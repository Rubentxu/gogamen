//! Aislamiento temporal del visor bajo carga (GOG-115).
//!
//! Ensaya cuatro modos de visor ([`ViewerMode`]) con la **misma** semilla y carga de trabajo
//! (rollout Grid headless, [`GridRollout`]) y mide, por tick: latencia del *sensor virtual*
//! (el paso del mundo Grid), coste de *publicación* hacia el visor ([`ObservationHub`] acotado),
//! latencia de ACK del puerto de control, fallos de presupuesto temporal explícito, contadores de
//! huecos/descartes, profundidad máxima de cola y (si el binario instala [`CountingAllocator`])
//! asignaciones de heap del hilo productor.
//!
//! Separación de responsabilidades:
//! - **sensor virtual** = `GridRollout::step` (observar + actuar sobre el mundo simulado);
//! - **visor** = hilo consumidor que sólo *lee* frames y renderiza SVG; nunca escribe en el
//!   rollout, así que no puede alterar decisiones (las decisiones dependen sólo de semilla+tick).
//!
//! # No es WCET
//! Los percentiles son observaciones sobre N muestras en este host; no son una cota del peor caso
//! ni una promesa de impacto cero del visor. Con presupuesto **lógico** (número de ticks) las
//! decisiones son idénticas entre modos; con plazo de **reloj real** el número de ticks completados
//! puede diferir y se registra ([`run_mode`] con [`Budget::WallClock`]).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use gogamen_core::{
    EntityId, EpistemicKind, EventSource, MasterSeed, MonotonicInstant, ObservationHub,
    PublishStatus, RunId, SimDuration, StreamMetrics, StreamRevision, Tick, VisualEntitySample,
    VisualFrame,
};
use gogamen_lab_grid::GridPos;
use gogamen_lab_grid::world::GridWorldConfig;

use crate::camera::Camera2D;
use crate::control::{
    AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    SessionHost, SessionOptions,
};
use crate::grid_lab::{GridRollout, GridWorldView, project_grid_scene};
use crate::viewport::SoftwareSvgBackend;

// ---------------------------------------------------------------------------------------------
// Allocator contador (opt-in: sólo cuenta si un binario lo instala como `#[global_allocator]`).
// ---------------------------------------------------------------------------------------------

thread_local! {
    static THREAD_ALLOCS: Cell<u64> = const { Cell::new(0) };
}

/// Allocator que delega en `System` y cuenta `alloc`/`realloc` por hilo (sin atómicos).
/// Instalar con `#[global_allocator] static A: CountingAllocator = CountingAllocator;`.
pub struct CountingAllocator;

// SAFETY: delega todas las operaciones en `std::alloc::System`; sólo incrementa un contador
// thread-local `Cell<u64>` con inicialización const (no asigna ni tiene destructor).
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = THREAD_ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

fn thread_allocs() -> u64 {
    THREAD_ALLOCS.try_with(Cell::get).unwrap_or(0)
}

/// `true` si el binario actual instaló [`CountingAllocator`] (se comprueba asignando).
pub fn counting_allocator_active() -> bool {
    let before = thread_allocs();
    let b = std::hint::black_box(Box::new(0u64));
    let after = thread_allocs();
    drop(b);
    after > before
}

// ---------------------------------------------------------------------------------------------
// Configuración
// ---------------------------------------------------------------------------------------------

/// Modo del visor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewerMode {
    /// Sin visor: ningún suscriptor; la publicación se omite.
    NoViewer,
    /// Visor que consume y renderiza SVG sin retraso artificial.
    Normal,
    /// Visor lento: duerme `slow_consumer_sleep` por frame (cola acotada desborda -> descartes).
    Slow,
    /// Visor que se desconecta a mitad de la ejecución (suscripción liberada + `close_viewer`).
    Disconnected,
}

impl ViewerMode {
    pub const ALL: [ViewerMode; 4] = [
        ViewerMode::NoViewer,
        ViewerMode::Normal,
        ViewerMode::Slow,
        ViewerMode::Disconnected,
    ];
    pub fn name(self) -> &'static str {
        match self {
            ViewerMode::NoViewer => "NoViewer",
            ViewerMode::Normal => "Normal",
            ViewerMode::Slow => "Slow",
            ViewerMode::Disconnected => "Disconnected",
        }
    }
}

/// Presupuesto de la ejecución.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    /// Presupuesto **lógico**: exactamente `ticks` pasos de decisión (determinista).
    Logical,
    /// Plazo de **reloj real**: se corta al pasar la duración (el nº de ticks puede diferir).
    WallClock(Duration),
}

/// Configuración común a todos los modos.
#[derive(Debug, Clone)]
pub struct IsolationConfig {
    pub seed: u64,
    /// Presupuesto lógico: ticks de decisión por ejecución (también tope en `WallClock`).
    pub ticks: u64,
    /// Capacidad de la cola visual acotada por suscriptor.
    pub frame_capacity: usize,
    /// Presupuesto de latencia por tick (ns) contra el que se cuentan los *misses*.
    pub tick_budget_ns: u64,
    /// Retraso por frame del visor lento.
    pub slow_consumer_sleep: Duration,
    /// Frames entregados tras los que el visor `Disconnected` se desconecta.
    pub disconnect_after_frames: u64,
    pub grid: GridWorldConfig,
}

impl Default for IsolationConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            ticks: 2000,
            frame_capacity: 8,
            tick_budget_ns: 1_000_000, // 1 ms por tick (configurado, no derivado de WCET)
            slow_consumer_sleep: Duration::from_millis(2),
            disconnect_after_frames: 16,
            grid: GridWorldConfig::default(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Estadísticas
// ---------------------------------------------------------------------------------------------

/// Resumen de N muestras en nanosegundos (percentil por rango más cercano).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    pub n: usize,
    pub p50: u64,
    pub p99: u64,
    pub max: u64,
}

impl Stats {
    pub fn from_samples(samples: &[u64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        let mut v = samples.to_vec();
        v.sort_unstable();
        let pick = |q: f64| {
            let rank = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len());
            v[rank - 1]
        };
        Self {
            n: v.len(),
            p50: pick(0.50),
            p99: pick(0.99),
            max: *v.last().expect("non-empty"),
        }
    }
}

/// Resultado de un modo.
#[derive(Debug, Clone)]
pub struct ModeResult {
    pub mode: ViewerMode,
    pub budget: Budget,
    /// Ticks de decisión completados.
    pub ticks_done: u64,
    pub episodes: u64,
    /// Hash acumulado tras cada tick (secuencia de decisiones/traza).
    pub hash_sequence: Vec<u64>,
    pub final_hash: u64,
    /// Sensor virtual: `GridRollout::step`.
    pub sensor_step: Stats,
    /// Publicación hacia el visor (construcción de frame + difusión al hub).
    pub publish: Stats,
    /// Tick completo (sensor + control + publicación).
    pub tick_total: Stats,
    /// Latencia del ACK del puerto de control.
    pub control_ack: Stats,
    pub tick_budget_ns: u64,
    /// Ticks cuyo `tick_total` superó `tick_budget_ns`.
    pub deadline_misses: u64,
    pub control_sent: u64,
    pub control_applied: u64,
    // --- huecos / cola ---
    pub frames_produced: u64,
    /// Frames realmente difundidos a algún suscriptor.
    pub frames_published: u64,
    /// Frames omitidos por no haber observadores (NoViewer o tras desconexión).
    pub frames_not_published: u64,
    pub frames_delivered: u64,
    /// Descartados por saturación de la cola (contador del ring).
    pub dropped_ring: u64,
    /// Publicados pero no entregados (quedaron en cola / en vuelo al cerrar).
    pub undelivered: u64,
    /// `produced - delivered` (debe ser `dropped_ring + undelivered + not_published`).
    pub gaps_total: u64,
    /// Huecos vistos por el consumidor (saltos de `revision`).
    pub consumer_observed_gaps: u64,
    pub queue_capacity: usize,
    /// Profundidad máxima muestreada por el consumidor (<= capacidad).
    pub queue_max_depth: usize,
    pub viewer_renders: u64,
    pub viewer_render: Stats,
    pub viewer_disconnected: bool,
    /// Asignaciones del hilo productor (`None` = NOT_MEASURED).
    pub allocs_sensor: Option<u64>,
    pub allocs_publish: Option<u64>,
    pub wall_ns: u128,
}

// ---------------------------------------------------------------------------------------------
// Ejecución
// ---------------------------------------------------------------------------------------------

fn fnv_fold(mut h: u64, words: &[u64]) -> u64 {
    for w in words {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

struct ConsumerReport {
    delivered: u64,
    observed_gaps: u64,
    max_depth: usize,
    renders: Vec<u64>,
    disconnected: bool,
    snapshot: StreamMetrics,
}

fn ns(d: Duration) -> u64 {
    d.as_nanos().min(u64::MAX as u128) as u64
}

/// Ejecuta un modo. El productor (motor) corre en el hilo llamante; el visor en otro hilo.
pub fn run_mode(cfg: &IsolationConfig, mode: ViewerMode, budget: Budget) -> ModeResult {
    let measure_allocs = counting_allocator_active();
    let host = SessionHost::new(1, SessionOptions::simulated());
    let control = host
        .control_port()
        .expect("simulated session is controllable");
    let observe = host.observation_port();
    let session = observe.observe().session;

    let hub = ObservationHub::new(RunId::new(cfg.seed), cfg.frame_capacity, 1).expect("capacity");
    let stop = Arc::new(AtomicBool::new(false));
    let disconnected_flag = Arc::new(AtomicBool::new(false));

    let consumer = if mode == ViewerMode::NoViewer {
        None
    } else {
        let sub = hub.subscribe(); // suscrito antes de producir
        let stop = Arc::clone(&stop);
        let disc = Arc::clone(&disconnected_flag);
        let host_c = host.clone();
        let obs_c = host.observation_port();
        let map = cfg.grid.map.clone();
        let sleep = cfg.slow_consumer_sleep;
        let after = cfg.disconnect_after_frames;
        Some(std::thread::spawn(move || {
            let backend = SoftwareSvgBackend;
            let camera = Camera2D::new(480.0, 360.0, true);
            let mut view = GridWorldView {
                map,
                agent: GridPos { x: 0, y: 0 },
                tick: Tick::ZERO,
                total_reward: 0.0,
                last_reward: 0.0,
                collisions: 0,
                terminal: Default::default(),
            };
            let mut rep = ConsumerReport {
                delivered: 0,
                observed_gaps: 0,
                max_depth: 0,
                renders: Vec::new(),
                disconnected: false,
                snapshot: StreamMetrics::default(),
            };
            let mut last_rev = 0u64;
            loop {
                rep.max_depth = rep.max_depth.max(sub.metrics().buffered_frames);
                match sub.try_recv_frame() {
                    Some(f) => {
                        rep.delivered += 1;
                        let rev = f.revision().as_u64();
                        rep.observed_gaps += rev.saturating_sub(last_rev + 1);
                        last_rev = rev;
                        let t0 = Instant::now();
                        if let Some(e) = f.entities().first() {
                            let p = e.position();
                            view.agent = GridPos {
                                x: p[0] as usize,
                                y: p[1] as usize,
                            };
                            view.tick = f.tick();
                        }
                        let svg = backend.render_svg(&project_grid_scene(&view), &camera);
                        std::hint::black_box(svg.len());
                        rep.renders.push(ns(t0.elapsed()));
                        let _ = obs_c.observe(); // el visor sólo lee
                        if mode == ViewerMode::Slow {
                            std::thread::sleep(sleep);
                            if stop.load(Ordering::Acquire) {
                                break;
                            }
                        }
                        if mode == ViewerMode::Disconnected && rep.delivered >= after {
                            host_c.close_viewer();
                            rep.disconnected = true;
                            break;
                        }
                    }
                    None => {
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        std::thread::sleep(Duration::from_micros(20));
                    }
                }
            }
            rep.snapshot = sub.metrics();
            drop(sub); // desconexión real (observer_count -> 0)
            disc.store(true, Ordering::Release);
            rep
        }))
    };

    // ---- bucle productor (motor) ----
    let mut episode = 0u64;
    let mut rollout = GridRollout::new(cfg.grid.clone(), MasterSeed::new(cfg.seed)).expect("cfg");
    let mut hash = FNV_OFFSET;
    let mut hash_sequence = Vec::new();
    let (mut s_step, mut s_pub, mut s_tot, mut s_ack) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a_sensor, mut a_publish) = (0u64, 0u64);
    let (mut misses, mut sent, mut applied) = (0u64, 0u64, 0u64);
    let (mut published, mut skipped) = (0u64, 0u64);
    let mut global = 0u64;
    let wall = Instant::now();
    let mid = cfg.ticks / 2;

    while global < cfg.ticks {
        if let Budget::WallClock(d) = budget
            && wall.elapsed() >= d
        {
            break;
        }
        // Harness-only (fuera de la medición): en Disconnected, esperar (acotado) a que el visor
        // se haya ido para garantizar que la desconexión ocurre a mitad de ejecución.
        if mode == ViewerMode::Disconnected && global == mid && matches!(budget, Budget::Logical) {
            let t = Instant::now();
            while !disconnected_flag.load(Ordering::Acquire) && t.elapsed() < Duration::from_secs(2)
            {
                std::thread::yield_now();
            }
        }
        let tick_start = Instant::now();
        let a0 = thread_allocs();
        // 1) sensor virtual
        let rec = loop {
            match rollout.step().expect("step") {
                Some(r) => break r,
                None => {
                    episode += 1;
                    rollout = GridRollout::new(
                        cfg.grid.clone(),
                        MasterSeed::new(cfg.seed.wrapping_add(episode)),
                    )
                    .expect("cfg");
                }
            }
        };
        let t1 = Instant::now();
        let a1 = thread_allocs();
        global += 1;
        hash = fnv_fold(hash, &[episode, global, rec.action as u64, rec.trace_hash]);
        hash_sequence.push(hash);
        // 2) control: nunca debe bloquearse por el visor
        let t2 = Instant::now();
        let ack = control.submit(ControlCommand {
            id: CommandId(global),
            session,
            action: ControlAction::SetSpeed(1.0 + (global % 3) as f64),
        });
        s_ack.push(ns(t2.elapsed()));
        sent += 1;
        if ack.outcome == AckOutcome::Applied {
            applied += 1;
        }
        // 3) publicación (sólo lectura hacia el visor)
        let t3 = Instant::now();
        let a2 = thread_allocs();
        let status = hub
            .publish_frame_if_observed(|| {
                let sample = VisualEntitySample::new(
                    EntityId::new(1),
                    [rec.position.x as f64, rec.position.y as f64, 0.0],
                    rec.total_reward,
                    rec.action as u32,
                )?;
                VisualFrame::new(
                    RunId::new(cfg.seed),
                    Tick::new(global),
                    SimDuration::from_nanos(global * 100_000_000),
                    MonotonicInstant::from_nanos(global as u128 * 1000),
                    StreamRevision::new(global),
                    EventSource::SimulationWorld,
                    EpistemicKind::Observed,
                    vec![sample],
                )
            })
            .expect("publish");
        let a3 = thread_allocs();
        let t4 = Instant::now();
        match status {
            PublishStatus::SkippedNoObservers => skipped += 1,
            _ => published += 1,
        }
        let step_ns = ns(t1 - tick_start);
        let pub_ns = ns(t4 - t3);
        let total = ns(t4 - tick_start);
        s_step.push(step_ns);
        s_pub.push(pub_ns);
        s_tot.push(total);
        if total > cfg.tick_budget_ns {
            misses += 1;
        }
        a_sensor += a1 - a0;
        a_publish += a3 - a2;
    }
    let wall_ns = wall.elapsed().as_nanos();
    stop.store(true, Ordering::Release);

    let cons = consumer.map(|h| h.join().expect("consumer thread"));
    let (delivered, observed_gaps, max_depth, renders, disc, snap) = match cons {
        Some(c) => (
            c.delivered,
            c.observed_gaps,
            c.max_depth,
            c.renders,
            c.disconnected,
            c.snapshot,
        ),
        None => (0, 0, 0, Vec::new(), false, StreamMetrics::default()),
    };
    let produced = global;
    // Invariante del ring (hard): lo publicado al suscriptor == entregado + descartado + en cola.
    // `snap` se tomó en el hilo consumidor, único que extrae, justo antes de liberar la suscripción.
    let in_flight_after_snapshot = published.saturating_sub(snap.published_frames);
    let undelivered = snap.buffered_frames as u64 + in_flight_after_snapshot;
    ModeResult {
        mode,
        budget,
        ticks_done: produced,
        episodes: episode + 1,
        final_hash: hash,
        hash_sequence,
        sensor_step: Stats::from_samples(&s_step),
        publish: Stats::from_samples(&s_pub),
        tick_total: Stats::from_samples(&s_tot),
        control_ack: Stats::from_samples(&s_ack),
        tick_budget_ns: cfg.tick_budget_ns,
        deadline_misses: misses,
        control_sent: sent,
        control_applied: applied,
        frames_produced: produced,
        frames_published: published,
        frames_not_published: skipped,
        frames_delivered: delivered,
        dropped_ring: snap.dropped_frames,
        undelivered,
        gaps_total: produced - delivered,
        consumer_observed_gaps: observed_gaps,
        queue_capacity: cfg.frame_capacity,
        queue_max_depth: max_depth,
        viewer_renders: renders.len() as u64,
        viewer_render: Stats::from_samples(&renders),
        viewer_disconnected: disc,
        allocs_sensor: measure_allocs.then_some(a_sensor),
        allocs_publish: measure_allocs.then_some(a_publish),
        wall_ns,
    }
}

/// `true` si todas las secuencias de hash de decisión son idénticas.
pub fn hash_sequences_identical(results: &[ModeResult]) -> bool {
    results
        .windows(2)
        .all(|w| w[0].hash_sequence == w[1].hash_sequence)
}

/// Longitud del prefijo común de las secuencias de hash de dos resultados.
pub fn common_prefix_len(a: &ModeResult, b: &ModeResult) -> usize {
    a.hash_sequence
        .iter()
        .zip(&b.hash_sequence)
        .take_while(|(x, y)| x == y)
        .count()
}

// ---------------------------------------------------------------------------------------------
// Informe
// ---------------------------------------------------------------------------------------------

/// Informe comparable de los cuatro modos.
#[derive(Debug, Clone)]
pub struct IsolationReport {
    pub platform: String,
    pub config: IsolationConfig,
    pub logical: Vec<ModeResult>,
    pub wall_clock: Vec<ModeResult>,
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// Descripción de plataforma (os/arch/cpus/perfil de compilación).
pub fn platform_description() -> String {
    format!(
        "os={} arch={} cpus={} cpu_model=\"{}\" profile={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        cpu_model(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
    )
}

impl IsolationReport {
    /// Ejecuta los cuatro modos con presupuesto lógico y, opcionalmente, con plazo de reloj real.
    pub fn run(cfg: IsolationConfig, wall_clock_deadline: Option<Duration>) -> Self {
        let logical = ViewerMode::ALL
            .iter()
            .map(|m| run_mode(&cfg, *m, Budget::Logical))
            .collect();
        let wall_clock = match wall_clock_deadline {
            Some(d) => ViewerMode::ALL
                .iter()
                .map(|m| run_mode(&cfg, *m, Budget::WallClock(d)))
                .collect(),
            None => Vec::new(),
        };
        Self {
            platform: platform_description(),
            config: cfg,
            logical,
            wall_clock,
        }
    }

    /// Markdown plano.
    pub fn to_markdown(&self) -> String {
        let c = &self.config;
        let mut o = String::new();
        let _ = writeln!(o, "# Viewer isolation report (GOG-115)\n");
        let _ = writeln!(
            o,
            "DISCLAIMER: not WCET. Percentiles are observations over N samples on this host; \
             no worst-case bound and no zero-impact promise are made.\n"
        );
        let _ = writeln!(o, "## Platform\n\n{}\n", self.platform);
        let _ = writeln!(o, "## Configuration\n");
        let _ = writeln!(
            o,
            "- seed (master): {} (episode k uses seed+k)\n- logical budget: {} ticks\n- queue capacity: {} frames\n- tick budget (configured, ns): {}\n- slow consumer sleep: {:?}/frame\n- disconnect after: {} delivered frames\n- workload: GridRollout (headless Grid, cul_de_sac map, seeded random policy)\n- sensor = GridRollout::step; viewer = consumer thread (read-only, SVG render)\n- allocation counting: {}\n",
            c.seed,
            c.ticks,
            c.frame_capacity,
            c.tick_budget_ns,
            c.slow_consumer_sleep,
            c.disconnect_after_frames,
            self.logical.first().and_then(|r| r.allocs_sensor).map_or(
                "NOT_MEASURED (CountingAllocator not installed in this binary)".to_string(),
                |_| "measured on the producer thread (CountingAllocator)".to_string()
            ),
        );
        section(&mut o, "Logical budget (deterministic)", &self.logical);
        if !self.wall_clock.is_empty() {
            section(
                &mut o,
                "Real-time deadline (may differ by mode)",
                &self.wall_clock,
            );
            let _ = writeln!(o, "### Real-time deadline vs decisions\n");
            let base = &self.wall_clock[0];
            for r in &self.wall_clock {
                let _ = writeln!(
                    o,
                    "- {}: ticks_done={} (vs NoViewer {}), common hash prefix with NoViewer={} (identical decisions on the common prefix: {})",
                    r.mode.name(),
                    r.ticks_done,
                    base.ticks_done,
                    common_prefix_len(base, r),
                    common_prefix_len(base, r) == base.ticks_done.min(r.ticks_done) as usize,
                );
            }
            let _ = writeln!(
                o,
                "\nA real-time deadline cuts runs at different logical ticks; any difference above \
                 is recorded, not hidden.\n"
            );
        }
        let _ = writeln!(
            o,
            "## Decision determinism (logical budget)\n\nhash sequences identical across modes: {}\nfinal hashes: {}\n",
            hash_sequences_identical(&self.logical),
            self.logical
                .iter()
                .map(|r| format!("{}={:016x}", r.mode.name(), r.final_hash))
                .collect::<Vec<_>>()
                .join(" ")
        );
        o
    }
}

fn section(o: &mut String, title: &str, rs: &[ModeResult]) {
    let _ = writeln!(o, "## {title}\n");
    let _ = writeln!(
        o,
        "Times in ns (p50 / p99 / max over N=ticks samples). misses = tick_total > budget.\n"
    );
    let _ = writeln!(
        o,
        "| mode | ticks | sensor_step p50/p99/max | publish p50/p99/max | tick_total p50/p99/max | ctl_ack p50/p99/max | misses |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|---|");
    for r in rs {
        let f = |s: &Stats| format!("{}/{}/{}", s.p50, s.p99, s.max);
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            r.mode.name(),
            r.ticks_done,
            f(&r.sensor_step),
            f(&r.publish),
            f(&r.tick_total),
            f(&r.control_ack),
            r.deadline_misses
        );
    }
    let _ = writeln!(o);
    let _ = writeln!(
        o,
        "| mode | produced | published | not_published | delivered | dropped(ring) | undelivered | gaps(prod-deliv) | consumer_gaps | q_max/cap | ctl applied/sent | viewer render p50/max | allocs sensor/publish |"
    );
    let _ = writeln!(o, "|---|---|---|---|---|---|---|---|---|---|---|---|---|");
    for r in rs {
        let al = |a: Option<u64>| a.map_or("NOT_MEASURED".to_string(), |v| v.to_string());
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {}/{} | {}/{} | {}/{} | {}/{} |",
            r.mode.name(),
            r.frames_produced,
            r.frames_published,
            r.frames_not_published,
            r.frames_delivered,
            r.dropped_ring,
            r.undelivered,
            r.gaps_total,
            r.consumer_observed_gaps,
            r.queue_max_depth,
            r.queue_capacity,
            r.control_applied,
            r.control_sent,
            r.viewer_render.p50,
            r.viewer_render.max,
            al(r.allocs_sensor),
            al(r.allocs_publish),
        );
    }
    let _ = writeln!(o);
}
