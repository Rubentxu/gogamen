//! Telemetría acotada de cumplimiento temporal y reporte estructurado (`std`-only).
//!
//! Este módulo mide el cumplimiento temporal de bucles de control y pasos críticos
//! sin bloquear la ruta de ejecución ni realizar E/S de disco o red en el registro:
//! - Soporta [`gogamen_core::VirtualMonotonicClock`] para oráculos deterministas en tests
//!   y [`gogamen_core::SystemMonotonicClock`] para medición wall-time monotónica real.
//! - Mantiene ventanas circulares preasignadas ([`PreallocatedRingBuffer`]), un histograma
//!   de buckets fijos ([`FixedLatencyHistogram`]) y contadores estáticos por motivo de
//!   respaldo ([`FallbackCounts`]), garantizando memoria acotada O(1) bajo alto volumen.
//! - Emite un [`TemporalComplianceReport`] con metadatos de plataforma, carga, plazos,
//!   percentiles (`p50`, `p95`, `p99`), `max_observed`, jitter, `deadline_misses`,
//!   conteos de fallback, histograma, transiciones de estado y la nota explícita
//!   [`NO_WCET_CLAIM_NOTICE`] indicando que las mediciones observadas **no** afirman ni
//!   certifican WCET (Worst-Case Execution Time).

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use gogamen_core::{MonotonicClock, MonotonicInstant, Tick};

use crate::multirate::{MultirateFallbackReason, MultirateTickRecord, MultirateTrace};
use crate::temporal_profile::{PreallocatedRingBuffer, STRICT_REALTIME_NON_CERTIFICATION_NOTICE};

/// Aviso explícito de que la telemetría observada no afirma ni certifica WCET.
pub const NO_WCET_CLAIM_NOTICE: &str = "Los resultados de telemetría observados reflejan mediciones empíricas bajo la carga ejecutada y no afirman, demuestran ni certifican WCET (Worst-Case Execution Time) ni garantías de tiempo real duro.";

/// Número de buckets fijos del histograma de latencia.
pub const HISTOGRAM_BUCKET_COUNT: usize = 16;

/// Límites superiores inclusivos (en nanosegundos) para los primeros 15 buckets del histograma.
/// El bucket 16 corresponde al desbordamiento (`> 1_000_000_000 ns`).
pub const DEFAULT_HISTOGRAM_BOUNDS_NANOS: [u64; HISTOGRAM_BUCKET_COUNT - 1] = [
    1_000,         // <= 1 us
    5_000,         // <= 5 us
    10_000,        // <= 10 us
    50_000,        // <= 50 us
    100_000,       // <= 100 us
    250_000,       // <= 250 us
    500_000,       // <= 500 us
    1_000_000,     // <= 1 ms
    2_000_000,     // <= 2 ms
    5_000_000,     // <= 5 ms
    10_000_000,    // <= 10 ms
    20_000_000,    // <= 20 ms
    50_000_000,    // <= 50 ms
    100_000_000,   // <= 100 ms
    1_000_000_000, // <= 1 s
];

const DEFAULT_WINDOW_CAPACITY: usize = 1024;
const DEFAULT_TRANSITION_CAPACITY: usize = 256;

/// Errores de configuración de la telemetría de cumplimiento temporal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalTelemetryError {
    /// Parámetro de configuración inválido.
    InvalidConfig(&'static str),
}

impl fmt::Display for TemporalTelemetryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid temporal telemetry config: {msg}"),
        }
    }
}

impl Error for TemporalTelemetryError {}

/// Metadatos de plataforma incluidos en [`TemporalComplianceReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlatformMetadata {
    /// Sistema operativo objetivo (`std::env::consts::OS`).
    pub os: &'static str,
    /// Arquitectura objetivo (`std::env::consts::ARCH`).
    pub arch: &'static str,
}

impl PlatformMetadata {
    /// Captura los metadatos de plataforma de compilación actual.
    pub const fn current() -> Self {
        Self {
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        }
    }
}

/// Configuración acotada de [`TemporalTelemetryRecorder`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalTelemetryConfig {
    /// Etiqueta descriptiva de la carga o escenario medido.
    pub load_label: String,
    /// Periodo objetivo por tick en nanosegundos.
    pub target_period_nanos: u64,
    /// Plazo máximo aceptable por paso en nanosegundos.
    pub deadline_nanos: u64,
    /// Capacidad fija de la ventana circular para cálculo de percentiles y jitter reciente.
    pub window_capacity: NonZeroUsize,
    /// Capacidad fija del buffer circular de transiciones de estado.
    pub transition_capacity: NonZeroUsize,
}

impl TemporalTelemetryConfig {
    /// Crea una configuración validando que `target_period` y `deadline` sean mayores que cero
    /// y quepan en `u64` nanosegundos.
    pub fn new(
        load_label: impl Into<String>,
        target_period: Duration,
        deadline: Duration,
    ) -> Result<Self, TemporalTelemetryError> {
        let load_label = load_label.into();
        if load_label.trim().is_empty() {
            return Err(TemporalTelemetryError::InvalidConfig(
                "load_label must not be empty",
            ));
        }
        let target_period_nanos = u64::try_from(target_period.as_nanos()).map_err(|_| {
            TemporalTelemetryError::InvalidConfig("target_period exceeds u64 nanoseconds")
        })?;
        if target_period_nanos == 0 {
            return Err(TemporalTelemetryError::InvalidConfig(
                "target_period must be greater than zero",
            ));
        }
        let deadline_nanos = u64::try_from(deadline.as_nanos()).map_err(|_| {
            TemporalTelemetryError::InvalidConfig("deadline exceeds u64 nanoseconds")
        })?;
        if deadline_nanos == 0 {
            return Err(TemporalTelemetryError::InvalidConfig(
                "deadline must be greater than zero",
            ));
        }

        Ok(Self {
            load_label,
            target_period_nanos,
            deadline_nanos,
            window_capacity: NonZeroUsize::new(DEFAULT_WINDOW_CAPACITY)
                .expect("default window capacity is non-zero"),
            transition_capacity: NonZeroUsize::new(DEFAULT_TRANSITION_CAPACITY)
                .expect("default transition capacity is non-zero"),
        })
    }

    /// Ajusta la capacidad fija de la ventana circular de muestras.
    pub const fn with_window_capacity(mut self, capacity: NonZeroUsize) -> Self {
        self.window_capacity = capacity;
        self
    }

    /// Ajusta la capacidad fija del historial de transiciones de estado.
    pub const fn with_transition_capacity(mut self, capacity: NonZeroUsize) -> Self {
        self.transition_capacity = capacity;
        self
    }
}

/// Bucket individual dentro de [`FixedLatencyHistogram`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistogramBucket {
    /// Límite superior inclusivo en nanosegundos (`None` para el bucket de desbordamiento).
    pub upper_bound_nanos: Option<u64>,
    /// Número de muestras acumuladas en este bucket.
    pub count: u64,
}

/// Histograma acotado de latencias con buckets fijos predefinidos (sin crecimiento dinámico).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedLatencyHistogram {
    counts: [u64; HISTOGRAM_BUCKET_COUNT],
    total_samples: u64,
    min_nanos: Option<u64>,
    max_nanos: u64,
}

impl FixedLatencyHistogram {
    /// Crea un histograma vacío con [`HISTOGRAM_BUCKET_COUNT`] buckets fijos.
    pub const fn new() -> Self {
        Self {
            counts: [0; HISTOGRAM_BUCKET_COUNT],
            total_samples: 0,
            min_nanos: None,
            max_nanos: 0,
        }
    }

    /// Registra una muestra de latencia en nanosegundos en O(1) y sin asignaciones.
    pub fn record(&mut self, latency_nanos: u64) {
        let bucket_idx = DEFAULT_HISTOGRAM_BOUNDS_NANOS
            .iter()
            .position(|&bound| latency_nanos <= bound)
            .unwrap_or(HISTOGRAM_BUCKET_COUNT - 1);

        self.counts[bucket_idx] = self.counts[bucket_idx].saturating_add(1);
        self.total_samples = self.total_samples.saturating_add(1);
        self.min_nanos = Some(match self.min_nanos {
            Some(current) => current.min(latency_nanos),
            None => latency_nanos,
        });
        self.max_nanos = self.max_nanos.max(latency_nanos);
    }

    /// Número total de muestras registradas en el histograma.
    pub const fn total_samples(&self) -> u64 {
        self.total_samples
    }

    /// Latencia mínima observada, si hay al menos una muestra.
    pub const fn min_nanos(&self) -> Option<u64> {
        self.min_nanos
    }

    /// Latencia máxima observada (`0` si no hay muestras).
    pub const fn max_nanos(&self) -> u64 {
        self.max_nanos
    }

    /// Arreglo crudo de conteos por bucket.
    pub const fn counts(&self) -> &[u64; HISTOGRAM_BUCKET_COUNT] {
        &self.counts
    }

    /// Devuelve todos los buckets con su límite superior y conteo.
    pub fn buckets(&self) -> [HistogramBucket; HISTOGRAM_BUCKET_COUNT] {
        let mut out = [HistogramBucket {
            upper_bound_nanos: None,
            count: 0,
        }; HISTOGRAM_BUCKET_COUNT];
        let mut idx = 0;
        while idx < HISTOGRAM_BUCKET_COUNT {
            let upper = if idx < HISTOGRAM_BUCKET_COUNT - 1 {
                Some(DEFAULT_HISTOGRAM_BOUNDS_NANOS[idx])
            } else {
                None
            };
            out[idx] = HistogramBucket {
                upper_bound_nanos: upper,
                count: self.counts[idx],
            };
            idx += 1;
        }
        out
    }
}

impl Default for FixedLatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Conteo fijo por cada variante de [`MultirateFallbackReason`] (sin asignaciones dinámicas).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FallbackCounts {
    counts: [u64; 7],
}

impl FallbackCounts {
    const ALL_REASONS: [MultirateFallbackReason; 7] = [
        MultirateFallbackReason::NoProposalYet,
        MultirateFallbackReason::WorkerFailed,
        MultirateFallbackReason::WatchdogExpired,
        MultirateFallbackReason::StalePlan,
        MultirateFallbackReason::StaleObservation,
        MultirateFallbackReason::DeadlineExpired,
        MultirateFallbackReason::GuardRejected,
    ];

    const fn reason_index(reason: MultirateFallbackReason) -> usize {
        match reason {
            MultirateFallbackReason::NoProposalYet => 0,
            MultirateFallbackReason::WorkerFailed => 1,
            MultirateFallbackReason::WatchdogExpired => 2,
            MultirateFallbackReason::StalePlan => 3,
            MultirateFallbackReason::StaleObservation => 4,
            MultirateFallbackReason::DeadlineExpired => 5,
            MultirateFallbackReason::GuardRejected => 6,
        }
    }

    /// Incrementa en 1 el contador asociado a `reason`.
    pub fn increment(&mut self, reason: MultirateFallbackReason) {
        let idx = Self::reason_index(reason);
        self.counts[idx] = self.counts[idx].saturating_add(1);
    }

    /// Devuelve el número de activaciones registradas para `reason`.
    pub const fn get(&self, reason: MultirateFallbackReason) -> u64 {
        self.counts[Self::reason_index(reason)]
    }

    /// Suma total de todas las activaciones de respaldo registradas.
    pub const fn total(&self) -> u64 {
        self.counts[0]
            .saturating_add(self.counts[1])
            .saturating_add(self.counts[2])
            .saturating_add(self.counts[3])
            .saturating_add(self.counts[4])
            .saturating_add(self.counts[5])
            .saturating_add(self.counts[6])
    }

    /// Itera sobre todos los motivos y su conteo observado.
    pub fn iter(&self) -> impl Iterator<Item = (MultirateFallbackReason, u64)> + '_ {
        Self::ALL_REASONS
            .iter()
            .copied()
            .enumerate()
            .map(|(idx, reason)| (reason, self.counts[idx]))
    }
}

/// Estado agregado de cumplimiento temporal en un tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemporalComplianceState {
    /// Paso dentro de plazo y sin respaldo activo.
    Nominal,
    /// Paso excedió `deadline_nanos` sin activar respaldo.
    DeadlineMissed,
    /// Paso dentro de plazo pero activó respaldo seguro.
    FallbackActive(MultirateFallbackReason),
    /// Paso excedió `deadline_nanos` y activó respaldo seguro.
    DegradedMissAndFallback(MultirateFallbackReason),
}

impl TemporalComplianceState {
    const fn classify(
        deadline_missed: bool,
        fallback_reason: Option<MultirateFallbackReason>,
    ) -> Self {
        match (deadline_missed, fallback_reason) {
            (false, None) => Self::Nominal,
            (true, None) => Self::DeadlineMissed,
            (false, Some(reason)) => Self::FallbackActive(reason),
            (true, Some(reason)) => Self::DegradedMissAndFallback(reason),
        }
    }
}

/// Transición registrada cuando cambia el [`TemporalComplianceState`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalTransition {
    /// Tick en que se produjo la transición.
    pub tick: Tick,
    /// Instante monotónico en nanosegundos al finalizar el registro del tick.
    pub timestamp_nanos: u128,
    /// Estado previo.
    pub from: TemporalComplianceState,
    /// Nuevo estado alcanzado.
    pub to: TemporalComplianceState,
    /// Latencia observada en el tick de transición (nanosegundos).
    pub latency_nanos: u64,
    /// Jitter firmado en nanosegundos asociado al tick de transición.
    pub jitter_nanos: i128,
}

/// Resumen de métricas de jitter incluidas en [`TemporalComplianceReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitterSummary {
    /// Jitter absoluto máximo observado globalmente en nanosegundos.
    pub max_abs_nanos: u64,
    /// Jitter absoluto medio acumulado en nanosegundos.
    pub mean_abs_nanos: u64,
    /// Percentil 95 del jitter absoluto en la ventana acotada (nanosegundos).
    pub p95_abs_nanos: u64,
    /// Último jitter firmado observado en nanosegundos.
    pub last_signed_nanos: i128,
}

/// Resultado inmediato devuelto por cada registro de tick en [`TemporalTelemetryRecorder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordedTickTelemetry {
    /// Tick registrado.
    pub tick: Tick,
    /// Latencia medida en nanosegundos.
    pub latency_nanos: u64,
    /// Jitter firmado en nanosegundos.
    pub jitter_nanos: i128,
    /// Indica si la latencia superó `deadline_nanos`.
    pub deadline_missed: bool,
    /// Motivo de fallback registrado, si existió.
    pub fallback_reason: Option<MultirateFallbackReason>,
    /// Estado de cumplimiento tras este tick.
    pub state: TemporalComplianceState,
    /// Indica si este tick produjo una transición respecto al estado anterior.
    pub transitioned: bool,
}

/// Reporte estructurado de cumplimiento temporal (AC #2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalComplianceReport {
    /// Metadatos de plataforma (`os`, `arch`).
    pub platform: PlatformMetadata,
    /// Etiqueta de carga / escenario medido.
    pub load_label: String,
    /// Periodo objetivo configurado en nanosegundos.
    pub target_period_nanos: u64,
    /// Plazo configurado en nanosegundos.
    pub deadline_nanos: u64,
    /// Número total de muestras procesadas.
    pub total_samples: u64,
    /// Número de muestras actualmente retenidas en la ventana acotada.
    pub window_samples: usize,
    /// Capacidad máxima fija de la ventana acotada.
    pub window_capacity: usize,
    /// Sobrescrituras producidas en la ventana circular.
    pub window_overwrites: u64,
    /// Percentil 50 de latencia en nanosegundos (ventana acotada).
    pub p50_nanos: u64,
    /// Percentil 95 de latencia en nanosegundos (ventana acotada).
    pub p95_nanos: u64,
    /// Percentil 99 de latencia en nanosegundos (ventana acotada).
    pub p99_nanos: u64,
    /// Latencia máxima observada globalmente en nanosegundos.
    pub max_observed_nanos: u64,
    /// Resumen de jitter.
    pub jitter: JitterSummary,
    /// Conteo total de incumplimientos de plazo (`latency_nanos > deadline_nanos`).
    pub deadline_misses: u64,
    /// Conteos por motivo específico de fallback.
    pub fallback_counts: FallbackCounts,
    /// Histograma acotado con buckets fijos predefinidos.
    pub histogram: FixedLatencyHistogram,
    /// Transiciones de estado retenidas en el buffer acotado.
    pub transitions: Vec<TemporalTransition>,
    /// Número total de transiciones observadas históricamente.
    pub total_transitions: u64,
    /// Sobrescrituras en el buffer circular de transiciones.
    pub transition_overwrites: u64,
    /// Estado de cumplimiento actual.
    pub current_state: TemporalComplianceState,
    /// Nota explícita de que las mediciones observadas no afirman ni certifican WCET.
    pub non_wcet_notice: &'static str,
    /// Nota explícita de no certificación en tiempo real estricto.
    pub non_realtime_notice: &'static str,
}

/// Registrador acotado en memoria de telemetría temporal desacoplado de render e I/O lento.
///
/// Toda la memoria de muestra se preasigna en [`TemporalTelemetryRecorder::new`].
/// Los métodos de registro (`finish_tick`, `record_sample`, `record_multirate_tick`)
/// operan en O(1) sobre buffers circulares y contadores fijos sin asignar en heap ni
/// ejecutar syscalls de disco o red.
#[derive(Debug, Clone)]
pub struct TemporalTelemetryRecorder<C: MonotonicClock> {
    clock: C,
    config: TemporalTelemetryConfig,
    platform: PlatformMetadata,
    latencies_window: PreallocatedRingBuffer<u64>,
    abs_jitters_window: PreallocatedRingBuffer<u64>,
    transitions: PreallocatedRingBuffer<TemporalTransition>,
    histogram: FixedLatencyHistogram,
    fallback_counts: FallbackCounts,
    total_samples: u64,
    deadline_misses: u64,
    max_observed_nanos: u64,
    max_abs_jitter_nanos: u64,
    sum_abs_jitter_nanos: u128,
    last_signed_jitter_nanos: i128,
    total_transitions: u64,
    current_state: TemporalComplianceState,
}

impl<C: MonotonicClock> TemporalTelemetryRecorder<C> {
    /// Construye un registrador preasignado con capacidad acotada usando `clock`.
    pub fn new(clock: C, config: TemporalTelemetryConfig) -> Self {
        let latencies_window = PreallocatedRingBuffer::with_capacity(config.window_capacity);
        let abs_jitters_window = PreallocatedRingBuffer::with_capacity(config.window_capacity);
        let transitions = PreallocatedRingBuffer::with_capacity(config.transition_capacity);
        Self {
            clock,
            config,
            platform: PlatformMetadata::current(),
            latencies_window,
            abs_jitters_window,
            transitions,
            histogram: FixedLatencyHistogram::new(),
            fallback_counts: FallbackCounts::default(),
            total_samples: 0,
            deadline_misses: 0,
            max_observed_nanos: 0,
            max_abs_jitter_nanos: 0,
            sum_abs_jitter_nanos: 0,
            last_signed_jitter_nanos: 0,
            total_transitions: 0,
            current_state: TemporalComplianceState::Nominal,
        }
    }

    /// Referencia al reloj monotónico configurado.
    pub const fn clock(&self) -> &C {
        &self.clock
    }

    /// Configuración acotada vigente.
    pub const fn config(&self) -> &TemporalTelemetryConfig {
        &self.config
    }

    /// Toma el instante de inicio del tick usando `self.clock`.
    pub fn start_tick(&self) -> MonotonicInstant {
        self.clock.now()
    }

    /// Finaliza un tick iniciado en `started_at`, mide la latencia mediante `self.clock.now()`
    /// y registra la muestra en O(1) sin asignaciones ni I/O.
    pub fn finish_tick(
        &mut self,
        tick: Tick,
        started_at: MonotonicInstant,
        target_at: Option<MonotonicInstant>,
        fallback_reason: Option<MultirateFallbackReason>,
    ) -> RecordedTickTelemetry {
        let finished_at = self.clock.now();
        let latency = finished_at.saturating_duration_since(started_at);
        let latency_nanos = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);
        let jitter_nanos = match target_at {
            Some(target) => signed_diff_nanos(started_at, target),
            None => (latency_nanos as i128) - (self.config.target_period_nanos as i128),
        };
        self.record_raw(
            tick,
            finished_at,
            latency_nanos,
            jitter_nanos,
            fallback_reason,
        )
    }

    /// Ejecuta `step_fn` midiendo su duración real con `self.clock` y registra el resultado.
    pub fn measure_step<F, R>(
        &mut self,
        tick: Tick,
        target_at: Option<MonotonicInstant>,
        fallback_reason: Option<MultirateFallbackReason>,
        step_fn: F,
    ) -> (R, RecordedTickTelemetry)
    where
        F: FnOnce() -> R,
    {
        let started_at = self.start_tick();
        let result = step_fn();
        let telemetry = self.finish_tick(tick, started_at, target_at, fallback_reason);
        (result, telemetry)
    }

    /// Registra una muestra explícita con latencia, jitter firmado y fallback opcional.
    pub fn record_sample(
        &mut self,
        tick: Tick,
        timestamp: MonotonicInstant,
        latency: Duration,
        jitter_nanos: i128,
        fallback_reason: Option<MultirateFallbackReason>,
    ) -> RecordedTickTelemetry {
        let latency_nanos = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);
        self.record_raw(
            tick,
            timestamp,
            latency_nanos,
            jitter_nanos,
            fallback_reason,
        )
    }

    /// Ingiere un registro de [`MultirateTrace`] en la telemetría acotada.
    pub fn record_multirate_tick(&mut self, record: &MultirateTickRecord) -> RecordedTickTelemetry {
        let latency_nanos =
            u64::try_from(record.control_step_latency.as_nanos()).unwrap_or(u64::MAX);
        self.record_raw(
            record.tick,
            record.finished_at,
            latency_nanos,
            record.jitter_nanos,
            record.fallback_reason,
        )
    }

    /// Ingiere todos los registros de una [`MultirateTrace`].
    pub fn record_multirate_trace(&mut self, trace: &MultirateTrace) {
        for record in trace.records() {
            let _ = self.record_multirate_tick(record);
        }
    }

    fn record_raw(
        &mut self,
        tick: Tick,
        timestamp: MonotonicInstant,
        latency_nanos: u64,
        jitter_nanos: i128,
        fallback_reason: Option<MultirateFallbackReason>,
    ) -> RecordedTickTelemetry {
        let deadline_missed = latency_nanos > self.config.deadline_nanos;
        let abs_jitter_nanos = u64::try_from(jitter_nanos.unsigned_abs()).unwrap_or(u64::MAX);

        self.total_samples = self.total_samples.saturating_add(1);
        self.max_observed_nanos = self.max_observed_nanos.max(latency_nanos);
        self.max_abs_jitter_nanos = self.max_abs_jitter_nanos.max(abs_jitter_nanos);
        self.sum_abs_jitter_nanos = self
            .sum_abs_jitter_nanos
            .saturating_add(u128::from(abs_jitter_nanos));
        self.last_signed_jitter_nanos = jitter_nanos;

        if deadline_missed {
            self.deadline_misses = self.deadline_misses.saturating_add(1);
        }
        if let Some(reason) = fallback_reason {
            self.fallback_counts.increment(reason);
        }

        self.histogram.record(latency_nanos);
        let _ = self.latencies_window.push_overwrite(latency_nanos);
        let _ = self.abs_jitters_window.push_overwrite(abs_jitter_nanos);

        let next_state = TemporalComplianceState::classify(deadline_missed, fallback_reason);
        let transitioned = next_state != self.current_state;
        if transitioned {
            let transition = TemporalTransition {
                tick,
                timestamp_nanos: timestamp.as_nanos(),
                from: self.current_state,
                to: next_state,
                latency_nanos,
                jitter_nanos,
            };
            let _ = self.transitions.push_overwrite(transition);
            self.total_transitions = self.total_transitions.saturating_add(1);
            self.current_state = next_state;
        }

        RecordedTickTelemetry {
            tick,
            latency_nanos,
            jitter_nanos,
            deadline_missed,
            fallback_reason,
            state: next_state,
            transitioned,
        }
    }

    /// Genera un [`TemporalComplianceReport`] estructurado a partir del estado acotado actual.
    ///
    /// Esta operación está desacoplada de la ruta crítica de registro y no realiza I/O ni render.
    pub fn report(&self) -> TemporalComplianceReport {
        let mut sorted_latencies: Vec<u64> = self.latencies_window.iter().copied().collect();
        sorted_latencies.sort_unstable();

        let mut sorted_jitters: Vec<u64> = self.abs_jitters_window.iter().copied().collect();
        sorted_jitters.sort_unstable();

        let mean_abs_nanos = if self.total_samples == 0 {
            0
        } else {
            u64::try_from(self.sum_abs_jitter_nanos / u128::from(self.total_samples))
                .unwrap_or(u64::MAX)
        };

        let transitions: Vec<TemporalTransition> = self.transitions.iter().copied().collect();

        TemporalComplianceReport {
            platform: self.platform,
            load_label: self.config.load_label.clone(),
            target_period_nanos: self.config.target_period_nanos,
            deadline_nanos: self.config.deadline_nanos,
            total_samples: self.total_samples,
            window_samples: self.latencies_window.len(),
            window_capacity: self.latencies_window.capacity(),
            window_overwrites: self.latencies_window.overwrites(),
            p50_nanos: percentile_from_sorted(&sorted_latencies, 50, 100),
            p95_nanos: percentile_from_sorted(&sorted_latencies, 95, 100),
            p99_nanos: percentile_from_sorted(&sorted_latencies, 99, 100),
            max_observed_nanos: self.max_observed_nanos,
            jitter: JitterSummary {
                max_abs_nanos: self.max_abs_jitter_nanos,
                mean_abs_nanos,
                p95_abs_nanos: percentile_from_sorted(&sorted_jitters, 95, 100),
                last_signed_nanos: self.last_signed_jitter_nanos,
            },
            deadline_misses: self.deadline_misses,
            fallback_counts: self.fallback_counts,
            histogram: self.histogram.clone(),
            transitions,
            total_transitions: self.total_transitions,
            transition_overwrites: self.transitions.overwrites(),
            current_state: self.current_state,
            non_wcet_notice: NO_WCET_CLAIM_NOTICE,
            non_realtime_notice: STRICT_REALTIME_NON_CERTIFICATION_NOTICE,
        }
    }
}

fn signed_diff_nanos(lhs: MonotonicInstant, rhs: MonotonicInstant) -> i128 {
    let a = lhs.as_nanos();
    let b = rhs.as_nanos();
    if a >= b {
        i128::try_from(a - b).unwrap_or(i128::MAX)
    } else {
        -i128::try_from(b - a).unwrap_or(i128::MAX)
    }
}

fn percentile_from_sorted(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let rank = n.saturating_mul(numerator).div_ceil(denominator).max(1);
    sorted[(rank - 1).min(n - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use gogamen_core::{SystemMonotonicClock, VirtualMonotonicClock};

    #[test]
    fn ac1_virtual_clock_fixture_with_induced_delays_counts_exact_misses_fallbacks_and_percentiles()
    {
        let clock = VirtualMonotonicClock::new();
        let config = TemporalTelemetryConfig::new(
            "deterministic_fixture_10ms",
            Duration::from_millis(10),
            Duration::from_millis(10),
        )
        .unwrap();
        let mut recorder = TemporalTelemetryRecorder::new(clock, config);

        // 20 ticks deterministas con retrasos inducidos conocidos (en ms):
        // 1..=17 dentro de plazo (1..=8 ms), 18..=20 exceden deadline de 10 ms (12, 15, 25 ms).
        let step_durations_ms: [u64; 20] = [
            1, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 8, 8, 9, 10, 12, 15, 25,
        ];
        let fallbacks: [Option<MultirateFallbackReason>; 20] = [
            None,
            None,
            None,
            None,
            Some(MultirateFallbackReason::NoProposalYet),
            None,
            None,
            None,
            None,
            Some(MultirateFallbackReason::StalePlan),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(MultirateFallbackReason::DeadlineExpired),
            Some(MultirateFallbackReason::DeadlineExpired),
            Some(MultirateFallbackReason::WatchdogExpired),
        ];

        for i in 0..20 {
            let target_at = MonotonicInstant::from_nanos((i as u128) * 10_000_000);
            // Inducimos jitter de inicio en ticks impares (+500 us) y pares (0 us).
            if i % 2 == 1 {
                recorder.clock().advance(Duration::from_micros(500));
            }
            let started_at = recorder.start_tick();
            recorder
                .clock()
                .advance(Duration::from_millis(step_durations_ms[i]));
            let recorded = recorder.finish_tick(
                Tick::new(i as u64),
                started_at,
                Some(target_at),
                fallbacks[i],
            );

            let expected_miss = step_durations_ms[i] > 10;
            assert_eq!(recorded.deadline_missed, expected_miss);

            // Alinear reloj virtual al siguiente periodo teórico si sobra tiempo.
            let next_target = MonotonicInstant::from_nanos(((i + 1) as u128) * 10_000_000);
            let now = recorder.clock().now();
            if now < next_target {
                recorder
                    .clock()
                    .advance(next_target.saturating_duration_since(now));
            }
        }

        let report = recorder.report();
        assert_eq!(report.total_samples, 20);
        assert_eq!(report.deadline_misses, 3); // 12ms, 15ms, 25ms
        assert_eq!(report.max_observed_nanos, 25_000_000);
        // Con N=20 ordenado: p50 -> idx 9 (5ms), p95 -> idx 18 (15ms), p99 -> idx 19 (25ms)
        assert_eq!(report.p50_nanos, 5_000_000);
        assert_eq!(report.p95_nanos, 15_000_000);
        assert_eq!(report.p99_nanos, 25_000_000);
        assert_eq!(report.fallback_counts.total(), 5);
        assert_eq!(
            report
                .fallback_counts
                .get(MultirateFallbackReason::NoProposalYet),
            1
        );
        assert_eq!(
            report
                .fallback_counts
                .get(MultirateFallbackReason::StalePlan),
            1
        );
        assert_eq!(
            report
                .fallback_counts
                .get(MultirateFallbackReason::DeadlineExpired),
            2
        );
        assert_eq!(
            report
                .fallback_counts
                .get(MultirateFallbackReason::WatchdogExpired),
            1
        );
        assert_eq!(report.jitter.max_abs_nanos, 8_000_000);
    }

    #[test]
    fn ac1_system_monotonic_clock_measures_real_wall_time_steps() {
        let clock = SystemMonotonicClock::new();
        let config = TemporalTelemetryConfig::new(
            "system_wall_time_probe",
            Duration::from_millis(5),
            Duration::from_millis(20),
        )
        .unwrap();
        let mut recorder = TemporalTelemetryRecorder::new(clock, config);

        for i in 0..8_u64 {
            let (_res, rec) = recorder.measure_step(Tick::new(i), None, None, || {
                // Trabajo breve determinista en CPU sin E/S.
                let mut acc = 0_u64;
                for x in 0..10_000_u64 {
                    acc = acc.wrapping_add(x);
                }
                acc
            });
            assert_eq!(rec.tick, Tick::new(i));
        }

        let report = recorder.report();
        assert_eq!(report.total_samples, 8);
        assert!(report.p50_nanos <= report.p95_nanos);
        assert!(report.p95_nanos <= report.p99_nanos);
        assert!(report.p99_nanos <= report.max_observed_nanos);
        assert_eq!(report.histogram.total_samples(), 8);
    }

    #[test]
    fn ac2_report_includes_platform_load_deadlines_histogram_transitions_and_non_wcet_notice() {
        let clock = VirtualMonotonicClock::new();
        let config = TemporalTelemetryConfig::new(
            "control_loop_100hz",
            Duration::from_millis(10),
            Duration::from_millis(10),
        )
        .unwrap();
        let mut recorder = TemporalTelemetryRecorder::new(clock, config);

        // Nominal -> DeadlineMissed -> DegradedMissAndFallback(StalePlan) -> Nominal
        recorder.record_sample(
            Tick::new(0),
            MonotonicInstant::from_nanos(2_000_000),
            Duration::from_millis(2),
            0,
            None,
        );
        recorder.record_sample(
            Tick::new(1),
            MonotonicInstant::from_nanos(14_000_000),
            Duration::from_millis(12),
            2_000_000,
            None,
        );
        recorder.record_sample(
            Tick::new(2),
            MonotonicInstant::from_nanos(28_000_000),
            Duration::from_millis(14),
            4_000_000,
            Some(MultirateFallbackReason::StalePlan),
        );
        recorder.record_sample(
            Tick::new(3),
            MonotonicInstant::from_nanos(31_000_000),
            Duration::from_millis(3),
            -1_000_000,
            None,
        );

        let report = recorder.report();
        assert_eq!(report.platform.os, std::env::consts::OS);
        assert_eq!(report.platform.arch, std::env::consts::ARCH);
        assert_eq!(report.load_label, "control_loop_100hz");
        assert_eq!(report.target_period_nanos, 10_000_000);
        assert_eq!(report.deadline_nanos, 10_000_000);

        let bucket_sum: u64 = report.histogram.buckets().iter().map(|b| b.count).sum();
        assert_eq!(bucket_sum, 4);
        assert_eq!(report.histogram.counts().len(), HISTOGRAM_BUCKET_COUNT);

        assert_eq!(report.transitions.len(), 3);
        assert_eq!(report.transitions[0].from, TemporalComplianceState::Nominal);
        assert_eq!(
            report.transitions[0].to,
            TemporalComplianceState::DeadlineMissed
        );
        assert_eq!(
            report.transitions[1].to,
            TemporalComplianceState::DegradedMissAndFallback(MultirateFallbackReason::StalePlan)
        );
        assert_eq!(report.transitions[2].to, TemporalComplianceState::Nominal);
        assert_eq!(report.current_state, TemporalComplianceState::Nominal);

        assert_eq!(report.non_wcet_notice, NO_WCET_CLAIM_NOTICE);
        assert!(report.non_wcet_notice.contains("WCET"));
        assert_eq!(
            report.non_realtime_notice,
            STRICT_REALTIME_NON_CERTIFICATION_NOTICE
        );
    }

    #[test]
    fn ac3_bounded_memory_under_high_volume_and_decoupled_from_slow_render_or_io() {
        let clock = VirtualMonotonicClock::new();
        let window_cap = NonZeroUsize::new(64).unwrap();
        let transition_cap = NonZeroUsize::new(16).unwrap();
        let config = TemporalTelemetryConfig::new(
            "high_volume_bounded_test",
            Duration::from_millis(1),
            Duration::from_millis(1),
        )
        .unwrap()
        .with_window_capacity(window_cap)
        .with_transition_capacity(transition_cap);

        let mut recorder = TemporalTelemetryRecorder::new(clock, config);

        // 20_000 muestras alternando nominal y miss+fallback para forzar miles de transiciones.
        for i in 0..20_000_u64 {
            let (lat_us, fb) = if i % 2 == 0 {
                (500, None)
            } else if i == 10_001 {
                (5_000, Some(MultirateFallbackReason::DeadlineExpired))
            } else {
                (1_500, Some(MultirateFallbackReason::StaleObservation))
            };
            recorder.record_sample(
                Tick::new(i),
                MonotonicInstant::from_nanos(u128::from(i) * 1_000_000),
                Duration::from_micros(lat_us),
                0,
                fb,
            );
        }

        let report = recorder.report();
        assert_eq!(report.total_samples, 20_000);
        assert_eq!(report.window_capacity, 64);
        assert_eq!(report.window_samples, 64);
        assert_eq!(report.window_overwrites, 20_000 - 64);
        assert_eq!(report.transitions.len(), 16);
        assert!(report.transition_overwrites > 0);
        assert_eq!(report.max_observed_nanos, 5_000_000);
        assert_eq!(report.histogram.total_samples(), 20_000);

        // Verificar desacoplamiento de un consumidor lento de render/I/O en otro hilo:
        // el productor publica snapshots sin bloquearse esperando al consumidor lento.
        let shared_snapshot: Arc<Mutex<Option<TemporalComplianceReport>>> =
            Arc::new(Mutex::new(None));
        let consumer_slot = Arc::clone(&shared_snapshot);
        let consumer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            consumer_slot.lock().unwrap().take()
        });

        // Ruta de registro publica el reporte con try_lock/overwrite no bloqueante.
        if let Ok(mut guard) = shared_snapshot.try_lock() {
            *guard = Some(report);
        }
        let consumed = consumer.join().unwrap();
        assert!(consumed.is_some());
        assert_eq!(consumed.unwrap().total_samples, 20_000);
    }
}
