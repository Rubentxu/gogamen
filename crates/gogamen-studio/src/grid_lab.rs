//! GridWorld en Studio (GOG-110): primer slice vertical con el motor Grid real.
//!
//! - El único motor Grid es el lab `gogamen-lab-grid` ([`GridWorldEnv`]); aquí no hay dinámica.
//! - [`GridRollout`] es el lado motor (headless-capable, sin tipos de Studio): política
//!   determinista sembrada + hash de traza idéntico al de `run_grid_episode`.
//! - [`GridWorldView`] / [`project_grid_scene`]: proyector puro de una vista inmutable a `Scene2D`.
//! - [`GridLabSession`] ([`LabSession`]) avanza el rollout *sólo* al sincronizarse con el
//!   `SessionHost` (`StudioShell::sync_active`): `Step` -> un paso lógico, `Reset` -> mundo inicial.
//!   El viewport/render nunca avanza la simulación.
//! - Si el episodio termina (meta/peligro) el tick del host puede seguir creciendo, pero el mundo
//!   Grid se queda en su estado terminal hasta `Reset`.

use std::sync::{Arc, Mutex, MutexGuard};

use gogamen_core::{
    Action, ActionSpace, DeterminismMetadata, DeterministicRng, DiscreteAction, DomainError,
    EnvironmentError, EnvironmentObserver, EpistemicKind, MasterSeed, RngSource, StreamKey,
    Terminal, Tick, TraceHasher,
};
use gogamen_lab_grid::world::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldEnv};
use gogamen_lab_grid::{CellKind, GridMap, GridPos};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};
use gogamen_sdk::extensions::ExtensionId;

use crate::control::Observation;
use crate::shell::{LabEntry, LabSession};

/// Id del lab en el registro del shell.
pub const GRID_LAB_ID: &str = "lab.grid";

/// Vista inmutable del mundo Grid (entrada del proyector).
#[derive(Debug, Clone, PartialEq)]
pub struct GridWorldView {
    pub map: GridMap,
    pub agent: GridPos,
    pub tick: Tick,
    pub total_reward: f64,
    pub last_reward: f64,
    pub collisions: u64,
    pub terminal: Terminal,
}

impl GridWorldView {
    pub fn of(env: &GridWorldEnv) -> Self {
        Self {
            map: env.config().map.clone(),
            agent: env.position(),
            tick: env.tick(),
            total_reward: env.total_reward(),
            last_reward: env.last_reward(),
            collisions: env.collisions(),
            terminal: env.terminal(),
        }
    }
}

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite grid coordinate")
}

/// Proyección pura: celda `(x, y)` -> cuadrado unidad con `y` hacia arriba (`-y`).
/// Capas: `Environment` (celdas, código de celda en `state_code`), `Labels` (`#`, `G`, `H`),
/// `Decisions` (agente como círculo).
pub fn project_grid_scene(view: &GridWorldView) -> Scene2D {
    let mut scene = Scene2D::new(
        SceneId::new(110),
        view.tick,
        EpistemicKind::Observed,
        CoordinateSpace::meters(),
    );
    for layer in [LayerId::Environment, LayerId::Labels, LayerId::Decisions] {
        scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }
    let (w, h) = (view.map.width(), view.map.height());
    let mut next_id = 1u64;
    let mut id = || {
        let n = next_id;
        next_id += 1;
        SceneNodeId::new(n)
    };
    for y in 0..h {
        for x in 0..w {
            let kind = view.map.cell(GridPos::new(x, y)).unwrap_or(CellKind::Empty);
            let (fx, fy) = (x as f64, -(y as f64) - 1.0);
            let rect = ScenePrimitive::rect(pt(fx, fy), 1.0, 1.0).expect("unit rect");
            let node = SceneNode::new(id(), None, EpistemicKind::Observed, rect, true)
                .with_state_code(kind.as_code() as u32);
            scene.add_node(LayerId::Environment, node).expect("node");
            let glyph = match kind {
                CellKind::Wall => Some("#"),
                CellKind::Goal => Some("G"),
                CellKind::Hazard => Some("H"),
                CellKind::Empty => None,
            };
            if let Some(g) = glyph {
                let lab = ScenePrimitive::label(pt(fx + 0.4, fy + 0.4), g);
                let node = SceneNode::new(id(), None, EpistemicKind::Observed, lab, false);
                scene.add_node(LayerId::Labels, node).expect("node");
            }
        }
    }
    let a = view.agent;
    let circle = ScenePrimitive::circle(pt(a.x as f64 + 0.5, -(a.y as f64) - 0.5), 0.35)
        .expect("agent circle");
    let node = SceneNode::new(id(), None, EpistemicKind::Observed, circle, true);
    scene.add_node(LayerId::Decisions, node).expect("node");
    scene
}

/// Registro lógico de un paso.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    /// Tick lógico del mundo Grid tras el paso (1-based).
    pub tick: u64,
    pub action: usize,
    pub position: GridPos,
    pub reward: f64,
    pub total_reward: f64,
    pub terminal: Terminal,
    /// Hash de traza acumulado tras el paso (comparable con `run_grid_episode`).
    pub trace_hash: u64,
}

/// Motor headless: mundo Grid real + política determinista sembrada
/// (`seed.derive(Episode(0))`, acción = `next_u64 % 5`, la misma que `run_grid_episode`).
pub struct GridRollout {
    config: GridWorldConfig,
    seed: MasterSeed,
    env: GridWorldEnv,
    rng: DeterministicRng,
    hasher: TraceHasher,
    records: Vec<StepRecord>,
}

impl GridRollout {
    pub fn new(config: GridWorldConfig, seed: MasterSeed) -> Result<Self, DomainError> {
        let env = GridWorldEnv::new(config.clone())?;
        Ok(Self {
            config,
            seed,
            env,
            rng: seed.derive(StreamKey::Episode(0)),
            hasher: TraceHasher::with_metadata(&DeterminismMetadata::current(seed)),
            records: Vec::new(),
        })
    }

    /// Restaura el mundo inicial sembrado (misma secuencia que una instancia nueva).
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone(), self.seed).expect("config already validated");
    }

    pub fn is_terminal(&self) -> bool {
        self.env.terminal().is_terminal()
    }

    pub fn steps(&self) -> u64 {
        self.records.len() as u64
    }

    pub fn records(&self) -> &[StepRecord] {
        &self.records
    }

    pub fn trace_hash(&self) -> u64 {
        self.hasher.finish()
    }

    pub fn view(&self) -> GridWorldView {
        GridWorldView::of(&self.env)
    }

    /// Un paso lógico. `Ok(None)` si el episodio ya terminó.
    pub fn step(&mut self) -> Result<Option<StepRecord>, DomainError> {
        if self.is_terminal() {
            return Ok(None);
        }
        let obs = EnvironmentObserver::observe(&self.env).map_err(|_| {
            DomainError::InvalidTransition {
                reason: "failed to observe grid environment",
            }
        })?;
        let idx = (self.rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
        let action = Action::Discrete(DiscreteAction::new(idx, GRID_ACTION_CARDINALITY)?);
        let _: &ActionSpace = gogamen_core::Actuator::action_space(&self.env);
        let outcome =
            gogamen_core::Actuator::apply(&mut self.env, &action).map_err(|e| match e {
                EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
                EnvironmentError::InvalidAction(d) => d,
                _ => DomainError::InvalidTransition {
                    reason: "failed to apply grid action",
                },
            })?;
        let h = &mut self.hasher;
        h.write_tick(self.env.tick());
        h.write_observation(&obs);
        h.write_u64(idx as u64);
        h.write_observation(outcome.observation());
        h.write_f64(self.env.last_reward());
        h.write_terminal(outcome.terminal());
        let rec = StepRecord {
            tick: self.env.tick().as_u64(),
            action: idx,
            position: self.env.position(),
            reward: self.env.last_reward(),
            total_reward: self.env.total_reward(),
            terminal: outcome.terminal(),
            trace_hash: h.finish(),
        };
        self.records.push(rec.clone());
        Ok(Some(rec))
    }
}

/// Estado compartido sólo-lectura de una sesión Grid (para observación y tests).
#[derive(Debug, Clone, PartialEq)]
pub struct GridSnapshot {
    pub scene: Scene2D,
    pub view: GridWorldView,
    pub records: Vec<StepRecord>,
    pub trace_hash: u64,
}

/// Sonda de observación de la sesión Grid (clonable, sólo lectura).
#[derive(Clone)]
pub struct GridProbe {
    shared: Arc<Mutex<Option<GridSnapshot>>>,
}

impl GridProbe {
    fn lock(&self) -> MutexGuard<'_, Option<GridSnapshot>> {
        self.shared.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Último snapshot publicado por la sesión activa (`None` si no hay sesión creada).
    pub fn snapshot(&self) -> Option<GridSnapshot> {
        self.lock().clone()
    }

    /// Observación combinada: control (GOG-90) + escena proyectada del mundo.
    pub fn observe_with(&self, control: Observation) -> GridObservation {
        GridObservation {
            control,
            grid: self.snapshot(),
        }
    }
}

/// Observación del lab Grid: la del puerto de control más la escena proyectada.
#[derive(Debug, Clone, PartialEq)]
pub struct GridObservation {
    pub control: Observation,
    pub grid: Option<GridSnapshot>,
}

/// Sesión Grid del shell. Sólo avanza el rollout en [`LabSession::sync_to`].
pub struct GridLabSession {
    rollout: GridRollout,
    generation: u64,
    scene: Scene2D,
    probe: GridProbe,
}

impl GridLabSession {
    fn publish(&mut self) {
        let view = self.rollout.view();
        self.scene = project_grid_scene(&view);
        *self.probe.lock() = Some(GridSnapshot {
            scene: self.scene.clone(),
            view,
            records: self.rollout.records().to_vec(),
            trace_hash: self.rollout.trace_hash(),
        });
    }
}

impl LabSession for GridLabSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, observation: &Observation) {
        if observation.session.generation != self.generation {
            self.generation = observation.session.generation;
            self.rollout.reset();
        }
        while self.rollout.steps() < observation.tick.as_u64() && !self.rollout.is_terminal() {
            if !matches!(self.rollout.step(), Ok(Some(_))) {
                break;
            }
        }
        self.publish();
    }
}

/// Entrada de registro del lab Grid con sonda de observación.
pub fn grid_lab_entry_with_probe(
    config: GridWorldConfig,
    seed: MasterSeed,
) -> Result<(LabEntry, GridProbe), DomainError> {
    GridRollout::new(config.clone(), seed)?; // valida config una vez
    let probe = GridProbe {
        shared: Arc::new(Mutex::new(None)),
    };
    let p = probe.clone();
    let entry = LabEntry::new(
        ExtensionId::new(GRID_LAB_ID),
        "GridWorld",
        Arc::new(move || {
            let rollout = GridRollout::new(config.clone(), seed).expect("validated config");
            let scene = project_grid_scene(&rollout.view());
            let mut s = GridLabSession {
                rollout,
                generation: 0,
                scene,
                probe: p.clone(),
            };
            s.publish();
            Box::new(s)
        }),
    );
    Ok((entry, probe))
}

/// Entrada de registro del lab Grid (sin sonda).
pub fn grid_lab_entry(config: GridWorldConfig, seed: MasterSeed) -> Result<LabEntry, DomainError> {
    grid_lab_entry_with_probe(config, seed).map(|(e, _)| e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    use crate::shell::StudioShell;
    use gogamen_lab_grid::run_grid_episode;

    const SEED: u64 = 42;
    const N: u64 = 12;

    fn cfg() -> GridWorldConfig {
        GridWorldConfig::default()
    }

    fn custom_cfg() -> GridWorldConfig {
        let map = GridMap::from_ascii(&["S..#", ".#..", "...G"]).unwrap();
        GridWorldConfig {
            map,
            ..GridWorldConfig::default()
        }
    }

    fn headless(config: GridWorldConfig, seed: u64, n: u64) -> Vec<StepRecord> {
        let mut r = GridRollout::new(config, MasterSeed::new(seed)).unwrap();
        for _ in 0..n {
            if r.step().unwrap().is_none() {
                break;
            }
        }
        r.records().to_vec()
    }

    struct Ui {
        shell: StudioShell,
        probe: GridProbe,
        next: u64,
    }

    impl Ui {
        fn new(config: GridWorldConfig, seed: u64) -> Self {
            let (entry, probe) = grid_lab_entry_with_probe(config, MasterSeed::new(seed)).unwrap();
            let mut shell = StudioShell::new(vec![entry], 400.0, 300.0).unwrap();
            shell.select_lab(GRID_LAB_ID).unwrap();
            Self {
                shell,
                probe,
                next: 1,
            }
        }
        fn send(&mut self, action: ControlAction) -> crate::control::Ack {
            let port = self.shell.control_port().unwrap().unwrap();
            let session = self.shell.observation_port().unwrap().observe().session;
            let id = CommandId(self.next);
            self.next += 1;
            let ack = port.submit(ControlCommand {
                id,
                session,
                action,
            });
            self.shell.sync_active();
            ack
        }
        fn svg(&self) -> String {
            self.shell.render_svg().unwrap()
        }
    }

    #[test]
    fn ac1_headless_matches_lab_run_grid_episode_hash_per_step() {
        let recs = headless(cfg(), SEED, N);
        assert_eq!(recs.len() as u64, N);
        for (i, r) in recs.iter().enumerate() {
            let ep = run_grid_episode(i as u64 + 1, MasterSeed::new(SEED), false).unwrap();
            assert_eq!(ep.trace_hash, r.trace_hash, "step {}", i + 1);
            assert_eq!(ep.final_pos, r.position);
            assert_eq!(ep.total_reward, r.total_reward);
        }
        assert_eq!(recs, headless(cfg(), SEED, N), "repeatable");
        assert_ne!(recs, headless(cfg(), SEED + 1, N), "seed matters");
    }

    #[test]
    fn ac1_studio_step_by_step_equals_headless() {
        for config in [cfg(), custom_cfg()] {
            let expected = headless(config.clone(), SEED, N);
            let mut ui = Ui::new(config, SEED);
            for i in 0..expected.len() {
                let ack = ui.send(ControlAction::Step);
                assert_eq!(ack.outcome, AckOutcome::Applied);
                assert_eq!(ack.tick.as_u64(), i as u64 + 1);
                let snap = ui.probe.snapshot().unwrap();
                assert_eq!(snap.records, expected[..=i], "records after step {}", i + 1);
                assert_eq!(snap.trace_hash, expected[i].trace_hash);
            }
        }
    }

    #[test]
    fn ac2_pause_step_reset_with_real_acks_and_svg() {
        let mut ui = Ui::new(cfg(), SEED);
        let svg0 = ui.svg();
        assert_eq!(svg0, ui.svg(), "render is pure and byte-stable");
        let obs0 = ui.shell.observation_port().unwrap().observe();
        assert_eq!(obs0.tick, Tick::ZERO);

        let a = ui.send(ControlAction::Pause);
        assert_eq!(
            (a.outcome, a.revision, a.tick.as_u64()),
            (AckOutcome::Applied, 1, 0)
        );
        assert_eq!(ui.svg(), svg0, "pause does not move the agent");

        // Step hasta que el agente se mueva (la política puede chocar con muros).
        let mut svg_prev = svg0.clone();
        let mut moved = false;
        let mut prev_pos = cfg().map.start();
        for i in 0..N {
            let a = ui.send(ControlAction::Step);
            assert_eq!(a.outcome, AckOutcome::Applied);
            assert_eq!((a.revision, a.tick.as_u64()), (i + 2, i + 1));
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.view.tick.as_u64(), i + 1, "exactly one grid step");
            assert_eq!(snap.scene, project_grid_scene(&snap.view));
            let svg = ui.svg();
            if svg != svg_prev {
                moved = true;
            }
            let pos = snap.records.last().unwrap().position;
            assert_eq!(
                svg != svg_prev,
                pos != prev_pos,
                "SVG tracks agent position"
            );
            prev_pos = pos;
            svg_prev = svg;
        }
        assert!(moved, "SVG changed after stepping");
        assert_ne!(ui.svg(), svg0);

        let obs = ui.shell.observation_port().unwrap().observe();
        let generation = obs.session.generation;
        let a = ui.send(ControlAction::Reset);
        assert_eq!(a.outcome, AckOutcome::Applied);
        assert_eq!(a.tick, Tick::ZERO);
        let obs = ui
            .probe
            .observe_with(ui.shell.observation_port().unwrap().observe());
        assert_eq!(obs.control.session.generation, generation + 1);
        let snap = obs.grid.unwrap();
        assert!(snap.records.is_empty());
        assert_eq!(
            snap.trace_hash,
            GridRollout::new(cfg(), MasterSeed::new(SEED))
                .unwrap()
                .trace_hash()
        );
        assert_eq!(ui.svg(), svg0, "reset restores the seeded initial world");

        // tras Reset la secuencia se repite idéntica
        let again = headless(cfg(), SEED, 3);
        for _ in 0..3 {
            ui.send(ControlAction::Step);
        }
        assert_eq!(ui.probe.snapshot().unwrap().records, again);
    }

    #[test]
    fn ac2_step_while_running_rejected_and_world_unchanged() {
        let mut ui = Ui::new(cfg(), SEED);
        ui.send(ControlAction::Resume);
        let before = ui.probe.snapshot().unwrap();
        let a = ui.send(ControlAction::Step);
        assert!(matches!(a.outcome, AckOutcome::Rejected { .. }));
        assert_eq!(ui.probe.snapshot().unwrap(), before);
        // pump del motor avanza el mundo vía sync
        let host = ui.shell.engine_handle().unwrap();
        assert_eq!(host.pump(3), 3);
        ui.shell.sync_active();
        assert_eq!(
            ui.probe.snapshot().unwrap().records,
            headless(cfg(), SEED, 3)
        );
    }

    #[test]
    fn scene_contains_cells_obstacles_goal_and_agent() {
        let ui = Ui::new(custom_cfg(), SEED);
        let scene = ui.probe.snapshot().unwrap().scene;
        let layer = |id| scene.layers().iter().find(|l| l.id() == id).unwrap();
        let cells = layer(LayerId::Environment).nodes();
        assert_eq!(cells.len(), 12);
        let walls = cells.iter().filter(|n| n.state_code() == Some(1)).count();
        let goals = cells.iter().filter(|n| n.state_code() == Some(2)).count();
        assert_eq!((walls, goals), (2, 1));
        assert_eq!(layer(LayerId::Decisions).nodes().len(), 1);
        assert_eq!(layer(LayerId::Labels).nodes().len(), 3);
        let svg = ui.svg();
        assert!(svg.contains("<circle") && svg.contains(">G</text>") && svg.contains(">#</text>"));
    }

    // GOG-146: la simulación no depende del refresco gráfico ni se observa obsoleta.

    fn gog_146_ui_running(config: GridWorldConfig, max_per_pump: Option<u64>) -> Ui {
        let mut ui = Ui::new(config, SEED);
        if let Some(m) = max_per_pump {
            ui.shell.engine_handle().unwrap().set_max_ticks_per_pump(m);
        }
        assert_eq!(ui.send(ControlAction::Resume).outcome, AckOutcome::Applied);
        ui
    }

    fn gog_146_tick(ui: &Ui) -> u64 {
        ui.shell.observation_port().unwrap().observe().tick.as_u64()
    }

    #[test]
    fn gog_146_no_viewer_ticks_and_hashes_match_headless() {
        for config in [cfg(), custom_cfg()] {
            let expected = headless(config.clone(), SEED, 64);
            // Visor cerrado: el motor avanza ticks sin ningún sync/render.
            let mut ui = gog_146_ui_running(config.clone(), None);
            let host = ui.shell.engine_handle().unwrap();
            host.close_viewer();
            for _ in 0..30 {
                assert_eq!(host.pump(1), 1);
            }
            let obs = ui.shell.observation_port().unwrap().observe();
            assert_eq!(obs.tick.as_u64(), 30);
            assert!(!obs.viewer_attached);
            assert_eq!(obs.run_state, crate::control::RunState::Running);
            // Sin sync la sesión no avanzó sola (el render no simula) ...
            assert!(ui.probe.snapshot().unwrap().records.is_empty());
            // ... y un único sync alcanza el mismo estado que el lab headless.
            ui.shell.sync_active();
            let want = &expected[..expected.len().min(30)];
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.records, want);
            // hash por tick idéntico
            for (i, r) in want.iter().enumerate() {
                assert_eq!(snap.records[i].trace_hash, r.trace_hash, "tick {}", i + 1);
            }
        }
    }

    #[test]
    fn gog_146_per_tick_sync_hash_equals_headless() {
        let expected = headless(cfg(), SEED, 40);
        let mut ui = gog_146_ui_running(cfg(), None);
        let host = ui.shell.engine_handle().unwrap();
        for t in 1..=40usize {
            host.pump(1);
            ui.shell.sync_active();
            let snap = ui.probe.snapshot().unwrap();
            let n = t.min(expected.len());
            assert_eq!(snap.records, expected[..n], "tick {t}");
            assert_eq!(snap.view.tick.as_u64(), n as u64, "no stale view at {t}");
            let _ = ui.svg();
        }
    }

    #[test]
    fn gog_146_slow_or_skipped_viewer_loses_no_records_nor_changes_physics() {
        let expected = headless(custom_cfg(), SEED, 90);
        let total = 90u64;
        let mut reference = None;
        // periodos de sync: cada tick, raro, y nunca hasta el final; con tope de pump pequeño.
        for (period, cap) in [(1u64, None), (7, None), (1000, None), (13, Some(5u64))] {
            let mut ui = gog_146_ui_running(custom_cfg(), cap);
            let host = ui.shell.engine_handle().unwrap();
            let mut done = 0u64;
            let mut i = 0u64;
            while done < total {
                let out = host.pump_bounded(3);
                done += out.executed;
                i += 1;
                if i.is_multiple_of(period) {
                    ui.shell.sync_active();
                    let snap = ui.probe.snapshot().unwrap();
                    let n = (done as usize).min(expected.len());
                    assert_eq!(snap.records, expected[..n], "period {period} tick {done}");
                }
            }
            // drena backlog residual
            while host.pump_bounded(0).pending > 0 {}
            let end = gog_146_tick(&ui);
            ui.shell.sync_active();
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.records, expected[..(end as usize).min(expected.len())]);
            assert_eq!(snap.view.tick.as_u64(), end.min(expected.len() as u64));
            if end >= total {
                match &reference {
                    None => reference = Some(snap.trace_hash),
                    Some(h) => {
                        assert_eq!(*h, snap.trace_hash, "period {period} changed physics")
                    }
                }
            }
        }
        assert!(reference.is_some());
    }

    #[test]
    fn gog_146_reset_between_skipped_syncs_replays_from_seed() {
        let mut ui = gog_146_ui_running(cfg(), None);
        let host = ui.shell.engine_handle().unwrap();
        host.pump(5);
        ui.shell.sync_active();
        host.pump(6); // sin sync
        // Reset + reanudar + más ticks, todo sin sync intermedio.
        assert_eq!(ui.send(ControlAction::Reset).outcome, AckOutcome::Applied);
        let a = ui.send(ControlAction::Resume);
        assert_eq!(a.outcome, AckOutcome::Applied);
        host.pump(4);
        ui.shell.sync_active();
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.records, headless(cfg(), SEED, 4));
        assert_eq!(gog_146_tick(&ui), 4);
    }

    fn strip_comments_and_tests(src: &str) -> String {
        let cut = src.find("#[cfg(test)]").unwrap_or(src.len());
        src[..cut]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn ac3_no_grid_stepping_in_viewport_and_single_grid_engine() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let step_tokens = [
            "GridWorldEnv",
            "GridWorldModel",
            "gogamen_lab_grid",
            "GridRollout",
            "step_grid_dynamics",
            ".apply(",
            ".step(",
        ];
        for f in ["viewport.rs", "camera.rs", "shell.rs"] {
            let text = strip_comments_and_tests(&std::fs::read_to_string(src.join(f)).unwrap());
            for t in step_tokens {
                assert!(!text.contains(t), "{f} contains `{t}`");
            }
        }
        // único motor: ningún módulo de studio define dinámica Grid propia.
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let text = strip_comments_and_tests(&std::fs::read_to_string(&path).unwrap());
            assert!(!text.contains("fn step_grid_dynamics"), "{path:?}");
            assert!(!text.contains("impl World for"), "{path:?}");
            assert!(!text.contains("impl Actuator for"), "{path:?}");
        }
        // el motor vive en el lab y studio lo importa vía path dependency
        let manifest = std::fs::read_to_string(src.join("..").join("Cargo.toml")).unwrap();
        assert!(manifest.contains("gogamen-lab-grid = { path = \"../../labs/grid\" }"));
        let lab = std::fs::read_to_string(src.join("../../../labs/grid/src/world.rs")).unwrap();
        assert!(
            lab.contains("fn step_grid_dynamics") && lab.contains("impl World for GridWorldEnv")
        );
        // el proyector sólo recibe una vista inmutable
        let me = std::fs::read_to_string(src.join("grid_lab.rs")).unwrap();
        assert!(me.contains("pub fn project_grid_scene(view: &GridWorldView)"));
    }
}
