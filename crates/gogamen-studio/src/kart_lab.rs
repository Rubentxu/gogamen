//! Kart en Studio (GOG-111): pista, coche, sensores y futuros FMC del mundo Kart real.
//!
//! - El único mundo y planificador Kart son los del lab `gogamen-lab-racing` ([`KartEnv`] y
//!   [`KartFmcController`], el mismo par que usa `KartFmcController::run_episode`); aquí no hay
//!   dinámica ni planificación propias.
//! - [`KartRollout`] es el lado motor (headless, sin tipos de Studio): por paso hace
//!   `decide_from_env` + `Actuator::apply`, el bucle interno de `run_episode`, y mantiene un hash
//!   de traza que incluye los cambios autorizados del presupuesto del planificador.
//! - [`KartWorldView`] / [`project_kart_layers`] / [`project_kart_scene`]: proyector puro de una
//!   vista inmutable a `Scene2D`. Pista, coche y rumbo salen de la vista; sensores, futuros
//!   elegido/rechazados, trayectoria ejecutada, colisiones y leyenda salen de la capa común
//!   `gogamen-projection::overlay` (no hay overlay específico de Kart).
//! - [`KartLabSession`] ([`LabSession`]) avanza el rollout *sólo* en `sync_to`; el render nunca
//!   avanza la simulación. Los parámetros (GOG-95) se aplican al sincronizar, antes de decidir
//!   (frontera de tick): el presupuesto autorizado se registra en [`KartRollout::budget_changes`].
//!
//! Los "futuros" son los de la decisión que produjo la acción aplicada (desde la pose previa):
//! la acción elegida frente a las acciones raíz rechazadas (ver
//! [`KartFmcController::candidate_futures`]). Los rayos de sensor son geometría de la pista
//! ([`TrackMap::cast_ray`]) desde la pose actual.

use std::sync::{Arc, Mutex, MutexGuard};

use gogamen_core::{
    Actuator, DeterminismMetadata, DomainError, EntityId, EnvironmentError, EnvironmentObserver,
    EpistemicKind, MasterSeed, Terminal, TerminalReason, Tick, TraceHasher,
};
use gogamen_lab_racing::{
    KartCandidateFuture, KartControl, KartEnv, KartFmcController, KartFmcControllerConfig,
    KartRunError, KartState, KinematicBicycleConfig, Point2 as TrackPoint, TrackMap,
};
use gogamen_projection::{
    DecisionOverlayFrame, ExecutedTrajectory, FutureKind, FutureTrajectorySample, LayerId,
    LayerVisibility, OverlayConfig, OverlayLegend, OverlayStats, Point2, ProjectedOverlay, Scene2D,
    SceneId, SceneNode, SceneNodeId, ScenePrimitive, SensorRaySample, TerminalMarker,
    TerminalOutcome, project_decision_overlay,
};
use gogamen_sdk::extensions::ExtensionId;

use crate::control::Observation;
use crate::inspector::{
    EntityRef, EntityView, Field, FieldDescriptor, FieldSchema, FieldValue, Lod, Provenance,
    SnapshotView,
};
use crate::params::{
    Mutability, ParamDescriptor, ParamId, ParamKind, ParamSchema, ParamUnit, ParamValue,
};
use crate::shell::{LabEntry, LabSession};

/// Id del lab en el registro del shell.
pub const KART_LAB_ID: &str = "lab.kart";
/// Id de la entidad coche (seleccionable en la escena y en el inspector).
pub const KART_CAR_ID: u64 = 1;
/// Parámetro mutable: presupuesto de transiciones simuladas por decisión.
pub const PARAM_BUDGET: &str = "planner.max_transitions";
/// Parámetro inmutable en caliente: horizonte del planificador.
pub const PARAM_HORIZON: &str = "planner.horizon";
/// Número de rayos de sensor del coche.
pub const SENSOR_RAY_COUNT: usize = 5;
/// Alcance de los rayos de sensor (m).
pub const SENSOR_RANGE: f64 = 12.0;
/// Abanico de rayos respecto al rumbo (rad): `-FAN..=+FAN`.
pub const SENSOR_FAN: f64 = std::f64::consts::FRAC_PI_3;

const SCENE_ID: u64 = 111;
const WALL_NODE_BASE: u64 = 1;
const CHECKPOINT_NODE_BASE: u64 = 1_000;
const CAR_NODE: u64 = 5_000;
const HEADING_NODE: u64 = 5_001;
const ACTION_LABEL_NODE: u64 = 5_002;
const STATUS_LABEL_NODE: u64 = 5_003;
const COLLISION_NODE: u64 = 5_004;
const CHECKPOINT_PENDING_CODE: u32 = 1;
const CHECKPOINT_CLEARED_CODE: u32 = 2;
const WALL_CODE: u32 = 10;
const CAR_CODE: u32 = 20;
const COLLISION_CODE: u32 = 900;

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite kart coordinate")
}

fn to_pt(p: TrackPoint) -> Point2 {
    pt(p.x, p.y)
}

/// Lectura de un rayo de sensor (geometría de pista desde la pose del coche).
#[derive(Debug, Clone, PartialEq)]
pub struct KartSensorReading {
    pub origin: TrackPoint,
    /// Punto de impacto, o el extremo del alcance si no hubo impacto.
    pub end: TrackPoint,
    pub hit: bool,
    pub distance: f64,
}

/// Vista inmutable del mundo Kart (entrada del proyector y del inspector).
#[derive(Debug, Clone, PartialEq)]
pub struct KartWorldView {
    pub track: TrackMap,
    pub kart: KartState,
    pub kart_radius: f64,
    pub tick: Tick,
    pub checkpoints_cleared: usize,
    pub collided: bool,
    pub terminal: Terminal,
    /// Última acción aplicada (`None` antes del primer paso).
    pub applied: Option<KartControl>,
    /// Presupuesto de transiciones por decisión vigente.
    pub budget: usize,
    /// Horizonte del planificador (inmutable).
    pub horizon: usize,
    /// Estado del plan de la última decisión.
    pub plan_status: Option<String>,
    /// Transiciones simuladas por la última decisión.
    pub transitions_used: usize,
    /// Posiciones ejecutadas (pose inicial incluida).
    pub trail: Vec<TrackPoint>,
    /// Futuros de la última decisión (elegido + rechazados).
    pub futures: Vec<KartCandidateFuture>,
    pub sensors: Vec<KartSensorReading>,
}

fn sensor_readings(track: &TrackMap, kart: KartState) -> Vec<KartSensorReading> {
    let origin = TrackPoint {
        x: kart.x,
        y: kart.y,
    };
    (0..SENSOR_RAY_COUNT)
        .map(|i| {
            let frac = i as f64 / (SENSOR_RAY_COUNT - 1) as f64;
            let angle = kart.yaw - SENSOR_FAN + 2.0 * SENSOR_FAN * frac;
            let hit = track.cast_ray(origin, angle, SENSOR_RANGE);
            let distance = hit.unwrap_or(SENSOR_RANGE);
            KartSensorReading {
                origin,
                end: TrackPoint {
                    x: origin.x + angle.cos() * distance,
                    y: origin.y + angle.sin() * distance,
                },
                hit: hit.is_some(),
                distance,
            }
        })
        .collect()
}

fn outcome_of(terminal: Terminal) -> TerminalOutcome {
    match terminal {
        Terminal::Ongoing => TerminalOutcome::HorizonReached,
        Terminal::Terminated(TerminalReason::GoalReached) => TerminalOutcome::Success,
        Terminal::Terminated(TerminalReason::Failure) => TerminalOutcome::Failure,
        Terminal::Terminated(_) => TerminalOutcome::Unknown,
    }
}

/// Configuración del overlay común para Kart: presupuesto holgado, todos los futuros exportados.
pub fn kart_overlay_config(visibility: LayerVisibility) -> OverlayConfig {
    let mut cfg = OverlayConfig {
        visibility,
        ..OverlayConfig::default()
    };
    cfg.budget.max_future_samples = 64;
    cfg.budget.max_sensor_samples = 64;
    cfg.budget.max_terminal_markers = 64;
    cfg.budget.max_bytes_per_frame = 1 << 20;
    cfg.budget.export_all_walkers = true;
    cfg
}

fn overlay_frame(view: &KartWorldView) -> DecisionOverlayFrame {
    let tick = view.tick;
    let mut frame = DecisionOverlayFrame::new(SceneId::new(SCENE_ID), tick, tick.as_u64());
    if view.trail.len() >= 2 {
        let pts = view.trail.iter().copied().map(to_pt).collect();
        let executed = ExecutedTrajectory::new(0, pts, Some(EntityId::new(KART_CAR_ID)))
            .expect("trail has >= 2 points");
        frame = frame.with_executed_trajectory(executed);
    }
    let futures = view
        .futures
        .iter()
        .enumerate()
        .filter(|(_, f)| f.points.len() >= 2)
        .map(|(i, f)| {
            let kind = if f.chosen {
                FutureKind::SelectedFuture
            } else {
                FutureKind::RejectedFuture
            };
            let pts: Vec<Point2> = f.points.iter().copied().map(to_pt).collect();
            let end = *pts.last().expect("non-empty");
            let marker = TerminalMarker::new(
                end,
                outcome_of(f.terminal),
                (pts.len() - 1) as u32,
                f.hold_return,
            )
            .expect("finite return");
            FutureTrajectorySample::new(i as u64, kind, pts, f.hold_return)
                .expect("finite score")
                .with_terminal(marker)
        })
        .collect();
    frame = frame.with_futures(futures);
    let sensors = view
        .sensors
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let conf = if s.hit { 1.0 } else { 0.25 };
            SensorRaySample::new(i as u64, to_pt(s.origin), to_pt(s.end), s.hit, conf)
                .expect("confidence in range")
        })
        .collect();
    frame = frame.with_sensors(sensors);
    if view.collided {
        let here = pt(view.kart.x, view.kart.y);
        frame = frame.with_terminals(vec![
            TerminalMarker::new(
                here,
                TerminalOutcome::Failure,
                view.tick.as_u64() as u32,
                0.0,
            )
            .expect("finite"),
        ]);
    }
    frame
}

/// Proyección pura con capas comunes: devuelve escena + leyenda + estadísticas del overlay.
///
/// Capas: `Environment` (muros, checkpoints; `state_code` 10 muro, 1/2 checkpoint pendiente/
/// superado), `Custom(1)` trayectoria ejecutada, `Sensors` rayos, `Custom(2)` futuros rechazados,
/// `Decisions` futuro elegido + coche (círculo seleccionable `EntityId(KART_CAR_ID)`), rumbo y
/// marca de colisión, `Custom(3)` marcadores terminales, `Labels` leyenda + acción aplicada.
pub fn project_kart_layers_with(
    view: &KartWorldView,
    visibility: LayerVisibility,
) -> ProjectedOverlay {
    let mut overlay =
        project_decision_overlay(&overlay_frame(view), &kart_overlay_config(visibility))
            .expect("kart overlay fits its budget");
    let scene = &mut overlay.scene;
    let mut add = |layer: LayerId, node: SceneNode| {
        scene.add_node(layer, node).expect("unique kart node id");
    };
    for (i, w) in view.track.walls().iter().enumerate() {
        let line = ScenePrimitive::polyline(vec![to_pt(w.start), to_pt(w.end)], false)
            .expect("wall polyline");
        add(
            LayerId::Environment,
            SceneNode::new(
                SceneNodeId::new(WALL_NODE_BASE + i as u64),
                None,
                EpistemicKind::Observed,
                line,
                false,
            )
            .with_state_code(WALL_CODE),
        );
    }
    for (i, c) in view.track.checkpoints().iter().enumerate() {
        let circle = ScenePrimitive::circle(to_pt(c.center), c.radius).expect("checkpoint circle");
        let code = if i < view.checkpoints_cleared {
            CHECKPOINT_CLEARED_CODE
        } else {
            CHECKPOINT_PENDING_CODE
        };
        add(
            LayerId::Environment,
            SceneNode::new(
                SceneNodeId::new(CHECKPOINT_NODE_BASE + i as u64),
                None,
                EpistemicKind::Observed,
                circle,
                false,
            )
            .with_state_code(code),
        );
    }
    let here = pt(view.kart.x, view.kart.y);
    let car = ScenePrimitive::circle(here, view.kart_radius).expect("car circle");
    add(
        LayerId::Decisions,
        SceneNode::new(
            SceneNodeId::new(CAR_NODE),
            Some(EntityId::new(KART_CAR_ID)),
            EpistemicKind::Observed,
            car,
            true,
        )
        .with_state_code(CAR_CODE),
    );
    let nose = pt(
        view.kart.x + view.kart.yaw.cos() * view.kart_radius * 2.5,
        view.kart.y + view.kart.yaw.sin() * view.kart_radius * 2.5,
    );
    add(
        LayerId::Decisions,
        SceneNode::new(
            SceneNodeId::new(HEADING_NODE),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::polyline(vec![here, nose], false).expect("heading polyline"),
            false,
        ),
    );
    if view.collided {
        add(
            LayerId::Decisions,
            SceneNode::new(
                SceneNodeId::new(COLLISION_NODE),
                None,
                EpistemicKind::Observed,
                ScenePrimitive::circle(here, view.kart_radius * 2.0).expect("collision circle"),
                false,
            )
            .with_state_code(COLLISION_CODE),
        );
    }
    let action_text = match view.applied {
        Some(c) => format!(
            "action steer={:.3} accel={:.3}",
            c.steering_angle, c.throttle_accel
        ),
        None => "action none".to_owned(),
    };
    add(
        LayerId::Labels,
        SceneNode::new(
            SceneNodeId::new(ACTION_LABEL_NODE),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::label(pt(view.kart.x + 0.8, view.kart.y + 0.8), action_text),
            false,
        ),
    );
    let status = format!(
        "tick {} ckpt {}/{} budget {} {}{}",
        view.tick.as_u64(),
        view.checkpoints_cleared,
        view.track.checkpoints().len(),
        view.budget,
        view.plan_status.as_deref().unwrap_or("-"),
        if view.collided { " COLLIDED" } else { "" },
    );
    let b = (view.track.bounds_min(), view.track.bounds_max());
    add(
        LayerId::Labels,
        SceneNode::new(
            SceneNodeId::new(STATUS_LABEL_NODE),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::label(pt(b.0.x + 0.5, b.1.y + 1.0), status),
            false,
        ),
    );
    overlay
}

/// [`project_kart_layers_with`] con todas las capas visibles.
pub fn project_kart_layers(view: &KartWorldView) -> ProjectedOverlay {
    project_kart_layers_with(view, LayerVisibility::all_visible())
}

/// Proyección pura de la vista a `Scene2D` (y arriba, metros).
pub fn project_kart_scene(view: &KartWorldView) -> Scene2D {
    project_kart_layers(view).scene
}

/// Cambio autorizado del presupuesto registrado en la traza del rollout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetChange {
    /// Pasos lógicos ya ejecutados cuando se aplicó (afecta a la decisión siguiente).
    pub at_step: u64,
    pub from: usize,
    pub to: usize,
}

/// Registro lógico de un paso.
#[derive(Debug, Clone, PartialEq)]
pub struct KartStepRecord {
    /// Tick lógico del mundo tras el paso (1-based).
    pub tick: u64,
    pub control: KartControl,
    pub kart: KartState,
    pub checkpoints_cleared: usize,
    pub terminal: Terminal,
    /// Presupuesto con el que se decidió.
    pub budget: usize,
    pub transitions_used: usize,
    /// Hash de traza acumulado tras el paso.
    pub trace_hash: u64,
}

struct LastDecision {
    control: KartControl,
    plan_status: String,
    transitions_used: usize,
    futures: Vec<KartCandidateFuture>,
}

/// Motor headless: mundo Kart real + planificador FMC real (`seed` = semilla del planificador,
/// como `KartFmcControllerConfig::with_dt_and_seed`).
pub struct KartRollout {
    kinematics: KinematicBicycleConfig,
    track: TrackMap,
    seed: MasterSeed,
    env: KartEnv,
    controller: KartFmcController,
    hasher: TraceHasher,
    records: Vec<KartStepRecord>,
    budget_changes: Vec<BudgetChange>,
    trail: Vec<TrackPoint>,
    last: Option<LastDecision>,
    halted: bool,
}

impl KartRollout {
    pub fn new(
        kinematics: KinematicBicycleConfig,
        track: TrackMap,
        seed: MasterSeed,
    ) -> Result<Self, DomainError> {
        let env = KartEnv::new(kinematics.clone(), track.clone())?;
        let config = KartFmcControllerConfig::with_dt_and_seed(kinematics.dt, seed);
        let controller = KartFmcController::new(kinematics.clone(), track.clone(), config)?;
        let start = env.kart_state();
        Ok(Self {
            kinematics,
            track,
            seed,
            env,
            controller,
            hasher: TraceHasher::with_metadata(&DeterminismMetadata::current(seed)),
            records: Vec::new(),
            budget_changes: Vec::new(),
            trail: vec![TrackPoint {
                x: start.x,
                y: start.y,
            }],
            last: None,
            halted: false,
        })
    }

    /// Restaura el mundo inicial sembrado y el presupuesto por defecto.
    pub fn reset(&mut self) {
        *self = Self::new(self.kinematics.clone(), self.track.clone(), self.seed)
            .expect("config already validated");
    }

    pub fn is_terminal(&self) -> bool {
        self.halted || self.env.terminal().is_terminal()
    }

    pub fn steps(&self) -> u64 {
        self.records.len() as u64
    }

    pub fn records(&self) -> &[KartStepRecord] {
        &self.records
    }

    pub fn trace_hash(&self) -> u64 {
        self.hasher.finish()
    }

    pub fn env(&self) -> &KartEnv {
        &self.env
    }

    pub fn controller(&self) -> &KartFmcController {
        &self.controller
    }

    pub fn budget(&self) -> usize {
        self.controller.config().max_transitions_per_decision
    }

    pub fn horizon(&self) -> usize {
        self.controller.config().planner.horizon()
    }

    /// Cambios de presupuesto aplicados (traza autorizada).
    pub fn budget_changes(&self) -> &[BudgetChange] {
        &self.budget_changes
    }

    /// Aplica un nuevo presupuesto desde la siguiente decisión; queda en el log y en el hash.
    /// Sin efecto (ni traza) si el valor no cambia.
    pub fn set_budget(&mut self, budget: usize) -> Result<(), DomainError> {
        let from = self.budget();
        if budget == from {
            return Ok(());
        }
        self.controller.set_max_transitions_per_decision(budget)?;
        self.budget_changes.push(BudgetChange {
            at_step: self.steps(),
            from,
            to: budget,
        });
        self.hasher.write_str("budget");
        self.hasher.write_u64(self.steps());
        self.hasher.write_u64(budget as u64);
        Ok(())
    }

    pub fn view(&self) -> KartWorldView {
        let kart = self.env.kart_state();
        KartWorldView {
            track: self.track.clone(),
            kart,
            kart_radius: self.kinematics.kart_radius,
            tick: self.env.tick(),
            checkpoints_cleared: self.env.checkpoints_cleared(),
            collided: self.env.collided(),
            terminal: self.env.terminal(),
            applied: self.last.as_ref().map(|l| l.control),
            budget: self.budget(),
            horizon: self.horizon(),
            plan_status: self.last.as_ref().map(|l| l.plan_status.clone()),
            transitions_used: self.last.as_ref().map_or(0, |l| l.transitions_used),
            trail: self.trail.clone(),
            futures: self
                .last
                .as_ref()
                .map(|l| l.futures.clone())
                .unwrap_or_default(),
            sensors: sensor_readings(&self.track, kart),
        }
    }

    /// Un paso lógico (`decide_from_env` + `apply`). `Ok(None)` si el episodio ya terminó o el
    /// planificador no autorizó ninguna acción (`run_episode` termina igual con
    /// `FailedNoAuthorizedAction`).
    pub fn step(&mut self) -> Result<Option<KartStepRecord>, KartRunError> {
        if self.is_terminal() {
            return Ok(None);
        }
        let obs = EnvironmentObserver::observe(&self.env).map_err(KartRunError::Environment)?;
        let start = gogamen_core::State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
            .map_err(|e| KartRunError::Model(gogamen_core::ModelError::TransitionFailed(e)))?;
        let report = self
            .controller
            .decide_from_env(&self.env)
            .map_err(KartRunError::Model)?;
        let (Some(action), Some(control)) = (report.action.clone(), report.control) else {
            self.halted = true;
            return Ok(None);
        };
        let futures = self
            .controller
            .candidate_futures(&start, &report)
            .map_err(KartRunError::Model)?;
        let budget = self.budget();
        let outcome = Actuator::apply(&mut self.env, &action).map_err(|e| match e {
            EnvironmentError::AlreadyTerminal => {
                KartRunError::Environment(EnvironmentError::AlreadyTerminal)
            }
            other => KartRunError::Environment(other),
        })?;
        let kart = self.env.kart_state();
        let h = &mut self.hasher;
        h.write_tick(self.env.tick());
        h.write_observation(&obs);
        h.write_f64(control.steering_angle);
        h.write_f64(control.throttle_accel);
        h.write_u64(report.transitions_used as u64);
        h.write_u64(budget as u64);
        h.write_observation(outcome.observation());
        h.write_terminal(outcome.terminal());
        self.trail.push(TrackPoint {
            x: kart.x,
            y: kart.y,
        });
        self.last = Some(LastDecision {
            control,
            plan_status: format!("{:?}", report.plan_status),
            transitions_used: report.transitions_used,
            futures,
        });
        let rec = KartStepRecord {
            tick: self.env.tick().as_u64(),
            control,
            kart,
            checkpoints_cleared: self.env.checkpoints_cleared(),
            terminal: outcome.terminal(),
            budget,
            transitions_used: report.transitions_used,
            trace_hash: self.hasher.finish(),
        };
        self.records.push(rec.clone());
        Ok(Some(rec))
    }
}

/// Esquema de parámetros del lab Kart: presupuesto mutable + horizonte inmutable en caliente.
pub fn kart_param_schema(rollout: &KartRollout) -> ParamSchema {
    ParamSchema::new(vec![
        ParamDescriptor {
            id: ParamId::new(PARAM_BUDGET),
            label: "Planner budget".into(),
            kind: ParamKind::Int {
                min: 1,
                max: 4096,
                step: 1,
            },
            unit: ParamUnit::Custom("transitions"),
            mutability: Mutability::Mutable,
            default: ParamValue::Int(rollout.budget() as i64),
        },
        ParamDescriptor {
            id: ParamId::new(PARAM_HORIZON),
            label: "Planner horizon".into(),
            kind: ParamKind::Int {
                min: 1,
                max: 256,
                step: 1,
            },
            unit: ParamUnit::Custom("steps"),
            mutability: Mutability::Immutable,
            default: ParamValue::Int(rollout.horizon() as i64),
        },
    ])
    .expect("kart param schema is valid")
}

/// Estado compartido sólo-lectura de una sesión Kart (para observación y tests).
#[derive(Debug, Clone, PartialEq)]
pub struct KartSnapshot {
    pub scene: Scene2D,
    pub legend: OverlayLegend,
    pub overlay_stats: OverlayStats,
    pub view: KartWorldView,
    pub records: Vec<KartStepRecord>,
    pub budget_changes: Vec<BudgetChange>,
    pub trace_hash: u64,
}

/// Sonda de observación de la sesión Kart (clonable, sólo lectura).
#[derive(Clone)]
pub struct KartProbe {
    shared: Arc<Mutex<Option<KartSnapshot>>>,
}

impl KartProbe {
    fn lock(&self) -> MutexGuard<'_, Option<KartSnapshot>> {
        self.shared.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Último snapshot publicado por la sesión activa (`None` si no hay sesión creada).
    pub fn snapshot(&self) -> Option<KartSnapshot> {
        self.lock().clone()
    }

    /// Observación combinada: control (GOG-90) + escena proyectada del mundo.
    pub fn observe_with(&self, control: Observation) -> KartObservation {
        KartObservation {
            control,
            kart: self.snapshot(),
        }
    }
}

/// Observación del lab Kart: la del puerto de control más la escena proyectada.
#[derive(Debug, Clone, PartialEq)]
pub struct KartObservation {
    pub control: Observation,
    pub kart: Option<KartSnapshot>,
}

/// Sesión Kart del shell. Sólo avanza el rollout en [`LabSession::sync_to`].
pub struct KartLabSession {
    rollout: KartRollout,
    generation: u64,
    scene: Scene2D,
    probe: KartProbe,
}

impl KartLabSession {
    fn publish(&mut self) {
        let view = self.rollout.view();
        let overlay = project_kart_layers(&view);
        self.scene = overlay.scene.clone();
        *self.probe.lock() = Some(KartSnapshot {
            scene: overlay.scene,
            legend: overlay.legend,
            overlay_stats: overlay.stats,
            view,
            records: self.rollout.records().to_vec(),
            budget_changes: self.rollout.budget_changes().to_vec(),
            trace_hash: self.rollout.trace_hash(),
        });
    }
}

impl LabSession for KartLabSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, observation: &Observation) {
        if observation.session.generation != self.generation {
            self.generation = observation.session.generation;
            self.rollout.reset();
        }
        // Frontera de tick: los parámetros vigentes ya fueron autorizados por el host.
        let budget = observation
            .params
            .iter()
            .find(|(id, _)| id.0 == PARAM_BUDGET)
            .and_then(|(_, v)| match v {
                ParamValue::Int(b) if *b >= 1 => Some(*b as usize),
                _ => None,
            });
        if let Some(b) = budget {
            let _ = self.rollout.set_budget(b);
        }
        while self.rollout.steps() < observation.tick.as_u64() && !self.rollout.is_terminal() {
            if !matches!(self.rollout.step(), Ok(Some(_))) {
                break;
            }
        }
        self.publish();
    }
}

/// Entrada de registro del lab Kart con sonda de observación y esquema de parámetros.
pub fn kart_lab_entry_with_probe(
    kinematics: KinematicBicycleConfig,
    track: TrackMap,
    seed: MasterSeed,
) -> Result<(LabEntry, KartProbe), DomainError> {
    let schema = kart_param_schema(&KartRollout::new(kinematics.clone(), track.clone(), seed)?);
    let probe = KartProbe {
        shared: Arc::new(Mutex::new(None)),
    };
    let p = probe.clone();
    let entry = LabEntry::new(
        ExtensionId::new(KART_LAB_ID),
        "Kart",
        Arc::new(move || {
            let rollout = KartRollout::new(kinematics.clone(), track.clone(), seed)
                .expect("validated config");
            let scene = project_kart_scene(&rollout.view());
            let mut s = KartLabSession {
                rollout,
                generation: 0,
                scene,
                probe: p.clone(),
            };
            s.publish();
            Box::new(s)
        }),
    )
    .with_param_schema(schema);
    Ok((entry, probe))
}

/// Entrada de registro del lab Kart (sin sonda).
pub fn kart_lab_entry(
    kinematics: KinematicBicycleConfig,
    track: TrackMap,
    seed: MasterSeed,
) -> Result<LabEntry, DomainError> {
    kart_lab_entry_with_probe(kinematics, track, seed).map(|(e, _)| e)
}

// ---------------------------------------------------------------------------------------------
// Cableado al inspector (GOG-92)
// ---------------------------------------------------------------------------------------------

const CHECKPOINT_ENTITY_BASE: u64 = 1_000;
const FUTURE_ENTITY_BASE: u64 = 2_000;

/// Id de la entidad checkpoint `i`.
pub fn kart_checkpoint_id(i: usize) -> u64 {
    CHECKPOINT_ENTITY_BASE + i as u64
}

/// Id de la entidad futuro `i` (índice en `KartWorldView::futures`).
pub fn kart_future_id(i: usize) -> u64 {
    FUTURE_ENTITY_BASE + i as u64
}

/// Epoch de entidades Kart: sesión (`epoch`, `generation`); cambia con cada `Reset`.
pub fn kart_epoch(obs: &Observation) -> u64 {
    (obs.session.epoch << 32) | (obs.session.generation & 0xffff_ffff)
}

/// Descriptores del lab Kart.
pub fn kart_schema() -> FieldSchema {
    let mut s = FieldSchema::new();
    s.register(FieldDescriptor::new("position", "Position", Lod::Minimal))
        .register(FieldDescriptor::new("heading", "Heading", Lod::Minimal))
        .register(FieldDescriptor::new("tick", "Tick", Lod::Minimal))
        .register(FieldDescriptor::new("speed", "Speed", Lod::Standard))
        .register(FieldDescriptor::new("steering", "Steering", Lod::Standard))
        .register(FieldDescriptor::new(
            "throttle_accel",
            "Throttle accel",
            Lod::Standard,
        ))
        .register(FieldDescriptor::new(
            "checkpoints",
            "Checkpoints",
            Lod::Standard,
        ))
        .register(FieldDescriptor::new("terminal", "Terminal", Lod::Standard))
        .register(FieldDescriptor::new("collided", "Collided", Lod::Standard))
        .register(FieldDescriptor::new("budget", "Planner budget", Lod::Full))
        .register(FieldDescriptor::new(
            "plan_status",
            "Plan status",
            Lod::Full,
        ))
        .register(FieldDescriptor::new(
            "transitions_used",
            "Transitions used",
            Lod::Full,
        ))
        .register(FieldDescriptor::new(
            "sensor_min",
            "Nearest wall",
            Lod::Full,
        ))
        .register(FieldDescriptor::new("chosen", "Chosen", Lod::Minimal))
        .register(FieldDescriptor::new(
            "hold_return",
            "Hold return",
            Lod::Standard,
        ))
        .register(FieldDescriptor::new("support", "Support", Lod::Full))
        .register(FieldDescriptor::new("cleared", "Cleared", Lod::Minimal));
    s
}

/// Adapta la vista real del mundo Kart + la observación de sesión a un [`SnapshotView`]:
/// una entidad coche (id [`KART_CAR_ID`]), una por checkpoint y una por futuro candidato.
pub fn kart_snapshot_view(world: &KartWorldView, obs: &Observation) -> SnapshotView {
    let tick = world.tick.as_u64();
    let epoch = kart_epoch(obs);
    let mut car = vec![
        Field::new(
            "position",
            FieldValue::Vector(vec![world.kart.x, world.kart.y]),
            tick,
        )
        .unit("m"),
        Field::new("heading", FieldValue::Number(world.kart.yaw), tick).unit("rad"),
        Field::new("tick", FieldValue::Integer(tick as i64), tick).unit("tick"),
        Field::new("speed", FieldValue::Number(world.kart.speed), tick).unit("m/s"),
        Field::new(
            "checkpoints",
            FieldValue::Integer(world.checkpoints_cleared as i64),
            tick,
        ),
        Field::new("collided", FieldValue::Bool(world.collided), tick),
        Field::new(
            "terminal",
            FieldValue::Text(format!("{:?}", world.terminal)),
            tick,
        ),
        Field::new("budget", FieldValue::Integer(world.budget as i64), tick)
            .unit("transitions")
            .provenance(Provenance::Derived),
        Field::new(
            "transitions_used",
            FieldValue::Integer(world.transitions_used as i64),
            tick,
        )
        .unit("transitions")
        .provenance(Provenance::Derived),
    ];
    if let Some(a) = world.applied {
        car.push(Field::new("steering", FieldValue::Number(a.steering_angle), tick).unit("rad"));
        car.push(
            Field::new("throttle_accel", FieldValue::Number(a.throttle_accel), tick).unit("m/s2"),
        );
    }
    if let Some(s) = &world.plan_status {
        car.push(
            Field::new("plan_status", FieldValue::Text(s.clone()), tick)
                .provenance(Provenance::Derived),
        );
    }
    if let Some(min) = world.sensors.iter().map(|s| s.distance).reduce(f64::min) {
        car.push(Field::new("sensor_min", FieldValue::Number(min), tick).unit("m"));
    }
    let mut entities = vec![EntityView {
        entity: EntityRef::new(KART_CAR_ID, epoch),
        kind: "car".into(),
        fields: car,
    }];
    for (i, c) in world.track.checkpoints().iter().enumerate() {
        entities.push(EntityView {
            entity: EntityRef::new(kart_checkpoint_id(i), epoch),
            kind: "checkpoint".into(),
            fields: vec![
                Field::new(
                    "position",
                    FieldValue::Vector(vec![c.center.x, c.center.y]),
                    tick,
                )
                .unit("m"),
                Field::new(
                    "cleared",
                    FieldValue::Bool(i < world.checkpoints_cleared),
                    tick,
                ),
            ],
        });
    }
    for (i, f) in world.futures.iter().enumerate() {
        let mut fields = vec![
            Field::new("chosen", FieldValue::Bool(f.chosen), tick),
            Field::new("hold_return", FieldValue::Number(f.hold_return), tick)
                .provenance(Provenance::Derived),
            Field::new("support", FieldValue::Integer(f.support_count as i64), tick)
                .provenance(Provenance::Derived),
            Field::new(
                "terminal",
                FieldValue::Text(format!("{:?}", f.terminal)),
                tick,
            ),
        ];
        if let Some(c) = f.control {
            fields.push(
                Field::new("steering", FieldValue::Number(c.steering_angle), tick).unit("rad"),
            );
            fields.push(
                Field::new("throttle_accel", FieldValue::Number(c.throttle_accel), tick)
                    .unit("m/s2"),
            );
        }
        entities.push(EntityView {
            entity: EntityRef::new(kart_future_id(i), epoch),
            kind: "future".into(),
            fields,
        });
    }
    SnapshotView {
        revision: obs.revision,
        tick,
        entities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    use crate::inspector::{Inspector, InspectorStatus, VanishReason};
    use crate::param_ui::{UiPermissions, control_descriptors};
    use crate::params::{
        ParamChange, ParamCommand, ParamControlPort, ParamOutcome, ParamRejectReason, ParamRequest,
    };
    use crate::shell::StudioShell;
    use gogamen_projection::LegendCategory;

    const SEED: u64 = 7;
    const N: u64 = 6;

    fn kin() -> KinematicBicycleConfig {
        KinematicBicycleConfig::default()
    }

    fn headless(seed: u64, n: u64) -> KartRollout {
        let mut r = KartRollout::new(
            kin(),
            TrackMap::rectangular_circuit(),
            MasterSeed::new(seed),
        )
        .unwrap();
        for _ in 0..n {
            if r.step().unwrap().is_none() {
                break;
            }
        }
        r
    }

    struct Ui {
        shell: StudioShell,
        probe: KartProbe,
        next: u64,
    }

    impl Ui {
        fn new(seed: u64) -> Self {
            let (entry, probe) = kart_lab_entry_with_probe(
                kin(),
                TrackMap::rectangular_circuit(),
                MasterSeed::new(seed),
            )
            .unwrap();
            let mut shell = StudioShell::new(vec![entry], 640.0, 480.0).unwrap();
            shell.select_lab(KART_LAB_ID).unwrap();
            Self {
                shell,
                probe,
                next: 1,
            }
        }
        fn obs(&self) -> Observation {
            self.shell.observation_port().unwrap().observe()
        }
        fn send(&mut self, action: ControlAction) -> crate::control::Ack {
            let port = self.shell.control_port().unwrap().unwrap();
            let session = self.obs().session;
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
        fn set_budget(&mut self, value: ParamValue, param: &str) -> ParamOutcome {
            let id = CommandId(self.next);
            self.next += 1;
            let session = self.obs().session;
            self.shell
                .param_port()
                .unwrap()
                .submit_params(ParamCommand {
                    id,
                    session,
                    request: ParamRequest::Set(vec![ParamChange {
                        id: ParamId::new(param),
                        value,
                    }]),
                })
                .outcome
        }
        fn snapshot_view(&self) -> SnapshotView {
            kart_snapshot_view(&self.probe.snapshot().unwrap().view, &self.obs())
        }
    }

    #[test]
    fn ac1_scene_matches_snapshot_track_vehicle_action() {
        let mut ui = Ui::new(SEED);
        for _ in 0..N {
            assert_eq!(ui.send(ControlAction::Step).outcome, AckOutcome::Applied);
        }
        let snap = ui.probe.snapshot().unwrap();
        let v = &snap.view;
        assert_eq!(snap.scene, project_kart_scene(v));
        let node = |id: u64| {
            snap.scene
                .layers()
                .iter()
                .flat_map(|l| l.nodes())
                .find(|n| n.node_id() == SceneNodeId::new(id))
                .cloned()
        };
        // pista: un nodo por muro y checkpoint
        let walls = v.track.walls().len();
        for i in 0..walls {
            assert!(node(WALL_NODE_BASE + i as u64).is_some());
        }
        assert_eq!(
            snap.scene
                .layers()
                .iter()
                .flat_map(|l| l.nodes())
                .filter(|n| n.state_code() == Some(WALL_CODE))
                .count(),
            walls
        );
        // coche en la pose del snapshot y seleccionable
        let car = node(CAR_NODE).unwrap();
        assert!(car.is_selectable());
        assert_eq!(car.entity_id(), Some(EntityId::new(KART_CAR_ID)));
        let hit = snap
            .scene
            .pick_at_world(pt(v.kart.x, v.kart.y), 0.05)
            .unwrap()
            .unwrap();
        assert_eq!(hit.entity_id, Some(EntityId::new(KART_CAR_ID)));
        // la vista es el mundo real tras N pasos del planificador real
        let last = snap.records.last().unwrap();
        assert_eq!(snap.records.len() as u64, N);
        assert_eq!(v.kart, last.kart);
        assert_eq!(v.applied, Some(last.control));
        assert_eq!(v.kart, ui_world_kart(&headless(SEED, N)));
        // acción aplicada en el texto
        let txt = format!("{:?}", node(ACTION_LABEL_NODE).unwrap());
        assert!(txt.contains(&format!("{:.3}", last.control.steering_angle)));
        // sensores: un rayo por sensor, impacto = TrackMap::cast_ray
        assert_eq!(v.sensors.len(), SENSOR_RAY_COUNT);
        for s in &v.sensors {
            assert!(s.distance <= SENSOR_RANGE);
        }
        // futuros: exactamente uno elegido, el resto rechazados, con la acción aplicada
        let chosen: Vec<_> = v.futures.iter().filter(|f| f.chosen).collect();
        assert_eq!(chosen.len(), 1);
        assert_eq!(chosen[0].control, Some(last.control));
        let rejected = v.futures.iter().filter(|f| !f.chosen).count();
        assert!(rejected >= 1, "rejected futures are shown");
        // leyenda con las categorías
        for cat in [
            LegendCategory::SelectedFuture,
            LegendCategory::RejectedFuture,
            LegendCategory::SensorRay,
            LegendCategory::ExecutedTrajectory,
        ] {
            let e = snap.legend.find(cat).unwrap_or_else(|| panic!("{cat:?}"));
            assert!(e.visible && !e.label.is_empty());
        }
        assert_eq!(
            snap.legend
                .find(LegendCategory::SelectedFuture)
                .unwrap()
                .sample_count,
            1
        );
        assert_eq!(
            snap.legend
                .find(LegendCategory::RejectedFuture)
                .unwrap()
                .sample_count,
            rejected
        );
        assert_eq!(
            snap.legend
                .find(LegendCategory::SensorRay)
                .unwrap()
                .sample_count,
            SENSOR_RAY_COUNT
        );
        let svg = ui.shell.render_svg().unwrap();
        assert_eq!(svg, ui.shell.render_svg().unwrap(), "render is pure");
        assert!(svg.starts_with("<svg") && svg.contains("<circle"));
    }

    fn ui_world_kart(r: &KartRollout) -> KartState {
        r.env().kart_state()
    }

    #[test]
    fn ac3_studio_equals_headless_and_lab_controller_per_step() {
        let expected = headless(SEED, N);
        let mut ui = Ui::new(SEED);
        for i in 0..N as usize {
            ui.send(ControlAction::Step);
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.records, expected.records()[..=i]);
            assert_eq!(snap.trace_hash, expected.records()[i].trace_hash);
        }
        // mismo par mundo+planificador que el lab: decide_from_env + apply a mano
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(kin(), track.clone()).unwrap();
        let cfg = KartFmcControllerConfig::with_dt_and_seed(kin().dt, MasterSeed::new(SEED));
        let mut ctl = KartFmcController::new(kin(), track, cfg).unwrap();
        for rec in expected.records() {
            let report = ctl.decide_from_env(&env).unwrap();
            assert_eq!(report.control, Some(rec.control));
            Actuator::apply(&mut env, &report.action.unwrap()).unwrap();
            assert_eq!(env.kart_state(), rec.kart);
        }
        // misma semilla repite; otra semilla (misma pista) puede diferir pero se repite igual
        assert_eq!(headless(SEED, N).records(), expected.records());
        assert_eq!(headless(SEED, N).trace_hash(), expected.trace_hash());
    }

    #[test]
    fn ac3_run_episode_matches_rollout_final_state() {
        let track = TrackMap::rectangular_circuit();
        let cfg = KartFmcControllerConfig::with_dt_and_seed(kin().dt, MasterSeed::new(SEED));
        let mut ctl = KartFmcController::new(kin(), track.clone(), cfg).unwrap();
        let mut env = KartEnv::new(kin(), track.clone()).unwrap();
        let m = ctl.run_episode(&mut env, 8).unwrap();
        let mut r = KartRollout::new(kin(), track, MasterSeed::new(SEED)).unwrap();
        while r.steps() < m.episode_ticks as u64 && r.step().unwrap().is_some() {}
        assert_eq!(r.env().kart_state(), env.kart_state());
        assert_eq!(r.env().checkpoints_cleared(), env.checkpoints_cleared());
    }

    #[test]
    fn render_does_not_advance_and_pause_keeps_world() {
        let mut ui = Ui::new(SEED);
        let svg0 = ui.shell.render_svg().unwrap();
        ui.send(ControlAction::Pause);
        assert_eq!(ui.shell.render_svg().unwrap(), svg0);
        assert!(ui.probe.snapshot().unwrap().records.is_empty());
        ui.send(ControlAction::Step);
        assert_ne!(ui.shell.render_svg().unwrap(), svg0);
        ui.send(ControlAction::Reset);
        assert_eq!(ui.shell.render_svg().unwrap(), svg0, "reset restores world");
        assert!(ui.probe.snapshot().unwrap().records.is_empty());
    }

    #[test]
    fn ac2_selecting_car_reuses_inspector_and_layers() {
        let mut ui = Ui::new(SEED);
        for _ in 0..3 {
            ui.send(ControlAction::Step);
        }
        let snap = ui.probe.snapshot().unwrap();
        // seleccionar vía la escena (capas) y alimentar el inspector con la entidad
        let hit = snap
            .scene
            .pick_at_world(pt(snap.view.kart.x, snap.view.kart.y), 0.05)
            .unwrap()
            .unwrap();
        let entity = hit.entity_id.unwrap();
        let obs = ui.obs();
        let mut i = Inspector::new(kart_schema(), Lod::Full, 0);
        i.select(EntityRef::new(entity.as_u64(), kart_epoch(&obs)));
        let v = i.inspect(&ui.snapshot_view());
        assert_eq!(v.status, InspectorStatus::Live);
        assert_eq!(v.kind.as_deref(), Some("car"));
        let by = |n: &str| v.rows.iter().find(|r| r.name == n).unwrap().clone();
        assert_eq!(
            by("position").value,
            FieldValue::Vector(vec![snap.view.kart.x, snap.view.kart.y])
        );
        assert_eq!(by("heading").value, FieldValue::Number(snap.view.kart.yaw));
        assert_eq!(by("speed").value, FieldValue::Number(snap.view.kart.speed));
        let a = snap.view.applied.unwrap();
        assert_eq!(by("steering").value, FieldValue::Number(a.steering_angle));
        assert_eq!(
            by("budget").value,
            FieldValue::Integer(snap.view.budget as i64)
        );
        // futuro elegido accesible por el mismo inspector
        let fi = snap.view.futures.iter().position(|f| f.chosen).unwrap();
        i.select(EntityRef::new(kart_future_id(fi), kart_epoch(&obs)));
        let v = i.inspect(&ui.snapshot_view());
        assert_eq!(v.kind.as_deref(), Some("future"));
        assert_eq!(by_name(&v, "chosen"), FieldValue::Bool(true));
        // LOD mínimo oculta campos de planificador
        i.select(EntityRef::new(KART_CAR_ID, kart_epoch(&obs)));
        i.set_lod(Lod::Minimal);
        assert!(
            i.inspect(&ui.snapshot_view())
                .rows
                .iter()
                .all(|r| r.name != "budget")
        );
        // Reset: la selección vieja desaparece (reciclada)
        ui.send(ControlAction::Reset);
        let v = i.inspect(&ui.snapshot_view());
        assert!(matches!(
            v.status,
            InspectorStatus::Vanished(VanishReason::Recycled { .. })
        ));
    }

    fn by_name(v: &crate::inspector::InspectorView, n: &str) -> FieldValue {
        v.rows.iter().find(|r| r.name == n).unwrap().value.clone()
    }

    #[test]
    fn ac2_authorized_budget_change_is_traced_and_unauthorized_rejected() {
        let mut ui = Ui::new(SEED);
        let default_budget = ui.probe.snapshot().unwrap().view.budget;
        assert_eq!(default_budget, 32 * 16);
        ui.send(ControlAction::Step);
        // autorizado: se encola y se aplica en la frontera del siguiente Step
        let out = ui.set_budget(ParamValue::Int(64), PARAM_BUDGET);
        assert!(matches!(out, ParamOutcome::Queued { .. }), "{out:?}");
        ui.send(ControlAction::Step);
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.view.budget, 64);
        assert_eq!(snap.records[0].budget, default_budget);
        assert_eq!(snap.records[1].budget, 64);
        assert_eq!(snap.records[2].budget, 64);
        assert!(snap.records[1].transitions_used <= 64);
        assert_eq!(
            snap.budget_changes,
            vec![BudgetChange {
                at_step: 1,
                from: default_budget,
                to: 64
            }]
        );
        // log de parámetros del host (autoridad) coincide
        let events = ui.shell.observation_port().unwrap().param_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].param, ParamId::new(PARAM_BUDGET));
        assert_eq!(events[0].new, ParamValue::Int(64));
        // el cambio altera el hash de traza respecto a la ejecución sin cambio
        assert_ne!(snap.trace_hash, headless(SEED, 3).trace_hash());

        // no autorizados: inmutable, fuera de rango, tipo, desconocido
        let n_before = ui.probe.snapshot().unwrap().budget_changes.len();
        for (val, p, want) in [
            (
                ParamValue::Int(5),
                PARAM_HORIZON,
                ParamRejectReason::Immutable(ParamId::new(PARAM_HORIZON)),
            ),
            (
                ParamValue::Int(0),
                PARAM_BUDGET,
                ParamRejectReason::OutOfRange(ParamId::new(PARAM_BUDGET)),
            ),
            (
                ParamValue::Bool(true),
                PARAM_BUDGET,
                ParamRejectReason::WrongType(ParamId::new(PARAM_BUDGET)),
            ),
            (
                ParamValue::Int(1),
                "planner.nope",
                ParamRejectReason::UnknownParam(ParamId::new("planner.nope")),
            ),
        ] {
            assert_eq!(
                ui.set_budget(val, p),
                ParamOutcome::Rejected { reason: want }
            );
        }
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.budget_changes.len(), n_before);
        assert_eq!(snap.view.budget, 64);
        assert_eq!(ui.shell.observation_port().unwrap().param_events().len(), 1);
        assert_eq!(ui.obs().param_rejection_count, 4);
    }

    #[test]
    fn param_ui_exposes_budget_enabled_and_horizon_disabled() {
        let ui = Ui::new(SEED);
        let schema = ui.shell.observation_port().unwrap().param_schema();
        let d = control_descriptors(
            &schema,
            &UiPermissions {
                session_controllable: true,
            },
            &ui.obs().params,
        );
        let budget = d.iter().find(|c| c.id.0 == PARAM_BUDGET).unwrap();
        let horizon = d.iter().find(|c| c.id.0 == PARAM_HORIZON).unwrap();
        assert!(budget.enabled && !horizon.enabled);
        assert_eq!(budget.value, ParamValue::Int(512));
    }

    #[test]
    fn budget_is_applied_at_tick_boundary_via_pump_too() {
        let mut ui = Ui::new(SEED);
        ui.send(ControlAction::Resume);
        let out = ui.set_budget(ParamValue::Int(96), PARAM_BUDGET);
        assert!(matches!(out, ParamOutcome::Queued { .. }));
        ui.shell.engine_handle().unwrap().pump(3);
        ui.shell.sync_active();
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.view.budget, 96);
        assert_eq!(snap.budget_changes.len(), 1);
    }

    #[test]
    fn rollout_rejects_zero_budget_and_noop_is_not_traced() {
        let mut r = headless(SEED, 1);
        assert!(r.set_budget(0).is_err());
        let b = r.budget();
        r.set_budget(b).unwrap();
        assert!(r.budget_changes().is_empty());
    }

    #[test]
    fn collision_marker_when_car_hits_wall() {
        let mut v = headless(SEED, 2).view();
        v.collided = true;
        let scene = project_kart_scene(&v);
        let n = scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter(|n| n.state_code() == Some(COLLISION_CODE))
            .count();
        assert_eq!(n, 1);
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
    fn ac3_no_kart_dynamics_or_planner_in_renderer_modules_and_single_engine() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let tokens = [
            "KartEnv",
            "KartModel",
            "KartFmcController",
            "gogamen_lab_racing",
            "KartRollout",
            "kinematic_bicycle",
            ".apply(",
            ".step(",
        ];
        for f in ["viewport.rs", "camera.rs", "shell.rs"] {
            let text = strip_comments_and_tests(&std::fs::read_to_string(src.join(f)).unwrap());
            for t in tokens {
                assert!(!text.contains(t), "{f} contains `{t}`");
            }
        }
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let text = strip_comments_and_tests(&std::fs::read_to_string(&path).unwrap());
            assert!(!text.contains("impl World for"), "{path:?}");
            assert!(!text.contains("impl Actuator for"), "{path:?}");
            assert!(!text.contains("impl GenerativeModel for"), "{path:?}");
            assert!(!text.contains("fn kinematic_step"), "{path:?}");
            assert!(!text.contains("struct KartPlanner"), "{path:?}");
        }
        let manifest = std::fs::read_to_string(src.join("..").join("Cargo.toml")).unwrap();
        assert!(manifest.contains("gogamen-lab-racing = { path = \"../../labs/racing\" }"));
        let me = std::fs::read_to_string(src.join("kart_lab.rs")).unwrap();
        assert!(me.contains("pub fn project_kart_scene(view: &KartWorldView)"));
        // el overlay es la capa común, no un dibujo propio
        assert!(me.contains("project_decision_overlay("));
        let lab = std::fs::read_to_string(src.join("../../../labs/racing/src/lib.rs")).unwrap();
        assert!(lab.contains("KartFmcController") && lab.contains("KartEnv"));
    }
}
