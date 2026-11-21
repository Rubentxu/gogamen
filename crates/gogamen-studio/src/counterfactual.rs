//! Depurador contrafactual desde checkpoint (GOG-99), headless y sin Bevy/egui.
//!
//! Dada una [`Recording`] Grid (GOG-94) y un tick, [`Checkpoint::capture`] reconstruye el estado
//! del modelo en ese tick con el modelo ramificable real del núcleo
//! ([`GridWorldModel`] + `BranchableSnapshot`, GOG-5) verificándolo frame a frame contra lo
//! grabado. Cada ejecución ([`Checkpoint::run`]) hace `restore` del snapshot en un modelo nuevo:
//! nunca toca la grabación (se toma `&Recording`), ni un entorno vivo (aquí no existe ninguno),
//! y nunca rebobina nada físico ni produce efectos externos.
//!
//! - La acción original y la alternativa pasan por la guardia independiente de acción (GOG-7);
//!   una acción rechazada se registra como [`Rejection`] y **no** se ejecuta.
//! - La política original (semilla `Episode(0)`, `next_u64 % 5`) se regenera de la semilla
//!   grabada; en la rama alternativa sólo cambia la primera acción, el resto continúa igual.
//! - Un checkpoint insuficiente (hueco, tick fuera de rango, historia incompleta, estado que no
//!   coincide con lo grabado) se rechaza con [`CheckpointError`] tipado.
//! - [`CounterfactualView`] distingue **observación** (grabada) de **predicción** (modelo) y
//!   expone fidelidad por tick; [`project_counterfactual_scene`] es un proyector puro.

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, BranchableSnapshot, DeterministicRng,
    DiscreteAction, EpistemicKind, GuardContext, GuardDecision, GuardRejectionReason, MasterSeed,
    RngSource, SNAPSHOT_SCHEMA_VERSION, SimulationSnapshot, SnapshotError, State, StreamKey,
    Terminal, Tick,
};
use gogamen_lab_grid::world::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldModel};
use gogamen_lab_grid::{CellKind, GridMap, GridPos};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};

use crate::camera::{Camera2D, scene_bounds};
use crate::replay::{Frame, Gap, Recording};
use crate::viewport::SoftwareSvgBackend;

/// Nombre del modelo usado para predecir.
pub const MODEL_NAME: &str = "GridWorldModel";

/// Identificador del padre: FNV-1a 64 de los bytes serializados de la grabación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordingId(pub u64);

/// Hash FNV-1a 64 (determinista, sin dependencias).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl RecordingId {
    pub fn of(rec: &Recording) -> Self {
        Self(fnv1a(&rec.to_bytes()))
    }
}

/// Modelo con el que se predice (nombre + esquema de snapshot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelRef {
    pub name: &'static str,
    pub snapshot_schema: u32,
}

/// Qué acción inicial lleva la rama.
#[derive(Debug, Clone, PartialEq)]
pub enum BranchKind {
    /// Repite la acción grabada (la política original).
    Original,
    /// Sustituye la primera acción.
    Alternative(Action),
}

/// Cabecera de una rama: id del padre, tick, semilla y modelo (AC2).
#[derive(Debug, Clone, PartialEq)]
pub struct BranchHeader {
    pub id: u64,
    pub parent: RecordingId,
    pub tick: u64,
    pub seed: u64,
    pub model: ModelRef,
    pub kind: BranchKind,
}

/// Motivo por el que un checkpoint no es suficiente para ramificar.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointError {
    /// `tick` posterior al último tick grabado.
    TickOutOfRange { tick: u64, last_tick: u64 },
    /// El propio tick del checkpoint está en un hueco: no hay estado observado.
    MissingFrame(Gap),
    /// Hay un hueco antes del tick: el estado del modelo no es reconstruible/verificable.
    IncompleteHistory(Gap),
    /// El modelo no pudo construirse con la configuración grabada.
    ModelUnavailable(String),
    /// El snapshot del modelo es inválido o incompatible.
    Snapshot(SnapshotError),
    /// El modelo reconstruido no coincide con lo grabado en `tick` (`field`).
    ReconstructionMismatch { tick: u64, field: &'static str },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TickOutOfRange { tick, last_tick } => {
                write!(f, "tick {tick} out of range (last recorded {last_tick})")
            }
            Self::MissingFrame(g) => {
                write!(f, "no frame at checkpoint (gap {}..={})", g.from, g.to)
            }
            Self::IncompleteHistory(g) => {
                write!(
                    f,
                    "incomplete history before checkpoint (gap {}..={})",
                    g.from, g.to
                )
            }
            Self::ModelUnavailable(m) => write!(f, "model unavailable: {m}"),
            Self::Snapshot(e) => write!(f, "snapshot error: {e:?}"),
            Self::ReconstructionMismatch { tick, field } => {
                write!(
                    f,
                    "model state mismatches recording at tick {tick} ({field})"
                )
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Acción rechazada por la guardia: registrada, no ejecutada.
#[derive(Debug, Clone, PartialEq)]
pub struct Rejection {
    pub tick: u64,
    pub action: Action,
    pub reason: GuardRejectionReason,
}

/// Resultado de ejecutar una rama (datos propios; sin referencia al padre ni al modelo).
#[derive(Debug, Clone, PartialEq)]
pub struct BranchRun {
    pub header: BranchHeader,
    /// Frames predichos por el modelo tras el checkpoint (revisión = tick).
    pub predicted: Vec<Frame>,
    pub rejection: Option<Rejection>,
}

/// Resumen determinista de un frame (ignora `revision`).
fn frame_digest(frames: &[Frame]) -> u64 {
    let mut bytes = Vec::new();
    for f in frames {
        bytes.extend_from_slice(&f.tick.to_le_bytes());
        bytes.extend_from_slice(&(f.agent.x as u64).to_le_bytes());
        bytes.extend_from_slice(&(f.agent.y as u64).to_le_bytes());
        bytes.extend_from_slice(&f.last_reward.to_bits().to_le_bytes());
        bytes.extend_from_slice(&f.total_reward.to_bits().to_le_bytes());
        bytes.extend_from_slice(&f.collisions.to_le_bytes());
        bytes.push(match f.terminal {
            Terminal::Ongoing => 0,
            Terminal::Terminated(_) => 1,
        });
    }
    fnv1a(&bytes)
}

impl BranchRun {
    /// Hash de la traza predicha.
    pub fn trace_hash(&self) -> u64 {
        frame_digest(&self.predicted)
    }
}

fn same_state(a: &Frame, b: &Frame) -> bool {
    a.tick == b.tick
        && a.agent == b.agent
        && a.last_reward == b.last_reward
        && a.total_reward == b.total_reward
        && a.collisions == b.collisions
        && a.terminal == b.terminal
}

/// Frame (observable) del estado del modelo.
fn model_frame(state: &State, terminal: Terminal) -> Frame {
    let v = state.values();
    Frame {
        tick: state.tick().as_u64(),
        revision: state.tick().as_u64(),
        agent: GridPos::new(v[0] as usize, v[1] as usize),
        total_reward: v[2],
        last_reward: v[3],
        collisions: v[4] as u64,
        terminal,
    }
}

fn policy_action(rng: &mut DeterministicRng) -> Action {
    let idx = (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
    Action::Discrete(
        DiscreteAction::new(idx, GRID_ACTION_CARDINALITY).expect("index below cardinality"),
    )
}

/// Checkpoint verificado de una grabación (sólo lectura sobre el padre).
pub struct Checkpoint<'a> {
    rec: &'a Recording,
    parent: RecordingId,
    tick: u64,
    config: GridWorldConfig,
    snapshot: SimulationSnapshot,
    policy: DeterministicRng,
}

impl<'a> Checkpoint<'a> {
    /// Reconstruye y verifica el estado en `tick`; error tipado si es insuficiente.
    pub fn capture(rec: &'a Recording, tick: u64) -> Result<Self, CheckpointError> {
        if tick > rec.last_tick() {
            return Err(CheckpointError::TickOutOfRange {
                tick,
                last_tick: rec.last_tick(),
            });
        }
        for g in rec.gaps() {
            if g.from <= tick && tick <= g.to {
                return Err(CheckpointError::MissingFrame(*g));
            }
            if g.from < tick {
                return Err(CheckpointError::IncompleteHistory(*g));
            }
        }
        let config = GridWorldConfig {
            map: rec.map().clone(),
            ..GridWorldConfig::default()
        };
        let seed = MasterSeed::new(rec.seed());
        let mut model = GridWorldModel::new(config.clone(), seed)
            .map_err(|e| CheckpointError::ModelUnavailable(format!("{e:?}")))?;
        let mut policy = seed.derive(StreamKey::Episode(0));
        let check = |model: &GridWorldModel, at: u64| -> Result<(), CheckpointError> {
            let want = rec
                .frame_at(at)
                .ok_or(CheckpointError::ReconstructionMismatch {
                    tick: at,
                    field: "frame",
                })?;
            let got = model_frame(model.state(), model.terminal());
            let field = if got.tick != want.tick {
                "tick"
            } else if got.agent != want.agent {
                "agent"
            } else if got.total_reward != want.total_reward || got.last_reward != want.last_reward {
                "reward"
            } else if got.collisions != want.collisions {
                "collisions"
            } else if got.terminal != want.terminal {
                "terminal"
            } else {
                return Ok(());
            };
            Err(CheckpointError::ReconstructionMismatch { tick: at, field })
        };
        check(&model, 0)?;
        for at in 1..=tick {
            let action = policy_action(&mut policy);
            model
                .step_in_place(&action)
                .map_err(|_| CheckpointError::ReconstructionMismatch {
                    tick: at,
                    field: "step",
                })?;
            check(&model, at)?;
        }
        let snapshot = model.snapshot().map_err(CheckpointError::Snapshot)?;
        snapshot.verify().map_err(CheckpointError::Snapshot)?;
        Ok(Self {
            rec,
            parent: RecordingId::of(rec),
            tick,
            config,
            snapshot,
            policy,
        })
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn parent(&self) -> RecordingId {
        self.parent
    }

    /// Ejecuta una rama aislada hasta `horizon` pasos. `None` = acción original grabada;
    /// `Some(a)` = `a` sustituye sólo la primera acción. Cada ejecución usa un modelo nuevo
    /// restaurado desde el snapshot: dos ejecuciones iguales dan resultados idénticos y no se
    /// influyen entre sí. Toda acción (incluida la original) pasa por la guardia.
    pub fn run(&self, alternative: Option<Action>, horizon: u64) -> BranchRun {
        let seed = self.rec.seed();
        let mut model = GridWorldModel::new(self.config.clone(), MasterSeed::new(seed))
            .expect("config validated at capture");
        model
            .restore(&self.snapshot)
            .expect("snapshot verified at capture");
        let mut policy = self.policy.clone();
        let space = ActionSpace::discrete(GRID_ACTION_CARDINALITY).expect("non-empty space");
        let guard = ActionGuard::strict();
        let kind = alternative
            .clone()
            .map_or(BranchKind::Original, BranchKind::Alternative);
        let mut predicted = Vec::new();
        let mut rejection = None;
        for i in 0..horizon {
            if model.terminal().is_terminal() {
                break;
            }
            let drawn = policy_action(&mut policy);
            let action = match (i, &alternative) {
                (0, Some(a)) => a.clone(),
                _ => drawn,
            };
            let now = Tick::new(model.state().tick().as_u64());
            let proposal = ActionProposal::authorized(action.clone(), now);
            let ctx = GuardContext::now(&space, now, model.terminal());
            match guard.evaluate(&proposal, &ctx) {
                GuardDecision::Allow(a) | GuardDecision::Fallback(a) => {
                    if model.step_in_place(&a).is_err() {
                        break;
                    }
                    predicted.push(model_frame(model.state(), model.terminal()));
                }
                GuardDecision::NoOp => break,
                GuardDecision::Reject(reason) => {
                    rejection = Some(Rejection {
                        tick: now.as_u64(),
                        action,
                        reason,
                    });
                    break;
                }
            }
        }
        let mut header = BranchHeader {
            id: 0,
            parent: self.parent,
            tick: self.tick,
            seed,
            model: ModelRef {
                name: MODEL_NAME,
                snapshot_schema: SNAPSHOT_SCHEMA_VERSION,
            },
            kind,
        };
        header.id = fnv1a(
            format!(
                "{:x}|{}|{}|{:?}",
                header.parent.0, header.tick, seed, header.kind
            )
            .as_bytes(),
        );
        BranchRun {
            header,
            predicted,
            rejection,
        }
    }
}

/// Fidelidad de una predicción frente a lo observado, tick a tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Fidelity {
    /// Ticks predichos que también tienen frame observado.
    pub compared: u64,
    /// Ticks cuyo estado predicho difiere del observado.
    pub mismatched_ticks: Vec<u64>,
    /// Ticks predichos sin observación (hueco o fuera de la grabación): no se interpolan.
    pub unobserved: u64,
}

impl Fidelity {
    pub fn of(run: &BranchRun, rec: &Recording) -> Self {
        let (mut compared, mut unobserved, mut bad) = (0, 0, Vec::new());
        for p in &run.predicted {
            match rec.frame_at(p.tick) {
                Some(o) => {
                    compared += 1;
                    if !same_state(p, o) {
                        bad.push(p.tick);
                    }
                }
                None => unobserved += 1,
            }
        }
        Self {
            compared,
            mismatched_ticks: bad,
            unobserved,
        }
    }

    pub fn mismatches(&self) -> u64 {
        self.mismatched_ticks.len() as u64
    }

    /// Fracción de ticks comparados que coinciden (1.0 si no hubo comparación).
    pub fn ratio(&self) -> f64 {
        if self.compared == 0 {
            1.0
        } else {
            (self.compared - self.mismatches()) as f64 / self.compared as f64
        }
    }
}

/// Datos para la vista: observación vs predicciones, sin referencias vivas.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualView {
    pub map: GridMap,
    pub tick: u64,
    /// Frame observado en el checkpoint.
    pub checkpoint: Frame,
    pub original: BranchRun,
    pub alternative: BranchRun,
    /// Frames observados (grabados) desde el checkpoint, hasta el horizonte.
    pub observed: Vec<Frame>,
    /// Fidelidad del modelo con la acción original (predicción vs observación).
    pub model_fidelity: Fidelity,
    /// Divergencia de la rama alternativa respecto a lo observado.
    pub alternative_divergence: Fidelity,
}

/// Depurador contrafactual: compara acción original y alternativa desde un checkpoint.
pub struct CounterfactualDebugger;

impl CounterfactualDebugger {
    pub fn compare(
        rec: &Recording,
        tick: u64,
        alternative: Action,
        horizon: u64,
    ) -> Result<CounterfactualView, CheckpointError> {
        let cp = Checkpoint::capture(rec, tick)?;
        let original = cp.run(None, horizon);
        let alt = cp.run(Some(alternative), horizon);
        let observed = rec
            .frames()
            .iter()
            .filter(|f| f.tick > tick && f.tick <= tick + horizon)
            .cloned()
            .collect();
        Ok(CounterfactualView {
            map: rec.map().clone(),
            tick,
            checkpoint: rec.frame_at(tick).cloned().expect("verified at capture"),
            model_fidelity: Fidelity::of(&original, rec),
            alternative_divergence: Fidelity::of(&alt, rec),
            original,
            alternative: alt,
            observed,
        })
    }
}

// ===== RENDER SECTION: proyector puro; nada de dinámica, guardia ni modelos desde aquí =====

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite coordinate")
}

fn cell_center(p: GridPos) -> Point2 {
    pt(p.x as f64 + 0.5, -(p.y as f64) - 0.5)
}

fn path_primitive(points: &[GridPos]) -> Option<ScenePrimitive> {
    match points {
        [] => None,
        [p] => ScenePrimitive::circle(cell_center(*p), 0.12).ok(),
        many => {
            ScenePrimitive::polyline(many.iter().map(|p| cell_center(*p)).collect(), false).ok()
        }
    }
}

/// Líneas de texto de la vista (observado vs predicho, fidelidad, rechazo).
pub fn counterfactual_lines(view: &CounterfactualView) -> Vec<String> {
    let h = &view.alternative.header;
    let mut lines = vec![format!(
        "branch {:016x} parent={:016x} tick={} seed={} model={}/v{}",
        h.id, h.parent.0, h.tick, h.seed, h.model.name, h.model.snapshot_schema
    )];
    let end = |frames: &[Frame]| {
        frames.last().map_or("none".to_string(), |f| {
            format!("({},{}) reward={:.2}", f.agent.x, f.agent.y, f.total_reward)
        })
    };
    lines.push(format!("OBSERVED (recording): end {}", end(&view.observed)));
    let mf = &view.model_fidelity;
    lines.push(format!(
        "PREDICTED original: end {} | model fidelity {:.0}% ({}/{} ticks match, mismatches {:?})",
        end(&view.original.predicted),
        mf.ratio() * 100.0,
        mf.compared - mf.mismatches(),
        mf.compared,
        mf.mismatched_ticks
    ));
    let ad = &view.alternative_divergence;
    lines.push(format!(
        "PREDICTED alternative: end {} | diverges from observed on {}/{} ticks (first {:?})",
        end(&view.alternative.predicted),
        ad.mismatches(),
        ad.compared,
        ad.mismatched_ticks.first()
    ));
    if let Some(r) = &view.alternative.rejection {
        lines.push(format!(
            "alternative REJECTED by guard at tick {}: {} (not executed)",
            r.tick, r.reason
        ));
    }
    lines
}

/// Proyección pura. Capas: `Environment` (celdas + ruta observada, `Observed`),
/// `Imagination` (rutas predichas, `Hypothetical`), `Decisions` (checkpoint), `Labels`.
pub fn project_counterfactual_scene(view: &CounterfactualView) -> Scene2D {
    let mut scene = Scene2D::new(
        SceneId::new(99),
        Tick::new(view.tick),
        EpistemicKind::Hypothetical,
        CoordinateSpace::meters(),
    );
    for layer in [
        LayerId::Environment,
        LayerId::Imagination,
        LayerId::Decisions,
        LayerId::Labels,
    ] {
        scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }
    let mut next_id = 1u64;
    let mut id = || {
        let n = next_id;
        next_id += 1;
        SceneNodeId::new(n)
    };
    let (w, h) = (view.map.width(), view.map.height());
    for y in 0..h {
        for x in 0..w {
            let kind = view.map.cell(GridPos::new(x, y)).unwrap_or(CellKind::Empty);
            let rect =
                ScenePrimitive::rect(pt(x as f64, -(y as f64) - 1.0), 1.0, 1.0).expect("unit rect");
            let node = SceneNode::new(id(), None, EpistemicKind::Observed, rect, false)
                .with_state_code(kind.as_code() as u32);
            scene.add_node(LayerId::Environment, node).expect("node");
        }
    }
    let origin = Some(view.checkpoint.agent);
    let path = |frames: &[Frame]| -> Vec<GridPos> { frames.iter().map(|f| f.agent).collect() };
    let observed = path(&view.observed);
    let paths = [
        (LayerId::Environment, EpistemicKind::Observed, observed),
        (
            LayerId::Imagination,
            EpistemicKind::Hypothetical,
            path(&view.original.predicted),
        ),
        (
            LayerId::Imagination,
            EpistemicKind::Hypothetical,
            path(&view.alternative.predicted),
        ),
    ];
    for (layer, kind, points) in paths {
        if let Some(p) = path_primitive(&points) {
            scene
                .add_node(layer, SceneNode::new(id(), None, kind, p, false))
                .expect("node");
        }
    }
    if let Some(p) = origin {
        let c = ScenePrimitive::circle(cell_center(p), 0.3).expect("marker");
        scene
            .add_node(
                LayerId::Decisions,
                SceneNode::new(id(), None, EpistemicKind::Observed, c, true),
            )
            .expect("node");
    }
    for (i, line) in counterfactual_lines(view).into_iter().enumerate() {
        let lab = ScenePrimitive::label(pt(0.0, -(h as f64) - 1.0 - i as f64 * 0.6), line);
        scene
            .add_node(
                LayerId::Labels,
                SceneNode::new(id(), None, EpistemicKind::Observed, lab, false),
            )
            .expect("node");
    }
    scene
}

/// SVG determinista de la vista contrafactual.
pub fn render_counterfactual_svg(view: &CounterfactualView, width: f64, height: f64) -> String {
    let scene = project_counterfactual_scene(view);
    let mut cam = Camera2D::new(width, height, true);
    if let Some(bounds) = scene_bounds(&scene) {
        cam.fit_to_window(width, height, &bounds, 8.0);
    }
    SoftwareSvgBackend.render_svg(&scene, &cam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid_lab::GridRollout;
    use crate::recorder::record_grid;
    use gogamen_core::{DeterminismMetadata, TraceHasher};

    const SEED: u64 = 42;
    const TICKS: u64 = 12;

    fn rec() -> Recording {
        record_grid(SEED, GridMap::cul_de_sac_map(), TICKS).expect("record")
    }

    fn act(i: usize) -> Action {
        Action::Discrete(DiscreteAction::new(i, GRID_ACTION_CARDINALITY).unwrap())
    }

    /// Una acción alternativa distinta de la grabada en el tick siguiente al checkpoint.
    fn differing_alt(cp: &Checkpoint<'_>) -> Action {
        let base = cp.run(None, 6);
        (0..GRID_ACTION_CARDINALITY)
            .map(act)
            .find(|a| cp.run(Some(a.clone()), 6).predicted != base.predicted)
            .expect("some action changes the outcome")
    }

    #[test]
    fn ac1_original_reproduces_recording_alternative_differs_and_is_repeatable() {
        let rec = rec();
        let cp = Checkpoint::capture(&rec, 3).expect("checkpoint");
        let h = rec.last_tick() - 3; // horizonte = lo grabado tras el checkpoint
        let a = cp.run(None, h);
        // Outcome A == traza grabada tras el checkpoint, exactamente.
        let recorded: Vec<Frame> = rec
            .frames()
            .iter()
            .filter(|f| f.tick > 3)
            .cloned()
            .collect();
        assert!(!recorded.is_empty());
        assert_eq!(a.predicted.len(), recorded.len());
        for (p, o) in a.predicted.iter().zip(&recorded) {
            assert!(same_state(p, o), "tick {}", p.tick);
        }
        assert_eq!(a.trace_hash(), frame_digest(&recorded));
        assert_eq!(Fidelity::of(&a, &rec).mismatches(), 0);
        // Outcome B difiere.
        let alt = differing_alt(&cp);
        let b = cp.run(Some(alt.clone()), h);
        assert_ne!(b.trace_hash(), a.trace_hash());
        assert_ne!(b.predicted, a.predicted);
        // Repetible y sin influencia mutua (intercalado A B B A).
        assert_eq!(cp.run(Some(alt.clone()), h), b);
        assert_eq!(cp.run(None, h), a);
        assert_eq!(cp.run(Some(alt), h).trace_hash(), b.trace_hash());
        // Dos checkpoints independientes del mismo tick coinciden.
        let cp2 = Checkpoint::capture(&rec, 3).unwrap();
        assert_eq!(cp2.run(None, h), a);
    }

    #[test]
    fn ac2_branch_header_and_parent_untouched() {
        let rec = rec();
        let (bytes, hash) = (rec.to_bytes(), RecordingId::of(&rec));
        let cp = Checkpoint::capture(&rec, 4).unwrap();
        let alt = differing_alt(&cp);
        let b = cp.run(Some(alt.clone()), TICKS);
        let h = &b.header;
        assert_eq!((h.parent, h.tick, h.seed), (hash, 4, SEED));
        assert_eq!(h.model.name, MODEL_NAME);
        assert_eq!(h.model.snapshot_schema, SNAPSHOT_SCHEMA_VERSION);
        assert_eq!(h.kind, BranchKind::Alternative(alt));
        let a = cp.run(None, TICKS);
        assert_ne!(a.header.id, b.header.id);
        assert_eq!(a.header.kind, BranchKind::Original);
        // El padre no cambia (bytes y hash) tras ramificar y ejecutar.
        assert_eq!(rec.to_bytes(), bytes);
        assert_eq!(RecordingId::of(&rec), hash);
        assert_eq!(Recording::open(&bytes).unwrap(), rec);
    }

    #[test]
    fn ac2_live_environment_untouched_by_branching() {
        let rec = rec();
        let config = GridWorldConfig {
            map: rec.map().clone(),
            ..GridWorldConfig::default()
        };
        let mut live = GridRollout::new(config.clone(), MasterSeed::new(SEED)).unwrap();
        for _ in 0..4 {
            live.step().unwrap();
        }
        let (view, hash, n) = (live.view(), live.trace_hash(), live.steps());
        let dbg = CounterfactualDebugger::compare(&rec, 4, act(1), TICKS).unwrap();
        assert_eq!(live.view(), view);
        assert_eq!((live.trace_hash(), live.steps()), (hash, n));
        // El vivo sigue exactamente la grabación y no ve la rama alternativa.
        let next = live.step().unwrap().unwrap();
        let f = rec.frame_at(5).unwrap();
        assert_eq!(
            (next.position, next.total_reward),
            (f.agent, f.total_reward)
        );
        // sanity: el hash independiente no depende de la depuración
        let mut h =
            TraceHasher::with_metadata(&DeterminismMetadata::current(MasterSeed::new(SEED)));
        h.write_u64(1);
        let _ = (h.finish(), dbg);
    }

    #[test]
    fn ac2_guard_rejected_alternative_is_recorded_not_executed() {
        let rec = rec();
        let cp = Checkpoint::capture(&rec, 2).unwrap();
        // Índice 7 de un espacio de 9: fuera del ActionSpace de 5 acciones del Grid.
        let bad = Action::Discrete(DiscreteAction::new(7, 9).unwrap());
        let r = cp.run(Some(bad.clone()), TICKS);
        assert!(r.predicted.is_empty(), "rejected action must not execute");
        let rej = r.rejection.expect("rejection recorded");
        assert_eq!((rej.tick, rej.action), (2, bad));
        assert!(matches!(rej.reason, GuardRejectionReason::InvalidAction(_)));
        // La vista lo muestra.
        let view = CounterfactualDebugger::compare(
            &rec,
            2,
            Action::Discrete(DiscreteAction::new(7, 9).unwrap()),
            6,
        )
        .unwrap();
        assert!(
            counterfactual_lines(&view)
                .iter()
                .any(|l| l.contains("REJECTED by guard") && l.contains("not executed"))
        );
    }

    #[test]
    fn ac3_typed_errors_for_insufficient_checkpoint() {
        let rec = rec();
        let last = rec.last_tick();
        assert!(matches!(
            Checkpoint::capture(&rec, last + 5),
            Err(CheckpointError::TickOutOfRange { .. })
        ));
        // Hueco en el tick y hueco previo.
        let frames: Vec<Frame> = rec
            .frames()
            .iter()
            .filter(|f| f.tick != 3)
            .cloned()
            .collect();
        let holey =
            Recording::new(SEED, crate::replay::map_to_rows(rec.map()), last, frames).unwrap();
        assert!(
            matches!(Checkpoint::capture(&holey, 3), Err(CheckpointError::MissingFrame(g)) if g == Gap { from: 3, to: 3 })
        );
        assert!(matches!(
            Checkpoint::capture(&holey, 5),
            Err(CheckpointError::IncompleteHistory(_))
        ));
        assert!(Checkpoint::capture(&holey, 2).is_ok());
        // Cola truncada: el tick declarado más allá del último frame cae en hueco.
        let tail = Recording::new(
            SEED,
            crate::replay::map_to_rows(rec.map()),
            last + 3,
            rec.frames().to_vec(),
        )
        .unwrap();
        assert!(matches!(
            Checkpoint::capture(&tail, last + 2),
            Err(CheckpointError::MissingFrame(_))
        ));
        // Estado grabado que el modelo no reproduce (semilla alterada).
        let wrong = Recording::new(
            SEED + 1,
            crate::replay::map_to_rows(rec.map()),
            last,
            rec.frames().to_vec(),
        )
        .unwrap();
        let err = (1..=last).find_map(|t| Checkpoint::capture(&wrong, t).err());
        assert!(
            matches!(err, Some(CheckpointError::ReconstructionMismatch { .. })),
            "{err:?}"
        );
        // Un checkpoint rechazado no produce rama: compare propaga el error.
        assert!(CounterfactualDebugger::compare(&holey, 5, act(1), 4).is_err());
    }

    #[test]
    fn ac3_view_separates_prediction_from_observation_with_fidelity() {
        let rec = rec();
        let cp = Checkpoint::capture(&rec, 3).unwrap();
        let alt = differing_alt(&cp);
        let view = CounterfactualDebugger::compare(&rec, 3, alt, 6).unwrap();
        assert_eq!(view.model_fidelity.compared, view.observed.len() as u64);
        assert_eq!(view.model_fidelity.mismatches(), 0);
        assert!((view.model_fidelity.ratio() - 1.0).abs() < f64::EPSILON);
        assert!(view.alternative_divergence.mismatches() > 0);
        assert!(view.alternative_divergence.ratio() < 1.0);
        let lines = counterfactual_lines(&view);
        let all = lines.join("\n");
        for needle in [
            "OBSERVED (recording)",
            "PREDICTED original",
            "PREDICTED alternative",
            "model fidelity 100%",
            "parent=",
            "seed=42",
            "GridWorldModel",
        ] {
            assert!(all.contains(needle), "missing {needle}: {all}");
        }
        // La escena marca observado vs hipotético por nodo.
        let scene = project_counterfactual_scene(&view);
        let kinds: Vec<_> = scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes().iter().map(move |n| (l.id(), n.epistemic_kind())))
            .collect();
        assert!(
            kinds
                .iter()
                .any(|(l, k)| *l == LayerId::Imagination && *k == EpistemicKind::Hypothetical)
        );
        assert!(
            kinds
                .iter()
                .any(|(l, k)| *l == LayerId::Environment && *k == EpistemicKind::Observed)
        );
        let svg = render_counterfactual_svg(&view, 640.0, 480.0);
        assert_eq!(svg, render_counterfactual_svg(&view, 640.0, 480.0));
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn render_section_has_no_dynamics_planner_or_model() {
        let src = include_str!("counterfactual.rs");
        let start = src.find("// ===== RENDER SECTION").unwrap();
        let end = src.find("#[cfg(test)]").unwrap();
        let render: String = src[start..end]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "GridWorldModel",
            "GridWorldEnv",
            "GridRollout",
            "BranchableSnapshot",
            "ActionGuard",
            "step_in_place",
            "restore(",
            "fork(",
            "snapshot(",
            "Planner",
            "Actuator",
            "ControlPort",
            "SessionHost",
            "std::net",
            "std::fs",
            "gogamen_lab_",
            "RngSource",
            "next_u64",
            "Checkpoint",
        ] {
            assert!(
                !render.contains(banned),
                "render section references {banned}"
            );
        }
    }
}
