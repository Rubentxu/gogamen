//! Language World visual e interacción humana (GOG-97): mapa parcial de creencias vs verdad del evaluador.
//!
//! # Estructura
//! - **Vistas + proyectores (puros):** [`AgentView`] (sólo creencias y trazas; *sin* verdad de
//!   terreno) y [`EvaluatorView`] (la anterior + mapa real). Dos proyectores explícitos,
//!   [`project_agent_scene`] y [`project_evaluator_scene`], y el selector [`ViewMode`]. El código de
//!   esta mitad del fichero (hasta el marcador `ENGINE`) no contiene dinámica, planificador ni
//!   actualización de creencias (test textual de guarda).
//! - **Motor headless:** [`LanguageWorldRollout`] orquesta piezas *reales*: [`GroundTruthGrid`]
//!   (sensor con oclusión), [`GridBeliefMap`] (actualización bayesiana de core, mapa plano +
//!   ledger central), [`BeliefGridPlanner`] (sólo ve [`PlannerInput`]: creencias + posición propia),
//!   `language_world::ground_human_notice` (aviso humano -> grounding offline -> `HumanReport`) y las
//!   acciones epistémicas de core (`EpistemicController`, `AskSpec`, `LocalCommunicationBus`).
//!   Studio no reimplementa ninguna actualización de creencias.
//! - **Ask offline y no bloqueante:** [`OfflineAskProvider`] devuelve respuestas enlatadas con
//!   latencia *simulada* (reloj virtual, sin `sleep`). Permiso denegado, proveedor lento
//!   (`TimedOut`) o no disponible (`NoResponse`) quedan registrados en [`AskRecord`] y el ciclo
//!   observar-planificar-actuar continúa en el siguiente tick.
//!
//! `gogamen-core` y `gogamen-cli` no dependen de este módulo.

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use gogamen_core::{
    Action, AdvanceSpec, AskDispatch, AskPollOutcome, AskSpec, BeliefUpdateStatus, BinaryBelief,
    BinaryCommOutcomeMapper, CommCost, CommCostWeights, CommOutcomeMapper, CommPayload,
    CommunicationPort, ConversationScope, DecisionOption, DecisionPayoffModel, DiscreteAction,
    EpistemicActionKind, EpistemicController, EpistemicKind, EvidenceRef, FrameId, InterlocutorId,
    InterlocutorKind, LikelihoodModel, LocalCommunicationBus, MasterSeed, MessageId,
    MonotonicClock, MonotonicInstant, NoResponseReason, OptionId, Permission, PermissionSet,
    SensorId, SituatedMessage, UnitTag, VirtualMonotonicClock,
};
use gogamen_lab_grid::language_world::ground_human_notice;
use gogamen_lab_grid::partial_obs::{
    BeliefGridPlanner, CellHypothesis, DEFAULT_HUMAN_SENSOR_ID, GridBeliefConfig, GridBeliefMap,
    GroundTruthGrid, HumanReport, HypothesisKind,
};
use gogamen_lab_grid::world::GridAction;
use gogamen_lab_grid::{CellKind, GridMap, GridPos, GridSize};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};

use crate::camera::Camera2D;
use crate::cognitive::EvidenceSource;
use crate::viewport::SoftwareSvgBackend;

/// Permiso nominal necesario para que el agente pueda preguntar a un humano (`Ask`).
pub const ASK_PERMISSION: &str = "human.ask";

/// Selector explícito de vista: agente (sólo creencias) o evaluador (con verdad de terreno).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode {
    Agent,
    Evaluator,
}

/// Una celda del mapa parcial de creencias tal como la ve el agente.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefCellView {
    pub pos: GridPos,
    /// `P(Obstacle)` actual.
    pub p_obstacle: f64,
    pub human_updates: u64,
    pub direct_updates: u64,
}

/// Procedencia resumida de una celda.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellEvidence {
    /// Sólo prior.
    Prior,
    /// Sólo avisos humanos (sin corroborar por medición directa): se marca `human?`.
    HumanOnly,
    /// Hay medición directa.
    Direct,
}

impl BeliefCellView {
    pub fn evidence(&self) -> CellEvidence {
        if self.direct_updates > 0 {
            CellEvidence::Direct
        } else if self.human_updates > 0 {
            CellEvidence::HumanOnly
        } else {
            CellEvidence::Prior
        }
    }
}

/// Traza de una actualización de creencia (antes/después por celda afectada).
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateTrace {
    pub tick: u64,
    pub pos: GridPos,
    pub source: EvidenceSource,
    /// Canal: `human-notice`, `human-ask` o `direct-sensor`.
    pub channel: &'static str,
    pub before: f64,
    pub after: f64,
    pub status: BeliefUpdateStatus,
    /// Confianza declarada en la observación concreta.
    pub reported_confidence: f64,
    /// Fiabilidad calibrada de la fuente (modelo de verosimilitud de core).
    pub source_sensitivity: f64,
    pub source_specificity: f64,
    pub evidence: EvidenceRef,
}

/// Resultado registrado de un `Ask`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AskOutcome {
    /// El valorador epistémico prefirió otra acción.
    NotSelected(EpistemicActionKind),
    PermissionDenied,
    Pending,
    /// Respuesta aplicada a creencias (`true` = "bloqueada").
    Answered {
        obstacle: bool,
    },
    TimedOut,
    NoResponse(NoResponseReason),
}

/// Registro de una acción epistémica `Ask` (siempre visible, también si no se ejecutó).
#[derive(Debug, Clone, PartialEq)]
pub struct AskRecord {
    pub tick: u64,
    pub pos: GridPos,
    pub selected: EpistemicActionKind,
    pub evoi: f64,
    pub net_evoi: f64,
    pub outcome: AskOutcome,
}

/// Vista del agente: **sólo** creencias, trazas y acciones epistémicas. No hay mapa real.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentView {
    pub tick: u64,
    pub size: GridSize,
    pub agent: GridPos,
    pub cells: Vec<BeliefCellView>,
    pub traces: Vec<UpdateTrace>,
    pub asks: Vec<AskRecord>,
    pub last_action: Option<GridAction>,
}

/// Vista del evaluador: la del agente más el mapa real (nunca llega al planificador).
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatorView {
    pub agent: AgentView,
    pub truth: GridMap,
}

const MAX_TRACE_ROWS: usize = 12;

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite coordinate")
}

struct Sink {
    scene: Scene2D,
    next: u64,
}

impl Sink {
    fn new(id: u64, tick: u64) -> Self {
        let mut scene = Scene2D::new(
            SceneId::new(id),
            gogamen_core::Tick::new(tick),
            EpistemicKind::Observed,
            CoordinateSpace::meters(),
        );
        for layer in [LayerId::Environment, LayerId::Labels, LayerId::Decisions] {
            scene
                .add_layer(SceneLayer::new(layer))
                .expect("unique layer");
        }
        Self { scene, next: 1 }
    }

    fn add(
        &mut self,
        layer: LayerId,
        prim: ScenePrimitive,
        kind: EpistemicKind,
        state: Option<u32>,
    ) {
        let mut node = SceneNode::new(SceneNodeId::new(self.next), None, kind, prim, true);
        if let Some(code) = state {
            node = node.with_state_code(code);
        }
        self.next += 1;
        let _ = self.scene.add_node(layer, node);
    }

    fn label(&mut self, x: f64, y: f64, text: impl Into<String>, kind: EpistemicKind) {
        self.add(
            LayerId::Labels,
            ScenePrimitive::label(pt(x, y), text),
            kind,
            None,
        );
    }
}

fn source_label(s: EvidenceSource) -> &'static str {
    match s {
        EvidenceSource::Human => "human",
        EvidenceSource::DirectSensor => "direct-sensor",
        EvidenceSource::Agent => "agent",
        EvidenceSource::Unknown => "unknown",
    }
}

fn outcome_text(o: AskOutcome) -> String {
    match o {
        AskOutcome::NotSelected(k) => format!("not-asked(selected={k:?})"),
        AskOutcome::PermissionDenied => "permission-denied(non-blocking)".to_owned(),
        AskOutcome::Pending => "pending".to_owned(),
        AskOutcome::Answered { obstacle } => format!("answered(blocked={obstacle})"),
        AskOutcome::TimedOut => "timed-out(non-blocking)".to_owned(),
        AskOutcome::NoResponse(r) => format!("no-response({r:?})(non-blocking)"),
    }
}

/// Filas de texto compartidas: trazas de actualización y acciones epistémicas.
fn panel_rows(view: &AgentView) -> Vec<String> {
    let mut rows = vec![format!("== Belief updates ({}) ==", view.traces.len())];
    let skip = view.traces.len().saturating_sub(MAX_TRACE_ROWS);
    for t in view.traces.iter().skip(skip) {
        let mut note = String::new();
        if t.source == EvidenceSource::Human {
            let corrected = view.traces.iter().find(|d| {
                d.pos == t.pos && d.source == EvidenceSource::DirectSensor && d.tick >= t.tick
            });
            note = match corrected {
                Some(d) => format!(
                    " later-corrected-by-direct(E{}, {:.3})",
                    d.evidence.as_u64(),
                    d.after
                ),
                None => " FLAGGED uncorroborated(source=human)".to_owned(),
            };
        }
        rows.push(format!(
            "t={} ({},{}) src={} via={} P(obstacle) {:.3} -> {:.3} conf={:.2} reliability(sens={:.2},spec={:.2}) ev=E{} [{:?}]{note}",
            t.tick,
            t.pos.x,
            t.pos.y,
            source_label(t.source),
            t.channel,
            t.before,
            t.after,
            t.reported_confidence,
            t.source_sensitivity,
            t.source_specificity,
            t.evidence.as_u64(),
            t.status
        ));
    }
    rows.push(format!("== Epistemic actions ({}) ==", view.asks.len()));
    for a in &view.asks {
        rows.push(format!(
            "t={} ASK ({},{}) selected={:?} evoi={:.3} net={:.3} outcome={}",
            a.tick,
            a.pos.x,
            a.pos.y,
            a.selected,
            a.evoi,
            a.net_evoi,
            outcome_text(a.outcome)
        ));
    }
    rows
}

fn belief_cells(sink: &mut Sink, view: &AgentView, with_truth: Option<&GridMap>) {
    for c in &view.cells {
        let (fx, fy) = (c.pos.x as f64, c.pos.y as f64);
        let state = match with_truth {
            Some(map) => map.cell(c.pos).map_or(0, |k| k.as_code() as u32),
            None => (c.p_obstacle * 100.0).round() as u32,
        };
        sink.add(
            LayerId::Environment,
            ScenePrimitive::rect(pt(fx, fy), 1.0, 1.0).expect("unit rect"),
            if with_truth.is_some() {
                EpistemicKind::Observed
            } else {
                EpistemicKind::Inferred
            },
            Some(state),
        );
        sink.label(
            fx + 0.08,
            fy + 0.45,
            format!("{:.2}", c.p_obstacle),
            EpistemicKind::Inferred,
        );
        let tag = match c.evidence() {
            CellEvidence::Prior => "prior",
            CellEvidence::HumanOnly => "human?",
            CellEvidence::Direct => "seen",
        };
        sink.label(fx + 0.08, fy + 0.85, tag, EpistemicKind::Inferred);
        if let Some(map) = with_truth {
            let glyph = match map.cell(c.pos) {
                Some(CellKind::Wall) => Some("#"),
                Some(CellKind::Goal) => Some("G"),
                Some(CellKind::Hazard) => Some("H"),
                _ => None,
            };
            if let Some(g) = glyph {
                sink.label(fx + 0.7, fy + 0.2, g, EpistemicKind::Observed);
            }
        }
    }
    let a = view.agent;
    sink.add(
        LayerId::Decisions,
        ScenePrimitive::circle(pt(a.x as f64 + 0.5, a.y as f64 + 0.5), 0.2).expect("agent circle"),
        EpistemicKind::Observed,
        None,
    );
}

fn panel(sink: &mut Sink, view: &AgentView, extra: Vec<String>) {
    let mut y = view.size.height as f64 + 0.5;
    for r in panel_rows(view).into_iter().chain(extra) {
        sink.label(0.0, y, r, EpistemicKind::Observed);
        y += 0.3;
    }
}

/// Proyector de la vista del **agente** (sólo creencias; coordenadas con `y` hacia abajo).
pub fn project_agent_scene(view: &AgentView) -> Scene2D {
    let mut sink = Sink::new(97, view.tick);
    sink.label(
        0.0,
        -0.3,
        format!(
            "Language World [agent view: beliefs only] tick={} agent=({},{})",
            view.tick, view.agent.x, view.agent.y
        ),
        EpistemicKind::Observed,
    );
    belief_cells(&mut sink, view, None);
    panel(&mut sink, view, Vec::new());
    sink.scene
}

/// Proyector de la vista del **evaluador**: verdad de terreno + creencias + discrepancias.
pub fn project_evaluator_scene(view: &EvaluatorView) -> Scene2D {
    let mut sink = Sink::new(98, view.agent.tick);
    sink.label(
        0.0,
        -0.3,
        format!(
            "Language World [EVALUATOR view: GROUND TRUTH, not available to the agent] tick={}",
            view.agent.tick
        ),
        EpistemicKind::Observed,
    );
    belief_cells(&mut sink, &view.agent, Some(&view.truth));
    let mut extra = vec!["== Belief vs ground truth ==".to_owned()];
    for c in &view.agent.cells {
        let blocked = view.truth.cell(c.pos) == Some(CellKind::Wall);
        if (c.p_obstacle >= 0.5) != blocked {
            extra.push(format!(
                "MISMATCH ({},{}) P(obstacle)={:.3} truth_blocked={blocked}",
                c.pos.x, c.pos.y, c.p_obstacle
            ));
        }
    }
    panel(&mut sink, &view.agent, extra);
    sink.scene
}

/// Renderiza una escena a SVG determinista (`y` hacia abajo).
pub fn render_svg(scene: &Scene2D, width: f64, height: f64) -> String {
    let mut cam = Camera2D::new(width, height, false);
    if let Some(bounds) = crate::camera::scene_bounds(scene) {
        cam.fit_to_window(width, height, &bounds, 8.0);
    }
    SoftwareSvgBackend.render_svg(scene, &cam)
}

// ======================================================================================
// ENGINE: a partir de aquí hay motor headless (dinámica, planificador, creencias, Ask).
// El test de guarda exige que NADA de esto aparezca antes del marcador.
// ======================================================================================

/// Entrada del planificador: **sólo** creencias y posición propia. Sin verdad de terreno.
pub struct PlannerInput<'a> {
    pub beliefs: &'a GridBeliefMap,
    pub agent_pos: GridPos,
}

fn plan(planner: &BeliefGridPlanner, input: &PlannerInput<'_>) -> Result<GridAction, LabError> {
    planner
        .choose_action(input.beliefs, input.agent_pos)
        .map_err(LabError::of)
}

/// Error del lab (envuelve errores de core/lab en texto; sin pánicos).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabError(pub String);

impl LabError {
    fn of(e: impl fmt::Display) -> Self {
        Self(e.to_string())
    }
}

impl fmt::Display for LabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "language world lab: {}", self.0)
    }
}

impl std::error::Error for LabError {}

/// Comportamiento enlatado del proveedor offline ante un `Ask`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CannedBehavior {
    /// Respuesta fija tras `latency` simulada (`obstacle = true` -> "yes, blocked").
    Answer {
        obstacle: bool,
        confidence: f64,
        latency: Duration,
    },
    /// Interlocutor no disponible: responde `NoResponse(Unavailable)`.
    Unavailable,
}

/// Proveedor offline determinista (sin red, sin hilos, sin reloj real).
#[derive(Debug, Clone, PartialEq)]
pub struct OfflineAskProvider {
    per_cell: BTreeMap<(usize, usize), CannedBehavior>,
    default: CannedBehavior,
    calls: u64,
}

impl OfflineAskProvider {
    pub fn new(default: CannedBehavior) -> Self {
        Self {
            per_cell: BTreeMap::new(),
            default,
            calls: 0,
        }
    }

    pub fn with_cell(mut self, pos: GridPos, behavior: CannedBehavior) -> Self {
        self.per_cell.insert((pos.x, pos.y), behavior);
        self
    }

    /// Número de preguntas atendidas por el proveedor.
    pub const fn calls(&self) -> u64 {
        self.calls
    }

    fn behavior(&mut self, pos: GridPos) -> CannedBehavior {
        self.calls += 1;
        self.per_cell
            .get(&(pos.x, pos.y))
            .copied()
            .unwrap_or(self.default)
    }
}

/// Configuración del escenario headless.
#[derive(Debug, Clone, PartialEq)]
pub struct LanguageWorldLabConfig {
    /// Mapa real (sólo lo ven el sensor con oclusión y el evaluador).
    pub truth: GridMap,
    pub vision_radius: usize,
    pub prior_obstacle: f64,
    pub human_sensitivity: f64,
    pub human_specificity: f64,
    /// Tiempo simulado por tick.
    pub dt: Duration,
    /// Plazo de un `Ask` (tiempo simulado).
    pub ask_timeout: Duration,
}

impl LanguageWorldLabConfig {
    pub fn new(truth: GridMap) -> Self {
        Self {
            truth,
            vision_radius: 1,
            prior_obstacle: 0.4,
            human_sensitivity: 0.85,
            human_specificity: 0.85,
            dt: Duration::from_millis(100),
            ask_timeout: Duration::from_secs(1),
        }
    }
}

struct PendingAsk {
    record: usize,
    pos: GridPos,
    controller: EpistemicController,
    dispatch: AskDispatch,
    scheduled: Option<(MonotonicInstant, SituatedMessage)>,
    delivered: Option<SituatedMessage>,
}

/// Resumen de un tick del ciclo observar-planificar-actuar.
#[derive(Debug, Clone, PartialEq)]
pub struct TickReport {
    pub tick: u64,
    pub action: GridAction,
    pub agent: GridPos,
    pub direct_updates: usize,
    pub ask_outcomes: Vec<(usize, AskOutcome)>,
}

/// Motor headless: sensor real con oclusión + creencias reales + planificador sobre creencias.
pub struct LanguageWorldRollout {
    truth: GroundTruthGrid,
    truth_map: GridMap,
    beliefs: GridBeliefMap,
    planner: BeliefGridPlanner,
    clock: VirtualMonotonicClock,
    bus: LocalCommunicationBus,
    provider: OfflineAskProvider,
    permissions: PermissionSet,
    config: LanguageWorldLabConfig,
    pending: Vec<PendingAsk>,
    asks: Vec<AskRecord>,
    traces: Vec<UpdateTrace>,
    tick: u64,
    next_id: u64,
    last_action: Option<GridAction>,
    agent_id: InterlocutorId,
    human_id: InterlocutorId,
    _seed: MasterSeed,
}

fn reliability(model: &LikelihoodModel) -> (f64, f64) {
    (
        model.sensitivity().unwrap_or(f64::NAN),
        model.likelihood(0, 0).unwrap_or(f64::NAN),
    )
}

impl LanguageWorldRollout {
    pub fn new(
        config: LanguageWorldLabConfig,
        provider: OfflineAskProvider,
        permissions: PermissionSet,
    ) -> Result<Self, LabError> {
        let bcfg = GridBeliefConfig::new(
            config.prior_obstacle,
            0.01,
            0.01,
            config.human_sensitivity,
            config.human_specificity,
            0.99,
            0.99,
        )
        .map_err(LabError::of)?;
        let beliefs = GridBeliefMap::new(config.truth.size(), bcfg).map_err(LabError::of)?;
        let mut bus = LocalCommunicationBus::new(16, 16).map_err(LabError::of)?;
        let agent_id = InterlocutorId::new("gogamen-agent").map_err(LabError::of)?;
        let human_id = InterlocutorId::new("human-operator").map_err(LabError::of)?;
        bus.register_interlocutor(agent_id.clone(), InterlocutorKind::Agent);
        bus.register_interlocutor(human_id.clone(), InterlocutorKind::Human);
        Ok(Self {
            truth: GroundTruthGrid::new(config.truth.clone(), config.vision_radius),
            truth_map: config.truth.clone(),
            beliefs,
            planner: BeliefGridPlanner::default(),
            clock: VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000)),
            bus,
            provider,
            permissions,
            config,
            pending: Vec::new(),
            asks: Vec::new(),
            traces: Vec::new(),
            tick: 0,
            next_id: 5_000,
            last_action: None,
            agent_id,
            human_id,
            _seed: MasterSeed::new(0),
        })
    }

    fn fresh_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    pub const fn tick_count(&self) -> u64 {
        self.tick
    }

    pub fn asks(&self) -> &[AskRecord] {
        &self.asks
    }

    pub fn traces(&self) -> &[UpdateTrace] {
        &self.traces
    }

    pub const fn provider(&self) -> &OfflineAskProvider {
        &self.provider
    }

    /// Creencia actual `P(Obstacle)` (lectura del mapa de creencias de core).
    pub fn p_obstacle(&self, pos: GridPos) -> Result<f64, LabError> {
        self.beliefs
            .probability(pos, HypothesisKind::Obstacle)
            .map_err(LabError::of)
    }

    /// Aplica un `HumanReport` a través de core y registra la traza antes/después.
    fn apply_human(&mut self, report: &HumanReport, channel: &'static str) -> Result<(), LabError> {
        let cell = *report.payload();
        let before = self.p_obstacle(cell.pos)?;
        let status = self
            .beliefs
            .apply_human_report(report)
            .map_err(LabError::of)?;
        let after = self.p_obstacle(cell.pos)?;
        let (sens, spec) = reliability(&self.beliefs.config().human_model);
        self.traces.push(UpdateTrace {
            tick: self.tick,
            pos: cell.pos,
            source: EvidenceSource::Human,
            channel,
            before,
            after,
            status,
            reported_confidence: report.confidence().as_f64(),
            source_sensitivity: sens,
            source_specificity: spec,
            evidence: report.evidence_ref(),
        });
        Ok(())
    }

    /// Canal humano semántico: el texto se aterriza offline (`ground_human_notice`) y el informe
    /// resultante se aplica con el camino real `GridBeliefMap::apply_human_report`.
    pub fn submit_human_notice(&mut self, text: &str) -> Result<(), LabError> {
        let id = self.fresh_id();
        let now = self.clock.now();
        let (_outcome, report) = ground_human_notice(text, id, id, now).map_err(LabError::of)?;
        self.apply_human(&report, "human-notice")
    }

    /// Acción epistémica `Ask` sobre `pos`: valora (EVOI), comprueba permiso y, si procede, envía
    /// por el puerto de comunicación sin bloquear. Devuelve el índice del [`AskRecord`].
    pub fn ask_about(&mut self, pos: GridPos) -> Result<usize, LabError> {
        let p = self.p_obstacle(pos)?;
        let option = |s: &str| OptionId::new(s).map_err(LabError::of);
        let (yes, no) = (option("yes")?, option("no")?);
        // Avanzar a la celda: +10 si libre, -12 si bloqueada; abstenerse: 0.
        let payoff = DecisionPayoffModel::new(vec![
            DecisionOption::new(option("advance")?, 10.0, -12.0)
                .map_err(LabError::of)?
                .with_action(Action::Discrete(
                    DiscreteAction::new(1, 2).map_err(LabError::of)?,
                )),
            DecisionOption::new(option("abstain")?, 0.0, 0.0)
                .map_err(LabError::of)?
                .with_action(Action::Discrete(
                    DiscreteAction::new(0, 2).map_err(LabError::of)?,
                )),
        ])
        .map_err(LabError::of)?;
        let controller = EpistemicController::new(
            self.agent_id.clone(),
            BinaryBelief::new(p).map_err(LabError::of)?,
            payoff,
            CommCostWeights::from_unit_weight(0.5).map_err(LabError::of)?,
            FrameId::world(),
            UnitTag::dimensionless(),
        );
        let model =
            LikelihoodModel::from_sensitivity_specificity(0.9, 0.9).map_err(LabError::of)?;
        let scope = ConversationScope::session("language-world").map_err(LabError::of)?;
        let spec = AskSpec::new(
            self.human_id.clone(),
            scope.clone(),
            format!("Is cell ({},{}) blocked?", pos.x, pos.y),
            vec![yes.clone(), no.clone()],
            CommCost::from_units(2),
            0.0,
            model,
            1.0,
            self.config.ask_timeout,
        )
        .map_err(LabError::of)?;
        let report = controller
            .evaluate_actions(&AdvanceSpec::default(), None, Some(&spec))
            .map_err(LabError::of)?;
        let (evoi, net_evoi) = report
            .ask
            .as_ref()
            .map_or((0.0, 0.0), |a| (a.evoi, a.net_evoi));
        let mut record = AskRecord {
            tick: self.tick,
            pos,
            selected: report.selected,
            evoi,
            net_evoi,
            outcome: AskOutcome::Pending,
        };
        let idx = self.asks.len();
        if report.selected != EpistemicActionKind::Ask {
            record.outcome = AskOutcome::NotSelected(report.selected);
            self.asks.push(record);
            return Ok(idx);
        }
        if !self.permissions.contains(&Permission::new(ASK_PERMISSION)) {
            record.outcome = AskOutcome::PermissionDenied;
            self.asks.push(record);
            return Ok(idx);
        }
        let request_id = MessageId::new(self.fresh_id());
        let dispatch = controller
            .dispatch_ask(&mut self.bus, &self.clock, request_id, &spec)
            .map_err(LabError::of)?;
        let reply_id = MessageId::new(self.fresh_id());
        let scheduled = match self.provider.behavior(pos) {
            CannedBehavior::Answer {
                obstacle,
                confidence,
                latency,
            } => {
                let due = dispatch.sent_at.saturating_add(latency);
                let payload = CommPayload::answer(
                    if obstacle {
                        "yes, blocked"
                    } else {
                        "no, clear"
                    },
                    Some(if obstacle { yes } else { no }),
                    Some(gogamen_core::Confidence::new(confidence).map_err(LabError::of)?),
                )
                .map_err(LabError::of)?;
                (due, payload)
            }
            CannedBehavior::Unavailable => (
                dispatch.sent_at,
                CommPayload::no_response(NoResponseReason::Unavailable),
            ),
        };
        let reply = SituatedMessage::new(
            reply_id,
            self.human_id.clone(),
            self.agent_id.clone(),
            scope,
            scheduled.0,
            CommCost::from_units(1),
            scheduled.1,
        )
        .map_err(LabError::of)?
        .with_correlation_id(dispatch.request_id)
        .map_err(LabError::of)?;
        self.asks.push(record);
        self.pending.push(PendingAsk {
            record: idx,
            pos,
            controller,
            dispatch,
            scheduled: Some((scheduled.0, reply)),
            delivered: None,
        });
        Ok(idx)
    }

    /// Sondea (primero) y entrega (después) las respuestas pendientes. Nunca bloquea.
    fn pump_asks(&mut self, now: MonotonicInstant) -> Result<Vec<(usize, AskOutcome)>, LabError> {
        let mapper = BinaryCommOutcomeMapper::new(
            SensorId::new(DEFAULT_HUMAN_SENSOR_ID),
            OptionId::new("yes").map_err(LabError::of)?,
            OptionId::new("no").map_err(LabError::of)?,
        );
        let ask_model =
            LikelihoodModel::from_sensitivity_specificity(0.9, 0.9).map_err(LabError::of)?;
        let mut resolved = Vec::new();
        let mut still = Vec::new();
        for mut p in std::mem::take(&mut self.pending) {
            let polled = p
                .controller
                .poll_ask_reply(&mut self.bus, &self.clock, &p.dispatch, &ask_model, &mapper)
                .map_err(LabError::of)?;
            let outcome = match polled {
                AskPollOutcome::Pending => None,
                AskPollOutcome::TimedOut { .. } => Some(AskOutcome::TimedOut),
                AskPollOutcome::NoResponse { reason, .. } => Some(AskOutcome::NoResponse(reason)),
                AskPollOutcome::Observed { .. } => {
                    let msg = p.delivered.clone().ok_or_else(|| {
                        LabError("answer observed without delivered reply".to_owned())
                    })?;
                    let env = mapper
                        .to_observation(&msg, now, &FrameId::world(), &UnitTag::dimensionless())
                        .map_err(LabError::of)?;
                    let report: HumanReport = env.map(|present| CellHypothesis {
                        pos: p.pos,
                        hypothesis: HypothesisKind::Obstacle,
                        present,
                    });
                    let obstacle = report.payload().present;
                    self.apply_human(&report, "human-ask")?;
                    Some(AskOutcome::Answered { obstacle })
                }
            };
            match outcome {
                Some(o) => {
                    self.asks[p.record].outcome = o;
                    resolved.push((p.record, o));
                }
                None => {
                    if p.scheduled.as_ref().is_some_and(|(due, _)| *due <= now)
                        && let Some((_, reply)) = p.scheduled.take()
                    {
                        self.bus.send(reply.clone()).map_err(LabError::of)?;
                        p.delivered = Some(reply);
                    }
                    still.push(p);
                }
            }
        }
        self.pending = still;
        Ok(resolved)
    }

    /// Un tick: reloj simulado, respuestas Ask, observar (sensor con oclusión) -> planificar
    /// (sólo creencias) -> actuar (dinámica real del lab).
    pub fn tick(&mut self) -> Result<TickReport, LabError> {
        self.tick += 1;
        self.clock.advance(self.config.dt);
        let now = self.clock.now();
        let ask_outcomes = self.pump_asks(now)?;

        let obs = self.truth.observe_occluded(now);
        let (sens, spec) = reliability(&self.beliefs.config().direct_model);
        let mut direct_updates = 0;
        for env in obs.visible_cells() {
            let pos = env.payload().pos;
            let before = self.p_obstacle(pos)?;
            let outcome = self
                .beliefs
                .apply_direct_observation(env)
                .map_err(LabError::of)?;
            let after = self.p_obstacle(pos)?;
            if outcome.statuses[HypothesisKind::Obstacle.index()].is_updated() {
                direct_updates += 1;
                self.traces.push(UpdateTrace {
                    tick: self.tick,
                    pos,
                    source: EvidenceSource::DirectSensor,
                    channel: "direct-sensor",
                    before,
                    after,
                    status: outcome.statuses[HypothesisKind::Obstacle.index()],
                    reported_confidence: env.confidence().as_f64(),
                    source_sensitivity: sens,
                    source_specificity: spec,
                    evidence: env.evidence_ref(),
                });
            }
        }

        let action = plan(
            &self.planner,
            &PlannerInput {
                beliefs: &self.beliefs,
                agent_pos: obs.agent_pos(),
            },
        )?;
        let agent = self.truth.step_agent(action);
        self.last_action = Some(action);
        Ok(TickReport {
            tick: self.tick,
            action,
            agent,
            direct_updates,
            ask_outcomes,
        })
    }

    /// Vista del agente (sin verdad de terreno).
    pub fn agent_view(&self) -> AgentView {
        let size = self.beliefs.size();
        let mut cells = Vec::with_capacity(size.cell_count());
        for y in 0..size.height {
            for x in 0..size.width {
                let pos = GridPos::new(x, y);
                let get = |r: Result<f64, _>| r.unwrap_or(f64::NAN);
                cells.push(BeliefCellView {
                    pos,
                    p_obstacle: get(self.beliefs.probability(pos, HypothesisKind::Obstacle)),
                    human_updates: self
                        .beliefs
                        .human_update_count(pos, HypothesisKind::Obstacle)
                        .unwrap_or(0),
                    direct_updates: self
                        .beliefs
                        .direct_update_count(pos, HypothesisKind::Obstacle)
                        .unwrap_or(0),
                });
            }
        }
        AgentView {
            tick: self.tick,
            size,
            agent: self.truth.agent_pos(),
            cells,
            traces: self.traces.clone(),
            asks: self.asks.clone(),
            last_action: self.last_action,
        }
    }

    /// Vista del evaluador (agente + mapa real).
    pub fn evaluator_view(&self) -> EvaluatorView {
        EvaluatorView {
            agent: self.agent_view(),
            truth: self.truth_map.clone(),
        }
    }

    /// Escena según el modo explícito.
    pub fn scene(&self, mode: ViewMode) -> Scene2D {
        match mode {
            ViewMode::Agent => project_agent_scene(&self.agent_view()),
            ViewMode::Evaluator => project_evaluator_scene(&self.evaluator_view()),
        }
    }
}

/// Mapas de ejemplo del escenario: 7x5, inicio `(1,1)`, meta `(5,2)`, y la celda `(2,1)` es
/// obstáculo real (`true`, aviso verdadero) o libre (`false`, aviso falso).
pub fn scenario_map(obstacle_at_2_1: bool) -> GridMap {
    let row1 = if obstacle_at_2_1 {
        "#S#...#"
    } else {
        "#S....#"
    };
    GridMap::from_ascii(&["#######", row1, "#....G#", "#.....#", "#######"])
        .expect("valid scenario map")
}

/// Aviso humano canónico de la rejilla (el grounding offline lo aterriza en `(2,1)`).
pub const WARNING_TEXT: &str = "Aviso: obstáculo en celda (2,1) de la ruta directa";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive::scene_texts;

    fn perms(granted: bool) -> PermissionSet {
        if granted {
            PermissionSet::empty().with_permission(Permission::new(ASK_PERMISSION))
        } else {
            PermissionSet::empty()
        }
    }

    fn answer(obstacle: bool, latency_ms: u64) -> OfflineAskProvider {
        OfflineAskProvider::new(CannedBehavior::Answer {
            obstacle,
            confidence: 0.9,
            latency: Duration::from_millis(latency_ms),
        })
    }

    fn rollout(
        obstacle: bool,
        radius: usize,
        provider: OfflineAskProvider,
        granted: bool,
    ) -> LanguageWorldRollout {
        let mut cfg = LanguageWorldLabConfig::new(scenario_map(obstacle));
        cfg.vision_radius = radius;
        LanguageWorldRollout::new(cfg, provider, perms(granted)).expect("rollout")
    }

    const CELL: GridPos = GridPos::new(2, 1);

    #[test]
    fn ac1_true_warning_updates_beliefs_through_core_with_trace() {
        let mut r = rollout(true, 1, answer(true, 0), true);
        let prior = r.p_obstacle(CELL).unwrap();
        r.submit_human_notice(WARNING_TEXT).unwrap();
        let t = r.traces().last().unwrap().clone();
        assert_eq!(t.source, EvidenceSource::Human);
        assert_eq!(t.channel, "human-notice");
        assert_eq!(t.pos, CELL);
        assert_eq!(t.before, prior);
        assert!(t.after > t.before, "{t:?}");
        assert_eq!(t.status, BeliefUpdateStatus::Updated);
        assert!(t.source_sensitivity > 0.8 && t.source_specificity > 0.8);
        let scene = project_agent_scene(&r.agent_view());
        let text = scene_texts(&scene).join("\n");
        assert!(text.contains("src=human"), "{text}");
        assert!(text.contains(&format!("{:.3} -> {:.3}", t.before, t.after)));
        assert!(text.contains(&format!("ev=E{}", t.evidence.as_u64())));
        assert!(text.contains("conf=0.85"));
        assert!(text.contains("reliability(sens=0.85,spec=0.85)"));
        assert!(text.contains("FLAGGED uncorroborated(source=human)"));
        // la creencia que ve la vista es la del mapa de core
        let v = r.agent_view();
        let cell = v.cells.iter().find(|c| c.pos == CELL).unwrap();
        assert_eq!(cell.p_obstacle, t.after);
        assert_eq!(cell.evidence(), CellEvidence::HumanOnly);
        // tick: la medición directa confirma el muro
        r.tick().unwrap();
        assert!(r.p_obstacle(CELL).unwrap() > t.after);
    }

    #[test]
    fn ac1_false_warning_raises_then_is_corrected_by_direct_evidence() {
        let mut r = rollout(false, 1, answer(true, 0), true);
        r.submit_human_notice(WARNING_TEXT).unwrap();
        let human = r.traces()[0].clone();
        assert!(human.after > human.before);
        r.tick().unwrap();
        let direct = r
            .traces()
            .iter()
            .find(|t| t.source == EvidenceSource::DirectSensor && t.pos == CELL)
            .expect("direct trace for the warned cell")
            .clone();
        assert!((direct.before - human.after).abs() < 1e-12);
        assert!(direct.after < human.before, "corrected below the prior");
        let text = scene_texts(&project_agent_scene(&r.agent_view())).join("\n");
        assert!(text.contains("later-corrected-by-direct"), "{text}");
        assert!(!text.contains("FLAGGED"));
        assert!(text.contains("src=direct-sensor"));
    }

    #[test]
    fn ac1_uncorroborated_false_warning_stays_flagged_with_source() {
        // sin visión directa de (2,1) el aviso falso permanece, marcado con su fuente
        let mut r = rollout(false, 0, answer(true, 0), true);
        r.submit_human_notice(WARNING_TEXT).unwrap();
        r.tick().unwrap();
        let v = r.agent_view();
        assert_eq!(
            v.cells.iter().find(|c| c.pos == CELL).unwrap().evidence(),
            CellEvidence::HumanOnly
        );
        let text = scene_texts(&project_agent_scene(&v)).join("\n");
        assert!(text.contains("FLAGGED uncorroborated(source=human)"));
        assert!(text.contains("human?"));
        // el evaluador ve el desajuste creencia vs verdad sólo si P >= 0.5
        let ev = scene_texts(&project_evaluator_scene(&r.evaluator_view())).join("\n");
        assert!(ev.contains("== Belief vs ground truth =="));
    }

    #[test]
    fn ac1_ask_answer_updates_beliefs_with_human_source() {
        let mut r = rollout(true, 0, answer(true, 200), true);
        let before = r.p_obstacle(CELL).unwrap();
        let idx = r.ask_about(CELL).unwrap();
        assert_eq!(r.asks()[idx].selected, EpistemicActionKind::Ask);
        assert_eq!(r.asks()[idx].outcome, AskOutcome::Pending);
        for _ in 0..4 {
            r.tick().unwrap();
        }
        assert_eq!(
            r.asks()[idx].outcome,
            AskOutcome::Answered { obstacle: true }
        );
        let t = r
            .traces()
            .iter()
            .find(|t| t.channel == "human-ask")
            .unwrap();
        assert_eq!(t.source, EvidenceSource::Human);
        assert!(t.after > before && t.before == before);
        assert_eq!(r.provider().calls(), 1);
    }

    #[test]
    fn ac2_ask_is_visible_epistemic_action_with_offline_answer() {
        let mut r = rollout(true, 0, answer(true, 100), true);
        r.ask_about(CELL).unwrap();
        for _ in 0..3 {
            r.tick().unwrap();
        }
        let text = scene_texts(&project_agent_scene(&r.agent_view())).join("\n");
        assert!(text.contains("== Epistemic actions (1) =="));
        assert!(text.contains("ASK (2,1) selected=Ask"));
        assert!(text.contains("outcome=answered(blocked=true)"));
    }

    #[test]
    fn ac2_denied_permission_is_recorded_and_loop_continues() {
        let mut r = rollout(true, 1, answer(true, 0), false);
        let idx = r.ask_about(CELL).unwrap();
        assert_eq!(r.asks()[idx].outcome, AskOutcome::PermissionDenied);
        assert_eq!(r.provider().calls(), 0, "provider never contacted");
        let before = r.tick_count();
        let rep = r.tick().unwrap();
        assert_eq!(rep.tick, before + 1);
        assert!(rep.direct_updates > 0, "observe-plan-act still ran");
        let text = scene_texts(&project_agent_scene(&r.agent_view())).join("\n");
        assert!(text.contains("permission-denied(non-blocking)"));
    }

    #[test]
    fn ac2_slow_provider_times_out_without_blocking_and_beliefs_unchanged() {
        // latencia 5s > plazo 1s (tiempo simulado; no hay sleeps)
        let mut r = rollout(true, 0, answer(true, 5_000), true);
        let before = r.p_obstacle(CELL).unwrap();
        let idx = r.ask_about(CELL).unwrap();
        let mut agent_ticks = 0;
        for _ in 0..12 {
            let rep = r.tick().unwrap();
            agent_ticks += 1;
            assert!(r.last_action_is_some());
            assert_eq!(rep.tick, agent_ticks);
        }
        assert_eq!(r.asks()[idx].outcome, AskOutcome::TimedOut);
        assert_eq!(r.p_obstacle(CELL).unwrap(), before);
        assert!(r.traces().iter().all(|t| t.channel != "human-ask"));
    }

    #[test]
    fn ac2_unavailable_provider_records_no_response_and_continues() {
        let mut r = rollout(
            true,
            0,
            OfflineAskProvider::new(CannedBehavior::Unavailable),
            true,
        );
        let idx = r.ask_about(CELL).unwrap();
        for _ in 0..3 {
            r.tick().unwrap();
        }
        assert_eq!(
            r.asks()[idx].outcome,
            AskOutcome::NoResponse(NoResponseReason::Unavailable)
        );
        assert_eq!(r.tick_count(), 3);
    }

    #[test]
    fn ac2_ask_not_selected_when_belief_is_already_certain() {
        let mut r = rollout(true, 1, answer(true, 0), true);
        r.tick().unwrap(); // (2,1) observado directamente: P ~ 0.99
        let idx = r.ask_about(CELL).unwrap();
        assert!(matches!(r.asks()[idx].outcome, AskOutcome::NotSelected(_)));
        assert_eq!(r.provider().calls(), 0);
    }

    fn hidden_variant(wall: bool) -> GridMap {
        let row3 = if wall { "#....#." } else { "#......" };
        let _ = row3;
        let row = if wall { "#....##" } else { "#.....#" };
        GridMap::from_ascii(&["#######", "#S....#", "#....G#", row, "#######"]).unwrap()
    }

    #[test]
    fn ac3_agent_scene_has_no_ground_truth_and_ignores_hidden_cells() {
        let build = |wall: bool| {
            let mut cfg = LanguageWorldLabConfig::new(hidden_variant(wall));
            cfg.vision_radius = 1;
            let mut r = LanguageWorldRollout::new(cfg, answer(true, 0), perms(true)).unwrap();
            for _ in 0..3 {
                r.tick().unwrap();
            }
            r
        };
        let (a, b) = (build(true), build(false));
        // (5,3) no ha sido visto aún
        let hidden = GridPos::new(5, 3);
        let va = a.agent_view();
        assert_eq!(
            va.cells
                .iter()
                .find(|c| c.pos == hidden)
                .unwrap()
                .direct_updates,
            0
        );
        // misma historia observable => escena del agente idéntica pese a verdad distinta
        assert_eq!(va, b.agent_view());
        let sa = scene_texts(&a.scene(ViewMode::Agent));
        assert_eq!(sa, scene_texts(&b.scene(ViewMode::Agent)));
        let joined = sa.join("\n");
        for leak in ["#", "GROUND TRUTH", "MISMATCH", "truth"] {
            assert!(!joined.contains(leak), "agent scene leaks `{leak}`");
        }
        // el evaluador sí distingue ambas verdades y marca el muro
        let ea = scene_texts(&a.scene(ViewMode::Evaluator));
        let eb = scene_texts(&b.scene(ViewMode::Evaluator));
        assert_ne!(ea, eb);
        assert!(ea.join("\n").contains("GROUND TRUTH"));
        assert!(ea.iter().any(|t| t == "#"));
        assert_ne!(ea, sa);
    }

    #[test]
    fn ac3_no_ground_truth_used_as_sensor() {
        let mut r = rollout(true, 0, answer(true, 0), true);
        r.tick().unwrap();
        let v = r.agent_view();
        let touched: Vec<_> = v.cells.iter().filter(|c| c.direct_updates > 0).collect();
        assert_eq!(touched.len(), 1, "radius 0 sees only its own cell");
        for c in v.cells.iter().filter(|c| c.direct_updates == 0) {
            assert_eq!(c.p_obstacle, 0.4, "hidden cell keeps prior");
        }
    }

    #[test]
    fn ac3_planner_input_type_has_no_ground_truth_field_and_planner_only_gets_it() {
        let src = include_str!("language_world_lab.rs");
        let start = src.find("pub struct PlannerInput").unwrap();
        let block = &src[start..start + src[start..].find("\n}").unwrap()];
        for banned in ["GridMap", "GroundTruth", "truth", "Evaluator"] {
            assert!(!block.contains(banned), "PlannerInput has `{banned}`");
        }
        assert!(block.contains("beliefs: &'a GridBeliefMap"));
        let prod = &src[..src.find("#[cfg(test)]").unwrap()];
        assert_eq!(prod.matches(".choose_action(").count(), 1);
        assert!(prod.contains(".choose_action(input.beliefs, input.agent_pos)"));
    }

    #[test]
    fn ac3_render_code_has_no_dynamics_planner_or_belief_updates() {
        let src = include_str!("language_world_lab.rs");
        let render: String = src
            [src.find("pub const ASK_PERMISSION").unwrap()..src.find("// ENGINE:").unwrap()]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "std::net",
            "std::fs",
            "http",
            "LLM",
            "SystemOne",
            "Planner",
            "choose_action",
            "step_agent",
            "observe_occluded",
            "apply_human_report",
            "apply_direct_observation",
            "update_with_observation",
            "GroundTruthGrid",
            "GridBeliefMap",
            "CommunicationPort",
            "LocalCommunicationBus",
            "EpistemicController",
            "ground_human_notice",
            "async ",
        ] {
            assert!(!render.contains(banned), "render code contains `{banned}`");
        }
        assert!(render.contains("pub fn project_agent_scene(view: &AgentView) -> Scene2D"));
        assert!(render.contains("pub fn project_evaluator_scene(view: &EvaluatorView) -> Scene2D"));
    }

    #[test]
    fn projection_is_deterministic_and_renders_svg() {
        let mut r = rollout(false, 1, answer(false, 0), true);
        r.submit_human_notice(WARNING_TEXT).unwrap();
        r.tick().unwrap();
        for mode in [ViewMode::Agent, ViewMode::Evaluator] {
            let s1 = render_svg(&r.scene(mode), 800.0, 600.0);
            let s2 = render_svg(&r.scene(mode), 800.0, 600.0);
            assert_eq!(s1, s2);
            assert!(s1.starts_with("<svg"));
        }
    }

    impl LanguageWorldRollout {
        fn last_action_is_some(&self) -> bool {
            self.last_action.is_some()
        }
    }
}
