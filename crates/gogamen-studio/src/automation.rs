//! Vista de grafo y eventos de automatización (GOG-106): proyector puro `AutomationView -> Scene2D`.
//!
//! - **Datos, no motores:** el proyector ([`project_automation_scene`]) sólo lee un
//!   [`AutomationView`] inmutable. No ejecuta dinámica ni planificador (test textual de guarda).
//! - **Rollout headless sobre el motor real:** [`run_automation`] conduce el
//!   [`ServiceSupervisor`] real (GOG-49, sobre `StateMachineWorkflow` de GOG-48) con un reloj
//!   virtual; compila además la `SystemSpec` equivalente con `SpecCompiler` para obtener el
//!   grafo del plan compilado ([`gogamen_compiler::ExecutablePlan`]). No requiere FMC ni Jev real:
//!   ni proveedor cognitivo ni red; la traza mostrada es la del propio supervisor
//!   (`SupervisorTraceRecord` + `trace_hash`), idéntica a la de `ServiceSupervisor::replay`.
//! - **Ejecutado vs propuesto:** cada acción del workflow es un efecto *propuesto*
//!   ([`EffectStatus::Proposed`], `EpistemicKind::Hypothetical`) salvo que el actuador sintético
//!   registre su ejecución en ese tick ([`EffectStatus::Executed`], `Observed`).
//! - **Errores, reintentos y tiempos:** cada paso muestra su estado (`RETRY n/m`,
//!   `ERROR-BRANCH`, `TIMEOUT-BRANCH`), su instante monotónico virtual y el delta respecto al paso previo.
//! - **Layout simple y determinista:** profundidad topológica -> `x`, índice -> `y`. Sin editor de
//!   grafos. **Selección común:** cada nodo/arista/paso lleva un `EntityId` estable y
//!   [`AutomationView::snapshot_view`] lo expone al [`crate::Inspector`] genérico (GOG-92).
//!
//! `gogamen-core` y `gogamen-cli` no dependen de este módulo.

use std::fmt::Write;
use std::time::Duration;

use gogamen_core::{
    EntityId, EpistemicKind, GuardDecision, IdempotencyKey, MonotonicDeadline, MonotonicInstant,
    ProposedEffect, Tick, TransitionGuard, VirtualMonotonicClock, WorkflowStepError,
    WorkflowStepOutcome, WorkflowStepStatus, WorkflowTrigger,
};
use gogamen_lab_automation::{
    AutomationError, ServiceSupervisor, ServiceSupervisorConfig, SupervisorTraceRecord,
    SyntheticServiceEvent, compile_supervisor_spec,
};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};

use crate::camera::Camera2D;
use crate::inspector::{EntityRef, EntityView, Field, FieldValue, Provenance, SnapshotView};
use crate::viewport::SoftwareSvgBackend;

/// Base de `EntityId` de nodos del plan compilado.
pub const PLAN_NODE_BASE: u64 = 1000;
/// Base de `EntityId` de estados del workflow.
pub const STATE_BASE: u64 = 2000;
/// Base de `EntityId` de transiciones/ramas del workflow.
pub const TRANSITION_BASE: u64 = 3000;
/// Base de `EntityId` de pasos (resultados) de la traza.
pub const OUTCOME_BASE: u64 = 4000;
/// Época fija de las entidades de esta vista (la vista es inmutable).
pub const VIEW_EPOCH: u64 = 1;

/// Nodo del plan compilado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNodeView {
    pub id: String,
    /// `modulo::operacion` resuelto por el compilador.
    pub operation: String,
    /// Profundidad topológica (camino más largo desde una fuente).
    pub depth: usize,
}

/// Arista del plan compilado (`from`/`to` son índices en `plan_nodes`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanEdgeView {
    pub from: usize,
    pub to: usize,
    pub delayed: bool,
}

/// Estado del workflow del supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateView {
    pub name: String,
    pub terminal: bool,
    pub initial: bool,
    pub depth: usize,
}

/// Clase de transición.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionKind {
    Normal,
    OnError,
    OnTimeout,
}

/// Transición o rama del workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionView {
    /// `None` = desde cualquier estado no terminal (rama `OnError`/`OnTimeout` global).
    pub from: Option<String>,
    pub to: String,
    pub label: String,
    pub kind: TransitionKind,
}

/// Estado de un efecto del workflow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectStatus {
    /// Propuesto por el workflow; **no** ejecutado.
    Proposed,
    /// Ejecutado: el actuador sintético lo registró en ese tick.
    Executed,
}

/// Efecto asociado a un paso.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectView {
    pub action: String,
    pub target: String,
    pub effect: String,
    pub status: EffectStatus,
}

/// Resultado de un paso del workflow con su tiempo virtual.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OutcomeView {
    pub tick: u64,
    pub timestamp_ns: u128,
    /// Nanosegundos desde el paso anterior (0 en el primero).
    pub delta_ns: u128,
    pub from: String,
    pub to: String,
    pub status: WorkflowStepStatus,
    /// Origen: `event` o `error:<code>`.
    pub source: String,
    pub effects: Vec<EffectView>,
}

/// Decisión de la guardia (remediación) en un tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardView {
    pub tick: u64,
    pub decision: String,
    pub allowed: bool,
}

/// Vista completa e inmutable de la automatización.
#[derive(Debug, Clone, PartialEq)]
pub struct AutomationView {
    pub title: String,
    pub spec_id: String,
    pub plan_fingerprint: u64,
    pub plan_nodes: Vec<PlanNodeView>,
    pub plan_edges: Vec<PlanEdgeView>,
    pub states: Vec<StateView>,
    pub transitions: Vec<TransitionView>,
    pub max_retries: u32,
    pub active_state: String,
    /// Índice en `transitions` de la última transición realmente tomada.
    pub active_transition: Option<usize>,
    pub outcomes: Vec<OutcomeView>,
    pub guards: Vec<GuardView>,
    pub trace: Vec<SupervisorTraceRecord>,
    pub trace_hash: String,
    pub last_tick: u64,
}

impl AutomationView {
    /// Cuenta de efectos por estado.
    pub fn effect_count(&self, status: EffectStatus) -> usize {
        self.outcomes
            .iter()
            .flat_map(|o| &o.effects)
            .filter(|e| e.status == status)
            .count()
    }

    fn count_status(&self, f: impl Fn(&WorkflowStepStatus) -> bool) -> usize {
        self.outcomes.iter().filter(|o| f(&o.status)).count()
    }

    /// Adapta la vista al inspector genérico (selección común por `EntityRef`).
    pub fn snapshot_view(&self) -> SnapshotView {
        let t = self.last_tick;
        let text = |n: &str, v: &str| Field::new(n, FieldValue::Text(v.to_owned()), t);
        let int = |n: &str, v: i64| Field::new(n, FieldValue::Integer(v), t);
        let mut entities = Vec::new();
        for (i, n) in self.plan_nodes.iter().enumerate() {
            entities.push(EntityView {
                entity: EntityRef::new(PLAN_NODE_BASE + i as u64, VIEW_EPOCH),
                kind: "plan_node".into(),
                fields: vec![
                    text("node_id", &n.id),
                    text("operation", &n.operation),
                    int("depth", n.depth as i64),
                ],
            });
        }
        for (i, s) in self.states.iter().enumerate() {
            entities.push(EntityView {
                entity: EntityRef::new(STATE_BASE + i as u64, VIEW_EPOCH),
                kind: "workflow_state".into(),
                fields: vec![
                    text("name", &s.name),
                    Field::new("terminal", FieldValue::Bool(s.terminal), t),
                    Field::new("active", FieldValue::Bool(s.name == self.active_state), t),
                ],
            });
        }
        for (i, tr) in self.transitions.iter().enumerate() {
            entities.push(EntityView {
                entity: EntityRef::new(TRANSITION_BASE + i as u64, VIEW_EPOCH),
                kind: "transition".into(),
                fields: vec![
                    text("from", tr.from.as_deref().unwrap_or("*")),
                    text("to", &tr.to),
                    text("trigger", &tr.label),
                    text("branch", &format!("{:?}", tr.kind)),
                    Field::new(
                        "active",
                        FieldValue::Bool(self.active_transition == Some(i)),
                        t,
                    ),
                ],
            });
        }
        for (i, o) in self.outcomes.iter().enumerate() {
            let count = |s| o.effects.iter().filter(|e| e.status == s).count() as i64;
            entities.push(EntityView {
                entity: EntityRef::new(OUTCOME_BASE + i as u64, VIEW_EPOCH),
                kind: "step".into(),
                fields: vec![
                    int("tick", o.tick as i64),
                    Field::new(
                        "timestamp_ns",
                        FieldValue::Integer(o.timestamp_ns as i64),
                        o.tick,
                    )
                    .unit("ns")
                    .provenance(Provenance::Observed),
                    text("from", &o.from),
                    text("to", &o.to),
                    text("status", &status_text(&o.status)),
                    text("source", &o.source),
                    int("effects_executed", count(EffectStatus::Executed)),
                    int(
                        "effects_proposed_not_executed",
                        count(EffectStatus::Proposed),
                    ),
                ],
            });
        }
        SnapshotView {
            revision: self.outcomes.len() as u64,
            tick: t,
            entities,
        }
    }
}

/// Entrada de un paso del guion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScriptInput {
    Event(SyntheticServiceEvent),
    /// Error de paso reportado al workflow (activa reintentos / `OnError`).
    Error {
        code: String,
        detail: String,
    },
}

/// Paso del guion: avanza el reloj virtual y entrega la entrada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptStep {
    pub tick: u64,
    pub advance_ms: u64,
    pub input: ScriptInput,
}

/// Escenario headless (sin FMC ni Jev real: sólo eventos sintéticos y reloj virtual).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomationScenario {
    pub config: ServiceSupervisorConfig,
    /// Plazo `OnTimeout` relativo al inicio (ms de reloj virtual).
    pub deadline_ms: Option<u64>,
    pub script: Vec<ScriptStep>,
}

fn status_text(s: &WorkflowStepStatus) -> String {
    match s {
        WorkflowStepStatus::Transitioned => "TRANSITIONED".into(),
        WorkflowStepStatus::DuplicateIgnored => "DUPLICATE-IGNORED".into(),
        WorkflowStepStatus::NoMatchingTransition => "NO-MATCH".into(),
        WorkflowStepStatus::Retrying {
            attempt,
            max_retries,
        } => format!("RETRY {attempt}/{max_retries}"),
        WorkflowStepStatus::ErrorBranchTaken => "ERROR-BRANCH".into(),
        WorkflowStepStatus::TimeoutBranchTaken => "TIMEOUT-BRANCH".into(),
        WorkflowStepStatus::AlreadyTerminal => "ALREADY-TERMINAL".into(),
    }
}

fn effect_text(e: &ProposedEffect) -> String {
    match e {
        ProposedEffect::EmitEvent { topic, payload } => format!("emit {topic}:{payload}"),
        ProposedEffect::Log { level, message } => format!("log {level:?}:{message}"),
        ProposedEffect::Custom { kind, detail } => format!("custom {kind}:{detail}"),
    }
}

fn guard_text(d: &GuardDecision) -> String {
    match d {
        GuardDecision::Allow(_) => "allow".into(),
        GuardDecision::Fallback(_) => "fallback".into(),
        GuardDecision::NoOp => "noop".into(),
        GuardDecision::Reject(r) => format!("reject:{r:?}"),
    }
}

fn trigger_label(g: &TransitionGuard) -> String {
    match g {
        TransitionGuard::Always => "always".into(),
        TransitionGuard::Trigger(WorkflowTrigger::Event(k)) => k.as_str().to_owned(),
        TransitionGuard::Trigger(WorkflowTrigger::CognitiveDecision { decision_code, .. }) => {
            format!("decision:{decision_code}")
        }
        TransitionGuard::PayloadEquals(p) => format!("payload=={p}"),
        TransitionGuard::PayloadContains(p) => format!("payload~{p}"),
        TransitionGuard::WithinRetryBudget => "within_retry_budget".into(),
        TransitionGuard::All(v) => v.iter().map(trigger_label).collect::<Vec<_>>().join("&"),
        TransitionGuard::Any(v) => v.iter().map(trigger_label).collect::<Vec<_>>().join("|"),
    }
}

/// Profundidad por camino más largo; `order` debe ser un orden topológico.
fn longest_depths(n: usize, edges: &[(usize, usize)], order: &[usize]) -> Vec<usize> {
    let mut depth = vec![0usize; n];
    for &u in order {
        for &(a, b) in edges {
            if a == u && depth[b] < depth[u] + 1 {
                depth[b] = depth[u] + 1;
            }
        }
    }
    depth
}

/// Ejecuta el escenario sobre el supervisor real y construye la vista.
pub fn run_automation(
    title: &str,
    scenario: &AutomationScenario,
) -> Result<AutomationView, AutomationError> {
    let spec = scenario.config.to_system_spec();
    let (_, plan) = compile_supervisor_spec(&spec)?;
    let mut sup = ServiceSupervisor::new(scenario.config.clone())?;
    let clock = VirtualMonotonicClock::new();
    if let Some(ms) = scenario.deadline_ms {
        sup.set_deadline(Some(MonotonicDeadline::from_now(
            &clock,
            Duration::from_millis(ms),
        )));
    }

    // Plan compilado: nodos (orden topológico) y aristas.
    let graph = plan.graph_plan();
    let order: Vec<usize> = plan.execution_order().iter().map(|n| n.0).collect();
    let mut plan_nodes: Vec<PlanNodeView> = Vec::new();
    for &i in &order {
        let idx = gogamen_core::NodeIndex(i);
        let id = graph.node_id(idx).map_or("?", |n| n.as_str()).to_owned();
        let operation = plan
            .node_operation(idx)
            .and_then(|op| plan.factory(op.factory))
            .map_or_else(
                || "?".to_owned(),
                |f| format!("{}::{}", f.module_id.as_str(), f.operation_id.as_str()),
            );
        plan_nodes.push(PlanNodeView {
            id,
            operation,
            depth: 0,
        });
    }
    // `plan_nodes[k]` corresponde a `order[k]`.
    let pos = |node: gogamen_core::NodeIndex| order.iter().position(|&x| x == node.0);
    let mut plan_edges = Vec::new();
    for &i in &order {
        for t in plan.instant_outgoing(gogamen_core::NodeIndex(i)) {
            if let (Some(a), Some(b)) = (pos(t.from_node), pos(t.to_node)) {
                plan_edges.push(PlanEdgeView {
                    from: a,
                    to: b,
                    delayed: false,
                });
            }
        }
    }
    for t in plan.delayed_transfers() {
        if let (Some(a), Some(b)) = (pos(t.from_node), pos(t.to_node)) {
            plan_edges.push(PlanEdgeView {
                from: a,
                to: b,
                delayed: true,
            });
        }
    }
    let pe: Vec<(usize, usize)> = plan_edges.iter().map(|e| (e.from, e.to)).collect();
    let topo: Vec<usize> = (0..plan_nodes.len()).collect();
    for (n, d) in plan_nodes
        .iter_mut()
        .zip(longest_depths(topo.len(), &pe, &topo))
    {
        n.depth = d;
    }

    // Workflow: estados y transiciones del spec real.
    let wf = sup.workflow_spec().clone();
    let mut transitions: Vec<TransitionView> = wf
        .transitions
        .iter()
        .map(|t| TransitionView {
            from: Some(t.from.as_str().to_owned()),
            to: t.to.as_str().to_owned(),
            label: trigger_label(&t.guard),
            kind: TransitionKind::Normal,
        })
        .collect();
    for b in &wf.error_branches {
        transitions.push(TransitionView {
            from: b.from.as_ref().map(|s| s.as_str().to_owned()),
            to: b.on_exhausted_state.as_str().to_owned(),
            label: format!("OnError(retries<={})", b.retry_policy.max_retries),
            kind: TransitionKind::OnError,
        });
    }
    for b in &wf.timeout_branches {
        transitions.push(TransitionView {
            from: b.from.as_ref().map(|s| s.as_str().to_owned()),
            to: b.to.as_str().to_owned(),
            label: "OnTimeout".into(),
            kind: TransitionKind::OnTimeout,
        });
    }
    let mut names: Vec<String> = vec![wf.initial_state.as_str().to_owned()];
    for t in &transitions {
        for n in t.from.iter().chain(std::iter::once(&t.to)) {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
    }
    for s in &wf.terminal_states {
        if !names.iter().any(|n| n == s.as_str()) {
            names.push(s.as_str().to_owned());
        }
    }
    // Profundidad: BFS sobre transiciones normales; ramas globales detrás del máximo.
    let mut depth: Vec<Option<usize>> = vec![None; names.len()];
    depth[0] = Some(0);
    let mut queue = std::collections::VecDeque::from([0usize]);
    while let Some(u) = queue.pop_front() {
        for t in transitions
            .iter()
            .filter(|t| t.kind == TransitionKind::Normal && t.from.as_deref() == Some(&names[u]))
        {
            let v = names.iter().position(|n| *n == t.to).expect("known state");
            if depth[v].is_none() {
                depth[v] = depth[u].map(|d| d + 1);
                queue.push_back(v);
            }
        }
    }
    let max_depth = depth.iter().flatten().copied().max().unwrap_or(0);
    let states: Vec<StateView> = names
        .iter()
        .enumerate()
        .map(|(i, n)| StateView {
            name: n.clone(),
            terminal: wf.terminal_states.iter().any(|s| s.as_str() == n),
            initial: i == 0,
            depth: depth[i].unwrap_or(max_depth + 1),
        })
        .collect();

    // Rollout sobre el supervisor real.
    let service = scenario.config.service_name.clone();
    let mut outcomes: Vec<OutcomeView> = Vec::new();
    let mut guards = Vec::new();
    let mut last_ts = MonotonicInstant::ZERO;
    let mut last_tick = 0u64;
    for (idx, step) in scenario.script.iter().enumerate() {
        clock.advance(Duration::from_millis(step.advance_ms));
        let tick = Tick::new(step.tick);
        last_tick = step.tick;
        let (raw, source): (Vec<WorkflowStepOutcome>, String) = match &step.input {
            ScriptInput::Event(ev) => {
                let key = IdempotencyKey::new(format!("lab-{idx}-t{}", step.tick))?;
                let report = sup.process_event(ev, Some(key), tick, &clock, None)?;
                if let Some(d) = &report.guard_decision {
                    guards.push(GuardView {
                        tick: step.tick,
                        decision: guard_text(d),
                        allowed: d.is_allowed(),
                    });
                }
                (report.workflow_outcomes, "event".into())
            }
            ScriptInput::Error { code, detail } => {
                let err = WorkflowStepError::new(code.clone(), detail.clone())?;
                let o = sup.report_error(&err, None, tick, &clock)?;
                (vec![o], format!("error:{code}"))
            }
        };
        for o in raw {
            let executed = sup.actuator().executed_actions();
            let effects = o
                .proposed_actions
                .iter()
                .map(|a| {
                    let ran = executed.iter().any(|x| {
                        x.tick == o.tick
                            && x.action_name == a.action_name()
                            && x.service_name == service
                    });
                    EffectView {
                        action: a.action_name().to_owned(),
                        target: a.target().to_owned(),
                        effect: effect_text(a.effect()),
                        status: if ran {
                            EffectStatus::Executed
                        } else {
                            EffectStatus::Proposed
                        },
                    }
                })
                .collect();
            let ts = o.timestamp;
            outcomes.push(OutcomeView {
                tick: o.tick.as_u64(),
                timestamp_ns: ts.as_nanos(),
                delta_ns: ts.as_nanos().saturating_sub(last_ts.as_nanos()),
                from: o.previous_state.as_str().to_owned(),
                to: o.current_state.as_str().to_owned(),
                status: o.status,
                source: source.clone(),
                effects,
            });
            last_ts = ts;
        }
    }

    // Transición activa: la última realmente tomada (cambia de estado).
    let active_transition = outcomes
        .iter()
        .rev()
        .find(|o| o.from != o.to)
        .and_then(|o| {
            let kind = match o.status {
                WorkflowStepStatus::ErrorBranchTaken => TransitionKind::OnError,
                WorkflowStepStatus::TimeoutBranchTaken => TransitionKind::OnTimeout,
                _ => TransitionKind::Normal,
            };
            transitions.iter().position(|t| {
                t.kind == kind && t.to == o.to && t.from.as_deref().is_none_or(|f| f == o.from)
            })
        });

    Ok(AutomationView {
        title: title.to_owned(),
        spec_id: plan.spec_id().to_owned(),
        plan_fingerprint: plan.semantic_fingerprint(),
        plan_nodes,
        plan_edges,
        states,
        transitions,
        max_retries: scenario.config.max_retries,
        active_state: sup.current_state().as_str().to_owned(),
        active_transition,
        outcomes,
        guards,
        trace: sup.trace().to_vec(),
        trace_hash: sup.trace_hash(),
        last_tick,
    })
}

// ---------------------------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------------------------

/// Incidente completo: alerta, diagnóstico, remediación (ejecutada) y resolución.
pub fn incident_scenario() -> AutomationScenario {
    let ev = |tick, advance_ms, e| ScriptStep {
        tick,
        advance_ms,
        input: ScriptInput::Event(e),
    };
    AutomationScenario {
        config: ServiceSupervisorConfig::default(),
        deadline_ms: None,
        script: vec![
            ev(1, 10, SyntheticServiceEvent::LatencySpike { p99_ms: 80 }),
            ev(2, 10, SyntheticServiceEvent::QueueBacklog { depth: 120 }),
            ev(
                3,
                25,
                SyntheticServiceEvent::ProbePassed { consecutive_ok: 2 },
            ),
        ],
    }
}

/// Errores con reintentos acotados hasta la rama `OnError`.
pub fn retry_error_scenario() -> AutomationScenario {
    let err = |tick, advance_ms| ScriptStep {
        tick,
        advance_ms,
        input: ScriptInput::Error {
            code: "E_IO".into(),
            detail: "synthetic failure".into(),
        },
    };
    AutomationScenario {
        config: ServiceSupervisorConfig {
            max_retries: 1,
            ..ServiceSupervisorConfig::default()
        },
        deadline_ms: None,
        script: vec![err(1, 5), err(2, 40)],
    }
}

/// Plazo vencido: rama `OnTimeout`.
pub fn timeout_scenario() -> AutomationScenario {
    AutomationScenario {
        config: ServiceSupervisorConfig::default(),
        deadline_ms: Some(50),
        script: vec![ScriptStep {
            tick: 1,
            advance_ms: 60,
            input: ScriptInput::Event(SyntheticServiceEvent::HealthDegraded { failed_checks: 1 }),
        }],
    }
}

// ---------------------------------------------------------------------------------------------
// Proyector puro
// ---------------------------------------------------------------------------------------------

const NODE_W: f64 = 170.0;
const NODE_H: f64 = 34.0;
const COL_W: f64 = 230.0;
const ROW_GAP: f64 = 62.0;
const LINE_H: f64 = 18.0;

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite layout point")
}

struct Builder {
    scene: Scene2D,
    next: u64,
}

impl Builder {
    fn add(
        &mut self,
        layer: LayerId,
        entity: Option<u64>,
        kind: EpistemicKind,
        prim: ScenePrimitive,
        active: bool,
    ) {
        let node = SceneNode::new(
            SceneNodeId::new(self.next),
            entity.map(EntityId::new),
            kind,
            prim,
            entity.is_some(),
        )
        .with_state_code(u32::from(active));
        self.next += 1;
        // las capas se crean de antemano; el id de nodo es único por construcción
        let _ = self.scene.add_node(layer, node);
    }
    fn label(&mut self, x: f64, y: f64, text: String, kind: EpistemicKind) {
        self.add(
            LayerId::Labels,
            None,
            kind,
            ScenePrimitive::label(pt(x, y), text),
            false,
        );
    }
}

fn ms(ns: u128) -> String {
    format!("{:.3}ms", ns as f64 / 1_000_000.0)
}

/// Posición `(x, y)` de nodo por `(depth, index)`.
fn slot(base_y: f64, depth: usize, index: usize) -> (f64, f64) {
    (20.0 + depth as f64 * COL_W, base_y + index as f64 * ROW_GAP)
}

/// Proyecta la vista a una [`Scene2D`] (píxeles, `y` hacia abajo). Función pura y determinista.
pub fn project_automation_scene(view: &AutomationView) -> Scene2D {
    let mut b = Builder {
        scene: Scene2D::new(
            SceneId::new(106),
            Tick::new(view.last_tick),
            EpistemicKind::Observed,
            CoordinateSpace::meters(),
        ),
        next: 1,
    };
    for layer in [
        LayerId::Environment,
        LayerId::Decisions,
        LayerId::Sensors,
        LayerId::Imagination,
        LayerId::Labels,
    ] {
        b.scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }
    let obs = EpistemicKind::Observed;
    b.label(
        10.0,
        16.0,
        format!(
            "Automation graph: {} spec={} plan_fp={:016x} tick={} active_state={}",
            view.title, view.spec_id, view.plan_fingerprint, view.last_tick, view.active_state
        ),
        obs,
    );

    // --- Plan compilado ---
    b.label(
        10.0,
        40.0,
        "== Compiled plan (nodes / dependencies) ==".into(),
        obs,
    );
    let plan_base = 56.0;
    let mut per_depth: Vec<usize> = Vec::new();
    let mut plan_pos = Vec::new();
    for n in &view.plan_nodes {
        if per_depth.len() <= n.depth {
            per_depth.resize(n.depth + 1, 0);
        }
        plan_pos.push(slot(plan_base, n.depth, per_depth[n.depth]));
        per_depth[n.depth] += 1;
    }
    for (i, (n, &(x, y))) in view.plan_nodes.iter().zip(&plan_pos).enumerate() {
        let id = PLAN_NODE_BASE + i as u64;
        b.add(
            LayerId::Environment,
            Some(id),
            obs,
            ScenePrimitive::rect(pt(x, y), NODE_W, NODE_H).expect("finite rect"),
            false,
        );
        b.label(x + 6.0, y + 14.0, format!("node {}", n.id), obs);
        b.label(x + 6.0, y + 28.0, n.operation.clone(), obs);
    }
    for (i, e) in view.plan_edges.iter().enumerate() {
        let ((x0, y0), (x1, y1)) = (plan_pos[e.from], plan_pos[e.to]);
        b.add(
            LayerId::Decisions,
            Some(TRANSITION_BASE + 500 + i as u64),
            obs,
            ScenePrimitive::polyline(
                vec![
                    pt(x0 + NODE_W, y0 + NODE_H / 2.0),
                    pt(x1, y1 + NODE_H / 2.0),
                ],
                false,
            )
            .expect("finite polyline"),
            false,
        );
        b.label(
            (x0 + NODE_W + x1) / 2.0 - 30.0,
            y0 + NODE_H / 2.0 - 4.0,
            format!(
                "{} -> {}",
                view.plan_nodes[e.from].id, view.plan_nodes[e.to].id
            ),
            obs,
        );
    }

    // --- Estados del workflow ---
    let plan_rows = per_depth.iter().copied().max().unwrap_or(1).max(1);
    let state_title_y = plan_base + plan_rows as f64 * ROW_GAP + 10.0;
    b.label(
        10.0,
        state_title_y,
        "== Supervisor workflow states / transitions ==".into(),
        obs,
    );
    let state_base = state_title_y + 16.0;
    let mut per_depth: Vec<usize> = Vec::new();
    let mut state_pos = Vec::new();
    for s in &view.states {
        if per_depth.len() <= s.depth {
            per_depth.resize(s.depth + 1, 0);
        }
        state_pos.push(slot(state_base, s.depth, per_depth[s.depth]));
        per_depth[s.depth] += 1;
    }
    let find = |name: &str| view.states.iter().position(|s| s.name == name);
    for (i, (s, &(x, y))) in view.states.iter().zip(&state_pos).enumerate() {
        let active = s.name == view.active_state;
        b.add(
            LayerId::Environment,
            Some(STATE_BASE + i as u64),
            obs,
            ScenePrimitive::rect(pt(x, y), NODE_W, NODE_H).expect("finite rect"),
            active,
        );
        if active {
            b.add(
                LayerId::Decisions,
                None,
                obs,
                ScenePrimitive::rect(pt(x - 3.0, y - 3.0), NODE_W + 6.0, NODE_H + 6.0)
                    .expect("finite rect"),
                true,
            );
        }
        let tags = format!(
            "{}{}{}",
            if s.initial { " [initial]" } else { "" },
            if s.terminal { " [terminal]" } else { "" },
            if active { " [ACTIVE]" } else { "" }
        );
        b.label(x + 6.0, y + 14.0, format!("state {}{tags}", s.name), obs);
        // ramas globales: anotación bajo el estado destino, sin polilíneas
        let mut line = 28.0;
        for t in view
            .transitions
            .iter()
            .filter(|t| t.from.is_none() && t.to == s.name)
        {
            b.label(
                x + 6.0,
                y + line,
                format!("<- {} from any state", t.label),
                obs,
            );
            line += 12.0;
        }
    }
    for (i, t) in view.transitions.iter().enumerate() {
        let (Some(from), Some(to)) = (t.from.as_deref().and_then(find), find(&t.to)) else {
            continue;
        };
        let active = view.active_transition == Some(i);
        let ((x0, y0), (x1, y1)) = (state_pos[from], state_pos[to]);
        b.add(
            LayerId::Decisions,
            Some(TRANSITION_BASE + i as u64),
            obs,
            ScenePrimitive::polyline(
                vec![
                    pt(x0 + NODE_W, y0 + NODE_H / 2.0),
                    pt(x1, y1 + NODE_H / 2.0),
                ],
                false,
            )
            .expect("finite polyline"),
            active,
        );
        b.label(
            x0 + NODE_W + 4.0,
            y0 + NODE_H / 2.0 - 4.0,
            format!(
                "{}{}",
                t.label,
                if active { " [ACTIVE TRANSITION]" } else { "" }
            ),
            obs,
        );
    }

    // --- Línea de eventos / tiempos ---
    let max_rows = per_depth.iter().copied().max().unwrap_or(1).max(1);
    let mut y = state_base + max_rows as f64 * ROW_GAP + 14.0;
    b.label(
        10.0,
        y,
        "== Events / timings (virtual clock; PROPOSED effects are NOT executed) ==".into(),
        obs,
    );
    y += LINE_H;
    for (i, o) in view.outcomes.iter().enumerate() {
        let kind = if matches!(
            o.status,
            WorkflowStepStatus::Retrying { .. }
                | WorkflowStepStatus::ErrorBranchTaken
                | WorkflowStepStatus::TimeoutBranchTaken
        ) {
            "ERR"
        } else {
            "ok "
        };
        b.add(
            LayerId::Environment,
            Some(OUTCOME_BASE + i as u64),
            obs,
            ScenePrimitive::rect(pt(6.0, y - 13.0), 1000.0, LINE_H - 2.0).expect("finite rect"),
            false,
        );
        b.label(
            10.0,
            y,
            format!(
                "#{i} tick={} t={} (+{}) {} -> {} {} [{kind}] src={}",
                o.tick,
                ms(o.timestamp_ns),
                ms(o.delta_ns),
                o.from,
                o.to,
                status_text(&o.status),
                o.source
            ),
            obs,
        );
        y += LINE_H;
        for e in &o.effects {
            let (tag, ek, layer) = match e.status {
                EffectStatus::Executed => ("EXECUTED", obs, LayerId::Sensors),
                EffectStatus::Proposed => (
                    "PROPOSED (not executed)",
                    EpistemicKind::Hypothetical,
                    LayerId::Imagination,
                ),
            };
            b.add(
                layer,
                None,
                ek,
                ScenePrimitive::rect(pt(30.0, y - 12.0), 10.0, 10.0).expect("finite rect"),
                false,
            );
            b.label(
                46.0,
                y,
                format!("{tag} {} -> {} [{}]", e.action, e.target, e.effect),
                ek,
            );
            y += LINE_H;
        }
    }
    let _ = &mut y;
    let mut summary = String::new();
    let _ = write!(
        summary,
        "== Summary: steps={} executed={} proposed_not_executed={} retries={} error_branch={} timeout_branch={} guard={} trace_hash={} ==",
        view.outcomes.len(),
        view.effect_count(EffectStatus::Executed),
        view.effect_count(EffectStatus::Proposed),
        view.count_status(|s| matches!(s, WorkflowStepStatus::Retrying { .. })),
        view.count_status(|s| matches!(s, WorkflowStepStatus::ErrorBranchTaken)),
        view.count_status(|s| matches!(s, WorkflowStepStatus::TimeoutBranchTaken)),
        if view.guards.is_empty() {
            "none".to_owned()
        } else {
            view.guards
                .iter()
                .map(|g| format!("t{}:{}", g.tick, g.decision))
                .collect::<Vec<_>>()
                .join(",")
        },
        view.trace_hash
    );
    b.label(10.0, y + 6.0, summary, obs);
    b.scene
}

/// Renderiza la vista a SVG determinista mediante el backend de software.
pub fn render_automation_svg(view: &AutomationView, width: f64, height: f64) -> String {
    let scene = project_automation_scene(view);
    let mut cam = Camera2D::new(width, height, false);
    if let Some(bounds) = crate::camera::scene_bounds(&scene) {
        cam.fit_to_window(width, height, &bounds, 8.0);
    }
    SoftwareSvgBackend.render_svg(&scene, &cam)
}

/// Textos (etiquetas) de la escena, en orden.
pub fn scene_texts(scene: &Scene2D) -> Vec<String> {
    crate::cognitive::scene_texts(scene)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inspector::{FieldSchema, Inspector, InspectorStatus, Lod};
    use gogamen_lab_automation::{ServiceSupervisor, SyntheticServiceEvent};

    fn texts(v: &AutomationView) -> Vec<String> {
        scene_texts(&project_automation_scene(v))
    }
    fn incident() -> AutomationView {
        run_automation("incident", &incident_scenario()).unwrap()
    }

    #[test]
    fn ac1_nodes_dependencies_and_active_transition_match_plan_and_events() {
        let v = incident();
        // plan compilado: supervisor -> actuator
        assert_eq!(v.plan_nodes.len(), 2);
        assert_eq!(v.plan_nodes[0].id, "supervisor");
        assert_eq!(v.plan_nodes[0].operation, "automation.supervisor::monitor");
        assert_eq!(v.plan_nodes[1].id, "actuator");
        assert_eq!(v.plan_nodes[1].depth, 1);
        assert_eq!(
            v.plan_edges,
            vec![PlanEdgeView {
                from: 0,
                to: 1,
                delayed: false
            }]
        );
        // estados del workflow real
        let names: Vec<_> = v.states.iter().map(|s| s.name.as_str()).collect();
        for s in [
            "Monitoring",
            "Alert",
            "Diagnose",
            "Remediate",
            "Resolved",
            "Error",
            "Timeout",
        ] {
            assert!(names.contains(&s), "{s} in {names:?}");
        }
        let depth = |n: &str| v.states.iter().find(|s| s.name == n).unwrap().depth;
        assert_eq!(
            [
                depth("Monitoring"),
                depth("Alert"),
                depth("Diagnose"),
                depth("Remediate"),
                depth("Resolved")
            ],
            [0, 1, 2, 3, 4]
        );
        // eventos del fixture: tick 2 recorre Alert/Diagnose/Remediate; tick 3 resuelve
        let path: Vec<_> = v
            .outcomes
            .iter()
            .map(|o| (o.tick, o.from.as_str(), o.to.as_str()))
            .collect();
        assert_eq!(
            path,
            vec![
                (2, "Monitoring", "Alert"),
                (2, "Alert", "Diagnose"),
                (2, "Diagnose", "Remediate"),
                (3, "Remediate", "Resolved"),
            ]
        );
        assert_eq!(v.active_state, "Resolved");
        let t = &v.transitions[v.active_transition.expect("active transition")];
        assert_eq!(
            (t.from.as_deref(), t.to.as_str(), t.kind),
            (Some("Remediate"), "Resolved", TransitionKind::Normal)
        );
        assert_eq!(t.label, "service.recovery.confirmed");
        let tx = texts(&v);
        assert!(tx.iter().any(|s| s.contains("active_state=Resolved")));
        assert!(tx.iter().any(|s| s.contains("[ACTIVE TRANSITION]")));
        assert!(
            tx.iter()
                .any(|s| s.starts_with("state Resolved") && s.contains("[ACTIVE]"))
        );
        assert_eq!(tx.iter().filter(|s| s.contains("[ACTIVE]")).count(), 1);
        assert!(tx.iter().any(|s| s == "node supervisor"));
        assert!(tx.iter().any(|s| s == "supervisor -> actuator"));
    }

    #[test]
    fn ac2_errors_retries_and_timings_shown_and_proposed_not_marked_executed() {
        let v = incident();
        // Sólo la remediación fue ejecutada por el actuador; el resto son propuestas.
        let executed: Vec<_> = v
            .outcomes
            .iter()
            .flat_map(|o| &o.effects)
            .filter(|e| e.status == EffectStatus::Executed)
            .map(|e| e.action.as_str())
            .collect();
        assert_eq!(executed, vec!["restart_worker"]);
        assert_eq!(v.effect_count(EffectStatus::Proposed), 3);
        let tx = texts(&v);
        for p in ["emit_alert", "record_diagnosis", "record_resolution"] {
            let line = tx.iter().find(|s| s.contains(p)).unwrap();
            assert!(line.starts_with("PROPOSED (not executed)"), "{line}");
        }
        assert!(
            tx.iter()
                .find(|s| s.contains("restart_worker"))
                .unwrap()
                .starts_with("EXECUTED")
        );
        // tiempos: reloj virtual 10, 20, 45 ms con delta respecto al paso previo
        let t3 = tx.iter().find(|s| s.starts_with("#3")).unwrap();
        assert!(
            t3.contains("t=45.000ms (+0.000ms)") || t3.contains("t=45.000ms"),
            "{t3}"
        );
        let t0 = tx.iter().find(|s| s.starts_with("#0")).unwrap();
        assert!(t0.contains("t=20.000ms (+20.000ms)"), "{t0}");
        assert!(t3.contains("(+25.000ms)"), "{t3}");

        // reintentos y rama OnError
        let e = run_automation("retry", &retry_error_scenario()).unwrap();
        let st: Vec<_> = e.outcomes.iter().map(|o| o.status.clone()).collect();
        assert_eq!(
            st,
            vec![
                WorkflowStepStatus::Retrying {
                    attempt: 1,
                    max_retries: 1
                },
                WorkflowStepStatus::ErrorBranchTaken
            ]
        );
        assert_eq!(e.active_state, "Error");
        // efectos del reintento/abort: propuestos, nunca ejecutados
        assert_eq!(e.effect_count(EffectStatus::Executed), 0);
        assert!(e.effect_count(EffectStatus::Proposed) >= 1);
        let tx = texts(&e);
        assert!(
            tx.iter()
                .any(|s| s.contains("RETRY 1/1") && s.contains("[ERR]"))
        );
        assert!(
            tx.iter()
                .any(|s| s.contains("ERROR-BRANCH") && s.contains("src=error:E_IO"))
        );
        assert!(tx.iter().any(|s| s.contains("(+40.000ms)")));
        assert!(tx.iter().any(|s| s.contains("retries=1 error_branch=1")));
        let at = &e.transitions[e.active_transition.unwrap()];
        assert_eq!(at.kind, TransitionKind::OnError);

        // timeout
        let t = run_automation("timeout", &timeout_scenario()).unwrap();
        assert_eq!(t.active_state, "Timeout");
        assert_eq!(t.outcomes[0].status, WorkflowStepStatus::TimeoutBranchTaken);
        assert_eq!(
            t.transitions[t.active_transition.unwrap()].kind,
            TransitionKind::OnTimeout
        );
        assert!(texts(&t).iter().any(|s| s.contains("TIMEOUT-BRANCH")));
        assert_eq!(t.effect_count(EffectStatus::Executed), 0);
    }

    #[test]
    fn proposed_never_executed_without_actuator_record_and_rejected_guard_kept() {
        // Una vista donde la guardia rechazó: efecto propuesto sigue siendo PROPOSED.
        let mut v = incident();
        for o in &mut v.outcomes {
            for e in &mut o.effects {
                e.status = EffectStatus::Proposed;
            }
        }
        v.guards = vec![GuardView {
            tick: 2,
            decision: "reject:Unauthorized".into(),
            allowed: false,
        }];
        let tx = texts(&v);
        assert!(!tx.iter().any(|s| s.starts_with("EXECUTED")));
        assert!(
            tx.iter()
                .any(|s| s.contains("executed=0 proposed_not_executed=4"))
        );
        assert!(
            tx.iter()
                .any(|s| s.contains("guard=t2:reject:Unauthorized"))
        );
        // epistemología de la escena: propuestas = Hypothetical
        let scene = project_automation_scene(&v);
        let hyp = scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter(|n| n.epistemic_kind() == EpistemicKind::Hypothetical)
            .count();
        assert!(hyp >= 8, "marker+label per proposed effect, got {hyp}");
    }

    #[test]
    fn ac3_lab_uses_same_runtime_and_trace_without_provider_or_network() {
        let sc = incident_scenario();
        let v = run_automation("a", &sc).unwrap();
        let v2 = run_automation("a", &sc).unwrap();
        assert_eq!(v, v2, "determinista");
        assert_eq!(texts(&v), texts(&v2));
        // la traza mostrada es la del supervisor real: replay con los mismos eventos
        let clock = gogamen_core::VirtualMonotonicClock::new();
        let mut script: Vec<(Tick, SyntheticServiceEvent)> = Vec::new();
        for s in &sc.script {
            if let ScriptInput::Event(e) = &s.input {
                script.push((Tick::new(s.tick), e.clone()));
            }
        }
        let rep = ServiceSupervisor::replay(sc.config.clone(), &script, &clock, None).unwrap();
        // el replay usa claves de idempotencia propias, pero la traza (tick/estado/detalle) coincide
        assert_eq!(rep.trace, v.trace);
        assert_eq!(rep.final_state.as_str(), v.active_state);
        assert!(!v.trace.is_empty());
        assert!(texts(&v).iter().any(|s| s.contains(&v.trace_hash)));
    }

    #[test]
    fn common_selection_pick_to_inspector() {
        let v = incident();
        let scene = project_automation_scene(&v);
        let snap = v.snapshot_view();
        // seleccionar por picking el nodo 'Resolved' y leerlo con el inspector genérico
        let id = STATE_BASE + v.states.iter().position(|s| s.name == "Resolved").unwrap() as u64;
        let (_, node) = scene.find_by_entity_id(EntityId::new(id))[0];
        let ScenePrimitive::Rect { min, width, height } = *node.primitive() else {
            panic!("state is a rect")
        };
        let hit = scene
            .pick_at(pt(min.x() + width / 2.0, min.y() + height / 2.0), 0.0)
            .expect("hit");
        assert_eq!(hit.entity_id, Some(EntityId::new(id)));
        let mut ins = Inspector::new(FieldSchema::new(), Lod::Full, 10);
        ins.select(EntityRef::new(hit.entity_id.unwrap().as_u64(), VIEW_EPOCH));
        let iv = ins.inspect(&snap);
        assert_eq!(iv.status, InspectorStatus::Live);
        assert_eq!(iv.kind.as_deref(), Some("workflow_state"));
        let active = iv.rows.iter().find(|r| r.name == "active").unwrap();
        assert_eq!(active.value, FieldValue::Bool(true));
        // un paso también es seleccionable y expone ejecutado/propuesto por separado
        let step = OUTCOME_BASE + 2;
        ins.select(EntityRef::new(step, VIEW_EPOCH));
        let iv = ins.inspect(&snap);
        assert_eq!(iv.kind.as_deref(), Some("step"));
        let row = |n: &str| iv.rows.iter().find(|r| r.name == n).unwrap().value.clone();
        assert_eq!(row("to"), FieldValue::Text("Remediate".into()));
        assert_eq!(row("effects_executed"), FieldValue::Integer(1));
        assert_eq!(row("effects_proposed_not_executed"), FieldValue::Integer(0));
    }

    #[test]
    fn layout_is_layered_and_deterministic() {
        let v = incident();
        let a = project_automation_scene(&v).to_canonical_json();
        assert_eq!(a, project_automation_scene(&v).to_canonical_json());
        let sv = render_automation_svg(&v, 1100.0, 700.0);
        assert!(sv.starts_with("<svg") && sv.contains("Automation graph"));
        // x crece con la profundidad topológica
        let scene = project_automation_scene(&v);
        let x_of = |name: &str| {
            let i = v.states.iter().position(|s| s.name == name).unwrap() as u64;
            let (_, n) = scene.find_by_entity_id(EntityId::new(STATE_BASE + i))[0];
            let ScenePrimitive::Rect { min, .. } = *n.primitive() else {
                unreachable!()
            };
            min.x()
        };
        assert!(x_of("Monitoring") < x_of("Alert") && x_of("Alert") < x_of("Remediate"));
        assert!(x_of("Remediate") < x_of("Resolved"));
    }

    #[test]
    fn projector_is_data_only_no_dynamics_or_planner() {
        let src = include_str!("automation.rs");
        let proj = &src[src.find("// Proyector puro").unwrap()..src.find("#[cfg(test)]").unwrap()];
        let prod: String = proj
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "std::net",
            "std::fs",
            "TcpStream",
            "Llm",
            "LLM",
            "SystemOne",
            "Planner",
            "ServiceSupervisor",
            "process_event",
            "report_error",
            "ControlPort",
            "SessionHost",
            "gogamen_lab_",
            ".step(",
            ".apply(",
            "Actuator",
        ] {
            assert!(!prod.contains(banned), "projector mentions {banned}");
        }
        assert!(prod.contains("pub fn project_automation_scene(view: &AutomationView) -> Scene2D"));
    }
}
