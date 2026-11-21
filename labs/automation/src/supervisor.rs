//! Supervisor de servicio local compuesto sobre [`StateMachineWorkflow`], [`ActionGuard`]
//! y [`SystemOneProvider`] opcional.

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, CognitiveDecisionEvent, DeterminismMetadata,
    EffectSeverity, ErrorBranch, EventKind, GuardContext, GuardDecision, IdempotencyKey,
    MasterSeed, MonotonicClock, MonotonicDeadline, OptionId, ProposedWorkflowAction, QueryId,
    ResponseStatus, RetryPolicy, SchedulerEvent, StateMachineWorkflow, SystemOneProvider, Terminal,
    Tick, TimeoutBranch, TraceHasher, TransitionGuard, TypedQuery, WorkflowError, WorkflowId,
    WorkflowInput, WorkflowSpec, WorkflowStateId, WorkflowStepError, WorkflowStepOutcome,
    WorkflowStepStatus, WorkflowTransition, WorkflowTrigger,
};
use gogamen_spec::SystemSpec;

use crate::events::{EventClassification, SyntheticServiceEvent};
use crate::spec::{AutomationError, ServiceSupervisorConfig};

/// Identificadores de estado canónicos del workflow de supervisión.
pub const STATE_MONITORING: &str = "Monitoring";
/// Estado de alerta activa.
pub const STATE_ALERT: &str = "Alert";
/// Estado de diagnóstico determinista o asistido por System One.
pub const STATE_DIAGNOSE: &str = "Diagnose";
/// Estado de remediación autorizada por [`ActionGuard`].
pub const STATE_REMEDIATE: &str = "Remediate";
/// Estado terminal de incidente resuelto.
pub const STATE_RESOLVED: &str = "Resolved";
/// Estado terminal por agotamiento de reintentos (`OnError`).
pub const STATE_ERROR: &str = "Error";
/// Estado terminal por expiración de plazo monotónico (`OnTimeout`).
pub const STATE_TIMEOUT: &str = "Timeout";

/// Código de decisión cognitiva emitida cuando System One clasifica un incidente ambiguo como alerta.
pub const COGNITIVE_DECISION_ESCALATE: &str = "escalate_alert";
/// Identificador de consulta tipada enviada a [`SystemOneProvider`].
pub const QUERY_INCIDENT_CLASSIFY: &str = "incident.classify";

/// Registro de una acción sintética ejecutada por [`FakeServiceActuator`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutedFakeAction {
    /// Tick lógico en que se ejecutó.
    pub tick: Tick,
    /// Nombre del servicio objetivo.
    pub service_name: String,
    /// Nombre lógico de la acción de remediación.
    pub action_name: String,
    /// Índice discreto autorizado.
    pub discrete_index: usize,
    /// Indica si la acción provino de una política de fallback.
    pub was_fallback: bool,
}

/// Actuador sintético local (sin red ni procesos del SO) gobernado por [`ActionGuard`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FakeServiceActuator {
    executed_actions: Vec<ExecutedFakeAction>,
    rejected_count: usize,
    noop_count: usize,
}

impl FakeServiceActuator {
    /// Crea un actuador sintético vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evalúa `proposal` mediante `guard` y ejecuta la acción fake únicamente si es autorizada.
    pub fn evaluate_and_execute(
        &mut self,
        guard: &ActionGuard,
        proposal: &ActionProposal,
        action_space: &ActionSpace,
        tick: Tick,
        service_name: &str,
        action_name: &str,
    ) -> GuardDecision {
        let ctx = GuardContext::now(action_space, tick, Terminal::Ongoing);
        let decision = guard.evaluate(proposal, &ctx);

        match &decision {
            GuardDecision::Allow(Action::Discrete(discrete)) => {
                self.executed_actions.push(ExecutedFakeAction {
                    tick,
                    service_name: service_name.to_owned(),
                    action_name: action_name.to_owned(),
                    discrete_index: discrete.index(),
                    was_fallback: false,
                });
            }
            GuardDecision::Fallback(Action::Discrete(discrete)) => {
                self.executed_actions.push(ExecutedFakeAction {
                    tick,
                    service_name: service_name.to_owned(),
                    action_name: action_name.to_owned(),
                    discrete_index: discrete.index(),
                    was_fallback: true,
                });
            }
            GuardDecision::Allow(_) | GuardDecision::Fallback(_) => {
                self.rejected_count += 1;
            }
            GuardDecision::NoOp => {
                self.noop_count += 1;
            }
            GuardDecision::Reject(_) => {
                self.rejected_count += 1;
            }
        }

        decision
    }

    /// Acciones fake ejecutadas tras autorización de [`ActionGuard`].
    pub fn executed_actions(&self) -> &[ExecutedFakeAction] {
        &self.executed_actions
    }

    /// Número de propuestas rechazadas por [`ActionGuard`].
    pub const fn rejected_count(&self) -> usize {
        self.rejected_count
    }

    /// Número de decisiones `NoOp` emitidas por [`ActionGuard`].
    pub const fn noop_count(&self) -> usize {
        self.noop_count
    }
}

/// Tipo de hito registrado en la traza determinista del supervisor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceRecordKind {
    /// Transición o evaluación del [`StateMachineWorkflow`].
    WorkflowStep,
    /// Clasificación cognitiva opcional vía [`SystemOneProvider`].
    CognitiveDecision,
    /// Decisión del [`ActionGuard`] y ejecución en [`FakeServiceActuator`].
    ActuatorDecision,
    /// Evento por debajo de umbral o ignorado.
    IgnoredEvent,
    /// Error reportado al workflow.
    ErrorReported,
}

/// Entrada inmutable de la traza determinista del supervisor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SupervisorTraceRecord {
    /// Tick lógico del evento.
    pub tick: Tick,
    /// Categoría del hito.
    pub kind: TraceRecordKind,
    /// Estado previo del workflow.
    pub from_state: String,
    /// Estado posterior del workflow.
    pub to_state: String,
    /// Detalle determinista del paso.
    pub detail: String,
}

/// Resultado estructurado de procesar un [`SyntheticServiceEvent`] en [`ServiceSupervisor`].
#[derive(Debug, Clone, PartialEq)]
pub struct SupervisorStepReport {
    /// Clasificación determinista inicial del evento.
    pub classification: EventClassification,
    /// Resultados de cada paso ejecutado en [`StateMachineWorkflow`].
    pub workflow_outcomes: Vec<WorkflowStepOutcome>,
    /// Decisión emitida por [`ActionGuard`] si se alcanzó la fase de remediación.
    pub guard_decision: Option<GuardDecision>,
    /// Evento cognitivo emitido si se consultó a [`SystemOneProvider`].
    pub cognitive_event: Option<CognitiveDecisionEvent>,
    /// Estado actual tras procesar el evento.
    pub current_state: WorkflowStateId,
}

/// Resultado de ejecutar un replay completo sobre una secuencia de eventos sintéticos.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayReport {
    /// Estado final del workflow tras el replay.
    pub final_state: WorkflowStateId,
    /// Hash determinista de la traza completa.
    pub trace_hash: String,
    /// Registros de traza generados.
    pub trace: Vec<SupervisorTraceRecord>,
    /// Acciones ejecutadas por el actuador fake.
    pub executed_actions: Vec<ExecutedFakeAction>,
    /// Eventos cognitivos emitidos durante el replay.
    pub cognitive_events: Vec<CognitiveDecisionEvent>,
}

/// Supervisor de servicio local sintético compuesto sobre [`StateMachineWorkflow`].
///
/// Soporta:
/// - **Modo sin IA (determinista puro):** procesa alertas, diagnóstico, remediación y
///   resolución con replay exacto sin ningún [`SystemOneProvider`].
/// - **Modo cognitivo opcional:** consulta un [`SystemOneProvider`] ante eventos ambiguos
///   y emite [`CognitiveDecisionEvent`] directamente al [`StateMachineWorkflow`] sin
///   modificar sus contratos core.
/// - **Extensibilidad por Spec:** construible desde [`SystemSpec`] o JSON.
#[derive(Debug, Clone)]
pub struct ServiceSupervisor {
    config: ServiceSupervisorConfig,
    workflow: StateMachineWorkflow,
    guard: ActionGuard,
    actuator: FakeServiceActuator,
    action_space: ActionSpace,
    trace: Vec<SupervisorTraceRecord>,
    trace_hasher: TraceHasher,
    cognitive_events: Vec<CognitiveDecisionEvent>,
}

impl ServiceSupervisor {
    /// Construye un supervisor a partir de una [`ServiceSupervisorConfig`] validada.
    pub fn new(config: ServiceSupervisorConfig) -> Result<Self, AutomationError> {
        config.validate()?;
        let workflow_spec = build_workflow_spec(&config)?;
        let workflow = workflow_spec.compile()?;
        let action_space = ActionSpace::discrete(config.action_space_size)?;
        let metadata = DeterminismMetadata::current(MasterSeed::new(0xA070_4E51));
        let mut trace_hasher = TraceHasher::with_metadata(&metadata);
        trace_hasher.write_str(&config.service_name);
        trace_hasher.write_str(config.remediation_action.action_name());
        trace_hasher.write_u64(u64::from(config.thresholds.max_failed_checks));
        trace_hasher.write_u64(config.thresholds.latency_threshold_ms);
        trace_hasher.write_u64(config.thresholds.queue_backlog_threshold);

        Ok(Self {
            config,
            workflow,
            guard: ActionGuard::strict(),
            actuator: FakeServiceActuator::new(),
            action_space,
            trace: Vec::new(),
            trace_hasher,
            cognitive_events: Vec::new(),
        })
    }

    /// Construye un supervisor directamente desde una [`SystemSpec`] declarativa.
    pub fn from_system_spec(spec: &SystemSpec) -> Result<Self, AutomationError> {
        let config = ServiceSupervisorConfig::from_system_spec(spec)?;
        Self::new(config)
    }

    /// Construye un supervisor desde un documento JSON de [`SystemSpec`].
    pub fn from_spec_json(json: &str) -> Result<Self, AutomationError> {
        let config = ServiceSupervisorConfig::from_spec_json(json)?;
        Self::new(config)
    }

    /// Configuración activa del supervisor.
    pub const fn config(&self) -> &ServiceSupervisorConfig {
        &self.config
    }

    /// Definición inmutable del workflow (estados, transiciones y ramas `OnError`/`OnTimeout`).
    pub fn workflow_spec(&self) -> &WorkflowSpec {
        self.workflow.spec()
    }

    /// Estado actual del [`StateMachineWorkflow`].
    pub fn current_state(&self) -> &WorkflowStateId {
        self.workflow.current_state()
    }

    /// Indica si el workflow ha alcanzado un estado terminal (`Resolved`, `Error`, `Timeout`).
    pub fn is_terminal(&self) -> bool {
        self.workflow.is_terminal()
    }

    /// Configura o limpia el plazo monotónico (`OnTimeout`) del workflow.
    pub fn set_deadline(&mut self, deadline: Option<MonotonicDeadline>) {
        self.workflow.set_deadline(deadline);
    }

    /// Acceso de solo lectura al actuador sintético y sus acciones ejecutadas.
    pub const fn actuator(&self) -> &FakeServiceActuator {
        &self.actuator
    }

    /// Traza determinista acumulada.
    pub fn trace(&self) -> &[SupervisorTraceRecord] {
        &self.trace
    }

    /// Eventos cognitivos emitidos por [`SystemOneProvider`] durante la ejecución.
    pub fn cognitive_events(&self) -> &[CognitiveDecisionEvent] {
        &self.cognitive_events
    }

    /// Huella determinista hexadecimal de toda la traza acumulada.
    pub fn trace_hash(&self) -> String {
        self.trace_hasher.finish_hex()
    }

    /// Procesa un [`SyntheticServiceEvent`] en `tick` usando `clock` y un proveedor cognitivo opcional.
    pub fn process_event(
        &mut self,
        event: &SyntheticServiceEvent,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        clock: &dyn MonotonicClock,
        provider: Option<&dyn SystemOneProvider>,
    ) -> Result<SupervisorStepReport, AutomationError> {
        let classification = self.config.thresholds.classify(event);
        let mut workflow_outcomes = Vec::new();
        let mut guard_decision = None;
        let mut cognitive_event = None;

        match &classification {
            EventClassification::BelowThreshold => {
                if self
                    .workflow
                    .deadline()
                    .is_some_and(|deadline| deadline.is_expired(clock))
                {
                    let timeout_probe =
                        SchedulerEvent::signal(EventKind::new("service.heartbeat")?);
                    let outcome = self.step_and_record(
                        timeout_probe,
                        idempotency_key,
                        tick,
                        clock,
                        "timeout_probe",
                    )?;
                    workflow_outcomes.push(outcome);
                } else {
                    self.record_trace_current(
                        tick,
                        TraceRecordKind::IgnoredEvent,
                        format!("below_threshold:{event}"),
                    );
                }
            }
            EventClassification::Breach { kind, detail } => {
                if self.workflow.current_state().as_str() == STATE_MONITORING {
                    let alert_input = SchedulerEvent::new(
                        EventKind::new("service.alert.triggered")?,
                        format!("{kind}:{detail}"),
                    );
                    let alert_key = derive_sub_key(idempotency_key.as_ref(), "alert")?;
                    let alert_outcome =
                        self.step_and_record(alert_input, alert_key, tick, clock, "alert")?;
                    let alerted = alert_outcome.status == WorkflowStepStatus::Transitioned;
                    workflow_outcomes.push(alert_outcome);

                    if alerted {
                        guard_decision = self.advance_diagnose_and_remediate(
                            kind,
                            detail,
                            idempotency_key.as_ref(),
                            tick,
                            clock,
                            &mut workflow_outcomes,
                        )?;
                    }
                } else {
                    let probe = SchedulerEvent::new(
                        EventKind::new("service.alert.triggered")?,
                        format!("{kind}:{detail}"),
                    );
                    let outcome = self.step_and_record(
                        probe,
                        idempotency_key,
                        tick,
                        clock,
                        "breach_non_monitoring",
                    )?;
                    workflow_outcomes.push(outcome);
                }
            }
            EventClassification::Ambiguous { kind, detail } => {
                if self.workflow.current_state().as_str() == STATE_MONITORING {
                    if let Some(sys_one) = provider {
                        let decision = self
                            .classify_ambiguous_with_provider(sys_one, kind, detail, tick, clock)?;
                        if let Some(cog_event) = decision {
                            cognitive_event = Some(cog_event.clone());
                            self.cognitive_events.push(cog_event.clone());
                            let cog_key = derive_sub_key(idempotency_key.as_ref(), "cognitive")?;
                            let alert_outcome = self.step_and_record(
                                cog_event,
                                cog_key,
                                tick,
                                clock,
                                "cognitive_alert",
                            )?;
                            let alerted = alert_outcome.status == WorkflowStepStatus::Transitioned;
                            workflow_outcomes.push(alert_outcome);

                            if alerted {
                                guard_decision = self.advance_diagnose_and_remediate(
                                    kind,
                                    detail,
                                    idempotency_key.as_ref(),
                                    tick,
                                    clock,
                                    &mut workflow_outcomes,
                                )?;
                            }
                        } else {
                            self.record_trace_current(
                                tick,
                                TraceRecordKind::IgnoredEvent,
                                format!("cognitive_noise_ignored:{kind}:{detail}"),
                            );
                        }
                    } else {
                        self.record_trace_current(
                            tick,
                            TraceRecordKind::IgnoredEvent,
                            format!("ambiguous_without_provider:{kind}:{detail}"),
                        );
                    }
                }
            }
            EventClassification::ProbeReady { detail }
            | EventClassification::RecoveryReady { detail } => {
                let recovery_input = SchedulerEvent::new(
                    EventKind::new("service.recovery.confirmed")?,
                    detail.clone(),
                );
                let outcome =
                    self.step_and_record(recovery_input, idempotency_key, tick, clock, "recovery")?;
                workflow_outcomes.push(outcome);
            }
        }

        Ok(SupervisorStepReport {
            classification,
            workflow_outcomes,
            guard_decision,
            cognitive_event,
            current_state: self.workflow.current_state().clone(),
        })
    }

    /// Reporta un error de paso al [`StateMachineWorkflow`] respetando [`RetryPolicy`] y `OnError`.
    pub fn report_error(
        &mut self,
        error: &WorkflowStepError,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        clock: &dyn MonotonicClock,
    ) -> Result<WorkflowStepOutcome, AutomationError> {
        let outcome = self
            .workflow
            .report_error(error, idempotency_key, tick, clock)?;
        self.record_trace(
            tick,
            TraceRecordKind::ErrorReported,
            outcome.previous_state.as_str(),
            outcome.current_state.as_str(),
            format!("{}:{:?}", error.code, outcome.status),
        );
        Ok(outcome)
    }

    /// Ejecuta un replay determinista completo sobre una secuencia de eventos sintéticos.
    pub fn replay(
        config: ServiceSupervisorConfig,
        script: &[(Tick, SyntheticServiceEvent)],
        clock: &dyn MonotonicClock,
        provider: Option<&dyn SystemOneProvider>,
    ) -> Result<ReplayReport, AutomationError> {
        let mut supervisor = Self::new(config)?;
        for (idx, (tick, event)) in script.iter().enumerate() {
            let key = IdempotencyKey::new(format!("replay-{idx}-t{}", tick.as_u64()))?;
            supervisor.process_event(event, Some(key), *tick, clock, provider)?;
        }
        Ok(ReplayReport {
            final_state: supervisor.current_state().clone(),
            trace_hash: supervisor.trace_hash(),
            trace: supervisor.trace,
            executed_actions: supervisor.actuator.executed_actions().to_vec(),
            cognitive_events: supervisor.cognitive_events,
        })
    }

    fn advance_diagnose_and_remediate(
        &mut self,
        kind: &str,
        detail: &str,
        base_key: Option<&IdempotencyKey>,
        tick: Tick,
        clock: &dyn MonotonicClock,
        workflow_outcomes: &mut Vec<WorkflowStepOutcome>,
    ) -> Result<Option<GuardDecision>, AutomationError> {
        let diagnose_input = SchedulerEvent::new(
            EventKind::new("service.diagnose.requested")?,
            format!("diagnose:{kind}:{detail}"),
        );
        let diagnose_key = derive_sub_key(base_key, "diagnose")?;
        let diagnose_outcome =
            self.step_and_record(diagnose_input, diagnose_key, tick, clock, "diagnose")?;
        let diagnosed = diagnose_outcome.status == WorkflowStepStatus::Transitioned;
        workflow_outcomes.push(diagnose_outcome);

        if !diagnosed {
            return Ok(None);
        }

        let action_name = self.config.remediation_action.action_name().to_owned();
        let remediate_input = SchedulerEvent::new(
            EventKind::new("service.remediate.requested")?,
            format!("remediate:{action_name}"),
        );
        let remediate_key = derive_sub_key(base_key, "remediate")?;
        let remediate_outcome =
            self.step_and_record(remediate_input, remediate_key, tick, clock, "remediate")?;
        let remediated = remediate_outcome.status == WorkflowStepStatus::Transitioned;
        workflow_outcomes.push(remediate_outcome);

        if !remediated {
            return Ok(None);
        }

        let proposal = ActionProposal::authorized(
            Action::Discrete(
                self.config
                    .remediation_action
                    .discrete_action(self.config.action_space_size)?,
            ),
            tick,
        );
        let decision = self.actuator.evaluate_and_execute(
            &self.guard,
            &proposal,
            &self.action_space,
            tick,
            &self.config.service_name,
            &action_name,
        );
        self.record_trace_current(
            tick,
            TraceRecordKind::ActuatorDecision,
            format!("action={action_name};allowed={}", decision.is_allowed()),
        );
        Ok(Some(decision))
    }

    fn classify_ambiguous_with_provider(
        &mut self,
        provider: &dyn SystemOneProvider,
        kind: &str,
        detail: &str,
        tick: Tick,
        clock: &dyn MonotonicClock,
    ) -> Result<Option<CognitiveDecisionEvent>, AutomationError> {
        let escalate_opt = OptionId::new(COGNITIVE_DECISION_ESCALATE)?;
        let noise_opt = OptionId::new("transient_noise")?;
        let query = TypedQuery::choice(
            QueryId::new(QUERY_INCIDENT_CLASSIFY)?,
            vec![escalate_opt.clone(), noise_opt],
            None,
        )?;

        let response = provider.query(&query, clock)?;
        let selected = if response.status() == ResponseStatus::Answered {
            response.selected_option().cloned()
        } else {
            None
        };

        self.record_trace_current(
            tick,
            TraceRecordKind::CognitiveDecision,
            format!(
                "provider={};status={:?};selected={}",
                provider.provider_id(),
                response.status(),
                selected.as_ref().map_or("none", |opt| opt.as_str())
            ),
        );

        if selected.as_ref() == Some(&escalate_opt) {
            let cog = CognitiveDecisionEvent::new(
                COGNITIVE_DECISION_ESCALATE,
                format!("{kind}:{detail}"),
                tick,
            )?;
            Ok(Some(cog))
        } else {
            Ok(None)
        }
    }

    fn step_and_record(
        &mut self,
        input: impl Into<WorkflowInput>,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        clock: &dyn MonotonicClock,
        label: &str,
    ) -> Result<WorkflowStepOutcome, AutomationError> {
        let outcome = self.workflow.step(input, idempotency_key, tick, clock)?;
        self.record_trace(
            tick,
            TraceRecordKind::WorkflowStep,
            outcome.previous_state.as_str(),
            outcome.current_state.as_str(),
            format!("{label}:{:?}", outcome.status),
        );
        Ok(outcome)
    }

    fn record_trace_current(&mut self, tick: Tick, kind: TraceRecordKind, detail: String) {
        let state = self.workflow.current_state().as_str().to_owned();
        self.record_trace(tick, kind, &state, &state, detail);
    }

    fn record_trace(
        &mut self,
        tick: Tick,
        kind: TraceRecordKind,
        from_state: &str,
        to_state: &str,
        detail: String,
    ) {
        self.trace_hasher.write_tick(tick);
        self.trace_hasher.write_u8(kind as u8);
        self.trace_hasher.write_str(from_state);
        self.trace_hasher.write_str(to_state);
        self.trace_hasher.write_str(&detail);
        self.trace.push(SupervisorTraceRecord {
            tick,
            kind,
            from_state: from_state.to_owned(),
            to_state: to_state.to_owned(),
            detail,
        });
    }
}

fn derive_sub_key(
    base: Option<&IdempotencyKey>,
    suffix: &str,
) -> Result<Option<IdempotencyKey>, WorkflowError> {
    match base {
        Some(key) => Ok(Some(IdempotencyKey::new(format!(
            "{}:{suffix}",
            key.as_str()
        ))?)),
        None => Ok(None),
    }
}

fn build_workflow_spec(config: &ServiceSupervisorConfig) -> Result<WorkflowSpec, AutomationError> {
    let monitoring = WorkflowStateId::new(STATE_MONITORING)?;
    let alert = WorkflowStateId::new(STATE_ALERT)?;
    let diagnose = WorkflowStateId::new(STATE_DIAGNOSE)?;
    let remediate = WorkflowStateId::new(STATE_REMEDIATE)?;
    let resolved = WorkflowStateId::new(STATE_RESOLVED)?;
    let error_state = WorkflowStateId::new(STATE_ERROR)?;
    let timeout_state = WorkflowStateId::new(STATE_TIMEOUT)?;

    let mut spec = WorkflowSpec::new(
        WorkflowId::new(format!("service-supervisor-{}", config.service_name))?,
        monitoring.clone(),
        vec![resolved.clone(), error_state.clone(), timeout_state.clone()],
    );
    spec.default_retry_policy = RetryPolicy::new(config.max_retries);

    let alert_event = EventKind::new("service.alert.triggered")?;
    let diagnose_event = EventKind::new("service.diagnose.requested")?;
    let remediate_event = EventKind::new("service.remediate.requested")?;
    let recovery_event = EventKind::new("service.recovery.confirmed")?;

    let action_name = config.remediation_action.action_name();
    let target = config.service_name.as_str();

    spec.transitions = vec![
        WorkflowTransition::new(
            monitoring.clone(),
            alert.clone(),
            TransitionGuard::Trigger(WorkflowTrigger::Event(alert_event)),
            vec![ProposedWorkflowAction::emit_event(
                "emit_alert",
                target,
                "service.alert",
                "alert_triggered",
            )?],
        ),
        WorkflowTransition::new(
            monitoring,
            alert.clone(),
            TransitionGuard::Trigger(WorkflowTrigger::cognitive_decision(
                COGNITIVE_DECISION_ESCALATE,
            )?),
            vec![ProposedWorkflowAction::emit_event(
                "emit_cognitive_alert",
                target,
                "service.alert",
                "cognitive_escalation",
            )?],
        ),
        WorkflowTransition::new(
            alert,
            diagnose.clone(),
            TransitionGuard::Trigger(WorkflowTrigger::Event(diagnose_event)),
            vec![ProposedWorkflowAction::log(
                "record_diagnosis",
                target,
                EffectSeverity::Warn,
                "diagnosis_started",
            )?],
        ),
        WorkflowTransition::new(
            diagnose,
            remediate.clone(),
            TransitionGuard::Trigger(WorkflowTrigger::Event(remediate_event)),
            vec![ProposedWorkflowAction::emit_event(
                action_name,
                target,
                "service.remediate",
                action_name,
            )?],
        ),
        WorkflowTransition::new(
            remediate,
            resolved,
            TransitionGuard::Trigger(WorkflowTrigger::Event(recovery_event)),
            vec![ProposedWorkflowAction::log(
                "record_resolution",
                target,
                EffectSeverity::Info,
                "incident_resolved",
            )?],
        ),
    ];

    spec.error_branches = vec![ErrorBranch {
        from: None,
        retry_policy: RetryPolicy::new(config.max_retries),
        retry_actions: vec![ProposedWorkflowAction::log(
            "retry_step",
            target,
            EffectSeverity::Warn,
            "retrying_step",
        )?],
        on_exhausted_state: error_state,
        on_exhausted_actions: vec![ProposedWorkflowAction::log(
            "abort_on_error",
            target,
            EffectSeverity::Error,
            "retries_exhausted",
        )?],
    }];

    spec.timeout_branches = vec![TimeoutBranch {
        from: None,
        to: timeout_state,
        actions: vec![ProposedWorkflowAction::log(
            "abort_on_timeout",
            target,
            EffectSeverity::Error,
            "deadline_expired",
        )?],
    }];

    spec.validate()?;
    Ok(spec)
}
