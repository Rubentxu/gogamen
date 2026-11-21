//! Motor puro de automatización por eventos y máquinas de estado tipadas (`std`-only).
//!
//! Este módulo implementa un motor determinista de flujos de trabajo y máquinas de estado
//! sin requerir [`crate::Planner`], proveedores externos ni ejecución de IO:
//! - [`WorkflowSpec`] y [`StateMachineWorkflow`]: definición validable y máquina de estados
//!   tipada con estados explícitos ([`WorkflowStateId`]) y transiciones con guardas
//!   ([`TransitionGuard`]).
//! - [`ProposedWorkflowAction`]: acciones propuestas puras basadas en [`ProposedEffect`],
//!   sin efectos secundarios directos.
//! - Ramas explícitas [`ErrorBranch`] (`OnError`) y [`TimeoutBranch`] (`OnTimeout`)
//!   respaldadas por [`MonotonicDeadline`] y [`MonotonicClock`], junto con [`RetryPolicy`]
//!   acotado (`max_retries`).
//! - Idempotencia acotada mediante [`IdempotencyKey`] para evitar ejecuciones duplicadas
//!   ante reentrega de eventos.
//! - Integración nativa con [`SchedulerEvent`] y [`CognitiveDecisionEvent`] mediante
//!   [`WorkflowTrigger`] y [`WorkflowInput`], sin puentes ad hoc.

use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::clock::{MonotonicClock, MonotonicDeadline, MonotonicInstant};
use crate::scheduler::{EventKind, SchedulerEvent};
use crate::system::{EffectSeverity, ProposedEffect};
use crate::units::Tick;

/// Identificador no vacío de una especificación de workflow.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkflowId(String);

impl WorkflowId {
    /// Crea un [`WorkflowId`] a partir de una cadena no vacía.
    pub fn new(id: impl Into<String>) -> Result<Self, WorkflowError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "workflow id must not be empty",
            });
        }
        Ok(Self(id))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador explícito y no vacío de un estado dentro de [`StateMachineWorkflow`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WorkflowStateId(String);

impl WorkflowStateId {
    /// Crea un [`WorkflowStateId`] a partir de una cadena no vacía.
    pub fn new(id: impl Into<String>) -> Result<Self, WorkflowError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "workflow state id must not be empty",
            });
        }
        Ok(Self(id))
    }

    /// Devuelve el nombre del estado.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for WorkflowStateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Clave no vacía de idempotencia para deduplicar entradas repetidas en un workflow.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IdempotencyKey(String);

impl IdempotencyKey {
    /// Crea una [`IdempotencyKey`] a partir de una cadena no vacía.
    pub fn new(key: impl Into<String>) -> Result<Self, WorkflowError> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(WorkflowError::InvalidInput {
                reason: "idempotency key must not be empty",
            });
        }
        Ok(Self(key))
    }

    /// Devuelve el valor textual de la clave.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IdempotencyKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Evento nativo de decisión cognitiva que puede activar directamente un workflow.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CognitiveDecisionEvent {
    decision_code: String,
    payload: String,
    tick: Tick,
}

impl CognitiveDecisionEvent {
    /// Crea un evento de decisión cognitiva con código no vacío, carga útil y [`Tick`].
    pub fn new(
        decision_code: impl Into<String>,
        payload: impl Into<String>,
        tick: Tick,
    ) -> Result<Self, WorkflowError> {
        let decision_code = decision_code.into();
        if decision_code.trim().is_empty() {
            return Err(WorkflowError::InvalidInput {
                reason: "cognitive decision code must not be empty",
            });
        }
        Ok(Self {
            decision_code,
            payload: payload.into(),
            tick,
        })
    }

    /// Código de decisión cognitiva.
    pub fn decision_code(&self) -> &str {
        &self.decision_code
    }

    /// Carga útil asociada a la decisión.
    pub fn payload(&self) -> &str {
        &self.payload
    }

    /// Tick lógico en el que se emitió la decisión.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Convierte la decisión en un [`SchedulerEvent`] con prefijo `cognitive.decision.`.
    pub fn to_scheduler_event(&self) -> Result<SchedulerEvent, WorkflowError> {
        let kind =
            EventKind::new(format!("cognitive.decision.{}", self.decision_code)).map_err(|_| {
                WorkflowError::InvalidInput {
                    reason: "failed to construct cognitive decision EventKind",
                }
            })?;
        Ok(SchedulerEvent::new(kind, self.payload.clone()))
    }
}

/// Entrada tipada admitida directamente por [`StateMachineWorkflow`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkflowInput {
    /// Evento proveniente de [`crate::LocalScheduler`].
    Event(SchedulerEvent),
    /// Evento nativo de decisión cognitiva sin puente ad hoc.
    CognitiveDecision(CognitiveDecisionEvent),
}

impl WorkflowInput {
    /// Devuelve la carga útil textual de la entrada.
    pub fn payload(&self) -> &str {
        match self {
            Self::Event(ev) => ev.payload(),
            Self::CognitiveDecision(dec) => dec.payload(),
        }
    }
}

impl From<SchedulerEvent> for WorkflowInput {
    fn from(value: SchedulerEvent) -> Self {
        Self::Event(value)
    }
}

impl From<CognitiveDecisionEvent> for WorkflowInput {
    fn from(value: CognitiveDecisionEvent) -> Self {
        Self::CognitiveDecision(value)
    }
}

/// Condición de disparo para transiciones activadas por eventos o decisiones cognitivas.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WorkflowTrigger {
    /// Disparo por [`EventKind`] de [`SchedulerEvent`].
    Event(EventKind),
    /// Disparo directo por código de decisión cognitiva (y carga útil opcional).
    CognitiveDecision {
        /// Código de decisión esperado.
        decision_code: String,
        /// Carga útil exacta requerida (si es `Some`).
        required_payload: Option<String>,
    },
}

impl WorkflowTrigger {
    /// Construye un trigger para una decisión cognitiva sin restricción de payload.
    pub fn cognitive_decision(decision_code: impl Into<String>) -> Result<Self, WorkflowError> {
        let decision_code = decision_code.into();
        if decision_code.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "cognitive decision trigger code must not be empty",
            });
        }
        Ok(Self::CognitiveDecision {
            decision_code,
            required_payload: None,
        })
    }

    /// Construye un trigger para una decisión cognitiva con payload exacto requerido.
    pub fn cognitive_decision_with_payload(
        decision_code: impl Into<String>,
        required_payload: impl Into<String>,
    ) -> Result<Self, WorkflowError> {
        let decision_code = decision_code.into();
        if decision_code.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "cognitive decision trigger code must not be empty",
            });
        }
        Ok(Self::CognitiveDecision {
            decision_code,
            required_payload: Some(required_payload.into()),
        })
    }

    /// Evalúa si la entrada [`WorkflowInput`] coincide con el trigger.
    pub fn matches(&self, input: &WorkflowInput) -> bool {
        match (self, input) {
            (Self::Event(expected), WorkflowInput::Event(actual)) => actual.kind() == expected,
            (
                Self::CognitiveDecision {
                    decision_code,
                    required_payload,
                },
                WorkflowInput::CognitiveDecision(actual),
            ) => {
                actual.decision_code() == decision_code
                    && required_payload
                        .as_deref()
                        .is_none_or(|req| actual.payload() == req)
            }
            (
                Self::CognitiveDecision {
                    decision_code,
                    required_payload,
                },
                WorkflowInput::Event(actual),
            ) => {
                let expected_kind = format!("cognitive.decision.{decision_code}");
                actual.kind().as_str() == expected_kind
                    && required_payload
                        .as_deref()
                        .is_none_or(|req| actual.payload() == req)
            }
            (Self::Event(_), WorkflowInput::CognitiveDecision(_)) => false,
        }
    }
}

/// Política explícita y acotada de reintentos ante errores de paso.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct RetryPolicy {
    /// Número máximo de reintentos permitidos antes de tomar la rama `OnError`.
    pub max_retries: u32,
}

impl RetryPolicy {
    /// Política sin reintentos (`max_retries = 0`).
    pub const ZERO: Self = Self { max_retries: 0 };

    /// Crea una política con `max_retries` reintentos máximos.
    pub const fn new(max_retries: u32) -> Self {
        Self { max_retries }
    }

    /// Indica si aún se permite otro reintento dado el contador actual `current_retries`.
    pub const fn can_retry(&self, current_retries: u32) -> bool {
        current_retries < self.max_retries
    }
}

/// Contexto inmutable disponible durante la evaluación de una [`TransitionGuard`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowContext<'a> {
    /// Estado actual de la máquina.
    pub current_state: &'a WorkflowStateId,
    /// Tick lógico actual.
    pub tick: Tick,
    /// Instante monotónico actual.
    pub now: MonotonicInstant,
    /// Número de reintentos acumulados en la ejecución actual.
    pub retry_count: u32,
    /// Entrada que originó la evaluación.
    pub input: &'a WorkflowInput,
}

/// Guarda tipada y pura que condiciona una [`WorkflowTransition`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TransitionGuard {
    /// Siempre verdadera.
    Always,
    /// Verdadera si la entrada coincide con [`WorkflowTrigger`].
    Trigger(WorkflowTrigger),
    /// Verdadera si el payload de la entrada es exactamente igual al esperado.
    PayloadEquals(String),
    /// Verdadera si el payload de la entrada contiene la subcadena esperada.
    PayloadContains(String),
    /// Verdadera si el número de reintentos actual es estrictamente menor que `max_retries`.
    WithinRetryBudget,
    /// Conjunción lógica de todas las guardas contenidas.
    All(Vec<TransitionGuard>),
    /// Disyunción lógica de las guardas contenidas.
    Any(Vec<TransitionGuard>),
}

impl TransitionGuard {
    /// Evalúa la guarda de forma pura y determinista.
    pub fn evaluate(&self, ctx: &WorkflowContext<'_>, retry_policy: RetryPolicy) -> bool {
        match self {
            Self::Always => true,
            Self::Trigger(trigger) => trigger.matches(ctx.input),
            Self::PayloadEquals(expected) => ctx.input.payload() == expected,
            Self::PayloadContains(substr) => ctx.input.payload().contains(substr.as_str()),
            Self::WithinRetryBudget => retry_policy.can_retry(ctx.retry_count),
            Self::All(guards) => guards.iter().all(|guard| guard.evaluate(ctx, retry_policy)),
            Self::Any(guards) => guards.iter().any(|guard| guard.evaluate(ctx, retry_policy)),
        }
    }
}

/// Acción propuesta pura emitida por una transición del workflow sin ejecutar IO.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProposedWorkflowAction {
    action_name: String,
    target: String,
    effect: ProposedEffect,
}

impl ProposedWorkflowAction {
    /// Crea una acción propuesta pura con nombre, destino y [`ProposedEffect`].
    pub fn new(
        action_name: impl Into<String>,
        target: impl Into<String>,
        effect: ProposedEffect,
    ) -> Result<Self, WorkflowError> {
        let action_name = action_name.into();
        if action_name.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "proposed workflow action name must not be empty",
            });
        }
        let target = target.into();
        if target.trim().is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "proposed workflow action target must not be empty",
            });
        }
        Ok(Self {
            action_name,
            target,
            effect,
        })
    }

    /// Crea una acción propuesta que emite un [`ProposedEffect::EmitEvent`].
    pub fn emit_event(
        action_name: impl Into<String>,
        target: impl Into<String>,
        topic: impl Into<String>,
        payload: impl Into<String>,
    ) -> Result<Self, WorkflowError> {
        Self::new(
            action_name,
            target,
            ProposedEffect::emit_event(topic, payload),
        )
    }

    /// Crea una acción propuesta que registra un [`ProposedEffect::Log`].
    pub fn log(
        action_name: impl Into<String>,
        target: impl Into<String>,
        level: EffectSeverity,
        message: impl Into<String>,
    ) -> Result<Self, WorkflowError> {
        Self::new(action_name, target, ProposedEffect::log(level, message))
    }

    /// Nombre lógico de la acción propuesta.
    pub fn action_name(&self) -> &str {
        &self.action_name
    }

    /// Destino lógico de la acción propuesta.
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Efecto puro asociado a la acción.
    pub fn effect(&self) -> &ProposedEffect {
        &self.effect
    }
}

/// Transición tipada entre dos estados del workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowTransition {
    /// Estado origen requerido.
    pub from: WorkflowStateId,
    /// Estado destino al cumplirse la guarda.
    pub to: WorkflowStateId,
    /// Condición pura que habilita la transición.
    pub guard: TransitionGuard,
    /// Acciones propuestas puras emitidas al tomar la transición.
    pub actions: Vec<ProposedWorkflowAction>,
}

impl WorkflowTransition {
    /// Crea una transición tipada entre `from` y `to`.
    pub fn new(
        from: WorkflowStateId,
        to: WorkflowStateId,
        guard: TransitionGuard,
        actions: Vec<ProposedWorkflowAction>,
    ) -> Self {
        Self {
            from,
            to,
            guard,
            actions,
        }
    }
}

/// Rama explícita `OnError` con política de reintento acotado y estado de fallo/recuperación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorBranch {
    /// Estado origen al que aplica (o `None` para cualquier estado no terminal).
    pub from: Option<WorkflowStateId>,
    /// Política de reintento acotado para esta rama.
    pub retry_policy: RetryPolicy,
    /// Acciones propuestas durante cada intento de reintento.
    pub retry_actions: Vec<ProposedWorkflowAction>,
    /// Estado destino cuando se agotan los reintentos.
    pub on_exhausted_state: WorkflowStateId,
    /// Acciones propuestas al agotar los reintentos y tomar la rama `OnError`.
    pub on_exhausted_actions: Vec<ProposedWorkflowAction>,
}

/// Rama explícita `OnTimeout` activada cuando expira un [`MonotonicDeadline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimeoutBranch {
    /// Estado origen al que aplica (o `None` para cualquier estado no terminal).
    pub from: Option<WorkflowStateId>,
    /// Estado destino al expirar el plazo monotónico.
    pub to: WorkflowStateId,
    /// Acciones propuestas al activar la rama `OnTimeout`.
    pub actions: Vec<ProposedWorkflowAction>,
}

/// Error tipado reportado durante un paso de workflow que activa [`ErrorBranch`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WorkflowStepError {
    /// Código corto del fallo.
    pub code: String,
    /// Descripción legible del fallo.
    pub detail: String,
}

impl WorkflowStepError {
    /// Crea un error de paso con código no vacío y detalle.
    pub fn new(code: impl Into<String>, detail: impl Into<String>) -> Result<Self, WorkflowError> {
        let code = code.into();
        if code.trim().is_empty() {
            return Err(WorkflowError::InvalidInput {
                reason: "workflow step error code must not be empty",
            });
        }
        Ok(Self {
            code,
            detail: detail.into(),
        })
    }
}

impl fmt::Display for WorkflowStepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code, self.detail)
    }
}

/// Error de validación o ejecución del motor de workflows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowError {
    /// La especificación del workflow es inválida.
    InvalidSpec {
        /// Motivo de invalidez.
        reason: &'static str,
    },
    /// Una entrada o clave proporcionada es inválida.
    InvalidInput {
        /// Motivo de invalidez.
        reason: &'static str,
    },
    /// No existe una rama `OnError` aplicable para el estado actual.
    MissingErrorBranch {
        /// Estado en el que ocurrió el error.
        state: WorkflowStateId,
    },
    /// No existe una rama `OnTimeout` aplicable para el estado actual.
    MissingTimeoutBranch {
        /// Estado en el que expiró el deadline.
        state: WorkflowStateId,
    },
}

impl fmt::Display for WorkflowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpec { reason } => write!(f, "invalid workflow spec: {reason}"),
            Self::InvalidInput { reason } => write!(f, "invalid workflow input: {reason}"),
            Self::MissingErrorBranch { state } => {
                write!(f, "no OnError branch configured for state '{state}'")
            }
            Self::MissingTimeoutBranch { state } => {
                write!(f, "no OnTimeout branch configured for state '{state}'")
            }
        }
    }
}

impl Error for WorkflowError {}

/// Especificación declarativa y validable de una máquina de estados de workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowSpec {
    /// Identificador del workflow.
    pub id: WorkflowId,
    /// Estado inicial de la máquina.
    pub initial_state: WorkflowStateId,
    /// Conjunto de estados terminales explícitos.
    pub terminal_states: Vec<WorkflowStateId>,
    /// Transiciones normales evaluadas en orden de declaración.
    pub transitions: Vec<WorkflowTransition>,
    /// Ramas explícitas `OnError`.
    pub error_branches: Vec<ErrorBranch>,
    /// Ramas explícitas `OnTimeout`.
    pub timeout_branches: Vec<TimeoutBranch>,
    /// Política de reintento por defecto para evaluaciones de guardas.
    pub default_retry_policy: RetryPolicy,
    /// Capacidad acotada de claves de idempotencia recordadas.
    pub idempotency_capacity: usize,
}

impl WorkflowSpec {
    /// Crea una especificación básica con capacidad de idempotencia acotada por defecto (`256`).
    pub fn new(
        id: WorkflowId,
        initial_state: WorkflowStateId,
        terminal_states: Vec<WorkflowStateId>,
    ) -> Self {
        Self {
            id,
            initial_state,
            terminal_states,
            transitions: Vec::new(),
            error_branches: Vec::new(),
            timeout_branches: Vec::new(),
            default_retry_policy: RetryPolicy::ZERO,
            idempotency_capacity: 256,
        }
    }

    /// Valida invariantes estructurales de la especificación.
    pub fn validate(&self) -> Result<(), WorkflowError> {
        if self.terminal_states.is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "workflow must declare at least one terminal state",
            });
        }
        if self.idempotency_capacity == 0 {
            return Err(WorkflowError::InvalidSpec {
                reason: "idempotency_capacity must be greater than zero",
            });
        }
        if self.transitions.is_empty() {
            return Err(WorkflowError::InvalidSpec {
                reason: "workflow must declare at least one transition",
            });
        }
        Ok(())
    }

    /// Compila la especificación validada en una instancia ejecutable [`StateMachineWorkflow`].
    pub fn compile(self) -> Result<StateMachineWorkflow, WorkflowError> {
        StateMachineWorkflow::new(self)
    }
}

/// Estado resultado de procesar un evento, error o timeout en [`StateMachineWorkflow`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStepStatus {
    /// Se tomó una transición normal válida.
    Transitioned,
    /// La entrada fue ignorada por idempotencia al repetirse [`IdempotencyKey`].
    DuplicateIgnored,
    /// Ninguna transición normal cumplió su guarda para el estado y entrada actuales.
    NoMatchingTransition,
    /// Un error activó un reintento acotado dentro de la política configurada.
    Retrying {
        /// Número de intento actual (`1..=max_retries`).
        attempt: u32,
        /// Máximo de reintentos configurados.
        max_retries: u32,
    },
    /// Se agotaron los reintentos y se tomó la rama explícita `OnError`.
    ErrorBranchTaken,
    /// Expiró [`MonotonicDeadline`] y se tomó la rama explícita `OnTimeout`.
    TimeoutBranchTaken,
    /// La máquina ya estaba en un estado terminal y no procesó más transiciones.
    AlreadyTerminal,
}

/// Resultado inmutable de un paso en [`StateMachineWorkflow`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowStepOutcome {
    /// Resultado de la evaluación del paso.
    pub status: WorkflowStepStatus,
    /// Estado previo al paso.
    pub previous_state: WorkflowStateId,
    /// Estado posterior al paso.
    pub current_state: WorkflowStateId,
    /// Acciones propuestas puras generadas por la transición (vacío si no hubo transición o fue duplicada).
    pub proposed_actions: Vec<ProposedWorkflowAction>,
    /// Tick lógico del paso.
    pub tick: Tick,
    /// Instante monotónico en que se evaluó el paso.
    pub timestamp: MonotonicInstant,
}

impl WorkflowStepOutcome {
    /// Extrae únicamente los [`ProposedEffect`] puros de las acciones propuestas.
    pub fn proposed_effects(&self) -> Vec<ProposedEffect> {
        self.proposed_actions
            .iter()
            .map(|action| action.effect().clone())
            .collect()
    }
}

/// Registro acotado FIFO de claves de idempotencia ya procesadas.
#[derive(Debug, Clone)]
struct IdempotencyCache {
    capacity: usize,
    order: VecDeque<IdempotencyKey>,
    set: HashSet<IdempotencyKey>,
}

impl IdempotencyCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::with_capacity(capacity),
            set: HashSet::with_capacity(capacity),
        }
    }

    fn contains(&self, key: &IdempotencyKey) -> bool {
        self.set.contains(key)
    }

    fn record(&mut self, key: IdempotencyKey) {
        if self.set.contains(&key) {
            return;
        }
        if self.order.len() == self.capacity
            && let Some(evicted) = self.order.pop_front()
        {
            self.set.remove(&evicted);
        }
        self.order.push_back(key.clone());
        self.set.insert(key);
    }
}

/// Motor determinista de máquina de estados para automatización por eventos.
///
/// No requiere [`crate::Planner`], proveedores externos ni ejecución de IO.
#[derive(Debug, Clone)]
pub struct StateMachineWorkflow {
    spec: WorkflowSpec,
    current_state: WorkflowStateId,
    retry_count: u32,
    deadline: Option<MonotonicDeadline>,
    idempotency: IdempotencyCache,
}

impl StateMachineWorkflow {
    /// Crea y valida una instancia de [`StateMachineWorkflow`] a partir de `spec`.
    pub fn new(spec: WorkflowSpec) -> Result<Self, WorkflowError> {
        spec.validate()?;
        let current_state = spec.initial_state.clone();
        let idempotency = IdempotencyCache::new(spec.idempotency_capacity);
        Ok(Self {
            spec,
            current_state,
            retry_count: 0,
            deadline: None,
            idempotency,
        })
    }

    /// Devuelve la especificación del workflow.
    pub fn spec(&self) -> &WorkflowSpec {
        &self.spec
    }

    /// Devuelve el estado actual de la máquina.
    pub fn current_state(&self) -> &WorkflowStateId {
        &self.current_state
    }

    /// Indica si el estado actual pertenece al conjunto de estados terminales.
    pub fn is_terminal(&self) -> bool {
        self.spec
            .terminal_states
            .iter()
            .any(|s| s == &self.current_state)
    }

    /// Devuelve el número de reintentos acumulados.
    pub const fn retry_count(&self) -> u32 {
        self.retry_count
    }

    /// Configura o limpia el [`MonotonicDeadline`] activo para evaluar timeouts (`OnTimeout`).
    pub fn set_deadline(&mut self, deadline: Option<MonotonicDeadline>) {
        self.deadline = deadline;
    }

    /// Devuelve el [`MonotonicDeadline`] activo, si existe.
    pub const fn deadline(&self) -> Option<MonotonicDeadline> {
        self.deadline
    }

    /// Procesa una entrada tipada (`SchedulerEvent` o `CognitiveDecisionEvent`) leyendo `clock`.
    pub fn step<C: MonotonicClock + ?Sized>(
        &mut self,
        input: impl Into<WorkflowInput>,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        clock: &C,
    ) -> Result<WorkflowStepOutcome, WorkflowError> {
        self.step_at(input, idempotency_key, tick, clock.now())
    }

    /// Procesa una entrada tipada en un [`MonotonicInstant`] explícito.
    pub fn step_at(
        &mut self,
        input: impl Into<WorkflowInput>,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        now: MonotonicInstant,
    ) -> Result<WorkflowStepOutcome, WorkflowError> {
        let input = input.into();
        let previous_state = self.current_state.clone();

        if let Some(outcome) = self.check_duplicate_or_terminal(&idempotency_key, tick, now) {
            return Ok(outcome);
        }

        if let Some(outcome) = self.check_timeout_at(&idempotency_key, tick, now)? {
            return Ok(outcome);
        }

        let ctx = WorkflowContext {
            current_state: &self.current_state,
            tick,
            now,
            retry_count: self.retry_count,
            input: &input,
        };

        let matched = self.spec.transitions.iter().find(|transition| {
            transition.from == self.current_state
                && transition
                    .guard
                    .evaluate(&ctx, self.spec.default_retry_policy)
        });

        let Some(transition) = matched.cloned() else {
            return Ok(WorkflowStepOutcome {
                status: WorkflowStepStatus::NoMatchingTransition,
                previous_state,
                current_state: self.current_state.clone(),
                proposed_actions: Vec::new(),
                tick,
                timestamp: now,
            });
        };

        self.current_state = transition.to;
        self.retry_count = 0;
        if let Some(key) = idempotency_key {
            self.idempotency.record(key);
        }

        Ok(WorkflowStepOutcome {
            status: WorkflowStepStatus::Transitioned,
            previous_state,
            current_state: self.current_state.clone(),
            proposed_actions: transition.actions,
            tick,
            timestamp: now,
        })
    }

    /// Reporta un error de ejecución en el estado actual, aplicando [`RetryPolicy`] y `OnError`.
    pub fn report_error<C: MonotonicClock + ?Sized>(
        &mut self,
        _error: &WorkflowStepError,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        clock: &C,
    ) -> Result<WorkflowStepOutcome, WorkflowError> {
        self.report_error_at(_error, idempotency_key, tick, clock.now())
    }

    /// Reporta un error de ejecución en un [`MonotonicInstant`] explícito.
    pub fn report_error_at(
        &mut self,
        _error: &WorkflowStepError,
        idempotency_key: Option<IdempotencyKey>,
        tick: Tick,
        now: MonotonicInstant,
    ) -> Result<WorkflowStepOutcome, WorkflowError> {
        let previous_state = self.current_state.clone();

        if let Some(outcome) = self.check_duplicate_or_terminal(&idempotency_key, tick, now) {
            return Ok(outcome);
        }

        let branch = self
            .find_error_branch(&self.current_state)
            .cloned()
            .ok_or_else(|| WorkflowError::MissingErrorBranch {
                state: self.current_state.clone(),
            })?;

        if let Some(key) = idempotency_key {
            self.idempotency.record(key);
        }

        if branch.retry_policy.can_retry(self.retry_count) {
            self.retry_count += 1;
            return Ok(WorkflowStepOutcome {
                status: WorkflowStepStatus::Retrying {
                    attempt: self.retry_count,
                    max_retries: branch.retry_policy.max_retries,
                },
                previous_state,
                current_state: self.current_state.clone(),
                proposed_actions: branch.retry_actions,
                tick,
                timestamp: now,
            });
        }

        self.current_state = branch.on_exhausted_state;
        Ok(WorkflowStepOutcome {
            status: WorkflowStepStatus::ErrorBranchTaken,
            previous_state,
            current_state: self.current_state.clone(),
            proposed_actions: branch.on_exhausted_actions,
            tick,
            timestamp: now,
        })
    }

    fn check_duplicate_or_terminal(
        &self,
        idempotency_key: &Option<IdempotencyKey>,
        tick: Tick,
        now: MonotonicInstant,
    ) -> Option<WorkflowStepOutcome> {
        if let Some(key) = idempotency_key
            && self.idempotency.contains(key)
        {
            return Some(WorkflowStepOutcome {
                status: WorkflowStepStatus::DuplicateIgnored,
                previous_state: self.current_state.clone(),
                current_state: self.current_state.clone(),
                proposed_actions: Vec::new(),
                tick,
                timestamp: now,
            });
        }

        if self.is_terminal() {
            return Some(WorkflowStepOutcome {
                status: WorkflowStepStatus::AlreadyTerminal,
                previous_state: self.current_state.clone(),
                current_state: self.current_state.clone(),
                proposed_actions: Vec::new(),
                tick,
                timestamp: now,
            });
        }

        None
    }

    fn check_timeout_at(
        &mut self,
        idempotency_key: &Option<IdempotencyKey>,
        tick: Tick,
        now: MonotonicInstant,
    ) -> Result<Option<WorkflowStepOutcome>, WorkflowError> {
        let Some(deadline) = self.deadline else {
            return Ok(None);
        };
        if !deadline.is_expired_at(now) {
            return Ok(None);
        }

        let branch = self
            .find_timeout_branch(&self.current_state)
            .cloned()
            .ok_or_else(|| WorkflowError::MissingTimeoutBranch {
                state: self.current_state.clone(),
            })?;

        let previous_state = self.current_state.clone();
        self.current_state = branch.to;
        self.deadline = None;
        if let Some(key) = idempotency_key.clone() {
            self.idempotency.record(key);
        }

        Ok(Some(WorkflowStepOutcome {
            status: WorkflowStepStatus::TimeoutBranchTaken,
            previous_state,
            current_state: self.current_state.clone(),
            proposed_actions: branch.actions,
            tick,
            timestamp: now,
        }))
    }

    fn find_error_branch(&self, state: &WorkflowStateId) -> Option<&ErrorBranch> {
        self.spec
            .error_branches
            .iter()
            .find(|b| b.from.as_ref() == Some(state))
            .or_else(|| self.spec.error_branches.iter().find(|b| b.from.is_none()))
    }

    fn find_timeout_branch(&self, state: &WorkflowStateId) -> Option<&TimeoutBranch> {
        self.spec
            .timeout_branches
            .iter()
            .find(|b| b.from.as_ref() == Some(state))
            .or_else(|| self.spec.timeout_branches.iter().find(|b| b.from.is_none()))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::clock::VirtualMonotonicClock;

    use super::*;

    fn state(name: &str) -> WorkflowStateId {
        WorkflowStateId::new(name).expect("valid state id")
    }

    fn event_kind(name: &str) -> EventKind {
        EventKind::new(name).expect("valid event kind")
    }

    fn three_step_spec() -> WorkflowSpec {
        let init = state("Init");
        let step1 = state("Step1");
        let step2 = state("Step2");
        let completed = state("Completed");
        let failed = state("Failed");
        let timed_out = state("TimedOut");

        let mut spec = WorkflowSpec::new(
            WorkflowId::new("three-step-pipeline").expect("valid id"),
            init.clone(),
            vec![completed.clone(), failed.clone(), timed_out.clone()],
        );

        spec.transitions = vec![
            WorkflowTransition::new(
                init,
                step1.clone(),
                TransitionGuard::Trigger(WorkflowTrigger::Event(event_kind("workflow.start"))),
                vec![
                    ProposedWorkflowAction::emit_event(
                        "prepare",
                        "pipeline.stage1",
                        "stage1.ready",
                        "init->step1",
                    )
                    .expect("valid action"),
                ],
            ),
            WorkflowTransition::new(
                step1,
                step2.clone(),
                TransitionGuard::All(vec![
                    TransitionGuard::Trigger(WorkflowTrigger::Event(event_kind("stage1.done"))),
                    TransitionGuard::PayloadEquals("ok".to_string()),
                ]),
                vec![
                    ProposedWorkflowAction::emit_event(
                        "advance",
                        "pipeline.stage2",
                        "stage2.ready",
                        "step1->step2",
                    )
                    .expect("valid action"),
                ],
            ),
            WorkflowTransition::new(
                step2,
                completed,
                TransitionGuard::Trigger(WorkflowTrigger::Event(event_kind("stage2.done"))),
                vec![
                    ProposedWorkflowAction::log(
                        "complete",
                        "pipeline.audit",
                        EffectSeverity::Info,
                        "workflow completed",
                    )
                    .expect("valid action"),
                ],
            ),
        ];

        spec.error_branches = vec![ErrorBranch {
            from: None,
            retry_policy: RetryPolicy::new(2),
            retry_actions: vec![
                ProposedWorkflowAction::log(
                    "retry_step",
                    "pipeline.retry",
                    EffectSeverity::Warn,
                    "retrying after error",
                )
                .expect("valid action"),
            ],
            on_exhausted_state: failed,
            on_exhausted_actions: vec![
                ProposedWorkflowAction::log(
                    "abort_on_error",
                    "pipeline.error",
                    EffectSeverity::Error,
                    "retries exhausted",
                )
                .expect("valid action"),
            ],
        }];

        spec.timeout_branches = vec![TimeoutBranch {
            from: None,
            to: timed_out,
            actions: vec![
                ProposedWorkflowAction::log(
                    "abort_on_timeout",
                    "pipeline.timeout",
                    EffectSeverity::Error,
                    "deadline expired",
                )
                .expect("valid action"),
            ],
        }];

        spec
    }

    #[test]
    fn ac1_three_step_workflow_executes_without_planner_or_jev() {
        let clock = VirtualMonotonicClock::new();
        let mut workflow = three_step_spec().compile().expect("valid workflow");

        assert_eq!(workflow.current_state(), &state("Init"));
        assert!(!workflow.is_terminal());

        let o1 = workflow
            .step(
                SchedulerEvent::signal(event_kind("workflow.start")),
                Some(IdempotencyKey::new("k-1").expect("valid key")),
                Tick::new(1),
                &clock,
            )
            .expect("step 1 ok");
        assert_eq!(o1.status, WorkflowStepStatus::Transitioned);
        assert_eq!(o1.current_state, state("Step1"));
        assert_eq!(o1.proposed_actions.len(), 1);
        assert_eq!(o1.proposed_actions[0].action_name(), "prepare");

        let o2 = workflow
            .step(
                SchedulerEvent::new(event_kind("stage1.done"), "ok"),
                Some(IdempotencyKey::new("k-2").expect("valid key")),
                Tick::new(2),
                &clock,
            )
            .expect("step 2 ok");
        assert_eq!(o2.status, WorkflowStepStatus::Transitioned);
        assert_eq!(o2.current_state, state("Step2"));
        assert_eq!(o2.proposed_actions[0].action_name(), "advance");

        let o3 = workflow
            .step(
                SchedulerEvent::signal(event_kind("stage2.done")),
                Some(IdempotencyKey::new("k-3").expect("valid key")),
                Tick::new(3),
                &clock,
            )
            .expect("step 3 ok");
        assert_eq!(o3.status, WorkflowStepStatus::Transitioned);
        assert_eq!(o3.current_state, state("Completed"));
        assert!(workflow.is_terminal());
        assert_eq!(
            o3.proposed_effects(),
            vec![ProposedEffect::log(
                EffectSeverity::Info,
                "workflow completed"
            )]
        );
    }

    #[test]
    fn ac2_error_triggers_bounded_retry_then_explicit_error_branch_with_idempotency() {
        let clock = VirtualMonotonicClock::new();
        let mut workflow = three_step_spec().compile().expect("valid workflow");

        // Advance to Step1 first.
        workflow
            .step(
                SchedulerEvent::signal(event_kind("workflow.start")),
                Some(IdempotencyKey::new("ev-start").expect("valid key")),
                Tick::new(1),
                &clock,
            )
            .expect("start ok");

        // Duplicate delivery of the exact same IdempotencyKey is ignored with no duplicate action.
        let dup = workflow
            .step(
                SchedulerEvent::signal(event_kind("workflow.start")),
                Some(IdempotencyKey::new("ev-start").expect("valid key")),
                Tick::new(2),
                &clock,
            )
            .expect("dup ok");
        assert_eq!(dup.status, WorkflowStepStatus::DuplicateIgnored);
        assert_eq!(dup.current_state, state("Step1"));
        assert!(dup.proposed_actions.is_empty());

        let step_err = WorkflowStepError::new("IO_TRANSIENT", "temporary failure").expect("err");

        // Retry 1 of 2
        let r1 = workflow
            .report_error(
                &step_err,
                Some(IdempotencyKey::new("err-1").expect("valid key")),
                Tick::new(3),
                &clock,
            )
            .expect("retry 1");
        assert_eq!(
            r1.status,
            WorkflowStepStatus::Retrying {
                attempt: 1,
                max_retries: 2
            }
        );
        assert_eq!(workflow.current_state(), &state("Step1"));

        // Duplicate error with same key does not consume retry budget.
        let r1_dup = workflow
            .report_error(
                &step_err,
                Some(IdempotencyKey::new("err-1").expect("valid key")),
                Tick::new(4),
                &clock,
            )
            .expect("dup err");
        assert_eq!(r1_dup.status, WorkflowStepStatus::DuplicateIgnored);
        assert_eq!(workflow.retry_count(), 1);

        // Retry 2 of 2
        let r2 = workflow
            .report_error(
                &step_err,
                Some(IdempotencyKey::new("err-2").expect("valid key")),
                Tick::new(5),
                &clock,
            )
            .expect("retry 2");
        assert_eq!(
            r2.status,
            WorkflowStepStatus::Retrying {
                attempt: 2,
                max_retries: 2
            }
        );

        // Retries exhausted -> explicit OnError branch.
        let exhausted = workflow
            .report_error(
                &step_err,
                Some(IdempotencyKey::new("err-3").expect("valid key")),
                Tick::new(6),
                &clock,
            )
            .expect("exhausted");
        assert_eq!(exhausted.status, WorkflowStepStatus::ErrorBranchTaken);
        assert_eq!(exhausted.current_state, state("Failed"));
        assert!(workflow.is_terminal());
        assert_eq!(
            exhausted.proposed_actions[0].action_name(),
            "abort_on_error"
        );
    }

    #[test]
    fn ac2_timeout_activates_explicit_timeout_branch_using_monotonic_deadline() {
        let clock = VirtualMonotonicClock::new();
        let mut workflow = three_step_spec().compile().expect("valid workflow");

        workflow.set_deadline(Some(MonotonicDeadline::from_now(
            &clock,
            Duration::from_millis(50),
        )));

        // Before deadline: normal transition proceeds.
        clock.advance(Duration::from_millis(20));
        let ok = workflow
            .step(
                SchedulerEvent::signal(event_kind("workflow.start")),
                None,
                Tick::new(1),
                &clock,
            )
            .expect("before deadline");
        assert_eq!(ok.status, WorkflowStepStatus::Transitioned);
        assert_eq!(ok.current_state, state("Step1"));

        // Advance past deadline: explicit OnTimeout branch is taken.
        clock.advance(Duration::from_millis(40));
        let timed_out = workflow
            .step(
                SchedulerEvent::new(event_kind("stage1.done"), "ok"),
                None,
                Tick::new(2),
                &clock,
            )
            .expect("timeout branch");
        assert_eq!(timed_out.status, WorkflowStepStatus::TimeoutBranchTaken);
        assert_eq!(timed_out.current_state, state("TimedOut"));
        assert!(workflow.is_terminal());
        assert_eq!(
            timed_out.proposed_actions[0].action_name(),
            "abort_on_timeout"
        );
    }

    #[test]
    fn ac3_cognitive_decision_event_triggers_workflow_directly_without_bridge() {
        let clock = VirtualMonotonicClock::new();
        let init = state("Idle");
        let executing = state("Executing");
        let completed = state("Completed");

        let mut spec = WorkflowSpec::new(
            WorkflowId::new("cognitive-triggered").expect("valid id"),
            init.clone(),
            vec![completed.clone()],
        );
        spec.transitions = vec![
            WorkflowTransition::new(
                init,
                executing.clone(),
                TransitionGuard::Trigger(
                    WorkflowTrigger::cognitive_decision_with_payload("PLAN_COMMIT", "route=north")
                        .expect("valid trigger"),
                ),
                vec![
                    ProposedWorkflowAction::emit_event(
                        "dispatch_route",
                        "actuator.nav",
                        "nav.execute",
                        "route=north",
                    )
                    .expect("valid action"),
                ],
            ),
            WorkflowTransition::new(
                executing,
                completed.clone(),
                TransitionGuard::Trigger(WorkflowTrigger::Event(event_kind("nav.arrived"))),
                vec![
                    ProposedWorkflowAction::log(
                        "mark_done",
                        "nav.audit",
                        EffectSeverity::Info,
                        "arrived",
                    )
                    .expect("valid action"),
                ],
            ),
        ];

        let mut workflow = spec.compile().expect("valid workflow");
        let decision = CognitiveDecisionEvent::new("PLAN_COMMIT", "route=north", Tick::new(10))
            .expect("valid decision");

        let outcome = workflow
            .step(
                decision,
                Some(IdempotencyKey::new("cog-10").expect("valid key")),
                Tick::new(10),
                &clock,
            )
            .expect("cognitive trigger ok");

        assert_eq!(outcome.status, WorkflowStepStatus::Transitioned);
        assert_eq!(outcome.current_state, state("Executing"));
        assert_eq!(outcome.proposed_actions[0].action_name(), "dispatch_route");

        let final_outcome = workflow
            .step(
                SchedulerEvent::signal(event_kind("nav.arrived")),
                Some(IdempotencyKey::new("nav-11").expect("valid key")),
                Tick::new(11),
                &clock,
            )
            .expect("final step ok");
        assert_eq!(final_outcome.status, WorkflowStepStatus::Transitioned);
        assert_eq!(final_outcome.current_state, completed);
        assert!(workflow.is_terminal());
    }
}
