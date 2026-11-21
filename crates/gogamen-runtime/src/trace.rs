//! Registro determinista de trazas por tick para el runtime headless de Gogamen.

use std::fmt;

use gogamen_core::{
    Action, ApplyOutcome, DeterminismMetadata, GuardDecision, GuardRejectionReason, MasterSeed,
    Observation, PlanStatus, Tick, TraceHasher,
};

/// Causa explícita de rechazo o activación de respaldo registrada en la traza.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionCause {
    /// Rechazo emitido directamente por [`gogamen_core::ActionGuard`].
    Guard(GuardRejectionReason),
    /// El planificador agotó su presupuesto de transiciones o su deadline monotónico.
    PlanBudgetExhausted,
    /// La planificación fue cancelada antes de completarse.
    PlanCancelled,
    /// El planificador no encontró ninguna acción válida para el objetivo.
    PlanNoValidAction,
    /// Fallo al observar o actuar sobre el entorno vivo.
    EnvironmentError(String),
    /// Fallo del modelo generativo durante la planificación.
    ModelError(String),
}

impl fmt::Display for RejectionCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Guard(reason) => write!(f, "guard: {reason}"),
            Self::PlanBudgetExhausted => write!(f, "plan: budget or monotonic deadline exhausted"),
            Self::PlanCancelled => write!(f, "plan: cancelled"),
            Self::PlanNoValidAction => write!(f, "plan: no valid action"),
            Self::EnvironmentError(msg) => write!(f, "environment: {msg}"),
            Self::ModelError(msg) => write!(f, "model: {msg}"),
        }
    }
}

/// Entrada inmutable de traza correspondiente a un único [`Tick`] del bucle
/// `observe -> plan -> guard -> act`.
#[derive(Debug, Clone, PartialEq)]
pub struct TickTraceEntry {
    /// Tick lógico ejecutado.
    pub tick: Tick,
    /// Semilla maestra de la ejecución.
    pub seed: MasterSeed,
    /// Observación capturada al inicio del tick.
    pub observation: Observation,
    /// Estado emitido por el planificador.
    pub plan_status: PlanStatus,
    /// Acción propuesta por el planificador, si la hubo.
    pub selected_action: Option<Action>,
    /// Decisión emitida por [`gogamen_core::ActionGuard`].
    pub guard_decision: GuardDecision,
    /// Acción efectivamente aplicada al entorno vivo (`None` en `NoOp` o `Reject`).
    pub applied_action: Option<Action>,
    /// Causa de rechazo o respaldo si la propuesta no fue autorizada normalmente.
    pub rejection_reason: Option<RejectionCause>,
    /// Indica si se ejecutó una acción de respaldo seguro en lugar del plan nominal.
    pub used_fallback: bool,
    /// Resultado de aplicar la acción en el entorno vivo (`None` si no se actuó).
    pub outcome: Option<ApplyOutcome>,
}

/// Traza completa y determinista de una ejecución del runtime headless.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionTrace {
    seed: MasterSeed,
    metadata: DeterminismMetadata,
    entries: Vec<TickTraceEntry>,
    hasher: TraceHasher,
}

impl ExecutionTrace {
    /// Crea una nueva traza vacía asociada a `seed`.
    pub fn new(seed: MasterSeed) -> Self {
        let metadata = DeterminismMetadata::current(seed);
        let hasher = TraceHasher::with_metadata(&metadata);
        Self {
            seed,
            metadata,
            entries: Vec::new(),
            hasher,
        }
    }

    /// Registra una entrada de tick y actualiza el hash determinista de traza.
    pub fn push(&mut self, entry: TickTraceEntry) {
        self.hasher.write_tick(entry.tick);
        self.hasher.write_observation(&entry.observation);
        self.hasher.write_u8(plan_status_code(entry.plan_status));
        self.hasher.write_u8(u8::from(entry.used_fallback));

        match &entry.applied_action {
            Some(action) => {
                self.hasher.write_u8(1);
                write_action(&mut self.hasher, action);
            }
            None => self.hasher.write_u8(0),
        }

        if let Some(reason) = &entry.rejection_reason {
            self.hasher.write_u8(1);
            self.hasher.write_str(&reason.to_string());
        } else {
            self.hasher.write_u8(0);
        }

        if let Some(outcome) = &entry.outcome {
            self.hasher.write_u8(1);
            self.hasher.write_observation(outcome.observation());
            self.hasher.write_terminal(outcome.terminal());
        } else {
            self.hasher.write_u8(0);
        }

        self.entries.push(entry);
    }

    /// Devuelve la [`MasterSeed`] de esta ejecución.
    pub const fn seed(&self) -> MasterSeed {
        self.seed
    }

    /// Devuelve los [`DeterminismMetadata`] asociados a la traza.
    pub const fn metadata(&self) -> &DeterminismMetadata {
        &self.metadata
    }

    /// Devuelve todas las entradas de tick registradas en orden cronológico.
    pub fn entries(&self) -> &[TickTraceEntry] {
        &self.entries
    }

    /// Número de ticks registrados en la traza.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Indica si la traza está vacía.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Busca la entrada correspondiente a un [`Tick`] concreto.
    pub fn get_tick(&self, tick: Tick) -> Option<&TickTraceEntry> {
        self.entries.iter().find(|entry| entry.tick == tick)
    }

    /// Número de ticks en los que se aplicó una acción al entorno vivo.
    pub fn applied_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.applied_action.is_some())
            .count()
    }

    /// Número de ticks en los que se utilizó una acción de respaldo (`used_fallback == true`).
    pub fn fallback_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.used_fallback)
            .count()
    }

    /// Número de ticks en los que la guardia rechazó la propuesta (`GuardDecision::Reject`).
    pub fn rejected_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.guard_decision.is_rejected())
            .count()
    }

    /// Número de ticks en los que la guardia decidió `NoOp`.
    pub fn noop_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.guard_decision.is_no_op())
            .count()
    }

    /// Devuelve el hash FNV-1a de 64 bits de la traza acumulada.
    pub const fn trace_hash(&self) -> u64 {
        self.hasher.finish()
    }

    /// Devuelve el hash de la traza formateado en hexadecimal de 16 dígitos.
    pub fn trace_hash_hex(&self) -> String {
        self.hasher.finish_hex()
    }
}

const fn plan_status_code(status: PlanStatus) -> u8 {
    match status {
        PlanStatus::Completed => 0,
        PlanStatus::BudgetExhausted => 1,
        PlanStatus::Cancelled => 2,
        PlanStatus::NoValidAction => 3,
    }
}

fn write_action(hasher: &mut TraceHasher, action: &Action) {
    if let Some(discrete) = action.as_discrete() {
        hasher.write_u8(0);
        hasher.write_u64(discrete.index() as u64);
    } else if let Some(continuous) = action.as_continuous() {
        hasher.write_u8(1);
        hasher.write_u64(continuous.len() as u64);
        for &val in continuous {
            hasher.write_f64(val);
        }
    }
}
