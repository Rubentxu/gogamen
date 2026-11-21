//! Memoria procedimental, versionado y promoción/reversión de estrategias (`std`-only).
//!
//! Este módulo implementa almacenamiento determinista de políticas tabulares versionadas,
//! compuerta de promoción basada en evidencia de fixture y reversión íntegra de versiones:
//! - [`StrategyId`] y [`StrategyVersion`]: identificadores tipados y monotónicos de estrategia.
//! - [`TabularPolicy`]: política tabular determinista `(state_bucket, action_idx) -> value`
//!   con selección argmax determinista (desempate por menor índice de acción).
//! - [`StrategyStats`]: estadísticas de evaluación asociadas a cada versión de estrategia
//!   (`episodes_evaluated`, `mean_return`, `holdout_score`, `failures`, `evidence_refs`).
//! - [`StrategyLifecycle`]: ciclo de vida explícito (`CandidateUntested`, `EvaluatedRejected`,
//!   `Active`, `ArchivedRolledBack`).
//! - [`EvaluationCriteria`], [`EvaluationEvidence`] y [`PromotionGate`]: compuerta de promoción
//!   que impide que una candidata no probada sustituya a la política activa y registra de forma
//!   permanente y auditable los resultados negativos cuando una candidata es rechazada.
//! - [`ProceduralMemory`]: mantiene la estrategia activa, el historial completo de versiones
//!   (activas previas, candidatas, rechazadas y archivadas), el log de auditoría y [`ProceduralMemory::rollback`],
//!   que restaura la versión anterior íntegra incluidas sus [`StrategyStats`].
//!
//! # Aislamiento de seguridad
//! Ni [`TabularPolicy`], ni [`PromotionGate`], ni [`ProceduralMemory`] reciben, almacenan o
//! mutan [`crate::ActionGuard`], [`crate::ActionSpace`], [`crate::DomainPrecondition`] ni
//! permisos de ejecución. Toda acción seleccionada por una estrategia debe pasar igualmente
//! por las restricciones inviolables del guardia de acción.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::determinism::TraceHasher;
use crate::provenance::EvidenceRef;

/// Versión de esquema usada para hashing determinista de [`ProceduralMemory`].
pub const PROCEDURAL_MEMORY_SCHEMA_VERSION: u32 = 1;

/// Identificador tipado de una estrategia procedimental.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StrategyId(u64);

impl StrategyId {
    /// Crea un [`StrategyId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for StrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "strategy({})", self.0)
    }
}

/// Versión monotónica tipada de una estrategia procedimental (`>= 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StrategyVersion(u64);

impl StrategyVersion {
    /// Versión inicial (`1`).
    pub const INITIAL: Self = Self(1);

    /// Crea una [`StrategyVersion`] validando que sea `>= 1`.
    pub fn new(value: u64) -> Result<Self, ProceduralError> {
        if value == 0 {
            return Err(ProceduralError::InvalidVersion { version: value });
        }
        Ok(Self(value))
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Devuelve la siguiente versión monotónica o error por desbordamiento.
    pub fn next(self) -> Result<Self, ProceduralError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(ProceduralError::VersionOverflow)
    }
}

impl fmt::Display for StrategyVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// Ciclo de vida explícito de una versión de estrategia en [`ProceduralMemory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StrategyLifecycle {
    /// Candidata registrada sin evaluación aprobada en fixture; jamás sustituye a la activa.
    CandidateUntested,
    /// Candidata evaluada en fixture que falló [`EvaluationCriteria`]; sus resultados negativos
    /// permanecen visibles y auditables.
    EvaluatedRejected,
    /// Estrategia actualmente activa.
    Active,
    /// Estrategia previamente activa que fue sustituida por una promoción o revertida en rollback.
    ArchivedRolledBack,
}

impl StrategyLifecycle {
    /// Devuelve `true` si la estrategia está en estado [`Self::CandidateUntested`].
    pub const fn is_untested(self) -> bool {
        matches!(self, Self::CandidateUntested)
    }

    /// Devuelve `true` si la estrategia fue rechazada tras evaluación ([`Self::EvaluatedRejected`]).
    pub const fn is_rejected(self) -> bool {
        matches!(self, Self::EvaluatedRejected)
    }

    /// Devuelve `true` si la estrategia está actualmente activa ([`Self::Active`]).
    pub const fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Devuelve `true` si la estrategia está archivada o revertida ([`Self::ArchivedRolledBack`]).
    pub const fn is_archived(self) -> bool {
        matches!(self, Self::ArchivedRolledBack)
    }

    const fn as_u8(self) -> u8 {
        match self {
            Self::CandidateUntested => 0,
            Self::EvaluatedRejected => 1,
            Self::Active => 2,
            Self::ArchivedRolledBack => 3,
        }
    }
}

impl fmt::Display for StrategyLifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::CandidateUntested => "candidate_untested",
            Self::EvaluatedRejected => "evaluated_rejected",
            Self::Active => "active",
            Self::ArchivedRolledBack => "archived_rolled_back",
        };
        f.write_str(label)
    }
}

/// Errores explícitos de validación, promoción y reversión en memoria procedimental.
#[derive(Debug, Clone, PartialEq)]
pub enum ProceduralError {
    /// La versión de estrategia debe ser `>= 1`.
    InvalidVersion {
        /// Valor inválido recibido.
        version: u64,
    },
    /// Desbordamiento aritmético al asignar una nueva versión de estrategia.
    VersionOverflow,
    /// El número de buckets de estado debe ser al menos 1.
    InvalidStateCount {
        /// Valor recibido.
        num_states: usize,
    },
    /// El número de acciones debe ser al menos 1.
    InvalidActionCount {
        /// Valor recibido.
        num_actions: usize,
    },
    /// Un bucket de estado está fuera del rango `[0, num_states)`.
    StateOutOfRange {
        /// Bucket recibido.
        state_bucket: usize,
        /// Número de estados configurados.
        num_states: usize,
    },
    /// Un índice de acción está fuera del rango `[0, num_actions)`.
    ActionOutOfRange {
        /// Índice recibido.
        action_idx: usize,
        /// Número de acciones configuradas.
        num_actions: usize,
    },
    /// Un valor numérico (`f64`) es `NaN` o infinito.
    NonFiniteValue {
        /// Campo inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Parámetro inválido en [`StrategyStats`] (por ejemplo `failures > episodes_evaluated`).
    InvalidStats {
        /// Motivo de invalidez.
        reason: &'static str,
    },
    /// Parámetro inválido en [`EvaluationCriteria`].
    InvalidCriteria {
        /// Motivo de invalidez.
        reason: &'static str,
    },
    /// Las dimensiones `(num_states, num_actions)` de la política candidata no coinciden con las de la memoria.
    PolicyDimensionMismatch {
        /// Estados esperados.
        expected_states: usize,
        /// Acciones esperadas.
        expected_actions: usize,
        /// Estados recibidos.
        actual_states: usize,
        /// Acciones recibidas.
        actual_actions: usize,
    },
    /// No existe una estrategia registrada con la clave `(strategy_id, version)`.
    StrategyNotFound {
        /// Identificador consultado.
        strategy_id: StrategyId,
        /// Versión consultada.
        version: StrategyVersion,
    },
    /// Intento de promover una candidata que aún no tiene evidencia de evaluación en fixture.
    UntestedCandidate {
        /// Identificador de la candidata no probada.
        strategy_id: StrategyId,
        /// Versión de la candidata no probada.
        version: StrategyVersion,
    },
    /// Transición de ciclo de vida inválida para la operación solicitada.
    InvalidLifecycleTransition {
        /// Identificador de estrategia.
        strategy_id: StrategyId,
        /// Versión de estrategia.
        version: StrategyVersion,
        /// Estado actual del ciclo de vida.
        current: StrategyLifecycle,
        /// Operación rechazada.
        operation: &'static str,
    },
    /// No existe ninguna versión activa previa a la cual revertir mediante `rollback`.
    NoPreviousVersionToRollback,
}

impl fmt::Display for ProceduralError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion { version } => {
                write!(f, "invalid strategy version ({version}): must be >= 1")
            }
            Self::VersionOverflow => write!(f, "strategy version counter overflow"),
            Self::InvalidStateCount { num_states } => {
                write!(f, "invalid policy state count ({num_states}): must be >= 1")
            }
            Self::InvalidActionCount { num_actions } => {
                write!(
                    f,
                    "invalid policy action count ({num_actions}): must be >= 1"
                )
            }
            Self::StateOutOfRange {
                state_bucket,
                num_states,
            } => write!(
                f,
                "state_bucket ({state_bucket}) out of range for {num_states} states"
            ),
            Self::ActionOutOfRange {
                action_idx,
                num_actions,
            } => write!(
                f,
                "action_idx ({action_idx}) out of range for {num_actions} actions"
            ),
            Self::NonFiniteValue { field, value } => {
                write!(f, "non-finite value in '{field}': {value}")
            }
            Self::InvalidStats { reason } => write!(f, "invalid strategy stats: {reason}"),
            Self::InvalidCriteria { reason } => write!(f, "invalid evaluation criteria: {reason}"),
            Self::PolicyDimensionMismatch {
                expected_states,
                expected_actions,
                actual_states,
                actual_actions,
            } => write!(
                f,
                "policy dimension mismatch: expected ({expected_states} states, {expected_actions} actions), got ({actual_states} states, {actual_actions} actions)"
            ),
            Self::StrategyNotFound {
                strategy_id,
                version,
            } => write!(f, "strategy {strategy_id} {version} not found"),
            Self::UntestedCandidate {
                strategy_id,
                version,
            } => write!(
                f,
                "cannot promote untested candidate {strategy_id} {version}: fixture evaluation required"
            ),
            Self::InvalidLifecycleTransition {
                strategy_id,
                version,
                current,
                operation,
            } => write!(
                f,
                "invalid lifecycle transition for {strategy_id} {version} in state '{current}' during '{operation}'"
            ),
            Self::NoPreviousVersionToRollback => {
                write!(
                    f,
                    "no previous active strategy version available for rollback"
                )
            }
        }
    }
}

impl Error for ProceduralError {}

/// Política tabular determinista que mapea `(state_bucket, action_idx) -> value`.
///
/// La selección de acción escoge deterministamente la acción de mayor valor para el
/// estado dado, desempatando siempre por el menor índice de acción.
///
/// # Aislamiento de seguridad
/// `TabularPolicy` no contiene referencias a [`crate::ActionGuard`] ni puede convertirse
/// en una [`crate::ActionProposal`] o [`crate::Permission`]:
///
/// ```compile_fail
/// use gogamen_core::{ActionProposal, TabularPolicy};
/// let policy = TabularPolicy::new(2, 2).unwrap();
/// let _proposal: ActionProposal = policy.into();
/// ```
///
/// ```compile_fail
/// use gogamen_core::{Permission, TabularPolicy};
/// let policy = TabularPolicy::new(2, 2).unwrap();
/// let _perm: Permission = policy.into();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TabularPolicy {
    num_states: usize,
    num_actions: usize,
    values: BTreeMap<(usize, usize), f64>,
}

impl TabularPolicy {
    /// Crea una política tabular inicializada con valor `0.0` para todo par `(estado, acción)`.
    pub fn new(num_states: usize, num_actions: usize) -> Result<Self, ProceduralError> {
        if num_states == 0 {
            return Err(ProceduralError::InvalidStateCount { num_states });
        }
        if num_actions == 0 {
            return Err(ProceduralError::InvalidActionCount { num_actions });
        }
        Ok(Self {
            num_states,
            num_actions,
            values: BTreeMap::new(),
        })
    }

    /// Crea una política tabular a partir de una lista de entradas `(state_bucket, action_idx, value)`.
    pub fn from_entries(
        num_states: usize,
        num_actions: usize,
        entries: &[(usize, usize, f64)],
    ) -> Result<Self, ProceduralError> {
        let mut policy = Self::new(num_states, num_actions)?;
        for &(state_bucket, action_idx, value) in entries {
            policy.set_action_value(state_bucket, action_idx, value)?;
        }
        Ok(policy)
    }

    /// Devuelve el número de buckets de estado configurados.
    pub const fn num_states(&self) -> usize {
        self.num_states
    }

    /// Devuelve el número de acciones discretas configuradas.
    pub const fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Asigna un valor finito para el par `(state_bucket, action_idx)`.
    pub fn set_action_value(
        &mut self,
        state_bucket: usize,
        action_idx: usize,
        value: f64,
    ) -> Result<(), ProceduralError> {
        self.validate_state_action(state_bucket, action_idx)?;
        if !value.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "policy.action_value",
                value,
            });
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        self.values.insert((state_bucket, action_idx), normalized);
        Ok(())
    }

    /// Devuelve el valor registrado para `(state_bucket, action_idx)` (`0.0` si no se ha asignado).
    pub fn action_value(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<f64, ProceduralError> {
        self.validate_state_action(state_bucket, action_idx)?;
        Ok(self
            .values
            .get(&(state_bucket, action_idx))
            .copied()
            .unwrap_or(0.0))
    }

    /// Devuelve el vector completo de valores de acción para `state_bucket`.
    pub fn action_values(&self, state_bucket: usize) -> Result<Vec<f64>, ProceduralError> {
        self.validate_state(state_bucket)?;
        let mut row = Vec::with_capacity(self.num_actions);
        for action_idx in 0..self.num_actions {
            row.push(
                self.values
                    .get(&(state_bucket, action_idx))
                    .copied()
                    .unwrap_or(0.0),
            );
        }
        Ok(row)
    }

    /// Selecciona deterministamente `(best_action_idx, best_value)` para `state_bucket`.
    ///
    /// En caso de empate exacto de valor, elige siempre el menor índice de acción.
    pub fn select_action(&self, state_bucket: usize) -> Result<(usize, f64), ProceduralError> {
        self.validate_state(state_bucket)?;
        let mut best_action = 0_usize;
        let mut best_value = self.values.get(&(state_bucket, 0)).copied().unwrap_or(0.0);
        for action_idx in 1..self.num_actions {
            let value = self
                .values
                .get(&(state_bucket, action_idx))
                .copied()
                .unwrap_or(0.0);
            if value > best_value {
                best_value = value;
                best_action = action_idx;
            }
        }
        Ok((best_action, best_value))
    }

    /// Calcula el checksum determinista FNV-1a de la tabla de política mediante [`TraceHasher`].
    pub fn checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.procedural.tabular_policy");
        hasher.write_u64(self.num_states as u64);
        hasher.write_u64(self.num_actions as u64);
        hasher.write_u64(self.values.len() as u64);
        for (&(state_bucket, action_idx), &value) in &self.values {
            hasher.write_u64(state_bucket as u64);
            hasher.write_u64(action_idx as u64);
            hasher.write_f64(value);
        }
        hasher.finish()
    }

    fn validate_state(&self, state_bucket: usize) -> Result<(), ProceduralError> {
        if state_bucket >= self.num_states {
            return Err(ProceduralError::StateOutOfRange {
                state_bucket,
                num_states: self.num_states,
            });
        }
        Ok(())
    }

    fn validate_state_action(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<(), ProceduralError> {
        self.validate_state(state_bucket)?;
        if action_idx >= self.num_actions {
            return Err(ProceduralError::ActionOutOfRange {
                action_idx,
                num_actions: self.num_actions,
            });
        }
        Ok(())
    }
}

/// Estadísticas de evaluación asociadas a una versión de estrategia.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyStats {
    episodes_evaluated: u64,
    mean_return: f64,
    holdout_score: f64,
    failures: u64,
    evidence_refs: Vec<EvidenceRef>,
}

impl StrategyStats {
    /// Construye y valida estadísticas de evaluación para una estrategia.
    pub fn new(
        episodes_evaluated: u64,
        mean_return: f64,
        holdout_score: f64,
        failures: u64,
        evidence_refs: Vec<EvidenceRef>,
    ) -> Result<Self, ProceduralError> {
        if !mean_return.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "stats.mean_return",
                value: mean_return,
            });
        }
        if !holdout_score.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "stats.holdout_score",
                value: holdout_score,
            });
        }
        if failures > episodes_evaluated {
            return Err(ProceduralError::InvalidStats {
                reason: "failures cannot exceed episodes_evaluated",
            });
        }
        Ok(Self {
            episodes_evaluated,
            mean_return: if mean_return == 0.0 { 0.0 } else { mean_return },
            holdout_score: if holdout_score == 0.0 {
                0.0
            } else {
                holdout_score
            },
            failures,
            evidence_refs,
        })
    }

    /// Estadísticas vacías para una candidata aún no evaluada.
    pub const fn untested() -> Self {
        Self {
            episodes_evaluated: 0,
            mean_return: 0.0,
            holdout_score: 0.0,
            failures: 0,
            evidence_refs: Vec::new(),
        }
    }

    /// Número de episodios evaluados.
    pub const fn episodes_evaluated(&self) -> u64 {
        self.episodes_evaluated
    }

    /// Retorno medio observado en la evaluación.
    pub const fn mean_return(&self) -> f64 {
        self.mean_return
    }

    /// Puntuación obtenida sobre el conjunto holdout.
    pub const fn holdout_score(&self) -> f64 {
        self.holdout_score
    }

    /// Número de episodios fallidos observados.
    pub const fn failures(&self) -> u64 {
        self.failures
    }

    /// Tasa de fallos `failures / episodes_evaluated` (`0.0` si `episodes_evaluated == 0`).
    pub fn failure_rate(&self) -> f64 {
        if self.episodes_evaluated == 0 {
            0.0
        } else {
            (self.failures as f64) / (self.episodes_evaluated as f64)
        }
    }

    /// Referencias de evidencia que respaldan estas estadísticas.
    pub fn evidence_refs(&self) -> &[EvidenceRef] {
        &self.evidence_refs
    }
}

/// Criterios cuantitativos mínimos exigidos por [`PromotionGate`] para autorizar una promoción.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvaluationCriteria {
    min_episodes: u64,
    min_return_improvement: f64,
    max_failure_rate: f64,
    min_holdout_score: Option<f64>,
}

impl EvaluationCriteria {
    /// Construye y valida los criterios base de promoción.
    pub fn new(
        min_episodes: u64,
        min_return_improvement: f64,
        max_failure_rate: f64,
    ) -> Result<Self, ProceduralError> {
        if min_episodes == 0 {
            return Err(ProceduralError::InvalidCriteria {
                reason: "min_episodes must be >= 1",
            });
        }
        if !min_return_improvement.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "criteria.min_return_improvement",
                value: min_return_improvement,
            });
        }
        if min_return_improvement < 0.0 {
            return Err(ProceduralError::InvalidCriteria {
                reason: "min_return_improvement must be >= 0.0",
            });
        }
        if !max_failure_rate.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "criteria.max_failure_rate",
                value: max_failure_rate,
            });
        }
        if !(0.0..=1.0).contains(&max_failure_rate) {
            return Err(ProceduralError::InvalidCriteria {
                reason: "max_failure_rate must be within [0.0, 1.0]",
            });
        }
        Ok(Self {
            min_episodes,
            min_return_improvement: if min_return_improvement == 0.0 {
                0.0
            } else {
                min_return_improvement
            },
            max_failure_rate: if max_failure_rate == 0.0 {
                0.0
            } else {
                max_failure_rate
            },
            min_holdout_score: None,
        })
    }

    /// Configura un umbral mínimo opcional para `holdout_score`.
    pub fn with_min_holdout_score(
        mut self,
        min_holdout_score: f64,
    ) -> Result<Self, ProceduralError> {
        if !min_holdout_score.is_finite() {
            return Err(ProceduralError::NonFiniteValue {
                field: "criteria.min_holdout_score",
                value: min_holdout_score,
            });
        }
        self.min_holdout_score = Some(if min_holdout_score == 0.0 {
            0.0
        } else {
            min_holdout_score
        });
        Ok(self)
    }

    /// Número mínimo de episodios evaluados en fixture.
    pub const fn min_episodes(&self) -> u64 {
        self.min_episodes
    }

    /// Mejora mínima requerida de `mean_return` respecto a la estrategia activa vigente.
    pub const fn min_return_improvement(&self) -> f64 {
        self.min_return_improvement
    }

    /// Tasa máxima de fallos permitida en `[0.0, 1.0]`.
    pub const fn max_failure_rate(&self) -> f64 {
        self.max_failure_rate
    }

    /// Umbral mínimo opcional de `holdout_score`.
    pub const fn min_holdout_score(&self) -> Option<f64> {
        self.min_holdout_score
    }
}

/// Evidencia estructurada producida al evaluar una estrategia candidata sobre un fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationEvidence {
    stats: StrategyStats,
    fixture_tag: Option<String>,
}

impl EvaluationEvidence {
    /// Crea evidencia de evaluación a partir de [`StrategyStats`] validadas.
    pub const fn new(stats: StrategyStats) -> Self {
        Self {
            stats,
            fixture_tag: None,
        }
    }

    /// Asocia una etiqueta identificativa del fixture usado en la evaluación.
    pub fn with_fixture_tag(mut self, fixture_tag: impl Into<String>) -> Self {
        self.fixture_tag = Some(fixture_tag.into());
        self
    }

    /// Devuelve las [`StrategyStats`] medidas en el fixture.
    pub const fn stats(&self) -> &StrategyStats {
        &self.stats
    }

    /// Devuelve la etiqueta opcional del fixture.
    pub fn fixture_tag(&self) -> Option<&str> {
        self.fixture_tag.as_deref()
    }
}

/// Motivo cuantitativo explícito por el cual una candidata fue rechazada por [`PromotionGate`].
#[derive(Debug, Clone, PartialEq)]
pub enum RejectionReason {
    /// La candidata se evaluó en menos episodios que los exigidos por `min_episodes`.
    InsufficientEpisodes {
        /// Episodios observados.
        observed: u64,
        /// Mínimo requerido.
        required: u64,
    },
    /// La mejora de retorno medio no alcanzó `min_return_improvement`.
    InsufficientReturnImprovement {
        /// Retorno medio de la estrategia activa de referencia.
        baseline_return: f64,
        /// Retorno medio de la candidata.
        candidate_return: f64,
        /// Diferencia observada (`candidate_return - baseline_return`).
        improvement: f64,
        /// Mejora mínima exigida.
        required: f64,
    },
    /// La tasa de fallos superó `max_failure_rate`.
    FailureRateExceeded {
        /// Tasa de fallos observada.
        observed: f64,
        /// Máximo permitido.
        max_allowed: f64,
    },
    /// La puntuación en holdout quedó por debajo de `min_holdout_score`.
    HoldoutScoreBelowMinimum {
        /// Puntuación holdout observada.
        observed: f64,
        /// Mínimo requerido.
        required: f64,
    },
}

impl fmt::Display for RejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientEpisodes { observed, required } => write!(
                f,
                "insufficient episodes evaluated: observed {observed}, required >= {required}"
            ),
            Self::InsufficientReturnImprovement {
                baseline_return,
                candidate_return,
                improvement,
                required,
            } => write!(
                f,
                "insufficient return improvement: baseline={baseline_return:.4}, candidate={candidate_return:.4}, delta={improvement:.4} < required {required:.4}"
            ),
            Self::FailureRateExceeded {
                observed,
                max_allowed,
            } => write!(
                f,
                "failure rate exceeded: observed {observed:.4} > max_allowed {max_allowed:.4}"
            ),
            Self::HoldoutScoreBelowMinimum { observed, required } => write!(
                f,
                "holdout score below minimum: observed {observed:.4} < required {required:.4}"
            ),
        }
    }
}

/// Veredicto determinista emitido por [`PromotionGate::evaluate`].
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionGateVerdict {
    /// La evidencia satisface todos los criterios de [`EvaluationCriteria`].
    Approved,
    /// La evidencia incumple uno o más criterios; todos los motivos quedan expuestos.
    Rejected(Vec<RejectionReason>),
}

impl PromotionGateVerdict {
    /// Devuelve `true` si la compuerta aprobó la evidencia.
    pub const fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// Devuelve los motivos de rechazo si la compuerta rechazó la evidencia.
    pub fn rejection_reasons(&self) -> &[RejectionReason] {
        match self {
            Self::Approved => &[],
            Self::Rejected(reasons) => reasons,
        }
    }
}

/// Compuerta determinista de promoción basada en [`EvaluationCriteria`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PromotionGate {
    criteria: EvaluationCriteria,
}

impl PromotionGate {
    /// Crea una compuerta de promoción con los criterios especificados.
    pub const fn new(criteria: EvaluationCriteria) -> Self {
        Self { criteria }
    }

    /// Devuelve los [`EvaluationCriteria`] configurados en la compuerta.
    pub const fn criteria(&self) -> EvaluationCriteria {
        self.criteria
    }

    /// Evalúa las estadísticas de la candidata frente a las de la estrategia activa vigente.
    pub fn evaluate(
        &self,
        baseline: &StrategyStats,
        candidate: &StrategyStats,
    ) -> PromotionGateVerdict {
        let mut reasons = Vec::new();

        if candidate.episodes_evaluated() < self.criteria.min_episodes {
            reasons.push(RejectionReason::InsufficientEpisodes {
                observed: candidate.episodes_evaluated(),
                required: self.criteria.min_episodes,
            });
        }

        let improvement = candidate.mean_return() - baseline.mean_return();
        if improvement < self.criteria.min_return_improvement {
            reasons.push(RejectionReason::InsufficientReturnImprovement {
                baseline_return: baseline.mean_return(),
                candidate_return: candidate.mean_return(),
                improvement,
                required: self.criteria.min_return_improvement,
            });
        }

        let failure_rate = candidate.failure_rate();
        if failure_rate > self.criteria.max_failure_rate {
            reasons.push(RejectionReason::FailureRateExceeded {
                observed: failure_rate,
                max_allowed: self.criteria.max_failure_rate,
            });
        }

        if let Some(min_holdout) = self.criteria.min_holdout_score
            && candidate.holdout_score() < min_holdout
        {
            reasons.push(RejectionReason::HoldoutScoreBelowMinimum {
                observed: candidate.holdout_score(),
                required: min_holdout,
            });
        }

        if reasons.is_empty() {
            PromotionGateVerdict::Approved
        } else {
            PromotionGateVerdict::Rejected(reasons)
        }
    }
}

/// Registro completo e inmutablemente auditable de una versión de estrategia.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyRecord {
    strategy_id: StrategyId,
    version: StrategyVersion,
    policy: TabularPolicy,
    stats: StrategyStats,
    lifecycle: StrategyLifecycle,
    evaluation_recorded: bool,
    fixture_tag: Option<String>,
    rejection_reasons: Vec<RejectionReason>,
}

impl StrategyRecord {
    /// Identificador de la estrategia.
    pub const fn strategy_id(&self) -> StrategyId {
        self.strategy_id
    }

    /// Versión monotónica asignada a este registro.
    pub const fn version(&self) -> StrategyVersion {
        self.version
    }

    /// Política tabular asociada a esta versión.
    pub const fn policy(&self) -> &TabularPolicy {
        &self.policy
    }

    /// Estadísticas [`StrategyStats`] íntegras de esta versión.
    pub const fn stats(&self) -> &StrategyStats {
        &self.stats
    }

    /// Estado actual del ciclo de vida ([`StrategyLifecycle`]).
    pub const fn lifecycle(&self) -> StrategyLifecycle {
        self.lifecycle
    }

    /// Indica si ya se registró [`EvaluationEvidence`] en fixture para esta versión.
    pub const fn has_evaluation(&self) -> bool {
        self.evaluation_recorded
    }

    /// Etiqueta opcional del fixture donde se evaluó esta versión.
    pub fn fixture_tag(&self) -> Option<&str> {
        self.fixture_tag.as_deref()
    }

    /// Motivos de rechazo si la estrategia fue marcada como [`StrategyLifecycle::EvaluatedRejected`].
    pub fn rejection_reasons(&self) -> &[RejectionReason] {
        &self.rejection_reasons
    }
}

/// Resultado de intentar promover una candidata evaluada mediante [`ProceduralMemory::promote_candidate`].
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionDecision {
    /// La candidata superó [`PromotionGate`] y pasó a ser la estrategia [`StrategyLifecycle::Active`].
    Promoted {
        /// Clave de la versión anterior archivada.
        previous_strategy_id: StrategyId,
        /// Versión anterior archivada.
        previous_version: StrategyVersion,
        /// Identificador de la nueva estrategia activa.
        active_strategy_id: StrategyId,
        /// Versión de la nueva estrategia activa.
        active_version: StrategyVersion,
    },
    /// La candidata falló [`EvaluationCriteria`] y quedó registrada como [`StrategyLifecycle::EvaluatedRejected`]
    /// con sus estadísticas y motivos visibles.
    Rejected {
        /// Identificador de la candidata rechazada.
        strategy_id: StrategyId,
        /// Versión de la candidata rechazada.
        version: StrategyVersion,
        /// Motivos cuantitativos del rechazo.
        reasons: Vec<RejectionReason>,
        /// Estadísticas negativas medidas en el fixture.
        stats: StrategyStats,
    },
}

impl PromotionDecision {
    /// Devuelve `true` si la promoción fue autorizada.
    pub const fn is_promoted(&self) -> bool {
        matches!(self, Self::Promoted { .. })
    }

    /// Devuelve `true` si la candidata fue rechazada.
    pub const fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }
}

/// Evento estructurado del registro de auditoría de [`ProceduralMemory`].
#[derive(Debug, Clone, PartialEq)]
pub enum PromotionAuditEvent {
    /// Se registró una candidata sin evaluar (`CandidateUntested`).
    CandidateRegistered {
        /// Identificador de la candidata.
        strategy_id: StrategyId,
        /// Versión asignada.
        version: StrategyVersion,
    },
    /// Se registró evidencia de fixture para una candidata (aprobada o rechazada).
    CandidateEvaluated {
        /// Identificador de la candidata.
        strategy_id: StrategyId,
        /// Versión evaluada.
        version: StrategyVersion,
        /// Etiqueta opcional del fixture.
        fixture_tag: Option<String>,
        /// Estadísticas registradas.
        stats: StrategyStats,
        /// Veredicto frente a la estrategia activa en el momento de evaluar.
        verdict: PromotionGateVerdict,
    },
    /// Una candidata evaluada sustituyó a la estrategia activa anterior.
    Promoted {
        /// Estrategia anterior archivada.
        from: (StrategyId, StrategyVersion),
        /// Nueva estrategia activa.
        to: (StrategyId, StrategyVersion),
    },
    /// Se ejecutó un rollback restaurando la versión activa anterior con sus estadísticas íntegras.
    RolledBack {
        /// Estrategia activa revertida y archivada.
        from: (StrategyId, StrategyVersion),
        /// Estrategia anterior restaurada como activa.
        to: (StrategyId, StrategyVersion),
    },
}

/// Memoria procedimental determinista con compuerta de promoción, historial completo y rollback.
///
/// # Invariantes
/// - Una candidata en estado [`StrategyLifecycle::CandidateUntested`] **nunca** sustituye a la
///   estrategia activa (`ProceduralError::UntestedCandidate`).
/// - Las candidatas rechazadas permanecen registradas en estado [`StrategyLifecycle::EvaluatedRejected`]
///   con todas sus [`StrategyStats`], [`RejectionReason`] y eventos en [`Self::audit_log`];
///   nunca se borran ni ocultan.
/// - [`Self::rollback`] restaura la versión activa anterior con su [`TabularPolicy`] y
///   [`StrategyStats`] íntegras.
/// - **Aislamiento de seguridad:** `ProceduralMemory` no modifica ni posee acceso a
///   [`crate::ActionGuard`] ni puede convertirse en [`crate::ActionProposal`] o [`crate::Permission`]:
///
/// ```compile_fail
/// use gogamen_core::{
///     ActionProposal, EvaluationCriteria, ProceduralMemory, StrategyId, StrategyStats, TabularPolicy,
/// };
/// let policy = TabularPolicy::new(2, 2).unwrap();
/// let stats = StrategyStats::new(10, 1.0, 0.9, 0, vec![]).unwrap();
/// let criteria = EvaluationCriteria::new(5, 0.1, 0.1).unwrap();
/// let mem = ProceduralMemory::new(StrategyId::new(1), policy, stats, criteria);
/// let _proposal: ActionProposal = mem.into();
/// ```
#[derive(Debug, Clone)]
pub struct ProceduralMemory {
    num_states: usize,
    num_actions: usize,
    gate: PromotionGate,
    next_version: u64,
    active_key: (StrategyId, StrategyVersion),
    previous_active_stack: Vec<(StrategyId, StrategyVersion)>,
    records: BTreeMap<(StrategyId, StrategyVersion), StrategyRecord>,
    audit_log: Vec<PromotionAuditEvent>,
}

impl ProceduralMemory {
    /// Inicializa la memoria procedimental con una estrategia base activa (`v1`), sus
    /// estadísticas iniciales y la compuerta de promoción [`PromotionGate`].
    pub fn new(
        initial_strategy_id: StrategyId,
        initial_policy: TabularPolicy,
        initial_stats: StrategyStats,
        criteria: EvaluationCriteria,
    ) -> Self {
        let num_states = initial_policy.num_states();
        let num_actions = initial_policy.num_actions();
        let version = StrategyVersion::INITIAL;
        let active_key = (initial_strategy_id, version);
        let record = StrategyRecord {
            strategy_id: initial_strategy_id,
            version,
            policy: initial_policy,
            stats: initial_stats,
            lifecycle: StrategyLifecycle::Active,
            evaluation_recorded: true,
            fixture_tag: None,
            rejection_reasons: Vec::new(),
        };

        let mut records = BTreeMap::new();
        records.insert(active_key, record);

        Self {
            num_states,
            num_actions,
            gate: PromotionGate::new(criteria),
            next_version: 2,
            active_key,
            previous_active_stack: Vec::new(),
            records,
            audit_log: Vec::new(),
        }
    }

    /// Devuelve la [`PromotionGate`] vigente.
    pub const fn gate(&self) -> PromotionGate {
        self.gate
    }

    /// Devuelve los [`EvaluationCriteria`] vigentes.
    pub const fn criteria(&self) -> EvaluationCriteria {
        self.gate.criteria()
    }

    /// Devuelve el número de estados de las políticas gestionadas.
    pub const fn num_states(&self) -> usize {
        self.num_states
    }

    /// Devuelve el número de acciones de las políticas gestionadas.
    pub const fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Devuelve el registro completo de la estrategia actualmente activa.
    pub fn active_strategy(&self) -> &StrategyRecord {
        self.records
            .get(&self.active_key)
            .expect("active strategy key must always exist in procedural memory")
    }

    /// Devuelve la política tabular actualmente activa.
    pub fn active_policy(&self) -> &TabularPolicy {
        self.active_strategy().policy()
    }

    /// Devuelve las estadísticas [`StrategyStats`] de la estrategia actualmente activa.
    pub fn active_stats(&self) -> &StrategyStats {
        self.active_strategy().stats()
    }

    /// Consulta cualquier versión registrada por `(strategy_id, version)` (activa, candidata,
    /// rechazada o archivada).
    pub fn get_strategy(
        &self,
        strategy_id: StrategyId,
        version: StrategyVersion,
    ) -> Option<&StrategyRecord> {
        self.records.get(&(strategy_id, version))
    }

    /// Devuelve todas las versiones registradas en orden determinista por `(StrategyId, StrategyVersion)`.
    pub fn all_strategies(&self) -> Vec<&StrategyRecord> {
        self.records.values().collect()
    }

    /// Devuelve todas las candidatas rechazadas ([`StrategyLifecycle::EvaluatedRejected`])
    /// con sus resultados negativos visibles y auditables.
    pub fn rejected_strategies(&self) -> Vec<&StrategyRecord> {
        self.records
            .values()
            .filter(|rec| rec.lifecycle.is_rejected())
            .collect()
    }

    /// Devuelve las versiones activas previas disponibles en la pila de rollback (de más antigua a más reciente).
    pub fn rollback_history(&self) -> Vec<&StrategyRecord> {
        self.previous_active_stack
            .iter()
            .filter_map(|key| self.records.get(key))
            .collect()
    }

    /// Devuelve el número de versiones previas disponibles para [`Self::rollback`].
    pub fn rollback_depth(&self) -> usize {
        self.previous_active_stack.len()
    }

    /// Devuelve el log completo e inmutable de auditoría de registros, evaluaciones, promociones y rollbacks.
    pub fn audit_log(&self) -> &[PromotionAuditEvent] {
        &self.audit_log
    }

    /// Registra una nueva estrategia candidata en estado [`StrategyLifecycle::CandidateUntested`].
    ///
    /// Esta operación **nunca** altera la estrategia activa vigente.
    pub fn register_candidate(
        &mut self,
        strategy_id: StrategyId,
        policy: TabularPolicy,
    ) -> Result<StrategyVersion, ProceduralError> {
        if policy.num_states() != self.num_states || policy.num_actions() != self.num_actions {
            return Err(ProceduralError::PolicyDimensionMismatch {
                expected_states: self.num_states,
                expected_actions: self.num_actions,
                actual_states: policy.num_states(),
                actual_actions: policy.num_actions(),
            });
        }

        let version = StrategyVersion::new(self.next_version)?;
        self.next_version = self
            .next_version
            .checked_add(1)
            .ok_or(ProceduralError::VersionOverflow)?;

        let record = StrategyRecord {
            strategy_id,
            version,
            policy,
            stats: StrategyStats::untested(),
            lifecycle: StrategyLifecycle::CandidateUntested,
            evaluation_recorded: false,
            fixture_tag: None,
            rejection_reasons: Vec::new(),
        };

        self.records.insert((strategy_id, version), record);
        self.audit_log
            .push(PromotionAuditEvent::CandidateRegistered {
                strategy_id,
                version,
            });

        Ok(version)
    }

    /// Registra la [`EvaluationEvidence`] obtenida en fixture para una candidata y la evalúa
    /// frente a [`PromotionGate`].
    ///
    /// - Si la evidencia incumple [`EvaluationCriteria`], la candidata pasa inmediatamente a
    ///   [`StrategyLifecycle::EvaluatedRejected`] conservando sus estadísticas negativas y
    ///   motivos de rechazo de forma visible y auditable.
    /// - Si la evidencia cumple [`EvaluationCriteria`], queda registrada y lista para ser
    ///   autorizada explícitamente mediante [`Self::promote_candidate`].
    pub fn record_evaluation(
        &mut self,
        strategy_id: StrategyId,
        version: StrategyVersion,
        evidence: EvaluationEvidence,
    ) -> Result<PromotionGateVerdict, ProceduralError> {
        let key = (strategy_id, version);
        let baseline_stats = self.active_stats().clone();

        let record = self
            .records
            .get_mut(&key)
            .ok_or(ProceduralError::StrategyNotFound {
                strategy_id,
                version,
            })?;

        if record.lifecycle != StrategyLifecycle::CandidateUntested {
            return Err(ProceduralError::InvalidLifecycleTransition {
                strategy_id,
                version,
                current: record.lifecycle,
                operation: "record_evaluation",
            });
        }

        let verdict = self.gate.evaluate(&baseline_stats, evidence.stats());
        record.stats = evidence.stats.clone();
        record.fixture_tag = evidence.fixture_tag.clone();
        record.evaluation_recorded = true;

        match &verdict {
            PromotionGateVerdict::Approved => {
                record.rejection_reasons.clear();
            }
            PromotionGateVerdict::Rejected(reasons) => {
                record.lifecycle = StrategyLifecycle::EvaluatedRejected;
                record.rejection_reasons = reasons.clone();
            }
        }

        self.audit_log
            .push(PromotionAuditEvent::CandidateEvaluated {
                strategy_id,
                version,
                fixture_tag: evidence.fixture_tag,
                stats: evidence.stats,
                verdict: verdict.clone(),
            });

        Ok(verdict)
    }

    /// Intenta promover una candidata a [`StrategyLifecycle::Active`].
    ///
    /// - Si la candidata no tiene evidencia registrada (`CandidateUntested` sin evaluación),
    ///   devuelve [`ProceduralError::UntestedCandidate`] y **no** sustituye la política activa.
    /// - Si la candidata ya fue marcada como [`StrategyLifecycle::EvaluatedRejected`], devuelve
    ///   [`PromotionDecision::Rejected`] preservando intactos sus resultados negativos y la
    ///   política activa.
    /// - Si la candidata tiene evidencia registrada, re-evalúa [`PromotionGate`] contra la
    ///   estrategia activa vigente:
    ///   - Si cumple los criterios, archiva la versión activa anterior completa (incluidas sus
    ///     [`StrategyStats`]) en el historial de rollback y promueve la candidata a `Active`.
    ///   - Si falla los criterios, marca la candidata como `EvaluatedRejected` con sus motivos
    ///     visibles y devuelve [`PromotionDecision::Rejected`].
    pub fn promote_candidate(
        &mut self,
        strategy_id: StrategyId,
        version: StrategyVersion,
    ) -> Result<PromotionDecision, ProceduralError> {
        let key = (strategy_id, version);
        let baseline_stats = self.active_stats().clone();

        let (lifecycle, evaluation_recorded, candidate_stats, existing_reasons) = {
            let record = self
                .records
                .get(&key)
                .ok_or(ProceduralError::StrategyNotFound {
                    strategy_id,
                    version,
                })?;
            (
                record.lifecycle,
                record.evaluation_recorded,
                record.stats.clone(),
                record.rejection_reasons.clone(),
            )
        };

        match lifecycle {
            StrategyLifecycle::CandidateUntested => {
                if !evaluation_recorded {
                    return Err(ProceduralError::UntestedCandidate {
                        strategy_id,
                        version,
                    });
                }
            }
            StrategyLifecycle::EvaluatedRejected => {
                return Ok(PromotionDecision::Rejected {
                    strategy_id,
                    version,
                    reasons: existing_reasons,
                    stats: candidate_stats,
                });
            }
            StrategyLifecycle::Active | StrategyLifecycle::ArchivedRolledBack => {
                return Err(ProceduralError::InvalidLifecycleTransition {
                    strategy_id,
                    version,
                    current: lifecycle,
                    operation: "promote_candidate",
                });
            }
        }

        match self.gate.evaluate(&baseline_stats, &candidate_stats) {
            PromotionGateVerdict::Rejected(reasons) => {
                let record = self
                    .records
                    .get_mut(&key)
                    .expect("candidate key verified to exist");
                record.lifecycle = StrategyLifecycle::EvaluatedRejected;
                record.rejection_reasons = reasons.clone();
                Ok(PromotionDecision::Rejected {
                    strategy_id,
                    version,
                    reasons,
                    stats: candidate_stats,
                })
            }
            PromotionGateVerdict::Approved => {
                let previous_key = self.active_key;
                if let Some(prev_record) = self.records.get_mut(&previous_key) {
                    prev_record.lifecycle = StrategyLifecycle::ArchivedRolledBack;
                }
                self.previous_active_stack.push(previous_key);

                let promoted_record = self
                    .records
                    .get_mut(&key)
                    .expect("candidate key verified to exist");
                promoted_record.lifecycle = StrategyLifecycle::Active;
                promoted_record.rejection_reasons.clear();
                self.active_key = key;

                self.audit_log.push(PromotionAuditEvent::Promoted {
                    from: previous_key,
                    to: key,
                });

                Ok(PromotionDecision::Promoted {
                    previous_strategy_id: previous_key.0,
                    previous_version: previous_key.1,
                    active_strategy_id: strategy_id,
                    active_version: version,
                })
            }
        }
    }

    /// Atajo que registra la evidencia de fixture y ejecuta inmediatamente [`Self::promote_candidate`].
    pub fn evaluate_and_promote(
        &mut self,
        strategy_id: StrategyId,
        version: StrategyVersion,
        evidence: EvaluationEvidence,
    ) -> Result<PromotionDecision, ProceduralError> {
        let _ = self.record_evaluation(strategy_id, version, evidence)?;
        self.promote_candidate(strategy_id, version)
    }

    /// Revierte a la versión activa anterior, restaurando íntegramente su [`TabularPolicy`]
    /// y sus [`StrategyStats`].
    ///
    /// La estrategia que estaba activa queda archivada con estado
    /// [`StrategyLifecycle::ArchivedRolledBack`] y permanece visible en el historial.
    pub fn rollback(&mut self) -> Result<&StrategyRecord, ProceduralError> {
        let Some(previous_key) = self.previous_active_stack.pop() else {
            return Err(ProceduralError::NoPreviousVersionToRollback);
        };

        let current_key = self.active_key;
        if let Some(current_record) = self.records.get_mut(&current_key) {
            current_record.lifecycle = StrategyLifecycle::ArchivedRolledBack;
        }
        if let Some(restored_record) = self.records.get_mut(&previous_key) {
            restored_record.lifecycle = StrategyLifecycle::Active;
        }
        self.active_key = previous_key;

        self.audit_log.push(PromotionAuditEvent::RolledBack {
            from: current_key,
            to: previous_key,
        });

        Ok(self.active_strategy())
    }

    /// Calcula un checksum determinista FNV-1a del estado completo de la memoria procedimental.
    pub fn checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.procedural.memory");
        hasher.write_u64(u64::from(PROCEDURAL_MEMORY_SCHEMA_VERSION));
        hasher.write_u64(self.num_states as u64);
        hasher.write_u64(self.num_actions as u64);
        hasher.write_u64(self.active_key.0.as_u64());
        hasher.write_u64(self.active_key.1.as_u64());
        hasher.write_u64(self.previous_active_stack.len() as u64);
        for &(sid, ver) in &self.previous_active_stack {
            hasher.write_u64(sid.as_u64());
            hasher.write_u64(ver.as_u64());
        }
        hasher.write_u64(self.records.len() as u64);
        for (&(sid, ver), rec) in &self.records {
            hasher.write_u64(sid.as_u64());
            hasher.write_u64(ver.as_u64());
            hasher.write_u8(rec.lifecycle.as_u8());
            hasher.write_u8(u8::from(rec.evaluation_recorded));
            hasher.write_u64(rec.policy.checksum());
            hasher.write_u64(rec.stats.episodes_evaluated());
            hasher.write_f64(rec.stats.mean_return());
            hasher.write_f64(rec.stats.holdout_score());
            hasher.write_u64(rec.stats.failures());
            hasher.write_u64(rec.stats.evidence_refs().len() as u64);
            for &ev in rec.stats.evidence_refs() {
                hasher.write_u64(ev.as_u64());
            }
        }
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::action::{Action, ActionSpace};
    use crate::domain::{Observation, Terminal};
    use crate::guard::{
        ActionGuard, ActionProposal, GuardContext, GuardDecision, SafeFallbackPolicy,
    };
    use crate::units::{SimDuration, Tick};

    fn base_memory() -> ProceduralMemory {
        let initial_policy =
            TabularPolicy::from_entries(3, 3, &[(0, 0, 1.0), (1, 1, 0.8), (2, 0, 0.5)]).unwrap();
        let initial_stats = StrategyStats::new(
            20,
            1.0,
            0.75,
            2,
            vec![EvidenceRef::new(100), EvidenceRef::new(101)],
        )
        .unwrap();
        let criteria = EvaluationCriteria::new(10, 0.25, 0.15)
            .unwrap()
            .with_min_holdout_score(0.70)
            .unwrap();
        ProceduralMemory::new(StrategyId::new(1), initial_policy, initial_stats, criteria)
    }

    #[test]
    fn ac1_untested_candidate_never_replaces_active_and_fixture_evaluation_gates_promotion() {
        let mut mem = base_memory();
        let initial_policy_checksum = mem.active_policy().checksum();

        // 1. Registrar candidata sin evaluar: permanece CandidateUntested y no cambia la activa.
        let candidate_policy =
            TabularPolicy::from_entries(3, 3, &[(0, 1, 2.5), (1, 1, 1.2), (2, 1, 1.0)]).unwrap();
        let v2 = mem
            .register_candidate(StrategyId::new(2), candidate_policy.clone())
            .unwrap();
        assert_eq!(v2, StrategyVersion::new(2).unwrap());
        assert_eq!(
            mem.get_strategy(StrategyId::new(2), v2)
                .unwrap()
                .lifecycle(),
            StrategyLifecycle::CandidateUntested
        );
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(1));
        assert_eq!(mem.active_policy().checksum(), initial_policy_checksum);

        // 2. Intentar promover candidata no probada falla con UntestedCandidate y no altera la activa.
        let err = mem
            .promote_candidate(StrategyId::new(2), v2)
            .expect_err("untested candidate must never replace active policy");
        assert_eq!(
            err,
            ProceduralError::UntestedCandidate {
                strategy_id: StrategyId::new(2),
                version: v2,
            }
        );
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(1));
        assert_eq!(mem.active_policy().checksum(), initial_policy_checksum);

        // 3. Registrar evaluación en fixture que CUMPLE EvaluationCriteria y promover.
        let passing_stats = StrategyStats::new(
            15,
            1.6,
            0.88,
            1,
            vec![EvidenceRef::new(200), EvidenceRef::new(201)],
        )
        .unwrap();
        let verdict = mem
            .record_evaluation(
                StrategyId::new(2),
                v2,
                EvaluationEvidence::new(passing_stats.clone()).with_fixture_tag("fixture_nav_v1"),
            )
            .unwrap();
        assert!(verdict.is_approved());
        // Registrar evidencia por sí sola aún no promueve hasta invocar promote_candidate.
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(1));

        let decision = mem.promote_candidate(StrategyId::new(2), v2).unwrap();
        assert_eq!(
            decision,
            PromotionDecision::Promoted {
                previous_strategy_id: StrategyId::new(1),
                previous_version: StrategyVersion::INITIAL,
                active_strategy_id: StrategyId::new(2),
                active_version: v2,
            }
        );
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(2));
        assert_eq!(mem.active_strategy().version(), v2);
        assert_eq!(mem.active_strategy().lifecycle(), StrategyLifecycle::Active);
        assert_eq!(mem.active_policy(), &candidate_policy);
        assert_eq!(mem.active_stats(), &passing_stats);
        assert_eq!(
            mem.get_strategy(StrategyId::new(1), StrategyVersion::INITIAL)
                .unwrap()
                .lifecycle(),
            StrategyLifecycle::ArchivedRolledBack
        );

        // 4. Candidata con evaluación en fixture que FALLA criterios es rechazada y no sustituye activa.
        let weak_policy = TabularPolicy::from_entries(3, 3, &[(0, 2, 0.9)]).unwrap();
        let v3 = mem
            .register_candidate(StrategyId::new(3), weak_policy)
            .unwrap();
        let failing_stats =
            StrategyStats::new(5, 1.5, 0.50, 2, vec![EvidenceRef::new(300)]).unwrap();
        let decision_reject = mem
            .evaluate_and_promote(
                StrategyId::new(3),
                v3,
                EvaluationEvidence::new(failing_stats.clone()).with_fixture_tag("fixture_nav_v1"),
            )
            .unwrap();
        assert!(decision_reject.is_rejected());
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(2));
        assert_eq!(mem.active_strategy().version(), v2);
    }

    #[test]
    fn ac2_rollback_restores_previous_version_including_complete_strategy_stats() {
        let mut mem = base_memory();
        let v1_policy = mem.active_policy().clone();
        let v1_stats = mem.active_stats().clone();

        // Rollback sin historial previo falla limpiamente.
        assert_eq!(
            mem.rollback().expect_err("no previous version yet"),
            ProceduralError::NoPreviousVersionToRollback
        );

        // Promover v2.
        let v2_policy =
            TabularPolicy::from_entries(3, 3, &[(0, 1, 2.0), (1, 0, 1.5), (2, 1, 1.1)]).unwrap();
        let v2_stats = StrategyStats::new(
            12,
            1.4,
            0.82,
            1,
            vec![EvidenceRef::new(210), EvidenceRef::new(211)],
        )
        .unwrap();
        let v2 = mem
            .register_candidate(StrategyId::new(2), v2_policy.clone())
            .unwrap();
        let d2 = mem
            .evaluate_and_promote(
                StrategyId::new(2),
                v2,
                EvaluationEvidence::new(v2_stats.clone()),
            )
            .unwrap();
        assert!(d2.is_promoted());

        // Promover v3.
        let v3_policy =
            TabularPolicy::from_entries(3, 3, &[(0, 2, 3.0), (1, 2, 2.0), (2, 2, 1.8)]).unwrap();
        let v3_stats = StrategyStats::new(18, 1.9, 0.91, 0, vec![EvidenceRef::new(310)]).unwrap();
        let v3 = mem
            .register_candidate(StrategyId::new(3), v3_policy.clone())
            .unwrap();
        let d3 = mem
            .evaluate_and_promote(
                StrategyId::new(3),
                v3,
                EvaluationEvidence::new(v3_stats.clone()),
            )
            .unwrap();
        assert!(d3.is_promoted());
        assert_eq!(mem.rollback_depth(), 2);

        // Primer rollback: v3 -> v2 con política y estadísticas íntegras de v2.
        {
            let restored = mem.rollback().unwrap();
            assert_eq!(restored.strategy_id(), StrategyId::new(2));
            assert_eq!(restored.version(), v2);
            assert_eq!(restored.lifecycle(), StrategyLifecycle::Active);
            assert_eq!(restored.policy(), &v2_policy);
            assert_eq!(restored.stats(), &v2_stats);
            assert_eq!(restored.stats().episodes_evaluated(), 12);
            assert_eq!(restored.stats().mean_return(), 1.4);
            assert_eq!(restored.stats().holdout_score(), 0.82);
            assert_eq!(restored.stats().failures(), 1);
            assert_eq!(
                restored.stats().evidence_refs(),
                &[EvidenceRef::new(210), EvidenceRef::new(211)]
            );
        }
        assert_eq!(
            mem.get_strategy(StrategyId::new(3), v3)
                .unwrap()
                .lifecycle(),
            StrategyLifecycle::ArchivedRolledBack
        );
        assert_eq!(
            mem.get_strategy(StrategyId::new(3), v3).unwrap().stats(),
            &v3_stats
        );

        // Segundo rollback: v2 -> v1 con política y estadísticas íntegras de v1.
        {
            let restored = mem.rollback().unwrap();
            assert_eq!(restored.strategy_id(), StrategyId::new(1));
            assert_eq!(restored.version(), StrategyVersion::INITIAL);
            assert_eq!(restored.lifecycle(), StrategyLifecycle::Active);
            assert_eq!(restored.policy(), &v1_policy);
            assert_eq!(restored.stats(), &v1_stats);
            assert_eq!(restored.stats().episodes_evaluated(), 20);
            assert_eq!(restored.stats().mean_return(), 1.0);
            assert_eq!(restored.stats().holdout_score(), 0.75);
            assert_eq!(restored.stats().failures(), 2);
            assert_eq!(
                restored.stats().evidence_refs(),
                &[EvidenceRef::new(100), EvidenceRef::new(101)]
            );
        }

        // Sin más versiones previas, rollback falla sin corromper v1.
        assert_eq!(
            mem.rollback().expect_err("stack exhausted"),
            ProceduralError::NoPreviousVersionToRollback
        );
        assert_eq!(mem.active_strategy().strategy_id(), StrategyId::new(1));
        assert_eq!(mem.active_stats(), &v1_stats);
    }

    #[test]
    fn ac3_procedural_learning_does_not_alter_guard_constraints_and_negative_results_stay_visible()
    {
        let space = ActionSpace::discrete(3).unwrap();
        let safe_idle = Action::discrete_in_space(0, &space).unwrap();
        let now = Instant::now();
        let obs = Observation::new(Tick::new(5), SimDuration::ZERO, vec![0.1]).unwrap();
        let ctx =
            GuardContext::new(&space, Tick::new(5), now, Terminal::Ongoing).with_observation(&obs);

        // Restricción inviolable en ActionGuard: acción 2 está prohibida permanentemente.
        let guard = ActionGuard::with_precondition(
            SafeFallbackPolicy::ExplicitFallback(safe_idle.clone()),
            |action: &Action, _ctx: &GuardContext<'_>| -> Result<(), &'static str> {
                if action.as_discrete().map(|d| d.index()) == Some(2) {
                    return Err("action_2_is_forbidden_by_safety_invariant");
                }
                Ok(())
            },
        );

        let mut mem = base_memory();

        // 1. Registrar y rechazar una candidata con resultados negativos (múltiples incumplimientos).
        let bad_policy = TabularPolicy::from_entries(3, 3, &[(0, 1, 0.2)]).unwrap();
        let v_bad = mem
            .register_candidate(StrategyId::new(10), bad_policy)
            .unwrap();
        let negative_stats = StrategyStats::new(
            4,
            0.5,
            0.40,
            2,
            vec![EvidenceRef::new(901), EvidenceRef::new(902)],
        )
        .unwrap();
        let reject_verdict = mem
            .record_evaluation(
                StrategyId::new(10),
                v_bad,
                EvaluationEvidence::new(negative_stats.clone())
                    .with_fixture_tag("fixture_regression_check"),
            )
            .unwrap();
        assert!(!reject_verdict.is_approved());
        assert_eq!(reject_verdict.rejection_reasons().len(), 4);

        // 2. Promover una estrategia que asigna valor muy alto a la acción prohibida (acción 2).
        let aggressive_policy = TabularPolicy::from_entries(3, 3, &[(0, 2, 99.0)]).unwrap();
        let v_agg = mem
            .register_candidate(StrategyId::new(11), aggressive_policy)
            .unwrap();
        let good_stats = StrategyStats::new(15, 2.5, 0.95, 0, vec![EvidenceRef::new(910)]).unwrap();
        let promo = mem
            .evaluate_and_promote(
                StrategyId::new(11),
                v_agg,
                EvaluationEvidence::new(good_stats),
            )
            .unwrap();
        assert!(promo.is_promoted());

        // La política activa propone acción 2, pero ActionGuard mantiene inviolable su restricción.
        let (selected_action_idx, selected_val) = mem.active_policy().select_action(0).unwrap();
        assert_eq!(selected_action_idx, 2);
        assert_eq!(selected_val, 99.0);

        let proposed_action = Action::discrete_in_space(selected_action_idx, &space).unwrap();
        let proposal = ActionProposal::authorized(proposed_action, Tick::new(5));
        assert_eq!(
            guard.evaluate(&proposal, &ctx),
            GuardDecision::Fallback(safe_idle)
        );

        // 3. Incluso tras promociones posteriores y rollback, los resultados negativos siguen visibles y auditables.
        let _ = mem.rollback().unwrap();
        let rejected = mem.rejected_strategies();
        assert_eq!(rejected.len(), 1);
        let bad_rec = rejected[0];
        assert_eq!(bad_rec.strategy_id(), StrategyId::new(10));
        assert_eq!(bad_rec.version(), v_bad);
        assert_eq!(bad_rec.lifecycle(), StrategyLifecycle::EvaluatedRejected);
        assert_eq!(bad_rec.stats(), &negative_stats);
        assert_eq!(bad_rec.fixture_tag(), Some("fixture_regression_check"));
        assert_eq!(bad_rec.rejection_reasons().len(), 4);
        assert!(bad_rec.rejection_reasons().iter().any(|r| matches!(
            r,
            RejectionReason::InsufficientEpisodes {
                observed: 4,
                required: 10
            }
        )));
        assert!(
            bad_rec
                .rejection_reasons()
                .iter()
                .any(|r| matches!(r, RejectionReason::InsufficientReturnImprovement { .. }))
        );
        assert!(
            bad_rec
                .rejection_reasons()
                .iter()
                .any(|r| matches!(r, RejectionReason::FailureRateExceeded { .. }))
        );
        assert!(
            bad_rec
                .rejection_reasons()
                .iter()
                .any(|r| matches!(r, RejectionReason::HoldoutScoreBelowMinimum { .. }))
        );

        // También presentes en el log de auditoría completo.
        assert!(mem.audit_log().iter().any(|ev| matches!(
            ev,
            PromotionAuditEvent::CandidateEvaluated {
                strategy_id,
                version,
                verdict: PromotionGateVerdict::Rejected(_),
                ..
            } if *strategy_id == StrategyId::new(10) && *version == v_bad
        )));
    }

    #[test]
    fn tabular_policy_selection_is_deterministic_and_validates_inputs() {
        assert!(matches!(
            TabularPolicy::new(0, 2),
            Err(ProceduralError::InvalidStateCount { num_states: 0 })
        ));
        assert!(matches!(
            TabularPolicy::new(2, 0),
            Err(ProceduralError::InvalidActionCount { num_actions: 0 })
        ));

        let mut policy = TabularPolicy::new(2, 3).unwrap();
        assert!(matches!(
            policy.set_action_value(0, 0, f64::NAN),
            Err(ProceduralError::NonFiniteValue { .. })
        ));
        assert!(matches!(
            policy.set_action_value(2, 0, 1.0),
            Err(ProceduralError::StateOutOfRange {
                state_bucket: 2,
                num_states: 2
            })
        ));
        assert!(matches!(
            policy.set_action_value(0, 3, 1.0),
            Err(ProceduralError::ActionOutOfRange {
                action_idx: 3,
                num_actions: 3
            })
        ));

        // Empate exacto entre acción 1 y acción 2 -> desempate determinista por menor índice (1).
        policy.set_action_value(0, 1, 1.5).unwrap();
        policy.set_action_value(0, 2, 1.5).unwrap();
        assert_eq!(policy.select_action(0).unwrap(), (1, 1.5));
        assert_eq!(policy.action_values(0).unwrap(), vec![0.0, 1.5, 1.5]);
        assert_eq!(policy.select_action(1).unwrap(), (0, 0.0));
    }

    #[test]
    fn validates_stats_criteria_and_dimension_mismatch() {
        assert!(matches!(
            StrategyStats::new(5, 1.0, 0.5, 6, vec![]),
            Err(ProceduralError::InvalidStats { .. })
        ));
        assert!(matches!(
            StrategyStats::new(5, f64::INFINITY, 0.5, 0, vec![]),
            Err(ProceduralError::NonFiniteValue {
                field: "stats.mean_return",
                ..
            })
        ));
        assert!(matches!(
            EvaluationCriteria::new(0, 0.1, 0.1),
            Err(ProceduralError::InvalidCriteria { .. })
        ));
        assert!(matches!(
            EvaluationCriteria::new(5, -0.1, 0.1),
            Err(ProceduralError::InvalidCriteria { .. })
        ));
        assert!(matches!(
            EvaluationCriteria::new(5, 0.1, 1.5),
            Err(ProceduralError::InvalidCriteria { .. })
        ));

        let mut mem = base_memory();
        let wrong_dim_policy = TabularPolicy::new(2, 2).unwrap();
        assert!(matches!(
            mem.register_candidate(StrategyId::new(9), wrong_dim_policy),
            Err(ProceduralError::PolicyDimensionMismatch {
                expected_states: 3,
                expected_actions: 3,
                actual_states: 2,
                actual_actions: 2
            })
        ));
    }

    #[test]
    fn procedural_memory_checksum_is_deterministic_across_identical_runs() {
        fn build_and_run() -> u64 {
            let mut mem = base_memory();
            let p2 = TabularPolicy::from_entries(3, 3, &[(0, 1, 2.0)]).unwrap();
            let s2 = StrategyStats::new(12, 1.5, 0.85, 1, vec![EvidenceRef::new(50)]).unwrap();
            let v2 = mem.register_candidate(StrategyId::new(2), p2).unwrap();
            let _ = mem
                .evaluate_and_promote(StrategyId::new(2), v2, EvaluationEvidence::new(s2))
                .unwrap();
            mem.checksum()
        }

        assert_eq!(build_and_run(), build_and_run());
    }
}
