//! Aprendizaje incremental tabular por observación con calibración Dirichlet/Laplace (`std`-only).
//!
//! Este módulo implementa aprendizaje incremental acotado y reproducible de frecuencias
//! de transición `(state_bucket, action_idx) -> next_state_bucket` directamente desde
//! observaciones con procedencia, sin entrenamiento externo ni dependencias fuera de `std`:
//! - [`LearningPolicy`]: parámetros inmutables de suavización Dirichlet/Laplace (`prior_alpha`),
//!   cotas de actualización (`max_updates_per_pair`, `max_total_updates`) y estabilidad
//!   numérica de `log_loss`.
//! - [`ObservationId`] y [`TransitionObservation`]: unidad tipada de observación de transición
//!   vinculada a [`EvidenceRef`] y opcionalmente a `(SensorId, MessageId)` de un
//!   [`ObservationEnvelope`].
//! - [`TabularTransitionLearner`] (que implementa [`IncrementalLearner`]): mantiene conteos
//!   deterministas ordenados en `BTreeMap`, deduplica por [`EvidenceRef`], [`ObservationId`]
//!   y `(SensorId, MessageId)` devolviendo [`LearnUpdateStatus::DuplicateIgnored`], versiona
//!   cada actualización efectiva (`model_version: u64`) y produce un checksum reproducible
//!   mediante [`TraceHasher`].
//! - [`PredictionEval`] y [`HoldoutEval`]: evaluación explícita de predicción frente a
//!   resultado (`brier_score`, `log_loss`, `holdout_error`, `accuracy`).
//!
//! # Aislamiento de seguridad
//! Ni [`LearningPolicy`] ni [`TabularTransitionLearner`] reciben, almacenan o mutan
//! [`crate::ActionGuard`], [`crate::ActionSpace`], [`crate::ActionProposal`] ni permisos
//! de ejecución. El aprendizaje únicamente actualiza estimaciones probabilísticas de
//! transición y jamás amplía el conjunto de acciones autorizables.

use std::collections::{BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::belief::DiscreteBeliefState;
use crate::determinism::TraceHasher;
use crate::provenance::{EvidenceRef, MessageId, ObservationEnvelope, ObservationKind, SensorId};

/// Versión de esquema usada para hashing determinista de [`TabularTransitionLearner`].
pub const LEARNING_SCHEMA_VERSION: u32 = 1;

/// Probabilidad mínima por defecto para acotar `-ln(p)` en el cálculo de `log_loss`.
pub const DEFAULT_MIN_LOG_LOSS_PROB: f64 = 1e-12;

/// Identificador tipado único de una observación usada para aprendizaje incremental.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ObservationId(u64);

impl ObservationId {
    /// Construye un [`ObservationId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ObservationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "obs({})", self.0)
    }
}

/// Errores explícitos de validación y evaluación del módulo de aprendizaje incremental.
#[derive(Debug, Clone, PartialEq)]
pub enum LearningError {
    /// El número de buckets de estado debe ser al menos 1.
    InvalidStateCount {
        /// Valor recibido.
        num_states: usize,
    },
    /// El número de índices de acción debe ser al menos 1.
    InvalidActionCount {
        /// Valor recibido.
        num_actions: usize,
    },
    /// El parámetro `prior_alpha` de Dirichlet/Laplace debe ser finito y estrictamente positivo.
    InvalidPriorAlpha {
        /// Valor recibido.
        prior_alpha: f64,
    },
    /// El piso de probabilidad `min_log_loss_prob` debe ser finito y pertenecer a `(0.0, 1.0)`.
    InvalidMinLogLossProb {
        /// Valor recibido.
        value: f64,
    },
    /// Un límite de capacidad en [`LearningPolicy`] no puede ser `Some(0)`.
    InvalidCapacityLimit {
        /// Nombre del parámetro inválido.
        field: &'static str,
    },
    /// Un índice de estado está fuera del rango `[0, num_states)`.
    StateBucketOutOfRange {
        /// Nombre del campo (`state_bucket` o `next_state_bucket`).
        field: &'static str,
        /// Índice recibido.
        bucket: usize,
        /// Número de estados configurados.
        num_states: usize,
    },
    /// Un índice de acción está fuera del rango `[0, num_actions)`.
    ActionIndexOutOfRange {
        /// Índice de acción recibido.
        action_idx: usize,
        /// Número de acciones configuradas.
        num_actions: usize,
    },
    /// El conjunto holdout pasado a evaluación está vacío.
    EmptyHoldout,
    /// Desbordamiento aritmético en conteo de observaciones o versión de modelo.
    CounterOverflow {
        /// Contexto del desbordamiento.
        context: &'static str,
    },
}

impl fmt::Display for LearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStateCount { num_states } => {
                write!(f, "invalid state bucket count ({num_states}): must be >= 1")
            }
            Self::InvalidActionCount { num_actions } => {
                write!(f, "invalid action count ({num_actions}): must be >= 1")
            }
            Self::InvalidPriorAlpha { prior_alpha } => {
                write!(
                    f,
                    "invalid Dirichlet prior_alpha ({prior_alpha}): must be finite and > 0.0"
                )
            }
            Self::InvalidMinLogLossProb { value } => {
                write!(
                    f,
                    "invalid min_log_loss_prob ({value}): must be finite and in (0.0, 1.0)"
                )
            }
            Self::InvalidCapacityLimit { field } => {
                write!(
                    f,
                    "invalid learning policy limit '{field}': capacity cannot be Some(0)"
                )
            }
            Self::StateBucketOutOfRange {
                field,
                bucket,
                num_states,
            } => {
                write!(
                    f,
                    "{field} ({bucket}) out of range for {num_states} state buckets"
                )
            }
            Self::ActionIndexOutOfRange {
                action_idx,
                num_actions,
            } => {
                write!(
                    f,
                    "action_idx ({action_idx}) out of range for {num_actions} actions"
                )
            }
            Self::EmptyHoldout => write!(f, "holdout evaluation dataset cannot be empty"),
            Self::CounterOverflow { context } => {
                write!(f, "learning counter overflow in {context}")
            }
        }
    }
}

impl Error for LearningError {}

/// Política inmutable de aprendizaje incremental acotado.
///
/// Controla únicamente el prior de Dirichlet/Laplace, el habilitado de aprendizaje,
/// cotas máximas de actualizaciones y el piso de probabilidad para `log_loss`.
///
/// # Aislamiento de seguridad
/// `LearningPolicy` **no** tiene acceso a [`crate::ActionGuard`] ni puede convertirse en
/// una [`crate::ActionProposal`] o permiso de actuación:
///
/// ```compile_fail
/// use gogamen_core::{ActionProposal, LearningPolicy};
/// let policy = LearningPolicy::default();
/// let _proposal: ActionProposal = policy.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningPolicy {
    prior_alpha: f64,
    learning_enabled: bool,
    max_updates_per_pair: Option<u64>,
    max_total_updates: Option<u64>,
    min_log_loss_prob: f64,
    max_dedup_window: usize,
}

/// Capacidad por defecto de la ventana circular de deduplicación de [`TabularTransitionLearner`].
pub const DEFAULT_MAX_DEDUP_WINDOW: usize = 4096;

impl LearningPolicy {
    /// Crea una política con prior de Dirichlet `prior_alpha` (> 0.0, finito),
    /// aprendizaje habilitado y sin cotas de capacidad adicionales.
    pub fn new(prior_alpha: f64) -> Result<Self, LearningError> {
        if !prior_alpha.is_finite() || prior_alpha <= 0.0 {
            return Err(LearningError::InvalidPriorAlpha { prior_alpha });
        }
        Ok(Self {
            prior_alpha,
            learning_enabled: true,
            max_updates_per_pair: None,
            max_total_updates: None,
            min_log_loss_prob: DEFAULT_MIN_LOG_LOSS_PROB,
            max_dedup_window: DEFAULT_MAX_DEDUP_WINDOW,
        })
    }

    /// Política estándar con prior de Laplace (`prior_alpha = 1.0`).
    pub const fn laplace() -> Self {
        Self {
            prior_alpha: 1.0,
            learning_enabled: true,
            max_updates_per_pair: None,
            max_total_updates: None,
            min_log_loss_prob: DEFAULT_MIN_LOG_LOSS_PROB,
            max_dedup_window: DEFAULT_MAX_DEDUP_WINDOW,
        }
    }

    /// Habilita o deshabilita la actualización incremental de conteos.
    pub const fn with_learning_enabled(mut self, enabled: bool) -> Self {
        self.learning_enabled = enabled;
        self
    }

    /// Configura el máximo número de actualizaciones permitidas por par `(state_bucket, action_idx)`.
    pub fn with_max_updates_per_pair(mut self, limit: Option<u64>) -> Result<Self, LearningError> {
        if matches!(limit, Some(0)) {
            return Err(LearningError::InvalidCapacityLimit {
                field: "max_updates_per_pair",
            });
        }
        self.max_updates_per_pair = limit;
        Ok(self)
    }

    /// Configura el máximo número total de actualizaciones permitidas en el modelo.
    pub fn with_max_total_updates(mut self, limit: Option<u64>) -> Result<Self, LearningError> {
        if matches!(limit, Some(0)) {
            return Err(LearningError::InvalidCapacityLimit {
                field: "max_total_updates",
            });
        }
        self.max_total_updates = limit;
        Ok(self)
    }

    /// Configura la capacidad (> 0) de la ventana circular de deduplicación
    /// (por defecto [`DEFAULT_MAX_DEDUP_WINDOW`]).
    ///
    /// El aprendiz recuerda únicamente las últimas `limit` claves de identidad aceptadas
    /// ([`EvidenceRef`], [`ObservationId`] y `(SensorId, MessageId)`, cada una con su propia
    /// ventana); al superar la capacidad se evicta la más antigua. **Consecuencia semántica:**
    /// un duplicado cuya clave original salió de la ventana ya no se detecta y se reaprende
    /// como observación nueva. A cambio, la memoria de deduplicación es O(`limit`) en
    /// ejecuciones largas. No afecta al checksum del modelo.
    pub fn with_max_dedup_window(mut self, limit: usize) -> Result<Self, LearningError> {
        if limit == 0 {
            return Err(LearningError::InvalidCapacityLimit {
                field: "max_dedup_window",
            });
        }
        self.max_dedup_window = limit;
        Ok(self)
    }

    /// Devuelve la capacidad de la ventana circular de deduplicación.
    pub const fn max_dedup_window(&self) -> usize {
        self.max_dedup_window
    }

    /// Configura el piso de probabilidad `eps` en `(0.0, 1.0)` usado al calcular `log_loss`.
    pub fn with_min_log_loss_prob(mut self, value: f64) -> Result<Self, LearningError> {
        if !value.is_finite() || value <= 0.0 || value >= 1.0 {
            return Err(LearningError::InvalidMinLogLossProb { value });
        }
        self.min_log_loss_prob = value;
        Ok(self)
    }

    /// Devuelve el parámetro `prior_alpha` del prior de Dirichlet/Laplace.
    pub const fn prior_alpha(&self) -> f64 {
        self.prior_alpha
    }

    /// Indica si el aprendizaje incremental está habilitado.
    pub const fn learning_enabled(&self) -> bool {
        self.learning_enabled
    }

    /// Devuelve la cota máxima de actualizaciones por par `(state_bucket, action_idx)`.
    pub const fn max_updates_per_pair(&self) -> Option<u64> {
        self.max_updates_per_pair
    }

    /// Devuelve la cota máxima total de actualizaciones del modelo.
    pub const fn max_total_updates(&self) -> Option<u64> {
        self.max_total_updates
    }

    /// Devuelve el piso de probabilidad usado para acotar `log_loss`.
    pub const fn min_log_loss_prob(&self) -> f64 {
        self.min_log_loss_prob
    }
}

impl Default for LearningPolicy {
    fn default() -> Self {
        Self::laplace()
    }
}

/// Muestra discreta de transición `(state_bucket, action_idx) -> next_state_bucket`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionSample {
    state_bucket: usize,
    action_idx: usize,
    next_state_bucket: usize,
    /// `true` when the transition ended the episode in failure (additive field; `false` for
    /// every sample built with [`TransitionSample::new`]).
    terminal_failure: bool,
}

impl TransitionSample {
    /// Crea una muestra discreta de transición (no terminal).
    pub const fn new(state_bucket: usize, action_idx: usize, next_state_bucket: usize) -> Self {
        Self {
            state_bucket,
            action_idx,
            next_state_bucket,
            terminal_failure: false,
        }
    }

    /// Creates a sample whose transition ended in terminal failure; `failure_bucket` is the
    /// absorbing failure bucket reserved by the caller's bucket layout (see
    /// `learned_model::TabularScale`).
    pub const fn failure(state_bucket: usize, action_idx: usize, failure_bucket: usize) -> Self {
        Self {
            state_bucket,
            action_idx,
            next_state_bucket: failure_bucket,
            terminal_failure: true,
        }
    }

    /// Returns `true` if the observed transition was a terminal failure.
    pub const fn is_terminal_failure(&self) -> bool {
        self.terminal_failure
    }

    /// Devuelve el bucket del estado de origen.
    pub const fn state_bucket(&self) -> usize {
        self.state_bucket
    }

    /// Devuelve el índice de la acción aplicada.
    pub const fn action_idx(&self) -> usize {
        self.action_idx
    }

    /// Devuelve el bucket del estado siguiente observado.
    pub const fn next_state_bucket(&self) -> usize {
        self.next_state_bucket
    }
}

/// Observación de transición con identidad y procedencia para aprendizaje deduplicado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TransitionObservation {
    observation_id: ObservationId,
    evidence_ref: EvidenceRef,
    sensor_message: Option<(SensorId, MessageId)>,
    sample: TransitionSample,
}

impl TransitionObservation {
    /// Crea una observación de transición con [`ObservationId`] y [`EvidenceRef`] explícitos.
    pub const fn new(
        observation_id: ObservationId,
        evidence_ref: EvidenceRef,
        state_bucket: usize,
        action_idx: usize,
        next_state_bucket: usize,
    ) -> Self {
        Self {
            observation_id,
            evidence_ref,
            sensor_message: None,
            sample: TransitionSample::new(state_bucket, action_idx, next_state_bucket),
        }
    }

    /// Creates an observation of a transition that ended in terminal failure
    /// (see [`TransitionSample::failure`]).
    pub const fn failure(
        observation_id: ObservationId,
        evidence_ref: EvidenceRef,
        state_bucket: usize,
        action_idx: usize,
        failure_bucket: usize,
    ) -> Self {
        Self {
            observation_id,
            evidence_ref,
            sensor_message: None,
            sample: TransitionSample::failure(state_bucket, action_idx, failure_bucket),
        }
    }

    /// Asocia una procedencia explícita `(SensorId, MessageId)` a la observación.
    pub const fn with_sensor_message(mut self, sensor_id: SensorId, message_id: MessageId) -> Self {
        self.sensor_message = Some((sensor_id, message_id));
        self
    }

    /// Construye una [`TransitionObservation`] a partir de un [`ObservationEnvelope<TransitionSample>`].
    pub fn from_envelope(
        envelope: &ObservationEnvelope<TransitionSample>,
        observation_id: ObservationId,
    ) -> Self {
        Self {
            observation_id,
            evidence_ref: envelope.evidence_ref(),
            sensor_message: Some((envelope.sensor_id(), envelope.message_id())),
            sample: *envelope.payload(),
        }
    }

    /// Devuelve el [`ObservationId`] de esta observación.
    pub const fn observation_id(&self) -> ObservationId {
        self.observation_id
    }

    /// Devuelve la referencia de evidencia [`EvidenceRef`] asociada.
    pub const fn evidence_ref(&self) -> EvidenceRef {
        self.evidence_ref
    }

    /// Devuelve la tupla `(SensorId, MessageId)` si la observación proviene de un envelope de sensor.
    pub const fn sensor_message(&self) -> Option<(SensorId, MessageId)> {
        self.sensor_message
    }

    /// Devuelve la muestra discreta subyacente.
    pub const fn sample(&self) -> TransitionSample {
        self.sample
    }

    /// Devuelve el bucket del estado de origen.
    pub const fn state_bucket(&self) -> usize {
        self.sample.state_bucket
    }

    /// Devuelve el índice de la acción aplicada.
    pub const fn action_idx(&self) -> usize {
        self.sample.action_idx
    }

    /// Devuelve el bucket del estado siguiente observado.
    pub const fn next_state_bucket(&self) -> usize {
        self.sample.next_state_bucket
    }
}

/// Estado explícito devuelto tras intentar incorporar una observación al modelo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LearnUpdateStatus {
    /// La observación era nueva y actualizó las frecuencias y la versión del modelo.
    Updated,
    /// La observación ya había sido vista por [`EvidenceRef`], [`ObservationId`] o `(SensorId, MessageId)` y se ignoró.
    DuplicateIgnored,
    /// El aprendizaje está deshabilitado en [`LearningPolicy`]; no se actualizó el modelo.
    LearningDisabled,
    /// Se alcanzó el límite de capacidad configurado en [`LearningPolicy`]; no se actualizó el modelo.
    CapacityReached,
    /// El tipo epistémico del [`ObservationEnvelope`] (`Intention`, `Proposal`, `Unknown`) no es apto para entrenar.
    UnsupportedObservationKind,
}

impl LearnUpdateStatus {
    /// Devuelve `true` si el modelo fue actualizado.
    pub const fn is_updated(self) -> bool {
        matches!(self, Self::Updated)
    }

    /// Devuelve `true` si la observación fue ignorada por duplicado.
    pub const fn is_duplicate(self) -> bool {
        matches!(self, Self::DuplicateIgnored)
    }
}

/// Evaluación explícita de una predicción probabilística frente al estado siguiente observado.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictionEval {
    state_bucket: usize,
    action_idx: usize,
    actual_next_state: usize,
    predicted_distribution: Vec<f64>,
    predicted_prob_actual: f64,
    predicted_argmax: usize,
    brier_score: f64,
    log_loss: f64,
    hit: bool,
}

impl PredictionEval {
    /// Devuelve el estado de origen evaluado.
    pub const fn state_bucket(&self) -> usize {
        self.state_bucket
    }

    /// Devuelve la acción evaluada.
    pub const fn action_idx(&self) -> usize {
        self.action_idx
    }

    /// Devuelve el estado siguiente real observado.
    pub const fn actual_next_state(&self) -> usize {
        self.actual_next_state
    }

    /// Devuelve la distribución predicha `P(next_state | state, action)`.
    pub fn predicted_distribution(&self) -> &[f64] {
        &self.predicted_distribution
    }

    /// Devuelve la probabilidad asignada al estado siguiente real.
    pub const fn predicted_prob_actual(&self) -> f64 {
        self.predicted_prob_actual
    }

    /// Devuelve el estado con mayor probabilidad predicha (desempate determinista por menor índice).
    pub const fn predicted_argmax(&self) -> usize {
        self.predicted_argmax
    }

    /// Devuelve el Brier score multiclase `sum_k (p_k - 1_{k == actual})^2` en `[0.0, 2.0]` (menor es mejor).
    pub const fn brier_score(&self) -> f64 {
        self.brier_score
    }

    /// Devuelve la pérdida logarítmica `-ln(max(p_actual, min_log_loss_prob))` (>= 0.0, menor es mejor).
    pub const fn log_loss(&self) -> f64 {
        self.log_loss
    }

    /// Indica si `predicted_argmax == actual_next_state`.
    pub const fn hit(&self) -> bool {
        self.hit
    }
}

/// Evaluación agregada del modelo sobre un conjunto holdout de transiciones.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldoutEval {
    sample_count: usize,
    mean_brier_score: f64,
    mean_log_loss: f64,
    accuracy: f64,
    model_version: u64,
    model_checksum: u64,
}

impl HoldoutEval {
    /// Número de muestras evaluadas en el holdout.
    pub const fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Brier score medio en el holdout (menor es mejor).
    pub const fn mean_brier_score(&self) -> f64 {
        self.mean_brier_score
    }

    /// Error holdout principal del modelo (alias explícito de [`Self::mean_brier_score`]).
    pub const fn holdout_error(&self) -> f64 {
        self.mean_brier_score
    }

    /// Log-loss medio en el holdout (menor es mejor).
    pub const fn mean_log_loss(&self) -> f64 {
        self.mean_log_loss
    }

    /// Exactitud top-1 (`hits / sample_count`) en `[0.0, 1.0]`.
    pub const fn accuracy(&self) -> f64 {
        self.accuracy
    }

    /// Versión del modelo en el instante de la evaluación.
    pub const fn model_version(&self) -> u64 {
        self.model_version
    }

    /// Checksum determinista del modelo en el instante de la evaluación.
    pub const fn model_checksum(&self) -> u64 {
        self.model_checksum
    }
}

/// Resultado completo de intentar incorporar una observación al aprendiz incremental.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnOutcome {
    status: LearnUpdateStatus,
    model_version: u64,
    checksum: u64,
    pre_update_eval: Option<PredictionEval>,
}

impl LearnOutcome {
    /// Estado de la actualización ([`LearnUpdateStatus`]).
    pub const fn status(&self) -> LearnUpdateStatus {
        self.status
    }

    /// Versión del modelo tras procesar la observación.
    pub const fn model_version(&self) -> u64 {
        self.model_version
    }

    /// Checksum determinista del modelo tras procesar la observación.
    pub const fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Evaluación prequential (predicción emitida *antes* de incorporar la observación frente
    /// al resultado real) cuando la observación fue procesada para entrenamiento.
    pub const fn pre_update_eval(&self) -> Option<&PredictionEval> {
        self.pre_update_eval.as_ref()
    }
}

/// Contrato genérico de aprendizaje incremental por observación.
pub trait IncrementalLearner {
    /// Devuelve la versión monotónica actual del modelo (`0` para el prior inicial).
    fn model_version(&self) -> u64;

    /// Devuelve el checksum determinista actual del estado aprendible del modelo.
    fn checksum(&self) -> u64;

    /// Predice la distribución posterior `P(next_state | state_bucket, action_idx)`.
    fn predict(&self, state_bucket: usize, action_idx: usize) -> Result<Vec<f64>, LearningError>;

    /// Evalúa la predicción actual del modelo frente a una observación sin mutar estado.
    fn evaluate_observation(
        &self,
        observation: &TransitionObservation,
    ) -> Result<PredictionEval, LearningError>;

    /// Evalúa el modelo actual sobre un conjunto holdout de observaciones sin mutar estado.
    fn evaluate_holdout(
        &self,
        holdout: &[TransitionObservation],
    ) -> Result<HoldoutEval, LearningError>;

    /// Incorpora una observación de transición respetando deduplicación y política acotada.
    fn observe(
        &mut self,
        observation: &TransitionObservation,
    ) -> Result<LearnOutcome, LearningError>;
}

/// Aprendiz tabular incremental de frecuencias de transición con prior de Dirichlet/Laplace.
///
/// Mantiene conteos deterministas ordenados para pares `(state_bucket, action_idx) -> next_state_bucket`,
/// deduplica observaciones por [`EvidenceRef`], [`ObservationId`] y `(SensorId, MessageId)`,
/// versiona cada actualización efectiva (`model_version`) y calcula un checksum reproducible
/// mediante [`TraceHasher`].
///
/// # Aislamiento de seguridad
/// `TabularTransitionLearner` no posee referencias ni capacidad alguna para alterar
/// [`crate::ActionGuard`] ni el espacio de acciones autorizables:
///
/// ```compile_fail
/// use gogamen_core::{ActionProposal, LearningPolicy, TabularTransitionLearner};
/// let learner = TabularTransitionLearner::new(2, 2, LearningPolicy::default()).unwrap();
/// let _proposal: ActionProposal = learner.into();
/// ```
#[derive(Debug, Clone)]
pub struct TabularTransitionLearner {
    num_states: usize,
    num_actions: usize,
    policy: LearningPolicy,
    /// Tabla densa contigua `[(state * num_actions + action) * num_states + next_state]`.
    counts: Vec<u64>,
    /// Total de transiciones por par `state * num_actions + action`, actualizado en O(1).
    pair_totals: Vec<u64>,
    /// Número de pares con al menos una observación (filas "presentes" en el checksum).
    present_pairs: u64,
    /// Checksum del estado actual; se refresca únicamente tras una actualización efectiva.
    cached_checksum: u64,
    /// Contador de recomputaciones completas del checksum (solo tests).
    #[cfg(test)]
    checksum_recomputes: u64,
    seen_evidence: DedupWindow<EvidenceRef>,
    seen_observations: DedupWindow<ObservationId>,
    seen_sensor_messages: DedupWindow<(SensorId, MessageId)>,
    model_version: u64,
    total_updates: u64,
}

/// Ventana circular de deduplicación: conjunto ordenado para consulta más cola FIFO para
/// evicción de la clave más antigua cuando se supera la capacidad.
#[derive(Debug, Clone)]
struct DedupWindow<K: Ord + Copy> {
    capacity: usize,
    order: VecDeque<K>,
    set: BTreeSet<K>,
}

impl<K: Ord + Copy> DedupWindow<K> {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            order: VecDeque::new(),
            set: BTreeSet::new(),
        }
    }

    fn contains(&self, key: &K) -> bool {
        self.set.contains(key)
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    fn insert(&mut self, key: K) {
        if !self.set.insert(key) {
            return;
        }
        self.order.push_back(key);
        while self.order.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.set.remove(&oldest);
            }
        }
    }
}

impl TabularTransitionLearner {
    /// Crea un aprendiz tabular incremental inicializado únicamente con el prior configurado.
    pub fn new(
        num_states: usize,
        num_actions: usize,
        policy: LearningPolicy,
    ) -> Result<Self, LearningError> {
        if num_states == 0 {
            return Err(LearningError::InvalidStateCount { num_states });
        }
        if num_actions == 0 {
            return Err(LearningError::InvalidActionCount { num_actions });
        }
        // La tabla densa requiere `num_states^2 * num_actions` celdas; dimensiones que
        // desbordan `usize` se rechazan en lugar de abortar por asignación.
        let pairs = num_states
            .checked_mul(num_actions)
            .ok_or(LearningError::InvalidStateCount { num_states })?;
        let cells = pairs
            .checked_mul(num_states)
            .ok_or(LearningError::InvalidStateCount { num_states })?;
        let window = policy.max_dedup_window;
        let mut learner = Self {
            num_states,
            num_actions,
            policy,
            counts: vec![0_u64; cells],
            pair_totals: vec![0_u64; pairs],
            present_pairs: 0,
            cached_checksum: 0,
            #[cfg(test)]
            checksum_recomputes: 0,
            seen_evidence: DedupWindow::new(window),
            seen_observations: DedupWindow::new(window),
            seen_sensor_messages: DedupWindow::new(window),
            model_version: 0,
            total_updates: 0,
        };
        learner.refresh_checksum();
        Ok(learner)
    }

    fn pair_index(&self, state_bucket: usize, action_idx: usize) -> usize {
        state_bucket * self.num_actions + action_idx
    }

    fn row(&self, pair: usize) -> &[u64] {
        &self.counts[pair * self.num_states..(pair + 1) * self.num_states]
    }

    fn refresh_checksum(&mut self) {
        self.cached_checksum = self.compute_checksum();
        #[cfg(test)]
        {
            self.checksum_recomputes += 1;
        }
    }

    /// Crea un aprendiz tabular con prior de Laplace por defecto (`prior_alpha = 1.0`).
    pub fn with_laplace_prior(
        num_states: usize,
        num_actions: usize,
    ) -> Result<Self, LearningError> {
        Self::new(num_states, num_actions, LearningPolicy::laplace())
    }

    /// Devuelve el número de buckets de estado configurados.
    pub const fn num_states(&self) -> usize {
        self.num_states
    }

    /// Devuelve el número de índices de acción configurados.
    pub const fn num_actions(&self) -> usize {
        self.num_actions
    }

    /// Devuelve la [`LearningPolicy`] inmutable vigente.
    pub const fn policy(&self) -> LearningPolicy {
        self.policy
    }

    /// Devuelve la versión monotónica actual del modelo (`0` en el prior inicial).
    pub const fn model_version(&self) -> u64 {
        self.model_version
    }

    /// Devuelve el número total de observaciones efectivamente aprendidas.
    pub const fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Devuelve el número de referencias de evidencia únicas registradas.
    pub fn seen_evidence_count(&self) -> usize {
        self.seen_evidence.len()
    }

    /// Devuelve el número de [`ObservationId`] únicos registrados.
    pub fn seen_observation_count(&self) -> usize {
        self.seen_observations.len()
    }

    /// Indica si la observación ya fue contabilizada por cualquiera de sus claves de identidad.
    pub fn is_duplicate(&self, observation: &TransitionObservation) -> bool {
        if self.seen_evidence.contains(&observation.evidence_ref) {
            return true;
        }
        if self.seen_observations.contains(&observation.observation_id) {
            return true;
        }
        if let Some(sensor_msg) = observation.sensor_message
            && self.seen_sensor_messages.contains(&sensor_msg)
        {
            return true;
        }
        false
    }

    /// Devuelve el número de transiciones observadas para un par `(state_bucket, action_idx)`.
    pub fn pair_observation_count(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<u64, LearningError> {
        self.validate_state_action(state_bucket, action_idx)?;
        Ok(self.pair_totals[self.pair_index(state_bucket, action_idx)])
    }

    /// Devuelve el conteo crudo observado para `(state_bucket, action_idx) -> next_state_bucket`.
    pub fn transition_count(
        &self,
        state_bucket: usize,
        action_idx: usize,
        next_state_bucket: usize,
    ) -> Result<u64, LearningError> {
        self.validate_transition(state_bucket, action_idx, next_state_bucket)?;
        let pair = self.pair_index(state_bucket, action_idx);
        Ok(self.counts[pair * self.num_states + next_state_bucket])
    }

    /// Predice la distribución posterior `P(next_state | state_bucket, action_idx)`
    /// usando la media predictiva de Dirichlet-Multinomial.
    pub fn predict(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<Vec<f64>, LearningError> {
        self.validate_state_action(state_bucket, action_idx)?;
        let alpha = self.policy.prior_alpha;
        let prior_mass = alpha * (self.num_states as f64);

        let mut probs = Vec::with_capacity(self.num_states);
        let pair = self.pair_index(state_bucket, action_idx);
        let total_count = self.pair_totals[pair];
        if total_count > 0 {
            let row = self.row(pair);
            let denom = prior_mass + (total_count as f64);
            for &count in row {
                probs.push((alpha + (count as f64)) / denom);
            }
        } else {
            let uniform = 1.0 / (self.num_states as f64);
            probs.resize(self.num_states, uniform);
        }
        Ok(probs)
    }

    /// Predice la distribución posterior validada como [`DiscreteBeliefState`].
    pub fn predict_belief(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<DiscreteBeliefState, LearningError> {
        let probs = self.predict(state_bucket, action_idx)?;
        // La media predictiva de Dirichlet con prior_alpha > 0 y conteos >= 0 siempre
        // produce probabilidades finitas, estrictamente positivas y normalizadas a 1.0.
        Ok(DiscreteBeliefState::new(probs)
            .expect("Dirichlet predictive distribution must always be normalized"))
    }

    /// Evalúa la predicción actual del modelo frente a una muestra discreta sin mutar estado.
    pub fn evaluate_sample(
        &self,
        sample: TransitionSample,
    ) -> Result<PredictionEval, LearningError> {
        self.validate_transition(
            sample.state_bucket,
            sample.action_idx,
            sample.next_state_bucket,
        )?;
        let predicted = self.predict(sample.state_bucket, sample.action_idx)?;
        Ok(compute_prediction_eval(
            sample,
            predicted,
            self.policy.min_log_loss_prob,
        ))
    }

    /// Evalúa la predicción actual del modelo frente a una [`TransitionObservation`] sin mutar estado.
    pub fn evaluate_observation(
        &self,
        observation: &TransitionObservation,
    ) -> Result<PredictionEval, LearningError> {
        self.evaluate_sample(observation.sample)
    }

    /// Evalúa el modelo actual sobre un conjunto holdout de observaciones sin mutar estado.
    pub fn evaluate_holdout(
        &self,
        holdout: &[TransitionObservation],
    ) -> Result<HoldoutEval, LearningError> {
        if holdout.is_empty() {
            return Err(LearningError::EmptyHoldout);
        }

        let mut brier_sum = 0.0_f64;
        let mut log_loss_sum = 0.0_f64;
        let mut hits = 0_usize;

        for observation in holdout {
            let eval = self.evaluate_observation(observation)?;
            brier_sum += eval.brier_score;
            log_loss_sum += eval.log_loss;
            if eval.hit {
                hits += 1;
            }
        }

        let count = holdout.len() as f64;
        Ok(HoldoutEval {
            sample_count: holdout.len(),
            mean_brier_score: brier_sum / count,
            mean_log_loss: log_loss_sum / count,
            accuracy: (hits as f64) / count,
            model_version: self.model_version,
            model_checksum: self.checksum(),
        })
    }

    /// Incorpora una observación de transición aplicando validación, política acotada y deduplicación.
    ///
    /// Si la observación ya fue registrada por [`EvidenceRef`], [`ObservationId`] o
    /// `(SensorId, MessageId)`, no entrena dos veces y devuelve [`LearnUpdateStatus::DuplicateIgnored`].
    pub fn observe(
        &mut self,
        observation: &TransitionObservation,
    ) -> Result<LearnOutcome, LearningError> {
        self.validate_transition(
            observation.state_bucket(),
            observation.action_idx(),
            observation.next_state_bucket(),
        )?;

        if self.is_duplicate(observation) {
            return Ok(LearnOutcome {
                status: LearnUpdateStatus::DuplicateIgnored,
                model_version: self.model_version,
                checksum: self.checksum(),
                pre_update_eval: None,
            });
        }

        if !self.policy.learning_enabled {
            return Ok(LearnOutcome {
                status: LearnUpdateStatus::LearningDisabled,
                model_version: self.model_version,
                checksum: self.checksum(),
                pre_update_eval: None,
            });
        }

        if let Some(max_total) = self.policy.max_total_updates
            && self.total_updates >= max_total
        {
            return Ok(LearnOutcome {
                status: LearnUpdateStatus::CapacityReached,
                model_version: self.model_version,
                checksum: self.checksum(),
                pre_update_eval: None,
            });
        }

        let pair = self.pair_index(observation.state_bucket(), observation.action_idx());
        let pair_total = self.pair_totals[pair];
        if let Some(max_pair) = self.policy.max_updates_per_pair
            && pair_total >= max_pair
        {
            return Ok(LearnOutcome {
                status: LearnUpdateStatus::CapacityReached,
                model_version: self.model_version,
                checksum: self.checksum(),
                pre_update_eval: None,
            });
        }

        let next_version =
            self.model_version
                .checked_add(1)
                .ok_or(LearningError::CounterOverflow {
                    context: "model_version",
                })?;
        let next_total_updates =
            self.total_updates
                .checked_add(1)
                .ok_or(LearningError::CounterOverflow {
                    context: "total_updates",
                })?;

        let pre_update_eval = self.evaluate_observation(observation)?;

        let cell = pair * self.num_states + observation.next_state_bucket();
        let next_cell = self.counts[cell]
            .checked_add(1)
            .ok_or(LearningError::CounterOverflow {
                context: "transition_count",
            })?;
        let next_pair_total = pair_total
            .checked_add(1)
            .ok_or(LearningError::CounterOverflow {
                context: "transition_count",
            })?;
        self.counts[cell] = next_cell;
        self.pair_totals[pair] = next_pair_total;
        if pair_total == 0 {
            self.present_pairs += 1;
        }

        self.seen_evidence.insert(observation.evidence_ref);
        self.seen_observations.insert(observation.observation_id);
        if let Some(sensor_msg) = observation.sensor_message {
            self.seen_sensor_messages.insert(sensor_msg);
        }
        self.model_version = next_version;
        self.total_updates = next_total_updates;
        self.refresh_checksum();

        Ok(LearnOutcome {
            status: LearnUpdateStatus::Updated,
            model_version: self.model_version,
            checksum: self.checksum(),
            pre_update_eval: Some(pre_update_eval),
        })
    }

    /// Incorpora una observación envuelta en [`ObservationEnvelope<TransitionSample>`],
    /// aceptando únicamente reclamos epistémicos fácticos (`Measurement` y `Assertion`).
    pub fn observe_envelope(
        &mut self,
        envelope: &ObservationEnvelope<TransitionSample>,
        observation_id: ObservationId,
    ) -> Result<LearnOutcome, LearningError> {
        let observation = TransitionObservation::from_envelope(envelope, observation_id);
        self.validate_transition(
            observation.state_bucket(),
            observation.action_idx(),
            observation.next_state_bucket(),
        )?;

        match envelope.claim_kind() {
            ObservationKind::Measurement | ObservationKind::Assertion => self.observe(&observation),
            ObservationKind::Intention | ObservationKind::Proposal | ObservationKind::Unknown => {
                Ok(LearnOutcome {
                    status: LearnUpdateStatus::UnsupportedObservationKind,
                    model_version: self.model_version,
                    checksum: self.checksum(),
                    pre_update_eval: None,
                })
            }
        }
    }

    /// Calcula el checksum determinista FNV-1a de 64 bits del modelo mediante [`TraceHasher`].
    ///
    /// El orden de recorrido es estrictamente canónico (pares ordenados por `(state, action)`,
    /// únicamente los pares con observaciones), garantizando reproducibilidad determinista ante
    /// trazas equivalentes. El valor se cachea y solo se recalcula tras una actualización
    /// efectiva, por lo que esta consulta es O(1).
    pub const fn checksum(&self) -> u64 {
        self.cached_checksum
    }

    fn compute_checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.learning.tabular_transition");
        hasher.write_u64(u64::from(LEARNING_SCHEMA_VERSION));
        hasher.write_u64(self.num_states as u64);
        hasher.write_u64(self.num_actions as u64);
        hasher.write_f64(self.policy.prior_alpha);
        hasher.write_u8(u8::from(self.policy.learning_enabled));
        write_optional_u64(&mut hasher, self.policy.max_updates_per_pair);
        write_optional_u64(&mut hasher, self.policy.max_total_updates);
        hasher.write_f64(self.policy.min_log_loss_prob);
        hasher.write_u64(self.model_version);
        hasher.write_u64(self.total_updates);
        hasher.write_u64(self.present_pairs);
        for state_bucket in 0..self.num_states {
            for action_idx in 0..self.num_actions {
                let pair = self.pair_index(state_bucket, action_idx);
                if self.pair_totals[pair] == 0 {
                    continue;
                }
                let row = self.row(pair);
                hasher.write_u64(state_bucket as u64);
                hasher.write_u64(action_idx as u64);
                hasher.write_u64(row.len() as u64);
                for &count in row {
                    hasher.write_u64(count);
                }
            }
        }
        hasher.finish()
    }

    /// Devuelve [`Self::checksum`] formateado en hexadecimal de 16 dígitos.
    pub fn checksum_hex(&self) -> String {
        format!("{:016x}", self.checksum())
    }

    fn validate_state_action(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<(), LearningError> {
        if state_bucket >= self.num_states {
            return Err(LearningError::StateBucketOutOfRange {
                field: "state_bucket",
                bucket: state_bucket,
                num_states: self.num_states,
            });
        }
        if action_idx >= self.num_actions {
            return Err(LearningError::ActionIndexOutOfRange {
                action_idx,
                num_actions: self.num_actions,
            });
        }
        Ok(())
    }

    fn validate_transition(
        &self,
        state_bucket: usize,
        action_idx: usize,
        next_state_bucket: usize,
    ) -> Result<(), LearningError> {
        self.validate_state_action(state_bucket, action_idx)?;
        if next_state_bucket >= self.num_states {
            return Err(LearningError::StateBucketOutOfRange {
                field: "next_state_bucket",
                bucket: next_state_bucket,
                num_states: self.num_states,
            });
        }
        Ok(())
    }
}

impl IncrementalLearner for TabularTransitionLearner {
    fn model_version(&self) -> u64 {
        Self::model_version(self)
    }

    fn checksum(&self) -> u64 {
        Self::checksum(self)
    }

    fn predict(&self, state_bucket: usize, action_idx: usize) -> Result<Vec<f64>, LearningError> {
        Self::predict(self, state_bucket, action_idx)
    }

    fn evaluate_observation(
        &self,
        observation: &TransitionObservation,
    ) -> Result<PredictionEval, LearningError> {
        Self::evaluate_observation(self, observation)
    }

    fn evaluate_holdout(
        &self,
        holdout: &[TransitionObservation],
    ) -> Result<HoldoutEval, LearningError> {
        Self::evaluate_holdout(self, holdout)
    }

    fn observe(
        &mut self,
        observation: &TransitionObservation,
    ) -> Result<LearnOutcome, LearningError> {
        Self::observe(self, observation)
    }
}

fn write_optional_u64(hasher: &mut TraceHasher, value: Option<u64>) {
    match value {
        Some(v) => {
            hasher.write_u8(1);
            hasher.write_u64(v);
        }
        None => {
            hasher.write_u8(0);
        }
    }
}

fn compute_prediction_eval(
    sample: TransitionSample,
    predicted_distribution: Vec<f64>,
    min_log_loss_prob: f64,
) -> PredictionEval {
    let actual = sample.next_state_bucket;
    let predicted_prob_actual = predicted_distribution[actual];

    let mut predicted_argmax = 0_usize;
    let mut best_prob = f64::NEG_INFINITY;
    let mut brier_score = 0.0_f64;

    for (idx, &prob) in predicted_distribution.iter().enumerate() {
        if prob > best_prob {
            best_prob = prob;
            predicted_argmax = idx;
        }
        let target = if idx == actual { 1.0 } else { 0.0 };
        let diff = prob - target;
        brier_score += diff * diff;
    }

    let clamped_prob = predicted_prob_actual.max(min_log_loss_prob);
    let log_loss = -clamped_prob.ln();
    let hit = predicted_argmax == actual;

    PredictionEval {
        state_bucket: sample.state_bucket,
        action_idx: sample.action_idx,
        actual_next_state: actual,
        predicted_distribution,
        predicted_prob_actual,
        predicted_argmax,
        brier_score,
        log_loss,
        hit,
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::action::{Action, ActionSpace, DiscreteAction};
    use crate::clock::MonotonicInstant;
    use crate::domain::{Observation, Terminal};
    use crate::guard::{
        ActionGuard, ActionProposal, DomainPrecondition, GuardContext, GuardDecision,
        GuardRejectionReason, SafeFallbackPolicy,
    };
    use crate::provenance::{Confidence, FrameId, ProvenanceMetadata, UnitTag};
    use crate::units::{SimDuration, Tick};

    /// Fixture determinista: genera observaciones de entrenamiento y holdout a partir de
    /// frecuencias de transición fijadas para 3 estados y 2 acciones.
    fn build_transition_fixture() -> (Vec<TransitionObservation>, Vec<TransitionObservation>) {
        // Patrón determinista por (state, action):
        // (0, 0) -> 90% estado 1, 10% estado 2
        // (0, 1) -> 85% estado 2, 15% estado 0
        // (1, 0) -> 80% estado 0, 20% estado 2
        // (1, 1) -> 90% estado 2, 10% estado 1
        // (2, 0) -> 85% estado 0, 15% estado 1
        // (2, 1) -> 80% estado 1, 20% estado 0
        let specs: &[(usize, usize, usize, usize, usize)] = &[
            (0, 0, 1, 2, 18), // 18 veces mayoritario (1), 2 veces minoritario (2)
            (0, 1, 2, 0, 17), // 17 veces 2, 3 veces 0
            (1, 0, 0, 2, 16), // 16 veces 0, 4 veces 2
            (1, 1, 2, 1, 18), // 18 veces 2, 2 veces 1
            (2, 0, 0, 1, 17), // 17 veces 0, 3 veces 1
            (2, 1, 1, 0, 16), // 16 veces 1, 4 veces 0
        ];

        let mut train = Vec::new();
        let mut holdout = Vec::new();
        let mut next_id = 1_u64;

        for &(s, a, major_next, minor_next, major_count) in specs {
            for i in 0..20_usize {
                let next_s = if i < major_count {
                    major_next
                } else {
                    minor_next
                };
                train.push(TransitionObservation::new(
                    ObservationId::new(next_id),
                    EvidenceRef::new(10_000 + next_id),
                    s,
                    a,
                    next_s,
                ));
                next_id += 1;
            }

            // Holdout fijado con la misma distribución aproximada (10 muestras por par)
            let holdout_major = major_count / 2;
            for i in 0..10_usize {
                let next_s = if i < holdout_major {
                    major_next
                } else {
                    minor_next
                };
                holdout.push(TransitionObservation::new(
                    ObservationId::new(next_id),
                    EvidenceRef::new(10_000 + next_id),
                    s,
                    a,
                    next_s,
                ));
                next_id += 1;
            }
        }

        (train, holdout)
    }

    #[test]
    fn fixture_learns_transition_frequencies_and_improves_holdout_error_over_prior() {
        let (train, holdout) = build_transition_fixture();
        let mut learner = TabularTransitionLearner::with_laplace_prior(3, 2).unwrap();

        assert_eq!(learner.model_version(), 0);
        let prior_eval = learner.evaluate_holdout(&holdout).unwrap();
        assert_eq!(prior_eval.sample_count(), holdout.len());
        assert_eq!(prior_eval.model_version(), 0);

        // Con prior uniforme sobre 3 estados: p = [1/3, 1/3, 1/3], Brier = (2/3)^2 + 2*(1/3)^2 = 2/3.
        assert!((prior_eval.holdout_error() - (2.0 / 3.0)).abs() < 1e-12);
        assert!((prior_eval.mean_log_loss() - 3.0_f64.ln()).abs() < 1e-12);

        for obs in &train {
            let outcome = learner.observe(obs).unwrap();
            assert_eq!(outcome.status(), LearnUpdateStatus::Updated);
            assert!(outcome.pre_update_eval().is_some());
        }

        assert_eq!(learner.model_version(), train.len() as u64);
        assert_eq!(learner.total_updates(), train.len() as u64);

        // Verificar frecuencias aprendidas para (0, 0): 18 hacia estado 1, 2 hacia estado 2, 0 hacia estado 0
        // Con Laplace prior (alpha=1.0, K=3): denom = 3 + 20 = 23.
        let pred_0_0 = learner.predict(0, 0).unwrap();
        assert!((pred_0_0[0] - (1.0 / 23.0)).abs() < 1e-12);
        assert!((pred_0_0[1] - (19.0 / 23.0)).abs() < 1e-12);
        assert!((pred_0_0[2] - (3.0 / 23.0)).abs() < 1e-12);

        // Verificar conversión a DiscreteBeliefState normalizado.
        let belief_0_0 = learner.predict_belief(0, 0).unwrap();
        assert_eq!(belief_0_0.num_states(), 3);
        assert!((belief_0_0.probabilities()[1] - (19.0 / 23.0)).abs() < 1e-12);

        // Verificar mejora sustancial en holdout frente al prior.
        let trained_eval = learner.evaluate_holdout(&holdout).unwrap();
        assert_eq!(trained_eval.model_version(), train.len() as u64);
        assert!(
            trained_eval.holdout_error() < prior_eval.holdout_error() * 0.5,
            "expected holdout Brier error to drop significantly: trained={}, prior={}",
            trained_eval.holdout_error(),
            prior_eval.holdout_error()
        );
        assert!(
            trained_eval.mean_log_loss() < prior_eval.mean_log_loss() * 0.6,
            "expected holdout log-loss to drop significantly: trained={}, prior={}",
            trained_eval.mean_log_loss(),
            prior_eval.mean_log_loss()
        );
        assert!(trained_eval.accuracy() > prior_eval.accuracy());
        assert!(trained_eval.accuracy() >= 0.80);
    }

    #[test]
    fn duplicate_observation_never_trains_twice_and_updates_are_versioned_and_reproducible() {
        let mut learner_a = TabularTransitionLearner::with_laplace_prior(3, 2).unwrap();
        let mut learner_b = TabularTransitionLearner::with_laplace_prior(3, 2).unwrap();

        let obs1 =
            TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(101), 0, 0, 1)
                .with_sensor_message(SensorId::new(7), MessageId::new(50));
        let obs2 =
            TransitionObservation::new(ObservationId::new(2), EvidenceRef::new(102), 0, 0, 2)
                .with_sensor_message(SensorId::new(7), MessageId::new(51));

        // 1. Primera observación actualiza versión 0 -> 1.
        let out1 = learner_a.observe(&obs1).unwrap();
        assert_eq!(out1.status(), LearnUpdateStatus::Updated);
        assert_eq!(out1.model_version(), 1);
        let checksum_v1 = out1.checksum();
        let pred_v1 = learner_a.predict(0, 0).unwrap();

        // 2. Duplicado exacto: ignorado, sin cambios en versión, checksum ni predicciones.
        let dup_exact = learner_a.observe(&obs1).unwrap();
        assert_eq!(dup_exact.status(), LearnUpdateStatus::DuplicateIgnored);
        assert!(dup_exact.status().is_duplicate());
        assert_eq!(dup_exact.model_version(), 1);
        assert_eq!(dup_exact.checksum(), checksum_v1);
        assert!(dup_exact.pre_update_eval().is_none());
        assert_eq!(learner_a.predict(0, 0).unwrap(), pred_v1);

        // 3. Mismo EvidenceRef con distinto ObservationId y distinto (SensorId, MessageId): ignorado.
        let dup_evidence =
            TransitionObservation::new(ObservationId::new(999), EvidenceRef::new(101), 0, 0, 1)
                .with_sensor_message(SensorId::new(8), MessageId::new(1));
        let out_dup_ev = learner_a.observe(&dup_evidence).unwrap();
        assert_eq!(out_dup_ev.status(), LearnUpdateStatus::DuplicateIgnored);
        assert_eq!(learner_a.model_version(), 1);
        assert_eq!(learner_a.checksum(), checksum_v1);

        // 4. Mismo ObservationId con distinto EvidenceRef: ignorado.
        let dup_obs_id =
            TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(999), 0, 0, 1);
        let out_dup_id = learner_a.observe(&dup_obs_id).unwrap();
        assert_eq!(out_dup_id.status(), LearnUpdateStatus::DuplicateIgnored);
        assert_eq!(learner_a.model_version(), 1);
        assert_eq!(learner_a.checksum(), checksum_v1);

        // 5. Mismo (SensorId, MessageId) con distinto ObservationId y EvidenceRef: ignorado.
        let dup_sensor_msg =
            TransitionObservation::new(ObservationId::new(888), EvidenceRef::new(888), 0, 0, 1)
                .with_sensor_message(SensorId::new(7), MessageId::new(50));
        let out_dup_sm = learner_a.observe(&dup_sensor_msg).unwrap();
        assert_eq!(out_dup_sm.status(), LearnUpdateStatus::DuplicateIgnored);
        assert_eq!(learner_a.model_version(), 1);
        assert_eq!(learner_a.checksum(), checksum_v1);

        // 6. Segunda observación única avanza versión 1 -> 2.
        let out2 = learner_a.observe(&obs2).unwrap();
        assert_eq!(out2.status(), LearnUpdateStatus::Updated);
        assert_eq!(out2.model_version(), 2);
        assert_ne!(out2.checksum(), checksum_v1);

        // Reproducibilidad: learner_b recibe solo obs1 y obs2 sin duplicados -> estado bit-a-bit idéntico.
        assert_eq!(
            learner_b.observe(&obs1).unwrap().status(),
            LearnUpdateStatus::Updated
        );
        assert_eq!(
            learner_b.observe(&obs2).unwrap().status(),
            LearnUpdateStatus::Updated
        );
        assert_eq!(learner_a.model_version(), learner_b.model_version());
        assert_eq!(learner_a.checksum(), learner_b.checksum());
        assert_eq!(learner_a.checksum_hex(), learner_b.checksum_hex());
        assert_eq!(
            learner_a.predict(0, 0).unwrap(),
            learner_b.predict(0, 0).unwrap()
        );
        assert_eq!(learner_a.transition_count(0, 0, 1).unwrap(), 1);
        assert_eq!(learner_a.transition_count(0, 0, 2).unwrap(), 1);
    }

    struct OnlyActionZeroAllowed;

    impl DomainPrecondition for OnlyActionZeroAllowed {
        fn check(&self, action: &Action, _context: &GuardContext<'_>) -> Result<(), &'static str> {
            match action {
                Action::Discrete(discrete) if discrete.index() == 0 => Ok(()),
                _ => Err("only discrete action 0 is authorized by safety guard"),
            }
        }
    }

    #[test]
    fn learning_policy_never_changes_guard_or_authorizable_actions() {
        let action_space = ActionSpace::discrete(2).unwrap();
        let guard =
            ActionGuard::with_precondition(SafeFallbackPolicy::RejectOnly, OnlyActionZeroAllowed);
        let now = Instant::now();
        let obs = Observation::new(Tick::new(5), SimDuration::ZERO, vec![1.0]).unwrap();
        let ctx = GuardContext::new(&action_space, Tick::new(5), now, Terminal::Ongoing)
            .with_observation(&obs);

        let action_0 = Action::Discrete(DiscreteAction::new(0, 2).unwrap());
        let action_1 = Action::Discrete(DiscreteAction::new(1, 2).unwrap());
        let action_out_of_space = Action::Discrete(DiscreteAction::new(2, 3).unwrap());

        let proposals = [
            ActionProposal::authorized(action_0.clone(), Tick::new(5)),
            ActionProposal::authorized(action_1, Tick::new(5)),
            ActionProposal::authorized(action_out_of_space, Tick::new(5)),
            ActionProposal::unauthorized(Tick::new(5)),
        ];

        let decisions_before: Vec<GuardDecision> = proposals
            .iter()
            .map(|proposal| guard.evaluate(proposal, &ctx))
            .collect();

        assert_eq!(decisions_before[0], GuardDecision::Allow(action_0));
        assert_eq!(
            decisions_before[1],
            GuardDecision::Reject(GuardRejectionReason::PreconditionFailed {
                reason: "only discrete action 0 is authorized by safety guard",
            })
        );
        assert!(matches!(
            decisions_before[2],
            GuardDecision::Reject(GuardRejectionReason::InvalidAction(_))
        ));
        assert_eq!(
            decisions_before[3],
            GuardDecision::Reject(GuardRejectionReason::Unauthorized)
        );

        // Entrenar intensivamente el aprendiz con observaciones sobre acción 1 (prohibida por el guard)
        // y probar con distinta LearningPolicy.
        let aggressive_policy = LearningPolicy::new(0.25).unwrap();
        let mut learner = TabularTransitionLearner::new(3, 2, aggressive_policy).unwrap();
        for i in 1..=50_u64 {
            let sample = TransitionObservation::new(
                ObservationId::new(i),
                EvidenceRef::new(500 + i),
                0,
                1,
                2,
            );
            assert_eq!(
                learner.observe(&sample).unwrap().status(),
                LearnUpdateStatus::Updated
            );
        }
        assert!(learner.predict(0, 1).unwrap()[2] > 0.95);

        // Reevaluar exactamente las mismas propuestas con el mismo ActionGuard: cero cambios.
        let decisions_after: Vec<GuardDecision> = proposals
            .iter()
            .map(|proposal| guard.evaluate(proposal, &ctx))
            .collect();
        assert_eq!(decisions_before, decisions_after);
    }

    #[test]
    fn envelope_filters_non_factual_observation_kinds() {
        let mut learner = TabularTransitionLearner::with_laplace_prior(2, 2).unwrap();
        let make_env = |msg_id: u64, ev_id: u64, kind: ObservationKind| {
            let metadata = ProvenanceMetadata {
                message_id: MessageId::new(msg_id),
                observed_at: MonotonicInstant::from_nanos(u128::from(100 * msg_id)),
                received_at: MonotonicInstant::from_nanos(u128::from(100 * msg_id + 5)),
                sensor_id: SensorId::new(1),
                frame_id: FrameId::world(),
                units: UnitTag::dimensionless(),
                confidence: Confidence::FULL,
                revision: 1,
                evidence_ref: EvidenceRef::new(ev_id),
                claim_kind: kind,
            };
            ObservationEnvelope::new(metadata, TransitionSample::new(0, 0, 1))
        };

        for (idx, kind) in [
            ObservationKind::Intention,
            ObservationKind::Proposal,
            ObservationKind::Unknown,
        ]
        .into_iter()
        .enumerate()
        {
            let env = make_env((idx + 1) as u64, (100 + idx) as u64, kind);
            let out = learner
                .observe_envelope(&env, ObservationId::new((idx + 1) as u64))
                .unwrap();
            assert_eq!(out.status(), LearnUpdateStatus::UnsupportedObservationKind);
            assert_eq!(learner.model_version(), 0);
        }

        let env_measurement = make_env(10, 200, ObservationKind::Measurement);
        let out_m = learner
            .observe_envelope(&env_measurement, ObservationId::new(10))
            .unwrap();
        assert_eq!(out_m.status(), LearnUpdateStatus::Updated);
        assert_eq!(learner.model_version(), 1);

        let env_assertion = make_env(11, 201, ObservationKind::Assertion);
        let out_a = learner
            .observe_envelope(&env_assertion, ObservationId::new(11))
            .unwrap();
        assert_eq!(out_a.status(), LearnUpdateStatus::Updated);
        assert_eq!(learner.model_version(), 2);
    }

    #[test]
    fn bounded_policy_and_disabled_learning_are_enforced() {
        let disabled_policy = LearningPolicy::laplace().with_learning_enabled(false);
        let mut disabled_learner = TabularTransitionLearner::new(2, 2, disabled_policy).unwrap();
        let obs = TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(1), 0, 0, 1);
        let out = disabled_learner.observe(&obs).unwrap();
        assert_eq!(out.status(), LearnUpdateStatus::LearningDisabled);
        assert_eq!(disabled_learner.model_version(), 0);

        let bounded_policy = LearningPolicy::laplace()
            .with_max_updates_per_pair(Some(2))
            .unwrap()
            .with_max_total_updates(Some(3))
            .unwrap();
        let mut bounded = TabularTransitionLearner::new(2, 2, bounded_policy).unwrap();

        for i in 1..=2_u64 {
            let o = TransitionObservation::new(ObservationId::new(i), EvidenceRef::new(i), 0, 0, 1);
            assert_eq!(
                bounded.observe(&o).unwrap().status(),
                LearnUpdateStatus::Updated
            );
        }
        // Tercer intento en par (0, 0) alcanza max_updates_per_pair=2.
        let o3 = TransitionObservation::new(ObservationId::new(3), EvidenceRef::new(3), 0, 0, 1);
        assert_eq!(
            bounded.observe(&o3).unwrap().status(),
            LearnUpdateStatus::CapacityReached
        );
        assert_eq!(bounded.model_version(), 2);

        // Otro par (0, 1) sí puede entrenar hasta max_total_updates=3.
        let o4 = TransitionObservation::new(ObservationId::new(4), EvidenceRef::new(4), 0, 1, 0);
        assert_eq!(
            bounded.observe(&o4).unwrap().status(),
            LearnUpdateStatus::Updated
        );
        assert_eq!(bounded.model_version(), 3);

        // Cuarta actualización global alcanza max_total_updates=3.
        let o5 = TransitionObservation::new(ObservationId::new(5), EvidenceRef::new(5), 1, 0, 0);
        assert_eq!(
            bounded.observe(&o5).unwrap().status(),
            LearnUpdateStatus::CapacityReached
        );
        assert_eq!(bounded.model_version(), 3);
    }

    #[test]
    fn validates_dimensions_policy_and_out_of_range_indices() {
        assert_eq!(
            TabularTransitionLearner::with_laplace_prior(0, 2).unwrap_err(),
            LearningError::InvalidStateCount { num_states: 0 }
        );
        assert_eq!(
            TabularTransitionLearner::with_laplace_prior(2, 0).unwrap_err(),
            LearningError::InvalidActionCount { num_actions: 0 }
        );
        assert!(matches!(
            LearningPolicy::new(0.0),
            Err(LearningError::InvalidPriorAlpha { .. })
        ));
        assert!(matches!(
            LearningPolicy::new(f64::NAN),
            Err(LearningError::InvalidPriorAlpha { .. })
        ));
        assert!(matches!(
            LearningPolicy::laplace().with_max_updates_per_pair(Some(0)),
            Err(LearningError::InvalidCapacityLimit {
                field: "max_updates_per_pair"
            })
        ));
        assert!(matches!(
            LearningPolicy::laplace().with_min_log_loss_prob(0.0),
            Err(LearningError::InvalidMinLogLossProb { .. })
        ));

        let mut learner = TabularTransitionLearner::with_laplace_prior(2, 2).unwrap();
        assert_eq!(
            learner.evaluate_holdout(&[]).unwrap_err(),
            LearningError::EmptyHoldout
        );

        let bad_state =
            TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(1), 2, 0, 0);
        assert_eq!(
            learner.observe(&bad_state).unwrap_err(),
            LearningError::StateBucketOutOfRange {
                field: "state_bucket",
                bucket: 2,
                num_states: 2,
            }
        );
        // Un error de validación no contamina el deduplicador ni la versión.
        assert_eq!(learner.model_version(), 0);
        assert_eq!(learner.seen_evidence_count(), 0);
    }

    #[test]
    fn incremental_learner_trait_works_generically() {
        fn step_and_eval<L: IncrementalLearner>(
            learner: &mut L,
            obs: &TransitionObservation,
        ) -> (LearnOutcome, PredictionEval) {
            let outcome = learner.observe(obs).unwrap();
            let post_eval = learner.evaluate_observation(obs).unwrap();
            (outcome, post_eval)
        }

        let mut learner = TabularTransitionLearner::with_laplace_prior(2, 1).unwrap();
        let obs = TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(1), 0, 0, 1);
        let (outcome, post_eval) = step_and_eval(&mut learner, &obs);
        assert_eq!(outcome.status(), LearnUpdateStatus::Updated);
        assert_eq!(learner.model_version(), 1);
        assert!(
            post_eval.brier_score() < outcome.pre_update_eval().unwrap().brier_score(),
            "post-update Brier score should improve on the observed transition"
        );
    }

    /// Referencia: recomputa el checksum con el algoritmo histórico basado en `BTreeMap`.
    fn reference_checksum(
        learner: &TabularTransitionLearner,
        model: &std::collections::BTreeMap<(usize, usize), Vec<u64>>,
    ) -> u64 {
        let policy = learner.policy();
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.learning.tabular_transition");
        hasher.write_u64(u64::from(LEARNING_SCHEMA_VERSION));
        hasher.write_u64(learner.num_states() as u64);
        hasher.write_u64(learner.num_actions() as u64);
        hasher.write_f64(policy.prior_alpha());
        hasher.write_u8(u8::from(policy.learning_enabled()));
        write_optional_u64(&mut hasher, policy.max_updates_per_pair());
        write_optional_u64(&mut hasher, policy.max_total_updates());
        hasher.write_f64(policy.min_log_loss_prob());
        hasher.write_u64(learner.model_version());
        hasher.write_u64(learner.total_updates());
        hasher.write_u64(model.len() as u64);
        for (&(s, a), row) in model {
            hasher.write_u64(s as u64);
            hasher.write_u64(a as u64);
            hasher.write_u64(row.len() as u64);
            for &c in row {
                hasher.write_u64(c);
            }
        }
        hasher.finish()
    }

    #[test]
    fn flat_table_checksum_matches_btreemap_reference_and_pair_totals_consistent() {
        let (ns, na) = (5_usize, 3_usize);
        let mut learner = TabularTransitionLearner::with_laplace_prior(ns, na).unwrap();
        let mut model = std::collections::BTreeMap::<(usize, usize), Vec<u64>>::new();
        assert_eq!(learner.checksum(), reference_checksum(&learner, &model));

        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for i in 1..=300_u64 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let s = (x % ns as u64) as usize;
            let a = ((x >> 16) % na as u64) as usize;
            let n = ((x >> 32) % ns as u64) as usize;
            let obs =
                TransitionObservation::new(ObservationId::new(i), EvidenceRef::new(i), s, a, n);
            let out = learner.observe(&obs).unwrap();
            assert_eq!(out.status(), LearnUpdateStatus::Updated);
            model.entry((s, a)).or_insert_with(|| vec![0; ns])[n] += 1;
            assert_eq!(out.checksum(), reference_checksum(&learner, &model));
            assert_eq!(learner.checksum(), out.checksum());
        }
        for s in 0..ns {
            for a in 0..na {
                let row_sum: u64 = (0..ns)
                    .map(|n| learner.transition_count(s, a, n).unwrap())
                    .sum();
                assert_eq!(learner.pair_observation_count(s, a).unwrap(), row_sum);
            }
        }
    }

    #[test]
    fn non_updating_paths_return_cached_checksum_without_recompute() {
        let policy = LearningPolicy::laplace()
            .with_max_total_updates(Some(1))
            .unwrap();
        let mut learner = TabularTransitionLearner::new(2, 2, policy).unwrap();
        let first = TransitionObservation::new(ObservationId::new(1), EvidenceRef::new(1), 0, 0, 1);
        let updated = learner.observe(&first).unwrap();
        let recomputes = learner.checksum_recomputes;

        let dup = learner.observe(&first).unwrap();
        assert_eq!(dup.status(), LearnUpdateStatus::DuplicateIgnored);
        let other = TransitionObservation::new(ObservationId::new(2), EvidenceRef::new(2), 1, 1, 0);
        let capped = learner.observe(&other).unwrap();
        assert_eq!(capped.status(), LearnUpdateStatus::CapacityReached);
        assert_eq!(dup.checksum(), updated.checksum());
        assert_eq!(capped.checksum(), updated.checksum());
        assert_eq!(learner.checksum(), updated.checksum());
        assert_eq!(learner.checksum_recomputes, recomputes);

        let mut disabled = TabularTransitionLearner::new(
            2,
            2,
            LearningPolicy::laplace().with_learning_enabled(false),
        )
        .unwrap();
        let before = disabled.checksum();
        let recomputes = disabled.checksum_recomputes;
        let out = disabled.observe(&first).unwrap();
        assert_eq!(out.status(), LearnUpdateStatus::LearningDisabled);
        assert_eq!(out.checksum(), before);
        assert_eq!(disabled.checksum_recomputes, recomputes);
    }

    #[test]
    fn dedup_window_is_bounded_and_evicts_oldest() {
        let policy = LearningPolicy::laplace().with_max_dedup_window(3).unwrap();
        assert_eq!(policy.max_dedup_window(), 3);
        assert!(LearningPolicy::laplace().with_max_dedup_window(0).is_err());
        assert_eq!(
            LearningPolicy::default().max_dedup_window(),
            DEFAULT_MAX_DEDUP_WINDOW
        );
        let mut learner = TabularTransitionLearner::new(2, 1, policy).unwrap();
        let mk = |i: u64| {
            TransitionObservation::new(ObservationId::new(i), EvidenceRef::new(i), 0, 0, 1)
                .with_sensor_message(SensorId::new(1), MessageId::new(i))
        };
        for i in 1..=10 {
            assert_eq!(
                learner.observe(&mk(i)).unwrap().status(),
                LearnUpdateStatus::Updated
            );
            assert!(learner.seen_evidence_count() <= 3);
            assert!(learner.seen_observation_count() <= 3);
            assert!(learner.seen_sensor_messages.len() <= 3);
        }
        // Dentro de la ventana (8, 9, 10): duplicado ignorado.
        assert_eq!(
            learner.observe(&mk(10)).unwrap().status(),
            LearnUpdateStatus::DuplicateIgnored
        );
        assert_eq!(
            learner.observe(&mk(8)).unwrap().status(),
            LearnUpdateStatus::DuplicateIgnored
        );
        // Fuera de la ventana: reaceptado.
        let before = learner.total_updates();
        assert_eq!(
            learner.observe(&mk(1)).unwrap().status(),
            LearnUpdateStatus::Updated
        );
        assert_eq!(learner.total_updates(), before + 1);
    }
}
