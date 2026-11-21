//! Creencias discretas y actualización bayesiana calibrada (`std`-only).
//!
//! Este módulo implementa representaciones de creencias discretas normalizadas y su
//! actualización bayesiana calibrada a partir de observaciones con procedencia
//! ([`ObservationEnvelope`]):
//! - [`DiscreteBeliefState`] (y su alias [`CategoricalBelief`]) y [`BinaryBelief`]:
//!   distribuciones discretas normalizadas (`sum == 1.0`, finitas y no negativas).
//! - [`LikelihoodModel`]: matriz estocástica explícita `P(observación | estado)` o
//!   sensibilidad/especificidad explícitas, **calibradas independientemente** de
//!   [`ObservationEnvelope::confidence`](crate::provenance::ObservationEnvelope::confidence).
//! - [`EvidenceDeduplicator`]: deduplicación de rumor u origen compartido por
//!   [`EvidenceRef`] y `(SensorId, MessageId)` para evitar doble conteo de evidencia.
//! - [`DiscreteBeliefFilter`]: filtro bayesiano discreto que aplica actualización
//!   calibrada y produce estados explícitos ([`BeliefUpdateStatus`]) ante duplicados,
//!   observaciones imposibles (`P(obs) == 0.0`), mensajes desactualizados
//!   (`observed_at < current_watermark`) o tipos epistémicos no soportados
//!   (`ObservationKind::Unknown`, `Intention`, `Proposal`) sin corromper la creencia previa.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use crate::clock::MonotonicInstant;
use crate::entropy::DEFAULT_PROBABILITY_SUM_TOLERANCE;
use crate::provenance::{EvidenceRef, MessageId, ObservationEnvelope, ObservationKind, SensorId};

/// Errores explícitos de validación o incompatibilidad estructural en creencias discretas.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefError {
    /// La distribución de creencia está vacía.
    EmptyDistribution,
    /// Una probabilidad o verosimilitud no es finita (`NaN`, `+inf`, `-inf`).
    NonFiniteProbability {
        /// Índice de la entrada inválida.
        index: usize,
        /// Valor no finito recibido.
        value: f64,
    },
    /// Una probabilidad o verosimilitud es estrictamente negativa (`< 0.0`).
    NegativeProbability {
        /// Índice de la entrada negativa.
        index: usize,
        /// Valor negativo recibido.
        value: f64,
    },
    /// La suma de probabilidades difiere de `1.0` más allá de la tolerancia permitida.
    ProbabilitySumMismatch {
        /// Suma observada.
        sum: f64,
        /// Tolerancia máxima permitida.
        tolerance: f64,
    },
    /// Un parámetro escalar de probabilidad (`p_true`, `sensitivity`, `specificity`) está fuera de `[0.0, 1.0]` o no es finito.
    InvalidProbability {
        /// Nombre del parámetro inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// La matriz de verosimilitud tiene dimensiones inválidas o inconsistentes.
    InvalidLikelihoodMatrix {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// El número de estados de la creencia no coincide con el del [`LikelihoodModel`].
    StateCountMismatch {
        /// Número de estados en la creencia.
        belief_states: usize,
        /// Número de estados en el modelo de verosimilitud.
        model_states: usize,
    },
    /// El índice de resultado observado excede el número de resultados del [`LikelihoodModel`].
    OutcomeOutOfRange {
        /// Índice de resultado observado.
        outcome: usize,
        /// Número de resultados definidos en el modelo.
        num_outcomes: usize,
    },
    /// Se intentó convertir a [`BinaryBelief`] una distribución que no tiene exactamente 2 estados.
    NotBinaryDistribution {
        /// Número real de estados de la distribución.
        actual_states: usize,
    },
}

impl fmt::Display for BeliefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => write!(f, "belief distribution cannot be empty"),
            Self::NonFiniteProbability { index, value } => {
                write!(f, "non-finite probability at index {index}: {value}")
            }
            Self::NegativeProbability { index, value } => {
                write!(f, "negative probability at index {index}: {value}")
            }
            Self::ProbabilitySumMismatch { sum, tolerance } => {
                write!(
                    f,
                    "belief probability sum mismatch: sum {sum} differs from 1.0 by more than {tolerance}"
                )
            }
            Self::InvalidProbability { field, value } => {
                write!(
                    f,
                    "invalid probability '{field}' ({value}): must be finite and within [0.0, 1.0]"
                )
            }
            Self::InvalidLikelihoodMatrix { reason } => {
                write!(f, "invalid likelihood matrix: {reason}")
            }
            Self::StateCountMismatch {
                belief_states,
                model_states,
            } => {
                write!(
                    f,
                    "state count mismatch: belief has {belief_states} states, model expects {model_states}"
                )
            }
            Self::OutcomeOutOfRange {
                outcome,
                num_outcomes,
            } => {
                write!(
                    f,
                    "observation outcome index {outcome} out of range for {num_outcomes} outcomes"
                )
            }
            Self::NotBinaryDistribution { actual_states } => {
                write!(
                    f,
                    "expected binary distribution with 2 states, got {actual_states}"
                )
            }
        }
    }
}

impl Error for BeliefError {}

fn validate_unit_scalar(field: &'static str, value: f64) -> Result<f64, BeliefError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(BeliefError::InvalidProbability { field, value });
    }
    Ok(if value == 0.0 { 0.0 } else { value })
}

fn validate_normalized_distribution(probs: Vec<f64>) -> Result<Vec<f64>, BeliefError> {
    if probs.is_empty() {
        return Err(BeliefError::EmptyDistribution);
    }

    let mut sum = 0.0_f64;
    let mut normalized = Vec::with_capacity(probs.len());
    for (index, value) in probs.into_iter().enumerate() {
        if !value.is_finite() {
            return Err(BeliefError::NonFiniteProbability { index, value });
        }
        if value < 0.0 {
            return Err(BeliefError::NegativeProbability { index, value });
        }
        let clean = if value == 0.0 { 0.0 } else { value };
        sum += clean;
        normalized.push(clean);
    }

    if !sum.is_finite() || (sum - 1.0).abs() > DEFAULT_PROBABILITY_SUM_TOLERANCE {
        return Err(BeliefError::ProbabilitySumMismatch {
            sum,
            tolerance: DEFAULT_PROBABILITY_SUM_TOLERANCE,
        });
    }

    Ok(normalized)
}

/// Estado de creencia discreta normalizada sobre `N >= 1` estados mutuamente excluyentes.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteBeliefState {
    probabilities: Vec<f64>,
}

/// Alias semántico de [`DiscreteBeliefState`] para distribuciones categóricas sobre `N` estados.
pub type CategoricalBelief = DiscreteBeliefState;

impl DiscreteBeliefState {
    /// Construye una creencia discreta validando finitud, no negatividad y suma `1.0`.
    pub fn new(probabilities: Vec<f64>) -> Result<Self, BeliefError> {
        let probabilities = validate_normalized_distribution(probabilities)?;
        Ok(Self { probabilities })
    }

    /// Construye una creencia uniforme sobre `num_states` estados (`>= 1`).
    pub fn uniform(num_states: usize) -> Result<Self, BeliefError> {
        if num_states == 0 {
            return Err(BeliefError::EmptyDistribution);
        }
        let p = 1.0 / (num_states as f64);
        Ok(Self {
            probabilities: vec![p; num_states],
        })
    }

    /// Devuelve el slice inmutable de probabilidades por estado.
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Devuelve el número de estados discretos de la distribución.
    pub fn num_states(&self) -> usize {
        self.probabilities.len()
    }

    /// Devuelve la probabilidad asignada al estado `state`, si existe.
    pub fn get(&self, state: usize) -> Option<f64> {
        self.probabilities.get(state).copied()
    }

    /// Devuelve el índice del estado más probable (desempate determinista por menor índice).
    pub fn map_state(&self) -> usize {
        let mut best_idx = 0;
        let mut best_prob = self.probabilities[0];
        for (idx, &p) in self.probabilities.iter().enumerate().skip(1) {
            if p > best_prob {
                best_idx = idx;
                best_prob = p;
            }
        }
        best_idx
    }

    /// Aplica un paso bayesiano puro dado un índice de resultado observado y un [`LikelihoodModel`].
    ///
    /// Si la probabilidad marginal `P(observation) == 0.0`, devuelve
    /// [`BeliefUpdateStatus::ConflictImpossibleObservation`] sin alterar la creencia.
    pub fn update_with_outcome(
        &mut self,
        outcome: usize,
        model: &LikelihoodModel,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        self.update_with_outcome_policy(outcome, model, EpistemicFloorPolicy::None)
    }

    /// Como [`Self::update_with_outcome`], pero aplicando una [`EpistemicFloorPolicy`].
    ///
    /// Con [`EpistemicFloorPolicy::None`] el resultado es idéntico bit a bit al de
    /// [`Self::update_with_outcome`]. Con [`EpistemicFloorPolicy::ClampedFloor`] el producto
    /// bayesiano se calcula en log-espacio (logsumexp) y se aplica el piso `min_prob` a todo
    /// estado no estructuralmente imposible (`P(o|s) > 0.0`). La política se valida contra el
    /// número de estados antes de modificar nada.
    pub fn update_with_outcome_policy(
        &mut self,
        outcome: usize,
        model: &LikelihoodModel,
        policy: EpistemicFloorPolicy,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        policy.validate(self.num_states())?;
        if self.num_states() != model.num_states() {
            return Err(BeliefError::StateCountMismatch {
                belief_states: self.num_states(),
                model_states: model.num_states(),
            });
        }
        if outcome >= model.num_outcomes() {
            return Err(BeliefError::OutcomeOutOfRange {
                outcome,
                num_outcomes: model.num_outcomes(),
            });
        }

        if let EpistemicFloorPolicy::ClampedFloor { min_prob } = policy {
            return Ok(self.update_log_space_floored(outcome, model, min_prob));
        }

        let mut unnormalized = Vec::with_capacity(self.probabilities.len());
        let mut marginal = 0.0_f64;
        for (state_idx, &prior_p) in self.probabilities.iter().enumerate() {
            let likelihood = model.matrix[state_idx][outcome];
            let product = likelihood * prior_p;
            marginal += product;
            unnormalized.push(product);
        }

        if !marginal.is_finite() || marginal <= 0.0 {
            return Ok(BeliefUpdateStatus::ConflictImpossibleObservation);
        }

        for p in &mut unnormalized {
            let posterior = *p / marginal;
            *p = if posterior == 0.0 { 0.0 } else { posterior };
        }
        self.probabilities = unnormalized;
        Ok(BeliefUpdateStatus::Updated)
    }

    /// Paso bayesiano en log-espacio con piso epistémico. Precondiciones validadas por el llamador.
    ///
    /// Estados con `P(o|s) == 0.0` quedan en `0.0` (imposibilidad estructural del modelo); el resto
    /// se pondera con `ln P(o|s) + ln p_s`, se normaliza con logsumexp y se les aplica el piso.
    /// Si ningún estado posible conserva masa, devuelve conflicto sin alterar la creencia.
    fn update_log_space_floored(
        &mut self,
        outcome: usize,
        model: &LikelihoodModel,
        min_prob: f64,
    ) -> BeliefUpdateStatus {
        let log_weights: Vec<f64> = self
            .probabilities
            .iter()
            .enumerate()
            .map(|(state_idx, &prior_p)| {
                let likelihood = model.matrix[state_idx][outcome];
                if likelihood > 0.0 && prior_p > 0.0 {
                    likelihood.ln() + prior_p.ln()
                } else {
                    f64::NEG_INFINITY
                }
            })
            .collect();
        let max_log = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if !max_log.is_finite() {
            // Sin masa posterior (todas imposibles o prior nulo donde el modelo permite).
            return BeliefUpdateStatus::ConflictImpossibleObservation;
        }
        let log_marginal = max_log
            + log_weights
                .iter()
                .map(|&w| (w - max_log).exp())
                .sum::<f64>()
                .ln();

        // `free[i]`: estado elegible para piso (verosimilitud estructuralmente > 0).
        let free: Vec<bool> = (0..self.probabilities.len())
            .map(|i| model.matrix[i][outcome] > 0.0)
            .collect();
        let mut posterior: Vec<f64> = log_weights
            .iter()
            .map(|&w| (w - log_marginal).exp())
            .collect();

        // Water-filling: fija en `min_prob` los estados bajo el piso y reescala el resto.
        let mut pinned = vec![false; posterior.len()];
        loop {
            let mut changed = false;
            for i in 0..posterior.len() {
                if free[i] && !pinned[i] && posterior[i] < min_prob {
                    pinned[i] = true;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            let pinned_mass = pinned.iter().filter(|&&p| p).count() as f64 * min_prob;
            let rest_sum: f64 = posterior
                .iter()
                .zip(&pinned)
                .filter(|&(_, &p)| !p)
                .map(|(&v, _)| v)
                .sum();
            let scale = if rest_sum > 0.0 {
                (1.0 - pinned_mass) / rest_sum
            } else {
                0.0
            };
            for (v, &p) in posterior.iter_mut().zip(&pinned) {
                *v = if p { min_prob } else { *v * scale };
            }
        }
        self.probabilities = posterior;
        BeliefUpdateStatus::Updated
    }

    /// Actualiza la creencia con un [`ObservationEnvelope<usize>`], aplicando verificación de tipo
    /// epistémico, marca temporal (`observed_at >= current_watermark`), deduplicación de evidencia
    /// y actualización bayesiana calibrada.
    ///
    /// Nunca usa [`ObservationEnvelope::confidence`](crate::provenance::ObservationEnvelope::confidence)
    /// como sustituto de la calibración de [`LikelihoodModel`].
    pub fn update_with_observation(
        &mut self,
        envelope: &ObservationEnvelope<usize>,
        model: &LikelihoodModel,
        deduplicator: &mut EvidenceDeduplicator,
        current_watermark: &mut Option<MonotonicInstant>,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        self.update_with_observation_policy(
            envelope,
            model,
            deduplicator,
            current_watermark,
            EpistemicFloorPolicy::None,
        )
    }

    /// Como [`Self::update_with_observation`], aplicando una [`EpistemicFloorPolicy`].
    pub fn update_with_observation_policy(
        &mut self,
        envelope: &ObservationEnvelope<usize>,
        model: &LikelihoodModel,
        deduplicator: &mut EvidenceDeduplicator,
        current_watermark: &mut Option<MonotonicInstant>,
        policy: EpistemicFloorPolicy,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        policy.validate(self.num_states())?;
        if self.num_states() != model.num_states() {
            return Err(BeliefError::StateCountMismatch {
                belief_states: self.num_states(),
                model_states: model.num_states(),
            });
        }
        let outcome = *envelope.payload();
        if outcome >= model.num_outcomes() {
            return Err(BeliefError::OutcomeOutOfRange {
                outcome,
                num_outcomes: model.num_outcomes(),
            });
        }

        match envelope.claim_kind() {
            ObservationKind::Measurement | ObservationKind::Assertion => {}
            ObservationKind::Unknown | ObservationKind::Intention | ObservationKind::Proposal => {
                return Ok(BeliefUpdateStatus::AbstainedUnsupportedKind);
            }
        }

        if let Some(watermark) = *current_watermark
            && envelope.observed_at() < watermark
        {
            return Ok(BeliefUpdateStatus::AbstainedStaleObservation);
        }

        if deduplicator.is_duplicate(
            envelope.sensor_id(),
            envelope.message_id(),
            envelope.evidence_ref(),
        ) {
            return Ok(BeliefUpdateStatus::DuplicateEvidenceIgnored);
        }

        let status = self.update_with_outcome_policy(outcome, model, policy)?;
        if status == BeliefUpdateStatus::Updated {
            deduplicator.record(
                envelope.sensor_id(),
                envelope.message_id(),
                envelope.evidence_ref(),
            );
            *current_watermark = Some(envelope.observed_at());
        }
        Ok(status)
    }
}

/// Creencia binaria normalizada sobre una hipótesis (`false` en índice 0, `true` en índice 1).
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryBelief {
    inner: DiscreteBeliefState,
}

impl BinaryBelief {
    /// Construye una [`BinaryBelief`] a partir de `P(true)` en `[0.0, 1.0]`.
    pub fn new(p_true: f64) -> Result<Self, BeliefError> {
        let p_true = validate_unit_scalar("p_true", p_true)?;
        let p_false = if 1.0 - p_true == 0.0 {
            0.0
        } else {
            1.0 - p_true
        };
        Ok(Self {
            inner: DiscreteBeliefState {
                probabilities: vec![p_false, p_true],
            },
        })
    }

    /// Convierte una [`DiscreteBeliefState`] de 2 estados en [`BinaryBelief`].
    pub fn from_discrete(discrete: DiscreteBeliefState) -> Result<Self, BeliefError> {
        if discrete.num_states() != 2 {
            return Err(BeliefError::NotBinaryDistribution {
                actual_states: discrete.num_states(),
            });
        }
        Ok(Self { inner: discrete })
    }

    /// Devuelve la probabilidad posterior de la hipótesis verdadera (`P(true)`).
    pub fn p_true(&self) -> f64 {
        self.inner.probabilities[1]
    }

    /// Devuelve la probabilidad posterior de la hipótesis falsa (`P(false)`).
    pub fn p_false(&self) -> f64 {
        self.inner.probabilities[0]
    }

    /// Devuelve una referencia a la [`DiscreteBeliefState`] subyacente.
    pub const fn as_discrete(&self) -> &DiscreteBeliefState {
        &self.inner
    }

    /// Consume la creencia binaria y devuelve la [`DiscreteBeliefState`] subyacente.
    pub fn into_discrete(self) -> DiscreteBeliefState {
        self.inner
    }

    /// Actualiza la creencia binaria con un [`ObservationEnvelope<bool>`].
    ///
    /// Convierte `false -> 0` (observación negativa) y `true -> 1` (observación positiva).
    pub fn update_with_observation(
        &mut self,
        envelope: &ObservationEnvelope<bool>,
        model: &LikelihoodModel,
        deduplicator: &mut EvidenceDeduplicator,
        current_watermark: &mut Option<MonotonicInstant>,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        let mapped = envelope.clone().map(usize::from);
        self.inner
            .update_with_observation(&mapped, model, deduplicator, current_watermark)
    }
}

/// Modelo calibrado de verosimilitud discreta `P(observación | estado)`.
///
/// Cada fila `matrix[state]` es una distribución normalizada sobre los resultados observables,
/// calibrada explícitamente con independencia de [`Confidence`](crate::provenance::Confidence).
#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodModel {
    matrix: Vec<Vec<f64>>,
    num_outcomes: usize,
}

impl LikelihoodModel {
    /// Construye un modelo binario calibrado con sensibilidad y especificidad explícitas.
    ///
    /// Convención de índices:
    /// - Estado `0`: hipótesis falsa (`false`); Estado `1`: hipótesis verdadera (`true`).
    /// - Resultado `0`: observación negativa (`false`); Resultado `1`: observación positiva (`true`).
    /// - `sensitivity = P(obs = true | state = true)`
    /// - `specificity = P(obs = false | state = false)`
    pub fn from_sensitivity_specificity(
        sensitivity: f64,
        specificity: f64,
    ) -> Result<Self, BeliefError> {
        let sensitivity = validate_unit_scalar("sensitivity", sensitivity)?;
        let specificity = validate_unit_scalar("specificity", specificity)?;
        let false_positive = if 1.0 - specificity == 0.0 {
            0.0
        } else {
            1.0 - specificity
        };
        let false_negative = if 1.0 - sensitivity == 0.0 {
            0.0
        } else {
            1.0 - sensitivity
        };
        Ok(Self {
            matrix: vec![
                vec![specificity, false_positive],
                vec![false_negative, sensitivity],
            ],
            num_outcomes: 2,
        })
    }

    /// Construye un modelo general a partir de una matriz `P(outcome | state)` donde cada
    /// fila corresponde a un estado y debe sumar `1.0`.
    pub fn from_matrix(matrix: Vec<Vec<f64>>) -> Result<Self, BeliefError> {
        if matrix.is_empty() {
            return Err(BeliefError::InvalidLikelihoodMatrix {
                reason: "matrix must contain at least one state row",
            });
        }
        let num_outcomes = matrix[0].len();
        if num_outcomes == 0 {
            return Err(BeliefError::InvalidLikelihoodMatrix {
                reason: "each state row must contain at least one outcome",
            });
        }

        let mut validated_rows = Vec::with_capacity(matrix.len());
        for row in matrix {
            if row.len() != num_outcomes {
                return Err(BeliefError::InvalidLikelihoodMatrix {
                    reason: "all state rows must have the same number of outcomes",
                });
            }
            validated_rows.push(validate_normalized_distribution(row)?);
        }

        Ok(Self {
            matrix: validated_rows,
            num_outcomes,
        })
    }

    /// Devuelve el número de estados soportados por el modelo.
    pub fn num_states(&self) -> usize {
        self.matrix.len()
    }

    /// Devuelve el número de resultados observables del modelo.
    pub const fn num_outcomes(&self) -> usize {
        self.num_outcomes
    }

    /// Devuelve `P(outcome | state)` si los índices están dentro de rango.
    pub fn likelihood(&self, state: usize, outcome: usize) -> Option<f64> {
        self.matrix
            .get(state)
            .and_then(|row| row.get(outcome))
            .copied()
    }

    /// Si el modelo es binario (`2 x 2`), devuelve la sensibilidad `P(obs=1 | state=1)`.
    pub fn sensitivity(&self) -> Option<f64> {
        if self.num_states() == 2 && self.num_outcomes == 2 {
            Some(self.matrix[1][1])
        } else {
            None
        }
    }

    /// Si el modelo es binario (`2 x 2`), devuelve la especificidad `P(obs=0 | state=0)`.
    pub fn specificity(&self) -> Option<f64> {
        if self.num_states() == 2 && self.num_outcomes == 2 {
            Some(self.matrix[0][0])
        } else {
            None
        }
    }
}

/// Deduplicador de evidencia para evitar doble conteo de rumores u orígenes compartidos.
///
/// Registra referencias de evidencia ([`EvidenceRef`]) y pares `(SensorId, MessageId)` ya
/// contabilizados en una creencia.
#[derive(Debug, Clone, Default)]
pub struct EvidenceDeduplicator {
    seen_evidence: HashSet<EvidenceRef>,
    seen_messages: HashSet<(SensorId, MessageId)>,
}

impl EvidenceDeduplicator {
    /// Crea un deduplicador vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Indica si `evidence_ref` ya fue contabilizada.
    pub fn has_seen_evidence(&self, evidence_ref: EvidenceRef) -> bool {
        self.seen_evidence.contains(&evidence_ref)
    }

    /// Indica si el mensaje `(sensor_id, message_id)` ya fue contabilizado.
    pub fn has_seen_message(&self, sensor_id: SensorId, message_id: MessageId) -> bool {
        self.seen_messages.contains(&(sensor_id, message_id))
    }

    /// Devuelve el número de referencias de evidencia únicas contabilizadas.
    pub fn seen_evidence_count(&self) -> usize {
        self.seen_evidence.len()
    }

    /// Devuelve el número de mensajes únicos `(SensorId, MessageId)` contabilizados.
    pub fn seen_message_count(&self) -> usize {
        self.seen_messages.len()
    }

    /// Indica si la observación corresponde a evidencia o mensaje ya contabilizado.
    pub fn is_duplicate(
        &self,
        sensor_id: SensorId,
        message_id: MessageId,
        evidence_ref: EvidenceRef,
    ) -> bool {
        self.has_seen_evidence(evidence_ref) || self.has_seen_message(sensor_id, message_id)
    }

    /// Registra `(sensor_id, message_id)` y `evidence_ref`; devuelve `true` si era evidencia nueva.
    pub fn record(
        &mut self,
        sensor_id: SensorId,
        message_id: MessageId,
        evidence_ref: EvidenceRef,
    ) -> bool {
        if self.is_duplicate(sensor_id, message_id, evidence_ref) {
            return false;
        }
        self.seen_evidence.insert(evidence_ref);
        self.seen_messages.insert((sensor_id, message_id));
        true
    }
}

/// Resultado explícito de intentar actualizar una creencia discreta con una observación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeliefUpdateStatus {
    /// La creencia se actualizó correctamente mediante regla de Bayes calibrada.
    Updated,
    /// La observación compartía [`EvidenceRef`] o `(SensorId, MessageId)` ya contabilizado y se ignoró.
    DuplicateEvidenceIgnored,
    /// La observación tiene verosimilitud marginal cero (`P(observation) == 0.0`); se preservó el prior.
    ConflictImpossibleObservation,
    /// La observación llegó desactualizada (`observed_at < current_watermark`); se preservó el prior.
    AbstainedStaleObservation,
    /// La clasificación epistémica (`Unknown`, `Intention`, `Proposal`) no permite actualizar hechos.
    AbstainedUnsupportedKind,
}

impl BeliefUpdateStatus {
    /// Devuelve `true` si la creencia fue actualizada.
    pub const fn is_updated(self) -> bool {
        matches!(self, Self::Updated)
    }

    /// Devuelve `true` si la observación fue ignorada por evidencia duplicada.
    pub const fn is_duplicate(self) -> bool {
        matches!(self, Self::DuplicateEvidenceIgnored)
    }

    /// Devuelve `true` si la observación produjo conflicto por probabilidad marginal cero.
    pub const fn is_conflict(self) -> bool {
        matches!(self, Self::ConflictImpossibleObservation)
    }

    /// Devuelve `true` si el filtro se abstuvo por obsolescencia o tipo no soportado.
    pub const fn is_abstained(self) -> bool {
        matches!(
            self,
            Self::AbstainedStaleObservation | Self::AbstainedUnsupportedKind
        )
    }
}

/// Política opcional de regularización epistémica contra el colapso dogmático de creencias.
///
/// La actualización lineal repetida `p_i <- p_i * P(o|s_i)` puede hacer underflow a `0.0`
/// exacto tras decenas de lecturas consistentes; un `0.0` es irreversible (`0.0 * L == 0.0`).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EpistemicFloorPolicy {
    /// Sin regularización: actualización lineal clásica (comportamiento histórico).
    #[default]
    None,
    /// Actualización en log-espacio con piso `min_prob` para todo estado con `P(o|s) > 0.0`.
    /// Requiere `0.0 < min_prob * num_states < 1.0`.
    ClampedFloor {
        /// Probabilidad mínima por estado.
        min_prob: f64,
    },
}

impl EpistemicFloorPolicy {
    /// Valida la política para una creencia de `num_states` estados.
    pub fn validate(self, num_states: usize) -> Result<(), BeliefError> {
        if let Self::ClampedFloor { min_prob } = self {
            let total = min_prob * num_states as f64;
            if !min_prob.is_finite() || !(total > 0.0 && total < 1.0) {
                return Err(BeliefError::InvalidProbability {
                    field: "min_prob",
                    value: min_prob,
                });
            }
        }
        Ok(())
    }
}

/// Filtro bayesiano discreto con estado de creencia, modelo calibrado, deduplicador y marca temporal.
#[derive(Debug, Clone)]
pub struct DiscreteBeliefFilter {
    belief: DiscreteBeliefState,
    model: LikelihoodModel,
    deduplicator: EvidenceDeduplicator,
    current_watermark: Option<MonotonicInstant>,
    floor_policy: EpistemicFloorPolicy,
}

impl DiscreteBeliefFilter {
    /// Crea un filtro discreto validando la compatibilidad entre `prior` y `model`.
    pub fn new(prior: DiscreteBeliefState, model: LikelihoodModel) -> Result<Self, BeliefError> {
        if prior.num_states() != model.num_states() {
            return Err(BeliefError::StateCountMismatch {
                belief_states: prior.num_states(),
                model_states: model.num_states(),
            });
        }
        Ok(Self {
            belief: prior,
            model,
            deduplicator: EvidenceDeduplicator::new(),
            current_watermark: None,
            floor_policy: EpistemicFloorPolicy::None,
        })
    }

    /// Configura la política de piso epistémico (validada contra el número de estados).
    pub fn with_epistemic_floor(
        mut self,
        policy: EpistemicFloorPolicy,
    ) -> Result<Self, BeliefError> {
        policy.validate(self.belief.num_states())?;
        self.floor_policy = policy;
        Ok(self)
    }

    /// Devuelve la política de piso epistémico activa.
    pub const fn epistemic_floor(&self) -> EpistemicFloorPolicy {
        self.floor_policy
    }

    /// Crea un filtro binario calibrado con `prior` (`P(true)`), `sensitivity` y `specificity`.
    pub fn binary(
        prior: BinaryBelief,
        sensitivity: f64,
        specificity: f64,
    ) -> Result<Self, BeliefError> {
        let model = LikelihoodModel::from_sensitivity_specificity(sensitivity, specificity)?;
        Self::new(prior.into_discrete(), model)
    }

    /// Devuelve una referencia inmutable al estado de creencia actual.
    pub const fn belief(&self) -> &DiscreteBeliefState {
        &self.belief
    }

    /// Devuelve una referencia inmutable al modelo de verosimilitud calibrado.
    pub const fn model(&self) -> &LikelihoodModel {
        &self.model
    }

    /// Devuelve el deduplicador de evidencia asociado al filtro.
    pub const fn deduplicator(&self) -> &EvidenceDeduplicator {
        &self.deduplicator
    }

    /// Devuelve la marca temporal monotónica más reciente que actualizó la creencia.
    pub const fn current_watermark(&self) -> Option<MonotonicInstant> {
        self.current_watermark
    }

    /// Procesa una observación categórica (`usize`) preservando la creencia previa ante conflictos,
    /// duplicados o abstenciones.
    pub fn update_with_observation(
        &mut self,
        envelope: &ObservationEnvelope<usize>,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        self.belief.update_with_observation_policy(
            envelope,
            &self.model,
            &mut self.deduplicator,
            &mut self.current_watermark,
            self.floor_policy,
        )
    }

    /// Procesa una observación binaria (`bool`: `false -> 0`, `true -> 1`).
    pub fn update_with_binary_observation(
        &mut self,
        envelope: &ObservationEnvelope<bool>,
    ) -> Result<BeliefUpdateStatus, BeliefError> {
        let mapped = envelope.clone().map(usize::from);
        self.update_with_observation(&mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{Confidence, FrameId, ProvenanceMetadata, UnitTag};

    fn make_envelope<T>(
        sensor_id: u64,
        message_id: u64,
        evidence_ref: u64,
        observed_ns: u128,
        confidence: f64,
        claim_kind: ObservationKind,
        payload: T,
    ) -> ObservationEnvelope<T> {
        ObservationEnvelope::new(
            ProvenanceMetadata {
                message_id: MessageId::new(message_id),
                observed_at: MonotonicInstant::from_nanos(observed_ns),
                received_at: MonotonicInstant::from_nanos(observed_ns + 10),
                sensor_id: SensorId::new(sensor_id),
                frame_id: FrameId::world(),
                units: UnitTag::dimensionless(),
                confidence: Confidence::new(confidence).unwrap(),
                revision: message_id,
                evidence_ref: EvidenceRef::new(evidence_ref),
                claim_kind,
            },
            payload,
        )
    }

    #[test]
    fn validates_belief_distributions_and_likelihood_models() {
        assert!(matches!(
            DiscreteBeliefState::new(vec![]),
            Err(BeliefError::EmptyDistribution)
        ));
        assert!(matches!(
            DiscreteBeliefState::new(vec![0.5, f64::NAN]),
            Err(BeliefError::NonFiniteProbability { index: 1, .. })
        ));
        assert!(matches!(
            DiscreteBeliefState::new(vec![-0.1, 1.1]),
            Err(BeliefError::NegativeProbability { index: 0, .. })
        ));
        assert!(matches!(
            DiscreteBeliefState::new(vec![0.3, 0.3]),
            Err(BeliefError::ProbabilitySumMismatch { .. })
        ));
        assert!(matches!(
            BinaryBelief::new(1.2),
            Err(BeliefError::InvalidProbability {
                field: "p_true",
                ..
            })
        ));
        assert!(matches!(
            LikelihoodModel::from_sensitivity_specificity(f64::INFINITY, 0.8),
            Err(BeliefError::InvalidProbability {
                field: "sensitivity",
                ..
            })
        ));
        assert!(matches!(
            LikelihoodModel::from_matrix(vec![vec![0.5, 0.5], vec![1.0]]),
            Err(BeliefError::InvalidLikelihoodMatrix { .. })
        ));

        let uniform = DiscreteBeliefState::uniform(4).unwrap();
        assert_eq!(uniform.probabilities(), &[0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn ac1_manual_prior_0_2_sensitivity_0_8_specificity_0_8_positive_gives_0_5() {
        let prior = BinaryBelief::new(0.2).unwrap();
        let mut filter = DiscreteBeliefFilter::binary(prior, 0.8, 0.8).unwrap();

        // Aunque Confidence de interpretación sea baja (0.15) o alta (0.99),
        // la actualización usa la calibración explícita del LikelihoodModel.
        let obs = make_envelope(1, 1, 100, 1_000, 0.15, ObservationKind::Measurement, true);
        let status = filter.update_with_binary_observation(&obs).unwrap();

        assert_eq!(status, BeliefUpdateStatus::Updated);
        assert!(status.is_updated());

        let posterior_true = filter.belief().get(1).unwrap();
        let posterior_false = filter.belief().get(0).unwrap();
        assert!(
            (posterior_true - 0.5).abs() < 1e-12,
            "expected posterior P(true) == 0.5, got {posterior_true}"
        );
        assert!(
            (posterior_false - 0.5).abs() < 1e-12,
            "expected posterior P(false) == 0.5, got {posterior_false}"
        );
        assert_eq!(
            filter.current_watermark(),
            Some(MonotonicInstant::from_nanos(1_000))
        );
    }

    #[test]
    fn ac1_calibration_is_independent_of_envelope_confidence() {
        let model = LikelihoodModel::from_sensitivity_specificity(0.8, 0.8).unwrap();
        let mut b_low = BinaryBelief::new(0.2).unwrap();
        let mut b_high = BinaryBelief::new(0.2).unwrap();

        let mut dedup_a = EvidenceDeduplicator::new();
        let mut dedup_b = EvidenceDeduplicator::new();
        let mut wm_a = None;
        let mut wm_b = None;

        let obs_low_conf = make_envelope(1, 1, 10, 100, 0.05, ObservationKind::Measurement, true);
        let obs_high_conf = make_envelope(1, 1, 10, 100, 0.99, ObservationKind::Measurement, true);

        assert_eq!(
            b_low
                .update_with_observation(&obs_low_conf, &model, &mut dedup_a, &mut wm_a)
                .unwrap(),
            BeliefUpdateStatus::Updated
        );
        assert_eq!(
            b_high
                .update_with_observation(&obs_high_conf, &model, &mut dedup_b, &mut wm_b)
                .unwrap(),
            BeliefUpdateStatus::Updated
        );
        assert!((b_low.p_true() - b_high.p_true()).abs() < 1e-12);
        assert!((b_low.p_true() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ac2_repeated_rumor_with_shared_evidence_ref_does_not_double_count() {
        let prior = BinaryBelief::new(0.2).unwrap();
        let mut filter = DiscreteBeliefFilter::binary(prior, 0.8, 0.8).unwrap();

        // Primera observación con EvidenceRef(777) desde sensor 1.
        let obs1 = make_envelope(1, 1, 777, 1_000, 0.8, ObservationKind::Assertion, true);
        assert_eq!(
            filter.update_with_binary_observation(&obs1).unwrap(),
            BeliefUpdateStatus::Updated
        );
        let after_first = filter.belief().get(1).unwrap();
        assert!((after_first - 0.5).abs() < 1e-12);

        // Rumor repetido por otro sensor (sensor 2, distinto message_id) pero con el mismo EvidenceRef(777).
        let rumor = make_envelope(2, 99, 777, 2_000, 0.95, ObservationKind::Assertion, true);
        let rumor_status = filter.update_with_binary_observation(&rumor).unwrap();
        assert_eq!(rumor_status, BeliefUpdateStatus::DuplicateEvidenceIgnored);
        assert!(rumor_status.is_duplicate());
        assert!((filter.belief().get(1).unwrap() - after_first).abs() < 1e-12);
        // La marca de agua no avanza por evidencia duplicada ignorada.
        assert_eq!(
            filter.current_watermark(),
            Some(MonotonicInstant::from_nanos(1_000))
        );

        // Evidencia genuinamente independiente (distinto EvidenceRef y distinto mensaje) sí actualiza.
        let independent =
            make_envelope(2, 100, 778, 2_500, 0.8, ObservationKind::Measurement, true);
        assert_eq!(
            filter.update_with_binary_observation(&independent).unwrap(),
            BeliefUpdateStatus::Updated
        );
        // P(true) = (0.8 * 0.5) / (0.8 * 0.5 + 0.2 * 0.5) = 0.8
        assert!((filter.belief().get(1).unwrap() - 0.8).abs() < 1e-12);
    }

    #[test]
    fn ac3_impossible_observation_produces_conflict_without_corrupting_prior() {
        // Estado determinista P(false)=1.0, P(true)=0.0 con detector perfecto (sensitivity=1.0, specificity=1.0).
        // Una observación positiva tiene P(obs=true) = 0.0*1.0 + 1.0*0.0 = 0.0 -> imposible.
        let prior = BinaryBelief::new(0.0).unwrap();
        let mut filter = DiscreteBeliefFilter::binary(prior, 1.0, 1.0).unwrap();

        let impossible = make_envelope(1, 1, 50, 1_000, 0.9, ObservationKind::Measurement, true);
        let status = filter.update_with_binary_observation(&impossible).unwrap();

        assert_eq!(status, BeliefUpdateStatus::ConflictImpossibleObservation);
        assert!(status.is_conflict());
        // Creencia intacta (sin NaN ni división por cero).
        assert_eq!(filter.belief().probabilities(), &[1.0, 0.0]);
        assert_eq!(filter.current_watermark(), None);
    }

    #[test]
    fn ac3_stale_observation_and_unknown_kind_abstain_without_corrupting_prior() {
        let prior = CategoricalBelief::new(vec![0.6, 0.3, 0.1]).unwrap();
        let model =
            LikelihoodModel::from_matrix(vec![vec![0.7, 0.3], vec![0.2, 0.8], vec![0.5, 0.5]])
                .unwrap();
        let mut filter = DiscreteBeliefFilter::new(prior, model).unwrap();

        // Observación válida en t = 2_000 ns.
        let valid = make_envelope(1, 1, 10, 2_000, 0.9, ObservationKind::Measurement, 1_usize);
        assert_eq!(
            filter.update_with_observation(&valid).unwrap(),
            BeliefUpdateStatus::Updated
        );
        let snapshot_after_valid = filter.belief().clone();

        // Observación desactualizada (observed_at = 1_500 < 2_000) -> AbstainedStaleObservation.
        let stale = make_envelope(1, 2, 11, 1_500, 0.9, ObservationKind::Measurement, 0_usize);
        let stale_status = filter.update_with_observation(&stale).unwrap();
        assert_eq!(stale_status, BeliefUpdateStatus::AbstainedStaleObservation);
        assert!(stale_status.is_abstained());
        assert_eq!(filter.belief(), &snapshot_after_valid);

        // Observación con ObservationKind::Unknown -> AbstainedUnsupportedKind.
        let unknown = make_envelope(1, 3, 12, 3_000, 0.9, ObservationKind::Unknown, 0_usize);
        let unknown_status = filter.update_with_observation(&unknown).unwrap();
        assert_eq!(unknown_status, BeliefUpdateStatus::AbstainedUnsupportedKind);
        assert!(unknown_status.is_abstained());
        assert_eq!(filter.belief(), &snapshot_after_valid);
        assert_eq!(
            filter.current_watermark(),
            Some(MonotonicInstant::from_nanos(2_000))
        );
    }

    fn binary_filter_with(
        prior: f64,
        sens: f64,
        spec: f64,
        policy: EpistemicFloorPolicy,
    ) -> DiscreteBeliefFilter {
        DiscreteBeliefFilter::binary(BinaryBelief::new(prior).unwrap(), sens, spec)
            .unwrap()
            .with_epistemic_floor(policy)
            .unwrap()
    }

    #[test]
    fn gog124_ac1_none_policy_is_bit_identical_to_legacy_update() {
        assert_eq!(
            DiscreteBeliefFilter::binary(BinaryBelief::new(0.2).unwrap(), 0.8, 0.8)
                .unwrap()
                .epistemic_floor(),
            EpistemicFloorPolicy::None
        );
        let mut legacy = DiscreteBeliefState::new(vec![0.8, 0.2]).unwrap();
        let mut explicit = legacy.clone();
        let model = LikelihoodModel::from_sensitivity_specificity(0.8, 0.8).unwrap();
        for outcome in [1, 1, 0, 1, 0, 0] {
            legacy.update_with_outcome(outcome, &model).unwrap();
            explicit
                .update_with_outcome_policy(outcome, &model, EpistemicFloorPolicy::None)
                .unwrap();
            assert_eq!(legacy.probabilities(), explicit.probabilities());
        }
        let mut filter = binary_filter_with(0.2, 0.8, 0.8, EpistemicFloorPolicy::None);
        let obs = make_envelope(1, 1, 100, 1_000, 0.5, ObservationKind::Measurement, true);
        filter.update_with_binary_observation(&obs).unwrap();
        assert!((filter.belief().get(1).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gog124_invalid_floor_config_is_rejected() {
        for bad in [0.0, -0.1, 0.5, 0.6, f64::NAN, f64::INFINITY] {
            let r = DiscreteBeliefFilter::binary(BinaryBelief::new(0.5).unwrap(), 0.8, 0.8)
                .unwrap()
                .with_epistemic_floor(EpistemicFloorPolicy::ClampedFloor { min_prob: bad });
            assert!(
                matches!(
                    r,
                    Err(BeliefError::InvalidProbability {
                        field: "min_prob",
                        ..
                    })
                ),
                "min_prob {bad} should be rejected"
            );
        }
        assert!(
            EpistemicFloorPolicy::ClampedFloor { min_prob: 0.49 }
                .validate(2)
                .is_ok()
        );
    }

    /// Escenario: 200 lecturas `false` y luego `true`; devuelve P(true) tras cada fase.
    fn run_dogma_scenario(policy: EpistemicFloorPolicy) -> (DiscreteBeliefFilter, Vec<f64>) {
        // Detector muy fiable: razón de verosimilitud 99 por lectura -> 99^-200 ~ 1e-400 < f64::MIN.
        let mut filter = binary_filter_with(0.5, 0.99, 0.99, policy);
        let mut id = 0_u64;
        for _ in 0..200 {
            id += 1;
            let obs = make_envelope(
                1,
                id,
                id,
                id as u128,
                0.9,
                ObservationKind::Measurement,
                false,
            );
            assert!(
                filter
                    .update_with_binary_observation(&obs)
                    .unwrap()
                    .is_updated()
            );
        }
        let mut trace = Vec::new();
        for _ in 0..10 {
            id += 1;
            let obs = make_envelope(
                1,
                id,
                id,
                id as u128,
                0.9,
                ObservationKind::Measurement,
                true,
            );
            filter.update_with_binary_observation(&obs).unwrap();
            trace.push(filter.belief().get(1).unwrap());
        }
        (filter, trace)
    }

    #[test]
    fn gog124_ac2_none_policy_stays_dogmatically_stuck_at_zero() {
        let (filter, trace) = run_dogma_scenario(EpistemicFloorPolicy::None);
        assert!(trace.iter().all(|&p| p == 0.0), "bug demo: {trace:?}");
        assert_eq!(filter.belief().probabilities(), &[1.0, 0.0]);
    }

    #[test]
    fn gog124_ac2_clamped_floor_avoids_underflow_and_recovers() {
        let floor = 1e-6;
        let policy = EpistemicFloorPolicy::ClampedFloor { min_prob: floor };

        // Tras 200 falsas: sin underflow, P(true) pegado al piso, suma 1.
        let mut filter = binary_filter_with(0.5, 0.99, 0.99, policy);
        for i in 1..=200_u64 {
            let obs = make_envelope(1, i, i, i as u128, 0.9, ObservationKind::Measurement, false);
            filter.update_with_binary_observation(&obs).unwrap();
            assert!(filter.belief().probabilities().iter().all(|&p| p >= floor));
        }
        assert_eq!(filter.belief().get(1).unwrap(), floor);
        let sum: f64 = filter.belief().probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);

        // Recuperación: cruza 0.5 en <= 5 lecturas true y sigue subiendo.
        let (_, trace) = run_dogma_scenario(policy);
        assert!(trace[..5].iter().any(|&p| p > 0.5), "trace: {trace:?}");
        let first = trace.iter().position(|&p| p > 0.5).unwrap();
        assert!(first < 5);
        // Sube estrictamente hasta saturar en el techo 1 - floor, sin retroceder jamás.
        assert!(trace.windows(2).all(|w| w[1] > w[0] || w[1] == 1.0 - floor));
        assert!(trace.iter().all(|&p| p >= floor));
        assert!(*trace.last().unwrap() > 0.99);
    }

    #[test]
    fn gog124_floor_keeps_structural_zeros_and_floors_the_rest() {
        let model =
            LikelihoodModel::from_matrix(vec![vec![0.0, 1.0], vec![0.9, 0.1], vec![0.9, 0.1]])
                .unwrap();
        let mut b = DiscreteBeliefState::new(vec![0.4, 0.3, 0.3]).unwrap();
        let policy = EpistemicFloorPolicy::ClampedFloor { min_prob: 0.01 };
        for _ in 0..50 {
            b.update_with_outcome_policy(0, &model, policy).unwrap();
        }
        assert_eq!(b.get(0), Some(0.0));
        assert!(b.get(1).unwrap() > 0.0 && b.get(2).unwrap() > 0.0);
        let sum: f64 = b.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        // Outcome 1 hace dominar al estado 0 y respeta el piso del resto.
        for _ in 0..50 {
            b.update_with_outcome_policy(1, &model, policy).unwrap();
        }
        assert!(b.get(1).unwrap() >= 0.01 - 1e-15 && b.get(2).unwrap() >= 0.01 - 1e-15);
    }

    #[test]
    fn gog124_ac3_impossible_observation_conflicts_under_both_policies() {
        let policies = [
            EpistemicFloorPolicy::None,
            EpistemicFloorPolicy::ClampedFloor { min_prob: 1e-6 },
        ];
        for policy in policies {
            // (a) P(o|s) == 0.0 para todos los estados (columna 1 nula).
            let model = LikelihoodModel::from_matrix(vec![vec![1.0, 0.0], vec![1.0, 0.0]]).unwrap();
            let prior = DiscreteBeliefState::new(vec![0.3, 0.7]).unwrap();
            let mut filter = DiscreteBeliefFilter::new(prior.clone(), model)
                .unwrap()
                .with_epistemic_floor(policy)
                .unwrap();
            let obs = make_envelope(1, 1, 1, 1_000, 0.9, ObservationKind::Measurement, 1_usize);
            let status = filter.update_with_observation(&obs).unwrap();
            assert_eq!(status, BeliefUpdateStatus::ConflictImpossibleObservation);
            assert_eq!(filter.belief(), &prior);
            assert_eq!(filter.current_watermark(), None);

            // (b) Prior determinista + detector perfecto: marginal cero.
            let mut filter = binary_filter_with(0.0, 1.0, 1.0, policy);
            let obs = make_envelope(1, 1, 1, 1_000, 0.9, ObservationKind::Measurement, true);
            let status = filter.update_with_binary_observation(&obs).unwrap();
            assert_eq!(status, BeliefUpdateStatus::ConflictImpossibleObservation);
            assert_eq!(filter.belief().probabilities(), &[1.0, 0.0]);
            assert_eq!(filter.current_watermark(), None);
        }
    }
}
