//! Puerto tipado System One y proveedor offline determinista (`std`-only).
//!
//! Este módulo define el contrato del puerto System One para consultas rápidas
//! estructuradas sin dependencias de red, tokens, Jev, `Planner` ni `Actuator`:
//! - [`TypedQuery`]: preguntas tipadas cerradas (`Choice`, `Noul`, `Score`) con
//!   identificadores validados ([`QueryId`], [`OptionId`], [`HypothesisId`]) y plazo
//!   monotónico opcional ([`MonotonicDeadline`]).
//! - [`SystemOneResponse`]: respuesta tipada con procedencia ([`ProvenanceMetadata`] /
//!   [`EvidenceRef`]), estado explícito ([`ResponseStatus`]: `Answered`, `Unknown`,
//!   `TimedOut`), distribución normalizada validada ([`ValidatedDistribution`]) y
//!   puntuaciones crudas preservadas por separado ([`RawScores`]).
//! - [`SystemOneError`]: validación estricta que rechaza opciones inexistentes,
//!   puntuaciones o probabilidades no finitas (`NaN`/infinito), distribuciones
//!   inválidas e inconsistencias estructurales.
//! - [`SystemOneProvider`]: trait inyectable (compatible con `&dyn SystemOneProvider` y
//!   genéricos `P: SystemOneProvider`) sin referencia al planificador ni actuadores.
//! - [`ReplaySystemOneProvider`] (y su alias [`FixtureSystemOneProvider`]): proveedor
//!   offline determinista basado en fixtures registrados en memoria.

use std::collections::{BTreeMap, HashSet};
use std::error::Error;
use std::fmt;
use std::time::Duration as StdDuration;

use crate::belief::{BeliefError, DiscreteBeliefState};
use crate::clock::{MonotonicClock, MonotonicDeadline};
use crate::entropy::DEFAULT_PROBABILITY_SUM_TOLERANCE;
use crate::provenance::{EvidenceRef, ProvenanceMetadata};

const NOUL_TRUE_OPTION: &str = "true";
const NOUL_FALSE_OPTION: &str = "false";

/// Identificador no vacío de una consulta tipada hacia System One.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueryId(String);

impl QueryId {
    /// Crea un [`QueryId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, SystemOneError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SystemOneError::EmptyIdentifier { field: "query_id" });
        }
        Ok(Self(id))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for QueryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador no vacío de una opción dentro de una consulta `Choice` o `Score`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OptionId(String);

impl OptionId {
    /// Crea un [`OptionId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, SystemOneError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SystemOneError::EmptyIdentifier { field: "option_id" });
        }
        Ok(Self(id))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for OptionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador no vacío de una hipótesis evaluada en una consulta `Noul`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HypothesisId(String);

impl HypothesisId {
    /// Crea un [`HypothesisId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, SystemOneError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(SystemOneError::EmptyIdentifier {
                field: "hypothesis_id",
            });
        }
        Ok(Self(id))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for HypothesisId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Errores explícitos de validación o ejecución del puerto System One.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemOneError {
    /// Un identificador requerido (`query_id`, `option_id`, `hypothesis_id`) está vacío.
    EmptyIdentifier {
        /// Nombre del campo inválido.
        field: &'static str,
    },
    /// Una consulta `Choice` o `Score` declaró un conjunto vacío de opciones.
    EmptyOptions {
        /// Identificador de la consulta.
        query_id: QueryId,
    },
    /// Se detectó una opción duplicada en una consulta, distribución o lista de scores.
    DuplicateOption {
        /// Opción duplicada.
        option: OptionId,
    },
    /// La respuesta o fixture hace referencia a una opción inexistente en la consulta.
    UnknownOption {
        /// Identificador de la consulta.
        query_id: QueryId,
        /// Opción no declarada en la consulta.
        option: OptionId,
    },
    /// Falta una opción declarada en la consulta dentro de la distribución o scores requeridos.
    MissingOption {
        /// Identificador de la consulta.
        query_id: QueryId,
        /// Opción faltante.
        option: OptionId,
    },
    /// Una puntuación cruda no es finita (`NaN`, `+inf` o `-inf`).
    NonFiniteScore {
        /// Opción asociada al score inválido.
        option: OptionId,
        /// Valor no finito recibido.
        value: f64,
    },
    /// Una probabilidad en la distribución no es finita (`NaN`, `+inf` o `-inf`).
    NonFiniteProbability {
        /// Opción asociada a la probabilidad inválida.
        option: OptionId,
        /// Valor no finito recibido.
        value: f64,
    },
    /// Una probabilidad en la distribución es estrictamente negativa.
    NegativeProbability {
        /// Opción asociada a la probabilidad negativa.
        option: OptionId,
        /// Valor negativo recibido.
        value: f64,
    },
    /// La distribución está vacía o no puede normalizarse.
    InvalidDistribution {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// La suma de probabilidades difiere de `1.0` más allá de la tolerancia permitida.
    InvalidDistributionSum {
        /// Suma observada.
        sum: f64,
        /// Tolerancia máxima permitida.
        tolerance: f64,
    },
    /// El identificador de consulta de la respuesta no coincide con el de la consulta.
    QueryIdMismatch {
        /// Identificador esperado.
        expected: QueryId,
        /// Identificador recibido.
        actual: QueryId,
    },
    /// El tipo de respuesta no corresponde al tipo de consulta (`Choice` / `Noul` / `Score`).
    QueryKindMismatch {
        /// Tipo esperado.
        expected: &'static str,
        /// Tipo recibido.
        actual: &'static str,
    },
    /// La hipótesis de la respuesta `Noul` no coincide con la hipótesis consultada.
    HypothesisMismatch {
        /// Hipótesis esperada.
        expected: HypothesisId,
        /// Hipótesis recibida.
        actual: HypothesisId,
    },
    /// La respuesta incumple una invariante estructural para su estado (`Answered`, `Unknown`, `TimedOut`).
    InvalidResponse {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// No existe un fixture registrado en el proveedor offline para el `query_id` solicitado.
    MissingFixture {
        /// Identificador de consulta no encontrado.
        query_id: QueryId,
    },
}

impl fmt::Display for SystemOneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { field } => {
                write!(f, "system one identifier '{field}' cannot be empty")
            }
            Self::EmptyOptions { query_id } => {
                write!(
                    f,
                    "system one query '{query_id}' requires at least one option"
                )
            }
            Self::DuplicateOption { option } => {
                write!(f, "duplicate option '{option}' in system one contract")
            }
            Self::UnknownOption { query_id, option } => {
                write!(
                    f,
                    "unknown option '{option}' for system one query '{query_id}'"
                )
            }
            Self::MissingOption { query_id, option } => {
                write!(
                    f,
                    "missing required option '{option}' for system one query '{query_id}'"
                )
            }
            Self::NonFiniteScore { option, value } => {
                write!(f, "non-finite raw score for option '{option}': {value}")
            }
            Self::NonFiniteProbability { option, value } => {
                write!(f, "non-finite probability for option '{option}': {value}")
            }
            Self::NegativeProbability { option, value } => {
                write!(f, "negative probability for option '{option}': {value}")
            }
            Self::InvalidDistribution { reason } => {
                write!(f, "invalid system one distribution: {reason}")
            }
            Self::InvalidDistributionSum { sum, tolerance } => {
                write!(
                    f,
                    "system one distribution sum mismatch: sum {sum} differs from 1.0 by more than {tolerance}"
                )
            }
            Self::QueryIdMismatch { expected, actual } => {
                write!(
                    f,
                    "system one query_id mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::QueryKindMismatch { expected, actual } => {
                write!(
                    f,
                    "system one query kind mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::HypothesisMismatch { expected, actual } => {
                write!(
                    f,
                    "system one hypothesis mismatch: expected '{expected}', got '{actual}'"
                )
            }
            Self::InvalidResponse { reason } => {
                write!(f, "invalid system one response: {reason}")
            }
            Self::MissingFixture { query_id } => {
                write!(f, "no offline fixture registered for query '{query_id}'")
            }
        }
    }
}

impl Error for SystemOneError {}

/// Pregunta tipada enviada al puerto System One.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedQuery {
    /// Selección discreta sobre un conjunto cerrado y no vacío de opciones.
    Choice {
        /// Identificador único de la consulta.
        query_id: QueryId,
        /// Opciones permitidas para la respuesta.
        options: Vec<OptionId>,
        /// Plazo monotónico opcional para responder.
        deadline: Option<MonotonicDeadline>,
    },
    /// Juicio epistémico ternario (`True` / `False` / `Unknown`) sobre una hipótesis.
    Noul {
        /// Identificador único de la consulta.
        query_id: QueryId,
        /// Hipótesis sometida a evaluación.
        hypothesis_id: HypothesisId,
        /// Plazo monotónico opcional para responder.
        deadline: Option<MonotonicDeadline>,
    },
    /// Puntuación cuantitativa de un conjunto cerrado y no vacío de opciones.
    Score {
        /// Identificador único de la consulta.
        query_id: QueryId,
        /// Opciones a puntuar.
        options: Vec<OptionId>,
        /// Plazo monotónico opcional para responder.
        deadline: Option<MonotonicDeadline>,
    },
}

impl TypedQuery {
    /// Construye y valida una consulta [`TypedQuery::Choice`].
    pub fn choice(
        query_id: QueryId,
        options: Vec<OptionId>,
        deadline: Option<MonotonicDeadline>,
    ) -> Result<Self, SystemOneError> {
        let query = Self::Choice {
            query_id,
            options,
            deadline,
        };
        query.validate()?;
        Ok(query)
    }

    /// Construye una consulta [`TypedQuery::Noul`].
    pub const fn noul(
        query_id: QueryId,
        hypothesis_id: HypothesisId,
        deadline: Option<MonotonicDeadline>,
    ) -> Self {
        Self::Noul {
            query_id,
            hypothesis_id,
            deadline,
        }
    }

    /// Construye y valida una consulta [`TypedQuery::Score`].
    pub fn score(
        query_id: QueryId,
        options: Vec<OptionId>,
        deadline: Option<MonotonicDeadline>,
    ) -> Result<Self, SystemOneError> {
        let query = Self::Score {
            query_id,
            options,
            deadline,
        };
        query.validate()?;
        Ok(query)
    }

    /// Devuelve el [`QueryId`] de la consulta.
    pub const fn query_id(&self) -> &QueryId {
        match self {
            Self::Choice { query_id, .. }
            | Self::Noul { query_id, .. }
            | Self::Score { query_id, .. } => query_id,
        }
    }

    /// Devuelve el plazo monotónico asociado a la consulta, si existe.
    pub const fn deadline(&self) -> Option<MonotonicDeadline> {
        match self {
            Self::Choice { deadline, .. }
            | Self::Noul { deadline, .. }
            | Self::Score { deadline, .. } => *deadline,
        }
    }

    /// Devuelve una copia de la consulta con el plazo monotónico actualizado.
    pub fn with_deadline(mut self, new_deadline: MonotonicDeadline) -> Self {
        match &mut self {
            Self::Choice { deadline, .. }
            | Self::Noul { deadline, .. }
            | Self::Score { deadline, .. } => *deadline = Some(new_deadline),
        }
        self
    }

    /// Valida las invariantes estructurales de la consulta (opciones no vacías y sin duplicados).
    pub fn validate(&self) -> Result<(), SystemOneError> {
        match self {
            Self::Choice {
                query_id, options, ..
            }
            | Self::Score {
                query_id, options, ..
            } => {
                if options.is_empty() {
                    return Err(SystemOneError::EmptyOptions {
                        query_id: query_id.clone(),
                    });
                }
                let mut seen = HashSet::with_capacity(options.len());
                for option in options {
                    if !seen.insert(option) {
                        return Err(SystemOneError::DuplicateOption {
                            option: option.clone(),
                        });
                    }
                }
                Ok(())
            }
            Self::Noul { .. } => Ok(()),
        }
    }
}

/// Estado explícito de resolución de una respuesta de System One.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseStatus {
    /// El proveedor respondió con evidencia y distribución validada dentro del plazo.
    Answered,
    /// El proveedor declaró explícitamente desconocimiento o ausencia de evidencia suficiente.
    Unknown,
    /// Se agotó el plazo monotónico ([`MonotonicDeadline`]) antes de obtener respuesta.
    TimedOut,
}

impl ResponseStatus {
    /// Devuelve `true` si el estado es [`ResponseStatus::Answered`].
    pub const fn is_answered(self) -> bool {
        matches!(self, Self::Answered)
    }

    /// Devuelve `true` si el estado es [`ResponseStatus::Unknown`].
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Devuelve `true` si el estado es [`ResponseStatus::TimedOut`].
    pub const fn is_timed_out(self) -> bool {
        matches!(self, Self::TimedOut)
    }
}

impl fmt::Display for ResponseStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Answered => "answered",
            Self::Unknown => "unknown",
            Self::TimedOut => "timed_out",
        };
        f.write_str(label)
    }
}

/// Veredicto ternario para consultas [`TypedQuery::Noul`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoulVerdict {
    /// La hipótesis se evalúa como verdadera.
    True,
    /// La hipótesis se evalúa como falsa.
    False,
    /// No hay evidencia suficiente o la respuesta expiró.
    Unknown,
}

impl fmt::Display for NoulVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::True => "true",
            Self::False => "false",
            Self::Unknown => "unknown",
        };
        f.write_str(label)
    }
}

/// Colección validada de puntuaciones crudas (`raw_scores`) preservadas sin normalizar.
///
/// Cada entrada garantiza finitud (`f64::is_finite`) y unicidad de [`OptionId`].
/// Nunca se mezcla ni sobreescribe con [`ValidatedDistribution`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RawScores {
    entries: Vec<(OptionId, f64)>,
}

impl RawScores {
    /// Crea una colección vacía de puntuaciones crudas.
    pub const fn empty() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Construye [`RawScores`] validando finitud de todos los valores y ausencia de duplicados.
    pub fn new(entries: Vec<(OptionId, f64)>) -> Result<Self, SystemOneError> {
        let mut seen = HashSet::with_capacity(entries.len());
        let mut normalized = Vec::with_capacity(entries.len());
        for (option, value) in entries {
            if !value.is_finite() {
                return Err(SystemOneError::NonFiniteScore { option, value });
            }
            if !seen.insert(option.clone()) {
                return Err(SystemOneError::DuplicateOption { option });
            }
            let clean = if value == 0.0 { 0.0 } else { value };
            normalized.push((option, clean));
        }
        Ok(Self {
            entries: normalized,
        })
    }

    /// Devuelve el slice ordenado de pares `(OptionId, score)`.
    pub fn entries(&self) -> &[(OptionId, f64)] {
        &self.entries
    }

    /// Busca la puntuación cruda asociada a `option`.
    pub fn score_of(&self, option: &OptionId) -> Option<f64> {
        self.entries
            .iter()
            .find(|(id, _)| id == option)
            .map(|(_, value)| *value)
    }

    /// Número de puntuaciones crudas registradas.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Indica si no hay puntuaciones crudas registradas.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Distribución de probabilidad discreta normalizada y validada sobre [`OptionId`].
///
/// Garantiza que la distribución no está vacía, todas las probabilidades son finitas y
/// están en `[0.0, 1.0]`, no existen opciones duplicadas y la suma es `1.0` dentro de
/// [`DEFAULT_PROBABILITY_SUM_TOLERANCE`].
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedDistribution {
    entries: Vec<(OptionId, f64)>,
}

impl ValidatedDistribution {
    /// Construye y valida una distribución de probabilidad discreta normalizada.
    pub fn new(entries: Vec<(OptionId, f64)>) -> Result<Self, SystemOneError> {
        if entries.is_empty() {
            return Err(SystemOneError::InvalidDistribution {
                reason: "distribution cannot be empty",
            });
        }

        let mut seen = HashSet::with_capacity(entries.len());
        let mut validated = Vec::with_capacity(entries.len());
        let mut sum = 0.0_f64;

        for (option, value) in entries {
            if !value.is_finite() {
                return Err(SystemOneError::NonFiniteProbability { option, value });
            }
            if value < 0.0 {
                return Err(SystemOneError::NegativeProbability { option, value });
            }
            if !seen.insert(option.clone()) {
                return Err(SystemOneError::DuplicateOption { option });
            }
            let clean = if value == 0.0 { 0.0 } else { value };
            sum += clean;
            validated.push((option, clean));
        }

        if !sum.is_finite() || (sum - 1.0).abs() > DEFAULT_PROBABILITY_SUM_TOLERANCE {
            return Err(SystemOneError::InvalidDistributionSum {
                sum,
                tolerance: DEFAULT_PROBABILITY_SUM_TOLERANCE,
            });
        }

        Ok(Self { entries: validated })
    }

    /// Deriva una [`ValidatedDistribution`] aplicando softmax numéricamente estable sobre [`RawScores`].
    ///
    /// Los [`RawScores`] originales no se modifican ni se sobrescriben.
    pub fn from_softmax_scores(raw_scores: &RawScores) -> Result<Self, SystemOneError> {
        if raw_scores.is_empty() {
            return Err(SystemOneError::InvalidDistribution {
                reason: "cannot derive distribution from empty raw scores",
            });
        }

        let max_score = raw_scores
            .entries()
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut weights = Vec::with_capacity(raw_scores.len());
        let mut sum = 0.0_f64;
        for (option, score) in raw_scores.entries() {
            let weight = (score - max_score).exp();
            if !weight.is_finite() {
                return Err(SystemOneError::NonFiniteScore {
                    option: option.clone(),
                    value: *score,
                });
            }
            sum += weight;
            weights.push((option.clone(), weight));
        }

        if !sum.is_finite() || sum <= 0.0 {
            return Err(SystemOneError::InvalidDistribution {
                reason: "softmax normalization produced non-positive or non-finite sum",
            });
        }

        let probs = weights
            .into_iter()
            .map(|(option, weight)| (option, weight / sum))
            .collect();
        Self::new(probs)
    }

    /// Devuelve el slice de pares `(OptionId, probabilidad)`.
    pub fn entries(&self) -> &[(OptionId, f64)] {
        &self.entries
    }

    /// Devuelve la probabilidad asociada a `option`, si existe.
    pub fn probability_of(&self, option: &OptionId) -> Option<f64> {
        self.entries
            .iter()
            .find(|(id, _)| id == option)
            .map(|(_, p)| *p)
    }

    /// Devuelve únicamente los valores de probabilidad en el orden declarado.
    pub fn probabilities(&self) -> Vec<f64> {
        self.entries.iter().map(|(_, p)| *p).collect()
    }

    /// Convierte la distribución validada en un [`DiscreteBeliefState`].
    pub fn to_belief_state(&self) -> Result<DiscreteBeliefState, BeliefError> {
        DiscreteBeliefState::new(self.probabilities())
    }

    /// Número de opciones en la distribución.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Indica si la distribución está vacía (siempre `false` en instancias válidas).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Carga útil específica del tipo de consulta respondida por System One.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemOnePayload {
    /// Resultado de una consulta [`TypedQuery::Choice`].
    Choice {
        /// Opción elegida cuando el estado es [`ResponseStatus::Answered`].
        selected: Option<OptionId>,
    },
    /// Resultado de una consulta [`TypedQuery::Noul`].
    Noul {
        /// Hipótesis evaluada.
        hypothesis_id: HypothesisId,
        /// Veredicto ternario (`True`, `False` o `Unknown`).
        verdict: NoulVerdict,
    },
    /// Resultado de una consulta [`TypedQuery::Score`].
    Score,
}

/// Respuesta tipada de System One con procedencia, distribución validada y scores crudos separados.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneResponse {
    query_id: QueryId,
    status: ResponseStatus,
    payload: SystemOnePayload,
    provenance: ProvenanceMetadata,
    validated_distribution: Option<ValidatedDistribution>,
    raw_scores: RawScores,
}

impl SystemOneResponse {
    /// Construye una [`SystemOneResponse`] y la valida estrictamente frente a `query`.
    pub fn new_validated(
        query: &TypedQuery,
        status: ResponseStatus,
        payload: SystemOnePayload,
        provenance: ProvenanceMetadata,
        validated_distribution: Option<ValidatedDistribution>,
        raw_scores: RawScores,
    ) -> Result<Self, SystemOneError> {
        let response = Self {
            query_id: query.query_id().clone(),
            status,
            payload,
            provenance,
            validated_distribution,
            raw_scores,
        };
        response.validate_against(query)?;
        Ok(response)
    }

    /// Devuelve el [`QueryId`] al que responde.
    pub const fn query_id(&self) -> &QueryId {
        &self.query_id
    }

    /// Devuelve el estado de resolución ([`ResponseStatus`]).
    pub const fn status(&self) -> ResponseStatus {
        self.status
    }

    /// Devuelve la carga útil tipada de la respuesta.
    pub const fn payload(&self) -> &SystemOnePayload {
        &self.payload
    }

    /// Devuelve los metadatos de procedencia de la respuesta.
    pub const fn provenance(&self) -> &ProvenanceMetadata {
        &self.provenance
    }

    /// Devuelve la referencia de evidencia asociada a la respuesta.
    pub const fn evidence_ref(&self) -> EvidenceRef {
        self.provenance.evidence_ref
    }

    /// Devuelve la distribución normalizada validada, si aplica.
    pub const fn validated_distribution(&self) -> Option<&ValidatedDistribution> {
        self.validated_distribution.as_ref()
    }

    /// Devuelve las puntuaciones crudas preservadas por separado.
    pub const fn raw_scores(&self) -> &RawScores {
        &self.raw_scores
    }

    /// Devuelve la opción seleccionada si la respuesta corresponde a `Choice` contestada.
    pub fn selected_option(&self) -> Option<&OptionId> {
        match &self.payload {
            SystemOnePayload::Choice { selected } => selected.as_ref(),
            _ => None,
        }
    }

    /// Devuelve el veredicto ternario si la respuesta corresponde a `Noul`.
    pub const fn noul_verdict(&self) -> Option<NoulVerdict> {
        match self.payload {
            SystemOnePayload::Noul { verdict, .. } => Some(verdict),
            _ => None,
        }
    }

    /// Valida estrictamente que esta respuesta cumple todas las invariantes de `query`.
    pub fn validate_against(&self, query: &TypedQuery) -> Result<(), SystemOneError> {
        query.validate()?;

        if self.query_id != *query.query_id() {
            return Err(SystemOneError::QueryIdMismatch {
                expected: query.query_id().clone(),
                actual: self.query_id.clone(),
            });
        }

        match (query, &self.payload) {
            (
                TypedQuery::Choice {
                    query_id, options, ..
                },
                SystemOnePayload::Choice { selected },
            ) => {
                let allowed: HashSet<&OptionId> = options.iter().collect();
                validate_raw_scores_membership(query_id, &self.raw_scores, &allowed)?;

                if let Some(dist) = &self.validated_distribution {
                    validate_distribution_membership(query_id, dist, options, &allowed)?;
                }

                match self.status {
                    ResponseStatus::Answered => {
                        let Some(chosen) = selected else {
                            return Err(SystemOneError::InvalidResponse {
                                reason: "answered choice response must include a selected option",
                            });
                        };
                        if !allowed.contains(chosen) {
                            return Err(SystemOneError::UnknownOption {
                                query_id: query_id.clone(),
                                option: chosen.clone(),
                            });
                        }
                        if self.validated_distribution.is_none() {
                            return Err(SystemOneError::InvalidResponse {
                                reason: "answered choice response must include a validated distribution",
                            });
                        }
                    }
                    ResponseStatus::Unknown | ResponseStatus::TimedOut => {
                        if let Some(chosen) = selected {
                            if !allowed.contains(chosen) {
                                return Err(SystemOneError::UnknownOption {
                                    query_id: query_id.clone(),
                                    option: chosen.clone(),
                                });
                            }
                            return Err(SystemOneError::InvalidResponse {
                                reason: "unknown or timed out choice response must not select an option",
                            });
                        }
                    }
                }
                Ok(())
            }
            (
                TypedQuery::Score {
                    query_id, options, ..
                },
                SystemOnePayload::Score,
            ) => {
                let allowed: HashSet<&OptionId> = options.iter().collect();
                validate_raw_scores_membership(query_id, &self.raw_scores, &allowed)?;

                if let Some(dist) = &self.validated_distribution {
                    validate_distribution_membership(query_id, dist, options, &allowed)?;
                }

                if self.status.is_answered() {
                    for option in options {
                        if self.raw_scores.score_of(option).is_none() {
                            return Err(SystemOneError::MissingOption {
                                query_id: query_id.clone(),
                                option: option.clone(),
                            });
                        }
                    }
                    if self.validated_distribution.is_none() {
                        return Err(SystemOneError::InvalidResponse {
                            reason: "answered score response must include a validated distribution",
                        });
                    }
                }
                Ok(())
            }
            (
                TypedQuery::Noul {
                    query_id,
                    hypothesis_id: expected_hyp,
                    ..
                },
                SystemOnePayload::Noul {
                    hypothesis_id: actual_hyp,
                    verdict,
                },
            ) => {
                if expected_hyp != actual_hyp {
                    return Err(SystemOneError::HypothesisMismatch {
                        expected: expected_hyp.clone(),
                        actual: actual_hyp.clone(),
                    });
                }

                let true_opt = OptionId(NOUL_TRUE_OPTION.to_owned());
                let false_opt = OptionId(NOUL_FALSE_OPTION.to_owned());
                let hyp_opt = OptionId(expected_hyp.as_str().to_owned());

                let mut allowed_raw = HashSet::new();
                allowed_raw.insert(&true_opt);
                allowed_raw.insert(&false_opt);
                allowed_raw.insert(&hyp_opt);
                validate_raw_scores_membership(query_id, &self.raw_scores, &allowed_raw)?;

                if let Some(dist) = &self.validated_distribution {
                    let noul_options = [true_opt.clone(), false_opt.clone()];
                    let allowed_dist: HashSet<&OptionId> = noul_options.iter().collect();
                    validate_distribution_membership(query_id, dist, &noul_options, &allowed_dist)?;
                }

                match self.status {
                    ResponseStatus::Answered => {
                        if matches!(verdict, NoulVerdict::Unknown) {
                            return Err(SystemOneError::InvalidResponse {
                                reason: "answered noul response must have True or False verdict",
                            });
                        }
                        if self.validated_distribution.is_none() {
                            return Err(SystemOneError::InvalidResponse {
                                reason: "answered noul response must include a validated distribution",
                            });
                        }
                    }
                    ResponseStatus::Unknown | ResponseStatus::TimedOut => {
                        if !matches!(verdict, NoulVerdict::Unknown) {
                            return Err(SystemOneError::InvalidResponse {
                                reason: "unknown or timed out noul response must report Unknown verdict",
                            });
                        }
                    }
                }
                Ok(())
            }
            (expected_query, actual_payload) => Err(SystemOneError::QueryKindMismatch {
                expected: query_kind_label(expected_query),
                actual: payload_kind_label(actual_payload),
            }),
        }
    }
}

const fn query_kind_label(query: &TypedQuery) -> &'static str {
    match query {
        TypedQuery::Choice { .. } => "choice",
        TypedQuery::Noul { .. } => "noul",
        TypedQuery::Score { .. } => "score",
    }
}

const fn payload_kind_label(payload: &SystemOnePayload) -> &'static str {
    match payload {
        SystemOnePayload::Choice { .. } => "choice",
        SystemOnePayload::Noul { .. } => "noul",
        SystemOnePayload::Score => "score",
    }
}

fn validate_raw_scores_membership(
    query_id: &QueryId,
    raw_scores: &RawScores,
    allowed: &HashSet<&OptionId>,
) -> Result<(), SystemOneError> {
    for (option, _) in raw_scores.entries() {
        if !allowed.contains(option) {
            return Err(SystemOneError::UnknownOption {
                query_id: query_id.clone(),
                option: option.clone(),
            });
        }
    }
    Ok(())
}

fn validate_distribution_membership(
    query_id: &QueryId,
    distribution: &ValidatedDistribution,
    expected_options: &[OptionId],
    allowed: &HashSet<&OptionId>,
) -> Result<(), SystemOneError> {
    for (option, _) in distribution.entries() {
        if !allowed.contains(option) {
            return Err(SystemOneError::UnknownOption {
                query_id: query_id.clone(),
                option: option.clone(),
            });
        }
    }
    for option in expected_options {
        if distribution.probability_of(option).is_none() {
            return Err(SystemOneError::MissingOption {
                query_id: query_id.clone(),
                option: option.clone(),
            });
        }
    }
    Ok(())
}

/// Puerto inyectable de proveedor System One.
///
/// Es deliberadamente independiente de planificadores, actuadores, transporte de red y
/// proveedores externos, y es `object-safe` para permitir inyección tanto dinámica
/// (`&dyn SystemOneProvider`) como estática (`P: SystemOneProvider`).
pub trait SystemOneProvider {
    /// Devuelve un identificador estable del proveedor para auditoría y trazabilidad.
    fn provider_id(&self) -> &str;

    /// Evalúa una [`TypedQuery`] respetando su [`MonotonicDeadline`] según `clock`.
    fn query(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
    ) -> Result<SystemOneResponse, SystemOneError>;
}

/// Comportamiento determinista de un fixture registrado en [`ReplaySystemOneProvider`].
#[derive(Debug, Clone, PartialEq)]
pub enum FixtureBehavior {
    /// Respuesta `Choice` con opción seleccionada, scores crudos y distribución opcional.
    ///
    /// Si `validated_distribution` es `None`, se calcula automáticamente mediante softmax
    /// estable sobre `raw_scores` preservando los scores crudos intactos.
    Choice {
        /// Opción elegida por el fixture.
        selected: OptionId,
        /// Puntuaciones crudas sin normalizar.
        raw_scores: Vec<(OptionId, f64)>,
        /// Distribución normalizada explícita (opcional).
        validated_distribution: Option<Vec<(OptionId, f64)>>,
    },
    /// Respuesta `Noul` con veredicto ternario, scores crudos y distribución opcional.
    Noul {
        /// Hipótesis a la que responde el fixture.
        hypothesis_id: HypothesisId,
        /// Veredicto (`True`, `False` o `Unknown`).
        verdict: NoulVerdict,
        /// Puntuaciones crudas preservadas por separado.
        raw_scores: Vec<(OptionId, f64)>,
        /// Distribución binaria normalizada sobre `["true", "false"]` (opcional).
        validated_distribution: Option<Vec<(OptionId, f64)>>,
    },
    /// Respuesta `Score` con puntuaciones crudas y distribución opcional.
    ///
    /// Si `validated_distribution` es `None`, se deriva mediante softmax estable sobre
    /// `raw_scores` manteniendo los scores crudos por separado.
    Score {
        /// Puntuaciones crudas sin normalizar.
        raw_scores: Vec<(OptionId, f64)>,
        /// Distribución normalizada explícita (opcional).
        validated_distribution: Option<Vec<(OptionId, f64)>>,
    },
    /// Respuesta explícita [`ResponseStatus::Unknown`].
    Unknown {
        /// Puntuaciones crudas opcionales (normalmente vacías).
        raw_scores: Vec<(OptionId, f64)>,
    },
    /// Respuesta explícita [`ResponseStatus::TimedOut`].
    TimedOut,
}

/// Fixture offline determinista asociado a un [`QueryId`].
#[derive(Debug, Clone, PartialEq)]
pub struct SystemOneFixture {
    /// Comportamiento y datos devueltos por el fixture.
    pub behavior: FixtureBehavior,
    /// Metadatos de procedencia asociados a la respuesta emitida.
    pub provenance: ProvenanceMetadata,
    /// Latencia monotónica simulada del proveedor (sin dormir hilos).
    pub simulated_latency: StdDuration,
}

impl SystemOneFixture {
    /// Crea un fixture con latencia simulada cero.
    pub const fn new(behavior: FixtureBehavior, provenance: ProvenanceMetadata) -> Self {
        Self {
            behavior,
            provenance,
            simulated_latency: StdDuration::ZERO,
        }
    }

    /// Asigna una latencia monotónica simulada al fixture.
    pub const fn with_simulated_latency(mut self, simulated_latency: StdDuration) -> Self {
        self.simulated_latency = simulated_latency;
        self
    }
}

/// Proveedor offline determinista de System One respaldado por fixtures en memoria.
///
/// No utiliza red, tokens, hilos ni dependencias externas; valida estrictamente cada
/// respuesta frente a la [`TypedQuery`] y respeta [`MonotonicDeadline`] usando el reloj
/// monotónico inyectado.
#[derive(Debug, Clone)]
pub struct ReplaySystemOneProvider {
    provider_id: String,
    fixtures: BTreeMap<QueryId, SystemOneFixture>,
}

/// Alias semántico de [`ReplaySystemOneProvider`] para bancos de fixtures offline.
pub type FixtureSystemOneProvider = ReplaySystemOneProvider;

impl ReplaySystemOneProvider {
    /// Crea un proveedor offline determinista con un identificador explícito.
    pub fn new(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            fixtures: BTreeMap::new(),
        }
    }

    /// Registra o reemplaza un fixture para `query_id`.
    pub fn register_fixture(&mut self, query_id: QueryId, fixture: SystemOneFixture) -> &mut Self {
        self.fixtures.insert(query_id, fixture);
        self
    }

    /// Versión encadenable de [`ReplaySystemOneProvider::register_fixture`].
    pub fn with_fixture(mut self, query_id: QueryId, fixture: SystemOneFixture) -> Self {
        self.register_fixture(query_id, fixture);
        self
    }

    /// Devuelve el número de fixtures registrados.
    pub fn fixture_count(&self) -> usize {
        self.fixtures.len()
    }

    /// Indica si existe un fixture registrado para `query_id`.
    pub fn contains_fixture(&self, query_id: &QueryId) -> bool {
        self.fixtures.contains_key(query_id)
    }
}

impl Default for ReplaySystemOneProvider {
    fn default() -> Self {
        Self::new("replay-offline")
    }
}

impl SystemOneProvider for ReplaySystemOneProvider {
    fn provider_id(&self) -> &str {
        &self.provider_id
    }

    fn query(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
    ) -> Result<SystemOneResponse, SystemOneError> {
        query.validate()?;

        let query_id = query.query_id();
        let fixture =
            self.fixtures
                .get(query_id)
                .ok_or_else(|| SystemOneError::MissingFixture {
                    query_id: query_id.clone(),
                })?;

        let now = clock.now();
        let completion_instant = now.saturating_add(fixture.simulated_latency);
        let mut provenance = fixture.provenance.clone();
        provenance.received_at = completion_instant;

        let deadline_exceeded = query.deadline().is_some_and(|deadline| {
            deadline.is_expired_at(now) || deadline.is_expired_at(completion_instant)
        });

        if deadline_exceeded || matches!(fixture.behavior, FixtureBehavior::TimedOut) {
            let payload = empty_payload_for_query(query);
            return SystemOneResponse::new_validated(
                query,
                ResponseStatus::TimedOut,
                payload,
                provenance,
                None,
                RawScores::empty(),
            );
        }

        match &fixture.behavior {
            FixtureBehavior::Choice {
                selected,
                raw_scores,
                validated_distribution,
            } => {
                let raw = RawScores::new(raw_scores.clone())?;
                let dist = match validated_distribution {
                    Some(explicit) => ValidatedDistribution::new(explicit.clone())?,
                    None => ValidatedDistribution::from_softmax_scores(&raw)?,
                };
                SystemOneResponse::new_validated(
                    query,
                    ResponseStatus::Answered,
                    SystemOnePayload::Choice {
                        selected: Some(selected.clone()),
                    },
                    provenance,
                    Some(dist),
                    raw,
                )
            }
            FixtureBehavior::Score {
                raw_scores,
                validated_distribution,
            } => {
                let raw = RawScores::new(raw_scores.clone())?;
                let dist = match validated_distribution {
                    Some(explicit) => ValidatedDistribution::new(explicit.clone())?,
                    None => ValidatedDistribution::from_softmax_scores(&raw)?,
                };
                SystemOneResponse::new_validated(
                    query,
                    ResponseStatus::Answered,
                    SystemOnePayload::Score,
                    provenance,
                    Some(dist),
                    raw,
                )
            }
            FixtureBehavior::Noul {
                hypothesis_id,
                verdict,
                raw_scores,
                validated_distribution,
            } => {
                let raw = RawScores::new(raw_scores.clone())?;
                let status = match verdict {
                    NoulVerdict::True | NoulVerdict::False => ResponseStatus::Answered,
                    NoulVerdict::Unknown => ResponseStatus::Unknown,
                };
                let dist = match (status, validated_distribution) {
                    (_, Some(explicit)) => Some(ValidatedDistribution::new(explicit.clone())?),
                    (ResponseStatus::Answered, None) => {
                        Some(ValidatedDistribution::from_softmax_scores(&raw)?)
                    }
                    (ResponseStatus::Unknown | ResponseStatus::TimedOut, None) => None,
                };
                SystemOneResponse::new_validated(
                    query,
                    status,
                    SystemOnePayload::Noul {
                        hypothesis_id: hypothesis_id.clone(),
                        verdict: *verdict,
                    },
                    provenance,
                    dist,
                    raw,
                )
            }
            FixtureBehavior::Unknown { raw_scores } => {
                let raw = RawScores::new(raw_scores.clone())?;
                let payload = empty_payload_for_query(query);
                SystemOneResponse::new_validated(
                    query,
                    ResponseStatus::Unknown,
                    payload,
                    provenance,
                    None,
                    raw,
                )
            }
            FixtureBehavior::TimedOut => unreachable!("handled above"),
        }
    }
}

fn empty_payload_for_query(query: &TypedQuery) -> SystemOnePayload {
    match query {
        TypedQuery::Choice { .. } => SystemOnePayload::Choice { selected: None },
        TypedQuery::Noul { hypothesis_id, .. } => SystemOnePayload::Noul {
            hypothesis_id: hypothesis_id.clone(),
            verdict: NoulVerdict::Unknown,
        },
        TypedQuery::Score { .. } => SystemOnePayload::Score,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{MonotonicInstant, VirtualMonotonicClock};
    use crate::provenance::{
        Confidence, EvidenceRef, FrameId, MessageId, ObservationKind, SensorId, UnitTag,
    };

    fn sample_provenance(message_id: u64, evidence_ref: u64) -> ProvenanceMetadata {
        ProvenanceMetadata {
            message_id: MessageId::new(message_id),
            observed_at: MonotonicInstant::from_nanos(1_000),
            received_at: MonotonicInstant::from_nanos(1_000),
            sensor_id: SensorId::new(101),
            frame_id: FrameId::new("system_one_offline").unwrap(),
            units: UnitTag::new("score").unwrap(),
            confidence: Confidence::new(0.8).unwrap(),
            revision: 1,
            evidence_ref: EvidenceRef::new(evidence_ref),
            claim_kind: ObservationKind::Proposal,
        }
    }

    fn opt(name: &str) -> OptionId {
        OptionId::new(name).unwrap()
    }

    fn qid(name: &str) -> QueryId {
        QueryId::new(name).unwrap()
    }

    fn hid(name: &str) -> HypothesisId {
        HypothesisId::new(name).unwrap()
    }

    #[test]
    fn ac1_replay_provider_answers_choice_noul_score_unknown_and_timeout_offline() {
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(10_000));

        let choice_query = TypedQuery::choice(
            qid("q-choice"),
            vec![opt("left"), opt("right")],
            Some(MonotonicDeadline::from_now(
                &clock,
                StdDuration::from_millis(100),
            )),
        )
        .unwrap();

        let noul_query = TypedQuery::noul(
            qid("q-noul"),
            hid("door_is_locked"),
            Some(MonotonicDeadline::from_now(
                &clock,
                StdDuration::from_millis(100),
            )),
        );

        let score_query = TypedQuery::score(
            qid("q-score"),
            vec![opt("alpha"), opt("beta"), opt("gamma")],
            Some(MonotonicDeadline::from_now(
                &clock,
                StdDuration::from_millis(100),
            )),
        )
        .unwrap();

        let unknown_query =
            TypedQuery::choice(qid("q-unknown"), vec![opt("a"), opt("b")], None).unwrap();

        let timeout_query = TypedQuery::score(
            qid("q-timeout"),
            vec![opt("fast"), opt("slow")],
            Some(MonotonicDeadline::from_now(
                &clock,
                StdDuration::from_millis(50),
            )),
        )
        .unwrap();

        let provider = FixtureSystemOneProvider::new("offline-fixture-bank")
            .with_fixture(
                qid("q-choice"),
                SystemOneFixture::new(
                    FixtureBehavior::Choice {
                        selected: opt("right"),
                        raw_scores: vec![(opt("left"), 1.0), (opt("right"), 3.0)],
                        validated_distribution: None,
                    },
                    sample_provenance(1, 501),
                )
                .with_simulated_latency(StdDuration::from_millis(10)),
            )
            .with_fixture(
                qid("q-noul"),
                SystemOneFixture::new(
                    FixtureBehavior::Noul {
                        hypothesis_id: hid("door_is_locked"),
                        verdict: NoulVerdict::True,
                        raw_scores: vec![(opt("true"), 2.5), (opt("false"), -1.5)],
                        validated_distribution: Some(vec![(opt("true"), 0.9), (opt("false"), 0.1)]),
                    },
                    sample_provenance(2, 502),
                ),
            )
            .with_fixture(
                qid("q-score"),
                SystemOneFixture::new(
                    FixtureBehavior::Score {
                        raw_scores: vec![
                            (opt("alpha"), 10.0),
                            (opt("beta"), 10.0),
                            (opt("gamma"), 10.0),
                        ],
                        validated_distribution: None,
                    },
                    sample_provenance(3, 503),
                ),
            )
            .with_fixture(
                qid("q-unknown"),
                SystemOneFixture::new(
                    FixtureBehavior::Unknown {
                        raw_scores: Vec::new(),
                    },
                    sample_provenance(4, 504),
                ),
            )
            .with_fixture(
                qid("q-timeout"),
                SystemOneFixture::new(
                    FixtureBehavior::Score {
                        raw_scores: vec![(opt("fast"), 1.0), (opt("slow"), 0.5)],
                        validated_distribution: None,
                    },
                    sample_provenance(5, 505),
                )
                .with_simulated_latency(StdDuration::from_millis(80)),
            );

        assert_eq!(provider.provider_id(), "offline-fixture-bank");
        assert_eq!(provider.fixture_count(), 5);
        assert!(provider.contains_fixture(&qid("q-choice")));

        // Choice
        let choice_resp = provider.query(&choice_query, &clock).unwrap();
        assert_eq!(choice_resp.status(), ResponseStatus::Answered);
        assert_eq!(choice_resp.selected_option(), Some(&opt("right")));
        assert_eq!(choice_resp.evidence_ref(), EvidenceRef::new(501));
        let choice_dist = choice_resp.validated_distribution().unwrap();
        assert!(choice_dist.probability_of(&opt("right")).unwrap() > 0.8);
        assert!(choice_dist.to_belief_state().is_ok());

        // Noul
        let noul_resp = provider.query(&noul_query, &clock).unwrap();
        assert_eq!(noul_resp.status(), ResponseStatus::Answered);
        assert_eq!(noul_resp.noul_verdict(), Some(NoulVerdict::True));
        assert_eq!(noul_resp.evidence_ref(), EvidenceRef::new(502));
        let noul_dist = noul_resp.validated_distribution().unwrap();
        assert!((noul_dist.probability_of(&opt("true")).unwrap() - 0.9).abs() < 1e-12);
        assert!((noul_dist.probability_of(&opt("false")).unwrap() - 0.1).abs() < 1e-12);

        // Score
        let score_resp = provider.query(&score_query, &clock).unwrap();
        assert_eq!(score_resp.status(), ResponseStatus::Answered);
        assert_eq!(score_resp.evidence_ref(), EvidenceRef::new(503));
        let score_dist = score_resp.validated_distribution().unwrap();
        for id in [opt("alpha"), opt("beta"), opt("gamma")] {
            assert!((score_dist.probability_of(&id).unwrap() - (1.0 / 3.0)).abs() < 1e-12);
            assert_eq!(score_resp.raw_scores().score_of(&id), Some(10.0));
        }

        // Unknown
        let unknown_resp = provider.query(&unknown_query, &clock).unwrap();
        assert_eq!(unknown_resp.status(), ResponseStatus::Unknown);
        assert_eq!(unknown_resp.selected_option(), None);
        assert!(unknown_resp.validated_distribution().is_none());

        // Timeout by monotonic simulated latency exceeding deadline (80 ms > 50 ms)
        let timeout_resp = provider.query(&timeout_query, &clock).unwrap();
        assert_eq!(timeout_resp.status(), ResponseStatus::TimedOut);
        assert!(timeout_resp.validated_distribution().is_none());
        assert!(timeout_resp.raw_scores().is_empty());

        // Advancing virtual clock past choice deadline turns choice_query into TimedOut
        clock.advance(StdDuration::from_millis(120));
        let expired_choice = provider.query(&choice_query, &clock).unwrap();
        assert_eq!(expired_choice.status(), ResponseStatus::TimedOut);
        assert_eq!(expired_choice.selected_option(), None);
        assert!(expired_choice.validated_distribution().is_none());
    }

    #[test]
    fn ac2_rejects_unknown_options_non_finite_scores_and_preserves_raw_scores_separately() {
        let clock = VirtualMonotonicClock::new();
        let query = TypedQuery::score(qid("q-validate"), vec![opt("a"), opt("b")], None).unwrap();

        // 1. Rechaza opción inexistente en raw_scores.
        let bad_option_provider = ReplaySystemOneProvider::default().with_fixture(
            qid("q-validate"),
            SystemOneFixture::new(
                FixtureBehavior::Score {
                    raw_scores: vec![(opt("a"), 1.0), (opt("ghost"), 2.0)],
                    validated_distribution: None,
                },
                sample_provenance(10, 601),
            ),
        );
        assert!(matches!(
            bad_option_provider.query(&query, &clock),
            Err(SystemOneError::UnknownOption { option, .. }) if option == opt("ghost")
        ));

        // 2. Rechaza opción inexistente en Choice::selected.
        let choice_query =
            TypedQuery::choice(qid("q-choice-bad"), vec![opt("a"), opt("b")], None).unwrap();
        let bad_selected_provider = ReplaySystemOneProvider::default().with_fixture(
            qid("q-choice-bad"),
            SystemOneFixture::new(
                FixtureBehavior::Choice {
                    selected: opt("nonexistent"),
                    raw_scores: vec![(opt("a"), 1.0), (opt("b"), 2.0)],
                    validated_distribution: None,
                },
                sample_provenance(11, 602),
            ),
        );
        assert!(matches!(
            bad_selected_provider.query(&choice_query, &clock),
            Err(SystemOneError::UnknownOption { option, .. }) if option == opt("nonexistent")
        ));

        // 3. Rechaza scores no finitos (NaN, +inf, -inf).
        for bad_score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let non_finite_provider = ReplaySystemOneProvider::default().with_fixture(
                qid("q-validate"),
                SystemOneFixture::new(
                    FixtureBehavior::Score {
                        raw_scores: vec![(opt("a"), bad_score), (opt("b"), 1.0)],
                        validated_distribution: None,
                    },
                    sample_provenance(12, 603),
                ),
            );
            assert!(matches!(
                non_finite_provider.query(&query, &clock),
                Err(SystemOneError::NonFiniteScore { option, .. }) if option == opt("a")
            ));
        }

        // 4. Rechaza distribución inválida (suma != 1.0 o probabilidad negativa).
        let bad_dist_provider = ReplaySystemOneProvider::default().with_fixture(
            qid("q-validate"),
            SystemOneFixture::new(
                FixtureBehavior::Score {
                    raw_scores: vec![(opt("a"), 5.0), (opt("b"), 15.0)],
                    validated_distribution: Some(vec![(opt("a"), 0.5), (opt("b"), 0.2)]),
                },
                sample_provenance(13, 604),
            ),
        );
        assert!(matches!(
            bad_dist_provider.query(&query, &clock),
            Err(SystemOneError::InvalidDistributionSum { .. })
        ));

        // 5. Preserva raw_scores por separado de validated_distribution sin mezclarlos ni borrarlos.
        let valid_provider = ReplaySystemOneProvider::default().with_fixture(
            qid("q-validate"),
            SystemOneFixture::new(
                FixtureBehavior::Score {
                    raw_scores: vec![(opt("a"), -4.25), (opt("b"), 12.75)],
                    validated_distribution: Some(vec![(opt("a"), 0.2), (opt("b"), 0.8)]),
                },
                sample_provenance(14, 605),
            ),
        );
        let resp = valid_provider.query(&query, &clock).unwrap();
        let dist = resp.validated_distribution().unwrap();
        let raw = resp.raw_scores();

        assert_eq!(raw.score_of(&opt("a")), Some(-4.25));
        assert_eq!(raw.score_of(&opt("b")), Some(12.75));
        assert_eq!(dist.probability_of(&opt("a")), Some(0.2));
        assert_eq!(dist.probability_of(&opt("b")), Some(0.8));
        assert_ne!(
            raw.score_of(&opt("a")).unwrap(),
            dist.probability_of(&opt("a")).unwrap()
        );
        assert_ne!(
            raw.score_of(&opt("b")).unwrap(),
            dist.probability_of(&opt("b")).unwrap()
        );
    }

    struct UniformFallbackProvider {
        id: &'static str,
    }

    impl SystemOneProvider for UniformFallbackProvider {
        fn provider_id(&self) -> &str {
            self.id
        }

        fn query(
            &self,
            query: &TypedQuery,
            clock: &dyn MonotonicClock,
        ) -> Result<SystemOneResponse, SystemOneError> {
            query.validate()?;
            let mut provenance = sample_provenance(99, 999);
            provenance.received_at = clock.now();

            match query {
                TypedQuery::Choice { options, .. } => {
                    let p = 1.0 / (options.len() as f64);
                    let dist = ValidatedDistribution::new(
                        options.iter().cloned().map(|o| (o, p)).collect(),
                    )?;
                    let raw = RawScores::new(options.iter().cloned().map(|o| (o, 0.0)).collect())?;
                    SystemOneResponse::new_validated(
                        query,
                        ResponseStatus::Answered,
                        SystemOnePayload::Choice {
                            selected: Some(options[0].clone()),
                        },
                        provenance,
                        Some(dist),
                        raw,
                    )
                }
                TypedQuery::Score { options, .. } => {
                    let p = 1.0 / (options.len() as f64);
                    let dist = ValidatedDistribution::new(
                        options.iter().cloned().map(|o| (o, p)).collect(),
                    )?;
                    let raw = RawScores::new(options.iter().cloned().map(|o| (o, 1.0)).collect())?;
                    SystemOneResponse::new_validated(
                        query,
                        ResponseStatus::Answered,
                        SystemOnePayload::Score,
                        provenance,
                        Some(dist),
                        raw,
                    )
                }
                TypedQuery::Noul { hypothesis_id, .. } => SystemOneResponse::new_validated(
                    query,
                    ResponseStatus::Unknown,
                    SystemOnePayload::Noul {
                        hypothesis_id: hypothesis_id.clone(),
                        verdict: NoulVerdict::Unknown,
                    },
                    provenance,
                    None,
                    RawScores::empty(),
                ),
            }
        }
    }

    #[test]
    fn ac3_providers_are_swappable_via_trait_object_and_generics_without_planner() {
        fn evaluate_dyn(
            provider: &dyn SystemOneProvider,
            query: &TypedQuery,
            clock: &dyn MonotonicClock,
        ) -> Result<(String, Option<OptionId>, f64), SystemOneError> {
            let resp = provider.query(query, clock)?;
            let selected = resp.selected_option().cloned();
            let top_prob = resp
                .validated_distribution()
                .and_then(|d| d.entries().iter().map(|(_, p)| *p).reduce(f64::max))
                .unwrap_or(0.0);
            Ok((provider.provider_id().to_owned(), selected, top_prob))
        }

        fn evaluate_generic<P: SystemOneProvider>(
            provider: &P,
            query: &TypedQuery,
            clock: &dyn MonotonicClock,
        ) -> Result<SystemOneResponse, SystemOneError> {
            provider.query(query, clock)
        }

        let clock = VirtualMonotonicClock::new();
        let query =
            TypedQuery::choice(qid("q-swap"), vec![opt("stay"), opt("move")], None).unwrap();

        let replay_provider = ReplaySystemOneProvider::new("replay-a").with_fixture(
            qid("q-swap"),
            SystemOneFixture::new(
                FixtureBehavior::Choice {
                    selected: opt("move"),
                    raw_scores: vec![(opt("stay"), 0.0), (opt("move"), 4.0)],
                    validated_distribution: Some(vec![(opt("stay"), 0.1), (opt("move"), 0.9)]),
                },
                sample_provenance(20, 700),
            ),
        );
        let fallback_provider = UniformFallbackProvider {
            id: "uniform-fallback-b",
        };

        // Inyección dinámica (&dyn SystemOneProvider).
        let (id_a, sel_a, prob_a) = evaluate_dyn(&replay_provider, &query, &clock).unwrap();
        let (id_b, sel_b, prob_b) = evaluate_dyn(&fallback_provider, &query, &clock).unwrap();

        assert_eq!(id_a, "replay-a");
        assert_eq!(sel_a, Some(opt("move")));
        assert!((prob_a - 0.9).abs() < 1e-12);

        assert_eq!(id_b, "uniform-fallback-b");
        assert_eq!(sel_b, Some(opt("stay")));
        assert!((prob_b - 0.5).abs() < 1e-12);

        // Inyección estática (P: SystemOneProvider).
        let resp_a = evaluate_generic(&replay_provider, &query, &clock).unwrap();
        let resp_b = evaluate_generic(&fallback_provider, &query, &clock).unwrap();
        assert_ne!(resp_a.selected_option(), resp_b.selected_option());
        assert_ne!(resp_a.evidence_ref(), resp_b.evidence_ref());
    }
}
