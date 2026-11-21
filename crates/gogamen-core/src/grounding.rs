//! Grounding semántico de mensajes situados sobre ontología y catálogo locales (`std`-only).
//!
//! Este módulo conecta [`SituatedMessage`] con [`SemanticKnowledgeStore`],
//! [`CapabilityCatalog`] y [`SystemOneProvider`] bajo un contrato estricto de cierre
//! ontológico:
//! - Toda consulta hacia System One se construye únicamente con opciones candidatas
//!   previamente registradas en el store semántico o en el catálogo de capacidades.
//! - Ninguna respuesta del proveedor puede fabricar [`EntityId`] ni [`OperationId`]
//!   fuera de los candidatos registrados.
//! - Ante `ResponseStatus::Unknown`, agotamiento de plazo o ambigüedad entre varios
//!   candidatos, el grounder se abstiene explícitamente ([`GroundedOutcome::Abstained`])
//!   o emite una pregunta de aclaración ([`GroundedOutcome::ClarificationNeeded`]).

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::capability::{CapabilityCatalog, OperationId};
use crate::clock::{MonotonicClock, MonotonicDeadline};
use crate::communication::{CommPayload, CommunicationError, SituatedMessage};
use crate::observability::EntityId;
use crate::provenance::{Confidence, EvidenceRef};
use crate::semantic_memory::{
    GroundingCandidate, GroundingFilter, PredicateId, PropositionKey, SemanticKnowledgeStore,
};
use crate::system_one::{
    OptionId, QueryId, ResponseStatus, SystemOneError, SystemOneProvider, SystemOneResponse,
    TypedQuery,
};

/// Acto de habla clasificado para un [`SituatedMessage`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessageSpeechAct {
    /// Aviso informativo o alerta.
    Notice,
    /// Orden o mandato imperativo.
    Order,
    /// Preferencia o deseo expresado por el interlocutor.
    Preference,
    /// Negación o rechazo explícito.
    Negation,
    /// Referente ambiguo que requiere desambiguación o abstención.
    AmbiguousReferent,
    /// Acto de habla no determinado.
    Unknown,
}

impl MessageSpeechAct {
    /// Devuelve `true` si el acto corresponde a una orden ([`MessageSpeechAct::Order`]).
    pub const fn is_order(self) -> bool {
        matches!(self, Self::Order)
    }

    /// Devuelve `true` si el acto corresponde a una negación ([`MessageSpeechAct::Negation`]).
    pub const fn is_negation(self) -> bool {
        matches!(self, Self::Negation)
    }

    /// Devuelve `true` si el acto corresponde a un referente ambiguo.
    pub const fn is_ambiguous_referent(self) -> bool {
        matches!(self, Self::AmbiguousReferent)
    }
}

impl fmt::Display for MessageSpeechAct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Notice => "notice",
            Self::Order => "order",
            Self::Preference => "preference",
            Self::Negation => "negation",
            Self::AmbiguousReferent => "ambiguous_referent",
            Self::Unknown => "unknown",
        };
        f.write_str(label)
    }
}

/// Motivo explícito de abstención durante el grounding semántico.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstentionReason {
    /// El proveedor System One devolvió `ResponseStatus::Unknown` o el mensaje es indeterminado.
    UnknownResponse,
    /// Existen múltiples candidatos ambiguos y no se pudo seleccionar uno único.
    AmbiguousCandidates(Vec<EntityId>),
    /// No hay entidades registradas coincidentes o el proveedor devolvió una entidad no registrada.
    UnregisteredEntity,
    /// No hay operaciones registradas o el proveedor devolvió una operación no registrada.
    UnregisteredOperation,
    /// El proveedor System One agotó su plazo monotónico (`ResponseStatus::TimedOut`).
    ProviderTimedOut,
}

/// Resultado tipado del proceso de grounding semántico de un mensaje.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundedOutcome {
    /// El mensaje se aterrizó sobre entidades, proposiciones u operaciones registradas.
    Grounded {
        /// Acto de habla clasificado del mensaje.
        speech_act: MessageSpeechAct,
        /// Entidad registrada seleccionada, si aplica.
        entity_id: Option<EntityId>,
        /// Clave de proposición emparejada en la ontología local, si aplica.
        proposition_key: Option<PropositionKey>,
        /// Operación registrada en [`CapabilityCatalog`], si aplica.
        operation: Option<OperationId>,
        /// Confianza asociada a la resolución (si el proveedor la aporta).
        confidence: Option<Confidence>,
        /// Referencia de evidencia de procedencia del proveedor System One.
        evidence_ref: Option<EvidenceRef>,
    },
    /// Abstención explícita sin fabricar entidades ni acciones arbitrarias.
    Abstained {
        /// Motivo determinista de la abstención.
        reason: AbstentionReason,
    },
    /// Solicitud explícita de aclaración ante múltiples candidatos registrados.
    ClarificationNeeded {
        /// Lista determinista de entidades candidatas registradas.
        candidates: Vec<EntityId>,
        /// Pregunta tipada [`CommPayload::Question`] para desambiguación.
        question: CommPayload,
    },
}

/// Errores estructurales del módulo de grounding semántico.
#[derive(Debug, Clone, PartialEq)]
pub enum GroundingError {
    /// Error al construir o consultar [`SystemOneProvider`].
    SystemOne(SystemOneError),
    /// Error al construir la pregunta tipada de aclaración [`CommPayload::Question`].
    Communication(CommunicationError),
    /// Margen de ambigüedad inválido (debe ser finito y estar en `[0.0, 1.0]`).
    InvalidAmbiguityMargin(f64),
}

impl fmt::Display for GroundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SystemOne(err) => write!(f, "system one grounding error: {err}"),
            Self::Communication(err) => write!(f, "communication payload error: {err}"),
            Self::InvalidAmbiguityMargin(margin) => {
                write!(
                    f,
                    "invalid ambiguity_margin {margin}: must be finite and within [0.0, 1.0]"
                )
            }
        }
    }
}

impl Error for GroundingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::SystemOne(err) => Some(err),
            Self::Communication(err) => Some(err),
            Self::InvalidAmbiguityMargin(_) => None,
        }
    }
}

impl From<SystemOneError> for GroundingError {
    fn from(value: SystemOneError) -> Self {
        Self::SystemOne(value)
    }
}

impl From<CommunicationError> for GroundingError {
    fn from(value: CommunicationError) -> Self {
        Self::Communication(value)
    }
}

/// Modo de consulta tipada enviado a [`SystemOneProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GroundingQueryMode {
    /// Selección discreta mediante [`TypedQuery::Choice`].
    #[default]
    Choice,
    /// Puntuación cuantitativa mediante [`TypedQuery::Score`].
    Score,
}

/// Política para resolver una [`OperationId`] desde [`CapabilityCatalog`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperationResolutionPolicy {
    /// Resuelve operación únicamente cuando el acto de habla es [`MessageSpeechAct::Order`].
    #[default]
    AutoForOrders,
    /// Intenta resolver operación para cualquier mensaje aterrizado.
    Always,
    /// No consulta operaciones en el catálogo de capacidades.
    Never,
}

/// Configuración determinista de [`SemanticGrounder`].
#[derive(Debug, Clone, PartialEq)]
pub struct GroundingConfig {
    /// Tipo de consulta tipada (`Choice` o `Score`).
    pub query_mode: GroundingQueryMode,
    /// Si es `true`, múltiples candidatos ambiguos emiten [`GroundedOutcome::ClarificationNeeded`];
    /// si es `false`, emiten [`GroundedOutcome::Abstained`] con [`AbstentionReason::AmbiguousCandidates`].
    pub clarify_on_ambiguity: bool,
    /// Margen de probabilidad para considerar empate ambiguo en consultas `Score`.
    pub ambiguity_margin: f64,
    /// Política de resolución de operaciones registradas en [`CapabilityCatalog`].
    pub operation_policy: OperationResolutionPolicy,
}

impl Default for GroundingConfig {
    fn default() -> Self {
        Self {
            query_mode: GroundingQueryMode::Choice,
            clarify_on_ambiguity: true,
            ambiguity_margin: 0.05,
            operation_policy: OperationResolutionPolicy::AutoForOrders,
        }
    }
}

impl GroundingConfig {
    /// Valida los parámetros de configuración.
    pub fn validate(&self) -> Result<(), GroundingError> {
        if !self.ambiguity_margin.is_finite() || !(0.0..=1.0).contains(&self.ambiguity_margin) {
            return Err(GroundingError::InvalidAmbiguityMargin(
                self.ambiguity_margin,
            ));
        }
        Ok(())
    }
}

/// Parámetros de una solicitud de grounding sobre un [`SituatedMessage`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroundingRequest {
    /// Predicado opcional para filtrar proposiciones en [`SemanticKnowledgeStore`].
    pub predicate: Option<PredicateId>,
    /// Filtro estructural sobre entidades y proposiciones candidatas.
    pub filter: GroundingFilter,
    /// Identificador explícito para la consulta de entidad hacia [`SystemOneProvider`].
    pub entity_query_id: Option<QueryId>,
    /// Identificador explícito para la consulta de operación hacia [`SystemOneProvider`].
    pub operation_query_id: Option<QueryId>,
    /// Plazo monotónico opcional para las consultas a [`SystemOneProvider`].
    pub deadline: Option<MonotonicDeadline>,
    /// Enunciado opcional para la pregunta de aclaración si hay ambigüedad.
    pub clarification_prompt: Option<String>,
}

impl GroundingRequest {
    /// Crea una solicitud de grounding con valores por defecto.
    pub fn new() -> Self {
        Self::default()
    }

    /// Asigna un predicado para filtrar candidatos en [`SemanticKnowledgeStore`].
    pub fn with_predicate(mut self, predicate: PredicateId) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Asigna un [`GroundingFilter`] explícito.
    pub fn with_filter(mut self, filter: GroundingFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Asigna un [`QueryId`] explícito para la consulta de entidad.
    pub fn with_entity_query_id(mut self, query_id: QueryId) -> Self {
        self.entity_query_id = Some(query_id);
        self
    }

    /// Asigna un [`QueryId`] explícito para la consulta de operación.
    pub fn with_operation_query_id(mut self, query_id: QueryId) -> Self {
        self.operation_query_id = Some(query_id);
        self
    }

    /// Asigna un plazo monotónico explícito.
    pub const fn with_deadline(mut self, deadline: MonotonicDeadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Asigna un enunciado personalizado para preguntas de desambiguación.
    pub fn with_clarification_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.clarification_prompt = Some(prompt.into());
        self
    }
}

/// Grounder semántico determinista que aterriza mensajes situados sobre ontología y catálogo locales.
#[derive(Debug, Clone, Default)]
pub struct SemanticGrounder {
    config: GroundingConfig,
}

impl SemanticGrounder {
    /// Crea un [`SemanticGrounder`] con la configuración por defecto.
    pub fn new() -> Self {
        Self::default()
    }

    /// Crea un [`SemanticGrounder`] validando una [`GroundingConfig`] explícita.
    pub fn with_config(config: GroundingConfig) -> Result<Self, GroundingError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Devuelve la configuración activa del grounder.
    pub const fn config(&self) -> &GroundingConfig {
        &self.config
    }

    /// Clasifica de forma determinista y bilingüe (ES/EN) el [`MessageSpeechAct`] de un mensaje.
    pub fn classify_speech_act(message: &SituatedMessage) -> MessageSpeechAct {
        classify_payload_speech_act(message.payload())
    }

    /// Aterriza un [`SituatedMessage`] usando únicamente candidatos registrados en `store` y
    /// operaciones registradas en `catalog`, consultando a `provider` mediante [`TypedQuery`].
    pub fn ground(
        &self,
        message: &SituatedMessage,
        store: &SemanticKnowledgeStore,
        catalog: &CapabilityCatalog,
        provider: &dyn SystemOneProvider,
        clock: &dyn MonotonicClock,
        request: &GroundingRequest,
    ) -> Result<GroundedOutcome, GroundingError> {
        self.config.validate()?;

        let speech_act = Self::classify_speech_act(message);
        if matches!(message.payload(), CommPayload::NoResponse { .. })
            || speech_act == MessageSpeechAct::Unknown
        {
            return Ok(GroundedOutcome::Abstained {
                reason: AbstentionReason::UnknownResponse,
            });
        }

        let candidates =
            store.resolve_grounding_candidates(request.predicate.as_ref(), &request.filter);
        if candidates.is_empty() {
            return Ok(GroundedOutcome::Abstained {
                reason: AbstentionReason::UnregisteredEntity,
            });
        }

        let mut entity_options = Vec::with_capacity(candidates.len());
        let mut entity_by_option: BTreeMap<OptionId, &GroundingCandidate> = BTreeMap::new();
        for candidate in &candidates {
            let opt_id = entity_option_id(candidate.entity.id())?;
            entity_options.push(opt_id.clone());
            entity_by_option.insert(opt_id, candidate);
        }

        let entity_query_id = match &request.entity_query_id {
            Some(id) => id.clone(),
            None => QueryId::new(format!("ground-entity-{}", message.message_id().as_u64()))?,
        };
        let deadline = request.deadline.or_else(|| message.deadline());
        let entity_query = build_typed_query(
            self.config.query_mode,
            entity_query_id,
            entity_options,
            deadline,
        )?;
        let entity_response = provider.query(&entity_query, clock)?;

        let (selected_candidate, entity_confidence) = match self.resolve_entity_response(
            &entity_response,
            &entity_by_option,
            &candidates,
            speech_act,
            request,
        )? {
            EntityResolution::Resolved {
                candidate,
                confidence,
            } => (candidate, confidence),
            EntityResolution::Outcome(outcome) => return Ok(outcome),
        };

        // Verificación defensiva frente a la ontología local: nunca aceptar un EntityId ausente.
        let resolved_entity_id = selected_candidate.entity.id();
        if store.get_entity(resolved_entity_id).is_none() {
            return Ok(GroundedOutcome::Abstained {
                reason: AbstentionReason::UnregisteredEntity,
            });
        }

        let proposition_key = selected_candidate
            .matched_propositions
            .first()
            .map(|prop| prop.key().clone());

        let should_resolve_operation = match self.config.operation_policy {
            OperationResolutionPolicy::Always => true,
            OperationResolutionPolicy::Never => false,
            OperationResolutionPolicy::AutoForOrders => speech_act.is_order(),
        };

        let mut final_evidence = Some(entity_response.evidence_ref());
        let mut final_confidence = entity_confidence;
        let operation = if should_resolve_operation {
            match self.resolve_operation(message, catalog, provider, clock, request, deadline)? {
                OperationResolution::Resolved {
                    operation_id,
                    confidence,
                    evidence_ref,
                } => {
                    if !catalog.contains_operation(&operation_id) {
                        return Ok(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnregisteredOperation,
                        });
                    }
                    final_evidence = Some(evidence_ref);
                    if final_confidence.is_none() {
                        final_confidence = confidence;
                    }
                    Some(operation_id)
                }
                OperationResolution::Abstained(reason) => {
                    return Ok(GroundedOutcome::Abstained { reason });
                }
            }
        } else {
            None
        };

        Ok(GroundedOutcome::Grounded {
            speech_act,
            entity_id: Some(resolved_entity_id),
            proposition_key,
            operation,
            confidence: final_confidence,
            evidence_ref: final_evidence,
        })
    }

    fn resolve_entity_response<'a>(
        &self,
        response: &SystemOneResponse,
        entity_by_option: &BTreeMap<OptionId, &'a GroundingCandidate>,
        candidates: &[GroundingCandidate],
        speech_act: MessageSpeechAct,
        request: &GroundingRequest,
    ) -> Result<EntityResolution<'a>, GroundingError> {
        match response.status() {
            ResponseStatus::TimedOut => Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                reason: AbstentionReason::ProviderTimedOut,
            })),
            ResponseStatus::Unknown => {
                if speech_act.is_ambiguous_referent()
                    && candidates.len() > 1
                    && self.config.clarify_on_ambiguity
                {
                    let ids: Vec<EntityId> = candidates.iter().map(|c| c.entity.id()).collect();
                    let question = build_clarification_question(request, &ids)?;
                    Ok(EntityResolution::Outcome(
                        GroundedOutcome::ClarificationNeeded {
                            candidates: ids,
                            question,
                        },
                    ))
                } else {
                    Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                        reason: AbstentionReason::UnknownResponse,
                    }))
                }
            }
            ResponseStatus::Answered => match self.config.query_mode {
                GroundingQueryMode::Choice => {
                    let Some(selected) = response.selected_option() else {
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnknownResponse,
                        }));
                    };
                    let Some(candidate) = entity_by_option.get(selected) else {
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnregisteredEntity,
                        }));
                    };
                    let confidence = response
                        .validated_distribution()
                        .and_then(|dist| dist.probability_of(selected))
                        .and_then(|p| Confidence::new(p).ok());
                    Ok(EntityResolution::Resolved {
                        candidate,
                        confidence,
                    })
                }
                GroundingQueryMode::Score => {
                    let Some(dist) = response.validated_distribution() else {
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnknownResponse,
                        }));
                    };
                    let mut ranked: Vec<(&OptionId, f64)> = dist
                        .entries()
                        .iter()
                        .map(|(opt, prob)| (opt, *prob))
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

                    let Some(&(top_opt, top_prob)) = ranked.first() else {
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnknownResponse,
                        }));
                    };

                    let ambiguous_opts: Vec<&OptionId> = ranked
                        .iter()
                        .filter(|(_, prob)| (top_prob - *prob) <= self.config.ambiguity_margin)
                        .map(|(opt, _)| *opt)
                        .collect();

                    if ambiguous_opts.len() > 1 {
                        let mut ambiguous_ids = Vec::with_capacity(ambiguous_opts.len());
                        for opt in ambiguous_opts {
                            let Some(cand) = entity_by_option.get(opt) else {
                                return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                                    reason: AbstentionReason::UnregisteredEntity,
                                }));
                            };
                            ambiguous_ids.push(cand.entity.id());
                        }
                        if self.config.clarify_on_ambiguity {
                            let question = build_clarification_question(request, &ambiguous_ids)?;
                            return Ok(EntityResolution::Outcome(
                                GroundedOutcome::ClarificationNeeded {
                                    candidates: ambiguous_ids,
                                    question,
                                },
                            ));
                        }
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::AmbiguousCandidates(ambiguous_ids),
                        }));
                    }

                    let Some(candidate) = entity_by_option.get(top_opt) else {
                        return Ok(EntityResolution::Outcome(GroundedOutcome::Abstained {
                            reason: AbstentionReason::UnregisteredEntity,
                        }));
                    };
                    let confidence = Confidence::new(top_prob).ok();
                    Ok(EntityResolution::Resolved {
                        candidate,
                        confidence,
                    })
                }
            },
        }
    }

    fn resolve_operation(
        &self,
        message: &SituatedMessage,
        catalog: &CapabilityCatalog,
        provider: &dyn SystemOneProvider,
        clock: &dyn MonotonicClock,
        request: &GroundingRequest,
        deadline: Option<MonotonicDeadline>,
    ) -> Result<OperationResolution, GroundingError> {
        let valid_ops = catalog.operation_ids();
        if valid_ops.is_empty() {
            return Ok(OperationResolution::Abstained(
                AbstentionReason::UnregisteredOperation,
            ));
        }

        let mut op_options = Vec::with_capacity(valid_ops.len());
        let mut op_by_option: BTreeMap<OptionId, OperationId> = BTreeMap::new();
        for op_id in valid_ops {
            let opt_id = operation_option_id(&op_id)?;
            op_options.push(opt_id.clone());
            op_by_option.insert(opt_id, op_id);
        }

        let op_query_id = match &request.operation_query_id {
            Some(id) => id.clone(),
            None => QueryId::new(format!("ground-op-{}", message.message_id().as_u64()))?,
        };
        let op_query =
            build_typed_query(self.config.query_mode, op_query_id, op_options, deadline)?;
        let op_response = provider.query(&op_query, clock)?;

        match op_response.status() {
            ResponseStatus::TimedOut => Ok(OperationResolution::Abstained(
                AbstentionReason::ProviderTimedOut,
            )),
            ResponseStatus::Unknown => Ok(OperationResolution::Abstained(
                AbstentionReason::UnknownResponse,
            )),
            ResponseStatus::Answered => match self.config.query_mode {
                GroundingQueryMode::Choice => {
                    let Some(selected) = op_response.selected_option() else {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnknownResponse,
                        ));
                    };
                    let Some(op_id) = op_by_option.get(selected).cloned() else {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnregisteredOperation,
                        ));
                    };
                    let confidence = op_response
                        .validated_distribution()
                        .and_then(|dist| dist.probability_of(selected))
                        .and_then(|p| Confidence::new(p).ok());
                    Ok(OperationResolution::Resolved {
                        operation_id: op_id,
                        confidence,
                        evidence_ref: op_response.evidence_ref(),
                    })
                }
                GroundingQueryMode::Score => {
                    let Some(dist) = op_response.validated_distribution() else {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnknownResponse,
                        ));
                    };
                    let mut ranked: Vec<(&OptionId, f64)> = dist
                        .entries()
                        .iter()
                        .map(|(opt, prob)| (opt, *prob))
                        .collect();
                    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));

                    let Some(&(top_opt, top_prob)) = ranked.first() else {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnknownResponse,
                        ));
                    };
                    let ambiguous_count = ranked
                        .iter()
                        .filter(|(_, prob)| (top_prob - *prob) <= self.config.ambiguity_margin)
                        .count();
                    if ambiguous_count > 1 {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnknownResponse,
                        ));
                    }
                    let Some(op_id) = op_by_option.get(top_opt).cloned() else {
                        return Ok(OperationResolution::Abstained(
                            AbstentionReason::UnregisteredOperation,
                        ));
                    };
                    Ok(OperationResolution::Resolved {
                        operation_id: op_id,
                        confidence: Confidence::new(top_prob).ok(),
                        evidence_ref: op_response.evidence_ref(),
                    })
                }
            },
        }
    }
}

enum EntityResolution<'a> {
    Resolved {
        candidate: &'a GroundingCandidate,
        confidence: Option<Confidence>,
    },
    Outcome(GroundedOutcome),
}

enum OperationResolution {
    Resolved {
        operation_id: OperationId,
        confidence: Option<Confidence>,
        evidence_ref: EvidenceRef,
    },
    Abstained(AbstentionReason),
}

fn entity_option_id(entity_id: EntityId) -> Result<OptionId, SystemOneError> {
    OptionId::new(format!("entity:{}", entity_id.as_u64()))
}

fn operation_option_id(op_id: &OperationId) -> Result<OptionId, SystemOneError> {
    OptionId::new(format!("op:{}", op_id.as_str()))
}

fn build_typed_query(
    mode: GroundingQueryMode,
    query_id: QueryId,
    options: Vec<OptionId>,
    deadline: Option<MonotonicDeadline>,
) -> Result<TypedQuery, SystemOneError> {
    match mode {
        GroundingQueryMode::Choice => TypedQuery::choice(query_id, options, deadline),
        GroundingQueryMode::Score => TypedQuery::score(query_id, options, deadline),
    }
}

fn build_clarification_question(
    request: &GroundingRequest,
    candidates: &[EntityId],
) -> Result<CommPayload, GroundingError> {
    let prompt = request
        .clarification_prompt
        .as_deref()
        .unwrap_or("Which registered entity do you mean?");
    let mut expected_options = Vec::with_capacity(candidates.len());
    for id in candidates {
        expected_options.push(entity_option_id(*id)?);
    }
    Ok(CommPayload::question(prompt, expected_options)?)
}

fn classify_payload_speech_act(payload: &CommPayload) -> MessageSpeechAct {
    match payload {
        CommPayload::NoResponse { .. } => MessageSpeechAct::Unknown,
        CommPayload::Notice { summary, detail } => {
            let combined = match detail {
                Some(d) => format!("{summary} {d}"),
                None => summary.clone(),
            };
            match classify_text_speech_act(&combined) {
                MessageSpeechAct::Unknown => MessageSpeechAct::Notice,
                other => other,
            }
        }
        CommPayload::Question { prompt, .. } => classify_text_speech_act(prompt),
        CommPayload::Answer { text, .. } => classify_text_speech_act(text),
        CommPayload::StructuredFact {
            fact_key,
            fact_value,
            attributes,
            ..
        } => {
            if let Some(tag) = attributes.get("speech_act") {
                return match tag.trim().to_ascii_lowercase().as_str() {
                    "notice" | "aviso" => MessageSpeechAct::Notice,
                    "order" | "orden" => MessageSpeechAct::Order,
                    "preference" | "preferencia" => MessageSpeechAct::Preference,
                    "negation" | "negacion" | "negación" => MessageSpeechAct::Negation,
                    "ambiguous_referent" | "referente_ambiguo" => {
                        MessageSpeechAct::AmbiguousReferent
                    }
                    _ => MessageSpeechAct::Unknown,
                };
            }
            classify_text_speech_act(&format!("{fact_key} {fact_value}"))
        }
    }
}

fn classify_text_speech_act(text: &str) -> MessageSpeechAct {
    let lower = text.trim().to_ascii_lowercase();
    if lower.is_empty() {
        return MessageSpeechAct::Unknown;
    }

    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|tok| !tok.is_empty())
        .collect();

    if tokens.is_empty() {
        return MessageSpeechAct::Unknown;
    }

    // 1. Negación (prioridad alta para evitar falsos positivos con órdenes negadas).
    const NEGATION_TOKENS: &[&str] = &[
        "no",
        "not",
        "don't",
        "never",
        "cancel",
        "abort",
        "reject",
        "avoid",
        "forbid",
        "nunca",
        "jamás",
        "jamas",
        "tampoco",
        "cancela",
        "anula",
        "rechaza",
        "prohibido",
        "evita",
    ];
    if contains_any_token(&tokens, NEGATION_TOKENS) || lower.contains("do not") {
        return MessageSpeechAct::Negation;
    }

    // 2. Referente ambiguo (frases deícticas o anafóricas sin desambiguación).
    const AMBIGUOUS_PHRASES: &[&str] = &[
        "that one",
        "this one",
        "one of them",
        "some of them",
        "not sure which",
        "uno de ellos",
        "una de ellas",
        "cualquiera de",
        "ese de ahí",
        "ese de ahi",
        "esa de ahí",
        "esa de ahi",
    ];
    const AMBIGUOUS_TOKENS: &[&str] = &[
        "either",
        "whichever",
        "ese",
        "esa",
        "eso",
        "aquel",
        "aquella",
        "cualquiera",
    ];
    if AMBIGUOUS_PHRASES.iter().any(|p| lower.contains(p))
        || contains_any_token(&tokens, AMBIGUOUS_TOKENS)
    {
        return MessageSpeechAct::AmbiguousReferent;
    }

    // 3. Preferencia.
    const PREFERENCE_PHRASES: &[&str] = &["would like", "me gustaría", "me gustaria", "opto por"];
    const PREFERENCE_TOKENS: &[&str] = &[
        "prefer",
        "rather",
        "want",
        "favorite",
        "favourite",
        "better",
        "prefiero",
        "preferiría",
        "preferiria",
        "quiero",
        "mejor",
        "favorito",
        "favorita",
        "ojalá",
        "ojala",
    ];
    if PREFERENCE_PHRASES.iter().any(|p| lower.contains(p))
        || contains_any_token(&tokens, PREFERENCE_TOKENS)
    {
        return MessageSpeechAct::Preference;
    }

    // 4. Orden / mandato imperativo.
    const ORDER_TOKENS: &[&str] = &[
        "open",
        "close",
        "move",
        "stop",
        "execute",
        "run",
        "pick",
        "drop",
        "activate",
        "deactivate",
        "start",
        "send",
        "abre",
        "cierra",
        "mueve",
        "detén",
        "deten",
        "para",
        "ejecuta",
        "recoge",
        "suelta",
        "activa",
        "desactiva",
        "inicia",
        "envía",
        "envia",
        "haz",
    ];
    if contains_any_token(&tokens, ORDER_TOKENS) {
        return MessageSpeechAct::Order;
    }

    // 5. Aviso informativo.
    const NOTICE_TOKENS: &[&str] = &[
        "notice",
        "warning",
        "alert",
        "fyi",
        "inform",
        "report",
        "note",
        "aviso",
        "alerta",
        "atención",
        "atencion",
        "informa",
        "informo",
        "notifico",
        "recuerda",
        "locked",
        "blocked",
        "cerrada",
        "bloqueada",
    ];
    if contains_any_token(&tokens, NOTICE_TOKENS) {
        return MessageSpeechAct::Notice;
    }

    MessageSpeechAct::Unknown
}

fn contains_any_token(tokens: &[&str], targets: &[&str]) -> bool {
    tokens.iter().any(|tok| targets.contains(tok))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief::BinaryBelief;
    use crate::capability::{
        CapabilityVersion, EffectKind, ModuleManifest, ModuleOrigin, OperationSpec, PermissionSet,
    };
    use crate::clock::{MonotonicInstant, VirtualMonotonicClock};
    use crate::communication::{CommCost, ConversationScope, InterlocutorId};
    use crate::provenance::{
        FrameId, MessageId, ObservationKind, ProvenanceMetadata, SensorId, UnitTag,
    };
    use crate::semantic_memory::{
        EntityKind, EpistemicStatus, EvidenceInput, EvidencePolarity, PropositionValue,
        SemanticEntity,
    };
    use crate::system_one::{
        FixtureBehavior, RawScores, ReplaySystemOneProvider, SystemOneFixture, SystemOnePayload,
        ValidatedDistribution,
    };
    use std::time::Duration as StdDuration;

    fn sample_provenance(id: u64) -> ProvenanceMetadata {
        ProvenanceMetadata {
            message_id: MessageId::new(id),
            observed_at: MonotonicInstant::from_nanos(100),
            received_at: MonotonicInstant::from_nanos(105),
            sensor_id: SensorId::new(1),
            frame_id: FrameId::new("world").unwrap(),
            units: UnitTag::new("none").unwrap(),
            confidence: Confidence::new(0.9).unwrap(),
            revision: 1,
            evidence_ref: EvidenceRef::new(id),
            claim_kind: ObservationKind::Measurement,
        }
    }

    fn sample_message(id: u64, payload: CommPayload) -> SituatedMessage {
        SituatedMessage::new(
            MessageId::new(id),
            InterlocutorId::new("human-1").unwrap(),
            InterlocutorId::new("agent-1").unwrap(),
            ConversationScope::Direct,
            MonotonicInstant::from_nanos(1_000),
            CommCost::new(0, 0, StdDuration::ZERO),
            payload,
        )
        .unwrap()
    }

    fn sample_store_with_two_doors() -> (SemanticKnowledgeStore, EntityId, EntityId, PredicateId) {
        let mut store = SemanticKnowledgeStore::new();
        let door_north = EntityId::new(10);
        let door_south = EntityId::new(20);
        let kind = EntityKind::new("door").unwrap();
        let pred = PredicateId::new("state").unwrap();

        let e1 = SemanticEntity::new(door_north, kind.clone())
            .with_tag("north")
            .unwrap();
        let e2 = SemanticEntity::new(door_south, kind)
            .with_tag("south")
            .unwrap();
        store.upsert_entity(e1);
        store.upsert_entity(e2);

        let ev = EvidenceInput::new(
            EvidenceRef::new(1),
            MonotonicInstant::from_nanos(10),
            SensorId::new(1),
            EvidencePolarity::Supporting,
            ObservationKind::Measurement,
            Confidence::new(0.95).unwrap(),
        );
        store
            .record_proposition(
                PropositionKey::new(
                    door_north,
                    pred.clone(),
                    PropositionValue::symbol("closed").unwrap(),
                ),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.9).unwrap(),
                ev,
            )
            .unwrap();
        store
            .record_proposition(
                PropositionKey::new(
                    door_south,
                    pred.clone(),
                    PropositionValue::symbol("open").unwrap(),
                ),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.85).unwrap(),
                ev,
            )
            .unwrap();

        (store, door_north, door_south, pred)
    }

    fn sample_catalog_with_door_ops() -> (CapabilityCatalog, OperationId, OperationId) {
        let mut catalog = CapabilityCatalog::new();
        let op_open = OperationId::new("door.open");
        let op_close = OperationId::new("door.close");

        let manifest = ModuleManifest::new(
            "world.doors",
            ModuleOrigin::Builtin,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permissions(PermissionSet::empty())
        .with_operation(
            OperationSpec::new(op_open.as_str(), CapabilityVersion::new(1, 0, 0))
                .with_effect(EffectKind::StateMutation),
        )
        .with_operation(
            OperationSpec::new(op_close.as_str(), CapabilityVersion::new(1, 0, 0))
                .with_effect(EffectKind::StateMutation),
        );
        catalog.register_module(manifest).unwrap();
        (catalog, op_open, op_close)
    }

    #[test]
    fn bilingual_fixtures_distinguish_all_five_speech_acts() {
        let spanish_cases = [
            (
                CommPayload::notice("Aviso: la puerta norte está cerrada").unwrap(),
                MessageSpeechAct::Notice,
            ),
            (
                CommPayload::answer("Abre la puerta norte", None, None).unwrap(),
                MessageSpeechAct::Order,
            ),
            (
                CommPayload::answer("Prefiero la puerta sur", None, None).unwrap(),
                MessageSpeechAct::Preference,
            ),
            (
                CommPayload::answer("No abras la puerta norte", None, None).unwrap(),
                MessageSpeechAct::Negation,
            ),
            (
                CommPayload::answer("Abre ese de ahí", None, None).unwrap(),
                MessageSpeechAct::AmbiguousReferent,
            ),
        ];

        let english_cases = [
            (
                CommPayload::notice("Notice: north door is locked").unwrap(),
                MessageSpeechAct::Notice,
            ),
            (
                CommPayload::answer("Open the north door", None, None).unwrap(),
                MessageSpeechAct::Order,
            ),
            (
                CommPayload::answer("I prefer the south door", None, None).unwrap(),
                MessageSpeechAct::Preference,
            ),
            (
                CommPayload::answer("Do not open the north door", None, None).unwrap(),
                MessageSpeechAct::Negation,
            ),
            (
                CommPayload::answer("Open that one", None, None).unwrap(),
                MessageSpeechAct::AmbiguousReferent,
            ),
        ];

        for (idx, (payload, expected)) in spanish_cases.into_iter().chain(english_cases).enumerate()
        {
            let msg = sample_message((idx + 1) as u64, payload);
            assert_eq!(SemanticGrounder::classify_speech_act(&msg), expected);
        }
    }

    #[test]
    fn grounds_order_to_registered_entity_and_operation_bilingually() {
        let (store, door_north, door_south, pred) = sample_store_with_two_doors();
        let (catalog, op_open, _op_close) = sample_catalog_with_door_ops();
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000));
        let grounder = SemanticGrounder::new();

        for (msg_id, text) in [
            (1_u64, "Abre la puerta norte"),
            (2_u64, "Open the north door"),
        ] {
            let msg = sample_message(msg_id, CommPayload::answer(text, None, None).unwrap());
            let eq = QueryId::new(format!("ground-entity-{msg_id}")).unwrap();
            let oq = QueryId::new(format!("ground-op-{msg_id}")).unwrap();

            let mut provider = ReplaySystemOneProvider::new("bilingual-fixture");
            provider.register_fixture(
                eq.clone(),
                SystemOneFixture::new(
                    FixtureBehavior::Choice {
                        selected: entity_option_id(door_north).unwrap(),
                        raw_scores: vec![
                            (entity_option_id(door_north).unwrap(), 3.0),
                            (entity_option_id(door_south).unwrap(), 0.2),
                        ],
                        validated_distribution: None,
                    },
                    sample_provenance(100 + msg_id),
                ),
            );
            provider.register_fixture(
                oq.clone(),
                SystemOneFixture::new(
                    FixtureBehavior::Choice {
                        selected: operation_option_id(&op_open).unwrap(),
                        raw_scores: vec![
                            (operation_option_id(&op_open).unwrap(), 2.5),
                            (
                                operation_option_id(&OperationId::new("door.close")).unwrap(),
                                0.1,
                            ),
                        ],
                        validated_distribution: None,
                    },
                    sample_provenance(200 + msg_id),
                ),
            );

            let req = GroundingRequest::new().with_predicate(pred.clone());
            let outcome = grounder
                .ground(&msg, &store, &catalog, &provider, &clock, &req)
                .unwrap();

            match outcome {
                GroundedOutcome::Grounded {
                    speech_act,
                    entity_id,
                    proposition_key,
                    operation,
                    ..
                } => {
                    assert_eq!(speech_act, MessageSpeechAct::Order);
                    assert_eq!(entity_id, Some(door_north));
                    assert_eq!(
                        proposition_key,
                        Some(PropositionKey::new(
                            door_north,
                            pred.clone(),
                            PropositionValue::symbol("closed").unwrap()
                        ))
                    );
                    assert_eq!(operation, Some(op_open.clone()));
                }
                other => panic!("expected Grounded outcome, got {other:?}"),
            }
        }
    }

    /// Proveedor malicioso/defectuoso que ignora las opciones de la consulta y devuelve
    /// un `EntityId` u `OperationId` inventado fuera del catálogo/store.
    struct ForgedOptionProvider {
        forged_entity_opt: OptionId,
        forged_op_opt: OptionId,
    }

    impl SystemOneProvider for ForgedOptionProvider {
        fn provider_id(&self) -> &str {
            "forged-provider"
        }

        fn query(
            &self,
            query: &TypedQuery,
            _clock: &dyn MonotonicClock,
        ) -> Result<SystemOneResponse, SystemOneError> {
            let forged = if query.query_id().as_str().contains("entity") {
                self.forged_entity_opt.clone()
            } else {
                self.forged_op_opt.clone()
            };
            let dummy_query = TypedQuery::choice(
                query.query_id().clone(),
                vec![forged.clone()],
                query.deadline(),
            )?;
            let dist = ValidatedDistribution::new(vec![(forged.clone(), 1.0)])?;
            SystemOneResponse::new_validated(
                &dummy_query,
                ResponseStatus::Answered,
                SystemOnePayload::Choice {
                    selected: Some(forged.clone()),
                },
                sample_provenance(999),
                Some(dist),
                RawScores::new(vec![(forged, 1.0)])?,
            )
        }
    }

    #[test]
    fn never_creates_arbitrary_entity_or_operation_from_model_response() {
        let (store, door_north, _door_south, _) = sample_store_with_two_doors();
        let (catalog, _op_open, _op_close) = sample_catalog_with_door_ops();
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000));
        let grounder = SemanticGrounder::new();

        // Caso 1: proveedor devuelve EntityId inventado (entity:9999).
        let msg = sample_message(
            10,
            CommPayload::answer("Open the door", None, None).unwrap(),
        );
        let forged_entity_provider = ForgedOptionProvider {
            forged_entity_opt: OptionId::new("entity:9999").unwrap(),
            forged_op_opt: OptionId::new("op:door.open").unwrap(),
        };
        let outcome = grounder
            .ground(
                &msg,
                &store,
                &catalog,
                &forged_entity_provider,
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        assert_eq!(
            outcome,
            GroundedOutcome::Abstained {
                reason: AbstentionReason::UnregisteredEntity,
            }
        );

        // Caso 2: entidad válida, pero proveedor devuelve operación inventada (op:self_destruct).
        let mut mixed_provider = ReplaySystemOneProvider::new("mixed-forged-op");
        mixed_provider.register_fixture(
            QueryId::new("ground-entity-11").unwrap(),
            SystemOneFixture::new(
                FixtureBehavior::Choice {
                    selected: entity_option_id(door_north).unwrap(),
                    raw_scores: vec![
                        (entity_option_id(door_north).unwrap(), 2.0),
                        (entity_option_id(EntityId::new(20)).unwrap(), 0.1),
                    ],
                    validated_distribution: None,
                },
                sample_provenance(11),
            ),
        );

        struct EntityOkForgedOpProvider(ReplaySystemOneProvider);
        impl SystemOneProvider for EntityOkForgedOpProvider {
            fn provider_id(&self) -> &str {
                "entity-ok-forged-op"
            }
            fn query(
                &self,
                query: &TypedQuery,
                clock: &dyn MonotonicClock,
            ) -> Result<SystemOneResponse, SystemOneError> {
                if query.query_id().as_str().contains("entity") {
                    self.0.query(query, clock)
                } else {
                    let forged = OptionId::new("op:self_destruct").unwrap();
                    let dummy =
                        TypedQuery::choice(query.query_id().clone(), vec![forged.clone()], None)?;
                    let dist = ValidatedDistribution::new(vec![(forged.clone(), 1.0)])?;
                    SystemOneResponse::new_validated(
                        &dummy,
                        ResponseStatus::Answered,
                        SystemOnePayload::Choice {
                            selected: Some(forged.clone()),
                        },
                        sample_provenance(12),
                        Some(dist),
                        RawScores::new(vec![(forged, 1.0)])?,
                    )
                }
            }
        }

        let msg_op = sample_message(
            11,
            CommPayload::answer("Open the north door", None, None).unwrap(),
        );
        let outcome_op = grounder
            .ground(
                &msg_op,
                &store,
                &catalog,
                &EntityOkForgedOpProvider(mixed_provider),
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        assert_eq!(
            outcome_op,
            GroundedOutcome::Abstained {
                reason: AbstentionReason::UnregisteredOperation,
            }
        );
    }

    #[test]
    fn unknown_and_multiple_candidates_trigger_abstention_or_clarification() {
        let (store, door_north, door_south, _) = sample_store_with_two_doors();
        let (catalog, _, _) = sample_catalog_with_door_ops();
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000));

        // 1. Unknown response en Choice -> Abstained(UnknownResponse)
        let grounder = SemanticGrounder::new();
        let msg_notice = sample_message(
            20,
            CommPayload::notice("Aviso: hay una puerta abierta").unwrap(),
        );
        let mut provider_unknown = ReplaySystemOneProvider::new("unknown-provider");
        provider_unknown.register_fixture(
            QueryId::new("ground-entity-20").unwrap(),
            SystemOneFixture::new(
                FixtureBehavior::Unknown { raw_scores: vec![] },
                sample_provenance(20),
            ),
        );
        let out_unknown = grounder
            .ground(
                &msg_notice,
                &store,
                &catalog,
                &provider_unknown,
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        assert_eq!(
            out_unknown,
            GroundedOutcome::Abstained {
                reason: AbstentionReason::UnknownResponse,
            }
        );

        // 2. Referente ambiguo con múltiples candidatos y respuesta Unknown -> ClarificationNeeded
        let msg_ambig = sample_message(
            21,
            CommPayload::answer("Open that one", None, None).unwrap(),
        );
        let mut provider_ambig = ReplaySystemOneProvider::new("ambig-provider");
        provider_ambig.register_fixture(
            QueryId::new("ground-entity-21").unwrap(),
            SystemOneFixture::new(
                FixtureBehavior::Unknown { raw_scores: vec![] },
                sample_provenance(21),
            ),
        );
        let out_clarify = grounder
            .ground(
                &msg_ambig,
                &store,
                &catalog,
                &provider_ambig,
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        match out_clarify {
            GroundedOutcome::ClarificationNeeded {
                candidates,
                question,
            } => {
                assert_eq!(candidates, vec![door_north, door_south]);
                assert!(matches!(question, CommPayload::Question { .. }));
            }
            other => panic!("expected ClarificationNeeded, got {other:?}"),
        }

        // 3. Empate en Score con clarify_on_ambiguity = false -> Abstained(AmbiguousCandidates)
        let score_grounder = SemanticGrounder::with_config(GroundingConfig {
            query_mode: GroundingQueryMode::Score,
            clarify_on_ambiguity: false,
            ambiguity_margin: 0.05,
            operation_policy: OperationResolutionPolicy::Never,
        })
        .unwrap();
        let msg_pref = sample_message(
            22,
            CommPayload::answer("Prefiero esa puerta", None, None).unwrap(),
        );
        let mut provider_tie = ReplaySystemOneProvider::new("tie-provider");
        provider_tie.register_fixture(
            QueryId::new("ground-entity-22").unwrap(),
            SystemOneFixture::new(
                FixtureBehavior::Score {
                    raw_scores: vec![
                        (entity_option_id(door_north).unwrap(), 1.0),
                        (entity_option_id(door_south).unwrap(), 1.0),
                    ],
                    validated_distribution: None,
                },
                sample_provenance(22),
            ),
        );
        let out_tie = score_grounder
            .ground(
                &msg_pref,
                &store,
                &catalog,
                &provider_tie,
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        assert_eq!(
            out_tie,
            GroundedOutcome::Abstained {
                reason: AbstentionReason::AmbiguousCandidates(vec![door_north, door_south]),
            }
        );

        // 4. TimedOut -> Abstained(ProviderTimedOut)
        let msg_timeout = sample_message(
            23,
            CommPayload::answer("I prefer the north door", None, None).unwrap(),
        );
        let mut provider_timeout = ReplaySystemOneProvider::new("timeout-provider");
        provider_timeout.register_fixture(
            QueryId::new("ground-entity-23").unwrap(),
            SystemOneFixture::new(FixtureBehavior::TimedOut, sample_provenance(23))
                .with_simulated_latency(StdDuration::from_millis(5)),
        );
        let out_timeout = grounder
            .ground(
                &msg_timeout,
                &store,
                &catalog,
                &provider_timeout,
                &clock,
                &GroundingRequest::new(),
            )
            .unwrap();
        assert_eq!(
            out_timeout,
            GroundedOutcome::Abstained {
                reason: AbstentionReason::ProviderTimedOut,
            }
        );
    }
}
