//! Comunicación situada reutilizable, acotada, determinista y no bloqueante (`std`-only).
//!
//! Este módulo define el puerto de comunicación situada entre interlocutores humanos,
//! agentes y canales simbólicos multiagente sin depender de red, LLM, ni otorgar
//! autoridad o permisos de actuación al controlador:
//! - [`InterlocutorId`], [`InterlocutorKind`] (`Human` | `Agent` | `SymbolChannel`),
//!   [`ConversationScope`] y [`CommCost`] (unidades, tokens y tiempo explícito).
//! - [`CommPayload`]: carga útil tipada y validada (`Notice`, `Question`, `Answer`,
//!   `NoResponse`, `StructuredFact`), rechazando campos vacíos o inválidos.
//! - [`SituatedMessage`]: mensaje inmutable con correlación pregunta-respuesta
//!   (`correlation_id`), reloj monotónico (`sent_at`, `deadline`) y coste explícito.
//! - [`CommunicationPort`] y [`LocalCommunicationBus`]: puerto desacoplado e
//!   implementación local en memoria, acotada, determinista y **no bloqueante**, con
//!   correlación exacta `request_response` / `poll_reply` y manejo explícito de
//!   `TimedOut` / `NoResponse` mediante [`MonotonicClock`].
//! - [`OfflineTemplate`] y [`TemplateRegistry`]: plantillas offline reutilizables
//!   para interlocutores humanos y canales simbólicos multiagente con validación de
//!   parámetros y payload resultante.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::time::Duration as StdDuration;

use crate::clock::{MonotonicClock, MonotonicDeadline, MonotonicInstant};
use crate::provenance::{Confidence, EvidenceRef, MessageId};
use crate::system_one::OptionId;

/// Errores explícitos de validación, transporte y plantillas de comunicación situada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommunicationError {
    /// Un identificador requerido está vacío o solo contiene espacios.
    EmptyIdentifier {
        /// Nombre del campo inválido.
        field: &'static str,
    },
    /// Un campo textual obligatorio del payload está vacío o solo contiene espacios.
    EmptyPayloadField {
        /// Nombre del campo inválido.
        field: &'static str,
    },
    /// Una consulta `Question` contiene opciones esperadas duplicadas.
    DuplicateOptionId {
        /// Identificador de opción repetido.
        option_id: OptionId,
    },
    /// El emisor y el receptor del mensaje son el mismo interlocutor.
    SelfAddressing {
        /// Interlocutor duplicado.
        interlocutor: InterlocutorId,
    },
    /// Un mensaje intenta correlacionarse consigo mismo.
    SelfCorrelation {
        /// Identificador del mensaje.
        message_id: MessageId,
    },
    /// Se intentó enviar un mensaje con un [`MessageId`] ya registrado en el bus.
    DuplicateMessageId {
        /// Identificador duplicado.
        message_id: MessageId,
    },
    /// `request_response` exige un payload de tipo [`CommPayloadKind::Question`].
    ExpectedQuestionPayload {
        /// Tipo real recibido.
        actual_kind: CommPayloadKind,
    },
    /// No existe una solicitud pendiente asociada al `correlation_id` consultado.
    UnknownCorrelationId {
        /// Identificador de correlación desconocido.
        correlation_id: MessageId,
    },
    /// La respuesta correlacionada no coincide con el emisor/receptor de la solicitud.
    CorrelationMismatch {
        /// Identificador de correlación afectado.
        correlation_id: MessageId,
        /// Motivo determinista del desacuerdo.
        reason: &'static str,
    },
    /// La capacidad configurada para colas o solicitudes pendientes debe ser >= 1.
    InvalidCapacity {
        /// Campo de configuración inválido.
        field: &'static str,
        /// Valor recibido.
        value: usize,
    },
    /// La bandeja de entrada del receptor alcanzó su capacidad máxima acotada.
    ChannelFull {
        /// Receptor saturado.
        recipient: InterlocutorId,
        /// Capacidad máxima configurada.
        capacity: usize,
    },
    /// El registro de solicitudes pendientes alcanzó su límite máximo acotado.
    PendingLimitReached {
        /// Capacidad máxima de solicitudes pendientes.
        capacity: usize,
    },
    /// Se intentó registrar una plantilla con un [`TemplateId`] ya existente.
    DuplicateTemplateId {
        /// Identificador duplicado.
        template_id: TemplateId,
    },
    /// No existe una plantilla registrada con el [`TemplateId`] solicitado.
    UnknownTemplateId {
        /// Identificador desconocido.
        template_id: TemplateId,
    },
    /// Falta un parámetro requerido al renderizar una [`OfflineTemplate`].
    MissingTemplateParam {
        /// Plantilla afectada.
        template_id: TemplateId,
        /// Nombre del parámetro faltante.
        param: String,
    },
    /// Un parámetro provisto para renderizar una plantilla está vacío o es inválido.
    InvalidTemplateParam {
        /// Plantilla afectada.
        template_id: TemplateId,
        /// Nombre del parámetro inválido.
        param: String,
    },
    /// La plantilla no corresponde al [`InterlocutorKind`] esperado.
    TemplateKindMismatch {
        /// Plantilla solicitada.
        template_id: TemplateId,
        /// Tipo esperado por el invocador.
        expected: InterlocutorKind,
        /// Tipo declarado por la plantilla.
        actual: InterlocutorKind,
    },
}

impl fmt::Display for CommunicationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { field } => {
                write!(f, "communication identifier '{field}' cannot be empty")
            }
            Self::EmptyPayloadField { field } => {
                write!(f, "communication payload field '{field}' cannot be empty")
            }
            Self::DuplicateOptionId { option_id } => {
                write!(f, "duplicate expected option_id '{option_id}' in question")
            }
            Self::SelfAddressing { interlocutor } => {
                write!(
                    f,
                    "sender and recipient cannot be the same interlocutor '{interlocutor}'"
                )
            }
            Self::SelfCorrelation { message_id } => {
                write!(f, "message {message_id} cannot correlate with itself")
            }
            Self::DuplicateMessageId { message_id } => {
                write!(f, "duplicate message_id {message_id} on communication bus")
            }
            Self::ExpectedQuestionPayload { actual_kind } => {
                write!(
                    f,
                    "request_response requires Question payload, got {actual_kind}"
                )
            }
            Self::UnknownCorrelationId { correlation_id } => {
                write!(f, "unknown correlation_id {correlation_id}")
            }
            Self::CorrelationMismatch {
                correlation_id,
                reason,
            } => {
                write!(f, "correlation mismatch for {correlation_id}: {reason}")
            }
            Self::InvalidCapacity { field, value } => {
                write!(f, "invalid capacity '{field}' ({value}): must be >= 1")
            }
            Self::ChannelFull {
                recipient,
                capacity,
            } => {
                write!(
                    f,
                    "inbox for recipient '{recipient}' is full (capacity={capacity})"
                )
            }
            Self::PendingLimitReached { capacity } => {
                write!(
                    f,
                    "pending correlated requests limit reached (capacity={capacity})"
                )
            }
            Self::DuplicateTemplateId { template_id } => {
                write!(f, "duplicate offline template_id '{template_id}'")
            }
            Self::UnknownTemplateId { template_id } => {
                write!(f, "unknown offline template_id '{template_id}'")
            }
            Self::MissingTemplateParam { template_id, param } => {
                write!(
                    f,
                    "missing parameter '{param}' for offline template '{template_id}'"
                )
            }
            Self::InvalidTemplateParam { template_id, param } => {
                write!(
                    f,
                    "invalid empty parameter '{param}' for offline template '{template_id}'"
                )
            }
            Self::TemplateKindMismatch {
                template_id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "template '{template_id}' targets {actual}, expected {expected}"
                )
            }
        }
    }
}

impl Error for CommunicationError {}

fn validate_non_empty_id(
    field: &'static str,
    value: impl Into<String>,
) -> Result<String, CommunicationError> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(CommunicationError::EmptyIdentifier { field });
    }
    Ok(value)
}

fn validate_non_empty_payload(
    field: &'static str,
    value: impl Into<String>,
) -> Result<String, CommunicationError> {
    let value = value.into();
    if value.trim().is_empty() {
        return Err(CommunicationError::EmptyPayloadField { field });
    }
    Ok(value)
}

/// Identificador validado y no vacío de un interlocutor en comunicación situada.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterlocutorId(String);

impl InterlocutorId {
    /// Crea un [`InterlocutorId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, CommunicationError> {
        Ok(Self(validate_non_empty_id("interlocutor_id", id)?))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for InterlocutorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Naturaleza explícita del interlocutor o canal comunicativo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InterlocutorKind {
    /// Operador o usuario humano.
    Human,
    /// Otro agente autónomo o módulo cognitivo.
    Agent,
    /// Canal simbólico compartido multiagente.
    SymbolChannel,
}

impl fmt::Display for InterlocutorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Human => f.write_str("Human"),
            Self::Agent => f.write_str("Agent"),
            Self::SymbolChannel => f.write_str("SymbolChannel"),
        }
    }
}

/// Ámbito explícito de una conversación situada.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversationScope {
    /// Intercambio directo entre dos interlocutores.
    Direct,
    /// Conversación acotada a una sesión o episodio identificado.
    Session(String),
    /// Canal temático o de símbolos compartidos multiagente.
    Channel(String),
}

impl ConversationScope {
    /// Construye un ámbito directo (`Direct`).
    pub const fn direct() -> Self {
        Self::Direct
    }

    /// Construye un ámbito de sesión validando que el identificador no sea vacío.
    pub fn session(id: impl Into<String>) -> Result<Self, CommunicationError> {
        Ok(Self::Session(validate_non_empty_id("scope.session", id)?))
    }

    /// Construye un ámbito de canal validando que el nombre no sea vacío.
    pub fn channel(name: impl Into<String>) -> Result<Self, CommunicationError> {
        Ok(Self::Channel(validate_non_empty_id("scope.channel", name)?))
    }
}

impl fmt::Display for ConversationScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => f.write_str("direct"),
            Self::Session(id) => write!(f, "session:{id}"),
            Self::Channel(name) => write!(f, "channel:{name}"),
        }
    }
}

/// Coste explícito de un acto comunicativo en unidades, tokens y tiempo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct CommCost {
    units: u64,
    tokens: u64,
    duration: StdDuration,
}

impl CommCost {
    /// Coste nulo (`0` unidades, `0` tokens, `0 ns`).
    pub const ZERO: Self = Self {
        units: 0,
        tokens: 0,
        duration: StdDuration::ZERO,
    };

    /// Construye un [`CommCost`] explícito.
    pub const fn new(units: u64, tokens: u64, duration: StdDuration) -> Self {
        Self {
            units,
            tokens,
            duration,
        }
    }

    /// Construye un [`CommCost`] a partir de unidades discretas de comunicación.
    pub const fn from_units(units: u64) -> Self {
        Self::new(units, 0, StdDuration::ZERO)
    }

    /// Construye un [`CommCost`] a partir de tokens estimados o consumidos.
    pub const fn from_tokens(tokens: u64) -> Self {
        Self::new(0, tokens, StdDuration::ZERO)
    }

    /// Construye un [`CommCost`] a partir de una duración temporal explícita.
    pub const fn from_duration(duration: StdDuration) -> Self {
        Self::new(0, 0, duration)
    }

    /// Unidades de comunicación.
    pub const fn units(self) -> u64 {
        self.units
    }

    /// Tokens estimados o consumidos.
    pub const fn tokens(self) -> u64 {
        self.tokens
    }

    /// Tiempo explícito asociado al mensaje.
    pub const fn duration(self) -> StdDuration {
        self.duration
    }

    /// Suma verificada de dos costes comunicativos.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            units: self.units.checked_add(other.units)?,
            tokens: self.tokens.checked_add(other.tokens)?,
            duration: self.duration.checked_add(other.duration)?,
        })
    }

    /// Suma saturante de dos costes comunicativos.
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            units: self.units.saturating_add(other.units),
            tokens: self.tokens.saturating_add(other.tokens),
            duration: self.duration.saturating_add(other.duration),
        }
    }
}

impl fmt::Display for CommCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "cost(units={}, tokens={}, ms={})",
            self.units,
            self.tokens,
            self.duration.as_millis()
        )
    }
}

/// Discriminante de tipo de [`CommPayload`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CommPayloadKind {
    /// Aviso informativo sin respuesta obligatoria.
    Notice,
    /// Pregunta que espera respuesta correlacionada.
    Question,
    /// Respuesta a una pregunta previa.
    Answer,
    /// Ausencia explícita de respuesta.
    NoResponse,
    /// Hecho estructurado con atributos clave-valor y procedencia opcional.
    StructuredFact,
}

impl fmt::Display for CommPayloadKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Notice => f.write_str("Notice"),
            Self::Question => f.write_str("Question"),
            Self::Answer => f.write_str("Answer"),
            Self::NoResponse => f.write_str("NoResponse"),
            Self::StructuredFact => f.write_str("StructuredFact"),
        }
    }
}

/// Motivo explícito de ausencia de respuesta en [`CommPayload::NoResponse`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NoResponseReason {
    /// El plazo de respuesta expiró antes de recibir contestación.
    TimedOut,
    /// El interlocutor no está disponible en este ciclo.
    Unavailable,
    /// El interlocutor declinó responder explícitamente.
    Declined,
    /// Se agotó el presupuesto de comunicación asignado.
    BudgetExhausted,
}

impl fmt::Display for NoResponseReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TimedOut => f.write_str("TimedOut"),
            Self::Unavailable => f.write_str("Unavailable"),
            Self::Declined => f.write_str("Declined"),
            Self::BudgetExhausted => f.write_str("BudgetExhausted"),
        }
    }
}

/// Payload tipado y validado para comunicación situada sin red ni LLM.
#[derive(Debug, Clone, PartialEq)]
pub enum CommPayload {
    /// Aviso informativo.
    Notice {
        /// Resumen no vacío del aviso.
        summary: String,
        /// Detalle adicional opcional (si existe, no puede ser vacío).
        detail: Option<String>,
    },
    /// Pregunta tipada hacia un interlocutor o canal.
    Question {
        /// Enunciado no vacío de la pregunta.
        prompt: String,
        /// Opciones cerradas esperadas (opcional; si se proveen, sin duplicados).
        expected_options: Vec<OptionId>,
    },
    /// Respuesta explícita a una pregunta.
    Answer {
        /// Contenido textual no vacío de la respuesta.
        text: String,
        /// Opción seleccionada si la pregunta era cerrada.
        selected_option: Option<OptionId>,
        /// Confianza declarada opcional en `[0.0, 1.0]` (no implica autoridad).
        confidence: Option<Confidence>,
    },
    /// Ausencia explícita de respuesta.
    NoResponse {
        /// Motivo determinista de la no respuesta.
        reason: NoResponseReason,
    },
    /// Hecho estructurado intercambiable entre humano y agentes.
    StructuredFact {
        /// Clave del hecho (no vacía).
        fact_key: String,
        /// Valor del hecho (no vacío).
        fact_value: String,
        /// Atributos clave-valor adicionales (ninguna clave o valor puede estar vacío).
        attributes: BTreeMap<String, String>,
        /// Confianza epistémica opcional (no implica permiso ni verdad absoluta).
        confidence: Option<Confidence>,
        /// Referencia opcional de evidencia.
        evidence: Option<EvidenceRef>,
    },
}

impl CommPayload {
    /// Construye un [`CommPayload::Notice`] validado.
    pub fn notice(summary: impl Into<String>) -> Result<Self, CommunicationError> {
        let payload = Self::Notice {
            summary: validate_non_empty_payload("notice.summary", summary)?,
            detail: None,
        };
        payload.validate()?;
        Ok(payload)
    }

    /// Construye un [`CommPayload::Notice`] con detalle validado.
    pub fn notice_with_detail(
        summary: impl Into<String>,
        detail: impl Into<String>,
    ) -> Result<Self, CommunicationError> {
        let payload = Self::Notice {
            summary: validate_non_empty_payload("notice.summary", summary)?,
            detail: Some(validate_non_empty_payload("notice.detail", detail)?),
        };
        payload.validate()?;
        Ok(payload)
    }

    /// Construye un [`CommPayload::Question`] validado.
    pub fn question(
        prompt: impl Into<String>,
        expected_options: Vec<OptionId>,
    ) -> Result<Self, CommunicationError> {
        let payload = Self::Question {
            prompt: validate_non_empty_payload("question.prompt", prompt)?,
            expected_options,
        };
        payload.validate()?;
        Ok(payload)
    }

    /// Construye un [`CommPayload::Answer`] validado.
    pub fn answer(
        text: impl Into<String>,
        selected_option: Option<OptionId>,
        confidence: Option<Confidence>,
    ) -> Result<Self, CommunicationError> {
        let payload = Self::Answer {
            text: validate_non_empty_payload("answer.text", text)?,
            selected_option,
            confidence,
        };
        payload.validate()?;
        Ok(payload)
    }

    /// Construye un [`CommPayload::NoResponse`].
    pub const fn no_response(reason: NoResponseReason) -> Self {
        Self::NoResponse { reason }
    }

    /// Construye un [`CommPayload::StructuredFact`] validado.
    pub fn structured_fact(
        fact_key: impl Into<String>,
        fact_value: impl Into<String>,
        attributes: BTreeMap<String, String>,
        confidence: Option<Confidence>,
        evidence: Option<EvidenceRef>,
    ) -> Result<Self, CommunicationError> {
        let payload = Self::StructuredFact {
            fact_key: validate_non_empty_payload("structured_fact.fact_key", fact_key)?,
            fact_value: validate_non_empty_payload("structured_fact.fact_value", fact_value)?,
            attributes,
            confidence,
            evidence,
        };
        payload.validate()?;
        Ok(payload)
    }

    /// Devuelve el tipo discriminante del payload.
    pub const fn kind(&self) -> CommPayloadKind {
        match self {
            Self::Notice { .. } => CommPayloadKind::Notice,
            Self::Question { .. } => CommPayloadKind::Question,
            Self::Answer { .. } => CommPayloadKind::Answer,
            Self::NoResponse { .. } => CommPayloadKind::NoResponse,
            Self::StructuredFact { .. } => CommPayloadKind::StructuredFact,
        }
    }

    /// Valida todas las invariantes estructurales del payload.
    pub fn validate(&self) -> Result<(), CommunicationError> {
        match self {
            Self::Notice { summary, detail } => {
                if summary.trim().is_empty() {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "notice.summary",
                    });
                }
                if let Some(d) = detail
                    && d.trim().is_empty()
                {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "notice.detail",
                    });
                }
            }
            Self::Question {
                prompt,
                expected_options,
            } => {
                if prompt.trim().is_empty() {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "question.prompt",
                    });
                }
                let mut seen = BTreeSet::new();
                for opt in expected_options {
                    if !seen.insert(opt.clone()) {
                        return Err(CommunicationError::DuplicateOptionId {
                            option_id: opt.clone(),
                        });
                    }
                }
            }
            Self::Answer { text, .. } => {
                if text.trim().is_empty() {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "answer.text",
                    });
                }
            }
            Self::NoResponse { .. } => {}
            Self::StructuredFact {
                fact_key,
                fact_value,
                attributes,
                ..
            } => {
                if fact_key.trim().is_empty() {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "structured_fact.fact_key",
                    });
                }
                if fact_value.trim().is_empty() {
                    return Err(CommunicationError::EmptyPayloadField {
                        field: "structured_fact.fact_value",
                    });
                }
                for (k, v) in attributes {
                    if k.trim().is_empty() {
                        return Err(CommunicationError::EmptyPayloadField {
                            field: "structured_fact.attribute_key",
                        });
                    }
                    if v.trim().is_empty() {
                        return Err(CommunicationError::EmptyPayloadField {
                            field: "structured_fact.attribute_value",
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Mensaje inmutable de comunicación situada.
///
/// Nunca transporta acciones autorizadas, permisos ni capacidades de mutación del entorno.
#[derive(Debug, Clone, PartialEq)]
pub struct SituatedMessage {
    message_id: MessageId,
    correlation_id: Option<MessageId>,
    sender: InterlocutorId,
    recipient: InterlocutorId,
    scope: ConversationScope,
    sent_at: MonotonicInstant,
    deadline: Option<MonotonicDeadline>,
    cost: CommCost,
    payload: CommPayload,
}

impl SituatedMessage {
    /// Construye un [`SituatedMessage`] validando direccionamiento y payload.
    pub fn new(
        message_id: MessageId,
        sender: InterlocutorId,
        recipient: InterlocutorId,
        scope: ConversationScope,
        sent_at: MonotonicInstant,
        cost: CommCost,
        payload: CommPayload,
    ) -> Result<Self, CommunicationError> {
        if sender == recipient {
            return Err(CommunicationError::SelfAddressing {
                interlocutor: sender,
            });
        }
        payload.validate()?;
        Ok(Self {
            message_id,
            correlation_id: None,
            sender,
            recipient,
            scope,
            sent_at,
            deadline: None,
            cost,
            payload,
        })
    }

    /// Asocia un `correlation_id` (por ejemplo el [`MessageId`] de una pregunta previa).
    pub fn with_correlation_id(
        mut self,
        correlation_id: MessageId,
    ) -> Result<Self, CommunicationError> {
        if correlation_id == self.message_id {
            return Err(CommunicationError::SelfCorrelation {
                message_id: self.message_id,
            });
        }
        self.correlation_id = Some(correlation_id);
        Ok(self)
    }

    /// Asocia un plazo monotónico explícito al mensaje.
    pub const fn with_deadline(mut self, deadline: MonotonicDeadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Identificador único del mensaje.
    pub const fn message_id(&self) -> MessageId {
        self.message_id
    }

    /// Identificador de correlación pregunta-respuesta si aplica.
    pub const fn correlation_id(&self) -> Option<MessageId> {
        self.correlation_id
    }

    /// Emisor del mensaje.
    pub fn sender(&self) -> &InterlocutorId {
        &self.sender
    }

    /// Receptor del mensaje.
    pub fn recipient(&self) -> &InterlocutorId {
        &self.recipient
    }

    /// Ámbito de la conversación.
    pub fn scope(&self) -> &ConversationScope {
        &self.scope
    }

    /// Instante monotónico de emisión.
    pub const fn sent_at(&self) -> MonotonicInstant {
        self.sent_at
    }

    /// Plazo monotónico opcional del mensaje.
    pub const fn deadline(&self) -> Option<MonotonicDeadline> {
        self.deadline
    }

    /// Coste explícito declarado del mensaje.
    pub const fn cost(&self) -> CommCost {
        self.cost
    }

    /// Payload tipado del mensaje.
    pub fn payload(&self) -> &CommPayload {
        &self.payload
    }
}

/// Resultado no bloqueante de consultar una respuesta correlacionada.
#[derive(Debug, Clone, PartialEq)]
pub enum ReplyOutcome {
    /// Aún no ha llegado respuesta y el plazo no ha expirado.
    Pending,
    /// Llegó una respuesta correlacionada válida.
    Answered(SituatedMessage),
    /// El interlocutor respondió explícitamente con [`CommPayload::NoResponse`].
    NoResponse(SituatedMessage),
    /// El plazo monotónico de la pregunta expiró sin recibir respuesta.
    TimedOut {
        /// Identificador de la pregunta expirada.
        request_id: MessageId,
        /// Plazo que fue alcanzado o superado.
        deadline: MonotonicDeadline,
        /// Instante monotónico en que se observó la expiración.
        observed_at: MonotonicInstant,
    },
}

/// Puerto desacoplado y no bloqueante de transporte para comunicación situada.
///
/// Ninguna implementación de este puerto otorga autoridad de actuación ni bloquea
/// al controlador al enviar o sondear mensajes.
pub trait CommunicationPort {
    /// Envía un mensaje validado de forma no bloqueante.
    fn send(&mut self, message: SituatedMessage) -> Result<MessageId, CommunicationError>;

    /// Extrae hasta `max_messages` mensajes de la bandeja de `recipient` en orden FIFO.
    fn poll_inbox(
        &mut self,
        recipient: &InterlocutorId,
        max_messages: usize,
    ) -> Vec<SituatedMessage>;

    /// Envía una pregunta correlacionada y registra su seguimiento sin bloquear.
    fn request_response(
        &mut self,
        request: SituatedMessage,
        clock: &dyn MonotonicClock,
    ) -> Result<MessageId, CommunicationError>;

    /// Consulta de forma no bloqueante el estado o respuesta de una pregunta correlacionada.
    fn poll_reply(
        &mut self,
        correlation_id: MessageId,
        clock: &dyn MonotonicClock,
    ) -> Result<ReplyOutcome, CommunicationError>;

    /// Número de solicitudes pregunta-respuesta actualmente pendientes.
    fn pending_count(&self) -> usize;

    /// Número de mensajes actualmente en espera en la bandeja de `recipient`.
    fn inbox_len(&self, recipient: &InterlocutorId) -> usize;
}

#[derive(Debug, Clone)]
struct PendingRequest {
    sender: InterlocutorId,
    recipient: InterlocutorId,
    deadline: Option<MonotonicDeadline>,
    reply: Option<SituatedMessage>,
}

/// Implementación local en memoria, acotada, determinista y no bloqueante de [`CommunicationPort`].
///
/// Sirve simultáneamente a interlocutores humanos, agentes y canales simbólicos multiagente
/// sin hilos, red ni bloqueo del controlador.
#[derive(Debug, Clone)]
pub struct LocalCommunicationBus {
    max_inbox_per_recipient: usize,
    max_pending_requests: usize,
    interlocutor_kinds: BTreeMap<InterlocutorId, InterlocutorKind>,
    inboxes: BTreeMap<InterlocutorId, VecDeque<SituatedMessage>>,
    pending_requests: BTreeMap<MessageId, PendingRequest>,
    seen_message_ids: HashSet<MessageId>,
    total_cost: CommCost,
}

impl LocalCommunicationBus {
    /// Crea un bus local acotado con capacidades explícitas.
    pub fn new(
        max_inbox_per_recipient: usize,
        max_pending_requests: usize,
    ) -> Result<Self, CommunicationError> {
        if max_inbox_per_recipient == 0 {
            return Err(CommunicationError::InvalidCapacity {
                field: "max_inbox_per_recipient",
                value: max_inbox_per_recipient,
            });
        }
        if max_pending_requests == 0 {
            return Err(CommunicationError::InvalidCapacity {
                field: "max_pending_requests",
                value: max_pending_requests,
            });
        }
        Ok(Self {
            max_inbox_per_recipient,
            max_pending_requests,
            interlocutor_kinds: BTreeMap::new(),
            inboxes: BTreeMap::new(),
            pending_requests: BTreeMap::new(),
            seen_message_ids: HashSet::new(),
            total_cost: CommCost::ZERO,
        })
    }

    /// Registra o actualiza el [`InterlocutorKind`] de un interlocutor en el bus.
    pub fn register_interlocutor(&mut self, interlocutor: InterlocutorId, kind: InterlocutorKind) {
        self.interlocutor_kinds.insert(interlocutor, kind);
    }

    /// Devuelve el [`InterlocutorKind`] registrado para un interlocutor, si existe.
    pub fn interlocutor_kind(&self, interlocutor: &InterlocutorId) -> Option<InterlocutorKind> {
        self.interlocutor_kinds.get(interlocutor).copied()
    }

    /// Coste comunicativo total acumulado en este bus.
    pub const fn total_cost(&self) -> CommCost {
        self.total_cost
    }

    /// Capacidad máxima configurada por bandeja de receptor.
    pub const fn max_inbox_per_recipient(&self) -> usize {
        self.max_inbox_per_recipient
    }

    /// Capacidad máxima configurada de solicitudes correlacionadas pendientes.
    pub const fn max_pending_requests(&self) -> usize {
        self.max_pending_requests
    }
}

impl CommunicationPort for LocalCommunicationBus {
    fn send(&mut self, message: SituatedMessage) -> Result<MessageId, CommunicationError> {
        message.payload().validate()?;
        let msg_id = message.message_id();
        if self.seen_message_ids.contains(&msg_id) {
            return Err(CommunicationError::DuplicateMessageId { message_id: msg_id });
        }

        if let Some(corr_id) = message.correlation_id()
            && let Some(pending) = self.pending_requests.get_mut(&corr_id)
        {
            if message.sender() != &pending.recipient || message.recipient() != &pending.sender {
                return Err(CommunicationError::CorrelationMismatch {
                    correlation_id: corr_id,
                    reason: "reply sender/recipient do not match original request",
                });
            }
            if pending.reply.is_some() {
                return Err(CommunicationError::CorrelationMismatch {
                    correlation_id: corr_id,
                    reason: "correlated request already received a reply",
                });
            }
            self.seen_message_ids.insert(msg_id);
            self.total_cost = self.total_cost.saturating_add(message.cost());
            pending.reply = Some(message);
            return Ok(msg_id);
        }

        let current_len = self
            .inboxes
            .get(message.recipient())
            .map_or(0, VecDeque::len);
        if current_len >= self.max_inbox_per_recipient {
            return Err(CommunicationError::ChannelFull {
                recipient: message.recipient().clone(),
                capacity: self.max_inbox_per_recipient,
            });
        }

        self.seen_message_ids.insert(msg_id);
        self.total_cost = self.total_cost.saturating_add(message.cost());
        self.inboxes
            .entry(message.recipient().clone())
            .or_default()
            .push_back(message);
        Ok(msg_id)
    }

    fn poll_inbox(
        &mut self,
        recipient: &InterlocutorId,
        max_messages: usize,
    ) -> Vec<SituatedMessage> {
        if max_messages == 0 {
            return Vec::new();
        }
        let Some(queue) = self.inboxes.get_mut(recipient) else {
            return Vec::new();
        };
        let count = max_messages.min(queue.len());
        let mut drained = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(msg) = queue.pop_front() {
                drained.push(msg);
            }
        }
        drained
    }

    fn request_response(
        &mut self,
        request: SituatedMessage,
        _clock: &dyn MonotonicClock,
    ) -> Result<MessageId, CommunicationError> {
        if request.payload().kind() != CommPayloadKind::Question {
            return Err(CommunicationError::ExpectedQuestionPayload {
                actual_kind: request.payload().kind(),
            });
        }
        if self.pending_requests.len() >= self.max_pending_requests {
            return Err(CommunicationError::PendingLimitReached {
                capacity: self.max_pending_requests,
            });
        }

        let req_id = request.message_id();
        let pending = PendingRequest {
            sender: request.sender().clone(),
            recipient: request.recipient().clone(),
            deadline: request.deadline(),
            reply: None,
        };
        self.send(request)?;
        self.pending_requests.insert(req_id, pending);
        Ok(req_id)
    }

    fn poll_reply(
        &mut self,
        correlation_id: MessageId,
        clock: &dyn MonotonicClock,
    ) -> Result<ReplyOutcome, CommunicationError> {
        let Some(pending) = self.pending_requests.get(&correlation_id) else {
            return Err(CommunicationError::UnknownCorrelationId { correlation_id });
        };

        if pending.reply.is_some() {
            let resolved = self
                .pending_requests
                .remove(&correlation_id)
                .and_then(|p| p.reply)
                .expect("reply was just verified as Some");
            return match resolved.payload() {
                CommPayload::NoResponse { .. } => Ok(ReplyOutcome::NoResponse(resolved)),
                _ => Ok(ReplyOutcome::Answered(resolved)),
            };
        }

        if let Some(deadline) = pending.deadline {
            let now = clock.now();
            if deadline.is_expired_at(now) {
                self.pending_requests.remove(&correlation_id);
                return Ok(ReplyOutcome::TimedOut {
                    request_id: correlation_id,
                    deadline,
                    observed_at: now,
                });
            }
        }

        Ok(ReplyOutcome::Pending)
    }

    fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    fn inbox_len(&self, recipient: &InterlocutorId) -> usize {
        self.inboxes.get(recipient).map_or(0, VecDeque::len)
    }
}

/// Identificador validado y no vacío de una plantilla offline de comunicación.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemplateId(String);

impl TemplateId {
    /// Crea un [`TemplateId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, CommunicationError> {
        Ok(Self(validate_non_empty_id("template_id", id)?))
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for TemplateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Especificación estructural de una plantilla offline para construir [`CommPayload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfflineTemplateKind {
    /// Plantilla de aviso (`Notice`).
    Notice {
        /// Plantilla de resumen con marcadores `{param}`.
        summary_template: String,
        /// Plantilla opcional de detalle con marcadores `{param}`.
        detail_template: Option<String>,
    },
    /// Plantilla de pregunta (`Question`).
    Question {
        /// Plantilla de enunciado con marcadores `{param}`.
        prompt_template: String,
        /// Opciones cerradas esperadas.
        expected_options: Vec<OptionId>,
    },
    /// Plantilla de hecho estructurado (`StructuredFact`).
    StructuredFact {
        /// Clave fija del hecho.
        fact_key: String,
        /// Plantilla del valor del hecho con marcadores `{param}`.
        fact_value_template: String,
        /// Plantillas de atributos clave-valor adicionales.
        attribute_templates: BTreeMap<String, String>,
    },
}

/// Plantilla offline reutilizable para interlocutores humanos o canales simbólicos multiagente.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OfflineTemplate {
    id: TemplateId,
    target_kind: InterlocutorKind,
    default_cost: CommCost,
    kind: OfflineTemplateKind,
}

impl OfflineTemplate {
    /// Crea una nueva plantilla offline validando que su contenido base no sea vacío.
    pub fn new(
        id: TemplateId,
        target_kind: InterlocutorKind,
        default_cost: CommCost,
        kind: OfflineTemplateKind,
    ) -> Result<Self, CommunicationError> {
        match &kind {
            OfflineTemplateKind::Notice {
                summary_template,
                detail_template,
            } => {
                validate_non_empty_payload("template.summary", summary_template)?;
                if let Some(detail) = detail_template {
                    validate_non_empty_payload("template.detail", detail)?;
                }
            }
            OfflineTemplateKind::Question {
                prompt_template,
                expected_options,
            } => {
                validate_non_empty_payload("template.prompt", prompt_template)?;
                let mut seen = BTreeSet::new();
                for opt in expected_options {
                    if !seen.insert(opt.clone()) {
                        return Err(CommunicationError::DuplicateOptionId {
                            option_id: opt.clone(),
                        });
                    }
                }
            }
            OfflineTemplateKind::StructuredFact {
                fact_key,
                fact_value_template,
                attribute_templates,
            } => {
                validate_non_empty_payload("template.fact_key", fact_key)?;
                validate_non_empty_payload("template.fact_value", fact_value_template)?;
                for (k, v) in attribute_templates {
                    validate_non_empty_payload("template.attribute_key", k)?;
                    validate_non_empty_payload("template.attribute_value", v)?;
                }
            }
        }
        Ok(Self {
            id,
            target_kind,
            default_cost,
            kind,
        })
    }

    /// Identificador de la plantilla.
    pub fn id(&self) -> &TemplateId {
        &self.id
    }

    /// Tipo de interlocutor al que está destinada la plantilla.
    pub const fn target_kind(&self) -> InterlocutorKind {
        self.target_kind
    }

    /// Coste por defecto asociado a esta plantilla.
    pub const fn default_cost(&self) -> CommCost {
        self.default_cost
    }

    /// Renderiza la plantilla con `params` produciendo un [`CommPayload`] validado.
    pub fn render(
        &self,
        params: &BTreeMap<String, String>,
    ) -> Result<CommPayload, CommunicationError> {
        match &self.kind {
            OfflineTemplateKind::Notice {
                summary_template,
                detail_template,
            } => {
                let summary = render_pattern(&self.id, summary_template, params)?;
                let detail = match detail_template {
                    Some(t) => Some(render_pattern(&self.id, t, params)?),
                    None => None,
                };
                match detail {
                    Some(d) => CommPayload::notice_with_detail(summary, d),
                    None => CommPayload::notice(summary),
                }
            }
            OfflineTemplateKind::Question {
                prompt_template,
                expected_options,
            } => {
                let prompt = render_pattern(&self.id, prompt_template, params)?;
                CommPayload::question(prompt, expected_options.clone())
            }
            OfflineTemplateKind::StructuredFact {
                fact_key,
                fact_value_template,
                attribute_templates,
            } => {
                let fact_value = render_pattern(&self.id, fact_value_template, params)?;
                let mut attributes = BTreeMap::new();
                for (k, v_tpl) in attribute_templates {
                    let rendered = render_pattern(&self.id, v_tpl, params)?;
                    attributes.insert(k.clone(), rendered);
                }
                CommPayload::structured_fact(fact_key.clone(), fact_value, attributes, None, None)
            }
        }
    }
}

fn render_pattern(
    template_id: &TemplateId,
    pattern: &str,
    params: &BTreeMap<String, String>,
) -> Result<String, CommunicationError> {
    let mut out = String::with_capacity(pattern.len());
    let mut rest = pattern;
    while let Some(open_idx) = rest.find('{') {
        out.push_str(&rest[..open_idx]);
        let after_open = &rest[open_idx + 1..];
        let Some(close_idx) = after_open.find('}') else {
            out.push('{');
            rest = after_open;
            continue;
        };
        let key = &after_open[..close_idx];
        if key.trim().is_empty() {
            return Err(CommunicationError::InvalidTemplateParam {
                template_id: template_id.clone(),
                param: key.to_owned(),
            });
        }
        let Some(value) = params.get(key) else {
            return Err(CommunicationError::MissingTemplateParam {
                template_id: template_id.clone(),
                param: key.to_owned(),
            });
        };
        if value.trim().is_empty() {
            return Err(CommunicationError::InvalidTemplateParam {
                template_id: template_id.clone(),
                param: key.to_owned(),
            });
        }
        out.push_str(value);
        rest = &after_open[close_idx + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Registro determinista de plantillas offline reutilizables.
#[derive(Debug, Clone, Default)]
pub struct TemplateRegistry {
    templates: BTreeMap<TemplateId, OfflineTemplate>,
}

impl TemplateRegistry {
    /// Crea un registro vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una nueva [`OfflineTemplate`], rechazando duplicados.
    pub fn register(&mut self, template: OfflineTemplate) -> Result<(), CommunicationError> {
        let id = template.id().clone();
        if self.templates.contains_key(&id) {
            return Err(CommunicationError::DuplicateTemplateId { template_id: id });
        }
        self.templates.insert(id, template);
        Ok(())
    }

    /// Obtiene una plantilla por su identificador.
    pub fn get(&self, id: &TemplateId) -> Option<&OfflineTemplate> {
        self.templates.get(id)
    }

    /// Renderiza una plantilla validando existencia, [`InterlocutorKind`] opcional y payload.
    pub fn render(
        &self,
        id: &TemplateId,
        expected_kind: Option<InterlocutorKind>,
        params: &BTreeMap<String, String>,
    ) -> Result<CommPayload, CommunicationError> {
        let Some(template) = self.templates.get(id) else {
            return Err(CommunicationError::UnknownTemplateId {
                template_id: id.clone(),
            });
        };
        if let Some(expected) = expected_kind
            && template.target_kind() != expected
        {
            return Err(CommunicationError::TemplateKindMismatch {
                template_id: id.clone(),
                expected,
                actual: template.target_kind(),
            });
        }
        template.render(params)
    }

    /// Número de plantillas registradas.
    pub fn len(&self) -> usize {
        self.templates.len()
    }

    /// Indica si el registro está vacío.
    pub fn is_empty(&self) -> bool {
        self.templates.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionSpace;
    use crate::clock::VirtualMonotonicClock;
    use crate::domain::Terminal;
    use crate::guard::{
        ActionGuard, ActionProposal, GuardContext, GuardDecision, GuardRejectionReason,
        SafeFallbackPolicy,
    };
    use crate::units::Tick;

    #[test]
    fn rejects_empty_identifiers_payloads_and_self_addressing() {
        assert!(matches!(
            InterlocutorId::new("   "),
            Err(CommunicationError::EmptyIdentifier {
                field: "interlocutor_id"
            })
        ));
        assert!(matches!(
            ConversationScope::session(""),
            Err(CommunicationError::EmptyIdentifier {
                field: "scope.session"
            })
        ));
        assert!(matches!(
            ConversationScope::channel("  "),
            Err(CommunicationError::EmptyIdentifier {
                field: "scope.channel"
            })
        ));
        assert!(matches!(
            CommPayload::notice("   "),
            Err(CommunicationError::EmptyPayloadField {
                field: "notice.summary"
            })
        ));
        assert!(matches!(
            CommPayload::notice_with_detail("ok", " "),
            Err(CommunicationError::EmptyPayloadField {
                field: "notice.detail"
            })
        ));
        assert!(matches!(
            CommPayload::question("", vec![]),
            Err(CommunicationError::EmptyPayloadField {
                field: "question.prompt"
            })
        ));
        let opt = OptionId::new("yes").unwrap();
        assert!(matches!(
            CommPayload::question("Proceed?", vec![opt.clone(), opt]),
            Err(CommunicationError::DuplicateOptionId { .. })
        ));
        assert!(matches!(
            CommPayload::answer("  ", None, None),
            Err(CommunicationError::EmptyPayloadField {
                field: "answer.text"
            })
        ));
        assert!(matches!(
            CommPayload::structured_fact("", "val", BTreeMap::new(), None, None),
            Err(CommunicationError::EmptyPayloadField {
                field: "structured_fact.fact_key"
            })
        ));
        let mut bad_attrs = BTreeMap::new();
        bad_attrs.insert("zone".to_owned(), "  ".to_owned());
        assert!(matches!(
            CommPayload::structured_fact("status", "ready", bad_attrs, None, None),
            Err(CommunicationError::EmptyPayloadField {
                field: "structured_fact.attribute_value"
            })
        ));

        let alice = InterlocutorId::new("alice").unwrap();
        let err = SituatedMessage::new(
            MessageId::new(1),
            alice.clone(),
            alice,
            ConversationScope::direct(),
            MonotonicInstant::ZERO,
            CommCost::from_units(1),
            CommPayload::notice("self").unwrap(),
        )
        .unwrap_err();
        assert!(matches!(err, CommunicationError::SelfAddressing { .. }));
    }

    #[test]
    fn request_response_and_timeout_reproduce_correlation_and_deterministic_order() {
        let clock = VirtualMonotonicClock::new();
        let mut bus = LocalCommunicationBus::new(8, 8).unwrap();
        let controller = InterlocutorId::new("controller").unwrap();
        let human = InterlocutorId::new("operator").unwrap();

        let q1_id = MessageId::new(10);
        let q2_id = MessageId::new(11);
        let q1 = SituatedMessage::new(
            q1_id,
            controller.clone(),
            human.clone(),
            ConversationScope::session("ep-1").unwrap(),
            clock.now(),
            CommCost::new(1, 12, StdDuration::from_millis(5)),
            CommPayload::question(
                "Confirm sector A?",
                vec![OptionId::new("yes").unwrap(), OptionId::new("no").unwrap()],
            )
            .unwrap(),
        )
        .unwrap()
        .with_deadline(MonotonicDeadline::from_now(
            &clock,
            StdDuration::from_millis(50),
        ));

        let q2 = SituatedMessage::new(
            q2_id,
            controller.clone(),
            human.clone(),
            ConversationScope::session("ep-1").unwrap(),
            clock.now(),
            CommCost::new(1, 10, StdDuration::from_millis(5)),
            CommPayload::question("Confirm sector B?", vec![]).unwrap(),
        )
        .unwrap()
        .with_deadline(MonotonicDeadline::from_now(
            &clock,
            StdDuration::from_millis(100),
        ));

        assert_eq!(bus.request_response(q1, &clock).unwrap(), q1_id);
        assert_eq!(bus.request_response(q2, &clock).unwrap(), q2_id);
        assert_eq!(bus.pending_count(), 2);

        // El humano recibe ambas preguntas en orden FIFO determinista.
        let inbox = bus.poll_inbox(&human, 10);
        assert_eq!(inbox.len(), 2);
        assert_eq!(inbox[0].message_id(), q1_id);
        assert_eq!(inbox[1].message_id(), q2_id);

        // El humano responde primero a q2 (fuera de orden respecto al envío).
        clock.advance(StdDuration::from_millis(20));
        let ans_q2 = SituatedMessage::new(
            MessageId::new(12),
            human.clone(),
            controller.clone(),
            ConversationScope::session("ep-1").unwrap(),
            clock.now(),
            CommCost::new(1, 4, StdDuration::from_millis(2)),
            CommPayload::answer("Sector B clear", None, Some(Confidence::new(0.95).unwrap()))
                .unwrap(),
        )
        .unwrap()
        .with_correlation_id(q2_id)
        .unwrap();
        bus.send(ans_q2.clone()).unwrap();

        // q1 sigue pendiente, q2 recibe su respuesta exacta.
        assert_eq!(
            bus.poll_reply(q1_id, &clock).unwrap(),
            ReplyOutcome::Pending
        );
        assert_eq!(
            bus.poll_reply(q2_id, &clock).unwrap(),
            ReplyOutcome::Answered(ans_q2)
        );
        assert_eq!(bus.pending_count(), 1);

        // Avanzamos el reloj monotónico más allá del deadline de q1 -> TimedOut explícito.
        clock.advance(StdDuration::from_millis(35));
        assert!(matches!(
            bus.poll_reply(q1_id, &clock).unwrap(),
            ReplyOutcome::TimedOut { request_id, .. } if request_id == q1_id
        ));
        assert_eq!(bus.pending_count(), 0);
        assert!(matches!(
            bus.poll_reply(q1_id, &clock),
            Err(CommunicationError::UnknownCorrelationId { .. })
        ));
    }

    #[test]
    fn channels_are_non_blocking_bounded_and_grant_no_authority() {
        let clock = VirtualMonotonicClock::new();
        let mut bus = LocalCommunicationBus::new(1, 1).unwrap();
        let agent_a = InterlocutorId::new("agent_a").unwrap();
        let agent_b = InterlocutorId::new("agent_b").unwrap();

        // Bandeja vacía -> poll_inbox retorna inmediatamente vacío sin bloquear.
        assert!(bus.poll_inbox(&agent_b, 4).is_empty());

        // Llenamos la capacidad (1) y verificamos rechazo inmediato (no bloqueante).
        let m1 = SituatedMessage::new(
            MessageId::new(100),
            agent_a.clone(),
            agent_b.clone(),
            ConversationScope::direct(),
            clock.now(),
            CommCost::from_units(1),
            CommPayload::notice("ping").unwrap(),
        )
        .unwrap();
        let m2 = SituatedMessage::new(
            MessageId::new(101),
            agent_a.clone(),
            agent_b.clone(),
            ConversationScope::direct(),
            clock.now(),
            CommCost::from_units(1),
            CommPayload::notice("overflow").unwrap(),
        )
        .unwrap();

        bus.send(m1).unwrap();
        assert!(matches!(
            bus.send(m2),
            Err(CommunicationError::ChannelFull { capacity: 1, .. })
        ));

        // Un mensaje en el canal (incluso con confianza 1.0) jamás autoriza una acción en ActionGuard.
        let space = ActionSpace::discrete(2).unwrap();
        let guard = ActionGuard::new(SafeFallbackPolicy::RejectOnly);
        let proposal = ActionProposal::unauthorized(Tick::ZERO);
        let ctx = GuardContext::new(
            &space,
            Tick::ZERO,
            std::time::Instant::now(),
            Terminal::Ongoing,
        );
        let decision = guard.evaluate(&proposal, &ctx);
        assert_eq!(
            decision,
            GuardDecision::Reject(GuardRejectionReason::Unauthorized)
        );
    }

    #[test]
    fn same_local_bus_serves_human_and_multiagent_symbol_channel_with_offline_templates() {
        let clock = VirtualMonotonicClock::new();
        let mut bus = LocalCommunicationBus::new(8, 4).unwrap();

        let controller = InterlocutorId::new("controller").unwrap();
        let human = InterlocutorId::new("human_supervisor").unwrap();
        let symbol_channel = InterlocutorId::new("swarm_symbols").unwrap();
        let peer_agent = InterlocutorId::new("agent_beta").unwrap();

        bus.register_interlocutor(controller.clone(), InterlocutorKind::Agent);
        bus.register_interlocutor(human.clone(), InterlocutorKind::Human);
        bus.register_interlocutor(symbol_channel.clone(), InterlocutorKind::SymbolChannel);
        bus.register_interlocutor(peer_agent.clone(), InterlocutorKind::Agent);

        let mut registry = TemplateRegistry::new();
        let human_tpl_id = TemplateId::new("human_clarification").unwrap();
        let symbol_tpl_id = TemplateId::new("symbol_hazard_fact").unwrap();

        registry
            .register(
                OfflineTemplate::new(
                    human_tpl_id.clone(),
                    InterlocutorKind::Human,
                    CommCost::new(1, 20, StdDuration::from_millis(10)),
                    OfflineTemplateKind::Question {
                        prompt_template: "Route {route} blocked by {obstacle}. Override?"
                            .to_owned(),
                        expected_options: vec![
                            OptionId::new("approve").unwrap(),
                            OptionId::new("abort").unwrap(),
                        ],
                    },
                )
                .unwrap(),
            )
            .unwrap();

        let mut symbol_attrs = BTreeMap::new();
        symbol_attrs.insert("zone".to_owned(), "{zone}".to_owned());
        registry
            .register(
                OfflineTemplate::new(
                    symbol_tpl_id.clone(),
                    InterlocutorKind::SymbolChannel,
                    CommCost::new(1, 6, StdDuration::from_millis(1)),
                    OfflineTemplateKind::StructuredFact {
                        fact_key: "hazard_symbol".to_owned(),
                        fact_value_template: "{symbol_code}".to_owned(),
                        attribute_templates: symbol_attrs,
                    },
                )
                .unwrap(),
            )
            .unwrap();

        let mut human_params = BTreeMap::new();
        human_params.insert("route".to_owned(), "north-corridor".to_owned());
        human_params.insert("obstacle".to_owned(), "gate-locked".to_owned());
        let human_payload = registry
            .render(&human_tpl_id, Some(InterlocutorKind::Human), &human_params)
            .unwrap();

        let mut sym_params = BTreeMap::new();
        sym_params.insert("symbol_code".to_owned(), "SYM_LOCK_04".to_owned());
        sym_params.insert("zone".to_owned(), "north-corridor".to_owned());
        let symbol_payload = registry
            .render(
                &symbol_tpl_id,
                Some(InterlocutorKind::SymbolChannel),
                &sym_params,
            )
            .unwrap();

        // Rechaza desajuste de tipo de interlocutor o parámetro faltante.
        assert!(matches!(
            registry.render(&symbol_tpl_id, Some(InterlocutorKind::Human), &sym_params),
            Err(CommunicationError::TemplateKindMismatch { .. })
        ));
        assert!(matches!(
            registry.render(&human_tpl_id, None, &BTreeMap::new()),
            Err(CommunicationError::MissingTemplateParam { .. })
        ));

        // Ambos mensajes viajan por el mismo LocalCommunicationBus.
        let human_q = SituatedMessage::new(
            MessageId::new(201),
            controller.clone(),
            human.clone(),
            ConversationScope::direct(),
            clock.now(),
            registry.get(&human_tpl_id).unwrap().default_cost(),
            human_payload,
        )
        .unwrap();
        let sym_fact = SituatedMessage::new(
            MessageId::new(202),
            peer_agent,
            symbol_channel.clone(),
            ConversationScope::channel("multiagent-grid").unwrap(),
            clock.now(),
            registry.get(&symbol_tpl_id).unwrap().default_cost(),
            symbol_payload,
        )
        .unwrap();

        bus.request_response(human_q, &clock).unwrap();
        bus.send(sym_fact).unwrap();

        assert_eq!(bus.inbox_len(&human), 1);
        assert_eq!(bus.inbox_len(&symbol_channel), 1);
        assert_eq!(bus.total_cost().units(), 2);
        assert_eq!(bus.total_cost().tokens(), 26);

        let received_sym = bus.poll_inbox(&symbol_channel, 1);
        assert_eq!(received_sym.len(), 1);
        assert_eq!(
            received_sym[0].payload().kind(),
            CommPayloadKind::StructuredFact
        );
    }
}
