//! Envelope tipado de observaciones, procedencia y detección de ingreso (`std`-only).
//!
//! Este módulo define los contratos de procedencia epistémica para observaciones:
//! - [`ObservationKind`]: distingue explícitamente entre medición (`Measurement`),
//!   afirmación (`Assertion`), intención (`Intention`), propuesta (`Proposal`) y
//!   desconocido (`Unknown`).
//! - [`Confidence`]: valor finito validado en `[0.0, 1.0]` **sin** conversión implícita
//!   a `bool` de verdad ni a permiso/autorización de acción.
//! - [`ObservationEnvelope<T>`]: envuelve una carga útil `T` con metadatos inmutables de
//!   procedencia ([`MessageId`], [`SensorId`], [`FrameId`], [`UnitTag`], [`EvidenceRef`],
//!   `observed_at`, `received_at`, `revision`, `confidence`, `claim_kind`).
//! - [`ProvenanceTracker`] / [`IngressWindow`]: detecta mensajes duplicados
//!   ([`IngressStatus::Duplicate`]) y tardíos o fuera de orden
//!   ([`IngressStatus::LateOutOfOrder`]) por `(sensor_id, message_id)`, tiempo observado
//!   y revisión.

use std::collections::{HashMap, HashSet, VecDeque};
use std::error::Error;
use std::fmt;

use crate::clock::MonotonicInstant;

/// Error de validación en la construcción de tipos de procedencia o ventanas de ingreso.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceError {
    /// El valor de confianza no es finito o está fuera del rango `[0.0, 1.0]`.
    InvalidConfidence {
        /// Valor recibido.
        value: f64,
    },
    /// Un identificador textual requerido (`FrameId` o `UnitTag`) está vacío o solo contiene espacios.
    EmptyIdentifier {
        /// Nombre del campo inválido.
        field: &'static str,
    },
    /// La capacidad máxima de historial de la ventana de ingreso debe ser al menos 1.
    InvalidWindowCapacity {
        /// Capacidad recibida.
        capacity: usize,
    },
}

impl fmt::Display for ProvenanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidence { value } => {
                write!(
                    f,
                    "invalid confidence ({value}): must be finite and within [0.0, 1.0]"
                )
            }
            Self::EmptyIdentifier { field } => {
                write!(f, "provenance identifier '{field}' cannot be empty")
            }
            Self::InvalidWindowCapacity { capacity } => {
                write!(
                    f,
                    "invalid ingress window capacity ({capacity}): must be >= 1"
                )
            }
        }
    }
}

impl Error for ProvenanceError {}

/// Identificador único de mensaje emitido por un sensor o fuente de observación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MessageId(u64);

impl MessageId {
    /// Crea un [`MessageId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for MessageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "msg({})", self.0)
    }
}

/// Identificador estructurado del sensor o fuente emisora de la observación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SensorId(u64);

impl SensorId {
    /// Crea un [`SensorId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SensorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sensor({})", self.0)
    }
}

/// Marco de referencia espacial o lógico asociado a una observación.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameId(String);

impl FrameId {
    /// Construye un [`FrameId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, ProvenanceError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ProvenanceError::EmptyIdentifier { field: "frame_id" });
        }
        Ok(Self(id))
    }

    /// Marco de referencia raíz por defecto (`"world"`).
    pub fn world() -> Self {
        Self("world".to_owned())
    }

    /// Devuelve la representación textual del marco de referencia.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for FrameId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Etiqueta explícita de unidades físicas o lógicas de la observación.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnitTag(String);

impl UnitTag {
    /// Construye un [`UnitTag`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(tag: impl Into<String>) -> Result<Self, ProvenanceError> {
        let tag = tag.into();
        if tag.trim().is_empty() {
            return Err(ProvenanceError::EmptyIdentifier { field: "units" });
        }
        Ok(Self(tag))
    }

    /// Etiqueta estándar para magnitudes adimensionales (`"dimensionless"`).
    pub fn dimensionless() -> Self {
        Self("dimensionless".to_owned())
    }

    /// Devuelve la representación textual de la etiqueta de unidades.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for UnitTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Referencia estructurada a la evidencia que respalda la observación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EvidenceRef(u64);

impl EvidenceRef {
    /// Crea un [`EvidenceRef`] a partir de un identificador numérico de traza o artefacto.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EvidenceRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "evidence({})", self.0)
    }
}

/// Nivel de confianza escalar finito en el intervalo cerrado `[0.0, 1.0]`.
///
/// # Seguridad epistémica
/// `Confidence` expresa únicamente una medida graduada de incertidumbre/soporte en `[0.0, 1.0]`.
/// **No existe ninguna conversión implícita** (`From`/`Into`, `Deref`, `PartialEq<bool>`,
/// ni métodos de acceso a `bool`) de `Confidence` hacia un valor de verdad ni hacia un
/// permiso o autorización de acción.
///
/// ```compile_fail
/// use gogamen_core::Confidence;
/// let c = Confidence::FULL;
/// let _truth: bool = c.into();
/// ```
///
/// ```compile_fail
/// use gogamen_core::Confidence;
/// let c = Confidence::FULL;
/// if c {
///     // Prohibido: Confidence no es un booleano de verdad.
/// }
/// ```
///
/// ```compile_fail
/// use gogamen_core::{Confidence, Permission};
/// let c = Confidence::FULL;
/// let _perm: Permission = c.into();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f64);

impl Confidence {
    /// Confianza mínima (`0.0`).
    pub const ZERO: Self = Self(0.0);

    /// Confianza máxima (`1.0`).
    pub const FULL: Self = Self(1.0);

    /// Construye un [`Confidence`] validando que sea finito y pertenezca a `[0.0, 1.0]`.
    pub fn new(value: f64) -> Result<Self, ProvenanceError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(ProvenanceError::InvalidConfidence { value });
        }
        let normalized = if value == 0.0 { 0.0 } else { value };
        Ok(Self(normalized))
    }

    /// Devuelve el valor escalar en `[0.0, 1.0]`.
    pub const fn as_f64(self) -> f64 {
        self.0
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

/// Clasificación epistémica explícita del tipo de reclamo transportado por una observación.
///
/// Permite distinguir sin ambigüedad entre mediciones directas, afirmaciones, intenciones,
/// propuestas y origen desconocido.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ObservationKind {
    /// Medición directa obtenida por un sensor o instrumento sobre el entorno o sistema.
    Measurement,
    /// Afirmación declarativa emitida por un módulo o agente sobre un hecho o estado.
    Assertion,
    /// Intención declarada de comportamiento futuro; no representa un hecho observado.
    Intention,
    /// Propuesta candidata sujeta a validación o arbitraje; no otorga permiso ni verdad.
    Proposal,
    /// Origen o naturaleza epistémica desconocida o no clasificada.
    Unknown,
}

impl ObservationKind {
    /// Devuelve `true` si la observación es una medición directa (`Measurement`).
    pub const fn is_measurement(self) -> bool {
        matches!(self, Self::Measurement)
    }

    /// Devuelve `true` si la observación es una afirmación (`Assertion`).
    pub const fn is_assertion(self) -> bool {
        matches!(self, Self::Assertion)
    }

    /// Devuelve `true` si la observación expresa una intención (`Intention`).
    pub const fn is_intention(self) -> bool {
        matches!(self, Self::Intention)
    }

    /// Devuelve `true` si la observación es una propuesta (`Proposal`).
    pub const fn is_proposal(self) -> bool {
        matches!(self, Self::Proposal)
    }

    /// Devuelve `true` si la clasificación es desconocida (`Unknown`).
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }
}

impl fmt::Display for ObservationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Measurement => "measurement",
            Self::Assertion => "assertion",
            Self::Intention => "intention",
            Self::Proposal => "proposal",
            Self::Unknown => "unknown",
        };
        f.write_str(label)
    }
}

/// Metadatos de procedencia y calidad epistémica de una observación.
#[derive(Debug, Clone, PartialEq)]
pub struct ProvenanceMetadata {
    /// Identificador único del mensaje.
    pub message_id: MessageId,
    /// Instante monotónico en que se realizó la observación.
    pub observed_at: MonotonicInstant,
    /// Instante monotónico en que se recibió el mensaje.
    pub received_at: MonotonicInstant,
    /// Identificador del sensor o fuente emisora.
    pub sensor_id: SensorId,
    /// Marco de referencia espacial o lógico.
    pub frame_id: FrameId,
    /// Unidades físicas o lógicas de la carga útil.
    pub units: UnitTag,
    /// Confianza escalar validada en `[0.0, 1.0]`.
    pub confidence: Confidence,
    /// Número de revisión monotónica de la fuente o mensaje.
    pub revision: u64,
    /// Referencia a la evidencia de respaldo.
    pub evidence_ref: EvidenceRef,
    /// Clasificación epistémica explícita del reclamo.
    pub claim_kind: ObservationKind,
}

/// Envelope inmutable de observación con metadatos completos de procedencia.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationEnvelope<T> {
    metadata: ProvenanceMetadata,
    payload: T,
}

/// Alias semántico de [`ObservationEnvelope`] para registros de observación con procedencia.
pub type ObservationRecord<T> = ObservationEnvelope<T>;

impl<T> ObservationEnvelope<T> {
    /// Construye un [`ObservationEnvelope`] combinando metadatos validados y carga útil.
    pub const fn new(metadata: ProvenanceMetadata, payload: T) -> Self {
        Self { metadata, payload }
    }

    /// Devuelve el identificador único del mensaje.
    pub const fn message_id(&self) -> MessageId {
        self.metadata.message_id
    }

    /// Devuelve el instante monotónico en que se observó el dato.
    pub const fn observed_at(&self) -> MonotonicInstant {
        self.metadata.observed_at
    }

    /// Devuelve el instante monotónico en que se recibió el mensaje.
    pub const fn received_at(&self) -> MonotonicInstant {
        self.metadata.received_at
    }

    /// Devuelve el identificador del sensor o fuente emisora.
    pub const fn sensor_id(&self) -> SensorId {
        self.metadata.sensor_id
    }

    /// Devuelve el marco de referencia espacial o lógico.
    pub const fn frame_id(&self) -> &FrameId {
        &self.metadata.frame_id
    }

    /// Devuelve la etiqueta de unidades de la carga útil.
    pub const fn units(&self) -> &UnitTag {
        &self.metadata.units
    }

    /// Devuelve el nivel de confianza validado en `[0.0, 1.0]`.
    pub const fn confidence(&self) -> Confidence {
        self.metadata.confidence
    }

    /// Devuelve la revisión asociada al mensaje.
    pub const fn revision(&self) -> u64 {
        self.metadata.revision
    }

    /// Devuelve la referencia a la evidencia que respalda la observación.
    pub const fn evidence_ref(&self) -> EvidenceRef {
        self.metadata.evidence_ref
    }

    /// Devuelve la clasificación epistémica del reclamo.
    pub const fn claim_kind(&self) -> ObservationKind {
        self.metadata.claim_kind
    }

    /// Devuelve una referencia inmutable a todos los metadatos de procedencia.
    pub const fn metadata(&self) -> &ProvenanceMetadata {
        &self.metadata
    }

    /// Devuelve una referencia inmutable a la carga útil observada.
    pub const fn payload(&self) -> &T {
        &self.payload
    }

    /// Consume el envelope y devuelve la carga útil.
    pub fn into_payload(self) -> T {
        self.payload
    }

    /// Transforma la carga útil preservando intactos los metadatos de procedencia.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> ObservationEnvelope<U> {
        ObservationEnvelope {
            metadata: self.metadata,
            payload: f(self.payload),
        }
    }
}

/// Estado resultante de evaluar un mensaje entrante en [`ProvenanceTracker`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IngressStatus {
    /// El mensaje es nuevo y mantiene orden monotónico de tiempo observado y revisión.
    Accepted,
    /// Ya se registró previamente un mensaje con el mismo `(sensor_id, message_id)`.
    Duplicate,
    /// El mensaje tiene un ID nuevo, pero llegó fuera de orden por `observed_at` o `revision`.
    LateOutOfOrder,
}

impl IngressStatus {
    /// Devuelve `true` si el mensaje fue aceptado en orden.
    pub const fn is_accepted(self) -> bool {
        matches!(self, Self::Accepted)
    }

    /// Devuelve `true` si el mensaje fue detectado como duplicado.
    pub const fn is_duplicate(self) -> bool {
        matches!(self, Self::Duplicate)
    }

    /// Devuelve `true` si el mensaje fue detectado como tardío o fuera de orden.
    pub const fn is_late_out_of_order(self) -> bool {
        matches!(self, Self::LateOutOfOrder)
    }
}

/// Marca de agua temporal y de revisión observada para un sensor concreto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorWatermark {
    /// Último instante `observed_at` aceptado para el sensor.
    pub last_observed_at: MonotonicInstant,
    /// Última revisión aceptada para el sensor.
    pub last_revision: u64,
}

/// Rastreador de procedencia e ingreso para detectar duplicados y mensajes tardíos.
///
/// Registra pares `(sensor_id, message_id)` ya vistos y mantiene la última marca
/// `(observed_at, revision)` aceptada por cada [`SensorId`].
#[derive(Debug, Clone, Default)]
pub struct ProvenanceTracker {
    seen_messages: HashSet<(SensorId, MessageId)>,
    seen_order: VecDeque<(SensorId, MessageId)>,
    max_seen_messages: Option<usize>,
    watermarks: HashMap<SensorId, SensorWatermark>,
}

/// Alias semántico de [`ProvenanceTracker`] para ventanas de ingreso de observaciones.
pub type IngressWindow = ProvenanceTracker;

impl ProvenanceTracker {
    /// Crea un [`ProvenanceTracker`] sin límite de historial de IDs vistos.
    pub fn new() -> Self {
        Self::default()
    }

    /// Crea un [`ProvenanceTracker`] acotado que retiene como máximo `max_seen_messages` pares
    /// `(sensor_id, message_id)` en orden FIFO.
    pub fn with_max_seen_messages(max_seen_messages: usize) -> Result<Self, ProvenanceError> {
        if max_seen_messages == 0 {
            return Err(ProvenanceError::InvalidWindowCapacity {
                capacity: max_seen_messages,
            });
        }
        Ok(Self {
            seen_messages: HashSet::with_capacity(max_seen_messages.min(1024)),
            seen_order: VecDeque::with_capacity(max_seen_messages.min(1024)),
            max_seen_messages: Some(max_seen_messages),
            watermarks: HashMap::new(),
        })
    }

    /// Evalúa y registra un [`ObservationEnvelope`], devolviendo su [`IngressStatus`].
    pub fn record<T>(&mut self, envelope: &ObservationEnvelope<T>) -> IngressStatus {
        self.record_metadata(envelope.metadata())
    }

    /// Evalúa y registra [`ProvenanceMetadata`], devolviendo su [`IngressStatus`].
    pub fn record_metadata(&mut self, metadata: &ProvenanceMetadata) -> IngressStatus {
        let key = (metadata.sensor_id, metadata.message_id);
        if self.seen_messages.contains(&key) {
            return IngressStatus::Duplicate;
        }

        if let Some(limit) = self.max_seen_messages
            && self.seen_order.len() == limit
            && let Some(evicted) = self.seen_order.pop_front()
        {
            self.seen_messages.remove(&evicted);
        }

        self.seen_messages.insert(key);
        self.seen_order.push_back(key);

        match self.watermarks.get_mut(&metadata.sensor_id) {
            Some(watermark) => {
                if metadata.observed_at < watermark.last_observed_at
                    || metadata.revision < watermark.last_revision
                {
                    IngressStatus::LateOutOfOrder
                } else {
                    watermark.last_observed_at = metadata.observed_at;
                    watermark.last_revision = metadata.revision;
                    IngressStatus::Accepted
                }
            }
            None => {
                self.watermarks.insert(
                    metadata.sensor_id,
                    SensorWatermark {
                        last_observed_at: metadata.observed_at,
                        last_revision: metadata.revision,
                    },
                );
                IngressStatus::Accepted
            }
        }
    }

    /// Indica si un par `(sensor_id, message_id)` se encuentra actualmente registrado.
    pub fn contains(&self, sensor_id: SensorId, message_id: MessageId) -> bool {
        self.seen_messages.contains(&(sensor_id, message_id))
    }

    /// Devuelve la cantidad de pares `(sensor_id, message_id)` actualmente retenidos.
    pub fn seen_count(&self) -> usize {
        self.seen_messages.len()
    }

    /// Devuelve la última marca de agua aceptada para un sensor, si existe.
    pub fn last_watermark(&self, sensor_id: SensorId) -> Option<SensorWatermark> {
        self.watermarks.get(&sensor_id).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Observation;
    use crate::units::{SimDuration, Tick};

    fn sample_meta(
        sensor_id: u64,
        message_id: u64,
        observed_ns: u128,
        received_ns: u128,
        revision: u64,
        claim_kind: ObservationKind,
    ) -> ProvenanceMetadata {
        ProvenanceMetadata {
            message_id: MessageId::new(message_id),
            observed_at: MonotonicInstant::from_nanos(observed_ns),
            received_at: MonotonicInstant::from_nanos(received_ns),
            sensor_id: SensorId::new(sensor_id),
            frame_id: FrameId::new("base_link").unwrap(),
            units: UnitTag::new("m/s").unwrap(),
            confidence: Confidence::new(0.85).unwrap(),
            revision,
            evidence_ref: EvidenceRef::new(42),
            claim_kind,
        }
    }

    #[test]
    fn ac1_distinguishes_all_observation_kinds_and_preserves_provenance() {
        let obs = Observation::new(Tick::new(3), SimDuration::ZERO, vec![1.25]).unwrap();
        let kinds = [
            ObservationKind::Measurement,
            ObservationKind::Assertion,
            ObservationKind::Intention,
            ObservationKind::Proposal,
            ObservationKind::Unknown,
        ];

        let envelopes: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(idx, &kind)| {
                let meta = sample_meta(1, idx as u64 + 1, 100, 110, idx as u64 + 1, kind);
                ObservationEnvelope::new(meta, obs.clone())
            })
            .collect();

        assert!(envelopes[0].claim_kind().is_measurement());
        assert!(envelopes[1].claim_kind().is_assertion());
        assert!(envelopes[2].claim_kind().is_intention());
        assert!(envelopes[3].claim_kind().is_proposal());
        assert!(envelopes[4].claim_kind().is_unknown());

        // Cada variante es distinta entre sí y preserva sus metadatos.
        for (i, env_i) in envelopes.iter().enumerate() {
            assert_eq!(env_i.sensor_id(), SensorId::new(1));
            assert_eq!(env_i.frame_id().as_str(), "base_link");
            assert_eq!(env_i.units().as_str(), "m/s");
            assert_eq!(env_i.evidence_ref(), EvidenceRef::new(42));
            assert_eq!(env_i.confidence().as_f64(), 0.85);
            for (j, env_j) in envelopes.iter().enumerate() {
                if i != j {
                    assert_ne!(env_i.claim_kind(), env_j.claim_kind());
                }
            }
        }
    }

    #[test]
    fn ac2_detects_duplicates_and_late_out_of_order_messages_by_id_and_time() {
        let mut tracker = ProvenanceTracker::new();

        // Primer mensaje del sensor 1 aceptado.
        let m1 = ObservationEnvelope::new(
            sample_meta(1, 10, 1_000, 1_050, 1, ObservationKind::Measurement),
            42_i32,
        );
        assert_eq!(tracker.record(&m1), IngressStatus::Accepted);

        // Duplicado exacto por (sensor_id, message_id), incluso si cambia tiempo o payload.
        let m1_dup = ObservationEnvelope::new(
            sample_meta(1, 10, 2_000, 2_050, 2, ObservationKind::Measurement),
            99_i32,
        );
        assert_eq!(tracker.record(&m1_dup), IngressStatus::Duplicate);

        // Mismo message_id en otro sensor no es duplicado.
        let s2_m1 = ObservationEnvelope::new(
            sample_meta(2, 10, 500, 520, 1, ObservationKind::Assertion),
            7_i32,
        );
        assert_eq!(tracker.record(&s2_m1), IngressStatus::Accepted);

        // Mensaje nuevo del sensor 1 con observed_at anterior al último aceptado -> LateOutOfOrder.
        let m_late_time = ObservationEnvelope::new(
            sample_meta(1, 11, 900, 1_100, 2, ObservationKind::Measurement),
            43_i32,
        );
        assert_eq!(tracker.record(&m_late_time), IngressStatus::LateOutOfOrder);
        // La marca de agua del sensor 1 no retrocede.
        assert_eq!(
            tracker.last_watermark(SensorId::new(1)),
            Some(SensorWatermark {
                last_observed_at: MonotonicInstant::from_nanos(1_000),
                last_revision: 1,
            })
        );

        // Reenvío del mensaje tardío sigue detectándose como Duplicate.
        assert_eq!(tracker.record(&m_late_time), IngressStatus::Duplicate);

        // Mensaje nuevo con tiempo válido pero revisión regresiva -> LateOutOfOrder.
        let m_advance = ObservationEnvelope::new(
            sample_meta(1, 12, 1_500, 1_550, 5, ObservationKind::Measurement),
            44_i32,
        );
        assert_eq!(tracker.record(&m_advance), IngressStatus::Accepted);

        let m_late_rev = ObservationEnvelope::new(
            sample_meta(1, 13, 1_600, 1_650, 4, ObservationKind::Measurement),
            45_i32,
        );
        assert_eq!(tracker.record(&m_late_rev), IngressStatus::LateOutOfOrder);
    }

    #[test]
    fn ac2_bounded_ingress_window_evicts_oldest_seen_ids() {
        assert!(matches!(
            IngressWindow::with_max_seen_messages(0),
            Err(ProvenanceError::InvalidWindowCapacity { capacity: 0 })
        ));

        let mut window = IngressWindow::with_max_seen_messages(2).unwrap();
        let m1 = sample_meta(1, 1, 100, 100, 1, ObservationKind::Measurement);
        let m2 = sample_meta(1, 2, 200, 200, 2, ObservationKind::Measurement);
        let m3 = sample_meta(1, 3, 300, 300, 3, ObservationKind::Measurement);

        assert_eq!(window.record_metadata(&m1), IngressStatus::Accepted);
        assert_eq!(window.record_metadata(&m2), IngressStatus::Accepted);
        assert_eq!(window.seen_count(), 2);

        // Al insertar m3 se expulsa (1, 1).
        assert_eq!(window.record_metadata(&m3), IngressStatus::Accepted);
        assert_eq!(window.seen_count(), 2);
        assert!(!window.contains(SensorId::new(1), MessageId::new(1)));
        assert!(window.contains(SensorId::new(1), MessageId::new(2)));
        assert!(window.contains(SensorId::new(1), MessageId::new(3)));
    }

    #[test]
    fn ac3_confidence_validates_finite_unit_interval_and_has_no_truth_or_permission_conversion() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0001, 1.0001] {
            assert!(matches!(
                Confidence::new(invalid),
                Err(ProvenanceError::InvalidConfidence { .. })
            ));
        }

        let zero = Confidence::new(-0.0).unwrap();
        assert_eq!(zero.as_f64(), 0.0);
        assert!(zero.as_f64().is_sign_positive());
        assert_eq!(Confidence::ZERO.as_f64(), 0.0);
        assert_eq!(Confidence::FULL.as_f64(), 1.0);

        // Verificación estática adicional de que Confidence no implementa From<Confidence> for bool
        // ni Into<Permission>. Las pruebas compile_fail en rustdoc verifican el rechazo en compilación.
        let c = Confidence::new(0.99).unwrap();
        let scalar: f64 = c.as_f64();
        assert!((scalar - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn validates_identifiers_and_maps_payload_preserving_metadata() {
        assert!(matches!(
            FrameId::new("   "),
            Err(ProvenanceError::EmptyIdentifier { field: "frame_id" })
        ));
        assert!(matches!(
            UnitTag::new(""),
            Err(ProvenanceError::EmptyIdentifier { field: "units" })
        ));

        let meta = sample_meta(7, 77, 10, 20, 1, ObservationKind::Proposal);
        let env = ObservationEnvelope::new(meta.clone(), 10_u32);
        let mapped = env.map(|v| v * 2);
        assert_eq!(mapped.metadata(), &meta);
        assert_eq!(*mapped.payload(), 20);
        assert_eq!(mapped.into_payload(), 20);
    }
}
