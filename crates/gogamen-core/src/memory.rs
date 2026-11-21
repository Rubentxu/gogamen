//! Puerto de memoria episódica acotada e implementación local determinista (`std`-only).
//!
//! Este módulo define el contrato [`MemoryPort`] y la implementación en memoria
//! [`BoundedEpisodicMemory`] con las siguientes garantías:
//! - **Aislamiento estricto por individuo/agente:** todo episodio pertenece a un [`OwnerId`]
//!   y toda consulta requiere ese propietario explícito; opcionalmente [`AgentMemoryScope`]
//!   encapsula y valida el ámbito en tiempo de ejecución.
//! - **Sin texto personal bruto por defecto:** [`EpisodicRecord`] no almacena texto libre
//!   ni PII. Ante entradas con texto bruto, [`RawTextPolicy::RejectByDefault`] las rechaza
//!   sin mutar estado y [`RawTextPolicy::RedactAndDrop`] descarta el texto marcando
//!   `raw_text_redacted = true`.
//! - **Retención y expiración acotadas y deterministas:** [`RetentionPolicy`] limita la
//!   capacidad por propietario (`max_episodes_per_owner`), la capacidad global
//!   (`max_total_episodes`) y el tiempo de vida monotónico (`ttl_nanos`).
//! - **Snapshot y replay determinista:** [`MemorySnapshot`] y [`BoundedEpisodicMemory::replay_operations`]
//!   permiten reconstruir y verificar exactamente el estado de un fixture.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::clock::MonotonicInstant;
use crate::observability::EntityId;
use crate::provenance::{
    Confidence, EvidenceRef, MessageId, ObservationKind, ProvenanceMetadata, SensorId,
};
use crate::units::Tick;

/// Versión de esquema actual de los snapshots de memoria episódica.
pub const MEMORY_SCHEMA_VERSION: u32 = 1;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Identificador único de propietario (individuo o agente) de episodios en memoria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct OwnerId(u64);

impl OwnerId {
    /// Crea un [`OwnerId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for OwnerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "owner({})", self.0)
    }
}

/// Identificador único y monotónico asignado a cada episodio registrado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EpisodeId(u64);

impl EpisodeId {
    /// Crea un [`EpisodeId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EpisodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "episode({})", self.0)
    }
}

/// Código estructurado de resumen episódico sin texto libre.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SummaryCode(u64);

impl SummaryCode {
    /// Crea un [`SummaryCode`] a partir de un código numérico estructurado.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SummaryCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "summary({})", self.0)
    }
}

/// Política de tratamiento de texto bruto o potencial PII en el ingreso de episodios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RawTextPolicy {
    /// Rechaza cualquier entrada que incluya texto bruto (valor por defecto).
    #[default]
    RejectByDefault,
    /// Descarta explícitamente el texto bruto sin almacenarlo, dejando constancia en el registro.
    RedactAndDrop,
}

impl RawTextPolicy {
    const fn as_u8(self) -> u8 {
        match self {
            Self::RejectByDefault => 0,
            Self::RedactAndDrop => 1,
        }
    }
}

/// Política acotada de retención y expiración de episodios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RetentionPolicy {
    /// Máximo de episodios retenidos por cada [`OwnerId`].
    pub max_episodes_per_owner: usize,
    /// Máximo total de episodios retenidos en toda la memoria.
    pub max_total_episodes: usize,
    /// Tiempo de vida en nanosegundos monotónicos (`None` desactiva la expiración por TTL).
    pub ttl_nanos: Option<u128>,
}

impl RetentionPolicy {
    /// Construye y valida una [`RetentionPolicy`].
    pub fn new(
        max_episodes_per_owner: usize,
        max_total_episodes: usize,
        ttl_nanos: Option<u128>,
    ) -> Result<Self, MemoryError> {
        if max_episodes_per_owner == 0 {
            return Err(MemoryError::InvalidRetentionPolicy {
                reason: "max_episodes_per_owner must be >= 1",
            });
        }
        if max_total_episodes == 0 {
            return Err(MemoryError::InvalidRetentionPolicy {
                reason: "max_total_episodes must be >= 1",
            });
        }
        if matches!(ttl_nanos, Some(0)) {
            return Err(MemoryError::InvalidRetentionPolicy {
                reason: "ttl_nanos, when Some, must be >= 1",
            });
        }
        Ok(Self {
            max_episodes_per_owner,
            max_total_episodes,
            ttl_nanos,
        })
    }
}

/// Errores del puerto y almacenamiento de memoria episódica.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryError {
    /// Parámetros inválidos en [`RetentionPolicy`].
    InvalidRetentionPolicy {
        /// Motivo del rechazo.
        reason: &'static str,
    },
    /// Se intentó registrar texto bruto con [`RawTextPolicy::RejectByDefault`].
    RawTextRejected {
        /// Política activa que rechazó el texto.
        policy: RawTextPolicy,
    },
    /// Un rasgo numérico (`f64`) es `NaN` o infinito.
    NonFiniteFeature {
        /// Índice del elemento inválido en `numeric_features`.
        index: usize,
        /// Valor no finito recibido.
        value: f64,
    },
    /// El rango temporal `[start, end]` de la consulta es inválido (`start > end`).
    InvalidTimeRange {
        /// Inicio del intervalo.
        start: MonotonicInstant,
        /// Fin del intervalo.
        end: MonotonicInstant,
    },
    /// Intento de operar sobre un [`OwnerId`] distinto al del [`AgentMemoryScope`].
    ScopeMismatch {
        /// Propietario esperado por el ámbito.
        expected: OwnerId,
        /// Propietario recibido en la operación.
        actual: OwnerId,
    },
    /// El snapshot es inválido, inconsistente o tiene el checksum corrupto.
    InvalidSnapshot {
        /// Motivo de invalidez.
        reason: &'static str,
    },
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRetentionPolicy { reason } => {
                write!(f, "invalid retention policy: {reason}")
            }
            Self::RawTextRejected { policy } => {
                write!(f, "raw text rejected by memory policy ({policy:?})")
            }
            Self::NonFiniteFeature { index, value } => {
                write!(
                    f,
                    "non-finite numeric feature at index {index} ({value}): must be finite"
                )
            }
            Self::InvalidTimeRange { start, end } => {
                write!(f, "invalid query time range: start ({start}) > end ({end})")
            }
            Self::ScopeMismatch { expected, actual } => {
                write!(
                    f,
                    "agent memory scope mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidSnapshot { reason } => {
                write!(f, "invalid memory snapshot: {reason}")
            }
        }
    }
}

impl Error for MemoryError {}

/// Procedencia resumida y libre de texto asociada a un episodio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpisodeProvenance {
    /// Referencia estructurada a la evidencia de respaldo.
    pub evidence_ref: EvidenceRef,
    /// Identificador único del mensaje de origen.
    pub message_id: MessageId,
    /// Identificador del sensor o fuente emisora.
    pub sensor_id: SensorId,
    /// Clasificación epistémica de la observación.
    pub claim_kind: ObservationKind,
    /// Confianza escalar validada en `[0.0, 1.0]`.
    pub confidence: Confidence,
}

impl EpisodeProvenance {
    /// Crea una procedencia completa para un episodio.
    pub const fn new(
        evidence_ref: EvidenceRef,
        message_id: MessageId,
        sensor_id: SensorId,
        claim_kind: ObservationKind,
        confidence: Confidence,
    ) -> Self {
        Self {
            evidence_ref,
            message_id,
            sensor_id,
            claim_kind,
            confidence,
        }
    }

    /// Construye una procedencia mínima a partir de una [`EvidenceRef`].
    pub const fn from_evidence(evidence_ref: EvidenceRef) -> Self {
        Self {
            evidence_ref,
            message_id: MessageId::new(0),
            sensor_id: SensorId::new(0),
            claim_kind: ObservationKind::Unknown,
            confidence: Confidence::ZERO,
        }
    }
}

impl From<&ProvenanceMetadata> for EpisodeProvenance {
    fn from(meta: &ProvenanceMetadata) -> Self {
        Self {
            evidence_ref: meta.evidence_ref,
            message_id: meta.message_id,
            sensor_id: meta.sensor_id,
            claim_kind: meta.claim_kind,
            confidence: meta.confidence,
        }
    }
}

/// Solicitud de registro de un episodio en memoria.
///
/// Puede transportar opcionalmente `raw_text` de entrada para que la política
/// [`RawTextPolicy`] decida si rechazarla o redactarla antes de persistir el
/// [`EpisodicRecord`] sin texto.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeInput {
    /// Propietario del episodio.
    pub owner: OwnerId,
    /// Entidad observada.
    pub entity: EntityId,
    /// Tick discreto de simulación asociado.
    pub tick: Tick,
    /// Instante monotónico de observación.
    pub observed_at: MonotonicInstant,
    /// Revisión monotónica de la fuente o flujo.
    pub revision: u64,
    /// Procedencia epistémica resumida.
    pub provenance: EpisodeProvenance,
    /// Código estructurado de resumen sin texto libre.
    pub summary_code: SummaryCode,
    /// Vector acotado de rasgos numéricos finitos.
    pub numeric_features: Vec<f64>,
    /// Texto bruto opcional (rechazado por defecto según [`RawTextPolicy`]).
    pub raw_text: Option<String>,
}

impl EpisodeInput {
    /// Crea una entrada de episodio sin texto bruto.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        owner: OwnerId,
        entity: EntityId,
        tick: Tick,
        observed_at: MonotonicInstant,
        revision: u64,
        provenance: EpisodeProvenance,
        summary_code: SummaryCode,
        numeric_features: Vec<f64>,
    ) -> Self {
        Self {
            owner,
            entity,
            tick,
            observed_at,
            revision,
            provenance,
            summary_code,
            numeric_features,
            raw_text: None,
        }
    }

    /// Adjunta texto bruto opcional a la solicitud para validación/redacción según política.
    pub fn with_raw_text(mut self, raw_text: impl Into<String>) -> Self {
        self.raw_text = Some(raw_text.into());
        self
    }
}

/// Registro episódico almacenado en memoria, estructuralmente libre de texto personal bruto.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodicRecord {
    /// Identificador único del episodio.
    pub episode_id: EpisodeId,
    /// Propietario aislado del episodio.
    pub owner: OwnerId,
    /// Entidad observada.
    pub entity: EntityId,
    /// Tick discreto asociado.
    pub tick: Tick,
    /// Instante monotónico de observación.
    pub observed_at: MonotonicInstant,
    /// Revisión monotónica.
    pub revision: u64,
    /// Procedencia estructurada.
    pub provenance: EpisodeProvenance,
    /// Código estructurado de resumen.
    pub summary_code: SummaryCode,
    /// Rasgos numéricos finitos.
    pub numeric_features: Vec<f64>,
    /// Indica si la entrada original contenía texto bruto que fue redactado/descartado.
    pub raw_text_redacted: bool,
}

/// Clave de orden total de un episodio: `(observed_at, tick, revision, episode_id)`.
///
/// El orden derivado (lexicográfico por campos) es idéntico al de la antigua tupla usada por
/// la evicción lineal; `episode_id` desempata y permite recuperar el registro en `records`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct EpisodeSortKey {
    observed_at: MonotonicInstant,
    tick: Tick,
    revision: u64,
    episode_id: EpisodeId,
}

impl EpisodicRecord {
    fn sort_key(&self) -> EpisodeSortKey {
        EpisodeSortKey {
            observed_at: self.observed_at,
            tick: self.tick,
            revision: self.revision,
            episode_id: self.episode_id,
        }
    }

    fn is_expired_at(&self, now: MonotonicInstant, ttl_nanos: u128) -> bool {
        now.as_nanos().saturating_sub(self.observed_at.as_nanos()) >= ttl_nanos
    }
}

/// Contrato de puerto de memoria episódica acotada.
pub trait MemoryPort {
    /// Registra un episodio aplicando la política de texto bruto y los límites de retención.
    fn record_episode(&mut self, input: EpisodeInput) -> Result<EpisodeId, MemoryError>;

    /// Consulta los episodios del `owner` para `entity` en el intervalo cerrado
    /// `[start, end]`, devueltos en orden cronológico determinista.
    fn query_by_time_and_entity(
        &self,
        owner: OwnerId,
        entity: EntityId,
        start: MonotonicInstant,
        end: MonotonicInstant,
    ) -> Result<Vec<EpisodicRecord>, MemoryError>;

    /// Elimina episodios expirados respecto a `now` según la política TTL y devuelve cuántos se eliminaron.
    fn prune_expired(&mut self, now: MonotonicInstant) -> usize;

    /// Devuelve la cantidad de episodios retenidos actualmente para `owner`.
    fn len_for_owner(&self, owner: OwnerId) -> usize;
}

/// Ámbito aislado de memoria para un único individuo/agente (`OwnerId`).
///
/// Garantiza que toda lectura o escritura pase por el `OwnerId` del ámbito y rechaza
/// explícitamente intentos de suplantación con [`MemoryError::ScopeMismatch`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AgentMemoryScope {
    owner: OwnerId,
}

impl AgentMemoryScope {
    /// Crea un ámbito aislado para `owner`.
    pub const fn new(owner: OwnerId) -> Self {
        Self { owner }
    }

    /// Devuelve el [`OwnerId`] propietario de este ámbito.
    pub const fn owner(self) -> OwnerId {
        self.owner
    }

    /// Registra un episodio validando que `input.owner == self.owner`.
    pub fn record_episode<M: MemoryPort + ?Sized>(
        &self,
        memory: &mut M,
        input: EpisodeInput,
    ) -> Result<EpisodeId, MemoryError> {
        if input.owner != self.owner {
            return Err(MemoryError::ScopeMismatch {
                expected: self.owner,
                actual: input.owner,
            });
        }
        memory.record_episode(input)
    }

    /// Consulta episodios del propio `owner` para `entity` en `[start, end]`.
    pub fn query_by_time_and_entity<M: MemoryPort + ?Sized>(
        &self,
        memory: &M,
        entity: EntityId,
        start: MonotonicInstant,
        end: MonotonicInstant,
    ) -> Result<Vec<EpisodicRecord>, MemoryError> {
        memory.query_by_time_and_entity(self.owner, entity, start, end)
    }

    /// Expira episodios obsoletos respecto a `now`.
    pub fn prune_expired<M: MemoryPort + ?Sized>(
        &self,
        memory: &mut M,
        now: MonotonicInstant,
    ) -> usize {
        memory.prune_expired(now)
    }

    /// Devuelve cuántos episodios están almacenados para el propietario de este ámbito.
    pub fn len<M: MemoryPort + ?Sized>(&self, memory: &M) -> usize {
        memory.len_for_owner(self.owner)
    }

    /// Indica si el propietario de este ámbito no tiene episodios almacenados.
    pub fn is_empty<M: MemoryPort + ?Sized>(&self, memory: &M) -> bool {
        self.len(memory) == 0
    }
}

/// Operación determinista aplicable durante el replay de un fixture de memoria.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryOperation {
    /// Registro de un episodio.
    Record(EpisodeInput),
    /// Expiración por TTL en un instante monotónico dado.
    PruneExpired(MonotonicInstant),
}

/// Snapshot serializable y verificable del estado de [`BoundedEpisodicMemory`].
#[derive(Debug, Clone, PartialEq)]
pub struct MemorySnapshot {
    /// Versión de esquema del snapshot.
    pub schema_version: u32,
    /// Política de retención activa.
    pub policy: RetentionPolicy,
    /// Política de texto bruto activa.
    pub raw_text_policy: RawTextPolicy,
    /// Siguiente identificador de episodio a asignar.
    pub next_episode_id: u64,
    /// Episodios en orden canónico determinista.
    pub records: Vec<EpisodicRecord>,
    /// Checksum FNV-1a de integridad del snapshot.
    pub checksum: u64,
}

impl MemorySnapshot {
    /// Verifica la integridad estructural y el checksum del snapshot.
    pub fn verify(&self) -> Result<(), MemoryError> {
        if self.schema_version != MEMORY_SCHEMA_VERSION {
            return Err(MemoryError::InvalidSnapshot {
                reason: "unsupported memory snapshot schema_version",
            });
        }
        let _ = RetentionPolicy::new(
            self.policy.max_episodes_per_owner,
            self.policy.max_total_episodes,
            self.policy.ttl_nanos,
        )?;
        let expected = compute_snapshot_checksum(
            self.schema_version,
            self.policy,
            self.raw_text_policy,
            self.next_episode_id,
            &self.records,
        );
        if self.checksum != expected {
            return Err(MemoryError::InvalidSnapshot {
                reason: "snapshot checksum mismatch",
            });
        }
        if self.records.len() > self.policy.max_total_episodes {
            return Err(MemoryError::InvalidSnapshot {
                reason: "snapshot exceeds max_total_episodes",
            });
        }

        let mut seen_ids = BTreeSet::new();
        let mut owner_counts: BTreeMap<OwnerId, usize> = BTreeMap::new();
        for record in &self.records {
            if record.episode_id.as_u64() >= self.next_episode_id {
                return Err(MemoryError::InvalidSnapshot {
                    reason: "record episode_id >= next_episode_id",
                });
            }
            if !seen_ids.insert(record.episode_id) {
                return Err(MemoryError::InvalidSnapshot {
                    reason: "duplicate episode_id in snapshot",
                });
            }
            validate_features(&record.numeric_features)?;
            let count = owner_counts.entry(record.owner).or_insert(0);
            *count += 1;
            if *count > self.policy.max_episodes_per_owner {
                return Err(MemoryError::InvalidSnapshot {
                    reason: "owner count exceeds max_episodes_per_owner",
                });
            }
        }
        Ok(())
    }
}

/// Implementación local `in-memory` de [`MemoryPort`] con límites estrictos,
/// aislamiento por [`OwnerId`], evicción determinista y soporte de snapshot/replay.
#[derive(Debug, Clone)]
pub struct BoundedEpisodicMemory {
    policy: RetentionPolicy,
    raw_text_policy: RawTextPolicy,
    next_episode_id: u64,
    records: BTreeMap<EpisodeId, EpisodicRecord>,
    /// Índice global de evicción: claves de todos los registros en orden de antigüedad.
    global_order: BTreeSet<EpisodeSortKey>,
    /// Índice de evicción por propietario (sin conjuntos vacíos); su tamaño es el recuento.
    owner_order: BTreeMap<OwnerId, BTreeSet<EpisodeSortKey>>,
}

impl BoundedEpisodicMemory {
    /// Crea una memoria episódica acotada con [`RawTextPolicy::RejectByDefault`].
    pub fn new(policy: RetentionPolicy) -> Self {
        Self::with_raw_text_policy(policy, RawTextPolicy::RejectByDefault)
    }

    /// Crea una memoria episódica acotada especificando explícitamente [`RawTextPolicy`].
    pub fn with_raw_text_policy(policy: RetentionPolicy, raw_text_policy: RawTextPolicy) -> Self {
        Self {
            policy,
            raw_text_policy,
            next_episode_id: 1,
            records: BTreeMap::new(),
            global_order: BTreeSet::new(),
            owner_order: BTreeMap::new(),
        }
    }

    /// Devuelve la política de retención configurada.
    pub const fn policy(&self) -> RetentionPolicy {
        self.policy
    }

    /// Devuelve la política de texto bruto configurada.
    pub const fn raw_text_policy(&self) -> RawTextPolicy {
        self.raw_text_policy
    }

    /// Devuelve el número total de episodios retenidos en todos los propietarios.
    pub fn total_len(&self) -> usize {
        self.records.len()
    }

    /// Devuelve `true` si no hay episodios almacenados.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Captura un [`MemorySnapshot`] determinista y verificable del estado actual.
    pub fn snapshot(&self) -> MemorySnapshot {
        let mut records: Vec<EpisodicRecord> = self.records.values().cloned().collect();
        records.sort_by_key(EpisodicRecord::sort_key);
        let checksum = compute_snapshot_checksum(
            MEMORY_SCHEMA_VERSION,
            self.policy,
            self.raw_text_policy,
            self.next_episode_id,
            &records,
        );
        MemorySnapshot {
            schema_version: MEMORY_SCHEMA_VERSION,
            policy: self.policy,
            raw_text_policy: self.raw_text_policy,
            next_episode_id: self.next_episode_id,
            records,
            checksum,
        }
    }

    /// Reconstruye la memoria a partir de un [`MemorySnapshot`] tras verificar su integridad.
    pub fn restore_from_snapshot(snapshot: &MemorySnapshot) -> Result<Self, MemoryError> {
        snapshot.verify()?;
        let mut memory = Self::with_raw_text_policy(snapshot.policy, snapshot.raw_text_policy);
        memory.next_episode_id = snapshot.next_episode_id;
        for record in &snapshot.records {
            memory.insert_record(record.clone());
        }
        Ok(memory)
    }

    /// Aplica una [`MemoryOperation`] individual sobre la memoria.
    pub fn apply_operation(&mut self, op: &MemoryOperation) -> Result<(), MemoryError> {
        match op {
            MemoryOperation::Record(input) => {
                let _ = self.record_episode(input.clone())?;
                Ok(())
            }
            MemoryOperation::PruneExpired(now) => {
                let _ = self.prune_expired(*now);
                Ok(())
            }
        }
    }

    /// Reconstruye deterministamente una memoria ejecutando una secuencia de operaciones de fixture.
    pub fn replay_operations(
        policy: RetentionPolicy,
        raw_text_policy: RawTextPolicy,
        operations: &[MemoryOperation],
    ) -> Result<Self, MemoryError> {
        let mut memory = Self::with_raw_text_policy(policy, raw_text_policy);
        for op in operations {
            memory.apply_operation(op)?;
        }
        Ok(memory)
    }

    /// Inserta un registro manteniendo sincronizados `records` y los índices de evicción.
    fn insert_record(&mut self, record: EpisodicRecord) {
        let episode_id = record.episode_id;
        // Un id repetido reemplaza al registro previo: retira antes sus claves de índice.
        self.remove_record(episode_id);
        let key = record.sort_key();
        self.global_order.insert(key);
        self.owner_order
            .entry(record.owner)
            .or_default()
            .insert(key);
        self.records.insert(episode_id, record);
    }

    fn remove_record(&mut self, episode_id: EpisodeId) {
        let Some(removed) = self.records.remove(&episode_id) else {
            return;
        };
        let key = removed.sort_key();
        self.global_order.remove(&key);
        if let Some(set) = self.owner_order.get_mut(&removed.owner) {
            set.remove(&key);
            if set.is_empty() {
                self.owner_order.remove(&removed.owner);
            }
        }
    }

    /// Evicción O(log M) del registro más antiguo del propietario mientras exceda su cupo.
    fn evict_oldest_for_owner(&mut self, owner: OwnerId) {
        while self.len_for_owner(owner) > self.policy.max_episodes_per_owner {
            let Some(oldest) = self
                .owner_order
                .get(&owner)
                .and_then(|set| set.first().copied())
            else {
                break;
            };
            self.remove_record(oldest.episode_id);
        }
    }

    /// Evicción O(log M) del registro global más antiguo mientras se exceda el cupo total.
    fn evict_oldest_globally(&mut self) {
        while self.records.len() > self.policy.max_total_episodes {
            let Some(oldest) = self.global_order.first().copied() else {
                break;
            };
            self.remove_record(oldest.episode_id);
        }
    }
}

impl MemoryPort for BoundedEpisodicMemory {
    fn record_episode(&mut self, input: EpisodeInput) -> Result<EpisodeId, MemoryError> {
        let raw_text_redacted = match (self.raw_text_policy, input.raw_text.is_some()) {
            (RawTextPolicy::RejectByDefault, true) => {
                return Err(MemoryError::RawTextRejected {
                    policy: self.raw_text_policy,
                });
            }
            (RawTextPolicy::RedactAndDrop, true) => true,
            (_, false) => false,
        };
        validate_features(&input.numeric_features)?;

        let episode_id = EpisodeId::new(self.next_episode_id);
        self.next_episode_id = self.next_episode_id.saturating_add(1);

        let record = EpisodicRecord {
            episode_id,
            owner: input.owner,
            entity: input.entity,
            tick: input.tick,
            observed_at: input.observed_at,
            revision: input.revision,
            provenance: input.provenance,
            summary_code: input.summary_code,
            numeric_features: input.numeric_features,
            raw_text_redacted,
        };

        self.insert_record(record);

        self.evict_oldest_for_owner(input.owner);
        self.evict_oldest_globally();

        Ok(episode_id)
    }

    fn query_by_time_and_entity(
        &self,
        owner: OwnerId,
        entity: EntityId,
        start: MonotonicInstant,
        end: MonotonicInstant,
    ) -> Result<Vec<EpisodicRecord>, MemoryError> {
        if start > end {
            return Err(MemoryError::InvalidTimeRange { start, end });
        }
        let mut results: Vec<EpisodicRecord> = self
            .records
            .values()
            .filter(|r| {
                r.owner == owner
                    && r.entity == entity
                    && r.observed_at >= start
                    && r.observed_at <= end
            })
            .cloned()
            .collect();
        results.sort_by_key(EpisodicRecord::sort_key);
        Ok(results)
    }

    fn prune_expired(&mut self, now: MonotonicInstant) -> usize {
        let Some(ttl) = self.policy.ttl_nanos else {
            return 0;
        };
        let expired_ids: Vec<EpisodeId> = self
            .records
            .values()
            .filter(|r| r.is_expired_at(now, ttl))
            .map(|r| r.episode_id)
            .collect();
        let removed = expired_ids.len();
        for id in expired_ids {
            self.remove_record(id);
        }
        removed
    }

    fn len_for_owner(&self, owner: OwnerId) -> usize {
        self.owner_order.get(&owner).map_or(0, BTreeSet::len)
    }
}

fn validate_features(features: &[f64]) -> Result<(), MemoryError> {
    for (index, &value) in features.iter().enumerate() {
        if !value.is_finite() {
            return Err(MemoryError::NonFiniteFeature { index, value });
        }
    }
    Ok(())
}

fn fnv1a_update(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0f64.to_bits()
    } else {
        value.to_bits()
    }
}

fn compute_snapshot_checksum(
    schema_version: u32,
    policy: RetentionPolicy,
    raw_text_policy: RawTextPolicy,
    next_episode_id: u64,
    records: &[EpisodicRecord],
) -> u64 {
    let mut hash = FNV_OFFSET_BASIS;
    fnv1a_update(&mut hash, &schema_version.to_le_bytes());
    fnv1a_update(
        &mut hash,
        &(policy.max_episodes_per_owner as u64).to_le_bytes(),
    );
    fnv1a_update(&mut hash, &(policy.max_total_episodes as u64).to_le_bytes());
    match policy.ttl_nanos {
        Some(ttl) => {
            fnv1a_update(&mut hash, &[1]);
            fnv1a_update(&mut hash, &ttl.to_le_bytes());
        }
        None => fnv1a_update(&mut hash, &[0]),
    }
    fnv1a_update(&mut hash, &[raw_text_policy.as_u8()]);
    fnv1a_update(&mut hash, &next_episode_id.to_le_bytes());
    fnv1a_update(&mut hash, &(records.len() as u64).to_le_bytes());

    for r in records {
        fnv1a_update(&mut hash, &r.episode_id.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.owner.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.entity.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.tick.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.observed_at.as_nanos().to_le_bytes());
        fnv1a_update(&mut hash, &r.revision.to_le_bytes());
        fnv1a_update(&mut hash, &r.provenance.evidence_ref.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.provenance.message_id.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &r.provenance.sensor_id.as_u64().to_le_bytes());
        let kind_tag = match r.provenance.claim_kind {
            ObservationKind::Measurement => 0u8,
            ObservationKind::Assertion => 1u8,
            ObservationKind::Intention => 2u8,
            ObservationKind::Proposal => 3u8,
            ObservationKind::Unknown => 4u8,
        };
        fnv1a_update(&mut hash, &[kind_tag]);
        fnv1a_update(
            &mut hash,
            &canonical_f64_bits(r.provenance.confidence.as_f64()).to_le_bytes(),
        );
        fnv1a_update(&mut hash, &r.summary_code.as_u64().to_le_bytes());
        fnv1a_update(&mut hash, &(r.numeric_features.len() as u64).to_le_bytes());
        for &feat in &r.numeric_features {
            fnv1a_update(&mut hash, &canonical_f64_bits(feat).to_le_bytes());
        }
        fnv1a_update(&mut hash, &[u8::from(r.raw_text_redacted)]);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::{FrameId, UnitTag};

    fn sample_provenance(evidence: u64, msg: u64, sensor: u64) -> EpisodeProvenance {
        let full_meta = ProvenanceMetadata {
            message_id: MessageId::new(msg),
            observed_at: MonotonicInstant::from_nanos(100),
            received_at: MonotonicInstant::from_nanos(105),
            sensor_id: SensorId::new(sensor),
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence: Confidence::new(0.85).unwrap(),
            revision: 1,
            evidence_ref: EvidenceRef::new(evidence),
            claim_kind: ObservationKind::Measurement,
        };
        EpisodeProvenance::from(&full_meta)
    }

    #[test]
    fn ac1_query_by_time_and_entity_returns_episode_with_provenance() {
        let policy = RetentionPolicy::new(8, 16, None).unwrap();
        let mut memory = BoundedEpisodicMemory::new(policy);

        let owner = OwnerId::new(1);
        let entity_a = EntityId::new(42);
        let entity_b = EntityId::new(99);

        let prov1 = sample_provenance(1001, 10, 3);
        let prov2 = sample_provenance(1002, 11, 3);

        let id1 = memory
            .record_episode(EpisodeInput::new(
                owner,
                entity_a,
                Tick::new(5),
                MonotonicInstant::from_nanos(500),
                2,
                prov1,
                SummaryCode::new(77),
                vec![1.5, -2.0],
            ))
            .unwrap();
        let _other_entity = memory
            .record_episode(EpisodeInput::new(
                owner,
                entity_b,
                Tick::new(6),
                MonotonicInstant::from_nanos(600),
                3,
                sample_provenance(9001, 12, 4),
                SummaryCode::new(88),
                vec![9.0],
            ))
            .unwrap();
        let id2 = memory
            .record_episode(EpisodeInput::new(
                owner,
                entity_a,
                Tick::new(2),
                MonotonicInstant::from_nanos(200),
                1,
                prov2,
                SummaryCode::new(76),
                vec![0.5],
            ))
            .unwrap();

        // Consulta en [150, 550] para entity_a devuelve en orden cronológico determinista.
        let results = memory
            .query_by_time_and_entity(
                owner,
                entity_a,
                MonotonicInstant::from_nanos(150),
                MonotonicInstant::from_nanos(550),
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].episode_id, id2);
        assert_eq!(results[0].provenance, prov2);
        assert_eq!(results[0].provenance.evidence_ref, EvidenceRef::new(1002));
        assert_eq!(results[0].summary_code, SummaryCode::new(76));
        assert!(!results[0].raw_text_redacted);

        assert_eq!(results[1].episode_id, id1);
        assert_eq!(results[1].provenance, prov1);
        assert_eq!(results[1].provenance.evidence_ref, EvidenceRef::new(1001));
        assert_eq!(results[1].numeric_features, vec![1.5, -2.0]);

        // Rango invertido se rechaza explícitamente.
        let err = memory
            .query_by_time_and_entity(
                owner,
                entity_a,
                MonotonicInstant::from_nanos(600),
                MonotonicInstant::from_nanos(100),
            )
            .unwrap_err();
        assert!(matches!(err, MemoryError::InvalidTimeRange { .. }));
    }

    #[test]
    fn ac2_retention_and_expiration_respect_bounds_and_reject_raw_text_by_default() {
        let policy = RetentionPolicy::new(2, 3, Some(100)).unwrap();
        let mut memory = BoundedEpisodicMemory::new(policy);
        let owner1 = OwnerId::new(1);
        let owner2 = OwnerId::new(2);
        let entity = EntityId::new(10);

        // Por defecto rechaza texto personal bruto sin mutar la memoria.
        let raw_attempt = EpisodeInput::new(
            owner1,
            entity,
            Tick::new(1),
            MonotonicInstant::from_nanos(10),
            1,
            EpisodeProvenance::from_evidence(EvidenceRef::new(1)),
            SummaryCode::new(1),
            vec![1.0],
        )
        .with_raw_text("Juan Perez DNI 12345678A");

        assert_eq!(
            memory.record_episode(raw_attempt),
            Err(MemoryError::RawTextRejected {
                policy: RawTextPolicy::RejectByDefault
            })
        );
        assert_eq!(memory.total_len(), 0);

        // Rechaza rasgos no finitos sin mutar la memoria.
        let nan_attempt = EpisodeInput::new(
            owner1,
            entity,
            Tick::new(1),
            MonotonicInstant::from_nanos(10),
            1,
            EpisodeProvenance::from_evidence(EvidenceRef::new(1)),
            SummaryCode::new(1),
            vec![1.0, f64::NAN],
        );
        assert!(matches!(
            memory.record_episode(nan_attempt),
            Err(MemoryError::NonFiniteFeature { index: 1, .. })
        ));
        assert_eq!(memory.total_len(), 0);

        // Inserta 3 episodios para owner1 -> max_episodes_per_owner = 2 evicta el más antiguo (t=10).
        for (tick, ts) in [(1, 10), (2, 20), (3, 30)] {
            memory
                .record_episode(EpisodeInput::new(
                    owner1,
                    entity,
                    Tick::new(tick),
                    MonotonicInstant::from_nanos(ts),
                    tick,
                    EpisodeProvenance::from_evidence(EvidenceRef::new(tick)),
                    SummaryCode::new(tick),
                    vec![tick as f64],
                ))
                .unwrap();
        }
        assert_eq!(memory.len_for_owner(owner1), 2);
        let owner1_records = memory
            .query_by_time_and_entity(
                owner1,
                entity,
                MonotonicInstant::ZERO,
                MonotonicInstant::from_nanos(1000),
            )
            .unwrap();
        assert_eq!(owner1_records.len(), 2);
        assert_eq!(
            owner1_records[0].observed_at,
            MonotonicInstant::from_nanos(20)
        );
        assert_eq!(
            owner1_records[1].observed_at,
            MonotonicInstant::from_nanos(30)
        );

        // Inserta 2 episodios para owner2 (t=25, t=40) -> max_total_episodes = 3 evicta t=20 (owner1).
        for (tick, ts) in [(4, 25), (5, 40)] {
            memory
                .record_episode(EpisodeInput::new(
                    owner2,
                    entity,
                    Tick::new(tick),
                    MonotonicInstant::from_nanos(ts),
                    tick,
                    EpisodeProvenance::from_evidence(EvidenceRef::new(tick)),
                    SummaryCode::new(tick),
                    vec![tick as f64],
                ))
                .unwrap();
        }
        assert_eq!(memory.total_len(), 3);
        assert_eq!(memory.len_for_owner(owner1), 1);
        assert_eq!(memory.len_for_owner(owner2), 2);

        // Expiración TTL = 100: en now=125 expira t=25 (125-25=100 >= 100), permanecen t=30 y t=40.
        let pruned = memory.prune_expired(MonotonicInstant::from_nanos(125));
        assert_eq!(pruned, 1);
        assert_eq!(memory.total_len(), 2);
        assert_eq!(memory.len_for_owner(owner1), 1);
        assert_eq!(memory.len_for_owner(owner2), 1);

        // Política explícita RedactAndDrop descarta el texto sin almacenarlo.
        let mut redacting = BoundedEpisodicMemory::with_raw_text_policy(
            RetentionPolicy::new(4, 8, None).unwrap(),
            RawTextPolicy::RedactAndDrop,
        );
        let redacted_id = redacting
            .record_episode(
                EpisodeInput::new(
                    owner1,
                    entity,
                    Tick::new(1),
                    MonotonicInstant::from_nanos(50),
                    1,
                    EpisodeProvenance::from_evidence(EvidenceRef::new(7)),
                    SummaryCode::new(9),
                    vec![3.25],
                )
                .with_raw_text("dato personal sensible"),
            )
            .unwrap();
        let stored = redacting
            .query_by_time_and_entity(
                owner1,
                entity,
                MonotonicInstant::ZERO,
                MonotonicInstant::from_nanos(100),
            )
            .unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].episode_id, redacted_id);
        assert!(stored[0].raw_text_redacted);
    }

    #[test]
    fn ac3_owner_isolation_and_deterministic_snapshot_replay_reconstruct_fixture() {
        let policy = RetentionPolicy::new(3, 5, Some(500)).unwrap();
        let alice = OwnerId::new(101);
        let bob = OwnerId::new(202);
        let alice_scope = AgentMemoryScope::new(alice);
        let bob_scope = AgentMemoryScope::new(bob);
        let shared_entity = EntityId::new(7);

        let ops = vec![
            MemoryOperation::Record(EpisodeInput::new(
                alice,
                shared_entity,
                Tick::new(1),
                MonotonicInstant::from_nanos(100),
                1,
                sample_provenance(1, 1, 1),
                SummaryCode::new(10),
                vec![1.0, 2.0],
            )),
            MemoryOperation::Record(EpisodeInput::new(
                bob,
                shared_entity,
                Tick::new(2),
                MonotonicInstant::from_nanos(150),
                1,
                sample_provenance(2, 2, 2),
                SummaryCode::new(20),
                vec![3.0],
            )),
            MemoryOperation::Record(EpisodeInput::new(
                alice,
                shared_entity,
                Tick::new(3),
                MonotonicInstant::from_nanos(200),
                2,
                sample_provenance(3, 3, 1),
                SummaryCode::new(11),
                vec![4.0],
            )),
            MemoryOperation::PruneExpired(MonotonicInstant::from_nanos(600)),
            MemoryOperation::Record(EpisodeInput::new(
                bob,
                shared_entity,
                Tick::new(4),
                MonotonicInstant::from_nanos(650),
                2,
                sample_provenance(4, 4, 2),
                SummaryCode::new(21),
                vec![5.0],
            )),
        ];

        let mut mem_a = BoundedEpisodicMemory::new(policy);
        for op in &ops {
            mem_a.apply_operation(op).unwrap();
        }

        // Aislamiento estricto: el ámbito de Alice no puede escribir para Bob ni leer los episodios de Bob.
        let spoof = EpisodeInput::new(
            bob,
            shared_entity,
            Tick::new(9),
            MonotonicInstant::from_nanos(700),
            9,
            sample_provenance(9, 9, 9),
            SummaryCode::new(99),
            vec![0.0],
        );
        assert_eq!(
            alice_scope.record_episode(&mut mem_a, spoof),
            Err(MemoryError::ScopeMismatch {
                expected: alice,
                actual: bob
            })
        );

        let alice_view = alice_scope
            .query_by_time_and_entity(
                &mem_a,
                shared_entity,
                MonotonicInstant::ZERO,
                MonotonicInstant::from_nanos(1000),
            )
            .unwrap();
        let bob_view = bob_scope
            .query_by_time_and_entity(
                &mem_a,
                shared_entity,
                MonotonicInstant::ZERO,
                MonotonicInstant::from_nanos(1000),
            )
            .unwrap();

        assert_eq!(alice_view.len(), 1);
        assert!(alice_view.iter().all(|r| r.owner == alice));
        assert_eq!(alice_view[0].summary_code, SummaryCode::new(11));

        assert_eq!(bob_view.len(), 2);
        assert!(bob_view.iter().all(|r| r.owner == bob));

        // Replay determinista del fixture produce exactamente el mismo snapshot y checksum.
        let mem_replayed =
            BoundedEpisodicMemory::replay_operations(policy, RawTextPolicy::RejectByDefault, &ops)
                .unwrap();
        let snap_a = mem_a.snapshot();
        let snap_replayed = mem_replayed.snapshot();
        assert_eq!(snap_a, snap_replayed);

        // Restauración desde snapshot reconstruye exactamente el estado.
        let restored = BoundedEpisodicMemory::restore_from_snapshot(&snap_a).unwrap();
        assert_eq!(restored.snapshot(), snap_a);

        // Checksum corrupto se rechaza.
        let mut corrupted = snap_a.clone();
        corrupted.checksum ^= 0xDEAD;
        assert!(matches!(
            BoundedEpisodicMemory::restore_from_snapshot(&corrupted),
            Err(MemoryError::InvalidSnapshot { .. })
        ));
    }

    // ---- GOG-128: índices secundarios de evicción ----

    use crate::model::{DeterministicRng, RngSource};

    impl BoundedEpisodicMemory {
        /// Verifica la coherencia entre `records`, `global_order` y `owner_order`.
        fn assert_index_invariants(&self) {
            assert_eq!(self.global_order.len(), self.records.len());
            let mut owner_total = 0;
            for (owner, set) in &self.owner_order {
                assert!(!set.is_empty(), "conjunto vacío para {owner:?}");
                owner_total += set.len();
                for key in set {
                    let rec = self.records.get(&key.episode_id).expect("clave huérfana");
                    assert_eq!(rec.owner, *owner);
                    assert_eq!(rec.sort_key(), *key);
                    assert!(self.global_order.contains(key));
                }
            }
            assert_eq!(owner_total, self.records.len());
            for (id, rec) in &self.records {
                assert_eq!(*id, rec.episode_id);
                assert!(self.global_order.contains(&rec.sort_key()));
            }
        }
    }

    /// Implementación de referencia (solo tests) con la evicción lineal original.
    struct LinearRef {
        policy: RetentionPolicy,
        next_id: u64,
        records: BTreeMap<EpisodeId, EpisodicRecord>,
    }

    impl LinearRef {
        fn new(policy: RetentionPolicy) -> Self {
            Self {
                policy,
                next_id: 1,
                records: BTreeMap::new(),
            }
        }

        fn record(&mut self, input: EpisodeInput) -> EpisodeId {
            let id = EpisodeId::new(self.next_id);
            self.next_id += 1;
            let owner = input.owner;
            self.records.insert(
                id,
                EpisodicRecord {
                    episode_id: id,
                    owner,
                    entity: input.entity,
                    tick: input.tick,
                    observed_at: input.observed_at,
                    revision: input.revision,
                    provenance: input.provenance,
                    summary_code: input.summary_code,
                    numeric_features: input.numeric_features,
                    raw_text_redacted: false,
                },
            );
            while self.records.values().filter(|r| r.owner == owner).count()
                > self.policy.max_episodes_per_owner
            {
                let id = self
                    .records
                    .values()
                    .filter(|r| r.owner == owner)
                    .min_by_key(|r| r.sort_key())
                    .map(|r| r.episode_id)
                    .unwrap();
                self.records.remove(&id);
            }
            while self.records.len() > self.policy.max_total_episodes {
                let id = self
                    .records
                    .values()
                    .min_by_key(|r| r.sort_key())
                    .map(|r| r.episode_id)
                    .unwrap();
                self.records.remove(&id);
            }
            id
        }

        fn prune(&mut self, now: MonotonicInstant) -> usize {
            let Some(ttl) = self.policy.ttl_nanos else {
                return 0;
            };
            let ids: Vec<_> = self
                .records
                .values()
                .filter(|r| r.is_expired_at(now, ttl))
                .map(|r| r.episode_id)
                .collect();
            for id in &ids {
                self.records.remove(id);
            }
            ids.len()
        }

        fn snapshot(&self) -> MemorySnapshot {
            let mut records: Vec<_> = self.records.values().cloned().collect();
            records.sort_by_key(EpisodicRecord::sort_key);
            let checksum = compute_snapshot_checksum(
                MEMORY_SCHEMA_VERSION,
                self.policy,
                RawTextPolicy::RejectByDefault,
                self.next_id,
                &records,
            );
            MemorySnapshot {
                schema_version: MEMORY_SCHEMA_VERSION,
                policy: self.policy,
                raw_text_policy: RawTextPolicy::RejectByDefault,
                next_episode_id: self.next_id,
                records,
                checksum,
            }
        }
    }

    fn random_input(rng: &mut DeterministicRng, owners: u64, span: u64) -> EpisodeInput {
        // Rangos pequeños para provocar empates en observed_at/tick/revision.
        EpisodeInput::new(
            OwnerId::new(rng.next_u64() % owners),
            EntityId::new(rng.next_u64() % 3),
            Tick::new(rng.next_u64() % 4),
            MonotonicInstant::from_nanos(u128::from(rng.next_u64() % span)),
            rng.next_u64() % 3,
            sample_provenance(1, 1, 1),
            SummaryCode::new(rng.next_u64() % 100),
            vec![(rng.next_u64() % 10) as f64],
        )
    }

    #[test]
    fn gog128_ac2_eviction_matches_linear_reference_on_seeded_workloads() {
        for seed in [1u64, 7, 42, 0xDEAD_BEEF, 2026] {
            let policy = RetentionPolicy::new(5, 17, Some(60)).unwrap();
            let mut real = BoundedEpisodicMemory::new(policy);
            let mut reference = LinearRef::new(policy);
            let mut rng = DeterministicRng::new(seed);
            for step in 0..1500u64 {
                if step % 50 == 49 {
                    let now = MonotonicInstant::from_nanos(u128::from(rng.next_u64() % 120));
                    assert_eq!(real.prune_expired(now), reference.prune(now));
                } else {
                    let input = random_input(&mut rng, 4, 80);
                    let a = real.record_episode(input.clone()).unwrap();
                    let b = reference.record(input);
                    assert_eq!(a, b);
                }
                // Misma secuencia de supervivientes tras cada paso => misma secuencia de evicción.
                assert!(
                    real.records.keys().eq(reference.records.keys()),
                    "seed {seed} step {step}"
                );
                if step % 100 == 0 {
                    assert_eq!(real.snapshot(), reference.snapshot());
                }
            }
            let (rs, fs) = (real.snapshot(), reference.snapshot());
            assert_eq!(rs.checksum, fs.checksum);
            assert_eq!(rs, fs);
            // Consultas por rango idénticas al filtrado lineal de referencia.
            for owner in 0..4 {
                let (s, e) = (
                    MonotonicInstant::from_nanos(10),
                    MonotonicInstant::from_nanos(60),
                );
                let got = real
                    .query_by_time_and_entity(OwnerId::new(owner), EntityId::new(1), s, e)
                    .unwrap();
                let mut want: Vec<_> = reference
                    .records
                    .values()
                    .filter(|r| {
                        r.owner == OwnerId::new(owner)
                            && r.entity == EntityId::new(1)
                            && r.observed_at >= s
                            && r.observed_at <= e
                    })
                    .cloned()
                    .collect();
                want.sort_by_key(EpisodicRecord::sort_key);
                assert_eq!(got, want);
            }
        }
    }

    #[test]
    fn gog128_ac3_stress_indices_stay_consistent() {
        let policy = RetentionPolicy::new(40, 300, Some(500)).unwrap();
        let mut mem = BoundedEpisodicMemory::new(policy);
        let mut rng = DeterministicRng::new(0x600D_5EED);
        let mut ids: Vec<EpisodeId> = Vec::new();
        for step in 0..6000u64 {
            match rng.next_u64() % 40 {
                0 => {
                    let now = MonotonicInstant::from_nanos(u128::from(rng.next_u64() % 1200));
                    mem.prune_expired(now);
                }
                1 | 2 => {
                    if !ids.is_empty() {
                        let idx = (rng.next_u64() as usize) % ids.len();
                        mem.remove_record(ids.swap_remove(idx));
                    }
                }
                3 => {
                    let snap = mem.snapshot();
                    mem = BoundedEpisodicMemory::restore_from_snapshot(&snap).unwrap();
                    assert_eq!(mem.snapshot(), snap);
                }
                _ => {
                    let input = random_input(&mut rng, 12, 1000);
                    ids.push(mem.record_episode(input).unwrap());
                }
            }
            assert!(mem.total_len() <= 300);
            if step % 20 == 0 {
                mem.assert_index_invariants();
            }
        }
        mem.assert_index_invariants();
        for (owner, set) in &mem.owner_order {
            assert!(set.len() <= 40);
            assert_eq!(mem.len_for_owner(*owner), set.len());
        }
    }

    #[test]
    fn gog128_ac1_evictions_pop_first_from_indices() {
        // Con M grande y cupo pequeño cada inserción evicta exactamente el primero del índice.
        let policy = RetentionPolicy::new(3, 2000, None).unwrap();
        let mut mem = BoundedEpisodicMemory::new(policy);
        let mut rng = DeterministicRng::new(5);
        for _ in 0..4000 {
            let input = random_input(&mut rng, 1000, 1_000_000);
            let owner = input.owner;
            let before = mem.owner_order.get(&owner).and_then(|s| s.first().copied());
            let full = mem.len_for_owner(owner) == 3;
            let id = mem.record_episode(input).unwrap();
            if full {
                let victim = before.unwrap();
                // La víctima es el mínimo previo; si el nuevo es aún más antiguo, se evicta el nuevo.
                match mem.records.get(&id).map(EpisodicRecord::sort_key) {
                    Some(k) => {
                        assert!(k > victim);
                        assert!(!mem.records.contains_key(&victim.episode_id));
                    }
                    None => assert!(mem.records.contains_key(&victim.episode_id)),
                }
            }
            assert!(mem.total_len() <= 2000);
        }
        mem.assert_index_invariants();
    }
}
