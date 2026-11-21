//! Traza científica durable y snapshots de replay (`std`-only).
//!
//! Este módulo implementa la persistencia científica determinista separada de los
//! frames visuales descartables ([`gogamen_core::VisualFrame`]):
//! - Formato binario versionado ([`SCIENTIFIC_TRACE_VERSION`], [`TraceManifest`])
//!   con `run_id`, `build_version`, `spec_id`, `model_id`, `seed`, estado de
//!   completitud (`is_complete`), contadores de registros, huecos ([`TraceGap`])
//!   y checksum FNV-1a.
//! - Registros científicos ([`ScientificRecord`]) con [`TickInputs`],
//!   [`DecisionAction`], [`StreamSeeds`], [`StreamRevision`] y
//!   [`CheckpointRef`] sobre [`SimulationSnapshot`], con serialización y
//!   deserialización determinista bit-a-bit en `std` puro.
//! - Política de campos sensibles ([`SensitiveFieldPolicy`]) que rechaza o
//!   redacta credenciales y secretos antes de persistir checkpoints o manifiestos.
//! - Escritura acotada fuera del bucle de control crítico mediante
//!   [`BufferedTraceSink`] y [`AsyncTraceWriter`], con cuotas ([`TraceQuota`]),
//!   política explícita ante saturación o fallo de disco
//!   ([`OverflowPolicy::MarkIncomplete`] | [`OverflowPolicy::AbortExperiment`])
//!   y soporte de simulación de fallos ([`TraceWriteError::DiskFailure`]).

use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use gogamen_core::{
    Action, DeterminismMetadata, DiscreteAction, GuardDecision, MasterSeed, Observation,
    PlanStatus, RunId, SimDuration, SimulationSnapshot, StreamRevision, Terminal, TerminalReason,
    Tick, TraceHasher,
};

use crate::trace::TickTraceEntry;

/// Versión actual del formato binario de traza científica.
pub const SCIENTIFIC_TRACE_VERSION: u32 = 1;

const TRACE_MAGIC: &[u8; 8] = b"GOGTRACE";
const MANIFEST_MAGIC: &[u8; 8] = b"GOGMANIF";
const ARCHIVE_MAGIC: &[u8; 8] = b"GOGARCHV";
const REDACTED_VALUE: &str = "[REDACTED]";

/// Política de manejo de campos sensibles en checkpoints y metadatos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SensitiveFieldMode {
    /// Rechaza el registro o manifiesto si detecta un campo o valor sensible.
    Reject,
    /// Redacta el valor sensible sustituyéndolo por `"[REDACTED]"`.
    Redact,
}

/// Política configurable para impedir que credenciales o secretos entren en
/// checkpoints científicos o manifiestos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensitiveFieldPolicy {
    mode: SensitiveFieldMode,
    forbidden_key_patterns: Vec<String>,
    secret_value_prefixes: Vec<String>,
}

impl SensitiveFieldPolicy {
    /// Crea una política con el modo indicado y los patrones por defecto.
    pub fn new(mode: SensitiveFieldMode) -> Self {
        Self {
            mode,
            forbidden_key_patterns: default_forbidden_key_patterns(),
            secret_value_prefixes: default_secret_value_prefixes(),
        }
    }

    /// Devuelve el modo de la política ([`SensitiveFieldMode::Reject`] o [`SensitiveFieldMode::Redact`]).
    pub const fn mode(&self) -> SensitiveFieldMode {
        self.mode
    }

    /// Añade un patrón de clave prohibida (comparación insensible a mayúsculas).
    pub fn with_forbidden_key_pattern(mut self, pattern: impl Into<String>) -> Self {
        self.forbidden_key_patterns
            .push(pattern.into().to_ascii_lowercase());
        self
    }

    /// Añade un prefijo de valor considerado secreto (p. ej. `"sk-"`).
    pub fn with_secret_value_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.secret_value_prefixes.push(prefix.into());
        self
    }

    /// Indica si una clave coincide con algún patrón sensible.
    pub fn is_sensitive_key(&self, key: &str) -> bool {
        let lower = key.to_ascii_lowercase();
        self.forbidden_key_patterns
            .iter()
            .any(|pattern| lower.contains(pattern))
    }

    /// Indica si un valor coincide con algún prefijo de secreto conocido.
    pub fn is_sensitive_value(&self, value: &str) -> bool {
        let trimmed = value.trim_start();
        self.secret_value_prefixes
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
    }

    /// Valida o redacta un texto simple asociado a un campo nombrado.
    pub fn sanitize_field(&self, field: &str, value: &str) -> Result<String, ScientificTraceError> {
        if self.is_sensitive_key(field) || self.is_sensitive_value(value) {
            match self.mode {
                SensitiveFieldMode::Reject => Err(ScientificTraceError::SensitiveDataDetected {
                    field: field.to_owned(),
                }),
                SensitiveFieldMode::Redact => Ok(REDACTED_VALUE.to_owned()),
            }
        } else {
            Ok(value.to_owned())
        }
    }

    /// Valida o redacta un mapa de anotaciones clave-valor (por ejemplo en [`CheckpointRef`]).
    pub fn sanitize_map(
        &self,
        annotations: &BTreeMap<String, String>,
    ) -> Result<BTreeMap<String, String>, ScientificTraceError> {
        let mut sanitized = BTreeMap::new();
        for (key, value) in annotations {
            let is_sensitive = self.is_sensitive_key(key) || self.is_sensitive_value(value);
            if is_sensitive {
                match self.mode {
                    SensitiveFieldMode::Reject => {
                        return Err(ScientificTraceError::SensitiveDataDetected {
                            field: key.clone(),
                        });
                    }
                    SensitiveFieldMode::Redact => {
                        sanitized.insert(key.clone(), REDACTED_VALUE.to_owned());
                    }
                }
            } else {
                sanitized.insert(key.clone(), value.clone());
            }
        }
        Ok(sanitized)
    }
}

impl Default for SensitiveFieldPolicy {
    fn default() -> Self {
        Self::new(SensitiveFieldMode::Reject)
    }
}

fn default_forbidden_key_patterns() -> Vec<String> {
    [
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "credential",
        "private_key",
        "auth",
        "bearer",
        "jwt",
        "session_id",
        "cookie",
        "access_key",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn default_secret_value_prefixes() -> Vec<String> {
    ["sk-", "ghp_", "gho_", "Bearer ", "AKIA", "xoxb-"]
        .into_iter()
        .map(str::to_owned)
        .collect()
}

/// Motivo explícito de un hueco ([`TraceGap`]) en la traza científica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceGapReason {
    /// La cola acotada en memoria alcanzó su límite de registros o bytes.
    QueueOverflow,
    /// Se superó la cuota total configurada para la traza.
    QuotaExceeded,
    /// Se produjo un fallo de escritura en el almacenamiento durable.
    DiskFailure(String),
}

impl fmt::Display for TraceGapReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueOverflow => write!(f, "queue overflow"),
            Self::QuotaExceeded => write!(f, "quota exceeded"),
            Self::DiskFailure(msg) => write!(f, "disk failure: {msg}"),
        }
    }
}

/// Hueco explícito registrado cuando faltan registros en una traza incompleta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceGap {
    /// Primera revisión faltante o afectada.
    pub from_revision: StreamRevision,
    /// Última revisión faltante o afectada (inclusiva).
    pub to_revision: StreamRevision,
    /// Número de registros perdidos en este intervalo.
    pub dropped_count: u64,
    /// Motivo de la pérdida.
    pub reason: TraceGapReason,
}

/// Configuración de cabecera para crear un [`TraceManifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceManifestConfig {
    /// Identificador único de la ejecución.
    pub run_id: RunId,
    /// Versión de compilación del binario/crate.
    pub build_version: String,
    /// Identificador de especificación del experimento.
    pub spec_id: String,
    /// Identificador del modelo evaluado.
    pub model_id: String,
    /// Semilla maestra determinista de la ejecución.
    pub seed: MasterSeed,
}

impl TraceManifestConfig {
    /// Crea una configuración tomando `build_version` desde [`DeterminismMetadata::current`].
    pub fn new(
        run_id: RunId,
        spec_id: impl Into<String>,
        model_id: impl Into<String>,
        seed: MasterSeed,
    ) -> Self {
        let meta = DeterminismMetadata::current(seed);
        Self {
            run_id,
            build_version: meta.build_version().to_owned(),
            spec_id: spec_id.into(),
            model_id: model_id.into(),
            seed,
        }
    }
}

/// Manifiesto versionado de una traza científica durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceManifest {
    /// Versión del formato de traza ([`SCIENTIFIC_TRACE_VERSION`]).
    pub version: u32,
    /// Identificador único de la ejecución.
    pub run_id: RunId,
    /// Versión de compilación registrada.
    pub build_version: String,
    /// Identificador de especificación del experimento.
    pub spec_id: String,
    /// Identificador del modelo utilizado.
    pub model_id: String,
    /// Semilla maestra de la ejecución.
    pub seed: MasterSeed,
    /// Indica si la traza está completa sin pérdidas ni fallos (`false` ante cualquier hueco).
    pub is_complete: bool,
    /// Total de registros efectivamente persistidos.
    pub total_records: u64,
    /// Total de registros descartados por saturación, cuota o fallo de disco.
    pub dropped_records: u64,
    /// Lista cronológica de huecos registrados.
    pub gaps: Vec<TraceGap>,
    /// Checksum FNV-1a de integridad del manifiesto.
    pub checksum: u64,
}

impl TraceManifest {
    /// Construye un manifiesto inicial completo (0 registros, 0 pérdidas) con checksum calculado.
    pub fn new(config: TraceManifestConfig) -> Self {
        let mut manifest = Self {
            version: SCIENTIFIC_TRACE_VERSION,
            run_id: config.run_id,
            build_version: config.build_version,
            spec_id: config.spec_id,
            model_id: config.model_id,
            seed: config.seed,
            is_complete: true,
            total_records: 0,
            dropped_records: 0,
            gaps: Vec::new(),
            checksum: 0,
        };
        manifest.recompute_checksum();
        manifest
    }

    /// Calcula el checksum FNV-1a canónico sobre todos los campos del manifiesto.
    pub fn compute_checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_u64(u64::from(self.version));
        hasher.write_u64(self.run_id.as_u64());
        hasher.write_str(&self.build_version);
        hasher.write_str(&self.spec_id);
        hasher.write_str(&self.model_id);
        hasher.write_u64(self.seed.as_u64());
        hasher.write_u8(u8::from(self.is_complete));
        hasher.write_u64(self.total_records);
        hasher.write_u64(self.dropped_records);
        hasher.write_u64(self.gaps.len() as u64);
        for gap in &self.gaps {
            hasher.write_u64(gap.from_revision.as_u64());
            hasher.write_u64(gap.to_revision.as_u64());
            hasher.write_u64(gap.dropped_count);
            match &gap.reason {
                TraceGapReason::QueueOverflow => hasher.write_u8(0),
                TraceGapReason::QuotaExceeded => hasher.write_u8(1),
                TraceGapReason::DiskFailure(msg) => {
                    hasher.write_u8(2);
                    hasher.write_str(msg);
                }
            }
        }
        hasher.finish()
    }

    /// Recalcula y actualiza `self.checksum`.
    pub fn recompute_checksum(&mut self) {
        #[cfg(test)]
        tests::RECOMPUTE_COUNT.with(|c| c.set(c.get() + 1));
        self.checksum = self.compute_checksum();
    }

    /// Verifica versión de formato, invariante de completitud y checksum del manifiesto.
    pub fn verify(&self) -> Result<(), ScientificTraceError> {
        if self.version != SCIENTIFIC_TRACE_VERSION {
            return Err(ScientificTraceError::UnsupportedVersion {
                expected: SCIENTIFIC_TRACE_VERSION,
                actual: self.version,
            });
        }
        if self.dropped_records > 0 && self.is_complete {
            return Err(ScientificTraceError::InconsistentCompleteness);
        }
        if !self.gaps.is_empty() && self.is_complete {
            return Err(ScientificTraceError::InconsistentCompleteness);
        }
        let actual = self.compute_checksum();
        if self.checksum != actual {
            return Err(ScientificTraceError::ChecksumMismatch {
                expected: self.checksum,
                actual,
            });
        }
        Ok(())
    }

    /// Registra un hueco en el manifiesto, forzando `is_complete = false`.
    pub fn record_gap(
        &mut self,
        from_revision: StreamRevision,
        to_revision: StreamRevision,
        dropped_count: u64,
        reason: TraceGapReason,
    ) {
        if dropped_count == 0 {
            return;
        }
        self.is_complete = false;
        self.dropped_records = self.dropped_records.saturating_add(dropped_count);
        if let Some(last) = self.gaps.last_mut()
            && last.reason == reason
            && to_revision >= last.to_revision
            && from_revision.as_u64() <= last.to_revision.as_u64().saturating_add(1)
        {
            last.to_revision = to_revision;
            last.dropped_count = last.dropped_count.saturating_add(dropped_count);
        } else {
            self.gaps.push(TraceGap {
                from_revision,
                to_revision,
                dropped_count,
                reason,
            });
        }
        self.recompute_checksum();
    }

    /// Serializa el manifiesto en formato binario determinista.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MANIFEST_MAGIC);
        write_u32(&mut out, self.version);
        write_u64(&mut out, self.run_id.as_u64());
        write_string(&mut out, &self.build_version);
        write_string(&mut out, &self.spec_id);
        write_string(&mut out, &self.model_id);
        write_u64(&mut out, self.seed.as_u64());
        write_u8(&mut out, u8::from(self.is_complete));
        write_u64(&mut out, self.total_records);
        write_u64(&mut out, self.dropped_records);
        write_u64(&mut out, self.gaps.len() as u64);
        for gap in &self.gaps {
            write_u64(&mut out, gap.from_revision.as_u64());
            write_u64(&mut out, gap.to_revision.as_u64());
            write_u64(&mut out, gap.dropped_count);
            match &gap.reason {
                TraceGapReason::QueueOverflow => write_u8(&mut out, 0),
                TraceGapReason::QuotaExceeded => write_u8(&mut out, 1),
                TraceGapReason::DiskFailure(msg) => {
                    write_u8(&mut out, 2);
                    write_string(&mut out, msg);
                }
            }
        }
        write_u64(&mut out, self.checksum);
        out
    }

    /// Deserializa y verifica un manifiesto binario.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ScientificTraceError> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader.read_exact::<8>()?;
        if &magic != MANIFEST_MAGIC {
            return Err(ScientificTraceError::InvalidMagic);
        }
        let version = reader.read_u32()?;
        if version != SCIENTIFIC_TRACE_VERSION {
            return Err(ScientificTraceError::UnsupportedVersion {
                expected: SCIENTIFIC_TRACE_VERSION,
                actual: version,
            });
        }
        let run_id = RunId::new(reader.read_u64()?);
        let build_version = reader.read_string()?;
        let spec_id = reader.read_string()?;
        let model_id = reader.read_string()?;
        let seed = MasterSeed::new(reader.read_u64()?);
        let is_complete = reader.read_bool()?;
        let total_records = reader.read_u64()?;
        let dropped_records = reader.read_u64()?;
        let gap_count = reader.read_len()?;
        let mut gaps = Vec::with_capacity(gap_count);
        for _ in 0..gap_count {
            let from_revision = StreamRevision::new(reader.read_u64()?);
            let to_revision = StreamRevision::new(reader.read_u64()?);
            let dropped_count = reader.read_u64()?;
            let reason = match reader.read_u8()? {
                0 => TraceGapReason::QueueOverflow,
                1 => TraceGapReason::QuotaExceeded,
                2 => TraceGapReason::DiskFailure(reader.read_string()?),
                other => {
                    return Err(ScientificTraceError::CorruptedFormat(format!(
                        "unknown TraceGapReason tag: {other}"
                    )));
                }
            };
            gaps.push(TraceGap {
                from_revision,
                to_revision,
                dropped_count,
                reason,
            });
        }
        let checksum = reader.read_u64()?;
        reader.ensure_eof()?;

        let manifest = Self {
            version,
            run_id,
            build_version,
            spec_id,
            model_id,
            seed,
            is_complete,
            total_records,
            dropped_records,
            gaps,
            checksum,
        };
        manifest.verify()?;
        Ok(manifest)
    }
}

/// Entradas del tick capturadas para reproducibilidad científica.
#[derive(Debug, Clone, PartialEq)]
pub struct TickInputs {
    /// Tick lógico de la ejecución.
    pub tick: Tick,
    /// Observación capturada al inicio del tick.
    pub observation: Observation,
}

impl TickInputs {
    /// Construye un [`TickInputs`] a partir de `tick` y `observation`.
    pub fn new(tick: Tick, observation: Observation) -> Self {
        Self { tick, observation }
    }
}

/// Representación propia y serializable de la decisión de [`gogamen_core::ActionGuard`].
#[derive(Debug, Clone, PartialEq)]
pub enum RecordedGuardDecision {
    /// Autoriza la acción propuesta.
    Allow(Action),
    /// Autoriza una acción de respaldo seguro.
    Fallback(Action),
    /// No-operación explícita.
    NoOp,
    /// Rechazo determinista con descripción textual.
    Reject(String),
}

impl From<&GuardDecision> for RecordedGuardDecision {
    fn from(decision: &GuardDecision) -> Self {
        match decision {
            GuardDecision::Allow(action) => Self::Allow(action.clone()),
            GuardDecision::Fallback(action) => Self::Fallback(action.clone()),
            GuardDecision::NoOp => Self::NoOp,
            GuardDecision::Reject(reason) => Self::Reject(reason.to_string()),
        }
    }
}

/// Decisión y acción efectiva de un tick en la traza científica.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionAction {
    /// Estado emitido por el planificador.
    pub plan_status: PlanStatus,
    /// Acción propuesta por el planificador, si existió.
    pub proposed_action: Option<Action>,
    /// Decisión emitida por la guardia.
    pub guard_decision: RecordedGuardDecision,
    /// Acción efectivamente aplicada al entorno (`None` en `NoOp` o `Reject`).
    pub applied_action: Option<Action>,
    /// Indica si se ejecutó una acción de respaldo.
    pub used_fallback: bool,
}

impl DecisionAction {
    /// Construye un [`DecisionAction`] a partir de una entrada [`TickTraceEntry`].
    pub fn from_trace_entry(entry: &TickTraceEntry) -> Self {
        Self {
            plan_status: entry.plan_status,
            proposed_action: entry.selected_action.clone(),
            guard_decision: RecordedGuardDecision::from(&entry.guard_decision),
            applied_action: entry.applied_action.clone(),
            used_fallback: entry.used_fallback,
        }
    }
}

/// Semillas deterministas asociadas al tick (maestra y streams derivados).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamSeeds {
    /// Semilla maestra de la ejecución.
    pub master_seed: MasterSeed,
    /// Semillas derivadas indexadas por nombre de stream en orden determinista.
    pub derived: BTreeMap<String, u64>,
}

impl StreamSeeds {
    /// Crea [`StreamSeeds`] solo con la semilla maestra.
    pub fn new(master_seed: MasterSeed) -> Self {
        Self {
            master_seed,
            derived: BTreeMap::new(),
        }
    }

    /// Añade una semilla derivada nombrada y devuelve `self` encadenable.
    pub fn with_derived(mut self, name: impl Into<String>, seed: u64) -> Self {
        self.derived.insert(name.into(), seed);
        self
    }
}

/// Referencia a un checkpoint de replay completo ([`SimulationSnapshot`]) con
/// anotaciones opcionales sujetas a [`SensitiveFieldPolicy`].
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointRef {
    /// Snapshot completo e íntegro de simulación para replay/ramificación.
    pub snapshot: SimulationSnapshot,
    /// Anotaciones auxiliares del checkpoint (validadas por [`SensitiveFieldPolicy`]).
    pub annotations: BTreeMap<String, String>,
}

impl CheckpointRef {
    /// Crea un [`CheckpointRef`] sin anotaciones adicionales.
    pub fn new(snapshot: SimulationSnapshot) -> Self {
        Self {
            snapshot,
            annotations: BTreeMap::new(),
        }
    }

    /// Crea un [`CheckpointRef`] con anotaciones adicionales.
    pub fn with_annotations(
        snapshot: SimulationSnapshot,
        annotations: BTreeMap<String, String>,
    ) -> Self {
        Self {
            snapshot,
            annotations,
        }
    }

    /// Devuelve una versión saneada del checkpoint aplicando `policy`.
    pub fn sanitized(&self, policy: &SensitiveFieldPolicy) -> Result<Self, ScientificTraceError> {
        self.snapshot
            .verify()
            .map_err(|err| ScientificTraceError::InvalidSnapshot(err.to_string()))?;
        let annotations = policy.sanitize_map(&self.annotations)?;
        Ok(Self {
            snapshot: self.snapshot.clone(),
            annotations,
        })
    }
}

impl From<SimulationSnapshot> for CheckpointRef {
    fn from(snapshot: SimulationSnapshot) -> Self {
        Self::new(snapshot)
    }
}

/// Registro científico durable correspondiente a un paso o evento de replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ScientificRecord {
    /// Revisión monotónica del registro dentro de la ejecución.
    pub revision: StreamRevision,
    /// Entradas del tick (tick y observación).
    pub inputs: TickInputs,
    /// Decisión del planificador y de la guardia junto con la acción aplicada.
    pub decision: DecisionAction,
    /// Semillas deterministas activas en el tick.
    pub stream_seeds: StreamSeeds,
    /// Checkpoint opcional de replay para reanudar o bifurcar exactamente desde este punto.
    pub checkpoint: Option<CheckpointRef>,
}

impl ScientificRecord {
    /// Construye un [`ScientificRecord`] a partir de un [`TickTraceEntry`] del runtime.
    pub fn from_trace_entry(
        revision: StreamRevision,
        entry: &TickTraceEntry,
        stream_seeds: StreamSeeds,
        checkpoint: Option<CheckpointRef>,
    ) -> Self {
        Self {
            revision,
            inputs: TickInputs::new(entry.tick, entry.observation.clone()),
            decision: DecisionAction::from_trace_entry(entry),
            stream_seeds,
            checkpoint,
        }
    }

    /// Aplica `policy` al registro (incluyendo su [`CheckpointRef`] si existe) y
    /// valida la integridad del snapshot embebido.
    pub fn sanitized(&self, policy: &SensitiveFieldPolicy) -> Result<Self, ScientificTraceError> {
        let checkpoint = match &self.checkpoint {
            Some(cp) => Some(cp.sanitized(policy)?),
            None => None,
        };
        Ok(Self {
            revision: self.revision,
            inputs: self.inputs.clone(),
            decision: self.decision.clone(),
            stream_seeds: self.stream_seeds.clone(),
            checkpoint,
        })
    }

    /// Serializa el registro aplicando primero `policy` para evitar persistir credenciales.
    ///
    /// El encoding es single-pass: se reserva un único `Vec<u8>` con capacidad exacta,
    /// se escribe la cabecera `[MAGIC(8), VERSION(4), LEN(8)]`, el payload desde el
    /// offset 20, se parchea `LEN` y se añade el checksum FNV-1a (8 bytes) calculado
    /// sobre `out[20..]`. No hay payload intermedio ni copias. El formato binario es
    /// idéntico al histórico y compatible con [`ScientificRecord::from_bytes`].
    pub fn to_bytes_with_policy(
        &self,
        policy: &SensitiveFieldPolicy,
    ) -> Result<Vec<u8>, ScientificTraceError> {
        // Solo el checkpoint requiere sanitización; el resto se serializa por referencia
        // para evitar clonar inputs/decisión/semillas.
        match &self.checkpoint {
            Some(cp) => {
                let clean = cp.sanitized(policy)?;
                Ok(self.encode_single_pass(Some(&clean)))
            }
            None => Ok(self.encode_single_pass(None)),
        }
    }

    /// Serializa el registro usando la política estricta por defecto ([`SensitiveFieldPolicy::default`]).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ScientificTraceError> {
        self.to_bytes_with_policy(&SensitiveFieldPolicy::default())
    }

    /// Tamaño exacto en bytes del payload (sin cabecera ni checksum).
    fn payload_len(&self, checkpoint: Option<&CheckpointRef>) -> usize {
        fn f64_slice(values: &[f64]) -> usize {
            8 + 8 * values.len()
        }
        fn string(s: &str) -> usize {
            8 + s.len()
        }
        fn action(a: &Action) -> usize {
            match a {
                Action::Discrete(_) => 1 + 8,
                Action::Continuous(vals) => 1 + f64_slice(vals),
            }
        }
        fn opt_action(a: Option<&Action>) -> usize {
            1 + a.map_or(0, action)
        }
        let mut n = 8 + 8; // revision, tick
        n += 8 + 8 + f64_slice(self.inputs.observation.features());
        n += 1 + opt_action(self.decision.proposed_action.as_ref());
        n += match &self.decision.guard_decision {
            RecordedGuardDecision::Allow(a) | RecordedGuardDecision::Fallback(a) => 1 + action(a),
            RecordedGuardDecision::NoOp => 1,
            RecordedGuardDecision::Reject(msg) => 1 + string(msg),
        };
        n += opt_action(self.decision.applied_action.as_ref());
        n += 1; // used_fallback
        n += 8 + 8; // master seed, derived len
        for name in self.stream_seeds.derived.keys() {
            n += string(name) + 8;
        }
        n += 1; // checkpoint tag
        if let Some(cp) = checkpoint {
            let snap = &cp.snapshot;
            n += 8 + 8 + f64_slice(snap.state().values()) + 8;
            n += match snap.terminal() {
                Terminal::Ongoing => 1,
                Terminal::Terminated(_) => 2,
            };
            n += 8 + 8 + 8;
            for (k, v) in &cp.annotations {
                n += string(k) + string(v);
            }
        }
        n
    }

    fn encode_single_pass(&self, checkpoint: Option<&CheckpointRef>) -> Vec<u8> {
        const HEADER_LEN: usize = 8 + 4 + 8;
        let payload_len = self.payload_len(checkpoint);
        let mut out = Vec::with_capacity(HEADER_LEN + payload_len + 8);
        out.extend_from_slice(TRACE_MAGIC);
        write_u32(&mut out, SCIENTIFIC_TRACE_VERSION);
        write_u64(&mut out, 0); // LEN placeholder, patched below.
        debug_assert_eq!(out.len(), HEADER_LEN);

        write_u64(&mut out, self.revision.as_u64());
        write_u64(&mut out, self.inputs.tick.as_u64());
        write_observation(&mut out, &self.inputs.observation);
        write_decision_action(&mut out, &self.decision);
        write_stream_seeds(&mut out, &self.stream_seeds);
        match checkpoint {
            Some(cp) => {
                write_u8(&mut out, 1);
                write_checkpoint(&mut out, cp);
            }
            None => write_u8(&mut out, 0),
        }

        let written = out.len() - HEADER_LEN;
        debug_assert_eq!(written, payload_len);
        out[12..HEADER_LEN].copy_from_slice(&(written as u64).to_le_bytes());

        let mut hasher = TraceHasher::new();
        hasher.write_u64(u64::from(SCIENTIFIC_TRACE_VERSION));
        hasher.write_bytes(&out[HEADER_LEN..]);
        let checksum = hasher.finish();
        write_u64(&mut out, checksum);
        out
    }

    /// Deserializa un [`ScientificRecord`] validando cabecera, versión y checksum.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ScientificTraceError> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader.read_exact::<8>()?;
        if &magic != TRACE_MAGIC {
            return Err(ScientificTraceError::InvalidMagic);
        }
        let version = reader.read_u32()?;
        if version != SCIENTIFIC_TRACE_VERSION {
            return Err(ScientificTraceError::UnsupportedVersion {
                expected: SCIENTIFIC_TRACE_VERSION,
                actual: version,
            });
        }
        let payload_len = reader.read_len()?;
        let payload = reader.read_slice(payload_len)?;
        let expected_checksum = reader.read_u64()?;
        reader.ensure_eof()?;

        let mut hasher = TraceHasher::new();
        hasher.write_u64(u64::from(version));
        hasher.write_bytes(payload);
        let actual_checksum = hasher.finish();
        if expected_checksum != actual_checksum {
            return Err(ScientificTraceError::ChecksumMismatch {
                expected: expected_checksum,
                actual: actual_checksum,
            });
        }

        let mut p = ByteReader::new(payload);
        let revision = StreamRevision::new(p.read_u64()?);
        let tick = Tick::new(p.read_u64()?);
        let observation = read_observation(&mut p)?;
        let decision = read_decision_action(&mut p)?;
        let stream_seeds = read_stream_seeds(&mut p)?;
        let has_checkpoint = p.read_bool()?;
        let checkpoint = if has_checkpoint {
            Some(read_checkpoint(&mut p)?)
        } else {
            None
        };
        p.ensure_eof()?;

        Ok(Self {
            revision,
            inputs: TickInputs::new(tick, observation),
            decision,
            stream_seeds,
            checkpoint,
        })
    }
}

/// Archivo completo autocontenido (manifiesto + registros) para exportación y replay de fixtures.
#[derive(Debug, Clone, PartialEq)]
pub struct ScientificTraceArchive {
    /// Manifiesto versionado de la ejecución.
    pub manifest: TraceManifest,
    /// Registros científicos ordenados cronológicamente.
    pub records: Vec<ScientificRecord>,
}

impl ScientificTraceArchive {
    /// Serializa el archivo completo en formato binario determinista.
    pub fn to_bytes(&self, policy: &SensitiveFieldPolicy) -> Result<Vec<u8>, ScientificTraceError> {
        let mut manifest = self.manifest.clone();
        manifest.spec_id = policy.sanitize_field("spec_id", &manifest.spec_id)?;
        manifest.model_id = policy.sanitize_field("model_id", &manifest.model_id)?;
        manifest.recompute_checksum();

        let manifest_bytes = manifest.to_bytes();
        let mut encoded_records = Vec::with_capacity(self.records.len());
        for record in &self.records {
            encoded_records.push(record.to_bytes_with_policy(policy)?);
        }

        let mut out = Vec::new();
        out.extend_from_slice(ARCHIVE_MAGIC);
        write_u32(&mut out, SCIENTIFIC_TRACE_VERSION);
        write_u64(&mut out, manifest_bytes.len() as u64);
        out.extend_from_slice(&manifest_bytes);
        write_u64(&mut out, encoded_records.len() as u64);
        for rec in encoded_records {
            write_u64(&mut out, rec.len() as u64);
            out.extend_from_slice(&rec);
        }
        Ok(out)
    }

    /// Deserializa y verifica un [`ScientificTraceArchive`] completo.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ScientificTraceError> {
        let mut reader = ByteReader::new(bytes);
        let magic = reader.read_exact::<8>()?;
        if &magic != ARCHIVE_MAGIC {
            return Err(ScientificTraceError::InvalidMagic);
        }
        let version = reader.read_u32()?;
        if version != SCIENTIFIC_TRACE_VERSION {
            return Err(ScientificTraceError::UnsupportedVersion {
                expected: SCIENTIFIC_TRACE_VERSION,
                actual: version,
            });
        }
        let manifest_len = reader.read_len()?;
        let manifest_bytes = reader.read_slice(manifest_len)?;
        let manifest = TraceManifest::from_bytes(manifest_bytes)?;

        let record_count = reader.read_len()?;
        let mut records = Vec::with_capacity(record_count);
        for _ in 0..record_count {
            let rec_len = reader.read_len()?;
            let rec_bytes = reader.read_slice(rec_len)?;
            records.push(ScientificRecord::from_bytes(rec_bytes)?);
        }
        reader.ensure_eof()?;

        Ok(Self { manifest, records })
    }
}

/// Errores de formato, validación o sensibilidad en trazas científicas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScientificTraceError {
    /// Cabecera mágica inválida.
    InvalidMagic,
    /// Versión de formato no soportada.
    UnsupportedVersion {
        /// Versión esperada.
        expected: u32,
        /// Versión encontrada.
        actual: u32,
    },
    /// Checksum inconsistente respecto al contenido.
    ChecksumMismatch {
        /// Checksum almacenado.
        expected: u64,
        /// Checksum recalculado.
        actual: u64,
    },
    /// El manifiesto declara completitud `true` a pesar de tener pérdidas o huecos.
    InconsistentCompleteness,
    /// Se detectó un campo o valor sensible prohibido por [`SensitiveFieldPolicy`].
    SensitiveDataDetected {
        /// Nombre del campo afectado.
        field: String,
    },
    /// Snapshot inválido o corrupto.
    InvalidSnapshot(String),
    /// Buffer truncado o formato binario corrupto.
    CorruptedFormat(String),
}

impl fmt::Display for ScientificTraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid scientific trace magic header"),
            Self::UnsupportedVersion { expected, actual } => {
                write!(
                    f,
                    "unsupported scientific trace version {actual} (expected {expected})"
                )
            }
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "scientific trace checksum mismatch: expected 0x{expected:016x}, got 0x{actual:016x}"
                )
            }
            Self::InconsistentCompleteness => {
                write!(
                    f,
                    "manifest claims completeness despite non-zero dropped records or gaps"
                )
            }
            Self::SensitiveDataDetected { field } => {
                write!(
                    f,
                    "sensitive credential or secret detected in field '{field}'"
                )
            }
            Self::InvalidSnapshot(msg) => write!(f, "invalid snapshot in checkpoint: {msg}"),
            Self::CorruptedFormat(msg) => write!(f, "corrupted scientific trace format: {msg}"),
        }
    }
}

impl Error for ScientificTraceError {}

/// Política explícita ante saturación de cola, exceso de cuota o fallo de disco.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverflowPolicy {
    /// Nunca bloquea el bucle de control: descarta/registra hueco y marca `is_complete = false`.
    MarkIncomplete,
    /// Marca `is_complete = false` y detiene explícitamente el experimento con error.
    AbortExperiment,
}

/// Cuotas acotadas de buffer y persistencia para la traza científica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TraceQuota {
    /// Máximo de registros en la cola acotada en memoria (`>= 1`).
    pub max_queue_records: usize,
    /// Máximo de bytes acumulados en la cola acotada en memoria (`>= 1`).
    pub max_queue_bytes: usize,
    /// Límite opcional de registros totales persistidos en el experimento.
    pub max_total_records: Option<u64>,
    /// Límite opcional de bytes totales persistidos en el experimento.
    pub max_total_bytes: Option<u64>,
}

impl TraceQuota {
    /// Crea una cuota acotada por registros y bytes en cola.
    pub fn new(max_queue_records: usize, max_queue_bytes: usize) -> Result<Self, TraceWriteError> {
        if max_queue_records == 0 {
            return Err(TraceWriteError::InvalidQuota(
                "max_queue_records must be >= 1",
            ));
        }
        if max_queue_bytes == 0 {
            return Err(TraceWriteError::InvalidQuota(
                "max_queue_bytes must be >= 1",
            ));
        }
        Ok(Self {
            max_queue_records,
            max_queue_bytes,
            max_total_records: None,
            max_total_bytes: None,
        })
    }

    /// Añade límites totales opcionales de registros y bytes para toda la ejecución.
    pub const fn with_total_limits(
        mut self,
        max_total_records: Option<u64>,
        max_total_bytes: Option<u64>,
    ) -> Self {
        self.max_total_records = max_total_records;
        self.max_total_bytes = max_total_bytes;
        self
    }
}

impl Default for TraceQuota {
    fn default() -> Self {
        Self {
            max_queue_records: 1024,
            max_queue_bytes: 4 * 1024 * 1024,
            max_total_records: None,
            max_total_bytes: None,
        }
    }
}

/// Resultado no bloqueante al encolar un registro en [`BufferedTraceSink`] o [`AsyncTraceWriter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnqueueOutcome {
    /// El registro fue aceptado en la cola acotada.
    Enqueued,
    /// El registro fue descartado bajo [`OverflowPolicy::MarkIncomplete`] sin bloquear el control.
    DroppedMarkIncomplete,
}

/// Errores explícitos del subsistema de escritura durable de trazas científicas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceWriteError {
    /// Configuración de cuota inválida.
    InvalidQuota(&'static str),
    /// Error de validación o política de sensibilidad del registro/manifiesto.
    Trace(ScientificTraceError),
    /// Fallo explícito de disco o almacenamiento subyacente.
    DiskFailure {
        /// Descripción del fallo de almacenamiento.
        message: String,
    },
    /// El experimento fue detenido explícitamente según [`OverflowPolicy::AbortExperiment`].
    ExperimentAborted {
        /// Motivo determinista de la detención.
        reason: String,
    },
    /// El escritor ya fue cerrado.
    WriterClosed,
}

impl fmt::Display for TraceWriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuota(msg) => write!(f, "invalid trace quota: {msg}"),
            Self::Trace(err) => write!(f, "trace validation error: {err}"),
            Self::DiskFailure { message } => write!(f, "disk failure: {message}"),
            Self::ExperimentAborted { reason } => {
                write!(f, "experiment aborted by trace policy: {reason}")
            }
            Self::WriterClosed => write!(f, "trace writer is already closed"),
        }
    }
}

impl Error for TraceWriteError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Trace(err) => Some(err),
            _ => None,
        }
    }
}

impl From<ScientificTraceError> for TraceWriteError {
    fn from(value: ScientificTraceError) -> Self {
        Self::Trace(value)
    }
}

/// Contrato de almacenamiento durable fuera del bucle de control.
pub trait TraceStorageBackend: Send {
    /// Persiste un registro científico ya serializado.
    fn write_record(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError>;
    /// Persiste el manifiesto serializado de la ejecución.
    fn write_manifest(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError>;
    /// Fuerza el volcado de buffers al medio subyacente.
    fn flush(&mut self) -> Result<(), TraceWriteError>;
}

/// Backend en memoria con soporte de simulación determinista de fallos de disco.
#[derive(Debug, Default)]
pub struct InMemoryTraceBackend {
    records: Vec<Vec<u8>>,
    manifest: Option<Vec<u8>>,
    fail_flag: Arc<AtomicBool>,
    fail_after_writes: Option<usize>,
    write_attempts: Arc<AtomicUsize>,
}

impl InMemoryTraceBackend {
    /// Crea un backend en memoria limpio.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configura el backend para simular [`TraceWriteError::DiskFailure`] tras `n` escrituras exitosas.
    pub fn with_failure_after_writes(mut self, successful_writes: usize) -> Self {
        self.fail_after_writes = Some(successful_writes);
        self
    }

    /// Devuelve un disparador atómico para inyectar un fallo de disco en caliente.
    pub fn failure_trigger(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.fail_flag)
    }

    /// Devuelve los registros crudos persistidos.
    pub fn raw_records(&self) -> &[Vec<u8>] {
        &self.records
    }

    /// Devuelve el manifiesto crudo persistido, si se escribió.
    pub fn raw_manifest(&self) -> Option<&[u8]> {
        self.manifest.as_deref()
    }

    /// Decodifica todos los registros persistidos en el backend.
    pub fn decode_records(&self) -> Result<Vec<ScientificRecord>, ScientificTraceError> {
        self.records
            .iter()
            .map(|bytes| ScientificRecord::from_bytes(bytes))
            .collect()
    }

    /// Decodifica el manifiesto persistido, si existe.
    pub fn decode_manifest(&self) -> Result<Option<TraceManifest>, ScientificTraceError> {
        self.manifest
            .as_deref()
            .map(TraceManifest::from_bytes)
            .transpose()
    }

    fn check_failure(&self) -> Result<(), TraceWriteError> {
        if self.fail_flag.load(Ordering::Acquire) {
            return Err(TraceWriteError::DiskFailure {
                message: "simulated disk write failure".to_owned(),
            });
        }
        let attempt = self.write_attempts.fetch_add(1, Ordering::AcqRel);
        if let Some(limit) = self.fail_after_writes
            && attempt >= limit
        {
            return Err(TraceWriteError::DiskFailure {
                message: format!("simulated disk failure at write #{attempt}"),
            });
        }
        Ok(())
    }
}

impl TraceStorageBackend for InMemoryTraceBackend {
    fn write_record(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError> {
        self.check_failure()?;
        self.records.push(encoded.to_vec());
        Ok(())
    }

    fn write_manifest(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError> {
        self.check_failure()?;
        self.manifest = Some(encoded.to_vec());
        Ok(())
    }

    fn flush(&mut self) -> Result<(), TraceWriteError> {
        if self.fail_flag.load(Ordering::Acquire) {
            return Err(TraceWriteError::DiskFailure {
                message: "simulated disk flush failure".to_owned(),
            });
        }
        Ok(())
    }
}

/// Backend durable en disco (`std::fs`) que escribe `records.gtrace` y `manifest.gmanif`.
#[derive(Debug)]
pub struct FileTraceBackend {
    dir: PathBuf,
    records_writer: BufWriter<File>,
}

impl FileTraceBackend {
    /// Abre o crea el directorio `dir` para almacenar la traza científica.
    pub fn create(dir: impl AsRef<Path>) -> Result<Self, TraceWriteError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|err| TraceWriteError::DiskFailure {
            message: err.to_string(),
        })?;
        let records_file = File::create(dir.join("records.gtrace")).map_err(|err| {
            TraceWriteError::DiskFailure {
                message: err.to_string(),
            }
        })?;
        Ok(Self {
            dir,
            records_writer: BufWriter::new(records_file),
        })
    }

    /// Devuelve el directorio de almacenamiento.
    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

impl TraceStorageBackend for FileTraceBackend {
    fn write_record(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError> {
        let len_bytes = (encoded.len() as u64).to_le_bytes();
        self.records_writer
            .write_all(&len_bytes)
            .and_then(|()| self.records_writer.write_all(encoded))
            .map_err(|err| TraceWriteError::DiskFailure {
                message: err.to_string(),
            })
    }

    fn write_manifest(&mut self, encoded: &[u8]) -> Result<(), TraceWriteError> {
        let manifest_path = self.dir.join("manifest.gmanif");
        let mut file =
            File::create(&manifest_path).map_err(|err| TraceWriteError::DiskFailure {
                message: err.to_string(),
            })?;
        file.write_all(encoded)
            .and_then(|()| file.flush())
            .map_err(|err| TraceWriteError::DiskFailure {
                message: err.to_string(),
            })
    }

    fn flush(&mut self) -> Result<(), TraceWriteError> {
        self.records_writer
            .flush()
            .map_err(|err| TraceWriteError::DiskFailure {
                message: err.to_string(),
            })
    }
}

/// Sink acotado y no bloqueante que amortigua registros fuera del bucle de control crítico.
#[derive(Debug)]
pub struct BufferedTraceSink {
    manifest: TraceManifest,
    quota: TraceQuota,
    policy: OverflowPolicy,
    sensitive_policy: SensitiveFieldPolicy,
    queue: VecDeque<(StreamRevision, Vec<u8>)>,
    queued_bytes: usize,
    accepted_records: u64,
    accepted_bytes: u64,
    aborted_reason: Option<String>,
}

impl BufferedTraceSink {
    /// Crea un nuevo [`BufferedTraceSink`] validando el manifiesto inicial contra `sensitive_policy`.
    pub fn new(
        mut manifest_config: TraceManifestConfig,
        quota: TraceQuota,
        policy: OverflowPolicy,
        sensitive_policy: SensitiveFieldPolicy,
    ) -> Result<Self, TraceWriteError> {
        manifest_config.spec_id =
            sensitive_policy.sanitize_field("spec_id", &manifest_config.spec_id)?;
        manifest_config.model_id =
            sensitive_policy.sanitize_field("model_id", &manifest_config.model_id)?;
        let manifest = TraceManifest::new(manifest_config);
        Ok(Self {
            manifest,
            quota,
            policy,
            sensitive_policy,
            queue: VecDeque::new(),
            queued_bytes: 0,
            accepted_records: 0,
            accepted_bytes: 0,
            aborted_reason: None,
        })
    }

    /// Devuelve el manifiesto actual (refleja siempre cualquier pérdida o hueco).
    pub const fn manifest(&self) -> &TraceManifest {
        &self.manifest
    }

    /// Número de registros actualmente en cola pendientes de volcado.
    pub fn queued_records(&self) -> usize {
        self.queue.len()
    }

    /// Bytes actualmente en cola pendientes de volcado.
    pub const fn queued_bytes(&self) -> usize {
        self.queued_bytes
    }

    /// Encola un registro sin bloquear ni realizar E/S en el bucle de control.
    pub fn record(&mut self, record: &ScientificRecord) -> Result<EnqueueOutcome, TraceWriteError> {
        if let Some(reason) = &self.aborted_reason {
            return Err(TraceWriteError::ExperimentAborted {
                reason: reason.clone(),
            });
        }

        let encoded = record.to_bytes_with_policy(&self.sensitive_policy)?;
        self.enqueue_encoded(record.revision, encoded)
    }

    /// Aplica cuotas y encola un registro ya serializado y sanitizado.
    ///
    /// Es la sección crítica de [`AsyncTraceWriter::try_record`]: solo contabilidad
    /// de cuotas y un `push_back` O(1) (más, en caso de descarte, un `record_gap`).
    fn enqueue_encoded(
        &mut self,
        revision: StreamRevision,
        encoded: Vec<u8>,
    ) -> Result<EnqueueOutcome, TraceWriteError> {
        if let Some(reason) = &self.aborted_reason {
            return Err(TraceWriteError::ExperimentAborted {
                reason: reason.clone(),
            });
        }
        let encoded_len = encoded.len();

        let total_records_exceeded = self
            .quota
            .max_total_records
            .is_some_and(|limit| self.accepted_records.saturating_add(1) > limit);
        let total_bytes_exceeded = self
            .quota
            .max_total_bytes
            .is_some_and(|limit| self.accepted_bytes.saturating_add(encoded_len as u64) > limit);

        if total_records_exceeded || total_bytes_exceeded {
            return self.handle_drop(revision, 1, TraceGapReason::QuotaExceeded);
        }

        let queue_records_exceeded =
            self.queue.len().saturating_add(1) > self.quota.max_queue_records;
        let queue_bytes_exceeded =
            self.queued_bytes.saturating_add(encoded_len) > self.quota.max_queue_bytes;

        if queue_records_exceeded || queue_bytes_exceeded {
            return self.handle_drop(revision, 1, TraceGapReason::QueueOverflow);
        }

        self.queued_bytes = self.queued_bytes.saturating_add(encoded_len);
        self.accepted_records = self.accepted_records.saturating_add(1);
        self.accepted_bytes = self.accepted_bytes.saturating_add(encoded_len as u64);
        self.queue.push_back((revision, encoded));
        Ok(EnqueueOutcome::Enqueued)
    }

    /// Vuelca los registros pendientes al `backend` fuera del bucle crítico.
    pub fn flush_to(
        &mut self,
        backend: &mut dyn TraceStorageBackend,
    ) -> Result<usize, TraceWriteError> {
        let mut flushed = 0usize;
        // `total_records` y checksum se actualizan una sola vez por lote.
        let result = self.flush_queue(backend, &mut flushed);
        self.apply_flushed(flushed);
        result?;
        if let Err(err) = backend.flush() {
            let msg = err.to_string();
            let rev = StreamRevision::new(self.manifest.total_records);
            self.manifest
                .record_gap(rev, rev, 1, TraceGapReason::DiskFailure(msg.clone()));
            if self.policy == OverflowPolicy::AbortExperiment {
                self.aborted_reason = Some(msg.clone());
                return Err(TraceWriteError::ExperimentAborted { reason: msg });
            }
        }
        Ok(flushed)
    }

    fn apply_flushed(&mut self, flushed: usize) {
        if flushed > 0 {
            self.manifest.total_records =
                self.manifest.total_records.saturating_add(flushed as u64);
            self.manifest.recompute_checksum();
        }
    }

    fn flush_queue(
        &mut self,
        backend: &mut dyn TraceStorageBackend,
        flushed: &mut usize,
    ) -> Result<(), TraceWriteError> {
        while let Some((revision, encoded)) = self.queue.pop_front() {
            self.queued_bytes = self.queued_bytes.saturating_sub(encoded.len());
            match backend.write_record(&encoded) {
                Ok(()) => {
                    *flushed += 1;
                }
                Err(err) => {
                    let msg = err.to_string();
                    self.manifest.record_gap(
                        revision,
                        revision,
                        1,
                        TraceGapReason::DiskFailure(msg.clone()),
                    );
                    if self.policy == OverflowPolicy::AbortExperiment {
                        self.aborted_reason = Some(msg.clone());
                        return Err(TraceWriteError::ExperimentAborted { reason: msg });
                    }
                }
            }
        }
        Ok(())
    }

    /// Vuelca los registros pendientes y persiste el manifiesto final en `backend`.
    pub fn finalize(
        &mut self,
        backend: &mut dyn TraceStorageBackend,
    ) -> Result<TraceManifest, TraceWriteError> {
        self.flush_to(backend)?;
        self.manifest.recompute_checksum();
        let manifest_bytes = self.manifest.to_bytes();
        if let Err(err) = backend.write_manifest(&manifest_bytes) {
            let msg = err.to_string();
            let rev = StreamRevision::new(self.manifest.total_records);
            self.manifest
                .record_gap(rev, rev, 1, TraceGapReason::DiskFailure(msg.clone()));
            if self.policy == OverflowPolicy::AbortExperiment {
                self.aborted_reason = Some(msg.clone());
                return Err(TraceWriteError::ExperimentAborted { reason: msg });
            }
        }
        Ok(self.manifest.clone())
    }

    fn handle_drop(
        &mut self,
        revision: StreamRevision,
        count: u64,
        reason: TraceGapReason,
    ) -> Result<EnqueueOutcome, TraceWriteError> {
        let reason_text = reason.to_string();
        self.manifest.record_gap(revision, revision, count, reason);
        match self.policy {
            OverflowPolicy::MarkIncomplete => Ok(EnqueueOutcome::DroppedMarkIncomplete),
            OverflowPolicy::AbortExperiment => {
                self.aborted_reason = Some(reason_text.clone());
                Err(TraceWriteError::ExperimentAborted {
                    reason: reason_text,
                })
            }
        }
    }
}

#[derive(Debug)]
struct AsyncSharedState {
    sink: BufferedTraceSink,
    closed: bool,
    worker_error: Option<TraceWriteError>,
}

/// Escritor asíncrono en hilo dedicado que drena la cola acotada hacia un
/// [`TraceStorageBackend`] sin bloquear el bucle de control.
#[derive(Debug)]
pub struct AsyncTraceWriter {
    shared: Arc<(Mutex<AsyncSharedState>, Condvar)>,
    worker: Option<JoinHandle<()>>,
    /// Política de sanitización inmutable, copiada fuera del mutex para serializar sin lock.
    sensitive_policy: SensitiveFieldPolicy,
}

impl AsyncTraceWriter {
    /// Inicia un escritor asíncrono en segundo plano sobre `backend`.
    pub fn start(
        manifest_config: TraceManifestConfig,
        quota: TraceQuota,
        policy: OverflowPolicy,
        sensitive_policy: SensitiveFieldPolicy,
        mut backend: Box<dyn TraceStorageBackend>,
    ) -> Result<Self, TraceWriteError> {
        let sink =
            BufferedTraceSink::new(manifest_config, quota, policy, sensitive_policy.clone())?;
        let shared = Arc::new((
            Mutex::new(AsyncSharedState {
                sink,
                closed: false,
                worker_error: None,
            }),
            Condvar::new(),
        ));
        let worker_shared = Arc::clone(&shared);
        let handle = thread::Builder::new()
            .name("gogamen-sci-trace".to_owned())
            .spawn(move || {
                let (lock, cvar) = &*worker_shared;
                loop {
                    let mut guard = match lock.lock() {
                        Ok(g) => g,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    while guard.sink.queued_records() == 0 && !guard.closed {
                        guard = match cvar.wait(guard) {
                            Ok(g) => g,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                    }

                    // Extraemos un lote bajo lock y escribimos fuera del lock para no bloquear el bucle de control.
                    let mut batch = Vec::new();
                    while let Some(item) = guard.sink.queue.pop_front() {
                        guard.sink.queued_bytes =
                            guard.sink.queued_bytes.saturating_sub(item.1.len());
                        batch.push(item);
                    }
                    let is_closed = guard.closed;
                    drop(guard);

                    // Escribimos el lote fuera del lock y acumulamos resultados; el manifiesto
                    // (`total_records` + checksum) se actualiza una única vez al final del lote.
                    let mut ok_count = 0u64;
                    let mut failures: Vec<(StreamRevision, String)> = Vec::new();
                    for (revision, encoded) in batch {
                        match backend.write_record(&encoded) {
                            Ok(()) => ok_count += 1,
                            Err(err) => failures.push((revision, err.to_string())),
                        }
                    }
                    if ok_count > 0 || !failures.is_empty() {
                        let mut guard = match lock.lock() {
                            Ok(g) => g,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        for (revision, msg) in failures {
                            guard.sink.manifest.record_gap(
                                revision,
                                revision,
                                1,
                                TraceGapReason::DiskFailure(msg.clone()),
                            );
                            if guard.sink.policy == OverflowPolicy::AbortExperiment {
                                guard.sink.aborted_reason = Some(msg.clone());
                                guard.worker_error =
                                    Some(TraceWriteError::ExperimentAborted { reason: msg });
                            }
                        }
                        guard.sink.apply_flushed(ok_count as usize);
                    }

                    if is_closed {
                        let flush_res = backend.flush();
                        let mut guard = match lock.lock() {
                            Ok(g) => g,
                            Err(poisoned) => poisoned.into_inner(),
                        };
                        if let Err(err) = flush_res {
                            let msg = err.to_string();
                            let rev = StreamRevision::new(guard.sink.manifest.total_records);
                            guard.sink.manifest.record_gap(
                                rev,
                                rev,
                                1,
                                TraceGapReason::DiskFailure(msg.clone()),
                            );
                            if guard.sink.policy == OverflowPolicy::AbortExperiment {
                                guard.sink.aborted_reason = Some(msg.clone());
                                guard.worker_error =
                                    Some(TraceWriteError::ExperimentAborted { reason: msg });
                            }
                        }
                        guard.sink.manifest.recompute_checksum();
                        let manifest_bytes = guard.sink.manifest.to_bytes();
                        drop(guard);

                        if let Err(err) = backend.write_manifest(&manifest_bytes) {
                            let msg = err.to_string();
                            let mut guard = match lock.lock() {
                                Ok(g) => g,
                                Err(poisoned) => poisoned.into_inner(),
                            };
                            let rev = StreamRevision::new(guard.sink.manifest.total_records);
                            guard.sink.manifest.record_gap(
                                rev,
                                rev,
                                1,
                                TraceGapReason::DiskFailure(msg.clone()),
                            );
                            if guard.sink.policy == OverflowPolicy::AbortExperiment {
                                guard.sink.aborted_reason = Some(msg.clone());
                                guard.worker_error =
                                    Some(TraceWriteError::ExperimentAborted { reason: msg });
                            }
                        }
                        break;
                    }
                }
            })
            .map_err(|err| TraceWriteError::DiskFailure {
                message: format!("failed to spawn async trace writer thread: {err}"),
            })?;

        Ok(Self {
            shared,
            worker: Some(handle),
            sensitive_policy,
        })
    }

    /// Encola un registro sin bloquear por E/S de disco.
    ///
    /// Contrato de concurrencia:
    /// - La serialización y sanitización (la parte cara) se ejecutan **fuera** del mutex
    ///   usando una copia inmutable de la [`SensitiveFieldPolicy`]; el resultado es
    ///   idéntico al de [`BufferedTraceSink::record`].
    /// - Bajo el mutex solo queda el estado mutable compartido: comprobación de
    ///   cierre/abort, contabilidad de cuotas y `VecDeque::push_back` O(1).
    /// - Ambas políticas usan `lock()` bloqueante: la sección crítica es O(1) y sin E/S,
    ///   así que la espera es breve. Un registro **nunca** se descarta porque el mutex
    ///   esté momentáneamente ocupado (p. ej. por el hilo consumidor).
    /// - Solo se descarta por desbordamiento real de cuota: con
    ///   [`OverflowPolicy::MarkIncomplete`] se devuelve
    ///   [`EnqueueOutcome::DroppedMarkIncomplete`] y se registra el hueco
    ///   (`QueueOverflow`); con [`OverflowPolicy::AbortExperiment`] se devuelve error.
    pub fn try_record(&self, record: &ScientificRecord) -> Result<EnqueueOutcome, TraceWriteError> {
        let (lock, cvar) = &*self.shared;
        let encoded = record.to_bytes_with_policy(&self.sensitive_policy);

        let mut guard = match lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.closed {
            return Err(TraceWriteError::WriterClosed);
        }
        if let Some(err) = &guard.worker_error {
            return Err(err.clone());
        }
        let outcome = guard.sink.enqueue_encoded(record.revision, encoded?)?;
        cvar.notify_one();
        Ok(outcome)
    }

    /// Devuelve una instantánea del [`TraceManifest`] actual.
    pub fn manifest_snapshot(&self) -> TraceManifest {
        let (lock, _) = &*self.shared;
        let guard = match lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.sink.manifest().clone()
    }

    /// Cierra el escritor, espera al hilo en segundo plano y devuelve el manifiesto final.
    pub fn finish(mut self) -> Result<TraceManifest, TraceWriteError> {
        self.shutdown_worker()?;
        let (lock, _) = &*self.shared;
        let guard = match lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Some(err) = &guard.worker_error {
            return Err(err.clone());
        }
        Ok(guard.sink.manifest().clone())
    }

    fn shutdown_worker(&mut self) -> Result<(), TraceWriteError> {
        let (lock, cvar) = &*self.shared;
        {
            let mut guard = match lock.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            guard.closed = true;
            cvar.notify_all();
        }
        if let Some(handle) = self.worker.take() {
            handle.join().map_err(|_| TraceWriteError::DiskFailure {
                message: "async trace writer thread panicked".to_owned(),
            })?;
        }
        Ok(())
    }
}

impl Drop for AsyncTraceWriter {
    fn drop(&mut self) {
        let _ = self.shutdown_worker();
    }
}

// --- Helpers de serialización binaria determinista en std puro ---

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_f64(out: &mut Vec<u8>, v: f64) {
    out.extend_from_slice(&v.to_bits().to_le_bytes());
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

fn write_f64_slice(out: &mut Vec<u8>, values: &[f64]) {
    write_u64(out, values.len() as u64);
    for &v in values {
        write_f64(out, v);
    }
}

fn write_observation(out: &mut Vec<u8>, obs: &Observation) {
    write_u64(out, obs.tick().as_u64());
    write_f64(out, obs.elapsed().as_secs_f64());
    write_f64_slice(out, obs.features());
}

fn write_action(out: &mut Vec<u8>, action: &Action) {
    match action {
        Action::Discrete(d) => {
            write_u8(out, 0);
            write_u64(out, d.index() as u64);
        }
        Action::Continuous(vals) => {
            write_u8(out, 1);
            write_f64_slice(out, vals);
        }
    }
}

fn write_opt_action(out: &mut Vec<u8>, action: Option<&Action>) {
    match action {
        Some(a) => {
            write_u8(out, 1);
            write_action(out, a);
        }
        None => write_u8(out, 0),
    }
}

fn write_decision_action(out: &mut Vec<u8>, decision: &DecisionAction) {
    let status_tag = match decision.plan_status {
        PlanStatus::Completed => 0,
        PlanStatus::BudgetExhausted => 1,
        PlanStatus::Cancelled => 2,
        PlanStatus::NoValidAction => 3,
    };
    write_u8(out, status_tag);
    write_opt_action(out, decision.proposed_action.as_ref());
    match &decision.guard_decision {
        RecordedGuardDecision::Allow(a) => {
            write_u8(out, 0);
            write_action(out, a);
        }
        RecordedGuardDecision::Fallback(a) => {
            write_u8(out, 1);
            write_action(out, a);
        }
        RecordedGuardDecision::NoOp => write_u8(out, 2),
        RecordedGuardDecision::Reject(msg) => {
            write_u8(out, 3);
            write_string(out, msg);
        }
    }
    write_opt_action(out, decision.applied_action.as_ref());
    write_u8(out, u8::from(decision.used_fallback));
}

fn write_stream_seeds(out: &mut Vec<u8>, seeds: &StreamSeeds) {
    write_u64(out, seeds.master_seed.as_u64());
    write_u64(out, seeds.derived.len() as u64);
    for (name, &seed) in &seeds.derived {
        write_string(out, name);
        write_u64(out, seed);
    }
}

fn write_terminal(out: &mut Vec<u8>, terminal: Terminal) {
    match terminal {
        Terminal::Ongoing => write_u8(out, 0),
        Terminal::Terminated(reason) => {
            write_u8(out, 1);
            let tag = match reason {
                TerminalReason::GoalReached => 0,
                TerminalReason::Failure => 1,
                TerminalReason::HorizonReached => 2,
                TerminalReason::Cancelled => 3,
                TerminalReason::InvariantViolated => 4,
            };
            write_u8(out, tag);
        }
    }
}

fn write_checkpoint(out: &mut Vec<u8>, cp: &CheckpointRef) {
    let snap = &cp.snapshot;
    write_u64(out, snap.state().tick().as_u64());
    write_f64(out, snap.state().elapsed().as_secs_f64());
    write_f64_slice(out, snap.state().values());
    write_u64(out, snap.rng_state());
    write_terminal(out, snap.terminal());
    write_u64(out, snap.tick().as_u64());
    write_f64(out, snap.elapsed().as_secs_f64());
    write_u64(out, cp.annotations.len() as u64);
    for (k, v) in &cp.annotations {
        write_string(out, k);
        write_string(out, v);
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_slice(&mut self, len: usize) -> Result<&'a [u8], ScientificTraceError> {
        let end = self.pos.checked_add(len).ok_or_else(|| {
            ScientificTraceError::CorruptedFormat("length overflow while reading slice".to_owned())
        })?;
        if end > self.bytes.len() {
            return Err(ScientificTraceError::CorruptedFormat(
                "unexpected end of buffer".to_owned(),
            ));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_exact<const N: usize>(&mut self) -> Result<[u8; N], ScientificTraceError> {
        let slice = self.read_slice(N)?;
        let mut buf = [0u8; N];
        buf.copy_from_slice(slice);
        Ok(buf)
    }

    fn read_u8(&mut self) -> Result<u8, ScientificTraceError> {
        Ok(self.read_exact::<1>()?[0])
    }

    fn read_bool(&mut self) -> Result<bool, ScientificTraceError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            other => Err(ScientificTraceError::CorruptedFormat(format!(
                "invalid boolean byte: {other}"
            ))),
        }
    }

    fn read_u32(&mut self) -> Result<u32, ScientificTraceError> {
        Ok(u32::from_le_bytes(self.read_exact::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64, ScientificTraceError> {
        Ok(u64::from_le_bytes(self.read_exact::<8>()?))
    }

    fn read_f64(&mut self) -> Result<f64, ScientificTraceError> {
        Ok(f64::from_bits(self.read_u64()?))
    }

    fn read_len(&mut self) -> Result<usize, ScientificTraceError> {
        let raw = self.read_u64()?;
        usize::try_from(raw).map_err(|_| {
            ScientificTraceError::CorruptedFormat("length exceeds usize::MAX".to_owned())
        })
    }

    fn read_string(&mut self) -> Result<String, ScientificTraceError> {
        let len = self.read_len()?;
        let slice = self.read_slice(len)?;
        std::str::from_utf8(slice)
            .map(str::to_owned)
            .map_err(|err| ScientificTraceError::CorruptedFormat(err.to_string()))
    }

    fn read_f64_vec(&mut self) -> Result<Vec<f64>, ScientificTraceError> {
        let len = self.read_len()?;
        let mut out = Vec::with_capacity(len.min(4096));
        for _ in 0..len {
            out.push(self.read_f64()?);
        }
        Ok(out)
    }

    fn ensure_eof(&self) -> Result<(), ScientificTraceError> {
        if self.pos != self.bytes.len() {
            return Err(ScientificTraceError::CorruptedFormat(format!(
                "trailing {} bytes after end of record",
                self.bytes.len() - self.pos
            )));
        }
        Ok(())
    }
}

fn read_sim_duration(reader: &mut ByteReader<'_>) -> Result<SimDuration, ScientificTraceError> {
    let secs = reader.read_f64()?;
    SimDuration::from_secs_f64(secs)
        .map_err(|err| ScientificTraceError::CorruptedFormat(err.to_string()))
}

fn read_observation(reader: &mut ByteReader<'_>) -> Result<Observation, ScientificTraceError> {
    let tick = Tick::new(reader.read_u64()?);
    let elapsed = read_sim_duration(reader)?;
    let features = reader.read_f64_vec()?;
    Observation::new(tick, elapsed, features)
        .map_err(|err| ScientificTraceError::CorruptedFormat(err.to_string()))
}

fn read_action(reader: &mut ByteReader<'_>) -> Result<Action, ScientificTraceError> {
    match reader.read_u8()? {
        0 => {
            let raw_idx = reader.read_u64()?;
            let idx = usize::try_from(raw_idx).map_err(|_| {
                ScientificTraceError::CorruptedFormat("discrete action index overflow".to_owned())
            })?;
            let cardinality = idx.checked_add(1).ok_or_else(|| {
                ScientificTraceError::CorruptedFormat(
                    "discrete action index cannot be usize::MAX".to_owned(),
                )
            })?;
            let discrete = DiscreteAction::new(idx, cardinality)
                .map_err(|err| ScientificTraceError::CorruptedFormat(err.to_string()))?;
            Ok(Action::Discrete(discrete))
        }
        1 => {
            let vals = reader.read_f64_vec()?;
            Action::continuous(vals)
                .map_err(|err| ScientificTraceError::CorruptedFormat(err.to_string()))
        }
        other => Err(ScientificTraceError::CorruptedFormat(format!(
            "unknown Action tag: {other}"
        ))),
    }
}

fn read_opt_action(reader: &mut ByteReader<'_>) -> Result<Option<Action>, ScientificTraceError> {
    if reader.read_bool()? {
        Ok(Some(read_action(reader)?))
    } else {
        Ok(None)
    }
}

fn read_decision_action(
    reader: &mut ByteReader<'_>,
) -> Result<DecisionAction, ScientificTraceError> {
    let plan_status = match reader.read_u8()? {
        0 => PlanStatus::Completed,
        1 => PlanStatus::BudgetExhausted,
        2 => PlanStatus::Cancelled,
        3 => PlanStatus::NoValidAction,
        other => {
            return Err(ScientificTraceError::CorruptedFormat(format!(
                "unknown PlanStatus tag: {other}"
            )));
        }
    };
    let proposed_action = read_opt_action(reader)?;
    let guard_decision = match reader.read_u8()? {
        0 => RecordedGuardDecision::Allow(read_action(reader)?),
        1 => RecordedGuardDecision::Fallback(read_action(reader)?),
        2 => RecordedGuardDecision::NoOp,
        3 => RecordedGuardDecision::Reject(reader.read_string()?),
        other => {
            return Err(ScientificTraceError::CorruptedFormat(format!(
                "unknown RecordedGuardDecision tag: {other}"
            )));
        }
    };
    let applied_action = read_opt_action(reader)?;
    let used_fallback = reader.read_bool()?;
    Ok(DecisionAction {
        plan_status,
        proposed_action,
        guard_decision,
        applied_action,
        used_fallback,
    })
}

fn read_stream_seeds(reader: &mut ByteReader<'_>) -> Result<StreamSeeds, ScientificTraceError> {
    let master_seed = MasterSeed::new(reader.read_u64()?);
    let count = reader.read_len()?;
    let mut derived = BTreeMap::new();
    for _ in 0..count {
        let name = reader.read_string()?;
        let seed = reader.read_u64()?;
        derived.insert(name, seed);
    }
    Ok(StreamSeeds {
        master_seed,
        derived,
    })
}

fn read_terminal(reader: &mut ByteReader<'_>) -> Result<Terminal, ScientificTraceError> {
    match reader.read_u8()? {
        0 => Ok(Terminal::Ongoing),
        1 => {
            let reason = match reader.read_u8()? {
                0 => TerminalReason::GoalReached,
                1 => TerminalReason::Failure,
                2 => TerminalReason::HorizonReached,
                3 => TerminalReason::Cancelled,
                4 => TerminalReason::InvariantViolated,
                other => {
                    return Err(ScientificTraceError::CorruptedFormat(format!(
                        "unknown TerminalReason tag: {other}"
                    )));
                }
            };
            Ok(Terminal::Terminated(reason))
        }
        other => Err(ScientificTraceError::CorruptedFormat(format!(
            "unknown Terminal tag: {other}"
        ))),
    }
}

fn read_checkpoint(reader: &mut ByteReader<'_>) -> Result<CheckpointRef, ScientificTraceError> {
    let state_tick = Tick::new(reader.read_u64()?);
    let state_elapsed = read_sim_duration(reader)?;
    let state_values = reader.read_f64_vec()?;
    let state = gogamen_core::State::new(state_tick, state_elapsed, state_values)
        .map_err(|err| ScientificTraceError::InvalidSnapshot(err.to_string()))?;
    let rng_state = reader.read_u64()?;
    let terminal = read_terminal(reader)?;
    let snap_tick = Tick::new(reader.read_u64()?);
    let snap_elapsed = read_sim_duration(reader)?;
    let snapshot =
        SimulationSnapshot::from_parts(state, rng_state, terminal, snap_tick, snap_elapsed)
            .map_err(|err| ScientificTraceError::InvalidSnapshot(err.to_string()))?;

    let ann_count = reader.read_len()?;
    let mut annotations = BTreeMap::new();
    for _ in 0..ann_count {
        let k = reader.read_string()?;
        let v = reader.read_string()?;
        annotations.insert(k, v);
    }
    Ok(CheckpointRef::with_annotations(snapshot, annotations))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LineWorld1DConfig, run_1d_episode};
    use gogamen_core::{DeterministicRng, State, StepDt, StreamKey};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        pub(super) static RECOMPUTE_COUNT: Cell<usize> = const { Cell::new(0) };
        static ALLOC_ARMED: Cell<bool> = const { Cell::new(false) };
        static ALLOC_COUNT: Cell<usize> = const { Cell::new(0) };
    }

    /// Allocator that counts allocations only on threads that armed `ALLOC_ARMED`.
    struct CountingAlloc;

    // SAFETY: delegates every operation to `System`; the counters are thread-local `Cell`s
    // with const initialisers (no allocation, no destructor).
    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ALLOC_ARMED.with(Cell::get) {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            if ALLOC_ARMED.with(Cell::get) {
                ALLOC_COUNT.with(|c| c.set(c.get() + 1));
            }
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    #[global_allocator]
    static GLOBAL: CountingAlloc = CountingAlloc;

    /// Legacy two-buffer encoder, kept verbatim as the byte-for-byte reference.
    fn legacy_to_bytes(rec: &ScientificRecord, policy: &SensitiveFieldPolicy) -> Vec<u8> {
        let clean = rec.sanitized(policy).unwrap();
        let mut payload = Vec::new();
        write_u64(&mut payload, clean.revision.as_u64());
        write_u64(&mut payload, clean.inputs.tick.as_u64());
        write_observation(&mut payload, &clean.inputs.observation);
        write_decision_action(&mut payload, &clean.decision);
        write_stream_seeds(&mut payload, &clean.stream_seeds);
        match &clean.checkpoint {
            Some(cp) => {
                write_u8(&mut payload, 1);
                write_checkpoint(&mut payload, cp);
            }
            None => write_u8(&mut payload, 0),
        }
        let mut hasher = TraceHasher::new();
        hasher.write_u64(u64::from(SCIENTIFIC_TRACE_VERSION));
        hasher.write_bytes(&payload);
        let checksum = hasher.finish();
        let mut out = Vec::with_capacity(8 + 4 + payload.len() + 8);
        out.extend_from_slice(TRACE_MAGIC);
        write_u32(&mut out, SCIENTIFIC_TRACE_VERSION);
        write_u64(&mut out, payload.len() as u64);
        out.extend_from_slice(&payload);
        write_u64(&mut out, checksum);
        out
    }

    fn sample_snapshot(tick: u64, pos: f64, rng_state: u64) -> SimulationSnapshot {
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let elapsed = SimDuration::from_secs_f64(dt.as_secs_f64() * tick as f64).unwrap();
        let state = State::new(Tick::new(tick), elapsed, vec![pos, 0.0]).unwrap();
        let rng = DeterministicRng::new(rng_state);
        SimulationSnapshot::new(state, &rng, Terminal::Ongoing)
    }

    fn sample_record(
        rev: u64,
        tick: u64,
        seed: MasterSeed,
        checkpoint: Option<CheckpointRef>,
    ) -> ScientificRecord {
        let elapsed = SimDuration::from_secs_f64(0.1 * tick as f64).unwrap();
        let obs = Observation::new(Tick::new(tick), elapsed, vec![tick as f64 * 0.5]).unwrap();
        let action = Action::continuous(vec![0.25]).unwrap();
        ScientificRecord {
            revision: StreamRevision::new(rev),
            inputs: TickInputs::new(Tick::new(tick), obs),
            decision: DecisionAction {
                plan_status: PlanStatus::Completed,
                proposed_action: Some(action.clone()),
                guard_decision: RecordedGuardDecision::Allow(action.clone()),
                applied_action: Some(action),
                used_fallback: false,
            },
            stream_seeds: StreamSeeds::new(seed)
                .with_derived("episode", seed.derive_seed(StreamKey::episode(0))),
            checkpoint,
        }
    }

    #[test]
    fn fixture_round_trip_reconstructs_ticks_actions_checkpoints_and_manifest() {
        let seed = MasterSeed::new(77);
        let episode = run_1d_episode(10, seed).unwrap();
        let config = TraceManifestConfig::new(
            RunId::new(1001),
            "spec-line1d-v1",
            "model-receding-h5",
            seed,
        );

        let mut sink = BufferedTraceSink::new(
            config,
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();

        let mut original_records = Vec::new();
        for (idx, entry) in episode.trace.entries().iter().enumerate() {
            let rev = StreamRevision::new((idx + 1) as u64);
            let cp = if idx % 5 == 0 {
                let mut anns = BTreeMap::new();
                anns.insert("phase".to_owned(), format!("step-{idx}"));
                Some(CheckpointRef::with_annotations(
                    sample_snapshot(entry.tick.as_u64(), idx as f64 * 0.1, 900 + idx as u64),
                    anns,
                ))
            } else {
                None
            };
            let seeds = StreamSeeds::new(seed)
                .with_derived("planner", seed.derive_seed(StreamKey::agent(0, 1)));
            let record = ScientificRecord::from_trace_entry(rev, entry, seeds, cp);
            assert_eq!(sink.record(&record).unwrap(), EnqueueOutcome::Enqueued);
            original_records.push(record);
        }

        let mut backend = InMemoryTraceBackend::new();
        let manifest = sink.finalize(&mut backend).unwrap();

        assert_eq!(manifest.version, SCIENTIFIC_TRACE_VERSION);
        assert_eq!(manifest.run_id, RunId::new(1001));
        assert_eq!(manifest.build_version, gogamen_core::core_version());
        assert_eq!(manifest.spec_id, "spec-line1d-v1");
        assert_eq!(manifest.model_id, "model-receding-h5");
        assert_eq!(manifest.seed, seed);
        assert!(manifest.is_complete);
        assert_eq!(manifest.total_records, 10);
        assert_eq!(manifest.dropped_records, 0);
        assert!(manifest.gaps.is_empty());
        manifest.verify().unwrap();

        let decoded_manifest = backend.decode_manifest().unwrap().unwrap();
        assert_eq!(decoded_manifest, manifest);

        let decoded_records = backend.decode_records().unwrap();
        assert_eq!(decoded_records, original_records);

        // Verifica también round-trip completo de ScientificTraceArchive.
        let archive = ScientificTraceArchive {
            manifest,
            records: original_records,
        };
        let bytes = archive.to_bytes(&SensitiveFieldPolicy::default()).unwrap();
        let restored = ScientificTraceArchive::from_bytes(&bytes).unwrap();
        assert_eq!(restored, archive);
        let world_cfg = LineWorld1DConfig::default();
        assert!(world_cfg.dt.as_secs_f64() > 0.0);
    }

    #[test]
    fn detects_corrupted_record_or_manifest_checksum() {
        let seed = MasterSeed::new(5);
        let rec = sample_record(
            1,
            1,
            seed,
            Some(CheckpointRef::new(sample_snapshot(1, 0.2, 42))),
        );
        let mut bytes = rec.to_bytes().unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0xFF;
        let err = ScientificRecord::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, ScientificTraceError::ChecksumMismatch { .. }));

        let mut manifest = TraceManifest::new(TraceManifestConfig::new(
            RunId::new(1),
            "spec",
            "model",
            seed,
        ));
        manifest.dropped_records = 1;
        // Fuerza inconsistencia simulada de completitud.
        manifest.is_complete = true;
        assert_eq!(
            manifest.verify().unwrap_err(),
            ScientificTraceError::InconsistentCompleteness
        );
    }

    #[test]
    fn overflow_mark_incomplete_never_blocks_and_records_gaps() {
        let seed = MasterSeed::new(11);
        let quota = TraceQuota::new(2, 64 * 1024).unwrap();
        let mut sink = BufferedTraceSink::new(
            TraceManifestConfig::new(RunId::new(2), "spec-ov", "model-ov", seed),
            quota,
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();

        assert_eq!(
            sink.record(&sample_record(1, 1, seed, None)).unwrap(),
            EnqueueOutcome::Enqueued
        );
        assert_eq!(
            sink.record(&sample_record(2, 2, seed, None)).unwrap(),
            EnqueueOutcome::Enqueued
        );
        // El tercero desborda la cola acotada de 2 registros sin bloquear.
        assert_eq!(
            sink.record(&sample_record(3, 3, seed, None)).unwrap(),
            EnqueueOutcome::DroppedMarkIncomplete
        );

        let mut backend = InMemoryTraceBackend::new();
        let manifest = sink.finalize(&mut backend).unwrap();
        assert!(!manifest.is_complete);
        assert_eq!(manifest.total_records, 2);
        assert_eq!(manifest.dropped_records, 1);
        assert_eq!(manifest.gaps.len(), 1);
        assert_eq!(manifest.gaps[0].from_revision, StreamRevision::new(3));
        assert_eq!(manifest.gaps[0].reason, TraceGapReason::QueueOverflow);
    }

    #[test]
    fn overflow_abort_experiment_stops_explicitly() {
        let seed = MasterSeed::new(12);
        let quota = TraceQuota::new(1, 64 * 1024).unwrap();
        let mut sink = BufferedTraceSink::new(
            TraceManifestConfig::new(RunId::new(3), "spec-abort", "model-abort", seed),
            quota,
            OverflowPolicy::AbortExperiment,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();

        sink.record(&sample_record(1, 1, seed, None)).unwrap();
        let err = sink.record(&sample_record(2, 2, seed, None)).unwrap_err();
        assert!(matches!(err, TraceWriteError::ExperimentAborted { .. }));
        assert!(!sink.manifest().is_complete);
        assert_eq!(sink.manifest().dropped_records, 1);
    }

    #[test]
    fn async_writer_handles_disk_failure_according_to_policy() {
        let seed = MasterSeed::new(99);
        // Caso 1: MarkIncomplete ante DiskFailure tras 1 escritura exitosa.
        let backend_mark = InMemoryTraceBackend::new().with_failure_after_writes(1);
        let writer_mark = AsyncTraceWriter::start(
            TraceManifestConfig::new(RunId::new(10), "spec-df", "model-df", seed),
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
            Box::new(backend_mark),
        )
        .unwrap();

        writer_mark
            .try_record(&sample_record(1, 1, seed, None))
            .unwrap();
        writer_mark
            .try_record(&sample_record(2, 2, seed, None))
            .unwrap();
        let manifest = writer_mark.finish().unwrap();
        assert!(!manifest.is_complete);
        assert!(manifest.dropped_records >= 1);
        assert!(
            manifest
                .gaps
                .iter()
                .any(|g| matches!(g.reason, TraceGapReason::DiskFailure(_)))
        );

        // Caso 2: AbortExperiment ante DiskFailure.
        let backend_abort = InMemoryTraceBackend::new().with_failure_after_writes(0);
        let writer_abort = AsyncTraceWriter::start(
            TraceManifestConfig::new(RunId::new(11), "spec-df-abort", "model-df", seed),
            TraceQuota::default(),
            OverflowPolicy::AbortExperiment,
            SensitiveFieldPolicy::default(),
            Box::new(backend_abort),
        )
        .unwrap();
        let _ = writer_abort.try_record(&sample_record(1, 1, seed, None));
        let err = writer_abort.finish().unwrap_err();
        assert!(matches!(err, TraceWriteError::ExperimentAborted { .. }));
    }

    #[test]
    fn sensitive_field_policy_rejects_or_redacts_credentials_in_checkpoints() {
        let seed = MasterSeed::new(42);
        let mut anns = BTreeMap::new();
        anns.insert("api_key".to_owned(), "sk-live-secret-12345".to_owned());
        anns.insert("checkpoint_tag".to_owned(), "nominal".to_owned());

        let cp = CheckpointRef::with_annotations(sample_snapshot(5, 1.0, 123), anns);
        let rec = sample_record(1, 5, seed, Some(cp));

        // Modo Reject por defecto impide serializar el checkpoint con credenciales.
        let err = rec.to_bytes().unwrap_err();
        assert!(matches!(
            err,
            ScientificTraceError::SensitiveDataDetected { .. }
        ));

        // Modo Redact sustituye la credencial por "[REDACTED]" antes de persistir.
        let redact_policy = SensitiveFieldPolicy::new(SensitiveFieldMode::Redact);
        let bytes = rec.to_bytes_with_policy(&redact_policy).unwrap();
        let raw_text = String::from_utf8_lossy(&bytes);
        assert!(!raw_text.contains("sk-live-secret-12345"));

        let decoded = ScientificRecord::from_bytes(&bytes).unwrap();
        let decoded_cp = decoded.checkpoint.unwrap();
        assert_eq!(
            decoded_cp.annotations.get("api_key").map(String::as_str),
            Some(REDACTED_VALUE)
        );
        assert_eq!(
            decoded_cp
                .annotations
                .get("checkpoint_tag")
                .map(String::as_str),
            Some("nominal")
        );
    }

    #[test]
    fn quota_enforces_byte_limits_and_file_backend_persists_cleanly() {
        let seed = MasterSeed::new(33);
        let rec1 = sample_record(1, 1, seed, None);
        let rec1_len = rec1.to_bytes().unwrap().len();

        // Cuota de bytes que solo permite 1 registro en cola.
        let quota = TraceQuota::new(16, rec1_len).unwrap();
        let mut sink = BufferedTraceSink::new(
            TraceManifestConfig::new(RunId::new(77), "spec-bytes", "model-bytes", seed),
            quota,
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();

        assert_eq!(sink.record(&rec1).unwrap(), EnqueueOutcome::Enqueued);
        assert_eq!(
            sink.record(&sample_record(2, 2, seed, None)).unwrap(),
            EnqueueOutcome::DroppedMarkIncomplete
        );

        let dir =
            std::env::temp_dir().join(format!("gogamen_sci_trace_test_{}", std::process::id()));
        let mut file_backend = FileTraceBackend::create(&dir).unwrap();
        let manifest = sink.finalize(&mut file_backend).unwrap();
        assert!(!manifest.is_complete);
        assert_eq!(manifest.total_records, 1);
        assert_eq!(manifest.dropped_records, 1);
        assert!(dir.join("records.gtrace").exists());
        assert!(dir.join("manifest.gmanif").exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn single_pass_encoding_is_byte_identical_to_legacy_and_round_trips() {
        let seed = MasterSeed::new(5);
        let policy = SensitiveFieldPolicy::default();
        let mut anns = BTreeMap::new();
        anns.insert("note".to_owned(), "hello".to_owned());
        let cp = CheckpointRef::with_annotations(sample_snapshot(3, 0.7, 9), anns);
        for rec in [
            sample_record(1, 1, seed, None),
            sample_record(2, 2, seed, Some(cp)),
        ] {
            let fast = rec.to_bytes_with_policy(&policy).unwrap();
            assert_eq!(fast, legacy_to_bytes(&rec, &policy));
            assert_eq!(fast.capacity(), fast.len(), "exact-size single buffer");
            assert_eq!(ScientificRecord::from_bytes(&fast).unwrap(), rec);
        }
    }

    #[test]
    fn single_pass_encoding_performs_exactly_one_allocation() {
        let seed = MasterSeed::new(6);
        let rec = sample_record(1, 1, seed, None);
        let policy = SensitiveFieldPolicy::default();
        ALLOC_COUNT.with(|c| c.set(0));
        ALLOC_ARMED.with(|a| a.set(true));
        let bytes = rec.to_bytes_with_policy(&policy);
        ALLOC_ARMED.with(|a| a.set(false));
        let bytes = bytes.unwrap();
        assert_eq!(ALLOC_COUNT.with(Cell::get), 1);
        assert!(!bytes.is_empty());
    }

    #[test]
    fn flush_to_recomputes_manifest_checksum_once_per_batch() {
        let seed = MasterSeed::new(21);
        let mut sink = BufferedTraceSink::new(
            TraceManifestConfig::new(RunId::new(21), "spec", "model", seed),
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();
        for i in 1..=5u64 {
            sink.record(&sample_record(i, i, seed, None)).unwrap();
        }
        let mut backend = InMemoryTraceBackend::new();
        RECOMPUTE_COUNT.with(|c| c.set(0));
        assert_eq!(sink.flush_to(&mut backend).unwrap(), 5);
        assert_eq!(RECOMPUTE_COUNT.with(Cell::get), 1);
        assert_eq!(sink.manifest().total_records, 5);
        sink.manifest().verify().unwrap();
    }

    #[test]
    fn try_record_does_not_drop_on_mutex_contention() {
        let seed = MasterSeed::new(31);
        let writer = Arc::new(
            AsyncTraceWriter::start(
                TraceManifestConfig::new(RunId::new(31), "spec", "model", seed),
                TraceQuota::default(),
                OverflowPolicy::MarkIncomplete,
                SensitiveFieldPolicy::default(),
                Box::new(InMemoryTraceBackend::new()),
            )
            .unwrap(),
        );

        let holder = Arc::clone(&writer);
        let (tx, rx) = std::sync::mpsc::channel();
        let t = std::thread::spawn(move || {
            let _held = holder.shared.0.lock().unwrap();
            tx.send(()).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(100));
        });
        rx.recv().unwrap();
        // The mutex is busy: the producer waits and the record is NOT dropped.
        assert_eq!(
            writer.try_record(&sample_record(1, 1, seed, None)).unwrap(),
            EnqueueOutcome::Enqueued
        );
        t.join().unwrap();
        let writer = Arc::try_unwrap(writer).unwrap();
        let manifest = writer.finish().unwrap();
        assert!(manifest.is_complete);
        assert_eq!(manifest.dropped_records, 0);
        assert_eq!(manifest.total_records, 1);
        assert!(manifest.gaps.is_empty());
        manifest.verify().unwrap();
    }

    #[test]
    fn async_writer_manifest_matches_sync_sink_manifest() {
        let seed = MasterSeed::new(41);
        let cfg = || TraceManifestConfig::new(RunId::new(41), "spec", "model", seed);
        let mut sink = BufferedTraceSink::new(
            cfg(),
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )
        .unwrap();
        let writer = AsyncTraceWriter::start(
            cfg(),
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
            Box::new(InMemoryTraceBackend::new()),
        )
        .unwrap();
        for i in 1..=4u64 {
            let rec = sample_record(i, i, seed, None);
            sink.record(&rec).unwrap();
            writer.try_record(&rec).unwrap();
        }
        let expected = sink.finalize(&mut InMemoryTraceBackend::new()).unwrap();
        let actual = writer.finish().unwrap();
        assert_eq!(actual, expected);
    }
}
