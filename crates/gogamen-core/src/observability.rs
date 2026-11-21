//! Contrato común de observabilidad independiente de UI (`std`-only).
//!
//! Este módulo define los tipos de procedencia, separación entre frames visuales
//! descartables ([`VisualFrame`]) y eventos científicos acotados ([`ScientificTraceEvent`]),
//! y canales/streams acotados ([`VisualFrameRing`], [`BoundedObservationStream`],
//! [`ObservationHub`]) sin dependencias de Bevy/egui ni campos de texto libre humano o secretos.
//!
//! # Invariantes clave
//! - **Publicación solo con observadores:** si `observer_count() == 0`, los métodos
//!   `publish_*_if_observed` no ejecutan el constructor perezoso ni almacenan datos.
//! - **Consumidor lento sin bloqueo ni cola ilimitada:** los buffers tienen capacidad fija
//!   (`capacity >= 1`). Ante saturación se descarta el elemento más antiguo en `O(1)`
//!   incrementando contadores explícitos (`dropped_frames`, `dropped_events`) y marcando
//!   pérdida en los eventos científicos ([`SaturationLossMarker`]).
//! - **Separación snapshot vs frame visual:** [`VisualFrame`] es explícitamente descartable
//!   y **no** es apto como checkpoint completo de simulación (`is_checkpoint_capable() == false`).

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use crate::clock::MonotonicInstant;
use crate::units::{SimDuration, Tick};

/// Versión de esquema actual del contrato de observabilidad.
pub const OBSERVABILITY_SCHEMA_VERSION: u32 = 1;

/// Límite máximo de valores numéricos por cada [`ScientificTraceEvent`].
pub const MAX_EVENT_VALUES: usize = 32;

/// Identificador único de una ejecución o episodio observado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(u64);

impl RunId {
    /// Crea un [`RunId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "run({})", self.0)
    }
}

/// Revisión monotónica de un stream de observabilidad dentro de una ejecución.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct StreamRevision(u64);

impl StreamRevision {
    /// Revisión inicial (`0`).
    pub const ZERO: Self = Self(0);

    /// Crea una [`StreamRevision`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Devuelve la revisión siguiente (`self + 1`), fallando si hay desbordamiento.
    pub const fn checked_next(self) -> Option<Self> {
        match self.0.checked_add(1) {
            Some(next) => Some(Self(next)),
            None => None,
        }
    }
}

impl fmt::Display for StreamRevision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rev({})", self.0)
    }
}

/// Identificador numérico estructurado de una entidad observada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId(u64);

impl EntityId {
    /// Crea un [`EntityId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "entity({})", self.0)
    }
}

/// Procedencia estructurada de un frame o evento sin texto libre humano.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EventSource {
    /// Mundo o simulación principal.
    SimulationWorld,
    /// Entorno vivo / sensores / actuadores reales.
    LiveEnvironment,
    /// Modelo generativo interno del planificador.
    GenerativeModel,
    /// Subsistema de planificación.
    Planner,
    /// Planificador de tareas / scheduler determinista.
    Scheduler,
    /// Guardia o política de seguridad de acciones.
    Guard,
    /// Módulo externo identificado únicamente por código numérico.
    Module(u64),
}

/// Clasificación epistémica explícita del dato observado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicKind {
    /// Dato observado directamente de la ejecución efectiva.
    Observed,
    /// Dato inferido o estimado a partir de observaciones parciales.
    Inferred,
    /// Dato hipotético generado por rollouts, búsqueda o ramas especulativas.
    Hypothetical,
}

impl EpistemicKind {
    /// Indica si el dato es directamente observado.
    pub const fn is_observed(self) -> bool {
        matches!(self, Self::Observed)
    }

    /// Indica si el dato es inferido.
    pub const fn is_inferred(self) -> bool {
        matches!(self, Self::Inferred)
    }

    /// Indica si el dato es hipotético.
    pub const fn is_hypothetical(self) -> bool {
        matches!(self, Self::Hypothetical)
    }
}

/// Errores explícitos de validación y operación en observabilidad.
#[derive(Debug, Clone, PartialEq)]
pub enum ObservabilityError {
    /// La versión de esquema recibida no coincide con [`OBSERVABILITY_SCHEMA_VERSION`].
    UnsupportedSchemaVersion {
        /// Versión esperada.
        expected: u32,
        /// Versión recibida.
        actual: u32,
    },
    /// Capacidad de buffer inválida (debe ser `>= 1`).
    InvalidCapacity {
        /// Capacidad recibida.
        capacity: usize,
    },
    /// El identificador de ejecución no coincide con el configurado en el stream.
    RunIdMismatch {
        /// Identificador esperado.
        expected: RunId,
        /// Identificador recibido.
        actual: RunId,
    },
    /// El par `(tick, revision)` no avanzó de forma estrictamente monotónica.
    NonMonotonicOrder {
        /// Tick previo aceptado.
        prev_tick: Tick,
        /// Revisión previa aceptada.
        prev_revision: StreamRevision,
        /// Tick recibido.
        next_tick: Tick,
        /// Revisión recibida.
        next_revision: StreamRevision,
    },
    /// El reloj monotónico retrocedió respecto a la muestra anterior.
    ClockRegression {
        /// Instante monotónico previo.
        prev: MonotonicInstant,
        /// Instante monotónico recibido.
        next: MonotonicInstant,
    },
    /// Un valor numérico continuo no es finito (`NaN`, `+inf` o `-inf`).
    NonFiniteValue {
        /// Campo donde se detectó el valor no finito.
        field: &'static str,
        /// Valor inválido.
        value: f64,
    },
    /// El número de valores de un [`ScientificTraceEvent`] supera [`MAX_EVENT_VALUES`].
    EventValuesLimitExceeded {
        /// Máximo permitido.
        max: usize,
        /// Cantidad recibida.
        actual: usize,
    },
}

impl fmt::Display for ObservabilityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion { expected, actual } => {
                write!(
                    f,
                    "unsupported observability schema version {actual} (expected {expected})"
                )
            }
            Self::InvalidCapacity { capacity } => {
                write!(
                    f,
                    "invalid bounded stream capacity {capacity}: must be >= 1"
                )
            }
            Self::RunIdMismatch { expected, actual } => {
                write!(f, "run id mismatch: expected {expected}, got {actual}")
            }
            Self::NonMonotonicOrder {
                prev_tick,
                prev_revision,
                next_tick,
                next_revision,
            } => {
                write!(
                    f,
                    "non-monotonic stream order: ({next_tick}, {next_revision}) must be > ({prev_tick}, {prev_revision})"
                )
            }
            Self::ClockRegression { prev, next } => {
                write!(
                    f,
                    "monotonic clock regression: {next} is earlier than {prev}"
                )
            }
            Self::NonFiniteValue { field, value } => {
                write!(f, "non-finite value in '{field}': {value}")
            }
            Self::EventValuesLimitExceeded { max, actual } => {
                write!(
                    f,
                    "scientific trace event values count {actual} exceeds bounded limit {max}"
                )
            }
        }
    }
}

impl Error for ObservabilityError {}

fn validate_finite(field: &'static str, value: f64) -> Result<(), ObservabilityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ObservabilityError::NonFiniteValue { field, value })
    }
}

/// Muestra visual numérica de una entidad individual dentro de un [`VisualFrame`].
///
/// No contiene texto libre humano ni secretos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VisualEntitySample {
    entity_id: EntityId,
    position: [f64; 3],
    scalar: f64,
    state_code: u32,
}

impl VisualEntitySample {
    /// Construye una muestra visual validando que todas sus coordenadas sean finitas.
    pub fn new(
        entity_id: EntityId,
        position: [f64; 3],
        scalar: f64,
        state_code: u32,
    ) -> Result<Self, ObservabilityError> {
        for coord in position {
            validate_finite("visual_entity.position", coord)?;
        }
        validate_finite("visual_entity.scalar", scalar)?;
        Ok(Self {
            entity_id,
            position,
            scalar,
            state_code,
        })
    }

    /// Identificador de la entidad.
    pub const fn entity_id(&self) -> EntityId {
        self.entity_id
    }

    /// Posición espacial `[x, y, z]`.
    pub const fn position(&self) -> [f64; 3] {
        self.position
    }

    /// Magnitud escalar auxiliar para representación visual.
    pub const fn scalar(&self) -> f64 {
        self.scalar
    }

    /// Código numérico de estado visual.
    pub const fn state_code(&self) -> u32 {
        self.state_code
    }

    /// Valida que los valores numéricos sigan siendo finitos.
    pub fn validate(&self) -> Result<(), ObservabilityError> {
        for coord in self.position {
            validate_finite("visual_entity.position", coord)?;
        }
        validate_finite("visual_entity.scalar", self.scalar)
    }
}

/// Snapshot visual descartable para representación/telemetría ligera.
///
/// **Importante:** un [`VisualFrame`] es explícitamente descartable ante consumidores lentos
/// y **NO** es apto como checkpoint completo de simulación ([`VisualFrame::is_checkpoint_capable`]
/// devuelve siempre `false`).
#[derive(Debug, Clone, PartialEq)]
pub struct VisualFrame {
    schema_version: u32,
    run_id: RunId,
    tick: Tick,
    sim_time: SimDuration,
    monotonic_instant: MonotonicInstant,
    revision: StreamRevision,
    source: EventSource,
    epistemic_kind: EpistemicKind,
    entities: Vec<VisualEntitySample>,
}

impl VisualFrame {
    /// Construye un [`VisualFrame`] con la versión de esquema vigente [`OBSERVABILITY_SCHEMA_VERSION`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        tick: Tick,
        sim_time: SimDuration,
        monotonic_instant: MonotonicInstant,
        revision: StreamRevision,
        source: EventSource,
        epistemic_kind: EpistemicKind,
        entities: Vec<VisualEntitySample>,
    ) -> Result<Self, ObservabilityError> {
        Self::with_schema_version(
            OBSERVABILITY_SCHEMA_VERSION,
            run_id,
            tick,
            sim_time,
            monotonic_instant,
            revision,
            source,
            epistemic_kind,
            entities,
        )
    }

    /// Construye un [`VisualFrame`] validando explícitamente la versión de esquema.
    #[allow(clippy::too_many_arguments)]
    pub fn with_schema_version(
        schema_version: u32,
        run_id: RunId,
        tick: Tick,
        sim_time: SimDuration,
        monotonic_instant: MonotonicInstant,
        revision: StreamRevision,
        source: EventSource,
        epistemic_kind: EpistemicKind,
        entities: Vec<VisualEntitySample>,
    ) -> Result<Self, ObservabilityError> {
        let frame = Self {
            schema_version,
            run_id,
            tick,
            sim_time,
            monotonic_instant,
            revision,
            source,
            epistemic_kind,
            entities,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Valida versión de esquema y finitud de las muestras visuales.
    pub fn validate(&self) -> Result<(), ObservabilityError> {
        if self.schema_version != OBSERVABILITY_SCHEMA_VERSION {
            return Err(ObservabilityError::UnsupportedSchemaVersion {
                expected: OBSERVABILITY_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        for entity in &self.entities {
            entity.validate()?;
        }
        Ok(())
    }

    /// Indica explícitamente que este frame visual **no** es apto como checkpoint completo de simulación.
    pub const fn is_checkpoint_capable(&self) -> bool {
        false
    }

    /// Indica explícitamente que este frame visual es descartable ante saturación.
    pub const fn is_discardable(&self) -> bool {
        true
    }

    /// Versión de esquema del frame.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Identificador de ejecución.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Tick lógico asociado.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Tiempo físico acumulado de simulación.
    pub const fn sim_time(&self) -> SimDuration {
        self.sim_time
    }

    /// Instante monotónico de emisión.
    pub const fn monotonic_instant(&self) -> MonotonicInstant {
        self.monotonic_instant
    }

    /// Revisión monotónica del stream.
    pub const fn revision(&self) -> StreamRevision {
        self.revision
    }

    /// Procedencia estructurada del frame.
    pub const fn source(&self) -> EventSource {
        self.source
    }

    /// Clasificación epistémica (`Observed`, `Inferred`, `Hypothetical`).
    pub const fn epistemic_kind(&self) -> EpistemicKind {
        self.epistemic_kind
    }

    /// Muestras visuales de entidades incluidas en el frame.
    pub fn entities(&self) -> &[VisualEntitySample] {
        &self.entities
    }
}

/// Marca explícita de pérdida por saturación en un [`ScientificTraceEvent`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SaturationLossMarker {
    /// Número de eventos previos descartados por saturación antes de este evento.
    pub dropped_events: u64,
}

/// Evento científico acotado para análisis y traza estructurada sin texto libre humano.
#[derive(Debug, Clone, PartialEq)]
pub struct ScientificTraceEvent {
    schema_version: u32,
    run_id: RunId,
    tick: Tick,
    sim_time: SimDuration,
    monotonic_instant: MonotonicInstant,
    revision: StreamRevision,
    source: EventSource,
    epistemic_kind: EpistemicKind,
    entity_id: Option<EntityId>,
    metric_code: u32,
    values: Vec<f64>,
    dropped_before: u64,
}

impl ScientificTraceEvent {
    /// Construye un [`ScientificTraceEvent`] con la versión [`OBSERVABILITY_SCHEMA_VERSION`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        run_id: RunId,
        tick: Tick,
        sim_time: SimDuration,
        monotonic_instant: MonotonicInstant,
        revision: StreamRevision,
        source: EventSource,
        epistemic_kind: EpistemicKind,
        entity_id: Option<EntityId>,
        metric_code: u32,
        values: Vec<f64>,
    ) -> Result<Self, ObservabilityError> {
        Self::with_schema_version(
            OBSERVABILITY_SCHEMA_VERSION,
            run_id,
            tick,
            sim_time,
            monotonic_instant,
            revision,
            source,
            epistemic_kind,
            entity_id,
            metric_code,
            values,
            0,
        )
    }

    /// Construye un [`ScientificTraceEvent`] especificando versión de esquema y marca de pérdida.
    #[allow(clippy::too_many_arguments)]
    pub fn with_schema_version(
        schema_version: u32,
        run_id: RunId,
        tick: Tick,
        sim_time: SimDuration,
        monotonic_instant: MonotonicInstant,
        revision: StreamRevision,
        source: EventSource,
        epistemic_kind: EpistemicKind,
        entity_id: Option<EntityId>,
        metric_code: u32,
        values: Vec<f64>,
        dropped_before: u64,
    ) -> Result<Self, ObservabilityError> {
        let event = Self {
            schema_version,
            run_id,
            tick,
            sim_time,
            monotonic_instant,
            revision,
            source,
            epistemic_kind,
            entity_id,
            metric_code,
            values,
            dropped_before,
        };
        event.validate()?;
        Ok(event)
    }

    /// Valida versión de esquema, cota de longitud y finitud de valores.
    pub fn validate(&self) -> Result<(), ObservabilityError> {
        if self.schema_version != OBSERVABILITY_SCHEMA_VERSION {
            return Err(ObservabilityError::UnsupportedSchemaVersion {
                expected: OBSERVABILITY_SCHEMA_VERSION,
                actual: self.schema_version,
            });
        }
        if self.values.len() > MAX_EVENT_VALUES {
            return Err(ObservabilityError::EventValuesLimitExceeded {
                max: MAX_EVENT_VALUES,
                actual: self.values.len(),
            });
        }
        for &val in &self.values {
            validate_finite("scientific_event.values", val)?;
        }
        Ok(())
    }

    /// Versión de esquema del evento.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Identificador de ejecución.
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Tick lógico asociado.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Tiempo físico acumulado de simulación.
    pub const fn sim_time(&self) -> SimDuration {
        self.sim_time
    }

    /// Instante monotónico de emisión.
    pub const fn monotonic_instant(&self) -> MonotonicInstant {
        self.monotonic_instant
    }

    /// Revisión monotónica del stream.
    pub const fn revision(&self) -> StreamRevision {
        self.revision
    }

    /// Procedencia estructurada del evento.
    pub const fn source(&self) -> EventSource {
        self.source
    }

    /// Clasificación epistémica (`Observed`, `Inferred`, `Hypothetical`).
    pub const fn epistemic_kind(&self) -> EpistemicKind {
        self.epistemic_kind
    }

    /// Entidad opcional asociada al evento.
    pub const fn entity_id(&self) -> Option<EntityId> {
        self.entity_id
    }

    /// Código numérico de métrica o tipo de evento científico.
    pub const fn metric_code(&self) -> u32 {
        self.metric_code
    }

    /// Valores numéricos finitos del evento.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Número de eventos descartados por saturación justo antes de aceptar este evento.
    pub const fn dropped_before(&self) -> u64 {
        self.dropped_before
    }

    /// Devuelve la marca explícita de pérdida si hubo saturación previa.
    pub const fn loss_marker(&self) -> Option<SaturationLossMarker> {
        if self.dropped_before > 0 {
            Some(SaturationLossMarker {
                dropped_events: self.dropped_before,
            })
        } else {
            None
        }
    }

    fn record_prior_loss(&mut self, additional_dropped: u64) {
        self.dropped_before = self.dropped_before.saturating_add(additional_dropped);
    }
}

/// Estado resultante de una publicación condicional en un stream o hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishStatus {
    /// No había observadores activos (`observer_count == 0`); no se construyó ni almacenó nada.
    SkippedNoObservers,
    /// Publicado sin pérdidas.
    Published,
    /// Publicado descartando elementos antiguos por saturación de consumidor lento.
    PublishedWithDrops {
        /// Cantidad de elementos descartados en esta operación o acumulados en el ring.
        dropped_count: u64,
    },
}

/// Métricas agregadas de un stream acotado de observabilidad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StreamMetrics {
    /// Frames visuales actualmente en buffer.
    pub buffered_frames: usize,
    /// Eventos científicos actualmente en buffer.
    pub buffered_events: usize,
    /// Total de frames visuales descartados por saturación.
    pub dropped_frames: u64,
    /// Total de eventos científicos descartados por saturación.
    pub dropped_events: u64,
    /// Total de frames visuales publicados.
    pub published_frames: u64,
    /// Total de eventos científicos publicados.
    pub published_events: u64,
}

#[derive(Debug, Clone, Copy)]
struct StreamOrderTracker {
    run_id: RunId,
    last_order: Option<(Tick, StreamRevision)>,
    last_instant: Option<MonotonicInstant>,
}

impl StreamOrderTracker {
    const fn new(run_id: RunId) -> Self {
        Self {
            run_id,
            last_order: None,
            last_instant: None,
        }
    }

    fn validate_and_advance(
        &mut self,
        run_id: RunId,
        tick: Tick,
        revision: StreamRevision,
        instant: MonotonicInstant,
    ) -> Result<(), ObservabilityError> {
        if run_id != self.run_id {
            return Err(ObservabilityError::RunIdMismatch {
                expected: self.run_id,
                actual: run_id,
            });
        }
        if let Some((prev_tick, prev_revision)) = self.last_order
            && (tick, revision) <= (prev_tick, prev_revision)
        {
            return Err(ObservabilityError::NonMonotonicOrder {
                prev_tick,
                prev_revision,
                next_tick: tick,
                next_revision: revision,
            });
        }
        if let Some(prev_instant) = self.last_instant
            && instant < prev_instant
        {
            return Err(ObservabilityError::ClockRegression {
                prev: prev_instant,
                next: instant,
            });
        }
        self.last_order = Some((tick, revision));
        self.last_instant = Some(instant);
        Ok(())
    }
}

/// Ring buffer acotado de [`VisualFrame`] para consumidores lentos.
///
/// Nunca bloquea al productor ni crece por encima de `capacity`. Cuando está lleno,
/// descarta el frame más antiguo e incrementa [`VisualFrameRing::dropped_frames`].
#[derive(Debug, Clone)]
pub struct VisualFrameRing {
    capacity: usize,
    buffer: VecDeque<VisualFrame>,
    tracker: StreamOrderTracker,
    dropped_frames: u64,
    published_frames: u64,
}

impl VisualFrameRing {
    /// Crea un ring buffer visual de capacidad fija (`capacity >= 1`).
    pub fn new(run_id: RunId, capacity: usize) -> Result<Self, ObservabilityError> {
        if capacity == 0 {
            return Err(ObservabilityError::InvalidCapacity { capacity });
        }
        Ok(Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            tracker: StreamOrderTracker::new(run_id),
            dropped_frames: 0,
            published_frames: 0,
        })
    }

    /// Inserta un [`VisualFrame`] validando esquema, orden y reloj.
    ///
    /// Si el buffer está lleno, descarta el frame más antiguo sin bloquear y
    /// devuelve `Ok(true)` indicando que hubo descarte (`Ok(false)` si no hubo descarte).
    pub fn push(&mut self, frame: VisualFrame) -> Result<bool, ObservabilityError> {
        frame.validate()?;
        self.tracker.validate_and_advance(
            frame.run_id(),
            frame.tick(),
            frame.revision(),
            frame.monotonic_instant(),
        )?;

        let mut dropped = false;
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
            self.dropped_frames = self.dropped_frames.saturating_add(1);
            dropped = true;
        }
        self.buffer.push_back(frame);
        self.published_frames = self.published_frames.saturating_add(1);
        Ok(dropped)
    }

    /// Extrae el frame más antiguo disponible.
    pub fn pop(&mut self) -> Option<VisualFrame> {
        self.buffer.pop_front()
    }

    /// Drena todos los frames actualmente almacenados en orden FIFO.
    pub fn drain_all(&mut self) -> Vec<VisualFrame> {
        self.buffer.drain(..).collect()
    }

    /// Capacidad máxima fija del ring.
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Número de frames actualmente en buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Indica si el ring está vacío.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Total de frames visuales descartados por saturación.
    pub const fn dropped_frames(&self) -> u64 {
        self.dropped_frames
    }

    /// Total de frames visuales aceptados en el ring.
    pub const fn published_frames(&self) -> u64 {
        self.published_frames
    }
}

/// Ring buffer acotado de [`ScientificTraceEvent`] con marca de pérdida por saturación.
#[derive(Debug, Clone)]
struct ScientificEventRing {
    capacity: usize,
    buffer: VecDeque<ScientificTraceEvent>,
    tracker: StreamOrderTracker,
    dropped_events: u64,
    pending_loss: u64,
    published_events: u64,
}

impl ScientificEventRing {
    fn new(run_id: RunId, capacity: usize) -> Result<Self, ObservabilityError> {
        if capacity == 0 {
            return Err(ObservabilityError::InvalidCapacity { capacity });
        }
        Ok(Self {
            capacity,
            buffer: VecDeque::with_capacity(capacity),
            tracker: StreamOrderTracker::new(run_id),
            dropped_events: 0,
            pending_loss: 0,
            published_events: 0,
        })
    }

    fn push(&mut self, mut event: ScientificTraceEvent) -> Result<bool, ObservabilityError> {
        event.validate()?;
        self.tracker.validate_and_advance(
            event.run_id(),
            event.tick(),
            event.revision(),
            event.monotonic_instant(),
        )?;

        let mut dropped = false;
        if self.buffer.len() == self.capacity {
            self.buffer.pop_front();
            self.dropped_events = self.dropped_events.saturating_add(1);
            self.pending_loss = self.pending_loss.saturating_add(1);
            dropped = true;
        }
        if self.pending_loss > 0 {
            event.record_prior_loss(self.pending_loss);
            self.pending_loss = 0;
        }
        self.buffer.push_back(event);
        self.published_events = self.published_events.saturating_add(1);
        Ok(dropped)
    }

    fn pop(&mut self) -> Option<ScientificTraceEvent> {
        self.buffer.pop_front()
    }

    fn drain_all(&mut self) -> Vec<ScientificTraceEvent> {
        self.buffer.drain(..).collect()
    }
}

/// Stream acotado único que combina [`VisualFrameRing`] y eventos científicos acotados,
/// con conteo explícito de observadores para evitar trabajo cuando `observer_count() == 0`.
#[derive(Debug)]
pub struct BoundedObservationStream {
    frames: VisualFrameRing,
    events: ScientificEventRing,
    observer_count: usize,
}

impl BoundedObservationStream {
    /// Crea un stream acotado con capacidades fijas (`>= 1`) para frames y eventos.
    pub fn new(
        run_id: RunId,
        frame_capacity: usize,
        event_capacity: usize,
    ) -> Result<Self, ObservabilityError> {
        Ok(Self {
            frames: VisualFrameRing::new(run_id, frame_capacity)?,
            events: ScientificEventRing::new(run_id, event_capacity)?,
            observer_count: 0,
        })
    }

    /// Número de observadores registrados actualmente.
    pub const fn observer_count(&self) -> usize {
        self.observer_count
    }

    /// Registra un observador activo y devuelve el total actualizado.
    pub fn register_observer(&mut self) -> usize {
        self.observer_count = self.observer_count.saturating_add(1);
        self.observer_count
    }

    /// Da de baja un observador activo (saturando en 0) y devuelve el total actualizado.
    pub fn unregister_observer(&mut self) -> usize {
        self.observer_count = self.observer_count.saturating_sub(1);
        self.observer_count
    }

    /// Construye y publica un [`VisualFrame`] **solo** si `observer_count() > 0`.
    pub fn publish_frame_if_observed<F>(
        &mut self,
        build_frame: F,
    ) -> Result<PublishStatus, ObservabilityError>
    where
        F: FnOnce() -> Result<VisualFrame, ObservabilityError>,
    {
        if self.observer_count == 0 {
            return Ok(PublishStatus::SkippedNoObservers);
        }
        let frame = build_frame()?;
        let dropped = self.frames.push(frame)?;
        if dropped {
            Ok(PublishStatus::PublishedWithDrops {
                dropped_count: self.frames.dropped_frames(),
            })
        } else {
            Ok(PublishStatus::Published)
        }
    }

    /// Construye y publica un [`ScientificTraceEvent`] **solo** si `observer_count() > 0`.
    pub fn publish_event_if_observed<F>(
        &mut self,
        build_event: F,
    ) -> Result<PublishStatus, ObservabilityError>
    where
        F: FnOnce() -> Result<ScientificTraceEvent, ObservabilityError>,
    {
        if self.observer_count == 0 {
            return Ok(PublishStatus::SkippedNoObservers);
        }
        let event = build_event()?;
        let dropped = self.events.push(event)?;
        if dropped {
            Ok(PublishStatus::PublishedWithDrops {
                dropped_count: self.events.dropped_events,
            })
        } else {
            Ok(PublishStatus::Published)
        }
    }

    /// Extrae el [`VisualFrame`] más antiguo.
    pub fn pop_frame(&mut self) -> Option<VisualFrame> {
        self.frames.pop()
    }

    /// Extrae el [`ScientificTraceEvent`] más antiguo.
    pub fn pop_event(&mut self) -> Option<ScientificTraceEvent> {
        self.events.pop()
    }

    /// Drena todos los frames visuales almacenados.
    pub fn drain_frames(&mut self) -> Vec<VisualFrame> {
        self.frames.drain_all()
    }

    /// Drena todos los eventos científicos almacenados.
    pub fn drain_events(&mut self) -> Vec<ScientificTraceEvent> {
        self.events.drain_all()
    }

    /// Métricas actuales de buffer, publicación y pérdidas.
    pub fn metrics(&self) -> StreamMetrics {
        StreamMetrics {
            buffered_frames: self.frames.len(),
            buffered_events: self.events.buffer.len(),
            dropped_frames: self.frames.dropped_frames(),
            dropped_events: self.events.dropped_events,
            published_frames: self.frames.published_frames(),
            published_events: self.events.published_events,
        }
    }
}

#[derive(Debug)]
struct HubSubscriber {
    stream: Mutex<BoundedObservationStream>,
}

#[derive(Debug)]
struct ObservationHubInner {
    run_id: RunId,
    frame_capacity: usize,
    event_capacity: usize,
    observer_count: AtomicUsize,
    total_dropped_frames: AtomicU64,
    total_dropped_events: AtomicU64,
    subscribers: Mutex<Vec<Weak<HubSubscriber>>>,
}

/// Hub compartible (`Send + Sync`) para publicar frames visuales y eventos científicos
/// hacia múltiples observadores sin bloquear el bucle de control.
#[derive(Debug, Clone)]
pub struct ObservationHub {
    inner: Arc<ObservationHubInner>,
}

impl ObservationHub {
    /// Crea un nuevo [`ObservationHub`] para `run_id` con capacidades fijas por suscriptor.
    pub fn new(
        run_id: RunId,
        frame_capacity: usize,
        event_capacity: usize,
    ) -> Result<Self, ObservabilityError> {
        if frame_capacity == 0 {
            return Err(ObservabilityError::InvalidCapacity {
                capacity: frame_capacity,
            });
        }
        if event_capacity == 0 {
            return Err(ObservabilityError::InvalidCapacity {
                capacity: event_capacity,
            });
        }
        Ok(Self {
            inner: Arc::new(ObservationHubInner {
                run_id,
                frame_capacity,
                event_capacity,
                observer_count: AtomicUsize::new(0),
                total_dropped_frames: AtomicU64::new(0),
                total_dropped_events: AtomicU64::new(0),
                subscribers: Mutex::new(Vec::new()),
            }),
        })
    }

    /// Identificador de ejecución del hub.
    pub fn run_id(&self) -> RunId {
        self.inner.run_id
    }

    /// Número de observadores activos suscritos actualmente.
    pub fn observer_count(&self) -> usize {
        self.inner.observer_count.load(Ordering::Acquire)
    }

    /// Total acumulado de frames visuales descartados en todos los suscriptores.
    pub fn total_dropped_frames(&self) -> u64 {
        self.inner.total_dropped_frames.load(Ordering::Acquire)
    }

    /// Total acumulado de eventos científicos descartados en todos los suscriptores.
    pub fn total_dropped_events(&self) -> u64 {
        self.inner.total_dropped_events.load(Ordering::Acquire)
    }

    /// Suscribe un nuevo observador con buffers acotados propios.
    ///
    /// Al liberar [`ObservationSubscription`], el conteo de observadores se decrementa
    /// automáticamente.
    pub fn subscribe(&self) -> ObservationSubscription {
        let mut stream = BoundedObservationStream::new(
            self.inner.run_id,
            self.inner.frame_capacity,
            self.inner.event_capacity,
        )
        .expect("hub capacities were validated at construction");
        stream.register_observer();

        let sub = Arc::new(HubSubscriber {
            stream: Mutex::new(stream),
        });
        let weak = Arc::downgrade(&sub);
        let mut list = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        list.retain(|entry| entry.strong_count() > 0);
        list.push(weak);
        self.inner.observer_count.fetch_add(1, Ordering::AcqRel);

        ObservationSubscription {
            hub: Arc::clone(&self.inner),
            subscriber: sub,
        }
    }

    /// Construye y difunde un [`VisualFrame`] **solo** si `observer_count() > 0`.
    pub fn publish_frame_if_observed<F>(
        &self,
        build_frame: F,
    ) -> Result<PublishStatus, ObservabilityError>
    where
        F: FnOnce() -> Result<VisualFrame, ObservabilityError>,
    {
        if self.observer_count() == 0 {
            return Ok(PublishStatus::SkippedNoObservers);
        }
        let frame = build_frame()?;
        frame.validate()?;

        let mut list = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        list.retain(|entry| entry.strong_count() > 0);
        if list.is_empty() {
            return Ok(PublishStatus::SkippedNoObservers);
        }

        let mut drops_this_publish = 0_u64;
        for weak in list.iter() {
            if let Some(sub) = weak.upgrade() {
                let mut stream = sub
                    .stream
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let status = stream.publish_frame_if_observed(|| Ok(frame.clone()))?;
                if matches!(status, PublishStatus::PublishedWithDrops { .. }) {
                    drops_this_publish = drops_this_publish.saturating_add(1);
                }
            }
        }

        if drops_this_publish > 0 {
            self.inner
                .total_dropped_frames
                .fetch_add(drops_this_publish, Ordering::AcqRel);
            Ok(PublishStatus::PublishedWithDrops {
                dropped_count: drops_this_publish,
            })
        } else {
            Ok(PublishStatus::Published)
        }
    }

    /// Construye y difunde un [`ScientificTraceEvent`] **solo** si `observer_count() > 0`.
    pub fn publish_event_if_observed<F>(
        &self,
        build_event: F,
    ) -> Result<PublishStatus, ObservabilityError>
    where
        F: FnOnce() -> Result<ScientificTraceEvent, ObservabilityError>,
    {
        if self.observer_count() == 0 {
            return Ok(PublishStatus::SkippedNoObservers);
        }
        let event = build_event()?;
        event.validate()?;

        let mut list = self
            .inner
            .subscribers
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        list.retain(|entry| entry.strong_count() > 0);
        if list.is_empty() {
            return Ok(PublishStatus::SkippedNoObservers);
        }

        let mut drops_this_publish = 0_u64;
        for weak in list.iter() {
            if let Some(sub) = weak.upgrade() {
                let mut stream = sub
                    .stream
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                let status = stream.publish_event_if_observed(|| Ok(event.clone()))?;
                if matches!(status, PublishStatus::PublishedWithDrops { .. }) {
                    drops_this_publish = drops_this_publish.saturating_add(1);
                }
            }
        }

        if drops_this_publish > 0 {
            self.inner
                .total_dropped_events
                .fetch_add(drops_this_publish, Ordering::AcqRel);
            Ok(PublishStatus::PublishedWithDrops {
                dropped_count: drops_this_publish,
            })
        } else {
            Ok(PublishStatus::Published)
        }
    }
}

/// Suscripción RAII de un observador sobre [`ObservationHub`].
#[derive(Debug)]
pub struct ObservationSubscription {
    hub: Arc<ObservationHubInner>,
    subscriber: Arc<HubSubscriber>,
}

impl ObservationSubscription {
    /// Extrae el [`VisualFrame`] más antiguo sin bloquear.
    pub fn try_recv_frame(&self) -> Option<VisualFrame> {
        let mut stream = self
            .subscriber
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stream.pop_frame()
    }

    /// Extrae el [`ScientificTraceEvent`] más antiguo sin bloquear.
    pub fn try_recv_event(&self) -> Option<ScientificTraceEvent> {
        let mut stream = self
            .subscriber
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stream.pop_event()
    }

    /// Drena todos los frames visuales pendientes.
    pub fn drain_frames(&self) -> Vec<VisualFrame> {
        let mut stream = self
            .subscriber
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stream.drain_frames()
    }

    /// Drena todos los eventos científicos pendientes.
    pub fn drain_events(&self) -> Vec<ScientificTraceEvent> {
        let mut stream = self
            .subscriber
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stream.drain_events()
    }

    /// Métricas locales de esta suscripción.
    pub fn metrics(&self) -> StreamMetrics {
        let stream = self
            .subscriber
            .stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        stream.metrics()
    }
}

impl Drop for ObservationSubscription {
    fn drop(&mut self) {
        self.hub.observer_count.fetch_sub(1, Ordering::AcqRel);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{MonotonicClock, VirtualMonotonicClock};
    use std::time::Duration as StdDuration;

    fn sample_entity(id: u64, x: f64) -> VisualEntitySample {
        VisualEntitySample::new(EntityId::new(id), [x, 0.0, 0.0], x * 2.0, 1).unwrap()
    }

    #[test]
    fn ac1_validates_schema_order_clock_and_epistemic_distinction() {
        let run = RunId::new(42);
        let clock = VirtualMonotonicClock::new();
        let mut ring = VisualFrameRing::new(run, 8).unwrap();

        // 1. Versión de esquema inválida es rechazada explícitamente.
        let bad_version = VisualFrame::with_schema_version(
            OBSERVABILITY_SCHEMA_VERSION + 1,
            run,
            Tick::new(1),
            SimDuration::from_secs_f64(0.1).unwrap(),
            clock.now(),
            StreamRevision::new(1),
            EventSource::SimulationWorld,
            EpistemicKind::Observed,
            vec![sample_entity(1, 1.0)],
        );
        assert!(matches!(
            bad_version,
            Err(ObservabilityError::UnsupportedSchemaVersion {
                expected: OBSERVABILITY_SCHEMA_VERSION,
                actual: 2
            })
        ));

        // 2. Publica 3 frames con distinta clasificación epistémica en orden monotónico.
        let t1 = clock.advance(StdDuration::from_millis(10));
        let f_obs = VisualFrame::new(
            run,
            Tick::new(1),
            SimDuration::from_secs_f64(0.1).unwrap(),
            t1,
            StreamRevision::new(1),
            EventSource::LiveEnvironment,
            EpistemicKind::Observed,
            vec![sample_entity(1, 1.0)],
        )
        .unwrap();
        assert!(!f_obs.is_checkpoint_capable());
        assert!(f_obs.is_discardable());

        let t2 = clock.advance(StdDuration::from_millis(10));
        let f_inf = VisualFrame::new(
            run,
            Tick::new(1),
            SimDuration::from_secs_f64(0.1).unwrap(),
            t2,
            StreamRevision::new(2),
            EventSource::Guard,
            EpistemicKind::Inferred,
            vec![sample_entity(1, 1.1)],
        )
        .unwrap();

        let t3 = clock.advance(StdDuration::from_millis(10));
        let f_hyp = VisualFrame::new(
            run,
            Tick::new(2),
            SimDuration::from_secs_f64(0.2).unwrap(),
            t3,
            StreamRevision::new(3),
            EventSource::Planner,
            EpistemicKind::Hypothetical,
            vec![sample_entity(1, 1.5)],
        )
        .unwrap();

        assert!(!ring.push(f_obs).unwrap());
        assert!(!ring.push(f_inf).unwrap());
        assert!(!ring.push(f_hyp).unwrap());

        // 3. Orden no monotónico de (tick, revision) es rechazado.
        let f_stale = VisualFrame::new(
            run,
            Tick::new(2),
            SimDuration::from_secs_f64(0.2).unwrap(),
            clock.advance(StdDuration::from_millis(5)),
            StreamRevision::new(3),
            EventSource::Planner,
            EpistemicKind::Hypothetical,
            vec![sample_entity(1, 1.6)],
        )
        .unwrap();
        assert!(matches!(
            ring.push(f_stale),
            Err(ObservabilityError::NonMonotonicOrder { .. })
        ));

        // 4. Regresión de reloj monotónico es rechazada.
        let f_clock_reg = VisualFrame::new(
            run,
            Tick::new(3),
            SimDuration::from_secs_f64(0.3).unwrap(),
            MonotonicInstant::from_nanos(1),
            StreamRevision::new(4),
            EventSource::SimulationWorld,
            EpistemicKind::Observed,
            vec![sample_entity(1, 2.0)],
        )
        .unwrap();
        assert!(matches!(
            ring.push(f_clock_reg),
            Err(ObservabilityError::ClockRegression { .. })
        ));

        // 5. La distinción Observed | Inferred | Hypothetical se preserva intacta al consumir.
        let drained = ring.drain_all();
        assert_eq!(drained.len(), 3);
        assert!(drained[0].epistemic_kind().is_observed());
        assert!(drained[1].epistemic_kind().is_inferred());
        assert!(drained[2].epistemic_kind().is_hypothetical());
        assert_ne!(drained[0].epistemic_kind(), drained[1].epistemic_kind());
        assert_ne!(drained[1].epistemic_kind(), drained[2].epistemic_kind());
    }

    #[test]
    fn ac2_lazy_publish_skips_without_observers_and_slow_consumer_drops_without_blocking() {
        let run = RunId::new(99);
        let hub = ObservationHub::new(run, 3, 2).unwrap();
        let clock = VirtualMonotonicClock::new();

        // Sin observadores: no se ejecuta el constructor perezoso.
        assert_eq!(hub.observer_count(), 0);
        let skipped = hub
            .publish_frame_if_observed(|| -> Result<VisualFrame, ObservabilityError> {
                panic!("builder must not run when observer_count == 0");
            })
            .unwrap();
        assert_eq!(skipped, PublishStatus::SkippedNoObservers);

        // Con observador lento que no drena: capacidad fija=3, descarta los más antiguos sin bloquear.
        let sub = hub.subscribe();
        assert_eq!(hub.observer_count(), 1);

        for idx in 1..=7_u64 {
            let instant = clock.advance(StdDuration::from_millis(5));
            let status = hub
                .publish_frame_if_observed(|| {
                    VisualFrame::new(
                        run,
                        Tick::new(idx),
                        SimDuration::from_secs_f64((idx as f64) * 0.05).unwrap(),
                        instant,
                        StreamRevision::new(idx),
                        EventSource::SimulationWorld,
                        EpistemicKind::Observed,
                        vec![sample_entity(1, idx as f64)],
                    )
                })
                .unwrap();
            if idx <= 3 {
                assert_eq!(status, PublishStatus::Published);
            } else {
                assert_eq!(
                    status,
                    PublishStatus::PublishedWithDrops { dropped_count: 1 }
                );
            }
        }

        let metrics = sub.metrics();
        assert_eq!(metrics.buffered_frames, 3);
        assert_eq!(metrics.dropped_frames, 4);
        assert_eq!(metrics.published_frames, 7);
        assert_eq!(hub.total_dropped_frames(), 4);

        let frames = sub.drain_frames();
        let ticks: Vec<u64> = frames.iter().map(|f| f.tick().as_u64()).collect();
        assert_eq!(ticks, vec![5, 6, 7]);

        // Saturación en ScientificTraceEvent deja marca explícita de pérdida.
        for idx in 1..=5_u64 {
            let instant = clock.advance(StdDuration::from_millis(5));
            hub.publish_event_if_observed(|| {
                ScientificTraceEvent::new(
                    run,
                    Tick::new(idx),
                    SimDuration::from_secs_f64((idx as f64) * 0.05).unwrap(),
                    instant,
                    StreamRevision::new(idx),
                    EventSource::GenerativeModel,
                    EpistemicKind::Hypothetical,
                    Some(EntityId::new(1)),
                    1001,
                    vec![idx as f64],
                )
            })
            .unwrap();
        }
        let events = sub.drain_events();
        assert_eq!(events.len(), 2);
        assert_eq!(sub.metrics().dropped_events, 3);
        assert_eq!(
            events[0].loss_marker(),
            Some(SaturationLossMarker { dropped_events: 1 })
        );
        assert_eq!(
            events[1].loss_marker(),
            Some(SaturationLossMarker { dropped_events: 1 })
        );

        drop(sub);
        assert_eq!(hub.observer_count(), 0);
    }

    #[test]
    fn ac3_rejects_non_finite_and_excess_event_values_without_text_or_secrets() {
        let run = RunId::new(7);
        assert!(matches!(
            VisualEntitySample::new(EntityId::new(1), [f64::NAN, 0.0, 0.0], 0.0, 0),
            Err(ObservabilityError::NonFiniteValue {
                field: "visual_entity.position",
                ..
            })
        ));
        assert!(matches!(
            ScientificTraceEvent::new(
                run,
                Tick::new(1),
                SimDuration::ZERO,
                MonotonicInstant::ZERO,
                StreamRevision::new(1),
                EventSource::Planner,
                EpistemicKind::Inferred,
                None,
                500,
                vec![0.0; MAX_EVENT_VALUES + 1],
            ),
            Err(ObservabilityError::EventValuesLimitExceeded {
                max: MAX_EVENT_VALUES,
                actual: 33
            })
        ));
    }
}
