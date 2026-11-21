//! Capas reutilizables de sensores, futuros FMC por decisión y trayectoria ejecutada (`std`-only).
//!
//! Este módulo proyecta muestras de planificación FMC ([`FutureTrajectorySample`]),
//! lecturas de sensores ([`SensorRaySample`]), marcadores terminales ([`TerminalMarker`])
//! y trayectoria ejecutada ([`ExecutedTrajectory`]) hacia [`Scene2D`] con:
//! - separación epistémica y visual explícita entre realidad observada, trayectoria ejecutada,
//!   sensores, hipótesis, futuro seleccionado y futuros rechazados;
//! - leyenda explícita ([`OverlayLegend`]) y estilos diferenciados ([`OverlayStyle`]);
//! - invariancia de planificación (transformación pura `&DecisionOverlayFrame` sin RNG);
//! - presupuesto estricto de muestras y bytes por frame ([`OverlayBudget`]).

use std::fmt::Write;

use gogamen_core::{EntityId, EpistemicKind, Tick};

use crate::coords::{CoordinateSpace, Point2, ScaleTransform, normalize_zero};
use crate::error::{ProjectionError, ensure_finite, ensure_positive};
use crate::scene::{LayerId, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId, ScenePrimitive};

const EXECUTED_NODE_BASE: u64 = 10_000;
const FUTURE_NODE_BASE: u64 = 100_000;
const SENSOR_NODE_BASE: u64 = 200_000;
const SENSOR_HIT_NODE_BASE: u64 = 250_000;
const TERMINAL_NODE_BASE: u64 = 300_000;
const LEGEND_SWATCH_NODE_BASE: u64 = 400_000;
const LEGEND_LABEL_NODE_BASE: u64 = 410_000;

/// Clasificación epistémica y de decisión para trayectorias proyectadas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FutureKind {
    /// Realidad observada directamente en el entorno o estado actual.
    ObservedReality,
    /// Futuro seleccionado como ganador por el planificador.
    SelectedFuture,
    /// Hipótesis o rollout exploratorio evaluado por el planificador.
    Hypothesis,
    /// Futuro evaluado y descartado/rechazado por el planificador.
    RejectedFuture,
}

impl FutureKind {
    /// Clasificación epistémica correspondiente en `gogamen-core`.
    pub const fn epistemic_kind(self) -> EpistemicKind {
        match self {
            Self::ObservedReality => EpistemicKind::Observed,
            Self::SelectedFuture | Self::Hypothesis | Self::RejectedFuture => {
                EpistemicKind::Hypothetical
            }
        }
    }

    /// Nombre canónico estable para serialización determinista.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ObservedReality => "observed_reality",
            Self::SelectedFuture => "selected_future",
            Self::Hypothesis => "hypothesis",
            Self::RejectedFuture => "rejected_future",
        }
    }

    /// Etiqueta legible para leyenda.
    pub const fn display_label(self) -> &'static str {
        match self {
            Self::ObservedReality => "Observed Reality",
            Self::SelectedFuture => "Selected Future",
            Self::Hypothesis => "Hypothesis",
            Self::RejectedFuture => "Rejected Future",
        }
    }

    const fn priority_rank(self) -> u8 {
        match self {
            Self::ObservedReality => 0,
            Self::SelectedFuture => 1,
            Self::Hypothesis => 2,
            Self::RejectedFuture => 3,
        }
    }
}

/// Resultado terminal asociado a un marcador de trayectoria.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TerminalOutcome {
    /// Objetivo alcanzado o terminación favorable.
    Success,
    /// Colisión, fallo o terminación desfavorable.
    Failure,
    /// Horizonte máximo de planificación alcanzado sin terminación.
    HorizonReached,
    /// Estado terminal no clasificado.
    Unknown,
}

impl TerminalOutcome {
    /// Nombre canónico estable.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::HorizonReached => "horizon_reached",
            Self::Unknown => "unknown",
        }
    }
}

/// Marcador terminal posicionado en metros del mundo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerminalMarker {
    position_meters: Point2,
    outcome: TerminalOutcome,
    step_index: u32,
    reward: f64,
}

impl TerminalMarker {
    /// Crea un [`TerminalMarker`] validando que `reward` sea finito.
    pub fn new(
        position_meters: Point2,
        outcome: TerminalOutcome,
        step_index: u32,
        reward: f64,
    ) -> Result<Self, ProjectionError> {
        let reward = ensure_finite("terminal_marker.reward", reward)?;
        Ok(Self {
            position_meters,
            outcome,
            step_index,
            reward: normalize_zero(reward),
        })
    }

    /// Posición en metros del mundo.
    pub const fn position_meters(&self) -> Point2 {
        self.position_meters
    }

    /// Resultado terminal.
    pub const fn outcome(&self) -> TerminalOutcome {
        self.outcome
    }

    /// Índice de paso lógico donde ocurrió.
    pub const fn step_index(&self) -> u32 {
        self.step_index
    }

    /// Recompensa o valor asociado.
    pub const fn reward(&self) -> f64 {
        self.reward
    }
}

/// Muestra de trayectoria futura o observada asociada a una decisión FMC.
#[derive(Debug, Clone, PartialEq)]
pub struct FutureTrajectorySample {
    trajectory_id: u64,
    kind: FutureKind,
    points_meters: Vec<Point2>,
    score: f64,
    walker_index: Option<usize>,
    terminal: Option<TerminalMarker>,
}

impl FutureTrajectorySample {
    /// Crea una muestra de trayectoria validando `points_meters.len() >= 2` y `score` finito.
    pub fn new(
        trajectory_id: u64,
        kind: FutureKind,
        points_meters: Vec<Point2>,
        score: f64,
    ) -> Result<Self, ProjectionError> {
        if points_meters.len() < 2 {
            return Err(ProjectionError::InsufficientPolylinePoints {
                count: points_meters.len(),
            });
        }
        let score = ensure_finite("future_trajectory.score", score)?;
        Ok(Self {
            trajectory_id,
            kind,
            points_meters,
            score: normalize_zero(score),
            walker_index: None,
            terminal: None,
        })
    }

    /// Asocia el índice de walker de origen, si proviene de una población FMC.
    pub const fn with_walker_index(mut self, walker_index: usize) -> Self {
        self.walker_index = Some(walker_index);
        self
    }

    /// Asocia un marcador terminal al final de la trayectoria.
    pub const fn with_terminal(mut self, terminal: TerminalMarker) -> Self {
        self.terminal = Some(terminal);
        self
    }

    /// Identificador de trayectoria.
    pub const fn trajectory_id(&self) -> u64 {
        self.trajectory_id
    }

    /// Categoría de futuro.
    pub const fn kind(&self) -> FutureKind {
        self.kind
    }

    /// Puntos en metros del mundo.
    pub fn points_meters(&self) -> &[Point2] {
        &self.points_meters
    }

    /// Puntuación o valor virtual de la trayectoria.
    pub const fn score(&self) -> f64 {
        self.score
    }

    /// Índice de walker si aplica.
    pub const fn walker_index(&self) -> Option<usize> {
        self.walker_index
    }

    /// Marcador terminal asociado, si existe.
    pub const fn terminal(&self) -> Option<TerminalMarker> {
        self.terminal
    }
}

/// Trayectoria efectivamente ejecutada en el mundo, separada de los futuros hipotéticos.
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutedTrajectory {
    trajectory_id: u64,
    points_meters: Vec<Point2>,
    entity_id: Option<EntityId>,
}

impl ExecutedTrajectory {
    /// Crea una trayectoria ejecutada validando `points_meters.len() >= 2`.
    pub fn new(
        trajectory_id: u64,
        points_meters: Vec<Point2>,
        entity_id: Option<EntityId>,
    ) -> Result<Self, ProjectionError> {
        if points_meters.len() < 2 {
            return Err(ProjectionError::InsufficientPolylinePoints {
                count: points_meters.len(),
            });
        }
        Ok(Self {
            trajectory_id,
            points_meters,
            entity_id,
        })
    }

    /// Identificador de la trayectoria ejecutada.
    pub const fn trajectory_id(&self) -> u64 {
        self.trajectory_id
    }

    /// Puntos en metros del mundo.
    pub fn points_meters(&self) -> &[Point2] {
        &self.points_meters
    }

    /// Entidad asociada, si existe.
    pub const fn entity_id(&self) -> Option<EntityId> {
        self.entity_id
    }
}

/// Muestra de rayo o lectura sensorial en metros del mundo.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorRaySample {
    sensor_id: u64,
    origin_meters: Point2,
    hit_meters: Point2,
    hit_detected: bool,
    confidence: f64,
}

/// Alias conveniente para muestras de sensores.
pub type SensorSample = SensorRaySample;

impl SensorRaySample {
    /// Crea una muestra sensorial validando `confidence` finito en `[0.0, 1.0]`.
    pub fn new(
        sensor_id: u64,
        origin_meters: Point2,
        hit_meters: Point2,
        hit_detected: bool,
        confidence: f64,
    ) -> Result<Self, ProjectionError> {
        let confidence = ensure_finite("sensor_ray.confidence", confidence)?;
        if !(0.0..=1.0).contains(&confidence) {
            return Err(ProjectionError::InvalidSamplingConfig {
                field: "sensor_ray.confidence",
                value: confidence,
            });
        }
        Ok(Self {
            sensor_id,
            origin_meters,
            hit_meters,
            hit_detected,
            confidence: normalize_zero(confidence),
        })
    }

    /// Identificador del sensor.
    pub const fn sensor_id(&self) -> u64 {
        self.sensor_id
    }

    /// Origen del rayo en metros.
    pub const fn origin_meters(&self) -> Point2 {
        self.origin_meters
    }

    /// Punto extremo / impacto del rayo en metros.
    pub const fn hit_meters(&self) -> Point2 {
        self.hit_meters
    }

    /// Indica si hubo detección de impacto.
    pub const fn hit_detected(&self) -> bool {
        self.hit_detected
    }

    /// Confianza normalizada en `[0.0, 1.0]`.
    pub const fn confidence(&self) -> f64 {
        self.confidence
    }
}

/// Frame completo de sensores y futuros FMC por decisión listo para proyección.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOverlayFrame {
    scene_id: SceneId,
    tick: Tick,
    decision_id: u64,
    executed_trajectory: Option<ExecutedTrajectory>,
    futures: Vec<FutureTrajectorySample>,
    sensors: Vec<SensorRaySample>,
    terminals: Vec<TerminalMarker>,
}

impl DecisionOverlayFrame {
    /// Crea un frame de overlay vacío para una decisión en `tick`.
    pub const fn new(scene_id: SceneId, tick: Tick, decision_id: u64) -> Self {
        Self {
            scene_id,
            tick,
            decision_id,
            executed_trajectory: None,
            futures: Vec::new(),
            sensors: Vec::new(),
            terminals: Vec::new(),
        }
    }

    /// Adjunta la trayectoria ejecutada separada.
    pub fn with_executed_trajectory(mut self, executed: ExecutedTrajectory) -> Self {
        self.executed_trajectory = Some(executed);
        self
    }

    /// Adjunta muestras de futuros.
    pub fn with_futures(mut self, futures: Vec<FutureTrajectorySample>) -> Self {
        self.futures = futures;
        self
    }

    /// Adjunta muestras de sensores.
    pub fn with_sensors(mut self, sensors: Vec<SensorRaySample>) -> Self {
        self.sensors = sensors;
        self
    }

    /// Adjunta marcadores terminales independientes.
    pub fn with_terminals(mut self, terminals: Vec<TerminalMarker>) -> Self {
        self.terminals = terminals;
        self
    }

    /// Identificador de escena destino.
    pub const fn scene_id(&self) -> SceneId {
        self.scene_id
    }

    /// Tick lógico del frame.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Identificador de decisión FMC.
    pub const fn decision_id(&self) -> u64 {
        self.decision_id
    }

    /// Trayectoria ejecutada separada, si existe.
    pub const fn executed_trajectory(&self) -> Option<&ExecutedTrajectory> {
        self.executed_trajectory.as_ref()
    }

    /// Futuros declarados en el frame.
    pub fn futures(&self) -> &[FutureTrajectorySample] {
        &self.futures
    }

    /// Lecturas sensoriales declaradas en el frame.
    pub fn sensors(&self) -> &[SensorRaySample] {
        &self.sensors
    }

    /// Marcadores terminales independientes.
    pub fn terminals(&self) -> &[TerminalMarker] {
        &self.terminals
    }
}

/// Color RGBA de 8 bits por canal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RgbaColor {
    /// Canal rojo.
    pub r: u8,
    /// Canal verde.
    pub g: u8,
    /// Canal azul.
    pub b: u8,
    /// Canal alfa.
    pub a: u8,
}

impl RgbaColor {
    /// Construye un color RGBA explícito.
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Representación hexadecimal canónica `#RRGGBBAA`.
    pub fn to_hex(self) -> String {
        format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
    }
}

/// Patrón de trazo para diferenciar categorías visuales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DashPattern {
    /// Línea continua.
    Solid,
    /// Línea discontinua (segmentos largos).
    Dashed,
    /// Línea punteada.
    Dotted,
    /// Patrón raya-punto.
    DashDot,
}

impl DashPattern {
    /// Nombre canónico estable.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Solid => "solid",
            Self::Dashed => "dashed",
            Self::Dotted => "dotted",
            Self::DashDot => "dash_dot",
        }
    }
}

/// Estilo visual explícito asignado a una categoría del overlay.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayStyle {
    /// Color RGBA.
    pub color: RgbaColor,
    /// Grosor de trazo lógico (`> 0.0`).
    pub stroke_width: f64,
    /// Patrón de trazo.
    pub dash: DashPattern,
    /// Opacidad en `[0.0, 1.0]`.
    pub opacity: f64,
    /// Capa destino en [`Scene2D`].
    pub layer_id: LayerId,
    /// Orden Z explícito dentro de [`Scene2D`].
    pub z_order: i32,
    /// Código numérico de estado asociado a los nodos emitidos.
    pub state_code: u32,
}

/// Paleta diferenciada por defecto para todas las categorías de sensores y futuros.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayPalette {
    /// Estilo para [`FutureKind::ObservedReality`].
    pub observed_reality: OverlayStyle,
    /// Estilo para [`ExecutedTrajectory`].
    pub executed_trajectory: OverlayStyle,
    /// Estilo para [`SensorRaySample`].
    pub sensor_ray: OverlayStyle,
    /// Estilo para [`FutureKind::RejectedFuture`].
    pub rejected_future: OverlayStyle,
    /// Estilo para [`FutureKind::Hypothesis`].
    pub hypothesis: OverlayStyle,
    /// Estilo para [`FutureKind::SelectedFuture`].
    pub selected_future: OverlayStyle,
    /// Estilo para [`TerminalMarker`].
    pub terminal_marker: OverlayStyle,
}

impl Default for OverlayPalette {
    fn default() -> Self {
        Self {
            observed_reality: OverlayStyle {
                color: RgbaColor::new(0x2E, 0x8B, 0x57, 0xF2),
                stroke_width: 2.0,
                dash: DashPattern::Solid,
                opacity: 0.95,
                layer_id: LayerId::Environment,
                z_order: 0,
                state_code: 101,
            },
            executed_trajectory: OverlayStyle {
                color: RgbaColor::new(0x00, 0x5A, 0xB5, 0xFF),
                stroke_width: 3.2,
                dash: DashPattern::Solid,
                opacity: 1.0,
                layer_id: LayerId::Custom(1),
                z_order: 5,
                state_code: 102,
            },
            sensor_ray: OverlayStyle {
                color: RgbaColor::new(0xE6, 0x9F, 0x00, 0xBF),
                stroke_width: 1.2,
                dash: DashPattern::Dotted,
                opacity: 0.75,
                layer_id: LayerId::Sensors,
                z_order: 10,
                state_code: 103,
            },
            rejected_future: OverlayStyle {
                color: RgbaColor::new(0xCC, 0x79, 0xA7, 0x59),
                stroke_width: 1.0,
                dash: DashPattern::Dotted,
                opacity: 0.35,
                layer_id: LayerId::Custom(2),
                z_order: 15,
                state_code: 104,
            },
            hypothesis: OverlayStyle {
                color: RgbaColor::new(0x56, 0xB4, 0xE9, 0xA6),
                stroke_width: 1.5,
                dash: DashPattern::Dashed,
                opacity: 0.65,
                layer_id: LayerId::Imagination,
                z_order: 20,
                state_code: 105,
            },
            selected_future: OverlayStyle {
                color: RgbaColor::new(0x00, 0x9E, 0x73, 0xFF),
                stroke_width: 2.8,
                dash: DashPattern::DashDot,
                opacity: 1.0,
                layer_id: LayerId::Decisions,
                z_order: 30,
                state_code: 106,
            },
            terminal_marker: OverlayStyle {
                color: RgbaColor::new(0xD5, 0x5E, 0x00, 0xF2),
                stroke_width: 2.0,
                dash: DashPattern::Solid,
                opacity: 0.95,
                layer_id: LayerId::Custom(3),
                z_order: 35,
                state_code: 107,
            },
        }
    }
}

impl OverlayPalette {
    /// Devuelve el estilo asociado a un [`FutureKind`].
    pub const fn style_for_future(&self, kind: FutureKind) -> &OverlayStyle {
        match kind {
            FutureKind::ObservedReality => &self.observed_reality,
            FutureKind::SelectedFuture => &self.selected_future,
            FutureKind::Hypothesis => &self.hypothesis,
            FutureKind::RejectedFuture => &self.rejected_future,
        }
    }
}

/// Categoría de leyenda para distinguir capas en el overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LegendCategory {
    /// Realidad observada.
    ObservedReality,
    /// Trayectoria ejecutada separada.
    ExecutedTrajectory,
    /// Rayos / lecturas de sensores.
    SensorRay,
    /// Futuros rechazados.
    RejectedFuture,
    /// Hipótesis exploratorias.
    Hypothesis,
    /// Futuro seleccionado.
    SelectedFuture,
    /// Marcadores terminales.
    TerminalMarker,
}

impl LegendCategory {
    /// Nombre canónico estable.
    pub const fn canonical_name(self) -> &'static str {
        match self {
            Self::ObservedReality => "observed_reality",
            Self::ExecutedTrajectory => "executed_trajectory",
            Self::SensorRay => "sensor_ray",
            Self::RejectedFuture => "rejected_future",
            Self::Hypothesis => "hypothesis",
            Self::SelectedFuture => "selected_future",
            Self::TerminalMarker => "terminal_marker",
        }
    }

    /// Etiqueta legible para UI/inspección.
    pub const fn display_label(self) -> &'static str {
        match self {
            Self::ObservedReality => FutureKind::ObservedReality.display_label(),
            Self::ExecutedTrajectory => "Executed Trajectory",
            Self::SensorRay => "Sensor Rays",
            Self::RejectedFuture => FutureKind::RejectedFuture.display_label(),
            Self::Hypothesis => FutureKind::Hypothesis.display_label(),
            Self::SelectedFuture => FutureKind::SelectedFuture.display_label(),
            Self::TerminalMarker => "Terminal Markers",
        }
    }
}

/// Entrada individual de la leyenda visual del overlay.
#[derive(Debug, Clone, PartialEq)]
pub struct LegendEntry {
    /// Categoría representada.
    pub category: LegendCategory,
    /// Etiqueta descriptiva.
    pub label: String,
    /// Estilo visual diferenciado.
    pub style: OverlayStyle,
    /// Indica si la capa está visible según [`LayerVisibility`].
    pub visible: bool,
    /// Número de muestras proyectadas en esta categoría.
    pub sample_count: usize,
}

/// Leyenda explícita y ordenada del overlay de decisión.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayLegend {
    entries: Vec<LegendEntry>,
}

impl OverlayLegend {
    /// Entradas de leyenda ordenadas de forma determinista por `z_order`.
    pub fn entries(&self) -> &[LegendEntry] {
        &self.entries
    }

    /// Busca la entrada correspondiente a `category`.
    pub fn find(&self, category: LegendCategory) -> Option<&LegendEntry> {
        self.entries.iter().find(|e| e.category == category)
    }

    /// Serializa la leyenda en formato JSON canónico determinista.
    pub fn to_canonical_json(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("{\"entries\":[");
        for (idx, entry) in self.entries.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"category\":\"{}\",\"label\":\"{}\",\"visible\":{},\"sample_count\":{},\"style\":{{\"color\":\"{}\",\"stroke_width\":{},\"dash\":\"{}\",\"opacity\":{},\"layer\":\"{}\",\"z_order\":{},\"state_code\":{}}}}}",
                entry.category.canonical_name(),
                entry.label,
                entry.visible,
                entry.sample_count,
                entry.style.color.to_hex(),
                normalize_zero(entry.style.stroke_width),
                entry.style.dash.as_str(),
                normalize_zero(entry.style.opacity),
                entry.style.layer_id.canonical_name(),
                entry.style.z_order,
                entry.style.state_code
            );
        }
        out.push_str("]}");
        out
    }
}

/// Control de visibilidad por capa de observabilidad sin afectar la planificación.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerVisibility {
    /// Visibilidad de [`FutureKind::ObservedReality`].
    pub observed_reality: bool,
    /// Visibilidad de [`ExecutedTrajectory`].
    pub executed_trajectory: bool,
    /// Visibilidad de [`SensorRaySample`].
    pub sensors: bool,
    /// Visibilidad de [`FutureKind::RejectedFuture`].
    pub rejected_futures: bool,
    /// Visibilidad de [`FutureKind::Hypothesis`].
    pub hypotheses: bool,
    /// Visibilidad de [`FutureKind::SelectedFuture`].
    pub selected_future: bool,
    /// Visibilidad de [`TerminalMarker`].
    pub terminals: bool,
    /// Visibilidad de los nodos de leyenda en [`LayerId::Labels`].
    pub legend: bool,
}

impl Default for LayerVisibility {
    fn default() -> Self {
        Self::all_visible()
    }
}

impl LayerVisibility {
    /// Activa todas las capas.
    pub const fn all_visible() -> Self {
        Self {
            observed_reality: true,
            executed_trajectory: true,
            sensors: true,
            rejected_futures: true,
            hypotheses: true,
            selected_future: true,
            terminals: true,
            legend: true,
        }
    }

    /// Configuración minimalista (solo trayectoria ejecutada, futuro seleccionado y leyenda).
    pub const fn minimal() -> Self {
        Self {
            observed_reality: false,
            executed_trajectory: true,
            sensors: false,
            rejected_futures: false,
            hypotheses: false,
            selected_future: true,
            terminals: false,
            legend: true,
        }
    }

    /// Indica si un [`FutureKind`] concreto está visible.
    pub const fn is_future_visible(self, kind: FutureKind) -> bool {
        match kind {
            FutureKind::ObservedReality => self.observed_reality,
            FutureKind::SelectedFuture => self.selected_future,
            FutureKind::Hypothesis => self.hypotheses,
            FutureKind::RejectedFuture => self.rejected_futures,
        }
    }
}

/// Presupuesto explícito de muestras y bytes por frame para observabilidad acotada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayBudget {
    /// Máximo número de trayectorias futuras exportadas por frame.
    pub max_future_samples: usize,
    /// Máximo número de rayos sensoriales exportados por frame.
    pub max_sensor_samples: usize,
    /// Máximo número de marcadores terminales exportados por frame.
    pub max_terminal_markers: usize,
    /// Máximo número de bytes serializados permitidos en `Scene2D::to_canonical_json()`.
    pub max_bytes_per_frame: usize,
    /// Si es `false` (por defecto), no se exportan todos los walkers sino un subconjunto acotado.
    pub export_all_walkers: bool,
}

impl Default for OverlayBudget {
    fn default() -> Self {
        Self {
            max_future_samples: 16,
            max_sensor_samples: 32,
            max_terminal_markers: 16,
            max_bytes_per_frame: 16_384,
            export_all_walkers: false,
        }
    }
}

/// Configuración de proyección para [`DecisionOverlayFrame`].
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayConfig {
    /// Espacio de coordenadas destino.
    pub coordinate_space: CoordinateSpace,
    /// Visibilidad por capa.
    pub visibility: LayerVisibility,
    /// Presupuesto de muestras y bytes por frame.
    pub budget: OverlayBudget,
    /// Paleta de estilos diferenciados.
    pub palette: OverlayPalette,
    /// Paso determinista de muestreo (`>= 1`).
    pub sample_stride: usize,
    /// Tasa determinista de muestreo en `(0.0, 1.0]`.
    pub sampling_rate: f64,
    /// Radio de marcadores terminales en metros.
    pub terminal_marker_radius_meters: f64,
    /// Radio de impactos de sensor en metros.
    pub sensor_hit_radius_meters: f64,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            coordinate_space: CoordinateSpace::meters(),
            visibility: LayerVisibility::default(),
            budget: OverlayBudget::default(),
            palette: OverlayPalette::default(),
            sample_stride: 1,
            sampling_rate: 1.0,
            terminal_marker_radius_meters: 0.18,
            sensor_hit_radius_meters: 0.10,
        }
    }
}

impl OverlayConfig {
    /// Valida los parámetros numéricos de muestreo y radios.
    pub fn validate(&self) -> Result<(), ProjectionError> {
        if self.sample_stride == 0 {
            return Err(ProjectionError::InvalidSamplingConfig {
                field: "overlay.sample_stride",
                value: 0.0,
            });
        }
        let rate = ensure_finite("overlay.sampling_rate", self.sampling_rate)?;
        if rate <= 0.0 || rate > 1.0 {
            return Err(ProjectionError::InvalidSamplingConfig {
                field: "overlay.sampling_rate",
                value: rate,
            });
        }
        ensure_positive(
            "overlay.terminal_marker_radius_meters",
            self.terminal_marker_radius_meters,
        )?;
        ensure_positive(
            "overlay.sensor_hit_radius_meters",
            self.sensor_hit_radius_meters,
        )?;
        Ok(())
    }
}

/// Métricas de exportación y truncado del overlay proyectado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayStats {
    /// Número total de futuros recibidos en el frame.
    pub input_futures: usize,
    /// Número de futuros finalmente exportados a [`Scene2D`].
    pub exported_futures: usize,
    /// Número total de sensores recibidos en el frame.
    pub input_sensors: usize,
    /// Número de sensores finalmente exportados a [`Scene2D`].
    pub exported_sensors: usize,
    /// Número total de marcadores terminales recibidos (incluyendo los adjuntos a futuros).
    pub input_terminals: usize,
    /// Número de marcadores terminales finalmente exportados.
    pub exported_terminals: usize,
    /// Indica si se aplicó recorte por límite de muestras o `export_all_walkers == false`.
    pub truncated_by_sample_cap: bool,
    /// Indica si se aplicó recorte adicional para respetar `max_bytes_per_frame`.
    pub truncated_by_byte_cap: bool,
    /// Tamaño exacto en bytes de `scene.to_canonical_json()`.
    pub serialized_bytes: usize,
}

/// Resultado de proyectar un [`DecisionOverlayFrame`] a [`Scene2D`].
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedOverlay {
    /// Escena 2D resultante con capas ordenadas y visibles/ocultas según configuración.
    pub scene: Scene2D,
    /// Leyenda explícita con estilo diferenciado por categoría.
    pub legend: OverlayLegend,
    /// Métricas verificables de cap de muestras y bytes.
    pub stats: OverlayStats,
}

impl ProjectedOverlay {
    /// Serializa la escena y la leyenda conjuntamente de forma determinista.
    pub fn to_canonical_json(&self) -> String {
        format!(
            "{{\"scene\":{},\"legend\":{},\"serialized_bytes\":{}}}",
            self.scene.to_canonical_json(),
            self.legend.to_canonical_json(),
            self.stats.serialized_bytes
        )
    }
}

/// Proyecta un [`DecisionOverlayFrame`] hacia [`Scene2D`] de forma pura, determinista y acotada.
pub fn project_decision_overlay(
    frame: &DecisionOverlayFrame,
    config: &OverlayConfig,
) -> Result<ProjectedOverlay, ProjectionError> {
    config.validate()?;
    let transform = ScaleTransform::from_meters(config.coordinate_space);

    let mut futures_sorted: Vec<&FutureTrajectorySample> = frame.futures().iter().collect();
    futures_sorted.sort_by_key(|f| (f.kind().priority_rank(), f.trajectory_id()));

    let mut sensors_sorted: Vec<&SensorRaySample> = frame.sensors().iter().collect();
    sensors_sorted.sort_by_key(|s| s.sensor_id());

    let mut all_terminals: Vec<TerminalMarker> = frame.terminals().to_vec();
    for f in &futures_sorted {
        if let Some(term) = f.terminal() {
            all_terminals.push(term);
        }
    }
    all_terminals.sort_by_key(|t| (t.step_index(), t.outcome()));

    let input_futures = futures_sorted.len();
    let input_sensors = sensors_sorted.len();
    let input_terminals = all_terminals.len();

    let (selected_futures, mut truncated_by_sample_cap) = select_futures(&futures_sorted, config);
    let (selected_sensors, sensors_capped) =
        select_evenly(&sensors_sorted, config, config.budget.max_sensor_samples);
    let (selected_terminals, terminals_capped) =
        select_evenly(&all_terminals, config, config.budget.max_terminal_markers);
    truncated_by_sample_cap |= sensors_capped || terminals_capped;

    let state = ShedState {
        include_executed: frame.executed_trajectory().is_some()
            && config.visibility.executed_trajectory,
        include_legend_nodes: config.visibility.legend,
        futures: selected_futures,
        sensors: selected_sensors,
        terminals: selected_terminals,
    };
    let counters = InputCounters {
        input_futures,
        input_sensors,
        input_terminals,
        truncated_by_sample_cap,
    };

    // Primera pasada: igual que el bucle original, una construcción y una serialización.
    let (scene, legend) = build_state(frame, config, &transform, &state)?;
    let serialized_bytes = serialized_len(&scene);
    if serialized_bytes <= config.budget.max_bytes_per_frame {
        return Ok(finish(
            scene,
            legend,
            &state,
            &counters,
            false,
            serialized_bytes,
        ));
    }

    // Recorte exacto: el orden de recorte es determinista e independiente del tamaño, así que
    // se simula sobre longitudes de fragmentos por nodo (O(F)) y se confirma con una sola
    // serialización canónica adicional. Si el modelo no aplica o discrepa, se usa el bucle lineal.
    if let Some(plan) = plan_exact_shed(config, &transform, &state, &scene, serialized_bytes) {
        let mut shed = state.clone();
        for _ in 0..plan.steps {
            shed.shed_one();
        }
        let (scene, legend) = build_state(frame, config, &transform, &shed)?;
        let confirmed = serialized_len(&scene);
        if confirmed == plan.predicted_bytes {
            if plan.fits {
                return Ok(finish(scene, legend, &shed, &counters, true, confirmed));
            }
            return Err(ProjectionError::OverlayBudgetTooSmall {
                required_bytes: confirmed,
                max_bytes: config.budget.max_bytes_per_frame,
            });
        }
    }

    enforce_budget_linear(frame, config, &transform, state, &counters)
}

struct InputCounters {
    input_futures: usize,
    input_sensors: usize,
    input_terminals: usize,
    truncated_by_sample_cap: bool,
}

/// Selección vigente del overlay durante el recorte por presupuesto de bytes.
#[derive(Clone)]
struct ShedState<'a> {
    include_executed: bool,
    include_legend_nodes: bool,
    futures: Vec<&'a FutureTrajectorySample>,
    sensors: Vec<&'a SensorRaySample>,
    terminals: Vec<TerminalMarker>,
}

impl ShedState<'_> {
    fn shed_one(&mut self) -> bool {
        shed_lowest_priority(
            &mut self.futures,
            &mut self.sensors,
            &mut self.terminals,
            &mut self.include_executed,
            &mut self.include_legend_nodes,
        )
    }
}

fn build_state(
    frame: &DecisionOverlayFrame,
    config: &OverlayConfig,
    transform: &ScaleTransform,
    state: &ShedState<'_>,
) -> Result<(Scene2D, OverlayLegend), ProjectionError> {
    build_scene_and_legend(
        frame,
        config,
        transform,
        state.include_executed,
        &state.futures,
        &state.sensors,
        &state.terminals,
        state.include_legend_nodes,
    )
}

/// Única vía de serialización canónica de escena en la ruta de enforcement (contable en tests).
fn serialized_len(scene: &Scene2D) -> usize {
    #[cfg(test)]
    enforcement_tests::CANONICAL_SERIALIZATIONS.with(|c| c.set(c.get() + 1));
    scene.to_canonical_json().len()
}

fn finish(
    scene: Scene2D,
    legend: OverlayLegend,
    state: &ShedState<'_>,
    counters: &InputCounters,
    truncated_by_byte_cap: bool,
    serialized_bytes: usize,
) -> ProjectedOverlay {
    let stats = OverlayStats {
        input_futures: counters.input_futures,
        exported_futures: state.futures.len(),
        input_sensors: counters.input_sensors,
        exported_sensors: state.sensors.len(),
        input_terminals: counters.input_terminals,
        exported_terminals: state.terminals.len(),
        truncated_by_sample_cap: counters.truncated_by_sample_cap,
        truncated_by_byte_cap,
        serialized_bytes,
    };
    ProjectedOverlay {
        scene,
        legend,
        stats,
    }
}

/// Bucle lineal de referencia (una reconstrucción por paso); fallback del recorte exacto.
fn enforce_budget_linear(
    frame: &DecisionOverlayFrame,
    config: &OverlayConfig,
    transform: &ScaleTransform,
    mut state: ShedState<'_>,
    counters: &InputCounters,
) -> Result<ProjectedOverlay, ProjectionError> {
    let mut truncated_by_byte_cap = false;
    loop {
        let (scene, legend) = build_state(frame, config, transform, &state)?;
        let serialized_bytes = serialized_len(&scene);
        if serialized_bytes <= config.budget.max_bytes_per_frame {
            return Ok(finish(
                scene,
                legend,
                &state,
                counters,
                truncated_by_byte_cap,
                serialized_bytes,
            ));
        }
        truncated_by_byte_cap = true;
        if !state.shed_one() {
            return Err(ProjectionError::OverlayBudgetTooSmall {
                required_bytes: serialized_bytes,
                max_bytes: config.budget.max_bytes_per_frame,
            });
        }
    }
}

struct ShedPlan {
    /// Número de pasos de `shed_lowest_priority` que ejecutaba el bucle original.
    steps: usize,
    /// `true` si el estado resultante cabe; `false` si se agotó el recorte.
    fits: bool,
    predicted_bytes: usize,
}

fn decimal_digits(value: u64) -> u32 {
    value.checked_ilog10().unwrap_or(0) + 1
}

/// Calcula, sin reserializar, el primer estado de la secuencia de recorte que cabe en el
/// presupuesto. El tamaño canónico es `constante + sum(capas: nodos + separadores)`; los nodos
/// conservan su longitud al recortar (los ids `BASE + idx` no cambian de nº de dígitos mientras
/// `BASE + n - 1` conserve los dígitos de `BASE`, condición que se comprueba) salvo los nodos
/// de leyenda, cuyo texto depende de los contadores y se recalculan (14 nodos como máximo).
/// Devuelve `None` si el modelo no aplica o no reproduce el tamaño inicial.
fn plan_exact_shed(
    config: &OverlayConfig,
    transform: &ScaleTransform,
    initial: &ShedState<'_>,
    scene: &Scene2D,
    initial_len: usize,
) -> Option<ShedPlan> {
    let id_stable = |base: u64, n: usize| {
        n == 0 || decimal_digits(base.wrapping_add(n as u64 - 1)) == decimal_digits(base)
    };
    if !(id_stable(FUTURE_NODE_BASE, initial.futures.len())
        && id_stable(SENSOR_NODE_BASE, initial.sensors.len())
        && id_stable(SENSOR_HIT_NODE_BASE, initial.sensors.len())
        && id_stable(TERMINAL_NODE_BASE, initial.terminals.len()))
    {
        return None;
    }
    let p = &config.palette;
    let layer_of = |id: LayerId| scene.layers().iter().find(|l| l.id() == id);
    let observed = layer_of(p.observed_reality.layer_id)?;
    let executed = layer_of(p.executed_trajectory.layer_id)?;
    let sensors_layer = layer_of(p.sensor_ray.layer_id)?;
    let rejected = layer_of(p.rejected_future.layer_id)?;
    let hypothesis = layer_of(p.hypothesis.layer_id)?;
    let selected = layer_of(p.selected_future.layer_id)?;
    let terminal = layer_of(p.terminal_marker.layer_id)?;
    let labels = layer_of(LayerId::Labels)?;

    // Longitud de cada elemento seleccionado, en orden de vector.
    // Capas: 0 observed, 1 selected, 2 rejected, 3 hypothesis.
    let mut cursors = [0usize; 4];
    let layer_nodes = [observed, selected, rejected, hypothesis];
    let mut primary: Vec<(usize, usize)> = Vec::new(); // (capa, len) Observed/Selected
    let mut rejected_lens: Vec<usize> = Vec::new();
    let mut hypothesis_lens: Vec<usize> = Vec::new();
    for f in &initial.futures {
        let slot = match f.kind() {
            FutureKind::ObservedReality => 0,
            FutureKind::SelectedFuture => 1,
            FutureKind::RejectedFuture => 2,
            FutureKind::Hypothesis => 3,
        };
        let node = layer_nodes[slot].nodes().get(cursors[slot])?;
        cursors[slot] += 1;
        let len = node.canonical_json_len();
        match slot {
            0 | 1 => primary.push((slot, len)),
            2 => rejected_lens.push(len),
            _ => hypothesis_lens.push(len),
        }
    }
    if cursors[0] != observed.nodes().len()
        || cursors[1] != selected.nodes().len()
        || cursors[2] != rejected.nodes().len()
        || cursors[3] != hypothesis.nodes().len()
    {
        return None;
    }

    let prefix = |lens: Vec<(usize, usize)>| -> Vec<(usize, usize)> {
        let mut acc = vec![(0usize, 0usize)];
        for (len, n) in lens {
            let (l, c) = *acc.last().unwrap_or(&(0, 0));
            acc.push((l + len, c + n));
        }
        acc
    };
    // Prefijos (suma de longitudes de nodo, nº de nodos) por nº de elementos retenidos.
    let mut s_cursor = 0usize;
    let mut sensor_items = Vec::new();
    if config.visibility.sensors {
        for s in &initial.sensors {
            let n = if s.hit_detected() { 2 } else { 1 };
            let mut len = 0;
            for _ in 0..n {
                len += sensors_layer.nodes().get(s_cursor)?.canonical_json_len();
                s_cursor += 1;
            }
            sensor_items.push((len, n));
        }
        if s_cursor != sensors_layer.nodes().len() {
            return None;
        }
    } else {
        sensor_items.resize(initial.sensors.len(), (0, 0));
    }
    let sensor_prefix = prefix(sensor_items);
    let mut terminal_items = Vec::new();
    if config.visibility.terminals {
        if terminal.nodes().len() != initial.terminals.len() {
            return None;
        }
        for node in terminal.nodes() {
            terminal_items.push((node.canonical_json_len(), 1));
        }
    } else {
        terminal_items.resize(initial.terminals.len(), (0, 0));
    }
    let terminal_prefix = prefix(terminal_items);

    let executed_len = if initial.include_executed {
        executed.nodes().first()?.canonical_json_len()
    } else {
        0
    };
    let contrib = |sum: usize, n: usize| sum + n.saturating_sub(1);

    // Simulación del estado.
    let mut primary_stack = primary;
    let mut sums = [0usize; 4]; // observed, selected, rejected, hypothesis
    let mut ns = [0usize; 4];
    for &(slot, len) in &primary_stack {
        sums[slot] += len;
        ns[slot] += 1;
    }
    sums[2] = rejected_lens.iter().sum();
    ns[2] = rejected_lens.len();
    sums[3] = hypothesis_lens.iter().sum();
    ns[3] = hypothesis_lens.len();
    let mut sensors_n = initial.sensors.len();
    let mut terminals_n = initial.terminals.len();
    let mut include_executed = initial.include_executed;
    let mut include_legend = initial.include_legend_nodes;

    let legend_contrib = |counts: [usize; 7]| -> Option<usize> {
        let legend = build_legend(config, counts);
        let mut layer = SceneLayer::with_visibility(
            LayerId::Labels,
            LayerId::Labels.default_z_order(),
            config.visibility.legend,
        );
        populate_legend_nodes(&mut layer, &legend, transform).ok()?;
        Some(layer.canonical_nodes_len())
    };

    let mut constant: Option<usize> = None;
    let mut steps = 0usize;
    loop {
        let counts = [
            ns[0],
            usize::from(include_executed),
            if config.visibility.sensors {
                sensors_n
            } else {
                0
            },
            ns[2],
            ns[3],
            ns[1],
            if config.visibility.terminals {
                terminals_n
            } else {
                0
            },
        ];
        let (s_len, s_nodes) = sensor_prefix[sensors_n];
        let (t_len, t_nodes) = terminal_prefix[terminals_n];
        let (e_len, e_nodes) = if include_executed {
            (executed_len, 1)
        } else {
            (0, 0)
        };
        let legend_part = if include_legend {
            legend_contrib(counts)?
        } else {
            0
        };
        let variable = contrib(sums[0], ns[0])
            + contrib(e_len, e_nodes)
            + contrib(s_len, s_nodes)
            + contrib(sums[2], ns[2])
            + contrib(sums[3], ns[3])
            + contrib(sums[1], ns[1])
            + contrib(t_len, t_nodes)
            + legend_part;

        let base = match constant {
            Some(c) => c,
            None => {
                // Estado inicial: la constante se deduce del tamaño real ya medido.
                let c = initial_len.checked_sub(variable)?;
                // Comprobación cruzada de la capa de leyenda con la escena real.
                if initial.include_legend_nodes && labels.canonical_nodes_len() != legend_part {
                    return None;
                }
                constant = Some(c);
                c
            }
        };
        let size = base + variable;
        if steps > 0 && size <= config.budget.max_bytes_per_frame {
            return Some(ShedPlan {
                steps,
                fits: true,
                predicted_bytes: size,
            });
        }

        // Siguiente paso: espejo de `shed_lowest_priority`.
        if let Some(len) = rejected_lens.pop() {
            sums[2] -= len;
            ns[2] -= 1;
        } else if let Some(len) = hypothesis_lens.pop() {
            sums[3] -= len;
            ns[3] -= 1;
        } else if sensors_n > 0 {
            sensors_n /= 2;
        } else if terminals_n > 0 {
            terminals_n /= 2;
        } else if let Some((slot, len)) = primary_stack.pop() {
            sums[slot] -= len;
            ns[slot] -= 1;
        } else if include_executed {
            include_executed = false;
        } else if include_legend {
            include_legend = false;
        } else {
            return Some(ShedPlan {
                steps,
                fits: false,
                predicted_bytes: size,
            });
        }
        steps += 1;
    }
}

fn select_futures<'a>(
    sorted: &[&'a FutureTrajectorySample],
    config: &OverlayConfig,
) -> (Vec<&'a FutureTrajectorySample>, bool) {
    let visible: Vec<&'a FutureTrajectorySample> = sorted
        .iter()
        .copied()
        .filter(|f| config.visibility.is_future_visible(f.kind()))
        .collect();

    if visible.is_empty() || config.budget.max_future_samples == 0 {
        return (Vec::new(), !visible.is_empty());
    }

    let mut priority = Vec::new();
    let mut secondary = Vec::new();
    for &item in &visible {
        match item.kind() {
            FutureKind::ObservedReality | FutureKind::SelectedFuture => priority.push(item),
            FutureKind::Hypothesis | FutureKind::RejectedFuture => secondary.push(item),
        }
    }

    let mut out = Vec::new();
    let mut truncated = false;

    for item in priority {
        if out.len() < config.budget.max_future_samples {
            out.push(item);
        } else {
            truncated = true;
        }
    }

    let remaining_cap = config.budget.max_future_samples.saturating_sub(out.len());
    let effective_secondary_cap = if config.budget.export_all_walkers {
        remaining_cap
    } else {
        // Por defecto no se exportan todos los walkers: se acota al presupuesto restante.
        remaining_cap
    };

    let (sampled_secondary, sec_truncated) =
        select_evenly(&secondary, config, effective_secondary_cap);
    truncated |= sec_truncated;
    out.extend(sampled_secondary);
    (out, truncated)
}

fn select_evenly<T: Copy>(items: &[T], config: &OverlayConfig, max_count: usize) -> (Vec<T>, bool) {
    if items.is_empty() || max_count == 0 {
        return (Vec::new(), !items.is_empty());
    }

    let strided: Vec<T> = items
        .iter()
        .enumerate()
        .filter(|(idx, _)| idx % config.sample_stride == 0)
        .map(|(_, &item)| item)
        .collect();

    let mut truncated = strided.len() < items.len();
    if strided.is_empty() {
        return (Vec::new(), truncated);
    }

    let target_by_rate = if config.sampling_rate < 1.0 {
        let count = ((strided.len() as f64) * config.sampling_rate).floor() as usize;
        count.max(1).min(strided.len())
    } else {
        strided.len()
    };

    let target = target_by_rate.min(max_count);
    if target < strided.len() {
        truncated = true;
    }

    if target == strided.len() {
        return (strided, truncated);
    }

    let mut selected = Vec::with_capacity(target);
    for slot in 0..target {
        let idx = (slot * strided.len()) / target;
        selected.push(strided[idx]);
    }
    (selected, truncated)
}

fn shed_lowest_priority(
    futures: &mut Vec<&FutureTrajectorySample>,
    sensors: &mut Vec<&SensorRaySample>,
    terminals: &mut Vec<TerminalMarker>,
    include_executed: &mut bool,
    include_legend_nodes: &mut bool,
) -> bool {
    if let Some(pos) = futures
        .iter()
        .rposition(|f| f.kind() == FutureKind::RejectedFuture)
    {
        futures.remove(pos);
        return true;
    }
    if let Some(pos) = futures
        .iter()
        .rposition(|f| f.kind() == FutureKind::Hypothesis)
    {
        futures.remove(pos);
        return true;
    }
    if !sensors.is_empty() {
        let new_len = sensors.len() / 2;
        sensors.truncate(new_len);
        return true;
    }
    if !terminals.is_empty() {
        let new_len = terminals.len() / 2;
        terminals.truncate(new_len);
        return true;
    }
    if !futures.is_empty() {
        futures.pop();
        return true;
    }
    if *include_executed {
        *include_executed = false;
        return true;
    }
    if *include_legend_nodes {
        *include_legend_nodes = false;
        return true;
    }
    false
}

#[allow(clippy::too_many_arguments)]
fn build_scene_and_legend(
    frame: &DecisionOverlayFrame,
    config: &OverlayConfig,
    transform: &ScaleTransform,
    include_executed: bool,
    futures: &[&FutureTrajectorySample],
    sensors: &[&SensorRaySample],
    terminals: &[TerminalMarker],
    include_legend_nodes: bool,
) -> Result<(Scene2D, OverlayLegend), ProjectionError> {
    let mut scene = Scene2D::new(
        frame.scene_id(),
        frame.tick(),
        EpistemicKind::Observed,
        config.coordinate_space,
    );

    let p = &config.palette;
    let mut observed_layer = SceneLayer::with_visibility(
        p.observed_reality.layer_id,
        p.observed_reality.z_order,
        config.visibility.observed_reality,
    );
    let mut executed_layer = SceneLayer::with_visibility(
        p.executed_trajectory.layer_id,
        p.executed_trajectory.z_order,
        config.visibility.executed_trajectory,
    );
    let mut sensors_layer = SceneLayer::with_visibility(
        p.sensor_ray.layer_id,
        p.sensor_ray.z_order,
        config.visibility.sensors,
    );
    let mut rejected_layer = SceneLayer::with_visibility(
        p.rejected_future.layer_id,
        p.rejected_future.z_order,
        config.visibility.rejected_futures,
    );
    let mut hypothesis_layer = SceneLayer::with_visibility(
        p.hypothesis.layer_id,
        p.hypothesis.z_order,
        config.visibility.hypotheses,
    );
    let mut selected_layer = SceneLayer::with_visibility(
        p.selected_future.layer_id,
        p.selected_future.z_order,
        config.visibility.selected_future,
    );
    let mut terminal_layer = SceneLayer::with_visibility(
        p.terminal_marker.layer_id,
        p.terminal_marker.z_order,
        config.visibility.terminals,
    );

    let mut counts = [0usize; 7];

    if include_executed && let Some(executed) = frame.executed_trajectory() {
        let local_pts = transform_points(executed.points_meters(), transform)?;
        let node = SceneNode::new(
            SceneNodeId::new(EXECUTED_NODE_BASE.wrapping_add(executed.trajectory_id())),
            executed.entity_id(),
            EpistemicKind::Observed,
            ScenePrimitive::polyline(local_pts, false)?,
            true,
        )
        .with_state_code(p.executed_trajectory.state_code);
        executed_layer.add_node(node)?;
        counts[1] += 1;
    }

    for (idx, future) in futures.iter().enumerate() {
        let style = p.style_for_future(future.kind());
        let local_pts = transform_points(future.points_meters(), transform)?;
        let node = SceneNode::new(
            SceneNodeId::new(FUTURE_NODE_BASE.wrapping_add(idx as u64)),
            None,
            future.kind().epistemic_kind(),
            ScenePrimitive::polyline(local_pts, false)?,
            true,
        )
        .with_state_code(style.state_code);

        match future.kind() {
            FutureKind::ObservedReality => {
                observed_layer.add_node(node)?;
                counts[0] += 1;
            }
            FutureKind::RejectedFuture => {
                rejected_layer.add_node(node)?;
                counts[3] += 1;
            }
            FutureKind::Hypothesis => {
                hypothesis_layer.add_node(node)?;
                counts[4] += 1;
            }
            FutureKind::SelectedFuture => {
                selected_layer.add_node(node)?;
                counts[5] += 1;
            }
        }
    }

    if config.visibility.sensors {
        let hit_radius = transform.transform_length(config.sensor_hit_radius_meters)?;
        for (idx, sensor) in sensors.iter().enumerate() {
            let origin = transform.transform_point(sensor.origin_meters())?;
            let hit = transform.transform_point(sensor.hit_meters())?;
            let ray_node = SceneNode::new(
                SceneNodeId::new(SENSOR_NODE_BASE.wrapping_add(idx as u64)),
                None,
                EpistemicKind::Inferred,
                ScenePrimitive::polyline(vec![origin, hit], false)?,
                false,
            )
            .with_state_code(p.sensor_ray.state_code);
            sensors_layer.add_node(ray_node)?;

            if sensor.hit_detected() {
                let hit_node = SceneNode::new(
                    SceneNodeId::new(SENSOR_HIT_NODE_BASE.wrapping_add(idx as u64)),
                    None,
                    EpistemicKind::Inferred,
                    ScenePrimitive::circle(hit, hit_radius)?,
                    false,
                )
                .with_state_code(p.sensor_ray.state_code);
                sensors_layer.add_node(hit_node)?;
            }
            counts[2] += 1;
        }
    }

    if config.visibility.terminals {
        let term_radius = transform.transform_length(config.terminal_marker_radius_meters)?;
        for (idx, term) in terminals.iter().enumerate() {
            let pos = transform.transform_point(term.position_meters())?;
            let node = SceneNode::new(
                SceneNodeId::new(TERMINAL_NODE_BASE.wrapping_add(idx as u64)),
                None,
                EpistemicKind::Hypothetical,
                ScenePrimitive::circle(pos, term_radius)?,
                false,
            )
            .with_state_code(p.terminal_marker.state_code);
            terminal_layer.add_node(node)?;
            counts[6] += 1;
        }
    }

    scene.add_layer(observed_layer)?;
    scene.add_layer(executed_layer)?;
    scene.add_layer(sensors_layer)?;
    scene.add_layer(rejected_layer)?;
    scene.add_layer(hypothesis_layer)?;
    scene.add_layer(selected_layer)?;
    scene.add_layer(terminal_layer)?;

    let legend = build_legend(config, counts);
    let mut labels_layer = SceneLayer::with_visibility(
        LayerId::Labels,
        LayerId::Labels.default_z_order(),
        config.visibility.legend,
    );
    if include_legend_nodes {
        populate_legend_nodes(&mut labels_layer, &legend, transform)?;
    }
    scene.add_layer(labels_layer)?;

    Ok((scene, legend))
}

fn build_legend(config: &OverlayConfig, counts: [usize; 7]) -> OverlayLegend {
    let p = &config.palette;
    let v = config.visibility;
    let mut entries = vec![
        LegendEntry {
            category: LegendCategory::ObservedReality,
            label: LegendCategory::ObservedReality.display_label().to_string(),
            style: p.observed_reality.clone(),
            visible: v.observed_reality,
            sample_count: counts[0],
        },
        LegendEntry {
            category: LegendCategory::ExecutedTrajectory,
            label: LegendCategory::ExecutedTrajectory
                .display_label()
                .to_string(),
            style: p.executed_trajectory.clone(),
            visible: v.executed_trajectory,
            sample_count: counts[1],
        },
        LegendEntry {
            category: LegendCategory::SensorRay,
            label: LegendCategory::SensorRay.display_label().to_string(),
            style: p.sensor_ray.clone(),
            visible: v.sensors,
            sample_count: counts[2],
        },
        LegendEntry {
            category: LegendCategory::RejectedFuture,
            label: LegendCategory::RejectedFuture.display_label().to_string(),
            style: p.rejected_future.clone(),
            visible: v.rejected_futures,
            sample_count: counts[3],
        },
        LegendEntry {
            category: LegendCategory::Hypothesis,
            label: LegendCategory::Hypothesis.display_label().to_string(),
            style: p.hypothesis.clone(),
            visible: v.hypotheses,
            sample_count: counts[4],
        },
        LegendEntry {
            category: LegendCategory::SelectedFuture,
            label: LegendCategory::SelectedFuture.display_label().to_string(),
            style: p.selected_future.clone(),
            visible: v.selected_future,
            sample_count: counts[5],
        },
        LegendEntry {
            category: LegendCategory::TerminalMarker,
            label: LegendCategory::TerminalMarker.display_label().to_string(),
            style: p.terminal_marker.clone(),
            visible: v.terminals,
            sample_count: counts[6],
        },
    ];
    entries.sort_by_key(|e| (e.style.z_order, e.category));
    OverlayLegend { entries }
}

fn populate_legend_nodes(
    layer: &mut SceneLayer,
    legend: &OverlayLegend,
    transform: &ScaleTransform,
) -> Result<(), ProjectionError> {
    let swatch_size = transform.transform_length(0.20)?;
    for (idx, entry) in legend.entries().iter().enumerate() {
        let y_meters = -(idx as f64) * 0.45;
        let swatch_pos = transform.transform_point(Point2::new(0.0, y_meters)?)?;
        let label_pos = transform.transform_point(Point2::new(0.35, y_meters)?)?;

        let swatch = SceneNode::new(
            SceneNodeId::new(LEGEND_SWATCH_NODE_BASE.wrapping_add(idx as u64)),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::rect(swatch_pos, swatch_size, swatch_size)?,
            false,
        )
        .with_state_code(entry.style.state_code);
        layer.add_node(swatch)?;

        let text = format!(
            "{} | {} {} w={:.1} a={:.2} z={} n={}",
            entry.label,
            entry.style.color.to_hex(),
            entry.style.dash.as_str(),
            entry.style.stroke_width,
            entry.style.opacity,
            entry.style.z_order,
            entry.sample_count
        );
        let label = SceneNode::new(
            SceneNodeId::new(LEGEND_LABEL_NODE_BASE.wrapping_add(idx as u64)),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::label(label_pos, text),
            false,
        )
        .with_state_code(entry.style.state_code);
        layer.add_node(label)?;
    }
    Ok(())
}

fn transform_points(
    points_meters: &[Point2],
    transform: &ScaleTransform,
) -> Result<Vec<Point2>, ProjectionError> {
    let mut local = Vec::with_capacity(points_meters.len());
    for &pt in points_meters {
        local.push(transform.transform_point(pt)?);
    }
    Ok(local)
}

#[cfg(test)]
mod enforcement_tests;
