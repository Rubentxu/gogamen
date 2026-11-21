//! Grid parcialmente observable con oclusión, mapa de creencias y fuente humana falible (`std`-only).
//!
//! Este módulo impone una separación estricta entre el estado real del mundo ([`GroundTruthGrid`])
//! y la vista observable expuesta al agente/planificador ([`PartialGridObserver`], [`OccludedObservation`]):
//! - [`GroundTruthGrid`] mantiene el [`GridMap`] real de forma privada y solo emite
//!   [`OccludedObservation`] con las celdas dentro de `vision_radius` y con línea de visión
//!   libre de muros ([`CellKind::Wall`]).
//! - [`GridBeliefMap`] mantiene creencias binarias calibradas por celda e hipótesis
//!   ([`HypothesisKind::Obstacle`], [`HypothesisKind::Hazard`], [`HypothesisKind::Goal`])
//!   usando [`BinaryBelief`], [`LikelihoodModel`], [`EvidenceDeduplicator`] y
//!   [`BeliefUpdateStatus`] de `gogamen-core`.
//!   - [`HumanReport`] (`ObservationEnvelope<CellHypothesis>`, `ObservationKind::Assertion`)
//!     actualiza hipótesis con la sensibilidad/especificidad falibles configuradas.
//!   - [`DirectObservation`] (`ObservationEnvelope<VisibleCell>`, `ObservationKind::Measurement`)
//!     actualiza con alta fiabilidad y corrige explícitamente cualquier creencia previa
//!     contradicha ([`BeliefCorrection`]).
//! - [`FallibleHumanFixture`] genera casos seeded reproducibles mediante [`MasterSeed`] y
//!   [`DeterministicRng`] que cubren falso positivo, falso negativo y dato antiguo (`Stale`).
//! - [`BeliefGridPlanner`] planifica acciones consultando exclusivamente [`GridBeliefMap`]
//!   y nunca recibe acceso a [`GroundTruthGrid`] ni a [`GridMap`].

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use gogamen_core::{
    BeliefError, BeliefUpdateStatus, BinaryBelief, Confidence, DeterministicRng,
    DiscreteBeliefFilter, DiscreteBeliefState, DomainError, EvidenceDeduplicator, EvidenceRef,
    FrameId, LikelihoodModel, MasterSeed, MessageId, MonotonicInstant, ObservationEnvelope,
    ObservationKind, ProvenanceError, ProvenanceMetadata, RngSource, SensorId, StreamKey, UnitTag,
};

use crate::map::{CellKind, GridMap, GridPos, GridSize};
use crate::world::GridAction;

/// Identificador por defecto del sensor local de observación directa.
pub const DEFAULT_DIRECT_SENSOR_ID: u64 = 1;
/// Identificador por defecto de la fuente humana falible.
pub const DEFAULT_HUMAN_SENSOR_ID: u64 = 100;

/// Errores explícitos del subsistema de Grid parcialmente observable.
#[derive(Debug, Clone, PartialEq)]
pub enum PartialObsError {
    /// Una coordenada consultada o reportada cae fuera de los límites de la rejilla.
    OutOfBounds {
        /// Posición inválida.
        pos: GridPos,
        /// Tamaño de la rejilla.
        size: GridSize,
    },
    /// Error en validación o actualización bayesiana de `gogamen-core`.
    Belief(BeliefError),
    /// Error de metadatos de procedencia de `gogamen-core`.
    Provenance(ProvenanceError),
    /// Error de dominio de `gogamen-core`.
    Domain(DomainError),
    /// Precondición incumplida al construir un caso controlado en el fixture.
    InvalidFixturePrecondition {
        /// Motivo concreto de la precondición incumplida.
        reason: &'static str,
    },
}

impl fmt::Display for PartialObsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds { pos, size } => write!(
                f,
                "grid position ({}, {}) out of bounds for {}x{} grid",
                pos.x, pos.y, size.width, size.height
            ),
            Self::Belief(err) => write!(f, "belief error: {err}"),
            Self::Provenance(err) => write!(f, "provenance error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::InvalidFixturePrecondition { reason } => {
                write!(f, "invalid fixture precondition: {reason}")
            }
        }
    }
}

impl Error for PartialObsError {}

impl From<BeliefError> for PartialObsError {
    fn from(value: BeliefError) -> Self {
        Self::Belief(value)
    }
}

impl From<ProvenanceError> for PartialObsError {
    fn from(value: ProvenanceError) -> Self {
        Self::Provenance(value)
    }
}

impl From<DomainError> for PartialObsError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

/// Tipo de hipótesis binaria mantenida por celda en [`GridBeliefMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HypothesisKind {
    /// Presencia de un muro u obstáculo infranqueable (`CellKind::Wall`).
    Obstacle,
    /// Presencia de una celda de peligro (`CellKind::Hazard`).
    Hazard,
    /// Presencia de la celda objetivo (`CellKind::Goal`).
    Goal,
}

impl HypothesisKind {
    /// Lista ordenada de todas las hipótesis rastreadas por celda.
    pub const ALL: [Self; 3] = [Self::Obstacle, Self::Hazard, Self::Goal];

    /// Índice estable en `[0, 3)` para indexación densa determinista.
    pub const fn index(self) -> usize {
        match self {
            Self::Obstacle => 0,
            Self::Hazard => 1,
            Self::Goal => 2,
        }
    }

    /// Evalúa si el [`CellKind`] real satisface esta hipótesis.
    pub const fn matches_cell(self, cell: CellKind) -> bool {
        match self {
            Self::Obstacle => matches!(cell, CellKind::Wall),
            Self::Hazard => matches!(cell, CellKind::Hazard),
            Self::Goal => matches!(cell, CellKind::Goal),
        }
    }
}

/// Carga útil de una afirmación humana sobre una hipótesis de celda.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellHypothesis {
    /// Posición de la celda a la que refiere la afirmación.
    pub pos: GridPos,
    /// Hipótesis evaluada sobre la celda.
    pub hypothesis: HypothesisKind,
    /// `true` si afirma presencia de la hipótesis; `false` si afirma ausencia.
    pub present: bool,
}

/// Informe humano falible envuelto con metadatos de procedencia (`ObservationKind::Assertion`).
pub type HumanReport = ObservationEnvelope<CellHypothesis>;

/// Carga útil de una celda medida directamente dentro de la línea de visión no ocluida.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisibleCell {
    /// Posición de la celda visible.
    pub pos: GridPos,
    /// Tipo de celda medido directamente por el sensor local.
    pub kind: CellKind,
}

/// Observación directa de alta fiabilidad con procedencia (`ObservationKind::Measurement`).
pub type DirectObservation = ObservationEnvelope<VisibleCell>;

/// Observación parcial emitida por [`GroundTruthGrid`] con oclusión por muros.
///
/// Solo contiene las celdas visibles dentro de `vision_radius` cuya línea de visión
/// desde `agent_pos` no queda bloqueada por [`CellKind::Wall`]. No retiene ninguna
/// referencia al [`GridMap`] real.
#[derive(Debug, Clone, PartialEq)]
pub struct OccludedObservation {
    agent_pos: GridPos,
    grid_size: GridSize,
    vision_radius: usize,
    observed_at: MonotonicInstant,
    visible_cells: Vec<DirectObservation>,
}

impl OccludedObservation {
    /// Posición del agente desde la que se tomó la observación.
    pub const fn agent_pos(&self) -> GridPos {
        self.agent_pos
    }

    /// Dimensiones totales de la rejilla.
    pub const fn grid_size(&self) -> GridSize {
        self.grid_size
    }

    /// Radio de visión local (distancia de Chebyshev).
    pub const fn vision_radius(&self) -> usize {
        self.vision_radius
    }

    /// Instante monotónico en que se tomó la observación.
    pub const fn observed_at(&self) -> MonotonicInstant {
        self.observed_at
    }

    /// Slice de todas las celdas visibles no ocluidas como [`DirectObservation`].
    pub fn visible_cells(&self) -> &[DirectObservation] {
        &self.visible_cells
    }

    /// Devuelve `true` si `pos` fue visible en esta observación.
    pub fn contains_pos(&self, pos: GridPos) -> bool {
        self.visible_cells
            .iter()
            .any(|obs| obs.payload().pos == pos)
    }

    /// Devuelve el [`CellKind`] observado en `pos`, o `None` si estuvo ocluida o fuera de radio.
    pub fn cell_at(&self, pos: GridPos) -> Option<CellKind> {
        self.visible_cells
            .iter()
            .find(|obs| obs.payload().pos == pos)
            .map(|obs| obs.payload().kind)
    }
}

/// API de observación parcialmente observable expuesta al planificador.
///
/// # Aislamiento del ground truth en tiempo de compilación
/// Un planificador que solo recibe `&mut impl PartialGridObserver` no puede consultar
/// el [`GridMap`] interno ni leer celdas ocluidas:
///
/// ```compile_fail
/// use gogamen_lab_grid::{GridPos, PartialGridObserver};
///
/// fn cheat_planner(obs: &mut impl PartialGridObserver, hidden: GridPos) {
///     let _ = obs.map();
///     let _ = obs.truth_cell(hidden);
/// }
/// ```
pub trait PartialGridObserver {
    /// Dimensiones de la rejilla (sin exponer su contenido).
    fn grid_size(&self) -> GridSize;

    /// Posición actual del agente.
    fn agent_pos(&self) -> GridPos;

    /// Radio máximo de visión local.
    fn vision_radius(&self) -> usize;

    /// Emite una [`OccludedObservation`] en `observed_at` con solo las celdas visibles.
    fn observe_occluded(&mut self, observed_at: MonotonicInstant) -> OccludedObservation;
}

/// Estado ground truth del Grid, encapsulado y aislado de los planificadores.
///
/// Mantiene privado el [`GridMap`] real. La única vía pública para percibir celdas es
/// [`GroundTruthGrid::observe_occluded`] (o [`PartialGridObserver`] mediante
/// [`GroundTruthGrid::observer_view`]), que aplica radio de visión y oclusión por muros.
#[derive(Debug, Clone)]
pub struct GroundTruthGrid {
    map: GridMap,
    agent_pos: GridPos,
    vision_radius: usize,
    direct_sensor_id: SensorId,
    next_message_id: u64,
    next_evidence_ref: u64,
}

impl GroundTruthGrid {
    /// Crea un nuevo entorno ground truth posicionado en `map.start()`.
    pub fn new(map: GridMap, vision_radius: usize) -> Self {
        let agent_pos = map.start();
        Self {
            map,
            agent_pos,
            vision_radius,
            direct_sensor_id: SensorId::new(DEFAULT_DIRECT_SENSOR_ID),
            next_message_id: 1,
            next_evidence_ref: 10_000,
        }
    }

    /// Configura un [`SensorId`] explícito para las mediciones directas emitidas.
    pub const fn with_sensor_id(mut self, sensor_id: SensorId) -> Self {
        self.direct_sensor_id = sensor_id;
        self
    }

    /// Dimensiones de la rejilla.
    pub const fn grid_size(&self) -> GridSize {
        self.map.size()
    }

    /// Posición actual del agente en el mundo real.
    pub const fn agent_pos(&self) -> GridPos {
        self.agent_pos
    }

    /// Radio máximo de visión local (distancia de Chebyshev).
    pub const fn vision_radius(&self) -> usize {
        self.vision_radius
    }

    /// Reposiciona al agente en una celda transitable del mundo real.
    pub fn set_agent_pos(&mut self, pos: GridPos) -> Result<(), PartialObsError> {
        if !self.map.in_bounds(pos) {
            return Err(PartialObsError::OutOfBounds {
                pos,
                size: self.map.size(),
            });
        }
        if !self.map.is_passable(pos) {
            return Err(PartialObsError::Domain(DomainError::InvalidTransition {
                reason: "agent position cannot be inside a wall",
            }));
        }
        self.agent_pos = pos;
        Ok(())
    }

    /// Ejecuta un movimiento discreto del agente respetando muros y bordes del ground truth.
    pub fn step_agent(&mut self, action: GridAction) -> GridPos {
        let (dx, dy) = action.delta();
        let nx = self.agent_pos.x as isize + dx;
        let ny = self.agent_pos.y as isize + dy;
        if nx >= 0 && ny >= 0 {
            let candidate = GridPos::new(nx as usize, ny as usize);
            if self.map.is_passable(candidate) {
                self.agent_pos = candidate;
            }
        }
        self.agent_pos
    }

    /// Evalúa si `target` es visible desde la posición actual del agente (dentro de
    /// `vision_radius` y con línea de visión no bloqueada por muros).
    pub fn is_visible(&self, target: GridPos) -> bool {
        if !self.map.in_bounds(target) {
            return false;
        }
        let chebyshev = self
            .agent_pos
            .x
            .abs_diff(target.x)
            .max(self.agent_pos.y.abs_diff(target.y));
        if chebyshev > self.vision_radius {
            return false;
        }
        has_line_of_sight(&self.map, self.agent_pos, target)
    }

    /// Construye una [`OccludedObservation`] en `observed_at` con solo las celdas visibles.
    pub fn observe_occluded(&mut self, observed_at: MonotonicInstant) -> OccludedObservation {
        let r = self.vision_radius as isize;
        let cx = self.agent_pos.x as isize;
        let cy = self.agent_pos.y as isize;
        let mut visible_cells = Vec::new();

        for dy in -r..=r {
            for dx in -r..=r {
                let tx = cx + dx;
                let ty = cy + dy;
                if tx < 0 || ty < 0 {
                    continue;
                }
                let pos = GridPos::new(tx as usize, ty as usize);
                if !self.is_visible(pos) {
                    continue;
                }
                if let Some(kind) = self.map.cell(pos) {
                    let message_id = MessageId::new(self.next_message_id);
                    let evidence_ref = EvidenceRef::new(self.next_evidence_ref);
                    self.next_message_id += 1;
                    self.next_evidence_ref += 1;

                    let meta = ProvenanceMetadata {
                        message_id,
                        observed_at,
                        received_at: observed_at,
                        sensor_id: self.direct_sensor_id,
                        frame_id: FrameId::world(),
                        units: UnitTag::dimensionless(),
                        confidence: Confidence::FULL,
                        revision: message_id.as_u64(),
                        evidence_ref,
                        claim_kind: ObservationKind::Measurement,
                    };
                    visible_cells.push(ObservationEnvelope::new(meta, VisibleCell { pos, kind }));
                }
            }
        }

        OccludedObservation {
            agent_pos: self.agent_pos,
            grid_size: self.map.size(),
            vision_radius: self.vision_radius,
            observed_at,
            visible_cells,
        }
    }

    /// Devuelve una vista de observación restringida que solo implementa [`PartialGridObserver`].
    pub fn observer_view(&mut self) -> GroundTruthObservationView<'_> {
        GroundTruthObservationView { inner: self }
    }

    /// Acceso privado de módulo usado exclusivamente por [`FallibleHumanFixture`].
    fn truth_cell(&self, pos: GridPos) -> Option<CellKind> {
        self.map.cell(pos)
    }

    /// Acceso privado de módulo usado exclusivamente por [`FallibleHumanFixture`].
    fn map(&self) -> &GridMap {
        &self.map
    }
}

impl PartialGridObserver for GroundTruthGrid {
    fn grid_size(&self) -> GridSize {
        self.grid_size()
    }

    fn agent_pos(&self) -> GridPos {
        self.agent_pos()
    }

    fn vision_radius(&self) -> usize {
        self.vision_radius()
    }

    fn observe_occluded(&mut self, observed_at: MonotonicInstant) -> OccludedObservation {
        GroundTruthGrid::observe_occluded(self, observed_at)
    }
}

/// Vista prestada de [`GroundTruthGrid`] que solo expone la interfaz [`PartialGridObserver`].
#[derive(Debug)]
pub struct GroundTruthObservationView<'a> {
    inner: &'a mut GroundTruthGrid,
}

impl PartialGridObserver for GroundTruthObservationView<'_> {
    fn grid_size(&self) -> GridSize {
        self.inner.grid_size()
    }

    fn agent_pos(&self) -> GridPos {
        self.inner.agent_pos()
    }

    fn vision_radius(&self) -> usize {
        self.inner.vision_radius()
    }

    fn observe_occluded(&mut self, observed_at: MonotonicInstant) -> OccludedObservation {
        self.inner.observe_occluded(observed_at)
    }
}

/// Verifica línea de visión discreta (Bresenham) desde `origin` hasta `target`.
///
/// - Las celdas estrictamente intermedias con `CellKind::Wall` bloquean la visión.
/// - En pasos diagonales puros, un rincón cerrado por dos muros ortogonales también bloquea
///   la visión para evitar fugas diagonales entre esquinas.
/// - La propia celda `target` (aunque sea `Wall`) sí es visible si nada la bloquea antes.
pub(crate) fn has_line_of_sight(map: &GridMap, origin: GridPos, target: GridPos) -> bool {
    if origin == target {
        return true;
    }

    let mut x = origin.x as isize;
    let mut y = origin.y as isize;
    let x1 = target.x as isize;
    let y1 = target.y as isize;

    let dx = (x1 - x).abs();
    let dy = -(y1 - y).abs();
    let sx = if x < x1 { 1 } else { -1 };
    let sy = if y < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        let prev_x = x;
        let prev_y = y;
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }

        if x != prev_x && y != prev_y {
            let side_a = GridPos::new(x as usize, prev_y as usize);
            let side_b = GridPos::new(prev_x as usize, y as usize);
            if map.cell(side_a).is_none_or(CellKind::is_blocking)
                && map.cell(side_b).is_none_or(CellKind::is_blocking)
            {
                return false;
            }
        }

        let pos = GridPos::new(x as usize, y as usize);
        if pos == target {
            return true;
        }
        if map.cell(pos).is_none_or(CellKind::is_blocking) {
            return false;
        }
    }
}

/// Configuración de priors y modelos calibrados de verosimilitud para [`GridBeliefMap`].
#[derive(Debug, Clone, PartialEq)]
pub struct GridBeliefConfig {
    /// Prior inicial `P(Obstacle)` por celda.
    pub prior_obstacle: f64,
    /// Prior inicial `P(Hazard)` por celda.
    pub prior_hazard: f64,
    /// Prior inicial `P(Goal)` por celda.
    pub prior_goal: f64,
    /// Modelo calibrado de verosimilitud para informes humanos (`Assertion`).
    pub human_model: LikelihoodModel,
    /// Modelo calibrado de alta fiabilidad para observaciones directas (`Measurement`).
    pub direct_model: LikelihoodModel,
}

impl Default for GridBeliefConfig {
    fn default() -> Self {
        Self {
            prior_obstacle: 0.25,
            prior_hazard: 0.25,
            prior_goal: 0.25,
            human_model: LikelihoodModel::from_sensitivity_specificity(0.80, 0.80)
                .expect("default human likelihood model must be valid"),
            direct_model: LikelihoodModel::from_sensitivity_specificity(0.99, 0.99)
                .expect("default direct likelihood model must be valid"),
        }
    }
}

impl GridBeliefConfig {
    /// Construye una configuración validando priors y modelos binarios `2x2`.
    pub fn new(
        prior_obstacle: f64,
        prior_hazard: f64,
        prior_goal: f64,
        human_sensitivity: f64,
        human_specificity: f64,
        direct_sensitivity: f64,
        direct_specificity: f64,
    ) -> Result<Self, PartialObsError> {
        let _ = BinaryBelief::new(prior_obstacle)?;
        let _ = BinaryBelief::new(prior_hazard)?;
        let _ = BinaryBelief::new(prior_goal)?;
        let human_model =
            LikelihoodModel::from_sensitivity_specificity(human_sensitivity, human_specificity)?;
        let direct_model =
            LikelihoodModel::from_sensitivity_specificity(direct_sensitivity, direct_specificity)?;
        Ok(Self {
            prior_obstacle,
            prior_hazard,
            prior_goal,
            human_model,
            direct_model,
        })
    }

    fn prior_for(&self, hypothesis: HypothesisKind) -> Result<BinaryBelief, BeliefError> {
        let p = match hypothesis {
            HypothesisKind::Obstacle => self.prior_obstacle,
            HypothesisKind::Hazard => self.prior_hazard,
            HypothesisKind::Goal => self.prior_goal,
        };
        BinaryBelief::new(p)
    }
}

/// Registro de una contradicción observable en la que una observación directa corrigió la hipótesis.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefCorrection {
    /// Celda corregida.
    pub pos: GridPos,
    /// Hipótesis cuya clasificación MAP cambió tras la observación directa.
    pub hypothesis: HypothesisKind,
    /// Probabilidad `P(true)` previa a la observación directa.
    pub prior_p_true: f64,
    /// Probabilidad `P(true)` posterior a la observación directa.
    pub posterior_p_true: f64,
    /// Valor observado directamente (`true` = presente, `false` = ausente).
    pub observed_present: bool,
    /// Instante monotónico de la medición directa correctora.
    pub observed_at: MonotonicInstant,
}

/// Resultado de aplicar una [`DirectObservation`] sobre las tres hipótesis de una celda.
#[derive(Debug, Clone, PartialEq)]
pub struct DirectObservationOutcome {
    /// Posición de la celda actualizada.
    pub pos: GridPos,
    /// Estado de actualización por cada hipótesis (`[Obstacle, Hazard, Goal]`).
    pub statuses: [BeliefUpdateStatus; 3],
    /// Correcciones por contradicción detectadas en esta celda.
    pub corrections: Vec<BeliefCorrection>,
}

/// Resumen agregado de aplicar una [`OccludedObservation`] completa sobre [`GridBeliefMap`].
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBatchReport {
    /// Número de celdas visibles procesadas.
    pub cells_processed: usize,
    /// Número de actualizaciones de hipótesis aceptadas (`Updated`).
    pub hypotheses_updated: usize,
    /// Número de abstenciones por dato antiguo (`AbstainedStaleObservation`).
    pub stale_abstained: usize,
    /// Número de duplicados ignorados (`DuplicateEvidenceIgnored`).
    pub duplicates_ignored: usize,
    /// Correcciones por contradicción registradas durante el lote.
    pub corrections: Vec<BeliefCorrection>,
}

/// Estado escalar `Copy` por `(celda, hipótesis)`; almacenado de forma plana en [`GridBeliefMap`].
///
/// Guarda `[P(false), P(true)]` tal cual lo produce [`BinaryBelief`] para preservar la
/// igualdad bit a bit con la implementación previa.
#[derive(Debug, Clone, Copy, PartialEq)]
struct CellHypothesisState {
    probs: [f64; 2],
    watermark: Option<MonotonicInstant>,
    human_updates: u64,
    direct_updates: u64,
}

impl CellHypothesisState {
    fn from_belief(belief: &BinaryBelief) -> Self {
        Self {
            probs: [belief.p_false(), belief.p_true()],
            watermark: None,
            human_updates: 0,
            direct_updates: 0,
        }
    }

    fn belief(&self) -> Result<BinaryBelief, BeliefError> {
        BinaryBelief::from_discrete(DiscreteBeliefState::new(self.probs.to_vec())?)
    }
}

/// Deduplicador centralizado: reemplaza un `EvidenceDeduplicator` por `(celda, hipótesis)`.
///
/// Mantiene la semántica previa: una observación es duplicada para el índice `idx` si su
/// `EvidenceRef` o su `(SensorId, MessageId)` ya fueron registrados **para ese mismo índice**.
#[derive(Debug, Clone, Default)]
struct CentralEvidenceLedger {
    seen_evidence: HashSet<(usize, EvidenceRef)>,
    seen_messages: HashSet<(usize, SensorId, MessageId)>,
}

impl CentralEvidenceLedger {
    fn is_duplicate(
        &self,
        idx: usize,
        sensor_id: SensorId,
        message_id: MessageId,
        evidence_ref: EvidenceRef,
    ) -> bool {
        self.seen_evidence.contains(&(idx, evidence_ref))
            || self.seen_messages.contains(&(idx, sensor_id, message_id))
    }

    fn record(
        &mut self,
        idx: usize,
        sensor_id: SensorId,
        message_id: MessageId,
        evidence_ref: EvidenceRef,
    ) {
        self.seen_evidence.insert((idx, evidence_ref));
        self.seen_messages.insert((idx, sensor_id, message_id));
    }
}

/// Mapa de creencias por celda e hipótesis actualizado por informes humanos y observaciones directas.
///
/// El estado escalar vive en un único `Vec` plano de structs `Copy` y la evidencia observada en
/// un único [`CentralEvidenceLedger`]; clonar el mapa no clona tablas hash por celda.
#[derive(Debug, Clone)]
pub struct GridBeliefMap {
    size: GridSize,
    config: GridBeliefConfig,
    entries: Vec<CellHypothesisState>,
    ledger: CentralEvidenceLedger,
    corrections: Vec<BeliefCorrection>,
}

/// Aplica una observación sobre `(entry, idx)` usando el ledger centralizado.
///
/// Delega en [`BinaryBelief::update_with_observation`] con un deduplicador local que solo
/// contiene la evidencia de `envelope` si el ledger ya la considera duplicada, de modo que el
/// orden de validaciones y el resultado coinciden exactamente con el deduplicador por celda.
fn update_entry(
    entry: &mut CellHypothesisState,
    ledger: &mut CentralEvidenceLedger,
    idx: usize,
    envelope: &ObservationEnvelope<bool>,
    model: &LikelihoodModel,
) -> Result<BeliefUpdateStatus, PartialObsError> {
    let (sensor, message, evidence) = (
        envelope.sensor_id(),
        envelope.message_id(),
        envelope.evidence_ref(),
    );
    let mut local = EvidenceDeduplicator::new();
    if ledger.is_duplicate(idx, sensor, message, evidence) {
        local.record(sensor, message, evidence);
    }
    let mut belief = entry.belief()?;
    let mut watermark = entry.watermark;
    let status = belief.update_with_observation(envelope, model, &mut local, &mut watermark)?;
    if status.is_updated() {
        ledger.record(idx, sensor, message, evidence);
        entry.probs = [belief.p_false(), belief.p_true()];
        entry.watermark = watermark;
    }
    Ok(status)
}

impl GridBeliefMap {
    /// Crea un mapa de creencias inicializado con los priors configurados para cada celda.
    pub fn new(size: GridSize, config: GridBeliefConfig) -> Result<Self, PartialObsError> {
        if size.width == 0 || size.height == 0 {
            return Err(PartialObsError::Domain(DomainError::InvalidTransition {
                reason: "grid belief map dimensions must be non-zero",
            }));
        }
        let prior_obs = config.prior_for(HypothesisKind::Obstacle)?;
        let prior_haz = config.prior_for(HypothesisKind::Hazard)?;
        let prior_goal = config.prior_for(HypothesisKind::Goal)?;

        let total_cells = size.cell_count();
        let mut entries = Vec::with_capacity(total_cells * HypothesisKind::ALL.len());
        for _ in 0..total_cells {
            for belief in [&prior_obs, &prior_haz, &prior_goal] {
                entries.push(CellHypothesisState::from_belief(belief));
            }
        }

        Ok(Self {
            size,
            config,
            entries,
            ledger: CentralEvidenceLedger::default(),
            corrections: Vec::new(),
        })
    }

    /// Número de estados escalares planos `(celda, hipótesis)` almacenados.
    #[cfg(test)]
    fn scalar_state_len(&self) -> usize {
        self.entries.len()
    }

    /// Número de pares `(índice, EvidenceRef)` y `(índice, SensorId, MessageId)` registrados.
    #[cfg(test)]
    fn ledger_sizes(&self) -> (usize, usize) {
        (
            self.ledger.seen_evidence.len(),
            self.ledger.seen_messages.len(),
        )
    }

    /// Dimensiones de la rejilla de creencias.
    pub const fn size(&self) -> GridSize {
        self.size
    }

    /// Configuración de priors y verosimilitudes del mapa de creencias.
    pub const fn config(&self) -> &GridBeliefConfig {
        &self.config
    }

    /// Historial acumulado de correcciones por contradicción observable.
    pub fn corrections(&self) -> &[BeliefCorrection] {
        &self.corrections
    }

    fn entry_index(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<usize, PartialObsError> {
        if pos.x >= self.size.width || pos.y >= self.size.height {
            return Err(PartialObsError::OutOfBounds {
                pos,
                size: self.size,
            });
        }
        let cell_idx = pos.y * self.size.width + pos.x;
        Ok(cell_idx * HypothesisKind::ALL.len() + hypothesis.index())
    }

    /// Devuelve (por valor) la [`BinaryBelief`] actual para `(pos, hypothesis)`.
    pub fn belief(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<BinaryBelief, PartialObsError> {
        let idx = self.entry_index(pos, hypothesis)?;
        Ok(self.entries[idx].belief()?)
    }

    /// Devuelve `P(true)` actual para `(pos, hypothesis)`.
    pub fn probability(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<f64, PartialObsError> {
        let idx = self.entry_index(pos, hypothesis)?;
        Ok(self.entries[idx].probs[1])
    }

    /// Devuelve el watermark monotónico más reciente aceptado para `(pos, hypothesis)`.
    pub fn watermark(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<Option<MonotonicInstant>, PartialObsError> {
        let idx = self.entry_index(pos, hypothesis)?;
        Ok(self.entries[idx].watermark)
    }

    /// Número de actualizaciones humanas aceptadas para `(pos, hypothesis)`.
    pub fn human_update_count(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<u64, PartialObsError> {
        let idx = self.entry_index(pos, hypothesis)?;
        Ok(self.entries[idx].human_updates)
    }

    /// Número de actualizaciones por observación directa aceptadas para `(pos, hypothesis)`.
    pub fn direct_update_count(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
    ) -> Result<u64, PartialObsError> {
        let idx = self.entry_index(pos, hypothesis)?;
        Ok(self.entries[idx].direct_updates)
    }

    /// Construye un [`DiscreteBeliefFilter`] inicializado con la creencia actual de `(pos, hypothesis)`
    /// y el [`LikelihoodModel`] especificado.
    pub fn to_discrete_filter(
        &self,
        pos: GridPos,
        hypothesis: HypothesisKind,
        model: LikelihoodModel,
    ) -> Result<DiscreteBeliefFilter, PartialObsError> {
        let belief = self.belief(pos, hypothesis)?;
        Ok(DiscreteBeliefFilter::new(belief.into_discrete(), model)?)
    }

    /// Aplica un [`HumanReport`] (`ObservationKind::Assertion`) calibrado con `config.human_model`.
    ///
    /// - Rechaza con [`BeliefUpdateStatus::AbstainedUnsupportedKind`] cualquier envelope que no
    ///   sea `ObservationKind::Assertion`.
    /// - Rechaza con [`BeliefUpdateStatus::AbstainedStaleObservation`] cualquier reporte con
    ///   `observed_at < watermark` sin degradar la creencia más reciente.
    /// - Deduplica rumores/reenvíos por `EvidenceRef` y `(SensorId, MessageId)`.
    pub fn apply_human_report(
        &mut self,
        report: &HumanReport,
    ) -> Result<BeliefUpdateStatus, PartialObsError> {
        let payload = *report.payload();
        let idx = self.entry_index(payload.pos, payload.hypothesis)?;

        if report.claim_kind() != ObservationKind::Assertion {
            return Ok(BeliefUpdateStatus::AbstainedUnsupportedKind);
        }

        let bool_env = report.clone().map(|h| h.present);
        let status = update_entry(
            &mut self.entries[idx],
            &mut self.ledger,
            idx,
            &bool_env,
            &self.config.human_model,
        )?;
        if status.is_updated() {
            self.entries[idx].human_updates += 1;
        }
        Ok(status)
    }

    /// Aplica una [`DirectObservation`] (`ObservationKind::Measurement`) con `config.direct_model`.
    ///
    /// Actualiza las tres hipótesis (`Obstacle`, `Hazard`, `Goal`) de la celda visible y registra
    /// una [`BeliefCorrection`] cada vez que la observación directa contradice y revierte la
    /// clasificación previa (`prior_p_true >= 0.5`).
    pub fn apply_direct_observation(
        &mut self,
        obs: &DirectObservation,
    ) -> Result<DirectObservationOutcome, PartialObsError> {
        let payload = *obs.payload();
        // Valida límites antes de actualizar cualquier hipótesis.
        let _ = self.entry_index(payload.pos, HypothesisKind::Obstacle)?;

        if obs.claim_kind() != ObservationKind::Measurement {
            return Ok(DirectObservationOutcome {
                pos: payload.pos,
                statuses: [BeliefUpdateStatus::AbstainedUnsupportedKind; 3],
                corrections: Vec::new(),
            });
        }

        let mut statuses = [BeliefUpdateStatus::AbstainedUnsupportedKind; 3];
        let mut cell_corrections = Vec::new();

        for hypothesis in HypothesisKind::ALL {
            let idx = self.entry_index(payload.pos, hypothesis)?;
            let observed_present = hypothesis.matches_cell(payload.kind);
            let bool_env = obs.clone().map(|_| observed_present);

            let entry = &mut self.entries[idx];
            let prior_p = entry.probs[1];
            let status = update_entry(
                entry,
                &mut self.ledger,
                idx,
                &bool_env,
                &self.config.direct_model,
            )?;
            statuses[hypothesis.index()] = status;

            if status.is_updated() {
                entry.direct_updates += 1;
                let posterior_p = entry.probs[1];
                let prior_map = prior_p >= 0.5;
                let posterior_map = posterior_p >= 0.5;
                if prior_map != observed_present && posterior_map == observed_present {
                    let correction = BeliefCorrection {
                        pos: payload.pos,
                        hypothesis,
                        prior_p_true: prior_p,
                        posterior_p_true: posterior_p,
                        observed_present,
                        observed_at: obs.observed_at(),
                    };
                    cell_corrections.push(correction.clone());
                    self.corrections.push(correction);
                }
            }
        }

        Ok(DirectObservationOutcome {
            pos: payload.pos,
            statuses,
            corrections: cell_corrections,
        })
    }

    /// Aplica todas las celdas visibles de una [`OccludedObservation`] sobre el mapa de creencias.
    pub fn apply_occluded_observation(
        &mut self,
        obs: &OccludedObservation,
    ) -> Result<ObservationBatchReport, PartialObsError> {
        let mut hypotheses_updated = 0usize;
        let mut stale_abstained = 0usize;
        let mut duplicates_ignored = 0usize;
        let mut corrections = Vec::new();

        for direct in obs.visible_cells() {
            let outcome = self.apply_direct_observation(direct)?;
            for status in outcome.statuses {
                match status {
                    BeliefUpdateStatus::Updated => hypotheses_updated += 1,
                    BeliefUpdateStatus::AbstainedStaleObservation => stale_abstained += 1,
                    BeliefUpdateStatus::DuplicateEvidenceIgnored => duplicates_ignored += 1,
                    BeliefUpdateStatus::ConflictImpossibleObservation
                    | BeliefUpdateStatus::AbstainedUnsupportedKind => {}
                }
            }
            corrections.extend(outcome.corrections);
        }

        Ok(ObservationBatchReport {
            cells_processed: obs.visible_cells().len(),
            hypotheses_updated,
            stale_abstained,
            duplicates_ignored,
            corrections,
        })
    }
}

/// Planificador que decide acciones exclusivamente a partir de [`GridBeliefMap`].
///
/// Por construcción de firma, nunca recibe [`GroundTruthGrid`] ni [`GridMap`]; solo
/// consulta creencias locales actualizadas mediante [`PartialGridObserver`] e informes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeliefGridPlanner {
    /// Peso negativo aplicado a `P(Obstacle)` al evaluar una celda candidata.
    pub obstacle_penalty: f64,
    /// Peso negativo aplicado a `P(Hazard)` al evaluar una celda candidata.
    pub hazard_penalty: f64,
    /// Peso positivo aplicado a `P(Goal)` al evaluar una celda candidata.
    pub goal_bonus: f64,
    /// Peso positivo para reducir incertidumbre (preferencia por explorar celdas poco observadas).
    pub exploration_bonus: f64,
}

impl Default for BeliefGridPlanner {
    fn default() -> Self {
        Self {
            obstacle_penalty: 10.0,
            hazard_penalty: 8.0,
            goal_bonus: 15.0,
            exploration_bonus: 1.0,
        }
    }
}

impl BeliefGridPlanner {
    /// Observa a través de [`PartialGridObserver`], actualiza [`GridBeliefMap`] y elige
    /// la mejor [`GridAction`] usando únicamente las creencias.
    pub fn observe_and_choose_action(
        &self,
        observer: &mut impl PartialGridObserver,
        beliefs: &mut GridBeliefMap,
        observed_at: MonotonicInstant,
    ) -> Result<(GridAction, ObservationBatchReport), PartialObsError> {
        let obs = observer.observe_occluded(observed_at);
        let agent_pos = obs.agent_pos();
        let report = beliefs.apply_occluded_observation(&obs)?;
        let action = self.choose_action(beliefs, agent_pos)?;
        Ok((action, report))
    }

    /// Elige una [`GridAction`] determinista evaluando únicamente [`GridBeliefMap`].
    pub fn choose_action(
        &self,
        beliefs: &GridBeliefMap,
        agent_pos: GridPos,
    ) -> Result<GridAction, PartialObsError> {
        let candidates = [
            GridAction::Up,
            GridAction::Right,
            GridAction::Down,
            GridAction::Left,
            GridAction::Stay,
        ];
        let size = beliefs.size();
        let mut best_action = GridAction::Stay;
        let mut best_score = f64::NEG_INFINITY;

        for action in candidates {
            let (dx, dy) = action.delta();
            let nx = agent_pos.x as isize + dx;
            let ny = agent_pos.y as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let target = GridPos::new(nx as usize, ny as usize);
            if target.x >= size.width || target.y >= size.height {
                continue;
            }

            let p_obs = beliefs.probability(target, HypothesisKind::Obstacle)?;
            let p_haz = beliefs.probability(target, HypothesisKind::Hazard)?;
            let p_goal = beliefs.probability(target, HypothesisKind::Goal)?;
            let direct_seen = beliefs.direct_update_count(target, HypothesisKind::Obstacle)?;
            let novelty = if direct_seen == 0 {
                self.exploration_bonus
            } else {
                0.0
            };

            let score = self.goal_bonus * p_goal + novelty
                - self.obstacle_penalty * p_obs
                - self.hazard_penalty * p_haz;

            if score > best_score {
                best_score = score;
                best_action = action;
            }
        }

        Ok(best_action)
    }
}

/// Clasificación de un informe humano generado por [`FallibleHumanFixture`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HumanReportCase {
    /// El humano reportó presencia y el hecho era verdadero.
    TruePositive,
    /// El humano reportó ausencia y el hecho era falso.
    TrueNegative,
    /// El humano reportó presencia donde el hecho era falso.
    FalsePositive,
    /// El humano reportó ausencia donde el hecho era verdadero.
    FalseNegative,
    /// El humano envió un informe con marca temporal anterior al watermark actual.
    Stale,
}

/// Informe humano generado junto con su etiqueta de caso de prueba.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedHumanReport {
    /// Envelope de informe humano (`ObservationKind::Assertion`).
    pub report: HumanReport,
    /// Caso epistémico que representa este informe.
    pub case: HumanReportCase,
}

/// Suite seeded reproducible que garantiza cobertura de falso positivo, falso negativo y dato antiguo.
#[derive(Debug, Clone, PartialEq)]
pub struct SeededHumanReportSuite {
    /// Informe de falso positivo (reporta hipótesis presente donde es falsa en el ground truth).
    pub false_positive: GeneratedHumanReport,
    /// Informe de falso negativo (reporta hipótesis ausente donde es verdadera en el ground truth).
    pub false_negative: GeneratedHumanReport,
    /// Informe con timestamp anterior al watermark (`observed_at < base_time`).
    pub stale: GeneratedHumanReport,
}

/// Configuración de la fuente humana falible de fixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FallibleHumanConfig {
    /// Identificador de sensor/fuente del informante humano.
    pub sensor_id: SensorId,
    /// Sensibilidad simulada `P(reporta presente | presente real)` en `[0.0, 1.0]`.
    pub sensitivity: f64,
    /// Especificidad simulada `P(reporta ausente | ausente real)` en `[0.0, 1.0]`.
    pub specificity: f64,
    /// Retraso monotónico aplicado a informes antiguos (`Stale`) en nanosegundos.
    pub stale_lag_nanos: u128,
}

impl Default for FallibleHumanConfig {
    fn default() -> Self {
        Self {
            sensor_id: SensorId::new(DEFAULT_HUMAN_SENSOR_ID),
            sensitivity: 0.80,
            specificity: 0.80,
            stale_lag_nanos: 500_000_000,
        }
    }
}

/// Fuente humana falible determinista (seeded con [`MasterSeed`] / [`DeterministicRng`]).
///
/// Genera [`HumanReport`] reproducibles con casos de falso positivo, falso negativo
/// y dato antiguo (`Stale`) para validar la actualización bayesiana y corrección por
/// observación directa.
#[derive(Debug, Clone)]
pub struct FallibleHumanFixture {
    rng: DeterministicRng,
    config: FallibleHumanConfig,
    next_message_id: u64,
    next_evidence_ref: u64,
}

impl FallibleHumanFixture {
    /// Crea un fixture humano falible determinista a partir de `seed` y `config`.
    pub fn new(seed: MasterSeed, config: FallibleHumanConfig) -> Result<Self, PartialObsError> {
        let _ =
            LikelihoodModel::from_sensitivity_specificity(config.sensitivity, config.specificity)?;
        let rng = seed.derive(StreamKey::Agent {
            episode: 0,
            agent: config.sensor_id.as_u64(),
        });
        Ok(Self {
            rng,
            config,
            next_message_id: 1,
            next_evidence_ref: 100_000,
        })
    }

    /// Devuelve la configuración activa del fixture humano.
    pub const fn config(&self) -> &FallibleHumanConfig {
        &self.config
    }

    fn build_report(
        &mut self,
        pos: GridPos,
        hypothesis: HypothesisKind,
        present: bool,
        observed_at: MonotonicInstant,
        received_at: MonotonicInstant,
        case: HumanReportCase,
    ) -> GeneratedHumanReport {
        let message_id = MessageId::new(self.next_message_id);
        let evidence_ref = EvidenceRef::new(self.next_evidence_ref);
        self.next_message_id += 1;
        self.next_evidence_ref += 1;

        // Confianza subjetiva reportada por el humano (0.55..0.95), independiente de la
        // calibración real de `LikelihoodModel`.
        let conf_scalar = 0.55 + 0.40 * self.rng.next_f64();
        let confidence =
            Confidence::new(conf_scalar.clamp(0.0, 1.0)).expect("clamped confidence must be valid");

        let meta = ProvenanceMetadata {
            message_id,
            observed_at,
            received_at,
            sensor_id: self.config.sensor_id,
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence,
            revision: message_id.as_u64(),
            evidence_ref,
            claim_kind: ObservationKind::Assertion,
        };
        let report = ObservationEnvelope::new(
            meta,
            CellHypothesis {
                pos,
                hypothesis,
                present,
            },
        );
        GeneratedHumanReport { report, case }
    }

    /// Genera un informe estocástico seeded sobre `(pos, hypothesis)` según `sensitivity`/`specificity`.
    pub fn sample_report(
        &mut self,
        ground_truth: &GroundTruthGrid,
        pos: GridPos,
        hypothesis: HypothesisKind,
        observed_at: MonotonicInstant,
    ) -> Result<GeneratedHumanReport, PartialObsError> {
        let cell = ground_truth
            .truth_cell(pos)
            .ok_or(PartialObsError::OutOfBounds {
                pos,
                size: ground_truth.grid_size(),
            })?;
        let actual_present = hypothesis.matches_cell(cell);
        let draw = self.rng.next_f64();

        let (reported_present, case) = if actual_present {
            if draw < self.config.sensitivity {
                (true, HumanReportCase::TruePositive)
            } else {
                (false, HumanReportCase::FalseNegative)
            }
        } else if draw < self.config.specificity {
            (false, HumanReportCase::TrueNegative)
        } else {
            (true, HumanReportCase::FalsePositive)
        };

        Ok(self.build_report(
            pos,
            hypothesis,
            reported_present,
            observed_at,
            observed_at,
            case,
        ))
    }

    /// Emite explícitamente un caso de **falso positivo** verificando que `hypothesis` es falsa en `pos`.
    pub fn emit_false_positive(
        &mut self,
        ground_truth: &GroundTruthGrid,
        pos: GridPos,
        hypothesis: HypothesisKind,
        observed_at: MonotonicInstant,
    ) -> Result<GeneratedHumanReport, PartialObsError> {
        let cell = ground_truth
            .truth_cell(pos)
            .ok_or(PartialObsError::OutOfBounds {
                pos,
                size: ground_truth.grid_size(),
            })?;
        if hypothesis.matches_cell(cell) {
            return Err(PartialObsError::InvalidFixturePrecondition {
                reason: "false positive requires hypothesis to be false in ground truth",
            });
        }
        Ok(self.build_report(
            pos,
            hypothesis,
            true,
            observed_at,
            observed_at,
            HumanReportCase::FalsePositive,
        ))
    }

    /// Emite explícitamente un caso de **falso negativo** verificando que `hypothesis` es verdadera en `pos`.
    pub fn emit_false_negative(
        &mut self,
        ground_truth: &GroundTruthGrid,
        pos: GridPos,
        hypothesis: HypothesisKind,
        observed_at: MonotonicInstant,
    ) -> Result<GeneratedHumanReport, PartialObsError> {
        let cell = ground_truth
            .truth_cell(pos)
            .ok_or(PartialObsError::OutOfBounds {
                pos,
                size: ground_truth.grid_size(),
            })?;
        if !hypothesis.matches_cell(cell) {
            return Err(PartialObsError::InvalidFixturePrecondition {
                reason: "false negative requires hypothesis to be true in ground truth",
            });
        }
        Ok(self.build_report(
            pos,
            hypothesis,
            false,
            observed_at,
            observed_at,
            HumanReportCase::FalseNegative,
        ))
    }

    /// Emite un informe con dato antiguo (`observed_at` retrasado respecto a `received_at`).
    pub fn emit_stale_report(
        &mut self,
        pos: GridPos,
        hypothesis: HypothesisKind,
        present: bool,
        stale_observed_at: MonotonicInstant,
        received_at: MonotonicInstant,
    ) -> GeneratedHumanReport {
        self.build_report(
            pos,
            hypothesis,
            present,
            stale_observed_at,
            received_at,
            HumanReportCase::Stale,
        )
    }

    /// Genera una suite seeded reproducible con un caso de falso positivo, uno de falso negativo
    /// y uno de dato antiguo (`observed_at < base_time`) seleccionados deterministamente del mapa.
    pub fn generate_seeded_suite(
        &mut self,
        ground_truth: &GroundTruthGrid,
        base_time: MonotonicInstant,
    ) -> Result<SeededHumanReportSuite, PartialObsError> {
        let map = ground_truth.map();
        let mut empty_cells = Vec::new();
        let mut wall_cells = Vec::new();
        let mut goal_cells = Vec::new();

        for y in 0..map.height() {
            for x in 0..map.width() {
                let pos = GridPos::new(x, y);
                match map.cell(pos) {
                    Some(CellKind::Empty) => empty_cells.push(pos),
                    Some(CellKind::Wall) => wall_cells.push(pos),
                    Some(CellKind::Goal) => goal_cells.push(pos),
                    Some(CellKind::Hazard) | None => {}
                }
            }
        }

        if empty_cells.is_empty() {
            return Err(PartialObsError::InvalidFixturePrecondition {
                reason: "grid map needs at least one Empty cell for false positive case",
            });
        }
        let fp_idx = (self.rng.next_u64() as usize) % empty_cells.len();
        let fp_pos = empty_cells[fp_idx];
        let false_positive =
            self.emit_false_positive(ground_truth, fp_pos, HypothesisKind::Obstacle, base_time)?;

        let (fn_pos, fn_hyp) = if !wall_cells.is_empty() {
            let fn_idx = (self.rng.next_u64() as usize) % wall_cells.len();
            (wall_cells[fn_idx], HypothesisKind::Obstacle)
        } else if !goal_cells.is_empty() {
            let fn_idx = (self.rng.next_u64() as usize) % goal_cells.len();
            (goal_cells[fn_idx], HypothesisKind::Goal)
        } else {
            return Err(PartialObsError::InvalidFixturePrecondition {
                reason: "grid map needs at least one Wall or Goal cell for false negative case",
            });
        };
        let fn_time = MonotonicInstant::from_nanos(base_time.as_nanos().saturating_add(1_000));
        let false_negative = self.emit_false_negative(ground_truth, fn_pos, fn_hyp, fn_time)?;

        if base_time.as_nanos() == 0 {
            return Err(PartialObsError::InvalidFixturePrecondition {
                reason: "base_time must be strictly greater than 0 to construct a stale timestamp",
            });
        }
        let lag = self.config.stale_lag_nanos.min(base_time.as_nanos());
        let lag = if lag == 0 { 1 } else { lag };
        let stale_observed_at = MonotonicInstant::from_nanos(base_time.as_nanos() - lag);
        let stale_received_at =
            MonotonicInstant::from_nanos(base_time.as_nanos().saturating_add(2_000));
        let stale = self.emit_stale_report(
            fp_pos,
            HypothesisKind::Obstacle,
            true,
            stale_observed_at,
            stale_received_at,
        );

        Ok(SeededHumanReportSuite {
            false_positive,
            false_negative,
            stale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::cul_de_sac_map;

    #[test]
    fn ac1_occlusion_hides_cells_behind_walls_and_planner_is_invariant_to_hidden_ground_truth() {
        // Dos mundos idénticos en la región visible desde S=(1,1) con radio=3,
        // pero completamente distintos detrás del muro en x=2 y más allá del radio.
        let map_a = GridMap::from_ascii(&["#######", "#S.#.G#", "#..#..#", "#######"]).unwrap();
        let map_b = GridMap::from_ascii(&["#######", "#S.#HG#", "#..#H.#", "#######"]).unwrap();

        let mut gt_a = GroundTruthGrid::new(map_a, 3);
        let mut gt_b = GroundTruthGrid::new(map_b, 3);
        let t = MonotonicInstant::from_nanos(1_000);

        let obs_a = gt_a.observer_view().observe_occluded(t);
        let obs_b = gt_b.observer_view().observe_occluded(t);

        // El muro en (3,1) es visible, pero (4,1) y (5,1) detrás del muro están ocluidas.
        assert!(obs_a.contains_pos(GridPos::new(1, 1)));
        assert!(obs_a.contains_pos(GridPos::new(2, 1)));
        assert!(obs_a.contains_pos(GridPos::new(3, 1)));
        assert_eq!(obs_a.cell_at(GridPos::new(3, 1)), Some(CellKind::Wall));
        assert!(!obs_a.contains_pos(GridPos::new(4, 1)));
        assert!(!obs_a.contains_pos(GridPos::new(5, 1)));
        assert!(!obs_a.contains_pos(GridPos::new(4, 2)));

        // Las observaciones parciales de ambos mundos son idénticas porque solo difieren en celdas ocluidas.
        assert_eq!(obs_a, obs_b);

        // El planificador basado en creencias toma exactamente la misma decisión sin leer ground truth.
        let planner = BeliefGridPlanner::default();
        let mut beliefs_a =
            GridBeliefMap::new(gt_a.grid_size(), GridBeliefConfig::default()).unwrap();
        let mut beliefs_b =
            GridBeliefMap::new(gt_b.grid_size(), GridBeliefConfig::default()).unwrap();

        let t2 = MonotonicInstant::from_nanos(2_000);
        let (act_a, _) = planner
            .observe_and_choose_action(&mut gt_a.observer_view(), &mut beliefs_a, t2)
            .unwrap();
        let (act_b, _) = planner
            .observe_and_choose_action(&mut gt_b.observer_view(), &mut beliefs_b, t2)
            .unwrap();

        assert_eq!(act_a, act_b);
        // Las celdas ocluidas (4,1) conservan exactamente el prior sin filtración de ground truth.
        assert!(
            (beliefs_a
                .probability(GridPos::new(4, 1), HypothesisKind::Hazard)
                .unwrap()
                - beliefs_a.config().prior_hazard)
                .abs()
                < 1e-12
        );
        assert!(
            (beliefs_b
                .probability(GridPos::new(4, 1), HypothesisKind::Hazard)
                .unwrap()
                - beliefs_b.config().prior_hazard)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn ac1_diagonal_closed_corner_blocks_line_of_sight() {
        // S en (1,1), muros en (2,1) y (1,2) formando rincón cerrado -> (2,2) no debe ser visible.
        let map = GridMap::from_ascii(&["#####", "#S#G#", "##..#", "#####"]).unwrap();
        let mut gt = GroundTruthGrid::new(map, 2);
        let obs = gt.observe_occluded(MonotonicInstant::from_nanos(500));

        assert!(obs.contains_pos(GridPos::new(2, 1)));
        assert!(obs.contains_pos(GridPos::new(1, 2)));
        assert!(
            !obs.contains_pos(GridPos::new(2, 2)),
            "diagonal closed corner must occlude (2,2)"
        );
    }

    #[test]
    fn ac2_human_report_updates_hypothesis_and_direct_contradiction_corrects_it() {
        let map = GridMap::from_ascii(&["######", "#S..G#", "######"]).unwrap();
        let mut gt = GroundTruthGrid::new(map, 1);
        let mut beliefs = GridBeliefMap::new(gt.grid_size(), GridBeliefConfig::default()).unwrap();
        let mut human =
            FallibleHumanFixture::new(MasterSeed::new(42), FallibleHumanConfig::default()).unwrap();

        // Celda (3,1) está fuera del radio=1 inicial desde S=(1,1) y realmente es Empty.
        let target = GridPos::new(3, 1);
        let prior_p = beliefs
            .probability(target, HypothesisKind::Obstacle)
            .unwrap();
        assert!((prior_p - 0.25).abs() < 1e-12);

        // 1) Informe humano falso positivo afirma que (3,1) tiene un obstáculo.
        let fp = human
            .emit_false_positive(
                &gt,
                target,
                HypothesisKind::Obstacle,
                MonotonicInstant::from_nanos(1_000),
            )
            .unwrap();
        assert_eq!(fp.case, HumanReportCase::FalsePositive);
        assert_eq!(fp.report.claim_kind(), ObservationKind::Assertion);

        let status = beliefs.apply_human_report(&fp.report).unwrap();
        assert_eq!(status, BeliefUpdateStatus::Updated);
        let after_human = beliefs
            .probability(target, HypothesisKind::Obstacle)
            .unwrap();
        assert!(
            after_human > 0.5,
            "human assertion must push P(Obstacle) above 0.5, got {after_human}"
        );

        // Reenvío del mismo informe humano es ignorado como duplicado sin doble conteo.
        let dup_status = beliefs.apply_human_report(&fp.report).unwrap();
        assert_eq!(dup_status, BeliefUpdateStatus::DuplicateEvidenceIgnored);
        assert!(
            (beliefs
                .probability(target, HypothesisKind::Obstacle)
                .unwrap()
                - after_human)
                .abs()
                < 1e-12
        );

        // 2) El agente avanza hasta (2,1), con lo que (3,1) entra en su radio de visión directa.
        gt.step_agent(GridAction::Right);
        assert_eq!(gt.agent_pos(), GridPos::new(2, 1));
        let obs = gt.observe_occluded(MonotonicInstant::from_nanos(2_000));
        assert_eq!(obs.cell_at(target), Some(CellKind::Empty));

        let batch = beliefs.apply_occluded_observation(&obs).unwrap();
        let after_direct = beliefs
            .probability(target, HypothesisKind::Obstacle)
            .unwrap();
        assert!(
            after_direct < 0.05,
            "direct observation must correct false positive below 0.05, got {after_direct}"
        );
        assert!(
            batch.corrections.iter().any(|c| c.pos == target
                && c.hypothesis == HypothesisKind::Obstacle
                && !c.observed_present),
            "contradiction correction must be recorded for {target:?}"
        );
    }

    #[test]
    fn ac2_rejects_unsupported_or_mismatched_observation_kinds() {
        let map = GridMap::from_ascii(&["#####", "#S.G#", "#####"]).unwrap();
        let mut beliefs = GridBeliefMap::new(map.size(), GridBeliefConfig::default()).unwrap();

        let proposal = ObservationEnvelope::new(
            ProvenanceMetadata {
                message_id: MessageId::new(1),
                observed_at: MonotonicInstant::from_nanos(1_000),
                received_at: MonotonicInstant::from_nanos(1_000),
                sensor_id: SensorId::new(100),
                frame_id: FrameId::world(),
                units: UnitTag::dimensionless(),
                confidence: Confidence::FULL,
                revision: 1,
                evidence_ref: EvidenceRef::new(999),
                claim_kind: ObservationKind::Proposal,
            },
            CellHypothesis {
                pos: GridPos::new(2, 1),
                hypothesis: HypothesisKind::Obstacle,
                present: true,
            },
        );
        let status = beliefs.apply_human_report(&proposal).unwrap();
        assert_eq!(status, BeliefUpdateStatus::AbstainedUnsupportedKind);
        assert!(
            (beliefs
                .probability(GridPos::new(2, 1), HypothesisKind::Obstacle)
                .unwrap()
                - 0.25)
                .abs()
                < 1e-12
        );
    }

    #[test]
    fn ac3_seeded_fixture_covers_false_positive_false_negative_and_stale_reproducibly() {
        let map = cul_de_sac_map();
        let gt_a = GroundTruthGrid::new(map.clone(), 2);
        let gt_b = GroundTruthGrid::new(map.clone(), 2);
        let base_time = MonotonicInstant::from_nanos(10_000_000_000);

        let mut fx_a =
            FallibleHumanFixture::new(MasterSeed::new(2026), FallibleHumanConfig::default())
                .unwrap();
        let mut fx_b =
            FallibleHumanFixture::new(MasterSeed::new(2026), FallibleHumanConfig::default())
                .unwrap();
        let mut fx_other =
            FallibleHumanFixture::new(MasterSeed::new(9999), FallibleHumanConfig::default())
                .unwrap();

        let suite_a = fx_a.generate_seeded_suite(&gt_a, base_time).unwrap();
        let suite_b = fx_b.generate_seeded_suite(&gt_b, base_time).unwrap();
        let suite_other = fx_other.generate_seeded_suite(&gt_a, base_time).unwrap();

        // Reproducibilidad exacta para la misma semilla.
        assert_eq!(suite_a, suite_b);
        assert_ne!(suite_a, suite_other);

        assert_eq!(suite_a.false_positive.case, HumanReportCase::FalsePositive);
        assert_eq!(suite_a.false_negative.case, HumanReportCase::FalseNegative);
        assert_eq!(suite_a.stale.case, HumanReportCase::Stale);
        assert!(suite_a.stale.report.observed_at() < base_time);

        let mut beliefs = GridBeliefMap::new(map.size(), GridBeliefConfig::default()).unwrap();

        // --- Caso 1: Falso positivo ---
        let fp_payload = *suite_a.false_positive.report.payload();
        assert!(fp_payload.present);
        assert_eq!(
            beliefs
                .apply_human_report(&suite_a.false_positive.report)
                .unwrap(),
            BeliefUpdateStatus::Updated
        );
        let p_after_fp = beliefs
            .probability(fp_payload.pos, fp_payload.hypothesis)
            .unwrap();
        assert!(p_after_fp > 0.5);

        // Observación directa en la celda del falso positivo lo corrige.
        let mut gt_fp = GroundTruthGrid::new(map.clone(), 1);
        gt_fp.set_agent_pos(fp_payload.pos).unwrap();
        let obs_fp =
            gt_fp.observe_occluded(MonotonicInstant::from_nanos(base_time.as_nanos() + 5_000));
        let batch_fp = beliefs.apply_occluded_observation(&obs_fp).unwrap();
        let p_corrected_fp = beliefs
            .probability(fp_payload.pos, fp_payload.hypothesis)
            .unwrap();
        assert!(p_corrected_fp < 0.05);
        assert!(
            batch_fp
                .corrections
                .iter()
                .any(|c| c.pos == fp_payload.pos && c.hypothesis == fp_payload.hypothesis)
        );

        // --- Caso 2: Falso negativo ---
        let fn_payload = *suite_a.false_negative.report.payload();
        assert!(!fn_payload.present);
        assert_eq!(
            beliefs
                .apply_human_report(&suite_a.false_negative.report)
                .unwrap(),
            BeliefUpdateStatus::Updated
        );
        let p_after_fn = beliefs
            .probability(fn_payload.pos, fn_payload.hypothesis)
            .unwrap();
        assert!(
            p_after_fn < 0.15,
            "false negative must lower P(true) below 0.15, got {p_after_fn}"
        );

        // Observación directa sobre la celda real (Wall/Goal) corrige el falso negativo.
        let direct_fn = ObservationEnvelope::new(
            ProvenanceMetadata {
                message_id: MessageId::new(777),
                observed_at: MonotonicInstant::from_nanos(base_time.as_nanos() + 10_000),
                received_at: MonotonicInstant::from_nanos(base_time.as_nanos() + 10_000),
                sensor_id: SensorId::new(DEFAULT_DIRECT_SENSOR_ID),
                frame_id: FrameId::world(),
                units: UnitTag::dimensionless(),
                confidence: Confidence::FULL,
                revision: 777,
                evidence_ref: EvidenceRef::new(777_000),
                claim_kind: ObservationKind::Measurement,
            },
            VisibleCell {
                pos: fn_payload.pos,
                kind: map.cell(fn_payload.pos).unwrap(),
            },
        );
        let outcome_fn = beliefs.apply_direct_observation(&direct_fn).unwrap();
        let p_corrected_fn = beliefs
            .probability(fn_payload.pos, fn_payload.hypothesis)
            .unwrap();
        assert!(
            p_corrected_fn > 0.85,
            "direct observation must correct false negative above 0.85, got {p_corrected_fn}"
        );
        assert!(
            outcome_fn
                .corrections
                .iter()
                .any(|c| c.pos == fn_payload.pos
                    && c.hypothesis == fn_payload.hypothesis
                    && c.observed_present)
        );

        // --- Caso 3: Dato antiguo (Stale) ---
        // La celda fp_payload.pos ya tiene watermark `base_time + 5_000` tras la observación directa.
        let watermark_before = beliefs
            .watermark(fp_payload.pos, fp_payload.hypothesis)
            .unwrap();
        let belief_before = beliefs
            .probability(fp_payload.pos, fp_payload.hypothesis)
            .unwrap();

        let stale_status = beliefs.apply_human_report(&suite_a.stale.report).unwrap();
        assert_eq!(stale_status, BeliefUpdateStatus::AbstainedStaleObservation);
        assert!(stale_status.is_abstained());
        assert_eq!(
            beliefs
                .watermark(fp_payload.pos, fp_payload.hypothesis)
                .unwrap(),
            watermark_before
        );
        assert!(
            (beliefs
                .probability(fp_payload.pos, fp_payload.hypothesis)
                .unwrap()
                - belief_before)
                .abs()
                < 1e-12,
            "stale report must not degrade the newer belief"
        );

        // Verificación adicional directa con `DiscreteBeliefFilter` de `gogamen-core`.
        let mut filter = beliefs
            .to_discrete_filter(
                fp_payload.pos,
                fp_payload.hypothesis,
                beliefs.config().human_model.clone(),
            )
            .unwrap();
        let fresh_env = suite_a.false_positive.report.clone().map(|h| h.present);
        let stale_env = suite_a.stale.report.clone().map(|h| h.present);
        assert_eq!(
            filter.update_with_binary_observation(&fresh_env).unwrap(),
            BeliefUpdateStatus::Updated
        );
        let p_fresh = filter.belief().get(1).unwrap();
        assert_eq!(
            filter.update_with_binary_observation(&stale_env).unwrap(),
            BeliefUpdateStatus::AbstainedStaleObservation
        );
        assert!((filter.belief().get(1).unwrap() - p_fresh).abs() < 1e-12);
    }

    #[test]
    fn ac3_stochastic_human_fixture_samples_reproducibly_by_seed() {
        let map = cul_de_sac_map();
        let gt = GroundTruthGrid::new(map, 2);
        let mut fx1 =
            FallibleHumanFixture::new(MasterSeed::new(7), FallibleHumanConfig::default()).unwrap();
        let mut fx2 =
            FallibleHumanFixture::new(MasterSeed::new(7), FallibleHumanConfig::default()).unwrap();

        let mut cases1 = Vec::new();
        let mut cases2 = Vec::new();
        for i in 0..40u64 {
            let t = MonotonicInstant::from_nanos((i as u128 + 1) * 1_000);
            let pos = if i % 2 == 0 {
                GridPos::new(1, 1) // Empty
            } else {
                GridPos::new(0, 0) // Wall
            };
            let r1 = fx1
                .sample_report(&gt, pos, HypothesisKind::Obstacle, t)
                .unwrap();
            let r2 = fx2
                .sample_report(&gt, pos, HypothesisKind::Obstacle, t)
                .unwrap();
            assert_eq!(r1, r2);
            cases1.push(r1.case);
            cases2.push(r2.case);
        }

        assert_eq!(cases1, cases2);
        // Con sensitivity=0.80, specificity=0.80 y 40 muestras, deben observarse tanto aciertos como errores.
        assert!(cases1.contains(&HumanReportCase::TruePositive));
        assert!(cases1.contains(&HumanReportCase::TrueNegative));
        assert!(
            cases1.contains(&HumanReportCase::FalsePositive)
                || cases1.contains(&HumanReportCase::FalseNegative)
        );
    }

    // ---- GOG-131: deduplicador centralizado y almacenamiento plano ----

    fn assert_copy<T: Copy>() {}

    fn meta(
        sensor: u64,
        message: u64,
        evidence: u64,
        observed: u128,
        kind: ObservationKind,
    ) -> ProvenanceMetadata {
        ProvenanceMetadata {
            message_id: MessageId::new(message),
            observed_at: MonotonicInstant::from_nanos(observed),
            received_at: MonotonicInstant::from_nanos(observed + 1),
            sensor_id: SensorId::new(sensor),
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence: Confidence::new(0.9).unwrap(),
            revision: message,
            evidence_ref: EvidenceRef::new(evidence),
            claim_kind: kind,
        }
    }

    fn human(
        sensor: u64,
        msg: u64,
        ev: u64,
        at: u128,
        pos: GridPos,
        hypothesis: HypothesisKind,
        present: bool,
    ) -> HumanReport {
        ObservationEnvelope::new(
            meta(sensor, msg, ev, at, ObservationKind::Assertion),
            CellHypothesis {
                pos,
                hypothesis,
                present,
            },
        )
    }

    fn direct(
        sensor: u64,
        msg: u64,
        ev: u64,
        at: u128,
        pos: GridPos,
        kind: CellKind,
    ) -> DirectObservation {
        ObservationEnvelope::new(
            meta(sensor, msg, ev, at, ObservationKind::Measurement),
            VisibleCell { pos, kind },
        )
    }

    #[test]
    fn gog131_ac1_single_central_deduplicator_and_flat_copy_state() {
        assert_copy::<CellHypothesisState>();
        let size = GridSize {
            width: 5,
            height: 4,
        };
        let mut beliefs = GridBeliefMap::new(size, GridBeliefConfig::default()).unwrap();
        // Un único vector plano con 3 entradas por celda; ledger único y vacío al inicio.
        assert_eq!(beliefs.scalar_state_len(), 3 * 5 * 4);
        assert_eq!(beliefs.ledger_sizes(), (0, 0));

        // Observar muchas celdas no crea deduplicadores por celda: solo crecen los dos sets
        // del ledger único (una clave por (celda, hipótesis) aceptada).
        let pos = GridPos::new(2, 1);
        let obs = direct(1, 1, 1, 10, pos, CellKind::Wall);
        beliefs.apply_direct_observation(&obs).unwrap();
        assert_eq!(beliefs.ledger_sizes(), (3, 3));
        let rep = human(100, 7, 7, 20, pos, HypothesisKind::Obstacle, true);
        beliefs.apply_human_report(&rep).unwrap();
        assert_eq!(beliefs.ledger_sizes(), (4, 4));
        assert_eq!(beliefs.scalar_state_len(), 60);
    }

    #[test]
    fn gog131_ac2_clone_copies_flat_state_and_is_independent() {
        let size = GridSize {
            width: 4,
            height: 4,
        };
        let mut beliefs = GridBeliefMap::new(size, GridBeliefConfig::default()).unwrap();
        let pos = GridPos::new(1, 2);
        beliefs
            .apply_direct_observation(&direct(1, 1, 1, 10, pos, CellKind::Hazard))
            .unwrap();
        let mut clone = beliefs.clone();
        assert_eq!(clone.entries, beliefs.entries);
        assert_eq!(clone.ledger_sizes(), beliefs.ledger_sizes());
        assert_eq!(clone.scalar_state_len(), beliefs.scalar_state_len());

        // Mutar el clon no afecta al original, y la evidencia ya vista se conserva en el clon.
        let dup = clone
            .apply_direct_observation(&direct(1, 1, 1, 10, pos, CellKind::Hazard))
            .unwrap();
        assert!(
            dup.statuses
                .iter()
                .all(|st| *st == BeliefUpdateStatus::DuplicateEvidenceIgnored)
        );
        let p_before = beliefs.probability(pos, HypothesisKind::Hazard).unwrap();
        clone
            .apply_direct_observation(&direct(1, 2, 2, 20, pos, CellKind::Hazard))
            .unwrap();
        assert_eq!(
            beliefs.probability(pos, HypothesisKind::Hazard).unwrap(),
            p_before
        );
        assert_ne!(
            clone.probability(pos, HypothesisKind::Hazard).unwrap(),
            p_before
        );
        assert_eq!(beliefs.ledger_sizes(), (3, 3));
        assert_eq!(clone.ledger_sizes(), (6, 6));
        // El original sigue aceptando la evidencia nueva (no contaminada por el clon).
        let st = beliefs
            .apply_direct_observation(&direct(1, 2, 2, 20, pos, CellKind::Hazard))
            .unwrap();
        assert!(st.statuses.iter().all(|s| s.is_updated()));
    }

    /// Referencia de test del comportamiento previo: un deduplicador por (celda, hipótesis).
    struct PerCellReference {
        width: usize,
        cells: Vec<(BinaryBelief, EvidenceDeduplicator, Option<MonotonicInstant>)>,
        config: GridBeliefConfig,
    }

    impl PerCellReference {
        fn new(size: GridSize, config: GridBeliefConfig) -> Self {
            let mut cells = Vec::new();
            for _ in 0..size.cell_count() {
                for h in HypothesisKind::ALL {
                    cells.push((
                        config.prior_for(h).unwrap(),
                        EvidenceDeduplicator::new(),
                        None,
                    ));
                }
            }
            Self {
                width: size.width,
                cells,
                config,
            }
        }

        fn idx(&self, pos: GridPos, h: HypothesisKind) -> usize {
            (pos.y * self.width + pos.x) * 3 + h.index()
        }

        fn human(&mut self, r: &HumanReport) -> BeliefUpdateStatus {
            let p = *r.payload();
            let i = self.idx(p.pos, p.hypothesis);
            let env = r.clone().map(|h| h.present);
            let (b, d, w) = &mut self.cells[i];
            b.update_with_observation(&env, &self.config.human_model, d, w)
                .unwrap()
        }

        fn direct(&mut self, o: &DirectObservation) -> [BeliefUpdateStatus; 3] {
            let p = *o.payload();
            let mut out = [BeliefUpdateStatus::AbstainedUnsupportedKind; 3];
            for h in HypothesisKind::ALL {
                let i = self.idx(p.pos, h);
                let present = h.matches_cell(p.kind);
                let env = o.clone().map(|_| present);
                let (b, d, w) = &mut self.cells[i];
                out[h.index()] = b
                    .update_with_observation(&env, &self.config.direct_model, d, w)
                    .unwrap();
            }
            out
        }
    }

    #[test]
    fn gog131_ac3_equivalence_with_per_cell_reference_on_seeded_sequence() {
        let size = GridSize {
            width: 4,
            height: 3,
        };
        let cfg = GridBeliefConfig::default();
        let mut new_map = GridBeliefMap::new(size, cfg.clone()).unwrap();
        let mut reference = PerCellReference::new(size, cfg);
        let mut rng = DeterministicRng::new(0x600D_5EED);
        let kinds = [
            CellKind::Empty,
            CellKind::Wall,
            CellKind::Hazard,
            CellKind::Goal,
        ];
        let mut now: u128 = 1_000;
        let mut duplicates = 0usize;
        let mut stale = 0usize;
        let mut updated = 0usize;
        let mut tally = |sts: &[BeliefUpdateStatus]| {
            for st in sts {
                match st {
                    BeliefUpdateStatus::DuplicateEvidenceIgnored => duplicates += 1,
                    BeliefUpdateStatus::AbstainedStaleObservation => stale += 1,
                    BeliefUpdateStatus::Updated => updated += 1,
                    _ => {}
                }
            }
        };

        for _ in 0..4_000 {
            let pos = GridPos::new((rng.next_u64() % 4) as usize, (rng.next_u64() % 3) as usize);
            // Ids en rangos pequeños para provocar colisiones de evidencia y de mensaje.
            let sensor = 100 + rng.next_u64() % 2;
            let msg = rng.next_u64() % 12;
            let ev = rng.next_u64() % 12;
            let jitter = rng.next_u64() % 3;
            now += 10;
            let at = if jitter == 0 { now - 200 } else { now };
            if rng.next_u64().is_multiple_of(2) {
                let h = HypothesisKind::ALL[(rng.next_u64() % 3) as usize];
                let present = rng.next_u64().is_multiple_of(2);
                let r = human(sensor, msg, ev, at, pos, h, present);
                let got = new_map.apply_human_report(&r).unwrap();
                let want = reference.human(&r);
                assert_eq!(got, want);
                tally(&[got]);
            } else {
                let kind = kinds[(rng.next_u64() % 4) as usize];
                let o = direct(sensor, msg, ev, at, pos, kind);
                let got = new_map.apply_direct_observation(&o).unwrap().statuses;
                let want = reference.direct(&o);
                assert_eq!(got, want);
                tally(&got);
            }
        }
        // La secuencia ejercita realmente los tres caminos.
        assert!(duplicates > 50 && stale > 50 && updated > 50);

        for y in 0..3 {
            for x in 0..4 {
                for h in HypothesisKind::ALL {
                    let pos = GridPos::new(x, y);
                    let i = reference.idx(pos, h);
                    let (rb, _, rw) = &reference.cells[i];
                    let nb = new_map.belief(pos, h).unwrap();
                    assert_eq!(
                        nb.p_true().to_bits(),
                        rb.p_true().to_bits(),
                        "p_true {pos:?} {h:?}"
                    );
                    assert_eq!(nb.p_false().to_bits(), rb.p_false().to_bits());
                    assert_eq!(new_map.watermark(pos, h).unwrap(), *rw);
                }
            }
        }
    }
}
