//! Suite de ablación reproducible offline de sensor semántico, creencias, FMC y
//! comunicación activa (`std`-only + `gogamen-core`).
//!
//! Evalúa bajo **semillas `holdout` idénticas** (separadas explícitamente de `dev`) y
//! **presupuestos idénticos** (`max_steps` y `PlanBudget` por paso) las 6 configuraciones:
//! 1. [`AblationVariant::PhysicalSensorOnly`]: solo sensor físico directo ocluido.
//! 2. [`AblationVariant::PerfectHint`]: sensor físico + avisos humanos exactos.
//! 3. [`AblationVariant::FallibleSource`]: sensor físico + fuente humana falible (`FallibleHumanFixture`).
//! 4. [`AblationVariant::NominalFmc`]: FMC nominal sobre mapa puntual derivado de creencias.
//! 5. [`AblationVariant::BeliefFmc`]: FMC sobre `K` partículas muestreadas del posterior de creencias.
//! 6. [`AblationVariant::ActiveCommunication`]: comunicación activa (`Ask` / `Inspect` / `Advance`)
//!    mediada por [`EpistemicController`] y [`LocalCommunicationBus`] (`CommunicationPort`) + creencias.
//!
//! Incluye métricas de calibración/fiabilidad (**Brier score**, **log-loss**, **ECE** y bins de
//! fiabilidad) sobre fixtures offline etiquetados con descargo explícito de que **no afirman
//! calibración Jev real**, además de explicación determinista de hallazgos (sin ocultar
//! degradaciones bajo ruido o fuente falible).

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::time::Duration as StdDuration;

use gogamen_core::{
    ActionGuard, ActionSpace, BeliefError, BinaryBelief, BinaryCommOutcomeMapper, CommCost,
    CommCostWeights, CommPayload, CommunicationError, CommunicationPort, Confidence,
    ConversationScope, DecisionOption, DecisionPayoffModel, DiscreteAction, DomainError,
    EpistemicActionError, EpistemicActionKind, EpistemicController, EvidenceRef, FmcPlannerConfig,
    FmcPlannerError, FmcScoringConfig, FrameId, GuardContext, GuardDecision, InterlocutorId,
    InterlocutorKind, LikelihoodModel, LocalCommunicationBus, MasterSeed, MessageId,
    MonotonicClock, MonotonicInstant, ObservationEnvelope, ObservationKind, OptionId, PlanBudget,
    ProvenanceMetadata, SafeFallbackPolicy, SensorId, SituatedMessage, StepDt, StreamKey,
    SystemOneError, Terminal, Tick, UnitTag, VirtualMonotonicClock,
};

use crate::belief_fmc::{
    BeliefFmcConfig, BeliefFmcError, BeliefFmcPlanner, belief_epistemic_uncertainty_bits,
};
use crate::experiment::SeedSplit;
use crate::map::{CellKind, GridMap, GridPos};
use crate::partial_obs::{
    BeliefGridPlanner, CellHypothesis, DEFAULT_HUMAN_SENSOR_ID, FallibleHumanConfig,
    FallibleHumanFixture, GridBeliefConfig, GridBeliefMap, GroundTruthGrid, HypothesisKind,
    PartialGridObserver, PartialObsError,
};
use crate::world::{GRID_ACTION_CARDINALITY, GridAction};

/// Descargo explícito sobre métricas de calibración en fixtures offline etiquetados.
pub const CALIBRATION_FIXTURE_DISCLAIMER: &str = "Métricas de Brier score, log-loss y fiabilidad/calibración calculadas exclusivamente sobre fixtures offline etiquetados en Grid; no constituyen ni afirman calibración Jev real en entornos abiertos.";

const LOG_LOSS_EPS: f64 = 1e-12;
const DEFAULT_CALIBRATION_BINS: usize = 5;

/// Errores explícitos de la suite de ablación.
#[derive(Debug, Clone, PartialEq)]
pub enum AblationError {
    /// Las semillas `dev` o `holdout` están vacías.
    EmptySeedSplit {
        /// Split que resultó vacío.
        split: SeedSplit,
    },
    /// Existe solapamiento entre semillas `dev` y `holdout`.
    OverlappingSeedSplits {
        /// Primera semilla repetida encontrada en ambos splits.
        seed: MasterSeed,
    },
    /// Parámetro numérico de configuración inválido.
    InvalidConfig {
        /// Campo inválido.
        field: &'static str,
        /// Motivo legible.
        reason: &'static str,
    },
    /// Error de dominio de `gogamen-core`.
    Domain(DomainError),
    /// Error de observabilidad parcial o mapa de creencias.
    PartialObs(PartialObsError),
    /// Error del planificador FMC sobre creencias.
    BeliefFmc(BeliefFmcError),
    /// Error de configuración del planificador FMC.
    FmcPlanner(FmcPlannerError),
    /// Error de acciones epistémicas.
    Epistemic(EpistemicActionError),
    /// Error de comunicación situada.
    Communication(CommunicationError),
    /// Error de actualización bayesiana.
    Belief(BeliefError),
    /// Error de identificadores de sistema.
    SystemOne(SystemOneError),
}

impl fmt::Display for AblationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeedSplit { split } => {
                write!(f, "seed split {split:?} cannot be empty")
            }
            Self::OverlappingSeedSplits { seed } => {
                write!(
                    f,
                    "dev and holdout seed splits must be disjoint (overlap at seed {})",
                    seed.as_u64()
                )
            }
            Self::InvalidConfig { field, reason } => {
                write!(f, "invalid ablation config '{field}': {reason}")
            }
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::PartialObs(err) => write!(f, "partial observation error: {err}"),
            Self::BeliefFmc(err) => write!(f, "belief fmc error: {err}"),
            Self::FmcPlanner(err) => write!(f, "fmc planner error: {err}"),
            Self::Epistemic(err) => write!(f, "epistemic action error: {err}"),
            Self::Communication(err) => write!(f, "communication error: {err}"),
            Self::Belief(err) => write!(f, "belief error: {err}"),
            Self::SystemOne(err) => write!(f, "system one error: {err}"),
        }
    }
}

impl Error for AblationError {}

impl From<DomainError> for AblationError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<PartialObsError> for AblationError {
    fn from(value: PartialObsError) -> Self {
        Self::PartialObs(value)
    }
}

impl From<BeliefFmcError> for AblationError {
    fn from(value: BeliefFmcError) -> Self {
        Self::BeliefFmc(value)
    }
}

impl From<FmcPlannerError> for AblationError {
    fn from(value: FmcPlannerError) -> Self {
        Self::FmcPlanner(value)
    }
}

impl From<EpistemicActionError> for AblationError {
    fn from(value: EpistemicActionError) -> Self {
        Self::Epistemic(value)
    }
}

impl From<CommunicationError> for AblationError {
    fn from(value: CommunicationError) -> Self {
        Self::Communication(value)
    }
}

impl From<BeliefError> for AblationError {
    fn from(value: BeliefError) -> Self {
        Self::Belief(value)
    }
}

impl From<SystemOneError> for AblationError {
    fn from(value: SystemOneError) -> Self {
        Self::SystemOne(value)
    }
}

/// Las 6 configuraciones evaluadas en la suite de ablación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AblationVariant {
    /// 1. Solo sensor físico directo ocluido (`BeliefGridPlanner`).
    PhysicalSensorOnly,
    /// 2. Sensor físico + avisos humanos exactos (`NominalFmc`).
    PerfectHint,
    /// 3. Sensor físico + fuente humana falible (`FallibleHumanFixture` + `NominalFmc`).
    FallibleSource,
    /// 4. FMC nominal sobre estimación puntual de creencias (sin avisos humanos).
    NominalFmc,
    /// 5. FMC sobre `K` muestras del posterior de creencias (sin avisos humanos).
    BeliefFmc,
    /// 6. Comunicación activa (`Ask`/`Inspect`/`Advance` sobre `CommunicationPort` + creencias + `BeliefFmc`).
    ActiveCommunication,
}

impl AblationVariant {
    /// Todas las configuraciones en orden canónico de ablación.
    pub const ALL: [Self; 6] = [
        Self::PhysicalSensorOnly,
        Self::PerfectHint,
        Self::FallibleSource,
        Self::NominalFmc,
        Self::BeliefFmc,
        Self::ActiveCommunication,
    ];

    /// Etiqueta estable de la configuración.
    pub const fn label(self) -> &'static str {
        match self {
            Self::PhysicalSensorOnly => "PhysicalSensorOnly",
            Self::PerfectHint => "PerfectHint",
            Self::FallibleSource => "FallibleSource",
            Self::NominalFmc => "NominalFmc",
            Self::BeliefFmc => "BeliefFmc",
            Self::ActiveCommunication => "ActiveCommunication",
        }
    }
}

/// Mapa por defecto de ablación (`7x5`) con peligro ocluido `(3,1)` en la ruta corta y desvío seguro inferior.
pub fn default_ablation_map() -> GridMap {
    GridMap::from_ascii(&["#######", "#S.H.G#", "#.#.#.#", "#.....#", "#######"])
        .expect("default ablation map must be valid")
}

/// Configuración compartida por las 6 variantes de la suite de ablación.
#[derive(Debug, Clone, PartialEq)]
pub struct AblationConfig {
    /// Mapa discreto compartido por todas las configuraciones.
    pub map: GridMap,
    /// Radio de visión ocluida del sensor físico directo.
    pub vision_radius: usize,
    /// Presupuesto máximo de pasos por episodio.
    pub max_steps: u64,
    /// Presupuesto de transiciones simuladas por paso de decisión (`PlanBudget`).
    pub plan_budget_per_step: PlanBudget,
    /// Población de walkers para `FmcPlanner`.
    pub fmc_population_size: usize,
    /// Horizonte recedente para `FmcPlanner`.
    pub fmc_horizon: usize,
    /// Número de partículas `K` para `BeliefFmc` y `ActiveCommunication`.
    pub belief_particle_count: usize,
    /// Configuración de creencias de celda.
    pub belief_config: GridBeliefConfig,
    /// Configuración de la fuente humana falible.
    pub fallible_human_config: FallibleHumanConfig,
    /// Peso escalar por unidad de comunicación en `ActiveCommunication`.
    pub comm_cost_unit_weight: f64,
    /// Número de bins para la curva de fiabilidad / ECE.
    pub calibration_bins: usize,
    /// Semillas reservadas de desarrollo (`SeedSplit::Dev`).
    pub dev_seeds: Vec<MasterSeed>,
    /// Semillas reservadas de evaluación ciega (`SeedSplit::Holdout`).
    pub holdout_seeds: Vec<MasterSeed>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self::from_master_seed(MasterSeed::new(2_026_046))
    }
}

impl AblationConfig {
    /// Construye una configuración determinista con splits `dev` y `holdout` disjuntos
    /// derivados a partir de `master_seed`.
    pub fn from_master_seed(master_seed: MasterSeed) -> Self {
        let dev_seeds = (0..4_u64)
            .map(|idx| MasterSeed::new(master_seed.derive_seed(StreamKey::Episode(idx + 1))))
            .collect();
        let holdout_seeds = (0..6_u64)
            .map(|idx| {
                MasterSeed::new(master_seed.derive_seed(StreamKey::Agent {
                    episode: idx + 100,
                    agent: 46,
                }))
            })
            .collect();

        Self {
            map: default_ablation_map(),
            vision_radius: 1,
            max_steps: 24,
            plan_budget_per_step: PlanBudget::with_transitions(192),
            fmc_population_size: 8,
            fmc_horizon: 8,
            belief_particle_count: 4,
            belief_config: GridBeliefConfig::default(),
            fallible_human_config: FallibleHumanConfig {
                sensor_id: SensorId::new(DEFAULT_HUMAN_SENSOR_ID),
                sensitivity: 0.72,
                specificity: 0.72,
                stale_lag_nanos: 500_000_000,
            },
            comm_cost_unit_weight: 0.4,
            calibration_bins: DEFAULT_CALIBRATION_BINS,
            dev_seeds,
            holdout_seeds,
        }
    }

    /// Valida que los splits `dev` y `holdout` sean no vacíos y disjuntos, y que los
    /// presupuestos sean válidos.
    pub fn validate(&self) -> Result<(), AblationError> {
        if self.dev_seeds.is_empty() {
            return Err(AblationError::EmptySeedSplit {
                split: SeedSplit::Dev,
            });
        }
        if self.holdout_seeds.is_empty() {
            return Err(AblationError::EmptySeedSplit {
                split: SeedSplit::Holdout,
            });
        }
        let dev_set: BTreeSet<u64> = self.dev_seeds.iter().map(|s| s.as_u64()).collect();
        for seed in &self.holdout_seeds {
            if dev_set.contains(&seed.as_u64()) {
                return Err(AblationError::OverlappingSeedSplits { seed: *seed });
            }
        }
        if self.max_steps == 0 {
            return Err(AblationError::InvalidConfig {
                field: "max_steps",
                reason: "must be >= 1",
            });
        }
        if self.plan_budget_per_step.max_transitions() == 0 {
            return Err(AblationError::InvalidConfig {
                field: "plan_budget_per_step",
                reason: "max_transitions must be >= 1",
            });
        }
        if self.belief_particle_count == 0 {
            return Err(AblationError::InvalidConfig {
                field: "belief_particle_count",
                reason: "must be >= 1",
            });
        }
        if self.fmc_population_size == 0 || self.fmc_horizon == 0 {
            return Err(AblationError::InvalidConfig {
                field: "fmc_population_size/fmc_horizon",
                reason: "must be >= 1",
            });
        }
        if self.calibration_bins == 0 {
            return Err(AblationError::InvalidConfig {
                field: "calibration_bins",
                reason: "must be >= 1",
            });
        }
        if !self.comm_cost_unit_weight.is_finite() || self.comm_cost_unit_weight < 0.0 {
            return Err(AblationError::InvalidConfig {
                field: "comm_cost_unit_weight",
                reason: "must be finite and >= 0.0",
            });
        }
        Ok(())
    }
}

/// Bin individual de fiabilidad / calibración sobre predicciones binarias `[0.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReliabilityBin {
    /// Índice del bin (`0..calibration_bins`).
    pub bin_index: usize,
    /// Límite inferior del intervalo de probabilidad predicha.
    pub lower_bound: f64,
    /// Límite superior del intervalo de probabilidad predicha.
    pub upper_bound: f64,
    /// Número de predicciones asignadas a este bin.
    pub count: usize,
    /// Probabilidad media predicha en este bin (`0.0` si `count == 0`).
    pub mean_predicted_prob: f64,
    /// Frecuencia empírica observada en ground truth (`0.0` si `count == 0`).
    pub empirical_frequency: f64,
    /// Brecha absoluta `|mean_predicted_prob - empirical_frequency|`.
    pub abs_gap: f64,
}

/// Métricas de calibración y scoring propio sobre fixtures offline etiquetados.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefCalibrationMetrics {
    /// Número total de predicciones binarias evaluadas `(celda, hipótesis)`.
    pub sample_count: usize,
    /// Brier score medio en `[0.0, 1.0]`.
    pub brier_score: f64,
    /// Log-loss medio (entropía cruzada binaria con recorte numérico) `>= 0.0`.
    pub log_loss: f64,
    /// Error de calibración esperado (ECE) en `[0.0, 1.0]`.
    pub expected_calibration_error: f64,
    /// Bins uniformes de fiabilidad.
    pub reliability_bins: Vec<ReliabilityBin>,
    /// Descargo explícito de que se trata de fixtures offline y no calibración Jev real.
    pub calibration_disclaimer: &'static str,
}

/// Resultado individual de ejecutar una variante sobre una semilla `holdout`.
#[derive(Debug, Clone, PartialEq)]
pub struct AblationSeedRun {
    /// Configuración evaluada.
    pub variant: AblationVariant,
    /// Split de semilla utilizado (en evaluación ablatoria siempre `SeedSplit::Holdout`).
    pub split: SeedSplit,
    /// Semilla determinista ejecutada.
    pub seed: MasterSeed,
    /// Presupuesto máximo de pasos compartido.
    pub max_steps_budget: u64,
    /// Presupuesto máximo de transiciones por paso compartido.
    pub plan_budget_per_step: usize,
    /// Si alcanzó `CellKind::Goal` dentro del presupuesto.
    pub success: bool,
    /// Si falló (impactó `Hazard` o agotó `max_steps`).
    pub failed: bool,
    /// Número de colisiones contra muro/borde en el episodio.
    pub wall_collisions: u64,
    /// Si pisó una celda `CellKind::Hazard`.
    pub hit_hazard: bool,
    /// Pasos ejecutados en el entorno.
    pub steps: u64,
    /// Transiciones de rollout evaluadas por el planificador.
    pub transitions_evaluated: u64,
    /// Coste comunicativo acumulado en unidades (`CommCost::units()`).
    pub comm_cost_units: u64,
    /// Preguntas `Ask` enviadas por `CommunicationPort`.
    pub asks_dispatched: u64,
    /// Decisiones en las que `EpistemicController` se abstuvo de preguntar (`Advance`/`Abstain`).
    pub asks_abstained: u64,
    /// Incertidumbre epistémica inicial del mapa de creencias (bits).
    pub initial_uncertainty_bits: f64,
    /// Incertidumbre epistémica final del mapa de creencias (bits).
    pub final_uncertainty_bits: f64,
    /// Reducción de incertidumbre `initial - final` (bits).
    pub uncertainty_reduction_bits: f64,
    /// Motivo explícito en caso de fallo.
    pub failure_reason: Option<String>,
}

/// Agregado por variante sobre todas las semillas `holdout` idénticas.
#[derive(Debug, Clone, PartialEq)]
pub struct AblationVariantSummary {
    /// Configuración agregada.
    pub variant: AblationVariant,
    /// Número de semillas `holdout` ejecutadas.
    pub runs_count: usize,
    /// Semillas exactas ejecutadas en orden.
    pub holdout_seeds: Vec<MasterSeed>,
    /// Presupuesto máximo de pasos compartido.
    pub max_steps_budget: u64,
    /// Presupuesto máximo de transiciones por paso compartido.
    pub plan_budget_per_step: usize,
    /// Número de episodios exitosos.
    pub successes_count: usize,
    /// Tasa de éxito en `[0.0, 1.0]`.
    pub success_rate: f64,
    /// Tasa de fallo en `[0.0, 1.0]`.
    pub failure_rate: f64,
    /// Tasa de episodios con colisión contra muro (`wall_collisions > 0`).
    pub wall_collision_rate: f64,
    /// Tasa de episodios con impacto en `Hazard`.
    pub hazard_hit_rate: f64,
    /// Media de pasos ejecutados.
    pub mean_steps: f64,
    /// Media de transiciones simuladas evaluadas por episodio.
    pub mean_transitions_evaluated: f64,
    /// Razón de utilización del presupuesto máximo total de transiciones `steps * budget`.
    pub budget_utilization_ratio: f64,
    /// Media de unidades de coste comunicativo.
    pub mean_comm_cost_units: f64,
    /// Total de preguntas `Ask` enviadas en el split.
    pub total_asks_dispatched: u64,
    /// Total de abstenciones de pregunta en `ActiveCommunication`.
    pub total_asks_abstained: u64,
    /// Incertidumbre inicial media en bits.
    pub mean_initial_uncertainty_bits: f64,
    /// Incertidumbre final media en bits.
    pub mean_final_uncertainty_bits: f64,
    /// Reducción media de incertidumbre en bits.
    pub mean_uncertainty_reduction_bits: f64,
    /// Métricas de Brier, log-loss y fiabilidad/calibración sobre fixtures offline etiquetados.
    pub calibration: BeliefCalibrationMetrics,
}

/// Informe completo y reproducible de la suite de ablación.
#[derive(Debug, Clone, PartialEq)]
pub struct AblationReport {
    /// Semillas de desarrollo declaradas (separadas de `holdout`).
    pub dev_seeds: Vec<MasterSeed>,
    /// Semillas `holdout` compartidas idénticamente por las 6 variantes.
    pub holdout_seeds: Vec<MasterSeed>,
    /// Presupuesto máximo de pasos compartido.
    pub max_steps_budget: u64,
    /// Presupuesto máximo de transiciones por paso compartido.
    pub plan_budget_per_step: usize,
    /// Resúmenes agregados de las 6 configuraciones.
    pub summaries: Vec<AblationVariantSummary>,
    /// Trazas individuales por variante y semilla.
    pub runs: Vec<AblationSeedRun>,
    /// Descargo explícito sobre calibración offline vs calibración Jev real.
    pub calibration_disclaimer: &'static str,
    /// Explicación determinista y honesta de los hallazgos (incluyendo degradaciones).
    pub findings_explanation: String,
}

impl AblationReport {
    /// Busca el resumen agregado de una variante concreta.
    pub fn summary_for(&self, variant: AblationVariant) -> Option<&AblationVariantSummary> {
        self.summaries.iter().find(|s| s.variant == variant)
    }
}

/// Calcula Brier score, log-loss, ECE y bins de fiabilidad sobre muestras `(prob_pred, actual)`.
pub fn compute_calibration_metrics(
    samples: &[(f64, bool)],
    num_bins: usize,
) -> Result<BeliefCalibrationMetrics, AblationError> {
    if num_bins == 0 {
        return Err(AblationError::InvalidConfig {
            field: "calibration_bins",
            reason: "must be >= 1",
        });
    }
    if samples.is_empty() {
        return Err(AblationError::InvalidConfig {
            field: "calibration_samples",
            reason: "cannot compute calibration metrics on empty sample set",
        });
    }

    let n = samples.len() as f64;
    let mut brier_sum = 0.0_f64;
    let mut log_loss_sum = 0.0_f64;
    let mut bin_counts = vec![0_usize; num_bins];
    let mut bin_pred_sum = vec![0.0_f64; num_bins];
    let mut bin_true_sum = vec![0.0_f64; num_bins];

    for &(raw_p, y_bool) in samples {
        let p = raw_p.clamp(0.0, 1.0);
        let y = if y_bool { 1.0 } else { 0.0 };
        let diff = p - y;
        brier_sum += diff * diff;

        let p_clipped = p.clamp(LOG_LOSS_EPS, 1.0 - LOG_LOSS_EPS);
        log_loss_sum += -(y * p_clipped.ln() + (1.0 - y) * (1.0 - p_clipped).ln());

        let mut idx = (p * (num_bins as f64)).floor() as usize;
        if idx >= num_bins {
            idx = num_bins - 1;
        }
        bin_counts[idx] += 1;
        bin_pred_sum[idx] += p;
        bin_true_sum[idx] += y;
    }

    let mut reliability_bins = Vec::with_capacity(num_bins);
    let mut ece = 0.0_f64;
    for b in 0..num_bins {
        let lower_bound = (b as f64) / (num_bins as f64);
        let upper_bound = ((b + 1) as f64) / (num_bins as f64);
        let count = bin_counts[b];
        let (mean_predicted_prob, empirical_frequency, abs_gap) = if count > 0 {
            let c = count as f64;
            let mean_p = bin_pred_sum[b] / c;
            let emp_f = bin_true_sum[b] / c;
            let gap = (mean_p - emp_f).abs();
            ece += (c / n) * gap;
            (mean_p, emp_f, gap)
        } else {
            (0.0, 0.0, 0.0)
        };
        reliability_bins.push(ReliabilityBin {
            bin_index: b,
            lower_bound,
            upper_bound,
            count,
            mean_predicted_prob,
            empirical_frequency,
            abs_gap,
        });
    }

    Ok(BeliefCalibrationMetrics {
        sample_count: samples.len(),
        brier_score: brier_sum / n,
        log_loss: log_loss_sum / n,
        expected_calibration_error: ece,
        reliability_bins,
        calibration_disclaimer: CALIBRATION_FIXTURE_DISCLAIMER,
    })
}

fn collect_belief_predictions(
    beliefs: &GridBeliefMap,
    map: &GridMap,
    out: &mut Vec<(f64, bool)>,
) -> Result<(), AblationError> {
    let size = map.size();
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            let cell = map.cell(pos).expect("in-bounds cell must exist");
            for hypothesis in HypothesisKind::ALL {
                let pred = beliefs.probability(pos, hypothesis)?;
                let actual = hypothesis.matches_cell(cell);
                out.push((pred, actual));
            }
        }
    }
    Ok(())
}

fn inject_perfect_hints(
    beliefs: &mut GridBeliefMap,
    map: &GridMap,
    observed_at: MonotonicInstant,
) -> Result<(), AblationError> {
    let mut msg_id = 50_000_u64;
    let mut ev_id = 500_000_u64;
    let size = map.size();
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            let cell = map.cell(pos).expect("in-bounds cell must exist");
            if !matches!(cell, CellKind::Hazard | CellKind::Goal | CellKind::Wall) {
                continue;
            }
            for hypothesis in HypothesisKind::ALL {
                let present = hypothesis.matches_cell(cell);
                let meta = ProvenanceMetadata {
                    message_id: MessageId::new(msg_id),
                    observed_at,
                    received_at: observed_at,
                    sensor_id: SensorId::new(DEFAULT_HUMAN_SENSOR_ID),
                    frame_id: FrameId::world(),
                    units: UnitTag::dimensionless(),
                    confidence: Confidence::FULL,
                    revision: msg_id,
                    evidence_ref: EvidenceRef::new(ev_id),
                    claim_kind: ObservationKind::Assertion,
                };
                msg_id += 1;
                ev_id += 1;
                let report = ObservationEnvelope::new(
                    meta,
                    CellHypothesis {
                        pos,
                        hypothesis,
                        present,
                    },
                );
                let _ = beliefs.apply_human_report(&report)?;
            }
        }
    }
    Ok(())
}

fn inject_fallible_hints(
    beliefs: &mut GridBeliefMap,
    truth: &GroundTruthGrid,
    fixture: &mut FallibleHumanFixture,
    observed_at: MonotonicInstant,
) -> Result<(), AblationError> {
    let size = truth.grid_size();
    let _sens = fixture.config().sensitivity;
    let _spec = fixture.config().specificity;

    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            if pos == truth.agent_pos() {
                continue;
            }
            for hypothesis in [HypothesisKind::Hazard, HypothesisKind::Goal] {
                let generated = fixture.sample_report(truth, pos, hypothesis, observed_at)?;
                let _ = beliefs.apply_human_report(&generated.report)?;
            }
        }
    }
    Ok(())
}

fn build_belief_fmc_planner(
    config: &AblationConfig,
    seed: MasterSeed,
    particle_count: usize,
) -> Result<BeliefFmcPlanner, AblationError> {
    let per_particle = (config.plan_budget_per_step.max_transitions() / particle_count).max(1);
    let fmc_config = FmcPlannerConfig::new(
        config.fmc_population_size,
        config.fmc_horizon,
        StepDt::from_millis_f64(100.0).map_err(|_| AblationError::InvalidConfig {
            field: "fmc_dt",
            reason: "invalid default step dt",
        })?,
        FmcScoringConfig::default(),
        seed,
    )?;

    let bfc = BeliefFmcConfig {
        particle_count,
        fmc_config,
        budget_per_particle: PlanBudget::with_transitions(per_particle),
        ..BeliefFmcConfig::default()
    };
    Ok(BeliefFmcPlanner::new(bfc)?)
}

fn find_uncertain_query_cell(
    beliefs: &GridBeliefMap,
    agent_pos: GridPos,
    asked_cells: &BTreeSet<(usize, usize)>,
) -> Result<Option<(GridPos, f64)>, AblationError> {
    let size = beliefs.size();
    let mut best: Option<(GridPos, usize, f64)> = None;
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            if pos == agent_pos || asked_cells.contains(&(pos.x, pos.y)) {
                continue;
            }
            if beliefs.direct_update_count(pos, HypothesisKind::Hazard)? > 0 {
                continue;
            }
            let p_haz = beliefs.probability(pos, HypothesisKind::Hazard)?;
            let uncertainty = 0.5 - (p_haz - 0.5).abs();
            let dist = pos.x.abs_diff(agent_pos.x) + pos.y.abs_diff(agent_pos.y);
            if dist <= 3 {
                let replace = match best {
                    None => true,
                    Some((_, best_dist, best_unc)) => {
                        uncertainty > best_unc + 1e-9
                            || ((uncertainty - best_unc).abs() <= 1e-9 && dist < best_dist)
                    }
                };
                if replace {
                    best = Some((pos, dist, p_haz));
                }
            }
        }
    }
    Ok(best.map(|(pos, _, p)| (pos, p)))
}

struct ActiveCommContext {
    asked_cells: BTreeSet<(usize, usize)>,
    clock: VirtualMonotonicClock,
    bus: LocalCommunicationBus,
    comm_cost_units: u64,
    asks_dispatched: u64,
    asks_abstained: u64,
}

fn maybe_run_active_communication(
    config: &AblationConfig,
    beliefs: &mut GridBeliefMap,
    truth: &GroundTruthGrid,
    fixture: &mut FallibleHumanFixture,
    step: u64,
    ctx_comm: &mut ActiveCommContext,
) -> Result<(), AblationError> {
    let Some((target_pos, p_haz)) =
        find_uncertain_query_cell(beliefs, truth.agent_pos(), &ctx_comm.asked_cells)?
    else {
        ctx_comm.asks_abstained += 1;
        return Ok(());
    };

    let agent_id = InterlocutorId::new("ablation-agent")?;
    let oracle_id = InterlocutorId::new("human-oracle")?;
    let opt_enter = OptionId::new("enter")?;
    let opt_detour = OptionId::new("detour")?;
    let opt_yes = OptionId::new("hazard_yes")?;
    let opt_no = OptionId::new("hazard_no")?;

    let payoff_model = DecisionPayoffModel::new(vec![
        DecisionOption::new(opt_enter, 4.0, -14.0)?.with_action(gogamen_core::Action::Discrete(
            DiscreteAction::new(1, GRID_ACTION_CARDINALITY)?,
        )),
        DecisionOption::new(opt_detour, 0.0, 0.0)?,
    ])?;

    let mut controller = EpistemicController::new(
        agent_id.clone(),
        BinaryBelief::new(p_haz.clamp(0.01, 0.99))?,
        payoff_model,
        CommCostWeights::from_unit_weight(config.comm_cost_unit_weight)?,
        FrameId::world(),
        UnitTag::dimensionless(),
    );

    let ask_model = LikelihoodModel::from_sensitivity_specificity(
        config.fallible_human_config.sensitivity,
        config.fallible_human_config.specificity,
    )?;
    let scope = ConversationScope::session("grid-hazard-check")?;
    let ask_spec = gogamen_core::AskSpec::new(
        oracle_id.clone(),
        scope.clone(),
        "Is target cell hazardous?",
        vec![opt_yes.clone(), opt_no.clone()],
        CommCost::from_units(1),
        0.0,
        ask_model.clone(),
        1.0,
        StdDuration::from_secs(2),
    )?;
    let advance_spec = gogamen_core::AdvanceSpec::default();

    let eval = controller.evaluate_actions(&advance_spec, None, Some(&ask_spec))?;
    if eval.selected != EpistemicActionKind::Ask {
        ctx_comm.asks_abstained += 1;
        return Ok(());
    }

    let req_msg_id = MessageId::new(10_000 + step * 10 + 1);
    let reply_msg_id = MessageId::new(10_000 + step * 10 + 2);
    let dispatch =
        controller.dispatch_ask(&mut ctx_comm.bus, &ctx_comm.clock, req_msg_id, &ask_spec)?;
    let _inbox = ctx_comm.bus.poll_inbox(&oracle_id, 4);
    ctx_comm.asks_dispatched += 1;

    ctx_comm.clock.advance(StdDuration::from_millis(50));
    let generated = fixture.sample_report(
        truth,
        target_pos,
        HypothesisKind::Hazard,
        ctx_comm.clock.now(),
    )?;
    let reported_present = generated.report.payload().present;
    let selected_option = if reported_present {
        opt_yes.clone()
    } else {
        opt_no.clone()
    };

    let reply = SituatedMessage::new(
        reply_msg_id,
        oracle_id,
        agent_id,
        scope,
        ctx_comm.clock.now(),
        CommCost::from_units(1),
        CommPayload::answer(
            if reported_present { "hazard" } else { "clear" },
            Some(selected_option),
            Some(generated.report.confidence()),
        )?,
    )?
    .with_correlation_id(dispatch.request_id)?;
    ctx_comm.bus.send(reply)?;

    let mapper =
        BinaryCommOutcomeMapper::new(config.fallible_human_config.sensor_id, opt_yes, opt_no);
    let _poll = controller.poll_ask_reply(
        &mut ctx_comm.bus,
        &ctx_comm.clock,
        &dispatch,
        &ask_model,
        &mapper,
    )?;

    // Frontera estricta de observación: la respuesta solo entra vía ObservationEnvelope a creencias.
    let _ = beliefs.apply_human_report(&generated.report)?;

    // Verificación simbólica de ActionGuard sobre la propuesta derivada de creencia.
    let action_space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
    let guard = ActionGuard::new(SafeFallbackPolicy::RejectOnly);
    let wall_now = std::time::Instant::now();
    let ctx = GuardContext::new(
        &action_space,
        Tick::new(step + 1),
        wall_now,
        Terminal::Ongoing,
    );
    let _guard_decision: GuardDecision =
        controller.propose_guarded_action(Tick::new(step + 1), &guard, &ctx);

    ctx_comm.asked_cells.insert((target_pos.x, target_pos.y));
    ctx_comm.comm_cost_units = ctx_comm.bus.total_cost().units();
    Ok(())
}

fn run_single_ablation_episode(
    config: &AblationConfig,
    variant: AblationVariant,
    seed: MasterSeed,
    episode_idx: u64,
    calibration_samples: &mut Vec<(f64, bool)>,
) -> Result<AblationSeedRun, AblationError> {
    let mut truth = GroundTruthGrid::new(config.map.clone(), config.vision_radius);
    let mut beliefs = GridBeliefMap::new(truth.grid_size(), config.belief_config.clone())?;
    let initial_uncertainty_bits = belief_epistemic_uncertainty_bits(&beliefs)?;

    let clock = VirtualMonotonicClock::new();
    clock.advance(StdDuration::from_millis(10));

    let mut fixture = FallibleHumanFixture::new(seed, config.fallible_human_config)?;
    match variant {
        AblationVariant::PerfectHint => {
            inject_perfect_hints(&mut beliefs, &config.map, clock.now())?;
        }
        AblationVariant::FallibleSource => {
            inject_fallible_hints(&mut beliefs, &truth, &mut fixture, clock.now())?;
        }
        AblationVariant::PhysicalSensorOnly
        | AblationVariant::NominalFmc
        | AblationVariant::BeliefFmc
        | AblationVariant::ActiveCommunication => {}
    }

    let reactive_planner = BeliefGridPlanner::default();
    let fmc_planner = match variant {
        AblationVariant::PhysicalSensorOnly => None,
        AblationVariant::PerfectHint
        | AblationVariant::FallibleSource
        | AblationVariant::NominalFmc => Some(build_belief_fmc_planner(config, seed, 1)?),
        AblationVariant::BeliefFmc | AblationVariant::ActiveCommunication => Some(
            build_belief_fmc_planner(config, seed, config.belief_particle_count)?,
        ),
    };

    let mut bus = LocalCommunicationBus::new(16, 16)?;
    bus.register_interlocutor(
        InterlocutorId::new("ablation-agent")?,
        InterlocutorKind::Agent,
    );
    bus.register_interlocutor(
        InterlocutorId::new("human-oracle")?,
        InterlocutorKind::Human,
    );
    let mut comm_ctx = ActiveCommContext {
        asked_cells: BTreeSet::new(),
        clock: VirtualMonotonicClock::new(),
        bus,
        comm_cost_units: 0,
        asks_dispatched: 0,
        asks_abstained: 0,
    };
    comm_ctx.clock.advance(StdDuration::from_millis(10));

    let mut steps = 0_u64;
    let mut transitions_evaluated = 0_u64;
    let mut wall_collisions = 0_u64;
    let mut hit_hazard = false;
    let mut success = false;
    let mut failure_reason = None;

    for step in 0..config.max_steps {
        clock.advance(StdDuration::from_millis(100));
        let now = clock.now();

        if variant == AblationVariant::ActiveCommunication {
            let obs = truth.observer_view().observe_occluded(now);
            let _ = beliefs.apply_occluded_observation(&obs)?;
            maybe_run_active_communication(
                config,
                &mut beliefs,
                &truth,
                &mut fixture,
                step,
                &mut comm_ctx,
            )?;
        }

        let action = match variant {
            AblationVariant::PhysicalSensorOnly => {
                let mut view = truth.observer_view();
                let (act, _) =
                    reactive_planner.observe_and_choose_action(&mut view, &mut beliefs, now)?;
                act
            }
            AblationVariant::PerfectHint
            | AblationVariant::FallibleSource
            | AblationVariant::NominalFmc => {
                let planner = fmc_planner
                    .as_ref()
                    .expect("nominal FMC planner must exist");
                let mut view = truth.observer_view();
                let decision =
                    planner.observe_and_plan(&mut view, &mut beliefs, now, episode_idx)?;
                transitions_evaluated +=
                    decision.comparison.nominal.trace.rollout_transitions_used() as u64;
                decision.comparison.nominal.first_action
            }
            AblationVariant::BeliefFmc | AblationVariant::ActiveCommunication => {
                let planner = fmc_planner.as_ref().expect("belief FMC planner must exist");
                let mut view = truth.observer_view();
                let decision =
                    planner.observe_and_plan(&mut view, &mut beliefs, now, episode_idx)?;
                transitions_evaluated +=
                    decision.comparison.belief_sampled.total_rollout_transitions as u64;
                decision.selected_action
            }
        };

        let prev_pos = truth.agent_pos();
        let next_pos = truth.step_agent(action);
        steps += 1;

        if action != GridAction::Stay && next_pos == prev_pos {
            wall_collisions += 1;
        }

        if let Some(cell) = config.map.cell(next_pos) {
            match cell {
                CellKind::Goal => {
                    success = true;
                    break;
                }
                CellKind::Hazard => {
                    hit_hazard = true;
                    failure_reason = Some(format!(
                        "stepped on hazard at ({}, {})",
                        next_pos.x, next_pos.y
                    ));
                    break;
                }
                CellKind::Empty | CellKind::Wall => {}
            }
        }
    }

    if !success && failure_reason.is_none() {
        failure_reason = Some(format!(
            "step budget exhausted ({} steps)",
            config.max_steps
        ));
    }

    collect_belief_predictions(&beliefs, &config.map, calibration_samples)?;
    let final_uncertainty_bits = belief_epistemic_uncertainty_bits(&beliefs)?;

    Ok(AblationSeedRun {
        variant,
        split: SeedSplit::Holdout,
        seed,
        max_steps_budget: config.max_steps,
        plan_budget_per_step: config.plan_budget_per_step.max_transitions(),
        success,
        failed: !success,
        wall_collisions,
        hit_hazard,
        steps,
        transitions_evaluated,
        comm_cost_units: comm_ctx.comm_cost_units,
        asks_dispatched: comm_ctx.asks_dispatched,
        asks_abstained: comm_ctx.asks_abstained,
        initial_uncertainty_bits,
        final_uncertainty_bits,
        uncertainty_reduction_bits: initial_uncertainty_bits - final_uncertainty_bits,
        failure_reason,
    })
}

fn build_findings_explanation(
    config: &AblationConfig,
    summaries: &[AblationVariantSummary],
) -> String {
    let find = |v: AblationVariant| summaries.iter().find(|s| s.variant == v);
    let mut lines = Vec::new();
    lines.push(format!(
        "Evaluación ablatoria reproducible sobre {} semillas holdout idénticas (disjuntas de {} semillas dev), presupuesto max_steps={} y PlanBudget={} transiciones/paso.",
        config.holdout_seeds.len(),
        config.dev_seeds.len(),
        config.max_steps,
        config.plan_budget_per_step.max_transitions()
    ));

    if let (Some(phys), Some(perf), Some(fall)) = (
        find(AblationVariant::PhysicalSensorOnly),
        find(AblationVariant::PerfectHint),
        find(AblationVariant::FallibleSource),
    ) {
        lines.push(format!(
            "Sensor físico vs avisos: PhysicalSensorOnly éxito={:.2}% (riesgo={:.2}%), PerfectHint éxito={:.2}% (Brier={:.4}, ECE={:.4}), FallibleSource éxito={:.2}% (riesgo={:.2}%, Brier={:.4}, log-loss={:.4}, ECE={:.4}).",
            phys.success_rate * 100.0,
            phys.hazard_hit_rate * 100.0,
            perf.success_rate * 100.0,
            perf.calibration.brier_score,
            perf.calibration.expected_calibration_error,
            fall.success_rate * 100.0,
            fall.hazard_hit_rate * 100.0,
            fall.calibration.brier_score,
            fall.calibration.log_loss,
            fall.calibration.expected_calibration_error
        ));
        if fall.success_rate < perf.success_rate
            || fall.hazard_hit_rate > perf.hazard_hit_rate
            || fall.calibration.brier_score > perf.calibration.brier_score
        {
            lines.push(
                "Hallazgo negativo/degradación reportado explícitamente: FallibleSource degrada el rendimiento o la calibración respecto a PerfectHint debido a falsos positivos/negativos en los informes humanos ruidosos."
                    .to_string(),
            );
        }
    }

    if let (Some(nom), Some(bel)) = (
        find(AblationVariant::NominalFmc),
        find(AblationVariant::BeliefFmc),
    ) {
        lines.push(format!(
            "FMC nominal vs creencias bajo presupuesto idéntico: NominalFmc éxito={:.2}% (transiciones={:.1}), BeliefFmc éxito={:.2}% (transiciones={:.1}, K={} partículas).",
            nom.success_rate * 100.0,
            nom.mean_transitions_evaluated,
            bel.success_rate * 100.0,
            bel.mean_transitions_evaluated,
            config.belief_particle_count
        ));
        if bel.success_rate < nom.success_rate {
            lines.push(
                "Hallazgo negativo reportado: bajo presupuesto fijo dividido entre K partículas (budget/K), BeliefFmc dispone de menos rollouts por partícula que NominalFmc, pudiendo perder precisión de búsqueda en mapas deterministas."
                    .to_string(),
            );
        }
    }

    if let Some(comm) = find(AblationVariant::ActiveCommunication) {
        lines.push(format!(
            "ActiveCommunication (Ask/Inspect/Advance): éxito={:.2}%, asks={}, abstenciones={}, coste_medio={:.2} unidades, reducción_incertidumbre={:.2} bits.",
            comm.success_rate * 100.0,
            comm.total_asks_dispatched,
            comm.total_asks_abstained,
            comm.mean_comm_cost_units,
            comm.mean_uncertainty_reduction_bits
        ));
    }

    lines.push(CALIBRATION_FIXTURE_DISCLAIMER.to_string());
    lines.join(" ")
}

/// Ejecuta la suite completa de ablación sobre las 6 configuraciones usando exactamente
/// las mismas semillas `holdout` y el mismo presupuesto (`max_steps` + `PlanBudget`).
pub fn run_ablation_suite(config: &AblationConfig) -> Result<AblationReport, AblationError> {
    config.validate()?;

    let mut summaries = Vec::with_capacity(AblationVariant::ALL.len());
    let mut all_runs = Vec::with_capacity(AblationVariant::ALL.len() * config.holdout_seeds.len());

    for variant in AblationVariant::ALL {
        let mut variant_runs = Vec::with_capacity(config.holdout_seeds.len());
        let mut calibration_samples = Vec::new();

        for (idx, &seed) in config.holdout_seeds.iter().enumerate() {
            let run = run_single_ablation_episode(
                config,
                variant,
                seed,
                idx as u64,
                &mut calibration_samples,
            )?;
            variant_runs.push(run);
        }

        let n = variant_runs.len() as f64;
        let successes_count = variant_runs.iter().filter(|r| r.success).count();
        let failures_count = variant_runs.iter().filter(|r| r.failed).count();
        let wall_collision_episodes = variant_runs
            .iter()
            .filter(|r| r.wall_collisions > 0)
            .count();
        let hazard_episodes = variant_runs.iter().filter(|r| r.hit_hazard).count();

        let sum_steps: u64 = variant_runs.iter().map(|r| r.steps).sum();
        let sum_transitions: u64 = variant_runs.iter().map(|r| r.transitions_evaluated).sum();
        let sum_comm_cost: u64 = variant_runs.iter().map(|r| r.comm_cost_units).sum();
        let total_asks_dispatched: u64 = variant_runs.iter().map(|r| r.asks_dispatched).sum();
        let total_asks_abstained: u64 = variant_runs.iter().map(|r| r.asks_abstained).sum();

        let sum_init_unc: f64 = variant_runs
            .iter()
            .map(|r| r.initial_uncertainty_bits)
            .sum();
        let sum_final_unc: f64 = variant_runs.iter().map(|r| r.final_uncertainty_bits).sum();
        let sum_red_unc: f64 = variant_runs
            .iter()
            .map(|r| r.uncertainty_reduction_bits)
            .sum();

        let max_possible_transitions =
            (sum_steps as f64) * (config.plan_budget_per_step.max_transitions() as f64);
        let budget_utilization_ratio = if max_possible_transitions > 0.0 {
            (sum_transitions as f64) / max_possible_transitions
        } else {
            0.0
        };

        let calibration =
            compute_calibration_metrics(&calibration_samples, config.calibration_bins)?;

        summaries.push(AblationVariantSummary {
            variant,
            runs_count: variant_runs.len(),
            holdout_seeds: config.holdout_seeds.clone(),
            max_steps_budget: config.max_steps,
            plan_budget_per_step: config.plan_budget_per_step.max_transitions(),
            successes_count,
            success_rate: (successes_count as f64) / n,
            failure_rate: (failures_count as f64) / n,
            wall_collision_rate: (wall_collision_episodes as f64) / n,
            hazard_hit_rate: (hazard_episodes as f64) / n,
            mean_steps: (sum_steps as f64) / n,
            mean_transitions_evaluated: (sum_transitions as f64) / n,
            budget_utilization_ratio,
            mean_comm_cost_units: (sum_comm_cost as f64) / n,
            total_asks_dispatched,
            total_asks_abstained,
            mean_initial_uncertainty_bits: sum_init_unc / n,
            mean_final_uncertainty_bits: sum_final_unc / n,
            mean_uncertainty_reduction_bits: sum_red_unc / n,
            calibration,
        });
        all_runs.extend(variant_runs);
    }

    let findings_explanation = build_findings_explanation(config, &summaries);

    Ok(AblationReport {
        dev_seeds: config.dev_seeds.clone(),
        holdout_seeds: config.holdout_seeds.clone(),
        max_steps_budget: config.max_steps,
        plan_budget_per_step: config.plan_budget_per_step.max_transitions(),
        summaries,
        runs: all_runs,
        calibration_disclaimer: CALIBRATION_FIXTURE_DISCLAIMER,
        findings_explanation,
    })
}

/// Demostración determinista reproducible por [`MasterSeed`] con splits `dev`/`holdout`
/// disjuntos y explicación explícita de hallazgos.
pub fn run_ablation_demo(master_seed: MasterSeed) -> Result<AblationReport, AblationError> {
    let config = AblationConfig::from_master_seed(master_seed);
    run_ablation_suite(&config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_all_six_variants_share_identical_holdout_seeds_and_budgets() {
        let config = AblationConfig::default();
        let report = run_ablation_suite(&config).expect("ablation suite must succeed");

        assert_eq!(report.summaries.len(), 6);
        assert_eq!(report.holdout_seeds, config.holdout_seeds);
        assert!(!report.dev_seeds.is_empty());

        let dev_set: BTreeSet<u64> = report.dev_seeds.iter().map(|s| s.as_u64()).collect();
        for seed in &report.holdout_seeds {
            assert!(
                !dev_set.contains(&seed.as_u64()),
                "holdout seed must be disjoint from dev seeds"
            );
        }

        for variant in AblationVariant::ALL {
            let summary = report
                .summary_for(variant)
                .expect("every variant must have a summary");
            assert_eq!(summary.holdout_seeds, config.holdout_seeds);
            assert_eq!(summary.runs_count, config.holdout_seeds.len());
            assert_eq!(summary.max_steps_budget, config.max_steps);
            assert_eq!(
                summary.plan_budget_per_step,
                config.plan_budget_per_step.max_transitions()
            );
            assert!(summary.success_rate >= 0.0 && summary.success_rate <= 1.0);
            assert!(summary.failure_rate >= 0.0 && summary.failure_rate <= 1.0);
            assert!(summary.wall_collision_rate >= 0.0 && summary.wall_collision_rate <= 1.0);
            assert!(summary.hazard_hit_rate >= 0.0 && summary.hazard_hit_rate <= 1.0);
            assert!(summary.mean_uncertainty_reduction_bits.is_finite());
            assert!(summary.budget_utilization_ratio <= 1.0 + 1e-9);
        }

        let comm_summary = report
            .summary_for(AblationVariant::ActiveCommunication)
            .expect("ActiveCommunication summary must exist");
        assert!(comm_summary.total_asks_dispatched > 0);
        assert!(comm_summary.mean_comm_cost_units > 0.0);
    }

    #[test]
    fn ac2_calibration_metrics_compute_exact_brier_logloss_ece_with_non_jev_disclaimer() {
        let samples = vec![(0.8, true), (0.2, false), (0.8, false), (0.2, true)];
        let metrics = compute_calibration_metrics(&samples, 5).expect("metrics must compute");

        // Brier: ((0.2)^2 + (0.2)^2 + (0.8)^2 + (0.8)^2) / 4 = (0.04 + 0.04 + 0.64 + 0.64)/4 = 0.34
        assert!((metrics.brier_score - 0.34).abs() < 1e-12);
        assert!(metrics.log_loss.is_finite() && metrics.log_loss > 0.0);
        // En bins 0.2 (2 muestras, emp=0.5, gap=0.3) y 0.8 (2 muestras, emp=0.5, gap=0.3) -> ECE=0.3
        assert!((metrics.expected_calibration_error - 0.30).abs() < 1e-12);
        assert_eq!(metrics.reliability_bins.len(), 5);
        assert!(metrics.calibration_disclaimer.contains("no constituyen"));
        assert!(metrics.calibration_disclaimer.contains("Jev"));

        let report = run_ablation_demo(MasterSeed::new(777)).expect("demo must succeed");
        assert_eq!(
            report.calibration_disclaimer,
            CALIBRATION_FIXTURE_DISCLAIMER
        );
        for summary in &report.summaries {
            assert!(summary.calibration.brier_score.is_finite());
            assert!(summary.calibration.log_loss.is_finite());
            assert!(summary.calibration.expected_calibration_error.is_finite());
            assert_eq!(
                summary.calibration.calibration_disclaimer,
                CALIBRATION_FIXTURE_DISCLAIMER
            );
        }
    }

    #[test]
    fn ac3_demo_is_deterministic_and_reports_findings_without_hiding_degradation() {
        let seed = MasterSeed::new(2_026_046);
        let r1 = run_ablation_demo(seed).expect("first run");
        let r2 = run_ablation_demo(seed).expect("second run");

        assert_eq!(r1, r2);
        assert!(!r1.findings_explanation.is_empty());
        assert!(r1.findings_explanation.contains("PhysicalSensorOnly"));
        assert!(r1.findings_explanation.contains("FallibleSource"));
        assert!(r1.findings_explanation.contains("NominalFmc"));
        assert!(r1.findings_explanation.contains("BeliefFmc"));
        assert!(r1.findings_explanation.contains("ActiveCommunication"));
        assert!(r1.findings_explanation.contains("Jev"));

        // Verifica que si los splits dev/holdout se solapan se rechaza explícitamente.
        let mut bad_cfg = AblationConfig::from_master_seed(seed);
        bad_cfg.holdout_seeds[0] = bad_cfg.dev_seeds[0];
        let err = run_ablation_suite(&bad_cfg).expect_err("overlapping seeds must fail");
        assert!(matches!(err, AblationError::OverlappingSeedSplits { .. }));
    }
}
