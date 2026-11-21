//! Runner headless de comparación científica pareada entre dos planificadores.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

use gogamen_core::{
    Action, Actuator, CancelToken, DomainError, EnvironmentError, EnvironmentObserver, FmcPlanner,
    FmcPlannerConfig, FmcScoringConfig, GenerativeModel, MasterSeed, ModelError, PlanBudget,
    PlanRequest, PlanStatus, PlanningContext, RngSource, RunId, State, StepDt, StreamKey,
    StreamRevision, Terminal, TerminalReason, Tick, Transition,
};
use gogamen_lab_cartpole::{
    CartPoleConfig, CartPoleEnv, CartPolePolicy, EntropicExperimentConfig, EntropicProxyController,
    LinearFeedbackController,
};
use gogamen_lab_grid::{
    GridAction, GridMap, GridPos, GridWorldConfig, GridWorldEnv, GridWorldModel, MonteCarloConfig,
    MonteCarloPlanner, ObservationMode, SeedSplit,
};
use gogamen_runtime::scientific_trace::{
    BufferedTraceSink, DecisionAction, InMemoryTraceBackend, OverflowPolicy, RecordedGuardDecision,
    ScientificRecord, ScientificTraceArchive, ScientificTraceError, SensitiveFieldPolicy,
    StreamSeeds, TickInputs, TraceManifest, TraceManifestConfig, TraceQuota, TraceWriteError,
};

use crate::stats::{
    PairedComparisonSummary, PairedWinner, PlannerSummary, RunOutcomeStatus,
    summarize_paired_outcomes, summarize_planner_runs,
};

/// Errores explícitos de configuración o ejecución de la arena científica.
#[derive(Debug, Clone, PartialEq)]
pub enum ArenaError {
    /// El conjunto de semillas está vacío.
    EmptySeedSet,
    /// Una misma semilla aparece repetida dentro de un mismo split.
    DuplicateSeed {
        /// Split donde se detectó el duplicado.
        split: SeedSplit,
        /// Semilla duplicada.
        seed: MasterSeed,
    },
    /// Una semilla de `holdout` se solapa con `dev`, violando la separación estricta.
    OverlappingHoldoutSeeds {
        /// Semilla que aparece tanto en `dev` como en `holdout`.
        seed: MasterSeed,
    },
    /// Presupuesto inválido (`max_ticks == 0` o `max_transitions == 0`).
    InvalidBudget {
        /// Descripción del parámetro inválido.
        reason: &'static str,
    },
    /// El planificador no es compatible con el escenario configurado.
    IncompatiblePlanner {
        /// Lado (`"left"` o `"right"`).
        side: &'static str,
        /// Identificador del planificador incompatible.
        planner_id: String,
        /// Escenario objetivo.
        scenario: &'static str,
    },
    /// Error de dominio o validación de entorno.
    Domain(DomainError),
    /// Error de traza científica durable.
    Trace(String),
}

impl fmt::Display for ArenaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeedSet => write!(f, "arena seed set cannot be empty"),
            Self::DuplicateSeed { split, seed } => {
                write!(f, "duplicate seed {} in {:?} split", seed.as_u64(), split)
            }
            Self::OverlappingHoldoutSeeds { seed } => write!(
                f,
                "holdout seed {} overlaps with dev seeds; sets must be strictly disjoint",
                seed.as_u64()
            ),
            Self::InvalidBudget { reason } => write!(f, "invalid arena budget: {reason}"),
            Self::IncompatiblePlanner {
                side,
                planner_id,
                scenario,
            } => write!(
                f,
                "planner '{planner_id}' on {side} side is incompatible with {scenario} scenario"
            ),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Trace(msg) => write!(f, "scientific trace error: {msg}"),
        }
    }
}

impl Error for ArenaError {}

impl From<DomainError> for ArenaError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

impl From<ScientificTraceError> for ArenaError {
    fn from(err: ScientificTraceError) -> Self {
        Self::Trace(err.to_string())
    }
}

impl From<TraceWriteError> for ArenaError {
    fn from(err: TraceWriteError) -> Self {
        Self::Trace(err.to_string())
    }
}

/// Presupuesto compartido impuesto por igual a ambos planificadores en cada episodio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArenaBudget {
    /// Número máximo de pasos (ticks) por episodio.
    pub max_ticks: u64,
    /// Número máximo de transiciones totales (entorno + simulaciones) por episodio.
    pub max_transitions: u64,
    /// Presupuesto máximo de transiciones simuladas por tick (`PlanBudget`).
    pub max_transitions_per_tick: usize,
    /// Timeout opcional de tiempo real por episodio.
    pub per_episode_timeout: Option<Duration>,
}

impl ArenaBudget {
    /// Construye un presupuesto compartido explícito.
    pub const fn new(
        max_ticks: u64,
        max_transitions: u64,
        max_transitions_per_tick: usize,
    ) -> Self {
        Self {
            max_ticks,
            max_transitions,
            max_transitions_per_tick,
            per_episode_timeout: None,
        }
    }

    /// Asocia un timeout opcional de pared por episodio.
    pub const fn with_episode_timeout(mut self, timeout: Duration) -> Self {
        self.per_episode_timeout = Some(timeout);
        self
    }

    /// Valida que el presupuesto sea positivo y coherente.
    pub fn validate(&self) -> Result<(), ArenaError> {
        if self.max_ticks == 0 {
            return Err(ArenaError::InvalidBudget {
                reason: "max_ticks must be > 0",
            });
        }
        if self.max_transitions == 0 {
            return Err(ArenaError::InvalidBudget {
                reason: "max_transitions must be > 0",
            });
        }
        Ok(())
    }
}

impl Default for ArenaBudget {
    fn default() -> Self {
        Self::new(80, 16_000, 160)
    }
}

/// Conjuntos de semillas `dev` y `holdout` estrictamente disjuntos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArenaSeedSet {
    /// Semillas de desarrollo y calibración (`SeedSplit::Dev`).
    pub dev_seeds: Vec<MasterSeed>,
    /// Semillas reservadas de evaluación ciega (`SeedSplit::Holdout`).
    pub holdout_seeds: Vec<MasterSeed>,
}

impl ArenaSeedSet {
    /// Crea un conjunto validando que `dev_seeds` y `holdout_seeds` sean disjuntos y sin duplicados.
    pub fn new(
        dev_seeds: Vec<MasterSeed>,
        holdout_seeds: Vec<MasterSeed>,
    ) -> Result<Self, ArenaError> {
        let set = Self {
            dev_seeds,
            holdout_seeds,
        };
        set.validate()?;
        Ok(set)
    }

    /// Valida separación estricta `dev` vs `holdout`, ausencia de duplicados y conjunto no vacío.
    pub fn validate(&self) -> Result<(), ArenaError> {
        if self.dev_seeds.is_empty() && self.holdout_seeds.is_empty() {
            return Err(ArenaError::EmptySeedSet);
        }

        let mut dev_seen = BTreeSet::new();
        for &seed in &self.dev_seeds {
            if !dev_seen.insert(seed) {
                return Err(ArenaError::DuplicateSeed {
                    split: SeedSplit::Dev,
                    seed,
                });
            }
        }

        let mut holdout_seen = BTreeSet::new();
        for &seed in &self.holdout_seeds {
            if !holdout_seen.insert(seed) {
                return Err(ArenaError::DuplicateSeed {
                    split: SeedSplit::Holdout,
                    seed,
                });
            }
            if dev_seen.contains(&seed) {
                return Err(ArenaError::OverlappingHoldoutSeeds { seed });
            }
        }
        Ok(())
    }

    /// Itera sobre todas las semillas etiquetadas con su [`SeedSplit`] (`Dev` seguido de `Holdout`).
    pub fn iter_tagged(&self) -> impl Iterator<Item = (SeedSplit, MasterSeed)> + '_ {
        self.dev_seeds
            .iter()
            .copied()
            .map(|s| (SeedSplit::Dev, s))
            .chain(
                self.holdout_seeds
                    .iter()
                    .copied()
                    .map(|s| (SeedSplit::Holdout, s)),
            )
    }
}

impl Default for ArenaSeedSet {
    fn default() -> Self {
        Self {
            dev_seeds: (1..=5).map(MasterSeed::new).collect(),
            holdout_seeds: (101..=105).map(MasterSeed::new).collect(),
        }
    }
}

/// Configuración de escenario Grid compartido para comparación en arena.
#[derive(Debug, Clone, PartialEq)]
pub struct GridArenaConfig {
    /// Etiqueta descriptiva del mapa/escenario.
    pub label: String,
    /// Mapa discreto compartido por ambos planificadores.
    pub map: GridMap,
    /// Reglas y contrato de observación compartido (`ObservationMode`).
    pub world_config: GridWorldConfig,
}

impl GridArenaConfig {
    /// Construye un escenario Grid asegurando que `world_config.map` coincida con `map`.
    pub fn new(
        label: impl Into<String>,
        map: GridMap,
        mut world_config: GridWorldConfig,
    ) -> Result<Self, ArenaError> {
        world_config.map = map.clone();
        world_config.validate()?;
        Ok(Self {
            label: label.into(),
            map,
            world_config,
        })
    }

    /// Configura el [`ObservationMode`] compartido por ambos planificadores.
    pub fn with_observation_mode(mut self, mode: ObservationMode) -> Self {
        self.world_config.observation_mode = mode;
        self
    }
}

impl Default for GridArenaConfig {
    fn default() -> Self {
        let world_config = GridWorldConfig::default();
        let map = world_config.map.clone();
        Self {
            label: "grid-cul-de-sac".to_owned(),
            map,
            world_config,
        }
    }
}

/// Configuración de escenario CartPole compartido para comparación en arena.
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleArenaConfig {
    /// Etiqueta descriptiva del escenario CartPole.
    pub label: String,
    /// Parámetros físicos y estado inicial compartidos.
    pub world_config: CartPoleConfig,
    /// Paso temporal de simulación.
    pub dt: StepDt,
}

impl CartPoleArenaConfig {
    /// Crea un escenario CartPole validando su configuración física.
    pub fn new(
        label: impl Into<String>,
        world_config: CartPoleConfig,
        dt: StepDt,
    ) -> Result<Self, ArenaError> {
        world_config.validate()?;
        Ok(Self {
            label: label.into(),
            world_config,
            dt,
        })
    }
}

impl Default for CartPoleArenaConfig {
    fn default() -> Self {
        Self {
            label: "cartpole-classic".to_owned(),
            world_config: CartPoleConfig::default(),
            dt: StepDt::from_secs_f64(0.02).expect("0.02s is a valid StepDt"),
        }
    }
}

/// Escenario compartido por ambos planificadores en una comparación de arena.
#[derive(Debug, Clone, PartialEq)]
pub enum ArenaScenario {
    /// Mundo discreto Grid con `ObservationMode` compartido.
    Grid(GridArenaConfig),
    /// Péndulo invertido continuo CartPole.
    CartPole(CartPoleArenaConfig),
}

impl ArenaScenario {
    /// Etiqueta legible del escenario.
    pub fn label(&self) -> &str {
        match self {
            Self::Grid(cfg) => &cfg.label,
            Self::CartPole(cfg) => &cfg.label,
        }
    }

    /// Contrato explícito de observación compartido por ambos planificadores.
    pub fn observation_contract_label(&self) -> String {
        match self {
            Self::Grid(cfg) => match cfg.world_config.observation_mode {
                ObservationMode::Full => "GridObservation::Full".to_owned(),
                ObservationMode::Partial { radius } => {
                    format!("GridObservation::Partial(radius={radius})")
                }
            },
            Self::CartPole(_) => "CartPoleObservation::FullKinematics".to_owned(),
        }
    }

    const fn kind_label(&self) -> &'static str {
        match self {
            Self::Grid(_) => "Grid",
            Self::CartPole(_) => "CartPole",
        }
    }
}

/// Planificador o controlador existente seleccionable en la arena.
#[derive(Debug, Clone, PartialEq)]
pub enum ArenaPlanner {
    /// Política aleatoria uniforme reproducible sobre Grid.
    GridRandomSeeded,
    /// Planificador Monte Carlo random shooting sobre Grid (`MonteCarloPlanner`).
    GridMonteCarlo {
        /// Horizonte de cada rollout.
        horizon: usize,
        /// Número de rollouts por decisión.
        rollouts: usize,
    },
    /// Planificador Fractal Monte Carlo recedente sobre Grid (`FmcPlanner`).
    GridFmc {
        /// Tamaño de población de walkers.
        population_size: usize,
        /// Horizonte recedente.
        horizon: usize,
        /// Configuración de scoring y resampling.
        scoring: FmcScoringConfig,
    },
    /// Techo privilegiado BFS con acceso completo al grafo del mapa (`is_privileged_full_info = true`).
    GridPrivilegedBfs,
    /// Controlador clásico por realimentación lineal para CartPole.
    CartPoleLinearFeedback(LinearFeedbackController),
    /// Controlador entrópico para CartPole (`EntropicProxyController`).
    CartPoleEntropicProxy(Box<EntropicExperimentConfig>),
}

impl ArenaPlanner {
    /// Identificador legible del planificador.
    pub fn id(&self) -> String {
        match self {
            Self::GridRandomSeeded => "GridRandomSeeded".to_owned(),
            Self::GridMonteCarlo { horizon, rollouts } => {
                format!("GridMonteCarlo(h={horizon},r={rollouts})")
            }
            Self::GridFmc {
                population_size,
                horizon,
                ..
            } => format!("GridFmc(pop={population_size},h={horizon})"),
            Self::GridPrivilegedBfs => "GridPrivilegedBfs".to_owned(),
            Self::CartPoleLinearFeedback(_) => "CartPoleLinearFeedback".to_owned(),
            Self::CartPoleEntropicProxy(_) => "CartPoleEntropicProxy".to_owned(),
        }
    }

    /// Indica si el planificador usa información privilegiada completa.
    pub const fn is_privileged_full_info(&self) -> bool {
        matches!(self, Self::GridPrivilegedBfs)
    }

    fn is_compatible_with(&self, scenario: &ArenaScenario) -> bool {
        matches!(
            (self, scenario),
            (
                Self::GridRandomSeeded
                    | Self::GridMonteCarlo { .. }
                    | Self::GridFmc { .. }
                    | Self::GridPrivilegedBfs,
                ArenaScenario::Grid(_)
            ) | (
                Self::CartPoleLinearFeedback(_) | Self::CartPoleEntropicProxy(_),
                ArenaScenario::CartPole(_)
            )
        )
    }
}

/// Configuración completa de una comparación científica pareada en arena headless.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaComparisonConfig {
    /// Identificador de especificación del experimento para trazabilidad científica.
    pub spec_id: String,
    /// Escenario compartido (reglas y contrato de observación).
    pub scenario: ArenaScenario,
    /// Presupuesto compartido por episodio y por tick.
    pub budget: ArenaBudget,
    /// Conjuntos de semillas `dev` y `holdout` disjuntos.
    pub seeds: ArenaSeedSet,
    /// Primer planificador a comparar (`left`).
    pub left_planner: ArenaPlanner,
    /// Segundo planificador a comparar (`right`).
    pub right_planner: ArenaPlanner,
}

impl ArenaComparisonConfig {
    /// Valida la configuración completa antes de ejecutar ningún episodio.
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.budget.validate()?;
        self.seeds.validate()?;
        if let ArenaScenario::Grid(grid) = &self.scenario {
            grid.world_config.validate()?;
        } else if let ArenaScenario::CartPole(cp) = &self.scenario {
            cp.world_config.validate()?;
        }
        if !self.left_planner.is_compatible_with(&self.scenario) {
            return Err(ArenaError::IncompatiblePlanner {
                side: "left",
                planner_id: self.left_planner.id(),
                scenario: self.scenario.kind_label(),
            });
        }
        if !self.right_planner.is_compatible_with(&self.scenario) {
            return Err(ArenaError::IncompatiblePlanner {
                side: "right",
                planner_id: self.right_planner.id(),
                scenario: self.scenario.kind_label(),
            });
        }
        Ok(())
    }
}

/// Resultado individual de un planificador sobre una semilla y split concretos,
/// incluyendo su traza científica verificada compatible con `gogamen_runtime::scientific_trace`.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerRunResult {
    /// Identificador del planificador.
    pub planner_id: String,
    /// Marca explícita de acceso privilegiado completo.
    pub is_privileged_full_info: bool,
    /// Split de la semilla (`Dev` o `Holdout`).
    pub split: SeedSplit,
    /// Semilla determinista de la ejecución.
    pub seed: MasterSeed,
    /// Estado explícito de finalización (`Succeeded`, `Failed`, `TimedOut`, `BudgetExhausted`).
    pub status: RunOutcomeStatus,
    /// Atajo booleano de `status.is_succeeded()`.
    pub succeeded: bool,
    /// Pasos ejecutados en el entorno vivo.
    pub steps_executed: u64,
    /// Transiciones totales evaluadas (entorno + simulaciones de planificación).
    pub transitions_evaluated: u64,
    /// Fracción del presupuesto de pasos consumida (`steps_executed / max_ticks`).
    pub budget_utilization_steps: f64,
    /// Fracción del presupuesto de transiciones consumida (`transitions_evaluated / max_transitions`).
    pub budget_utilization_transitions: f64,
    /// Recompensa acumulada en el episodio.
    pub total_reward: f64,
    /// Latencia medida en nanosegundos.
    pub latency_ns: u128,
    /// Manifiesto de la traza científica generada.
    pub trace_manifest: TraceManifest,
    /// Archivo completo autocontenido (manifiesto + registros científicos por tick).
    pub trace_archive: ScientificTraceArchive,
}

/// Resultado pareado de ambos planificadores sobre exactamente la misma semilla, escenario y presupuesto.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedSeedOutcome {
    /// Split de la semilla (`Dev` o `Holdout`).
    pub split: SeedSplit,
    /// Semilla compartida.
    pub seed: MasterSeed,
    /// Resultado del planificador izquierdo.
    pub left: PlannerRunResult,
    /// Resultado del planificador derecho.
    pub right: PlannerRunResult,
    /// Diferencia pareada de pasos (`left.steps - right.steps`).
    pub steps_delta: i64,
    /// Diferencia pareada de transiciones (`left.transitions - right.transitions`).
    pub transitions_delta: i64,
    /// Diferencia pareada de recompensa (`left.total_reward - right.total_reward`).
    pub reward_delta: f64,
    /// Ganador pareado en esta semilla.
    pub outcome_winner: PairedWinner,
}

/// Informe científico completo de la comparación pareada en arena headless.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaComparisonReport {
    /// Identificador de especificación del experimento.
    pub spec_id: String,
    /// Etiqueta del escenario compartido.
    pub scenario_label: String,
    /// Contrato de observación compartido.
    pub observation_contract: String,
    /// Presupuesto compartido.
    pub budget: ArenaBudget,
    /// Conjunto de semillas validado (`dev` y `holdout` disjuntos).
    pub seeds: ArenaSeedSet,
    /// Identificador del planificador izquierdo.
    pub left_planner_id: String,
    /// Identificador del planificador derecho.
    pub right_planner_id: String,
    /// Resultados pareados por semilla (en orden `Dev` seguido de `Holdout`).
    pub paired_outcomes: Vec<PairedSeedOutcome>,
    /// Agregado global del planificador izquierdo (denominador intacto).
    pub left_overall: PlannerSummary,
    /// Agregado global del planificador derecho (denominador intacto).
    pub right_overall: PlannerSummary,
    /// Agregado `Dev` del planificador izquierdo (`None` si no hubo semillas `Dev`).
    pub left_dev: Option<PlannerSummary>,
    /// Agregado `Dev` del planificador derecho (`None` si no hubo semillas `Dev`).
    pub right_dev: Option<PlannerSummary>,
    /// Agregado `Holdout` del planificador izquierdo (`None` si no hubo semillas `Holdout`).
    pub left_holdout: Option<PlannerSummary>,
    /// Agregado `Holdout` del planificador derecho (`None` si no hubo semillas `Holdout`).
    pub right_holdout: Option<PlannerSummary>,
    /// Resumen pareado global.
    pub paired_overall: PairedComparisonSummary,
    /// Resumen pareado sobre `Dev`.
    pub paired_dev: Option<PairedComparisonSummary>,
    /// Resumen pareado sobre `Holdout`.
    pub paired_holdout: Option<PairedComparisonSummary>,
}

impl ArenaComparisonReport {
    /// Formatea un resumen legible para inspección headless o exportación a texto.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "=== Gogamen Arena Comparison [{}] ===\nScenario: {} | Observation: {}\nBudget: max_ticks={}, max_transitions={}, max_transitions_per_tick={}\nLeft: {} | Right: {}\n",
            self.spec_id,
            self.scenario_label,
            self.observation_contract,
            self.budget.max_ticks,
            self.budget.max_transitions,
            self.budget.max_transitions_per_tick,
            self.left_planner_id,
            self.right_planner_id,
        ));
        for summary in [&self.left_overall, &self.right_overall] {
            out.push_str(&format!(
                "- {} (priv={}): success={}/{} ({:.1}%), failed={}, timed_out={}, budget_exhausted={}, steps_med={:.1} (iqr={:.1}, std={:.2}), trans_mean={:.1}\n",
                summary.planner_id,
                summary.is_privileged_full_info,
                summary.succeeded_count,
                summary.total_runs,
                summary.success_rate * 100.0,
                summary.failed_count,
                summary.timed_out_count,
                summary.budget_exhausted_count,
                summary.steps_dispersion.median,
                summary.steps_dispersion.iqr,
                summary.steps_dispersion.std_dev,
                summary.transitions_dispersion.mean,
            ));
        }
        out.push_str(&format!(
            "Paired (n={}): left_wins={}, right_wins={}, ties={}, both_ok={}, left_only={}, right_only={}, both_fail={}\n",
            self.paired_overall.total_pairs,
            self.paired_overall.left_wins,
            self.paired_overall.right_wins,
            self.paired_overall.ties,
            self.paired_overall.both_succeeded,
            self.paired_overall.left_only_succeeded,
            self.paired_overall.right_only_succeeded,
            self.paired_overall.both_non_success,
        ));
        out
    }
}

/// Ejecuta la comparación científica pareada en modo 100% headless (sin UI ni Bevy).
pub fn run_arena_comparison(
    config: &ArenaComparisonConfig,
) -> Result<ArenaComparisonReport, ArenaError> {
    config.validate()?;

    let mut paired_outcomes = Vec::new();
    for (idx, (split, seed)) in config.seeds.iter_tagged().enumerate() {
        let left_run_id = RunId::new(((idx as u64) + 1) * 2 - 1);
        let right_run_id = RunId::new(((idx as u64) + 1) * 2);

        let left = execute_single_run(
            &config.spec_id,
            &config.scenario,
            config.budget,
            &config.left_planner,
            split,
            seed,
            left_run_id,
        )?;
        let right = execute_single_run(
            &config.spec_id,
            &config.scenario,
            config.budget,
            &config.right_planner,
            split,
            seed,
            right_run_id,
        )?;

        let outcome_winner = determine_paired_winner(&left, &right);
        let steps_delta = (left.steps_executed as i64) - (right.steps_executed as i64);
        let transitions_delta =
            (left.transitions_evaluated as i64) - (right.transitions_evaluated as i64);
        let reward_delta = left.total_reward - right.total_reward;

        paired_outcomes.push(PairedSeedOutcome {
            split,
            seed,
            left,
            right,
            steps_delta,
            transitions_delta,
            reward_delta,
            outcome_winner,
        });
    }

    let left_id = config.left_planner.id();
    let right_id = config.right_planner.id();
    let left_priv = config.left_planner.is_privileged_full_info();
    let right_priv = config.right_planner.is_privileged_full_info();

    let all_left: Vec<&PlannerRunResult> = paired_outcomes.iter().map(|p| &p.left).collect();
    let all_right: Vec<&PlannerRunResult> = paired_outcomes.iter().map(|p| &p.right).collect();
    let all_pairs: Vec<&PairedSeedOutcome> = paired_outcomes.iter().collect();

    let dev_pairs: Vec<&PairedSeedOutcome> = paired_outcomes
        .iter()
        .filter(|p| p.split == SeedSplit::Dev)
        .collect();
    let holdout_pairs: Vec<&PairedSeedOutcome> = paired_outcomes
        .iter()
        .filter(|p| p.split == SeedSplit::Holdout)
        .collect();

    let left_dev = (!dev_pairs.is_empty()).then(|| {
        let runs: Vec<&PlannerRunResult> = dev_pairs.iter().map(|p| &p.left).collect();
        summarize_planner_runs(&left_id, left_priv, Some(SeedSplit::Dev), &runs)
    });
    let right_dev = (!dev_pairs.is_empty()).then(|| {
        let runs: Vec<&PlannerRunResult> = dev_pairs.iter().map(|p| &p.right).collect();
        summarize_planner_runs(&right_id, right_priv, Some(SeedSplit::Dev), &runs)
    });
    let left_holdout = (!holdout_pairs.is_empty()).then(|| {
        let runs: Vec<&PlannerRunResult> = holdout_pairs.iter().map(|p| &p.left).collect();
        summarize_planner_runs(&left_id, left_priv, Some(SeedSplit::Holdout), &runs)
    });
    let right_holdout = (!holdout_pairs.is_empty()).then(|| {
        let runs: Vec<&PlannerRunResult> = holdout_pairs.iter().map(|p| &p.right).collect();
        summarize_planner_runs(&right_id, right_priv, Some(SeedSplit::Holdout), &runs)
    });

    Ok(ArenaComparisonReport {
        spec_id: config.spec_id.clone(),
        scenario_label: config.scenario.label().to_owned(),
        observation_contract: config.scenario.observation_contract_label(),
        budget: config.budget,
        seeds: config.seeds.clone(),
        left_planner_id: left_id.clone(),
        right_planner_id: right_id.clone(),
        left_overall: summarize_planner_runs(&left_id, left_priv, None, &all_left),
        right_overall: summarize_planner_runs(&right_id, right_priv, None, &all_right),
        left_dev,
        right_dev,
        left_holdout,
        right_holdout,
        paired_overall: summarize_paired_outcomes(None, &all_pairs),
        paired_dev: (!dev_pairs.is_empty())
            .then(|| summarize_paired_outcomes(Some(SeedSplit::Dev), &dev_pairs)),
        paired_holdout: (!holdout_pairs.is_empty())
            .then(|| summarize_paired_outcomes(Some(SeedSplit::Holdout), &holdout_pairs)),
        paired_outcomes,
    })
}

fn determine_paired_winner(left: &PlannerRunResult, right: &PlannerRunResult) -> PairedWinner {
    match (left.succeeded, right.succeeded) {
        (true, false) => return PairedWinner::Left,
        (false, true) => return PairedWinner::Right,
        (true, true) => {
            if left.steps_executed < right.steps_executed {
                return PairedWinner::Left;
            }
            if right.steps_executed < left.steps_executed {
                return PairedWinner::Right;
            }
        }
        (false, false) => {}
    }

    if (left.total_reward - right.total_reward).abs() > 1e-9 {
        if left.total_reward > right.total_reward {
            PairedWinner::Left
        } else {
            PairedWinner::Right
        }
    } else {
        PairedWinner::Tie
    }
}

struct TraceRecorder {
    sink: BufferedTraceSink,
    backend: InMemoryTraceBackend,
    next_revision: u64,
}

impl TraceRecorder {
    fn new(
        run_id: RunId,
        spec_id: &str,
        model_id: &str,
        seed: MasterSeed,
    ) -> Result<Self, ArenaError> {
        let cfg = TraceManifestConfig::new(run_id, spec_id, model_id, seed);
        let sink = BufferedTraceSink::new(
            cfg,
            TraceQuota::default(),
            OverflowPolicy::MarkIncomplete,
            SensitiveFieldPolicy::default(),
        )?;
        Ok(Self {
            sink,
            backend: InMemoryTraceBackend::new(),
            next_revision: 1,
        })
    }

    fn record_step(
        &mut self,
        tick: Tick,
        observation: gogamen_core::Observation,
        plan_status: PlanStatus,
        action: Action,
        used_fallback: bool,
        stream_seeds: StreamSeeds,
    ) -> Result<(), ArenaError> {
        let revision = StreamRevision::new(self.next_revision);
        self.next_revision += 1;
        let guard_decision = if used_fallback {
            RecordedGuardDecision::Fallback(action.clone())
        } else {
            RecordedGuardDecision::Allow(action.clone())
        };
        let record = ScientificRecord {
            revision,
            inputs: TickInputs::new(tick, observation),
            decision: DecisionAction {
                plan_status,
                proposed_action: Some(action.clone()),
                guard_decision,
                applied_action: Some(action),
                used_fallback,
            },
            stream_seeds,
            checkpoint: None,
        };
        self.sink.record(&record)?;
        if self.sink.queued_records() >= 256 {
            self.sink.flush_to(&mut self.backend)?;
        }
        Ok(())
    }

    fn finalize(mut self) -> Result<(TraceManifest, ScientificTraceArchive), ArenaError> {
        let manifest = self.sink.finalize(&mut self.backend)?;
        let decoded_manifest = self
            .backend
            .decode_manifest()?
            .unwrap_or_else(|| manifest.clone());
        let records = self.backend.decode_records()?;
        let archive = ScientificTraceArchive {
            manifest: decoded_manifest.clone(),
            records,
        };
        Ok((decoded_manifest, archive))
    }
}

fn execute_single_run(
    spec_id: &str,
    scenario: &ArenaScenario,
    budget: ArenaBudget,
    planner: &ArenaPlanner,
    split: SeedSplit,
    seed: MasterSeed,
    run_id: RunId,
) -> Result<PlannerRunResult, ArenaError> {
    let model_id = format!("{}/{}", scenario.label(), planner.id());
    let mut recorder = TraceRecorder::new(run_id, spec_id, &model_id, seed)?;
    let started = Instant::now();

    let (status, steps_executed, transitions_evaluated, total_reward) = match scenario {
        ArenaScenario::Grid(grid_cfg) => {
            run_grid_episode(grid_cfg, budget, planner, seed, started, &mut recorder)?
        }
        ArenaScenario::CartPole(cp_cfg) => {
            run_cartpole_episode(cp_cfg, budget, planner, seed, started, &mut recorder)?
        }
    };

    let latency_ns = started.elapsed().as_nanos();
    let (trace_manifest, trace_archive) = recorder.finalize()?;
    let succeeded = status.is_succeeded();
    let budget_utilization_steps = (steps_executed as f64) / (budget.max_ticks as f64);
    let budget_utilization_transitions =
        (transitions_evaluated as f64) / (budget.max_transitions as f64);

    Ok(PlannerRunResult {
        planner_id: planner.id(),
        is_privileged_full_info: planner.is_privileged_full_info(),
        split,
        seed,
        status,
        succeeded,
        steps_executed,
        transitions_evaluated,
        budget_utilization_steps,
        budget_utilization_transitions,
        total_reward,
        latency_ns,
        trace_manifest,
        trace_archive,
    })
}

fn grid_step_reward(prev: &State, _action: &Action, transition: &Transition) -> f64 {
    transition.next_state().values()[2] - prev.values()[2]
}

fn grid_phi(state: &State) -> Vec<f64> {
    vec![state.values()[0], state.values()[1]]
}

fn map_grid_model_error(err: ModelError) -> DomainError {
    match err {
        ModelError::InvalidAction(d) | ModelError::TransitionFailed(d) => d,
        ModelError::TerminalState => DomainError::AlreadyTerminal,
        _ => DomainError::InvalidTransition {
            reason: "grid planning model error in arena",
        },
    }
}

fn map_env_error(err: EnvironmentError) -> DomainError {
    match err {
        EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
        EnvironmentError::InvalidAction(d) => d,
        _ => DomainError::InvalidTransition {
            reason: "environment apply error in arena",
        },
    }
}

fn bfs_shortest_path(map: &GridMap) -> (Option<Vec<GridAction>>, u64) {
    let start = map.start();
    let goal = map.goal();
    let mut queue = VecDeque::new();
    let mut visited = BTreeSet::new();
    let mut parent: BTreeMap<(usize, usize), ((usize, usize), GridAction)> = BTreeMap::new();
    let mut expansions = 0_u64;

    queue.push_back(start);
    visited.insert((start.x, start.y));

    while let Some(cur) = queue.pop_front() {
        expansions += 1;
        if cur == goal {
            let mut path = Vec::new();
            let mut node = (goal.x, goal.y);
            while node != (start.x, start.y) {
                let (prev, act) = parent[&node];
                path.push(act);
                node = prev;
            }
            path.reverse();
            return (Some(path), expansions);
        }

        for act in [
            GridAction::Up,
            GridAction::Down,
            GridAction::Left,
            GridAction::Right,
        ] {
            let (dx, dy) = act.delta();
            let nx = cur.x as isize + dx;
            let ny = cur.y as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let next = GridPos::new(nx as usize, ny as usize);
            if map.is_passable(next) && visited.insert((next.x, next.y)) {
                parent.insert((next.x, next.y), ((cur.x, cur.y), act));
                queue.push_back(next);
            }
        }
    }

    (None, expansions)
}

fn run_grid_episode(
    grid_cfg: &GridArenaConfig,
    budget: ArenaBudget,
    planner: &ArenaPlanner,
    seed: MasterSeed,
    started: Instant,
    recorder: &mut TraceRecorder,
) -> Result<(RunOutcomeStatus, u64, u64, f64), ArenaError> {
    let mut world_config = grid_cfg.world_config.clone();
    world_config.map = grid_cfg.map.clone();
    let mut env = GridWorldEnv::new(world_config.clone())?;
    let action_space = Actuator::action_space(&env).clone();
    let fallback_action = GridAction::Stay.to_action(&action_space)?;

    let mut ticks_executed = 0_u64;
    let mut transitions_evaluated = 0_u64;
    let mut wall_timed_out = false;
    let mut no_path = false;
    let mut last_plan_budget_exhausted = false;

    let agent_stream = StreamKey::agent(0, 0);
    let mut rng = seed.derive(agent_stream);
    let agent_seed_u64 = seed.derive(agent_stream).next_u64();
    let stream_seeds = StreamSeeds::new(seed).with_derived("agent_0_0", agent_seed_u64);

    let mut model = match planner {
        ArenaPlanner::GridMonteCarlo { .. } | ArenaPlanner::GridFmc { .. } => {
            Some(GridWorldModel::new(world_config.clone(), seed)?)
        }
        _ => None,
    };

    let mut mc_planner = if let ArenaPlanner::GridMonteCarlo { horizon, rollouts } = planner {
        Some(MonteCarloPlanner::new(MonteCarloConfig::new(
            *horizon,
            *rollouts,
            budget.max_transitions_per_tick,
            GridAction::Stay,
        )))
    } else {
        None
    };

    let fmc_planner = if let ArenaPlanner::GridFmc {
        population_size,
        horizon,
        scoring,
    } = planner
    {
        let cfg =
            FmcPlannerConfig::new(*population_size, *horizon, world_config.dt, *scoring, seed)
                .map_err(|_| DomainError::InvalidTransition {
                    reason: "invalid FmcPlannerConfig in arena",
                })?;
        Some(FmcPlanner::new(cfg))
    } else {
        None
    };

    let bfs_path = if matches!(planner, ArenaPlanner::GridPrivilegedBfs) {
        let (path, expansions) = bfs_shortest_path(&grid_cfg.map);
        transitions_evaluated = expansions.min(budget.max_transitions);
        if expansions > budget.max_transitions {
            last_plan_budget_exhausted = true;
        }
        if path.is_none() {
            no_path = true;
        }
        path
    } else {
        None
    };

    if !no_path && !last_plan_budget_exhausted {
        let cancel = CancelToken::new();
        while ticks_executed < budget.max_ticks && transitions_evaluated < budget.max_transitions {
            if let Some(limit) = budget.per_episode_timeout
                && started.elapsed() >= limit
            {
                wall_timed_out = true;
                break;
            }

            let obs = EnvironmentObserver::observe(&env).map_err(|_| {
                ArenaError::Domain(DomainError::InvalidTransition {
                    reason: "failed to observe GridWorldEnv in arena",
                })
            })?;
            let tick = Tick::new(ticks_executed);

            let (action, plan_status, used_fallback, sim_transitions) = match planner {
                ArenaPlanner::GridRandomSeeded => {
                    const GRID_ACTIONS: [GridAction; 5] = [
                        GridAction::Up,
                        GridAction::Down,
                        GridAction::Left,
                        GridAction::Right,
                        GridAction::Stay,
                    ];
                    let idx = (rng.next_u64() as usize) % GRID_ACTIONS.len();
                    let act = GRID_ACTIONS[idx].to_action(&action_space)?;
                    (act, PlanStatus::Completed, false, 0_u64)
                }
                ArenaPlanner::GridPrivilegedBfs => {
                    let Some(path) = &bfs_path else {
                        no_path = true;
                        break;
                    };
                    let idx = ticks_executed as usize;
                    if idx >= path.len() {
                        break;
                    }
                    let act = path[idx].to_action(&action_space)?;
                    (act, PlanStatus::Completed, false, 0_u64)
                }
                ArenaPlanner::GridMonteCarlo { .. } => {
                    let m = model
                        .as_ref()
                        .expect("model initialized for GridMonteCarlo");
                    let mc = mc_planner
                        .as_mut()
                        .expect("mc_planner initialized for GridMonteCarlo");
                    let remaining_global =
                        (budget.max_transitions - transitions_evaluated) as usize;
                    let step_budget = budget
                        .max_transitions_per_tick
                        .min(remaining_global.saturating_sub(1));
                    let step_res = mc
                        .plan_grid(
                            m,
                            &mut rng,
                            PlanBudget::with_transitions(step_budget),
                            &cancel,
                        )
                        .map_err(map_grid_model_error)?;
                    let act = step_res
                        .plan_result
                        .authorized_action()
                        .cloned()
                        .unwrap_or_else(|| fallback_action.clone());
                    (
                        act,
                        step_res.plan_result.status(),
                        step_res.used_fallback,
                        step_res.transitions_evaluated as u64,
                    )
                }
                ArenaPlanner::GridFmc { .. } => {
                    let m = model.as_ref().expect("model initialized for GridFmc");
                    let fmc = fmc_planner
                        .as_ref()
                        .expect("fmc_planner initialized for GridFmc");
                    let remaining_global =
                        (budget.max_transitions - transitions_evaluated) as usize;
                    let step_budget = budget
                        .max_transitions_per_tick
                        .min(remaining_global.saturating_sub(1));
                    let request = PlanRequest::from_state(
                        m.state().clone(),
                        m.action_space().clone(),
                        PlanBudget::with_transitions(step_budget),
                    )
                    .with_fallback(fallback_action.clone())?;
                    let mut ctx = PlanningContext::new(m, &mut rng);
                    let (result, trace) = fmc
                        .plan_with_trace(&mut ctx, &request, grid_step_reward, grid_phi)
                        .map_err(map_grid_model_error)?;
                    let act = trace
                        .first_action()
                        .cloned()
                        .unwrap_or_else(|| fallback_action.clone());
                    let used_fb =
                        trace.first_action().is_none() || result.status() != PlanStatus::Completed;
                    (
                        act,
                        result.status(),
                        used_fb,
                        trace.rollout_transitions_used() as u64,
                    )
                }
                ArenaPlanner::CartPoleLinearFeedback(_)
                | ArenaPlanner::CartPoleEntropicProxy(_) => {
                    unreachable!("validated scenario compatibility")
                }
            };

            transitions_evaluated += sim_transitions;
            if transitions_evaluated >= budget.max_transitions {
                last_plan_budget_exhausted = true;
                break;
            }

            if plan_status == PlanStatus::BudgetExhausted {
                last_plan_budget_exhausted = true;
            }

            let outcome = env.apply(&action).map_err(map_env_error)?;
            if let Some(m) = model.as_mut() {
                m.step_in_place(&action).map_err(map_grid_model_error)?;
            }

            recorder.record_step(
                tick,
                obs,
                plan_status,
                action,
                used_fallback,
                stream_seeds.clone(),
            )?;

            ticks_executed += 1;
            transitions_evaluated += 1;
            if outcome.terminal().is_terminal() {
                break;
            }
        }
    }

    let status = classify_grid_outcome(
        env.terminal(),
        no_path,
        wall_timed_out,
        last_plan_budget_exhausted,
        ticks_executed,
        transitions_evaluated,
        budget,
    );

    Ok((
        status,
        ticks_executed,
        transitions_evaluated,
        env.total_reward(),
    ))
}

fn classify_grid_outcome(
    terminal: Terminal,
    no_path: bool,
    wall_timed_out: bool,
    last_plan_budget_exhausted: bool,
    ticks_executed: u64,
    transitions_evaluated: u64,
    budget: ArenaBudget,
) -> RunOutcomeStatus {
    if terminal == Terminal::Terminated(TerminalReason::GoalReached) {
        return RunOutcomeStatus::Succeeded;
    }
    if no_path {
        return RunOutcomeStatus::Failed {
            reason: "no_path_to_goal".to_owned(),
        };
    }
    match terminal {
        Terminal::Terminated(TerminalReason::Failure) => RunOutcomeStatus::Failed {
            reason: "terminated_failure_or_hazard".to_owned(),
        },
        Terminal::Terminated(TerminalReason::InvariantViolated) => RunOutcomeStatus::Failed {
            reason: "invariant_violated".to_owned(),
        },
        _ if wall_timed_out => RunOutcomeStatus::TimedOut {
            reason: "per_episode_timeout_exceeded".to_owned(),
        },
        _ if transitions_evaluated >= budget.max_transitions || last_plan_budget_exhausted => {
            RunOutcomeStatus::BudgetExhausted {
                reason: "max_transitions_exhausted_without_goal".to_owned(),
            }
        }
        _ if ticks_executed >= budget.max_ticks => RunOutcomeStatus::TimedOut {
            reason: "max_ticks_exhausted_without_goal".to_owned(),
        },
        _ => RunOutcomeStatus::Failed {
            reason: "stopped_without_goal".to_owned(),
        },
    }
}

fn run_cartpole_episode(
    cp_cfg: &CartPoleArenaConfig,
    budget: ArenaBudget,
    planner: &ArenaPlanner,
    seed: MasterSeed,
    started: Instant,
    recorder: &mut TraceRecorder,
) -> Result<(RunOutcomeStatus, u64, u64, f64), ArenaError> {
    let mut env = CartPoleEnv::new(cp_cfg.world_config.clone())?;
    let action_space = Actuator::action_space(&env).clone();
    let stream_seeds = StreamSeeds::new(seed).with_derived(
        "cartpole_controller",
        seed.derive(StreamKey::agent(0, 0)).next_u64(),
    );

    let mut linear_ctrl = match planner {
        ArenaPlanner::CartPoleLinearFeedback(c) => Some(*c),
        _ => None,
    };
    let mut entropic_ctrl = match planner {
        ArenaPlanner::CartPoleEntropicProxy(cfg) => {
            Some(EntropicProxyController::new(cfg.as_ref().clone(), seed)?)
        }
        _ => None,
    };

    let mut ticks_executed = 0_u64;
    let mut transitions_evaluated = 0_u64;
    let mut total_reward = 0.0_f64;
    let mut wall_timed_out = false;

    while ticks_executed < budget.max_ticks && transitions_evaluated < budget.max_transitions {
        if let Some(limit) = budget.per_episode_timeout
            && started.elapsed() >= limit
        {
            wall_timed_out = true;
            break;
        }

        let obs = EnvironmentObserver::observe(&env).map_err(|_| {
            ArenaError::Domain(DomainError::InvalidTransition {
                reason: "failed to observe CartPoleEnv in arena",
            })
        })?;
        let tick = Tick::new(ticks_executed);

        let action = if let Some(ctrl) = linear_ctrl.as_mut() {
            ctrl.select_action(&obs, &action_space)?
        } else if let Some(ctrl) = entropic_ctrl.as_mut() {
            ctrl.select_action(&obs, &action_space)?
        } else {
            unreachable!("validated scenario compatibility")
        };

        let outcome = env.apply(&action).map_err(map_env_error)?;
        if !outcome.terminal().is_terminal() {
            total_reward += 1.0;
        }

        recorder.record_step(
            tick,
            obs,
            PlanStatus::Completed,
            action,
            false,
            stream_seeds.clone(),
        )?;

        ticks_executed += 1;
        transitions_evaluated += 1;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let status = match env.terminal() {
        Terminal::Terminated(TerminalReason::Failure) => RunOutcomeStatus::Failed {
            reason: "cartpole_terminal_failure".to_owned(),
        },
        Terminal::Terminated(TerminalReason::InvariantViolated) => RunOutcomeStatus::Failed {
            reason: "cartpole_invariant_violated".to_owned(),
        },
        _ if wall_timed_out => RunOutcomeStatus::TimedOut {
            reason: "per_episode_timeout_exceeded".to_owned(),
        },
        _ if ticks_executed >= budget.max_ticks => RunOutcomeStatus::Succeeded,
        _ if transitions_evaluated >= budget.max_transitions => RunOutcomeStatus::BudgetExhausted {
            reason: "max_transitions_exhausted_before_horizon".to_owned(),
        },
        _ => RunOutcomeStatus::Succeeded,
    };

    Ok((status, ticks_executed, transitions_evaluated, total_reward))
}
