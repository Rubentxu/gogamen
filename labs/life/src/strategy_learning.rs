//! Comparación reproducible de estrategias de aprendizaje, ablación causal del canal
//! y verificación de invariantes de seguridad para `gogamen-lab-life` (`std`-only).
//!
//! Implementa sobre una tarea acotada de forrajeo en [`LifeWorld`]:
//! - Separación estricta entre `train_seeds` y `holdout_seeds` disjuntos.
//! - Registro explícito de semillas y linajes ([`LineageId`], [`StrategyLineageRecord`]).
//! - Comparación bajo el mismo presupuesto entre:
//!   - [`StrategyKind::GeneticInheritance`] (herencia y mutación acotada de [`Genome`] + reflejo genético).
//!   - [`StrategyKind::AssociativeLearning`] (aprendizaje asociativo con [`TabularTransitionLearner`]).
//!   - [`StrategyKind::FmcTrajectoryDistillation`] (trayectorias de [`HomeostaticFmcAgent`] destiladas a [`TabularPolicy`] y promovidas en [`ProceduralMemory`]).
//! - Ablación causal del canal comunicativo (`Intact` vs `Permuted` vs `Silenced`) con
//!   coste explícito ([`CommCost`]), aceptando y explicando resultados neutros o negativos.
//! - Inmutabilidad estricta de [`ActionGuard`] frente a recompensas/aprendizaje y nota
//!   explícita de no-consciencia ni lenguaje composicional ([`SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE`]).

use std::collections::BTreeSet;
use std::fmt;
use std::time::{Duration as StdDuration, Instant};

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, CommCost, CommunicationError,
    DeterministicRng, EvaluationCriteria, EvaluationEvidence, EvidenceRef, GuardContext,
    GuardDecision, LearningError, LearningPolicy, MasterSeed, ObservationId, ProceduralError,
    ProceduralMemory, RngSource, SafeFallbackPolicy, StrategyId, StrategyStats, StrategyVersion,
    StreamKey, TabularPolicy, TabularTransitionLearner, Terminal, Tick, TraceHasher,
    TransitionObservation,
};

use crate::adaptive_agent::{HomeostaticFmcAgent, LOCAL_LIFE_ACTIONS, LifeAgentError};
use crate::communication::{ForageCue, SignalCodebook, Symbol};
use crate::config::{LifeError, LifeWorldConfig};
use crate::genetics::Genome;
use crate::organism::LifeAction;
use crate::world::LifeWorld;

/// Versión de esquema del reporte de aprendizaje de estrategias.
pub const STRATEGY_LEARNING_SCHEMA_VERSION: &str = "gogamen.life.strategy_learning.v1";

/// Nota epistémica y de seguridad obligatoria:
/// el aprendizaje y la recompensa nunca modifican políticas de seguridad ni implican consciencia.
pub const SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE: &str = "Reward and strategy learning never modify ActionGuard safety policies or forbidden action rules; no consciousness or compositional language is claimed.";

const NUM_SYMBOL_BUCKETS: usize = 5; // A, B, C, D, Silenced
const NUM_ENERGY_BUCKETS: usize = 3; // Low, Medium, High
const NUM_STATE_BUCKETS: usize = NUM_SYMBOL_BUCKETS * NUM_ENERGY_BUCKETS; // 15
const NUM_ACTIONS: usize = 5; // Stay, MoveUp, MoveDown, MoveLeft, MoveRight

/// Identificador único determinista de un linaje de estrategia.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineageId(pub u64);

impl LineageId {
    /// Crea un [`LineageId`] a partir de su identificador numérico.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LineageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "lineage-{}", self.0)
    }
}

/// Familia de estrategia comparada en evaluación `holdout`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StrategyKind {
    /// Herencia genética de genoma y reflejos seleccionados sobre `train_seeds`.
    GeneticInheritance,
    /// Aprendizaje asociativo tabular sobre `train_seeds` mediante [`TabularTransitionLearner`].
    AssociativeLearning,
    /// Destilación de trayectorias de [`HomeostaticFmcAgent`] a [`TabularPolicy`] en [`ProceduralMemory`].
    FmcTrajectoryDistillation,
}

impl StrategyKind {
    /// Todas las estrategias comparadas en orden canónico.
    pub const ALL: [Self; 3] = [
        Self::GeneticInheritance,
        Self::AssociativeLearning,
        Self::FmcTrajectoryDistillation,
    ];

    /// Identificador textual canónico.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::GeneticInheritance => "genetic_inheritance",
            Self::AssociativeLearning => "associative_learning",
            Self::FmcTrajectoryDistillation => "fmc_trajectory_distillation",
        }
    }
}

impl fmt::Display for StrategyKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Modo de ablación causal aplicado sobre el canal de comunicación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ChannelAblationKind {
    /// Canal intacto (sujeto únicamente al ruido y coste configurados).
    Intact,
    /// Canal permutado: conserva la distribución marginal de símbolos pero rompe el vínculo causal.
    Permuted,
    /// Canal silenciado: no se transmiten símbolos ni se paga coste de comunicación.
    Silenced,
}

impl ChannelAblationKind {
    /// Identificador textual canónico.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Intact => "intact",
            Self::Permuted => "permuted",
            Self::Silenced => "silenced",
        }
    }
}

impl fmt::Display for ChannelAblationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Clasificación explícita del efecto causal del canal tras descontar coste y ruido.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelAblationVerdict {
    /// El canal intacto supera tanto al canal permutado como al silenciado en utilidad neta.
    PositiveAdvantage,
    /// La diferencia neta es nula o indistinguible dentro de tolerancia numérica.
    NeutralOrNull,
    /// El coste comunicativo o el ruido anulan la ventaja, produciendo utilidad neta negativa.
    NegativeDueToCostOrNoise,
}

/// Errores explícitos del módulo de aprendizaje y comparación de estrategias.
#[derive(Debug, Clone, PartialEq)]
pub enum StrategyLearningError {
    /// Parámetro de configuración inválido.
    InvalidConfig(&'static str),
    /// Se detectó al menos una semilla presente simultáneamente en `train_seeds` y `holdout_seeds`.
    OverlappingTrainHoldoutSeeds {
        /// Valor de semilla solapada.
        seed: u64,
    },
    /// Error del entorno [`LifeWorld`].
    Life(LifeError),
    /// Error del agente homeostático FMC.
    Agent(LifeAgentError),
    /// Error del aprendiz tabular.
    Learning(LearningError),
    /// Error de memoria procedimental.
    Procedural(ProceduralError),
    /// Error de comunicación en `gogamen-core`.
    Communication(CommunicationError),
}

impl fmt::Display for StrategyLearningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid strategy learning config: {msg}"),
            Self::OverlappingTrainHoldoutSeeds { seed } => {
                write!(
                    f,
                    "train_seeds and holdout_seeds must be disjoint (overlap: {seed})"
                )
            }
            Self::Life(err) => write!(f, "life error: {err}"),
            Self::Agent(err) => write!(f, "agent error: {err}"),
            Self::Learning(err) => write!(f, "learning error: {err}"),
            Self::Procedural(err) => write!(f, "procedural memory error: {err}"),
            Self::Communication(err) => write!(f, "communication error: {err}"),
        }
    }
}

impl std::error::Error for StrategyLearningError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConfig(_) | Self::OverlappingTrainHoldoutSeeds { .. } => None,
            Self::Life(err) => Some(err),
            Self::Agent(err) => Some(err),
            Self::Learning(err) => Some(err),
            Self::Procedural(err) => Some(err),
            Self::Communication(err) => Some(err),
        }
    }
}

impl From<LifeError> for StrategyLearningError {
    fn from(value: LifeError) -> Self {
        Self::Life(value)
    }
}

impl From<LifeAgentError> for StrategyLearningError {
    fn from(value: LifeAgentError) -> Self {
        Self::Agent(value)
    }
}

impl From<LearningError> for StrategyLearningError {
    fn from(value: LearningError) -> Self {
        Self::Learning(value)
    }
}

impl From<ProceduralError> for StrategyLearningError {
    fn from(value: ProceduralError) -> Self {
        Self::Procedural(value)
    }
}

impl From<CommunicationError> for StrategyLearningError {
    fn from(value: CommunicationError) -> Self {
        Self::Communication(value)
    }
}

/// Presupuesto común bajo el cual se entrenan y evalúan las estrategias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyBudget {
    /// Ticks por semilla durante entrenamiento.
    pub train_ticks_per_seed: u64,
    /// Ticks por semilla durante evaluación `holdout`.
    pub holdout_ticks_per_seed: u64,
    /// Máximo de mensajes comunicativos permitidos por episodio.
    pub max_comm_messages_per_episode: usize,
    /// Tamaño de población de trayectorias del planificador FMC.
    pub fmc_population_size: usize,
    /// Horizonte de planificación FMC.
    pub fmc_horizon: usize,
    /// Presupuesto máximo de transiciones simuladas por decisión FMC.
    pub fmc_max_transitions: usize,
}

impl Default for StrategyBudget {
    fn default() -> Self {
        Self {
            train_ticks_per_seed: 36,
            holdout_ticks_per_seed: 36,
            max_comm_messages_per_episode: 36,
            fmc_population_size: 6,
            fmc_horizon: 3,
            fmc_max_transitions: 24,
        }
    }
}

impl StrategyBudget {
    /// Valida que todos los límites del presupuesto sean estrictamente positivos.
    pub fn validate(&self) -> Result<(), StrategyLearningError> {
        if self.train_ticks_per_seed == 0 {
            return Err(StrategyLearningError::InvalidConfig(
                "train_ticks_per_seed must be >= 1",
            ));
        }
        if self.holdout_ticks_per_seed == 0 {
            return Err(StrategyLearningError::InvalidConfig(
                "holdout_ticks_per_seed must be >= 1",
            ));
        }
        if self.fmc_population_size == 0 || self.fmc_horizon == 0 || self.fmc_max_transitions == 0 {
            return Err(StrategyLearningError::InvalidConfig(
                "fmc budget parameters must be >= 1",
            ));
        }
        Ok(())
    }
}

/// Política de seguridad inmutable aplicada mediante [`ActionGuard`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SafetyPolicySpec {
    /// Acción discreta permanentemente prohibida por seguridad (si `Some`).
    pub forbidden_action: Option<LifeAction>,
    /// Acción de respaldo segura cuando una propuesta es rechazada.
    pub fallback_action: LifeAction,
}

impl Default for SafetyPolicySpec {
    fn default() -> Self {
        Self {
            forbidden_action: None,
            fallback_action: LifeAction::Stay,
        }
    }
}

impl SafetyPolicySpec {
    /// Calcula una huella determinista de la política de seguridad.
    pub fn fingerprint(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.life.safety_policy");
        match self.forbidden_action {
            Some(act) => {
                hasher.write_u8(1);
                hasher.write_u64(action_to_index(act) as u64);
            }
            None => hasher.write_u8(0),
        }
        hasher.write_u64(action_to_index(self.fallback_action) as u64);
        hasher.finish()
    }
}

/// Configuración completa del experimento comparativo de estrategias.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyComparisonConfig {
    /// Identificador del experimento.
    pub experiment_id: String,
    /// Configuración física base de [`LifeWorld`].
    pub world_config: LifeWorldConfig,
    /// Semillas de entrenamiento (disjuntas de `holdout_seeds`).
    pub train_seeds: Vec<MasterSeed>,
    /// Semillas de evaluación `holdout` (disjuntas de `train_seeds`).
    pub holdout_seeds: Vec<MasterSeed>,
    /// Presupuesto compartido para las 3 estrategias.
    pub budget: StrategyBudget,
    /// Coste explícito [`CommCost`] por mensaje enviado.
    pub comm_cost_per_message: CommCost,
    /// Coste energético escalar por unidad comunicativa.
    pub energy_cost_per_unit: f64,
    /// Coste energético escalar por token comunicativo.
    pub energy_cost_per_token: f64,
    /// Probabilidad de ruido en el símbolo del canal (`[0.0, 1.0]`).
    pub channel_noise_prob: f64,
    /// Alimento colocado en la dirección objetivo en cada ronda.
    pub food_reward: f64,
    /// Política de seguridad inmutable aplicada vía [`ActionGuard`].
    pub safety_policy: SafetyPolicySpec,
}

impl Default for StrategyComparisonConfig {
    fn default() -> Self {
        let world_config = LifeWorldConfig {
            width: 7,
            height: 7,
            initial_population: 0,
            max_population: 4,
            initial_energy: 14.0,
            max_energy: 28.0,
            basal_cost: 0.45,
            move_cost: 0.2,
            max_eat_per_tick: 4.0,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            ..LifeWorldConfig::default()
        };
        Self {
            experiment_id: "gog84_strategy_comparison".to_string(),
            world_config,
            train_seeds: vec![
                MasterSeed::new(101),
                MasterSeed::new(102),
                MasterSeed::new(103),
            ],
            holdout_seeds: vec![
                MasterSeed::new(201),
                MasterSeed::new(202),
                MasterSeed::new(203),
                MasterSeed::new(204),
            ],
            budget: StrategyBudget::default(),
            comm_cost_per_message: CommCost::new(1, 1, StdDuration::from_millis(1)),
            energy_cost_per_unit: 0.15,
            energy_cost_per_token: 0.05,
            channel_noise_prob: 0.0,
            food_reward: 2.5,
            safety_policy: SafetyPolicySpec::default(),
        }
    }
}

impl StrategyComparisonConfig {
    /// Valida invariantes de configuración y separación estricta `train`/`holdout`.
    pub fn validate(&self) -> Result<(), StrategyLearningError> {
        if self.experiment_id.trim().is_empty() {
            return Err(StrategyLearningError::InvalidConfig(
                "experiment_id must not be empty",
            ));
        }
        self.world_config.validate()?;
        if self.world_config.width < 3 || self.world_config.height < 3 {
            return Err(StrategyLearningError::InvalidConfig(
                "world_config must be at least 3x3",
            ));
        }
        self.budget.validate()?;
        if self.train_seeds.is_empty() || self.holdout_seeds.is_empty() {
            return Err(StrategyLearningError::InvalidConfig(
                "train_seeds and holdout_seeds must both be non-empty",
            ));
        }
        let mut train_set = BTreeSet::new();
        for seed in &self.train_seeds {
            train_set.insert(seed.as_u64());
        }
        for seed in &self.holdout_seeds {
            if train_set.contains(&seed.as_u64()) {
                return Err(StrategyLearningError::OverlappingTrainHoldoutSeeds {
                    seed: seed.as_u64(),
                });
            }
        }
        if !self.energy_cost_per_unit.is_finite() || self.energy_cost_per_unit < 0.0 {
            return Err(StrategyLearningError::InvalidConfig(
                "energy_cost_per_unit must be finite and >= 0.0",
            ));
        }
        if !self.energy_cost_per_token.is_finite() || self.energy_cost_per_token < 0.0 {
            return Err(StrategyLearningError::InvalidConfig(
                "energy_cost_per_token must be finite and >= 0.0",
            ));
        }
        if !self.channel_noise_prob.is_finite()
            || self.channel_noise_prob < 0.0
            || self.channel_noise_prob > 1.0
        {
            return Err(StrategyLearningError::InvalidConfig(
                "channel_noise_prob must be finite in [0.0, 1.0]",
            ));
        }
        if !self.food_reward.is_finite() || self.food_reward <= 0.0 {
            return Err(StrategyLearningError::InvalidConfig(
                "food_reward must be finite and > 0.0",
            ));
        }
        if self.safety_policy.forbidden_action == Some(self.safety_policy.fallback_action) {
            return Err(StrategyLearningError::InvalidConfig(
                "safety_policy.fallback_action cannot equal forbidden_action",
            ));
        }
        Ok(())
    }

    /// Coste energético escalar por cada mensaje enviado.
    pub fn scalar_comm_cost(&self) -> f64 {
        (self.comm_cost_per_message.units() as f64) * self.energy_cost_per_unit
            + (self.comm_cost_per_message.tokens() as f64) * self.energy_cost_per_token
    }
}

/// Registro trazable de linaje de una estrategia entrenada o heredada.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyLineageRecord {
    /// Identificador único del linaje.
    pub lineage_id: LineageId,
    /// Estrategia asociada.
    pub strategy: StrategyKind,
    /// Linaje progenitor (`None` para linajes fundadores).
    pub parent_lineage_id: Option<LineageId>,
    /// Generación dentro de la genealogía (`0` para fundadores).
    pub generation: u64,
    /// Semilla de origen o semilla inicial de entrenamiento.
    pub origin_seed: MasterSeed,
    /// Genoma asociado (presente en `GeneticInheritance`).
    pub genome: Option<Genome>,
    /// Checksum determinista de los parámetros/política resultantes.
    pub policy_checksum: u64,
    /// Identificador en [`ProceduralMemory`] (para `FmcTrajectoryDistillation`).
    pub procedural_strategy_id: Option<StrategyId>,
    /// Versión promovida en [`ProceduralMemory`] (para `FmcTrajectoryDistillation`).
    pub procedural_version: Option<StrategyVersion>,
}

/// Métricas agregadas de una estrategia sobre todas las semillas `holdout`.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyHoldoutMetrics {
    /// Estrategia evaluada.
    pub strategy: StrategyKind,
    /// Linaje evaluado.
    pub lineage_id: LineageId,
    /// Modo de ablación del canal usado en esta evaluación.
    pub ablation: ChannelAblationKind,
    /// Número de episodios `holdout` ejecutados.
    pub holdout_episodes: usize,
    /// Tasa de supervivencia al final del presupuesto `holdout`.
    pub survival_rate: f64,
    /// Recompensa bruta media (alimento consumido).
    pub mean_gross_reward: f64,
    /// Coste comunicativo medio descontado.
    pub mean_comm_cost: f64,
    /// Utilidad neta media (`mean_gross_reward - mean_comm_cost`).
    pub mean_net_utility: f64,
    /// Ticks sobrevividos de media.
    pub mean_ticks_survived: f64,
    /// Intervenciones del [`ActionGuard`] bloqueando acciones prohibidas.
    pub guard_interventions: usize,
    /// Hash determinista de la traza de evaluación.
    pub trace_hash: u64,
}

/// Resultado de la ablación causal del canal de comunicación para una estrategia.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelAblationResult {
    /// Estrategia evaluada bajo las tres condiciones del canal.
    pub strategy: StrategyKind,
    /// Utilidad neta con canal intacto.
    pub intact_net_utility: f64,
    /// Utilidad neta con canal permutado (misma distribución marginal, sin vínculo causal).
    pub permuted_net_utility: f64,
    /// Utilidad neta con canal silenciado (sin mensajes ni coste comunicativo).
    pub silenced_net_utility: f64,
    /// Recompensa bruta con canal intacto.
    pub intact_gross_reward: f64,
    /// Recompensa bruta con canal permutado.
    pub permuted_gross_reward: f64,
    /// Recompensa bruta con canal silenciado.
    pub silenced_gross_reward: f64,
    /// Coste comunicativo medio con canal intacto.
    pub intact_comm_cost: f64,
    /// Coste comunicativo medio con canal permutado.
    pub permuted_comm_cost: f64,
    /// Ventaja causal neta frente a canal permutado (`intact_net - permuted_net`).
    pub causal_advantage_vs_permuted: f64,
    /// Ventaja causal neta frente a canal silenciado (`intact_net - silenced_net`).
    pub causal_advantage_vs_silenced: f64,
    /// Veredicto explícito que acepta resultados neutros o negativos.
    pub verdict: ChannelAblationVerdict,
    /// Explicación legible del resultado observado.
    pub explanation: String,
}

/// Verificación explícita de inmutabilidad de políticas de seguridad frente al aprendizaje.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SafetyInvariantCheck {
    /// Huella de la política de seguridad antes de entrenar.
    pub fingerprint_before: u64,
    /// Huella de la política de seguridad tras entrenar y evaluar.
    pub fingerprint_after: u64,
    /// `true` si la política permaneció idéntica (`fingerprint_before == fingerprint_after`).
    pub policy_immutable: bool,
    /// Número total de ejecuciones de acción prohibida observadas (siempre debe ser `0`).
    pub forbidden_executions: usize,
    /// Total de propuestas interceptadas y redirigidas al respaldo seguro por [`ActionGuard`].
    pub total_guard_interventions: usize,
}

/// Reporte completo y reproducible de comparación de estrategias y ablación causal.
#[derive(Debug, Clone, PartialEq)]
pub struct StrategyComparisonReport {
    /// Versión de esquema del reporte.
    pub schema_version: &'static str,
    /// Identificador del experimento.
    pub experiment_id: String,
    /// Semillas usadas exclusivamente para entrenamiento.
    pub train_seeds: Vec<MasterSeed>,
    /// Semillas disjuntas usadas exclusivamente para evaluación `holdout`.
    pub holdout_seeds: Vec<MasterSeed>,
    /// Presupuesto compartido de entrenamiento y evaluación.
    pub budget: StrategyBudget,
    /// Registros de linaje de todas las estrategias (incluyendo generaciones genéticas).
    pub lineage_records: Vec<StrategyLineageRecord>,
    /// Métricas `holdout` con canal intacto para las 3 estrategias.
    pub holdout_metrics: Vec<StrategyHoldoutMetrics>,
    /// Resultados de ablación causal del canal (`Intact` vs `Permuted` vs `Silenced`).
    pub channel_ablations: Vec<ChannelAblationResult>,
    /// Verificación explícita de inmutabilidad de [`ActionGuard`].
    pub safety_check: SafetyInvariantCheck,
    /// Nota obligatoria de seguridad inmutable y no-consciencia.
    pub notice: &'static str,
}

impl StrategyComparisonReport {
    /// Devuelve las métricas `holdout` (canal intacto) para una estrategia concreta.
    pub fn metrics_for(&self, strategy: StrategyKind) -> Option<&StrategyHoldoutMetrics> {
        self.holdout_metrics
            .iter()
            .find(|m| m.strategy == strategy && m.ablation == ChannelAblationKind::Intact)
    }

    /// Devuelve el resultado de ablación causal para una estrategia concreta.
    pub fn ablation_for(&self, strategy: StrategyKind) -> Option<&ChannelAblationResult> {
        self.channel_ablations
            .iter()
            .find(|a| a.strategy == strategy)
    }

    /// Renderiza un resumen legible del experimento.
    pub fn render_report(&self) -> String {
        let mut out = format!(
            "=== Strategy Learning & Causal Ablation Report ({}) ===\n\
             train_seeds={:?}, holdout_seeds={:?}\n",
            self.experiment_id,
            self.train_seeds
                .iter()
                .map(|s| s.as_u64())
                .collect::<Vec<_>>(),
            self.holdout_seeds
                .iter()
                .map(|s| s.as_u64())
                .collect::<Vec<_>>(),
        );
        for m in &self.holdout_metrics {
            out.push_str(&format!(
                "  [{}] lineage={} survival={:.3} gross={:.3} comm={:.3} net={:.3} guard_blocks={}\n",
                m.strategy,
                m.lineage_id,
                m.survival_rate,
                m.mean_gross_reward,
                m.mean_comm_cost,
                m.mean_net_utility,
                m.guard_interventions
            ));
        }
        for a in &self.channel_ablations {
            out.push_str(&format!(
                "  [ablation:{}] intact={:.3} permuted={:.3} silenced={:.3} verdict={:?} ({})\n",
                a.strategy,
                a.intact_net_utility,
                a.permuted_net_utility,
                a.silenced_net_utility,
                a.verdict,
                a.explanation
            ));
        }
        out.push_str(&format!(
            "  safety_immutable={} forbidden_executions={} guard_interventions={}\n  notice: {}\n",
            self.safety_check.policy_immutable,
            self.safety_check.forbidden_executions,
            self.safety_check.total_guard_interventions,
            self.notice
        ));
        out
    }
}

#[derive(Debug, Clone)]
struct GeneticIndividual {
    lineage_id: LineageId,
    parent_lineage_id: Option<LineageId>,
    generation: u64,
    origin_seed: MasterSeed,
    genome: Genome,
    /// Acción preferida para cada bucket de símbolo `[A, B, C, D, Silenced]`.
    symbol_action_genes: [usize; NUM_SYMBOL_BUCKETS],
}

impl GeneticIndividual {
    fn checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_str("gogamen.life.genetic_individual");
        hasher.write_u64(self.lineage_id.as_u64());
        self.genome.write_to_hasher(&mut hasher);
        for &act in &self.symbol_action_genes {
            hasher.write_u64(act as u64);
        }
        hasher.finish()
    }
}

enum TrainedPolicy<'a> {
    Genetic(&'a GeneticIndividual),
    Associative(&'a TabularTransitionLearner),
    Distilled(&'a TabularPolicy),
}

impl TrainedPolicy<'_> {
    fn propose_action(&self, state_bucket: usize, symbol_bucket: usize) -> usize {
        match self {
            Self::Genetic(ind) => {
                ind.symbol_action_genes[symbol_bucket.min(NUM_SYMBOL_BUCKETS - 1)]
            }
            Self::Associative(learner) => {
                let mut best_act = 0_usize;
                let mut best_prob = -1.0_f64;
                for action_idx in 0..NUM_ACTIONS {
                    if let Ok(probs) = learner.predict(state_bucket, action_idx) {
                        let hit_prob = probs.get(1).copied().unwrap_or(0.0);
                        if hit_prob > best_prob {
                            best_prob = hit_prob;
                            best_act = action_idx;
                        }
                    }
                }
                best_act
            }
            Self::Distilled(policy) => policy
                .select_action(state_bucket)
                .map_or(0, |(action_idx, _)| action_idx),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct EpisodeOutcome {
    survived: bool,
    gross_reward: f64,
    comm_cost: f64,
    net_utility: f64,
    ticks_survived: u64,
    guard_interventions: usize,
    forbidden_executions: usize,
    trace_hash: u64,
}

/// Ejecuta la comparación reproducible de las 3 estrategias y la ablación causal del canal.
pub fn run_strategy_comparison(
    config: &StrategyComparisonConfig,
) -> Result<StrategyComparisonReport, StrategyLearningError> {
    config.validate()?;

    let safety_before = config.safety_policy.fingerprint();
    let action_space = ActionSpace::discrete(NUM_ACTIONS).map_err(|_| {
        StrategyLearningError::InvalidConfig("failed to build discrete ActionSpace")
    })?;
    let fallback_action = Action::discrete_in_space(
        action_to_index(config.safety_policy.fallback_action),
        &action_space,
    )
    .map_err(|_| StrategyLearningError::InvalidConfig("invalid safety fallback action"))?;

    let forbidden_idx = config.safety_policy.forbidden_action.map(action_to_index);
    let guard = ActionGuard::with_precondition(
        SafeFallbackPolicy::ExplicitFallback(fallback_action),
        move |action: &Action, _ctx: &GuardContext<'_>| -> Result<(), &'static str> {
            if let Some(forb) = forbidden_idx
                && action.as_discrete().map(|d| d.index()) == Some(forb)
            {
                return Err("action_forbidden_by_immutable_safety_policy");
            }
            Ok(())
        },
    );

    let codebook = SignalCodebook::from_seed(MasterSeed::new(0xC0DE_B004));
    let mut lineage_records = Vec::new();
    let mut next_lineage_id = 1_u64;

    // 1. Entrenar estrategia genética (selección + mutación acotada sobre train_seeds).
    let best_genetic = train_genetic_strategy(
        config,
        &codebook,
        &guard,
        &action_space,
        &mut next_lineage_id,
        &mut lineage_records,
    )?;

    // 2. Entrenar estrategia asociativa (TabularTransitionLearner sobre train_seeds).
    let associative_learner = train_associative_strategy(config, &codebook, &guard, &action_space)?;
    let assoc_lineage_id = LineageId::new(next_lineage_id);
    next_lineage_id += 1;
    lineage_records.push(StrategyLineageRecord {
        lineage_id: assoc_lineage_id,
        strategy: StrategyKind::AssociativeLearning,
        parent_lineage_id: None,
        generation: 0,
        origin_seed: config.train_seeds[0],
        genome: None,
        policy_checksum: associative_learner.checksum(),
        procedural_strategy_id: None,
        procedural_version: None,
    });

    // 3. Entrenar estrategia FMC + destilación a ProceduralMemory sobre train_seeds.
    let procedural_memory =
        train_fmc_distillation_strategy(config, &codebook, &guard, &action_space)?;
    let fmc_lineage_id = LineageId::new(next_lineage_id);
    let active_rec = procedural_memory.active_strategy();
    lineage_records.push(StrategyLineageRecord {
        lineage_id: fmc_lineage_id,
        strategy: StrategyKind::FmcTrajectoryDistillation,
        parent_lineage_id: None,
        generation: 0,
        origin_seed: config.train_seeds[0],
        genome: None,
        policy_checksum: procedural_memory.active_policy().checksum(),
        procedural_strategy_id: Some(active_rec.strategy_id()),
        procedural_version: Some(active_rec.version()),
    });

    // 4. Evaluar las 3 estrategias sobre holdout_seeds bajo el mismo presupuesto y ablaciones.
    let mut holdout_metrics = Vec::new();
    let mut channel_ablations = Vec::new();
    let mut total_forbidden_executions = 0_usize;
    let mut total_guard_interventions = 0_usize;

    let strategies_with_lineage = [
        (
            StrategyKind::GeneticInheritance,
            best_genetic.lineage_id,
            best_genetic.genome,
            TrainedPolicy::Genetic(&best_genetic),
        ),
        (
            StrategyKind::AssociativeLearning,
            assoc_lineage_id,
            Genome::default(),
            TrainedPolicy::Associative(&associative_learner),
        ),
        (
            StrategyKind::FmcTrajectoryDistillation,
            fmc_lineage_id,
            Genome::default(),
            TrainedPolicy::Distilled(procedural_memory.active_policy()),
        ),
    ];

    for (strategy, lineage_id, genome, policy) in strategies_with_lineage {
        let intact = evaluate_holdout_ablation(
            config,
            strategy,
            lineage_id,
            genome,
            &policy,
            &codebook,
            &guard,
            &action_space,
            ChannelAblationKind::Intact,
        )?;
        let permuted = evaluate_holdout_ablation(
            config,
            strategy,
            lineage_id,
            genome,
            &policy,
            &codebook,
            &guard,
            &action_space,
            ChannelAblationKind::Permuted,
        )?;
        let silenced = evaluate_holdout_ablation(
            config,
            strategy,
            lineage_id,
            genome,
            &policy,
            &codebook,
            &guard,
            &action_space,
            ChannelAblationKind::Silenced,
        )?;

        total_forbidden_executions += intact.1 + permuted.1 + silenced.1;
        total_guard_interventions += intact.0.guard_interventions
            + permuted.0.guard_interventions
            + silenced.0.guard_interventions;

        let ablation_result = build_ablation_result(strategy, &intact.0, &permuted.0, &silenced.0);
        holdout_metrics.push(intact.0);
        channel_ablations.push(ablation_result);
    }

    let safety_after = config.safety_policy.fingerprint();

    Ok(StrategyComparisonReport {
        schema_version: STRATEGY_LEARNING_SCHEMA_VERSION,
        experiment_id: config.experiment_id.clone(),
        train_seeds: config.train_seeds.clone(),
        holdout_seeds: config.holdout_seeds.clone(),
        budget: config.budget,
        lineage_records,
        holdout_metrics,
        channel_ablations,
        safety_check: SafetyInvariantCheck {
            fingerprint_before: safety_before,
            fingerprint_after: safety_after,
            policy_immutable: safety_before == safety_after,
            forbidden_executions: total_forbidden_executions,
            total_guard_interventions,
        },
        notice: SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE,
    })
}

fn train_genetic_strategy<P: gogamen_core::DomainPrecondition>(
    config: &StrategyComparisonConfig,
    codebook: &SignalCodebook,
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
    next_lineage_id: &mut u64,
    lineage_records: &mut Vec<StrategyLineageRecord>,
) -> Result<GeneticIndividual, StrategyLearningError> {
    let mut rng = DeterministicRng::new(config.train_seeds[0].as_u64() ^ 0x9E37_79B9);
    let pop_size = 4_usize;
    let generations = 3_u64;

    let mut population = Vec::with_capacity(pop_size);
    for idx in 0..pop_size {
        let lid = LineageId::new(*next_lineage_id);
        *next_lineage_id += 1;
        let mut genes = [0_usize; NUM_SYMBOL_BUCKETS];
        for (slot, gene) in genes.iter_mut().enumerate() {
            *gene = if idx == 0 {
                (slot % 4) + 1
            } else {
                (rng.next_u64() as usize) % NUM_ACTIONS
            };
        }
        let ind = GeneticIndividual {
            lineage_id: lid,
            parent_lineage_id: None,
            generation: 0,
            origin_seed: config.train_seeds[idx % config.train_seeds.len()],
            genome: Genome::random(&mut rng),
            symbol_action_genes: genes,
        };
        lineage_records.push(StrategyLineageRecord {
            lineage_id: ind.lineage_id,
            strategy: StrategyKind::GeneticInheritance,
            parent_lineage_id: ind.parent_lineage_id,
            generation: ind.generation,
            origin_seed: ind.origin_seed,
            genome: Some(ind.genome),
            policy_checksum: ind.checksum(),
            procedural_strategy_id: None,
            procedural_version: None,
        });
        population.push(ind);
    }

    for generation in 1..=generations {
        let mut scored = Vec::with_capacity(population.len());
        for ind in &population {
            let mut total_net = 0.0_f64;
            for &seed in &config.train_seeds {
                let outcome = run_episode(
                    config,
                    seed,
                    config.budget.train_ticks_per_seed,
                    ind.genome,
                    &TrainedPolicy::Genetic(ind),
                    codebook,
                    guard,
                    action_space,
                    ChannelAblationKind::Intact,
                )?;
                total_net += outcome.net_utility;
            }
            scored.push((total_net, ind.clone()));
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0));
        let elite = &scored[0].1;

        let mut next_pop = Vec::with_capacity(pop_size);
        next_pop.push(elite.clone());
        for _ in 1..pop_size {
            let lid = LineageId::new(*next_lineage_id);
            *next_lineage_id += 1;
            let mut child_genes = elite.symbol_action_genes;
            let mutate_slot = (rng.next_u64() as usize) % NUM_SYMBOL_BUCKETS;
            child_genes[mutate_slot] = (rng.next_u64() as usize) % NUM_ACTIONS;
            let child = GeneticIndividual {
                lineage_id: lid,
                parent_lineage_id: Some(elite.lineage_id),
                generation,
                origin_seed: elite.origin_seed,
                genome: elite.genome.mutate(0.4, 0.15, &mut rng),
                symbol_action_genes: child_genes,
            };
            lineage_records.push(StrategyLineageRecord {
                lineage_id: child.lineage_id,
                strategy: StrategyKind::GeneticInheritance,
                parent_lineage_id: child.parent_lineage_id,
                generation: child.generation,
                origin_seed: child.origin_seed,
                genome: Some(child.genome),
                policy_checksum: child.checksum(),
                procedural_strategy_id: None,
                procedural_version: None,
            });
            next_pop.push(child);
        }
        population = next_pop;
    }

    let mut best = population[0].clone();
    let mut best_score = f64::NEG_INFINITY;
    for ind in &population {
        let mut total_net = 0.0_f64;
        for &seed in &config.train_seeds {
            let outcome = run_episode(
                config,
                seed,
                config.budget.train_ticks_per_seed,
                ind.genome,
                &TrainedPolicy::Genetic(ind),
                codebook,
                guard,
                action_space,
                ChannelAblationKind::Intact,
            )?;
            total_net += outcome.net_utility;
        }
        if total_net > best_score {
            best_score = total_net;
            best = ind.clone();
        }
    }
    Ok(best)
}

fn train_associative_strategy<P: gogamen_core::DomainPrecondition>(
    config: &StrategyComparisonConfig,
    codebook: &SignalCodebook,
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
) -> Result<TabularTransitionLearner, StrategyLearningError> {
    let mut learner =
        TabularTransitionLearner::new(NUM_STATE_BUCKETS, NUM_ACTIONS, LearningPolicy::new(0.5)?)?;
    let mut obs_seq = 1_u64;

    for &seed in &config.train_seeds {
        let mut world = LifeWorld::empty(config.world_config.clone(), seed)?;
        let cx = config.world_config.width / 2;
        let cy = config.world_config.height / 2;
        world.spawn_organism_with_genome(
            cx,
            cy,
            config.world_config.initial_energy,
            Genome::default(),
        )?;

        let mut episode_rng = seed.derive(StreamKey::episode(10));
        let schedule = build_episode_schedule(
            config,
            seed,
            config.budget.train_ticks_per_seed,
            codebook,
            ChannelAblationKind::Intact,
        );

        for (tick_idx, &(cue, maybe_symbol)) in schedule.iter().enumerate() {
            if !world.privileged_global_view().organisms().alive()[0] {
                break;
            }
            let (ox, oy) = (
                world.privileged_global_view().organisms().pos_x()[0],
                world.privileged_global_view().organisms().pos_y()[0],
            );
            let valid_cue = ensure_in_bounds_cue(
                ox,
                oy,
                config.world_config.width,
                config.world_config.height,
                cue,
            );
            let (tx, ty) = cue_target_cell(ox, oy, valid_cue);
            world.set_cell_food(tx, ty, config.food_reward)?;

            let energy = world.privileged_global_view().organisms().energy()[0];
            let sym_bucket = symbol_to_bucket(maybe_symbol);
            let state_bucket =
                encode_state_bucket(sym_bucket, energy, config.world_config.max_energy);

            // Exploración determinista para cubrir (símbolo, acción) durante entrenamiento.
            let proposed_idx = if (episode_rng.next_f64()) < 0.35 {
                (episode_rng.next_u64() as usize) % NUM_ACTIONS
            } else {
                TrainedPolicy::Associative(&learner).propose_action(state_bucket, sym_bucket)
            };

            let (executed_action, _) = apply_action_guard(
                guard,
                action_space,
                LOCAL_LIFE_ACTIONS[proposed_idx],
                Tick::new(tick_idx as u64),
                config.safety_policy.fallback_action,
            )?;
            let exec_idx = action_to_index(executed_action);

            let report = world.step_tick(&[executed_action])?;
            let outcome_bucket = if report.food_consumed > 0.0 { 1 } else { 0 };

            let transition_obs = TransitionObservation::new(
                ObservationId::new(obs_seq),
                EvidenceRef::new(obs_seq),
                state_bucket,
                exec_idx,
                outcome_bucket,
            );
            obs_seq += 1;
            learner.observe(&transition_obs)?;
        }
    }
    Ok(learner)
}

fn train_fmc_distillation_strategy<P: gogamen_core::DomainPrecondition>(
    config: &StrategyComparisonConfig,
    codebook: &SignalCodebook,
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
) -> Result<ProceduralMemory, StrategyLearningError> {
    let mut sum_returns = [[0.0_f64; NUM_ACTIONS]; NUM_STATE_BUCKETS];
    let mut counts = [[0_u64; NUM_ACTIONS]; NUM_STATE_BUCKETS];

    let mut total_episodes = 0_u64;
    let mut total_return = 0.0_f64;
    let mut total_hits = 0_u64;
    let mut total_steps = 0_u64;

    for &seed in &config.train_seeds {
        let mut world = LifeWorld::empty(config.world_config.clone(), seed)?;
        let cx = config.world_config.width / 2;
        let cy = config.world_config.height / 2;
        world.spawn_organism_with_genome(
            cx,
            cy,
            config.world_config.initial_energy,
            Genome::default(),
        )?;

        let mut fmc_agent = HomeostaticFmcAgent::new(
            &config.world_config,
            config.budget.fmc_population_size,
            config.budget.fmc_horizon,
            config.budget.fmc_max_transitions,
            seed,
        )?;
        let schedule = build_episode_schedule(
            config,
            seed,
            config.budget.train_ticks_per_seed,
            codebook,
            ChannelAblationKind::Intact,
        );
        let mut ep_return = 0.0_f64;
        total_episodes += 1;

        for (tick_idx, &(cue, maybe_symbol)) in schedule.iter().enumerate() {
            if !world.privileged_global_view().organisms().alive()[0] {
                break;
            }
            let (ox, oy) = (
                world.privileged_global_view().organisms().pos_x()[0],
                world.privileged_global_view().organisms().pos_y()[0],
            );
            let valid_cue = ensure_in_bounds_cue(
                ox,
                oy,
                config.world_config.width,
                config.world_config.height,
                cue,
            );
            let (tx, ty) = cue_target_cell(ox, oy, valid_cue);
            world.set_cell_food(tx, ty, config.food_reward)?;

            let obs = world
                .observe_organism(0)
                .ok_or(StrategyLearningError::InvalidConfig(
                    "missing organism observation",
                ))?;
            let decision = fmc_agent.decide_with_metrics(&obs)?;
            let sym_bucket = symbol_to_bucket(maybe_symbol);
            let state_bucket =
                encode_state_bucket(sym_bucket, obs.energy(), config.world_config.max_energy);

            let (executed_action, _) = apply_action_guard(
                guard,
                action_space,
                decision.action,
                Tick::new(tick_idx as u64),
                config.safety_policy.fallback_action,
            )?;
            let exec_idx = action_to_index(executed_action);
            let report = world.step_tick(&[executed_action])?;

            total_steps += 1;
            if report.food_consumed > 0.0 {
                total_hits += 1;
            }
            ep_return += report.food_consumed;
            sum_returns[state_bucket][exec_idx] += report.food_consumed;
            counts[state_bucket][exec_idx] += 1;
        }
        total_return += ep_return;
    }

    let mut distilled_policy = TabularPolicy::new(NUM_STATE_BUCKETS, NUM_ACTIONS)?;
    for s in 0..NUM_STATE_BUCKETS {
        for a in 0..NUM_ACTIONS {
            let mean_val = if counts[s][a] > 0 {
                sum_returns[s][a] / (counts[s][a] as f64)
            } else {
                0.0
            };
            distilled_policy.set_action_value(s, a, mean_val)?;
        }
    }

    let criteria = EvaluationCriteria::new(1, 0.0, 0.0)?;
    let baseline_policy = TabularPolicy::new(NUM_STATE_BUCKETS, NUM_ACTIONS)?;
    let baseline_stats = StrategyStats::new(1, 0.0, 0.0, 0, vec![EvidenceRef::new(1)])?;
    let mut memory = ProceduralMemory::new(
        StrategyId::new(1),
        baseline_policy,
        baseline_stats,
        criteria,
    );

    let mean_ep_return = total_return / (total_episodes.max(1) as f64);
    let success_rate = (total_hits as f64) / (total_steps.max(1) as f64);
    let distilled_stats = StrategyStats::new(
        total_episodes,
        mean_ep_return,
        success_rate.clamp(0.0, 1.0),
        0,
        vec![EvidenceRef::new(2)],
    )?;
    let version = memory.register_candidate(StrategyId::new(2), distilled_policy)?;
    let _ = memory.evaluate_and_promote(
        StrategyId::new(2),
        version,
        EvaluationEvidence::new(distilled_stats).with_fixture_tag("fmc_trajectory_distillation"),
    )?;

    Ok(memory)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_holdout_ablation<P: gogamen_core::DomainPrecondition>(
    config: &StrategyComparisonConfig,
    strategy: StrategyKind,
    lineage_id: LineageId,
    genome: Genome,
    policy: &TrainedPolicy<'_>,
    codebook: &SignalCodebook,
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
    ablation: ChannelAblationKind,
) -> Result<(StrategyHoldoutMetrics, usize), StrategyLearningError> {
    let n = config.holdout_seeds.len() as f64;
    let mut survivors = 0_usize;
    let mut sum_gross = 0.0_f64;
    let mut sum_comm = 0.0_f64;
    let mut sum_net = 0.0_f64;
    let mut sum_ticks = 0_u64;
    let mut total_interventions = 0_usize;
    let mut total_forbidden = 0_usize;

    let mut hasher = TraceHasher::new();
    hasher.write_str("gogamen.life.holdout_ablation");
    hasher.write_str(strategy.as_str());
    hasher.write_str(ablation.as_str());

    for &seed in &config.holdout_seeds {
        let ep = run_episode(
            config,
            seed,
            config.budget.holdout_ticks_per_seed,
            genome,
            policy,
            codebook,
            guard,
            action_space,
            ablation,
        )?;
        if ep.survived {
            survivors += 1;
        }
        sum_gross += ep.gross_reward;
        sum_comm += ep.comm_cost;
        sum_net += ep.net_utility;
        sum_ticks += ep.ticks_survived;
        total_interventions += ep.guard_interventions;
        total_forbidden += ep.forbidden_executions;
        hasher.write_u64(ep.trace_hash);
    }

    let metrics = StrategyHoldoutMetrics {
        strategy,
        lineage_id,
        ablation,
        holdout_episodes: config.holdout_seeds.len(),
        survival_rate: (survivors as f64) / n,
        mean_gross_reward: sum_gross / n,
        mean_comm_cost: sum_comm / n,
        mean_net_utility: sum_net / n,
        mean_ticks_survived: (sum_ticks as f64) / n,
        guard_interventions: total_interventions,
        trace_hash: hasher.finish(),
    };
    Ok((metrics, total_forbidden))
}

#[allow(clippy::too_many_arguments)]
fn run_episode<P: gogamen_core::DomainPrecondition>(
    config: &StrategyComparisonConfig,
    seed: MasterSeed,
    ticks: u64,
    genome: Genome,
    policy: &TrainedPolicy<'_>,
    codebook: &SignalCodebook,
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
    ablation: ChannelAblationKind,
) -> Result<EpisodeOutcome, StrategyLearningError> {
    let mut world = LifeWorld::empty(config.world_config.clone(), seed)?;
    let cx = config.world_config.width / 2;
    let cy = config.world_config.height / 2;
    world.spawn_organism_with_genome(cx, cy, config.world_config.initial_energy, genome)?;

    let schedule = build_episode_schedule(config, seed, ticks, codebook, ablation);
    let step_comm_cost = config.scalar_comm_cost();

    let mut gross_reward = 0.0_f64;
    let mut comm_cost = 0.0_f64;
    let mut ticks_survived = 0_u64;
    let mut guard_interventions = 0_usize;
    let mut forbidden_executions = 0_usize;

    for (tick_idx, &(cue, maybe_symbol)) in schedule.iter().enumerate() {
        if !world.privileged_global_view().organisms().alive()[0] {
            break;
        }
        let (ox, oy) = (
            world.privileged_global_view().organisms().pos_x()[0],
            world.privileged_global_view().organisms().pos_y()[0],
        );
        let valid_cue = ensure_in_bounds_cue(
            ox,
            oy,
            config.world_config.width,
            config.world_config.height,
            cue,
        );
        let (tx, ty) = cue_target_cell(ox, oy, valid_cue);
        world.set_cell_food(tx, ty, config.food_reward)?;

        if maybe_symbol.is_some() {
            comm_cost += step_comm_cost;
        }

        let energy = world.privileged_global_view().organisms().energy()[0];
        let sym_bucket = symbol_to_bucket(maybe_symbol);
        let state_bucket = encode_state_bucket(sym_bucket, energy, config.world_config.max_energy);
        let proposed_idx = policy.propose_action(state_bucket, sym_bucket);

        let (executed_action, intervened) = apply_action_guard(
            guard,
            action_space,
            LOCAL_LIFE_ACTIONS[proposed_idx],
            Tick::new(tick_idx as u64),
            config.safety_policy.fallback_action,
        )?;
        if intervened {
            guard_interventions += 1;
        }
        if config.safety_policy.forbidden_action == Some(executed_action) {
            forbidden_executions += 1;
        }

        let report = world.step_tick(&[executed_action])?;
        gross_reward += report.food_consumed;
        ticks_survived += 1;
    }

    let survived = world.privileged_global_view().organisms().alive()[0];
    Ok(EpisodeOutcome {
        survived,
        gross_reward,
        comm_cost,
        net_utility: gross_reward - comm_cost,
        ticks_survived,
        guard_interventions,
        forbidden_executions,
        trace_hash: world.trace_hash(),
    })
}

fn build_episode_schedule(
    config: &StrategyComparisonConfig,
    seed: MasterSeed,
    ticks: u64,
    codebook: &SignalCodebook,
    ablation: ChannelAblationKind,
) -> Vec<(ForageCue, Option<Symbol>)> {
    let mut rng = seed.derive(StreamKey::episode(1));
    let mut cues = Vec::with_capacity(ticks as usize);
    let mut symbols = Vec::with_capacity(ticks as usize);

    for t in 0..ticks {
        let cue = ForageCue::ALL[(rng.next_u64() as usize) % ForageCue::ALL.len()];
        cues.push(cue);
        if ablation == ChannelAblationKind::Silenced
            || (t as usize) >= config.budget.max_comm_messages_per_episode
        {
            symbols.push(None);
        } else {
            let mut sym = codebook.symbol_for(cue);
            if config.channel_noise_prob > 0.0 && rng.next_f64() < config.channel_noise_prob {
                let offset = 1 + ((rng.next_u64() as usize) % 3);
                sym = Symbol::ALL[(sym.index() + offset) % 4];
            }
            symbols.push(Some(sym));
        }
    }

    if ablation == ChannelAblationKind::Permuted && symbols.len() > 1 {
        let mut perm_rng = seed.derive(StreamKey::episode(99));
        for i in (1..symbols.len()).rev() {
            let j = (perm_rng.next_u64() as usize) % (i + 1);
            symbols.swap(i, j);
        }
    }

    cues.into_iter().zip(symbols).collect()
}

fn build_ablation_result(
    strategy: StrategyKind,
    intact: &StrategyHoldoutMetrics,
    permuted: &StrategyHoldoutMetrics,
    silenced: &StrategyHoldoutMetrics,
) -> ChannelAblationResult {
    let adv_perm = intact.mean_net_utility - permuted.mean_net_utility;
    let adv_sil = intact.mean_net_utility - silenced.mean_net_utility;
    let eps = 1e-6;

    let (verdict, explanation) = if adv_perm > eps && adv_sil > eps {
        (
            ChannelAblationVerdict::PositiveAdvantage,
            format!(
                "Intact channel improves net utility by {adv_perm:.3} vs permuted and {adv_sil:.3} vs silenced."
            ),
        )
    } else if adv_sil < -eps || adv_perm < -eps {
        (
            ChannelAblationVerdict::NegativeDueToCostOrNoise,
            format!(
                "Channel cost or noise cancels advantage (vs_permuted={adv_perm:.3}, vs_silenced={adv_sil:.3})."
            ),
        )
    } else {
        (
            ChannelAblationVerdict::NeutralOrNull,
            format!(
                "No net causal advantage observed (vs_permuted={adv_perm:.3}, vs_silenced={adv_sil:.3})."
            ),
        )
    };

    ChannelAblationResult {
        strategy,
        intact_net_utility: intact.mean_net_utility,
        permuted_net_utility: permuted.mean_net_utility,
        silenced_net_utility: silenced.mean_net_utility,
        intact_gross_reward: intact.mean_gross_reward,
        permuted_gross_reward: permuted.mean_gross_reward,
        silenced_gross_reward: silenced.mean_gross_reward,
        intact_comm_cost: intact.mean_comm_cost,
        permuted_comm_cost: permuted.mean_comm_cost,
        causal_advantage_vs_permuted: adv_perm,
        causal_advantage_vs_silenced: adv_sil,
        verdict,
        explanation,
    }
}

fn apply_action_guard<P: gogamen_core::DomainPrecondition>(
    guard: &ActionGuard<P>,
    action_space: &ActionSpace,
    proposed: LifeAction,
    tick: Tick,
    fallback: LifeAction,
) -> Result<(LifeAction, bool), StrategyLearningError> {
    let action = Action::discrete_in_space(action_to_index(proposed), action_space)
        .map_err(|_| StrategyLearningError::InvalidConfig("failed to construct discrete Action"))?;
    let proposal = ActionProposal::authorized(action, tick);
    let ctx = GuardContext::new(action_space, tick, Instant::now(), Terminal::Ongoing);
    match guard.evaluate(&proposal, &ctx) {
        GuardDecision::Allow(allowed) => {
            let idx = allowed.as_discrete().map_or(0, |d| d.index());
            Ok((LOCAL_LIFE_ACTIONS[idx], false))
        }
        GuardDecision::Fallback(fb) => {
            let idx = fb
                .as_discrete()
                .map_or(action_to_index(fallback), |d| d.index());
            Ok((LOCAL_LIFE_ACTIONS[idx], true))
        }
        GuardDecision::NoOp | GuardDecision::Reject(_) => Ok((fallback, true)),
    }
}

const fn action_to_index(action: LifeAction) -> usize {
    match action {
        LifeAction::Stay | LifeAction::Reproduce => 0,
        LifeAction::MoveUp => 1,
        LifeAction::MoveDown => 2,
        LifeAction::MoveLeft => 3,
        LifeAction::MoveRight => 4,
    }
}

const fn symbol_to_bucket(symbol: Option<Symbol>) -> usize {
    match symbol {
        Some(sym) => sym.index(),
        None => 4,
    }
}

fn encode_state_bucket(symbol_bucket: usize, energy: f64, max_energy: f64) -> usize {
    let ratio = if max_energy > 0.0 {
        (energy / max_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let energy_bucket = if ratio < 0.35 {
        0
    } else if ratio < 0.75 {
        1
    } else {
        2
    };
    symbol_bucket.min(NUM_SYMBOL_BUCKETS - 1) * NUM_ENERGY_BUCKETS + energy_bucket
}

fn ensure_in_bounds_cue(
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    cue: ForageCue,
) -> ForageCue {
    let (dx, dy) = cue_delta(cue);
    let nx = ox as isize + dx;
    let ny = oy as isize + dy;
    if nx >= 0 && nx < width as isize && ny >= 0 && ny < height as isize {
        return cue;
    }
    for candidate in ForageCue::ALL {
        let (cdx, cdy) = cue_delta(candidate);
        let cx = ox as isize + cdx;
        let cy = oy as isize + cdy;
        if cx >= 0 && cx < width as isize && cy >= 0 && cy < height as isize {
            return candidate;
        }
    }
    cue
}

const fn cue_delta(cue: ForageCue) -> (isize, isize) {
    match cue {
        ForageCue::North => (0, -1),
        ForageCue::South => (0, 1),
        ForageCue::East => (1, 0),
        ForageCue::West => (-1, 0),
    }
}

fn cue_target_cell(ox: usize, oy: usize, cue: ForageCue) -> (usize, usize) {
    let (dx, dy) = cue_delta(cue);
    ((ox as isize + dx) as usize, (oy as isize + dy) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_disjoint_train_holdout_lineage_records_and_three_strategy_comparison() {
        let mut overlapping = StrategyComparisonConfig::default();
        overlapping.holdout_seeds.push(overlapping.train_seeds[0]);
        let err = run_strategy_comparison(&overlapping).unwrap_err();
        assert_eq!(
            err,
            StrategyLearningError::OverlappingTrainHoldoutSeeds {
                seed: overlapping.train_seeds[0].as_u64()
            }
        );

        let cfg = StrategyComparisonConfig::default();
        let report_a = run_strategy_comparison(&cfg).unwrap();
        let report_b = run_strategy_comparison(&cfg).unwrap();
        assert_eq!(report_a, report_b);

        assert_eq!(report_a.train_seeds, cfg.train_seeds);
        assert_eq!(report_a.holdout_seeds, cfg.holdout_seeds);
        assert_eq!(report_a.holdout_metrics.len(), 3);
        assert!(
            report_a
                .metrics_for(StrategyKind::GeneticInheritance)
                .is_some()
        );
        assert!(
            report_a
                .metrics_for(StrategyKind::AssociativeLearning)
                .is_some()
        );
        assert!(
            report_a
                .metrics_for(StrategyKind::FmcTrajectoryDistillation)
                .is_some()
        );

        // Verificar registro explícito de linajes incluyendo descendencia genética y memoria procedimental.
        assert!(report_a.lineage_records.iter().any(|r| r.strategy
            == StrategyKind::GeneticInheritance
            && r.parent_lineage_id.is_some()));
        assert!(
            report_a
                .lineage_records
                .iter()
                .any(|r| r.strategy == StrategyKind::FmcTrajectoryDistillation
                    && r.procedural_strategy_id.is_some()
                    && r.procedural_version.is_some())
        );
    }

    #[test]
    fn ac2_causal_channel_ablation_and_negative_cost_noise_case() {
        // Caso 1: canal útil con coste moderado y sin ruido.
        let good_cfg = StrategyComparisonConfig::default();
        let good_report = run_strategy_comparison(&good_cfg).unwrap();
        let assoc_ablation = good_report
            .ablation_for(StrategyKind::AssociativeLearning)
            .unwrap();
        assert!(assoc_ablation.intact_gross_reward >= assoc_ablation.permuted_gross_reward);
        assert!(assoc_ablation.intact_gross_reward >= assoc_ablation.silenced_gross_reward);

        // Caso 2: coste comunicativo excesivo + ruido alto -> resultado negativo explícito y explicado.
        let bad_cfg = StrategyComparisonConfig {
            energy_cost_per_unit: 3.0,
            energy_cost_per_token: 1.0,
            channel_noise_prob: 0.85,
            ..StrategyComparisonConfig::default()
        };
        let bad_report = run_strategy_comparison(&bad_cfg).unwrap();
        let fmc_ablation = bad_report
            .ablation_for(StrategyKind::FmcTrajectoryDistillation)
            .unwrap();
        assert_eq!(
            fmc_ablation.verdict,
            ChannelAblationVerdict::NegativeDueToCostOrNoise
        );
        assert!(fmc_ablation.causal_advantage_vs_silenced < 0.0);
        assert!(
            fmc_ablation.explanation.contains("cost") || fmc_ablation.explanation.contains("noise")
        );
    }

    #[test]
    fn ac3_safety_policy_immutable_under_reward_and_no_consciousness_notice() {
        let cfg = StrategyComparisonConfig {
            safety_policy: SafetyPolicySpec {
                forbidden_action: Some(LifeAction::MoveLeft),
                fallback_action: LifeAction::Stay,
            },
            food_reward: 4.0,
            ..StrategyComparisonConfig::default()
        };

        let report = run_strategy_comparison(&cfg).unwrap();
        assert!(report.safety_check.policy_immutable);
        assert_eq!(
            report.safety_check.fingerprint_before,
            report.safety_check.fingerprint_after
        );
        assert_eq!(report.safety_check.forbidden_executions, 0);
        assert!(report.safety_check.total_guard_interventions > 0);
        assert_eq!(report.notice, SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE);
        assert!(
            report
                .render_report()
                .contains(SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE)
        );
    }
}
