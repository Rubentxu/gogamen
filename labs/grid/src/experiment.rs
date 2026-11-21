//! Experimento de comparación justa en Grid (`std`-only + `gogamen-core`).
//!
//! Evalúa bajo exactamente el mismo mapa, reglas (`GridWorldConfig`), estados
//! iniciales y presupuesto documentado (`BaselineBudget` + `max_transitions_per_tick`)
//! a cuatro algoritmos separados explícitamente por acceso a información:
//! - No privilegiados: `RandomSeeded`, `MonteCarloShooting`, `Fmc` (`FmcPlanner`).
//! - Techo privilegiado (`is_privileged_full_info = true`): `PrivilegedBfs`.
//!
//! Separa explícitamente semillas de desarrollo (`SeedSplit::Dev`) y evaluación
//! ciega (`SeedSplit::Holdout`), registra latencia, transiciones, pasos,
//! recompensa, fallos por seed y agregados estadísticos con análisis honesto.

use std::time::Instant;

use gogamen_core::{
    Action, Actuator, DomainError, FmcPlanner, FmcPlannerConfig, FmcScoringConfig, GenerativeModel,
    MasterSeed, ModelError, PlanBudget, PlanRequest, PlanningContext, State, StreamKey, Terminal,
    TerminalReason, Transition,
};

use crate::baselines::{BaselineBudget, run_bfs_baseline, run_random_baseline};
use crate::map::GridMap;
use crate::monte_carlo::{MonteCarloConfig, run_monte_carlo_shooting_baseline};
use crate::world::{GridAction, GridWorldConfig, GridWorldEnv, GridWorldModel};

/// División explícita de semillas entre desarrollo y holdout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SeedSplit {
    /// Semillas de desarrollo usadas durante ajuste o diagnóstico.
    Dev,
    /// Semillas reservadas de evaluación ciega (holdout).
    Holdout,
}

/// Algoritmo evaluado en el experimento de comparación justa en Grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExperimentAlgorithm {
    /// Política aleatoria uniforme reproducible (`run_random_baseline`).
    RandomSeeded,
    /// Planificador Monte Carlo random shooting (`run_monte_carlo_shooting_baseline`).
    MonteCarloShooting,
    /// Techo privilegiado con acceso completo al grafo del mapa (`run_bfs_baseline`).
    PrivilegedBfs,
    /// Planificador Fractal Monte Carlo recedente (`FmcPlanner` sobre `PlanningContext`).
    Fmc,
}

impl ExperimentAlgorithm {
    /// Todos los algoritmos en orden canónico de evaluación.
    pub const ALL: [Self; 4] = [
        Self::RandomSeeded,
        Self::MonteCarloShooting,
        Self::PrivilegedBfs,
        Self::Fmc,
    ];

    /// Indica si el algoritmo utiliza información privilegiada completa del mapa.
    pub const fn is_privileged_full_info(self) -> bool {
        matches!(self, Self::PrivilegedBfs)
    }

    /// Nombre legible del algoritmo para reportes y trazas.
    pub const fn label(self) -> &'static str {
        match self {
            Self::RandomSeeded => "RandomSeeded",
            Self::MonteCarloShooting => "MonteCarloShooting",
            Self::PrivilegedBfs => "PrivilegedBfs",
            Self::Fmc => "Fmc",
        }
    }
}

/// Configuración explícita y documentada de la comparación justa en Grid.
#[derive(Debug, Clone, PartialEq)]
pub struct FairComparisonConfig {
    /// Mapa discreto compartido por todos los algoritmos.
    pub map: GridMap,
    /// Reglas físicas, recompensas y modo de observación compartidos (`GridWorldConfig`).
    pub world_config: GridWorldConfig,
    /// Presupuesto global por episodio (`max_ticks` y `max_transitions`).
    pub budget: BaselineBudget,
    /// Presupuesto máximo de transiciones simuladas por decisión/tick para planificadores.
    pub max_transitions_per_tick: usize,
    /// Horizonte de simulación para `MonteCarloShooting`.
    pub mc_horizon: usize,
    /// Número de rollouts por decisión para `MonteCarloShooting`.
    pub mc_rollouts: usize,
    /// Tamaño de población de walkers para `FmcPlanner`.
    pub fmc_population_size: usize,
    /// Horizonte recedente para `FmcPlanner`.
    pub fmc_horizon: usize,
    /// Configuración de scoring y resampling para `FmcPlanner`.
    pub fmc_scoring: FmcScoringConfig,
    /// Semillas de desarrollo (`SeedSplit::Dev`).
    pub dev_seeds: Vec<MasterSeed>,
    /// Semillas de evaluación ciega (`SeedSplit::Holdout`).
    pub holdout_seeds: Vec<MasterSeed>,
}

impl Default for FairComparisonConfig {
    fn default() -> Self {
        Self {
            map: GridMap::cul_de_sac_map(),
            world_config: GridWorldConfig::default(),
            budget: BaselineBudget::new(80, 16_000),
            max_transitions_per_tick: 160,
            mc_horizon: 10,
            mc_rollouts: 16,
            fmc_population_size: 16,
            fmc_horizon: 10,
            fmc_scoring: FmcScoringConfig::default(),
            dev_seeds: (1..=5).map(MasterSeed::new).collect(),
            holdout_seeds: (101..=105).map(MasterSeed::new).collect(),
        }
    }
}

/// Resultado individual de ejecutar un algoritmo sobre una semilla y split concretos.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedRunResult {
    /// Algoritmo ejecutado.
    pub algorithm: ExperimentAlgorithm,
    /// División de semilla (`Dev` o `Holdout`).
    pub split: SeedSplit,
    /// Semilla determinista de la ejecución.
    pub seed: MasterSeed,
    /// Marca explícita de acceso privilegiado (`true` solo para `PrivilegedBfs`).
    pub is_privileged_full_info: bool,
    /// Indica si alcanzó `CellKind::Goal`.
    pub success: bool,
    /// Pasos (ticks) ejecutados en `GridWorldEnv`.
    pub steps: u64,
    /// Transiciones totales evaluadas (simulación + entorno).
    pub transitions_evaluated: u64,
    /// Latencia de ejecución medida con `std::time::Instant` en nanosegundos.
    pub latency_ns: u128,
    /// Recompensa acumulada en el episodio.
    pub total_reward: f64,
    /// Indica si la ejecución falló (no alcanzó la meta o terminó en fallo/error).
    pub failed: bool,
    /// Motivo explícito del fallo cuando `failed == true`.
    pub failure_reason: Option<String>,
}

/// Agregado estadístico por algoritmo (global o por `SeedSplit`).
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmAggregate {
    /// Algoritmo agregado.
    pub algorithm: ExperimentAlgorithm,
    /// `None` para agregado global; `Some(split)` para desglose `Dev`/`Holdout`.
    pub split: Option<SeedSplit>,
    /// Marca explícita de si el algoritmo usa información privilegiada completa.
    pub is_privileged_full_info: bool,
    /// Número total de ejecuciones agregadas.
    pub runs_count: usize,
    /// Número de ejecuciones que alcanzaron la meta.
    pub successes_count: usize,
    /// Número de ejecuciones fallidas.
    pub failures_count: usize,
    /// Tasa de éxito en `[0.0, 1.0]`.
    pub success_rate: f64,
    /// Media de pasos ejecutados.
    pub mean_steps: f64,
    /// Mediana de pasos ejecutados.
    pub median_steps: f64,
    /// Rango intercuartílico (`Q75 - Q25`) de pasos ejecutados.
    pub iqr_steps: f64,
    /// Mínimo y máximo de pasos ejecutados (`None` si `runs_count == 0`).
    pub min_max_steps: Option<(u64, u64)>,
    /// Media de transiciones evaluadas.
    pub mean_transitions: f64,
    /// Media de latencia en nanosegundos.
    pub mean_latency_ns: f64,
    /// Media de recompensa acumulada.
    pub mean_total_reward: f64,
    /// Notas honestas sobre cada fallo observado (sin ocultarlos ni maquillarlos).
    pub failure_notes: Vec<String>,
}

/// Informe completo del experimento de comparación justa en Grid.
#[derive(Debug, Clone, PartialEq)]
pub struct FairComparisonReport {
    /// Configuración compartida documentada.
    pub config: FairComparisonConfig,
    /// Resultados individuales por `(algoritmo, split, seed)`.
    pub seed_results: Vec<SeedRunResult>,
    /// Agregados globales por algoritmo (combinando `Dev` y `Holdout`).
    pub aggregates: Vec<AlgorithmAggregate>,
    /// Agregados desglosados por `(algoritmo, split)`.
    pub split_aggregates: Vec<AlgorithmAggregate>,
    /// Notas comparativas honestas (incluyendo análisis de derrotas o fallos de FMC).
    pub analysis_notes: Vec<String>,
}

fn grid_step_reward(prev: &State, _action: &Action, transition: &Transition) -> f64 {
    transition.next_state().values()[2] - prev.values()[2]
}

fn grid_phi(state: &State) -> Vec<f64> {
    vec![state.values()[0], state.values()[1]]
}

fn map_model_error(err: ModelError) -> DomainError {
    match err {
        ModelError::InvalidAction(d) | ModelError::TransitionFailed(d) => d,
        ModelError::TerminalState => DomainError::AlreadyTerminal,
        _ => DomainError::InvalidTransition {
            reason: "fmc grid planning model error",
        },
    }
}

fn run_fmc_episode(
    config: &FairComparisonConfig,
    seed: MasterSeed,
) -> Result<(bool, u64, u64, f64, Option<String>), DomainError> {
    let mut world_config = config.world_config.clone();
    world_config.map = config.map.clone();
    world_config.validate()?;

    let mut env = GridWorldEnv::new(world_config.clone())?;
    let mut model = GridWorldModel::new(world_config.clone(), seed)?;
    let fmc_config = FmcPlannerConfig::new(
        config.fmc_population_size,
        config.fmc_horizon,
        world_config.dt,
        config.fmc_scoring,
        seed,
    )
    .map_err(|_| DomainError::InvalidTransition {
        reason: "invalid FmcPlannerConfig in FairComparisonConfig",
    })?;
    let planner = FmcPlanner::new(fmc_config);
    let mut planning_rng = seed.derive(StreamKey::agent(0, 0));
    let fallback = GridAction::Stay.to_action(model.action_space())?;

    let mut ticks_executed = 0_u64;
    let mut transitions_evaluated = 0_u64;

    while ticks_executed < config.budget.max_ticks
        && transitions_evaluated < config.budget.max_transitions
    {
        let remaining_global = (config.budget.max_transitions - transitions_evaluated) as usize;
        let step_budget_transitions = config
            .max_transitions_per_tick
            .min(remaining_global.saturating_sub(1));

        let request = PlanRequest::from_state(
            model.state().clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(step_budget_transitions),
        )
        .with_fallback(fallback.clone())?;

        let mut ctx = PlanningContext::new(&model, &mut planning_rng);
        let (_result, trace) = planner
            .plan_with_trace(&mut ctx, &request, grid_step_reward, grid_phi)
            .map_err(map_model_error)?;

        transitions_evaluated += trace.rollout_transitions_used() as u64;
        if transitions_evaluated >= config.budget.max_transitions {
            break;
        }

        let action = trace
            .first_action()
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let outcome = env.apply(&action).map_err(|err| match err {
            gogamen_core::EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
            gogamen_core::EnvironmentError::InvalidAction(d) => d,
            _ => DomainError::InvalidTransition {
                reason: "failed to apply fmc action in grid environment",
            },
        })?;
        model.step_in_place(&action).map_err(map_model_error)?;

        ticks_executed += 1;
        transitions_evaluated += 1;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let reached_goal = env.terminal() == Terminal::Terminated(TerminalReason::GoalReached);
    let failure_reason = if reached_goal {
        None
    } else {
        Some(classify_non_goal_reason(
            env.terminal(),
            false,
            ticks_executed,
            transitions_evaluated,
            config.budget,
        ))
    };

    Ok((
        reached_goal,
        ticks_executed,
        transitions_evaluated,
        env.total_reward(),
        failure_reason,
    ))
}

fn classify_non_goal_reason(
    terminal: Terminal,
    no_path: bool,
    ticks_executed: u64,
    transitions_evaluated: u64,
    budget: BaselineBudget,
) -> String {
    if no_path {
        return "no_path_to_goal".to_string();
    }
    match terminal {
        Terminal::Terminated(TerminalReason::Failure) => "terminated_failure_or_hazard".to_string(),
        Terminal::Terminated(TerminalReason::InvariantViolated) => "invariant_violated".to_string(),
        _ if ticks_executed >= budget.max_ticks => "max_ticks_exhausted_without_goal".to_string(),
        _ if transitions_evaluated >= budget.max_transitions => {
            "max_transitions_exhausted_without_goal".to_string()
        }
        _ => "stopped_without_goal".to_string(),
    }
}

fn run_single_seed(
    config: &FairComparisonConfig,
    algorithm: ExperimentAlgorithm,
    split: SeedSplit,
    seed: MasterSeed,
) -> SeedRunResult {
    let mut shared_world = config.world_config.clone();
    shared_world.map = config.map.clone();

    let started = Instant::now();
    let outcome: Result<(bool, u64, u64, f64, Option<String>), DomainError> = match algorithm {
        ExperimentAlgorithm::RandomSeeded => {
            run_random_baseline(&config.map, shared_world, seed, config.budget).map(|m| {
                let reason = if m.reached_goal {
                    None
                } else {
                    Some(classify_non_goal_reason(
                        Terminal::Ongoing,
                        m.no_path,
                        m.ticks_executed,
                        m.transitions_evaluated,
                        config.budget,
                    ))
                };
                (
                    m.reached_goal,
                    m.ticks_executed,
                    m.transitions_evaluated,
                    m.total_reward,
                    reason,
                )
            })
        }
        ExperimentAlgorithm::MonteCarloShooting => {
            let mc_config = MonteCarloConfig::new(
                config.mc_horizon,
                config.mc_rollouts,
                config.max_transitions_per_tick,
                GridAction::Stay,
            );
            run_monte_carlo_shooting_baseline(
                &config.map,
                shared_world,
                seed,
                mc_config,
                config.budget,
            )
            .map(|m| {
                let reason = if m.reached_goal {
                    None
                } else {
                    Some(classify_non_goal_reason(
                        Terminal::Ongoing,
                        false,
                        m.ticks_executed,
                        m.transitions_evaluated,
                        config.budget,
                    ))
                };
                (
                    m.reached_goal,
                    m.ticks_executed,
                    m.transitions_evaluated,
                    m.total_reward,
                    reason,
                )
            })
        }
        ExperimentAlgorithm::PrivilegedBfs => {
            run_bfs_baseline(&config.map, shared_world, config.budget).map(|m| {
                let reason = if m.reached_goal {
                    None
                } else {
                    Some(classify_non_goal_reason(
                        Terminal::Ongoing,
                        m.no_path,
                        m.ticks_executed,
                        m.transitions_evaluated,
                        config.budget,
                    ))
                };
                (
                    m.reached_goal,
                    m.ticks_executed,
                    m.transitions_evaluated,
                    m.total_reward,
                    reason,
                )
            })
        }
        ExperimentAlgorithm::Fmc => run_fmc_episode(config, seed),
    };
    let latency_ns = started.elapsed().as_nanos();

    match outcome {
        Ok((success, steps, transitions_evaluated, total_reward, failure_reason)) => {
            SeedRunResult {
                algorithm,
                split,
                seed,
                is_privileged_full_info: algorithm.is_privileged_full_info(),
                success,
                steps,
                transitions_evaluated,
                latency_ns,
                total_reward,
                failed: !success,
                failure_reason,
            }
        }
        Err(err) => SeedRunResult {
            algorithm,
            split,
            seed,
            is_privileged_full_info: algorithm.is_privileged_full_info(),
            success: false,
            steps: 0,
            transitions_evaluated: 0,
            latency_ns,
            total_reward: 0.0,
            failed: true,
            failure_reason: Some(format!("execution_error: {err}")),
        },
    }
}

fn percentile_sorted_u64(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0] as f64;
    }
    let pos = p * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    (sorted[lo] as f64) * (1.0 - frac) + (sorted[hi] as f64) * frac
}

fn build_aggregate(
    algorithm: ExperimentAlgorithm,
    split: Option<SeedSplit>,
    runs: &[&SeedRunResult],
) -> AlgorithmAggregate {
    let runs_count = runs.len();
    let successes_count = runs.iter().filter(|r| r.success).count();
    let failures_count = runs.iter().filter(|r| r.failed).count();
    let success_rate = if runs_count == 0 {
        0.0
    } else {
        (successes_count as f64) / (runs_count as f64)
    };

    let mut sorted_steps: Vec<u64> = runs.iter().map(|r| r.steps).collect();
    sorted_steps.sort_unstable();

    let mean_steps = if runs_count == 0 {
        0.0
    } else {
        (sorted_steps.iter().sum::<u64>() as f64) / (runs_count as f64)
    };
    let median_steps = percentile_sorted_u64(&sorted_steps, 0.5);
    let iqr_steps =
        percentile_sorted_u64(&sorted_steps, 0.75) - percentile_sorted_u64(&sorted_steps, 0.25);
    let min_max_steps = sorted_steps
        .first()
        .copied()
        .zip(sorted_steps.last().copied());

    let mean_transitions = if runs_count == 0 {
        0.0
    } else {
        (runs.iter().map(|r| r.transitions_evaluated).sum::<u64>() as f64) / (runs_count as f64)
    };
    let mean_latency_ns = if runs_count == 0 {
        0.0
    } else {
        (runs.iter().map(|r| r.latency_ns).sum::<u128>() as f64) / (runs_count as f64)
    };
    let mean_total_reward = if runs_count == 0 {
        0.0
    } else {
        runs.iter().map(|r| r.total_reward).sum::<f64>() / (runs_count as f64)
    };

    let failure_notes = runs
        .iter()
        .filter(|r| r.failed)
        .map(|r| {
            format!(
                "{} {:?} seed={}: {} (steps={}, transitions={})",
                algorithm.label(),
                r.split,
                r.seed.as_u64(),
                r.failure_reason.as_deref().unwrap_or("unknown_failure"),
                r.steps,
                r.transitions_evaluated,
            )
        })
        .collect();

    AlgorithmAggregate {
        algorithm,
        split,
        is_privileged_full_info: algorithm.is_privileged_full_info(),
        runs_count,
        successes_count,
        failures_count,
        success_rate,
        mean_steps,
        median_steps,
        iqr_steps,
        min_max_steps,
        mean_transitions,
        mean_latency_ns,
        mean_total_reward,
        failure_notes,
    }
}

fn build_analysis_notes(aggregates: &[AlgorithmAggregate]) -> Vec<String> {
    let mut notes = Vec::new();
    let find_agg = |algo: ExperimentAlgorithm| aggregates.iter().find(|a| a.algorithm == algo);

    if let Some(bfs) = find_agg(ExperimentAlgorithm::PrivilegedBfs) {
        notes.push(format!(
            "PrivilegedBfs (is_privileged_full_info={}) ceiling: success_rate={:.3}, mean_steps={:.2}.",
            bfs.is_privileged_full_info, bfs.success_rate, bfs.mean_steps
        ));
    }

    if let Some(fmc) = find_agg(ExperimentAlgorithm::Fmc) {
        if fmc.failures_count > 0 {
            notes.push(format!(
                "FMC recorded {}/{} failures (success_rate={:.3}): receding-horizon planning without privileged map knowledge can exhaust budget or hit local traps when horizon/per-tick transitions are insufficient.",
                fmc.failures_count, fmc.runs_count, fmc.success_rate
            ));
        }

        for peer_kind in [
            ExperimentAlgorithm::MonteCarloShooting,
            ExperimentAlgorithm::RandomSeeded,
        ] {
            if let Some(peer) = find_agg(peer_kind) {
                if fmc.success_rate < peer.success_rate {
                    notes.push(format!(
                        "FMC lost on success_rate against {} ({:.3} < {:.3}).",
                        peer.algorithm.label(),
                        fmc.success_rate,
                        peer.success_rate
                    ));
                } else if (fmc.success_rate - peer.success_rate).abs() < f64::EPSILON
                    && fmc.successes_count > 0
                    && fmc.mean_steps > peer.mean_steps
                {
                    notes.push(format!(
                        "FMC tied on success_rate ({:.3}) with {} but needed more mean_steps ({:.2} > {:.2}).",
                        fmc.success_rate,
                        peer.algorithm.label(),
                        fmc.mean_steps,
                        peer.mean_steps
                    ));
                }
            }
        }
    }

    notes
}

/// Ejecuta la comparación justa en Grid sobre exactamente los mismos estados
/// iniciales, mapa, reglas y presupuesto documentado para todos los algoritmos.
pub fn run_fair_grid_comparison(config: &FairComparisonConfig) -> FairComparisonReport {
    let mut seed_results = Vec::with_capacity(
        ExperimentAlgorithm::ALL.len() * (config.dev_seeds.len() + config.holdout_seeds.len()),
    );

    let all_seeds: Vec<(SeedSplit, MasterSeed)> = config
        .dev_seeds
        .iter()
        .copied()
        .map(|s| (SeedSplit::Dev, s))
        .chain(
            config
                .holdout_seeds
                .iter()
                .copied()
                .map(|s| (SeedSplit::Holdout, s)),
        )
        .collect();

    for algorithm in ExperimentAlgorithm::ALL {
        for &(split, seed) in &all_seeds {
            seed_results.push(run_single_seed(config, algorithm, split, seed));
        }
    }

    let mut aggregates = Vec::with_capacity(ExperimentAlgorithm::ALL.len());
    let mut split_aggregates = Vec::with_capacity(ExperimentAlgorithm::ALL.len() * 2);

    for algorithm in ExperimentAlgorithm::ALL {
        let algo_runs: Vec<&SeedRunResult> = seed_results
            .iter()
            .filter(|r| r.algorithm == algorithm)
            .collect();
        aggregates.push(build_aggregate(algorithm, None, &algo_runs));

        for split in [SeedSplit::Dev, SeedSplit::Holdout] {
            let split_runs: Vec<&SeedRunResult> = algo_runs
                .iter()
                .copied()
                .filter(|r| r.split == split)
                .collect();
            split_aggregates.push(build_aggregate(algorithm, Some(split), &split_runs));
        }
    }

    let analysis_notes = build_analysis_notes(&aggregates);

    FairComparisonReport {
        config: config.clone(),
        seed_results,
        aggregates,
        split_aggregates,
        analysis_notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_all_algorithms_share_map_rules_initial_state_and_documented_budget() {
        let config = FairComparisonConfig::default();
        let report = run_fair_grid_comparison(&config);

        assert_eq!(report.config, config);
        let expected_seeds = config.dev_seeds.len() + config.holdout_seeds.len();
        assert_eq!(
            report.seed_results.len(),
            ExperimentAlgorithm::ALL.len() * expected_seeds
        );

        for algo in ExperimentAlgorithm::ALL {
            let algo_runs: Vec<&SeedRunResult> = report
                .seed_results
                .iter()
                .filter(|r| r.algorithm == algo)
                .collect();
            assert_eq!(algo_runs.len(), expected_seeds);
            for run in &algo_runs {
                assert_eq!(run.is_privileged_full_info, algo.is_privileged_full_info());
                assert!(run.steps <= config.budget.max_ticks);
                assert!(run.transitions_evaluated <= config.budget.max_transitions);
            }
        }

        // Techo privilegiado explícitamente separado.
        let bfs_agg = report
            .aggregates
            .iter()
            .find(|a| a.algorithm == ExperimentAlgorithm::PrivilegedBfs)
            .unwrap();
        assert!(bfs_agg.is_privileged_full_info);
        assert_eq!(bfs_agg.success_rate, 1.0);

        for non_priv in [
            ExperimentAlgorithm::RandomSeeded,
            ExperimentAlgorithm::MonteCarloShooting,
            ExperimentAlgorithm::Fmc,
        ] {
            let agg = report
                .aggregates
                .iter()
                .find(|a| a.algorithm == non_priv)
                .unwrap();
            assert!(!agg.is_privileged_full_info);
        }

        // Splits Dev y Holdout separados explícitamente.
        assert_eq!(
            report.split_aggregates.len(),
            ExperimentAlgorithm::ALL.len() * 2
        );
        let dev_count = report
            .seed_results
            .iter()
            .filter(|r| r.split == SeedSplit::Dev)
            .count();
        let holdout_count = report
            .seed_results
            .iter()
            .filter(|r| r.split == SeedSplit::Holdout)
            .count();
        assert_eq!(
            dev_count,
            ExperimentAlgorithm::ALL.len() * config.dev_seeds.len()
        );
        assert_eq!(
            holdout_count,
            ExperimentAlgorithm::ALL.len() * config.holdout_seeds.len()
        );
    }

    #[test]
    fn ac2_exports_success_steps_transitions_latency_intervals_and_failures_per_seed() {
        let config = FairComparisonConfig::default();
        let report = run_fair_grid_comparison(&config);

        for run in &report.seed_results {
            assert!(run.latency_ns > 0);
            assert!(run.total_reward.is_finite());
            assert_eq!(run.failed, !run.success);
            if run.failed {
                assert!(run.failure_reason.is_some());
            } else {
                assert!(run.failure_reason.is_none());
            }
        }

        for agg in &report.aggregates {
            assert_eq!(
                agg.runs_count,
                config.dev_seeds.len() + config.holdout_seeds.len()
            );
            assert_eq!(agg.successes_count + agg.failures_count, agg.runs_count);
            assert!(agg.success_rate >= 0.0 && agg.success_rate <= 1.0);
            assert!(agg.mean_steps >= 0.0);
            assert!(agg.median_steps >= 0.0);
            assert!(agg.iqr_steps >= 0.0);
            assert!(agg.min_max_steps.is_some());
            let (min_s, max_s) = agg.min_max_steps.unwrap();
            assert!(min_s <= max_s);
            assert!(agg.mean_transitions >= 0.0);
            assert!(agg.mean_latency_ns > 0.0);
            assert_eq!(agg.failure_notes.len(), agg.failures_count);
        }
    }

    #[test]
    fn ac3_honestly_records_and_analyzes_fmc_losses_and_failures_without_hiding_them() {
        // Fixture exigente con presupuesto ajustado donde FMC no puede llegar al Goal
        // en cul_de_sac_map (requiere >= 16 pasos), mientras PrivilegedBfs o un
        // presupuesto menor fuerza fallo explícito de FMC.
        let tight_config = FairComparisonConfig {
            map: GridMap::cul_de_sac_map(),
            world_config: GridWorldConfig::default(),
            budget: BaselineBudget::new(6, 300),
            max_transitions_per_tick: 40,
            mc_horizon: 4,
            mc_rollouts: 8,
            fmc_population_size: 8,
            fmc_horizon: 4,
            fmc_scoring: FmcScoringConfig::default(),
            dev_seeds: vec![MasterSeed::new(1), MasterSeed::new(2)],
            holdout_seeds: vec![MasterSeed::new(101)],
        };

        let report = run_fair_grid_comparison(&tight_config);
        let fmc_agg = report
            .aggregates
            .iter()
            .find(|a| a.algorithm == ExperimentAlgorithm::Fmc)
            .unwrap();

        assert_eq!(fmc_agg.successes_count, 0);
        assert_eq!(fmc_agg.failures_count, 3);
        assert_eq!(fmc_agg.success_rate, 0.0);
        assert_eq!(fmc_agg.failure_notes.len(), 3);
        assert!(
            fmc_agg
                .failure_notes
                .iter()
                .all(|note| note.contains("Fmc") && note.contains("exhausted"))
        );
        assert!(
            report
                .analysis_notes
                .iter()
                .any(|note| note.contains("FMC recorded 3/3 failures"))
        );

        // También verificamos determinismo exacto de métricas (salvo latency_ns).
        let report2 = run_fair_grid_comparison(&tight_config);
        for (a, b) in report.seed_results.iter().zip(report2.seed_results.iter()) {
            assert_eq!(a.algorithm, b.algorithm);
            assert_eq!(a.split, b.split);
            assert_eq!(a.seed, b.seed);
            assert_eq!(a.success, b.success);
            assert_eq!(a.steps, b.steps);
            assert_eq!(a.transitions_evaluated, b.transitions_evaluated);
            assert_eq!(a.total_reward, b.total_reward);
            assert_eq!(a.failed, b.failed);
            assert_eq!(a.failure_reason, b.failure_reason);
        }
    }
}
