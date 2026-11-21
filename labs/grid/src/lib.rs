//! Laboratorio de mundo Grid discreto con callejón sin salida y obstáculos (`gogamen-lab-grid`).
//!
//! Implementa:
//! - [`GridMap`], [`CellKind`] y [`cul_de_sac_map()`] con un callejón sin salida que exige retroceder.
//! - [`GridWorldEnv`] (`World` + `LiveEnvironment`) y [`GridWorldModel`] (`GenerativeModel` + `BranchableSnapshot`).
//! - [`ObservationMode::Full`] y [`ObservationMode::Partial`] desactivable por configuración.
//! - [`run_grid_episode`] para episodios deterministas reproducibles por [`gogamen_core::MasterSeed`].

pub mod ablation;
pub mod baselines;
pub mod belief_fmc;
pub mod experiment;
pub mod freedom_proxy;
pub mod language_world;
pub mod map;
pub mod monte_carlo;
pub mod partial_obs;
pub mod pomcp;
pub mod scenario;
pub mod world;

pub use ablation::{
    AblationConfig, AblationError, AblationReport, AblationSeedRun, AblationVariant,
    AblationVariantSummary, BeliefCalibrationMetrics, CALIBRATION_FIXTURE_DISCLAIMER,
    ReliabilityBin, compute_calibration_metrics, default_ablation_map, run_ablation_demo,
    run_ablation_suite,
};
pub use baselines::{
    BaselineBudget, BaselineKind, BaselineMetrics, run_bfs_baseline, run_random_baseline,
};
pub use belief_fmc::{
    BELIEF_FMC_PROTOTYPE_NOTE, BeliefFmcComparison, BeliefFmcConfig, BeliefFmcDecision,
    BeliefFmcError, BeliefFmcMode, BeliefFmcPlanner, BeliefParticle, BeliefSampledFmcSummary,
    NominalFmcSummary, ParticleRolloutTrace, belief_epistemic_uncertainty_bits,
    build_nominal_map_from_belief, sample_belief_particles,
};
pub use experiment::{
    AlgorithmAggregate, ExperimentAlgorithm, FairComparisonConfig, FairComparisonReport,
    SeedRunResult, SeedSplit, run_fair_grid_comparison,
};
pub use freedom_proxy::{
    FUTURE_FREEDOM_EPISTEMOLOGICAL_NOTE, FreedomProxyError, FutureFreedomConfig,
    FutureFreedomReport, PureNoiseFixture, SamplingPolicy, default_grid_bin_spec,
    measure_future_freedom, run_grid_freedom_experiment, run_pure_noise_freedom_experiment,
};
pub use language_world::{
    LanguageWorldConfig, LanguageWorldError, LanguageWorldReport, LanguageWorldStepTrace,
    run_language_world_demo, run_language_world_with_config,
};
pub use map::{CellKind, GridMap, GridPos, GridSize, cul_de_sac_map};
pub use monte_carlo::{
    MonteCarloBaselineMetrics, MonteCarloConfig, MonteCarloPlanner, MonteCarloStepResult,
    run_monte_carlo_shooting_baseline,
};
pub use partial_obs::{
    BeliefCorrection, BeliefGridPlanner, CellHypothesis, DEFAULT_DIRECT_SENSOR_ID,
    DEFAULT_HUMAN_SENSOR_ID, DirectObservation, DirectObservationOutcome, FallibleHumanConfig,
    FallibleHumanFixture, GeneratedHumanReport, GridBeliefConfig, GridBeliefMap, GroundTruthGrid,
    GroundTruthObservationView, HumanReport, HumanReportCase, HypothesisKind,
    ObservationBatchReport, OccludedObservation, PartialGridObserver, PartialObsError,
    SeededHumanReportSuite, VisibleCell,
};
pub use pomcp::{
    AlgorithmComparisonSummary, AlgorithmEpisodeMetrics, EpisodeComparisonRow,
    HistoryVsOmniscientReport, NextInvestmentChoice, OmniscientRolloutSummary,
    POMCP_INFORMATION_GAP_NOTE, POMCP_RECOMMENDATION_DISCLAIMER, PomcpComparisonConfig,
    PomcpComparisonRecommendation, PomcpConfig, PomcpDecision, PomcpError, PomcpObservationKey,
    PomcpPlanner, PomcpRolloutPolicy, PomcpRootActionStat, PomcpVsBeliefFmcReport,
    build_pomcp_recommendation, run_history_vs_omniscient_fixture, run_omniscient_rollout,
    run_pomcp_vs_belief_fmc_comparison,
};
pub use scenario::{
    GRID_MAP_CONFIG_KEY, GRID_WORLD_NODE_KIND, GridScenario, GridScenarioError,
    grid_experiment_spec, grid_scenario_from_experiment, map_to_ascii_rows, run_grid_scenario,
};
pub use world::{
    ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_UP, GRID_ACTION_CARDINALITY,
    GridAction, GridEpisodeSummary, GridWorldConfig, GridWorldEnv, GridWorldModel, ObservationMode,
    run_grid_episode, run_grid_episode_with_config,
};

#[cfg(test)]
mod tests {
    use std::collections::{BTreeSet, VecDeque};

    use gogamen_core::{
        Action, ActionSpace, Actuator, BranchableSnapshot, DomainError, EnvironmentError,
        EnvironmentObserver, GenerativeModel, LiveEnvironment, MasterSeed, ModelError,
        SimulationSnapshot, StepDt, Terminal, TerminalReason, World,
    };

    use super::*;

    fn shortest_path_actions(map: &GridMap) -> Option<Vec<GridAction>> {
        let start = map.start();
        let goal = map.goal();
        let mut queue = VecDeque::new();
        let mut visited = BTreeSet::new();
        let mut parent: std::collections::BTreeMap<(usize, usize), ((usize, usize), GridAction)> =
            std::collections::BTreeMap::new();

        queue.push_back(start);
        visited.insert((start.x, start.y));

        while let Some(cur) = queue.pop_front() {
            if cur == goal {
                let mut path = Vec::new();
                let mut node = (goal.x, goal.y);
                while node != (start.x, start.y) {
                    let (prev, act) = parent[&node];
                    path.push(act);
                    node = prev;
                }
                path.reverse();
                return Some(path);
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
        None
    }

    #[test]
    fn grid_smoke_computes_cell_count() {
        let size = GridSize {
            width: 4,
            height: 3,
        };
        assert_eq!(size.cell_count(), 12);
    }

    #[test]
    fn cul_de_sac_map_requires_backtracking_to_reach_goal() {
        let map = cul_de_sac_map();
        let start = map.start();
        let goal = map.goal();
        let initial_dist = start.manhattan_distance(goal);

        // Un agente voraz que nunca acepta alejarse de la meta entra en el callejón (Right -> (4,3) -> (5,3))
        // y queda atrapado sin poder alcanzar `Goal`.
        let mut greedy_pos = start;
        for _ in 0..20 {
            let cur_dist = greedy_pos.manhattan_distance(goal);
            let mut best = None;
            for act in [
                GridAction::Up,
                GridAction::Down,
                GridAction::Left,
                GridAction::Right,
            ] {
                let (dx, dy) = act.delta();
                let nx = greedy_pos.x as isize + dx;
                let ny = greedy_pos.y as isize + dy;
                if nx < 0 || ny < 0 {
                    continue;
                }
                let cand = GridPos::new(nx as usize, ny as usize);
                if map.is_passable(cand) && cand.manhattan_distance(goal) < cur_dist {
                    best = Some(cand);
                    break;
                }
            }
            match best {
                Some(next) => greedy_pos = next,
                None => break,
            }
        }
        assert_ne!(
            greedy_pos, goal,
            "greedy forward-only policy must get trapped in cul-de-sac"
        );
        assert!(
            greedy_pos.manhattan_distance(goal) < initial_dist,
            "greedy agent must have entered the cul-de-sac closer to goal"
        );

        // El camino real existe mediante retroceso, aumentando temporalmente la distancia Manhattan.
        let path = shortest_path_actions(&map).expect("cul-de-sac map must have a valid path");
        let mut env = GridWorldEnv::new(GridWorldConfig::default()).unwrap();
        let space = Actuator::action_space(&env).clone();
        let mut max_dist_along_path = initial_dist;

        for act in path {
            let action = act.to_action(&space).unwrap();
            let outcome = env.apply(&action).unwrap();
            max_dist_along_path = max_dist_along_path.max(env.position().manhattan_distance(goal));
            if env.position() == goal {
                assert_eq!(
                    outcome.terminal(),
                    Terminal::Terminated(TerminalReason::GoalReached)
                );
            }
        }

        assert!(
            max_dist_along_path > initial_dist,
            "path must backtrack away from goal (max_dist={max_dist_along_path} > initial={initial_dist})"
        );
        assert_eq!(env.position(), goal);
        assert_eq!(
            env.terminal(),
            Terminal::Terminated(TerminalReason::GoalReached)
        );
    }

    #[test]
    fn fixture_covers_walls_terminal_reward_and_toggleable_partial_observation() {
        let map = GridMap::from_ascii(&["#####", "#S.G#", "#####"]).unwrap();

        // 1) Observación parcial activada (`radius = 1`, ventana 3x3 -> 3 + 9 = 12 features).
        let cfg_partial = GridWorldConfig {
            map: map.clone(),
            ..GridWorldConfig::default()
        }
        .with_partial_observation(Some(1));
        let mut env = GridWorldEnv::new(cfg_partial).unwrap();
        assert_eq!(
            Actuator::action_space(&env),
            &ActionSpace::discrete(5).unwrap()
        );

        let obs_partial = EnvironmentObserver::observe(&env).unwrap();
        assert_eq!(obs_partial.features().len(), 3 + 3 * 3);
        assert_eq!(obs_partial.features()[0], 1.0); // agent_x
        assert_eq!(obs_partial.features()[1], 1.0); // agent_y
        assert_eq!(obs_partial.features()[2], 1.0); // radius

        // Colisión contra muro superior (`Up`) mantiene posición y aplica `wall_collision_reward`.
        let space = Actuator::action_space(&env).clone();
        let up = GridAction::Up.to_action(&space).unwrap();
        let outcome = env.apply(&up).unwrap();
        assert_eq!(outcome.terminal(), Terminal::Ongoing);
        assert_eq!(env.position(), GridPos::new(1, 1));
        assert_eq!(env.collisions(), 1);
        assert_eq!(env.last_reward(), env.config().wall_collision_reward);
        assert!((env.total_reward() - (-0.5)).abs() < 1e-12);

        // Avanzar hasta `Goal` (`Right`, `Right`) produce `GoalReached` y recompensa positiva.
        let right = GridAction::Right.to_action(&space).unwrap();
        let step1 = env.apply(&right).unwrap();
        assert_eq!(step1.terminal(), Terminal::Ongoing);
        assert_eq!(env.position(), GridPos::new(2, 1));
        assert_eq!(env.last_reward(), env.config().step_reward);

        let step2 = env.apply(&right).unwrap();
        assert_eq!(
            step2.terminal(),
            Terminal::Terminated(TerminalReason::GoalReached)
        );
        assert_eq!(env.position(), GridPos::new(3, 1));
        assert_eq!(env.last_reward(), env.config().goal_reward);
        assert!(env.total_reward() > 0.0);

        // Pasos posteriores en estado terminal son rechazados hasta `reset()`.
        let err_live = env.apply(&right).unwrap_err();
        assert_eq!(err_live, EnvironmentError::AlreadyTerminal);
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let err_world = World::step(&mut env, &right, dt).unwrap_err();
        assert_eq!(err_world, DomainError::AlreadyTerminal);

        // Reinicio restaura posición, reward y estado `Ongoing`.
        let reset_obs = LiveEnvironment::reset(&mut env).unwrap();
        assert_eq!(reset_obs.features()[0], 1.0);
        assert_eq!(env.position(), GridPos::new(1, 1));
        assert_eq!(env.total_reward(), 0.0);
        assert_eq!(env.collisions(), 0);
        assert_eq!(env.terminal(), Terminal::Ongoing);

        // 2) Observación parcial desactivable (`None` -> `ObservationMode::Full`).
        let cfg_full = GridWorldConfig {
            map,
            ..GridWorldConfig::default()
        }
        .with_partial_observation(None);
        assert_eq!(cfg_full.observation_mode, ObservationMode::Full);
        let env_full = GridWorldEnv::new(cfg_full).unwrap();
        let obs_full = EnvironmentObserver::observe(&env_full).unwrap();
        // Full: [agent_x, agent_y, goal_x, goal_y, ...5*3 cells] = 4 + 15 = 19 features.
        assert_eq!(obs_full.features().len(), 4 + 15);
        assert_eq!(obs_full.features()[2], 3.0); // goal_x
        assert_eq!(obs_full.features()[3], 1.0); // goal_y
    }

    #[test]
    fn grid_world_model_supports_generative_step_fork_and_restore() {
        let config = GridWorldConfig::default();
        let mut model = GridWorldModel::new(config, MasterSeed::new(7)).unwrap();
        let space = model.action_space().clone();

        let snap = model.snapshot().unwrap();
        let mut branch = model.fork().unwrap();

        let right = GridAction::Right.to_action(&space).unwrap();
        let left = GridAction::Left.to_action(&space).unwrap();

        // Avanzamos la rama; `model` original no se contamina.
        let t_branch = branch.step_in_place(&right).unwrap();
        assert_eq!(branch.position(), GridPos::new(4, 3));
        assert_eq!(model.position(), GridPos::new(3, 3));
        assert_eq!(t_branch.terminal(), Terminal::Ongoing);

        // Avanzamos `model` contra muro izquierdo (`Left` desde (3,3) choca con `#` en (2,3)).
        model.step_in_place(&left).unwrap();
        assert_eq!(model.position(), GridPos::new(3, 3));
        assert_eq!(model.total_reward(), -0.5);

        // Restaurar `model` al snapshot previo devuelve estado exacto.
        model.restore(&snap).unwrap();
        assert_eq!(model.position(), GridPos::new(3, 3));
        assert_eq!(model.total_reward(), 0.0);

        // Snapshot corrupto debe fallar verificación.
        let mut env = snap.to_envelope();
        env.checksum = Some(0xdead_beef);
        assert!(model.restore_from_envelope(env).is_err());

        // Verificar `GenerativeModel::step` puro con estado terminal falla.
        let mut env_goal = GridWorldEnv::new(GridWorldConfig::default()).unwrap();
        let goal_state = gogamen_core::State::new(
            gogamen_core::Tick::ZERO,
            gogamen_core::SimDuration::ZERO,
            vec![
                env_goal.config().map.goal().x as f64,
                env_goal.config().map.goal().y as f64,
                0.0,
                0.0,
                0.0,
            ],
        )
        .unwrap();
        let mut rng = gogamen_core::DeterministicRng::new(1);
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let err = model
            .step(
                &goal_state,
                &Action::discrete_in_space(0, &space).unwrap(),
                dt,
                &mut rng,
            )
            .unwrap_err();
        assert_eq!(err, ModelError::TerminalState);
        let _ = World::reset(&mut env_goal).unwrap();
    }

    #[test]
    fn run_grid_episode_is_reproducible_by_seed_and_reflects_partial_obs() {
        let a = run_grid_episode(40, MasterSeed::new(42), false).unwrap();
        let b = run_grid_episode(40, MasterSeed::new(42), false).unwrap();
        assert_eq!(a, b);

        let other_seed = run_grid_episode(40, MasterSeed::new(99), false).unwrap();
        assert_ne!(a.trace_hash, other_seed.trace_hash);

        let partial_a = run_grid_episode(40, MasterSeed::new(42), true).unwrap();
        let partial_b = run_grid_episode(40, MasterSeed::new(42), true).unwrap();
        assert_eq!(partial_a, partial_b);
        assert_ne!(a.trace_hash, partial_a.trace_hash);
    }

    #[test]
    fn snapshot_schema_roundtrip_works() {
        let model = GridWorldModel::new(GridWorldConfig::default(), MasterSeed::new(123)).unwrap();
        let snap: SimulationSnapshot = model.snapshot().unwrap();
        snap.verify().unwrap();
    }

    #[test]
    fn bfs_baseline_reaches_goal_in_cul_de_sac_and_reports_no_path_when_closed() {
        let solvable = cul_de_sac_map();
        let budget = BaselineBudget::default();
        let bfs = run_bfs_baseline(&solvable, GridWorldConfig::default(), budget).unwrap();

        assert_eq!(bfs.kind, BaselineKind::PrivilegedBfs);
        assert!(bfs.is_privileged_full_info);
        assert_eq!(bfs.seed, None);
        assert!(bfs.reached_goal);
        assert!(!bfs.no_path);
        assert_eq!(bfs.collisions, 0);
        assert!(bfs.ticks_executed > 0);
        assert_eq!(bfs.path_len, Some(bfs.ticks_executed));
        assert!(bfs.transitions_evaluated >= bfs.ticks_executed);
        assert!(bfs.total_reward > 0.0);

        // Mapa completamente cerrado entre S y G: debe reportar `no_path: true` sin colgarse.
        let closed = GridMap::from_ascii(&["#####", "#S#G#", "#####"]).unwrap();
        let closed_bfs = run_bfs_baseline(&closed, GridWorldConfig::default(), budget).unwrap();
        assert_eq!(closed_bfs.kind, BaselineKind::PrivilegedBfs);
        assert!(closed_bfs.is_privileged_full_info);
        assert!(!closed_bfs.reached_goal);
        assert!(closed_bfs.no_path);
        assert_eq!(closed_bfs.ticks_executed, 0);
        assert_eq!(closed_bfs.path_len, None);
        assert_eq!(closed_bfs.total_reward, 0.0);
    }

    #[test]
    fn random_baseline_is_reproducible_by_seed_and_unprivileged() {
        let map = cul_de_sac_map();
        let budget = BaselineBudget::new(50, 50);
        let r1 = run_random_baseline(
            &map,
            GridWorldConfig::default(),
            MasterSeed::new(42),
            budget,
        )
        .unwrap();
        let r2 = run_random_baseline(
            &map,
            GridWorldConfig::default(),
            MasterSeed::new(42),
            budget,
        )
        .unwrap();
        let r_other = run_random_baseline(
            &map,
            GridWorldConfig::default(),
            MasterSeed::new(99),
            budget,
        )
        .unwrap();

        assert_eq!(r1, r2);
        assert_ne!(r1.trace_hash, r_other.trace_hash);
        assert_eq!(r1.kind, BaselineKind::RandomSeeded);
        assert!(!r1.is_privileged_full_info);
        assert_eq!(r1.seed, Some(MasterSeed::new(42)));
        assert!(!r1.no_path);
        assert!(r1.ticks_executed > 0);
        assert_eq!(r1.transitions_evaluated, r1.ticks_executed);
    }

    #[test]
    fn baselines_export_comparable_metrics_under_shared_budget() {
        let map = GridMap::from_ascii(&["######", "#S...#", "#.##.#", "#...G#", "######"]).unwrap();
        let budget = BaselineBudget::new(200, 1_000);
        let bfs = run_bfs_baseline(&map, GridWorldConfig::default(), budget).unwrap();
        let rnd = run_random_baseline(&map, GridWorldConfig::default(), MasterSeed::new(7), budget)
            .unwrap();

        assert!(bfs.is_privileged_full_info);
        assert!(!rnd.is_privileged_full_info);
        assert!(bfs.reached_goal);
        assert!(bfs.ticks_executed <= budget.max_ticks);
        assert!(rnd.ticks_executed <= budget.max_ticks);
        assert!(bfs.transitions_evaluated <= budget.max_transitions);
        assert!(rnd.transitions_evaluated <= budget.max_transitions);
        assert!(bfs.collisions == 0);
        // En este mapa abierto con presupuesto 200, el camino óptimo de BFS es <= al de Random (si llega).
        if rnd.reached_goal {
            assert!(bfs.ticks_executed <= rnd.ticks_executed);
        }
    }

    #[test]
    fn monte_carlo_selects_goal_trajectory_in_controlled_fixture_without_mutating_model() {
        let map = GridMap::from_ascii(&["######", "#S..G#", "######"]).unwrap();
        let config = GridWorldConfig {
            map: map.clone(),
            ..GridWorldConfig::default()
        };
        let model = GridWorldModel::new(config.clone(), MasterSeed::new(11)).unwrap();
        let initial_pos = model.position();
        let initial_reward = model.total_reward();

        let mc_config = MonteCarloConfig::new(4, 128, 512, GridAction::Stay);
        let mut planner = MonteCarloPlanner::new(mc_config);
        let mut rng = gogamen_core::DeterministicRng::new(11);
        let cancel = gogamen_core::CancelToken::new();

        let step_res = planner
            .plan_grid(
                &model,
                &mut rng,
                gogamen_core::PlanBudget::with_transitions(mc_config.max_transitions),
                &cancel,
            )
            .unwrap();

        // El modelo original no debe haber sido mutado por las simulaciones.
        assert_eq!(model.position(), initial_pos);
        assert_eq!(model.total_reward(), initial_reward);
        assert_eq!(
            step_res.plan_result.status(),
            gogamen_core::PlanStatus::Completed
        );
        assert!(!step_res.used_fallback);
        assert_eq!(step_res.chosen_action, GridAction::Right);
        assert_eq!(step_res.simulations_run, 128);
        assert!(step_res.transitions_evaluated <= mc_config.max_transitions);
        assert!(step_res.best_return.unwrap() > 0.0);

        // Ejecución de episodio completo alcanza la meta en el corredor controlado.
        let budget = BaselineBudget::new(10, 5_000);
        let mc_run =
            run_monte_carlo_shooting_baseline(&map, config, MasterSeed::new(11), mc_config, budget)
                .unwrap();

        assert!(mc_run.reached_goal);
        assert_eq!(mc_run.fallbacks_used, 0);
        assert!(!mc_run.selected_actions.is_empty());
        assert_eq!(mc_run.selected_actions[0], GridAction::Right);
        assert!(mc_run.total_reward > 0.0);
        assert!(mc_run.transitions_evaluated <= budget.max_transitions);
        assert_eq!(
            mc_run.to_baseline_metrics().kind,
            BaselineKind::MonteCarloShooting
        );
        assert!(!mc_run.to_baseline_metrics().is_privileged_full_info);
    }

    #[test]
    fn monte_carlo_counts_exact_simulations_and_uses_fallback_on_budget_exhaustion() {
        let map = GridMap::from_ascii(&["#####", "#S.G#", "#####"]).unwrap();
        let config = GridWorldConfig {
            map: map.clone(),
            ..GridWorldConfig::default()
        };
        let model = GridWorldModel::new(config.clone(), MasterSeed::new(5)).unwrap();

        // Presupuesto de 7 transiciones con horizon=5, rollouts=4:
        // Rollout 0 consume 5 transiciones (1 simulación); Rollout 1 consume 2 y agota en el paso 3.
        let mc_partial = MonteCarloConfig::new(5, 4, 7, GridAction::Left);
        let mut planner = MonteCarloPlanner::new(mc_partial);
        let mut rng = gogamen_core::DeterministicRng::new(5);
        let cancel = gogamen_core::CancelToken::new();

        let step_res = planner
            .plan_grid(
                &model,
                &mut rng,
                gogamen_core::PlanBudget::with_transitions(7),
                &cancel,
            )
            .unwrap();

        assert_eq!(
            step_res.plan_result.status(),
            gogamen_core::PlanStatus::BudgetExhausted
        );
        assert!(step_res.used_fallback);
        assert_eq!(step_res.chosen_action, GridAction::Left);
        assert_eq!(
            step_res.plan_result.authorized_action(),
            Some(&GridAction::Left.to_action(model.action_space()).unwrap())
        );
        assert_eq!(step_res.simulations_run, 2);
        assert_eq!(step_res.transitions_evaluated, 7);

        // Episodio con `max_transitions = 0` en `mc_config`: 0 simulaciones y 1 fallback por tick.
        let mc_zero = MonteCarloConfig::new(4, 8, 0, GridAction::Stay);
        let budget = BaselineBudget::new(3, 10);
        let ep_zero =
            run_monte_carlo_shooting_baseline(&map, config, MasterSeed::new(5), mc_zero, budget)
                .unwrap();

        assert!(!ep_zero.reached_goal);
        assert_eq!(ep_zero.ticks_executed, 3);
        assert_eq!(ep_zero.simulations_run, 0);
        assert_eq!(ep_zero.fallbacks_used, 3);
        assert_eq!(ep_zero.transitions_evaluated, 3);
        assert_eq!(
            ep_zero.selected_actions,
            vec![GridAction::Stay, GridAction::Stay, GridAction::Stay]
        );

        // Verificar también contrato `gogamen_core::Planner` sobre `PlanningContext`.
        let mut ctx_rng = gogamen_core::DeterministicRng::new(17);
        let mut ctx = gogamen_core::PlanningContext::new(&model, &mut ctx_rng);
        let mut trait_planner =
            MonteCarloPlanner::new(MonteCarloConfig::new(3, 16, 64, GridAction::Stay));
        let req = gogamen_core::PlanRequest::from_state(
            model.state().clone(),
            model.action_space().clone(),
            gogamen_core::PlanBudget::with_transitions(64),
        );
        let plan_res = gogamen_core::Planner::plan(&mut trait_planner, &mut ctx, &req).unwrap();
        assert_eq!(plan_res.status(), gogamen_core::PlanStatus::Completed);
        assert!(plan_res.authorized_action().is_some());
        assert!(plan_res.transitions_used() <= 64);
    }

    #[test]
    fn monte_carlo_is_reproducible_for_equal_seed_and_budget() {
        let map = GridMap::from_ascii(&["######", "#S..G#", "######"]).unwrap();
        let config = GridWorldConfig {
            map: map.clone(),
            ..GridWorldConfig::default()
        };
        let mc_config = MonteCarloConfig::new(4, 32, 128, GridAction::Stay);
        let budget = BaselineBudget::new(10, 1_000);

        let run_a = run_monte_carlo_shooting_baseline(
            &map,
            config.clone(),
            MasterSeed::new(77),
            mc_config,
            budget,
        )
        .unwrap();
        let run_b = run_monte_carlo_shooting_baseline(
            &map,
            config.clone(),
            MasterSeed::new(77),
            mc_config,
            budget,
        )
        .unwrap();
        let run_other_seed =
            run_monte_carlo_shooting_baseline(&map, config, MasterSeed::new(88), mc_config, budget)
                .unwrap();

        assert_eq!(run_a.selected_actions, run_b.selected_actions);
        assert_eq!(run_a.simulations_run, run_b.simulations_run);
        assert_eq!(run_a.transitions_evaluated, run_b.transitions_evaluated);
        assert_eq!(run_a.fallbacks_used, run_b.fallbacks_used);
        assert_eq!(run_a.trace_hash, run_b.trace_hash);
        assert_eq!(run_a, run_b);

        assert_ne!(run_a.trace_hash, run_other_seed.trace_hash);
    }
}
