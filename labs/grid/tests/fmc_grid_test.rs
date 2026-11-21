use gogamen_core::{
    Actuator, CancelToken, DeterministicRng, FmcPlanner, FmcPlannerConfig, FmcScoringConfig,
    GenerativeModel, MasterSeed, PlanBudget, PlanRequest, PlanStatus, PlanningContext, State,
    Terminal, TerminalReason, Transition,
};
use gogamen_lab_grid::{GridAction, GridMap, GridWorldConfig, GridWorldEnv, GridWorldModel};

fn grid_step_reward(prev: &State, _action: &gogamen_core::Action, transition: &Transition) -> f64 {
    transition.next_state().values()[2] - prev.values()[2]
}

fn grid_phi(state: &State) -> Vec<f64> {
    vec![state.values()[0], state.values()[1]]
}

#[test]
fn ac1_grid_episode_runs_without_live_env_side_effects_during_planning() {
    let map =
        GridMap::from_ascii(&["#######", "#S.#.G#", "#..#..#", "#.....#", "#######"]).unwrap();
    let config = GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    };
    let mut env = GridWorldEnv::new(config.clone()).unwrap();
    let model = GridWorldModel::new(config.clone(), MasterSeed::new(11)).unwrap();

    let fmc_config = FmcPlannerConfig::new(
        24,
        10,
        config.dt,
        FmcScoringConfig::default(),
        MasterSeed::new(77),
    )
    .unwrap();
    let planner = FmcPlanner::new(fmc_config);
    let mut rng = DeterministicRng::new(999);
    let fallback = GridAction::Stay.to_action(model.action_space()).unwrap();

    let max_episode_steps = 30;
    let mut reached_goal = false;

    for _ in 0..max_episode_steps {
        if env.terminal().is_terminal() {
            break;
        }

        let current_state = env.current_state().unwrap();
        let snap_tick = env.tick();
        let snap_pos = env.position();
        let snap_reward = env.total_reward();
        let snap_collisions = env.collisions();
        let snap_terminal = env.terminal();

        let request = PlanRequest::from_state(
            current_state,
            model.action_space().clone(),
            PlanBudget::with_transitions(24 * 10),
        )
        .with_fallback(fallback.clone())
        .unwrap();

        let mut ctx = PlanningContext::new(&model, &mut rng);
        let (result, trace) = planner
            .plan_with_trace(&mut ctx, &request, grid_step_reward, grid_phi)
            .unwrap();

        // AC1: El entorno real NO debe sufrir ningún efecto durante las simulaciones del planner.
        assert_eq!(env.tick(), snap_tick);
        assert_eq!(env.position(), snap_pos);
        assert_eq!(env.total_reward(), snap_reward);
        assert_eq!(env.collisions(), snap_collisions);
        assert_eq!(env.terminal(), snap_terminal);

        // AC3: Trace consistente con la decisión tomada.
        assert_eq!(trace.status(), result.status());
        assert_eq!(trace.first_action(), result.authorized_action());
        assert!(trace.rollout_transitions_used() > 0);
        assert!(trace.mean_virtual_reward().is_finite());

        let action = result
            .authorized_action()
            .expect("completed plan must authorize an action");
        let outcome = env.apply(action).unwrap();
        if outcome.terminal() == Terminal::Terminated(TerminalReason::GoalReached) {
            reached_goal = true;
            break;
        }
    }

    assert!(
        reached_goal,
        "receding-horizon FMC planner should reach Goal on cul_de_sac_map within {max_episode_steps} steps"
    );
}

#[test]
fn ac2_respects_budget_terminal_cancellation_and_fallback() {
    let config = GridWorldConfig::default();
    let env = GridWorldEnv::new(config.clone()).unwrap();
    let model = GridWorldModel::new(config.clone(), MasterSeed::new(5)).unwrap();

    let fmc_config = FmcPlannerConfig::new(
        16,
        6,
        config.dt,
        FmcScoringConfig::default(),
        MasterSeed::new(33),
    )
    .unwrap();
    let planner = FmcPlanner::new(fmc_config);
    let mut rng = DeterministicRng::new(12);
    let fallback = GridAction::Stay.to_action(model.action_space()).unwrap();
    let start_state = env.current_state().unwrap();

    // 1. Budget 0 -> BudgetExhausted + fallback explícito, 0 transiciones.
    let req_zero = PlanRequest::from_state(
        start_state.clone(),
        model.action_space().clone(),
        PlanBudget::with_transitions(0),
    )
    .with_fallback(fallback.clone())
    .unwrap();
    let mut ctx = PlanningContext::new(&model, &mut rng);
    let (res_zero, trace_zero) = planner
        .plan_with_trace(&mut ctx, &req_zero, grid_step_reward, grid_phi)
        .unwrap();
    assert_eq!(res_zero.status(), PlanStatus::BudgetExhausted);
    assert_eq!(res_zero.transitions_used(), 0);
    assert_eq!(res_zero.authorized_action(), Some(&fallback));
    assert_eq!(trace_zero.first_action(), Some(&fallback));
    assert_eq!(trace_zero.rollout_transitions_used(), 0);

    // 2. Budget parcial (< population_size) -> BudgetExhausted + fallback, cap estricto.
    let req_partial = PlanRequest::from_state(
        start_state.clone(),
        model.action_space().clone(),
        PlanBudget::with_transitions(5),
    )
    .with_fallback(fallback.clone())
    .unwrap();
    let (res_partial, trace_partial) = planner
        .plan_with_trace(&mut ctx, &req_partial, grid_step_reward, grid_phi)
        .unwrap();
    assert_eq!(res_partial.status(), PlanStatus::BudgetExhausted);
    assert!(res_partial.transitions_used() <= 5);
    assert_eq!(res_partial.authorized_action(), Some(&fallback));
    assert_eq!(trace_partial.status(), PlanStatus::BudgetExhausted);

    // 3. Cancelación -> Cancelled, nunca autoriza acción ni fallback.
    let token = CancelToken::cancelled();
    let req_cancel = PlanRequest::from_state(
        start_state,
        model.action_space().clone(),
        PlanBudget::with_transitions(200),
    )
    .with_cancel_token(token)
    .with_fallback(fallback)
    .unwrap();
    let (res_cancel, trace_cancel) = planner
        .plan_with_trace(&mut ctx, &req_cancel, grid_step_reward, grid_phi)
        .unwrap();
    assert_eq!(res_cancel.status(), PlanStatus::Cancelled);
    assert_eq!(res_cancel.authorized_action(), None);
    assert_eq!(res_cancel.transitions_used(), 0);
    assert_eq!(trace_cancel.status(), PlanStatus::Cancelled);
    assert_eq!(trace_cancel.first_action(), None);

    // 4. Estado terminal en la meta -> NoValidAction, 0 transiciones.
    let goal = config.map.goal();
    let goal_state = State::new(
        env.tick(),
        gogamen_core::SimDuration::ZERO,
        vec![goal.x as f64, goal.y as f64, 10.0, 10.0, 0.0],
    )
    .unwrap();
    let req_terminal = PlanRequest::from_state(
        goal_state,
        model.action_space().clone(),
        PlanBudget::with_transitions(200),
    );
    let (res_term, trace_term) = planner
        .plan_with_trace(&mut ctx, &req_terminal, grid_step_reward, grid_phi)
        .unwrap();
    assert_eq!(res_term.status(), PlanStatus::NoValidAction);
    assert_eq!(res_term.authorized_action(), None);
    assert_eq!(res_term.transitions_used(), 0);
    assert_eq!(trace_term.status(), PlanStatus::NoValidAction);

    // Confirmar que `env` no fue mutado en ninguna de las llamadas anteriores.
    assert_eq!(env.tick().as_u64(), 0);
    assert_eq!(env.position(), config.map.start());
}

#[test]
fn ac3_trace_reports_first_action_counts_virtual_reward_and_diversity() {
    let config = GridWorldConfig::default();
    let env = GridWorldEnv::new(config.clone()).unwrap();
    let model = GridWorldModel::new(config.clone(), MasterSeed::new(19)).unwrap();

    let fmc_config = FmcPlannerConfig::new(
        20,
        5,
        config.dt,
        FmcScoringConfig::default(),
        MasterSeed::new(101),
    )
    .unwrap();
    let planner = FmcPlanner::new(fmc_config);
    let mut rng = DeterministicRng::new(55);

    let request = PlanRequest::from_state(
        env.current_state().unwrap(),
        model.action_space().clone(),
        PlanBudget::with_transitions(200),
    );
    let mut ctx = PlanningContext::new(&model, &mut rng);
    let (result, trace) = planner
        .plan_with_trace(&mut ctx, &request, grid_step_reward, grid_phi)
        .unwrap();

    assert_eq!(result.status(), PlanStatus::Completed);
    assert_eq!(trace.status(), PlanStatus::Completed);
    assert_eq!(trace.first_action(), result.authorized_action());
    assert!(trace.first_action().is_some());
    assert_eq!(trace.rollout_transitions_used(), result.transitions_used());
    assert!(trace.rollout_transitions_used() > 0 && trace.rollout_transitions_used() <= 100);
    assert!(trace.mean_virtual_reward().is_finite() && trace.mean_virtual_reward() > 0.0);
    assert!(trace.unique_first_actions() >= 1 && trace.unique_first_actions() <= 5);
    assert!(trace.action_diversity() > 0.0 && trace.action_diversity() <= 1.0);
    assert!(trace.cloning_events() <= 20 * 4);
}
