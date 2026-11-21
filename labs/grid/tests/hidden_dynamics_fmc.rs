//! GOG-168: FMC planning on Grid with hidden physics (learned, versioned model only).

use gogamen_core::{
    CancelToken, LearnedModelConfig, LearningPolicy, MasterSeed, PlanStatus, Tick, UnseenPairPolicy,
};
use gogamen_lab_grid::hidden_dynamics::{
    FallbackReason, GridTask, HiddenCurveConfig, PlanGrid, PlanGridConfig,
    collect_hidden_experience, freeze_grid_model, grid_cell_scale, run_hidden_dynamics_curve,
    run_plan_grid_episode,
};
use gogamen_lab_grid::{GridMap, GridPos, GridWorldConfig};

fn agent(world: &GridWorldConfig, n: usize, policy: UnseenPairPolicy, budget: usize) -> PlanGrid {
    let scale = grid_cell_scale(&world.map).unwrap();
    let learner = collect_hidden_experience(
        world,
        &scale,
        n,
        MasterSeed::new(5),
        LearningPolicy::new(0.05).unwrap(),
    )
    .unwrap();
    let model = freeze_grid_model(
        &learner,
        scale,
        LearnedModelConfig::new(0.5, 2, policy).unwrap(),
    )
    .unwrap();
    let mut config = PlanGridConfig::new(GridTask::from_world(world));
    config.budget_transitions = budget;
    PlanGrid::new(model, config).unwrap()
}

fn detour_world() -> GridWorldConfig {
    let map =
        GridMap::from_ascii(&["#######", "#S.#.G#", "#..#..#", "#.....#", "#######"]).unwrap();
    GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    }
}

#[test]
fn hidden_dynamics_fmc_trace_carries_model_version() {
    let world = GridWorldConfig::default();
    let a = agent(&world, 800, UnseenPairPolicy::SamplePrior, 160);
    let d = a.decide(world.map.start(), Tick::ZERO, None);
    let v = a.model().version();
    assert_eq!(d.trace.model_revision, v.revision());
    assert_eq!(d.trace.model_checksum, v.checksum());
    assert_eq!(d.trace.status, Some(PlanStatus::Completed));
    assert!(d.trace.transitions_used <= 160 && d.trace.transitions_used > 0);
    // Different experience -> different version checksum recorded in the trace.
    let b = agent(&world, 200, UnseenPairPolicy::SamplePrior, 160);
    let d2 = b.decide(world.map.start(), Tick::ZERO, None);
    assert_ne!(d.trace.model_checksum, d2.trace.model_checksum);
}

#[test]
fn hidden_dynamics_fmc_abstains_with_exact_typed_reasons() {
    let world = GridWorldConfig::default();
    let start = world.map.start();

    // Unseen pair + Reject policy.
    let a = agent(&world, 0, UnseenPairPolicy::Reject, 160);
    let d = a.decide(start, Tick::ZERO, None);
    assert_eq!(d.trace.fallback, Some(FallbackReason::UnseenPairRejected));
    assert_eq!(d.trace.action_index, 0);

    // Uncertain model sampled from the prior: abstain instead of acting on invented dynamics.
    let a = agent(&world, 0, UnseenPairPolicy::SamplePrior, 160);
    let d = a.decide(start, Tick::ZERO, None);
    assert_eq!(d.trace.fallback, Some(FallbackReason::UncertainModel));
    assert!(d.trace.uncertain_steps > 0);
    assert_eq!(d.trace.uncertain_steps, d.trace.simulated_steps);

    // Budget expiry (budget below one population step).
    let a = agent(&world, 800, UnseenPairPolicy::SamplePrior, 5);
    let d = a.decide(start, Tick::ZERO, None);
    assert_eq!(d.trace.fallback, Some(FallbackReason::BudgetExhausted));
    assert!(d.trace.transitions_used <= 5);
    let a0 = agent(&world, 800, UnseenPairPolicy::SamplePrior, 0);
    let d = a0.decide(start, Tick::ZERO, None);
    assert_eq!(d.trace.fallback, Some(FallbackReason::BudgetExhausted));
    assert_eq!(d.trace.transitions_used, 0);

    // Cancellation.
    let a = agent(&world, 800, UnseenPairPolicy::SamplePrior, 160);
    let token = CancelToken::new();
    token.cancel();
    let d = a.decide(start, Tick::ZERO, Some(&token));
    assert_eq!(d.trace.fallback, Some(FallbackReason::Cancelled));
}

#[test]
fn hidden_dynamics_fmc_learns_hazard_as_failure() {
    let map = GridMap::from_ascii(&["#####", "#SHG#", "#...#", "#####"]).unwrap();
    let world = GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    };
    let scale = grid_cell_scale(&world.map).unwrap();
    let learner = collect_hidden_experience(
        &world,
        &scale,
        1500,
        MasterSeed::new(9),
        LearningPolicy::new(0.05).unwrap(),
    )
    .unwrap();
    let model = freeze_grid_model(&learner, scale.clone(), LearnedModelConfig::default()).unwrap();
    let start = scale.bucket_of(&[1.5, 1.5]).unwrap();
    let right = gogamen_lab_grid::world::ACTION_RIGHT;
    let p = model.predict(start, right).unwrap();
    assert!(p.distribution[scale.failure_bucket()] > 0.8, "{p:?}");
    // The learned-model agent avoids stepping right into the hazard.
    let mut cfg = PlanGridConfig::new(GridTask::from_world(&world));
    cfg.max_uncertain_fraction = 1.0;
    let agent = PlanGrid::new(model, cfg).unwrap();
    let d = agent.decide(GridPos::new(1, 1), Tick::ZERO, None);
    assert_ne!(d.trace.action_index, right);
}

#[test]
fn hidden_dynamics_fmc_curve_on_holdout_seeds() {
    let config = HiddenCurveConfig::new(detour_world());
    let budget = config.plan.budget_transitions as f64;
    let curve = run_hidden_dynamics_curve(&config).unwrap();
    assert_eq!(curve.len(), config.observation_counts.len());

    println!(
        "obs | planner goal/fail/reward | random goal | greedy goal | ceiling goal/reward | regret | fallbacks"
    );
    for p in &curve {
        println!(
            "{:>5} | {:.2}/{:.2}/{:>6.2} | {:.2} | {:.2} | {:.2}/{:>6.2} | {:>6.2} | {:?}",
            p.observations,
            p.planner.stats.goal_rate,
            p.planner.stats.failure_rate,
            p.planner.stats.mean_reward,
            p.random.goal_rate,
            p.greedy.goal_rate,
            p.ceiling.goal_rate,
            p.ceiling.mean_reward,
            p.regret_vs_ceiling,
            p.planner.fallbacks
        );
        // Equal-cost cap: nobody exceeds the shared per-decision transition budget.
        assert!(p.planner.stats.mean_transitions_per_decision <= budget);
        assert!(p.ceiling.mean_transitions_per_decision <= budget);
        assert_eq!(p.random.mean_transitions_per_decision, 0.0);
        assert!(p.regret_vs_ceiling.is_finite());
    }
    // The ceiling does not depend on experience.
    assert!(curve.windows(2).all(|w| w[0].ceiling == w[1].ceiling));
    // Zero experience: the agent abstains on every decision (no invented dynamics).
    let zero = &curve[0];
    assert_eq!(zero.planner.stats.goal_rate, 0.0);
    assert_eq!(
        zero.planner
            .fallbacks
            .get(&FallbackReason::UncertainModel)
            .copied()
            .unwrap_or(0),
        zero.planner.decisions
    );
    // Observed result on these fixed holdout seeds (deterministic): with enough experience the
    // planner reaches the goal where equal-cost random and learner-greedy do not. The gap to the
    // privileged ceiling stays large; this is reported above, not thresholded.
    let best = curve
        .iter()
        .map(|p| p.planner.stats.goal_rate)
        .fold(0.0, f64::max);
    let last = curve.last().unwrap();
    assert!(best > last.random.goal_rate && best > last.greedy.goal_rate);
    assert!(
        curve
            .iter()
            .all(|p| p.planner.stats.goal_rate <= p.ceiling.goal_rate)
    );
}

#[test]
fn hidden_dynamics_fmc_curve_is_deterministic() {
    let mut config = HiddenCurveConfig::new(detour_world());
    config.observation_counts = vec![0, 200];
    config.holdout_seeds.truncate(2);
    assert_eq!(
        run_hidden_dynamics_curve(&config).unwrap(),
        run_hidden_dynamics_curve(&config).unwrap()
    );
}

#[test]
fn hidden_dynamics_fmc_episode_runs_without_simulator_access() {
    let world = GridWorldConfig::default();
    let a = agent(&world, 3000, UnseenPairPolicy::SamplePrior, 160);
    let ep = run_plan_grid_episode(&world, &a, 40).unwrap();
    assert_eq!(ep.traces.len() as u64, ep.ticks);
    assert!(
        ep.traces
            .iter()
            .all(|t| t.model_checksum == a.model().version().checksum())
    );
}
