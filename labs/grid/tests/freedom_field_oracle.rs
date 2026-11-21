//! GOG-182 (DEM-034): the core freedom field against the exact DEM-001 oracle (`BoolWorld`) on
//! pocket worlds, and its use as an optional FMC leaf value on hidden-physics Grid.

use std::sync::Arc;

use gogamen_core::{
    ActionMeasure, BinSpec, BinningVersion, FreedomField, FreedomFieldConfig, FreedomFieldKey,
    FreedomKernel, LearnedModelConfig, LearningPolicy, MasterSeed, TabularScale, UnseenPairPolicy,
};
use gogamen_lab_grid::bool_world::{BoolSamplingPolicy, BoolWorld, MicroState, OracleLimits};
use gogamen_lab_grid::hidden_dynamics::{
    GridTask, PlanGrid, PlanGridConfig, PolicyStats, collect_hidden_experience, freeze_grid_model,
    grid_cell_scale, run_plan_grid_episode,
};
use gogamen_lab_grid::{GridMap, GridWorldConfig};

/// Dense kernel of a BoolWorld whose viable states all have the same number of actions.
/// Inviable destinations map to the absorbing failure column.
fn kernel_of(world: &BoolWorld) -> (FreedomKernel, Vec<MicroState>, usize) {
    let specs: Vec<_> = world.state_specs().collect();
    let viable: Vec<MicroState> = specs.iter().filter(|s| s.viable).map(|s| s.state).collect();
    let actions = specs.iter().find(|s| s.viable).unwrap().actions.len();
    let n = viable.len();
    let mut kernel = FreedomKernel::new(n, actions, 1_000_000).unwrap();
    for (i, st) in viable.iter().enumerate() {
        let spec = specs.iter().find(|s| s.state == *st).unwrap();
        assert_eq!(spec.actions.len(), actions);
        for (a, dist) in spec.actions.iter().enumerate() {
            let mut row = vec![0.0; n + 1];
            for &(dest, p) in dist {
                match viable.iter().position(|v| *v == dest) {
                    Some(j) => row[j] += p,
                    None => row[n] += p,
                }
            }
            kernel.set_row(i, a, &row).unwrap();
        }
    }
    (kernel, viable, actions)
}

fn key(n: usize) -> FreedomFieldKey {
    let spec = BinSpec::uniform(
        "pocket",
        BinningVersion::new(1, 0),
        vec![(0.0, n as f64, n)],
    )
    .unwrap();
    FreedomFieldKey {
        revision: 0,
        checksum: 0,
        scale: TabularScale::new(spec).unwrap(),
    }
}

fn s(i: u8) -> MicroState {
    MicroState(i)
}

/// 3 viable states + 1 inviable, 2 actions each.
fn pocket_world() -> BoolWorld {
    BoolWorld::new(1)
        .with_state(
            s(0),
            true,
            vec![vec![(s(0), 0.5), (s(1), 0.5)], vec![(s(2), 1.0)]],
        )
        .unwrap()
        .with_state(
            s(1),
            true,
            vec![vec![(s(1), 0.25), (s(3), 0.75)], vec![(s(0), 1.0)]],
        )
        .unwrap()
        .with_state(
            s(2),
            true,
            vec![vec![(s(3), 1.0)], vec![(s(2), 0.5), (s(0), 0.5)]],
        )
        .unwrap()
        .with_state(s(3), false, vec![])
        .unwrap()
}

/// 2 viable states + 1 inviable; eigenvalues of the uniform-action transfer matrix are known in
/// closed form: `T = [[0.5, 0.25], [0.125, 0.5]]`.
fn two_state_world() -> BoolWorld {
    BoolWorld::new(2)
        .with_state(
            s(0),
            true,
            vec![
                vec![(s(0), 0.5), (s(1), 0.25), (s(2), 0.25)],
                vec![(s(0), 0.5), (s(1), 0.25), (s(2), 0.25)],
            ],
        )
        .unwrap()
        .with_state(
            s(1),
            true,
            vec![
                vec![(s(0), 0.125), (s(1), 0.5), (s(2), 0.375)],
                vec![(s(0), 0.125), (s(1), 0.5), (s(2), 0.375)],
            ],
        )
        .unwrap()
        .with_state(s(2), false, vec![])
        .unwrap()
}

fn limits() -> OracleLimits {
    OracleLimits {
        max_trajectories: 1_000_000,
        ..OracleLimits::default()
    }
}

fn assert_matches_oracle(world: &BoolWorld, horizons: &[usize]) {
    let (kernel, viable, actions) = kernel_of(world);
    for &h in horizons {
        let mean = FreedomField::compute_from_kernel(
            &kernel,
            key(viable.len()),
            FreedomFieldConfig::with_horizon(h),
        )
        .unwrap();
        let count = FreedomField::compute_from_kernel(
            &kernel,
            key(viable.len()),
            FreedomFieldConfig {
                measure: ActionMeasure::CountActions,
                ..FreedomFieldConfig::with_horizon(h)
            },
        )
        .unwrap();
        for (i, st) in viable.iter().enumerate() {
            let v = mean.value(i).unwrap();
            match world.oracle(*st, &BoolSamplingPolicy::UniformActions, h, limits()) {
                Ok(report) => {
                    // exp(V_h) is exactly the oracle's (unnormalized) viable mass.
                    let want = report.viable_mass.ln();
                    assert!((v - want).abs() < 1e-12, "h={h} s={st:?}: {v} vs {want}");
                    let c = count.value(i).unwrap();
                    assert!((c - v - h as f64 * (actions as f64).ln()).abs() < 1e-12);
                }
                Err(_) => assert_eq!(v, f64::NEG_INFINITY, "h={h} s={st:?}"),
            }
        }
    }
}

#[test]
fn freedom_field_matches_oracle_on_pocket_worlds() {
    assert_matches_oracle(&pocket_world(), &[1, 2, 3, 5, 8]);
    assert_matches_oracle(&two_state_world(), &[1, 4, 9]);
    // Seeded worlds (no inviable states: viable mass is 1, V = 0).
    for seed in 1..=6 {
        let w = BoolWorld::from_seed(seed, 6).unwrap();
        assert_matches_oracle(&w, &[1, 4, 7]);
    }
}

#[test]
fn freedom_field_reports_convergence_and_eigenvalue_separation() {
    let (kernel, viable, _) = kernel_of(&two_state_world());
    let f = FreedomField::compute_from_kernel(
        &kernel,
        key(viable.len()),
        FreedomFieldConfig::default(),
    )
    .unwrap();
    assert!(f.converged());
    // T = [[.5,.25],[.125,.5]]: eigenvalues 0.5 +- sqrt(.25*.125) = 0.5 +- 0.1767767.
    let (l1, l2) = (
        0.5 + (0.25f64 * 0.125).sqrt(),
        0.5 - (0.25f64 * 0.125).sqrt(),
    );
    let ratio = f.eigenvalue_ratio_estimate().unwrap();
    println!(
        "two-state world: iterations={} lambda1={l1:.6} lambda2/lambda1={:.6} estimate={ratio:.6}",
        f.iterations().len(),
        l2 / l1
    );
    for it in f.iterations().iter().take(6) {
        println!(
            "  iter {:>2} residual {:.3e} log-growth {:.6} contraction {:?}",
            it.iteration, it.residual, it.log_growth, it.contraction
        );
    }
    assert!((f.log_growth_estimate().unwrap() - l1.ln()).abs() < 1e-8);
    assert!((ratio - l2 / l1).abs() < 1e-3, "{ratio} vs {}", l2 / l1);

    // Pocket world with 3 states: converges, residual recorded per iteration.
    let (kernel, viable, _) = kernel_of(&pocket_world());
    let f = FreedomField::compute_from_kernel(
        &kernel,
        key(viable.len()),
        FreedomFieldConfig::default(),
    )
    .unwrap();
    println!(
        "pocket world: converged={} iterations={} final residual {:.3e} ratio estimate {:?}",
        f.converged(),
        f.iterations().len(),
        f.final_residual(),
        f.eigenvalue_ratio_estimate()
    );
    assert!(f.converged());
    assert_eq!(
        f.iterations().len(),
        f.iterations().last().unwrap().iteration
    );
}

// ---------------------------------------------------------------------------------------------
// AC2: FMC with the freedom field as leaf value on hidden-physics Grid (7x5, Holdout seeds).
// ---------------------------------------------------------------------------------------------

fn detour_world() -> GridWorldConfig {
    let map =
        GridMap::from_ascii(&["#######", "#S.#.G#", "#..#..#", "#.....#", "#######"]).unwrap();
    GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    }
}

#[derive(Clone, Copy)]
struct Variant {
    label: &'static str,
    population: usize,
    horizon: usize,
    budget: usize,
    leaf: bool,
}

fn run_variant(
    world: &GridWorldConfig,
    observations: usize,
    v: Variant,
    seeds: usize,
) -> PolicyStats {
    let scale = grid_cell_scale(&world.map).unwrap();
    let mut eps = Vec::new();
    for seed in (101..).take(seeds).map(MasterSeed::new) {
        let learner = collect_hidden_experience(
            world,
            &scale,
            observations,
            seed,
            LearningPolicy::new(0.05).unwrap(),
        )
        .unwrap();
        let model = freeze_grid_model(
            &learner,
            scale.clone(),
            LearnedModelConfig::new(0.5, 2, UnseenPairPolicy::SamplePrior).unwrap(),
        )
        .unwrap();
        let mut cfg = PlanGridConfig::new(GridTask::from_world(world));
        cfg.seed = seed;
        cfg.population_size = v.population;
        cfg.horizon = v.horizon;
        cfg.budget_transitions = v.budget;
        let mut agent = PlanGrid::new(model.clone(), cfg).unwrap();
        if v.leaf {
            let field =
                Arc::new(FreedomField::compute(&model, FreedomFieldConfig::default()).unwrap());
            agent = agent.with_freedom_leaf(field, 1.0, 5.0).unwrap();
        }
        eps.push(run_plan_grid_episode(world, &agent, 30).unwrap());
    }
    let n = eps.len() as f64;
    PolicyStats {
        goal_rate: eps.iter().filter(|e| e.reached_goal).count() as f64 / n,
        failure_rate: eps.iter().filter(|e| e.failed).count() as f64 / n,
        mean_reward: eps.iter().map(|e| e.total_reward).sum::<f64>() / n,
        mean_transitions_per_decision: eps.iter().map(|e| e.planning_transitions).sum::<u64>()
            as f64
            / eps.iter().map(|e| e.ticks).sum::<u64>().max(1) as f64,
    }
}

fn hazard_world() -> GridWorldConfig {
    GridWorldConfig {
        map: gogamen_lab_grid::ablation::default_ablation_map(),
        ..GridWorldConfig::default()
    }
}

const VARIANTS: [Variant; 6] = [
    Variant {
        label: "baseline  pop16 h10 b160, no leaf",
        population: 16,
        horizon: 10,
        budget: 160,
        leaf: false,
    },
    Variant {
        label: "leaf      pop16 h10 b160",
        population: 16,
        horizon: 10,
        budget: 160,
        leaf: true,
    },
    Variant {
        label: "no leaf   pop4  h10 b40 ",
        population: 4,
        horizon: 10,
        budget: 40,
        leaf: false,
    },
    Variant {
        label: "leaf      pop4  h10 b40 ",
        population: 4,
        horizon: 10,
        budget: 40,
        leaf: true,
    },
    Variant {
        label: "no leaf   pop16 h2  b40 ",
        population: 16,
        horizon: 2,
        budget: 40,
        leaf: false,
    },
    Variant {
        label: "leaf      pop16 h2  b40 ",
        population: 16,
        horizon: 2,
        budget: 40,
        leaf: true,
    },
];

/// Quick, thresholdless smoke: the leaf agent runs deterministically within the budget cap.
#[test]
fn freedom_field_leaf_agent_is_deterministic_and_within_budget() {
    let world = hazard_world();
    let v = VARIANTS[3];
    let a = run_variant(&world, 400, v, 2);
    assert_eq!(a, run_variant(&world, 400, v, 2));
    assert!(a.mean_transitions_per_decision <= v.budget as f64);
}

/// Full report on the 6 Holdout seeds (slow in debug: run with `--ignored --nocapture`).
/// Pre-declared before looking at results: leaf weight 1.0 reward/nat, cap 5 nats; "quarter
/// budget" = 40 of 160 transitions per decision, spent as 4 walkers x horizon 10 or 16 walkers x
/// horizon 2. Nothing is tuned on these seeds and no threshold is asserted.
#[test]
#[ignore = "slow report; run explicitly with --ignored --nocapture"]
fn freedom_field_leaf_value_report_on_hidden_grid_holdout() {
    for (name, world) in [
        ("hazard map (ablation 7x5)", hazard_world()),
        ("detour map (GOG-168 7x5, no hazard)", detour_world()),
    ] {
        for obs in [800usize, 2000] {
            println!("{name}: observations = {obs} (holdout seeds 101..=106)");
            for v in VARIANTS {
                let st = run_variant(&world, obs, v, 6);
                println!(
                    "  {} | goal {:.2} fail {:.2} viability {:.2} reward {:>6.2} trans/dec {:>6.1}",
                    v.label,
                    st.goal_rate,
                    st.failure_rate,
                    1.0 - st.failure_rate,
                    st.mean_reward,
                    st.mean_transitions_per_decision
                );
            }
        }
    }
}

#[test]
fn freedom_field_leaf_rejects_field_of_another_model_version() {
    let world = detour_world();
    let scale = grid_cell_scale(&world.map).unwrap();
    let build = |n| {
        let learner = collect_hidden_experience(
            &world,
            &scale,
            n,
            MasterSeed::new(5),
            LearningPolicy::new(0.05).unwrap(),
        )
        .unwrap();
        freeze_grid_model(&learner, scale.clone(), LearnedModelConfig::default()).unwrap()
    };
    let (a, b) = (build(200), build(400));
    let field_a = Arc::new(FreedomField::compute(&a, FreedomFieldConfig::default()).unwrap());
    let cfg = PlanGridConfig::new(GridTask::from_world(&world));
    assert!(
        PlanGrid::new(b, cfg.clone())
            .unwrap()
            .with_freedom_leaf(field_a.clone(), 1.0, 5.0)
            .is_err()
    );
    assert!(
        PlanGrid::new(a.clone(), cfg.clone())
            .unwrap()
            .with_freedom_leaf(field_a.clone(), -1.0, 5.0)
            .is_err()
    );
    assert!(
        PlanGrid::new(a, cfg)
            .unwrap()
            .with_freedom_leaf(field_a, 1.0, 5.0)
            .is_ok()
    );
}
