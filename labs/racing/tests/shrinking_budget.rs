//! GOG-188 shrinking-budget tests (package `gogamen-lab-racing`).

use std::sync::OnceLock;

use gogamen_core::{Action, ActionSpace, Actuator, TabularPolicy, Terminal, Tick};
use gogamen_lab_racing::shrinking_budget::*;
use gogamen_lab_racing::{KartEnv, KinematicBicycleConfig, TrackMap};

fn report() -> &'static ShrinkingReport {
    static R: OnceLock<ShrinkingReport> = OnceLock::new();
    R.get_or_init(|| {
        let r = run_experiment(&ExperimentConfig::default()).unwrap();
        println!("{}", r.format_tables());
        r
    })
}

fn tiny() -> ExperimentConfig {
    ExperimentConfig {
        seeds: vec![5],
        initial_population: 4,
        horizon: 4,
        halvings: 2,
        episodes_per_stage: 1,
        ticks: 20,
        fixture_episodes: 1,
    }
}

#[test]
fn shrinking_budget_preregistration_constants_and_ledger() {
    assert_eq!(VIABILITY_TOLERANCE, 0.05);
    assert_eq!(SAVING_MIN, 0.25);
    assert!(PREREG.contains("-0.05") && PREREG.contains("0.25"));
    let cfg = ExperimentConfig::default();
    assert_eq!(cfg.halvings, 5);
    let pops: Vec<usize> = (0..=5).map(|s| cfg.population_at(s)).collect();
    assert_eq!(pops, vec![32, 16, 8, 4, 2, 1]);
    assert!(
        endowment_ledger(&cfg)
            .iter()
            .any(|e| e.name == "saving_min")
    );
}

#[test]
fn shrinking_budget_reflex_not_used_before_promotion() {
    // Plan-only arm never has a reflex.
    let r = report();
    for run in &r.plan_only {
        assert!(
            run.stages
                .iter()
                .all(|s| s.reflex_fraction == 0.0 && !s.reflex_active)
        );
    }
    for run in &r.distilled {
        // Stage 0 always precedes any promotion; stages without an active promoted reflex use 0.
        assert_eq!(run.stages[0].reflex_fraction, 0.0);
        for s in &run.stages {
            if !s.reflex_active {
                assert_eq!(s.reflex_fraction, 0.0, "reflex used without promotion");
            }
        }
        // A reflex can only be active after at least one approved gate attempt.
        if run.stages.iter().any(|s| s.reflex_active) {
            assert!(run.promotions > 0 && run.gates.iter().any(|g| g.promoted));
        }
    }
}

#[test]
fn shrinking_budget_every_action_goes_through_guard() {
    let r = report();
    for run in r.plan_only.iter().chain(&r.distilled) {
        for s in &run.stages {
            assert_eq!(
                s.guard_checked, s.decisions,
                "each decision checked by the guard"
            );
        }
    }
}

#[test]
fn shrinking_budget_guard_replaces_invalid_and_terminal_actions() {
    let kin = KinematicBicycleConfig::default();
    let env = KartEnv::new(kin.clone(), TrackMap::rectangular_circuit()).unwrap();
    let space: ActionSpace = Actuator::action_space(&env).clone();
    let brake = Action::continuous_in_space(vec![0.0, kin.min_accel], &space).unwrap();
    let guard = gogamen_core::ActionGuard::new(gogamen_core::SafeFallbackPolicy::ExplicitFallback(
        brake.clone(),
    ));
    // Out-of-range action (unchecked against the space) is replaced by the fallback.
    let bad = Action::continuous(vec![kin.max_steering_angle + 1.0, 0.0]).unwrap();
    let (a, intervened) =
        guarded_action(&guard, &space, bad, &brake, Tick::new(0), Terminal::Ongoing);
    assert!(intervened);
    assert_eq!(a, brake);
    // Valid action passes untouched.
    let ok = Action::continuous_in_space(vec![0.1, 1.0], &space).unwrap();
    let (a, intervened) = guarded_action(
        &guard,
        &space,
        ok.clone(),
        &brake,
        Tick::new(0),
        Terminal::Ongoing,
    );
    assert!(!intervened);
    assert_eq!(a, ok);
}

#[test]
fn shrinking_budget_gate_rejects_crashing_reflex() {
    // A table that always floors the throttle straight (index 5) drives into a wall: must be rejected.
    let mut entries = Vec::new();
    // Support every bucket in a coarse sweep so coverage is high.
    for b in 0..NUM_STATE_BUCKETS {
        entries.push((b, 5, 10.0));
    }
    let policy = TabularPolicy::from_entries(NUM_STATE_BUCKETS, NUM_ACTIONS, &entries).unwrap();
    let rec = evaluate_candidate_through_gate(
        &tiny_gate_cfg(),
        5,
        &TrackMap::rectangular_circuit(),
        0,
        policy,
    )
    .unwrap();
    assert!(!rec.promoted, "crashing reflex promoted: {rec:?}");
}

fn tiny_gate_cfg() -> ExperimentConfig {
    ExperimentConfig {
        seeds: vec![5],
        initial_population: 16,
        horizon: 8,
        halvings: 0,
        episodes_per_stage: 1,
        ticks: 200,
        fixture_episodes: 2,
    }
}

#[test]
fn shrinking_budget_empty_reflex_not_promoted_and_deterministic() {
    let empty = TabularPolicy::new(NUM_STATE_BUCKETS, NUM_ACTIONS).unwrap();
    let rec =
        evaluate_candidate_through_gate(&tiny(), 5, &TrackMap::rectangular_circuit(), 0, empty)
            .unwrap();
    assert!(!rec.promoted, "zero-coverage reflex must fail min coverage");
    let a = run_agent(&tiny(), 5, &TrackMap::rectangular_circuit(), true).unwrap();
    let b = run_agent(&tiny(), 5, &TrackMap::rectangular_circuit(), true).unwrap();
    assert_eq!(a, b);
}

#[test]
fn shrinking_budget_paired_results_report_budget_halving_and_verdict() {
    let r = report();
    let cfg = &r.config;
    for run in r.plan_only.iter().chain(&r.distilled) {
        assert_eq!(run.stages.len(), 6);
        for s in &run.stages {
            // decision-level transitions never exceed the nominal budget
            assert!(s.transitions_per_decision <= cfg.nominal_budget(s.stage) as f64 + 1e-9);
            assert!((0.0..=1.0).contains(&s.viability));
        }
    }
    // Plan-only transitions/decision are strictly non-increasing across stages (budget halves).
    for run in &r.plan_only {
        for w in run.stages.windows(2) {
            assert!(w[1].transitions_per_decision <= w[0].transitions_per_decision);
        }
    }
    // Null / negative / positive are all valid: only require a coherent, reported verdict.
    println!("VERDICT {:?}", r.verdict);
    assert!(r.saving_fraction.is_finite() && r.reflex_fraction.is_finite());
}
