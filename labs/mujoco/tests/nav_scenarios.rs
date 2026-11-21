//! GOG-65 acceptance tests: closed-loop navigation on the deterministic Rust backend.
use gogamen_lab_mujoco::nav::*;

fn grid() -> GridPlanner {
    GridPlanner::new(VelocityLimits::default())
}

fn run(cfg: ScenarioConfig, planner: impl NavPlanner) -> Report {
    Sim::new(cfg, planner).run().unwrap()
}

// ---------------------------------------------------------------- AC1

#[test]
fn nominal_scenario_reaches_goal_without_collision() {
    for seed in 1..=3 {
        let r = run(ScenarioConfig::corridor(seed), grid());
        assert_eq!(r.outcome, Outcome::Reached, "seed {seed}: {r:?}");
        assert!(r.min_clearance > 0.0, "seed {seed}: {r:?}");
        assert!(r.true_goal_error < 0.3, "seed {seed}: {r:?}");
    }
}

#[test]
fn odometry_drift_is_corrected_by_map_fixes_or_fails_safe() {
    for seed in 1..=3 {
        let mut c = ScenarioConfig::corridor(seed);
        c.odom.v_scale = 1.05;
        c.odom.w_bias0 = 0.02;
        let with_fix = run(c.clone(), grid());
        assert_eq!(
            with_fix.outcome,
            Outcome::Reached,
            "seed {seed}: {with_fix:?}"
        );
        c.fusion.fix_gain = 0.0;
        let no_fix = run(c, grid());
        assert_ne!(
            no_fix.outcome,
            Outcome::Collision,
            "seed {seed}: {no_fix:?}"
        );
    }
}

#[test]
fn shove_and_kick_complete_or_fail_safe() {
    for seed in 1..=3 {
        let mut c = ScenarioConfig::corridor(seed);
        c.perturbations = vec![
            Perturbation::Shove {
                tick: 150,
                dx: 0.0,
                dy: -0.3,
                dyaw: 0.3,
            },
            Perturbation::Kick {
                tick: 300,
                dv: 0.8,
                dw: 1.0,
            },
        ];
        let r = run(c, grid());
        assert!(
            matches!(
                r.outcome,
                Outcome::Reached | Outcome::SafeStop | Outcome::Timeout
            ),
            "seed {seed}: {r:?}"
        );
        assert!(r.min_clearance > 0.0, "seed {seed}: {r:?}");
    }
}

#[test]
fn sensor_latency_degrades_progress_but_never_safety() {
    for seed in 1..=3 {
        let mut times = Vec::new();
        for lat in [0u64, 10, 30, 45, 80] {
            let mut c = ScenarioConfig::corridor(seed);
            c.range.latency_ticks = lat;
            let r = run(c, grid());
            assert!(r.min_clearance > 0.0, "seed {seed} latency {lat}: {r:?}");
            times.push((lat, r));
        }
        let (_, base) = &times[0];
        assert_eq!(base.outcome, Outcome::Reached);
        // Moderate latency is slower than no latency (or does not finish in the budget).
        let (_, mid) = &times[1];
        assert!(
            mid.outcome != Outcome::Reached || mid.ticks > base.ticks,
            "seed {seed}: {mid:?} vs {base:?}"
        );
        // Stale sensing (age above `stale_ticks`) => full stop, never moving.
        for (lat, r) in &times[3..] {
            assert_eq!(
                r.outcome,
                Outcome::SafeStop,
                "seed {seed} latency {lat}: {r:?}"
            );
            assert!(r.max_speed < 0.01, "seed {seed} latency {lat}: {r:?}");
        }
    }
}

#[test]
fn sensor_dropout_mid_run_stops_safely() {
    for seed in 1..=3 {
        let mut c = ScenarioConfig::corridor(seed);
        c.range.dropout_from_tick = Some(200);
        let mut sim = Sim::new(c, grid());
        let r = sim.run().unwrap();
        assert_eq!(r.outcome, Outcome::SafeStop, "seed {seed}: {r:?}");
        assert!(r.min_clearance > 0.0);
        let last = sim.trace().last().unwrap();
        assert!(last.safety && last.cmd == [0.0, 0.0]);
    }
}

// ---------------------------------------------------------------- AC2

#[test]
fn bounded_command_clamps_any_input() {
    let l = VelocityLimits::default();
    for (v, w) in [
        (1e9, -1e9),
        (f64::NAN, 3.0),
        (f64::INFINITY, f64::NEG_INFINITY),
        (-5.0, 0.2),
        (0.3, -0.4),
    ] {
        let c = BoundedCommand::new(v, w, &l);
        assert!(
            c.v().abs() <= l.v_max && c.w().abs() <= l.w_max,
            "{v} {w} -> {c:?}"
        );
        assert!(c.v().is_finite() && c.w().is_finite());
    }
    assert_eq!(BoundedCommand::new(0.3, -0.4, &l).to_vec(), vec![0.3, -0.4]);
}

#[test]
fn planner_output_is_always_bounded() {
    let l = VelocityLimits::default();
    let mut sim = Sim::new(ScenarioConfig::corridor(1), grid());
    sim.run().unwrap();
    for row in sim.trace() {
        assert!(
            row.cmd[0].abs() <= l.v_max && row.cmd[1].abs() <= l.w_max,
            "{row:?}"
        );
    }
}

#[test]
fn independent_stop_contains_adversarial_planners() {
    let l = VelocityLimits::default();
    for seed in 1..=3 {
        for m in [
            Misbehaviour::OutOfBounds,
            Misbehaviour::Nan,
            Misbehaviour::WrongDim,
            Misbehaviour::Hang,
            Misbehaviour::Charge,
            Misbehaviour::HangAfter(150),
        ] {
            let mut sim = Sim::new(
                ScenarioConfig::corridor(seed),
                AdversarialPlanner::new(m, l),
            );
            let r = sim.run().unwrap();
            assert_ne!(r.outcome, Outcome::Collision, "seed {seed} {m:?}: {r:?}");
            assert!(r.min_clearance > 0.0, "seed {seed} {m:?}: {r:?}");
            // Real (physical) speed stays within limits plus small controller overshoot.
            assert!(
                r.max_speed <= l.v_max * 1.05 && r.max_yaw_rate <= l.w_max * 1.1,
                "seed {seed} {m:?}: {r:?}"
            );
            for row in sim.trace() {
                assert!(row.cmd[0].abs() <= l.v_max && row.cmd[1].abs() <= l.w_max);
            }
        }
    }
}

#[test]
fn invalid_or_stale_proposals_are_replaced_by_guard_fallback() {
    let l = VelocityLimits::default();
    for m in [
        Misbehaviour::OutOfBounds,
        Misbehaviour::Nan,
        Misbehaviour::WrongDim,
        Misbehaviour::Hang,
    ] {
        let mut sim = Sim::new(ScenarioConfig::corridor(1), AdversarialPlanner::new(m, l));
        let r = sim.run().unwrap();
        assert!(r.fallbacks > 0, "{m:?}: {r:?}");
        assert_eq!(r.outcome, Outcome::SafeStop, "{m:?}: {r:?}");
        assert!(r.max_speed < 0.01, "{m:?}: {r:?}");
    }
    // A planner that dies mid-run: its last proposal ages out and the robot is stopped.
    let mut sim = Sim::new(
        ScenarioConfig::corridor(1),
        AdversarialPlanner::new(Misbehaviour::HangAfter(150), l),
    );
    let r = sim.run().unwrap();
    assert_eq!(r.outcome, Outcome::SafeStop);
    assert!(r.fallbacks > 100);
    let last = sim.trace().last().unwrap();
    assert!(last.true_pose[0] < 4.5 && last.cmd == [0.0, 0.0]);
}

#[test]
fn planner_budget_is_enforced_per_decision() {
    // Exhausting the core PlanBudget (transitions) makes the planner return no proposal; the guard
    // then stops the robot rather than letting it run on an unchecked plan.
    let mut c = ScenarioConfig::corridor(1);
    c.planner_budget = 1;
    let r = run(c, grid());
    assert_ne!(r.outcome, Outcome::Collision, "{r:?}");
    assert_ne!(r.outcome, Outcome::Reached, "{r:?}");
}

// ---------------------------------------------------------------- AC3

#[test]
fn trace_records_prediction_minus_physics_per_tick() {
    let mut sim = Sim::new(ScenarioConfig::corridor(2), grid());
    let r = sim.run().unwrap();
    let t = sim.trace();
    assert_eq!(t.len() as u64, r.ticks);
    let mut nonzero = false;
    for (i, row) in t.iter().enumerate() {
        assert_eq!(row.tick, i as u64 + 1);
        // Bit-exact reconstruction of the recorded difference from the recorded poses.
        let d = row.recompute_diff();
        for (k, (a, b)) in d.iter().zip(&row.diff).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "tick {} axis {k}", row.tick);
        }
        assert_eq!(d, pose_diff(row.pred_pose, row.true_pose));
        nonzero |= row.diff.iter().any(|x| x.abs() > 1e-6);
        assert!(row.diff[2].abs() <= std::f64::consts::PI);
    }
    assert!(
        nonzero,
        "the planner model should differ measurably from the physics"
    );
}

#[test]
fn same_seed_gives_bit_identical_trace_and_other_seed_differs() {
    let a = run(ScenarioConfig::corridor(7), grid());
    let b = run(ScenarioConfig::corridor(7), grid());
    let c = run(ScenarioConfig::corridor(8), grid());
    assert_eq!(a.trace_hash, b.trace_hash);
    assert_eq!(a, b);
    assert_ne!(a.trace_hash, c.trace_hash);
}

#[test]
fn snapshot_restore_reproduces_the_trace() {
    let mut sim = Sim::new(ScenarioConfig::corridor(3), grid());
    sim.run_ticks(300).unwrap();
    let snap = sim.snapshot().unwrap();
    sim.run_ticks(500).unwrap();
    let first: Vec<TraceRow> = sim.trace().to_vec();
    let h1 = trace_hash(&first);
    sim.restore(&snap).unwrap();
    assert_eq!(sim.tick(), 300);
    sim.run_ticks(500).unwrap();
    assert_eq!(trace_hash(sim.trace()), h1);
    assert_eq!(sim.trace(), &first[..]);
}
