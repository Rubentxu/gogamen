//! GOG-80 acceptance tests: seeded fault injection + external fail-safe policy (simulation only).
use gogamen_lab_mujoco::nav::*;

const DT: f64 = 0.01;
const SEEDS: [u64; 3] = [1, 2, 3];
/// Detection deadlines in ticks after fault onset, per domain (measured: sensor 11-14, actuator 4,
/// model 20-29; these bounds leave margin but are still tight).
const SENSOR_DEADLINE: u64 = 20;
const ACTUATOR_DEADLINE: u64 = 8;
const MODEL_DEADLINE: u64 = 60;
/// After detection the robot must be standing (< 0.02 m/s) within this many ticks.
const STOP_DEADLINE: u64 = 100;

fn limits() -> VelocityLimits {
    VelocityLimits::default()
}

fn grid() -> GridPlanner {
    GridPlanner::new(limits())
}

fn charge() -> AdversarialPlanner {
    AdversarialPlanner::new(Misbehaviour::Charge, limits())
}

fn wrong_model(gain: f64, radius_error: f64, from: u64) -> WrongModelPlanner<GridPlanner> {
    WrongModelPlanner::new(
        grid(),
        ModelError {
            gain,
            radius_error,
            from_tick: from,
        },
        limits().v_max,
        limits().w_max,
    )
}

fn policy(on: bool) -> Option<FailSafeConfig> {
    on.then(FailSafeConfig::default)
}

/// First tick after `t` at which the true speed is below 0.02 m/s.
fn standing_after(sim: &Sim<impl NavPlanner>, t: u64) -> Option<u64> {
    sim.trace()
        .windows(2)
        .find(|w| {
            w[1].tick > t
                && (w[1].true_pose[0] - w[0].true_pose[0])
                    .hypot(w[1].true_pose[1] - w[0].true_pose[1])
                    / DT
                    < 0.02
        })
        .map(|w| w[1].tick)
}

struct Measured {
    report: Report,
    detect: Option<u64>,
    standing: Option<u64>,
}

fn run<P: NavPlanner>(cfg: ScenarioConfig, planner: P, d: Domain) -> Measured {
    let mut sim = Sim::new(cfg, planner);
    let report = sim.run().unwrap();
    let detect = sim.failsafe().and_then(|f| f.first_trip(d));
    let standing = detect.and_then(|t| standing_after(&sim, t));
    Measured {
        report,
        detect,
        standing,
    }
}

// ---------------------------------------------------------------- AC1: bounded stop under faults

fn assert_bounded_stop(label: &str, m: &Measured, onset: u64, deadline: u64) {
    let detect = m
        .detect
        .unwrap_or_else(|| panic!("{label}: not detected: {:?}", m.report));
    assert!(detect >= onset, "{label}: detected before onset");
    assert!(
        detect - onset <= deadline,
        "{label}: detection took {} ticks (> {deadline})",
        detect - onset
    );
    let standing = m
        .standing
        .unwrap_or_else(|| panic!("{label}: never stopped"));
    assert!(
        standing - detect <= STOP_DEADLINE,
        "{label}: stop took {} ticks",
        standing - detect
    );
    assert!(
        m.report.min_clearance > 0.0 && m.report.outcome != Outcome::Collision,
        "{label}: {:?}",
        m.report
    );
}

fn sensor_modes() -> [SensorFaultMode; 2] {
    [SensorFaultMode::OldTimestamp, SensorFaultMode::FrozenValues]
}

#[test]
fn sensor_faults_trigger_bounded_stop_for_any_planner() {
    for seed in SEEDS {
        for mode in sensor_modes() {
            let plan = FaultPlan::sensor_seeded(seed, mode);
            let onset = plan.sensor.unwrap().from_tick;
            let mk = || blocked_corridor(seed, plan, policy(true));
            let l = limits();
            let hang = AdversarialPlanner::new(Misbehaviour::HangAfter(100), l);
            let oob = AdversarialPlanner::new(Misbehaviour::OutOfBounds, l);
            let tag = |n: &str| format!("sensor {mode:?} seed {seed} {n}");
            assert_bounded_stop(
                &tag("grid"),
                &run(mk(), grid(), Domain::Sensor),
                onset,
                SENSOR_DEADLINE,
            );
            assert_bounded_stop(
                &tag("charge"),
                &run(mk(), charge(), Domain::Sensor),
                onset,
                SENSOR_DEADLINE,
            );
            assert_bounded_stop(
                &tag("hang"),
                &run(mk(), hang, Domain::Sensor),
                onset,
                SENSOR_DEADLINE,
            );
            assert_bounded_stop(
                &tag("oob"),
                &run(mk(), oob, Domain::Sensor),
                onset,
                SENSOR_DEADLINE,
            );
        }
    }
}

#[test]
fn actuator_timeout_triggers_bounded_stop_with_estop_for_any_planner() {
    let modes = [
        (ActuatorFaultMode::Drop, Some(60)),
        (ActuatorFaultMode::Drop, None),
        (ActuatorFaultMode::Delay { ticks: 20 }, None),
    ];
    for seed in SEEDS {
        for (mode, dur) in modes {
            let plan = FaultPlan::actuator_seeded(seed, mode, dur);
            let onset = plan.actuator.unwrap().from_tick;
            let mk = || blocked_corridor(seed, plan, policy(true));
            let l = limits();
            let hang = AdversarialPlanner::new(Misbehaviour::HangAfter(100), l);
            let tag = |n: &str| format!("actuator {mode:?}/{dur:?} seed {seed} {n}");
            assert_bounded_stop(
                &tag("grid"),
                &run(mk(), grid(), Domain::Actuator),
                onset,
                ACTUATOR_DEADLINE,
            );
            assert_bounded_stop(
                &tag("charge"),
                &run(mk(), charge(), Domain::Actuator),
                onset,
                ACTUATOR_DEADLINE,
            );
            assert_bounded_stop(
                &tag("hang"),
                &run(mk(), hang, Domain::Actuator),
                onset,
                ACTUATOR_DEADLINE,
            );
        }
    }
}

#[test]
fn wrong_model_triggers_reduced_speed_fallback() {
    for seed in SEEDS {
        for gain in [0.5, 1.8] {
            let from = FaultPlan::onset(seed, 120);
            let cfg = blocked_corridor(seed, FaultPlan::none(), policy(true));
            let mut sim = Sim::new(cfg, wrong_model(gain, 0.0, from));
            let r = sim.run().unwrap();
            let detect = sim
                .failsafe()
                .unwrap()
                .first_trip(Domain::Model)
                .expect("model fault detected");
            assert!(
                detect >= from && detect - from <= MODEL_DEADLINE,
                "gain {gain} seed {seed}: {detect} vs {from}"
            );
            assert!(
                r.min_clearance > 0.0 && r.outcome != Outcome::Collision,
                "{r:?}"
            );
            // Fallback table says ReduceSpeed(0.3): after the ramp-down the speed stays capped.
            let cap = match FailSafeConfig::default().fallback[1 + 1] {
                Fallback::ReduceSpeed(v) => v,
                other => panic!("unexpected model fallback {other:?}"),
            };
            let v = max_speed_from(sim.trace(), detect + 60, DT);
            assert!(
                v <= cap + 0.02,
                "gain {gain} seed {seed}: speed {v} above cap {cap}"
            );
        }
    }
}

// ---------------------------------------------------------------- AC1: no false stops, negatives

#[test]
fn nominal_has_no_false_stops() {
    let mut runs = 0;
    let mut false_ticks = 0;
    let mut total_ticks = 0;
    for seed in 1..=20u64 {
        // Clean corridor, mild-drift variant and blocked corridor; all with the policy enabled.
        let mut drift = ScenarioConfig::corridor(seed);
        drift.odom.v_scale = 1.02;
        drift.odom.w_bias0 = 0.005;
        let mut cfgs = vec![ScenarioConfig::corridor(seed), drift];
        cfgs.push(blocked_corridor(seed, FaultPlan::none(), None));
        for (k, mut cfg) in cfgs.into_iter().enumerate() {
            cfg.failsafe = Some(FailSafeConfig::default());
            let mut sim = Sim::new(cfg, grid());
            let r = sim.run().unwrap();
            let fs = sim.failsafe().unwrap();
            assert!(
                fs.events().is_empty(),
                "seed {seed} scenario {k}: false trip {:?} residual {}",
                fs.events(),
                fs.residual()
            );
            assert!(r.min_clearance > 0.0);
            false_ticks += fs.active_ticks();
            total_ticks += fs.ticks();
            runs += 1;
        }
    }
    assert_eq!(runs, 60);
    assert_eq!(
        false_ticks, 0,
        "false-stop rate {false_ticks}/{total_ticks} ticks"
    );
    // The nominal corridor still reaches the goal with the policy on.
    let mut c = ScenarioConfig::corridor(1);
    c.failsafe = Some(FailSafeConfig::default());
    assert_eq!(Sim::new(c, grid()).run().unwrap().outcome, Outcome::Reached);
}

#[test]
fn negative_control_without_policy_the_faults_cause_collision_or_violation() {
    for seed in SEEDS {
        // Frozen sensor values and a permanently dead actuator => collision with the wall.
        let p = FaultPlan::sensor_seeded(seed, SensorFaultMode::FrozenValues);
        let r = run(
            blocked_corridor(seed, p, policy(false)),
            charge(),
            Domain::Sensor,
        )
        .report;
        assert_eq!(r.outcome, Outcome::Collision, "sensor seed {seed}: {r:?}");
        let p = FaultPlan::actuator_seeded(seed, ActuatorFaultMode::Drop, None);
        let r = run(
            blocked_corridor(seed, p, policy(false)),
            charge(),
            Domain::Actuator,
        )
        .report;
        assert_eq!(r.outcome, Outcome::Collision, "actuator seed {seed}: {r:?}");
        // Wrong model: no collision (the independent safety stop still works), but the speed cap
        // the policy enforces is violated.
        let from = FaultPlan::onset(seed, 120);
        let mut sim = Sim::new(
            blocked_corridor(seed, FaultPlan::none(), policy(false)),
            wrong_model(0.5, 0.0, from),
        );
        sim.run().unwrap();
        let v = max_speed_from(sim.trace(), from + 100, DT);
        assert!(
            v > 0.5,
            "seed {seed}: model fault without policy still capped? {v}"
        );
        assert!(sim.failsafe().is_none());
    }
}

#[test]
fn faulty_runs_are_deterministic() {
    let p = FaultPlan::sensor_seeded(2, SensorFaultMode::FrozenDithered { amplitude: 0.01 });
    let h = |seed| {
        let mut s = Sim::new(blocked_corridor(seed, p, policy(true)), charge());
        s.run().unwrap().trace_hash
    };
    assert_eq!(h(2), h(2));
    assert_ne!(h(2), h(3));
}

// ---------------------------------------------------------------- latch and re-arm

#[test]
fn latch_holds_after_fault_ends_and_rearm_is_explicit() {
    let seed = 1;
    let plan = FaultPlan::actuator_seeded(seed, ActuatorFaultMode::Drop, Some(60));
    let a = plan.actuator.unwrap();
    let mut sim = Sim::new(blocked_corridor(seed, plan, policy(true)), grid());
    // During the fault: latched, and re-arm is refused while the condition persists.
    sim.run_ticks(a.from_tick + 30).unwrap();
    assert!(sim.failsafe().unwrap().is_latched(Domain::Actuator));
    assert_eq!(
        sim.rearm_failsafe(),
        Err(RearmRefused::StillFaulted(Domain::Actuator))
    );
    assert!(sim.failsafe().unwrap().is_latched(Domain::Actuator));
    // Long after the fault ended, the latch still holds and the robot is standing.
    sim.run_ticks(200).unwrap();
    assert!(sim.failsafe().unwrap().is_latched(Domain::Actuator));
    let n = sim.trace().len();
    let (a0, a1) = (sim.trace()[n - 1].true_pose, sim.trace()[n - 50].true_pose);
    assert!((a0[0] - a1[0]).hypot(a0[1] - a1[1]) < 1e-3);
    // Explicit re-arm now succeeds; the policy does not re-trip afterwards.
    assert_eq!(sim.rearm_failsafe(), Ok(()));
    assert!(!sim.failsafe().unwrap().any_latched());
    let before = sim.failsafe().unwrap().events().len();
    sim.run_ticks(100).unwrap();
    assert_eq!(sim.failsafe().unwrap().events().len(), before);
}

#[test]
fn fallback_table_is_explicit() {
    let c = FailSafeConfig::default();
    assert_eq!(c.fallback[0], Fallback::Stop); // sensor
    assert_eq!(c.fallback[1], Fallback::StopAndEStop); // actuator
    assert_eq!(c.fallback[2], Fallback::ReduceSpeed(0.3)); // model
    assert_eq!(
        Domain::ALL,
        [Domain::Sensor, Domain::Actuator, Domain::Model]
    );
}

// ---------------------------------------------------------------- AC2: residual risk

#[test]
fn residual_risk_fault_inside_the_detection_thresholds_is_not_detected() {
    let seed = 1;
    // Actuator delay within the echo tolerance.
    let plan = FaultPlan::actuator_seeded(seed, ActuatorFaultMode::Delay { ticks: 3 }, None);
    let mut s = Sim::new(blocked_corridor(seed, plan, policy(true)), grid());
    s.run().unwrap();
    assert!(s.failsafe().unwrap().events().is_empty());
    // Mild model gain error whose residual stays below the threshold.
    let from = FaultPlan::onset(seed, 120);
    let mut s = Sim::new(
        blocked_corridor(seed, FaultPlan::none(), policy(true)),
        wrong_model(0.9, 0.0, from),
    );
    s.run().unwrap();
    assert!(s.failsafe().unwrap().first_trip(Domain::Model).is_none());
    // Sensor latency just under the age threshold.
    let mut c = blocked_corridor(seed, FaultPlan::none(), policy(true));
    c.range.latency_ticks = 9; // age <= 9 + 4 < 15
    let mut s = Sim::new(c, grid());
    s.run().unwrap();
    assert!(s.failsafe().unwrap().first_trip(Domain::Sensor).is_none());
}

#[test]
fn residual_risk_heavy_odometry_drift_causes_a_benign_false_model_trip() {
    // 5 % odometry scale error + 0.02 rad/s gyro bias pushes the nominal residual above the
    // threshold (see README). The fallback is only a speed cap, so safety is kept, but this is a
    // false positive of the model detector and it is documented as such.
    let mut c = ScenarioConfig::corridor(1);
    c.odom.v_scale = 1.05;
    c.odom.w_bias0 = 0.02;
    c.failsafe = Some(FailSafeConfig::default());
    let mut s = Sim::new(c, grid());
    let r = s.run().unwrap();
    let ev = s.failsafe().unwrap().events();
    assert!(
        ev.iter().all(|e| e.domain == Domain::Model) && !ev.is_empty(),
        "{ev:?}"
    );
    assert!(r.min_clearance > 0.0, "{r:?}");
}

#[test]
fn residual_risk_radius_error_is_invisible_to_the_residual() {
    for seed in SEEDS {
        let from = FaultPlan::onset(seed, 120);
        let mut s = Sim::new(
            blocked_corridor(seed, FaultPlan::none(), policy(true)),
            wrong_model(1.0, 0.3, from),
        );
        let r = s.run().unwrap();
        assert!(s.failsafe().unwrap().events().is_empty(), "seed {seed}");
        // Only the independent safety stop (not this policy) protects here.
        assert!(r.min_clearance > 0.0, "seed {seed}: {r:?}");
    }
}

#[test]
fn residual_risk_dithered_frozen_sensor_defeats_the_repeat_detector() {
    for seed in SEEDS {
        let p = FaultPlan::sensor_seeded(seed, SensorFaultMode::FrozenDithered { amplitude: 0.01 });
        let m = run(
            blocked_corridor(seed, p, policy(true)),
            charge(),
            Domain::Sensor,
        );
        assert!(m.detect.is_none(), "seed {seed}: unexpectedly detected");
        assert_eq!(
            m.report.outcome,
            Outcome::Collision,
            "seed {seed}: {:?}",
            m.report
        );
    }
}

#[test]
fn residual_risk_broken_estop_path_is_not_covered() {
    for seed in SEEDS {
        let mut p = FaultPlan::actuator_seeded(seed, ActuatorFaultMode::Drop, None);
        p.estop_broken = true;
        let m = run(
            blocked_corridor(seed, p, policy(true)),
            charge(),
            Domain::Actuator,
        );
        assert!(m.detect.is_some(), "the fault is detected...");
        assert_eq!(
            m.report.outcome,
            Outcome::Collision,
            "...but the stop itself failed: {:?}",
            m.report
        );
    }
}

#[test]
fn simultaneous_faults_are_detected_in_this_configuration_but_not_proven_in_general() {
    for seed in SEEDS {
        let mut p = FaultPlan::actuator_seeded(seed, ActuatorFaultMode::Drop, Some(60));
        p.sensor = Some(SensorFault {
            from_tick: FaultPlan::onset(seed, 120),
            until_tick: None,
            mode: SensorFaultMode::FrozenValues,
        });
        let from = FaultPlan::onset(seed, 120);
        let mut s = Sim::new(
            blocked_corridor(seed, p, policy(true)),
            wrong_model(0.5, 0.0, from),
        );
        let r = s.run().unwrap();
        let fs = s.failsafe().unwrap();
        // The model fault is masked here: the sensor/actuator stop leaves no motion to compare.
        assert!(fs.is_latched(Domain::Sensor), "seed {seed}");
        assert!(fs.is_latched(Domain::Actuator), "seed {seed}");
        assert!(r.min_clearance > 0.0, "seed {seed}: {r:?}");
    }
}

// ---------------------------------------------------------------- AC3: not a certification, simulation only

const README: &str = include_str!("../README.md");
const CARGO: &str = include_str!("../Cargo.toml");

#[test]
fn readme_states_this_is_not_certification_and_not_hardware_enabling() {
    let lc = README.to_lowercase();
    assert!(lc.contains("not a safety certification"));
    assert!(lc.contains("does not enable real hardware"));
    assert!(lc.contains("simulation only"));
    // Every mention of a safety standard must be a disclaimer.
    for line in README.lines() {
        let l = line.to_lowercase();
        let mentions = [
            "sil ",
            "sil-",
            "iso 13849",
            "iec 61508",
            "iso 26262",
            "iec 62061",
        ]
        .iter()
        .any(|t| l.contains(t));
        if mentions {
            assert!(
                l.contains("no claim") || l.contains("not ") || l.contains("no sil"),
                "README line makes a standards mention without a disclaimer: {line}"
            );
        }
    }
}

#[test]
fn no_hardware_dependencies_features_or_code_paths() {
    const { assert!(SIMULATION_ONLY) };
    // Dependencies: only the core crate.
    let deps: Vec<&str> = CARGO
        .split("[dependencies]")
        .nth(1)
        .unwrap()
        .lines()
        .skip(1)
        .take_while(|l| !l.trim_start().starts_with('['))
        .filter(|l| l.contains('=') && !l.trim_start().starts_with('#'))
        .collect();
    assert!(
        deps.len() == 1 && deps[0].starts_with("gogamen-core"),
        "{deps:?}"
    );
    // Features: only `default` and the pre-existing MuJoCo Python sidecar (no hardware feature).
    let feats: Vec<&str> = CARGO
        .split("[features]")
        .nth(1)
        .unwrap()
        .split("[dependencies]")
        .next()
        .unwrap()
        .lines()
        .filter(|l| l.contains('=') && !l.trim_start().starts_with('#'))
        .collect();
    assert_eq!(feats.len(), 2, "{feats:?}");
    for f in &feats {
        let name = f.split('=').next().unwrap().trim();
        assert!(
            name == "default" || name == "mujoco-sidecar",
            "unexpected feature {name}"
        );
    }
    // Source of the nav stack touches no I/O or hardware APIs.
    let sources = [
        include_str!("../src/nav/faults.rs"),
        include_str!("../src/nav/failsafe.rs"),
        include_str!("../src/nav/sim.rs"),
        include_str!("../src/nav/lowlevel.rs"),
        include_str!("../src/nav/sensors.rs"),
    ];
    let forbidden = [
        "std::net",
        "std::fs",
        "std::process",
        "std::os::",
        "serialport",
        "tokio",
        "ros2",
        "rclrs",
        "socketcan",
        "gpio",
        "/dev/",
        "TcpStream",
        "UdpSocket",
        "extern \"C\"",
    ];
    for (i, s) in sources.iter().enumerate() {
        // Only look at non-test, non-comment code.
        let code: String = s
            .split("#[cfg(test)]")
            .next()
            .unwrap()
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for t in forbidden {
            assert!(!code.contains(t), "source #{i} references `{t}`");
        }
    }
}
