//! GOG-80 fault injection table. Simulation only.
//! Run: `cargo run --release -p gogamen-lab-mujoco --example fault_injection`
use gogamen_lab_mujoco::nav::*;

const DT: f64 = 0.01;

fn grid() -> GridPlanner {
    GridPlanner::new(VelocityLimits::default())
}

fn wrong_model(gain: f64, radius_error: f64, from: u64) -> WrongModelPlanner<GridPlanner> {
    let l = VelocityLimits::default();
    WrongModelPlanner::new(
        grid(),
        ModelError {
            gain,
            radius_error,
            from_tick: from,
        },
        l.v_max,
        l.w_max,
    )
}

struct Row {
    outcome: Outcome,
    onset: Option<u64>,
    detect: Option<u64>,
    still_ticks: Option<u64>,
    min_clear: f64,
    max_v_after: f64,
    fs_active: u64,
    ticks: u64,
}

fn execute<P: NavPlanner>(cfg: ScenarioConfig, planner: P, onset: Option<u64>, d: Domain) -> Row {
    let mut sim = Sim::new(cfg, planner);
    let r = sim.run().unwrap();
    let detect = sim.failsafe().and_then(|f| f.first_trip(d));
    let still_ticks = detect.and_then(|t| {
        sim.trace()
            .windows(2)
            .find(|w| {
                w[1].tick > t
                    && (w[1].true_pose[0] - w[0].true_pose[0])
                        .hypot(w[1].true_pose[1] - w[0].true_pose[1])
                        / DT
                        < 0.02
            })
            .map(|w| w[1].tick - t)
    });
    Row {
        outcome: r.outcome,
        onset,
        detect,
        still_ticks,
        min_clear: r.min_clearance,
        max_v_after: max_speed_from(sim.trace(), onset.unwrap_or(0) + 110, DT),
        fs_active: sim.failsafe().map_or(0, |f| f.active_ticks()),
        ticks: r.ticks,
    }
}

fn print(label: &str, seed: u64, policy: bool, r: &Row) {
    let opt = |o: Option<u64>| o.map_or("-".to_string(), |v| v.to_string());
    println!(
        "{:<34} {:>4} {:<4} {:<9?} onset={:>4} detect={:>4} dt_ticks={:>4} still_after={:>4} min_clear={:>6.3} max_v_after(+110)={:>5.2} fs_ticks={:>4} ticks={}",
        label,
        seed,
        if policy { "on" } else { "off" },
        r.outcome,
        opt(r.onset),
        opt(r.detect),
        match (r.onset, r.detect) {
            (Some(o), Some(d)) => (d - o).to_string(),
            _ => "-".into(),
        },
        opt(r.still_ticks),
        r.min_clear,
        r.max_v_after,
        r.fs_active,
        r.ticks
    );
}

fn main() {
    println!(
        "label                              seed pol  outcome   onset detect dt_ticks still_after min_clear max_v_after fs_ticks ticks"
    );
    let fs = |on: bool| on.then(FailSafeConfig::default);
    for seed in 1..=3u64 {
        for policy in [true, false] {
            let sensor = [
                ("sensor old-timestamp", SensorFaultMode::OldTimestamp),
                ("sensor frozen-values", SensorFaultMode::FrozenValues),
                (
                    "sensor frozen+dither (undetected)",
                    SensorFaultMode::FrozenDithered { amplitude: 0.01 },
                ),
            ];
            for (name, mode) in sensor {
                let plan = FaultPlan::sensor_seeded(seed, mode);
                let onset = plan.sensor.map(|s| s.from_tick);
                let cfg = blocked_corridor(seed, plan, fs(policy));
                print(
                    name,
                    seed,
                    policy,
                    &execute(
                        cfg,
                        AdversarialPlanner::new(Misbehaviour::Charge, VelocityLimits::default()),
                        onset,
                        Domain::Sensor,
                    ),
                );
            }
            let act = [
                ("actuator drop 60 ticks", ActuatorFaultMode::Drop, Some(60)),
                ("actuator drop permanent", ActuatorFaultMode::Drop, None),
                (
                    "actuator delay 20 ticks",
                    ActuatorFaultMode::Delay { ticks: 20 },
                    None,
                ),
            ];
            for (name, mode, dur) in act {
                let plan = FaultPlan::actuator_seeded(seed, mode, dur);
                let onset = plan.actuator.map(|a| a.from_tick);
                let cfg = blocked_corridor(seed, plan, fs(policy));
                print(
                    name,
                    seed,
                    policy,
                    &execute(
                        cfg,
                        AdversarialPlanner::new(Misbehaviour::Charge, VelocityLimits::default()),
                        onset,
                        Domain::Actuator,
                    ),
                );
            }
            let from = FaultPlan::onset(seed, 120);
            for (name, g, rad) in [
                ("model gain 0.5", 0.5, 0.0),
                ("model gain 1.8", 1.8, 0.0),
                ("model radius -0.3 (undetected)", 1.0, 0.3),
            ] {
                let cfg = blocked_corridor(seed, FaultPlan::none(), fs(policy));
                print(
                    name,
                    seed,
                    policy,
                    &execute(cfg, wrong_model(g, rad, from), Some(from), Domain::Model),
                );
            }
        }
        let r = execute(
            blocked_corridor(seed, FaultPlan::none(), Some(FailSafeConfig::default())),
            grid(),
            None,
            Domain::Sensor,
        );
        print("nominal (no fault)", seed, true, &r);
    }
}
