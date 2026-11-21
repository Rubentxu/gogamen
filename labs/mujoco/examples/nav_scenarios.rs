//! Navigation scenarios on the Rust backend. Run: `cargo run --release -p gogamen-lab-mujoco --example nav_scenarios`
use gogamen_lab_mujoco::nav::*;

fn row(name: &str, cfg: ScenarioConfig, planner: impl NavPlanner) {
    let mut sim = Sim::new(cfg, planner);
    let r = sim.run().unwrap();
    println!(
        "{:<34} {:?} t={:>5.2}s goal_err={:.3} min_clear={:>6.3} guard_fb={:>4} safety_ticks={:>4} max_v={:.2} max_pred_err={:.3}",
        name,
        r.outcome,
        r.ticks as f64 * 0.01,
        r.true_goal_error,
        r.min_clearance,
        r.fallbacks,
        r.safety_ticks,
        r.max_speed,
        r.max_pred_err
    );
}

fn main() {
    let l = VelocityLimits::default();
    let g = || GridPlanner::new(l);
    for seed in 1..=3u64 {
        println!("--- seed {seed}");
        row("nominal", ScenarioConfig::corridor(seed), g());
        for lat in [0u64, 10, 30, 45, 80] {
            let mut c = ScenarioConfig::corridor(seed);
            c.range.latency_ticks = lat;
            row(&format!("latency {} ms", lat * 10), c, g());
        }
        let mut c = ScenarioConfig::corridor(seed);
        c.odom.v_scale = 1.05;
        c.odom.w_bias0 = 0.02;
        row("drift (scale 1.05, bias 0.02)", c, g());
        let mut c = ScenarioConfig::corridor(seed);
        c.odom.v_scale = 1.05;
        c.odom.w_bias0 = 0.02;
        c.fusion.fix_gain = 0.0;
        row("drift, no map fix", c, g());
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
        row("shove + kick", c, g());
        let mut c = ScenarioConfig::corridor(seed);
        c.range.dropout_from_tick = Some(200);
        row("sensor dropout @2s", c, g());
        for m in [
            Misbehaviour::OutOfBounds,
            Misbehaviour::Nan,
            Misbehaviour::WrongDim,
            Misbehaviour::Hang,
            Misbehaviour::Charge,
            Misbehaviour::HangAfter(150),
        ] {
            row(
                &format!("adversarial {m:?}"),
                ScenarioConfig::corridor(seed),
                AdversarialPlanner::new(m, l),
            );
        }
    }
}
