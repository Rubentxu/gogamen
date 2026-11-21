//! Classic PD vs budgeted sampling planner. Run:
//! `cargo run --release -p gogamen-lab-mujoco --example compare_control`
//! (add `--features mujoco-sidecar` to also run the MuJoCo backend).
use gogamen_lab_mujoco::*;

const GOALS: [(f64, f64); 3] = [(2.0, 1.5), (-1.5, 2.0), (0.5, -2.5)];
const DECISIONS: usize = 80; // 8 s of simulated time
const SEG_STEPS: usize = 10; // 0.1 s control period

fn report(port: &mut dyn ContinuousPhysicsEnginePort, budgets: &[usize], seeds: &[u64]) {
    println!("\nbackend: {}", port.name());
    println!(
        "{:<26} {:>12} {:>10} {:>10} {:>12} {:>12} {:>9}",
        "controller", "goal", "final_err", "reach_t", "lat_mean_us", "lat_max_us", "sims/dec"
    );
    for (gx, gy) in GOALS {
        let goal = Goal { x: gx, y: gy };
        let mut rows = vec![
            run_episode(
                port,
                &mut PdController::default(),
                goal,
                DECISIONS,
                SEG_STEPS,
            )
            .unwrap(),
        ];
        for &b in budgets {
            for &s in seeds {
                let mut p = SamplingPlanner::new(s, b, 6, SEG_STEPS);
                let mut r = run_episode(port, &mut p, goal, DECISIONS, SEG_STEPS).unwrap();
                r.controller = format!("sampler B={b} seed={s}");
                rows.push(r);
            }
        }
        for r in rows {
            println!(
                "{:<26} {:>12} {:>10.4} {:>10} {:>12.1} {:>12.1} {:>9}",
                r.controller,
                format!("({gx},{gy})"),
                r.final_error,
                r.reach_time.map_or("-".into(), |t| format!("{t:.2}s")),
                r.latency_mean_us,
                r.latency_max_us,
                r.sims_per_decision_max
            );
        }
    }
}

fn main() {
    let mut rust = UnicycleEngine::default();
    report(&mut rust, &[16, 64, 256], &[1, 2, 3]);

    #[cfg(feature = "mujoco-sidecar")]
    {
        use gogamen_lab_mujoco::sidecar::{MujocoSidecarEngine, probe};
        match probe() {
            Ok(v) => {
                println!("\nmujoco {v} sidecar found");
                let mut mj = MujocoSidecarEngine::spawn().expect("spawn");
                report(&mut mj, &[16, 64], &[1]);
            }
            Err(why) => println!("\nMuJoCo backend skipped: {why}"),
        }
    }
}
