//! `gogamen-lab-mujoco`: continuous physics engine port with full integration-state
//! snapshot/restore, a deterministic pure-Rust reference solver and an optional MuJoCo
//! Python sidecar (feature `mujoco-sidecar`). Depends on no other crate.

pub mod control;
pub mod dialog;
pub mod json;
pub mod nav;
pub mod port;
pub mod unicycle;

#[cfg(feature = "mujoco-sidecar")]
pub mod sidecar;

pub use control::{
    Controller, EpisodeReport, Goal, PdController, REACH_RADIUS, Rng, SamplingPlanner, run_episode,
};
pub use port::{ContinuousPhysicsEnginePort, Dims, EngineSnapshot, PortError};
pub use unicycle::{MAX_THRUST, MAX_TORQUE, UnicycleEngine, UnicycleParams};

/// Replay tolerance (max abs difference over every state component) for the Rust solver: exact.
pub const RUST_REPLAY_TOLERANCE: f64 = 0.0;
/// Declared replay tolerance for MuJoCo restore-and-replay (same machine/build; observed 0.0).
pub const MUJOCO_REPLAY_TOLERANCE: f64 = 1e-9;

/// The fixed action script used by the replay tests: `(thrust, torque, steps)`.
pub const REPLAY_SCRIPT: [(f64, f64, usize); 4] = [
    (3.0, 0.4, 25),
    (-1.0, -0.8, 17),
    (4.0, 0.9, 40),
    (0.0, 0.0, 10),
];

/// Apply [`REPLAY_SCRIPT`] to a port.
pub fn apply_script(port: &mut dyn ContinuousPhysicsEnginePort) -> Result<(), PortError> {
    for (f, t, n) in REPLAY_SCRIPT {
        port.set_ctrl(&[f, t])?;
        port.step(n)?;
    }
    Ok(())
}

/// Full observable state: `[time, qpos.., qvel.., ctrl..]`.
pub fn observe(port: &mut dyn ContinuousPhysicsEnginePort) -> Result<Vec<f64>, PortError> {
    let mut v = vec![port.time()?];
    v.extend(port.qpos()?);
    v.extend(port.qvel()?);
    v.extend(port.ctrl()?);
    Ok(v)
}

/// Max absolute component difference.
pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warmup(e: &mut UnicycleEngine) {
        e.set_ctrl(&[2.0, 0.3]).unwrap();
        e.step(33).unwrap();
    }

    #[test]
    fn rust_restore_replay_is_bit_exact() {
        let mut e = UnicycleEngine::default();
        warmup(&mut e);
        let snap = e.snapshot().unwrap();
        apply_script(&mut e).unwrap();
        let first = observe(&mut e).unwrap();
        e.restore(&snap).unwrap();
        apply_script(&mut e).unwrap();
        let second = observe(&mut e).unwrap();
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(max_abs_diff(&first, &second), RUST_REPLAY_TOLERANCE);
    }

    #[test]
    fn rust_without_restore_state_differs() {
        let mut e = UnicycleEngine::default();
        warmup(&mut e);
        let _snap = e.snapshot().unwrap();
        apply_script(&mut e).unwrap();
        let first = observe(&mut e).unwrap();
        apply_script(&mut e).unwrap(); // no restore
        let second = observe(&mut e).unwrap();
        assert!(max_abs_diff(&first, &second) > 0.1);
    }

    #[test]
    fn rust_snapshot_includes_time_and_actuator_state() {
        let mut e = UnicycleEngine::default();
        warmup(&mut e);
        let snap = e.snapshot().unwrap();
        assert!((snap.time - 0.33).abs() < 1e-9);
        assert!(e.act()[0] != 0.0);
        let act = e.act();
        e.step(5).unwrap();
        e.restore(&snap).unwrap();
        assert_eq!(e.act(), act);
        assert_eq!(e.time().unwrap(), snap.time);
    }

    #[test]
    fn rust_restore_rejects_foreign_snapshot() {
        let mut e = UnicycleEngine::default();
        let mut s = e.snapshot().unwrap();
        s.backend = "other".into();
        assert!(matches!(e.restore(&s), Err(PortError::SnapshotMismatch(_))));
        assert!(e.set_ctrl(&[1.0]).is_err());
        assert!(e.set_ctrl(&[f64::NAN, 0.0]).is_err());
    }

    #[test]
    fn planner_respects_budget_and_is_deterministic() {
        let goal = Goal { x: 2.0, y: 1.5 };
        let run = || {
            let mut e = UnicycleEngine::default();
            let mut p = SamplingPlanner::new(7, 32, 6, 10);
            run_episode(&mut e, &mut p, goal, 40, 10).unwrap()
        };
        let (a, b) = (run(), run());
        assert!(a.sims_per_decision_max <= 32);
        assert_eq!(a.sims_total, 40 * 32);
        assert_eq!(a.final_error.to_bits(), b.final_error.to_bits());
    }

    #[test]
    fn both_controllers_make_progress_on_rust_solver() {
        let goal = Goal { x: 2.0, y: 1.5 };
        let mut e = UnicycleEngine::default();
        let pd = run_episode(&mut e, &mut PdController::default(), goal, 80, 10).unwrap();
        let mut p = SamplingPlanner::new(1, 64, 6, 10);
        let sp = run_episode(&mut e, &mut p, goal, 80, 10).unwrap();
        assert!(pd.final_error < 0.5, "pd {pd:?}");
        assert!(sp.final_error < 0.5, "sp {sp:?}");
    }
}
