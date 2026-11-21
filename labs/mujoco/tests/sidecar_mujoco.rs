//! MuJoCo sidecar tests. Skipped (with an explicit message) when python3/mujoco are missing.
use gogamen_lab_mujoco::sidecar::{MujocoSidecarEngine, probe};
use gogamen_lab_mujoco::*;

fn engine() -> Option<MujocoSidecarEngine> {
    match probe() {
        Ok(v) => {
            eprintln!("mujoco sidecar available (mujoco {v})");
            Some(MujocoSidecarEngine::spawn().expect("spawn sidecar"))
        }
        Err(why) => {
            eprintln!("SKIPPED: mujoco sidecar tests: {why}");
            None
        }
    }
}

fn warmup(e: &mut MujocoSidecarEngine) {
    e.set_ctrl(&[2.0, 0.3]).unwrap();
    e.step(33).unwrap();
}

#[test]
fn mujoco_dims_and_dt() {
    let Some(e) = engine() else { return };
    assert_eq!(
        e.dims(),
        Dims {
            nq: 3,
            nv: 3,
            nu: 2,
            na: 2
        }
    );
    assert!((e.dt() - 0.01).abs() < 1e-12);
}

#[test]
fn mujoco_restore_replay_within_tolerance() {
    let Some(mut e) = engine() else { return };
    warmup(&mut e);
    let snap = e.snapshot().unwrap();
    assert!((snap.time - 0.33).abs() < 1e-9);
    apply_script(&mut e).unwrap();
    let first = observe(&mut e).unwrap();
    e.restore(&snap).unwrap();
    apply_script(&mut e).unwrap();
    let second = observe(&mut e).unwrap();
    let d = max_abs_diff(&first, &second);
    eprintln!("mujoco replay max abs diff = {d:e}");
    assert!(d <= MUJOCO_REPLAY_TOLERANCE, "diff {d}");
    // The robot actually moved (the test is not vacuous).
    assert!(first[1].abs() + first[2].abs() > 0.1);
}

#[test]
fn mujoco_without_restore_state_differs() {
    let Some(mut e) = engine() else { return };
    warmup(&mut e);
    apply_script(&mut e).unwrap();
    let first = observe(&mut e).unwrap();
    apply_script(&mut e).unwrap(); // no restore
    let second = observe(&mut e).unwrap();
    assert!(max_abs_diff(&first, &second) > 0.1);
}

#[test]
fn mujoco_partial_restore_misses_actuator_state() {
    // Restoring only qpos/qvel (no time/act/warmstart) does not reproduce the trajectory:
    // this is why the complete integration state is snapshotted.
    let Some(mut e) = engine() else { return };
    warmup(&mut e);
    let snap = e.snapshot().unwrap();
    let (q, v) = (e.qpos().unwrap(), e.qvel().unwrap());
    let act_at_snap = e.act().unwrap();
    assert!(act_at_snap[0] != 0.0);
    e.set_ctrl(&[0.0, 0.0]).unwrap();
    e.step(60).unwrap(); // actuator state decays to ~0
    e.restore_qpos_qvel_only(&q, &v).unwrap();
    e.set_ctrl(&[2.0, 0.3]).unwrap();
    e.step(20).unwrap();
    let partial = observe(&mut e).unwrap();

    e.restore(&snap).unwrap();
    e.set_ctrl(&[2.0, 0.3]).unwrap();
    e.step(20).unwrap();
    let full = observe(&mut e).unwrap();
    // time differs by construction; compare poses.
    let d = max_abs_diff(&partial[1..4], &full[1..4]);
    eprintln!("partial-vs-full pose diff = {d:e}");
    assert!(d > 1e-6);
}

#[test]
fn mujoco_foreign_snapshot_rejected_and_planner_runs() {
    let Some(mut e) = engine() else { return };
    let mut r = UnicycleEngine::default();
    let rs = r.snapshot().unwrap();
    assert!(matches!(
        e.restore(&rs),
        Err(PortError::SnapshotMismatch(_))
    ));
    let goal = Goal { x: 1.0, y: 0.5 };
    let mut p = SamplingPlanner::new(3, 16, 4, 10);
    let rep = run_episode(&mut e, &mut p, goal, 10, 10).unwrap();
    assert!(rep.sims_per_decision_max <= 16);
    assert!(rep.final_error < goal.x.hypot(goal.y));
}
