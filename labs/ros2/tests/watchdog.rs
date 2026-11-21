use gogamen_lab_ros2::bridge::Emulator;
use gogamen_lab_ros2::contract::{Mode, Twist};
use gogamen_lab_ros2::controller::*;
use gogamen_lab_ros2::hil::*;

/// Planner that sends cmd + heartbeat every 100 ms for `ms`.
fn drive(e: &mut Emulator, ms: u64, cmd: Twist) {
    for _ in 0..ms / 100 {
        e.planner_cmd(cmd);
        e.planner_heartbeat();
        e.run_ms(100);
    }
}

fn emu() -> Emulator {
    Emulator::new(LocalConfig::default(), 42)
}

#[test]
fn command_arrives_and_moves() {
    let mut e = emu();
    drive(&mut e, 1000, Twist::planar(0.2, 0.0));
    assert_eq!(e.last.status, Status::Run);
    assert!(e.x > 0.15 && e.x < 0.25, "x={}", e.x);
}

#[test]
fn no_commands_beyond_timeout_stops() {
    let mut e = emu();
    drive(&mut e, 500, Twist::planar(0.2, 0.0));
    // heartbeats continue, commands stop
    for _ in 0..5 {
        e.planner_heartbeat();
        e.run_ms(100);
    }
    assert_eq!(e.last.status, Status::Stop(StopCause::CommandTimeout));
    assert!(e.last.twist.is_zero());
    assert!(!e.last.estop_line);
    let x = e.x;
    e.run_ms(200);
    assert_eq!(e.x, x);
}

#[test]
fn heartbeat_loss_stops_and_latches() {
    let mut e = emu();
    drive(&mut e, 500, Twist::planar(0.2, 0.0));
    e.faults.hb_cut = true;
    for _ in 0..10 {
        e.planner_cmd(Twist::planar(0.2, 0.0)); // commands keep coming
        e.planner_heartbeat();
        e.run_ms(100);
    }
    assert_eq!(e.last.status, Status::EStop(EStopCause::HeartbeatLost));
    assert!(e.last.twist.is_zero() && e.last.estop_line);
    // heartbeat restored: must NOT rearm by itself
    e.faults.hb_cut = false;
    drive(&mut e, 1000, Twist::planar(0.2, 0.0));
    assert_eq!(e.last.status.mode(), Mode::EStop);
    assert!(e.last.twist.is_zero());
    // explicit rearm works only with a fresh heartbeat and then needs a fresh command
    assert_eq!(e.ctrl.rearm(e.now()), Ok(()));
    e.run_ms(10);
    assert_eq!(e.last.status, Status::Stop(StopCause::NoCommand));
    drive(&mut e, 300, Twist::planar(0.2, 0.0));
    assert_eq!(e.last.status, Status::Run);
}

#[test]
fn rearm_refused_without_fresh_heartbeat() {
    let mut e = emu();
    drive(&mut e, 300, Twist::planar(0.1, 0.0));
    e.faults.hb_cut = true;
    e.run_ms(1000);
    assert_eq!(e.last.status.mode(), Mode::EStop);
    assert_eq!(e.ctrl.rearm(e.now()), Err(RearmRefused::HeartbeatNotFresh));
    assert_eq!(e.last.status.mode(), Mode::EStop);
}

#[test]
fn latency_above_deadline_drops_command() {
    let mut e = emu();
    drive(&mut e, 300, Twist::planar(0.2, 0.0));
    e.faults.cmd_delay_ms = 200; // > 150 ms deadline
    let x0 = e.x;
    drive(&mut e, 1000, Twist::planar(0.2, 0.0));
    assert!(e.ctrl.stats.dropped_late >= 8, "{:?}", e.ctrl.stats);
    // motion only from commands accepted before the delay (<= timeout window)
    assert!(e.x - x0 < 0.1, "moved {}", e.x - x0);
    assert_eq!(e.last.status, Status::Stop(StopCause::CommandTimeout));
}

#[test]
fn latency_below_deadline_is_accepted() {
    let mut e = emu();
    e.faults.cmd_delay_ms = 100;
    drive(&mut e, 1000, Twist::planar(0.2, 0.0));
    assert_eq!(e.ctrl.stats.dropped_late, 0);
    assert_eq!(e.last.status, Status::Run);
}

#[test]
fn deterministic_message_loss() {
    let run = || {
        let mut e = Emulator::new(LocalConfig::default(), 7);
        e.faults.cmd_loss_permille = 400;
        drive(&mut e, 3000, Twist::planar(0.2, 0.0));
        (e.x, e.ctrl.stats)
    };
    assert_eq!(run(), run());
    let (_, s) = run();
    assert!(s.accepted > 0 && s.accepted < 30);
}

#[test]
fn speed_limits_enforced_locally() {
    let mut e = emu();
    drive(&mut e, 500, Twist::planar(5.0, 10.0));
    assert!(e.last.twist.linear.x <= 0.3 + 1e-12);
    assert!(e.last.twist.angular.z <= 0.5 + 1e-12);
    assert!(e.ctrl.stats.clamped > 0);
    // non-planar and non-finite components never reach the drive
    let mut c = LocalController::new(LocalConfig::default());
    assert_eq!(c.on_cmd(10, 10, Twist::planar(f64::NAN, 0.0)), CmdVerdict::DroppedNonFinite);
    let mut t = Twist::planar(0.1, 0.0);
    t.linear.z = 9.0;
    t.angular.x = 9.0;
    assert_eq!(c.on_cmd(10, 10, t), CmdVerdict::Accepted);
}

#[test]
fn geofence_enforced_locally_and_latches() {
    let mut e = emu();
    // Fence at x=2.0; planner insists on driving forward at an absurd speed.
    drive(&mut e, 12_000, Twist::planar(5.0, 0.0));
    assert!(e.x <= 2.0, "x={}", e.x);
    assert_eq!(e.last.status, Status::Stop(StopCause::GeofenceLookahead));
    // Pose pushed outside the fence (e.g. external disturbance) latches e-stop.
    e.x = 2.5;
    drive(&mut e, 100, Twist::planar(0.1, 0.0));
    assert_eq!(e.last.status, Status::EStop(EStopCause::GeofenceBreach));
    assert_eq!(e.ctrl.rearm(e.now()), Err(RearmRefused::OutsideGeofence));
}

#[test]
fn awaiting_heartbeat_does_not_move() {
    let mut e = emu();
    for _ in 0..10 {
        e.planner_cmd(Twist::planar(0.2, 0.0));
        e.run_ms(100);
    }
    assert_eq!(e.last.status, Status::Stop(StopCause::AwaitingHeartbeat));
    assert_eq!(e.x, 0.0);
}

#[test]
fn physical_run_is_never_executed() {
    let flags = std::hint::black_box((HARDWARE_ENABLED, gogamen_lab_ros2::SIMULATION_ONLY));
    assert_eq!(flags, (false, true));
    let empty = HilRequest::default();
    assert_eq!(run_physical(&empty), Err(HilError::NotAuthorized(NotRunReason::NoAuthorization)));
    let full = HilRequest {
        authorization: Some("ticket".into()),
        device_id: Some("dev-1".into()),
        checklist_ack: PHYSICAL_TEST_CHECKLIST.iter().map(|s| s.to_string()).collect(),
    };
    assert_eq!(run_physical(&full), Err(HilError::NotAuthorized(NotRunReason::NoPhysicalBackend)));
    let partial = HilRequest { checklist_ack: vec![], ..full };
    assert_eq!(run_physical(&partial), Err(HilError::NotAuthorized(NotRunReason::ChecklistIncomplete)));
}

#[test]
fn contract_is_consistent() {
    use gogamen_lab_ros2::contract::*;
    let j = contract_json();
    for t in TOPICS {
        assert!(j.contains(t.name));
        assert!(t.qos.deadline_ms * t.rate_hz as u64 >= 1000, "{} deadline shorter than period", t.name);
    }
    assert!(j.contains("\"linear.x\"") && j.contains("\"angular.z\""));
    // controller timeouts must exceed the contract periods
    let c = LocalConfig::default();
    assert!(c.cmd_timeout_ms >= 2 * (1000 / TOPICS[0].rate_hz as u64));
    assert!(c.heartbeat_timeout_ms >= 2 * (1000 / TOPICS[1].rate_hz as u64));
}
