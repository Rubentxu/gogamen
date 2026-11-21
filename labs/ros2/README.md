# gogamen-lab-ros2 (GOG-81 spike)

ROS 2 sim/HIL contract, robot-side watchdog controller and a deterministic pure-Rust link emulator.
**SIMULATION ONLY. Not for production. Not a certified safety function** (no SIL / ISO 13849 /
IEC 61508 claim). No ROS dependency, no dependency on gogamen-core/cli, no hardware code.

## Contract (`src/contract.rs`, `contract_json` example)

| Topic | Type | Dir | QoS | Rate | Deadline |
|---|---|---|---|---|---|
| `/cmd_vel` | `geometry_msgs/msg/Twist` | planner->robot | best-effort, volatile, depth 1 | 10 Hz | 100 ms |
| `/gogamen/heartbeat` | `gogamen_msgs/msg/Heartbeat` (custom) | planner->robot | best-effort, depth 1 | 5 Hz | 200 ms |
| `/odom` | `nav_msgs/msg/Odometry` | robot->planner | best-effort, depth 5 | 20 Hz | 100 ms |
| `/scan` | `sensor_msgs/msg/LaserScan` | robot->planner | best-effort, depth 5 | 10 Hz | 200 ms |
| `/gogamen/robot_state` | `gogamen_msgs/msg/RobotState` (custom) | robot->planner | reliable, transient-local, depth 1 | 5 Hz | 500 ms |

Commands are never queued or retransmitted: a stale command is worse than none. Time is virtual ms.

## Supervision: who watches whom

1. **Planner (remote, best effort)** proposes `Twist` + heartbeat. Trusted for nothing.
2. **Local controller (robot side, `controller.rs`)** watches the planner: it enforces everything
   below using only messages that arrive, independent of the planner process and the network.
3. **Hardware e-stop (physical, outside this crate)** watches everything: it must cut drive power
   without software. The `estop_line` output models the signal that would drive it; it is NOT a
   substitute for the physical button/contactor.

## Local limits and defaults (conservative)

| Item | Default | Behaviour |
|---|---|---|
| max linear / angular | 0.3 m/s / 0.5 rad/s | clamp; non-planar components zeroed; non-finite dropped |
| command deadline | 150 ms (`now - stamp`) | late command dropped |
| command timeout | 300 ms | safe Stop (unlatched; resumes on fresh command) |
| heartbeat timeout | 500 ms | **latched** EStop |
| no heartbeat yet | - | Stop (`AwaitingHeartbeat`) |
| odometry timeout | 200 ms | Stop (geofence cannot be evaluated) |
| geofence | box +-2 m, 500 ms lookahead | translation blocked at the edge; pose outside -> latched EStop |

Latched e-stop never clears by itself. `LocalController::rearm` is explicit, refused unless the
heartbeat and odometry are fresh and the pose is inside the geofence, and requires a fresh command afterwards.

## Physical stop

Software stop (zero Twist) is a first layer only. Real hardware needs an independent e-stop circuit
(contactor/brake) reachable by an operator, tested before each session.

## Physical test: NOT_RUN

`hil::run_physical` always returns `Err(NotAuthorized(..))` (`HARDWARE_ENABLED = false`; no backend exists,
even when authorization, device and checklist are supplied). Required before any physical run:
see `hil::PHYSICAL_TEST_CHECKLIST` (authorization, identified low-risk device, hardware e-stop,
fenced area, limits verified on device, trained operator, abort criteria, separately reviewed backend).

## Tests and optional smoke

```sh
cargo test -p gogamen-lab-ros2 --offline
scripts/ros2_contract_smoke.sh   # optional: ros:jazzy container, Twist layout + real topic round trip
```
The smoke exits 0 PASS, 2 NOT_RUN (no runtime/image), 1 FAIL. It is not part of default checks.
