# gogamen-lab-mujoco

Continuous physics engine port (`ContinuousPhysicsEnginePort`) for a simple mobile robot, with two backends:

| Backend | Availability | Notes |
|---|---|---|
| `UnicycleEngine` (pure Rust) | always, offline, no deps | dynamic unicycle, semi-implicit Euler, bit-exact deterministic |
| `MujocoSidecarEngine` (feature `mujoco-sidecar`, default off) | needs `python3` + `pip` module `mujoco` | Python process using `mj_getState`/`mj_setState` with `mjSTATE_INTEGRATION`, JSON lines on stdin/stdout |

The crate depends only on `gogamen-core` (the `nav` module reuses its GOG-7 `ActionGuard`/`ActionSpace`/`ActionProposal` and GOG-6 `PlanBudget`); `gogamen-core`/`gogamen-cli` do not depend on it (checked: `cargo tree -p gogamen-core -p gogamen-cli` has no mujoco/bevy/egui/wasmtime/rerun/studio).

## Port contract

`qpos` = `[x, y, yaw]`; `qvel` is backend specific (Rust: `[v, w]` body frame; MuJoCo: `[vx, vy, wz]` world frame; use `body_twist()` for frame-independent `(forward speed, yaw rate)`); `ctrl` = `[thrust (N, |.|<=5), torque (N m, |.|<=1)]`, clamped, held until changed; actuators have a first-order lag (tau 0.05 s) that is part of the state (`act`). `dt` = 0.01 s.

`snapshot()` captures the complete integration state (time, qpos, qvel, actuator state, ctrl; for MuJoCo also warmstart and the rest of `mjSTATE_INTEGRATION`). `restore()` restores it; snapshots from another backend are rejected (`SnapshotMismatch`). Same snapshot + same `set_ctrl`/`step` sequence => same state.

## Tolerances

* Rust solver: `RUST_REPLAY_TOLERANCE = 0.0` (bit-exact, asserted with `to_bits`).
* MuJoCo: `MUJOCO_REPLAY_TOLERANCE = 1e-9` max abs diff over `[time, qpos, qvel, ctrl]`. Observed on this machine (MuJoCo 4.0.1): `0e0`. The tolerance is declared, not proven portable across builds/CPUs.

## Running

```
export CARGO_TARGET_DIR=...            # optional
cargo test -p gogamen-lab-mujoco                          # Rust backend only
cargo test -p gogamen-lab-mujoco --features mujoco-sidecar -- --nocapture   # + MuJoCo
cargo run --release -p gogamen-lab-mujoco --example compare_control --features mujoco-sidecar
```

Install the module with `pip install --user mujoco`. The sidecar is started as `python3 -P -B -c <embedded script>` (`GOGAMEN_PYTHON` overrides the interpreter). `python3 -I` is not used: isolated mode also drops the per-user site-packages where `pip --user` installs `mujoco`. If python or the module is missing, `sidecar::probe()` returns `Err(reason)` and the sidecar tests print `SKIPPED: ...` and pass. The MJCF model (slide x, slide y, hinge yaw; thrust/torque `filter` actuators in the site frame; a stiff lateral damper approximating the no-skid constraint) lives in `sidecar/mobile_robot.py`. No hardware is accessed.

## Tests (what they show)

* Replay after restore reproduces the state (Rust: exact; MuJoCo: within tolerance).
* Negative: repeating the script without restoring yields a different state (> 0.1 abs).
* MuJoCo negative: restoring only `qpos/qvel` (no time/act/warmstart) yields a different trajectory (pose diff 4.4e-2 measured) - the full integration state matters.
* Planner: simulation budget is a hard cap and runs are deterministic for a fixed seed.

## Classic vs. sampling ("FMC-style") control

Task: drive from the origin to a goal in 8 s (80 decisions, one per 0.1 s). `classic-PD` is a proportional-derivative go-to-goal controller. The sampling planner forks the engine by `snapshot`/`restore` for every simulation (rollout of 6 segments x 0.1 s), uses 1 warm-start candidate (shifted previous plan) + uniform random shooting for half the budget + Gaussian refinement around the best for the rest, and receding horizon. The budget is a hard cap on rollouts per decision (`sims/dec`), fixed seeds. Latency is wall-clock per `decide` call (release build).

It does **not** reuse `gogamen_core::FmcPlanner`: that planner is built on the discrete-action `GenerativeModel`/`State`/`Action` types (walker population + virtual-reward cloning), whereas this port is continuous (`qpos/qvel/ctrl`); bridging would need an action-discretisation adapter and a dependency on core. So this is a budgeted sampling planner in the FMC spirit (fork/restore + sample + select), not the full FMC algorithm.

Measured results (`compare_control`, release, this machine, goals `(2,1.5) (-1.5,2) (0.5,-2.5)`; `final_err` in m after 8 s; `reach_t` = first time within 0.05 m). Full output, all seeds 1-3:

Rust solver, ranges over the 3 goals (x3 seeds for the sampler):

| controller | sims/decision | final_err | reach_t | mean latency / decision |
|---|---|---|---|---|
| classic-PD | 0 | 0.0002 - 0.0005 | 2.1 - 3.7 s | 0.2 - 0.6 us |
| sampler B=16 | 16 | 0.011 - 0.075 | 2.1 - 4.3 s | 31 - 38 us |
| sampler B=64 | 64 | 0.016 - 0.062 | 1.9 - 2.1 s | 123 - 144 us |
| sampler B=256 | 256 | 0.006 - 0.031 | 1.9 - 2.0 s | 523 - 592 us |

MuJoCo sidecar (seed 1 only), via JSON pipe:

| controller | sims/decision | final_err | reach_t | mean latency / decision |
|---|---|---|---|---|
| classic-PD | 0 | 0.0011 - 0.0014 | 2.0 - 4.2 s | ~123 us (state reads over the pipe) |
| sampler B=16 | 16 | 0.037 - 0.065 | 2.1 - 2.4 s | 24.7 - 26.1 ms |
| sampler B=64 | 64 | 0.016 - 0.047 | 2.0 - 2.1 s | 102 - 105 ms |

Reading: the sampler tends to reach the goal region about as fast or faster (budget >= 64 reaches in ~2 s because it uses full thrust with look-ahead, the PD is gain-limited), but the PD converges to a much smaller final error and is 2-3 orders of magnitude cheaper. The sampler never fine-converges because its cost only needs a 5 cm neighbourhood; this is a property of the cost and sampling noise, not a fundamental limit.

## Navigation stack (GOG-65, `nav` module, Rust backend only)

Simulated mobile robot in a 2D corridor (walls y = +-1.5, end wall at x = 6, a static pillar at (2.0, 0.85) r 0.25 and a moving obstacle ping-ponging along x = 3.5, y in [-1.2, 1.2] at 0.3 m/s). Start (0,0,0), goal (5,0). No hardware. The MuJoCo sidecar is not part of this stack (it has no sensor model); everything below runs on the deterministic `UnicycleEngine`.

Tick pipeline (dt = 0.01 s): perturbation -> odometry -> fusion predict -> range scan (+latency FIFO) -> fusion update -> planner every 10 ticks (under a `PlanBudget`) -> **core `ActionGuard`** -> **independent safety stop** -> `LowLevelController` -> physics step -> prediction / trace / collision / outcome.

Contract:

* The planner (`NavPlanner::plan`) returns an untrusted `Option<Vec<f64>>`. It is wrapped as an `ActionProposal` (max age `proposal_max_age` = 30 ticks) and checked by the core guard against `ActionSpace::continuous([v in +-1.0, w in +-1.5])`; wrong dimension, NaN or out-of-bounds proposals, and no/stale proposals (a hung planner), become `Fallback(stop)`. Whatever passes is converted into a `BoundedCommand`, a type whose fields are private and whose only constructor saturates (NaN -> 0); the low level only accepts that type, so the clamp holds by construction (unit + integration tests).
* The safety stop does not depend on the planner: it uses only the fused ranges. Stale sensing (no scan, or scan age > `stale_ticks` = 40, i.e. 0.4 s) => full stop. Obstacle inside a trigger distance (`radius + margin + v^2/(2 brake) + (v + obstacle_speed)(react + age)`, with a lateral reach that grows for rays not explained by the known static map) => forward velocity forced to 0 (turning in place allowed), with latch hysteresis; a moving obstacle about to touch the stopped robot makes it back away at 0.4 m/s.
* Planner-hang is modelled as "the planner stops producing proposals" (the loop is tick driven and never waits for it); a real blocked thread is not exercised.
* `SafeStop` = stopped for >= 400 consecutive ticks. Other outcomes: `Reached`, `Timeout` (4000 ticks), `Collision` (true body distance < 0).

Models and parameters (defaults of `ScenarioConfig::corridor(seed)`): unicycle plant tau 0.05 s; odometry noise v sigma 0.02 m/s, w sigma 0.01 rad/s, optional scale error `v_scale` and gyro bias `w_bias0`, bias random walk 0.002; 21 rays over 210 deg, range 3 m, sigma 0.02 m, period 5 ticks, `latency_ticks`, optional `dropout_from_tick`; fusion = odometry pose + map position fixes (each ray that hits the known static map within a 0.3 m gate nudges the pose with gain 0.3, applied at the scan's sample tick) and ego-motion age compensation of ranges with complementary blending (alpha 0.6); low level = ramped (accel 1, brake 2 m/s^2) feed-forward + P tracking, `VelocityLimits` v 1.0 m/s, w 1.5 rad/s; planner = 4 speeds x 9 yaw rates, 1 s horizon, budget 64 transitions per decision. All noise comes from a seeded RNG; same seed => bit-identical run.

Replay / trace: one `TraceRow` per tick holds true pose, estimated pose, the planner-model prediction (kinematic model reset to the estimate at each decision and integrated with the commanded velocity), `diff = pred - true` (yaw wrapped), executed command, guard verdict, safety flag, and true clearance. `TraceRow::recompute_diff()` reconstructs the difference from the recorded poses; tests check bit equality with the recorded value, equality of `trace_hash` (FNV-1a over bits) for the same seed, inequality for another seed, and that `Sim::snapshot`/`restore` replays the identical trace.

Run: `cargo test -p gogamen-lab-mujoco` (`tests/nav_scenarios.rs`: 13 tests) and `cargo run --release -p gogamen-lab-mujoco --example nav_scenarios`.

Measured results (release, seeds 1 / 2 / 3; time to the end of the episode, `min_clear` = minimum true body-to-obstacle distance; no `Collision` in any of the 51 rows):

| scenario | outcome (seeds 1/2/3) | time | min_clear |
|---|---|---|---|
| nominal (= latency 0) | Reached / Reached / Reached | 8.58 / 8.57 / 8.58 s | 0.40 m |
| latency 100 ms | Reached x3 | 24.5 / 24.6 / 24.8 s | 0.38 - 0.39 m |
| latency 300 ms | Timeout / Reached / Timeout | 40 / 27.7 / 40 s | 0.34 - 0.38 m |
| latency 450 and 800 ms (scan older than 0.4 s) | SafeStop x3 (never moves) | 4.0 s | 1.30 m |
| odometry drift (scale 1.05, bias 0.02 rad/s) with map fixes | Reached x3 | 9.0 - 9.1 s | 0.45 m |
| same drift, fixes disabled | Reached x3 (final true error 0.48 m vs 0.15 m with fixes) | 8.1 s | 0.45 m |
| shove (0, -0.3 m, +0.3 rad at 1.5 s) + kick (v +0.8, w +1.0 at 3 s) | Reached / SafeStop / SafeStop | 16.3 / 12.4 / 9.7 s | 0.29 - 0.46 m |
| sensor dropout from 2 s | SafeStop x3 | 7.3 s | 0.40 m |
| adversarial planner OutOfBounds / NaN / WrongDim / Hang | SafeStop x3, robot never moves (400 guard fallbacks each) | 4.0 s | 1.30 m |
| adversarial HangAfter(150) | SafeStop x3 (about 500 fallbacks) | 6.7 s | 0.54 m |
| adversarial Charge (always full forward) | Reached x3 (only the safety stop and the moving obstacle phase matter) | 6.2 s | 0.40 m |

Planner model error (max position error over an episode): 0.10 - 0.11 m nominal, 0.18 - 0.65 m with latency, up to 4.7 m for HangAfter (the model keeps integrating a stale command while the guard stops the robot).

Caveats:

* Liveness is not guaranteed, only safety: with 300 ms latency two of three seeds time out because the conservative safety stop keeps the robot waiting; shove/kick leaves two of three seeds in `SafeStop` (stopped for 400 ticks, goal not reached). These were not tuned further.
* The safety/threat test, the mover speed (0.3 m/s) and the margins were tuned on this single corridor over seeds 1-3 (the moving obstacle was slowed from 0.5 to 0.3 m/s because at 0.5 m/s the safety stop found no crossing window for a 1 m/s robot and the nominal case never reached the goal); results do not generalise to other worlds, faster obstacles or other robots. The safety stop assumes obstacles move at most `obstacle_speed` = 0.4 m/s.
* Heading is not corrected by fusion (only position fixes along rays); the gyro bias is not estimated. The fusion is a hand-rolled complementary filter, not an EKF.
* The retreat manoeuvre assumes free space behind the robot (the rear is poorly covered by the 210 deg field of view).
* A planner "hang" is a simulated absence of proposals, not a real blocked thread; the guard plus the tick-driven loop is what bounds it.
* Sensor noise is Gaussian-ish only; no outliers or multipath. Latency is a fixed number of ticks.
* The MuJoCo sidecar was not run through this stack.

## Fault injection and external fail-safe policy (GOG-80, `nav::faults` / `nav::failsafe`)

**Scope and non-claims.** This is a simulation only experiment on the pure-Rust backend. It is **not a safety certification**: there is no claim of SIL, ISO 13849, IEC 61508 or any other conformance, no hazard analysis, no diagnostic-coverage figure. It **does not enable real hardware**: the crate has no hardware dependency, feature or API (the only feature is the pre-existing MuJoCo Python sidecar), and `failsafe::SIMULATION_ONLY` is `true`. Guarded by `tests/fault_injection.rs`.

Fault injection (deterministic: pure function of plan, seed and tick; fault onsets are `120 + seed-derived jitter in [0, 40)`):

* Sensor: `OldTimestamp` (last scan re-delivered with its old stamp), `FrozenValues` (last values with a fresh stamp), `FrozenDithered` (frozen plus tiny seeded noise).
* Actuator: `Drop` (commands discarded, last command held; optional duration N ticks) and `Delay{ticks}` (commands applied and acknowledged late).
* Model: `WrongModelPlanner` wraps any planner; from `from_tick` the planner believes a wrong effective gain (a wrong plant mass is represented only through this gain, mass is not modelled separately) and a smaller robot radius (obstacles believed farther).

The policy (`FailSafe`) lives in its own module, has no planner handle and only sees `Observables` (scan stamp/values, actuator acknowledgement stamp, fused position, the displacement the planner model predicts for the executed command). It can only stop or cap speed. Explicit table:

| Domain | Detector (observable) | Default threshold | Fallback |
|---|---|---|---|
| Sensor | newest scan stamp older than N ticks, or K bit-identical consecutive scans | age > 15 ticks, 3 repeats | `Stop` through the normal path, latched |
| Actuator | acknowledged command lags the issued one | lag > 3 ticks | `StopAndEStop` (independent active-brake channel that bypasses the command path), latched |
| Model | windowed open-loop prediction vs fused displacement | residual > 0.18 m over 50 ticks, 3 consecutive ticks | `ReduceSpeed(0.3 m/s)`, latched |

Latch/re-arm: a tripped domain stays latched even when the fault ends; `rearm()` is refused while the fault condition is still observed and only succeeds once it has cleared.

Measured (corridor with an extra full-width wall at x = 3.0, seeds 1-3, `cargo run --release -p gogamen-lab-mujoco --example fault_injection`; ticks are 10 ms):

| Fault (policy on) | Detection after onset | Standing after detection | Result |
|---|---|---|---|
| sensor old-timestamp | 11-14 ticks | 46-51 ticks | SafeStop, min clearance 0.58-0.67 m |
| sensor frozen-values | 11-14 ticks | 46-51 ticks | SafeStop |
| actuator drop (60 ticks or permanent), delay 20 | 4 ticks | 62-65 ticks | SafeStop, min clearance 0.64-0.74 m |
| model gain 0.5 / 1.8 | 38-47 ticks | speed capped at 0.3 m/s after ramp-down | SafeStop, no collision |
| no fault (60 runs: 20 seeds x {clean, mild drift 2 % / 0.005 rad/s, blocked corridor}) | none | none | 0 false trips, 0 override ticks |

Negative control (policy off): frozen values and a permanently dead actuator end in `Collision` on all 3 seeds with the charging planner; with the wrong model the 0.3 m/s cap is violated (max speed above 0.5 m/s). Old-timestamp scans are caught by the older independent safety stop after 40 ticks, so they do not collide even with the policy off.

Residual risk, what this does NOT cover (each has a test in `tests/fault_injection.rs`):

* Faults inside the thresholds are invisible: actuator delay of 3 ticks, a model gain of 0.9, scan latency below the age limit never trip.
* Faults the detectors cannot observe: a frozen sensor with a small dither is not bit-identical, so it is not detected and the robot collides even with the policy on; a robot radius error changes planning but not the prediction, so the residual does not see it (only the independent safety stop protects there).
* Failure of the stop itself: if the e-stop channel shares the dead actuator path (`estop_broken`), the fault is detected but the robot still collides.
* Simultaneous faults: only one combination (sensor + actuator + model, same seeds) was run; sensor and actuator were detected and contained, the model fault was masked because the robot had already stopped (nothing left to compare); combinations are not analysed exhaustively and detectors are not proven independent.
* The model residual shares the odometry/fusion estimate with the rest of the stack: with heavy drift (5 % odometry scale error plus 0.02 rad/s gyro bias, seed 1) the nominal residual reaches about 0.19 m and the policy falsely trips the model domain (benign: speed cap only, no collision; a 0.10 m threshold tripped even earlier). The 0.18 m threshold was tuned on this one corridor and three seeds.
* Thresholds, timings and braking distances are tuned for this 1 m/s unicycle in this world and do not generalise. Latency of the real stop path, sensor/actuator electronics, watchdog hardware and common-cause failures are not modelled.

## Dialog: linguistic perception + AskHuman (GOG-66, `dialog` module)

A simulated robot must reach "the target door" (left door at (5, 0.8) or right door at (5, -0.8)); the target is only known through a human. Beliefs are a core `BinaryBelief` (`P(target is left)`), updated by core primitives (`LikelihoodModel`, `EvidenceDeduplicator`, `ObservationEnvelope`); the ask/act decision uses core `EpistemicActionSelector` (EVOI minus ask cost).

Contract:

* Utterance to evidence: each utterance goes through the core `SystemOneProvider` port (the "Jev" oracle) as a `Noul` query. `Replay` mode answers from recorded verdicts (default, offline). `Live` is an opt-in stub that always reports "not available" (NOE-029 / real Jev / credentials are not available here). A true/false verdict becomes evidence with speaker reliability 0.9; unknown, timeout or unavailable gives no evidence.
* `AskHuman`: one question, answered with reliability 0.95 after the fixture latency, or timeout (120 ticks) which gives no evidence.
* Policy (`DialogPolicy`): prior 0.5, act threshold 0.85 on confidence in the MAP door, reward +10 / wrong door -20 / wait 0, ask cost 2, response probability 0.9. Variants: `no-ask (always guess)`, `no-ask (threshold)` (abstain below threshold), `ask-when-worth-it` (EVOI > cost), `always-ask`.
* No authority over the guard: the cognition modules (`oracle`, `perception`, `policy`, `cognition`, `fixtures`) never reference the simulator, `ActionGuard` or safety stop (textual test). They only emit a typed goal `Command::Go(door)`. `episode.rs` calls `Sim::step_tick` exactly once per control tick regardless of cognition; until a goal is adopted the planner proposes a stop. The goal change still goes through the core `ActionGuard` and the independent safety stop.
* Time is virtual (10 ms/tick); oracle and human latencies are simulated and do not block the control loop.

Tests (`tests/dialog.rs`): AC1 one test per fixture kind with concrete beliefs (correct 0.5 -> 0.9 and go; ambiguous stays 0.5 then ask -> 0.95, or mute human -> hold at 0.5; contradictory 0.9 -> 0.5 then ask -> 0.05, or silent human -> hold). AC2 never-answering oracle and 1 000 / 50 000-tick slow oracles give the same control trace hash, ticks, decision and release tick; ticks are contiguous for every oracle mode; a fast oracle's control rows are identical to the never-oracle's up to the release tick. AC3 metrics and policy comparison over 16 fixtures.

Measured (`cargo run --release -p gogamen-lab-mujoco --example dialog_eval`; 16 annotated, seeded fixtures: 8 correct, 4 ambiguous, 4 contradictory; Brier of final `P(left)` vs annotation, ECE with 5 bins on MAP confidence; 0 collisions in every run):

| Policy (Replay oracle) | succ | wrong | abstain | asks | utility | Brier | ECE | MAP acc |
|---|---|---|---|---|---|---|---|---|
| no-ask (always guess) | 9 | 7 | 0 | 0 | -50 | 0.2800 | 0.1375 | 0.5625 |
| no-ask (threshold) | 5 | 3 | 8 | 0 | -10 | 0.2800 | 0.1375 | 0.5625 |
| ask-when-worth-it (EVOI) | 9 | 5 | 2 | 8 | -26 | 0.2997 | 0.3062 | 0.5625 |
| always-ask | 9 | 3 | 4 | 16 | -2 | 0.2082 | 0.2634 | 0.6875 |

With a never-answering oracle (no utterance evidence): no-ask guess 7/9/0 (utility -110), no-ask threshold 0/0/16, ask-when-worth-it and always-ask both 11 succ / 2 wrong / 3 abstain, 16 asks, utility 38, Brier 0.1614, ECE 0.1781. Constant 0.5 has Brier 0.25.

Reading: always asking is the best policy on this set under the Replay oracle; the EVOI policy does not ask when a confident (0.9) but wrong utterance arrives (3 fixtures), so it does worse than always-ask and its Brier (0.2997) is worse than the constant 0.5 reference.

Caveats (honest):

* Fixtures are synthetic and hand-annotated, 16 episodes, one corridor, one seed per episode; differences of 1-2 episodes are not statistically meaningful. The fixtures deliberately include speaker errors (about 3 of 8 "correct" ones are wrong), so the assumed 0.9 speaker reliability is overconfident: this is what drives the poor ECE.
* Jev is not real: only recorded replay verdicts (0.8/0.2 distributions invented for the fixture) and an unavailable live stub. Nothing here says anything about real language-model behaviour.
* Evidence items are assumed independent; the core `SemanticGrounder` and communication bus are not used; the human is a recorded reply, not an interactive agent; ask timeouts and reply latencies are fixed numbers.
* An `Unavailable` oracle is known immediately, whereas a timeout is only known after 50 ticks, so their decision ticks differ (both leave control timing unchanged while holding).
* The control stack, `GatedPlanner` and the `Sim::set_goal` / `planner_mut` additions are the same nav stack as GOG-65; the guard is not weakened.

## Caveats

* Results come from one task, three goals, three seeds, untuned gains for both controllers; do not generalise.
* The two backends are different models (Rust: abstract unicycle; MuJoCo: slide/slide/hinge chain with thrust along the body axis and lateral damping): they are not expected to match each other numerically, and nothing here compares them.
* MuJoCo cost per simulation is dominated by JSON round trips (~5 per rollout step/segment), not by physics; a batched `rollout` command would cut it. Latencies are single-run wall-clock numbers, not statistically characterised.
* Replay tolerance for MuJoCo was observed on one machine/build only.
* The Rust crate `mujoco-rs`/C bindings were not tried; only the Python module was used (the sidecar route from the task).
