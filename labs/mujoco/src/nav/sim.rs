//! Closed-loop scenario: engine + odometry + range sensor + fusion + planner + guard + safety stop
//! + low-level controller, with a per-tick prediction-vs-physics trace and snapshot/restore.
//!
//! Per-tick order: perturbations -> odometry -> fusion predict -> sensor (+latency) -> fusion update
//! -> planner (every `decision_period` ticks, bounded by `PlanBudget`) -> `ActionGuard` (limits,
//! dimension, finiteness, staleness; fallback = stop) -> independent `SafetyStop` (can only reduce
//! the command to STOP) -> low-level controller -> one physics step -> prediction/trace/collision.

use gogamen_core::{
    Action, ActionBounds, ActionGuard, ActionProposal, ActionSpace, GuardContext, GuardDecision,
    PlanBudget, SafeFallbackPolicy, Terminal, Tick,
};

use super::failsafe::{FailSafe, FailSafeConfig, Observables};
use super::faults::{ActuatorInjector, FaultPlan, SensorInjector};
use super::fusion::{Fusion, FusionConfig};
use super::geometry::{MovingObstacle, Shape, World};
use super::lowlevel::{BoundedCommand, LowLevelConfig, LowLevelController, VelocityLimits};
use super::planner::{NavObservation, NavPlanner};
use super::sensors::{OdomConfig, Odometry, RangeConfig, RangeSensor};
use crate::port::{ContinuousPhysicsEnginePort, EngineSnapshot, PortError};
use crate::unicycle::{UnicycleEngine, UnicycleParams};

/// Independent safety-stop parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafetyConfig {
    /// A scan older than this many ticks (or no scan yet) forces a stop.
    pub stale_ticks: u64,
    /// Extra distance kept from obstacles (m).
    pub margin: f64,
    /// Corridor half-width checked ahead = robot radius + this (m).
    pub lateral_margin: f64,
    /// Deceleration assumed when computing the stopping distance (m/s^2).
    pub brake_decel: f64,
    /// Reaction time added to the stopping distance (s).
    pub react_time: f64,
    /// Extra clearance needed to release the latch (m).
    pub hysteresis: f64,
    /// Assumed worst-case speed of an unknown moving obstacle (m/s); inflates the stopping
    /// distance and the checked corridor by `speed * (react_time + scan age)`.
    pub obstacle_speed: f64,
    /// Cap (s) on the time horizon over which an obstacle may drift laterally into the corridor.
    pub lookahead_time: f64,
    /// Speed assumed for the time-to-reach of a hit while the robot is slow or stopped (it will accelerate).
    pub assumed_speed: f64,
    /// Extra path length (m) the robot needs to travel to get past a hit (body + obstacle extent).
    pub pass_length: f64,
    /// A moving obstacle closer than `robot_radius + retreat_margin` makes the blocked robot back away.
    pub retreat_margin: f64,
    /// Reverse speed (m/s) used while backing away.
    pub retreat_speed: f64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            stale_ticks: 40,
            margin: 0.25,
            lateral_margin: 0.1,
            brake_decel: 1.5,
            react_time: 0.15,
            hysteresis: 0.15,
            obstacle_speed: 0.4,
            lookahead_time: 1.5,
            assumed_speed: 0.9,
            pass_length: 0.6,
            retreat_margin: 0.45,
            retreat_speed: 0.4,
        }
    }
}

/// External disturbance applied at a tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Perturbation {
    /// Instantaneous pose shove (invisible to odometry).
    Shove {
        /// Tick at which it happens.
        tick: u64,
        /// World dx (m).
        dx: f64,
        /// World dy (m).
        dy: f64,
        /// Yaw change (rad).
        dyaw: f64,
    },
    /// Instantaneous velocity kick.
    Kick {
        /// Tick at which it happens.
        tick: u64,
        /// Forward speed change (m/s).
        dv: f64,
        /// Yaw-rate change (rad/s).
        dw: f64,
    },
}

/// Full scenario description.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    /// Obstacles.
    pub world: World,
    /// Goal `[x, y]`.
    pub goal: [f64; 2],
    /// Start pose.
    pub start: [f64; 3],
    /// Plant.
    pub plant: UnicycleParams,
    /// Odometry model.
    pub odom: OdomConfig,
    /// Range sensor.
    pub range: RangeConfig,
    /// Fusion.
    pub fusion: FusionConfig,
    /// Velocity limits (action space of the guard, clamp of the low-level controller).
    pub limits: VelocityLimits,
    /// Safety stop.
    pub safety: SafetyConfig,
    /// Ticks between planner calls.
    pub decision_period: u64,
    /// A proposal older than this many ticks is rejected by the guard (covers a hung planner).
    pub proposal_max_age: u64,
    /// Candidate evaluations allowed per planner call (`PlanBudget`).
    pub planner_budget: usize,
    /// Robot radius (m) for true collision checks.
    pub robot_radius: f64,
    /// Estimated distance to the goal that counts as reached (m).
    pub reach_tol: f64,
    /// Episode length limit.
    pub max_ticks: u64,
    /// Consecutive stopped ticks after which the episode ends as a safe stop.
    pub safe_stop_hold_ticks: u64,
    /// Disturbances.
    pub perturbations: Vec<Perturbation>,
    /// Seed for odometry and sensor noise.
    pub seed: u64,
    /// Injected faults (default: none).
    pub faults: FaultPlan,
    /// External fail-safe policy (default: disabled).
    pub failsafe: Option<FailSafeConfig>,
}

impl ScenarioConfig {
    /// Corridor scenario: walls at y = +-1.5, end wall at x = 6.2, a static pillar and a mover
    /// crossing the corridor at x = 3.5. Start (0,0), goal (5,0).
    pub fn corridor(seed: u64) -> Self {
        let wall = |min: [f64; 2], max: [f64; 2]| Shape::Aabb { min, max };
        Self {
            world: World {
                statics: vec![
                    wall([-1.0, 1.5], [6.0, 1.7]),
                    wall([-1.0, -1.7], [6.0, -1.5]),
                    wall([6.0, -1.7], [6.2, 1.7]),
                    Shape::Circle {
                        c: [2.0, 0.85],
                        r: 0.25,
                    },
                ],
                movers: vec![MovingObstacle {
                    a: [3.5, -1.2],
                    b: [3.5, 1.2],
                    speed: 0.3,
                    phase: 1.2,
                    radius: 0.2,
                }],
            },
            goal: [5.0, 0.0],
            start: [0.0, 0.0, 0.0],
            plant: UnicycleParams::default(),
            odom: OdomConfig::default(),
            range: RangeConfig::default(),
            fusion: FusionConfig::default(),
            limits: VelocityLimits::default(),
            safety: SafetyConfig::default(),
            decision_period: 10,
            proposal_max_age: 30,
            planner_budget: 64,
            robot_radius: 0.2,
            reach_tol: 0.15,
            max_ticks: 4000,
            safe_stop_hold_ticks: 400,
            perturbations: vec![],
            seed,
            faults: FaultPlan::none(),
            failsafe: None,
        }
    }
}

/// Episode status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Still running.
    Running,
    /// Estimated position within tolerance of the goal.
    Reached,
    /// Stopped (by guard fallback or safety stop) for `safe_stop_hold_ticks` without collision: fails safe.
    SafeStop,
    /// Hit `max_ticks` while moving/waiting.
    Timeout,
    /// The robot body touched an obstacle (the failure this stack exists to prevent).
    Collision,
}

/// What the guard did this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardTag {
    /// Proposal allowed as is.
    Allow,
    /// Proposal rejected, explicit stop fallback applied.
    Fallback,
    /// Rejected without fallback.
    Reject,
    /// No-op.
    NoOp,
}

/// One trace row (one per physics tick).
#[derive(Debug, Clone, PartialEq)]
pub struct TraceRow {
    /// Tick index after the step (1-based count of steps taken).
    pub tick: u64,
    /// Physical pose `[x, y, yaw]`.
    pub true_pose: [f64; 3],
    /// Fused estimate at decision time of this tick.
    pub est_pose: [f64; 3],
    /// Pose predicted by the planner's model for this tick.
    pub pred_pose: [f64; 3],
    /// Recorded `pred_pose - true_pose` (yaw wrapped to (-pi, pi]).
    pub diff: [f64; 3],
    /// Command executed by the low level (after guard + safety).
    pub cmd: [f64; 2],
    /// Guard verdict.
    pub guard: GuardTag,
    /// Safety stop active.
    pub safety: bool,
    /// True distance from the robot body to the nearest obstacle (m).
    pub clearance: f64,
}

fn wrap(a: f64) -> f64 {
    (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI
}

/// `pred - truth` with yaw wrapped.
pub fn pose_diff(pred: [f64; 3], truth: [f64; 3]) -> [f64; 3] {
    [
        pred[0] - truth[0],
        pred[1] - truth[1],
        wrap(pred[2] - truth[2]),
    ]
}

impl TraceRow {
    /// Recompute the prediction error from the stored poses (must equal `diff` bit for bit).
    pub fn recompute_diff(&self) -> [f64; 3] {
        pose_diff(self.pred_pose, self.true_pose)
    }
}

/// Summary of an episode.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// Outcome.
    pub outcome: Outcome,
    /// Ticks executed.
    pub ticks: u64,
    /// True distance from the final pose to the goal (m).
    pub true_goal_error: f64,
    /// Minimum true body-to-obstacle distance over the episode (m); negative = collision.
    pub min_clearance: f64,
    /// Guard counts.
    pub allowed: u64,
    /// Guard fallbacks (rejected proposals replaced by stop).
    pub fallbacks: u64,
    /// Ticks in which the safety stop overrode the command.
    pub safety_ticks: u64,
    /// Max |true forward speed| seen (m/s).
    pub max_speed: f64,
    /// Max |true yaw rate| seen (rad/s).
    pub max_yaw_rate: f64,
    /// Max planner-model error norm (position, m) over the trace.
    pub max_pred_err: f64,
    /// Hash of the whole trace (bit-level).
    pub trace_hash: u64,
}

/// The closed-loop simulation.
#[derive(Debug, Clone)]
pub struct Sim<P: NavPlanner> {
    cfg: ScenarioConfig,
    engine: UnicycleEngine,
    map: World,
    odom: Odometry,
    sensor: RangeSensor,
    fusion: Fusion,
    low: LowLevelController,
    planner: P,
    space: ActionSpace,
    guard: ActionGuard,
    tick: u64,
    proposal: ActionProposal,
    pred_pose: [f64; 3],
    pred_cmd: BoundedCommand,
    latched: bool,
    stopped_ticks: u64,
    outcome: Outcome,
    min_clearance: f64,
    counts: [u64; 3],
    max_speed: f64,
    max_yaw_rate: f64,
    trace: Vec<TraceRow>,
    sensor_inj: SensorInjector,
    act_inj: ActuatorInjector,
    failsafe: Option<FailSafe>,
}

/// Active-braking gains of the independent e-stop channel (N per m/s, N m per rad/s).
const ESTOP_KV: f64 = 10.0;
const ESTOP_KW: f64 = 2.0;

/// Snapshot of the full simulation state (engine through the port + all controller/sensor state).
#[derive(Debug, Clone)]
pub struct SimSnapshot<P: NavPlanner> {
    engine: EngineSnapshot,
    sim: Box<Sim<P>>,
}

impl<P: NavPlanner> Sim<P> {
    /// Build the simulation. Panics only on invalid static configuration (limits not finite).
    pub fn new(cfg: ScenarioConfig, planner: P) -> Self {
        let engine = UnicycleEngine::with_pose(cfg.plant, cfg.start[0], cfg.start[1], cfg.start[2]);
        let sensor = RangeSensor::new(cfg.range, cfg.seed ^ 0xA5A5_5A5A);
        let fusion = Fusion::new(cfg.fusion, sensor.angles(), cfg.range.max_range, cfg.start);
        let space = ActionSpace::continuous(vec![
            ActionBounds::new(-cfg.limits.v_max, cfg.limits.v_max).expect("finite v limits"),
            ActionBounds::new(-cfg.limits.w_max, cfg.limits.w_max).expect("finite w limits"),
        ])
        .expect("non-empty action space");
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(Action::Continuous(
            vec![0.0, 0.0],
        )));
        let mut ll = LowLevelConfig::for_plant(&cfg.plant);
        ll.limits = cfg.limits;
        Self {
            map: World {
                statics: cfg.world.statics.clone(),
                movers: vec![],
            },
            odom: Odometry::new(cfg.odom, cfg.seed, cfg.start),
            sensor,
            fusion,
            low: LowLevelController::new(ll),
            planner,
            space,
            guard,
            tick: 0,
            proposal: ActionProposal::unauthorized(Tick::ZERO),
            pred_pose: cfg.start,
            pred_cmd: BoundedCommand::STOP,
            latched: false,
            stopped_ticks: 0,
            outcome: Outcome::Running,
            min_clearance: f64::INFINITY,
            counts: [0; 3],
            max_speed: 0.0,
            max_yaw_rate: 0.0,
            trace: Vec::new(),
            sensor_inj: SensorInjector::new(&cfg.faults),
            act_inj: ActuatorInjector::new(&cfg.faults),
            failsafe: cfg.failsafe.map(FailSafe::new),
            engine,
            cfg,
        }
    }

    /// Mutable access to the (untrusted) planner, e.g. to release a gated planner. The guard and
    /// the safety stop still sit between whatever the planner proposes and the actuators.
    pub fn planner_mut(&mut self) -> &mut P {
        &mut self.planner
    }

    /// Replace the task goal (task input, not a safety parameter: the guard and the safety stop
    /// are unaffected). Used by the dialog layer once the target has been disambiguated.
    pub fn set_goal(&mut self, goal: [f64; 2]) {
        self.cfg.goal = goal;
    }

    /// Per-tick trace.
    pub fn trace(&self) -> &[TraceRow] {
        &self.trace
    }

    /// Current outcome.
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }

    /// Ticks executed.
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Physical engine (read-only).
    pub fn engine(&self) -> &UnicycleEngine {
        &self.engine
    }

    /// Fusion (read-only).
    pub fn fusion(&self) -> &Fusion {
        &self.fusion
    }

    /// External fail-safe policy state (if enabled).
    pub fn failsafe(&self) -> Option<&FailSafe> {
        self.failsafe.as_ref()
    }

    /// Explicit operator re-arm of the fail-safe policy (`Ok` also when it is disabled).
    pub fn rearm_failsafe(&mut self) -> Result<(), super::failsafe::RearmRefused> {
        self.failsafe.as_mut().map_or(Ok(()), FailSafe::rearm)
    }

    /// Capture everything needed to replay from here.
    pub fn snapshot(&mut self) -> Result<SimSnapshot<P>, PortError> {
        Ok(SimSnapshot {
            engine: self.engine.snapshot()?,
            sim: Box::new(self.clone()),
        })
    }

    /// Restore a snapshot taken from a simulation with the same configuration.
    pub fn restore(&mut self, s: &SimSnapshot<P>) -> Result<(), PortError> {
        *self = (*s.sim).clone();
        self.engine.restore(&s.engine)
    }

    fn sanitize(&self, v: &[f64]) -> BoundedCommand {
        if v.len() == 2 {
            BoundedCommand::new(v[0], v[1], &self.cfg.limits)
        } else {
            BoundedCommand::STOP
        }
    }

    /// Returns `(stale, blocked)`. `stale` (no/old scan) forces a full stop; `blocked` (obstacle
    /// within the stopping distance, latched with hysteresis) forces zero forward speed.
    fn retreat_needed(&self) -> bool {
        self.fusion.nearest_dynamic() < self.cfg.robot_radius + self.cfg.safety.retreat_margin
    }

    fn safety_check(&mut self, v_est: f64) -> (bool, bool) {
        let s = self.cfg.safety;
        let age = self.fusion.scan_age(self.tick);
        let stale = age.is_none_or(|a| a > s.stale_ticks);
        let age_s = age.unwrap_or(0) as f64 * self.engine.dt();
        let vf = v_est.max(0.0);
        let delay = s.react_time + age_s;
        let trig = self.cfg.robot_radius
            + s.margin
            + vf * vf / (2.0 * s.brake_decel)
            + (vf + s.obstacle_speed) * delay;
        let half = self.cfg.robot_radius + s.lateral_margin;
        let clear = self.fusion.forward_clearance(
            half,
            s.obstacle_speed,
            delay,
            vf.max(s.assumed_speed),
            s.lookahead_time,
            s.pass_length,
        );
        if clear < trig {
            self.latched = true;
        } else if clear > trig + s.hysteresis {
            self.latched = false;
        }
        (stale, self.latched)
    }

    /// Advance one physics tick. No-op once the episode has ended.
    pub fn step_tick(&mut self) -> Result<(), PortError> {
        if self.outcome != Outcome::Running {
            return Ok(());
        }
        let dt = self.engine.dt();
        let tick = self.tick;
        let cfg_p = self.cfg.perturbations.clone();
        for p in cfg_p {
            match p {
                Perturbation::Shove {
                    tick: t,
                    dx,
                    dy,
                    dyaw,
                } if t == tick => self.engine.shove(dx, dy, dyaw),
                Perturbation::Kick { tick: t, dv, dw } if t == tick => self.engine.kick(dv, dw),
                _ => {}
            }
        }
        let time = self.engine.time()?;
        let q = self.engine.qpos()?;
        let (tv, tw) = self.engine.body_twist()?;
        self.max_speed = self.max_speed.max(tv.abs());
        self.max_yaw_rate = self.max_yaw_rate.max(tw.abs());

        // Sensing and fusion.
        self.odom.update(tv, tw, dt);
        let (ov, _) = self.odom.twist();
        self.fusion.predict(tick, self.odom.pose(), ov, dt);
        let raw_scan = self
            .sensor
            .tick(tick, time, [q[0], q[1], q[2]], &self.cfg.world);
        let delivered = self.sensor_inj.filter(tick, raw_scan);
        if let Some(scan) = &delivered {
            self.fusion.on_scan(tick, scan, &self.map, ov, dt);
        }
        let est = self.fusion.pose();

        // Planner (untrusted), bounded by PlanBudget.
        if tick.is_multiple_of(self.cfg.decision_period) && self.fusion.scan_age(tick).is_some() {
            let obs = NavObservation {
                tick,
                pose: est,
                ranges: self.fusion.ranges(),
                angles: self.fusion.angles(),
                max_range: self.fusion.max_range(),
                goal: self.cfg.goal,
            };
            let mut tracker = PlanBudget::with_transitions(self.cfg.planner_budget).tracker();
            if let Some(raw) = self.planner.plan(&obs, &mut tracker) {
                self.pred_cmd = self.sanitize(&raw);
                self.pred_pose = est;
                self.proposal = match Action::continuous(raw) {
                    Ok(a) => ActionProposal::authorized(a, Tick::new(tick)),
                    Err(_) => ActionProposal::unauthorized(Tick::new(tick)),
                }
                .with_max_age_ticks(self.cfg.proposal_max_age);
            }
        }

        // Independent guard, then independent safety stop (can only reduce to STOP).
        let ctx = GuardContext::now(&self.space, Tick::new(tick), Terminal::Ongoing);
        let decision = self.guard.evaluate(&self.proposal, &ctx);
        let (mut cmd, tag) = match &decision {
            GuardDecision::Allow(a) => (
                a.as_continuous()
                    .map_or(BoundedCommand::STOP, |v| self.sanitize(v)),
                GuardTag::Allow,
            ),
            GuardDecision::Fallback(_) => (BoundedCommand::STOP, GuardTag::Fallback),
            GuardDecision::NoOp => (BoundedCommand::STOP, GuardTag::NoOp),
            GuardDecision::Reject(_) => (BoundedCommand::STOP, GuardTag::Reject),
        };
        self.counts[0] += u64::from(tag == GuardTag::Allow);
        self.counts[1] += u64::from(tag != GuardTag::Allow);
        let (stale, blocked) = self.safety_check(ov);
        let safety = stale || blocked;
        if stale {
            cmd = BoundedCommand::STOP;
        } else if blocked && self.retreat_needed() {
            // A moving obstacle is about to touch the (stopped) robot: back away along the way we came.
            cmd = BoundedCommand::new(-self.cfg.safety.retreat_speed, 0.0, &self.cfg.limits);
        } else if blocked {
            // Blocked ahead: no forward motion; turning in place stays allowed so the planner can
            // find a way out (a disc robot cannot hit anything by rotating).
            cmd = BoundedCommand::new(0.0, cmd.w(), &self.cfg.limits);
        }
        // External fail-safe policy: sees only observables, can only force stop / cap the speed.
        let mut estop = false;
        if let Some(fs) = self.failsafe.as_mut() {
            let (rv, rw) = self.low.reference();
            let next = self.planner.predict(est, [rv, rw], dt);
            let obs = Observables {
                tick,
                scan: delivered
                    .as_ref()
                    .map(|s| (s.sample_tick, s.ranges.as_slice())),
                echo_tick: self.act_inj.echo(),
                est_xy: [est[0], est[1]],
                pred_step: [next[0] - est[0], next[1] - est[1]],
            };
            let o = fs.update(&obs);
            if o.stop {
                cmd = BoundedCommand::STOP;
            }
            if let Some(cap) = o.speed_cap {
                cmd = BoundedCommand::new(cmd.v().clamp(-cap, cap), cmd.w(), &self.cfg.limits);
            }
            estop = o.estop;
        }
        let safety = safety || estop;
        self.counts[2] += u64::from(safety);
        self.stopped_ticks = if cmd.v() == 0.0 && cmd.w() == 0.0 && tv.abs() < 0.02 {
            self.stopped_ticks + 1
        } else {
            0
        };

        // Actuate and step.
        let ow = self.odom.twist().1;
        let u = self.low.step(cmd, ov, ow, dt);
        // The command path is always exercised (its acknowledgement is what the policy watches).
        let applied = self.act_inj.issue(tick, u);
        if estop && !self.cfg.faults.estop_broken {
            // Independent e-stop channel: active braking that bypasses the (possibly faulty) command path.
            self.engine.set_ctrl(&[-ESTOP_KV * ov, -ESTOP_KW * ow])?;
        } else if let Some(c) = applied.ctrl {
            self.engine.set_ctrl(&c)?;
        }
        self.engine.step(1)?;
        self.tick += 1;

        // Prediction vs physics, collision, termination.
        self.pred_pose =
            self.planner
                .predict(self.pred_pose, [self.pred_cmd.v(), self.pred_cmd.w()], dt);
        let q = self.engine.qpos()?;
        let truth = [q[0], q[1], q[2]];
        let clearance = self
            .cfg
            .world
            .min_distance([q[0], q[1]], self.engine.time()?)
            - self.cfg.robot_radius;
        self.min_clearance = self.min_clearance.min(clearance);
        self.trace.push(TraceRow {
            tick: self.tick,
            true_pose: truth,
            est_pose: est,
            pred_pose: self.pred_pose,
            diff: pose_diff(self.pred_pose, truth),
            cmd: [cmd.v(), cmd.w()],
            guard: tag,
            safety,
            clearance,
        });
        let e = self.fusion.pose();
        self.outcome = if clearance < 0.0 {
            Outcome::Collision
        } else if (e[0] - self.cfg.goal[0]).hypot(e[1] - self.cfg.goal[1]) < self.cfg.reach_tol {
            Outcome::Reached
        } else if self.stopped_ticks >= self.cfg.safe_stop_hold_ticks {
            Outcome::SafeStop
        } else if self.tick >= self.cfg.max_ticks {
            Outcome::Timeout
        } else {
            Outcome::Running
        };
        Ok(())
    }

    /// Run until the episode ends or `max_ticks`.
    pub fn run(&mut self) -> Result<Report, PortError> {
        while self.outcome == Outcome::Running {
            self.step_tick()?;
        }
        Ok(self.report())
    }

    /// Run exactly `n` more ticks (or until the episode ends).
    pub fn run_ticks(&mut self, n: u64) -> Result<(), PortError> {
        for _ in 0..n {
            self.step_tick()?;
        }
        Ok(())
    }

    /// Summary report.
    pub fn report(&self) -> Report {
        let q = self.engine.clone().qpos().unwrap_or_default();
        Report {
            outcome: self.outcome,
            ticks: self.tick,
            true_goal_error: (q[0] - self.cfg.goal[0]).hypot(q[1] - self.cfg.goal[1]),
            min_clearance: self.min_clearance,
            allowed: self.counts[0],
            fallbacks: self.counts[1],
            safety_ticks: self.counts[2],
            max_speed: self.max_speed,
            max_yaw_rate: self.max_yaw_rate,
            max_pred_err: self
                .trace
                .iter()
                .map(|r| r.diff[0].hypot(r.diff[1]))
                .fold(0.0, f64::max),
            trace_hash: trace_hash(&self.trace),
        }
    }
}

/// FNV-1a over the bit patterns of every trace value.
pub fn trace_hash(trace: &[TraceRow]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |x: u64| {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for r in trace {
        mix(r.tick);
        for v in r
            .true_pose
            .iter()
            .chain(&r.est_pose)
            .chain(&r.pred_pose)
            .chain(&r.diff)
            .chain(&r.cmd)
        {
            mix(v.to_bits());
        }
        mix(r.guard as u64);
        mix(u64::from(r.safety));
        mix(r.clearance.to_bits());
    }
    h
}
