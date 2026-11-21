//! Deterministic, seeded fault injection for the navigation stack (GOG-80).
//!
//! Three fault classes: a stale/frozen sensor, an actuator that times out (commands dropped or
//! delayed) and a planner whose prediction model has wrong parameters. Everything is a pure
//! function of `(plan, seed, tick)`; nothing here touches the wall clock or an OS RNG.

use std::collections::VecDeque;

use gogamen_core::BudgetTracker;

use super::planner::{NavObservation, NavPlanner, kinematic_step};
use super::sensors::Scan;
use super::sim::TraceRow;
use crate::control::Rng;

/// How the sensor misbehaves once the fault is active.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SensorFaultMode {
    /// The last good scan is re-delivered with its original (old) timestamp.
    OldTimestamp,
    /// The last good values are re-delivered with a fresh timestamp (stuck-at, repeated values).
    FrozenValues,
    /// Like `FrozenValues` but with a small seeded dither (amplitude in m) added to each range,
    /// so the values are never bit-identical. A fault the repeat detector cannot see.
    FrozenDithered {
        /// Dither amplitude (m).
        amplitude: f64,
    },
}

/// Sensor fault window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensorFault {
    /// First faulty tick.
    pub from_tick: u64,
    /// Fault ends before this tick (`None` = permanent).
    pub until_tick: Option<u64>,
    /// Behaviour.
    pub mode: SensorFaultMode,
}

/// How the actuator misbehaves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActuatorFaultMode {
    /// Commands are discarded; the actuator keeps holding the last applied command.
    Drop,
    /// Commands are applied `ticks` late (and acknowledged late).
    Delay {
        /// Delay in ticks.
        ticks: u64,
    },
}

/// Actuator fault window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActuatorFault {
    /// First faulty tick.
    pub from_tick: u64,
    /// Number of faulty ticks (`None` = permanent).
    pub duration_ticks: Option<u64>,
    /// Behaviour.
    pub mode: ActuatorFaultMode,
}

/// All faults of one scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaultPlan {
    /// Sensor fault.
    pub sensor: Option<SensorFault>,
    /// Actuator fault.
    pub actuator: Option<ActuatorFault>,
    /// Residual-risk switch: the independent e-stop channel itself is dead (it goes through the
    /// same faulty actuator path). Only used to demonstrate what the policy cannot cover.
    pub estop_broken: bool,
    /// Seed of the injection RNG (dither).
    pub seed: u64,
}

impl FaultPlan {
    /// No faults.
    pub fn none() -> Self {
        Self {
            sensor: None,
            actuator: None,
            estop_broken: false,
            seed: 0,
        }
    }

    /// Seeded onset jitter: a tick in `[base, base + 40)` that depends only on `seed`.
    pub fn onset(seed: u64, base: u64) -> u64 {
        base + (Rng::new(seed ^ 0xFA17_FA17).unit() * 40.0) as u64
    }

    /// Sensor fault whose onset is derived from `seed`.
    pub fn sensor_seeded(seed: u64, mode: SensorFaultMode) -> Self {
        Self {
            sensor: Some(SensorFault {
                from_tick: Self::onset(seed, 120),
                until_tick: None,
                mode,
            }),
            seed,
            ..Self::none()
        }
    }

    /// Actuator fault whose onset is derived from `seed`.
    pub fn actuator_seeded(seed: u64, mode: ActuatorFaultMode, duration: Option<u64>) -> Self {
        Self {
            actuator: Some(ActuatorFault {
                from_tick: Self::onset(seed, 120),
                duration_ticks: duration,
                mode,
            }),
            seed,
            ..Self::none()
        }
    }
}

/// Sensor-side injector (sits between the sensor and the fusion).
#[derive(Debug, Clone)]
pub struct SensorInjector {
    fault: Option<SensorFault>,
    last_good: Option<Scan>,
    rng: Rng,
}

impl SensorInjector {
    /// New injector.
    pub fn new(plan: &FaultPlan) -> Self {
        Self {
            fault: plan.sensor,
            last_good: None,
            rng: Rng::new(plan.seed ^ 0x5E75_0001),
        }
    }

    /// Pass the sensor output through the fault model.
    pub fn filter(&mut self, tick: u64, scan: Option<Scan>) -> Option<Scan> {
        let active = self
            .fault
            .is_some_and(|f| tick >= f.from_tick && f.until_tick.is_none_or(|u| tick < u));
        let scan = scan?;
        if !active {
            self.last_good = Some(scan.clone());
            return Some(scan);
        }
        let f = self.fault.expect("active implies a fault");
        let held = self.last_good.clone().unwrap_or(scan);
        Some(match f.mode {
            SensorFaultMode::OldTimestamp => held,
            SensorFaultMode::FrozenValues => Scan {
                sample_tick: tick,
                ranges: held.ranges,
            },
            SensorFaultMode::FrozenDithered { amplitude } => Scan {
                sample_tick: tick,
                ranges: held
                    .ranges
                    .iter()
                    .map(|r| r + amplitude * self.rng.normal())
                    .collect(),
            },
        })
    }
}

/// What the actuator physically applies at a tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Applied {
    /// New control to apply (`None` = keep holding the previous one).
    pub ctrl: Option<[f64; 2]>,
    /// Stamp of the command now in effect (acknowledgement / echo).
    pub echo_tick: Option<u64>,
}

/// Actuator-side injector (sits between the low-level controller and the engine).
#[derive(Debug, Clone)]
pub struct ActuatorInjector {
    fault: Option<ActuatorFault>,
    line: VecDeque<(u64, u64, [f64; 2])>,
    echo: Option<u64>,
}

impl ActuatorInjector {
    /// New injector.
    pub fn new(plan: &FaultPlan) -> Self {
        Self {
            fault: plan.actuator,
            line: VecDeque::new(),
            echo: None,
        }
    }

    /// Latest acknowledgement.
    pub fn echo(&self) -> Option<u64> {
        self.echo
    }

    /// Issue command `u` at `tick`.
    pub fn issue(&mut self, tick: u64, u: [f64; 2]) -> Applied {
        let active = self.fault.filter(|f| {
            tick >= f.from_tick && f.duration_ticks.is_none_or(|d| tick < f.from_tick + d)
        });
        match active.map(|f| f.mode) {
            None => {
                // Healthy: apply now. Anything still in the delay line is superseded.
                self.line.clear();
                self.echo = Some(tick);
                Applied {
                    ctrl: Some(u),
                    echo_tick: self.echo,
                }
            }
            Some(ActuatorFaultMode::Drop) => Applied {
                ctrl: None,
                echo_tick: self.echo,
            },
            Some(ActuatorFaultMode::Delay { ticks }) => {
                self.line.push_back((tick + ticks, tick, u));
                let mut ctrl = None;
                while self.line.front().is_some_and(|(due, _, _)| *due <= tick) {
                    let (_, stamp, v) = self.line.pop_front().expect("checked");
                    ctrl = Some(v);
                    self.echo = Some(stamp);
                }
                Applied {
                    ctrl,
                    echo_tick: self.echo,
                }
            }
        }
    }
}

/// Wrong parameters of the planner's internal model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelError {
    /// The planner believes `cmd -> motion` has this gain (1.0 = correct). A wrong plant mass
    /// shows up to the planner only through this effective gain; mass is not modelled separately.
    pub gain: f64,
    /// The planner believes obstacles are this much farther than the sensor says (m), i.e. an
    /// underestimated robot radius. Affects planning only, never the prediction.
    pub radius_error: f64,
    /// Tick from which the wrong model applies (before it the wrapper is transparent).
    pub from_tick: u64,
}

/// Wraps a planner and corrupts its model from `from_tick` on.
#[derive(Debug, Clone)]
pub struct WrongModelPlanner<P: NavPlanner> {
    inner: P,
    /// Model error.
    pub error: ModelError,
    /// Output clamp so that proposals stay representable (v, w).
    pub v_max: f64,
    /// Yaw-rate clamp.
    pub w_max: f64,
    tick: u64,
}

impl<P: NavPlanner> WrongModelPlanner<P> {
    /// Wrap `inner`.
    pub fn new(inner: P, error: ModelError, v_max: f64, w_max: f64) -> Self {
        Self {
            inner,
            error,
            v_max,
            w_max,
            tick: 0,
        }
    }

    fn active(&self) -> bool {
        self.tick >= self.error.from_tick
    }
}

impl<P: NavPlanner> NavPlanner for WrongModelPlanner<P> {
    fn name(&self) -> String {
        format!("wrong-model({})", self.inner.name())
    }

    fn plan(&mut self, obs: &NavObservation<'_>, budget: &mut BudgetTracker) -> Option<Vec<f64>> {
        self.tick = obs.tick;
        if !self.active() {
            return self.inner.plan(obs, budget);
        }
        let inflated: Vec<f64> = obs
            .ranges
            .iter()
            .map(|r| {
                if *r < obs.max_range {
                    (r + self.error.radius_error).min(obs.max_range)
                } else {
                    *r
                }
            })
            .collect();
        let o2 = NavObservation {
            ranges: &inflated,
            ..obs.clone()
        };
        let out = self.inner.plan(&o2, budget)?;
        if out.len() != 2 {
            return Some(out);
        }
        // To obtain velocity v the planner believes it must command v / gain.
        let g = self.error.gain;
        Some(vec![
            (out[0] / g).clamp(-self.v_max, self.v_max),
            (out[1] / g).clamp(-self.w_max, self.w_max),
        ])
    }

    fn predict(&self, pose: [f64; 3], cmd: [f64; 2], dt: f64) -> [f64; 3] {
        if self.active() {
            kinematic_step(pose, cmd[0] * self.error.gain, cmd[1] * self.error.gain, dt)
        } else {
            self.inner.predict(pose, cmd, dt)
        }
    }
}

/// Corridor with an additional full-width static wall at x = 3.0..3.2 (it is in the known map).
/// The goal is unreachable, so a healthy stack ends in a safe stop in front of it, while a robot
/// driving blind (stale sensor, uncommandable actuator) hits it. Used by the GOG-80 fixtures.
pub fn blocked_corridor(
    seed: u64,
    faults: FaultPlan,
    failsafe: Option<super::failsafe::FailSafeConfig>,
) -> super::sim::ScenarioConfig {
    let mut c = super::sim::ScenarioConfig::corridor(seed);
    c.world.statics.push(super::geometry::Shape::Aabb {
        min: [3.0, -1.5],
        max: [3.2, 1.5],
    });
    c.faults = faults;
    c.failsafe = failsafe;
    c
}

/// Max true speed (m/s, from consecutive true poses) over trace rows with `tick >= from_tick`.
pub fn max_speed_from(trace: &[TraceRow], from_tick: u64, dt: f64) -> f64 {
    trace
        .windows(2)
        .filter(|w| w[1].tick >= from_tick)
        .map(|w| {
            (w[1].true_pose[0] - w[0].true_pose[0]).hypot(w[1].true_pose[1] - w[0].true_pose[1])
                / dt
        })
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(t: u64, v: f64) -> Option<Scan> {
        Some(Scan {
            sample_tick: t,
            ranges: vec![v, v + 1.0],
        })
    }

    #[test]
    fn sensor_modes_are_deterministic_and_distinct() {
        let mk = |mode| {
            let plan = FaultPlan {
                sensor: Some(SensorFault {
                    from_tick: 10,
                    until_tick: None,
                    mode,
                }),
                seed: 9,
                ..FaultPlan::none()
            };
            let mut inj = SensorInjector::new(&plan);
            (0..20u64)
                .step_by(5)
                .map(|t| inj.filter(t, scan(t, t as f64)).unwrap())
                .collect::<Vec<_>>()
        };
        let old = mk(SensorFaultMode::OldTimestamp);
        assert_eq!(old[2].sample_tick, 5);
        assert_eq!(old[2].ranges, old[1].ranges);
        let frozen = mk(SensorFaultMode::FrozenValues);
        assert_eq!(frozen[3].sample_tick, 15);
        assert_eq!(frozen[3].ranges, frozen[1].ranges);
        let d = mk(SensorFaultMode::FrozenDithered { amplitude: 0.01 });
        assert_ne!(d[3].ranges, d[1].ranges);
        assert_eq!(d, mk(SensorFaultMode::FrozenDithered { amplitude: 0.01 }));
    }

    #[test]
    fn actuator_drop_holds_and_delay_acks_late() {
        let plan = FaultPlan::actuator_seeded(1, ActuatorFaultMode::Drop, Some(5));
        let from = plan.actuator.unwrap().from_tick;
        let mut a = ActuatorInjector::new(&plan);
        for t in 0..from {
            assert_eq!(a.issue(t, [1.0, 0.0]).echo_tick, Some(t));
        }
        let r = a.issue(from, [2.0, 0.0]);
        assert_eq!((r.ctrl, r.echo_tick), (None, Some(from - 1)));
        let r = a.issue(from + 5, [3.0, 0.0]);
        assert_eq!((r.ctrl, r.echo_tick), (Some([3.0, 0.0]), Some(from + 5)));

        let plan = FaultPlan {
            actuator: Some(ActuatorFault {
                from_tick: 0,
                duration_ticks: None,
                mode: ActuatorFaultMode::Delay { ticks: 3 },
            }),
            ..FaultPlan::none()
        };
        let mut a = ActuatorInjector::new(&plan);
        assert_eq!(a.issue(0, [0.0; 2]).ctrl, None);
        let r = a.issue(3, [9.0, 0.0]);
        assert_eq!((r.ctrl, r.echo_tick), (Some([0.0; 2]), Some(0)));
    }

    #[test]
    fn onset_is_seeded() {
        assert_eq!(FaultPlan::onset(3, 100), FaultPlan::onset(3, 100));
        assert!((100..140).contains(&FaultPlan::onset(7, 100)));
    }
}
