//! External fail-safe policy (GOG-80).
//!
//! This module is deliberately separate from the planner: it has no reference to any
//! [`NavPlanner`](super::NavPlanner), it only receives plain observable data
//! ([`Observables`]) and returns an [`Override`] that can only make the robot *less* mobile
//! (stop or cap the speed). Each failure domain has an explicit detector and an explicit fallback
//! (see [`FailSafeConfig::fallback`]); once tripped, a domain stays latched until an explicit
//! [`FailSafe::rearm`] succeeds.
//!
//! This is a simulation-only research mechanism. It is NOT a certified safety function and makes
//! no claim of conformance to any safety standard (see the README, "Fault injection").

use std::collections::VecDeque;

/// This crate path is simulation only: nothing in the fail-safe or fault-injection code talks to
/// hardware, and no feature or API enables it. Guarded by `tests/fault_injection.rs`.
pub const SIMULATION_ONLY: bool = true;

/// Failure domains the policy monitors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    /// Stale or frozen sensor data.
    Sensor,
    /// Actuator does not acknowledge (timeout / dropped / delayed commands).
    Actuator,
    /// Planner prediction model diverges from the observed motion.
    Model,
}

impl Domain {
    /// All domains, in table order.
    pub const ALL: [Domain; 3] = [Domain::Sensor, Domain::Actuator, Domain::Model];

    fn idx(self) -> usize {
        match self {
            Domain::Sensor => 0,
            Domain::Actuator => 1,
            Domain::Model => 2,
        }
    }
}

/// Fallback applied while a domain is latched.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fallback {
    /// Command STOP through the normal command path.
    Stop,
    /// Command STOP and assert the independent e-stop channel (active braking that bypasses the
    /// command path). Needed when the command path itself is the failed element.
    StopAndEStop,
    /// Cap |forward speed| (m/s); yaw is left to the other layers.
    ReduceSpeed(f64),
}

/// Thresholds and the domain -> fallback table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FailSafeConfig {
    /// Sensor: newest delivered scan older than this many ticks trips the domain.
    pub max_scan_age: u64,
    /// Sensor: this many consecutive delivered scans bit-identical to their predecessor trip the domain.
    pub max_repeats: u32,
    /// Actuator: the acknowledged command lags the issued one by more than this many ticks.
    pub max_echo_lag: u64,
    /// Model: window length (ticks) of the prediction-residual test.
    pub residual_window: usize,
    /// Model: position residual (m) over the window above which a tick counts as a violation.
    pub residual_threshold: f64,
    /// Model: consecutive violating ticks needed to trip.
    pub residual_ticks: u32,
    /// Explicit fallback table (index order = [`Domain::ALL`]).
    pub fallback: [Fallback; 3],
}

impl Default for FailSafeConfig {
    fn default() -> Self {
        Self {
            max_scan_age: 15,
            max_repeats: 3,
            max_echo_lag: 3,
            residual_window: 50,
            residual_threshold: 0.18,
            residual_ticks: 3,
            fallback: [
                Fallback::Stop,
                Fallback::StopAndEStop,
                Fallback::ReduceSpeed(0.3),
            ],
        }
    }
}

/// Observable data for one tick. Contains no planner handle and no ground truth.
#[derive(Debug, Clone, Copy)]
pub struct Observables<'a> {
    /// Current tick.
    pub tick: u64,
    /// Scan delivered by the sensor at this tick, if any: `(sample_tick, ranges)`.
    pub scan: Option<(u64, &'a [f64])>,
    /// Tick stamp of the command the actuator last acknowledged (`None` before the first ack).
    pub echo_tick: Option<u64>,
    /// Fused position estimate `[x, y]`.
    pub est_xy: [f64; 2],
    /// Displacement over this tick predicted by the planner's model for the command actually executed.
    pub pred_step: [f64; 2],
}

/// What the policy forces this tick.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Override {
    /// Force a stop command.
    pub stop: bool,
    /// Assert the independent e-stop channel.
    pub estop: bool,
    /// Forward-speed cap (m/s), if any.
    pub speed_cap: Option<f64>,
}

/// Why a re-arm was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RearmRefused {
    /// A fault condition is still observed in this domain.
    StillFaulted(Domain),
}

/// A trip event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TripEvent {
    /// Tick of detection.
    pub tick: u64,
    /// Domain.
    pub domain: Domain,
}

/// The policy state machine.
#[derive(Debug, Clone)]
pub struct FailSafe {
    cfg: FailSafeConfig,
    latched: [bool; 3],
    condition: [bool; 3],
    newest_sample: Option<u64>,
    last_ranges: Option<Vec<f64>>,
    repeats: u32,
    residual_run: u32,
    window: VecDeque<([f64; 2], [f64; 2])>,
    residual: f64,
    events: Vec<TripEvent>,
    active_ticks: u64,
    ticks: u64,
}

impl FailSafe {
    /// New armed policy.
    pub fn new(cfg: FailSafeConfig) -> Self {
        Self {
            cfg,
            latched: [false; 3],
            condition: [false; 3],
            newest_sample: None,
            last_ranges: None,
            repeats: 0,
            residual_run: 0,
            window: VecDeque::new(),
            residual: 0.0,
            events: Vec::new(),
            active_ticks: 0,
            ticks: 0,
        }
    }

    /// Configuration.
    pub fn config(&self) -> &FailSafeConfig {
        &self.cfg
    }

    /// Is `d` latched?
    pub fn is_latched(&self, d: Domain) -> bool {
        self.latched[d.idx()]
    }

    /// Is any domain latched?
    pub fn any_latched(&self) -> bool {
        self.latched.iter().any(|b| *b)
    }

    /// Is the independent e-stop currently demanded?
    pub fn estop_demanded(&self) -> bool {
        Domain::ALL
            .iter()
            .any(|d| self.latched[d.idx()] && self.cfg.fallback[d.idx()] == Fallback::StopAndEStop)
    }

    /// Trip events in order.
    pub fn events(&self) -> &[TripEvent] {
        &self.events
    }

    /// Tick at which `d` first tripped.
    pub fn first_trip(&self, d: Domain) -> Option<u64> {
        self.events.iter().find(|e| e.domain == d).map(|e| e.tick)
    }

    /// Ticks in which any override was active.
    pub fn active_ticks(&self) -> u64 {
        self.active_ticks
    }

    /// Ticks observed.
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Last model residual (m).
    pub fn residual(&self) -> f64 {
        self.residual
    }

    /// Explicit re-arm of every latched domain. Refused while any fault condition is *currently*
    /// observed; on success all latches are cleared and the detectors restart from scratch.
    pub fn rearm(&mut self) -> Result<(), RearmRefused> {
        for d in Domain::ALL {
            if self.latched[d.idx()] && self.condition[d.idx()] {
                return Err(RearmRefused::StillFaulted(d));
            }
        }
        self.latched = [false; 3];
        self.repeats = 0;
        self.residual_run = 0;
        self.window.clear();
        Ok(())
    }

    /// Process one tick of observables and return the override to apply.
    pub fn update(&mut self, o: &Observables<'_>) -> Override {
        self.ticks += 1;
        let c = self.cfg;

        // Sensor: age of the newest stamp and bit-identical repeats.
        if let Some((s, r)) = o.scan {
            self.newest_sample = Some(self.newest_sample.map_or(s, |n| n.max(s)));
            if self.last_ranges.as_deref() == Some(r) {
                self.repeats += 1;
            } else {
                self.repeats = 0;
            }
            self.last_ranges = Some(r.to_vec());
        }
        let age = self.newest_sample.map(|s| o.tick.saturating_sub(s));
        let sensor_bad = age.is_some_and(|a| a > c.max_scan_age) || self.repeats >= c.max_repeats;

        // Actuator: acknowledged command lagging the issued one (issued at tick-1).
        let act_bad = o
            .echo_tick
            .is_some_and(|e| o.tick.saturating_sub(1).saturating_sub(e) > c.max_echo_lag);

        // Model: windowed open-loop prediction vs estimated displacement.
        self.window.push_back((o.est_xy, o.pred_step));
        if self.window.len() > c.residual_window + 1 {
            self.window.pop_front();
        }
        let mut model_bad = false;
        if self.window.len() > c.residual_window {
            let first = self.window.front().expect("non-empty").0;
            let sum = self
                .window
                .iter()
                .take(c.residual_window)
                .fold([0.0; 2], |a, w| [a[0] + w.1[0], a[1] + w.1[1]]);
            let moved = [o.est_xy[0] - first[0], o.est_xy[1] - first[1]];
            self.residual = (sum[0] - moved[0]).hypot(sum[1] - moved[1]);
            if self.residual > c.residual_threshold {
                self.residual_run += 1;
            } else {
                self.residual_run = 0;
            }
            model_bad = self.residual_run >= c.residual_ticks;
        }

        let bad = [sensor_bad, act_bad, model_bad];
        for d in Domain::ALL {
            let i = d.idx();
            self.condition[i] = bad[i];
            if bad[i] && !self.latched[i] {
                self.latched[i] = true;
                self.events.push(TripEvent {
                    tick: o.tick,
                    domain: d,
                });
            }
        }

        let mut ov = Override::default();
        for d in Domain::ALL {
            if !self.latched[d.idx()] {
                continue;
            }
            match c.fallback[d.idx()] {
                Fallback::Stop => ov.stop = true,
                Fallback::StopAndEStop => {
                    ov.stop = true;
                    ov.estop = true;
                }
                Fallback::ReduceSpeed(v) => {
                    ov.speed_cap = Some(ov.speed_cap.map_or(v, |x: f64| x.min(v)));
                }
            }
        }
        if ov != Override::default() {
            self.active_ticks += 1;
        }
        ov
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(tick: u64) -> Observables<'static> {
        Observables {
            tick,
            scan: None,
            echo_tick: Some(tick.saturating_sub(1)),
            est_xy: [0.0; 2],
            pred_step: [0.0; 2],
        }
    }

    #[test]
    fn healthy_inputs_never_trip() {
        let mut f = FailSafe::new(FailSafeConfig::default());
        for t in 0..500 {
            let r = [t as f64 * 1e-3, 1.0];
            let mut o = obs(t);
            if t % 5 == 0 {
                o.scan = Some((t, &r));
            }
            assert_eq!(f.update(&o), Override::default());
        }
        assert!(f.events().is_empty());
    }

    #[test]
    fn stale_stamp_trips_and_latches_until_rearm() {
        let mut f = FailSafe::new(FailSafeConfig::default());
        let r = [1.0, 2.0];
        let mut o = obs(0);
        o.scan = Some((0, &r));
        f.update(&o);
        let mut tripped = None;
        for t in 1..40 {
            let ov = f.update(&obs(t));
            if ov.stop && tripped.is_none() {
                tripped = Some(t);
            }
        }
        assert_eq!(tripped, Some(16));
        // Fresh data arrives, but the latch holds until an explicit re-arm.
        let r2 = [1.1, 2.1];
        let mut o = obs(40);
        o.scan = Some((40, &r2));
        assert!(f.update(&o).stop);
        assert!(f.rearm().is_ok());
        assert!(!f.any_latched());
    }

    #[test]
    fn rearm_is_refused_while_condition_persists() {
        let mut f = FailSafe::new(FailSafeConfig::default());
        let r = [1.0];
        let mut o = obs(0);
        o.scan = Some((0, &r));
        f.update(&o);
        for t in 1..30 {
            f.update(&obs(t));
        }
        assert_eq!(f.rearm(), Err(RearmRefused::StillFaulted(Domain::Sensor)));
        assert!(f.is_latched(Domain::Sensor));
    }
}
