//! Planner interface (untrusted) and reference planners.
//!
//! The planner only proposes a velocity `[v, w]` (an *objective*, never raw actuator torques). The
//! supervisor treats every proposal as untrusted and routes it through `gogamen_core::ActionGuard`.
//! Planner compute is bounded with `gogamen_core::PlanBudget`/`BudgetTracker` (one "transition" =
//! one candidate evaluated with the planner's own model).

use gogamen_core::BudgetTracker;

use super::lowlevel::{BoundedCommand, VelocityLimits};

/// What the planner sees: the fused estimate, never ground truth.
#[derive(Debug, Clone)]
pub struct NavObservation<'a> {
    /// Current tick.
    pub tick: u64,
    /// Fused pose estimate `[x, y, yaw]`.
    pub pose: [f64; 3],
    /// Fused per-ray ranges.
    pub ranges: &'a [f64],
    /// Ray bearings relative to heading.
    pub angles: &'a [f64],
    /// Sensor maximum range.
    pub max_range: f64,
    /// Goal position.
    pub goal: [f64; 2],
}

/// Planner model: ideal kinematic unicycle step (no actuator lag, no noise).
pub fn kinematic_step(pose: [f64; 3], v: f64, w: f64, dt: f64) -> [f64; 3] {
    let (s, c) = pose[2].sin_cos();
    [pose[0] + v * c * dt, pose[1] + v * s * dt, pose[2] + w * dt]
}

/// A planner. `plan` returns a raw action vector (expected `[v, w]`) or `None` when it has no answer
/// (e.g. it is hung). The result is *not trusted*.
pub trait NavPlanner: Clone {
    /// Name for reports.
    fn name(&self) -> String;
    /// Propose a command for the next decision interval.
    fn plan(&mut self, obs: &NavObservation<'_>, budget: &mut BudgetTracker) -> Option<Vec<f64>>;
    /// The planner's own prediction model, used to record prediction vs. physical result.
    fn predict(&self, pose: [f64; 3], cmd: [f64; 2], dt: f64) -> [f64; 3] {
        kinematic_step(pose, cmd[0], cmd[1], dt)
    }
}

/// Reference planner: evaluates a grid of `(v, w)` candidates over a horizon with the kinematic
/// model, scoring goal distance and fused-range collision risk. Its output is built exclusively
/// through [`BoundedCommand`], so it is within limits by type.
#[derive(Debug, Clone)]
pub struct GridPlanner {
    /// Velocity limits.
    pub limits: VelocityLimits,
    /// Rollout horizon (s).
    pub horizon: f64,
    /// Clearance margin added to a rollout point's range before it counts as free (m).
    pub margin: f64,
}

impl GridPlanner {
    /// Planner with a 1 s horizon and 0.45 m margin.
    pub fn new(limits: VelocityLimits) -> Self {
        Self {
            limits,
            horizon: 1.0,
            margin: 0.45,
        }
    }

    fn risk(&self, obs: &NavObservation<'_>, p: [f64; 3]) -> f64 {
        let (dx, dy) = (p[0] - obs.pose[0], p[1] - obs.pose[1]);
        let dist = dx.hypot(dy);
        if dist < 1e-6 {
            return 0.0;
        }
        let mut rel = dy.atan2(dx) - obs.pose[2];
        rel = (rel + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        // Nearest ray and its immediate neighbours (obstacles are wider than the ray spacing).
        let (mut bi, mut bd) = (0, f64::INFINITY);
        for (i, a) in obs.angles.iter().enumerate() {
            if (a - rel).abs() < bd {
                bd = (a - rel).abs();
                bi = i;
            }
        }
        if bd > 0.3 {
            return 0.0; // outside the sensor FOV: unknown, treated as free
        }
        let lo = bi.saturating_sub(1);
        let hi = (bi + 1).min(obs.ranges.len() - 1);
        let r = obs.ranges[lo..=hi]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        if r < obs.max_range && r < dist + self.margin {
            1.0
        } else {
            0.0
        }
    }
}

impl NavPlanner for GridPlanner {
    fn name(&self) -> String {
        "grid-planner".into()
    }

    fn plan(&mut self, obs: &NavObservation<'_>, budget: &mut BudgetTracker) -> Option<Vec<f64>> {
        let l = self.limits;
        let goal_dist = (obs.goal[0] - obs.pose[0]).hypot(obs.goal[1] - obs.pose[1]);
        let vcap = l.v_max.min(goal_dist * 1.5);
        let mut best: Option<(f64, f64, f64)> = None;
        for vi in 0..4 {
            let v = vcap * vi as f64 / 3.0;
            for wi in 0..9 {
                let w = -l.w_max + 2.0 * l.w_max * wi as f64 / 8.0;
                if !budget.try_consume_transition() {
                    break;
                }
                let mut p = obs.pose;
                let mut collide = 0.0;
                let n = 10;
                for k in 1..=n {
                    p = kinematic_step(p, v, w, self.horizon / n as f64);
                    if k % 2 == 0 {
                        collide += self.risk(obs, p);
                    }
                }
                let bearing = (obs.goal[1] - p[1]).atan2(obs.goal[0] - p[0]);
                let herr = (bearing - p[2] + std::f64::consts::PI)
                    .rem_euclid(std::f64::consts::TAU)
                    - std::f64::consts::PI;
                let cost = (obs.goal[0] - p[0]).hypot(obs.goal[1] - p[1])
                    + 5.0 * collide
                    + 0.05 * w.abs()
                    + 0.3 * herr.abs();
                if best.is_none_or(|b| cost < b.0) {
                    best = Some((cost, v, w));
                }
            }
        }
        best.map(|(_, v, w)| BoundedCommand::new(v, w, &l).to_vec())
    }
}

/// Misbehaviour modes for the adversarial planner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Misbehaviour {
    /// Always proposes a huge, out-of-limit velocity.
    OutOfBounds,
    /// Proposes NaN.
    Nan,
    /// Proposes a vector of the wrong dimension.
    WrongDim,
    /// Never answers (models a hung planner: the supervisor does not wait for it).
    Hang,
    /// Proposes valid full-speed straight-ahead, ignoring every obstacle.
    Charge,
    /// Behaves like [`GridPlanner`] until `after_tick`, then hangs.
    HangAfter(u64),
}

/// Adversarial planner for guard / safety tests.
#[derive(Debug, Clone)]
pub struct AdversarialPlanner {
    /// What to do wrong.
    pub mode: Misbehaviour,
    inner: GridPlanner,
}

impl AdversarialPlanner {
    /// New adversarial planner.
    pub fn new(mode: Misbehaviour, limits: VelocityLimits) -> Self {
        Self {
            mode,
            inner: GridPlanner::new(limits),
        }
    }
}

impl NavPlanner for AdversarialPlanner {
    fn name(&self) -> String {
        format!("adversarial-{:?}", self.mode)
    }

    fn plan(&mut self, obs: &NavObservation<'_>, budget: &mut BudgetTracker) -> Option<Vec<f64>> {
        match self.mode {
            Misbehaviour::OutOfBounds => Some(vec![50.0, 40.0]),
            Misbehaviour::Nan => Some(vec![f64::NAN, 0.0]),
            Misbehaviour::WrongDim => Some(vec![1.0, 0.0, 0.0]),
            Misbehaviour::Hang => None,
            Misbehaviour::Charge => {
                Some(BoundedCommand::new(self.inner.limits.v_max, 0.0, &self.inner.limits).to_vec())
            }
            Misbehaviour::HangAfter(t) => {
                if obs.tick >= t {
                    None
                } else {
                    self.inner.plan(obs, budget)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::PlanBudget;

    #[test]
    fn planner_budget_is_a_hard_cap_and_output_is_bounded() {
        let l = VelocityLimits::default();
        let mut p = GridPlanner::new(l);
        let angles = [-0.5, 0.0, 0.5];
        let ranges = [3.0, 3.0, 3.0];
        let obs = NavObservation {
            tick: 0,
            pose: [0.0; 3],
            ranges: &ranges,
            angles: &angles,
            max_range: 3.0,
            goal: [5.0, 0.0],
        };
        let mut b = PlanBudget::with_transitions(10).tracker();
        let out = p.plan(&obs, &mut b).unwrap();
        assert_eq!(b.transitions_used(), 10);
        assert!(out[0].abs() <= l.v_max && out[1].abs() <= l.w_max);
        let mut b = PlanBudget::with_transitions(0).tracker();
        assert!(p.plan(&obs, &mut b).is_none());
    }
}
