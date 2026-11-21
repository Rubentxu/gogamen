//! Bounded velocity command type and the low-level velocity controller (ramps + feedforward + P).

use crate::unicycle::{MAX_THRUST, MAX_TORQUE, UnicycleParams};

/// Velocity limits of the robot.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityLimits {
    /// Max |forward speed| (m/s).
    pub v_max: f64,
    /// Max |yaw rate| (rad/s).
    pub w_max: f64,
}

impl Default for VelocityLimits {
    fn default() -> Self {
        Self {
            v_max: 1.0,
            w_max: 1.5,
        }
    }
}

/// A velocity command that is within [`VelocityLimits`] *by construction*: the only ways to build one
/// are [`BoundedCommand::new`] (saturates; NaN becomes 0) and [`BoundedCommand::STOP`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedCommand {
    v: f64,
    w: f64,
}

impl BoundedCommand {
    /// Zero velocity.
    pub const STOP: Self = Self { v: 0.0, w: 0.0 };

    /// Saturate `(v, w)` into `limits`; non-finite inputs are mapped to 0 (infinities saturate).
    pub fn new(v: f64, w: f64, limits: &VelocityLimits) -> Self {
        fn sat(x: f64, m: f64) -> f64 {
            if x.is_nan() { 0.0 } else { x.clamp(-m, m) }
        }
        Self {
            v: sat(v, limits.v_max),
            w: sat(w, limits.w_max),
        }
    }

    /// Forward speed (m/s).
    pub fn v(self) -> f64 {
        self.v
    }

    /// Yaw rate (rad/s).
    pub fn w(self) -> f64 {
        self.w
    }

    /// As a `[v, w]` action vector.
    pub fn to_vec(self) -> Vec<f64> {
        vec![self.v, self.w]
    }
}

/// Low-level controller parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LowLevelConfig {
    /// Velocity limits applied to the reference (second line of defence after the guard).
    pub limits: VelocityLimits,
    /// Forward acceleration ramp (m/s^2).
    pub a_accel: f64,
    /// Forward braking ramp (m/s^2), used whenever |v_ref| decreases.
    pub a_brake: f64,
    /// Yaw acceleration ramp (rad/s^2).
    pub alpha: f64,
    /// P gain on forward speed error (N per m/s).
    pub kp_v: f64,
    /// P gain on yaw-rate error (N m per rad/s).
    pub kp_w: f64,
    /// Plant mass (kg) for feedforward.
    pub mass: f64,
    /// Plant yaw inertia (kg m^2).
    pub inertia: f64,
    /// Plant linear damping.
    pub lin_damping: f64,
    /// Plant angular damping.
    pub ang_damping: f64,
}

impl LowLevelConfig {
    /// Feedforward from the plant parameters, with tuned feedback gains.
    pub fn for_plant(p: &UnicycleParams) -> Self {
        Self {
            limits: VelocityLimits::default(),
            a_accel: 1.0,
            a_brake: 2.0,
            alpha: 4.0,
            kp_v: 4.0,
            kp_w: 0.3,
            mass: p.mass,
            inertia: p.inertia,
            lin_damping: p.lin_damping,
            ang_damping: p.ang_damping,
        }
    }
}

fn ramp(cur: f64, target: f64, up: f64, down: f64, dt: f64) -> f64 {
    let rate = if target.abs() < cur.abs() || target * cur < 0.0 {
        down
    } else {
        up
    };
    let step = rate * dt;
    cur + (target - cur).clamp(-step, step)
}

/// Velocity reference ramps + feedforward/P tracking, producing `[thrust, torque]`.
#[derive(Debug, Clone)]
pub struct LowLevelController {
    cfg: LowLevelConfig,
    v_ref: f64,
    w_ref: f64,
}

impl LowLevelController {
    /// New controller at rest.
    pub fn new(cfg: LowLevelConfig) -> Self {
        Self {
            cfg,
            v_ref: 0.0,
            w_ref: 0.0,
        }
    }

    /// Current ramped reference `(v, w)`.
    pub fn reference(&self) -> (f64, f64) {
        (self.v_ref, self.w_ref)
    }

    /// One control step: track `cmd` (re-saturated to the limits) given the measured twist.
    /// Output is clamped to the actuator range.
    pub fn step(&mut self, cmd: BoundedCommand, meas_v: f64, meas_w: f64, dt: f64) -> [f64; 2] {
        let c = self.cfg;
        let cmd = BoundedCommand::new(cmd.v(), cmd.w(), &c.limits);
        let (pv, pw) = (self.v_ref, self.w_ref);
        self.v_ref = ramp(pv, cmd.v(), c.a_accel, c.a_brake, dt);
        self.w_ref = ramp(pw, cmd.w(), c.alpha, 2.0 * c.alpha, dt);
        let (dv, dw) = ((self.v_ref - pv) / dt, (self.w_ref - pw) / dt);
        let thrust = c.lin_damping * self.v_ref + c.mass * dv + c.kp_v * (self.v_ref - meas_v);
        let torque = c.ang_damping * self.w_ref + c.inertia * dw + c.kp_w * (self.w_ref - meas_w);
        [
            thrust.clamp(-MAX_THRUST, MAX_THRUST),
            torque.clamp(-MAX_TORQUE, MAX_TORQUE),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::ContinuousPhysicsEnginePort;
    use crate::unicycle::UnicycleEngine;

    #[test]
    fn bounded_command_always_within_limits() {
        let l = VelocityLimits::default();
        let vals = [
            0.0,
            0.5,
            -3.0,
            1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -0.0,
        ];
        for &v in &vals {
            for &w in &vals {
                let c = BoundedCommand::new(v, w, &l);
                assert!(c.v().abs() <= l.v_max && c.w().abs() <= l.w_max);
                assert!(c.v().is_finite() && c.w().is_finite());
            }
        }
        assert_eq!(BoundedCommand::new(f64::NAN, 9.0, &l).v(), 0.0);
        assert_eq!(BoundedCommand::new(9.0, -9.0, &l).w(), -l.w_max);
    }

    #[test]
    fn tracks_velocity_with_ramps_and_bounds() {
        let mut e = UnicycleEngine::default();
        let mut c = LowLevelController::new(LowLevelConfig::for_plant(&e.params()));
        let cmd = BoundedCommand::new(1.0, 0.5, &VelocityLimits::default());
        let mut v_at_half_s = 0.0;
        for k in 0..400 {
            let (v, w) = e.body_twist().unwrap();
            e.set_ctrl(&c.step(cmd, v, w, 0.01)).unwrap();
            e.step(1).unwrap();
            if k == 49 {
                v_at_half_s = e.body_twist().unwrap().0;
            }
        }
        let (v, w) = e.body_twist().unwrap();
        assert!((v - 1.0).abs() < 0.05 && (w - 0.5).abs() < 0.05, "{v} {w}");
        // accel ramp 1 m/s^2 => about 0.5 m/s after 0.5 s (not an instant jump to 1.0)
        assert!(v_at_half_s > 0.35 && v_at_half_s < 0.65, "{v_at_half_s}");
        // brake
        for _ in 0..300 {
            let (v, w) = e.body_twist().unwrap();
            e.set_ctrl(&c.step(BoundedCommand::STOP, v, w, 0.01))
                .unwrap();
            e.step(1).unwrap();
        }
        assert!(e.body_twist().unwrap().0.abs() < 0.02);
    }
}
