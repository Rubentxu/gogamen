//! Deterministic pure-Rust reference solver: dynamic unicycle (differential-drive abstraction).
//!
//! State: `qpos=[x,y,yaw]`, `qvel=[v,w]` (body forward speed, yaw rate), `act=[a_thrust,a_torque]`
//! (first-order actuator lag, like a MuJoCo `filter` actuator), `time`, and held `ctrl`.
//! Semi-implicit Euler with fixed `dt`; no hidden state beyond that listed, so snapshots are exact.

use crate::port::{ContinuousPhysicsEnginePort, Dims, EngineSnapshot, PortError, check_finite};

/// Maximum thrust (N).
pub const MAX_THRUST: f64 = 5.0;
/// Maximum torque (N m).
pub const MAX_TORQUE: f64 = 1.0;

/// Physical parameters of the unicycle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnicycleParams {
    /// Mass (kg).
    pub mass: f64,
    /// Yaw inertia (kg m^2).
    pub inertia: f64,
    /// Linear damping (N s/m).
    pub lin_damping: f64,
    /// Angular damping (N m s/rad).
    pub ang_damping: f64,
    /// Actuator time constant (s).
    pub tau: f64,
    /// Physics timestep (s).
    pub dt: f64,
}

impl Default for UnicycleParams {
    fn default() -> Self {
        Self {
            mass: 2.0,
            inertia: 0.05,
            lin_damping: 1.0,
            ang_damping: 0.05,
            tau: 0.05,
            dt: 0.01,
        }
    }
}

/// Reference solver, `Clone` + bit-exact deterministic.
#[derive(Debug, Clone, PartialEq)]
pub struct UnicycleEngine {
    params: UnicycleParams,
    time: f64,
    qpos: [f64; 3],
    qvel: [f64; 2],
    act: [f64; 2],
    ctrl: [f64; 2],
}

const TAG: &str = "rust-unicycle";

impl UnicycleEngine {
    /// New engine at the origin.
    pub fn new(params: UnicycleParams) -> Self {
        Self {
            params,
            time: 0.0,
            qpos: [0.0; 3],
            qvel: [0.0; 2],
            act: [0.0; 2],
            ctrl: [0.0; 2],
        }
    }

    /// Engine with an initial pose.
    pub fn with_pose(params: UnicycleParams, x: f64, y: f64, yaw: f64) -> Self {
        let mut e = Self::new(params);
        e.qpos = [x, y, yaw];
        e
    }

    /// Actuator state `[a_thrust, a_torque]`.
    pub fn act(&self) -> [f64; 2] {
        self.act
    }

    /// External disturbance: instantaneous velocity kick (`dv` m/s forward, `dw` rad/s yaw).
    /// Not part of the port; used by perturbation tests. Deterministic and snapshot-safe.
    pub fn kick(&mut self, dv: f64, dw: f64) {
        self.qvel[0] += dv;
        self.qvel[1] += dw;
    }

    /// External disturbance: instantaneous pose shove (world-frame `dx`, `dy`, yaw `dyaw`).
    pub fn shove(&mut self, dx: f64, dy: f64, dyaw: f64) {
        self.qpos[0] += dx;
        self.qpos[1] += dy;
        self.qpos[2] += dyaw;
    }

    /// Physical parameters.
    pub fn params(&self) -> UnicycleParams {
        self.params
    }

    fn step_once(&mut self) {
        let p = self.params;
        let dt = p.dt;
        for i in 0..2 {
            self.act[i] += dt * (self.ctrl[i] - self.act[i]) / p.tau;
        }
        self.qvel[0] += dt * (self.act[0] - p.lin_damping * self.qvel[0]) / p.mass;
        self.qvel[1] += dt * (self.act[1] - p.ang_damping * self.qvel[1]) / p.inertia;
        let (s, c) = self.qpos[2].sin_cos();
        self.qpos[0] += dt * self.qvel[0] * c;
        self.qpos[1] += dt * self.qvel[0] * s;
        self.qpos[2] += dt * self.qvel[1];
        self.time += dt;
    }
}

impl Default for UnicycleEngine {
    fn default() -> Self {
        Self::new(UnicycleParams::default())
    }
}

impl ContinuousPhysicsEnginePort for UnicycleEngine {
    fn name(&self) -> &str {
        TAG
    }
    fn dims(&self) -> Dims {
        Dims {
            nq: 3,
            nv: 2,
            nu: 2,
            na: 2,
        }
    }
    fn dt(&self) -> f64 {
        self.params.dt
    }
    fn time(&mut self) -> Result<f64, PortError> {
        Ok(self.time)
    }
    fn qpos(&mut self) -> Result<Vec<f64>, PortError> {
        Ok(self.qpos.to_vec())
    }
    fn qvel(&mut self) -> Result<Vec<f64>, PortError> {
        Ok(self.qvel.to_vec())
    }
    fn ctrl(&mut self) -> Result<Vec<f64>, PortError> {
        Ok(self.ctrl.to_vec())
    }
    fn body_twist(&mut self) -> Result<(f64, f64), PortError> {
        Ok((self.qvel[0], self.qvel[1]))
    }
    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PortError> {
        check_finite("ctrl", ctrl, 2)?;
        self.ctrl = [
            ctrl[0].clamp(-MAX_THRUST, MAX_THRUST),
            ctrl[1].clamp(-MAX_TORQUE, MAX_TORQUE),
        ];
        Ok(())
    }
    fn step(&mut self, n: usize) -> Result<(), PortError> {
        for _ in 0..n {
            self.step_once();
        }
        Ok(())
    }
    fn snapshot(&mut self) -> Result<EngineSnapshot, PortError> {
        let mut state = Vec::with_capacity(8);
        state.extend_from_slice(&self.qpos);
        state.extend_from_slice(&self.qvel);
        state.extend_from_slice(&self.act);
        Ok(EngineSnapshot {
            backend: TAG.into(),
            time: self.time,
            ctrl: self.ctrl.to_vec(),
            state,
        })
    }
    fn restore(&mut self, s: &EngineSnapshot) -> Result<(), PortError> {
        if s.backend != TAG || s.state.len() != 7 || s.ctrl.len() != 2 {
            return Err(PortError::SnapshotMismatch(format!(
                "expected {TAG} snapshot with 7 state values, got {} / {}",
                s.backend,
                s.state.len()
            )));
        }
        self.time = s.time;
        self.qpos.copy_from_slice(&s.state[0..3]);
        self.qvel.copy_from_slice(&s.state[3..5]);
        self.act.copy_from_slice(&s.state[5..7]);
        self.ctrl.copy_from_slice(&s.ctrl);
        Ok(())
    }
    fn reset(&mut self) -> Result<(), PortError> {
        *self = Self::new(self.params);
        Ok(())
    }
}
