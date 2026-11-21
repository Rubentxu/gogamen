//! The `ContinuousPhysicsEnginePort` boundary.

use std::fmt;

/// Error returned by any engine backend.
#[derive(Debug, Clone, PartialEq)]
pub enum PortError {
    /// A vector had the wrong length or a non-finite value.
    InvalidArgument(String),
    /// A snapshot produced by a different backend / layout was supplied.
    SnapshotMismatch(String),
    /// Backend (process, protocol, engine) failure.
    Backend(String),
}

impl fmt::Display for PortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::SnapshotMismatch(m) => write!(f, "snapshot mismatch: {m}"),
            Self::Backend(m) => write!(f, "backend error: {m}"),
        }
    }
}

impl std::error::Error for PortError {}

/// Dimensions of the engine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    /// Generalized positions.
    pub nq: usize,
    /// Generalized velocities.
    pub nv: usize,
    /// User-controllable inputs.
    pub nu: usize,
    /// Actuator internal states (e.g. motor lag).
    pub na: usize,
}

/// Snapshot of the *complete integration state*.
///
/// `state` is backend-opaque (for MuJoCo it is the `mj_getState` vector with
/// `mjSTATE_INTEGRATION`: time, qpos, qvel, act, warmstart, ctrl, ...). `time` and
/// `ctrl` are duplicated in readable form.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineSnapshot {
    /// Backend tag; `restore` rejects snapshots from another backend.
    pub backend: String,
    /// Simulation time in seconds.
    pub time: f64,
    /// Control vector held at snapshot time.
    pub ctrl: Vec<f64>,
    /// Opaque full integration state.
    pub state: Vec<f64>,
}

/// Continuous-state physics engine boundary used by planners and controllers.
///
/// Contract:
/// * `qpos` for the mobile robot is always `[x, y, yaw]`.
/// * `snapshot` + `restore` + the same `set_ctrl`/`step` sequence reproduces the same
///   state (bit-exact for the Rust solver; within [`MUJOCO_REPLAY_TOLERANCE`](crate::MUJOCO_REPLAY_TOLERANCE)
///   for MuJoCo). Restoring also restores time and actuator/warmstart state.
/// * Object-safe; all methods take `&mut self` because a backend may do I/O.
pub trait ContinuousPhysicsEnginePort {
    /// Human-readable backend name.
    fn name(&self) -> &str;
    /// State dimensions.
    fn dims(&self) -> Dims;
    /// Fixed physics timestep in seconds.
    fn dt(&self) -> f64;
    /// Simulation time in seconds.
    fn time(&mut self) -> Result<f64, PortError>;
    /// Generalized positions.
    fn qpos(&mut self) -> Result<Vec<f64>, PortError>;
    /// Generalized velocities (backend-specific frame; see backend docs).
    fn qvel(&mut self) -> Result<Vec<f64>, PortError>;
    /// Currently held control.
    fn ctrl(&mut self) -> Result<Vec<f64>, PortError>;
    /// Forward speed (m/s) and yaw rate (rad/s) of the robot body.
    fn body_twist(&mut self) -> Result<(f64, f64), PortError>;
    /// Set control `[thrust, torque]` (clamped to the actuator range). Held until changed.
    fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), PortError>;
    /// Advance `n` physics steps with the currently held control.
    fn step(&mut self, n: usize) -> Result<(), PortError>;
    /// Capture the full integration state.
    fn snapshot(&mut self) -> Result<EngineSnapshot, PortError>;
    /// Restore a snapshot taken from the same backend.
    fn restore(&mut self, snapshot: &EngineSnapshot) -> Result<(), PortError>;
    /// Reset to the initial state (time 0, zero velocity, zero control).
    fn reset(&mut self) -> Result<(), PortError>;
}

pub(crate) fn check_finite(name: &str, v: &[f64], len: usize) -> Result<(), PortError> {
    if v.len() != len {
        return Err(PortError::InvalidArgument(format!(
            "{name}: expected {len} values, got {}",
            v.len()
        )));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(PortError::InvalidArgument(format!(
            "{name}: non-finite value"
        )));
    }
    Ok(())
}
