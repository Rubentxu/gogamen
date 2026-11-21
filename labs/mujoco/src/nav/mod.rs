//! Simulated mobile robot navigation stack on the pure-Rust unicycle backend: odometry, range
//! sensors with latency, moving obstacles, fusion, low-level control, an independent action guard
//! (`gogamen_core::ActionGuard`) plus safety stop, and a prediction-vs-physics trace.

pub mod failsafe;
pub mod faults;
pub mod fusion;
pub mod geometry;
pub mod lowlevel;
pub mod planner;
pub mod sensors;
pub mod sim;

pub use failsafe::{
    Domain, FailSafe, FailSafeConfig, Fallback, Observables, Override, RearmRefused,
    SIMULATION_ONLY, TripEvent,
};
pub use faults::{
    ActuatorFault, ActuatorFaultMode, ActuatorInjector, FaultPlan, ModelError, SensorFault,
    SensorFaultMode, SensorInjector, WrongModelPlanner, blocked_corridor, max_speed_from,
};
pub use fusion::{Fusion, FusionConfig};
pub use geometry::{MovingObstacle, Shape, World};
pub use lowlevel::{BoundedCommand, LowLevelConfig, LowLevelController, VelocityLimits};
pub use planner::{
    AdversarialPlanner, GridPlanner, Misbehaviour, NavObservation, NavPlanner, kinematic_step,
};
pub use sensors::{OdomConfig, Odometry, RangeConfig, RangeSensor, Scan};
pub use sim::{
    GuardTag, Outcome, Perturbation, Report, SafetyConfig, ScenarioConfig, Sim, SimSnapshot,
    TraceRow, pose_diff, trace_hash,
};
