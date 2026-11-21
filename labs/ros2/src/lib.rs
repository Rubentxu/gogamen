//! ROS 2 sim/HIL spike (GOG-81): typed contract, robot-side watchdog, deterministic emulator.
//!
//! SIMULATION ONLY. Nothing here links ROS, talks to hardware or is a certified safety function.
//! See `README.md` for layers, default limits and the physical-test protocol.

pub mod bridge;
pub mod contract;
pub mod controller;
pub mod hil;

/// This crate never drives physical hardware; see [`hil`].
pub const SIMULATION_ONLY: bool = true;
