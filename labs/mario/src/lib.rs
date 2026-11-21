//! GOG-62 Mario lab.
//!
//! * [`EmulatorPort`]: typed boundary for an emulator (reset, step with pad
//!   input, typed features, RAM view, optional snapshot/restore).
//! * [`rom`]: user-ROM hashing (SHA-256) and the configurable known-hash list.
//! * [`fixture`]: `SyntheticPlatformer`, a deterministic side-scroller written
//!   for CI. It is a synthetic fixture, NOT Mario, and uses no third-party asset.
//! * [`policy`] / [`metrics`]: baselines, a snapshot Monte Carlo planner and
//!   reproducible episode metrics (progress, deaths, actions, cost, frames/s).
//! * [`command`]: the `mario_run` command logic (testable, prints NOT_RUN).
//!
//! No ROM or BIOS is distributed; no emulator backend is compiled in.

pub mod coach;
pub mod command;
pub mod fixture;
pub mod metrics;
pub mod policy;
pub mod port;
pub mod rom;

pub use fixture::{SYNTHETIC_LABEL, SyntheticPlatformer};
pub use metrics::{EpisodeMetrics, RunConfig, run_episode};
pub use policy::{AlwaysRight, MonteCarloPlanner, Policy, RandomPolicy};
pub use port::{EmulatorPort, MarioFeatures, PadState, PortError, StepReport};
pub use rom::{KnownRoms, NotRunReason, RomError, RomVerified, sha256_hex};
