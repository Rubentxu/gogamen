//! Simulated robot with a human collaborator (GOG-66): linguistic perception, `AskHuman`, and an
//! optional Jev-like language oracle that has no authority over the control loop.
//!
//! Layering (enforced by a textual test in `tests/dialog.rs`):
//!
//! * [`oracle`], [`perception`], [`policy`], [`cognition`], [`fixtures`]: *cognition side*. They
//!   know nothing about the simulator, the action guard or the safety stop.
//! * [`episode`]: the only place that owns both the control loop (`nav::Sim`) and the cognition
//!   state machine. The control loop advances exactly one tick per iteration; cognition only
//!   emits a goal [`cognition::Command`] when it has one. The goal still goes through the
//!   core `ActionGuard` and the independent safety stop inside `Sim`.
//! * [`eval`]: calibration metrics and the policy comparison.

pub mod cognition;
pub mod episode;
pub mod eval;
pub mod fixtures;
pub mod oracle;
pub mod perception;
pub mod policy;

pub use cognition::{Cognition, Command, Event, EventKind};
pub use episode::{DIALOG_HOLD_SETTLE_TICKS, DialogEpisode, EpisodeResult, EpisodeVerdict};
pub use eval::{PolicyRow, brier_score, expected_calibration_error, run_policy};
pub use fixtures::{Claim, DialogFixture, FixtureKind, HumanReply, annotated_fixtures};
pub use oracle::{
    LiveJevUnavailable, OracleMode, OracleOutcome, TICK_NANOS, build_oracle, consult,
};
pub use perception::{Perception, Source, UpdateRecord};
pub use policy::{DialogPolicy, Door, PolicyKind, PreAsk};
