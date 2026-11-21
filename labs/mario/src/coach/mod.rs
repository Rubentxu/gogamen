//! GOG-63 Mario Coach: asynchronous human-like advice that can only change the
//! planner's hypothesis or an authorised preference, never the pad.
//!
//! * [`advice`]: untrusted text -> catalog -> `CoachAdvice`, virtual-clock queue.
//! * [`planner`]: `CoachedPlanner`, a Monte Carlo planner whose hypothesis and
//!   preference are modified by delivered advice; its pad passes a `PadGuard`.
//! * [`mock`]: deterministic coach mocks (truthful, late, lying, mute).
//! * [`eval`]: with/without coach comparison on the synthetic fixture.

pub mod advice;
pub mod eval;
pub mod mock;
pub mod planner;

pub use advice::{
    AdviceCatalog, AdviceError, AdviceKind, AdviceQueue, AdviceRecord, CoachAdvice,
    CoachObservation, CoachProvider, Emission, Outcome, RawAdvice, Timing,
};
pub use mock::{MockCoach, MockMode};
pub use planner::{CoachReport, CoachedPlanner, Hypothesis, PadGuard};
