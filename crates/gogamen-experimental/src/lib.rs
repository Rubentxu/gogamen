//! Experimental, opt-in capabilities (GOG-82): visual/Box perception and generative "System Two".
//!
//! Design rules:
//!
//! * Nothing here is reachable from `gogamen-core`, `gogamen-cli` or the labs; they do not depend
//!   on this crate and behave identically whether or not it exists.
//! * Every capability is **off by default** ([`ExperimentalConfig::default`]). A disabled, absent
//!   or unavailable capability yields [`Outcome::Abstain`] with a typed [`AbstainReason`].
//! * Generated output is untrusted. It only becomes usable after [`CandidateValidator`] checks the
//!   capability catalog, the core `ActionGuard` and provenance/grounding. A generated action can
//!   only be obtained as a [`GuardedAction`], whose sole constructor sits behind the guard.
//! * Providers here are deterministic mocks/fixtures: no network, no real models.

pub mod candidate;
pub mod capability;
pub mod mock;

#[cfg(feature = "lab-baseline")]
pub mod compare;

pub use candidate::{
    CandidateBody, CandidateRejection, CandidateValidator, Claim, DetectionRejection,
    GenerativeCandidate, GuardedAction, ValidatedCandidate, ValidationContext, VisualDetection,
};
pub use capability::{
    AbstainReason, Availability, CapabilityReport, CapabilityStatus, DeclaredCost,
    ExperimentalCapability, ExperimentalConfig, ExperimentalHub, GenerationOutput,
    GenerativeProposalPort, Outcome, ProposalRequest, ProviderError, VisualFrame, VisualOutput,
    VisualPerceptionPort,
};
