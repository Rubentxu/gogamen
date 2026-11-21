//! Generated candidates and their validation (catalog + `ActionGuard` + provenance/grounding).

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, CapabilityCatalog, Confidence, EvidenceRef,
    GuardContext, GuardDecision, GuardRejectionReason, ModuleContractError, ObservationEnvelope,
    ObservationKind, OperationBinding, SensorId, TemporalProfile, Terminal, Tick,
};

/// One factual statement made by a generated candidate, with the evidence it cites.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    /// Statement text.
    pub statement: String,
    /// Evidence references. Must all resolve to non-generative evidence.
    pub evidence: Vec<EvidenceRef>,
}

/// What a generated candidate proposes.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateBody {
    /// Execute a catalog operation with a concrete action (still needs the guard).
    Action {
        /// Operation binding looked up in the capability catalog.
        binding: OperationBinding,
        /// Concrete action for the guard.
        action: Action,
    },
    /// A notice/advice for a human; asserts facts, so it needs grounded claims.
    Notice {
        /// Notice text.
        text: String,
    },
    /// A proposed goal name; advisory only (a goal manager must still authorize it).
    Goal {
        /// Goal name, must be in the known goal set.
        name: String,
    },
}

/// A candidate produced by a generative provider. Untrusted.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerativeCandidate {
    /// Candidate id (for reporting).
    pub id: String,
    /// Proposal.
    pub body: CandidateBody,
    /// Claims backing the proposal.
    pub claims: Vec<Claim>,
}

/// One raw visual detection. Untrusted.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualDetection {
    /// Detected label.
    pub label: String,
    /// Reported confidence.
    pub confidence: Confidence,
    /// Frame evidence this detection claims to come from.
    pub evidence: EvidenceRef,
}

/// Typed reason a candidate was rejected.
#[derive(Debug, Clone, PartialEq)]
pub enum CandidateRejection {
    /// Envelope sensor id is not a registered generative origin (origin must be labelled).
    UnlabelledOrigin {
        /// Offending sensor.
        sensor: SensorId,
    },
    /// Envelope claim kind is not `Proposal` (generated output must never pose as measurement).
    WrongClaimKind {
        /// Kind found.
        found: ObservationKind,
    },
    /// A claim has no evidence, or cites evidence that does not resolve (including the
    /// generator's own evidence ref).
    UngroundedClaim {
        /// Index of the claim.
        claim_index: usize,
    },
    /// Notice without any claim (an assertion without evidence).
    NoticeWithoutClaims,
    /// Notice text empty.
    EmptyNotice,
    /// Operation unknown/incompatible/not permitted according to the catalog.
    OutOfCatalog(ModuleContractError),
    /// Goal name not in the known goal set.
    UnknownGoal {
        /// Goal name.
        name: String,
    },
    /// The core `ActionGuard` did not return `Allow` (rejection, fallback or no-op).
    GuardDenied {
        /// Guard rejection reason when available (`None` for fallback/no-op).
        reason: Option<GuardRejectionReason>,
    },
}

impl fmt::Display for CandidateRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnlabelledOrigin { sensor } => {
                write!(f, "sensor {sensor:?} is not a labelled generative origin")
            }
            Self::WrongClaimKind { found } => {
                write!(f, "generated output must be a proposal, found {found}")
            }
            Self::UngroundedClaim { claim_index } => {
                write!(f, "claim #{claim_index} has no resolvable evidence")
            }
            Self::NoticeWithoutClaims => f.write_str("notice carries no grounded claim"),
            Self::EmptyNotice => f.write_str("notice text is empty"),
            Self::OutOfCatalog(e) => write!(f, "operation outside catalog contract: {e:?}"),
            Self::UnknownGoal { name } => write!(f, "unknown goal `{name}`"),
            Self::GuardDenied { reason } => write!(f, "action guard denied: {reason:?}"),
        }
    }
}

impl Error for CandidateRejection {}

/// Typed reason a visual detection was rejected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionRejection {
    /// Label not in the known label catalog.
    UnknownLabel(String),
    /// Evidence does not resolve to a known frame.
    UngroundedDetection,
}

/// An action that went through `ActionGuard` and was explicitly allowed.
///
/// Fields are private and there is no public constructor: the only way to obtain one is
/// [`CandidateValidator::validate`]. This is how "no executable action bypasses the guard" is
/// enforced at the type level.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardedAction {
    action: Action,
    binding: OperationBinding,
}

impl GuardedAction {
    /// The guard-approved action.
    pub const fn action(&self) -> &Action {
        &self.action
    }

    /// The catalog binding it was validated under.
    pub const fn binding(&self) -> &OperationBinding {
        &self.binding
    }
}

/// A candidate that passed every check.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidatedCandidate {
    /// Guard-approved action.
    Action(GuardedAction),
    /// Grounded notice text (advisory).
    Notice(String),
    /// Known goal name (advisory; needs separate authorization).
    Goal(String),
}

/// Runtime context for the guard.
#[derive(Debug, Clone)]
pub struct ValidationContext<'a> {
    /// Action space of the live environment.
    pub action_space: &'a ActionSpace,
    /// Current tick.
    pub tick: Tick,
    /// Terminal state.
    pub terminal: Terminal,
}

/// Validates generated candidates. Holds only references to trusted structures.
pub struct CandidateValidator<'a> {
    catalog: &'a CapabilityCatalog,
    guard: &'a ActionGuard,
    generative_sensors: BTreeSet<SensorId>,
    known_evidence: BTreeSet<EvidenceRef>,
    known_goals: BTreeSet<String>,
    known_labels: BTreeSet<String>,
}

impl<'a> CandidateValidator<'a> {
    /// New validator with empty registries (everything is rejected until populated).
    pub fn new(catalog: &'a CapabilityCatalog, guard: &'a ActionGuard) -> Self {
        Self {
            catalog,
            guard,
            generative_sensors: BTreeSet::new(),
            known_evidence: BTreeSet::new(),
            known_goals: BTreeSet::new(),
            known_labels: BTreeSet::new(),
        }
    }

    /// Registers a sensor id as a labelled generative origin.
    pub fn with_generative_sensor(mut self, sensor: SensorId) -> Self {
        self.generative_sensors.insert(sensor);
        self
    }

    /// Registers non-generative evidence (real observations) that claims may cite.
    pub fn with_evidence(mut self, evidence: impl IntoIterator<Item = EvidenceRef>) -> Self {
        self.known_evidence.extend(evidence);
        self
    }

    /// Registers goal names that may be proposed.
    pub fn with_goals<S: Into<String>>(mut self, goals: impl IntoIterator<Item = S>) -> Self {
        self.known_goals.extend(goals.into_iter().map(Into::into));
        self
    }

    /// Registers visual labels that detections may use.
    pub fn with_labels<S: Into<String>>(mut self, labels: impl IntoIterator<Item = S>) -> Self {
        self.known_labels.extend(labels.into_iter().map(Into::into));
        self
    }

    /// Validates one visual detection (label catalog + evidence).
    pub fn validate_detection(&self, d: &VisualDetection) -> Result<(), DetectionRejection> {
        if !self.known_labels.contains(&d.label) {
            return Err(DetectionRejection::UnknownLabel(d.label.clone()));
        }
        if !self.known_evidence.contains(&d.evidence) {
            return Err(DetectionRejection::UngroundedDetection);
        }
        Ok(())
    }

    /// Validates a generated candidate: provenance label, grounding, catalog, then guard.
    ///
    /// Operations are resolved under `TemporalProfile::BestEffort`, i.e. generated candidates are
    /// treated as an advisory path and never as a bounded control path.
    pub fn validate(
        &self,
        envelope: &ObservationEnvelope<GenerativeCandidate>,
        ctx: &ValidationContext<'_>,
    ) -> Result<ValidatedCandidate, CandidateRejection> {
        if !self.generative_sensors.contains(&envelope.sensor_id()) {
            return Err(CandidateRejection::UnlabelledOrigin {
                sensor: envelope.sensor_id(),
            });
        }
        if envelope.claim_kind() != ObservationKind::Proposal {
            return Err(CandidateRejection::WrongClaimKind {
                found: envelope.claim_kind(),
            });
        }
        let cand = envelope.payload();
        for (claim_index, claim) in cand.claims.iter().enumerate() {
            if claim.evidence.is_empty()
                || !claim
                    .evidence
                    .iter()
                    .all(|e| self.known_evidence.contains(e))
            {
                return Err(CandidateRejection::UngroundedClaim { claim_index });
            }
        }
        match &cand.body {
            CandidateBody::Notice { text } => {
                if text.trim().is_empty() {
                    return Err(CandidateRejection::EmptyNotice);
                }
                if cand.claims.is_empty() {
                    return Err(CandidateRejection::NoticeWithoutClaims);
                }
                Ok(ValidatedCandidate::Notice(text.clone()))
            }
            CandidateBody::Goal { name } => {
                if !self.known_goals.contains(name) {
                    return Err(CandidateRejection::UnknownGoal { name: name.clone() });
                }
                Ok(ValidatedCandidate::Goal(name.clone()))
            }
            CandidateBody::Action { binding, action } => {
                self.catalog
                    .validate_operation(binding, TemporalProfile::BestEffort)
                    .map_err(CandidateRejection::OutOfCatalog)?;
                let proposal = ActionProposal::authorized(action.clone(), ctx.tick);
                let gctx = GuardContext::now(ctx.action_space, ctx.tick, ctx.terminal);
                match self.guard.evaluate(&proposal, &gctx) {
                    GuardDecision::Allow(action) => Ok(ValidatedCandidate::Action(GuardedAction {
                        action,
                        binding: binding.clone(),
                    })),
                    GuardDecision::Reject(reason) => Err(CandidateRejection::GuardDenied {
                        reason: Some(reason),
                    }),
                    // A fallback/no-op replaces the candidate: the candidate itself is not usable.
                    GuardDecision::Fallback(_) | GuardDecision::NoOp => {
                        Err(CandidateRejection::GuardDenied { reason: None })
                    }
                }
            }
        }
    }
}
