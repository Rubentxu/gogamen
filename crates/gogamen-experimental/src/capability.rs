//! Opt-in ports, configuration, capability discovery and the abstaining [`ExperimentalHub`].

use std::error::Error;
use std::fmt;
use std::time::Duration;

use gogamen_core::ObservationEnvelope;

use crate::candidate::{GenerativeCandidate, VisualDetection};

/// Experimental capabilities that can be discovered and enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExperimentalCapability {
    /// Visual perception plugin ("Box").
    VisualBox,
    /// Generative deliberation output ("System Two").
    GenerativeSystemTwo,
}

impl fmt::Display for ExperimentalCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::VisualBox => "visual-box",
            Self::GenerativeSystemTwo => "generative-system-two",
        })
    }
}

/// Explicit opt-in switches. `Default` enables nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExperimentalConfig {
    visual_box: bool,
    generative_system_two: bool,
}

impl ExperimentalConfig {
    /// Everything disabled (same as `default()`).
    pub const fn disabled() -> Self {
        Self {
            visual_box: false,
            generative_system_two: false,
        }
    }

    /// Enables `capability`.
    pub const fn enable(mut self, capability: ExperimentalCapability) -> Self {
        match capability {
            ExperimentalCapability::VisualBox => self.visual_box = true,
            ExperimentalCapability::GenerativeSystemTwo => self.generative_system_two = true,
        }
        self
    }

    /// Whether `capability` was explicitly enabled.
    pub const fn is_enabled(&self, capability: ExperimentalCapability) -> bool {
        match capability {
            ExperimentalCapability::VisualBox => self.visual_box,
            ExperimentalCapability::GenerativeSystemTwo => self.generative_system_two,
        }
    }
}

/// Provider-reported availability (for "use if available" discovery).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Ready to serve requests.
    Available,
    /// Not usable now, with a static reason (missing model, credentials, ...).
    Unavailable(&'static str),
}

/// Typed reason for abstaining.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstainReason {
    /// The capability was not enabled in [`ExperimentalConfig`] (default).
    DisabledByConfig(ExperimentalCapability),
    /// Enabled, but no provider was registered.
    NoProvider(ExperimentalCapability),
    /// Enabled and registered, but the provider reports it is unavailable.
    ProviderUnavailable {
        /// Capability involved.
        capability: ExperimentalCapability,
        /// Provider-supplied reason.
        reason: &'static str,
    },
    /// The provider failed while serving the request.
    ProviderFailed {
        /// Capability involved.
        capability: ExperimentalCapability,
        /// Provider error.
        error: ProviderError,
    },
}

impl fmt::Display for AbstainReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DisabledByConfig(c) => write!(f, "{c} disabled by config"),
            Self::NoProvider(c) => write!(f, "{c} enabled but no provider registered"),
            Self::ProviderUnavailable { capability, reason } => {
                write!(f, "{capability} provider unavailable: {reason}")
            }
            Self::ProviderFailed { capability, error } => {
                write!(f, "{capability} provider failed: {error}")
            }
        }
    }
}

impl Error for AbstainReason {}

/// Error returned by a provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderError(pub String);

impl fmt::Display for ProviderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for ProviderError {}

/// Either a value or a typed abstention. Never panics, never fabricates a value.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome<T> {
    /// The capability produced a value (still untrusted until validated).
    Value(T),
    /// The capability abstained.
    Abstain(AbstainReason),
}

impl<T> Outcome<T> {
    /// `true` for [`Outcome::Abstain`].
    pub const fn is_abstain(&self) -> bool {
        matches!(self, Self::Abstain(_))
    }

    /// Abstention reason, if any.
    pub const fn abstain_reason(&self) -> Option<&AbstainReason> {
        match self {
            Self::Abstain(r) => Some(r),
            Self::Value(_) => None,
        }
    }
}

/// Declared (not measured) cost of one provider call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeclaredCost {
    /// Simulated latency the provider declares for this call.
    pub latency: Duration,
    /// Abstract token/work units the provider declares.
    pub units: u64,
}

/// Input frame for the visual port (opaque bytes; no decoding here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VisualFrame {
    /// Stable frame identifier.
    pub frame_id: String,
    /// Raw payload.
    pub bytes: Vec<u8>,
}

/// Visual port output: detections plus declared cost. Detections are untrusted.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualOutput {
    /// Raw detections.
    pub detections: Vec<VisualDetection>,
    /// Declared cost.
    pub cost: DeclaredCost,
}

/// Request to the generative port: free-text prompt plus a stable key for fixtures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProposalRequest {
    /// Stable key (fixtures are keyed by it).
    pub key: String,
    /// Prompt text.
    pub prompt: String,
}

/// Generative port output: candidates wrapped in provenance envelopes plus declared cost.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationOutput {
    /// Candidates, each in an [`ObservationEnvelope`]. Untrusted until validated.
    pub candidates: Vec<ObservationEnvelope<GenerativeCandidate>>,
    /// Declared cost.
    pub cost: DeclaredCost,
}

/// Visual/Box perception port.
pub trait VisualPerceptionPort {
    /// Whether the provider can serve requests now.
    fn availability(&self) -> Availability;
    /// Produces raw detections for `frame`.
    fn perceive(&self, frame: &VisualFrame) -> Result<VisualOutput, ProviderError>;
}

/// Generative System Two port (deliberative proposals, outside the critical loop).
pub trait GenerativeProposalPort {
    /// Whether the provider can serve requests now.
    fn availability(&self) -> Availability;
    /// Produces candidate actions/notices/goals for `request`.
    fn propose(&self, request: &ProposalRequest) -> Result<GenerationOutput, ProviderError>;
}

/// Discovery status of one capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityStatus {
    /// Capability.
    pub capability: ExperimentalCapability,
    /// Enabled in config.
    pub enabled: bool,
    /// A provider was registered.
    pub provider_registered: bool,
    /// Final verdict: `Available` only if enabled, registered and provider-available.
    pub availability: Availability,
}

/// Snapshot of all capability statuses, in stable order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityReport {
    /// One entry per [`ExperimentalCapability`].
    pub statuses: Vec<CapabilityStatus>,
}

impl CapabilityReport {
    /// Whether `capability` can be used right now.
    pub fn is_available(&self, capability: ExperimentalCapability) -> bool {
        self.statuses
            .iter()
            .any(|s| s.capability == capability && s.availability == Availability::Available)
    }
}

/// Holds config and optional providers; abstains unless everything lines up.
#[derive(Default)]
pub struct ExperimentalHub {
    config: ExperimentalConfig,
    visual: Option<Box<dyn VisualPerceptionPort>>,
    generative: Option<Box<dyn GenerativeProposalPort>>,
}

impl fmt::Debug for ExperimentalHub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExperimentalHub")
            .field("config", &self.config)
            .field("visual", &self.visual.is_some())
            .field("generative", &self.generative.is_some())
            .finish()
    }
}

impl ExperimentalHub {
    /// Hub with the given config and no providers.
    pub fn new(config: ExperimentalConfig) -> Self {
        Self {
            config,
            visual: None,
            generative: None,
        }
    }

    /// Registers a visual provider (does not enable it).
    pub fn with_visual(mut self, provider: Box<dyn VisualPerceptionPort>) -> Self {
        self.visual = Some(provider);
        self
    }

    /// Registers a generative provider (does not enable it).
    pub fn with_generative(mut self, provider: Box<dyn GenerativeProposalPort>) -> Self {
        self.generative = Some(provider);
        self
    }

    fn gate(
        &self,
        capability: ExperimentalCapability,
        registered: bool,
        availability: impl FnOnce() -> Availability,
    ) -> Result<(), AbstainReason> {
        if !self.config.is_enabled(capability) {
            return Err(AbstainReason::DisabledByConfig(capability));
        }
        if !registered {
            return Err(AbstainReason::NoProvider(capability));
        }
        match availability() {
            Availability::Available => Ok(()),
            Availability::Unavailable(reason) => {
                Err(AbstainReason::ProviderUnavailable { capability, reason })
            }
        }
    }

    /// Capability discovery. A disabled capability never calls its provider.
    pub fn discover(&self) -> CapabilityReport {
        let mk = |capability, registered: bool, avail: &dyn Fn() -> Availability| {
            let enabled = self.config.is_enabled(capability);
            let availability = match self.gate(capability, registered, avail) {
                Ok(()) => Availability::Available,
                Err(AbstainReason::DisabledByConfig(_)) => {
                    Availability::Unavailable("disabled by config")
                }
                Err(AbstainReason::NoProvider(_)) => Availability::Unavailable("no provider"),
                Err(AbstainReason::ProviderUnavailable { reason, .. }) => {
                    Availability::Unavailable(reason)
                }
                Err(AbstainReason::ProviderFailed { .. }) => {
                    Availability::Unavailable("provider failed")
                }
            };
            CapabilityStatus {
                capability,
                enabled,
                provider_registered: registered,
                availability,
            }
        };
        let visual = &self.visual;
        let generative = &self.generative;
        CapabilityReport {
            statuses: vec![
                mk(ExperimentalCapability::VisualBox, visual.is_some(), &|| {
                    visual
                        .as_ref()
                        .map_or(Availability::Unavailable("no provider"), |p| {
                            p.availability()
                        })
                }),
                mk(
                    ExperimentalCapability::GenerativeSystemTwo,
                    generative.is_some(),
                    &|| {
                        generative
                            .as_ref()
                            .map_or(Availability::Unavailable("no provider"), |p| {
                                p.availability()
                            })
                    },
                ),
            ],
        }
    }

    /// Raw visual detections, or a typed abstention.
    pub fn perceive(&self, frame: &VisualFrame) -> Outcome<VisualOutput> {
        let cap = ExperimentalCapability::VisualBox;
        let provider = self.visual.as_ref();
        if let Err(reason) = self.gate(cap, provider.is_some(), || {
            provider.map_or(Availability::Unavailable("no provider"), |p| {
                p.availability()
            })
        }) {
            return Outcome::Abstain(reason);
        }
        match provider.map(|p| p.perceive(frame)) {
            Some(Ok(out)) => Outcome::Value(out),
            Some(Err(error)) => Outcome::Abstain(AbstainReason::ProviderFailed {
                capability: cap,
                error,
            }),
            None => Outcome::Abstain(AbstainReason::NoProvider(cap)),
        }
    }

    /// Raw generative candidates, or a typed abstention.
    pub fn propose(&self, request: &ProposalRequest) -> Outcome<GenerationOutput> {
        let cap = ExperimentalCapability::GenerativeSystemTwo;
        let provider = self.generative.as_ref();
        if let Err(reason) = self.gate(cap, provider.is_some(), || {
            provider.map_or(Availability::Unavailable("no provider"), |p| {
                p.availability()
            })
        }) {
            return Outcome::Abstain(reason);
        }
        match provider.map(|p| p.propose(request)) {
            Some(Ok(out)) => Outcome::Value(out),
            Some(Err(error)) => Outcome::Abstain(AbstainReason::ProviderFailed {
                capability: cap,
                error,
            }),
            None => Outcome::Abstain(AbstainReason::NoProvider(cap)),
        }
    }
}
