//! Deterministic mock providers (fixtures). No network, no models.

use std::collections::BTreeMap;
use std::time::Duration;

use gogamen_core::{
    Confidence, EvidenceRef, FrameId, MessageId, MonotonicInstant, ObservationEnvelope,
    ObservationKind, ProvenanceMetadata, SensorId, UnitTag,
};

use crate::candidate::{GenerativeCandidate, VisualDetection};
use crate::capability::{
    Availability, DeclaredCost, GenerationOutput, GenerativeProposalPort, ProposalRequest,
    ProviderError, VisualFrame, VisualOutput, VisualPerceptionPort,
};

/// Sensor id used by [`wrap_generated`] for generative origin.
pub const MOCK_GENERATIVE_SENSOR: SensorId = SensorId::new(9_000);

/// Wraps a candidate in an envelope labelled as generative (`Proposal`) origin.
pub fn wrap_generated(
    seq: u64,
    candidate: GenerativeCandidate,
    kind: ObservationKind,
    sensor: SensorId,
) -> ObservationEnvelope<GenerativeCandidate> {
    let meta = ProvenanceMetadata {
        message_id: MessageId::new(seq),
        observed_at: MonotonicInstant::from_nanos(u128::from(seq)),
        received_at: MonotonicInstant::from_nanos(u128::from(seq)),
        sensor_id: sensor,
        frame_id: FrameId::world(),
        units: UnitTag::dimensionless(),
        confidence: Confidence::new(0.5).unwrap_or(Confidence::ZERO),
        revision: 0,
        // Generator-owned evidence namespace, deliberately disjoint from real evidence.
        evidence_ref: EvidenceRef::new(1_000_000 + seq),
        claim_kind: kind,
    };
    ObservationEnvelope::new(meta, candidate)
}

/// Fixture-driven generative provider: returns the scripted candidates per request key.
#[derive(Debug, Clone, Default)]
pub struct MockGenerative {
    scripts: BTreeMap<String, Vec<GenerativeCandidate>>,
    kind: Option<ObservationKind>,
    sensor: Option<SensorId>,
    unavailable: Option<&'static str>,
    latency: Duration,
    units: u64,
}

impl MockGenerative {
    /// New empty mock with declared latency/units per call.
    pub fn new(latency: Duration, units: u64) -> Self {
        Self {
            latency,
            units,
            ..Self::default()
        }
    }

    /// Scripts the answer for `key`.
    pub fn script(mut self, key: impl Into<String>, candidates: Vec<GenerativeCandidate>) -> Self {
        self.scripts.insert(key.into(), candidates);
        self
    }

    /// Overrides the claim kind of produced envelopes (to simulate mislabelled output).
    pub fn with_kind(mut self, kind: ObservationKind) -> Self {
        self.kind = Some(kind);
        self
    }

    /// Overrides the sensor id of produced envelopes (to simulate unlabelled origin).
    pub fn with_sensor(mut self, sensor: SensorId) -> Self {
        self.sensor = Some(sensor);
        self
    }

    /// Makes the provider report itself unavailable.
    pub fn unavailable(mut self, reason: &'static str) -> Self {
        self.unavailable = Some(reason);
        self
    }
}

impl GenerativeProposalPort for MockGenerative {
    fn availability(&self) -> Availability {
        self.unavailable
            .map_or(Availability::Available, Availability::Unavailable)
    }

    fn propose(&self, request: &ProposalRequest) -> Result<GenerationOutput, ProviderError> {
        let list = self
            .scripts
            .get(&request.key)
            .ok_or_else(|| ProviderError(format!("no fixture for key `{}`", request.key)))?;
        let kind = self.kind.unwrap_or(ObservationKind::Proposal);
        let sensor = self.sensor.unwrap_or(MOCK_GENERATIVE_SENSOR);
        let candidates = list
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, c)| wrap_generated(i as u64, c, kind, sensor))
            .collect();
        Ok(GenerationOutput {
            candidates,
            cost: DeclaredCost {
                latency: self.latency,
                units: self.units,
            },
        })
    }
}

/// Fixture-driven visual provider: returns the same detections for every frame.
#[derive(Debug, Clone, Default)]
pub struct MockVisual {
    detections: Vec<VisualDetection>,
    unavailable: Option<&'static str>,
}

impl MockVisual {
    /// Mock returning `detections` for any frame.
    pub fn new(detections: Vec<VisualDetection>) -> Self {
        Self {
            detections,
            unavailable: None,
        }
    }

    /// Makes the provider report itself unavailable.
    pub fn unavailable(mut self, reason: &'static str) -> Self {
        self.unavailable = Some(reason);
        self
    }
}

impl VisualPerceptionPort for MockVisual {
    fn availability(&self) -> Availability {
        self.unavailable
            .map_or(Availability::Available, Availability::Unavailable)
    }

    fn perceive(&self, _frame: &VisualFrame) -> Result<VisualOutput, ProviderError> {
        Ok(VisualOutput {
            detections: self.detections.clone(),
            cost: DeclaredCost::default(),
        })
    }
}
