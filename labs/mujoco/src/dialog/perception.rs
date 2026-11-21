//! Linguistic perception: oracle verdicts and human answers become observations with
//! provenance, and update a core [`BinaryBelief`] under an explicit calibrated likelihood.
//!
//! Reuses `gogamen_core::{BinaryBelief, LikelihoodModel, EvidenceDeduplicator,
//! ObservationEnvelope, ProvenanceMetadata}`. Speaker reliability and human reliability are the
//! sensitivity/specificity of the likelihood models; the envelope `Confidence` is *not* used as a
//! likelihood (core rule).

use gogamen_core::{
    BeliefError, BeliefUpdateStatus, BinaryBelief, Confidence, EvidenceDeduplicator, EvidenceRef,
    FrameId, LikelihoodModel, MessageId, MonotonicInstant, ObservationEnvelope, ObservationKind,
    ProvenanceMetadata, SensorId, UnitTag,
};

use super::oracle::TICK_NANOS;

/// Who produced the evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// An utterance of the speaker, as interpreted by the oracle.
    Speaker,
    /// The reply to an `AskHuman` question.
    Human,
}

/// What one evidence item did to the belief.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateRecord {
    /// Evidence source.
    pub source: Source,
    /// Tick at which it was applied.
    pub tick: u64,
    /// The claim: target is the left door.
    pub says_left: bool,
    /// `P(left)` before.
    pub p_before: f64,
    /// `P(left)` after.
    pub p_after: f64,
    /// Core update status.
    pub status: BeliefUpdateStatus,
}

/// Belief about "the target is the left door" plus the provenance bookkeeping.
#[derive(Debug, Clone)]
pub struct Perception {
    belief: BinaryBelief,
    speaker_model: LikelihoodModel,
    human_model: LikelihoodModel,
    dedup: EvidenceDeduplicator,
    watermark: Option<MonotonicInstant>,
    next_id: u64,
    history: Vec<UpdateRecord>,
    claims_left: u32,
    claims_right: u32,
}

impl Perception {
    /// Perception with prior `P(left)`, speaker reliability and human reliability (each used as
    /// both sensitivity and specificity).
    pub fn new(
        prior_p_left: f64,
        speaker_reliability: f64,
        human_reliability: f64,
    ) -> Result<Self, BeliefError> {
        Ok(Self {
            belief: BinaryBelief::new(prior_p_left)?,
            speaker_model: LikelihoodModel::from_sensitivity_specificity(
                speaker_reliability,
                speaker_reliability,
            )?,
            human_model: LikelihoodModel::from_sensitivity_specificity(
                human_reliability,
                human_reliability,
            )?,
            dedup: EvidenceDeduplicator::new(),
            watermark: None,
            next_id: 1,
            history: Vec::new(),
            claims_left: 0,
            claims_right: 0,
        })
    }

    /// Current belief.
    pub fn belief(&self) -> &BinaryBelief {
        &self.belief
    }

    /// `P(left)`.
    pub fn p_left(&self) -> f64 {
        self.belief.p_true()
    }

    /// Confidence in the MAP door: `max(P(left), P(right))`.
    pub fn confidence(&self) -> f64 {
        self.belief.p_true().max(self.belief.p_false())
    }

    /// Update history.
    pub fn history(&self) -> &[UpdateRecord] {
        &self.history
    }

    /// True if the speaker claimed both doors (self-contradiction).
    pub fn speaker_contradicted(&self) -> bool {
        self.claims_left > 0 && self.claims_right > 0
    }

    /// Ingest one piece of evidence at `tick`.
    pub fn ingest(
        &mut self,
        source: Source,
        says_left: bool,
        tick: u64,
    ) -> Result<UpdateRecord, BeliefError> {
        let id = self.next_id;
        self.next_id += 1;
        let at = MonotonicInstant::from_nanos(tick as u128 * TICK_NANOS);
        let (sensor, model) = match source {
            Source::Speaker => (1, &self.speaker_model),
            Source::Human => (2, &self.human_model),
        };
        let meta = ProvenanceMetadata {
            message_id: MessageId::new(id),
            observed_at: at,
            received_at: at,
            sensor_id: SensorId::new(sensor),
            frame_id: FrameId::new("dialog").expect("static frame id"),
            units: UnitTag::dimensionless(),
            confidence: Confidence::new(0.5).expect("static confidence"),
            revision: id,
            evidence_ref: EvidenceRef::new(id),
            claim_kind: ObservationKind::Assertion,
        };
        let env = ObservationEnvelope::new(meta, says_left);
        let p_before = self.p_left();
        let status = self.belief.update_with_observation(
            &env,
            model,
            &mut self.dedup,
            &mut self.watermark,
        )?;
        if source == Source::Speaker {
            if says_left {
                self.claims_left += 1;
            } else {
                self.claims_right += 1;
            }
        }
        let rec = UpdateRecord {
            source,
            tick,
            says_left,
            p_before,
            p_after: self.p_left(),
            status,
        };
        self.history.push(rec);
        Ok(rec)
    }
}
