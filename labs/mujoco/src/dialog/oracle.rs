//! Language oracle port (a "Jev"-like System One) for the dialog layer.
//!
//! The port is the existing core [`SystemOneProvider`] (GOG-35): a typed query goes in and a
//! typed, validated response comes out. This module deliberately has no access to the simulator,
//! the action guard or the safety stop: the only thing an oracle can produce is an
//! [`OracleOutcome`], i.e. data that cognition may or may not turn into evidence.
//!
//! * Default mode is [`OracleMode::Replay`]: recorded verdicts with a simulated latency.
//! * [`OracleMode::Live`] is opt-in and, since NOE-029 (a real Jev client) does not exist in this
//!   repository, resolves to [`LiveJevUnavailable`], which always reports "not available".

use std::time::Duration;

use gogamen_core::{
    Confidence, EvidenceRef, FixtureBehavior, FrameId, HypothesisId, MessageId, MonotonicClock,
    MonotonicDeadline, MonotonicInstant, NoulVerdict, ObservationKind, OptionId,
    ProvenanceMetadata, QueryId, ReplaySystemOneProvider, ResponseStatus, SensorId, SystemOneError,
    SystemOneFixture, SystemOneProvider, SystemOneResponse, TypedQuery, UnitTag,
    VirtualMonotonicClock,
};

use super::fixtures::Claim;

/// Virtual-time length of one control tick (`dt = 0.01 s`).
pub const TICK_NANOS: u128 = 10_000_000;

/// Hypothesis every utterance is interpreted against: "the target is the left door".
pub const HYPOTHESIS: &str = "target_is_left_door";

/// How the oracle behaves in an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OracleMode {
    /// Recorded verdicts with their recorded latency (default; fully offline).
    #[default]
    Replay,
    /// Recorded verdicts but every response takes `latency_ticks` (may exceed the timeout).
    ReplaySlow {
        /// Simulated response latency in control ticks.
        latency_ticks: u64,
    },
    /// The oracle never answers.
    Never,
    /// Live Jev. Opt-in; NOE-029 is not available, so this is a stub that reports unavailable.
    Live,
}

/// Live Jev stand-in: always "not available". No network, credentials or provider API involved.
#[derive(Debug, Clone, Copy, Default)]
pub struct LiveJevUnavailable;

impl SystemOneProvider for LiveJevUnavailable {
    fn provider_id(&self) -> &str {
        "jev-live-unavailable"
    }

    fn query(
        &self,
        _query: &TypedQuery,
        _clock: &dyn MonotonicClock,
    ) -> Result<SystemOneResponse, SystemOneError> {
        Err(SystemOneError::InvalidResponse {
            reason: "live Jev (NOE-029) is not available in this build; replay only",
        })
    }
}

/// What a consultation yielded, and at which control tick the robot learns it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleOutcome {
    /// Answered `True`/`False` for the hypothesis.
    Answered {
        /// Verdict (never [`NoulVerdict::Unknown`]).
        verdict: NoulVerdict,
        /// Control tick at which the answer arrives.
        at_tick: u64,
    },
    /// The oracle explicitly said it does not know.
    Unknown {
        /// Control tick at which the answer arrives.
        at_tick: u64,
    },
    /// No answer before the deadline; cognition learns this at the deadline tick.
    TimedOut {
        /// Deadline tick.
        at_tick: u64,
    },
    /// The provider is unavailable (error); known immediately.
    Unavailable {
        /// Control tick of the failed call.
        at_tick: u64,
    },
}

impl OracleOutcome {
    /// Tick at which cognition learns this outcome.
    pub const fn at_tick(&self) -> u64 {
        match self {
            Self::Answered { at_tick, .. }
            | Self::Unknown { at_tick }
            | Self::TimedOut { at_tick }
            | Self::Unavailable { at_tick } => *at_tick,
        }
    }
}

fn provenance(index: u64, received_nanos: u128) -> ProvenanceMetadata {
    ProvenanceMetadata {
        message_id: MessageId::new(10_000 + index),
        observed_at: MonotonicInstant::from_nanos(0),
        received_at: MonotonicInstant::from_nanos(received_nanos),
        sensor_id: SensorId::new(900),
        frame_id: FrameId::new("jev_replay").expect("static frame id"),
        units: UnitTag::dimensionless(),
        confidence: Confidence::new(0.5).expect("static confidence"),
        revision: 1,
        evidence_ref: EvidenceRef::new(10_000 + index),
        claim_kind: ObservationKind::Proposal,
    }
}

/// Query id of the `index`-th utterance of a fixture.
pub fn query_id(fixture_id: &str, index: usize) -> QueryId {
    QueryId::new(format!("{fixture_id}/{index}")).expect("non-empty id")
}

/// Build the oracle for a fixture's claims under `mode`.
pub fn build_oracle(
    mode: OracleMode,
    fixture_id: &str,
    claims: &[Claim],
) -> Box<dyn SystemOneProvider> {
    if mode == OracleMode::Live {
        return Box::new(LiveJevUnavailable);
    }
    let mut provider = ReplaySystemOneProvider::new("jev-replay");
    let hyp = HypothesisId::new(HYPOTHESIS).expect("static hypothesis");
    let (t, f) = (
        OptionId::new("true").expect("static"),
        OptionId::new("false").expect("static"),
    );
    for (i, c) in claims.iter().enumerate() {
        let latency_ticks = match mode {
            OracleMode::ReplaySlow { latency_ticks } => latency_ticks,
            _ => c.latency_ticks,
        };
        let behavior = if mode == OracleMode::Never {
            FixtureBehavior::TimedOut
        } else {
            let (pt, pf) = match c.verdict {
                NoulVerdict::True => (0.8, 0.2),
                NoulVerdict::False => (0.2, 0.8),
                NoulVerdict::Unknown => (0.5, 0.5),
            };
            FixtureBehavior::Noul {
                hypothesis_id: hyp.clone(),
                verdict: c.verdict,
                raw_scores: vec![(t.clone(), pt), (f.clone(), pf)],
                validated_distribution: (c.verdict != NoulVerdict::Unknown)
                    .then(|| vec![(t.clone(), pt), (f.clone(), pf)]),
            }
        };
        let fixture = SystemOneFixture::new(behavior, provenance(i as u64, 0))
            .with_simulated_latency(Duration::from_nanos(
                (latency_ticks as u128 * TICK_NANOS) as u64,
            ));
        provider.register_fixture(query_id(fixture_id, i), fixture);
    }
    Box::new(provider)
}

/// Consult the oracle about utterance `index` at control tick `issue_tick`, with a deadline of
/// `timeout_ticks`. Runs on a virtual clock: nothing here sleeps or blocks, and the control loop
/// is never involved. Returns when (in control ticks) cognition learns the outcome.
pub fn consult(
    provider: &dyn SystemOneProvider,
    fixture_id: &str,
    index: usize,
    issue_tick: u64,
    timeout_ticks: u64,
) -> OracleOutcome {
    let now = issue_tick as u128 * TICK_NANOS;
    let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(now));
    let deadline = MonotonicDeadline::from_now(
        &clock,
        Duration::from_nanos((timeout_ticks as u128 * TICK_NANOS) as u64),
    );
    let deadline_tick = issue_tick + timeout_ticks;
    let query = TypedQuery::noul(
        query_id(fixture_id, index),
        HypothesisId::new(HYPOTHESIS).expect("static hypothesis"),
        Some(deadline),
    );
    match provider.query(&query, &clock) {
        Err(_) => OracleOutcome::Unavailable {
            at_tick: issue_tick,
        },
        Ok(resp) => {
            let arrival = (resp.provenance().received_at.as_nanos() / TICK_NANOS) as u64;
            match (resp.status(), resp.noul_verdict()) {
                (ResponseStatus::Answered, Some(v @ (NoulVerdict::True | NoulVerdict::False)))
                    if arrival < deadline_tick =>
                {
                    OracleOutcome::Answered {
                        verdict: v,
                        at_tick: arrival,
                    }
                }
                (ResponseStatus::Unknown, _) if arrival < deadline_tick => {
                    OracleOutcome::Unknown { at_tick: arrival }
                }
                _ => OracleOutcome::TimedOut {
                    at_tick: deadline_tick,
                },
            }
        }
    }
}
