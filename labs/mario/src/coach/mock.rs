//! Deterministic coach mocks. They see only a `CoachObservation` and a level
//! description (the "human" knows the level layout); they output text.

use super::advice::{CoachObservation, CoachProvider, Emission, RawAdvice, Timing};
use crate::fixture::{GAPS, SPIKES, WALLS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MockMode {
    /// Correct advice for the next hazard, after `latency_ticks`.
    Truthful,
    /// Wrong/hallucinated advice, including out-of-catalog "control" text.
    Lying,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MockCoach {
    pub mode: MockMode,
    pub latency_ticks: u64,
    pub ttl_ticks: u64,
    pub max_age_ticks: u64,
    /// Network pause: nothing is emitted for ticks in `[from, from + len)`.
    pub mute: Option<(u64, u64)>,
    /// How far ahead (x units) a hazard is announced.
    pub lookahead: i32,
}

impl MockCoach {
    pub fn truthful(latency_ticks: u64) -> Self {
        Self {
            mode: MockMode::Truthful,
            latency_ticks,
            ttl_ticks: 6,
            max_age_ticks: 8,
            mute: None,
            lookahead: 40,
        }
    }
    pub fn lying(latency_ticks: u64) -> Self {
        Self {
            mode: MockMode::Lying,
            ..Self::truthful(latency_ticks)
        }
    }
    pub fn muted(mut self, from: u64, len: u64) -> Self {
        self.mute = Some((from, len));
        self
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Hz {
    Gap,
    Spike,
    Wall,
}

fn next_hazard(x: i32, look: i32) -> Option<Hz> {
    let all = GAPS
        .iter()
        .map(|&(a, _)| (a, Hz::Gap))
        .chain(SPIKES.iter().map(|&(a, _)| (a, Hz::Spike)))
        .chain(WALLS.iter().map(|&(a, _)| (a, Hz::Wall)));
    all.filter(|&(a, _)| a > x && a - x <= look)
        .min_by_key(|&(a, _)| a)
        .map(|(_, h)| h)
}

impl CoachProvider for MockCoach {
    fn name(&self) -> &'static str {
        match self.mode {
            MockMode::Truthful => "mock_truthful",
            MockMode::Lying => "mock_lying",
        }
    }
    fn emit(&mut self, obs: &CoachObservation) -> Vec<Emission> {
        if let Some((from, len)) = self.mute
            && obs.tick >= from
            && obs.tick < from + len
        {
            return Vec::new();
        }
        // One announcement per 4 ticks while a hazard is near (stateless, so
        // determinism needs no hidden state).
        if !obs.tick.is_multiple_of(4) {
            return Vec::new();
        }
        let timing = Timing {
            latency_ticks: self.latency_ticks,
            ttl_ticks: self.ttl_ticks,
            max_age_ticks: self.max_age_ticks,
        };
        let mk = |t: &str| Emission {
            raw: RawAdvice::new(t),
            timing,
        };
        let hz = next_hazard(obs.x, self.lookahead);
        match (self.mode, hz) {
            (MockMode::Truthful, Some(Hz::Gap)) => vec![mk("gap_ahead")],
            (MockMode::Truthful, Some(Hz::Spike)) => vec![mk("avoid_spikes")],
            (MockMode::Truthful, Some(Hz::Wall)) => vec![mk("climb_platform")],
            (MockMode::Truthful, None) => Vec::new(),
            (MockMode::Lying, Some(Hz::Gap)) => vec![mk("slow_down")],
            (MockMode::Lying, Some(_)) => vec![mk("slow_down"), mk("emit_pad:RIGHT+JUMP")],
            // Hallucination on flat ground.
            (MockMode::Lying, None) => vec![mk("gap_ahead"), mk("climb_platform")],
        }
    }
}
