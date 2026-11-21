//! Calibration metrics and the policy comparison over annotated fixtures.

use super::episode::{DialogEpisode, EpisodeResult, EpisodeVerdict};
use super::fixtures::DialogFixture;
use super::oracle::OracleMode;
use super::policy::{DialogPolicy, PolicyKind};

/// Brier score of probabilistic forecasts `p` of binary outcomes `y` (lower is better).
pub fn brier_score(pairs: &[(f64, bool)]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    pairs
        .iter()
        .map(|(p, y)| (p - f64::from(u8::from(*y))).powi(2))
        .sum::<f64>()
        / pairs.len() as f64
}

/// Expected calibration error of the confidence in the MAP hypothesis: items are
/// `(confidence in [0.5, 1], map_was_correct)`, grouped into `bins` equal-width bins over
/// `[0.5, 1.0]`; ECE = sum over bins of `n_b / n * |mean_conf_b - accuracy_b|`.
pub fn expected_calibration_error(items: &[(f64, bool)], bins: usize) -> f64 {
    if items.is_empty() || bins == 0 {
        return 0.0;
    }
    let mut acc = vec![(0usize, 0.0f64, 0usize); bins];
    for (conf, ok) in items {
        let idx = (((conf - 0.5) / 0.5 * bins as f64) as usize).min(bins - 1);
        acc[idx].0 += 1;
        acc[idx].1 += conf;
        acc[idx].2 += usize::from(*ok);
    }
    let n = items.len() as f64;
    acc.iter()
        .filter(|(c, _, _)| *c > 0)
        .map(|(c, sum, ok)| {
            let c = *c as f64;
            (c / n) * (sum / c - *ok as f64 / c).abs()
        })
        .sum()
}

/// Aggregated measurements of one policy over a fixture set.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyRow {
    /// Policy family.
    pub kind: PolicyKind,
    /// Fixtures run.
    pub episodes: usize,
    /// Reached the true door.
    pub successes: usize,
    /// Reached the wrong door.
    pub wrong_door: usize,
    /// Waited without moving.
    pub abstained: usize,
    /// Body collisions.
    pub collisions: usize,
    /// Timeouts and safe stops.
    pub other_failures: usize,
    /// Total `AskHuman` questions.
    pub queries: u32,
    /// Summed task utility (reward/penalty per policy, 0 for abstention/other).
    pub utility: f64,
    /// Brier score of the final `P(left)` against the annotation.
    pub brier: f64,
    /// ECE (5 bins) of the MAP confidence against MAP correctness.
    pub ece: f64,
    /// Mean final confidence in the MAP door.
    pub mean_confidence: f64,
    /// MAP accuracy (fraction of fixtures where the MAP door equals the truth).
    pub map_accuracy: f64,
    /// Mean control ticks per episode.
    pub mean_ticks: f64,
}

/// Run `policy` over `fixtures` (episode seed = fixture index + 1) and aggregate.
pub fn run_policy(
    fixtures: &[DialogFixture],
    policy: DialogPolicy,
    mode: OracleMode,
) -> (PolicyRow, Vec<EpisodeResult>) {
    let results: Vec<EpisodeResult> = fixtures
        .iter()
        .enumerate()
        .map(|(i, f)| DialogEpisode::run(f, policy, mode, i as u64 + 1))
        .collect();
    let count = |v: EpisodeVerdict| results.iter().filter(|r| r.verdict == v).count();
    let brier_pairs: Vec<(f64, bool)> = results
        .iter()
        .zip(fixtures)
        .map(|(r, f)| (r.p_left, f.truth_left))
        .collect();
    let ece_items: Vec<(f64, bool)> = brier_pairs
        .iter()
        .map(|(p, truth)| (p.max(1.0 - p), (*p >= 0.5) == *truth))
        .collect();
    let n = results.len().max(1) as f64;
    let utility = results
        .iter()
        .map(|r| match r.verdict {
            EpisodeVerdict::Success => policy.reward,
            EpisodeVerdict::WrongDoor => policy.penalty,
            _ => 0.0,
        })
        .sum::<f64>()
        - results
            .iter()
            .map(|r| f64::from(r.queries) * policy.ask_cost_units as f64)
            .sum::<f64>();
    let row = PolicyRow {
        kind: policy.kind,
        episodes: results.len(),
        successes: count(EpisodeVerdict::Success),
        wrong_door: count(EpisodeVerdict::WrongDoor),
        abstained: count(EpisodeVerdict::Abstained),
        collisions: count(EpisodeVerdict::Collision),
        other_failures: count(EpisodeVerdict::Timeout) + count(EpisodeVerdict::SafeStop),
        queries: results.iter().map(|r| r.queries).sum(),
        utility,
        brier: brier_score(&brier_pairs),
        ece: expected_calibration_error(&ece_items, 5),
        mean_confidence: ece_items.iter().map(|(c, _)| c).sum::<f64>() / n,
        map_accuracy: ece_items.iter().filter(|(_, ok)| *ok).count() as f64 / n,
        mean_ticks: results.iter().map(|r| r.ticks as f64).sum::<f64>() / n,
    };
    (row, results)
}
