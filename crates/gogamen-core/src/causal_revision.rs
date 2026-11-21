//! Counterexample-driven revision of causal hypotheses (GOG-165 / DEM-012, first slice).
//!
//! A hypothesis is "under action `a`, bit `t` of the next state is `true` with probability
//! `p`", optionally split into disjoint *regions* by context literals over state bits. When
//! a rule fails (counterexamples), [`CausalRevisor::revise`] searches for a missing context
//! condition, specializes within a complexity limit, and promotes the specialization only with
//! holdout evidence. Rejected candidates and downgraded rules keep their evidence. Every
//! change appends an immutable [`ModelRevision`]; [`CausalRevisor::rollback`] re-installs an
//! earlier one as a *new* revision (history is never rewritten).
//!
//! # Preregistration (fixed BEFORE the first run; see `causal_revision` tests)
//! - Complexity limit: [`RevisionConfig::default`] (`max_literals = 2`, `max_regions = 4`).
//! - Anti-overfitting: a split needs both children with `>= min_support` samples (3) and an
//!   error reduction `>= min_gain` (2), so an isolated case can never get its own region.
//! - Promotion rule: holdout size `>= min_holdout` (20) AND candidate holdout Brier
//!   `<= promote_ratio (0.9) x min(current Brier, prior 0.25)`, i.e. it must also beat the prior.
//!   Otherwise the candidate is rejected (logged, not installed).
//! - Downgrade: an installed rule whose holdout Brier exceeds the uninformed prior (0.25)
//!   is marked `Downgraded` (abstains) in a new revision, evidence retained.
//! - Abstention: no covering region, downgraded rule, or `max(p, 1-p) < confidence_floor (0.7)`.
//!
//! This is a bookkeeping/inference component over discrete bits; it makes no claim about
//! emergent behaviour.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::conditional_effects::{ConditionalError, ConditionalObservation};
use crate::provenance::EvidenceRef;

/// Maximum retained evidence references per region.
pub const MAX_REGION_EVIDENCE: usize = 32;
/// Brier score of the uninformed prior `p = 0.5`.
pub const PRIOR_BRIER: f64 = 0.25;

/// Revision limits and thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevisionConfig {
    /// Maximum context literals per region.
    pub max_literals: usize,
    /// Maximum regions per (action, target) rule.
    pub max_regions: usize,
    /// Minimum samples per region child of a split.
    pub min_support: usize,
    /// Minimum training error reduction of a split.
    pub min_gain: usize,
    /// Minimum holdout samples to promote.
    pub min_holdout: usize,
    /// Candidate Brier must be `<= promote_ratio x` current Brier.
    pub promote_ratio: f64,
    /// Below this confidence the rule abstains.
    pub confidence_floor: f64,
}

impl Default for RevisionConfig {
    fn default() -> Self {
        Self {
            max_literals: 2,
            max_regions: 4,
            min_support: 3,
            min_gain: 2,
            min_holdout: 20,
            promote_ratio: 0.9,
            confidence_floor: 0.7,
        }
    }
}

/// A region of a rule: conjunction of bit literals and the counts observed inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Region {
    /// `(bit index, required value)` literals.
    pub context: Vec<(u16, bool)>,
    /// Samples covered.
    pub n: u64,
    /// Samples where the target bit was `true`.
    pub t: u64,
    /// Retained supporting evidence (capped).
    pub evidence: Vec<EvidenceRef>,
}

impl Region {
    /// Laplace-smoothed probability that the target bit is `true`.
    pub fn p_true(&self) -> f64 {
        (self.t as f64 + 1.0) / (self.n as f64 + 2.0)
    }

    fn covers(&self, state: &[bool]) -> bool {
        self.context
            .iter()
            .all(|&(i, v)| state.get(usize::from(i)) == Some(&v))
    }
}

/// Lifecycle of a rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuleStatus {
    /// Used for prediction.
    Active,
    /// Kept for evidence but abstains.
    Downgraded,
}

/// Rule for one `(action, target bit)` pair.
#[derive(Debug, Clone, PartialEq)]
pub struct RegionRule {
    /// Disjoint regions.
    pub regions: Vec<Region>,
    /// Status.
    pub status: RuleStatus,
}

/// Why a revision was created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevisionCause {
    /// A holdout-validated candidate replaced the current rule.
    Promoted,
    /// A rule worse than the prior was downgraded.
    Downgraded,
    /// An earlier revision was re-installed.
    Rollback {
        /// Revision whose rules were restored.
        restored: u64,
    },
}

/// Immutable snapshot of all rules.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRevision {
    /// Monotonic revision number (`0` = empty).
    pub revision: u64,
    /// Cause of this revision.
    pub cause: Option<RevisionCause>,
    /// Rules keyed by `(action, target bit)`.
    pub rules: BTreeMap<(usize, usize), RegionRule>,
}

/// Result of one revision attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum RevisionOutcome {
    /// Candidate promoted into the given revision.
    Promoted {
        /// New revision.
        revision: u64,
    },
    /// Candidate rejected; current model unchanged, evidence logged.
    Rejected {
        /// Machine-readable reason.
        reason: RejectReason,
    },
    /// Current rule was worse than the prior on holdout and was downgraded.
    Downgraded {
        /// New revision.
        revision: u64,
    },
}

/// Why a candidate was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    /// Holdout too small to promote anything.
    InsufficientHoldout,
    /// No training sample for this action.
    NoTrainingData,
    /// Candidate did not beat the current rule enough on holdout.
    NoHoldoutImprovement,
}

/// A rejected candidate retained with its evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct RejectedCandidate {
    /// Revision that was current when rejected.
    pub at_revision: u64,
    /// Key of the rule.
    pub key: (usize, usize),
    /// Reason.
    pub reason: RejectReason,
    /// Candidate rule (kept for audit).
    pub candidate: RegionRule,
    /// Candidate holdout Brier.
    pub candidate_brier: Option<f64>,
    /// Current holdout Brier.
    pub current_brier: f64,
}

/// Prediction of the revised model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RevisedPrediction {
    /// Calibrated-ish estimate with its support.
    Outcome {
        /// Probability the target bit is `true`.
        p_true: f64,
        /// Samples behind the region.
        n: u64,
    },
    /// The model declines to answer.
    Abstain,
}

/// Counterexample-driven revisor with versioned rules and rollback.
#[derive(Debug, Clone)]
pub struct CausalRevisor {
    cfg: RevisionConfig,
    num_bits: usize,
    num_actions: usize,
    versions: Vec<Arc<ModelRevision>>,
    rejected: Vec<RejectedCandidate>,
}

impl CausalRevisor {
    /// Creates a revisor at revision 0 (no rules).
    pub fn new(
        num_bits: usize,
        num_actions: usize,
        cfg: RevisionConfig,
    ) -> Result<Self, ConditionalError> {
        if num_bits == 0 || num_actions == 0 {
            return Err(ConditionalError::InvalidConfig("shape"));
        }
        if cfg.max_regions == 0 || cfg.min_support == 0 || !(0.0..=1.0).contains(&cfg.promote_ratio)
        {
            return Err(ConditionalError::InvalidConfig("revision"));
        }
        Ok(Self {
            cfg,
            num_bits,
            num_actions,
            versions: vec![Arc::new(ModelRevision {
                revision: 0,
                cause: None,
                rules: BTreeMap::new(),
            })],
            rejected: Vec::new(),
        })
    }

    /// Current revision number.
    pub fn revision(&self) -> u64 {
        self.current().revision
    }

    /// Current immutable revision.
    pub fn current(&self) -> &Arc<ModelRevision> {
        self.versions.last().expect("revision 0 always present")
    }

    /// All stored revisions, oldest first.
    pub fn versions(&self) -> &[Arc<ModelRevision>] {
        &self.versions
    }

    /// Rejected candidates retained for audit.
    pub fn rejected(&self) -> &[RejectedCandidate] {
        &self.rejected
    }

    fn check(&self, o: &ConditionalObservation, target: usize) -> Result<(), ConditionalError> {
        if o.state.len() != self.num_bits
            || o.next.len() != self.num_bits
            || o.action >= self.num_actions
            || target >= self.num_bits
        {
            return Err(ConditionalError::ShapeMismatch("revision observation"));
        }
        Ok(())
    }

    /// Predicts `next[target]` given `state` and `action`; abstains when unsure.
    pub fn predict(&self, state: &[bool], action: usize, target: usize) -> RevisedPrediction {
        predict_with(
            self.current(),
            self.cfg.confidence_floor,
            state,
            action,
            target,
        )
    }

    /// Re-installs revision `to` as a new revision (rollback keeps history).
    pub fn rollback(&mut self, to: u64) -> Option<u64> {
        let old = self.versions.iter().find(|v| v.revision == to)?.clone();
        let revision = self.revision() + 1;
        self.versions.push(Arc::new(ModelRevision {
            revision,
            cause: Some(RevisionCause::Rollback { restored: to }),
            rules: old.rules.clone(),
        }));
        Some(revision)
    }

    /// Attempts to revise the rule for `(action, target)` from `train`, validating on `holdout`.
    pub fn revise(
        &mut self,
        action: usize,
        target: usize,
        train: &[ConditionalObservation],
        holdout: &[ConditionalObservation],
    ) -> Result<RevisionOutcome, ConditionalError> {
        for o in train.iter().chain(holdout) {
            self.check(o, target)?;
        }
        let key = (action, target);
        let samples: Vec<&ConditionalObservation> =
            train.iter().filter(|o| o.action == action).collect();
        let hold: Vec<&ConditionalObservation> =
            holdout.iter().filter(|o| o.action == action).collect();
        let at_revision = self.revision();
        let current_rule = self.current().rules.get(&key).cloned();
        let current_brier = rule_brier(
            current_rule.as_ref(),
            &hold,
            target,
            self.cfg.confidence_floor,
        );

        // Downgrade a failing installed rule even if no candidate can be built.
        let downgrade = |this: &mut Self| -> Option<RevisionOutcome> {
            let rule = current_rule.as_ref()?;
            if rule.status == RuleStatus::Active
                && hold.len() >= this.cfg.min_holdout
                && current_brier > PRIOR_BRIER
            {
                let mut next = this.current().rules.clone();
                next.get_mut(&key)?.status = RuleStatus::Downgraded;
                let revision = this.push(RevisionCause::Downgraded, next);
                return Some(RevisionOutcome::Downgraded { revision });
            }
            None
        };

        if samples.is_empty() {
            return Ok(downgrade(self).unwrap_or_else(|| {
                self.reject(
                    at_revision,
                    key,
                    RejectReason::NoTrainingData,
                    RegionRule {
                        regions: vec![],
                        status: RuleStatus::Active,
                    },
                    None,
                    current_brier,
                )
            }));
        }
        let candidate = self.grow_rule(&samples, target);
        if hold.len() < self.cfg.min_holdout {
            return Ok(self.reject(
                at_revision,
                key,
                RejectReason::InsufficientHoldout,
                candidate,
                None,
                current_brier,
            ));
        }
        let cand_brier = rule_brier(Some(&candidate), &hold, target, self.cfg.confidence_floor);
        if cand_brier <= self.cfg.promote_ratio * current_brier.min(PRIOR_BRIER) {
            let mut next = self.current().rules.clone();
            next.insert(key, candidate);
            let revision = self.push(RevisionCause::Promoted, next);
            return Ok(RevisionOutcome::Promoted { revision });
        }
        if let Some(o) = downgrade(self) {
            return Ok(o);
        }
        Ok(self.reject(
            at_revision,
            key,
            RejectReason::NoHoldoutImprovement,
            candidate,
            Some(cand_brier),
            current_brier,
        ))
    }

    fn reject(
        &mut self,
        at_revision: u64,
        key: (usize, usize),
        reason: RejectReason,
        candidate: RegionRule,
        candidate_brier: Option<f64>,
        current_brier: f64,
    ) -> RevisionOutcome {
        self.rejected.push(RejectedCandidate {
            at_revision,
            key,
            reason,
            candidate,
            candidate_brier,
            current_brier,
        });
        RevisionOutcome::Rejected { reason }
    }

    fn push(&mut self, cause: RevisionCause, rules: BTreeMap<(usize, usize), RegionRule>) -> u64 {
        let revision = self.revision() + 1;
        self.versions.push(Arc::new(ModelRevision {
            revision,
            cause: Some(cause),
            rules,
        }));
        revision
    }

    fn grow_rule(&self, samples: &[&ConditionalObservation], target: usize) -> RegionRule {
        let mut regions = Vec::new();
        let mut leaves_left = self.cfg.max_regions;
        self.grow(samples, Vec::new(), target, &mut regions, &mut leaves_left);
        RegionRule {
            regions,
            status: RuleStatus::Active,
        }
    }

    /// `budget` = leaves still allowed including the ones pending for this node.
    fn grow(
        &self,
        samples: &[&ConditionalObservation],
        context: Vec<(u16, bool)>,
        target: usize,
        out: &mut Vec<Region>,
        budget: &mut usize,
    ) {
        let n = samples.len();
        let t = samples.iter().filter(|o| o.next[target]).count();
        let err = t.min(n - t);
        let mut best: Option<(usize, u16)> = None; // (gain, bit)
        if context.len() < self.cfg.max_literals && *budget >= 2 && err >= self.cfg.min_gain {
            for b in 0..self.num_bits {
                let bi = b as u16;
                if context.iter().any(|&(i, _)| i == bi) {
                    continue;
                }
                let (l, r): (Vec<_>, Vec<_>) = samples.iter().partition(|o| o.state[b]);
                if l.len() < self.cfg.min_support || r.len() < self.cfg.min_support {
                    continue;
                }
                let e = |s: &[&&ConditionalObservation]| {
                    let k = s.iter().filter(|o| o.next[target]).count();
                    k.min(s.len() - k)
                };
                let child = e(&l) + e(&r);
                let gain = err.saturating_sub(child);
                if gain >= self.cfg.min_gain && best.is_none_or(|(g, _)| gain > g) {
                    best = Some((gain, bi));
                }
            }
        }
        if let Some((_, bit)) = best {
            *budget -= 1; // one leaf becomes two
            for value in [false, true] {
                let part: Vec<&ConditionalObservation> = samples
                    .iter()
                    .copied()
                    .filter(|o| o.state[usize::from(bit)] == value)
                    .collect();
                let mut ctx = context.clone();
                ctx.push((bit, value));
                self.grow(&part, ctx, target, out, budget);
            }
        } else {
            out.push(Region {
                context,
                n: n as u64,
                t: t as u64,
                evidence: samples
                    .iter()
                    .take(MAX_REGION_EVIDENCE)
                    .map(|o| o.evidence)
                    .collect(),
            });
        }
    }
}

fn predict_with(
    rev: &ModelRevision,
    floor: f64,
    state: &[bool],
    action: usize,
    target: usize,
) -> RevisedPrediction {
    let Some(rule) = rev.rules.get(&(action, target)) else {
        return RevisedPrediction::Abstain;
    };
    if rule.status != RuleStatus::Active {
        return RevisedPrediction::Abstain;
    }
    match rule.regions.iter().find(|r| r.covers(state)) {
        Some(r) => {
            let p = r.p_true();
            if p.max(1.0 - p) < floor {
                RevisedPrediction::Abstain
            } else {
                RevisedPrediction::Outcome { p_true: p, n: r.n }
            }
        }
        None => RevisedPrediction::Abstain,
    }
}

/// Mean Brier on `hold`; abstentions and absent rules score as the uninformed prior.
fn rule_brier(
    rule: Option<&RegionRule>,
    hold: &[&ConditionalObservation],
    target: usize,
    floor: f64,
) -> f64 {
    if hold.is_empty() {
        return PRIOR_BRIER;
    }
    let sum: f64 = hold
        .iter()
        .map(|o| {
            let p = rule
                .filter(|r| r.status == RuleStatus::Active)
                .and_then(|r| r.regions.iter().find(|g| g.covers(&o.state)))
                .map(Region::p_true)
                .filter(|p| p.max(1.0 - *p) >= floor)
                .unwrap_or(0.5);
            let y = if o.next[target] { 1.0 } else { 0.0 };
            (p - y) * (p - y)
        })
        .sum();
    sum / hold.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::ObservationId;

    const BITS: usize = 3;

    fn obs(id: u64, s: [bool; 3], moved: bool) -> ConditionalObservation {
        ConditionalObservation {
            id: ObservationId::new(id),
            evidence: EvidenceRef::new(9000 + id),
            state: s.to_vec(),
            action: 0,
            next: vec![moved, false, false],
        }
    }

    /// bit1 = context ("blocked"): push moves iff NOT blocked. bit0 irrelevant, bit2 marks the
    /// single isolated outlier (unblocked yet did not move).
    fn fixture(start: u64, n: u64, outlier_at: Option<u64>) -> Vec<ConditionalObservation> {
        (0..n)
            .map(|i| {
                let blocked = i % 2 == 1;
                let b0 = (i / 2) % 2 == 1;
                let outlier = outlier_at == Some(i);
                obs(start + i, [b0, blocked, outlier], !blocked && !outlier)
            })
            .collect()
    }

    fn revisor() -> CausalRevisor {
        CausalRevisor::new(BITS, 1, RevisionConfig::default()).unwrap()
    }

    #[test]
    fn causal_revision_learns_two_regions_without_overfitting_outlier() {
        let mut r = revisor();
        let train = fixture(1, 60, Some(10)); // i=10 is even => unblocked outlier
        let hold = fixture(1000, 40, None);
        let out = r.revise(0, 0, &train, &hold).unwrap();
        assert_eq!(out, RevisionOutcome::Promoted { revision: 1 });
        let rule = &r.current().rules[&(0, 0)];
        // Exactly two regions, split on bit1 only; outlier (bit2) did not get a region.
        assert_eq!(rule.regions.len(), 2, "{:?}", rule.regions);
        assert!(
            rule.regions
                .iter()
                .all(|g| g.context == vec![(1, false)] || g.context == vec![(1, true)])
        );
        let free = rule
            .regions
            .iter()
            .find(|g| g.context == vec![(1, false)])
            .unwrap();
        assert_eq!((free.n, free.t), (30, 29));
        assert!(matches!(
            r.predict(&[false, false, false], 0, 0),
            RevisedPrediction::Outcome { p_true, .. } if p_true > 0.9
        ));
        assert!(matches!(
            r.predict(&[false, true, false], 0, 0),
            RevisedPrediction::Outcome { p_true, .. } if p_true < 0.1
        ));
    }

    #[test]
    fn causal_revision_versions_rollback_and_holdout_gate() {
        let mut r = revisor();
        // Naive hypothesis learned only from unblocked cases: "push always moves".
        let naive_train: Vec<_> = fixture(1, 60, None)
            .into_iter()
            .filter(|o| !o.state[1])
            .collect();
        let hold = fixture(1000, 40, None);
        let hold_unblocked: Vec<_> = hold.iter().filter(|o| !o.state[1]).cloned().collect();
        // Prior-only current model => naive rule promoted on (unblocked-only) holdout evidence.
        assert_eq!(
            r.revise(0, 0, &naive_train, &hold_unblocked).unwrap(),
            RevisionOutcome::Promoted { revision: 1 }
        );
        let naive = r.current().clone();
        // Counterexamples (blocked cases) arrive: specialize, validated on holdout.
        let train = fixture(2000, 60, Some(10));
        assert_eq!(
            r.revise(0, 0, &train, &hold).unwrap(),
            RevisionOutcome::Promoted { revision: 2 }
        );
        assert_eq!(r.versions().len(), 3);
        assert_eq!(r.versions()[1].rules, naive.rules, "previous rule stored");
        assert_eq!(r.current().rules[&(0, 0)].regions.len(), 2);
        // Holdout gate: too-small holdout never promotes.
        let before = r.revision();
        let small = fixture(3000, 10, None);
        assert_eq!(
            r.revise(0, 0, &train, &small).unwrap(),
            RevisionOutcome::Rejected {
                reason: RejectReason::InsufficientHoldout
            }
        );
        assert_eq!(r.revision(), before);
        assert_eq!(r.rejected().len(), 1);
        assert!(!r.rejected()[0].candidate.regions[0].evidence.is_empty());
        // Rollback restores the naive rules as a new revision.
        assert_eq!(r.rollback(1), Some(3));
        assert_eq!(r.current().rules, naive.rules);
        assert_eq!(
            r.current().cause,
            Some(RevisionCause::Rollback { restored: 1 })
        );
        assert_eq!(r.rollback(99), None);
        assert_eq!(r.versions().len(), 4);
    }

    /// Deterministic splitmix64 coin.
    fn coin(state: &mut u64) -> bool {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) & 1 == 1
    }

    #[test]
    fn causal_revision_noise_and_contradiction_abstain_without_explosion() {
        let mut r = revisor();
        let mut seed = 0xC0FFEE_u64;
        let mut make = |start: u64, n: u64| -> Vec<ConditionalObservation> {
            (0..n)
                .map(|i| {
                    let s = [coin(&mut seed), coin(&mut seed), coin(&mut seed)];
                    obs(start + i, s, coin(&mut seed)) // outcome independent of state
                })
                .collect()
        };
        let train = make(1, 200);
        let hold = make(1000, 200);
        let out = r.revise(0, 0, &train, &hold).unwrap();
        // Either rejected, or (if promoted) region count is bounded; never explosive.
        let regions = r.rejected().last().map_or_else(
            || r.current().rules[&(0, 0)].regions.len(),
            |c| c.candidate.regions.len(),
        );
        assert!(
            regions <= RevisionConfig::default().max_regions,
            "regions={regions}"
        );
        println!(
            "noise outcome={out:?} regions={regions} rejected={}",
            r.rejected().len()
        );
        // Whatever happened, predictions on noise must abstain (p near 0.5 or no rule).
        let abstained = (0..8u8)
            .filter(|m| {
                let s = [m & 1 == 1, m & 2 == 2, m & 4 == 4];
                r.predict(&s, 0, 0) == RevisedPrediction::Abstain
            })
            .count();
        println!("noise abstentions={abstained}/8");
        assert!(abstained >= 6, "abstained={abstained}");
        assert_eq!(r.revision(), 0, "noise must not be promoted: {out:?}");
        assert!(matches!(out, RevisionOutcome::Rejected { .. }));

        // Contradiction: identical contexts with 50/50 outcomes -> no split, abstain.
        let contra: Vec<_> = (0..60)
            .map(|i| obs(5000 + i, [false, false, false], i % 2 == 0))
            .collect();
        let chold: Vec<_> = (0..40)
            .map(|i| obs(6000 + i, [false, false, false], i % 2 == 0))
            .collect();
        let _ = r.revise(0, 0, &contra, &chold).unwrap();
        assert_eq!(r.predict(&[false; 3], 0, 0), RevisedPrediction::Abstain);
        assert!(
            r.current()
                .rules
                .get(&(0, 0))
                .is_none_or(|g| g.regions.len() <= 1)
        );
    }

    #[test]
    fn causal_revision_downgrades_rule_worse_than_prior_keeping_evidence() {
        let mut r = revisor();
        let train = fixture(1, 60, None);
        let hold = fixture(1000, 40, None);
        r.revise(0, 0, &train, &hold).unwrap();
        // Dynamics flip: outcomes inverted on holdout; training now contradictory noise.
        let flipped: Vec<_> = fixture(4000, 40, None)
            .into_iter()
            .map(|mut o| {
                o.next[0] = !o.next[0];
                o
            })
            .collect();
        let mixed: Vec<_> = (0..40)
            .map(|i| obs(7000 + i, [false, false, false], i % 2 == 0))
            .collect();
        let out = r.revise(0, 0, &mixed, &flipped).unwrap();
        assert!(matches!(out, RevisionOutcome::Downgraded { .. }), "{out:?}");
        let rule = &r.current().rules[&(0, 0)];
        assert_eq!(rule.status, RuleStatus::Downgraded);
        assert!(!rule.regions.is_empty() && !rule.regions[0].evidence.is_empty());
        assert_eq!(
            r.predict(&[false, false, false], 0, 0),
            RevisedPrediction::Abstain
        );
    }
}
