//! Bounded predicate invention with an MDL criterion (GOG-166 / DEM-013, first slice).
//!
//! When a rule keeps failing because the missing context is a *compound* of state bits (for
//! example `b0 XOR b1`, where neither bit alone is informative), [`PredicateInventor`]
//! enumerates compound predicates from a small relational vocabulary ([`PredicateOp`]) over
//! the base bits, keeps only those that pay for their description, and re-runs the existing
//! [`CausalRevisor`] on the state augmented with the invented bits. Holdout decides adoption.
//!
//! # Preregistration (fixed BEFORE the first run; see the `predicate_invention` tests)
//! - Vocabulary: arity-2 `And, Or, Xor, Iff, AndNot, NotAnd`; arity-3 `And3, Or3, Parity3,
//!   Majority3`; arguments are distinct base bits in ascending order. Nothing else.
//! - Limits ([`InventionConfig::default`]): `max_arity = 3`, `max_candidates = 256`,
//!   `max_evals = 2048` scoring evaluations (a deterministic work budget standing in for time:
//!   no wall clock, so reports are reproducible), `max_train = 400`, `min_evidence = 20`,
//!   `max_accepted = 2`.
//! - Cost: `description_bits = log2(#ops) + arity * log2(num_bits)`. `benefit_bits` is the
//!   reduction in two-part code length (data entropy bits + `0.5*log2(n+1)` per leaf) of the best
//!   single split on the candidate versus the best split on the current features (base bits
//!   plus accepted predicates), on TRAIN only.
//! - Accept a candidate iff `benefit_bits - description_bits >= min_net_bits (2.0)`.
//! - Redundant: a candidate whose train truth table equals (or is the complement of) a base bit
//!   or an accepted predicate is rejected as `Redundant` without scoring.
//! - Adoption (HOLDOUT): the augmented revisor must promote a rule and its holdout Brier must be
//!   `<= adopt_ratio (0.9) x` the baseline (no-invention) holdout Brier; otherwise the invention
//!   is discarded and reported as a negative.
//!
//! This is a bounded search over discrete bits, not a theorem prover; it makes no claim about
//! emergent behaviour. Predicates are only searched over the given base bits.

use std::collections::BTreeMap;

use crate::causal_revision::{CausalRevisor, RevisedPrediction, RevisionConfig, RevisionOutcome};
use crate::conditional_effects::{ConditionalError, ConditionalObservation};

/// Relational vocabulary (bounded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PredicateOp {
    /// `a & b`.
    And,
    /// `a | b`.
    Or,
    /// `a ^ b`.
    Xor,
    /// `a == b`.
    Iff,
    /// `a & !b`.
    AndNot,
    /// `!a & b`.
    NotAnd,
    /// `a & b & c`.
    And3,
    /// `a | b | c`.
    Or3,
    /// odd number of true among `a, b, c`.
    Parity3,
    /// at least two of `a, b, c`.
    Majority3,
}

impl PredicateOp {
    const ALL: [Self; 10] = [
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Iff,
        Self::AndNot,
        Self::NotAnd,
        Self::And3,
        Self::Or3,
        Self::Parity3,
        Self::Majority3,
    ];

    /// Number of arguments.
    pub const fn arity(self) -> usize {
        match self {
            Self::And | Self::Or | Self::Xor | Self::Iff | Self::AndNot | Self::NotAnd => 2,
            Self::And3 | Self::Or3 | Self::Parity3 | Self::Majority3 => 3,
        }
    }

    fn eval(self, v: &[bool]) -> bool {
        match self {
            Self::And => v[0] && v[1],
            Self::Or => v[0] || v[1],
            Self::Xor => v[0] ^ v[1],
            Self::Iff => v[0] == v[1],
            Self::AndNot => v[0] && !v[1],
            Self::NotAnd => !v[0] && v[1],
            Self::And3 => v[0] && v[1] && v[2],
            Self::Or3 => v[0] || v[1] || v[2],
            Self::Parity3 => v[0] ^ v[1] ^ v[2],
            Self::Majority3 => usize::from(v[0]) + usize::from(v[1]) + usize::from(v[2]) >= 2,
        }
    }
}

/// A compound predicate over base bits.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Predicate {
    /// Operator.
    pub op: PredicateOp,
    /// Distinct base-bit indices, ascending.
    pub args: Vec<u16>,
}

impl Predicate {
    /// Evaluates on a base state; `None` if an index is out of range.
    pub fn eval(&self, state: &[bool]) -> Option<bool> {
        let vals: Option<Vec<bool>> = self
            .args
            .iter()
            .map(|&i| state.get(usize::from(i)).copied())
            .collect();
        Some(self.op.eval(&vals?))
    }

    /// Human-readable form, e.g. `Xor(b0,b1)`.
    pub fn describe(&self) -> String {
        let a: Vec<String> = self.args.iter().map(|i| format!("b{i}")).collect();
        format!("{:?}({})", self.op, a.join(","))
    }
}

/// Limits and thresholds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InventionConfig {
    /// Maximum predicate arity (2 or 3).
    pub max_arity: usize,
    /// Maximum enumerated candidates (size limit).
    pub max_candidates: usize,
    /// Maximum scoring evaluations (deterministic work budget).
    pub max_evals: usize,
    /// Maximum training samples used (evidence limit).
    pub max_train: usize,
    /// Minimum training samples for the action; fewer refuses to invent.
    pub min_evidence: usize,
    /// Maximum accepted predicates.
    pub max_accepted: usize,
    /// Minimum `benefit - description` in bits to accept.
    pub min_net_bits: f64,
    /// Holdout Brier must be `<= adopt_ratio x` baseline.
    pub adopt_ratio: f64,
}

impl Default for InventionConfig {
    fn default() -> Self {
        Self {
            max_arity: 3,
            max_candidates: 256,
            max_evals: 2048,
            max_train: 400,
            min_evidence: 20,
            max_accepted: 2,
            min_net_bits: 2.0,
            adopt_ratio: 0.9,
        }
    }
}

/// Verdict for one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CandidateVerdict {
    /// Kept: pays for its description.
    Accepted,
    /// Same truth table (or complement) as an existing feature.
    Redundant {
        /// Description of the feature it duplicates.
        duplicate_of: String,
    },
    /// Benefit does not cover the description cost.
    NoNetBenefit,
    /// Evaluation budget exhausted before scoring.
    BudgetExhausted,
}

/// One candidate hypothesis in the report.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateReport {
    /// The predicate.
    pub predicate: Predicate,
    /// Description cost in bits.
    pub description_bits: f64,
    /// Predictive benefit in bits (train, last scoring round; 0 if unscored).
    pub benefit_bits: f64,
    /// Verdict.
    pub verdict: CandidateVerdict,
}

/// Reproducible report of one invention run.
#[derive(Debug, Clone, PartialEq)]
pub struct InventionReport {
    /// Candidates in enumeration order.
    pub candidates: Vec<CandidateReport>,
    /// Candidates that were not enumerated because of `max_candidates`.
    pub enumeration_truncated: bool,
    /// Scoring evaluations used.
    pub evals: usize,
    /// Training samples used.
    pub train_used: usize,
    /// Baseline (no invention) holdout Brier.
    pub baseline_brier: f64,
    /// Baseline holdout 0/1 error (abstention counts 0.5).
    pub baseline_error: f64,
    /// Holdout Brier with invented predicates (equals baseline if none accepted).
    pub invented_brier: f64,
    /// Holdout 0/1 error with invented predicates.
    pub invented_error: f64,
    /// Whether the invention was adopted.
    pub adopted: bool,
}

impl InventionReport {
    /// Deterministic textual rendering (stable across runs).
    pub fn render(&self) -> String {
        let mut s = format!(
            "train={} evals={} truncated={} baseline(brier={:.4},err={:.4}) invented(brier={:.4},err={:.4}) adopted={}\n",
            self.train_used,
            self.evals,
            self.enumeration_truncated,
            self.baseline_brier,
            self.baseline_error,
            self.invented_brier,
            self.invented_error,
            self.adopted
        );
        for c in &self.candidates {
            s.push_str(&format!(
                "{} cost={:.2} benefit={:.2} {:?}\n",
                c.predicate.describe(),
                c.description_bits,
                c.benefit_bits,
                c.verdict
            ));
        }
        s
    }
}

/// A model with adopted invented predicates.
#[derive(Debug, Clone)]
pub struct InventedModel {
    /// Adopted predicates, in invention order.
    pub predicates: Vec<Predicate>,
    revisor: CausalRevisor,
    base_bits: usize,
}

impl InventedModel {
    fn augment(&self, state: &[bool]) -> Vec<bool> {
        augment(state, &self.predicates)
    }

    /// Predicts `next[target]`; abstains if the state shape is wrong.
    pub fn predict(&self, state: &[bool], action: usize, target: usize) -> RevisedPrediction {
        if state.len() != self.base_bits {
            return RevisedPrediction::Abstain;
        }
        self.revisor.predict(&self.augment(state), action, target)
    }

    /// The underlying revisor (rules, rejected log, versions).
    pub fn revisor(&self) -> &CausalRevisor {
        &self.revisor
    }
}

/// Result of an invention run.
#[derive(Debug, Clone)]
pub struct InventionOutcome {
    /// Reproducible report.
    pub report: InventionReport,
    /// Present only when the invention was adopted.
    pub model: Option<InventedModel>,
}

/// Bounded predicate inventor.
#[derive(Debug, Clone)]
pub struct PredicateInventor {
    num_bits: usize,
    num_actions: usize,
    cfg: InventionConfig,
    revision: RevisionConfig,
}

fn augment(state: &[bool], preds: &[Predicate]) -> Vec<bool> {
    let mut v = state.to_vec();
    v.extend(preds.iter().map(|p| p.eval(state).unwrap_or(false)));
    v
}

fn lg(x: f64) -> f64 {
    x.log2()
}

/// Two-part code length of a leaf with `t` positives out of `n`.
fn leaf_bits(t: u64, n: u64) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let (t, nf) = (t as f64, n as f64);
    let h = |k: f64| if k <= 0.0 { 0.0 } else { -k * lg(k / nf) };
    h(t) + h(nf - t) + 0.5 * lg(nf + 1.0)
}

/// Code length of splitting by a feature column (`ys` = target, `col` = feature).
fn split_bits(col: &[bool], ys: &[bool]) -> f64 {
    let mut c = [[0u64; 2]; 2]; // [feature][n, t]
    for (&f, &y) in col.iter().zip(ys) {
        c[usize::from(f)][0] += 1;
        c[usize::from(f)][1] += u64::from(y);
    }
    leaf_bits(c[0][1], c[0][0]) + leaf_bits(c[1][1], c[1][0])
}

fn unsplit_bits(ys: &[bool]) -> f64 {
    leaf_bits(ys.iter().filter(|&&y| y).count() as u64, ys.len() as u64)
}

/// Normalized truth-table signature (complement-invariant: first sample forced `false`).
fn signature(col: &[bool]) -> Vec<bool> {
    match col.first() {
        Some(true) => col.iter().map(|b| !b).collect(),
        _ => col.to_vec(),
    }
}

fn holdout_scores(
    hold: &[&ConditionalObservation],
    target: usize,
    pred: impl Fn(&[bool]) -> RevisedPrediction,
) -> (f64, f64) {
    if hold.is_empty() {
        return (crate::causal_revision::PRIOR_BRIER, 0.5);
    }
    let (mut brier, mut err) = (0.0, 0.0);
    for o in hold {
        let y = f64::from(u8::from(o.next[target]));
        match pred(&o.state) {
            RevisedPrediction::Outcome { p_true, .. } => {
                brier += (p_true - y).powi(2);
                err += f64::from(u8::from((p_true >= 0.5) != o.next[target]));
            }
            RevisedPrediction::Abstain => {
                brier += crate::causal_revision::PRIOR_BRIER;
                err += 0.5;
            }
        }
    }
    let n = hold.len() as f64;
    (brier / n, err / n)
}

impl PredicateInventor {
    /// Creates an inventor over `num_bits` base bits.
    pub fn new(
        num_bits: usize,
        num_actions: usize,
        cfg: InventionConfig,
        revision: RevisionConfig,
    ) -> Result<Self, ConditionalError> {
        if num_bits < 2 || num_actions == 0 || !(2..=3).contains(&cfg.max_arity) {
            return Err(ConditionalError::InvalidConfig("invention shape"));
        }
        if cfg.max_candidates == 0 || cfg.max_train == 0 || cfg.max_accepted == 0 {
            return Err(ConditionalError::InvalidConfig("invention limits"));
        }
        Ok(Self {
            num_bits,
            num_actions,
            cfg,
            revision,
        })
    }

    fn enumerate(&self) -> (Vec<Predicate>, bool) {
        let n = self.num_bits as u16;
        let mut out = Vec::new();
        for op in PredicateOp::ALL {
            if op.arity() > self.cfg.max_arity {
                continue;
            }
            let mut push = |args: Vec<u16>| -> bool {
                if out.len() >= self.cfg.max_candidates {
                    return false;
                }
                out.push(Predicate { op, args });
                true
            };
            for a in 0..n {
                for b in a + 1..n {
                    if op.arity() == 2 {
                        if !push(vec![a, b]) {
                            return (out, true);
                        }
                    } else {
                        for c in b + 1..n {
                            if !push(vec![a, b, c]) {
                                return (out, true);
                            }
                        }
                    }
                }
            }
        }
        (out, false)
    }

    fn description_bits(&self, p: &Predicate) -> f64 {
        lg(PredicateOp::ALL.len() as f64) + p.args.len() as f64 * lg(self.num_bits as f64)
    }

    /// Proposes, scores (MDL on train) and validates (holdout) compound predicates for the rule
    /// `(action, target)`.
    pub fn invent(
        &self,
        action: usize,
        target: usize,
        train: &[ConditionalObservation],
        holdout: &[ConditionalObservation],
    ) -> Result<InventionOutcome, ConditionalError> {
        for o in train.iter().chain(holdout) {
            if o.state.len() != self.num_bits
                || o.next.len() != self.num_bits
                || o.action >= self.num_actions
                || target >= self.num_bits
            {
                return Err(ConditionalError::ShapeMismatch("invention observation"));
            }
        }
        let used: Vec<&ConditionalObservation> = train
            .iter()
            .filter(|o| o.action == action)
            .take(self.cfg.max_train)
            .collect();
        let hold: Vec<&ConditionalObservation> =
            holdout.iter().filter(|o| o.action == action).collect();
        let train_used: Vec<ConditionalObservation> = used.iter().map(|&o| o.clone()).collect();

        // Baseline: the existing revisor without invention.
        let mut base = CausalRevisor::new(self.num_bits, self.num_actions, self.revision)?;
        base.revise(action, target, &train_used, holdout)?;
        let (baseline_brier, baseline_error) =
            holdout_scores(&hold, target, |s| base.predict(s, action, target));

        let mut report = InventionReport {
            candidates: Vec::new(),
            enumeration_truncated: false,
            evals: 0,
            train_used: used.len(),
            baseline_brier,
            baseline_error,
            invented_brier: baseline_brier,
            invented_error: baseline_error,
            adopted: false,
        };
        if used.len() < self.cfg.min_evidence {
            return Ok(InventionOutcome {
                report,
                model: None,
            });
        }

        let ys: Vec<bool> = used.iter().map(|o| o.next[target]).collect();
        let (cands, truncated) = self.enumerate();
        report.enumeration_truncated = truncated;
        let cols: Vec<Vec<bool>> = cands
            .iter()
            .map(|p| {
                used.iter()
                    .map(|o| p.eval(&o.state).unwrap_or(false))
                    .collect()
            })
            .collect();
        report.candidates = cands
            .iter()
            .map(|p| CandidateReport {
                predicate: p.clone(),
                description_bits: self.description_bits(p),
                benefit_bits: 0.0,
                verdict: CandidateVerdict::NoNetBenefit,
            })
            .collect();

        // Existing features: base bits (by signature) then accepted predicates.
        let mut known: BTreeMap<Vec<bool>, String> = BTreeMap::new();
        let mut best_existing = unsplit_bits(&ys);
        for b in 0..self.num_bits {
            let col: Vec<bool> = used.iter().map(|o| o.state[b]).collect();
            best_existing = best_existing.min(split_bits(&col, &ys));
            known
                .entry(signature(&col))
                .or_insert_with(|| format!("b{b}"));
        }
        let mut accepted: Vec<usize> = Vec::new();
        let mut done = vec![false; cands.len()];

        for _round in 0..self.cfg.max_accepted {
            let mut best: Option<(usize, f64)> = None;
            for i in 0..cands.len() {
                if done[i] {
                    continue;
                }
                if let Some(dup) = known.get(&signature(&cols[i])) {
                    report.candidates[i].verdict = CandidateVerdict::Redundant {
                        duplicate_of: dup.clone(),
                    };
                    done[i] = true;
                    continue;
                }
                if report.evals >= self.cfg.max_evals {
                    report.candidates[i].verdict = CandidateVerdict::BudgetExhausted;
                    continue;
                }
                report.evals += 1;
                let benefit = best_existing - split_bits(&cols[i], &ys);
                report.candidates[i].benefit_bits = benefit;
                let net = benefit - report.candidates[i].description_bits;
                if net >= self.cfg.min_net_bits && best.is_none_or(|(_, bn)| net > bn) {
                    best = Some((i, net));
                }
            }
            let Some((i, _)) = best else { break };
            report.candidates[i].verdict = CandidateVerdict::Accepted;
            done[i] = true;
            best_existing = best_existing.min(split_bits(&cols[i], &ys));
            known.insert(signature(&cols[i]), cands[i].describe());
            accepted.push(i);
        }

        if accepted.is_empty() {
            return Ok(InventionOutcome {
                report,
                model: None,
            });
        }

        // Holdout validation through the existing revisor on augmented states.
        let preds: Vec<Predicate> = accepted.iter().map(|&i| cands[i].clone()).collect();
        let aug_bits = self.num_bits + preds.len();
        let lift = |o: &ConditionalObservation| {
            let mut n = o.clone();
            n.state = augment(&o.state, &preds);
            n.next.resize(aug_bits, false);
            n
        };
        let tr: Vec<ConditionalObservation> = train_used.iter().map(lift).collect();
        let ho: Vec<ConditionalObservation> = holdout.iter().map(lift).collect();
        let mut rev = CausalRevisor::new(aug_bits, self.num_actions, self.revision)?;
        let outcome = rev.revise(action, target, &tr, &ho)?;
        let model = InventedModel {
            predicates: preds,
            revisor: rev,
            base_bits: self.num_bits,
        };
        let (b, e) = holdout_scores(&hold, target, |s| model.predict(s, action, target));
        report.invented_brier = b;
        report.invented_error = e;
        report.adopted = matches!(outcome, RevisionOutcome::Promoted { .. })
            && b <= self.cfg.adopt_ratio * baseline_brier;
        let model = report.adopted.then_some(model);
        Ok(InventionOutcome { report, model })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::ObservationId;
    use crate::provenance::EvidenceRef;

    const BITS: usize = 4;

    /// Deterministic LCG bit source.
    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// `rule(state)` decides next[0]; bits 0..4 are uniform random (b2,b3 are distractors).
    fn data(
        seed: u64,
        start: u64,
        n: u64,
        mut rule: impl FnMut(&[bool]) -> bool,
    ) -> Vec<ConditionalObservation> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let r = lcg(&mut s);
                let state: Vec<bool> = (0..BITS).map(|k| (r >> k) & 1 == 1).collect();
                let y = rule(&state);
                ConditionalObservation {
                    id: ObservationId::new(start + i),
                    evidence: EvidenceRef::new(7000 + start + i),
                    state,
                    action: 0,
                    next: vec![y, false, false, false],
                }
            })
            .collect()
    }

    fn xor_rule(s: &[bool]) -> bool {
        s[0] ^ s[1]
    }

    fn inventor(cfg: InventionConfig) -> PredicateInventor {
        PredicateInventor::new(BITS, 1, cfg, RevisionConfig::default()).unwrap()
    }

    #[test]
    fn predicate_invention_improves_holdout_on_hidden_xor_context() {
        let train = data(1, 1, 240, xor_rule);
        let hold = data(2, 10_000, 120, xor_rule);
        let out = inventor(InventionConfig::default())
            .invent(0, 0, &train, &hold)
            .unwrap();
        let r = &out.report;
        println!("{}", r.render());
        assert!(r.adopted, "{}", r.render());
        // single literals carry no information about XOR: baseline stays near the prior
        assert!(r.baseline_brier > 0.2, "baseline {}", r.baseline_brier);
        assert!(r.invented_brier < 0.05, "invented {}", r.invented_brier);
        assert!(r.invented_error < r.baseline_error);
        let m = out.model.unwrap();
        assert_eq!(
            m.predicates[0],
            Predicate {
                op: PredicateOp::Xor,
                args: vec![0, 1]
            }
        );
        // the invented context transfers to a fresh state it never saw as a literal
        match m.predict(&[true, false, true, true], 0, 0) {
            RevisedPrediction::Outcome { p_true, .. } => assert!(p_true > 0.9),
            RevisedPrediction::Abstain => panic!("must not abstain"),
        }
    }

    #[test]
    fn predicate_invention_rejects_redundant_and_reports_cost_vs_benefit() {
        let train = data(1, 1, 240, xor_rule);
        let hold = data(2, 10_000, 120, xor_rule);
        let r = inventor(InventionConfig::default())
            .invent(0, 0, &train, &hold)
            .unwrap()
            .report;
        let get = |op, args: &[u16]| {
            r.candidates
                .iter()
                .find(|c| c.predicate.op == op && c.predicate.args == args)
                .unwrap()
        };
        // Iff(b0,b1) is the complement of the accepted Xor(b0,b1): redundant.
        assert!(matches!(
            get(PredicateOp::Iff, &[0, 1]).verdict,
            CandidateVerdict::Redundant { .. }
        ));
        let x = get(PredicateOp::Xor, &[0, 1]);
        assert_eq!(x.verdict, CandidateVerdict::Accepted);
        assert!(x.benefit_bits > x.description_bits + 2.0);
        // distractor-only compound predicates do not pay for their description
        let d = get(PredicateOp::Xor, &[2, 3]);
        assert_eq!(d.verdict, CandidateVerdict::NoNetBenefit);
        assert!(d.benefit_bits < d.description_bits);
        // exactly one predicate accepted: nothing else adds net benefit once Xor is known
        let acc = r
            .candidates
            .iter()
            .filter(|c| c.verdict == CandidateVerdict::Accepted)
            .count();
        assert_eq!(acc, 1);
    }

    #[test]
    fn predicate_invention_limits_bound_search_and_report_is_reproducible() {
        let train = data(1, 1, 240, xor_rule);
        let hold = data(2, 10_000, 120, xor_rule);
        // size limit
        let small = InventionConfig {
            max_candidates: 5,
            ..InventionConfig::default()
        };
        let r = inventor(small).invent(0, 0, &train, &hold).unwrap().report;
        assert!(r.enumeration_truncated);
        assert_eq!(r.candidates.len(), 5);
        // arity limit: 6 ops x C(4,2) = 36 candidates, no arity-3
        let a2 = InventionConfig {
            max_arity: 2,
            ..InventionConfig::default()
        };
        let r = inventor(a2).invent(0, 0, &train, &hold).unwrap().report;
        assert_eq!(r.candidates.len(), 36);
        assert!(r.candidates.iter().all(|c| c.predicate.args.len() == 2));
        // work (time) budget
        let tiny = InventionConfig {
            max_evals: 3,
            ..InventionConfig::default()
        };
        let r = inventor(tiny).invent(0, 0, &train, &hold).unwrap().report;
        assert!(r.evals <= 3);
        assert!(
            r.candidates
                .iter()
                .any(|c| c.verdict == CandidateVerdict::BudgetExhausted)
        );
        // evidence limit: too little evidence refuses to invent
        let few = &train[..10];
        let r = inventor(InventionConfig::default())
            .invent(0, 0, few, &hold)
            .unwrap()
            .report;
        assert!(r.candidates.is_empty() && !r.adopted);
        let capped = InventionConfig {
            max_train: 50,
            ..InventionConfig::default()
        };
        let r = inventor(capped).invent(0, 0, &train, &hold).unwrap().report;
        assert_eq!(r.train_used, 50);
        // reproducibility
        let a = inventor(InventionConfig::default())
            .invent(0, 0, &train, &hold)
            .unwrap()
            .report
            .render();
        let b = inventor(InventionConfig::default())
            .invent(0, 0, &train, &hold)
            .unwrap()
            .report
            .render();
        assert_eq!(a, b);
    }

    #[test]
    fn predicate_invention_negative_pure_noise_is_not_adopted() {
        let mut s = 99u64;
        let noise = |_: &[bool]| lcg(&mut s) & 1 == 1;
        let train = data(1, 1, 240, noise);
        let mut s2 = 77u64;
        let noise2 = |_: &[bool]| lcg(&mut s2) & 1 == 1;
        let hold = data(2, 10_000, 120, noise2);
        let r = inventor(InventionConfig::default())
            .invent(0, 0, &train, &hold)
            .unwrap();
        println!("{}", r.report.render());
        assert!(!r.report.adopted);
        assert!(r.model.is_none());
    }
}
