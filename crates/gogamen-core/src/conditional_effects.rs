//! Budget-bounded conditional-effects learner over propositional macrostates (GOG-164 / DEM-011).
//!
//! First slice of the hypothesis space: for every bit of the next macrostate the learner keeps
//! *clauses* (conjunctions of at most [`ConditionalBudget::max_literals`] literals over the
//! current state bits and the chosen action) with Beta counts of "next bit is true". It generalizes
//! the tabular `(bucket, action) -> bucket` learner: a clause that omits irrelevant bits shares
//! evidence across contexts.
//!
//! # Guarantees
//! - **No live model, no goal rewards.** Inputs are [`ConditionalObservation`]s only.
//! - **Budget.** Literals per clause, clauses per bit and total updates are bounded; clauses
//!   refused by the budget are counted, never silently dropped.
//! - **Dedup + audit.** An observation whose id or evidence was already applied is ignored and
//!   leaves counts, version and checksum untouched. Every applied update appends an
//!   [`UpdateRecord`] (version, observation id, evidence ref).
//! - **Explicit failure.** Each update is first scored prequentially (Brier of the prediction
//!   made *before* seeing the outcome). A sliding window over those scores yields
//!   [`PredictionHealth::PredictionFailure`]; [`ConditionalEffectsLearner::gate`] then returns an
//!   error instead of any "safe" default action.
//!
//! This is a propositional rule learner; it claims nothing beyond the bits it is given.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::learning::ObservationId;
use crate::provenance::EvidenceRef;

/// Literal of a clause: a state bit with a polarity, or "the action is `a`".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Literal {
    /// State bit `index` equals `value`.
    Bit {
        /// Bit index in the state vector.
        index: u16,
        /// Required value.
        value: bool,
    },
    /// The chosen discrete action index equals this value.
    Action(u16),
}

/// Errors of the conditional-effects learner.
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionalError {
    /// A budget or config value is invalid (`name` identifies it).
    InvalidConfig(&'static str),
    /// Vector lengths or action index do not match the learner shape.
    ShapeMismatch(&'static str),
    /// The prediction-failure gate is closed; no action may be presented as safe.
    PredictionFailure {
        /// Mean prequential Brier over the window.
        mean_brier: f64,
        /// Configured failure threshold.
        threshold: f64,
    },
}

impl fmt::Display for ConditionalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(n) => write!(f, "invalid conditional-effects config `{n}`"),
            Self::ShapeMismatch(n) => write!(f, "shape mismatch: {n}"),
            Self::PredictionFailure {
                mean_brier,
                threshold,
            } => write!(
                f,
                "prediction failure: mean Brier {mean_brier:.4} > threshold {threshold:.4}"
            ),
        }
    }
}

impl Error for ConditionalError {}

/// Resource budget of the learner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalBudget {
    /// Maximum literals per clause (hypothesis complexity).
    pub max_literals: usize,
    /// Maximum tracked clauses per output bit.
    pub max_clauses_per_bit: usize,
    /// Maximum number of applied (non-duplicate) updates.
    pub max_updates: u64,
}

/// Prediction and failure-monitoring parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConditionalConfig {
    /// Beta pseudocount per outcome (`> 0`).
    pub alpha: f64,
    /// Minimum observations before a clause may drive a prediction (`>= 1`).
    pub min_evidence: u64,
    /// Sliding window (observations) of the failure monitor (`>= 1`).
    pub window: usize,
    /// Window mean Brier above which prediction failure is declared (`> 0`).
    pub failure_threshold: f64,
    /// Minimum scored observations in the window before failure may be declared (`>= 1`).
    pub min_window_samples: usize,
}

impl Default for ConditionalConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            min_evidence: 3,
            window: 40,
            failure_threshold: 0.12,
            min_window_samples: 20,
        }
    }
}

/// One observed transition of the propositional macrostate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConditionalObservation {
    /// Unique observation id (dedup key).
    pub id: ObservationId,
    /// Evidence reference (dedup key, recorded in the audit log).
    pub evidence: EvidenceRef,
    /// State bits before the action.
    pub state: Vec<bool>,
    /// Discrete action index.
    pub action: usize,
    /// State bits after the action.
    pub next: Vec<bool>,
}

/// Outcome of [`ConditionalEffectsLearner::observe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionalUpdateStatus {
    /// Counts and version were updated.
    Updated,
    /// Id or evidence already applied; nothing changed.
    DuplicateIgnored,
    /// `max_updates` reached; nothing changed.
    CapacityReached,
}

/// Audit entry of one applied update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateRecord {
    /// Model version produced by this update.
    pub version: u64,
    /// Observation applied.
    pub observation: ObservationId,
    /// Evidence supporting it.
    pub evidence: EvidenceRef,
    /// Prequential Brier of the prediction made before the update (`None` if no informed bit).
    pub prequential_brier: Option<f64>,
    /// Clauses this update could not admit because of the budget.
    pub budget_rejected: u32,
}

/// Prediction for one next-state bit.
#[derive(Debug, Clone, PartialEq)]
pub struct BitPrediction {
    /// `P(next bit = true)`.
    pub probability: f64,
    /// `false` when no clause had enough evidence (prior-only prediction).
    pub informed: bool,
    /// Clause that drove the prediction.
    pub clause: Option<Vec<Literal>>,
    /// Observations behind that clause.
    pub evidence: u64,
}

/// Health of the learner's own predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredictionHealth {
    /// Not enough scored observations to judge.
    Warmup,
    /// Window Brier is within threshold.
    Healthy {
        /// Mean prequential Brier in the window.
        mean_brier: f64,
    },
    /// Window Brier exceeds the threshold: dynamics changed or the hypothesis space is wrong.
    PredictionFailure {
        /// Mean prequential Brier in the window.
        mean_brier: f64,
    },
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Counts {
    n: u64,
    ones: u64,
}

/// Mutable learner. Freeze with [`ConditionalEffectsLearner::freeze`] for an immutable version.
#[derive(Debug, Clone)]
pub struct ConditionalEffectsLearner {
    num_bits: usize,
    num_actions: usize,
    budget: ConditionalBudget,
    config: ConditionalConfig,
    clauses: Vec<BTreeMap<Vec<Literal>, Counts>>,
    seen_ids: BTreeSet<ObservationId>,
    seen_evidence: BTreeSet<EvidenceRef>,
    version: u64,
    log: Vec<UpdateRecord>,
    window: VecDeque<f64>,
    duplicates_ignored: u64,
    budget_rejections: u64,
}

impl ConditionalEffectsLearner {
    /// Creates an empty learner over `num_bits` state bits and `num_actions` actions.
    pub fn new(
        num_bits: usize,
        num_actions: usize,
        budget: ConditionalBudget,
        config: ConditionalConfig,
    ) -> Result<Self, ConditionalError> {
        if num_bits == 0 || num_bits > u16::MAX as usize {
            return Err(ConditionalError::InvalidConfig("num_bits"));
        }
        if num_actions == 0 || num_actions > u16::MAX as usize {
            return Err(ConditionalError::InvalidConfig("num_actions"));
        }
        if budget.max_literals == 0 || budget.max_clauses_per_bit == 0 || budget.max_updates == 0 {
            return Err(ConditionalError::InvalidConfig("budget"));
        }
        if !config.alpha.is_finite() || config.alpha <= 0.0 {
            return Err(ConditionalError::InvalidConfig("alpha"));
        }
        if !config.failure_threshold.is_finite() || config.failure_threshold <= 0.0 {
            return Err(ConditionalError::InvalidConfig("failure_threshold"));
        }
        if config.min_evidence == 0 || config.window == 0 || config.min_window_samples == 0 {
            return Err(ConditionalError::InvalidConfig("config counts"));
        }
        Ok(Self {
            num_bits,
            num_actions,
            budget,
            config,
            clauses: vec![BTreeMap::new(); num_bits],
            seen_ids: BTreeSet::new(),
            seen_evidence: BTreeSet::new(),
            version: 0,
            log: Vec::new(),
            window: VecDeque::new(),
            duplicates_ignored: 0,
            budget_rejections: 0,
        })
    }

    /// Monotonic model version (number of applied updates).
    pub const fn version(&self) -> u64 {
        self.version
    }

    /// Audit log of applied updates, in order.
    pub fn log(&self) -> &[UpdateRecord] {
        &self.log
    }

    /// Duplicate observations ignored so far.
    pub const fn duplicates_ignored(&self) -> u64 {
        self.duplicates_ignored
    }

    /// Clauses refused by the budget so far.
    pub const fn budget_rejections(&self) -> u64 {
        self.budget_rejections
    }

    /// Number of tracked clauses for output bit `bit`.
    pub fn clause_count(&self, bit: usize) -> usize {
        self.clauses.get(bit).map_or(0, BTreeMap::len)
    }

    /// FNV-1a checksum over the clause counts and version (excludes log and dedup sets).
    pub fn checksum(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        eat(self.version);
        for (bit, map) in self.clauses.iter().enumerate() {
            eat(bit as u64);
            for (clause, c) in map {
                for lit in clause {
                    match *lit {
                        Literal::Bit { index, value } => {
                            eat(1 + (u64::from(index) << 2) + (u64::from(value) << 1));
                        }
                        Literal::Action(a) => eat(0x8000_0000 + u64::from(a)),
                    }
                }
                eat(c.n);
                eat(c.ones);
            }
        }
        h
    }

    fn check_shape(&self, state: &[bool], action: usize) -> Result<(), ConditionalError> {
        if state.len() != self.num_bits {
            return Err(ConditionalError::ShapeMismatch("state length"));
        }
        if action >= self.num_actions {
            return Err(ConditionalError::ShapeMismatch("action index"));
        }
        Ok(())
    }

    /// Literals true in `(state, action)`.
    fn firing_literals(state: &[bool], action: usize) -> Vec<Literal> {
        let mut lits: Vec<Literal> = state
            .iter()
            .enumerate()
            .map(|(i, &v)| Literal::Bit {
                index: i as u16,
                value: v,
            })
            .collect();
        lits.push(Literal::Action(action as u16));
        lits.sort();
        lits
    }

    /// All sub-conjunctions of the firing literals with `<= max_literals`, ascending size.
    fn firing_clauses(&self, state: &[bool], action: usize) -> Vec<Vec<Literal>> {
        let lits = Self::firing_literals(state, action);
        let mut out: Vec<Vec<Literal>> = vec![Vec::new()];
        let mut frontier: Vec<(usize, Vec<Literal>)> = vec![(0, Vec::new())];
        for _ in 0..self.budget.max_literals {
            let mut next_frontier = Vec::new();
            for (start, prefix) in &frontier {
                for (i, lit) in lits.iter().enumerate().skip(*start) {
                    let mut c = prefix.clone();
                    c.push(*lit);
                    out.push(c.clone());
                    next_frontier.push((i + 1, c));
                }
            }
            frontier = next_frontier;
        }
        out
    }

    /// Predicts `P(next bit = true)` for output bit `bit`.
    ///
    /// Among firing clauses with at least `min_evidence` observations, the one with the lowest
    /// posterior Bernoulli variance wins (ties: more evidence, then fewer literals). With none,
    /// the result is the uninformed prior, flagged `informed = false`.
    pub fn predict_bit(
        &self,
        state: &[bool],
        action: usize,
        bit: usize,
    ) -> Result<BitPrediction, ConditionalError> {
        self.check_shape(state, action)?;
        if bit >= self.num_bits {
            return Err(ConditionalError::ShapeMismatch("output bit"));
        }
        let a = self.config.alpha;
        let mut best: Option<(f64, u64, usize, Vec<Literal>, f64)> = None;
        for clause in self.firing_clauses(state, action) {
            let Some(c) = self.clauses[bit].get(&clause) else {
                continue;
            };
            if c.n < self.config.min_evidence {
                continue;
            }
            let p = (c.ones as f64 + a) / (c.n as f64 + 2.0 * a);
            let var = p * (1.0 - p);
            let better = match &best {
                None => true,
                Some((bv, bn, bl, _, _)) => {
                    var < *bv - 1e-12
                        || ((var - *bv).abs() <= 1e-12
                            && (c.n > *bn || (c.n == *bn && clause.len() < *bl)))
                }
            };
            if better {
                best = Some((var, c.n, clause.len(), clause, p));
            }
        }
        Ok(match best {
            Some((_, n, _, clause, p)) => BitPrediction {
                probability: p,
                informed: true,
                clause: Some(clause),
                evidence: n,
            },
            None => BitPrediction {
                probability: 0.5,
                informed: false,
                clause: None,
                evidence: 0,
            },
        })
    }

    /// Predicts all next-state bits.
    pub fn predict(
        &self,
        state: &[bool],
        action: usize,
    ) -> Result<Vec<BitPrediction>, ConditionalError> {
        (0..self.num_bits)
            .map(|b| self.predict_bit(state, action, b))
            .collect()
    }

    /// Mean Brier and mean log-loss of the learner on `holdout` (never updates the model).
    ///
    /// Returns `None` for an empty holdout.
    pub fn evaluate(
        &self,
        holdout: &[ConditionalObservation],
    ) -> Result<Option<HoldoutScores>, ConditionalError> {
        if holdout.is_empty() {
            return Ok(None);
        }
        let mut scorer = ScoreAcc::default();
        for o in holdout {
            for (b, p) in self.predict(&o.state, o.action)?.iter().enumerate() {
                scorer.add(p.probability, o.next[b]);
            }
        }
        Ok(Some(scorer.finish()))
    }

    /// Applies one observation: dedup, prequential scoring, clause update, audit record.
    pub fn observe(
        &mut self,
        obs: &ConditionalObservation,
    ) -> Result<ConditionalUpdateStatus, ConditionalError> {
        self.check_shape(&obs.state, obs.action)?;
        if obs.next.len() != self.num_bits {
            return Err(ConditionalError::ShapeMismatch("next length"));
        }
        if self.seen_ids.contains(&obs.id) || self.seen_evidence.contains(&obs.evidence) {
            self.duplicates_ignored += 1;
            return Ok(ConditionalUpdateStatus::DuplicateIgnored);
        }
        if self.version >= self.budget.max_updates {
            return Ok(ConditionalUpdateStatus::CapacityReached);
        }
        // Prequential score: prediction before learning from this outcome.
        let mut acc = ScoreAcc::default();
        for (b, p) in self.predict(&obs.state, obs.action)?.iter().enumerate() {
            if p.informed {
                acc.add(p.probability, obs.next[b]);
            }
        }
        let preq = (acc.n > 0).then(|| acc.finish().brier);

        let clauses = self.firing_clauses(&obs.state, obs.action);
        let mut rejected = 0u32;
        for bit in 0..self.num_bits {
            for clause in &clauses {
                let map = &mut self.clauses[bit];
                if !map.contains_key(clause) && map.len() >= self.budget.max_clauses_per_bit {
                    rejected += 1;
                    continue;
                }
                let c = map.entry(clause.clone()).or_default();
                c.n += 1;
                c.ones += u64::from(obs.next[bit]);
            }
        }
        self.budget_rejections += u64::from(rejected);
        self.seen_ids.insert(obs.id);
        self.seen_evidence.insert(obs.evidence);
        self.version += 1;
        if let Some(b) = preq {
            self.window.push_back(b);
            while self.window.len() > self.config.window {
                self.window.pop_front();
            }
        }
        self.log.push(UpdateRecord {
            version: self.version,
            observation: obs.id,
            evidence: obs.evidence,
            prequential_brier: preq,
            budget_rejected: rejected,
        });
        Ok(ConditionalUpdateStatus::Updated)
    }

    /// Current health of the learner's own predictions.
    pub fn health(&self) -> PredictionHealth {
        if self.window.len() < self.config.min_window_samples {
            return PredictionHealth::Warmup;
        }
        let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
        if mean > self.config.failure_threshold {
            PredictionHealth::PredictionFailure { mean_brier: mean }
        } else {
            PredictionHealth::Healthy { mean_brier: mean }
        }
    }

    /// Fails explicitly under prediction failure; never substitutes a "safe" action.
    pub fn gate(&self) -> Result<(), ConditionalError> {
        match self.health() {
            PredictionHealth::PredictionFailure { mean_brier } => {
                Err(ConditionalError::PredictionFailure {
                    mean_brier,
                    threshold: self.config.failure_threshold,
                })
            }
            _ => Ok(()),
        }
    }

    /// Freezes an immutable copy for planning/evaluation.
    pub fn freeze(&self) -> Arc<ConditionalModelVersion> {
        Arc::new(ConditionalModelVersion {
            learner: self.clone(),
            checksum: self.checksum(),
        })
    }
}

/// Immutable frozen revision of a [`ConditionalEffectsLearner`].
#[derive(Debug, Clone)]
pub struct ConditionalModelVersion {
    learner: ConditionalEffectsLearner,
    checksum: u64,
}

impl ConditionalModelVersion {
    /// Version captured.
    pub const fn revision(&self) -> u64 {
        self.learner.version
    }

    /// Checksum captured.
    pub const fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Read-only prediction for all next bits.
    pub fn predict(
        &self,
        state: &[bool],
        action: usize,
    ) -> Result<Vec<BitPrediction>, ConditionalError> {
        self.learner.predict(state, action)
    }
}

/// Aggregate scores over holdout bit predictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HoldoutScores {
    /// Mean squared error of probabilities.
    pub brier: f64,
    /// Mean negative log-likelihood (nats), probabilities clamped to `[1e-12, 1 - 1e-12]`.
    pub log_loss: f64,
    /// Number of bit predictions scored.
    pub n: usize,
}

#[derive(Debug, Default)]
struct ScoreAcc {
    brier: f64,
    ll: f64,
    n: usize,
}

impl ScoreAcc {
    fn add(&mut self, p: f64, y: bool) {
        let t = if y { 1.0 } else { 0.0 };
        self.brier += (p - t) * (p - t);
        let q = p.clamp(1e-12, 1.0 - 1e-12);
        self.ll -= if y { q.ln() } else { (1.0 - q).ln() };
        self.n += 1;
    }

    fn finish(&self) -> HoldoutScores {
        let n = self.n.max(1) as f64;
        HoldoutScores {
            brier: self.brier / n,
            log_loss: self.ll / n,
            n: self.n,
        }
    }
}

/// Scores of the uninformed prior (`p = 0.5` for every bit) on `holdout`.
pub fn prior_scores(holdout: &[ConditionalObservation]) -> Option<HoldoutScores> {
    if holdout.is_empty() {
        return None;
    }
    let mut acc = ScoreAcc::default();
    for o in holdout {
        for &y in &o.next {
            acc.add(0.5, y);
        }
    }
    Some(acc.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn learner(budget: ConditionalBudget) -> ConditionalEffectsLearner {
        ConditionalEffectsLearner::new(2, 2, budget, ConditionalConfig::default()).unwrap()
    }

    fn budget() -> ConditionalBudget {
        ConditionalBudget {
            max_literals: 2,
            max_clauses_per_bit: 200,
            max_updates: 1000,
        }
    }

    fn obs(i: u64, s: [bool; 2], a: usize, n: [bool; 2]) -> ConditionalObservation {
        ConditionalObservation {
            id: ObservationId::new(i),
            evidence: EvidenceRef::new(1000 + i),
            state: s.to_vec(),
            action: a,
            next: n.to_vec(),
        }
    }

    #[test]
    fn conditional_effects_core_dedup_and_audit() {
        let mut l = learner(budget());
        let o = obs(1, [true, false], 1, [false, false]);
        assert_eq!(l.observe(&o).unwrap(), ConditionalUpdateStatus::Updated);
        let (v, c) = (l.version(), l.checksum());
        assert_eq!(
            l.observe(&o).unwrap(),
            ConditionalUpdateStatus::DuplicateIgnored
        );
        // Same evidence, different id: also a duplicate.
        let mut o2 = o.clone();
        o2.id = ObservationId::new(2);
        assert_eq!(
            l.observe(&o2).unwrap(),
            ConditionalUpdateStatus::DuplicateIgnored
        );
        assert_eq!((l.version(), l.checksum()), (v, c));
        assert_eq!(l.log().len(), 1);
        assert_eq!(l.log()[0].version, 1);
        assert_eq!(l.log()[0].evidence, EvidenceRef::new(1001));
        assert_eq!(l.duplicates_ignored(), 2);
    }

    #[test]
    fn conditional_effects_core_learns_irrelevant_bit_generalization() {
        // next[0] = !state[0] when action 1, independent of state[1].
        let mut l = learner(budget());
        for i in 0..40u64 {
            let s0 = i % 2 == 0;
            let s1 = (i / 2) % 2 == 0;
            l.observe(&obs(i, [s0, s1], 1, [!s0, s1])).unwrap();
        }
        let p = l.predict_bit(&[true, true], 1, 0).unwrap();
        assert!(p.informed && p.probability < 0.1, "{p:?}");
    }

    #[test]
    fn conditional_effects_core_budget_bounds_clauses() {
        let mut l = learner(ConditionalBudget {
            max_clauses_per_bit: 5,
            ..budget()
        });
        l.observe(&obs(1, [true, false], 1, [false, false]))
            .unwrap();
        assert!(l.clause_count(0) <= 5);
        assert!(l.budget_rejections() > 0);
        assert!(l.log()[0].budget_rejected > 0);
    }

    #[test]
    fn conditional_effects_core_capacity_and_shape() {
        let mut l = learner(ConditionalBudget {
            max_updates: 1,
            ..budget()
        });
        l.observe(&obs(1, [true, false], 1, [false, false]))
            .unwrap();
        assert_eq!(
            l.observe(&obs(2, [true, false], 1, [false, false]))
                .unwrap(),
            ConditionalUpdateStatus::CapacityReached
        );
        assert!(l.predict_bit(&[true], 0, 0).is_err());
    }
}
