//! Decisive moments and predicates born from bifurcations (GOG-181, DEM-032).
//!
//! # What it does
//! 1. **Viability spread.** The FMC planner publishes, per decision and *before cloning*, the
//!    viable fraction of walkers per root action ([`crate::FmcStepRootViability`]). The spread of
//!    a decision is `max - min` of those fractions over the root actions that have walkers.
//! 2. **Decisive marks.** [`mark_decisive`] flags the top-`q` decisions by spread (ties broken by
//!    decision order, spread `0` is never decisive). It is a pure function of the spreads, so the
//!    same seed gives the same marks.
//! 3. **Predicates nobody gave the agent.** [`fit_pair_predicates`] fits, for every pair of
//!    actions, one threshold on one sense (state dimension) that separates "planner chose `a`"
//!    from "planner chose `b`", using the decisive states. Candidate thresholds are either born
//!    from the decisive states ([`ThresholdSource::Decisive`]) or a uniform mesh of the same size
//!    ([`ThresholdSource::UniformMesh`]); only the placement of candidates differs.
//! 4. **Export.** [`DecisiveTimeline::events`] is a plain data export (tick, spread, mark,
//!    optional control bits) that a viewer can consume without touching the simulation.
//!
//! No type here grants permissions: everything is read-only data over already-simulated plans.

use std::error::Error;
use std::fmt;

use crate::fmc_planner::FmcStepRootViability;

/// Errors of the decisive-moment helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum DecisiveError {
    /// `top_quantile` must lie in `(0, 1]`.
    InvalidQuantile(f64),
    /// A spread was not finite.
    NonFiniteSpread { index: usize },
    /// A sample had a different number of senses than the first one.
    SenseCountMismatch { expected: usize, actual: usize },
    /// The candidate count per sense must be `>= 1`.
    InvalidCandidateCount,
}

impl fmt::Display for DecisiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuantile(q) => write!(f, "top quantile {q} must be in (0, 1]"),
            Self::NonFiniteSpread { index } => write!(f, "non-finite spread at decision {index}"),
            Self::SenseCountMismatch { expected, actual } => {
                write!(f, "expected {expected} senses, found {actual}")
            }
            Self::InvalidCandidateCount => write!(f, "candidates per sense must be >= 1"),
        }
    }
}

impl Error for DecisiveError {}

/// `max - min` of the viable fractions over root actions with walkers (`0.0` with < 2 roots).
pub fn viability_spread(step: &FmcStepRootViability) -> f64 {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut n = 0usize;
    for r in step.roots.iter().filter(|r| r.walkers > 0) {
        let f = r.fraction();
        min = min.min(f);
        max = max.max(f);
        n += 1;
    }
    if n < 2 { 0.0 } else { max - min }
}

/// Marks the top `top_quantile` share of decisions by spread: exactly `ceil(q * n)` decisions
/// (fewer if some have spread `0`, which is never decisive). Ties are broken by lower index.
pub fn mark_decisive(spreads: &[f64], top_quantile: f64) -> Result<Vec<bool>, DecisiveError> {
    if !(top_quantile.is_finite() && top_quantile > 0.0 && top_quantile <= 1.0) {
        return Err(DecisiveError::InvalidQuantile(top_quantile));
    }
    if let Some(index) = spreads.iter().position(|s| !s.is_finite()) {
        return Err(DecisiveError::NonFiniteSpread { index });
    }
    let n = spreads.len();
    let k = ((top_quantile * n as f64).ceil() as usize).min(n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| spreads[b].total_cmp(&spreads[a]).then(a.cmp(&b)));
    let mut marks = vec![false; n];
    for &i in order.iter().take(k) {
        if spreads[i] > 0.0 {
            marks[i] = true;
        }
    }
    Ok(marks)
}

/// One planner decision, as recorded by the caller (pure data).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionInput {
    /// Tick of the decision.
    pub tick: u64,
    /// State values (the senses) the planner decided from.
    pub state: Vec<f64>,
    /// Index of the action the planner chose.
    pub chosen: usize,
    /// Pre-cloning viability of the decision (`FmcTraceSummary::decision_viability`).
    pub viability: FmcStepRootViability,
    /// Optional control bits of the decision (`ControlBitsEstimate::control_corrected`).
    pub control_bits: Option<f64>,
}

/// A decision with its spread and decisive mark.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionSample {
    /// The recorded decision.
    pub input: DecisionInput,
    /// [`viability_spread`] of the decision.
    pub spread: f64,
    /// Whether the decision is in the declared top quantile.
    pub decisive: bool,
}

/// Plain exportable event for a timeline viewer (no live references).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisiveEvent {
    /// Tick of the decision.
    pub tick: u64,
    /// Viability spread between root actions.
    pub spread: f64,
    /// Decisive mark.
    pub decisive: bool,
    /// Control bits at the decision, if estimated.
    pub control_bits: Option<f64>,
}

/// Decisions of one seed set with their decisive marks.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisiveTimeline {
    top_quantile: f64,
    samples: Vec<DecisionSample>,
}

impl DecisiveTimeline {
    /// Computes spreads and marks the top `top_quantile` over **all** given decisions.
    pub fn build(inputs: Vec<DecisionInput>, top_quantile: f64) -> Result<Self, DecisiveError> {
        let spreads: Vec<f64> = inputs
            .iter()
            .map(|d| viability_spread(&d.viability))
            .collect();
        let marks = mark_decisive(&spreads, top_quantile)?;
        let samples = inputs
            .into_iter()
            .zip(spreads)
            .zip(marks)
            .map(|((input, spread), decisive)| DecisionSample {
                input,
                spread,
                decisive,
            })
            .collect();
        Ok(Self {
            top_quantile,
            samples,
        })
    }

    /// Declared quantile.
    pub const fn top_quantile(&self) -> f64 {
        self.top_quantile
    }

    /// All decisions in input order.
    pub fn samples(&self) -> &[DecisionSample] {
        &self.samples
    }

    /// Only the decisive decisions.
    pub fn decisive_samples(&self) -> impl Iterator<Item = &DecisionSample> {
        self.samples.iter().filter(|s| s.decisive)
    }

    /// Plain data export consumed by viewers (Studio timeline).
    pub fn events(&self) -> Vec<DecisiveEvent> {
        self.samples
            .iter()
            .map(|s| DecisiveEvent {
                tick: s.input.tick,
                spread: s.spread,
                decisive: s.decisive,
                control_bits: s.input.control_bits,
            })
            .collect()
    }
}

/// Where candidate thresholds come from. Both give the same number per sense.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThresholdSource {
    /// `m` quantile points of the sense values at decisive states.
    Decisive,
    /// `m` equally spaced points strictly inside the range of the sense over all states.
    UniformMesh,
}

/// One threshold on one sense for a pair of actions `(a, b)` with `a < b`.
///
/// Predicts `b` when `(state[sense] > threshold) == greater_picks_b`, else `a`.
#[derive(Debug, Clone, PartialEq)]
pub struct PairPredicate {
    /// Lower action index of the pair.
    pub a: usize,
    /// Higher action index of the pair.
    pub b: usize,
    /// Sense (state dimension) tested.
    pub sense: usize,
    /// Threshold value.
    pub threshold: f64,
    /// Orientation of the test.
    pub greater_picks_b: bool,
    /// Accuracy on the fitting samples.
    pub fit_accuracy: f64,
    /// Number of fitting samples.
    pub fit_samples: usize,
}

impl PairPredicate {
    /// Predicted action index for `state`.
    pub fn predict(&self, state: &[f64]) -> usize {
        let above = state.get(self.sense).is_some_and(|v| *v > self.threshold);
        if above == self.greater_picks_b {
            self.b
        } else {
            self.a
        }
    }
}

fn senses_of(samples: &[&DecisionSample]) -> Result<usize, DecisiveError> {
    let Some(first) = samples.first() else {
        return Ok(0);
    };
    let n = first.input.state.len();
    for s in samples {
        if s.input.state.len() != n {
            return Err(DecisiveError::SenseCountMismatch {
                expected: n,
                actual: s.input.state.len(),
            });
        }
    }
    Ok(n)
}

fn candidates(
    source: ThresholdSource,
    sense: usize,
    m: usize,
    decisive: &[&DecisionSample],
    all: &[&DecisionSample],
) -> Vec<f64> {
    match source {
        ThresholdSource::Decisive => {
            let mut v: Vec<f64> = decisive.iter().map(|s| s.input.state[sense]).collect();
            v.sort_by(f64::total_cmp);
            if v.is_empty() {
                return Vec::new();
            }
            (0..m)
                .map(|i| {
                    let idx = (((i as f64 + 0.5) / m as f64) * v.len() as f64) as usize;
                    v[idx.min(v.len() - 1)]
                })
                .collect()
        }
        ThresholdSource::UniformMesh => {
            let lo = all
                .iter()
                .map(|s| s.input.state[sense])
                .fold(f64::INFINITY, f64::min);
            let hi = all
                .iter()
                .map(|s| s.input.state[sense])
                .fold(f64::NEG_INFINITY, f64::max);
            if !(lo.is_finite() && hi.is_finite()) {
                return Vec::new();
            }
            (0..m)
                .map(|i| lo + (hi - lo) * (i as f64 + 1.0) / (m as f64 + 1.0))
                .collect()
        }
    }
}

/// Fits one [`PairPredicate`] per pair of chosen actions.
///
/// * `all` are the train decisions (the mesh range comes from them);
/// * fitting samples of a pair are the **decisive** decisions whose `chosen` is `a` or `b`;
///   pairs with fewer than `min_samples` of them, or with a single class, get no predicate;
/// * the best `(sense, threshold, orientation)` by fitting accuracy wins; ties are broken by
///   lower sense, lower threshold, `greater_picks_b = true` first (deterministic).
pub fn fit_pair_predicates(
    all: &[&DecisionSample],
    source: ThresholdSource,
    candidates_per_sense: usize,
    min_samples: usize,
) -> Result<Vec<PairPredicate>, DecisiveError> {
    if candidates_per_sense == 0 {
        return Err(DecisiveError::InvalidCandidateCount);
    }
    let senses = senses_of(all)?;
    let decisive: Vec<&DecisionSample> = all.iter().copied().filter(|s| s.decisive).collect();
    let mut actions: Vec<usize> = all.iter().map(|s| s.input.chosen).collect();
    actions.sort_unstable();
    actions.dedup();

    let cand: Vec<Vec<f64>> = (0..senses)
        .map(|d| candidates(source, d, candidates_per_sense, &decisive, all))
        .collect();

    let mut out = Vec::new();
    for (i, &a) in actions.iter().enumerate() {
        for &b in &actions[i + 1..] {
            let fit: Vec<&DecisionSample> = decisive
                .iter()
                .copied()
                .filter(|s| s.input.chosen == a || s.input.chosen == b)
                .collect();
            let has_a = fit.iter().any(|s| s.input.chosen == a);
            let has_b = fit.iter().any(|s| s.input.chosen == b);
            if fit.len() < min_samples || !has_a || !has_b {
                continue;
            }
            let mut best: Option<PairPredicate> = None;
            for (sense, thresholds) in cand.iter().enumerate() {
                for &threshold in thresholds {
                    for greater_picks_b in [true, false] {
                        let p = PairPredicate {
                            a,
                            b,
                            sense,
                            threshold,
                            greater_picks_b,
                            fit_accuracy: 0.0,
                            fit_samples: fit.len(),
                        };
                        let hits = fit
                            .iter()
                            .filter(|s| p.predict(&s.input.state) == s.input.chosen)
                            .count();
                        let acc = hits as f64 / fit.len() as f64;
                        if best.as_ref().is_none_or(|b| acc > b.fit_accuracy) {
                            best = Some(PairPredicate {
                                fit_accuracy: acc,
                                ..p
                            });
                        }
                    }
                }
            }
            out.extend(best);
        }
    }
    Ok(out)
}

/// Accuracy of `predicates` on `samples`: for every sample whose `chosen` belongs to a fitted
/// pair, the pair's predicate predicts the choice. Returns `(hits, total)`.
pub fn predicate_accuracy(
    predicates: &[PairPredicate],
    samples: &[&DecisionSample],
) -> (usize, usize) {
    let mut hits = 0;
    let mut total = 0;
    for s in samples {
        for p in predicates {
            if s.input.chosen == p.a || s.input.chosen == p.b {
                total += 1;
                hits += usize::from(p.predict(&s.input.state) == s.input.chosen);
            }
        }
    }
    (hits, total)
}

#[cfg(test)]
mod tests {
    //! Experiment protocol, declared BEFORE looking at Holdout results:
    //! * decisive quantile `q = 0.25`, `M = 4` candidate thresholds per sense for both sources
    //!   (a small budget on purpose: "same number of thresholds"), `min_samples = 4`;
    //! * Train seeds `0..40`, Holdout seeds `1000..1040`; marks are computed per seed set;
    //! * metric: accuracy of the pair predicates on **all** Holdout decisions whose choice
    //!   belongs to the pair (see [`predicate_accuracy`]); the verdict is reported either way.
    use super::*;
    use crate::action::{Action, ActionSpace, DiscreteAction};
    use crate::control_bits::{ControlBitsConfig, control_bits_from_walkers};
    use crate::determinism::MasterSeed;
    use crate::domain::{Observation, State, Terminal, TerminalReason, Transition};
    use crate::entropy::{BinSpec, BinningVersion};
    use crate::error::ModelError;
    use crate::fmc_planner::{FmcPlanner, FmcPlannerConfig};
    use crate::fmc_scoring::FmcScoringConfig;
    use crate::model::{DeterministicRng, GenerativeModel, PlanningContext, RngSource};
    use crate::planner::{PlanBudget, PlanRequest};
    use crate::units::{SimDuration, StepDt, Tick};

    const Q: f64 = 0.25;
    const M: usize = 4;
    const MIN_SAMPLES: usize = 4;
    const HAZARD: i64 = 2;
    const GOAL_LEFT: i64 = -6;
    const GOAL_RIGHT: i64 = 7;

    /// 1-D line with a hazard at `x = 2` and goals at both ends. State = `[x, y]`, `y` is a
    /// constant distractor sense. Right of the hazard the planner must go right, left of it left.
    struct HazardLine {
        space: ActionSpace,
    }

    impl GenerativeModel for HazardLine {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            let x = state.values()[0] as i64;
            if x == HAZARD || x == GOAL_LEFT || x == GOAL_RIGHT {
                return Err(ModelError::TerminalState);
            }
            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            let nx = if idx == 1 { x + 1 } else { x - 1 };
            let terminal = if nx == HAZARD {
                Terminal::Terminated(TerminalReason::Failure)
            } else if nx == GOAL_LEFT || nx == GOAL_RIGHT {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else {
                Terminal::Ongoing
            };
            let vals = vec![nx as f64, state.values()[1]];
            let next = state
                .advance(dt, vals.clone())
                .map_err(ModelError::TransitionFailed)?;
            let obs = Observation::new(next.tick(), next.elapsed(), vals)
                .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    fn planner(seed: u64, capture: bool) -> FmcPlanner {
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let cfg = FmcPlannerConfig::new(
            48,
            8,
            dt,
            FmcScoringConfig::default(),
            MasterSeed::new(seed),
        )
        .unwrap();
        FmcPlanner::new(cfg).with_walker_capture(capture)
    }

    fn episode(seed: u64, capture: bool) -> Vec<DecisionInput> {
        let model = HazardLine {
            space: ActionSpace::discrete(2).unwrap(),
        };
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let mut rng = DeterministicRng::new(seed ^ 0x5EED);
        let starts: Vec<i64> = (-5..=6).filter(|x| *x != HAZARD).collect();
        let x0 = starts[(rng.next_u64() % starts.len() as u64) as usize];
        let y = (rng.next_u64() % 1000) as f64 / 1000.0;
        let mut state = State::new(Tick::ZERO, SimDuration::ZERO, vec![x0 as f64, y]).unwrap();
        let scale = BinSpec::uniform(
            "hazard-line",
            BinningVersion::new(1, 0),
            vec![(-8.0, 8.0, 16), (0.0, 1.0, 1)],
        )
        .unwrap();
        let cb_cfg = ControlBitsConfig {
            resamples: 20,
            ..ControlBitsConfig::default()
        };
        let mut out = Vec::new();
        for _ in 0..10 {
            let p = planner(seed, capture);
            let mut ctx = PlanningContext::new(&model, &mut rng);
            let request = PlanRequest::from_state(
                state.clone(),
                model.action_space().clone(),
                PlanBudget::with_transitions(100_000),
            );
            let (result, trace) = p.plan_with_context(&mut ctx, &request).unwrap();
            let Some(action) = result.authorized_action().cloned() else {
                break;
            };
            let chosen = action.as_discrete().unwrap().index();
            let control_bits = trace.final_walkers().and_then(|w| {
                let (a, s): (Vec<_>, Vec<_>) = w.iter().cloned().unzip();
                control_bits_from_walkers(&a, &s, &scale, &cb_cfg)
                    .ok()
                    .map(|e| e.control_corrected)
            });
            out.push(DecisionInput {
                tick: state.tick().as_u64(),
                state: state.values().to_vec(),
                chosen,
                viability: trace.decision_viability().cloned().expect("viability"),
                control_bits,
            });
            match model.step(&state, &action, dt, &mut rng) {
                Ok(t) => {
                    let done = !matches!(t.terminal(), Terminal::Ongoing);
                    state = t.into_next_state();
                    if done {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
        out
    }

    fn collect(seeds: std::ops::Range<u64>) -> Vec<DecisionInput> {
        seeds.flat_map(|s| episode(s, false)).collect()
    }

    #[test]
    fn decisive_moments_mark_is_top_quantile_and_reproducible() {
        let spreads = [0.0, 0.5, 0.5, 1.0, 0.2, 0.0, 0.9, 0.1];
        let marks = mark_decisive(&spreads, 0.25).unwrap();
        // ceil(0.25 * 8) = 2: the two largest spreads (1.0 and 0.9).
        assert_eq!(
            marks,
            vec![false, false, false, true, false, false, true, false]
        );
        // Ties: lower index wins; zero spread never decisive.
        let marks = mark_decisive(&[0.5, 0.5, 0.0, 0.0], 0.5).unwrap();
        assert_eq!(marks, vec![true, true, false, false]);
        assert!(mark_decisive(&[1.0], 0.0).is_err());
        assert!(mark_decisive(&[f64::NAN], 0.5).is_err());

        let a = DecisiveTimeline::build(collect(0..6), Q).unwrap();
        let b = DecisiveTimeline::build(collect(0..6), Q).unwrap();
        assert_eq!(a, b, "same seeds must give the same marks");
        assert!(a.decisive_samples().count() > 0);
        assert!(a.samples().iter().any(|s| s.spread > 0.0));
    }

    #[test]
    fn decisive_moments_publish_pre_clone_viability_without_changing_plan() {
        let model = HazardLine {
            space: ActionSpace::discrete(2).unwrap(),
        };
        let state = State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 0.5]).unwrap();
        let request = PlanRequest::from_state(
            state,
            model.action_space().clone(),
            PlanBudget::with_transitions(100_000),
        );
        let run = |capture: bool| {
            let mut rng = DeterministicRng::new(3);
            let mut ctx = PlanningContext::new(&model, &mut rng);
            planner(11, capture)
                .plan_with_context(&mut ctx, &request)
                .unwrap()
        };
        let (r0, t0) = run(false);
        let (r1, t1) = run(true);
        assert_eq!(r0, r1);
        assert_eq!(t0.root_action_stats(), t1.root_action_stats());
        assert_eq!(t0.cloning_events(), t1.cloning_events());
        assert!(t0.final_walkers().is_none() && t1.final_walkers().is_some());
        assert_eq!(t0.pre_clone_viability(), t1.pre_clone_viability());

        let steps = t0.pre_clone_viability();
        assert!(!steps.is_empty());
        let first = &steps[0];
        assert_eq!(first.step, 0);
        // Step 0 is before any cloning: every walker is counted once, with its root action.
        assert_eq!(first.roots.iter().map(|r| r.walkers).sum::<usize>(), 48);
        // At x = 1 moving right (+1) lands on the hazard: that root has no viable walker.
        let right = first
            .roots
            .iter()
            .find(|r| r.action.as_discrete().map(DiscreteAction::index) == Some(1))
            .unwrap();
        assert_eq!(right.viable, 0);
        assert_eq!(viability_spread(first), 1.0);
    }

    #[test]
    fn decisive_moments_control_bits_ride_the_timeline_events() {
        let inputs = episode(5, true);
        assert!(inputs.iter().any(|d| d.control_bits.is_some()));
        let timeline = DecisiveTimeline::build(inputs, Q).unwrap();
        let events = timeline.events();
        assert_eq!(events.len(), timeline.samples().len());
        assert!(events.iter().all(|e| e.control_bits.is_some()));
    }

    #[test]
    fn decisive_moments_holdout_predicates_vs_uniform_mesh() {
        let train = DecisiveTimeline::build(collect(0..40), Q).unwrap();
        let hold = DecisiveTimeline::build(collect(1000..1040), Q).unwrap();
        let tr: Vec<&DecisionSample> = train.samples().iter().collect();
        let ho: Vec<&DecisionSample> = hold.samples().iter().collect();

        let born = fit_pair_predicates(&tr, ThresholdSource::Decisive, M, MIN_SAMPLES).unwrap();
        let mesh = fit_pair_predicates(&tr, ThresholdSource::UniformMesh, M, MIN_SAMPLES).unwrap();
        assert!(!born.is_empty() && !mesh.is_empty());
        for p in born.iter().chain(&mesh) {
            assert!(p.a < p.b && p.sense < 2);
        }

        let (bh, bt) = predicate_accuracy(&born, &ho);
        let (mh, mt) = predicate_accuracy(&mesh, &ho);
        assert_eq!(bt, mt, "same pairs, same evaluation set");
        let (born_acc, mesh_acc) = (bh as f64 / bt as f64, mh as f64 / mt as f64);
        eprintln!(
            "HOLDOUT decisions={} decisive={} born={born_acc:.4} ({bh}/{bt}) mesh={mesh_acc:.4} ({mh}/{mt}) born={born:?} mesh={mesh:?}",
            ho.len(),
            hold.decisive_samples().count()
        );
        // Honest reporting: the comparison is printed above; only structural facts are asserted.
        assert!(born_acc.is_finite() && mesh_acc.is_finite());
        // Reproducible end to end.
        let again = fit_pair_predicates(&tr, ThresholdSource::Decisive, M, MIN_SAMPLES).unwrap();
        assert_eq!(born, again);
    }
}
