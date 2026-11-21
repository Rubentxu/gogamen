//! Learned tabular generative model over a frozen, immutable model version (`std`-only).
//!
//! [`TabularLearnedModel`] implements [`GenerativeModel`] by sampling from a
//! [`ModelVersion`]: an immutable snapshot of a [`TabularTransitionLearner`] plus the
//! [`TabularScale`] that maps continuous [`State`] vectors onto discrete buckets.
//!
//! # Design
//! - **Off the critical path learning.** Learning happens on a mutable
//!   [`TabularTransitionLearner`] owned by the caller. The adapter only ever reads a
//!   [`ModelVersion`] behind an [`Arc`]: it has no access to the learner, so a model revision
//!   published later cannot change a decision already in flight.
//! - **Absorbing failure bucket.** The scale reserves one extra bucket (index
//!   [`TabularScale::failure_bucket`]) as a learned, absorbing destination. A transition
//!   recorded with [`TransitionSample::failure`] lands there and the adapter turns a sampled
//!   failure bucket into `Terminal::Terminated(TerminalReason::Failure)`. The failure bucket has
//!   its own Dirichlet pseudocount ([`LearnedModelConfig::failure_pseudocount`]) so unseen pairs
//!   keep a small, explicit failure probability instead of an invented safe prediction.
//! - **Explicit uncertainty.** Pairs without evidence are reported through
//!   [`BucketPrediction::uncertain`] / [`LearnedStepReport::uncertain`] and can be rejected with
//!   [`UnseenPairPolicy::Reject`]. By default they are sampled from the prior (high entropy).
//! - **Causal rules** (`causal_rule`) are intentionally not implemented here; they will plug in
//!   behind the same [`GenerativeModel`] trait.
//!
//! # Scale contract (GOG-200)
//! [`TabularScale`] is built on the typed [`crate::Scale`] (identifier,
//! [`crate::BinningVersion`] and
//! canonical hash). [`TabularScale::from_scale`] shares one contract with the planner and
//! [`plan_fmc_on_version`] records its identity in the decision trace, so the learner's buckets
//! and the planner's projection cannot diverge through separately rebuilt bin parameters.
//!
//! # Planner isolation
//! [`plan_fmc_on_version`] accepts only a [`TabularLearnedModel`]; a live environment or any
//! other simulator is rejected at compile time:
//!
//! ```no_run
//! use gogamen_core::*;
//! use std::sync::Arc;
//! fn ok(model: &TabularLearnedModel, planner: &FmcPlanner, request: &PlanRequest) {
//!     let mut rng = DeterministicRng::new(1);
//!     let _ = plan_fmc_on_version(planner, model, request, &mut rng, |_, _, _| 0.0);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! struct Env { space: ActionSpace }
//! impl EnvironmentObserver for Env {
//!     fn observe(&self) -> Result<Observation, EnvironmentError> { unimplemented!() }
//! }
//! impl Actuator for Env {
//!     fn action_space(&self) -> &ActionSpace { &self.space }
//!     fn apply(&mut self, _: &Action) -> Result<ApplyOutcome, EnvironmentError> { unimplemented!() }
//! }
//! impl LiveEnvironment for Env {
//!     fn reset(&mut self) -> Result<Observation, EnvironmentError> { unimplemented!() }
//! }
//! fn bad(env: &Env, planner: &FmcPlanner, request: &PlanRequest) {
//!     let mut rng = DeterministicRng::new(1);
//!     let _ = plan_fmc_on_version(planner, env, request, &mut rng, |_, _, _| 0.0);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! // Any other simulator (a grid world, a kart model...) is not a versioned learned model.
//! struct OtherSim { space: ActionSpace }
//! impl GenerativeModel for OtherSim {
//!     fn action_space(&self) -> &ActionSpace { &self.space }
//!     fn step(&self, _: &State, _: &Action, _: StepDt, _: &mut dyn RngSource)
//!         -> Result<Transition, ModelError> { unimplemented!() }
//! }
//! fn bad(sim: &OtherSim, planner: &FmcPlanner, request: &PlanRequest) {
//!     let mut rng = DeterministicRng::new(1);
//!     let _ = plan_fmc_on_version(planner, sim, request, &mut rng, |_, _, _| 0.0);
//! }
//! ```

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::action::{Action, ActionSpace};
use crate::domain::{Observation, State, Terminal, TerminalReason, Transition};
use crate::entropy::{BinAxisSpec, BinSpec, EntropyError};
use crate::error::ModelError;
use crate::fmc_planner::{FmcPlanner, FmcTraceSummary};
use crate::fmc_scoring::is_failure_terminal;
use crate::learning::{LearningError, TabularTransitionLearner, TransitionSample};
use crate::model::{GenerativeModel, PlanningContext, RngSource};
use crate::planner::{PlanRequest, PlanResult};
use crate::scale::{Scale, ScaleError};
use crate::units::StepDt;

/// Errors raised while building a scale, a version or a sample for the learned model.
#[derive(Debug, Clone, PartialEq)]
pub enum LearnedModelError {
    /// The bin layout has too many buckets to index.
    ScaleTooLarge,
    /// The learner shape does not match the scale (`num_bins + 1` states, same action count).
    ShapeMismatch {
        /// Number of states the scale requires (bins plus the failure bucket).
        expected_states: usize,
        /// Number of states the learner has.
        actual_states: usize,
    },
    /// A configuration value is not finite and strictly positive.
    InvalidConfig(&'static str),
    /// A state could not be projected onto the scale.
    Projection(EntropyError),
    /// The typed [`Scale`] contract could not be built from the bin specification.
    Scale(ScaleError),
    /// The underlying learner rejected the request.
    Learning(LearningError),
}

impl fmt::Display for LearnedModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScaleTooLarge => write!(f, "tabular scale has too many buckets"),
            Self::ShapeMismatch {
                expected_states,
                actual_states,
            } => write!(
                f,
                "learner has {actual_states} states, scale requires {expected_states}"
            ),
            Self::InvalidConfig(field) => {
                write!(
                    f,
                    "invalid learned model config `{field}`: must be finite and > 0"
                )
            }
            Self::Projection(err) => write!(f, "state projection failed: {err}"),
            Self::Scale(err) => write!(f, "invalid scale: {err}"),
            Self::Learning(err) => write!(f, "learner error: {err}"),
        }
    }
}

impl Error for LearnedModelError {}

impl From<ScaleError> for LearnedModelError {
    fn from(value: ScaleError) -> Self {
        Self::Scale(value)
    }
}

/// Discretization of `State` vectors into tabular buckets plus one absorbing failure bucket.
///
/// Buckets `0..num_bins()` are the row-major cells of the [`BinSpec`]; bucket
/// [`TabularScale::failure_bucket`] (`== num_bins()`) is reserved for terminal failure.
///
/// Since GOG-200 the discretization is owned by a typed [`Scale`] ([`TabularScale::as_scale`]):
/// the same contract supplies the learner's buckets and the identity the planner records, so both
/// sides cannot diverge through separately rebuilt bin parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TabularScale {
    contract: Scale,
    strides: Vec<usize>,
    num_bins: usize,
}

impl TabularScale {
    /// Builds a scale from a [`BinSpec`] (the reference projection of `microstate`/`entropy`).
    ///
    /// The [`BinSpec`] is lifted into the typed [`Scale`] contract; prefer
    /// [`TabularScale::from_scale`] when a shared contract already exists.
    pub fn new(bins: BinSpec) -> Result<Self, LearnedModelError> {
        Self::from_scale(Scale::from_bin_spec(bins)?)
    }

    /// Builds the tabular bucket layout from an existing [`Scale`] contract.
    pub fn from_scale(contract: Scale) -> Result<Self, LearnedModelError> {
        let mut strides = vec![0usize; contract.bins().dimension()];
        let mut total = 1usize;
        for (axis, stride) in contract.bins().axes().iter().zip(strides.iter_mut()).rev() {
            *stride = total;
            total = total
                .checked_mul(axis.num_bins())
                .ok_or(LearnedModelError::ScaleTooLarge)?;
        }
        // The failure bucket needs one more slot.
        total
            .checked_add(1)
            .ok_or(LearnedModelError::ScaleTooLarge)?;
        Ok(Self {
            contract,
            strides,
            num_bins: total,
        })
    }

    /// Number of regular (non-failure) buckets.
    pub const fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Index of the absorbing failure bucket.
    pub const fn failure_bucket(&self) -> usize {
        self.num_bins
    }

    /// Number of learner states required: regular buckets plus the failure bucket.
    pub const fn num_states(&self) -> usize {
        self.num_bins + 1
    }

    /// The underlying bin specification.
    pub const fn bins(&self) -> &BinSpec {
        self.contract.bins()
    }

    /// The typed [`Scale`] contract shared with the planner.
    pub const fn as_scale(&self) -> &Scale {
        &self.contract
    }

    /// Projects state values onto a regular bucket.
    pub fn bucket_of(&self, values: &[f64]) -> Result<usize, LearnedModelError> {
        let coords = self
            .bins()
            .bin_coords(values)
            .map_err(LearnedModelError::Projection)?;
        Ok(coords.iter().zip(&self.strides).map(|(c, s)| c * s).sum())
    }

    /// Representative state values (cell centers) of a regular bucket.
    ///
    /// Returns `None` for the failure bucket or an out-of-range index.
    pub fn center_of(&self, bucket: usize) -> Option<Vec<f64>> {
        if bucket >= self.num_bins {
            return None;
        }
        let mut rest = bucket;
        let mut out = Vec::with_capacity(self.strides.len());
        for (axis, stride) in self.contract.bins().axes().iter().zip(&self.strides) {
            let coord = rest / stride;
            rest %= stride;
            out.push(axis_center(axis, coord));
        }
        Some(out)
    }

    /// Builds the learner sample for an observed transition, recording terminal failure.
    ///
    /// A failure terminal (`Failure` or `InvariantViolated`) is stored as a transition into
    /// the absorbing failure bucket; any other transition is projected onto its next bucket.
    pub fn sample_from_transition(
        &self,
        prev_values: &[f64],
        action_idx: usize,
        next_values: &[f64],
        terminal: Terminal,
    ) -> Result<TransitionSample, LearnedModelError> {
        let from = self.bucket_of(prev_values)?;
        if is_failure_terminal(terminal) {
            return Ok(TransitionSample::failure(
                from,
                action_idx,
                self.failure_bucket(),
            ));
        }
        let to = self.bucket_of(next_values)?;
        Ok(TransitionSample::new(from, action_idx, to))
    }
}

fn axis_center(axis: &BinAxisSpec, coord: usize) -> f64 {
    match axis {
        BinAxisSpec::Uniform { min, max, num_bins } => {
            let width = (max - min) / (*num_bins as f64);
            min + (coord as f64 + 0.5) * width
        }
        BinAxisSpec::ExplicitEdges { edges } => 0.5 * (edges[coord] + edges[coord + 1]),
    }
}

/// Immutable frozen revision of a learner and its scale.
///
/// There is no mutating API: to publish a new revision, freeze the learner again.
#[derive(Debug, Clone)]
pub struct ModelVersion {
    learner: TabularTransitionLearner,
    scale: TabularScale,
}

impl ModelVersion {
    /// Freezes a clone of `learner` (the caller keeps learning on its own copy).
    pub fn freeze(
        learner: &TabularTransitionLearner,
        scale: TabularScale,
    ) -> Result<Arc<Self>, LearnedModelError> {
        if learner.num_states() != scale.num_states() {
            return Err(LearnedModelError::ShapeMismatch {
                expected_states: scale.num_states(),
                actual_states: learner.num_states(),
            });
        }
        Ok(Arc::new(Self {
            learner: learner.clone(),
            scale,
        }))
    }

    /// Monotonic learner revision captured by this version.
    pub const fn revision(&self) -> u64 {
        self.learner.model_version()
    }

    /// Learner checksum captured by this version.
    pub const fn checksum(&self) -> u64 {
        self.learner.checksum()
    }

    /// The scale of this version.
    pub const fn scale(&self) -> &TabularScale {
        &self.scale
    }

    /// Number of actions the frozen learner covers.
    pub const fn num_actions(&self) -> usize {
        self.learner.num_actions()
    }
}

/// What to do when a `(bucket, action)` pair has no recorded evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnseenPairPolicy {
    /// Sample from the Dirichlet prior (near-uniform plus failure pseudocount) and flag it.
    #[default]
    SamplePrior,
    /// Fail the step with [`ModelError::InvalidState`].
    Reject,
}

/// Adapter parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnedModelConfig {
    failure_pseudocount: f64,
    min_evidence: u64,
    unseen: UnseenPairPolicy,
}

impl LearnedModelConfig {
    /// Builds a config. `failure_pseudocount` is the Dirichlet pseudocount of the failure
    /// bucket (the regular buckets use the learner's `prior_alpha`); `min_evidence >= 1` is
    /// the number of observations below which a pair is flagged as uncertain.
    pub fn new(
        failure_pseudocount: f64,
        min_evidence: u64,
        unseen: UnseenPairPolicy,
    ) -> Result<Self, LearnedModelError> {
        if !failure_pseudocount.is_finite() || failure_pseudocount <= 0.0 {
            return Err(LearnedModelError::InvalidConfig("failure_pseudocount"));
        }
        if min_evidence == 0 {
            return Err(LearnedModelError::InvalidConfig("min_evidence"));
        }
        Ok(Self {
            failure_pseudocount,
            min_evidence,
            unseen,
        })
    }

    /// Failure bucket pseudocount.
    pub const fn failure_pseudocount(&self) -> f64 {
        self.failure_pseudocount
    }
}

impl Default for LearnedModelConfig {
    fn default() -> Self {
        Self {
            failure_pseudocount: 0.5,
            min_evidence: 1,
            unseen: UnseenPairPolicy::SamplePrior,
        }
    }
}

/// Predictive distribution over buckets for one pair, with explicit uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct BucketPrediction {
    /// Probabilities over `0..num_states()`; the last entry is the failure bucket.
    pub distribution: Vec<f64>,
    /// Number of observations behind this pair.
    pub evidence: u64,
    /// `true` when `evidence < min_evidence`: the distribution is prior, not data.
    pub uncertain: bool,
}

/// Provenance of one simulated step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LearnedStepReport {
    /// Bucket the source state was projected to.
    pub state_bucket: usize,
    /// Sampled destination bucket (may be the failure bucket).
    pub next_bucket: usize,
    /// Evidence count of the pair.
    pub evidence: u64,
    /// `true` when the pair had insufficient evidence.
    pub uncertain: bool,
    /// Learner revision of the version that produced the step.
    pub revision: u64,
    /// Checksum of that version.
    pub checksum: u64,
}

/// [`GenerativeModel`] backed by a frozen [`ModelVersion`].
#[derive(Debug, Clone)]
pub struct TabularLearnedModel {
    version: Arc<ModelVersion>,
    space: ActionSpace,
    config: LearnedModelConfig,
}

impl TabularLearnedModel {
    /// Builds the adapter over a frozen version; it retains the `Arc`, so later revisions do
    /// not affect it.
    pub fn new(
        version: Arc<ModelVersion>,
        config: LearnedModelConfig,
    ) -> Result<Self, LearnedModelError> {
        let space = ActionSpace::discrete(version.num_actions())
            .map_err(|_| LearnedModelError::InvalidConfig("num_actions"))?;
        Ok(Self {
            version,
            space,
            config,
        })
    }

    /// The frozen version this model samples from.
    pub fn version(&self) -> &Arc<ModelVersion> {
        &self.version
    }

    /// Dirichlet posterior concentration of `(state_bucket, action_idx)` over next buckets:
    /// prior pseudocount (learner `prior_alpha`, or the failure pseudocount for the failure
    /// bucket) plus the observed transition count. [`Self::predict`] is its normalisation.
    pub fn concentration(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<Vec<f64>, LearnedModelError> {
        let learner = &self.version.learner;
        let failure = self.version.scale.failure_bucket();
        let alpha = learner.policy().prior_alpha();
        (0..learner.num_states())
            .map(|next| {
                let count = learner
                    .transition_count(state_bucket, action_idx, next)
                    .map_err(LearnedModelError::Learning)? as f64;
                let prior = if next == failure {
                    self.config.failure_pseudocount
                } else {
                    alpha
                };
                Ok(prior + count)
            })
            .collect()
    }

    /// Predictive distribution for a pair: Dirichlet means with the learner's `prior_alpha`
    /// on regular buckets and the failure pseudocount on the failure bucket.
    pub fn predict(
        &self,
        state_bucket: usize,
        action_idx: usize,
    ) -> Result<BucketPrediction, LearnedModelError> {
        let evidence = self
            .version
            .learner
            .pair_observation_count(state_bucket, action_idx)
            .map_err(LearnedModelError::Learning)?;
        let mut weights = self.concentration(state_bucket, action_idx)?;
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
        }
        Ok(BucketPrediction {
            distribution: weights,
            evidence,
            uncertain: evidence < self.config.min_evidence,
        })
    }

    /// Like [`GenerativeModel::step`] but also returns the step provenance and uncertainty.
    pub fn step_with_report(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<(Transition, LearnedStepReport), ModelError> {
        self.space
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        let action_idx = action
            .as_discrete()
            .ok_or(ModelError::InvalidState {
                reason: "learned model requires a discrete action",
            })?
            .index();
        let scale = &self.version.scale;
        let from = scale
            .bucket_of(state.values())
            .map_err(|_| ModelError::InvalidState {
                reason: "state cannot be projected onto the model scale",
            })?;
        let prediction = self
            .predict(from, action_idx)
            .map_err(|_| ModelError::InvalidState {
                reason: "state bucket outside the frozen model",
            })?;
        if prediction.uncertain && self.config.unseen == UnseenPairPolicy::Reject {
            return Err(ModelError::InvalidState {
                reason: "no evidence for this (bucket, action) pair",
            });
        }

        let draw = rng.next_f64();
        let mut acc = 0.0;
        let mut next = prediction.distribution.len() - 1;
        for (idx, p) in prediction.distribution.iter().enumerate() {
            acc += p;
            if draw < acc {
                next = idx;
                break;
            }
        }

        let transition = transition_to_bucket(scale, state, dt, next)?;
        let report = LearnedStepReport {
            state_bucket: from,
            next_bucket: next,
            evidence: prediction.evidence,
            uncertain: prediction.uncertain,
            revision: self.version.revision(),
            checksum: self.version.checksum(),
        };
        Ok((transition, report))
    }
}

/// Builds the transition that lands in `next` (a bucket of `scale`) from `state`.
///
/// The failure bucket is absorbing: the state does not move and the transition is a terminal
/// failure. Any other bucket moves the state to the bucket centre.
pub(crate) fn transition_to_bucket(
    scale: &TabularScale,
    state: &State,
    dt: StepDt,
    next: usize,
) -> Result<Transition, ModelError> {
    let (next_values, terminal) = if next == scale.failure_bucket() {
        (
            state.values().to_vec(),
            Terminal::Terminated(TerminalReason::Failure),
        )
    } else {
        (
            scale.center_of(next).ok_or(ModelError::InvalidState {
                reason: "sampled bucket outside the scale",
            })?,
            Terminal::Ongoing,
        )
    };
    let next_state = state
        .advance(dt, next_values.clone())
        .map_err(ModelError::TransitionFailed)?;
    let observation = Observation::new(next_state.tick(), next_state.elapsed(), next_values)
        .map_err(ModelError::TransitionFailed)?;
    Transition::new(state.tick(), dt, next_state, observation, terminal)
        .map_err(ModelError::TransitionFailed)
}

impl GenerativeModel for TabularLearnedModel {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.step_with_report(state, action, dt, rng)
            .map(|(transition, _)| transition)
    }
}

/// Runs FMC planning on a versioned learned model only.
///
/// The planner receives the model (and the request's belief/state data) and nothing else: no
/// live environment, actuator or foreign simulator can be passed here. The decision trace records
/// the identity of the model's [`Scale`] (GOG-200).
pub fn plan_fmc_on_version<R>(
    planner: &FmcPlanner,
    model: &TabularLearnedModel,
    request: &PlanRequest,
    rng: &mut dyn RngSource,
    reward_fn: R,
) -> Result<(PlanResult, FmcTraceSummary), ModelError>
where
    R: FnMut(&State, &Action, &Transition) -> f64,
{
    let scale = model.version().scale().as_scale();
    plan_fmc_with_scale(planner, model, scale, request, rng, reward_fn)
}

/// Runs FMC planning on a versioned learned model with an explicit [`Scale`] handle.
///
/// `scale` must be the same contract the model was built on
/// ([`TabularScale::as_scale`]); a divergent scale is rejected instead of silently projecting
/// states with a different discretization. The recorded scale identity comes from `scale`, not
/// from loose parameters, so the learner-side and planner-side identities cannot be forged
/// independently.
pub fn plan_fmc_with_scale<R>(
    planner: &FmcPlanner,
    model: &TabularLearnedModel,
    scale: &Scale,
    request: &PlanRequest,
    rng: &mut dyn RngSource,
    reward_fn: R,
) -> Result<(PlanResult, FmcTraceSummary), ModelError>
where
    R: FnMut(&State, &Action, &Transition) -> f64,
{
    if model.version().scale().as_scale().identity() != scale.identity() {
        return Err(ModelError::InvalidState {
            reason: "planner scale does not match the learned model scale",
        });
    }
    let mut ctx = PlanningContext::new(model, rng);
    let (result, trace) =
        planner.plan_with_trace(&mut ctx, request, reward_fn, |s| s.values().to_vec())?;
    Ok((result, trace.with_scale_identity(scale.identity())))
}

#[cfg(test)]
mod learned_model_tests {
    use super::*;
    use crate::determinism::MasterSeed;
    use crate::domain::State;
    use crate::entropy::BinningVersion;
    use crate::fmc_planner::FmcPlannerConfig;
    use crate::fmc_scoring::FmcScoringConfig;
    use crate::learning::{LearningPolicy, ObservationId, TransitionObservation};
    use crate::model::DeterministicRng;
    use crate::planner::{CancelToken, PlanBudget, PlanTarget};
    use crate::provenance::EvidenceRef;
    use crate::units::{SimDuration, Tick};

    const BINS: usize = 4;
    const SAFE: usize = 0;
    const LETHAL: usize = 1;

    fn scale() -> TabularScale {
        let spec =
            BinSpec::uniform("t", BinningVersion::new(1, 0), vec![(0.0, 4.0, BINS)]).unwrap();
        TabularScale::new(spec).unwrap()
    }

    fn dt() -> StepDt {
        StepDt::from_millis_f64(100.0).unwrap()
    }

    fn state(x: f64) -> State {
        State::new(Tick::ZERO, SimDuration::ZERO, vec![x]).unwrap()
    }

    /// Feeds the learner: SAFE keeps the bucket; LETHAL either fails (`record_failure`) or, when
    /// failure is not recorded as terminal, is stored as "stays in bucket".
    fn train(learner: &mut TabularTransitionLearner, sc: &TabularScale, record_failure: bool) {
        let mut id = 0;
        for b in 0..BINS {
            let x = b as f64 + 0.5;
            for _ in 0..40 {
                for action in [SAFE, LETHAL] {
                    id += 1;
                    let terminal = if action == LETHAL && record_failure {
                        Terminal::Terminated(TerminalReason::Failure)
                    } else {
                        Terminal::Ongoing
                    };
                    let sample = sc
                        .sample_from_transition(&[x], action, &[x], terminal)
                        .unwrap();
                    let obs = if sample.is_terminal_failure() {
                        TransitionObservation::failure(
                            ObservationId::new(id),
                            EvidenceRef::new(1000 + id),
                            sample.state_bucket(),
                            sample.action_idx(),
                            sample.next_state_bucket(),
                        )
                    } else {
                        TransitionObservation::new(
                            ObservationId::new(id),
                            EvidenceRef::new(1000 + id),
                            sample.state_bucket(),
                            sample.action_idx(),
                            sample.next_state_bucket(),
                        )
                    };
                    learner.observe(&obs).unwrap();
                }
            }
        }
    }

    fn learner(sc: &TabularScale) -> TabularTransitionLearner {
        TabularTransitionLearner::new(sc.num_states(), 2, LearningPolicy::laplace()).unwrap()
    }

    fn model(l: &TabularTransitionLearner, fp: f64) -> TabularLearnedModel {
        let v = ModelVersion::freeze(l, scale()).unwrap();
        TabularLearnedModel::new(
            v,
            LearnedModelConfig::new(fp, 1, UnseenPairPolicy::SamplePrior).unwrap(),
        )
        .unwrap()
    }

    fn fmc_stats(m: &TabularLearnedModel) -> FmcTraceSummary {
        let cfg = FmcPlannerConfig::new(
            24,
            3,
            dt(),
            FmcScoringConfig::default(),
            MasterSeed::new(11),
        )
        .unwrap();
        let planner = FmcPlanner::new(cfg);
        let request = PlanRequest::new(
            PlanTarget::State(state(1.5)),
            ActionSpace::discrete(2).unwrap(),
            PlanBudget::with_transitions(10_000),
            CancelToken::new(),
            None,
        )
        .unwrap();
        let mut rng = DeterministicRng::new(5);
        // Pure modality: constant reward, only viability matters.
        plan_fmc_on_version(&planner, m, &request, &mut rng, |_, _, _| 0.0)
            .unwrap()
            .1
    }

    #[test]
    fn learned_model_failure_sample_is_additive() {
        let plain = TransitionSample::new(1, 0, 2);
        assert!(!plain.is_terminal_failure());
        assert!(TransitionSample::failure(1, 0, 4).is_terminal_failure());
        assert_ne!(plain, TransitionSample::failure(1, 0, 2));
    }

    #[test]
    fn learned_model_failure_bucket_separates_lethal_from_safe_in_pure_fmc() {
        let sc = scale();
        // With the failure bucket: lethal action ends terminally and is excluded as non-viable.
        let mut with = learner(&sc);
        train(&mut with, &sc, true);
        let trace = fmc_stats(&model(&with, 0.5));
        let stats = trace.root_action_stats();
        let support = |idx: usize| {
            stats
                .iter()
                .find(|s| s.action.as_discrete().map(|d| d.index()) == Some(idx))
                .map_or(0, |s| s.support_count)
        };
        assert!(
            support(LETHAL) < support(SAFE),
            "lethal action must have strictly less viable support: {stats:?}"
        );
        assert_eq!(
            trace.first_action().unwrap().as_discrete().unwrap().index(),
            SAFE
        );

        // Without it (failure never recorded as terminal, negligible failure pseudocount):
        // both actions stay viable with identical constant return, so Pure FMC cannot tell them apart.
        let mut without = learner(&sc);
        train(&mut without, &sc, false);
        let trace = fmc_stats(&model(&without, 1e-12));
        let stats = trace.root_action_stats();
        assert_eq!(stats.len(), 2, "both actions look viable: {stats:?}");
        assert!(
            stats
                .iter()
                .all(|s| s.support_count > 0 && s.mean_return == 0.0)
        );
    }

    #[test]
    fn learned_model_unseen_pairs_report_uncertainty_and_failure_pseudocount() {
        let sc = scale();
        let l = learner(&sc);
        let m = model(&l, 0.5);
        let p = m.predict(0, 0).unwrap();
        assert!(p.uncertain);
        assert_eq!(p.evidence, 0);
        let fail = p.distribution[sc.failure_bucket()];
        // 0.5 / (0.5 + 4 * 1.0): own pseudocount, not a confident prediction.
        assert!((fail - 0.5 / 4.5).abs() < 1e-12);
        assert!((p.distribution.iter().sum::<f64>() - 1.0).abs() < 1e-12);

        let (_, report) = m
            .step_with_report(
                &state(0.5),
                &Action::discrete_in_space(0, m.action_space()).unwrap(),
                dt(),
                &mut DeterministicRng::new(1),
            )
            .unwrap();
        assert!(report.uncertain);

        let strict = TabularLearnedModel::new(
            m.version().clone(),
            LearnedModelConfig::new(0.5, 1, UnseenPairPolicy::Reject).unwrap(),
        )
        .unwrap();
        let err = strict
            .step(
                &state(0.5),
                &Action::discrete_in_space(0, strict.action_space()).unwrap(),
                dt(),
                &mut DeterministicRng::new(1),
            )
            .unwrap_err();
        assert!(matches!(err, ModelError::InvalidState { .. }));
    }

    fn rollout(m: &TabularLearnedModel, seed: u64) -> Vec<(usize, bool)> {
        let mut rng = DeterministicRng::new(seed);
        let mut s = state(1.5);
        let mut out = Vec::new();
        for _ in 0..12 {
            let a = Action::discrete_in_space(LETHAL, m.action_space()).unwrap();
            let t = m.step(&s, &a, dt(), &mut rng).unwrap();
            out.push((
                SimBucket::of(t.next_state().values()),
                t.terminal().is_terminal(),
            ));
            if t.terminal().is_terminal() {
                break;
            }
            s = t.into_next_state();
        }
        out
    }

    struct SimBucket;
    impl SimBucket {
        fn of(v: &[f64]) -> usize {
            scale().bucket_of(v).unwrap()
        }
    }

    #[test]
    fn learned_model_revisions_are_reproducible_and_version_is_retained() {
        let sc = scale();
        let mut l = learner(&sc);
        train(&mut l, &sc, true);
        let v1_model = model(&l, 0.5);
        let before = v1_model.predict(1, LETHAL).unwrap();
        let checksum1 = v1_model.version().checksum();

        // Same seed, same version: identical rollouts.
        assert_eq!(rollout(&v1_model, 99), rollout(&v1_model, 99));

        // The learner moves on (revision 2) while v1 is held by an in-flight decision.
        let more = TransitionObservation::new(
            ObservationId::new(9_999),
            EvidenceRef::new(99_999),
            1,
            SAFE,
            2,
        );
        l.observe(&more).unwrap();
        let v2_model = model(&l, 0.5);

        assert_eq!(v1_model.version().checksum(), checksum1);
        assert_eq!(v1_model.predict(1, LETHAL).unwrap(), before);
        assert_eq!(rollout(&v1_model, 99), rollout(&v1_model, 99));
        assert!(v2_model.version().revision() > v1_model.version().revision());
        assert_ne!(v2_model.version().checksum(), v1_model.version().checksum());
        assert_ne!(
            v2_model.predict(1, SAFE).unwrap().distribution,
            v1_model.predict(1, SAFE).unwrap().distribution
        );
        // Same seed on revision 2 is also reproducible.
        assert_eq!(rollout(&v2_model, 7), rollout(&v2_model, 7));
    }

    #[test]
    fn learned_model_freeze_rejects_shape_mismatch() {
        let sc = scale();
        let bad = TabularTransitionLearner::new(BINS, 2, LearningPolicy::laplace()).unwrap();
        assert!(matches!(
            ModelVersion::freeze(&bad, sc),
            Err(LearnedModelError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn learned_model_scale_round_trips_centers() {
        let spec = BinSpec::uniform(
            "2d",
            BinningVersion::new(1, 0),
            vec![(0.0, 2.0, 2), (0.0, 3.0, 3)],
        )
        .unwrap();
        let sc = TabularScale::new(spec).unwrap();
        assert_eq!(sc.num_bins(), 6);
        for b in 0..6 {
            assert_eq!(sc.bucket_of(&sc.center_of(b).unwrap()).unwrap(), b);
        }
        assert!(sc.center_of(sc.failure_bucket()).is_none());
    }
}
