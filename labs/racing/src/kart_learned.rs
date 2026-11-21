//! Learned, observation-only Kart controller (experimental, `std`-only).
//!
//! # What this module is
//! A discrete-action agent that *learns online* how its heading (`yaw`) and `speed` change under
//! macro actions, together with an explicit collision (failure) bucket, using the GOG-167
//! building blocks: [`TabularTransitionLearner`], [`ModelVersion`] (immutable snapshots) and
//! [`TabularLearnedModel`] planned through [`plan_fmc_on_version`].
//!
//! # Architectural boundary (hidden physics)
//! The real physics and the obstacle map live only on the live-environment side
//! (the sibling lab module). This file receives exclusively [`KartObservable`] values
//! (`yaw`, `speed`) and [`KartStepOutcome`]s produced by the caller after it executed a macro
//! action in the real environment. It deliberately names none of the privileged simulator, map
//! or environment types, and the test `kart_learned_agent_source_has_no_privileged_tokens`
//! scans this very file for them.
//!
//! What the scan guarantees: this source cannot mention (hence cannot call) the privileged
//! kinematic model, the obstacle map or the live environment by name, and the public API only
//! accepts observables. What it does NOT guarantee: that a caller cannot smuggle privileged
//! information through the numbers it passes in (the observables are plain `f64`s), nor that other
//! files in the crate stay clean; it is a source-level tripwire, not a capability system.
//!
//! # Known limitations of the state abstraction
//! The model state is only `(yaw, speed)`. Position is not observed by the agent, so a collision is
//! learned as an aliased event of `(yaw, speed, action)`; the model cannot know *where* obstacles are.

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use gogamen_core::EvidenceRef;
use gogamen_core::{
    Action, ActionSpace, BinSpec, BinningVersion, CancelToken, DeterministicRng, FmcPlanner,
    FmcPlannerConfig, FmcScoringConfig, LearnedModelConfig, LearnedModelError, LearningError,
    LearningPolicy, MasterSeed, ModelVersion, ObservationId, PlanBudget, PlanRequest, PlanTarget,
    SimDuration, State, StepDt, TabularLearnedModel, TabularScale, TabularTransitionLearner,
    Terminal, TerminalReason, Tick, TransitionObservation, UnseenPairPolicy, plan_fmc_on_version,
};

/// Number of yaw bins over `[-PI, PI]`.
pub const YAW_BINS: usize = 8;
/// Number of speed bins over `[SPEED_MIN, SPEED_MAX]`.
pub const SPEED_BINS: usize = 6;
/// Lower speed edge of the model scale (m/s).
pub const SPEED_MIN: f64 = -3.0;
/// Upper speed edge of the model scale (m/s).
pub const SPEED_MAX: f64 = 18.0;

/// What the agent sees of the kart: heading and speed. Nothing else.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartObservable {
    /// Heading in radians.
    pub yaw: f64,
    /// Longitudinal speed in m/s.
    pub speed: f64,
}

impl KartObservable {
    /// Builds an observable; rejects non-finite components.
    pub fn new(yaw: f64, speed: f64) -> Result<Self, KartLearnedError> {
        if !yaw.is_finite() || !speed.is_finite() {
            return Err(KartLearnedError::new("observable must be finite"));
        }
        Ok(Self { yaw, speed })
    }

    fn values(self) -> [f64; 2] {
        [self.yaw, self.speed]
    }
}

/// Discrete macro action: steering level x acceleration level, each held for several ticks by the
/// caller. Levels are fractions of the control range in `[-1, 1]`; the caller scales them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KartMacroAction {
    /// Steer left, brake.
    LeftBrake,
    /// Steer left, accelerate.
    LeftThrottle,
    /// Straight, brake.
    StraightBrake,
    /// Straight, accelerate.
    StraightThrottle,
    /// Steer right, brake.
    RightBrake,
    /// Steer right, accelerate.
    RightThrottle,
}

impl KartMacroAction {
    /// All actions in index order.
    pub const ALL: [Self; 6] = [
        Self::LeftBrake,
        Self::LeftThrottle,
        Self::StraightBrake,
        Self::StraightThrottle,
        Self::RightBrake,
        Self::RightThrottle,
    ];

    /// Learner action index.
    pub fn index(self) -> usize {
        Self::ALL.iter().position(|a| *a == self).unwrap_or(0)
    }

    /// Inverse of [`Self::index`].
    pub fn from_index(index: usize) -> Option<Self> {
        Self::ALL.get(index).copied()
    }

    /// Steering as a fraction of the steering range (positive
    /// is left, matching a positive yaw rate).
    pub fn steer_fraction(self) -> f64 {
        match self {
            Self::LeftBrake | Self::LeftThrottle => 0.5,
            Self::StraightBrake | Self::StraightThrottle => 0.0,
            Self::RightBrake | Self::RightThrottle => -0.5,
        }
    }

    /// Acceleration as a fraction of the accel range (`-1` brake, `+1` throttle).
    pub fn accel_fraction(self) -> f64 {
        match self {
            Self::LeftBrake | Self::StraightBrake | Self::RightBrake => -0.5,
            _ => 0.5,
        }
    }
}

/// Result of one executed macro action, as reported by the environment side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KartStepOutcome {
    /// The kart kept going and was observed at this new heading/speed.
    Continued(KartObservable),
    /// The macro action ended in a collision (terminal failure).
    Collided,
}

/// One executed (not synthetic) transition, kept for held-out evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartExecutedTransition {
    /// Observable before the macro action.
    pub before: KartObservable,
    /// Macro action executed.
    pub action: KartMacroAction,
    /// What happened.
    pub outcome: KartStepOutcome,
}

/// Error type of this module.
#[derive(Debug, Clone, PartialEq)]
pub struct KartLearnedError(String);

impl KartLearnedError {
    /// Builds an error from a message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self(msg.into())
    }
}

impl fmt::Display for KartLearnedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kart learned: {}", self.0)
    }
}

impl Error for KartLearnedError {}

impl From<LearnedModelError> for KartLearnedError {
    fn from(e: LearnedModelError) -> Self {
        Self::new(e.to_string())
    }
}

impl From<LearningError> for KartLearnedError {
    fn from(e: LearningError) -> Self {
        Self::new(e.to_string())
    }
}

/// Prediction error of one frozen model version over a set of executed transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct KartPredictionError {
    /// Learner revision of the evaluated version.
    pub revision: u64,
    /// Number of evaluated transitions that continued (non-collision).
    pub continued: usize,
    /// Mean absolute circular yaw error (rad) of the predictive mean, continued transitions.
    pub mean_yaw_error: f64,
    /// Mean absolute speed error (m/s) of the predictive mean, continued transitions.
    pub mean_speed_error: f64,
    /// Mean Brier score over the full bucket distribution (all transitions, failure included).
    pub mean_brier: f64,
    /// Number of evaluated transitions (continued + collided).
    pub total: usize,
    /// Fraction of evaluated pairs flagged uncertain (insufficient evidence).
    pub uncertain_fraction: f64,
}

/// Record of one published model version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KartVersionRecord {
    /// Learner revision.
    pub revision: u64,
    /// Learner checksum.
    pub checksum: u64,
    /// Updates applied when the version was frozen.
    pub updates: u64,
}

/// Online learning cost counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KartUpdateCost {
    /// Number of effective updates.
    pub updates: u64,
    /// Total elapsed nanoseconds spent in updates (informational, not deterministic).
    pub total_nanos: u128,
}

/// Observation-only learning agent.
#[derive(Debug, Clone)]
pub struct KartLearnedAgent {
    scale: TabularScale,
    learner: TabularTransitionLearner,
    next_id: u64,
    cost: KartUpdateCost,
    versions: Vec<KartVersionRecord>,
}

fn model_config() -> LearnedModelConfig {
    LearnedModelConfig::new(0.5, 1, UnseenPairPolicy::SamplePrior)
        .expect("static learned model config is valid")
}

impl KartLearnedAgent {
    /// Creates an agent holding only the prior (revision 0).
    pub fn new() -> Result<Self, KartLearnedError> {
        let spec = BinSpec::uniform(
            "kart-yaw-speed",
            BinningVersion::new(1, 0),
            vec![
                (-std::f64::consts::PI, std::f64::consts::PI, YAW_BINS),
                (SPEED_MIN, SPEED_MAX, SPEED_BINS),
            ],
        )
        .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let scale = TabularScale::new(spec)?;
        let learner = TabularTransitionLearner::new(
            scale.num_states(),
            KartMacroAction::ALL.len(),
            LearningPolicy::laplace(),
        )?;
        Ok(Self {
            scale,
            learner,
            next_id: 0,
            cost: KartUpdateCost::default(),
            versions: Vec::new(),
        })
    }

    /// Online update from one executed transition; records its cost.
    pub fn learn(&mut self, t: &KartExecutedTransition) -> Result<(), KartLearnedError> {
        let started = Instant::now();
        let terminal = match t.outcome {
            KartStepOutcome::Collided => Terminal::Terminated(TerminalReason::Failure),
            KartStepOutcome::Continued(_) => Terminal::Ongoing,
        };
        let next = match t.outcome {
            KartStepOutcome::Continued(o) => o.values(),
            KartStepOutcome::Collided => t.before.values(),
        };
        let sample = self.scale.sample_from_transition(
            &t.before.values(),
            t.action.index(),
            &next,
            terminal,
        )?;
        self.next_id += 1;
        let id = ObservationId::new(self.next_id);
        let ev = EvidenceRef::new(self.next_id);
        let obs = if sample.is_terminal_failure() {
            TransitionObservation::failure(
                id,
                ev,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        } else {
            TransitionObservation::new(
                id,
                ev,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        };
        self.learner.observe(&obs)?;
        self.cost.updates += 1;
        self.cost.total_nanos += started.elapsed().as_nanos();
        Ok(())
    }

    /// Freezes the current learner into an immutable version and records it.
    pub fn publish(&mut self) -> Result<TabularLearnedModel, KartLearnedError> {
        let version = ModelVersion::freeze(&self.learner, self.scale.clone())?;
        self.versions.push(KartVersionRecord {
            revision: version.revision(),
            checksum: version.checksum(),
            updates: self.cost.updates,
        });
        Ok(TabularLearnedModel::new(
            Arc::clone(&version),
            model_config(),
        )?)
    }

    /// Update cost counters.
    pub fn cost(&self) -> KartUpdateCost {
        self.cost
    }

    /// Versions published so far.
    pub fn versions(&self) -> &[KartVersionRecord] {
        &self.versions
    }

    /// Evaluates a frozen model against executed transitions (predictive mean vs. actual).
    pub fn prediction_error(
        &self,
        model: &TabularLearnedModel,
        held: &[KartExecutedTransition],
    ) -> Result<KartPredictionError, KartLearnedError> {
        let scale = model.version().scale();
        let failure = scale.failure_bucket();
        let (mut yaw_sum, mut speed_sum, mut cont) = (0.0, 0.0, 0usize);
        let (mut brier, mut uncertain) = (0.0, 0usize);
        for t in held {
            let from = scale.bucket_of(&t.before.values())?;
            let pred = model.predict(from, t.action.index())?;
            if pred.uncertain {
                uncertain += 1;
            }
            let actual_bucket = match t.outcome {
                KartStepOutcome::Collided => failure,
                KartStepOutcome::Continued(o) => scale.bucket_of(&o.values())?,
            };
            brier += pred
                .distribution
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let y = if i == actual_bucket { 1.0 } else { 0.0 };
                    (p - y) * (p - y)
                })
                .sum::<f64>();
            if let KartStepOutcome::Continued(actual) = t.outcome {
                let (yaw, speed) = predictive_mean(scale, &pred.distribution);
                yaw_sum += circular_abs_diff(yaw, actual.yaw);
                speed_sum += (speed - actual.speed).abs();
                cont += 1;
            }
        }
        let total = held.len().max(1) as f64;
        let cont_f = cont.max(1) as f64;
        Ok(KartPredictionError {
            revision: model.version().revision(),
            continued: cont,
            mean_yaw_error: yaw_sum / cont_f,
            mean_speed_error: speed_sum / cont_f,
            mean_brier: brier / total,
            total: held.len(),
            uncertain_fraction: uncertain as f64 / total,
        })
    }

    /// Chooses a macro action by planning on the frozen learned model only.
    ///
    /// The reward is the predicted speed after the step; collision avoidance enters only through
    /// the learned failure bucket. This is a proxy, not a racing objective.
    pub fn decide(
        model: &TabularLearnedModel,
        observed: KartObservable,
        macro_dt: StepDt,
        seed: u64,
    ) -> Result<KartMacroAction, KartLearnedError> {
        let space = ActionSpace::discrete(KartMacroAction::ALL.len())
            .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let fallback = Action::discrete_in_space(KartMacroAction::StraightBrake.index(), &space)
            .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let cfg = FmcPlannerConfig::new(
            24,
            3,
            macro_dt,
            FmcScoringConfig::default(),
            MasterSeed::new(seed),
        )
        .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let planner = FmcPlanner::new(cfg);
        let state = State::new(Tick::ZERO, SimDuration::ZERO, observed.values().to_vec())
            .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let request = PlanRequest::new(
            PlanTarget::State(state),
            space,
            PlanBudget::with_transitions(10_000),
            CancelToken::new(),
            Some(fallback),
        )
        .map_err(|e| KartLearnedError::new(e.to_string()))?;
        let mut rng = DeterministicRng::new(seed);
        let (_, trace) = plan_fmc_on_version(&planner, model, &request, &mut rng, |_, _, t| {
            t.next_state().values().get(1).copied().unwrap_or(0.0)
        })
        .map_err(|e| KartLearnedError::new(e.to_string()))?;
        Ok(trace
            .first_action()
            .and_then(|a| a.as_discrete())
            .and_then(|d| KartMacroAction::from_index(d.index()))
            .unwrap_or(KartMacroAction::StraightBrake))
    }
}

/// Predictive mean of `(yaw, speed)` over the non-failure buckets (circular mean for yaw).
fn predictive_mean(scale: &TabularScale, distribution: &[f64]) -> (f64, f64) {
    let (mut s, mut c, mut speed, mut mass) = (0.0, 0.0, 0.0, 0.0);
    for (b, &p) in distribution.iter().enumerate().take(scale.num_bins()) {
        if let Some(center) = scale.center_of(b) {
            s += p * center[0].sin();
            c += p * center[0].cos();
            speed += p * center[1];
            mass += p;
        }
    }
    if mass <= 0.0 {
        return (0.0, 0.0);
    }
    (s.atan2(c), speed / mass)
}

fn circular_abs_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(2.0 * std::f64::consts::PI);
    d.min(2.0 * std::f64::consts::PI - d)
}

/// Mean error of the "nothing changes" predictor on the same transitions (reference baseline).
pub fn persistence_error(held: &[KartExecutedTransition]) -> (f64, f64) {
    let (mut y, mut s, mut n) = (0.0, 0.0, 0usize);
    for t in held {
        if let KartStepOutcome::Continued(a) = t.outcome {
            y += circular_abs_diff(t.before.yaw, a.yaw);
            s += (t.before.speed - a.speed).abs();
            n += 1;
        }
    }
    let n = n.max(1) as f64;
    (y / n, s / n)
}
