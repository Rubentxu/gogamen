//! FMC planning on Grid with **hidden physics** (GOG-168 / DEM-015).
//!
//! [`PlanGrid`] reuses the core [`FmcPlanner`] (through [`plan_fmc_on_version`]) over a
//! [`TabularLearnedModel`]: a frozen, versioned model learned from experience. The agent never
//! touches the simulator ([`crate::GridWorldModel`] / [`crate::GridWorldEnv`]); it only receives
//! its observed cell and the task specification ([`GridTask`]: goal cell and reward constants,
//! which are the *objective*, not the physics).
//!
//! # Architecture guarantee
//! `PlanGrid` can only be built from, and only plans over, a [`TabularLearnedModel`]:
//!
//! ```no_run
//! use gogamen_lab_grid::hidden_dynamics::{GridTask, PlanGrid, PlanGridConfig};
//! use gogamen_lab_grid::GridWorldConfig;
//! use gogamen_core::TabularLearnedModel;
//! fn ok(model: TabularLearnedModel, world: &GridWorldConfig) {
//!     let config = PlanGridConfig::new(GridTask::from_world(world));
//!     let _ = PlanGrid::new(model, config);
//! }
//! ```
//!
//! A privileged simulator is rejected at compile time:
//!
//! ```compile_fail
//! use gogamen_lab_grid::hidden_dynamics::{GridTask, PlanGrid, PlanGridConfig};
//! use gogamen_lab_grid::{GridWorldConfig, GridWorldModel};
//! fn bad(sim: GridWorldModel, world: &GridWorldConfig) {
//!     let config = PlanGridConfig::new(GridTask::from_world(world));
//!     let _ = PlanGrid::new(sim, config);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_lab_grid::hidden_dynamics::{GridTask, PlanGrid, PlanGridConfig};
//! use gogamen_lab_grid::{GridWorldConfig, GridWorldEnv};
//! fn bad(env: GridWorldEnv, world: &GridWorldConfig) {
//!     let config = PlanGridConfig::new(GridTask::from_world(world));
//!     let _ = PlanGrid::new(env, config);
//! }
//! ```
//!
//! # Abstention
//! Every decision carries a [`PlanGridTrace`] with the model revision/checksum and, when the
//! agent does not trust the plan, a typed [`FallbackReason`]; the executed action is then the
//! safe `Stay` fallback.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use gogamen_core::{
    Action, ActionSpace, Actuator, BinSpec, BinningVersion, CancelToken, DeterministicRng,
    DomainError, EvidenceRef, FmcPlanner, FmcPlannerConfig, FmcScoringConfig, FreedomField,
    GenerativeModel, LearnedModelConfig, LearnedModelError, LearningError, LearningPolicy,
    MasterSeed, ModelError, ModelVersion, ObservationId, PlanBudget, PlanRequest, PlanStatus,
    PlanningContext, RngSource, SimDuration, State, StepDt, TabularLearnedModel, TabularScale,
    TabularTransitionLearner, Terminal, TerminalReason, Tick, Transition, TransitionObservation,
    plan_fmc_on_version,
};

use crate::map::{GridMap, GridPos};
use crate::world::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldEnv, GridWorldModel};

/// Errors of the hidden-physics lab.
#[derive(Debug, Clone, PartialEq)]
pub enum HiddenGridError {
    /// Domain-level failure (invalid config, action or terminal state).
    Domain(DomainError),
    /// The learned model could not be built.
    Model(LearnedModelError),
    /// The learner rejected an observation.
    Learning(LearningError),
    /// The planner configuration is invalid.
    PlannerConfig,
    /// The configuration is invalid (`reason` names the field).
    InvalidConfig(&'static str),
}

impl fmt::Display for HiddenGridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(e) => write!(f, "domain error: {e}"),
            Self::Model(e) => write!(f, "learned model error: {e}"),
            Self::Learning(e) => write!(f, "learner error: {e}"),
            Self::PlannerConfig => write!(f, "invalid FMC planner configuration"),
            Self::InvalidConfig(field) => write!(f, "invalid configuration: {field}"),
        }
    }
}

impl std::error::Error for HiddenGridError {}

impl From<DomainError> for HiddenGridError {
    fn from(e: DomainError) -> Self {
        Self::Domain(e)
    }
}
impl From<LearnedModelError> for HiddenGridError {
    fn from(e: LearnedModelError) -> Self {
        Self::Model(e)
    }
}
impl From<LearningError> for HiddenGridError {
    fn from(e: LearningError) -> Self {
        Self::Learning(e)
    }
}

/// Task specification given to the agent: the objective, not the physics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridTask {
    /// Goal cell.
    pub goal: GridPos,
    /// Reward for an ordinary move.
    pub step_reward: f64,
    /// Reward when the agent does not move (bumped into something).
    pub wall_collision_reward: f64,
    /// Reward for entering the goal cell.
    pub goal_reward: f64,
    /// Reward for a transition that ends in terminal failure.
    pub hazard_reward: f64,
    /// Planning time step.
    pub dt: StepDt,
}

impl GridTask {
    /// Extracts the task spec (goal and reward constants) from a world config. The map layout
    /// (walls, hazards) is deliberately not copied.
    pub fn from_world(world: &GridWorldConfig) -> Self {
        Self {
            goal: world.map.goal(),
            step_reward: world.step_reward,
            wall_collision_reward: world.wall_collision_reward,
            goal_reward: world.goal_reward,
            hazard_reward: world.hazard_reward,
            dt: world.dt,
        }
    }
}

/// [`PlanGrid`] parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGridConfig {
    /// Task specification.
    pub task: GridTask,
    /// FMC walker population.
    pub population_size: usize,
    /// FMC horizon.
    pub horizon: usize,
    /// Simulated-transition budget per decision.
    pub budget_transitions: usize,
    /// Abstain when more than this fraction of simulated steps used a pair without evidence.
    pub max_uncertain_fraction: f64,
    /// FMC scoring.
    pub scoring: FmcScoringConfig,
    /// FMC seed.
    pub seed: MasterSeed,
}

impl PlanGridConfig {
    /// Defaults: 16 walkers, horizon 10, budget `16 * 10`, abstain above 50 % uncertain steps.
    pub fn new(task: GridTask) -> Self {
        Self {
            task,
            population_size: 16,
            horizon: 10,
            budget_transitions: 160,
            max_uncertain_fraction: 0.5,
            scoring: FmcScoringConfig::default(),
            seed: MasterSeed::new(7),
        }
    }
}

/// Typed reason for not executing the planned action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FallbackReason {
    /// The plan relied on too many `(cell, action)` pairs without evidence.
    UncertainModel,
    /// The model rejected an unseen pair ([`gogamen_core::UnseenPairPolicy::Reject`]).
    UnseenPairRejected,
    /// The transition budget expired before the plan completed.
    BudgetExhausted,
    /// The plan was cancelled.
    Cancelled,
    /// The planner found no valid action.
    NoValidAction,
    /// Any other model failure.
    ModelError,
}

impl FallbackReason {
    /// Stable label for reports.
    pub const fn label(self) -> &'static str {
        match self {
            Self::UncertainModel => "uncertain_model",
            Self::UnseenPairRejected => "unseen_pair_rejected",
            Self::BudgetExhausted => "budget_exhausted",
            Self::Cancelled => "cancelled",
            Self::NoValidAction => "no_valid_action",
            Self::ModelError => "model_error",
        }
    }
}

/// Trace of one [`PlanGrid`] decision, tagged with the model version.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGridTrace {
    /// Episode tick of the decision.
    pub tick: u64,
    /// Learner revision of the model version used.
    pub model_revision: u64,
    /// Checksum of the model version used.
    pub model_checksum: u64,
    /// Planner status (`None` when the model errored before a plan existed).
    pub status: Option<PlanStatus>,
    /// Simulated transitions consumed (never above the request budget).
    pub transitions_used: usize,
    /// Simulated steps seen by the reward callback.
    pub simulated_steps: usize,
    /// Simulated steps whose pair had insufficient evidence.
    pub uncertain_steps: usize,
    /// Index of the action actually returned.
    pub action_index: usize,
    /// `Some` when the returned action is the fallback, with the exact reason.
    pub fallback: Option<FallbackReason>,
}

/// One decision: the action to execute and its trace.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGridDecision {
    /// Action to apply (planned or fallback `Stay`).
    pub action: Action,
    /// Trace with model version and abstention reason.
    pub trace: PlanGridTrace,
}

/// FMC planner over a learned, versioned model: the agent has no simulator access.
#[derive(Debug, Clone)]
pub struct PlanGrid {
    model: TabularLearnedModel,
    planner: FmcPlanner,
    config: PlanGridConfig,
    freedom_leaf: Option<FreedomLeaf>,
}

/// Optional leaf value: the derived freedom field of the agent's own model version, applied as
/// potential-based shaping `weight * (phi(next) - phi(prev))` on every simulated step. The sum
/// over a rollout telescopes to `weight * (phi(last) - phi(first))`, i.e. a leaf value at the
/// horizon, so the planner itself is unchanged. The field is only read as a number: it grants
/// no permission and never replaces the guard.
#[derive(Debug, Clone)]
struct FreedomLeaf {
    field: Arc<FreedomField>,
    weight: f64,
    cap: f64,
}

impl PlanGrid {
    /// Builds the agent. The only dynamics it will ever see are those of `model`.
    pub fn new(
        model: TabularLearnedModel,
        config: PlanGridConfig,
    ) -> Result<Self, HiddenGridError> {
        if !(0.0..=1.0).contains(&config.max_uncertain_fraction) {
            return Err(HiddenGridError::InvalidConfig("max_uncertain_fraction"));
        }
        let planner_config = FmcPlannerConfig::new(
            config.population_size,
            config.horizon,
            config.task.dt,
            config.scoring,
            config.seed,
        )
        .map_err(|_| HiddenGridError::PlannerConfig)?;
        Ok(Self {
            model,
            planner: FmcPlanner::new(planner_config),
            config,
            freedom_leaf: None,
        })
    }

    /// Enables the freedom field as leaf value (default: disabled, behaviour unchanged).
    ///
    /// The field must have been computed from exactly this agent's model version (revision,
    /// checksum and scale); otherwise [`HiddenGridError::InvalidConfig`] is returned. `weight`
    /// (reward units per nat of potential) must be finite and non-negative; `cap` (nats, `> 0`)
    /// bounds the penalty of a zero-viability bucket.
    pub fn with_freedom_leaf(
        mut self,
        field: Arc<FreedomField>,
        weight: f64,
        cap: f64,
    ) -> Result<Self, HiddenGridError> {
        if !field.key().matches(self.model.version()) {
            return Err(HiddenGridError::InvalidConfig("freedom field version"));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(HiddenGridError::InvalidConfig("freedom leaf weight"));
        }
        if !cap.is_finite() || cap <= 0.0 {
            return Err(HiddenGridError::InvalidConfig("freedom leaf cap"));
        }
        self.freedom_leaf = Some(FreedomLeaf { field, weight, cap });
        Ok(self)
    }

    /// The versioned model in use.
    pub fn model(&self) -> &TabularLearnedModel {
        &self.model
    }

    /// Plans one action from the observed cell. Never fails: on any problem it returns the
    /// `Stay` fallback with a typed reason.
    pub fn decide(
        &self,
        observed: GridPos,
        tick: Tick,
        cancel: Option<&CancelToken>,
    ) -> PlanGridDecision {
        let task = self.config.task;
        let version = self.model.version();
        let scale = version.scale();
        let stay = grid_action(0).expect("stay is a valid action");
        let base_trace = PlanGridTrace {
            tick: tick.as_u64(),
            model_revision: version.revision(),
            model_checksum: version.checksum(),
            status: None,
            transitions_used: 0,
            simulated_steps: 0,
            uncertain_steps: 0,
            action_index: 0,
            fallback: None,
        };
        let abstain = |reason: FallbackReason, mut trace: PlanGridTrace| PlanGridDecision {
            action: stay.clone(),
            trace: {
                trace.fallback = Some(reason);
                trace.action_index = 0;
                trace
            },
        };

        let state = match State::new(
            tick,
            SimDuration::ZERO,
            vec![observed.x as f64 + 0.5, observed.y as f64 + 0.5],
        ) {
            Ok(s) => s,
            Err(_) => return abstain(FallbackReason::ModelError, base_trace),
        };
        let mut request = PlanRequest::from_state(
            state,
            self.model.action_space().clone(),
            PlanBudget::with_transitions(self.config.budget_transitions),
        )
        .with_fallback(stay.clone())
        .expect("stay fallback is valid");
        if let Some(token) = cancel {
            request = request.with_cancel_token(token.clone());
        }

        let mut simulated = 0usize;
        let mut uncertain = 0usize;
        let model = &self.model;
        let leaf = self.freedom_leaf.as_ref();
        let reward_fn = |prev: &State, action: &Action, tr: &Transition| {
            simulated += 1;
            if let (Ok(bucket), Some(a)) = (scale.bucket_of(prev.values()), action.as_discrete())
                && model.predict(bucket, a.index()).is_ok_and(|p| p.uncertain)
            {
                uncertain += 1;
            }
            let failure = matches!(tr.terminal(), Terminal::Terminated(TerminalReason::Failure));
            let cell = |s: &State| {
                (
                    s.values()[0].floor() as usize,
                    s.values()[1].floor() as usize,
                )
            };
            let (px, py) = cell(prev);
            let (nx, ny) = cell(tr.next_state());
            let shaping = leaf.map_or(0.0, |l| {
                let phi =
                    |b: Option<usize>| l.weight * b.map_or(-l.cap, |b| l.field.potential(b, l.cap));
                let from = scale.bucket_of(prev.values()).ok();
                let to = if failure {
                    None
                } else {
                    scale.bucket_of(tr.next_state().values()).ok()
                };
                phi(to) - phi(from)
            });
            shaping
                + if failure {
                    task.hazard_reward
                } else if (nx, ny) == (task.goal.x, task.goal.y) {
                    task.goal_reward
                } else if (nx, ny) == (px, py) {
                    task.wall_collision_reward
                } else {
                    task.step_reward
                }
        };

        let mut rng = DeterministicRng::new(
            self.config
                .seed
                .derive_seed(gogamen_core::StreamKey::Episode(tick.as_u64())),
        );
        let outcome = plan_fmc_on_version(&self.planner, model, &request, &mut rng, reward_fn);
        let mut trace = base_trace;
        trace.simulated_steps = simulated;
        trace.uncertain_steps = uncertain;

        match outcome {
            Err(ModelError::InvalidState { reason }) if reason.contains("no evidence") => {
                abstain(FallbackReason::UnseenPairRejected, trace)
            }
            Err(_) => abstain(FallbackReason::ModelError, trace),
            Ok((result, summary)) => {
                trace.status = Some(result.status());
                trace.transitions_used = summary.rollout_transitions_used();
                match result.status() {
                    PlanStatus::BudgetExhausted => abstain(FallbackReason::BudgetExhausted, trace),
                    PlanStatus::Cancelled => abstain(FallbackReason::Cancelled, trace),
                    PlanStatus::NoValidAction => abstain(FallbackReason::NoValidAction, trace),
                    PlanStatus::Completed => {
                        let too_uncertain = simulated > 0
                            && (uncertain as f64)
                                > self.config.max_uncertain_fraction * (simulated as f64);
                        match result.authorized_action() {
                            None => abstain(FallbackReason::NoValidAction, trace),
                            Some(_) if too_uncertain => {
                                abstain(FallbackReason::UncertainModel, trace)
                            }
                            Some(action) => {
                                trace.action_index = action.as_discrete().map_or(0, |a| a.index());
                                PlanGridDecision {
                                    action: action.clone(),
                                    trace,
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn grid_action(index: usize) -> Result<Action, DomainError> {
    let space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
    Action::discrete_in_space(index, &space)
}

/// Scale over cell coordinates: one bucket per grid cell (+ the failure bucket).
pub fn grid_cell_scale(map: &GridMap) -> Result<TabularScale, HiddenGridError> {
    let spec = BinSpec::uniform(
        "grid-cell",
        BinningVersion::new(1, 0),
        vec![
            (0.0, map.width() as f64, map.width()),
            (0.0, map.height() as f64, map.height()),
        ],
    )
    .map_err(|_| HiddenGridError::InvalidConfig("grid scale"))?;
    Ok(TabularScale::new(spec)?)
}

/// Collects `observations` transitions of random-policy experience from the real world into a
/// fresh learner. Hazard terminals are recorded as learned failure.
pub fn collect_hidden_experience(
    world: &GridWorldConfig,
    scale: &TabularScale,
    observations: usize,
    seed: MasterSeed,
    policy: LearningPolicy,
) -> Result<TabularTransitionLearner, HiddenGridError> {
    const EPISODE_CAP: u64 = 60;
    let mut learner =
        TabularTransitionLearner::new(scale.num_states(), GRID_ACTION_CARDINALITY, policy)?;
    let mut env = GridWorldEnv::new(world.clone())?;
    let space = env.action_space().clone();
    let mut rng = seed.derive(gogamen_core::StreamKey::Episode(0xE0));
    let mut ticks_in_episode = 0u64;
    for id in 0..observations {
        if env.terminal().is_terminal() || ticks_in_episode >= EPISODE_CAP {
            gogamen_core::LiveEnvironment::reset(&mut env).map_err(|_| {
                HiddenGridError::Domain(DomainError::InvalidTransition {
                    reason: "reset failed while collecting experience",
                })
            })?;
            ticks_in_episode = 0;
        }
        let before = env.position();
        let idx = (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
        let action = grid_action(idx)?;
        debug_assert!(space.validate(&action).is_ok());
        let outcome = env.apply(&action).map_err(|_| {
            HiddenGridError::Domain(DomainError::InvalidTransition {
                reason: "action rejected while collecting experience",
            })
        })?;
        ticks_in_episode += 1;
        let after = env.position();
        let sample = scale.sample_from_transition(
            &[before.x as f64 + 0.5, before.y as f64 + 0.5],
            idx,
            &[after.x as f64 + 0.5, after.y as f64 + 0.5],
            outcome.terminal(),
        )?;
        let oid = ObservationId::new(id as u64 + 1);
        let evidence = EvidenceRef::new(10_000 + id as u64);
        let obs = if sample.is_terminal_failure() {
            TransitionObservation::failure(
                oid,
                evidence,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        } else {
            TransitionObservation::new(
                oid,
                evidence,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        };
        learner.observe(&obs)?;
    }
    Ok(learner)
}

/// Builds the frozen model version and its adapter from a learner.
pub fn freeze_grid_model(
    learner: &TabularTransitionLearner,
    scale: TabularScale,
    config: LearnedModelConfig,
) -> Result<TabularLearnedModel, HiddenGridError> {
    let version = ModelVersion::freeze(learner, scale)?;
    Ok(TabularLearnedModel::new(version, config)?)
}

/// Outcome of one episode.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenEpisode {
    /// Goal reached.
    pub reached_goal: bool,
    /// Episode ended in terminal failure (hazard).
    pub failed: bool,
    /// Ticks executed.
    pub ticks: u64,
    /// Total reward collected in the real world.
    pub total_reward: f64,
    /// Simulated transitions consumed by planning (0 for non-planning policies).
    pub planning_transitions: u64,
    /// Decisions that fell back, by exact reason.
    pub fallbacks: BTreeMap<FallbackReason, u64>,
    /// Per-decision traces (empty for baselines).
    pub traces: Vec<PlanGridTrace>,
}

fn finish_episode(env: &GridWorldEnv, ticks: u64) -> HiddenEpisode {
    HiddenEpisode {
        reached_goal: env.terminal() == Terminal::Terminated(TerminalReason::GoalReached),
        failed: env.terminal() == Terminal::Terminated(TerminalReason::Failure),
        ticks,
        total_reward: env.total_reward(),
        planning_transitions: 0,
        fallbacks: BTreeMap::new(),
        traces: Vec::new(),
    }
}

fn apply_to_env(env: &mut GridWorldEnv, action: &Action) -> Result<(), HiddenGridError> {
    env.apply(action).map(|_| ()).map_err(|_| {
        HiddenGridError::Domain(DomainError::InvalidTransition {
            reason: "failed to apply action in grid environment",
        })
    })
}

/// Runs [`PlanGrid`] against the real world. The world only receives actions; the agent only
/// receives the observed cell.
pub fn run_plan_grid_episode(
    world: &GridWorldConfig,
    agent: &PlanGrid,
    max_ticks: u64,
) -> Result<HiddenEpisode, HiddenGridError> {
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut ticks = 0;
    let mut planning = 0u64;
    let mut fallbacks = BTreeMap::new();
    let mut traces = Vec::new();
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let decision = agent.decide(env.position(), Tick::new(ticks), None);
        planning += decision.trace.transitions_used as u64;
        if let Some(reason) = decision.trace.fallback {
            *fallbacks.entry(reason).or_insert(0) += 1;
        }
        apply_to_env(&mut env, &decision.action)?;
        traces.push(decision.trace);
        ticks += 1;
    }
    let mut ep = finish_episode(&env, ticks);
    ep.planning_transitions = planning;
    ep.fallbacks = fallbacks;
    ep.traces = traces;
    Ok(ep)
}

/// Learner-only baseline policies (no planning engine).
pub enum LearnerPolicy<'a> {
    /// Uniform random actions.
    Random(MasterSeed),
    /// Per decision, the action maximizing the model's expected negative Manhattan distance to
    /// the goal after one step (one `predict` per action, no lookahead).
    GreedyOnLearner(&'a TabularLearnedModel, GridTask),
}

/// Runs a learner-only baseline against the real world.
pub fn run_learner_policy_episode(
    world: &GridWorldConfig,
    policy: &LearnerPolicy<'_>,
    max_ticks: u64,
) -> Result<HiddenEpisode, HiddenGridError> {
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut rng = match policy {
        LearnerPolicy::Random(seed) => seed.derive(gogamen_core::StreamKey::Episode(0xB0)),
        LearnerPolicy::GreedyOnLearner(..) => {
            MasterSeed::new(0).derive(gogamen_core::StreamKey::Episode(0))
        }
    };
    let mut ticks = 0;
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let idx = match policy {
            LearnerPolicy::Random(_) => (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize,
            LearnerPolicy::GreedyOnLearner(model, task) => {
                greedy_action(model, task, env.position())
            }
        };
        apply_to_env(&mut env, &grid_action(idx)?)?;
        ticks += 1;
    }
    Ok(finish_episode(&env, ticks))
}

fn greedy_action(model: &TabularLearnedModel, task: &GridTask, pos: GridPos) -> usize {
    let scale = model.version().scale();
    let Ok(bucket) = scale.bucket_of(&[pos.x as f64 + 0.5, pos.y as f64 + 0.5]) else {
        return 0;
    };
    let mut best = (0usize, f64::NEG_INFINITY);
    for a in 0..GRID_ACTION_CARDINALITY {
        let Ok(pred) = model.predict(bucket, a) else {
            continue;
        };
        let mut score = 0.0;
        for (next, p) in pred.distribution.iter().enumerate() {
            let value = match scale.center_of(next) {
                Some(c) => {
                    let cell = GridPos::new(c[0].floor() as usize, c[1].floor() as usize);
                    -(cell.manhattan_distance(task.goal) as f64)
                }
                None => task.hazard_reward,
            };
            score += p * value;
        }
        if score > best.1 {
            best = (a, score);
        }
    }
    best.0
}

/// Privileged ceiling: the same [`FmcPlanner`] and budget over the real [`GridWorldModel`].
/// This policy reads the simulator and exists only to measure the distance to the ceiling.
pub fn run_privileged_fmc_episode(
    world: &GridWorldConfig,
    config: &PlanGridConfig,
    max_ticks: u64,
) -> Result<HiddenEpisode, HiddenGridError> {
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut model = GridWorldModel::new(world.clone(), config.seed)?;
    let planner = FmcPlanner::new(
        FmcPlannerConfig::new(
            config.population_size,
            config.horizon,
            world.dt,
            config.scoring,
            config.seed,
        )
        .map_err(|_| HiddenGridError::PlannerConfig)?,
    );
    let stay = grid_action(0)?;
    let mut rng =
        DeterministicRng::new(config.seed.derive_seed(gogamen_core::StreamKey::Episode(1)));
    let mut ticks = 0;
    let mut planning = 0u64;
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let request = PlanRequest::from_state(
            model.state().clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(config.budget_transitions),
        )
        .with_fallback(stay.clone())?;
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let (_, trace) = planner
            .plan_with_trace(
                &mut ctx,
                &request,
                |prev, _, tr| tr.next_state().values()[2] - prev.values()[2],
                |s| vec![s.values()[0], s.values()[1]],
            )
            .map_err(|_| HiddenGridError::PlannerConfig)?;
        planning += trace.rollout_transitions_used() as u64;
        let action = trace
            .first_action()
            .cloned()
            .unwrap_or_else(|| stay.clone());
        apply_to_env(&mut env, &action)?;
        model
            .step_in_place(&action)
            .map_err(|_| HiddenGridError::PlannerConfig)?;
        ticks += 1;
    }
    let mut ep = finish_episode(&env, ticks);
    ep.planning_transitions = planning;
    Ok(ep)
}

/// Seed split used by the experience curve.
pub use crate::experiment::SeedSplit;

/// Configuration of the experience curve.
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenCurveConfig {
    /// World (the hidden physics).
    pub world: GridWorldConfig,
    /// Numbers of random-experience transitions to learn from.
    pub observation_counts: Vec<usize>,
    /// Reserve (holdout) seeds; every run uses them blind.
    pub holdout_seeds: Vec<MasterSeed>,
    /// Planner settings (also the per-decision transition budget cap shared by all policies).
    pub plan: PlanGridConfig,
    /// Model adapter settings.
    pub model: LearnedModelConfig,
    /// Dirichlet prior of the learner (small values keep the model sharp with little data).
    pub prior_alpha: f64,
    /// Ticks per episode.
    pub max_ticks: u64,
}

impl HiddenCurveConfig {
    /// Default: the 7x5 small-detour map of the FMC lab tests (solvable within the planner
    /// horizon, so the privileged ceiling is meaningful), observation counts `0..=2000` and
    /// holdout seeds `101..=106`.
    pub fn new(world: GridWorldConfig) -> Self {
        let plan = PlanGridConfig::new(GridTask::from_world(&world));
        Self {
            world,
            observation_counts: vec![0, 50, 200, 800, 2000],
            holdout_seeds: (101..=106).map(MasterSeed::new).collect(),
            plan,
            model: LearnedModelConfig::new(0.5, 2, gogamen_core::UnseenPairPolicy::SamplePrior)
                .expect("valid default model config"),
            prior_alpha: 0.05,
            max_ticks: 30,
        }
    }
}

/// Aggregate of one policy at one experience level.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyStats {
    /// Fraction of episodes that reached the goal.
    pub goal_rate: f64,
    /// Fraction of episodes that ended in hazard failure.
    pub failure_rate: f64,
    /// Mean real-world return.
    pub mean_reward: f64,
    /// Mean simulated transitions per decision (planning cost).
    pub mean_transitions_per_decision: f64,
}

/// One point of the curve (number of observations -> viability / regret).
#[derive(Debug, Clone, PartialEq)]
pub struct HiddenCurvePoint {
    /// Observations the model learned from.
    pub observations: usize,
    /// Always [`SeedSplit::Holdout`].
    pub split: SeedSplit,
    /// `PlanGrid` over the learned model.
    pub planner: PlanGridStats,
    /// Uniform random policy.
    pub random: PolicyStats,
    /// Greedy policy on the learner (no planning).
    pub greedy: PolicyStats,
    /// Privileged ceiling (real `GridWorldModel`), independent of the observation count.
    pub ceiling: PolicyStats,
    /// `ceiling.mean_reward - planner.stats.mean_reward` (paired by seed, can be negative).
    pub regret_vs_ceiling: f64,
    /// Model checksums per holdout seed (frozen versions that were evaluated).
    pub model_checksums: Vec<u64>,
}

/// Planner aggregate plus abstention counts.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanGridStats {
    /// Common stats.
    pub stats: PolicyStats,
    /// Decisions that fell back, by exact reason.
    pub fallbacks: BTreeMap<FallbackReason, u64>,
    /// Total decisions.
    pub decisions: u64,
}

fn stats(eps: &[HiddenEpisode]) -> PolicyStats {
    let n = eps.len().max(1) as f64;
    let decisions: u64 = eps.iter().map(|e| e.ticks).sum();
    let transitions: u64 = eps.iter().map(|e| e.planning_transitions).sum();
    PolicyStats {
        goal_rate: eps.iter().filter(|e| e.reached_goal).count() as f64 / n,
        failure_rate: eps.iter().filter(|e| e.failed).count() as f64 / n,
        mean_reward: eps.iter().map(|e| e.total_reward).sum::<f64>() / n,
        mean_transitions_per_decision: transitions as f64 / decisions.max(1) as f64,
    }
}

/// Evaluates the experience curve on holdout seeds. Nothing is tuned on these seeds and no
/// threshold is applied: the report carries the measured numbers.
pub fn run_hidden_dynamics_curve(
    config: &HiddenCurveConfig,
) -> Result<Vec<HiddenCurvePoint>, HiddenGridError> {
    config.world.validate()?;
    let scale = grid_cell_scale(&config.world.map)?;
    let mut ceiling_eps = Vec::new();
    for seed in &config.holdout_seeds {
        let mut plan = config.plan.clone();
        plan.seed = *seed;
        ceiling_eps.push(run_privileged_fmc_episode(
            &config.world,
            &plan,
            config.max_ticks,
        )?);
    }
    let ceiling = stats(&ceiling_eps);

    let mut points = Vec::new();
    for &n in &config.observation_counts {
        let (mut planner_eps, mut random_eps, mut greedy_eps) =
            (Vec::new(), Vec::new(), Vec::new());
        let mut checksums = Vec::new();
        let mut fallbacks: BTreeMap<FallbackReason, u64> = BTreeMap::new();
        let mut decisions = 0;
        for seed in &config.holdout_seeds {
            let learner = collect_hidden_experience(
                &config.world,
                &scale,
                n,
                *seed,
                LearningPolicy::new(config.prior_alpha)?,
            )?;
            let model = freeze_grid_model(&learner, scale.clone(), config.model)?;
            checksums.push(model.version().checksum());
            let mut plan = config.plan.clone();
            plan.seed = *seed;
            let agent = PlanGrid::new(model.clone(), plan.clone())?;
            let ep = run_plan_grid_episode(&config.world, &agent, config.max_ticks)?;
            decisions += ep.ticks;
            for (reason, count) in &ep.fallbacks {
                *fallbacks.entry(*reason).or_insert(0) += count;
            }
            planner_eps.push(ep);
            random_eps.push(run_learner_policy_episode(
                &config.world,
                &LearnerPolicy::Random(*seed),
                config.max_ticks,
            )?);
            greedy_eps.push(run_learner_policy_episode(
                &config.world,
                &LearnerPolicy::GreedyOnLearner(&model, plan.task),
                config.max_ticks,
            )?);
        }
        let planner = stats(&planner_eps);
        points.push(HiddenCurvePoint {
            observations: n,
            split: SeedSplit::Holdout,
            regret_vs_ceiling: ceiling.mean_reward - planner.mean_reward,
            planner: PlanGridStats {
                stats: planner,
                fallbacks,
                decisions,
            },
            random: stats(&random_eps),
            greedy: stats(&greedy_eps),
            ceiling: ceiling.clone(),
            model_checksums: checksums,
        });
    }
    Ok(points)
}
