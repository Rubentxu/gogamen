//! Nested scales: the upper level emits the goal of the lower one (GOG-175, DEM-022).
//!
//! # What it does
//! Two levels run the SAME [`FmcPlanner`] with different `(scale, horizon)` pairs over the same
//! [`BoolWorld`]:
//!
//! 1. The **coarse** level plans over a [`ScaledModel`] of its working scale with the pure
//!    principle (default FMC reward: only the failure penalty, no task reward) and emits a
//!    macrostate: the modal final macro of the viable walkers that share the winning first action.
//! 2. The **fine** level plans over the finest scale (identity) with a goal term: `+goal_reward`
//!    for a viable transition that lands in a microstate of the emitted macro.
//!
//! # Working scale
//! [`select_scale`] returns the *coarsest* candidate that (a) does not hide a hard limit (no macro
//! mixes viable and inviable microstates) and (b) whose exact predictive error at the requested
//! horizon (`PredictiveError::quotient_max_abs_deviation` of [`measure_at_scale`]) is `<= eps`.
//! The finest candidate (identity, exact by construction) is the fallback, so a predictive
//! failure at a long horizon sends the level back to the fine representation. The predictive
//! error does not depend on the initial state, so the selection is computed once per agent.
//!
//! # Hard limits
//! Viability is defined on microstates. An aggregated scale never hides it: a scale whose macros
//! mix viable and inviable microstates is rejected ([`RejectReason::HidesHardLimit`]), the coarse
//! model reports `Failure` whenever a sampled microstate is inviable, a goal macro is never one
//! that contains an inviable microstate, and the fine level always plans on the exact
//! microstates, so it keeps every hard limit regardless of the goal it receives.
//!
//! # Preregistered decision rule (declared BEFORE the first evaluation run)
//! See [`PREREG`] and [`compare_levels`]. Reserve seeds are only evaluated once, after the
//! parameters below are frozen; development seeds (`0..8`) are used for sanity checks only.
//!
//! # Limits and non-claims
//! Tiny exact [`BoolWorld`]s only; the result says nothing about emergence, AGI or any real
//! environment. The coarse model is the uniform quotient of GOG-161 (it can be wrong, which is
//! what the predictive error measures). No online predictive monitor is implemented (the
//! failure is detected from the exact error at the requested horizon, not from live surprise).

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, ActionSpace, DeterministicRng, DomainError, FmcPlanner, FmcPlannerConfig,
    FmcScoringConfig, GenerativeModel, MasterSeed, ModelError, Observation, PlanBudget,
    PlanRequest, PlanningContext, RngSource, SimDuration, State, StepDt, Terminal, TerminalReason,
    Tick, Transition,
};

use crate::bool_world::{BoolWorld, BoolWorldError, MicroState};
use crate::multiscale::{MultiscaleConfig, MultiscaleError, Scale, measure_at_scale};

/// Preregistered protocol (frozen before the first reserve-seed run).
pub const PREREG: &str = "\
eps=0.05 (max quotient deviation at the requested horizon); coarse level pop=16,h=6; \
fine level pop=32,h=3; one-level baseline pop=64,h=3 (192 simulated transitions per decision in \
both arms, enforced as PlanBudget caps and reported as consumed); episodes of 24 steps, 3 \
episode seeds per world; development seeds 0..8 (sanity only), reserve seeds 1000..1040 \
(evaluated once). Metric: viable steps survived (higher is better), paired by (world, episode \
seed) and averaged per world. Verdict: Improves iff mean paired difference > 0 and > 2*SE \
(SE over worlds); Worse iff < 0 and < -2*SE; otherwise NoDifference. Non-improvement is \
reported as such.";

/// Errors of the nested-scale layer (fail-closed).
#[derive(Debug)]
pub enum NestedError {
    /// Invalid configuration.
    InvalidConfig(&'static str),
    /// Underlying world error.
    World(BoolWorldError),
    /// Underlying multiscale error.
    Multiscale(MultiscaleError),
    /// Underlying domain error.
    Domain(DomainError),
    /// Planner/model error.
    Model(ModelError),
}

impl fmt::Display for NestedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            Self::World(e) => write!(f, "world error: {e}"),
            Self::Multiscale(e) => write!(f, "multiscale error: {e}"),
            Self::Domain(e) => write!(f, "domain error: {e}"),
            Self::Model(e) => write!(f, "model error: {e}"),
        }
    }
}

impl Error for NestedError {}

impl From<BoolWorldError> for NestedError {
    fn from(e: BoolWorldError) -> Self {
        Self::World(e)
    }
}
impl From<MultiscaleError> for NestedError {
    fn from(e: MultiscaleError) -> Self {
        Self::Multiscale(e)
    }
}
impl From<DomainError> for NestedError {
    fn from(e: DomainError) -> Self {
        Self::Domain(e)
    }
}
impl From<ModelError> for NestedError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}

// ---------------------------------------------------------------------------------------------
// Scale selection
// ---------------------------------------------------------------------------------------------

/// Why a candidate scale was not admitted.
#[derive(Debug, Clone, PartialEq)]
pub enum RejectReason {
    /// A macro mixes viable and inviable microstates (it would hide a hard limit).
    HidesHardLimit,
    /// Predictive error at the requested horizon exceeds the budget.
    PredictiveError {
        /// Measured error.
        error: f64,
    },
    /// The exact measurement was not possible (fail-closed: not admitted).
    Unmeasurable,
}

/// Result of [`select_scale`].
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleSelection {
    /// Index into the candidate list ordered coarse -> fine (by macro count).
    pub index: usize,
    /// Label of the chosen scale.
    pub label: String,
    /// Version of the chosen scale.
    pub version: (u32, u32),
    /// Predictive error of the chosen scale at the horizon (`0.0` for the identity fallback).
    pub predictive_error: f64,
    /// Requested horizon.
    pub horizon: usize,
    /// Error budget.
    pub eps: f64,
    /// Coarser candidates rejected before the chosen one, with reasons.
    pub rejected: Vec<(String, RejectReason)>,
}

/// `true` if some macro of `scale` mixes viable and inviable microstates.
pub fn scale_hides_hard_limit(world: &BoolWorld, scale: &Scale) -> bool {
    let viable: BTreeMap<MicroState, bool> =
        world.state_specs().map(|s| (s.state, s.viable)).collect();
    (0..scale.macro_count()).any(|m| {
        let mut flags = scale
            .phi()
            .iter()
            .filter(|(_, id)| **id == m)
            .map(|(s, _)| viable[s]);
        let first = flags.next();
        !flags.all(|f| Some(f) == first)
    })
}

/// Picks the coarsest admissible scale among `candidates` for `horizon` (see module docs).
///
/// `candidates` may come in any order; the finest one (most macros) must be the identity
/// (`macro_count == micro count`) and is the exact fallback.
pub fn select_scale(
    world: &BoolWorld,
    candidates: &[Scale],
    initial: MicroState,
    horizon: usize,
    eps: f64,
) -> Result<ScaleSelection, NestedError> {
    if candidates.is_empty() || horizon == 0 || !eps.is_finite() || eps < 0.0 {
        return Err(NestedError::InvalidConfig(
            "need candidates, horizon >= 1 and finite eps >= 0",
        ));
    }
    let mut order: Vec<&Scale> = candidates.iter().collect();
    order.sort_by_key(|s| s.macro_count());
    let micro_count = world.state_specs().count();
    let finest = order.last().expect("non-empty");
    if finest.macro_count() != micro_count {
        return Err(NestedError::InvalidConfig(
            "the finest candidate must be the identity scale",
        ));
    }
    let mut rejected = Vec::new();
    let last = order.len() - 1;
    for (i, scale) in order.iter().enumerate() {
        if i == last {
            return Ok(ScaleSelection {
                index: i,
                label: scale.label().to_string(),
                version: scale.version(),
                predictive_error: 0.0,
                horizon,
                eps,
                rejected,
            });
        }
        if scale_hides_hard_limit(world, scale) {
            rejected.push((scale.label().to_string(), RejectReason::HidesHardLimit));
            continue;
        }
        match measure_at_scale(world, scale, &MultiscaleConfig::new(initial, horizon)) {
            Ok(m) => {
                let error = m.predictive.quotient_max_abs_deviation;
                if error <= eps {
                    return Ok(ScaleSelection {
                        index: i,
                        label: scale.label().to_string(),
                        version: scale.version(),
                        predictive_error: error,
                        horizon,
                        eps,
                        rejected,
                    });
                }
                rejected.push((
                    scale.label().to_string(),
                    RejectReason::PredictiveError { error },
                ));
            }
            Err(_) => rejected.push((scale.label().to_string(), RejectReason::Unmeasurable)),
        }
    }
    unreachable!("the last candidate always returns")
}

// ---------------------------------------------------------------------------------------------
// Scaled generative model
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Row {
    viable: bool,
    actions: Vec<Vec<(usize, f64)>>,
}

/// [`GenerativeModel`] of a [`BoolWorld`] seen at a [`Scale`].
///
/// State values are `[macro id]`. A step picks a viable member of the macro uniformly (the GOG-161
/// quotient), applies the action (index wrapped modulo the member's action count) and reports the
/// macro of the sampled successor; the terminal is `Failure` iff that successor is inviable.
/// With the identity scale it is the exact world.
#[derive(Debug, Clone)]
pub struct ScaledModel {
    space: ActionSpace,
    dt: StepDt,
    rows: Vec<Row>,
    macro_of: Vec<usize>,
    members: Vec<Vec<usize>>,
    index_of: BTreeMap<MicroState, usize>,
    micros: Vec<MicroState>,
}

impl ScaledModel {
    /// Builds the model of `world` at `scale`.
    pub fn new(world: &BoolWorld, scale: &Scale, dt: StepDt) -> Result<Self, NestedError> {
        let micros: Vec<MicroState> = world.state_specs().map(|s| s.state).collect();
        let index_of: BTreeMap<MicroState, usize> =
            micros.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let k = world
            .state_specs()
            .map(|s| s.actions.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let rows: Vec<Row> = world
            .state_specs()
            .map(|s| Row {
                viable: s.viable,
                actions: s
                    .actions
                    .iter()
                    .map(|a| a.iter().map(|(d, p)| (index_of[d], *p)).collect())
                    .collect(),
            })
            .collect();
        let macro_of: Vec<usize> = micros
            .iter()
            .map(|s| {
                scale
                    .phi()
                    .get(s)
                    .copied()
                    .ok_or(NestedError::InvalidConfig("scale does not cover the world"))
            })
            .collect::<Result<_, _>>()?;
        let mut members = vec![Vec::new(); scale.macro_count()];
        for (i, m) in macro_of.iter().enumerate() {
            members[*m].push(i);
        }
        Ok(Self {
            space: ActionSpace::discrete(k)?,
            dt,
            rows,
            macro_of,
            members,
            index_of,
            micros,
        })
    }

    /// Simulation step.
    pub fn dt(&self) -> StepDt {
        self.dt
    }

    /// Action space (`k` = maximum action count of the world).
    pub fn space(&self) -> &ActionSpace {
        &self.space
    }

    /// Macro id of a microstate at this scale.
    pub fn macro_id(&self, micro: MicroState) -> Option<usize> {
        self.index_of.get(&micro).map(|i| self.macro_of[*i])
    }

    /// Microstates of a macro id.
    pub fn members_of(&self, macro_id: usize) -> Vec<MicroState> {
        self.members
            .get(macro_id)
            .map(|v| v.iter().map(|i| self.micros[*i]).collect())
            .unwrap_or_default()
    }

    /// `true` if every member of the macro is viable (never contains a hard limit).
    pub fn macro_is_hard_limit_free(&self, macro_id: usize) -> bool {
        self.members
            .get(macro_id)
            .is_some_and(|v| !v.is_empty() && v.iter().all(|i| self.rows[*i].viable))
    }

    /// Initial state for the observed microstate.
    pub fn state_of(&self, micro: MicroState) -> Result<State, NestedError> {
        let id = self
            .macro_id(micro)
            .ok_or(NestedError::InvalidConfig("microstate not in the world"))?;
        Ok(State::new(Tick::ZERO, SimDuration::ZERO, vec![id as f64])?)
    }

    fn sample_successor(&self, micro: usize, action: usize, rng: &mut dyn RngSource) -> usize {
        let acts = &self.rows[micro].actions;
        if acts.is_empty() {
            return micro;
        }
        let dests = &acts[action % acts.len()];
        let u = rng.next_f64();
        let mut acc = 0.0;
        for (d, p) in dests {
            acc += p;
            if u < acc {
                return *d;
            }
        }
        dests.last().map_or(micro, |(d, _)| *d)
    }
}

impl GenerativeModel for ScaledModel {
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
        let id = state.values()[0] as usize;
        let viable: Vec<usize> = self
            .members
            .get(id)
            .map(|v| v.iter().copied().filter(|i| self.rows[*i].viable).collect())
            .unwrap_or_default();
        if viable.is_empty() {
            return Err(ModelError::TerminalState);
        }
        let micro = viable[rng.next_usize_unbiased(viable.len())];
        let a = action.as_discrete().map_or(0, |d| d.index());
        let dest = self.sample_successor(micro, a, rng);
        let terminal = if self.rows[dest].viable {
            Terminal::Ongoing
        } else {
            Terminal::Terminated(TerminalReason::Failure)
        };
        let vals = vec![self.macro_of[dest] as f64];
        let next = state
            .advance(dt, vals.clone())
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(next.tick(), next.elapsed(), vals)
            .map_err(ModelError::TransitionFailed)?;
        Transition::new(state.tick(), dt, next, obs, terminal).map_err(ModelError::TransitionFailed)
    }
}

// ---------------------------------------------------------------------------------------------
// Agents and decision records
// ---------------------------------------------------------------------------------------------

/// Level of the hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Upper level: emits the goal.
    Coarse,
    /// Lower level: receives the goal and acts.
    Fine,
}

/// Goal emitted by the coarse level (a macro at its working scale).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GoalRecord {
    /// Scale label the macro belongs to.
    pub scale_label: String,
    /// Macro id.
    pub macro_id: usize,
    /// Member microstates (the fine level rewards landing in them).
    pub members: Vec<MicroState>,
}

/// Uncertainty recorded with a decision.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uncertainty {
    /// Predictive error of the level's scale at its horizon.
    pub predictive_error: f64,
    /// Fraction of candidate walkers that voted for the emitted goal (coarse level only).
    pub goal_support: Option<f64>,
    /// Normalised diversity of first actions in the FMC population.
    pub action_diversity: f64,
}

/// Record of one level's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelDecision {
    /// Level.
    pub level: Level,
    /// Scale label used by the level.
    pub scale_label: String,
    /// Scale version.
    pub scale_version: (u32, u32),
    /// Horizon.
    pub horizon: usize,
    /// Goal emitted (coarse) or received (fine); `None` if no viable goal existed.
    pub goal: Option<GoalRecord>,
    /// Uncertainty.
    pub uncertainty: Uncertainty,
    /// Simulated transitions consumed.
    pub transitions_used: usize,
    /// First action recommended by this level.
    pub action_index: usize,
    /// `true` if the working scale of this level mixes viable and inviable microstates.
    pub hides_hard_limit: bool,
    /// Candidate scales rejected for this level and why.
    pub rejected: Vec<(String, RejectReason)>,
    /// Number of final walkers excluded as goal votes because their macro contains a hard limit.
    pub goal_votes_excluded_hard_limit: usize,
}

/// A full nested decision.
#[derive(Debug, Clone, PartialEq)]
pub struct NestedDecision {
    /// Executed action index (the fine level's).
    pub action_index: usize,
    /// One record per level (coarse first for the two-level agent).
    pub levels: Vec<LevelDecision>,
}

impl NestedDecision {
    /// Total simulated transitions of this decision.
    pub fn transitions_used(&self) -> usize {
        self.levels.iter().map(|l| l.transitions_used).sum()
    }
}

/// Parameters of one planning level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LevelParams {
    /// FMC population.
    pub population: usize,
    /// FMC horizon.
    pub horizon: usize,
}

impl LevelParams {
    /// Transition budget of a level (`population * horizon`).
    pub const fn budget(&self) -> usize {
        self.population * self.horizon
    }
}

/// Configuration of the nested agent.
#[derive(Debug, Clone)]
pub struct NestedConfig {
    /// Predictive-error budget `eps`.
    pub eps: f64,
    /// Coarse level parameters.
    pub coarse: LevelParams,
    /// Fine level parameters.
    pub fine: LevelParams,
    /// Reward for a viable transition landing in the goal macro.
    pub goal_reward: f64,
    /// Master seed.
    pub seed: u64,
}

impl NestedConfig {
    /// The preregistered configuration ([`PREREG`]).
    pub fn preregistered(seed: u64) -> Self {
        Self {
            eps: 0.05,
            coarse: LevelParams {
                population: 16,
                horizon: 6,
            },
            fine: LevelParams {
                population: 32,
                horizon: 3,
            },
            goal_reward: 1.0,
            seed,
        }
    }

    /// Total transitions per decision of the two-level agent.
    pub const fn total_budget(&self) -> usize {
        self.coarse.budget() + self.fine.budget()
    }
}

fn dt() -> StepDt {
    StepDt::from_millis_f64(100.0).expect("100 ms is a valid StepDt")
}

fn planner(p: LevelParams, seed: u64, capture: bool) -> Result<FmcPlanner, NestedError> {
    let cfg = FmcPlannerConfig::new(
        p.population,
        p.horizon,
        dt(),
        FmcScoringConfig::default(),
        MasterSeed::new(seed),
    )
    .map_err(|_| NestedError::InvalidConfig("invalid planner parameters"))?;
    Ok(FmcPlanner::new(cfg).with_walker_capture(capture))
}

fn mix(seed: u64, tick: u64, salt: u64) -> u64 {
    let mut z = seed
        .wrapping_add(tick.wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .wrapping_add(salt.wrapping_mul(0xbf58_476d_1ce4_e5b9));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Two-level agent: the coarse level emits a macro goal, the fine level receives it.
#[derive(Debug, Clone)]
pub struct NestedAgent {
    cfg: NestedConfig,
    coarse_model: ScaledModel,
    fine_model: ScaledModel,
    coarse_selection: ScaleSelection,
    coarse_hides_hard_limit: bool,
}

impl NestedAgent {
    /// Builds the agent; `candidates` are the available scales (must include the identity).
    pub fn new(
        world: &BoolWorld,
        candidates: &[Scale],
        initial: MicroState,
        cfg: NestedConfig,
    ) -> Result<Self, NestedError> {
        let selection = select_scale(world, candidates, initial, cfg.coarse.horizon, cfg.eps)?;
        let mut order: Vec<&Scale> = candidates.iter().collect();
        order.sort_by_key(|s| s.macro_count());
        let coarse_scale = order[selection.index];
        let fine_scale = order[order.len() - 1];
        Ok(Self {
            coarse_hides_hard_limit: scale_hides_hard_limit(world, coarse_scale),
            coarse_model: ScaledModel::new(world, coarse_scale, dt())?,
            fine_model: ScaledModel::new(world, fine_scale, dt())?,
            coarse_selection: selection,
            cfg,
        })
    }

    /// Selection made for the coarse level.
    pub fn coarse_selection(&self) -> &ScaleSelection {
        &self.coarse_selection
    }

    /// Decides for the observed microstate at `tick`. Deterministic in `(seed, observed, tick)`.
    pub fn decide(&self, observed: MicroState, tick: u64) -> Result<NestedDecision, NestedError> {
        // --- coarse level: pure principle, emits a macro goal ------------------------------
        let cp = planner(self.cfg.coarse, mix(self.cfg.seed, tick, 1), true)?;
        let cstate = self.coarse_model.state_of(observed)?;
        let creq = PlanRequest::from_state(
            cstate,
            self.coarse_model.space().clone(),
            PlanBudget::with_transitions(self.cfg.coarse.budget()),
        );
        let mut crng = DeterministicRng::new(mix(self.cfg.seed, tick, 2));
        let (cres, ctrace) = {
            let mut ctx = PlanningContext::new(&self.coarse_model, &mut crng);
            cp.plan_with_context(&mut ctx, &creq)?
        };
        let c_action = cres
            .authorized_action()
            .and_then(|a| a.as_discrete())
            .map_or(0, |d| d.index());
        let mut votes: BTreeMap<usize, usize> = BTreeMap::new();
        let mut excluded = 0usize;
        let mut considered = 0usize;
        for (first, state) in ctrace.final_walkers().unwrap_or(&[]) {
            if first
                .as_ref()
                .and_then(|a| a.as_discrete())
                .map(|d| d.index())
                != Some(c_action)
            {
                continue;
            }
            let id = state.values()[0] as usize;
            if self.coarse_model.macro_is_hard_limit_free(id) {
                *votes.entry(id).or_insert(0) += 1;
                considered += 1;
            } else {
                excluded += 1;
            }
        }
        // Modal macro; ties -> lowest id (BTreeMap order + strict `>`).
        let mut best: Option<(usize, usize)> = None;
        for (id, n) in &votes {
            if best.is_none_or(|(_, bn)| *n > bn) {
                best = Some((*id, *n));
            }
        }
        let goal = best.map(|(id, _)| GoalRecord {
            scale_label: self.coarse_selection.label.clone(),
            macro_id: id,
            members: self.coarse_model.members_of(id),
        });
        let support = best.map(|(_, n)| n as f64 / considered.max(1) as f64);
        let coarse_rec = LevelDecision {
            level: Level::Coarse,
            scale_label: self.coarse_selection.label.clone(),
            scale_version: self.coarse_selection.version,
            horizon: self.cfg.coarse.horizon,
            goal: goal.clone(),
            uncertainty: Uncertainty {
                predictive_error: self.coarse_selection.predictive_error,
                goal_support: support,
                action_diversity: ctrace.action_diversity(),
            },
            transitions_used: ctrace.rollout_transitions_used(),
            action_index: c_action,
            hides_hard_limit: self.coarse_hides_hard_limit,
            rejected: self.coarse_selection.rejected.clone(),
            goal_votes_excluded_hard_limit: excluded,
        };

        // --- fine level: same planner type, goal term, exact microstates -------------------
        let fp = planner(self.cfg.fine, mix(self.cfg.seed, tick, 3), false)?;
        let fstate = self.fine_model.state_of(observed)?;
        let freq = PlanRequest::from_state(
            fstate,
            self.fine_model.space().clone(),
            PlanBudget::with_transitions(self.cfg.fine.budget()),
        );
        let goal_ids: Vec<usize> = goal
            .as_ref()
            .map(|g| {
                g.members
                    .iter()
                    .filter_map(|m| self.fine_model.macro_id(*m))
                    .collect()
            })
            .unwrap_or_default();
        let goal_reward = self.cfg.goal_reward;
        let reward = move |_p: &State, _a: &Action, t: &Transition| -> f64 {
            match t.terminal() {
                Terminal::Terminated(
                    TerminalReason::Failure | TerminalReason::InvariantViolated,
                ) => -1.0,
                _ => {
                    let id = t.next_state().values()[0] as usize;
                    if goal_ids.contains(&id) {
                        goal_reward
                    } else {
                        0.0
                    }
                }
            }
        };
        let mut frng = DeterministicRng::new(mix(self.cfg.seed, tick, 4));
        let (fres, ftrace) = {
            let mut ctx = PlanningContext::new(&self.fine_model, &mut frng);
            fp.plan_with_trace(&mut ctx, &freq, reward, |s| s.values().to_vec())?
        };
        let f_action = fres
            .authorized_action()
            .and_then(|a| a.as_discrete())
            .map_or(0, |d| d.index());
        let fine_rec = LevelDecision {
            level: Level::Fine,
            scale_label: "identity".to_string(),
            scale_version: (1, 0),
            horizon: self.cfg.fine.horizon,
            goal,
            uncertainty: Uncertainty {
                predictive_error: 0.0,
                goal_support: support,
                action_diversity: ftrace.action_diversity(),
            },
            transitions_used: ftrace.rollout_transitions_used(),
            action_index: f_action,
            hides_hard_limit: false,
            rejected: Vec::new(),
            goal_votes_excluded_hard_limit: 0,
        };
        Ok(NestedDecision {
            action_index: f_action,
            levels: vec![coarse_rec, fine_rec],
        })
    }
}

/// One-level baseline: the same planner, fine scale, pure principle, equal total budget.
#[derive(Debug, Clone)]
pub struct OneLevelAgent {
    params: LevelParams,
    model: ScaledModel,
    seed: u64,
}

impl OneLevelAgent {
    /// Builds the baseline over the identity scale of `world`.
    pub fn new(
        world: &BoolWorld,
        identity: &Scale,
        params: LevelParams,
        seed: u64,
    ) -> Result<Self, NestedError> {
        Ok(Self {
            params,
            model: ScaledModel::new(world, identity, dt())?,
            seed,
        })
    }

    /// Decides for the observed microstate; returns `(action index, transitions used)`.
    pub fn decide(&self, observed: MicroState, tick: u64) -> Result<(usize, usize), NestedError> {
        let p = planner(self.params, mix(self.seed, tick, 5), false)?;
        let req = PlanRequest::from_state(
            self.model.state_of(observed)?,
            self.model.space().clone(),
            PlanBudget::with_transitions(self.params.budget()),
        );
        let mut rng = DeterministicRng::new(mix(self.seed, tick, 6));
        let (res, trace) = {
            let mut ctx = PlanningContext::new(&self.model, &mut rng);
            p.plan_with_context(&mut ctx, &req)?
        };
        let a = res
            .authorized_action()
            .and_then(|a| a.as_discrete())
            .map_or(0, |d| d.index());
        Ok((a, trace.rollout_transitions_used()))
    }
}

// ---------------------------------------------------------------------------------------------
// Fixture worlds
// ---------------------------------------------------------------------------------------------

/// Kind of generated block world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlockWorldKind {
    /// Exactly lumpable on the block partition.
    Lumpable,
    /// One microstate leaks `leak` probability mass to a different block (non-lumpable).
    Leaky {
        /// Leaked mass.
        leak: f64,
    },
    /// One microstate of a safe block is inviable: the block scale would hide a hard limit.
    MixedHardLimit,
}

/// A generated world with its block scale and identity scale.
#[derive(Debug, Clone)]
pub struct BlockWorld {
    /// The world.
    pub world: BoolWorld,
    /// Coarse partition: one macro per block.
    pub blocks: Scale,
    /// Identity scale.
    pub identity: Scale,
    /// Viable microstates usable as starts.
    pub starts: Vec<MicroState>,
}

const BLOCKS: usize = 5;
const BLOCK_SIZE: usize = 3;
const ACTIONS: usize = 3;

/// Deterministic block world: `BLOCKS` blocks of `BLOCK_SIZE` microstates, the last block is an
/// inviable absorbing trap. Block-level dynamics are random per `seed`; microstate `j` of a block
/// goes to microstate `(j + a) % BLOCK_SIZE` of the destination block.
pub fn block_world(seed: u64, kind: BlockWorldKind) -> Result<BlockWorld, NestedError> {
    let mut state = seed ^ 0xD1B5_4A32_D192_ED03;
    let mut next = move || {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let trap = BLOCKS - 1;
    // kernel[b][a] = (d1, d2, p2)
    let mut kernel = vec![vec![(0usize, 0usize, 0.0f64); ACTIONS]; BLOCKS - 1];
    for row in kernel.iter_mut() {
        for cell in row.iter_mut() {
            let d1 = (next() % (BLOCKS as u64 - 1)) as usize;
            let d2 = (next() % BLOCKS as u64) as usize;
            let p2 = if next() % 2 == 0 { 0.25 } else { 0.5 };
            *cell = (d1, d2, p2);
        }
    }
    let ms = |b: usize, j: usize| MicroState((b * BLOCK_SIZE + j) as u8);
    let mut world = BoolWorld::new(seed);
    let mut phi = BTreeMap::new();
    let mixed = (1usize, 2usize); // (block, member) made inviable in MixedHardLimit worlds
    #[allow(clippy::needless_range_loop)] // trap block has no kernel row
    for b in 0..BLOCKS {
        for j in 0..BLOCK_SIZE {
            phi.insert(ms(b, j), b);
            if b == trap || (kind == BlockWorldKind::MixedHardLimit && (b, j) == mixed) {
                world = world.with_state(ms(b, j), false, Vec::new())?;
                continue;
            }
            let mut actions = Vec::new();
            for (a, &(d1, d2, p2)) in kernel[b].iter().enumerate() {
                let t1 = ms(d1, (j + a) % BLOCK_SIZE);
                let t2 = ms(d2, (j + a) % BLOCK_SIZE);
                let mut dests: BTreeMap<MicroState, f64> = BTreeMap::new();
                *dests.entry(t1).or_insert(0.0) += 1.0 - p2;
                *dests.entry(t2).or_insert(0.0) += p2;
                if let BlockWorldKind::Leaky { leak } = kind
                    && (b, j, a) == (0, 0, 0)
                {
                    let leak_to = ms((d1 + 1) % (BLOCKS - 1), (j + a) % BLOCK_SIZE);
                    *dests.entry(t1).or_insert(0.0) -= leak.min(1.0 - p2);
                    *dests.entry(leak_to).or_insert(0.0) += leak.min(1.0 - p2);
                }
                actions.push(dests.into_iter().filter(|(_, p)| *p > 1e-12).collect());
            }
            world = world.with_state(ms(b, j), true, actions)?;
        }
    }
    let starts: Vec<MicroState> = world
        .state_specs()
        .filter(|s| s.viable)
        .map(|s| s.state)
        .collect();
    let blocks = Scale::from_partition("blocks", &world, phi)?;
    let identity = Scale::identity(&world);
    Ok(BlockWorld {
        world,
        blocks,
        identity,
        starts,
    })
}

// ---------------------------------------------------------------------------------------------
// Episodes and the two-level vs one-level comparison
// ---------------------------------------------------------------------------------------------

/// Episode length of the preregistered evaluation.
pub const EPISODE_LEN: u64 = 24;
/// Episode seeds per world.
pub const EPISODES_PER_WORLD: u64 = 3;

/// Outcome of one episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpisodeOutcome {
    /// Viable steps survived (at most [`EPISODE_LEN`]).
    pub survived: u64,
    /// Total simulated transitions consumed by the planner.
    pub transitions: u64,
    /// Decisions taken.
    pub decisions: u64,
}

fn true_step(
    model: &ScaledModel,
    micro: MicroState,
    action: usize,
    rng: &mut DeterministicRng,
) -> MicroState {
    let i = model.index_of[&micro];
    model.micros[model.sample_successor(i, action, rng)]
}

fn viable(world: &BoolWorld, s: MicroState) -> bool {
    world.state_specs().any(|v| v.state == s && v.viable)
}

/// Which arm to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// One fine level, `pop = total_budget / fine.horizon`.
    OneLevel,
    /// Coarse level emits the goal, fine level acts.
    TwoLevel,
}

/// Agents of one world, built once (selection is episode-independent).
#[derive(Debug, Clone)]
pub struct WorldAgents {
    truth: ScaledModel,
    nested: NestedAgent,
    one: OneLevelAgent,
}

impl WorldAgents {
    /// Builds both arms for `bw`; `start` only seeds the (initial-independent) scale selection.
    pub fn new(
        bw: &BlockWorld,
        cfg: &NestedConfig,
        start: MicroState,
    ) -> Result<Self, NestedError> {
        let candidates = [bw.blocks.clone(), bw.identity.clone()];
        Ok(Self {
            truth: ScaledModel::new(&bw.world, &bw.identity, dt())?,
            nested: NestedAgent::new(&bw.world, &candidates, start, cfg.clone())?,
            one: OneLevelAgent::new(
                &bw.world,
                &bw.identity,
                LevelParams {
                    population: cfg.total_budget() / cfg.fine.horizon,
                    horizon: cfg.fine.horizon,
                },
                cfg.seed,
            )?,
        })
    }

    /// The two-level agent.
    pub fn nested(&self) -> &NestedAgent {
        &self.nested
    }
}

/// Runs one episode of `arm` on `bw` from `start`, with true dynamics sampled from the world.
pub fn run_episode(
    bw: &BlockWorld,
    agents: &WorldAgents,
    arm: Arm,
    cfg: &NestedConfig,
    start: MicroState,
    episode_seed: u64,
) -> Result<EpisodeOutcome, NestedError> {
    let mut rng = DeterministicRng::new(mix(cfg.seed, episode_seed, 99));
    let mut pos = start;
    let mut out = EpisodeOutcome {
        survived: 0,
        transitions: 0,
        decisions: 0,
    };
    for tick in 0..EPISODE_LEN {
        let (a, used) = match arm {
            Arm::OneLevel => agents.one.decide(pos, tick)?,
            Arm::TwoLevel => {
                let d = agents.nested.decide(pos, tick)?;
                (d.action_index, d.transitions_used())
            }
        };
        out.transitions += used as u64;
        out.decisions += 1;
        pos = true_step(&agents.truth, pos, a, &mut rng);
        if !viable(&bw.world, pos) {
            return Ok(out);
        }
        out.survived += 1;
    }
    Ok(out)
}

/// World kind assigned to a world seed: `seed % 5` in {0,1,2} lumpable, 3 leaky(1/8), 4 mixed.
pub fn kind_for_seed(seed: u64) -> BlockWorldKind {
    match seed % 5 {
        3 => BlockWorldKind::Leaky { leak: 0.125 },
        4 => BlockWorldKind::MixedHardLimit,
        _ => BlockWorldKind::Lumpable,
    }
}

/// Verdict of the preregistered rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelsVerdict {
    /// Two levels survive significantly longer.
    Improves,
    /// Two levels survive significantly less.
    Worse,
    /// Neither (inside 2 standard errors).
    NoDifference,
}

/// Per-world row of the comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldRow {
    /// World seed.
    pub seed: u64,
    /// Kind.
    pub kind: BlockWorldKind,
    /// Label of the coarse level's chosen scale.
    pub coarse_scale: String,
    /// Mean survival, one level.
    pub one: f64,
    /// Mean survival, two levels.
    pub two: f64,
}

/// Report of [`compare_levels`].
#[derive(Debug, Clone, PartialEq)]
pub struct LevelsReport {
    /// Rows per world.
    pub rows: Vec<WorldRow>,
    /// Mean survival, one level.
    pub mean_one: f64,
    /// Mean survival, two levels.
    pub mean_two: f64,
    /// Mean paired difference two - one.
    pub mean_diff: f64,
    /// Standard error of the paired difference over worlds.
    pub se: f64,
    /// Mean simulated transitions per decision, one level.
    pub transitions_per_decision_one: f64,
    /// Mean simulated transitions per decision, two levels.
    pub transitions_per_decision_two: f64,
    /// Worlds where the coarse level used the block scale.
    pub worlds_with_coarse_scale: usize,
    /// Verdict.
    pub verdict: LevelsVerdict,
}

/// Preregistered comparison over `seeds` (see [`PREREG`]).
pub fn compare_levels(seeds: &[u64]) -> Result<LevelsReport, NestedError> {
    let mut rows = Vec::new();
    let (mut t1, mut t2, mut d1, mut d2) = (0u64, 0u64, 0u64, 0u64);
    let mut coarse_used = 0usize;
    for &seed in seeds {
        let kind = kind_for_seed(seed);
        let bw = block_world(seed, kind)?;
        let cfg = NestedConfig::preregistered(seed);
        let start = bw.starts[(mix(seed, 0, 7) % bw.starts.len() as u64) as usize];
        let agents = WorldAgents::new(&bw, &cfg, start)?;
        let coarse_label = agents.nested().coarse_selection().label.clone();
        if coarse_label == "blocks" {
            coarse_used += 1;
        }
        let (mut one, mut two) = (0.0, 0.0);
        for ep in 0..EPISODES_PER_WORLD {
            let a = run_episode(&bw, &agents, Arm::OneLevel, &cfg, start, ep)?;
            let b = run_episode(&bw, &agents, Arm::TwoLevel, &cfg, start, ep)?;
            one += a.survived as f64;
            two += b.survived as f64;
            t1 += a.transitions;
            d1 += a.decisions;
            t2 += b.transitions;
            d2 += b.decisions;
        }
        rows.push(WorldRow {
            seed,
            kind,
            coarse_scale: coarse_label,
            one: one / EPISODES_PER_WORLD as f64,
            two: two / EPISODES_PER_WORLD as f64,
        });
    }
    let n = rows.len().max(1) as f64;
    let mean_one = rows.iter().map(|r| r.one).sum::<f64>() / n;
    let mean_two = rows.iter().map(|r| r.two).sum::<f64>() / n;
    let diffs: Vec<f64> = rows.iter().map(|r| r.two - r.one).collect();
    let mean_diff = diffs.iter().sum::<f64>() / n;
    let var = if rows.len() > 1 {
        diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let se = (var / n).sqrt();
    let verdict = if mean_diff > 0.0 && mean_diff > 2.0 * se {
        LevelsVerdict::Improves
    } else if mean_diff < 0.0 && mean_diff < -2.0 * se {
        LevelsVerdict::Worse
    } else {
        LevelsVerdict::NoDifference
    };
    Ok(LevelsReport {
        rows,
        mean_one,
        mean_two,
        mean_diff,
        se,
        transitions_per_decision_one: t1 as f64 / d1.max(1) as f64,
        transitions_per_decision_two: t2 as f64 / d2.max(1) as f64,
        worlds_with_coarse_scale: coarse_used,
        verdict,
    })
}
