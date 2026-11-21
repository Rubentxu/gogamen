//! GOG-203: the closed-loop Grid arms missing from the GOG-176 first slice.
//!
//! Three small, auditable policies, each with an explicit access level (declared in
//! [`crate::emergence::arm_table`]):
//!
//! - [`entropic_pure_decision`] — reward-free entropic controller on Grid. It is the Grid
//!   adaptation of the Kart `EntropicPureController` decision rule: candidate actions are the
//!   five discrete grid moves, each is expanded into `population` rollouts of `horizon` steps
//!   (root action first, uniform random tail) over the **true** [`GridWorldModel`], and the
//!   action maximizing `H(final cells of viable futures) + log2(1 + viable futures)` is chosen.
//!   No reward, goal weight or map layout is read: only the observed cell and the dynamics.
//! - [`empowerment_decision`] — closed-loop empowerment on Grid. It uses a [`TabularLearnedModel`]
//!   frozen from executed experience (no simulator access) and scores each action by its
//!   contribution to the control information `I(A; S_T)`: `KL(p(S_T | a) || p(S_T))` in bits.
//!   The reported `entropy_gain_bits` is the plug-in mutual information of the decision.
//! - [`GridCausalAgent`] — causal-rule learner. It records executed `(cell, action) -> cell`
//!   outcomes, induces typed [`CausalRule`] hypotheses (from `gogamen_core::causal_rule`) with
//!   their evidence, and acts only when the induced rule applies **and** the harness grants
//!   [`ActionPermission`]. The rule never grants itself permission; without a permitted,
//!   evidence-backed rule the arm falls back to a seeded random move (counted, never hidden).
//!
//! # What is NOT claimed
//! These are bounded lab policies. Entropy maximization is not intelligence, mutual information
//! over a short rollout is not exact empowerment, and an induced rule is a hypothesis, not a
//! proven law. Negative outcomes (no goal, rule transfer failures) are reported as measured.

use std::collections::BTreeMap;

use gogamen_core::causal_rule::{
    ActionPermission, CausalRule, CausalRuleError, Condition, Effect, Observed, Outcome,
    OutcomeLabel, RuleContext, RuleEvaluation, RuleId, RuleValue, TemporalScope, VariableName,
};
use gogamen_core::{
    Action, ActionSpace, Actuator, DeterministicRng, DomainError, EpistemicStatus, EvidenceRef,
    GenerativeModel, GuardDecision, LearnedModelError, LiveEnvironment, MasterSeed, ModelError,
    RngSource, SimDuration, State, StreamKey, TabularLearnedModel, Terminal, TerminalReason, Tick,
    shannon_entropy_bits,
};
use gogamen_lab_grid::hidden_dynamics::HiddenEpisode;
use gogamen_lab_grid::{
    GRID_ACTION_CARDINALITY, GridPos, GridWorldConfig, GridWorldEnv, GridWorldModel,
};

/// Arm label of the reward-free entropic Grid controller.
pub const GRID_ENTROPIC_PURE_ARM: &str = "entropic_pure";
/// Arm label of the learned-model empowerment controller.
pub const GRID_EMPOWERMENT_ARM: &str = "empowerment_learned";
/// Arm label of the causal-rule learner.
pub const GRID_CAUSAL_RULE_ARM: &str = "causal_rule_learned";
/// The three GOG-203 arms, in arm-table order.
pub const GOG_203_ARMS: [&str; 3] = [
    GRID_ENTROPIC_PURE_ARM,
    GRID_EMPOWERMENT_ARM,
    GRID_CAUSAL_RULE_ARM,
];

/// Rollouts per candidate action of the entropic Grid controller.
pub const GRID_ENTROPIC_POPULATION: usize = 8;
/// Rollout horizon (steps) of the entropic Grid controller.
pub const GRID_ENTROPIC_HORIZON: usize = 4;
/// Rollouts per candidate action of the empowerment controller.
pub const GRID_EMPOWERMENT_POPULATION: usize = 10;
/// Rollout horizon (steps) of the empowerment controller.
pub const GRID_EMPOWERMENT_HORIZON: usize = 5;
/// Hard cap on provenance references kept per induced rule (matches the core limit).
pub const GRID_CAUSAL_MAX_PROVENANCE: usize = 16;
/// Hard cap on induced-experience evaluation episodes (same cap as the Grid learners).
const EXPLORE_EPISODE_CAP: u64 = 60;

/// Errors of the GOG-203 Grid arms (fail-closed, typed).
#[derive(Debug, Clone, PartialEq)]
pub enum GridClosedLoopError {
    /// Domain validation failure.
    Domain(DomainError),
    /// Generative-model failure during a rollout.
    Model(ModelError),
    /// Learned-model failure (bucketing, prediction).
    Learned(LearnedModelError),
    /// Causal-rule validation failure.
    Rule(CausalRuleError),
    /// Invalid configuration (`reason` names the field).
    InvalidConfig(&'static str),
}

impl std::fmt::Display for GridClosedLoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(e) => write!(f, "domain error: {e}"),
            Self::Model(e) => write!(f, "model error: {e}"),
            Self::Learned(e) => write!(f, "learned model error: {e}"),
            Self::Rule(e) => write!(f, "causal rule error: {e}"),
            Self::InvalidConfig(reason) => write!(f, "invalid configuration: {reason}"),
        }
    }
}

impl std::error::Error for GridClosedLoopError {}

impl From<DomainError> for GridClosedLoopError {
    fn from(e: DomainError) -> Self {
        Self::Domain(e)
    }
}
impl From<ModelError> for GridClosedLoopError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}
impl From<LearnedModelError> for GridClosedLoopError {
    fn from(e: LearnedModelError) -> Self {
        Self::Learned(e)
    }
}
impl From<CausalRuleError> for GridClosedLoopError {
    fn from(e: CausalRuleError) -> Self {
        Self::Rule(e)
    }
}

/// Bounded rollout budget of one decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RolloutConfig {
    /// Rollouts per candidate action (`1..=64`).
    pub population: usize,
    /// Steps per rollout (`1..=32`).
    pub horizon: usize,
}

impl RolloutConfig {
    /// Validates the bounds (fail-closed).
    pub fn new(population: usize, horizon: usize) -> Result<Self, GridClosedLoopError> {
        if population == 0 || population > 64 {
            return Err(GridClosedLoopError::InvalidConfig(
                "rollout population must be in 1..=64",
            ));
        }
        if horizon == 0 || horizon > 32 {
            return Err(GridClosedLoopError::InvalidConfig(
                "rollout horizon must be in 1..=32",
            ));
        }
        Ok(Self {
            population,
            horizon,
        })
    }

    /// Simulated transitions consumed by one decision over all candidate actions.
    pub const fn transitions_per_decision(&self) -> usize {
        GRID_ACTION_CARDINALITY * self.population * self.horizon
    }
}

/// Rollout statistics of one candidate action.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRollout {
    /// Discrete action index.
    pub action: usize,
    /// Rollouts that ended without a terminal failure.
    pub viable: usize,
    /// Rollouts that ended in a terminal failure (hazard / learned failure bucket).
    pub failures: usize,
    /// Shannon entropy (bits) of the final cells of the viable rollouts.
    pub entropy_bits: f64,
    /// Final-cell counts of the viable rollouts (`(x, y) -> count`).
    pub final_cells: BTreeMap<(usize, usize), usize>,
}

impl CandidateRollout {
    fn new(action: usize) -> Self {
        Self {
            action,
            viable: 0,
            failures: 0,
            entropy_bits: 0.0,
            final_cells: BTreeMap::new(),
        }
    }

    fn finish(mut self) -> Self {
        self.entropy_bits = entropy_of(&self.final_cells);
        self
    }

    /// Number of distinct final cells reached by viable rollouts.
    pub fn distinct_cells(&self) -> usize {
        self.final_cells.len()
    }
}

/// One auditable closed-loop decision of the entropic or the empowerment arm.
#[derive(Debug, Clone, PartialEq)]
pub struct ClosedLoopDecision {
    /// Chosen action index.
    pub action: usize,
    /// Selection score (bits): entropic score or per-action `KL(p(S_T|a) || p(S_T))`.
    pub score_bits: f64,
    /// Simulated transitions consumed by this decision.
    pub transitions: usize,
    /// Rollout statistics of every candidate action (fixed order `0..5`).
    pub candidates: Vec<CandidateRollout>,
    /// Control information of the decision (`I(A; S_T)`, empowerment arm only; `0` otherwise).
    pub entropy_gain_bits: f64,
    /// `true` when no candidate had a viable rollout and `Stay` was applied.
    pub fail_closed: bool,
}

fn entropy_of(counts: &BTreeMap<(usize, usize), usize>) -> f64 {
    let total: usize = counts.values().sum();
    if total == 0 {
        return 0.0;
    }
    let probs: Vec<f64> = counts.values().map(|c| *c as f64 / total as f64).collect();
    shannon_entropy_bits(&probs).unwrap_or(0.0)
}

/// `KL(p || q)` in bits, with `p` given as counts (`p_i = count_i / total`).
fn kl_bits(
    counts: &BTreeMap<(usize, usize), usize>,
    total: usize,
    q: &BTreeMap<(usize, usize), f64>,
) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let mut kl = 0.0;
    for (cell, count) in counts {
        let p = *count as f64 / total as f64;
        let qc = q.get(cell).copied().unwrap_or(0.0);
        if p > 0.0 && qc > 0.0 {
            kl += p * (p / qc).log2();
        }
    }
    kl.max(0.0)
}

fn grid_action(index: usize) -> Result<Action, GridClosedLoopError> {
    let space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
    Ok(grid_action_in(&space, index)?)
}

fn grid_action_in(space: &ActionSpace, index: usize) -> Result<Action, DomainError> {
    Action::discrete_in_space(index, space)
}

fn state_at(pos: GridPos) -> Result<State, GridClosedLoopError> {
    Ok(State::new(
        Tick::ZERO,
        SimDuration::ZERO,
        vec![pos.x as f64, pos.y as f64, 0.0, 0.0, 0.0],
    )?)
}

/// One candidate rollout over the true Grid dynamics (no reward is read).
fn rollout_known<M: GenerativeModel>(
    model: &M,
    space: &ActionSpace,
    dt: gogamen_core::StepDt,
    from: GridPos,
    action: usize,
    cfg: RolloutConfig,
    rng: &mut DeterministicRng,
) -> Result<CandidateRollout, GridClosedLoopError> {
    let init = state_at(from)?;
    let mut out = CandidateRollout::new(action);
    for _ in 0..cfg.population {
        let mut state = init.clone();
        let mut failed = false;
        for step in 0..cfg.horizon {
            let idx = if step == 0 {
                action
            } else {
                (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize
            };
            let act = grid_action_in(space, idx)?;
            match model.step(&state, &act, dt, rng) {
                Err(ModelError::TerminalState) => break,
                Err(e) => return Err(e.into()),
                Ok(tr) => {
                    if matches!(tr.terminal(), Terminal::Terminated(TerminalReason::Failure)) {
                        failed = true;
                        break;
                    }
                    state = tr.next_state().clone();
                    if tr.terminal().is_terminal() {
                        break;
                    }
                }
            }
        }
        if failed {
            out.failures += 1;
        } else {
            out.viable += 1;
            let vals = state.values();
            *out.final_cells
                .entry((vals[0] as usize, vals[1] as usize))
                .or_insert(0) += 1;
        }
    }
    Ok(out.finish())
}

fn sample_categorical(dist: &[f64], rng: &mut DeterministicRng) -> usize {
    let u = rng.next_f64();
    let mut acc = 0.0;
    for (i, p) in dist.iter().enumerate() {
        acc += *p;
        if u <= acc {
            return i;
        }
    }
    dist.len().saturating_sub(1)
}

/// One candidate rollout over a frozen tabular learned model (no simulator access).
fn rollout_learned(
    model: &TabularLearnedModel,
    from: GridPos,
    action: usize,
    cfg: RolloutConfig,
    rng: &mut DeterministicRng,
) -> Result<CandidateRollout, GridClosedLoopError> {
    let version = model.version();
    let scale = version.scale();
    let failure = scale.failure_bucket();
    let mut out = CandidateRollout::new(action);
    for _ in 0..cfg.population {
        let mut bucket = scale.bucket_of(&[from.x as f64 + 0.5, from.y as f64 + 0.5])?;
        let mut pos = from;
        let mut failed = false;
        for step in 0..cfg.horizon {
            let idx = if step == 0 {
                action
            } else {
                (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize
            };
            let pred = model.predict(bucket, idx)?;
            let next = sample_categorical(&pred.distribution, rng);
            if next == failure {
                failed = true;
                break;
            }
            let center = scale
                .center_of(next)
                .ok_or(GridClosedLoopError::InvalidConfig("learned bucket"))?;
            pos = GridPos::new(center[0].floor() as usize, center[1].floor() as usize);
            bucket = next;
        }
        if failed {
            out.failures += 1;
        } else {
            out.viable += 1;
            *out.final_cells.entry((pos.x, pos.y)).or_insert(0) += 1;
        }
    }
    Ok(out.finish())
}

/// Reward-free entropic decision over the true Grid dynamics.
///
/// Score: `H(final cells of viable futures) + log2(1 + viable futures)`, the Grid adaptation of
/// the Kart `EntropicPureController` rule. Ties keep the lowest action index; no viable candidate
/// means `Stay` (fail-closed).
pub fn entropic_pure_decision(
    model: &GridWorldModel,
    from: GridPos,
    cfg: RolloutConfig,
    rng: &mut DeterministicRng,
) -> Result<ClosedLoopDecision, GridClosedLoopError> {
    let space = model.action_space().clone();
    let dt = model.config().dt;
    let mut candidates = Vec::with_capacity(GRID_ACTION_CARDINALITY);
    for action in 0..GRID_ACTION_CARDINALITY {
        candidates.push(rollout_known(model, &space, dt, from, action, cfg, rng)?);
    }
    let mut best: Option<(usize, f64)> = None;
    for c in &candidates {
        if c.viable == 0 {
            continue;
        }
        let score = c.entropy_bits + (1.0 + c.viable as f64).log2();
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((c.action, score));
        }
    }
    let (action, score_bits, fail_closed) = match best {
        Some((a, s)) => (a, s, false),
        None => (0, 0.0, true),
    };
    Ok(ClosedLoopDecision {
        action,
        score_bits,
        transitions: cfg.transitions_per_decision(),
        candidates,
        entropy_gain_bits: 0.0,
        fail_closed,
    })
}

/// Closed-loop empowerment decision over a learned model.
///
/// Each candidate action is scored by its control information contribution
/// `KL(p(S_T | a) || p(S_T))` in bits; the reported `entropy_gain_bits` is
/// `I(A; S_T) = sum_a p(a) * KL_a` for the decision (a plug-in estimate over the rollouts).
pub fn empowerment_decision(
    model: &TabularLearnedModel,
    from: GridPos,
    cfg: RolloutConfig,
    rng: &mut DeterministicRng,
) -> Result<ClosedLoopDecision, GridClosedLoopError> {
    let mut candidates = Vec::with_capacity(GRID_ACTION_CARDINALITY);
    for action in 0..GRID_ACTION_CARDINALITY {
        candidates.push(rollout_learned(model, from, action, cfg, rng)?);
    }
    let total_viable: usize = candidates.iter().map(|c| c.viable).sum();
    if total_viable == 0 {
        return Ok(ClosedLoopDecision {
            action: 0,
            score_bits: 0.0,
            transitions: cfg.transitions_per_decision(),
            candidates,
            entropy_gain_bits: 0.0,
            fail_closed: true,
        });
    }
    // Marginal distribution of final cells over all rollouts (uniform action prior).
    let mut mixture: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    for c in &candidates {
        if c.viable == 0 {
            continue;
        }
        let w = c.viable as f64 / total_viable as f64;
        for (cell, count) in &c.final_cells {
            *mixture.entry(*cell).or_insert(0.0) += w * *count as f64 / c.viable as f64;
        }
    }
    let mut best: Option<(usize, f64)> = None;
    let mut control_bits = 0.0;
    for c in &candidates {
        if c.viable == 0 {
            continue;
        }
        let kl = kl_bits(&c.final_cells, c.viable, &mixture);
        control_bits += (c.viable as f64 / total_viable as f64) * kl;
        if best.is_none_or(|(_, b)| kl > b) {
            best = Some((c.action, kl));
        }
    }
    let (action, score_bits) = best.unwrap_or((0, 0.0));
    Ok(ClosedLoopDecision {
        action,
        score_bits,
        transitions: cfg.transitions_per_decision(),
        candidates,
        entropy_gain_bits: control_bits,
        fail_closed: false,
    })
}

/// Runs one reward-free entropic Grid episode against the real world.
pub fn run_entropic_pure_grid_episode(
    world: &GridWorldConfig,
    seed: MasterSeed,
    cfg: RolloutConfig,
    max_ticks: u64,
) -> Result<HiddenEpisode, GridClosedLoopError> {
    let model = GridWorldModel::new(world.clone(), seed)?;
    let mut rng = seed.derive(StreamKey::Episode(0x203));
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut ticks = 0u64;
    let mut planning = 0u64;
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let d = entropic_pure_decision(&model, env.position(), cfg, &mut rng)?;
        planning += d.transitions as u64;
        let act = grid_action(d.action)?;
        env.apply(&act).map_err(|_| {
            GridClosedLoopError::Domain(DomainError::InvalidTransition {
                reason: "entropic grid arm could not apply its action",
            })
        })?;
        ticks += 1;
    }
    Ok(finish(env, ticks, planning))
}

/// Runs one empowerment episode against the real world using only the learned `model`.
pub fn run_empowerment_episode(
    world: &GridWorldConfig,
    model: &TabularLearnedModel,
    seed: MasterSeed,
    cfg: RolloutConfig,
    max_ticks: u64,
) -> Result<HiddenEpisode, GridClosedLoopError> {
    let mut rng = seed.derive(StreamKey::Episode(0xE203));
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut ticks = 0u64;
    let mut planning = 0u64;
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let d = empowerment_decision(model, env.position(), cfg, &mut rng)?;
        planning += d.transitions as u64;
        let act = grid_action(d.action)?;
        env.apply(&act).map_err(|_| {
            GridClosedLoopError::Domain(DomainError::InvalidTransition {
                reason: "empowerment grid arm could not apply its action",
            })
        })?;
        ticks += 1;
    }
    Ok(finish(env, ticks, planning))
}

fn finish(env: GridWorldEnv, ticks: u64, planning: u64) -> HiddenEpisode {
    HiddenEpisode {
        reached_goal: env.terminal() == Terminal::Terminated(TerminalReason::GoalReached),
        failed: env.terminal() == Terminal::Terminated(TerminalReason::Failure),
        ticks,
        total_reward: env.total_reward(),
        planning_transitions: planning,
        fallbacks: BTreeMap::new(),
        traces: Vec::new(),
    }
}

/// Executed evidence for one `(cell, action)` pair.
#[derive(Debug, Clone, Default, PartialEq)]
struct PairEvidence {
    moved: u64,
    blocked: u64,
    successors: BTreeMap<(usize, usize), u64>,
    provenance: Vec<u64>,
}

impl PairEvidence {
    fn total(&self) -> u64 {
        self.moved + self.blocked
    }

    fn modal_successor(&self) -> Option<(usize, usize)> {
        self.successors
            .iter()
            .max_by_key(|(cell, count)| (**count, std::cmp::Reverse(**cell)))
            .map(|(cell, _)| *cell)
    }
}

/// Outcome of one causal-rule decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalDecision {
    /// Action index chosen.
    pub action: usize,
    /// `true` when no evidence-backed, permitted rule applied and a seeded random move was used.
    pub fallback: bool,
    /// Induced rules evaluated for this decision (one per action with evidence).
    pub rules_considered: usize,
}

/// Causal-rule learning agent: induces typed rules from executed transitions and acts on them.
#[derive(Debug, Clone)]
pub struct GridCausalAgent {
    evidence: BTreeMap<(usize, usize, usize), PairEvidence>,
    visits: BTreeMap<(usize, usize), u64>,
    rng: DeterministicRng,
    next_evidence_id: u64,
    decisions: u64,
    fallback_decisions: u64,
}

impl GridCausalAgent {
    /// Empty agent with a deterministic decision RNG.
    pub fn new(seed: MasterSeed) -> Self {
        Self {
            evidence: BTreeMap::new(),
            visits: BTreeMap::new(),
            rng: seed.derive(StreamKey::Episode(0xCA05A)),
            next_evidence_id: 0,
            decisions: 0,
            fallback_decisions: 0,
        }
    }

    /// Collects `transitions` random-policy transitions from `world` (resetting on terminal).
    pub fn explore(
        world: &GridWorldConfig,
        transitions: usize,
        seed: MasterSeed,
    ) -> Result<Self, GridClosedLoopError> {
        if transitions == 0 {
            return Err(GridClosedLoopError::InvalidConfig(
                "causal exploration needs at least one transition",
            ));
        }
        let mut agent = Self::new(seed);
        let mut env = GridWorldEnv::new(world.clone())?;
        let space = Actuator::action_space(&env).clone();
        let mut rng = seed.derive(StreamKey::Episode(0x203A));
        let mut ticks_in_episode = 0u64;
        for _ in 0..transitions {
            if env.terminal().is_terminal() || ticks_in_episode >= EXPLORE_EPISODE_CAP {
                LiveEnvironment::reset(&mut env).map_err(|_| {
                    GridClosedLoopError::Domain(DomainError::InvalidTransition {
                        reason: "causal exploration could not reset the grid environment",
                    })
                })?;
                ticks_in_episode = 0;
            }
            let from = env.position();
            let idx = (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
            let act = grid_action_in(&space, idx)?;
            env.apply(&act).map_err(|_| {
                GridClosedLoopError::Domain(DomainError::InvalidTransition {
                    reason: "causal exploration could not apply an action",
                })
            })?;
            agent.observe_transition(from, idx, env.position());
            ticks_in_episode += 1;
        }
        Ok(agent)
    }

    /// Records one executed transition as evidence.
    pub fn observe_transition(&mut self, from: GridPos, action: usize, to: GridPos) {
        let entry = self.evidence.entry((from.x, from.y, action)).or_default();
        if to == from {
            entry.blocked += 1;
        } else {
            entry.moved += 1;
            *entry.successors.entry((to.x, to.y)).or_insert(0) += 1;
        }
        if entry.provenance.len() < GRID_CAUSAL_MAX_PROVENANCE {
            self.next_evidence_id += 1;
            entry.provenance.push(self.next_evidence_id);
        }
    }

    /// Number of `(cell, action)` pairs with executed evidence.
    pub fn evidence_pairs(&self) -> usize {
        self.evidence.len()
    }

    /// Decisions taken so far.
    pub const fn decisions(&self) -> u64 {
        self.decisions
    }

    /// Decisions taken without an applicable, permitted rule (seeded random fallback).
    pub const fn fallback_decisions(&self) -> u64 {
        self.fallback_decisions
    }

    /// Induces the typed causal rule of `action` at `cell` from the recorded evidence.
    ///
    /// Returns `Ok(None)` when the pair has no evidence. The rule carries its provenance
    /// ([`EvidenceRef`]) and an epistemic status (`Hypothesis` with one observation,
    /// `CorroboratedObservation` with two or more). Outcomes are the empirical `moved`/`blocked`
    /// distribution; the effect records the modal successor cell.
    pub fn rule_for(
        &self,
        cell: GridPos,
        action: usize,
    ) -> Result<Option<CausalRule>, GridClosedLoopError> {
        let Some(entry) = self.evidence.get(&(cell.x, cell.y, action)) else {
            return Ok(None);
        };
        let total = entry.total();
        if total == 0 {
            return Ok(None);
        }
        let here = VariableName::new("here")?;
        let next = VariableName::new("next")?;
        let condition = Condition::Equals(here, here_value(cell)?);
        let mut outcomes = Vec::new();
        if entry.moved > 0 {
            let succ = entry
                .modal_successor()
                .ok_or(GridClosedLoopError::InvalidConfig("causal successor"))?;
            outcomes.push(Outcome::new(
                OutcomeLabel::new("moved")?,
                entry.moved as f64 / total as f64,
                vec![Effect {
                    variable: next.clone(),
                    value: cell_value(succ)?,
                }],
            )?);
        }
        if entry.blocked > 0 {
            outcomes.push(Outcome::new(
                OutcomeLabel::new("blocked")?,
                entry.blocked as f64 / total as f64,
                vec![Effect {
                    variable: next,
                    value: cell_value((cell.x, cell.y))?,
                }],
            )?);
        }
        let status = if total >= 2 {
            EpistemicStatus::CorroboratedObservation
        } else {
            EpistemicStatus::Hypothesis
        };
        let rule = CausalRule::new(
            RuleId::new(format!("grid.causal.a{action}.x{}.y{}", cell.x, cell.y))?,
            1,
            Some(action),
            vec![condition],
            TemporalScope::ALWAYS,
            outcomes,
            entry
                .provenance
                .iter()
                .map(|id| EvidenceRef::new(*id))
                .collect(),
            status,
        )?;
        Ok(Some(rule))
    }

    /// Decides one action from the observed cell.
    ///
    /// A rule is used only when it applies (all conditions observed and true) **and** the
    /// harness grants [`ActionPermission`] for exactly that discrete action. Among usable rules
    /// the agent prefers a move to an unvisited cell, then the least-visited successor, then the
    /// highest empirical `moved` probability, then the lowest action index. Without a usable rule
    /// it falls back to a seeded random move.
    pub fn decide(
        &mut self,
        cell: GridPos,
        tick: u64,
    ) -> Result<CausalDecision, GridClosedLoopError> {
        let space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
        let ctx = RuleContext::new(tick).with(
            VariableName::new("here")?,
            Observed::Known(here_value(cell)?),
        );
        let mut best: Option<((bool, u32, u32), usize)> = None;
        let mut considered = 0usize;
        for action in 0..GRID_ACTION_CARDINALITY {
            let Some(rule) = self.rule_for(cell, action)? else {
                continue;
            };
            considered += 1;
            let permission = permission_for(&space, action)?;
            let RuleEvaluation::Applies { outcomes } = rule.evaluate(&ctx, permission) else {
                continue;
            };
            let p_moved = outcomes
                .iter()
                .find(|(label, _)| label.as_str() == "moved")
                .map_or(0.0, |(_, p)| *p);
            if p_moved <= 0.0 {
                continue;
            }
            let entry = self
                .evidence
                .get(&(cell.x, cell.y, action))
                .ok_or(GridClosedLoopError::InvalidConfig("causal evidence"))?;
            let succ = entry
                .modal_successor()
                .ok_or(GridClosedLoopError::InvalidConfig("causal successor"))?;
            let visits = self.visits.get(&succ).copied().unwrap_or(0);
            let key = (
                succ != (cell.x, cell.y) && visits == 0,
                u32::MAX - visits.min(u64::from(u32::MAX)) as u32,
                (p_moved * 1000.0) as u32,
            );
            if best.is_none_or(|(bk, _)| key > bk) {
                best = Some((key, action));
            }
        }
        self.decisions += 1;
        let (action, fallback) = match best {
            Some((_, a)) => (a, false),
            None => {
                self.fallback_decisions += 1;
                (
                    (self.rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize,
                    true,
                )
            }
        };
        self.visits
            .entry((cell.x, cell.y))
            .and_modify(|v| *v += 1)
            .or_insert(1);
        Ok(CausalDecision {
            action,
            fallback,
            rules_considered: considered,
        })
    }
}

/// Runs one causal-rule episode: the agent only sees its cell and its own induced rules.
pub fn run_causal_grid_episode(
    world: &GridWorldConfig,
    agent: &mut GridCausalAgent,
    max_ticks: u64,
) -> Result<HiddenEpisode, GridClosedLoopError> {
    let mut env = GridWorldEnv::new(world.clone())?;
    let mut ticks = 0u64;
    while ticks < max_ticks && !env.terminal().is_terminal() {
        let d = agent.decide(env.position(), ticks)?;
        let act = grid_action(d.action)?;
        env.apply(&act).map_err(|_| {
            GridClosedLoopError::Domain(DomainError::InvalidTransition {
                reason: "causal grid arm could not apply its action",
            })
        })?;
        ticks += 1;
    }
    Ok(finish(env, ticks, 0))
}

fn cell_value(cell: (usize, usize)) -> Result<RuleValue, CausalRuleError> {
    RuleValue::symbol(format!("x{}.y{}", cell.0, cell.1))
}

fn here_value(cell: GridPos) -> Result<RuleValue, CausalRuleError> {
    cell_value((cell.x, cell.y))
}

/// External authorization: the validated discrete action space of the environment grants the
/// permission. The rule itself never grants it.
fn permission_for(
    space: &ActionSpace,
    action: usize,
) -> Result<ActionPermission, GridClosedLoopError> {
    let act = grid_action_in(space, action)?;
    Ok(ActionPermission::from_decision(
        &GuardDecision::Allow(act),
        action,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_a() -> GridWorldConfig {
        GridWorldConfig {
            map: gogamen_lab_grid::GridMap::from_ascii(&[
                "#######", "#S.#.G#", "#..#H.#", "#.....#", "#######",
            ])
            .unwrap(),
            hazard_is_terminal: true,
            ..GridWorldConfig::default()
        }
    }

    #[test]
    fn gog_203_entropic_pure_grid_scores_only_entropy_and_viability() {
        let world = map_a();
        let model = GridWorldModel::new(world.clone(), MasterSeed::new(7)).unwrap();
        let cfg = RolloutConfig::new(GRID_ENTROPIC_POPULATION, GRID_ENTROPIC_HORIZON).unwrap();
        let mut rng = MasterSeed::new(7).derive(StreamKey::Episode(0x203));
        let from = world.map.start();
        let a = entropic_pure_decision(&model, from, cfg, &mut rng).unwrap();
        let mut rng = MasterSeed::new(7).derive(StreamKey::Episode(0x203));
        let b = entropic_pure_decision(&model, from, cfg, &mut rng).unwrap();
        assert_eq!(a, b, "same seed same decision");
        assert_eq!(a.transitions, 5 * 8 * 4);
        assert_eq!(a.candidates.len(), 5);
        assert_eq!(a.entropy_gain_bits, 0.0);
        assert!(!a.fail_closed);
        // The reward of the world is never read: changing it must not change the decision.
        let mut other = world.clone();
        other.step_reward = 1_000.0;
        other.goal_reward = -1_000.0;
        let model_other = GridWorldModel::new(other, MasterSeed::new(7)).unwrap();
        let mut rng = MasterSeed::new(7).derive(StreamKey::Episode(0x203));
        let c = entropic_pure_decision(&model_other, from, cfg, &mut rng).unwrap();
        assert_eq!(a.action, c.action);
        assert_eq!(a.score_bits.to_bits(), c.score_bits.to_bits());
    }

    #[test]
    fn gog_203_empowerment_kl_is_control_information_not_diversity() {
        // Two actions, one deterministic cell each: KL_a = 1 bit and I(A;S_T) = 1 bit.
        let mut left = CandidateRollout::new(0);
        left.viable = 4;
        left.final_cells = BTreeMap::from([((1, 1), 4)]);
        let mut right = CandidateRollout::new(1);
        right.viable = 4;
        right.final_cells = BTreeMap::from([((2, 1), 4)]);
        let total = left.viable + right.viable;
        let mut mixture: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for c in [&left, &right] {
            let w = c.viable as f64 / total as f64;
            for (cell, n) in &c.final_cells {
                *mixture.entry(*cell).or_insert(0.0) += w * *n as f64 / c.viable as f64;
            }
        }
        let kl_left = kl_bits(&left.final_cells, left.viable, &mixture);
        let kl_right = kl_bits(&right.final_cells, right.viable, &mixture);
        assert!((kl_left - 1.0).abs() < 1e-12, "{kl_left}");
        assert!((kl_right - 1.0).abs() < 1e-12, "{kl_right}");
        // Pure noise (the same spread final-cell distribution for every action) is
        // diverse but carries no control information: KL(p || p) = 0.
        let spread = BTreeMap::from([((1, 1), 2), ((2, 1), 2)]);
        assert!((entropy_of(&spread) - 1.0).abs() < 1e-12);
        let mut noise_mixture: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        for _ in 0..2 {
            for (cell, n) in &spread {
                *noise_mixture.entry(*cell).or_insert(0.0) += 0.5 * *n as f64 / 4.0;
            }
        }
        assert_eq!(kl_bits(&spread, 4, &noise_mixture), 0.0);
    }

    #[test]
    fn gog_203_causal_agent_induces_rules_and_requires_permission() {
        let world = map_a();
        let agent = GridCausalAgent::explore(&world, 200, MasterSeed::new(7)).unwrap();
        assert!(agent.evidence_pairs() > 0);
        let start = world.map.start();
        // Evaluate the induced rule of a known pair with and without explicit permission.
        let (cell, action) = agent
            .evidence
            .keys()
            .map(|(x, y, a)| (GridPos::new(*x, *y), *a))
            .next()
            .unwrap();
        let rule = agent.rule_for(cell, action).unwrap().unwrap();
        assert_eq!(rule.action(), Some(action));
        assert!(!rule.provenance().is_empty());
        let ctx = RuleContext::new(0).with(
            VariableName::new("here").unwrap(),
            Observed::Known(here_value(cell).unwrap()),
        );
        assert_eq!(
            rule.evaluate(&ctx, ActionPermission::NotPermitted),
            RuleEvaluation::ActionNotPermitted,
            "a rule never grants its own permission"
        );
        assert!(matches!(
            rule.evaluate(&ctx, ActionPermission::Permitted),
            RuleEvaluation::Applies { .. }
        ));
        // A different cell contradicts the condition (the rule is contextual, not universal).
        let other = RuleContext::new(0).with(
            VariableName::new("here").unwrap(),
            Observed::Known(here_value(GridPos::new(start.x + 1, start.y)).unwrap()),
        );
        assert!(matches!(
            rule.evaluate(&other, ActionPermission::Permitted),
            RuleEvaluation::Contradicted { .. }
        ));
        // Decisions never exceed the harness actions and are deterministic.
        let mut a = GridCausalAgent::explore(&world, 100, MasterSeed::new(11)).unwrap();
        let mut b = GridCausalAgent::explore(&world, 100, MasterSeed::new(11)).unwrap();
        for tick in 0..20 {
            let da = a.decide(start, tick).unwrap();
            let db = b.decide(start, tick).unwrap();
            assert_eq!(da, db);
            assert!(da.action < GRID_ACTION_CARDINALITY);
            assert!(da.rules_considered <= GRID_ACTION_CARDINALITY);
        }
    }

    #[test]
    fn gog_203_grid_arm_episodes_report_denominators_and_planning_cost() {
        let world = map_a();
        let cfg = RolloutConfig::new(GRID_ENTROPIC_POPULATION, GRID_ENTROPIC_HORIZON).unwrap();
        let entropic = run_entropic_pure_grid_episode(&world, MasterSeed::new(7), cfg, 12).unwrap();
        assert_eq!(entropic.ticks, 12);
        assert_eq!(entropic.planning_transitions, 12 * 5 * 8 * 4);
        assert!(entropic.traces.is_empty());

        let mut agent = GridCausalAgent::explore(&world, 120, MasterSeed::new(7)).unwrap();
        let causal = run_causal_grid_episode(&world, &mut agent, 12).unwrap();
        assert_eq!(causal.ticks, 12);
        assert_eq!(
            causal.planning_transitions, 0,
            "rules plan without simulation"
        );
        assert_eq!(agent.decisions(), 12);

        assert!(matches!(
            RolloutConfig::new(0, 4),
            Err(GridClosedLoopError::InvalidConfig(_))
        ));
        assert!(RolloutConfig::new(4, 0).is_err());
    }
}
