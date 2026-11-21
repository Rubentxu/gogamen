//! GOG-188: "shrinking budget" on the Kart with its privileged model (EXPERIMENTAL, `std`-only).
//!
//! Question: if the planning budget (simulated transitions per decision) is halved every `N`
//! episodes, can an agent that **distils its own planner into a reflex** (a [`TabularPolicy`]
//! learned from the planner's decisions) keep the same viability with fewer transitions than an
//! agent that only plans?
//!
//! - "Privileged model" = the Kart's true physics ([`KartModel`]), used by the planner. The
//!   learned model plays no role here.
//! - Distillation reuses the pattern of `labs/life/src/strategy_learning.rs`: decisions of the
//!   planner are tallied into a [`TabularPolicy`] and promoted through [`ProceduralMemory`] /
//!   [`PromotionGate`]. The reflex only replaces the planner **after** the gate approves it, and
//!   it abstains (the planner decides) on state buckets with too little support.
//! - EVERY action (planner, reflex, warm-up) goes through [`ActionGuard`] before the environment.
//!
//! # Preregistration (written before any result was seen)
//! See [`PREREG`]. Equal viability = paired mean difference of per-episode viability
//! (`distilled - plan_only`, pooled over all stages and seeds) `>= -VIABILITY_TOLERANCE`, with
//! the distilled agent also visiting at least `MIN_CELLS_RATIO` of the cells the planner visits
//! (a stopped kart is trivially "viable", so behaviour must not be degenerate). Saving criterion:
//! fraction of decision transitions saved (`1 - distilled / plan_only`, decision cost only) is
//! `>= SAVING_MIN` and the reflex was promoted in at least `MIN_PROMOTED_SEED_FRACTION` of the
//! seeds. The cost of the promotion fixtures is reported separately as `gate_transitions` and a
//! net saving is also reported. Anything else is reported as a null / negative result.
//!
//! No agency or emergence claim: a tabular behaviour-cloning of a sampling planner on two toy
//! circuits, a few seeds, short episodes.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, Actuator, DeterministicRng, DomainError,
    EndowmentProvenance, EnvironmentError, EnvironmentObserver, EvaluationCriteria,
    EvaluationEvidence, EvidenceRef, GenerativeModel, GuardContext, GuardDecision, LiveEnvironment,
    MasterSeed, ModelError, ProceduralError, ProceduralMemory, PromotionDecision, RngSource,
    SafeFallbackPolicy, State, StepDt, StrategyId, StrategyStats, StreamKey, TabularPolicy,
    Terminal, TerminalReason, Tick, shannon_entropy_bits,
};

use crate::physics::{KartState, KinematicBicycleConfig, wrap_angle};
use crate::track::TrackMap;
use crate::world::{KartEnv, KartModel};

/// Preregistered tolerance: distilled viability may be at most this far below plan-only (paired).
pub const VIABILITY_TOLERANCE: f64 = 0.05;
/// Preregistered saving criterion: minimum fraction of decision transitions saved.
pub const SAVING_MIN: f64 = 0.25;
/// Preregistered: fraction of seeds in which the reflex must be promoted at least once.
pub const MIN_PROMOTED_SEED_FRACTION: f64 = 0.5;
/// Preregistered non-degeneracy: distilled agent must visit this fraction of planner's cells.
pub const MIN_CELLS_RATIO: f64 = 0.8;
/// Gate: extra failure rate tolerated over the planner on the fixture.
pub const GATE_FAILURE_TOLERANCE: f64 = 0.0;
/// Gate: minimum fraction of fixture decisions that the reflex must resolve (`holdout_score`).
pub const GATE_MIN_COVERAGE: f64 = 0.3;
/// A state bucket answers only with at least this many tallied planner decisions.
pub const MIN_BUCKET_SUPPORT: f64 = 3.0;

/// Human-readable preregistration (kept next to the constants it describes).
pub const PREREG: &str = "equal viability: paired mean(viab_distilled - viab_plan_only) >= -0.05 \
    pooled over stages and seeds AND cells_distilled/cells_plan >= 0.8; saving: 1 - \
    transitions_distilled/transitions_plan >= 0.25 AND reflex promoted in >= 50% of seeds. \
    Otherwise null/negative. Fixed before running.";

/// Candidate actions: fixed 3x3 grid `{-max,0,+max}` steering x `{min,0,max}` acceleration.
pub const NUM_ACTIONS: usize = 9;
const GRID: usize = 16;
const YAW_BINS: usize = 8;
const SPEED_BINS: usize = 3;
/// Number of state buckets of the reflex table.
pub const NUM_STATE_BUCKETS: usize = GRID * GRID * YAW_BINS * SPEED_BINS;

/// One endowment-ledger entry (minimal duplicate of the `labs/grid` pattern; racing does not
/// depend on grid).
#[derive(Debug, Clone, PartialEq)]
pub struct EndowmentEntry {
    /// Quantity name.
    pub name: &'static str,
    /// Value handed to the agent / experiment.
    pub value: f64,
    /// Origin.
    pub source: EndowmentProvenance,
}

/// Ledger of design numbers.
pub fn endowment_ledger(cfg: &ExperimentConfig) -> Vec<EndowmentEntry> {
    let given = |name, value| EndowmentEntry {
        name,
        value,
        source: EndowmentProvenance::Given,
    };
    vec![
        given("initial_population", cfg.initial_population as f64),
        given("horizon", cfg.horizon as f64),
        given("halvings", cfg.halvings as f64),
        given("episodes_per_stage", cfg.episodes_per_stage as f64),
        given("ticks_per_episode", cfg.ticks as f64),
        given("viability_tolerance", VIABILITY_TOLERANCE),
        given("saving_min", SAVING_MIN),
        given("min_bucket_support", MIN_BUCKET_SUPPORT),
        given("gate_min_coverage", GATE_MIN_COVERAGE),
    ]
}

/// Errors of the experiment.
#[derive(Debug, Clone, PartialEq)]
pub enum ShrinkingError {
    /// Invalid configuration.
    InvalidConfig(&'static str),
    /// Domain error.
    Domain(DomainError),
    /// Environment error.
    Environment(EnvironmentError),
    /// Model error.
    Model(ModelError),
    /// Procedural memory error.
    Procedural(ProceduralError),
}

impl From<DomainError> for ShrinkingError {
    fn from(e: DomainError) -> Self {
        Self::Domain(e)
    }
}
impl From<EnvironmentError> for ShrinkingError {
    fn from(e: EnvironmentError) -> Self {
        Self::Environment(e)
    }
}
impl From<ModelError> for ShrinkingError {
    fn from(e: ModelError) -> Self {
        Self::Model(e)
    }
}
impl From<ProceduralError> for ShrinkingError {
    fn from(e: ProceduralError) -> Self {
        Self::Procedural(e)
    }
}

/// Experiment configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentConfig {
    /// Paired seeds (even index: rectangular circuit, odd: chicane circuit).
    pub seeds: Vec<u64>,
    /// Rollouts per candidate at stage 0; halved at every stage (min 1).
    pub initial_population: usize,
    /// Rollout length in steps (fixed).
    pub horizon: usize,
    /// Number of halvings (stages = halvings + 1).
    pub halvings: usize,
    /// Episodes per stage ("every N episodes").
    pub episodes_per_stage: usize,
    /// Ticks per episode.
    pub ticks: usize,
    /// Promotion-fixture episodes per gate attempt.
    pub fixture_episodes: usize,
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self {
            seeds: vec![11, 23, 37, 41],
            initial_population: 32,
            horizon: 8,
            halvings: 5,
            episodes_per_stage: 3,
            ticks: 100,
            fixture_episodes: 2,
        }
    }
}

impl ExperimentConfig {
    fn validate(&self) -> Result<(), ShrinkingError> {
        if self.seeds.is_empty()
            || self.initial_population == 0
            || self.horizon == 0
            || self.episodes_per_stage == 0
            || self.ticks == 0
            || self.fixture_episodes == 0
        {
            return Err(ShrinkingError::InvalidConfig("zero-sized parameter"));
        }
        Ok(())
    }

    /// Rollouts per candidate at `stage`.
    pub fn population_at(&self, stage: usize) -> usize {
        (self.initial_population >> stage.min(63)).max(1)
    }

    /// Nominal transitions per decision at `stage` (upper bound: `9 * pop * horizon`).
    pub fn nominal_budget(&self, stage: usize) -> usize {
        NUM_ACTIONS * self.population_at(stage) * self.horizon
    }
}

/// Fixed grid of the 9 candidate actions.
pub fn action_grid(
    kin: &KinematicBicycleConfig,
    space: &ActionSpace,
) -> Result<Vec<Action>, DomainError> {
    let mut v = Vec::with_capacity(NUM_ACTIONS);
    for s in [-kin.max_steering_angle, 0.0, kin.max_steering_angle] {
        for a in [kin.min_accel, 0.0, kin.max_accel] {
            v.push(Action::continuous_in_space(vec![s, a], space)?);
        }
    }
    Ok(v)
}

/// State bucket of a Kart pose (position 4 m cells, 8 yaw bins, 3 speed bins).
pub fn state_bucket(k: &KartState) -> usize {
    let cell = |v: f64| ((v / 4.0).floor().max(0.0) as usize).min(GRID - 1);
    let yaw = wrap_angle(k.yaw);
    let yi = (((yaw + std::f64::consts::PI) / std::f64::consts::TAU) * YAW_BINS as f64)
        .floor()
        .clamp(0.0, (YAW_BINS - 1) as f64) as usize;
    let si = if k.speed < 3.0 {
        0
    } else if k.speed < 8.0 {
        1
    } else {
        2
    };
    ((cell(k.x) * GRID + cell(k.y)) * YAW_BINS + yi) * SPEED_BINS + si
}

/// Passes a proposed action through the [`ActionGuard`]; returns the action to execute and
/// whether the guard intervened (fallback / no-op / reject -> brake fail-closed).
pub fn guarded_action(
    guard: &ActionGuard,
    space: &ActionSpace,
    proposed: Action,
    brake: &Action,
    tick: Tick,
    terminal: Terminal,
) -> (Action, bool) {
    let proposal = ActionProposal::authorized(proposed, tick);
    let ctx = GuardContext::now(space, tick, terminal);
    match guard.evaluate(&proposal, &ctx) {
        GuardDecision::Allow(a) => (a, false),
        GuardDecision::Fallback(a) => (a, true),
        GuardDecision::NoOp | GuardDecision::Reject(_) => (brake.clone(), true),
    }
}

/// Sampling planner over the true Kart model (the "privileged model"). Counts transitions.
struct Planner {
    model: KartModel,
    dt: StepDt,
    space: ActionSpace,
    grid: Vec<Action>,
    kin: KinematicBicycleConfig,
    rng: DeterministicRng,
    population: usize,
    horizon: usize,
    transitions: u64,
}

impl Planner {
    fn new(
        kin: &KinematicBicycleConfig,
        track: &TrackMap,
        seed: MasterSeed,
    ) -> Result<Self, ShrinkingError> {
        let model = KartModel::new(kin.clone(), track.clone(), seed)?;
        let space = model.action_space().clone();
        let grid = action_grid(kin, &space)?;
        Ok(Self {
            model,
            dt: kin.dt,
            space,
            grid,
            kin: kin.clone(),
            rng: seed.derive(StreamKey::Agent {
                episode: 0,
                agent: 0,
            }),
            population: 1,
            horizon: 1,
            transitions: 0,
        })
    }

    fn configure(&mut self, seed: MasterSeed, tag: u64, population: usize, horizon: usize) {
        self.rng = seed.derive(StreamKey::Agent {
            episode: tag,
            agent: 1,
        });
        self.population = population;
        self.horizon = horizon;
    }

    /// Best root action index by `H_bits(final cells of viable futures) + log2(1 + viable)`.
    /// Rollouts: root action at step 0, then one random constant action held for the horizon.
    fn decide(&mut self, state: &State) -> Result<usize, ShrinkingError> {
        let mut best: Option<(f64, usize)> = None;
        for idx in 0..NUM_ACTIONS {
            let root = self.grid[idx].clone();
            let mut cells: BTreeMap<(i64, i64), usize> = BTreeMap::new();
            let mut viable = 0usize;
            for _ in 0..self.population {
                let st = (2.0 * self.rng.next_f64() - 1.0) * self.kin.max_steering_angle;
                let ac = self.kin.min_accel
                    + self.rng.next_f64() * (self.kin.max_accel - self.kin.min_accel);
                let held = Action::continuous_in_space(vec![st, ac], &self.space)
                    .map_err(ModelError::InvalidAction)?;
                let mut s = state.clone();
                let mut alive = true;
                for step in 0..self.horizon {
                    let act = if step == 0 { &root } else { &held };
                    self.transitions += 1;
                    let tr =
                        match GenerativeModel::step(&self.model, &s, act, self.dt, &mut self.rng) {
                            Ok(tr) => tr,
                            Err(ModelError::TerminalState) => break,
                            Err(e) => return Err(e.into()),
                        };
                    s = tr.next_state().clone();
                    match tr.terminal() {
                        Terminal::Ongoing | Terminal::Terminated(TerminalReason::GoalReached) => {}
                        Terminal::Terminated(_) => {
                            alive = false;
                            break;
                        }
                    }
                    if tr.terminal().is_terminal() {
                        break;
                    }
                }
                if alive {
                    viable += 1;
                    let k = KartState::from_slice(&s.values()[0..4])
                        .map_err(ModelError::TransitionFailed)?;
                    *cells
                        .entry((k.x.floor() as i64, k.y.floor() as i64))
                        .or_insert(0) += 1;
                }
            }
            if viable == 0 {
                continue;
            }
            let probs: Vec<f64> = cells.values().map(|c| *c as f64 / viable as f64).collect();
            let h = shannon_entropy_bits(&probs).map_err(|_| ModelError::InvalidState {
                reason: "entropy failed",
            })?;
            let score = h + (1.0 + viable as f64).log2();
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, idx));
            }
        }
        // Fail-closed: no viable future anywhere -> full brake, straight (index 1 = steer -max
        // would be wrong; brake straight is grid index 3).
        Ok(best.map_or(3, |b| b.1))
    }
}

/// Statistics of one episode.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EpisodeStats {
    /// Fraction of ticks survived (1.0 if no collision).
    pub viability: f64,
    /// `true` if the Kart collided.
    pub collided: bool,
    /// Decisions taken (excluding warm-up).
    pub decisions: u64,
    /// Decisions resolved by the reflex.
    pub reflex_decisions: u64,
    /// Planner transitions spent.
    pub transitions: u64,
    /// Distinct 1 m cells visited.
    pub cells: usize,
    /// Actions that passed through the guard.
    pub guard_checked: u64,
    /// Guard interventions (fallback / reject).
    pub guard_interventions: u64,
}

/// Reflex = promoted tabular policy + abstention on unsupported buckets.
fn reflex_pick(policy: &TabularPolicy, bucket: usize) -> Option<usize> {
    let row = policy.action_values(bucket).ok()?;
    let support: f64 = row.iter().sum();
    if support < MIN_BUCKET_SUPPORT {
        return None;
    }
    policy.select_action(bucket).ok().map(|(a, _)| a)
}

struct Runner<'a> {
    kin: &'a KinematicBicycleConfig,
    track: &'a TrackMap,
    guard: ActionGuard,
    space: ActionSpace,
    brake: Action,
}

impl<'a> Runner<'a> {
    fn new(kin: &'a KinematicBicycleConfig, track: &'a TrackMap) -> Result<Self, ShrinkingError> {
        let env = KartEnv::new(kin.clone(), track.clone())?;
        let space = Actuator::action_space(&env).clone();
        let brake = Action::continuous_in_space(vec![0.0, kin.min_accel], &space)?;
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(brake.clone()));
        Ok(Self {
            kin,
            track,
            guard,
            space,
            brake,
        })
    }

    fn apply_guarded(
        &self,
        env: &mut KartEnv,
        action: Action,
        tick: usize,
        stats: &mut EpisodeStats,
    ) -> Result<(), ShrinkingError> {
        let (a, intervened) = guarded_action(
            &self.guard,
            &self.space,
            action,
            &self.brake,
            Tick::new(tick as u64),
            env.terminal(),
        );
        stats.guard_checked += 1;
        stats.guard_interventions += u64::from(intervened);
        env.apply(&a)?;
        Ok(())
    }

    /// Runs one episode with a seeded mild random warm-up (start diversity; guarded).
    #[allow(clippy::too_many_arguments)]
    fn episode(
        &self,
        seed: u64,
        tag: u64,
        ticks: usize,
        planner: &mut Planner,
        reflex: Option<&TabularPolicy>,
        mut record: Option<&mut BTreeMap<(usize, usize), f64>>,
    ) -> Result<EpisodeStats, ShrinkingError> {
        let mut env = KartEnv::new(self.kin.clone(), self.track.clone())?;
        let mut stats = EpisodeStats::default();
        let mut wrng = MasterSeed::new(seed).derive(StreamKey::Agent {
            episode: tag,
            agent: 7,
        });
        let warm = 5 + (wrng.next_f64() * 15.0) as usize;
        for t in 0..warm {
            let st = (2.0 * wrng.next_f64() - 1.0) * 0.3;
            let ac = 1.0 + wrng.next_f64() * (self.kin.max_accel - 1.0);
            let a = Action::continuous_in_space(vec![st, ac], &self.space)?;
            self.apply_guarded(&mut env, a, t, &mut stats)?;
            if env.terminal().is_terminal() {
                break;
            }
        }
        if env.terminal().is_terminal() {
            LiveEnvironment::reset(&mut env)?;
        }
        stats.guard_checked = 0;
        stats.guard_interventions = 0;
        let start_transitions = planner.transitions;
        let mut cells = BTreeSet::new();
        let mut survived = ticks;
        for t in 0..ticks {
            let k = env.kart_state();
            cells.insert((k.x.floor() as i64, k.y.floor() as i64));
            let bucket = state_bucket(&k);
            let idx = if let Some(i) = reflex.and_then(|p| reflex_pick(p, bucket)) {
                stats.reflex_decisions += 1;
                i
            } else {
                let obs = EnvironmentObserver::observe(&env)?;
                let state = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
                    .map_err(ModelError::TransitionFailed)?;
                let i = planner.decide(&state)?;
                if let Some(rec) = record.as_deref_mut() {
                    *rec.entry((bucket, i)).or_insert(0.0) += 1.0;
                }
                i
            };
            stats.decisions += 1;
            let proposed = planner.grid[idx].clone();
            self.apply_guarded(&mut env, proposed, t, &mut stats)?;
            match env.terminal() {
                Terminal::Ongoing => {}
                Terminal::Terminated(TerminalReason::GoalReached) => break,
                Terminal::Terminated(_) => {
                    stats.collided = true;
                    survived = t;
                    break;
                }
            }
        }
        stats.viability = survived as f64 / ticks as f64;
        stats.transitions = planner.transitions - start_transitions;
        stats.cells = cells.len();
        Ok(stats)
    }
}

/// Outcome of one promotion-gate attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct GateRecord {
    /// Stage after which the attempt was made.
    pub stage: usize,
    /// Planner viability on the fixture.
    pub plan_viability: f64,
    /// Hybrid (reflex + planner on abstention) viability on the fixture.
    pub reflex_viability: f64,
    /// Fraction of fixture decisions resolved by the reflex.
    pub coverage: f64,
    /// Cells ratio hybrid/planner.
    pub cells_ratio: f64,
    /// `true` if promoted.
    pub promoted: bool,
    /// Rejection reasons (empty if promoted).
    pub reasons: Vec<String>,
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn sd(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

/// Evaluates a candidate policy on fixture episodes and runs it through [`ProceduralMemory`]'s
/// promotion gate. Baseline = the planner measured on the same fixture, with the preregistered
/// viability tolerance folded in (`baseline.mean_return = plan_viability - tolerance`,
/// `min_return_improvement = 0`). Returns the record and the transitions spent on the fixture.
fn gate_attempt(
    runner: &Runner<'_>,
    planner: &mut Planner,
    cfg: &ExperimentConfig,
    seed: u64,
    stage: usize,
    candidate: TabularPolicy,
) -> Result<(GateRecord, u64), ShrinkingError> {
    let before = planner.transitions;
    let n = cfg.fixture_episodes;
    let mut plan = Vec::new();
    let mut hyb = Vec::new();
    let master = MasterSeed::new(seed);
    for j in 0..n {
        let tag = 10_000 + (stage * 100 + j) as u64;
        let fseed = seed ^ 0xF1F1_0000 ^ j as u64;
        planner.configure(master, tag, cfg.population_at(stage), cfg.horizon);
        plan.push(runner.episode(fseed, tag, cfg.ticks, planner, None, None)?);
        planner.configure(master, tag, cfg.population_at(stage), cfg.horizon);
        hyb.push(runner.episode(fseed, tag, cfg.ticks, planner, Some(&candidate), None)?);
    }
    let spent = planner.transitions - before;
    let pv = mean(&plan.iter().map(|e| e.viability).collect::<Vec<_>>());
    let hv = mean(&hyb.iter().map(|e| e.viability).collect::<Vec<_>>());
    let pf = plan.iter().filter(|e| e.collided).count() as u64;
    let hf = hyb.iter().filter(|e| e.collided).count() as u64;
    let dec: u64 = hyb.iter().map(|e| e.decisions).sum();
    let refl: u64 = hyb.iter().map(|e| e.reflex_decisions).sum();
    let coverage = refl as f64 / dec.max(1) as f64;
    let pc = mean(&plan.iter().map(|e| e.cells as f64).collect::<Vec<_>>());
    let hc = mean(&hyb.iter().map(|e| e.cells as f64).collect::<Vec<_>>());
    let cells_ratio = if pc > 0.0 { hc / pc } else { 1.0 };

    let criteria = EvaluationCriteria::new(
        n as u64,
        0.0,
        (pf as f64 / n as f64 + GATE_FAILURE_TOLERANCE).min(1.0),
    )?
    .with_min_holdout_score(GATE_MIN_COVERAGE)?;
    let baseline = StrategyStats::new(
        n as u64,
        pv - VIABILITY_TOLERANCE,
        0.0,
        pf,
        vec![EvidenceRef::new(1)],
    )?;
    let mut memory = ProceduralMemory::new(
        StrategyId::new(1),
        TabularPolicy::new(NUM_STATE_BUCKETS, NUM_ACTIONS)?,
        baseline,
        criteria,
    );
    let version = memory.register_candidate(StrategyId::new(2), candidate)?;
    let evidence = StrategyStats::new(n as u64, hv, coverage, hf, vec![EvidenceRef::new(2)])?;
    let decision = memory.evaluate_and_promote(
        StrategyId::new(2),
        version,
        EvaluationEvidence::new(evidence).with_fixture_tag("kart_reflex_distillation"),
    )?;
    let mut reasons = Vec::new();
    let mut promoted = matches!(decision, PromotionDecision::Promoted { .. });
    if let PromotionDecision::Rejected { reasons: r, .. } = &decision {
        reasons.extend(r.iter().map(|x| format!("{x:?}")));
    }
    if promoted && cells_ratio < MIN_CELLS_RATIO {
        promoted = false;
        reasons.push(format!("degenerate_behaviour cells_ratio={cells_ratio:.2}"));
    }
    Ok((
        GateRecord {
            stage,
            plan_viability: pv,
            reflex_viability: hv,
            coverage,
            cells_ratio,
            promoted,
            reasons,
        },
        spent,
    ))
}

/// Public hook to run the promotion gate for an arbitrary candidate (used by tests).
pub fn evaluate_candidate_through_gate(
    cfg: &ExperimentConfig,
    seed: u64,
    track: &TrackMap,
    stage: usize,
    candidate: TabularPolicy,
) -> Result<GateRecord, ShrinkingError> {
    cfg.validate()?;
    let kin = KinematicBicycleConfig::default();
    let runner = Runner::new(&kin, track)?;
    let mut planner = Planner::new(&kin, track, MasterSeed::new(seed))?;
    Ok(gate_attempt(&runner, &mut planner, cfg, seed, stage, candidate)?.0)
}

/// Per-stage aggregate of one agent on one seed.
#[derive(Debug, Clone, PartialEq)]
pub struct StageStats {
    /// Stage index.
    pub stage: usize,
    /// Nominal budget (transitions per decision upper bound).
    pub nominal_budget: usize,
    /// Mean viability.
    pub viability: f64,
    /// Mean distinct cells.
    pub cells: f64,
    /// Planner transitions per decision (decision cost only).
    pub transitions_per_decision: f64,
    /// Fraction of decisions resolved by the reflex.
    pub reflex_fraction: f64,
    /// Decisions.
    pub decisions: u64,
    /// Transitions.
    pub transitions: u64,
    /// Whether a promoted reflex existed at the start of the stage.
    pub reflex_active: bool,
    /// Guard-checked actions.
    pub guard_checked: u64,
    /// Guard interventions.
    pub guard_interventions: u64,
}

/// Result of one agent (one seed).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRun {
    /// `true` for the distilling agent.
    pub distill: bool,
    /// Per-stage stats.
    pub stages: Vec<StageStats>,
    /// Gate attempts.
    pub gates: Vec<GateRecord>,
    /// Transitions spent on promotion fixtures (not in `stages`).
    pub gate_transitions: u64,
    /// Number of promotions.
    pub promotions: usize,
}

fn track_for(index: usize) -> TrackMap {
    if index.is_multiple_of(2) {
        TrackMap::rectangular_circuit()
    } else {
        TrackMap::chicane_circuit()
    }
}

/// Runs one agent over the whole shrinking schedule on one seed / track.
pub fn run_agent(
    cfg: &ExperimentConfig,
    seed: u64,
    track: &TrackMap,
    distill: bool,
) -> Result<AgentRun, ShrinkingError> {
    cfg.validate()?;
    let kin = KinematicBicycleConfig::default();
    let runner = Runner::new(&kin, track)?;
    let master = MasterSeed::new(seed);
    let mut planner = Planner::new(&kin, track, master)?;
    let mut counts: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut promoted: Option<TabularPolicy> = None;
    let mut run = AgentRun {
        distill,
        stages: Vec::new(),
        gates: Vec::new(),
        gate_transitions: 0,
        promotions: 0,
    };
    for stage in 0..=cfg.halvings {
        let pop = cfg.population_at(stage);
        let mut eps = Vec::new();
        let reflex_active = promoted.is_some();
        for e in 0..cfg.episodes_per_stage {
            let tag = (stage * cfg.episodes_per_stage + e) as u64;
            planner.configure(master, tag, pop, cfg.horizon);
            let ep_seed = seed ^ (tag << 20);
            let rec = if distill { Some(&mut counts) } else { None };
            eps.push(runner.episode(
                ep_seed,
                tag,
                cfg.ticks,
                &mut planner,
                promoted.as_ref(),
                rec,
            )?);
        }
        let decisions: u64 = eps.iter().map(|e| e.decisions).sum();
        let transitions: u64 = eps.iter().map(|e| e.transitions).sum();
        let reflex: u64 = eps.iter().map(|e| e.reflex_decisions).sum();
        run.stages.push(StageStats {
            stage,
            nominal_budget: cfg.nominal_budget(stage),
            viability: mean(&eps.iter().map(|e| e.viability).collect::<Vec<_>>()),
            cells: mean(&eps.iter().map(|e| e.cells as f64).collect::<Vec<_>>()),
            transitions_per_decision: transitions as f64 / decisions.max(1) as f64,
            reflex_fraction: reflex as f64 / decisions.max(1) as f64,
            decisions,
            transitions,
            reflex_active,
            guard_checked: eps.iter().map(|e| e.guard_checked).sum(),
            guard_interventions: eps.iter().map(|e| e.guard_interventions).sum(),
        });
        if distill && stage < cfg.halvings && !counts.is_empty() {
            let entries: Vec<(usize, usize, f64)> =
                counts.iter().map(|(&(s, a), &c)| (s, a, c)).collect();
            let candidate = TabularPolicy::from_entries(NUM_STATE_BUCKETS, NUM_ACTIONS, &entries)?;
            // The fixture is evaluated at the budget of the NEXT stage (what the reflex replaces).
            let (rec, spent) = gate_attempt(
                &runner,
                &mut planner,
                cfg,
                seed,
                stage + 1,
                candidate.clone(),
            )?;
            run.gate_transitions += spent;
            if rec.promoted {
                promoted = Some(candidate);
                run.promotions += 1;
            }
            run.gates.push(rec);
        }
    }
    Ok(run)
}

/// Verdict of the preregistered rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Equal viability and saving criterion both met.
    SavingConfirmed,
    /// Reflex promoted too rarely.
    NotPromoted,
    /// Viability (or behaviour diversity) fell below tolerance.
    ViabilityLoss,
    /// Equal viability held but the saving was below the criterion.
    NoSaving,
}

/// Full report.
#[derive(Debug, Clone, PartialEq)]
pub struct ShrinkingReport {
    /// Configuration.
    pub config: ExperimentConfig,
    /// Plan-only runs (per seed).
    pub plan_only: Vec<AgentRun>,
    /// Distilling runs (per seed).
    pub distilled: Vec<AgentRun>,
    /// Pooled paired viability difference (distilled - plan_only), mean over seeds.
    pub viability_diff_mean: f64,
    /// Dispersion (sd over seeds) of the per-seed viability difference.
    pub viability_diff_sd: f64,
    /// Cells ratio distilled/plan (pooled).
    pub cells_ratio: f64,
    /// Fraction of decision transitions saved.
    pub saving_fraction: f64,
    /// Saving including promotion-fixture cost.
    pub net_saving_fraction: f64,
    /// Overall fraction of decisions resolved by the reflex (distilled agent).
    pub reflex_fraction: f64,
    /// Seeds in which the reflex was promoted at least once.
    pub promoted_seeds: usize,
    /// Verdict.
    pub verdict: Verdict,
}

/// Runs the full paired experiment.
pub fn run_experiment(cfg: &ExperimentConfig) -> Result<ShrinkingReport, ShrinkingError> {
    cfg.validate()?;
    let mut plan_only = Vec::new();
    let mut distilled = Vec::new();
    for (i, &seed) in cfg.seeds.iter().enumerate() {
        let track = track_for(i);
        plan_only.push(run_agent(cfg, seed, &track, false)?);
        distilled.push(run_agent(cfg, seed, &track, true)?);
    }
    let diffs: Vec<f64> = plan_only
        .iter()
        .zip(&distilled)
        .map(|(p, d)| {
            mean(&d.stages.iter().map(|s| s.viability).collect::<Vec<_>>())
                - mean(&p.stages.iter().map(|s| s.viability).collect::<Vec<_>>())
        })
        .collect();
    let sum_t = |runs: &[AgentRun]| -> f64 {
        runs.iter()
            .flat_map(|r| &r.stages)
            .map(|s| s.transitions as f64)
            .sum()
    };
    let sum_d = |runs: &[AgentRun]| -> f64 {
        runs.iter()
            .flat_map(|r| &r.stages)
            .map(|s| s.decisions as f64)
            .sum()
    };
    let cells = |runs: &[AgentRun]| -> f64 {
        mean(
            &runs
                .iter()
                .flat_map(|r| r.stages.iter().map(|s| s.cells))
                .collect::<Vec<_>>(),
        )
    };
    let tpd_p = sum_t(&plan_only) / sum_d(&plan_only).max(1.0);
    let tpd_d = sum_t(&distilled) / sum_d(&distilled).max(1.0);
    let saving_fraction = if tpd_p > 0.0 {
        1.0 - tpd_d / tpd_p
    } else {
        0.0
    };
    let gate_t: f64 = distilled.iter().map(|r| r.gate_transitions as f64).sum();
    let net_saving_fraction = if sum_t(&plan_only) > 0.0 {
        1.0 - (sum_t(&distilled) + gate_t) / sum_t(&plan_only)
    } else {
        0.0
    };
    let reflex_total: f64 = distilled
        .iter()
        .flat_map(|r| &r.stages)
        .map(|s| s.reflex_fraction * s.decisions as f64)
        .sum();
    let reflex_fraction = reflex_total / sum_d(&distilled).max(1.0);
    let promoted_seeds = distilled.iter().filter(|r| r.promotions > 0).count();
    let viability_diff_mean = mean(&diffs);
    let cells_ratio = if cells(&plan_only) > 0.0 {
        cells(&distilled) / cells(&plan_only)
    } else {
        1.0
    };
    let verdict = if (promoted_seeds as f64) < MIN_PROMOTED_SEED_FRACTION * cfg.seeds.len() as f64 {
        Verdict::NotPromoted
    } else if viability_diff_mean < -VIABILITY_TOLERANCE || cells_ratio < MIN_CELLS_RATIO {
        Verdict::ViabilityLoss
    } else if saving_fraction < SAVING_MIN {
        Verdict::NoSaving
    } else {
        Verdict::SavingConfirmed
    };
    Ok(ShrinkingReport {
        config: cfg.clone(),
        plan_only,
        distilled,
        viability_diff_mean,
        viability_diff_sd: sd(&diffs),
        cells_ratio,
        saving_fraction,
        net_saving_fraction,
        reflex_fraction,
        promoted_seeds,
        verdict,
    })
}

impl ShrinkingReport {
    /// Plain-text tables (per-stage mean +- sd over seeds for both agents).
    pub fn format_tables(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "PREREG: {PREREG}");
        let _ = writeln!(
            out,
            "stage | nominal | pop | arm      | viability       | trans/decision    | reflex frac     | cells"
        );
        for s in 0..=self.config.halvings {
            for (name, runs) in [("plan", &self.plan_only), ("distill", &self.distilled)] {
                let col = |f: fn(&StageStats) -> f64| -> Vec<f64> {
                    runs.iter().map(|r| f(&r.stages[s])).collect()
                };
                let v = col(|x| x.viability);
                let t = col(|x| x.transitions_per_decision);
                let r = col(|x| x.reflex_fraction);
                let c = col(|x| x.cells);
                let _ = writeln!(
                    out,
                    "{s:>5} | {:>7} | {:>3} | {name:<8} | {:.3} +- {:.3}   | {:>8.1} +- {:>7.1} | {:.3} +- {:.3}   | {:.1}",
                    self.config.nominal_budget(s),
                    self.config.population_at(s),
                    mean(&v),
                    sd(&v),
                    mean(&t),
                    sd(&t),
                    mean(&r),
                    sd(&r),
                    mean(&c),
                );
            }
        }
        let _ = writeln!(
            out,
            "viability diff (distill - plan) = {:+.4} +- {:.4} (sd over seeds); cells ratio = {:.3}",
            self.viability_diff_mean, self.viability_diff_sd, self.cells_ratio
        );
        let _ = writeln!(
            out,
            "transitions saved at decision level = {:.3}; net of gate fixtures = {:.3}; reflex fraction overall = {:.3}; promoted seeds = {}/{}",
            self.saving_fraction,
            self.net_saving_fraction,
            self.reflex_fraction,
            self.promoted_seeds,
            self.config.seeds.len()
        );
        for (i, r) in self.distilled.iter().enumerate() {
            for g in &r.gates {
                let _ = writeln!(
                    out,
                    "gate seed#{i} stage<={} plan_v={:.3} hyb_v={:.3} cov={:.3} cells_ratio={:.2} promoted={} {:?}",
                    g.stage,
                    g.plan_viability,
                    g.reflex_viability,
                    g.coverage,
                    g.cells_ratio,
                    g.promoted,
                    g.reasons
                );
            }
        }
        let _ = writeln!(out, "VERDICT: {:?}", self.verdict);
        out
    }
}
