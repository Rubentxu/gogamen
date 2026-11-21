//! Unknown model as hidden state: posterior sampling in FMC (GOG-195, first slice).
//!
//! Instead of planning on the mean table of the learned model, each planning call draws ONE
//! transition table from the learner's Dirichlet posterior
//! ([`gogamen_core::sample_posterior_table`]) and runs the unchanged [`FmcPlanner`] on it. There
//! is **no exploration bonus** in this arm: any exploration is a side effect of acting on a
//! plausible random model (Thompson-style). Every decision records the sample it used
//! ([`SampleRecord`]: seed, sample id, table hash, model revision/checksum).
//!
//! # Preregistration (fixed in code BEFORE the first run)
//! - World: the 7x5 detour map of the hidden-physics lab. The agent learns the physics online
//!   from its own real transitions; budget `B` real transitions per `(arm, seed)`, identical
//!   for every arm (equal budget). Per-decision planning budget and FMC settings are identical.
//! - Ceiling: the privileged FMC planner over the true simulator
//!   ([`run_privileged_fmc_episode`]) on the same seed. Gap = `ceiling_return - eval_return`.
//! - Evaluation: every [`StudyConfig::checkpoint_every`] real transitions the frozen learner is
//!   evaluated by planning **on the mean table** (same for all arms: it measures the quality of
//!   the experience each arm collected). A checkpoint "closes the gap" iff
//!   `gap <= GAP_TOLERANCE`. Metric: experience-to-ceiling = first such checkpoint
//!   (transitions); right-censored at `B + checkpoint_every` when never reached.
//! - Seeds: development seeds are only for smoke tests; the report uses RESERVE seeds
//!   ([`StudyConfig::reserve_seeds`]) that nothing was tuned on, with identical seeds in
//!   every arm (paired).
//! - Rule (posterior vs each comparator, on paired per-seed differences
//!   `d = etc(comparator) - etc(posterior)`; positive = posterior is faster):
//!   BETTER iff `mean(d) >= MIN_GAIN` AND paired `z >= MIN_Z` AND wins > losses AND posterior
//!   censored count <= comparator censored count; WORSE iff the mirror image; otherwise NULL.
//!   Null and worse are valid, reported results.
//!
//! # Honest scoping
//! The progress-drive and Inspect/Ask arms are *ports* of GOG-187 / GOG-169 to this hidden
//! Grid (their original fixtures are a noise-cell grid and a door fixture): the progress arm
//! uses the core [`LearningProgressReader`] with the same one-step candidate scoring, and the
//! inspect arm is an EVOI-free "probe the least-evidenced action while some exist" rule. Both
//! are pure data collectors; they are compared on the experience they gather. No claim of
//! emergence or agency is made.

use std::fmt;

use gogamen_core::{
    Action, ActionSpace, Actuator, DeterministicRng, DriveSignal, EvidenceRef, FmcPlanner,
    FmcPlannerConfig, LearnedModelConfig, LearningPolicy, LearningProgressConfig,
    LearningProgressReader, LiveEnvironment, MasterSeed, ObservationId, PlanBudget, PlanRequest,
    PlanningContext, RngSource, SampledTableModel, SimDuration, State, StreamKey,
    TabularTransitionLearner, Terminal, TerminalReason, Tick, Transition, TransitionObservation,
    UnseenPairPolicy, mix_seed, posterior_mean_table, sample_posterior_table,
};

use crate::hidden_dynamics::{
    GridTask, HiddenGridError, PlanGridConfig, freeze_grid_model, grid_cell_scale,
    run_privileged_fmc_episode,
};
use crate::map::GridMap;
use crate::world::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldEnv};

/// Gap (reward units) under which an evaluation counts as having reached the ceiling.
pub const GAP_TOLERANCE: f64 = 1.0;
/// Minimum mean paired gain (transitions, one checkpoint) to call a difference BETTER / WORSE.
pub const MIN_GAIN: f64 = 150.0;
/// Minimum paired z (standard errors) to call a difference BETTER / WORSE.
pub const MIN_Z: f64 = 2.0;

/// Error of the study.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorFmcError(pub String);

impl fmt::Display for PosteriorFmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "posterior sampling fmc: {}", self.0)
    }
}
impl std::error::Error for PosteriorFmcError {}

fn err<E: fmt::Debug>(e: E) -> PosteriorFmcError {
    PosteriorFmcError(format!("{e:?}"))
}

impl From<HiddenGridError> for PosteriorFmcError {
    fn from(e: HiddenGridError) -> Self {
        err(e)
    }
}

/// Experimental arm; all arms get the same number of real transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Arm {
    /// FMC on the mean table of the learner (online).
    MeanPlan,
    /// FMC on one posterior sample per planning call, no bonus.
    PosteriorSample,
    /// GOG-187 port: one-step choice by learning-progress reading.
    ProgressDrive,
    /// GOG-169 port: probe the least-evidenced action while any exists, else plan on the mean.
    InspectAsk,
    /// Uniform random reference.
    Random,
}

impl Arm {
    /// All arms in report order.
    pub const ALL: [Self; 5] = [
        Self::MeanPlan,
        Self::PosteriorSample,
        Self::ProgressDrive,
        Self::InspectAsk,
        Self::Random,
    ];
    /// Stable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::MeanPlan => "mean_plan",
            Self::PosteriorSample => "posterior_sample",
            Self::ProgressDrive => "progress_drive",
            Self::InspectAsk => "inspect_ask",
            Self::Random => "random",
        }
    }
}

/// What a decision planned on: identity of the sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SampleRecord {
    /// Decision (real transition) index in the run.
    pub decision: u64,
    /// Sampler seed.
    pub seed: u64,
    /// Sample id within the seed's stream.
    pub sample_id: u64,
    /// Hash of the sampled table.
    pub table_hash: u64,
    /// Model revision the sample was drawn from.
    pub model_revision: u64,
    /// Model checksum the sample was drawn from.
    pub model_checksum: u64,
}

/// Study configuration.
#[derive(Debug, Clone)]
pub struct StudyConfig {
    /// World (the hidden physics).
    pub world: GridWorldConfig,
    /// Development seeds (smoke tests only).
    pub dev_seeds: Vec<u64>,
    /// Reserve (held-out) seeds used for the report.
    pub reserve_seeds: Vec<u64>,
    /// Real-transition budget per `(arm, seed)`.
    pub budget: usize,
    /// Evaluation period in real transitions.
    pub checkpoint_every: usize,
    /// Per-episode tick cap (learning and evaluation).
    pub max_ticks: u64,
    /// Dirichlet prior of the learner.
    pub prior_alpha: f64,
    /// Failure pseudocount of the model adapter.
    pub failure_pseudocount: f64,
    /// Planner settings shared by every planning arm and the ceiling.
    pub plan: PlanGridConfig,
    /// Evidence under which [`Arm::InspectAsk`] keeps probing a pair.
    pub inspect_min_evidence: u64,
    /// Learning-progress reader window.
    pub progress_window: usize,
}

impl StudyConfig {
    /// Default: detour map, budget 900, checkpoints every 150, 6 reserve seeds (reduced for
    /// runtime). Budget and checkpoint period were chosen on DEV seeds 1..=3 after a first
    /// 300-transition configuration left every arm at the floor (all runs censored); the
    /// reserve seeds were run once with this configuration.
    pub fn new() -> Self {
        let map =
            GridMap::from_ascii(&["#######", "#S.#.G#", "#..#..#", "#.....#", "#######"]).unwrap();
        let world = GridWorldConfig {
            map,
            ..GridWorldConfig::default()
        };
        let plan = PlanGridConfig::new(GridTask::from_world(&world));
        Self {
            world,
            dev_seeds: vec![1, 2, 3],
            reserve_seeds: (101..=106).collect(),
            budget: 900,
            checkpoint_every: 150,
            max_ticks: 30,
            prior_alpha: 0.05,
            failure_pseudocount: 0.5,
            plan,
            inspect_min_evidence: 2,
            progress_window: 4,
        }
    }
}

impl Default for StudyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one `(arm, seed)` run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunResult {
    /// Seed.
    pub seed: u64,
    /// Ceiling return of the privileged planner on this seed.
    pub ceiling: f64,
    /// First checkpoint (transitions) at which the gap closed; `budget + checkpoint_every` when censored.
    pub experience_to_ceiling: usize,
    /// Whether the gap never closed within the budget.
    pub censored: bool,
    /// Gap at the final checkpoint.
    pub final_gap: f64,
    /// Gap at every checkpoint.
    pub gaps: Vec<f64>,
    /// Distinct `(cell, action)` pairs with evidence at the end (coverage of experience).
    pub pairs_covered: usize,
    /// Samples used per decision (empty unless [`Arm::PosteriorSample`]).
    pub samples: Vec<SampleRecord>,
}

fn stay() -> Action {
    Action::discrete_in_space(0, &ActionSpace::discrete(GRID_ACTION_CARDINALITY).unwrap()).unwrap()
}

#[derive(Clone, Copy)]
enum Mode {
    Mean,
    Sample { seed: u64, id: u64 },
}

/// One FMC decision over a freshly frozen model of `learner`. Returns the action index and,
/// for sample mode, the record of the sample used.
fn plan_decision(
    cfg: &StudyConfig,
    learner: &TabularTransitionLearner,
    scale: &gogamen_core::TabularScale,
    pos: (usize, usize),
    tick: u64,
    planner_seed: MasterSeed,
    mode: Mode,
) -> Result<(usize, Option<SampleRecord>), PosteriorFmcError> {
    let model_cfg =
        LearnedModelConfig::new(cfg.failure_pseudocount, 2, UnseenPairPolicy::SamplePrior)
            .map_err(err)?;
    let model = freeze_grid_model(learner, scale.clone(), model_cfg)?;
    let (table, record) = match mode {
        Mode::Mean => (posterior_mean_table(&model).map_err(err)?, None),
        Mode::Sample { seed, id } => {
            let t = sample_posterior_table(&model, seed, id).map_err(err)?;
            let r = SampleRecord {
                decision: tick,
                seed: t.seed(),
                sample_id: t.sample_id(),
                table_hash: t.hash(),
                model_revision: t.revision(),
                model_checksum: t.checksum(),
            };
            (t, Some(r))
        }
    };
    let sampled = SampledTableModel::new(&model, table);
    let task = cfg.plan.task;
    let planner = FmcPlanner::new(
        FmcPlannerConfig::new(
            cfg.plan.population_size,
            cfg.plan.horizon,
            task.dt,
            cfg.plan.scoring,
            planner_seed,
        )
        .map_err(err)?,
    );
    let state = State::new(
        Tick::new(tick),
        SimDuration::ZERO,
        vec![pos.0 as f64 + 0.5, pos.1 as f64 + 0.5],
    )
    .map_err(err)?;
    let request = PlanRequest::from_state(
        state,
        gogamen_core::GenerativeModel::action_space(&sampled).clone(),
        PlanBudget::with_transitions(cfg.plan.budget_transitions),
    )
    .with_fallback(stay())
    .map_err(err)?;
    let cell = |s: &State| {
        (
            s.values()[0].floor() as usize,
            s.values()[1].floor() as usize,
        )
    };
    let reward = |prev: &State, _a: &Action, tr: &Transition| {
        if matches!(tr.terminal(), Terminal::Terminated(TerminalReason::Failure)) {
            task.hazard_reward
        } else {
            let (px, py) = cell(prev);
            let (nx, ny) = cell(tr.next_state());
            if (nx, ny) == (task.goal.x, task.goal.y) {
                task.goal_reward
            } else if (nx, ny) == (px, py) {
                task.wall_collision_reward
            } else {
                task.step_reward
            }
        }
    };
    let mut rng = DeterministicRng::new(planner_seed.derive_seed(StreamKey::Episode(tick)));
    let mut ctx = PlanningContext::new(&sampled, &mut rng);
    let (result, _) = planner
        .plan_with_trace(&mut ctx, &request, reward, |s| s.values().to_vec())
        .map_err(err)?;
    let idx = result
        .authorized_action()
        .and_then(|a| a.as_discrete())
        .map_or(0, |a| a.index());
    Ok((idx, record))
}

fn grid_action(idx: usize) -> Action {
    Action::discrete_in_space(
        idx,
        &ActionSpace::discrete(GRID_ACTION_CARDINALITY).unwrap(),
    )
    .unwrap()
}

/// Frozen-learner evaluation episode (no learning). Returns the real-world return.
fn eval_return(
    cfg: &StudyConfig,
    learner: &TabularTransitionLearner,
    scale: &gogamen_core::TabularScale,
    seed: MasterSeed,
    own_sampling: bool,
) -> Result<f64, PosteriorFmcError> {
    let mut env = GridWorldEnv::new(cfg.world.clone()).map_err(err)?;
    let mut tick = 0;
    while tick < cfg.max_ticks && !env.terminal().is_terminal() {
        let mode = if own_sampling {
            Mode::Sample {
                seed: seed.as_u64(),
                id: tick,
            }
        } else {
            Mode::Mean
        };
        let (a, _) = plan_decision(
            cfg,
            learner,
            scale,
            (env.position().x, env.position().y),
            tick,
            seed,
            mode,
        )?;
        env.apply(&grid_action(a)).map_err(err)?;
        tick += 1;
    }
    Ok(env.total_reward())
}

/// Runs one `(arm, seed)`: online learning for `budget` real transitions with periodic
/// frozen mean-plan evaluation against the privileged ceiling.
pub fn run_arm(cfg: &StudyConfig, arm: Arm, seed: u64) -> Result<RunResult, PosteriorFmcError> {
    let scale = grid_cell_scale(&cfg.world.map)?;
    let master = MasterSeed::new(seed);
    let mut ceiling_plan = cfg.plan.clone();
    ceiling_plan.seed = master;
    let ceiling =
        run_privileged_fmc_episode(&cfg.world, &ceiling_plan, cfg.max_ticks)?.total_reward;

    let mut learner = TabularTransitionLearner::new(
        scale.num_states(),
        GRID_ACTION_CARDINALITY,
        LearningPolicy::new(cfg.prior_alpha).map_err(err)?,
    )
    .map_err(err)?;
    let reader_cfg = LearningProgressConfig::new(
        cfg.progress_window,
        scale.num_states(),
        (scale.num_states() as f64).ln(),
    )
    .map_err(err)?;
    let mut reader = LearningProgressReader::new(reader_cfg);
    let mut env = GridWorldEnv::new(cfg.world.clone()).map_err(err)?;
    let mut arm_rng = DeterministicRng::new(mix_seed(seed, 0xA12 + arm as u64));
    let sample_seed = mix_seed(seed, 0x5A3D);
    let mut ep_tick = 0u64;
    let mut samples = Vec::new();
    let mut gaps = Vec::new();
    let mut etc: Option<usize> = None;
    let bucket = |p: crate::map::GridPos| {
        scale
            .bucket_of(&[p.x as f64 + 0.5, p.y as f64 + 0.5])
            .map_err(err)
    };

    for t in 0..cfg.budget {
        if env.terminal().is_terminal() || ep_tick >= cfg.max_ticks {
            LiveEnvironment::reset(&mut env).map_err(err)?;
            ep_tick = 0;
        }
        let before = env.position();
        let from = bucket(before)?;
        let pos = (before.x, before.y);
        let planner_seed = MasterSeed::new(mix_seed(seed, 0x9000 + t as u64));
        let idx = match arm {
            Arm::Random => (arm_rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize,
            Arm::MeanPlan => {
                plan_decision(
                    cfg,
                    &learner,
                    &scale,
                    pos,
                    ep_tick,
                    planner_seed,
                    Mode::Mean,
                )?
                .0
            }
            Arm::PosteriorSample => {
                let (a, rec) = plan_decision(
                    cfg,
                    &learner,
                    &scale,
                    pos,
                    ep_tick,
                    planner_seed,
                    Mode::Sample {
                        seed: sample_seed,
                        id: t as u64,
                    },
                )?;
                let mut rec = rec.expect("sample mode records its sample");
                rec.decision = t as u64;
                samples.push(rec);
                a
            }
            Arm::InspectAsk => {
                let mut least: Option<(u64, usize)> = None;
                for a in 0..GRID_ACTION_CARDINALITY {
                    let n = learner.pair_observation_count(from, a).map_err(err)?;
                    if n < cfg.inspect_min_evidence && least.is_none_or(|(m, _)| n < m) {
                        least = Some((n, a));
                    }
                }
                match least {
                    Some((_, a)) => a,
                    None => {
                        plan_decision(
                            cfg,
                            &learner,
                            &scale,
                            pos,
                            ep_tick,
                            planner_seed,
                            Mode::Mean,
                        )?
                        .0
                    }
                }
            }
            Arm::ProgressDrive => {
                let mut best = (0usize, f64::NEG_INFINITY);
                for a in 0..GRID_ACTION_CARDINALITY {
                    let explore = if learner.pair_observation_count(from, a).map_err(err)? > 0 {
                        let p = learner.predict(from, a).map_err(err)?;
                        let next = p
                            .iter()
                            .enumerate()
                            .fold((0usize, f64::NEG_INFINITY), |b, (i, &q)| {
                                if q > b.1 { (i, q) } else { b }
                            })
                            .0;
                        reader
                            .read(next, DriveSignal::LearningProgress)
                            .map_err(err)?
                            .value_or(1.0)
                    } else {
                        1.0
                    };
                    let score = explore + 0.01 * arm_rng.next_f64();
                    if score > best.1 {
                        best = (a, score);
                    }
                }
                best.0
            }
        };

        let outcome = env.apply(&grid_action(idx)).map_err(err)?;
        ep_tick += 1;
        let after = env.position();
        let sample = scale
            .sample_from_transition(
                &[before.x as f64 + 0.5, before.y as f64 + 0.5],
                idx,
                &[after.x as f64 + 0.5, after.y as f64 + 0.5],
                outcome.terminal(),
            )
            .map_err(err)?;
        let oid = ObservationId::new(t as u64 + 1);
        let ev = EvidenceRef::new(20_000 + t as u64);
        let obs = if sample.is_terminal_failure() {
            TransitionObservation::failure(
                oid,
                ev,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        } else {
            TransitionObservation::new(
                oid,
                ev,
                sample.state_bucket(),
                sample.action_idx(),
                sample.next_state_bucket(),
            )
        };
        let out = learner.observe(&obs).map_err(err)?;
        if let Some(e) = out.pre_update_eval() {
            reader.record(from, e.log_loss()).map_err(err)?;
        }

        let used = t + 1;
        if used % cfg.checkpoint_every == 0 {
            let ret = eval_return(
                cfg,
                &learner,
                &scale,
                MasterSeed::new(mix_seed(seed, 0xE000 + used as u64)),
                false,
            )?;
            let gap = ceiling - ret;
            gaps.push(gap);
            if etc.is_none() && gap <= GAP_TOLERANCE {
                etc = Some(used);
            }
        }
    }
    let mut pairs = 0;
    for s in 0..scale.num_states() {
        for a in 0..GRID_ACTION_CARDINALITY {
            if learner.pair_observation_count(s, a).map_err(err)? > 0 {
                pairs += 1;
            }
        }
    }
    Ok(RunResult {
        seed,
        ceiling,
        experience_to_ceiling: etc.unwrap_or(cfg.budget + cfg.checkpoint_every),
        censored: etc.is_none(),
        final_gap: gaps.last().copied().unwrap_or(f64::NAN),
        gaps,
        pairs_covered: pairs,
        samples,
    })
}

/// Mean / sample standard deviation / min / median / max.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation (n-1).
    pub std: f64,
    /// Minimum.
    pub min: f64,
    /// Median.
    pub median: f64,
    /// Maximum.
    pub max: f64,
}

/// Dispersion summary.
pub fn summarize(xs: &[f64]) -> Summary {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let std = if xs.len() > 1 {
        (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    Summary {
        mean,
        std,
        min: s[0],
        median: s[s.len() / 2],
        max: s[s.len() - 1],
    }
}

/// Paired z of `d` (mean / standard error); 0 when the differences have no spread and no mean.
pub fn paired_z(d: &[f64]) -> f64 {
    let s = summarize(d);
    let se = s.std / (d.len() as f64).sqrt();
    if se == 0.0 {
        if s.mean == 0.0 {
            0.0
        } else {
            s.mean.signum() * f64::INFINITY
        }
    } else {
        s.mean / se
    }
}

/// Verdict of the preregistered rule for posterior sampling vs one comparator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Posterior sampling closes the gap with less experience, per the rule.
    Better,
    /// No difference meeting the rule (the null result).
    Null,
    /// The comparator closes the gap with less experience, per the mirrored rule.
    Worse,
}

/// Paired comparison of posterior sampling against one comparator.
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    /// Comparator arm.
    pub versus: Arm,
    /// Mean of `etc(comparator) - etc(posterior)`.
    pub mean_gain: f64,
    /// Paired z.
    pub z: f64,
    /// Seeds where posterior was faster.
    pub wins: usize,
    /// Seeds where the comparator was faster.
    pub losses: usize,
    /// Seeds tied.
    pub ties: usize,
    /// Verdict per the preregistered rule.
    pub verdict: Verdict,
}

/// Per-arm aggregate.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmReport {
    /// Arm.
    pub arm: Arm,
    /// Per-seed runs (same seeds in every arm).
    pub runs: Vec<RunResult>,
    /// Experience-to-ceiling summary (censored values at `budget + checkpoint_every`).
    pub etc: Summary,
    /// Seeds that never closed the gap.
    pub censored: usize,
    /// Final gap summary.
    pub final_gap: Summary,
    /// Covered `(cell, action)` pairs summary.
    pub pairs: Summary,
}

/// Full study report.
#[derive(Debug, Clone, PartialEq)]
pub struct StudyReport {
    /// One report per arm in [`Arm::ALL`] order.
    pub arms: Vec<ArmReport>,
    /// Posterior sampling vs each other arm.
    pub comparisons: Vec<Comparison>,
}

impl StudyReport {
    /// Report of `arm`.
    pub fn arm(&self, arm: Arm) -> &ArmReport {
        self.arms
            .iter()
            .find(|a| a.arm == arm)
            .expect("arm present")
    }
}

/// Applies the preregistered rule to paired per-seed experience-to-ceiling values.
pub fn compare(versus: Arm, posterior: &ArmReport, other: &ArmReport) -> Comparison {
    let d: Vec<f64> = other
        .runs
        .iter()
        .zip(&posterior.runs)
        .map(|(o, p)| o.experience_to_ceiling as f64 - p.experience_to_ceiling as f64)
        .collect();
    let mean_gain = summarize(&d).mean;
    let z = paired_z(&d);
    let wins = d.iter().filter(|x| **x > 0.0).count();
    let losses = d.iter().filter(|x| **x < 0.0).count();
    let verdict = if mean_gain >= MIN_GAIN
        && z >= MIN_Z
        && wins > losses
        && posterior.censored <= other.censored
    {
        Verdict::Better
    } else if mean_gain <= -MIN_GAIN
        && z <= -MIN_Z
        && losses > wins
        && posterior.censored >= other.censored
    {
        Verdict::Worse
    } else {
        Verdict::Null
    };
    Comparison {
        versus,
        mean_gain,
        z,
        wins,
        losses,
        ties: d.len() - wins - losses,
        verdict,
    }
}

/// Runs every arm on `seeds` (paired) and applies the preregistered rule.
pub fn run_study(cfg: &StudyConfig, seeds: &[u64]) -> Result<StudyReport, PosteriorFmcError> {
    let mut arms = Vec::new();
    for arm in Arm::ALL {
        let runs = seeds
            .iter()
            .map(|&s| run_arm(cfg, arm, s))
            .collect::<Result<Vec<_>, _>>()?;
        let col = |f: fn(&RunResult) -> f64| -> Vec<f64> { runs.iter().map(f).collect() };
        arms.push(ArmReport {
            arm,
            etc: summarize(&col(|r| r.experience_to_ceiling as f64)),
            censored: runs.iter().filter(|r| r.censored).count(),
            final_gap: summarize(&col(|r| r.final_gap)),
            pairs: summarize(&col(|r| r.pairs_covered as f64)),
            runs,
        });
    }
    let post = arms
        .iter()
        .find(|a| a.arm == Arm::PosteriorSample)
        .expect("posterior arm")
        .clone();
    let comparisons = arms
        .iter()
        .filter(|a| a.arm != Arm::PosteriorSample)
        .map(|a| compare(a.arm, &post, a))
        .collect();
    Ok(StudyReport { arms, comparisons })
}
