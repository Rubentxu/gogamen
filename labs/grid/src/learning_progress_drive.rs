//! GOG-187: a drive read from learning progress versus a drive read from raw prediction error.
//!
//! World: a small open grid with **one pure-noise cell** ("TV static"): whatever the action,
//! leaving that cell lands uniformly in the 3x3 block around it. Every other cell is
//! deterministic. The agent learns `P(next cell | cell, action)` with the core
//! [`TabularTransitionLearner`]; the prequential `log_loss` of each executed transition is
//! fed (per starting cell) to a [`LearningProgressReader`].
//!
//! Three arms share everything (world, learner, utility machinery, seeds) and differ only in
//! the signal the exploration drive reads:
//! - [`Arm::ErrorDriven`]: recent raw `log_loss` ([`DriveSignal::RawError`]),
//! - [`Arm::ProgressDriven`]: recent drop of `log_loss` ([`DriveSignal::LearningProgress`]),
//! - [`Arm::Random`]: no drive, only seeded tie-breaking jitter (reference for "uniform").
//!
//! Exploration needs no dedicated mechanism: the `explore` objective is a
//! [`WeightedObjective`] modulated by the drive's deficit inside [`DynamicUtility`]; each
//! candidate action is scored by the reading of the region it is predicted to reach.
//!
//! Every action goes through [`ActionGuard`]: [`GuardedGrid`] is the only way to move the world
//! and executes nothing that the guard does not `Allow` (or replace by an explicit `Fallback`).
//!
//! Nothing here claims agency or emergence: it is a sensitivity study of a 1-step greedy
//! agent on a 5x5 grid. [`TrapVerdict::Refuted`] / [`TrapVerdict::Opposite`] are valid results.

use gogamen_core::{
    Action, ActionCandidate, ActionGuard, ActionProposal, ActionSpace, DeterministicRng,
    DomainPrecondition, DriveId, DriveModality, DriveSignal, DynamicUtility, EndowmentProvenance,
    EvidenceRef, GuardContext, GuardDecision, GuardRejectionReason, HomeostaticState,
    LearningPolicy, LearningProgressConfig, LearningProgressReader, NoPrecondition, ObservationId,
    RegionReading, RngSource, TabularTransitionLearner, Terminal, Tick, TransitionObservation,
    UtilityEvaluator, WeightedObjective, signal_drive_spec,
};

use crate::viability_prior::EndowmentLedger;

/// Number of actions: North, South, East, West, Stay.
pub const N_ACTIONS: usize = 5;

/// Error of this lab (stringly, the lab is not a library surface).
#[derive(Debug, Clone, PartialEq)]
pub struct LpDriveError(pub String);

impl std::fmt::Display for LpDriveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "learning progress drive lab: {}", self.0)
    }
}

impl<E: std::error::Error> From<E> for LpDriveError {
    fn from(e: E) -> Self {
        Self(e.to_string())
    }
}

// ---------------------------------------------------------------------------------------
// Preregistration (written before any result was observed).
// ---------------------------------------------------------------------------------------

/// Preregistered decision rule. Fixed in code before the first run of the study.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreregisteredRule {
    /// H1: the error-driven noise-cell visit share must be at least this multiple of the
    /// uniform share `1 / cells`.
    pub error_min_ratio_vs_uniform: f64,
    /// H1: the progress-driven share must be at most this multiple of the uniform share.
    pub progress_max_ratio_vs_uniform: f64,
    /// H1: minimum paired z (error - progress share) in standard errors; `<= -z` means
    /// [`TrapVerdict::Opposite`].
    pub min_paired_z: f64,
    /// H2: normalised non-noise progress level at which the drive counts as "reappeared".
    pub reappear_threshold: f64,
    /// H2: error counts as recovered when its moving average is back within this fraction
    /// of the spike above the pre-change baseline.
    pub recovery_fraction_of_spike: f64,
    /// H2: minimum fraction of seeds in which both events must occur before the horizon.
    pub recovery_min_fraction: f64,
}

impl PreregisteredRule {
    /// The rule:
    /// - H1 CONFIRMED iff error share >= 2x uniform AND progress share <= 1.5x uniform AND
    ///   paired z >= 2; OPPOSITE iff paired z <= -2; otherwise REFUTED.
    /// - H2 CONFIRMED iff in >= 80% of seeds the progress drive reaches 0.03 after the rule
    ///   change AND the error returns within 25% of its spike, both before the horizon.
    pub const PREREGISTERED: Self = Self {
        error_min_ratio_vs_uniform: 2.0,
        progress_max_ratio_vs_uniform: 1.5,
        min_paired_z: 2.0,
        reappear_threshold: 0.03,
        recovery_fraction_of_spike: 0.25,
        recovery_min_fraction: 0.8,
    };
}

/// H1 verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrapVerdict {
    /// Error-driven trapped, progress-driven not.
    Confirmed,
    /// Neither confirmed nor opposite.
    Refuted,
    /// Progress-driven visits the noise cell significantly MORE than error-driven.
    Opposite,
}

/// H2 verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryVerdict {
    /// Drive reappears and error drops again in enough seeds.
    Confirmed,
    /// At least one of the two events fails the rule.
    Refuted,
}

// ---------------------------------------------------------------------------------------
// World.
// ---------------------------------------------------------------------------------------

/// Transition rule of the deterministic cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// North/South/East/West move as named.
    Nominal,
    /// Controls reversed: North<->South and East<->West swapped.
    Reversed,
}

/// Hidden grid with one pure-noise cell. The world state is private; only
/// [`GuardedGrid::step`] can advance it.
#[derive(Debug, Clone)]
pub struct NoiseGrid {
    w: usize,
    h: usize,
    noise: usize,
    rule: Rule,
    pos: usize,
    rng: DeterministicRng,
}

impl NoiseGrid {
    fn new(w: usize, h: usize, noise: (usize, usize), start: (usize, usize), seed: u64) -> Self {
        Self {
            w,
            h,
            noise: noise.1 * w + noise.0,
            rule: Rule::Nominal,
            pos: start.1 * w + start.0,
            rng: DeterministicRng::new(seed),
        }
    }

    /// Number of cells (= learner states = regions).
    pub const fn cells(&self) -> usize {
        self.w * self.h
    }

    /// Current cell.
    pub const fn position(&self) -> usize {
        self.pos
    }

    /// Noise cell index.
    pub const fn noise_cell(&self) -> usize {
        self.noise
    }

    /// Current rule.
    pub const fn rule(&self) -> Rule {
        self.rule
    }

    /// Changes the physics (the "rule change").
    pub fn set_rule(&mut self, rule: Rule) {
        self.rule = rule;
    }

    fn shift(&self, cell: usize, a: usize) -> usize {
        let (x, y) = ((cell % self.w) as i64, (cell / self.w) as i64);
        let (dx, dy) = match a {
            0 => (0, -1),
            1 => (0, 1),
            2 => (1, 0),
            3 => (-1, 0),
            _ => (0, 0),
        };
        let (nx, ny) = (x + dx, y + dy);
        if nx < 0 || ny < 0 || nx >= self.w as i64 || ny >= self.h as i64 {
            cell
        } else {
            (ny as usize) * self.w + nx as usize
        }
    }

    /// Nominal geometry (what the agent assumes for never-tried pairs).
    pub fn nominal_next(&self, cell: usize, a: usize) -> usize {
        self.shift(cell, a)
    }

    fn apply(&mut self, a: usize) -> (usize, usize) {
        let from = self.pos;
        let to = if from == self.noise {
            let (nx, ny) = ((from % self.w) as i64, (from / self.w) as i64);
            let mut block = Vec::new();
            for y in (ny - 1).max(0)..=(ny + 1).min(self.h as i64 - 1) {
                for x in (nx - 1).max(0)..=(nx + 1).min(self.w as i64 - 1) {
                    block.push((y as usize) * self.w + x as usize);
                }
            }
            block[self.rng.next_usize_unbiased(block.len())]
        } else {
            let eff = match (self.rule, a) {
                (Rule::Reversed, 0) => 1,
                (Rule::Reversed, 1) => 0,
                (Rule::Reversed, 2) => 3,
                (Rule::Reversed, 3) => 2,
                (_, other) => other,
            };
            self.shift(from, eff)
        };
        self.pos = to;
        (from, to)
    }
}

/// Outcome of a guarded step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepOutcome {
    /// The guard authorised the action (or an explicit fallback) and it was executed.
    Executed {
        /// Executed action index.
        action: usize,
        /// Cell before.
        from: usize,
        /// Cell after.
        to: usize,
    },
    /// The guard rejected the proposal; the world did not move.
    Denied(GuardRejectionReason),
    /// The guard ordered an explicit no-op; the world did not move.
    NoOp,
}

/// The world behind an [`ActionGuard`]: the only path that mutates it.
pub struct GuardedGrid<P = NoPrecondition> {
    world: NoiseGrid,
    guard: ActionGuard<P>,
    space: ActionSpace,
    executed: u64,
    denied: u64,
    noops: u64,
}

impl<P: DomainPrecondition> GuardedGrid<P> {
    fn new(world: NoiseGrid, guard: ActionGuard<P>) -> Result<Self, LpDriveError> {
        Ok(Self {
            world,
            guard,
            space: ActionSpace::discrete(N_ACTIONS)?,
            executed: 0,
            denied: 0,
            noops: 0,
        })
    }

    /// Read-only world view.
    pub const fn world(&self) -> &NoiseGrid {
        &self.world
    }

    /// Changes the physics.
    pub fn set_rule(&mut self, rule: Rule) {
        self.world.set_rule(rule);
    }

    /// Counters `(executed, denied, noop)`.
    pub const fn counters(&self) -> (u64, u64, u64) {
        (self.executed, self.denied, self.noops)
    }

    /// Evaluates `proposal` with the guard and executes only what it authorises.
    pub fn step(&mut self, proposal: &ActionProposal, tick: Tick) -> StepOutcome {
        let ctx = GuardContext::now(&self.space, tick, Terminal::Ongoing);
        let action = match self.guard.evaluate(proposal, &ctx) {
            GuardDecision::Allow(a) | GuardDecision::Fallback(a) => a,
            GuardDecision::NoOp => {
                self.noops += 1;
                return StepOutcome::NoOp;
            }
            GuardDecision::Reject(reason) => {
                self.denied += 1;
                return StepOutcome::Denied(reason);
            }
        };
        let idx = action.as_discrete().map_or(0, |d| d.index());
        let (from, to) = self.world.apply(idx);
        self.executed += 1;
        StepOutcome::Executed {
            action: idx,
            from,
            to,
        }
    }
}

/// Builds a guarded world for the configuration (used by tests and the study).
pub fn new_guarded_grid<P: DomainPrecondition>(
    cfg: &StudyConfig,
    seed: u64,
    guard: ActionGuard<P>,
) -> Result<GuardedGrid<P>, LpDriveError> {
    let world = NoiseGrid::new(
        cfg.width,
        cfg.height,
        cfg.noise,
        cfg.start,
        seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA5A5,
    );
    GuardedGrid::new(world, guard)
}

// ---------------------------------------------------------------------------------------
// Study configuration and per-run metrics.
// ---------------------------------------------------------------------------------------

/// Experimental arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// Drive reads raw prediction error.
    ErrorDriven,
    /// Drive reads learning progress.
    ProgressDriven,
    /// No drive (seeded jitter only).
    Random,
}

impl Arm {
    /// Stable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::ErrorDriven => "error_driven",
            Self::ProgressDriven => "progress_driven",
            Self::Random => "random",
        }
    }

    const fn signal(self) -> Option<DriveSignal> {
        match self {
            Self::ErrorDriven => Some(DriveSignal::RawError),
            Self::ProgressDriven => Some(DriveSignal::LearningProgress),
            Self::Random => None,
        }
    }
}

/// Study configuration.
#[derive(Debug, Clone)]
pub struct StudyConfig {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// Noise cell `(x, y)`.
    pub noise: (usize, usize),
    /// Start cell `(x, y)`.
    pub start: (usize, usize),
    /// Total ticks per run.
    pub horizon: u64,
    /// Tick at which the rule changes to [`Rule::Reversed`].
    pub change_tick: u64,
    /// Ticks excluded from the trap measure (initial learning).
    pub warmup: u64,
    /// Half-window of the reader (samples per region).
    pub window: usize,
    /// Moving-average length (non-noise samples) for the error recovery.
    pub error_ma: usize,
    /// Length (ticks) of the pre-change reference period.
    pub baseline_ticks: u64,
    /// Dirichlet prior of the learner.
    pub prior_alpha: f64,
    /// Weight of the seeded jitter tie-break term.
    pub jitter_weight: f64,
    /// Deficit gain of the exploration objective.
    pub deficit_gain: f64,
    /// Paired seeds.
    pub seeds: Vec<u64>,
    /// Preregistered rule.
    pub rule: PreregisteredRule,
}

impl StudyConfig {
    /// Default study: 5x5 grid, noise at (3,1), 16 paired seeds.
    pub fn new() -> Self {
        Self {
            width: 5,
            height: 5,
            noise: (3, 1),
            start: (0, 4),
            horizon: 1500,
            change_tick: 600,
            warmup: 150,
            window: 4,
            error_ma: 50,
            baseline_ticks: 100,
            prior_alpha: 0.05,
            jitter_weight: 0.01,
            deficit_gain: 1.0,
            seeds: (1..=16).collect(),
            rule: PreregisteredRule::PREREGISTERED,
        }
    }
}

impl Default for StudyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics of one `(arm, seed)` run.
#[derive(Debug, Clone, PartialEq)]
pub struct RunMetrics {
    /// Seed.
    pub seed: u64,
    /// Ticks spent in the noise cell during `[warmup, change_tick)`.
    pub noise_visits: u64,
    /// Share of those ticks (`noise_visits / (change_tick - warmup)`).
    pub noise_share: f64,
    /// Mean non-noise `log_loss` in the reference period before the change.
    pub baseline_loss: f64,
    /// Peak moving-average non-noise `log_loss` after the change.
    pub peak_loss: f64,
    /// Ticks from the change until the error is back within the preregistered fraction of
    /// the spike (`None` = not before the horizon).
    pub error_recovery_delay: Option<u64>,
    /// Mean non-noise progress level in the reference period before the change.
    pub pre_progress_level: f64,
    /// Ticks from the change until the non-noise progress level reaches the threshold.
    pub drive_reappear_delay: Option<u64>,
    /// Guard counters `(executed, denied, noop)`.
    pub guard_counters: (u64, u64, u64),
    /// Final mean non-noise progress level.
    pub final_progress_level: f64,
}

/// Mean/dispersion summary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// Sample size.
    pub n: usize,
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

/// Summarises a sample (`n = 0` gives NaN fields).
pub fn summarize(xs: &[f64]) -> Summary {
    if xs.is_empty() {
        return Summary {
            n: 0,
            mean: f64::NAN,
            std: f64::NAN,
            min: f64::NAN,
            median: f64::NAN,
            max: f64::NAN,
        };
    }
    let n = xs.len();
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = if n > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)
    } else {
        0.0
    };
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let median = if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    };
    Summary {
        n,
        mean,
        std: var.sqrt(),
        min: s[0],
        median,
        max: s[n - 1],
    }
}

/// Paired z of `a - b` (mean difference over its standard error).
pub fn paired_z(a: &[f64], b: &[f64]) -> f64 {
    let d: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let s = summarize(&d);
    let se = s.std / (d.len() as f64).sqrt();
    if se == 0.0 {
        if s.mean > 0.0 {
            f64::INFINITY
        } else if s.mean < 0.0 {
            f64::NEG_INFINITY
        } else {
            0.0
        }
    } else {
        s.mean / se
    }
}

fn progress_level(reader: &LearningProgressReader, noise: usize) -> f64 {
    let (mut sum, mut k) = (0.0, 0usize);
    for r in (0..reader.config().regions()).filter(|&r| r != noise) {
        if let Ok(RegionReading::Value(v)) = reader.read(r, DriveSignal::LearningProgress) {
            sum += v;
            k += 1;
        }
    }
    if k == 0 { 0.0 } else { sum / k as f64 }
}

/// The identifier of the exploration drive of an arm.
fn drive_id_of(signal: DriveSignal) -> Result<DriveId, LpDriveError> {
    Ok(DriveId::new(signal.label())?)
}

/// Runs one episode with the strict guard and no extra precondition.
pub fn run_episode(cfg: &StudyConfig, arm: Arm, seed: u64) -> Result<RunMetrics, LpDriveError> {
    run_episode_guarded(cfg, arm, seed, ActionGuard::strict())
}

/// Runs one episode with a caller-supplied guard.
pub fn run_episode_guarded<P: DomainPrecondition>(
    cfg: &StudyConfig,
    arm: Arm,
    seed: u64,
    guard: ActionGuard<P>,
) -> Result<RunMetrics, LpDriveError> {
    let mut env = new_guarded_grid(cfg, seed, guard)?;
    let cells = env.world().cells();
    let noise = env.world().noise_cell();
    let mut agent_rng = DeterministicRng::new(seed ^ 0x5EED_0000_0000_0001);

    let mut learner = TabularTransitionLearner::new(
        cells,
        N_ACTIONS,
        LearningPolicy::new(cfg.prior_alpha).map_err(LpDriveError::from)?,
    )?;
    let reader_cfg = LearningProgressConfig::new(cfg.window, cells, (cells as f64).ln())?;
    let mut reader = LearningProgressReader::new(reader_cfg);

    // Drive machinery: the explore objective is modulated by the drive's deficit.
    let signal = arm.signal();
    let (drive_id, mut state, utility) = match signal {
        Some(sig) => {
            let id = drive_id_of(sig)?;
            let spec = signal_drive_spec(id.clone(), 0.0, cfg.deficit_gain)?;
            let state = HomeostaticState::new(vec![spec.clone()])?;
            let utility = DynamicUtility::new(vec![
                WeightedObjective::from_drive("explore", &spec)?,
                WeightedObjective::static_weight("jitter", cfg.jitter_weight)?,
            ])?;
            (Some(id), state, utility)
        }
        None => (
            None,
            HomeostaticState::new(vec![])?,
            DynamicUtility::new(vec![WeightedObjective::static_weight(
                "jitter",
                cfg.jitter_weight,
            )?])?,
        ),
    };
    let evaluator = UtilityEvaluator::new(utility);
    let space = ActionSpace::discrete(N_ACTIONS)?;

    let mut noise_visits = 0u64;
    let mut nn_losses: Vec<(u64, f64)> = Vec::new(); // (tick, loss) for non-noise starts
    let mut levels: Vec<f64> = Vec::with_capacity(cfg.horizon as usize);

    for t in 0..cfg.horizon {
        if t == cfg.change_tick {
            env.set_rule(Rule::Reversed);
        }
        let s = env.world().position();
        if t >= cfg.warmup && t < cfg.change_tick && s == noise {
            noise_visits += 1;
        }

        // Drive reading (global level): optimistic for regions without enough data.
        if let (Some(sig), Some(id)) = (signal, &drive_id) {
            let level = (0..cells)
                .map(|r| reader.read(r, sig).map_or(1.0, |x| x.value_or(1.0)))
                .fold(0.0_f64, f64::max);
            state.set_reading(id, level)?;
        }

        // Candidates: reading of the region each action is predicted to reach, using only
        // the agent's own learned model (nominal geometry for never-tried pairs).
        let mut candidates = Vec::with_capacity(N_ACTIONS);
        for a in 0..N_ACTIONS {
            let next = if learner.pair_observation_count(s, a)? > 0 {
                let p = learner.predict(s, a)?;
                p.iter()
                    .enumerate()
                    .fold((0usize, f64::NEG_INFINITY), |best, (i, &q)| {
                        if q > best.1 { (i, q) } else { best }
                    })
                    .0
            } else {
                env.world().nominal_next(s, a)
            };
            let explore = match signal {
                Some(sig) => reader.read(next, sig)?.value_or(1.0),
                None => 0.0,
            };
            let mut cand = ActionCandidate::new(Action::discrete_in_space(a, &space)?)
                .with_score("jitter", agent_rng.next_f64())?;
            if signal.is_some() {
                cand = cand.with_score("explore", explore)?;
            }
            candidates.push(cand);
        }
        let outcome = evaluator.evaluate_candidates(&candidates, &state)?;
        let chosen = outcome
            .selected()
            .ok_or_else(|| LpDriveError("no selectable action".into()))?
            .action()
            .clone();

        // Every action goes through the guard.
        let proposal = ActionProposal::authorized(chosen, Tick::new(t));
        if let StepOutcome::Executed { action, from, to } = env.step(&proposal, Tick::new(t)) {
            let obs = TransitionObservation::new(
                ObservationId::new(t + 1),
                EvidenceRef::new(1_000_000 + t),
                from,
                action,
                to,
            );
            let out = learner.observe(&obs)?;
            if let Some(eval) = out.pre_update_eval() {
                reader.record(from, eval.log_loss())?;
                if from != noise {
                    nn_losses.push((t, eval.log_loss()));
                }
            }
        }
        levels.push(progress_level(&reader, noise));
    }

    Ok(compute_metrics(
        cfg,
        seed,
        noise_visits,
        &nn_losses,
        &levels,
        env.counters(),
    ))
}

fn compute_metrics(
    cfg: &StudyConfig,
    seed: u64,
    noise_visits: u64,
    nn_losses: &[(u64, f64)],
    levels: &[f64],
    guard_counters: (u64, u64, u64),
) -> RunMetrics {
    let pre_from = cfg.change_tick.saturating_sub(cfg.baseline_ticks);
    let base: Vec<f64> = nn_losses
        .iter()
        .filter(|(t, _)| *t >= pre_from && *t < cfg.change_tick)
        .map(|(_, l)| *l)
        .collect();
    let baseline_loss = if base.is_empty() {
        0.0
    } else {
        summarize(&base).mean
    };

    // Moving average of the post-change non-noise losses (window of `error_ma` samples,
    // spanning the change so the pre-change samples dilute the first points).
    let post_start = nn_losses.partition_point(|(t, _)| *t < cfg.change_tick);
    let mut ma: Vec<(u64, f64)> = Vec::new();
    for i in post_start..nn_losses.len() {
        let lo = (i + 1).saturating_sub(cfg.error_ma);
        let m = nn_losses[lo..=i].iter().map(|(_, l)| *l).sum::<f64>() / (i + 1 - lo) as f64;
        ma.push((nn_losses[i].0, m));
    }
    let peak = ma
        .iter()
        .copied()
        .fold(None, |best: Option<(u64, f64)>, x| match best {
            Some(b) if b.1 >= x.1 => Some(b),
            _ => Some(x),
        });
    let (peak_tick, peak_loss) = peak.unwrap_or((cfg.change_tick, baseline_loss));
    let target = baseline_loss + cfg.rule.recovery_fraction_of_spike * (peak_loss - baseline_loss);
    let error_recovery_delay = if peak_loss > baseline_loss {
        ma.iter()
            .find(|(t, m)| *t > peak_tick && *m <= target)
            .map(|(t, _)| t - cfg.change_tick)
    } else {
        None
    };

    let pre: Vec<f64> = levels[pre_from as usize..cfg.change_tick as usize].to_vec();
    let pre_progress_level = summarize(&pre).mean;
    let drive_reappear_delay = levels
        .iter()
        .enumerate()
        .skip(cfg.change_tick as usize)
        .find(|(_, l)| **l >= cfg.rule.reappear_threshold)
        .map(|(t, _)| t as u64 - cfg.change_tick);

    let noise_share = noise_visits as f64 / (cfg.change_tick - cfg.warmup) as f64;
    RunMetrics {
        seed,
        noise_visits,
        noise_share,
        baseline_loss,
        peak_loss,
        error_recovery_delay,
        pre_progress_level,
        drive_reappear_delay,
        guard_counters,
        final_progress_level: *levels.last().unwrap_or(&0.0),
    }
}

// ---------------------------------------------------------------------------------------
// Study.
// ---------------------------------------------------------------------------------------

/// Declaration of a drive in the report: identity, signal and modality.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveDeclaration {
    /// Drive id.
    pub drive_id: String,
    /// Signal read.
    pub signal: DriveSignal,
    /// Modality (both signals are [`DriveModality::Situated`]).
    pub modality: DriveModality,
}

/// Results of one arm over all seeds.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmReport {
    /// Arm.
    pub arm: Arm,
    /// Per-seed metrics (same seeds in every arm).
    pub runs: Vec<RunMetrics>,
    /// Noise-cell visit share summary.
    pub share: Summary,
    /// Noise-cell visit count summary.
    pub visits: Summary,
    /// Error recovery delay (ticks) over seeds that recovered.
    pub error_delay: Summary,
    /// Drive reappearance delay (ticks) over seeds where it reappeared.
    pub drive_delay: Summary,
    /// Seeds in which the error recovered.
    pub error_recovered: usize,
    /// Seeds in which the drive reappeared.
    pub drive_reappeared: usize,
    /// Pre-change progress level summary.
    pub pre_level: Summary,
}

/// Full study report.
#[derive(Debug, Clone, PartialEq)]
pub struct StudyReport {
    /// Arms in order error, progress, random.
    pub arms: Vec<ArmReport>,
    /// Uniform share `1 / cells`.
    pub uniform_share: f64,
    /// Paired z of (error - progress) noise shares.
    pub z_error_vs_progress: f64,
    /// H1 verdict.
    pub trap_verdict: TrapVerdict,
    /// H2 verdict (evaluated on the progress-driven arm).
    pub recovery_verdict: RecoveryVerdict,
    /// Endowment ledger (drive weights as `Given`, observed level as `Learned`).
    pub ledger: EndowmentLedger,
    /// Drives recorded as situated.
    pub drives: Vec<DriveDeclaration>,
}

impl StudyReport {
    /// Arm lookup.
    pub fn arm(&self, arm: Arm) -> &ArmReport {
        self.arms
            .iter()
            .find(|a| a.arm == arm)
            .expect("all arms are present")
    }
}

fn arm_report(arm: Arm, runs: Vec<RunMetrics>) -> ArmReport {
    let col = |f: fn(&RunMetrics) -> f64| runs.iter().map(f).collect::<Vec<_>>();
    let opt = |f: fn(&RunMetrics) -> Option<u64>| {
        runs.iter()
            .filter_map(|r| f(r).map(|d| d as f64))
            .collect::<Vec<_>>()
    };
    let error_delay = opt(|r| r.error_recovery_delay);
    let drive_delay = opt(|r| r.drive_reappear_delay);
    ArmReport {
        arm,
        share: summarize(&col(|r| r.noise_share)),
        visits: summarize(&col(|r| r.noise_visits as f64)),
        error_recovered: error_delay.len(),
        drive_reappeared: drive_delay.len(),
        error_delay: summarize(&error_delay),
        drive_delay: summarize(&drive_delay),
        pre_level: summarize(&col(|r| r.pre_progress_level)),
        runs,
    }
}

/// Runs the paired study and applies the preregistered rule.
pub fn run_study(cfg: &StudyConfig) -> Result<StudyReport, LpDriveError> {
    let mut arms = Vec::new();
    for arm in [Arm::ErrorDriven, Arm::ProgressDriven, Arm::Random] {
        let runs = cfg
            .seeds
            .iter()
            .map(|&s| run_episode(cfg, arm, s))
            .collect::<Result<Vec<_>, _>>()?;
        arms.push(arm_report(arm, runs));
    }
    let uniform_share = 1.0 / (cfg.width * cfg.height) as f64;
    let shares = |a: &ArmReport| a.runs.iter().map(|r| r.noise_share).collect::<Vec<_>>();
    let (err, prog) = (&arms[0], &arms[1]);
    let z = paired_z(&shares(err), &shares(prog));
    let r = cfg.rule;
    let trap_verdict = if z >= r.min_paired_z
        && err.share.mean >= r.error_min_ratio_vs_uniform * uniform_share
        && prog.share.mean <= r.progress_max_ratio_vs_uniform * uniform_share
    {
        TrapVerdict::Confirmed
    } else if z <= -r.min_paired_z {
        TrapVerdict::Opposite
    } else {
        TrapVerdict::Refuted
    };
    let n = cfg.seeds.len() as f64;
    let recovery_verdict = if prog.drive_reappeared as f64 >= r.recovery_min_fraction * n
        && prog.error_recovered as f64 >= r.recovery_min_fraction * n
    {
        RecoveryVerdict::Confirmed
    } else {
        RecoveryVerdict::Refuted
    };

    let mut ledger = EndowmentLedger::default();
    ledger.record(
        "drive:learning_progress.deficit_gain",
        cfg.deficit_gain,
        EndowmentProvenance::Given,
    );
    ledger.record(
        "drive:learning_progress.window",
        cfg.window as f64,
        EndowmentProvenance::Given,
    );
    ledger.record(
        "drive:raw_error.deficit_gain",
        cfg.deficit_gain,
        EndowmentProvenance::Given,
    );
    ledger.record(
        "drive:learning_progress.final_level",
        summarize(
            &prog
                .runs
                .iter()
                .map(|r| r.final_progress_level)
                .collect::<Vec<_>>(),
        )
        .mean,
        EndowmentProvenance::Learned,
    );
    let drives = [DriveSignal::LearningProgress, DriveSignal::RawError]
        .into_iter()
        .map(|s| DriveDeclaration {
            drive_id: s.label().to_owned(),
            signal: s,
            modality: DriveModality::of_signal(s),
        })
        .collect();

    Ok(StudyReport {
        arms,
        uniform_share,
        z_error_vs_progress: z,
        trap_verdict,
        recovery_verdict,
        ledger,
        drives,
    })
}
