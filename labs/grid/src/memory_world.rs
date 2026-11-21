//! MemoryWorld (GOG-171 / DEM-018): bounded stigmergy laboratory on a Grid fixture.
//!
//! When an agent decides, it leaves in its own cell the *control measure it just estimated*
//! (`control_bits_from_pairs` over walkers' `(first action, next cell)` pairs, GOG-180) on an
//! [`AmbientSurface`] (GOG-170, no new `MemoryPort`). A reader uses the mark found at the final
//! cell of each imagined future as leaf value. Two agents share nothing but the surface; there is
//! no cooperative policy, no message code and no sender/receiver identity in the read path.
//!
//! # Preregistration (written before the first run of the decisive tests)
//!
//! Setting: 9x7 room, goal on the right, two agents; a "fog" band (slip probability 0.85) sits
//! across the middle rows (rules R1) and is moved to the detour rows after the rules change (R2).
//! Reader: random-shooting with a goal-biased continuation policy, leaf = goal bonus or
//! `lambda * mark` (mark = estimated control bits / 2, absent mark = 0).
//!
//! * **H1 (marks help a poor reader).** Tenth-budget reader + measurement marks has a higher
//!   success rate than the same reader without marks AND than the same reader with permuted marks.
//! * **H2 (gap).** Marks recover part of the gap to the full-budget reader without marks.
//! * **H3 (stale damage).** After R1 -> R2, a surface that never evaporates (stale) is worse than
//!   an evaporating surface; damage = `success(no marks) - success(marks)` in phase B.
//!
//! Decision rule: paired per-(episode, agent) success differences (common random numbers: same
//! start cell, same environment/planning seeds in every condition); mean +- 1.96 standard errors.
//! `Effect` iff the lower bound is `> 0`, `Harm` iff the upper bound is `< 0`, otherwise `Null`.
//! H1 needs `Effect` against both controls; otherwise H1 is reported as not supported. Nulls and
//! harms are reported as such. No claim of emergence, communication or "language" is made: the
//! mark is a scalar estimate written in the world, and the leaf value is a function of that scalar.
//!
//! # Controls
//!
//! All conditions perform the identical control estimation at every decision (same walkers, same
//! cost), so the *only* difference between no-marks / measurement / permuted is what the reader
//! can see. Grants, TTL and surface limits are identical; "permuted" reads the mark of a different
//! cell (a derangement of cell ids), keeping the multiset of values. Safety (walls) lives in the
//! environment and is never expressible through marks; writer identity is recorded as provenance
//! only and is not read when valuing a leaf.

use std::collections::BTreeSet;

use gogamen_core::{
    AmbientSurface, BinningVersion, ControlBitsConfig, DeterministicRng, EvidenceRef,
    MemoryViewError, OwnerId, RngSource, SurfaceAccess, SurfaceGrant, SurfaceId, SurfaceLimits,
    Tick, control_bits_from_pairs,
};

/// Grid width.
pub const MW_W: usize = 9;
/// Grid height.
pub const MW_H: usize = 7;
const N_CELLS: usize = MW_W * MW_H;
const GOAL: (usize, usize) = (7, 3);
const N_ACTIONS: usize = 4;
const SLIP: f64 = 0.85;
/// Maximum control bits with four actions (normalizer of marks).
const MAX_BITS: f64 = 2.0;
/// Honest scope note carried by every report.
pub const MEMORY_WORLD_NOTE: &str = "MemoryWorld: a scalar control estimate left in the world and \
read as leaf value. Not language, not communication protocol, not emergence.";

/// Which rules govern the fog band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rules {
    /// Fog across the middle rows (rows 2..=4), columns 3..=5.
    R1,
    /// After the change: fog across the detour rows (0,1,5,6), middle clear.
    R2,
}

impl Rules {
    fn fog(self, x: usize, y: usize) -> bool {
        if !(3..=5).contains(&x) {
            return false;
        }
        let middle = (2..=4).contains(&y);
        match self {
            Rules::R1 => middle,
            Rules::R2 => !middle,
        }
    }
}

/// Condition of the mark surface as seen by the reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkCondition {
    /// Marks never visible to the reader (estimation work is still done and paid).
    NoMarks,
    /// Reader sees the mark written in the cell.
    Measurement,
    /// Reader sees the mark of a different cell (derangement); same values/cost/TTL/access.
    Permuted,
}

/// Evaporation policy of the surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Evaporation {
    /// Cells expire after `ttl` ticks and visible value decays by `decay^age`.
    Evaporating {
        /// TTL in ticks.
        ttl: u64,
        /// Per-tick multiplicative decay in `(0, 1]`.
        decay: f64,
    },
    /// Marks never expire nor decay.
    Stale,
}

/// One experimental variant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Variant {
    /// Report label.
    pub label: &'static str,
    /// Rollouts per decision (cost knob).
    pub rollouts: usize,
    /// Mark visibility.
    pub marks: MarkCondition,
    /// Evaporation.
    pub evaporation: Evaporation,
}

/// Experiment configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MemoryWorldConfig {
    /// Exploratory warm-up episodes (R1, epsilon-random, marks written).
    pub warm_episodes: usize,
    /// Evaluation episodes per phase (each has two agents).
    pub eval_episodes: usize,
    /// Steps per episode.
    pub episode_len: usize,
    /// Imagination horizon.
    pub plan_h: usize,
    /// Full-budget rollouts per decision (tenth = this / 10).
    pub rollouts_full: usize,
    /// Walkers per control estimation.
    pub est_walkers: usize,
    /// Weight of the mark in the leaf value.
    pub lambda: f64,
    /// Discount for goal bonus.
    pub gamma: f64,
    /// Warm-up epsilon.
    pub warm_epsilon: f64,
    /// TTL (ticks) used by evaporating variants.
    pub ttl: u64,
    /// Decay used by evaporating variants.
    pub decay: f64,
    /// Master seed.
    pub seed: u64,
}

impl Default for MemoryWorldConfig {
    fn default() -> Self {
        Self {
            warm_episodes: 6,
            eval_episodes: 40,
            episode_len: 30,
            plan_h: 10,
            rollouts_full: 80,
            est_walkers: 64,
            lambda: 0.3,
            gamma: 0.95,
            warm_epsilon: 0.5,
            ttl: 90,
            decay: 0.98,
            seed: 171,
        }
    }
}

impl MemoryWorldConfig {
    /// Rollouts of the tenth-budget reader.
    pub fn rollouts_tenth(&self) -> usize {
        (self.rollouts_full / 10).max(N_ACTIONS)
    }
    fn variant(
        &self,
        label: &'static str,
        full: bool,
        marks: MarkCondition,
        stale: bool,
    ) -> Variant {
        Variant {
            label,
            rollouts: if full {
                self.rollouts_full
            } else {
                self.rollouts_tenth()
            },
            marks,
            evaporation: if stale {
                Evaporation::Stale
            } else {
                Evaporation::Evaporating {
                    ttl: self.ttl,
                    decay: self.decay,
                }
            },
        }
    }
    /// The five preregistered variants.
    pub fn variants(&self) -> [Variant; 5] {
        [
            self.variant("tenth/no-marks", false, MarkCondition::NoMarks, false),
            self.variant(
                "tenth/measurement+evap",
                false,
                MarkCondition::Measurement,
                false,
            ),
            self.variant("tenth/permuted+evap", false, MarkCondition::Permuted, false),
            self.variant("full/no-marks", true, MarkCondition::NoMarks, false),
            self.variant(
                "tenth/measurement+stale",
                false,
                MarkCondition::Measurement,
                true,
            ),
        ]
    }
}

/// Typed errors of the lab.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryWorldError {
    /// Surface refused an operation.
    Surface(MemoryViewError),
    /// Control estimation failed.
    Control(String),
    /// Invalid configuration.
    InvalidConfig(&'static str),
}

impl std::fmt::Display for MemoryWorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Surface(e) => write!(f, "surface: {e}"),
            Self::Control(e) => write!(f, "control estimation: {e}"),
            Self::InvalidConfig(r) => write!(f, "invalid config: {r}"),
        }
    }
}

impl std::error::Error for MemoryWorldError {}

impl From<MemoryViewError> for MemoryWorldError {
    fn from(e: MemoryViewError) -> Self {
        Self::Surface(e)
    }
}

fn cell_id(x: usize, y: usize) -> usize {
    y * MW_W + x
}

fn is_wall(x: i64, y: i64) -> bool {
    x < 0 || y < 0 || x >= MW_W as i64 || y >= MW_H as i64
}

/// Environment step. Walls (borders) are a safety rule of the world: moving into one is a no-op,
/// regardless of any mark.
fn env_step(
    rules: Rules,
    pos: (usize, usize),
    action: usize,
    rng: &mut DeterministicRng,
) -> (usize, usize) {
    let a = if rules.fog(pos.0, pos.1) && rng.next_f64() < SLIP {
        rng.next_usize_unbiased(N_ACTIONS)
    } else {
        action
    };
    let (dx, dy) = [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)][a];
    let (nx, ny) = (pos.0 as i64 + dx, pos.1 as i64 + dy);
    if is_wall(nx, ny) {
        pos
    } else {
        (nx as usize, ny as usize)
    }
}

/// Goal-biased continuation policy of imagined futures.
fn rollout_action(pos: (usize, usize), rng: &mut DeterministicRng) -> usize {
    if rng.next_f64() < 0.6 {
        let (dx, dy) = (GOAL.0 as i64 - pos.0 as i64, GOAL.1 as i64 - pos.1 as i64);
        if dx.abs() >= dy.abs() && dx != 0 {
            return if dx > 0 { 0 } else { 1 };
        }
        if dy != 0 {
            return if dy > 0 { 2 } else { 3 };
        }
        if dx != 0 {
            return if dx > 0 { 0 } else { 1 };
        }
    }
    rng.next_usize_unbiased(N_ACTIONS)
}

fn mix(parts: &[u64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        h ^= *p;
        h = h.wrapping_mul(0x1000_0000_01b3);
        h ^= h >> 29;
    }
    h
}

/// Mark surface wrapper: [`AmbientSurface`] + visibility condition + evaporation.
#[derive(Debug, Clone)]
pub struct MarkBoard {
    surface: AmbientSurface,
    condition: MarkCondition,
    evaporation: Evaporation,
    perm: Vec<usize>,
    writes: u64,
}

impl MarkBoard {
    /// Builds a board where every id in `agents` may read and write every cell.
    pub fn new(
        agents: &[OwnerId],
        condition: MarkCondition,
        evaporation: Evaporation,
        seed: u64,
    ) -> Result<Self, MemoryWorldError> {
        let mut grants = Vec::new();
        for &agent in agents {
            for c in 0..N_CELLS {
                grants.push(SurfaceGrant {
                    agent,
                    surface: SurfaceId::new(c as u64),
                    access: SurfaceAccess::Write,
                });
            }
        }
        let ttl_ticks = match evaporation {
            Evaporation::Evaporating { ttl, .. } => Some(ttl),
            Evaporation::Stale => None,
        };
        let surface = AmbientSurface::new(
            grants,
            SurfaceLimits {
                max_cells: N_CELLS,
                max_writes: 1_000_000,
                ttl_ticks,
            },
        )?;
        // Derangement: shuffle an order, map each cell to the next one in that order.
        let mut order: Vec<usize> = (0..N_CELLS).collect();
        let mut rng = DeterministicRng::new(mix(&[seed, 0x9e37]));
        for i in (1..N_CELLS).rev() {
            order.swap(i, rng.next_usize_unbiased(i + 1));
        }
        let mut perm = vec![0; N_CELLS];
        for i in 0..N_CELLS {
            perm[order[i]] = order[(i + 1) % N_CELLS];
        }
        Ok(Self {
            surface,
            condition,
            evaporation,
            perm,
            writes: 0,
        })
    }

    /// Leaves `bits` (control estimate) in `cell`. Written in every condition (equal cost).
    pub fn write(
        &mut self,
        agent: OwnerId,
        cell: usize,
        bits: f64,
        tick: u64,
    ) -> Result<(), MemoryWorldError> {
        self.surface.write(
            agent,
            SurfaceId::new(cell as u64),
            bits,
            Tick::new(tick),
            EvidenceRef::new(tick),
        )?;
        self.writes += 1;
        Ok(())
    }

    /// Applies TTL evaporation.
    pub fn expire(&mut self, now: u64) -> usize {
        self.surface.expire(Tick::new(now))
    }

    /// Normalized mark in `[0, 1]` visible to `reader` at `cell` (0 when absent or hidden).
    /// The writer's identity is deliberately not consulted.
    pub fn read(&self, reader: OwnerId, cell: usize, now: u64) -> Result<f64, MemoryWorldError> {
        let target = match self.condition {
            MarkCondition::NoMarks => return Ok(0.0),
            MarkCondition::Measurement => cell,
            MarkCondition::Permuted => self.perm[cell],
        };
        let Some(c) = self.surface.read(reader, SurfaceId::new(target as u64))? else {
            return Ok(0.0);
        };
        let age = now.saturating_sub(c.provenance.tick.as_u64());
        let factor = match self.evaporation {
            Evaporation::Evaporating { decay, .. } => decay.powi(age.min(10_000) as i32),
            Evaporation::Stale => 1.0,
        };
        Ok((c.value * factor / MAX_BITS).clamp(0.0, 1.0))
    }

    /// Writes performed.
    pub fn writes(&self) -> u64 {
        self.writes
    }

    /// Raw (undecayed) value stored for `cell`, for diagnostics.
    pub fn raw(&self, reader: OwnerId, cell: usize) -> Result<Option<f64>, MemoryWorldError> {
        Ok(self
            .surface
            .read(reader, SurfaceId::new(cell as u64))?
            .map(|c| c.value))
    }

    /// Underlying surface (read-only), e.g. for replay checks.
    pub fn surface(&self) -> &AmbientSurface {
        &self.surface
    }

    /// Multiset of raw values currently stored, sorted (permutation invariance checks).
    pub fn sorted_values(&self, reader: OwnerId) -> Result<Vec<f64>, MemoryWorldError> {
        let mut v = Vec::new();
        for c in 0..N_CELLS {
            if let Some(x) = self.raw(reader, c)? {
                v.push(x);
            }
        }
        v.sort_by(f64::total_cmp);
        Ok(v)
    }
}

/// Simulation-step accounting (the cost unit).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepCost {
    /// Steps spent estimating control (walkers).
    pub estimation: u64,
    /// Steps spent in imagined rollouts.
    pub planning: u64,
}

/// Outcome of one agent in one episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpisodeOutcome {
    /// Goal reached within the episode.
    pub success: bool,
    /// Steps used (episode length if failed).
    pub steps: usize,
}

/// Estimates control bits at `pos` with `n` walkers (uniform first action, one real step).
fn estimate_control(
    rules: Rules,
    pos: (usize, usize),
    n: usize,
    cfg: &ControlBitsConfig,
    rng: &mut DeterministicRng,
) -> Result<f64, MemoryWorldError> {
    let pairs: Vec<(usize, usize)> = (0..n)
        .map(|_| {
            let a = rng.next_usize_unbiased(N_ACTIONS);
            let q = env_step(rules, pos, a, rng);
            (a, cell_id(q.0, q.1))
        })
        .collect();
    let e = control_bits_from_pairs(&pairs, "mw_cell", BinningVersion::V1_0, cfg)
        .map_err(|e| MemoryWorldError::Control(format!("{e:?}")))?;
    Ok(e.control_corrected.clamp(0.0, MAX_BITS))
}

/// Reader decision: random shooting with leaf = goal bonus or `lambda * mark(final cell)`.
#[allow(clippy::too_many_arguments)]
fn decide(
    cfg: &MemoryWorldConfig,
    variant: &Variant,
    board: &MarkBoard,
    reader: OwnerId,
    rules: Rules,
    pos: (usize, usize),
    now: u64,
    rng: &mut DeterministicRng,
    cost: &mut StepCost,
) -> Result<usize, MemoryWorldError> {
    let per_action = (variant.rollouts / N_ACTIONS).max(1);
    let mut best = (f64::NEG_INFINITY, 0usize);
    let tie = rng.next_usize_unbiased(N_ACTIONS);
    for k in 0..N_ACTIONS {
        let a = (k + tie) % N_ACTIONS;
        let mut total = 0.0;
        for _ in 0..per_action {
            let mut p = env_step(rules, pos, a, rng);
            cost.planning += 1;
            let mut value = None;
            if p == GOAL {
                value = Some(1.0);
            }
            let mut t = 1;
            while value.is_none() && t < cfg.plan_h {
                let act = rollout_action(p, rng);
                p = env_step(rules, p, act, rng);
                cost.planning += 1;
                t += 1;
                if p == GOAL {
                    value = Some(cfg.gamma.powi(t as i32 - 1));
                }
            }
            total += match value {
                Some(v) => v,
                None => cfg.lambda * board.read(reader, cell_id(p.0, p.1), now)?,
            };
        }
        let mean = total / per_action as f64;
        if mean > best.0 {
            best = (mean, a);
        }
    }
    Ok(best.1)
}

/// Result of one variant over warm-up, phase A (R1) and phase B (R2).
#[derive(Debug, Clone, PartialEq)]
pub struct VariantRun {
    /// Variant.
    pub variant: Variant,
    /// Phase A outcomes (R1), indexed `episode * 2 + agent`.
    pub phase_a: Vec<EpisodeOutcome>,
    /// Phase B outcomes (R2 after rules change), same indexing.
    pub phase_b: Vec<EpisodeOutcome>,
    /// Cost during the whole run.
    pub cost: StepCost,
    /// Surface writes performed.
    pub writes: u64,
    /// Hash of the final surface state (determinism checks).
    pub surface_hash: u64,
}

impl VariantRun {
    /// Success rate of a phase.
    pub fn success_rate(outcomes: &[EpisodeOutcome]) -> f64 {
        outcomes.iter().filter(|o| o.success).count() as f64 / outcomes.len().max(1) as f64
    }
    /// Mean steps of a phase.
    pub fn mean_steps(outcomes: &[EpisodeOutcome]) -> f64 {
        outcomes.iter().map(|o| o.steps as f64).sum::<f64>() / outcomes.len().max(1) as f64
    }
}

fn start_cell(seed: u64, phase: u64, ep: usize, agent: usize) -> (usize, usize) {
    let mut rng = DeterministicRng::new(mix(&[seed, 0x57a7, phase, ep as u64, agent as u64]));
    (rng.next_usize_unbiased(2), rng.next_usize_unbiased(MW_H))
}

/// Runs one variant: warm-up in R1, evaluation in R1 (phase A), rules change, evaluation in R2
/// (phase B) with the same surface (marks written before the change remain).
pub fn run_variant(
    cfg: &MemoryWorldConfig,
    variant: &Variant,
) -> Result<VariantRun, MemoryWorldError> {
    if cfg.eval_episodes == 0 || cfg.episode_len == 0 || cfg.est_walkers < 2 {
        return Err(MemoryWorldError::InvalidConfig(
            "episodes, length and walkers must be positive",
        ));
    }
    let agents = [OwnerId::new(1), OwnerId::new(2)];
    let mut board = MarkBoard::new(&agents, variant.marks, variant.evaporation, cfg.seed)?;
    let cb_cfg = ControlBitsConfig {
        resamples: 8,
        confidence: 0.95,
        seed: cfg.seed,
    };
    let mut cost = StepCost::default();
    let mut tick = 0u64;
    let mut phase_a = Vec::new();
    let mut phase_b = Vec::new();

    let phases: [(u64, Rules, usize, bool); 3] = [
        (0, Rules::R1, cfg.warm_episodes, true),
        (1, Rules::R1, cfg.eval_episodes, false),
        (2, Rules::R2, cfg.eval_episodes, false),
    ];
    for (phase, rules, episodes, warm) in phases {
        for ep in 0..episodes {
            let mut pos = [
                start_cell(cfg.seed, phase, ep, 0),
                start_cell(cfg.seed, phase, ep, 1),
            ];
            let mut done = [None::<usize>; 2];
            for step in 0..cfg.episode_len {
                tick += 1;
                board.expire(tick);
                for (ai, &agent) in agents.iter().enumerate() {
                    if done[ai].is_some() {
                        continue;
                    }
                    let key = [cfg.seed, phase, ep as u64, ai as u64, step as u64];
                    let mut plan_rng =
                        DeterministicRng::new(mix(&[key[0], key[1], key[2], key[3], key[4], 1]));
                    let mut env_rng =
                        DeterministicRng::new(mix(&[key[0], key[1], key[2], key[3], key[4], 2]));
                    let mut est_rng =
                        DeterministicRng::new(mix(&[key[0], key[1], key[2], key[3], key[4], 3]));
                    // Estimate control here and leave it in this cell (all conditions pay this).
                    let bits =
                        estimate_control(rules, pos[ai], cfg.est_walkers, &cb_cfg, &mut est_rng)?;
                    cost.estimation += cfg.est_walkers as u64;
                    board.write(agent, cell_id(pos[ai].0, pos[ai].1), bits, tick)?;
                    let action = if warm && plan_rng.next_f64() < cfg.warm_epsilon {
                        plan_rng.next_usize_unbiased(N_ACTIONS)
                    } else {
                        decide(
                            cfg,
                            variant,
                            &board,
                            agent,
                            rules,
                            pos[ai],
                            tick,
                            &mut plan_rng,
                            &mut cost,
                        )?
                    };
                    pos[ai] = env_step(rules, pos[ai], action, &mut env_rng);
                    if pos[ai] == GOAL {
                        done[ai] = Some(step + 1);
                    }
                }
            }
            if !warm {
                let out = if phase == 1 {
                    &mut phase_a
                } else {
                    &mut phase_b
                };
                for d in done {
                    out.push(EpisodeOutcome {
                        success: d.is_some(),
                        steps: d.unwrap_or(cfg.episode_len),
                    });
                }
            }
        }
    }
    Ok(VariantRun {
        variant: *variant,
        phase_a,
        phase_b,
        cost,
        writes: board.writes(),
        surface_hash: board.surface().state_hash(),
    })
}

/// Verdict of the preregistered decision rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// 95% interval of the paired difference is above zero.
    Effect,
    /// Interval contains zero: null result.
    Null,
    /// Interval below zero: the first condition is worse.
    Harm,
}

/// Paired comparison `a - b` of success indicators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairedDiff {
    /// Mean difference in success rate (a minus b).
    pub mean: f64,
    /// Lower 95% bound (normal approximation).
    pub ci_low: f64,
    /// Upper 95% bound.
    pub ci_high: f64,
    /// Preregistered verdict.
    pub verdict: Verdict,
}

/// Paired difference of success between two runs' outcome lists (same indexing).
pub fn paired_diff(a: &[EpisodeOutcome], b: &[EpisodeOutcome]) -> PairedDiff {
    let n = a.len().min(b.len()).max(1) as f64;
    let d: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(u8::from(x.success)) - f64::from(u8::from(y.success)))
        .collect();
    let mean = d.iter().sum::<f64>() / n;
    let var = d.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let se = (var / n).sqrt();
    let (ci_low, ci_high) = (mean - 1.96 * se, mean + 1.96 * se);
    let verdict = if ci_low > 0.0 {
        Verdict::Effect
    } else if ci_high < 0.0 {
        Verdict::Harm
    } else {
        Verdict::Null
    };
    PairedDiff {
        mean,
        ci_low,
        ci_high,
        verdict,
    }
}

/// Full report of the preregistered experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryWorldReport {
    /// Runs in [`MemoryWorldConfig::variants`] order.
    pub runs: Vec<VariantRun>,
    /// H1a: tenth+marks minus tenth no-marks (phase A).
    pub marks_vs_none: PairedDiff,
    /// H1b: tenth+marks minus tenth permuted marks (phase A).
    pub marks_vs_permuted: PairedDiff,
    /// H2: full no-marks minus tenth+marks (phase A); `Effect` means a gap remains.
    pub full_vs_marks: PairedDiff,
    /// Fraction of the full-vs-tenth-no-marks gap recovered by marks (None if no gap).
    pub gap_recovered: Option<f64>,
    /// H1 supported (both controls `Effect`).
    pub h1_supported: bool,
    /// Phase B: no-marks minus evaporating marks (damage of marks after change; positive = harm).
    pub damage_evaporating: PairedDiff,
    /// Phase B: no-marks minus stale marks.
    pub damage_stale: PairedDiff,
    /// Phase B: evaporating minus stale marks (H3 supported iff `Effect`).
    pub evap_vs_stale: PairedDiff,
    /// Scope note.
    pub note: &'static str,
}

/// Runs the five variants and applies the preregistered decision rule.
pub fn run_memory_world(cfg: &MemoryWorldConfig) -> Result<MemoryWorldReport, MemoryWorldError> {
    let vs = cfg.variants();
    let runs = vs
        .iter()
        .map(|v| run_variant(cfg, v))
        .collect::<Result<Vec<_>, _>>()?;
    let (none, meas, perm, full, stale) = (&runs[0], &runs[1], &runs[2], &runs[3], &runs[4]);
    let marks_vs_none = paired_diff(&meas.phase_a, &none.phase_a);
    let marks_vs_permuted = paired_diff(&meas.phase_a, &perm.phase_a);
    let full_vs_marks = paired_diff(&full.phase_a, &meas.phase_a);
    let gap = VariantRun::success_rate(&full.phase_a) - VariantRun::success_rate(&none.phase_a);
    let gap_recovered = (gap.abs() > 1e-9).then(|| {
        (VariantRun::success_rate(&meas.phase_a) - VariantRun::success_rate(&none.phase_a)) / gap
    });
    let h1_supported =
        marks_vs_none.verdict == Verdict::Effect && marks_vs_permuted.verdict == Verdict::Effect;
    Ok(MemoryWorldReport {
        marks_vs_none,
        marks_vs_permuted,
        full_vs_marks,
        gap_recovered,
        h1_supported,
        damage_evaporating: paired_diff(&none.phase_b, &meas.phase_b),
        damage_stale: paired_diff(&none.phase_b, &stale.phase_b),
        evap_vs_stale: paired_diff(&meas.phase_b, &stale.phase_b),
        runs,
        note: MEMORY_WORLD_NOTE,
    })
}

/// Cells where marks were written (diagnostic).
pub fn written_cells(
    board: &MarkBoard,
    reader: OwnerId,
) -> Result<BTreeSet<usize>, MemoryWorldError> {
    let mut s = BTreeSet::new();
    for c in 0..N_CELLS {
        if board.raw(reader, c)?.is_some() {
            s.insert(c);
        }
    }
    Ok(s)
}
