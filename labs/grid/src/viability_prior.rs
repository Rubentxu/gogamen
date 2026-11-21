//! GOG-186: sweep of the failure-cube pseudocount (the "viability prior") of the learned model.
//!
//! The prior is a design number ([`LearnedModelConfig::failure_pseudocount`]): it sets how much
//! failure mass the model assigns to `(cell, action)` pairs it has never observed. The reward
//! (`GridTask`) is held fixed across the whole sweep; only the prior changes.
//!
//! Hypothesis H (preregistered, see [`PreregisteredRule`], written before any result was seen):
//! a higher prior makes the agent avoid the unknown, so across the interval
//! - mean deaths are non-increasing in the prior, and
//! - mean explored terrain (distinct cells visited) is non-increasing in the prior.
//!
//! Nothing here claims agency or emergence: it is a one-parameter sensitivity study of a
//! planner on a small grid. A [`Verdict::Refuted`] outcome is a valid, reported result.

use gogamen_core::{
    Actuator, EndowmentProvenance, LearnedModelConfig, LearningPolicy, MasterSeed, Tick,
    UnseenPairPolicy,
};

use crate::hidden_dynamics::{
    GridTask, HiddenGridError, PlanGrid, PlanGridConfig, collect_hidden_experience,
    freeze_grid_model, grid_cell_scale,
};
use crate::world::{GridWorldConfig, GridWorldEnv};
use crate::{GridMap, GridPos};

/// One entry of the endowment ledger ("libro de dotacion"): a design number and its origin.
///
/// No pre-existing ledger record type exists in the repo (only the
/// [`EndowmentProvenance`] marker), so this minimal typed record lives next to the lab code.
/// The provenance of the prior is [`EndowmentProvenance::Given`] (by design).
#[derive(Debug, Clone, PartialEq)]
pub struct EndowmentEntry {
    /// Name of the endowed quantity.
    pub name: &'static str,
    /// Value handed to the agent.
    pub value: f64,
    /// Origin of the value.
    pub source: EndowmentProvenance,
}

/// Ledger of endowed quantities.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EndowmentLedger {
    /// Entries in insertion order.
    pub entries: Vec<EndowmentEntry>,
}

impl EndowmentLedger {
    /// Records an arbitrary endowed quantity with its provenance (GOG-187).
    pub fn record(&mut self, name: &'static str, value: f64, source: EndowmentProvenance) {
        self.entries.push(EndowmentEntry {
            name,
            value,
            source,
        });
    }

    /// Records the failure pseudocount as given by design.
    pub fn record_failure_prior(&mut self, value: f64) {
        self.entries.push(EndowmentEntry {
            name: "failure_pseudocount",
            value,
            source: EndowmentProvenance::Given,
        });
    }
}

/// Preregistered decision rule (fixed before running the sweep).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PreregisteredRule {
    /// Minimum `-rho` (Spearman between prior rank and per-prior mean) for each metric.
    pub min_neg_spearman: f64,
    /// Minimum drop (first prior minus last prior) in units of the paired standard error.
    pub min_endpoint_z: f64,
}

impl PreregisteredRule {
    /// The rule: Spearman rho <= -0.7 AND endpoint paired drop >= 2 standard errors, for BOTH
    /// deaths and explored terrain. Otherwise the hypothesis is REFUTED (no partial credit).
    pub const PREREGISTERED: Self = Self {
        min_neg_spearman: 0.7,
        min_endpoint_z: 2.0,
    };
}

/// Verdict of the monotonicity check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Both metrics decrease with the prior per the rule.
    Confirmed,
    /// At least one metric fails the rule.
    Refuted,
}

/// Sweep configuration. Same seeds for every prior value (paired design).
#[derive(Debug, Clone)]
pub struct PriorSweepConfig {
    /// World (reward fixed).
    pub world: GridWorldConfig,
    /// Prior values, ascending.
    pub priors: Vec<f64>,
    /// Paired seeds (experience collection + planner seed).
    pub seeds: Vec<u64>,
    /// Random-policy observations per learner.
    pub observations: usize,
    /// Episode tick cap.
    pub max_ticks: u64,
    /// Rule applied to the result.
    pub rule: PreregisteredRule,
}

/// Hazard map: start top-left, goal bottom-right, hazards scattered in the middle.
pub fn hazard_map() -> GridMap {
    GridMap::from_ascii(&[
        "#########",
        "#S......#",
        "#.H.H.H.#",
        "#.......#",
        "#.H.H.H.#",
        "#......G#",
        "#########",
    ])
    .expect("static map is valid")
}

impl PriorSweepConfig {
    /// Default small sweep.
    pub fn new() -> Self {
        Self {
            world: GridWorldConfig {
                map: hazard_map(),
                ..GridWorldConfig::default()
            },
            priors: vec![0.01, 0.05, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0],
            seeds: (1..=24).collect(),
            observations: 120,
            max_ticks: 40,
            rule: PreregisteredRule::PREREGISTERED,
        }
    }
}

impl Default for PriorSweepConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of one (prior, seed) episode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedOutcome {
    /// Seed.
    pub seed: u64,
    /// Died in a hazard.
    pub died: bool,
    /// Distinct cells visited (including start).
    pub explored: usize,
}

/// Dispersion summary of a sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dispersion {
    /// Mean.
    pub mean: f64,
    /// Sample standard deviation (n-1).
    pub std: f64,
    /// Minimum.
    pub min: f64,
    /// Maximum.
    pub max: f64,
}

fn dispersion(xs: &[f64]) -> Dispersion {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = if xs.len() > 1 {
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    Dispersion {
        mean,
        std: var.sqrt(),
        min: xs.iter().cloned().fold(f64::INFINITY, f64::min),
        max: xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Row of the sweep for one prior value.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorRow {
    /// Prior value.
    pub prior: f64,
    /// Per-seed outcomes (same seeds in every row).
    pub outcomes: Vec<SeedOutcome>,
    /// Deaths per episode (0/1 per seed).
    pub deaths: Dispersion,
    /// Explored distinct cells.
    pub explored: Dispersion,
}

/// Full sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorSweepReport {
    /// Rows by prior.
    pub rows: Vec<PriorRow>,
    /// Ledger with every prior value as given by design.
    pub ledger: EndowmentLedger,
    /// Spearman (prior, mean deaths).
    pub rho_deaths: f64,
    /// Spearman (prior, mean explored).
    pub rho_explored: f64,
    /// Endpoint paired drop z for deaths (first - last).
    pub z_deaths: f64,
    /// Endpoint paired drop z for explored.
    pub z_explored: f64,
    /// Verdict under the preregistered rule.
    pub verdict: Verdict,
}

fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut r = vec![0.0; xs.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && xs[idx[j + 1]] == xs[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation; `0.0` when either series is constant (no monotone evidence).
pub fn spearman(xs: &[f64], ys: &[f64]) -> f64 {
    let (rx, ry) = (ranks(xs), ranks(ys));
    let n = rx.len() as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
    let cov: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let vx: f64 = rx.iter().map(|a| (a - mx).powi(2)).sum();
    let vy: f64 = ry.iter().map(|b| (b - my).powi(2)).sum();
    if vx == 0.0 || vy == 0.0 {
        0.0
    } else {
        cov / (vx * vy).sqrt()
    }
}

fn paired_drop_z(first: &[f64], last: &[f64]) -> f64 {
    let d: Vec<f64> = first.iter().zip(last).map(|(a, b)| a - b).collect();
    let s = dispersion(&d);
    let se = s.std / (d.len() as f64).sqrt();
    if se == 0.0 {
        if s.mean > 0.0 { f64::INFINITY } else { 0.0 }
    } else {
        s.mean / se
    }
}

fn run_one(
    config: &PriorSweepConfig,
    prior: f64,
    seed: u64,
) -> Result<SeedOutcome, HiddenGridError> {
    let world = &config.world;
    let scale = grid_cell_scale(&world.map)?;
    let learner = collect_hidden_experience(
        world,
        &scale,
        config.observations,
        MasterSeed::new(seed),
        LearningPolicy::new(0.05)?,
    )?;
    let model = freeze_grid_model(
        &learner,
        scale,
        LearnedModelConfig::new(prior, 2, UnseenPairPolicy::SamplePrior)?,
    )?;
    let mut plan = PlanGridConfig::new(GridTask::from_world(world));
    plan.max_uncertain_fraction = 1.0; // never abstain: the prior alone drives behaviour
    plan.seed = MasterSeed::new(seed.wrapping_mul(0x9E37_79B9).wrapping_add(11));
    let agent = PlanGrid::new(model, plan)?;

    let mut env = GridWorldEnv::new(world.clone())?;
    let mut visited: Vec<GridPos> = vec![env.position()];
    let mut ticks = 0;
    while ticks < config.max_ticks && !env.terminal().is_terminal() {
        let d = agent.decide(env.position(), Tick::new(ticks), None);
        Actuator::apply(&mut env, &d.action).map_err(|_| {
            HiddenGridError::Domain(gogamen_core::DomainError::InvalidTransition {
                reason: "action rejected in viability prior sweep",
            })
        })?;
        let p = env.position();
        if !visited.contains(&p) {
            visited.push(p);
        }
        ticks += 1;
    }
    Ok(SeedOutcome {
        seed,
        died: env.terminal()
            == gogamen_core::Terminal::Terminated(gogamen_core::TerminalReason::Failure),
        explored: visited.len(),
    })
}

/// Runs the paired sweep and applies the preregistered rule.
pub fn run_prior_sweep(config: &PriorSweepConfig) -> Result<PriorSweepReport, HiddenGridError> {
    let mut rows = Vec::new();
    let mut ledger = EndowmentLedger::default();
    for &prior in &config.priors {
        ledger.record_failure_prior(prior);
        let outcomes = config
            .seeds
            .iter()
            .map(|&s| run_one(config, prior, s))
            .collect::<Result<Vec<_>, _>>()?;
        let d: Vec<f64> = outcomes
            .iter()
            .map(|o| f64::from(u8::from(o.died)))
            .collect();
        let e: Vec<f64> = outcomes.iter().map(|o| o.explored as f64).collect();
        rows.push(PriorRow {
            prior,
            outcomes,
            deaths: dispersion(&d),
            explored: dispersion(&e),
        });
    }
    let priors: Vec<f64> = rows.iter().map(|r| r.prior).collect();
    let md: Vec<f64> = rows.iter().map(|r| r.deaths.mean).collect();
    let me: Vec<f64> = rows.iter().map(|r| r.explored.mean).collect();
    let rho_deaths = spearman(&priors, &md);
    let rho_explored = spearman(&priors, &me);
    let series = |r: &PriorRow, f: fn(&SeedOutcome) -> f64| -> Vec<f64> {
        r.outcomes.iter().map(f).collect()
    };
    let (first, last) = (&rows[0], &rows[rows.len() - 1]);
    let dth = |o: &SeedOutcome| f64::from(u8::from(o.died));
    let exp = |o: &SeedOutcome| o.explored as f64;
    let z_deaths = paired_drop_z(&series(first, dth), &series(last, dth));
    let z_explored = paired_drop_z(&series(first, exp), &series(last, exp));
    let r = config.rule;
    let ok = rho_deaths <= -r.min_neg_spearman
        && rho_explored <= -r.min_neg_spearman
        && z_deaths >= r.min_endpoint_z
        && z_explored >= r.min_endpoint_z;
    Ok(PriorSweepReport {
        rows,
        ledger,
        rho_deaths,
        rho_explored,
        z_deaths,
        z_explored,
        verdict: if ok {
            Verdict::Confirmed
        } else {
            Verdict::Refuted
        },
    })
}
