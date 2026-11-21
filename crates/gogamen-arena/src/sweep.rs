//! Parameter sweeps over the arena and phase diagram of the behaviour (`GOG-185`).
//!
//! A sweep is the Cartesian product of four axes (horizon, scale, budget, endowment) evaluated
//! with the same paired `Dev`/`Holdout` seeds in every cell. Outputs are a CSV (denominators,
//! failures, dispersion), an SVG heat map and a frontier report: the smallest horizon at which the
//! phenomenon appears, as an interval across seeds, next to the exact oracle of a pocket world
//! ([`gogamen_lab_grid::bool_world`]).
//!
//! The executor is generic over [`SweepWorld`]; [`CartPoleEntropicSweep`] is the first adapter.
//! The total cost is declared before anything runs and enforced by
//! [`SweepError::BudgetExceeded`].

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use gogamen_core::{BinSpec, BinningVersion, MasterSeed};
use gogamen_lab_cartpole::{
    DEFAULT_THETA_THRESHOLD_RADIANS, DEFAULT_X_THRESHOLD, EntropicConditionKind,
    EntropicExperimentConfig, run_entropic_comparison,
};
use gogamen_lab_grid::bool_world::{BoolSamplingPolicy, BoolWorld, MicroState, OracleLimits};

use crate::runner::{ArenaError, ArenaSeedSet};
use crate::stats::DispersionStats;
use gogamen_lab_grid::SeedSplit;

/// Named candidate-action set (the "endowment" axis).
#[derive(Debug, Clone, PartialEq)]
pub struct Endowment {
    /// Short label used in CSV rows.
    pub label: String,
    /// Candidate forces available to the controller.
    pub actions: Vec<f64>,
}

/// Axes of a sweep; every combination is one cell.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepAxes {
    /// Planning horizons.
    pub horizons: Vec<usize>,
    /// Scales: uniform bins per axis of the `BinSpec`.
    pub scales: Vec<usize>,
    /// Budgets: rollouts per step and candidate action.
    pub budgets: Vec<usize>,
    /// Endowments: candidate action sets.
    pub endowments: Vec<Endowment>,
}

/// One point of the Cartesian product.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepCell {
    /// Horizon.
    pub horizon: usize,
    /// Bins per axis.
    pub scale: usize,
    /// Rollouts per step.
    pub budget: usize,
    /// Index into [`SweepAxes::endowments`].
    pub endowment_index: usize,
}

/// Measure of one seed in one cell.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedMeasure {
    /// Fraction of the episode spent upright by the treated controller, in `[0, 1]`.
    pub treated_score: f64,
    /// Same for the baseline controller (paired on the seed).
    pub baseline_score: f64,
    /// Whether the treated episode ended in failure.
    pub treated_failed: bool,
}

/// World evaluated by a sweep.
pub trait SweepWorld {
    /// Declared cost of one seed in one cell (in abstract model steps).
    fn seed_cost(&self, cell: &SweepCell, endowment: &Endowment) -> u64;
    /// Runs one seed in one cell; `Err` is counted as a failed run, never dropped.
    fn run_seed(
        &self,
        cell: &SweepCell,
        endowment: &Endowment,
        seed: MasterSeed,
    ) -> Result<SeedMeasure, String>;
}

/// Sweep configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepConfig {
    /// Identifier echoed in the report.
    pub sweep_id: String,
    /// Axes of the product.
    pub axes: SweepAxes,
    /// Paired seeds, shared by every cell.
    pub seeds: ArenaSeedSet,
    /// Maximum total declared cost.
    pub budget_limit: u64,
    /// The phenomenon is present for a seed when `treated - baseline > margin`.
    pub margin: f64,
}

/// Sweep errors.
#[derive(Debug, Clone, PartialEq)]
pub enum SweepError {
    /// Declared total cost exceeds the limit; nothing was run.
    BudgetExceeded {
        /// Declared total cost.
        declared: u64,
        /// Configured limit.
        limit: u64,
    },
    /// An axis has no values.
    EmptyAxis(&'static str),
    /// Invalid seed set.
    Seeds(ArenaError),
}

impl fmt::Display for SweepError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BudgetExceeded { declared, limit } => {
                write!(
                    f,
                    "sweep budget exceeded: declared {declared} > limit {limit}"
                )
            }
            Self::EmptyAxis(name) => write!(f, "sweep axis `{name}` is empty"),
            Self::Seeds(e) => write!(f, "invalid sweep seeds: {e}"),
        }
    }
}

impl std::error::Error for SweepError {}

/// Aggregate of one cell over one split.
#[derive(Debug, Clone, PartialEq)]
pub struct CellResult {
    /// Cell coordinates.
    pub cell: SweepCell,
    /// Split.
    pub split: SeedSplit,
    /// Denominator: seeds attempted.
    pub total: usize,
    /// Seeds whose run errored (kept in the denominator).
    pub run_errors: usize,
    /// Seeds whose treated episode failed.
    pub treated_failures: usize,
    /// Seeds where the phenomenon is present.
    pub present: usize,
    /// Dispersion of `treated - baseline` over the seeds that ran.
    pub advantage: DispersionStats,
    /// Per-seed `(seed, present)` in input order.
    pub per_seed: Vec<(u64, bool)>,
}

/// Interval of the horizon where the phenomenon first appears, across seeds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontierInterval {
    /// Smallest first-appearance horizon over seeds, if any seed showed it.
    pub min: Option<usize>,
    /// Largest first-appearance horizon over seeds, if any seed showed it.
    pub max: Option<usize>,
    /// Seeds that never showed the phenomenon on the swept horizons.
    pub never: usize,
    /// Seeds considered.
    pub total: usize,
}

impl FrontierInterval {
    fn from_firsts(firsts: &[Option<usize>]) -> Self {
        let vals: Vec<usize> = firsts.iter().flatten().copied().collect();
        Self {
            min: vals.iter().min().copied(),
            max: vals.iter().max().copied(),
            never: firsts.len() - vals.len(),
            total: firsts.len(),
        }
    }

    /// Whether both intervals exist and overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        match (self.min, self.max, other.min, other.max) {
            (Some(a0), Some(a1), Some(b0), Some(b1)) => a0 <= b1 && b0 <= a1,
            _ => false,
        }
    }

    fn render(&self) -> String {
        match (self.min, self.max) {
            (Some(a), Some(b)) => format!("[{a}, {b}] (never: {}/{})", self.never, self.total),
            _ => format!("not observed (never: {}/{})", self.never, self.total),
        }
    }
}

/// Frontier of one slice `(scale, budget, endowment)` in one split.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceFrontier {
    /// Split.
    pub split: SeedSplit,
    /// Scale.
    pub scale: usize,
    /// Budget.
    pub budget: usize,
    /// Endowment label.
    pub endowment: String,
    /// Interval across seeds.
    pub interval: FrontierInterval,
}

/// Full sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct SweepReport {
    /// Sweep id.
    pub sweep_id: String,
    /// Cost declared before running.
    pub declared_cost: u64,
    /// Number of cells in the product.
    pub cell_count: usize,
    /// One entry per `(cell, split)`.
    pub results: Vec<CellResult>,
    /// Frontiers per slice and split.
    pub frontiers: Vec<SliceFrontier>,
    /// Axes echoed for rendering.
    pub axes: SweepAxes,
}

/// Declared total cost of a sweep, computed without running anything.
pub fn declared_cost(config: &SweepConfig, world: &dyn SweepWorld) -> Result<u64, SweepError> {
    validate(config)?;
    let n_seeds = (config.seeds.dev_seeds.len() + config.seeds.holdout_seeds.len()) as u64;
    let mut total = 0u64;
    for cell in cells(&config.axes) {
        let e = &config.axes.endowments[cell.endowment_index];
        total = total.saturating_add(world.seed_cost(&cell, e).saturating_mul(n_seeds));
    }
    Ok(total)
}

fn validate(config: &SweepConfig) -> Result<(), SweepError> {
    let a = &config.axes;
    if a.horizons.is_empty() {
        return Err(SweepError::EmptyAxis("horizons"));
    }
    if a.scales.is_empty() {
        return Err(SweepError::EmptyAxis("scales"));
    }
    if a.budgets.is_empty() {
        return Err(SweepError::EmptyAxis("budgets"));
    }
    if a.endowments.is_empty() {
        return Err(SweepError::EmptyAxis("endowments"));
    }
    config.seeds.validate().map_err(SweepError::Seeds)
}

/// Cartesian product in canonical order (endowment, budget, scale, horizon; horizon fastest).
fn cells(axes: &SweepAxes) -> Vec<SweepCell> {
    let mut out = Vec::new();
    for endowment_index in 0..axes.endowments.len() {
        for &budget in &axes.budgets {
            for &scale in &axes.scales {
                for &horizon in &axes.horizons {
                    out.push(SweepCell {
                        horizon,
                        scale,
                        budget,
                        endowment_index,
                    });
                }
            }
        }
    }
    out
}

/// Runs the sweep. The budget is checked first; on error nothing is executed.
pub fn run_sweep(config: &SweepConfig, world: &dyn SweepWorld) -> Result<SweepReport, SweepError> {
    let declared = declared_cost(config, world)?;
    if declared > config.budget_limit {
        return Err(SweepError::BudgetExceeded {
            declared,
            limit: config.budget_limit,
        });
    }
    let all_cells = cells(&config.axes);
    let splits = [
        (SeedSplit::Dev, &config.seeds.dev_seeds),
        (SeedSplit::Holdout, &config.seeds.holdout_seeds),
    ];
    let mut results = Vec::new();
    for cell in &all_cells {
        let e = &config.axes.endowments[cell.endowment_index];
        for (split, seeds) in &splits {
            let mut run_errors = 0;
            let mut treated_failures = 0;
            let mut present = 0;
            let mut adv = Vec::new();
            let mut per_seed = Vec::new();
            for &seed in seeds.iter() {
                match world.run_seed(cell, e, seed) {
                    Ok(m) => {
                        let a = m.treated_score - m.baseline_score;
                        let p = a > config.margin;
                        adv.push(a);
                        present += usize::from(p);
                        treated_failures += usize::from(m.treated_failed);
                        per_seed.push((seed.as_u64(), p));
                    }
                    Err(_) => {
                        run_errors += 1;
                        per_seed.push((seed.as_u64(), false));
                    }
                }
            }
            results.push(CellResult {
                cell: cell.clone(),
                split: *split,
                total: seeds.len(),
                run_errors,
                treated_failures,
                present,
                advantage: DispersionStats::compute(&adv),
                per_seed,
            });
        }
    }
    let frontiers = compute_frontiers(config, &results);
    Ok(SweepReport {
        sweep_id: config.sweep_id.clone(),
        declared_cost: declared,
        cell_count: all_cells.len(),
        results,
        frontiers,
        axes: config.axes.clone(),
    })
}

fn compute_frontiers(config: &SweepConfig, results: &[CellResult]) -> Vec<SliceFrontier> {
    let mut horizons = config.axes.horizons.clone();
    horizons.sort_unstable();
    horizons.dedup();
    let mut out = Vec::new();
    for split in [SeedSplit::Dev, SeedSplit::Holdout] {
        for (ei, e) in config.axes.endowments.iter().enumerate() {
            for &budget in &config.axes.budgets {
                for &scale in &config.axes.scales {
                    // seed -> horizon -> present
                    let mut by_seed: BTreeMap<u64, BTreeMap<usize, bool>> = BTreeMap::new();
                    for r in results.iter().filter(|r| {
                        r.split == split
                            && r.cell.endowment_index == ei
                            && r.cell.budget == budget
                            && r.cell.scale == scale
                    }) {
                        for &(s, p) in &r.per_seed {
                            by_seed.entry(s).or_default().insert(r.cell.horizon, p);
                        }
                    }
                    let firsts: Vec<Option<usize>> = by_seed
                        .values()
                        .map(|m| horizons.iter().copied().find(|h| m.get(h) == Some(&true)))
                        .collect();
                    if firsts.is_empty() {
                        continue;
                    }
                    out.push(SliceFrontier {
                        split,
                        scale,
                        budget,
                        endowment: e.label.clone(),
                        interval: FrontierInterval::from_firsts(&firsts),
                    });
                }
            }
        }
    }
    out
}

fn split_label(s: SeedSplit) -> &'static str {
    match s {
        SeedSplit::Dev => "dev",
        SeedSplit::Holdout => "holdout",
    }
}

impl SweepReport {
    /// CSV with one row per `(cell, split)`: denominators, failures and dispersion.
    /// Floats use fixed precision so the text (and its hash) is reproducible.
    pub fn to_csv(&self) -> String {
        let mut s = String::from(
            "split,horizon,scale,budget,endowment,total,run_errors,treated_failures,present,\
             present_rate,adv_mean,adv_std,adv_min,adv_q25,adv_median,adv_q75,adv_max\n",
        );
        for r in &self.results {
            let d = &r.advantage;
            let rate = if r.total == 0 {
                0.0
            } else {
                r.present as f64 / r.total as f64
            };
            let _ = writeln!(
                s,
                "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                split_label(r.split),
                r.cell.horizon,
                r.cell.scale,
                r.cell.budget,
                self.axes.endowments[r.cell.endowment_index].label,
                r.total,
                r.run_errors,
                r.treated_failures,
                r.present,
                rate,
                d.mean,
                d.std_dev,
                d.min,
                d.q25,
                d.median,
                d.q75,
                d.max
            );
        }
        s
    }

    /// FNV-1a 64-bit hash of the CSV text.
    pub fn csv_hash(&self) -> u64 {
        fnv1a(self.to_csv().as_bytes())
    }

    /// SVG heat map of the phenomenon rate (rows = horizon, columns = scale), one panel per
    /// split, averaged over budgets and endowments.
    pub fn to_svg(&self) -> String {
        let mut hs = self.axes.horizons.clone();
        hs.sort_unstable();
        hs.dedup();
        let mut sc = self.axes.scales.clone();
        sc.sort_unstable();
        sc.dedup();
        let (cw, ch, ox, oy) = (60usize, 28usize, 70usize, 40usize);
        let panel_h = oy + hs.len() * ch + 20;
        let width = ox + sc.len() * cw + 20;
        let mut s = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{}\" \
             font-family=\"monospace\" font-size=\"11\">\n",
            panel_h * 2
        );
        for (pi, split) in [SeedSplit::Dev, SeedSplit::Holdout].into_iter().enumerate() {
            let y0 = pi * panel_h;
            let _ = writeln!(
                s,
                "<text x=\"4\" y=\"{}\">{} {} (rate; rows=horizon, cols=bins/axis)</text>",
                y0 + 14,
                self.sweep_id,
                split_label(split)
            );
            for (ci, &scale) in sc.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "<text x=\"{}\" y=\"{}\">{scale}</text>",
                    ox + ci * cw + 4,
                    y0 + oy - 6
                );
            }
            for (ri, &h) in hs.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "<text x=\"4\" y=\"{}\">H={h}</text>",
                    y0 + oy + ri * ch + 18
                );
                for (ci, &scale) in sc.iter().enumerate() {
                    let (p, t) = self
                        .results
                        .iter()
                        .filter(|r| {
                            r.split == split && r.cell.horizon == h && r.cell.scale == scale
                        })
                        .fold((0usize, 0usize), |(p, t), r| (p + r.present, t + r.total));
                    let rate = if t == 0 { 0.0 } else { p as f64 / t as f64 };
                    let g = (255.0 - 200.0 * rate).round() as u8;
                    let _ = writeln!(
                        s,
                        "<rect x=\"{}\" y=\"{}\" width=\"{cw}\" height=\"{ch}\" \
                         fill=\"rgb({g},{g},255)\" stroke=\"#444\"/>\
                         <text x=\"{}\" y=\"{}\">{rate:.2}</text>",
                        ox + ci * cw,
                        y0 + oy + ri * ch,
                        ox + ci * cw + 14,
                        y0 + oy + ri * ch + 18
                    );
                }
            }
        }
        s.push_str("</svg>\n");
        s
    }

    /// Text report: declared cost, frontier per slice with its seed interval, and the
    /// comparison against the pocket-world oracle when given.
    pub fn render_frontier_report(&self, oracle: Option<&OracleFrontier>) -> String {
        let mut s = format!(
            "sweep {} cells={} declared_cost={} csv_hash={:016x}\n",
            self.sweep_id,
            self.cell_count,
            self.declared_cost,
            self.csv_hash()
        );
        for f in &self.frontiers {
            let _ = writeln!(
                s,
                "frontier split={} scale={} budget={} endowment={}: horizon {}",
                split_label(f.split),
                f.scale,
                f.budget,
                f.endowment,
                f.interval.render()
            );
        }
        if let Some(o) = oracle {
            let _ = writeln!(
                s,
                "oracle (pocket world, full viable reachability) horizon {}",
                o.interval.render()
            );
            for f in &self.frontiers {
                let _ = writeln!(
                    s,
                    "  vs split={} scale={} budget={} endowment={}: overlap={}",
                    split_label(f.split),
                    f.scale,
                    f.budget,
                    f.endowment,
                    f.interval.overlaps(&o.interval)
                );
            }
            s.push_str(
                "note: the oracle observable (reachability of viable microstates) differs from the \
                 CartPole observable (upright advantage); overlap is a coarse horizon-scale \
                 comparison, not an equivalence.\n",
            );
        }
        s
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Exact pocket-world frontier: smallest horizon at which every viable microstate is reachable
/// from state 0, per world seed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OracleFrontier {
    /// Interval across world seeds.
    pub interval: FrontierInterval,
}

/// Computes the oracle frontier over `world_seeds` with `n_states` microstates and `horizons`.
/// Worlds whose oracle errors are counted as `never`.
pub fn oracle_frontier(world_seeds: &[u64], n_states: u8, horizons: &[usize]) -> OracleFrontier {
    let mut hs = horizons.to_vec();
    hs.sort_unstable();
    hs.dedup();
    let firsts: Vec<Option<usize>> = world_seeds
        .iter()
        .map(|&ws| {
            let world = BoolWorld::from_seed(ws, n_states).ok()?;
            hs.iter().copied().find(|&h| {
                world
                    .oracle(
                        MicroState(0),
                        &BoolSamplingPolicy::UniformActions,
                        h,
                        OracleLimits::default(),
                    )
                    .map(|r| r.reachability.fraction() >= 1.0)
                    .unwrap_or(false)
            })
        })
        .collect();
    OracleFrontier {
        interval: FrontierInterval::from_firsts(&firsts),
    }
}

/// CartPole entropic adapter: treated = entropic proxy, baseline = no control.
#[derive(Debug, Clone)]
pub struct CartPoleEntropicSweep {
    /// Base configuration; horizon, rollouts, bins and actions are overridden per cell.
    pub base: EntropicExperimentConfig,
}

impl CartPoleEntropicSweep {
    fn cell_config(
        &self,
        cell: &SweepCell,
        e: &Endowment,
    ) -> Result<EntropicExperimentConfig, String> {
        let mut c = self.base.clone();
        c.horizon = cell.horizon;
        c.rollouts_per_step = cell.budget;
        c.candidate_actions = e.actions.clone();
        c.bin_spec = BinSpec::uniform(
            format!("cartpole-x-theta-{}", cell.scale),
            BinningVersion::V1_0,
            vec![
                (-DEFAULT_X_THRESHOLD, DEFAULT_X_THRESHOLD, cell.scale),
                (
                    -DEFAULT_THETA_THRESHOLD_RADIANS,
                    DEFAULT_THETA_THRESHOLD_RADIANS,
                    cell.scale,
                ),
            ],
        )
        .map_err(|e| e.to_string())?;
        Ok(c)
    }
}

impl SweepWorld for CartPoleEntropicSweep {
    fn seed_cost(&self, cell: &SweepCell, e: &Endowment) -> u64 {
        let per_tick = 2 + (e.actions.len() * cell.budget * cell.horizon) as u64;
        per_tick.saturating_mul(self.base.episode_ticks)
    }

    fn run_seed(
        &self,
        cell: &SweepCell,
        e: &Endowment,
        seed: MasterSeed,
    ) -> Result<SeedMeasure, String> {
        let cfg = self.cell_config(cell, e)?;
        let report = run_entropic_comparison(&[seed], &cfg).map_err(|e| e.to_string())?;
        debug_assert_eq!(
            report.entropic_proxy.condition,
            EntropicConditionKind::EntropicProxy
        );
        let t = report
            .entropic_proxy
            .runs
            .first()
            .ok_or("no entropic run")?;
        let b = report.no_control.runs.first().ok_or("no baseline run")?;
        let ticks = cfg.episode_ticks as f64;
        Ok(SeedMeasure {
            treated_score: t.upright_ticks as f64 / ticks,
            baseline_score: b.upright_ticks as f64 / ticks,
            treated_failed: t.failed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endowments() -> Vec<Endowment> {
        vec![Endowment {
            label: "pm10".to_owned(),
            actions: vec![-10.0, 0.0, 10.0],
        }]
    }

    fn config(limit: u64) -> SweepConfig {
        SweepConfig {
            sweep_id: "sweep-cartpole-test".to_owned(),
            axes: SweepAxes {
                horizons: vec![1, 4, 8],
                scales: vec![4, 8],
                budgets: vec![2],
                endowments: endowments(),
            },
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(1), MasterSeed::new(2)],
                vec![MasterSeed::new(101), MasterSeed::new(102)],
            )
            .unwrap(),
            budget_limit: limit,
            margin: 0.0,
        }
    }

    fn world() -> CartPoleEntropicSweep {
        let base = EntropicExperimentConfig {
            episode_ticks: 20,
            ..EntropicExperimentConfig::default()
        };
        CartPoleEntropicSweep { base }
    }

    #[test]
    fn sweep_phase_diagram() {
        let w = world();
        let cfg = config(1_000_000);
        let declared = declared_cost(&cfg, &w).unwrap();
        let report = run_sweep(&cfg, &w).unwrap();
        assert_eq!(report.declared_cost, declared);
        assert_eq!(report.cell_count, 6);
        assert_eq!(report.results.len(), 12);
        for r in &report.results {
            assert_eq!(r.total, 2);
            assert!(r.present <= r.total);
        }
        let csv = report.to_csv();
        assert_eq!(csv.lines().count(), 13);
        assert!(csv.contains("holdout") && csv.contains("dev"));
        // Reproducible by hash.
        let again = run_sweep(&cfg, &w).unwrap();
        assert_eq!(report.csv_hash(), again.csv_hash());
        assert_eq!(csv, again.to_csv());
        // SVG heat map.
        let svg = report.to_svg();
        assert!(svg.starts_with("<svg") && svg.contains("<rect"));
        // Frontier + oracle comparison (result reported as-is, whatever it is).
        let oracle = oracle_frontier(&[1, 2, 3, 4], 4, &[1, 2, 3, 4, 6, 8]);
        assert_eq!(oracle.interval.total, 4);
        let text = report.render_frontier_report(Some(&oracle));
        assert!(text.contains("oracle") && text.contains("frontier"));
        assert!(!report.frontiers.is_empty());
    }

    #[test]
    fn sweep_budget_exceeded_before_running() {
        let w = world();
        let cfg = config(10);
        match run_sweep(&cfg, &w) {
            Err(SweepError::BudgetExceeded { declared, limit }) => {
                assert!(declared > limit);
                assert_eq!(limit, 10);
            }
            other => panic!("expected BudgetExceeded, got {other:?}"),
        }
    }

    #[test]
    fn sweep_empty_axis_is_rejected() {
        let mut cfg = config(1_000);
        cfg.axes.horizons.clear();
        assert_eq!(
            run_sweep(&cfg, &world()).unwrap_err(),
            SweepError::EmptyAxis("horizons")
        );
    }

    #[test]
    fn oracle_frontier_interval_is_well_formed() {
        let o = oracle_frontier(&[1, 2, 3], 2, &[1, 2, 3]);
        if let (Some(a), Some(b)) = (o.interval.min, o.interval.max) {
            assert!(a <= b);
        }
        assert_eq!(o.interval.total, 3);
    }

    #[test]
    fn frontier_intervals_overlap_logic() {
        let a = FrontierInterval::from_firsts(&[Some(2), Some(4)]);
        let b = FrontierInterval::from_firsts(&[Some(4), None]);
        let c = FrontierInterval::from_firsts(&[None]);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
        assert_eq!(b.never, 1);
    }
}
