//! Control bits from already-simulated FMC walkers (GOG-180, DEM-027).
//!
//! # What it computes
//! `control = H(Z) - H(Z | A0) = I(A0; Z)` in **bits**, where `A0` is each walker's first action
//! (`WalkerPopulation::first_actions`) and `Z` its final macrostate under a scale. No extra
//! transition is simulated: the function is pure over `(first_action, final_state)` pairs the
//! planner already produced.
//!
//! # Scale
//! Since GOG-200 `phi` is a typed [`crate::Scale`] contract with identifier, version and hash;
//! [`crate::Scale::bins`] exposes the canonical [`BinSpec`] this module consumes. The generic
//! entry points stay on `&BinSpec` (they are the macrostate-key core), and the
//! [`control_bits_from_walkers_on_scale`] / [`control_bits_from_population_on_scale`] helpers
//! bridge the typified scale so callers never rebuild bin parameters by hand. The macrostate key
//! is the vector of bin coordinates; the scale label and [`BinningVersion`] are copied into the
//! estimate so records stay attributable.
//!
//! # Estimator
//! - Plug-in entropies reuse [`shannon_entropy_bits`].
//! - Bias correction: Miller-Madow, `H_MM = H + (K - 1) / (2 n ln 2)` with `K` the number of
//!   observed cells. Control is `H_MM(Z) + H_MM(A) - H_MM(Z, A)` (equal to
//!   `H_MM(Z) - H_MM(Z | A)` with `H(Z|A) = H(Z,A) - H(A)`). The point estimate is **not**
//!   clamped at 0, so under independence it fluctuates around 0 (possibly slightly negative).
//! - Coverage: Good-Turing `C = 1 - f1 / n`, `f1` = macrostates observed exactly once.
//! - Interval: deterministic bootstrap of the corrected estimate `t`: percentile interval
//!   `[q_lo, q_hi]` shifted by the mean bootstrap bias (`mean(boot) - t`; resamples with
//!   duplicated pairs inflate the estimate, and the *basic* interval `[2t - q_hi, 2t - q_lo]`
//!   was measured to exclude 0 on pure noise, so it is not used), plus a deterministic permutation null (actions shuffled) whose
//!   upper quantile is reported as `null_upper_bits`. Both use [`DeterministicRng`] seeded from
//!   the config.
//!
//! # Cloning bias (FMC)
//! FMC cloning copies `(first_action, state)` from a source walker to a destination, so the
//! walker set is neither independent nor identically distributed w.r.t. the open-loop sampling
//! distribution: duplicated pairs inflate the effective sample size and reward-driven selection
//! tilts the action marginal. The estimator treats pairs as i.i.d., so on a cloned population
//! its interval is too narrow and its value is biased. The size of the effect on the pocket
//! fixture is measured in the test `control_bits_cloning_bias_is_quantified`; see the rustdoc
//! of [`CLONING_BIAS_NOTE`] for the recorded result (about -0.09 bits vs plain sampling).

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::action::Action;
use crate::domain::State;
use crate::entropy::{BinSpec, BinningVersion, EntropyError, shannon_entropy_bits};
use crate::fmc_walkers::WalkerPopulation;
use crate::model::{DeterministicRng, RngSource};

/// Version of the estimator (Miller-Madow + Good-Turing + bias-shifted bootstrap); bump on any change.
pub const CONTROL_BITS_ESTIMATOR_VERSION: u32 = 1;

/// Pre-registered minimum Spearman rank correlation between estimated control and the reference
/// empowerment (channel capacity) over pocket worlds (GOG-180 AC1). Fixed before measuring.
pub const CONTROL_BITS_MIN_SPEARMAN: f64 = 0.8;

/// Recorded result of the cloning-bias measurement (test
/// `control_bits_cloning_bias_is_quantified`, deterministic seeds 1000..1300).
///
/// Pocket fixture: 4 actions, 4 macrostates, noise 0.4, `n = 64` walkers, 300 replicates; the
/// cloned arm applies one synchronous clone round (real `clone_walkers_synchronously`) that
/// overwrites the 25% lowest-reward walkers with copies of the 25% highest-reward ones
/// (reward = macrostate index). True `I(A0; Z)` (uniform prior = capacity here) = 0.6432 bits.
///
/// - Sampling without cloning: mean corrected control 0.6863 (bias **+0.043** bits: Miller-Madow
///   leaves a residual positive bias at `n = 64`), mean interval width 0.623.
/// - With cloning: mean corrected control 0.5959 (bias **-0.047** bits), mean interval width
///   0.545.
///
/// So on this fixture cloning moves the estimate by about -0.09 bits relative to plain sampling
/// (sign: down, because reward-driven copying concentrates `Z`) and shrinks the reported
/// interval by ~13% even though the effective sample size is smaller, i.e. the interval is
/// over-confident on cloned populations. One fixture, one clone rule: not a general bound.
pub const CLONING_BIAS_NOTE: &str = "FMC cloning (25% reward-driven, n=64, pocket 4x4): corrected control bias -0.047 bits vs +0.043 without cloning (truth 0.6432); mean interval width 0.545 vs 0.623 (over-confident). Single fixture; see test control_bits_cloning_bias_is_quantified.";

/// Errors of the control-bits estimator (fail-closed).
#[derive(Debug, Clone, PartialEq)]
pub enum ControlBitsError {
    /// No samples.
    EmptySample,
    /// `first_actions` and `states` have different lengths.
    LengthMismatch {
        /// Number of first actions.
        actions: usize,
        /// Number of states.
        states: usize,
    },
    /// A walker has no recorded first action (it never stepped).
    MissingFirstAction {
        /// Walker index.
        walker: usize,
    },
    /// Invalid configuration.
    InvalidConfig {
        /// Reason.
        reason: &'static str,
    },
    /// Entropy or scale-projection error.
    Entropy(EntropyError),
}

impl fmt::Display for ControlBitsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySample => write!(f, "control bits need at least one sample"),
            Self::LengthMismatch { actions, states } => {
                write!(f, "{actions} first actions but {states} states")
            }
            Self::MissingFirstAction { walker } => {
                write!(f, "walker {walker} has no first action")
            }
            Self::InvalidConfig { reason } => write!(f, "invalid control bits config: {reason}"),
            Self::Entropy(e) => write!(f, "entropy error: {e}"),
        }
    }
}

impl Error for ControlBitsError {}

impl From<EntropyError> for ControlBitsError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Bootstrap / permutation configuration (deterministic given `seed`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlBitsConfig {
    /// Bootstrap and permutation replicates (`>= 1`).
    pub resamples: usize,
    /// Two-sided confidence level in `(0, 1)`.
    pub confidence: f64,
    /// Seed of the deterministic resampling stream.
    pub seed: u64,
}

impl Default for ControlBitsConfig {
    fn default() -> Self {
        Self {
            resamples: 400,
            confidence: 0.95,
            seed: 180,
        }
    }
}

impl ControlBitsConfig {
    fn validate(&self) -> Result<(), ControlBitsError> {
        if self.resamples == 0 {
            return Err(ControlBitsError::InvalidConfig {
                reason: "resamples must be >= 1",
            });
        }
        if !(self.confidence.is_finite() && self.confidence > 0.0 && self.confidence < 1.0) {
            return Err(ControlBitsError::InvalidConfig {
                reason: "confidence must be in (0, 1)",
            });
        }
        Ok(())
    }
}

/// Result of the estimator. All entropies/control are in **bits**.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlBitsEstimate {
    /// Label of the scale (`BinSpec::label`) or `""` for raw pairs.
    pub scale_label: String,
    /// Version of the scale (`BinSpec::version`).
    pub scale_version: BinningVersion,
    /// Estimator version ([`CONTROL_BITS_ESTIMATOR_VERSION`]).
    pub estimator_version: u32,
    /// Number of `(a0, z)` samples.
    pub samples: usize,
    /// Distinct first actions observed.
    pub distinct_actions: usize,
    /// Distinct macrostates observed.
    pub distinct_macrostates: usize,
    /// Macrostates observed exactly once (`f1`).
    pub singletons_f1: usize,
    /// Good-Turing coverage `1 - f1 / n` of the macrostate distribution.
    pub coverage: f64,
    /// Plug-in `H(Z)`.
    pub h_z_plugin: f64,
    /// Miller-Madow `H(Z)`.
    pub h_z_corrected: f64,
    /// Plug-in `H(Z | A0)`.
    pub h_z_given_a_plugin: f64,
    /// Miller-Madow `H(Z | A0)`.
    pub h_z_given_a_corrected: f64,
    /// Plug-in `I(A0; Z)` (biased upwards).
    pub control_plugin: f64,
    /// Bias-corrected control `H_MM(Z) - H_MM(Z | A0)` (not clamped).
    pub control_corrected: f64,
    /// Lower end of the bias-shifted bootstrap percentile interval of `control_corrected`.
    pub ci_low: f64,
    /// Upper end of the bias-shifted bootstrap percentile interval of `control_corrected`.
    pub ci_high: f64,
    /// Upper `confidence` quantile of the permutation null of `control_corrected`.
    pub null_upper_bits: f64,
    /// Confidence level used.
    pub confidence: f64,
    /// Resamples used.
    pub resamples: usize,
    /// Seed used.
    pub seed: u64,
}

impl ControlBitsEstimate {
    /// `true` when `0` lies in the bootstrap interval.
    pub fn zero_in_interval(&self) -> bool {
        self.ci_low <= 0.0 && 0.0 <= self.ci_high
    }
}

struct Stats {
    h_z: f64,
    h_z_mm: f64,
    h_z_given_a: f64,
    h_z_given_a_mm: f64,
    control_mm: f64,
}

fn entropy_of_counts(counts: &[usize], n: usize) -> Result<(f64, usize), EntropyError> {
    let probs: Vec<f64> = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| c as f64 / n as f64)
        .collect();
    let k = probs.len();
    // Sum may differ from 1 by float rounding only; use the default tolerance.
    Ok((shannon_entropy_bits(&probs)?, k))
}

fn mm(h: f64, k: usize, n: usize) -> f64 {
    h + (k.saturating_sub(1)) as f64 / (2.0 * n as f64 * std::f64::consts::LN_2)
}

fn stats(pairs: &[(usize, usize)], na: usize, nz: usize) -> Result<Stats, EntropyError> {
    let n = pairs.len();
    let mut ca = vec![0usize; na];
    let mut cz = vec![0usize; nz];
    let mut cza = vec![0usize; na * nz];
    for &(a, z) in pairs {
        ca[a] += 1;
        cz[z] += 1;
        cza[a * nz + z] += 1;
    }
    let (h_a, k_a) = entropy_of_counts(&ca, n)?;
    let (h_z, k_z) = entropy_of_counts(&cz, n)?;
    let (h_za, k_za) = entropy_of_counts(&cza, n)?;
    let (h_a_mm, h_z_mm, h_za_mm) = (mm(h_a, k_a, n), mm(h_z, k_z, n), mm(h_za, k_za, n));
    Ok(Stats {
        h_z,
        h_z_mm,
        h_z_given_a: h_za - h_a,
        h_z_given_a_mm: h_za_mm - h_a_mm,
        control_mm: h_z_mm + h_a_mm - h_za_mm,
    })
}

fn control_mm_only(pairs: &[(usize, usize)], na: usize, nz: usize) -> Result<f64, EntropyError> {
    Ok(stats(pairs, na, nz)?.control_mm)
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn dense_ids<K: Ord + Clone>(keys: &[K]) -> (Vec<usize>, usize) {
    let mut map: BTreeMap<K, usize> = BTreeMap::new();
    let mut ids = Vec::with_capacity(keys.len());
    for k in keys {
        let next = map.len();
        ids.push(*map.entry(k.clone()).or_insert(next));
    }
    (ids, map.len())
}

/// Control bits over generic `(first action key, macrostate key)` pairs.
///
/// `scale_label` / `scale_version` are copied into the estimate for attribution only.
pub fn control_bits_from_pairs<A: Ord + Clone, Z: Ord + Clone>(
    pairs: &[(A, Z)],
    scale_label: &str,
    scale_version: BinningVersion,
    config: &ControlBitsConfig,
) -> Result<ControlBitsEstimate, ControlBitsError> {
    config.validate()?;
    if pairs.is_empty() {
        return Err(ControlBitsError::EmptySample);
    }
    let n = pairs.len();
    let a_keys: Vec<A> = pairs.iter().map(|p| p.0.clone()).collect();
    let z_keys: Vec<Z> = pairs.iter().map(|p| p.1.clone()).collect();
    let (a_ids, na) = dense_ids(&a_keys);
    let (z_ids, nz) = dense_ids(&z_keys);
    let ids: Vec<(usize, usize)> = a_ids.iter().copied().zip(z_ids.iter().copied()).collect();

    let s = stats(&ids, na, nz)?;
    let mut count_z = vec![0usize; nz];
    for &z in &z_ids {
        count_z[z] += 1;
    }
    let f1 = count_z.iter().filter(|&&c| c == 1).count();
    let coverage = 1.0 - f1 as f64 / n as f64;

    let mut rng = DeterministicRng::new(config.seed);
    // Bootstrap of the corrected estimate.
    let mut boot = Vec::with_capacity(config.resamples);
    let mut buf = vec![(0usize, 0usize); n];
    for _ in 0..config.resamples {
        for slot in buf.iter_mut() {
            *slot = ids[rng.next_usize_unbiased(n)];
        }
        boot.push(control_mm_only(&buf, na, nz)?);
    }
    boot.sort_by(|a, b| a.total_cmp(b));
    let alpha = (1.0 - config.confidence) / 2.0;
    let q_lo = quantile(&boot, alpha);
    let q_hi = quantile(&boot, 1.0 - alpha);
    let t = s.control_mm;
    // Bootstrap resamples contain duplicated pairs, which inflates the (even corrected) estimate
    // near the null; the basic interval `[2t - q_hi, 2t - q_lo]` then excludes the true value.
    // Instead shift the percentile interval by the mean bootstrap bias.
    let bias = boot.iter().sum::<f64>() / boot.len() as f64 - t;
    let (ci_low, ci_high) = (q_lo - bias, q_hi - bias);

    // Permutation null: shuffle the action column (breaks any A-Z dependence).
    let mut null = Vec::with_capacity(config.resamples);
    let mut a_perm = a_ids.clone();
    for _ in 0..config.resamples {
        for i in (1..n).rev() {
            let j = rng.next_usize_unbiased(i + 1);
            a_perm.swap(i, j);
        }
        for (slot, (&a, &z)) in buf.iter_mut().zip(a_perm.iter().zip(z_ids.iter())) {
            *slot = (a, z);
        }
        null.push(control_mm_only(&buf, na, nz)?);
    }
    null.sort_by(|a, b| a.total_cmp(b));
    let null_upper_bits = quantile(&null, config.confidence);

    Ok(ControlBitsEstimate {
        scale_label: scale_label.to_string(),
        scale_version,
        estimator_version: CONTROL_BITS_ESTIMATOR_VERSION,
        samples: n,
        distinct_actions: na,
        distinct_macrostates: nz,
        singletons_f1: f1,
        coverage,
        h_z_plugin: s.h_z,
        h_z_corrected: s.h_z_mm,
        h_z_given_a_plugin: s.h_z_given_a,
        h_z_given_a_corrected: s.h_z_given_a_mm,
        control_plugin: s.h_z - s.h_z_given_a,
        control_corrected: t,
        ci_low,
        ci_high,
        null_upper_bits,
        confidence: config.confidence,
        resamples: config.resamples,
        seed: config.seed,
    })
}

fn action_key(action: &Action) -> Vec<u64> {
    match action {
        Action::Discrete(d) => vec![d.index() as u64],
        Action::Continuous(v) => v.iter().map(|x| x.to_bits()).collect(),
    }
}

/// Control bits from walkers' first actions and final states projected through `scale`.
///
/// `scale` is the `phi` projection: a [`BinSpec`] over `State::values()` (see module docs).
pub fn control_bits_from_walkers(
    first_actions: &[Option<Action>],
    states: &[State],
    scale: &BinSpec,
    config: &ControlBitsConfig,
) -> Result<ControlBitsEstimate, ControlBitsError> {
    if first_actions.len() != states.len() {
        return Err(ControlBitsError::LengthMismatch {
            actions: first_actions.len(),
            states: states.len(),
        });
    }
    let mut pairs = Vec::with_capacity(states.len());
    for (i, (a, s)) in first_actions.iter().zip(states).enumerate() {
        let a = a
            .as_ref()
            .ok_or(ControlBitsError::MissingFirstAction { walker: i })?;
        pairs.push((action_key(a), scale.bin_coords(s.values())?));
    }
    control_bits_from_pairs(&pairs, scale.label(), scale.version(), config)
}

/// Convenience over [`control_bits_from_walkers`] for a [`WalkerPopulation`].
pub fn control_bits_from_population(
    population: &WalkerPopulation,
    scale: &BinSpec,
    config: &ControlBitsConfig,
) -> Result<ControlBitsEstimate, ControlBitsError> {
    control_bits_from_walkers(
        population.first_actions(),
        population.states(),
        scale,
        config,
    )
}

/// Control bits from walkers projected through a typed [`crate::Scale`] (GOG-200).
///
/// Equivalent to [`control_bits_from_walkers`] with `scale.bins()`; keeps a single typed scale
/// handle across learner, planner and post-hoc analysis.
pub fn control_bits_from_walkers_on_scale(
    first_actions: &[Option<Action>],
    states: &[State],
    scale: &crate::Scale,
    config: &ControlBitsConfig,
) -> Result<ControlBitsEstimate, ControlBitsError> {
    control_bits_from_walkers(first_actions, states, scale.bins(), config)
}

/// Convenience over [`control_bits_from_walkers_on_scale`] for a [`WalkerPopulation`].
pub fn control_bits_from_population_on_scale(
    population: &WalkerPopulation,
    scale: &crate::Scale,
    config: &ControlBitsConfig,
) -> Result<ControlBitsEstimate, ControlBitsError> {
    control_bits_from_walkers_on_scale(
        population.first_actions(),
        population.states(),
        scale,
        config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionSpace, DiscreteAction};
    use crate::determinism::MasterSeed;
    use crate::domain::{Observation, Terminal, Transition};
    use crate::empowerment::{
        BlahutArimotoConfig, DiscreteChannel, ObservationSpaceSpec, compute_reference_empowerment,
    };
    use crate::error::ModelError;
    use crate::model::GenerativeModel;
    use crate::planner::{BudgetTracker, PlanBudget};
    use crate::units::{SimDuration, StepDt, Tick};

    const K: usize = 4;

    /// Pocket channel: action `a` lands on `z = a` with prob `1 - noise`, otherwise uniform.
    /// `rows` = per-action distribution over `K` macrostates.
    fn pocket_rows(actions: usize, noise: f64) -> Vec<Vec<f64>> {
        (0..actions)
            .map(|a| {
                (0..K)
                    .map(|z| {
                        let hit = if z == a % K { 1.0 - noise } else { 0.0 };
                        hit + noise / K as f64
                    })
                    .collect()
            })
            .collect()
    }

    fn sample_z(row: &[f64], rng: &mut DeterministicRng) -> usize {
        let u = rng.next_f64();
        let mut acc = 0.0;
        for (z, p) in row.iter().enumerate() {
            acc += p;
            if u < acc {
                return z;
            }
        }
        row.len() - 1
    }

    fn reference_capacity(rows: &[Vec<f64>]) -> f64 {
        let ch = DiscreteChannel::from_matrix(
            rows.len(),
            1,
            ObservationSpaceSpec::with_cardinality(K).unwrap(),
            rows.to_vec(),
            1024,
            1e-9,
        )
        .unwrap();
        compute_reference_empowerment(&ch, &BlahutArimotoConfig::default())
            .unwrap()
            .capacity_bits
    }

    fn spearman(x: &[f64], y: &[f64]) -> f64 {
        fn ranks(v: &[f64]) -> Vec<f64> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
            let mut r = vec![0.0; v.len()];
            let mut i = 0;
            while i < idx.len() {
                let mut j = i;
                while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
                    j += 1;
                }
                let avg = (i + j) as f64 / 2.0 + 1.0;
                for t in i..=j {
                    r[idx[t]] = avg;
                }
                i = j + 1;
            }
            r
        }
        let (rx, ry) = (ranks(x), ranks(y));
        let n = x.len() as f64;
        let (mx, my) = (rx.iter().sum::<f64>() / n, ry.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in rx.iter().zip(&ry) {
            sxy += (a - mx) * (b - my);
            sxx += (a - mx).powi(2);
            syy += (b - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    fn sample_pairs(rows: &[Vec<f64>], n: usize, seed: u64) -> Vec<(usize, usize)> {
        let mut rng = DeterministicRng::new(seed);
        (0..n)
            .map(|_| {
                let a = rng.next_usize_unbiased(rows.len());
                (a, sample_z(&rows[a], &mut rng))
            })
            .collect()
    }

    #[test]
    fn control_bits_rank_matches_reference_empowerment_in_pocket_worlds() {
        // Pocket "states" = channels with different action counts and noise levels.
        let worlds: Vec<(usize, f64)> = vec![
            (1, 0.0),
            (2, 0.6),
            (2, 0.2),
            (3, 0.5),
            (3, 0.1),
            (4, 0.7),
            (4, 0.3),
            (4, 0.0),
        ];
        let mut est = Vec::new();
        let mut reference = Vec::new();
        for (i, &(na, noise)) in worlds.iter().enumerate() {
            let rows = pocket_rows(na, noise);
            let cap = reference_capacity(&rows);
            let pairs = sample_pairs(&rows, 2000, 7 + i as u64);
            let e = control_bits_from_pairs(
                &pairs,
                "pocket",
                BinningVersion::V1_0,
                &ControlBitsConfig::default(),
            )
            .unwrap();
            // Estimate sits below the reference (capacity >= uniform-prior I) within interval.
            assert!(
                e.ci_low <= cap + 1e-9,
                "world {i}: ci_low {} above reference {cap}",
                e.ci_low
            );
            assert!(e.coverage > 0.9);
            est.push(e.control_corrected);
            reference.push(cap);
        }
        let rho = spearman(&est, &reference);
        println!("spearman={rho} est={est:?} ref={reference:?}");
        assert!(rho >= CONTROL_BITS_MIN_SPEARMAN, "spearman {rho}");
    }

    #[test]
    fn control_bits_pure_noise_has_high_entropy_and_zero_control() {
        // Pure noise: z uniform over 16 macrostates regardless of the 4 actions.
        let mut rng = DeterministicRng::new(55);
        let pairs: Vec<(usize, usize)> = (0..600)
            .map(|_| (rng.next_usize_unbiased(4), rng.next_usize_unbiased(16)))
            .collect();
        let noise = control_bits_from_pairs(
            &pairs,
            "noise",
            BinningVersion::V1_0,
            &ControlBitsConfig::default(),
        )
        .unwrap();
        // Controlled comparison: same marginal range of Z, deterministic in a (z = a).
        let ctrl_pairs: Vec<(usize, usize)> = (0..600)
            .map(|_| {
                let a = rng.next_usize_unbiased(4);
                (a, a)
            })
            .collect();
        let ctrl = control_bits_from_pairs(
            &ctrl_pairs,
            "ctrl",
            BinningVersion::V1_0,
            &ControlBitsConfig::default(),
        )
        .unwrap();
        println!("noise={noise:?}");
        assert!(noise.h_z_plugin > 3.8 && noise.h_z_plugin > ctrl.h_z_plugin + 1.5);
        assert!(
            noise.zero_in_interval(),
            "interval [{}, {}]",
            noise.ci_low,
            noise.ci_high
        );
        assert!(noise.control_corrected.abs() < 0.05);
        assert!(noise.control_plugin > noise.control_corrected);
        assert!(noise.coverage > 0.95 && noise.coverage <= 1.0);
        assert_eq!(
            noise.singletons_f1,
            (noise.samples as f64 * (1.0 - noise.coverage)).round() as usize
        );
        assert!(ctrl.control_corrected > 1.8);
    }

    #[test]
    fn control_bits_is_deterministic_and_fail_closed() {
        let rows = pocket_rows(3, 0.2);
        let pairs = sample_pairs(&rows, 300, 1);
        let cfg = ControlBitsConfig::default();
        let a = control_bits_from_pairs(&pairs, "s", BinningVersion::V1_0, &cfg).unwrap();
        let b = control_bits_from_pairs(&pairs, "s", BinningVersion::V1_0, &cfg).unwrap();
        assert_eq!(a, b);
        let empty: Vec<(usize, usize)> = Vec::new();
        assert_eq!(
            control_bits_from_pairs(&empty, "s", BinningVersion::V1_0, &cfg),
            Err(ControlBitsError::EmptySample)
        );
        let bad = ControlBitsConfig {
            resamples: 0,
            ..cfg
        };
        assert!(control_bits_from_pairs(&pairs, "s", BinningVersion::V1_0, &bad).is_err());
        let scale = BinSpec::uniform("s", BinningVersion::V1_0, vec![(0.0, 4.0, 4)]).unwrap();
        assert!(matches!(
            control_bits_from_walkers(&[None], &[], &scale, &cfg),
            Err(ControlBitsError::LengthMismatch { .. })
        ));
    }

    // ---- Population-level fixture for the cloning-bias measurement --------------------------

    struct PocketModel {
        space: ActionSpace,
        rows: Vec<Vec<f64>>,
    }

    impl GenerativeModel for PocketModel {
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
            let a = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            let row = &self.rows[a];
            let u = rng.next_f64();
            let (mut acc, mut z) = (0.0, row.len() - 1);
            for (i, p) in row.iter().enumerate() {
                acc += p;
                if u < acc {
                    z = i;
                    break;
                }
            }
            let next = state
                .advance(dt, vec![z as f64 + 0.5])
                .map_err(ModelError::TransitionFailed)?;
            let obs = Observation::new(next.tick(), next.elapsed(), next.values().to_vec())
                .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next, obs, Terminal::Ongoing)
                .map_err(ModelError::TransitionFailed)
        }
    }

    /// Runs one population of `n` walkers one step, optionally with a synchronous clone round.
    /// Reward of a walker is its macrostate index (so cloning is reward-driven toward high `z`).
    fn run_population(
        model: &PocketModel,
        n: usize,
        seed: u64,
        clone_fraction: f64,
    ) -> WalkerPopulation {
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let mut pop = WalkerPopulation::new(root, n, MasterSeed::new(seed), 0, 0, false).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut rng = DeterministicRng::new(seed ^ 0xABCD);
        let actions: Vec<Action> = (0..n)
            .map(|_| {
                Action::discrete_in_space(
                    rng.next_usize_unbiased(model.rows.len()),
                    model.action_space(),
                )
                .unwrap()
            })
            .collect();
        let mut budget = BudgetTracker::new(PlanBudget::new(n * 4, None));
        pop.step_population(model, &actions, dt, &mut budget, |_, _, t| {
            t.next_state().values()[0]
        })
        .unwrap();
        if clone_fraction > 0.0 {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                pop.state(crate::fmc_walkers::WalkerId::new(a))
                    .unwrap()
                    .values()[0]
                    .total_cmp(
                        &pop.state(crate::fmc_walkers::WalkerId::new(b))
                            .unwrap()
                            .values()[0],
                    )
                    .then(a.cmp(&b))
            });
            let m = ((n as f64) * clone_fraction) as usize;
            let clones: Vec<_> = (0..m)
                .map(|i| {
                    (
                        crate::fmc_walkers::WalkerId::new(order[n - 1 - i]),
                        crate::fmc_walkers::WalkerId::new(order[i]),
                    )
                })
                .collect();
            pop.clone_walkers_synchronously(&clones, seed).unwrap();
        }
        pop
    }

    #[test]
    fn control_bits_cloning_bias_is_quantified() {
        let model = PocketModel {
            space: ActionSpace::discrete(K).unwrap(),
            rows: pocket_rows(K, 0.4),
        };
        let truth = reference_capacity(&model.rows); // uniform prior is optimal here (symmetric)
        let scale =
            BinSpec::uniform("pocket_z", BinningVersion::V1_0, vec![(0.0, K as f64, K)]).unwrap();
        let cfg = ControlBitsConfig {
            resamples: 100,
            ..ControlBitsConfig::default()
        };
        let reps = 300u64;
        let n = 64;
        let (mut plain, mut cloned) = (0.0, 0.0);
        let (mut plain_w, mut cloned_w) = (0.0, 0.0);
        for r in 0..reps {
            let p = run_population(&model, n, 1000 + r, 0.0);
            let c = run_population(&model, n, 1000 + r, 0.25);
            let ep = control_bits_from_population(&p, &scale, &cfg).unwrap();
            let ec = control_bits_from_population(&c, &scale, &cfg).unwrap();
            plain += ep.control_corrected;
            cloned += ec.control_corrected;
            plain_w += ep.ci_high - ep.ci_low;
            cloned_w += ec.ci_high - ec.ci_low;
        }
        let reps_f = reps as f64;
        let (plain, cloned) = (plain / reps_f, cloned / reps_f);
        let (plain_w, cloned_w) = (plain_w / reps_f, cloned_w / reps_f);
        println!(
            "truth={truth:.4} plain_mean={plain:.4} (bias {:.4}) cloned_mean={cloned:.4} \
             (bias {:.4}) mean_ci_width plain={plain_w:.4} cloned={cloned_w:.4}",
            plain - truth,
            cloned - truth
        );
        // Recorded measurement (see CLONING_BIAS_NOTE): plain +0.043, cloned -0.047 bits.
        assert!(
            (plain - truth - 0.043).abs() < 0.01,
            "plain bias {}",
            plain - truth
        );
        assert!(
            (cloned - truth + 0.047).abs() < 0.01,
            "cloned bias {}",
            cloned - truth
        );
        assert!(
            cloned < plain,
            "cloning must lower the estimate on this fixture"
        );
        assert!(
            cloned_w < plain_w,
            "cloned interval should be (over-confidently) narrower"
        );
    }
}
