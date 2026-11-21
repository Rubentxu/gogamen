//! Freedom field: exact soft-Bellman value of viable futures over a learned coarse model
//! (GOG-182 / DEM-034, `std`-only).
//!
//! # What it computes
//! Over a table `P(s' | s, a)` on a coarse scale (the [`TabularLearnedModel`] of a frozen
//! [`ModelVersion`], or any explicit [`FreedomKernel`]) with one **absorbing failure bucket**,
//! the field is the log of the mass of viable trajectories, by log-domain value iteration:
//!
//! ```text
//! V_0(s)     = 0                                   (s regular),  V(failure) = -inf
//! Q_k(s, a)  = log sum_{s' regular} P(s'|s,a) exp(V_k(s'))
//! V_{k+1}(s) = log sum_a  w_a exp(Q_k(s, a))       (log-sum-exp over actions)
//! ```
//!
//! with `w_a = 1/|A|` ([`ActionMeasure::UniformPolicy`]) or `w_a = 1`
//! ([`ActionMeasure::CountActions`], the maximum-entropy count of viable action sequences). With
//! the uniform measure `exp(V_k(s))` is exactly the probability of surviving `k` steps without
//! entering the failure bucket under uniform actions, i.e. the unnormalized `viable_mass` of an
//! exhaustive oracle; `V = -inf` means zero viability. The failure bucket is absorbing: it
//! contributes zero mass and never gets a finite value.
//!
//! # Convergence report
//! Every iteration records a residual: the span `max - min` over states of the increment
//! `V_k(s) - V_{k-1}(s)`. For a primitive transfer matrix the increments converge to a common
//! `ln lambda_1` and the span decays like `|lambda_2 / lambda_1|^k`, so the ratio of successive
//! residuals estimates the eigenvalue separation ([`FreedomField::eigenvalue_ratio_estimate`]).
//! This is a power-iteration estimate, not an eigendecomposition: it is meaningless (`None` or
//! non-convergent) for periodic or reducible chains, and `converged` is only `true` when the
//! residual fell below the tolerance.
//!
//! # Versioning, authority, provenance
//! - A field is bound to a [`FreedomFieldKey`]: `(revision, checksum)` of the [`ModelVersion`]
//!   and its [`TabularScale`]. [`FreedomFieldStore::refresh`] recomputes (off the critical path,
//!   on the caller's thread) only when that key changes.
//! - **A field never grants permission.** This module does not import or reference `ActionGuard`,
//!   proposals or decisions: a [`FreedomField`] can only be *read* as a number per bucket (for
//!   example as a planner leaf value). Whether an action is allowed remains the guard's decision.
//! - It is a **derived** quantity ([`EndowmentProvenance::Derived`]), not given and not
//!   measured: it inherits the model's uncertainty, reported as
//!   [`FreedomField::uncertain_pairs`].

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use crate::learned_model::{LearnedModelError, ModelVersion, TabularLearnedModel, TabularScale};
use crate::provenance::EndowmentProvenance;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;
const ROW_SUM_TOLERANCE: f64 = 1e-6;

/// Errors of the freedom field computation.
#[derive(Debug, Clone, PartialEq)]
pub enum FreedomFieldError {
    /// A configuration value is invalid.
    InvalidConfig(&'static str),
    /// A kernel row is not a probability distribution of the right length.
    InvalidRow {
        /// Source bucket.
        state: usize,
        /// Action index.
        action: usize,
    },
    /// `states^2 * actions` exceeds the configured cell limit.
    TooLarge {
        /// Configured limit.
        limit: usize,
    },
    /// The model rejected a prediction.
    Model(LearnedModelError),
}

impl fmt::Display for FreedomFieldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(what) => write!(f, "invalid freedom field config: {what}"),
            Self::InvalidRow { state, action } => {
                write!(f, "invalid kernel row for state {state}, action {action}")
            }
            Self::TooLarge { limit } => write!(f, "kernel exceeds {limit} cells"),
            Self::Model(e) => write!(f, "model error: {e}"),
        }
    }
}

impl Error for FreedomFieldError {}

impl From<LearnedModelError> for FreedomFieldError {
    fn from(e: LearnedModelError) -> Self {
        Self::Model(e)
    }
}

/// How actions are weighted in the log-sum-exp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ActionMeasure {
    /// `w_a = 1/|A|`: viability under uniformly random actions.
    #[default]
    UniformPolicy,
    /// `w_a = 1`: maximum-entropy count of viable action sequences (`V` grows by `ln |A|` per
    /// step relative to [`Self::UniformPolicy`]).
    CountActions,
}

/// Iteration controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreedomFieldConfig {
    /// Action weighting.
    pub measure: ActionMeasure,
    /// `Some(k)`: run exactly `k` iterations (finite-horizon field). `None`: iterate until the
    /// residual is below `tolerance` or `max_iterations` is reached.
    pub horizon: Option<usize>,
    /// Iteration cap for the convergent mode.
    pub max_iterations: usize,
    /// Residual (increment span, nats) below which the convergent mode stops.
    pub tolerance: f64,
    /// Maximum `states^2 * actions` cells of the dense kernel.
    pub max_cells: usize,
}

impl Default for FreedomFieldConfig {
    fn default() -> Self {
        Self {
            measure: ActionMeasure::UniformPolicy,
            horizon: None,
            max_iterations: 500,
            tolerance: 1e-9,
            max_cells: 4_000_000,
        }
    }
}

impl FreedomFieldConfig {
    /// Finite-horizon configuration (exactly `horizon` iterations).
    pub fn with_horizon(horizon: usize) -> Self {
        Self {
            horizon: Some(horizon),
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<(), FreedomFieldError> {
        if self.horizon == Some(0) {
            return Err(FreedomFieldError::InvalidConfig("horizon"));
        }
        if self.max_iterations == 0 {
            return Err(FreedomFieldError::InvalidConfig("max_iterations"));
        }
        if !self.tolerance.is_finite() || self.tolerance < 0.0 {
            return Err(FreedomFieldError::InvalidConfig("tolerance"));
        }
        Ok(())
    }
}

/// Dense transition table over `n` regular buckets plus one absorbing failure bucket
/// (index `n`). Rows are `P(. | s, a)` of length `n + 1`.
#[derive(Debug, Clone, PartialEq)]
pub struct FreedomKernel {
    n: usize,
    actions: usize,
    rows: Vec<f64>,
    uncertain_pairs: usize,
}

impl FreedomKernel {
    /// Empty kernel (all rows zero until set); every `(state, action)` row must be set.
    pub fn new(
        regular_states: usize,
        actions: usize,
        max_cells: usize,
    ) -> Result<Self, FreedomFieldError> {
        if regular_states == 0 {
            return Err(FreedomFieldError::InvalidConfig("regular_states"));
        }
        if actions == 0 {
            return Err(FreedomFieldError::InvalidConfig("actions"));
        }
        let cells = regular_states
            .checked_mul(regular_states + 1)
            .and_then(|c| c.checked_mul(actions))
            .filter(|c| *c <= max_cells)
            .ok_or(FreedomFieldError::TooLarge { limit: max_cells })?;
        Ok(Self {
            n: regular_states,
            actions,
            rows: vec![0.0; cells],
            uncertain_pairs: 0,
        })
    }

    /// Number of regular buckets (the failure bucket is index `regular_states()`).
    pub const fn regular_states(&self) -> usize {
        self.n
    }

    /// Sets the row `P(. | state, action)` (length `n + 1`, last entry = failure probability).
    pub fn set_row(
        &mut self,
        state: usize,
        action: usize,
        row: &[f64],
    ) -> Result<(), FreedomFieldError> {
        let bad = FreedomFieldError::InvalidRow { state, action };
        if state >= self.n || action >= self.actions || row.len() != self.n + 1 {
            return Err(bad);
        }
        if row.iter().any(|p| !p.is_finite() || *p < 0.0)
            || (row.iter().sum::<f64>() - 1.0).abs() > ROW_SUM_TOLERANCE
        {
            return Err(bad);
        }
        let start = (state * self.actions + action) * (self.n + 1);
        self.rows[start..start + self.n + 1].copy_from_slice(row);
        Ok(())
    }

    /// Reads the predictive table of a learned model (`predict` for every bucket and action).
    pub fn from_model(
        model: &TabularLearnedModel,
        max_cells: usize,
    ) -> Result<Self, FreedomFieldError> {
        let scale = model.version().scale();
        let mut kernel = Self::new(scale.num_bins(), model.version().num_actions(), max_cells)?;
        for s in 0..kernel.n {
            for a in 0..kernel.actions {
                let p = model.predict(s, a)?;
                if p.uncertain {
                    kernel.uncertain_pairs += 1;
                }
                kernel.set_row(s, a, &p.distribution)?;
            }
        }
        Ok(kernel)
    }

    fn row(&self, state: usize, action: usize) -> &[f64] {
        let start = (state * self.actions + action) * (self.n + 1);
        &self.rows[start..start + self.n + 1]
    }
}

/// Identity of the model a field was computed from.
#[derive(Debug, Clone, PartialEq)]
pub struct FreedomFieldKey {
    /// Learner revision of the [`ModelVersion`].
    pub revision: u64,
    /// Learner checksum of the [`ModelVersion`].
    pub checksum: u64,
    /// Scale (bucketization) the field is defined on.
    pub scale: TabularScale,
}

impl FreedomFieldKey {
    /// Key of a frozen version.
    pub fn of_version(version: &ModelVersion) -> Self {
        Self {
            revision: version.revision(),
            checksum: version.checksum(),
            scale: version.scale().clone(),
        }
    }

    /// `true` when `version` is exactly the version this key names (revision, checksum, scale).
    pub fn matches(&self, version: &ModelVersion) -> bool {
        self.revision == version.revision()
            && self.checksum == version.checksum()
            && &self.scale == version.scale()
    }
}

/// Convergence record of one iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreedomIteration {
    /// Iteration number (`1` is the first update).
    pub iteration: usize,
    /// Span `max - min` of `V_k - V_{k-1}` over states finite in both iterates (nats).
    pub residual: f64,
    /// Midpoint of that increment range: estimate of `ln lambda_1`.
    pub log_growth: f64,
    /// `residual_k / residual_{k-1}` when the previous residual is positive.
    pub contraction: Option<f64>,
}

/// Versioned freedom field: `V_k` per bucket (the failure bucket is `-inf`).
#[derive(Debug, Clone, PartialEq)]
pub struct FreedomField {
    key: FreedomFieldKey,
    config: FreedomFieldConfig,
    values: Vec<f64>,
    iterations: Vec<FreedomIteration>,
    converged: bool,
    uncertain_pairs: usize,
    field_checksum: u64,
}

fn log_sum_exp(terms: impl Iterator<Item = f64>) -> f64 {
    let terms: Vec<f64> = terms.filter(|t| *t > f64::NEG_INFINITY).collect();
    let max = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    max + terms.iter().map(|t| (t - max).exp()).sum::<f64>().ln()
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

impl FreedomField {
    /// Computes the field of a learned model (the kernel is its predictive table).
    pub fn compute(
        model: &TabularLearnedModel,
        config: FreedomFieldConfig,
    ) -> Result<Self, FreedomFieldError> {
        config.validate()?;
        let kernel = FreedomKernel::from_model(model, config.max_cells)?;
        Self::compute_from_kernel(
            &kernel,
            FreedomFieldKey::of_version(model.version()),
            config,
        )
    }

    /// Computes the field of an explicit kernel (a known model), labelled with `key`.
    pub fn compute_from_kernel(
        kernel: &FreedomKernel,
        key: FreedomFieldKey,
        config: FreedomFieldConfig,
    ) -> Result<Self, FreedomFieldError> {
        config.validate()?;
        let n = kernel.n;
        let log_w = match config.measure {
            ActionMeasure::UniformPolicy => -(kernel.actions as f64).ln(),
            ActionMeasure::CountActions => 0.0,
        };
        let steps = config.horizon.unwrap_or(config.max_iterations);
        let mut v = vec![0.0f64; n];
        let mut iterations: Vec<FreedomIteration> = Vec::new();
        let mut converged = false;
        for k in 1..=steps {
            let next: Vec<f64> = (0..n)
                .map(|s| {
                    log_sum_exp((0..kernel.actions).map(|a| {
                        let row = kernel.row(s, a);
                        let q = log_sum_exp(
                            row[..n]
                                .iter()
                                .zip(&v)
                                .filter(|(p, _)| **p > 0.0)
                                .map(|(p, vs)| p.ln() + vs),
                        );
                        log_w + q
                    }))
                })
                .collect();
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for (new, old) in next.iter().zip(&v) {
                if new.is_finite() && old.is_finite() {
                    lo = lo.min(new - old);
                    hi = hi.max(new - old);
                }
            }
            let (residual, log_growth) = if lo <= hi {
                (hi - lo, 0.5 * (hi + lo))
            } else {
                (0.0, f64::NEG_INFINITY)
            };
            let contraction = iterations
                .last()
                .filter(|p| p.residual > 0.0)
                .map(|p| residual / p.residual);
            iterations.push(FreedomIteration {
                iteration: k,
                residual,
                log_growth,
                contraction,
            });
            v = next;
            if config.horizon.is_none() && residual <= config.tolerance {
                converged = true;
                break;
            }
        }
        if config.horizon.is_some() {
            // A finite horizon is exact by construction; `converged` stays false unless the
            // last residual happens to be within tolerance.
            converged = iterations
                .last()
                .is_some_and(|it| it.residual <= config.tolerance);
        }
        let mut values = v;
        values.push(f64::NEG_INFINITY);

        let mut h = FNV_OFFSET_BASIS;
        fnv(&mut h, &key.revision.to_le_bytes());
        fnv(&mut h, &key.checksum.to_le_bytes());
        fnv(&mut h, &(values.len() as u64).to_le_bytes());
        fnv(&mut h, &[config.measure as u8]);
        fnv(&mut h, &(iterations.len() as u64).to_le_bytes());
        for x in &values {
            fnv(&mut h, &x.to_bits().to_le_bytes());
        }
        Ok(Self {
            key,
            config,
            values,
            iterations,
            converged,
            uncertain_pairs: kernel.uncertain_pairs,
            field_checksum: h,
        })
    }

    /// Provenance marker: always [`EndowmentProvenance::Derived`].
    pub const fn provenance(&self) -> EndowmentProvenance {
        EndowmentProvenance::Derived
    }

    /// Version identity (model revision/checksum and scale).
    pub const fn key(&self) -> &FreedomFieldKey {
        &self.key
    }

    /// Configuration used.
    pub const fn config(&self) -> &FreedomFieldConfig {
        &self.config
    }

    /// Checksum of key, measure and values (stable across runs).
    pub const fn field_checksum(&self) -> u64 {
        self.field_checksum
    }

    /// Per-iteration convergence record.
    pub fn iterations(&self) -> &[FreedomIteration] {
        &self.iterations
    }

    /// Residual of the last iteration.
    pub fn final_residual(&self) -> f64 {
        self.iterations.last().map_or(f64::INFINITY, |i| i.residual)
    }

    /// `true` when the last residual is within the configured tolerance.
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Number of `(bucket, action)` pairs of the source model flagged uncertain.
    pub const fn uncertain_pairs(&self) -> usize {
        self.uncertain_pairs
    }

    /// Estimate of `|lambda_2 / lambda_1|`: the last finite contraction ratio of successive
    /// residuals. `None` when fewer than two positive residuals exist (e.g. the residual hit
    /// zero immediately). Only meaningful for a primitive transfer matrix; see module docs.
    pub fn eigenvalue_ratio_estimate(&self) -> Option<f64> {
        self.iterations.iter().rev().find_map(|i| {
            i.contraction
                .filter(|c| c.is_finite() && i.residual > 1e-13)
        })
    }

    /// Estimate of `ln lambda_1` (growth of `V` per step) from the last iteration.
    pub fn log_growth_estimate(&self) -> Option<f64> {
        self.iterations
            .last()
            .map(|i| i.log_growth)
            .filter(|g| g.is_finite())
    }

    /// `V_k(bucket)`: `-inf` for the failure bucket and for buckets with no viable future;
    /// `None` for an out-of-range bucket.
    pub fn value(&self, bucket: usize) -> Option<f64> {
        self.values.get(bucket).copied()
    }

    /// Normalized potential `max(V(bucket) - max_s V(s), -cap)` (`<= 0`; `-cap` for
    /// zero-viability buckets). Differences of potentials are the leaf-value signal; the
    /// per-step growth `ln lambda_1` cancels in a difference of two buckets of the same `V_k`.
    pub fn potential(&self, bucket: usize, cap: f64) -> f64 {
        let top = self
            .values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        match self.values.get(bucket) {
            Some(v) if v.is_finite() && top.is_finite() => (v - top).max(-cap),
            _ => -cap,
        }
    }
}

/// Holds the latest field and recomputes it only when the model key changes.
#[derive(Debug, Clone)]
pub struct FreedomFieldStore {
    config: FreedomFieldConfig,
    current: Option<Arc<FreedomField>>,
}

impl FreedomFieldStore {
    /// Empty store.
    pub fn new(config: FreedomFieldConfig) -> Self {
        Self {
            config,
            current: None,
        }
    }

    /// Latest field, if any.
    pub fn current(&self) -> Option<&Arc<FreedomField>> {
        self.current.as_ref()
    }

    /// Recomputes when the model's `(revision, checksum, scale)` differs from the stored key.
    /// Returns whether a recomputation happened. Intended to be called off the critical path.
    pub fn refresh(&mut self, model: &TabularLearnedModel) -> Result<bool, FreedomFieldError> {
        if let Some(f) = &self.current
            && f.key().matches(model.version())
        {
            return Ok(false);
        }
        self.current = Some(Arc::new(FreedomField::compute(model, self.config)?));
        Ok(true)
    }
}

#[cfg(test)]
mod freedom_field_tests {
    use super::*;
    use crate::domain::{Terminal, TerminalReason};
    use crate::entropy::{BinSpec, BinningVersion};
    use crate::learned_model::{LearnedModelConfig, UnseenPairPolicy};
    use crate::learning::{
        LearningPolicy, ObservationId, TabularTransitionLearner, TransitionObservation,
    };
    use crate::provenance::EvidenceRef;

    fn key(n: usize) -> FreedomFieldKey {
        let spec =
            BinSpec::uniform("t", BinningVersion::new(1, 0), vec![(0.0, n as f64, n)]).unwrap();
        FreedomFieldKey {
            revision: 1,
            checksum: 2,
            scale: TabularScale::new(spec).unwrap(),
        }
    }

    fn two_state_kernel() -> FreedomKernel {
        // Substochastic [[.7,.2],[.1,.6]]: eigenvalues 0.8 and 0.5 (the all-ones start is not an
        // eigenvector because the failure leak differs per state).
        let mut k = FreedomKernel::new(2, 1, 100).unwrap();
        k.set_row(0, 0, &[0.7, 0.2, 0.1]).unwrap();
        k.set_row(1, 0, &[0.1, 0.6, 0.3]).unwrap();
        k
    }

    #[test]
    fn freedom_field_matches_closed_form_and_reports_eigenvalue_ratio() {
        let f = FreedomField::compute_from_kernel(
            &two_state_kernel(),
            key(2),
            FreedomFieldConfig::default(),
        )
        .unwrap();
        assert!(f.converged());
        // lambda_1 = 0.8, lambda_2 = 0.5.
        let growth = f.log_growth_estimate().unwrap();
        assert!((growth - 0.8f64.ln()).abs() < 1e-8, "{growth}");
        let ratio = f.eigenvalue_ratio_estimate().unwrap();
        assert!((ratio - 0.5 / 0.8).abs() < 1e-3, "{ratio}");
        // Residual per iteration decays monotonically here.
        let r: Vec<f64> = f.iterations().iter().map(|i| i.residual).collect();
        assert!(r.windows(2).all(|w| w[1] <= w[0] + 1e-15), "{r:?}");
        // Finite horizon is exact: V_1 = ln(0.9) and ln(0.7) (row survival masses).
        let h = FreedomField::compute_from_kernel(
            &two_state_kernel(),
            key(2),
            FreedomFieldConfig::with_horizon(1),
        )
        .unwrap();
        assert_eq!(h.iterations().len(), 1);
        assert!((h.value(0).unwrap() - 0.9f64.ln()).abs() < 1e-12);
        assert!((h.value(1).unwrap() - 0.7f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn freedom_field_failure_is_absorbing_and_dead_ends_get_neg_inf() {
        // State 0 -> 1 -> failure (certain); state 2 stays alive forever.
        let mut k = FreedomKernel::new(3, 1, 100).unwrap();
        k.set_row(0, 0, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        k.set_row(1, 0, &[0.0, 0.0, 0.0, 1.0]).unwrap();
        k.set_row(2, 0, &[0.0, 0.0, 1.0, 0.0]).unwrap();
        let f = FreedomField::compute_from_kernel(&k, key(3), FreedomFieldConfig::with_horizon(3))
            .unwrap();
        assert_eq!(f.value(3), Some(f64::NEG_INFINITY));
        assert_eq!(f.value(1), Some(f64::NEG_INFINITY));
        assert_eq!(f.value(0), Some(f64::NEG_INFINITY));
        assert_eq!(f.value(2), Some(0.0));
        assert_eq!(f.potential(2, 5.0), 0.0);
        assert_eq!(f.potential(3, 5.0), -5.0);
        // With horizon 1 state 0 is still viable (it survives one step).
        let f1 = FreedomField::compute_from_kernel(&k, key(3), FreedomFieldConfig::with_horizon(1))
            .unwrap();
        assert_eq!(f1.value(0), Some(0.0));
        assert_eq!(f1.value(1), Some(f64::NEG_INFINITY));
    }

    #[test]
    fn freedom_field_count_measure_adds_ln_actions_per_step() {
        let mut k = FreedomKernel::new(2, 2, 100).unwrap();
        for s in 0..2 {
            k.set_row(s, 0, &[1.0, 0.0, 0.0]).unwrap();
            k.set_row(s, 1, &[0.0, 0.5, 0.5]).unwrap();
        }
        let mean =
            FreedomField::compute_from_kernel(&k, key(2), FreedomFieldConfig::with_horizon(4))
                .unwrap();
        let count = FreedomField::compute_from_kernel(
            &k,
            key(2),
            FreedomFieldConfig {
                measure: ActionMeasure::CountActions,
                ..FreedomFieldConfig::with_horizon(4)
            },
        )
        .unwrap();
        for s in 0..2 {
            let d = count.value(s).unwrap() - mean.value(s).unwrap();
            assert!((d - 4.0 * 2.0f64.ln()).abs() < 1e-12, "{d}");
        }
    }

    #[test]
    fn freedom_field_rejects_invalid_kernel_rows_and_config() {
        let mut k = FreedomKernel::new(2, 1, 100).unwrap();
        assert!(k.set_row(0, 0, &[0.5, 0.2, 0.1]).is_err());
        assert!(k.set_row(0, 0, &[1.0, 0.0]).is_err());
        assert!(k.set_row(0, 0, &[f64::NAN, 0.0, 0.0]).is_err());
        assert!(k.set_row(5, 0, &[1.0, 0.0, 0.0]).is_err());
        assert!(FreedomKernel::new(100, 4, 100).is_err());
        let bad = FreedomFieldConfig::with_horizon(0);
        assert!(FreedomField::compute_from_kernel(&k, key(2), bad).is_err());
    }

    fn learner(fail_in_bucket_1: bool, extra: u64) -> (TabularTransitionLearner, TabularScale) {
        let sc = key(3).scale;
        let mut l =
            TabularTransitionLearner::new(sc.num_states(), 2, LearningPolicy::laplace()).unwrap();
        let mut id = 0;
        for b in 0..3 {
            let x = b as f64 + 0.5;
            for _ in 0..(20 + extra) {
                for action in [0usize, 1] {
                    id += 1;
                    let terminal = if action == 1 && b == 1 && fail_in_bucket_1 {
                        Terminal::Terminated(TerminalReason::Failure)
                    } else {
                        Terminal::Ongoing
                    };
                    let s = sc
                        .sample_from_transition(&[x], action, &[x], terminal)
                        .unwrap();
                    let (o, e) = (ObservationId::new(id), EvidenceRef::new(900 + id));
                    let obs = if s.is_terminal_failure() {
                        TransitionObservation::failure(
                            o,
                            e,
                            s.state_bucket(),
                            s.action_idx(),
                            s.next_state_bucket(),
                        )
                    } else {
                        TransitionObservation::new(
                            o,
                            e,
                            s.state_bucket(),
                            s.action_idx(),
                            s.next_state_bucket(),
                        )
                    };
                    l.observe(&obs).unwrap();
                }
            }
        }
        (l, sc)
    }

    fn model(l: &TabularTransitionLearner, sc: TabularScale) -> TabularLearnedModel {
        let v = ModelVersion::freeze(l, sc).unwrap();
        TabularLearnedModel::new(
            v,
            LearnedModelConfig::new(0.01, 1, UnseenPairPolicy::SamplePrior).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn freedom_field_is_versioned_derived_and_recomputed_only_on_new_version() {
        let (l, sc) = learner(true, 0);
        let m = model(&l, sc.clone());
        let mut store = FreedomFieldStore::new(FreedomFieldConfig::with_horizon(6));
        assert!(store.refresh(&m).unwrap());
        assert!(
            !store.refresh(&m).unwrap(),
            "same version: no recomputation"
        );
        let f = store.current().unwrap().clone();
        assert_eq!(f.provenance(), EndowmentProvenance::Derived);
        assert_eq!(f.key().revision, m.version().revision());
        assert_eq!(f.key().checksum, m.version().checksum());
        assert_eq!(f.key().scale, sc);
        assert!(f.key().matches(m.version()));
        assert_eq!(f.uncertain_pairs(), 0);
        // Bucket 1 has a lethal action learned as failure: lower freedom than bucket 0.
        assert!(f.value(1).unwrap() < f.value(0).unwrap());
        assert_eq!(f.value(sc.failure_bucket()), Some(f64::NEG_INFINITY));
        // New experience: new version key, recomputation, different field checksum.
        let (l2, _) = learner(true, 5);
        let m2 = model(&l2, sc);
        assert!(!f.key().matches(m2.version()));
        assert!(store.refresh(&m2).unwrap());
        let f2 = store.current().unwrap();
        assert!(f2.key().matches(m2.version()));
        assert_ne!(f.field_checksum(), f2.field_checksum());
        // Deterministic.
        let again = FreedomField::compute(&m, FreedomFieldConfig::with_horizon(6)).unwrap();
        assert_eq!(again.field_checksum(), f.field_checksum());
        assert_eq!(&again, &*f);
    }

    #[test]
    fn freedom_field_marks_uncertain_pairs_of_unseen_buckets() {
        let sc = key(3).scale;
        let l =
            TabularTransitionLearner::new(sc.num_states(), 2, LearningPolicy::laplace()).unwrap();
        let m = model(&l, sc);
        let f = FreedomField::compute(&m, FreedomFieldConfig::default()).unwrap();
        assert_eq!(f.uncertain_pairs(), 3 * 2);
        assert!(f.converged());
    }
}
