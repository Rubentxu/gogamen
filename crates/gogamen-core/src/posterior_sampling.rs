//! Posterior sampling over a [`TabularLearnedModel`] (Thompson-style, `std`-only; GOG-195).
//!
//! The learner's belief about each `(state bucket, action)` pair is a Dirichlet over next
//! buckets ([`TabularLearnedModel::concentration`]). Instead of planning on the mean table,
//! [`sample_posterior_table`] draws **one full transition table** from that posterior with a
//! seeded [`DeterministicRng`]; [`SampledTableModel`] then exposes the table as a
//! [`GenerativeModel`] for the planner. No exploration bonus is involved: any exploration
//! comes only from the planner acting on a plausible-but-random model.
//!
//! # Determinism and provenance
//! A [`PosteriorTable`] is a pure function of `(model version checksum, seed, sample_id)`. It
//! carries `seed`, `sample_id`, the source model `revision`/`checksum` and a content `hash` of
//! the table bits, so every decision can record exactly which sample it planned on.
//!
//! # Sampler
//! Gamma variates use Marsaglia-Tsang with the `U^(1/a)` boost for shapes below one. Shapes
//! can be tiny (e.g. `0.05`), which underflows in linear space, so Dirichlet draws are
//! normalised in log space.

use crate::action::{Action, ActionSpace};
use crate::domain::State;
use crate::domain::Transition;
use crate::error::ModelError;
use crate::learned_model::{
    LearnedModelError, TabularLearnedModel, TabularScale, transition_to_bucket,
};
use crate::model::{DeterministicRng, GenerativeModel, RngSource};
use crate::units::StepDt;

/// Provenance of a table: the posterior mean or a seeded posterior draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    /// Normalised Dirichlet concentration (what planning "on the mean" uses).
    Mean,
    /// One draw from the Dirichlet posterior.
    Sample,
}

/// A complete transition table with the identity of the sample that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorTable {
    rows: Vec<Vec<f64>>,
    num_actions: usize,
    kind: TableKind,
    seed: u64,
    sample_id: u64,
    revision: u64,
    checksum: u64,
    hash: u64,
}

impl PosteriorTable {
    /// Mean table or posterior sample.
    pub const fn kind(&self) -> TableKind {
        self.kind
    }
    /// Seed of the sampler (0 for the mean table).
    pub const fn seed(&self) -> u64 {
        self.seed
    }
    /// Identifier of the sample within the seed's stream (0 for the mean table).
    pub const fn sample_id(&self) -> u64 {
        self.sample_id
    }
    /// Revision of the model version the table was derived from.
    pub const fn revision(&self) -> u64 {
        self.revision
    }
    /// Checksum of the model version the table was derived from.
    pub const fn checksum(&self) -> u64 {
        self.checksum
    }
    /// FNV-1a hash over the bit patterns of every probability in the table.
    pub const fn hash(&self) -> u64 {
        self.hash
    }
    /// Next-bucket distribution of `(state_bucket, action_idx)`.
    pub fn row(&self, state_bucket: usize, action_idx: usize) -> Option<&[f64]> {
        if action_idx >= self.num_actions {
            return None;
        }
        self.rows
            .get(state_bucket * self.num_actions + action_idx)
            .map(Vec::as_slice)
    }
}

fn fnv_table(rows: &[Vec<f64>]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for p in rows.iter().flatten() {
        for byte in p.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn build_table(
    model: &TabularLearnedModel,
    mut row_fn: impl FnMut(&[f64]) -> Vec<f64>,
    kind: TableKind,
    seed: u64,
    sample_id: u64,
) -> Result<PosteriorTable, LearnedModelError> {
    let version = model.version();
    let num_actions = version.num_actions();
    let num_states = version.scale().num_states();
    let mut rows = Vec::with_capacity(num_states * num_actions);
    for s in 0..num_states {
        for a in 0..num_actions {
            rows.push(row_fn(&model.concentration(s, a)?));
        }
    }
    let hash = fnv_table(&rows);
    Ok(PosteriorTable {
        rows,
        num_actions,
        kind,
        seed,
        sample_id,
        revision: version.revision(),
        checksum: version.checksum(),
        hash,
    })
}

/// The posterior-mean table (normalised concentrations): the "plan on the mean" baseline.
pub fn posterior_mean_table(
    model: &TabularLearnedModel,
) -> Result<PosteriorTable, LearnedModelError> {
    build_table(
        model,
        |alpha| {
            let total: f64 = alpha.iter().sum();
            alpha.iter().map(|a| a / total).collect()
        },
        TableKind::Mean,
        0,
        0,
    )
}

/// Draws one table from the Dirichlet posterior. Pure in `(model, seed, sample_id)`.
pub fn sample_posterior_table(
    model: &TabularLearnedModel,
    seed: u64,
    sample_id: u64,
) -> Result<PosteriorTable, LearnedModelError> {
    let mut rng = DeterministicRng::new(mix_seed(seed, sample_id));
    build_table(
        model,
        |alpha| sample_dirichlet(alpha, &mut rng),
        TableKind::Sample,
        seed,
        sample_id,
    )
}

/// SplitMix64-style mixing of `(seed, id)` into an independent stream seed.
pub const fn mix_seed(seed: u64, id: u64) -> u64 {
    let mut z = seed ^ id.wrapping_mul(0x9e37_79b9_7f4a_7c15).rotate_left(17);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn standard_normal(rng: &mut dyn RngSource) -> f64 {
    let u1 = loop {
        let u = rng.next_f64();
        if u > 0.0 {
            break u;
        }
    };
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// `ln X` for `X ~ Gamma(shape, 1)`; computed in log space so tiny shapes do not underflow.
///
/// `shape` must be finite and `> 0`.
pub fn sample_ln_gamma(shape: f64, rng: &mut dyn RngSource) -> f64 {
    if shape < 1.0 {
        // G(a) = G(a + 1) * U^(1/a)
        let u = loop {
            let u = rng.next_f64();
            if u > 0.0 {
                break u;
            }
        };
        return sample_ln_gamma(shape + 1.0, rng) + u.ln() / shape;
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let t = 1.0 + c * x;
        if t <= 0.0 {
            continue;
        }
        let v = t * t * t;
        let u = rng.next_f64();
        if u > 0.0 && u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d.ln() + v.ln();
        }
    }
}

/// One draw from `Dirichlet(alpha)` (all entries finite and `> 0`), as probabilities.
pub fn sample_dirichlet(alpha: &[f64], rng: &mut dyn RngSource) -> Vec<f64> {
    let logs: Vec<f64> = alpha.iter().map(|&a| sample_ln_gamma(a, rng)).collect();
    let max = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = logs.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = out.iter().sum();
    for p in &mut out {
        *p /= total;
    }
    out
}

/// [`GenerativeModel`] over a fixed [`PosteriorTable`] (mean or sample) and a model scale.
///
/// It reads no simulator and no learner: the table is the only dynamics it knows.
#[derive(Debug, Clone)]
pub struct SampledTableModel {
    scale: TabularScale,
    space: ActionSpace,
    table: PosteriorTable,
}

impl SampledTableModel {
    /// Wraps `table` (derived from `model`) for planning.
    pub fn new(model: &TabularLearnedModel, table: PosteriorTable) -> Self {
        Self {
            scale: model.version().scale().clone(),
            space: model.action_space().clone(),
            table,
        }
    }

    /// The table (and its provenance) this model plans on.
    pub const fn table(&self) -> &PosteriorTable {
        &self.table
    }

    /// The bucket scale shared with the source model.
    pub const fn scale(&self) -> &TabularScale {
        &self.scale
    }
}

impl GenerativeModel for SampledTableModel {
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
        self.space
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        let action_idx = action
            .as_discrete()
            .ok_or(ModelError::InvalidState {
                reason: "sampled table model requires a discrete action",
            })?
            .index();
        let from = self
            .scale
            .bucket_of(state.values())
            .map_err(|_| ModelError::InvalidState {
                reason: "state cannot be projected onto the model scale",
            })?;
        let row = self
            .table
            .row(from, action_idx)
            .ok_or(ModelError::InvalidState {
                reason: "state bucket outside the sampled table",
            })?;
        let draw = rng.next_f64();
        let mut acc = 0.0;
        let mut next = row.len() - 1;
        for (idx, p) in row.iter().enumerate() {
            acc += p;
            if draw < acc {
                next = idx;
                break;
            }
        }
        transition_to_bucket(&self.scale, state, dt, next)
    }
}

#[cfg(test)]
mod posterior_sampling_tests {
    use super::*;
    use crate::entropy::{BinSpec, BinningVersion};
    use crate::learned_model::{LearnedModelConfig, ModelVersion};
    use crate::learning::{
        LearningPolicy, ObservationId, TabularTransitionLearner, TransitionObservation,
    };
    use crate::provenance::EvidenceRef;

    fn model(observations: u64) -> TabularLearnedModel {
        let spec = BinSpec::uniform("p", BinningVersion::new(1, 0), vec![(0.0, 4.0, 4)]).unwrap();
        let scale = TabularScale::new(spec).unwrap();
        let mut learner =
            TabularTransitionLearner::new(scale.num_states(), 2, LearningPolicy::new(0.1).unwrap())
                .unwrap();
        for i in 0..observations {
            let obs = TransitionObservation::new(
                ObservationId::new(i + 1),
                EvidenceRef::new(100 + i),
                0,
                0,
                1,
            );
            learner.observe(&obs).unwrap();
        }
        let version = ModelVersion::freeze(&learner, scale).unwrap();
        TabularLearnedModel::new(version, LearnedModelConfig::default()).unwrap()
    }

    #[test]
    fn dirichlet_is_a_distribution_even_for_tiny_shapes() {
        let mut rng = DeterministicRng::new(5);
        for shape in [0.01, 0.05, 0.5, 1.0, 7.0] {
            let p = sample_dirichlet(&[shape; 6], &mut rng);
            assert!(p.iter().all(|x| x.is_finite() && *x >= 0.0));
            assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gamma_mean_matches_shape() {
        for shape in [0.3, 2.5] {
            let mut rng = DeterministicRng::new(11);
            let n = 20_000;
            let mean = (0..n)
                .map(|_| sample_ln_gamma(shape, &mut rng).exp())
                .sum::<f64>()
                / f64::from(n);
            assert!(
                (mean - shape).abs() < 0.08 * shape.max(1.0),
                "{shape}: {mean}"
            );
        }
    }

    #[test]
    fn samples_are_deterministic_and_identified() {
        let m = model(6);
        let a = sample_posterior_table(&m, 9, 3).unwrap();
        let b = sample_posterior_table(&m, 9, 3).unwrap();
        assert_eq!(a, b);
        assert_eq!(
            (a.seed(), a.sample_id(), a.kind()),
            (9, 3, TableKind::Sample)
        );
        assert_eq!(a.checksum(), m.version().checksum());
        assert_ne!(a.hash(), sample_posterior_table(&m, 9, 4).unwrap().hash());
        assert_ne!(a.hash(), sample_posterior_table(&m, 10, 3).unwrap().hash());
    }

    #[test]
    fn mean_table_equals_predict_and_sample_average_converges() {
        let m = model(6);
        let mean = posterior_mean_table(&m).unwrap();
        assert_eq!(mean.kind(), TableKind::Mean);
        let pred = m.predict(0, 0).unwrap().distribution;
        for (x, y) in mean.row(0, 0).unwrap().iter().zip(&pred) {
            assert!((x - y).abs() < 1e-12);
        }
        let n = 400;
        let mut acc = vec![0.0; pred.len()];
        for id in 0..n {
            let t = sample_posterior_table(&m, 1, id).unwrap();
            for (a, p) in acc.iter_mut().zip(t.row(0, 0).unwrap()) {
                *a += p / n as f64;
            }
        }
        for (a, p) in acc.iter().zip(&pred) {
            assert!((a - p).abs() < 0.06, "{a} vs {p}");
        }
    }

    #[test]
    fn sampled_model_steps_follow_the_table() {
        let m = model(30);
        let t = sample_posterior_table(&m, 2, 0).unwrap();
        let sm = SampledTableModel::new(&m, t);
        let state = State::new(
            crate::units::Tick::ZERO,
            crate::units::SimDuration::ZERO,
            vec![0.5],
        )
        .unwrap();
        let action = Action::discrete_in_space(0, m.action_space()).unwrap();
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let mut rng = DeterministicRng::new(3);
        let tr = sm.step(&state, &action, dt, &mut rng).unwrap();
        let mut rng2 = DeterministicRng::new(3);
        assert_eq!(tr, sm.step(&state, &action, dt, &mut rng2).unwrap());
    }
}
