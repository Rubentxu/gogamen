//! Multiscale measures over [`BoolWorld`] without false equivalences (GOG-162, DEM-009).
//!
//! # What it does
//! A [`Scale`] is a labelled, versioned projection `phi: micro -> macro` (identity, trivial or
//! an arbitrary partition). [`measure_at_scale`] evaluates, exactly and under ONE explicit
//! intervention policy ([`InterventionPolicy::OpenLoopUniformSequences`]: a uniform prior over
//! open-loop action-index sequences), four quantities that live in SEPARATE types and fields:
//!
//! - [`TrajectoryEntropy`]: `H` in bits of the mixture of projected trajectories
//!   `(phi(s_0), ..., phi(s_T))`. This is an *apparent* entropy: it depends on the scale.
//! - [`CausalControl`]: `I(A; phi(S_T))` in bits (prior uniform over action sequences). A lower
//!   bound on empowerment of the projected variable, NOT empowerment itself (no capacity
//!   maximisation over the prior is done) and NOT effective information.
//! - [`Viability`]: mean raw probability mass that stays viable up to the horizon, plus whether
//!   every macro is viability-homogeneous. Viability is defined on microstates, so a coarse scale
//!   cannot create it.
//! - [`PredictiveError`]: exact predictive loss of the projection, reusing
//!   [`crate::macrostate::analyze_partition`] (GOG-161) with the macro id as observation.
//!
//! [`compare_scales`] relates a micro-side scale and a macro-side scale ([`ScaleRelation`]),
//! reports the per-quantity deltas as separate fields ([`ScaleDeltas`]) and a
//! [`ScaleVerdict`]. There is deliberately no function or field that merges these quantities
//! into a single number, and no type here grants any permission.
//!
//! # Recording the scale
//! Every [`ScaleMeasures`] carries a [`ScaleTag`] (`label`, `version`, `micro_count`,
//! `macro_count`). `label` and `version` have the same shape as the existing scale fields of the
//! scientific trace (`ControlBitsRecord::scale_label` / `scale_version`), so a caller can copy
//! them there; this crate does not depend on the runtime, so it does not write the trace.
//!
//! # Limits and non-claims
//! Exact enumeration with fail-closed limits ([`MultiscaleConfig`]); no randomness, no clock.
//! Coarsening can lower `H` and (by data processing) never raises control; that says nothing
//! about emergence or real-world causality. Tiny worlds only.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_core::{EntropyError, shannon_entropy_bits_with_tolerance};

use crate::bool_world::{BoolWorld, BoolWorldError, MicroState};
use crate::macrostate::{MacroError, MacroLimits, analyze_partition};

/// Current scale-record version `(major, minor)`.
pub const MULTISCALE_SCALE_VERSION: (u32, u32) = (1, 0);

/// Errors of the multiscale analysis (fail-closed).
#[derive(Debug, Clone, PartialEq)]
pub enum MultiscaleError {
    /// Invalid configuration.
    InvalidConfig(&'static str),
    /// Invalid scale (partition does not match the world).
    InvalidScale(&'static str),
    /// Some action sequence leaves no viable mass.
    NoViableMass,
    /// Too many action sequences.
    TooManySequences {
        /// Limit.
        limit: usize,
    },
    /// Too many live paths.
    TooManyPaths {
        /// Limit.
        limit: usize,
    },
    /// Non-finite or inconsistent result.
    Inconsistent(&'static str),
    /// Underlying world error.
    World(BoolWorldError),
    /// Underlying macrostate analysis error.
    Macro(MacroError),
    /// Entropy computation error.
    Entropy(EntropyError),
}

impl fmt::Display for MultiscaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            Self::InvalidScale(m) => write!(f, "invalid scale: {m}"),
            Self::NoViableMass => write!(f, "an action sequence leaves no viable mass"),
            Self::TooManySequences { limit } => write!(f, "too many sequences (limit {limit})"),
            Self::TooManyPaths { limit } => write!(f, "too many paths (limit {limit})"),
            Self::Inconsistent(m) => write!(f, "inconsistent result: {m}"),
            Self::World(e) => write!(f, "world error: {e}"),
            Self::Macro(e) => write!(f, "macrostate error: {e}"),
            Self::Entropy(e) => write!(f, "entropy error: {e}"),
        }
    }
}

impl Error for MultiscaleError {}

impl From<BoolWorldError> for MultiscaleError {
    fn from(e: BoolWorldError) -> Self {
        Self::World(e)
    }
}
impl From<MacroError> for MultiscaleError {
    fn from(e: MacroError) -> Self {
        Self::Macro(e)
    }
}
impl From<EntropyError> for MultiscaleError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// A labelled, versioned projection `phi: micro -> macro` with dense macro ids `0..k`.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    label: String,
    version: (u32, u32),
    phi: BTreeMap<MicroState, usize>,
    macro_count: usize,
}

impl Scale {
    /// Identity scale: every microstate is its own macro (the micro limit).
    pub fn identity(world: &BoolWorld) -> Self {
        let phi: BTreeMap<_, _> = world
            .state_specs()
            .enumerate()
            .map(|(i, v)| (v.state, i))
            .collect();
        let macro_count = phi.len();
        Self {
            label: "identity".to_string(),
            version: MULTISCALE_SCALE_VERSION,
            phi,
            macro_count,
        }
    }

    /// Trivial scale: a single macro containing every microstate (the coarsest limit).
    pub fn trivial(world: &BoolWorld) -> Self {
        let phi: BTreeMap<_, _> = world.state_specs().map(|v| (v.state, 0)).collect();
        Self {
            label: "trivial".to_string(),
            version: MULTISCALE_SCALE_VERSION,
            phi,
            macro_count: 1,
        }
    }

    /// Arbitrary partition: must cover exactly the world's microstates with dense ids.
    pub fn from_partition(
        label: &str,
        world: &BoolWorld,
        phi: BTreeMap<MicroState, usize>,
    ) -> Result<Self, MultiscaleError> {
        if label.is_empty() {
            return Err(MultiscaleError::InvalidScale("empty label"));
        }
        let covers = phi.len() == world.state_specs().count()
            && world.state_specs().all(|v| phi.contains_key(&v.state));
        if !covers {
            return Err(MultiscaleError::InvalidScale(
                "partition must cover exactly the microstates",
            ));
        }
        let macro_count = phi.values().max().map_or(0, |m| m + 1);
        if macro_count == 0 || (0..macro_count).any(|m| !phi.values().any(|v| *v == m)) {
            return Err(MultiscaleError::InvalidScale("macro ids are not dense"));
        }
        Ok(Self {
            label: label.to_string(),
            version: MULTISCALE_SCALE_VERSION,
            phi,
            macro_count,
        })
    }

    /// Scale label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Scale version `(major, minor)`.
    pub fn version(&self) -> (u32, u32) {
        self.version
    }

    /// Number of macrostates.
    pub fn macro_count(&self) -> usize {
        self.macro_count
    }

    /// Projection `micro -> macro`.
    pub fn phi(&self) -> &BTreeMap<MicroState, usize> {
        &self.phi
    }

    fn tag(&self) -> ScaleTag {
        ScaleTag {
            label: self.label.clone(),
            version: self.version,
            micro_count: self.phi.len(),
            macro_count: self.macro_count,
        }
    }
}

/// Scale identification recorded with every measurement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScaleTag {
    /// Scale label (same shape as the trace's `scale_label`).
    pub label: String,
    /// Scale version `(major, minor)` (same shape as the trace's `scale_version`).
    pub version: (u32, u32),
    /// Number of microstates.
    pub micro_count: usize,
    /// Number of macrostates.
    pub macro_count: usize,
}

/// Intervention policy under which every quantity is computed (always reported).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterventionPolicy {
    /// Uniform prior over all open-loop action-index sequences of length `horizon`
    /// (`k^horizon`, `k` = maximum action count); indices wrap modulo each state's action count.
    OpenLoopUniformSequences {
        /// Horizon.
        horizon: usize,
        /// Number of sequences enumerated.
        sequences: usize,
    },
}

/// Exact evaluation configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MultiscaleConfig {
    /// Initial microstate (must be viable).
    pub initial: MicroState,
    /// Horizon (>= 1).
    pub horizon: usize,
    /// Maximum number of action sequences.
    pub max_sequences: usize,
    /// Maximum number of live paths per sequence.
    pub max_paths: usize,
    /// Tolerance in bits / probability for comparisons.
    pub tolerance: f64,
}

impl MultiscaleConfig {
    /// Default limits for `initial` and `horizon`.
    pub fn new(initial: MicroState, horizon: usize) -> Self {
        Self {
            initial,
            horizon,
            max_sequences: 4_096,
            max_paths: 100_000,
            tolerance: 1e-9,
        }
    }
}

/// Apparent entropy `H` of projected trajectories, in bits. Scale-dependent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrajectoryEntropy {
    /// Bits.
    pub bits: f64,
}

/// Causal control `I(A; phi(S_T))` in bits (uniform open-loop prior). Not empowerment, not EI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CausalControl {
    /// Bits.
    pub bits: f64,
}

/// Viability, defined on microstates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Viability {
    /// Mean raw probability mass remaining viable at the horizon over action sequences.
    pub mean_viable_mass: f64,
    /// `true` if every macro contains only viable or only inviable microstates.
    pub macros_viability_homogeneous: bool,
}

/// Exact predictive loss of the projection (GOG-161 analysis, macro id as observation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictiveError {
    /// `H(micro | macro)` in bits with uniform microstates (information discarded).
    pub information_loss_bits: f64,
    /// Max absolute deviation between micro predictions and the quotient prediction.
    pub quotient_max_abs_deviation: f64,
    /// Whether the projection is Markovian in the GOG-161 sense.
    pub markovian: bool,
}

/// All measures at one scale. Fields are intentionally separate; there is no aggregate.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleMeasures {
    /// Scale recorded with the measurement.
    pub scale: ScaleTag,
    /// Intervention policy used.
    pub policy: InterventionPolicy,
    /// Apparent trajectory entropy.
    pub trajectory_entropy: TrajectoryEntropy,
    /// Causal control.
    pub control: CausalControl,
    /// Viability.
    pub viability: Viability,
    /// Predictive error.
    pub predictive: PredictiveError,
}

type SpecMap<'a> = BTreeMap<MicroState, (bool, &'a [Vec<(MicroState, f64)>])>;

fn entropy(dist: impl Iterator<Item = f64>, tol: f64) -> Result<f64, MultiscaleError> {
    let p: Vec<f64> = dist.collect();
    Ok(shannon_entropy_bits_with_tolerance(&p, tol)?)
}

/// Measures `world` at `scale` (exact, deterministic).
pub fn measure_at_scale(
    world: &BoolWorld,
    scale: &Scale,
    cfg: &MultiscaleConfig,
) -> Result<ScaleMeasures, MultiscaleError> {
    if cfg.horizon == 0 || cfg.max_sequences == 0 || cfg.max_paths == 0 {
        return Err(MultiscaleError::InvalidConfig(
            "horizon, max_sequences and max_paths must be >= 1",
        ));
    }
    if !cfg.tolerance.is_finite() || cfg.tolerance < 0.0 {
        return Err(MultiscaleError::InvalidConfig("invalid tolerance"));
    }
    let specs: SpecMap<'_> = world
        .state_specs()
        .map(|v| (v.state, (v.viable, v.actions)))
        .collect();
    if specs.len() != scale.phi.len() || !specs.keys().all(|s| scale.phi.contains_key(s)) {
        return Err(MultiscaleError::InvalidScale(
            "scale does not match the world's microstates",
        ));
    }
    if !specs.get(&cfg.initial).is_some_and(|(v, _)| *v) {
        return Err(BoolWorldError::InvalidInitialState(cfg.initial).into());
    }
    let k = specs
        .values()
        .map(|(_, a)| a.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let n_seq = u32::try_from(cfg.horizon)
        .ok()
        .and_then(|h| k.checked_pow(h))
        .filter(|n| *n <= cfg.max_sequences)
        .ok_or(MultiscaleError::TooManySequences {
            limit: cfg.max_sequences,
        })?;

    let w = 1.0 / n_seq as f64;
    let mut traj_mix: BTreeMap<Vec<usize>, f64> = BTreeMap::new();
    let mut final_mix: BTreeMap<usize, f64> = BTreeMap::new();
    let mut cond_final = 0.0;
    let mut viable_sum = 0.0;
    for idx in 0..n_seq {
        let mut seq = vec![0usize; cfg.horizon];
        let mut r = idx;
        for slot in seq.iter_mut() {
            *slot = r % k;
            r /= k;
        }
        let (paths, mass) = propagate(&specs, cfg, &seq)?;
        viable_sum += mass;
        let mut finals: BTreeMap<usize, f64> = BTreeMap::new();
        for (path, p) in paths {
            let projected: Vec<usize> = path.iter().map(|s| scale.phi[s]).collect();
            let last = *projected.last().expect("path is never empty");
            let p = p / mass;
            *finals.entry(last).or_insert(0.0) += p;
            *traj_mix.entry(projected).or_insert(0.0) += w * p;
        }
        cond_final += w * entropy(finals.values().copied(), cfg.tolerance)?;
        for (s, p) in finals {
            *final_mix.entry(s).or_insert(0.0) += w * p;
        }
    }
    let h_traj = entropy(traj_mix.values().copied(), cfg.tolerance)?;
    let h_final = entropy(final_mix.values().copied(), cfg.tolerance)?;
    let mut mi = h_final - cond_final;
    if !mi.is_finite() || !h_traj.is_finite() || mi < -cfg.tolerance {
        return Err(MultiscaleError::Inconsistent(
            "I(A; macro) negative or not finite",
        ));
    }
    mi = mi.max(0.0);

    let homogeneous = (0..scale.macro_count).all(|m| {
        let mut flags = scale
            .phi
            .iter()
            .filter(|(_, id)| **id == m)
            .map(|(s, _)| specs[s].0);
        let first = flags.next();
        flags.all(|f| Some(f) == first)
    });

    let phi = scale.phi.clone();
    let observe = |s: MicroState| u8::try_from(phi[&s]).unwrap_or(u8::MAX);
    let macro_report = analyze_partition(
        world,
        &observe,
        &scale.phi,
        cfg.horizon,
        MacroLimits {
            tolerance: cfg.tolerance,
            ..MacroLimits::default()
        },
    )?;

    Ok(ScaleMeasures {
        scale: scale.tag(),
        policy: InterventionPolicy::OpenLoopUniformSequences {
            horizon: cfg.horizon,
            sequences: n_seq,
        },
        trajectory_entropy: TrajectoryEntropy { bits: h_traj },
        control: CausalControl { bits: mi },
        viability: Viability {
            mean_viable_mass: viable_sum * w,
            macros_viability_homogeneous: homogeneous,
        },
        predictive: PredictiveError {
            information_loss_bits: macro_report.information_loss_bits,
            quotient_max_abs_deviation: macro_report.quotient_max_abs_deviation,
            markovian: macro_report.markovian,
        },
    })
}

/// Micro paths surviving viability for one action sequence, with their raw total mass.
type PathMass = (BTreeMap<Vec<MicroState>, f64>, f64);

fn propagate(
    specs: &SpecMap<'_>,
    cfg: &MultiscaleConfig,
    seq: &[usize],
) -> Result<PathMass, MultiscaleError> {
    let mut paths: BTreeMap<Vec<MicroState>, f64> = BTreeMap::new();
    paths.insert(vec![cfg.initial], 1.0);
    for &a in seq {
        let mut next: BTreeMap<Vec<MicroState>, f64> = BTreeMap::new();
        for (path, p) in &paths {
            let last = *path.last().expect("path is never empty");
            let (_, acts) = specs.get(&last).ok_or(BoolWorldError::InvalidWorld {
                reason: "transition to undeclared microstate",
            })?;
            let dests: Vec<(MicroState, f64)> = if acts.is_empty() {
                vec![(last, 1.0)] // absorbing
            } else {
                acts[a % acts.len()].clone()
            };
            for (d, q) in dests {
                let viable = specs
                    .get(&d)
                    .ok_or(BoolWorldError::InvalidWorld {
                        reason: "transition to undeclared microstate",
                    })?
                    .0;
                if !viable || q <= 0.0 {
                    continue;
                }
                if next.len() >= cfg.max_paths {
                    return Err(MultiscaleError::TooManyPaths {
                        limit: cfg.max_paths,
                    });
                }
                let mut np = path.clone();
                np.push(d);
                *next.entry(np).or_insert(0.0) += p * q;
            }
        }
        paths = next;
    }
    let mass: f64 = paths.values().sum();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(MultiscaleError::NoViableMass);
    }
    Ok((paths, mass))
}

/// How two scales relate as partitions of the same microstates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleRelation {
    /// Same partition (macro ids may be renamed).
    Identical,
    /// The macro-side scale is strictly coarser: it is a function of the micro-side scale.
    MacroCoarsensMicro,
    /// The macro-side scale is strictly finer than the micro-side one.
    MacroRefinesMicro,
    /// Neither is a function of the other.
    Incomparable,
}

fn is_function_of(a: &Scale, b: &Scale) -> bool {
    // True if `b` is a function of `a` (a refines b).
    let mut map: BTreeMap<usize, usize> = BTreeMap::new();
    a.phi
        .iter()
        .all(|(s, ia)| *map.entry(*ia).or_insert(b.phi[s]) == b.phi[s])
}

/// Classifies the relation between a micro-side and a macro-side scale.
pub fn relate_scales(micro: &Scale, macro_side: &Scale) -> Result<ScaleRelation, MultiscaleError> {
    if micro.phi.len() != macro_side.phi.len()
        || !micro.phi.keys().all(|s| macro_side.phi.contains_key(s))
    {
        return Err(MultiscaleError::InvalidScale(
            "scales are defined over different microstates",
        ));
    }
    let coarsens = is_function_of(micro, macro_side);
    let refines = is_function_of(macro_side, micro);
    Ok(match (coarsens, refines) {
        (true, true) => ScaleRelation::Identical,
        (true, false) => ScaleRelation::MacroCoarsensMicro,
        (false, true) => ScaleRelation::MacroRefinesMicro,
        (false, false) => ScaleRelation::Incomparable,
    })
}

/// Per-quantity differences `macro - micro`. Never summed: units/meaning differ.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleDeltas {
    /// Change of apparent trajectory entropy (bits).
    pub trajectory_entropy_bits: f64,
    /// Change of causal control (bits).
    pub control_bits: f64,
    /// Change of mean viable mass (probability).
    pub viable_mass: f64,
    /// Change of information discarded `H(micro | macro)` (bits).
    pub information_loss_bits: f64,
    /// Change of the quotient prediction deviation (probability).
    pub quotient_deviation: f64,
}

/// Verdict about what changing scale did (reports control separately from entropy).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleVerdict {
    /// Apparent entropy changed while causal control did not (within tolerance).
    EntropyChangedWithoutControlChange {
        /// Entropy change in bits (macro - micro).
        trajectory_entropy_delta_bits: f64,
    },
    /// Causal control changed (negative when coarsening discards controllable distinctions).
    ControlChanged {
        /// Control change in bits (macro - micro).
        control_delta_bits: f64,
    },
    /// Neither changed within tolerance.
    Unchanged,
}

/// Exact comparison of the same world at two scales.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleComparison {
    /// Measures at the micro-side scale.
    pub micro: ScaleMeasures,
    /// Measures at the macro-side scale.
    pub macro_side: ScaleMeasures,
    /// Partition relation between both scales.
    pub relation: ScaleRelation,
    /// Intervention policy shared by both sides.
    pub policy: InterventionPolicy,
    /// Separate per-quantity deltas.
    pub deltas: ScaleDeltas,
    /// Tolerance used.
    pub tolerance: f64,
    /// Verdict.
    pub verdict: ScaleVerdict,
}

/// Compares `micro` and `macro_side` on `world` under one shared intervention policy.
pub fn compare_scales(
    world: &BoolWorld,
    micro: &Scale,
    macro_side: &Scale,
    cfg: &MultiscaleConfig,
) -> Result<ScaleComparison, MultiscaleError> {
    let relation = relate_scales(micro, macro_side)?;
    let a = measure_at_scale(world, micro, cfg)?;
    let b = measure_at_scale(world, macro_side, cfg)?;
    let deltas = ScaleDeltas {
        trajectory_entropy_bits: b.trajectory_entropy.bits - a.trajectory_entropy.bits,
        control_bits: b.control.bits - a.control.bits,
        viable_mass: b.viability.mean_viable_mass - a.viability.mean_viable_mass,
        information_loss_bits: b.predictive.information_loss_bits
            - a.predictive.information_loss_bits,
        quotient_deviation: b.predictive.quotient_max_abs_deviation
            - a.predictive.quotient_max_abs_deviation,
    };
    let tol = cfg.tolerance;
    let verdict = if deltas.control_bits.abs() > tol {
        ScaleVerdict::ControlChanged {
            control_delta_bits: deltas.control_bits,
        }
    } else if deltas.trajectory_entropy_bits.abs() > tol {
        ScaleVerdict::EntropyChangedWithoutControlChange {
            trajectory_entropy_delta_bits: deltas.trajectory_entropy_bits,
        }
    } else {
        ScaleVerdict::Unchanged
    };
    Ok(ScaleComparison {
        policy: a.policy,
        micro: a,
        macro_side: b,
        relation,
        deltas,
        tolerance: tol,
        verdict,
    })
}

#[cfg(test)]
mod multiscale_tests {
    use super::*;
    use crate::entropy_controls::add_exogenous_noise;

    const NOISE_BIT: u8 = 3;

    /// Fork world: state 0 chooses 1 or 2; both return to 0.
    fn fork() -> BoolWorld {
        let s = MicroState;
        BoolWorld::new(162)
            .with_state(s(0), true, vec![vec![(s(1), 1.0)], vec![(s(2), 1.0)]])
            .unwrap()
            .with_state(s(1), true, vec![vec![(s(0), 1.0)], vec![(s(0), 1.0)]])
            .unwrap()
            .with_state(s(2), true, vec![vec![(s(0), 1.0)], vec![(s(0), 1.0)]])
            .unwrap()
    }

    fn drop_noise(world: &BoolWorld) -> Scale {
        let mask = 1u8 << NOISE_BIT;
        let mut ids: BTreeMap<u8, usize> = BTreeMap::new();
        let mut phi = BTreeMap::new();
        for v in world.state_specs() {
            let core = v.state.0 & !mask;
            let next = ids.len();
            let id = *ids.entry(core).or_insert(next);
            phi.insert(v.state, id);
        }
        Scale::from_partition("drop_noise_bit3", world, phi).unwrap()
    }

    #[test]
    fn multiscale_changing_scale_changes_h_not_control() {
        let noisy = add_exogenous_noise(&fork(), NOISE_BIT).unwrap();
        let cfg = MultiscaleConfig::new(MicroState(0), 1);
        let cmp =
            compare_scales(&noisy, &Scale::identity(&noisy), &drop_noise(&noisy), &cfg).unwrap();
        assert_eq!(cmp.relation, ScaleRelation::MacroCoarsensMicro);
        // Micro: 4 equiprobable paths -> 2 bits; control I = 2 - 1 = 1 bit.
        assert!((cmp.micro.trajectory_entropy.bits - 2.0).abs() < 1e-9);
        assert!((cmp.micro.control.bits - 1.0).abs() < 1e-9);
        // Macro (noise bit projected out): 1 bit of H, same 1 bit of control.
        assert!((cmp.macro_side.trajectory_entropy.bits - 1.0).abs() < 1e-9);
        assert!((cmp.macro_side.control.bits - 1.0).abs() < 1e-9);
        assert!((cmp.deltas.trajectory_entropy_bits + 1.0).abs() < 1e-9);
        assert!(cmp.deltas.control_bits.abs() < 1e-9);
        assert!(matches!(
            cmp.verdict,
            ScaleVerdict::EntropyChangedWithoutControlChange { .. }
        ));
        // The coarse scale predicts exactly even though it discards 1 bit of micro detail.
        assert!(cmp.macro_side.predictive.markovian);
        assert!(cmp.macro_side.predictive.quotient_max_abs_deviation < 1e-9);
        assert!((cmp.macro_side.predictive.information_loss_bits - 1.0).abs() < 1e-9);
        assert_eq!(cmp.micro.predictive.information_loss_bits, 0.0);
        // Both viability numbers are micro-defined and equal.
        assert!(cmp.deltas.viable_mass.abs() < 1e-12);
        assert_eq!(cmp.policy, cmp.micro.policy);
    }

    #[test]
    fn multiscale_scale_is_recorded_in_measures() {
        let noisy = add_exogenous_noise(&fork(), NOISE_BIT).unwrap();
        let cfg = MultiscaleConfig::new(MicroState(0), 1);
        let m = measure_at_scale(&noisy, &drop_noise(&noisy), &cfg).unwrap();
        assert_eq!(m.scale.label, "drop_noise_bit3");
        assert_eq!(m.scale.version, MULTISCALE_SCALE_VERSION);
        assert_eq!((m.scale.micro_count, m.scale.macro_count), (6, 3));
        assert_eq!(
            m.policy,
            InterventionPolicy::OpenLoopUniformSequences {
                horizon: 1,
                sequences: 2
            }
        );
    }

    #[test]
    fn multiscale_ablation_trivial_and_identity_limits() {
        let w = fork();
        let cfg = MultiscaleConfig::new(MicroState(0), 1);
        let id = measure_at_scale(&w, &Scale::identity(&w), &cfg).unwrap();
        let tr = measure_at_scale(&w, &Scale::trivial(&w), &cfg).unwrap();
        // Identity: full control 1 bit, H 1 bit, no information lost, exact prediction.
        assert!((id.control.bits - 1.0).abs() < 1e-9);
        assert!((id.trajectory_entropy.bits - 1.0).abs() < 1e-9);
        assert_eq!(id.predictive.information_loss_bits, 0.0);
        assert!(id.predictive.markovian);
        // Trivial: one macro -> zero H, zero control; loses log2(3) bits; trivially Markovian.
        assert_eq!(tr.scale.macro_count, 1);
        assert_eq!(tr.trajectory_entropy.bits, 0.0);
        assert_eq!(tr.control.bits, 0.0);
        assert!((tr.predictive.information_loss_bits - 3f64.log2()).abs() < 1e-9);
        assert!(tr.predictive.markovian);
        let cmp = compare_scales(&w, &Scale::identity(&w), &Scale::trivial(&w), &cfg).unwrap();
        assert_eq!(cmp.relation, ScaleRelation::MacroCoarsensMicro);
        assert!(
            matches!(cmp.verdict, ScaleVerdict::ControlChanged { control_delta_bits } if (control_delta_bits + 1.0).abs() < 1e-9)
        );
        // Control never exceeds the identity limit at any coarsening, and relation is symmetric.
        let rev = relate_scales(&Scale::trivial(&w), &Scale::identity(&w)).unwrap();
        assert_eq!(rev, ScaleRelation::MacroRefinesMicro);
        assert_eq!(
            relate_scales(&Scale::identity(&w), &Scale::identity(&w)).unwrap(),
            ScaleRelation::Identical
        );
    }

    #[test]
    fn multiscale_rejects_invalid_scales_and_configs() {
        let w = fork();
        let cfg = MultiscaleConfig::new(MicroState(0), 1);
        let mut phi = BTreeMap::new();
        phi.insert(MicroState(0), 0);
        assert!(Scale::from_partition("x", &w, phi).is_err());
        let sparse: BTreeMap<_, _> = [(0u8, 0usize), (1, 2), (2, 2)]
            .into_iter()
            .map(|(s, m)| (MicroState(s), m))
            .collect();
        assert!(Scale::from_partition("x", &w, sparse).is_err());
        assert!(
            measure_at_scale(
                &w,
                &Scale::identity(&w),
                &MultiscaleConfig { horizon: 0, ..cfg }
            )
            .is_err()
        );
        let other = add_exogenous_noise(&w, NOISE_BIT).unwrap();
        assert!(measure_at_scale(&other, &Scale::identity(&w), &cfg).is_err());
        let tight = MultiscaleConfig {
            max_sequences: 1,
            ..cfg
        };
        assert!(matches!(
            measure_at_scale(&w, &Scale::identity(&w), &tight),
            Err(MultiscaleError::TooManySequences { .. })
        ));
    }
}
