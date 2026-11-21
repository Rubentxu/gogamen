//! Memory entropy-inflation guard (GOG-172 / DEM-019).
//!
//! Oracle that keeps two numbers apart: the entropy `H_mem` of what an agent *writes* into the
//! ambient memory surface ([`MarkBoard`], GOG-170/171) and the *world control* score
//! `I(A0; Z)` (bias-corrected, [`control_bits_from_pairs`], GOG-180) measured over the world.
//!
//! Writers: honest (writes the control estimate), random bits, duplicated marks, and a
//! no-causal-effect predicate appended to the observed state. None of them acts on the world, so
//! the `(first action, next state)` pairs are generated from streams independent of the writer:
//! the control score of random-bit and duplicated writers is *identical* to the honest one, while
//! `H_mem` of random bits is high. Predicates with no causal effect inflate the plug-in
//! `H(Z)` (and, when the world has no control, plug-in control through finite-sample bias); the
//! corrected score does not follow.
//!
//! Scope: small deterministic fixtures. It shows that these inflation routes are detectable; it
//! makes no claim about emergence or about any other inflation route.

use gogamen_core::{
    BinningVersion, ControlBitsConfig, DeterministicRng, OwnerId, RngSource, shannon_entropy_bits,
};

use crate::memory_world::{Evaporation, MarkBoard, MarkCondition, MemoryWorldError};

const N_ACTIONS: usize = 4;
/// Number of cells of the fixture surface (matches the MemoryWorld board).
const CELLS: usize = crate::memory_world::MW_W * crate::memory_world::MW_H;
/// Max value of a mark (bits with four actions).
const MARK_MAX: f64 = 2.0;
/// Histogram bins used to measure the entropy of written marks.
const MARK_BINS: usize = 8;

/// Scope note carried by every report.
pub const MEMORY_ENTROPY_GUARD_NOTE: &str = "Memory entropy H_mem and world control I(A0;Z) are \
reported separately; fixtures only, no emergence claim.";

/// World dynamics of the fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardWorld {
    /// Next state `z = a`: four actions fully determine four outcomes (2 bits of control).
    Causal,
    /// Next state uniform and independent of the action (0 bits of control).
    Null,
}

/// What the agent writes into memory (and, for the predicate, appends to the observed state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardWriter {
    /// Writes its own control estimate (constant mark) once per cell.
    Honest,
    /// Writes an independent uniform random value in every cell.
    RandomBits,
    /// Writes the same mark many times (every cell, `DUPLICATES` rounds).
    Duplicated,
    /// Honest marks, plus `predicate_bits` random predicates with no effect on the world,
    /// appended to the observed next state.
    NoEffectPredicate,
}

/// Duplicate rounds of [`GuardWriter::Duplicated`].
pub const DUPLICATES: u64 = 5;

/// Fixture configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardConfig {
    /// `(a0, z)` samples.
    pub samples: usize,
    /// Seed of every stream.
    pub seed: u64,
    /// Number of no-effect predicate bits (used by [`GuardWriter::NoEffectPredicate`]).
    pub predicate_bits: u32,
}

impl Default for GuardConfig {
    fn default() -> Self {
        Self {
            samples: 400,
            seed: 172,
            predicate_bits: 3,
        }
    }
}

/// Report with memory entropy and world control kept in separate groups.
#[derive(Debug, Clone, PartialEq)]
pub struct GuardReport {
    /// Fixture world.
    pub world: GuardWorld,
    /// Fixture writer.
    pub writer: GuardWriter,
    /// Shannon entropy (bits) of the histogram of values stored on the surface.
    pub memory_entropy_bits: f64,
    /// Cells holding a value.
    pub memory_cells: usize,
    /// Write operations performed (duplicates count here, not in `memory_entropy_bits`).
    pub memory_writes: u64,
    /// Plug-in `H(Z)` of the observed next state (world side).
    pub world_h_z_plugin: f64,
    /// Plug-in `I(A0; Z)` (biased upwards).
    pub world_control_plugin: f64,
    /// Bias-corrected `I(A0; Z)`: the world control score.
    pub world_control_corrected: f64,
    /// Upper permutation-null quantile of the corrected score.
    pub world_null_upper_bits: f64,
}

fn xor_seed(seed: u64, salt: u64) -> u64 {
    seed ^ salt.wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

/// World pairs `(a0, z)`; `extra` predicates (if any) are folded in as `z * 2^k + p`.
/// Drawn from streams that do not depend on the writer.
fn world_pairs(world: GuardWorld, cfg: &GuardConfig, predicate_bits: u32) -> Vec<(usize, usize)> {
    let mut act = DeterministicRng::new(xor_seed(cfg.seed, 1));
    let mut env = DeterministicRng::new(xor_seed(cfg.seed, 2));
    let mut pred = DeterministicRng::new(xor_seed(cfg.seed, 3));
    (0..cfg.samples)
        .map(|_| {
            let a = act.next_usize_unbiased(N_ACTIONS);
            let noise = env.next_usize_unbiased(N_ACTIONS);
            let z = match world {
                GuardWorld::Causal => a,
                GuardWorld::Null => noise,
            };
            let p = if predicate_bits == 0 {
                0
            } else {
                pred.next_usize_unbiased(1usize << predicate_bits)
            };
            (a, (z << predicate_bits) | p)
        })
        .collect()
}

fn hist_entropy(values: &[f64]) -> Result<f64, MemoryWorldError> {
    if values.is_empty() {
        return Ok(0.0);
    }
    let mut counts = [0usize; MARK_BINS];
    for v in values {
        let b = ((v / MARK_MAX) * MARK_BINS as f64) as usize;
        counts[b.min(MARK_BINS - 1)] += 1;
    }
    let n = values.len() as f64;
    let probs: Vec<f64> = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| c as f64 / n)
        .collect();
    shannon_entropy_bits(&probs).map_err(|e| MemoryWorldError::Control(e.to_string()))
}

fn control_cfg(cfg: &GuardConfig) -> ControlBitsConfig {
    ControlBitsConfig {
        resamples: 200,
        confidence: 0.95,
        seed: cfg.seed,
    }
}

/// Runs one deterministic fixture and reports memory entropy and world control separately.
pub fn run_guard_fixture(
    world: GuardWorld,
    writer: GuardWriter,
    cfg: &GuardConfig,
) -> Result<GuardReport, MemoryWorldError> {
    if cfg.samples < 8 {
        return Err(MemoryWorldError::InvalidConfig("samples must be >= 8"));
    }
    let agent = OwnerId::new(1);
    let mut board = MarkBoard::new(
        &[agent],
        MarkCondition::Measurement,
        Evaporation::Stale,
        cfg.seed,
    )?;

    let pbits = if writer == GuardWriter::NoEffectPredicate {
        cfg.predicate_bits
    } else {
        0
    };
    let pairs = world_pairs(world, cfg, pbits);
    // Honest mark = the control estimate over the physical pairs (no predicates).
    let honest_pairs = world_pairs(world, cfg, 0);
    let honest = gogamen_core::control_bits_from_pairs(
        &honest_pairs,
        "guard_physical",
        BinningVersion::V1_0,
        &control_cfg(cfg),
    )
    .map_err(|e| MemoryWorldError::Control(e.to_string()))?;
    let mark = honest.control_corrected.clamp(0.0, MARK_MAX);

    let mut wrng = DeterministicRng::new(xor_seed(cfg.seed, 4));
    let mut tick = 0u64;
    let rounds = if writer == GuardWriter::Duplicated {
        DUPLICATES
    } else {
        1
    };
    for _ in 0..rounds {
        for cell in 0..CELLS {
            tick += 1;
            let v = match writer {
                GuardWriter::RandomBits => wrng.next_f64() * MARK_MAX * 0.999_999,
                _ => mark,
            };
            board.write(agent, cell, v, tick)?;
        }
    }
    let values = board.sorted_values(agent)?;
    let memory_entropy_bits = hist_entropy(&values)?;

    let est = gogamen_core::control_bits_from_pairs(
        &pairs,
        "guard_world",
        BinningVersion::V1_0,
        &control_cfg(cfg),
    )
    .map_err(|e| MemoryWorldError::Control(e.to_string()))?;

    Ok(GuardReport {
        world,
        writer,
        memory_entropy_bits,
        memory_cells: values.len(),
        memory_writes: board.writes(),
        world_h_z_plugin: est.h_z_plugin,
        world_control_plugin: est.control_plugin,
        world_control_corrected: est.control_corrected,
        world_null_upper_bits: est.null_upper_bits,
    })
}

/// Auxiliary (memory-derived) channel appended to the physical observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuxChannel {
    /// Scalar mark that is a deterministic function of the action (`a >= 2` -> high).
    Useful,
    /// Scalar mark drawn at random, independent of the action.
    Random,
}

/// Control over the physical observation only vs. with the normalized auxiliary channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AuxProjectionReport {
    /// Corrected control over the physical state `z = a % 2` alone (also what projecting the
    /// auxiliary channel away yields).
    pub control_physical_only: f64,
    /// Corrected control over `(z, normalize(aux))`.
    pub control_with_aux: f64,
}

/// Normalizes a raw mark in `[0, MARK_MAX]` to `[0, 1]` and projects to one bit (threshold 0.5).
pub fn normalize_project_mark(raw: f64) -> usize {
    usize::from((raw / MARK_MAX).clamp(0.0, 1.0) >= 0.5)
}

/// Measures whether normalizing/projecting an auxiliary channel preserves a useful signal and
/// does not manufacture control from a useless one.
pub fn run_aux_projection(
    aux: AuxChannel,
    cfg: &GuardConfig,
) -> Result<AuxProjectionReport, MemoryWorldError> {
    if cfg.samples < 8 {
        return Err(MemoryWorldError::InvalidConfig("samples must be >= 8"));
    }
    let mut act = DeterministicRng::new(xor_seed(cfg.seed, 11));
    let mut rnd = DeterministicRng::new(xor_seed(cfg.seed, 12));
    let mut phys = Vec::with_capacity(cfg.samples);
    let mut both = Vec::with_capacity(cfg.samples);
    for _ in 0..cfg.samples {
        let a = act.next_usize_unbiased(N_ACTIONS);
        let z = a % 2;
        let raw = match aux {
            AuxChannel::Useful => {
                if a >= 2 {
                    1.5
                } else {
                    0.5
                }
            }
            AuxChannel::Random => rnd.next_f64() * MARK_MAX * 0.999_999,
        };
        phys.push((a, z));
        both.push((a, (z, normalize_project_mark(raw))));
    }
    let c = control_cfg(cfg);
    let ctl = |r: Result<gogamen_core::ControlBitsEstimate, gogamen_core::ControlBitsError>| {
        r.map(|e| e.control_corrected)
            .map_err(|e| MemoryWorldError::Control(e.to_string()))
    };
    Ok(AuxProjectionReport {
        control_physical_only: ctl(gogamen_core::control_bits_from_pairs(
            &phys,
            "phys",
            BinningVersion::V1_0,
            &c,
        ))?,
        control_with_aux: ctl(gogamen_core::control_bits_from_pairs(
            &both,
            "phys+aux",
            BinningVersion::V1_0,
            &c,
        ))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(w: GuardWorld, k: GuardWriter) -> GuardReport {
        run_guard_fixture(w, k, &GuardConfig::default()).expect("fixture")
    }

    #[test]
    fn memory_entropy_guard_random_control_high_h_null_control() {
        // AC1: in a world where actions have no effect, random bits give high H_mem, ~0 control.
        let r = run(GuardWorld::Null, GuardWriter::RandomBits);
        let honest = run(GuardWorld::Null, GuardWriter::Honest);
        eprintln!("AC1 null/random {r:?}\nAC1 null/honest {honest:?}");
        assert!(
            r.memory_entropy_bits > 2.5,
            "H_mem={}",
            r.memory_entropy_bits
        );
        assert!(honest.memory_entropy_bits < 0.01);
        assert!(
            r.world_control_corrected.abs() < 0.1,
            "{}",
            r.world_control_corrected
        );
        assert!(r.world_control_corrected <= r.world_null_upper_bits + 0.05);

        // Causal world: writing random bits leaves control exactly unchanged.
        let rc = run(GuardWorld::Causal, GuardWriter::RandomBits);
        let hc = run(GuardWorld::Causal, GuardWriter::Honest);
        eprintln!("AC1 causal/random {rc:?}\nAC1 causal/honest {hc:?}");
        assert!(rc.memory_entropy_bits > hc.memory_entropy_bits + 2.0);
        assert_eq!(rc.world_control_corrected, hc.world_control_corrected);
        assert_eq!(rc.world_control_plugin, hc.world_control_plugin);
        assert!(rc.world_control_corrected > 1.8);

        // Duplicated marks: more writes, no extra H_mem, no extra control.
        let d = run(GuardWorld::Causal, GuardWriter::Duplicated);
        eprintln!("AC1 causal/dup {d:?}");
        assert_eq!(d.memory_writes, DUPLICATES * CELLS as u64);
        assert_eq!(d.memory_cells, CELLS);
        assert_eq!(d.memory_entropy_bits, hc.memory_entropy_bits);
        assert_eq!(d.world_control_corrected, hc.world_control_corrected);

        // No-effect predicates inflate plug-in numbers, not the corrected score.
        let p = run(GuardWorld::Causal, GuardWriter::NoEffectPredicate);
        eprintln!("AC1 causal/predicate {p:?}");
        assert!(p.world_h_z_plugin > hc.world_h_z_plugin + 2.0);
        assert!(
            p.world_control_corrected <= hc.world_control_corrected + 0.1,
            "corrected {} vs {}",
            p.world_control_corrected,
            hc.world_control_corrected
        );
        let pn = run(GuardWorld::Null, GuardWriter::NoEffectPredicate);
        eprintln!("AC1 null/predicate {pn:?}");
        // In the null world the plug-in control IS inflated by the predicates; corrected is not.
        assert!(pn.world_control_plugin > honest.world_control_plugin + 0.1);
        assert!(pn.world_control_corrected.abs() < 0.15);
        assert!(pn.world_control_corrected < pn.world_control_plugin - 0.1);
    }

    #[test]
    fn memory_entropy_guard_projection_keeps_useful_signal() {
        // AC2.
        let cfg = GuardConfig::default();
        let useful = run_aux_projection(AuxChannel::Useful, &cfg).unwrap();
        let random = run_aux_projection(AuxChannel::Random, &cfg).unwrap();
        eprintln!("AC2 useful {useful:?}\nAC2 random {random:?}");
        assert!((useful.control_physical_only - 1.0).abs() < 0.1);
        assert!(useful.control_with_aux > useful.control_physical_only + 0.8);
        assert!((random.control_with_aux - random.control_physical_only).abs() < 0.1);
        assert_eq!(normalize_project_mark(0.5), 0);
        assert_eq!(normalize_project_mark(1.5), 1);
        assert_eq!(normalize_project_mark(99.0), 1);
        assert_eq!(normalize_project_mark(-3.0), 0);
    }

    #[test]
    fn memory_entropy_guard_reports_separately_and_deterministically() {
        // AC3: two runs are bit-identical; the report holds two disjoint metric groups.
        let a = run(GuardWorld::Causal, GuardWriter::RandomBits);
        let b = run(GuardWorld::Causal, GuardWriter::RandomBits);
        assert_eq!(a, b);
        assert!(MEMORY_ENTROPY_GUARD_NOTE.contains("separately"));
        let hc = run(GuardWorld::Causal, GuardWriter::Honest);
        // Same world side, different memory side.
        assert_eq!(a.world_control_corrected, hc.world_control_corrected);
        assert_ne!(a.memory_entropy_bits, hc.memory_entropy_bits);
        // Fixed expected values for the honest causal fixture (2 bits, constant mark).
        assert!((hc.world_control_corrected - 2.0).abs() < 0.05);
        assert_eq!(hc.memory_entropy_bits, 0.0);
    }
}
