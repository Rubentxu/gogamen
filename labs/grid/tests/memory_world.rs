//! GOG-171 (DEM-018) MemoryWorld decisive tests. Hypotheses and decision rule are preregistered in
//! the module docs of `gogamen_lab_grid::memory_world`; assertions here check controls and
//! determinism, never that an effect exists. Effects/nulls are reported (run with `--nocapture`).

use gogamen_core::{MemoryViewError, OwnerId};
use gogamen_lab_grid::{
    Evaporation, MEMORY_WORLD_NOTE, MarkBoard, MarkCondition, MemoryWorldConfig, MemoryWorldError,
    Verdict, run_memory_world, run_variant,
};

const A: OwnerId = OwnerId::new(1);
const B: OwnerId = OwnerId::new(2);

fn evap() -> Evaporation {
    Evaporation::Evaporating {
        ttl: 90,
        decay: 0.98,
    }
}

#[test]
fn memory_world_three_paired_conditions_have_controlled_cost_ttl_and_access() {
    let cfg = MemoryWorldConfig {
        warm_episodes: 2,
        eval_episodes: 4,
        ..MemoryWorldConfig::default()
    };
    let v = cfg.variants();
    // Conditions 0..3 = no marks / measurement / permuted: same budget, TTL and decay.
    assert_eq!(v[0].rollouts, v[1].rollouts);
    assert_eq!(v[1].rollouts, v[2].rollouts);
    assert_eq!(v[0].evaporation, v[1].evaporation);
    assert_eq!(v[1].evaporation, v[2].evaporation);
    let runs: Vec<_> = v[..3]
        .iter()
        .map(|x| run_variant(&cfg, x).unwrap())
        .collect();
    for r in &runs {
        // Every decision pays exactly one estimation and leaves exactly one mark, in all three.
        assert_eq!(
            r.cost.estimation,
            r.writes * cfg.est_walkers as u64,
            "{}",
            r.variant.label
        );
        assert!(r.writes > 0);
    }

    // Same values written -> same multiset; permuted reads a different cell (derangement).
    let mut m = MarkBoard::new(&[A, B], MarkCondition::Measurement, evap(), 9).unwrap();
    let mut p = MarkBoard::new(&[A, B], MarkCondition::Permuted, evap(), 9).unwrap();
    let mut n = MarkBoard::new(&[A, B], MarkCondition::NoMarks, evap(), 9).unwrap();
    for c in 0..20 {
        let bits = (c as f64) / 10.0;
        for b in [&mut m, &mut p, &mut n] {
            b.write(A, c, bits, 5).unwrap();
        }
    }
    assert_eq!(m.sorted_values(A).unwrap(), p.sorted_values(A).unwrap());
    assert!((0..20).all(|c| n.read(B, c, 5).unwrap() == 0.0));
    let differing = (0..20)
        .filter(|&c| (m.read(B, c, 5).unwrap() - p.read(B, c, 5).unwrap()).abs() > 1e-12)
        .count();
    assert!(
        differing >= 15,
        "permuted reads must mostly differ, got {differing}"
    );

    // Access is controlled: an agent without grant can neither read nor write.
    let outsider = OwnerId::new(99);
    assert!(matches!(
        m.write(outsider, 0, 1.0, 6),
        Err(MemoryWorldError::Surface(
            MemoryViewError::SurfaceWriteDenied { .. }
        ))
    ));
    assert!(matches!(
        m.read(outsider, 0, 6),
        Err(MemoryWorldError::Surface(
            MemoryViewError::SurfaceReadDenied { .. }
        ))
    ));
}

#[test]
fn memory_world_tenth_reader_with_marks_vs_without_and_full_budget_reports_null_honestly() {
    let cfg = MemoryWorldConfig::default();
    let rep = run_memory_world(&cfg).unwrap();
    assert_eq!(rep, run_memory_world(&cfg).unwrap(), "deterministic");
    let (none, meas, perm, full, _) = (
        &rep.runs[0],
        &rep.runs[1],
        &rep.runs[2],
        &rep.runs[3],
        &rep.runs[4],
    );
    assert!(full.variant.rollouts >= 10 * none.variant.rollouts - 10 * 3);
    assert!(
        full.cost.planning > none.cost.planning,
        "full budget must cost more"
    );
    for r in &rep.runs {
        println!(
            "{:28} A: success={:.3} steps={:.1} | B: success={:.3} steps={:.1} | est={} plan={} writes={}",
            r.variant.label,
            gogamen_lab_grid::VariantRun::success_rate(&r.phase_a),
            gogamen_lab_grid::VariantRun::mean_steps(&r.phase_a),
            gogamen_lab_grid::VariantRun::success_rate(&r.phase_b),
            gogamen_lab_grid::VariantRun::mean_steps(&r.phase_b),
            r.cost.estimation,
            r.cost.planning,
            r.writes
        );
    }
    println!("H1a marks-none      : {:?}", rep.marks_vs_none);
    println!("H1b marks-permuted  : {:?}", rep.marks_vs_permuted);
    println!(
        "H2  full-marks      : {:?} gap_recovered={:?}",
        rep.full_vs_marks, rep.gap_recovered
    );
    println!(
        "H1 supported = {} (null/harm are valid outcomes)",
        rep.h1_supported
    );
    assert_eq!(
        rep.h1_supported,
        rep.marks_vs_none.verdict == Verdict::Effect
            && rep.marks_vs_permuted.verdict == Verdict::Effect
    );
    let _ = (meas, perm);
}

#[test]
fn memory_world_rules_change_measures_stale_damage_against_evaporation() {
    // Mechanism: after a long gap a stale surface still serves the old mark; evaporation does not.
    let mut stale =
        MarkBoard::new(&[A], MarkCondition::Measurement, Evaporation::Stale, 1).unwrap();
    let mut ev = MarkBoard::new(&[A], MarkCondition::Measurement, evap(), 1).unwrap();
    for b in [&mut stale, &mut ev] {
        b.write(A, 4, 2.0, 1).unwrap();
    }
    ev.expire(500);
    stale.expire(500);
    assert!(
        (stale.read(A, 4, 500).unwrap() - 1.0).abs() < 1e-12,
        "stale mark persists"
    );
    assert_eq!(ev.read(A, 4, 500).unwrap(), 0.0, "evaporated mark is gone");
    // Decay shrinks the visible value before TTL.
    let mut ev2 = MarkBoard::new(&[A], MarkCondition::Measurement, evap(), 1).unwrap();
    ev2.write(A, 4, 2.0, 1).unwrap();
    assert!(ev2.read(A, 4, 40).unwrap() < ev2.read(A, 4, 2).unwrap());

    // Behavioural damage after the R1 -> R2 change, reported with the preregistered rule.
    let rep = run_memory_world(&MemoryWorldConfig::default()).unwrap();
    println!(
        "damage evaporating (none - marks, phase B): {:?}",
        rep.damage_evaporating
    );
    println!(
        "damage stale       (none - marks, phase B): {:?}",
        rep.damage_stale
    );
    println!(
        "H3 evaporating - stale                    : {:?}",
        rep.evap_vs_stale
    );
    assert!(rep.damage_stale.ci_low <= rep.damage_stale.mean);
}

#[test]
fn memory_world_identity_safety_and_signal_stay_separate_and_not_language() {
    // Value read is independent of who wrote it (identity is provenance only).
    let mut x = MarkBoard::new(&[A, B], MarkCondition::Measurement, evap(), 3).unwrap();
    let mut y = MarkBoard::new(&[A, B], MarkCondition::Measurement, evap(), 3).unwrap();
    x.write(A, 7, 1.5, 10).unwrap();
    y.write(B, 7, 1.5, 10).unwrap();
    assert_eq!(x.read(A, 7, 12).unwrap(), y.read(A, 7, 12).unwrap());
    // Marks are finite scalars only; a non-finite mark is rejected, cannot encode a message.
    assert!(x.write(A, 7, f64::NAN, 11).is_err());
    // Safety is in the environment: even a maximal mark on every cell never moves anyone out of
    // the grid (walls are no-ops in `env_step`); runs with adversarially hot boards stay in-bounds
    // by construction of `run_variant` (indices would panic otherwise).
    let cfg = MemoryWorldConfig {
        warm_episodes: 1,
        eval_episodes: 2,
        ..MemoryWorldConfig::default()
    };
    let v = cfg.variants()[1];
    let r = run_variant(&cfg, &v).unwrap();
    assert_eq!(r.phase_a.len(), 4);
    assert!(MEMORY_WORLD_NOTE.contains("Not language"));
}
