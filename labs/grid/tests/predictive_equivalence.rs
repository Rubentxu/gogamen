//! Decisive suite for GOG-201 / DEM-008 v2: tolerance-stable predictive-equivalence
//! macrostates and `phi` as a core scale.
//!
//! Run with:
//! `cargo test -p gogamen-lab-grid --offline predictive_equivalence`

use std::collections::BTreeMap;

use gogamen_core::{BinningVersion, ScaleIdentity};
use gogamen_lab_grid::bool_world::{BoolWorld, MicroState};
use gogamen_lab_grid::macrostate::{
    MACRO_PHI_SCALE_VERSION, MacroLimits, NonMarkovCounterexample, analyze_partition,
    exact_predictive_macrostates, predictive_macrostates,
};

fn micro(i: u8) -> MicroState {
    MicroState(i)
}

/// World where 0 and 1 have identical predictive signatures (same action outcomes) while 4
/// starts with the same observation as 0 but predicts a different observable outcome.
///
/// `delta` perturbs the action-0 probabilities of state 0 only, keeping the sum at 1, so the
/// partition is expected to be stable while `delta <= tolerance`.
fn fixture(delta: f64) -> BoolWorld {
    let to_two = 0.25 + delta;
    let to_three = 0.75 - delta;
    BoolWorld::new(1)
        .with_state(
            micro(0),
            true,
            vec![
                vec![(micro(2), to_two), (micro(3), to_three)],
                vec![(micro(0), 1.0)],
            ],
        )
        .unwrap()
        .with_state(
            micro(1),
            true,
            vec![
                vec![(micro(2), 0.25), (micro(3), 0.75)],
                vec![(micro(1), 1.0)],
            ],
        )
        .unwrap()
        .with_state(
            micro(2),
            true,
            vec![vec![(micro(2), 1.0)], vec![(micro(2), 1.0)]],
        )
        .unwrap()
        .with_state(
            micro(3),
            true,
            vec![vec![(micro(3), 1.0)], vec![(micro(3), 1.0)]],
        )
        .unwrap()
        .with_state(
            micro(4),
            true,
            vec![vec![(micro(2), 1.0)], vec![(micro(4), 1.0)]],
        )
        .unwrap()
        .with_state(
            micro(5),
            true,
            vec![vec![(micro(5), 1.0)], vec![(micro(5), 1.0)]],
        )
        .unwrap()
}

fn obs(state: MicroState) -> u8 {
    match state.0 {
        0 | 1 | 4 => 0,
        2 => 1,
        3 => 2,
        _ => 3,
    }
}

#[test]
fn predictive_equivalence_oracle_groups_equivalent_and_separates_distinct() {
    let limits = MacroLimits::default();
    let report = exact_predictive_macrostates(&fixture(0.0), &obs, 3, limits).unwrap();

    // Predictively equivalent states share a macro.
    assert_eq!(report.phi[&micro(0)], report.phi[&micro(1)]);
    assert_eq!(
        report.macros[report.phi[&micro(0)]],
        vec![micro(0), micro(1)]
    );
    // Same initial observation, different observable outcome distribution: kept apart.
    assert_ne!(report.phi[&micro(0)], report.phi[&micro(4)]);
    assert_eq!(obs(micro(0)), obs(micro(4)));
    // Distinct observable outcomes are never merged.
    assert_ne!(report.phi[&micro(2)], report.phi[&micro(3)]);
    assert_ne!(obs(micro(2)), obs(micro(3)));
    assert_ne!(report.phi[&micro(5)], report.phi[&micro(2)]);
    assert_eq!(report.partition_size, 5);
    assert_eq!(report.micro_count, 6);
}

#[test]
fn predictive_equivalence_quotient_matches_and_sub_tolerance_perturbation_is_stable() {
    let limits = MacroLimits::default();
    let reference = predictive_macrostates(&fixture(0.0), &obs, 3, limits).unwrap();
    let exact = exact_predictive_macrostates(&fixture(0.0), &obs, 3, limits).unwrap();

    // On an exact fixture the production (tolerant) partition coincides with the oracle.
    assert_eq!(reference.phi, exact.phi);
    // Quotient predictions match the originals within tolerance.
    assert!(reference.quotient_max_abs_deviation <= limits.tolerance);
    assert!(reference.max_within_macro_deviation <= limits.tolerance);
    assert!(reference.markovian);
    assert_eq!(reference.partition_size, 5);
    // H(micro | macro) = (2/6) * log2(2) bits.
    assert!((reference.information_loss_bits - 1.0 / 3.0).abs() < 1e-12);

    // A probability perturbation strictly below the tolerance must not change the partition.
    let perturbed = predictive_macrostates(&fixture(5e-13), &obs, 3, limits).unwrap();
    assert_eq!(perturbed.phi, reference.phi);
    assert_eq!(perturbed.partition_size, reference.partition_size);
    assert_eq!(perturbed.phi_identity(), reference.phi_identity());
    assert_eq!(perturbed.phi_hash(), reference.phi_hash());
    assert!(perturbed.max_within_macro_deviation <= limits.tolerance);
    assert!(perturbed.quotient_max_abs_deviation <= limits.tolerance);
}

#[test]
fn predictive_equivalence_above_tolerance_perturbation_changes_partition() {
    let limits = MacroLimits::default();
    let reference = predictive_macrostates(&fixture(0.0), &obs, 3, limits).unwrap();
    let perturbed = predictive_macrostates(&fixture(1e-3), &obs, 3, limits).unwrap();

    // The same probability move above tolerance splits the formerly equivalent pair.
    assert_ne!(perturbed.phi[&micro(0)], perturbed.phi[&micro(1)]);
    assert_eq!(perturbed.partition_size, 6);
    assert_ne!(perturbed.phi_hash(), reference.phi_hash());
    assert_ne!(perturbed.phi_identity(), reference.phi_identity());
    assert!(perturbed.max_within_macro_deviation <= limits.tolerance);
}

#[test]
fn predictive_equivalence_phi_is_a_core_scale() {
    let limits = MacroLimits::default();
    let report = predictive_macrostates(&fixture(0.0), &obs, 2, limits).unwrap();
    let identity: ScaleIdentity = report.phi_identity();

    // `phi` carries the core scale contract: id, version and content hash.
    assert_eq!(identity, report.phi_scale.identity());
    assert_eq!(identity.id(), report.phi_scale.id());
    assert_eq!(identity.version(), MACRO_PHI_SCALE_VERSION);
    assert_eq!(identity.version(), BinningVersion::new(1, 0));
    assert_eq!(identity.hash(), report.phi_scale.hash());
    assert_eq!(identity.hash(), report.phi_hash());
    // Id derives from the microstate declaration (BoolWorld seed), horizon and tolerance.
    assert!(
        identity
            .id()
            .starts_with("gogamen.macro.phi/seed=1/horizon=2/tolerance=1e-9")
    );

    // Reproducible for the same declaration.
    let again = predictive_macrostates(&fixture(0.0), &obs, 2, limits).unwrap();
    assert_eq!(again.phi_identity(), identity);
    assert_eq!(again.phi_hash(), identity.hash());

    // A different horizon is a different scale declaration.
    let deeper = predictive_macrostates(&fixture(0.0), &obs, 3, limits).unwrap();
    assert_ne!(deeper.phi_identity().id(), identity.id());
}

#[test]
fn predictive_equivalence_non_markov_counterexample_exposes_distributions() {
    // (a) Two microstates of one macro with different observable predictions at horizon 2.
    let mut merged: BTreeMap<MicroState, usize> = BTreeMap::new();
    for (state, macro_id) in [(0, 0), (4, 0), (1, 1), (2, 2), (3, 3), (5, 4)] {
        merged.insert(micro(state), macro_id);
    }
    let report =
        analyze_partition(&fixture(0.0), &obs, &merged, 2, MacroLimits::default()).unwrap();
    assert!(!report.markovian);
    match report.counterexample.expect("non-Markov partition") {
        NonMarkovCounterexample::PredictionMismatch {
            a,
            b,
            action_sequence,
            a_outcomes,
            b_outcomes,
            deviation,
            ..
        } => {
            assert_eq!((a, b), (micro(0), micro(4)));
            assert_eq!(action_sequence, vec![0, 0]);
            assert!((deviation - 0.75).abs() < 1e-12);
            assert_eq!(
                a_outcomes,
                vec![(vec![0, 1, 1], 0.25), (vec![0, 2, 2], 0.75)]
            );
            assert_eq!(b_outcomes, vec![(vec![0, 1, 1], 1.0)]);
        }
        other => panic!("unexpected counterexample: {other:?}"),
    }

    // (b) Same one-step observation but the macro-successor distribution depends on the
    // representative microstate (0 -> 2, 1 -> 3; 2 and 3 share an observation but differ).
    let world = successor_world();
    let mut partition: BTreeMap<MicroState, usize> = BTreeMap::new();
    for (state, macro_id) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4)] {
        partition.insert(micro(state), macro_id);
    }
    let report = analyze_partition(&world, &obs2, &partition, 1, MacroLimits::default()).unwrap();
    assert!(!report.markovian);
    match report.counterexample.expect("non-Markov partition") {
        NonMarkovCounterexample::SuccessorDependsOnRepresentative {
            macro_id,
            action,
            a,
            b,
            successor,
            a_distribution,
            b_distribution,
            deviation,
        } => {
            assert_eq!((macro_id, action, a, b), (0, 0, micro(0), micro(1)));
            assert_eq!(successor, 1);
            assert_eq!(a_distribution, vec![(1, 1.0)]);
            assert_eq!(b_distribution, vec![(2, 1.0)]);
            assert!((deviation - 1.0).abs() < 1e-12);
        }
        other => panic!("unexpected counterexample: {other:?}"),
    }
}

// --- AC1/AC3 auxiliary fixture: successors differ inside a merged macro. -------------

fn successor_world() -> BoolWorld {
    BoolWorld::new(2)
        .with_state(micro(0), true, vec![vec![(micro(2), 1.0)]])
        .unwrap()
        .with_state(micro(1), true, vec![vec![(micro(3), 1.0)]])
        .unwrap()
        .with_state(micro(2), true, vec![vec![(micro(4), 1.0)]])
        .unwrap()
        .with_state(micro(3), true, vec![vec![(micro(5), 1.0)]])
        .unwrap()
        .with_state(micro(4), true, vec![vec![(micro(4), 1.0)]])
        .unwrap()
        .with_state(micro(5), true, vec![vec![(micro(5), 1.0)]])
        .unwrap()
}

fn obs2(state: MicroState) -> u8 {
    match state.0 {
        0 | 1 => 0,
        2 | 3 => 1,
        4 => 2,
        _ => 3,
    }
}
