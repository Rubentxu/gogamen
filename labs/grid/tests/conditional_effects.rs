//! GOG-164 decisive tests (preregistered holdout and rule in `conditional_effects` module docs).

use gogamen_core::{ConditionalUpdateStatus, PredictionHealth, prior_scores};
use gogamen_lab_grid::GridMap;
use gogamen_lab_grid::conditional_effects::*;

fn train() -> (gogamen_core::ConditionalEffectsLearner, GridMap) {
    let map = GridMap::cul_de_sac_map();
    let mut l = new_lab_learner().unwrap();
    for o in generate_stream(&map, Dynamics::Normal, TRAIN_SEED, TRAIN_STEPS, 0) {
        assert_eq!(l.observe(&o).unwrap(), ConditionalUpdateStatus::Updated);
    }
    (l, map)
}

#[test]
fn conditional_effects_ac1_improves_over_prior_on_fixed_holdout() {
    let (l, map) = train();
    let holdout = generate_stream(
        &map,
        Dynamics::Normal,
        HOLDOUT_SEED,
        HOLDOUT_STEPS,
        1_000_000,
    );
    let prior = prior_scores(&holdout).unwrap();
    let got = l.evaluate(&holdout).unwrap().unwrap();
    println!(
        "AC1 holdout n={} prior brier={:.4} ll={:.4} | learned brier={:.4} ll={:.4} ratio={:.3} v={}",
        got.n,
        prior.brier,
        prior.log_loss,
        got.brier,
        got.log_loss,
        got.brier / prior.brier,
        l.version()
    );
    assert!(
        got.brier <= BRIER_RATIO_MAX * prior.brier,
        "Brier rule failed"
    );
    assert!(got.log_loss < prior.log_loss, "log-loss rule failed");
}

#[test]
fn conditional_effects_ac2_dedup_and_audit() {
    let (mut l, map) = train();
    let (v, c) = (l.version(), l.checksum());
    assert_eq!(l.log().len(), TRAIN_STEPS);
    assert!(
        l.log()
            .iter()
            .enumerate()
            .all(|(i, r)| r.version == i as u64 + 1)
    );
    for o in generate_stream(&map, Dynamics::Normal, TRAIN_SEED, TRAIN_STEPS, 0) {
        assert_eq!(
            l.observe(&o).unwrap(),
            ConditionalUpdateStatus::DuplicateIgnored
        );
    }
    assert_eq!((l.version(), l.checksum()), (v, c));
    assert_eq!(l.duplicates_ignored(), TRAIN_STEPS as u64);
    assert_eq!(l.log().len(), TRAIN_STEPS);
    println!(
        "AC2 version={v} checksum={c:#x} duplicates={}",
        l.duplicates_ignored()
    );
}

#[test]
fn conditional_effects_ac3_changed_dynamics_is_explicit_failure() {
    let (mut l, map) = train();
    assert!(
        matches!(l.health(), PredictionHealth::Healthy { .. }),
        "pre-shift health: {:?}",
        l.health()
    );
    assert_eq!(gated_choice(&l, 1), GatedChoice::Act(1));
    println!("AC3 pre-shift health={:?}", l.health());
    let mut first_failure = None;
    for (i, o) in generate_stream(&map, Dynamics::Flipped, SHIFT_SEED, SHIFT_STEPS, 2_000_000)
        .iter()
        .enumerate()
    {
        l.observe(o).unwrap();
        if first_failure.is_none()
            && matches!(l.health(), PredictionHealth::PredictionFailure { .. })
        {
            first_failure = Some(i + 1);
        }
    }
    println!(
        "AC3 first_failure_after={first_failure:?} final={:?}",
        l.health()
    );
    assert!(
        first_failure.is_some(),
        "failure never declared: {:?}",
        l.health()
    );
    assert!(matches!(
        gated_choice(&l, 0),
        GatedChoice::PredictionFailure(_)
    ));
}
