use gogamen_experimental::compare::{local_detector, run_comparison};

#[test]
fn detector_handles_negation_and_ambiguity() {
    assert_eq!(local_detector("go to the left door"), Some(true));
    assert_eq!(local_detector("not the left door"), Some(false));
    assert_eq!(local_detector("that one"), None);
    assert_eq!(local_detector("left or right"), None);
}

#[test]
fn comparison_is_deterministic_and_counts_every_rejection() {
    let a = run_comparison();
    let b = run_comparison();
    // Everything except measured wall time is deterministic.
    assert_eq!(a.baseline.correct, b.baseline.correct);
    assert_eq!(a.generative.valid, b.generative.valid);
    assert_eq!(a.generative.correct, b.generative.correct);
    assert_eq!(
        a.generative.declared_units_total,
        b.generative.declared_units_total
    );

    let g = &a.generative;
    assert_eq!(
        g.produced,
        g.valid + g.rejected_ungrounded + g.rejected_unknown_goal
    );
    assert!(g.rejected_ungrounded > 0 && g.rejected_unknown_goal > 0);
    assert!(g.valid_rate() < 1.0);
    assert!(g.declared_units_total > 0);
    assert!(g.declared_latency_total > a.baseline.declared_latency_total);
    // Baseline has no declared cost, and is not worse on synthetic keyword fixtures.
    assert_eq!(a.baseline.declared_units_total, 0);
    assert!(a.baseline.accuracy() >= g.accuracy());
    // Disabled by default: every request abstained.
    assert_eq!(a.disabled_abstentions, a.baseline.items);
}
