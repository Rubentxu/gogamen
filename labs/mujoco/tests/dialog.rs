//! GOG-66: linguistic perception + AskHuman in a simulated robot (replay oracle, synthetic fixtures).

use gogamen_lab_mujoco::dialog::fixtures::first_of;
use gogamen_lab_mujoco::dialog::*;
use gogamen_lab_mujoco::nav::sim::GuardTag;

const EPS: f64 = 1e-9;

fn run(f: &DialogFixture, kind: PolicyKind, mode: OracleMode) -> EpisodeResult {
    DialogEpisode::run(f, DialogPolicy::new(kind), mode, 7)
}

fn find(id: &str) -> DialogFixture {
    annotated_fixtures()
        .into_iter()
        .find(|f| f.id == id)
        .expect("fixture")
}

// ---------------------------------------------------------------- AC1

#[test]
fn correct_utterance_raises_confidence_and_decides_without_asking() {
    let f = first_of(FixtureKind::Correct); // "left door", truth left
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    assert!((r.p_left - 0.9).abs() < EPS, "p_left={}", r.p_left);
    assert_eq!(r.queries, 0);
    assert_eq!(r.chosen, Some(Door::Left));
    assert_eq!(r.verdict, EpisodeVerdict::Success);
    let upd = r.events.iter().find_map(|e| match &e.kind {
        EventKind::OracleEvidence(u) => Some(*u),
        _ => None,
    });
    let u = upd.expect("evidence event");
    assert!((u.p_before - 0.5).abs() < EPS && (u.p_after - 0.9).abs() < EPS);
}

#[test]
fn ambiguous_utterance_keeps_belief_flat_and_asks_the_human() {
    let f = find("a01-that-door"); // human answers "left", truth left
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    assert_eq!(r.queries, 1);
    // utterance carries no evidence (0.5), human reliability 0.95 -> 0.95
    assert!((r.p_left - 0.95).abs() < EPS, "p_left={}", r.p_left);
    assert_eq!(r.verdict, EpisodeVerdict::Success);
    assert!(
        r.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::OracleUnknown))
    );
    assert!(
        r.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::AskSent))
    );
}

#[test]
fn ambiguous_utterance_with_mute_human_holds_at_prior() {
    let f = find("a04-pointing");
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    assert_eq!(r.queries, 1);
    assert!((r.p_left - 0.5).abs() < EPS);
    assert_eq!(r.verdict, EpisodeVerdict::Abstained);
    assert!(r.chosen.is_none());
    assert!(
        r.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::AskTimedOut))
    );
}

#[test]
fn contradictory_utterances_lower_confidence_then_ask() {
    let f = first_of(FixtureKind::Contradictory); // left, then "no wait, right"
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    let ups: Vec<UpdateRecord> = r
        .events
        .iter()
        .filter_map(|e| match &e.kind {
            EventKind::OracleEvidence(u) | EventKind::HumanEvidence(u) => Some(*u),
            _ => None,
        })
        .collect();
    assert!((ups[0].p_after - 0.9).abs() < EPS);
    // second, contradicting claim: confidence collapses back to the prior
    assert!(
        (ups[1].p_after - 0.5).abs() < EPS,
        "after contradiction {}",
        ups[1].p_after
    );
    assert_eq!(r.queries, 1);
    // human says right with reliability 0.95
    assert!((r.p_left - 0.05).abs() < EPS, "p_left={}", r.p_left);
    assert_eq!(r.verdict, EpisodeVerdict::Success);
}

#[test]
fn contradictory_with_silent_human_rejects_the_command() {
    let f = find("x04-silent");
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    assert!((r.p_left - 0.5).abs() < EPS);
    assert_eq!(r.chosen, None);
    assert_eq!(r.verdict, EpisodeVerdict::Abstained);
}

// ---------------------------------------------------------------- AC2

#[test]
fn never_and_slow_oracles_give_identical_control_trace() {
    let f = first_of(FixtureKind::Correct);
    for kind in [
        PolicyKind::EpistemicAsk,
        PolicyKind::AlwaysAsk,
        PolicyKind::NoAskAbstain,
    ] {
        let never = run(&f, kind, OracleMode::Never);
        let slow = run(
            &f,
            kind,
            OracleMode::ReplaySlow {
                latency_ticks: 1_000,
            },
        );
        let slower = run(
            &f,
            kind,
            OracleMode::ReplaySlow {
                latency_ticks: 50_000,
            },
        );
        assert_eq!(never.trace_hash, slow.trace_hash, "{kind:?}");
        assert_eq!(never.trace_hash, slower.trace_hash, "{kind:?}");
        assert_eq!(never.ticks, slow.ticks);
        assert_eq!(never.decided_at, slow.decided_at);
        assert_eq!(never.released_at, slow.released_at);
    }
}

#[test]
fn control_ticks_are_contiguous_whatever_the_oracle_does() {
    let f = first_of(FixtureKind::Ambiguous);
    for mode in [
        OracleMode::Replay,
        OracleMode::Never,
        OracleMode::ReplaySlow {
            latency_ticks: 10_000,
        },
        OracleMode::Live,
    ] {
        let r = run(&f, PolicyKind::EpistemicAsk, mode);
        for (i, row) in r.trace.iter().enumerate() {
            assert_eq!(
                row.tick,
                i as u64 + 1,
                "{mode:?}: one control step per tick"
            );
        }
        assert_eq!(r.trace.len() as u64, r.ticks);
    }
}

#[test]
fn oracle_latency_never_delays_control_before_a_decision() {
    // Fast oracle releases the planner at tick 5; the never-oracle later. Up to the earlier
    // release the control rows must be identical (holding, guard allows, same timing).
    let f = first_of(FixtureKind::Correct);
    let fast = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    let never = run(&f, PolicyKind::EpistemicAsk, OracleMode::Never);
    let rel = fast.released_at.expect("fast releases") as usize;
    assert!(never.released_at.map(|t| t as usize).unwrap_or(usize::MAX) > rel);
    assert_eq!(fast.trace[..rel], never.trace[..rel]);
    for row in &never.trace[..rel] {
        assert_eq!(row.guard, GuardTag::Allow);
        assert_eq!(row.cmd, [0.0, 0.0]);
    }
}

#[test]
fn live_jev_is_an_unavailable_stub_and_off_by_default() {
    assert_eq!(OracleMode::default(), OracleMode::Replay);
    let f = first_of(FixtureKind::Correct);
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Live);
    assert!(
        r.events
            .iter()
            .any(|e| matches!(e.kind, EventKind::OracleUnavailable))
    );
    assert!(
        r.queries >= 1,
        "with no usable oracle the policy falls back to asking"
    );
    assert!(r.trace.iter().all(|row| row.guard != GuardTag::Reject));
}

#[test]
fn oracle_has_no_authority_over_the_guard_textual() {
    for (name, src) in [
        ("oracle", include_str!("../src/dialog/oracle.rs")),
        ("cognition", include_str!("../src/dialog/cognition.rs")),
        ("perception", include_str!("../src/dialog/perception.rs")),
        ("policy", include_str!("../src/dialog/policy.rs")),
        ("fixtures", include_str!("../src/dialog/fixtures.rs")),
    ] {
        // Strip comments so documentation can mention the guard.
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "ActionGuard",
            "GuardTag",
            "safety",
            "Safety",
            "nav::",
            "Sim",
            "set_goal",
            "planner_mut",
        ] {
            assert!(
                !code.contains(banned),
                "{name} must not reference `{banned}`"
            );
        }
    }
    // The only way cognition influences motion is the typed goal command.
    let cog = include_str!("../src/dialog/cognition.rs");
    assert!(cog.contains("Command::Go"));
}

#[test]
fn guard_remains_in_the_loop_after_the_oracle_decides() {
    let f = first_of(FixtureKind::Correct);
    let r = run(&f, PolicyKind::EpistemicAsk, OracleMode::Replay);
    assert!(r.trace.iter().any(|row| row.guard == GuardTag::Allow));
    assert_eq!(r.verdict, EpisodeVerdict::Success);
}

// ---------------------------------------------------------------- AC3

#[test]
fn at_least_twelve_annotated_fixtures_cover_all_kinds() {
    let fx = annotated_fixtures();
    assert!(fx.len() >= 12);
    for k in [
        FixtureKind::Correct,
        FixtureKind::Ambiguous,
        FixtureKind::Contradictory,
    ] {
        assert!(fx.iter().filter(|f| f.kind == k).count() >= 3);
    }
    let mut ids: Vec<_> = fx.iter().map(|f| f.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), fx.len(), "unique ids");
}

#[test]
fn metrics_known_values() {
    assert!((brier_score(&[(1.0, true), (0.0, false)])).abs() < EPS);
    assert!((brier_score(&[(0.5, true), (0.5, false)]) - 0.25).abs() < EPS);
    // perfectly calibrated: 0.8 confidence, 4 of 5 correct
    let items = vec![
        (0.8, true),
        (0.8, true),
        (0.8, true),
        (0.8, true),
        (0.8, false),
    ];
    assert!(expected_calibration_error(&items, 5).abs() < EPS);
    // overconfident: 0.95 confidence, 50% correct
    let items = vec![(0.95, true), (0.95, false)];
    assert!((expected_calibration_error(&items, 5) - 0.45).abs() < EPS);
}

#[test]
fn policy_comparison_is_deterministic_and_consistent() {
    let fx = annotated_fixtures();
    let (a, _) = run_policy(
        &fx,
        DialogPolicy::new(PolicyKind::EpistemicAsk),
        OracleMode::Replay,
    );
    let (b, _) = run_policy(
        &fx,
        DialogPolicy::new(PolicyKind::EpistemicAsk),
        OracleMode::Replay,
    );
    assert_eq!(a, b);
    let (noask, _) = run_policy(
        &fx,
        DialogPolicy::new(PolicyKind::NoAskGuess),
        OracleMode::Replay,
    );
    let (always, _) = run_policy(
        &fx,
        DialogPolicy::new(PolicyKind::AlwaysAsk),
        OracleMode::Replay,
    );
    assert_eq!(noask.queries, 0);
    assert_eq!(always.queries as usize, fx.len());
    assert!(a.queries > 0 && a.queries < always.queries);
    for row in [&a, &noask, &always] {
        assert_eq!(row.episodes, fx.len());
        assert_eq!(
            row.successes + row.wrong_door + row.abstained + row.collisions + row.other_failures,
            fx.len()
        );
        assert!((0.0..=1.0).contains(&row.brier) && (0.0..=1.0).contains(&row.ece));
    }
    // asking the (mostly right) human improves calibration and accuracy over never asking
    assert!(always.brier < noask.brier);
    assert!(always.map_accuracy > noask.map_accuracy);
}
