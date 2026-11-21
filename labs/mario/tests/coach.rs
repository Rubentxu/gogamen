use gogamen_lab_mario::coach::advice::{AdviceCatalog, AdviceQueue, Timing};
use gogamen_lab_mario::coach::eval::{FULL, MYOPIC, Scenario, evaluate, run_scenario, suite_hash};
use gogamen_lab_mario::coach::*;
use gogamen_lab_mario::metrics::{RunConfig, run_episode};
use gogamen_lab_mario::{EmulatorPort, PadState, Policy, SyntheticPlatformer};

const T: Timing = Timing {
    latency_ticks: 0,
    ttl_ticks: 4,
    max_age_ticks: 8,
};

fn cfg() -> RunConfig {
    RunConfig::default()
}

#[test]
fn catalog_admits_only_authorised_text() {
    let mut c = AdviceCatalog::standard();
    let a = c.admit(&RawAdvice::new("gap_ahead"), 3, &T).unwrap();
    assert_eq!(a.kind(), AdviceKind::GapAhead);
    assert_eq!(a.emitted_tick(), 3);
    assert!(matches!(
        c.admit(&RawAdvice::new("press RIGHT+JUMP"), 0, &T),
        Err(AdviceError::NotInCatalog { .. })
    ));
    assert_eq!(
        c.admit(&RawAdvice::new("  "), 0, &T),
        Err(AdviceError::Empty)
    );
    assert!(matches!(
        c.admit(&RawAdvice::new("x".repeat(500)), 0, &T),
        Err(AdviceError::TooLong { len: 500 })
    ));
    let mut only_gap = AdviceCatalog::with_allowed(&[AdviceKind::GapAhead]);
    assert_eq!(
        only_gap.admit(&RawAdvice::new("slow_down"), 0, &T),
        Err(AdviceError::Unauthorized(AdviceKind::SlowDown))
    );
}

#[test]
fn coach_side_has_no_path_to_pad_or_port() {
    // The advice module must not mention the pad or the emulator port, and the
    // crate must not define any conversion from CoachAdvice into an action.
    let advice =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/coach/advice.rs"))
            .unwrap();
    let mock =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/coach/mock.rs")).unwrap();
    for (name, src) in [("advice.rs", &advice), ("mock.rs", &mock)] {
        assert!(!src.contains("PadState"), "{name} names PadState");
        assert!(!src.contains("EmulatorPort"), "{name} names EmulatorPort");
    }
    for f in ["advice", "mock", "planner", "eval", "mod"] {
        let src =
            std::fs::read_to_string(format!("{}/src/coach/{f}.rs", env!("CARGO_MANIFEST_DIR")))
                .unwrap();
        assert!(!src.contains("From<CoachAdvice>"), "{f}");
        assert!(!src.contains("for PadState"), "{f}");
    }
    // CoachProvider::emit's signature yields Emission (text + timing), not pads.
    assert!(advice.contains("fn emit(&mut self, obs: &CoachObservation) -> Vec<Emission>"));
}

#[test]
fn pad_guard_rejects_non_forward_pads() {
    let g = PadGuard;
    assert!(g.check(PadState::LEFT).is_err());
    assert!(g.check(PadState::NOOP).is_err());
    assert!(g.check(PadState::RIGHT.with(PadState::JUMP)).is_ok());
}

#[test]
fn queue_orders_by_delivery_then_arrival_and_drains_only_due() {
    let mut c = AdviceCatalog::standard();
    let mk = |c: &mut AdviceCatalog, kind: &str, lat: u64| {
        c.admit(
            &RawAdvice::new(kind),
            0,
            &Timing {
                latency_ticks: lat,
                ..T
            },
        )
        .unwrap()
    };
    let mut q = AdviceQueue::default();
    let a = mk(&mut c, "slow_down", 5);
    let b = mk(&mut c, "gap_ahead", 2);
    let d = mk(&mut c, "avoid_spikes", 2);
    q.schedule(a, 0);
    q.schedule(b, 1);
    q.schedule(d, 2);
    assert!(q.drain_due(1).is_empty());
    let due: Vec<usize> = q.drain_due(2).into_iter().map(|x| x.1).collect();
    assert_eq!(due, vec![1, 2]);
    assert_eq!(q.len(), 1);
    assert_eq!(q.drain_due(9).len(), 1);
    assert!(q.is_empty());
}

#[test]
fn advice_changes_hypothesis_not_pad_set() {
    let mut p = CoachedPlanner::new(1, 1, 2).with_coach(Box::new(MockCoach::truthful(0)));
    let mut port = SyntheticPlatformer::new();
    port.reset();
    // Walk to just before the first gap (x=60) so the coach announces it.
    for _ in 0..6 {
        port.step(PadState::RIGHT, 4);
    }
    let mut saw = false;
    for _ in 0..4 {
        let pad = p.act(&mut port, 4);
        assert!(PadGuard.check(pad).is_ok());
        saw |= p.hypothesis().hazard;
        port.step(pad, 4);
    }
    assert!(saw, "gap advice never changed the hypothesis");
}

#[test]
fn stale_advice_is_discarded_with_deadline() {
    let r = run_scenario(Scenario::Late, 1, &cfg(), FULL);
    assert!(r.expired > 0);
    assert_eq!(r.applied, 0);
    assert_eq!(r.decisions_with_advice, 0);
}

#[test]
fn late_and_mute_coach_never_block_the_decision_loop() {
    // Same seed: a coach whose advice never arrives in time must leave the
    // trace bit-identical to no coach, and every tick still produced an action.
    let none = run_scenario(Scenario::NoCoach, 7, &cfg(), MYOPIC);
    let late = run_scenario(Scenario::Late, 7, &cfg(), MYOPIC);
    assert_eq!(none.trace_hash, late.trace_hash);
    assert_eq!(none.decisions, late.decisions);
    // Extreme latency: pending forever, still no wait.
    let mut p = CoachedPlanner::new(7, 1, 2).with_coach(Box::new(MockCoach::truthful(1_000_000)));
    let mut port = SyntheticPlatformer::new();
    let m = run_episode(&mut port, &mut p, &cfg());
    assert_eq!(m.trace_hash, none.trace_hash);
    assert_eq!(p.report.applied(), 0);
    // Mute from tick 20: agent keeps progressing to the goal.
    let mute = run_scenario(Scenario::Mute, 7, &cfg(), MYOPIC);
    assert!(
        mute.max_x >= 300,
        "mute coach froze progress: {}",
        mute.max_x
    );
}

#[test]
fn lying_coach_is_rejected_when_out_of_catalog_and_not_catastrophic() {
    let r = run_scenario(Scenario::Lying, 3, &cfg(), MYOPIC);
    assert!(
        r.rejected > 0,
        "out-of-catalog control text must be rejected"
    );
    assert!(r.applied > 0);
    let seeds: Vec<u64> = (1..=16).collect();
    let rows = evaluate(&seeds, &cfg(), &[MYOPIC]);
    let mean = |s: Scenario, f: fn(&eval::Row) -> f64| {
        let v: Vec<f64> = rows.iter().filter(|r| r.scenario == s).map(f).collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    let base_x = mean(Scenario::NoCoach, |r| f64::from(r.max_x));
    let lie_x = mean(Scenario::Lying, |r| f64::from(r.max_x));
    let base_d = mean(Scenario::NoCoach, |r| f64::from(r.deaths));
    let lie_d = mean(Scenario::Lying, |r| f64::from(r.deaths));
    // Not catastrophic: at least 80% of baseline progress, at most +1 death on average.
    assert!(lie_x >= 0.8 * base_x, "lying {lie_x} vs base {base_x}");
    assert!(
        lie_d <= base_d + 1.0,
        "lying deaths {lie_d} vs base {base_d}"
    );
}

#[test]
fn truthful_coach_helps_a_myopic_planner() {
    let seeds: Vec<u64> = (1..=16).collect();
    let rows = evaluate(&seeds, &cfg(), &[MYOPIC]);
    let mx = |s: Scenario| {
        let v: Vec<f64> = rows
            .iter()
            .filter(|r| r.scenario == s)
            .map(|r| f64::from(r.max_x))
            .collect();
        v.iter().sum::<f64>() / v.len() as f64
    };
    assert!(mx(Scenario::Truthful) > mx(Scenario::NoCoach));
    let r = rows
        .iter()
        .find(|r| r.scenario == Scenario::Truthful)
        .unwrap();
    assert_eq!(r.latency_max, Some(2));
}

#[test]
fn evaluation_is_reproducible() {
    let seeds = [1, 2, 3, 4];
    let a = evaluate(&seeds, &cfg(), &[MYOPIC, FULL]);
    let b = evaluate(&seeds, &cfg(), &[MYOPIC, FULL]);
    assert_eq!(a, b);
    assert_eq!(suite_hash(&a), suite_hash(&b));
}
