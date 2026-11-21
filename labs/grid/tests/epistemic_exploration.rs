//! GOG-169 decisive tests (preregistration in `epistemic_exploration` module docs).

use gogamen_core::{ActionGuard, EpistemicActionKind, SafeFallbackPolicy};
use gogamen_lab_grid::epistemic_exploration::*;
use gogamen_lab_grid::world::ACTION_RIGHT;

#[test]
fn epistemic_exploration_ac1_hidden_door_inspects_and_learns_causality() {
    let cfg = DoorConfig::default();
    let sel = select_for_prior(&cfg, 0.5).unwrap();
    println!(
        "AC1 selector p=0.5: selected={:?} evoi={:.3} cost={:.3} net={:.3}",
        sel.selected,
        sel.inspect.as_ref().unwrap().evoi,
        sel.inspect.as_ref().unwrap().explicit_cost,
        sel.inspect.as_ref().unwrap().net_evoi
    );
    assert_eq!(sel.selected, EpistemicActionKind::Inspect);

    let rep = run_door_ablation(&cfg).unwrap();
    let e = &rep.epistemic;
    println!(
        "AC1 epistemic: net={:.3} probes={} (early={} late={}) cost={:.1} commits={} detours={} \
         learned p(open|lit)={:?} p(open|unlit)={:?} holdout moved-brier={:.4} (prior {:.2}) improves={}",
        e.mean_net_utility,
        e.probes,
        e.early_probes,
        e.late_probes,
        e.total_probe_cost,
        e.commits,
        e.detours,
        e.learned_p_open_lit,
        e.learned_p_open_unlit,
        e.holdout_moved_brier,
        rep.prior_moved_brier,
        rep.improves
    );
    assert!(e.learned_p_open_lit.unwrap() > 0.75, "lit rule not learned");
    assert!(
        e.learned_p_open_unlit.unwrap() < 0.25,
        "unlit rule not learned"
    );
    assert!(rep.improves, "preregistered AC1 rule not met");
}

#[test]
fn epistemic_exploration_ac2_abstains_when_redundant_or_costly() {
    let cfg = DoorConfig::default();
    // Redundant: the belief is already confident, EVOI < probe cost.
    for p in [0.95, 0.9, 0.02] {
        let s = select_for_prior(&cfg, p).unwrap();
        println!(
            "AC2 redundant p={p}: selected={:?} evoi={:.3}",
            s.selected,
            s.inspect.as_ref().unwrap().evoi
        );
        assert_ne!(s.selected, EpistemicActionKind::Inspect, "p={p}");
    }
    // High cost: EVOI at p=0.5 is positive but below cost.
    let costly = DoorConfig {
        probe_cost: 5.0,
        ..cfg
    };
    let s = select_for_prior(&costly, 0.5).unwrap();
    println!(
        "AC2 costly: selected={:?} evoi={:.3} cost=5",
        s.selected,
        s.inspect.as_ref().unwrap().evoi
    );
    assert!(s.inspect.as_ref().unwrap().evoi > 0.0);
    assert_ne!(s.selected, EpistemicActionKind::Inspect);
    let rep = run_door_ablation(&costly).unwrap();
    println!(
        "AC2 costly run: probes={} net={:.3} ablation net={:.3}",
        rep.epistemic.probes, rep.epistemic.mean_net_utility, rep.no_epistemic.mean_net_utility
    );
    assert_eq!(rep.epistemic.probes, 0);
    assert_eq!(
        rep.epistemic.mean_net_utility,
        rep.no_epistemic.mean_net_utility
    );
    assert_eq!(rep.epistemic.detours, costly.episodes, "safe policy kept");

    // In-run redundancy: late probe rate collapses once the rule is learned.
    let r = run_door_ablation(&cfg).unwrap().epistemic;
    println!(
        "AC2 in-run: early={} late={} abstentions={}",
        r.early_probes, r.late_probes, r.inspect_abstentions
    );
    assert!(r.late_probes as f64 <= LATE_PROBE_RATIO_MAX * r.early_probes as f64);
}

#[test]
fn epistemic_exploration_ac3_ablation_reports_effect_without_holdout_leak() {
    let cfg = DoorConfig::default();
    let rep = run_door_ablation(&cfg).unwrap();
    let (e, n, a) = (&rep.epistemic, &rep.no_epistemic, &rep.always_inspect);
    println!("AC3 holdout fingerprint={:#x}", rep.holdout_fingerprint);
    for r in [e, n, a] {
        println!(
            "AC3 {:?}: net={:.3} probe_cost={:.1} probes={} updates={} moved-brier={:.4} all-bit-brier={:.4} leaked={}",
            r.arm,
            r.mean_net_utility,
            r.total_probe_cost,
            r.probes,
            r.learner_updates,
            r.holdout_moved_brier,
            r.holdout_all_brier,
            r.holdout_ids_in_training
        );
        assert_eq!(r.holdout_ids_in_training, 0);
    }
    // Ablation: no probes, no learning, prior-level holdout error, zero probe cost.
    assert_eq!(n.probes, 0);
    assert_eq!(n.learner_updates, 0);
    assert_eq!(n.total_probe_cost, 0.0);
    assert!((n.holdout_moved_brier - 0.25).abs() < 1e-9);
    // Effect of removing the component.
    assert!(e.holdout_moved_brier < n.holdout_moved_brier);
    assert!(e.total_probe_cost > n.total_probe_cost);
    // Selector spends less than the cost-blind reference.
    assert!(e.total_probe_cost < a.total_probe_cost);
    // Holdout disjoint from training streams.
    assert_ne!(DOOR_SEED, DOOR_HOLDOUT_SEED);
    assert!(
        door_holdout(&cfg)
            .iter()
            .all(|o| o.id.as_u64() > DOOR_HOLDOUT_ID_BASE)
    );
}

#[test]
fn epistemic_exploration_safety_guard_not_bypassed() {
    let cfg = DoorConfig::default();
    let holdout = door_holdout(&cfg);
    // Probing (push) is forbidden: exploration must not execute; safe policy kept.
    let guard = ActionGuard::with_precondition(
        SafeFallbackPolicy::RejectOnly,
        DoorSafety {
            forbidden: vec![ACTION_RIGHT],
        },
    );
    let r = run_door_arm(&cfg, DoorArm::Epistemic, &guard, &holdout).unwrap();
    println!(
        "SAFETY forbidden push: probes={} blocked_probes={} blocked_commits={} detours={} learner_updates={}",
        r.probes, r.guard_blocked_probes, r.guard_blocked_commits, r.detours, r.learner_updates
    );
    assert_eq!(r.probes, 0);
    assert_eq!(r.commits, 0);
    assert!(r.guard_blocked_probes > 0);
    assert_eq!(r.learner_updates, 0);
    assert_eq!(r.detours, cfg.episodes);
    // Even the cost-blind arm cannot bypass the guard.
    let a = run_door_arm(&cfg, DoorArm::AlwaysInspect, &guard, &holdout).unwrap();
    assert_eq!(a.probes, 0);
    assert_eq!(a.commits, 0);
}
