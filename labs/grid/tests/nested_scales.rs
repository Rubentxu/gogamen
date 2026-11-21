//! GOG-175 (DEM-022): nested scales. Protocol: `gogamen_lab_grid::nested_scales::PREREG`.

use gogamen_lab_grid::nested_scales::{
    BlockWorldKind, Level, LevelsVerdict, NestedAgent, NestedConfig, RejectReason, ScaledModel,
    block_world, compare_levels, scale_hides_hard_limit, select_scale,
};
use gogamen_lab_grid::{MicroState, Scale};

fn agent_for(
    kind: BlockWorldKind,
    seed: u64,
) -> (NestedAgent, gogamen_lab_grid::nested_scales::BlockWorld) {
    let bw = block_world(seed, kind).unwrap();
    let a = NestedAgent::new(
        &bw.world,
        &[bw.blocks.clone(), bw.identity.clone()],
        bw.starts[0],
        NestedConfig::preregistered(7),
    )
    .unwrap();
    (a, bw)
}

#[test]
fn nested_scales_coarse_when_preserved_fine_on_predictive_failure() {
    // Two-scale fixture: same dynamics, only the leak of one microstate changes.
    let (preserved, bw) = agent_for(BlockWorldKind::Leaky { leak: 1.0 / 64.0 }, 11);
    let s = preserved.coarse_selection();
    assert_eq!(s.label, "blocks", "{s:?}");
    assert!(s.predictive_error <= s.eps && s.predictive_error > 0.0);

    let (failed, _) = agent_for(BlockWorldKind::Leaky { leak: 0.125 }, 11);
    let f = failed.coarse_selection();
    assert_eq!(f.label, "identity", "{f:?}");
    assert!(matches!(
        f.rejected[0].1,
        RejectReason::PredictiveError { error } if error > f.eps
    ));

    // Same seed and observation => same selection and same decision.
    let again = NestedAgent::new(
        &bw.world,
        &[bw.blocks.clone(), bw.identity.clone()],
        bw.starts[0],
        NestedConfig::preregistered(7),
    )
    .unwrap();
    assert_eq!(again.coarse_selection(), preserved.coarse_selection());
    let d1 = preserved.decide(bw.starts[0], 3).unwrap();
    let d2 = again.decide(bw.starts[0], 3).unwrap();
    assert_eq!(d1, d2);
    let d3 = preserved.decide(bw.starts[0], 3).unwrap();
    assert_eq!(d1, d3);
}

#[test]
fn nested_scales_decisions_record_level_scale_goal_uncertainty() {
    let (a, bw) = agent_for(BlockWorldKind::Lumpable, 21);
    let mut saw_goal = false;
    for (tick, s) in bw.starts.iter().enumerate() {
        let d = a.decide(*s, tick as u64).unwrap();
        assert_eq!(d.levels.len(), 2);
        let (c, f) = (&d.levels[0], &d.levels[1]);
        assert_eq!((c.level, f.level), (Level::Coarse, Level::Fine));
        assert_eq!(c.scale_label, "blocks");
        assert_eq!(f.scale_label, "identity");
        assert_eq!((c.horizon, f.horizon), (6, 3));
        assert_eq!(c.goal, f.goal, "fine receives exactly what coarse emitted");
        assert!(c.uncertainty.predictive_error <= 0.05);
        assert!(d.transitions_used() <= NestedConfig::preregistered(7).total_budget());
        assert!(d.transitions_used() > 0);
        if let Some(g) = &c.goal {
            saw_goal = true;
            assert!(
                c.uncertainty
                    .goal_support
                    .is_some_and(|x| x > 0.0 && x <= 1.0)
            );
            // Hard limit: the goal never contains an inviable microstate.
            for m in &g.members {
                assert!(bw.world.state_specs().any(|v| v.state == *m && v.viable));
            }
        }
    }
    assert!(saw_goal);
}

#[test]
fn nested_scales_aggregation_does_not_hide_hard_limits() {
    let mixed = block_world(4, BlockWorldKind::MixedHardLimit).unwrap();
    assert!(scale_hides_hard_limit(&mixed.world, &mixed.blocks));
    let sel = select_scale(
        &mixed.world,
        &[mixed.blocks.clone(), mixed.identity.clone()],
        mixed.starts[0],
        6,
        1.0, // even a huge error budget must not admit the hiding scale
    )
    .unwrap();
    assert_eq!(sel.label, "identity");
    assert_eq!(sel.rejected[0].1, RejectReason::HidesHardLimit);

    // The coarse model still reports failure from the inviable member and refuses to step a
    // macro made only of hard limits.
    let ok = block_world(5, BlockWorldKind::Lumpable).unwrap();
    let m = ScaledModel::new(&ok.world, &ok.blocks, gogamen_core_dt()).unwrap();
    let trap_macro = 4; // last block is the inviable trap
    assert!(!m.macro_is_hard_limit_free(trap_macro));
    assert!(m.macro_is_hard_limit_free(0));
    // Identity scale never hides anything.
    assert!(!scale_hides_hard_limit(
        &ok.world,
        &Scale::identity(&ok.world)
    ));
    // In the mixed world the viable members of the mixed macro are still distinguishable fine-side.
    assert!(!mixed.world.state_specs().all(|v| v.viable));
    let _ = MicroState(0);
}

fn gogamen_core_dt() -> gogamen_core::StepDt {
    gogamen_core::StepDt::from_millis_f64(100.0).unwrap()
}

#[test]
fn nested_scales_two_levels_vs_one_on_reserve_seeds() {
    let seeds: Vec<u64> = (1000..1040).collect();
    let r = compare_levels(&seeds).unwrap();
    println!(
        "reserve: one={:.3} two={:.3} diff={:+.3} se={:.3} verdict={:?} coarse_worlds={}/{} \
         transitions/decision one={:.1} two={:.1}",
        r.mean_one,
        r.mean_two,
        r.mean_diff,
        r.se,
        r.verdict,
        r.worlds_with_coarse_scale,
        r.rows.len(),
        r.transitions_per_decision_one,
        r.transitions_per_decision_two
    );
    // Equal-budget condition (caps are identical; consumed transitions reported).
    let budget = NestedConfig::preregistered(0).total_budget() as f64;
    assert!(r.transitions_per_decision_one <= budget + 1e-9);
    assert!(r.transitions_per_decision_two <= budget + 1e-9);
    assert!(
        (r.transitions_per_decision_one - r.transitions_per_decision_two).abs() <= 0.1 * budget,
        "arms must consume comparable transitions"
    );
    assert!(r.worlds_with_coarse_scale > 0, "coarse level never used");
    // Verdict is reported whichever way it goes (including no improvement).
    assert!(matches!(
        r.verdict,
        LevelsVerdict::Improves | LevelsVerdict::Worse | LevelsVerdict::NoDifference
    ));
}
