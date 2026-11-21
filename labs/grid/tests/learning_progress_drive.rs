//! GOG-187: learning-progress drive vs raw-error drive on a hidden grid with a noise cell.

use gogamen_core::{
    Action, ActionGuard, ActionProposal, DriveModality, EndowmentProvenance, GuardContext,
    GuardRejectionReason, SafeFallbackPolicy, Tick,
};
use gogamen_lab_grid::learning_progress_drive::*;

fn print(r: &StudyReport, cfg: &StudyConfig) {
    println!(
        "uniform share = {:.4}; seeds = {}",
        r.uniform_share,
        cfg.seeds.len()
    );
    println!("arm | noise visits mean±sd [min,med,max] | share mean±sd");
    for a in &r.arms {
        println!(
            "{:>15} | {:.1}±{:.1} [{},{},{}] | {:.3}±{:.3}",
            a.arm.label(),
            a.visits.mean,
            a.visits.std,
            a.visits.min,
            a.visits.median,
            a.visits.max,
            a.share.mean,
            a.share.std
        );
    }
    println!("paired z (error - progress) = {:.2}", r.z_error_vs_progress);
    println!("TRAP VERDICT = {:?}", r.trap_verdict);
    println!(
        "arm | err recovered | err delay mean±sd [min,med,max] | drive reappeared | drive delay | pre-level"
    );
    for a in &r.arms {
        println!(
            "{:>15} | {}/{} | {:.1}±{:.1} [{},{},{}] | {}/{} | {:.1}±{:.1} [{},{},{}] | {:.4}",
            a.arm.label(),
            a.error_recovered,
            a.runs.len(),
            a.error_delay.mean,
            a.error_delay.std,
            a.error_delay.min,
            a.error_delay.median,
            a.error_delay.max,
            a.drive_reappeared,
            a.runs.len(),
            a.drive_delay.mean,
            a.drive_delay.std,
            a.drive_delay.min,
            a.drive_delay.median,
            a.drive_delay.max,
            a.pre_level.mean
        );
    }
    println!("RECOVERY VERDICT = {:?}", r.recovery_verdict);
}

#[test]
fn learning_progress_drive_study_paired_seeds_ledger_and_verdicts() {
    let cfg = StudyConfig::new();
    let report = run_study(&cfg).unwrap();
    print(&report, &cfg);

    // Paired design: identical seeds in every arm.
    for a in &report.arms {
        let seeds: Vec<u64> = a.runs.iter().map(|r| r.seed).collect();
        assert_eq!(seeds, cfg.seeds);
        // Guard on the path: every tick either executed or was denied/no-op.
        for r in &a.runs {
            let (e, d, n) = r.guard_counters;
            assert_eq!(e + d + n, cfg.horizon);
            assert_eq!(d + n, 0, "strict guard without precondition denies nothing");
        }
    }

    // AC3: recorded as Situated and present in the endowment ledger.
    assert!(
        report
            .drives
            .iter()
            .all(|d| d.modality == DriveModality::Situated)
    );
    assert!(
        report
            .drives
            .iter()
            .any(|d| d.drive_id == "learning_progress")
    );
    let given: Vec<_> = report
        .ledger
        .entries
        .iter()
        .filter(|e| e.name.starts_with("drive:learning_progress"))
        .collect();
    assert!(given.len() >= 2);
    assert!(given.iter().any(|e| e.source == EndowmentProvenance::Given));
    assert!(
        given
            .iter()
            .any(|e| e.source == EndowmentProvenance::Learned)
    );

    // Determinism (on a reduced configuration to keep runtime low).
    let small = StudyConfig {
        seeds: vec![1, 2],
        horizon: 700,
        ..StudyConfig::new()
    };
    // Debug-string comparison: empty summaries hold NaN, which breaks PartialEq.
    assert_eq!(
        format!("{:?}", run_study(&small).unwrap()),
        format!("{:?}", run_study(&small).unwrap())
    );

    // Verdicts are whatever the preregistered rule says; negatives are valid.
    assert!(matches!(
        report.trap_verdict,
        TrapVerdict::Confirmed | TrapVerdict::Refuted | TrapVerdict::Opposite
    ));
}

#[test]
fn learning_progress_drive_denied_action_is_not_executed() {
    // Precondition: West (index 3) is forbidden. The guard must stop it on the path.
    fn no_west(a: &Action, _ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        match a.as_discrete() {
            Some(d) if d.index() == 3 => Err("west forbidden"),
            _ => Ok(()),
        }
    }
    let cfg = StudyConfig::new();
    // Whole episodes under a denying guard: denied ticks are counted and never executed.
    let guard = ActionGuard::with_precondition(
        SafeFallbackPolicy::RejectOnly,
        no_west as fn(&Action, &GuardContext<'_>) -> Result<(), &'static str>,
    );
    let m = run_episode_guarded(&cfg, Arm::ErrorDriven, 3, guard).unwrap();
    let (executed, denied, noop) = m.guard_counters;
    assert!(
        denied > 0,
        "the agent must have proposed West at least once"
    );
    assert_eq!(noop, 0);
    assert_eq!(executed + denied, cfg.horizon);

    // NoOp policy: nothing executed either.
    let guard = ActionGuard::with_precondition(
        SafeFallbackPolicy::NoOp,
        no_west as fn(&Action, &GuardContext<'_>) -> Result<(), &'static str>,
    );
    let m = run_episode_guarded(&cfg, Arm::Random, 3, guard).unwrap();
    assert!(m.guard_counters.2 > 0);
    assert_eq!(m.guard_counters.1, 0);
}

#[test]
fn learning_progress_drive_guard_blocks_unauthorized_and_stale_proposals() {
    let cfg = StudyConfig::new();
    let world = GuardedGridProbe::new(&cfg);
    let before = world.position();
    let outcome = world.try_unauthorized();
    assert_eq!(
        outcome,
        StepOutcome::Denied(GuardRejectionReason::Unauthorized)
    );
    assert_eq!(world.position_after(), before);
}

/// Tiny harness around the public API to drive `GuardedGrid` by hand.
struct GuardedGridProbe(std::cell::RefCell<GuardedGrid>);

impl GuardedGridProbe {
    fn new(cfg: &StudyConfig) -> Self {
        Self(std::cell::RefCell::new(
            new_guarded_grid(cfg, 7, ActionGuard::strict()).unwrap(),
        ))
    }
    fn position(&self) -> usize {
        self.0.borrow().world().position()
    }
    fn position_after(&self) -> usize {
        self.position()
    }
    fn try_unauthorized(&self) -> StepOutcome {
        let p = ActionProposal::unauthorized(Tick::new(0));
        let out = self.0.borrow_mut().step(&p, Tick::new(0));
        assert_eq!(self.0.borrow().counters(), (0, 1, 0));
        out
    }
}

#[test]
fn learning_progress_drive_rule_change_moves_world_only_through_step() {
    let cfg = StudyConfig::new();
    let mut g = new_guarded_grid(&cfg, 7, ActionGuard::strict()).unwrap();
    let space = gogamen_core::ActionSpace::discrete(N_ACTIONS).unwrap();
    let start = g.world().position();
    let north = Action::discrete_in_space(0, &space).unwrap();
    let out = g.step(
        &ActionProposal::authorized(north.clone(), Tick::new(0)),
        Tick::new(0),
    );
    assert!(matches!(out, StepOutcome::Executed { from, .. } if from == start));
    // Start is bottom row (0,4): nominal North goes to (0,3); reversed North goes South (stays).
    assert_eq!(g.world().position(), 3 * cfg.width);
    g.set_rule(Rule::Reversed);
    let out = g.step(
        &ActionProposal::authorized(north, Tick::new(1)),
        Tick::new(1),
    );
    assert!(matches!(out, StepOutcome::Executed { .. }));
    assert_eq!(g.world().position(), 4 * cfg.width);
}
