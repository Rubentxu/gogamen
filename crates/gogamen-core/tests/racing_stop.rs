//! GOG-190: paired experiment, fixed budget vs racing stop, on held-out (reserve) seeds.
//!
//! The core crate cannot depend on `gogamen-lab-grid` (BoolWorld), so the oracle here is a small
//! self-contained stochastic hazard world defined below whose per-(context, action) hazards are
//! known exactly. Viability = the real episode never hits a failure terminal.
//!
//! PREREGISTERED BEFORE THE FIRST RUN (thresholds below were not tuned on reserve seeds):
//! - equal viability: `mean_paired(racing_viable - fixed_viable) >= -EQUAL_VIABILITY_TOLERANCE`.
//! - saving: `1 - mean_tx_racing / mean_tx_fixed >= MIN_SAVING_FRACTION` AND the paired one-sided
//!   t statistic of the per-episode saving exceeds `SAVING_T_MIN`.
//! - overall verdict = both hold; otherwise it is reported as NOT met (negative result).

use std::collections::BTreeMap;

use gogamen_core::action::{Action, ActionSpace, DiscreteAction};
use gogamen_core::determinism::MasterSeed;
use gogamen_core::domain::{Observation, State, Terminal, TerminalReason, Transition};
use gogamen_core::error::ModelError;
use gogamen_core::fmc_planner::{FmcActionSelectionPolicy, FmcPlannerConfig};
use gogamen_core::fmc_scoring::FmcScoringConfig;
use gogamen_core::guard::ActionGuard;
use gogamen_core::model::{DeterministicRng, GenerativeModel, RngSource};
use gogamen_core::racing_stop::{RacingDecision, RacingStopConfig, StopReason, decide};
use gogamen_core::units::{SimDuration, StepDt, Tick};

const EQUAL_VIABILITY_TOLERANCE: f64 = 0.02;
const MIN_SAVING_FRACTION: f64 = 0.10;
const SAVING_T_MIN: f64 = 1.645;

const RESERVE_SEEDS: std::ops::Range<u64> = 10_000..10_200;
const CALIBRATION_SEEDS: std::ops::Range<u64> = 0..8;
const EPISODE_TICKS: u64 = 16;
const POP: usize = 24;
const HORIZON: usize = 6;

/// Hazard per (context, action); context cycles 0..4 every tick.
const HAZARD: [[f64; 3]; 4] = [
    [0.015, 0.30, 0.30],
    [0.05, 0.05, 0.05],
    [0.025, 0.15, 0.025],
    [0.20, 0.04, 0.20],
];

struct HazardWorld {
    space: ActionSpace,
}

impl GenerativeModel for HazardWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        let ctx = state.values()[0] as usize % 4;
        let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
        let failed = rng.next_f64() < HAZARD[ctx][idx];
        let terminal = if failed {
            Terminal::Terminated(TerminalReason::Failure)
        } else {
            Terminal::Ongoing
        };
        let next = vec![((ctx + 1) % 4) as f64];
        let ns = state
            .advance(dt, next.clone())
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(ns.tick(), ns.elapsed(), next)
            .map_err(ModelError::TransitionFailed)?;
        Transition::new(state.tick(), dt, ns, obs, terminal).map_err(ModelError::TransitionFailed)
    }
}

fn planner_config(seed: u64) -> FmcPlannerConfig {
    FmcPlannerConfig::new(
        POP,
        HORIZON,
        StepDt::from_millis_f64(100.0).unwrap(),
        FmcScoringConfig::default(),
        MasterSeed::new(seed),
    )
    .unwrap()
    .with_action_selection(FmcActionSelectionPolicy::LowerConfidenceBound {
        lcb_c: 1.0,
        prior_weight: 2.0,
    })
    .unwrap()
}

fn racing_cfg() -> RacingStopConfig {
    // min_budget = 2 horizon steps; confidence 0.90; coverage 0.75; support >= 2.
    RacingStopConfig::new(2 * POP, POP * HORIZON, 0.90, 0.75, 2).unwrap()
}

struct Episode {
    viable: bool,
    decisions: usize,
    transitions: usize,
    reasons: BTreeMap<StopReason, usize>,
    min_tx_seen: usize,
    guard_evaluated_all: bool,
}

fn run_episode(world: &HazardWorld, seed: u64, cfg: &RacingStopConfig) -> Episode {
    let guard = ActionGuard::strict();
    let fallback = Action::discrete_in_space(0, &world.space).unwrap();
    let dt = StepDt::from_millis_f64(100.0).unwrap();
    let mut env_rng = DeterministicRng::new(seed);
    let mut state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
    let mut ep = Episode {
        viable: true,
        decisions: 0,
        transitions: 0,
        reasons: BTreeMap::new(),
        min_tx_seen: usize::MAX,
        guard_evaluated_all: true,
    };
    for t in 0..EPISODE_TICKS {
        let d: RacingDecision = decide(
            cfg,
            &planner_config(seed.wrapping_mul(1000).wrapping_add(t)),
            world,
            state.clone(),
            &world.space,
            Some(fallback.clone()),
            &guard,
        )
        .unwrap();
        ep.decisions += 1;
        ep.transitions += d.transitions_used;
        ep.min_tx_seen = ep.min_tx_seen.min(d.transitions_used);
        *ep.reasons.entry(d.reason).or_default() += 1;
        // The guard verdict is always present and is what authorizes execution.
        ep.guard_evaluated_all &=
            d.guard.action().is_some() || d.guard.is_no_op() || d.guard.is_rejected();
        let action = d
            .guard
            .action()
            .cloned()
            .unwrap_or_else(|| fallback.clone());
        let tr = world.step(&state, &action, dt, &mut env_rng).unwrap();
        if matches!(tr.terminal(), Terminal::Terminated(_)) {
            ep.viable = false;
            break;
        }
        state = tr.next_state().clone();
    }
    ep
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    (m, var.sqrt())
}

#[test]
fn racing_stop_decide_is_deterministic_and_respects_min_budget() {
    let world = HazardWorld {
        space: ActionSpace::discrete(3).unwrap(),
    };
    let cfg = racing_cfg();
    for seed in CALIBRATION_SEEDS {
        let a = run_episode(&world, seed, &cfg);
        let b = run_episode(&world, seed, &cfg);
        assert_eq!(a.viable, b.viable);
        assert_eq!(a.transitions, b.transitions);
        assert_eq!(a.reasons, b.reasons);
        assert!(a.min_tx_seen >= cfg.min_budget(), "below min budget");
        assert!(a.guard_evaluated_all);
        assert!(a.transitions <= a.decisions * cfg.max_budget());
    }
}

#[test]
fn racing_stop_fixed_budget_uses_full_budget_and_records_budget_reason() {
    let world = HazardWorld {
        space: ActionSpace::discrete(3).unwrap(),
    };
    let fixed = RacingStopConfig::fixed(POP * HORIZON).unwrap();
    for seed in CALIBRATION_SEEDS {
        let e = run_episode(&world, seed, &fixed);
        // Terminal walkers stop consuming transitions, so usage can be slightly below the cap.
        assert!(e.transitions <= e.decisions * POP * HORIZON);
        assert!(e.min_tx_seen >= POP * HORIZON * 3 / 4);
        assert_eq!(e.reasons.len(), 1);
        assert!(e.reasons.contains_key(&StopReason::BudgetExhausted));
    }
}

#[test]
fn racing_stop_paired_reserve_seeds_report() {
    let world = HazardWorld {
        space: ActionSpace::discrete(3).unwrap(),
    };
    let fixed_cfg = RacingStopConfig::fixed(POP * HORIZON).unwrap();
    let race_cfg = racing_cfg();

    let (mut fx_tx, mut rc_tx, mut saving, mut dviab) = (vec![], vec![], vec![], vec![]);
    let (mut fx_v, mut rc_v) = (0.0, 0.0);
    let mut reasons: BTreeMap<StopReason, usize> = BTreeMap::new();
    let mut n_dec = 0usize;
    for seed in RESERVE_SEEDS {
        let f = run_episode(&world, seed, &fixed_cfg);
        let r = run_episode(&world, seed, &race_cfg);
        assert!(r.min_tx_seen >= race_cfg.min_budget());
        let f_per = f.transitions as f64 / f.decisions as f64;
        let r_per = r.transitions as f64 / r.decisions as f64;
        fx_tx.push(f_per);
        rc_tx.push(r_per);
        saving.push(1.0 - r_per / f_per);
        fx_v += f64::from(u8::from(f.viable));
        rc_v += f64::from(u8::from(r.viable));
        dviab.push(f64::from(u8::from(r.viable)) - f64::from(u8::from(f.viable)));
        for (k, c) in r.reasons {
            *reasons.entry(k).or_default() += c;
            n_dec += c;
        }
    }
    let n = fx_tx.len() as f64;
    let (fm, fs) = mean_std(&fx_tx);
    let (rm, rs) = mean_std(&rc_tx);
    let (sm, ss) = mean_std(&saving);
    let (dm, ds) = mean_std(&dviab);
    let t_saving = sm / (ss / n.sqrt()).max(1e-12);
    let equal_viab = dm >= -EQUAL_VIABILITY_TOLERANCE;
    let saves = (1.0 - rm / fm) >= MIN_SAVING_FRACTION && t_saving > SAVING_T_MIN;

    println!("== GOG-190 racing stop, reserve seeds {RESERVE_SEEDS:?} (n={n}) ==");
    println!(
        "fixed : mean tx/decision {fm:.2} (std across episodes {fs:.2}), viability {:.3}",
        fx_v / n
    );
    println!(
        "racing: mean tx/decision {rm:.2} (std across episodes {rs:.2}), viability {:.3}",
        rc_v / n
    );
    println!("paired saving per episode: mean {sm:.4} std {ss:.4} t={t_saving:.2}");
    println!(
        "paired viability diff (racing-fixed): mean {dm:.4} std {ds:.4} se {:.4}",
        ds / n.sqrt()
    );
    for (k, c) in &reasons {
        println!(
            "stop reason {:<16} {:>6} ({:.1}%)",
            k.as_str(),
            c,
            100.0 * *c as f64 / n_dec as f64
        );
    }
    println!(
        "VERDICT equal-viability (diff >= -{EQUAL_VIABILITY_TOLERANCE}): {}",
        if equal_viab { "MET" } else { "NOT MET" }
    );
    println!(
        "VERDICT saving (>= {MIN_SAVING_FRACTION} and t > {SAVING_T_MIN}): {}",
        if saves { "MET" } else { "NOT MET" }
    );
    // Invariants (not the verdict): racing never spends more than fixed per decision on average.
    assert!(rm <= fm + 1e-9);
    assert_eq!(reasons.values().sum::<usize>(), n_dec);
}
