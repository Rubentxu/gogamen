//! Epistemic exploration on a hidden-door Grid fixture (GOG-169 / DEM-016, first slice).
//!
//! The agent stands before a door. A visible context bit (`lit`) is the hidden causal law's
//! antecedent: the door opens **iff** `lit` (the agent is never told this). It chooses between
//! *Commit* (push through: `+open` if the door opens, `closed` penalty otherwise) and the safe
//! *Detour* (constant payoff). Before deciding it may *Inspect*: push the door as a probe
//! (explicit cost, noisy outcome). The choice is made by the core
//! [`EpistemicActionSelector`] (EVOI net of cost) over a [`DecisionPayoffModel`]; the prior comes
//! from the [`ConditionalEffectsLearner`] (uninformed `0.5` until it has evidence), and every
//! executed action (probe, commit, detour) goes through [`ActionGuard`]. Exploration therefore
//! never bypasses safety: a guard rejection makes the agent keep the safe policy.
//!
//! # Preregistration (fixed BEFORE the first run)
//! - Payoffs: commit `+10` (open) / `-6` (closed), detour `+4`, probe cost `0.5`, probe slip `0.1`
//!   (sensor modelled as sensitivity = specificity = `0.9`). Episodes: [`DoorConfig::episodes`]
//!   (400) from [`DOOR_SEED`]; `lit` is a fair coin.
//! - Arms: `Epistemic` (selector), `NoEpistemic` (ablation: selector and probes disabled, decision
//!   under the static `0.5` prior), `AlwaysInspect` (cost-blind reference).
//! - Holdout: [`DOOR_HOLDOUT_SEED`], [`DOOR_HOLDOUT_N`] push transitions, ids from
//!   [`DOOR_HOLDOUT_ID_BASE`]; generated once, never passed to `observe`, never used for
//!   selection; only scored at the end with [`ConditionalEffectsLearner::evaluate`].
//! - AC1 rule: epistemic "improves" iff mean net utility (payoff minus probe cost) >=
//!   ablation + [`MIN_NET_GAIN`] AND holdout Brier of the `moved` bit <= [`MOVED_BRIER_MAX`]
//!   AND probes happened.
//! - AC2 rule: abstain iff the selector does not choose Inspect (redundant belief or probe cost
//!   above EVOI); with probe cost 5 the run must perform 0 probes and equal the ablation.
//!   In-run redundancy: probes in the last quarter <= [`LATE_PROBE_RATIO_MAX`] x first quarter.
//! - AC3: ablation effect is reported as differences in holdout Brier, net utility and cost.
//!   Anything that fails these rules is reported as NOT improved.
//!
//! This is a controlled fixture; it makes no claim of emergence or general capability.

use std::fmt;

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, AdvanceSpec, BinaryBelief, CommCostWeights,
    ConditionalEffectsLearner, ConditionalObservation, DecisionOption, DecisionPayoffModel,
    DeterministicRng, DiscreteAction, DomainPrecondition, EpistemicActionKind,
    EpistemicActionSelector, EvidenceRef, GuardContext, GuardDecision, InspectSpec,
    LikelihoodModel, MasterSeed, ObservationId, OptionId, RngSource, SafeFallbackPolicy, SensorId,
    StreamKey, Terminal, Tick,
};

use crate::conditional_effects::new_lab_learner;
use crate::world::{ACTION_RIGHT, ACTION_STAY, GRID_ACTION_CARDINALITY};

/// Seed of the online episode stream.
pub const DOOR_SEED: u64 = 0x169_0001;
/// Seed of the fixed holdout stream (disjoint from [`DOOR_SEED`]).
pub const DOOR_HOLDOUT_SEED: u64 = 0x169_0002;
/// Holdout transitions.
pub const DOOR_HOLDOUT_N: usize = 300;
/// First id of holdout observations (training ids stay far below).
pub const DOOR_HOLDOUT_ID_BASE: u64 = 1_000_000;
/// Minimum mean net-utility gain over the ablation to claim improvement.
pub const MIN_NET_GAIN: f64 = 0.5;
/// Maximum holdout Brier of the `moved` bit to claim causal learning (prior = 0.25).
pub const MOVED_BRIER_MAX: f64 = 0.15;
/// Maximum late/early probe ratio to claim redundancy-driven abstention.
pub const LATE_PROBE_RATIO_MAX: f64 = 0.5;

/// Bit layout of the macrostate: `[blocked N, blocked S, lit, blocked E, moved]`.
const BIT_LIT: usize = 2;
const BIT_MOVED: usize = 4;
const NUM_BITS: usize = 5;

/// Error of the exploration lab.
#[derive(Debug, Clone, PartialEq)]
pub struct DoorError(pub String);

impl fmt::Display for DoorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epistemic exploration error: {}", self.0)
    }
}

impl std::error::Error for DoorError {}

fn err<E: fmt::Debug>(e: E) -> DoorError {
    DoorError(format!("{e:?}"))
}

/// Fixture parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoorConfig {
    /// Episodes in the online run.
    pub episodes: usize,
    /// Payoff of committing when the door opens.
    pub payoff_open: f64,
    /// Payoff of committing when the door is closed.
    pub payoff_closed: f64,
    /// Payoff of the safe detour.
    pub payoff_detour: f64,
    /// Explicit cost of one probe.
    pub probe_cost: f64,
    /// Probability that a probe reports the wrong outcome.
    pub probe_slip: f64,
}

impl Default for DoorConfig {
    fn default() -> Self {
        Self {
            episodes: 400,
            payoff_open: 10.0,
            payoff_closed: -6.0,
            payoff_detour: 4.0,
            probe_cost: 0.5,
            probe_slip: 0.1,
        }
    }
}

/// Policy arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DoorArm {
    /// Selector-driven (EVOI net of cost).
    Epistemic,
    /// Ablation: no selector, no probes.
    NoEpistemic,
    /// Cost-blind reference: probe whenever the guard allows.
    AlwaysInspect,
}

/// Domain precondition that can forbid actions (safety constraint exploration must respect).
#[derive(Debug, Clone, Default)]
pub struct DoorSafety {
    /// Discrete action indices that are forbidden.
    pub forbidden: Vec<usize>,
}

impl DomainPrecondition for DoorSafety {
    fn check(&self, action: &Action, _ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        match action.as_discrete() {
            Some(a) if self.forbidden.contains(&a.index()) => Err("forbidden by door safety"),
            _ => Ok(()),
        }
    }
}

/// Metrics of one arm.
#[derive(Debug, Clone, PartialEq)]
pub struct DoorArmReport {
    /// Arm.
    pub arm: DoorArm,
    /// Episodes run.
    pub episodes: usize,
    /// Mean payoff minus probe cost per episode.
    pub mean_net_utility: f64,
    /// Total explicit probe cost.
    pub total_probe_cost: f64,
    /// Probes executed.
    pub probes: usize,
    /// Probes in the first quarter of episodes.
    pub early_probes: usize,
    /// Probes in the last quarter of episodes.
    pub late_probes: usize,
    /// Episodes where the selector declined to inspect.
    pub inspect_abstentions: usize,
    /// Commits executed.
    pub commits: usize,
    /// Detours executed.
    pub detours: usize,
    /// Probes the guard refused (executed = 0 for those).
    pub guard_blocked_probes: usize,
    /// Commits the guard refused (safe detour taken instead).
    pub guard_blocked_commits: usize,
    /// Observations applied by the learner.
    pub learner_updates: u64,
    /// Learned `P(moved | lit, push)`; `None` if uninformed.
    pub learned_p_open_lit: Option<f64>,
    /// Learned `P(moved | unlit, push)`; `None` if uninformed.
    pub learned_p_open_unlit: Option<f64>,
    /// Holdout Brier of the `moved` bit.
    pub holdout_moved_brier: f64,
    /// Holdout Brier over all bits.
    pub holdout_all_brier: f64,
    /// Holdout observations that reached the learner (must be 0).
    pub holdout_ids_in_training: usize,
}

/// Full ablation report.
#[derive(Debug, Clone, PartialEq)]
pub struct DoorAblationReport {
    /// Selector arm.
    pub epistemic: DoorArmReport,
    /// Ablation without the epistemic component.
    pub no_epistemic: DoorArmReport,
    /// Cost-blind reference.
    pub always_inspect: DoorArmReport,
    /// Brier of the uninformed prior on the holdout, `moved` bit.
    pub prior_moved_brier: f64,
    /// Fingerprint of the holdout (identical for every arm).
    pub holdout_fingerprint: u64,
    /// AC1 rule verdict for the epistemic arm.
    pub improves: bool,
}

fn uniform(rng: &mut DeterministicRng) -> f64 {
    (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

fn door_state(lit: bool) -> Vec<bool> {
    vec![true, true, lit, false, false]
}

fn push_observation(id: u64, lit: bool, moved: bool) -> ConditionalObservation {
    let mut next = door_state(lit);
    next[BIT_MOVED] = moved;
    ConditionalObservation {
        id: ObservationId::new(id),
        evidence: EvidenceRef::new(0x169_0000_0000 + id),
        state: door_state(lit),
        action: ACTION_RIGHT,
        next,
    }
}

/// Fixed holdout of noisy push transitions (never learned).
pub fn door_holdout(cfg: &DoorConfig) -> Vec<ConditionalObservation> {
    let mut rng: DeterministicRng =
        MasterSeed::new(DOOR_HOLDOUT_SEED).derive(StreamKey::Episode(0x169));
    (0..DOOR_HOLDOUT_N)
        .map(|i| {
            let lit = uniform(&mut rng) < 0.5;
            let moved = lit ^ (uniform(&mut rng) < cfg.probe_slip);
            push_observation(DOOR_HOLDOUT_ID_BASE + i as u64 + 1, lit, moved)
        })
        .collect()
}

fn fingerprint(h: &[ConditionalObservation]) -> u64 {
    let mut x = 0xcbf2_9ce4_8422_2325_u64;
    for o in h {
        for b in o.state.iter().chain(o.next.iter()) {
            x = (x ^ u64::from(*b)).wrapping_mul(0x100_0000_01b3);
        }
        x = (x ^ o.id.as_u64()).wrapping_mul(0x100_0000_01b3);
    }
    x
}

fn payoff_model(cfg: &DoorConfig) -> Result<DecisionPayoffModel, DoorError> {
    let commit = DecisionOption::new(
        OptionId::new("commit").map_err(err)?,
        cfg.payoff_closed,
        cfg.payoff_open,
    )
    .map_err(err)?
    .with_action(Action::Discrete(
        DiscreteAction::new(ACTION_RIGHT, GRID_ACTION_CARDINALITY).map_err(err)?,
    ));
    let detour = DecisionOption::new(
        OptionId::new("detour").map_err(err)?,
        cfg.payoff_detour,
        cfg.payoff_detour,
    )
    .map_err(err)?
    .with_action(Action::Discrete(
        DiscreteAction::new(ACTION_STAY, GRID_ACTION_CARDINALITY).map_err(err)?,
    ));
    DecisionPayoffModel::new(vec![commit, detour]).map_err(err)
}

fn probe_likelihood(cfg: &DoorConfig) -> Result<LikelihoodModel, DoorError> {
    LikelihoodModel::from_sensitivity_specificity(1.0 - cfg.probe_slip, 1.0 - cfg.probe_slip)
        .map_err(err)
}

/// Result of one preregistered selector evaluation (used by AC2 unit-level checks).
pub fn select_for_prior(
    cfg: &DoorConfig,
    p_open: f64,
) -> Result<gogamen_core::EpistemicSelectionReport, DoorError> {
    let belief = BinaryBelief::new(p_open).map_err(err)?;
    let inspect = InspectSpec::new(SensorId::new(169), cfg.probe_cost, probe_likelihood(cfg)?)
        .map_err(err)?;
    EpistemicActionSelector::evaluate(
        &belief,
        &payoff_model(cfg)?,
        CommCostWeights::default(),
        &AdvanceSpec::default(),
        Some(&inspect),
        None,
    )
    .map_err(err)
}

fn guarded(
    guard: &ActionGuard<DoorSafety>,
    space: &ActionSpace,
    action: Action,
    tick: Tick,
) -> GuardDecision {
    let proposal = ActionProposal::authorized(action, tick);
    let ctx = GuardContext::now(space, tick, Terminal::Ongoing);
    guard.evaluate(&proposal, &ctx)
}

fn discrete(i: usize) -> Result<Action, DoorError> {
    Ok(Action::Discrete(
        DiscreteAction::new(i, GRID_ACTION_CARDINALITY).map_err(err)?,
    ))
}

fn clamp_p(p: f64) -> f64 {
    p.clamp(0.01, 0.99)
}

/// Runs one arm online, then scores the fixed `holdout` without learning from it.
pub fn run_door_arm(
    cfg: &DoorConfig,
    arm: DoorArm,
    guard: &ActionGuard<DoorSafety>,
    holdout: &[ConditionalObservation],
) -> Result<DoorArmReport, DoorError> {
    let space = ActionSpace::discrete(GRID_ACTION_CARDINALITY).map_err(err)?;
    let payoff = payoff_model(cfg)?;
    let likelihood = probe_likelihood(cfg)?;
    let inspect =
        InspectSpec::new(SensorId::new(169), cfg.probe_cost, likelihood.clone()).map_err(err)?;
    let mut learner: ConditionalEffectsLearner = new_lab_learner().map_err(err)?;
    let mut rng: DeterministicRng = MasterSeed::new(DOOR_SEED).derive(StreamKey::Episode(0x169));
    let mut next_id = 0_u64;
    let mut r = DoorArmReport {
        arm,
        episodes: cfg.episodes,
        mean_net_utility: 0.0,
        total_probe_cost: 0.0,
        probes: 0,
        early_probes: 0,
        late_probes: 0,
        inspect_abstentions: 0,
        commits: 0,
        detours: 0,
        guard_blocked_probes: 0,
        guard_blocked_commits: 0,
        learner_updates: 0,
        learned_p_open_lit: None,
        learned_p_open_unlit: None,
        holdout_moved_brier: 0.0,
        holdout_all_brier: 0.0,
        holdout_ids_in_training: 0,
    };
    let mut total = 0.0;
    for ep in 0..cfg.episodes {
        let tick = Tick::new(ep as u64);
        let lit = uniform(&mut rng) < 0.5;
        let state = door_state(lit);
        let p_prior = if arm == DoorArm::NoEpistemic {
            0.5
        } else {
            let p = learner
                .predict_bit(&state, ACTION_RIGHT, BIT_MOVED)
                .map_err(err)?;
            if p.informed {
                clamp_p(p.probability)
            } else {
                0.5
            }
        };
        let mut belief = BinaryBelief::new(p_prior).map_err(err)?;

        let wants_inspect = match arm {
            DoorArm::NoEpistemic => false,
            DoorArm::AlwaysInspect => true,
            DoorArm::Epistemic => {
                let rep = EpistemicActionSelector::evaluate(
                    &belief,
                    &payoff,
                    CommCostWeights::default(),
                    &AdvanceSpec::default(),
                    Some(&inspect),
                    None,
                )
                .map_err(err)?;
                let w = rep.selected == EpistemicActionKind::Inspect;
                if !w {
                    r.inspect_abstentions += 1;
                }
                w
            }
        };
        let mut net = 0.0;
        if wants_inspect {
            match guarded(guard, &space, discrete(ACTION_RIGHT)?, tick) {
                GuardDecision::Allow(_) => {
                    let slipped = uniform(&mut rng) < cfg.probe_slip;
                    let observed = lit ^ slipped;
                    next_id += 1;
                    learner
                        .observe(&push_observation(next_id, lit, observed))
                        .map_err(err)?;
                    let mut d = belief.as_discrete().clone();
                    d.update_with_outcome(usize::from(observed), &likelihood)
                        .map_err(err)?;
                    belief = BinaryBelief::from_discrete(d).map_err(err)?;
                    net -= cfg.probe_cost;
                    r.total_probe_cost += cfg.probe_cost;
                    r.probes += 1;
                    if ep < cfg.episodes / 4 {
                        r.early_probes += 1;
                    }
                    if ep >= cfg.episodes - cfg.episodes / 4 {
                        r.late_probes += 1;
                    }
                }
                _ => r.guard_blocked_probes += 1,
            }
        }
        let decision = payoff.optimal_decision(&belief);
        let proposed = decision
            .action
            .clone()
            .ok_or_else(|| DoorError("no action".into()))?;
        let is_commit = proposed.as_discrete().map(|a| a.index()) == Some(ACTION_RIGHT);
        let committed = is_commit
            && match guarded(guard, &space, proposed, tick) {
                GuardDecision::Allow(_) => true,
                _ => {
                    r.guard_blocked_commits += 1;
                    false
                }
            };
        if committed {
            r.commits += 1;
            net += if lit {
                cfg.payoff_open
            } else {
                cfg.payoff_closed
            };
            next_id += 1;
            learner
                .observe(&push_observation(next_id, lit, lit))
                .map_err(err)?;
        } else {
            r.detours += 1;
            net += cfg.payoff_detour;
        }
        total += net;
    }
    r.mean_net_utility = total / cfg.episodes as f64;
    r.learner_updates = learner.version();
    r.holdout_ids_in_training = learner
        .log()
        .iter()
        .filter(|u| u.observation.as_u64() >= DOOR_HOLDOUT_ID_BASE)
        .count();
    let lit_p = learner
        .predict_bit(&door_state(true), ACTION_RIGHT, BIT_MOVED)
        .map_err(err)?;
    let unlit_p = learner
        .predict_bit(&door_state(false), ACTION_RIGHT, BIT_MOVED)
        .map_err(err)?;
    r.learned_p_open_lit = lit_p.informed.then_some(lit_p.probability);
    r.learned_p_open_unlit = unlit_p.informed.then_some(unlit_p.probability);
    let mut sq = 0.0;
    for o in holdout {
        let p = learner
            .predict_bit(&o.state, o.action, BIT_MOVED)
            .map_err(err)?
            .probability;
        let t = f64::from(u8::from(o.next[BIT_MOVED]));
        sq += (p - t) * (p - t);
    }
    r.holdout_moved_brier = sq / holdout.len() as f64;
    r.holdout_all_brier = learner
        .evaluate(holdout)
        .map_err(err)?
        .map_or(f64::NAN, |s| s.brier);
    debug_assert_eq!(NUM_BITS, 5);
    let _ = BIT_LIT;
    Ok(r)
}

/// Runs the three arms against the same fixed holdout and applies the preregistered AC1 rule.
pub fn run_door_ablation(cfg: &DoorConfig) -> Result<DoorAblationReport, DoorError> {
    let holdout = door_holdout(cfg);
    let fp = fingerprint(&holdout);
    let guard =
        ActionGuard::with_precondition(SafeFallbackPolicy::RejectOnly, DoorSafety::default());
    let epistemic = run_door_arm(cfg, DoorArm::Epistemic, &guard, &holdout)?;
    let no_epistemic = run_door_arm(cfg, DoorArm::NoEpistemic, &guard, &holdout)?;
    let always_inspect = run_door_arm(cfg, DoorArm::AlwaysInspect, &guard, &holdout)?;
    if fingerprint(&holdout) != fp {
        return Err(DoorError("holdout mutated".into()));
    }
    let prior_moved_brier = 0.25;
    let improves = epistemic.probes > 0
        && epistemic.mean_net_utility >= no_epistemic.mean_net_utility + MIN_NET_GAIN
        && epistemic.holdout_moved_brier <= MOVED_BRIER_MAX;
    Ok(DoorAblationReport {
        epistemic,
        no_epistemic,
        always_inspect,
        prior_moved_brier,
        holdout_fingerprint: fp,
        improves,
    })
}
