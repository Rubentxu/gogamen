//! GOG-176 / DEM-023: final emergence-and-transferability comparison on the Arena.
//!
//! GOG-203 adds the three closed-loop arms of its first slice (see [`crate::grid_arms`]):
//! a reward-free entropic Grid controller, a learned-model empowerment controller and a causal-rule
//! learner. They are registered in [`arm_table`] with their access level, evaluated on the same
//! Grid holdout seeds and budgets, and reported as descriptive rows. The pre-registered
//! hypotheses H1-H6 and their hash are **not** changed, so their verdicts do not depend on the new
//! arms.
//!
//! This module **extends** the existing arena instead of adding a parallel science runner:
//! - the fair Kart arms (random, geometric, pure reward-free entropic, goal-oriented FMC) and the
//!   privileged ceiling run through [`run_kart_arena_preregistered`], stamping the pre-registered
//!   hypotheses hash into every trace manifest before execution;
//! - the learned arms (GOG-173 `KartLearnedAgent`, GOG-168 `PlanGrid`) are executed by their lab
//!   harnesses, which only hand observables and outcomes to the agent;
//! - verdicts come from [`evaluate_hypotheses`] and are reported next to, never derived from,
//!   the build status.
//!
//! # Pre-declared protocol (fixed in this file before the first run)
//! - Seeds: dev [`DEV_SEEDS`] (training / calibration), holdout [`HOLDOUT_SEEDS`] (never used
//!   to train or choose parameters). Model-quality (Brier) holdout transitions use
//!   `UNSEEN_SEEDS` of the racing lab (disjoint from the dev seeds).
//! - Kart budget: `max_ticks = 320` (= 40 macro steps x 8 ticks), planning budget of **72
//!   simulated transitions per decision** for every planning arm (pure: 1 walker x 8 steps x 9
//!   root candidates; goal FMC: 9 walkers x 8 steps; learned FMC: 24 walkers x 3 macro steps).
//!   Decision cadence differs (arena roles decide every tick, the learned arm every 8 ticks), so
//!   *total* simulated transitions differ and are reported per arm.
//! - Learning budget: 45 random-exploration episodes (3 dev seeds x 15) on the training track,
//!   <= 40 macro transitions each. The reference "in-domain" learner gets the same budget on the
//!   evaluation track.
//! - Grid: models learn from 800 random-policy transitions per holdout seed; every planning arm
//!   has the same 160 simulated transitions per decision.
//!
//! # What is NOT claimed
//! Numbers describe these concrete tracks, maps and seeds. Nothing here shows general
//! intelligence, universal competence or emergence; a result is only what the pre-registered
//! hypotheses say about it, and negative results are reported as such.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gogamen_core::{LearningPolicy, MasterSeed};
use gogamen_lab_grid::hidden_dynamics::{
    HiddenCurveConfig, HiddenEpisode, LearnerPolicy, PlanGrid, collect_hidden_experience,
    freeze_grid_model, grid_cell_scale, run_learner_policy_episode, run_plan_grid_episode,
    run_privileged_fmc_episode,
};
use gogamen_lab_grid::{GridMap, GridWorldConfig, SeedSplit};
use gogamen_lab_racing::TrackMap;
use gogamen_lab_racing::kart_learned::{KartExecutedTransition, KartLearnedAgent};
use gogamen_lab_racing::kart_learned_lab::{
    KART_EPISODES_PER_SEED, KART_HOLD_TICKS, KART_LEARNED_HOLDOUT_SEEDS, KART_LEARNED_TRAIN_SEEDS,
    KART_MAX_MACRO_STEPS, KartEpisodeCounters, explore_episodes, run_controller_episode,
};
use gogamen_spec::{HypothesisComparator as Cmp, HypothesisSpec, SpecError, hypotheses_hash};

use crate::grid_arms::{
    GRID_CAUSAL_RULE_ARM, GRID_EMPOWERMENT_ARM, GRID_EMPOWERMENT_HORIZON,
    GRID_EMPOWERMENT_POPULATION, GRID_ENTROPIC_HORIZON, GRID_ENTROPIC_POPULATION,
    GRID_ENTROPIC_PURE_ARM, GridCausalAgent, RolloutConfig, run_causal_grid_episode,
    run_empowerment_episode, run_entropic_pure_grid_episode,
};
use crate::hypothesis::{BuildStatus, HypothesisEvidence, VerdictEntry, evaluate_hypotheses};
use crate::kart::{
    KartArenaBudget, KartArenaConfig, KartArenaReport, KartRole, run_kart_arena_preregistered,
};
use crate::runner::{ArenaError, ArenaSeedSet};

/// Dev seeds of the Kart arena (calibration only).
pub const DEV_SEEDS: [u64; 3] = [7, 19, 42];
/// Holdout seeds of the closed-loop evaluation (never trained or tuned on).
pub const HOLDOUT_SEEDS: [u64; 8] = [101, 102, 103, 104, 105, 106, 107, 108];
/// Ticks per Kart episode (`40 x 8`).
pub const KART_MAX_TICKS: usize = KART_MAX_MACRO_STEPS * KART_HOLD_TICKS;
/// Planning horizon (ticks) of the arena planning roles.
pub const KART_HORIZON: usize = 8;
/// Simulated transitions per decision shared by all fair planning arms (24 x 3 = 9 x 8 = 72).
pub const KART_TRANSITIONS_PER_DECISION: usize = 72;
/// Random-transition budget of the grid learners.
pub const GRID_OBSERVATIONS: usize = 800;

/// Error of the emergence experiment.
#[derive(Debug)]
pub enum EmergenceError {
    /// Arena configuration or run error.
    Arena(ArenaError),
    /// Hypothesis specification error.
    Spec(SpecError),
    /// Lab harness error (message only).
    Lab(String),
}

impl std::fmt::Display for EmergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arena(e) => write!(f, "arena: {e:?}"),
            Self::Spec(e) => write!(f, "spec: {e}"),
            Self::Lab(m) => write!(f, "lab: {m}"),
        }
    }
}

impl std::error::Error for EmergenceError {}

impl From<ArenaError> for EmergenceError {
    fn from(e: ArenaError) -> Self {
        Self::Arena(e)
    }
}
impl From<SpecError> for EmergenceError {
    fn from(e: SpecError) -> Self {
        Self::Spec(e)
    }
}

fn lab<E: std::fmt::Display>(e: E) -> EmergenceError {
    EmergenceError::Lab(e.to_string())
}

/// Pre-registered hypotheses. Thresholds were fixed before the first run; any later change
/// changes [`emergence_hypotheses_hash`] and makes verdicts `INCONCLUSIVE` (`HASH_MISMATCH`).
///
/// All claims are *positive* claims (learning helps / transfers); the refutation condition is the
/// matching failure. Values between thresholds stay `INCONCLUSIVE`.
pub fn emergence_hypotheses() -> Vec<HypothesisSpec> {
    vec![
        HypothesisSpec::new(
            "H1-learning-helps",
            "Online-learned model + FMC clears more checkpoints on the training track than the same agent with the untrained prior model",
            "kart_learning_gain_efficacy_same",
            Cmp::Ge,
            0.10,
            Cmp::Le,
            0.0,
        ),
        HypothesisSpec::new(
            "H2-model-transfers",
            "One-step Brier score of the learned model on an unseen track is not much worse than on the training track",
            "kart_brier_transfer_gap",
            Cmp::Le,
            0.05,
            Cmp::Ge,
            0.15,
        ),
        HypothesisSpec::new(
            "H3-control-transfers",
            "Closed-loop efficacy of the learned agent drops by at most 0.10 on an unseen track",
            "kart_efficacy_transfer_drop",
            Cmp::Le,
            0.10,
            Cmp::Ge,
            0.30,
        ),
        HypothesisSpec::new(
            "H4-beats-reward-free",
            "Learned agent beats the reward-free pure entropic controller on the unseen track",
            "kart_learned_minus_pure_efficacy_other",
            Cmp::Ge,
            0.10,
            Cmp::Le,
            0.0,
        ),
        HypothesisSpec::new(
            "H5-safer-than-random",
            "Learned agent collides at least 0.2 less often than random control on the unseen track",
            "kart_collision_rate_learned_minus_random_other",
            Cmp::Le,
            -0.2,
            Cmp::Ge,
            0.0,
        ),
        HypothesisSpec::new(
            "H6-grid-map-transfer",
            "Grid planner over a model learned on another map loses at most 0.15 goal rate versus an in-domain model",
            "grid_goal_rate_transfer_drop",
            Cmp::Le,
            0.15,
            Cmp::Ge,
            0.40,
        ),
    ]
}

/// Hash of [`emergence_hypotheses`] (what is stamped into the manifests before running).
pub fn emergence_hypotheses_hash() -> Result<Option<u64>, SpecError> {
    hypotheses_hash(&emergence_hypotheses())
}

/// Access level of an arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArmAccess {
    /// No environment information.
    NoInformation,
    /// Fair: knows task geometry only (no dynamics simulation).
    TaskGeometry,
    /// Fair: simulates futures with the *given* true dynamics model (known-model planner).
    KnownDynamics,
    /// Fair: knows nothing about dynamics or map; learns from executed transitions only.
    LearnedFromExperience,
    /// Privileged ceiling: more budget and/or the true simulator. Never part of fair comparisons.
    PrivilegedCeiling,
}

impl ArmAccess {
    /// Stable label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NoInformation => "no-information",
            Self::TaskGeometry => "task-geometry",
            Self::KnownDynamics => "known-dynamics",
            Self::LearnedFromExperience => "learned-from-experience",
            Self::PrivilegedCeiling => "PRIVILEGED-CEILING",
        }
    }
}

/// One row of the arm table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArmDescriptor {
    /// Environment family.
    pub family: &'static str,
    /// Arm identifier.
    pub arm: &'static str,
    /// What the arm observes.
    pub observes: &'static str,
    /// Access level.
    pub access: ArmAccess,
    /// Planning budget statement.
    pub budget: &'static str,
}

/// Arms that exist as comparable agents and are included.
pub fn arm_table() -> Vec<ArmDescriptor> {
    use ArmAccess::*;
    vec![
        ArmDescriptor {
            family: "kart",
            arm: "random",
            observes: "nothing",
            access: NoInformation,
            budget: "0 transitions",
        },
        ArmDescriptor {
            family: "kart",
            arm: "geometric_baseline",
            observes: "kinematic state + checkpoint positions",
            access: TaskGeometry,
            budget: "0 transitions",
        },
        ArmDescriptor {
            family: "kart",
            arm: "pure_reward_free_entropic",
            observes: "kinematic state; no reward/goal",
            access: KnownDynamics,
            budget: "72/decision (1x8x9)",
        },
        ArmDescriptor {
            family: "kart",
            arm: "goal_oriented_fmc",
            observes: "kinematic state + goal weights; true KartModel",
            access: KnownDynamics,
            budget: "72/decision (9x8)",
        },
        ArmDescriptor {
            family: "kart",
            arm: "learned_fmc",
            observes: "(yaw, speed) + collision outcome only; objective = predicted speed (no checkpoints, no map)",
            access: LearnedFromExperience,
            budget: "72/decision nominal (24x3 macro) after 45 exploration episodes",
        },
        ArmDescriptor {
            family: "kart",
            arm: "learned_prior_ablation",
            observes: "same as learned_fmc, zero training (revision-0 prior)",
            access: LearnedFromExperience,
            budget: "72/decision nominal, 0 episodes",
        },
        ArmDescriptor {
            family: "kart",
            arm: "learned_fmc_in_domain",
            observes: "same as learned_fmc, trained on the evaluation track (transfer reference)",
            access: LearnedFromExperience,
            budget: "72/decision nominal, 45 episodes on that track",
        },
        ArmDescriptor {
            family: "kart",
            arm: "privileged_fmc_ceiling",
            observes: "true KartModel, 4x budget",
            access: PrivilegedCeiling,
            budget: "288/decision",
        },
        ArmDescriptor {
            family: "grid",
            arm: "random",
            observes: "nothing",
            access: NoInformation,
            budget: "0 transitions",
        },
        ArmDescriptor {
            family: "grid",
            arm: "tabular_greedy",
            observes: "own cell + goal cell; learned tabular model, 1-step lookahead",
            access: LearnedFromExperience,
            budget: "800 experience transitions, 1 predict/action",
        },
        ArmDescriptor {
            family: "grid",
            arm: "planner_learned_fmc",
            observes: "own cell + goal cell; learned tabular model + FMC",
            access: LearnedFromExperience,
            budget: "160/decision after 800 experience transitions",
        },
        ArmDescriptor {
            family: "grid",
            arm: "planner_prior_ablation",
            observes: "as planner_learned_fmc with 0 experience",
            access: LearnedFromExperience,
            budget: "160/decision, 0 experience",
        },
        ArmDescriptor {
            family: "grid",
            arm: "privileged_fmc_ceiling",
            observes: "true GridWorldModel",
            access: PrivilegedCeiling,
            budget: "160/decision (same FMC, real simulator)",
        },
        // GOG-203 first slice: closed-loop arms that did not exist in the GOG-176 report.
        ArmDescriptor {
            family: "grid",
            arm: GRID_ENTROPIC_PURE_ARM,
            observes: "own cell only; true GridWorldModel for rollouts; no reward weight, no goal weight",
            access: KnownDynamics,
            budget: "160/decision (5 actions x 8 rollouts x 4 steps)",
        },
        ArmDescriptor {
            family: "grid",
            arm: GRID_EMPOWERMENT_ARM,
            observes: "own cell; tabular model frozen from executed transitions; no simulator, no reward",
            access: LearnedFromExperience,
            budget: "250/decision (5 x 10 x 5) after 800 experience transitions",
        },
        ArmDescriptor {
            family: "grid",
            arm: GRID_CAUSAL_RULE_ARM,
            observes: "own cell + induced typed causal rules; acts only with explicit ActionPermission",
            access: LearnedFromExperience,
            budget: "0 planning transitions; 800 experience transitions",
        },
    ]
}

/// Arms the task asked for that do **not** exist as comparable closed-loop agents (not invented).
pub fn missing_arms() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "known-model Grid FMC/MC arms in this report",
            "exist in run_arena_comparison (GridFmc, GridMonteCarlo, GridPrivilegedBfs) but are not re-run here; the Grid family uses the hidden-dynamics harness arms",
        ),
        (
            "kart_adaptive / multiscale / freedom_field arms",
            "exist but concern drift or macro-state analysis, not map/track transfer; not included in this slice",
        ),
        (
            "learned-arm scientific trace",
            "learned arms are executed by lab harnesses and produce no ScientificTraceArchive; the verdict trace_ref is the Kart-arena manifest stamped with the same hypotheses hash",
        ),
    ]
}

/// Aggregated closed-loop row for one arm on one track/map (holdout seeds only).
#[derive(Debug, Clone, PartialEq)]
pub struct ArmRow {
    /// Environment label (track or map).
    pub environment: String,
    /// Arm identifier.
    pub arm: String,
    /// Access level.
    pub access: ArmAccess,
    /// Episodes (denominator).
    pub episodes: usize,
    /// Episodes that reached the goal (all checkpoints / goal cell).
    pub goals: usize,
    /// Episodes ending in collision / hazard failure.
    pub failures: usize,
    /// Mean checkpoint efficacy (Kart) or goal rate (Grid).
    pub mean_efficacy: f64,
    /// Simulated transitions per decision (nominal for learned arms).
    pub transitions_per_decision: f64,
    /// Total simulated planning transitions over all episodes (nominal for learned arms).
    pub total_transitions: f64,
}

impl ArmRow {
    /// Failure rate (collisions / hazard failures per episode).
    pub fn failure_rate(&self) -> f64 {
        self.failures as f64 / self.episodes.max(1) as f64
    }
    /// Goal rate.
    pub fn goal_rate(&self) -> f64 {
        self.goals as f64 / self.episodes.max(1) as f64
    }
}

/// One-step model quality of a learner on held-out transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelQuality {
    /// Learner label.
    pub learner: String,
    /// Track whose transitions were evaluated.
    pub evaluated_on: String,
    /// Evaluated transitions.
    pub transitions: usize,
    /// Mean Brier score over the bucket distribution (collisions included).
    pub brier: f64,
    /// Mean abs. yaw error (rad).
    pub yaw_error: f64,
    /// Mean abs. speed error (m/s).
    pub speed_error: f64,
}

/// Full emergence/transfer report.
#[derive(Debug, Clone)]
pub struct EmergenceReport {
    /// Hash of the pre-registered hypotheses stamped before running.
    pub hypotheses_hash: Option<u64>,
    /// Kart arena on the training track (rectangular).
    pub kart_train_track: KartArenaReport,
    /// Kart arena on the unseen track (chicane).
    pub kart_other_track: KartArenaReport,
    /// Closed-loop rows (Kart then Grid), holdout seeds only.
    pub rows: Vec<ArmRow>,
    /// One-step model quality rows.
    pub model_quality: Vec<ModelQuality>,
    /// Exploration counters used to train the Kart learners (per training track).
    pub kart_training: Vec<(String, KartEpisodeCounters)>,
    /// Metrics evaluated by the hypotheses.
    pub metrics: BTreeMap<String, f64>,
}

impl EmergenceReport {
    fn row(&self, env: &str, arm: &str) -> Option<&ArmRow> {
        self.rows
            .iter()
            .find(|r| r.environment == env && r.arm == arm)
    }
}

const RECT: &str = "kart:rectangular";
const CHICANE: &str = "kart:chicane";
const GRID_A: &str = "grid:A";
const GRID_B: &str = "grid:B";

fn kart_rows_from_arena(env: &str, report: &KartArenaReport) -> Vec<ArmRow> {
    let mut rows = Vec::new();
    let mut roles: Vec<(KartRole, bool)> = report
        .fair_summaries
        .iter()
        .map(|s| (s.role, false))
        .collect();
    roles.extend(report.ceiling_summaries.iter().map(|s| (s.role, true)));
    for (role, _) in roles {
        let runs: Vec<&crate::kart::KartRunResult> = report
            .outcomes
            .iter()
            .filter(|o| o.split == SeedSplit::Holdout)
            .flat_map(|o| o.fair_runs.iter().chain(&o.ceiling_runs))
            .filter(|r| r.role == role)
            .collect();
        let n = runs.len().max(1) as f64;
        rows.push(ArmRow {
            environment: env.to_owned(),
            arm: role.label().to_owned(),
            access: match role {
                KartRole::Random => ArmAccess::NoInformation,
                KartRole::GeometricBaseline => ArmAccess::TaskGeometry,
                KartRole::PureEntropic | KartRole::GoalOrientedFmc => ArmAccess::KnownDynamics,
                KartRole::PrivilegedFmcCeiling => ArmAccess::PrivilegedCeiling,
            },
            episodes: runs.len(),
            goals: runs.iter().filter(|r| r.status.is_succeeded()).count(),
            failures: runs.iter().map(|r| r.collisions).sum(),
            mean_efficacy: runs.iter().map(|r| r.efficacy).sum::<f64>() / n,
            transitions_per_decision: runs.iter().map(|r| r.cost_per_decision).sum::<f64>() / n,
            total_transitions: runs.iter().map(|r| r.transitions_total as f64).sum(),
        });
    }
    rows
}

fn train_kart_agent(
    track: &TrackMap,
    episodes_per_seed: usize,
) -> Result<(KartLearnedAgent, KartEpisodeCounters), EmergenceError> {
    let mut counters = KartEpisodeCounters::default();
    let mut agent = KartLearnedAgent::new().map_err(lab)?;
    for seed in KART_LEARNED_TRAIN_SEEDS {
        let eps = explore_episodes(track, seed, episodes_per_seed, &mut counters).map_err(lab)?;
        for t in eps.iter().flatten() {
            agent.learn(t).map_err(lab)?;
        }
    }
    Ok((agent, counters))
}

fn eval_kart_learned(
    arm: &str,
    env: &str,
    agent: &KartLearnedAgent,
    track: &TrackMap,
) -> Result<ArmRow, EmergenceError> {
    let n_cp = track.checkpoints().len().max(1);
    let (mut goals, mut collisions, mut eff, mut decisions) = (0usize, 0usize, 0.0, 0u64);
    for seed in HOLDOUT_SEEDS {
        // Frozen evaluation: a clone per episode, `learn = false`.
        let mut a = agent.clone();
        let ep = run_controller_episode(&mut a, track, seed, false).map_err(lab)?;
        goals += usize::from(ep.checkpoints >= n_cp);
        collisions += usize::from(ep.collided);
        eff += ep.checkpoints as f64 / n_cp as f64;
        decisions += ep.macro_steps as u64;
    }
    let n = HOLDOUT_SEEDS.len();
    Ok(ArmRow {
        environment: env.to_owned(),
        arm: arm.to_owned(),
        access: ArmAccess::LearnedFromExperience,
        episodes: n,
        goals,
        failures: collisions,
        mean_efficacy: eff / n as f64,
        transitions_per_decision: KART_TRANSITIONS_PER_DECISION as f64,
        total_transitions: (decisions * KART_TRANSITIONS_PER_DECISION as u64) as f64,
    })
}

fn held_transitions(track: &TrackMap) -> Result<Vec<KartExecutedTransition>, EmergenceError> {
    let mut c = KartEpisodeCounters::default();
    let mut out = Vec::new();
    for seed in KART_LEARNED_HOLDOUT_SEEDS {
        let eps = explore_episodes(track, seed, KART_EPISODES_PER_SEED, &mut c).map_err(lab)?;
        out.extend(eps.into_iter().flatten());
    }
    Ok(out)
}

fn quality(
    learner: &str,
    on: &str,
    agent: &mut KartLearnedAgent,
    held: &[KartExecutedTransition],
) -> Result<ModelQuality, EmergenceError> {
    let model = agent.publish().map_err(lab)?;
    let e = agent.prediction_error(&model, held).map_err(lab)?;
    Ok(ModelQuality {
        learner: learner.to_owned(),
        evaluated_on: on.to_owned(),
        transitions: e.total,
        brier: e.mean_brier,
        yaw_error: e.mean_yaw_error,
        speed_error: e.mean_speed_error,
    })
}

fn grid_map_a() -> Result<GridMap, EmergenceError> {
    GridMap::from_ascii(&["#######", "#S.#.G#", "#..#H.#", "#.....#", "#######"]).map_err(lab)
}

fn grid_map_b() -> Result<GridMap, EmergenceError> {
    GridMap::from_ascii(&["#######", "#S..#G#", "#.#.H.#", "#.....#", "#######"]).map_err(lab)
}

fn grid_world(map: GridMap) -> GridWorldConfig {
    GridWorldConfig {
        map,
        hazard_is_terminal: true,
        ..GridWorldConfig::default()
    }
}

fn grid_row(env: &str, arm: &str, access: ArmAccess, eps: &[HiddenEpisode]) -> ArmRow {
    let n = eps.len().max(1) as f64;
    let decisions: u64 = eps.iter().map(|e| e.ticks).sum();
    let transitions: u64 = eps.iter().map(|e| e.planning_transitions).sum();
    let goals = eps.iter().filter(|e| e.reached_goal).count();
    ArmRow {
        environment: env.to_owned(),
        arm: arm.to_owned(),
        access,
        episodes: eps.len(),
        goals,
        failures: eps.iter().filter(|e| e.failed).count(),
        mean_efficacy: goals as f64 / n,
        transitions_per_decision: transitions as f64 / decisions.max(1) as f64,
        total_transitions: transitions as f64,
    }
}

/// Grid family: train on `train`, evaluate on `eval` (holdout seeds only).
fn grid_family(
    train: &GridWorldConfig,
    train_label: &str,
    eval: &GridWorldConfig,
    eval_label: &str,
    rows: &mut Vec<ArmRow>,
) -> Result<(), EmergenceError> {
    let curve = HiddenCurveConfig::new(eval.clone());
    let scale = grid_cell_scale(&train.map).map_err(lab)?;
    let in_domain = train_label == eval_label;
    let tag = |s: &str| format!("{s}[trained_on={train_label}]");
    let (mut plan_eps, mut greedy_eps, mut random_eps, mut ceil_eps, mut prior_eps) =
        (vec![], vec![], vec![], vec![], vec![]);
    for seed in &curve.holdout_seeds {
        let mut plan = curve.plan.clone();
        plan.seed = *seed;
        for (n, sink) in [(GRID_OBSERVATIONS, &mut plan_eps), (0, &mut prior_eps)] {
            let learner = collect_hidden_experience(
                train,
                &scale,
                n,
                *seed,
                LearningPolicy::new(curve.prior_alpha).map_err(lab)?,
            )
            .map_err(lab)?;
            let model = freeze_grid_model(&learner, scale.clone(), curve.model).map_err(lab)?;
            let agent = PlanGrid::new(model.clone(), plan.clone()).map_err(lab)?;
            sink.push(run_plan_grid_episode(eval, &agent, curve.max_ticks).map_err(lab)?);
            if n == GRID_OBSERVATIONS {
                greedy_eps.push(
                    run_learner_policy_episode(
                        eval,
                        &LearnerPolicy::GreedyOnLearner(&model, plan.task),
                        curve.max_ticks,
                    )
                    .map_err(lab)?,
                );
            }
        }
        random_eps.push(
            run_learner_policy_episode(eval, &LearnerPolicy::Random(*seed), curve.max_ticks)
                .map_err(lab)?,
        );
        ceil_eps.push(run_privileged_fmc_episode(eval, &plan, curve.max_ticks).map_err(lab)?);
    }
    use ArmAccess::*;
    rows.push(grid_row(
        eval_label,
        &tag("planner_learned_fmc"),
        LearnedFromExperience,
        &plan_eps,
    ));
    rows.push(grid_row(
        eval_label,
        &tag("tabular_greedy"),
        LearnedFromExperience,
        &greedy_eps,
    ));
    if in_domain {
        rows.push(grid_row(
            eval_label,
            "planner_prior_ablation",
            LearnedFromExperience,
            &prior_eps,
        ));
        rows.push(grid_row(eval_label, "random", NoInformation, &random_eps));
        rows.push(grid_row(
            eval_label,
            "privileged_fmc_ceiling",
            PrivilegedCeiling,
            &ceil_eps,
        ));
    }
    Ok(())
}

/// GOG-203 first slice: evaluates the three closed-loop Grid arms on `eval` (holdout seeds of the
/// hidden-dynamics harness), with experience collected on `train`.
///
/// Learning budgets match the other Grid learners (800 random transitions per holdout seed).
/// `entropic_pure` needs no experience (it rolls out over the true model and is only reported
/// in-domain); the other two rows are tagged with their training map so transfer rows are explicit.
pub fn run_grid_closed_loop_arms(
    train: &GridWorldConfig,
    train_label: &str,
    eval: &GridWorldConfig,
    eval_label: &str,
) -> Result<Vec<ArmRow>, EmergenceError> {
    let curve = HiddenCurveConfig::new(eval.clone());
    let entropic_cfg =
        RolloutConfig::new(GRID_ENTROPIC_POPULATION, GRID_ENTROPIC_HORIZON).map_err(lab)?;
    let empowerment_cfg =
        RolloutConfig::new(GRID_EMPOWERMENT_POPULATION, GRID_EMPOWERMENT_HORIZON).map_err(lab)?;
    let scale = grid_cell_scale(&train.map).map_err(lab)?;
    let (mut entropic, mut empower, mut causal) = (Vec::new(), Vec::new(), Vec::new());
    for seed in &curve.holdout_seeds {
        entropic.push(
            run_entropic_pure_grid_episode(eval, *seed, entropic_cfg, curve.max_ticks)
                .map_err(lab)?,
        );
        let learner = collect_hidden_experience(
            train,
            &scale,
            GRID_OBSERVATIONS,
            *seed,
            LearningPolicy::new(curve.prior_alpha).map_err(lab)?,
        )
        .map_err(lab)?;
        let model = freeze_grid_model(&learner, scale.clone(), curve.model).map_err(lab)?;
        empower.push(
            run_empowerment_episode(eval, &model, *seed, empowerment_cfg, curve.max_ticks)
                .map_err(lab)?,
        );
        let mut agent = GridCausalAgent::explore(train, GRID_OBSERVATIONS, *seed).map_err(lab)?;
        causal.push(run_causal_grid_episode(eval, &mut agent, curve.max_ticks).map_err(lab)?);
    }
    let tag = |s: &str| format!("{s}[trained_on={train_label}]");
    let mut rows = Vec::new();
    if train_label == eval_label {
        rows.push(grid_row(
            eval_label,
            GRID_ENTROPIC_PURE_ARM,
            ArmAccess::KnownDynamics,
            &entropic,
        ));
    }
    rows.push(grid_row(
        eval_label,
        &tag(GRID_EMPOWERMENT_ARM),
        ArmAccess::LearnedFromExperience,
        &empower,
    ));
    rows.push(grid_row(
        eval_label,
        &tag(GRID_CAUSAL_RULE_ARM),
        ArmAccess::LearnedFromExperience,
        &causal,
    ));
    Ok(rows)
}

/// Runs the whole pre-registered comparison once.
pub fn run_emergence_transfer() -> Result<EmergenceReport, EmergenceError> {
    let hash = emergence_hypotheses_hash()?;
    let rect = TrackMap::rectangular_circuit();
    let chicane = TrackMap::chicane_circuit();

    // Fair + ceiling arms through the existing Kart arena runner, hash stamped before running.
    let seeds = ArenaSeedSet::new(
        DEV_SEEDS.map(MasterSeed::new).to_vec(),
        HOLDOUT_SEEDS.map(MasterSeed::new).to_vec(),
    )?;
    let budget = KartArenaBudget::new(KART_MAX_TICKS, KART_HORIZON, KART_TRANSITIONS_PER_DECISION)?;
    let kart_train_track = run_kart_arena_preregistered(
        &KartArenaConfig::new("gog-176-rect", rect.clone(), budget, seeds.clone()),
        hash,
    )?;
    let kart_other_track = run_kart_arena_preregistered(
        &KartArenaConfig::new("gog-176-chicane", chicane.clone(), budget, seeds),
        hash,
    )?;

    let mut rows = kart_rows_from_arena(RECT, &kart_train_track);
    rows.extend(kart_rows_from_arena(CHICANE, &kart_other_track));

    // Learned arms: trained on the rectangular track only (transfer), plus references.
    let (mut learned_rect, c_rect) = train_kart_agent(&rect, KART_EPISODES_PER_SEED)?;
    let (mut learned_chic, c_chic) = train_kart_agent(&chicane, KART_EPISODES_PER_SEED)?;
    let mut prior = KartLearnedAgent::new().map_err(lab)?;
    rows.push(eval_kart_learned(
        "learned_fmc",
        RECT,
        &learned_rect,
        &rect,
    )?);
    rows.push(eval_kart_learned(
        "learned_prior_ablation",
        RECT,
        &prior,
        &rect,
    )?);
    rows.push(eval_kart_learned(
        "learned_fmc",
        CHICANE,
        &learned_rect,
        &chicane,
    )?);
    rows.push(eval_kart_learned(
        "learned_prior_ablation",
        CHICANE,
        &prior,
        &chicane,
    )?);
    rows.push(eval_kart_learned(
        "learned_fmc_in_domain",
        CHICANE,
        &learned_chic,
        &chicane,
    )?);

    let held_rect = held_transitions(&rect)?;
    let held_chic = held_transitions(&chicane)?;
    let model_quality = vec![
        quality("trained_on_rect", "rect", &mut learned_rect, &held_rect)?,
        quality("trained_on_rect", "chicane", &mut learned_rect, &held_chic)?,
        quality(
            "trained_on_chicane",
            "chicane",
            &mut learned_chic,
            &held_chic,
        )?,
        quality("prior_untrained", "rect", &mut prior, &held_rect)?,
        quality("prior_untrained", "chicane", &mut prior, &held_chic)?,
    ];

    // Grid family: map A <-> map B.
    let (wa, wb) = (grid_world(grid_map_a()?), grid_world(grid_map_b()?));
    grid_family(&wa, "grid:A", &wa, GRID_A, &mut rows)?;
    grid_family(&wb, "grid:B", &wb, GRID_B, &mut rows)?;
    grid_family(&wa, "grid:A", &wb, GRID_B, &mut rows)?;
    grid_family(&wb, "grid:B", &wa, GRID_A, &mut rows)?;

    // GOG-203 closed-loop arms: same holdout seeds and budgets, descriptive rows only.
    rows.extend(run_grid_closed_loop_arms(&wa, "grid:A", &wa, GRID_A)?);
    rows.extend(run_grid_closed_loop_arms(&wb, "grid:B", &wb, GRID_B)?);
    rows.extend(run_grid_closed_loop_arms(&wa, "grid:A", &wb, GRID_B)?);

    let mut report = EmergenceReport {
        hypotheses_hash: hash,
        kart_train_track,
        kart_other_track,
        rows,
        model_quality,
        kart_training: vec![
            ("rectangular".to_owned(), c_rect),
            ("chicane".to_owned(), c_chic),
        ],
        metrics: BTreeMap::new(),
    };
    report.metrics = observed_metrics(&report);
    Ok(report)
}

/// Derives the metrics named by [`emergence_hypotheses`] from the measured rows.
pub fn observed_metrics(r: &EmergenceReport) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::new();
    let eff = |env: &str, arm: &str| r.row(env, arm).map(|x| x.mean_efficacy);
    let fail = |env: &str, arm: &str| r.row(env, arm).map(ArmRow::failure_rate);
    if let (Some(l), Some(p)) = (
        eff(RECT, "learned_fmc"),
        eff(RECT, "learned_prior_ablation"),
    ) {
        m.insert("kart_learning_gain_efficacy_same".to_owned(), l - p);
    }
    let brier = |l: &str, on: &str| {
        r.model_quality
            .iter()
            .find(|q| q.learner == l && q.evaluated_on == on)
            .map(|q| q.brier)
    };
    if let (Some(o), Some(s)) = (
        brier("trained_on_rect", "chicane"),
        brier("trained_on_rect", "rect"),
    ) {
        m.insert("kart_brier_transfer_gap".to_owned(), o - s);
    }
    if let (Some(s), Some(o)) = (eff(RECT, "learned_fmc"), eff(CHICANE, "learned_fmc")) {
        m.insert("kart_efficacy_transfer_drop".to_owned(), s - o);
    }
    if let (Some(l), Some(p)) = (
        eff(CHICANE, "learned_fmc"),
        eff(CHICANE, "pure_reward_free_entropic"),
    ) {
        m.insert("kart_learned_minus_pure_efficacy_other".to_owned(), l - p);
    }
    if let (Some(l), Some(p)) = (fail(CHICANE, "learned_fmc"), fail(CHICANE, "random")) {
        m.insert(
            "kart_collision_rate_learned_minus_random_other".to_owned(),
            l - p,
        );
    }
    // Grid: eval on B, in-domain (trained on B) vs transferred (trained on A).
    let g_in = eff(GRID_B, "planner_learned_fmc[trained_on=grid:B]");
    let g_tr = eff(GRID_B, "planner_learned_fmc[trained_on=grid:A]");
    if let (Some(i), Some(t)) = (g_in, g_tr) {
        m.insert("grid_goal_rate_transfer_drop".to_owned(), i - t);
    }
    m
}

/// Adjudicates the hypotheses against the Kart-arena manifest stamped with the registered hash.
pub fn adjudicate(
    r: &EmergenceReport,
    build: BuildStatus,
) -> Result<Vec<VerdictEntry>, EmergenceError> {
    let manifest = r
        .kart_train_track
        .outcomes
        .first()
        .and_then(|o| o.fair_runs.first())
        .map(|x| &x.trace_manifest);
    let evidence = HypothesisEvidence {
        manifest,
        trace_ref: None,
        observed: r.metrics.clone(),
        build,
    };
    Ok(evaluate_hypotheses(&emergence_hypotheses(), &evidence)?)
}

/// Post-hoc caveats computed from the rows. They never change a verdict; they flag results that
/// are formally decided but weakly informative (floor effects, trivial baselines).
pub fn reading_notes(r: &EmergenceReport) -> Vec<String> {
    let mut notes = vec![
        "No AGI, universal-intelligence or emergence claim is made; verdicts only concern the pre-registered metrics on these tracks, maps and seeds.".to_owned(),
        "Privileged-ceiling rows are upper references and are excluded from every hypothesis metric.".to_owned(),
    ];
    let eff = |env: &str, arm: &str| r.row(env, arm).map(|x| x.mean_efficacy);
    if eff(RECT, "learned_fmc") == Some(0.0) && eff(CHICANE, "learned_fmc") == Some(0.0) {
        notes.push("FLOOR EFFECT: the learned Kart agent clears 0 checkpoints on both tracks, so any SUPPORTED transfer verdict on efficacy (e.g. H3: drop = 0) is vacuous: there is no competence to transfer. The agent observes only (yaw, speed) and optimizes predicted speed, so it cannot navigate checkpoints by construction.".to_owned());
    }
    if r.row(CHICANE, "random")
        .is_some_and(|x| x.failure_rate() >= 0.99)
    {
        notes.push("H5 compares against a random controller that collides in every episode; beating it is a weak safety result, not evidence of safe driving (the learned agent still collides in most episodes).".to_owned());
    }
    notes.push("Kart Brier is nearly identical on both tracks because the model state is (yaw, speed) with no position: it cannot encode track layout, so 'model transfer' here is expected from the abstraction, not a demonstrated scale transfer.".to_owned());
    notes.push("Grid: 6 holdout seeds and 800 random transitions; goal rates are small-count (one episode = 0.167) and differences of one episode are within noise.".to_owned());
    notes.push("GOG-203 slice: the grid entropic_pure, empowerment_learned and causal_rule_learned rows are descriptive; none of them enters the H1-H6 pre-registered metrics, and the hypothesis spec (hence the registered hash) was left untouched, so no verdict depends on the new arms.".to_owned());
    notes.push("GOG-203 access: entropic_pure rolls out over the true GridWorldModel (known-dynamics); empowerment_learned and causal_rule_learned use only executed experience (learned-from-experience) and act without simulator access. None of them reads reward constants as an objective and none is tuned on holdout seeds.".to_owned());
    for (env, trained_on) in [(GRID_A, "grid:A"), (GRID_B, "grid:A"), (GRID_B, "grid:B")] {
        for arm in [GRID_EMPOWERMENT_ARM, GRID_CAUSAL_RULE_ARM] {
            let name = format!("{arm}[trained_on={trained_on}]");
            if let Some(row) = r.row(env, &name)
                && row.goals == 0
            {
                notes.push(format!(
                    "NEGATIVE RESULT (GOG-203): {name} on {env} reached the goal in {} of {} episodes ({} hazard failures); recorded as measured, without retuning.",
                    row.goals, row.episodes, row.failures
                ));
            }
        }
    }
    notes
}

/// Renders the receipt-style Markdown report (verdicts are separate from the build status).
pub fn render_report(r: &EmergenceReport, verdicts: &[VerdictEntry]) -> String {
    let mut o = String::new();
    let _ = writeln!(
        o,
        "# GOG-176 / GOG-203 emergence and transferability (Arena)\n"
    );
    let _ = writeln!(
        o,
        "hypotheses_hash (registered before running): {}\n",
        r.hypotheses_hash
            .map_or("none".to_owned(), |h| format!("{h:016x}"))
    );
    let _ = writeln!(
        o,
        "GOG-203 slice: the closed-loop Grid arms entropic_pure, empowerment_learned and causal_rule_learned are included as descriptive rows below. The pre-registered H1-H6 spec and its hash are unchanged, so no verdict depends on them.\n"
    );
    let _ = writeln!(o, "## Arms (what each observes, access level)\n");
    let _ = writeln!(
        o,
        "| family | arm | observes | access | budget |\n|---|---|---|---|---|"
    );
    for a in arm_table() {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} |",
            a.family,
            a.arm,
            a.observes,
            a.access.label(),
            a.budget
        );
    }
    let _ = writeln!(o, "\n## Missing arms (not invented)\n");
    for (name, why) in missing_arms() {
        let _ = writeln!(o, "- {name}: {why}");
    }
    let _ = writeln!(
        o,
        "\n## Closed loop, holdout seeds only (Kart: {} seeds, max_ticks={KART_MAX_TICKS}; Grid: 6 seeds)\n",
        HOLDOUT_SEEDS.len()
    );
    let _ = writeln!(
        o,
        "| environment | arm | access | episodes | goals | failures(collision/hazard) | mean efficacy | transitions/decision | total transitions |\n|---|---|---|---|---|---|---|---|---|"
    );
    for x in &r.rows {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {:.3} | {:.1} | {:.0} |",
            x.environment,
            x.arm,
            x.access.label(),
            x.episodes,
            x.goals,
            x.failures,
            x.mean_efficacy,
            x.transitions_per_decision,
            x.total_transitions
        );
    }
    let _ = writeln!(
        o,
        "\nPrivileged ceiling rows are an upper reference, never part of the fair comparison. Learned-arm transitions are nominal (decisions x 72). Arena roles decide every tick, the learned arm every {KART_HOLD_TICKS} ticks."
    );
    let _ = writeln!(
        o,
        "\n## Learned-model quality on held-out random transitions (Kart)\n"
    );
    let _ = writeln!(
        o,
        "| learner | evaluated on | transitions | Brier | yaw err | speed err |\n|---|---|---|---|---|---|"
    );
    for q in &r.model_quality {
        let _ = writeln!(
            o,
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} |",
            q.learner, q.evaluated_on, q.transitions, q.brier, q.yaw_error, q.speed_error
        );
    }
    for (t, c) in &r.kart_training {
        let _ = writeln!(o, "\ntraining on {t}: {c:?}");
    }
    let _ = writeln!(o, "\n## Metrics\n");
    for (k, v) in &r.metrics {
        let _ = writeln!(o, "- {k} = {v:.4}");
    }
    let _ = writeln!(
        o,
        "\n## Reading notes (post-hoc interpretation; verdicts above are unchanged)\n"
    );
    for n in reading_notes(r) {
        let _ = writeln!(o, "- {n}");
    }
    let _ = writeln!(o, "\n## Verdicts (separate from build status)\n");
    o.push_str(&crate::hypothesis::render_verdict_table(verdicts));
    o
}

/// Training map of the grid transfer demo.
pub const GRID_DEMO_TRAIN: &str = "grid:A";
/// Holdout map of the grid transfer demo (never used for training in the transfer arm).
pub const GRID_DEMO_HOLDOUT: &str = "grid:B";

/// One episode of the grid transfer demo (a trace row; per-decision data is summarised).
#[derive(Debug, Clone, PartialEq)]
pub struct GridDemoEpisode {
    /// Arm label.
    pub arm: String,
    /// Map the model was trained on (`none` for model-free arms).
    pub trained_on: String,
    /// Map where the episode ran.
    pub evaluated_on: String,
    /// Holdout seed.
    pub seed: u64,
    /// Goal reached.
    pub reached_goal: bool,
    /// Terminal failure (hazard).
    pub failed: bool,
    /// Ticks executed.
    pub ticks: u64,
    /// Simulated planning transitions consumed.
    pub planning_transitions: u64,
    /// Model revision used (planning arms).
    pub model_revision: Option<u64>,
    /// Model checksum used (planning arms).
    pub model_checksum: Option<u64>,
    /// Decisions that fell back to `Stay` (planning arms).
    pub fallback_decisions: u64,
    /// Decisions traced (planning arms).
    pub decisions: u64,
    /// Replayable per-decision trace: `(tick, action_index, transitions_used, fallback_label)`.
    pub decision_trace: Vec<(u64, usize, usize, Option<&'static str>)>,
}

/// Result of [`run_grid_transfer_demo`].
#[derive(Debug, Clone, PartialEq)]
pub struct GridTransferDemo {
    /// Holdout seeds used.
    pub holdout_seeds: Vec<u64>,
    /// Random transitions the learners observe per seed.
    pub observations: usize,
    /// Ticks per episode (limit).
    pub max_ticks: u64,
    /// Simulated transitions per decision (limit).
    pub budget_transitions_per_decision: usize,
    /// Every episode in a stable order.
    pub episodes: Vec<GridDemoEpisode>,
}

impl GridTransferDemo {
    /// `(goals, episodes)` of an arm trained on `trained_on` and evaluated on the holdout map.
    pub fn goals(&self, arm: &str, trained_on: &str) -> (usize, usize) {
        let sel = self
            .episodes
            .iter()
            .filter(|e| e.arm == arm && e.trained_on == trained_on);
        (sel.clone().filter(|e| e.reached_goal).count(), sel.count())
    }
}

/// Grid transfer demo: the learned model (tabular, cell abstraction) is trained from random
/// experience on map A only and then plans on the unseen map B with holdout seeds, next to
/// baselines (random, untrained prior, tabular-greedy) and an in-domain reference. The planner
/// never sees a simulator; the privileged ceiling is reported separately and never compared.
pub fn run_grid_transfer_demo() -> Result<GridTransferDemo, EmergenceError> {
    let (wa, wb) = (grid_world(grid_map_a()?), grid_world(grid_map_b()?));
    let curve = HiddenCurveConfig::new(wb.clone());
    let mut episodes = Vec::new();
    let mut push =
        |arm: &str, trained: &str, seed: u64, e: &HiddenEpisode, model: Option<(u64, u64)>| {
            episodes.push(GridDemoEpisode {
                arm: arm.to_owned(),
                trained_on: trained.to_owned(),
                evaluated_on: GRID_DEMO_HOLDOUT.to_owned(),
                seed,
                reached_goal: e.reached_goal,
                failed: e.failed,
                ticks: e.ticks,
                planning_transitions: e.planning_transitions,
                model_revision: model.map(|m| m.0),
                model_checksum: model.map(|m| m.1),
                fallback_decisions: e.traces.iter().filter(|t| t.fallback.is_some()).count() as u64,
                decisions: e.traces.len() as u64,
                decision_trace: e
                    .traces
                    .iter()
                    .map(|t| {
                        (
                            t.tick,
                            t.action_index,
                            t.transitions_used,
                            t.fallback.map(|f| f.label()),
                        )
                    })
                    .collect(),
            });
        };
    for ms in &curve.holdout_seeds {
        let seed = ms.as_u64();
        let mut plan = curve.plan.clone();
        plan.seed = *ms;
        let planners: [(&str, &GridWorldConfig, &str, usize); 3] = [
            (
                "planner_learned_fmc",
                &wa,
                GRID_DEMO_TRAIN,
                GRID_OBSERVATIONS,
            ),
            ("planner_prior_ablation", &wa, "none", 0),
            (
                "planner_learned_fmc",
                &wb,
                GRID_DEMO_HOLDOUT,
                GRID_OBSERVATIONS,
            ),
        ];
        for (arm, train, label, n) in planners {
            let scale = grid_cell_scale(&train.map).map_err(lab)?;
            let learner = collect_hidden_experience(
                train,
                &scale,
                n,
                *ms,
                LearningPolicy::new(curve.prior_alpha).map_err(lab)?,
            )
            .map_err(lab)?;
            let model = freeze_grid_model(&learner, scale, curve.model).map_err(lab)?;
            let version = (model.version().revision(), model.version().checksum());
            let agent = PlanGrid::new(model.clone(), plan.clone()).map_err(lab)?;
            let ep = run_plan_grid_episode(&wb, &agent, curve.max_ticks).map_err(lab)?;
            push(arm, label, seed, &ep, Some(version));
            if arm == "planner_learned_fmc" && label == GRID_DEMO_TRAIN {
                let g = run_learner_policy_episode(
                    &wb,
                    &LearnerPolicy::GreedyOnLearner(&model, plan.task),
                    curve.max_ticks,
                )
                .map_err(lab)?;
                push("tabular_greedy", label, seed, &g, Some(version));
            }
        }
        let r = run_learner_policy_episode(&wb, &LearnerPolicy::Random(*ms), curve.max_ticks)
            .map_err(lab)?;
        push("random", "none", seed, &r, None);
        let c = run_privileged_fmc_episode(&wb, &plan, curve.max_ticks).map_err(lab)?;
        push("privileged_fmc_ceiling", "none", seed, &c, None);
    }
    Ok(GridTransferDemo {
        holdout_seeds: curve.holdout_seeds.iter().map(|s| s.as_u64()).collect(),
        observations: GRID_OBSERVATIONS,
        max_ticks: curve.max_ticks,
        budget_transitions_per_decision: curve.plan.budget_transitions,
        episodes,
    })
}

#[cfg(test)]
mod tests {
    use crate::grid_arms::GOG_203_ARMS;
    use crate::hypothesis::HypothesisVerdict;

    use super::*;

    #[test]
    fn emergence_transfer_hypotheses_are_hashed_and_complete() {
        let specs = emergence_hypotheses();
        assert_eq!(specs.len(), 6);
        let h = emergence_hypotheses_hash().unwrap();
        assert!(h.is_some());
        assert_eq!(h, emergence_hypotheses_hash().unwrap());
        let mut edited = specs;
        edited[0].threshold += 0.01;
        assert_ne!(h, hypotheses_hash(&edited).unwrap());
    }

    #[test]
    fn emergence_transfer_arm_table_declares_access_and_missing() {
        let t = arm_table();
        assert!(t.iter().any(|a| a.access == ArmAccess::PrivilegedCeiling));
        assert!(
            t.iter()
                .any(|a| a.access == ArmAccess::LearnedFromExperience)
        );
        let names: Vec<_> = missing_arms().iter().map(|m| m.0).collect();
        assert!(!names.contains(&"empowerment agent"));
        assert!(!names.contains(&"causal-rule learning agent"));
        assert!(!names.contains(&"pure entropic on Grid"));
        assert!(names.contains(&"kart_adaptive / multiscale / freedom_field arms"));
    }

    /// GOG-203: the three first-slice arms exist with controlled access and left `missing_arms`
    /// only for the genuinely out-of-scope entries.
    #[test]
    fn emergence_gog_203_arms_are_registered_with_controlled_access() {
        let t = arm_table();
        let get = |arm: &str| {
            t.iter()
                .find(|a| a.family == "grid" && a.arm == arm)
                .copied()
                .unwrap_or_else(|| panic!("arm {arm} missing from arm_table"))
        };
        let entropic = get(GRID_ENTROPIC_PURE_ARM);
        assert_eq!(entropic.access, ArmAccess::KnownDynamics);
        assert!(entropic.budget.contains("160/decision"));
        let empowerment = get(GRID_EMPOWERMENT_ARM);
        assert_eq!(empowerment.access, ArmAccess::LearnedFromExperience);
        assert!(empowerment.budget.contains("800 experience transitions"));
        let causal = get(GRID_CAUSAL_RULE_ARM);
        assert_eq!(causal.access, ArmAccess::LearnedFromExperience);
        assert!(causal.budget.contains("0 planning transitions"));
        // Only the genuinely out-of-scope entries stay documented as missing.
        let names: Vec<_> = missing_arms().iter().map(|m| m.0).collect();
        assert!(names.contains(&"kart_adaptive / multiscale / freedom_field arms"));
        assert!(names.contains(&"learned-arm scientific trace"));
        assert_eq!(
            GOG_203_ARMS,
            [
                GRID_ENTROPIC_PURE_ARM,
                GRID_EMPOWERMENT_ARM,
                GRID_CAUSAL_RULE_ARM
            ]
        );
    }

    /// GOG-203: the new arms run on the same holdout seeds with intact denominators, are
    /// deterministic, and their transfer rows are explicitly tagged.
    #[test]
    fn emergence_gog_203_rows_keep_denominators_and_are_deterministic() {
        let (wa, wb) = (
            grid_world(grid_map_a().unwrap()),
            grid_world(grid_map_b().unwrap()),
        );
        let rows = run_grid_closed_loop_arms(&wa, "grid:A", &wa, GRID_A).unwrap();
        assert_eq!(rows.len(), 3);
        for row in &rows {
            assert_eq!(row.episodes, 6, "holdout denominator intact: {row:?}");
            assert!(row.mean_efficacy.is_finite());
            assert!(row.goals <= row.episodes);
            assert!(row.failures <= row.episodes);
        }
        let entropic = rows
            .iter()
            .find(|r| r.arm == GRID_ENTROPIC_PURE_ARM)
            .unwrap();
        assert_eq!(entropic.access, ArmAccess::KnownDynamics);
        assert!((entropic.transitions_per_decision - 160.0).abs() < 1e-9);
        assert!(entropic.total_transitions > 0.0);
        let causal = rows
            .iter()
            .find(|r| r.arm.starts_with(GRID_CAUSAL_RULE_ARM))
            .unwrap();
        assert_eq!(causal.total_transitions, 0.0);
        assert_eq!(causal.transitions_per_decision, 0.0);
        // Same seeds and budgets produce exactly the same rows.
        assert_eq!(
            run_grid_closed_loop_arms(&wa, "grid:A", &wa, GRID_A).unwrap(),
            rows
        );
        // Transfer rows (trained on A, evaluated on B) are explicit; entropic is in-domain only.
        let transfer = run_grid_closed_loop_arms(&wa, "grid:A", &wb, GRID_B).unwrap();
        assert_eq!(transfer.len(), 2);
        assert!(transfer.iter().all(|r| r.environment == GRID_B));
        assert!(
            transfer
                .iter()
                .any(|r| r.arm == format!("{GRID_EMPOWERMENT_ARM}[trained_on=grid:A]"))
        );
        assert!(
            transfer
                .iter()
                .any(|r| r.arm == format!("{GRID_CAUSAL_RULE_ARM}[trained_on=grid:A]"))
        );
    }

    /// GOG-203: H1-H6 are re-evaluated against the unchanged pre-registered hash. Verdicts are
    /// produced from the manifest hash (never from the build status); incomplete metric evidence
    /// stays `INCONCLUSIVE`, and a missing registration never becomes `SUPPORTED`.
    #[test]
    fn emergence_gog_203_hypotheses_reevaluated_with_same_hash() {
        let hash = emergence_hypotheses_hash().unwrap();
        assert!(hash.is_some(), "the pre-registered hash must exist");
        let ids: Vec<String> = emergence_hypotheses()
            .iter()
            .map(|h| h.hypothesis_id.clone())
            .collect();
        assert_eq!(
            ids,
            vec![
                "H1-learning-helps",
                "H2-model-transfers",
                "H3-control-transfers",
                "H4-beats-reward-free",
                "H5-safer-than-random",
                "H6-grid-map-transfer",
            ]
        );
        // Executed harness run with the hash stamped before running.
        let seeds =
            ArenaSeedSet::new(vec![MasterSeed::new(7)], vec![MasterSeed::new(101)]).unwrap();
        let report = run_kart_arena_preregistered(
            &KartArenaConfig::new(
                "gog-203-reeval",
                TrackMap::rectangular_circuit(),
                KartArenaBudget::new(24, 4, 36).unwrap(),
                seeds,
            ),
            hash,
        )
        .unwrap();
        let manifest = report
            .outcomes
            .first()
            .and_then(|o| o.fair_runs.first())
            .map(|r| &r.trace_manifest);
        assert_eq!(manifest.and_then(|m| m.hypotheses_hash), hash);
        let evidence = HypothesisEvidence {
            manifest,
            trace_ref: None,
            observed: BTreeMap::new(),
            build: BuildStatus::Pass,
        };
        let verdicts = evaluate_hypotheses(&emergence_hypotheses(), &evidence).unwrap();
        assert_eq!(verdicts.len(), 6);
        assert!(
            verdicts
                .iter()
                .all(|v| v.registered_hash == hash && v.spec_hash == hash)
        );
        assert!(
            verdicts
                .iter()
                .all(|v| v.verdict == HypothesisVerdict::Inconclusive),
            "missing metrics are honestly inconclusive"
        );
        // No manifest: INCONCLUSIVE even for a metric that would satisfy support.
        let none = HypothesisEvidence {
            manifest: None,
            trace_ref: None,
            observed: BTreeMap::from([("kart_learning_gain_efficacy_same".to_owned(), 1.0)]),
            build: BuildStatus::Pass,
        };
        let v = evaluate_hypotheses(&emergence_hypotheses(), &none).unwrap();
        assert!(
            v.iter()
                .all(|x| x.verdict == HypothesisVerdict::Inconclusive)
        );
    }

    #[test]
    fn emergence_transfer_verdict_is_inconclusive_when_not_preregistered() {
        // Evidence without a manifest can never be SUPPORTED.
        let ev = HypothesisEvidence {
            manifest: None,
            trace_ref: None,
            observed: BTreeMap::from([("kart_brier_transfer_gap".to_owned(), 0.0)]),
            build: BuildStatus::Pass,
        };
        let rows = evaluate_hypotheses(&emergence_hypotheses(), &ev).unwrap();
        assert!(
            rows.iter()
                .all(|r| r.verdict == crate::hypothesis::HypothesisVerdict::Inconclusive)
        );
    }

    /// Full experiment, run once on demand:
    /// `cargo test -p gogamen-arena --offline --release emergence_transfer_full -- --ignored --nocapture`
    #[test]
    #[ignore = "long experiment; prints the real report"]
    fn emergence_transfer_full() {
        let r = run_emergence_transfer().unwrap();
        let again = run_emergence_transfer().unwrap();
        assert_eq!(r.metrics, again.metrics, "experiment must be deterministic");
        assert_eq!(r.rows, again.rows, "arm rows must be deterministic");
        assert_eq!(
            r.kart_train_track.report_hash(),
            again.kart_train_track.report_hash()
        );
        let v = adjudicate(&r, BuildStatus::Pass).unwrap();
        println!("{}", render_report(&r, &v));
    }
}
