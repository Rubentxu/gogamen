//! Environment-side experiment for the drift-aware Kart agent (`kart_adaptive`).
//!
//! Only this module (and `kart_learned_lab`) touches the live `KartEnv`. The evaluator changes
//! the live dynamics at an episode boundary that *it* knows; the agent only receives observables
//! and step outcomes and is never told about the change.
//!
//! # Pre-declared protocol (fixed before any holdout result was inspected)
//! - Pre-change training: 45 random-exploration episodes on the rectangular circuit
//!   ([`KART_LEARNED_TRAIN_SEEDS`](crate::kart_learned_lab::KART_LEARNED_TRAIN_SEEDS), as GOG-173).
//! - Calibration stream (pre-change dynamics, rectangular): [`CALIBRATION_SEEDS`] x
//!   [`STREAM_EPISODES_PER_SEED`] episodes; the change happens right after it.
//! - Adaptation stream (post-change dynamics, rectangular): `DEV_SEEDS` x
//!   [`STREAM_EPISODES_PER_SEED`] episodes, truncated to the scenario `budget` (transitions).
//!   All modes receive the very same transitions. If the budget exceeds that stream it is extended
//!   with [`EXTRA_ADAPT_SEEDS`] (also never a reserve seed).
//! - Reserve seeds `UNSEEN_SEEDS` (never used to train, calibrate or adapt) drive the held-out
//!   post-change transitions ([`HELD_EPISODES_PER_SEED`] episodes per seed, rectangular) and the
//!   planner seeds of the closed-loop episodes on the never-trained chicane circuit.
//! - Recovery criterion: the model's Brier error on the held-out post-change transitions is
//!   `<= RECOVERY_FACTOR x` the pre-change model's Brier error on pre-change held-out
//!   transitions; checked every [`CHECK_EVERY`] adaptation transitions.
//! - Detector parameters: `KartAdaptConfig::default()` (v2: mean-error signal, baseline = whole
//!   calibration stream, window 12, z 3, peak rule, margin 0.05).
//!
//! # Design change between v1 and v2 (declared, measured)
//! v1 used the single-transition Brier score with `baseline_len 60`, `window 12`, margin 0.15 and
//! threshold `mean + max(margin, z*se)`. Measured (reserve-seed numbers, `v1` run): the detector
//! fired once *before* the change in every scenario (false alarm at step 122, on the pre-change
//! stream), never after it, and the Brier error of the frozen model moved only 0.646 -> 0.737 for
//! the x2 steering change, so the recovery bound (1.25 x pre-change) was met by the frozen model
//! itself (vacuous criterion). v2 changes: signal = yaw error + speed error / 10 of the
//! predictive mean, baseline = the whole 350-step calibration stream, threshold also floored at
//! the highest rolling mean seen while calibrating, margin 0.05. The v2 detector parameters were
//! chosen with `detection_probe`, which only reads calibration and `DEV_SEEDS` streams (never
//! reserve seeds); the recovery criterion moved to the same signal. Quantisation of the model
//! (8 yaw x 6 speed bins) was NOT changed.
//!
//! Known limits: the post-change stream and the held-out set are drawn from the same circuit,
//! the environment is deterministic given the macro-action sequence, and the closed-loop episodes
//! are only 3 (one per reserve seed). Closed-loop numbers are measurements, not a performance claim.

use gogamen_core::{LiveEnvironment, StepDt};

use crate::autonomous_fmc::{DEV_SEEDS, UNSEEN_SEEDS};
use crate::kart_adaptive::{
    KartAdaptConfig, KartAdaptMode, KartAdaptiveAgent, KartAdaptiveVersion, KartSurpriseSignal,
    KartVersionReason, surprise_of,
};
use crate::kart_learned::{
    KartExecutedTransition, KartLearnedAgent, KartLearnedError, KartObservable, persistence_error,
};
use crate::kart_learned_lab::{
    KART_HOLD_TICKS, KART_LEARNED_TRAIN_SEEDS, KART_MAX_MACRO_STEPS, KartEpisodeCounters,
    execute_macro, explore_episodes_with, observable_of,
};
use crate::physics::KinematicBicycleConfig;
use crate::track::TrackMap;
use crate::world::KartEnv;

/// Seeds of the pre-change calibration stream (declared in advance).
pub const CALIBRATION_SEEDS: [u64; 3] = [11, 23, 47];
/// Extra adaptation seeds, appended after the `DEV_SEEDS` stream when a budget exceeds it.
pub const EXTRA_ADAPT_SEEDS: [u64; 6] = [1007, 1019, 1042, 2007, 2019, 2042];
/// Episodes per seed in the calibration and adaptation streams.
pub const STREAM_EPISODES_PER_SEED: usize = 5;
/// Episodes per reserve seed in held-out transition sets.
pub const HELD_EPISODES_PER_SEED: usize = 5;
/// Recovery is checked every this many adaptation transitions.
pub const CHECK_EVERY: usize = 10;
/// Allowed error ratio vs. the pre-change model on pre-change data to count as recovered.
pub const RECOVERY_FACTOR: f64 = 1.25;
/// Default adaptation budget (post-change transitions).
pub const DEFAULT_BUDGET: usize = 300;
/// Required calibration length (must equal `KartAdaptConfig::default().baseline_len`).
pub const CALIBRATION_LEN: usize = 350;

/// Change of the live dynamics, chosen by the evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KartDynamicsChange {
    /// Control: dynamics stay as they were (checks false alarms).
    NoChange,
    /// Steering-to-rotation ratio doubled (wheelbase halved).
    SteeringRatioDoubled,
    /// Heavier / lower-grip kart: accel limits x0.4 and drag x6.
    HeavyLowGrip,
    /// Steering-to-rotation ratio x12 (wheelbase / 12): deliberately beyond what the
    /// yaw/speed abstraction can follow (expected failure case).
    SteeringRatioExtreme,
    /// Steering-to-rotation ratio x4 (wheelbase / 4).
    SteeringRatioX4,
}

impl KartDynamicsChange {
    /// Live dynamics after the change.
    pub fn dynamics(self) -> KinematicBicycleConfig {
        let mut c = KinematicBicycleConfig::default();
        match self {
            Self::NoChange => {}
            Self::SteeringRatioDoubled => c.wheelbase *= 0.5,
            Self::HeavyLowGrip => {
                c.max_accel *= 0.4;
                c.min_accel *= 0.4;
                c.linear_drag *= 6.0;
            }
            Self::SteeringRatioExtreme => c.wheelbase /= 12.0,
            Self::SteeringRatioX4 => c.wheelbase /= 4.0,
        }
        c
    }
}

/// One declared scenario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartAdaptScenario {
    /// Short name.
    pub name: &'static str,
    /// Evaluator-side change.
    pub change: KartDynamicsChange,
    /// Post-change transitions the agents may consume.
    pub budget: usize,
}

/// The pre-declared scenarios.
pub const KART_ADAPT_SCENARIOS: [KartAdaptScenario; 7] = [
    KartAdaptScenario {
        name: "no_change_control",
        change: KartDynamicsChange::NoChange,
        budget: DEFAULT_BUDGET,
    },
    KartAdaptScenario {
        name: "steering_ratio_x2",
        change: KartDynamicsChange::SteeringRatioDoubled,
        budget: DEFAULT_BUDGET,
    },
    KartAdaptScenario {
        name: "heavy_low_grip",
        change: KartDynamicsChange::HeavyLowGrip,
        budget: DEFAULT_BUDGET,
    },
    KartAdaptScenario {
        name: "steering_ratio_x4",
        change: KartDynamicsChange::SteeringRatioX4,
        budget: DEFAULT_BUDGET,
    },
    KartAdaptScenario {
        name: "steering_ratio_x4_tiny_budget",
        change: KartDynamicsChange::SteeringRatioX4,
        budget: 20,
    },
    KartAdaptScenario {
        name: "steering_ratio_x4_large_budget",
        change: KartDynamicsChange::SteeringRatioX4,
        budget: 1500,
    },
    KartAdaptScenario {
        name: "steering_ratio_x12_extreme",
        change: KartDynamicsChange::SteeringRatioExtreme,
        budget: DEFAULT_BUDGET,
    },
];

/// Closed-loop statistics over the reserve seeds on the chicane circuit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartClosedLoopStats {
    /// Episodes run.
    pub episodes: usize,
    /// Episodes ending in collision.
    pub collisions: usize,
    /// Mean checkpoints cleared.
    pub mean_checkpoints: f64,
    /// Mean macro steps survived.
    pub mean_macro_steps: f64,
}

/// One error checkpoint on the held-out post-change set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartErrorPoint {
    /// Adaptation transitions consumed.
    pub consumed: usize,
    /// Brier error.
    pub brier: f64,
    /// Mean abs. circular yaw error (rad).
    pub yaw: f64,
    /// Mean abs. speed error (m/s).
    pub speed: f64,
    /// Detector signal (yaw + speed / 10), the recovery criterion.
    pub surprise: f64,
}

/// Result of one mode in one scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct KartModeResult {
    /// Learning policy.
    pub mode: KartAdaptMode,
    /// Alarm indices that fired before the change (false alarms).
    pub false_alarms: usize,
    /// Transitions after the change until the first alarm (1 = on the first post-change one).
    pub detection_delay: Option<usize>,
    /// Error trace on the held-out post-change set.
    pub trace: Vec<KartErrorPoint>,
    /// Adaptation transitions until the Brier bound was met (first checkpoint), if ever.
    pub recovery_delay: Option<usize>,
    /// Published versions (including the initial one).
    pub versions: Vec<KartAdaptiveVersion>,
    /// Closed-loop result on the unseen chicane with post-change dynamics.
    pub chicane: KartClosedLoopStats,
}

/// Result of one scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct KartScenarioResult {
    /// Scenario.
    pub scenario: KartAdaptScenario,
    /// Transitions actually consumed after the change.
    pub consumed: usize,
    /// Persistence `(yaw, speed)` error on the held-out post-change set.
    pub persistence: (f64, f64),
    /// Per-mode results (Frozen, ForgetOnDrift, AccumulateNoForget).
    pub modes: Vec<KartModeResult>,
}

/// Full report.
#[derive(Debug, Clone, PartialEq)]
pub struct KartAdaptReport {
    /// Detector-signal error of the pre-change model on pre-change held-out data.
    pub prechange_surprise: f64,
    /// Bound (RECOVERY_FACTOR x pre-change error) used as recovery criterion.
    pub recovery_bound: f64,
    /// Calibration stream length (the change happens right after it).
    pub calibration_len: usize,
    /// Closed loop of the pre-change model on the chicane with pre-change dynamics (reference).
    pub chicane_reference: KartClosedLoopStats,
    /// Scenario results.
    pub scenarios: Vec<KartScenarioResult>,
}

impl KartAdaptReport {
    /// Human-readable measurements only.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "kart_adaptive report\n  prechange_err={:.4} bound={:.4} calibration_len={}\n  chicane reference (pre-change model, pre-change dynamics) {:?}\n",
            self.prechange_surprise,
            self.recovery_bound,
            self.calibration_len,
            self.chicane_reference
        );
        for sc in &self.scenarios {
            s.push_str(&format!(
                "  scenario {} budget={} consumed={} persistence yaw={:.3} speed={:.3}\n",
                sc.scenario.name,
                sc.scenario.budget,
                sc.consumed,
                sc.persistence.0,
                sc.persistence.1
            ));
            for m in &sc.modes {
                let first = m.trace.first().map(|p| p.surprise).unwrap_or(f64::NAN);
                let last = m.trace.last().copied();
                let relearns = m
                    .versions
                    .iter()
                    .filter(|v| v.reason == KartVersionReason::DriftRelearn)
                    .count();
                s.push_str(&format!(
                    "    {:?}: false_alarms={} detect_delay={:?} recovery_delay={:?} err {:.3}->{:.3} yaw={:.3} speed={:.3} versions={} relearns={} chicane={:?}\n",
                    m.mode,
                    m.false_alarms,
                    m.detection_delay,
                    m.recovery_delay,
                    first,
                    last.map(|p| p.surprise).unwrap_or(f64::NAN),
                    last.map(|p| p.yaw).unwrap_or(f64::NAN),
                    last.map(|p| p.speed).unwrap_or(f64::NAN),
                    m.versions.len(),
                    relearns,
                    m.chicane
                ));
            }
        }
        s
    }
}

fn flatten(eps: Vec<Vec<KartExecutedTransition>>) -> Vec<KartExecutedTransition> {
    eps.into_iter().flatten().collect()
}

fn stream(
    dynamics: &KinematicBicycleConfig,
    track: &TrackMap,
    seeds: &[u64],
    episodes: usize,
) -> Result<Vec<KartExecutedTransition>, KartLearnedError> {
    let mut c = KartEpisodeCounters::default();
    let mut out = Vec::new();
    for &seed in seeds {
        out.extend(flatten(explore_episodes_with(
            dynamics, track, seed, episodes, &mut c,
        )?));
    }
    Ok(out)
}

fn closed_loop(
    agent: &KartAdaptiveAgent,
    dynamics: &KinematicBicycleConfig,
    track: &TrackMap,
) -> Result<KartClosedLoopStats, KartLearnedError> {
    let nominal = KinematicBicycleConfig::default();
    let macro_dt = StepDt::from_secs_f64(nominal.dt.as_secs_f64() * KART_HOLD_TICKS as f64)
        .map_err(|_| KartLearnedError::new("macro dt invalid"))?;
    let (mut coll, mut cps, mut steps) = (0usize, 0usize, 0usize);
    for seed in UNSEEN_SEEDS {
        let mut env = KartEnv::new(dynamics.clone(), track.clone())
            .map_err(|_| KartLearnedError::new("env build failed"))?;
        LiveEnvironment::reset(&mut env).map_err(|_| KartLearnedError::new("reset failed"))?;
        for step in 0..KART_MAX_MACRO_STEPS {
            let before: KartObservable = observable_of(&env)?;
            let action = agent.decide(before, macro_dt, seed.wrapping_add(step as u64))?;
            let r = execute_macro(&mut env, action, KART_HOLD_TICKS, &nominal)?;
            steps += 1;
            if r.episode_over {
                break;
            }
        }
        coll += usize::from(env.collided());
        cps += env.checkpoints_cleared();
    }
    let n = UNSEEN_SEEDS.len() as f64;
    Ok(KartClosedLoopStats {
        episodes: UNSEEN_SEEDS.len(),
        collisions: coll,
        mean_checkpoints: cps as f64 / n,
        mean_macro_steps: steps as f64 / n,
    })
}

fn point(
    agent: &KartAdaptiveAgent,
    consumed: usize,
    held: &[KartExecutedTransition],
) -> Result<KartErrorPoint, KartLearnedError> {
    let e = agent.prediction_error(held)?;
    Ok(KartErrorPoint {
        consumed,
        brier: e.mean_brier,
        yaw: e.mean_yaw_error,
        speed: e.mean_speed_error,
        surprise: surprise_of(KartSurpriseSignal::MeanError, &e),
    })
}

#[allow(clippy::too_many_arguments)]
fn run_mode(
    trained: &KartLearnedAgent,
    mode: KartAdaptMode,
    calibration: &[KartExecutedTransition],
    post: &[KartExecutedTransition],
    held_post: &[KartExecutedTransition],
    bound: f64,
    post_dynamics: &KinematicBicycleConfig,
    chicane: &TrackMap,
) -> Result<KartModeResult, KartLearnedError> {
    let mut agent = KartAdaptiveAgent::new(trained.clone(), mode, KartAdaptConfig::default())?;
    for t in calibration {
        agent.observe(t)?;
    }
    let false_alarms = agent.alarms().len();
    let mut trace = vec![point(&agent, 0, held_post)?];
    for (i, t) in post.iter().enumerate() {
        agent.observe(t)?;
        let consumed = i + 1;
        if consumed % CHECK_EVERY == 0 || consumed == post.len() {
            trace.push(point(&agent, consumed, held_post)?);
        }
    }
    let detection_delay = agent
        .alarms()
        .iter()
        .find(|a| a.at_step >= calibration.len())
        .map(|a| a.at_step - calibration.len() + 1);
    let recovery_delay = trace
        .iter()
        .find(|p| p.surprise <= bound)
        .map(|p| p.consumed);
    let chicane_stats = closed_loop(&agent, post_dynamics, chicane)?;
    Ok(KartModeResult {
        mode,
        false_alarms,
        detection_delay,
        trace,
        recovery_delay,
        versions: agent.versions().to_vec(),
        chicane: chicane_stats,
    })
}

/// Runs the whole pre-declared protocol.
pub fn run_kart_adaptive_experiment() -> Result<KartAdaptReport, KartLearnedError> {
    let rect = TrackMap::rectangular_circuit();
    let chicane = TrackMap::chicane_circuit();
    let nominal = KinematicBicycleConfig::default();

    // Pre-change training (same layout as GOG-173).
    let mut trained = KartLearnedAgent::new()?;
    let mut c = KartEpisodeCounters::default();
    for seed in KART_LEARNED_TRAIN_SEEDS {
        for ep in explore_episodes_with(&nominal, &rect, seed, 15, &mut c)? {
            for t in &ep {
                trained.learn(t)?;
            }
        }
    }
    let pre_model = trained.publish()?;
    let held_pre = stream(&nominal, &rect, &UNSEEN_SEEDS, HELD_EPISODES_PER_SEED)?;
    let prechange_surprise = surprise_of(
        KartSurpriseSignal::MeanError,
        &trained.prediction_error(&pre_model, &held_pre)?,
    );
    let recovery_bound = RECOVERY_FACTOR * prechange_surprise;

    let calibration = stream(
        &nominal,
        &rect,
        &CALIBRATION_SEEDS,
        STREAM_EPISODES_PER_SEED,
    )?;
    if calibration.len() != CALIBRATION_LEN {
        return Err(KartLearnedError::new("calibration stream length changed"));
    }
    let reference = KartAdaptiveAgent::new(
        trained.clone(),
        KartAdaptMode::Frozen,
        KartAdaptConfig::default(),
    )?;
    let chicane_reference = closed_loop(&reference, &nominal, &chicane)?;

    let mut scenarios = Vec::new();
    for scenario in KART_ADAPT_SCENARIOS {
        let dyn_post = scenario.change.dynamics();
        let mut post = stream(&dyn_post, &rect, &DEV_SEEDS, STREAM_EPISODES_PER_SEED)?;
        if post.len() < scenario.budget {
            post.extend(stream(
                &dyn_post,
                &rect,
                &EXTRA_ADAPT_SEEDS,
                STREAM_EPISODES_PER_SEED,
            )?);
        }
        post.truncate(scenario.budget);
        let held_post = stream(&dyn_post, &rect, &UNSEEN_SEEDS, HELD_EPISODES_PER_SEED)?;
        let mut modes = Vec::new();
        for mode in [
            KartAdaptMode::Frozen,
            KartAdaptMode::ForgetOnDrift,
            KartAdaptMode::AccumulateNoForget,
        ] {
            modes.push(run_mode(
                &trained,
                mode,
                &calibration,
                &post,
                &held_post,
                recovery_bound,
                &dyn_post,
                &chicane,
            )?);
        }
        scenarios.push(KartScenarioResult {
            scenario,
            consumed: post.len(),
            persistence: persistence_error(&held_post),
            modes,
        });
    }
    Ok(KartAdaptReport {
        prechange_surprise,
        recovery_bound,
        calibration_len: calibration.len(),
        chicane_reference,
        scenarios,
    })
}

/// Detection-only probe: uses only calibration and adaptation streams (never reserve seeds).
/// Returns `(false_alarm_step, detection_delay)` for one change and one detector config.
pub fn detection_probe(
    cfg: KartAdaptConfig,
    change: KartDynamicsChange,
) -> Result<(Option<usize>, Option<usize>), KartLearnedError> {
    let rect = TrackMap::rectangular_circuit();
    let nominal = KinematicBicycleConfig::default();
    let mut trained = KartLearnedAgent::new()?;
    let mut c = KartEpisodeCounters::default();
    for seed in KART_LEARNED_TRAIN_SEEDS {
        for ep in explore_episodes_with(&nominal, &rect, seed, 15, &mut c)? {
            for t in &ep {
                trained.learn(t)?;
            }
        }
    }
    let cal = stream(
        &nominal,
        &rect,
        &CALIBRATION_SEEDS,
        STREAM_EPISODES_PER_SEED,
    )?;
    let post = stream(
        &change.dynamics(),
        &rect,
        &DEV_SEEDS,
        STREAM_EPISODES_PER_SEED,
    )?;
    let mut agent = KartAdaptiveAgent::new(trained, KartAdaptMode::ForgetOnDrift, cfg)?;
    for t in &cal {
        agent.observe(t)?;
    }
    let false_alarm = agent.alarms().first().map(|a| a.at_step);
    for t in &post {
        agent.observe(t)?;
    }
    let delay = agent
        .alarms()
        .iter()
        .find(|a| a.at_step >= cal.len())
        .map(|a| a.at_step - cal.len() + 1);
    Ok((false_alarm, delay))
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT_SRC: &str = include_str!("kart_adaptive.rs");

    #[test]
    fn kart_adaptive_agent_source_has_no_privileged_tokens() {
        let lowered = AGENT_SRC.to_lowercase();
        for token in [
            "kartmodel",
            "kartenv",
            "trackmap",
            "liveenvironment",
            "actuator",
            "environmentobserver",
            "branchablesnapshot",
            "integrate_bicycle",
            "compute_bicycle",
            "kinematicbicycle",
            "kartstate",
            "kartcontrol",
            "track",
            "checkpoint",
            "crate::world",
            "crate::track",
            "crate::physics",
            "crate::kart_adaptive_lab",
            "crate::kart_learned_lab",
            "world",
            "wall",
            "wheelbase",
            "linear_drag",
            "max_accel",
            "dynamicschange",
            "scenario",
            "secret",
        ] {
            assert!(
                !lowered.contains(token),
                "adaptive agent source must not mention privileged token `{token}`"
            );
        }
        // The only crate-internal import allowed is the observation-only GOG-173 agent module.
        for (idx, _) in lowered.match_indices("crate::") {
            assert!(
                lowered[idx..].starts_with("crate::kart_learned::"),
                "only `crate::kart_learned::` may be imported"
            );
        }
    }

    fn report() -> &'static KartAdaptReport {
        static REPORT: std::sync::OnceLock<KartAdaptReport> = std::sync::OnceLock::new();
        REPORT.get_or_init(|| run_kart_adaptive_experiment().unwrap())
    }

    fn mode(r: &KartAdaptReport, scenario: &str, m: KartAdaptMode) -> KartModeResult {
        r.scenarios
            .iter()
            .find(|s| s.scenario.name == scenario)
            .unwrap_or_else(|| panic!("scenario {scenario}"))
            .modes
            .iter()
            .find(|x| x.mode == m)
            .unwrap()
            .clone()
    }

    #[test]
    fn kart_adaptive_experiment_is_deterministic() {
        let again = run_kart_adaptive_experiment().unwrap();
        assert_eq!(report(), &again);
    }

    #[test]
    fn kart_adaptive_no_false_alarms_and_frozen_never_changes() {
        let r = report();
        println!("{}", r.summary());
        for sc in &r.scenarios {
            for m in &sc.modes {
                assert_eq!(m.false_alarms, 0, "{} {:?}", sc.scenario.name, m.mode);
            }
            let frozen = mode(r, sc.scenario.name, KartAdaptMode::Frozen);
            assert_eq!(
                frozen.versions.len(),
                1,
                "frozen controller never republishes"
            );
            assert_eq!(frozen.detection_delay, None);
            let first = frozen.trace.first().unwrap().surprise;
            assert!(frozen.trace.iter().all(|p| p.surprise == first));
        }
        let control = mode(r, "no_change_control", KartAdaptMode::ForgetOnDrift);
        assert_eq!(control.detection_delay, None, "no alarm without a change");
        assert_eq!(control.versions.len(), 1);
    }

    #[test]
    fn kart_adaptive_detects_large_changes_from_own_error_only() {
        let r = report();
        for (name, max_delay) in [
            ("steering_ratio_x4", 40),
            ("steering_ratio_x12_extreme", 40),
        ] {
            let m = mode(r, name, KartAdaptMode::ForgetOnDrift);
            let d = m
                .detection_delay
                .unwrap_or_else(|| panic!("{name} undetected"));
            assert!(d <= max_delay, "{name}: delay {d}");
            assert!(
                m.versions
                    .iter()
                    .any(|v| v.reason == KartVersionReason::DriftRelearn && v.generation == 1),
                "a new model generation must be published after the alarm"
            );
        }
        // Detection-only probe (calibration + DEV streams only) agrees with the report.
        let (fa, delay) = detection_probe(
            KartAdaptConfig::default(),
            KartDynamicsChange::SteeringRatioX4,
        )
        .unwrap();
        assert_eq!(fa, None);
        assert_eq!(delay, Some(22));
    }

    /// Documented failures (measured, pinned so that they cannot silently become claims):
    /// small changes are not detected, a too-small budget prevents adaptation, and no scenario
    /// reaches the declared recovery bound with the forgetting policy.
    #[test]
    fn kart_adaptive_reports_failures_and_no_recovery() {
        let r = report();
        for name in ["steering_ratio_x2", "heavy_low_grip"] {
            let m = mode(r, name, KartAdaptMode::ForgetOnDrift);
            assert_eq!(m.detection_delay, None, "{name}: change not detected by v2");
            assert_eq!(m.versions.len(), 1);
        }
        let tiny = mode(
            r,
            "steering_ratio_x4_tiny_budget",
            KartAdaptMode::ForgetOnDrift,
        );
        assert_eq!(
            tiny.detection_delay, None,
            "budget shorter than detection delay"
        );
        assert_eq!(tiny.recovery_delay, None);
        for name in [
            "steering_ratio_x4",
            "steering_ratio_x4_large_budget",
            "steering_ratio_x12_extreme",
        ] {
            let m = mode(r, name, KartAdaptMode::ForgetOnDrift);
            assert_eq!(m.recovery_delay, None, "{name}: recovery not demonstrated");
        }
        // The extreme change leaves the model far above the bound for every policy.
        for m in &r.scenarios.last().unwrap().modes {
            assert!(m.trace.last().unwrap().surprise > 1.5 * r.recovery_bound);
        }
    }

    #[test]
    fn kart_adaptive_more_data_reduces_error_without_recovering() {
        let r = report();
        let forget = mode(
            r,
            "steering_ratio_x4_large_budget",
            KartAdaptMode::ForgetOnDrift,
        );
        let frozen = mode(r, "steering_ratio_x4_large_budget", KartAdaptMode::Frozen);
        let last = forget.trace.last().unwrap().surprise;
        assert!(last < frozen.trace.last().unwrap().surprise);
        assert!(last > r.recovery_bound);
        // Closed loop: no policy cleared a checkpoint on the unseen chicane (no performance claim).
        for sc in &r.scenarios {
            for m in &sc.modes {
                assert_eq!(m.chicane.mean_checkpoints, 0.0);
            }
        }
    }
}
