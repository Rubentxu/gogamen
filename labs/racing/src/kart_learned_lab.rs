//! Environment-side harness for the learned Kart controller (`kart_learned`).
//!
//! This is the only place where the real physics ([`KartEnv`]) is touched. It executes macro
//! actions in the live environment, converts the result into [`KartObservable`] /
//! [`KartStepOutcome`] and hands only those to the agent.
//!
//! # Pre-declared protocol (fixed before any result was inspected)
//! - Training seeds: [`KART_LEARNED_TRAIN_SEEDS`] (`DEV_SEEDS`); holdout seeds:
//!   [`KART_LEARNED_HOLDOUT_SEEDS`] (`UNSEEN_SEEDS`). Seeds drive the random exploration of macro
//!   actions (the environment itself is deterministic).
//! - Holdout transitions are executed in `KartEnv` on the training circuit (rectangular) and on
//!   a second circuit (chicane) that is never trained on.
//! - Versions are frozen after [`KART_LEARNED_VERSION_EPISODES`] cumulative training episodes.
//! - Error: mean abs. circular yaw error, mean abs. speed error of the predictive mean, and
//!   Brier score over all buckets (collisions included). Reference baselines: the revision-0
//!   prior and "nothing changes" persistence.
//!
//! Nothing here measures or claims racing performance.

use std::time::Instant;

use gogamen_core::{
    Action, Actuator, EnvironmentObserver, LiveEnvironment, RngSource, StepDt, Terminal,
};

use crate::autonomous_fmc::{DEV_SEEDS, UNSEEN_SEEDS};
use crate::kart_learned::{
    KartExecutedTransition, KartLearnedAgent, KartLearnedError, KartMacroAction, KartObservable,
    KartPredictionError, KartStepOutcome, KartUpdateCost, KartVersionRecord, persistence_error,
};
use crate::physics::KinematicBicycleConfig;
use crate::track::TrackMap;
use crate::world::KartEnv;

/// Training seeds (declared in advance).
pub const KART_LEARNED_TRAIN_SEEDS: [u64; 3] = DEV_SEEDS;
/// Reserve (holdout) seeds (declared in advance, never trained on).
pub const KART_LEARNED_HOLDOUT_SEEDS: [u64; 3] = UNSEEN_SEEDS;
/// Cumulative training episodes at which a model version is frozen.
pub const KART_LEARNED_VERSION_EPISODES: [usize; 4] = [0, 3, 12, 45];
/// Environment ticks each macro action is held.
pub const KART_HOLD_TICKS: usize = 8;
/// Maximum macro steps per episode.
pub const KART_MAX_MACRO_STEPS: usize = 40;
/// Episodes per seed (train: 15 x 3 seeds = 45; holdout: same layout).
pub const KART_EPISODES_PER_SEED: usize = 15;

fn lab_err(msg: &'static str) -> KartLearnedError {
    KartLearnedError::new(msg)
}

pub(crate) fn observable_of(env: &KartEnv) -> Result<KartObservable, KartLearnedError> {
    let obs = EnvironmentObserver::observe(env).map_err(|_| lab_err("observe failed"))?;
    let f = obs.features();
    KartObservable::new(f[2], f[3])
}

/// Result of executing one macro action in the live environment.
pub(crate) struct MacroResult {
    pub(crate) outcome: KartStepOutcome,
    pub(crate) episode_over: bool,
}

/// Executes a macro action. Commands are scaled by the `nominal` (design-time) limits and then
/// clamped to what the live environment accepts, like a saturating actuator.
pub(crate) fn execute_macro(
    env: &mut KartEnv,
    action: KartMacroAction,
    hold_ticks: usize,
    nominal: &KinematicBicycleConfig,
) -> Result<MacroResult, KartLearnedError> {
    let cfg = env.config().clone();
    let steer = (action.steer_fraction() * nominal.max_steering_angle)
        .clamp(-cfg.max_steering_angle, cfg.max_steering_angle);
    let accel_frac = action.accel_fraction();
    let accel = if accel_frac >= 0.0 {
        accel_frac * nominal.max_accel
    } else {
        -accel_frac * nominal.min_accel
    }
    .clamp(cfg.min_accel, cfg.max_accel);
    let command = Action::continuous_in_space(vec![steer, accel], Actuator::action_space(env))
        .map_err(|_| lab_err("action out of range"))?;
    let mut terminal = Terminal::Ongoing;
    for _ in 0..hold_ticks {
        let applied = env.apply(&command).map_err(|_| lab_err("apply failed"))?;
        terminal = applied.terminal();
        if terminal.is_terminal() {
            break;
        }
    }
    let outcome = if env.collided() {
        KartStepOutcome::Collided
    } else {
        KartStepOutcome::Continued(observable_of(env)?)
    };
    Ok(MacroResult {
        outcome,
        episode_over: terminal.is_terminal(),
    })
}

fn pick_random(rng: &mut gogamen_core::DeterministicRng) -> KartMacroAction {
    let idx = (rng.next_f64() * KartMacroAction::ALL.len() as f64) as usize;
    KartMacroAction::ALL[idx.min(KartMacroAction::ALL.len() - 1)]
}

/// Aggregate counters of executed episodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KartEpisodeCounters {
    /// Episodes run.
    pub episodes: usize,
    /// Episodes that ended in collision (failed episodes).
    pub collisions: usize,
    /// Executed macro transitions.
    pub transitions: usize,
    /// Episodes that reached the goal (all checkpoints).
    pub goals: usize,
}

/// Executes random-exploration episodes in `KartEnv`; returns the executed transitions.
pub fn explore_episodes(
    track: &TrackMap,
    seed: u64,
    episodes: usize,
    counters: &mut KartEpisodeCounters,
) -> Result<Vec<Vec<KartExecutedTransition>>, KartLearnedError> {
    explore_episodes_with(
        &KinematicBicycleConfig::default(),
        track,
        seed,
        episodes,
        counters,
    )
}

/// Like [`explore_episodes`] but with explicit live dynamics (evaluator-side only).
pub(crate) fn explore_episodes_with(
    dynamics: &KinematicBicycleConfig,
    track: &TrackMap,
    seed: u64,
    episodes: usize,
    counters: &mut KartEpisodeCounters,
) -> Result<Vec<Vec<KartExecutedTransition>>, KartLearnedError> {
    let mut out = Vec::with_capacity(episodes);
    for ep in 0..episodes {
        let mut env = KartEnv::new(dynamics.clone(), track.clone())
            .map_err(|_| lab_err("env build failed"))?;
        LiveEnvironment::reset(&mut env).map_err(|_| lab_err("reset failed"))?;
        let mut rng =
            gogamen_core::DeterministicRng::new(seed.wrapping_mul(1_000).wrapping_add(ep as u64));
        let mut steps = Vec::new();
        counters.episodes += 1;
        for _ in 0..KART_MAX_MACRO_STEPS {
            let before = observable_of(&env)?;
            let action = pick_random(&mut rng);
            let result = execute_macro(
                &mut env,
                action,
                KART_HOLD_TICKS,
                &KinematicBicycleConfig::default(),
            )?;
            steps.push(KartExecutedTransition {
                before,
                action,
                outcome: result.outcome,
            });
            counters.transitions += 1;
            if result.outcome == KartStepOutcome::Collided {
                counters.collisions += 1;
            }
            if result.episode_over {
                if env.checkpoints_cleared() >= env.track().checkpoints().len() {
                    counters.goals += 1;
                }
                break;
            }
        }
        out.push(steps);
    }
    Ok(out)
}

/// Statistics of one controller-driven episode (measured, no claim implied).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KartControllerEpisode {
    /// Macro steps executed.
    pub macro_steps: usize,
    /// Whether the episode ended in collision.
    pub collided: bool,
    /// Checkpoints cleared.
    pub checkpoints: usize,
    /// Number of model versions published during the episode.
    pub versions_published: usize,
}

/// Runs the learned controller in `KartEnv`, learning online from what it executes.
pub fn run_controller_episode(
    agent: &mut KartLearnedAgent,
    track: &TrackMap,
    seed: u64,
    learn: bool,
) -> Result<KartControllerEpisode, KartLearnedError> {
    let cfg = KinematicBicycleConfig::default();
    let macro_dt = StepDt::from_secs_f64(cfg.dt.as_secs_f64() * KART_HOLD_TICKS as f64)
        .map_err(|_| lab_err("macro dt invalid"))?;
    let mut env = KartEnv::new(cfg, track.clone()).map_err(|_| lab_err("env build failed"))?;
    LiveEnvironment::reset(&mut env).map_err(|_| lab_err("reset failed"))?;
    let mut model = agent.publish()?;
    let mut published = 1;
    let mut macro_steps = 0;
    for step in 0..KART_MAX_MACRO_STEPS {
        let before = observable_of(&env)?;
        let action =
            KartLearnedAgent::decide(&model, before, macro_dt, seed.wrapping_add(step as u64))?;
        let result = execute_macro(
            &mut env,
            action,
            KART_HOLD_TICKS,
            &KinematicBicycleConfig::default(),
        )?;
        macro_steps += 1;
        if learn {
            agent.learn(&KartExecutedTransition {
                before,
                action,
                outcome: result.outcome,
            })?;
            if step % 10 == 9 {
                model = agent.publish()?;
                published += 1;
            }
        }
        if result.episode_over {
            break;
        }
    }
    Ok(KartControllerEpisode {
        macro_steps,
        collided: env.collided(),
        checkpoints: env.checkpoints_cleared(),
        versions_published: published,
    })
}

/// Error of one version on one holdout set.
#[derive(Debug, Clone, PartialEq)]
pub struct KartVersionEval {
    /// Training episodes consumed before this version was frozen.
    pub train_episodes: usize,
    /// Prediction error on the rectangular (trained circuit) holdout.
    pub same_circuit: KartPredictionError,
    /// Prediction error on the chicane (never trained) holdout.
    pub other_circuit: KartPredictionError,
}

/// Full experiment report.
#[derive(Debug, Clone, PartialEq)]
pub struct KartLearnedReport {
    /// Per-version holdout errors (first entry is the prior, revision 0).
    pub versions: Vec<KartVersionEval>,
    /// Persistence baseline `(yaw, speed)` error on the same-circuit holdout.
    pub persistence_same: (f64, f64),
    /// Persistence baseline `(yaw, speed)` error on the other-circuit holdout.
    pub persistence_other: (f64, f64),
    /// Training counters.
    pub train: KartEpisodeCounters,
    /// Holdout counters (same circuit).
    pub holdout_same: KartEpisodeCounters,
    /// Holdout counters (other circuit).
    pub holdout_other: KartEpisodeCounters,
    /// Update cost.
    pub cost: KartUpdateCost,
    /// Published versions.
    pub version_records: Vec<KartVersionRecord>,
    /// Controller episodes with the final model, learning online (holdout seeds).
    pub controller: Vec<KartControllerEpisode>,
    /// Wall-clock total of the experiment in milliseconds (informational).
    pub wall_millis: u128,
}

impl KartLearnedReport {
    /// Human-readable summary; reports measurements only.
    pub fn summary(&self) -> String {
        let mut s = String::from("kart_learned report\n");
        for v in &self.versions {
            for (name, e) in [("same", &v.same_circuit), ("other", &v.other_circuit)] {
                s.push_str(&format!(
                    "  rev={} train_eps={} [{}] n={} yaw_err={:.4} speed_err={:.4} brier={:.4} uncertain={:.2}\n",
                    e.revision, v.train_episodes, name, e.total, e.mean_yaw_error,
                    e.mean_speed_error, e.mean_brier, e.uncertain_fraction
                ));
            }
        }
        s.push_str(&format!(
            "  persistence same yaw={:.4} speed={:.4}; other yaw={:.4} speed={:.4}\n",
            self.persistence_same.0,
            self.persistence_same.1,
            self.persistence_other.0,
            self.persistence_other.1
        ));
        s.push_str(&format!(
            "  train {:?}\n  holdout_same {:?}\n  holdout_other {:?}\n  cost {:?} versions {:?}\n  controller {:?}\n  wall_ms={}\n",
            self.train, self.holdout_same, self.holdout_other, self.cost,
            self.version_records, self.controller, self.wall_millis
        ));
        s
    }
}

fn flatten(eps: Vec<Vec<KartExecutedTransition>>) -> Vec<KartExecutedTransition> {
    eps.into_iter().flatten().collect()
}

/// Runs the pre-declared protocol end to end.
pub fn run_kart_learned_experiment() -> Result<KartLearnedReport, KartLearnedError> {
    let started = Instant::now();
    let rect = TrackMap::rectangular_circuit();
    let chicane = TrackMap::chicane_circuit();

    // Holdout transitions (executed in KartEnv with reserve seeds).
    let mut holdout_same_c = KartEpisodeCounters::default();
    let mut holdout_other_c = KartEpisodeCounters::default();
    let mut held_same = Vec::new();
    let mut held_other = Vec::new();
    for seed in KART_LEARNED_HOLDOUT_SEEDS {
        held_same.extend(flatten(explore_episodes(
            &rect,
            seed,
            KART_EPISODES_PER_SEED,
            &mut holdout_same_c,
        )?));
        held_other.extend(flatten(explore_episodes(
            &chicane,
            seed,
            KART_EPISODES_PER_SEED,
            &mut holdout_other_c,
        )?));
    }

    // Training episodes, in a fixed order across seeds.
    let mut train_c = KartEpisodeCounters::default();
    let mut train_eps: Vec<Vec<KartExecutedTransition>> = Vec::new();
    for seed in KART_LEARNED_TRAIN_SEEDS {
        train_eps.extend(explore_episodes(
            &rect,
            seed,
            KART_EPISODES_PER_SEED,
            &mut train_c,
        )?);
    }

    let mut agent = KartLearnedAgent::new()?;
    let mut versions = Vec::new();
    let mut consumed = 0usize;
    for &target in &KART_LEARNED_VERSION_EPISODES {
        while consumed < target {
            for t in &train_eps[consumed] {
                agent.learn(t)?;
            }
            consumed += 1;
        }
        let model = agent.publish()?;
        versions.push(KartVersionEval {
            train_episodes: consumed,
            same_circuit: agent.prediction_error(&model, &held_same)?,
            other_circuit: agent.prediction_error(&model, &held_other)?,
        });
    }
    let cost = agent.cost();

    // Controller episodes (online learning continues; versions above are already frozen).
    let mut controller = Vec::new();
    for seed in KART_LEARNED_HOLDOUT_SEEDS {
        controller.push(run_controller_episode(&mut agent, &rect, seed, true)?);
    }

    Ok(KartLearnedReport {
        versions,
        persistence_same: persistence_error(&held_same),
        persistence_other: persistence_error(&held_other),
        train: train_c,
        holdout_same: holdout_same_c,
        holdout_other: holdout_other_c,
        cost,
        version_records: agent.versions().to_vec(),
        controller,
        wall_millis: started.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const AGENT_SRC: &str = include_str!("kart_learned.rs");

    #[test]
    fn kart_learned_agent_source_has_no_privileged_tokens() {
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
            "world",
            "wall",
            "use crate::",
        ] {
            assert!(
                !lowered.contains(token),
                "agent source must not mention privileged token `{token}`"
            );
        }
    }

    #[test]
    fn kart_learned_prediction_improves_over_prior_on_executed_holdout() {
        let report = run_kart_learned_experiment().unwrap();
        println!("{}", report.summary());
        let prior = &report.versions[0];
        let last = report.versions.last().unwrap();
        assert_eq!(prior.same_circuit.revision, 0);
        assert!(last.same_circuit.revision > 0);
        assert!(last.same_circuit.total > 100, "holdout must be non-trivial");
        assert!(last.same_circuit.mean_brier < prior.same_circuit.mean_brier);
        assert!(last.same_circuit.mean_yaw_error < prior.same_circuit.mean_yaw_error);
        assert!(last.same_circuit.mean_speed_error < prior.same_circuit.mean_speed_error);
        // Counters are recorded, and the update count equals the executed training transitions.
        assert_eq!(report.cost.updates as usize, report.train.transitions);
        assert!(
            report.train.collisions > 0,
            "collisions must be observed and recorded"
        );
        assert!(report.version_records.len() >= KART_LEARNED_VERSION_EPISODES.len());
    }

    #[test]
    fn kart_learned_experiment_is_deterministic_apart_from_timing() {
        let mut a = run_kart_learned_experiment().unwrap();
        let mut b = run_kart_learned_experiment().unwrap();
        a.wall_millis = 0;
        b.wall_millis = 0;
        a.cost.total_nanos = 0;
        b.cost.total_nanos = 0;
        assert_eq!(a, b);
    }

    #[test]
    fn kart_learned_controller_uses_only_valid_actions() {
        let mut agent = KartLearnedAgent::new().unwrap();
        let ep =
            run_controller_episode(&mut agent, &TrackMap::rectangular_circuit(), 7, true).unwrap();
        assert!(ep.macro_steps >= 1);
        assert!(agent.cost().updates as usize >= 1);
    }
}
