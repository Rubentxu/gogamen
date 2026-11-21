//! Reproducible episode metrics. Everything except `wall_secs`/`fps` is a
//! pure function of (port, policy, config) and is covered by `trace_hash`.

use crate::policy::Policy;
use crate::port::EmulatorPort;
use crate::rom::sha256_hex;
use std::time::Instant;

#[derive(Debug, Clone, Copy)]
pub struct RunConfig {
    pub seed: u64,
    /// Maximum live decisions (actions) per episode.
    pub max_decisions: u32,
    /// Frames each action is held.
    pub frames_per_action: u32,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            max_decisions: 600,
            frames_per_action: 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeMetrics {
    pub policy: &'static str,
    pub backend: &'static str,
    pub synthetic: bool,
    /// Furthest x reached.
    pub max_x: i32,
    pub deaths: u32,
    /// Live actions applied.
    pub actions_used: u32,
    /// Planning cost: decision-steps simulated on snapshots (0 for baselines).
    pub planning_steps: u64,
    /// Live frames emulated.
    pub live_frames: u64,
    pub level_complete: bool,
    /// SHA-256 over the (pad, x, y, deaths) trace; deterministic.
    pub trace_hash: String,
    /// Wall clock seconds for the whole episode incl. planning (not hashed).
    pub wall_secs: f64,
    /// Live frames per wall second (not hashed).
    pub fps: f64,
}

/// Run one episode from reset and gather metrics.
pub fn run_episode(
    port: &mut dyn EmulatorPort,
    policy: &mut dyn Policy,
    cfg: &RunConfig,
) -> EpisodeMetrics {
    port.reset();
    let mut trace = Vec::new();
    let mut max_x = 0;
    let mut actions = 0u32;
    let start = Instant::now();
    while actions < cfg.max_decisions && !port.features().is_terminal() {
        let pad = policy.act(port, cfg.frames_per_action);
        let r = port.step(pad, cfg.frames_per_action);
        actions += 1;
        let f = r.features;
        max_x = max_x.max(f.x);
        trace.push(pad.0);
        trace.extend_from_slice(&f.x.to_le_bytes());
        trace.extend_from_slice(&f.y.to_le_bytes());
        trace.extend_from_slice(&f.deaths.to_le_bytes());
    }
    let wall = start.elapsed().as_secs_f64();
    let f = port.features();
    EpisodeMetrics {
        policy: policy.name(),
        backend: port.backend_name(),
        synthetic: port.is_synthetic(),
        max_x,
        deaths: f.deaths,
        actions_used: actions,
        planning_steps: policy.sim_steps(),
        live_frames: f.frame,
        level_complete: f.level_complete,
        trace_hash: sha256_hex(&trace),
        wall_secs: wall,
        fps: if wall > 0.0 {
            f.frame as f64 / wall
        } else {
            0.0
        },
    }
}
