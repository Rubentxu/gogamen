//! Classic PD go-to-goal controller vs. a budgeted sampling planner on the engine port.
//!
//! The sampler is an "FMC-style" fork/restore planner: each simulation restores the root
//! snapshot, rolls a candidate control sequence forward and scores it. It deliberately does
//! NOT reuse `gogamen_core::FmcPlanner`: that planner is built on the discrete-action
//! `GenerativeModel`/`State`/`Action` types (walker population, virtual-reward cloning),
//! while this port exposes continuous `qpos/qvel/ctrl`. Bridging would require a discretised
//! action adapter and core types that this lab (and the core-free dependency invariant)
//! avoids. The comparison is therefore "classic vs. simulation-budgeted random-shooting +
//! refinement", not a claim about the full FMC algorithm.

use std::time::Instant;

use crate::port::{ContinuousPhysicsEnginePort, PortError};
use crate::unicycle::{MAX_THRUST, MAX_TORQUE};

/// Target position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Goal {
    /// x (m)
    pub x: f64,
    /// y (m)
    pub y: f64,
}

/// Deterministic splitmix64 RNG (fixed seed => identical sequences on every platform).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    /// Seeded RNG.
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0,1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Uniform in [lo,hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.unit().max(1e-300);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// A controller deciding the control for the next control period.
pub trait Controller {
    /// Name for reports.
    fn name(&self) -> String;
    /// Decide `[thrust, torque]`. May fork/restore the port but must leave it at the root state.
    fn decide(
        &mut self,
        port: &mut dyn ContinuousPhysicsEnginePort,
        goal: Goal,
    ) -> Result<Vec<f64>, PortError>;
    /// Simulations (rollouts) consumed by the last `decide`.
    fn last_sims(&self) -> usize {
        0
    }
}

fn wrap(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    let mut r = (a + std::f64::consts::PI) % t;
    if r < 0.0 {
        r += t;
    }
    r - std::f64::consts::PI
}

/// Proportional-derivative go-to-goal controller (no simulation).
#[derive(Debug, Clone, Copy)]
pub struct PdController {
    /// Thrust gain on distance.
    pub kp_dist: f64,
    /// Thrust damping on speed.
    pub kd_speed: f64,
    /// Torque gain on heading error.
    pub kp_head: f64,
    /// Torque damping on yaw rate.
    pub kd_head: f64,
}

impl Default for PdController {
    fn default() -> Self {
        Self {
            kp_dist: 4.0,
            kd_speed: 2.0,
            kp_head: 1.5,
            kd_head: 0.25,
        }
    }
}

impl Controller for PdController {
    fn name(&self) -> String {
        "classic-PD".into()
    }
    fn decide(
        &mut self,
        port: &mut dyn ContinuousPhysicsEnginePort,
        goal: Goal,
    ) -> Result<Vec<f64>, PortError> {
        let q = port.qpos()?;
        let (v, w) = port.body_twist()?;
        let (dx, dy) = (goal.x - q[0], goal.y - q[1]);
        let dist = dx.hypot(dy);
        let err = wrap(dy.atan2(dx) - q[2]);
        let thrust = (self.kp_dist * dist * err.cos().max(0.0) - self.kd_speed * v)
            .clamp(-MAX_THRUST, MAX_THRUST);
        let torque = (self.kp_head * err - self.kd_head * w).clamp(-MAX_TORQUE, MAX_TORQUE);
        Ok(vec![thrust, torque])
    }
}

/// Budgeted sampling planner (random shooting + one Gaussian refinement round, receding horizon).
#[derive(Debug, Clone)]
pub struct SamplingPlanner {
    /// Hard cap on simulations (rollouts) per decision.
    pub sim_budget: usize,
    /// Control segments per rollout.
    pub segments: usize,
    /// Physics steps per segment (== control period).
    pub seg_steps: usize,
    rng: Rng,
    warm: Vec<[f64; 2]>,
    sims: usize,
}

impl SamplingPlanner {
    /// New planner with fixed seed.
    pub fn new(seed: u64, sim_budget: usize, segments: usize, seg_steps: usize) -> Self {
        Self {
            sim_budget,
            segments,
            seg_steps,
            rng: Rng::new(seed),
            warm: Vec::new(),
            sims: 0,
        }
    }

    fn rollout(
        &mut self,
        port: &mut dyn ContinuousPhysicsEnginePort,
        root: &crate::port::EngineSnapshot,
        seq: &[[f64; 2]],
        goal: Goal,
    ) -> Result<f64, PortError> {
        self.sims += 1;
        port.restore(root)?;
        let seg_dt = port.dt() * self.seg_steps as f64;
        let mut cost = 0.0;
        for c in seq {
            port.set_ctrl(c)?;
            port.step(self.seg_steps)?;
            let q = port.qpos()?;
            let d2 = (goal.x - q[0]).powi(2) + (goal.y - q[1]).powi(2);
            cost += d2 * seg_dt + 0.002 * (c[0] * c[0] + c[1] * c[1]) * seg_dt;
        }
        let q = port.qpos()?;
        let (v, w) = port.body_twist()?;
        let d2 = (goal.x - q[0]).powi(2) + (goal.y - q[1]).powi(2);
        Ok(cost + 5.0 * d2 + 0.5 * v * v + 0.05 * w * w)
    }
}

impl Controller for SamplingPlanner {
    fn name(&self) -> String {
        format!("sampling-planner(B={})", self.sim_budget)
    }
    fn last_sims(&self) -> usize {
        self.sims
    }
    fn decide(
        &mut self,
        port: &mut dyn ContinuousPhysicsEnginePort,
        goal: Goal,
    ) -> Result<Vec<f64>, PortError> {
        self.sims = 0;
        let root = port.snapshot()?;
        let s = self.segments;
        let mut best: Vec<[f64; 2]> = if self.warm.len() == s {
            self.warm.clone()
        } else {
            vec![[0.0, 0.0]; s]
        };
        let mut best_cost = self.rollout(port, &root, &best.clone(), goal)?;
        let explore = self.sim_budget.saturating_sub(1) / 2;
        for _ in 0..explore {
            let cand: Vec<[f64; 2]> = (0..s)
                .map(|_| {
                    [
                        self.rng.uniform(-MAX_THRUST, MAX_THRUST),
                        self.rng.uniform(-MAX_TORQUE, MAX_TORQUE),
                    ]
                })
                .collect();
            let c = self.rollout(port, &root, &cand, goal)?;
            if c < best_cost {
                best_cost = c;
                best = cand;
            }
        }
        while self.sims < self.sim_budget {
            let cand: Vec<[f64; 2]> = best
                .iter()
                .map(|b| {
                    [
                        (b[0] + 0.3 * MAX_THRUST * self.rng.normal())
                            .clamp(-MAX_THRUST, MAX_THRUST),
                        (b[1] + 0.3 * MAX_TORQUE * self.rng.normal())
                            .clamp(-MAX_TORQUE, MAX_TORQUE),
                    ]
                })
                .collect();
            let c = self.rollout(port, &root, &cand, goal)?;
            if c < best_cost {
                best_cost = c;
                best = cand;
            }
        }
        debug_assert!(self.sims <= self.sim_budget);
        port.restore(&root)?;
        // Receding horizon: shift the plan by one segment for the next decision.
        let first = best[0];
        let mut next = best[1..].to_vec();
        next.push(*best.last().unwrap_or(&[0.0, 0.0]));
        self.warm = next;
        Ok(first.to_vec())
    }
}

/// Result of one closed-loop episode.
#[derive(Debug, Clone)]
pub struct EpisodeReport {
    /// Controller name.
    pub controller: String,
    /// Backend name.
    pub backend: String,
    /// Number of decisions.
    pub decisions: usize,
    /// Distance to goal at the end (m).
    pub final_error: f64,
    /// Minimum distance reached (m).
    pub min_error: f64,
    /// First time within `reach_radius` (s).
    pub reach_time: Option<f64>,
    /// Mean latency per decision (microseconds, wall clock).
    pub latency_mean_us: f64,
    /// Max latency per decision (microseconds).
    pub latency_max_us: f64,
    /// Simulations per decision (max over the episode).
    pub sims_per_decision_max: usize,
    /// Total simulations over the episode.
    pub sims_total: usize,
}

/// Radius (m) that counts as "reached".
pub const REACH_RADIUS: f64 = 0.05;

/// Run a closed-loop episode: `decisions` control periods of `seg_steps` physics steps each.
pub fn run_episode(
    port: &mut dyn ContinuousPhysicsEnginePort,
    controller: &mut dyn Controller,
    goal: Goal,
    decisions: usize,
    seg_steps: usize,
) -> Result<EpisodeReport, PortError> {
    port.reset()?;
    let mut lat = Vec::with_capacity(decisions);
    let (mut min_err, mut reach, mut sims_total, mut sims_max) = (f64::INFINITY, None, 0, 0);
    for _ in 0..decisions {
        let t0 = Instant::now();
        let u = controller.decide(port, goal)?;
        lat.push(t0.elapsed().as_secs_f64() * 1e6);
        sims_total += controller.last_sims();
        sims_max = sims_max.max(controller.last_sims());
        port.set_ctrl(&u)?;
        port.step(seg_steps)?;
        let q = port.qpos()?;
        let e = (goal.x - q[0]).hypot(goal.y - q[1]);
        min_err = min_err.min(e);
        if reach.is_none() && e <= REACH_RADIUS {
            reach = Some(port.time()?);
        }
    }
    let q = port.qpos()?;
    Ok(EpisodeReport {
        controller: controller.name(),
        backend: port.name().to_string(),
        decisions,
        final_error: (goal.x - q[0]).hypot(goal.y - q[1]),
        min_error: min_err,
        reach_time: reach,
        latency_mean_us: lat.iter().sum::<f64>() / lat.len().max(1) as f64,
        latency_max_us: lat.iter().cloned().fold(0.0, f64::max),
        sims_per_decision_max: sims_max,
        sims_total,
    })
}
