//! Episode runner: the one place where the control loop (`nav::Sim`) and cognition meet.
//!
//! Per iteration the control loop advances exactly one tick, whatever cognition is doing. Before
//! each tick cognition is polled; it may answer with a goal [`Command`], which only changes
//! `Sim`'s task goal and releases a (still untrusted) planner. The core `ActionGuard` and the
//! independent safety stop remain inside `Sim` and see every command.

use crate::nav::{
    GridPlanner, NavObservation, NavPlanner, Outcome, ScenarioConfig, Sim, TraceRow,
    VelocityLimits, trace_hash,
};
use gogamen_core::BudgetTracker;

use super::cognition::{Cognition, Command, Event};
use super::fixtures::DialogFixture;
use super::oracle::OracleMode;
use super::policy::{DialogPolicy, Door};

/// Ticks the robot keeps holding after an "abstain" decision before the episode is closed.
pub const DIALOG_HOLD_SETTLE_TICKS: u64 = 100;

/// Planner that proposes an explicit stop until released, then delegates to the grid planner.
#[derive(Debug, Clone)]
pub struct GatedPlanner {
    inner: GridPlanner,
    released: bool,
}

impl GatedPlanner {
    /// Gated planner, initially holding.
    pub fn new(limits: VelocityLimits) -> Self {
        Self {
            inner: GridPlanner::new(limits),
            released: false,
        }
    }

    /// Let the inner planner drive.
    pub fn release(&mut self) {
        self.released = true;
    }
}

impl NavPlanner for GatedPlanner {
    fn name(&self) -> String {
        "gated-grid".into()
    }

    fn plan(&mut self, obs: &NavObservation<'_>, budget: &mut BudgetTracker) -> Option<Vec<f64>> {
        if self.released {
            self.inner.plan(obs, budget)
        } else {
            Some(vec![0.0, 0.0])
        }
    }
}

/// Final classification of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpisodeVerdict {
    /// Reached the true door.
    Success,
    /// Reached the other door.
    WrongDoor,
    /// Decided to wait; never moved.
    Abstained,
    /// The body touched an obstacle.
    Collision,
    /// Navigation timed out.
    Timeout,
    /// The stack stopped itself (guard or safety stop) without reaching a door.
    SafeStop,
}

/// Everything measured in one episode.
#[derive(Debug, Clone)]
pub struct EpisodeResult {
    /// Classification.
    pub verdict: EpisodeVerdict,
    /// Door the robot was sent to, if any.
    pub chosen: Option<Door>,
    /// Number of `AskHuman` questions.
    pub queries: u32,
    /// Final belief `P(left)`.
    pub p_left: f64,
    /// Tick at which cognition decided (go or wait).
    pub decided_at: Option<u64>,
    /// Tick at which the planner was released (goal adopted), if ever.
    pub released_at: Option<u64>,
    /// Control ticks executed.
    pub ticks: u64,
    /// Full control trace (one row per tick).
    pub trace: Vec<TraceRow>,
    /// Hash of the whole control trace.
    pub trace_hash: u64,
    /// Cognition events.
    pub events: Vec<Event>,
    /// Number of control ticks on which the guard did not simply allow the proposal.
    pub guard_non_allow_ticks: u64,
    /// Number of ticks on which the independent safety stop was active.
    pub safety_ticks: u64,
}

/// Runs dialog episodes.
#[derive(Debug, Clone, Copy)]
pub struct DialogEpisode;

impl DialogEpisode {
    /// Run `fixture` under `policy` with the oracle in `mode`; `seed` drives sensor noise.
    pub fn run(
        fixture: &DialogFixture,
        policy: DialogPolicy,
        mode: OracleMode,
        seed: u64,
    ) -> EpisodeResult {
        let cog = Cognition::new(fixture, policy, mode);
        Self::run_with(fixture, cog, seed)
    }

    /// Run with a prepared cognition (custom oracle port).
    pub fn run_with(fixture: &DialogFixture, mut cog: Cognition, seed: u64) -> EpisodeResult {
        let mut cfg = ScenarioConfig::corridor(seed);
        cfg.goal = [5.0, 0.0]; // between the doors; never "reached" while holding at the start
        cfg.safe_stop_hold_ticks = u64::MAX / 2; // holding while the human thinks is not a failure
        cfg.max_ticks = 8000;
        let mut sim = Sim::new(cfg.clone(), GatedPlanner::new(cfg.limits));
        let mut chosen = None;
        let mut released_at = None;
        loop {
            let t = sim.tick();
            if let Some(Command::Go(d)) = cog.poll(t) {
                sim.set_goal(d.position());
                sim.planner_mut().release();
                chosen = Some(d);
                released_at = Some(t);
            }
            if sim.outcome() != Outcome::Running {
                break;
            }
            if let Some(d) = cog.decided_at()
                && chosen.is_none()
                && t >= d + DIALOG_HOLD_SETTLE_TICKS
            {
                break;
            }
            sim.step_tick().expect("pure-rust engine does not fail");
        }
        let verdict = match (sim.outcome(), chosen) {
            (Outcome::Collision, _) => EpisodeVerdict::Collision,
            (Outcome::Reached, Some(d)) if d.is_left() == fixture.truth_left => {
                EpisodeVerdict::Success
            }
            (Outcome::Reached, Some(_)) => EpisodeVerdict::WrongDoor,
            (Outcome::SafeStop, _) => EpisodeVerdict::SafeStop,
            (Outcome::Timeout, _) => EpisodeVerdict::Timeout,
            (_, None) => EpisodeVerdict::Abstained,
            (_, Some(_)) => EpisodeVerdict::Timeout,
        };
        let report = sim.report();
        let trace = sim.trace().to_vec();
        EpisodeResult {
            verdict,
            chosen,
            queries: cog.asks(),
            p_left: cog.perception().p_left(),
            decided_at: cog.decided_at(),
            released_at,
            ticks: report.ticks,
            trace_hash: trace_hash(&trace),
            trace,
            events: cog.events().to_vec(),
            guard_non_allow_ticks: report.fallbacks,
            safety_ticks: report.safety_ticks,
        }
    }
}
