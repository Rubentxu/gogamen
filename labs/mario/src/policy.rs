//! Baselines and a snapshot-based Monte Carlo planner.
//!
//! The planner is a small local Monte Carlo rollout planner over
//! `EmulatorPort::snapshot/restore`. It is NOT wired to `gogamen-core`'s FMC
//! (that needs a `GenerativeModel`/`State` encoding; see README).

use crate::port::{EmulatorPort, PadState};

/// Deterministic xorshift64* generator (seeded via splitmix64).
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Self((z ^ (z >> 31)) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next_u64() >> 11) % n
    }
}

/// Chooses the next pad state. `sim_steps` accumulates planning cost
/// (emulator decision-steps simulated on forks/restores, not on the live run).
pub trait Policy {
    fn name(&self) -> &'static str;
    fn act(&mut self, port: &mut dyn EmulatorPort, frames_per_action: u32) -> PadState;
    fn sim_steps(&self) -> u64 {
        0
    }
}

const ALL_PADS: [PadState; 8] = [
    PadState::NOOP,
    PadState::RIGHT,
    PadState::LEFT,
    PadState::JUMP,
    PadState::RIGHT.with(PadState::JUMP),
    PadState::RIGHT.with(PadState::RUN),
    PadState::RIGHT.with(PadState::RUN).with(PadState::JUMP),
    PadState::LEFT.with(PadState::JUMP),
];

/// Uniform random over all pad combinations.
pub struct RandomPolicy(Rng);

impl RandomPolicy {
    pub fn new(seed: u64) -> Self {
        Self(Rng::new(seed))
    }
}

impl Policy for RandomPolicy {
    fn name(&self) -> &'static str {
        "random"
    }
    fn act(&mut self, _: &mut dyn EmulatorPort, _: u32) -> PadState {
        ALL_PADS[self.0.below(ALL_PADS.len() as u64) as usize]
    }
}

/// Always holds right.
pub struct AlwaysRight;

impl Policy for AlwaysRight {
    fn name(&self) -> &'static str {
        "always_right"
    }
    fn act(&mut self, _: &mut dyn EmulatorPort, _: u32) -> PadState {
        PadState::RIGHT
    }
}

const FORWARD: [PadState; 4] = [
    PadState::RIGHT,
    PadState::RIGHT.with(PadState::JUMP),
    PadState::RIGHT.with(PadState::RUN),
    PadState::RIGHT.with(PadState::RUN).with(PadState::JUMP),
];

/// Monte Carlo rollout planner: for each candidate first action runs
/// `rollouts` random rollouts of `horizon` decisions from a snapshot, scores
/// them by progress minus a death penalty, and picks the best first action.
pub struct MonteCarloPlanner {
    rng: Rng,
    pub rollouts: u32,
    pub horizon: u32,
    steps: u64,
}

const DEATH_PENALTY: i64 = 1_000;

impl MonteCarloPlanner {
    pub fn new(seed: u64, rollouts: u32, horizon: u32) -> Self {
        Self {
            rng: Rng::new(seed),
            rollouts,
            horizon,
            steps: 0,
        }
    }
}

impl Policy for MonteCarloPlanner {
    fn name(&self) -> &'static str {
        "monte_carlo"
    }
    fn sim_steps(&self) -> u64 {
        self.steps
    }
    fn act(&mut self, port: &mut dyn EmulatorPort, frames: u32) -> PadState {
        let Ok(root) = port.snapshot() else {
            return PadState::RIGHT; // no snapshot support: degrade, never panic
        };
        let base = port.features();
        let mut best = (i64::MIN, PadState::RIGHT);
        for &first in &FORWARD {
            let mut total = 0i64;
            for _ in 0..self.rollouts {
                let mut pad = first;
                for _ in 0..self.horizon {
                    let r = port.step(pad, frames);
                    self.steps += 1;
                    if r.features.is_terminal() || r.features.deaths > base.deaths {
                        break;
                    }
                    pad = FORWARD[self.rng.below(FORWARD.len() as u64) as usize];
                }
                let f = port.features();
                let mut score = i64::from(f.x - base.x);
                score -= DEATH_PENALTY * i64::from(f.deaths - base.deaths);
                if f.level_complete {
                    score += DEATH_PENALTY;
                }
                total += score;
                if port.restore(&root).is_err() {
                    return PadState::RIGHT;
                }
            }
            if total > best.0 {
                best = (total, first);
            }
        }
        best.1
    }
}
