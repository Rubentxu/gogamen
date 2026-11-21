//! Conditional-effects learning on partially observable Grid (GOG-164 / DEM-011, first slice).
//!
//! The agent sees only a radius-1 cross (is the cell N/S/W/E blocked?) plus whether its last
//! move displaced it; its position is hidden. The learner ([`ConditionalEffectsLearner`]) never
//! touches the simulator or any goal reward: it receives [`ConditionalObservation`]s.
//!
//! # Preregistration (fixed BEFORE the first run)
//! - Map: [`GridMap::cul_de_sac_map`]. Policy: uniform random over the 5 actions.
//! - Train stream: [`TRAIN_SEED`], [`TRAIN_STEPS`] transitions, ids `1..`.
//!   Holdout stream: [`HOLDOUT_SEED`], [`HOLDOUT_STEPS`] transitions (disjoint seed, never learned).
//! - Baseline: uninformed prior `p = 0.5` per bit (Brier `0.25`, log-loss `ln 2`).
//! - Decision rule: improvement is claimed iff holdout Brier <= [`BRIER_RATIO_MAX`] x prior Brier
//!   AND holdout log-loss < prior log-loss. Otherwise the result is reported as NOT improved.
//! - Changing dynamics: after training, actions Up<->Down and Left<->Right are swapped for
//!   [`SHIFT_STEPS`] transitions; success means health reaches `PredictionFailure` and the gate
//!   refuses (no safe-action substitution).

use gogamen_core::{
    ConditionalBudget, ConditionalConfig, ConditionalEffectsLearner, ConditionalError,
    ConditionalObservation, DeterministicRng, EvidenceRef, MasterSeed, ObservationId, RngSource,
    StreamKey,
};

use crate::map::{GridMap, GridPos};
use crate::world::{ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, GRID_ACTION_CARDINALITY};

/// Number of macrostate bits: blocked N, S, W, E and "moved".
pub const NUM_BITS: usize = 5;
/// Seed of the training stream.
pub const TRAIN_SEED: u64 = 0x164_0001;
/// Seed of the fixed holdout stream.
pub const HOLDOUT_SEED: u64 = 0x164_0002;
/// Seed of the shifted-dynamics stream.
pub const SHIFT_SEED: u64 = 0x164_0003;
/// Training transitions.
pub const TRAIN_STEPS: usize = 600;
/// Holdout transitions.
pub const HOLDOUT_STEPS: usize = 300;
/// Shifted-dynamics transitions.
pub const SHIFT_STEPS: usize = 120;
/// Required Brier ratio (learned / prior) to claim improvement.
pub const BRIER_RATIO_MAX: f64 = 0.9;

/// Dynamics variant of the hidden physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dynamics {
    /// Actions move as named.
    Normal,
    /// Up<->Down and Left<->Right are swapped (hidden physics change).
    Flipped,
}

fn delta(action: usize, dynamics: Dynamics) -> (isize, isize) {
    let eff = match (dynamics, action) {
        (Dynamics::Flipped, ACTION_UP) => ACTION_DOWN,
        (Dynamics::Flipped, ACTION_DOWN) => ACTION_UP,
        (Dynamics::Flipped, ACTION_LEFT) => ACTION_RIGHT,
        (Dynamics::Flipped, ACTION_RIGHT) => ACTION_LEFT,
        (_, a) => a,
    };
    match eff {
        ACTION_UP => (0, -1),
        ACTION_DOWN => (0, 1),
        ACTION_LEFT => (-1, 0),
        ACTION_RIGHT => (1, 0),
        _ => (0, 0),
    }
}

fn shifted(pos: GridPos, d: (isize, isize)) -> Option<GridPos> {
    let x = pos.x.checked_add_signed(d.0)?;
    let y = pos.y.checked_add_signed(d.1)?;
    Some(GridPos::new(x, y))
}

fn blocked(map: &GridMap, pos: GridPos, d: (isize, isize)) -> bool {
    shifted(pos, d).is_none_or(|p| !map.is_passable(p))
}

/// Partial observation: blocked N/S/W/E around `pos` plus the `moved` flag.
fn observe(map: &GridMap, pos: GridPos, moved: bool) -> Vec<bool> {
    vec![
        blocked(map, pos, (0, -1)),
        blocked(map, pos, (0, 1)),
        blocked(map, pos, (-1, 0)),
        blocked(map, pos, (1, 0)),
        moved,
    ]
}

/// Generates `steps` transitions of random-policy experience (ids start at `id_base + 1`).
pub fn generate_stream(
    map: &GridMap,
    dynamics: Dynamics,
    seed: u64,
    steps: usize,
    id_base: u64,
) -> Vec<ConditionalObservation> {
    let mut rng: DeterministicRng = MasterSeed::new(seed).derive(StreamKey::Episode(0x164));
    let mut pos = map.start();
    let mut moved = false;
    let mut out = Vec::with_capacity(steps);
    for i in 0..steps {
        let state = observe(map, pos, moved);
        let action = (rng.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
        let d = delta(action, dynamics);
        moved = false;
        if let Some(p) = shifted(pos, d).filter(|p| d != (0, 0) && map.is_passable(*p)) {
            pos = p;
            moved = true;
        }
        let id = id_base + i as u64 + 1;
        out.push(ConditionalObservation {
            id: ObservationId::new(id),
            evidence: EvidenceRef::new(0x164_0000_0000 + id),
            state,
            action,
            next: observe(map, pos, moved),
        });
    }
    out
}

/// Budget used by the lab (2-literal clauses, bounded clause table).
pub fn lab_budget() -> ConditionalBudget {
    ConditionalBudget {
        max_literals: 2,
        max_clauses_per_bit: 64,
        max_updates: 5_000,
    }
}

/// Builds the lab learner.
pub fn new_lab_learner() -> Result<ConditionalEffectsLearner, ConditionalError> {
    ConditionalEffectsLearner::new(
        NUM_BITS,
        GRID_ACTION_CARDINALITY,
        lab_budget(),
        ConditionalConfig::default(),
    )
}

/// Action decision that never hides a prediction failure behind a default action.
#[derive(Debug, Clone, PartialEq)]
pub enum GatedChoice {
    /// Predictions are trustworthy enough to act on `action`.
    Act(usize),
    /// Predictions are failing; no action is offered.
    PredictionFailure(ConditionalError),
}

/// Proposes `candidate` only if the learner's gate is open.
pub fn gated_choice(learner: &ConditionalEffectsLearner, candidate: usize) -> GatedChoice {
    match learner.gate() {
        Ok(()) => GatedChoice::Act(candidate),
        Err(e) => GatedChoice::PredictionFailure(e),
    }
}
