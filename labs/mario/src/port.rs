//! The emulator boundary used by metrics and policies.

use std::fmt;

/// Joypad state (action space of the port). Bit layout is stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PadState(pub u8);

impl PadState {
    pub const NOOP: Self = Self(0);
    pub const RIGHT: Self = Self(1);
    pub const LEFT: Self = Self(2);
    /// Jump (A button in the usual mapping).
    pub const JUMP: Self = Self(4);
    /// Run (B button in the usual mapping).
    pub const RUN: Self = Self(8);

    pub const fn with(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    pub const fn has(self, other: Self) -> bool {
        self.0 & other.0 == other.0 && other.0 != 0
    }
}

/// Typed observation extracted from RAM (or from the synthetic state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MarioFeatures {
    /// Horizontal position in level units.
    pub x: i32,
    /// Vertical position above the ground (units).
    pub y: i32,
    /// Vertical velocity.
    pub vy: i32,
    pub on_ground: bool,
    pub lives: u8,
    /// Cumulative deaths since reset.
    pub deaths: u32,
    /// True only on the step in which a death happened.
    pub died_this_step: bool,
    /// Level goal reached.
    pub level_complete: bool,
    /// No lives left or time out.
    pub game_over: bool,
    /// Frames emulated since reset.
    pub frame: u64,
}

impl MarioFeatures {
    pub fn is_terminal(&self) -> bool {
        self.level_complete || self.game_over
    }
}

/// Result of one `step`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepReport {
    pub frames_run: u32,
    pub features: MarioFeatures,
}

/// Typed port failures (never panics on bad input).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortError {
    /// Backend has no snapshot support.
    SnapshotUnsupported,
    /// Snapshot data is malformed.
    BadSnapshot(&'static str),
}

impl fmt::Display for PortError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SnapshotUnsupported => write!(f, "snapshot/restore unsupported by backend"),
            Self::BadSnapshot(r) => write!(f, "bad snapshot: {r}"),
        }
    }
}

impl std::error::Error for PortError {}

/// Emulator boundary: reset, step with pad input for `frames` frames,
/// typed features + RAM view, optional snapshot/restore.
pub trait EmulatorPort {
    /// Human-readable backend name (e.g. `synthetic-fixture`).
    fn backend_name(&self) -> &'static str;
    /// True when the backend is a synthetic stand-in and not a real game.
    fn is_synthetic(&self) -> bool;
    fn reset(&mut self);
    /// Hold `pad` for `frames` frames (stops early if terminal).
    fn step(&mut self, pad: PadState, frames: u32) -> StepReport;
    fn features(&self) -> MarioFeatures;
    /// Raw RAM view the features are decoded from.
    fn ram(&self) -> &[u8];
    /// Serialize full state; `Err(SnapshotUnsupported)` if not supported.
    fn snapshot(&self) -> Result<Vec<u8>, PortError>;
    /// Restore a state; on error the port must be left unchanged.
    fn restore(&mut self, data: &[u8]) -> Result<(), PortError>;
}
