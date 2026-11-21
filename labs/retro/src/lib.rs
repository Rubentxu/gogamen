//! GOG-60 emulation spike. See `README.md` for the technical decision
//! (dependency: none; license: original code, no ROM; platform: Linux x86_64;
//! alternatives: Stable-Retro vs libretro via libloading).
//!
//! [`LibretroCorePort`] is the boundary where a real libretro C-ABI core
//! (loaded with `libloading`) would be plugged later. No FFI is written here.

mod adapter;
mod core;
mod session;

pub use adapter::{
    ACTION_TABLE_VERSION, ADAPTER_VERSION, AccessMode, Adapter, AdapterConfig, AdapterError,
    AdapterSpec, SnapshotForkModel, StepOutcome, planning_features, planning_reward,
};
pub use core::{FB_HEIGHT, FB_WIDTH, FreePlayCore, RAM_SIZE, buttons};
pub use session::{HISTORY_LEN, Session};

/// Errors from state (de)serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StateError {
    /// Data shorter than required.
    Truncated,
    /// Magic/version mismatch.
    BadMagic,
    /// Length fields inconsistent with the data.
    BadLength,
    /// Integrity checksum failed.
    BadChecksum,
}

impl std::fmt::Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid state: {self:?}")
    }
}

impl std::error::Error for StateError {}

/// Mirror of the libretro core contract (`retro_run`, `retro_reset`,
/// `retro_get_memory_data`, `retro_serialize`, `retro_unserialize`,
/// video refresh). Frames are 8-bit palette indices, row-major.
pub trait LibretroCorePort {
    /// Framebuffer dimensions `(width, height)`.
    fn frame_size(&self) -> (usize, usize);
    /// Current framebuffer (`width * height` bytes).
    fn framebuffer(&self) -> &[u8];
    /// System RAM view (`RETRO_MEMORY_SYSTEM_RAM`).
    fn ram(&self) -> &[u8];
    /// Advance exactly one frame with the given joypad bitmask.
    fn run_frame(&mut self, input_mask: u16);
    /// Reset to power-on state.
    fn reset(&mut self);
    /// Serialize full emulator state.
    fn serialize_state(&self) -> Vec<u8>;
    /// Restore state; must return an error (never panic) on bad data, and
    /// leave the core unchanged on error.
    fn unserialize_state(&mut self, data: &[u8]) -> Result<(), StateError>;
}

/// FNV-1a 64-bit hash, used for frame hashes and checksums.
pub fn fnv1a(data: &[u8]) -> u64 {
    data.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}
