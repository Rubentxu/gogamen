//! Wrapper around a core: frame counter and recent input history live outside
//! the emulator, so a snapshot must carry both to be able to continue.

use crate::{LibretroCorePort, StateError, fnv1a};

/// Number of recent inputs retained in the wrapper.
pub const HISTORY_LEN: usize = 8;

const MAGIC: &[u8; 4] = b"GSN1";

pub struct Session<C: LibretroCorePort> {
    core: C,
    frame: u64,
    history: [u16; HISTORY_LEN],
}

impl<C: LibretroCorePort> Session<C> {
    pub fn new(core: C) -> Self {
        Self {
            core,
            frame: 0,
            history: [0; HISTORY_LEN],
        }
    }

    pub fn core(&self) -> &C {
        &self.core
    }

    /// Frames stepped since creation/reset/restore point.
    pub fn frame(&self) -> u64 {
        self.frame
    }

    /// Most recent inputs, oldest first.
    pub fn history(&self) -> &[u16; HISTORY_LEN] {
        &self.history
    }

    pub fn step(&mut self, mask: u16) {
        self.core.run_frame(mask);
        self.frame += 1;
        self.history.rotate_left(1);
        self.history[HISTORY_LEN - 1] = mask;
    }

    pub fn reset(&mut self) {
        self.core.reset();
        self.frame = 0;
        self.history = [0; HISTORY_LEN];
    }

    /// Hash of the current framebuffer.
    pub fn frame_hash(&self) -> u64 {
        fnv1a(self.core.framebuffer())
    }

    /// Snapshot = magic | frame u64 | history | core_len u32 | core | fnv1a u64.
    pub fn snapshot(&self) -> Vec<u8> {
        let core = self.core.serialize_state();
        let mut v = Vec::with_capacity(core.len() + 40);
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&self.frame.to_le_bytes());
        for h in self.history {
            v.extend_from_slice(&h.to_le_bytes());
        }
        v.extend_from_slice(&(core.len() as u32).to_le_bytes());
        v.extend_from_slice(&core);
        let sum = fnv1a(&v);
        v.extend_from_slice(&sum.to_le_bytes());
        v
    }

    /// Restore; on error the session is left unchanged.
    pub fn restore(&mut self, data: &[u8]) -> Result<(), StateError> {
        const HEAD: usize = 4 + 8 + HISTORY_LEN * 2 + 4;
        if data.len() < HEAD + 8 {
            return Err(StateError::Truncated);
        }
        if &data[..4] != MAGIC {
            return Err(StateError::BadMagic);
        }
        let (body, sum) = data.split_at(data.len() - 8);
        if fnv1a(body) != u64::from_le_bytes(sum.try_into().unwrap()) {
            return Err(StateError::BadChecksum);
        }
        let frame = u64::from_le_bytes(body[4..12].try_into().unwrap());
        let mut history = [0u16; HISTORY_LEN];
        for (i, h) in history.iter_mut().enumerate() {
            *h = u16::from_le_bytes(body[12 + i * 2..14 + i * 2].try_into().unwrap());
        }
        let len = u32::from_le_bytes(body[HEAD - 4..HEAD].try_into().unwrap()) as usize;
        if body.len() - HEAD != len {
            return Err(StateError::BadLength);
        }
        self.core.unserialize_state(&body[HEAD..])?;
        self.frame = frame;
        self.history = history;
        Ok(())
    }
}
