//! Original tiny deterministic game: collect coins while a drifting enemy
//! (driven by a fixed-seed xorshift PRNG) bounces around. No wall-clock.

use crate::{LibretroCorePort, StateError};

pub const FB_WIDTH: usize = 32;
pub const FB_HEIGHT: usize = 32;
pub const RAM_SIZE: usize = 32;

/// Joypad bits.
pub mod buttons {
    pub const UP: u16 = 1;
    pub const DOWN: u16 = 2;
    pub const LEFT: u16 = 4;
    pub const RIGHT: u16 = 8;
    pub const A: u16 = 16;
}

// RAM layout.
const PX: usize = 0;
const PY: usize = 1;
const CX: usize = 2;
const CY: usize = 3;
const SCORE: usize = 4; // u16 LE
const RNG: usize = 6; // u32 LE
const EX: usize = 10;
const EY: usize = 11;
const EDX: usize = 12; // i8 as u8
const EDY: usize = 13;
const FRAMES: usize = 14; // u32 LE
const LIVES: usize = 18;

const MAGIC: &[u8; 4] = b"GFP1";
const SEED: u32 = 0x1234_5678;

#[derive(Clone)]
pub struct FreePlayCore {
    ram: [u8; RAM_SIZE],
    fb: [u8; FB_WIDTH * FB_HEIGHT],
}

impl Default for FreePlayCore {
    fn default() -> Self {
        let mut c = Self {
            ram: [0; RAM_SIZE],
            fb: [0; FB_WIDTH * FB_HEIGHT],
        };
        c.reset();
        c
    }
}

impl FreePlayCore {
    pub fn new() -> Self {
        Self::default()
    }

    fn rng(&mut self) -> u32 {
        let mut x = u32::from_le_bytes(self.ram[RNG..RNG + 4].try_into().unwrap());
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.ram[RNG..RNG + 4].copy_from_slice(&x.to_le_bytes());
        x
    }

    fn respawn_coin(&mut self) {
        let r = self.rng();
        self.ram[CX] = (r % FB_WIDTH as u32) as u8;
        self.ram[CY] = ((r >> 8) % FB_HEIGHT as u32) as u8;
    }

    fn render(&mut self) {
        self.fb.fill(0);
        let set = |fb: &mut [u8], x: u8, y: u8, c: u8| fb[y as usize * FB_WIDTH + x as usize] = c;
        set(&mut self.fb, self.ram[CX], self.ram[CY], 2);
        set(&mut self.fb, self.ram[EX], self.ram[EY], 3);
        set(&mut self.fb, self.ram[PX], self.ram[PY], 1);
    }

    /// Score as displayed in RAM.
    pub fn score(&self) -> u16 {
        u16::from_le_bytes([self.ram[SCORE], self.ram[SCORE + 1]])
    }
}

impl LibretroCorePort for FreePlayCore {
    fn frame_size(&self) -> (usize, usize) {
        (FB_WIDTH, FB_HEIGHT)
    }
    fn framebuffer(&self) -> &[u8] {
        &self.fb
    }
    fn ram(&self) -> &[u8] {
        &self.ram
    }

    fn reset(&mut self) {
        self.ram = [0; RAM_SIZE];
        self.ram[PX] = 16;
        self.ram[PY] = 16;
        self.ram[RNG..RNG + 4].copy_from_slice(&SEED.to_le_bytes());
        self.ram[EX] = 4;
        self.ram[EY] = 5;
        self.ram[EDX] = 1;
        self.ram[EDY] = 1;
        self.ram[LIVES] = 3;
        self.respawn_coin();
        self.render();
    }

    fn run_frame(&mut self, m: u16) {
        let w = FB_WIDTH as u8;
        let h = FB_HEIGHT as u8;
        // Player movement (wraps around the screen); A moves twice as fast.
        let step: u8 = if m & buttons::A != 0 { 2 } else { 1 };
        if m & buttons::UP != 0 {
            self.ram[PY] = (self.ram[PY] + h - step) % h;
        }
        if m & buttons::DOWN != 0 {
            self.ram[PY] = (self.ram[PY] + step) % h;
        }
        if m & buttons::LEFT != 0 {
            self.ram[PX] = (self.ram[PX] + w - step) % w;
        }
        if m & buttons::RIGHT != 0 {
            self.ram[PX] = (self.ram[PX] + step) % w;
        }
        // Enemy bounces; PRNG occasionally flips its direction.
        let r = self.rng();
        if r.is_multiple_of(7) {
            self.ram[EDX] = self.ram[EDX].wrapping_neg();
        }
        if (r >> 8).is_multiple_of(11) {
            self.ram[EDY] = self.ram[EDY].wrapping_neg();
        }
        let nx = (self.ram[EX] as i8).wrapping_add(self.ram[EDX] as i8);
        let ny = (self.ram[EY] as i8).wrapping_add(self.ram[EDY] as i8);
        if nx < 0 || nx >= w as i8 {
            self.ram[EDX] = self.ram[EDX].wrapping_neg();
        } else {
            self.ram[EX] = nx as u8;
        }
        if ny < 0 || ny >= h as i8 {
            self.ram[EDY] = self.ram[EDY].wrapping_neg();
        } else {
            self.ram[EY] = ny as u8;
        }
        // Coin pickup.
        if self.ram[PX] == self.ram[CX] && self.ram[PY] == self.ram[CY] {
            let s = self.score().wrapping_add(1);
            self.ram[SCORE..SCORE + 2].copy_from_slice(&s.to_le_bytes());
            self.respawn_coin();
        }
        // Enemy hit: lose a life (wrapping back to 3), reposition player.
        if self.ram[PX] == self.ram[EX] && self.ram[PY] == self.ram[EY] {
            self.ram[LIVES] = if self.ram[LIVES] <= 1 {
                3
            } else {
                self.ram[LIVES] - 1
            };
            self.ram[PX] = 16;
            self.ram[PY] = 16;
        }
        let f =
            u32::from_le_bytes(self.ram[FRAMES..FRAMES + 4].try_into().unwrap()).wrapping_add(1);
        self.ram[FRAMES..FRAMES + 4].copy_from_slice(&f.to_le_bytes());
        self.render();
    }

    fn serialize_state(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(MAGIC.len() + RAM_SIZE + self.fb.len());
        v.extend_from_slice(MAGIC);
        v.extend_from_slice(&self.ram);
        v.extend_from_slice(&self.fb);
        v
    }

    fn unserialize_state(&mut self, data: &[u8]) -> Result<(), StateError> {
        let total = MAGIC.len() + RAM_SIZE + FB_WIDTH * FB_HEIGHT;
        if data.len() < total {
            return Err(StateError::Truncated);
        }
        if data.len() != total {
            return Err(StateError::BadLength);
        }
        if &data[..4] != MAGIC {
            return Err(StateError::BadMagic);
        }
        let (ram, fb) = data[4..].split_at(RAM_SIZE);
        // Validate coordinates so corrupt data cannot cause out-of-range rendering.
        let ok = [
            (PX, FB_WIDTH),
            (CX, FB_WIDTH),
            (EX, FB_WIDTH),
            (PY, FB_HEIGHT),
            (CY, FB_HEIGHT),
            (EY, FB_HEIGHT),
        ]
        .iter()
        .all(|&(i, lim)| (ram[i] as usize) < lim);
        if !ok {
            return Err(StateError::BadLength);
        }
        self.ram.copy_from_slice(ram);
        self.fb.copy_from_slice(fb);
        Ok(())
    }
}
