//! Lado motor de GOG-94: escribe una [`Recording`] desde el [`GridRollout`] real.
//! El reproductor (`replay.rs`) nunca importa este módulo.

use gogamen_core::MasterSeed;
use gogamen_lab_grid::GridMap;
use gogamen_lab_grid::world::GridWorldConfig;

use crate::grid_lab::GridRollout;
use crate::replay::{Frame, OpenError, Recording, map_to_rows};

/// Graba hasta `ticks` pasos (o hasta el estado terminal) del Grid real. El frame del tick 0 es
/// el mundo inicial; `revision` = índice de frame (un paso = una revisión).
pub fn record_grid(seed: u64, map: GridMap, ticks: u64) -> Result<Recording, OpenError> {
    let rows = map_to_rows(&map);
    let config = GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    };
    let mut rollout = GridRollout::new(config, MasterSeed::new(seed))
        .map_err(|e| OpenError::Corrupt(format!("engine: {e}")))?;
    let frame = |r: &GridRollout, revision: u64| {
        let v = r.view();
        Frame {
            tick: v.tick.as_u64(),
            revision,
            agent: v.agent,
            last_reward: v.last_reward,
            total_reward: v.total_reward,
            collisions: v.collisions,
            terminal: v.terminal,
        }
    };
    let mut frames = vec![frame(&rollout, 0)];
    for i in 0..ticks {
        match rollout.step() {
            Ok(Some(_)) => frames.push(frame(&rollout, i + 1)),
            Ok(None) => break,
            Err(e) => return Err(OpenError::Corrupt(format!("engine: {e}"))),
        }
    }
    let last = frames.last().map_or(0, |f| f.tick);
    Recording::new(seed, rows, last, frames)
}
