//! Graba un run Grid y lo reabre sin motor (`cargo run -p gogamen-studio --example replay_svg -- [ticks] [seek] [seed]`).
//! SVG del tick buscado por stdout; resumen por stderr.

use gogamen_lab_grid::GridMap;
use gogamen_studio::recorder::record_grid;
use gogamen_studio::replay::{Recording, ReplayPlayer};

fn main() {
    let mut a = std::env::args().skip(1);
    let ticks: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(12);
    let seek: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let bytes = record_grid(seed, GridMap::cul_de_sac_map(), ticks)
        .expect("record")
        .to_bytes();
    let rec = Recording::open(&bytes).expect("open");
    let mut player = ReplayPlayer::new(&rec, 480.0, 360.0);
    player.seek(seek);
    eprintln!("{:?}", player.view());
    print!("{}", player.render_svg().unwrap_or_default());
}
