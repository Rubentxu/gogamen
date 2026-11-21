//! Compara dos runs Grid en vistas sincronizadas (`cargo run -p gogamen-studio --example arena_svg -- [tick] [seed_a] [seed_b]`).
//! SVG por stdout; veredicto y marcadores por stderr. Sin informe de arena => "not comparable".

use gogamen_lab_grid::GridMap;
use gogamen_studio::arena_view::{
    ArenaSession, project_arena_scene, render_arena_svg, verdict_line,
};
use gogamen_studio::recorder::record_grid;

fn main() {
    let mut a = std::env::args().skip(1);
    let tick: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(6);
    let sa: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let sb: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let left = record_grid(sa, GridMap::cul_de_sac_map(), 12).expect("record a");
    let right = record_grid(sb, GridMap::cul_de_sac_map(), 8).expect("record b");
    let mut session = ArenaSession::new(&left, &right, None);
    session.seek(tick);
    let view = session.view();
    eprintln!("{}", verdict_line(&view.verdict));
    eprintln!("{} | {}", view.left.run_id, view.right.run_id);
    let _ = project_arena_scene(&view);
    print!("{}", render_arena_svg(&view, 800.0, 400.0));
}
