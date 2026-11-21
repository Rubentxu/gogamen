//! Depurador contrafactual (`cargo run -p gogamen-studio --example counterfactual_svg -- [tick] [alt_action 0..4] [seed]`).
//! SVG por stdout; resumen por stderr. Sólo lee una grabación: no hay rebobinado ni efectos externos.

use gogamen_core::{Action, DiscreteAction};
use gogamen_lab_grid::GridMap;
use gogamen_studio::counterfactual::{
    CounterfactualDebugger, counterfactual_lines, render_counterfactual_svg,
};
use gogamen_studio::recorder::record_grid;
use gogamen_studio::replay::Recording;

fn main() {
    let mut a = std::env::args().skip(1);
    let tick: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(3);
    let alt: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(4);
    let seed: u64 = a.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let bytes = record_grid(seed, GridMap::cul_de_sac_map(), 12)
        .expect("record")
        .to_bytes();
    let rec = Recording::open(&bytes).expect("open");
    let action = Action::Discrete(DiscreteAction::new(alt, 5).expect("alt action in 0..5"));
    match CounterfactualDebugger::compare(&rec, tick, action, 8) {
        Ok(view) => {
            for l in counterfactual_lines(&view) {
                eprintln!("{l}");
            }
            print!("{}", render_counterfactual_svg(&view, 640.0, 480.0));
        }
        Err(e) => eprintln!("checkpoint rejected: {e}"),
    }
}
