//! Ensayo de aislamiento temporal del visor (GOG-115).
//! `cargo run --release -p gogamen-studio --offline --example viewer_isolation -- [ticks] [seed] [wall_ms]`
//! Imprime el informe markdown (no es WCET).

use std::time::Duration;

use gogamen_studio::viewer_isolation::{CountingAllocator, IsolationConfig, IsolationReport};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut cfg = IsolationConfig::default();
    if let Some(t) = args.next().and_then(|s| s.parse().ok()) {
        cfg.ticks = t;
    }
    if let Some(s) = args.next().and_then(|s| s.parse().ok()) {
        cfg.seed = s;
    }
    let wall = args
        .next()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(20);
    let report = IsolationReport::run(cfg, Some(Duration::from_millis(wall)));
    print!("{}", report.to_markdown());
}
