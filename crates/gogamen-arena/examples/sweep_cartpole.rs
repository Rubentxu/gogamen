//! Minimal sweep example: writes `sweep.csv` and `sweep.svg` to the directory given as the
//! first argument (default: current directory) and prints the frontier report.
//!
//! `cargo run -p gogamen-arena --offline --example sweep_cartpole -- <out_dir>`

use std::path::PathBuf;

use gogamen_arena::ArenaSeedSet;
use gogamen_arena::sweep::{
    CartPoleEntropicSweep, Endowment, SweepAxes, SweepConfig, oracle_frontier, run_sweep,
};
use gogamen_core::MasterSeed;
use gogamen_lab_cartpole::EntropicExperimentConfig;

fn main() {
    let out: PathBuf = std::env::args_os()
        .nth(1)
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    let base = EntropicExperimentConfig {
        episode_ticks: 40,
        ..EntropicExperimentConfig::default()
    };
    let horizons = vec![1, 2, 4, 6, 8];
    let config = SweepConfig {
        sweep_id: "cartpole-entropic-horizon".to_owned(),
        axes: SweepAxes {
            horizons: horizons.clone(),
            scales: vec![4, 8, 12],
            budgets: vec![4],
            endowments: vec![Endowment {
                label: "pm10".to_owned(),
                actions: vec![-10.0, 0.0, 10.0],
            }],
        },
        seeds: ArenaSeedSet::new(
            (1..=4).map(MasterSeed::new).collect(),
            (101..=104).map(MasterSeed::new).collect(),
        )
        .expect("disjoint seeds"),
        budget_limit: 5_000_000,
        margin: 0.0,
    };
    let world = CartPoleEntropicSweep { base };
    let report = run_sweep(&config, &world).expect("sweep");
    let oracle = oracle_frontier(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &horizons);
    std::fs::write(out.join("sweep.csv"), report.to_csv()).expect("csv");
    std::fs::write(out.join("sweep.svg"), report.to_svg()).expect("svg");
    print!("{}", report.render_frontier_report(Some(&oracle)));
}
