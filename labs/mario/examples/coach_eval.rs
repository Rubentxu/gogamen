//! `cargo run -p gogamen-lab-mario --example coach_eval [--seeds N]`
//! With/without coach comparison on the SYNTHETIC fixture (not Mario).
//! With a real ROM this is NOT_RUN (no emulator backend).
use gogamen_lab_mario::SYNTHETIC_LABEL;
use gogamen_lab_mario::coach::eval::{FULL, MYOPIC, evaluate, suite_hash, summary};
use gogamen_lab_mario::metrics::RunConfig;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n: u64 = args
        .iter()
        .position(|a| a == "--seeds")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(16);
    let seeds: Vec<u64> = (1..=n).collect();
    let rows = evaluate(&seeds, &RunConfig::default(), &[MYOPIC, FULL]);
    println!("backend: {SYNTHETIC_LABEL}; real ROM: NOT_RUN");
    println!("seeds: {seeds:?}");
    print!("{}", summary(&rows));
    println!("suite_hash: {}", suite_hash(&rows));
}
