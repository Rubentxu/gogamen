//! Spike measurements (informational, no assertions).
//! Run: cargo run --release -p gogamen-lab-retro --example bench
use gogamen_lab_retro::{FreePlayCore, Session};
use std::time::Instant;

fn main() {
    let mut s = Session::new(FreePlayCore::new());
    let n = 100_000u64;
    let t = Instant::now();
    for i in 0..n {
        s.step(((i * 2654435761) >> 7) as u16 & 0x1f);
    }
    let secs = t.elapsed().as_secs_f64();
    println!(
        "throughput: {:.0} frames/s ({n} frames in {:.1} ms)",
        n as f64 / secs,
        secs * 1e3
    );

    let reps = 10_000;
    let t = Instant::now();
    let mut snap = Vec::new();
    for _ in 0..reps {
        snap = std::hint::black_box(s.snapshot());
    }
    let snap_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    let t = Instant::now();
    for _ in 0..reps {
        s.restore(std::hint::black_box(&snap)).unwrap();
    }
    let rest_us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    println!(
        "snapshot: {} bytes, {snap_us:.2} us; restore: {rest_us:.2} us",
        snap.len()
    );
    println!("extra processes: 0 (in-process core)");
}
