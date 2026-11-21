//! Integrated Studio tour (GOG-116): the six real labs (Grid, CartPole, Racing/Kart, Life,
//! Language World, Automation) plus a replay session in the common shell.
//!
//! ```text
//! cargo run -p gogamen-studio --features window --example studio_tour                   # interactive
//! cargo run -p gogamen-studio --features window --example studio_tour -- --script /tmp/g116
//! ```
//! `--script DIR`: for each lab it selects it, advances it a fixed number of ticks through the
//! control port (deterministic), and captures the real window as `DIR/NN_<lab>.png`. It prints one
//! `shot` line per lab, then `render_test_status=...`. Without a display it prints `NOT_RUN` and
//! exits 0 (never a pass). Interactive: lab selector, Step/Reset buttons, drag = pan, wheel = zoom.

use gogamen_studio::window::{run_tour, run_tour_interactive};
use gogamen_studio::{RenderTestStatus, has_graphical_display};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !has_graphical_display(|k| std::env::var(k).ok()) {
        println!(
            "NOT_RUN: {:?} (no DISPLAY/WAYLAND_DISPLAY)",
            RenderTestStatus::NotRunHeadlessHost
        );
        return;
    }
    let Some(i) = args.iter().position(|a| a == "--script") else {
        run_tour_interactive().expect("window session");
        return;
    };
    let dir = args.get(i + 1).map_or("/tmp/g116", String::as_str);
    let report = run_tour(std::path::Path::new(dir)).expect("window session");
    println!("gl_renderer={}", report.gl_renderer);
    println!("gl_version={}", report.gl_version);
    for (s, f) in report.shots.iter().zip(&report.tour) {
        println!(
            "shot name={} lab={} tick={} size={}x{} hash={:016x} scene_hash={:016x} scene_pixels={} colors={} programmatic={} path={}",
            s.name,
            s.lab,
            f.tick,
            s.size.0,
            s.size.1,
            s.hash,
            f.scene_hash,
            s.scene_pixels,
            s.distinct_colors,
            s.programmatic,
            s.path
                .as_ref()
                .map_or("-".into(), |p| p.display().to_string())
        );
    }
    let r = report.to_render_test_report();
    println!("render_test_status={:?} backend={:?}", r.status, r.backend);
    if r.status != RenderTestStatus::Pass
        || report.shots.len() != gogamen_studio::tour::TOUR_PLAN.len()
    {
        std::process::exit(1);
    }
}
