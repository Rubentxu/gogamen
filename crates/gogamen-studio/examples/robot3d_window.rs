//! Real-window 3D robot view (GOG-105, feature `window`).
//!
//! ```text
//! cargo run -p gogamen-studio --features window --example robot3d_window                  # interactive
//! cargo run -p gogamen-studio --features window --example robot3d_window -- --script /tmp/g105
//! ```
//! Interactive: left-drag = orbit, right-drag (or shift+drag) = pan, wheel = zoom, click = select,
//! double-click = reset view; Step/Reset buttons drive the control port. `--script DIR`: drives
//! the view programmatically (initial, orbit+zoom, pan/orthographic, stale/uncertainty) and
//! captures the window framebuffer as `DIR/NN_<state>.png`. Without a display it prints `NOT_RUN`.

use gogamen_studio::window::{run_robot3d_interactive, run_robot3d_shots};
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
        run_robot3d_interactive().expect("window session");
        return;
    };
    let dir = args.get(i + 1).map_or("/tmp/g105", String::as_str);
    let report = run_robot3d_shots(std::path::Path::new(dir)).expect("window session");
    println!("gl_renderer={}", report.gl_renderer);
    println!("gl_version={}", report.gl_version);
    for s in &report.shots {
        println!(
            "shot name={} size={}x{} hash={:016x} scene_pixels={} colors={} programmatic={} path={}",
            s.name,
            s.size.0,
            s.size.1,
            s.hash,
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
    if r.status != RenderTestStatus::Pass || report.shots.len() < 3 {
        std::process::exit(1);
    }
}
