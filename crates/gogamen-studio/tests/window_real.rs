//! Real graphical check (AC1 of GOG-89). `harness = false` because winit must own the main
//! thread. Without a display it prints `NOT_RUN` (never a pass) and exits 0; with a display it
//! really opens a window, drives the script and asserts on the captured pixels.
//! Output dir: `GOGAMEN_STUDIO_CAPTURE_DIR` (default `/tmp/g89`).

use gogamen_studio::fixtures::fixture_entries;
use gogamen_studio::window::run_script;
use gogamen_studio::{RenderTestStatus, StudioShell, has_graphical_display};

fn main() {
    if !has_graphical_display(|k| std::env::var(k).ok()) {
        println!(
            "NOT_RUN: {:?} (no DISPLAY/WAYLAND_DISPLAY)",
            RenderTestStatus::NotRunHeadlessHost
        );
        return;
    }
    let dir = std::env::var("GOGAMEN_STUDIO_CAPTURE_DIR").unwrap_or_else(|_| "/tmp/g89".into());
    let shell = StudioShell::new(fixture_entries(), 900.0, 600.0).unwrap();
    let report = run_script(shell, std::path::Path::new(&dir)).expect("window session");
    for s in &report.shots {
        println!(
            "shot {} {}x{} hash={:016x} scene_pixels={} lab={}",
            s.name, s.size.0, s.size.1, s.hash, s.scene_pixels, s.lab
        );
    }
    assert_eq!(report.shots.len(), 4, "expected 4 captures");
    assert!(
        report.shots.iter().all(|s| s.scene_pixels > 0),
        "scene not visible"
    );
    let (a, b, c, d) = (
        &report.shots[0],
        &report.shots[1],
        &report.shots[2],
        &report.shots[3],
    );
    assert_ne!(a.hash, b.hash, "zoom/pan did not change the frame");
    assert_ne!(a.hash, c.hash, "resize did not change the frame");
    assert_ne!(b.hash, c.hash);
    assert_ne!(c.size, a.size, "window size did not change");
    assert_ne!(d.hash, c.hash, "lab switch did not change the frame");
    assert_eq!(
        report.to_render_test_report().status,
        RenderTestStatus::Pass
    );
    println!(
        "PASS: real window ({} / {})",
        report.gl_renderer, report.gl_version
    );
}
