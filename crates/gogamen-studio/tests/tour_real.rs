//! Real graphical tour check (GOG-116). `harness = false` because winit must own the main
//! thread, and a process can open only one event loop: the parent re-executes this binary twice
//! (child mode, `GOGAMEN_TOUR_CHILD=<dir>`) to prove that the window captures are stable across
//! two independent runs.
//!
//! Without a display it prints `NOT_RUN` (never a pass) and exits 0. With a display it asserts:
//! one capture per lab, visible scene, pairwise distinct captures, hash-stable across 2 runs.
//! Output dir: `GOGAMEN_STUDIO_CAPTURE_DIR` (default `/tmp/g116`; run 2 uses `<dir>/run2`).

use std::path::Path;
use std::process::Command;

use gogamen_studio::tour::TOUR_PLAN;
use gogamen_studio::window::run_tour;
use gogamen_studio::{RenderTestStatus, has_graphical_display};

fn child(dir: &str) {
    let report = run_tour(Path::new(dir)).expect("window session");
    println!("gl_renderer={}", report.gl_renderer);
    for (s, f) in report.shots.iter().zip(&report.tour) {
        println!(
            "SHOT lab={} tick={} hash={:016x} scene_hash={:016x} scene_pixels={} size={}x{}",
            s.lab, f.tick, s.hash, f.scene_hash, s.scene_pixels, s.size.0, s.size.1
        );
    }
    println!("STATUS {:?}", report.to_render_test_report().status);
}

fn run_child(dir: &str) -> Vec<String> {
    let exe = std::env::current_exe().expect("current exe");
    let out = Command::new(exe)
        .env("GOGAMEN_TOUR_CHILD", dir)
        .output()
        .expect("spawn child");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "child failed: {}\n{text}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    text.lines().map(str::to_owned).collect()
}

fn main() {
    if let Ok(dir) = std::env::var("GOGAMEN_TOUR_CHILD") {
        child(&dir);
        return;
    }
    if !has_graphical_display(|k| std::env::var(k).ok()) {
        println!(
            "NOT_RUN: {:?} (no DISPLAY/WAYLAND_DISPLAY)",
            RenderTestStatus::NotRunHeadlessHost
        );
        return;
    }
    let dir = std::env::var("GOGAMEN_STUDIO_CAPTURE_DIR").unwrap_or_else(|_| "/tmp/g116".into());
    let run1 = run_child(&dir);
    let run2 = run_child(&format!("{dir}/run2"));
    let shots = |lines: &[String]| -> Vec<String> {
        lines
            .iter()
            .filter(|l| l.starts_with("SHOT "))
            .cloned()
            .collect()
    };
    let (a, b) = (shots(&run1), shots(&run2));
    for l in &a {
        println!("run1 {l}");
    }
    for l in &b {
        println!("run2 {l}");
    }
    assert_eq!(a.len(), TOUR_PLAN.len(), "one capture per lab");
    assert!(run1.iter().any(|l| l == "STATUS Pass"), "run1 status");
    assert!(run2.iter().any(|l| l == "STATUS Pass"), "run2 status");
    assert_eq!(a, b, "captures must be hash-stable across two runs");
    for l in &a {
        let px: usize = l
            .split("scene_pixels=")
            .nth(1)
            .and_then(|r| r.split_whitespace().next())
            .and_then(|v| v.parse().ok())
            .expect("scene_pixels");
        assert!(px > 0, "scene not visible: {l}");
    }
    let hashes: Vec<&str> = a
        .iter()
        .map(|l| {
            l.split(" hash=")
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
                .unwrap()
        })
        .collect();
    for i in 0..hashes.len() {
        for j in i + 1..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "captures {i} and {j} identical");
        }
    }
    println!(
        "PASS: real window tour, {} labs, stable across 2 runs",
        a.len()
    );
}
