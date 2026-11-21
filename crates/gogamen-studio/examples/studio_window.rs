//! Real-window Studio viewport (needs the `window` feature and a graphical session).
//!
//! ```text
//! cargo run -p gogamen-studio --features window --example studio_window            # interactive
//! cargo run -p gogamen-studio --features window --example studio_window -- --script /tmp/g89
//! ```
//! Interactive: drag = pan, wheel = zoom at cursor, double-click or `Fit` = re-fit, resizing the
//! window re-fits, the top bar switches between the two fixture labs.
//! `--script DIR`: deterministic zoom/pan, window resize and lab switch (all PROGRAMMATIC input),
//! writing one PNG per step plus a summary on stdout.

use gogamen_studio::StudioShell;
use gogamen_studio::fixtures::fixture_entries;
use gogamen_studio::window::{run_interactive, run_script};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let shell = StudioShell::new(fixture_entries(), 900.0, 600.0).expect("fixture labs");
    if let Some(i) = args.iter().position(|a| a == "--script") {
        let dir = args.get(i + 1).map_or("/tmp/g89", String::as_str);
        let report = run_script(shell, std::path::Path::new(dir)).expect("window session");
        println!("gl_renderer={}", report.gl_renderer);
        println!("gl_version={}", report.gl_version);
        for s in &report.shots {
            println!(
                "shot name={} size={}x{} hash={:016x} scene_pixels={} colors={} lab={} zoom={:.3} programmatic={} path={}",
                s.name,
                s.size.0,
                s.size.1,
                s.hash,
                s.scene_pixels,
                s.distinct_colors,
                s.lab,
                f64::from_bits(s.camera_zoom),
                s.programmatic,
                s.path
                    .as_ref()
                    .map_or("-".into(), |p| p.display().to_string())
            );
        }
        let r = report.to_render_test_report();
        println!("render_test_status={:?} backend={:?}", r.status, r.backend);
        if !report.all_distinct_and_visible() {
            std::process::exit(1);
        }
    } else {
        run_interactive(shell).expect("window session");
    }
}
