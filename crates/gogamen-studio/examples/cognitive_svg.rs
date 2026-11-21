//! Inspector cognitivo a SVG (`cargo run -p gogamen-studio --example cognitive_svg -- [grid|life] [before|after] [confidential|authorized]`).
//! SVG por stdout.

use gogamen_studio::cognitive::{Authorization, render_cognitive_svg};
use gogamen_studio::cognitive_fixtures::{grid_language_fixture, life_signal_fixture};

fn main() {
    let mut a = std::env::args().skip(1);
    let which = a.next().unwrap_or_else(|| "grid".into());
    let phase = a.next().unwrap_or_else(|| "after".into());
    let auth = match a.next().as_deref() {
        Some("authorized") => Authorization::Authorized,
        _ => Authorization::default(),
    };
    let view = if which == "life" {
        life_signal_fixture(auth)
    } else {
        grid_language_fixture(phase != "before", auth)
    };
    print!("{}", render_cognitive_svg(&view, 1000.0, 420.0));
}
