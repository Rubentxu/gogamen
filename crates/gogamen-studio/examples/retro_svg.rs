//! Framebuffer retro + overlays a SVG (`cargo run -p gogamen-studio --example retro_svg -- [privileged|pixel] [detect]`).
//! SVG por stdout. Fuente: `FreePlayCore` (homebrew generado por código, sin ROM).

use gogamen_lab_retro::AdapterConfig;
use gogamen_studio::retro_session::{EmuCommand, RetroHost};
use gogamen_studio::retro_view::{PerceptionMode, render_retro_svg};

fn main() {
    let mut a = std::env::args().skip(1);
    let mode = match a.next().as_deref() {
        Some("pixel") => PerceptionMode::PixelOnly,
        _ => PerceptionMode::Privileged,
    };
    let detect = a.next().as_deref() == Some("detect");
    let cfg = AdapterConfig {
        goal_score: u16::MAX,
        max_steps: 10_000,
        ..AdapterConfig::default()
    };
    let mut host = RetroHost::headless(cfg).expect("adapter");
    host.submit(EmuCommand::Input(8)).expect("input");
    for _ in 0..12 {
        host.tick().expect("tick");
    }
    let view = host.current_view(detect).expect("view");
    print!("{}", render_retro_svg(&view, mode, 330, 360).expect("svg"));
}
