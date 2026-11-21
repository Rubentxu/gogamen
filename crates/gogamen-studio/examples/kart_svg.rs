//! Renderiza el lab Kart a SVG tras N pasos
//! (`cargo run -p gogamen-studio --example kart_svg -- [steps] [seed]`). Escribe el SVG por stdout.

use gogamen_core::MasterSeed;
use gogamen_lab_racing::{KinematicBicycleConfig, TrackMap};
use gogamen_studio::StudioShell;
use gogamen_studio::control::{
    CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
};
use gogamen_studio::kart_lab::{KART_LAB_ID, kart_lab_entry};

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let entry = kart_lab_entry(
        KinematicBicycleConfig::default(),
        TrackMap::rectangular_circuit(),
        MasterSeed::new(seed),
    )
    .expect("config");
    let mut shell = StudioShell::new(vec![entry], 800.0, 600.0).expect("shell");
    shell.select_lab(KART_LAB_ID).expect("lab");
    let control = shell
        .control_port()
        .expect("session")
        .expect("controllable");
    let observe = shell.observation_port().expect("session");
    for i in 0..steps {
        control.submit(ControlCommand {
            id: CommandId(i + 1),
            session: observe.observe().session,
            action: ControlAction::Step,
        });
    }
    shell.sync_active();
    print!("{}", shell.render_svg().expect("active session"));
}
