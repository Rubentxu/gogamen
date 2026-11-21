//! Renderiza el lab CartPole a SVG tras N pasos
//! (`cargo run -p gogamen-studio --example cartpole_svg -- [steps] [seed]`). SVG por stdout.

use gogamen_core::MasterSeed;
use gogamen_studio::StudioShell;
use gogamen_studio::cartpole_lab::{
    CARTPOLE_LAB_ID, cartpole_demo_config, cartpole_lab_entry_with_probe,
};
use gogamen_studio::control::{
    CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(42);
    let (entry, _probe) =
        cartpole_lab_entry_with_probe(cartpole_demo_config(), MasterSeed::new(seed))
            .expect("config");
    let mut shell = StudioShell::new(vec![entry], 800.0, 500.0).expect("shell");
    shell.select_lab(CARTPOLE_LAB_ID).expect("lab");
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
        shell.sync_active();
    }
    print!("{}", shell.render_svg().expect("active session"));
}
