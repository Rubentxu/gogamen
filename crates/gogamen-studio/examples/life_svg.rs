//! Renderiza el lab Life a SVG tras N pasos, con un organismo seleccionado
//! (`cargo run -p gogamen-studio --example life_svg -- [steps] [seed]`). SVG por stdout.

use gogamen_core::MasterSeed;
use gogamen_studio::StudioShell;
use gogamen_studio::control::{
    CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
};
use gogamen_studio::inspector::EntityRef;
use gogamen_studio::life_lab::{
    LIFE_LAB_ID, life_demo_config, life_epoch, life_lab_entry_with_probe,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(7);
    let (entry, probe) =
        life_lab_entry_with_probe(life_demo_config(), MasterSeed::new(seed)).expect("config");
    let mut shell = StudioShell::new(vec![entry], 800.0, 800.0).expect("shell");
    shell.select_lab(LIFE_LAB_ID).expect("lab");
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
        // seleccionar el primer organismo vivo: sólo él muestra futuros FMC
        if let Some(o) = probe
            .snapshot()
            .and_then(|s| s.view.organisms.first().cloned())
        {
            probe.select(Some(EntityRef::new(o.id, life_epoch(&observe.observe()))));
        }
    }
    shell.sync_active();
    print!("{}", shell.render_svg().expect("active session"));
}
