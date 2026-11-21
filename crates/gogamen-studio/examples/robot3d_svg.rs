//! Deterministic SVG of the 3D robot view (GOG-105), no window and no GPU.
//!
//! ```text
//! cargo run -p gogamen-studio --example robot3d_svg -- [out_dir]
//! ```
//! Renders three states through the common shell (`lab.robot3d`): `initial` (fresh scan),
//! `orbited` (camera orbit + zoom, same physics) and `stale` (after the scanner dropout: STALE
//! label + uncertainty ellipse). Prints one `svg` line per state with its FNV-1a hash; with
//! `out_dir` it also writes `NN_<state>.svg`. Running it twice prints identical hashes.

use gogamen_studio::control::{
    CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
};
use gogamen_studio::tour::fnv64;
use gogamen_studio::{ROBOT3D_ID, StudioShell, ViewInput, robot3d_lab_entry};

fn main() {
    let out_dir = std::env::args().nth(1);
    let mut shell = StudioShell::new(vec![robot3d_lab_entry()], 900.0, 600.0).expect("shell");
    shell.select_lab(ROBOT3D_ID).expect("lab");
    let control = shell
        .control_port()
        .expect("session")
        .expect("controllable");
    let observe = shell.observation_port().expect("session");
    let mut next = 0;
    let mut step_to = |shell: &mut StudioShell, ticks: u64| {
        while observe.observe().tick.as_u64() < ticks {
            next += 1;
            control.submit(ControlCommand {
                id: CommandId(next),
                session: observe.observe().session,
                action: ControlAction::Step,
            });
            shell.sync_active();
        }
    };
    let emit = |idx: usize, name: &str, shell: &StudioShell| {
        let svg = shell.render_svg().expect("active session");
        println!(
            "svg name={name} bytes={} hash={:016x}",
            svg.len(),
            fnv64(svg.as_bytes())
        );
        if let Some(dir) = &out_dir {
            std::fs::create_dir_all(dir).expect("create out dir");
            std::fs::write(format!("{dir}/{idx:02}_{name}.svg"), svg).expect("write svg");
        }
    };
    step_to(&mut shell, 6);
    emit(1, "initial", &shell);
    shell.view_input(ViewInput::Orbit {
        dx_px: 180.0,
        dy_px: -30.0,
    });
    shell.view_input(ViewInput::Zoom { factor: 1.25 });
    emit(2, "orbited", &shell);
    step_to(&mut shell, 30);
    emit(3, "stale", &shell);
}
