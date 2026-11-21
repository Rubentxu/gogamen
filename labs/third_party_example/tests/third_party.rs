//! AC1 (GOG-103): un consumidor externo registra lab + escena + campos de inspector usando sólo
//! APIs públicas y el shell lo lista, lo carga y lo renderiza.

use gogamen_example_third_party::{LAB_ID, orbit_registration};
use gogamen_studio::control::{CommandId, ControlAction, ControlCommand, ControlPort};
use gogamen_studio::inspector::Lod;
use gogamen_studio::{ObservationPort, StudioShell, register_all};

#[test]
fn shell_lists_loads_and_renders_third_party_lab() {
    let mut labs = register_all([orbit_registration()]).unwrap();
    let lab = labs.remove(0);
    // campos de inspector declarados por el tercero
    let d = lab.inspector_schema().get("angle").expect("angle field");
    assert_eq!(d.min_lod, Lod::Minimal);
    assert!(d.label.contains("rad"));
    // el esquema de parámetros procede del descriptor de dominio
    assert_eq!(
        lab.into_entry().param_schema().descriptors()[0].id.0,
        "radius"
    );

    let entry = register_all([orbit_registration()])
        .unwrap()
        .remove(0)
        .into_entry();
    let mut shell = StudioShell::new(vec![entry], 320.0, 240.0).unwrap();
    assert_eq!(shell.lab_ids(), [LAB_ID]);
    shell.select_lab(LAB_ID).unwrap();
    let before = shell.render_svg().unwrap();
    assert!(before.contains("<svg") && before.contains("<circle"));

    // avanzar un tick por el puerto de control y sincronizar: la escena cambia
    let control = shell.control_port().unwrap().unwrap();
    let obs = shell.observation_port().unwrap();
    for i in 0..3 {
        control.submit(ControlCommand {
            id: CommandId(i + 1),
            session: obs.observe().session,
            action: ControlAction::Step,
        });
    }
    shell.sync_active();
    assert_ne!(shell.render_svg().unwrap(), before);
}

#[test]
fn headless_run_needs_no_shell() {
    use gogamen_example_third_party::{OrbitProjector, orbit_projection};
    let scenes =
        gogamen_projection::run_headless(&mut OrbitProjector::default(), &orbit_projection(), 4)
            .unwrap();
    assert_eq!(scenes.len(), 5);
}
