//! Gogamen Studio: shell de laboratorios + cámara 2D + viewport por software (`std`-only).
//!
//! # Backends (GOG-89)
//! - [`viewport::SoftwareSvgBackend`] renderiza una `Scene2D` a SVG de forma determinista.
//! - [`viewport::ViewportBackend`] es la frontera de backends.
//! - Con la feature opcional `window` (off por defecto) el módulo `window` aporta un backend
//!   `BackendKind::Window` real basado en eframe/egui 0.37 (fallback del spike Bevy +
//!   `bevy_egui`; ver README). Sin la feature el build es `std`-only.
//!
//! `gogamen-core` y `gogamen-cli` no dependen de este crate.

pub mod arena_view;
pub mod automation;
pub mod camera;
pub mod cartpole_lab;
pub mod cognitive;
pub mod cognitive_fixtures;
pub mod control;
pub mod counterfactual;
pub mod fixtures;
pub mod grid_lab;
pub mod inspector;
pub mod kart_lab;
pub mod language_world_lab;
pub mod life_lab;
pub mod param_ui;
pub mod params;
pub mod recorder;
pub mod replay;
pub mod retro_session;
pub mod retro_view;
pub mod robot3d;
pub mod scenario_editor;
pub mod shell;
pub mod status;
pub mod third_party;
pub mod tour;
pub mod viewer_isolation;
pub mod viewport;
#[cfg(feature = "window")]
pub mod window;

pub use automation::{
    AutomationScenario, AutomationView, EffectStatus, ScriptInput, ScriptStep, incident_scenario,
    project_automation_scene, render_automation_svg, retry_error_scenario, run_automation,
    timeout_scenario,
};
pub use camera::{Bounds, Camera2D, MAX_ZOOM, MIN_ZOOM, scene_bounds};
pub use cognitive::{
    Authorization, CausalLink, CogRef, CognitiveView, EvidenceSource, EvidenceView, HypothesisView,
    IntentionView, Known, MessageView, project_cognitive_scene, render_cognitive_svg,
};
pub use control::{
    Ack, AckOutcome, CommandId, ControlAction, ControlCommand, ControlHandle, ControlPort,
    ControlUnavailable, MAX_SPEED, Observation, ObservationHandle, ObservationPort, ParamHandle,
    RejectReason, RejectionRecord, RunState, SessionHost, SessionId, SessionKind, SessionOptions,
};
pub use inspector::{
    EntityRef, EntityView, Field, FieldDescriptor, FieldSchema, FieldValue, Inspector,
    InspectorRow, InspectorStatus, InspectorView, Lod, Provenance, Quality, SnapshotView,
    StaleReason, VanishReason,
};
pub use param_ui::{
    ControlDescriptor, ControlWidget, DisabledReason, UiPermissions, control_descriptors,
};
pub use params::{
    Mutability, PARAM_REJECTION_LOG_CAP, ParamAck, ParamChange, ParamCommand, ParamControlPort,
    ParamDescriptor, ParamEvent, ParamId, ParamKind, ParamOutcome, ParamRejectReason,
    ParamRejectionRecord, ParamRequest, ParamSchema, ParamUnit, ParamValue, SchemaError,
};
pub use retro_session::{
    CommandOutcome, CommandRecord, EmuCommand, FramePacket, HOMEBREW_LICENSE, RetroHost,
};
pub use retro_view::{
    CommandCounters, FrameCounters, OverlayItem, PerceptionMode, PerceptionSource, RetroFrameView,
    project_retro_scene, render_retro_svg,
};
pub use robot3d::{
    ROBOT3D_ID, Robot3dFrame, Robot3dPhysics, Robot3dSession, Robot3dSnapshot, ScanStatus,
    Selection, ViewState, build_scene, robot3d_lab_entry,
};
pub use scenario_editor::{EditError, ResourceKind, ScenarioDraft, ScenarioEdit};
pub use shell::{
    LabEntry, LabFactory, LabSession, SelectOutcome, ShellError, StudioShell, ViewInput,
};
pub use status::{
    RenderTestReport, RenderTestStatus, has_graphical_display, run_window_render_test,
};
pub use third_party::{
    LabRegistration, ProjectorFactory, RegisterError, RegisteredLab, SpecializedView, register_all,
};
pub use viewport::{BackendKind, SoftwareSvgBackend, ViewportBackend};

#[cfg(test)]
mod dependency_isolation {
    /// AC3: ni `gogamen-core` ni `gogamen-cli` dependen de studio ni de crates gráficos.
    #[test]
    fn core_and_cli_manifests_do_not_reference_studio_or_graphics() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for krate in ["gogamen-core", "gogamen-cli"] {
            let text = std::fs::read_to_string(root.join(krate).join("Cargo.toml")).unwrap();
            for banned in ["gogamen-studio", "bevy", "egui", "eframe", "wgpu", "winit"] {
                assert!(!text.contains(banned), "{krate} references {banned}");
            }
        }
        let own = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"),
        )
        .unwrap();
        // Graphics may only appear as an optional dependency behind the `window` feature.
        for banned in ["bevy", "egui", "wgpu", "winit"] {
            assert!(
                !own.contains(&format!("{banned} =")),
                "studio declares {banned}"
            );
        }
        let eframe_line = own.lines().find(|l| l.starts_with("eframe =")).unwrap();
        assert!(eframe_line.contains("optional = true"));
        assert!(own.contains("default = []"));
    }
}
