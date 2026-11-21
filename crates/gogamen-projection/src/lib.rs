//! SDK de proyección 2D independiente del renderer para Gogamen (`std`-only).
//!
//! Este crate proyecta [`gogamen_core::SimulationSnapshot`] y
//! [`gogamen_core::VisualFrame`] hacia una escena 2D tipada ([`Scene2D`]) con
//! capas explícitas, primitivas geométricas (`Circle`, `Polyline`, `Rect`, `Label`),
//! soporte de escala/unidades ([`CoordinateSpace`], [`ScaleTransform`]), selección
//! estable por `EntityId` / `SceneNodeId` y serialización JSON canónica determinista.
//!
//! # Aislamiento de dependencias
//! `gogamen-projection` depende únicamente de `std`, `gogamen-core` y `gogamen-spec`
//! (primitivas JSON canónicas de la `VisualizationSpec` declarativa, [`viz_spec`]). No incluye motores gráficos (Bevy), GPU (`wgpu`), UI ni acceso a entorno vivo (`LiveEnvironment` / `Actuator`).

pub mod coords;
pub mod error;
pub mod image;
pub mod overlay;
pub mod projector;
pub mod registration;
pub mod scene;
pub mod scene3d;
pub mod viz_spec;

pub use coords::{CoordinateSpace, Point2, ScaleTransform, ScaleUnits};
pub use error::ProjectionError;
pub use image::{ImageViewport, MAX_TEXTURE_DIM, PixelFormat, Texture};
pub use overlay::{
    DashPattern, DecisionOverlayFrame, ExecutedTrajectory, FutureKind, FutureTrajectorySample,
    LayerVisibility, LegendCategory, LegendEntry, OverlayBudget, OverlayConfig, OverlayLegend,
    OverlayPalette, OverlayStats, OverlayStyle, ProjectedOverlay, RgbaColor, SensorRaySample,
    SensorSample, TerminalMarker, TerminalOutcome, project_decision_overlay,
};
pub use projector::{
    ProjectionConfig, Projector2D, project_frame_to_scene, project_snapshot_to_scene,
};
pub use registration::{
    FrameLimits, InspectorFieldDecl, LabProjector, ProjectionRegistration, ProjectionRegistry,
    RegistrationDiagnostic, RegistrationIssue, check_registration, check_scene, run_headless,
};
pub use scene::{
    LayerId, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId, ScenePrimitive, SelectionHit,
};
pub use scene3d::{
    MAX_DISTANCE, MAX_ELEVATION, MIN_DISTANCE, NEAR, Node3D, OrbitCamera, Primitive3D, Projection,
    Scene3D, ScreenPoint, Vec3,
};
pub use viz_spec::{
    InspectorFieldSpec, InspectorLod, InspectorSpec, LayerBinding, LayerDescriptor,
    ParamDescriptor, ParamKind, ParamValue, ProjectionDescriptor, SystemWithVisualization,
    VISUALIZATION_SPEC_SCHEMA_VERSION, ViewSpec, VisualizationCatalog, VisualizationSpec,
    VizDiagnostic, VizDiagnosticKind, VizSpecError, canonical_visualization_fixture,
    canonical_visualization_fixture_json,
};
