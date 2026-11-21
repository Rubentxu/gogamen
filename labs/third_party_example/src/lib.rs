//! Lab de ejemplo de un tercero (GOG-103, API **v0**): un punto que orbita.
//!
//! Usa **sólo** APIs públicas: `gogamen_sdk::lab` (dominio), `gogamen_projection` (contrato
//! gráfico y escena) y `gogamen_studio::LabRegistration` (registro). No modifica Studio.
//! [`OrbitProjector`] y [`orbit_projection`] no necesitan Studio: ejecutan headless con
//! `gogamen_projection::run_headless`.

use std::sync::Arc;

use gogamen_core::{EpistemicKind, Tick};
use gogamen_projection::coords::ScaleUnits;
use gogamen_projection::{
    CoordinateSpace, FrameLimits, InspectorFieldDecl, LabProjector, LayerId, Point2,
    ProjectionError, ProjectionRegistration, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive, viz_spec::InspectorLod,
};
use gogamen_sdk::lab::{LabDescriptor, LabParam, LabUnit};
use gogamen_studio::LabRegistration;

/// Id del lab de ejemplo.
pub const LAB_ID: &str = "example.orbit";
/// Radio de la órbita en metros.
pub const RADIUS_M: f64 = 2.0;

/// Contrato de dominio (sin tipos gráficos).
pub fn orbit_descriptor() -> LabDescriptor {
    LabDescriptor::new(LAB_ID, "Orbit (third party)", "0.1.0").with_param(LabParam {
        id: "radius".into(),
        label: "Radius".into(),
        unit: LabUnit::Meters,
        min: 0.5,
        max: 5.0,
        default: RADIUS_M,
    })
}

/// Contrato gráfico.
pub fn orbit_projection() -> ProjectionRegistration {
    ProjectionRegistration::new(
        LAB_ID,
        "example.orbit.view",
        ScaleUnits::Meters,
        FrameLimits {
            max_nodes: 8,
            max_bytes: 4096,
        },
    )
    .with_field(InspectorFieldDecl {
        name: "angle".into(),
        label: "Angle".into(),
        unit: "rad".into(),
        min_lod: InspectorLod::Minimal,
    })
}

/// Proyector: el ángulo avanza `0.1 rad` por tick.
#[derive(Default)]
pub struct OrbitProjector {
    tick: u64,
}

impl LabProjector for OrbitProjector {
    fn advance_to(&mut self, tick: Tick) {
        self.tick = tick.as_u64();
    }

    fn project(&self) -> Result<Scene2D, ProjectionError> {
        let a = self.tick as f64 * 0.1;
        let mut s = Scene2D::new(
            SceneId::new(self.tick + 1),
            Tick::new(self.tick),
            EpistemicKind::Observed,
            CoordinateSpace::meters(),
        );
        s.add_layer(SceneLayer::new(LayerId::Environment))?;
        let nodes = [
            ScenePrimitive::circle(Point2::new(0.0, 0.0)?, RADIUS_M)?,
            ScenePrimitive::circle(Point2::new(RADIUS_M * a.cos(), RADIUS_M * a.sin())?, 0.2)?,
        ];
        for (i, p) in nodes.into_iter().enumerate() {
            s.add_node(
                LayerId::Environment,
                SceneNode::new(
                    SceneNodeId::new(i as u64 + 1),
                    None,
                    EpistemicKind::Observed,
                    p,
                    true,
                ),
            )?;
        }
        Ok(s)
    }
}

/// Paquete de registro completo, listo para `LabRegistration::register`.
pub fn orbit_registration() -> LabRegistration {
    LabRegistration {
        descriptor: orbit_descriptor(),
        projection: orbit_projection(),
        factory: Arc::new(|| Box::new(OrbitProjector::default())),
        specialized_view: None,
    }
}
