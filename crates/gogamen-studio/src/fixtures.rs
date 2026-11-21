//! Laboratorios fixture pequeños para el shell (no ligados a labs pesados).

use std::sync::Arc;

use gogamen_core::{EpistemicKind, Tick};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};
use gogamen_sdk::extensions::ExtensionId;

use crate::shell::{LabEntry, LabSession};

fn p(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite fixture point")
}

fn build(id: u64, nodes: Vec<(LayerId, ScenePrimitive, EpistemicKind)>) -> Scene2D {
    let mut s = Scene2D::new(
        SceneId::new(id),
        Tick::new(0),
        EpistemicKind::Observed,
        CoordinateSpace::meters(),
    );
    let mut seen = Vec::new();
    for (i, (layer, prim, kind)) in nodes.into_iter().enumerate() {
        if !seen.contains(&layer) {
            s.add_layer(SceneLayer::new(layer)).expect("unique layer");
            seen.push(layer);
        }
        s.add_node(
            layer,
            SceneNode::new(SceneNodeId::new(i as u64 + 1), None, kind, prim, true),
        )
        .expect("unique node");
    }
    s
}

/// Escena fixture A: un círculo y una etiqueta.
pub fn scene_alpha() -> Scene2D {
    build(
        1,
        vec![
            (
                LayerId::Environment,
                ScenePrimitive::circle(p(0.0, 0.0), 1.0).unwrap(),
                EpistemicKind::Observed,
            ),
            (
                LayerId::Environment,
                ScenePrimitive::label(p(1.0, 1.0), "a&b"),
                EpistemicKind::Observed,
            ),
        ],
    )
}

/// Escena fixture B: rectángulo y trayectoria hipotética.
pub fn scene_beta() -> Scene2D {
    build(
        2,
        vec![
            (
                LayerId::Environment,
                ScenePrimitive::rect(p(-5.0, -2.0), 10.0, 4.0).unwrap(),
                EpistemicKind::Observed,
            ),
            (
                LayerId::Imagination,
                ScenePrimitive::polyline(vec![p(-5.0, -2.0), p(0.0, 3.0), p(5.0, 2.0)], false)
                    .unwrap(),
                EpistemicKind::Hypothetical,
            ),
        ],
    )
}

struct FixtureSession {
    scene: Scene2D,
    _token: Option<Arc<()>>,
}

impl LabSession for FixtureSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }
}

/// Entrada de registro para un fixture; `token` (opcional) permite verificar liberación.
pub fn fixture_entry(
    id: &str,
    title: &str,
    scene: fn() -> Scene2D,
    token: Option<Arc<()>>,
) -> LabEntry {
    LabEntry::new(
        ExtensionId::new(id),
        title,
        Arc::new(move || {
            Box::new(FixtureSession {
                scene: scene(),
                _token: token.clone(),
            })
        }),
    )
}

/// Los dos labs fixture estándar (`fixture.alpha`, `fixture.beta`).
pub fn fixture_entries() -> Vec<LabEntry> {
    vec![
        fixture_entry("fixture.alpha", "Alpha", scene_alpha, None),
        fixture_entry("fixture.beta", "Beta", scene_beta, None),
    ]
}
