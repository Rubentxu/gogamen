//! GOG-111: nodo de imagen texturizada: validación y JSON canónico estable.

use gogamen_core::{EpistemicKind, Tick};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive, Texture,
};

fn scene() -> Scene2D {
    let tex = Texture::indexed8(
        2,
        2,
        vec![[0, 0, 0, 255], [255, 0, 0, 255]],
        vec![0, 1, 1, 0],
    )
    .unwrap();
    let prim = ScenePrimitive::image(Point2::ZERO, 2.0, 2.0, tex).unwrap();
    let mut s = Scene2D::new(
        SceneId::new(1),
        Tick::new(0),
        EpistemicKind::Observed,
        CoordinateSpace::meters(),
    );
    s.add_layer(SceneLayer::new(LayerId::Environment)).unwrap();
    s.add_node(
        LayerId::Environment,
        SceneNode::new(
            SceneNodeId::new(1),
            None,
            EpistemicKind::Observed,
            prim,
            false,
        ),
    )
    .unwrap();
    s
}

#[test]
fn image_json_is_canonical_and_stable() {
    let a = scene().to_canonical_json();
    assert_eq!(a, scene().to_canonical_json());
    assert!(a.contains("\"kind\":\"image\""));
    assert!(a.contains("\"format\":\"indexed8\""));
    assert!(a.contains("\"palette\":\"000000ffff0000ff\""));
    assert!(a.contains("\"pixels\":\"01000100\"") || a.contains("\"pixels\":\"00010100\""));
}

#[test]
fn invalid_images_are_rejected() {
    let tex = Texture::rgba8(1, 1, vec![0; 4]).unwrap();
    assert!(ScenePrimitive::image(Point2::ZERO, -1.0, 1.0, tex.clone()).is_err());
    assert!(ScenePrimitive::image(Point2::ZERO, f64::NAN, 1.0, tex).is_err());
    assert!(Texture::indexed8(1, 1, vec![[0; 4]], vec![1]).is_err());
}
