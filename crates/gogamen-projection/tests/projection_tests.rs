use gogamen_core::{
    DeterministicRng, EntityId, EpistemicKind, EventSource, MonotonicInstant, RunId, SimDuration,
    SimulationSnapshot, State, StreamRevision, Terminal, Tick, VisualEntitySample, VisualFrame,
};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, ProjectionConfig, ProjectionError, Projector2D,
    ScaleTransform, ScaleUnits, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive, project_frame_to_scene, project_snapshot_to_scene,
};

fn build_fixture_frame(reversed: bool) -> VisualFrame {
    let mut entities = vec![
        VisualEntitySample::new(EntityId::new(10), [1.5, 2.0, 0.0], 0.4, 1).unwrap(),
        VisualEntitySample::new(EntityId::new(20), [-3.0, 4.5, 1.0], 0.8, 2).unwrap(),
        VisualEntitySample::new(EntityId::new(30), [0.0, -1.25, 0.0], 0.0, 3).unwrap(),
    ];
    if reversed {
        entities.reverse();
    }
    VisualFrame::new(
        RunId::new(42),
        Tick::new(7),
        SimDuration::from_secs_f64(0.7).unwrap(),
        MonotonicInstant::from_nanos(700_000_000),
        StreamRevision::new(7),
        EventSource::SimulationWorld,
        EpistemicKind::Observed,
        entities,
    )
    .unwrap()
}

fn build_fixture_snapshot() -> SimulationSnapshot {
    let state = State::new(
        Tick::new(12),
        SimDuration::from_secs_f64(1.2).unwrap(),
        vec![1.0, 2.0, 3.5, -1.5],
    )
    .unwrap();
    let rng = DeterministicRng::new(999);
    SimulationSnapshot::new(state, &rng, Terminal::Ongoing)
}

#[test]
fn ac1_frame_and_snapshot_projection_is_deterministic_and_serializable() {
    let config = ProjectionConfig::new(SceneId::new(100), CoordinateSpace::meters());
    let frame_a = build_fixture_frame(false);
    let frame_b = build_fixture_frame(true);

    let scene_a = project_frame_to_scene(&frame_a, &config).unwrap();
    let scene_b = project_frame_to_scene(&frame_b, &config).unwrap();

    assert_eq!(scene_a, scene_b);
    let json_a = scene_a.to_canonical_json();
    let json_b = scene_b.to_canonical_json();
    assert_eq!(json_a, json_b);
    assert!(json_a.contains("\"scene_id\":100"));
    assert!(json_a.contains("\"tick\":7"));
    assert!(json_a.contains("\"epistemic_kind\":\"observed\""));
    assert!(json_a.contains("\"id\":\"environment\""));
    assert!(json_a.contains("\"id\":\"labels\""));
    assert!(json_a.contains("\"kind\":\"circle\""));
    assert!(json_a.contains("\"kind\":\"label\""));

    let snapshot = build_fixture_snapshot();
    let snap_scene_1 = project_snapshot_to_scene(&snapshot, &config).unwrap();
    let snap_scene_2 = project_snapshot_to_scene(&snapshot, &config).unwrap();
    assert_eq!(
        snap_scene_1.to_canonical_json(),
        snap_scene_2.to_canonical_json()
    );
    let snap_json = snap_scene_1.to_canonical_json();
    assert!(snap_json.contains("\"kind\":\"polyline\""));
    assert!(snap_json.contains("\"kind\":\"rect\""));
    assert!(snap_json.contains("\"id\":\"decisions\""));
}

#[test]
fn ac1_hypothetical_and_inferred_frames_route_to_explicit_layers() {
    let frame_hyp = VisualFrame::new(
        RunId::new(1),
        Tick::new(3),
        SimDuration::ZERO,
        MonotonicInstant::from_nanos(10),
        StreamRevision::new(1),
        EventSource::Planner,
        EpistemicKind::Hypothetical,
        vec![VisualEntitySample::new(EntityId::new(5), [1.0, 1.0, 0.0], 0.2, 0).unwrap()],
    )
    .unwrap();

    let scene = Projector2D::default().project_frame(&frame_hyp).unwrap();
    let matches = scene.find_by_entity_id(EntityId::new(5));
    assert!(
        matches
            .iter()
            .any(|(layer, _)| *layer == LayerId::Imagination)
    );
    assert!(matches.iter().any(|(layer, _)| *layer == LayerId::Labels));
}

#[test]
fn ac2_different_scales_preserve_logical_positions_and_stable_id_picking() {
    let frame = build_fixture_frame(false);

    let meters_space = CoordinateSpace::meters();
    let cm_space = CoordinateSpace::new(
        ScaleUnits::Centimeters,
        0.01,
        Point2::new(1.0, -2.0).unwrap(),
        false,
    )
    .unwrap();

    let scene_m = project_frame_to_scene(
        &frame,
        &ProjectionConfig::new(SceneId::new(1), meters_space),
    )
    .unwrap();
    let scene_cm =
        project_frame_to_scene(&frame, &ProjectionConfig::new(SceneId::new(1), cm_space)).unwrap();

    for entity in frame.entities() {
        let id = entity.entity_id();
        let m_nodes = scene_m.find_by_entity_id(id);
        let cm_nodes = scene_cm.find_by_entity_id(id);
        assert_eq!(m_nodes.len(), cm_nodes.len());

        let (m_layer, m_node) = m_nodes
            .iter()
            .find(|(layer, _)| *layer == LayerId::Environment)
            .unwrap();
        let (cm_layer, cm_node) = cm_nodes
            .iter()
            .find(|(layer, _)| *layer == LayerId::Environment)
            .unwrap();
        assert_eq!(m_layer, cm_layer);
        assert_eq!(m_node.node_id(), cm_node.node_id());

        let world_from_m = meters_space
            .local_to_world(m_node.primitive().anchor_point())
            .unwrap();
        let world_from_cm = cm_space
            .local_to_world(cm_node.primitive().anchor_point())
            .unwrap();
        assert!(
            world_from_m.distance_to(world_from_cm) < 1e-12,
            "world positions differed across scales: {world_from_m} vs {world_from_cm}"
        );

        let hit_m = scene_m
            .pick_at_world(world_from_m, 0.05)
            .unwrap()
            .expect("expected hit in meters scene");
        let hit_cm = scene_cm
            .pick_at_world(world_from_cm, 0.05)
            .unwrap()
            .expect("expected hit in centimeters scene");

        assert_eq!(hit_m.entity_id, Some(id));
        assert_eq!(hit_cm.entity_id, Some(id));
        assert_eq!(hit_m.node_id, hit_cm.node_id);
        assert_eq!(hit_m.layer_id, hit_cm.layer_id);
    }

    let xform = ScaleTransform::new(meters_space, cm_space);
    let p_world = Point2::new(2.5, 1.0).unwrap();
    let p_cm = xform.transform_point(p_world).unwrap();
    let roundtrip = xform.inverse_transform_point(p_cm).unwrap();
    assert!(p_world.distance_to(roundtrip) < 1e-12);
}

#[test]
fn ac3_headless_scene_construction_and_validation_errors_work_without_ui() {
    let mut scene = Scene2D::new(
        SceneId::new(9),
        Tick::ZERO,
        EpistemicKind::Observed,
        CoordinateSpace::from_units(ScaleUnits::GridCells),
    );
    let mut layer = SceneLayer::new(LayerId::Custom(7));
    layer
        .add_node(SceneNode::new(
            SceneNodeId::new(1),
            Some(EntityId::new(1)),
            EpistemicKind::Observed,
            ScenePrimitive::rect(Point2::ZERO, 2.0, 3.0).unwrap(),
            true,
        ))
        .unwrap();
    scene.add_layer(layer).unwrap();

    let hit = scene
        .pick_at(Point2::new(1.0, 1.5).unwrap(), 0.0)
        .expect("inside rect");
    assert_eq!(hit.entity_id, Some(EntityId::new(1)));
    assert_eq!(hit.distance, 0.0);

    assert!(matches!(
        CoordinateSpace::new(ScaleUnits::Meters, 0.0, Point2::ZERO, true),
        Err(ProjectionError::InvalidDimension { .. })
    ));
    assert!(matches!(
        Point2::new(f64::NAN, 1.0),
        Err(ProjectionError::NonFiniteValue { .. })
    ));
    assert!(matches!(
        ScenePrimitive::polyline(vec![Point2::ZERO], false),
        Err(ProjectionError::InsufficientPolylinePoints { count: 1 })
    ));
}
