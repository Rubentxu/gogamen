//! GOG-102: `VisualizationSpec` (equivalencia JSON/Kotlin vía golden, diagnósticos localizables).
//!
//! El lado Kotlin (`kotlin/.../VisualizationDslTest`) compara su salida con los mismos goldens;
//! aquí no se ejecuta JVM: se verifica que el golden es exactamente el spec construido en Rust.

use gogamen_projection::{
    LayerBinding, ParamValue, ScaleUnits, ViewSpec, VisualizationCatalog, VisualizationSpec,
    VizDiagnosticKind, VizSpecError, canonical_visualization_fixture,
    canonical_visualization_fixture_json,
};
use gogamen_spec::{ExprUnit, SystemSpec, canonical_two_node_fixture};

const GOLDEN: &str = include_str!("../fixtures/visualization_canonical_v0.json");

fn view_with(layer: LayerBinding) -> VisualizationSpec {
    let mut v = ViewSpec::new("v", "scene2d", ScaleUnits::Meters);
    v.layers.push(layer);
    let mut s = VisualizationSpec::new("two_node_canonical_v0");
    s.views.push(v);
    s
}

fn first_diag(spec: &VisualizationSpec) -> (String, VizDiagnosticKind) {
    let d = spec.validate(&VisualizationCatalog::builtin());
    assert_eq!(d.len(), 1, "{d:?}");
    (d[0].path.clone(), d[0].kind.clone())
}

#[test]
fn ac1_golden_equals_rust_built_spec_with_identical_bytes_and_hash() {
    assert_eq!(GOLDEN, canonical_visualization_fixture_json());
    let parsed = VisualizationSpec::from_json(GOLDEN).expect("golden parses");
    let built = canonical_visualization_fixture();
    assert_eq!(parsed, built);
    assert_eq!(parsed.to_canonical_json().unwrap(), GOLDEN);
    assert_eq!(
        parsed.canonical_hash().unwrap(),
        built.canonical_hash().unwrap()
    );
}

#[test]
fn ac1_units_and_references_are_checked_against_catalog_and_system() {
    let spec = VisualizationSpec::from_json(GOLDEN).unwrap();
    let catalog = VisualizationCatalog::builtin();
    assert!(spec.validate(&catalog).is_empty());
    assert!(spec.check(&catalog).is_ok());
    assert!(
        spec.validate_against_system(&catalog, &canonical_two_node_fixture())
            .is_empty()
    );

    // Referencia a nodo inexistente y system_spec_id distinto.
    let other = SystemSpec::new("other_system");
    let d = spec.validate_against_system(&catalog, &other);
    let paths: Vec<_> = d.iter().map(|x| x.path.as_str()).collect();
    assert_eq!(paths, ["system_spec_id", "views[0].layers[1].source_node"]);
    assert!(matches!(
        d[1].kind,
        VizDiagnosticKind::UnknownNodeRef { .. }
    ));
}

#[test]
fn ac1_strict_parser_rejects_unknown_fields_and_bad_units_with_path() {
    let bad = GOLDEN.replace("\"visible\":true", "\"visible\":true,\"zzz\":1");
    match VisualizationSpec::from_json(&bad) {
        Err(VizSpecError::Field { path, .. }) => assert_eq!(path, "views[0].layers[0].zzz"),
        other => panic!("{other:?}"),
    }
    let bad = GOLDEN.replace("\"unit\":\"meters\"", "\"unit\":\"furlongs\"");
    match VisualizationSpec::from_json(&bad) {
        Err(VizSpecError::Field { path, .. }) => {
            assert_eq!(path, "views[0].layers[1].params.ray_length.unit");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn ac2_unregistered_projection() {
    let mut s = view_with(LayerBinding::new("environment"));
    s.views[0].projection = "hologram".to_owned();
    let (path, kind) = first_diag(&s);
    assert_eq!(path, "views[0].projection");
    assert_eq!(
        kind,
        VizDiagnosticKind::UnregisteredProjection {
            projection: "hologram".to_owned()
        }
    );
}

#[test]
fn ac2_unknown_layer() {
    let mut s = view_with(LayerBinding::new("environment"));
    s.views[0].layers.push(LayerBinding::new("aurora"));
    let (path, kind) = first_diag(&s);
    assert_eq!(path, "views[0].layers[1].layer");
    assert_eq!(
        kind,
        VizDiagnosticKind::UnknownLayer {
            layer: "aurora".to_owned()
        }
    );
}

#[test]
fn ac2_layer_incompatible_with_projection() {
    let mut s = view_with(LayerBinding::new("environment"));
    s.views[0].projection = "decision_overlay".to_owned();
    s.views[0].units = ScaleUnits::Meters;
    let (path, kind) = first_diag(&s);
    assert_eq!(path, "views[0].layers[0].layer");
    assert!(matches!(
        kind,
        VizDiagnosticKind::LayerNotInProjection { .. }
    ));
}

#[test]
fn ac2_incompatible_units_for_projection() {
    let mut s = view_with(LayerBinding::new("sensors"));
    s.views[0].projection = "decision_overlay".to_owned();
    s.views[0].units = ScaleUnits::Kilometers;
    let (path, kind) = first_diag(&s);
    assert_eq!(path, "views[0].units");
    assert!(matches!(kind, VizDiagnosticKind::UnsupportedUnits { .. }));
}

#[test]
fn ac2_parameter_descriptor_violations() {
    let num = |value, unit| ParamValue::Number { value, unit };
    let cases = [
        (
            LayerBinding::new("sensors").with_param("nope", ParamValue::Bool(true)),
            "views[0].layers[0].params.nope",
            "unknown_parameter",
        ),
        (
            LayerBinding::new("sensors")
                .with_param("ray_length", num(1.0, Some(ExprUnit::Seconds))),
            "views[0].layers[0].params.ray_length",
            "unit_mismatch",
        ),
        (
            LayerBinding::new("sensors").with_param("ray_length", num(1.0, None)),
            "views[0].layers[0].params.ray_length",
            "missing_unit",
        ),
        (
            LayerBinding::new("sensors")
                .with_param("ray_length", num(-1.0, Some(ExprUnit::Meters))),
            "views[0].layers[0].params.ray_length",
            "parameter_out_of_range",
        ),
        (
            LayerBinding::new("sensors").with_param("ray_length", ParamValue::Text("x".into())),
            "views[0].layers[0].params.ray_length",
            "parameter_type",
        ),
        (
            LayerBinding::new("imagination").with_param("horizon_ticks", ParamValue::Integer(0)),
            "views[0].layers[0].params.horizon_ticks",
            "parameter_out_of_range",
        ),
    ];
    for (binding, path, code) in cases {
        let (p, kind) = first_diag(&view_with(binding));
        assert_eq!((p.as_str(), kind.code()), (path, code));
    }
}

#[test]
fn ac2_unknown_inspector_field_and_deterministic_order() {
    let mut s = view_with(LayerBinding::new("aurora"));
    s.views[0].projection = "hologram".to_owned();
    s.views[0].layers.push(LayerBinding::new("zeta"));
    s.views
        .push(ViewSpec::new("v", "scene2d", ScaleUnits::Meters));
    let mut ins = gogamen_projection::InspectorSpec::default();
    ins.fields.push(gogamen_projection::InspectorFieldSpec {
        name: "mass".to_owned(),
        label: "Mass".to_owned(),
        min_lod: gogamen_projection::InspectorLod::Full,
        stale_after: None,
    });
    s.views[1].inspector = Some(ins);
    let catalog = VisualizationCatalog::builtin();
    let d = s.validate(&catalog);
    let got: Vec<_> = d.iter().map(|x| (x.path.as_str(), x.kind.code())).collect();
    assert_eq!(
        got,
        [
            ("views[0].layers[0].layer", "unknown_layer"),
            ("views[0].layers[1].layer", "unknown_layer"),
            ("views[0].projection", "unregistered_projection"),
            ("views[1].id", "duplicate_id"),
            (
                "views[1].inspector.fields[0].name",
                "unknown_inspector_field"
            ),
        ]
    );
    assert_eq!(d, s.validate(&catalog), "deterministic");
    assert!(
        d[0].to_string()
            .starts_with("views[0].layers[0].layer: [unknown_layer]")
    );
}
