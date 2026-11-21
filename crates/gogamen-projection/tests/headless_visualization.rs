//! GOG-102 AC #3: la sección opcional `visualization` es ignorada por el plan headless.
//!
//! Sin JVM ni dependencias gráficas: el artefacto compilado y su ejecución son idénticos con y sin sección.

use gogamen_core::MasterSeed;
use gogamen_projection::{SystemWithVisualization, canonical_visualization_fixture};
use gogamen_runtime::PlanRuntimeEngine;
use gogamen_spec::{SystemSpec, canonical_two_node_fixture};

const WITH_SECTION: &str = include_str!("../fixtures/system_with_visualization_v0.json");

#[test]
fn headless_artifact_ignores_optional_visualization_section() {
    let engine = PlanRuntimeEngine::new();
    let plain = canonical_two_node_fixture();

    let doc = SystemWithVisualization::from_json(WITH_SECTION).expect("doc parses");
    assert_eq!(doc.visualization, Some(canonical_visualization_fixture()));
    assert_eq!(doc.system, plain);
    assert_eq!(doc.to_canonical_json().unwrap(), WITH_SECTION);

    // Sin la sección el documento es el SystemSpec canónico intacto (hashes existentes no cambian).
    let without = SystemWithVisualization {
        system: plain.clone(),
        visualization: None,
    };
    assert_eq!(
        without.to_canonical_json().unwrap(),
        plain.to_canonical_json().unwrap()
    );

    let a_plain = engine.compile_artifact(&plain).unwrap();
    let a_with = engine.compile_artifact(&doc.system).unwrap();
    assert_eq!(a_plain.spec_hash(), a_with.spec_hash());
    assert_eq!(a_plain.artifact_hash(), a_with.artifact_hash());
    assert_eq!(a_plain.catalog_fingerprint(), a_with.catalog_fingerprint());
    assert_eq!(
        a_plain.to_canonical_json().unwrap(),
        a_with.to_canonical_json().unwrap()
    );

    // Ejecución headless: spec directo, spec del documento con sección y artefacto serializado.
    let direct = engine.run_spec(&plain, 25).unwrap();
    let via_doc = engine.run_spec(&doc.system, 25).unwrap();
    let via_artifact = engine
        .run_artifact_json(&a_with.to_canonical_json().unwrap(), 25)
        .unwrap();
    assert_eq!(direct, via_doc);
    assert_eq!(direct, via_artifact);
    let seeded = engine
        .run_spec_with_seed(&doc.system, 10, MasterSeed::new(7))
        .unwrap();
    let seeded_plain = engine
        .run_spec_with_seed(&plain, 10, MasterSeed::new(7))
        .unwrap();
    assert_eq!(seeded.trace_hash, seeded_plain.trace_hash);
}

#[test]
fn strict_system_spec_parser_still_rejects_the_section_and_envelope_validates_it() {
    // El parser estricto de SystemSpec no cambia: la sección sólo la entiende el envelope opcional.
    assert!(SystemSpec::from_json(WITH_SECTION).is_err());
    let doc = SystemWithVisualization::from_json(WITH_SECTION).unwrap();
    let catalog = gogamen_projection::VisualizationCatalog::builtin();
    let viz = doc.visualization.unwrap();
    assert!(
        viz.validate_against_system(&catalog, &doc.system)
            .is_empty()
    );
}

#[test]
fn invalid_section_reports_path_prefixed_with_section_key() {
    let bad = WITH_SECTION.replace("\"units\":\"meters\"", "\"units\":\"parsecs\"");
    match SystemWithVisualization::from_json(&bad) {
        Err(gogamen_projection::VizSpecError::Field { path, .. }) => {
            assert_eq!(path, "visualization.views[0].units");
        }
        other => panic!("{other:?}"),
    }
}
