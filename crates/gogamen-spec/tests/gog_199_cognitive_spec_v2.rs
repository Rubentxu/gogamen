//! GOG-199 (DEM-005 v2): contrato tipado v0.2, libro de dotación y migración.
//!
//! Cubre los tres criterios de aceptación con errores direccionados por ruta:
//! (a) un principio `Pure` que lee el observable de un fenómeno, (b) `modelAccess: hidden`
//! junto a un modelo privilegiado del laboratorio y (c) el libro de dotación canónico con
//! hash estable y errores precisos ante valores no finitos o permisos desconocidos.

use std::fs;
use std::path::PathBuf;

use gogamen_core::SelectionPrinciple;
use gogamen_spec::{
    ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION, ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
    ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION, ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2,
    ActionRegistry, AdaptiveAgentSpec, AdaptiveExperimentSpec, ENDOWMENT_BOOK_SCHEMA_VERSION,
    EndowmentBook, ModelAccess, PhenomenonPredicate, PrincipleSpec, SpecError, SystemSpec,
    canonical_grid_endowment_experiment_fixture,
    canonical_grid_endowment_experiment_fixture_json, canonical_grid_experiment_fixture,
    canonical_grid_experiment_fixture_json, canonical_kart_endowment_experiment_fixture,
    canonical_two_node_fixture_json,
    migrate_experiment_spec_json,
};

fn corpus(rel: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/golden/cognitive_dsl")
        .join(rel);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn registry() -> ActionRegistry {
    ActionRegistry::canonical_grid_defaults()
}

#[test]
fn gog_199_cognitive_spec_v2_grid_and_kart_contracts_are_versioned() {
    for (experiment, expected_id) in [
        (
            canonical_grid_endowment_experiment_fixture(),
            "grid_endowment_v0_2",
        ),
        (
            canonical_kart_endowment_experiment_fixture(),
            "kart_endowment_v0_2",
        ),
    ] {
        assert_eq!(experiment.experiment_id, expected_id);
        assert_eq!(
            experiment.schema_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(
            experiment.agents[0].schema_version,
            ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2
        );
        experiment.validate().expect("v0.2 fixture must validate");
        if experiment.experiment_id == "grid_endowment_v0_2" {
            experiment
                .validate_with_registry(&registry())
                .expect("grid actions must be registered");
        }

        let json = experiment.to_canonical_json().unwrap();
        assert!(json.contains("\"schema_version\":\"gogamen.experiment_spec.v0.2\""));
        assert!(json.contains("\"principle\":{"));
        assert!(json.contains("\"scale\":{"));
        assert!(json.contains("\"phenomena\":["));
        assert!(json.contains("\"vary\":["));

        let parsed = AdaptiveExperimentSpec::from_json(&json).unwrap();
        assert_eq!(parsed, experiment);
        assert_eq!(parsed.to_canonical_json().unwrap(), json);

        let book = experiment.endowment_book().unwrap();
        assert_eq!(book.schema_version, ENDOWMENT_BOOK_SCHEMA_VERSION);
        assert_eq!(book.experiment_id, expected_id);
        assert_eq!(book.agents.len(), 1);
        assert!(!book.phenomena.is_empty());
        assert!(!book.hypotheses.is_empty());
    }
}

#[test]
fn gog_199_cognitive_spec_v2_migration_reports_and_preserves_v0_fixtures() {
    // Los fixtures v0/v0.1 conservan sus bytes canónicos y no se reescriben solos.
    SystemSpec::from_json(&canonical_two_node_fixture_json()).unwrap();
    let v01 = canonical_grid_experiment_fixture_json();
    let parsed_v01 = AdaptiveExperimentSpec::from_json(&v01).unwrap();
    assert_eq!(
        parsed_v01.schema_version,
        ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION
    );
    assert_eq!(parsed_v01.to_canonical_json().unwrap(), v01);

    // El corpus dorado del DSL sigue siendo byte-idéntico en v0.1.
    for i in 0..=7 {
        let json = corpus(&format!("positive/h{i}.json"));
        let spec = AdaptiveExperimentSpec::from_json(&json).unwrap_or_else(|e| {
            panic!("h{i} rejected: {e}");
        });
        assert_eq!(
            spec.schema_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
            "h{i} must keep its historical v0.1 bytes"
        );
        assert_eq!(spec.to_canonical_json().unwrap(), json, "h{i} not canonical");
    }

    // Migrar h0 produce un reporte estructurado con origen v0.1 y destino v0.2.
    let migrated = migrate_experiment_spec_json(&corpus("positive/h0.json")).unwrap();
    assert_eq!(
        migrated.report.source_version,
        ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION
    );
    assert_eq!(
        migrated.report.target_version,
        ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
    );
    assert_eq!(
        migrated.report.applied_steps,
        vec!["upgrade_schema_version:v0.1->v0.2"]
    );
    assert_eq!(
        migrated.spec.schema_version,
        ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
    );
    migrated.spec.validate_with_registry(&registry()).unwrap();

    // h6 declara `phenomena_json`; h7 añade `mode = pure` y `readings_json`.
    let h6 = migrate_experiment_spec_json(&corpus("positive/h6.json")).unwrap();
    assert!(
        h6.spec.agents[0].phenomena.len() >= 4,
        "h6 phenomena must be lifted"
    );
    assert!(!h6.spec.config.contains_key("phenomena_json"));
    assert!(h6.spec.agents[0].phenomena.iter().all(|p| !p
        .observable
        .is_empty()));
    assert!(matches!(
        h6.spec.agents[0].phenomena[0].predicate,
        PhenomenonPredicate::After { .. }
    ));
    assert!(h6.report.applied_steps.iter().any(|s| s == "lift_phenomena"));

    let h7 = migrate_experiment_spec_json(&corpus("positive/h7.json")).unwrap();
    assert_eq!(
        h7.spec.agents[0].principle.as_ref().map(|p| p.kind),
        Some(SelectionPrinciple::Pure)
    );
    assert!(!h7.spec.agents[0].config.contains_key("mode"));
    assert!(h7.report.applied_steps.iter().any(|s| s == "lift_agent_principle"));
    // `readings_json` no es parte del contrato tipado: se conserva como config legado.
    assert!(h7.spec.config.contains_key("readings_json"));

    for migrated in [&h6, &h7] {
        let json = migrated.spec.to_canonical_json().unwrap();
        assert_eq!(AdaptiveExperimentSpec::from_json(&json).unwrap(), migrated.spec);
    }

    // Un artefacto v0.2 es passthrough sin pasos.
    let v02 = canonical_grid_endowment_experiment_fixture_json();
    let passthrough = migrate_experiment_spec_json(&v02).unwrap();
    assert_eq!(
        passthrough.report.applied_steps,
        vec!["already_at_target_version"]
    );
    assert_eq!(passthrough.spec.to_canonical_json().unwrap(), v02);
}

#[test]
fn gog_199_cognitive_spec_v2_rejects_pure_principle_reading_phenomenon_observable() {
    let mut experiment = canonical_grid_endowment_experiment_fixture();
    experiment.agents[0].principle = Some(PrincipleSpec {
        kind: SelectionPrinciple::Pure,
        observables: vec!["reward".to_owned()],
    });
    let err = experiment.validate().unwrap_err();
    match &err {
        SpecError::PurePrincipleReadsPhenomenonObservable { path, observable } => {
            assert_eq!(path, "agent.phenomena[0].observable");
            assert_eq!(observable, "reward");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(
        err.to_string().contains("agent.phenomena[0].observable"),
        "message must carry the path: {err}"
    );

    // El mismo principio puro sin declarar ese observable sigue siendo válido.
    let mut allowed = canonical_grid_endowment_experiment_fixture();
    allowed.agents[0].principle = Some(PrincipleSpec {
        kind: SelectionPrinciple::Pure,
        observables: vec!["progress".to_owned()],
    });
    allowed.validate().unwrap();
}

#[test]
fn gog_199_cognitive_spec_v2_rejects_privileged_model_with_hidden_access() {
    let mut experiment = canonical_grid_endowment_experiment_fixture();
    // Instantánea canónica del agente antes de marcarlo privilegiado/hidden.
    let agent_json = experiment.agents[0].to_canonical_json().unwrap();
    experiment.agents[0].model_access = ModelAccess::Hidden;
    assert!(
        experiment.agents[0]
            .model
            .config
            .get("privileged")
            .is_none()
    );
    // El fixture no marca el modelo como privilegiado: hidden solo es válido entonces.
    experiment.validate().unwrap();

    experiment.agents[0].model = experiment.agents[0].model.clone().with_privileged(true);
    let err = experiment.validate().unwrap_err();
    match &err {
        SpecError::PrivilegedModelWithHiddenAccess { path, model_id } => {
            assert_eq!(path, "agent.model_access");
            assert_eq!(model_id, "pomdp_belief_v0");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(err.to_string().contains("agent.model_access"));

    // Un `model_access` desconocido también falla con ruta precisa al parsear.
    // (El escritor omite `model_access` cuando es `visible`, así que se inyecta en el agente.)
    let json = agent_json.replacen(
        '{',
        "{\"model_access\":\"omniscient\",",
        1,
    );
    assert!(json.contains("omniscient"));
    let err = AdaptiveAgentSpec::from_json(&json).unwrap_err();
    match &err {
        SpecError::UnknownVariant { path, value, expected } => {
            assert_eq!(path, "agent.model_access");
            assert_eq!(value, "omniscient");
            assert_eq!(*expected, &["visible", "hidden"]);
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Los campos tipados exigen v0.2: declararlos en v0.1 es un error con ruta.
    let mut v01_agent = canonical_grid_experiment_fixture().agents.remove(0);
    v01_agent.principle = Some(PrincipleSpec {
        kind: SelectionPrinciple::Directed,
        observables: Vec::new(),
    });
    let err = v01_agent.validate().unwrap_err();
    match &err {
        SpecError::TypedFieldRequiresV2 { path, expected } => {
            assert_eq!(path, "agent.principle");
            assert_eq!(*expected, ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2);
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert_eq!(v01_agent.schema_version, ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION);
}

#[test]
fn gog_199_cognitive_spec_v2_endowment_book_is_canonical_and_hash_stable() {
    let grid = canonical_grid_endowment_experiment_fixture();
    let book = grid.endowment_book().unwrap();
    book.validate().unwrap();

    let json = book.to_canonical_json().unwrap();
    assert!(json.starts_with("{\"agents\":["));
    assert!(json.contains("\"schema_version\":\"gogamen.endowment_book.v1\""));
    // Determinismo: dos emisiones del mismo experimento son byte-idénticas.
    assert_eq!(book.to_canonical_json().unwrap(), json);
    let hash = book.hash().unwrap();
    assert_eq!(book.hash().unwrap(), hash);
    assert_eq!(EndowmentBook::from_experiment(&grid).unwrap(), book);
    assert!(json.contains("\"world\""));

    let kart = canonical_kart_endowment_experiment_fixture()
        .endowment_book()
        .unwrap();
    let kart_json = kart.to_canonical_json().unwrap();
    assert_ne!(kart_json, json);
    assert_ne!(kart.hash().unwrap(), hash);

    // Cambiar una hipótesis cambia el hash: el libro fija la dotación declarada.
    let mut edited = grid.clone();
    let mut hypothesis = edited.hypotheses[0].clone();
    hypothesis.threshold += 0.01;
    edited.hypotheses[0] = hypothesis;
    let edited_book = edited.endowment_book().unwrap();
    assert_ne!(edited_book.hash().unwrap(), hash);
}

#[test]
fn gog_199_cognitive_spec_v2_endowment_reports_non_finite_and_unknown_permission() {
    // Valor no finito en un eje de escala: ruta `agent.scale.axes[0].min`.
    let mut scale_experiment = canonical_grid_endowment_experiment_fixture();
    let scale = scale_experiment.agents[0].scale.as_mut().unwrap();
    scale.axes[0].min = f64::NAN;
    let err = scale_experiment.validate().unwrap_err();
    match &err {
        SpecError::NonFiniteFloat { context } => {
            assert_eq!(context, "agent.scale.axes[0].min");
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(err.to_string().contains("agent.scale.axes[0].min"));

    // Valor no finito en un eje de ablación: ruta `vary[0].values[1]`.
    let mut vary_experiment = canonical_grid_endowment_experiment_fixture();
    vary_experiment.vary[0].values[1] = f64::INFINITY;
    let err = vary_experiment.validate().unwrap_err();
    match &err {
        SpecError::NonFiniteFloat { context } => {
            assert_eq!(context, "vary[0].values[1]");
        }
        other => panic!("unexpected error: {other:?}"),
    }

    // Un permiso de memoria desconocido falla con la ruta del elemento concreto.
    let agent = canonical_grid_endowment_experiment_fixture().agents.remove(0);
    let agent_json = agent.to_canonical_json().unwrap();
    assert!(agent_json.contains("\"permissions\":[\"read\",\"append\"]"));
    let bad = agent_json.replace(
        "\"permissions\":[\"read\",\"append\"]",
        "\"permissions\":[\"read\",\"admin\"]",
    );
    let err = AdaptiveAgentSpec::from_json(&bad).unwrap_err();
    match &err {
        SpecError::UnknownVariant { path, value, expected } => {
            assert_eq!(path, "cognitive.memory.permissions[1]");
            assert_eq!(value, "admin");
            assert!(expected.contains(&"read"));
            assert!(expected.contains(&"append"));
        }
        other => panic!("unexpected error: {other:?}"),
    }
    assert!(err.to_string().contains("cognitive.memory.permissions[1]"));

    // El mismo agente sin tocar conserva los permisos y el libro sigue emitiéndose.
    let endowment = agent_endowment_book(&agent);
    assert!(endowment.hash().unwrap() != 0);
}

fn agent_endowment_book(agent: &AdaptiveAgentSpec) -> EndowmentBook {
    let mut experiment = canonical_grid_endowment_experiment_fixture();
    experiment.agents = vec![agent.clone()];
    experiment.endowment_book().unwrap()
}
