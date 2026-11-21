//! GOG-158: `CognitiveSpec` tipado, opcional y compatible con v0.

use gogamen_spec::{
    AdaptiveAgentSpec, AdaptiveExperimentSpec, MAX_RESOLUTION_LEVELS, SpecError, SystemSpec,
    canonical_cognitive_grid_experiment_fixture_json,
    canonical_grid_cognitive_spec_experiment_fixture,
    canonical_kart_cognitive_spec_experiment_fixture, canonical_language_world_fixture_json,
    canonical_two_node_fixture_json,
};

fn grid_json() -> String {
    canonical_grid_cognitive_spec_experiment_fixture()
        .to_canonical_json()
        .unwrap()
}

#[test]
fn gog_158_grid_and_kart_cognitive_round_trip() {
    for exp in [
        canonical_grid_cognitive_spec_experiment_fixture(),
        canonical_kart_cognitive_spec_experiment_fixture(),
    ] {
        let json = exp.to_canonical_json().unwrap();
        assert!(json.contains("\"cognitive\":{"));
        let parsed = AdaptiveExperimentSpec::from_json(&json).unwrap();
        assert_eq!(parsed, exp);
        assert_eq!(parsed.to_canonical_json().unwrap(), json);
        assert!(parsed.agents[0].cognitive.is_some());
    }
    let agent = canonical_kart_cognitive_spec_experiment_fixture()
        .agents
        .remove(0);
    let json = agent.to_canonical_json().unwrap();
    assert_eq!(AdaptiveAgentSpec::from_json(&json).unwrap(), agent);
}

#[test]
fn gog_158_v0_fixtures_stay_valid_and_cognitive_is_omitted() {
    SystemSpec::from_json(&canonical_two_node_fixture_json()).unwrap();
    SystemSpec::from_json(&canonical_language_world_fixture_json()).unwrap();
    let plain = canonical_cognitive_grid_experiment_fixture_json();
    assert!(!plain.contains("cognitive\""));
    let parsed = AdaptiveExperimentSpec::from_json(&plain).unwrap();
    assert!(parsed.cognitive.is_none());
    assert_eq!(parsed.to_canonical_json().unwrap(), plain);
}

#[test]
fn gog_158_invalid_reference_rejected_with_path() {
    let json = grid_json().replace("\"sensor\":\"local_fov_sensor\"", "\"sensor\":\"ghost\"");
    match AdaptiveExperimentSpec::from_json(&json).unwrap_err() {
        SpecError::InvalidReference { path, kind, id } => {
            assert_eq!(kind, "sensor");
            assert_eq!(id, "ghost");
            assert_eq!(path, "cognitive.resolution.levels[0].sensor");
        }
        other => panic!("unexpected {other:?}"),
    }
    let json = grid_json().replace(
        "\"model\":\"pomdp_belief_v0\",\"rule\"",
        "\"model\":\"nope\",\"rule\"",
    );
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::InvalidReference { kind: "model", .. }
    ));
    let json = grid_json().replace("\"port\":\"memory\"", "\"port\":\"missing\"");
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::InvalidReference { kind: "port", .. }
    ));
}

#[test]
fn gog_158_invalid_type_rejected() {
    // Tipo JSON incorrecto.
    let json = grid_json().replace("\"horizon\":16", "\"horizon\":\"16\"");
    match AdaptiveExperimentSpec::from_json(&json).unwrap_err() {
        SpecError::InvalidFieldValue { field, reason, .. } => {
            assert_eq!(field, "horizon");
            assert!(reason.contains("cognitive.entropic.horizon"));
        }
        other => panic!("unexpected {other:?}"),
    }
    // Puerto de memoria con tipo de puerto incorrecto (nodo/puerto existente pero no MemoryPort).
    let json = grid_json().replace(
        "\"port\":{\"node\":\"memory_store\",\"port\":\"memory\"}",
        "\"port\":{\"node\":\"belief_updater\",\"port\":\"belief_state\"}",
    );
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::InvalidFieldValue { field: "port", .. }
    ));
    // Variante de objetivo desconocida.
    let json = grid_json().replace("\"objective\":\"fmc\"", "\"objective\":\"magic\"");
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::UnknownVariant { ref path, .. } if path == "cognitive.entropic.objective"
    ));
}

#[test]
fn gog_158_non_finite_rejected() {
    let mut exp = canonical_grid_cognitive_spec_experiment_fixture();
    for bad in [f64::NAN, f64::INFINITY] {
        exp.cognitive.as_mut().unwrap().entropic.temperature = bad;
        match exp.validate().unwrap_err() {
            SpecError::NonFiniteFloat { context } => {
                assert_eq!(context, "cognitive.entropic.temperature");
            }
            other => panic!("unexpected {other:?}"),
        }
        assert!(exp.to_canonical_json().is_err());
    }
    // Desborde textual a infinito: nunca produce un spec válido.
    let json = grid_json().replace("\"temperature\":0.5", "\"temperature\":1e999");
    assert!(AdaptiveExperimentSpec::from_json(&json).is_err());
    // Fuera de rango.
    let json = grid_json().replace("\"learning_rate\":0.1", "\"learning_rate\":2.0");
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::InvalidFieldValue {
            field: "learning_rate",
            ..
        }
    ));
}

#[test]
fn gog_158_unknown_permission_rejected() {
    let json = grid_json().replace("\"append\"", "\"root\"");
    match AdaptiveExperimentSpec::from_json(&json).unwrap_err() {
        SpecError::UnknownVariant {
            path,
            value,
            expected,
        } => {
            assert_eq!(path, "cognitive.memory.permissions[1]");
            assert_eq!(value, "root");
            assert!(expected.contains(&"read"));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn gog_158_bounds_exceeded_rejected() {
    // Demasiados niveles (se rechaza antes de parsear cada nivel).
    let level = "{\"bins\":4,\"level_id\":\"x\",\"sensor\":\"local_fov_sensor\"}";
    let many = vec![level; MAX_RESOLUTION_LEVELS + 1].join(",");
    let json = grid_json();
    let start = json.find("\"levels\":[").unwrap() + "\"levels\":[".len();
    let end = start + json[start..].find("]}").unwrap();
    let mutated = format!("{}{}{}", &json[..start], many, &json[end..]);
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&mutated).unwrap_err(),
        SpecError::LimitExceeded { limit, found, .. }
            if limit == MAX_RESOLUTION_LEVELS && found == MAX_RESOLUTION_LEVELS + 1
    ));
    // Capacidad y bins fuera de límite.
    let json = grid_json().replace("\"capacity\":256", "\"capacity\":4000000000");
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::InvalidFieldValue {
            field: "capacity",
            ..
        }
    ));
    // Identificador demasiado largo.
    let long = "a".repeat(129);
    let json = grid_json().replace(
        "\"binning_version\":\"v0\"",
        &format!("\"binning_version\":\"{long}\""),
    );
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::LimitExceeded { limit: 128, .. }
    ));
}

#[test]
fn gog_158_unknown_cognitive_field_rejected_by_default() {
    let json = grid_json().replacen("\"entropic\":{", "\"entropic\":{\"callback\":\"x\",", 1);
    assert!(matches!(
        AdaptiveExperimentSpec::from_json(&json).unwrap_err(),
        SpecError::UnknownField { .. }
    ));
}
