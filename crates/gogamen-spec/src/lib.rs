//! Especificación declarativa portable `SystemSpec` v0 para Gogamen (`std`-only).
//!
//! Este crate define el formato declarativo JSON versionado ([`SYSTEM_SPEC_SCHEMA_VERSION`])
//! para describir sistemas de nodos y conexiones causales tipadas sin DSL ni código arbitrario,
//! junto con [`AgentSpec`] y [`ExperimentSpec`] compuestos sobre [`SystemSpec`].
//!
//! El bloque opcional [`CognitiveSpec`] (política entrópica, resolución, aprendizaje, memoria)
//! se adjunta de forma aditiva a [`AdaptiveAgentSpec`] y [`AdaptiveExperimentSpec`]; ver [`cognitive`].
//!
//! # Aislamiento de dependencias
//! `gogamen-spec` depende únicamente de `std` y `gogamen-core` (sin `serde` ni crates externos).

pub mod adaptive;
pub mod codec;
pub mod cognitive;
pub mod endowment;
pub mod epistemic;
pub mod error;
pub mod expr;
pub mod hypothesis;
pub mod json;
pub mod model;

pub use adaptive::{
    ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION, ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
    ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION, ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2,
    ActionRegistry, AdaptiveAgentSpec, AdaptiveExperimentSpec, BudgetSpec, CapabilityCatalog,
    ConstraintSpec, GoalSpec, MigratedExperimentSpec, ModelSpecRef, SchemaMigrationReport,
    SeedSpec, SensorBindingSpec, WorldSpecRef, canonical_cognitive_grid_experiment_fixture,
    canonical_cognitive_grid_experiment_fixture_json,
    canonical_grid_endowment_experiment_fixture,
    canonical_grid_endowment_experiment_fixture_json, canonical_grid_experiment_fixture,
    canonical_grid_experiment_fixture_json,
    canonical_kart_endowment_experiment_fixture,
    canonical_kart_endowment_experiment_fixture_json, migrate_experiment_spec_json,
};
pub use codec::{
    canonical_cognitive_agent_fixture, canonical_cognitive_agent_fixture_json,
    canonical_language_world_fixture, canonical_language_world_fixture_json,
    canonical_two_node_fixture, canonical_two_node_fixture_json,
};
pub use cognitive::{
    COGNITIVE_SPEC_SCHEMA_VERSION, CognitiveRefs, CognitiveSpec, EntropicObjective,
    EntropicPolicySpec, LearningRule, LearningSpec, MAX_COGNITIVE_ID_BYTES, MAX_ENTROPIC_HORIZON,
    MAX_MEMORY_CAPACITY, MAX_RESOLUTION_BINS, MAX_RESOLUTION_LEVELS, MEMORY_PORT_TYPE_NAME,
    MemoryKind, MemoryPermission, MemoryPortRef, MemorySpec, ResolutionLevelSpec, ResolutionSpec,
    canonical_grid_cognitive_spec_experiment_fixture,
    canonical_kart_cognitive_spec_experiment_fixture,
};
pub use endowment::{
    AgentEndowment, ENDOWMENT_BOOK_SCHEMA_VERSION, EndowmentBook, EndowmentPhenomenon,
    EndowmentWorld, ModelAccess, PhenomenonPredicate, PhenomenonSpec, PrincipleSpec, ScaleAxisSpec,
    ScaleSpec, VaryAxisSpec,
};
pub use epistemic::{
    AVAILABLE_EPISTEMIC_MODELS, AbstentionPolicySpec, AskActionSpec, EpistemicQuestionSpec,
    EpistemicSpec, GuardPolicySpec, InspectActionSpec, ProvenanceSpec,
};
pub use error::{SpecDiagnostic, SpecError, SpecMigrationError, UnknownFieldPolicy};
pub use expr::{EXPR_AST_MAX_DEPTH, ExprAst, ExprType, ExprUnit};
pub use hypothesis::{
    HypothesisComparator, HypothesisSpec, canonical_hypotheses_json, hypotheses_hash,
    validate_hypotheses,
};
pub use json::{
    DEFAULT_JSON_MAX_BYTES, DEFAULT_JSON_MAX_ELEMENTS, DEFAULT_JSON_MAX_STRING_BYTES, JsonLimits,
    JsonNumber, JsonValue, parse_json, parse_with_limits, write_canonical_f64, write_json_string,
};
pub use model::{
    AGENT_SPEC_SCHEMA_VERSION, AgentSpec, EXPERIMENT_SPEC_SCHEMA_VERSION, ExperimentSpec,
    SYSTEM_SPEC_SCHEMA_VERSION, SpecConfigMap, SpecConfigValue, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec,
};

#[cfg(test)]
mod tests {
    use gogamen_core::{CausalKind, ConnectionError, PortDirection, PortType};

    use super::*;

    #[test]
    fn two_nodes_and_one_connection_round_trip_without_semantic_loss() {
        let original = canonical_two_node_fixture();
        let json = original
            .to_canonical_json()
            .expect("canonical fixture must serialize");
        let parsed = SystemSpec::from_json(&json).expect("canonical fixture must parse");

        assert_eq!(parsed, original);
        assert_eq!(parsed.to_canonical_json().unwrap(), json);

        let graph = parsed
            .to_connection_graph()
            .expect("parsed spec must compile to ConnectionGraph");
        let plan = graph.compile().expect("graph must compile");
        assert_eq!(plan.node_count(), 2);
        assert_eq!(plan.execution_order().len(), 2);
    }

    #[test]
    fn round_trip_preserves_delayed_connection_custom_port_and_all_scalar_types() {
        let node_a = SystemNodeSpec::new("env", "world")
            .with_output("obs", PortType::custom("ObservationVec"))
            .with_input("act", PortType::I64)
            .with_config("flag", SpecConfigValue::Bool(false))
            .with_config("steps", SpecConfigValue::I64(-42))
            .with_config("whole_float", SpecConfigValue::f64(1.0).unwrap())
            .with_config("label", SpecConfigValue::String("cartpole\nv0".to_owned()));

        let node_b = SystemNodeSpec::new("agent", "policy")
            .with_input("obs", PortType::custom("ObservationVec"))
            .with_output("act", PortType::I64);

        let spec = SystemSpec::new("feedback_loop")
            .with_node(node_a)
            .with_node(node_b)
            .with_connection(SystemConnectionSpec::instant("env", "obs", "agent", "obs"))
            .with_connection(SystemConnectionSpec::delayed(
                "agent", "act", "env", "act", 1,
            ));

        let json = spec.to_canonical_json().unwrap();
        let parsed = SystemSpec::from_json(&json).unwrap();
        assert_eq!(parsed, spec);
        assert_eq!(parsed.connections[1].causal, CausalKind::Delayed(1));
        assert_eq!(
            parsed.nodes[0].config.get("whole_float"),
            Some(&SpecConfigValue::F64(1.0))
        );
        assert!(parsed.to_connection_graph().is_ok());
    }

    #[test]
    fn agent_and_experiment_specs_compose_system_spec_and_round_trip() {
        let base_system = canonical_two_node_fixture();
        let agent = AgentSpec::new("agent_alpha", base_system.clone())
            .with_config("exploration", SpecConfigValue::f64(0.1).unwrap());
        let experiment = ExperimentSpec::new("exp_001", 0xDEAD_BEEF_CAFE_BABE, base_system)
            .with_agent(agent)
            .with_config("episodes", SpecConfigValue::I64(10));

        let json = experiment.to_canonical_json().unwrap();
        let parsed = ExperimentSpec::from_json(&json).unwrap();
        assert_eq!(parsed, experiment);
        assert_eq!(parsed.seed, 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn rejects_incompatible_schema_version() {
        let json = canonical_two_node_fixture_json()
            .replace(SYSTEM_SPEC_SCHEMA_VERSION, "gogamen.system_spec.v99");
        let err = SystemSpec::from_json(&json).unwrap_err();
        assert!(matches!(
            err,
            SpecError::IncompatibleSchemaVersion {
                expected: SYSTEM_SPEC_SCHEMA_VERSION,
                ..
            }
        ));
    }

    #[test]
    fn rejects_invalid_ids_and_port_types() {
        let bad_id = canonical_two_node_fixture_json().replace(
            "\"spec_id\":\"two_node_canonical_v0\"",
            "\"spec_id\":\"   \"",
        );
        assert!(matches!(
            SystemSpec::from_json(&bad_id),
            Err(SpecError::EmptyIdentifier {
                context: "SystemSpec.spec_id"
            })
        ));

        let bad_type = canonical_two_node_fixture_json()
            .replace("\"port_type\":\"f64\"", "\"port_type\":\"tensor_magic\"");
        assert!(matches!(
            SystemSpec::from_json(&bad_type),
            Err(SpecError::InvalidFieldValue {
                field: "port_type",
                ..
            })
        ));

        let mismatched_type = canonical_two_node_fixture_json().replacen(
            "\"id\":\"observation\",\"port_type\":\"f64\"",
            "\"id\":\"observation\",\"port_type\":\"bool\"",
            1,
        );
        assert!(matches!(
            SystemSpec::from_json(&mismatched_type),
            Err(SpecError::Graph(ConnectionError::TypeMismatch(_)))
        ));

        let bad_direction = canonical_two_node_fixture_json().replace(
            "\"direction\":\"output\",\"id\":\"observation\"",
            "\"direction\":\"bidirectional\",\"id\":\"observation\"",
        );
        assert!(matches!(
            SystemSpec::from_json(&bad_direction),
            Err(SpecError::InvalidFieldValue {
                field: "direction",
                ..
            })
        ));
    }

    #[test]
    fn rejects_or_allows_unknown_fields_by_explicit_policy() {
        let with_unknown = canonical_two_node_fixture_json().replacen(
            "{",
            "{\"extra_frontend_hint\":\"kotlin_dsl\",",
            1,
        );

        let rejected = SystemSpec::from_json(&with_unknown).unwrap_err();
        assert!(matches!(
            rejected,
            SpecError::UnknownField {
                context: "SystemSpec",
                ref field
            } if field == "extra_frontend_hint"
        ));

        let rejected_explicit =
            SystemSpec::from_json_with_policy(&with_unknown, UnknownFieldPolicy::Reject)
                .unwrap_err();
        assert_eq!(rejected, rejected_explicit);

        let allowed =
            SystemSpec::from_json_with_policy(&with_unknown, UnknownFieldPolicy::AllowIgnored)
                .expect("AllowIgnored policy must accept unknown fields");
        assert_eq!(allowed, canonical_two_node_fixture());
    }

    #[test]
    fn rejects_duplicate_json_keys_and_non_scalar_config() {
        let dup_key = "{\"schema_version\":\"gogamen.system_spec.v0\",\"spec_id\":\"a\",\"spec_id\":\"b\",\"nodes\":[],\"connections\":[]}";
        assert!(matches!(
            SystemSpec::from_json(dup_key),
            Err(SpecError::DuplicateJsonKey { ref key }) if key == "spec_id"
        ));

        let nested_cfg = "{\"schema_version\":\"gogamen.system_spec.v0\",\"spec_id\":\"a\",\"nodes\":[],\"connections\":[],\"config\":{\"dsl\":[1,2]}}";
        assert!(matches!(
            SystemSpec::from_json(nested_cfg),
            Err(SpecError::InvalidFieldValue {
                field: "config",
                ..
            })
        ));
    }

    #[test]
    fn canonical_fixture_is_deterministic_and_valid() {
        let json_a = canonical_two_node_fixture_json();
        let json_b = canonical_two_node_fixture_json();
        assert_eq!(json_a, json_b);
        assert!(json_a.contains(SYSTEM_SPEC_SCHEMA_VERSION));

        let spec = SystemSpec::from_json(&json_a).unwrap();
        assert_eq!(spec.nodes.len(), 2);
        assert_eq!(spec.connections.len(), 1);
        assert_eq!(spec.nodes[0].ports[0].direction, PortDirection::Output);
        assert_eq!(spec.nodes[1].ports[0].direction, PortDirection::Input);
    }

    #[test]
    fn expr_ast_round_trips_canonical_json_and_rejects_non_finite_or_empty_var() {
        let ast = ExprAst::gt(
            ExprAst::mul(
                ExprAst::var("energia").unwrap(),
                ExprAst::var("curiosidad").unwrap(),
            ),
            ExprAst::var("umbral").unwrap(),
        );
        let json = ast.to_canonical_json().unwrap();
        let parsed = ExprAst::from_json(&json).unwrap();
        assert_eq!(parsed, ast);
        assert_eq!(parsed.to_canonical_json().unwrap(), json);

        assert!(matches!(
            ExprAst::const_f64(f64::NAN),
            Err(SpecError::NonFiniteFloat { .. })
        ));
        assert!(matches!(
            ExprAst::const_f64(f64::INFINITY),
            Err(SpecError::NonFiniteFloat { .. })
        ));
        assert!(matches!(
            ExprAst::var("   "),
            Err(SpecError::EmptyIdentifier { .. })
        ));

        let bad_unit_json = "{\"op\":\"const_f64\",\"unit\":\"magic\",\"value\":1.0}";
        assert!(matches!(
            ExprAst::from_json(bad_unit_json),
            Err(SpecError::InvalidFieldValue { field: "unit", .. })
        ));
    }

    #[test]
    fn adaptive_grid_and_cognitive_fixtures_round_trip_and_validate_without_ui() {
        let registry = ActionRegistry::canonical_grid_defaults();

        let grid = canonical_grid_experiment_fixture();
        grid.validate_with_registry(&registry)
            .expect("grid fixture must validate against canonical registry");
        let grid_json = canonical_grid_experiment_fixture_json();
        let grid_parsed = AdaptiveExperimentSpec::from_json(&grid_json)
            .expect("grid fixture JSON must round-trip");
        assert_eq!(grid_parsed, grid);
        assert_eq!(grid_parsed.to_canonical_json().unwrap(), grid_json);
        assert!(grid_parsed.seeds.is_disjoint());
        assert_eq!(grid_parsed.budget, BudgetSpec::new(500, 1000, 5000));
        assert!(grid_parsed.system.to_connection_graph().is_ok());

        let cognitive = canonical_cognitive_grid_experiment_fixture();
        cognitive
            .validate_with_registry(&registry)
            .expect("cognitive fixture must validate against canonical registry");
        let cog_json = canonical_cognitive_grid_experiment_fixture_json();
        let cog_parsed = AdaptiveExperimentSpec::from_json(&cog_json)
            .expect("cognitive fixture JSON must round-trip");
        assert_eq!(cog_parsed, cognitive);
        assert_eq!(cog_parsed.to_canonical_json().unwrap(), cog_json);
        assert!(cog_parsed.seeds.is_disjoint());
        assert_eq!(cog_parsed.agents[0].model.kind, "belief_state");
        assert_eq!(cog_parsed.agents[0].sensors[0].kind, "partial_fov");
        assert!(cog_parsed.system.to_connection_graph().is_ok());
    }

    #[test]
    fn adaptive_spec_rejects_zero_budget_and_overlapping_seeds() {
        let mut grid = canonical_grid_experiment_fixture();
        grid.budget.max_ticks = 0;
        assert!(matches!(
            grid.validate(),
            Err(SpecError::InvalidFieldValue {
                context: "BudgetSpec",
                field: "max_ticks",
                ..
            })
        ));

        let mut grid = canonical_grid_experiment_fixture();
        grid.seeds = SeedSpec::new(vec![10, 20], vec![20, 30]);
        assert!(!grid.seeds.is_disjoint());
        assert!(matches!(
            grid.validate(),
            Err(SpecError::OverlappingSeeds { seed: 20 })
        ));
    }

    #[test]
    fn adaptive_spec_rejects_unregistered_actions_and_code_eval() {
        let mut grid = canonical_grid_experiment_fixture();
        grid.agents[0].actions.push("teleport_anywhere".to_owned());

        let registry = ActionRegistry::canonical_grid_defaults();
        let err = grid.validate_with_registry(&registry).unwrap_err();
        assert!(matches!(
            err,
            SpecError::UnregisteredAction {
                context: "AdaptiveAgentSpec.actions",
                ref action
            } if action == "teleport_anywhere"
        ));

        let eval_expr_json = "{\"op\":\"eval\",\"code\":\"os.system('rm -rf /')\"}";
        assert!(matches!(
            ExprAst::from_json(eval_expr_json),
            Err(SpecError::CodeEvalRejected {
                context: "ExprAst",
                ref field,
                ..
            }) if field == "op"
        ));

        let mut eval_config_spec = canonical_grid_experiment_fixture();
        eval_config_spec.world.config.insert(
            "script".to_owned(),
            SpecConfigValue::String("return 42".to_owned()),
        );
        assert!(matches!(
            eval_config_spec.validate(),
            Err(SpecError::CodeEvalRejected {
                context: "WorldSpecRef.config",
                ref field,
                ..
            }) if field == "script"
        ));

        let mut eval_marker_spec = canonical_grid_experiment_fixture();
        eval_marker_spec.agents[0].model.config.insert(
            "hook".to_owned(),
            SpecConfigValue::String("eval(x + 1)".to_owned()),
        );
        assert!(matches!(
            eval_marker_spec.validate(),
            Err(SpecError::CodeEvalRejected {
                context: "ModelSpecRef.config",
                ref field,
                ..
            }) if field == "hook"
        ));
    }

    #[test]
    fn migrates_legacy_experiment_v0_to_v0_2_and_reports_diagnostics() {
        let base_system = canonical_two_node_fixture();
        let legacy_agent = AgentSpec::new("legacy_agent", base_system.clone())
            .with_config("exploration", SpecConfigValue::f64(0.2).unwrap());
        let legacy_exp = ExperimentSpec::new("exp_legacy_01", 42, base_system)
            .with_agent(legacy_agent)
            .with_config("episodes", SpecConfigValue::I64(5));
        let legacy_json = legacy_exp.to_canonical_json().unwrap();

        let migrated = migrate_experiment_spec_json(&legacy_json)
            .expect("v0 -> v0.2 migration must succeed");
        assert_eq!(
            migrated.spec.schema_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(
            migrated.spec.agents[0].schema_version,
            ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(migrated.spec.experiment_id, "exp_legacy_01");
        assert_eq!(migrated.spec.seeds.dev_seeds, vec![42]);
        assert_eq!(migrated.spec.seeds.holdout_seeds, vec![43]);
        assert!(migrated.spec.seeds.is_disjoint());
        assert_eq!(migrated.spec.agents.len(), 1);
        assert_eq!(migrated.spec.agents[0].actions, vec!["stay"]);
        assert_eq!(
            migrated.report.source_version,
            EXPERIMENT_SPEC_SCHEMA_VERSION
        );
        assert_eq!(
            migrated.report.target_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert!(migrated
            .report
            .applied_steps
            .iter()
            .any(|step| step == "upgrade_schema_version:v0->v0.1"));
        assert!(migrated
            .report
            .applied_steps
            .iter()
            .any(|step| step == "upgrade_schema_version:v0.1->v0.2"));
        assert!(!migrated.report.diagnostics.is_empty());

        let registry = ActionRegistry::canonical_grid_defaults();
        migrated
            .spec
            .validate_with_registry(&registry)
            .expect("migrated spec must validate with default registry");

        // Compatibilidad: un artefacto v0.1 (incluido el golden del DSL cognitivo) sigue
        // parseándose y ascendiendo hasta el contrato v0.2.
        let v01_json = canonical_grid_experiment_fixture_json();
        let lifted = migrate_experiment_spec_json(&v01_json).expect("v0.1 lift must succeed");
        assert_eq!(
            lifted.report.source_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION
        );
        assert_eq!(
            lifted.report.target_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(
            lifted.report.applied_steps,
            vec!["upgrade_schema_version:v0.1->v0.2"]
        );

        // Un artefacto ya v0.2 es un passthrough sin pasos aplicados.
        let v02_json = canonical_grid_endowment_experiment_fixture_json();
        let passthrough =
            migrate_experiment_spec_json(&v02_json).expect("v0.2 passthrough must succeed");
        assert_eq!(
            passthrough.report.source_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(
            passthrough.report.target_version,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
        );
        assert_eq!(
            passthrough.report.applied_steps,
            vec!["already_at_target_version"]
        );
        assert!(passthrough.report.diagnostics.is_empty());

        let bad_version_json = v01_json.replace(
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
            "gogamen.experiment_spec.v99",
        );
        let err = migrate_experiment_spec_json(&bad_version_json).unwrap_err();
        assert_eq!(err.diagnostic.field_path, "$.schema_version");
        assert_eq!(
            err.diagnostic.found_version.as_deref(),
            Some("gogamen.experiment_spec.v99")
        );
        assert!(
            err.diagnostic
                .expected_version
                .as_deref()
                .unwrap()
                .contains(ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION)
        );
        assert!(matches!(
            err.source.as_deref(),
            Some(SpecError::IncompatibleSchemaVersion { .. })
        ));
    }
}
