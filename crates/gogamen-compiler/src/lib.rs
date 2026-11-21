//! Compilador declarativo de especificaciones portables [`gogamen_spec::SystemSpec`]
//! hacia un [`ExecutablePlan`] inmutable y pre-resuelto (`std`-only).
//!
//! Este crate valida la topología causal, capacidades, permisos, versiones,
//! perfil temporal y constantes **antes** de inicializar o asignar ningún recurso,
//! fábrica o actuador:
//! - [`SpecCompiler::compile`]: valida y resuelve nodos, puertos, operaciones
//!   ([`OperationIndex`], [`FactoryIndex`]) y constantes ([`ConstantIndex`]).
//! - [`ExecutablePlan`]: plan inmutable separado de cualquier estado mutable de
//!   ejecución, con huella semántica canónica (`semantic_fingerprint: u64`)
//!   calculada mediante [`gogamen_core::TraceHasher`].

mod artifact;
mod compiler;
mod error;
pub mod expr_eval;
mod plan;

pub use artifact::{
    ArtifactError, ArtifactInspection, ArtifactLoadAndInitError, ArtifactLoadAndInitResult,
    COMPILED_ARTIFACT_RUNTIME_VERSION, COMPILED_ARTIFACT_VERSION, CompiledPlanArtifact,
    catalog_fingerprint, load_plan_from_artifact_json,
};
pub use compiler::{CompileAndInitError, CompileAndInitResult, SpecCompiler};
pub use error::CompileError;
pub use expr_eval::{
    BoundedExprEvaluator, CompiledExpr, ExprCompileError, ExprEvalError, ExprSchema, ExprValue,
    VarIndex, VariableSlotRegistry,
};
pub use plan::{
    COMPILER_PLAN_VERSION, ConstantIndex, ConstantValue, ExecutablePlan, FactoryIndex,
    InstantiatedPlan, OperationIndex, ResolvedFactory, ResolvedOperation, ResourceInitializer,
};

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::{
        CapabilityCatalog, CapabilityVersion, ConnectionError, EffectKind, ModuleContractError,
        ModuleManifest, ModuleOrigin, OperationSpec, Permission, PortType, TemporalProfile,
    };
    use gogamen_spec::{SpecConfigValue, SystemConnectionSpec, SystemNodeSpec, SystemSpec};

    #[derive(Default)]
    struct SpyInitializer {
        factory_inits: usize,
        actuator_inits: usize,
    }

    impl ResourceInitializer for SpyInitializer {
        type Resource = String;
        type Actuator = String;
        type Error = &'static str;

        fn init_factory(
            &mut self,
            factory: &ResolvedFactory,
        ) -> Result<Self::Resource, Self::Error> {
            self.factory_inits += 1;
            Ok(format!("{}::{}", factory.module_id, factory.operation_id))
        }

        fn init_actuator(
            &mut self,
            operation: &ResolvedOperation,
            factory: &ResolvedFactory,
        ) -> Result<Self::Actuator, Self::Error> {
            self.actuator_inits += 1;
            Ok(format!(
                "op#{}:{}",
                operation.index.as_usize(),
                factory.operation_id
            ))
        }
    }

    fn base_catalog() -> CapabilityCatalog {
        let mut catalog = CapabilityCatalog::new();

        let math = ModuleManifest::new(
            "core.math",
            ModuleOrigin::Builtin,
            CapabilityVersion::new(1, 2, 0),
        )
        .with_granted_permission(Permission::state_read())
        .with_operation(
            OperationSpec::new("integrate", CapabilityVersion::new(1, 2, 0))
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64)
                .with_required_permission(Permission::state_read())
                .with_effect(EffectKind::Pure)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        )
        .with_operation(
            OperationSpec::new("gain", CapabilityVersion::new(1, 1, 0))
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64)
                .with_effect(EffectKind::Pure)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        );

        let io = ModuleManifest::new(
            "ext.io",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permission(Permission::filesystem())
        .with_operation(
            OperationSpec::new("log", CapabilityVersion::new(1, 0, 0))
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64)
                .with_required_permission(Permission::filesystem())
                .with_effect(EffectKind::BlockingIo)
                .with_temporal_profile(TemporalProfile::UnboundedBackground),
        );

        catalog.register_module(math).expect("register core.math");
        catalog.register_module(io).expect("register ext.io");
        catalog
    }

    fn valid_two_node_spec() -> SystemSpec {
        SystemSpec::new("sys-01")
            .with_config("dt", SpecConfigValue::F64(0.01))
            .with_node(
                SystemNodeSpec::new("n1", "core.math::gain")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64)
                    .with_config("factor", SpecConfigValue::F64(2.5))
                    .with_config("version", SpecConfigValue::String("1.0.0".to_owned())),
            )
            .with_node(
                SystemNodeSpec::new("n2", "core.math::integrate")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64)
                    .with_config("bias", SpecConfigValue::F64(0.0))
                    .with_config("version", SpecConfigValue::String("1.2.0".to_owned())),
            )
            .with_connection(SystemConnectionSpec::instant("n1", "out", "n2", "in"))
            .with_connection(SystemConnectionSpec::delayed("n2", "out", "n1", "in", 1))
    }

    #[test]
    fn invalid_spec_fails_before_initializing_any_resource_or_actuator() {
        let catalog = base_catalog();

        // 1. Ciclo causal instantáneo.
        let cycle_spec = SystemSpec::new("cycle-sys")
            .with_node(
                SystemNodeSpec::new("a", "core.math::gain")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64),
            )
            .with_node(
                SystemNodeSpec::new("b", "core.math::integrate")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64),
            )
            .with_connection(SystemConnectionSpec::instant("a", "out", "b", "in"))
            .with_connection(SystemConnectionSpec::instant("b", "out", "a", "in"));

        let mut spy = SpyInitializer::default();
        let err = SpecCompiler::compile_and_instantiate(
            &cycle_spec,
            &catalog,
            TemporalProfile::BoundedTick,
            &mut spy,
        )
        .expect_err("causal cycle must fail");
        assert!(matches!(
            err,
            CompileAndInitError::Compile(CompileError::Spec(gogamen_spec::SpecError::Graph(
                ConnectionError::InstantCycleDetected { .. }
            )))
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // 2. BlockingIo en ruta acotada (BoundedTick).
        let blocking_spec = SystemSpec::new("blocking-sys").with_node(
            SystemNodeSpec::new("logger", "ext.io::log")
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64),
        );
        let err = SpecCompiler::compile_and_instantiate(
            &blocking_spec,
            &catalog,
            TemporalProfile::BoundedTick,
            &mut spy,
        )
        .expect_err("BlockingIo in bounded path must fail");
        assert!(matches!(
            err,
            CompileAndInitError::Compile(CompileError::Contract(
                ModuleContractError::BlockingIoInBoundedPath { .. }
            ))
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // 3. Versión incompatible.
        let bad_ver_spec = SystemSpec::new("bad-ver").with_node(
            SystemNodeSpec::new("n1", "core.math::integrate")
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64)
                .with_config("version", SpecConfigValue::String("2.0.0".to_owned())),
        );
        let err = SpecCompiler::compile_and_instantiate(
            &bad_ver_spec,
            &catalog,
            TemporalProfile::BoundedTick,
            &mut spy,
        )
        .expect_err("incompatible version must fail");
        assert!(matches!(
            err,
            CompileAndInitError::Compile(CompileError::Contract(
                ModuleContractError::IncompatibleVersion { .. }
            ))
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // 4. Puerto incompatible con contrato de operación.
        let bad_port_spec = SystemSpec::new("bad-port").with_node(
            SystemNodeSpec::new("n1", "core.math::gain")
                .with_input("in", PortType::I64)
                .with_output("out", PortType::F64),
        );
        let err = SpecCompiler::compile_and_instantiate(
            &bad_port_spec,
            &catalog,
            TemporalProfile::BoundedTick,
            &mut spy,
        )
        .expect_err("port type mismatch must fail");
        assert!(matches!(
            err,
            CompileAndInitError::Compile(CompileError::PortContractMismatch { .. })
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);
    }

    #[test]
    fn compiled_plan_resolves_indices_without_string_lookups_during_step() {
        let catalog = base_catalog();
        let spec = valid_two_node_spec();

        let mut spy = SpyInitializer::default();
        let (plan, instantiated) = SpecCompiler::compile_and_instantiate(
            &spec,
            &catalog,
            TemporalProfile::BoundedTick,
            &mut spy,
        )
        .expect("valid spec must compile and instantiate");

        assert_eq!(spy.factory_inits, 2);
        assert_eq!(spy.actuator_inits, 2);
        assert_eq!(instantiated.resources.len(), 2);
        assert_eq!(instantiated.actuators.len(), 2);

        // Setup: resolver constantes una sola vez fuera de step.
        let dt_idx = plan.system_constant_index("dt").expect("dt constant");
        assert_eq!(plan.constant(dt_idx), Some(&ConstantValue::F64(0.01)));

        // Simulación de step usando únicamente índices resueltos (NodeIndex, OperationIndex,
        // FactoryIndex, ConstantIndex, PortIndex, BufferSlot).
        let mut executed_ops = Vec::new();
        for &node_idx in plan.execution_order() {
            let op = plan.node_operation(node_idx).expect("op by NodeIndex");
            let factory = plan.factory(op.factory).expect("factory by FactoryIndex");
            assert_eq!(factory.temporal_profile, TemporalProfile::BoundedTick);

            for &const_idx in &op.constants {
                let _val = plan.constant(const_idx).expect("const by ConstantIndex");
            }
            for transfer in plan.instant_outgoing(node_idx) {
                assert_ne!(transfer.from_node, transfer.to_node);
            }
            executed_ops.push((op.index, op.factory));
        }

        assert_eq!(executed_ops.len(), 2);
        assert_eq!(plan.delayed_transfers().len(), 1);
    }

    #[test]
    fn equivalent_canonical_writings_produce_identical_semantic_fingerprint() {
        let catalog = base_catalog();

        let json_a = r#"{
            "schema_version": "gogamen.system_spec.v0",
            "spec_id": "canon-sys",
            "config": { "alpha": 1.0, "zero": -0.0 },
            "nodes": [
                {
                    "id": "n1",
                    "kind": "core.math::gain",
                    "ports": [
                        { "id": "in", "direction": "input", "port_type": "f64" },
                        { "id": "out", "direction": "output", "port_type": "f64" }
                    ],
                    "config": { "factor": 2.0, "version": "1.0.0" }
                },
                {
                    "id": "n2",
                    "kind": "core.math::integrate",
                    "ports": [
                        { "id": "in", "direction": "input", "port_type": "f64" },
                        { "id": "out", "direction": "output", "port_type": "f64" }
                    ],
                    "config": { "bias": 0.5, "version": "1.2.0" }
                }
            ],
            "connections": [
                { "from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in", "delay_ticks": 0 },
                { "from_node": "n2", "from_port": "out", "to_node": "n1", "to_port": "in", "delay_ticks": 2 }
            ]
        }"#;

        let json_b = r#"{
            "connections": [
                { "from_node": "n2", "from_port": "out", "to_node": "n1", "to_port": "in", "delay_ticks": 2 },
                { "from_node": "n1", "from_port": "out", "to_node": "n2", "to_port": "in", "delay_ticks": 0 }
            ],
            "nodes": [
                {
                    "config": { "version": "1.2.0", "bias": 0.5 },
                    "ports": [
                        { "id": "out", "direction": "output", "port_type": "f64" },
                        { "id": "in", "direction": "input", "port_type": "f64" }
                    ],
                    "kind": "core.math::integrate",
                    "id": "n2"
                },
                {
                    "config": { "version": "1.0.0", "factor": 2.0 },
                    "ports": [
                        { "id": "out", "direction": "output", "port_type": "f64" },
                        { "id": "in", "direction": "input", "port_type": "f64" }
                    ],
                    "kind": "core.math::gain",
                    "id": "n1"
                }
            ],
            "config": { "zero": 0.0, "alpha": 1.0 },
            "spec_id": "canon-sys",
            "schema_version": "gogamen.system_spec.v0"
        }"#;

        let spec_a = SystemSpec::from_json(json_a).expect("spec_a");
        let spec_b = SystemSpec::from_json(json_b).expect("spec_b");

        let plan_a =
            SpecCompiler::compile(&spec_a, &catalog, TemporalProfile::BoundedTick).expect("plan_a");
        let plan_b =
            SpecCompiler::compile(&spec_b, &catalog, TemporalProfile::BoundedTick).expect("plan_b");

        assert_eq!(plan_a.semantic_fingerprint(), plan_b.semantic_fingerprint());

        // Cualquier cambio semántico altera el fingerprint.
        let mut spec_c = spec_b;
        spec_c
            .config
            .insert("alpha".to_owned(), SpecConfigValue::F64(1.5));
        let plan_c =
            SpecCompiler::compile(&spec_c, &catalog, TemporalProfile::BoundedTick).expect("plan_c");
        assert_ne!(plan_a.semantic_fingerprint(), plan_c.semantic_fingerprint());
    }

    #[test]
    fn expr_energy_times_curiosity_and_threshold_matches_rust_calculation() {
        use gogamen_spec::{ExprAst, ExprType, ExprUnit};

        let mut schema = ExprSchema::new();
        let energia_idx = schema
            .register("energia", ExprType::Scalar(ExprUnit::Energy))
            .unwrap();
        let curiosidad_idx = schema
            .register("curiosidad", ExprType::Scalar(ExprUnit::Ratio))
            .unwrap();
        let umbral_idx = schema
            .register("umbral", ExprType::Scalar(ExprUnit::Energy))
            .unwrap();

        assert_eq!(energia_idx.as_usize(), 0);
        assert_eq!(curiosidad_idx.as_usize(), 1);
        assert_eq!(umbral_idx.as_usize(), 2);

        let mul_ast = ExprAst::mul(
            ExprAst::var("energia").unwrap(),
            ExprAst::var("curiosidad").unwrap(),
        );
        let mul_compiled = CompiledExpr::compile(&mul_ast, &schema).unwrap();
        assert_eq!(
            mul_compiled.result_type(),
            ExprType::Scalar(ExprUnit::Energy)
        );

        let threshold_ast = ExprAst::gt(mul_ast, ExprAst::var("umbral").unwrap());
        let threshold_compiled = CompiledExpr::compile(&threshold_ast, &schema).unwrap();
        assert_eq!(threshold_compiled.result_type(), ExprType::Bool);

        let cases = [
            (12.5_f64, 0.4_f64, 4.9_f64),
            (10.0, 0.5, 5.0),
            (3.25, 0.8, 3.0),
        ];

        for (energia, curiosidad, umbral) in cases {
            let values = [
                ExprValue::Scalar(energia),
                ExprValue::Scalar(curiosidad),
                ExprValue::Scalar(umbral),
            ];
            let eval_mul = BoundedExprEvaluator::eval(&mul_compiled, &values, 16).unwrap();
            let expected_mul = energia * curiosidad;
            assert_eq!(eval_mul, ExprValue::Scalar(expected_mul));

            let eval_rule = BoundedExprEvaluator::eval(&threshold_compiled, &values, 16).unwrap();
            let expected_rule = (energia * curiosidad) > umbral;
            assert_eq!(eval_rule, ExprValue::Bool(expected_rule));
        }
    }

    #[test]
    fn expr_rejects_invalid_types_units_unknown_vars_non_finite_and_div_zero() {
        use gogamen_spec::{ExprAst, ExprType, ExprUnit};

        let mut schema = ExprSchema::new();
        schema
            .register("energia", ExprType::Scalar(ExprUnit::Energy))
            .unwrap();
        schema
            .register("curiosidad", ExprType::Scalar(ExprUnit::Ratio))
            .unwrap();
        schema
            .register("segundos", ExprType::Scalar(ExprUnit::Seconds))
            .unwrap();
        schema.register("activo", ExprType::Bool).unwrap();

        // Unidades incompatibles en suma y comparación.
        let bad_add = ExprAst::add(
            ExprAst::var("energia").unwrap(),
            ExprAst::var("curiosidad").unwrap(),
        );
        assert!(matches!(
            CompiledExpr::compile(&bad_add, &schema),
            Err(ExprCompileError::IncompatibleUnits {
                op: "add",
                lhs: ExprUnit::Energy,
                rhs: ExprUnit::Ratio,
            })
        ));

        let bad_cmp = ExprAst::gt(
            ExprAst::var("energia").unwrap(),
            ExprAst::var("segundos").unwrap(),
        );
        assert!(matches!(
            CompiledExpr::compile(&bad_cmp, &schema),
            Err(ExprCompileError::IncompatibleUnits {
                op: "gt",
                lhs: ExprUnit::Energy,
                rhs: ExprUnit::Seconds,
            })
        ));

        // Tipos incompatibles en operadores booleanos y relacionales.
        let bad_and = ExprAst::and(
            ExprAst::var("energia").unwrap(),
            ExprAst::var("activo").unwrap(),
        );
        assert!(matches!(
            CompiledExpr::compile(&bad_and, &schema),
            Err(ExprCompileError::TypeMismatch { op: "and", .. })
        ));

        let bad_lt = ExprAst::lt(ExprAst::var("activo").unwrap(), ExprAst::const_bool(false));
        assert!(matches!(
            CompiledExpr::compile(&bad_lt, &schema),
            Err(ExprCompileError::TypeMismatch { op: "lt", .. })
        ));

        // Variable inexistente.
        let missing_var = ExprAst::var("inexistente").unwrap();
        assert!(matches!(
            CompiledExpr::compile(&missing_var, &schema),
            Err(ExprCompileError::UnknownVariable { ref name }) if name == "inexistente"
        ));

        // Constantes no finitas (NaN / Inf).
        let nan_ast = ExprAst::ConstF64 {
            value: f64::NAN,
            unit: ExprUnit::Dimensionless,
        };
        assert!(matches!(
            CompiledExpr::compile(&nan_ast, &schema),
            Err(ExprCompileError::NonFiniteConstant)
        ));

        let inf_ast = ExprAst::ConstF64 {
            value: f64::INFINITY,
            unit: ExprUnit::Energy,
        };
        assert!(matches!(
            CompiledExpr::compile(&inf_ast, &schema),
            Err(ExprCompileError::NonFiniteConstant)
        ));

        // División por cero en compilación y en evaluación.
        let div_zero_const = ExprAst::div(
            ExprAst::var("energia").unwrap(),
            ExprAst::const_f64(0.0).unwrap(),
        );
        assert!(matches!(
            CompiledExpr::compile(&div_zero_const, &schema),
            Err(ExprCompileError::DivisionByZero)
        ));

        let div_var = ExprAst::div(
            ExprAst::var("energia").unwrap(),
            ExprAst::var("curiosidad").unwrap(),
        );
        let compiled_div = CompiledExpr::compile(&div_var, &schema).unwrap();
        let err = BoundedExprEvaluator::eval(
            &compiled_div,
            &[ExprValue::Scalar(10.0), ExprValue::Scalar(0.0)],
            16,
        )
        .unwrap_err();
        assert_eq!(err, ExprEvalError::DivisionByZero);

        // Entrada no finita en tiempo de evaluación.
        let err_nan = BoundedExprEvaluator::eval(
            &compiled_div,
            &[ExprValue::Scalar(f64::NAN), ExprValue::Scalar(1.0)],
            16,
        )
        .unwrap_err();
        assert_eq!(
            err_nan,
            ExprEvalError::NonFiniteValue {
                context: "variable input"
            }
        );
    }

    #[test]
    fn expr_evaluator_enforces_max_steps_and_depth_bounds() {
        use gogamen_spec::{ExprAst, ExprType, ExprUnit};

        let mut schema = ExprSchema::new();
        schema
            .register("energia", ExprType::Scalar(ExprUnit::Energy))
            .unwrap();
        schema
            .register("curiosidad", ExprType::Scalar(ExprUnit::Ratio))
            .unwrap();
        schema
            .register("umbral", ExprType::Scalar(ExprUnit::Energy))
            .unwrap();

        let ast = ExprAst::gt(
            ExprAst::mul(
                ExprAst::var("energia").unwrap(),
                ExprAst::var("curiosidad").unwrap(),
            ),
            ExprAst::var("umbral").unwrap(),
        );

        // Profundidad del AST es 3 (gt -> mul -> var). Límite 2 debe rechazarse.
        assert!(matches!(
            CompiledExpr::compile_with_max_depth(&ast, &schema, 2),
            Err(ExprCompileError::DepthLimitExceeded {
                depth: 3,
                max_depth: 2
            })
        ));

        let compiled = CompiledExpr::compile(&ast, &schema).unwrap();
        assert_eq!(compiled.instruction_count(), 5);

        let values = [
            ExprValue::Scalar(10.0),
            ExprValue::Scalar(0.5),
            ExprValue::Scalar(4.0),
        ];
        let err = BoundedExprEvaluator::eval(&compiled, &values, 4).unwrap_err();
        assert_eq!(err, ExprEvalError::StepLimitExceeded { max_steps: 4 });

        let ok = BoundedExprEvaluator::eval(&compiled, &values, 5).unwrap();
        assert_eq!(ok, ExprValue::Bool(true));
    }
}
