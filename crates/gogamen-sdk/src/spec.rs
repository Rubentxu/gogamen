//! Superficie pública de autoría declarativa (`SystemSpec`) y compilación (`SpecCompiler`).
//!
//! # Estabilidad SemVer pre-1.0
//! Este módulo pertenece a `gogamen-sdk` (`0.1.0-alpha`, [`crate::SDK_SEMVER`]).
//! La API de autoría y compilación puede evolucionar en versiones `0.x` previas a `1.0.0`
//! según [`crate::API_STABILITY_POLICY`], manteniendo el formato de esquema versionado
//! (`SYSTEM_SPEC_SCHEMA_VERSION`).
//!
//! # Errores
//! - [`SpecError`]: errores de validación estructural, esquemas incompatibles, campos
//!   desconocidos o topología causal inválida al construir o parsear [`SystemSpec`],
//!   [`AgentSpec`] o [`ExperimentSpec`].
//! - [`CompileError`] / [`CompileAndInitError`]: errores de compilación estática
//!   (resolución de módulos/operaciones, permisos, perfiles temporales o inicialización).

pub use gogamen_compiler::{
    BoundedExprEvaluator, COMPILER_PLAN_VERSION, CompileAndInitError, CompileAndInitResult,
    CompileError, CompiledExpr, ConstantIndex, ConstantValue, ExecutablePlan, ExprCompileError,
    ExprEvalError, ExprSchema, ExprValue, FactoryIndex, InstantiatedPlan, OperationIndex,
    ResolvedFactory, ResolvedOperation, ResourceInitializer, SpecCompiler, VarIndex,
    VariableSlotRegistry,
};
pub use gogamen_spec::{
    AGENT_SPEC_SCHEMA_VERSION, AgentSpec, EXPERIMENT_SPEC_SCHEMA_VERSION, ExperimentSpec, ExprAst,
    ExprType, ExprUnit, SYSTEM_SPEC_SCHEMA_VERSION, SpecConfigMap, SpecConfigValue, SpecError,
    SystemConnectionSpec, SystemNodeSpec, SystemSpec, UnknownFieldPolicy,
    canonical_two_node_fixture, canonical_two_node_fixture_json,
};
