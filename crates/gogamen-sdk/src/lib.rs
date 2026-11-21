//! SDK público de Rust para Gogamen (`0.1.0-alpha`, `std`-only offline).
//!
//! `gogamen-sdk` es la fachada pública unificada para consumidores externos que implementan
//! mundos ([`World`]), entornos vivos ([`LiveEnvironment`]), modelos generativos
//! ([`GenerativeModel`]), snapshots ramificables ([`BranchableSnapshot`]) y planificadores
//! ([`Planner`]), así como autoría declarativa mediante [`SystemSpec`] y compilación estática
//! con [`SpecCompiler`]. No depende de `labs/*`, frontends externos ni crates de terceros.
//!
//! # Estabilidad SemVer pre-1.0
//! - **Versión publicada:** `0.1.0-alpha` (expuesta en tiempo de ejecución como [`SDK_SEMVER`]).
//! - **Política ([`API_STABILITY_POLICY`]):** este SDK se encuentra en fase **pre-1.0**.
//!   Las firmas de traits, estructuras auxiliares y diagnósticos pueden introducir cambios
//!   incompatibles entre versiones `0.x` menores o pre-releases (`0.1.0-alpha`, `0.2.0`, etc.)
//!   hasta la estabilización `1.0.0`.
//! - **Aislamiento:** el núcleo público no expone dependencias hacia `labs/*`, Bevy ni Jev.
//!
//! # Unidades explícitas
//! Todas las magnitudes físicas y de recursos usan tipos o convenciones explícitas:
//! - **`StepDt` ([`StepDt`]):** paso temporal físico de simulación en **segundos** (`f64`),
//!   siempre finito y estrictamente positivo (`dt > 0.0`). Rechaza `NaN`, infinitos y `<= 0.0`.
//! - **`SimDuration` / `Duration` ([`SimDuration`]):** tiempo acumulado de simulación en
//!   **segundos** (`f64`), siempre finito y no negativo (`>= 0.0`).
//! - **`Tick` ([`Tick`]):** contador discreto adimensional de pasos de simulación (`u64`).
//! - **`Reward` ([`Reward`]):** recompensa o utilidad escalar adimensional (`f64`), finita
//!   (valores mayores indican mejor desempeño; no asume unidad monetaria ni física).
//! - **`Budget` ([`PlanBudget`]):** presupuesto de planificación acotado por número máximo de
//!   transiciones simuladas (`max_transitions: usize`) y un deadline monotónico opcional
//!   (`std::time::Instant`).
//! - **`MasterSeed` ([`MasterSeed`]):** semilla raíz determinista (`u64`) para derivar
//!   streams reproducibles sin estado global compartido.
//!
//! # Errores y semántica de fallos
//! Los errores se separan por frontera arquitectónica para evitar mezclar simulación pura,
//! actuación viva, autoría declarativa y verificación:
//! - [`DomainError`]: invariantes de valor ([`StepDt`], [`SimDuration`], [`State`],
//!   [`Observation`], [`ActionSpace`]) o llamada a [`World::step`] tras [`Terminal::Terminated`].
//! - [`ModelError`]: fallos de simulación pura en [`GenerativeModel`] o [`Planner`]
//!   (no puede acceder a [`Actuator`] ni [`LiveEnvironment`]).
//! - [`EnvironmentError`]: fallos de observación, actuación o reinicio en [`LiveEnvironment`].
//! - [`SnapshotError`]: corrupción de checksum, versión incompatible o estado inconsistente
//!   en [`BranchableSnapshot`] / [`SimulationSnapshot`].
//! - [`SpecError`]: JSON inválido, versión de esquema incompatible, campos desconocidos o
//!   topología causal inválida en [`SystemSpec`], [`AgentSpec`] o [`ExperimentSpec`].
//! - [`CompileError`] / [`CompileAndInitError`]: errores de resolución de capacidades,
//!   permisos, perfiles temporales o inicialización en [`SpecCompiler`].
//! - [`RuntimeError`] / [`PlanRuntimeError`]: errores de ejecución del bucle headless
//!   (`observe -> plan -> guard -> act`) o del motor de planes compilados.
//! - [`ExtensionError`]: errores de registro estático, compatibilidad de capacidades/versiones,
//!   rechazo de orígenes remotos/dinámicos o permisos insuficientes en [`StaticExtensionRegistry`].
//! - [`conformance::ConformanceError`]: violaciones de determinismo local, aislamiento `fork`,
//!   presupuesto [`PlanBudget`] o rechazo de acción en [`ActionGuard`].

pub mod conformance;
pub mod extensions;
pub mod lab;
pub mod prelude;
pub mod runtime;
pub mod spec;
#[cfg(feature = "wasm")]
pub mod wasm;

/// Versión SemVer pre-1.0 publicada por `gogamen-sdk`.
pub const SDK_SEMVER: &str = env!("CARGO_PKG_VERSION");

/// Política explícita de estabilidad SemVer pre-1.0 del SDK público de Gogamen.
pub const API_STABILITY_POLICY: &str = "pre-1.0 (0.1.0-alpha): breaking API changes may occur in 0.x releases prior to 1.0.0 stabilization; no dependency on labs/* or external frontends.";

/// Recompensa o utilidad escalar adimensional (`f64` finito).
pub type Reward = f64;

pub use conformance::{
    CapabilityConformanceReport, ConformanceConfig, ConformanceError, ConformanceReport,
    ReplayReport, ReplayStepRecord, run_capability_conformance_suite, run_conformance_suite,
    verify_action_rejection, verify_authority_isolation, verify_capability_catalog_contract,
    verify_effect_replay, verify_extension_registry_contract, verify_fork_isolation,
    verify_live_env_determinism, verify_live_env_fork_isolation, verify_model_determinism,
    verify_planner_budget_compliance, verify_temporal_contract, verify_world_determinism,
};
pub use extensions::{
    AdapterFactory, CapabilityId, CapabilityRequirement, DynPlanner, EXTENSION_CONTRACT_SCHEMA_V1,
    EXTENSION_HOST_VERSION_V1, ExtensionActivationContext, ExtensionError, ExtensionId,
    ExtensionKind, ExtensionManifest, ExtensionSource, PlanKernelFactory, PlannerFactory,
    SensorFactory, StaticExtensionRegistry,
};
pub use gogamen_core::{
    Action, ActionBounds, ActionCandidate, ActionGuard, ActionProposal, ActionSpace, Actuator,
    AgentMemoryScope, ApplyOutcome, BeliefError, BeliefUpdateStatus, BinaryBelief,
    BoundedEpisodicMemory, BranchableSnapshot, BudgetTracker, CancelToken, CandidateEvaluation,
    CapabilityCatalog, CapabilityVersion, CategoricalBelief, CausalKind, Confidence,
    ConstraintViolation, DeclaredProperty, DeterministicRng, DiscreteAction, DiscreteBeliefFilter,
    DiscreteBeliefState, DomainError, DomainPrecondition, DriveId, DriveReading, DriveSpec,
    Duration, DynamicUtility, EffectKind, EntityId, EnvironmentError, EnvironmentObserver,
    EpisodeId, EpisodeInput, EpisodeProvenance, EpisodicRecord, EvaluationOutcome,
    EvidenceDeduplicator, EvidenceRef, FmcPlanner, FmcPlannerConfig, FmcPlannerError,
    FmcTraceSummary, ForbiddenActionRule, FrameId, GenerativeModel, GuardContext, GuardDecision,
    GuardRejectionReason, HardConstraint, HoldoutEval, HomeostaticState, IncrementalLearner,
    IngressStatus, IngressWindow, LEARNING_SCHEMA_VERSION, LearnOutcome, LearnUpdateStatus,
    LearningError, LearningPolicy, LikelihoodModel, LiveEnvironment, MEMORY_SCHEMA_VERSION,
    MasterSeed, MemoryError, MemoryOperation, MemoryPort, MemorySnapshot, MessageId, ModelError,
    ModuleContractError, ModuleId, ModuleManifest, ModuleOrigin, MonotonicClock, MonotonicDeadline,
    MonotonicInstant, MotivationError, NoPrecondition, ObjectiveTerm, Observation,
    ObservationEnvelope, ObservationId, ObservationKind, ObservationRecord, ObservedEvidence,
    OperationBinding, OperationId, OperationSpec, OwnerId, Permission, PermissionSet, PlanBudget,
    PlanRequest, PlanResult, PlanStatus, PlanTarget, Planner, PlanningContext, PortDirection,
    PortId, PortSpec, PortType, PredictionEval, ProvenanceError, ProvenanceMetadata,
    ProvenanceTracker, RawTextPolicy, RetentionPolicy, RngSource, RngStreamRegistry,
    SafeFallbackPolicy, Seed, SelectedAction, SensorId, SensorWatermark, SimDuration,
    SimulationSnapshot, SnapshotEnvelope, SnapshotError, SnapshotMetrics, Snapshotable, State,
    StepDt, StreamKey, SummaryCode, SystemMonotonicClock, TabularTransitionLearner,
    TemporalProfile, TemporalReplayVerdict, TemporalTrace, TemporalTraceEntry, Terminal,
    TerminalReason, Tick, TraceHasher, Transition, TransitionObservation, TransitionSample,
    UnitTag, UtilityBreakdown, UtilityEvaluator, ValidatedOperation, ValidationError,
    VirtualMonotonicClock, WallClockSample, WeightedObjective, WorkUnits, World, WorldSnapshot,
};
pub use lab::{LabDescriptor, LabDescriptorIssue, LabParam, LabUnit};
pub use runtime::{
    ExecutionTrace, ExternalKernel, ExternalNodeContext, HeadlessRuntime, KernelRegistry,
    KernelRegistryError, PlanInstance, PlanRunSummary, PlanRuntimeEngine, PlanRuntimeError,
    PortValue, ProposalMutator, RejectionCause, Run1DSummary, RuntimeConfig, RuntimeError,
    TickTraceEntry, builtin_plan_catalog, run_1d_episode,
};
pub use spec::{
    AGENT_SPEC_SCHEMA_VERSION, AgentSpec, BoundedExprEvaluator, COMPILER_PLAN_VERSION,
    CompileAndInitError, CompileAndInitResult, CompileError, CompiledExpr, ConstantIndex,
    ConstantValue, EXPERIMENT_SPEC_SCHEMA_VERSION, ExecutablePlan, ExperimentSpec, ExprAst,
    ExprCompileError, ExprEvalError, ExprSchema, ExprType, ExprUnit, FactoryIndex,
    InstantiatedPlan, OperationIndex, ResolvedFactory, ResolvedOperation, ResourceInitializer,
    SYSTEM_SPEC_SCHEMA_VERSION, SpecCompiler, SpecConfigMap, SpecConfigValue, SpecError,
    SystemConnectionSpec, SystemNodeSpec, SystemSpec, UnknownFieldPolicy, VarIndex,
    VariableSlotRegistry, canonical_two_node_fixture, canonical_two_node_fixture_json,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sdk_exposes_explicit_pre_1_0_semver_and_stability_policy() {
        assert!(SDK_SEMVER.starts_with("0.1.0-alpha"));
        assert!(API_STABILITY_POLICY.contains("pre-1.0"));
        assert!(API_STABILITY_POLICY.contains("0.1.0-alpha"));
    }

    #[test]
    fn unit_types_validate_invariants_at_construction() {
        assert!(StepDt::from_secs_f64(0.0).is_err());
        assert!(StepDt::from_secs_f64(-1.0).is_err());
        assert!(StepDt::from_secs_f64(f64::NAN).is_err());
        assert_eq!(StepDt::from_secs_f64(0.02).unwrap().as_secs_f64(), 0.02);

        assert!(SimDuration::from_secs_f64(-0.1).is_err());
        assert_eq!(SimDuration::ZERO.as_secs_f64(), 0.0);
        assert_eq!(Tick::ZERO.as_u64(), 0);
    }
}
