//! Preludio ergonómico del SDK público de Gogamen (`0.1.0-alpha`).
//!
//! # Estabilidad SemVer pre-1.0
//! Reexporta los traits, tipos de dominio, unidades, errores y el Conformance Kit
//! más frecuentes para consumidores externos. Sujeto a [`crate::API_STABILITY_POLICY`].

pub use crate::conformance::{
    CapabilityConformanceReport, ConformanceConfig, ConformanceError, ConformanceReport,
    ReplayReport, ReplayStepRecord, run_capability_conformance_suite, run_conformance_suite,
    verify_action_rejection, verify_authority_isolation, verify_capability_catalog_contract,
    verify_effect_replay, verify_extension_registry_contract, verify_fork_isolation,
    verify_live_env_determinism, verify_live_env_fork_isolation, verify_model_determinism,
    verify_planner_budget_compliance, verify_temporal_contract, verify_world_determinism,
};
pub use crate::extensions::{
    CapabilityId, CapabilityRequirement, DynPlanner, EXTENSION_CONTRACT_SCHEMA_V1,
    EXTENSION_HOST_VERSION_V1, ExtensionActivationContext, ExtensionError, ExtensionId,
    ExtensionKind, ExtensionManifest, ExtensionSource, StaticExtensionRegistry,
};
pub use crate::runtime::{
    ExecutionTrace, ExternalKernel, HeadlessRuntime, KernelRegistry, PlanInstance, PlanRunSummary,
    PlanRuntimeEngine, PlanRuntimeError, PortValue, RuntimeConfig, RuntimeError, TickTraceEntry,
    builtin_plan_catalog,
};
pub use crate::spec::{
    AgentSpec, CompileAndInitError, CompileError, ExecutablePlan, ExperimentSpec, InstantiatedPlan,
    ResourceInitializer, SpecCompiler, SpecConfigValue, SpecError, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec, UnknownFieldPolicy,
};
pub use crate::{
    API_STABILITY_POLICY, Action, ActionBounds, ActionCandidate, ActionGuard, ActionProposal,
    ActionSpace, Actuator, AgentMemoryScope, ApplyOutcome, BinaryBelief, BoundedEpisodicMemory,
    BranchableSnapshot, BudgetTracker, CancelToken, CandidateEvaluation, CapabilityCatalog,
    CapabilityVersion, CausalKind, Confidence, ConstraintViolation, DeterministicRng,
    DiscreteAction, DomainError, DomainPrecondition, DriveId, DriveReading, DriveSpec, Duration,
    DynamicUtility, EffectKind, EntityId, EnvironmentError, EnvironmentObserver, EpisodeId,
    EpisodeInput, EpisodeProvenance, EpisodicRecord, EvaluationOutcome, EvidenceRef, FmcPlanner,
    FmcPlannerConfig, ForbiddenActionRule, FrameId, GenerativeModel, GuardContext, GuardDecision,
    GuardRejectionReason, HardConstraint, HomeostaticState, IncrementalLearner, IngressStatus,
    LearningPolicy, LiveEnvironment, MasterSeed, MemoryError, MemoryOperation, MemoryPort,
    MemorySnapshot, MessageId, ModelError, ModuleContractError, ModuleId, ModuleManifest,
    ModuleOrigin, MonotonicClock, MonotonicDeadline, MonotonicInstant, MotivationError,
    NoPrecondition, ObjectiveTerm, Observation, ObservationEnvelope, ObservationId,
    ObservationKind, OperationBinding, OperationId, OperationSpec, OwnerId, Permission,
    PermissionSet, PlanBudget, PlanRequest, PlanResult, PlanStatus, PlanTarget, Planner,
    PlanningContext, PortSpec, PortType, ProvenanceTracker, RetentionPolicy, Reward, RngSource,
    RngStreamRegistry, SDK_SEMVER, SafeFallbackPolicy, Seed, SelectedAction, SensorId, SimDuration,
    SimulationSnapshot, SnapshotEnvelope, SnapshotError, SnapshotMetrics, Snapshotable, State,
    StepDt, StreamKey, SummaryCode, SystemMonotonicClock, TabularTransitionLearner,
    TemporalProfile, TemporalReplayVerdict, TemporalTrace, TemporalTraceEntry, Terminal,
    TerminalReason, Tick, TraceHasher, Transition, TransitionObservation, TransitionSample,
    UnitTag, UtilityBreakdown, UtilityEvaluator, ValidatedOperation, ValidationError,
    VirtualMonotonicClock, WeightedObjective, WorkUnits, World, WorldSnapshot,
};
