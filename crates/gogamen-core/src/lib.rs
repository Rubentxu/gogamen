//! Núcleo de dominio y contratos con unidades explícitas de Gogamen (`std`-only).
//!
//! Este crate no depende de motores gráficos (como Bevy), transporte HTTP ni
//! proveedores externos. Define los contratos comunes de dominio sin presuponer
//! agentes ni un `System` específico; las capacidades `world` y `action` forman
//! el núcleo sobre el cual se construirán capas superiores.
//!
//! # Convenciones transversales
//! - **Mutabilidad:** los tipos de datos de dominio ([`Tick`], [`StepDt`], [`SimDuration`],
//!   [`State`], [`Observation`], [`Action`], [`Transition`], [`Terminal`]) son valores
//!   inmutables tras su construcción. Solo las capacidades con efecto sobre un episodio
//!   ([`World::reset`] y [`World::step`]) toman `&mut self`.
//! - **Unidades:** [`StepDt`] y [`SimDuration`] expresan tiempo físico explícitamente en
//!   **segundos** (`f64`), garantizando finitud (`StepDt > 0.0`, `SimDuration >= 0.0`).
//!   [`Tick`] representa un conteo discreto adimensional (`u64`).
//! - **Semántica de terminal:** [`Terminal::Ongoing`] indica que el episodio puede
//!   continuar; [`Terminal::Terminated`] indica finalización definitiva de episodio y
//!   prohíbe nuevos pasos hasta invocar [`World::reset`].
//! - **Separación modelo simulado / entorno vivo:** [`GenerativeModel`] y [`PlanningContext`]
//!   solo permiten simulación pura (`&self`, sin mutar el estado de origen) usando [`ModelError`];
//!   [`EnvironmentObserver`], [`Actuator`] y [`LiveEnvironment`] encapsulan la observación y
//!   actuación real usando [`EnvironmentError`], sin exponer capacidad de actuación al planificador.

pub mod action;
pub mod belief;
pub mod capability;
pub mod causal_revision;
pub mod causal_rule;
pub mod clock;
pub mod communication;
pub mod conditional_effects;
pub mod connection;
pub mod control_bits;
pub mod decisive_moments;
pub mod determinism;
pub mod domain;
pub mod empowerment;
pub mod entropy;
pub mod environment;
pub mod epistemic_action;
pub mod error;
pub mod fmc_incremental;
pub mod fmc_planner;
pub mod fmc_scoring;
pub mod fmc_walkers;
pub mod freedom_field;
pub mod goals;
pub mod grounding;
pub mod guard;
pub mod learned_model;
pub mod learning;
pub mod learning_progress;
pub mod memory;
pub mod memory_views;
pub mod microstate;
pub mod microstate_profile;
pub mod model;
pub mod motivation;
pub mod observability;
pub mod planner;
pub mod posterior_sampling;
pub mod predicate_invention;
pub mod procedural_memory;
pub mod provenance;
pub mod question_tree;
pub mod racing_stop;
pub mod scale;
pub mod scheduler;
pub mod selection_principle;
pub mod semantic_memory;
pub mod snapshot;
pub mod system;
pub mod system_one;
pub mod units;
pub mod workflow;
pub mod world;

pub use action::{Action, ActionBounds, ActionSpace, DiscreteAction};
pub use belief::{
    BeliefError, BeliefUpdateStatus, BinaryBelief, CategoricalBelief, DiscreteBeliefFilter,
    DiscreteBeliefState, EpistemicFloorPolicy, EvidenceDeduplicator, LikelihoodModel,
};
pub use capability::{
    CapabilityCatalog, CapabilityVersion, DeclaredProperty, EffectKind, ModuleContractError,
    ModuleId, ModuleManifest, ModuleOrigin, ObservedEvidence, OperationBinding, OperationId,
    OperationSpec, Permission, PermissionSet, TemporalProfile, ValidatedOperation,
};
pub use causal_revision::{
    CausalRevisor, ModelRevision, RejectReason, RejectedCandidate, RevisedPrediction,
    RevisionCause, RevisionConfig, RevisionOutcome,
};
pub use clock::{
    MonotonicClock, MonotonicDeadline, MonotonicInstant, SystemMonotonicClock,
    TemporalReplayVerdict, TemporalTrace, TemporalTraceEntry, VirtualMonotonicClock,
    WallClockSample, WorkUnits,
};
pub use communication::{
    CommCost, CommPayload, CommPayloadKind, CommunicationError, CommunicationPort,
    ConversationScope, InterlocutorId, InterlocutorKind, LocalCommunicationBus, NoResponseReason,
    OfflineTemplate, OfflineTemplateKind, ReplyOutcome, SituatedMessage, TemplateId,
    TemplateRegistry,
};
pub use conditional_effects::{
    BitPrediction, ConditionalBudget, ConditionalConfig, ConditionalEffectsLearner,
    ConditionalError, ConditionalModelVersion, ConditionalObservation, ConditionalUpdateStatus,
    HoldoutScores, Literal, PredictionHealth, UpdateRecord, prior_scores,
};
pub use connection::{
    BufferSlot, CausalKind, CompiledDelayedTransfer, CompiledNodeMeta, CompiledPlan,
    CompiledPortMeta, CompiledTransfer, ConnectionError, ConnectionGraph, ConnectionIndex,
    ConnectionSpec, NodeId, NodeIndex, NodeSpec, PlanRuntime, PortDirection, PortId, PortIndex,
    PortSpec, PortType, TypeMismatchError,
};
pub use control_bits::{
    CLONING_BIAS_NOTE, CONTROL_BITS_ESTIMATOR_VERSION, CONTROL_BITS_MIN_SPEARMAN,
    ControlBitsConfig, ControlBitsError, ControlBitsEstimate, control_bits_from_pairs,
    control_bits_from_population, control_bits_from_population_on_scale, control_bits_from_walkers,
    control_bits_from_walkers_on_scale,
};
pub use decisive_moments::{
    DecisionInput, DecisionSample, DecisiveError, DecisiveEvent, DecisiveTimeline, PairPredicate,
    ThresholdSource, fit_pair_predicates, mark_decisive, predicate_accuracy, viability_spread,
};
pub use determinism::{
    DeterminismMetadata, DeterministicClock, LogicalClock, MasterSeed, RngStreamRegistry, Seed,
    StreamKey, TraceHasher,
};
pub use domain::{Observation, State, Terminal, TerminalReason, Transition};
pub use empowerment::{
    ActionSequenceSpace, BlahutArimotoConfig, DEFAULT_BLAHUT_ARIMOTO_MAX_ITERATIONS,
    DEFAULT_BLAHUT_ARIMOTO_TOLERANCE_BITS, DEFAULT_MAX_SEQUENCE_COUNT, DiscreteChannel,
    EMPOWERMENT_COMPLEXITY_NOTE, EMPOWERMENT_VS_DIVERSITY_NOTE, EmpowermentError,
    EmpowermentReport, EmpowermentScalability, EmpowermentVsDiversityComparison,
    ObservationSpaceSpec, compare_empowerment_with_diversity_proxy, compute_reference_empowerment,
};
pub use entropy::{
    BinAxisSpec, BinSpec, BinningVersion, DEFAULT_PROBABILITY_SUM_TOLERANCE, EmpiricalDistribution,
    EntropyError, EntropyProxyKind, TrajectoryEntropyEstimate, TrajectoryStep, TrajectorySummary,
    estimate_final_state_entropy_proxy, estimate_future_trajectory_entropy_proxy,
    estimate_trajectory_entropy_proxy, shannon_entropy_bits, shannon_entropy_bits_with_tolerance,
};
pub use environment::{Actuator, ApplyOutcome, EnvironmentObserver, LiveEnvironment};
pub use epistemic_action::{
    AdvanceSpec, AskDispatch, AskPollOutcome, AskSpec, BinaryCommOutcomeMapper, CommCostWeights,
    CommObservationReport, CommOutcomeMapper, DecisionOption, DecisionPayoffModel,
    EpistemicActionError, EpistemicActionKind, EpistemicActionSelector, EpistemicActionValuation,
    EpistemicController, EpistemicSelectionReport, InspectSpec, OptimalDecision,
};
pub use error::{DomainError, EnvironmentError, ModelError, SnapshotError, ValidationError};
pub use fmc_incremental::{
    CheckpointOutcome, CheckpointStatus, FmcIncrementalError, IncrementalFmcSession,
};
pub use fmc_planner::{
    FmcPlanner, FmcPlannerConfig, FmcPlannerError, FmcRootViability, FmcStepRootViability,
    FmcTraceSummary,
};
pub use fmc_scoring::{
    CompanionPair, FmcScoringConfig, FmcScoringError, FmcScratchWorkspace, PopulationScore,
    ResampleDecision, ResampleReport, cloning_probability, compute_virtual_rewards,
    is_failure_terminal, pair_companions, pair_companions_viable, relativize, relativize_viable,
    relativize_viable_into, resample_population, sample_companion_index,
    sample_companion_index_unbiased, score_population, score_population_in_place,
};
pub use fmc_walkers::{
    PopulationError, WalkerAncestry, WalkerId, WalkerPopulation, WalkerStepReport,
};
pub use freedom_field::{
    ActionMeasure, FreedomField, FreedomFieldConfig, FreedomFieldError, FreedomFieldKey,
    FreedomFieldStore, FreedomIteration, FreedomKernel,
};
pub use goals::{
    ActiveGoal, GoalAuthority, GoalAuthorityScope, GoalAuthorization, GoalChangeProposal,
    GoalError, GoalId, GoalManager, GoalMutation, GoalOrigin, GoalProposal, GoalTransitionOutcome,
    GoalTransitionRecord, PendingOrRejectedObservation,
};
pub use grounding::{
    AbstentionReason, GroundedOutcome, GroundingConfig, GroundingError, GroundingQueryMode,
    GroundingRequest, MessageSpeechAct, OperationResolutionPolicy, SemanticGrounder,
};
pub use guard::{
    ActionGuard, ActionProposal, DomainPrecondition, GuardContext, GuardDecision,
    GuardRejectionReason, NoPrecondition, SafeFallbackPolicy,
};
pub use learned_model::{
    BucketPrediction, LearnedModelConfig, LearnedModelError, LearnedStepReport, ModelVersion,
    TabularLearnedModel, TabularScale, UnseenPairPolicy, plan_fmc_on_version, plan_fmc_with_scale,
};
pub use learning::{
    DEFAULT_MAX_DEDUP_WINDOW, DEFAULT_MIN_LOG_LOSS_PROB, HoldoutEval, IncrementalLearner,
    LEARNING_SCHEMA_VERSION, LearnOutcome, LearnUpdateStatus, LearningError, LearningPolicy,
    ObservationId, PredictionEval, TabularTransitionLearner, TransitionObservation,
    TransitionSample,
};
pub use learning_progress::{
    DriveModality, DriveSignal, LearningProgressConfig, LearningProgressError,
    LearningProgressReader, RegionReading, signal_drive_spec,
};
pub use memory::{
    AgentMemoryScope, BoundedEpisodicMemory, EpisodeId, EpisodeInput, EpisodeProvenance,
    EpisodicRecord, MEMORY_SCHEMA_VERSION, MemoryError, MemoryOperation, MemoryPort,
    MemorySnapshot, OwnerId, RawTextPolicy, RetentionPolicy, SummaryCode,
};
pub use memory_views::{
    AmbientSurface, AugmentedState, MEMORY_SUMMARY_DIM, MemoryViewError, PrivateMemoryView,
    SurfaceAccess, SurfaceCell, SurfaceGrant, SurfaceId, SurfaceLimits, SurfaceProvenance,
    SurfaceWrite, physical_state_hash, records_digest,
};
pub use microstate::{
    BELIEF_COLLAPSE_TOLERANCE, Cell, MAX_DISCRETE_CARDINALITY, MAX_MICROSTATE_NAME_BYTES,
    MAX_MICROSTATE_VARIABLES, Microstate, MicrostateError, MicrostateVariable, MicrostateView,
    Truth, VariableKind,
};
pub use model::{DeterministicRng, GenerativeModel, PlanningContext, RngSource};
pub use motivation::{
    ActionCandidate, CandidateEvaluation, ConstraintViolation, DriveId, DriveReading, DriveSpec,
    DynamicUtility, EvaluationOutcome, ForbiddenActionRule, HardConstraint, HomeostaticState,
    MotivationError, ObjectiveTerm, SelectedAction, UtilityBreakdown, UtilityEvaluator,
    WeightedObjective,
};
pub use observability::{
    BoundedObservationStream, EntityId, EpistemicKind, EventSource, MAX_EVENT_VALUES,
    OBSERVABILITY_SCHEMA_VERSION, ObservabilityError, ObservationHub, ObservationSubscription,
    PublishStatus, RunId, SaturationLossMarker, ScientificTraceEvent, StreamMetrics,
    StreamRevision, VisualEntitySample, VisualFrame, VisualFrameRing,
};
pub use planner::{
    BudgetTracker, CancelToken, PlanBudget, PlanRequest, PlanResult, PlanStatus, PlanTarget,
    Planner,
};
pub use posterior_sampling::{
    PosteriorTable, SampledTableModel, TableKind, mix_seed, posterior_mean_table, sample_dirichlet,
    sample_ln_gamma, sample_posterior_table,
};
pub use procedural_memory::{
    EvaluationCriteria, EvaluationEvidence, PROCEDURAL_MEMORY_SCHEMA_VERSION, ProceduralError,
    ProceduralMemory, PromotionAuditEvent, PromotionDecision, PromotionGate, PromotionGateVerdict,
    RejectionReason, StrategyId, StrategyLifecycle, StrategyRecord, StrategyStats, StrategyVersion,
    TabularPolicy,
};
pub use provenance::{
    Confidence, EndowmentProvenance, EvidenceRef, FrameId, IngressStatus, IngressWindow, MessageId,
    ObservationEnvelope, ObservationKind, ObservationRecord, ProvenanceError, ProvenanceMetadata,
    ProvenanceTracker, SensorId, SensorWatermark, UnitTag,
};
pub use scale::{CanonicalScale, Scale, ScaleError, ScaleIdentity};
pub use scheduler::{
    EventKind, LifecycleTracker, LocalScheduler, ScheduleTrigger, ScheduledTask, SchedulerConfig,
    SchedulerError, SchedulerEvent, SchedulerLifecycle, StepReport, TaskContext,
    TaskExecutionRecord, TaskHandler, TrackedResource,
};
pub use selection_principle::SelectionPrinciple;
pub use semantic_memory::{
    EntityKind, EpistemicStatus, EvidenceEntry, EvidenceInput, EvidencePolarity,
    GroundingCandidate, GroundingFilter, PredicateId, PropositionKey, PropositionRecord,
    PropositionValue, RevisionStatus, SemanticEntity, SemanticKnowledgeStore, SemanticMemoryError,
};
pub use snapshot::{
    BranchableSnapshot, SNAPSHOT_SCHEMA_VERSION, SimulationSnapshot, SnapshotEnvelope,
    SnapshotMetrics, Snapshotable, WorldSnapshot,
};
pub use system::{
    ComposedEffect, ComposedError, ComposedState, ComposedSystem, EffectSeverity, ProposedEffect,
    SystemContext, SystemError, SystemResult, SystemStep, TypedSystem,
};
pub use system_one::{
    FixtureBehavior, FixtureSystemOneProvider, HypothesisId, NoulVerdict, OptionId, QueryId,
    RawScores, ReplaySystemOneProvider, ResponseStatus, SystemOneError, SystemOneFixture,
    SystemOnePayload, SystemOneProvider, SystemOneResponse, TypedQuery, ValidatedDistribution,
};
pub use units::{Duration, SimDuration, StepDt, Tick};
pub use workflow::{
    CognitiveDecisionEvent, ErrorBranch, IdempotencyKey, ProposedWorkflowAction, RetryPolicy,
    StateMachineWorkflow, TimeoutBranch, TransitionGuard, WorkflowContext, WorkflowError,
    WorkflowId, WorkflowInput, WorkflowSpec, WorkflowStateId, WorkflowStepError,
    WorkflowStepOutcome, WorkflowStepStatus, WorkflowTransition, WorkflowTrigger,
};
pub use world::World;

/// Devuelve el nombre canónico del proyecto.
pub const fn project_name() -> &'static str {
    "gogamen"
}

/// Devuelve la versión del crate `gogamen-core`.
pub const fn core_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_smoke_returns_expected_identity() {
        assert_eq!(project_name(), "gogamen");
        assert!(!core_version().is_empty());
    }

    #[test]
    fn step_dt_rejects_invalid_values() {
        for invalid in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            -1e-6,
            -1.0,
        ] {
            assert!(
                matches!(
                    StepDt::from_secs_f64(invalid),
                    Err(DomainError::InvalidStepDt { .. })
                ),
                "expected InvalidStepDt for {invalid:?}"
            );
        }

        assert!(matches!(
            StepDt::from_millis_f64(0.0),
            Err(DomainError::InvalidStepDt { .. })
        ));
        assert!(matches!(
            StepDt::from_millis_f64(f64::NAN),
            Err(DomainError::InvalidStepDt { .. })
        ));
        assert!(matches!(
            StepDt::from_nanos(0),
            Err(DomainError::InvalidStepDt { .. })
        ));

        let valid = StepDt::from_millis_f64(20.0).expect("20 ms should be valid");
        assert!((valid.as_secs_f64() - 0.02).abs() < 1e-12);
    }

    #[test]
    fn sim_duration_validates_non_negative_finite_seconds() {
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.001] {
            assert!(matches!(
                SimDuration::from_secs_f64(invalid),
                Err(DomainError::InvalidDuration { .. })
            ));
        }

        let zero = SimDuration::from_secs_f64(-0.0).expect("-0.0 normalizes to 0.0");
        assert_eq!(zero.as_secs_f64(), 0.0);
        assert!(zero.as_secs_f64().is_sign_positive());

        let dt = StepDt::from_secs_f64(0.5).unwrap();
        let next = zero.checked_add_dt(dt).unwrap();
        assert_eq!(next.as_secs_f64(), 0.5);
    }

    #[test]
    fn state_and_observation_reject_non_finite_values() {
        let tick = Tick::ZERO;
        let elapsed = SimDuration::ZERO;

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                State::new(tick, elapsed, vec![1.0, bad]),
                Err(DomainError::NonFiniteValue {
                    field: "state.values",
                    ..
                })
            ));
            assert!(matches!(
                Observation::new(tick, elapsed, vec![bad]),
                Err(DomainError::NonFiniteValue {
                    field: "observation.features",
                    ..
                })
            ));
        }

        let state = State::new(tick, elapsed, vec![1.0, 2.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.25).unwrap();
        let next_state = state.advance(dt, vec![1.5, 2.5]).unwrap();
        assert_eq!(next_state.tick(), Tick::new(1));
        assert_eq!(next_state.elapsed().as_secs_f64(), 0.25);
        assert_eq!(next_state.values(), &[1.5, 2.5]);
        // El estado original permanece inmutable.
        assert_eq!(state.tick(), Tick::ZERO);
        assert_eq!(state.values(), &[1.0, 2.0]);
    }

    #[test]
    fn actions_reject_nan_and_out_of_range() {
        assert!(matches!(
            ActionSpace::discrete(0),
            Err(DomainError::EmptyActionSpace)
        ));
        assert!(matches!(
            ActionSpace::continuous(vec![]),
            Err(DomainError::EmptyActionSpace)
        ));
        assert!(matches!(
            ActionBounds::new(1.0, -1.0),
            Err(DomainError::InvalidActionBounds { .. })
        ));
        assert!(matches!(
            ActionBounds::new(f64::NAN, 1.0),
            Err(DomainError::NonFiniteValue { .. })
        ));

        let discrete_space = ActionSpace::discrete(3).unwrap();
        assert!(Action::discrete_in_space(2, &discrete_space).is_ok());
        assert!(matches!(
            Action::discrete_in_space(3, &discrete_space),
            Err(DomainError::DiscreteActionOutOfRange {
                index: 3,
                cardinality: 3
            })
        ));
        assert!(matches!(
            DiscreteAction::new(5, 3),
            Err(DomainError::DiscreteActionOutOfRange {
                index: 5,
                cardinality: 3
            })
        ));

        let continuous_space = ActionSpace::continuous(vec![
            ActionBounds::new(-1.0, 1.0).unwrap(),
            ActionBounds::new(0.0, 10.0).unwrap(),
        ])
        .unwrap();

        assert!(matches!(
            Action::continuous(vec![f64::NAN, 1.0]),
            Err(DomainError::NonFiniteValue {
                field: "action.continuous",
                ..
            })
        ));
        assert!(matches!(
            Action::continuous_in_space(vec![1.5, 5.0], &continuous_space),
            Err(DomainError::ContinuousActionOutOfRange { dimension: 0, .. })
        ));
        assert!(matches!(
            Action::continuous_in_space(vec![0.0, -0.1], &continuous_space),
            Err(DomainError::ContinuousActionOutOfRange { dimension: 1, .. })
        ));
        assert!(matches!(
            Action::continuous_in_space(vec![0.0], &continuous_space),
            Err(DomainError::ActionDimensionMismatch {
                expected: 2,
                actual: 1
            })
        ));

        let discrete_act = Action::discrete_in_space(0, &discrete_space).unwrap();
        assert!(matches!(
            continuous_space.validate(&discrete_act),
            Err(DomainError::IncompatibleActionKind {
                expected: "continuous",
                actual: "discrete"
            })
        ));
    }

    struct DummyWorld {
        space: ActionSpace,
        state: State,
        terminal: Terminal,
    }

    impl DummyWorld {
        fn new() -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
                state: State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap(),
                terminal: Terminal::Ongoing,
            }
        }
    }

    impl World for DummyWorld {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn reset(&mut self) -> Result<State, DomainError> {
            self.state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0])?;
            self.terminal = Terminal::Ongoing;
            Ok(self.state.clone())
        }

        fn observe(&self, state: &State) -> Result<Observation, DomainError> {
            Observation::new(state.tick(), state.elapsed(), state.values().to_vec())
        }

        fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
            if self.terminal.is_terminal() {
                return Err(DomainError::AlreadyTerminal);
            }
            self.space.validate(action)?;

            let prev_tick = self.state.tick();
            let delta = match action.as_discrete().map(DiscreteAction::index) {
                Some(1) => 1.0,
                _ => 0.0,
            };
            let next_val = self.state.values()[0] + delta;
            let next_state = self.state.advance(dt, vec![next_val])?;
            let obs = self.observe(&next_state)?;
            let terminal = if next_val >= 1.0 {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else {
                Terminal::Ongoing
            };

            self.state = next_state.clone();
            self.terminal = terminal;

            Transition::new(prev_tick, dt, next_state, obs, terminal)
        }
    }

    #[test]
    fn world_enforces_terminal_and_action_space_semantics() {
        let mut world = DummyWorld::new();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let stay = Action::discrete_in_space(0, world.action_space()).unwrap();
        let move_goal = Action::discrete_in_space(1, world.action_space()).unwrap();

        let t1 = world.step(&stay, dt).unwrap();
        assert!(t1.terminal().is_ongoing());
        assert_eq!(t1.next_state().tick(), Tick::new(1));

        let t2 = world.step(&move_goal, dt).unwrap();
        assert_eq!(t2.terminal().reason(), Some(TerminalReason::GoalReached));

        assert!(matches!(
            world.step(&stay, dt),
            Err(DomainError::AlreadyTerminal)
        ));

        let reset_state = world.reset().unwrap();
        assert_eq!(reset_state.tick(), Tick::ZERO);
        assert!(world.step(&stay, dt).is_ok());
    }

    struct ToyGenerativeModel {
        space: ActionSpace,
    }

    impl ToyGenerativeModel {
        fn new() -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
            }
        }
    }

    impl GenerativeModel for ToyGenerativeModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            self.space
                .validate(action)
                .map_err(ModelError::InvalidAction)?;
            if state.values().len() != 1 {
                return Err(ModelError::InvalidState {
                    reason: "expected 1D scalar state",
                });
            }

            let step_delta = match action.as_discrete().map(DiscreteAction::index) {
                Some(1) => 1.0 + 0.1 * rng.next_f64(),
                _ => 0.0,
            };
            let next_val = state.values()[0] + step_delta;
            let next_state = state
                .advance(dt, vec![next_val])
                .map_err(ModelError::TransitionFailed)?;
            let obs = Observation::new(
                next_state.tick(),
                next_state.elapsed(),
                next_state.values().to_vec(),
            )
            .map_err(ModelError::TransitionFailed)?;
            let terminal = if next_val >= 2.0 {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else {
                Terminal::Ongoing
            };

            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    struct ToyLiveEnvironment {
        space: ActionSpace,
        tick: Tick,
        elapsed: SimDuration,
        position: f64,
        terminal: Terminal,
        dt: StepDt,
    }

    impl ToyLiveEnvironment {
        fn new(dt: StepDt) -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
                tick: Tick::ZERO,
                elapsed: SimDuration::ZERO,
                position: 0.0,
                terminal: Terminal::Ongoing,
                dt,
            }
        }
    }

    impl EnvironmentObserver for ToyLiveEnvironment {
        fn observe(&self) -> Result<Observation, EnvironmentError> {
            Observation::new(self.tick, self.elapsed, vec![self.position]).map_err(|_| {
                EnvironmentError::ObservationFailed {
                    reason: "non-finite sensor reading",
                }
            })
        }
    }

    impl Actuator for ToyLiveEnvironment {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
            if self.terminal.is_terminal() {
                return Err(EnvironmentError::AlreadyTerminal);
            }
            self.space
                .validate(action)
                .map_err(EnvironmentError::InvalidAction)?;

            let delta = match action.as_discrete().map(DiscreteAction::index) {
                Some(1) => 1.0,
                _ => 0.0,
            };
            self.tick = self
                .tick
                .next()
                .map_err(|_| EnvironmentError::ActuationFailed {
                    reason: "tick overflow",
                })?;
            self.elapsed = self.elapsed.checked_add_dt(self.dt).map_err(|_| {
                EnvironmentError::ActuationFailed {
                    reason: "elapsed overflow",
                }
            })?;
            self.position += delta;
            if self.position >= 2.0 {
                self.terminal = Terminal::Terminated(TerminalReason::GoalReached);
            }
            let obs = self.observe()?;
            Ok(ApplyOutcome::new(obs, self.terminal))
        }
    }

    impl LiveEnvironment for ToyLiveEnvironment {
        fn reset(&mut self) -> Result<Observation, EnvironmentError> {
            self.tick = Tick::ZERO;
            self.elapsed = SimDuration::ZERO;
            self.position = 0.0;
            self.terminal = Terminal::Ongoing;
            self.observe()
        }
    }

    #[test]
    fn generative_model_advances_state_without_mutating_origin_and_is_deterministic() {
        let model = ToyGenerativeModel::new();
        let origin = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.5]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let act = Action::discrete_in_space(1, model.action_space()).unwrap();

        let mut rng_a = DeterministicRng::new(12345);
        let mut rng_b = DeterministicRng::new(12345);

        let t_a = model.step(&origin, &act, dt, &mut rng_a).unwrap();
        let t_b = model.step(&origin, &act, dt, &mut rng_b).unwrap();

        // El estado de origen permanece intacto.
        assert_eq!(origin.tick(), Tick::ZERO);
        assert_eq!(origin.elapsed(), SimDuration::ZERO);
        assert_eq!(origin.values(), &[0.5]);

        // El estado sucesor avanza deterministamente.
        assert_eq!(t_a, t_b);
        assert_eq!(t_a.next_state().tick(), Tick::new(1));
        assert_eq!(t_a.next_state().elapsed().as_secs_f64(), 0.1);
        assert!(t_a.next_state().values()[0] >= 1.5);
        assert!(t_a.next_state().values()[0] < 1.6);
    }

    #[test]
    fn planning_context_isolates_model_from_live_environment_and_actuator() {
        fn simulated_rollout_score<M: GenerativeModel + ?Sized>(
            ctx: &mut PlanningContext<'_, M>,
            start: &State,
            action: &Action,
            dt: StepDt,
        ) -> Result<f64, ModelError> {
            let t = ctx.simulate_step(start, action, dt)?;
            Ok(t.next_state().values()[0])
        }

        let model = ToyGenerativeModel::new();
        let mut rng = DeterministicRng::new(7);
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.25).unwrap();
        let move_act = Action::discrete_in_space(1, ctx.model().action_space()).unwrap();

        let predicted = simulated_rollout_score(&mut ctx, &start, &move_act, dt).unwrap();
        assert!(predicted >= 1.0);
        assert_eq!(start.tick(), Tick::ZERO);

        // El entorno vivo mantiene su propio ciclo observe/apply/reset independiente.
        let mut live = ToyLiveEnvironment::new(dt);
        assert_eq!(live.observe().unwrap().features(), &[0.0]);
        let outcome = live.apply(&move_act).unwrap();
        assert_eq!(outcome.observation().features(), &[1.0]);
        assert!(outcome.terminal().is_ongoing());
        let outcome_2 = live.apply(&move_act).unwrap();
        assert_eq!(
            outcome_2.terminal().reason(),
            Some(TerminalReason::GoalReached)
        );
        assert!(matches!(
            live.apply(&move_act),
            Err(EnvironmentError::AlreadyTerminal)
        ));
        let reset_obs = live.reset().unwrap();
        assert_eq!(reset_obs.tick(), Tick::ZERO);
        assert_eq!(reset_obs.features(), &[0.0]);
    }

    #[test]
    fn model_and_environment_errors_are_distinct_and_report_context() {
        let model = ToyGenerativeModel::new();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let bad_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 1.0]).unwrap();
        let valid_act = Action::discrete_in_space(0, model.action_space()).unwrap();
        let mut rng = DeterministicRng::new(9);

        let model_err: ModelError = model
            .step(&bad_state, &valid_act, dt, &mut rng)
            .expect_err("2D state must be rejected by 1D toy model");
        assert!(matches!(model_err, ModelError::InvalidState { .. }));
        assert!(model_err.to_string().contains("model invalid state"));

        let mut live = ToyLiveEnvironment::new(dt);
        let invalid_continuous = Action::continuous(vec![1.0]).unwrap();
        let env_err: EnvironmentError = live
            .apply(&invalid_continuous)
            .expect_err("continuous action must be rejected by discrete actuator");
        assert!(matches!(
            env_err,
            EnvironmentError::InvalidAction(DomainError::IncompatibleActionKind { .. })
        ));
        assert!(env_err.to_string().contains("environment invalid action"));

        // Verificación de que ambos implementan std::error::Error de forma independiente.
        fn as_std_err(err: &dyn std::error::Error) -> String {
            err.to_string()
        }
        assert_ne!(as_std_err(&model_err), as_std_err(&env_err));
    }

    #[test]
    fn same_seed_and_config_produce_identical_trace_hash() {
        fn run_trace(seed_value: u64) -> (u64, String) {
            let seed = MasterSeed::new(seed_value);
            let metadata = DeterminismMetadata::current(seed);
            let mut hasher = TraceHasher::with_metadata(&metadata);
            let dt = StepDt::from_secs_f64(0.1).unwrap();
            let mut clock = LogicalClock::new(dt);
            let model = ToyGenerativeModel::new();
            let action = Action::discrete_in_space(1, model.action_space()).unwrap();
            let mut rng = seed.derive(StreamKey::walker(0, 0, 0));
            let mut state = State::new(clock.tick(), clock.elapsed(), vec![0.0]).unwrap();

            for _ in 0..5 {
                let transition = model.step(&state, &action, dt, &mut rng).unwrap();
                let (tick, elapsed) = clock.advance().unwrap();
                assert_eq!(transition.next_state().tick(), tick);
                hasher.write_tick(tick);
                hasher.write_duration(elapsed);
                hasher.write_transition(&transition);
                state = transition.next_state().clone();
            }

            (hasher.finish(), hasher.finish_hex())
        }

        let (h1, hex1) = run_trace(0xDEAD_BEEF_CAFE_BABE);
        let (h2, hex2) = run_trace(0xDEAD_BEEF_CAFE_BABE);
        let (h_other, _) = run_trace(0xDEAD_BEEF_CAFE_BABF);

        assert_eq!(h1, h2);
        assert_eq!(hex1, hex2);
        assert_eq!(hex1.len(), 16);
        assert_ne!(h1, h_other);
    }

    #[test]
    fn logical_clock_avoids_floating_point_drift() {
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut clock = LogicalClock::new(dt);
        for _ in 0..10_000 {
            clock.advance().unwrap();
        }
        assert_eq!(clock.tick(), Tick::new(10_000));
        assert_eq!(clock.elapsed().as_secs_f64(), 0.1 * 10_000.0);

        let mut jumped = DeterministicClock::new(dt);
        let (tick, elapsed) = jumped.advance_by(10_000).unwrap();
        assert_eq!(tick, clock.tick());
        assert_eq!(elapsed, clock.elapsed());

        let at_tick = LogicalClock::with_tick(dt, Tick::new(10_000)).unwrap();
        assert_eq!(at_tick.elapsed(), clock.elapsed());

        clock.reset();
        assert_eq!(clock.tick(), Tick::ZERO);
        assert_eq!(clock.elapsed(), SimDuration::ZERO);
    }

    #[test]
    fn reordering_independent_streams_does_not_alter_sequences() {
        let seed = Seed::new(123_456_789);
        let k_ep0 = StreamKey::episode(0);
        let k_ag0 = StreamKey::agent(0, 0);
        let k_wk0 = StreamKey::walker(0, 0, 0);
        let k_wk1 = StreamKey::walker(0, 0, 1);
        let k_ep1 = StreamKey::episode(1);

        // Separación de dominio: mismo índice numérico en distintos niveles no colisiona.
        assert_ne!(seed.derive_seed(k_ep0), seed.derive_seed(k_ag0));
        assert_ne!(seed.derive_seed(k_ag0), seed.derive_seed(k_wk0));
        assert_ne!(seed.derive_seed(k_ep0), seed.derive_seed(k_wk0));

        // Orden A: ep0 -> ag0 -> wk0 -> wk1 -> ep1
        let mut reg_a = seed.registry();
        let seq_a_ep0: Vec<u64> = (0..4).map(|_| reg_a.stream(k_ep0).next_u64()).collect();
        let seq_a_ag0: Vec<u64> = (0..4).map(|_| reg_a.stream(k_ag0).next_u64()).collect();
        let seq_a_wk0: Vec<u64> = (0..4).map(|_| reg_a.stream(k_wk0).next_u64()).collect();
        let seq_a_wk1: Vec<u64> = (0..4).map(|_| reg_a.stream(k_wk1).next_u64()).collect();
        let seq_a_ep1: Vec<u64> = (0..4).map(|_| reg_a.stream(k_ep1).next_u64()).collect();

        // Orden B (inverso e intercalado): ep1 -> wk1 -> wk0 -> ag0 -> ep0
        let mut reg_b = RngStreamRegistry::new(seed);
        let seq_b_ep1: Vec<u64> = (0..4).map(|_| reg_b.stream(k_ep1).next_u64()).collect();
        let seq_b_wk1: Vec<u64> = (0..4).map(|_| reg_b.stream(k_wk1).next_u64()).collect();
        let seq_b_wk0: Vec<u64> = (0..4).map(|_| reg_b.stream(k_wk0).next_u64()).collect();
        let seq_b_ag0: Vec<u64> = (0..4).map(|_| reg_b.stream(k_ag0).next_u64()).collect();
        let seq_b_ep0: Vec<u64> = (0..4).map(|_| reg_b.stream(k_ep0).next_u64()).collect();

        assert_eq!(seq_a_ep0, seq_b_ep0);
        assert_eq!(seq_a_ag0, seq_b_ag0);
        assert_eq!(seq_a_wk0, seq_b_wk0);
        assert_eq!(seq_a_wk1, seq_b_wk1);
        assert_eq!(seq_a_ep1, seq_b_ep1);

        // Derivación jerárquica desde DeterministicRng también es determinista e inmutable.
        let root_rng = DeterministicRng::new(seed.as_u64());
        let child_1 = root_rng.derive(k_wk0);
        let child_2 = root_rng.derive(k_wk0);
        assert_eq!(child_1, child_2);
        assert_eq!(root_rng.state(), seed.as_u64());
    }

    #[test]
    fn determinism_metadata_records_seed_build_platform_without_cross_platform_promise() {
        let seed = MasterSeed::new(42);
        let meta = DeterminismMetadata::current(seed);

        assert_eq!(meta.seed(), seed);
        assert_eq!(meta.build_version(), core_version());
        assert_eq!(meta.platform_os(), std::env::consts::OS);
        assert_eq!(meta.platform_arch(), std::env::consts::ARCH);
        assert_eq!(
            meta.platform(),
            format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
        );
        assert!(!meta.cross_platform_guaranteed());
        assert!(meta.to_string().contains("cross_platform=false"));

        let other_platform =
            DeterminismMetadata::new(seed, core_version(), "other-os", "other-arch");
        assert_ne!(
            TraceHasher::with_metadata(&meta).finish(),
            TraceHasher::with_metadata(&other_platform).finish()
        );
    }

    #[derive(Clone)]
    struct StochasticBranchingWorld {
        space: ActionSpace,
        state: State,
        rng: DeterministicRng,
        terminal: Terminal,
        initial_seed: u64,
    }

    impl StochasticBranchingWorld {
        fn new(seed: u64, dimensions: usize) -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
                state: State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0; dimensions]).unwrap(),
                rng: DeterministicRng::new(seed),
                terminal: Terminal::Ongoing,
                initial_seed: seed,
            }
        }
    }

    impl World for StochasticBranchingWorld {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn reset(&mut self) -> Result<State, DomainError> {
            self.state = State::new(
                Tick::ZERO,
                SimDuration::ZERO,
                vec![0.0; self.state.dimension()],
            )?;
            self.rng = DeterministicRng::new(self.initial_seed);
            self.terminal = Terminal::Ongoing;
            Ok(self.state.clone())
        }

        fn observe(&self, state: &State) -> Result<Observation, DomainError> {
            Observation::new(state.tick(), state.elapsed(), state.values().to_vec())
        }

        fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
            if self.terminal.is_terminal() {
                return Err(DomainError::AlreadyTerminal);
            }
            self.space.validate(action)?;

            let prev_tick = self.state.tick();
            let noise = self.rng.next_f64();
            let direction = match action.as_discrete().map(DiscreteAction::index) {
                Some(1) => 1.0,
                _ => -0.5,
            };
            let mut next_values = self.state.values().to_vec();
            next_values[0] += direction + 0.25 * noise;

            let next_state = self.state.advance(dt, next_values)?;
            let obs = self.observe(&next_state)?;
            let terminal = if next_state.values()[0] >= 3.0 {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else {
                Terminal::Ongoing
            };

            self.state = next_state.clone();
            self.terminal = terminal;

            Transition::new(prev_tick, dt, next_state, obs, terminal)
        }
    }

    impl BranchableSnapshot for StochasticBranchingWorld {
        fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
            Ok(SimulationSnapshot::new(
                self.state.clone(),
                &self.rng,
                self.terminal,
            ))
        }

        fn fork(&self) -> Result<Self, SnapshotError> {
            let snap = self.snapshot()?;
            snap.verify()?;
            let mut cloned = self.clone();
            cloned.restore(&snap)?;
            Ok(cloned)
        }

        fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
            snapshot.verify()?;
            self.state = snapshot.state().clone();
            self.rng = snapshot.rng();
            self.terminal = snapshot.terminal();
            Ok(())
        }
    }

    #[test]
    fn branchable_snapshot_forks_diverge_without_contamination_and_restore_reproduces_next_transition()
     {
        let mut origin = StochasticBranchingWorld::new(98765, 2);
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let act_stay = Action::discrete_in_space(0, origin.action_space()).unwrap();
        let act_advance = Action::discrete_in_space(1, origin.action_space()).unwrap();

        // Avanzamos un paso inicial antes de bifurcar.
        let _ = origin.step(&act_advance, dt).unwrap();
        let saved_snapshot = origin.snapshot().unwrap();

        let mut fork_a = origin.fork().unwrap();
        let mut fork_b = origin.fork().unwrap();

        // Fork A y Fork B divergen aplicando acciones distintas.
        let t_a1 = fork_a.step(&act_advance, dt).unwrap();
        let t_a2 = fork_a.step(&act_advance, dt).unwrap();
        let t_b1 = fork_b.step(&act_stay, dt).unwrap();

        assert_ne!(t_a1.next_state().values(), t_b1.next_state().values());
        assert_eq!(t_a2.next_state().tick(), Tick::new(3));
        assert_eq!(t_b1.next_state().tick(), Tick::new(2));

        // El origen no fue contaminado por las mutaciones en Fork A ni Fork B.
        assert_eq!(origin.state.tick(), Tick::new(1));
        assert_eq!(origin.state.values(), saved_snapshot.state().values());
        assert_eq!(origin.rng.state(), saved_snapshot.rng_state());

        // La próxima transición en `origin` antes y después de `restore` es idéntica.
        let expected_next = origin.step(&act_advance, dt).unwrap();
        origin.restore(&saved_snapshot).unwrap();
        let replayed_next = origin.step(&act_advance, dt).unwrap();
        assert_eq!(expected_next, replayed_next);

        // También `fork_a` restaurado al snapshot reproduce exactamente la misma transición.
        fork_a.restore(&saved_snapshot).unwrap();
        let fork_a_replayed = fork_a.step(&act_advance, dt).unwrap();
        assert_eq!(expected_next, fork_a_replayed);
    }

    #[test]
    fn snapshot_rejects_incomplete_or_corrupted_envelopes() {
        let mut world = StochasticBranchingWorld::new(42, 1);
        let dt = StepDt::from_secs_f64(0.2).unwrap();
        let act = Action::discrete_in_space(1, world.action_space()).unwrap();
        let _ = world.step(&act, dt).unwrap();
        let valid = world.snapshot().unwrap();
        let baseline_state = world.state.clone();

        // Ausencia de cada campo obligatorio debe rechazarse explícitamente.
        let mut missing_rng = valid.to_envelope();
        missing_rng.rng_state = None;
        assert!(matches!(
            world.restore_from_envelope(missing_rng),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "rng_state"
            })
        ));

        let mut missing_terminal = valid.to_envelope();
        missing_terminal.terminal = None;
        assert!(matches!(
            world.restore_from_envelope(missing_terminal),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "terminal"
            })
        ));

        let mut missing_state = valid.to_envelope();
        missing_state.state = None;
        assert!(matches!(
            world.restore_from_envelope(missing_state),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "state"
            })
        ));

        let mut missing_checksum = valid.to_envelope();
        missing_checksum.checksum = None;
        assert!(matches!(
            world.restore_from_envelope(missing_checksum),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "checksum"
            })
        ));

        let mut missing_tick = valid.to_envelope();
        missing_tick.tick = None;
        assert!(matches!(
            world.restore_from_envelope(missing_tick),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "tick"
            })
        ));

        let mut missing_elapsed = valid.to_envelope();
        missing_elapsed.elapsed = None;
        assert!(matches!(
            world.restore_from_envelope(missing_elapsed),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "elapsed"
            })
        ));

        let mut missing_version = valid.to_envelope();
        missing_version.version = None;
        assert!(matches!(
            world.restore_from_envelope(missing_version),
            Err(SnapshotError::IncompleteSnapshot {
                missing_field: "version"
            })
        ));

        // Checksum corrupto e inconsistencia de reloj también se rechazan.
        let mut corrupted = valid.to_envelope();
        corrupted.checksum = Some(valid.checksum() ^ 0xFFFF);
        assert!(matches!(
            world.restore_from_envelope(corrupted),
            Err(SnapshotError::CorruptedChecksum { .. })
        ));

        let mut bad_clock = valid.to_envelope();
        bad_clock.tick = Some(Tick::new(99));
        assert!(matches!(
            world.restore_from_envelope(bad_clock),
            Err(SnapshotError::InconsistentClock {
                state_tick: 1,
                snapshot_tick: 99
            })
        ));

        // Ninguno de los intentos fallidos debe mutar el estado del mundo.
        assert_eq!(world.state, baseline_state);
    }

    #[test]
    fn snapshot_metrics_measure_bytes_and_fork_latency() {
        let small_world = StochasticBranchingWorld::new(7, 2);
        let large_world = StochasticBranchingWorld::new(7, 64);

        let (_fork_small, metrics_small) = small_world.fork_with_metrics().unwrap();
        let (_fork_large, metrics_large) = large_world.fork_with_metrics().unwrap();

        assert!(metrics_small.bytes_per_snapshot() > 0);
        assert!(metrics_large.bytes_per_snapshot() > metrics_small.bytes_per_snapshot());
        assert_eq!(
            metrics_large.bytes_per_snapshot() - metrics_small.bytes_per_snapshot(),
            (64 - 2) * std::mem::size_of::<f64>()
        );
        // La latencia en nanosegundos se mide de forma no negativa mediante `Instant`.
        assert!(metrics_small.fork_latency_ns() < u128::MAX);
        assert!(metrics_small.to_string().contains("bytes_per_snapshot="));
        assert!(metrics_small.to_string().contains("fork_latency_ns="));
    }
}
