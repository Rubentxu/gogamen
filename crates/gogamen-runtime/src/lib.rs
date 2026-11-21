//! Runtime headless `observe -> plan -> guard -> act` de Gogamen (`std`-only).
//!
//! Compone el bucle de horizonte recedente sobre [`gogamen_core::LocalScheduler`],
//! registra una traza determinista por tick ([`ExecutionTrace`]) y proporciona un
//! mundo 1D reproducible ([`LineWorld1D`] / [`World1D`]).

pub mod channels;
pub mod linux_rt_profile;
pub mod multirate;
pub mod plan_runner;
pub mod runtime;
pub mod scientific_trace;
pub mod temporal_profile;
pub mod temporal_telemetry;
pub mod trace;
pub mod world1d;

pub use channels::{
    BoundedChannel, ChannelConfig, ChannelError, ChannelMessage, ChannelOverflowPolicy,
    ChannelStats, SendOutcome,
};
pub use linux_rt_profile::{
    KernelPreemptMode, LINUX_RT_NON_CERTIFICATION_NOTE, LinuxCpuInfo, LinuxHostProfile,
    LinuxHostSnapshot, LinuxKernelInfo, LinuxLatencyProbeConfig, LinuxLatencyProbeError,
    LinuxLatencyProbeReport, LinuxRtPermissions, LoadProfile, LoadProfileProbeResult, RtLimit,
    RtPlatformStatus, RtSchedulingStatus, run_linux_latency_probe,
    run_linux_latency_probe_with_snapshot,
};
pub use multirate::{
    BestEffortWorkerHandle, BoundedMailbox, ControlPeriod, ILLUSTRATIVE_PERIODS_NOTICE,
    MailboxStats, MonotonicClockRef, MultirateConfig, MultirateController, MultirateError,
    MultirateFallbackReason, MultirateTickRecord, MultirateTrace, SensorReading, Watchdog,
    WatchdogHealth, WatchdogStatus, WorkerProposal, WorkerReport,
};
pub use plan_runner::{
    ExternalKernel, ExternalNodeContext, KernelRegistry, KernelRegistryError, PlanInstance,
    PlanRunSummary, PlanRuntimeEngine, PlanRuntimeError, PortValue, builtin_plan_catalog,
};
pub use runtime::{HeadlessRuntime, ProposalMutator, RuntimeConfig, RuntimeError};
pub use scientific_trace::{
    AsyncTraceWriter, BufferedTraceSink, CheckpointRef, ControlBitsRecord, DecisionAction,
    EnqueueOutcome, FileTraceBackend, InMemoryTraceBackend, OverflowPolicy, RecordedGuardDecision,
    SCIENTIFIC_TRACE_VERSION, ScientificRecord, ScientificTraceArchive, ScientificTraceError,
    SelectionMeasurementRecord, SensitiveFieldMode, SensitiveFieldPolicy, StreamSeeds, TickInputs,
    TraceGap, TraceGapReason, TraceManifest, TraceManifestConfig, TraceQuota, TraceStorageBackend,
    TraceWriteError,
};
pub use temporal_profile::{
    CriticalBufferConfig, CriticalBufferLimits, CriticalOperationDescriptor, CriticalPathBuffers,
    CriticalPathLifecycle, CriticalPathPlan, DeclaredCapabilityClaim, MemoryBoundClaim,
    ObservedAllocationProof, PreallocatedRingBuffer, PreparedCriticalPath, RuntimeTemporalProfile,
    STRICT_REALTIME_NON_CERTIFICATION_NOTICE, TemporalProfileReport, TemporalValidationError,
    ValidatedCriticalPath, prepare_critical_path, validate_before_start,
};
pub use temporal_telemetry::{
    DEFAULT_HISTOGRAM_BOUNDS_NANOS, FallbackCounts, FixedLatencyHistogram, HISTOGRAM_BUCKET_COUNT,
    HistogramBucket, JitterSummary, NO_WCET_CLAIM_NOTICE, PlatformMetadata, RecordedTickTelemetry,
    TemporalComplianceReport, TemporalComplianceState, TemporalTelemetryConfig,
    TemporalTelemetryError, TemporalTelemetryRecorder, TemporalTransition,
};
pub use trace::{ExecutionTrace, RejectionCause, TickTraceEntry};
pub use world1d::{
    ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, LineWorld1D, LineWorld1DConfig, LineWorld1DModel,
    RecedingHorizonPlanner, World1D,
};

use std::sync::Arc;

use gogamen_core::{ActionGuard, MasterSeed, SafeFallbackPolicy, SystemMonotonicClock};

/// Resumen de una ejecución headless sobre [`LineWorld1D`].
#[derive(Debug, Clone, PartialEq)]
pub struct Run1DSummary {
    /// Número de ticks ejecutados.
    pub ticks_executed: usize,
    /// Semilla maestra utilizada.
    pub seed: MasterSeed,
    /// Número de acciones aplicadas sobre el entorno vivo.
    pub actions_applied: usize,
    /// Número de ticks que usaron respaldo seguro.
    pub fallbacks_used: usize,
    /// Número de ticks rechazados por la guardia.
    pub rejected_count: usize,
    /// Posición final en el eje 1D.
    pub final_position: f64,
    /// Traza completa de ejecución.
    pub trace: ExecutionTrace,
}

/// Ejecuta un episodio headless de `ticks` pasos en [`LineWorld1D`] con `seed`.
pub fn run_1d_episode(ticks: u64, seed: MasterSeed) -> Result<Run1DSummary, RuntimeError> {
    let world_config = LineWorld1DConfig::default();
    let env = LineWorld1D::new(world_config).map_err(|err| {
        RuntimeError::Environment(gogamen_core::EnvironmentError::InvalidAction(err))
    })?;
    let model = env.model();
    let fallback = env.safe_fallback_action();
    let planner = RecedingHorizonPlanner::from_config(&world_config, 5);
    let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(fallback));
    let clock = Arc::new(SystemMonotonicClock::new());
    let runtime_config = RuntimeConfig::new(seed, world_config.dt);

    let mut runtime = HeadlessRuntime::new(runtime_config, env, model, planner, guard, clock)?;
    runtime.run_ticks(ticks)?;

    let trace = runtime.trace_snapshot();
    let final_position = runtime.with_env(|e| e.position());

    Ok(Run1DSummary {
        ticks_executed: trace.len(),
        seed,
        actions_applied: trace.applied_count(),
        fallbacks_used: trace.fallback_count(),
        rejected_count: trace.rejected_count(),
        final_position,
        trace,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use gogamen_core::{
        Action, ActionSpace, DomainError, GenerativeModel, GuardDecision, GuardRejectionReason,
        ModelError, PlanRequest, PlanResult, PlanStatus, Planner, PlanningContext, Tick,
        VirtualMonotonicClock,
    };

    #[test]
    fn runtime_executes_100_ticks_in_world1d_with_single_action_per_tick() {
        let seed = MasterSeed::new(42);
        let world_config = LineWorld1DConfig::default();
        let env = LineWorld1D::new(world_config).unwrap();
        let model = env.model();
        let fallback = env.safe_fallback_action();
        let planner = RecedingHorizonPlanner::from_config(&world_config, 5);
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(fallback));
        let clock = Arc::new(VirtualMonotonicClock::new());
        let runtime_config = RuntimeConfig::new(seed, world_config.dt);

        let mut runtime =
            HeadlessRuntime::new(runtime_config, env, model, planner, guard, clock).unwrap();
        let reports = runtime.run_ticks(100).unwrap();

        assert_eq!(reports.len(), 100);
        // Comprueba composición real sobre LocalScheduler: observe -> plan -> guard -> act por tick.
        let first_tasks: Vec<&str> = reports[0]
            .executions
            .iter()
            .map(|r| r.task_name.as_str())
            .collect();
        assert_eq!(first_tasks, vec!["observe", "plan", "guard", "act"]);

        runtime.with_env(|e| {
            assert_eq!(e.apply_count(), 100);
            assert_eq!(e.tick(), Tick::new(100));
            assert!((e.position() - world_config.goal_pos).abs() < 1e-9);
        });

        runtime.with_trace(|trace| {
            assert_eq!(trace.len(), 100);
            assert_eq!(trace.seed(), seed);
            assert_eq!(trace.applied_count(), 100);
            assert_eq!(trace.fallback_count(), 0);
            assert_eq!(trace.rejected_count(), 0);
            for (idx, entry) in trace.entries().iter().enumerate() {
                assert_eq!(entry.tick, Tick::new((idx + 1) as u64));
                assert_eq!(entry.seed, seed);
                assert_eq!(entry.plan_status, PlanStatus::Completed);
                assert!(entry.guard_decision.is_allowed());
                assert!(entry.applied_action.is_some());
                assert!(!entry.used_fallback);
                assert!(entry.rejection_reason.is_none());
                assert!(entry.outcome.is_some());
            }
        });

        // Reproducibilidad determinista con la misma semilla.
        let summary_a = run_1d_episode(100, seed).unwrap();
        let summary_b = run_1d_episode(100, seed).unwrap();
        assert_eq!(summary_a.trace.trace_hash(), summary_b.trace.trace_hash());
        assert_eq!(summary_a.actions_applied, 100);
    }

    struct ClockAdvancingPlanner {
        clock: Arc<VirtualMonotonicClock>,
        advance_by: Duration,
        inner: RecedingHorizonPlanner,
    }

    impl Planner for ClockAdvancingPlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            ctx: &mut PlanningContext<'_, M>,
            request: &PlanRequest,
        ) -> Result<PlanResult, ModelError> {
            self.clock.advance(self.advance_by);
            self.inner.plan(ctx, request)
        }
    }

    #[test]
    fn expired_deadline_with_virtual_clock_uses_explicit_fallback_and_records_trace() {
        let seed = MasterSeed::new(77);
        let world_config = LineWorld1DConfig::default();
        let env = LineWorld1D::new(world_config).unwrap();
        let model = env.model();
        let fallback = env.safe_fallback_action();
        let clock = Arc::new(VirtualMonotonicClock::new());
        let planner = ClockAdvancingPlanner {
            clock: Arc::clone(&clock),
            advance_by: Duration::from_millis(25),
            inner: RecedingHorizonPlanner::from_config(&world_config, 5),
        };
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(fallback.clone()));

        let mut runtime_config = RuntimeConfig::new(seed, world_config.dt);
        runtime_config.planning_timeout = Duration::from_millis(10);

        let mut runtime =
            HeadlessRuntime::new(runtime_config, env, model, planner, guard, clock).unwrap();
        runtime.run_ticks(5).unwrap();

        runtime.with_trace(|trace| {
            assert_eq!(trace.len(), 5);
            assert_eq!(trace.fallback_count(), 5);
            assert_eq!(trace.applied_count(), 5);
            for entry in trace.entries() {
                assert_eq!(entry.seed, seed);
                assert_eq!(entry.plan_status, PlanStatus::BudgetExhausted);
                assert_eq!(
                    entry.guard_decision,
                    GuardDecision::Fallback(fallback.clone())
                );
                assert_eq!(entry.applied_action, Some(fallback.clone()));
                assert!(entry.used_fallback);
                assert_eq!(
                    entry.rejection_reason,
                    Some(RejectionCause::PlanBudgetExhausted)
                );
            }
        });

        // El agente permaneció en la posición inicial porque aplicó Stay en los 5 ticks.
        runtime.with_env(|e| {
            assert_eq!(e.apply_count(), 5);
            assert!((e.position() - world_config.initial_pos).abs() < 1e-12);
        });
    }

    struct FixedPlanPlanner {
        result: PlanResult,
    }

    impl Planner for FixedPlanPlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            _ctx: &mut PlanningContext<'_, M>,
            _request: &PlanRequest,
        ) -> Result<PlanResult, ModelError> {
            Ok(self.result.clone())
        }
    }

    #[test]
    fn trace_locates_decision_seed_and_rejection_reason_for_invalid_stale_and_missing_plan() {
        let seed = MasterSeed::new(999);
        let world_config = LineWorld1DConfig::default();

        // 1. Sin plan válido (NoValidAction) con guardia estricta -> Unauthorized.
        {
            let env = LineWorld1D::new(world_config).unwrap();
            let model = env.model();
            let planner = FixedPlanPlanner {
                result: PlanResult::no_valid_action(0),
            };
            let guard = ActionGuard::strict();
            let clock = Arc::new(VirtualMonotonicClock::new());
            let mut runtime = HeadlessRuntime::new(
                RuntimeConfig::new(seed, world_config.dt),
                env,
                model,
                planner,
                guard,
                clock,
            )
            .unwrap();

            runtime.run_ticks(1).unwrap();
            runtime.with_trace(|trace| {
                let entry = trace.get_tick(Tick::new(1)).expect("tick 1 must exist");
                assert_eq!(entry.seed, seed);
                assert_eq!(entry.plan_status, PlanStatus::NoValidAction);
                assert_eq!(
                    entry.guard_decision,
                    GuardDecision::Reject(GuardRejectionReason::Unauthorized)
                );
                assert!(entry.applied_action.is_none());
                assert!(!entry.used_fallback);
                assert_eq!(
                    entry.rejection_reason,
                    Some(RejectionCause::Guard(GuardRejectionReason::Unauthorized))
                );
            });
        }

        // 2. Propuesta inválida fuera de ActionSpace -> InvalidAction.
        {
            let env = LineWorld1D::new(world_config).unwrap();
            let model = env.model();
            let larger_space = ActionSpace::discrete(5).unwrap();
            let out_of_range = Action::discrete_in_space(4, &larger_space).unwrap();
            let planner = FixedPlanPlanner {
                result: PlanResult::completed(out_of_range.clone(), 1),
            };
            let guard = ActionGuard::strict();
            let clock = Arc::new(VirtualMonotonicClock::new());
            let mut runtime = HeadlessRuntime::new(
                RuntimeConfig::new(seed, world_config.dt),
                env,
                model,
                planner,
                guard,
                clock,
            )
            .unwrap();

            runtime.run_ticks(1).unwrap();
            runtime.with_trace(|trace| {
                let entry = trace.get_tick(Tick::new(1)).expect("tick 1 must exist");
                assert_eq!(entry.seed, seed);
                assert_eq!(entry.selected_action, Some(out_of_range));
                assert!(entry.applied_action.is_none());
                assert!(matches!(
                    entry.rejection_reason,
                    Some(RejectionCause::Guard(GuardRejectionReason::InvalidAction(
                        DomainError::DiscreteActionOutOfRange {
                            index: 4,
                            cardinality: 3
                        }
                    )))
                ));
            });
        }

        // 3. Propuesta caducada por edad en ticks -> StaleProposal.
        {
            let env = LineWorld1D::new(world_config).unwrap();
            let model = env.model();
            let planner = RecedingHorizonPlanner::from_config(&world_config, 3);
            let guard = ActionGuard::strict();
            let clock = Arc::new(VirtualMonotonicClock::new());
            let mut runtime = HeadlessRuntime::new(
                RuntimeConfig::new(seed, world_config.dt),
                env,
                model,
                planner,
                guard,
                clock,
            )
            .unwrap();

            runtime.set_proposal_mutator(Box::new(|_tick, _now, proposal| {
                let action = proposal
                    .authorized_action()
                    .expect("nominal planner produces action")
                    .clone();
                gogamen_core::ActionProposal::authorized(action, Tick::ZERO).with_max_age_ticks(0)
            }));

            runtime.run_ticks(1).unwrap();
            runtime.with_trace(|trace| {
                let entry = trace.get_tick(Tick::new(1)).expect("tick 1 must exist");
                assert_eq!(entry.seed, seed);
                assert!(entry.applied_action.is_none());
                assert_eq!(
                    entry.rejection_reason,
                    Some(RejectionCause::Guard(GuardRejectionReason::StaleProposal {
                        proposal_tick: Tick::ZERO,
                        current_tick: Tick::new(1),
                        max_age_ticks: 0,
                    }))
                );
            });
        }
    }
}
