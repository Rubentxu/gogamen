//! Suite reproducible de perfilado y optimización basada en evidencia (`std`-only).
//!
//! Mide explícitamente y por separado:
//! - Coste de simulación, snapshots y rollouts FMC (`simulation_nanos`,
//!   `snapshot_clone_nanos`, `rollout_nanos`, `throughput_steps_per_sec`).
//! - Coste de publicación y proyección 2D (`publication_nanos`, `projection_nanos`,
//!   `render_placeholder_nanos`).
//! - Percentiles `p50`, `p95`, `p99`, máximo observado y estimación de asignaciones.
//! - Identificación del cuello de botella medido ([`BottleneckStage`]).
//! - Comparación reproducible antes/después ([`BeforeAfterProfileComparison`]) con
//!   equivalencia determinista de oráculos (`oracle_equivalence_verified`) y política
//!   conservadora que mantiene el baseline ([`Decision::KeepBaseline`]) si no hay
//!   mejora demostrada en `p95` y throughput.
//! - Política explícita [`NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE`]: no se
//!   introduce GPU ni paralelismo (Rayon/threads/SIMD) sin beneficio medido demostrado.

use std::fmt::{self, Write as _};

use gogamen_core::{
    Action, Actuator, DeterministicRng, EnvironmentObserver, FmcPlanner, FmcPlannerConfig,
    FmcScoringConfig, FmcScratchWorkspace, GenerativeModel, MasterSeed, MonotonicClock, PlanBudget,
    PlanRequest, PlanningContext, SimDuration, SimulationSnapshot, State, StreamKey,
    SystemMonotonicClock, Tick, TraceHasher, Transition, WalkerPopulation, score_population,
    score_population_in_place,
};
use gogamen_lab_grid::{GridAction, GridMap, GridWorldConfig, GridWorldEnv, GridWorldModel};
use gogamen_projection::{ProjectionConfig, Projector2D, project_snapshot_to_scene};
use gogamen_runtime::{BoundedChannel, ChannelConfig, ChannelMessage, ChannelOverflowPolicy};

use crate::runner::ArenaError;

/// Aviso normativo de rendimiento: no se introduce GPU ni paralelismo sin evidencia medida.
pub const NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE: &str = "Política de rendimiento Gogamen: no se introduce GPU (wgpu) ni paralelismo (Rayon/threads/SIMD) sin beneficio medido y demostrado frente al baseline single-thread CPU.";

/// Etapa identificada como cuello de botella dominante en la ejecución perfilada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckStage {
    /// Observación y avance del entorno de simulación.
    Simulation,
    /// Captura, clonado y verificación de `SimulationSnapshot`.
    SnapshotClone,
    /// Evaluación de rollouts del planificador FMC.
    Rollout,
    /// Publicación y recepción en canal acotado sin bloqueo.
    Publication,
    /// Proyección pura 2D de snapshot hacia `Scene2D`.
    Projection,
}

impl BottleneckStage {
    /// Etiqueta estable de la etapa.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Simulation => "Simulation",
            Self::SnapshotClone => "SnapshotClone",
            Self::Rollout => "Rollout",
            Self::Publication => "Publication",
            Self::Projection => "Projection",
        }
    }
}

impl fmt::Display for BottleneckStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Política de ejecución CPU single-thread validada antes de ejecutar el perfilado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPolicy {
    /// Ejecución secuencial en CPU de un solo hilo.
    pub single_threaded_cpu: bool,
    /// Uso de GPU habilitado (requiere evidencia previa).
    pub gpu_enabled: bool,
    /// Uso de paralelismo (Rayon/threads/SIMD) habilitado (requiere evidencia previa).
    pub parallelism_enabled: bool,
    /// Indica si existe evidencia medida que justifique GPU.
    pub has_measured_gpu_evidence: bool,
    /// Indica si existe evidencia medida que justifique paralelismo.
    pub has_measured_parallel_evidence: bool,
    /// Aviso explícito de política sin GPU/paralelismo sin evidencia.
    pub notice: &'static str,
}

impl ExecutionPolicy {
    /// Política por defecto: CPU single-thread offline sin GPU ni paralelismo.
    pub const fn default_cpu_only() -> Self {
        Self {
            single_threaded_cpu: true,
            gpu_enabled: false,
            parallelism_enabled: false,
            has_measured_gpu_evidence: false,
            has_measured_parallel_evidence: false,
            notice: NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE,
        }
    }

    /// Valida que no se active GPU ni paralelismo sin evidencia medida demostrada.
    pub fn validate(&self) -> Result<(), ArenaError> {
        if self.gpu_enabled && !self.has_measured_gpu_evidence {
            return Err(ArenaError::InvalidBudget {
                reason: "gpu_enabled requires measured evidence",
            });
        }
        if self.parallelism_enabled && !self.has_measured_parallel_evidence {
            return Err(ArenaError::InvalidBudget {
                reason: "parallelism_enabled requires measured evidence",
            });
        }
        if !self.single_threaded_cpu && !self.gpu_enabled && !self.parallelism_enabled {
            return Err(ArenaError::InvalidBudget {
                reason: "execution policy must specify a valid execution mode",
            });
        }
        Ok(())
    }
}

impl Default for ExecutionPolicy {
    fn default() -> Self {
        Self::default_cpu_only()
    }
}

/// Estadísticas de latencia y asignaciones para una etapa individual.
#[derive(Debug, Clone, PartialEq)]
pub struct StageLatencyStats {
    /// Tiempo acumulado en nanosegundos.
    pub total_nanos: u64,
    /// Latencia media por paso en nanosegundos.
    pub mean_nanos: f64,
    /// Percentil 50 en nanosegundos.
    pub p50_nanos: u64,
    /// Percentil 95 en nanosegundos.
    pub p95_nanos: u64,
    /// Percentil 99 en nanosegundos.
    pub p99_nanos: u64,
    /// Máximo observado en nanosegundos.
    pub max_nanos: u64,
    /// Conteo/estimación de asignaciones en heap atribuidas a esta etapa.
    pub estimated_allocations: u64,
}

impl StageLatencyStats {
    /// Calcula percentiles y agregados deterministas a partir de las muestras en nanosegundos.
    pub fn from_samples(samples: &[u64], estimated_allocations: u64) -> Self {
        if samples.is_empty() {
            return Self {
                total_nanos: 0,
                mean_nanos: 0.0,
                p50_nanos: 0,
                p95_nanos: 0,
                p99_nanos: 0,
                max_nanos: 0,
                estimated_allocations,
            };
        }
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        let total_nanos = sorted.iter().copied().fold(0_u64, u64::saturating_add);
        let mean_nanos = total_nanos as f64 / sorted.len() as f64;
        Self {
            total_nanos,
            mean_nanos,
            p50_nanos: percentile_from_sorted(&sorted, 50, 100),
            p95_nanos: percentile_from_sorted(&sorted, 95, 100),
            p99_nanos: percentile_from_sorted(&sorted, 99, 100),
            max_nanos: *sorted.last().unwrap_or(&0),
            estimated_allocations,
        }
    }
}

/// Desglose completo de tiempos, percentiles, asignaciones y cuello de botella.
#[derive(Debug, Clone, PartialEq)]
pub struct StageBreakdown {
    /// Tiempo acumulado de simulación (`observe` + `apply`) en nanosegundos.
    pub simulation_nanos: u64,
    /// Tiempo acumulado de captura/clonado/verificación de snapshots en nanosegundos.
    pub snapshot_clone_nanos: u64,
    /// Tiempo acumulado de rollouts del planificador FMC en nanosegundos.
    pub rollout_nanos: u64,
    /// Tiempo acumulado de publicación en canal acotado en nanosegundos.
    pub publication_nanos: u64,
    /// Tiempo acumulado de proyección 2D (`project_snapshot`) en nanosegundos.
    pub projection_nanos: u64,
    /// Tiempo acumulado del placeholder de render headless en nanosegundos.
    pub render_placeholder_nanos: u64,
    /// Tiempo total acumulado de todas las etapas en nanosegundos.
    pub total_nanos: u64,
    /// Throughput efectivo en pasos por segundo.
    pub throughput_steps_per_sec: f64,
    /// Percentil 50 del coste total por paso en nanosegundos.
    pub p50_nanos: u64,
    /// Percentil 95 del coste total por paso en nanosegundos.
    pub p95_nanos: u64,
    /// Percentil 99 del coste total por paso en nanosegundos.
    pub p99_nanos: u64,
    /// Latencia máxima por paso en nanosegundos.
    pub max_nanos: u64,
    /// Conteo/estimación total de asignaciones en heap durante el perfilado.
    pub estimated_allocations: u64,
    /// Estadísticas detalladas de simulación.
    pub simulation_stats: StageLatencyStats,
    /// Estadísticas detalladas de clonado de snapshots.
    pub snapshot_clone_stats: StageLatencyStats,
    /// Estadísticas detalladas de rollouts FMC.
    pub rollout_stats: StageLatencyStats,
    /// Estadísticas detalladas de publicación.
    pub publication_stats: StageLatencyStats,
    /// Estadísticas detalladas de proyección 2D.
    pub projection_stats: StageLatencyStats,
    /// Estadísticas detalladas de render placeholder.
    pub render_placeholder_stats: StageLatencyStats,
    /// Cuello de botella dominante identificado sobre los costes medidos.
    pub bottleneck_stage: BottleneckStage,
}

impl StageBreakdown {
    /// Identifica el cuello de botella dominante entre las cinco etapas candidatas.
    pub fn identify_bottleneck(
        simulation_nanos: u64,
        snapshot_clone_nanos: u64,
        rollout_nanos: u64,
        publication_nanos: u64,
        projection_nanos: u64,
    ) -> BottleneckStage {
        let candidates = [
            (BottleneckStage::Simulation, simulation_nanos),
            (BottleneckStage::SnapshotClone, snapshot_clone_nanos),
            (BottleneckStage::Rollout, rollout_nanos),
            (BottleneckStage::Publication, publication_nanos),
            (BottleneckStage::Projection, projection_nanos),
        ];
        candidates
            .into_iter()
            .max_by_key(|(_, nanos)| *nanos)
            .map(|(stage, _)| stage)
            .unwrap_or(BottleneckStage::Simulation)
    }
}

/// Variante de ejecución perfilada para comparación antes/después.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileVariant {
    /// Baseline sin reutilización de buffers (asigna estructuras por iteración).
    Baseline,
    /// Micro-optimización basada en evidencia: reutiliza buffers preasignados.
    PreallocatedBufferReuse,
}

/// Configuración reproducible de la suite de perfilado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfilingConfig {
    /// Identificador de escenario.
    pub scenario_id: String,
    /// Número de pasos ejecutados por variante.
    pub steps: usize,
    /// Tamaño de población del planificador FMC.
    pub fmc_population_size: usize,
    /// Horizonte de rollout del planificador FMC.
    pub fmc_horizon: usize,
    /// Semilla determinista maestra.
    pub seed: MasterSeed,
    /// Capacidad del canal acotado de publicación.
    pub channel_capacity: usize,
    /// Política de ejecución CPU-only sin paralelismo/GPU sin evidencia.
    pub policy: ExecutionPolicy,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            scenario_id: "grid-fmc-projection-profile".to_owned(),
            steps: 32,
            fmc_population_size: 16,
            fmc_horizon: 6,
            seed: MasterSeed::new(42),
            channel_capacity: 8,
            policy: ExecutionPolicy::default_cpu_only(),
        }
    }
}

/// Reporte completo de una ejecución perfilada (antes o después).
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileReport {
    /// Variante ejecutada.
    pub variant: ProfileVariant,
    /// Número de pasos completados.
    pub steps_executed: usize,
    /// Desglose de tiempos, percentiles, asignaciones y cuello de botella.
    pub breakdown: StageBreakdown,
    /// Digest determinista de los oráculos funcionales (estados, acciones, escenas, render).
    pub oracle_digest: String,
    /// Valores finales del estado de simulación.
    pub final_state_values: Vec<f64>,
    /// Tick final alcanzado.
    pub final_tick: u64,
    /// Conteo acumulado de nodos proyectados en `Scene2D`.
    pub total_scene_nodes: usize,
    /// Política validada durante la ejecución.
    pub policy: ExecutionPolicy,
}

/// Decisión explícita de la comparación antes/después.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    /// Mantener el baseline porque la optimización no mejoró el objetivo o rompió equivalencia.
    KeepBaseline,
    /// Aceptar la optimización porque mejoró `p95` y throughput manteniendo equivalencia.
    AcceptOptimization,
}

/// Resultado de estado de la comparación antes/después.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOutcome {
    /// El baseline se mantuvo como versión vigente.
    BaselineKept,
    /// La optimización fue aceptada con evidencia medida.
    OptimizationAccepted,
}

/// Política cuantitativa de aceptación para cambios de rendimiento.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptimizationAcceptancePolicy {
    /// Mejora mínima relativa requerida en `p95_nanos` (`(before - after) / before`).
    pub min_p95_improvement_ratio: f64,
    /// Mejora mínima relativa requerida en `throughput_steps_per_sec` (`(after - before) / before`).
    pub min_throughput_improvement_ratio: f64,
}

impl Default for OptimizationAcceptancePolicy {
    fn default() -> Self {
        Self {
            min_p95_improvement_ratio: 0.0,
            min_throughput_improvement_ratio: 0.0,
        }
    }
}

/// Comparación reproducible antes/después con veredicto basado en evidencia.
#[derive(Debug, Clone, PartialEq)]
pub struct BeforeAfterProfileComparison {
    /// Reporte baseline ("antes").
    pub baseline: ProfileReport,
    /// Reporte candidato optimizado ("después").
    pub optimized: ProfileReport,
    /// Verifica que todos los oráculos funcionales y deterministas son idénticos.
    pub oracle_equivalence_verified: bool,
    /// Decisión final de la política de aceptación.
    pub decision: Decision,
    /// Estado de resultado (`BaselineKept` vs `OptimizationAccepted`).
    pub outcome: ComparisonOutcome,
    /// Mejora relativa de `p95_nanos` (`> 0.0` indica reducción de latencia).
    pub p95_improvement_ratio: f64,
    /// Mejora relativa de `throughput_steps_per_sec` (`> 0.0` indica mayor throughput).
    pub throughput_improvement_ratio: f64,
    /// Reducción absoluta de asignaciones estimadas (`baseline - optimized`).
    pub allocation_reduction: i64,
    /// Cuello de botella medido en el baseline.
    pub bottleneck_before: BottleneckStage,
    /// Cuello de botella medido tras la optimización.
    pub bottleneck_after: BottleneckStage,
    /// Justificación legible de la decisión tomada.
    pub reason: String,
    /// Aviso explícito de política sin GPU ni paralelismo sin evidencia.
    pub policy_notice: &'static str,
}

impl BeforeAfterProfileComparison {
    /// Evalúa un reporte baseline frente a uno optimizado aplicando `acceptance_policy`.
    ///
    /// Si `oracle_equivalence_verified` es falso o la optimización no mejora `p95` y
    /// throughput según la política, devuelve [`Decision::KeepBaseline`] y
    /// [`ComparisonOutcome::BaselineKept`].
    pub fn evaluate(
        baseline: ProfileReport,
        optimized: ProfileReport,
        acceptance_policy: OptimizationAcceptancePolicy,
    ) -> Self {
        let oracle_equivalence_verified = baseline.oracle_digest == optimized.oracle_digest
            && baseline.final_state_values == optimized.final_state_values
            && baseline.final_tick == optimized.final_tick
            && baseline.total_scene_nodes == optimized.total_scene_nodes
            && baseline.steps_executed == optimized.steps_executed;

        let base_p95 = baseline.breakdown.p95_nanos;
        let opt_p95 = optimized.breakdown.p95_nanos;
        let p95_improvement_ratio = if base_p95 > 0 {
            (base_p95 as f64 - opt_p95 as f64) / base_p95 as f64
        } else {
            0.0
        };

        let base_tp = baseline.breakdown.throughput_steps_per_sec;
        let opt_tp = optimized.breakdown.throughput_steps_per_sec;
        let throughput_improvement_ratio = if base_tp > 0.0 {
            (opt_tp - base_tp) / base_tp
        } else {
            0.0
        };

        let allocation_reduction = baseline.breakdown.estimated_allocations as i64
            - optimized.breakdown.estimated_allocations as i64;

        let p95_improved = opt_p95 < base_p95
            && p95_improvement_ratio >= acceptance_policy.min_p95_improvement_ratio;
        let throughput_improved = opt_tp > base_tp
            && throughput_improvement_ratio >= acceptance_policy.min_throughput_improvement_ratio;

        let (decision, outcome, reason) = if !oracle_equivalence_verified {
            (
                Decision::KeepBaseline,
                ComparisonOutcome::BaselineKept,
                "equivalence check failed between baseline and optimized oracles".to_owned(),
            )
        } else if p95_improved && throughput_improved {
            (
                Decision::AcceptOptimization,
                ComparisonOutcome::OptimizationAccepted,
                format!(
                    "accepted: p95 improved by {:.2}% ({}ns -> {}ns) and throughput by {:.2}% ({:.1} -> {:.1} steps/s), allocations reduced by {}",
                    p95_improvement_ratio * 100.0,
                    base_p95,
                    opt_p95,
                    throughput_improvement_ratio * 100.0,
                    base_tp,
                    opt_tp,
                    allocation_reduction
                ),
            )
        } else {
            (
                Decision::KeepBaseline,
                ComparisonOutcome::BaselineKept,
                format!(
                    "baseline kept: target p95/throughput did not improve (p95: {}ns -> {}ns, throughput: {:.1} -> {:.1} steps/s)",
                    base_p95, opt_p95, base_tp, opt_tp
                ),
            )
        };

        let bottleneck_before = baseline.breakdown.bottleneck_stage;
        let bottleneck_after = optimized.breakdown.bottleneck_stage;

        Self {
            baseline,
            optimized,
            oracle_equivalence_verified,
            decision,
            outcome,
            p95_improvement_ratio,
            throughput_improvement_ratio,
            allocation_reduction,
            bottleneck_before,
            bottleneck_after,
            reason,
            policy_notice: NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE,
        }
    }

    /// Formatea un resumen compacto de la comparación antes/después.
    pub fn format_summary(&self) -> String {
        format!(
            "BeforeAfterComparison: outcome={:?} decision={:?} bottleneck={}->{} p95_delta={:.2}% throughput_delta={:.2}% alloc_reduction={} oracle_eq={} | {}",
            self.outcome,
            self.decision,
            self.bottleneck_before,
            self.bottleneck_after,
            self.p95_improvement_ratio * 100.0,
            self.throughput_improvement_ratio * 100.0,
            self.allocation_reduction,
            self.oracle_equivalence_verified,
            self.reason
        )
    }
}

/// Ejecuta la suite completa antes/después usando [`SystemMonotonicClock`].
pub fn run_before_after_profiling(
    config: &ProfilingConfig,
) -> Result<BeforeAfterProfileComparison, ArenaError> {
    let clock = SystemMonotonicClock::new();
    run_before_after_profiling_with_clock(config, &clock, OptimizationAcceptancePolicy::default())
}

/// Ejecuta la suite completa antes/después con un [`MonotonicClock`] y política explícitos.
pub fn run_before_after_profiling_with_clock<C: MonotonicClock>(
    config: &ProfilingConfig,
    clock: &C,
    acceptance_policy: OptimizationAcceptancePolicy,
) -> Result<BeforeAfterProfileComparison, ArenaError> {
    let baseline = run_profile_variant(config, ProfileVariant::Baseline, clock)?;
    let optimized = run_profile_variant(config, ProfileVariant::PreallocatedBufferReuse, clock)?;
    Ok(BeforeAfterProfileComparison::evaluate(
        baseline,
        optimized,
        acceptance_policy,
    ))
}

/// Ejecuta una variante individual de perfilado bajo `config` y `clock`.
pub fn run_profile_variant<C: MonotonicClock>(
    config: &ProfilingConfig,
    variant: ProfileVariant,
    clock: &C,
) -> Result<ProfileReport, ArenaError> {
    config.policy.validate()?;
    if config.steps == 0 || config.channel_capacity == 0 {
        return Err(ArenaError::InvalidBudget {
            reason: "profiling steps and channel_capacity must be greater than zero",
        });
    }

    let map = GridMap::from_ascii(&[
        "#########",
        "#S......#",
        "#..###..#",
        "#......G#",
        "#########",
    ])?;
    let world_config = GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    };

    let mut env = GridWorldEnv::new(world_config.clone())?;
    let mut model = GridWorldModel::new(world_config.clone(), config.seed)?;
    let action_space = Actuator::action_space(&env).clone();
    let fallback_action = GridAction::Stay.to_action(&action_space)?;

    let fmc_cfg = FmcPlannerConfig::new(
        config.fmc_population_size,
        config.fmc_horizon,
        world_config.dt,
        FmcScoringConfig::default(),
        config.seed,
    )
    .map_err(|_| ArenaError::InvalidBudget {
        reason: "invalid FmcPlannerConfig in profiling",
    })?;
    let fmc = FmcPlanner::new(fmc_cfg);
    let mut rng = config.seed.derive(StreamKey::agent(0, 0));

    let proj_config = ProjectionConfig::default();
    let preallocated_projector = Projector2D::new(proj_config.clone());
    let channel_cfg =
        ChannelConfig::try_new(config.channel_capacity, ChannelOverflowPolicy::Latest, None)
            .map_err(|_| ArenaError::InvalidBudget {
                reason: "invalid channel_capacity in profiling",
            })?;
    let mut preallocated_channel: BoundedChannel<SimulationSnapshot> = channel_cfg.prepare();
    let mut preallocated_render_buf = String::with_capacity(512);
    let mut preallocated_scratch_counts: Vec<usize> = Vec::with_capacity(8);

    let mut sim_samples = Vec::with_capacity(config.steps);
    let mut snap_samples = Vec::with_capacity(config.steps);
    let mut rollout_samples = Vec::with_capacity(config.steps);
    let mut pub_samples = Vec::with_capacity(config.steps);
    let mut proj_samples = Vec::with_capacity(config.steps);
    let mut render_samples = Vec::with_capacity(config.steps);
    let mut step_total_samples = Vec::with_capacity(config.steps);

    let mut sim_allocs = 0_u64;
    let mut snap_allocs = 0_u64;
    let mut rollout_allocs = 0_u64;
    let mut pub_allocs = 0_u64;
    let mut proj_allocs = 0_u64;
    let mut render_allocs = 0_u64;

    let mut hasher = TraceHasher::default();
    let mut total_scene_nodes = 0_usize;
    let step_budget = config
        .fmc_population_size
        .saturating_mul(config.fmc_horizon)
        .saturating_mul(2)
        .max(16);

    for step_idx in 0..config.steps {
        // 1. Simulación: observar estado
        let t0 = clock.now();
        let obs = EnvironmentObserver::observe(&env).map_err(|_| ArenaError::InvalidBudget {
            reason: "failed to observe GridWorldEnv during profiling",
        })?;
        let t1 = clock.now();

        // 2. Rollout FMC
        let r0 = clock.now();
        let request = PlanRequest::from_state(
            model.state().clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(step_budget),
        )
        .with_fallback(fallback_action.clone())?;
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let (plan_result, plan_trace) = fmc
            .plan_with_trace(&mut ctx, &request, profile_step_reward, profile_phi)
            .map_err(|err| ArenaError::Trace(err.to_string()))?;
        let selected_action = plan_trace
            .first_action()
            .cloned()
            .unwrap_or_else(|| fallback_action.clone());
        let r1 = clock.now();
        rollout_allocs += 2;

        // Aplicar acción en entorno y modelo (parte de simulación)
        let t2 = clock.now();
        let outcome = env
            .apply(&selected_action)
            .map_err(|err| ArenaError::Trace(err.to_string()))?;
        model
            .step_in_place(&selected_action)
            .map_err(|err| ArenaError::Trace(err.to_string()))?;
        let t3 = clock.now();
        let sim_ns = elapsed_nanos(t0, t1).saturating_add(elapsed_nanos(t2, t3));
        let rollout_ns = elapsed_nanos(r0, r1);
        sim_allocs += 1;

        // 3. Snapshot capture + clone + verify
        let s0 = clock.now();
        let snapshot = SimulationSnapshot::new(model.state().clone(), &rng, env.terminal());
        let cloned_snapshot = snapshot.clone();
        cloned_snapshot
            .verify()
            .map_err(|err| ArenaError::Trace(err.to_string()))?;
        let s1 = clock.now();
        let snap_ns = elapsed_nanos(s0, s1);
        snap_allocs += 2;

        // 4. Publicación en canal acotado
        let p0 = clock.now();
        let now_msg = clock.now();
        let msg = ChannelMessage::new(cloned_snapshot, now_msg, (step_idx + 1) as u64);
        let received_snapshot = match variant {
            ProfileVariant::Baseline => {
                let mut transient_channel: BoundedChannel<SimulationSnapshot> =
                    channel_cfg.prepare();
                let _ = transient_channel.try_send(msg, now_msg);
                pub_allocs += 2;
                transient_channel
                    .try_recv(now_msg)
                    .map(|m| m.payload)
                    .unwrap_or(snapshot)
            }
            ProfileVariant::PreallocatedBufferReuse => {
                let _ = preallocated_channel.try_send(msg, now_msg);
                preallocated_channel
                    .try_recv(now_msg)
                    .map(|m| m.payload)
                    .unwrap_or(snapshot)
            }
        };
        let p1 = clock.now();
        let pub_ns = elapsed_nanos(p0, p1);

        // 5. Proyección 2D
        let pr0 = clock.now();
        let scene = match variant {
            ProfileVariant::Baseline => {
                proj_allocs += 4;
                project_snapshot_to_scene(&received_snapshot, &proj_config)
                    .map_err(|err| ArenaError::Trace(err.to_string()))?
            }
            ProfileVariant::PreallocatedBufferReuse => {
                proj_allocs += 2;
                preallocated_projector
                    .project_snapshot(&received_snapshot)
                    .map_err(|err| ArenaError::Trace(err.to_string()))?
            }
        };
        let pr1 = clock.now();
        let proj_ns = elapsed_nanos(pr0, pr1);

        // 6. Render placeholder (serialización/formateo headless sin GPU)
        let rn0 = clock.now();
        let (step_nodes, render_digest_bytes): (usize, Vec<u8>) = match variant {
            ProfileVariant::Baseline => {
                let mut layer_counts = Vec::new();
                for layer in scene.layers() {
                    layer_counts.push(layer.nodes().len());
                }
                let nodes: usize = layer_counts.iter().sum();
                let mut buf = String::new();
                let _ = write!(
                    buf,
                    "tick={};nodes={};layers={:?};status={:?}",
                    scene.tick().as_u64(),
                    nodes,
                    layer_counts,
                    plan_result.status()
                );
                render_allocs += 3;
                (nodes, buf.into_bytes())
            }
            ProfileVariant::PreallocatedBufferReuse => {
                preallocated_scratch_counts.clear();
                for layer in scene.layers() {
                    preallocated_scratch_counts.push(layer.nodes().len());
                }
                let nodes: usize = preallocated_scratch_counts.iter().sum();
                preallocated_render_buf.clear();
                let _ = write!(
                    preallocated_render_buf,
                    "tick={};nodes={};layers={:?};status={:?}",
                    scene.tick().as_u64(),
                    nodes,
                    preallocated_scratch_counts,
                    plan_result.status()
                );
                (nodes, preallocated_render_buf.as_bytes().to_vec())
            }
        };
        let rn1 = clock.now();
        let render_ns = elapsed_nanos(rn0, rn1);

        total_scene_nodes += step_nodes;

        sim_samples.push(sim_ns);
        snap_samples.push(snap_ns);
        rollout_samples.push(rollout_ns);
        pub_samples.push(pub_ns);
        proj_samples.push(proj_ns);
        render_samples.push(render_ns);
        step_total_samples.push(
            sim_ns
                .saturating_add(snap_ns)
                .saturating_add(rollout_ns)
                .saturating_add(pub_ns)
                .saturating_add(proj_ns)
                .saturating_add(render_ns),
        );

        // Registrar oráculo determinista completo
        hasher.write_u64(step_idx as u64);
        hasher.write_observation(&obs);
        hasher.write_u64(match &selected_action {
            Action::Discrete(idx) => idx.index() as u64,
            Action::Continuous(vals) => vals.len() as u64,
        });
        hasher.write_u64(received_snapshot.checksum());
        hasher.write_u64(step_nodes as u64);
        hasher.write_bytes(&render_digest_bytes);

        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let steps_executed = step_total_samples.len();
    let simulation_stats = StageLatencyStats::from_samples(&sim_samples, sim_allocs);
    let snapshot_clone_stats = StageLatencyStats::from_samples(&snap_samples, snap_allocs);
    let rollout_stats = StageLatencyStats::from_samples(&rollout_samples, rollout_allocs);
    let publication_stats = StageLatencyStats::from_samples(&pub_samples, pub_allocs);
    let projection_stats = StageLatencyStats::from_samples(&proj_samples, proj_allocs);
    let render_placeholder_stats = StageLatencyStats::from_samples(&render_samples, render_allocs);
    let total_stats = StageLatencyStats::from_samples(
        &step_total_samples,
        sim_allocs + snap_allocs + rollout_allocs + pub_allocs + proj_allocs + render_allocs,
    );

    let total_nanos = total_stats.total_nanos;
    let throughput_steps_per_sec = if total_nanos > 0 {
        (steps_executed as f64) * 1_000_000_000.0 / (total_nanos as f64)
    } else {
        0.0
    };

    let bottleneck_stage = StageBreakdown::identify_bottleneck(
        simulation_stats.total_nanos,
        snapshot_clone_stats.total_nanos,
        rollout_stats.total_nanos,
        publication_stats.total_nanos,
        projection_stats.total_nanos,
    );

    let breakdown = StageBreakdown {
        simulation_nanos: simulation_stats.total_nanos,
        snapshot_clone_nanos: snapshot_clone_stats.total_nanos,
        rollout_nanos: rollout_stats.total_nanos,
        publication_nanos: publication_stats.total_nanos,
        projection_nanos: projection_stats.total_nanos,
        render_placeholder_nanos: render_placeholder_stats.total_nanos,
        total_nanos,
        throughput_steps_per_sec,
        p50_nanos: total_stats.p50_nanos,
        p95_nanos: total_stats.p95_nanos,
        p99_nanos: total_stats.p99_nanos,
        max_nanos: total_stats.max_nanos,
        estimated_allocations: total_stats.estimated_allocations,
        simulation_stats,
        snapshot_clone_stats,
        rollout_stats,
        publication_stats,
        projection_stats,
        render_placeholder_stats,
        bottleneck_stage,
    };

    let _ = EnvironmentObserver::observe(&env).map_err(|_| ArenaError::InvalidBudget {
        reason: "failed final observe during profiling",
    })?;

    Ok(ProfileReport {
        variant,
        steps_executed,
        breakdown,
        oracle_digest: hasher.finish_hex(),
        final_state_values: model.state().values().to_vec(),
        final_tick: env.tick().as_u64(),
        total_scene_nodes,
        policy: config.policy,
    })
}

fn elapsed_nanos(
    start: gogamen_core::MonotonicInstant,
    end: gogamen_core::MonotonicInstant,
) -> u64 {
    let diff = end.as_nanos().saturating_sub(start.as_nanos());
    diff.min(u64::MAX as u128) as u64
}

fn percentile_from_sorted(sorted: &[u64], numerator: usize, denominator: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let n = sorted.len();
    let rank = n.saturating_mul(numerator).div_ceil(denominator).max(1);
    sorted[(rank - 1).min(n - 1)]
}

/// Before/after comparison of the FMC scoring hot loop (GOG-125).
///
/// `legacy` is `score_population` with a `Vec`-returning `phi` (one `Vec` per walker per step);
/// `in_place` is `score_population_in_place` with a borrowed-slice `phi` and a reused
/// [`FmcScratchWorkspace`]. Allocation figures are analytic and deterministic (no clock involved):
/// they count only the per-walker projection `Vec`s, the `O(N)` site eliminated by the change.
/// Measured allocator-level evidence lives in this module's tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoringHotLoopComparison {
    /// Number of walkers `N`.
    pub population_size: usize,
    /// Number of scored steps.
    pub steps: usize,
    /// Digest (hex) over distances, `D`, `R` and `VR` bit patterns of the legacy path.
    pub legacy_digest: String,
    /// Same digest for the in-place path.
    pub in_place_digest: String,
    /// `true` iff both digests are equal (bit-for-bit oracle equivalence).
    pub oracle_equivalent: bool,
    /// Per-walker projection allocations in the legacy path: `steps * N`.
    pub legacy_projection_allocations: u64,
    /// Per-walker projection allocations in the in-place path: always `0`.
    pub in_place_projection_allocations: u64,
}

/// Runs both scoring paths on identical populations and RNG streams and compares oracles.
pub fn run_scoring_hot_loop_comparison(
    population_size: usize,
    steps: usize,
    seed: MasterSeed,
) -> Result<ScoringHotLoopComparison, ArenaError> {
    if population_size < 2 || steps == 0 {
        return Err(ArenaError::InvalidBudget {
            reason: "scoring hot loop comparison needs population_size >= 2 and steps > 0",
        });
    }
    let population = synthetic_scoring_population(population_size, seed)?;
    let config = FmcScoringConfig::default();
    let scoring_err = |_| ArenaError::InvalidBudget {
        reason: "scoring failed in hot loop comparison",
    };

    let mut legacy_hasher = TraceHasher::default();
    let mut rng = DeterministicRng::new(0xC0FFEE ^ seed.as_u64());
    for _ in 0..steps {
        let score = score_population(&population, |s| s.values().to_vec(), &config, &mut rng)
            .map_err(scoring_err)?;
        for pair in score.companions() {
            legacy_hasher.write_u64(pair.distance().to_bits());
        }
        write_f64_bits(&mut legacy_hasher, score.relativized_distances());
        write_f64_bits(&mut legacy_hasher, score.relativized_rewards());
        write_f64_bits(&mut legacy_hasher, score.virtual_rewards());
    }

    let mut in_place_hasher = TraceHasher::default();
    let mut rng = DeterministicRng::new(0xC0FFEE ^ seed.as_u64());
    let mut ws = FmcScratchWorkspace::new();
    for _ in 0..steps {
        score_population_in_place(&population, |s| s.values(), &config, &mut rng, &mut ws)
            .map_err(scoring_err)?;
        write_f64_bits(&mut in_place_hasher, ws.distances());
        write_f64_bits(&mut in_place_hasher, ws.relativized_distances());
        write_f64_bits(&mut in_place_hasher, ws.relativized_rewards());
        write_f64_bits(&mut in_place_hasher, ws.virtual_rewards());
    }

    let legacy_digest = legacy_hasher.finish_hex();
    let in_place_digest = in_place_hasher.finish_hex();
    Ok(ScoringHotLoopComparison {
        population_size,
        steps,
        oracle_equivalent: legacy_digest == in_place_digest,
        legacy_digest,
        in_place_digest,
        legacy_projection_allocations: (steps as u64) * (population_size as u64),
        in_place_projection_allocations: 0,
    })
}

fn write_f64_bits(hasher: &mut TraceHasher, values: &[f64]) {
    for value in values {
        hasher.write_u64(value.to_bits());
    }
}

fn synthetic_scoring_population(
    population_size: usize,
    seed: MasterSeed,
) -> Result<WalkerPopulation, ArenaError> {
    let states = (0..population_size)
        .map(|i| {
            let x = i as f64;
            State::new(
                Tick::ZERO,
                SimDuration::ZERO,
                vec![x.sin() * 3.0, (x * 0.37).cos(), x * 0.1, (x * x) % 7.0],
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| ArenaError::InvalidBudget {
            reason: "failed to build synthetic scoring states",
        })?;
    WalkerPopulation::from_states(states, seed, 0, 0, false).map_err(|_| {
        ArenaError::InvalidBudget {
            reason: "failed to build synthetic scoring population",
        }
    })
}

fn profile_step_reward(_state: &State, _action: &Action, transition: &Transition) -> f64 {
    if transition.terminal().is_terminal() {
        10.0
    } else {
        -0.1
    }
}

fn profile_phi(state: &State) -> Vec<f64> {
    let vals = state.values();
    if vals.len() >= 2 {
        vec![-(vals[0].abs() + vals[1].abs()) * 0.05]
    } else {
        vec![0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_report(
        variant: ProfileVariant,
        p95_nanos: u64,
        throughput_steps_per_sec: f64,
        estimated_allocations: u64,
        oracle_digest: &str,
    ) -> ProfileReport {
        let stats = StageLatencyStats {
            total_nanos: p95_nanos * 10,
            mean_nanos: p95_nanos as f64,
            p50_nanos: p95_nanos / 2,
            p95_nanos,
            p99_nanos: p95_nanos + 100,
            max_nanos: p95_nanos + 200,
            estimated_allocations,
        };
        let breakdown = StageBreakdown {
            simulation_nanos: 1_000,
            snapshot_clone_nanos: 500,
            rollout_nanos: p95_nanos * 5,
            publication_nanos: 300,
            projection_nanos: 800,
            render_placeholder_nanos: 400,
            total_nanos: p95_nanos * 10,
            throughput_steps_per_sec,
            p50_nanos: p95_nanos / 2,
            p95_nanos,
            p99_nanos: p95_nanos + 100,
            max_nanos: p95_nanos + 200,
            estimated_allocations,
            simulation_stats: stats.clone(),
            snapshot_clone_stats: stats.clone(),
            rollout_stats: stats.clone(),
            publication_stats: stats.clone(),
            projection_stats: stats.clone(),
            render_placeholder_stats: stats,
            bottleneck_stage: BottleneckStage::Rollout,
        };
        ProfileReport {
            variant,
            steps_executed: 10,
            breakdown,
            oracle_digest: oracle_digest.to_owned(),
            final_state_values: vec![1.0, 2.0],
            final_tick: 10,
            total_scene_nodes: 40,
            policy: ExecutionPolicy::default_cpu_only(),
        }
    }

    #[test]
    fn ac1_measures_stages_separately_identifies_bottleneck_and_verifies_oracle_equivalence() {
        let cfg = ProfilingConfig {
            steps: 12,
            fmc_population_size: 8,
            fmc_horizon: 4,
            ..ProfilingConfig::default()
        };
        let comparison = run_before_after_profiling(&cfg).unwrap();

        // Medición separada de todas las etapas y percentiles ordenados.
        for report in [&comparison.baseline, &comparison.optimized] {
            let b = &report.breakdown;
            assert!(b.total_nanos > 0);
            assert!(b.throughput_steps_per_sec > 0.0);
            assert!(b.p50_nanos <= b.p95_nanos);
            assert!(b.p95_nanos <= b.p99_nanos);
            assert!(b.p99_nanos <= b.max_nanos);
            assert!(b.estimated_allocations > 0);
            assert!(
                b.simulation_nanos
                    + b.snapshot_clone_nanos
                    + b.rollout_nanos
                    + b.publication_nanos
                    + b.projection_nanos
                    + b.render_placeholder_nanos
                    > 0
            );
        }

        // Identificación determinista del cuello de botella.
        assert_eq!(
            StageBreakdown::identify_bottleneck(10, 20, 500, 15, 30),
            BottleneckStage::Rollout
        );
        assert_eq!(
            StageBreakdown::identify_bottleneck(10, 20, 30, 15, 900),
            BottleneckStage::Projection
        );

        // Equivalencia exacta de oráculos entre baseline y variante optimizada.
        assert!(comparison.oracle_equivalence_verified);
        assert_eq!(
            comparison.baseline.oracle_digest,
            comparison.optimized.oracle_digest
        );
        assert_eq!(
            comparison.baseline.final_state_values,
            comparison.optimized.final_state_values
        );
        assert!(comparison.allocation_reduction > 0);
        assert!(!comparison.format_summary().is_empty());
    }

    #[test]
    fn ac2_before_after_accepts_improvement_and_keeps_baseline_when_not_improved() {
        let baseline = synthetic_report(ProfileVariant::Baseline, 10_000, 1_000.0, 200, "abc123");

        // Caso 1: mejora real en p95, throughput y asignaciones con oráculo idéntico -> OptimizationAccepted.
        let better = synthetic_report(
            ProfileVariant::PreallocatedBufferReuse,
            7_500,
            1_350.0,
            80,
            "abc123",
        );
        let accepted = BeforeAfterProfileComparison::evaluate(
            baseline.clone(),
            better,
            OptimizationAcceptancePolicy::default(),
        );
        assert_eq!(accepted.decision, Decision::AcceptOptimization);
        assert_eq!(accepted.outcome, ComparisonOutcome::OptimizationAccepted);
        assert!(accepted.oracle_equivalence_verified);
        assert!(accepted.p95_improvement_ratio > 0.2);
        assert!(accepted.throughput_improvement_ratio > 0.3);
        assert_eq!(accepted.allocation_reduction, 120);

        // Caso 2: no mejora p95 ni throughput -> KeepBaseline / BaselineKept.
        let worse = synthetic_report(
            ProfileVariant::PreallocatedBufferReuse,
            11_000,
            950.0,
            120,
            "abc123",
        );
        let kept = BeforeAfterProfileComparison::evaluate(
            baseline.clone(),
            worse,
            OptimizationAcceptancePolicy::default(),
        );
        assert_eq!(kept.decision, Decision::KeepBaseline);
        assert_eq!(kept.outcome, ComparisonOutcome::BaselineKept);
        assert!(kept.oracle_equivalence_verified);

        // Caso 3: mejora tiempos pero rompe equivalencia de oráculo -> KeepBaseline.
        let diverged = synthetic_report(
            ProfileVariant::PreallocatedBufferReuse,
            5_000,
            2_000.0,
            50,
            "different-oracle",
        );
        let rejected_oracle = BeforeAfterProfileComparison::evaluate(
            baseline,
            diverged,
            OptimizationAcceptancePolicy::default(),
        );
        assert!(!rejected_oracle.oracle_equivalence_verified);
        assert_eq!(rejected_oracle.decision, Decision::KeepBaseline);
        assert_eq!(rejected_oracle.outcome, ComparisonOutcome::BaselineKept);
    }

    #[test]
    fn ac3_enforces_no_gpu_or_parallelism_without_measured_evidence() {
        let default_policy = ExecutionPolicy::default_cpu_only();
        assert!(default_policy.single_threaded_cpu);
        assert!(!default_policy.gpu_enabled);
        assert!(!default_policy.parallelism_enabled);
        assert_eq!(
            default_policy.notice,
            NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE
        );
        default_policy.validate().unwrap();

        let invalid_gpu = ExecutionPolicy {
            gpu_enabled: true,
            ..default_policy
        };
        assert!(invalid_gpu.validate().is_err());

        let invalid_parallel = ExecutionPolicy {
            parallelism_enabled: true,
            ..default_policy
        };
        assert!(invalid_parallel.validate().is_err());

        let cfg = ProfilingConfig {
            policy: invalid_parallel,
            ..ProfilingConfig::default()
        };
        assert!(run_before_after_profiling(&cfg).is_err());
    }

    mod alloc_counter {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static ALLOCS: Cell<u64> = const { Cell::new(0) };
        }

        pub struct CountingAllocator;

        // SAFETY: delegates to `System`; the thread-local counter is const-initialised and has
        // no destructor, so touching it from the allocator never allocates.
        unsafe impl GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
                unsafe { System.alloc(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }
            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }

        pub fn count() -> u64 {
            ALLOCS.with(Cell::get)
        }
    }

    #[global_allocator]
    static COUNTING: alloc_counter::CountingAllocator = alloc_counter::CountingAllocator;

    fn measure<R>(f: impl FnOnce() -> R) -> (u64, R) {
        let before = alloc_counter::count();
        let out = f();
        (alloc_counter::count() - before, out)
    }

    #[test]
    fn gog125_ac3_scoring_hot_loop_reduces_allocations_and_keeps_oracles_bit_identical() {
        let seed = MasterSeed::new(2024);
        let steps = 5;
        let mut per_step_legacy = Vec::new();
        for n in [16usize, 64] {
            let cmp = run_scoring_hot_loop_comparison(n, steps, seed).unwrap();
            assert!(cmp.oracle_equivalent, "{cmp:?}");
            assert_eq!(cmp.legacy_digest, cmp.in_place_digest);
            assert_eq!(cmp.legacy_projection_allocations, (steps * n) as u64);
            assert_eq!(cmp.in_place_projection_allocations, 0);

            // Measured with a counting allocator (thread-local, deterministic).
            let population = synthetic_scoring_population(n, seed).unwrap();
            let config = FmcScoringConfig::default();
            let mut rng = DeterministicRng::new(1);
            let (legacy, _) = measure(|| {
                for _ in 0..steps {
                    let _ =
                        score_population(&population, |s| s.values().to_vec(), &config, &mut rng)
                            .unwrap();
                }
            });
            let mut ws = FmcScratchWorkspace::new();
            score_population_in_place(&population, |s| s.values(), &config, &mut rng, &mut ws)
                .unwrap(); // warm-up sizes the buffers
            let (in_place, _) = measure(|| {
                for _ in 0..steps {
                    score_population_in_place(
                        &population,
                        |s| s.values(),
                        &config,
                        &mut rng,
                        &mut ws,
                    )
                    .unwrap();
                }
            });
            eprintln!(
                "gog125 N={n} steps={steps}: legacy_allocs={legacy} in_place_allocs={in_place}"
            );
            assert_eq!(in_place, 0, "warm in-place scoring must not allocate");
            assert!(
                legacy >= (steps * n) as u64,
                "legacy allocates >= N per step"
            );
            per_step_legacy.push(legacy / steps as u64);
        }
        // Legacy cost grows with N (O(N)); in-place stays at zero (O(1)).
        assert!(per_step_legacy[1] > per_step_legacy[0]);
    }
}
