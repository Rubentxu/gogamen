//! Interfaz anytime incremental de Fractal Monte Carlo (FMC) con checkpoints acotados (`std`-only).
//!
//! Permite ejecutar la planificación FMC por fragmentos acotados (`max_chunk_steps`)
//! manteniendo el estado completo de la búsqueda entre invocaciones:
//! - [`IncrementalFmcSession`]: conserva la población [`WalkerPopulation`], el paso de
//!   horizonte actual (`horizon_step`), el presupuesto exacto de transiciones
//!   ([`BudgetTracker`]), un plazo monotónico opcional ([`MonotonicDeadline`]) con
//!   margen de guardia (`guard_margin`), y el mejor candidato admisible conocido
//!   ([`Option<Action>`]).
//! - [`IncrementalFmcSession::step_checkpoint`]: avanza como máximo `max_chunk_steps`
//!   pasos de horizonte, pausando limpiamente si expira `deadline - guard_margin` en
//!   [`MonotonicClock`], si se agota [`BudgetTracker`] o si se activa [`CancelToken`].
//! - Si un modelo o fixture costoso excede el tiempo asignado por paso o invade el
//!   margen de guardia, registra explícitamente el sobrepaso (`overrun = true`,
//!   `overrun_nanos`) sin prometer límites en modelos no acotados.
//! - [`IncrementalFmcSession::advance_root`]: warm-start inter-tick con reciclaje de horizonte
//!   recedente. Al avanzar el entorno vivo a `t+1`, reutiliza los walkers cuyo primer paso
//!   simulado coincide con la acción ejecutada y el estado observado, evitando re-simular los
//!   `H-1` pasos ya cubiertos.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use crate::action::{Action, ActionSpace};
use crate::clock::{MonotonicClock, MonotonicDeadline, MonotonicInstant};
use crate::determinism::StreamKey;
use crate::domain::{State, Transition};
use crate::error::{DomainError, ModelError};
use crate::fmc_planner::{
    FmcPlannerConfig, default_phi, default_reward_fn, mean_slice, sample_population_actions,
    select_winning_first_action,
};
use crate::fmc_scoring::{is_failure_terminal, resample_population, score_population};
use crate::fmc_walkers::{PopulationError, WalkerId, WalkerPopulation};
use crate::model::{DeterministicRng, GenerativeModel, PlanningContext};
use crate::planner::{BudgetTracker, CancelToken, PlanStatus};

/// Errores de inicialización de [`IncrementalFmcSession`].
#[derive(Debug, Clone, PartialEq)]
pub enum FmcIncrementalError {
    /// Fallo al inicializar la población de walkers FMC.
    Population(PopulationError),
    /// El candidato admisible inicial no es válido para el [`ActionSpace`] configurado.
    InvalidInitialCandidate(DomainError),
}

impl fmt::Display for FmcIncrementalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Population(err) => write!(f, "failed to initialize incremental FMC: {err}"),
            Self::InvalidInitialCandidate(err) => {
                write!(f, "invalid initial admissible candidate: {err}")
            }
        }
    }
}

impl Error for FmcIncrementalError {}

/// Estado explícito devuelto por [`IncrementalFmcSession::step_checkpoint`].
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointStatus {
    /// Pausado limpiamente tras ejecutar el máximo de pasos asignados al chunk.
    PausedChunkLimit,
    /// Pausado limpiamente porque expiró `deadline - guard_margin` en el reloj monotónico.
    PausedDeadline,
    /// Se agotó el presupuesto exacto de transiciones en [`BudgetTracker`].
    BudgetExhausted,
    /// La sesión fue cancelada cooperativamente mediante [`CancelToken`].
    Cancelled,
    /// La planificación alcanzó el horizonte configurado o todos los walkers terminaron.
    Completed,
    /// No existe ninguna acción válida aplicable (p. ej. estado inicial terminal).
    NoValidAction,
    /// El modelo generativo o la función de recompensa produjo un error.
    Failed(ModelError),
}

impl CheckpointStatus {
    /// Indica si la sesión pausó de forma reanudable (`PausedChunkLimit` o `PausedDeadline`).
    pub const fn is_paused(&self) -> bool {
        matches!(self, Self::PausedChunkLimit | Self::PausedDeadline)
    }

    /// Indica si la sesión alcanzó un estado terminal definitivo.
    pub const fn is_finished(&self) -> bool {
        !self.is_paused()
    }

    /// Indica si la planificación finalizó con [`CheckpointStatus::Completed`].
    pub const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Indica si se agotó el presupuesto de transiciones.
    pub const fn is_budget_exhausted(&self) -> bool {
        matches!(self, Self::BudgetExhausted)
    }

    /// Indica si se pausó por expiración de `deadline - guard_margin`.
    pub const fn is_deadline_expired(&self) -> bool {
        matches!(self, Self::PausedDeadline)
    }

    /// Indica si la sesión fue cancelada.
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// Traduce este estado de checkpoint a [`PlanStatus`].
    pub const fn to_plan_status(&self) -> PlanStatus {
        match self {
            Self::Completed => PlanStatus::Completed,
            Self::PausedChunkLimit | Self::PausedDeadline | Self::BudgetExhausted => {
                PlanStatus::BudgetExhausted
            }
            Self::Cancelled => PlanStatus::Cancelled,
            Self::NoValidAction | Self::Failed(_) => PlanStatus::NoValidAction,
        }
    }
}

/// Resultado observable de una invocación a [`IncrementalFmcSession::step_checkpoint`].
#[derive(Debug, Clone, PartialEq)]
pub struct CheckpointOutcome {
    status: CheckpointStatus,
    best_candidate: Option<Action>,
    steps_executed_in_chunk: usize,
    horizon_step: usize,
    transitions_used: usize,
    cloning_events: usize,
    last_mean_vr: f64,
    overrun: bool,
    overrun_nanos: u128,
    elapsed_nanos: u128,
    warm_start_survivors: usize,
    warm_start_hit_ratio: f64,
}

impl CheckpointOutcome {
    /// Estado de parada o pausa del checkpoint.
    pub const fn status(&self) -> &CheckpointStatus {
        &self.status
    }

    /// Mejor candidato admisible conocido hasta este checkpoint, o `None` si no hay ninguno.
    pub const fn best_candidate(&self) -> Option<&Action> {
        self.best_candidate.as_ref()
    }

    /// Consume el resultado y devuelve el mejor candidato admisible conocido, si existe.
    pub fn into_best_candidate(self) -> Option<Action> {
        self.best_candidate
    }

    /// Pasos de horizonte ejecutados durante esta llamada a `step_checkpoint`.
    pub const fn steps_executed_in_chunk(&self) -> usize {
        self.steps_executed_in_chunk
    }

    /// Paso de horizonte acumulado tras este checkpoint.
    pub const fn horizon_step(&self) -> usize {
        self.horizon_step
    }

    /// Total acumulado de transiciones consumidas en [`BudgetTracker`].
    pub const fn transitions_used(&self) -> usize {
        self.transitions_used
    }

    /// Total acumulado de eventos de clonación durante el resampling FMC.
    pub const fn cloning_events(&self) -> usize {
        self.cloning_events
    }

    /// Media de recompensa virtual (`VR`) del último paso evaluado (`0.0` si no hubo evaluación).
    pub const fn last_mean_vr(&self) -> f64 {
        self.last_mean_vr
    }

    /// Indica si algún paso del chunk excedió el tiempo asignado o invadió `deadline - guard_margin`.
    pub const fn overrun(&self) -> bool {
        self.overrun
    }

    /// Máximo sobrepaso registrado en nanosegundos dentro de este chunk (`0` si `overrun == false`).
    pub const fn overrun_nanos(&self) -> u128 {
        self.overrun_nanos
    }

    /// Duración monotónica total consumida por esta llamada a `step_checkpoint` en nanosegundos.
    pub const fn elapsed_nanos(&self) -> u128 {
        self.elapsed_nanos
    }

    /// Walkers reciclados por el último [`IncrementalFmcSession::advance_root`] (`0` en cold-start).
    pub const fn warm_start_survivors(&self) -> usize {
        self.warm_start_survivors
    }

    /// Fracción `survivors / N` del último `advance_root` (`0.0` en cold-start o tras fallback).
    pub const fn warm_start_hit_ratio(&self) -> f64 {
        self.warm_start_hit_ratio
    }
}

/// Sesión incremental anytime de FMC con checkpoints acotados y observabilidad de sobrepasos.
#[derive(Debug, Clone)]
pub struct IncrementalFmcSession {
    config: FmcPlannerConfig,
    action_space: ActionSpace,
    population: WalkerPopulation,
    horizon_step: usize,
    transitions_budget: BudgetTracker,
    deadline: Option<MonotonicDeadline>,
    guard_margin: Duration,
    step_budget: Option<Duration>,
    best_admissible_candidate: Option<Action>,
    cancel_token: CancelToken,
    planner_rng: DeterministicRng,
    cloning_events: usize,
    last_mean_vr: f64,
    any_step_completed: bool,
    total_overruns: usize,
    max_overrun_nanos: u128,
    warm_start_survivors: usize,
    warm_start_hit_ratio: f64,
}

impl IncrementalFmcSession {
    /// Crea una nueva sesión incremental FMC a partir de `initial_state` y el presupuesto dado.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: FmcPlannerConfig,
        initial_state: State,
        action_space: ActionSpace,
        transitions_budget: BudgetTracker,
        deadline: Option<MonotonicDeadline>,
        guard_margin: Duration,
        best_admissible_candidate: Option<Action>,
        cancel_token: CancelToken,
    ) -> Result<Self, FmcIncrementalError> {
        if let Some(ref candidate) = best_admissible_candidate {
            action_space
                .validate(candidate)
                .map_err(FmcIncrementalError::InvalidInitialCandidate)?;
        }

        let episode_tag = initial_state.tick().as_u64();
        let population = WalkerPopulation::new(
            initial_state,
            config.population_size(),
            config.seed(),
            episode_tag,
            0,
            true,
        )
        .map_err(FmcIncrementalError::Population)?;

        let planner_rng = config.seed().derive(StreamKey::Agent {
            episode: episode_tag,
            agent: 0,
        });

        Ok(Self {
            config,
            action_space,
            population,
            horizon_step: 0,
            transitions_budget,
            deadline,
            guard_margin,
            step_budget: None,
            best_admissible_candidate,
            cancel_token,
            planner_rng,
            cloning_events: 0,
            last_mean_vr: 0.0,
            any_step_completed: false,
            total_overruns: 0,
            max_overrun_nanos: 0,
            warm_start_survivors: 0,
            warm_start_hit_ratio: 0.0,
        })
    }

    /// Advances the planning root to the live environment's next tick, recycling the horizon.
    ///
    /// A walker is *compatible* when it is viable (not in failure terminal), its `first_action`
    /// equals `executed_action`, it has simulated at least two steps, and
    /// `||step1_state - observed_state||_2 <= state_tolerance_l2`. The L2 distance is computed
    /// directly over [`State::values`] (the raw state vector, not `phi`); states with different
    /// dimensions or a non-finite distance/tolerance are never compatible.
    ///
    /// With `m >= 1` compatible walkers: incompatible walkers are replaced by synchronous clones
    /// of compatible ones (round-robin by index, deterministic), then every walker's root is
    /// shifted one step: the second action becomes `first_action` and the first step's reward
    /// (stored per walker in its prefix buffer) is subtracted from the cumulative reward, which
    /// is an undiscounted sum. `horizon_step` decreases by one, so only the missing final horizon
    /// step is simulated by the next `step_checkpoint`. Returns the number of survivors `m`.
    ///
    /// With no compatible walker (or `horizon_step == 0`), falls back to a clean cold start: a
    /// fresh population from `observed_state`, `horizon_step = 0`, survivors `0`, ratio `0.0`.
    /// The transition budget, cancel token, best candidate and planner RNG are never reset.
    pub fn advance_root(
        &mut self,
        executed_action: &Action,
        observed_state: &State,
        state_tolerance_l2: f64,
    ) -> usize {
        let n = self.population.len();
        let mut compatible: Vec<usize> = Vec::new();
        if self.horizon_step > 0 {
            for i in 0..n {
                let id = WalkerId::new(i);
                if is_failure_terminal(self.population.terminals()[i])
                    || self.population.first_actions()[i].as_ref() != Some(executed_action)
                    || self.population.prefix_len(id).unwrap_or(0) < 2
                {
                    continue;
                }
                let Ok(Some(step1)) = self.population.step1_state(id) else {
                    continue;
                };
                let dist = state_l2_distance(step1, observed_state);
                if dist.is_finite() && dist <= state_tolerance_l2 {
                    compatible.push(i);
                }
            }
        }

        if !compatible.is_empty() {
            let mut clones = Vec::new();
            let mut next_src = 0usize;
            for dst in 0..n {
                if compatible.binary_search(&dst).is_err() {
                    let src = compatible[next_src % compatible.len()];
                    next_src += 1;
                    clones.push((WalkerId::new(src), WalkerId::new(dst)));
                }
            }
            let salt = observed_state.tick().as_u64() ^ ((self.horizon_step as u64) << 32);
            if self
                .population
                .clone_walkers_synchronously(&clones, salt)
                .is_ok()
            {
                self.population.shift_root_one_step();
                self.horizon_step -= 1;
                self.warm_start_survivors = compatible.len();
                self.warm_start_hit_ratio = compatible.len() as f64 / n as f64;
                if let Some(winner) = select_winning_first_action(
                    &self.population,
                    self.config.action_selection(),
                    self.config.scoring(),
                    None,
                ) {
                    self.best_admissible_candidate = Some(winner.action);
                }
                return compatible.len();
            }
        }

        self.cold_restart(observed_state);
        0
    }

    fn cold_restart(&mut self, observed_state: &State) {
        let episode_tag = observed_state.tick().as_u64();
        // N and track_ancestry are unchanged from construction, so this cannot fail.
        if let Ok(population) = WalkerPopulation::new(
            observed_state.clone(),
            self.config.population_size(),
            self.config.seed(),
            episode_tag,
            0,
            true,
        ) {
            self.population = population;
        }
        self.horizon_step = 0;
        self.any_step_completed = false;
        self.warm_start_survivors = 0;
        self.warm_start_hit_ratio = 0.0;
    }

    /// Walkers recycled by the last [`Self::advance_root`] (`0` on cold start).
    pub const fn warm_start_survivors(&self) -> usize {
        self.warm_start_survivors
    }

    /// Ratio `survivors / N` of the last [`Self::advance_root`] (`0.0` on cold start).
    pub const fn warm_start_hit_ratio(&self) -> f64 {
        self.warm_start_hit_ratio
    }

    /// Configura un presupuesto temporal máximo esperado por paso de horizonte para detectar sobrepasos.
    pub const fn with_step_budget(mut self, step_budget: Duration) -> Self {
        self.step_budget = Some(step_budget);
        self
    }

    /// Devuelve la configuración del planificador FMC asociada.
    pub const fn config(&self) -> &FmcPlannerConfig {
        &self.config
    }

    /// Devuelve una referencia a la población de walkers actual.
    pub const fn population(&self) -> &WalkerPopulation {
        &self.population
    }

    /// Devuelve el paso de horizonte actual (`0..=config.horizon()`).
    pub const fn horizon_step(&self) -> usize {
        self.horizon_step
    }

    /// Devuelve el rastreador exacto de presupuesto de transiciones.
    pub const fn transitions_budget(&self) -> &BudgetTracker {
        &self.transitions_budget
    }

    /// Devuelve el deadline monotónico opcional de la sesión.
    pub const fn deadline(&self) -> Option<MonotonicDeadline> {
        self.deadline
    }

    /// Devuelve el margen de guardia reservado antes del deadline.
    pub const fn guard_margin(&self) -> Duration {
        self.guard_margin
    }

    /// Devuelve el mejor candidato admisible conocido hasta el momento, si existe.
    pub const fn best_admissible_candidate(&self) -> Option<&Action> {
        self.best_admissible_candidate.as_ref()
    }

    /// Devuelve el token de cancelación cooperativo asociado.
    pub const fn cancel_token(&self) -> &CancelToken {
        &self.cancel_token
    }

    /// Devuelve el total acumulado de eventos de clonación.
    pub const fn cloning_events(&self) -> usize {
        self.cloning_events
    }

    /// Devuelve la media de recompensa virtual (`VR`) del último paso evaluado.
    pub const fn last_mean_vr(&self) -> f64 {
        self.last_mean_vr
    }

    /// Número de pasos en los que se registró un sobrepaso temporal.
    pub const fn total_overruns(&self) -> usize {
        self.total_overruns
    }

    /// Máximo sobrepaso observado en nanosegundos a lo largo de toda la sesión.
    pub const fn max_overrun_nanos(&self) -> u128 {
        self.max_overrun_nanos
    }

    /// Indica si la búsqueda alcanzó el horizonte o si toda la población es terminal.
    pub fn is_complete(&self) -> bool {
        self.horizon_step >= self.config.horizon() || self.population.all_terminal()
    }

    /// Ejecuta hasta `max_chunk_steps` pasos de horizonte usando recompensa y proyección `phi` por defecto.
    pub fn step_checkpoint<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &PlanningContext<'_, M>,
        clock: &dyn MonotonicClock,
        max_chunk_steps: usize,
    ) -> CheckpointOutcome {
        self.step_checkpoint_with(ctx, clock, max_chunk_steps, default_reward_fn, default_phi)
    }

    /// Ejecuta hasta `max_chunk_steps` pasos de horizonte con `reward_fn` y proyección `phi` explícitas.
    ///
    /// Pausa limpiamente al agotar `max_chunk_steps`, expirar `deadline - guard_margin` en `clock`,
    /// agotar `transitions_budget`, o activarse `cancel_token`, retornando siempre el mejor candidato
    /// admisible conocido hasta ese momento o `None` si no hay ninguno.
    pub fn step_checkpoint_with<M, R, P>(
        &mut self,
        ctx: &PlanningContext<'_, M>,
        clock: &dyn MonotonicClock,
        max_chunk_steps: usize,
        mut reward_fn: R,
        mut phi: P,
    ) -> CheckpointOutcome
    where
        M: GenerativeModel + ?Sized,
        R: FnMut(&State, &Action, &Transition) -> f64,
        P: FnMut(&State) -> Vec<f64>,
    {
        let chunk_start = clock.now();
        let mut steps_executed = 0usize;
        let mut chunk_overrun = false;
        let mut chunk_overrun_nanos = 0u128;

        loop {
            if self.cancel_token.is_cancelled() {
                return self.build_outcome(
                    CheckpointStatus::Cancelled,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    clock
                        .now()
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }

            if self.is_complete() {
                return self.build_outcome(
                    CheckpointStatus::Completed,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    clock
                        .now()
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }

            if self.transitions_budget.is_transitions_exhausted() {
                return self.build_outcome(
                    CheckpointStatus::BudgetExhausted,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    clock
                        .now()
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }

            let step_before = clock.now();
            if self.is_guard_deadline_expired(step_before) {
                return self.build_outcome(
                    CheckpointStatus::PausedDeadline,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    step_before
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }

            if steps_executed >= max_chunk_steps {
                return self.build_outcome(
                    CheckpointStatus::PausedChunkLimit,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    step_before
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }

            let step_idx = self.horizon_step;
            let actions = match sample_population_actions(
                &self.action_space,
                self.config.population_size(),
                step_idx == 0,
                self.config.scoring().unbiased_sampling(),
                &mut self.planner_rng,
            ) {
                Ok(actions) => actions,
                Err(err) => {
                    return self.build_outcome(
                        CheckpointStatus::Failed(err),
                        steps_executed,
                        chunk_overrun,
                        chunk_overrun_nanos,
                        clock
                            .now()
                            .as_nanos()
                            .saturating_sub(chunk_start.as_nanos()),
                    );
                }
            };

            let step_report = match self.population.step_population(
                ctx.model(),
                &actions,
                self.config.dt(),
                &mut self.transitions_budget,
                &mut reward_fn,
            ) {
                Ok(report) => report,
                Err(PopulationError::ModelStepFailed {
                    source: ModelError::TerminalState,
                    ..
                }) if !self.any_step_completed => {
                    return self.build_outcome(
                        CheckpointStatus::NoValidAction,
                        steps_executed,
                        chunk_overrun,
                        chunk_overrun_nanos,
                        clock
                            .now()
                            .as_nanos()
                            .saturating_sub(chunk_start.as_nanos()),
                    );
                }
                Err(PopulationError::ModelStepFailed { source, .. }) => {
                    return self.build_outcome(
                        CheckpointStatus::Failed(source),
                        steps_executed,
                        chunk_overrun,
                        chunk_overrun_nanos,
                        clock
                            .now()
                            .as_nanos()
                            .saturating_sub(chunk_start.as_nanos()),
                    );
                }
                Err(PopulationError::NonFiniteReward { .. }) => {
                    return self.build_outcome(
                        CheckpointStatus::Failed(ModelError::InvalidState {
                            reason: "FMC reward function produced non-finite value",
                        }),
                        steps_executed,
                        chunk_overrun,
                        chunk_overrun_nanos,
                        clock
                            .now()
                            .as_nanos()
                            .saturating_sub(chunk_start.as_nanos()),
                    );
                }
                Err(_) => {
                    return self.build_outcome(
                        CheckpointStatus::Failed(ModelError::InvalidState {
                            reason: "FMC walker population step failed",
                        }),
                        steps_executed,
                        chunk_overrun,
                        chunk_overrun_nanos,
                        clock
                            .now()
                            .as_nanos()
                            .saturating_sub(chunk_start.as_nanos()),
                    );
                }
            };

            let step_after = clock.now();
            if let Some(over) = self.measure_step_overrun(step_before, step_after) {
                chunk_overrun = true;
                chunk_overrun_nanos = chunk_overrun_nanos.max(over);
                self.total_overruns += 1;
                self.max_overrun_nanos = self.max_overrun_nanos.max(over);
            }

            if step_report.stepped() > 0 {
                self.any_step_completed = true;
                let score = match score_population(
                    &self.population,
                    &mut phi,
                    self.config.scoring(),
                    &mut self.planner_rng,
                ) {
                    Ok(score) => score,
                    Err(_) => {
                        return self.build_outcome(
                            CheckpointStatus::Failed(ModelError::InvalidState {
                                reason: "FMC scoring failed",
                            }),
                            steps_executed,
                            chunk_overrun,
                            chunk_overrun_nanos,
                            clock
                                .now()
                                .as_nanos()
                                .saturating_sub(chunk_start.as_nanos()),
                        );
                    }
                };
                self.last_mean_vr = mean_slice(score.virtual_rewards());

                if let Some(winner) = select_winning_first_action(
                    &self.population,
                    self.config.action_selection(),
                    self.config.scoring(),
                    Some(score.virtual_rewards()),
                ) {
                    self.best_admissible_candidate = Some(winner.action);
                }

                if !step_report.budget_exhausted()
                    && step_idx + 1 < self.config.horizon()
                    && !self.population.all_terminal()
                {
                    let resample_report = match resample_population(
                        &mut self.population,
                        score.virtual_rewards(),
                        &mut self.planner_rng,
                        step_idx as u64,
                    ) {
                        Ok(report) => report,
                        Err(_) => {
                            return self.build_outcome(
                                CheckpointStatus::Failed(ModelError::InvalidState {
                                    reason: "FMC resampling failed",
                                }),
                                steps_executed,
                                chunk_overrun,
                                chunk_overrun_nanos,
                                clock
                                    .now()
                                    .as_nanos()
                                    .saturating_sub(chunk_start.as_nanos()),
                            );
                        }
                    };
                    self.cloning_events += resample_report.cloned_count();
                }

                self.horizon_step += 1;
                steps_executed += 1;
            }

            if step_report.budget_exhausted() {
                return self.build_outcome(
                    CheckpointStatus::BudgetExhausted,
                    steps_executed,
                    chunk_overrun,
                    chunk_overrun_nanos,
                    clock
                        .now()
                        .as_nanos()
                        .saturating_sub(chunk_start.as_nanos()),
                );
            }
        }
    }

    fn is_guard_deadline_expired(&self, now: MonotonicInstant) -> bool {
        let Some(dl) = self.deadline else {
            return false;
        };
        let guard_target = dl
            .target()
            .as_nanos()
            .saturating_sub(self.guard_margin.as_nanos());
        now.as_nanos() >= guard_target
    }

    fn measure_step_overrun(
        &self,
        before: MonotonicInstant,
        after: MonotonicInstant,
    ) -> Option<u128> {
        let mut max_overrun = 0u128;

        if let Some(step_budget) = self.step_budget {
            let elapsed = after.as_nanos().saturating_sub(before.as_nanos());
            let budget_nanos = step_budget.as_nanos();
            if elapsed > budget_nanos {
                max_overrun = max_overrun.max(elapsed - budget_nanos);
            }
        }

        if let Some(dl) = self.deadline {
            let guard_target = dl
                .target()
                .as_nanos()
                .saturating_sub(self.guard_margin.as_nanos());
            if after.as_nanos() > guard_target {
                max_overrun = max_overrun.max(after.as_nanos() - guard_target);
            }
        }

        if max_overrun > 0 {
            Some(max_overrun)
        } else {
            None
        }
    }

    fn build_outcome(
        &self,
        status: CheckpointStatus,
        steps_executed_in_chunk: usize,
        overrun: bool,
        overrun_nanos: u128,
        elapsed_nanos: u128,
    ) -> CheckpointOutcome {
        CheckpointOutcome {
            status,
            best_candidate: self.best_admissible_candidate.clone(),
            steps_executed_in_chunk,
            horizon_step: self.horizon_step,
            transitions_used: self.transitions_budget.transitions_used(),
            cloning_events: self.cloning_events,
            last_mean_vr: self.last_mean_vr,
            overrun,
            overrun_nanos,
            elapsed_nanos,
            warm_start_survivors: self.warm_start_survivors,
            warm_start_hit_ratio: self.warm_start_hit_ratio,
        }
    }
}

/// L2 distance between the raw value vectors of two states; `INFINITY` if dimensions differ.
fn state_l2_distance(a: &State, b: &State) -> f64 {
    let (a, b) = (a.values(), b.values());
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::DiscreteAction;
    use crate::clock::VirtualMonotonicClock;
    use crate::determinism::MasterSeed;
    use crate::domain::{Observation, Terminal, TerminalReason};
    use crate::fmc_planner::FmcPlanner;
    use crate::fmc_scoring::FmcScoringConfig;
    use crate::model::RngSource;
    use crate::planner::{PlanBudget, PlanRequest, PlanTarget, Planner};
    use crate::units::{SimDuration, StepDt, Tick};

    struct GoalLineModel {
        space: ActionSpace,
        goal_x: i64,
        fail_x: i64,
    }

    impl GoalLineModel {
        fn new(goal_x: i64, fail_x: i64) -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
                goal_x,
                fail_x,
            }
        }
    }

    impl GenerativeModel for GoalLineModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            self.space
                .validate(action)
                .map_err(ModelError::InvalidAction)?;
            let cur_x = state.values()[0] as i64;
            if cur_x == self.goal_x || cur_x == self.fail_x {
                return Err(ModelError::TerminalState);
            }
            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            let next_x = if idx == 1 { cur_x + 1 } else { cur_x - 1 };
            let terminal = if next_x == self.goal_x {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else if next_x == self.fail_x {
                Terminal::Terminated(TerminalReason::Failure)
            } else {
                Terminal::Ongoing
            };
            let next_state = state
                .advance(dt, vec![next_x as f64])
                .map_err(ModelError::TransitionFailed)?;
            let obs =
                Observation::new(next_state.tick(), next_state.elapsed(), vec![next_x as f64])
                    .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    struct ExpensiveClockModel<'a> {
        inner: GoalLineModel,
        clock: &'a VirtualMonotonicClock,
        cost_per_step: Duration,
    }

    impl<'a> GenerativeModel for ExpensiveClockModel<'a> {
        fn action_space(&self) -> &ActionSpace {
            self.inner.action_space()
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            self.clock.advance(self.cost_per_step);
            self.inner.step(state, action, dt, rng)
        }
    }

    fn base_config(population_size: usize, horizon: usize, seed: u64) -> FmcPlannerConfig {
        FmcPlannerConfig::new(
            population_size,
            horizon,
            StepDt::from_millis_f64(100.0).unwrap(),
            FmcScoringConfig::default(),
            MasterSeed::new(seed),
        )
        .unwrap()
    }

    #[test]
    fn interruption_at_any_checkpoint_returns_best_admissible_or_explicit_none() {
        let model = GoalLineModel::new(3, -3);
        let initial_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let config = base_config(8, 5, 42);
        let clock = VirtualMonotonicClock::new();

        // Caso A: cancelación inmediata sin candidato inicial -> None explícito.
        let cancel_now = CancelToken::cancelled();
        let mut session_none = IncrementalFmcSession::new(
            config.clone(),
            initial_state.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(100).tracker(),
            None,
            Duration::ZERO,
            None,
            cancel_now,
        )
        .unwrap();
        let mut rng_a = DeterministicRng::new(1);
        let ctx_a = PlanningContext::new(&model, &mut rng_a);
        let out_none = session_none.step_checkpoint(&ctx_a, &clock, 2);
        assert_eq!(out_none.status(), &CheckpointStatus::Cancelled);
        assert_eq!(out_none.best_candidate(), None);
        assert_eq!(out_none.steps_executed_in_chunk(), 0);

        // Caso B: interrupción inmediata con candidato inicial admisible -> lo conserva.
        let fallback_action = Action::discrete_in_space(0, model.action_space()).unwrap();
        let mut session_init = IncrementalFmcSession::new(
            config.clone(),
            initial_state.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(100).tracker(),
            None,
            Duration::ZERO,
            Some(fallback_action.clone()),
            CancelToken::new(),
        )
        .unwrap();
        let out_zero_chunk = session_init.step_checkpoint(&ctx_a, &clock, 0);
        assert_eq!(out_zero_chunk.status(), &CheckpointStatus::PausedChunkLimit);
        assert_eq!(out_zero_chunk.best_candidate(), Some(&fallback_action));

        // Caso C: ejecución incremental en chunks de 1 paso hasta completar,
        // equivalente al FmcPlanner monolítico.
        let mut outcomes = Vec::new();
        loop {
            let out = session_init.step_checkpoint(&ctx_a, &clock, 1);
            let finished = out.status().is_finished();
            outcomes.push(out);
            if finished {
                break;
            }
        }
        let last = outcomes.last().unwrap();
        assert_eq!(last.status(), &CheckpointStatus::Completed);
        let expected_goal_action = Action::discrete_in_space(1, model.action_space()).unwrap();
        assert_eq!(last.best_candidate(), Some(&expected_goal_action));

        let mut planner = FmcPlanner::new(config);
        let mut rng_b = DeterministicRng::new(1);
        let mut ctx_b = PlanningContext::new(&model, &mut rng_b);
        let req = PlanRequest::new(
            PlanTarget::State(initial_state),
            model.action_space().clone(),
            PlanBudget::with_transitions(100),
            CancelToken::new(),
            None,
        )
        .unwrap();
        let batch_res = planner.plan(&mut ctx_b, &req).unwrap();
        assert_eq!(last.best_candidate(), batch_res.authorized_action());
    }

    fn state_at(x: f64, tick: u64) -> State {
        State::new(Tick::new(tick), SimDuration::ZERO, vec![x]).unwrap()
    }

    fn warm_session(horizon: usize, n: usize) -> IncrementalFmcSession {
        IncrementalFmcSession::new(
            base_config(n, horizon, 5),
            state_at(0.0, 0),
            ActionSpace::discrete(2).unwrap(),
            PlanBudget::with_transitions(100_000).tracker(),
            None,
            Duration::ZERO,
            None,
            CancelToken::new(),
        )
        .unwrap()
    }

    #[test]
    fn advance_root_recycles_horizon_with_fraction_of_cold_budget() {
        let h = 8;
        let model = GoalLineModel::new(1000, -1000);
        let clock = VirtualMonotonicClock::new();
        let mut rng = DeterministicRng::new(1);
        let ctx = PlanningContext::new(&model, &mut rng);

        let mut session = warm_session(h, 16);
        let cold = session.step_checkpoint(&ctx, &clock, h);
        assert_eq!(cold.status(), &CheckpointStatus::Completed);
        let cold_cost = cold.transitions_used();
        assert_eq!(cold_cost, 16 * h);
        assert_eq!(cold.warm_start_survivors(), 0);
        assert_eq!(cold.warm_start_hit_ratio(), 0.0);

        let executed = session.best_admissible_candidate().unwrap().clone();
        let delta = if executed.as_discrete().unwrap().index() == 1 {
            1.0
        } else {
            -1.0
        };
        let survivors = session.advance_root(&executed, &state_at(delta, 1), 1e-9);
        assert!(survivors > 0);
        assert_eq!(session.horizon_step(), h - 1);
        let pop = session.population();
        for i in 0..pop.len() {
            let id = WalkerId::new(i);
            assert_eq!(
                pop.first_action(id).unwrap(),
                pop.first_actions()[i].as_ref()
            );
            assert!(pop.cumulative_reward(id).unwrap().is_finite());
            assert_eq!(pop.prefix_len(id).unwrap(), h - 1);
        }

        let warm = session.step_checkpoint(&ctx, &clock, h);
        assert_eq!(warm.status(), &CheckpointStatus::Completed);
        assert_eq!(warm.horizon_step(), h);
        assert_eq!(warm.steps_executed_in_chunk(), 1);
        let warm_cost = warm.transitions_used() - cold_cost;
        assert!(warm_cost * h <= 2 * cold_cost, "{warm_cost} vs {cold_cost}");
        assert_eq!(warm.warm_start_survivors(), survivors);
        assert!(warm.warm_start_hit_ratio() > 0.0 && warm.warm_start_hit_ratio() <= 1.0);
    }

    #[test]
    fn advance_root_invalidates_warm_start_outside_tolerance() {
        let h = 8;
        let n = 16;
        let model = GoalLineModel::new(1000, -1000);
        let clock = VirtualMonotonicClock::new();
        let mut rng = DeterministicRng::new(1);
        let ctx = PlanningContext::new(&model, &mut rng);

        let mut session = warm_session(h, n);
        session.step_checkpoint(&ctx, &clock, h);
        let executed = session.best_admissible_candidate().unwrap().clone();
        // Perturbed observation: far from any simulated step-1 state.
        let observed = state_at(50.0, 1);
        assert_eq!(session.advance_root(&executed, &observed, 0.5), 0);
        assert_eq!(session.horizon_step(), 0);
        assert_eq!(session.warm_start_hit_ratio(), 0.0);
        let pop = session.population();
        assert_eq!(pop.len(), n);
        assert!(pop.states().iter().all(|s| s.values() == [50.0]));
        assert!(pop.cumulative_rewards().iter().all(|r| *r == 0.0));
        assert!(pop.first_actions().iter().all(Option::is_none));

        // Non-finite tolerance also falls back cleanly; session remains usable.
        assert_eq!(session.advance_root(&executed, &observed, f64::NAN), 0);
        let out = session.step_checkpoint(&ctx, &clock, h);
        assert_eq!(out.status(), &CheckpointStatus::Completed);
        assert_eq!(out.warm_start_survivors(), 0);
        assert_eq!(out.warm_start_hit_ratio(), 0.0);
        assert!(
            session
                .population()
                .states()
                .iter()
                .all(|s| s.values()[0].is_finite())
        );
    }

    #[test]
    fn exact_transition_budget_and_monotonic_deadline_with_guard_margin() {
        let model = GoalLineModel::new(5, -5);
        let initial_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let config = base_config(4, 6, 7);
        let clock = VirtualMonotonicClock::new();

        // Deadline a 100ms, guard_margin de 25ms -> corte efectivo en 75ms.
        let deadline = MonotonicDeadline::from_now(&clock, Duration::from_millis(100));
        let guard_margin = Duration::from_millis(25);

        let mut session = IncrementalFmcSession::new(
            config,
            initial_state,
            model.action_space().clone(),
            PlanBudget::with_transitions(10).tracker(),
            Some(deadline),
            guard_margin,
            None,
            CancelToken::new(),
        )
        .unwrap();

        let mut rng = DeterministicRng::new(9);
        let ctx = PlanningContext::new(&model, &mut rng);

        // Chunk 1: avanza 1 paso (4 transiciones), reloj en 40ms (< 75ms).
        clock.advance(Duration::from_millis(40));
        let out1 = session.step_checkpoint(&ctx, &clock, 1);
        assert_eq!(out1.status(), &CheckpointStatus::PausedChunkLimit);
        assert_eq!(out1.transitions_used(), 4);
        assert_eq!(session.transitions_budget().remaining_transitions(), 6);
        assert!(!out1.overrun());

        // Avanzamos reloj a exactamente 75ms (límite del guard_margin): debe pausar limpiamente sin overrun.
        clock.advance(Duration::from_millis(35));
        let out2 = session.step_checkpoint(&ctx, &clock, 2);
        assert_eq!(out2.status(), &CheckpointStatus::PausedDeadline);
        assert_eq!(out2.steps_executed_in_chunk(), 0);
        assert_eq!(out2.transitions_used(), 4);
        assert!(!out2.overrun());
        assert_eq!(out2.overrun_nanos(), 0);
        assert!(out2.best_candidate().is_some());

        // Sesión separada sin deadline para verificar agotamiento exacto de BudgetTracker (10 transiciones).
        let mut budget_session = IncrementalFmcSession::new(
            base_config(4, 6, 7),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap(),
            model.action_space().clone(),
            PlanBudget::with_transitions(10).tracker(),
            None,
            Duration::ZERO,
            None,
            CancelToken::new(),
        )
        .unwrap();
        let out_budget = budget_session.step_checkpoint(&ctx, &clock, 6);
        assert_eq!(out_budget.status(), &CheckpointStatus::BudgetExhausted);
        assert_eq!(out_budget.transitions_used(), 10);
        assert_eq!(
            budget_session.transitions_budget().remaining_transitions(),
            0
        );
    }

    #[test]
    fn expensive_fixture_records_overrun_without_promising_unbounded_limits() {
        let clock = VirtualMonotonicClock::new();
        // Cada walker tarda 10ms -> con 4 walkers, 1 paso de horizonte consume 40ms.
        let model = ExpensiveClockModel {
            inner: GoalLineModel::new(4, -4),
            clock: &clock,
            cost_per_step: Duration::from_millis(10),
        };
        let initial_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let config = base_config(4, 4, 11);

        // Deadline en 50ms con guard_margin de 20ms (corte efectivo a los 30ms) y step_budget de 15ms.
        let deadline = MonotonicDeadline::from_now(&clock, Duration::from_millis(50));
        let mut session = IncrementalFmcSession::new(
            config,
            initial_state,
            model.action_space().clone(),
            PlanBudget::with_transitions(100).tracker(),
            Some(deadline),
            Duration::from_millis(20),
            None,
            CancelToken::new(),
        )
        .unwrap()
        .with_step_budget(Duration::from_millis(15));

        let mut rng = DeterministicRng::new(3);
        let ctx = PlanningContext::new(&model, &mut rng);

        // El primer paso arranca en t=0 (< 30ms), pero al simular 4 walkers tarda 40ms:
        // excede step_budget (15ms) por 25ms y sobrepasa el guard_target (30ms) por 10ms.
        let outcome = session.step_checkpoint(&ctx, &clock, 4);
        assert_eq!(outcome.status(), &CheckpointStatus::PausedDeadline);
        assert_eq!(outcome.steps_executed_in_chunk(), 1);
        assert!(outcome.overrun());
        assert_eq!(
            outcome.overrun_nanos(),
            Duration::from_millis(25).as_nanos()
        );
        assert_eq!(session.total_overruns(), 1);
        assert_eq!(
            session.max_overrun_nanos(),
            Duration::from_millis(25).as_nanos()
        );
        // Aunque hubo sobrepaso en el modelo no acotado, conserva el candidato válido obtenido.
        assert!(outcome.best_candidate().is_some());
    }
}
