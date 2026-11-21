//! Runtime headless `observe -> plan -> guard -> act` compuesto sobre [`LocalScheduler`].

use std::cell::RefCell;
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::rc::Rc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gogamen_core::{
    ActionGuard, ActionProposal, CancelToken, DomainPrecondition, EnvironmentError, EventKind,
    GenerativeModel, GuardContext, GuardDecision, GuardRejectionReason, LiveEnvironment,
    LocalScheduler, MasterSeed, ModelError, MonotonicClock, MonotonicDeadline, NoPrecondition,
    Observation, PlanBudget, PlanRequest, PlanResult, PlanStatus, PlanTarget, Planner,
    PlanningContext, ScheduleTrigger, ScheduledTask, SchedulerConfig, SchedulerError,
    SchedulerEvent, SchedulerLifecycle, State, StepDt, StepReport, StreamKey, TaskContext,
    TaskHandler, Terminal, Tick,
};

use crate::trace::{ExecutionTrace, RejectionCause, TickTraceEntry};

const EVENT_OBSERVED: &str = "runtime.observed";
const EVENT_PLANNED: &str = "runtime.planned";
const EVENT_GUARDED: &str = "runtime.guarded";

/// Configuración del runtime headless de horizonte recedente.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Semilla maestra determinista de la ejecución.
    pub seed: MasterSeed,
    /// Paso temporal físico de cada tick.
    pub dt: StepDt,
    /// Máximo de transiciones simuladas permitidas al planificador por tick.
    pub max_transitions_per_tick: usize,
    /// Presupuesto temporal monotónico máximo por tick para planificación.
    pub planning_timeout: Duration,
    /// Antigüedad máxima opcional en ticks para propuestas evaluadas por la guardia.
    pub proposal_max_age_ticks: Option<u64>,
    /// Configuración del [`LocalScheduler`] subyacente.
    pub scheduler_config: SchedulerConfig,
}

impl RuntimeConfig {
    /// Crea una configuración por defecto para `seed` y `dt`.
    pub fn new(seed: MasterSeed, dt: StepDt) -> Self {
        Self {
            seed,
            dt,
            max_transitions_per_tick: 64,
            planning_timeout: Duration::from_millis(50),
            proposal_max_age_ticks: Some(0),
            scheduler_config: SchedulerConfig::default(),
        }
    }
}

/// Errores tipados del runtime headless.
#[derive(Debug)]
pub enum RuntimeError {
    /// Error del [`LocalScheduler`] subyacente.
    Scheduler(SchedulerError),
    /// Error del entorno vivo al observar o actuar.
    Environment(EnvironmentError),
    /// Error del modelo generativo durante la planificación.
    Model(ModelError),
    /// Error al construir eventos internos del scheduler.
    InvalidEventKind(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scheduler(err) => write!(f, "runtime scheduler error: {err}"),
            Self::Environment(err) => write!(f, "runtime environment error: {err}"),
            Self::Model(err) => write!(f, "runtime model error: {err}"),
            Self::InvalidEventKind(msg) => write!(f, "runtime invalid event kind: {msg}"),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Scheduler(err) => Some(err),
            Self::Environment(err) => Some(err),
            Self::Model(err) => Some(err),
            Self::InvalidEventKind(_) => None,
        }
    }
}

impl From<SchedulerError> for RuntimeError {
    fn from(value: SchedulerError) -> Self {
        Self::Scheduler(value)
    }
}

impl From<EnvironmentError> for RuntimeError {
    fn from(value: EnvironmentError) -> Self {
        Self::Environment(value)
    }
}

impl From<ModelError> for RuntimeError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

/// Modificador opcional de propuesta para inyectar escenarios de caducidad o invalidez en tests.
pub type ProposalMutator = Box<dyn FnMut(Tick, Instant, ActionProposal) -> ActionProposal>;

struct TickScratch {
    observation: Option<Observation>,
    plan_result: Option<PlanResult>,
    proposal: Option<ActionProposal>,
    proposal_expired_by_clock: bool,
    guard_decision: Option<GuardDecision>,
    rejection_reason: Option<RejectionCause>,
    used_fallback: bool,
    tick_instant: Option<Instant>,
}

impl TickScratch {
    fn clear(&mut self) {
        self.observation = None;
        self.plan_result = None;
        self.proposal = None;
        self.proposal_expired_by_clock = false;
        self.guard_decision = None;
        self.rejection_reason = None;
        self.used_fallback = false;
        self.tick_instant = None;
    }
}

struct RuntimeShared<E, M, P, Pre> {
    config: RuntimeConfig,
    env: E,
    model: M,
    planner: P,
    guard: ActionGuard<Pre>,
    clock: Arc<dyn MonotonicClock>,
    cancel_token: CancelToken,
    last_terminal: Terminal,
    trace: ExecutionTrace,
    scratch: TickScratch,
    proposal_mutator: Option<ProposalMutator>,
}

struct ObserveTask<E, M, P, Pre> {
    shared: Rc<RefCell<RuntimeShared<E, M, P, Pre>>>,
    observed_event: EventKind,
}

impl<E, M, P, Pre> TaskHandler for ObserveTask<E, M, P, Pre>
where
    E: LiveEnvironment,
{
    fn execute(&mut self, _ctx: &TaskContext<'_>) -> Result<Vec<SchedulerEvent>, SchedulerError> {
        let mut state = self.shared.borrow_mut();
        state.scratch.clear();
        state.scratch.tick_instant = Some(Instant::now());
        let obs = state
            .env
            .observe()
            .map_err(|err| SchedulerError::TaskFailed {
                task_name: "observe".to_string(),
                reason: err.to_string(),
            })?;
        state.scratch.observation = Some(obs);
        Ok(vec![SchedulerEvent::signal(self.observed_event.clone())])
    }
}

struct PlanTask<E, M, P, Pre> {
    shared: Rc<RefCell<RuntimeShared<E, M, P, Pre>>>,
    planned_event: EventKind,
}

impl<E, M, P, Pre> TaskHandler for PlanTask<E, M, P, Pre>
where
    E: LiveEnvironment,
    M: GenerativeModel,
    P: Planner,
{
    fn execute(&mut self, ctx: &TaskContext<'_>) -> Result<Vec<SchedulerEvent>, SchedulerError> {
        let mut state = self.shared.borrow_mut();
        let obs = state
            .scratch
            .observation
            .clone()
            .ok_or_else(|| SchedulerError::TaskFailed {
                task_name: "plan".to_string(),
                reason: "missing observation before plan phase".to_string(),
            })?;

        let tick_instant = state.scratch.tick_instant.unwrap_or_else(Instant::now);
        let clock = Arc::clone(&state.clock);
        let deadline = MonotonicDeadline::from_now(clock.as_ref(), state.config.planning_timeout);

        let plan_result = if state.cancel_token.is_cancelled() {
            PlanResult::cancelled(0)
        } else if deadline.is_expired(clock.as_ref()) {
            state.scratch.proposal_expired_by_clock = true;
            PlanResult::budget_exhausted(None, 0)
        } else {
            let plan_state = State::new(ctx.tick(), ctx.elapsed(), obs.features().to_vec())
                .map_err(SchedulerError::Domain)?;
            let budget = PlanBudget::new(
                state.config.max_transitions_per_tick,
                tick_instant.checked_add(state.config.planning_timeout),
            );
            let request = PlanRequest::new(
                PlanTarget::StateAndObservation {
                    state: plan_state,
                    observation: obs,
                },
                state.env.action_space().clone(),
                budget,
                state.cancel_token.clone(),
                None,
            )
            .map_err(SchedulerError::Domain)?;

            let mut rng = state
                .config
                .seed
                .derive(StreamKey::agent(0, ctx.tick().as_u64()));

            let raw_result = {
                let RuntimeShared { model, planner, .. } = &mut *state;
                let mut planning_ctx = PlanningContext::new(model, &mut rng);
                planner.plan(&mut planning_ctx, &request).map_err(|err| {
                    SchedulerError::TaskFailed {
                        task_name: "plan".to_string(),
                        reason: err.to_string(),
                    }
                })?
            };

            if deadline.is_expired(clock.as_ref()) && raw_result.status().is_completed() {
                state.scratch.proposal_expired_by_clock = true;
                PlanResult::budget_exhausted(None, raw_result.transitions_used())
            } else {
                raw_result
            }
        };

        let mut proposal = ActionProposal::from_plan_result(&plan_result, ctx.tick());
        if let Some(max_age) = state.config.proposal_max_age_ticks {
            proposal = proposal.with_max_age_ticks(max_age);
        }
        if state.scratch.proposal_expired_by_clock {
            proposal = proposal.with_expires_at(tick_instant);
        }
        if let Some(mutator) = state.proposal_mutator.as_mut() {
            proposal = mutator(ctx.tick(), tick_instant, proposal);
        }

        state.scratch.plan_result = Some(plan_result);
        state.scratch.proposal = Some(proposal);
        Ok(vec![SchedulerEvent::signal(self.planned_event.clone())])
    }
}

struct GuardTask<E, M, P, Pre> {
    shared: Rc<RefCell<RuntimeShared<E, M, P, Pre>>>,
    guarded_event: EventKind,
}

impl<E, M, P, Pre> TaskHandler for GuardTask<E, M, P, Pre>
where
    E: LiveEnvironment,
    Pre: DomainPrecondition,
{
    fn execute(&mut self, ctx: &TaskContext<'_>) -> Result<Vec<SchedulerEvent>, SchedulerError> {
        let mut state = self.shared.borrow_mut();
        let obs = state
            .scratch
            .observation
            .as_ref()
            .ok_or_else(|| SchedulerError::TaskFailed {
                task_name: "guard".to_string(),
                reason: "missing observation before guard phase".to_string(),
            })?;
        let plan_result =
            state
                .scratch
                .plan_result
                .as_ref()
                .ok_or_else(|| SchedulerError::TaskFailed {
                    task_name: "guard".to_string(),
                    reason: "missing plan result before guard phase".to_string(),
                })?;
        let proposal =
            state
                .scratch
                .proposal
                .as_ref()
                .ok_or_else(|| SchedulerError::TaskFailed {
                    task_name: "guard".to_string(),
                    reason: "missing action proposal before guard phase".to_string(),
                })?;

        let now = state.scratch.tick_instant.unwrap_or_else(Instant::now);
        let guard_ctx = GuardContext::new(
            state.env.action_space(),
            ctx.tick(),
            now,
            state.last_terminal,
        )
        .with_observation(obs);
        let decision = state.guard.evaluate(proposal, &guard_ctx);

        let used_fallback = decision.is_fallback()
            || (plan_result.status().is_budget_exhausted() && decision.action().is_some());

        let rejection_reason = match &decision {
            GuardDecision::Reject(reason) => Some(RejectionCause::Guard(reason.clone())),
            GuardDecision::Fallback(_) | GuardDecision::NoOp => match plan_result.status() {
                PlanStatus::BudgetExhausted => Some(RejectionCause::PlanBudgetExhausted),
                PlanStatus::Cancelled => Some(RejectionCause::PlanCancelled),
                PlanStatus::NoValidAction => Some(RejectionCause::PlanNoValidAction),
                PlanStatus::Completed => {
                    if proposal.ttl_overflowed() {
                        Some(RejectionCause::Guard(
                            GuardRejectionReason::DeadlineOverflow,
                        ))
                    } else if proposal.expires_at().is_some_and(|exp| now >= exp) {
                        Some(RejectionCause::Guard(GuardRejectionReason::DeadlineExpired))
                    } else if proposal.authorized_action().is_none() {
                        Some(RejectionCause::Guard(GuardRejectionReason::Unauthorized))
                    } else {
                        Some(RejectionCause::Guard(
                            GuardRejectionReason::PreconditionFailed {
                                reason: "proposal rejected by guard policy",
                            },
                        ))
                    }
                }
            },
            GuardDecision::Allow(_) => {
                if plan_result.status().is_budget_exhausted() {
                    Some(RejectionCause::PlanBudgetExhausted)
                } else {
                    None
                }
            }
        };

        state.scratch.guard_decision = Some(decision);
        state.scratch.rejection_reason = rejection_reason;
        state.scratch.used_fallback = used_fallback;
        Ok(vec![SchedulerEvent::signal(self.guarded_event.clone())])
    }
}

struct ActTask<E, M, P, Pre> {
    shared: Rc<RefCell<RuntimeShared<E, M, P, Pre>>>,
}

impl<E, M, P, Pre> TaskHandler for ActTask<E, M, P, Pre>
where
    E: LiveEnvironment,
{
    fn execute(&mut self, ctx: &TaskContext<'_>) -> Result<Vec<SchedulerEvent>, SchedulerError> {
        let mut state = self.shared.borrow_mut();
        let observation =
            state
                .scratch
                .observation
                .take()
                .ok_or_else(|| SchedulerError::TaskFailed {
                    task_name: "act".to_string(),
                    reason: "missing observation in act phase".to_string(),
                })?;
        let plan_result =
            state
                .scratch
                .plan_result
                .take()
                .ok_or_else(|| SchedulerError::TaskFailed {
                    task_name: "act".to_string(),
                    reason: "missing plan result in act phase".to_string(),
                })?;
        let guard_decision =
            state
                .scratch
                .guard_decision
                .take()
                .ok_or_else(|| SchedulerError::TaskFailed {
                    task_name: "act".to_string(),
                    reason: "missing guard decision in act phase".to_string(),
                })?;
        let mut rejection_reason = state.scratch.rejection_reason.take();
        let used_fallback = state.scratch.used_fallback;

        let (applied_action, outcome) = match guard_decision.action() {
            Some(action) => {
                let apply_outcome =
                    state
                        .env
                        .apply(action)
                        .map_err(|err| SchedulerError::TaskFailed {
                            task_name: "act".to_string(),
                            reason: err.to_string(),
                        })?;
                state.last_terminal = apply_outcome.terminal();
                (Some(action.clone()), Some(apply_outcome))
            }
            None => (None, None),
        };

        if applied_action.is_none() && rejection_reason.is_none() {
            rejection_reason = Some(RejectionCause::Guard(GuardRejectionReason::Unauthorized));
        }

        let entry = TickTraceEntry {
            tick: ctx.tick(),
            seed: state.config.seed,
            observation,
            plan_status: plan_result.status(),
            selected_action: plan_result.selected_action().cloned(),
            guard_decision,
            applied_action,
            rejection_reason,
            used_fallback,
            outcome,
        };
        state.trace.push(entry);
        Ok(Vec::new())
    }
}

/// Runtime headless de horizonte recedente compuesto sobre [`LocalScheduler`].
pub struct HeadlessRuntime<E, M, P, Pre = NoPrecondition> {
    scheduler: LocalScheduler,
    shared: Rc<RefCell<RuntimeShared<E, M, P, Pre>>>,
}

impl<E, M, P, Pre> HeadlessRuntime<E, M, P, Pre>
where
    E: LiveEnvironment + 'static,
    M: GenerativeModel + 'static,
    P: Planner + 'static,
    Pre: DomainPrecondition + 'static,
{
    /// Construye un [`HeadlessRuntime`] componiendo las tareas `observe -> plan -> guard -> act`
    /// sobre una instancia de [`LocalScheduler`].
    pub fn new(
        config: RuntimeConfig,
        env: E,
        model: M,
        planner: P,
        guard: ActionGuard<Pre>,
        clock: Arc<dyn MonotonicClock>,
    ) -> Result<Self, RuntimeError> {
        let trace = ExecutionTrace::new(config.seed);
        let mut scheduler = LocalScheduler::new(config.scheduler_config)?;

        let shared = Rc::new(RefCell::new(RuntimeShared {
            config,
            env,
            model,
            planner,
            guard,
            clock,
            cancel_token: CancelToken::new(),
            last_terminal: Terminal::Ongoing,
            trace,
            scratch: TickScratch {
                observation: None,
                plan_result: None,
                proposal: None,
                proposal_expired_by_clock: false,
                guard_decision: None,
                rejection_reason: None,
                used_fallback: false,
                tick_instant: None,
            },
            proposal_mutator: None,
        }));

        let observed_event = EventKind::new(EVENT_OBSERVED)?;
        let planned_event = EventKind::new(EVENT_PLANNED)?;
        let guarded_event = EventKind::new(EVENT_GUARDED)?;
        let every_tick = NonZeroU64::new(1).expect("1 is non-zero");

        scheduler.register_task(
            ScheduledTask::new(
                "observe",
                ScheduleTrigger::Periodic {
                    every_ticks: every_tick,
                },
                ObserveTask {
                    shared: Rc::clone(&shared),
                    observed_event: observed_event.clone(),
                },
            )?
            .with_priority(40),
        )?;

        scheduler.register_task(
            ScheduledTask::new(
                "plan",
                ScheduleTrigger::OnEvent(observed_event),
                PlanTask {
                    shared: Rc::clone(&shared),
                    planned_event: planned_event.clone(),
                },
            )?
            .with_priority(30),
        )?;

        scheduler.register_task(
            ScheduledTask::new(
                "guard",
                ScheduleTrigger::OnEvent(planned_event),
                GuardTask {
                    shared: Rc::clone(&shared),
                    guarded_event: guarded_event.clone(),
                },
            )?
            .with_priority(20),
        )?;

        scheduler.register_task(
            ScheduledTask::new(
                "act",
                ScheduleTrigger::OnEvent(guarded_event),
                ActTask {
                    shared: Rc::clone(&shared),
                },
            )?
            .with_priority(10),
        )?;

        Ok(Self { scheduler, shared })
    }

    /// Configura un mutador de propuestas para pruebas de caducidad o propuestas inválidas.
    pub fn set_proposal_mutator(&mut self, mutator: ProposalMutator) {
        self.shared.borrow_mut().proposal_mutator = Some(mutator);
    }

    /// Devuelve el [`CancelToken`] compartido con el planificador.
    pub fn cancel_token(&self) -> CancelToken {
        self.shared.borrow().cancel_token.clone()
    }

    /// Arranca el [`LocalScheduler`] subyacente si todavía está en `Created` o `Prepared`.
    pub fn start(&mut self) -> Result<(), RuntimeError> {
        match self.scheduler.lifecycle() {
            SchedulerLifecycle::Created => {
                self.scheduler.prepare()?;
                self.scheduler.start()?;
            }
            SchedulerLifecycle::Prepared => {
                self.scheduler.start()?;
            }
            SchedulerLifecycle::Running | SchedulerLifecycle::Stopped => {}
        }
        Ok(())
    }

    /// Ejecuta un único tick del bucle `observe -> plan -> guard -> act` sobre [`LocalScheduler`].
    pub fn step(&mut self) -> Result<StepReport, RuntimeError> {
        self.start()?;
        let (dt, clock) = {
            let s = self.shared.borrow();
            (s.config.dt, Arc::clone(&s.clock))
        };
        Ok(self.scheduler.step(dt, clock.as_ref())?)
    }

    /// Ejecuta exactamente `ticks` pasos consecutivos y detiene el scheduler al finalizar.
    pub fn run_ticks(&mut self, ticks: u64) -> Result<Vec<StepReport>, RuntimeError> {
        self.start()?;
        let mut reports = Vec::with_capacity(ticks as usize);
        for _ in 0..ticks {
            reports.push(self.step()?);
        }
        self.stop()?;
        Ok(reports)
    }

    /// Detiene el [`LocalScheduler`] subyacente si está en ejecución.
    pub fn stop(&mut self) -> Result<(), RuntimeError> {
        if self.scheduler.lifecycle() == SchedulerLifecycle::Running {
            self.scheduler.stop()?;
        }
        Ok(())
    }

    /// Acceso de solo lectura al [`LocalScheduler`] subyacente.
    pub const fn scheduler(&self) -> &LocalScheduler {
        &self.scheduler
    }

    /// Ejecuta un closure de solo lectura sobre la [`ExecutionTrace`] acumulada.
    pub fn with_trace<R>(&self, f: impl FnOnce(&ExecutionTrace) -> R) -> R {
        let shared = self.shared.borrow();
        f(&shared.trace)
    }

    /// Ejecuta un closure de solo lectura sobre el entorno vivo.
    pub fn with_env<R>(&self, f: impl FnOnce(&E) -> R) -> R {
        let shared = self.shared.borrow();
        f(&shared.env)
    }

    /// Clona la [`ExecutionTrace`] acumulada hasta el momento.
    pub fn trace_snapshot(&self) -> ExecutionTrace {
        self.shared.borrow().trace.clone()
    }
}
