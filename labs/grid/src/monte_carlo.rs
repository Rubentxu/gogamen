//! Baseline Monte Carlo de horizonte fijo (random shooting) sobre [`GridWorldModel`].
//!
//! Evalúa trayectorias candidatas clonando/restaurando snapshots mediante [`BranchableSnapshot`]
//! y simulando pasos con [`PlanningContext`] y [`GenerativeModel`] sin mutar el entorno vivo.

use gogamen_core::{
    Action, ActionSpace, Actuator, BranchableSnapshot, CancelToken, DeterminismMetadata,
    DiscreteAction, DomainError, EnvironmentError, EnvironmentObserver, GenerativeModel,
    MasterSeed, ModelError, PlanBudget, PlanRequest, PlanResult, Planner, PlanningContext,
    RngSource, StepDt, StreamKey, Terminal, TerminalReason, TraceHasher,
};

use crate::baselines::{BaselineBudget, BaselineKind, BaselineMetrics};
use crate::map::GridMap;
use crate::world::{
    GRID_ACTION_CARDINALITY, GridAction, GridWorldConfig, GridWorldEnv, GridWorldModel,
};

/// Configuración del planificador Monte Carlo de horizonte fijo (random shooting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MonteCarloConfig {
    /// Longitud fija de cada trayectoria simulada en pasos.
    pub horizon: usize,
    /// Número de rollouts candidatos a evaluar por decisión.
    pub rollouts: usize,
    /// Límite máximo de transiciones simuladas por invocación de planificación.
    pub max_transitions: usize,
    /// Acción de respaldo explícita cuando el presupuesto se agota o no hay simulación válida.
    pub fallback_action: GridAction,
}

impl MonteCarloConfig {
    /// Construye una configuración explícita de Monte Carlo random shooting.
    pub const fn new(
        horizon: usize,
        rollouts: usize,
        max_transitions: usize,
        fallback_action: GridAction,
    ) -> Self {
        Self {
            horizon,
            rollouts,
            max_transitions,
            fallback_action,
        }
    }
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            horizon: 8,
            rollouts: 32,
            max_transitions: 256,
            fallback_action: GridAction::Stay,
        }
    }
}

/// Resultado de una invocación individual de [`MonteCarloPlanner::plan_grid`].
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloStepResult {
    /// Resultado de contrato del planificador (`Completed`, `BudgetExhausted`, etc.).
    pub plan_result: PlanResult,
    /// Acción resuelta para ejecutar (mejor trayectoria o `fallback_action`).
    pub chosen_action: GridAction,
    /// Indica si se recurrió explícitamente a `fallback_action`.
    pub used_fallback: bool,
    /// Número exacto de rollouts iniciados/ejecutados en esta planificación.
    pub simulations_run: usize,
    /// Número exacto de transiciones simuladas consumidas en esta planificación.
    pub transitions_evaluated: usize,
    /// Retorno acumulado de la mejor trayectoria evaluada, si se completó al menos una.
    pub best_return: Option<f64>,
    /// Secuencia completa de acciones de la mejor trayectoria evaluada.
    pub best_sequence: Vec<GridAction>,
}

/// Métricas y traza determinista de una ejecución completa del baseline Monte Carlo.
#[derive(Debug, Clone, PartialEq)]
pub struct MonteCarloBaselineMetrics {
    /// Semilla determinista utilizada para los rollouts.
    pub seed: MasterSeed,
    /// Indica si el agente alcanzó `CellKind::Goal`.
    pub reached_goal: bool,
    /// Ticks ejecutados sobre [`GridWorldEnv`].
    pub ticks_executed: u64,
    /// Rollouts simulados acumulados durante todo el episodio.
    pub simulations_run: u64,
    /// Transiciones totales evaluadas (simulaciones de planificación + pasos en entorno).
    pub transitions_evaluated: u64,
    /// Número de decisiones en las que se usó `fallback_action`.
    pub fallbacks_used: u64,
    /// Secuencia exacta de acciones ejecutadas en [`GridWorldEnv`].
    pub selected_actions: Vec<GridAction>,
    /// Recompensa acumulada en [`GridWorldEnv`].
    pub total_reward: f64,
    /// Colisiones contra muros o bordes durante la ejecución.
    pub collisions: u64,
    /// Longitud del camino en ticks si se alcanzó la meta.
    pub path_len: Option<u64>,
    /// Hash determinista de la traza de ejecución y decisiones.
    pub trace_hash: u64,
}

impl MonteCarloBaselineMetrics {
    /// Convierte estas métricas al formato comparable [`BaselineMetrics`].
    pub fn to_baseline_metrics(&self) -> BaselineMetrics {
        BaselineMetrics {
            kind: BaselineKind::MonteCarloShooting,
            is_privileged_full_info: BaselineKind::MonteCarloShooting.is_privileged_full_info(),
            seed: Some(self.seed),
            reached_goal: self.reached_goal,
            no_path: false,
            ticks_executed: self.ticks_executed,
            transitions_evaluated: self.transitions_evaluated,
            total_reward: self.total_reward,
            collisions: self.collisions,
            path_len: self.path_len,
            trace_hash: self.trace_hash,
        }
    }
}

/// Planificador Monte Carlo de horizonte fijo (random shooting).
#[derive(Debug, Clone)]
pub struct MonteCarloPlanner {
    config: MonteCarloConfig,
}

impl MonteCarloPlanner {
    /// Crea un planificador Monte Carlo con la configuración dada.
    pub const fn new(config: MonteCarloConfig) -> Self {
        Self { config }
    }

    /// Devuelve la configuración del planificador.
    pub const fn config(&self) -> &MonteCarloConfig {
        &self.config
    }

    /// Planifica una acción sobre [`GridWorldModel`] usando [`BranchableSnapshot`] y [`PlanningContext`].
    ///
    /// Nunca muta el `model` original; cada rollout restaura el snapshot raíz sobre una rama aislada.
    pub fn plan_grid(
        &mut self,
        model: &GridWorldModel,
        rng: &mut dyn RngSource,
        plan_budget: PlanBudget,
        cancel_token: &CancelToken,
    ) -> Result<MonteCarloStepResult, ModelError> {
        let effective_max = self
            .config
            .max_transitions
            .min(plan_budget.max_transitions());
        let effective_budget = match plan_budget.deadline() {
            Some(deadline) => PlanBudget::new(effective_max, Some(deadline)),
            None => PlanBudget::with_transitions(effective_max),
        };
        let mut tracker = effective_budget.tracker();

        let fallback_action = self.config.fallback_action;
        let fallback_obj = fallback_action
            .to_action(model.action_space())
            .map_err(ModelError::InvalidAction)?;

        if cancel_token.is_cancelled() {
            return Ok(MonteCarloStepResult {
                plan_result: PlanResult::cancelled(0),
                chosen_action: fallback_action,
                used_fallback: true,
                simulations_run: 0,
                transitions_evaluated: 0,
                best_return: None,
                best_sequence: Vec::new(),
            });
        }

        if self.config.rollouts == 0 || self.config.horizon == 0 || model.terminal().is_terminal() {
            return Ok(MonteCarloStepResult {
                plan_result: PlanResult::no_valid_action(0),
                chosen_action: fallback_action,
                used_fallback: true,
                simulations_run: 0,
                transitions_evaluated: 0,
                best_return: None,
                best_sequence: Vec::new(),
            });
        }

        if tracker.is_exhausted() {
            return Ok(MonteCarloStepResult {
                plan_result: PlanResult::budget_exhausted(Some(fallback_obj), 0),
                chosen_action: fallback_action,
                used_fallback: true,
                simulations_run: 0,
                transitions_evaluated: 0,
                best_return: None,
                best_sequence: Vec::new(),
            });
        }

        let root_snapshot = model.snapshot().map_err(|err| ModelError::InvalidState {
            reason: match err {
                gogamen_core::SnapshotError::CorruptedChecksum { .. } => {
                    "corrupted root snapshot checksum"
                }
                _ => "failed to capture root snapshot for monte carlo",
            },
        })?;
        let mut branch = model.fork().map_err(|_| ModelError::InvalidState {
            reason: "failed to fork grid world model for monte carlo",
        })?;
        let dt = model.config().dt;
        let action_space = model.action_space().clone();

        let mut simulations_run = 0_usize;
        let mut budget_exhausted = false;
        let mut best_return: Option<f64> = None;
        let mut best_sequence: Vec<GridAction> = Vec::new();

        for _ in 0..self.config.rollouts {
            if cancel_token.is_cancelled() {
                return Ok(MonteCarloStepResult {
                    plan_result: PlanResult::cancelled(tracker.transitions_used()),
                    chosen_action: fallback_action,
                    used_fallback: true,
                    simulations_run,
                    transitions_evaluated: tracker.transitions_used(),
                    best_return,
                    best_sequence,
                });
            }

            if tracker.is_exhausted() {
                budget_exhausted = true;
                break;
            }

            branch
                .restore(&root_snapshot)
                .map_err(|_| ModelError::InvalidState {
                    reason: "failed to restore root snapshot before rollout",
                })?;

            let mut sim_state = branch.state().clone();
            let mut rollout_return = 0.0_f64;
            let mut rollout_actions = Vec::with_capacity(self.config.horizon);
            let mut started_rollout = false;
            let mut rollout_truncated = false;

            let mut sampled_indices = Vec::with_capacity(self.config.horizon);
            for _ in 0..self.config.horizon {
                sampled_indices.push((rng.next_u64() % (GRID_ACTION_CARDINALITY as u64)) as usize);
            }

            {
                let mut ctx = PlanningContext::new(&branch, rng);
                for action_idx in sampled_indices {
                    if !tracker.try_consume_transition() {
                        budget_exhausted = true;
                        rollout_truncated = true;
                        break;
                    }
                    started_rollout = true;

                    let grid_action = idx_to_grid_action(action_idx)?;
                    let action = grid_action
                        .to_action(&action_space)
                        .map_err(ModelError::InvalidAction)?;
                    let transition = ctx.simulate_step(&sim_state, &action, dt)?;
                    let step_reward = transition.next_state().values()[3];
                    rollout_return += step_reward;
                    rollout_actions.push(grid_action);
                    sim_state = transition.next_state().clone();

                    if transition.terminal().is_terminal() {
                        break;
                    }
                }
            }

            if started_rollout {
                simulations_run += 1;
            }

            if !rollout_truncated
                && !rollout_actions.is_empty()
                && best_return.is_none_or(|best| rollout_return > best)
            {
                best_return = Some(rollout_return);
                best_sequence = rollout_actions;
            }

            if budget_exhausted {
                break;
            }
        }

        let transitions_evaluated = tracker.transitions_used();

        if budget_exhausted || best_sequence.is_empty() {
            let plan_result = if budget_exhausted {
                PlanResult::budget_exhausted(Some(fallback_obj), transitions_evaluated)
            } else {
                PlanResult::no_valid_action(transitions_evaluated)
            };
            return Ok(MonteCarloStepResult {
                plan_result,
                chosen_action: fallback_action,
                used_fallback: true,
                simulations_run,
                transitions_evaluated,
                best_return,
                best_sequence,
            });
        }

        let chosen_action = best_sequence[0];
        let action = chosen_action
            .to_action(&action_space)
            .map_err(ModelError::InvalidAction)?;
        Ok(MonteCarloStepResult {
            plan_result: PlanResult::completed(action, transitions_evaluated),
            chosen_action,
            used_fallback: false,
            simulations_run,
            transitions_evaluated,
            best_return,
            best_sequence,
        })
    }
}

impl Planner for MonteCarloPlanner {
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        if request.is_cancelled() {
            return Ok(PlanResult::cancelled(0));
        }

        let Some(root_state) = request.state() else {
            return Ok(PlanResult::no_valid_action(0));
        };

        let effective_max = self
            .config
            .max_transitions
            .min(request.budget().max_transitions());
        let effective_budget = match request.budget().deadline() {
            Some(deadline) => PlanBudget::new(effective_max, Some(deadline)),
            None => PlanBudget::with_transitions(effective_max),
        };
        let mut tracker = effective_budget.tracker();

        let fallback_action = request.fallback().cloned().or_else(|| {
            self.config
                .fallback_action
                .to_action(request.action_space())
                .ok()
        });

        if self.config.rollouts == 0 || self.config.horizon == 0 {
            return Ok(PlanResult::no_valid_action(0));
        }

        if tracker.is_exhausted() {
            return Ok(PlanResult::budget_exhausted(fallback_action, 0));
        }

        let dt = StepDt::from_millis_f64(100.0).expect("100 ms must be valid StepDt");
        let ActionSpace::Discrete { cardinality } = *request.action_space() else {
            return Ok(PlanResult::no_valid_action(0));
        };
        if cardinality == 0 {
            return Ok(PlanResult::no_valid_action(0));
        }

        let mut best_return: Option<f64> = None;
        let mut best_first_action: Option<Action> = None;
        let mut budget_exhausted = false;

        for rollout_idx in 0..self.config.rollouts {
            if request.is_cancelled() {
                return Ok(PlanResult::cancelled(tracker.transitions_used()));
            }
            if tracker.is_exhausted() {
                budget_exhausted = true;
                break;
            }

            let mut sim_state = root_state.clone();
            let mut rollout_return = 0.0_f64;
            let mut first_action: Option<Action> = None;
            let mut truncated = false;

            for step_idx in 0..self.config.horizon {
                if !tracker.try_consume_transition() {
                    budget_exhausted = true;
                    truncated = true;
                    break;
                }

                // Selección determinista por paso usando el estado y el índice de rollout.
                let seed_mix = (rollout_idx as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((step_idx as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                    ^ sim_state.tick().as_u64();
                let action_idx = (seed_mix % (cardinality as u64)) as usize;
                let action = Action::Discrete(
                    DiscreteAction::new(action_idx, cardinality)
                        .map_err(ModelError::InvalidAction)?,
                );
                if first_action.is_none() {
                    first_action = Some(action.clone());
                }

                let transition = ctx.simulate_step(&sim_state, &action, dt)?;
                let step_reward = transition
                    .next_state()
                    .values()
                    .get(3)
                    .copied()
                    .unwrap_or(0.0);
                rollout_return += step_reward;
                sim_state = transition.next_state().clone();

                if transition.terminal().is_terminal() {
                    break;
                }
            }

            if !truncated
                && first_action.is_some()
                && best_return.is_none_or(|best| rollout_return > best)
            {
                best_return = Some(rollout_return);
                best_first_action = first_action;
            }

            if budget_exhausted {
                break;
            }
        }

        let used = tracker.transitions_used();
        if budget_exhausted {
            return Ok(PlanResult::budget_exhausted(fallback_action, used));
        }

        match best_first_action {
            Some(action) => Ok(PlanResult::completed(action, used)),
            None => Ok(PlanResult::no_valid_action(used)),
        }
    }
}

fn idx_to_grid_action(idx: usize) -> Result<GridAction, ModelError> {
    match idx {
        0 => Ok(GridAction::Stay),
        1 => Ok(GridAction::Up),
        2 => Ok(GridAction::Down),
        3 => Ok(GridAction::Left),
        4 => Ok(GridAction::Right),
        index => Err(ModelError::InvalidAction(
            DomainError::DiscreteActionOutOfRange {
                index,
                cardinality: GRID_ACTION_CARDINALITY,
            },
        )),
    }
}

fn map_env_error(err: EnvironmentError) -> DomainError {
    match err {
        EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
        EnvironmentError::InvalidAction(d) => d,
        _ => DomainError::InvalidTransition {
            reason: "failed to apply monte carlo action in grid environment",
        },
    }
}

fn map_model_error(err: ModelError) -> DomainError {
    match err {
        ModelError::InvalidAction(d) => d,
        ModelError::TransitionFailed(d) => d,
        ModelError::TerminalState => DomainError::AlreadyTerminal,
        _ => DomainError::InvalidTransition {
            reason: "monte carlo generative model planning failed",
        },
    }
}

fn observe_env(env: &GridWorldEnv) -> Result<gogamen_core::Observation, DomainError> {
    EnvironmentObserver::observe(env).map_err(|_| DomainError::InvalidTransition {
        reason: "failed to observe grid environment",
    })
}

/// Ejecuta el baseline Monte Carlo de horizonte fijo sobre [`GridWorldEnv`] usando [`GridWorldModel`].
pub fn run_monte_carlo_shooting_baseline(
    map: &GridMap,
    mut config: GridWorldConfig,
    seed: MasterSeed,
    mc_config: MonteCarloConfig,
    budget: BaselineBudget,
) -> Result<MonteCarloBaselineMetrics, DomainError> {
    config.map = map.clone();
    let mut env = GridWorldEnv::new(config.clone())?;
    let mut model = GridWorldModel::new(config, seed)?;

    let mut planner = MonteCarloPlanner::new(mc_config);
    let mut planning_rng = seed.derive(StreamKey::agent(0, 0));
    let cancel_token = CancelToken::new();
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(seed));

    let mut ticks_executed = 0_u64;
    let mut simulations_run = 0_u64;
    let mut transitions_evaluated = 0_u64;
    let mut fallbacks_used = 0_u64;
    let mut selected_actions = Vec::new();

    while ticks_executed < budget.max_ticks && transitions_evaluated < budget.max_transitions {
        let remaining_global = (budget.max_transitions - transitions_evaluated) as usize;
        // Reservamos 1 transición para el paso real en el entorno vivo.
        let step_plan_budget = PlanBudget::with_transitions(
            mc_config
                .max_transitions
                .min(remaining_global.saturating_sub(1)),
        );

        let step_res = planner
            .plan_grid(&model, &mut planning_rng, step_plan_budget, &cancel_token)
            .map_err(map_model_error)?;

        simulations_run += step_res.simulations_run as u64;
        transitions_evaluated += step_res.transitions_evaluated as u64;
        if step_res.used_fallback {
            fallbacks_used += 1;
        }

        if transitions_evaluated >= budget.max_transitions {
            break;
        }

        let obs = observe_env(&env)?;
        let action_idx = step_res.chosen_action as usize;
        let action = Action::Discrete(DiscreteAction::new(action_idx, GRID_ACTION_CARDINALITY)?);
        let outcome = env.apply(&action).map_err(map_env_error)?;
        model.step_in_place(&action).map_err(map_model_error)?;

        hasher.write_tick(env.tick());
        hasher.write_observation(&obs);
        hasher.write_u64(action_idx as u64);
        hasher.write_u64(step_res.simulations_run as u64);
        hasher.write_u64(u64::from(step_res.used_fallback));
        hasher.write_observation(outcome.observation());
        hasher.write_f64(env.last_reward());
        hasher.write_terminal(outcome.terminal());

        selected_actions.push(step_res.chosen_action);
        ticks_executed += 1;
        transitions_evaluated += 1;

        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let reached_goal = env.terminal() == Terminal::Terminated(TerminalReason::GoalReached);
    Ok(MonteCarloBaselineMetrics {
        seed,
        reached_goal,
        ticks_executed,
        simulations_run,
        transitions_evaluated,
        fallbacks_used,
        selected_actions,
        total_reward: env.total_reward(),
        collisions: env.collisions(),
        path_len: reached_goal.then_some(ticks_executed),
        trace_hash: hasher.finish(),
    })
}
