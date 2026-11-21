//! Mundo 1D reproducible (`LineWorld1D` / `World1D`) y planificador de horizonte recedente.
//!
//! Separa estrictamente el entorno vivo ([`LineWorld1D`], que implementa
//! [`LiveEnvironment`]) del modelo generativo puro ([`LineWorld1DModel`], que
//! implementa [`GenerativeModel`]).

use gogamen_core::{
    Action, ActionSpace, Actuator, ApplyOutcome, DiscreteAction, DomainError, EnvironmentError,
    EnvironmentObserver, GenerativeModel, LiveEnvironment, ModelError, Observation, PlanBudget,
    PlanRequest, PlanResult, Planner, PlanningContext, RngSource, SimDuration, State, StepDt,
    Terminal, TerminalReason, Tick, Transition,
};

/// Alias conveniente para [`LineWorld1D`].
pub type World1D = LineWorld1D;

/// Índice de acción discreta para permanecer quieto (`delta = 0.0`).
pub const ACTION_STAY: usize = 0;
/// Índice de acción discreta para desplazarse a la izquierda (`delta = -step_size`).
pub const ACTION_LEFT: usize = 1;
/// Índice de acción discreta para desplazarse a la derecha (`delta = +step_size`).
pub const ACTION_RIGHT: usize = 2;

/// Configuración acotada y determinista de [`LineWorld1D`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LineWorld1DConfig {
    /// Límite inferior del eje 1D.
    pub min_pos: f64,
    /// Límite superior del eje 1D.
    pub max_pos: f64,
    /// Posición inicial al iniciar o reiniciar un episodio.
    pub initial_pos: f64,
    /// Posición objetivo del agente en el mundo 1D.
    pub goal_pos: f64,
    /// Magnitud de desplazamiento por paso discreto.
    pub step_size: f64,
    /// Paso temporal físico por tick.
    pub dt: StepDt,
    /// Si es `true`, el episodio termina al alcanzar la vecindad de `goal_pos`.
    pub terminate_on_goal: bool,
}

impl Default for LineWorld1DConfig {
    fn default() -> Self {
        Self {
            min_pos: -50.0,
            max_pos: 50.0,
            initial_pos: 0.0,
            goal_pos: 25.0,
            step_size: 1.0,
            dt: StepDt::from_millis_f64(100.0).expect("100 ms must be valid StepDt"),
            terminate_on_goal: false,
        }
    }
}

impl LineWorld1DConfig {
    /// Valida los parámetros geométricos de la configuración.
    pub fn validate(&self) -> Result<(), DomainError> {
        if !self.min_pos.is_finite() || !self.max_pos.is_finite() || self.min_pos >= self.max_pos {
            return Err(DomainError::InvalidActionBounds {
                min: self.min_pos,
                max: self.max_pos,
            });
        }
        if !self.initial_pos.is_finite() {
            return Err(DomainError::NonFiniteValue {
                field: "world1d.initial_pos",
                value: self.initial_pos,
            });
        }
        if !self.goal_pos.is_finite() {
            return Err(DomainError::NonFiniteValue {
                field: "world1d.goal_pos",
                value: self.goal_pos,
            });
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(DomainError::NonFiniteValue {
                field: "world1d.step_size",
                value: self.step_size,
            });
        }
        Ok(())
    }

    fn clamp_pos(&self, pos: f64) -> f64 {
        pos.clamp(self.min_pos, self.max_pos)
    }

    fn is_goal_reached(&self, pos: f64) -> bool {
        (pos - self.goal_pos).abs() <= self.step_size * 0.25
    }

    fn delta_for_action(&self, action: &Action) -> f64 {
        match action.as_discrete().map(DiscreteAction::index) {
            Some(ACTION_LEFT) => -self.step_size,
            Some(ACTION_RIGHT) => self.step_size,
            _ => 0.0,
        }
    }
}

/// Entorno vivo 1D acotado y reproducible (`LiveEnvironment`).
#[derive(Debug, Clone)]
pub struct LineWorld1D {
    config: LineWorld1DConfig,
    action_space: ActionSpace,
    tick: Tick,
    elapsed: SimDuration,
    position: f64,
    terminal: Terminal,
    apply_count: u64,
}

impl LineWorld1D {
    /// Crea un nuevo entorno [`LineWorld1D`] con la configuración dada.
    pub fn new(config: LineWorld1DConfig) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = ActionSpace::discrete(3)?;
        Ok(Self {
            position: config.clamp_pos(config.initial_pos),
            config,
            action_space,
            tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
            terminal: Terminal::Ongoing,
            apply_count: 0,
        })
    }

    /// Devuelve la configuración del mundo 1D.
    pub const fn config(&self) -> LineWorld1DConfig {
        self.config
    }

    /// Devuelve la posición actual del agente en el eje 1D.
    pub const fn position(&self) -> f64 {
        self.position
    }

    /// Devuelve el tick actual del entorno.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve el estado terminal actual del entorno.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Número total de veces que se aplicó una acción mediante [`Actuator::apply`].
    pub const fn apply_count(&self) -> u64 {
        self.apply_count
    }

    /// Devuelve la acción segura de respaldo (`Stay`) en el [`ActionSpace`] del mundo 1D.
    pub fn safe_fallback_action(&self) -> Action {
        Action::discrete_in_space(ACTION_STAY, &self.action_space)
            .expect("ACTION_STAY=0 is always valid in discrete(3)")
    }

    /// Construye un [`LineWorld1DModel`] puro alineado con la física de este entorno.
    pub fn model(&self) -> LineWorld1DModel {
        LineWorld1DModel {
            config: self.config,
            action_space: self.action_space.clone(),
        }
    }
}

impl EnvironmentObserver for LineWorld1D {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        Observation::new(self.tick, self.elapsed, vec![self.position]).map_err(|_| {
            EnvironmentError::ObservationFailed {
                reason: "non-finite 1D position reading",
            }
        })
    }
}

impl Actuator for LineWorld1D {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        if self.terminal.is_terminal() {
            return Err(EnvironmentError::AlreadyTerminal);
        }
        self.action_space
            .validate(action)
            .map_err(EnvironmentError::InvalidAction)?;

        let delta = self.config.delta_for_action(action);
        self.tick = self
            .tick
            .next()
            .map_err(|_| EnvironmentError::ActuationFailed {
                reason: "tick overflow",
            })?;
        self.elapsed = self.elapsed.checked_add_dt(self.config.dt).map_err(|_| {
            EnvironmentError::ActuationFailed {
                reason: "elapsed overflow",
            }
        })?;
        self.position = self.config.clamp_pos(self.position + delta);
        self.apply_count = self.apply_count.saturating_add(1);

        if self.config.terminate_on_goal && self.config.is_goal_reached(self.position) {
            self.terminal = Terminal::Terminated(TerminalReason::GoalReached);
        }

        let obs = self.observe()?;
        Ok(ApplyOutcome::new(obs, self.terminal))
    }
}

impl LiveEnvironment for LineWorld1D {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        self.tick = Tick::ZERO;
        self.elapsed = SimDuration::ZERO;
        self.position = self.config.clamp_pos(self.config.initial_pos);
        self.terminal = Terminal::Ongoing;
        self.apply_count = 0;
        self.observe()
    }
}

/// Modelo generativo puro del mundo 1D (`GenerativeModel`).
#[derive(Debug, Clone)]
pub struct LineWorld1DModel {
    config: LineWorld1DConfig,
    action_space: ActionSpace,
}

impl LineWorld1DModel {
    /// Crea un modelo generativo para la configuración dada.
    pub fn new(config: LineWorld1DConfig) -> Result<Self, DomainError> {
        config.validate()?;
        Ok(Self {
            config,
            action_space: ActionSpace::discrete(3)?,
        })
    }
}

impl GenerativeModel for LineWorld1DModel {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        _rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.action_space
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        if state.values().len() != 1 {
            return Err(ModelError::InvalidState {
                reason: "LineWorld1DModel expects 1D scalar state",
            });
        }

        let delta = self.config.delta_for_action(action);
        let next_pos = self.config.clamp_pos(state.values()[0] + delta);
        let next_state = state
            .advance(dt, vec![next_pos])
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            next_state.values().to_vec(),
        )
        .map_err(ModelError::TransitionFailed)?;
        let terminal = if self.config.terminate_on_goal && self.config.is_goal_reached(next_pos) {
            Terminal::Terminated(TerminalReason::GoalReached)
        } else {
            Terminal::Ongoing
        };

        Transition::new(state.tick(), dt, next_state, obs, terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

/// Planificador determinista de horizonte recedente para dominios 1D.
///
/// Evalúa cada acción candidata simulando rollouts de longitud `horizon` a través
/// de [`PlanningContext`] y selecciona la acción que minimiza la distancia a `goal_pos`.
#[derive(Debug, Clone)]
pub struct RecedingHorizonPlanner {
    horizon: usize,
    goal_pos: f64,
    dt: StepDt,
}

impl RecedingHorizonPlanner {
    /// Crea un planificador con horizonte `horizon`, objetivo `goal_pos` y paso `dt`.
    pub fn new(horizon: usize, goal_pos: f64, dt: StepDt) -> Self {
        Self {
            horizon: horizon.max(1),
            goal_pos,
            dt,
        }
    }

    /// Crea un planificador configurado a partir de [`LineWorld1DConfig`].
    pub fn from_config(config: &LineWorld1DConfig, horizon: usize) -> Self {
        Self::new(horizon, config.goal_pos, config.dt)
    }

    fn state_from_request(&self, request: &PlanRequest) -> Result<State, ModelError> {
        if let Some(state) = request.state() {
            return Ok(state.clone());
        }
        if let Some(obs) = request.observation() {
            return State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
                .map_err(ModelError::TransitionFailed);
        }
        Err(ModelError::InvalidState {
            reason: "PlanRequest has neither state nor observation",
        })
    }

    fn candidate_actions(&self, space: &ActionSpace) -> Vec<Action> {
        match space {
            ActionSpace::Discrete { cardinality } => (0..*cardinality)
                .filter_map(|idx| Action::discrete_in_space(idx, space).ok())
                .collect(),
            ActionSpace::Continuous { bounds } => {
                let zeros: Vec<f64> = bounds
                    .iter()
                    .map(|b| 0.0_f64.clamp(b.min(), b.max()))
                    .collect();
                Action::continuous_in_space(zeros, space)
                    .into_iter()
                    .collect()
            }
        }
    }
}

impl Planner for RecedingHorizonPlanner {
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        if request.is_cancelled() {
            return Ok(PlanResult::cancelled(0));
        }

        let mut tracker = PlanBudget::new(
            request.budget().max_transitions(),
            request.budget().deadline(),
        )
        .tracker();

        if tracker.is_exhausted() {
            return Ok(PlanResult::budget_exhausted(
                request.fallback().cloned(),
                tracker.transitions_used(),
            ));
        }

        let root_state = self.state_from_request(request)?;
        let candidates = self.candidate_actions(request.action_space());
        if candidates.is_empty() {
            return Ok(PlanResult::no_valid_action(0));
        }

        let mut best_action: Option<Action> = None;
        let mut best_cost = f64::INFINITY;

        for first_action in &candidates {
            if request.is_cancelled() {
                return Ok(PlanResult::cancelled(tracker.transitions_used()));
            }
            if !tracker.try_consume_transition() {
                return Ok(PlanResult::budget_exhausted(
                    request.fallback().cloned(),
                    tracker.transitions_used(),
                ));
            }

            let first_transition = ctx.simulate_step(&root_state, first_action, self.dt)?;
            let mut rollout_state = first_transition.next_state().clone();
            let mut total_cost = (rollout_state.values()[0] - self.goal_pos).abs();

            // Horizonte recedente: simula los pasos restantes repitiendo la acción greedy local.
            for _step in 1..self.horizon {
                if request.is_cancelled() {
                    return Ok(PlanResult::cancelled(tracker.transitions_used()));
                }
                if first_transition.terminal().is_terminal() {
                    break;
                }
                if !tracker.try_consume_transition() {
                    return Ok(PlanResult::budget_exhausted(
                        request.fallback().cloned(),
                        tracker.transitions_used(),
                    ));
                }

                let mut step_best_state = None;
                let mut step_best_cost = f64::INFINITY;
                for follow_action in &candidates {
                    let t = ctx.simulate_step(&rollout_state, follow_action, self.dt)?;
                    let pos = t.next_state().values()[0];
                    let dist = (pos - self.goal_pos).abs();
                    if dist < step_best_cost {
                        step_best_cost = dist;
                        step_best_state = Some(t.next_state().clone());
                    }
                    if !tracker.try_consume_transition() {
                        break;
                    }
                }
                if let Some(next_s) = step_best_state {
                    rollout_state = next_s;
                    total_cost += step_best_cost;
                } else {
                    break;
                }
            }

            if total_cost < best_cost {
                best_cost = total_cost;
                best_action = Some(first_action.clone());
            }
        }

        match best_action {
            Some(action) => Ok(PlanResult::completed(action, tracker.transitions_used())),
            None => Ok(PlanResult::no_valid_action(tracker.transitions_used())),
        }
    }
}
