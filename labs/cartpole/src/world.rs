//! Entorno vivo (`CartPoleEnv`), modelo generativo ramificable (`CartPoleModel`) y ejecución headless.

use gogamen_core::{
    Action, ActionBounds, ActionSpace, Actuator, ApplyOutcome, BranchableSnapshot,
    DeterminismMetadata, DeterministicRng, DomainError, EnvironmentError, EnvironmentObserver,
    GenerativeModel, LiveEnvironment, MasterSeed, ModelError, Observation, RngSource, SimDuration,
    SimulationSnapshot, SnapshotError, State, StepDt, StreamKey, Terminal, Tick, TraceHasher,
    Transition, World,
};

use crate::controller::CartPolePolicy;
use crate::physics::{CartPoleConfig, CartPoleKinematics, integrate_step};

fn build_action_space(config: &CartPoleConfig) -> Result<ActionSpace, DomainError> {
    let bound = ActionBounds::new(-config.max_force, config.max_force)?;
    ActionSpace::continuous(vec![bound])
}

fn extract_force(action_space: &ActionSpace, action: &Action) -> Result<f64, DomainError> {
    action_space.validate(action)?;
    let values = action
        .as_continuous()
        .ok_or(DomainError::IncompatibleActionKind {
            expected: "continuous",
            actual: "discrete",
        })?;
    if values.len() != 1 {
        return Err(DomainError::ActionDimensionMismatch {
            expected: 1,
            actual: values.len(),
        });
    }
    Ok(values[0])
}

/// Entorno vivo del laboratorio CartPole (`World` + `LiveEnvironment`).
#[derive(Debug, Clone)]
pub struct CartPoleEnv {
    config: CartPoleConfig,
    action_space: ActionSpace,
    kinematics: CartPoleKinematics,
    tick: Tick,
    elapsed: SimDuration,
    terminal: Terminal,
}

impl CartPoleEnv {
    /// Crea un nuevo entorno [`CartPoleEnv`] inicializado en `config.initial_state`.
    pub fn new(config: CartPoleConfig) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = build_action_space(&config)?;
        let kinematics = config.initial_state;
        let terminal = kinematics.terminal_status(&config);
        Ok(Self {
            config,
            action_space,
            kinematics,
            tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
            terminal,
        })
    }

    /// Devuelve la configuración física del entorno.
    pub const fn config(&self) -> &CartPoleConfig {
        &self.config
    }

    /// Devuelve el estado cinemático actual.
    pub const fn kinematics(&self) -> CartPoleKinematics {
        self.kinematics
    }

    /// Devuelve el tick lógico actual del episodio.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve la duración simulada acumulada en el episodio.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Devuelve el estado de terminación actual del episodio.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    fn current_state(&self) -> Result<State, DomainError> {
        State::new(self.tick, self.elapsed, self.kinematics.to_vec())
    }

    fn reset_internal(&mut self) {
        self.kinematics = self.config.initial_state;
        self.tick = Tick::ZERO;
        self.elapsed = SimDuration::ZERO;
        self.terminal = self.kinematics.terminal_status(&self.config);
    }
}

impl World for CartPoleEnv {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn reset(&mut self) -> Result<State, DomainError> {
        self.reset_internal();
        self.current_state()
    }

    fn observe(&self, state: &State) -> Result<Observation, DomainError> {
        let kinematics = CartPoleKinematics::from_slice(state.values())?;
        Observation::new(state.tick(), state.elapsed(), kinematics.to_vec())
    }

    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        if self.terminal.is_terminal() {
            return Err(DomainError::AlreadyTerminal);
        }
        let force = extract_force(&self.action_space, action)?;
        let prev_tick = self.tick;
        let next_kinematics = integrate_step(&self.config, self.kinematics, force, dt)?;
        let next_tick = self.tick.next()?;
        let next_elapsed = self.elapsed.checked_add_dt(dt)?;
        let next_terminal = next_kinematics.terminal_status(&self.config);

        let next_state = State::new(next_tick, next_elapsed, next_kinematics.to_vec())?;
        let obs = World::observe(self, &next_state)?;
        let transition = Transition::new(prev_tick, dt, next_state, obs, next_terminal)?;

        self.kinematics = next_kinematics;
        self.tick = next_tick;
        self.elapsed = next_elapsed;
        self.terminal = next_terminal;

        Ok(transition)
    }
}

impl EnvironmentObserver for CartPoleEnv {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        Observation::new(self.tick, self.elapsed, self.kinematics.to_vec()).map_err(|_| {
            EnvironmentError::ObservationFailed {
                reason: "cartpole observation contains non-finite values",
            }
        })
    }
}

impl Actuator for CartPoleEnv {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        let transition = World::step(self, action, self.config.dt).map_err(|err| match err {
            DomainError::AlreadyTerminal => EnvironmentError::AlreadyTerminal,
            DomainError::DiscreteActionOutOfRange { .. }
            | DomainError::ContinuousActionOutOfRange { .. }
            | DomainError::ActionDimensionMismatch { .. }
            | DomainError::IncompatibleActionKind { .. } => EnvironmentError::InvalidAction(err),
            _ => EnvironmentError::ActuationFailed {
                reason: "failed to integrate cartpole step",
            },
        })?;
        Ok(ApplyOutcome::new(
            transition.observation().clone(),
            transition.terminal(),
        ))
    }
}

impl LiveEnvironment for CartPoleEnv {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        self.reset_internal();
        EnvironmentObserver::observe(self)
    }
}

/// Modelo generativo puro y ramificable de CartPole (`GenerativeModel` + `BranchableSnapshot`).
#[derive(Debug, Clone)]
pub struct CartPoleModel {
    config: CartPoleConfig,
    action_space: ActionSpace,
    state: State,
    rng: DeterministicRng,
    terminal: Terminal,
}

impl CartPoleModel {
    /// Crea un nuevo [`CartPoleModel`] inicializado en `config.initial_state`.
    pub fn new(config: CartPoleConfig, seed: MasterSeed) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = build_action_space(&config)?;
        let terminal = config.initial_state.terminal_status(&config);
        let state = State::new(Tick::ZERO, SimDuration::ZERO, config.initial_state.to_vec())?;
        Ok(Self {
            config,
            action_space,
            state,
            rng: seed.derive(StreamKey::Episode(0)),
            terminal,
        })
    }

    /// Devuelve la configuración física del modelo.
    pub const fn config(&self) -> &CartPoleConfig {
        &self.config
    }

    /// Devuelve el [`State`] interno actual del modelo.
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// Devuelve la cinemática actual decodificada del estado interno.
    pub fn kinematics(&self) -> CartPoleKinematics {
        CartPoleKinematics::from_slice(self.state.values()).expect("internal model state is valid")
    }

    /// Devuelve el estado terminal interno del modelo.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Reinicia el estado interno del modelo a `config.initial_state`.
    pub fn reset(&mut self) -> Result<&State, DomainError> {
        self.state = State::new(
            Tick::ZERO,
            SimDuration::ZERO,
            self.config.initial_state.to_vec(),
        )?;
        self.terminal = self.config.initial_state.terminal_status(&self.config);
        Ok(&self.state)
    }

    /// Avanza el estado interno del modelo un paso usando `self.config.dt`.
    pub fn step_in_place(&mut self, action: &Action) -> Result<Transition, ModelError> {
        if self.terminal.is_terminal() {
            return Err(ModelError::TerminalState);
        }
        let mut rng = self.rng.clone();
        let transition =
            GenerativeModel::step(self, &self.state, action, self.config.dt, &mut rng)?;
        self.rng = rng;
        self.state = transition.next_state().clone();
        self.terminal = transition.terminal();
        Ok(transition)
    }
}

impl GenerativeModel for CartPoleModel {
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
        let force = extract_force(&self.action_space, action).map_err(ModelError::InvalidAction)?;
        let kinematics = CartPoleKinematics::from_slice(state.values()).map_err(|_| {
            ModelError::InvalidState {
                reason: "cartpole state must have 4 finite elements [x, x_dot, theta, theta_dot]",
            }
        })?;
        if kinematics.terminal_status(&self.config).is_terminal() {
            return Err(ModelError::TerminalState);
        }

        let next_kinematics = integrate_step(&self.config, kinematics, force, dt)
            .map_err(ModelError::TransitionFailed)?;
        let next_terminal = next_kinematics.terminal_status(&self.config);
        let next_state = state
            .advance(dt, next_kinematics.to_vec())
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            next_kinematics.to_vec(),
        )
        .map_err(ModelError::TransitionFailed)?;

        Transition::new(state.tick(), dt, next_state, obs, next_terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

impl BranchableSnapshot for CartPoleModel {
    fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
        Ok(SimulationSnapshot::new(
            self.state.clone(),
            &self.rng,
            self.terminal,
        ))
    }

    fn fork(&self) -> Result<Self, SnapshotError> {
        Ok(self.clone())
    }

    fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
        snapshot.verify()?;
        CartPoleKinematics::from_slice(snapshot.state().values()).map_err(|_| {
            SnapshotError::Domain(DomainError::InvalidTransition {
                reason: "snapshot state is incompatible with cartpole kinematics",
            })
        })?;
        self.state = snapshot.state().clone();
        self.rng = snapshot.rng();
        self.terminal = snapshot.terminal();
        Ok(())
    }
}

/// Resumen determinista de una ejecución headless de CartPole.
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleEpisodeSummary {
    /// Número de ticks ejecutados antes de terminar o alcanzar el horizonte.
    pub ticks_executed: u64,
    /// Tiempo simulado total acumulado.
    pub elapsed: SimDuration,
    /// Estado cinemático final.
    pub final_kinematics: CartPoleKinematics,
    /// Máximo desplazamiento absoluto `|x|` observado durante el episodio.
    pub max_abs_x: f64,
    /// Máximo ángulo absoluto `|theta|` observado durante el episodio.
    pub max_abs_theta: f64,
    /// Estado de terminación final.
    pub terminal: Terminal,
    /// Hash determinista de traza de 64 bits.
    pub trace_hash: u64,
}

impl CartPoleEpisodeSummary {
    /// Devuelve `trace_hash` formateado en hexadecimal de 16 dígitos.
    pub fn trace_hash_hex(&self) -> String {
        format!("{:016x}", self.trace_hash)
    }
}

/// Ejecuta un episodio headless reproducible en [`CartPoleEnv`] bajo una [`CartPolePolicy`].
pub fn run_cartpole_headless(
    config: CartPoleConfig,
    ticks: u64,
    seed: MasterSeed,
    policy: &mut dyn CartPolePolicy,
) -> Result<CartPoleEpisodeSummary, DomainError> {
    let mut env = CartPoleEnv::new(config)?;
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(seed));

    let init = env.kinematics();
    let mut max_abs_x = init.x.abs();
    let mut max_abs_theta = init.theta.abs();
    let mut ticks_executed = 0;

    if !env.terminal().is_terminal() {
        for _ in 0..ticks {
            let obs =
                EnvironmentObserver::observe(&env).map_err(|_| DomainError::InvalidTransition {
                    reason: "failed to observe cartpole environment",
                })?;
            let action = policy.select_action(&obs, Actuator::action_space(&env))?;
            let outcome = env.apply(&action).map_err(|err| match err {
                EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
                EnvironmentError::InvalidAction(d) => d,
                _ => DomainError::InvalidTransition {
                    reason: "failed to apply cartpole action",
                },
            })?;

            let kin = env.kinematics();
            max_abs_x = max_abs_x.max(kin.x.abs());
            max_abs_theta = max_abs_theta.max(kin.theta.abs());

            hasher.write_tick(env.tick());
            hasher.write_observation(&obs);
            if let Some(vals) = action.as_continuous() {
                for &v in vals {
                    hasher.write_f64(v);
                }
            }
            hasher.write_observation(outcome.observation());
            hasher.write_terminal(outcome.terminal());

            ticks_executed += 1;
            if outcome.terminal().is_terminal() {
                break;
            }
        }
    }

    Ok(CartPoleEpisodeSummary {
        ticks_executed,
        elapsed: env.elapsed(),
        final_kinematics: env.kinematics(),
        max_abs_x,
        max_abs_theta,
        terminal: env.terminal(),
        trace_hash: hasher.finish(),
    })
}
