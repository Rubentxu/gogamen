//! Mundo Grid discreto (`GridWorldEnv` y `GridWorldModel`) y corredor de episodios deterministas.

use gogamen_core::{
    Action, ActionSpace, Actuator, ApplyOutcome, BranchableSnapshot, DeterminismMetadata,
    DeterministicRng, DiscreteAction, DomainError, EnvironmentError, EnvironmentObserver,
    GenerativeModel, LiveEnvironment, MasterSeed, ModelError, Observation, RngSource, SimDuration,
    SimulationSnapshot, SnapshotError, State, StepDt, StreamKey, Terminal, TerminalReason, Tick,
    TraceHasher, Transition, World,
};

use crate::map::{CellKind, GridMap, GridPos};

/// Índice discreto para permanecer en la celda actual.
pub const ACTION_STAY: usize = 0;
/// Índice discreto para moverse una celda hacia arriba (`y - 1`).
pub const ACTION_UP: usize = 1;
/// Índice discreto para moverse una celda hacia abajo (`y + 1`).
pub const ACTION_DOWN: usize = 2;
/// Índice discreto para moverse una celda hacia la izquierda (`x - 1`).
pub const ACTION_LEFT: usize = 3;
/// Índice discreto para moverse una celda hacia la derecha (`x + 1`).
pub const ACTION_RIGHT: usize = 4;

/// Número de acciones discretas del mundo Grid (`Stay`, `Up`, `Down`, `Left`, `Right`).
pub const GRID_ACTION_CARDINALITY: usize = 5;

/// Acción discreta del mundo Grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GridAction {
    /// Permanecer en la posición actual (`(0, 0)`).
    Stay = 0,
    /// Moverse hacia arriba (`(0, -1)`).
    Up = 1,
    /// Moverse hacia abajo (`(0, +1)`).
    Down = 2,
    /// Moverse hacia la izquierda (`(-1, 0)`).
    Left = 3,
    /// Moverse hacia la derecha (`(+1, 0)`).
    Right = 4,
}

impl GridAction {
    /// Convierte una [`Action`] genérica en [`GridAction`].
    pub fn from_action(action: &Action) -> Result<Self, DomainError> {
        let discrete = action
            .as_discrete()
            .ok_or(DomainError::IncompatibleActionKind {
                expected: "discrete",
                actual: "continuous",
            })?;
        match discrete.index() {
            ACTION_STAY => Ok(Self::Stay),
            ACTION_UP => Ok(Self::Up),
            ACTION_DOWN => Ok(Self::Down),
            ACTION_LEFT => Ok(Self::Left),
            ACTION_RIGHT => Ok(Self::Right),
            index => Err(DomainError::DiscreteActionOutOfRange {
                index,
                cardinality: GRID_ACTION_CARDINALITY,
            }),
        }
    }

    /// Convierte esta [`GridAction`] en [`Action`] validada contra `space`.
    pub fn to_action(self, space: &ActionSpace) -> Result<Action, DomainError> {
        Action::discrete_in_space(self as usize, space)
    }

    /// Desplazamiento `(dx, dy)` asociado a esta acción.
    pub const fn delta(self) -> (isize, isize) {
        match self {
            Self::Stay => (0, 0),
            Self::Up => (0, -1),
            Self::Down => (0, 1),
            Self::Left => (-1, 0),
            Self::Right => (1, 0),
        }
    }
}

/// Modo de observación del mundo Grid (completa o parcial desactivable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ObservationMode {
    /// Observación completa: `[agent_x, agent_y, goal_x, goal_y, ...celdas del mapa]`.
    #[default]
    Full,
    /// Observación parcial local: `[agent_x, agent_y, radius, ...ventana (2r+1)x(2r+1)]`.
    Partial {
        /// Radio de visión en celdas alrededor del agente (`0` = solo celda actual).
        radius: usize,
    },
}

/// Configuración explícita de [`GridWorldEnv`] y [`GridWorldModel`].
#[derive(Debug, Clone, PartialEq)]
pub struct GridWorldConfig {
    /// Mapa discreto con obstáculos, inicio y meta.
    pub map: GridMap,
    /// Modo de observación (`Full` o `Partial { radius }`).
    pub observation_mode: ObservationMode,
    /// Paso temporal físico por tick en segundos.
    pub dt: StepDt,
    /// Recompensa por paso válido sobre celda libre.
    pub step_reward: f64,
    /// Recompensa (penalización) al colisionar contra muro o borde.
    pub wall_collision_reward: f64,
    /// Recompensa positiva al alcanzar `CellKind::Goal`.
    pub goal_reward: f64,
    /// Recompensa al pisar `CellKind::Hazard`.
    pub hazard_reward: f64,
    /// Si es `true`, pisar `CellKind::Hazard` termina el episodio con `Failure`.
    pub hazard_is_terminal: bool,
}

impl Default for GridWorldConfig {
    fn default() -> Self {
        Self {
            map: GridMap::cul_de_sac_map(),
            observation_mode: ObservationMode::Full,
            dt: StepDt::from_millis_f64(100.0).expect("100 ms must be valid StepDt"),
            step_reward: -0.05,
            wall_collision_reward: -0.5,
            goal_reward: 10.0,
            hazard_reward: -5.0,
            hazard_is_terminal: true,
        }
    }
}

impl GridWorldConfig {
    /// Valida que todas las recompensas configuradas sean finitas.
    pub fn validate(&self) -> Result<(), DomainError> {
        for (field, value) in [
            ("grid.step_reward", self.step_reward),
            ("grid.wall_collision_reward", self.wall_collision_reward),
            ("grid.goal_reward", self.goal_reward),
            ("grid.hazard_reward", self.hazard_reward),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFiniteValue { field, value });
            }
        }
        Ok(())
    }

    /// Configura el modo de observación (`Some(radius)` activa parcial; `None` la desactiva).
    pub const fn with_partial_observation(mut self, radius: Option<usize>) -> Self {
        self.observation_mode = match radius {
            Some(r) => ObservationMode::Partial { radius: r },
            None => ObservationMode::Full,
        };
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct StepOutcomeData {
    next_pos: GridPos,
    reward: f64,
    collided: bool,
    terminal: Terminal,
}

fn step_grid_dynamics(
    config: &GridWorldConfig,
    pos: GridPos,
    action: GridAction,
) -> StepOutcomeData {
    let (dx, dy) = action.delta();
    let nx = pos.x as isize + dx;
    let ny = pos.y as isize + dy;

    let candidate = if nx >= 0 && ny >= 0 {
        let cand = GridPos::new(nx as usize, ny as usize);
        if config.map.is_passable(cand) {
            Some(cand)
        } else {
            None
        }
    } else {
        None
    };

    match candidate {
        None => StepOutcomeData {
            next_pos: pos,
            reward: config.wall_collision_reward,
            collided: true,
            terminal: Terminal::Ongoing,
        },
        Some(next_pos) => match config.map.cell(next_pos) {
            Some(CellKind::Goal) => StepOutcomeData {
                next_pos,
                reward: config.goal_reward,
                collided: false,
                terminal: Terminal::Terminated(TerminalReason::GoalReached),
            },
            Some(CellKind::Hazard) => StepOutcomeData {
                next_pos,
                reward: config.hazard_reward,
                collided: false,
                terminal: if config.hazard_is_terminal {
                    Terminal::Terminated(TerminalReason::Failure)
                } else {
                    Terminal::Ongoing
                },
            },
            Some(CellKind::Empty) | Some(CellKind::Wall) | None => StepOutcomeData {
                next_pos,
                reward: config.step_reward,
                collided: false,
                terminal: Terminal::Ongoing,
            },
        },
    }
}

fn build_observation(
    config: &GridWorldConfig,
    pos: GridPos,
    tick: Tick,
    elapsed: SimDuration,
) -> Result<Observation, DomainError> {
    let features = match config.observation_mode {
        ObservationMode::Full => {
            let goal = config.map.goal();
            let mut vec = Vec::with_capacity(4 + config.map.cells().len());
            vec.push(pos.x as f64);
            vec.push(pos.y as f64);
            vec.push(goal.x as f64);
            vec.push(goal.y as f64);
            for cell in config.map.cells() {
                vec.push(cell.as_code());
            }
            vec
        }
        ObservationMode::Partial { radius } => {
            let side = 2 * radius + 1;
            let mut vec = Vec::with_capacity(3 + side * side);
            vec.push(pos.x as f64);
            vec.push(pos.y as f64);
            vec.push(radius as f64);
            let r = radius as isize;
            let cx = pos.x as isize;
            let cy = pos.y as isize;
            for dy in -r..=r {
                for dx in -r..=r {
                    let tx = cx + dx;
                    let ty = cy + dy;
                    if tx < 0 || ty < 0 {
                        vec.push(CellKind::OUT_OF_BOUNDS_CODE);
                    } else {
                        let target = GridPos::new(tx as usize, ty as usize);
                        let code = config
                            .map
                            .cell(target)
                            .map_or(CellKind::OUT_OF_BOUNDS_CODE, CellKind::as_code);
                        vec.push(code);
                    }
                }
            }
            vec
        }
    };
    Observation::new(tick, elapsed, features)
}

fn encode_state(pos: GridPos, total_reward: f64, last_reward: f64, collisions: u64) -> Vec<f64> {
    vec![
        pos.x as f64,
        pos.y as f64,
        total_reward,
        last_reward,
        collisions as f64,
    ]
}

fn decode_state(state: &State, map: &GridMap) -> Result<(GridPos, f64, u64), ModelError> {
    let vals = state.values();
    if vals.len() != 5 {
        return Err(ModelError::InvalidState {
            reason: "grid state must have 5 elements [x, y, total_reward, last_reward, collisions]",
        });
    }
    let fx = vals[0];
    let fy = vals[1];
    let total_reward = vals[2];
    let fcol = vals[4];

    if fx < 0.0 || fy < 0.0 || fx.fract() != 0.0 || fy.fract() != 0.0 {
        return Err(ModelError::InvalidState {
            reason: "grid state coordinates must be non-negative integers",
        });
    }
    let pos = GridPos::new(fx as usize, fy as usize);
    if !map.is_passable(pos) {
        return Err(ModelError::InvalidState {
            reason: "grid state position is out of bounds or inside a wall",
        });
    }
    if fcol < 0.0 || fcol.fract() != 0.0 {
        return Err(ModelError::InvalidState {
            reason: "grid state collision count must be a non-negative integer",
        });
    }
    Ok((pos, total_reward, fcol as u64))
}

/// Entorno vivo del mundo Grid (`World` + `LiveEnvironment`).
#[derive(Debug, Clone)]
pub struct GridWorldEnv {
    config: GridWorldConfig,
    action_space: ActionSpace,
    position: GridPos,
    tick: Tick,
    elapsed: SimDuration,
    total_reward: f64,
    last_reward: f64,
    collisions: u64,
    terminal: Terminal,
}

impl GridWorldEnv {
    /// Crea un nuevo entorno [`GridWorldEnv`] posicionado en `config.map.start()`.
    pub fn new(config: GridWorldConfig) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
        let position = config.map.start();
        Ok(Self {
            config,
            action_space,
            position,
            tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
            total_reward: 0.0,
            last_reward: 0.0,
            collisions: 0,
            terminal: Terminal::Ongoing,
        })
    }

    /// Devuelve la configuración del entorno.
    pub const fn config(&self) -> &GridWorldConfig {
        &self.config
    }

    /// Posición actual del agente en la rejilla.
    pub const fn position(&self) -> GridPos {
        self.position
    }

    /// Tick lógico actual del episodio.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Recompensa acumulada en el episodio actual.
    pub const fn total_reward(&self) -> f64 {
        self.total_reward
    }

    /// Recompensa obtenida en el último paso ejecutado.
    pub const fn last_reward(&self) -> f64 {
        self.last_reward
    }

    /// Número de colisiones contra muros o bordes en el episodio actual.
    pub const fn collisions(&self) -> u64 {
        self.collisions
    }

    /// Estado de terminación actual del episodio.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Construye el [`State`] actual del entorno.
    pub fn current_state(&self) -> Result<State, DomainError> {
        State::new(
            self.tick,
            self.elapsed,
            encode_state(
                self.position,
                self.total_reward,
                self.last_reward,
                self.collisions,
            ),
        )
    }

    fn reset_internal(&mut self) {
        self.position = self.config.map.start();
        self.tick = Tick::ZERO;
        self.elapsed = SimDuration::ZERO;
        self.total_reward = 0.0;
        self.last_reward = 0.0;
        self.collisions = 0;
        self.terminal = Terminal::Ongoing;
    }

    fn advance_step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        self.action_space.validate(action)?;
        if self.terminal.is_terminal() {
            return Err(DomainError::AlreadyTerminal);
        }

        let grid_action = GridAction::from_action(action)?;
        let prev_tick = self.tick;
        let outcome = step_grid_dynamics(&self.config, self.position, grid_action);

        let next_tick = prev_tick.next()?;
        let next_elapsed = self.elapsed.checked_add_dt(dt)?;
        let next_collisions = self.collisions + u64::from(outcome.collided);
        let next_total_reward = self.total_reward + outcome.reward;

        let next_state = State::new(
            next_tick,
            next_elapsed,
            encode_state(
                outcome.next_pos,
                next_total_reward,
                outcome.reward,
                next_collisions,
            ),
        )?;
        let obs = build_observation(&self.config, outcome.next_pos, next_tick, next_elapsed)?;
        let transition = Transition::new(prev_tick, dt, next_state, obs, outcome.terminal)?;

        self.position = outcome.next_pos;
        self.tick = next_tick;
        self.elapsed = next_elapsed;
        self.total_reward = next_total_reward;
        self.last_reward = outcome.reward;
        self.collisions = next_collisions;
        self.terminal = outcome.terminal;

        Ok(transition)
    }
}

impl World for GridWorldEnv {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn reset(&mut self) -> Result<State, DomainError> {
        self.reset_internal();
        self.current_state()
    }

    fn observe(&self, state: &State) -> Result<Observation, DomainError> {
        let (pos, _, _) =
            decode_state(state, &self.config.map).map_err(|_| DomainError::InvalidTransition {
                reason: "cannot observe invalid grid state",
            })?;
        build_observation(&self.config, pos, state.tick(), state.elapsed())
    }

    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        self.advance_step(action, dt)
    }
}

impl EnvironmentObserver for GridWorldEnv {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        build_observation(&self.config, self.position, self.tick, self.elapsed).map_err(|_| {
            EnvironmentError::ObservationFailed {
                reason: "failed to construct grid observation",
            }
        })
    }
}

impl Actuator for GridWorldEnv {
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

        let transition = self
            .advance_step(action, self.config.dt)
            .map_err(|err| match err {
                DomainError::AlreadyTerminal => EnvironmentError::AlreadyTerminal,
                other => EnvironmentError::InvalidAction(other),
            })?;

        Ok(ApplyOutcome::new(
            transition.observation().clone(),
            transition.terminal(),
        ))
    }
}

impl LiveEnvironment for GridWorldEnv {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        self.reset_internal();
        EnvironmentObserver::observe(self)
    }
}

/// Modelo generativo puro y ramificable del mundo Grid (`GenerativeModel` + `BranchableSnapshot`).
#[derive(Debug, Clone)]
pub struct GridWorldModel {
    config: GridWorldConfig,
    action_space: ActionSpace,
    state: State,
    rng: DeterministicRng,
    terminal: Terminal,
}

impl GridWorldModel {
    /// Crea un nuevo [`GridWorldModel`] inicializado en `config.map.start()`.
    pub fn new(config: GridWorldConfig, seed: MasterSeed) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
        let state = State::new(
            Tick::ZERO,
            SimDuration::ZERO,
            encode_state(config.map.start(), 0.0, 0.0, 0),
        )?;
        Ok(Self {
            config,
            action_space,
            state,
            rng: seed.derive(StreamKey::Episode(0)),
            terminal: Terminal::Ongoing,
        })
    }

    /// Devuelve la configuración del modelo.
    pub const fn config(&self) -> &GridWorldConfig {
        &self.config
    }

    /// Devuelve el [`State`] interno actual del modelo.
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// Devuelve el estado terminal actual del modelo.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Posición actual decodificada del estado interno.
    pub fn position(&self) -> GridPos {
        let (pos, _, _) =
            decode_state(&self.state, &self.config.map).expect("internal model state is valid");
        pos
    }

    /// Recompensa acumulada en el estado interno actual.
    pub fn total_reward(&self) -> f64 {
        self.state.values()[2]
    }

    /// Avanza el estado interno del modelo en un paso usando `self.config.dt`.
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

impl GenerativeModel for GridWorldModel {
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
        let (pos, total_reward, collisions) = decode_state(state, &self.config.map)?;

        if matches!(self.config.map.cell(pos), Some(CellKind::Goal))
            || (self.config.hazard_is_terminal
                && matches!(self.config.map.cell(pos), Some(CellKind::Hazard)))
        {
            return Err(ModelError::TerminalState);
        }

        let grid_action = GridAction::from_action(action).map_err(ModelError::InvalidAction)?;
        let outcome = step_grid_dynamics(&self.config, pos, grid_action);
        let next_collisions = collisions + u64::from(outcome.collided);
        let next_total_reward = total_reward + outcome.reward;

        let next_state = state
            .advance(
                dt,
                encode_state(
                    outcome.next_pos,
                    next_total_reward,
                    outcome.reward,
                    next_collisions,
                ),
            )
            .map_err(ModelError::TransitionFailed)?;
        let obs = build_observation(
            &self.config,
            outcome.next_pos,
            next_state.tick(),
            next_state.elapsed(),
        )
        .map_err(ModelError::TransitionFailed)?;

        Transition::new(state.tick(), dt, next_state, obs, outcome.terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

impl BranchableSnapshot for GridWorldModel {
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
        decode_state(snapshot.state(), &self.config.map).map_err(|_| {
            SnapshotError::Domain(DomainError::InvalidTransition {
                reason: "snapshot state is incompatible with grid map",
            })
        })?;
        self.state = snapshot.state().clone();
        self.rng = snapshot.rng();
        self.terminal = snapshot.terminal();
        Ok(())
    }
}

/// Resumen determinista de un episodio ejecutado en [`GridWorldEnv`].
#[derive(Debug, Clone, PartialEq)]
pub struct GridEpisodeSummary {
    /// Número de ticks ejecutados antes de alcanzar horizonte o estado terminal.
    pub ticks_executed: u64,
    /// Semilla maestra usada para el episodio.
    pub seed: MasterSeed,
    /// Indica si la observación parcial estuvo activada.
    pub partial_obs: bool,
    /// Posición final del agente al terminar la ejecución.
    pub final_pos: GridPos,
    /// Recompensa acumulada durante el episodio.
    pub total_reward: f64,
    /// Número de colisiones contra muros o bordes.
    pub collisions: u64,
    /// Estado de terminación final.
    pub terminal: Terminal,
    /// Hash determinista de traza de 64 bits.
    pub trace_hash: u64,
}

impl GridEpisodeSummary {
    /// Devuelve `trace_hash` formateado en hexadecimal de 16 dígitos.
    pub fn trace_hash_hex(&self) -> String {
        format!("{:016x}", self.trace_hash)
    }
}

/// Ejecuta un episodio reproducible en [`GridWorldEnv`] con el mapa de callejón sin salida.
pub fn run_grid_episode(
    ticks: u64,
    seed: MasterSeed,
    partial_obs: bool,
) -> Result<GridEpisodeSummary, DomainError> {
    let config = GridWorldConfig::default().with_partial_observation(if partial_obs {
        Some(2)
    } else {
        None
    });
    run_grid_episode_with_config(config, ticks, seed)
}

/// Igual que [`run_grid_episode`] pero con una [`GridWorldConfig`] explícita (p. ej. la derivada
/// de un escenario `ExperimentSpec`, ver [`crate::scenario`]).
pub fn run_grid_episode_with_config(
    config: GridWorldConfig,
    ticks: u64,
    seed: MasterSeed,
) -> Result<GridEpisodeSummary, DomainError> {
    let partial_obs = matches!(config.observation_mode, ObservationMode::Partial { .. });
    let mut env = GridWorldEnv::new(config)?;
    let mut rng = seed.derive(StreamKey::Episode(0));
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(seed));

    let mut ticks_executed = 0;
    for _ in 0..ticks {
        let obs =
            EnvironmentObserver::observe(&env).map_err(|_| DomainError::InvalidTransition {
                reason: "failed to observe grid environment",
            })?;

        let action_idx = (rng.next_u64() % (GRID_ACTION_CARDINALITY as u64)) as usize;
        let action = Action::Discrete(DiscreteAction::new(action_idx, GRID_ACTION_CARDINALITY)?);
        let outcome = env.apply(&action).map_err(|err| match err {
            EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
            EnvironmentError::InvalidAction(d) => d,
            _ => DomainError::InvalidTransition {
                reason: "failed to apply grid action",
            },
        })?;

        hasher.write_tick(env.tick());
        hasher.write_observation(&obs);
        hasher.write_u64(action_idx as u64);
        hasher.write_observation(outcome.observation());
        hasher.write_f64(env.last_reward());
        hasher.write_terminal(outcome.terminal());

        ticks_executed += 1;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    Ok(GridEpisodeSummary {
        ticks_executed,
        seed,
        partial_obs,
        final_pos: env.position(),
        total_reward: env.total_reward(),
        collisions: env.collisions(),
        terminal: env.terminal(),
        trace_hash: hasher.finish(),
    })
}
