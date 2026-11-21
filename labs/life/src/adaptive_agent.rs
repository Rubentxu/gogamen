//! Agente adaptativo homeostático para `gogamen-lab-life` (`std`-only + `gogamen-core`).
//!
//! Reutiliza directamente `gogamen_core::motivation`, `gogamen_core::fmc_planner` y
//! `gogamen_core::snapshot` sin duplicar la lógica de homeostasis ni acceder a
//! [`crate::PrivilegedGlobalView`]:
//! - [`LifeDriveMapping`]: mapea la energía observada en [`LocalOrganismObservation`] a
//!   [`DriveSpec`] / [`HomeostaticState`] y genera [`DynamicUtility`] (`forage_weight` crece
//!   con el déficit energético; `rest_weight` crece al estar saciado).
//! - [`LocalLifeModel`]: modelo generativo local acotado ([`GenerativeModel`] +
//!   [`BranchableSnapshot`]) construido exclusivamente desde [`LocalOrganismObservation`].
//! - [`HomeostaticFmcAgent`]: planificador FMC recedente cuya búsqueda de alimento **emerge**
//!   de la valoración dinámica de estados bajo déficit energético (sin `if` de acción fija
//!   ni recompensa artificial por conducta objetivo).
//! - [`HeuristicLifeBaseline`] y [`run_shared_world_comparison`]: comparación en el mismo
//!   [`LifeWorld`] y [`crate::FoodGrid`] midiendo `planning_transitions_used`,
//!   `planning_latency_ns`, `survival_ticks` y `food_consumed`.

use std::fmt;
use std::time::Instant;

use gogamen_core::{
    Action, ActionCandidate, ActionSpace, BranchableSnapshot, DeterministicRng, DomainError,
    DriveId, DriveSpec, DynamicUtility, FmcPlanner, FmcPlannerConfig, FmcScoringConfig,
    GenerativeModel, HomeostaticState, MasterSeed, ModelError, MotivationError, Observation,
    PlanBudget, PlanRequest, PlanStatus, PlanningContext, RngSource, SimDuration,
    SimulationSnapshot, SnapshotError, State, StepDt, Terminal, TerminalReason, Transition,
    WeightedObjective,
};

use crate::config::{LifeError, LifeWorldConfig};
use crate::organism::{LifeAction, LocalOrganismObservation};
use crate::world::LifeWorld;

/// Acciones locales simuladas en [`LocalLifeModel`].
pub const LOCAL_LIFE_ACTIONS: [LifeAction; 5] = [
    LifeAction::Stay,
    LifeAction::MoveUp,
    LifeAction::MoveDown,
    LifeAction::MoveLeft,
    LifeAction::MoveRight,
];

const STATE_RX_IDX: usize = 0;
const STATE_RY_IDX: usize = 1;
const STATE_ENERGY_IDX: usize = 2;
const STATE_MOVED_IDX: usize = 3;
const STATE_FOOD_EATEN_IDX: usize = 4;
const STATE_TOTAL_FOOD_EATEN_IDX: usize = 5;
const STATE_CELLS_OFFSET: usize = 6;

/// Errores explícitos del agente adaptativo y modelo local de Life.
#[derive(Debug, Clone, PartialEq)]
pub enum LifeAgentError {
    /// Error en la capa de motivación/homeostasis de `gogamen-core`.
    Motivation(MotivationError),
    /// Error en la capa de dominio/acción de `gogamen-core`.
    Domain(DomainError),
    /// Error del modelo generativo o planificador FMC.
    Model(ModelError),
    /// Error de snapshot o ramificación.
    Snapshot(SnapshotError),
    /// Error de configuración o simulación de `LifeWorld`.
    Life(LifeError),
    /// Configuración inválida del agente adaptativo.
    InvalidConfig(&'static str),
}

impl fmt::Display for LifeAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Motivation(err) => write!(f, "motivation error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Model(err) => write!(f, "model/planner error: {err}"),
            Self::Snapshot(err) => write!(f, "snapshot error: {err}"),
            Self::Life(err) => write!(f, "life world error: {err}"),
            Self::InvalidConfig(msg) => write!(f, "invalid adaptive agent config: {msg}"),
        }
    }
}

impl std::error::Error for LifeAgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Motivation(err) => Some(err),
            Self::Domain(err) => Some(err),
            Self::Model(err) => Some(err),
            Self::Snapshot(err) => Some(err),
            Self::Life(err) => Some(err),
            Self::InvalidConfig(_) => None,
        }
    }
}

impl From<MotivationError> for LifeAgentError {
    fn from(value: MotivationError) -> Self {
        Self::Motivation(value)
    }
}

impl From<DomainError> for LifeAgentError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<ModelError> for LifeAgentError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<SnapshotError> for LifeAgentError {
    fn from(value: SnapshotError) -> Self {
        Self::Snapshot(value)
    }
}

impl From<LifeError> for LifeAgentError {
    fn from(value: LifeError) -> Self {
        Self::Life(value)
    }
}

/// Parámetros físicos locales visibles para el modelo interno del organismo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalLifePhysics {
    /// Capacidad máxima de energía del organismo.
    pub max_energy: f64,
    /// Coste metabólico basal por tick.
    pub basal_cost: f64,
    /// Coste adicional por movimiento efectivo.
    pub move_cost: f64,
    /// Máximo alimento consumible en un tick.
    pub max_eat_per_tick: f64,
}

impl LocalLifePhysics {
    /// Extrae únicamente las constantes físicas del organismo desde [`LifeWorldConfig`].
    pub fn from_world_config(config: &LifeWorldConfig) -> Result<Self, LifeAgentError> {
        config.validate()?;
        Ok(Self {
            max_energy: config.max_energy,
            basal_cost: config.basal_cost,
            move_cost: config.move_cost,
            max_eat_per_tick: config.max_eat_per_tick,
        })
    }
}

impl Default for LocalLifePhysics {
    fn default() -> Self {
        let cfg = LifeWorldConfig::default();
        Self {
            max_energy: cfg.max_energy,
            basal_cost: cfg.basal_cost,
            move_cost: cfg.move_cost,
            max_eat_per_tick: cfg.max_eat_per_tick,
        }
    }
}

/// Mapeo homeostático de [`LocalOrganismObservation`] hacia `gogamen_core::motivation`.
///
/// Modela dos drives complementarios sobre `energy` en `[0.0, max_energy]`:
/// - `energy`: `target = max_energy`, cuyo déficit crece conforme baja la energía observada
///   y modula el término `"forage"`.
/// - `satiety`: `target = 0.0`, cuyo déficit crece conforme la energía se acerca a `max_energy`
///   y modula el término `"rest"` (conservar energía estando saciado).
#[derive(Debug, Clone, PartialEq)]
pub struct LifeDriveMapping {
    energy_drive: DriveSpec,
    satiety_drive: DriveSpec,
    max_energy: f64,
}

impl LifeDriveMapping {
    /// Construye un mapeo homeostático validado sobre `gogamen_core::motivation`.
    pub fn new(
        max_energy: f64,
        forage_base_weight: f64,
        forage_deficit_gain: f64,
        rest_base_weight: f64,
        rest_satiety_gain: f64,
    ) -> Result<Self, MotivationError> {
        let energy_id = DriveId::new("energy")?;
        let satiety_id = DriveId::new("satiety")?;

        let energy_drive = DriveSpec::new(
            energy_id,
            max_energy,
            0.0,
            max_energy,
            forage_base_weight,
            forage_deficit_gain,
        )?;
        let satiety_drive = DriveSpec::new(
            satiety_id,
            0.0,
            0.0,
            max_energy,
            rest_base_weight,
            rest_satiety_gain,
        )?;

        Ok(Self {
            energy_drive,
            satiety_drive,
            max_energy,
        })
    }

    /// Construye un mapeo por defecto calibrado para un [`LifeWorldConfig`].
    pub fn from_world_config(config: &LifeWorldConfig) -> Result<Self, LifeAgentError> {
        config.validate()?;
        Ok(Self::new(config.max_energy, 0.2, 6.0, 0.05, 1.2)?)
    }

    /// Especificación del drive energético (`target = max_energy`).
    pub const fn energy_drive(&self) -> &DriveSpec {
        &self.energy_drive
    }

    /// Especificación del drive de saciedad/reposo (`target = 0.0`).
    pub const fn satiety_drive(&self) -> &DriveSpec {
        &self.satiety_drive
    }

    /// Construye un [`HomeostaticState`] actualizado a partir de [`LocalOrganismObservation`].
    pub fn homeostatic_state(
        &self,
        obs: &LocalOrganismObservation,
    ) -> Result<HomeostaticState, MotivationError> {
        let mut state =
            HomeostaticState::new(vec![self.energy_drive.clone(), self.satiety_drive.clone()])?;
        state.update([
            (self.energy_drive.id(), obs.energy()),
            (self.satiety_drive.id(), obs.energy()),
        ])?;
        Ok(state)
    }

    /// Genera la [`DynamicUtility`] acoplada a los drives de `LifeDriveMapping`.
    pub fn dynamic_utility(&self) -> Result<DynamicUtility, MotivationError> {
        let forage = WeightedObjective::from_drive("forage", &self.energy_drive)?;
        let rest = WeightedObjective::from_drive("rest", &self.satiety_drive)?;
        DynamicUtility::new(vec![forage, rest])
    }

    /// Peso efectivo actual de `"forage"` para la observación dada.
    pub fn forage_weight(&self, obs: &LocalOrganismObservation) -> Result<f64, MotivationError> {
        let state = self.homeostatic_state(obs)?;
        let utility = self.dynamic_utility()?;
        utility.effective_weight(&utility.terms()[0], &state)
    }

    /// Peso efectivo actual de `"rest"` para la observación dada.
    pub fn rest_weight(&self, obs: &LocalOrganismObservation) -> Result<f64, MotivationError> {
        let state = self.homeostatic_state(obs)?;
        let utility = self.dynamic_utility()?;
        utility.effective_weight(&utility.terms()[1], &state)
    }

    /// Evalúa la utilidad homeostática de una transición simulada en [`LocalLifeModel`].
    ///
    /// Esta función **no** inspecciona la acción elegida ni recompensa conductas objetivo
    /// como "moverse hacia comida"; evalúa únicamente propiedades intrínsecas del estado
    /// resultante:
    /// - `"forage"`: nivel energético normalizado `next_energy / max_energy` en `[0, 1]`.
    /// - `"rest"`: conservación de posición sin gasto de desplazamiento (`1.0` si `moved == 0`,
    ///   `0.0` en caso contrario).
    pub fn evaluate_transition_utility(
        &self,
        _prev_state: &State,
        action: &Action,
        next_state: &State,
        terminal: &Terminal,
        homeostatic_state: &HomeostaticState,
        utility: &DynamicUtility,
    ) -> Result<f64, MotivationError> {
        if matches!(
            terminal.reason(),
            Some(TerminalReason::Failure | TerminalReason::InvariantViolated)
        ) {
            return Ok(-10.0);
        }
        let values = next_state.values();
        let next_energy = values.get(STATE_ENERGY_IDX).copied().unwrap_or(0.0);
        let moved = values.get(STATE_MOVED_IDX).copied().unwrap_or(0.0);
        let normalized_energy = (next_energy / self.max_energy).clamp(0.0, 1.0);
        let stillness = if moved < 0.5 { 1.0 } else { 0.0 };

        let candidate = ActionCandidate::new(action.clone())
            .with_score("forage", normalized_energy)?
            .with_score("rest", stillness)?;
        let breakdown = utility.score_candidate(&candidate, homeostatic_state)?;
        Ok(breakdown.total_utility())
    }
}

impl Default for LifeDriveMapping {
    fn default() -> Self {
        Self::new(LifeWorldConfig::default().max_energy, 0.2, 6.0, 0.05, 1.2)
            .expect("default LifeDriveMapping must be valid")
    }
}

/// Modelo generativo local acotado construido exclusivamente desde [`LocalOrganismObservation`].
///
/// Nunca accede a [`crate::PrivilegedGlobalView`], ni a coordenadas globales, ni a la grilla
/// global del mundo. Implementa [`GenerativeModel`] y [`BranchableSnapshot`].
#[derive(Debug, Clone, PartialEq)]
pub struct LocalLifeModel {
    vision_radius: usize,
    side: usize,
    in_bounds: Vec<bool>,
    physics: LocalLifePhysics,
    action_space: ActionSpace,
    state: State,
    rng: DeterministicRng,
    terminal: Terminal,
}

impl LocalLifeModel {
    /// Construye un [`LocalLifeModel`] únicamente a partir de `obs` y `physics`.
    pub fn from_observation(
        obs: &LocalOrganismObservation,
        physics: LocalLifePhysics,
        seed: MasterSeed,
    ) -> Result<Self, LifeAgentError> {
        let r = obs.vision_radius();
        let side = 2 * r + 1;
        let expected_cells = side * side;
        if obs.local_cells().len() != expected_cells {
            return Err(LifeAgentError::InvalidConfig(
                "local_cells length does not match vision_radius",
            ));
        }

        let mut in_bounds = vec![false; expected_cells];
        let mut cell_food = vec![0.0; expected_cells];
        for cell in obs.local_cells() {
            let ix = (cell.dx + r as isize) as usize;
            let iy = (cell.dy + r as isize) as usize;
            let idx = iy * side + ix;
            in_bounds[idx] = cell.in_bounds;
            cell_food[idx] = if cell.in_bounds {
                cell.food.max(0.0)
            } else {
                0.0
            };
        }

        let mut values = Vec::with_capacity(STATE_CELLS_OFFSET + expected_cells);
        values.push(0.0); // rx
        values.push(0.0); // ry
        values.push(obs.energy());
        values.push(0.0); // moved flag
        values.push(0.0); // food eaten this step
        values.push(0.0); // total food eaten
        values.extend(cell_food);

        let state = State::new(obs.tick(), SimDuration::ZERO, values)?;
        let action_space = ActionSpace::discrete(LOCAL_LIFE_ACTIONS.len())?;
        let terminal = if obs.energy() > 0.0 {
            Terminal::Ongoing
        } else {
            Terminal::Terminated(TerminalReason::Failure)
        };

        Ok(Self {
            vision_radius: r,
            side,
            in_bounds,
            physics,
            action_space,
            state,
            rng: DeterministicRng::new(seed.as_u64() ^ obs.organism_id()),
            terminal,
        })
    }

    /// Estado inicial codificado desde la observación local.
    pub const fn initial_state(&self) -> &State {
        &self.state
    }

    /// Radio de visión local del modelo.
    pub const fn vision_radius(&self) -> usize {
        self.vision_radius
    }

    /// Decodifica una [`Action`] discreta de `gogamen-core` en [`LifeAction`].
    pub fn decode_action(action: &Action) -> Option<LifeAction> {
        match action {
            Action::Discrete(d) => LOCAL_LIFE_ACTIONS.get(d.index()).copied(),
            Action::Continuous(_) => None,
        }
    }

    fn cell_index(&self, rx: isize, ry: isize) -> Option<usize> {
        let r = self.vision_radius as isize;
        if rx < -r || rx > r || ry < -r || ry > r {
            return None;
        }
        let ix = (rx + r) as usize;
        let iy = (ry + r) as usize;
        Some(iy * self.side + ix)
    }
}

impl GenerativeModel for LocalLifeModel {
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

        let life_action = Self::decode_action(action).ok_or(ModelError::InvalidState {
            reason: "expected discrete LifeAction",
        })?;

        let values = state.values();
        let expected_len = STATE_CELLS_OFFSET + self.side * self.side;
        if values.len() != expected_len {
            return Err(ModelError::InvalidState {
                reason: "LocalLifeModel state dimension mismatch",
            });
        }

        let cur_energy = values[STATE_ENERGY_IDX];
        if cur_energy <= 0.0 {
            return Err(ModelError::TerminalState);
        }

        let mut rx = values[STATE_RX_IDX].round() as isize;
        let mut ry = values[STATE_RY_IDX].round() as isize;
        let mut moved = false;

        if life_action.is_move() {
            let (dx, dy) = life_action.delta();
            let nx = rx + dx;
            let ny = ry + dy;
            if let Some(idx) = self.cell_index(nx, ny)
                && self.in_bounds[idx]
            {
                rx = nx;
                ry = ny;
                moved = true;
            }
        }

        let cost = self.physics.basal_cost + if moved { self.physics.move_cost } else { 0.0 };
        let mut next_values = values.to_vec();
        next_values[STATE_RX_IDX] = rx as f64;
        next_values[STATE_RY_IDX] = ry as f64;
        next_values[STATE_MOVED_IDX] = if moved { 1.0 } else { 0.0 };

        let (next_energy, eaten, terminal) = if cur_energy <= cost {
            (0.0, 0.0, Terminal::Terminated(TerminalReason::Failure))
        } else {
            let energy_after_cost = cur_energy - cost;
            let headroom = (self.physics.max_energy - energy_after_cost).max(0.0);
            let max_request = headroom.min(self.physics.max_eat_per_tick);
            let eaten = if let Some(cell_idx) = self.cell_index(rx, ry) {
                let avail = next_values[STATE_CELLS_OFFSET + cell_idx];
                let taken = avail.min(max_request).max(0.0);
                next_values[STATE_CELLS_OFFSET + cell_idx] = (avail - taken).max(0.0);
                taken
            } else {
                0.0
            };
            (energy_after_cost + eaten, eaten, Terminal::Ongoing)
        };

        next_values[STATE_ENERGY_IDX] = next_energy;
        next_values[STATE_FOOD_EATEN_IDX] = eaten;
        next_values[STATE_TOTAL_FOOD_EATEN_IDX] += eaten;

        let next_state = state
            .advance(dt, next_values)
            .map_err(|_| ModelError::InvalidState {
                reason: "failed to advance LocalLifeModel state",
            })?;
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            vec![rx as f64, ry as f64, next_energy, eaten],
        )
        .map_err(|_| ModelError::InvalidState {
            reason: "failed to build LocalLifeModel observation",
        })?;

        Transition::new(state.tick(), dt, next_state, obs, terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

impl BranchableSnapshot for LocalLifeModel {
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
        let mut forked = self.clone();
        forked.restore(&snap)?;
        Ok(forked)
    }

    fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
        snapshot.verify()?;
        self.state = snapshot.state().clone();
        self.rng = snapshot.rng();
        self.terminal = snapshot.terminal();
        Ok(())
    }
}

/// Resultado de una decisión de agente con métricas de planificación y pesos homeostáticos.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentDecision {
    /// Acción discreta elegida para `LifeWorld`.
    pub action: LifeAction,
    /// Transiciones simuladas consumidas en el planificador (`0` para baseline heurístico).
    pub planning_transitions_used: usize,
    /// Latencia de decisión medida en nanosegundos.
    pub planning_latency_ns: u128,
    /// Peso efectivo de `"forage"` en el instante de la decisión.
    pub forage_weight: f64,
    /// Peso efectivo de `"rest"` en el instante de la decisión.
    pub rest_weight: f64,
}

/// Agente adaptativo homeostático basado en [`FmcPlanner`] y [`LifeDriveMapping`].
#[derive(Debug, Clone)]
pub struct HomeostaticFmcAgent {
    drive_mapping: LifeDriveMapping,
    physics: LocalLifePhysics,
    planner: FmcPlanner,
    budget: PlanBudget,
    seed: MasterSeed,
}

impl HomeostaticFmcAgent {
    /// Construye un [`HomeostaticFmcAgent`] con configuración explícita de mundo y FMC.
    pub fn new(
        world_config: &LifeWorldConfig,
        population_size: usize,
        horizon: usize,
        max_transitions: usize,
        seed: MasterSeed,
    ) -> Result<Self, LifeAgentError> {
        let physics = LocalLifePhysics::from_world_config(world_config)?;
        let drive_mapping = LifeDriveMapping::from_world_config(world_config)?;
        let dt = StepDt::from_millis_f64(100.0)
            .map_err(|_| LifeAgentError::InvalidConfig("default FMC StepDt must be valid"))?;
        let planner_cfg = FmcPlannerConfig::new(
            population_size,
            horizon,
            dt,
            FmcScoringConfig::default(),
            seed,
        )
        .map_err(|_| LifeAgentError::InvalidConfig("invalid FmcPlannerConfig"))?;

        Ok(Self {
            drive_mapping,
            physics,
            planner: FmcPlanner::new(planner_cfg),
            budget: PlanBudget::with_transitions(max_transitions),
            seed,
        })
    }

    /// Acceso al mapeo homeostático subyacente.
    pub const fn drive_mapping(&self) -> &LifeDriveMapping {
        &self.drive_mapping
    }

    /// Decide la siguiente [`LifeAction`] usando únicamente [`LocalOrganismObservation`].
    ///
    /// No contiene reglas `if` de acción fija por umbral energético; la acción elegida
    /// proviene de [`FmcPlanner::plan_with_trace`] evaluando estados con [`DynamicUtility`].
    pub fn decide_with_metrics(
        &mut self,
        obs: &LocalOrganismObservation,
    ) -> Result<AgentDecision, LifeAgentError> {
        let start = Instant::now();
        let homeostatic_state = self.drive_mapping.homeostatic_state(obs)?;
        let utility = self.drive_mapping.dynamic_utility()?;
        let forage_weight = utility.effective_weight(&utility.terms()[0], &homeostatic_state)?;
        let rest_weight = utility.effective_weight(&utility.terms()[1], &homeostatic_state)?;

        let model = LocalLifeModel::from_observation(obs, self.physics, self.seed)?;
        let fallback = Action::discrete_in_space(0, model.action_space())?;
        let request = PlanRequest::from_state(
            model.initial_state().clone(),
            model.action_space().clone(),
            self.budget.clone(),
        )
        .with_fallback(fallback)?;

        let mut rng = DeterministicRng::new(self.seed.as_u64() ^ obs.tick().as_u64());
        let mut ctx = PlanningContext::new(&model, &mut rng);

        let drive_mapping = &self.drive_mapping;
        let (plan_result, trace) = self.planner.plan_with_trace(
            &mut ctx,
            &request,
            |prev_state, action, transition| {
                drive_mapping
                    .evaluate_transition_utility(
                        prev_state,
                        action,
                        transition.next_state(),
                        &transition.terminal(),
                        &homeostatic_state,
                        &utility,
                    )
                    .unwrap_or(-10.0)
            },
            |state| {
                let v = state.values();
                vec![
                    v.get(STATE_RX_IDX).copied().unwrap_or(0.0),
                    v.get(STATE_RY_IDX).copied().unwrap_or(0.0),
                    v.get(STATE_ENERGY_IDX).copied().unwrap_or(0.0),
                ]
            },
        )?;

        let action = match plan_result.status() {
            PlanStatus::Completed | PlanStatus::BudgetExhausted => plan_result
                .authorized_action()
                .and_then(LocalLifeModel::decode_action)
                .unwrap_or(LifeAction::Stay),
            PlanStatus::Cancelled | PlanStatus::NoValidAction => LifeAction::Stay,
        };

        let planning_latency_ns = start.elapsed().as_nanos();
        Ok(AgentDecision {
            action,
            planning_transitions_used: trace.rollout_transitions_used(),
            planning_latency_ns,
            forage_weight,
            rest_weight,
        })
    }

    /// Versión abreviada compatible con [`LifeWorld::step_tick_with`].
    pub fn decide(&mut self, obs: &LocalOrganismObservation) -> LifeAction {
        self.decide_with_metrics(obs)
            .map(|d| d.action)
            .unwrap_or(LifeAction::Stay)
    }
}

/// Futuro candidato de un organismo: una acción raíz mantenida `horizon` pasos en el
/// [`LocalLifeModel`] real (el mismo modelo que usa el planificador FMC del agente).
#[derive(Debug, Clone, PartialEq)]
pub struct LifeCandidateFuture {
    /// Acción raíz mantenida durante el horizonte.
    pub action: LifeAction,
    /// `true` si es la acción que el agente elige en esta observación.
    pub chosen: bool,
    /// Desplazamientos `(rx, ry)` en celdas respecto al organismo; el primero es `(0, 0)`.
    pub offsets: Vec<(f64, f64)>,
    /// Energía al final de la trayectoria simulada.
    pub final_energy: f64,
    /// `true` si el organismo muere (energía agotada) dentro del horizonte.
    pub dies: bool,
}

impl HomeostaticFmcAgent {
    /// Como [`HomeostaticFmcAgent::decide`], pero reproduce si la energía propia alcanza
    /// `reproduction_threshold` (los nacimientos de la población no los decide el modelo local).
    pub fn decide_with_reproduction(
        &mut self,
        obs: &LocalOrganismObservation,
        reproduction_threshold: f64,
    ) -> LifeAction {
        if obs.energy() >= reproduction_threshold {
            LifeAction::Reproduce
        } else {
            self.decide(obs)
        }
    }

    /// Futuros candidatos de `obs` (uno por [`LOCAL_LIFE_ACTIONS`]) simulados `horizon` pasos
    /// con [`LocalLifeModel`]; marca como elegida la acción que decidiría este agente.
    /// No muta el agente (decide sobre una copia).
    pub fn candidate_futures(
        &self,
        obs: &LocalOrganismObservation,
        horizon: usize,
    ) -> Result<Vec<LifeCandidateFuture>, LifeAgentError> {
        let model = LocalLifeModel::from_observation(obs, self.physics, self.seed)?;
        let chosen = self.clone().decide(obs);
        let dt = StepDt::from_millis_f64(100.0)
            .map_err(|_| LifeAgentError::InvalidConfig("default FMC StepDt must be valid"))?;
        let mut out = Vec::with_capacity(LOCAL_LIFE_ACTIONS.len());
        for (idx, &life_action) in LOCAL_LIFE_ACTIONS.iter().enumerate() {
            let action = Action::discrete_in_space(idx, model.action_space())?;
            let mut rng = DeterministicRng::new(self.seed.as_u64() ^ obs.organism_id());
            let mut state = model.initial_state().clone();
            let mut offsets = vec![(0.0, 0.0)];
            let mut dies = false;
            for _ in 0..horizon {
                match model.step(&state, &action, dt, &mut rng) {
                    Ok(t) => {
                        state = t.next_state().clone();
                        let v = state.values();
                        offsets.push((v[STATE_RX_IDX], v[STATE_RY_IDX]));
                        if t.terminal().is_terminal() {
                            dies = true;
                            break;
                        }
                    }
                    Err(_) => {
                        dies = true;
                        break;
                    }
                }
            }
            out.push(LifeCandidateFuture {
                action: life_action,
                chosen: life_action == chosen,
                offsets,
                final_energy: state.values()[STATE_ENERGY_IDX],
                dies,
            });
        }
        Ok(out)
    }
}

/// Baseline heurístico voraz local para comparación justa en el mismo [`LifeWorld`].
#[derive(Debug, Clone, Default)]
pub struct HeuristicLifeBaseline {
    step_counter: u64,
}

impl HeuristicLifeBaseline {
    /// Crea un nuevo baseline heurístico sin planificación basada en modelo.
    pub const fn new() -> Self {
        Self { step_counter: 0 }
    }

    /// Elige acción local voraz y mide latencia (`planning_transitions_used = 0`).
    pub fn decide_with_metrics(&mut self, obs: &LocalOrganismObservation) -> AgentDecision {
        let start = Instant::now();
        self.step_counter = self.step_counter.saturating_add(1);

        let current_food = obs.cell_at(0, 0).map_or(0.0, |c| c.food);
        let action = if current_food > 0.0 {
            LifeAction::Stay
        } else {
            let candidates = [
                (LifeAction::MoveUp, 0, -1),
                (LifeAction::MoveDown, 0, 1),
                (LifeAction::MoveLeft, -1, 0),
                (LifeAction::MoveRight, 1, 0),
            ];
            let mut best_action = LifeAction::Stay;
            let mut best_food = 0.0;
            for (cand, dx, dy) in candidates {
                if let Some(cell) = obs.cell_at(dx, dy)
                    && cell.in_bounds
                    && cell.food > best_food
                {
                    best_food = cell.food;
                    best_action = cand;
                }
            }
            best_action
        };

        AgentDecision {
            action,
            planning_transitions_used: 0,
            planning_latency_ns: start.elapsed().as_nanos(),
            forage_weight: 0.0,
            rest_weight: 0.0,
        }
    }
}

/// Métricas acumuladas por agente en un episodio compartido de [`LifeWorld`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgentRunMetrics {
    /// Identificador del organismo en [`LifeWorld`].
    pub organism_id: u64,
    /// Ticks en los que el organismo permaneció vivo.
    pub survival_ticks: u64,
    /// Alimento total consumido por el organismo durante el episodio.
    pub food_consumed: f64,
    /// Energía final al término del episodio (`0.0` si murió).
    pub final_energy: f64,
    /// Indica si seguía vivo al finalizar el episodio.
    pub alive_at_end: bool,
    /// Total de transiciones de rollout consumidas en planificación.
    pub planning_transitions_used: usize,
    /// Latencia acumulada de decisión en nanosegundos.
    pub planning_latency_ns: u128,
}

/// Resultado de comparar [`HeuristicLifeBaseline`] y [`HomeostaticFmcAgent`] en el mismo mundo.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedWorldComparison {
    /// Métricas del baseline heurístico.
    pub baseline: AgentRunMetrics,
    /// Métricas del agente adaptativo homeostático FMC.
    pub fmc: AgentRunMetrics,
    /// Ticks totales ejecutados en el mundo compartido.
    pub ticks_executed: u64,
}

/// Ejecuta [`HeuristicLifeBaseline`] y [`HomeostaticFmcAgent`] simultáneamente en el mismo
/// [`LifeWorld`] y [`crate::FoodGrid`], midiendo supervivencia, consumo y coste de planificación.
pub fn run_shared_world_comparison(
    world: &mut LifeWorld,
    baseline_id: u64,
    fmc_id: u64,
    baseline: &mut HeuristicLifeBaseline,
    fmc_agent: &mut HomeostaticFmcAgent,
    ticks: u64,
) -> Result<SharedWorldComparison, LifeAgentError> {
    if baseline_id == fmc_id {
        return Err(LifeAgentError::InvalidConfig(
            "baseline_id and fmc_id must be distinct organisms",
        ));
    }

    let cfg = world.config().clone();
    let mut baseline_metrics = AgentRunMetrics {
        organism_id: baseline_id,
        survival_ticks: 0,
        food_consumed: 0.0,
        final_energy: 0.0,
        alive_at_end: false,
        planning_transitions_used: 0,
        planning_latency_ns: 0,
    };
    let mut fmc_metrics = AgentRunMetrics {
        organism_id: fmc_id,
        survival_ticks: 0,
        food_consumed: 0.0,
        final_energy: 0.0,
        alive_at_end: false,
        planning_transitions_used: 0,
        planning_latency_ns: 0,
    };

    for _ in 0..ticks {
        let mut baseline_step: Option<(f64, f64)> = None;
        let mut fmc_step: Option<(f64, f64)> = None;

        world.step_tick_with(|obs| {
            if obs.organism_id() == baseline_id {
                let decision = baseline.decide_with_metrics(obs);
                let moved = decision.action.is_move() && {
                    let (dx, dy) = decision.action.delta();
                    obs.cell_at(dx, dy).is_some_and(|c| c.in_bounds)
                };
                let cost = cfg.basal_cost + if moved { cfg.move_cost } else { 0.0 };
                baseline_metrics.planning_transitions_used += decision.planning_transitions_used;
                baseline_metrics.planning_latency_ns += decision.planning_latency_ns;
                baseline_step = Some((obs.energy(), cost));
                decision.action
            } else if obs.organism_id() == fmc_id {
                let decision = fmc_agent.decide_with_metrics(obs).unwrap_or(AgentDecision {
                    action: LifeAction::Stay,
                    planning_transitions_used: 0,
                    planning_latency_ns: 0,
                    forage_weight: 0.0,
                    rest_weight: 0.0,
                });
                let moved = decision.action.is_move() && {
                    let (dx, dy) = decision.action.delta();
                    obs.cell_at(dx, dy).is_some_and(|c| c.in_bounds)
                };
                let cost = cfg.basal_cost + if moved { cfg.move_cost } else { 0.0 };
                fmc_metrics.planning_transitions_used += decision.planning_transitions_used;
                fmc_metrics.planning_latency_ns += decision.planning_latency_ns;
                fmc_step = Some((obs.energy(), cost));
                decision.action
            } else {
                LifeAction::Stay
            }
        });

        let view = world.privileged_global_view();
        let orgs = view.organisms();
        for i in 0..orgs.len() {
            let id = orgs.ids()[i];
            let alive = orgs.alive()[i];
            let energy_after = orgs.energy()[i];
            if id == baseline_id
                && let Some((e_before, cost)) = baseline_step
                && alive
            {
                baseline_metrics.survival_ticks += 1;
                baseline_metrics.food_consumed += (energy_after - (e_before - cost)).max(0.0);
            } else if id == fmc_id
                && let Some((e_before, cost)) = fmc_step
                && alive
            {
                fmc_metrics.survival_ticks += 1;
                fmc_metrics.food_consumed += (energy_after - (e_before - cost)).max(0.0);
            }
        }
    }

    let view = world.privileged_global_view();
    let orgs = view.organisms();
    for i in 0..orgs.len() {
        let id = orgs.ids()[i];
        if id == baseline_id {
            baseline_metrics.alive_at_end = orgs.alive()[i];
            baseline_metrics.final_energy = orgs.energy()[i];
        } else if id == fmc_id {
            fmc_metrics.alive_at_end = orgs.alive()[i];
            fmc_metrics.final_energy = orgs.energy()[i];
        }
    }

    Ok(SharedWorldComparison {
        baseline: baseline_metrics,
        fmc: fmc_metrics,
        ticks_executed: ticks,
    })
}

#[cfg(test)]
mod tests {

    #[test]
    fn candidate_futures_cover_actions_mark_exactly_one_chosen_and_match_decision() {
        let cfg = LifeWorldConfig::default();
        let mut world = LifeWorld::new(cfg.clone(), MasterSeed::new(5)).unwrap();
        let agent = HomeostaticFmcAgent::new(&cfg, 8, 3, 64, MasterSeed::new(9)).unwrap();
        let obs = world.observe_organism(0).unwrap();
        let futures = agent.candidate_futures(&obs, 3).unwrap();
        assert_eq!(futures.len(), LOCAL_LIFE_ACTIONS.len());
        assert_eq!(futures.iter().filter(|f| f.chosen).count(), 1);
        let chosen = futures.iter().find(|f| f.chosen).unwrap();
        assert_eq!(chosen.action, agent.clone().decide(&obs));
        assert!(futures.iter().all(|f| f.offsets[0] == (0.0, 0.0)));
        let again = agent.candidate_futures(&obs, 3).unwrap();
        assert_eq!(futures, again, "deterministic and non-mutating");
        let _ = world.step_tick_with(|_| LifeAction::Stay);
    }

    #[test]
    fn decide_with_reproduction_reproduces_at_threshold_only() {
        let cfg = LifeWorldConfig::default();
        let mut world = LifeWorld::new(cfg.clone(), MasterSeed::new(5)).unwrap();
        let mut agent = HomeostaticFmcAgent::new(&cfg, 8, 3, 64, MasterSeed::new(9)).unwrap();
        let obs = world.observe_organism(0).unwrap();
        assert_eq!(
            agent.decide_with_reproduction(&obs, obs.energy()),
            LifeAction::Reproduce
        );
        assert_ne!(
            agent.decide_with_reproduction(&obs, obs.energy() + 1.0),
            LifeAction::Reproduce
        );
        let _ = world.step_tick_with(|_| LifeAction::Stay);
    }
    use super::*;
    use crate::organism::LocalCellInfo;
    use gogamen_core::Tick;

    fn make_local_obs(
        tick: u64,
        energy: f64,
        vision_radius: usize,
        food_offsets: &[(isize, isize, f64)],
    ) -> LocalOrganismObservation {
        let r = vision_radius as isize;
        let mut cells = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                let food = food_offsets
                    .iter()
                    .find(|&&(fx, fy, _)| fx == dx && fy == dy)
                    .map_or(0.0, |&(_, _, f)| f);
                cells.push(LocalCellInfo {
                    dx,
                    dy,
                    in_bounds: true,
                    food,
                    occupied_by_other: false,
                });
            }
        }
        LocalOrganismObservation::new(
            Tick::new(tick),
            1,
            energy,
            tick,
            Default::default(),
            vision_radius,
            cells,
        )
    }

    #[test]
    fn ac1_food_seeking_emerges_from_homeostatic_deficit_without_fixed_if() {
        let world_cfg = LifeWorldConfig {
            width: 7,
            height: 7,
            initial_population: 0,
            max_population: 4,
            initial_energy: 5.0,
            max_energy: 10.0,
            basal_cost: 0.5,
            move_cost: 0.2,
            max_eat_per_tick: 4.0,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 2,
            ..LifeWorldConfig::default()
        };

        let mut agent =
            HomeostaticFmcAgent::new(&world_cfg, 40, 4, 256, MasterSeed::new(101)).unwrap();

        // Alimento a 1 paso a la derecha (+1, 0).
        let satiated_obs = make_local_obs(0, 10.0, 2, &[(1, 0, 5.0)]);
        let hungry_obs = make_local_obs(0, 3.0, 2, &[(1, 0, 5.0)]);

        let satiated_decision = agent.decide_with_metrics(&satiated_obs).unwrap();
        let hungry_decision = agent.decide_with_metrics(&hungry_obs).unwrap();

        // Cuando está saciado, rest_weight > forage_weight y elige conservar energía (Stay).
        assert!(satiated_decision.rest_weight > satiated_decision.forage_weight);
        assert_eq!(satiated_decision.action, LifeAction::Stay);

        // Con déficit energético, forage_weight domina y emerge MoveRight hacia el alimento.
        assert!(hungry_decision.forage_weight > satiated_decision.forage_weight);
        assert!(hungry_decision.forage_weight > hungry_decision.rest_weight);
        assert_eq!(hungry_decision.action, LifeAction::MoveRight);
        assert!(hungry_decision.planning_transitions_used > 0);
    }

    #[test]
    fn ac1_local_life_model_supports_branchable_snapshot_isolation() {
        let physics = LocalLifePhysics::default();
        let obs = make_local_obs(2, 6.0, 1, &[(1, 0, 3.0)]);
        let model = LocalLifeModel::from_observation(&obs, physics, MasterSeed::new(7)).unwrap();
        let dt = StepDt::from_millis_f64(100.0).unwrap();

        let mut fork_a = model.fork().unwrap();
        let fork_b = model.fork().unwrap();
        let snap = model.snapshot().unwrap();

        let mut rng_a = DeterministicRng::new(1);
        let mut rng_b = DeterministicRng::new(1);
        let move_right = Action::discrete_in_space(4, model.action_space()).unwrap();
        let stay = Action::discrete_in_space(0, model.action_space()).unwrap();

        let tr_a = fork_a
            .step(fork_a.initial_state(), &move_right, dt, &mut rng_a)
            .unwrap();
        let tr_b = fork_b
            .step(fork_b.initial_state(), &stay, dt, &mut rng_b)
            .unwrap();

        assert!(
            tr_a.next_state().values()[STATE_ENERGY_IDX]
                > tr_b.next_state().values()[STATE_ENERGY_IDX]
        );
        fork_a.restore(&snap).unwrap();
        assert_eq!(fork_a.initial_state(), model.initial_state());
    }

    #[test]
    fn ac2_baseline_and_fmc_share_world_and_measure_planning_cost() {
        let world_cfg = LifeWorldConfig {
            width: 7,
            height: 7,
            initial_population: 0,
            max_population: 4,
            initial_energy: 5.0,
            max_energy: 10.0,
            basal_cost: 0.4,
            move_cost: 0.2,
            max_eat_per_tick: 3.5,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.05,
            food_regen_prob: 0.2,
            vision_radius: 2,
            ..LifeWorldConfig::default()
        };

        let mut world = LifeWorld::empty(world_cfg.clone(), MasterSeed::new(555)).unwrap();
        // Sembramos alimento a 2 pasos de ambos organismos (fuera del radio inmediato 1 del baseline voraz).
        world.set_cell_food(3, 1, 8.0).unwrap();
        world.set_cell_food(3, 5, 8.0).unwrap();
        world.set_cell_food(5, 3, 8.0).unwrap();

        let baseline_id = world.spawn_organism(1, 1, 4.0).unwrap();
        let fmc_id = world.spawn_organism(1, 5, 4.0).unwrap();

        let mut baseline = HeuristicLifeBaseline::new();
        let mut fmc_agent =
            HomeostaticFmcAgent::new(&world_cfg, 32, 4, 256, MasterSeed::new(777)).unwrap();

        let report = run_shared_world_comparison(
            &mut world,
            baseline_id,
            fmc_id,
            &mut baseline,
            &mut fmc_agent,
            12,
        )
        .unwrap();

        assert_eq!(report.ticks_executed, 12);
        assert_eq!(report.baseline.planning_transitions_used, 0);
        assert!(report.fmc.planning_transitions_used > 0);
        assert!(report.fmc.planning_latency_ns > 0);
        assert!(report.fmc.survival_ticks >= report.baseline.survival_ticks);
        assert!(report.fmc.food_consumed > 0.0);
        assert!(world.energy_balance().discrepancy() < 1e-9);
    }

    #[test]
    fn ac3_emergence_evaluation_avoids_target_behavior_reward() {
        let mapping = LifeDriveMapping::default();
        let utility = mapping.dynamic_utility().unwrap();

        let hungry_obs = make_local_obs(0, 2.0, 1, &[]);
        let hungry_state = mapping.homeostatic_state(&hungry_obs).unwrap();

        let prev_state = State::new(
            Tick::ZERO,
            SimDuration::ZERO,
            vec![0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
        )
        .unwrap();
        // Dos estados siguientes con idéntica energía resultante y sin movimiento:
        // la evaluación de transición no depende de ninguna etiqueta de acción objetivo.
        let next_state_a = State::new(
            Tick::new(1),
            SimDuration::ZERO,
            vec![1.0, 0.0, 5.0, 1.0, 3.0, 3.0],
        )
        .unwrap();
        let next_state_b = State::new(
            Tick::new(1),
            SimDuration::ZERO,
            vec![-1.0, 0.0, 5.0, 1.0, 0.0, 0.0],
        )
        .unwrap();

        let dummy_action_a =
            Action::discrete_in_space(4, &ActionSpace::discrete(5).unwrap()).unwrap();
        let dummy_action_b =
            Action::discrete_in_space(3, &ActionSpace::discrete(5).unwrap()).unwrap();

        let u_a = mapping
            .evaluate_transition_utility(
                &prev_state,
                &dummy_action_a,
                &next_state_a,
                &Terminal::Ongoing,
                &hungry_state,
                &utility,
            )
            .unwrap();
        let u_b = mapping
            .evaluate_transition_utility(
                &prev_state,
                &dummy_action_b,
                &next_state_b,
                &Terminal::Ongoing,
                &hungry_state,
                &utility,
            )
            .unwrap();

        // La utilidad depende únicamente de la homeostasis (energía y conservación),
        // no de la dirección ni de una recompensa por "ir hacia comida".
        assert!((u_a - u_b).abs() < 1e-12);
    }
}
