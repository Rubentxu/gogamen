//! Prototipo separado de FMC sobre muestras de creencia (`std`-only + `gogamen-core`).
//!
//! Este módulo implementa un prototipo experimental de planificación Fractal Monte Carlo (FMC)
//! sobre partículas de mapa muestreadas exclusivamente a partir de [`GridBeliefMap`] y
//! [`OccludedObservation`], sin acceso al [`crate::partial_obs::GroundTruthGrid`] real.
//!
//! **Nota explícita:** este prototipo realiza rollouts FMC sobre muestras de creencia y
//! **no es POMCP** (no construye un árbol de creencias POMCP ni asume observabilidad completa).

use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, DeterministicRng, DomainError, FmcPlanner, FmcPlannerConfig, FmcTraceSummary,
    GenerativeModel, MasterSeed, ModelError, MonotonicInstant, PlanBudget, PlanRequest,
    PlanningContext, RngSource, State, StreamKey, Transition,
};

use crate::map::{CellKind, GridMap, GridPos, GridSize};
use crate::partial_obs::{
    GridBeliefMap, HypothesisKind, OccludedObservation, PartialGridObserver, PartialObsError,
};
use crate::world::{GridAction, GridWorldConfig, GridWorldModel};

/// Nota explícita de alcance epistémico del prototipo.
pub const BELIEF_FMC_PROTOTYPE_NOTE: &str = "Prototipo separado de FMC sobre muestras de creencia (no es POMCP; no asume observabilidad completa ni construye árbol de creencias POMCP).";

/// Orden determinista de acciones discretas del Grid para desempate y agregación.
const ORDERED_GRID_ACTIONS: [GridAction; 5] = [
    GridAction::Up,
    GridAction::Right,
    GridAction::Down,
    GridAction::Left,
    GridAction::Stay,
];

/// Errores explícitos del prototipo de FMC sobre muestras de creencia.
#[derive(Debug, Clone, PartialEq)]
pub enum BeliefFmcError {
    /// El número de partículas `K` debe ser al menos `1`.
    InvalidParticleCount(usize),
    /// Las dimensiones de [`GridBeliefMap`] y [`OccludedObservation`] no coinciden.
    GridSizeMismatch {
        /// Tamaño del mapa de creencias.
        belief_size: GridSize,
        /// Tamaño reportado por la observación ocluida.
        observation_size: GridSize,
    },
    /// Error del subsistema de observabilidad parcial.
    PartialObs(PartialObsError),
    /// Error de dominio de `gogamen-core`.
    Domain(DomainError),
    /// Error del modelo generativo o planificador de `gogamen-core`.
    Model(ModelError),
}

impl fmt::Display for BeliefFmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParticleCount(k) => {
                write!(f, "invalid belief particle count ({k}): must be >= 1")
            }
            Self::GridSizeMismatch {
                belief_size,
                observation_size,
            } => write!(
                f,
                "grid size mismatch: belief map is {}x{}, observation is {}x{}",
                belief_size.width,
                belief_size.height,
                observation_size.width,
                observation_size.height
            ),
            Self::PartialObs(err) => write!(f, "partial observation error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Model(err) => write!(f, "model error: {err}"),
        }
    }
}

impl Error for BeliefFmcError {}

impl From<PartialObsError> for BeliefFmcError {
    fn from(value: PartialObsError) -> Self {
        Self::PartialObs(value)
    }
}

impl From<DomainError> for BeliefFmcError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<ModelError> for BeliefFmcError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

/// Partícula de estado raíz muestreada únicamente desde el posterior de creencias.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefParticle {
    /// Mapa discreto muestreado a partir de [`GridBeliefMap`] y [`OccludedObservation`].
    pub sampled_map: GridMap,
    /// Peso normalizado de la partícula en `(0.0, 1.0]`.
    pub weight: f64,
}

impl BeliefParticle {
    /// Construye una [`BeliefParticle`] explícita.
    pub const fn new(sampled_map: GridMap, weight: f64) -> Self {
        Self {
            sampled_map,
            weight,
        }
    }
}

/// Modo de planificación comparado en [`BeliefFmcComparison`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BeliefFmcMode {
    /// FMC ejecutado sobre un único estado/mapa nominal fijo derivado del posterior.
    NominalFmc,
    /// FMC ejecutado sobre `K` partículas muestreadas del posterior de creencias.
    BeliefSampledFmc,
}

/// Configuración del planificador prototipo [`BeliefFmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefFmcConfig {
    /// Número `K` de partículas de mapa muestreadas por decisión.
    pub particle_count: usize,
    /// Configuración base de [`FmcPlanner`] aplicada a cada partícula y al modo nominal.
    pub fmc_config: FmcPlannerConfig,
    /// Presupuesto de transiciones simuladas concedido por partícula (y para el modo nominal).
    pub budget_per_particle: PlanBudget,
    /// Recompensa por paso válido en celda libre para los modelos de partícula.
    pub step_reward: f64,
    /// Penalización por colisión contra muro o borde en los modelos de partícula.
    pub wall_collision_reward: f64,
    /// Recompensa por alcanzar `CellKind::Goal` en los modelos de partícula.
    pub goal_reward: f64,
    /// Recompensa por pisar `CellKind::Hazard` en los modelos de partícula.
    pub hazard_reward: f64,
    /// Si es `true`, pisar `CellKind::Hazard` termina el rollout simulado.
    pub hazard_is_terminal: bool,
}

impl Default for BeliefFmcConfig {
    fn default() -> Self {
        let defaults = GridWorldConfig::default();
        Self {
            particle_count: 8,
            fmc_config: FmcPlannerConfig::default(),
            budget_per_particle: PlanBudget::with_transitions(256),
            step_reward: defaults.step_reward,
            wall_collision_reward: defaults.wall_collision_reward,
            goal_reward: defaults.goal_reward,
            hazard_reward: defaults.hazard_reward,
            hazard_is_terminal: defaults.hazard_is_terminal,
        }
    }
}

impl BeliefFmcConfig {
    fn to_world_config(&self, map: GridMap) -> GridWorldConfig {
        GridWorldConfig {
            map,
            observation_mode: crate::world::ObservationMode::Full,
            dt: self.fmc_config.dt(),
            step_reward: self.step_reward,
            wall_collision_reward: self.wall_collision_reward,
            goal_reward: self.goal_reward,
            hazard_reward: self.hazard_reward,
            hazard_is_terminal: self.hazard_is_terminal,
        }
    }
}

/// Traza aislada del rollout FMC ejecutado sobre una [`BeliefParticle`].
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleRolloutTrace {
    /// Índice de la partícula en `0..K`.
    pub particle_index: usize,
    /// Peso normalizado de la partícula.
    pub weight: f64,
    /// Primera acción seleccionada por [`FmcPlanner`] en esta partícula.
    pub first_action: GridAction,
    /// Resumen de traza de [`FmcPlanner`] para esta partícula.
    pub trace: FmcTraceSummary,
}

/// Resumen de ejecución del modo nominal (`NominalFmc`).
#[derive(Debug, Clone, PartialEq)]
pub struct NominalFmcSummary {
    /// Modo etiquetado explícitamente como [`BeliefFmcMode::NominalFmc`].
    pub mode: BeliefFmcMode,
    /// Primera acción seleccionada sobre el estado nominal fijo.
    pub first_action: GridAction,
    /// Traza de [`FmcPlanner`] sobre el modelo nominal.
    pub trace: FmcTraceSummary,
}

/// Resumen agregado de ejecución sobre muestras de creencia (`BeliefSampledFmc`).
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefSampledFmcSummary {
    /// Modo etiquetado explícitamente como [`BeliefFmcMode::BeliefSampledFmc`].
    pub mode: BeliefFmcMode,
    /// Primera acción seleccionada tras agregar las partículas ponderadas.
    pub first_action: GridAction,
    /// Número `K` de partículas evaluadas.
    pub particle_count: usize,
    /// Total de transiciones simuladas consumidas sumando todas las partículas.
    pub total_rollout_transitions: usize,
    /// Media ponderada de `mean_virtual_reward` across partículas.
    pub weighted_mean_virtual_reward: f64,
    /// Peso acumulado por cada acción candidata en orden determinista.
    pub action_weights: [(GridAction, f64); 5],
}

/// Comparación estructurada entre `NominalFmc` y `BeliefSampledFmc`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefFmcComparison {
    /// Resultado de FMC sobre estado nominal fijo.
    pub nominal: NominalFmcSummary,
    /// Resultado de FMC sobre muestras del posterior de creencias.
    pub belief_sampled: BeliefSampledFmcSummary,
    /// Incertidumbre epistémica total del mapa de creencias en bits (entropía de Shannon).
    pub epistemic_uncertainty_bits: f64,
    /// Indica si ambos modos eligieron la misma primera acción.
    pub actions_agree: bool,
    /// Nota explícita que aclara que es un prototipo de FMC sobre muestras de creencia y no POMCP.
    pub prototype_note: &'static str,
}

/// Resultado completo de una decisión de [`BeliefFmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefFmcDecision {
    /// Primera acción agregada seleccionada por `BeliefSampledFmc`.
    pub selected_action: GridAction,
    /// Partículas muestreadas del posterior de creencias.
    pub particles: Vec<BeliefParticle>,
    /// Trazas aisladas de rollout por partícula.
    pub particle_traces: Vec<ParticleRolloutTrace>,
    /// Comparación explícita `NominalFmc` vs `BeliefSampledFmc` con incertidumbre.
    pub comparison: BeliefFmcComparison,
}

/// Calcula la entropía binaria de Shannon en bits para una probabilidad `p` en `[0, 1]`.
fn binary_entropy_bits(p: f64) -> f64 {
    let p = p.clamp(0.0, 1.0);
    let q = 1.0 - p;
    let h_p = if p > 0.0 { -p * p.log2() } else { 0.0 };
    let h_q = if q > 0.0 { -q * q.log2() } else { 0.0 };
    h_p + h_q
}

/// Calcula la incertidumbre epistémica total en bits sobre todas las celdas e hipótesis de `beliefs`.
pub fn belief_epistemic_uncertainty_bits(beliefs: &GridBeliefMap) -> Result<f64, BeliefFmcError> {
    let size = beliefs.size();
    let mut total_bits = 0.0_f64;
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            for hypothesis in HypothesisKind::ALL {
                let p = beliefs.probability(pos, hypothesis)?;
                total_bits += binary_entropy_bits(p);
            }
        }
    }
    Ok(total_bits)
}

fn validate_belief_and_obs(
    beliefs: &GridBeliefMap,
    obs: &OccludedObservation,
) -> Result<GridBeliefMap, BeliefFmcError> {
    if beliefs.size() != obs.grid_size() {
        return Err(BeliefFmcError::GridSizeMismatch {
            belief_size: beliefs.size(),
            observation_size: obs.grid_size(),
        });
    }
    let mut effective = beliefs.clone();
    effective.apply_occluded_observation(obs)?;
    Ok(effective)
}

fn find_visible_goal(obs: &OccludedObservation) -> Option<GridPos> {
    obs.visible_cells()
        .iter()
        .find(|cell| cell.payload().kind == CellKind::Goal)
        .map(|cell| cell.payload().pos)
}

fn sample_goal_pos(
    beliefs: &GridBeliefMap,
    obs: &OccludedObservation,
    start: GridPos,
    rng: &mut dyn RngSource,
) -> Result<GridPos, BeliefFmcError> {
    if let Some(goal) = find_visible_goal(obs) {
        return Ok(goal);
    }

    let size = beliefs.size();
    let mut candidates: Vec<(GridPos, f64)> = Vec::new();
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            if pos == start {
                continue;
            }
            if matches!(obs.cell_at(pos), Some(CellKind::Wall)) {
                continue;
            }
            let p_goal = beliefs.probability(pos, HypothesisKind::Goal)?.max(1e-6);
            candidates.push((pos, p_goal));
        }
    }

    if candidates.is_empty() {
        return Ok(start);
    }

    let total_weight: f64 = candidates.iter().map(|(_, w)| *w).sum();
    let threshold = rng.next_f64() * total_weight;
    let mut acc = 0.0_f64;
    for &(pos, weight) in &candidates {
        acc += weight;
        if acc >= threshold {
            return Ok(pos);
        }
    }
    Ok(candidates[candidates.len() - 1].0)
}

fn sample_single_map(
    beliefs: &GridBeliefMap,
    obs: &OccludedObservation,
    rng: &mut dyn RngSource,
) -> Result<GridMap, BeliefFmcError> {
    let size = beliefs.size();
    let start = obs.agent_pos();
    let goal = sample_goal_pos(beliefs, obs, start, rng)?;
    let mut cells = Vec::with_capacity(size.cell_count());

    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            let cell = if pos == goal {
                CellKind::Goal
            } else if pos == start {
                obs.cell_at(pos)
                    .filter(|kind| !kind.is_blocking())
                    .unwrap_or(CellKind::Empty)
            } else if let Some(visible_kind) = obs.cell_at(pos) {
                visible_kind
            } else {
                let p_obs = beliefs
                    .probability(pos, HypothesisKind::Obstacle)?
                    .clamp(0.0, 1.0);
                let p_haz = beliefs
                    .probability(pos, HypothesisKind::Hazard)?
                    .clamp(0.0, 1.0);
                let u = rng.next_f64();
                if u < p_obs {
                    CellKind::Wall
                } else if u < p_obs + (1.0 - p_obs) * p_haz {
                    CellKind::Hazard
                } else {
                    CellKind::Empty
                }
            };
            cells.push(cell);
        }
    }

    Ok(GridMap::new(size.width, size.height, cells, start, goal)?)
}

/// Muestrea `particle_count` partículas [`BeliefParticle`] desde [`GridBeliefMap`] + [`OccludedObservation`].
///
/// Nunca recibe ni consulta [`crate::partial_obs::GroundTruthGrid`].
pub fn sample_belief_particles(
    beliefs: &GridBeliefMap,
    obs: &OccludedObservation,
    particle_count: usize,
    rng: &mut dyn RngSource,
) -> Result<Vec<BeliefParticle>, BeliefFmcError> {
    if particle_count == 0 {
        return Err(BeliefFmcError::InvalidParticleCount(particle_count));
    }
    let effective = validate_belief_and_obs(beliefs, obs)?;
    let weight = 1.0 / (particle_count as f64);
    let mut particles = Vec::with_capacity(particle_count);
    for _ in 0..particle_count {
        let sampled_map = sample_single_map(&effective, obs, rng)?;
        particles.push(BeliefParticle::new(sampled_map, weight));
    }
    Ok(particles)
}

/// Construye un mapa nominal determinista (MAP por celda) únicamente a partir de
/// [`GridBeliefMap`] y [`OccludedObservation`], sin acceso al ground truth.
pub fn build_nominal_map_from_belief(
    beliefs: &GridBeliefMap,
    obs: &OccludedObservation,
) -> Result<GridMap, BeliefFmcError> {
    let effective = validate_belief_and_obs(beliefs, obs)?;
    let size = effective.size();
    let start = obs.agent_pos();

    let goal = if let Some(visible_goal) = find_visible_goal(obs) {
        visible_goal
    } else {
        let mut best_pos = start;
        let mut best_p = -1.0_f64;
        for y in 0..size.height {
            for x in 0..size.width {
                let pos = GridPos::new(x, y);
                if pos == start || matches!(obs.cell_at(pos), Some(CellKind::Wall)) {
                    continue;
                }
                let p_goal = effective.probability(pos, HypothesisKind::Goal)?;
                if p_goal > best_p {
                    best_p = p_goal;
                    best_pos = pos;
                }
            }
        }
        best_pos
    };

    let mut cells = Vec::with_capacity(size.cell_count());
    for y in 0..size.height {
        for x in 0..size.width {
            let pos = GridPos::new(x, y);
            let cell = if pos == goal {
                CellKind::Goal
            } else if pos == start {
                obs.cell_at(pos)
                    .filter(|kind| !kind.is_blocking())
                    .unwrap_or(CellKind::Empty)
            } else if let Some(visible_kind) = obs.cell_at(pos) {
                visible_kind
            } else {
                let p_obs = effective.probability(pos, HypothesisKind::Obstacle)?;
                let p_haz = effective.probability(pos, HypothesisKind::Hazard)?;
                if p_obs >= 0.5 {
                    CellKind::Wall
                } else if p_haz >= 0.5 {
                    CellKind::Hazard
                } else {
                    CellKind::Empty
                }
            };
            cells.push(cell);
        }
    }

    Ok(GridMap::new(size.width, size.height, cells, start, goal)?)
}

fn grid_step_reward(prev: &State, _action: &Action, transition: &Transition) -> f64 {
    transition.next_state().values()[2] - prev.values()[2]
}

fn grid_phi(state: &State) -> Vec<f64> {
    vec![state.values()[0], state.values()[1]]
}

fn run_isolated_fmc_on_map(
    config: &BeliefFmcConfig,
    map: GridMap,
    seed: MasterSeed,
    episode: u64,
    agent_id: u64,
) -> Result<(GridAction, FmcTraceSummary), BeliefFmcError> {
    let world_config = config.to_world_config(map);
    let model = GridWorldModel::new(world_config, seed)?;
    let fallback = GridAction::Stay.to_action(model.action_space())?;
    let request = PlanRequest::from_state(
        model.state().clone(),
        model.action_space().clone(),
        config.budget_per_particle.clone(),
    )
    .with_fallback(fallback)?;

    let fmc_config = FmcPlannerConfig::new(
        config.fmc_config.population_size(),
        config.fmc_config.horizon(),
        config.fmc_config.dt(),
        *config.fmc_config.scoring(),
        seed,
    )
    .map_err(|_| {
        BeliefFmcError::Domain(DomainError::InvalidTransition {
            reason: "invalid FmcPlannerConfig in BeliefFmcPlanner",
        })
    })?;

    let planner = FmcPlanner::new(fmc_config);
    let mut planning_rng = seed.derive(StreamKey::Agent {
        episode,
        agent: agent_id,
    });
    let mut ctx = PlanningContext::new(&model, &mut planning_rng);
    let (_result, trace) =
        planner.plan_with_trace(&mut ctx, &request, grid_step_reward, grid_phi)?;
    let first_action = match trace.first_action() {
        Some(action) => GridAction::from_action(action)?,
        None => GridAction::Stay,
    };
    Ok((first_action, trace))
}

fn action_index(action: GridAction) -> usize {
    ORDERED_GRID_ACTIONS
        .iter()
        .position(|&candidate| candidate == action)
        .expect("all GridAction variants are in ORDERED_GRID_ACTIONS")
}

/// Planificador prototipo de FMC sobre muestras de creencia (no es POMCP).
///
/// Muestrea `K` partículas de mapa desde [`GridBeliefMap`] + [`OccludedObservation`]
/// usando [`DeterministicRng`] (`StreamKey::Agent`), ejecuta rollouts aislados con
/// [`FmcPlanner`] bajo [`PlanningContext`] y agrega los resultados ponderados,
/// registrando la comparación frente a `NominalFmc`.
#[derive(Debug, Clone, PartialEq)]
pub struct BeliefFmcPlanner {
    config: BeliefFmcConfig,
}

impl BeliefFmcPlanner {
    /// Crea un nuevo [`BeliefFmcPlanner`] validando `particle_count >= 1`.
    pub fn new(config: BeliefFmcConfig) -> Result<Self, BeliefFmcError> {
        if config.particle_count == 0 {
            return Err(BeliefFmcError::InvalidParticleCount(config.particle_count));
        }
        Ok(Self { config })
    }

    /// Devuelve la configuración del planificador.
    pub const fn config(&self) -> &BeliefFmcConfig {
        &self.config
    }

    /// Observa a través de [`PartialGridObserver`], actualiza [`GridBeliefMap`] y ejecuta
    /// [`BeliefFmcPlanner::plan`] sin acceso al ground truth.
    pub fn observe_and_plan(
        &self,
        observer: &mut impl PartialGridObserver,
        beliefs: &mut GridBeliefMap,
        observed_at: MonotonicInstant,
        episode: u64,
    ) -> Result<BeliefFmcDecision, BeliefFmcError> {
        let obs = observer.observe_occluded(observed_at);
        beliefs.apply_occluded_observation(&obs)?;
        self.plan(beliefs, &obs, episode)
    }

    /// Ejecuta FMC sobre `K` muestras de creencia y compara contra `NominalFmc`.
    pub fn plan(
        &self,
        beliefs: &GridBeliefMap,
        obs: &OccludedObservation,
        episode: u64,
    ) -> Result<BeliefFmcDecision, BeliefFmcError> {
        let effective_beliefs = validate_belief_and_obs(beliefs, obs)?;
        let epistemic_uncertainty_bits = belief_epistemic_uncertainty_bits(&effective_beliefs)?;

        let base_seed = self.config.fmc_config.seed();
        let mut sampling_rng: DeterministicRng =
            base_seed.derive(StreamKey::Agent { episode, agent: 0 });
        let particles = sample_belief_particles(
            &effective_beliefs,
            obs,
            self.config.particle_count,
            &mut sampling_rng,
        )?;

        let mut particle_traces = Vec::with_capacity(particles.len());
        let mut action_votes = [0.0_f64; 5];
        let mut action_values = [0.0_f64; 5];
        let mut total_rollout_transitions = 0_usize;
        let mut weighted_mean_virtual_reward = 0.0_f64;

        for (idx, particle) in particles.iter().enumerate() {
            let agent_id = (idx as u64) + 1;
            let particle_seed = MasterSeed::new(base_seed.derive_seed(StreamKey::Agent {
                episode,
                agent: agent_id,
            }));
            let (first_action, trace) = run_isolated_fmc_on_map(
                &self.config,
                particle.sampled_map.clone(),
                particle_seed,
                episode,
                agent_id,
            )?;

            let slot = action_index(first_action);
            action_votes[slot] += particle.weight;
            action_values[slot] += particle.weight * trace.mean_virtual_reward();
            total_rollout_transitions += trace.rollout_transitions_used();
            weighted_mean_virtual_reward += particle.weight * trace.mean_virtual_reward();

            particle_traces.push(ParticleRolloutTrace {
                particle_index: idx,
                weight: particle.weight,
                first_action,
                trace,
            });
        }

        let mut best_idx = 0_usize;
        for idx in 1..ORDERED_GRID_ACTIONS.len() {
            let better_votes = action_votes[idx] > action_votes[best_idx] + 1e-12;
            let tied_votes = (action_votes[idx] - action_votes[best_idx]).abs() <= 1e-12;
            let better_value = action_values[idx] > action_values[best_idx] + 1e-12;
            if better_votes || (tied_votes && better_value) {
                best_idx = idx;
            }
        }
        let selected_action = ORDERED_GRID_ACTIONS[best_idx];

        let action_weights = [
            (ORDERED_GRID_ACTIONS[0], action_votes[0]),
            (ORDERED_GRID_ACTIONS[1], action_votes[1]),
            (ORDERED_GRID_ACTIONS[2], action_votes[2]),
            (ORDERED_GRID_ACTIONS[3], action_votes[3]),
            (ORDERED_GRID_ACTIONS[4], action_votes[4]),
        ];

        let nominal_map = build_nominal_map_from_belief(&effective_beliefs, obs)?;
        let nominal_agent_id = (particles.len() as u64) + 100;
        let nominal_seed = MasterSeed::new(base_seed.derive_seed(StreamKey::Agent {
            episode,
            agent: nominal_agent_id,
        }));
        let (nominal_action, nominal_trace) = run_isolated_fmc_on_map(
            &self.config,
            nominal_map,
            nominal_seed,
            episode,
            nominal_agent_id,
        )?;

        let comparison = BeliefFmcComparison {
            nominal: NominalFmcSummary {
                mode: BeliefFmcMode::NominalFmc,
                first_action: nominal_action,
                trace: nominal_trace,
            },
            belief_sampled: BeliefSampledFmcSummary {
                mode: BeliefFmcMode::BeliefSampledFmc,
                first_action: selected_action,
                particle_count: particles.len(),
                total_rollout_transitions,
                weighted_mean_virtual_reward,
                action_weights,
            },
            epistemic_uncertainty_bits,
            actions_agree: nominal_action == selected_action,
            prototype_note: BELIEF_FMC_PROTOTYPE_NOTE,
        };

        Ok(BeliefFmcDecision {
            selected_action,
            particles,
            particle_traces,
            comparison,
        })
    }
}

#[cfg(test)]
mod tests {
    use gogamen_core::FmcScoringConfig;

    use super::*;
    use crate::partial_obs::{GridBeliefConfig, GroundTruthGrid};

    fn test_config(particle_count: usize, seed: u64) -> BeliefFmcConfig {
        BeliefFmcConfig {
            particle_count,
            fmc_config: FmcPlannerConfig::new(
                8,
                5,
                gogamen_core::StepDt::from_millis_f64(100.0).expect("valid dt"),
                FmcScoringConfig::default(),
                MasterSeed::new(seed),
            )
            .expect("valid fmc config"),
            budget_per_particle: PlanBudget::with_transitions(128),
            ..BeliefFmcConfig::default()
        }
    }

    #[test]
    fn weighted_belief_particles_execute_isolated_rollouts() {
        let map = GridMap::from_ascii(&["#####", "#S..#", "###.#", "#..G#", "#####"])
            .expect("valid test map");
        let mut truth = GroundTruthGrid::new(map, 1);
        let mut beliefs =
            GridBeliefMap::new(truth.grid_size(), GridBeliefConfig::default()).expect("beliefs");
        let t0 = MonotonicInstant::from_nanos(1_000_000);
        let obs = truth.observer_view().observe_occluded(t0);
        beliefs.apply_occluded_observation(&obs).expect("apply obs");

        let planner = BeliefFmcPlanner::new(test_config(6, 42)).expect("planner");
        let decision = planner.plan(&beliefs, &obs, 0).expect("plan");

        assert_eq!(decision.particles.len(), 6);
        assert_eq!(decision.particle_traces.len(), 6);

        let total_weight: f64 = decision.particles.iter().map(|p| p.weight).sum();
        assert!((total_weight - 1.0).abs() < 1e-12);

        for (idx, (particle, trace)) in decision
            .particles
            .iter()
            .zip(decision.particle_traces.iter())
            .enumerate()
        {
            assert_eq!(trace.particle_index, idx);
            assert!((trace.weight - particle.weight).abs() < 1e-12);
            assert!(trace.trace.rollout_transitions_used() > 0);
            for visible in obs.visible_cells() {
                let payload = visible.payload();
                assert_eq!(
                    particle.sampled_map.cell(payload.pos),
                    Some(payload.kind),
                    "sampled particle must respect observed visible cell {:?}",
                    payload.pos
                );
            }
        }

        // Reproducibilidad determinista exacta con la misma semilla y episodio.
        let decision_repeat = planner.plan(&beliefs, &obs, 0).expect("repeat plan");
        assert_eq!(decision, decision_repeat);
    }

    #[test]
    fn initial_indistinguishability_prevents_hidden_state_leak() {
        // Dos mapas reales distintos que solo difieren en la celda (4, 3) oculta tras el muro (3, *).
        let map_a = GridMap::from_ascii(&["#######", "#S..#.#", "#...#.#", "#...#G#", "#######"])
            .expect("map_a");
        let map_b = GridMap::from_ascii(&["#######", "#S..#.#", "#...#.#", "#...#G#", "#######"])
            .expect("map_b");

        // Modificamos una celda oculta tras el muro vertical x=4 en map_b (p.ej. (5, 1) Hazard vs Empty).
        let mut cells_b = map_b.cells().to_vec();
        let hidden_pos = GridPos::new(5, 1);
        cells_b[hidden_pos.y * map_b.width() + hidden_pos.x] = CellKind::Hazard;
        let map_b = GridMap::new(
            map_b.width(),
            map_b.height(),
            cells_b,
            map_b.start(),
            map_b.goal(),
        )
        .expect("modified map_b");

        let mut truth_a = GroundTruthGrid::new(map_a, 2);
        let mut truth_b = GroundTruthGrid::new(map_b, 2);
        assert!(!truth_a.is_visible(hidden_pos));
        assert!(!truth_b.is_visible(hidden_pos));

        let t0 = MonotonicInstant::from_nanos(5_000_000);
        let obs_a = truth_a.observer_view().observe_occluded(t0);
        let obs_b = truth_b.observer_view().observe_occluded(t0);
        assert_eq!(obs_a, obs_b);

        let mut beliefs_a = GridBeliefMap::new(truth_a.grid_size(), GridBeliefConfig::default())
            .expect("beliefs_a");
        let mut beliefs_b = GridBeliefMap::new(truth_b.grid_size(), GridBeliefConfig::default())
            .expect("beliefs_b");

        let planner = BeliefFmcPlanner::new(test_config(8, 2026)).expect("planner");
        let decision_a = planner
            .observe_and_plan(&mut truth_a.observer_view(), &mut beliefs_a, t0, 1)
            .expect("decision_a");
        let decision_b = planner
            .observe_and_plan(&mut truth_b.observer_view(), &mut beliefs_b, t0, 1)
            .expect("decision_b");

        assert_eq!(decision_a.selected_action, decision_b.selected_action);
        assert_eq!(decision_a.particles, decision_b.particles);
        assert_eq!(decision_a.particle_traces, decision_b.particle_traces);
        assert_eq!(decision_a.comparison, decision_b.comparison);
    }

    #[test]
    fn comparison_records_nominal_vs_belief_sampled_uncertainty_and_not_pomcp() {
        let map =
            GridMap::from_ascii(&["#####", "#S..#", "#.#.#", "#..G#", "#####"]).expect("valid map");
        let mut truth = GroundTruthGrid::new(map, 1);
        let mut beliefs =
            GridBeliefMap::new(truth.grid_size(), GridBeliefConfig::default()).expect("beliefs");

        let t0 = MonotonicInstant::from_nanos(1_000_000);
        let planner = BeliefFmcPlanner::new(test_config(5, 99)).expect("planner");
        let decision_initial = planner
            .observe_and_plan(&mut truth.observer_view(), &mut beliefs, t0, 0)
            .expect("initial decision");

        assert_eq!(
            decision_initial.comparison.nominal.mode,
            BeliefFmcMode::NominalFmc
        );
        assert_eq!(
            decision_initial.comparison.belief_sampled.mode,
            BeliefFmcMode::BeliefSampledFmc
        );
        assert!(decision_initial.comparison.epistemic_uncertainty_bits > 0.0);
        assert!(
            decision_initial
                .comparison
                .prototype_note
                .contains("no es POMCP"),
            "prototype_note must explicitly clarify it is not POMCP"
        );
        assert_eq!(decision_initial.comparison.belief_sampled.particle_count, 5);
        assert!(
            decision_initial
                .comparison
                .belief_sampled
                .total_rollout_transitions
                > 0
        );

        // Al moverse y observar más celdas, la incertidumbre epistémica debe disminuir.
        truth.step_agent(GridAction::Right);
        let t1 = MonotonicInstant::from_nanos(2_000_000);
        let decision_after = planner
            .observe_and_plan(&mut truth.observer_view(), &mut beliefs, t1, 0)
            .expect("after decision");
        assert!(
            decision_after.comparison.epistemic_uncertainty_bits
                < decision_initial.comparison.epistemic_uncertainty_bits
        );
    }
}
