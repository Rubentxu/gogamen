//! Baseline POMCP (Partially Observable Monte Carlo Planning) y comparación con `BeliefFmcPlanner`
//! (`std`-only + `gogamen-core`).
//!
//! Implementa:
//! - [`PomcpPlanner`]: árbol de historiales acción-observación con selección UCB1 sobre partículas
//!   de creencia muestreadas exclusivamente desde [`GridBeliefMap`] y [`OccludedObservation`].
//! - [`run_history_vs_omniscient_fixture`]: fixture que distingue explícitamente la política
//!   condicionada a historial de observaciones (POMCP) frente a un rollout omnisciente con acceso
//!   al estado real oculto.
//! - [`run_pomcp_vs_belief_fmc_comparison`]: comparación bajo igualdad estricta de presupuesto
//!   (`PlanBudget` / mismo límite de transiciones y mismas semillas holdout), informando
//!   incertidumbre y rendimiento, y emitiendo [`PomcpComparisonRecommendation`] basada en la
//!   evidencia empírica observada sin prometer superioridad a priori.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_core::{
    BudgetTracker, DeterministicRng, DomainError, FmcPlannerConfig, FmcScoringConfig, MasterSeed,
    ModelError, MonotonicInstant, PlanBudget, RngSource, StepDt, StreamKey,
};

use crate::belief_fmc::{
    BeliefFmcConfig, BeliefFmcError, BeliefFmcPlanner, BeliefParticle,
    belief_epistemic_uncertainty_bits, sample_belief_particles,
};
use crate::map::{CellKind, GridMap, GridPos};
use crate::partial_obs::{
    GridBeliefConfig, GridBeliefMap, GroundTruthGrid, OccludedObservation, PartialGridObserver,
    PartialObsError, has_line_of_sight,
};
use crate::world::{GridAction, GridWorldConfig};

/// Nota explícita sobre la diferencia de información entre historial parcial y rollout omnisciente.
pub const POMCP_INFORMATION_GAP_NOTE: &str = "POMCP solo condiciona su árbol en el historial de observaciones ocluidas y creencias; el rollout omnisciente accede al mapa real oculto desde t=0.";

/// Nota epistémica obligatoria de la recomendación empírica.
pub const POMCP_RECOMMENDATION_DISCLAIMER: &str = "Recomendación condicionada estrictamente al presupuesto, semillas holdout y mapas observados; no promete superioridad a priori de POMCP ni de Belief-sampled FMC.";

/// Orden determinista de acciones discretas del Grid para expansión y desempate.
const ORDERED_GRID_ACTIONS: [GridAction; 5] = [
    GridAction::Up,
    GridAction::Right,
    GridAction::Down,
    GridAction::Left,
    GridAction::Stay,
];

/// Errores explícitos del módulo POMCP y su comparativa.
#[derive(Debug, Clone, PartialEq)]
pub enum PomcpError {
    /// El presupuesto de transiciones debe ser al menos `1`.
    InvalidBudget(usize),
    /// El número de partículas debe ser al menos `1`.
    InvalidParticleCount(usize),
    /// La profundidad máxima de búsqueda debe ser al menos `1`.
    InvalidMaxDepth(usize),
    /// El factor de descuento `gamma` debe ser finito y estar en `(0.0, 1.0]`.
    InvalidDiscount(f64),
    /// La constante de exploración UCB1 debe ser finita y `>= 0.0`.
    InvalidExplorationConstant(f64),
    /// Se requieren al menos una semilla holdout para comparar.
    EmptyHoldoutSeeds,
    /// Error propagado desde [`BeliefFmcError`].
    BeliefFmc(BeliefFmcError),
    /// Error propagado desde [`PartialObsError`].
    PartialObs(PartialObsError),
    /// Error propagado desde [`DomainError`].
    Domain(DomainError),
    /// Error propagado desde [`ModelError`].
    Model(ModelError),
}

impl fmt::Display for PomcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget(n) => {
                write!(f, "invalid POMCP transition budget ({n}): must be >= 1")
            }
            Self::InvalidParticleCount(k) => {
                write!(f, "invalid POMCP particle count ({k}): must be >= 1")
            }
            Self::InvalidMaxDepth(d) => {
                write!(f, "invalid POMCP max depth ({d}): must be >= 1")
            }
            Self::InvalidDiscount(g) => {
                write!(f, "invalid POMCP discount gamma ({g}): must be in (0, 1]")
            }
            Self::InvalidExplorationConstant(c) => {
                write!(
                    f,
                    "invalid POMCP exploration constant ({c}): must be finite and >= 0"
                )
            }
            Self::EmptyHoldoutSeeds => write!(f, "holdout_seeds must not be empty"),
            Self::BeliefFmc(err) => write!(f, "belief fmc error: {err}"),
            Self::PartialObs(err) => write!(f, "partial observation error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Model(err) => write!(f, "model error: {err}"),
        }
    }
}

impl Error for PomcpError {}

impl From<BeliefFmcError> for PomcpError {
    fn from(value: BeliefFmcError) -> Self {
        Self::BeliefFmc(value)
    }
}

impl From<PartialObsError> for PomcpError {
    fn from(value: PartialObsError) -> Self {
        Self::PartialObs(value)
    }
}

impl From<DomainError> for PomcpError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<ModelError> for PomcpError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

/// Configuración explícita de [`PomcpPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpConfig {
    /// Semilla maestra determinista para muestreo de partículas y rollouts.
    pub seed: MasterSeed,
    /// Número de partículas muestreadas en la raíz desde la creencia actual.
    pub particle_count: usize,
    /// Profundidad máxima del árbol y rollout por simulación.
    pub max_depth: usize,
    /// Constante de exploración `c` de UCB1.
    pub exploration_constant: f64,
    /// Factor de descuento `gamma` en `(0.0, 1.0]`.
    pub discount_gamma: f64,
    /// Presupuesto de transiciones simuladas por decisión.
    pub budget: PlanBudget,
    /// Recompensa por paso válido en celda libre.
    pub step_reward: f64,
    /// Penalización por colisión contra muro o borde.
    pub wall_collision_reward: f64,
    /// Recompensa por alcanzar `CellKind::Goal`.
    pub goal_reward: f64,
    /// Recompensa por pisar `CellKind::Hazard`.
    pub hazard_reward: f64,
    /// Si es `true`, pisar `CellKind::Hazard` termina la simulación.
    pub hazard_is_terminal: bool,
    /// Máximo de partículas almacenadas por nodo de historial para inspección.
    pub max_stored_particles_per_node: usize,
    /// Política de selección de acciones durante el rollout (por defecto
    /// [`PomcpRolloutPolicy::UniformRandom`], el comportamiento histórico).
    pub rollout_policy: PomcpRolloutPolicy,
    /// Umbral mínimo de partículas por nodo interno de historial. Si al visitar por primera vez
    /// un nodo su conjunto queda por debajo (típico tras observaciones raras), se reinvigora de
    /// forma determinista (ver [`reinvigorate_particles`]) con un flujo RNG propio
    /// (`StreamKey::Agent { agent: 2 }`) que no altera el RNG de búsqueda. `0` desactiva la
    /// reinvigoración. Se acota a `max_stored_particles_per_node`.
    pub min_node_particles: usize,
}

/// Política de acciones del rollout POMCP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PomcpRolloutPolicy {
    /// Acción uniforme en las 5 acciones (comportamiento histórico; valor por defecto para
    /// preservar las expectativas deterministas existentes).
    #[default]
    UniformRandom,
    /// Uniforme entre las acciones cuyo destino es transitable en el mapa de la partícula
    /// (ya conocido como impasable); si todas están bloqueadas, cae a uniforme sobre las 5.
    AvoidPassableViolations,
}

impl Default for PomcpConfig {
    fn default() -> Self {
        let defaults = GridWorldConfig::default();
        Self {
            seed: MasterSeed::new(42),
            particle_count: 16,
            max_depth: 10,
            exploration_constant: 2.0,
            discount_gamma: 0.95,
            budget: PlanBudget::with_transitions(256),
            step_reward: defaults.step_reward,
            wall_collision_reward: defaults.wall_collision_reward,
            goal_reward: defaults.goal_reward,
            hazard_reward: defaults.hazard_reward,
            hazard_is_terminal: defaults.hazard_is_terminal,
            max_stored_particles_per_node: 16,
            rollout_policy: PomcpRolloutPolicy::UniformRandom,
            min_node_particles: 4,
        }
    }
}

/// Clave determinista de observación local usada para ramificar historiales en el árbol POMCP.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PomcpObservationKey {
    /// Posición observada del agente `(x, y)`.
    pub agent_pos: (usize, usize),
    /// Celdas visibles ordenadas determinísticamente `((x, y), cell_code)`.
    pub visible_cells: Vec<((usize, usize), u8)>,
}

fn cell_kind_code(kind: CellKind) -> u8 {
    match kind {
        CellKind::Empty => 0,
        CellKind::Wall => 1,
        CellKind::Goal => 2,
        CellKind::Hazard => 3,
    }
}

/// Estadísticas agregadas de una acción en la raíz del árbol POMCP.
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpRootActionStat {
    /// Acción discreta candidata.
    pub action: GridAction,
    /// Valor medio estimado `V(root, a)`.
    pub mean_value: f64,
    /// Número de visitas `N(root, a)`.
    pub visit_count: usize,
}

/// Resultado estructurado de una decisión de [`PomcpPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpDecision {
    /// Acción seleccionada en la raíz.
    pub selected_action: GridAction,
    /// Estadísticas por acción en la raíz.
    pub root_action_stats: [PomcpRootActionStat; 5],
    /// Número de simulaciones Monte Carlo completadas bajo el presupuesto.
    pub simulations_completed: usize,
    /// Transiciones simuladas consumidas del [`PlanBudget`].
    pub transitions_used: usize,
    /// Número total de nodos de historial creados en el árbol POMCP.
    pub tree_nodes: usize,
    /// Profundidad máxima alcanzada en el árbol.
    pub max_tree_depth_reached: usize,
    /// Incertidumbre epistémica de la creencia actual en bits (entropía de Shannon).
    pub epistemic_uncertainty_bits: f64,
    /// Varianza muestral de los retornos de simulación en la raíz.
    pub value_variance: f64,
    /// Número de nodos internos reinvigorados por debajo del umbral `min_node_particles`.
    pub reinvigorated_nodes: usize,
    /// Mínimo de partículas almacenadas entre los nodos internos visitados (`0` si no hay).
    pub min_internal_node_particles: usize,
}

#[derive(Debug, Clone)]
struct PomcpActionNode {
    visit_count: usize,
    mean_value: f64,
    children: BTreeMap<PomcpObservationKey, usize>,
}

impl Default for PomcpActionNode {
    fn default() -> Self {
        Self {
            visit_count: 0,
            mean_value: 0.0,
            children: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct PomcpNode {
    visit_count: usize,
    belief_particles: Vec<BeliefParticle>,
    /// Celdas visibles en la observación que define este historial (no se perturban al reinvigorar).
    protected_cells: Vec<(usize, usize)>,
    actions: [PomcpActionNode; 5],
}

/// Estado auxiliar de la búsqueda (reinvigoración).
struct SearchAux {
    reinvigoration_rng: DeterministicRng,
    reinvigorated_nodes: usize,
    root_protected: Vec<(usize, usize)>,
}

impl PomcpNode {
    fn new() -> Self {
        Self {
            visit_count: 0,
            belief_particles: Vec::new(),
            protected_cells: Vec::new(),
            actions: std::array::from_fn(|_| PomcpActionNode::default()),
        }
    }
}

/// Planificador baseline POMCP (árbol de historiales acción-observación + UCB1 sobre partículas).
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpPlanner {
    config: PomcpConfig,
}

impl PomcpPlanner {
    /// Construye un [`PomcpPlanner`] validando parámetros y presupuesto.
    pub fn new(config: PomcpConfig) -> Result<Self, PomcpError> {
        if config.budget.max_transitions() == 0 {
            return Err(PomcpError::InvalidBudget(config.budget.max_transitions()));
        }
        if config.particle_count == 0 {
            return Err(PomcpError::InvalidParticleCount(config.particle_count));
        }
        if config.max_depth == 0 {
            return Err(PomcpError::InvalidMaxDepth(config.max_depth));
        }
        if !config.discount_gamma.is_finite()
            || config.discount_gamma <= 0.0
            || config.discount_gamma > 1.0
        {
            return Err(PomcpError::InvalidDiscount(config.discount_gamma));
        }
        if !config.exploration_constant.is_finite() || config.exploration_constant < 0.0 {
            return Err(PomcpError::InvalidExplorationConstant(
                config.exploration_constant,
            ));
        }
        Ok(Self { config })
    }

    /// Devuelve la configuración del planificador.
    pub const fn config(&self) -> &PomcpConfig {
        &self.config
    }

    /// Observa a través de [`PartialGridObserver`], actualiza [`GridBeliefMap`] y ejecuta
    /// [`PomcpPlanner::plan`] sin acceso al ground truth.
    pub fn observe_and_plan(
        &self,
        observer: &mut impl PartialGridObserver,
        beliefs: &mut GridBeliefMap,
        observed_at: MonotonicInstant,
        episode: u64,
    ) -> Result<PomcpDecision, PomcpError> {
        let obs = observer.observe_occluded(observed_at);
        beliefs.apply_occluded_observation(&obs)?;
        self.plan(beliefs, &obs, episode)
    }

    /// Planifica mediante POMCP a partir únicamente de [`GridBeliefMap`] y [`OccludedObservation`].
    pub fn plan(
        &self,
        beliefs: &GridBeliefMap,
        obs: &OccludedObservation,
        episode: u64,
    ) -> Result<PomcpDecision, PomcpError> {
        let mut effective_beliefs = beliefs.clone();
        effective_beliefs.apply_occluded_observation(obs)?;
        let epistemic_uncertainty_bits = belief_epistemic_uncertainty_bits(&effective_beliefs)?;

        let mut sampling_rng: DeterministicRng = self
            .config
            .seed
            .derive(StreamKey::Agent { episode, agent: 0 });
        let particles = sample_belief_particles(
            &effective_beliefs,
            obs,
            self.config.particle_count,
            &mut sampling_rng,
        )?;

        self.search(
            &particles,
            obs.agent_pos(),
            obs.vision_radius(),
            episode,
            epistemic_uncertainty_bits,
        )
    }

    /// Búsqueda POMCP sobre un conjunto explícito de partículas raíz. La partícula de cada
    /// simulación se muestrea proporcionalmente a `weight` (CDF + búsqueda binaria sobre
    /// `next_f64`); con pesos iguales (o inválidos) se conserva el round-robin histórico.
    fn search(
        &self,
        particles: &[BeliefParticle],
        start_pos: GridPos,
        vision_radius: usize,
        episode: u64,
        epistemic_uncertainty_bits: f64,
    ) -> Result<PomcpDecision, PomcpError> {
        let mut search_rng: DeterministicRng = self
            .config
            .seed
            .derive(StreamKey::Agent { episode, agent: 1 });
        let mut tracker = self.config.budget.tracker();
        let mut nodes = vec![PomcpNode::new()];
        nodes[0].belief_particles = particles
            .iter()
            .take(self.config.max_stored_particles_per_node)
            .cloned()
            .collect();

        let root_protected =
            observe_particle_key(&particles[0].sampled_map, start_pos, vision_radius)
                .visible_cells
                .iter()
                .map(|(p, _)| *p)
                .collect();
        let mut aux = SearchAux {
            reinvigoration_rng: self
                .config
                .seed
                .derive(StreamKey::Agent { episode, agent: 2 }),
            reinvigorated_nodes: 0,
            root_protected,
        };
        let cdf = weight_cdf(particles);
        let mut max_depth_reached = 0_usize;
        let mut sim_returns = Vec::new();
        let mut sim_idx = 0_usize;

        while !tracker.is_exhausted() {
            let particle_idx = match &cdf {
                Some(cdf) => sample_cdf_index(cdf, search_rng.next_f64()),
                None => sim_idx % particles.len(),
            };
            let particle = &particles[particle_idx];
            sim_idx += 1;
            let ret = self.simulate(
                &mut nodes,
                0,
                particle,
                start_pos,
                vision_radius,
                0,
                &mut max_depth_reached,
                &mut tracker,
                &mut search_rng,
                &mut aux,
            );
            sim_returns.push(ret);
        }

        let root = &nodes[0];
        let mut best_idx = 0_usize;
        for idx in 1..ORDERED_GRID_ACTIONS.len() {
            let cur = &root.actions[idx];
            let best = &root.actions[best_idx];
            let cur_visited = cur.visit_count > 0;
            let best_visited = best.visit_count > 0;
            if (cur_visited && !best_visited)
                || (cur_visited == best_visited
                    && (cur.mean_value > best.mean_value + 1e-12
                        || ((cur.mean_value - best.mean_value).abs() <= 1e-12
                            && cur.visit_count > best.visit_count)))
            {
                best_idx = idx;
            }
        }

        let root_action_stats = std::array::from_fn(|i| PomcpRootActionStat {
            action: ORDERED_GRID_ACTIONS[i],
            mean_value: root.actions[i].mean_value,
            visit_count: root.actions[i].visit_count,
        });

        Ok(PomcpDecision {
            selected_action: ORDERED_GRID_ACTIONS[best_idx],
            root_action_stats,
            simulations_completed: sim_returns.len(),
            transitions_used: tracker.transitions_used(),
            tree_nodes: nodes.len(),
            max_tree_depth_reached: max_depth_reached,
            epistemic_uncertainty_bits,
            value_variance: sample_variance(&sim_returns),
            reinvigorated_nodes: aux.reinvigorated_nodes,
            min_internal_node_particles: nodes
                .iter()
                .skip(1)
                .filter(|n| n.visit_count > 0)
                .map(|n| n.belief_particles.len())
                .min()
                .unwrap_or(0),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn simulate(
        &self,
        nodes: &mut Vec<PomcpNode>,
        node_idx: usize,
        particle: &BeliefParticle,
        agent_pos: GridPos,
        vision_radius: usize,
        depth: usize,
        max_depth_reached: &mut usize,
        tracker: &mut BudgetTracker,
        rng: &mut dyn RngSource,
        aux: &mut SearchAux,
    ) -> f64 {
        if depth > *max_depth_reached {
            *max_depth_reached = depth;
        }
        if depth >= self.config.max_depth || tracker.is_exhausted() {
            return 0.0;
        }

        if node_idx != 0 && nodes[node_idx].visit_count == 0 {
            nodes[node_idx].visit_count = 1;
            if nodes[node_idx].belief_particles.len() < self.config.max_stored_particles_per_node {
                nodes[node_idx].belief_particles.push(particle.clone());
            }
            let min = self
                .config
                .min_node_particles
                .min(self.config.max_stored_particles_per_node);
            let node = &mut nodes[node_idx];
            if reinvigorate_particles(
                &mut node.belief_particles,
                min,
                &node.protected_cells,
                &aux.root_protected,
                &mut aux.reinvigoration_rng,
            ) {
                aux.reinvigorated_nodes += 1;
            }
            return self.rollout(&particle.sampled_map, agent_pos, depth, tracker, rng);
        }

        let action_idx = self.select_ucb1_action(&nodes[node_idx]);
        let action = ORDERED_GRID_ACTIONS[action_idx];

        if !tracker.try_consume_transition() {
            return 0.0;
        }

        let (next_pos, reward, terminal) =
            step_particle_map(&particle.sampled_map, agent_pos, action, self.config());
        let future_return = if terminal {
            0.0
        } else {
            let obs_key = observe_particle_key(&particle.sampled_map, next_pos, vision_radius);
            let existing_child = nodes[node_idx].actions[action_idx]
                .children
                .get(&obs_key)
                .copied();
            let child_idx = match existing_child {
                Some(idx) => idx,
                None => {
                    let new_idx = nodes.len();
                    let mut node = PomcpNode::new();
                    node.protected_cells = obs_key.visible_cells.iter().map(|(p, _)| *p).collect();
                    nodes.push(node);
                    nodes[node_idx].actions[action_idx]
                        .children
                        .insert(obs_key, new_idx);
                    new_idx
                }
            };
            self.simulate(
                nodes,
                child_idx,
                particle,
                next_pos,
                vision_radius,
                depth + 1,
                max_depth_reached,
                tracker,
                rng,
                aux,
            )
        };

        let total_return = reward + self.config.discount_gamma * future_return;
        let node = &mut nodes[node_idx];
        node.visit_count += 1;
        let action_node = &mut node.actions[action_idx];
        action_node.visit_count += 1;
        let n = action_node.visit_count as f64;
        action_node.mean_value += (total_return - action_node.mean_value) / n;
        total_return
    }

    fn select_ucb1_action(&self, node: &PomcpNode) -> usize {
        for (idx, action_node) in node.actions.iter().enumerate() {
            if action_node.visit_count == 0 {
                return idx;
            }
        }

        let log_parent = (node.visit_count.max(1) as f64).ln();
        let mut best_idx = 0_usize;
        let mut best_score = f64::NEG_INFINITY;
        for (idx, action_node) in node.actions.iter().enumerate() {
            let explore = self.config.exploration_constant
                * (log_parent / (action_node.visit_count as f64)).sqrt();
            let score = action_node.mean_value + explore;
            if score > best_score + 1e-12 {
                best_score = score;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn rollout(
        &self,
        map: &GridMap,
        pos: GridPos,
        start_depth: usize,
        tracker: &mut BudgetTracker,
        rng: &mut dyn RngSource,
    ) -> f64 {
        self.rollout_traced(map, pos, start_depth, tracker, rng).0
    }

    /// Rollout que además devuelve el número de colisiones (movimiento no-`Stay` contra celda
    /// impasable o borde).
    fn rollout_traced(
        &self,
        map: &GridMap,
        mut pos: GridPos,
        start_depth: usize,
        tracker: &mut BudgetTracker,
        rng: &mut dyn RngSource,
    ) -> (f64, usize) {
        let mut total_return = 0.0_f64;
        let mut discount = 1.0_f64;
        let mut collisions = 0_usize;

        for _ in start_depth..self.config.max_depth {
            if !tracker.try_consume_transition() {
                break;
            }
            let action = self.rollout_action(map, pos, rng);
            let (next_pos, reward, terminal) = step_particle_map(map, pos, action, self.config());
            if action != GridAction::Stay && next_pos == pos {
                collisions += 1;
            }
            total_return += discount * reward;
            pos = next_pos;
            if terminal {
                break;
            }
            discount *= self.config.discount_gamma;
        }
        (total_return, collisions)
    }

    fn rollout_action(&self, map: &GridMap, pos: GridPos, rng: &mut dyn RngSource) -> GridAction {
        if self.config.rollout_policy == PomcpRolloutPolicy::AvoidPassableViolations {
            let mut allowed = [GridAction::Stay; 5];
            let mut n = 0_usize;
            for action in ORDERED_GRID_ACTIONS {
                if move_is_passable(map, pos, action) {
                    allowed[n] = action;
                    n += 1;
                }
            }
            if n > 0 {
                return allowed[(rng.next_u64() as usize) % n];
            }
        }
        ORDERED_GRID_ACTIONS[(rng.next_u64() as usize) % ORDERED_GRID_ACTIONS.len()]
    }
}

fn move_is_passable(map: &GridMap, pos: GridPos, action: GridAction) -> bool {
    let (dx, dy) = action.delta();
    let nx = pos.x as isize + dx;
    let ny = pos.y as isize + dy;
    nx >= 0 && ny >= 0 && map.is_passable(GridPos::new(nx as usize, ny as usize))
}

/// Sanitiza pesos: no finitos o `<= 0` pasan a `0.0`.
fn sanitized_weight(w: f64) -> f64 {
    if w.is_finite() && w > 0.0 { w } else { 0.0 }
}

/// CDF acumulada de pesos de partículas. `None` si los pesos sanitizados son todos iguales o
/// suman 0 (caso uniforme: se usa round-robin determinista, idéntico al comportamiento previo).
fn weight_cdf(particles: &[BeliefParticle]) -> Option<Vec<f64>> {
    let first = sanitized_weight(particles.first()?.weight);
    if particles
        .iter()
        .all(|p| sanitized_weight(p.weight) == first)
    {
        return None;
    }
    let mut acc = 0.0_f64;
    let cdf: Vec<f64> = particles
        .iter()
        .map(|p| {
            acc += sanitized_weight(p.weight);
            acc
        })
        .collect();
    if acc.is_finite() && acc > 0.0 {
        Some(cdf)
    } else {
        None
    }
}

/// Índice muestreado con `u` en `[0, 1)` sobre una CDF no normalizada (búsqueda binaria).
fn sample_cdf_index(cdf: &[f64], u: f64) -> usize {
    let total = cdf[cdf.len() - 1];
    let target = u * total;
    cdf.partition_point(|c| *c <= target).min(cdf.len() - 1)
}

/// Reinvigoración determinista: si `particles.len() < min`, completa hasta `min` clonando
/// partículas existentes (elegidas con el RNG dado) y perturbando una celda libre/muro que no
/// pertenezca a `protected`/`root_protected` ni sea inicio/meta, de modo que la observación del
/// historial se conserve. Normaliza pesos a `1/len`. Devuelve `true` si añadió partículas.
fn reinvigorate_particles(
    particles: &mut Vec<BeliefParticle>,
    min: usize,
    protected: &[(usize, usize)],
    root_protected: &[(usize, usize)],
    rng: &mut dyn RngSource,
) -> bool {
    if min == 0 || particles.is_empty() || particles.len() >= min {
        return false;
    }
    while particles.len() < min {
        let src = (rng.next_u64() as usize) % particles.len();
        let base = &particles[src].sampled_map;
        let (w, h) = (base.width(), base.height());
        let mut cells = base.cells().to_vec();
        for _ in 0..8 {
            let idx = (rng.next_u64() as usize) % cells.len();
            let (x, y) = (idx % w, idx / w);
            let pos = GridPos::new(x, y);
            if pos == base.start()
                || pos == base.goal()
                || protected.contains(&(x, y))
                || root_protected.contains(&(x, y))
            {
                continue;
            }
            match cells[idx] {
                CellKind::Empty => cells[idx] = CellKind::Wall,
                CellKind::Wall => cells[idx] = CellKind::Empty,
                _ => continue,
            }
            break;
        }
        let map =
            GridMap::new(w, h, cells, base.start(), base.goal()).unwrap_or_else(|_| base.clone());
        particles.push(BeliefParticle::new(map, 0.0));
    }
    let w = 1.0 / particles.len() as f64;
    for p in particles.iter_mut() {
        p.weight = w;
    }
    true
}

fn step_particle_map(
    map: &GridMap,
    pos: GridPos,
    action: GridAction,
    config: &PomcpConfig,
) -> (GridPos, f64, bool) {
    let (dx, dy) = action.delta();
    let nx = pos.x as isize + dx;
    let ny = pos.y as isize + dy;
    if nx < 0 || ny < 0 {
        return (pos, config.wall_collision_reward, false);
    }
    let candidate = GridPos::new(nx as usize, ny as usize);
    if !map.is_passable(candidate) {
        return (pos, config.wall_collision_reward, false);
    }
    match map.cell(candidate) {
        Some(CellKind::Goal) => (candidate, config.goal_reward, true),
        Some(CellKind::Hazard) => (candidate, config.hazard_reward, config.hazard_is_terminal),
        _ => (candidate, config.step_reward, false),
    }
}

fn observe_particle_key(
    map: &GridMap,
    agent_pos: GridPos,
    vision_radius: usize,
) -> PomcpObservationKey {
    let r = vision_radius as isize;
    let cx = agent_pos.x as isize;
    let cy = agent_pos.y as isize;
    let mut visible_cells = Vec::new();

    for dy in -r..=r {
        for dx in -r..=r {
            let tx = cx + dx;
            let ty = cy + dy;
            if tx < 0 || ty < 0 {
                continue;
            }
            let pos = GridPos::new(tx as usize, ty as usize);
            if !map.in_bounds(pos) || !has_line_of_sight(map, agent_pos, pos) {
                continue;
            }
            if let Some(kind) = map.cell(pos) {
                visible_cells.push(((pos.x, pos.y), cell_kind_code(kind)));
            }
        }
    }

    PomcpObservationKey {
        agent_pos: (agent_pos.x, agent_pos.y),
        visible_cells,
    }
}

fn sample_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

// ============================================================================
// AC #1: Política condicionada a historial (POMCP) vs rollout omnisciente
// ============================================================================

/// Resumen de un rollout omnisciente que conoce el mapa real oculto desde `t=0`.
#[derive(Debug, Clone, PartialEq)]
pub struct OmniscientRolloutSummary {
    /// Primera acción elegida usando el estado real oculto.
    pub selected_action: GridAction,
    /// Retorno medio estimado de la mejor acción en el mapa real.
    pub best_return: f64,
    /// Transiciones consumidas en la evaluación omnisciente.
    pub transitions_used: usize,
    /// Retorno medio observado por primera acción candidata.
    pub action_returns: [(GridAction, f64); 5],
}

/// Ejecuta un rollout omnisciente directamente sobre el `GridMap` real oculto (acceso privilegiado).
pub fn run_omniscient_rollout(
    true_map: &GridMap,
    agent_pos: GridPos,
    config: &PomcpConfig,
    episode: u64,
) -> Result<OmniscientRolloutSummary, PomcpError> {
    if config.budget.max_transitions() == 0 {
        return Err(PomcpError::InvalidBudget(config.budget.max_transitions()));
    }
    let mut tracker = config.budget.tracker();
    let mut rng: DeterministicRng = config.seed.derive(StreamKey::Agent {
        episode,
        agent: 999,
    });

    let mut sums = [0.0_f64; 5];
    let mut counts = [0_usize; 5];
    let mut step_idx = 0_usize;

    while !tracker.is_exhausted() {
        let action_idx = step_idx % ORDERED_GRID_ACTIONS.len();
        step_idx += 1;
        if !tracker.try_consume_transition() {
            break;
        }
        let first_action = ORDERED_GRID_ACTIONS[action_idx];
        let (mut pos, first_reward, first_terminal) =
            step_particle_map(true_map, agent_pos, first_action, config);

        let mut total_return = first_reward;
        if !first_terminal {
            let mut discount = config.discount_gamma;
            for _ in 1..config.max_depth {
                if !tracker.try_consume_transition() {
                    break;
                }
                let next_idx = (rng.next_u64() as usize) % ORDERED_GRID_ACTIONS.len();
                let next_act = ORDERED_GRID_ACTIONS[next_idx];
                let (next_pos, reward, terminal) =
                    step_particle_map(true_map, pos, next_act, config);
                total_return += discount * reward;
                pos = next_pos;
                if terminal {
                    break;
                }
                discount *= config.discount_gamma;
            }
        }

        sums[action_idx] += total_return;
        counts[action_idx] += 1;
    }

    let mut action_returns = [(GridAction::Stay, 0.0_f64); 5];
    let mut best_idx = 0_usize;
    let mut best_return = f64::NEG_INFINITY;
    for i in 0..5 {
        let mean = if counts[i] > 0 {
            sums[i] / (counts[i] as f64)
        } else {
            0.0
        };
        action_returns[i] = (ORDERED_GRID_ACTIONS[i], mean);
        if counts[i] > 0 && mean > best_return + 1e-12 {
            best_return = mean;
            best_idx = i;
        }
    }

    Ok(OmniscientRolloutSummary {
        selected_action: ORDERED_GRID_ACTIONS[best_idx],
        best_return: if best_return.is_finite() {
            best_return
        } else {
            0.0
        },
        transitions_used: tracker.transitions_used(),
        action_returns,
    })
}

/// Reporte comparativo entre POMCP (historial de observaciones) y rollout omnisciente.
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryVsOmniscientReport {
    /// Decisión de POMCP en `world_a` antes de observar la zona oculta (`t=0`).
    pub pomcp_t0_world_a: PomcpDecision,
    /// Decisión de POMCP en `world_b` antes de observar la zona oculta (`t=0`).
    pub pomcp_t0_world_b: PomcpDecision,
    /// `true` si POMCP produce exactamente la misma decisión en ambos mundos en `t=0`.
    pub pomcp_indistinguishable_at_t0: bool,
    /// Decisión de POMCP en `world_a` tras avanzar y observar (`t=1`).
    pub pomcp_t1_world_a: PomcpDecision,
    /// Decisión de POMCP en `world_b` tras avanzar y observar (`t=1`).
    pub pomcp_t1_world_b: PomcpDecision,
    /// `true` si POMCP cambia de acción cuando el historial de observaciones distingue ambos mundos.
    pub pomcp_diverges_after_observation: bool,
    /// Rollout omnisciente en `world_a` desde `t=0`.
    pub omniscient_t0_world_a: OmniscientRolloutSummary,
    /// Rollout omnisciente en `world_b` desde `t=0`.
    pub omniscient_t0_world_b: OmniscientRolloutSummary,
    /// `true` si el rollout omnisciente ya distingue ambos mundos en `t=0`.
    pub omniscient_diverges_at_t0: bool,
    /// Nota explicativa de la diferencia de información disponible.
    pub information_gap_note: &'static str,
}

/// Fixture que demuestra la diferencia entre política condicionada a historial (POMCP) y
/// rollout omnisciente sobre dos mundos inicialmente indistinguibles.
pub fn run_history_vs_omniscient_fixture(
    config: &PomcpConfig,
) -> Result<HistoryVsOmniscientReport, PomcpError> {
    // En t=0 el agente está en (1,2) con vision_radius=1.
    // Las celdas (3,1) y (3,3) están fuera del radio de visión (dx=2 > 1).
    // Al avanzar por el corredor superior hasta (2,1), la celda (3,1) queda adyacente y visible.
    let map_a = GridMap::from_ascii(&["######", "#..G.#", "#S#..#", "#..H.#", "######"])?;
    let map_b = GridMap::from_ascii(&["######", "#..H.#", "#S#..#", "#..G.#", "######"])?;

    let mut truth_a = GroundTruthGrid::new(map_a.clone(), 1);
    let mut truth_b = GroundTruthGrid::new(map_b.clone(), 1);
    let mut beliefs_a = GridBeliefMap::new(truth_a.grid_size(), GridBeliefConfig::default())?;
    let mut beliefs_b = GridBeliefMap::new(truth_b.grid_size(), GridBeliefConfig::default())?;

    let planner = PomcpPlanner::new(config.clone())?;
    let t0 = MonotonicInstant::from_nanos(1_000_000);
    let pomcp_t0_world_a =
        planner.observe_and_plan(&mut truth_a.observer_view(), &mut beliefs_a, t0, 1)?;
    let pomcp_t0_world_b =
        planner.observe_and_plan(&mut truth_b.observer_view(), &mut beliefs_b, t0, 1)?;

    let pomcp_indistinguishable_at_t0 = pomcp_t0_world_a == pomcp_t0_world_b;

    let omniscient_t0_world_a = run_omniscient_rollout(&map_a, truth_a.agent_pos(), config, 1)?;
    let omniscient_t0_world_b = run_omniscient_rollout(&map_b, truth_b.agent_pos(), config, 1)?;
    let omniscient_diverges_at_t0 =
        omniscient_t0_world_a.selected_action != omniscient_t0_world_b.selected_action;

    // Avanzamos ambos mundos por el corredor superior hasta (2, 1) donde (4, 1) queda visible.
    truth_a.step_agent(GridAction::Up);
    truth_a.step_agent(GridAction::Right);
    truth_b.step_agent(GridAction::Up);
    truth_b.step_agent(GridAction::Right);

    let t1 = MonotonicInstant::from_nanos(2_000_000);
    let pomcp_t1_world_a =
        planner.observe_and_plan(&mut truth_a.observer_view(), &mut beliefs_a, t1, 2)?;
    let pomcp_t1_world_b =
        planner.observe_and_plan(&mut truth_b.observer_view(), &mut beliefs_b, t1, 2)?;
    let pomcp_diverges_after_observation =
        pomcp_t1_world_a.selected_action != pomcp_t1_world_b.selected_action;

    Ok(HistoryVsOmniscientReport {
        pomcp_t0_world_a,
        pomcp_t0_world_b,
        pomcp_indistinguishable_at_t0,
        pomcp_t1_world_a,
        pomcp_t1_world_b,
        pomcp_diverges_after_observation,
        omniscient_t0_world_a,
        omniscient_t0_world_b,
        omniscient_diverges_at_t0,
        information_gap_note: POMCP_INFORMATION_GAP_NOTE,
    })
}

// ============================================================================
// AC #2 & AC #3: Comparación bajo igualdad estricta de presupuesto y recomendación
// ============================================================================

/// Configuración de comparación justa entre [`PomcpPlanner`] y [`BeliefFmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpComparisonConfig {
    /// Semillas holdout compartidas exactamente por ambos algoritmos.
    pub holdout_seeds: Vec<u64>,
    /// Presupuesto estricto de transiciones simuladas por paso de decisión (`PlanBudget`).
    pub transition_budget_per_step: usize,
    /// Límite máximo de pasos por episodio.
    pub max_episode_steps: usize,
    /// Radio de visión local del observador parcial.
    pub vision_radius: usize,
    /// Número de partículas raíz en [`PomcpPlanner`].
    pub pomcp_particle_count: usize,
    /// Profundidad máxima de [`PomcpPlanner`].
    pub pomcp_max_depth: usize,
    /// Constante de exploración UCB1 de [`PomcpPlanner`].
    pub pomcp_exploration_constant: f64,
    /// Descuento `gamma` de [`PomcpPlanner`].
    pub pomcp_discount_gamma: f64,
    /// Número de partículas `K` en [`BeliefFmcPlanner`].
    pub belief_fmc_particle_count: usize,
    /// Horizonte de [`BeliefFmcPlanner`].
    pub belief_fmc_horizon: usize,
    /// Tamaño de población de [`BeliefFmcPlanner`].
    pub belief_fmc_population_size: usize,
}

impl Default for PomcpComparisonConfig {
    fn default() -> Self {
        Self {
            holdout_seeds: vec![101, 102, 103, 104, 105, 106],
            transition_budget_per_step: 256,
            max_episode_steps: 24,
            vision_radius: 2,
            pomcp_particle_count: 16,
            pomcp_max_depth: 10,
            pomcp_exploration_constant: 2.0,
            pomcp_discount_gamma: 0.95,
            belief_fmc_particle_count: 8,
            belief_fmc_horizon: 6,
            belief_fmc_population_size: 6,
        }
    }
}

/// Métricas por episodio para un algoritmo bajo presupuesto compartido.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmEpisodeMetrics {
    /// Semilla holdout del episodio.
    pub seed: u64,
    /// `true` si alcanzó `CellKind::Goal`.
    pub succeeded: bool,
    /// Retorno acumulado real del episodio.
    pub total_return: f64,
    /// Pasos ejecutados hasta terminar o agotar `max_episode_steps`.
    pub steps: usize,
    /// Total de transiciones simuladas consumidas en el episodio.
    pub total_transitions_consumed: usize,
    /// Máximo de transiciones consumidas en un solo paso de decisión.
    pub max_step_transitions_consumed: usize,
    /// Incertidumbre epistémica inicial en bits.
    pub initial_belief_entropy_bits: f64,
    /// Incertidumbre epistémica final en bits.
    pub final_belief_entropy_bits: f64,
    /// Varianza media de valor en las decisiones del episodio.
    pub mean_decision_value_variance: f64,
}

/// Resumen agregado de un algoritmo sobre todas las semillas holdout.
#[derive(Debug, Clone, PartialEq)]
pub struct AlgorithmComparisonSummary {
    /// Nombre explícito del algoritmo (`"PomcpPlanner"` o `"BeliefFmcPlanner"`).
    pub algorithm_name: &'static str,
    /// Número de episodios evaluados.
    pub episodes: usize,
    /// Número de episodios exitosos.
    pub success_count: usize,
    /// Tasa de éxito en `[0.0, 1.0]`.
    pub success_rate: f64,
    /// Retorno medio por episodio.
    pub mean_return: f64,
    /// Varianza muestral del retorno entre episodios.
    pub return_variance: f64,
    /// Pasos medios por episodio.
    pub mean_steps: f64,
    /// Total de transiciones consumidas en la suite.
    pub total_transitions_consumed: usize,
    /// Media de transiciones consumidas por episodio.
    pub mean_transitions_per_episode: f64,
    /// Máximo de transiciones consumidas en cualquier paso de decisión.
    pub max_step_transitions_consumed: usize,
    /// Entropía inicial media de la creencia en bits.
    pub mean_initial_belief_entropy_bits: f64,
    /// Entropía final media de la creencia en bits.
    pub mean_final_belief_entropy_bits: f64,
    /// Reducción media de entropía de creencia en bits.
    pub mean_entropy_reduction_bits: f64,
    /// Varianza media de valor en las decisiones.
    pub mean_decision_value_variance: f64,
}

/// Fila emparejada de un episodio holdout.
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeComparisonRow {
    /// Semilla holdout compartida.
    pub seed: u64,
    /// Métricas de [`PomcpPlanner`].
    pub pomcp: AlgorithmEpisodeMetrics,
    /// Métricas de [`BeliefFmcPlanner`].
    pub belief_fmc: AlgorithmEpisodeMetrics,
}

/// Decisión explícita sobre la siguiente inversión basada en el resultado empírico observado.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NextInvestmentChoice {
    /// Invertir en refinar POMCP (observado superior en la muestra evaluada).
    InvestInPomcp,
    /// Invertir en refinar Belief-sampled FMC (observado superior en la muestra evaluada).
    InvestInBeliefFmc,
    /// Empate técnico o trade-off mixto: investigar enfoque híbrido o ampliar evaluación.
    HybridOrFurtherInvestigation,
}

/// Recomendación empírica explícita sin prometer superioridad a priori.
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpComparisonRecommendation {
    /// Decisión recomendada.
    pub choice: NextInvestmentChoice,
    /// Diferencia de tasa de éxito (`pomcp - belief_fmc`).
    pub success_rate_delta_pomcp_minus_fmc: f64,
    /// Diferencia de retorno medio (`pomcp - belief_fmc`).
    pub mean_return_delta_pomcp_minus_fmc: f64,
    /// Razón de transiciones consumidas (`pomcp / belief_fmc`).
    pub transition_efficiency_ratio: f64,
    /// Diferencia en reducción de incertidumbre en bits (`pomcp - belief_fmc`).
    pub uncertainty_reduction_delta_bits: f64,
    /// Justificación textual basada estrictamente en los datos observados.
    pub rationale: String,
    /// Descargo epistémico que prohíbe prometer superioridad a priori.
    pub epistemic_disclaimer: &'static str,
}

/// Reporte completo de comparación entre [`PomcpPlanner`] y [`BeliefFmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct PomcpVsBeliefFmcReport {
    /// Configuración usada en la comparación.
    pub config: PomcpComparisonConfig,
    /// Resumen agregado de [`PomcpPlanner`].
    pub pomcp_summary: AlgorithmComparisonSummary,
    /// Resumen agregado de [`BeliefFmcPlanner`].
    pub belief_fmc_summary: AlgorithmComparisonSummary,
    /// Resultados por semilla holdout.
    pub rows: Vec<EpisodeComparisonRow>,
    /// Recomendación explícita sobre la siguiente inversión.
    pub recommendation: PomcpComparisonRecommendation,
}

fn generate_holdout_map(seed: u64) -> Result<GridMap, PomcpError> {
    let mut rng = MasterSeed::new(seed).derive(StreamKey::Agent {
        episode: seed,
        agent: 777,
    });
    let upper_goal = rng.next_u64().is_multiple_of(2);
    let hazard_mid = rng.next_u64().is_multiple_of(2);

    let mut rows = [
        *b"#######",
        *b"#..#..#",
        *b"#S....#",
        *b"#..#..#",
        *b"#######",
    ];

    if upper_goal {
        rows[1][5] = b'G';
        rows[3][5] = b'H';
    } else {
        rows[1][5] = b'H';
        rows[3][5] = b'G';
    }
    if hazard_mid {
        rows[2][5] = b'H';
    }

    let ascii: Vec<String> = rows
        .iter()
        .map(|row| String::from_utf8_lossy(row).into_owned())
        .collect();
    let refs: Vec<&str> = ascii.iter().map(String::as_str).collect();
    Ok(GridMap::from_ascii(&refs)?)
}

fn step_ground_truth_episode(
    truth: &mut GroundTruthGrid,
    map: &GridMap,
    action: GridAction,
    defaults: &GridWorldConfig,
) -> (f64, bool, bool) {
    let prev = truth.agent_pos();
    let (dx, dy) = action.delta();
    let nx = prev.x as isize + dx;
    let ny = prev.y as isize + dy;
    let blocked = nx < 0 || ny < 0 || !map.is_passable(GridPos::new(nx as usize, ny as usize));

    let next = truth.step_agent(action);
    if blocked {
        return (defaults.wall_collision_reward, false, false);
    }

    match map.cell(next) {
        Some(CellKind::Goal) => (defaults.goal_reward, true, true),
        Some(CellKind::Hazard) => (defaults.hazard_reward, defaults.hazard_is_terminal, false),
        _ => (defaults.step_reward, false, false),
    }
}

fn run_pomcp_episode(
    map: &GridMap,
    seed: u64,
    config: &PomcpComparisonConfig,
) -> Result<AlgorithmEpisodeMetrics, PomcpError> {
    let defaults = GridWorldConfig::default();
    let pomcp_cfg = PomcpConfig {
        seed: MasterSeed::new(seed),
        particle_count: config.pomcp_particle_count,
        max_depth: config.pomcp_max_depth,
        exploration_constant: config.pomcp_exploration_constant,
        discount_gamma: config.pomcp_discount_gamma,
        budget: PlanBudget::with_transitions(config.transition_budget_per_step),
        ..PomcpConfig::default()
    };
    let planner = PomcpPlanner::new(pomcp_cfg)?;

    let mut truth = GroundTruthGrid::new(map.clone(), config.vision_radius);
    let mut beliefs = GridBeliefMap::new(truth.grid_size(), GridBeliefConfig::default())?;
    let initial_entropy = belief_epistemic_uncertainty_bits(&beliefs)?;

    let mut total_return = 0.0_f64;
    let mut total_transitions = 0_usize;
    let mut max_step_transitions = 0_usize;
    let mut variances = Vec::new();
    let mut succeeded = false;
    let mut steps = 0_usize;

    for step in 0..config.max_episode_steps {
        steps += 1;
        let t = MonotonicInstant::from_nanos(((step as u128) + 1) * 1_000_000);
        let decision =
            planner.observe_and_plan(&mut truth.observer_view(), &mut beliefs, t, step as u64)?;
        total_transitions += decision.transitions_used;
        max_step_transitions = max_step_transitions.max(decision.transitions_used);
        variances.push(decision.value_variance);

        let (reward, terminal, reached_goal) =
            step_ground_truth_episode(&mut truth, map, decision.selected_action, &defaults);
        total_return += reward;
        if terminal {
            succeeded = reached_goal;
            break;
        }
    }

    let final_entropy = belief_epistemic_uncertainty_bits(&beliefs)?;
    let mean_var = if variances.is_empty() {
        0.0
    } else {
        variances.iter().sum::<f64>() / (variances.len() as f64)
    };

    Ok(AlgorithmEpisodeMetrics {
        seed,
        succeeded,
        total_return,
        steps,
        total_transitions_consumed: total_transitions,
        max_step_transitions_consumed: max_step_transitions,
        initial_belief_entropy_bits: initial_entropy,
        final_belief_entropy_bits: final_entropy,
        mean_decision_value_variance: mean_var,
    })
}

fn run_belief_fmc_episode(
    map: &GridMap,
    seed: u64,
    config: &PomcpComparisonConfig,
) -> Result<AlgorithmEpisodeMetrics, PomcpError> {
    let defaults = GridWorldConfig::default();
    let per_particle_budget =
        (config.transition_budget_per_step / config.belief_fmc_particle_count.max(1)).max(1);
    let fmc_cfg = BeliefFmcConfig {
        particle_count: config.belief_fmc_particle_count,
        fmc_config: FmcPlannerConfig::new(
            config.belief_fmc_population_size,
            config.belief_fmc_horizon,
            StepDt::from_millis_f64(100.0).expect("valid dt"),
            FmcScoringConfig::default(),
            MasterSeed::new(seed),
        )
        .map_err(|_| {
            PomcpError::Domain(DomainError::InvalidTransition {
                reason: "invalid FmcPlannerConfig in PomcpComparisonConfig",
            })
        })?,
        budget_per_particle: PlanBudget::with_transitions(per_particle_budget),
        ..BeliefFmcConfig::default()
    };
    let planner = BeliefFmcPlanner::new(fmc_cfg)?;

    let mut truth = GroundTruthGrid::new(map.clone(), config.vision_radius);
    let mut beliefs = GridBeliefMap::new(truth.grid_size(), GridBeliefConfig::default())?;
    let initial_entropy = belief_epistemic_uncertainty_bits(&beliefs)?;

    let mut total_return = 0.0_f64;
    let mut total_transitions = 0_usize;
    let mut max_step_transitions = 0_usize;
    let mut variances = Vec::new();
    let mut succeeded = false;
    let mut steps = 0_usize;

    for step in 0..config.max_episode_steps {
        steps += 1;
        let t = MonotonicInstant::from_nanos(((step as u128) + 1) * 1_000_000);
        let decision =
            planner.observe_and_plan(&mut truth.observer_view(), &mut beliefs, t, step as u64)?;
        let step_transitions = decision.comparison.belief_sampled.total_rollout_transitions;
        total_transitions += step_transitions;
        max_step_transitions = max_step_transitions.max(step_transitions);

        let particle_rewards: Vec<f64> = decision
            .particle_traces
            .iter()
            .map(|tr| tr.trace.mean_virtual_reward())
            .collect();
        variances.push(sample_variance(&particle_rewards));

        let (reward, terminal, reached_goal) =
            step_ground_truth_episode(&mut truth, map, decision.selected_action, &defaults);
        total_return += reward;
        if terminal {
            succeeded = reached_goal;
            break;
        }
    }

    let final_entropy = belief_epistemic_uncertainty_bits(&beliefs)?;
    let mean_var = if variances.is_empty() {
        0.0
    } else {
        variances.iter().sum::<f64>() / (variances.len() as f64)
    };

    Ok(AlgorithmEpisodeMetrics {
        seed,
        succeeded,
        total_return,
        steps,
        total_transitions_consumed: total_transitions,
        max_step_transitions_consumed: max_step_transitions,
        initial_belief_entropy_bits: initial_entropy,
        final_belief_entropy_bits: final_entropy,
        mean_decision_value_variance: mean_var,
    })
}

fn summarize_algorithm(
    algorithm_name: &'static str,
    episodes: &[AlgorithmEpisodeMetrics],
) -> AlgorithmComparisonSummary {
    let n = episodes.len().max(1) as f64;
    let success_count = episodes.iter().filter(|e| e.succeeded).count();
    let returns: Vec<f64> = episodes.iter().map(|e| e.total_return).collect();
    let mean_return = returns.iter().sum::<f64>() / n;
    let mean_steps = episodes.iter().map(|e| e.steps as f64).sum::<f64>() / n;
    let total_transitions_consumed: usize =
        episodes.iter().map(|e| e.total_transitions_consumed).sum();
    let max_step_transitions_consumed = episodes
        .iter()
        .map(|e| e.max_step_transitions_consumed)
        .max()
        .unwrap_or(0);
    let mean_initial_belief_entropy_bits = episodes
        .iter()
        .map(|e| e.initial_belief_entropy_bits)
        .sum::<f64>()
        / n;
    let mean_final_belief_entropy_bits = episodes
        .iter()
        .map(|e| e.final_belief_entropy_bits)
        .sum::<f64>()
        / n;
    let mean_decision_value_variance = episodes
        .iter()
        .map(|e| e.mean_decision_value_variance)
        .sum::<f64>()
        / n;

    AlgorithmComparisonSummary {
        algorithm_name,
        episodes: episodes.len(),
        success_count,
        success_rate: (success_count as f64) / n,
        mean_return,
        return_variance: sample_variance(&returns),
        mean_steps,
        total_transitions_consumed,
        mean_transitions_per_episode: (total_transitions_consumed as f64) / n,
        max_step_transitions_consumed,
        mean_initial_belief_entropy_bits,
        mean_final_belief_entropy_bits,
        mean_entropy_reduction_bits: mean_initial_belief_entropy_bits
            - mean_final_belief_entropy_bits,
        mean_decision_value_variance,
    }
}

/// Construye [`PomcpComparisonRecommendation`] estrictamente a partir de los resúmenes observados.
pub fn build_pomcp_recommendation(
    pomcp: &AlgorithmComparisonSummary,
    fmc: &AlgorithmComparisonSummary,
) -> PomcpComparisonRecommendation {
    let success_delta = pomcp.success_rate - fmc.success_rate;
    let return_delta = pomcp.mean_return - fmc.mean_return;
    let uncertainty_delta = pomcp.mean_entropy_reduction_bits - fmc.mean_entropy_reduction_bits;
    let efficiency_ratio = if fmc.total_transitions_consumed == 0 {
        1.0
    } else {
        (pomcp.total_transitions_consumed as f64) / (fmc.total_transitions_consumed as f64)
    };

    let choice = if success_delta > 0.05 || (success_delta.abs() <= 0.05 && return_delta > 0.5) {
        NextInvestmentChoice::InvestInPomcp
    } else if success_delta < -0.05 || (success_delta.abs() <= 0.05 && return_delta < -0.5) {
        NextInvestmentChoice::InvestInBeliefFmc
    } else {
        NextInvestmentChoice::HybridOrFurtherInvestigation
    };

    let rationale = format!(
        "En {} episodios holdout bajo presupuesto idéntico: POMCP obtuvo success_rate={:.3}, mean_return={:.3}, entropy_reduction={:.3} bits; BeliefFmc obtuvo success_rate={:.3}, mean_return={:.3}, entropy_reduction={:.3} bits (delta_success={:+.3}, delta_return={:+.3}, transition_ratio={:.3}).",
        pomcp.episodes,
        pomcp.success_rate,
        pomcp.mean_return,
        pomcp.mean_entropy_reduction_bits,
        fmc.success_rate,
        fmc.mean_return,
        fmc.mean_entropy_reduction_bits,
        success_delta,
        return_delta,
        efficiency_ratio
    );

    PomcpComparisonRecommendation {
        choice,
        success_rate_delta_pomcp_minus_fmc: success_delta,
        mean_return_delta_pomcp_minus_fmc: return_delta,
        transition_efficiency_ratio: efficiency_ratio,
        uncertainty_reduction_delta_bits: uncertainty_delta,
        rationale,
        epistemic_disclaimer: POMCP_RECOMMENDATION_DISCLAIMER,
    }
}

/// Ejecuta la comparación justa entre [`PomcpPlanner`] y [`BeliefFmcPlanner`] bajo igualdad
/// estricta de presupuesto y mismas semillas holdout.
pub fn run_pomcp_vs_belief_fmc_comparison(
    config: &PomcpComparisonConfig,
) -> Result<PomcpVsBeliefFmcReport, PomcpError> {
    if config.holdout_seeds.is_empty() {
        return Err(PomcpError::EmptyHoldoutSeeds);
    }
    if config.transition_budget_per_step == 0 {
        return Err(PomcpError::InvalidBudget(config.transition_budget_per_step));
    }
    if config.belief_fmc_particle_count == 0 {
        return Err(PomcpError::InvalidParticleCount(
            config.belief_fmc_particle_count,
        ));
    }

    let mut rows = Vec::with_capacity(config.holdout_seeds.len());
    let mut pomcp_all = Vec::with_capacity(config.holdout_seeds.len());
    let mut fmc_all = Vec::with_capacity(config.holdout_seeds.len());

    for &seed in &config.holdout_seeds {
        let map = generate_holdout_map(seed)?;
        let pomcp = run_pomcp_episode(&map, seed, config)?;
        let belief_fmc = run_belief_fmc_episode(&map, seed, config)?;
        pomcp_all.push(pomcp.clone());
        fmc_all.push(belief_fmc.clone());
        rows.push(EpisodeComparisonRow {
            seed,
            pomcp,
            belief_fmc,
        });
    }

    let pomcp_summary = summarize_algorithm("PomcpPlanner", &pomcp_all);
    let belief_fmc_summary = summarize_algorithm("BeliefFmcPlanner", &fmc_all);
    let recommendation = build_pomcp_recommendation(&pomcp_summary, &belief_fmc_summary);

    Ok(PomcpVsBeliefFmcReport {
        config: config.clone(),
        pomcp_summary,
        belief_fmc_summary,
        rows,
        recommendation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_history_conditioned_pomcp_is_invariant_at_t0_and_diverges_after_observation_unlike_omniscient()
     {
        let config = PomcpConfig {
            seed: MasterSeed::new(2026),
            particle_count: 12,
            max_depth: 10,
            exploration_constant: 2.0,
            discount_gamma: 0.95,
            budget: PlanBudget::with_transitions(256),
            ..PomcpConfig::default()
        };

        let report = run_history_vs_omniscient_fixture(&config).expect("fixture report");

        // En t=0, ambos mundos producen exactamente la misma observación ocluida: POMCP no puede distinguirlos.
        assert!(
            report.pomcp_indistinguishable_at_t0,
            "POMCP must produce identical decisions on indistinguishable observation histories at t=0"
        );
        assert!(
            report.pomcp_t0_world_a.tree_nodes > 1,
            "POMCP must build a non-trivial action-observation history tree"
        );

        // El rollout omnisciente sí accede al mapa real oculto y los distingue desde t=0.
        assert!(
            report.omniscient_diverges_at_t0,
            "omniscient rollout must distinguish world_a and world_b at t=0 using hidden state"
        );
        assert_eq!(report.omniscient_t0_world_a.selected_action, GridAction::Up);
        assert_eq!(
            report.omniscient_t0_world_b.selected_action,
            GridAction::Down
        );

        // Cuando POMCP avanza y observa la celda oculta en t=1, su política condicionada a historial diverge.
        assert!(
            report.pomcp_diverges_after_observation,
            "POMCP must diverge at t=1 once observation history reveals Goal vs Hazard"
        );
        assert!(
            report.information_gap_note.contains("rollout omnisciente"),
            "information_gap_note must document the information difference"
        );
    }

    #[test]
    fn ac2_equal_budget_comparison_reports_uncertainty_and_performance_for_same_holdout_seeds() {
        let config = PomcpComparisonConfig {
            holdout_seeds: vec![11, 22, 33, 44],
            transition_budget_per_step: 256,
            max_episode_steps: 20,
            vision_radius: 2,
            pomcp_particle_count: 16,
            pomcp_max_depth: 10,
            pomcp_exploration_constant: 2.0,
            pomcp_discount_gamma: 0.95,
            belief_fmc_particle_count: 8,
            belief_fmc_horizon: 6,
            belief_fmc_population_size: 6,
        };

        let report = run_pomcp_vs_belief_fmc_comparison(&config).expect("comparison report");
        let repeat = run_pomcp_vs_belief_fmc_comparison(&config).expect("repeat report");
        assert_eq!(report, repeat, "comparison must be deterministic");

        assert_eq!(report.rows.len(), config.holdout_seeds.len());
        assert_eq!(report.pomcp_summary.episodes, 4);
        assert_eq!(report.belief_fmc_summary.episodes, 4);

        // Igualdad estricta de presupuesto por paso en ambos algoritmos.
        assert!(
            report.pomcp_summary.max_step_transitions_consumed <= config.transition_budget_per_step
        );
        assert!(
            report.belief_fmc_summary.max_step_transitions_consumed
                <= config.transition_budget_per_step
        );

        // Incertidumbre (entropía y varianza) y rendimiento (tasa de éxito, retorno, transiciones) informados.
        assert!(report.pomcp_summary.mean_initial_belief_entropy_bits > 0.0);
        assert!(report.belief_fmc_summary.mean_initial_belief_entropy_bits > 0.0);
        assert!(report.pomcp_summary.mean_entropy_reduction_bits >= 0.0);
        assert!(report.belief_fmc_summary.mean_entropy_reduction_bits >= 0.0);
        assert!(report.pomcp_summary.total_transitions_consumed > 0);
        assert!(report.belief_fmc_summary.total_transitions_consumed > 0);
        assert!(report.pomcp_summary.mean_decision_value_variance >= 0.0);
        assert!(report.belief_fmc_summary.mean_decision_value_variance >= 0.0);

        for (row, &expected_seed) in report.rows.iter().zip(config.holdout_seeds.iter()) {
            assert_eq!(row.seed, expected_seed);
            assert_eq!(row.pomcp.seed, expected_seed);
            assert_eq!(row.belief_fmc.seed, expected_seed);
        }
    }

    #[test]
    fn ac3_recommendation_reflects_empirical_outcome_without_promising_a_priori_superiority() {
        let strong_pomcp = AlgorithmComparisonSummary {
            algorithm_name: "PomcpPlanner",
            episodes: 6,
            success_count: 5,
            success_rate: 5.0 / 6.0,
            mean_return: 8.0,
            return_variance: 1.0,
            mean_steps: 7.0,
            total_transitions_consumed: 3000,
            mean_transitions_per_episode: 500.0,
            max_step_transitions_consumed: 256,
            mean_initial_belief_entropy_bits: 10.0,
            mean_final_belief_entropy_bits: 4.0,
            mean_entropy_reduction_bits: 6.0,
            mean_decision_value_variance: 0.5,
        };
        let weak_fmc = AlgorithmComparisonSummary {
            algorithm_name: "BeliefFmcPlanner",
            episodes: 6,
            success_count: 2,
            success_rate: 2.0 / 6.0,
            mean_return: 1.0,
            return_variance: 2.0,
            mean_steps: 12.0,
            total_transitions_consumed: 3200,
            mean_transitions_per_episode: 533.3,
            max_step_transitions_consumed: 256,
            mean_initial_belief_entropy_bits: 10.0,
            mean_final_belief_entropy_bits: 6.0,
            mean_entropy_reduction_bits: 4.0,
            mean_decision_value_variance: 0.8,
        };

        let rec_pomcp = build_pomcp_recommendation(&strong_pomcp, &weak_fmc);
        assert_eq!(rec_pomcp.choice, NextInvestmentChoice::InvestInPomcp);
        assert!(
            rec_pomcp
                .epistemic_disclaimer
                .contains("no promete superioridad a priori")
        );

        let rec_fmc = build_pomcp_recommendation(&weak_fmc, &strong_pomcp);
        assert_eq!(rec_fmc.choice, NextInvestmentChoice::InvestInBeliefFmc);
        assert!(
            rec_fmc
                .epistemic_disclaimer
                .contains("no promete superioridad a priori")
        );

        let rec_tie = build_pomcp_recommendation(&strong_pomcp, &strong_pomcp);
        assert_eq!(
            rec_tie.choice,
            NextInvestmentChoice::HybridOrFurtherInvestigation
        );
        assert!((rec_tie.success_rate_delta_pomcp_minus_fmc).abs() < 1e-12);
    }

    // ---- GOG-122: muestreo ponderado, rollout informado y reinvigoración ----

    fn gog122_config(sims: usize, depth: usize) -> PomcpConfig {
        PomcpConfig {
            seed: MasterSeed::new(122),
            max_depth: depth,
            budget: PlanBudget::with_transitions(sims),
            ..PomcpConfig::default()
        }
    }

    /// Con `goal_above` el destino de `Up` desde (0,1) es la meta; si no, es un muro (colisión).
    fn two_world_map(goal_above: bool) -> GridMap {
        let rows: [&str; 2] = if goal_above {
            ["G#", "S."]
        } else {
            ["#G", "S."]
        };
        GridMap::from_ascii(&rows).expect("two world map")
    }

    #[test]
    fn gog122_ac1_weighted_root_sampling_converges_to_posterior_expectation() {
        // Mundo A (peso 0.9): Up -> meta (+10). Mundo B (peso 0.1): Up -> muro (-0.5).
        // max_depth = 1 => el valor de Up en la raíz es exactamente la recompensa inmediata.
        let cfg = gog122_config(20_000, 1);
        let (goal_r, wall_r) = (cfg.goal_reward, cfg.wall_collision_reward);
        let planner = PomcpPlanner::new(cfg).expect("planner");
        let start = GridPos::new(0, 1);
        let mk = |wa: f64, wb: f64| {
            vec![
                BeliefParticle::new(two_world_map(true), wa),
                BeliefParticle::new(two_world_map(false), wb),
            ]
        };
        let weighted = planner
            .search(&mk(0.9, 0.1), start, 1, 0, 0.0)
            .expect("weighted");
        let up = weighted.root_action_stats[0].mean_value;
        let posterior = 0.9 * goal_r + 0.1 * wall_r; // 8.95
        let legacy_unweighted = 0.5 * goal_r + 0.5 * wall_r; // 4.75
        // Tolerancia 0.15 (error estándar ~ 10*0.3/sqrt(n_up) << 0.15 con >= 4000 visitas a Up).
        assert!(weighted.root_action_stats[0].visit_count > 4_000);
        assert!(
            (up - posterior).abs() < 0.15,
            "up={up} posterior={posterior}"
        );
        assert!(
            (up - legacy_unweighted).abs() > 3.0,
            "weighted estimate must be far from legacy 50/50 value {legacy_unweighted}"
        );

        // Con pesos iguales se conserva el round-robin histórico: ~0.5/0.5.
        let uniform = planner
            .search(&mk(0.5, 0.5), start, 1, 0, 0.0)
            .expect("uniform");
        assert!((uniform.root_action_stats[0].mean_value - legacy_unweighted).abs() < 0.15);

        // Determinismo con la misma semilla.
        let again = planner
            .search(&mk(0.9, 0.1), start, 1, 0, 0.0)
            .expect("again");
        assert_eq!(weighted.root_action_stats, again.root_action_stats);
    }

    #[test]
    fn gog122_invalid_weights_fall_back_safely_without_panic() {
        let cdf_none = |ws: &[f64]| {
            let ps: Vec<_> = ws
                .iter()
                .map(|w| BeliefParticle::new(two_world_map(true), *w))
                .collect();
            weight_cdf(&ps).is_none()
        };
        assert!(cdf_none(&[0.0, 0.0]));
        assert!(cdf_none(&[f64::NAN, f64::NAN]));
        assert!(cdf_none(&[-1.0, f64::INFINITY]));
        assert!(!cdf_none(&[f64::NAN, 1.0]));
        let cdf = [0.0, 0.9, 1.0];
        // Peso 0 nunca se elige.
        assert_eq!(sample_cdf_index(&cdf, 0.0), 1);
        assert_eq!(sample_cdf_index(&cdf, 0.95), 2);
        assert_eq!(sample_cdf_index(&cdf, 0.999_999), 2);

        let planner = PomcpPlanner::new(gog122_config(200, 3)).expect("planner");
        let ps = vec![
            BeliefParticle::new(two_world_map(true), f64::NAN),
            BeliefParticle::new(two_world_map(false), -3.0),
        ];
        let d = planner
            .search(&ps, GridPos::new(0, 1), 1, 0, 0.0)
            .expect("no panic");
        assert!(d.simulations_completed > 0);
    }

    #[test]
    fn gog122_ac2_avoid_passable_violations_reduces_variance_and_collisions() {
        // Habitación cerrada de 2 celdas; la meta es inalcanzable: toda la varianza viene de colisiones.
        let map = GridMap::from_ascii(&["####G", "#S.##", "#####"]).expect("room");
        let start = GridPos::new(1, 1);
        let run = |policy: PomcpRolloutPolicy| {
            let planner = PomcpPlanner::new(PomcpConfig {
                rollout_policy: policy,
                ..gog122_config(10_000, 10)
            })
            .expect("planner");
            let mut returns = Vec::new();
            let mut collisions = 0_usize;
            for ep in 0..200_u64 {
                let mut rng: DeterministicRng = MasterSeed::new(7).derive(StreamKey::Agent {
                    episode: ep,
                    agent: 1,
                });
                let mut tracker = PlanBudget::with_transitions(10_000).tracker();
                let (r, c) = planner.rollout_traced(&map, start, 0, &mut tracker, &mut rng);
                returns.push(r);
                collisions += c;
            }
            (sample_variance(&returns), collisions)
        };
        let (var_uniform, col_uniform) = run(PomcpRolloutPolicy::UniformRandom);
        let (var_avoid, col_avoid) = run(PomcpRolloutPolicy::AvoidPassableViolations);
        assert!(col_uniform > 0);
        assert_eq!(col_avoid, 0, "informed rollout must never bump known walls");
        assert!(var_uniform > 0.0);
        assert!(
            var_avoid < var_uniform,
            "avoid variance {var_avoid} must be < uniform {var_uniform}"
        );
    }

    #[test]
    fn gog122_avoid_policy_falls_back_when_all_moves_blocked() {
        // Celda aislada: solo Stay es transitable (Stay permanece en la propia celda).
        let map = GridMap::from_ascii(&["#S#G"]).expect("map");
        let planner = PomcpPlanner::new(PomcpConfig {
            rollout_policy: PomcpRolloutPolicy::AvoidPassableViolations,
            ..gog122_config(100, 5)
        })
        .expect("planner");
        let mut rng: DeterministicRng = MasterSeed::new(1).derive(StreamKey::Agent {
            episode: 0,
            agent: 1,
        });
        for _ in 0..20 {
            assert_eq!(
                planner.rollout_action(&map, GridPos::new(1, 0), &mut rng),
                GridAction::Stay
            );
        }
        // Posición impasable (ningún destino transitable salvo vecinos): no entra en pánico.
        let _ = planner.rollout_action(&map, GridPos::new(0, 0), &mut rng);
    }

    #[test]
    fn gog122_ac3_reinvigoration_prevents_particle_collapse_after_rare_observation() {
        // Mundo raro (peso 0.03) con muro arriba: sus historiales casi no se visitan.
        let start = GridPos::new(2, 2);
        let rows_common =
            GridMap::from_ascii(&[".....", ".....", "..S..", ".....", "....G"]).expect("map");
        let mut rare_cells = rows_common.cells().to_vec();
        rare_cells[5 + 2] = CellKind::Wall; // celda (2,1) encima del agente
        let rare =
            GridMap::new(5, 5, rare_cells, rows_common.start(), rows_common.goal()).expect("rare");
        let particles = vec![
            BeliefParticle::new(rows_common.clone(), 0.97),
            BeliefParticle::new(rare, 0.03),
        ];
        let run = |min: usize| {
            PomcpPlanner::new(PomcpConfig {
                min_node_particles: min,
                ..gog122_config(300, 4)
            })
            .expect("planner")
            .search(&particles, start, 1, 0, 0.0)
            .expect("decision")
        };
        let off = run(0);
        let on = run(4);
        assert_eq!(
            off.min_internal_node_particles, 1,
            "collapse without reinvigoration"
        );
        assert_eq!(off.reinvigorated_nodes, 0);
        assert!(
            on.min_internal_node_particles >= 4,
            "{}",
            on.min_internal_node_particles
        );
        assert!(on.reinvigorated_nodes > 0);
        // La reinvigoración usa un flujo RNG aparte: la decisión de búsqueda no cambia.
        assert_eq!(off.root_action_stats, on.root_action_stats);
        assert_eq!(on, run(4), "deterministic");

        // No degenerado: desde 1 sola partícula se obtienen mapas distintos con pesos normalizados.
        let mut set = vec![BeliefParticle::new(rows_common.clone(), 1.0)];
        let mut rng: DeterministicRng = MasterSeed::new(9).derive(StreamKey::Agent {
            episode: 0,
            agent: 2,
        });
        assert!(reinvigorate_particles(
            &mut set,
            6,
            &[(2, 1)],
            &[],
            &mut rng
        ));
        assert_eq!(set.len(), 6);
        let distinct = set
            .iter()
            .skip(1)
            .filter(|p| p.sampled_map != rows_common)
            .count();
        assert!(distinct >= 3, "distinct perturbed maps: {distinct}");
        assert!(
            set.iter()
                .all(|p| p.sampled_map.cell(GridPos::new(2, 1)) == Some(CellKind::Empty))
        );
        let total: f64 = set.iter().map(|p| p.weight).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
}
