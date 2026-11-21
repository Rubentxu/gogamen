//! Experimento y proxy empírico de libertad futura y ruido sobre [`crate::world::GridWorldModel`].
//!
//! # Alcance epistemológico
//! - Calcula proxies empíricos discretizados por bins (`FutureTrajectoryProxy` y
//!   `FinalStateProxy`) usando [`gogamen_core::estimate_trajectory_entropy_proxy`].
//! - Excluye estados terminales inseguros (`Hazard` / [`TerminalReason::Failure`] /
//!   [`TerminalReason::InvariantViolated`]) de la alcanzabilidad viable.
//! - Distingue entre:
//!   1. Diversidad controlada (`H > 0` y separación causal entre acciones `> 0`),
//!   2. Ausencia de diversidad por bloqueo o restricción (`H == 0` / baja alcanzabilidad), y
//!   3. Diversidad por ruido puro sin control ([`PureNoiseFixture`], donde `H` es alto pero
//!      `controllability == 0.0` y `effective_control_separation == 0.0`).
//! - **Nota:** Este módulo reporta un **proxy de libertad futura** discretizado por bins,
//!   **no** empowerment exacto ni entropía causal física.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, ActionSpace, BinSpec, BinningVersion, DomainError, EntropyError, EntropyProxyKind,
    GenerativeModel, MasterSeed, ModelError, Observation, RngSource, SimDuration, State, StepDt,
    StreamKey, Terminal, TerminalReason, Tick, TrajectoryEntropyEstimate, TrajectoryStep,
    TrajectorySummary, Transition, estimate_trajectory_entropy_proxy,
};

use crate::map::{CellKind, GridMap, GridPos};
use crate::world::{GRID_ACTION_CARDINALITY, GridAction, GridWorldConfig, GridWorldModel};

/// Nota epistemológica explícita incluida en todo [`FutureFreedomReport`].
pub const FUTURE_FREEDOM_EPISTEMOLOGICAL_NOTE: &str = "Proxy empírico de libertad futura discretizado por bins (FutureTrajectoryProxy / FinalStateProxy); no constituye empowerment exacto ni entropía causal física.";

const ALL_GRID_ACTIONS: [GridAction; GRID_ACTION_CARDINALITY] = [
    GridAction::Stay,
    GridAction::Up,
    GridAction::Down,
    GridAction::Left,
    GridAction::Right,
];

/// Errores explícitos al configurar o ejecutar el experimento de proxy de libertad futura.
#[derive(Debug, Clone, PartialEq)]
pub enum FreedomProxyError {
    /// Parámetros de experimento inválidos (`horizon == 0`, `rollouts == 0`, subconjunto vacío, etc.).
    InvalidConfig {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// Error de dominio subyacente.
    Domain(DomainError),
    /// Error del modelo generativo durante la simulación.
    Model(ModelError),
    /// Error de discretización o cálculo de entropía.
    Entropy(EntropyError),
}

impl fmt::Display for FreedomProxyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig { reason } => {
                write!(f, "invalid future freedom experiment config: {reason}")
            }
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Model(err) => write!(f, "model error: {err}"),
            Self::Entropy(err) => write!(f, "entropy error: {err}"),
        }
    }
}

impl Error for FreedomProxyError {}

impl From<DomainError> for FreedomProxyError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

impl From<ModelError> for FreedomProxyError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<EntropyError> for FreedomProxyError {
    fn from(err: EntropyError) -> Self {
        Self::Entropy(err)
    }
}

/// Política explícita de muestreo de acciones para estimar el proxy de libertad futura.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SamplingPolicy {
    /// Muestreo uniforme sobre las 5 acciones discretas de [`GridAction`].
    UniformRandom,
    /// Muestreo uniforme sobre un subconjunto explícito y no vacío de acciones.
    ActionSubset(Vec<GridAction>),
    /// Muestreo no controlado por ruido puro: no ejerce diferenciación causal de acciones.
    NoisyUncontrolled,
}

impl SamplingPolicy {
    /// Valida que la política esté bien formada.
    pub fn validate(&self) -> Result<(), FreedomProxyError> {
        if let Self::ActionSubset(actions) = self
            && actions.is_empty()
        {
            return Err(FreedomProxyError::InvalidConfig {
                reason: "SamplingPolicy::ActionSubset cannot be empty",
            });
        }
        Ok(())
    }

    fn candidate_actions(&self) -> Result<Vec<GridAction>, FreedomProxyError> {
        self.validate()?;
        match self {
            Self::UniformRandom | Self::NoisyUncontrolled => Ok(ALL_GRID_ACTIONS.to_vec()),
            Self::ActionSubset(actions) => Ok(actions.clone()),
        }
    }

    fn distinct_control_actions(&self) -> Result<Vec<GridAction>, FreedomProxyError> {
        self.validate()?;
        match self {
            Self::UniformRandom => Ok(ALL_GRID_ACTIONS.to_vec()),
            Self::ActionSubset(actions) => {
                let mut seen = BTreeSet::new();
                let mut unique = Vec::new();
                for &act in actions {
                    if seen.insert(act as usize) {
                        unique.push(act);
                    }
                }
                Ok(unique)
            }
            Self::NoisyUncontrolled => Ok(Vec::new()),
        }
    }
}

/// Configuración explícita y reproducible del experimento de proxy de libertad futura.
#[derive(Debug, Clone, PartialEq)]
pub struct FutureFreedomConfig {
    /// Semilla determinista maestra del experimento.
    pub seed: MasterSeed,
    /// Horizonte en pasos por rollout (`>= 1`).
    pub horizon: usize,
    /// Número total de rollouts muestreados (`>= 1`).
    pub rollouts: usize,
    /// Política de muestreo aplicada durante los rollouts.
    pub sampling_policy: SamplingPolicy,
    /// Especificación explícita de bins; si es `None`, se usa [`default_grid_bin_spec`].
    pub bin_spec: Option<BinSpec>,
}

impl FutureFreedomConfig {
    /// Construye una configuración explícita de experimento de libertad futura.
    pub const fn new(
        seed: MasterSeed,
        horizon: usize,
        rollouts: usize,
        sampling_policy: SamplingPolicy,
    ) -> Self {
        Self {
            seed,
            horizon,
            rollouts,
            sampling_policy,
            bin_spec: None,
        }
    }

    /// Asocia una especificación de bins explícita a la configuración.
    pub fn with_bin_spec(mut self, bin_spec: BinSpec) -> Self {
        self.bin_spec = Some(bin_spec);
        self
    }

    /// Valida parámetros básicos de configuración.
    pub fn validate(&self) -> Result<(), FreedomProxyError> {
        if self.horizon == 0 {
            return Err(FreedomProxyError::InvalidConfig {
                reason: "horizon must be >= 1",
            });
        }
        if self.rollouts == 0 {
            return Err(FreedomProxyError::InvalidConfig {
                reason: "rollouts must be >= 1",
            });
        }
        self.sampling_policy.validate()?;
        Ok(())
    }
}

impl Default for FutureFreedomConfig {
    fn default() -> Self {
        Self::new(MasterSeed::new(1), 6, 64, SamplingPolicy::UniformRandom)
    }
}

/// Construye la especificación de bins 2D por defecto alineada 1-a-1 con las celdas `(x, y)`
/// del [`GridMap`] (centrando cada coordenada entera en su bin mediante `[-0.5, size - 0.5]`).
pub fn default_grid_bin_spec(map: &GridMap) -> Result<BinSpec, EntropyError> {
    BinSpec::uniform(
        "grid_xy_cells",
        BinningVersion::V1_0,
        vec![
            (-0.5, (map.width() as f64) - 0.5, map.width()),
            (-0.5, (map.height() as f64) - 0.5, map.height()),
        ],
    )
}

/// Fixture de ruido puro sin control causal: implementa [`GenerativeModel`] ignorando por
/// completo la acción recibida y generando transiciones únicamente a partir del `rng` recibido.
///
/// Sirve para demostrar experimentalmente que una entropía alta (`H > 0`) producida por ruido
/// estocástico no implica control efectivo (`controllability == 0.0`).
#[derive(Debug, Clone)]
pub struct PureNoiseFixture {
    config: GridWorldConfig,
    action_space: ActionSpace,
    state: State,
    terminal: Terminal,
}

impl PureNoiseFixture {
    /// Crea un [`PureNoiseFixture`] inicializado en `config.map.start()`.
    pub fn new(config: GridWorldConfig, _seed: MasterSeed) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = ActionSpace::discrete(GRID_ACTION_CARDINALITY)?;
        let start = config.map.start();
        let state = State::new(
            Tick::ZERO,
            SimDuration::ZERO,
            vec![start.x as f64, start.y as f64, 0.0, 0.0, 0.0],
        )?;
        Ok(Self {
            config,
            action_space,
            state,
            terminal: Terminal::Ongoing,
        })
    }

    /// Devuelve la configuración del mapa y recompensas del fixture.
    pub const fn config(&self) -> &GridWorldConfig {
        &self.config
    }

    /// Devuelve el [`State`] inicial/actual del fixture.
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// Devuelve el estado terminal actual del fixture.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }
}

impl GenerativeModel for PureNoiseFixture {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.action_space
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        let vals = state.values();
        if vals.len() != 5 {
            return Err(ModelError::InvalidState {
                reason: "grid state must have 5 elements [x, y, total_reward, last_reward, collisions]",
            });
        }
        let pos = GridPos::new(vals[0] as usize, vals[1] as usize);
        if !self.config.map.is_passable(pos) {
            return Err(ModelError::InvalidState {
                reason: "grid state position is out of bounds or inside a wall",
            });
        }
        if matches!(self.config.map.cell(pos), Some(CellKind::Goal))
            || (self.config.hazard_is_terminal
                && matches!(self.config.map.cell(pos), Some(CellKind::Hazard)))
        {
            return Err(ModelError::TerminalState);
        }

        // La acción es ignorada por completo: la transición depende únicamente del ruido RNG.
        let noise_idx = (rng.next_u64() as usize) % GRID_ACTION_CARDINALITY;
        let noise_action = ALL_GRID_ACTIONS[noise_idx];
        let (dx, dy) = noise_action.delta();
        let nx = pos.x as isize + dx;
        let ny = pos.y as isize + dy;

        let candidate = if nx >= 0 && ny >= 0 {
            let cand = GridPos::new(nx as usize, ny as usize);
            if self.config.map.is_passable(cand) {
                Some(cand)
            } else {
                None
            }
        } else {
            None
        };

        let (next_pos, reward, collided, terminal) = match candidate {
            None => (
                pos,
                self.config.wall_collision_reward,
                true,
                Terminal::Ongoing,
            ),
            Some(next_pos) => match self.config.map.cell(next_pos) {
                Some(CellKind::Goal) => (
                    next_pos,
                    self.config.goal_reward,
                    false,
                    Terminal::Terminated(TerminalReason::GoalReached),
                ),
                Some(CellKind::Hazard) => (
                    next_pos,
                    self.config.hazard_reward,
                    false,
                    if self.config.hazard_is_terminal {
                        Terminal::Terminated(TerminalReason::Failure)
                    } else {
                        Terminal::Ongoing
                    },
                ),
                Some(CellKind::Empty) | Some(CellKind::Wall) | None => {
                    (next_pos, self.config.step_reward, false, Terminal::Ongoing)
                }
            },
        };

        let next_collisions = (vals[4] as u64) + u64::from(collided);
        let next_total_reward = vals[2] + reward;
        let next_state = state
            .advance(
                dt,
                vec![
                    next_pos.x as f64,
                    next_pos.y as f64,
                    next_total_reward,
                    reward,
                    next_collisions as f64,
                ],
            )
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            vec![next_pos.x as f64, next_pos.y as f64],
        )
        .map_err(ModelError::TransitionFailed)?;

        Transition::new(state.tick(), dt, next_state, obs, terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

/// Informe completo del experimento de proxy de libertad futura y separación de ruido.
#[derive(Debug, Clone, PartialEq)]
pub struct FutureFreedomReport {
    /// Semilla determinista utilizada.
    pub seed: MasterSeed,
    /// Horizonte en pasos por rollout.
    pub horizon: usize,
    /// Número total de rollouts ejecutados.
    pub rollouts: usize,
    /// Política de muestreo empleada.
    pub sampling_policy: SamplingPolicy,
    /// Especificación de bins utilizada para discretizar estados/trayectorias.
    pub bin_spec: BinSpec,
    /// Versión explícita del esquema de discretización por bins.
    pub binning_version: BinningVersion,
    /// Número de rollouts viables (sin terminar en estado inseguro).
    pub viable_rollouts: usize,
    /// Número de estados alcanzables distintos y viables (excluyendo terminales inseguros).
    pub viable_reachable_states: usize,
    /// Número de rollouts/transiciones excluidos por alcanzar un terminal inseguro.
    pub excluded_unsafe_terminals: usize,
    /// Proxy de entropía sobre trayectorias futuras completas (`FutureTrajectoryProxy`), en bits.
    pub trajectory_entropy_bits: f64,
    /// Proxy de entropía sobre estados finales (`FinalStateProxy`), en bits.
    pub final_state_entropy_bits: f64,
    /// Estimación completa de `FutureTrajectoryProxy` cuando existen rollouts viables.
    pub trajectory_estimate: Option<TrajectoryEntropyEstimate>,
    /// Estimación completa de `FinalStateProxy` cuando existen rollouts viables.
    pub final_state_estimate: Option<TrajectoryEntropyEstimate>,
    /// Fracción de comparaciones pareadas entre acciones que producen resultados distintos `[0, 1]`.
    pub controllability: f64,
    /// Separación efectiva media (distancia de variación total en `[0, 1]`) entre distribuciones
    /// condicionadas por acción inicial.
    pub effective_control_separation: f64,
    /// Indica si existe diversidad observable en estados/trayectorias viables.
    pub has_diversity: bool,
    /// Indica si la diversidad observada proviene únicamente de ruido sin control (`H > 0` con
    /// separación causal nula).
    pub is_noise_only: bool,
    /// Nota epistemológica explícita: proxy de libertad futura, no empowerment exacto.
    pub epistemological_note: &'static str,
}

fn is_unsafe_terminal(terminal: Terminal) -> bool {
    matches!(
        terminal,
        Terminal::Terminated(TerminalReason::Failure | TerminalReason::InvariantViolated)
    )
}

fn project_state_values(state: &State, bin_dim: usize) -> Vec<f64> {
    let vals = state.values();
    if bin_dim == 2 && vals.len() >= 2 {
        vec![vals[0], vals[1]]
    } else {
        vals.to_vec()
    }
}

fn total_variation_distance(
    a: &BTreeMap<Vec<usize>, usize>,
    total_a: usize,
    b: &BTreeMap<Vec<usize>, usize>,
    total_b: usize,
) -> f64 {
    if total_a == 0 && total_b == 0 {
        return 0.0;
    }
    let inv_a = if total_a > 0 {
        1.0 / (total_a as f64)
    } else {
        0.0
    };
    let inv_b = if total_b > 0 {
        1.0 / (total_b as f64)
    } else {
        0.0
    };
    let mut keys = BTreeSet::new();
    for k in a.keys() {
        keys.insert(k);
    }
    for k in b.keys() {
        keys.insert(k);
    }
    let mut l1 = 0.0_f64;
    for k in keys {
        let pa = (*a.get(k).unwrap_or(&0) as f64) * inv_a;
        let pb = (*b.get(k).unwrap_or(&0) as f64) * inv_b;
        l1 += (pa - pb).abs();
    }
    0.5 * l1
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum PairedOutcome {
    Viable(Vec<Vec<usize>>),
    UnsafeTerminal,
}

fn compute_control_separation<M: GenerativeModel>(
    model: &M,
    initial_state: &State,
    dt: StepDt,
    config: &FutureFreedomConfig,
    bin_spec: &BinSpec,
) -> Result<(f64, f64), FreedomProxyError> {
    let control_actions = config.sampling_policy.distinct_control_actions()?;
    if control_actions.len() < 2 {
        return Ok((0.0, 0.0));
    }

    let action_space = model.action_space();
    let bin_dim = bin_spec.dimension();
    let num_actions = control_actions.len();
    let all_candidates = config.sampling_policy.candidate_actions()?;

    // Distribuciones de estado final (o marcador de terminal inseguro) por acción inicial.
    let mut final_counts: Vec<BTreeMap<Vec<usize>, usize>> = vec![BTreeMap::new(); num_actions];
    let mut action_totals: Vec<usize> = vec![0; num_actions];
    let mut paired_diff_count = 0_usize;
    let mut paired_total_count = 0_usize;

    for rollout_idx in 0..config.rollouts {
        let base_transition_rng = config
            .seed
            .derive(StreamKey::walker(0, 1, rollout_idx as u64));
        let mut cont_rng = config
            .seed
            .derive(StreamKey::walker(0, 2, rollout_idx as u64));

        // Secuencia de continuación compartida (pasos 1..horizon) para aislar el efecto de a_0.
        let mut continuation_actions = Vec::with_capacity(config.horizon.saturating_sub(1));
        for _ in 1..config.horizon {
            let idx = (cont_rng.next_u64() as usize) % all_candidates.len();
            continuation_actions.push(all_candidates[idx]);
        }

        let mut rollout_outcomes = Vec::with_capacity(num_actions);

        for (action_idx, &first_action) in control_actions.iter().enumerate() {
            let mut rng = base_transition_rng.clone();
            let mut state = initial_state.clone();
            let mut traj_bins = Vec::with_capacity(config.horizon);
            let mut hit_unsafe = false;

            for step_idx in 0..config.horizon {
                let grid_act = if step_idx == 0 {
                    first_action
                } else {
                    continuation_actions[step_idx - 1]
                };
                let action = grid_act.to_action(action_space)?;
                let transition = model.step(&state, &action, dt, &mut rng)?;
                if is_unsafe_terminal(transition.terminal()) {
                    hit_unsafe = true;
                    break;
                }
                let next_state = transition.next_state();
                let coords = bin_spec.bin_coords(&project_state_values(next_state, bin_dim))?;
                traj_bins.push(coords);
                state = next_state.clone();
                if transition.terminal().is_terminal() {
                    break;
                }
            }

            let outcome = if hit_unsafe {
                // Bin reservado imposible para separar terminales inseguros de estados viables.
                *final_counts[action_idx]
                    .entry(vec![usize::MAX])
                    .or_insert(0) += 1;
                PairedOutcome::UnsafeTerminal
            } else {
                let final_bin = traj_bins.last().cloned().unwrap_or_default();
                *final_counts[action_idx].entry(final_bin).or_insert(0) += 1;
                PairedOutcome::Viable(traj_bins)
            };
            action_totals[action_idx] += 1;
            rollout_outcomes.push(outcome);
        }

        for i in 0..num_actions {
            for j in (i + 1)..num_actions {
                paired_total_count += 1;
                if rollout_outcomes[i] != rollout_outcomes[j] {
                    paired_diff_count += 1;
                }
            }
        }
    }

    let controllability = if paired_total_count == 0 {
        0.0
    } else {
        (paired_diff_count as f64) / (paired_total_count as f64)
    };

    let mut tv_sum = 0.0_f64;
    let mut tv_pairs = 0_usize;
    for i in 0..num_actions {
        for j in (i + 1)..num_actions {
            tv_sum += total_variation_distance(
                &final_counts[i],
                action_totals[i],
                &final_counts[j],
                action_totals[j],
            );
            tv_pairs += 1;
        }
    }

    let effective_control_separation = if tv_pairs == 0 {
        0.0
    } else {
        tv_sum / (tv_pairs as f64)
    };

    Ok((controllability, effective_control_separation))
}

/// Mide el proxy de libertad futura y la separación de ruido sobre cualquier [`GenerativeModel`].
pub fn measure_future_freedom<M: GenerativeModel>(
    model: &M,
    initial_state: &State,
    map: &GridMap,
    dt: StepDt,
    config: &FutureFreedomConfig,
) -> Result<FutureFreedomReport, FreedomProxyError> {
    config.validate()?;
    let bin_spec = match &config.bin_spec {
        Some(spec) => spec.clone(),
        None => default_grid_bin_spec(map)?,
    };
    let binning_version = bin_spec.version();
    let bin_dim = bin_spec.dimension();
    let candidate_actions = config.sampling_policy.candidate_actions()?;
    let action_space = model.action_space();

    let mut viable_trajectories = Vec::new();
    let mut viable_bins: BTreeSet<Vec<usize>> = BTreeSet::new();
    let mut excluded_unsafe_terminals = 0_usize;

    for rollout_idx in 0..config.rollouts {
        let mut policy_rng = config
            .seed
            .derive(StreamKey::walker(0, 0, rollout_idx as u64));
        let mut transition_rng = config
            .seed
            .derive(StreamKey::walker(0, 1, rollout_idx as u64));

        let mut state = initial_state.clone();
        let mut steps = Vec::with_capacity(config.horizon);
        let mut rollout_bins = Vec::new();
        let mut hit_unsafe = false;

        for _step_idx in 0..config.horizon {
            let act_idx = (policy_rng.next_u64() as usize) % candidate_actions.len();
            let action = candidate_actions[act_idx].to_action(action_space)?;
            let transition = model.step(&state, &action, dt, &mut transition_rng)?;

            if is_unsafe_terminal(transition.terminal()) {
                hit_unsafe = true;
                excluded_unsafe_terminals += 1;
                break;
            }

            let next_state = transition.next_state();
            let proj = project_state_values(next_state, bin_dim);
            rollout_bins.push(bin_spec.bin_coords(&proj)?);
            steps.push(TrajectoryStep::new(next_state.tick(), proj)?);
            state = next_state.clone();

            if transition.terminal().is_terminal() {
                break;
            }
        }

        if !hit_unsafe && !steps.is_empty() {
            for bin in rollout_bins {
                viable_bins.insert(bin);
            }
            viable_trajectories.push(TrajectorySummary::new(steps)?);
        }
    }

    let viable_rollouts = viable_trajectories.len();
    let viable_reachable_states = viable_bins.len();

    let (
        trajectory_estimate,
        trajectory_entropy_bits,
        final_state_estimate,
        final_state_entropy_bits,
    ) = if viable_trajectories.is_empty() {
        (None, 0.0, None, 0.0)
    } else {
        let traj_est = estimate_trajectory_entropy_proxy(
            &viable_trajectories,
            &bin_spec,
            EntropyProxyKind::FutureTrajectoryProxy,
        )?;
        let final_est = estimate_trajectory_entropy_proxy(
            &viable_trajectories,
            &bin_spec,
            EntropyProxyKind::FinalStateProxy,
        )?;
        let traj_h = traj_est.entropy_bits();
        let final_h = final_est.entropy_bits();
        (Some(traj_est), traj_h, Some(final_est), final_h)
    };

    let (controllability, effective_control_separation) =
        compute_control_separation(model, initial_state, dt, config, &bin_spec)?;

    let has_diversity = viable_reachable_states > 1
        || trajectory_entropy_bits > 0.0
        || final_state_entropy_bits > 0.0;
    let is_noise_only =
        has_diversity && controllability == 0.0 && effective_control_separation == 0.0;

    Ok(FutureFreedomReport {
        seed: config.seed,
        horizon: config.horizon,
        rollouts: config.rollouts,
        sampling_policy: config.sampling_policy.clone(),
        bin_spec,
        binning_version,
        viable_rollouts,
        viable_reachable_states,
        excluded_unsafe_terminals,
        trajectory_entropy_bits,
        final_state_entropy_bits,
        trajectory_estimate,
        final_state_estimate,
        controllability,
        effective_control_separation,
        has_diversity,
        is_noise_only,
        epistemological_note: FUTURE_FREEDOM_EPISTEMOLOGICAL_NOTE,
    })
}

/// Ejecuta el experimento de proxy de libertad futura sobre [`GridWorldModel`].
pub fn run_grid_freedom_experiment(
    world_config: GridWorldConfig,
    experiment_config: &FutureFreedomConfig,
) -> Result<FutureFreedomReport, FreedomProxyError> {
    let dt = world_config.dt;
    let model = GridWorldModel::new(world_config, experiment_config.seed)?;
    measure_future_freedom(
        &model,
        model.state(),
        &model.config().map,
        dt,
        experiment_config,
    )
}

/// Ejecuta el experimento de proxy de libertad futura sobre [`PureNoiseFixture`].
pub fn run_pure_noise_freedom_experiment(
    world_config: GridWorldConfig,
    experiment_config: &FutureFreedomConfig,
) -> Result<FutureFreedomReport, FreedomProxyError> {
    let dt = world_config.dt;
    let fixture = PureNoiseFixture::new(world_config, experiment_config.seed)?;
    measure_future_freedom(
        &fixture,
        fixture.state(),
        &fixture.config().map,
        dt,
        experiment_config,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_noise_fixture_shows_high_entropy_does_not_imply_control() {
        let open_map =
            GridMap::from_ascii(&["#######", "#.....#", "#..S..#", "#....G#", "#######"]).unwrap();
        let world_cfg = GridWorldConfig {
            map: open_map,
            ..GridWorldConfig::default()
        };
        let exp_cfg =
            FutureFreedomConfig::new(MasterSeed::new(42), 5, 64, SamplingPolicy::UniformRandom);

        let controlled = run_grid_freedom_experiment(world_cfg.clone(), &exp_cfg).unwrap();
        let noise = run_pure_noise_freedom_experiment(world_cfg, &exp_cfg).unwrap();

        // En el mundo controlado hay diversidad Y separación real entre acciones.
        assert!(controlled.trajectory_entropy_bits > 0.0);
        assert!(controlled.final_state_entropy_bits > 0.0);
        assert!(controlled.controllability > 0.0);
        assert!(controlled.effective_control_separation > 0.0);
        assert!(!controlled.is_noise_only);

        // En el fixture de ruido puro, H es alto pero el control es exactamente nulo.
        assert!(noise.trajectory_entropy_bits > 1.0);
        assert!(noise.final_state_entropy_bits > 1.0);
        assert_eq!(noise.controllability, 0.0);
        assert_eq!(noise.effective_control_separation, 0.0);
        assert!(noise.has_diversity);
        assert!(noise.is_noise_only);
    }

    #[test]
    fn excludes_unsafe_terminals_and_distinguishes_absence_of_diversity() {
        // 1) Mapa con peligros (Hazard) adyacentes: los terminales inseguros no cuentan como viables.
        let hazard_map =
            GridMap::from_ascii(&["#####", "#HHH#", "#HSH#", "#H.G#", "#####"]).unwrap();
        let hazard_cfg = GridWorldConfig {
            map: hazard_map,
            ..GridWorldConfig::default()
        };
        let exp_cfg =
            FutureFreedomConfig::new(MasterSeed::new(7), 2, 50, SamplingPolicy::UniformRandom);
        let hazard_report = run_grid_freedom_experiment(hazard_cfg.clone(), &exp_cfg).unwrap();

        assert!(hazard_report.excluded_unsafe_terminals > 0);
        assert!(hazard_report.viable_rollouts < hazard_report.rollouts);
        // Solo (2,2)='S', (2,3)='.' y (3,3)='G' son viables; ninguna celda 'H' se incluye.
        assert!(hazard_report.viable_reachable_states <= 3);

        // Si solo se permite ir a la derecha (directamente a Hazard), 0 rollouts viables.
        let lethal_cfg = FutureFreedomConfig::new(
            MasterSeed::new(7),
            2,
            16,
            SamplingPolicy::ActionSubset(vec![GridAction::Right]),
        );
        let lethal_report = run_grid_freedom_experiment(hazard_cfg, &lethal_cfg).unwrap();
        assert_eq!(lethal_report.viable_rollouts, 0);
        assert_eq!(lethal_report.viable_reachable_states, 0);
        assert_eq!(lethal_report.excluded_unsafe_terminals, 16);
        assert_eq!(lethal_report.trajectory_entropy_bits, 0.0);
        assert_eq!(lethal_report.final_state_entropy_bits, 0.0);

        // 2) Ausencia de diversidad por bloqueo total (muros en las 4 direcciones): H == 0 y no es ruido.
        let boxed_map = GridMap::from_ascii(&["#####", "###G#", "#S###", "#####"]).unwrap();
        let boxed_cfg = GridWorldConfig {
            map: boxed_map,
            ..GridWorldConfig::default()
        };
        let boxed_report = run_grid_freedom_experiment(boxed_cfg, &exp_cfg).unwrap();
        assert_eq!(boxed_report.excluded_unsafe_terminals, 0);
        assert_eq!(boxed_report.viable_reachable_states, 1);
        assert_eq!(boxed_report.trajectory_entropy_bits, 0.0);
        assert_eq!(boxed_report.final_state_entropy_bits, 0.0);
        assert_eq!(boxed_report.controllability, 0.0);
        assert_eq!(boxed_report.effective_control_separation, 0.0);
        assert!(!boxed_report.has_diversity);
        assert!(!boxed_report.is_noise_only);
    }

    #[test]
    fn report_exports_parameters_proxies_and_epistemological_disclaimer() {
        let map = crate::map::cul_de_sac_map();
        let world_cfg = GridWorldConfig {
            map: map.clone(),
            ..GridWorldConfig::default()
        };
        let exp_cfg = FutureFreedomConfig::new(
            MasterSeed::new(99),
            4,
            32,
            SamplingPolicy::ActionSubset(vec![GridAction::Left, GridAction::Right]),
        );
        let report = run_grid_freedom_experiment(world_cfg, &exp_cfg).unwrap();

        assert_eq!(report.seed, MasterSeed::new(99));
        assert_eq!(report.horizon, 4);
        assert_eq!(report.rollouts, 32);
        assert_eq!(
            report.sampling_policy,
            SamplingPolicy::ActionSubset(vec![GridAction::Left, GridAction::Right])
        );
        assert_eq!(report.bin_spec, default_grid_bin_spec(&map).unwrap());
        assert_eq!(report.binning_version, BinningVersion::V1_0);
        assert_eq!(
            report.trajectory_estimate.as_ref().unwrap().proxy_kind(),
            EntropyProxyKind::FutureTrajectoryProxy
        );
        assert_eq!(
            report.final_state_estimate.as_ref().unwrap().proxy_kind(),
            EntropyProxyKind::FinalStateProxy
        );
        assert!(
            report.epistemological_note.contains("proxy")
                || report.epistemological_note.contains("Proxy")
        );
        assert!(
            report
                .epistemological_note
                .contains("no constituye empowerment exacto")
        );

        // Política NoisyUncontrolled no ejerce separación de control.
        let uncontrolled_cfg = FutureFreedomConfig::new(
            MasterSeed::new(99),
            4,
            32,
            SamplingPolicy::NoisyUncontrolled,
        );
        let uncontrolled = run_grid_freedom_experiment(
            GridWorldConfig {
                map,
                ..GridWorldConfig::default()
            },
            &uncontrolled_cfg,
        )
        .unwrap();
        assert_eq!(uncontrolled.controllability, 0.0);
        assert_eq!(uncontrolled.effective_control_separation, 0.0);
        assert!(uncontrolled.is_noise_only);
    }

    #[test]
    fn rejects_invalid_experiment_configs() {
        let world_cfg = GridWorldConfig::default();
        assert!(matches!(
            run_grid_freedom_experiment(
                world_cfg.clone(),
                &FutureFreedomConfig::new(MasterSeed::new(1), 0, 10, SamplingPolicy::UniformRandom),
            ),
            Err(FreedomProxyError::InvalidConfig { .. })
        ));
        assert!(matches!(
            run_grid_freedom_experiment(
                world_cfg.clone(),
                &FutureFreedomConfig::new(MasterSeed::new(1), 2, 0, SamplingPolicy::UniformRandom),
            ),
            Err(FreedomProxyError::InvalidConfig { .. })
        ));
        assert!(matches!(
            run_grid_freedom_experiment(
                world_cfg,
                &FutureFreedomConfig::new(
                    MasterSeed::new(1),
                    2,
                    10,
                    SamplingPolicy::ActionSubset(Vec::new()),
                ),
            ),
            Err(FreedomProxyError::InvalidConfig { .. })
        ));
    }
}
