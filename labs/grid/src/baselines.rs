//! Baselines de control y techo para el laboratorio Grid (`RandomSeeded` y `PrivilegedBfs`).

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use gogamen_core::{
    Action, Actuator, DeterminismMetadata, DiscreteAction, DomainError, EnvironmentError,
    EnvironmentObserver, MasterSeed, RngSource, StreamKey, Terminal, TerminalReason, TraceHasher,
};

use crate::map::{CellKind, GridMap, GridPos};
use crate::world::{GRID_ACTION_CARDINALITY, GridAction, GridWorldConfig, GridWorldEnv};

/// Tipo de política baseline ejecutada sobre el mundo Grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaselineKind {
    /// Política aleatoria uniforme reproducible mediante [`MasterSeed`].
    RandomSeeded,
    /// Búsqueda en anchura (BFS) con acceso privilegiado al mapa completo (`GridMap`).
    PrivilegedBfs,
    /// Baseline Monte Carlo de horizonte fijo (random shooting) sobre [`crate::GridWorldModel`].
    MonteCarloShooting,
}

impl BaselineKind {
    /// Indica si la política utiliza información privilegiada completa del mapa.
    pub const fn is_privileged_full_info(self) -> bool {
        match self {
            Self::RandomSeeded | Self::MonteCarloShooting => false,
            Self::PrivilegedBfs => true,
        }
    }
}

/// Presupuesto de ejecución comparable entre baselines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BaselineBudget {
    /// Número máximo de ticks a ejecutar en [`GridWorldEnv`].
    pub max_ticks: u64,
    /// Número máximo de transiciones o expansiones evaluables.
    pub max_transitions: u64,
}

impl BaselineBudget {
    /// Construye un presupuesto explícito de ticks y transiciones.
    pub const fn new(max_ticks: u64, max_transitions: u64) -> Self {
        Self {
            max_ticks,
            max_transitions,
        }
    }
}

impl Default for BaselineBudget {
    fn default() -> Self {
        Self {
            max_ticks: 1_000,
            max_transitions: 100_000,
        }
    }
}

/// Métricas comparables producidas por un baseline en [`GridWorldEnv`].
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineMetrics {
    /// Identificador del tipo de baseline ejecutado.
    pub kind: BaselineKind,
    /// Marca explícita de si el baseline usó información privilegiada completa.
    pub is_privileged_full_info: bool,
    /// Semilla usada cuando el baseline es estocástico (`RandomSeeded`).
    pub seed: Option<MasterSeed>,
    /// Indica si el agente alcanzó la meta `CellKind::Goal`.
    pub reached_goal: bool,
    /// Indica explícitamente que no existe camino alcanzable hasta la meta.
    pub no_path: bool,
    /// Ticks ejecutados en [`GridWorldEnv`].
    pub ticks_executed: u64,
    /// Transiciones evaluadas (pasos en entorno y/o expansiones de búsqueda).
    pub transitions_evaluated: u64,
    /// Recompensa acumulada en el episodio.
    pub total_reward: f64,
    /// Colisiones contra muros o bordes durante la ejecución.
    pub collisions: u64,
    /// Longitud del camino en acciones si se alcanzó meta o se halló ruta.
    pub path_len: Option<u64>,
    /// Hash determinista de la traza de ejecución.
    pub trace_hash: u64,
}

fn map_env_error(err: EnvironmentError) -> DomainError {
    match err {
        EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
        EnvironmentError::InvalidAction(d) => d,
        _ => DomainError::InvalidTransition {
            reason: "failed to apply baseline action in grid environment",
        },
    }
}

fn observe_env(env: &GridWorldEnv) -> Result<gogamen_core::Observation, DomainError> {
    EnvironmentObserver::observe(env).map_err(|_| DomainError::InvalidTransition {
        reason: "failed to observe grid environment",
    })
}

fn hash_step(
    hasher: &mut TraceHasher,
    env: &GridWorldEnv,
    obs: &gogamen_core::Observation,
    action_idx: usize,
    next_obs: &gogamen_core::Observation,
    terminal: Terminal,
) {
    hasher.write_tick(env.tick());
    hasher.write_observation(obs);
    hasher.write_u64(action_idx as u64);
    hasher.write_observation(next_obs);
    hasher.write_f64(env.last_reward());
    hasher.write_terminal(terminal);
}

/// Ejecuta el baseline aleatorio reproducible por `seed` en [`GridWorldEnv`].
pub fn run_random_baseline(
    map: &GridMap,
    mut config: GridWorldConfig,
    seed: MasterSeed,
    budget: BaselineBudget,
) -> Result<BaselineMetrics, DomainError> {
    config.map = map.clone();
    let mut env = GridWorldEnv::new(config)?;
    let mut rng = seed.derive(StreamKey::Episode(0));
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(seed));

    let mut ticks_executed = 0_u64;
    let mut transitions_evaluated = 0_u64;

    while ticks_executed < budget.max_ticks && transitions_evaluated < budget.max_transitions {
        let obs = observe_env(&env)?;
        let action_idx = (rng.next_u64() % (GRID_ACTION_CARDINALITY as u64)) as usize;
        let action = Action::Discrete(DiscreteAction::new(action_idx, GRID_ACTION_CARDINALITY)?);
        let outcome = env.apply(&action).map_err(map_env_error)?;

        hash_step(
            &mut hasher,
            &env,
            &obs,
            action_idx,
            outcome.observation(),
            outcome.terminal(),
        );

        ticks_executed += 1;
        transitions_evaluated += 1;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let reached_goal = env.terminal() == Terminal::Terminated(TerminalReason::GoalReached);
    Ok(BaselineMetrics {
        kind: BaselineKind::RandomSeeded,
        is_privileged_full_info: BaselineKind::RandomSeeded.is_privileged_full_info(),
        seed: Some(seed),
        reached_goal,
        no_path: false,
        ticks_executed,
        transitions_evaluated,
        total_reward: env.total_reward(),
        collisions: env.collisions(),
        path_len: reached_goal.then_some(ticks_executed),
        trace_hash: hasher.finish(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BfsSearchResult {
    Found {
        actions: Vec<GridAction>,
        transitions_evaluated: u64,
    },
    NoPath {
        transitions_evaluated: u64,
    },
    BudgetExhausted {
        transitions_evaluated: u64,
    },
}

fn bfs_shortest_path(
    map: &GridMap,
    avoid_terminal_hazards: bool,
    max_transitions: u64,
) -> BfsSearchResult {
    let start = map.start();
    let goal = map.goal();

    if start == goal {
        if max_transitions == 0 {
            return BfsSearchResult::BudgetExhausted {
                transitions_evaluated: 0,
            };
        }
        return BfsSearchResult::Found {
            actions: vec![GridAction::Stay],
            transitions_evaluated: 1,
        };
    }

    let mut queue = VecDeque::new();
    let mut visited = BTreeSet::new();
    let mut parent: BTreeMap<(usize, usize), ((usize, usize), GridAction)> = BTreeMap::new();
    let mut transitions_evaluated = 0_u64;

    queue.push_back(start);
    visited.insert((start.x, start.y));

    while let Some(cur) = queue.pop_front() {
        if cur == goal {
            let mut actions = Vec::new();
            let mut node = (goal.x, goal.y);
            while node != (start.x, start.y) {
                let (prev, act) = parent[&node];
                actions.push(act);
                node = prev;
            }
            actions.reverse();
            return BfsSearchResult::Found {
                actions,
                transitions_evaluated,
            };
        }

        for act in [
            GridAction::Up,
            GridAction::Down,
            GridAction::Left,
            GridAction::Right,
        ] {
            if transitions_evaluated >= max_transitions {
                return BfsSearchResult::BudgetExhausted {
                    transitions_evaluated,
                };
            }
            transitions_evaluated += 1;

            let (dx, dy) = act.delta();
            let nx = cur.x as isize + dx;
            let ny = cur.y as isize + dy;
            if nx < 0 || ny < 0 {
                continue;
            }
            let next = GridPos::new(nx as usize, ny as usize);
            if !map.is_passable(next) {
                continue;
            }
            if avoid_terminal_hazards && map.cell(next) == Some(CellKind::Hazard) {
                continue;
            }
            if visited.insert((next.x, next.y)) {
                parent.insert((next.x, next.y), ((cur.x, cur.y), act));
                queue.push_back(next);
            }
        }
    }

    BfsSearchResult::NoPath {
        transitions_evaluated,
    }
}

/// Ejecuta el baseline BFS privilegiado (información completa de `GridMap`) en [`GridWorldEnv`].
///
/// Si no existe camino alcanzable hasta `Goal`, devuelve `no_path: true` y `reached_goal: false`.
pub fn run_bfs_baseline(
    map: &GridMap,
    mut config: GridWorldConfig,
    budget: BaselineBudget,
) -> Result<BaselineMetrics, DomainError> {
    config.map = map.clone();
    config.validate()?;

    let bfs_seed = MasterSeed::new(0);
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(bfs_seed));

    let (actions, mut transitions_evaluated) =
        match bfs_shortest_path(map, config.hazard_is_terminal, budget.max_transitions) {
            BfsSearchResult::Found {
                actions,
                transitions_evaluated,
            } => (actions, transitions_evaluated),
            BfsSearchResult::NoPath {
                transitions_evaluated,
            } => {
                return Ok(BaselineMetrics {
                    kind: BaselineKind::PrivilegedBfs,
                    is_privileged_full_info: BaselineKind::PrivilegedBfs.is_privileged_full_info(),
                    seed: None,
                    reached_goal: false,
                    no_path: true,
                    ticks_executed: 0,
                    transitions_evaluated,
                    total_reward: 0.0,
                    collisions: 0,
                    path_len: None,
                    trace_hash: hasher.finish(),
                });
            }
            BfsSearchResult::BudgetExhausted {
                transitions_evaluated,
            } => {
                return Ok(BaselineMetrics {
                    kind: BaselineKind::PrivilegedBfs,
                    is_privileged_full_info: BaselineKind::PrivilegedBfs.is_privileged_full_info(),
                    seed: None,
                    reached_goal: false,
                    no_path: false,
                    ticks_executed: 0,
                    transitions_evaluated,
                    total_reward: 0.0,
                    collisions: 0,
                    path_len: None,
                    trace_hash: hasher.finish(),
                });
            }
        };

    let planned_path_len = actions.len() as u64;
    let mut env = GridWorldEnv::new(config)?;
    let mut ticks_executed = 0_u64;

    for act in actions {
        if ticks_executed >= budget.max_ticks || transitions_evaluated >= budget.max_transitions {
            break;
        }
        let obs = observe_env(&env)?;
        let action_idx = act as usize;
        let action = Action::Discrete(DiscreteAction::new(action_idx, GRID_ACTION_CARDINALITY)?);
        let outcome = env.apply(&action).map_err(map_env_error)?;

        hash_step(
            &mut hasher,
            &env,
            &obs,
            action_idx,
            outcome.observation(),
            outcome.terminal(),
        );

        ticks_executed += 1;
        transitions_evaluated += 1;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    let reached_goal = env.terminal() == Terminal::Terminated(TerminalReason::GoalReached);
    Ok(BaselineMetrics {
        kind: BaselineKind::PrivilegedBfs,
        is_privileged_full_info: BaselineKind::PrivilegedBfs.is_privileged_full_info(),
        seed: None,
        reached_goal,
        no_path: false,
        ticks_executed,
        transitions_evaluated,
        total_reward: env.total_reward(),
        collisions: env.collisions(),
        path_len: Some(planned_path_len),
        trace_hash: hasher.finish(),
    })
}
