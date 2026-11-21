//! Población de walkers en estructura SoA y transición batched para FMC (`std`-only).
//!
//! Este módulo implementa [`WalkerPopulation`] para mantener `N` trayectorias simuladas
//! en arrays paralelos (Structure of Arrays):
//! - `states`: estado actual de cada walker.
//! - `first_actions`: primera acción aplicada desde la raíz de planificación.
//! - `cumulative_rewards`: recompensa acumulada por walker.
//! - `terminals`: estado de terminación ([`Terminal::Ongoing`] o [`Terminal::Terminated`]).
//! - `rngs`: [`DeterministicRng`] aislado por walker derivado mediante [`StreamKey::walker`].
//! - `ancestry`: linaje opcional ([`WalkerAncestry`]) para trazabilidad de clonación/resampling.
//!
//! # Invariantes de seguridad y determinismo
//! - **Terminal no simula ilegalmente:** todo walker en [`Terminal::Terminated`] se omite
//!   en [`WalkerPopulation::step_population`] sin invocar [`GenerativeModel::step`] ni
//!   consumir presupuesto en [`BudgetTracker`].
//! - **Atomicidad ante error:** si [`GenerativeModel::step`] falla o la recompensa no es
//!   finita, ningún estado previo de la población es mutado.
//! - **Clonación aislada:** [`WalkerPopulation::clone_walker`] copia los valores de estado
//!   y deriva un nuevo [`DeterministicRng`] independiente para el destino, garantizando que
//!   origen y destino diverjan sin compartir mutabilidad ni secuencia RNG.

use std::error::Error;
use std::fmt;

use crate::action::Action;
use crate::determinism::{MasterSeed, StreamKey};
use crate::domain::{State, Terminal, Transition};
use crate::error::ModelError;
use crate::model::{DeterministicRng, GenerativeModel};
use crate::planner::BudgetTracker;
use crate::units::StepDt;

/// Identificador tipado de un walker dentro de una [`WalkerPopulation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WalkerId(usize);

impl WalkerId {
    /// Crea un identificador de walker a partir de su índice `0..N`.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Devuelve el índice subyacente del walker.
    pub const fn index(self) -> usize {
        self.0
    }

    /// Devuelve el índice convertido a `u64` para derivación de [`StreamKey`].
    pub const fn as_u64(self) -> u64 {
        self.0 as u64
    }
}

impl From<usize> for WalkerId {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl fmt::Display for WalkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "walker({})", self.0)
    }
}

/// Metadatos opcionales de linaje para un walker tras operaciones de clonación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WalkerAncestry {
    origin_walker: WalkerId,
    parent_walker: Option<WalkerId>,
    clone_count: u64,
}

impl WalkerAncestry {
    /// Crea el linaje inicial de un walker raíz.
    pub const fn root(walker: WalkerId) -> Self {
        Self {
            origin_walker: walker,
            parent_walker: None,
            clone_count: 0,
        }
    }

    /// Devuelve el walker raíz original del que desciende esta trayectoria.
    pub const fn origin_walker(self) -> WalkerId {
        self.origin_walker
    }

    /// Devuelve el walker padre inmediato si este walker fue producido por clonación.
    pub const fn parent_walker(self) -> Option<WalkerId> {
        self.parent_walker
    }

    /// Devuelve cuántas veces ha sido clonado este walker hacia otros destinos.
    pub const fn clone_count(self) -> u64 {
        self.clone_count
    }
}

/// Resumen de ejecución de un paso batched sobre [`WalkerPopulation`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkerStepReport {
    stepped: usize,
    skipped_terminal: usize,
    skipped_budget: usize,
    newly_terminated: usize,
    budget_exhausted: bool,
}

impl WalkerStepReport {
    /// Número de walkers no-terminales que ejecutaron una transición en este paso.
    pub const fn stepped(self) -> usize {
        self.stepped
    }

    /// Número de walkers que ya eran terminales y fueron omitidos sin simular ni gastar presupuesto.
    pub const fn skipped_terminal(self) -> usize {
        self.skipped_terminal
    }

    /// Número de walkers activos omitidos por agotamiento de [`BudgetTracker`].
    pub const fn skipped_budget(self) -> usize {
        self.skipped_budget
    }

    /// Número de walkers que pasaron de [`Terminal::Ongoing`] a [`Terminal::Terminated`] en este paso.
    pub const fn newly_terminated(self) -> usize {
        self.newly_terminated
    }

    /// Indica si el [`BudgetTracker`] se agotó durante el paso batched.
    pub const fn budget_exhausted(self) -> bool {
        self.budget_exhausted
    }
}

/// Errores explícitos de construcción, indexación, clonación o paso de [`WalkerPopulation`].
#[derive(Debug, Clone, PartialEq)]
pub enum PopulationError {
    /// Se intentó crear una población sin walkers (`N == 0`).
    EmptyPopulation,
    /// El número de acciones provistas no coincide con el tamaño de la población.
    ActionCountMismatch {
        /// Tamaño esperado de la población.
        expected: usize,
        /// Número de acciones recibido.
        actual: usize,
    },
    /// El índice de walker está fuera del rango `0..len`.
    WalkerOutOfBounds {
        /// Identificador inválido.
        walker: WalkerId,
        /// Tamaño actual de la población.
        len: usize,
    },
    /// Se intentó clonar un walker sobre sí mismo (`src == dst`).
    SelfClone {
        /// Identificador repetido en origen y destino.
        walker: WalkerId,
    },
    /// Falló la simulación de [`GenerativeModel::step`] para un walker concreto.
    ModelStepFailed {
        /// Walker que originó el error.
        walker: WalkerId,
        /// Error devuelto por el modelo generativo.
        source: ModelError,
    },
    /// La función de recompensa devolvió un valor no finito (`NaN` o infinito).
    NonFiniteReward {
        /// Walker cuya recompensa resultó inválida.
        walker: WalkerId,
        /// Valor no finito recibido.
        reward: f64,
    },
}

impl fmt::Display for PopulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPopulation => write!(f, "walker population cannot be empty"),
            Self::ActionCountMismatch { expected, actual } => write!(
                f,
                "action count mismatch: expected {expected} actions, got {actual}"
            ),
            Self::WalkerOutOfBounds { walker, len } => {
                write!(f, "{walker} out of bounds for population of length {len}")
            }
            Self::SelfClone { walker } => {
                write!(f, "cannot clone {walker} onto itself")
            }
            Self::ModelStepFailed { walker, source } => {
                write!(f, "model step failed for {walker}: {source}")
            }
            Self::NonFiniteReward { walker, reward } => {
                write!(f, "non-finite reward ({reward}) for {walker}")
            }
        }
    }
}

impl Error for PopulationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ModelStepFailed { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// One simulated step of a walker's trajectory since the current root: the action taken, the
/// state it produced and the reward it earned. Enables horizon-receding warm starts.
#[derive(Debug, Clone, PartialEq)]
struct PrefixStep {
    action: Action,
    state: State,
    reward: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct StagedWalkerUpdate {
    index: usize,
    prefix: PrefixStep,
    next_state: State,
    next_terminal: Terminal,
    reward: f64,
    first_action: Option<Action>,
    next_rng: DeterministicRng,
}

/// Población de `N` walkers almacenada en arrays paralelos (SoA) para FMC.
#[derive(Debug, Clone, PartialEq)]
pub struct WalkerPopulation {
    states: Vec<State>,
    first_actions: Vec<Option<Action>>,
    cumulative_rewards: Vec<f64>,
    terminals: Vec<Terminal>,
    rngs: Vec<DeterministicRng>,
    ancestry: Option<Vec<WalkerAncestry>>,
    /// Per-walker prefix buffer `[(a_0, s_1, r_0), (a_1, s_2, r_1), ...]` since the current root.
    /// Index 0 yields `step1_state` / first-step reward; index 1 yields the second action.
    prefixes: Vec<Vec<PrefixStep>>,
    /// Reusable staging buffer for `step_population`; always empty between calls.
    staging: Vec<StagedWalkerUpdate>,
}

impl WalkerPopulation {
    /// Crea una población uniforme de `num_walkers` walkers a partir de `initial_state`.
    ///
    /// Cada walker `w` recibe un [`DeterministicRng`] independiente derivado de `master_seed`
    /// con [`StreamKey::walker(episode, agent, w as u64)`].
    pub fn new(
        initial_state: State,
        num_walkers: usize,
        master_seed: MasterSeed,
        episode: u64,
        agent: u64,
        track_ancestry: bool,
    ) -> Result<Self, PopulationError> {
        if num_walkers == 0 {
            return Err(PopulationError::EmptyPopulation);
        }
        Self::from_states(
            vec![initial_state; num_walkers],
            master_seed,
            episode,
            agent,
            track_ancestry,
        )
    }

    /// Crea una población a partir de un vector explícito de estados iniciales.
    pub fn from_states(
        states: Vec<State>,
        master_seed: MasterSeed,
        episode: u64,
        agent: u64,
        track_ancestry: bool,
    ) -> Result<Self, PopulationError> {
        let num_walkers = states.len();
        if num_walkers == 0 {
            return Err(PopulationError::EmptyPopulation);
        }

        let first_actions = vec![None; num_walkers];
        let cumulative_rewards = vec![0.0; num_walkers];
        let terminals = vec![Terminal::Ongoing; num_walkers];
        let rngs = (0..num_walkers)
            .map(|w| master_seed.derive(StreamKey::walker(episode, agent, w as u64)))
            .collect();
        let ancestry = if track_ancestry {
            Some(
                (0..num_walkers)
                    .map(|w| WalkerAncestry::root(WalkerId::new(w)))
                    .collect(),
            )
        } else {
            None
        };

        Ok(Self {
            states,
            first_actions,
            cumulative_rewards,
            terminals,
            rngs,
            ancestry,
            prefixes: vec![Vec::new(); num_walkers],
            staging: Vec::with_capacity(num_walkers),
        })
    }

    /// State reached by `id` after its first transition from the root (`s_1`), if it has stepped.
    pub fn step1_state(&self, id: WalkerId) -> Result<Option<&State>, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.prefixes[id.index()].first().map(|p| &p.state))
    }

    /// Second action taken by `id` (`a_1`), if it has stepped at least twice.
    pub fn second_action(&self, id: WalkerId) -> Result<Option<&Action>, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.prefixes[id.index()].get(1).map(|p| &p.action))
    }

    /// Reward earned by `id` in its first transition from the root, if it has stepped.
    pub fn first_step_reward(&self, id: WalkerId) -> Result<Option<f64>, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.prefixes[id.index()].first().map(|p| p.reward))
    }

    /// Number of simulated steps recorded in the prefix buffer of `id`.
    pub fn prefix_len(&self, id: WalkerId) -> Result<usize, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.prefixes[id.index()].len())
    }

    /// Shifts the root of every walker one step forward (horizon-receding warm start).
    ///
    /// For each walker with at least two recorded steps: drops the first step, promotes the
    /// former second action to `first_action` and subtracts the dropped step's reward from the
    /// cumulative reward (undiscounted sum, so this is exact up to f64 rounding). Walkers with
    /// fewer than two recorded steps are left untouched; callers should only invoke this after
    /// cloning every such walker from a shiftable one.
    pub fn shift_root_one_step(&mut self) {
        for i in 0..self.len() {
            if self.prefixes[i].len() < 2 {
                continue;
            }
            let dropped = self.prefixes[i].remove(0);
            self.cumulative_rewards[i] -= dropped.reward;
            self.first_actions[i] = Some(self.prefixes[i][0].action.clone());
        }
    }

    /// Devuelve el número total de walkers `N` en la población.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Devuelve `true` si la población no contiene walkers.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Devuelve la vista SoA de todos los estados actuales.
    pub fn states(&self) -> &[State] {
        &self.states
    }

    /// Devuelve el estado actual del walker `id`.
    pub fn state(&self, id: WalkerId) -> Result<&State, PopulationError> {
        self.check_bounds(id)?;
        Ok(&self.states[id.index()])
    }

    /// Devuelve la vista SoA de las primeras acciones aplicadas desde la raíz.
    pub fn first_actions(&self) -> &[Option<Action>] {
        &self.first_actions
    }

    /// Devuelve la primera acción aplicada desde la raíz para el walker `id`.
    pub fn first_action(&self, id: WalkerId) -> Result<Option<&Action>, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.first_actions[id.index()].as_ref())
    }

    /// Devuelve la vista SoA de las recompensas acumuladas.
    pub fn cumulative_rewards(&self) -> &[f64] {
        &self.cumulative_rewards
    }

    /// Devuelve la recompensa acumulada del walker `id`.
    pub fn cumulative_reward(&self, id: WalkerId) -> Result<f64, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.cumulative_rewards[id.index()])
    }

    /// Devuelve la vista SoA de los estados de terminación.
    pub fn terminals(&self) -> &[Terminal] {
        &self.terminals
    }

    /// Devuelve el estado de terminación del walker `id`.
    pub fn terminal(&self, id: WalkerId) -> Result<Terminal, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.terminals[id.index()])
    }

    /// Devuelve la vista SoA de los generadores RNG de la población.
    pub fn rngs(&self) -> &[DeterministicRng] {
        &self.rngs
    }

    /// Devuelve el [`DeterministicRng`] del walker `id`.
    pub fn rng(&self, id: WalkerId) -> Result<&DeterministicRng, PopulationError> {
        self.check_bounds(id)?;
        Ok(&self.rngs[id.index()])
    }

    /// Devuelve la vista SoA de linaje si el seguimiento de `ancestry` está habilitado.
    pub fn ancestry(&self) -> Option<&[WalkerAncestry]> {
        self.ancestry.as_deref()
    }

    /// Devuelve el linaje del walker `id`, o `None` si `ancestry` está deshabilitado.
    pub fn walker_ancestry(&self, id: WalkerId) -> Result<Option<WalkerAncestry>, PopulationError> {
        self.check_bounds(id)?;
        Ok(self.ancestry.as_ref().map(|a| a[id.index()]))
    }

    /// Devuelve el número de walkers actualmente activos ([`Terminal::Ongoing`]).
    pub fn active_count(&self) -> usize {
        self.terminals.iter().filter(|t| t.is_ongoing()).count()
    }

    /// Devuelve el número de walkers actualmente terminales ([`Terminal::Terminated`]).
    pub fn terminal_count(&self) -> usize {
        self.terminals.iter().filter(|t| t.is_terminal()).count()
    }

    /// Devuelve `true` si todos los walkers de la población han alcanzado un estado terminal.
    pub fn all_terminal(&self) -> bool {
        self.terminals.iter().all(|t| t.is_terminal())
    }

    /// Ejecuta un paso batched sobre todos los walkers no-terminales de la población.
    ///
    /// - Si `actions.len() != self.len()`, devuelve [`PopulationError::ActionCountMismatch`] sin
    ///   mutar la población.
    /// - Todo walker en [`Terminal::Terminated`] se omite sin llamar a `model.step` ni
    ///   consumir presupuesto de `budget`.
    /// - Cada walker en [`Terminal::Ongoing`] consume 1 transición en `budget` antes de
    ///   simular; si el presupuesto está agotado, los walkers restantes se omiten registrando
    ///   `skipped_budget` y `budget_exhausted = true`.
    /// - Si `model.step` falla o `reward_fn` devuelve un valor no finito, la población no
    ///   sufre ninguna mutación parcial.
    pub fn step_population<M, F>(
        &mut self,
        model: &M,
        actions: &[Action],
        dt: StepDt,
        budget: &mut BudgetTracker,
        mut reward_fn: F,
    ) -> Result<WalkerStepReport, PopulationError>
    where
        M: GenerativeModel + ?Sized,
        F: FnMut(&State, &Action, &Transition) -> f64,
    {
        if actions.len() != self.len() {
            return Err(PopulationError::ActionCountMismatch {
                expected: self.len(),
                actual: actions.len(),
            });
        }

        // Take the reusable staging buffer (always empty between calls); it is put back on every
        // exit path, so an error leaves the population observably untouched.
        let mut staged = std::mem::take(&mut self.staging);
        staged.clear();
        let mut skipped_terminal = 0usize;
        let mut skipped_budget = 0usize;
        let mut newly_terminated = 0usize;
        let mut budget_exhausted = false;
        let mut consumed_count = 0usize;

        for (index, action) in actions.iter().enumerate() {
            if self.terminals[index].is_terminal() {
                skipped_terminal += 1;
                continue;
            }

            if budget.is_exhausted() {
                budget_exhausted = true;
                skipped_budget += 1;
                continue;
            }

            if consumed_count >= budget.remaining_transitions() {
                budget_exhausted = true;
                skipped_budget += 1;
                continue;
            }

            let walker_id = WalkerId::new(index);
            let mut rng_copy = self.rngs[index].clone();
            let transition = match model.step(&self.states[index], action, dt, &mut rng_copy) {
                Ok(transition) => transition,
                Err(source) => {
                    staged.clear();
                    self.staging = staged;
                    return Err(PopulationError::ModelStepFailed {
                        walker: walker_id,
                        source,
                    });
                }
            };

            let reward = reward_fn(&self.states[index], action, &transition);
            if !reward.is_finite() {
                staged.clear();
                self.staging = staged;
                return Err(PopulationError::NonFiniteReward {
                    walker: walker_id,
                    reward,
                });
            }

            consumed_count += 1;
            let next_terminal = transition.terminal();
            if next_terminal.is_terminal() {
                newly_terminated += 1;
            }

            let first_action = if self.first_actions[index].is_none() {
                Some(action.clone())
            } else {
                None
            };

            // One State copy remains (population state + prefix entry both need an owner);
            // the transition's own State is moved instead of cloned a second time.
            let next_state = transition.into_next_state();
            staged.push(StagedWalkerUpdate {
                index,
                prefix: PrefixStep {
                    action: action.clone(),
                    state: next_state.clone(),
                    reward,
                },
                next_state,
                next_terminal,
                reward,
                first_action,
                next_rng: rng_copy,
            });
        }

        // Contabilizar en BudgetTracker únicamente tras validar todas las transiciones staged.
        for _ in 0..consumed_count {
            let consumed = budget.try_consume_transition();
            debug_assert!(
                consumed,
                "remaining_transitions was verified before staging"
            );
        }

        let stepped = staged.len();
        for update in staged.drain(..) {
            let idx = update.index;
            self.states[idx] = update.next_state;
            self.terminals[idx] = update.next_terminal;
            self.cumulative_rewards[idx] += update.reward;
            if let Some(first) = update.first_action {
                self.first_actions[idx] = Some(first);
            }
            self.rngs[idx] = update.next_rng;
            self.prefixes[idx].push(update.prefix);
        }
        self.staging = staged;

        Ok(WalkerStepReport {
            stepped,
            skipped_terminal,
            skipped_budget,
            newly_terminated,
            budget_exhausted,
        })
    }

    /// Clona de forma aislada el estado lógico de `src` hacia `dst` y deriva un nuevo RNG.
    ///
    /// - Copia `state`, `first_action`, `cumulative_reward` y `terminal`.
    /// - Si `ancestry` está habilitada, incrementa `clone_count` en `src` y registra en `dst`
    ///   el mismo `origin_walker` con `parent_walker = Some(src)` y `clone_count = 0`.
    /// - Deriva un nuevo [`DeterministicRng`] independiente para `dst` usando `seed_salt`
    ///   sin mutar el RNG de `src`, garantizando divergencia independiente.
    pub fn clone_walker(
        &mut self,
        src: WalkerId,
        dst: WalkerId,
        seed_salt: u64,
    ) -> Result<(), PopulationError> {
        self.check_bounds(src)?;
        self.check_bounds(dst)?;
        if src == dst {
            return Err(PopulationError::SelfClone { walker: src });
        }

        let src_idx = src.index();
        let dst_idx = dst.index();

        self.states[dst_idx] = self.states[src_idx].clone();
        self.first_actions[dst_idx] = self.first_actions[src_idx].clone();
        self.cumulative_rewards[dst_idx] = self.cumulative_rewards[src_idx];
        self.terminals[dst_idx] = self.terminals[src_idx];
        self.prefixes[dst_idx] = self.prefixes[src_idx].clone();

        // Deriva un stream RNG completamente independiente para dst sin mutar src.
        self.rngs[dst_idx] =
            self.rngs[src_idx].derive(StreamKey::walker(seed_salt, src.as_u64(), dst.as_u64()));

        if let Some(ancestry) = self.ancestry.as_mut() {
            let origin_walker = ancestry[src_idx].origin_walker;
            ancestry[src_idx].clone_count = ancestry[src_idx].clone_count.saturating_add(1);
            ancestry[dst_idx] = WalkerAncestry {
                origin_walker,
                parent_walker: Some(src),
                clone_count: 0,
            };
        }

        Ok(())
    }

    /// Applies several `(src, dst)` clones synchronously against the pre-clone population.
    ///
    /// Every destination receives the state its source had *before* any write of this call,
    /// so chains (`A -> B`, `B -> C`) and cycles (`A -> B`, `B -> A`) are well defined and
    /// independent of the order of `clones`. Per-pair semantics (copied fields, RNG derivation
    /// from `seed_salt`, ancestry) match [`Self::clone_walker`]. Ancestry `clone_count`
    /// increments of sources are applied first and any walker that is also a destination is
    /// then reset by the overwrite.
    ///
    /// All pairs are validated before anything is mutated; on error the population is
    /// unchanged. If a destination appears more than once, the last pair wins.
    pub fn clone_walkers_synchronously(
        &mut self,
        clones: &[(WalkerId, WalkerId)],
        seed_salt: u64,
    ) -> Result<(), PopulationError> {
        for &(src, dst) in clones {
            self.check_bounds(src)?;
            self.check_bounds(dst)?;
            if src == dst {
                return Err(PopulationError::SelfClone { walker: src });
            }
        }

        // Read phase: snapshot every clone payload from the untouched population.
        let staged: Vec<_> = clones
            .iter()
            .map(|&(src, dst)| {
                let s = src.index();
                let rng =
                    self.rngs[s].derive(StreamKey::walker(seed_salt, src.as_u64(), dst.as_u64()));
                (
                    src,
                    dst,
                    self.states[s].clone(),
                    self.first_actions[s].clone(),
                    self.cumulative_rewards[s],
                    self.terminals[s],
                    self.prefixes[s].clone(),
                    rng,
                    self.ancestry.as_ref().map(|a| a[s].origin_walker),
                )
            })
            .collect();

        // Write phase.
        if let Some(ancestry) = self.ancestry.as_mut() {
            for &(src, _) in clones {
                let a = &mut ancestry[src.index()];
                a.clone_count = a.clone_count.saturating_add(1);
            }
        }
        for (src, dst, state, first_action, reward, terminal, prefix, rng, origin) in staged {
            let d = dst.index();
            self.states[d] = state;
            self.first_actions[d] = first_action;
            self.cumulative_rewards[d] = reward;
            self.terminals[d] = terminal;
            self.prefixes[d] = prefix;
            self.rngs[d] = rng;
            if let (Some(ancestry), Some(origin_walker)) = (self.ancestry.as_mut(), origin) {
                ancestry[d] = WalkerAncestry {
                    origin_walker,
                    parent_walker: Some(src),
                    clone_count: 0,
                };
            }
        }

        Ok(())
    }

    fn check_bounds(&self, id: WalkerId) -> Result<(), PopulationError> {
        if id.index() >= self.len() {
            return Err(PopulationError::WalkerOutOfBounds {
                walker: id,
                len: self.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::action::{ActionSpace, DiscreteAction};
    use crate::domain::{Observation, TerminalReason};
    use crate::model::RngSource;
    use crate::planner::PlanBudget;
    use crate::units::{SimDuration, Tick};

    struct StochasticStepModel {
        space: ActionSpace,
        goal_threshold: f64,
        fail_on_action_2: bool,
        step_calls: AtomicUsize,
    }

    impl StochasticStepModel {
        fn new(goal_threshold: f64) -> Self {
            Self {
                space: ActionSpace::discrete(3).unwrap(),
                goal_threshold,
                fail_on_action_2: false,
                step_calls: AtomicUsize::new(0),
            }
        }

        fn with_failure_on_action_2(mut self) -> Self {
            self.fail_on_action_2 = true;
            self
        }

        fn calls(&self) -> usize {
            self.step_calls.load(Ordering::Relaxed)
        }
    }

    impl GenerativeModel for StochasticStepModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            self.step_calls.fetch_add(1, Ordering::Relaxed);
            self.space
                .validate(action)
                .map_err(ModelError::InvalidAction)?;

            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            if self.fail_on_action_2 && idx == 2 {
                return Err(ModelError::InvalidState {
                    reason: "injected model error for action 2",
                });
            }

            let noise = 0.25 * rng.next_f64();
            let delta = match idx {
                0 => 0.1 + noise,
                1 => 1.0 + noise,
                _ => -0.5 + noise,
            };
            let next_val = state.values()[0] + delta;
            let next_state = state
                .advance(dt, vec![next_val])
                .map_err(ModelError::TransitionFailed)?;
            let obs = Observation::new(
                next_state.tick(),
                next_state.elapsed(),
                next_state.values().to_vec(),
            )
            .map_err(ModelError::TransitionFailed)?;
            let terminal = if next_val >= self.goal_threshold {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else {
                Terminal::Ongoing
            };

            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    #[test]
    fn ac1_population_step_matches_scalar_walker_by_walker_fixture() {
        let model = StochasticStepModel::new(2.5);
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let seed = MasterSeed::new(0x1234_5678_9ABC_DEF0);
        let n = 5;

        let mut pop = WalkerPopulation::new(root.clone(), n, seed, 3, 1, false).unwrap();
        assert!(pop.ancestry().is_none());

        let act0 = Action::discrete_in_space(0, model.action_space()).unwrap();
        let act1 = Action::discrete_in_space(1, model.action_space()).unwrap();
        let act2 = Action::discrete_in_space(2, model.action_space()).unwrap();

        let step_actions_1 = vec![
            act0.clone(),
            act1.clone(),
            act0.clone(),
            act1.clone(),
            act2.clone(),
        ];
        let step_actions_2 = vec![
            act1.clone(),
            act1.clone(),
            act2.clone(),
            act0.clone(),
            act1.clone(),
        ];

        // Fixture escalar equivalente walker a walker.
        let mut scalar_states = vec![root; n];
        let mut scalar_first_actions: Vec<Option<Action>> = vec![None; n];
        let mut scalar_rewards = vec![0.0; n];
        let mut scalar_terminals = vec![Terminal::Ongoing; n];
        let mut scalar_rngs: Vec<DeterministicRng> = (0..n)
            .map(|w| seed.derive(StreamKey::walker(3, 1, w as u64)))
            .collect();

        let mut budget = PlanBudget::with_transitions(100).tracker();

        for actions in [&step_actions_1, &step_actions_2] {
            let report = pop
                .step_population(&model, actions, dt, &mut budget, |_s, _a, t| {
                    t.next_state().values()[0] * 0.5
                })
                .unwrap();

            let mut expected_stepped = 0usize;
            for i in 0..n {
                if scalar_terminals[i].is_terminal() {
                    continue;
                }
                expected_stepped += 1;
                let tr = model
                    .step(&scalar_states[i], &actions[i], dt, &mut scalar_rngs[i])
                    .unwrap();
                scalar_rewards[i] += tr.next_state().values()[0] * 0.5;
                if scalar_first_actions[i].is_none() {
                    scalar_first_actions[i] = Some(actions[i].clone());
                }
                scalar_states[i] = tr.next_state().clone();
                scalar_terminals[i] = tr.terminal();
            }

            assert_eq!(report.stepped(), expected_stepped);
            assert_eq!(pop.states(), scalar_states.as_slice());
            assert_eq!(pop.first_actions(), scalar_first_actions.as_slice());
            assert_eq!(pop.cumulative_rewards(), scalar_rewards.as_slice());
            assert_eq!(pop.terminals(), scalar_terminals.as_slice());
            assert_eq!(pop.rngs(), scalar_rngs.as_slice());
        }
    }

    fn marked_population(n: usize) -> WalkerPopulation {
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let mut pop = WalkerPopulation::new(root, n, MasterSeed::new(42), 0, 0, true).unwrap();
        for i in 0..n {
            pop.states[i] =
                State::new(Tick::ZERO, SimDuration::ZERO, vec![i as f64 + 1.0]).unwrap();
            pop.cumulative_rewards[i] = 10.0 * (i as f64 + 1.0);
        }
        pop
    }

    #[test]
    fn gog118_ac1_chain_copies_pre_resampling_state_of_middle_walker() {
        let mut pop = marked_population(3);
        let (a, b, c) = (WalkerId::new(0), WalkerId::new(1), WalkerId::new(2));
        // A -> B and B -> C: C must receive B's original state, never A's.
        pop.clone_walkers_synchronously(&[(a, b), (b, c)], 7)
            .unwrap();
        assert_eq!(pop.state(b).unwrap().values(), &[1.0]);
        assert_eq!(pop.state(c).unwrap().values(), &[2.0]);
        assert_eq!(pop.cumulative_reward(c).unwrap(), 20.0);
        assert_eq!(
            pop.walker_ancestry(c).unwrap().unwrap().parent_walker(),
            Some(b)
        );
        assert_eq!(pop.walker_ancestry(c).unwrap().unwrap().origin_walker(), b);
    }

    #[test]
    fn gog118_ac2_cross_clone_swaps_pre_resampling_states() {
        let mut pop = marked_population(2);
        let (a, b) = (WalkerId::new(0), WalkerId::new(1));
        pop.clone_walkers_synchronously(&[(a, b), (b, a)], 7)
            .unwrap();
        assert_eq!(pop.state(a).unwrap().values(), &[2.0]);
        assert_eq!(pop.state(b).unwrap().values(), &[1.0]);
        assert_eq!(pop.cumulative_reward(a).unwrap(), 20.0);
        assert_eq!(pop.cumulative_reward(b).unwrap(), 10.0);
    }

    #[test]
    fn gog118_ac3_result_is_invariant_to_pair_order() {
        let w = |i| WalkerId::new(i);
        let pairs = [(w(0), w(1)), (w(1), w(2)), (w(2), w(3)), (w(3), w(0))];
        let mut reference = marked_population(4);
        reference.clone_walkers_synchronously(&pairs, 9).unwrap();

        // All 24 permutations of the pair list.
        let mut idx = [0usize, 1, 2, 3];
        let mut perms = Vec::new();
        fn heap(k: usize, a: &mut [usize; 4], out: &mut Vec<[usize; 4]>) {
            if k == 1 {
                out.push(*a);
                return;
            }
            for i in 0..k {
                heap(k - 1, a, out);
                let j = if k.is_multiple_of(2) { i } else { 0 };
                a.swap(j, k - 1);
            }
        }
        heap(4, &mut idx, &mut perms);
        assert_eq!(perms.len(), 24);
        for perm in perms {
            let shuffled: Vec<_> = perm.iter().map(|&i| pairs[i]).collect();
            let mut pop = marked_population(4);
            pop.clone_walkers_synchronously(&shuffled, 9).unwrap();
            assert_eq!(pop.states(), reference.states());
            assert_eq!(pop.cumulative_rewards(), reference.cumulative_rewards());
            assert_eq!(pop.rngs(), reference.rngs());
            assert_eq!(pop.ancestry(), reference.ancestry());
        }
    }

    #[test]
    fn gog118_resample_population_chain_uses_pre_resampling_state() {
        struct Scripted(Vec<u64>, usize);
        impl RngSource for Scripted {
            fn next_u64(&mut self) -> u64 {
                let v = self.0[self.1];
                self.1 += 1;
                v
            }
        }
        let mut pop = marked_population(3);
        // Walker 0 keeps (vr highest); walker 1 clones companion 0; walker 2 clones companion 1.
        let mut rng = Scripted(vec![0, 0, 0, 0, 1, 0], 0);
        let report =
            crate::fmc_scoring::resample_population(&mut pop, &[10.0, 5.0, 1.0], &mut rng, 3)
                .unwrap();
        assert_eq!(report.cloned_count(), 2);
        assert_eq!(pop.state(WalkerId::new(1)).unwrap().values(), &[1.0]);
        assert_eq!(pop.state(WalkerId::new(2)).unwrap().values(), &[2.0]);
    }

    #[test]
    fn gog118_conflict_free_matches_sequential_clone_walker() {
        let w = |i| WalkerId::new(i);
        let mut sync = marked_population(4);
        let mut seq = marked_population(4);
        sync.clone_walkers_synchronously(&[(w(0), w(1)), (w(0), w(2))], 5)
            .unwrap();
        seq.clone_walker(w(0), w(1), 5).unwrap();
        seq.clone_walker(w(0), w(2), 5).unwrap();
        assert_eq!(sync.states(), seq.states());
        assert_eq!(sync.rngs(), seq.rngs());
        assert_eq!(sync.ancestry(), seq.ancestry());
        assert!(matches!(
            sync.clone_walkers_synchronously(&[(w(0), w(1)), (w(2), w(2))], 5),
            Err(PopulationError::SelfClone { .. })
        ));
    }

    #[test]
    fn ac2_cloning_isolates_state_and_derives_independent_rng() {
        let model = StochasticStepModel::new(100.0);
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let seed = MasterSeed::new(42);
        let mut pop = WalkerPopulation::new(root, 3, seed, 0, 0, true).unwrap();

        let act0 = Action::discrete_in_space(0, model.action_space()).unwrap();
        let act1 = Action::discrete_in_space(1, model.action_space()).unwrap();
        let mut budget = PlanBudget::with_transitions(50).tracker();

        // Avanzamos un paso inicial para poblar state, first_action y cumulative_reward.
        pop.step_population(
            &model,
            &[act1.clone(), act0.clone(), act0.clone()],
            dt,
            &mut budget,
            |_s, _a, t| t.next_state().values()[0],
        )
        .unwrap();

        let w0 = WalkerId::new(0);
        let w1 = WalkerId::new(1);
        let w2 = WalkerId::new(2);
        let src_rng_before_clone = pop.rng(w0).unwrap().clone();

        // Clonamos w0 -> w1 y w0 -> w2 con el mismo seed_salt.
        pop.clone_walker(w0, w1, 7).unwrap();
        pop.clone_walker(w0, w2, 7).unwrap();

        // El RNG de origen no se muta al clonar y los destinos tienen RNGs distintos.
        assert_eq!(pop.rng(w0).unwrap(), &src_rng_before_clone);
        assert_ne!(pop.rng(w0).unwrap(), pop.rng(w1).unwrap());
        assert_ne!(pop.rng(w0).unwrap(), pop.rng(w2).unwrap());
        assert_ne!(pop.rng(w1).unwrap(), pop.rng(w2).unwrap());

        // Estado lógico copiado fielmente.
        assert_eq!(pop.state(w0).unwrap(), pop.state(w1).unwrap());
        assert_eq!(pop.first_action(w0).unwrap(), pop.first_action(w1).unwrap());
        assert_eq!(
            pop.cumulative_reward(w0).unwrap(),
            pop.cumulative_reward(w1).unwrap()
        );
        assert_eq!(pop.terminal(w0).unwrap(), pop.terminal(w1).unwrap());

        // Linaje actualizado correctamente.
        let anc0 = pop.walker_ancestry(w0).unwrap().unwrap();
        let anc1 = pop.walker_ancestry(w1).unwrap().unwrap();
        assert_eq!(anc0.origin_walker(), w0);
        assert_eq!(anc0.parent_walker(), None);
        assert_eq!(anc0.clone_count(), 2);
        assert_eq!(anc1.origin_walker(), w0);
        assert_eq!(anc1.parent_walker(), Some(w0));
        assert_eq!(anc1.clone_count(), 0);

        // Tras aplicar exactamente la misma acción, w0, w1 y w2 divergen por sus RNGs independientes.
        let same_actions = [act1.clone(), act1.clone(), act1];
        pop.step_population(&model, &same_actions, dt, &mut budget, |_s, _a, t| {
            t.next_state().values()[0]
        })
        .unwrap();

        assert_ne!(
            pop.state(w0).unwrap().values(),
            pop.state(w1).unwrap().values()
        );
        assert_ne!(
            pop.state(w1).unwrap().values(),
            pop.state(w2).unwrap().values()
        );

        // Errores explícitos de autoclonación y fuera de rango.
        assert!(matches!(
            pop.clone_walker(w0, w0, 1),
            Err(PopulationError::SelfClone { walker }) if walker == w0
        ));
        assert!(matches!(
            pop.clone_walker(w0, WalkerId::new(99), 1),
            Err(PopulationError::WalkerOutOfBounds { .. })
        ));
    }

    #[test]
    fn ac3_terminal_walkers_do_not_simulate_and_budget_is_accurately_tracked() {
        let model = StochasticStepModel::new(1.0);
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let seed = MasterSeed::new(99);
        let mut pop = WalkerPopulation::new(root, 3, seed, 0, 0, false).unwrap();

        let stay = Action::discrete_in_space(0, model.action_space()).unwrap();
        let reach_goal = Action::discrete_in_space(1, model.action_space()).unwrap();

        let mut budget = PlanBudget::with_transitions(10).tracker();

        // Paso 1: walker 0 alcanza terminal (>= 1.0), walkers 1 y 2 siguen Ongoing.
        let r1 = pop
            .step_population(
                &model,
                &[reach_goal.clone(), stay.clone(), stay.clone()],
                dt,
                &mut budget,
                |_s, _a, _t| 1.0,
            )
            .unwrap();
        assert_eq!(r1.stepped(), 3);
        assert_eq!(r1.skipped_terminal(), 0);
        assert_eq!(r1.newly_terminated(), 1);
        assert_eq!(budget.transitions_used(), 3);
        assert_eq!(model.calls(), 3);
        assert!(pop.terminal(WalkerId::new(0)).unwrap().is_terminal());

        let terminal_state_before = pop.state(WalkerId::new(0)).unwrap().clone();
        let terminal_rng_before = pop.rng(WalkerId::new(0)).unwrap().clone();

        // Paso 2: walker 0 ya es terminal -> no debe llamar a model.step ni consumir presupuesto.
        let r2 = pop
            .step_population(
                &model,
                &[reach_goal, stay.clone(), stay.clone()],
                dt,
                &mut budget,
                |_s, _a, _t| 1.0,
            )
            .unwrap();
        assert_eq!(r2.stepped(), 2);
        assert_eq!(r2.skipped_terminal(), 1);
        assert_eq!(r2.skipped_budget(), 0);
        assert!(!r2.budget_exhausted());
        assert_eq!(budget.transitions_used(), 5);
        assert_eq!(model.calls(), 5);
        assert_eq!(pop.state(WalkerId::new(0)).unwrap(), &terminal_state_before);
        assert_eq!(pop.rng(WalkerId::new(0)).unwrap(), &terminal_rng_before);
        assert_eq!(pop.cumulative_reward(WalkerId::new(0)).unwrap(), 1.0);

        // Paso 3: presupuesto con 1 transición restante para 2 walkers activos.
        let mut tight_budget = PlanBudget::with_transitions(1).tracker();
        let r3 = pop
            .step_population(
                &model,
                &[stay.clone(), stay.clone(), stay],
                dt,
                &mut tight_budget,
                |_s, _a, _t| 1.0,
            )
            .unwrap();
        assert_eq!(r3.stepped(), 1);
        assert_eq!(r3.skipped_terminal(), 1);
        assert_eq!(r3.skipped_budget(), 1);
        assert!(r3.budget_exhausted());
        assert_eq!(tight_budget.transitions_used(), 1);
    }

    #[test]
    fn step_population_does_not_mutate_previous_states_on_error() {
        let model = StochasticStepModel::new(10.0).with_failure_on_action_2();
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let seed = MasterSeed::new(77);
        let mut pop = WalkerPopulation::new(root, 3, seed, 0, 0, true).unwrap();
        let snapshot_before = pop.clone();

        let act0 = Action::discrete_in_space(0, model.action_space()).unwrap();
        let act2 = Action::discrete_in_space(2, model.action_space()).unwrap();
        let mut budget = PlanBudget::with_transitions(10).tracker();

        // Walker 0 tendría éxito, pero Walker 2 falla -> toda la población debe quedar intacta.
        let err = pop
            .step_population(
                &model,
                &[act0.clone(), act0.clone(), act2],
                dt,
                &mut budget,
                |_s, _a, _t| 1.0,
            )
            .expect_err("walker 2 failure must abort batch cleanly");
        assert!(matches!(
            err,
            PopulationError::ModelStepFailed { walker, .. } if walker == WalkerId::new(2)
        ));
        assert_eq!(pop, snapshot_before);
        assert_eq!(budget.transitions_used(), 0);

        // Recompensa no finita también aborta sin mutar estado previo ni presupuesto.
        let err_nan = pop
            .step_population(
                &model,
                &[act0.clone(), act0.clone(), act0],
                dt,
                &mut budget,
                |_s, _a, _t| f64::NAN,
            )
            .expect_err("non-finite reward must abort batch cleanly");
        assert!(matches!(err_nan, PopulationError::NonFiniteReward { .. }));
        assert_eq!(pop, snapshot_before);
        assert_eq!(budget.transitions_used(), 0);
    }

    /// Model that records the heap address of the `State` values it returns.
    struct PtrRecordingModel {
        space: ActionSpace,
        last_ptr: std::cell::Cell<usize>,
    }

    impl GenerativeModel for PtrRecordingModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            _action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            let next_state = state
                .advance(dt, vec![state.values()[0] + 1.0])
                .map_err(ModelError::TransitionFailed)?;
            self.last_ptr.set(next_state.values().as_ptr() as usize);
            let obs = Observation::new(
                next_state.tick(),
                next_state.elapsed(),
                next_state.values().to_vec(),
            )
            .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next_state, obs, Terminal::Ongoing)
                .map_err(ModelError::TransitionFailed)
        }
    }

    #[test]
    fn ac1_step_population_moves_transition_state_and_keeps_staging_reusable() {
        let model = PtrRecordingModel {
            space: ActionSpace::discrete(2).unwrap(),
            last_ptr: std::cell::Cell::new(0),
        };
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut pop = WalkerPopulation::new(root, 1, MasterSeed::new(5), 0, 0, false).unwrap();
        let action = Action::discrete_in_space(0, model.action_space()).unwrap();
        let mut budget = PlanBudget::with_transitions(10).tracker();

        for step in 1..=3u32 {
            pop.step_population(
                &model,
                std::slice::from_ref(&action),
                dt,
                &mut budget,
                |_s, _a, _t| 0.5,
            )
            .unwrap();
            // The population owns the very allocation the model produced (moved, not cloned).
            assert_eq!(
                pop.states()[0].values().as_ptr() as usize,
                model.last_ptr.get()
            );
            assert_eq!(pop.states()[0].values()[0], f64::from(step));
            assert!(pop.staging.is_empty());
        }
        assert!(pop.staging.capacity() >= 1);
        assert_eq!(pop.prefix_len(WalkerId::new(0)).unwrap(), 3);
    }

    #[test]
    fn ac1_staging_buffer_is_empty_and_population_intact_after_error() {
        let model = StochasticStepModel::new(10.0).with_failure_on_action_2();
        let root = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut pop = WalkerPopulation::new(root, 3, MasterSeed::new(77), 0, 0, true).unwrap();
        let before = pop.clone();
        let a0 = Action::discrete_in_space(0, model.action_space()).unwrap();
        let a2 = Action::discrete_in_space(2, model.action_space()).unwrap();
        let mut budget = PlanBudget::with_transitions(10).tracker();
        pop.step_population(
            &model,
            &[a0.clone(), a0, a2],
            dt,
            &mut budget,
            |_s, _a, _t| 1.0,
        )
        .expect_err("walker 2 must fail");
        assert!(pop.staging.is_empty());
        assert_eq!(pop, before);
    }
}
