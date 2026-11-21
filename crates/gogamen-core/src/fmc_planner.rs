//! Planificador Fractal Monte Carlo (FMC) recedente (`std`-only).
//!
//! Implementa el algoritmo de referencia FMC sobre cualquier [`GenerativeModel`]
//! encapsulado en [`PlanningContext`], sin acceso a [`crate::LiveEnvironment`] ni
//! [`crate::Actuator`]:
//! - [`FmcPlannerConfig`]: tamaño de población, horizonte `H`, paso `dt`,
//!   configuración de scoring [`FmcScoringConfig`] y semilla determinista [`MasterSeed`].
//! - [`FmcTraceSummary`]: resumen observable de la planificación (primera acción,
//!   transiciones consumidas, eventos de clonación, valor virtual medio, número de
//!   primeras acciones únicas / diversidad y estado [`PlanStatus`]).
//! - [`FmcPlanner`]: ejecuta el bucle recedente `walk -> score -> resample` hasta
//!   horizonte `H` o hasta agotar [`crate::BudgetTracker`], [`crate::CancelToken`] o alcanzar
//!   terminación en toda la población.

use std::error::Error;
use std::fmt;

use crate::action::{Action, ActionSpace};
use crate::determinism::{MasterSeed, StreamKey};
use crate::domain::{State, Terminal, TerminalReason, Transition};
use crate::error::ModelError;
use crate::fmc_scoring::{
    FmcScoringConfig, is_failure_terminal, relativize_viable, resample_population, score_population,
};
use crate::fmc_walkers::{PopulationError, WalkerPopulation};
use crate::model::{GenerativeModel, PlanningContext, RngSource};
use crate::planner::{PlanRequest, PlanResult, PlanStatus, Planner};
use crate::units::StepDt;

/// Errores de configuración del planificador FMC recedente.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FmcPlannerError {
    /// El tamaño de la población debe ser al menos `1`.
    InvalidPopulationSize(usize),
    /// El horizonte de planificación debe ser al menos `1`.
    InvalidHorizon(usize),
    /// Un parámetro de [`FmcActionSelectionPolicy`] no es finito o es negativo.
    InvalidActionSelection(&'static str),
}

impl fmt::Display for FmcPlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPopulationSize(n) => {
                write!(f, "invalid FMC population size ({n}): must be >= 1")
            }
            Self::InvalidHorizon(h) => {
                write!(f, "invalid FMC planning horizon ({h}): must be >= 1")
            }
            Self::InvalidActionSelection(param) => {
                write!(
                    f,
                    "invalid FMC action selection parameter `{param}`: must be finite and >= 0"
                )
            }
        }
    }
}

impl Error for FmcPlannerError {}

/// Política de selección de la acción raíz `a_0` entre los walkers viables (no en fallo terminal).
///
/// Todas las políticas son deterministas y nunca producen NaN. Las estadísticas se calculan solo
/// sobre walkers viables; el empate se resuelve por mayor soporte `N(a_0)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FmcActionSelectionPolicy {
    /// Comportamiento clásico: media simple `sum_reward / N(a_0)`. Es vulnerable al sesgo de
    /// superviviente único (una acción con `N = 1` puede derrotar a una robusta con `N` alto).
    #[default]
    EmpiricalMean,
    /// Cota inferior de confianza con shrinkage empírico hacia la media poblacional:
    ///
    /// `score = (N * mean_a + prior_weight * mean_pop) / (N + prior_weight) - lcb_c * std / sqrt(N)`
    ///
    /// - `mean_pop` y `std_pop` son la media y desviación estándar poblacionales de los retornos de
    ///   todos los walkers viables con primera acción.
    /// - `std` es la desviación estándar muestral (`n - 1`) de los retornos de los viables de `a_0`
    ///   si `N >= 2`; con `N < 2` usa `std_pop` como fallback, de modo que un único superviviente
    ///   recibe la penalización máxima `lcb_c * std_pop` y el shrinkage lo acerca a `mean_pop`.
    /// - `lcb_c` está expresado en unidades de retorno: debe ser suficientemente grande para que la
    ///   penalización supere el margen de media del superviviente único.
    LowerConfidenceBound {
        /// Multiplicador de la penalización de incertidumbre (finito, `>= 0`).
        lcb_c: f64,
        /// Peso del prior poblacional en pseudo-observaciones (finito, `>= 0`).
        prior_weight: f64,
    },
    /// Suma de `VR_i` (valor virtual del scoring) de los walkers viables descendientes de `a_0`.
    ///
    /// Si el vector de `VR` no está disponible en el punto de selección (p. ej. la población fue
    /// remuestreada tras el último scoring), usa la suma de retornos relativizados
    /// ([`relativize_viable`]) de dichos walkers como aproximación.
    VirtualValueMass,
}

impl FmcActionSelectionPolicy {
    fn validate(&self) -> Result<(), FmcPlannerError> {
        if let Self::LowerConfidenceBound {
            lcb_c,
            prior_weight,
        } = *self
        {
            if !lcb_c.is_finite() || lcb_c < 0.0 {
                return Err(FmcPlannerError::InvalidActionSelection("lcb_c"));
            }
            if !prior_weight.is_finite() || prior_weight < 0.0 {
                return Err(FmcPlannerError::InvalidActionSelection("prior_weight"));
            }
        }
        Ok(())
    }
}

/// Configuración explícita de [`FmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct FmcPlannerConfig {
    population_size: usize,
    horizon: usize,
    dt: StepDt,
    scoring: FmcScoringConfig,
    seed: MasterSeed,
    action_selection: FmcActionSelectionPolicy,
}

impl FmcPlannerConfig {
    /// Construye y valida una configuración de planificador FMC recedente.
    pub fn new(
        population_size: usize,
        horizon: usize,
        dt: StepDt,
        scoring: FmcScoringConfig,
        seed: MasterSeed,
    ) -> Result<Self, FmcPlannerError> {
        if population_size == 0 {
            return Err(FmcPlannerError::InvalidPopulationSize(population_size));
        }
        if horizon == 0 {
            return Err(FmcPlannerError::InvalidHorizon(horizon));
        }
        Ok(Self {
            population_size,
            horizon,
            dt,
            scoring,
            seed,
            action_selection: FmcActionSelectionPolicy::EmpiricalMean,
        })
    }

    /// Establece la política de selección de acción raíz (por defecto [`FmcActionSelectionPolicy::EmpiricalMean`]).
    ///
    /// Rechaza parámetros no finitos o negativos.
    pub fn with_action_selection(
        mut self,
        policy: FmcActionSelectionPolicy,
    ) -> Result<Self, FmcPlannerError> {
        policy.validate()?;
        self.action_selection = policy;
        Ok(self)
    }

    /// Política de selección de la acción raíz.
    pub const fn action_selection(&self) -> FmcActionSelectionPolicy {
        self.action_selection
    }

    /// Número de walkers `N` en la población FMC.
    pub const fn population_size(&self) -> usize {
        self.population_size
    }

    /// Horizonte máximo `H` de pasos simulados por decisión.
    pub const fn horizon(&self) -> usize {
        self.horizon
    }

    /// Paso temporal físico `dt` en segundos usado en cada transición simulada.
    pub const fn dt(&self) -> StepDt {
        self.dt
    }

    /// Hiperparámetros de scoring y resampling FMC.
    pub const fn scoring(&self) -> &FmcScoringConfig {
        &self.scoring
    }

    /// Semilla determinista maestra para derivar streams RNG de walkers y resampling.
    pub const fn seed(&self) -> MasterSeed {
        self.seed
    }
}

impl Default for FmcPlannerConfig {
    fn default() -> Self {
        Self {
            population_size: 16,
            horizon: 8,
            dt: StepDt::from_millis_f64(100.0).expect("100 ms must be valid StepDt"),
            scoring: FmcScoringConfig::default(),
            seed: MasterSeed::new(0),
            action_selection: FmcActionSelectionPolicy::EmpiricalMean,
        }
    }
}

/// Métricas de una acción raíz `a_0` evaluada en la selección final.
#[derive(Debug, Clone, PartialEq)]
pub struct FmcRootActionStat {
    /// Acción raíz evaluada.
    pub action: Action,
    /// Número de walkers viables que descienden de la acción (`N(a_0)`).
    pub support_count: usize,
    /// Retorno acumulado medio de dichos walkers.
    pub mean_return: f64,
    /// Score según la [`FmcActionSelectionPolicy`] activa.
    pub score: f64,
}

/// Resumen estructurado y observable de una invocación de [`FmcPlanner`].
#[derive(Debug, Clone, PartialEq)]
pub struct FmcTraceSummary {
    first_action: Option<Action>,
    rollout_transitions_used: usize,
    cloning_events: usize,
    mean_virtual_reward: f64,
    unique_first_actions: usize,
    action_diversity: f64,
    status: PlanStatus,
    selected_support: usize,
    selected_score: Option<f64>,
    root_action_stats: Vec<FmcRootActionStat>,
}

impl FmcTraceSummary {
    /// Crea un [`FmcTraceSummary`] explícito.
    pub const fn new(
        first_action: Option<Action>,
        rollout_transitions_used: usize,
        cloning_events: usize,
        mean_virtual_reward: f64,
        unique_first_actions: usize,
        action_diversity: f64,
        status: PlanStatus,
    ) -> Self {
        Self {
            first_action,
            rollout_transitions_used,
            cloning_events,
            mean_virtual_reward,
            unique_first_actions,
            action_diversity,
            status,
            selected_support: 0,
            selected_score: None,
            root_action_stats: Vec::new(),
        }
    }

    /// Adjunta las métricas de selección de acción raíz (soporte y score de la ganadora y
    /// estadísticas por acción raíz).
    pub fn with_selection(
        mut self,
        selected_support: usize,
        selected_score: f64,
        root_action_stats: Vec<FmcRootActionStat>,
    ) -> Self {
        self.selected_support = selected_support;
        self.selected_score = Some(selected_score);
        self.root_action_stats = root_action_stats;
        self
    }

    /// Soporte `N(a_0)` de la acción seleccionada (`0` si no hubo selección).
    pub const fn selected_action_support(&self) -> usize {
        self.selected_support
    }

    /// Score de la acción seleccionada según la política activa (`None` si no hubo selección).
    pub const fn selected_action_score(&self) -> Option<f64> {
        self.selected_score
    }

    /// Métricas por acción raíz evaluada (vacío si no hubo selección).
    pub fn root_action_stats(&self) -> &[FmcRootActionStat] {
        &self.root_action_stats
    }

    /// Primera acción autorizada (ganadora o `fallback` si aplica).
    pub const fn first_action(&self) -> Option<&Action> {
        self.first_action.as_ref()
    }

    /// Total de transiciones simuladas consumidas durante el rollout FMC.
    pub const fn rollout_transitions_used(&self) -> usize {
        self.rollout_transitions_used
    }

    /// Total de clonaciones ejecutadas durante el resampling FMC.
    pub const fn cloning_events(&self) -> usize {
        self.cloning_events
    }

    /// Media del valor virtual (`VR`) en el último paso evaluado (`0.0` si no hubo evaluación).
    pub const fn mean_virtual_reward(&self) -> f64 {
        self.mean_virtual_reward
    }

    /// Número de primeras acciones distintas representadas en la población final.
    pub const fn unique_first_actions(&self) -> usize {
        self.unique_first_actions
    }

    /// Diversidad normalizada de primeras acciones en `[0.0, 1.0]`.
    pub const fn action_diversity(&self) -> f64 {
        self.action_diversity
    }

    /// Estado final de la invocación del planificador.
    pub const fn status(&self) -> PlanStatus {
        self.status
    }
}

/// Planificador Fractal Monte Carlo (FMC) recedente de horizonte finito.
#[derive(Debug, Clone, PartialEq)]
pub struct FmcPlanner {
    config: FmcPlannerConfig,
    last_trace: Option<FmcTraceSummary>,
}

impl FmcPlanner {
    /// Crea un nuevo planificador FMC con la configuración validada `config`.
    pub const fn new(config: FmcPlannerConfig) -> Self {
        Self {
            config,
            last_trace: None,
        }
    }

    /// Devuelve la configuración del planificador.
    pub const fn config(&self) -> &FmcPlannerConfig {
        &self.config
    }

    /// Devuelve el último [`FmcTraceSummary`] registrado por [`Planner::plan`], si existe.
    pub const fn last_trace(&self) -> Option<&FmcTraceSummary> {
        self.last_trace.as_ref()
    }

    /// Ejecuta una planificación FMC recedente usando función de recompensa y proyección `phi` por defecto.
    pub fn plan_with_context<M: GenerativeModel + ?Sized>(
        &self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<(PlanResult, FmcTraceSummary), ModelError> {
        self.plan_with_trace(ctx, request, default_reward_fn, default_phi)
    }

    /// Ejecuta una planificación FMC recedente sobre `ctx` con `reward_fn` y proyección `phi` explícitas.
    ///
    /// Devuelve el par `(PlanResult, FmcTraceSummary)` sin acceder ni mutar ningún entorno vivo.
    pub fn plan_with_trace<M, R, P>(
        &self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
        mut reward_fn: R,
        mut phi: P,
    ) -> Result<(PlanResult, FmcTraceSummary), ModelError>
    where
        M: GenerativeModel + ?Sized,
        R: FnMut(&State, &Action, &Transition) -> f64,
        P: FnMut(&State) -> Vec<f64>,
    {
        let mut tracker = request.budget().tracker();

        if request.is_cancelled() {
            let result = PlanResult::cancelled(0);
            let trace = FmcTraceSummary::new(None, 0, 0, 0.0, 0, 0.0, PlanStatus::Cancelled);
            return Ok((result, trace));
        }

        let Some(initial_state) = request.state() else {
            let result = PlanResult::no_valid_action(0);
            let trace = FmcTraceSummary::new(None, 0, 0, 0.0, 0, 0.0, PlanStatus::NoValidAction);
            return Ok((result, trace));
        };

        if tracker.is_exhausted() {
            let result = PlanResult::budget_exhausted(request.fallback().cloned(), 0);
            let trace = FmcTraceSummary::new(
                result.authorized_action().cloned(),
                0,
                0,
                0.0,
                0,
                0.0,
                PlanStatus::BudgetExhausted,
            );
            return Ok((result, trace));
        }

        let episode_tag = initial_state.tick().as_u64();
        let mut population = WalkerPopulation::new(
            initial_state.clone(),
            self.config.population_size,
            self.config.seed,
            episode_tag,
            0,
            true,
        )
        .map_err(|_| ModelError::InvalidState {
            reason: "failed to initialize FMC walker population",
        })?;

        let mut planner_rng = self.config.seed.derive(StreamKey::Agent {
            episode: episode_tag,
            agent: 0,
        });

        let mut cloning_events = 0usize;
        let mut last_mean_vr = 0.0f64;
        let mut budget_exhausted = false;
        let mut any_step_completed = false;
        let mut last_vr: Option<Vec<f64>> = None;

        for step_idx in 0..self.config.horizon {
            if request.is_cancelled() {
                let used = tracker.transitions_used();
                let (unique_first, diversity) = summarize_first_action_diversity(&population);
                let result = PlanResult::cancelled(used);
                let trace = FmcTraceSummary::new(
                    None,
                    used,
                    cloning_events,
                    last_mean_vr,
                    unique_first,
                    diversity,
                    PlanStatus::Cancelled,
                );
                return Ok((result, trace));
            }

            if tracker.is_exhausted() {
                budget_exhausted = true;
                break;
            }

            if population.all_terminal() {
                break;
            }

            let actions = sample_population_actions(
                request.action_space(),
                self.config.population_size,
                step_idx == 0,
                self.config.scoring.unbiased_sampling(),
                &mut planner_rng,
            )?;

            let step_report = match population.step_population(
                ctx.model(),
                &actions,
                self.config.dt,
                &mut tracker,
                &mut reward_fn,
            ) {
                Ok(report) => report,
                Err(PopulationError::ModelStepFailed {
                    source: ModelError::TerminalState,
                    ..
                }) if !any_step_completed => {
                    let used = tracker.transitions_used();
                    let result = PlanResult::no_valid_action(used);
                    let trace = FmcTraceSummary::new(
                        None,
                        used,
                        cloning_events,
                        last_mean_vr,
                        0,
                        0.0,
                        PlanStatus::NoValidAction,
                    );
                    return Ok((result, trace));
                }
                Err(PopulationError::ModelStepFailed { source, .. }) => return Err(source),
                Err(PopulationError::NonFiniteReward { .. }) => {
                    return Err(ModelError::InvalidState {
                        reason: "FMC reward function produced non-finite value",
                    });
                }
                Err(_) => {
                    return Err(ModelError::InvalidState {
                        reason: "FMC walker population step failed",
                    });
                }
            };

            if step_report.stepped() > 0 {
                any_step_completed = true;
            }
            if step_report.budget_exhausted() {
                budget_exhausted = true;
            }
            if step_report.stepped() == 0 && budget_exhausted {
                break;
            }

            let score = score_population(
                &population,
                &mut phi,
                &self.config.scoring,
                &mut planner_rng,
            )
            .map_err(|_| ModelError::InvalidState {
                reason: "FMC scoring failed",
            })?;
            last_mean_vr = mean_slice(score.virtual_rewards());
            last_vr = Some(score.virtual_rewards().to_vec());

            if budget_exhausted {
                break;
            }

            if step_idx + 1 < self.config.horizon && !population.all_terminal() {
                let resample_report = resample_population(
                    &mut population,
                    score.virtual_rewards(),
                    &mut planner_rng,
                    step_idx as u64,
                )
                .map_err(|_| ModelError::InvalidState {
                    reason: "FMC resampling failed",
                })?;
                cloning_events += resample_report.cloned_count();
                // The population changed: the VR vector no longer aligns with it.
                last_vr = None;
            }
        }

        let transitions_used = tracker.transitions_used();
        let (unique_first, diversity) = summarize_first_action_diversity(&population);

        if budget_exhausted {
            let result =
                PlanResult::budget_exhausted(request.fallback().cloned(), transitions_used);
            let trace = FmcTraceSummary::new(
                result.authorized_action().cloned(),
                transitions_used,
                cloning_events,
                last_mean_vr,
                unique_first,
                diversity,
                PlanStatus::BudgetExhausted,
            );
            return Ok((result, trace));
        }

        match select_winning_first_action(
            &population,
            self.config.action_selection,
            &self.config.scoring,
            last_vr.as_deref(),
        ) {
            Some(selection) => {
                let best_action = selection.action.clone();
                let result = PlanResult::completed(best_action.clone(), transitions_used);
                let trace = FmcTraceSummary::new(
                    Some(best_action),
                    transitions_used,
                    cloning_events,
                    last_mean_vr,
                    unique_first,
                    diversity,
                    PlanStatus::Completed,
                )
                .with_selection(
                    selection.support_count,
                    selection.score,
                    selection.stats,
                );
                Ok((result, trace))
            }
            None => {
                let result = PlanResult::no_valid_action(transitions_used);
                let trace = FmcTraceSummary::new(
                    None,
                    transitions_used,
                    cloning_events,
                    last_mean_vr,
                    unique_first,
                    diversity,
                    PlanStatus::NoValidAction,
                );
                Ok((result, trace))
            }
        }
    }
}

impl Planner for FmcPlanner {
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        let (result, trace) = self.plan_with_context(ctx, request)?;
        self.last_trace = Some(trace);
        Ok(result)
    }
}

/// Default projection as a borrowed slice (usable with `score_population_in_place`).
pub(crate) fn default_phi_slice(state: &State) -> &[f64] {
    state.values()
}

pub(crate) fn default_phi(state: &State) -> Vec<f64> {
    default_phi_slice(state).to_vec()
}

pub(crate) fn default_reward_fn(_prev: &State, _action: &Action, transition: &Transition) -> f64 {
    match transition.terminal() {
        Terminal::Terminated(TerminalReason::GoalReached) => 1.0,
        Terminal::Terminated(TerminalReason::Failure | TerminalReason::InvariantViolated) => -1.0,
        Terminal::Ongoing | Terminal::Terminated(_) => 0.0,
    }
}

pub(crate) fn sample_population_actions(
    space: &ActionSpace,
    population_size: usize,
    stratify_discrete: bool,
    unbiased: bool,
    rng: &mut dyn RngSource,
) -> Result<Vec<Action>, ModelError> {
    let mut actions = Vec::with_capacity(population_size);
    match space {
        ActionSpace::Discrete { cardinality } => {
            // Legacy modulo path unless `unbiased` (opt-in, changes the RNG sequence).
            let draw = |rng: &mut dyn RngSource| {
                if unbiased {
                    rng.next_usize_unbiased(*cardinality)
                } else {
                    (rng.next_u64() as usize) % *cardinality
                }
            };
            let offset = draw(rng);
            for w in 0..population_size {
                let idx = if stratify_discrete {
                    (offset + w) % *cardinality
                } else {
                    draw(rng)
                };
                let act =
                    Action::discrete_in_space(idx, space).map_err(ModelError::InvalidAction)?;
                actions.push(act);
            }
        }
        ActionSpace::Continuous { bounds } => {
            for _ in 0..population_size {
                let mut values = Vec::with_capacity(bounds.len());
                for bound in bounds {
                    let span = bound.max() - bound.min();
                    let sample =
                        (bound.min() + rng.next_f64() * span).clamp(bound.min(), bound.max());
                    values.push(sample);
                }
                let act = Action::continuous_in_space(values, space)
                    .map_err(ModelError::InvalidAction)?;
                actions.push(act);
            }
        }
    }
    Ok(actions)
}

fn summarize_first_action_diversity(population: &WalkerPopulation) -> (usize, f64) {
    let mut distinct: Vec<&Action> = Vec::new();
    let mut with_action = 0usize;
    for action in population.first_actions().iter().flatten() {
        with_action += 1;
        if !distinct.contains(&action) {
            distinct.push(action);
        }
    }
    let unique = distinct.len();
    let diversity = if with_action == 0 {
        0.0
    } else {
        (unique as f64) / (with_action as f64)
    };
    (unique, diversity)
}

pub(crate) fn mean_slice(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / (values.len() as f64)
}

/// Resultado de [`select_winning_first_action`]: acción ganadora, su soporte/score y métricas por acción.
pub(crate) struct ActionSelection {
    pub(crate) action: Action,
    pub(crate) support_count: usize,
    pub(crate) score: f64,
    pub(crate) stats: Vec<FmcRootActionStat>,
}

struct ActionAcc {
    action: Action,
    returns: Vec<f64>,
    mass: f64,
}

fn mean_std(values: &[f64], sample: bool) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / (n as f64);
    let denom = if sample { n.saturating_sub(1) } else { n };
    if denom == 0 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (denom as f64);
    (mean, var.sqrt())
}

/// Selecciona la acción raíz ganadora entre los walkers viables según `policy`.
///
/// `virtual_rewards` (alineado con `population`) solo se usa en
/// [`FmcActionSelectionPolicy::VirtualValueMass`]; si es `None` o su longitud no coincide se usa
/// la aproximación por retornos relativizados. El empate se resuelve por mayor soporte.
pub(crate) fn select_winning_first_action(
    population: &WalkerPopulation,
    policy: FmcActionSelectionPolicy,
    scoring: &FmcScoringConfig,
    virtual_rewards: Option<&[f64]>,
) -> Option<ActionSelection> {
    let vr = virtual_rewards.filter(|v| v.len() == population.len());
    let relativized =
        if matches!(policy, FmcActionSelectionPolicy::VirtualValueMass) && vr.is_none() {
            relativize_viable(
                population.cumulative_rewards(),
                population.terminals(),
                scoring,
            )
            .ok()
        } else {
            None
        };

    let mut accs: Vec<ActionAcc> = Vec::new();
    let mut all_returns: Vec<f64> = Vec::new();
    for i in 0..population.len() {
        if is_failure_terminal(population.terminals()[i]) {
            continue;
        }
        let Some(ref first_action) = population.first_actions()[i] else {
            continue;
        };
        let reward = population.cumulative_rewards()[i];
        let mass = match (vr, relativized.as_deref()) {
            (Some(v), _) => v[i],
            (None, Some(r)) => r[i],
            (None, None) => 0.0,
        };
        let mass = if mass.is_finite() { mass.max(0.0) } else { 0.0 };
        all_returns.push(reward);
        if let Some(entry) = accs.iter_mut().find(|c| &c.action == first_action) {
            entry.returns.push(reward);
            entry.mass += mass;
        } else {
            accs.push(ActionAcc {
                action: first_action.clone(),
                returns: vec![reward],
                mass,
            });
        }
    }

    let (mean_pop, std_pop) = mean_std(&all_returns, false);
    let stats: Vec<FmcRootActionStat> = accs
        .into_iter()
        .map(|acc| {
            let n = acc.returns.len();
            let (mean_a, std_sample) = mean_std(&acc.returns, true);
            let score = match policy {
                FmcActionSelectionPolicy::EmpiricalMean => mean_a,
                FmcActionSelectionPolicy::LowerConfidenceBound {
                    lcb_c,
                    prior_weight,
                } => {
                    let nf = n as f64;
                    let shrunk = (nf * mean_a + prior_weight * mean_pop) / (nf + prior_weight);
                    let std = if n >= 2 { std_sample } else { std_pop };
                    shrunk - lcb_c * std / nf.sqrt()
                }
                FmcActionSelectionPolicy::VirtualValueMass => acc.mass,
            };
            FmcRootActionStat {
                action: acc.action,
                support_count: n,
                mean_return: mean_a,
                score,
            }
        })
        .collect();

    let best = stats
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.score
                .total_cmp(&b.score)
                .then_with(|| a.support_count.cmp(&b.support_count))
        })
        .map(|(idx, _)| idx)?;
    let winner = &stats[best];
    Some(ActionSelection {
        action: winner.action.clone(),
        support_count: winner.support_count,
        score: winner.score,
        stats: stats.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::DiscreteAction;
    use crate::domain::Observation;
    use crate::model::DeterministicRng;
    use crate::planner::{CancelToken, PlanBudget};
    use crate::units::{SimDuration, Tick};

    struct GoalLineModel {
        space: ActionSpace,
        goal_x: i64,
        fail_x: i64,
    }

    impl GoalLineModel {
        fn new(goal_x: i64, fail_x: i64) -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(), // 0: -1, 1: +1
                goal_x,
                fail_x,
            }
        }
    }

    impl GenerativeModel for GoalLineModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            self.space
                .validate(action)
                .map_err(ModelError::InvalidAction)?;
            let cur_x = state.values()[0] as i64;
            if cur_x == self.goal_x || cur_x == self.fail_x {
                return Err(ModelError::TerminalState);
            }
            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            let next_x = if idx == 1 { cur_x + 1 } else { cur_x - 1 };
            let terminal = if next_x == self.goal_x {
                Terminal::Terminated(TerminalReason::GoalReached)
            } else if next_x == self.fail_x {
                Terminal::Terminated(TerminalReason::Failure)
            } else {
                Terminal::Ongoing
            };
            let next_state = state
                .advance(dt, vec![next_x as f64])
                .map_err(ModelError::TransitionFailed)?;
            let obs =
                Observation::new(next_state.tick(), next_state.elapsed(), vec![next_x as f64])
                    .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    #[test]
    fn config_rejects_zero_population_or_horizon() {
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let scoring = FmcScoringConfig::default();
        let seed = MasterSeed::new(7);

        assert_eq!(
            FmcPlannerConfig::new(0, 4, dt, scoring, seed).unwrap_err(),
            FmcPlannerError::InvalidPopulationSize(0)
        );
        assert_eq!(
            FmcPlannerConfig::new(4, 0, dt, scoring, seed).unwrap_err(),
            FmcPlannerError::InvalidHorizon(0)
        );
    }

    #[test]
    fn planner_selects_action_toward_goal_and_records_trace() {
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let model = GoalLineModel::new(3, -3);
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0]).unwrap();
        let config =
            FmcPlannerConfig::new(12, 4, dt, FmcScoringConfig::default(), MasterSeed::new(42))
                .unwrap();
        let mut planner = FmcPlanner::new(config);
        let mut rng = DeterministicRng::new(99);
        let mut ctx = PlanningContext::new(&model, &mut rng);

        let request = PlanRequest::from_state(
            start,
            model.action_space().clone(),
            PlanBudget::with_transitions(100),
        );
        let result = planner.plan(&mut ctx, &request).unwrap();
        assert_eq!(result.status(), PlanStatus::Completed);
        let right = Action::discrete_in_space(1, model.action_space()).unwrap();
        assert_eq!(result.authorized_action(), Some(&right));

        let trace = planner.last_trace().expect("trace must be recorded");
        assert_eq!(trace.status(), PlanStatus::Completed);
        assert_eq!(trace.first_action(), Some(&right));
        assert!(trace.rollout_transitions_used() > 0);
        assert!(trace.rollout_transitions_used() <= 100);
        assert!(trace.mean_virtual_reward().is_finite());
        assert!(trace.unique_first_actions() >= 1);
        assert!(trace.action_diversity() > 0.0 && trace.action_diversity() <= 1.0);
    }

    #[test]
    fn planner_respects_budget_cancel_terminal_and_fallback() {
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let model = GoalLineModel::new(3, -3);
        let fallback = Action::discrete_in_space(0, model.action_space()).unwrap();
        let config =
            FmcPlannerConfig::new(8, 4, dt, FmcScoringConfig::default(), MasterSeed::new(123))
                .unwrap();
        let planner = FmcPlanner::new(config);
        let mut rng = DeterministicRng::new(1);
        let mut ctx = PlanningContext::new(&model, &mut rng);

        // 1. Budget exhaustion -> uses explicit fallback
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let req_budget = PlanRequest::from_state(
            start.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(3),
        )
        .with_fallback(fallback.clone())
        .unwrap();
        let (res_budget, trace_budget) = planner.plan_with_context(&mut ctx, &req_budget).unwrap();
        assert_eq!(res_budget.status(), PlanStatus::BudgetExhausted);
        assert_eq!(res_budget.transitions_used(), 3);
        assert_eq!(res_budget.authorized_action(), Some(&fallback));
        assert_eq!(trace_budget.status(), PlanStatus::BudgetExhausted);
        assert_eq!(trace_budget.first_action(), Some(&fallback));

        // 2. Cancellation -> Cancelled, no action authorized
        let token = CancelToken::cancelled();
        let req_cancel = PlanRequest::from_state(
            start,
            model.action_space().clone(),
            PlanBudget::with_transitions(50),
        )
        .with_cancel_token(token)
        .with_fallback(fallback)
        .unwrap();
        let (res_cancel, trace_cancel) = planner.plan_with_context(&mut ctx, &req_cancel).unwrap();
        assert_eq!(res_cancel.status(), PlanStatus::Cancelled);
        assert_eq!(res_cancel.authorized_action(), None);
        assert_eq!(res_cancel.transitions_used(), 0);
        assert_eq!(trace_cancel.status(), PlanStatus::Cancelled);
        assert_eq!(trace_cancel.first_action(), None);

        // 3. Terminal initial state -> NoValidAction, 0 transitions
        let terminal_start = State::new(Tick::ZERO, SimDuration::ZERO, vec![3.0]).unwrap();
        let req_terminal = PlanRequest::from_state(
            terminal_start,
            model.action_space().clone(),
            PlanBudget::with_transitions(50),
        );
        let (res_term, trace_term) = planner.plan_with_context(&mut ctx, &req_terminal).unwrap();
        assert_eq!(res_term.status(), PlanStatus::NoValidAction);
        assert_eq!(res_term.authorized_action(), None);
        assert_eq!(res_term.transitions_used(), 0);
        assert_eq!(trace_term.status(), PlanStatus::NoValidAction);
    }

    const LCB: FmcActionSelectionPolicy = FmcActionSelectionPolicy::LowerConfidenceBound {
        lcb_c: 4.0,
        prior_weight: 2.0,
    };

    /// Model where action 1 fails terminally from `x == 1` and is otherwise safe.
    struct RiskyModel {
        space: ActionSpace,
    }

    impl GenerativeModel for RiskyModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, ModelError> {
            let x = state.values()[0];
            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0);
            let terminal = if idx == 1 && x == 1.0 {
                Terminal::Terminated(TerminalReason::Failure)
            } else {
                Terminal::Ongoing
            };
            let next_state = state
                .advance(dt, vec![x])
                .map_err(ModelError::TransitionFailed)?;
            let obs = Observation::new(next_state.tick(), next_state.elapsed(), vec![x])
                .map_err(ModelError::TransitionFailed)?;
            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(ModelError::TransitionFailed)
        }
    }

    /// Scoring with `z_clip = 1`. With the default `z_clip = 5` the relativization of the lone
    /// survivor is `exp(~3.9)` (z-score over a nearly constant viable set), which by itself
    /// outweighs 15 average walkers in `VirtualValueMass`; bounding the clip keeps a single
    /// outlier from dominating the mass.
    fn survivor_bias_scoring() -> FmcScoringConfig {
        FmcScoringConfig::new(1.0, 1.0, 1e-8, 1.0).unwrap()
    }

    /// 15 safe walkers (action 0, R=1.00), 1 risky survivor (action 1, R=1.01), 14 risky failures.
    ///
    /// `VR` is computed with a tight `z_clip` (see [`survivor_bias_scoring`]).
    fn survivor_bias_population() -> (WalkerPopulation, Vec<f64>) {
        let model = RiskyModel {
            space: ActionSpace::discrete(2).unwrap(),
        };
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let mk = |x: f64| State::new(Tick::ZERO, SimDuration::ZERO, vec![x]).unwrap();
        let mut states = vec![mk(0.0); 16];
        states.extend(vec![mk(1.0); 14]);
        let mut population =
            WalkerPopulation::from_states(states, MasterSeed::new(5), 0, 0, false).unwrap();
        let space = model.action_space();
        let safe = Action::discrete_in_space(0, space).unwrap();
        let risky = Action::discrete_in_space(1, space).unwrap();
        let mut actions = vec![safe; 15];
        actions.extend(vec![risky; 15]);
        let mut tracker = PlanBudget::with_transitions(1000).tracker();
        population
            .step_population(&model, &actions, dt, &mut tracker, |_, a, _| {
                if a.as_discrete().map(DiscreteAction::index) == Some(1) {
                    1.01
                } else {
                    1.0
                }
            })
            .unwrap();
        let score = score_population(
            &population,
            default_phi,
            &survivor_bias_scoring(),
            &mut DeterministicRng::new(3),
        )
        .unwrap();
        let vr = score.virtual_rewards().to_vec();
        (population, vr)
    }

    #[test]
    fn ac1_lcb_and_vvm_prefer_robust_branch_over_lone_survivor() {
        let (population, vr) = survivor_bias_population();
        let scoring = survivor_bias_scoring();
        let space = ActionSpace::discrete(2).unwrap();
        let safe = Action::discrete_in_space(0, &space).unwrap();
        let risky = Action::discrete_in_space(1, &space).unwrap();

        // Contrast: the classic mean is fooled by the single survivor (R=1.01, N=1).
        let mean = select_winning_first_action(
            &population,
            FmcActionSelectionPolicy::EmpiricalMean,
            &scoring,
            Some(&vr),
        )
        .unwrap();
        assert_eq!(mean.action, risky);
        assert_eq!(mean.support_count, 1);

        for policy in [LCB, FmcActionSelectionPolicy::VirtualValueMass] {
            let sel =
                select_winning_first_action(&population, policy, &scoring, Some(&vr)).unwrap();
            assert_eq!(sel.action, safe, "{policy:?}");
            assert_eq!(sel.support_count, 15);
            assert!(sel.score.is_finite());
        }

        // VirtualValueMass fallback without VR (relativized returns) also prefers the robust branch.
        let fallback = select_winning_first_action(
            &population,
            FmcActionSelectionPolicy::VirtualValueMass,
            &scoring,
            None,
        )
        .unwrap();
        assert_eq!(fallback.action, safe);
    }

    #[test]
    fn ac2_policies_converge_without_variance_or_with_h1() {
        // Population without variance: one action, identical returns.
        let model = RiskyModel {
            space: ActionSpace::discrete(2).unwrap(),
        };
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let mut population =
            WalkerPopulation::new(start, 6, MasterSeed::new(1), 0, 0, false).unwrap();
        let act = Action::discrete_in_space(0, model.action_space()).unwrap();
        let mut tracker = PlanBudget::with_transitions(100).tracker();
        population
            .step_population(
                &model,
                &vec![act.clone(); 6],
                dt,
                &mut tracker,
                |_, _, _| 1.0,
            )
            .unwrap();
        let scoring = FmcScoringConfig::default();
        let policies = [
            FmcActionSelectionPolicy::EmpiricalMean,
            LCB,
            FmcActionSelectionPolicy::VirtualValueMass,
        ];
        for policy in policies {
            let sel = select_winning_first_action(&population, policy, &scoring, None).unwrap();
            assert_eq!(sel.action, act);
            assert!(sel.score.is_finite());
        }

        // Full planner with H=1: every policy picks the goal-reaching action.
        let goal_model = GoalLineModel::new(3, -3);
        let right = Action::discrete_in_space(1, goal_model.action_space()).unwrap();
        for policy in policies {
            let config =
                FmcPlannerConfig::new(12, 1, dt, FmcScoringConfig::default(), MasterSeed::new(42))
                    .unwrap()
                    .with_action_selection(policy)
                    .unwrap();
            let planner = FmcPlanner::new(config);
            let mut rng = DeterministicRng::new(99);
            let mut ctx = PlanningContext::new(&goal_model, &mut rng);
            let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![2.0]).unwrap();
            let request = PlanRequest::from_state(
                start,
                goal_model.action_space().clone(),
                PlanBudget::with_transitions(100),
            );
            let (res, _) = planner.plan_with_context(&mut ctx, &request).unwrap();
            assert_eq!(res.authorized_action(), Some(&right), "{policy:?}");
        }
    }

    #[test]
    fn ac3_trace_exposes_selected_support_and_score() {
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let model = GoalLineModel::new(3, -3);
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![2.0]).unwrap();
        let config =
            FmcPlannerConfig::new(12, 1, dt, FmcScoringConfig::default(), MasterSeed::new(42))
                .unwrap()
                .with_action_selection(LCB)
                .unwrap();
        let planner = FmcPlanner::new(config);
        let mut rng = DeterministicRng::new(99);
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let request = PlanRequest::from_state(
            start,
            model.action_space().clone(),
            PlanBudget::with_transitions(100),
        );
        let (_, trace) = planner.plan_with_context(&mut ctx, &request).unwrap();
        assert!(trace.selected_action_support() >= 1);
        assert!(trace.selected_action_score().is_some_and(f64::is_finite));
        let stats = trace.root_action_stats();
        assert!(!stats.is_empty());
        let winner = stats
            .iter()
            .find(|s| Some(&s.action) == trace.first_action())
            .unwrap();
        assert_eq!(winner.support_count, trace.selected_action_support());
        assert_eq!(Some(winner.score), trace.selected_action_score());
    }

    #[test]
    fn action_selection_defaults_and_validation() {
        assert_eq!(
            FmcPlannerConfig::default().action_selection(),
            FmcActionSelectionPolicy::EmpiricalMean
        );
        for (c, w, name) in [
            (f64::NAN, 1.0, "lcb_c"),
            (-1.0, 1.0, "lcb_c"),
            (1.0, f64::INFINITY, "prior_weight"),
            (1.0, -0.5, "prior_weight"),
        ] {
            let err = FmcPlannerConfig::default()
                .with_action_selection(FmcActionSelectionPolicy::LowerConfidenceBound {
                    lcb_c: c,
                    prior_weight: w,
                })
                .unwrap_err();
            assert_eq!(err, FmcPlannerError::InvalidActionSelection(name));
        }
    }
}
