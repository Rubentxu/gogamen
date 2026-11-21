//! Carrera entre acciones raíz con parada por dominancia o cobertura (GOG-190, `std`-only).
//!
//! Envuelve [`IncrementalFmcSession`] en trozos de un paso de horizonte (`population_size`
//! transiciones) y, tras cada trozo, aplica una regla de parada pura ([`evaluate_stop`]) sobre
//! las estadísticas por acción raíz ([`FmcRootActionStat`], las mismas que usa
//! [`crate::FmcActionSelectionPolicy`]):
//!
//! - **Dominancia**: la acción líder (mayor score de la política activa) tiene una cota inferior
//!   de cuota de población (Hoeffding, `dominance_confidence`) que supera la cota superior de la
//!   cuota del segundo. La cuota de descendientes es la señal nativa de FMC: el resampling
//!   concentra walkers en ramas viables. Los walkers clonados no son independientes, por lo que
//!   la confianza es heurística, no una garantía estadística exacta.
//! - **Cobertura**: `profundidad_completada * fracción_de_raíces_con_soporte >= umbral`.
//! - **Presupuesto**: si ninguna regla dispara, se consume el presupuesto completo
//!   ([`StopReason::BudgetExhausted`]); es el fallback ante cobertura insuficiente.
//!
//! Invariantes: la regla nunca se evalúa por debajo de `min_budget` transiciones; el
//! [`ActionGuard`] se evalúa SIEMPRE sobre la propuesta final (la parada solo decide cuánto
//! planificar, nunca autoriza); el resultado es determinista dado el planificador sembrado.
//! El presupuesto ahorrado es solo contabilidad: reservarlo para momentos decisivos
//! (`decisive_moments`) es decisión del llamador.

use std::error::Error;
use std::fmt;
use std::time::Duration;

use crate::action::{Action, ActionSpace};
use crate::clock::VirtualMonotonicClock;
use crate::domain::{State, Terminal};
use crate::fmc_incremental::{CheckpointStatus, IncrementalFmcSession};
use crate::fmc_planner::{FmcPlannerConfig, FmcRootActionStat, select_winning_first_action};
use crate::guard::{ActionGuard, ActionProposal, DomainPrecondition, GuardContext, GuardDecision};
use crate::model::{DeterministicRng, GenerativeModel, PlanningContext};
use crate::planner::{CancelToken, PlanBudget};

/// Motivo por el que terminó la planificación de un tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StopReason {
    /// El líder domina con la confianza declarada.
    Dominance,
    /// La cobertura estimada alcanzó el umbral declarado.
    Coverage,
    /// Se consumió el presupuesto completo (también fallback ante evidencia/cobertura insuficiente).
    BudgetExhausted,
}

impl StopReason {
    /// Etiqueta estable para tablas y logs.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dominance => "Dominance",
            Self::Coverage => "Coverage",
            Self::BudgetExhausted => "BudgetExhausted",
        }
    }
}

/// Errores de configuración de [`RacingStopConfig`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RacingStopError {
    /// `min_budget` debe ser `>= 1` y `<= max_budget`.
    InvalidBudget,
    /// `dominance_confidence` debe estar en `(0, 1)`.
    InvalidConfidence,
    /// `coverage_threshold` debe estar en `(0, 1]`.
    InvalidCoverage,
    /// `min_support` debe ser `>= 1`.
    InvalidSupport,
    /// El planificador no pudo construir la sesión incremental.
    Session(String),
}

impl fmt::Display for RacingStopError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget => write!(f, "racing stop: need 1 <= min_budget <= max_budget"),
            Self::InvalidConfidence => write!(f, "racing stop: dominance_confidence not in (0,1)"),
            Self::InvalidCoverage => write!(f, "racing stop: coverage_threshold not in (0,1]"),
            Self::InvalidSupport => write!(f, "racing stop: min_support must be >= 1"),
            Self::Session(e) => write!(f, "racing stop: session error: {e}"),
        }
    }
}

impl Error for RacingStopError {}

/// Configuración declarada de la parada anticipada.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RacingStopConfig {
    min_budget: usize,
    max_budget: usize,
    dominance_confidence: f64,
    coverage_threshold: f64,
    min_support: usize,
    stop_enabled: bool,
}

impl RacingStopConfig {
    /// Configuración con parada anticipada activa.
    pub fn new(
        min_budget: usize,
        max_budget: usize,
        dominance_confidence: f64,
        coverage_threshold: f64,
        min_support: usize,
    ) -> Result<Self, RacingStopError> {
        if min_budget == 0 || min_budget > max_budget {
            return Err(RacingStopError::InvalidBudget);
        }
        if !(dominance_confidence.is_finite()
            && dominance_confidence > 0.0
            && dominance_confidence < 1.0)
        {
            return Err(RacingStopError::InvalidConfidence);
        }
        if !(coverage_threshold.is_finite()
            && coverage_threshold > 0.0
            && coverage_threshold <= 1.0)
        {
            return Err(RacingStopError::InvalidCoverage);
        }
        if min_support == 0 {
            return Err(RacingStopError::InvalidSupport);
        }
        Ok(Self {
            min_budget,
            max_budget,
            dominance_confidence,
            coverage_threshold,
            min_support,
            stop_enabled: true,
        })
    }

    /// Línea base de presupuesto fijo: nunca para antes de `budget` (o del horizonte).
    pub fn fixed(budget: usize) -> Result<Self, RacingStopError> {
        let mut cfg = Self::new(budget.max(1), budget.max(1), 0.5, 1.0, 1)?;
        cfg.stop_enabled = false;
        Ok(cfg)
    }

    /// Presupuesto mínimo de transiciones antes de evaluar cualquier parada.
    pub const fn min_budget(&self) -> usize {
        self.min_budget
    }
    /// Presupuesto máximo de transiciones por decisión.
    pub const fn max_budget(&self) -> usize {
        self.max_budget
    }
    /// Confianza de dominancia (Hoeffding).
    pub const fn dominance_confidence(&self) -> f64 {
        self.dominance_confidence
    }
    /// Umbral de cobertura.
    pub const fn coverage_threshold(&self) -> f64 {
        self.coverage_threshold
    }
    /// Soporte mínimo (walkers viables) para contar una raíz como cubierta.
    pub const fn min_support(&self) -> usize {
        self.min_support
    }
    /// `false` en la línea base de presupuesto fijo.
    pub const fn stop_enabled(&self) -> bool {
        self.stop_enabled
    }
}

/// Cobertura estimada: `(horizon_step / horizon) * (raíces con soporte >= min_support / acciones)`.
pub fn coverage(
    stats: &[FmcRootActionStat],
    min_support: usize,
    horizon_step: usize,
    horizon: usize,
    num_actions: usize,
) -> f64 {
    if horizon == 0 || num_actions == 0 {
        return 0.0;
    }
    let supported = stats
        .iter()
        .filter(|s| s.support_count >= min_support)
        .count();
    let depth = (horizon_step as f64 / horizon as f64).min(1.0);
    depth * (supported.min(num_actions) as f64 / num_actions as f64)
}

/// Semiancho de Hoeffding para una cuota sobre `n` muestras con la confianza dada.
fn hoeffding_eps(confidence: f64, n: usize) -> f64 {
    if n == 0 {
        return f64::INFINITY;
    }
    ((1.0 / (1.0 - confidence)).ln() / (2.0 * n as f64)).sqrt()
}

/// Regla de parada pura. `stats` son las estadísticas por acción raíz (soporte y score),
/// `walkers` el tamaño de población. Devuelve `None` si hay que seguir planificando.
///
/// Nunca devuelve `Some` si `transitions_used < config.min_budget()` o si la parada está
/// desactivada. Es independiente del orden de `stats`.
pub fn evaluate_stop(
    config: &RacingStopConfig,
    stats: &[FmcRootActionStat],
    walkers: usize,
    transitions_used: usize,
    horizon_step: usize,
    horizon: usize,
    num_actions: usize,
) -> Option<StopReason> {
    if !config.stop_enabled || transitions_used < config.min_budget || stats.is_empty() {
        return None;
    }
    let leader = stats.iter().max_by(|a, b| {
        a.score
            .total_cmp(&b.score)
            .then_with(|| a.support_count.cmp(&b.support_count))
    })?;
    let eps = hoeffding_eps(config.dominance_confidence, walkers);
    let share = |s: &FmcRootActionStat| s.support_count as f64 / walkers.max(1) as f64;
    let runner_ucb = stats
        .iter()
        .filter(|s| s.action != leader.action)
        .map(|s| (share(s) + eps).min(1.0))
        .fold(0.0_f64, f64::max);
    if leader.support_count >= config.min_support && share(leader) - eps > runner_ucb {
        return Some(StopReason::Dominance);
    }
    let cov = coverage(
        stats,
        config.min_support,
        horizon_step,
        horizon,
        num_actions,
    );
    if cov >= config.coverage_threshold {
        return Some(StopReason::Coverage);
    }
    None
}

/// Resultado de una decisión con parada anticipada.
#[derive(Debug, Clone, PartialEq)]
pub struct RacingDecision {
    /// Acción propuesta al guard (candidato del planificador o fallback).
    pub proposed: Option<Action>,
    /// Veredicto del [`ActionGuard`] (siempre evaluado).
    pub guard: GuardDecision,
    /// Motivo de la parada.
    pub reason: StopReason,
    /// Transiciones consumidas.
    pub transitions_used: usize,
    /// Trozos (pasos de horizonte) ejecutados.
    pub chunks: usize,
    /// Cobertura en el momento de parar.
    pub coverage: f64,
    /// `true` si nunca hubo evidencia (ningún walker viable) y se propuso el fallback.
    pub used_fallback_action: bool,
}

/// Planifica un tick en trozos de un paso de horizonte con parada anticipada.
///
/// `planner_config` debe llevar la semilla específica del tick (determinismo). `fallback` se
/// usa solo si no hay ningún candidato. El guard se evalúa siempre sobre la propuesta final.
#[allow(clippy::too_many_arguments)]
pub fn decide<M, P>(
    config: &RacingStopConfig,
    planner_config: &FmcPlannerConfig,
    model: &M,
    state: State,
    action_space: &ActionSpace,
    fallback: Option<Action>,
    guard: &ActionGuard<P>,
) -> Result<RacingDecision, RacingStopError>
where
    M: GenerativeModel + ?Sized,
    P: DomainPrecondition,
{
    let tick = state.tick();
    let budget = PlanBudget::with_transitions(config.max_budget);
    let mut session = IncrementalFmcSession::new(
        planner_config.clone(),
        state,
        action_space.clone(),
        budget.tracker(),
        None,
        Duration::ZERO,
        fallback,
        CancelToken::new(),
    )
    .map_err(|e| RacingStopError::Session(e.to_string()))?;
    let clock = VirtualMonotonicClock::new();
    let mut rng = DeterministicRng::new(0);
    let ctx = PlanningContext::new(model, &mut rng);
    let num_actions = match action_space {
        ActionSpace::Discrete { cardinality } => *cardinality,
        ActionSpace::Continuous { .. } => planner_config.population_size(),
    };

    let mut chunks = 0usize;
    let mut any_evidence = false;
    let mut cov: f64;
    let reason = loop {
        let outcome = session.step_checkpoint(&ctx, &clock, 1);
        chunks += 1;
        let stats = select_winning_first_action(
            session.population(),
            planner_config.action_selection(),
            planner_config.scoring(),
            None,
        )
        .map(|s| s.stats)
        .unwrap_or_default();
        any_evidence |= !stats.is_empty();
        cov = coverage(
            &stats,
            config.min_support,
            session.horizon_step(),
            planner_config.horizon(),
            num_actions,
        );
        if !matches!(outcome.status(), CheckpointStatus::PausedChunkLimit) {
            break StopReason::BudgetExhausted;
        }
        if let Some(r) = evaluate_stop(
            config,
            &stats,
            planner_config.population_size(),
            session.transitions_budget().transitions_used(),
            session.horizon_step(),
            planner_config.horizon(),
            num_actions,
        ) {
            break r;
        }
    };

    let proposed = session.best_admissible_candidate().cloned();
    let proposal = match &proposed {
        Some(a) => ActionProposal::authorized(a.clone(), tick),
        None => ActionProposal::unauthorized(tick),
    };
    let gctx = GuardContext::now(action_space, tick, Terminal::Ongoing);
    let guard_decision = guard.evaluate(&proposal, &gctx);
    Ok(RacingDecision {
        proposed,
        guard: guard_decision,
        reason,
        transitions_used: session.transitions_budget().transitions_used(),
        chunks,
        coverage: cov,
        used_fallback_action: !any_evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stat(i: usize, support: usize, score: f64) -> FmcRootActionStat {
        let space = ActionSpace::discrete(3).unwrap();
        FmcRootActionStat {
            action: Action::discrete_in_space(i, &space).unwrap(),
            support_count: support,
            mean_return: score,
            score,
        }
    }

    fn cfg() -> RacingStopConfig {
        RacingStopConfig::new(48, 144, 0.9, 0.75, 2).unwrap()
    }

    #[test]
    fn racing_stop_config_validation() {
        assert_eq!(
            RacingStopConfig::new(0, 10, 0.9, 0.5, 1),
            Err(RacingStopError::InvalidBudget)
        );
        assert_eq!(
            RacingStopConfig::new(11, 10, 0.9, 0.5, 1),
            Err(RacingStopError::InvalidBudget)
        );
        assert_eq!(
            RacingStopConfig::new(1, 10, 1.0, 0.5, 1),
            Err(RacingStopError::InvalidConfidence)
        );
        assert_eq!(
            RacingStopConfig::new(1, 10, 0.9, 0.0, 1),
            Err(RacingStopError::InvalidCoverage)
        );
        assert_eq!(
            RacingStopConfig::new(1, 10, 0.9, 0.5, 0),
            Err(RacingStopError::InvalidSupport)
        );
    }

    #[test]
    fn racing_stop_rule_dominance_coverage_and_continue() {
        let c = cfg();
        // 22 of 24 walkers under action 0: dominance.
        let dom = [stat(0, 22, 1.0), stat(1, 1, 0.0), stat(2, 1, 0.0)];
        assert_eq!(
            evaluate_stop(&c, &dom, 24, 48, 2, 6, 3),
            Some(StopReason::Dominance)
        );
        // Even split, shallow depth: keep going.
        let even = [stat(0, 8, 0.1), stat(1, 8, 0.0), stat(2, 8, 0.0)];
        assert_eq!(evaluate_stop(&c, &even, 24, 48, 2, 6, 3), None);
        // Even split, deep enough: coverage.
        assert_eq!(
            evaluate_stop(&c, &even, 24, 120, 5, 6, 3),
            Some(StopReason::Coverage)
        );
        // Not enough root support: coverage low, no stop.
        let thin = [stat(0, 2, 0.1), stat(1, 1, 0.0)];
        assert_eq!(evaluate_stop(&c, &thin, 24, 120, 5, 6, 3), None);
        assert_eq!(evaluate_stop(&c, &[], 24, 120, 5, 6, 3), None);
    }

    #[test]
    fn racing_stop_rule_never_below_min_budget_and_fixed_never_stops() {
        let c = cfg();
        let dom = [stat(0, 24, 1.0)];
        assert_eq!(evaluate_stop(&c, &dom, 24, 47, 1, 6, 3), None);
        assert_eq!(
            evaluate_stop(&c, &dom, 24, 48, 2, 6, 3),
            Some(StopReason::Dominance)
        );
        let fixed = RacingStopConfig::fixed(144).unwrap();
        assert!(!fixed.stop_enabled());
        assert_eq!(evaluate_stop(&fixed, &dom, 24, 144, 6, 6, 3), None);
    }

    #[test]
    fn racing_stop_rule_is_order_independent() {
        let c = cfg();
        let a = [stat(0, 22, 1.0), stat(1, 1, 0.0), stat(2, 1, 0.0)];
        let b = [stat(2, 1, 0.0), stat(0, 22, 1.0), stat(1, 1, 0.0)];
        assert_eq!(
            evaluate_stop(&c, &a, 24, 48, 2, 6, 3),
            evaluate_stop(&c, &b, 24, 48, 2, 6, 3)
        );
    }
}
