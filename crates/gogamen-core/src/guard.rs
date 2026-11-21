//! Guardia independiente de acción para validar autorización, límites, precondiciones y caducidad (`std`-only).
//!
//! Este módulo implementa la última barrera determinista antes de actuar sobre un
//! [`crate::Actuator`] o [`crate::LiveEnvironment`]:
//! - [`ActionProposal`]: encapsula una acción propuesta con su origen de autorización explícito
//!   (por ejemplo desde [`PlanResult::authorized_action`]), su [`Tick`] de emisión y su
//!   caducidad por edad en ticks y/o deadline monotónico.
//! - [`GuardContext`]: contexto determinista de evaluación en el instante de actuación
//!   ([`ActionSpace`], `current_tick`, `now`, [`Terminal`] y [`Observation`] numérica opcional).
//! - [`DomainPrecondition`]: precondición determinista configurable por dominio.
//! - [`SafeFallbackPolicy`] y [`GuardDecision`]: política explícita de respaldo seguro
//!   (`RejectOnly`, `ExplicitFallback(Action)` o `NoOp`), validando siempre que cualquier
//!   fallback explícito cumpla también [`ActionSpace`] y [`DomainPrecondition`].
//! - **Aislamiento semántico:** ni [`ActionGuard`] ni [`GuardContext`] aceptan ni consultan
//!   etiquetas, metadatos o canales semánticos para autorizar o relajar reglas deterministas.

use std::error::Error;
use std::fmt;
use std::time::{Duration, Instant};

use crate::action::{Action, ActionSpace};
use crate::domain::{Observation, Terminal};
use crate::error::DomainError;
use crate::planner::PlanResult;
use crate::units::Tick;

/// Motivo explícito por el cual [`ActionGuard`] rechaza una [`ActionProposal`] o su respaldo.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardRejectionReason {
    /// La propuesta carece de autorización explícita (por ejemplo `PlanResult::authorized_action() == None`).
    Unauthorized,
    /// El entorno o episodio se encuentra en estado [`Terminal::Terminated`] y no admite actuación.
    TerminalState,
    /// La acción propuesta viola los límites, dimensionalidad, finitud o tipo de [`ActionSpace`].
    InvalidAction(DomainError),
    /// La acción propuesta incumple una precondición determinista del dominio.
    PreconditionFailed {
        /// Motivo determinista reportado por la precondición del dominio.
        reason: &'static str,
    },
    /// La propuesta superó la antigüedad máxima permitida en ticks (`current_tick - proposal_tick > max_age_ticks`).
    StaleProposal {
        /// Tick en el que se emitió la propuesta.
        proposal_tick: Tick,
        /// Tick actual en el instante de evaluación.
        current_tick: Tick,
        /// Antigüedad máxima permitida en ticks.
        max_age_ticks: u64,
    },
    /// El tick de la propuesta es posterior al tick actual del entorno (`proposal_tick > current_tick`).
    FutureProposalTick {
        /// Tick registrado en la propuesta.
        proposal_tick: Tick,
        /// Tick actual del contexto.
        current_tick: Tick,
    },
    /// El deadline monotónico (`expires_at`) de la propuesta fue alcanzado o superado.
    DeadlineExpired,
    /// The proposal's relative TTL overflowed the monotonic clock, so no valid deadline exists (fail-closed).
    DeadlineOverflow,
    /// El fallback explícito configurado en [`SafeFallbackPolicy`] es inválido para el [`ActionSpace`].
    InvalidFallback(DomainError),
    /// El fallback explícito configurado en [`SafeFallbackPolicy`] incumple la precondición del dominio.
    FallbackPreconditionFailed {
        /// Motivo determinista reportado por la precondición del dominio.
        reason: &'static str,
    },
}

impl fmt::Display for GuardRejectionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unauthorized => write!(f, "proposal has no explicit action authorization"),
            Self::TerminalState => {
                write!(f, "cannot act while environment is in a terminal state")
            }
            Self::InvalidAction(err) => write!(f, "proposed action is invalid: {err}"),
            Self::PreconditionFailed { reason } => {
                write!(f, "domain precondition failed: {reason}")
            }
            Self::StaleProposal {
                proposal_tick,
                current_tick,
                max_age_ticks,
            } => write!(
                f,
                "stale action proposal: proposal {proposal_tick}, current {current_tick}, max_age_ticks={max_age_ticks}"
            ),
            Self::FutureProposalTick {
                proposal_tick,
                current_tick,
            } => write!(
                f,
                "inconsistent proposal tick: proposal {proposal_tick} is ahead of current {current_tick}"
            ),
            Self::DeadlineExpired => write!(f, "action proposal monotonic deadline expired"),
            Self::DeadlineOverflow => write!(
                f,
                "action proposal ttl overflowed the monotonic clock; no valid deadline"
            ),
            Self::InvalidFallback(err) => write!(f, "explicit fallback action is invalid: {err}"),
            Self::FallbackPreconditionFailed { reason } => {
                write!(f, "explicit fallback failed domain precondition: {reason}")
            }
        }
    }
}

impl Error for GuardRejectionReason {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAction(err) | Self::InvalidFallback(err) => Some(err),
            Self::Unauthorized
            | Self::TerminalState
            | Self::PreconditionFailed { .. }
            | Self::StaleProposal { .. }
            | Self::FutureProposalTick { .. }
            | Self::DeadlineExpired
            | Self::DeadlineOverflow
            | Self::FallbackPreconditionFailed { .. } => None,
        }
    }
}

/// Propuesta inmutable de acción sometida a verificación por [`ActionGuard`].
///
/// Captura explícitamente si existe una acción autorizada (nunca asume autorización
/// implícita ante ausencia de plan), el [`Tick`] de emisión y las cotas de caducidad.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionProposal {
    authorized_action: Option<Action>,
    proposal_tick: Tick,
    max_age_ticks: Option<u64>,
    expires_at: Option<Instant>,
    /// Set when `with_ttl` overflowed; the guard rejects such proposals (fail-closed).
    ttl_overflow: bool,
}

impl ActionProposal {
    /// Construye una propuesta a partir de un [`PlanResult`] usando exclusivamente
    /// [`PlanResult::authorized_action`].
    pub fn from_plan_result(plan_result: &PlanResult, proposal_tick: Tick) -> Self {
        Self {
            authorized_action: plan_result.authorized_action().cloned(),
            proposal_tick,
            max_age_ticks: None,
            expires_at: None,
            ttl_overflow: false,
        }
    }

    /// Construye una propuesta con una acción explícitamente autorizada en `proposal_tick`.
    pub const fn authorized(action: Action, proposal_tick: Tick) -> Self {
        Self {
            authorized_action: Some(action),
            proposal_tick,
            max_age_ticks: None,
            expires_at: None,
            ttl_overflow: false,
        }
    }

    /// Construye una propuesta explícitamente sin autorización de acción.
    pub const fn unauthorized(proposal_tick: Tick) -> Self {
        Self {
            authorized_action: None,
            proposal_tick,
            max_age_ticks: None,
            expires_at: None,
            ttl_overflow: false,
        }
    }

    /// Configura la antigüedad máxima permitida en ticks respecto al `current_tick` de evaluación.
    pub const fn with_max_age_ticks(mut self, max_age_ticks: u64) -> Self {
        self.max_age_ticks = Some(max_age_ticks);
        self
    }

    /// Configura un deadline monotónico absoluto (`expires_at`) para la propuesta.
    pub const fn with_expires_at(mut self, expires_at: Instant) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Configura un tiempo de vida relativo (`ttl`) a partir del instante `now`.
    ///
    /// Fail-closed (GOG-136): if `now + ttl` overflows, the proposal is marked as
    /// `ttl_overflow` (any previous deadline is kept) and [`ActionGuard::evaluate`]
    /// rejects it with [`GuardRejectionReason::DeadlineOverflow`] instead of treating
    /// the missing deadline as "never expires".
    pub fn with_ttl(mut self, now: Instant, ttl: Duration) -> Self {
        match now.checked_add(ttl) {
            Some(deadline) => {
                self.expires_at = Some(deadline);
                self.ttl_overflow = false;
            }
            None => self.ttl_overflow = true,
        }
        self
    }

    /// Devuelve la acción autorizada si existe.
    pub const fn authorized_action(&self) -> Option<&Action> {
        self.authorized_action.as_ref()
    }

    /// Devuelve el [`Tick`] en el que se generó la propuesta.
    pub const fn proposal_tick(&self) -> Tick {
        self.proposal_tick
    }

    /// Devuelve la antigüedad máxima en ticks configurada, si existe.
    pub const fn max_age_ticks(&self) -> Option<u64> {
        self.max_age_ticks
    }

    /// Indicates whether a relative TTL overflowed in [`ActionProposal::with_ttl`].
    pub const fn ttl_overflowed(&self) -> bool {
        self.ttl_overflow
    }

    /// Devuelve el deadline monotónico configurado, si existe.
    pub const fn expires_at(&self) -> Option<Instant> {
        self.expires_at
    }
}

/// Contexto determinista de evaluación para [`ActionGuard`].
///
/// Contiene únicamente información estructural y temporal del dominio; no expone
/// canales semánticos ni capacidades de actuación.
#[derive(Debug, Clone)]
pub struct GuardContext<'a> {
    action_space: &'a ActionSpace,
    current_tick: Tick,
    now: Instant,
    terminal: Terminal,
    observation: Option<&'a Observation>,
}

impl<'a> GuardContext<'a> {
    /// Crea un contexto determinista de guardia para el tick, instante y estado terminal actuales.
    pub const fn new(
        action_space: &'a ActionSpace,
        current_tick: Tick,
        now: Instant,
        terminal: Terminal,
    ) -> Self {
        Self {
            action_space,
            current_tick,
            now,
            terminal,
            observation: None,
        }
    }

    /// Crea un contexto determinista usando `Instant::now()` como reloj monotónico actual.
    pub fn now(action_space: &'a ActionSpace, current_tick: Tick, terminal: Terminal) -> Self {
        Self::new(action_space, current_tick, Instant::now(), terminal)
    }

    /// Asocia una [`Observation`] numérica del dominio para evaluar precondiciones deterministas.
    pub const fn with_observation(mut self, observation: &'a Observation) -> Self {
        self.observation = Some(observation);
        self
    }

    /// Devuelve el [`ActionSpace`] vigente en el entorno.
    pub const fn action_space(&self) -> &'a ActionSpace {
        self.action_space
    }

    /// Devuelve el [`Tick`] actual del entorno en el momento de decidir.
    pub const fn current_tick(&self) -> Tick {
        self.current_tick
    }

    /// Devuelve el instante monotónico de evaluación.
    pub const fn now_instant(&self) -> Instant {
        self.now
    }

    /// Devuelve el estado terminal actual ([`Terminal`]).
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Devuelve la [`Observation`] numérica asociada, si está presente.
    pub const fn observation(&self) -> Option<&'a Observation> {
        self.observation
    }
}

/// Precondición determinista configurable por dominio antes de autorizar una acción.
///
/// Recibe únicamente la acción candidata y el [`GuardContext`] determinista.
pub trait DomainPrecondition {
    /// Verifica si `action` satisface las precondiciones del dominio en `ctx`.
    fn check(&self, action: &Action, ctx: &GuardContext<'_>) -> Result<(), &'static str>;
}

/// Precondición por defecto que no impone restricciones adicionales al [`ActionSpace`] y [`Terminal`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NoPrecondition;

impl DomainPrecondition for NoPrecondition {
    fn check(&self, _action: &Action, _ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        Ok(())
    }
}

impl<F> DomainPrecondition for F
where
    F: Fn(&Action, &GuardContext<'_>) -> Result<(), &'static str>,
{
    fn check(&self, action: &Action, ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        (self)(action, ctx)
    }
}

/// Política explícita de respaldo seguro aplicada cuando una propuesta es rechazada
/// en un estado no terminal.
#[derive(Debug, Clone, PartialEq)]
pub enum SafeFallbackPolicy {
    /// No sustituye la propuesta rechazada; devuelve [`GuardDecision::Reject`].
    RejectOnly,
    /// Sustituye la propuesta rechazada por una acción de respaldo segura explícita,
    /// validada contra [`ActionSpace`] y [`DomainPrecondition`].
    ExplicitFallback(Action),
    /// Emite una decisión explícita de no-operación ([`GuardDecision::NoOp`]).
    NoOp,
}

/// Resultado explícito e inmutable emitido por [`ActionGuard::evaluate`].
#[derive(Debug, Clone, PartialEq)]
pub enum GuardDecision {
    /// Autoriza ejecutar exactamente la acción propuesta.
    Allow(Action),
    /// Rechaza la propuesta original y autoriza el respaldo seguro explícito del dominio.
    Fallback(Action),
    /// No autoriza ninguna acción y ordena no-operación explícita según política de dominio.
    NoOp,
    /// Rechaza la propuesta con el motivo determinista especificado.
    Reject(GuardRejectionReason),
}

impl GuardDecision {
    /// Devuelve `true` si la decisión es [`GuardDecision::Allow`].
    pub const fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow(_))
    }

    /// Devuelve `true` si la decisión es [`GuardDecision::Fallback`].
    pub const fn is_fallback(&self) -> bool {
        matches!(self, Self::Fallback(_))
    }

    /// Devuelve `true` si la decisión es [`GuardDecision::NoOp`].
    pub const fn is_no_op(&self) -> bool {
        matches!(self, Self::NoOp)
    }

    /// Devuelve `true` si la decisión es [`GuardDecision::Reject`].
    pub const fn is_rejected(&self) -> bool {
        matches!(self, Self::Reject(_))
    }

    /// Devuelve la acción ejecutable autorizada (`Allow` o `Fallback`), o `None` para `NoOp` y `Reject`.
    pub const fn action(&self) -> Option<&Action> {
        match self {
            Self::Allow(action) | Self::Fallback(action) => Some(action),
            Self::NoOp | Self::Reject(_) => None,
        }
    }

    /// Devuelve el motivo de rechazo si la decisión es [`GuardDecision::Reject`].
    pub const fn rejection_reason(&self) -> Option<&GuardRejectionReason> {
        match self {
            Self::Reject(reason) => Some(reason),
            Self::Allow(_) | Self::Fallback(_) | Self::NoOp => None,
        }
    }
}

/// Guardia independiente de acción que valida autorización explícita, límites de
/// [`ActionSpace`], precondiciones de dominio y caducidad antes de actuar.
#[derive(Debug, Clone)]
pub struct ActionGuard<P = NoPrecondition> {
    policy: SafeFallbackPolicy,
    precondition: P,
}

impl ActionGuard<NoPrecondition> {
    /// Crea una guardia con la política de respaldo `policy` sin precondición adicional.
    pub const fn new(policy: SafeFallbackPolicy) -> Self {
        Self {
            policy,
            precondition: NoPrecondition,
        }
    }

    /// Crea una guardia estricta que rechaza cualquier propuesta inválida (`SafeFallbackPolicy::RejectOnly`).
    pub const fn strict() -> Self {
        Self::new(SafeFallbackPolicy::RejectOnly)
    }
}

impl<P: DomainPrecondition> ActionGuard<P> {
    /// Crea una guardia con política de respaldo y una precondición determinista de dominio.
    pub const fn with_precondition(policy: SafeFallbackPolicy, precondition: P) -> Self {
        Self {
            policy,
            precondition,
        }
    }

    /// Devuelve la política de respaldo configurada.
    pub const fn policy(&self) -> &SafeFallbackPolicy {
        &self.policy
    }

    /// Evalúa de forma determinista una [`ActionProposal`] bajo [`GuardContext`].
    ///
    /// Orden de validación:
    /// 1. Estado terminal (`Terminal::Terminated` siempre produce `Reject(TerminalState)`).
    /// 2. Autorización explícita (`proposal.authorized_action()`).
    /// 3. Caducidad por ticks (`proposal_tick` / `max_age_ticks` vs `current_tick`) y deadline monotónico (`expires_at` vs `now`).
    /// 4. Límites e invariantes de [`ActionSpace`].
    /// 5. Precondición determinista del dominio ([`DomainPrecondition`]).
    pub fn evaluate(&self, proposal: &ActionProposal, ctx: &GuardContext<'_>) -> GuardDecision {
        if ctx.terminal().is_terminal() {
            return GuardDecision::Reject(GuardRejectionReason::TerminalState);
        }

        let Some(action) = proposal.authorized_action() else {
            return self.resolve_fallback(GuardRejectionReason::Unauthorized, ctx);
        };

        if proposal.proposal_tick() > ctx.current_tick() {
            return self.resolve_fallback(
                GuardRejectionReason::FutureProposalTick {
                    proposal_tick: proposal.proposal_tick(),
                    current_tick: ctx.current_tick(),
                },
                ctx,
            );
        }

        if let Some(max_age_ticks) = proposal.max_age_ticks() {
            let age = ctx.current_tick().as_u64() - proposal.proposal_tick().as_u64();
            if age > max_age_ticks {
                return self.resolve_fallback(
                    GuardRejectionReason::StaleProposal {
                        proposal_tick: proposal.proposal_tick(),
                        current_tick: ctx.current_tick(),
                        max_age_ticks,
                    },
                    ctx,
                );
            }
        }

        if proposal.ttl_overflowed() {
            return self.resolve_fallback(GuardRejectionReason::DeadlineOverflow, ctx);
        }

        if let Some(expires_at) = proposal.expires_at()
            && ctx.now_instant() >= expires_at
        {
            return self.resolve_fallback(GuardRejectionReason::DeadlineExpired, ctx);
        }

        if let Err(err) = ctx.action_space().validate(action) {
            return self.resolve_fallback(GuardRejectionReason::InvalidAction(err), ctx);
        }

        if let Err(reason) = self.precondition.check(action, ctx) {
            return self.resolve_fallback(GuardRejectionReason::PreconditionFailed { reason }, ctx);
        }

        GuardDecision::Allow(action.clone())
    }

    fn resolve_fallback(
        &self,
        reason: GuardRejectionReason,
        ctx: &GuardContext<'_>,
    ) -> GuardDecision {
        match &self.policy {
            SafeFallbackPolicy::RejectOnly => GuardDecision::Reject(reason),
            SafeFallbackPolicy::NoOp => GuardDecision::NoOp,
            SafeFallbackPolicy::ExplicitFallback(fallback_action) => {
                if let Err(err) = ctx.action_space().validate(fallback_action) {
                    return GuardDecision::Reject(GuardRejectionReason::InvalidFallback(err));
                }
                if let Err(fb_reason) = self.precondition.check(fallback_action, ctx) {
                    return GuardDecision::Reject(
                        GuardRejectionReason::FallbackPreconditionFailed { reason: fb_reason },
                    );
                }
                GuardDecision::Fallback(fallback_action.clone())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::TerminalReason;
    use crate::units::SimDuration;

    #[test]
    fn rejects_invalid_action_unauthorized_terminal_and_stale_proposal_in_fixture() {
        let space = ActionSpace::discrete(2).unwrap();
        let now = Instant::now();
        let guard = ActionGuard::strict();

        // 1. Propuesta sin autorización explícita (por ejemplo PlanStatus::NoValidAction o Cancelled).
        let no_plan = PlanResult::no_valid_action(3);
        let unauthorized_proposal = ActionProposal::from_plan_result(&no_plan, Tick::new(5));
        let ctx = GuardContext::new(&space, Tick::new(5), now, Terminal::Ongoing);
        assert_eq!(
            guard.evaluate(&unauthorized_proposal, &ctx),
            GuardDecision::Reject(GuardRejectionReason::Unauthorized)
        );

        // 2. Acción inválida fuera de ActionSpace.
        let other_space = ActionSpace::discrete(5).unwrap();
        let out_of_bounds_action = Action::discrete_in_space(4, &other_space).unwrap();
        let invalid_proposal = ActionProposal::authorized(out_of_bounds_action, Tick::new(5));
        assert!(matches!(
            guard.evaluate(&invalid_proposal, &ctx),
            GuardDecision::Reject(GuardRejectionReason::InvalidAction(
                DomainError::DiscreteActionOutOfRange {
                    index: 4,
                    cardinality: 2
                }
            ))
        ));

        // 3. Propuesta antigua por edad de ticks (proposal_tick=1, current_tick=5, max_age=2).
        let valid_action = Action::discrete_in_space(1, &space).unwrap();
        let stale_tick_proposal =
            ActionProposal::authorized(valid_action.clone(), Tick::new(1)).with_max_age_ticks(2);
        assert_eq!(
            guard.evaluate(&stale_tick_proposal, &ctx),
            GuardDecision::Reject(GuardRejectionReason::StaleProposal {
                proposal_tick: Tick::new(1),
                current_tick: Tick::new(5),
                max_age_ticks: 2,
            })
        );

        // 4. Propuesta antigua por deadline monotónico expirado.
        let expired_proposal =
            ActionProposal::authorized(valid_action.clone(), Tick::new(5)).with_expires_at(now);
        assert_eq!(
            guard.evaluate(&expired_proposal, &ctx),
            GuardDecision::Reject(GuardRejectionReason::DeadlineExpired)
        );

        // 5. Estado terminal siempre rechaza.
        let terminal_ctx = GuardContext::new(
            &space,
            Tick::new(5),
            now,
            Terminal::Terminated(TerminalReason::GoalReached),
        );
        let fresh_proposal = ActionProposal::authorized(valid_action.clone(), Tick::new(5))
            .with_max_age_ticks(2)
            .with_ttl(now, Duration::from_secs(1));
        assert_eq!(
            guard.evaluate(&fresh_proposal, &terminal_ctx),
            GuardDecision::Reject(GuardRejectionReason::TerminalState)
        );

        // 6. Propuesta válida y vigente es permitida.
        assert_eq!(
            guard.evaluate(&fresh_proposal, &ctx),
            GuardDecision::Allow(valid_action)
        );
    }

    #[test]
    fn gog_136_ttl_overflow_is_rejected_and_valid_ttl_still_expires() {
        let space = ActionSpace::discrete(3).unwrap();
        let action = Action::discrete_in_space(1, &space).unwrap();
        let now = Instant::now();
        let guard = ActionGuard::new(SafeFallbackPolicy::NoOp);
        let ctx = GuardContext::new(&space, Tick::new(5), now, Terminal::Ongoing);

        let overflowed =
            ActionProposal::authorized(action.clone(), Tick::new(5)).with_ttl(now, Duration::MAX);
        assert!(overflowed.ttl_overflowed());
        assert_eq!(overflowed.expires_at(), None);
        let guard_reject = ActionGuard::new(SafeFallbackPolicy::RejectOnly);
        assert_eq!(
            guard_reject.evaluate(&overflowed, &ctx),
            GuardDecision::Reject(GuardRejectionReason::DeadlineOverflow)
        );
        assert!(
            guard.evaluate(&overflowed, &ctx).action().is_none(),
            "overflowed ttl must never allow an action"
        );
        assert!(
            !GuardRejectionReason::DeadlineOverflow
                .to_string()
                .is_empty()
        );

        let valid = ActionProposal::authorized(action.clone(), Tick::new(5))
            .with_ttl(now, Duration::from_secs(10));
        assert!(!valid.ttl_overflowed());
        assert_eq!(guard.evaluate(&valid, &ctx), GuardDecision::Allow(action));
        let later = GuardContext::new(
            &space,
            Tick::new(5),
            now + Duration::from_secs(10),
            Terminal::Ongoing,
        );
        assert_eq!(
            guard_reject.evaluate(&valid, &later),
            GuardDecision::Reject(GuardRejectionReason::DeadlineExpired)
        );
    }

    #[test]
    fn applies_explicit_safe_fallback_or_noop_according_to_domain_policy() {
        let space = ActionSpace::discrete(3).unwrap();
        let safe_idle = Action::discrete_in_space(0, &space).unwrap();
        let risky_move = Action::discrete_in_space(2, &space).unwrap();
        let now = Instant::now();

        // Precondición de dominio: si la observación indica energía < 0.2, solo se permite acción 0.
        let low_battery_obs = Observation::new(Tick::new(4), SimDuration::ZERO, vec![0.1]).unwrap();
        let precondition = |action: &Action, ctx: &GuardContext<'_>| -> Result<(), &'static str> {
            let energy = ctx
                .observation()
                .and_then(|o| o.features().first().copied())
                .unwrap_or(1.0);
            if energy < 0.2 && action.as_discrete().map(|d| d.index()) != Some(0) {
                return Err("low_battery_restricts_to_safe_idle");
            }
            Ok(())
        };

        let ctx_low_battery = GuardContext::new(&space, Tick::new(4), now, Terminal::Ongoing)
            .with_observation(&low_battery_obs);

        // Política 1: ExplicitFallback(safe_idle) -> devuelve Fallback(safe_idle) si el fallback cumple ActionSpace y precondición.
        let guard_with_fallback = ActionGuard::with_precondition(
            SafeFallbackPolicy::ExplicitFallback(safe_idle.clone()),
            precondition,
        );
        let risky_proposal = ActionProposal::authorized(risky_move.clone(), Tick::new(4));
        let decision_fb = guard_with_fallback.evaluate(&risky_proposal, &ctx_low_battery);
        assert_eq!(decision_fb, GuardDecision::Fallback(safe_idle.clone()));
        assert_eq!(decision_fb.action(), Some(&safe_idle));

        // Política 2: NoOp -> devuelve GuardDecision::NoOp ante rechazo en estado no terminal.
        let guard_noop = ActionGuard::with_precondition(SafeFallbackPolicy::NoOp, precondition);
        let decision_noop = guard_noop.evaluate(&risky_proposal, &ctx_low_battery);
        assert_eq!(decision_noop, GuardDecision::NoOp);
        assert_eq!(decision_noop.action(), None);

        // Si el fallback explícito también viola ActionSpace o precondición, se rechaza explícitamente.
        let invalid_fallback_guard = ActionGuard::with_precondition(
            SafeFallbackPolicy::ExplicitFallback(risky_move),
            precondition,
        );
        assert_eq!(
            invalid_fallback_guard.evaluate(&risky_proposal, &ctx_low_battery),
            GuardDecision::Reject(GuardRejectionReason::FallbackPreconditionFailed {
                reason: "low_battery_restricts_to_safe_idle",
            })
        );

        let wrong_space_fallback = Action::continuous(vec![0.0]).unwrap();
        let bad_space_guard =
            ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(wrong_space_fallback));
        let unauth = ActionProposal::unauthorized(Tick::new(4));
        assert!(matches!(
            bad_space_guard.evaluate(&unauth, &ctx_low_battery),
            GuardDecision::Reject(GuardRejectionReason::InvalidFallback(
                DomainError::IncompatibleActionKind { .. }
            ))
        ));
    }

    #[test]
    fn semantic_observations_cannot_alter_or_relax_guard_decisions() {
        /// Canal semántico externo simulado que intenta sugerir relajación o bloqueo.
        #[allow(dead_code)]
        struct ExternalSemanticObservation {
            label: &'static str,
            force_allow_override: bool,
            confidence: f64,
        }

        let space = ActionSpace::discrete(2).unwrap();
        let valid_action = Action::discrete_in_space(1, &space).unwrap();
        let now = Instant::now();
        let guard = ActionGuard::strict();

        let semantic_allow_override = ExternalSemanticObservation {
            label: "OVERRIDE_ALLOW_UNSAFE_STALE_ACTION",
            force_allow_override: true,
            confidence: 1.0,
        };
        let semantic_deny_hint = ExternalSemanticObservation {
            label: "ABORT_EVERYTHING",
            force_allow_override: false,
            confidence: 0.99,
        };

        // Caso A: propuesta caducada no se relaja aunque el canal semántico pida override.
        let stale_proposal =
            ActionProposal::authorized(valid_action.clone(), Tick::new(0)).with_max_age_ticks(1);
        let ctx_stale = GuardContext::new(&space, Tick::new(3), now, Terminal::Ongoing);

        let _ignored_override = &semantic_allow_override;
        let decision_with_override = guard.evaluate(&stale_proposal, &ctx_stale);
        let decision_without_semantic = guard.evaluate(&stale_proposal, &ctx_stale);
        assert_eq!(decision_with_override, decision_without_semantic);
        assert!(decision_with_override.is_rejected());

        // Caso B: propuesta válida y determinista no se altera por etiquetas semánticas adversas.
        let valid_proposal =
            ActionProposal::authorized(valid_action.clone(), Tick::new(3)).with_max_age_ticks(1);
        let _ignored_deny = &semantic_deny_hint;
        assert_eq!(
            guard.evaluate(&valid_proposal, &ctx_stale),
            GuardDecision::Allow(valid_action)
        );
    }
}
