//! Contratos de presupuesto, cancelación, resultado y ejecución segura del planificador (`std`-only).
//!
//! Este módulo define los tipos de frontera entre el ciclo de planificación y la
//! autorización de acciones en tiempo de ejecución:
//! - [`PlanBudget`] y [`BudgetTracker`]: acotan las transiciones simuladas y el tiempo
//!   monotónico máximo de planificación.
//! - [`CancelToken`]: permite solicitar cancelación cooperativa de una planificación.
//! - [`PlanRequest`]: agrupa el objetivo ([`PlanTarget`]), el [`ActionSpace`], el
//!   presupuesto, el token de cancelación y una acción de respaldo (`fallback`) explícita.
//! - [`PlanStatus`] y [`PlanResult`]: distinguen explícitamente entre planificación
//!   completada, presupuesto agotado, cancelación y ausencia de acción válida,
//!   garantizando mediante [`PlanResult::authorized_action`] que la ausencia de plan
//!   **nunca** se interprete como autorización implícita.
//! - [`Planner`]: contrato común de planificadores sobre [`PlanningContext`].

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::action::{Action, ActionSpace};
use crate::domain::{Observation, State};
use crate::error::{DomainError, ModelError};
use crate::model::{GenerativeModel, PlanningContext};

/// Presupuesto de recursos para una invocación del planificador.
///
/// Acota tanto el número máximo de transiciones simuladas (`max_transitions`) como
/// una fecha límite monotónica opcional (`deadline`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanBudget {
    max_transitions: usize,
    deadline: Option<Instant>,
}

impl PlanBudget {
    /// Crea un presupuesto con un máximo de transiciones y un deadline monotónico opcional.
    pub const fn new(max_transitions: usize, deadline: Option<Instant>) -> Self {
        Self {
            max_transitions,
            deadline,
        }
    }

    /// Crea un presupuesto acotado únicamente por número máximo de transiciones.
    pub const fn with_transitions(max_transitions: usize) -> Self {
        Self::new(max_transitions, None)
    }

    /// Crea un presupuesto con máximo de transiciones y un tiempo límite relativo a `Instant::now()`.
    ///
    /// Fail-closed (GOG-136): if `now + timeout` overflows the monotonic clock, the
    /// deadline is set to `now` so the budget is already expired, rather than
    /// dropping the deadline (which would mean "never expires").
    pub fn with_timeout(max_transitions: usize, timeout: Duration) -> Self {
        let now = Instant::now();
        let deadline = now.checked_add(timeout).unwrap_or(now);
        Self::new(max_transitions, Some(deadline))
    }

    /// Devuelve el máximo de transiciones simuladas permitidas.
    pub const fn max_transitions(&self) -> usize {
        self.max_transitions
    }

    /// Devuelve el deadline monotónico opcional, si está configurado.
    pub const fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// Indica si el deadline monotónico se ha alcanzado o superado en `now`.
    pub fn is_deadline_expired_at(&self, now: Instant) -> bool {
        self.deadline.is_some_and(|deadline| now >= deadline)
    }

    /// Indica si el deadline monotónico se ha alcanzado o superado respecto a `Instant::now()`.
    pub fn is_deadline_expired(&self) -> bool {
        self.is_deadline_expired_at(Instant::now())
    }

    /// Inicia un [`BudgetTracker`] para contabilizar el consumo de este presupuesto.
    pub const fn tracker(&self) -> BudgetTracker {
        BudgetTracker::new(self.clone_const())
    }

    const fn clone_const(&self) -> Self {
        Self {
            max_transitions: self.max_transitions,
            deadline: self.deadline,
        }
    }
}

/// Rastreador de consumo de un [`PlanBudget`] durante la planificación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BudgetTracker {
    budget: PlanBudget,
    transitions_used: usize,
}

impl BudgetTracker {
    /// Crea un rastreador con `0` transiciones consumidas para `budget`.
    pub const fn new(budget: PlanBudget) -> Self {
        Self {
            budget,
            transitions_used: 0,
        }
    }

    /// Devuelve el presupuesto asociado a este rastreador.
    pub const fn budget(&self) -> &PlanBudget {
        &self.budget
    }

    /// Devuelve el número de transiciones consumidas hasta el momento.
    pub const fn transitions_used(&self) -> usize {
        self.transitions_used
    }

    /// Devuelve el número de transiciones restantes antes de alcanzar `max_transitions`.
    pub const fn remaining_transitions(&self) -> usize {
        self.budget
            .max_transitions
            .saturating_sub(self.transitions_used)
    }

    /// Indica si se ha alcanzado o superado el límite de transiciones.
    pub const fn is_transitions_exhausted(&self) -> bool {
        self.transitions_used >= self.budget.max_transitions
    }

    /// Indica si el presupuesto (por transiciones o deadline) está agotado en el instante `now`.
    pub fn is_exhausted_at(&self, now: Instant) -> bool {
        self.is_transitions_exhausted() || self.budget.is_deadline_expired_at(now)
    }

    /// Indica si el presupuesto (por transiciones o deadline) está agotado respecto a `Instant::now()`.
    pub fn is_exhausted(&self) -> bool {
        self.is_exhausted_at(Instant::now())
    }

    /// Intenta consumir una transición en el instante `now`.
    ///
    /// Devuelve `true` si la transición fue autorizada y contabilizada; devuelve `false`
    /// si el presupuesto de transiciones o el deadline ya estaban agotados.
    pub fn try_consume_transition_at(&mut self, now: Instant) -> bool {
        if self.is_exhausted_at(now) {
            return false;
        }
        self.transitions_used += 1;
        true
    }

    /// Intenta consumir una transición comprobando el reloj monotónico actual (`Instant::now()`).
    pub fn try_consume_transition(&mut self) -> bool {
        self.try_consume_transition_at(Instant::now())
    }
}

/// Token cooperativo de cancelación compartible entre el coordinador y el planificador.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    /// Crea un nuevo token de cancelación inicialmente activo (no cancelado).
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Crea un token que ya nace en estado cancelado.
    pub fn cancelled() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Señaliza la cancelación de la planificación asociada.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    /// Indica si se ha solicitado la cancelación.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// Estado explícito del resultado de una invocación del planificador.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanStatus {
    /// El planificador completó su búsqueda y emitió un resultado.
    Completed,
    /// El presupuesto se agotó (por límite de transiciones o deadline monotónico).
    BudgetExhausted,
    /// La planificación fue cancelada antes de completarse.
    Cancelled,
    /// No existe ninguna acción válida aplicable para el objetivo dado.
    NoValidAction,
}

impl PlanStatus {
    /// Indica si la planificación finalizó con estado [`PlanStatus::Completed`].
    pub const fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Indica si la planificación finalizó por agotamiento de presupuesto.
    pub const fn is_budget_exhausted(self) -> bool {
        matches!(self, Self::BudgetExhausted)
    }

    /// Indica si la planificación fue cancelada.
    pub const fn is_cancelled(self) -> bool {
        matches!(self, Self::Cancelled)
    }

    /// Indica si no se encontró ninguna acción válida.
    pub const fn is_no_valid_action(self) -> bool {
        matches!(self, Self::NoValidAction)
    }
}

impl fmt::Display for PlanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Completed => write!(f, "completed"),
            Self::BudgetExhausted => write!(f, "budget_exhausted"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::NoValidAction => write!(f, "no_valid_action"),
        }
    }
}

/// Objetivo de entrada sobre el cual planifica el algoritmo (estado y/u observación).
#[derive(Debug, Clone, PartialEq)]
pub enum PlanTarget {
    /// Planificación a partir de un [`State`] completo del modelo.
    State(State),
    /// Planificación a partir de una [`Observation`].
    Observation(Observation),
    /// Planificación disponiendo tanto de [`State`] como de [`Observation`].
    StateAndObservation {
        /// Estado de partida.
        state: State,
        /// Observación asociada.
        observation: Observation,
    },
}

impl PlanTarget {
    /// Devuelve el [`State`] de entrada si está presente.
    pub const fn state(&self) -> Option<&State> {
        match self {
            Self::State(state) | Self::StateAndObservation { state, .. } => Some(state),
            Self::Observation(_) => None,
        }
    }

    /// Devuelve la [`Observation`] de entrada si está presente.
    pub const fn observation(&self) -> Option<&Observation> {
        match self {
            Self::Observation(observation) | Self::StateAndObservation { observation, .. } => {
                Some(observation)
            }
            Self::State(_) => None,
        }
    }
}

/// Petición inmutable de planificación con restricciones explícitas de presupuesto y seguridad.
#[derive(Debug, Clone)]
pub struct PlanRequest {
    target: PlanTarget,
    action_space: ActionSpace,
    budget: PlanBudget,
    cancel_token: CancelToken,
    fallback: Option<Action>,
}

impl PlanRequest {
    /// Construye una petición de planificación validando que `fallback` (si existe)
    /// pertenezca a `action_space`.
    pub fn new(
        target: PlanTarget,
        action_space: ActionSpace,
        budget: PlanBudget,
        cancel_token: CancelToken,
        fallback: Option<Action>,
    ) -> Result<Self, DomainError> {
        if let Some(ref action) = fallback {
            action_space.validate(action)?;
        }
        Ok(Self {
            target,
            action_space,
            budget,
            cancel_token,
            fallback,
        })
    }

    /// Crea una petición a partir de un [`State`] sin cancelación previa ni fallback.
    pub fn from_state(state: State, action_space: ActionSpace, budget: PlanBudget) -> Self {
        Self {
            target: PlanTarget::State(state),
            action_space,
            budget,
            cancel_token: CancelToken::new(),
            fallback: None,
        }
    }

    /// Crea una petición a partir de una [`Observation`] sin cancelación previa ni fallback.
    pub fn from_observation(
        observation: Observation,
        action_space: ActionSpace,
        budget: PlanBudget,
    ) -> Self {
        Self {
            target: PlanTarget::Observation(observation),
            action_space,
            budget,
            cancel_token: CancelToken::new(),
            fallback: None,
        }
    }

    /// Asocia un [`CancelToken`] a la petición.
    pub fn with_cancel_token(mut self, cancel_token: CancelToken) -> Self {
        self.cancel_token = cancel_token;
        self
    }

    /// Asocia una acción de respaldo explícita validándola contra [`Self::action_space`].
    pub fn with_fallback(mut self, fallback: Action) -> Result<Self, DomainError> {
        self.action_space.validate(&fallback)?;
        self.fallback = Some(fallback);
        Ok(self)
    }

    /// Devuelve el objetivo de planificación ([`PlanTarget`]).
    pub const fn target(&self) -> &PlanTarget {
        &self.target
    }

    /// Devuelve el [`State`] objetivo si está presente.
    pub const fn state(&self) -> Option<&State> {
        self.target.state()
    }

    /// Devuelve la [`Observation`] objetivo si está presente.
    pub const fn observation(&self) -> Option<&Observation> {
        self.target.observation()
    }

    /// Devuelve el espacio de acciones admitido para esta petición.
    pub const fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    /// Devuelve el presupuesto de planificación ([`PlanBudget`]).
    pub const fn budget(&self) -> &PlanBudget {
        &self.budget
    }

    /// Devuelve el token de cancelación asociado.
    pub const fn cancel_token(&self) -> &CancelToken {
        &self.cancel_token
    }

    /// Indica si la petición ya ha sido marcada como cancelada.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.is_cancelled()
    }

    /// Devuelve la acción de respaldo explícita si se configuró una.
    pub const fn fallback(&self) -> Option<&Action> {
        self.fallback.as_ref()
    }
}

/// Resultado explícito de una invocación del planificador.
///
/// Garantiza que la ausencia de plan no se interprete nunca como autorización
/// implícita para actuar sobre el entorno.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanResult {
    status: PlanStatus,
    selected_action: Option<Action>,
    fallback_action: Option<Action>,
    transitions_used: usize,
}

impl PlanResult {
    /// Construye un [`PlanResult`] explícito con todos sus campos.
    pub const fn new(
        status: PlanStatus,
        selected_action: Option<Action>,
        fallback_action: Option<Action>,
        transitions_used: usize,
    ) -> Self {
        Self {
            status,
            selected_action,
            fallback_action,
            transitions_used,
        }
    }

    /// Construye un resultado completado con la acción seleccionada.
    pub const fn completed(action: Action, transitions_used: usize) -> Self {
        Self::new(PlanStatus::Completed, Some(action), None, transitions_used)
    }

    /// Construye un resultado por agotamiento de presupuesto con fallback explícito opcional.
    pub const fn budget_exhausted(
        fallback_action: Option<Action>,
        transitions_used: usize,
    ) -> Self {
        Self::new(
            PlanStatus::BudgetExhausted,
            None,
            fallback_action,
            transitions_used,
        )
    }

    /// Construye un resultado cancelado (sin acción autorizada).
    pub const fn cancelled(transitions_used: usize) -> Self {
        Self::new(PlanStatus::Cancelled, None, None, transitions_used)
    }

    /// Construye un resultado sin acción válida (sin acción autorizada).
    pub const fn no_valid_action(transitions_used: usize) -> Self {
        Self::new(PlanStatus::NoValidAction, None, None, transitions_used)
    }

    /// Devuelve el estado del resultado de planificación.
    pub const fn status(&self) -> PlanStatus {
        self.status
    }

    /// Devuelve la acción seleccionada por el planificador, si la hubo.
    pub const fn selected_action(&self) -> Option<&Action> {
        self.selected_action.as_ref()
    }

    /// Devuelve la acción de respaldo explícita asociada al resultado, si existe.
    pub const fn fallback_action(&self) -> Option<&Action> {
        self.fallback_action.as_ref()
    }

    /// Devuelve el número de transiciones simuladas consumidas durante la planificación.
    pub const fn transitions_used(&self) -> usize {
        self.transitions_used
    }

    /// Resuelve la acción autorizada para ejecución en el entorno.
    ///
    /// # Regla de seguridad
    /// **Nunca** interpreta la ausencia de plan como autorización implícita:
    /// - [`PlanStatus::Completed`]: autoriza únicamente [`Self::selected_action`] (si es `Some`).
    /// - [`PlanStatus::BudgetExhausted`]: autoriza únicamente [`Self::fallback_action`] si
    ///   existe un respaldo explícito; si es `None`, devuelve `None`.
    /// - [`PlanStatus::Cancelled`] y [`PlanStatus::NoValidAction`]: siempre devuelven `None`.
    pub const fn authorized_action(&self) -> Option<&Action> {
        match self.status {
            PlanStatus::Completed => self.selected_action.as_ref(),
            PlanStatus::BudgetExhausted => self.fallback_action.as_ref(),
            PlanStatus::Cancelled | PlanStatus::NoValidAction => None,
        }
    }

    /// Alias semántico de [`Self::authorized_action`] para el runtime de ejecución.
    pub const fn resolve_for_execution(&self) -> Option<&Action> {
        self.authorized_action()
    }
}

/// Contrato de un planificador aislado que opera exclusivamente sobre [`PlanningContext`].
pub trait Planner {
    /// Planifica una acción para `request` simulando únicamente a través de `ctx`.
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError>;
}

#[cfg(test)]
mod tests {
    #[test]
    fn gog_136_with_timeout_overflow_is_fail_closed_expired() {
        let budget = PlanBudget::with_timeout(5, Duration::MAX);
        assert!(budget.deadline().is_some());
        assert!(budget.is_deadline_expired());
        // Transition limit is independent of the deadline.
        assert_eq!(budget.max_transitions(), 5);
        let ok = PlanBudget::with_timeout(5, Duration::from_secs(3600));
        assert!(!ok.is_deadline_expired());
    }

    use super::*;
    use crate::action::DiscreteAction;
    use crate::domain::Terminal;
    use crate::domain::Transition;
    use crate::model::{DeterministicRng, RngSource};
    use crate::units::{SimDuration, StepDt, Tick};

    struct LinearTestModel {
        space: ActionSpace,
        reject_all_actions: bool,
    }

    impl LinearTestModel {
        fn new(cardinality: usize) -> Self {
            Self {
                space: ActionSpace::discrete(cardinality).unwrap(),
                reject_all_actions: false,
            }
        }

        fn rejecting_all(cardinality: usize) -> Self {
            Self {
                space: ActionSpace::discrete(cardinality).unwrap(),
                reject_all_actions: true,
            }
        }
    }

    impl GenerativeModel for LinearTestModel {
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
            if self.reject_all_actions {
                return Err(ModelError::InvalidState {
                    reason: "no valid transitions available",
                });
            }
            let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(0) as f64;
            let next_val = state.values().first().copied().unwrap_or(0.0) + idx;
            let next_state = state
                .advance(dt, vec![next_val])
                .map_err(ModelError::TransitionFailed)?;
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

    /// Planificador de fixture que evalúa acciones candidatas respetando estrictamente
    /// `PlanBudget` y `CancelToken`.
    struct GreedyFixturePlanner {
        dt: StepDt,
    }

    impl GreedyFixturePlanner {
        fn new(dt: StepDt) -> Self {
            Self { dt }
        }
    }

    impl Planner for GreedyFixturePlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            ctx: &mut PlanningContext<'_, M>,
            request: &PlanRequest,
        ) -> Result<PlanResult, ModelError> {
            let mut tracker = request.budget().tracker();

            if request.is_cancelled() {
                return Ok(PlanResult::cancelled(tracker.transitions_used()));
            }

            let Some(state) = request.state() else {
                return Ok(PlanResult::no_valid_action(tracker.transitions_used()));
            };

            let candidates: Vec<Action> = match request.action_space() {
                ActionSpace::Discrete { cardinality } => (0..*cardinality)
                    .filter_map(|idx| Action::discrete_in_space(idx, request.action_space()).ok())
                    .collect(),
                ActionSpace::Continuous { bounds } => {
                    let midpoint: Vec<f64> =
                        bounds.iter().map(|b| 0.5 * (b.min() + b.max())).collect();
                    match Action::continuous_in_space(midpoint, request.action_space()) {
                        Ok(act) => vec![act],
                        Err(_) => Vec::new(),
                    }
                }
            };

            let mut best: Option<(f64, Action)> = None;

            for candidate in candidates {
                if request.is_cancelled() {
                    return Ok(PlanResult::cancelled(tracker.transitions_used()));
                }

                if !tracker.try_consume_transition() {
                    return Ok(PlanResult::budget_exhausted(
                        request.fallback().cloned(),
                        tracker.transitions_used(),
                    ));
                }

                if let Ok(transition) = ctx.simulate_step(state, &candidate, self.dt) {
                    let score = transition
                        .next_state()
                        .values()
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                    let replace = match &best {
                        Some((best_score, _)) => score > *best_score,
                        None => true,
                    };
                    if replace {
                        best = Some((score, candidate));
                    }
                }
            }

            match best {
                Some((_, action)) => Ok(PlanResult::completed(action, tracker.transitions_used())),
                None => Ok(PlanResult::no_valid_action(tracker.transitions_used())),
            }
        }
    }

    #[test]
    fn plan_status_distinguishes_completed_exhausted_cancelled_and_no_valid_action() {
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let mut rng = DeterministicRng::new(1);

        // 1. Completed
        let model = LinearTestModel::new(4);
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let mut planner = GreedyFixturePlanner::new(dt);
        let req_ok = PlanRequest::from_state(
            start.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(10),
        );
        let res_ok = planner.plan(&mut ctx, &req_ok).unwrap();
        assert_eq!(res_ok.status(), PlanStatus::Completed);
        assert!(res_ok.status().is_completed());
        assert_eq!(res_ok.transitions_used(), 4);
        assert_eq!(
            res_ok.authorized_action(),
            Some(&Action::discrete_in_space(3, model.action_space()).unwrap())
        );

        // 2. BudgetExhausted
        let req_exhausted = PlanRequest::from_state(
            start.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(2),
        );
        let res_exhausted = planner.plan(&mut ctx, &req_exhausted).unwrap();
        assert_eq!(res_exhausted.status(), PlanStatus::BudgetExhausted);
        assert!(res_exhausted.status().is_budget_exhausted());
        assert_eq!(res_exhausted.transitions_used(), 2);

        // 3. Cancelled
        let token = CancelToken::new();
        token.cancel();
        let req_cancelled = PlanRequest::from_state(
            start.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(10),
        )
        .with_cancel_token(token);
        let res_cancelled = planner.plan(&mut ctx, &req_cancelled).unwrap();
        assert_eq!(res_cancelled.status(), PlanStatus::Cancelled);
        assert!(res_cancelled.status().is_cancelled());
        assert_eq!(res_cancelled.transitions_used(), 0);

        // 4. NoValidAction
        let rejecting_model = LinearTestModel::rejecting_all(3);
        let mut rejecting_ctx = PlanningContext::new(&rejecting_model, &mut rng);
        let req_no_valid = PlanRequest::from_state(
            start,
            rejecting_model.action_space().clone(),
            PlanBudget::with_transitions(10),
        );
        let res_no_valid = planner.plan(&mut rejecting_ctx, &req_no_valid).unwrap();
        assert_eq!(res_no_valid.status(), PlanStatus::NoValidAction);
        assert!(res_no_valid.status().is_no_valid_action());
        assert_eq!(res_no_valid.transitions_used(), 3);
    }

    #[test]
    fn fixture_planner_respects_max_transitions_budget() {
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let model = LinearTestModel::new(6);
        let start = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let mut rng = DeterministicRng::new(99);
        let mut ctx = PlanningContext::new(&model, &mut rng);
        let mut planner = GreedyFixturePlanner::new(dt);

        // Presupuesto 0: no debe consumir ninguna transición.
        let req_zero = PlanRequest::from_state(
            start.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(0),
        );
        let res_zero = planner.plan(&mut ctx, &req_zero).unwrap();
        assert_eq!(res_zero.status(), PlanStatus::BudgetExhausted);
        assert_eq!(res_zero.transitions_used(), 0);

        // Presupuesto 3 (< 6 acciones candidatas): se detiene exactamente en 3.
        let req_three = PlanRequest::from_state(
            start,
            model.action_space().clone(),
            PlanBudget::with_transitions(3),
        );
        let res_three = planner.plan(&mut ctx, &req_three).unwrap();
        assert_eq!(res_three.status(), PlanStatus::BudgetExhausted);
        assert_eq!(res_three.transitions_used(), 3);
    }

    #[test]
    fn budget_tracker_enforces_monotonic_deadline_and_transition_cap() {
        let now = Instant::now();
        let future = now + Duration::from_secs(5);
        let budget = PlanBudget::new(2, Some(future));
        let mut tracker = budget.tracker();

        assert!(!tracker.is_exhausted_at(now));
        assert_eq!(tracker.remaining_transitions(), 2);

        assert!(tracker.try_consume_transition_at(now));
        assert_eq!(tracker.transitions_used(), 1);
        assert_eq!(tracker.remaining_transitions(), 1);

        // Al alcanzar el deadline monotónico, no permite consumir más transiciones.
        assert!(tracker.is_exhausted_at(future));
        assert!(!tracker.try_consume_transition_at(future));
        assert_eq!(tracker.transitions_used(), 1);

        // Con tiempo anterior al deadline, permite la segunda transición y luego se agota.
        assert!(tracker.try_consume_transition_at(now));
        assert_eq!(tracker.transitions_used(), 2);
        assert!(tracker.is_transitions_exhausted());
        assert!(!tracker.try_consume_transition_at(now));
        assert_eq!(tracker.transitions_used(), 2);
    }

    #[test]
    fn runtime_never_interprets_missing_plan_as_implicit_authorization() {
        let space = ActionSpace::discrete(2).unwrap();
        let action_0 = Action::discrete_in_space(0, &space).unwrap();
        let fallback_1 = Action::discrete_in_space(1, &space).unwrap();

        // BudgetExhausted sin fallback explícito -> None.
        let exhausted_no_fb = PlanResult::budget_exhausted(None, 2);
        assert_eq!(exhausted_no_fb.authorized_action(), None);
        assert_eq!(exhausted_no_fb.resolve_for_execution(), None);

        // BudgetExhausted con selected_action accidental pero sin fallback explícito -> None.
        let exhausted_with_rogue_selected =
            PlanResult::new(PlanStatus::BudgetExhausted, Some(action_0.clone()), None, 2);
        assert_eq!(exhausted_with_rogue_selected.authorized_action(), None);

        // BudgetExhausted con fallback explícito -> autoriza únicamente el fallback explícito.
        let exhausted_with_fb = PlanResult::budget_exhausted(Some(fallback_1.clone()), 2);
        assert_eq!(exhausted_with_fb.authorized_action(), Some(&fallback_1));
        assert_eq!(exhausted_with_fb.resolve_for_execution(), Some(&fallback_1));

        // Cancelled -> nunca autoriza, ni aunque hubiera acciones presentes.
        let cancelled = PlanResult::cancelled(1);
        assert_eq!(cancelled.authorized_action(), None);
        let cancelled_with_actions = PlanResult::new(
            PlanStatus::Cancelled,
            Some(action_0.clone()),
            Some(fallback_1.clone()),
            1,
        );
        assert_eq!(cancelled_with_actions.authorized_action(), None);

        // NoValidAction -> nunca autoriza, ni aunque hubiera acciones presentes.
        let no_valid = PlanResult::no_valid_action(3);
        assert_eq!(no_valid.authorized_action(), None);
        let no_valid_with_actions = PlanResult::new(
            PlanStatus::NoValidAction,
            Some(action_0.clone()),
            Some(fallback_1.clone()),
            3,
        );
        assert_eq!(no_valid_with_actions.authorized_action(), None);

        // Completed con selected_action: None -> nunca autoriza, ni usando fallback implícito.
        let completed_empty = PlanResult::new(PlanStatus::Completed, None, Some(fallback_1), 1);
        assert_eq!(completed_empty.authorized_action(), None);

        // Completed con selected_action: Some(action_0) -> autoriza action_0.
        let completed_ok = PlanResult::completed(action_0.clone(), 1);
        assert_eq!(completed_ok.authorized_action(), Some(&action_0));
    }

    #[test]
    fn plan_request_validates_fallback_against_action_space() {
        let discrete_space = ActionSpace::discrete(2).unwrap();
        let continuous_space =
            ActionSpace::continuous(vec![crate::ActionBounds::new(-1.0, 1.0).unwrap()]).unwrap();
        let obs = Observation::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap();
        let continuous_act = Action::continuous(vec![0.5]).unwrap();

        let req =
            PlanRequest::from_observation(obs, discrete_space, PlanBudget::with_transitions(4));
        let err = req.with_fallback(continuous_act).unwrap_err();
        assert!(matches!(err, DomainError::IncompatibleActionKind { .. }));

        let valid_fb = Action::continuous(vec![0.25]).unwrap();
        let req_valid = PlanRequest::from_observation(
            Observation::new(Tick::ZERO, SimDuration::ZERO, vec![1.0]).unwrap(),
            continuous_space,
            PlanBudget::with_transitions(4),
        )
        .with_fallback(valid_fb.clone())
        .unwrap();
        assert_eq!(req_valid.fallback(), Some(&valid_fb));
    }
}
