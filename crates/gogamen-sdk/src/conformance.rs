//! Conformance Kit reutilizable para contratos de Gogamen (`std`-only).
//!
//! # Estabilidad SemVer pre-1.0
//! Este módulo forma parte de `gogamen-sdk` (`0.1.0-alpha`, [`crate::SDK_SEMVER`]).
//! Sus firmas y reportes de diagnóstico pueden evolucionar en versiones `0.x` antes de `1.0.0`
//! según [`crate::API_STABILITY_POLICY`].
//!
//! # Unidades verificadas
//! - **`dt` ([`StepDt`]):** paso temporal físico en **segundos** (`f64 > 0.0`).
//! - **`elapsed` ([`crate::SimDuration`]):** tiempo acumulado en **segundos** (`f64 >= 0.0`).
//! - **`tick` ([`Tick`]):** contador discreto adimensional (`u64`).
//! - **`PlanBudget`:** presupuesto acotado en número máximo de transiciones simuladas (`usize`)
//!   y deadline monotónico opcional (`std::time::Instant`).
//! - **`MasterSeed`:** semilla raíz determinista (`u64`).
//!
//! # Errores
//! Todas las verificaciones devuelven [`ConformanceError`], distinguiendo explícitamente
//! violaciones de determinismo local, contaminación entre ramas `fork`, exceso de presupuesto
//! en planificadores, fallos de rechazo en [`ActionGuard`] y errores subyacentes de dominio,
//! entorno, modelo o snapshot.

use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, BranchableSnapshot, CapabilityCatalog,
    DeterministicRng, DiscreteAction, DomainError, EnvironmentError, GenerativeModel, GuardContext,
    GuardDecision, GuardRejectionReason, LiveEnvironment, MasterSeed, ModelError,
    ModuleContractError, MonotonicDeadline, Observation, OperationBinding, PermissionSet,
    PlanBudget, PlanRequest, PlanStatus, Planner, PlanningContext, SafeFallbackPolicy,
    SimulationSnapshot, SnapshotError, State, StepDt, TemporalProfile, TemporalReplayVerdict,
    TemporalTrace, Terminal, TerminalReason, Tick, TraceHasher, Transition, ValidatedOperation,
    VirtualMonotonicClock, WorkUnits, World,
};

use crate::extensions::{
    ExtensionActivationContext, ExtensionError, ExtensionId, ExtensionKind, StaticExtensionRegistry,
};

/// Configuración estándar para ejecutar el Conformance Kit sobre implementaciones de usuario.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConformanceConfig {
    /// Semilla raíz determinista (`u64`) usada para reproducibilidad local.
    pub seed: MasterSeed,
    /// Paso temporal físico en **segundos** (`> 0.0`) aplicado en las simulaciones.
    pub dt: StepDt,
    /// Número de pasos discretos ejecutados en las pruebas de determinismo.
    pub horizon_steps: usize,
    /// Presupuesto máximo de transiciones simuladas (`usize`) permitido al planificador.
    pub max_transitions_budget: usize,
}

impl ConformanceConfig {
    /// Construye una configuración de conformidad con semilla y paso temporal explícitos.
    pub const fn new(seed: MasterSeed, dt: StepDt) -> Self {
        Self {
            seed,
            dt,
            horizon_steps: 10,
            max_transitions_budget: 8,
        }
    }

    /// Ajusta el horizonte de pasos discretos de verificación.
    pub const fn with_horizon_steps(mut self, horizon_steps: usize) -> Self {
        self.horizon_steps = horizon_steps;
        self
    }

    /// Ajusta el límite máximo de transiciones simuladas del presupuesto de planificación.
    pub const fn with_max_transitions_budget(mut self, max_transitions_budget: usize) -> Self {
        self.max_transitions_budget = max_transitions_budget;
        self
    }
}

/// Error explícito emitido cuando una implementación incumple un contrato de conformidad
/// o falla una operación subyacente durante la verificación.
#[derive(Debug, Clone, PartialEq)]
pub enum ConformanceError {
    /// Dos ejecuciones locales con idéntica [`MasterSeed`] produjeron secuencias divergentes.
    DeterminismMismatch {
        /// Paso discreto (0 para estado/observación inicial, 1..N para transiciones) donde divergió.
        step: usize,
        /// Descripción precisa de la divergencia observada.
        detail: String,
    },
    /// Una bifurcación (`fork`) mutó o contaminó el snapshot padre o el entorno vivo.
    ForkIsolationViolation {
        /// Descripción de la contaminación o fallo de restauración detectado.
        detail: String,
    },
    /// Un [`Planner`] consumió más transiciones de las autorizadas por [`PlanBudget`] o
    /// autorizó una acción sin presupuesto válido.
    BudgetExceeded {
        /// Límite máximo de transiciones permitido por [`PlanBudget::max_transitions`].
        allowed: usize,
        /// Transiciones reportadas como consumidas por [`gogamen_core::PlanResult::transitions_used`].
        used: usize,
    },
    /// [`ActionGuard`] no rechazó una acción prohibida/inválida o no aplicó el respaldo seguro esperado.
    ActionGuardViolation {
        /// Descripción del incumplimiento observado en la guardia.
        detail: String,
    },
    /// Error de contrato de capacidades o catálogo en [`CapabilityCatalog`].
    CapabilityContract(ModuleContractError),
    /// Error de registro o activación de extensiones en [`StaticExtensionRegistry`].
    Extension(ExtensionError),
    /// Violación del contrato temporal (`VirtualMonotonicClock`, `StepDt`, deadlines o traza).
    TemporalViolation {
        /// Descripción de la anomalía temporal observada.
        detail: String,
    },
    /// Violación de separación de permisos o autoridad entre propuesta y guardia/catálogo.
    AuthorityIsolationViolation {
        /// Descripción de la elevación implícita o fuga de autoridad detectada.
        detail: String,
    },
    /// Divergencia durante el replay de efectos pertinentes o traza temporal.
    ReplayMismatch {
        /// Paso donde se detectó la divergencia (0 para estado inicial/longitud).
        step: usize,
        /// Descripción de la divergencia observada.
        detail: String,
    },
    /// Error de dominio propagado desde [`World`] o validación de tipos.
    Domain(DomainError),
    /// Error de entorno vivo propagado desde [`LiveEnvironment`].
    Environment(EnvironmentError),
    /// Error de modelo propagado desde [`GenerativeModel`] o [`Planner`].
    Model(ModelError),
    /// Error de snapshot propagado desde [`BranchableSnapshot`] o [`SimulationSnapshot`].
    Snapshot(SnapshotError),
}

impl fmt::Display for ConformanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DeterminismMismatch { step, detail } => {
                write!(f, "determinism mismatch at step {step}: {detail}")
            }
            Self::ForkIsolationViolation { detail } => {
                write!(f, "fork isolation violation: {detail}")
            }
            Self::BudgetExceeded { allowed, used } => {
                write!(
                    f,
                    "planner budget exceeded: allowed <= {allowed} transitions, used {used}"
                )
            }
            Self::ActionGuardViolation { detail } => {
                write!(f, "action guard violation: {detail}")
            }
            Self::CapabilityContract(err) => {
                write!(f, "capability catalog contract violation: {err}")
            }
            Self::Extension(err) => {
                write!(f, "extension registry violation: {err}")
            }
            Self::TemporalViolation { detail } => {
                write!(f, "temporal contract violation: {detail}")
            }
            Self::AuthorityIsolationViolation { detail } => {
                write!(f, "authority isolation violation: {detail}")
            }
            Self::ReplayMismatch { step, detail } => {
                write!(f, "effect replay mismatch at step {step}: {detail}")
            }
            Self::Domain(err) => write!(f, "domain error during conformance check: {err}"),
            Self::Environment(err) => {
                write!(f, "environment error during conformance check: {err}")
            }
            Self::Model(err) => write!(f, "model error during conformance check: {err}"),
            Self::Snapshot(err) => write!(f, "snapshot error during conformance check: {err}"),
        }
    }
}

impl Error for ConformanceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CapabilityContract(err) => Some(err),
            Self::Extension(err) => Some(err),
            Self::Domain(err) => Some(err),
            Self::Environment(err) => Some(err),
            Self::Model(err) => Some(err),
            Self::Snapshot(err) => Some(err),
            Self::DeterminismMismatch { .. }
            | Self::ForkIsolationViolation { .. }
            | Self::BudgetExceeded { .. }
            | Self::ActionGuardViolation { .. }
            | Self::TemporalViolation { .. }
            | Self::AuthorityIsolationViolation { .. }
            | Self::ReplayMismatch { .. } => None,
        }
    }
}

impl From<ModuleContractError> for ConformanceError {
    fn from(err: ModuleContractError) -> Self {
        Self::CapabilityContract(err)
    }
}

impl From<ExtensionError> for ConformanceError {
    fn from(err: ExtensionError) -> Self {
        Self::Extension(err)
    }
}

impl From<DomainError> for ConformanceError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

impl From<EnvironmentError> for ConformanceError {
    fn from(err: EnvironmentError) -> Self {
        Self::Environment(err)
    }
}

impl From<ModelError> for ConformanceError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<SnapshotError> for ConformanceError {
    fn from(err: SnapshotError) -> Self {
        Self::Snapshot(err)
    }
}

/// Resumen estructurado de los resultados verificados por el Conformance Kit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    /// Semilla usada en la verificación.
    pub seed: MasterSeed,
    /// Huella determinista FNV-1a de la traza ejecutada.
    pub trace_hash: u64,
    /// Número de pasos ejecutados y comparados en determinismo local.
    pub steps_verified: usize,
    /// Indica si se verificó determinismo local.
    pub local_determinism_passed: bool,
    /// Indica si se verificó aislamiento `fork`.
    pub fork_isolation_passed: bool,
    /// Indica si se verificó cumplimiento de [`PlanBudget`].
    pub budget_compliance_passed: bool,
    /// Indica si se verificó rechazo de acción y respaldo seguro en [`ActionGuard`].
    pub action_rejection_passed: bool,
}

impl ConformanceReport {
    /// Devuelve `true` si las cuatro propiedades de conformidad han sido verificadas con éxito.
    pub const fn all_passed(&self) -> bool {
        self.local_determinism_passed
            && self.fork_isolation_passed
            && self.budget_compliance_passed
            && self.action_rejection_passed
    }
}

/// Verifica **determinismo local** para cualquier [`World`] construido a partir de [`MasterSeed`].
///
/// Ejecuta dos episodios independientes con la misma `seed` aplicando la secuencia de `actions`
/// durante `dt` segundos por paso, comprobando igualdad exacta de:
/// - estado inicial (`State`) y observación inicial (`Observation`),
/// - cada `Transition` (`next_state`, `observation`, `terminal`),
/// - huella canónica [`TraceHasher`].
///
/// Devuelve la huella determinista de la traza si ambas ejecuciones coinciden.
pub fn verify_world_determinism<W, F>(
    mut build_world: F,
    seed: MasterSeed,
    dt: StepDt,
    actions: &[Action],
) -> Result<u64, ConformanceError>
where
    W: World,
    F: FnMut(MasterSeed) -> Result<W, DomainError>,
{
    let mut world_a = build_world(seed)?;
    let mut world_b = build_world(seed)?;

    let init_state_a = world_a.reset()?;
    let init_state_b = world_b.reset()?;
    if init_state_a != init_state_b {
        return Err(ConformanceError::DeterminismMismatch {
            step: 0,
            detail: format!("initial states differ: {init_state_a:?} vs {init_state_b:?}"),
        });
    }

    let init_obs_a = world_a.observe(&init_state_a)?;
    let init_obs_b = world_b.observe(&init_state_b)?;
    if init_obs_a != init_obs_b {
        return Err(ConformanceError::DeterminismMismatch {
            step: 0,
            detail: format!("initial observations differ: {init_obs_a:?} vs {init_obs_b:?}"),
        });
    }

    let mut hasher_a = TraceHasher::new();
    let mut hasher_b = TraceHasher::new();
    hasher_a.write_state(&init_state_a);
    hasher_a.write_observation(&init_obs_a);
    hasher_b.write_state(&init_state_b);
    hasher_b.write_observation(&init_obs_b);

    for (idx, action) in actions.iter().enumerate() {
        let step_num = idx + 1;
        let tr_a = world_a.step(action, dt)?;
        let tr_b = world_b.step(action, dt)?;

        if tr_a != tr_b {
            return Err(ConformanceError::DeterminismMismatch {
                step: step_num,
                detail: format!("transitions differ: {tr_a:?} vs {tr_b:?}"),
            });
        }

        hasher_a.write_transition(&tr_a);
        hasher_b.write_transition(&tr_b);

        if tr_a.terminal().is_terminal() {
            break;
        }
    }

    let hash_a = hasher_a.finish();
    let hash_b = hasher_b.finish();
    if hash_a != hash_b {
        return Err(ConformanceError::DeterminismMismatch {
            step: actions.len(),
            detail: format!("trace hashes differ: {hash_a:#x} vs {hash_b:#x}"),
        });
    }

    Ok(hash_a)
}

/// Verifica **determinismo local** para cualquier [`LiveEnvironment`] construido con [`MasterSeed`].
///
/// Compara la observación inicial tras [`LiveEnvironment::reset`] y cada [`gogamen_core::ApplyOutcome`]
/// al aplicar `actions` en dos instancias independientes inicializadas con la misma semilla.
pub fn verify_live_env_determinism<E, F>(
    mut build_env: F,
    seed: MasterSeed,
    actions: &[Action],
) -> Result<u64, ConformanceError>
where
    E: LiveEnvironment,
    F: FnMut(MasterSeed) -> Result<E, EnvironmentError>,
{
    let mut env_a = build_env(seed)?;
    let mut env_b = build_env(seed)?;

    let obs_a = env_a.reset()?;
    let obs_b = env_b.reset()?;
    if obs_a != obs_b {
        return Err(ConformanceError::DeterminismMismatch {
            step: 0,
            detail: format!("initial live observations differ: {obs_a:?} vs {obs_b:?}"),
        });
    }

    let mut hasher = TraceHasher::new();
    hasher.write_observation(&obs_a);

    for (idx, action) in actions.iter().enumerate() {
        let step_num = idx + 1;
        let out_a = env_a.apply(action)?;
        let out_b = env_b.apply(action)?;
        if out_a != out_b {
            return Err(ConformanceError::DeterminismMismatch {
                step: step_num,
                detail: format!("live apply outcomes differ: {out_a:?} vs {out_b:?}"),
            });
        }
        hasher.write_observation(out_a.observation());
        hasher.write_terminal(out_a.terminal());
        if out_a.terminal().is_terminal() {
            break;
        }
    }

    Ok(hasher.finish())
}

/// Verifica **determinismo local** para cualquier [`GenerativeModel`] usando dos
/// [`DeterministicRng`] inicializados con la misma [`MasterSeed`].
pub fn verify_model_determinism<M: GenerativeModel + ?Sized>(
    model: &M,
    initial_state: &State,
    seed: MasterSeed,
    dt: StepDt,
    actions: &[Action],
) -> Result<u64, ConformanceError> {
    let mut rng_a = DeterministicRng::new(seed.as_u64());
    let mut rng_b = DeterministicRng::new(seed.as_u64());

    let mut state_a = initial_state.clone();
    let mut state_b = initial_state.clone();
    let mut hasher = TraceHasher::new();
    hasher.write_state(initial_state);

    for (idx, action) in actions.iter().enumerate() {
        let step_num = idx + 1;
        let tr_a: Transition = model.step(&state_a, action, dt, &mut rng_a)?;
        let tr_b: Transition = model.step(&state_b, action, dt, &mut rng_b)?;
        if tr_a != tr_b {
            return Err(ConformanceError::DeterminismMismatch {
                step: step_num,
                detail: format!("model transitions differ: {tr_a:?} vs {tr_b:?}"),
            });
        }
        hasher.write_transition(&tr_a);
        state_a = tr_a.next_state().clone();
        state_b = tr_b.next_state().clone();
        if tr_a.terminal().is_terminal() {
            break;
        }
    }

    Ok(hasher.finish())
}

/// Verifica **aislamiento fork** sobre cualquier [`BranchableSnapshot`].
///
/// Pasos verificados:
/// 1. Captura `snap_before = target.snapshot()` y verifica su integridad (`verify()`).
/// 2. Crea `forked = target.fork()`.
/// 3. Ejecuta `mutate_fork(&mut forked)` para avanzar o mutar la rama bifurcada.
/// 4. Comprueba que `target.snapshot()` sigue siendo idéntico a `snap_before` (sin contaminación del padre).
/// 5. Comprueba que `forked.restore(&snap_before)` restaura exactamente el snapshot padre en la rama.
pub fn verify_fork_isolation<B, F>(
    target: &B,
    mut mutate_fork: F,
) -> Result<SimulationSnapshot, ConformanceError>
where
    B: BranchableSnapshot,
    F: FnMut(&mut B) -> Result<(), ConformanceError>,
{
    let snap_before = target.snapshot()?;
    snap_before.verify()?;

    let mut forked = target.fork()?;
    let fork_snap_initial = forked.snapshot()?;
    if fork_snap_initial != snap_before {
        return Err(ConformanceError::ForkIsolationViolation {
            detail: "forked snapshot does not match parent snapshot immediately after fork()"
                .to_owned(),
        });
    }

    mutate_fork(&mut forked)?;

    let snap_after = target.snapshot()?;
    if snap_after != snap_before {
        return Err(ConformanceError::ForkIsolationViolation {
            detail: "mutating forked branch contaminated parent snapshot".to_owned(),
        });
    }

    forked.restore(&snap_before)?;
    let restored_snap = forked.snapshot()?;
    if restored_snap != snap_before {
        return Err(ConformanceError::ForkIsolationViolation {
            detail: "restore() on forked branch did not reproduce parent snapshot".to_owned(),
        });
    }

    Ok(snap_after)
}

/// Verifica **aislamiento fork** entre una rama simulada [`BranchableSnapshot`] y un entorno
/// vivo [`LiveEnvironment`], asegurando que avanzar la simulación bifurcada no muta ni
/// contamina [`gogamen_core::EnvironmentObserver::observe`].
pub fn verify_live_env_fork_isolation<E, B, F>(
    live_env: &E,
    branchable: &B,
    mutate_fork: F,
) -> Result<Observation, ConformanceError>
where
    E: LiveEnvironment,
    B: BranchableSnapshot,
    F: FnMut(&mut B) -> Result<(), ConformanceError>,
{
    let live_obs_before = live_env.observe()?;
    verify_fork_isolation(branchable, mutate_fork)?;
    let live_obs_after = live_env.observe()?;

    if live_obs_before != live_obs_after {
        return Err(ConformanceError::ForkIsolationViolation {
            detail: "advancing forked simulation mutated live environment observation".to_owned(),
        });
    }

    Ok(live_obs_after)
}

/// Verifica **cumplimiento de presupuesto** ([`PlanBudget`]) para cualquier [`Planner`] y [`GenerativeModel`].
///
/// Comprueba dos escenarios:
/// 1. Con `max_transitions = budget_transitions`: `result.transitions_used() <= budget_transitions`.
///    Además, si el estado es [`PlanStatus::Cancelled`] o [`PlanStatus::NoValidAction`],
///    `result.authorized_action()` debe ser `None`.
/// 2. Con `max_transitions = 0` y un `fallback` explícito: `result.transitions_used() == 0` y si
///    el planificador reporta [`PlanStatus::BudgetExhausted`], solo puede autorizar `fallback`.
pub fn verify_planner_budget_compliance<P, M>(
    planner: &mut P,
    model: &M,
    initial_state: &State,
    seed: MasterSeed,
    budget_transitions: usize,
    fallback: &Action,
) -> Result<(), ConformanceError>
where
    P: Planner,
    M: GenerativeModel + ?Sized,
{
    let space = model.action_space().clone();
    space.validate(fallback)?;

    // Escenario 1: presupuesto acotado configurado.
    let budget = PlanBudget::with_transitions(budget_transitions);
    let request = PlanRequest::from_state(initial_state.clone(), space.clone(), budget)
        .with_fallback(fallback.clone())?;
    let mut rng = DeterministicRng::new(seed.as_u64());
    let mut ctx = PlanningContext::new(model, &mut rng);
    let result = planner.plan(&mut ctx, &request)?;

    if result.transitions_used() > budget_transitions {
        return Err(ConformanceError::BudgetExceeded {
            allowed: budget_transitions,
            used: result.transitions_used(),
        });
    }

    if matches!(
        result.status(),
        PlanStatus::Cancelled | PlanStatus::NoValidAction
    ) && result.authorized_action().is_some()
    {
        return Err(ConformanceError::BudgetExceeded {
            allowed: budget_transitions,
            used: result.transitions_used(),
        });
    }

    // Escenario 2: presupuesto cero (0 transiciones permitidas).
    let zero_budget = PlanBudget::with_transitions(0);
    let zero_request = PlanRequest::from_state(initial_state.clone(), space, zero_budget)
        .with_fallback(fallback.clone())?;
    let mut rng_zero = DeterministicRng::new(seed.as_u64());
    let mut ctx_zero = PlanningContext::new(model, &mut rng_zero);
    let zero_result = planner.plan(&mut ctx_zero, &zero_request)?;

    if zero_result.transitions_used() > 0 {
        return Err(ConformanceError::BudgetExceeded {
            allowed: 0,
            used: zero_result.transitions_used(),
        });
    }
    if zero_result.status() == PlanStatus::BudgetExhausted
        && zero_result.authorized_action() != Some(fallback)
    {
        return Err(ConformanceError::ActionGuardViolation {
            detail: "zero-budget BudgetExhausted result did not preserve explicit fallback"
                .to_owned(),
        });
    }

    Ok(())
}

/// Construye una acción garantizada como inválida para el [`ActionSpace`] dado.
fn build_invalid_action_for_space(space: &ActionSpace) -> Action {
    match space {
        ActionSpace::Discrete { .. } => {
            // Un vector continuo siempre es inválido en un ActionSpace discreto.
            Action::continuous(vec![0.0]).expect("finite constant")
        }
        ActionSpace::Continuous { .. } => {
            // Una acción discreta siempre es inválida en un ActionSpace continuo.
            Action::Discrete(DiscreteAction::new(0, 1).expect("cardinality 1"))
        }
    }
}

/// Verifica **rechazo de acción** y aplicación de respaldo seguro en [`ActionGuard`]
/// para un [`ActionSpace`] y `safe_fallback` dados.
///
/// Comprueba automáticamente:
/// 1. Propuesta sin autorización (`ActionProposal::unauthorized`) -> `Reject(Unauthorized)` bajo
///    `RejectOnly`, y `Fallback(safe_fallback)` bajo `ExplicitFallback(safe_fallback)`.
/// 2. Acción inválida/incompatible con `action_space` -> `Reject(InvalidAction(_))` bajo
///    `RejectOnly`, y `Fallback(safe_fallback)` bajo `ExplicitFallback(safe_fallback)`.
/// 3. Estado terminal (`Terminal::Terminated`) -> siempre `Reject(TerminalState)`, incluso si hay
///    `ExplicitFallback` configurado.
pub fn verify_action_rejection(
    action_space: &ActionSpace,
    safe_fallback: &Action,
) -> Result<(), ConformanceError> {
    action_space.validate(safe_fallback)?;

    let strict_guard = ActionGuard::strict();
    let fallback_guard =
        ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback.clone()));
    let tick = Tick::ZERO;
    let now = std::time::Instant::now();
    let ongoing_ctx = GuardContext::new(action_space, tick, now, Terminal::Ongoing);

    // 1. Propuesta no autorizada.
    let unauthorized = ActionProposal::unauthorized(tick);
    let strict_unauth = strict_guard.evaluate(&unauthorized, &ongoing_ctx);
    if !matches!(
        strict_unauth,
        GuardDecision::Reject(GuardRejectionReason::Unauthorized)
    ) {
        return Err(ConformanceError::ActionGuardViolation {
            detail: format!(
                "strict guard must reject unauthorized proposal, got {strict_unauth:?}"
            ),
        });
    }
    let fb_unauth = fallback_guard.evaluate(&unauthorized, &ongoing_ctx);
    if fb_unauth != GuardDecision::Fallback(safe_fallback.clone()) {
        return Err(ConformanceError::ActionGuardViolation {
            detail: format!(
                "fallback guard must emit ExplicitFallback for unauthorized proposal, got {fb_unauth:?}"
            ),
        });
    }

    // 2. Acción inválida para el ActionSpace.
    let invalid_action = build_invalid_action_for_space(action_space);
    let invalid_proposal = ActionProposal::authorized(invalid_action, tick);
    let strict_invalid = strict_guard.evaluate(&invalid_proposal, &ongoing_ctx);
    if !matches!(
        strict_invalid,
        GuardDecision::Reject(GuardRejectionReason::InvalidAction(_))
    ) {
        return Err(ConformanceError::ActionGuardViolation {
            detail: format!("strict guard must reject invalid action, got {strict_invalid:?}"),
        });
    }
    let fb_invalid = fallback_guard.evaluate(&invalid_proposal, &ongoing_ctx);
    if fb_invalid != GuardDecision::Fallback(safe_fallback.clone()) {
        return Err(ConformanceError::ActionGuardViolation {
            detail: format!(
                "fallback guard must apply ExplicitFallback on invalid action, got {fb_invalid:?}"
            ),
        });
    }

    // 3. Estado terminal nunca permite actuar ni aplicar fallback.
    let terminal_ctx = GuardContext::new(
        action_space,
        tick,
        now,
        Terminal::Terminated(TerminalReason::HorizonReached),
    );
    let valid_proposal = ActionProposal::authorized(safe_fallback.clone(), tick);
    let terminal_decision = fallback_guard.evaluate(&valid_proposal, &terminal_ctx);
    if terminal_decision != GuardDecision::Reject(GuardRejectionReason::TerminalState) {
        return Err(ConformanceError::ActionGuardViolation {
            detail: format!(
                "guard must reject any action in terminal state, got {terminal_decision:?}"
            ),
        });
    }

    Ok(())
}

/// Ejecuta la suite completa de conformidad (determinismo local, aislamiento `fork`,
/// cumplimiento de presupuesto y rechazo de acción) devolviendo [`ConformanceReport`].
#[allow(clippy::too_many_arguments)]
pub fn run_conformance_suite<W, FW, B, FM, M, P>(
    config: ConformanceConfig,
    build_world: FW,
    actions: &[Action],
    branchable: &B,
    mutate_fork: FM,
    model: &M,
    initial_state: &State,
    planner: &mut P,
    safe_fallback: &Action,
) -> Result<ConformanceReport, ConformanceError>
where
    W: World,
    FW: FnMut(MasterSeed) -> Result<W, DomainError>,
    B: BranchableSnapshot,
    FM: FnMut(&mut B) -> Result<(), ConformanceError>,
    M: GenerativeModel + ?Sized,
    P: Planner,
{
    let bounded_len = actions.len().min(config.horizon_steps);
    let slice = &actions[..bounded_len];

    let trace_hash = verify_world_determinism(build_world, config.seed, config.dt, slice)?;
    verify_fork_isolation(branchable, mutate_fork)?;
    verify_planner_budget_compliance(
        planner,
        model,
        initial_state,
        config.seed,
        config.max_transitions_budget,
        safe_fallback,
    )?;
    verify_action_rejection(model.action_space(), safe_fallback)?;

    Ok(ConformanceReport {
        seed: config.seed,
        trace_hash,
        steps_verified: bounded_len,
        local_determinism_passed: true,
        fork_isolation_passed: true,
        budget_compliance_passed: true,
        action_rejection_passed: true,
    })
}

/// Registro estructurado de efectos y decisiones pertinentes por paso para verificación de replay.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayStepRecord {
    /// Tick lógico en el que se ejecutó el paso.
    pub tick: Tick,
    /// Acción propuesta/ejecutada en el paso.
    pub action: Action,
    /// Decisión emitida por [`ActionGuard`] para la propuesta.
    pub guard_decision: GuardDecision,
    /// Observación resultante tras aplicar el efecto en el entorno/mundo.
    pub observation: Observation,
    /// Estado terminal resultante tras el paso.
    pub terminal: Terminal,
    /// Unidades de trabajo acumuladas hasta este paso.
    pub work_units: WorkUnits,
}

/// Resultado de verificar el replay determinista de efectos pertinentes y traza temporal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayReport {
    /// Semilla usada en ambas ejecuciones replay.
    pub seed: MasterSeed,
    /// Número de pasos comparados.
    pub steps_verified: usize,
    /// Huella canónica [`TraceHasher`] de las observaciones y estados terminales reproducidos.
    pub trace_hash: u64,
    /// Veredicto de comparación temporal por unidades de trabajo.
    pub temporal_verdict: TemporalReplayVerdict,
}

/// Reporte unificado de conformidad para capacidades core usadas desde extensión o consumidor externo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapabilityConformanceReport {
    /// Semilla raíz determinista verificada.
    pub seed: MasterSeed,
    /// Huella canónica de la traza reproducida.
    pub trace_hash: u64,
    /// Número de pasos verificados en replay.
    pub steps_verified: usize,
    /// Número de operaciones validadas en [`CapabilityCatalog`].
    pub validated_operations: usize,
    /// Indica si la validación del catálogo de capacidades tuvo éxito.
    pub catalog_validated: bool,
    /// Indica si la validación temporal (`VirtualMonotonicClock`, `StepDt`, deadlines) tuvo éxito.
    pub temporal_validated: bool,
    /// Indica si la validación de aislamiento (`fork` y separación de autoridad/permisos) tuvo éxito.
    pub isolation_validated: bool,
    /// Indica si la validación de replay de efectos pertinentes tuvo éxito.
    pub replay_validated: bool,
}

impl CapabilityConformanceReport {
    /// Devuelve `true` si los cuatro pilares de conformidad de capacidades pasaron.
    pub const fn all_passed(&self) -> bool {
        self.catalog_validated
            && self.temporal_validated
            && self.isolation_validated
            && self.replay_validated
    }
}

/// Verifica el contrato del catálogo de capacidades [`CapabilityCatalog`] para una ruta
/// con perfil temporal `path_profile` y una lista de `bindings`.
///
/// Comprueba que:
/// 1. Todas las operaciones vinculadas existen, tienen versión compatible, permisos suficientes
///    y perfil temporal compatible con `path_profile` (sin `BlockingIo` en rutas acotadas).
/// 2. Para cualquier operación en el catálogo cuyos permisos requeridos excedan los otorgados
///    al módulo, el catálogo rechaza explícitamente la validación con `PermissionDenied`
///    sin elevación implícita por origen (`Builtin` o `ThirdParty`).
pub fn verify_capability_catalog_contract(
    catalog: &CapabilityCatalog,
    path_profile: TemporalProfile,
    bindings: &[OperationBinding],
) -> Result<Vec<ValidatedOperation>, ConformanceError> {
    let validated = catalog.validate_temporal_path(path_profile, bindings)?;

    if path_profile.is_bounded() {
        for op in &validated {
            if op.operation.declares_blocking_io() || !op.operation.temporal_profile.is_bounded() {
                return Err(ConformanceError::TemporalViolation {
                    detail: format!(
                        "validated operation '{}.{}' violates bounded path profile {:?}",
                        op.module_id, op.operation.id, path_profile
                    ),
                });
            }
        }
    }

    // Verifica que no exista elevación implícita de permisos en ningún módulo registrado.
    for manifest in catalog.modules() {
        for op in &manifest.operations {
            if !op
                .required_permissions
                .is_subset_of(&manifest.granted_permissions)
            {
                let restricted_binding =
                    OperationBinding::new(manifest.id.clone(), op.id.clone(), op.version);
                let res = catalog.validate_operation(&restricted_binding, path_profile);
                if !matches!(res, Err(ModuleContractError::PermissionDenied { .. })) {
                    return Err(ConformanceError::AuthorityIsolationViolation {
                        detail: format!(
                            "operation '{}.{}' with missing permissions was not rejected by CapabilityCatalog",
                            manifest.id, op.id
                        ),
                    });
                }
            }
        }
    }

    Ok(validated)
}

/// Verifica registro, compatibilidad y aislamiento de permisos en [`StaticExtensionRegistry`].
///
/// Para cada `id` en `expected_ids`:
/// - Comprueba que su manifiesto esté registrado y supere `validate_manifest`.
/// - Si requiere permisos, comprueba que [`ExtensionActivationContext::strict_empty`] falle con
///   [`ExtensionError::InsufficientPermissions`] al intentar instanciarlo.
/// - Comprueba que un contexto con `granted_permissions` pueda instanciar la extensión sin errores
///   (cuando `granted_permissions` cubre los permisos requeridos).
pub fn verify_extension_registry_contract(
    registry: &StaticExtensionRegistry,
    expected_ids: &[ExtensionId],
    granted_permissions: &PermissionSet,
) -> Result<(), ConformanceError> {
    let empty_ctx = ExtensionActivationContext::strict_empty();
    let granted_ctx = ExtensionActivationContext::with_permissions(granted_permissions.clone());

    for id in expected_ids {
        let manifest =
            registry
                .manifest(id)
                .ok_or_else(|| ConformanceError::AuthorityIsolationViolation {
                    detail: format!("expected extension '{id}' is not registered"),
                })?;
        if manifest.contract_schema != registry.contract_schema() {
            return Err(ConformanceError::Extension(
                ExtensionError::IncompatibleSchema {
                    id: id.clone(),
                    expected: registry.contract_schema().to_owned(),
                    actual: manifest.contract_schema.clone(),
                },
            ));
        }
        for cap in &manifest.declared_capabilities {
            let Some(supported_ver) = registry.supported_capabilities().get(&cap.id) else {
                return Err(ConformanceError::Extension(
                    ExtensionError::UnsupportedCapability {
                        id: id.clone(),
                        capability: cap.id.clone(),
                    },
                ));
            };
            if !supported_ver.is_compatible_with(&cap.min_version) {
                return Err(ConformanceError::Extension(
                    ExtensionError::IncompatibleVersion {
                        id: id.clone(),
                        capability: cap.id.clone(),
                        required: cap.min_version,
                        available: *supported_ver,
                    },
                ));
            }
        }

        if !manifest.required_permissions.is_empty() {
            let unpriv_err = match manifest.kind {
                ExtensionKind::Adapter => registry.instantiate_adapter(id, &empty_ctx).err(),
                ExtensionKind::Planner => registry.instantiate_planner(id, &empty_ctx).err(),
                ExtensionKind::Sensor => registry.instantiate_sensor(id, &empty_ctx).err(),
                ExtensionKind::PlanKernel => registry.build_kernel_registry(&empty_ctx).err(),
            };
            if !matches!(
                unpriv_err,
                Some(ExtensionError::InsufficientPermissions { .. })
            ) {
                return Err(ConformanceError::AuthorityIsolationViolation {
                    detail: format!(
                        "extension '{id}' did not reject unprivileged activation context"
                    ),
                });
            }
        }

        if manifest
            .required_permissions
            .is_subset_of(granted_permissions)
        {
            match manifest.kind {
                ExtensionKind::Adapter => {
                    registry.instantiate_adapter(id, &granted_ctx)?;
                }
                ExtensionKind::Planner => {
                    registry.instantiate_planner(id, &granted_ctx)?;
                }
                ExtensionKind::Sensor => {
                    registry.instantiate_sensor(id, &granted_ctx)?;
                }
                ExtensionKind::PlanKernel => {
                    registry.build_kernel_registry(&granted_ctx)?;
                }
            }
        }
    }

    Ok(())
}

/// Verifica el contrato temporal sobre [`VirtualMonotonicClock`], [`StepDt`],
/// [`MonotonicDeadline`] y [`TemporalTrace`].
///
/// Avanza un reloj virtual determinista `steps` veces en incrementos `dt`, comprobando:
/// - ausencia de regresiones monotónicas,
/// - proporcionalidad estricta respecto a [`WorkUnits`],
/// - cumplimiento exacto de expiración de [`MonotonicDeadline`] en `deadline_steps`.
pub fn verify_temporal_contract(
    dt: StepDt,
    steps: usize,
    deadline_steps: usize,
) -> Result<TemporalTrace, ConformanceError> {
    if steps == 0 {
        return Err(ConformanceError::TemporalViolation {
            detail: "steps must be >= 1 for temporal contract verification".to_owned(),
        });
    }

    let clock = VirtualMonotonicClock::new();
    let dt_std = std::time::Duration::from_secs_f64(dt.as_secs_f64());
    let deadline_duration =
        dt_std.saturating_mul(u32::try_from(deadline_steps).unwrap_or(u32::MAX));
    let deadline = MonotonicDeadline::from_now(&clock, deadline_duration);

    let mut trace = TemporalTrace::new();
    trace.record_with_clock(Tick::ZERO, WorkUnits::ZERO, &clock);

    for step in 1..=steps {
        if step <= deadline_steps && deadline.is_expired(&clock) {
            return Err(ConformanceError::TemporalViolation {
                detail: format!(
                    "monotonic deadline expired early before step {step}/{deadline_steps}"
                ),
            });
        }
        clock.advance(dt_std);
        trace.record_with_clock(Tick::new(step as u64), WorkUnits::new(step as u64), &clock);
        if step >= deadline_steps && !deadline.is_expired(&clock) {
            return Err(ConformanceError::TemporalViolation {
                detail: format!(
                    "monotonic deadline did not expire at step {step}/{deadline_steps}"
                ),
            });
        }
    }

    if trace.has_monotonic_regression() {
        return Err(ConformanceError::TemporalViolation {
            detail: "virtual monotonic clock exhibited backward regression".to_owned(),
        });
    }
    if !trace.is_strictly_proportional_to_work() {
        return Err(ConformanceError::TemporalViolation {
            detail: "temporal trace is not strictly proportional to work units".to_owned(),
        });
    }

    Ok(trace)
}

/// Verifica **separación de permisos y autoridad**:
/// 1. [`ActionGuard`] rechaza propuestas no autorizadas y aplica respaldo seguro sin elevar autoridad.
/// 2. [`CapabilityCatalog`] rechaza explícitamente `restricted_binding` con `PermissionDenied`
///    cuando el módulo carece de permisos otorgados suficientes.
pub fn verify_authority_isolation(
    action_space: &ActionSpace,
    safe_fallback: &Action,
    catalog: &CapabilityCatalog,
    restricted_binding: &OperationBinding,
    path_profile: TemporalProfile,
) -> Result<(), ConformanceError> {
    verify_action_rejection(action_space, safe_fallback)?;

    let res = catalog.validate_operation(restricted_binding, path_profile);
    if !matches!(res, Err(ModuleContractError::PermissionDenied { .. })) {
        return Err(ConformanceError::AuthorityIsolationViolation {
            detail: format!(
                "expected PermissionDenied for restricted binding '{}.{}', got {res:?}",
                restricted_binding.module_id, restricted_binding.operation_id
            ),
        });
    }

    Ok(())
}

/// Verifica **replay de efectos pertinentes**: ejecuta `run_episode(seed)` dos veces de forma
/// independiente y comprueba igualdad exacta de cada [`ReplayStepRecord`] (acción, decisión de
/// guardia, observación, estado terminal y unidades de trabajo) y coincidencia determinista en
/// [`TemporalTrace::compare_work_replay`].
pub fn verify_effect_replay<F>(
    mut run_episode: F,
    seed: MasterSeed,
) -> Result<ReplayReport, ConformanceError>
where
    F: FnMut(MasterSeed) -> Result<(Vec<ReplayStepRecord>, TemporalTrace), ConformanceError>,
{
    let (records_a, trace_a) = run_episode(seed)?;
    let (records_b, trace_b) = run_episode(seed)?;

    if records_a.len() != records_b.len() {
        return Err(ConformanceError::ReplayMismatch {
            step: 0,
            detail: format!(
                "replay length differs: {} vs {}",
                records_a.len(),
                records_b.len()
            ),
        });
    }

    let mut hasher = TraceHasher::new();
    for (idx, (a, b)) in records_a.iter().zip(records_b.iter()).enumerate() {
        let step = idx + 1;
        if a != b {
            return Err(ConformanceError::ReplayMismatch {
                step,
                detail: format!("pertinent effects diverge: {a:?} vs {b:?}"),
            });
        }
        hasher.write_observation(&a.observation);
        hasher.write_terminal(a.terminal);
    }

    let temporal_verdict = trace_a.compare_work_replay(&trace_b);
    if temporal_verdict != TemporalReplayVerdict::DeterministicMatch {
        return Err(ConformanceError::ReplayMismatch {
            step: records_a.len(),
            detail: format!(
                "temporal replay verdict is not DeterministicMatch: {temporal_verdict:?}"
            ),
        });
    }

    Ok(ReplayReport {
        seed,
        steps_verified: records_a.len(),
        trace_hash: hasher.finish(),
        temporal_verdict,
    })
}

/// Ejecuta la suite completa de conformidad de capacidades core cubriendo:
/// - catálogo de capacidades ([`CapabilityCatalog`]),
/// - temporalidad ([`VirtualMonotonicClock`], [`StepDt`], [`MonotonicDeadline`], [`TemporalProfile`]),
/// - aislamiento (`fork` de [`BranchableSnapshot`] sin mutar [`LiveEnvironment`] + separación de autoridad),
/// - replay de efectos pertinentes.
#[allow(clippy::too_many_arguments)]
pub fn run_capability_conformance_suite<E, B, FM, FR>(
    config: ConformanceConfig,
    catalog: &CapabilityCatalog,
    path_profile: TemporalProfile,
    valid_bindings: &[OperationBinding],
    restricted_binding: &OperationBinding,
    live_env: &E,
    branchable: &B,
    mutate_fork: FM,
    action_space: &ActionSpace,
    safe_fallback: &Action,
    run_episode: FR,
) -> Result<CapabilityConformanceReport, ConformanceError>
where
    E: LiveEnvironment,
    B: BranchableSnapshot,
    FM: FnMut(&mut B) -> Result<(), ConformanceError>,
    FR: FnMut(MasterSeed) -> Result<(Vec<ReplayStepRecord>, TemporalTrace), ConformanceError>,
{
    let validated = verify_capability_catalog_contract(catalog, path_profile, valid_bindings)?;
    let _temporal_trace = verify_temporal_contract(
        config.dt,
        config.horizon_steps.max(1),
        (config.horizon_steps / 2).max(1),
    )?;
    verify_live_env_fork_isolation(live_env, branchable, mutate_fork)?;
    verify_authority_isolation(
        action_space,
        safe_fallback,
        catalog,
        restricted_binding,
        path_profile,
    )?;
    let replay = verify_effect_replay(run_episode, config.seed)?;

    Ok(CapabilityConformanceReport {
        seed: config.seed,
        trace_hash: replay.trace_hash,
        steps_verified: replay.steps_verified,
        validated_operations: validated.len(),
        catalog_validated: true,
        temporal_validated: true,
        isolation_validated: true,
        replay_validated: true,
    })
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;

    use super::*;
    use gogamen_core::RngSource;

    #[derive(Clone)]
    struct TestWorld {
        space: ActionSpace,
        state: State,
        rng: DeterministicRng,
        terminal: Terminal,
    }

    impl TestWorld {
        fn new(seed: MasterSeed) -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
                state: State::new(Tick::ZERO, gogamen_core::SimDuration::ZERO, vec![0.0]).unwrap(),
                rng: DeterministicRng::new(seed.as_u64()),
                terminal: Terminal::Ongoing,
            }
        }
    }

    impl World for TestWorld {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn reset(&mut self) -> Result<State, DomainError> {
            self.state =
                State::new(Tick::ZERO, gogamen_core::SimDuration::ZERO, vec![0.0]).unwrap();
            self.terminal = Terminal::Ongoing;
            Ok(self.state.clone())
        }

        fn observe(&self, state: &State) -> Result<Observation, DomainError> {
            Observation::new(state.tick(), state.elapsed(), state.values().to_vec())
        }

        fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
            if self.terminal.is_terminal() {
                return Err(DomainError::AlreadyTerminal);
            }
            self.space.validate(action)?;
            let idx = action.as_discrete().unwrap().index();
            let delta = if idx == 0 { -1.0 } else { 1.0 };
            let noise = (self.rng.next_f64() - 0.5) * 0.01;
            let next_val = self.state.values()[0] + delta + noise;
            let next_state = self.state.advance(dt, vec![next_val])?;
            let obs = self.observe(&next_state)?;
            let tr = Transition::new(
                self.state.tick(),
                dt,
                next_state.clone(),
                obs,
                Terminal::Ongoing,
            )?;
            self.state = next_state;
            Ok(tr)
        }
    }

    impl GenerativeModel for TestWorld {
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
            self.space
                .validate(action)
                .map_err(ModelError::TransitionFailed)?;
            let idx = action.as_discrete().unwrap().index();
            let delta = if idx == 0 { -1.0 } else { 1.0 };
            let noise = (rng.next_f64() - 0.5) * 0.01;
            let next_val = state.values()[0] + delta + noise;
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

    impl BranchableSnapshot for TestWorld {
        fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
            Ok(SimulationSnapshot::new(
                self.state.clone(),
                &self.rng,
                self.terminal,
            ))
        }

        fn fork(&self) -> Result<Self, SnapshotError> {
            Ok(self.clone())
        }

        fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
            snapshot.verify()?;
            self.state = snapshot.state().clone();
            self.rng = snapshot.rng();
            self.terminal = snapshot.terminal();
            Ok(())
        }
    }

    struct CompliantPlanner;

    impl Planner for CompliantPlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            ctx: &mut PlanningContext<'_, M>,
            request: &PlanRequest,
        ) -> Result<gogamen_core::PlanResult, ModelError> {
            let mut tracker = request.budget().tracker();
            let fallback = request
                .fallback()
                .cloned()
                .unwrap_or_else(|| Action::discrete_in_space(0, request.action_space()).unwrap());
            if !tracker.try_consume_transition() {
                return Ok(gogamen_core::PlanResult::budget_exhausted(
                    request.fallback().cloned(),
                    tracker.transitions_used(),
                ));
            }
            if let Some(state) = request.state() {
                let dt = StepDt::from_secs_f64(0.1).unwrap();
                let _ = ctx.simulate_step(state, &fallback, dt)?;
            }
            Ok(gogamen_core::PlanResult::completed(
                fallback,
                tracker.transitions_used(),
            ))
        }
    }

    struct GreedyOverBudgetPlanner;

    impl Planner for GreedyOverBudgetPlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            _ctx: &mut PlanningContext<'_, M>,
            request: &PlanRequest,
        ) -> Result<gogamen_core::PlanResult, ModelError> {
            let action = Action::discrete_in_space(0, request.action_space()).unwrap();
            Ok(gogamen_core::PlanResult::completed(
                action,
                request.budget().max_transitions() + 5,
            ))
        }
    }

    #[derive(Clone)]
    struct LeakyForkBranch {
        shared_counter: Rc<Cell<u64>>,
        state: State,
        rng: DeterministicRng,
    }

    impl BranchableSnapshot for LeakyForkBranch {
        fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
            let mut vals = self.state.values().to_vec();
            vals[0] = self.shared_counter.get() as f64;
            let s = State::new(self.state.tick(), self.state.elapsed(), vals).unwrap();
            Ok(SimulationSnapshot::new(s, &self.rng, Terminal::Ongoing))
        }

        fn fork(&self) -> Result<Self, SnapshotError> {
            Ok(self.clone())
        }

        fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
            snapshot.verify()?;
            self.shared_counter.set(snapshot.state().values()[0] as u64);
            self.state = snapshot.state().clone();
            self.rng = snapshot.rng();
            Ok(())
        }
    }

    #[test]
    fn conformance_suite_passes_for_compliant_world_and_planner() {
        let seed = MasterSeed::new(1234);
        let dt = StepDt::from_secs_f64(0.05).unwrap();
        let config = ConformanceConfig::new(seed, dt);
        let world = TestWorld::new(seed);
        let fallback = Action::discrete_in_space(0, World::action_space(&world)).unwrap();
        let act_1 = Action::discrete_in_space(1, World::action_space(&world)).unwrap();
        let actions = vec![fallback.clone(), act_1, fallback.clone()];
        let initial_state = world.state.clone();
        let mut planner = CompliantPlanner;

        let report = run_conformance_suite(
            config,
            |s| Ok(TestWorld::new(s)),
            &actions,
            &world,
            |forked| {
                forked.step(&actions[1], dt)?;
                Ok(())
            },
            &world,
            &initial_state,
            &mut planner,
            &fallback,
        )
        .expect("compliant implementation must pass full conformance suite");

        assert!(report.all_passed());
        assert_eq!(report.steps_verified, 3);
    }

    #[test]
    fn conformance_detects_non_deterministic_world() {
        let counter = Cell::new(0u64);
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let space = ActionSpace::discrete(2).unwrap();
        let action = Action::discrete_in_space(0, &space).unwrap();

        let err = verify_world_determinism(
            |_| {
                let c = counter.get();
                counter.set(c + 1);
                let mut w = TestWorld::new(MasterSeed::new(c + 1));
                w.state = State::new(Tick::ZERO, gogamen_core::SimDuration::ZERO, vec![c as f64])
                    .unwrap();
                Ok(w)
            },
            MasterSeed::new(99),
            dt,
            &[action],
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ConformanceError::DeterminismMismatch { step: 1, .. }
        ));
    }

    #[test]
    fn conformance_detects_leaky_fork() {
        let branch = LeakyForkBranch {
            shared_counter: Rc::new(Cell::new(10)),
            state: State::new(Tick::ZERO, gogamen_core::SimDuration::ZERO, vec![10.0]).unwrap(),
            rng: DeterministicRng::new(1),
        };

        let err = verify_fork_isolation(&branch, |forked| {
            forked.shared_counter.set(999);
            Ok(())
        })
        .unwrap_err();

        assert!(matches!(
            err,
            ConformanceError::ForkIsolationViolation { .. }
        ));
    }

    #[test]
    fn conformance_detects_over_budget_planner() {
        let seed = MasterSeed::new(42);
        let world = TestWorld::new(seed);
        let fallback = Action::discrete_in_space(0, World::action_space(&world)).unwrap();
        let mut bad_planner = GreedyOverBudgetPlanner;

        let err = verify_planner_budget_compliance(
            &mut bad_planner,
            &world,
            &world.state,
            seed,
            4,
            &fallback,
        )
        .unwrap_err();

        assert!(matches!(
            err,
            ConformanceError::BudgetExceeded {
                allowed: 4,
                used: 9
            }
        ));
    }

    #[test]
    fn conformance_verifies_catalog_temporal_and_replay_contracts() {
        use gogamen_core::{
            CapabilityVersion, EffectKind, ModuleManifest, ModuleOrigin, OperationSpec, Permission,
        };

        let mut catalog = CapabilityCatalog::new();
        let manifest = ModuleManifest::new(
            "ext.thermostat",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(0, 1, 0),
        )
        .with_granted_permission(Permission::state_read())
        .with_operation(
            OperationSpec::new("step", CapabilityVersion::new(0, 1, 0))
                .with_required_permission(Permission::state_read())
                .with_effect(EffectKind::StateMutation)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        )
        .with_operation(
            OperationSpec::new("blocking_io", CapabilityVersion::new(0, 1, 0))
                .with_required_permission(Permission::state_read())
                .with_effect(EffectKind::BlockingIo)
                .with_temporal_profile(TemporalProfile::BestEffort),
        );
        catalog.register_module(manifest).unwrap();

        let valid_binding =
            OperationBinding::new("ext.thermostat", "step", CapabilityVersion::new(0, 1, 0));
        let validated = verify_capability_catalog_contract(
            &catalog,
            TemporalProfile::BoundedTick,
            &[valid_binding],
        )
        .unwrap();
        assert_eq!(validated.len(), 1);

        let blocking_binding = OperationBinding::new(
            "ext.thermostat",
            "blocking_io",
            CapabilityVersion::new(0, 1, 0),
        );
        let blocking_err = verify_capability_catalog_contract(
            &catalog,
            TemporalProfile::BoundedTick,
            &[blocking_binding],
        )
        .unwrap_err();
        assert!(matches!(
            blocking_err,
            ConformanceError::CapabilityContract(
                ModuleContractError::BlockingIoInBoundedPath { .. }
            )
        ));

        let dt = StepDt::from_secs_f64(0.05).unwrap();
        let trace = verify_temporal_contract(dt, 4, 2).unwrap();
        assert_eq!(trace.len(), 5);
        assert!(!trace.has_monotonic_regression());
        assert!(trace.is_strictly_proportional_to_work());

        let zero_steps_err = verify_temporal_contract(dt, 0, 1).unwrap_err();
        assert!(matches!(
            zero_steps_err,
            ConformanceError::TemporalViolation { .. }
        ));

        let seed = MasterSeed::new(77);
        let space = ActionSpace::discrete(2).unwrap();
        let action = Action::discrete_in_space(1, &space).unwrap();
        let report = verify_effect_replay(
            |s| {
                let mut w = TestWorld::new(s);
                let _ = w.reset()?;
                let clock = VirtualMonotonicClock::new();
                let mut t = TemporalTrace::new();
                t.record_with_clock(Tick::ZERO, WorkUnits::ZERO, &clock);
                let tr = World::step(&mut w, &action, dt)?;
                clock.advance(std::time::Duration::from_millis(50));
                t.record_with_clock(Tick::new(1), WorkUnits::new(1), &clock);
                Ok((
                    vec![ReplayStepRecord {
                        tick: Tick::ZERO,
                        action: action.clone(),
                        guard_decision: GuardDecision::Allow(action.clone()),
                        observation: tr.observation().clone(),
                        terminal: tr.terminal(),
                        work_units: WorkUnits::new(1),
                    }],
                    t,
                ))
            },
            seed,
        )
        .unwrap();
        assert_eq!(report.steps_verified, 1);
        assert_eq!(
            report.temporal_verdict,
            TemporalReplayVerdict::DeterministicMatch
        );

        let counter = Cell::new(0u64);
        let mismatch_err = verify_effect_replay(
            |_| {
                let c = counter.get();
                counter.set(c + 1);
                let mut t = TemporalTrace::new();
                t.record(
                    Tick::ZERO,
                    WorkUnits::new(c),
                    gogamen_core::MonotonicInstant::ZERO,
                );
                Ok((Vec::new(), t))
            },
            seed,
        )
        .unwrap_err();
        assert!(matches!(
            mismatch_err,
            ConformanceError::ReplayMismatch { .. }
        ));
    }
}
