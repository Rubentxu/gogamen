//! Controlador multirritmo que aísla la ruta periódica de control respecto al worker
//! best-effort de cognición, planificación y sensado lento (`std`-only).
//!
//! # Advertencia sobre frecuencias y tiempo real
//!
//! Las frecuencias y periodos configurables en este módulo (por ejemplo, `1 ms`, `10 ms`
//! o `50 ms` mediante [`ControlPeriod::ONE_MS`], [`ControlPeriod::TEN_MS`] y
//! [`ControlPeriod::FIFTY_MS`]) son **estrictamente ilustrativos y no constituyen
//! garantías de tiempo real duro**. La ejecución sobre `std::thread` y temporizadores del
//! sistema operativo está sujeta a jitter del planificador del SO.
//!
//! # Arquitectura de aislamiento
//!
//! - **Ruta periódica de control** ([`MultirateController`]): ejecuta cada tick periódico
//!   sin esperar ni bloquearse jamás por el worker lento. Lee de forma no bloqueante los
//!   mailboxes acotados ([`BoundedMailbox`], política [`ChannelOverflowPolicy::Latest`]),
//!   verifica vigencia temporal (`MonotonicDeadline`, edad máxima, revisión y `epoch`) y
//!   evalúa toda decisión a través de [`ActionGuard`] y [`gogamen_core::SafeFallbackPolicy`].
//! - **Worker best-effort** ([`BestEffortWorkerHandle`]): ejecuta planificación/sensado
//!   potencialmente lento en un hilo separado y publica resultados o fallos en el mailbox
//!   acotado.
//! - **Watchdog** ([`Watchdog`]): detecta silencio prolongado o fallos explícitos del
//!   worker y activa el respaldo seguro distinguible ([`MultirateFallbackReason`]).
//! - **Traza** ([`MultirateTrace`] / [`MultirateTickRecord`]): registra tiempos
//!   monotónicos reales, latencias, jitter respecto al periodo objetivo, fallbacks
//!   específicos y estadísticas de mailbox y watchdog.

use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, DomainPrecondition, GuardContext,
    GuardDecision, GuardRejectionReason, MonotonicClock, MonotonicDeadline, MonotonicInstant,
    NoPrecondition, Observation, Terminal, Tick,
};

use crate::channels::{
    BoundedChannel, ChannelConfig, ChannelError, ChannelMessage, ChannelOverflowPolicy,
    ChannelStats, SendOutcome,
};

/// Aviso explícito sobre el carácter ilustrativo (no tiempo real duro) de los periodos.
pub const ILLUSTRATIVE_PERIODS_NOTICE: &str = "Los periodos y frecuencias (p. ej. 1 ms, 10 ms, 50 ms) son ilustrativos y no constituyen garantías de tiempo real duro.";

/// Periodo ilustrativo del bucle periódico de control.
///
/// Las cadencias predefinidas (`1 ms`, `10 ms`, `50 ms`) son ilustrativas y no
/// garantizan ejecución en tiempo real duro sobre sistemas operativos de propósito general.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ControlPeriod(Duration);

impl ControlPeriod {
    /// Periodo ilustrativo de `1 ms` (~1000 Hz, no tiempo real duro).
    pub const ONE_MS: Self = Self(Duration::from_millis(1));
    /// Periodo ilustrativo de `10 ms` (~100 Hz, no tiempo real duro).
    pub const TEN_MS: Self = Self(Duration::from_millis(10));
    /// Periodo ilustrativo de `50 ms` (~20 Hz, no tiempo real duro).
    pub const FIFTY_MS: Self = Self(Duration::from_millis(50));

    /// Crea un periodo de control validando que sea estrictamente mayor que cero.
    pub fn from_duration(duration: Duration) -> Result<Self, MultirateError> {
        if duration.is_zero() {
            return Err(MultirateError::InvalidConfig(
                "control_period must be greater than zero",
            ));
        }
        Ok(Self(duration))
    }

    /// Devuelve el periodo como [`Duration`].
    pub const fn as_duration(self) -> Duration {
        self.0
    }

    /// Devuelve el periodo en nanosegundos.
    pub const fn as_nanos(self) -> u128 {
        self.0.as_nanos()
    }
}

impl fmt::Display for ControlPeriod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} (illustrative, non-hard-RT)", self.0)
    }
}

/// Errores de configuración o ciclo de vida del controlador multirritmo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultirateError {
    /// Configuración inválida en parámetros del controlador.
    InvalidConfig(&'static str),
}

impl fmt::Display for MultirateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid multirate configuration: {msg}"),
        }
    }
}

impl Error for MultirateError {}

/// Configuración del controlador multirritmo aislado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultirateConfig {
    /// Periodo objetivo ilustrativo de la ruta crítica de control.
    pub control_period: ControlPeriod,
    /// Edad máxima permitida para una propuesta de planificación/cognición.
    pub proposal_max_age: Duration,
    /// Edad máxima permitida para una observación de sensor.
    pub observation_max_age: Duration,
    /// Tiempo máximo sin latido ni entrega del worker antes de activar el watchdog.
    pub watchdog_timeout: Duration,
    /// Capacidad acotada de cada mailbox (`Latest`).
    pub mailbox_capacity: NonZeroUsize,
    /// Antigüedad máxima opcional en ticks reenviada a [`ActionGuard`].
    pub proposal_max_age_ticks: Option<u64>,
}

impl MultirateConfig {
    /// Crea una configuración base para `control_period` con cotas por defecto seguras.
    pub fn new(control_period: ControlPeriod) -> Self {
        let base = control_period.as_duration();
        Self {
            control_period,
            proposal_max_age: base.saturating_mul(4),
            observation_max_age: base.saturating_mul(4),
            watchdog_timeout: base.saturating_mul(6),
            mailbox_capacity: NonZeroUsize::new(1).expect("1 is non-zero"),
            proposal_max_age_ticks: None,
        }
    }

    /// Configura la edad máxima permitida para propuestas de planificación.
    pub const fn with_proposal_max_age(mut self, max_age: Duration) -> Self {
        self.proposal_max_age = max_age;
        self
    }

    /// Configura la edad máxima permitida para observaciones de sensor.
    pub const fn with_observation_max_age(mut self, max_age: Duration) -> Self {
        self.observation_max_age = max_age;
        self
    }

    /// Configura el timeout del watchdog del worker best-effort.
    pub const fn with_watchdog_timeout(mut self, timeout: Duration) -> Self {
        self.watchdog_timeout = timeout;
        self
    }

    /// Configura la capacidad acotada de los mailboxes (`Latest`).
    pub const fn with_mailbox_capacity(mut self, capacity: NonZeroUsize) -> Self {
        self.mailbox_capacity = capacity;
        self
    }

    /// Configura la antigüedad máxima opcional en ticks para [`ActionGuard`].
    pub const fn with_proposal_max_age_ticks(mut self, max_age_ticks: Option<u64>) -> Self {
        self.proposal_max_age_ticks = max_age_ticks;
        self
    }
}

/// Lectura de sensor con marca monotónica, revisión y época.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorReading {
    /// Observación numérica del entorno.
    pub observation: Observation,
    /// Instante monotónico en que se tomó la observación.
    pub observed_at: MonotonicInstant,
    /// Revisión monotónica de la observación.
    pub revision: u64,
    /// Época de control/episodio a la que pertenece.
    pub epoch: u64,
    /// Expiración explícita opcional.
    pub expires_at: Option<MonotonicInstant>,
}

impl SensorReading {
    /// Crea una nueva lectura de sensor vigente.
    pub const fn new(
        observation: Observation,
        observed_at: MonotonicInstant,
        revision: u64,
        epoch: u64,
    ) -> Self {
        Self {
            observation,
            observed_at,
            revision,
            epoch,
            expires_at: None,
        }
    }

    /// Asocia una expiración monotónica explícita a la lectura.
    pub const fn with_expires_at(mut self, expires_at: MonotonicInstant) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
}

/// Propuesta semántica/de planificación producida por el worker best-effort.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkerProposal {
    /// Acción candidata autorizada por el planificador.
    pub action: Action,
    /// Tick lógico en el que se originó la propuesta.
    pub proposal_tick: Tick,
    /// Revisión monotónica de la propuesta.
    pub revision: u64,
    /// Época de control/episodio para la cual es válida.
    pub epoch: u64,
    /// Revisión de la observación usada para planificar.
    pub observation_revision: u64,
    /// Instante monotónico de la observación usada para planificar.
    pub observation_timestamp: MonotonicInstant,
    /// Instante monotónico en que el worker terminó de producir la propuesta.
    pub produced_at: MonotonicInstant,
    /// Plazo monotónico máximo hasta el cual la propuesta puede ejecutarse.
    pub deadline: MonotonicDeadline,
}

/// Mensaje emitido por el worker best-effort hacia la ruta crítica de control.
#[derive(Debug, Clone, PartialEq)]
pub enum WorkerReport {
    /// Propuesta de acción planificada con metadatos de vigencia.
    Proposal(WorkerProposal),
    /// Fallo explícito del worker best-effort.
    Failed {
        /// Época en la que ocurrió el fallo.
        epoch: u64,
        /// Revisión asociada al intento fallido.
        revision: u64,
        /// Instante monotónico del fallo.
        failed_at: MonotonicInstant,
        /// Descripción breve del fallo.
        reason: String,
    },
    /// Latido de vida del worker sin nueva propuesta todavía.
    Heartbeat {
        /// Época activa del worker.
        epoch: u64,
        /// Instante monotónico del latido.
        timestamp: MonotonicInstant,
    },
}

/// Estadísticas extendidas del mailbox acotado no bloqueante.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct MailboxStats {
    /// Contadores internos de [`BoundedChannel`].
    pub channel: ChannelStats,
    /// Veces que una operación no bloqueante (`try_lock`) encontró el mutex ocupado
    /// (contador atómico `Relaxed` fuera del mutex; una vez por llamada).
    pub nonblocking_contentions: u64,
}

/// Mailbox acotado seguro entre hilos basado en [`BoundedChannel`] (`Latest`).
///
/// La ruta periódica de control usa operaciones estrictamente no bloqueantes
/// (`try_recv_latest_nonblocking`) para no esperar jamás al worker lento.
#[derive(Debug)]
pub struct BoundedMailbox<T> {
    inner: Arc<Mutex<MailboxInner<T>>>,
    /// Contador de contenciones fuera del mutex: debe poder incrementarse cuando
    /// `try_lock` devuelve `WouldBlock`, es decir, sin poseer el lock.
    nonblocking_contentions: Arc<AtomicU64>,
}

#[derive(Debug)]
struct MailboxInner<T> {
    channel: BoundedChannel<T>,
}

impl<T> Clone for BoundedMailbox<T> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            nonblocking_contentions: Arc::clone(&self.nonblocking_contentions),
        }
    }
}

impl<T> BoundedMailbox<T> {
    /// Prepara un mailbox acotado con política [`ChannelOverflowPolicy::Latest`].
    pub fn new(capacity: NonZeroUsize, max_age: Option<Duration>) -> Self {
        let mut cfg = ChannelConfig::new(capacity, ChannelOverflowPolicy::Latest, None);
        if let Some(age) = max_age {
            cfg = cfg.with_max_age(age);
        }
        Self {
            inner: Arc::new(Mutex::new(MailboxInner {
                channel: cfg.prepare(),
            })),
            nonblocking_contentions: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Envía un mensaje desde el productor (worker o sensor).
    pub fn send(
        &self,
        msg: ChannelMessage<T>,
        now: MonotonicInstant,
    ) -> Result<SendOutcome, ChannelError> {
        let mut guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.channel.try_send(msg, now)
    }

    /// Drena el mailbox de forma **estrictamente no bloqueante** (`try_lock`) y devuelve
    /// el último mensaje vigente, si hay alguno.
    ///
    /// Si el lock estuviera momentáneamente ocupado por el productor, retorna `None`
    /// inmediatamente sin esperar y contabiliza la contención.
    pub fn try_recv_latest_nonblocking(&self, now: MonotonicInstant) -> Option<ChannelMessage<T>> {
        let mut guard = match self.inner.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                // No bloquear jamás la ruta de control: solo contabilizar (atómico, sin lock).
                self.nonblocking_contentions.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };
        let mut latest = None;
        while let Some(msg) = guard.channel.try_recv(now) {
            latest = Some(msg);
        }
        latest
    }

    /// Drena todos los mensajes disponibles de forma no bloqueante invocando `try_lock`.
    pub fn try_drain_nonblocking(&self, now: MonotonicInstant) -> Vec<ChannelMessage<T>> {
        let mut guard = match self.inner.try_lock() {
            Ok(g) => g,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => {
                self.nonblocking_contentions.fetch_add(1, Ordering::Relaxed);
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        while let Some(msg) = guard.channel.try_recv(now) {
            out.push(msg);
        }
        out
    }

    /// Devuelve una instantánea de las estadísticas del mailbox sin bloquear si hay contención.
    pub fn stats_nonblocking(&self) -> MailboxStats {
        match self.inner.try_lock() {
            Ok(guard) => MailboxStats {
                channel: guard.channel.stats(),
                nonblocking_contentions: self.nonblocking_contentions.load(Ordering::Relaxed),
            },
            Err(TryLockError::Poisoned(poisoned)) => {
                let guard = poisoned.into_inner();
                MailboxStats {
                    channel: guard.channel.stats(),
                    nonblocking_contentions: self.nonblocking_contentions.load(Ordering::Relaxed),
                }
            }
            Err(TryLockError::WouldBlock) => MailboxStats {
                channel: ChannelStats::default(),
                nonblocking_contentions: self
                    .nonblocking_contentions
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1),
            },
        }
    }

    /// Devuelve estadísticas adquiriendo el lock (para inspección tras finalizar o en tests).
    pub fn stats(&self) -> MailboxStats {
        let guard = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        MailboxStats {
            channel: guard.channel.stats(),
            nonblocking_contentions: self.nonblocking_contentions.load(Ordering::Relaxed),
        }
    }
}

/// Estado observable del watchdog del worker best-effort.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WatchdogHealth {
    /// El worker ha enviado latido o propuesta dentro del timeout configurado.
    Healthy,
    /// Aún no ha pasado el timeout inicial desde el arranque del controlador.
    WarmingUp,
    /// El worker superó el timeout sin latido ni entrega válida.
    Expired,
    /// El worker reportó un fallo explícito.
    Failed,
}

/// Instantánea observable de estado y contadores del [`Watchdog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchdogStatus {
    /// Salud actual del worker según el watchdog.
    pub health: WatchdogHealth,
    /// Último instante monotónico en que se recibió latido o propuesta.
    pub last_heartbeat: Option<MonotonicInstant>,
    /// Total de expiraciones del watchdog contabilizadas.
    pub timeouts_count: u64,
    /// Total de fallos explícitos reportados por el worker.
    pub failures_count: u64,
    /// Total de latidos/entregas recibidos.
    pub heartbeats_count: u64,
    /// Motivo del último fallo explícito, si existe.
    pub last_failure_reason: Option<String>,
}

/// Watchdog monotónico que supervisa la vivacidad y fallos del worker best-effort.
#[derive(Debug, Clone)]
pub struct Watchdog {
    timeout: Duration,
    started_at: MonotonicInstant,
    last_heartbeat: Option<MonotonicInstant>,
    in_timeout_episode: bool,
    failed: bool,
    last_failure_reason: Option<String>,
    timeouts_count: u64,
    failures_count: u64,
    heartbeats_count: u64,
}

impl Watchdog {
    /// Crea un watchdog iniciado en `started_at` con `timeout`.
    pub const fn new(timeout: Duration, started_at: MonotonicInstant) -> Self {
        Self {
            timeout,
            started_at,
            last_heartbeat: None,
            in_timeout_episode: false,
            failed: false,
            last_failure_reason: None,
            timeouts_count: 0,
            failures_count: 0,
            heartbeats_count: 0,
        }
    }

    /// Registra actividad saludable (propuesta o latido) del worker en `at`.
    pub fn record_activity(&mut self, at: MonotonicInstant) {
        self.last_heartbeat = Some(at);
        self.in_timeout_episode = false;
        self.failed = false;
        self.last_failure_reason = None;
        self.heartbeats_count += 1;
    }

    /// Registra un fallo explícito del worker en `at`.
    pub fn record_failure(&mut self, _at: MonotonicInstant, reason: String) {
        self.failed = true;
        self.last_failure_reason = Some(reason);
        self.failures_count += 1;
    }

    /// Evalúa la salud actual del worker en el instante monotónico `now`.
    pub fn evaluate(&mut self, now: MonotonicInstant) -> WatchdogHealth {
        if self.failed {
            return WatchdogHealth::Failed;
        }
        let reference = self.last_heartbeat.unwrap_or(self.started_at);
        let elapsed = now.saturating_duration_since(reference);
        if elapsed > self.timeout {
            if !self.in_timeout_episode {
                self.timeouts_count += 1;
                self.in_timeout_episode = true;
            }
            WatchdogHealth::Expired
        } else if self.last_heartbeat.is_some() {
            WatchdogHealth::Healthy
        } else {
            WatchdogHealth::WarmingUp
        }
    }

    /// Devuelve una instantánea completa del estado del watchdog en `now`.
    pub fn status(&mut self, now: MonotonicInstant) -> WatchdogStatus {
        let health = self.evaluate(now);
        WatchdogStatus {
            health,
            last_heartbeat: self.last_heartbeat,
            timeouts_count: self.timeouts_count,
            failures_count: self.failures_count,
            heartbeats_count: self.heartbeats_count,
            last_failure_reason: self.last_failure_reason.clone(),
        }
    }
}

/// Causa específica y distinguible de activación del respaldo seguro en [`MultirateController`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MultirateFallbackReason {
    /// El worker best-effort aún no ha emitido ninguna propuesta aceptable.
    NoProposalYet,
    /// El worker best-effort notificó un fallo explícito.
    WorkerFailed,
    /// El watchdog expiró por falta de latido o entrega dentro del timeout.
    WatchdogExpired,
    /// La propuesta de planificación está obsoleta (edad máxima excedida, revisión o época antigua).
    StalePlan,
    /// La observación de sensor asociada está obsoleta (edad máxima excedida, expirada o época antigua).
    StaleObservation,
    /// El plazo monotónico (`MonotonicDeadline`) de la propuesta venció.
    DeadlineExpired,
    /// [`ActionGuard`] rechazó la acción propuesta (límites de espacio de acción o precondición).
    GuardRejected,
}

impl fmt::Display for MultirateFallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoProposalYet => write!(f, "no valid proposal received from worker yet"),
            Self::WorkerFailed => write!(f, "best-effort worker reported explicit failure"),
            Self::WatchdogExpired => write!(f, "worker watchdog timeout expired"),
            Self::StalePlan => write!(
                f,
                "planning proposal is stale or from an obsolete epoch/revision"
            ),
            Self::StaleObservation => {
                write!(
                    f,
                    "sensor observation is stale, expired, or from an obsolete epoch"
                )
            }
            Self::DeadlineExpired => write!(f, "proposal monotonic deadline expired"),
            Self::GuardRejected => write!(f, "action guard rejected proposed action"),
        }
    }
}

/// Registro por tick de la ruta periódica de control en [`MultirateTrace`].
#[derive(Debug, Clone, PartialEq)]
pub struct MultirateTickRecord {
    /// Tick de control ejecutado.
    pub tick: Tick,
    /// Época de control activa en este tick.
    pub epoch: u64,
    /// Instante monotónico objetivo del inicio de este tick.
    pub target_at: MonotonicInstant,
    /// Instante monotónico real de inicio del tick.
    pub started_at: MonotonicInstant,
    /// Instante monotónico real de finalización del tick.
    pub finished_at: MonotonicInstant,
    /// Jitter firmado en nanosegundos (`started_at - target_at`).
    pub jitter_nanos: i128,
    /// Latencia real del paso de control (`finished_at - started_at`).
    pub control_step_latency: Duration,
    /// Revisión de la propuesta evaluada en este tick, si existe.
    pub proposal_revision: Option<u64>,
    /// Revisión de la observación vigente en este tick, si existe.
    pub observation_revision: Option<u64>,
    /// Edad de la propuesta respecto a `started_at`, si existe.
    pub proposal_age: Option<Duration>,
    /// Edad de la observación respecto a `started_at`, si existe.
    pub observation_age: Option<Duration>,
    /// Indica si este tick activó el respaldo seguro (o rechazo de guardia).
    pub used_fallback: bool,
    /// Motivo específico y distinguible del respaldo seguro, si se activó.
    pub fallback_reason: Option<MultirateFallbackReason>,
    /// Decisión final emitida por [`ActionGuard`].
    pub guard_decision: GuardDecision,
    /// Estadísticas del mailbox de observaciones en este tick.
    pub observation_mailbox_stats: MailboxStats,
    /// Estadísticas del mailbox de propuestas/reportes en este tick.
    pub proposal_mailbox_stats: MailboxStats,
    /// Estado del watchdog en este tick.
    pub watchdog_status: WatchdogStatus,
}

/// Traza completa de ejecución del controlador multirritmo.
#[derive(Debug, Clone, PartialEq)]
pub struct MultirateTrace {
    control_period: ControlPeriod,
    timing_guarantee_note: &'static str,
    records: Vec<MultirateTickRecord>,
}

impl MultirateTrace {
    /// Crea una traza vacía para `control_period` con la nota explícita de no tiempo real duro.
    pub const fn new(control_period: ControlPeriod) -> Self {
        Self {
            control_period,
            timing_guarantee_note: ILLUSTRATIVE_PERIODS_NOTICE,
            records: Vec::new(),
        }
    }

    /// Periodo ilustrativo configurado para la traza.
    pub const fn control_period(&self) -> ControlPeriod {
        self.control_period
    }

    /// Nota explícita indicando que las frecuencias son ilustrativas y no tiempo real duro.
    pub const fn timing_guarantee_note(&self) -> &'static str {
        self.timing_guarantee_note
    }

    /// Registros por tick almacenados en la traza.
    pub fn records(&self) -> &[MultirateTickRecord] {
        &self.records
    }

    /// Número de ticks registrados.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Indica si la traza está vacía.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Número total de ticks que activaron respaldo seguro / rechazo.
    pub fn fallback_count(&self) -> usize {
        self.records.iter().filter(|r| r.used_fallback).count()
    }

    /// Número de ticks con un [`MultirateFallbackReason`] específico.
    pub fn count_by_fallback_reason(&self, reason: MultirateFallbackReason) -> usize {
        self.records
            .iter()
            .filter(|r| r.fallback_reason == Some(reason))
            .count()
    }

    /// Jitter absoluto máximo observado en nanosegundos.
    pub fn max_abs_jitter_nanos(&self) -> u128 {
        self.records
            .iter()
            .map(|r| r.jitter_nanos.unsigned_abs())
            .max()
            .unwrap_or(0)
    }

    /// Jitter absoluto medio en nanosegundos.
    pub fn mean_abs_jitter_nanos(&self) -> f64 {
        if self.records.is_empty() {
            return 0.0;
        }
        let total: u128 = self
            .records
            .iter()
            .map(|r| r.jitter_nanos.unsigned_abs())
            .sum();
        total as f64 / self.records.len() as f64
    }

    /// Latencia máxima del paso periódico de control.
    pub fn max_control_latency(&self) -> Duration {
        self.records
            .iter()
            .map(|r| r.control_step_latency)
            .max()
            .unwrap_or(Duration::ZERO)
    }

    fn push(&mut self, record: MultirateTickRecord) {
        self.records.push(record);
    }
}

/// Controlador multirritmo que desacopla el bucle periódico de control del worker lento.
pub struct MultirateController<P = NoPrecondition> {
    config: MultirateConfig,
    action_space: ActionSpace,
    guard: ActionGuard<P>,
    clock: Arc<dyn MonotonicClock>,
    observation_mailbox: BoundedMailbox<SensorReading>,
    proposal_mailbox: BoundedMailbox<WorkerReport>,
    watchdog: Watchdog,
    current_epoch: u64,
    current_tick: Tick,
    next_target_at: Option<MonotonicInstant>,
    latest_observation: Option<SensorReading>,
    latest_observation_revision: Option<u64>,
    latest_proposal: Option<WorkerProposal>,
    latest_proposal_revision: Option<u64>,
    worker_failed_in_epoch: bool,
    trace: MultirateTrace,
}

impl<P: DomainPrecondition> MultirateController<P> {
    /// Crea un controlador multirritmo con reloj monotónico, guardia de acción y mailboxes acotados.
    pub fn new(
        config: MultirateConfig,
        action_space: ActionSpace,
        guard: ActionGuard<P>,
        clock: Arc<dyn MonotonicClock>,
    ) -> Self {
        let now = clock.now();
        // No filtramos por max_age dentro del BoundedChannel de reportes para que el controlador
        // pueda distinguir explícitamente entre StalePlan, StaleObservation y NoProposalYet.
        let observation_mailbox = BoundedMailbox::new(config.mailbox_capacity, None);
        let proposal_mailbox = BoundedMailbox::new(config.mailbox_capacity, None);
        let watchdog = Watchdog::new(config.watchdog_timeout, now);
        let trace = MultirateTrace::new(config.control_period);
        Self {
            config,
            action_space,
            guard,
            clock,
            observation_mailbox,
            proposal_mailbox,
            watchdog,
            current_epoch: 1,
            current_tick: Tick::ZERO,
            next_target_at: None,
            latest_observation: None,
            latest_observation_revision: None,
            latest_proposal: None,
            latest_proposal_revision: None,
            worker_failed_in_epoch: false,
            trace,
        }
    }

    /// Devuelve la configuración activa del controlador.
    pub const fn config(&self) -> MultirateConfig {
        self.config
    }

    /// Devuelve la época de control actual.
    pub const fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Avanza a una nueva época e invalida propuestas/fallos de épocas anteriores.
    pub fn advance_epoch(&mut self) -> u64 {
        self.current_epoch += 1;
        self.worker_failed_in_epoch = false;
        self.current_epoch
    }

    /// Devuelve un handle clonable al mailbox acotado de observaciones de sensor.
    pub fn observation_mailbox(&self) -> BoundedMailbox<SensorReading> {
        self.observation_mailbox.clone()
    }

    /// Devuelve un handle clonable al mailbox acotado de reportes/propuestas del worker.
    pub fn proposal_mailbox(&self) -> BoundedMailbox<WorkerReport> {
        self.proposal_mailbox.clone()
    }

    /// Envía una lectura de sensor al mailbox acotado usando el reloj del controlador.
    pub fn submit_observation(&self, reading: SensorReading) -> Result<SendOutcome, ChannelError> {
        let now = self.clock.now();
        let revision = reading.revision;
        let timestamp = reading.observed_at;
        let expires_at = reading.expires_at;
        let mut msg = ChannelMessage::new(reading, timestamp, revision);
        if let Some(exp) = expires_at {
            msg = msg.with_expires_at(exp);
        }
        self.observation_mailbox.send(msg, now)
    }

    /// Envía una propuesta de planificación al mailbox acotado usando el reloj del controlador.
    pub fn submit_proposal(&self, proposal: WorkerProposal) -> Result<SendOutcome, ChannelError> {
        let now = self.clock.now();
        let revision = proposal.revision;
        let timestamp = proposal.produced_at;
        let msg = ChannelMessage::new(WorkerReport::Proposal(proposal), timestamp, revision);
        self.proposal_mailbox.send(msg, now)
    }

    /// Envía un fallo explícito del worker al mailbox acotado.
    pub fn submit_worker_failure(
        &self,
        epoch: u64,
        revision: u64,
        reason: impl Into<String>,
    ) -> Result<SendOutcome, ChannelError> {
        let now = self.clock.now();
        let report = WorkerReport::Failed {
            epoch,
            revision,
            failed_at: now,
            reason: reason.into(),
        };
        let msg = ChannelMessage::new(report, now, revision);
        self.proposal_mailbox.send(msg, now)
    }

    /// Ejecuta **un tick de la ruta periódica de control** sin bloquearse ni esperar al worker.
    pub fn step_control_tick(&mut self, terminal: Terminal) -> MultirateTickRecord {
        let started_at = self.clock.now();
        self.current_tick = self.current_tick.next().unwrap_or(self.current_tick);

        let target_at = self.next_target_at.unwrap_or(started_at);
        let jitter_nanos = (started_at.as_nanos() as i128) - (target_at.as_nanos() as i128);
        self.next_target_at =
            Some(target_at.saturating_add(self.config.control_period.as_duration()));

        // 1. Drenaje no bloqueante del mailbox de observaciones.
        for msg in self.observation_mailbox.try_drain_nonblocking(started_at) {
            let reading = msg.payload;
            let is_newer = self
                .latest_observation_revision
                .is_none_or(|prev| reading.revision >= prev);
            if is_newer {
                self.latest_observation_revision = Some(reading.revision);
                self.latest_observation = Some(reading);
            }
        }

        // 2. Drenaje no bloqueante del mailbox de reportes del worker.
        for msg in self.proposal_mailbox.try_drain_nonblocking(started_at) {
            match msg.payload {
                WorkerReport::Proposal(proposal) => {
                    self.watchdog.record_activity(started_at);
                    let is_newer_rev = self
                        .latest_proposal_revision
                        .is_none_or(|prev| proposal.revision >= prev);
                    if is_newer_rev {
                        self.latest_proposal_revision = Some(proposal.revision);
                        if proposal.epoch == self.current_epoch {
                            self.worker_failed_in_epoch = false;
                        }
                        self.latest_proposal = Some(proposal);
                    }
                }
                WorkerReport::Failed {
                    epoch,
                    failed_at,
                    reason,
                    ..
                } => {
                    self.watchdog.record_failure(failed_at, reason);
                    if epoch == self.current_epoch {
                        self.worker_failed_in_epoch = true;
                        self.latest_proposal = None;
                    }
                }
                WorkerReport::Heartbeat { timestamp, .. } => {
                    self.watchdog.record_activity(timestamp);
                }
            }
        }

        let watchdog_status = self.watchdog.status(started_at);

        // 3. Validación de vigencia (worker health, epoch, staleness, deadline) y guardia.
        let (guard_decision, fallback_reason, proposal_revision, proposal_age, obs_age) =
            self.evaluate_tick(started_at, &watchdog_status, terminal);

        let used_fallback = fallback_reason.is_some()
            || guard_decision.is_fallback()
            || guard_decision.is_no_op()
            || guard_decision.is_rejected();

        let finished_at = self.clock.now();
        let control_step_latency = finished_at.saturating_duration_since(started_at);

        let record = MultirateTickRecord {
            tick: self.current_tick,
            epoch: self.current_epoch,
            target_at,
            started_at,
            finished_at,
            jitter_nanos,
            control_step_latency,
            proposal_revision,
            observation_revision: self.latest_observation.as_ref().map(|o| o.revision),
            proposal_age,
            observation_age: obs_age,
            used_fallback,
            fallback_reason,
            guard_decision,
            observation_mailbox_stats: self.observation_mailbox.stats_nonblocking(),
            proposal_mailbox_stats: self.proposal_mailbox.stats_nonblocking(),
            watchdog_status,
        };
        self.trace.push(record.clone());
        record
    }

    /// Ejecuta `ticks` pasos periódicos usando reloj real (`std::thread::sleep` entre periodos).
    ///
    /// Las pausas entre ticks aproximan [`MultirateConfig::control_period`] de forma
    /// ilustrativa y no ofrecen garantías de tiempo real duro.
    pub fn run_periodic_loop(
        &mut self,
        ticks: usize,
        terminal: Terminal,
    ) -> Vec<MultirateTickRecord> {
        let mut records = Vec::with_capacity(ticks);
        for idx in 0..ticks {
            let record = self.step_control_tick(terminal);
            records.push(record);
            if idx + 1 < ticks
                && let Some(next_target) = self.next_target_at
            {
                let now = self.clock.now();
                if now < next_target {
                    let sleep_for = next_target.saturating_duration_since(now);
                    thread::sleep(sleep_for);
                }
            }
        }
        records
    }

    /// Devuelve una referencia a la traza acumulada.
    pub const fn trace(&self) -> &MultirateTrace {
        &self.trace
    }

    fn evaluate_tick(
        &self,
        now: MonotonicInstant,
        watchdog_status: &WatchdogStatus,
        terminal: Terminal,
    ) -> (
        GuardDecision,
        Option<MultirateFallbackReason>,
        Option<u64>,
        Option<Duration>,
        Option<Duration>,
    ) {
        let obs_age = self
            .latest_observation
            .as_ref()
            .map(|o| now.saturating_duration_since(o.observed_at));

        if self.worker_failed_in_epoch || watchdog_status.health == WatchdogHealth::Failed {
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (
                decision,
                Some(MultirateFallbackReason::WorkerFailed),
                None,
                None,
                obs_age,
            );
        }

        let Some(proposal) = &self.latest_proposal else {
            let reason = if watchdog_status.health == WatchdogHealth::Expired {
                MultirateFallbackReason::WatchdogExpired
            } else {
                MultirateFallbackReason::NoProposalYet
            };
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (decision, Some(reason), None, None, obs_age);
        };

        let prop_age = now.saturating_duration_since(proposal.produced_at);
        let prop_obs_age = now.saturating_duration_since(proposal.observation_timestamp);

        if proposal.epoch != self.current_epoch || prop_age > self.config.proposal_max_age {
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (
                decision,
                Some(MultirateFallbackReason::StalePlan),
                Some(proposal.revision),
                Some(prop_age),
                obs_age.or(Some(prop_obs_age)),
            );
        }

        if let Some(obs) = &self.latest_observation {
            let latest_obs_age = now.saturating_duration_since(obs.observed_at);
            let obs_expired = obs.expires_at.is_some_and(|exp| now > exp);
            if obs.epoch != self.current_epoch
                || obs_expired
                || latest_obs_age > self.config.observation_max_age
            {
                let decision = self.evaluate_fallback_through_guard(terminal);
                return (
                    decision,
                    Some(MultirateFallbackReason::StaleObservation),
                    Some(proposal.revision),
                    Some(prop_age),
                    Some(latest_obs_age),
                );
            }
        } else if prop_obs_age > self.config.observation_max_age {
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (
                decision,
                Some(MultirateFallbackReason::StaleObservation),
                Some(proposal.revision),
                Some(prop_age),
                Some(prop_obs_age),
            );
        }

        if proposal.deadline.is_expired_at(now) {
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (
                decision,
                Some(MultirateFallbackReason::DeadlineExpired),
                Some(proposal.revision),
                Some(prop_age),
                obs_age.or(Some(prop_obs_age)),
            );
        }

        if watchdog_status.health == WatchdogHealth::Expired {
            let decision = self.evaluate_fallback_through_guard(terminal);
            return (
                decision,
                Some(MultirateFallbackReason::WatchdogExpired),
                Some(proposal.revision),
                Some(prop_age),
                obs_age.or(Some(prop_obs_age)),
            );
        }

        let mut action_proposal =
            ActionProposal::authorized(proposal.action.clone(), proposal.proposal_tick);
        if let Some(max_age_ticks) = self.config.proposal_max_age_ticks {
            action_proposal = action_proposal.with_max_age_ticks(max_age_ticks);
        }

        let mut ctx = GuardContext::new(
            &self.action_space,
            self.current_tick,
            Instant::now(),
            terminal,
        );
        if let Some(obs) = &self.latest_observation {
            ctx = ctx.with_observation(&obs.observation);
        }

        let decision = self.guard.evaluate(&action_proposal, &ctx);
        let fallback_reason = match &decision {
            GuardDecision::Allow(_) => None,
            GuardDecision::Fallback(_) | GuardDecision::NoOp => {
                Some(MultirateFallbackReason::GuardRejected)
            }
            GuardDecision::Reject(reason) => Some(match reason {
                GuardRejectionReason::DeadlineExpired => MultirateFallbackReason::DeadlineExpired,
                GuardRejectionReason::StaleProposal { .. } => MultirateFallbackReason::StalePlan,
                _ => MultirateFallbackReason::GuardRejected,
            }),
        };

        (
            decision,
            fallback_reason,
            Some(proposal.revision),
            Some(prop_age),
            obs_age.or(Some(prop_obs_age)),
        )
    }

    fn evaluate_fallback_through_guard(&self, terminal: Terminal) -> GuardDecision {
        let unauthorized = ActionProposal::unauthorized(self.current_tick);
        let mut ctx = GuardContext::new(
            &self.action_space,
            self.current_tick,
            Instant::now(),
            terminal,
        );
        if let Some(obs) = &self.latest_observation {
            ctx = ctx.with_observation(&obs.observation);
        }
        self.guard.evaluate(&unauthorized, &ctx)
    }
}

/// Handle del worker best-effort ejecutado en segundo plano.
pub struct BestEffortWorkerHandle {
    stop_flag: Arc<AtomicBool>,
    join_handle: Option<JoinHandle<()>>,
}

impl BestEffortWorkerHandle {
    /// Lanza un worker best-effort en un hilo separado que consume la última observación
    /// disponible y publica [`WorkerReport`] en el mailbox acotado.
    ///
    /// La ruta de control nunca espera a este hilo ni bloquea por su ejecución.
    pub fn spawn<C, F>(
        name: impl Into<String>,
        observation_mailbox: BoundedMailbox<SensorReading>,
        proposal_mailbox: BoundedMailbox<WorkerReport>,
        clock: Arc<C>,
        mut step_fn: F,
    ) -> Self
    where
        C: MonotonicClock + Send + Sync + 'static,
        F: FnMut(Option<SensorReading>, &MonotonicClockRef<'_>) -> Option<WorkerReport>
            + Send
            + 'static,
    {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_clone = Arc::clone(&stop_flag);
        let thread_name = name.into();

        let builder = thread::Builder::new().name(thread_name);
        let handle = builder
            .spawn(move || {
                let clock_ref = MonotonicClockRef(clock.as_ref());
                while !stop_clone.load(Ordering::Relaxed) {
                    let now = clock.now();
                    let latest_obs = observation_mailbox
                        .try_recv_latest_nonblocking(now)
                        .map(|m| m.payload);
                    if let Some(report) = step_fn(latest_obs, &clock_ref) {
                        let send_now = clock.now();
                        let rev = match &report {
                            WorkerReport::Proposal(p) => p.revision,
                            WorkerReport::Failed { revision, .. } => *revision,
                            WorkerReport::Heartbeat { .. } => 0,
                        };
                        let msg = ChannelMessage::new(report, send_now, rev);
                        let _ = proposal_mailbox.send(msg, send_now);
                    }
                }
            })
            .expect("failed to spawn best-effort worker thread");

        Self {
            stop_flag,
            join_handle: Some(handle),
        }
    }

    /// Solicita la detención del worker y espera a que el hilo termine limpiamente.
    pub fn stop_and_join(mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for BestEffortWorkerHandle {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

/// Referencia ligera a [`MonotonicClock`] para callbacks del worker best-effort.
pub struct MonotonicClockRef<'a>(&'a dyn MonotonicClock);

impl<'a> MonotonicClockRef<'a> {
    /// Lee el instante monotónico actual.
    pub fn now(&self) -> MonotonicInstant {
        self.0.now()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::{
        ActionBounds, ActionSpace, SafeFallbackPolicy, SimDuration, SystemMonotonicClock,
        VirtualMonotonicClock,
    };

    fn test_action_space() -> ActionSpace {
        ActionSpace::continuous(vec![ActionBounds::new(-1.0, 1.0).expect("valid bounds")])
            .expect("valid 1d action space")
    }

    fn safe_fallback() -> Action {
        Action::continuous(vec![0.0]).expect("valid fallback action")
    }

    fn proposed_action(val: f64) -> Action {
        Action::continuous(vec![val]).expect("valid proposed action")
    }

    #[test]
    fn ac1_periodic_control_1ms_10ms_50ms_never_waits_for_slow_planner_or_sensor() {
        for period in [
            ControlPeriod::ONE_MS,
            ControlPeriod::TEN_MS,
            ControlPeriod::FIFTY_MS,
        ] {
            let clock = Arc::new(SystemMonotonicClock::new());
            let mut config = MultirateConfig::new(period);
            config.watchdog_timeout = Duration::from_millis(20);
            let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback()));
            let mut controller =
                MultirateController::new(config, test_action_space(), guard, clock.clone());

            let obs_mb = controller.observation_mailbox();
            let prop_mb = controller.proposal_mailbox();

            // Worker best-effort intencionalmente lento (120 ms por ciclo), muy por encima
            // de los periodos ilustrativos de 1 ms, 10 ms y 50 ms.
            let worker = BestEffortWorkerHandle::spawn(
                "slow-planner-test",
                obs_mb,
                prop_mb,
                clock.clone(),
                |_, clk| {
                    thread::sleep(Duration::from_millis(120));
                    let now = clk.now();
                    Some(WorkerReport::Proposal(WorkerProposal {
                        action: proposed_action(0.5),
                        proposal_tick: Tick::new(1),
                        revision: 1,
                        epoch: 1,
                        observation_revision: 1,
                        observation_timestamp: now,
                        produced_at: now,
                        deadline: MonotonicDeadline::at(
                            now.saturating_add(Duration::from_millis(200)),
                        ),
                    }))
                },
            );

            // Ejecutamos 5 ticks de control; ninguno debe bloquearse 120 ms.
            let records = controller.run_periodic_loop(5, Terminal::Ongoing);
            assert_eq!(records.len(), 5);
            for rec in &records {
                assert!(
                    rec.control_step_latency < Duration::from_millis(5),
                    "control step blocked unexpectedly for {:?} on period {:?}",
                    rec.control_step_latency,
                    period
                );
            }
            // En 1 ms (4 ms totales) y 10 ms (40 ms totales) el worker de 120 ms aún no entregó:
            // todos los ticks deben usar respaldo seguro sin esperar.
            // En 50 ms (200 ms totales), al menos los primeros 3 ticks (0, 50, 100 ms) usan respaldo.
            let expected_initial_fallbacks = if period == ControlPeriod::FIFTY_MS {
                3
            } else {
                5
            };
            for rec in records.iter().take(expected_initial_fallbacks) {
                assert!(rec.used_fallback);
                assert!(matches!(
                    rec.fallback_reason,
                    Some(
                        MultirateFallbackReason::NoProposalYet
                            | MultirateFallbackReason::WatchdogExpired
                    )
                ));
                assert_eq!(rec.guard_decision, GuardDecision::Fallback(safe_fallback()));
            }

            worker.stop_and_join();
        }
    }

    #[test]
    fn ac2_distinguishes_worker_failure_stale_plan_stale_observation_and_expired_deadline() {
        let clock = Arc::new(VirtualMonotonicClock::new());
        let config = MultirateConfig::new(ControlPeriod::TEN_MS)
            .with_proposal_max_age(Duration::from_millis(25))
            .with_observation_max_age(Duration::from_millis(20))
            .with_watchdog_timeout(Duration::from_millis(100));
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback()));
        let mut controller =
            MultirateController::new(config, test_action_space(), guard, clock.clone());

        // 1. Sin propuesta aún -> NoProposalYet
        let r1 = controller.step_control_tick(Terminal::Ongoing);
        assert!(r1.used_fallback);
        assert_eq!(
            r1.fallback_reason,
            Some(MultirateFallbackReason::NoProposalYet)
        );
        assert_eq!(r1.guard_decision, GuardDecision::Fallback(safe_fallback()));

        // 2. Propuesta válida y observación vigente -> Allow
        let now = clock.now();
        controller
            .submit_observation(SensorReading::new(
                Observation::new(Tick::new(2), SimDuration::ZERO, vec![1.0])
                    .expect("valid observation"),
                now,
                1,
                1,
            ))
            .unwrap();
        controller
            .submit_proposal(WorkerProposal {
                action: proposed_action(0.8),
                proposal_tick: Tick::new(2),
                revision: 1,
                epoch: 1,
                observation_revision: 1,
                observation_timestamp: now,
                produced_at: now,
                deadline: MonotonicDeadline::at(now.saturating_add(Duration::from_millis(30))),
            })
            .unwrap();

        let r2 = controller.step_control_tick(Terminal::Ongoing);
        assert!(!r2.used_fallback);
        assert_eq!(r2.fallback_reason, None);
        assert_eq!(
            r2.guard_decision,
            GuardDecision::Allow(proposed_action(0.8))
        );

        // 3. Avanza reloj 22 ms: observación supera observation_max_age (20 ms) -> StaleObservation
        clock.advance(Duration::from_millis(22));
        let r3 = controller.step_control_tick(Terminal::Ongoing);
        assert!(r3.used_fallback);
        assert_eq!(
            r3.fallback_reason,
            Some(MultirateFallbackReason::StaleObservation)
        );
        assert_eq!(r3.guard_decision, GuardDecision::Fallback(safe_fallback()));

        // 4. Renovamos observación, pero propuesta supera proposal_max_age (25 ms) -> StalePlan
        clock.advance(Duration::from_millis(5)); // total +27 ms desde propuesta rev 1
        let now4 = clock.now();
        controller
            .submit_observation(SensorReading::new(
                Observation::new(Tick::new(4), SimDuration::ZERO, vec![1.1])
                    .expect("valid observation"),
                now4,
                2,
                1,
            ))
            .unwrap();
        let r4 = controller.step_control_tick(Terminal::Ongoing);
        assert!(r4.used_fallback);
        assert_eq!(r4.fallback_reason, Some(MultirateFallbackReason::StalePlan));

        // 5. Propuesta fresca en edad pero con deadline ya vencido -> DeadlineExpired
        let now5 = clock.now();
        controller
            .submit_proposal(WorkerProposal {
                action: proposed_action(0.4),
                proposal_tick: Tick::new(5),
                revision: 2,
                epoch: 1,
                observation_revision: 2,
                observation_timestamp: now5,
                produced_at: now5,
                deadline: MonotonicDeadline::at(now5.saturating_add(Duration::from_millis(5))),
            })
            .unwrap();
        clock.advance(Duration::from_millis(6)); // deadline vencido, pero age=6ms < 25ms
        let r5 = controller.step_control_tick(Terminal::Ongoing);
        assert!(r5.used_fallback);
        assert_eq!(
            r5.fallback_reason,
            Some(MultirateFallbackReason::DeadlineExpired)
        );

        // 6. Fallo explícito del worker -> WorkerFailed
        controller
            .submit_worker_failure(1, 3, "planner diverged")
            .unwrap();
        let r6 = controller.step_control_tick(Terminal::Ongoing);
        assert!(r6.used_fallback);
        assert_eq!(
            r6.fallback_reason,
            Some(MultirateFallbackReason::WorkerFailed)
        );
        assert_eq!(r6.watchdog_status.health, WatchdogHealth::Failed);
    }

    #[test]
    fn ac3_trace_records_real_times_jitter_fallbacks_and_illustrative_notice() {
        let clock = Arc::new(VirtualMonotonicClock::new());
        let config = MultirateConfig::new(ControlPeriod::TEN_MS)
            .with_watchdog_timeout(Duration::from_millis(15));
        let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback()));
        let mut controller =
            MultirateController::new(config, test_action_space(), guard, clock.clone());

        // Tick 1: inicio exacto en 0 ms -> jitter 0
        let t1 = controller.step_control_tick(Terminal::Ongoing);
        assert_eq!(t1.jitter_nanos, 0);

        // Tick 2: simulamos 12 ms reales en lugar de 10 ms -> jitter +2 ms
        clock.advance(Duration::from_millis(12));
        let t2 = controller.step_control_tick(Terminal::Ongoing);
        assert_eq!(t2.jitter_nanos, 2_000_000);

        // Tick 3: simulamos +18 ms (total 30 ms, objetivo 20 ms -> jitter +10 ms) y watchdog expira
        clock.advance(Duration::from_millis(18));
        let t3 = controller.step_control_tick(Terminal::Ongoing);
        assert_eq!(t3.jitter_nanos, 10_000_000);
        assert_eq!(
            t3.fallback_reason,
            Some(MultirateFallbackReason::WatchdogExpired)
        );

        let trace = controller.trace();
        assert_eq!(trace.len(), 3);
        assert_eq!(trace.control_period(), ControlPeriod::TEN_MS);
        assert!(trace.timing_guarantee_note().contains("ilustrativos"));
        assert!(
            trace
                .timing_guarantee_note()
                .contains("no constituyen garantías de tiempo real duro")
        );
        assert_eq!(trace.max_abs_jitter_nanos(), 10_000_000);
        assert_eq!(trace.fallback_count(), 3);
        assert_eq!(
            trace.count_by_fallback_reason(MultirateFallbackReason::NoProposalYet),
            2
        );
        assert_eq!(
            trace.count_by_fallback_reason(MultirateFallbackReason::WatchdogExpired),
            1
        );
    }

    #[test]
    fn nonblocking_contentions_counted_exactly_once_per_contended_call() {
        use std::sync::Barrier;

        let mailbox: BoundedMailbox<u32> = BoundedMailbox::new(NonZeroUsize::new(4).unwrap(), None);
        let now = MonotonicInstant::from_nanos(1);
        assert_eq!(mailbox.stats().nonblocking_contentions, 0);

        // Un hilo sostiene el mutex entre dos barreras mientras el test llama.
        let holder = mailbox.clone();
        let acquired = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let (a2, r2) = (Arc::clone(&acquired), Arc::clone(&release));
        let handle = thread::spawn(move || {
            let _guard = holder.inner.lock().unwrap();
            a2.wait();
            r2.wait();
        });
        acquired.wait();

        assert!(mailbox.try_recv_latest_nonblocking(now).is_none());
        assert_eq!(mailbox.nonblocking_contentions.load(Ordering::Relaxed), 1);
        assert!(mailbox.try_drain_nonblocking(now).is_empty());
        assert_eq!(mailbox.nonblocking_contentions.load(Ordering::Relaxed), 2);
        // stats_nonblocking también cuenta la contención una vez por llamada.
        assert_eq!(mailbox.stats_nonblocking().nonblocking_contentions, 3);

        release.wait();
        handle.join().unwrap();

        // Sin contención no se incrementa; stats() refleja el contador atómico.
        assert!(mailbox.try_recv_latest_nonblocking(now).is_none());
        assert!(mailbox.try_drain_nonblocking(now).is_empty());
        assert_eq!(mailbox.stats().nonblocking_contentions, 3);
        assert_eq!(mailbox.stats_nonblocking().nonblocking_contentions, 3);
    }
}
