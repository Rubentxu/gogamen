//! Scheduler local determinista de ticks y eventos (`std`-only).
//!
//! Este módulo implementa un planificador local sin red, sin UI y sin dependencias
//! externas, diseñado para ejecutar tareas periódicas y reactivas por eventos con
//! ciclo de vida explícito y acotamiento estricto.
//!
//! # Ciclo de vida explícito
//! [`LocalScheduler`] separa estrictamente las fases de su ciclo de vida
//! ([`SchedulerLifecycle`]):
//! 1. [`SchedulerLifecycle::Created`]: registro de tareas ([`LocalScheduler::register_task`]).
//! 2. [`LocalScheduler::prepare`]: valida la configuración y transiciona a
//!    [`SchedulerLifecycle::Prepared`].
//! 3. [`LocalScheduler::start`]: inicializa los recursos/estado de cada tarea mediante
//!    [`TaskHandler::on_start`] y transiciona a [`SchedulerLifecycle::Running`].
//! 4. [`LocalScheduler::step`]: avanza un [`Tick`] usando [`StepDt`] y un
//!    [`MonotonicClock`], ejecutando tareas periódicas y eventos en cola.
//!    Rechaza cualquier invocación si el estado no es [`SchedulerLifecycle::Running`].
//! 5. [`LocalScheduler::stop`]: invoca [`TaskHandler::on_stop`] en todas las tareas
//!    para liberar recursos de forma verificable y transiciona a
//!    [`SchedulerLifecycle::Stopped`]. Adicionalmente, [`Drop`] garantiza la liberación
//!    si el scheduler se destruye mientras sigue en ejecución.
//!
//! # Orden estable y determinista de ejecución
//! El primer [`LocalScheduler::step`] ejecuta el `Tick(1)`, el segundo el `Tick(2)`, etc.
//!
//! ## 1. Tareas periódicas (`ScheduleTrigger::Periodic`)
//! En cada tick `t`, son elegibles todas las tareas periódicas tales que
//! `t.as_u64() % every_ticks.get() == 0`.
//!
//! Cuando coinciden múltiples tareas periódicas en el mismo tick, el orden de
//! ejecución es **total, estable y determinista** según la clave:
//! `(priority_desc, registration_order_asc)`:
//! 1. Mayor `priority` (`i32`) primero.
//! 2. En caso de empate en `priority`, menor `registration_order` (`usize`, orden de registro).
//!
//! ## 2. Eventos y tareas reactivas (`ScheduleTrigger::OnEvent`)
//! Los eventos se procesan de forma **iterativa (FIFO)** en una cola acotada, nunca
//! mediante recursión:
//! 1. Los eventos externos encolados antes de `step()` entran con `depth = 0` en orden FIFO.
//! 2. Los eventos emitidos por tareas periódicas entran al final de la cola con `depth = 1`.
//! 3. Al extraer un evento de la cola (FIFO), se ejecutan todas las tareas `OnEvent(kind)`
//!    coincidentes ordenadas por `(priority_desc, registration_order_asc)`.
//! 4. Los eventos emitidos por una tarea reactiva que procesa un evento de profundidad `d`
//!    entran al final de la cola con `depth = d + 1`.
//!
//! ## 3. Límites explícitos de cola de eventos
//! Para evitar bucles infinitos o recursión sin límite cuando unas tareas disparan a otras:
//! - [`SchedulerConfig::max_events_per_tick`]: número máximo de eventos procesados en un
//!   mismo `step()`. Si se supera, `step()` devuelve [`SchedulerError::EventLimitExceeded`].
//! - [`SchedulerConfig::max_event_depth`]: profundidad causal máxima permitida para eventos
//!   encadenados. Si una tarea intenta emitir un evento con `depth > max_event_depth`,
//!   `step()` devuelve [`SchedulerError::EventDepthExceeded`].
//! - [`SchedulerConfig::max_queue_capacity`]: capacidad máxima de eventos externos en espera
//!   antes de ejecutar `step()`. Si se llena, [`LocalScheduler::enqueue_event`] devuelve
//!   [`SchedulerError::EventQueueFull`].

use std::cmp::Reverse;
use std::collections::{BTreeMap, VecDeque};
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::clock::{MonotonicClock, MonotonicInstant};
use crate::error::DomainError;
use crate::units::{SimDuration, StepDt, Tick};

/// Identificador lógico de tipo de evento dentro del [`LocalScheduler`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventKind(String);

impl EventKind {
    /// Crea un [`EventKind`] a partir de una cadena no vacía.
    pub fn new(kind: impl Into<String>) -> Result<Self, SchedulerError> {
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(SchedulerError::InvalidConfig {
                reason: "event kind must not be empty",
            });
        }
        Ok(Self(kind))
    }

    /// Devuelve el nombre del tipo de evento.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EventKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Evento inmutable procesado por el [`LocalScheduler`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SchedulerEvent {
    kind: EventKind,
    payload: String,
}

impl SchedulerEvent {
    /// Crea un evento con tipo y carga útil textual.
    pub fn new(kind: EventKind, payload: impl Into<String>) -> Self {
        Self {
            kind,
            payload: payload.into(),
        }
    }

    /// Crea un evento sin carga útil (`payload` vacío).
    pub fn signal(kind: EventKind) -> Self {
        Self {
            kind,
            payload: String::new(),
        }
    }

    /// Tipo del evento.
    pub fn kind(&self) -> &EventKind {
        &self.kind
    }

    /// Carga útil textual del evento.
    pub fn payload(&self) -> &str {
        &self.payload
    }
}

/// Condición de disparo de una [`ScheduledTask`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScheduleTrigger {
    /// Se dispara cada `every_ticks` ticks (`tick.as_u64() % every_ticks.get() == 0`).
    Periodic {
        /// Periodo positivo en ticks.
        every_ticks: NonZeroU64,
    },
    /// Se dispara cuando se procesa un evento del tipo indicado.
    OnEvent(EventKind),
}

/// Estado explícito del ciclo de vida de [`LocalScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SchedulerLifecycle {
    /// Recién creado; acepta registro de tareas antes de `prepare()`.
    Created,
    /// Validado y listo para arrancar mediante `start()`.
    Prepared,
    /// En ejecución; inicializado y habilitado para `step()`.
    Running,
    /// Detenido; recursos liberados mediante `stop()`.
    Stopped,
}

impl fmt::Display for SchedulerLifecycle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Created => f.write_str("Created"),
            Self::Prepared => f.write_str("Prepared"),
            Self::Running => f.write_str("Running"),
            Self::Stopped => f.write_str("Stopped"),
        }
    }
}

/// Configuración acotada para [`LocalScheduler`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedulerConfig {
    /// Máximo de eventos procesables en un mismo `step()`.
    pub max_events_per_tick: usize,
    /// Profundidad causal máxima de eventos encadenados dentro de un `step()`.
    pub max_event_depth: usize,
    /// Capacidad máxima de la cola de eventos externos previos a `step()`.
    pub max_queue_capacity: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_events_per_tick: 256,
            max_event_depth: 16,
            max_queue_capacity: 256,
        }
    }
}

impl SchedulerConfig {
    /// Valida que los límites configurados sean coherentes y positivos.
    pub fn validate(&self) -> Result<(), SchedulerError> {
        if self.max_events_per_tick == 0 {
            return Err(SchedulerError::InvalidConfig {
                reason: "max_events_per_tick must be greater than zero",
            });
        }
        if self.max_queue_capacity == 0 {
            return Err(SchedulerError::InvalidConfig {
                reason: "max_queue_capacity must be greater than zero",
            });
        }
        Ok(())
    }
}

/// Errores tipados del [`LocalScheduler`].
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulerError {
    /// Operación invocada en un estado de ciclo de vida incompatible.
    InvalidLifecycleState {
        /// Nombre de la operación rechazada.
        operation: &'static str,
        /// Estado requerido.
        expected: SchedulerLifecycle,
        /// Estado actual del scheduler.
        actual: SchedulerLifecycle,
    },
    /// Nombre de tarea vacío.
    EmptyTaskName,
    /// Configuración o parámetro inválido.
    InvalidConfig {
        /// Motivo del rechazo.
        reason: &'static str,
    },
    /// La cola externa de eventos alcanzó su capacidad máxima.
    EventQueueFull {
        /// Capacidad configurada.
        capacity: usize,
    },
    /// Se superó el límite de eventos por tick durante un `step()`.
    EventLimitExceeded {
        /// Tick en el que ocurrió el exceso.
        tick: Tick,
        /// Límite configurado de eventos por tick.
        limit: usize,
    },
    /// Se superó la profundidad causal máxima de eventos en cadena.
    EventDepthExceeded {
        /// Tick en el que ocurrió el exceso.
        tick: Tick,
        /// Profundidad que intentó encolarse.
        depth: usize,
        /// Profundidad máxima permitida.
        max_depth: usize,
        /// Tipo de evento rechazado.
        event_kind: EventKind,
    },
    /// Una tarea falló durante `start()`, `step()` o `stop()`.
    TaskFailed {
        /// Nombre de la tarea que falló.
        task_name: String,
        /// Motivo legible del fallo.
        reason: String,
    },
    /// Error de dominio subyacente (por ejemplo desbordamiento de tick o duración).
    Domain(DomainError),
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLifecycleState {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "scheduler invalid lifecycle state for '{operation}': expected {expected}, got {actual}"
            ),
            Self::EmptyTaskName => f.write_str("scheduler task name must not be empty"),
            Self::InvalidConfig { reason } => write!(f, "scheduler invalid config: {reason}"),
            Self::EventQueueFull { capacity } => {
                write!(f, "scheduler event queue full (capacity={capacity})")
            }
            Self::EventLimitExceeded { tick, limit } => {
                write!(f, "scheduler event limit exceeded at {tick}: limit={limit}")
            }
            Self::EventDepthExceeded {
                tick,
                depth,
                max_depth,
                event_kind,
            } => write!(
                f,
                "scheduler event depth exceeded at {tick} for '{event_kind}': depth={depth} > max_depth={max_depth}"
            ),
            Self::TaskFailed { task_name, reason } => {
                write!(f, "scheduler task '{task_name}' failed: {reason}")
            }
            Self::Domain(err) => write!(f, "scheduler domain error: {err}"),
        }
    }
}

impl Error for SchedulerError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DomainError> for SchedulerError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

/// Contexto inmutable entregado a una tarea durante su ejecución.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TaskContext<'a> {
    tick: Tick,
    dt: StepDt,
    elapsed: SimDuration,
    event: Option<&'a SchedulerEvent>,
    event_depth: usize,
}

impl<'a> TaskContext<'a> {
    /// Tick lógico actual del paso.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Paso temporal físico de este tick.
    pub const fn dt(&self) -> StepDt {
        self.dt
    }

    /// Tiempo acumulado de simulación hasta este tick.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Evento que disparó la tarea si su trigger es [`ScheduleTrigger::OnEvent`].
    pub const fn event(&self) -> Option<&'a SchedulerEvent> {
        self.event
    }

    /// Profundidad causal actual (`0` para tareas periódicas y eventos externos).
    pub const fn event_depth(&self) -> usize {
        self.event_depth
    }
}

/// Contrato de ejecución y ciclo de vida para una tarea del [`LocalScheduler`].
pub trait TaskHandler {
    /// Inicializa recursos o estado de la tarea al invocar [`LocalScheduler::start`].
    fn on_start(&mut self) -> Result<(), SchedulerError> {
        Ok(())
    }

    /// Ejecuta la lógica de la tarea y devuelve los eventos nuevos a encolar.
    fn execute(&mut self, ctx: &TaskContext<'_>) -> Result<Vec<SchedulerEvent>, SchedulerError>;

    /// Libera recursos o estado de la tarea al invocar [`LocalScheduler::stop`] o [`Drop`].
    fn on_stop(&mut self) -> Result<(), SchedulerError> {
        Ok(())
    }
}

/// Definición de tarea programada dentro de [`LocalScheduler`].
pub struct ScheduledTask {
    name: String,
    trigger: ScheduleTrigger,
    priority: i32,
    handler: Box<dyn TaskHandler>,
}

impl ScheduledTask {
    /// Crea una nueva tarea programada con prioridad por defecto `0`.
    pub fn new(
        name: impl Into<String>,
        trigger: ScheduleTrigger,
        handler: impl TaskHandler + 'static,
    ) -> Result<Self, SchedulerError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(SchedulerError::EmptyTaskName);
        }
        Ok(Self {
            name,
            trigger,
            priority: 0,
            handler: Box::new(handler),
        })
    }

    /// Asigna una prioridad explícita (mayor valor se ejecuta antes en empates).
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Nombre de la tarea.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Disparador configurado para la tarea.
    pub fn trigger(&self) -> &ScheduleTrigger {
        &self.trigger
    }

    /// Prioridad configurada (mayor valor tiene precedencia).
    pub const fn priority(&self) -> i32 {
        self.priority
    }
}

/// Registro individual de una ejecución de tarea dentro de [`StepReport`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskExecutionRecord {
    /// Nombre de la tarea ejecutada.
    pub task_name: String,
    /// Prioridad de la tarea.
    pub priority: i32,
    /// Orden de registro asignado al añadir la tarea.
    pub registration_order: usize,
    /// Profundidad causal de la ejecución (`0` en periódicas o eventos externos).
    pub event_depth: usize,
}

/// Resumen determinista devuelto por [`LocalScheduler::step`].
#[derive(Debug, Clone, PartialEq)]
pub struct StepReport {
    /// Tick ejecutado.
    pub tick: Tick,
    /// Tiempo acumulado de simulación tras el paso.
    pub elapsed: SimDuration,
    /// Tareas ejecutadas en orden exacto de invocación.
    pub executions: Vec<TaskExecutionRecord>,
    /// Cantidad total de eventos procesados durante este tick.
    pub events_processed: usize,
    /// Instante monotónico al iniciar el paso.
    pub started_at: MonotonicInstant,
    /// Instante monotónico al finalizar el paso.
    pub finished_at: MonotonicInstant,
}

#[derive(Debug, Clone)]
struct QueuedEvent {
    event: SchedulerEvent,
    depth: usize,
}

struct RegisteredTask {
    registration_order: usize,
    task: ScheduledTask,
}

/// Planificador local determinista de tareas periódicas y reactivas por eventos.
///
/// Ver la documentación del módulo para la especificación exacta de ciclo de vida,
/// desempate y límites de cola.
pub struct LocalScheduler {
    config: SchedulerConfig,
    lifecycle: SchedulerLifecycle,
    current_tick: Tick,
    elapsed: SimDuration,
    tasks: Vec<RegisteredTask>,
    /// Índices de tareas periódicas ordenados por `(priority desc, registration_order asc)`.
    /// Se calcula una sola vez en `prepare()`; el registro queda cerrado después.
    sorted_periodic_tasks: Vec<usize>,
    /// Índices de listeners por tipo de evento con el mismo orden determinista.
    sorted_event_listeners: BTreeMap<EventKind, Vec<usize>>,
    event_queue: VecDeque<QueuedEvent>,
}

impl LocalScheduler {
    /// Crea un scheduler en estado [`SchedulerLifecycle::Created`].
    pub fn new(config: SchedulerConfig) -> Result<Self, SchedulerError> {
        config.validate()?;
        Ok(Self {
            config,
            lifecycle: SchedulerLifecycle::Created,
            current_tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
            tasks: Vec::new(),
            sorted_periodic_tasks: Vec::new(),
            sorted_event_listeners: BTreeMap::new(),
            event_queue: VecDeque::new(),
        })
    }

    /// Estado actual del ciclo de vida.
    pub const fn lifecycle(&self) -> SchedulerLifecycle {
        self.lifecycle
    }

    /// Último tick ejecutado (`Tick::ZERO` antes del primer `step()`).
    pub const fn current_tick(&self) -> Tick {
        self.current_tick
    }

    /// Duración acumulada de simulación.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Configuración activa del scheduler.
    pub const fn config(&self) -> SchedulerConfig {
        self.config
    }

    /// Registra una tarea mientras el scheduler está en [`SchedulerLifecycle::Created`].
    pub fn register_task(&mut self, task: ScheduledTask) -> Result<usize, SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Created {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "register_task",
                expected: SchedulerLifecycle::Created,
                actual: self.lifecycle,
            });
        }
        let registration_order = self.tasks.len();
        self.tasks.push(RegisteredTask {
            registration_order,
            task,
        });
        Ok(registration_order)
    }

    /// Prepara el scheduler y valida la transición `Created -> Prepared`.
    pub fn prepare(&mut self) -> Result<(), SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Created {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "prepare",
                expected: SchedulerLifecycle::Created,
                actual: self.lifecycle,
            });
        }
        self.config.validate()?;

        // Orden determinista precomputado: (priority desc, registration_order asc).
        let mut order: Vec<usize> = (0..self.tasks.len()).collect();
        order.sort_by_key(|&idx| {
            let entry = &self.tasks[idx];
            (Reverse(entry.task.priority), entry.registration_order)
        });
        self.sorted_periodic_tasks.clear();
        self.sorted_event_listeners.clear();
        for idx in order {
            match &self.tasks[idx].task.trigger {
                ScheduleTrigger::Periodic { .. } => self.sorted_periodic_tasks.push(idx),
                ScheduleTrigger::OnEvent(kind) => self
                    .sorted_event_listeners
                    .entry(kind.clone())
                    .or_default()
                    .push(idx),
            }
        }

        self.lifecycle = SchedulerLifecycle::Prepared;
        Ok(())
    }

    /// Arranca el scheduler (`Prepared -> Running`) inicializando todas las tareas.
    ///
    /// Si una tarea falla durante [`TaskHandler::on_start`], intenta detener las tareas ya
    /// inicializadas y mantiene el scheduler fuera de `Running`.
    pub fn start(&mut self) -> Result<(), SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Prepared {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "start",
                expected: SchedulerLifecycle::Prepared,
                actual: self.lifecycle,
            });
        }

        for idx in 0..self.tasks.len() {
            if let Err(err) = self.tasks[idx].task.handler.on_start() {
                for rollback_idx in (0..idx).rev() {
                    let _ = self.tasks[rollback_idx].task.handler.on_stop();
                }
                return Err(err);
            }
        }

        self.lifecycle = SchedulerLifecycle::Running;
        Ok(())
    }

    /// Encola un evento externo (`depth = 0`) para ser procesado en el siguiente `step()`.
    pub fn enqueue_event(&mut self, event: SchedulerEvent) -> Result<(), SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Running
            && self.lifecycle != SchedulerLifecycle::Prepared
        {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "enqueue_event",
                expected: SchedulerLifecycle::Running,
                actual: self.lifecycle,
            });
        }
        if self.event_queue.len() >= self.config.max_queue_capacity {
            return Err(SchedulerError::EventQueueFull {
                capacity: self.config.max_queue_capacity,
            });
        }
        self.event_queue.push_back(QueuedEvent { event, depth: 0 });
        Ok(())
    }

    /// Ejecuta un paso discreto (`dt`) midiendo instantes monotónicos con `clock`.
    ///
    /// Solo válido en estado [`SchedulerLifecycle::Running`].
    pub fn step(
        &mut self,
        dt: StepDt,
        clock: &dyn MonotonicClock,
    ) -> Result<StepReport, SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Running {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "step",
                expected: SchedulerLifecycle::Running,
                actual: self.lifecycle,
            });
        }

        let started_at = clock.now();
        let next_tick = self.current_tick.next()?;
        let next_elapsed = self.elapsed.checked_add_dt(dt)?;
        self.current_tick = next_tick;
        self.elapsed = next_elapsed;

        let mut executions = Vec::new();
        let mut events_processed = 0usize;

        // 1. Tareas periódicas elegibles en orden determinista: (priority_desc, registration_order_asc).
        for pos in 0..self.sorted_periodic_tasks.len() {
            let idx = self.sorted_periodic_tasks[pos];
            match self.tasks[idx].task.trigger {
                ScheduleTrigger::Periodic { every_ticks }
                    if next_tick.as_u64() % every_ticks.get() == 0 => {}
                _ => continue,
            }
            let emitted = {
                let entry = &mut self.tasks[idx];
                let ctx = TaskContext {
                    tick: next_tick,
                    dt,
                    elapsed: next_elapsed,
                    event: None,
                    event_depth: 0,
                };
                executions.push(TaskExecutionRecord {
                    task_name: entry.task.name.clone(),
                    priority: entry.task.priority,
                    registration_order: entry.registration_order,
                    event_depth: 0,
                });
                entry.task.handler.execute(&ctx)?
            };

            self.push_emitted_events(next_tick, emitted, 1)?;
        }

        // 2. Cola iterativa FIFO acotada de eventos (sin recursión).
        while let Some(queued) = self.event_queue.pop_front() {
            if events_processed >= self.config.max_events_per_tick {
                return Err(SchedulerError::EventLimitExceeded {
                    tick: next_tick,
                    limit: self.config.max_events_per_tick,
                });
            }
            events_processed += 1;

            let listener_count = self
                .sorted_event_listeners
                .get(queued.event.kind())
                .map_or(0, Vec::len);
            let next_depth = queued.depth.saturating_add(1);
            for pos in 0..listener_count {
                let idx = self.sorted_event_listeners[queued.event.kind()][pos];
                let emitted = {
                    let entry = &mut self.tasks[idx];
                    let ctx = TaskContext {
                        tick: next_tick,
                        dt,
                        elapsed: next_elapsed,
                        event: Some(&queued.event),
                        event_depth: queued.depth,
                    };
                    executions.push(TaskExecutionRecord {
                        task_name: entry.task.name.clone(),
                        priority: entry.task.priority,
                        registration_order: entry.registration_order,
                        event_depth: queued.depth,
                    });
                    entry.task.handler.execute(&ctx)?
                };

                self.push_emitted_events(next_tick, emitted, next_depth)?;
            }
        }

        let finished_at = clock.now();
        Ok(StepReport {
            tick: next_tick,
            elapsed: next_elapsed,
            executions,
            events_processed,
            started_at,
            finished_at,
        })
    }

    /// Detiene el scheduler (`Running -> Stopped`) y libera los recursos de todas las tareas.
    pub fn stop(&mut self) -> Result<(), SchedulerError> {
        if self.lifecycle != SchedulerLifecycle::Running {
            return Err(SchedulerError::InvalidLifecycleState {
                operation: "stop",
                expected: SchedulerLifecycle::Running,
                actual: self.lifecycle,
            });
        }

        let mut first_error: Option<SchedulerError> = None;
        for entry in self.tasks.iter_mut().rev() {
            if let Err(err) = entry.task.handler.on_stop()
                && first_error.is_none()
            {
                first_error = Some(err);
            }
        }
        self.event_queue.clear();
        self.lifecycle = SchedulerLifecycle::Stopped;

        match first_error {
            Some(err) => Err(err),
            None => Ok(()),
        }
    }

    fn push_emitted_events(
        &mut self,
        tick: Tick,
        emitted: Vec<SchedulerEvent>,
        depth: usize,
    ) -> Result<(), SchedulerError> {
        for event in emitted {
            if depth > self.config.max_event_depth {
                return Err(SchedulerError::EventDepthExceeded {
                    tick,
                    depth,
                    max_depth: self.config.max_event_depth,
                    event_kind: event.kind().clone(),
                });
            }
            self.event_queue.push_back(QueuedEvent { event, depth });
        }
        Ok(())
    }
}

impl Drop for LocalScheduler {
    fn drop(&mut self) {
        if self.lifecycle == SchedulerLifecycle::Running {
            let _ = self.stop();
        }
    }
}

/// Rastreador verificable de inicialización y liberación de recursos para tareas/fixtures.
#[derive(Debug, Clone, Default)]
pub struct LifecycleTracker {
    active_resources: Arc<AtomicUsize>,
    starts: Arc<AtomicUsize>,
    stops: Arc<AtomicUsize>,
}

impl LifecycleTracker {
    /// Crea un rastreador con cero recursos activos.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adquiere un recurso rastreado que se liberará al destruirse o llamarse a `release()`.
    pub fn acquire(&self) -> TrackedResource {
        self.active_resources.fetch_add(1, Ordering::SeqCst);
        self.starts.fetch_add(1, Ordering::SeqCst);
        TrackedResource {
            active_resources: Arc::clone(&self.active_resources),
            stops: Arc::clone(&self.stops),
            released: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Número de recursos actualmente activos.
    pub fn active_count(&self) -> usize {
        self.active_resources.load(Ordering::SeqCst)
    }

    /// Número total de inicializaciones registradas.
    pub fn starts(&self) -> usize {
        self.starts.load(Ordering::SeqCst)
    }

    /// Número total de liberaciones registradas.
    pub fn stops(&self) -> usize {
        self.stops.load(Ordering::SeqCst)
    }
}

/// Guardia RAII de recurso adquirido desde [`LifecycleTracker`].
#[derive(Debug)]
pub struct TrackedResource {
    active_resources: Arc<AtomicUsize>,
    stops: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
}

impl TrackedResource {
    /// Libera explícitamente el recurso de forma idempotente.
    pub fn release(&mut self) {
        if !self.released.swap(true, Ordering::SeqCst) {
            self.active_resources.fetch_sub(1, Ordering::SeqCst);
            self.stops.fetch_add(1, Ordering::SeqCst);
        }
    }
}

impl Drop for TrackedResource {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::VirtualMonotonicClock;

    struct RecordingPeriodicTask {
        log: Arc<std::sync::Mutex<Vec<(String, u64)>>>,
        name: String,
    }

    impl TaskHandler for RecordingPeriodicTask {
        fn execute(
            &mut self,
            ctx: &TaskContext<'_>,
        ) -> Result<Vec<SchedulerEvent>, SchedulerError> {
            let mut guard = self.log.lock().unwrap();
            guard.push((self.name.clone(), ctx.tick().as_u64()));
            Ok(Vec::new())
        }
    }

    struct EmitterTask {
        emit_kind: EventKind,
        emits_remaining: usize,
        tracker: LifecycleTracker,
        resource: Option<TrackedResource>,
    }

    impl TaskHandler for EmitterTask {
        fn on_start(&mut self) -> Result<(), SchedulerError> {
            self.resource = Some(self.tracker.acquire());
            Ok(())
        }

        fn execute(
            &mut self,
            _ctx: &TaskContext<'_>,
        ) -> Result<Vec<SchedulerEvent>, SchedulerError> {
            if self.emits_remaining > 0 {
                self.emits_remaining -= 1;
                Ok(vec![SchedulerEvent::signal(self.emit_kind.clone())])
            } else {
                Ok(Vec::new())
            }
        }

        fn on_stop(&mut self) -> Result<(), SchedulerError> {
            if let Some(mut res) = self.resource.take() {
                res.release();
            }
            Ok(())
        }
    }

    struct LoopbackEventTask {
        emit_kind: EventKind,
        log: Arc<std::sync::Mutex<Vec<(String, usize)>>>,
        name: String,
    }

    impl TaskHandler for LoopbackEventTask {
        fn execute(
            &mut self,
            ctx: &TaskContext<'_>,
        ) -> Result<Vec<SchedulerEvent>, SchedulerError> {
            self.log
                .lock()
                .unwrap()
                .push((self.name.clone(), ctx.event_depth()));
            Ok(vec![SchedulerEvent::signal(self.emit_kind.clone())])
        }
    }

    #[test]
    fn ac1_two_periodic_tasks_different_frequencies_fire_on_expected_ticks() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let clock = VirtualMonotonicClock::new();
        let dt = StepDt::from_millis_f64(10.0).unwrap();

        let mut scheduler = LocalScheduler::new(SchedulerConfig::default()).unwrap();
        // Misma prioridad (0): desempate por registration_order_asc ("fast" antes que "slow").
        scheduler
            .register_task(
                ScheduledTask::new(
                    "fast_every_2",
                    ScheduleTrigger::Periodic {
                        every_ticks: NonZeroU64::new(2).unwrap(),
                    },
                    RecordingPeriodicTask {
                        log: Arc::clone(&log),
                        name: "fast".to_string(),
                    },
                )
                .unwrap(),
            )
            .unwrap();
        scheduler
            .register_task(
                ScheduledTask::new(
                    "slow_every_3",
                    ScheduleTrigger::Periodic {
                        every_ticks: NonZeroU64::new(3).unwrap(),
                    },
                    RecordingPeriodicTask {
                        log: Arc::clone(&log),
                        name: "slow".to_string(),
                    },
                )
                .unwrap(),
            )
            .unwrap();

        scheduler.prepare().unwrap();
        scheduler.start().unwrap();

        for _ in 0..6 {
            clock.advance(dt.try_into_std().unwrap());
            scheduler.step(dt, &clock).unwrap();
        }
        scheduler.stop().unwrap();

        let recorded = log.lock().unwrap().clone();
        assert_eq!(
            recorded,
            vec![
                ("fast".to_string(), 2),
                ("slow".to_string(), 3),
                ("fast".to_string(), 4),
                ("fast".to_string(), 6),
                ("slow".to_string(), 6),
            ]
        );
    }

    #[test]
    fn ac1_tie_breaking_uses_priority_desc_then_registration_order_asc() {
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let clock = VirtualMonotonicClock::new();
        let dt = StepDt::from_millis_f64(10.0).unwrap();
        let every_1 = ScheduleTrigger::Periodic {
            every_ticks: NonZeroU64::new(1).unwrap(),
        };

        let mut scheduler = LocalScheduler::new(SchedulerConfig::default()).unwrap();
        scheduler
            .register_task(
                ScheduledTask::new(
                    "low_first",
                    every_1.clone(),
                    RecordingPeriodicTask {
                        log: Arc::clone(&log),
                        name: "low_first".to_string(),
                    },
                )
                .unwrap()
                .with_priority(1),
            )
            .unwrap();
        scheduler
            .register_task(
                ScheduledTask::new(
                    "high_second",
                    every_1.clone(),
                    RecordingPeriodicTask {
                        log: Arc::clone(&log),
                        name: "high_second".to_string(),
                    },
                )
                .unwrap()
                .with_priority(10),
            )
            .unwrap();
        scheduler
            .register_task(
                ScheduledTask::new(
                    "high_third",
                    every_1,
                    RecordingPeriodicTask {
                        log: Arc::clone(&log),
                        name: "high_third".to_string(),
                    },
                )
                .unwrap()
                .with_priority(10),
            )
            .unwrap();

        scheduler.prepare().unwrap();
        scheduler.start().unwrap();
        let report = scheduler.step(dt, &clock).unwrap();
        scheduler.stop().unwrap();

        let names: Vec<&str> = report
            .executions
            .iter()
            .map(|r| r.task_name.as_str())
            .collect();
        assert_eq!(names, vec!["high_second", "high_third", "low_first"]);
    }

    #[test]
    fn ac2_queued_events_respect_max_depth_and_max_events_without_unbounded_recursion() {
        let clock = VirtualMonotonicClock::new();
        let dt = StepDt::from_millis_f64(10.0).unwrap();
        let ping = EventKind::new("ping").unwrap();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));

        let mut scheduler = LocalScheduler::new(SchedulerConfig {
            max_events_per_tick: 10,
            max_event_depth: 3,
            max_queue_capacity: 8,
        })
        .unwrap();

        scheduler
            .register_task(
                ScheduledTask::new(
                    "loopback",
                    ScheduleTrigger::OnEvent(ping.clone()),
                    LoopbackEventTask {
                        emit_kind: ping.clone(),
                        log: Arc::clone(&log),
                        name: "loopback".to_string(),
                    },
                )
                .unwrap(),
            )
            .unwrap();

        scheduler.prepare().unwrap();
        scheduler.start().unwrap();
        scheduler
            .enqueue_event(SchedulerEvent::signal(ping.clone()))
            .unwrap();

        let err = scheduler.step(dt, &clock).unwrap_err();
        assert!(matches!(
            err,
            SchedulerError::EventDepthExceeded {
                depth: 4,
                max_depth: 3,
                ..
            }
        ));
        // Profundidades 0, 1, 2, 3 ejecutadas iterativamente antes de rechazar depth 4.
        let depths: Vec<usize> = log.lock().unwrap().iter().map(|(_, d)| *d).collect();
        assert_eq!(depths, vec![0, 1, 2, 3]);
        scheduler.stop().unwrap();

        // Ahora probamos el límite max_events_per_tick con eventos externos depth=0.
        let mut bounded_sched = LocalScheduler::new(SchedulerConfig {
            max_events_per_tick: 2,
            max_event_depth: 5,
            max_queue_capacity: 4,
        })
        .unwrap();
        bounded_sched.prepare().unwrap();
        bounded_sched.start().unwrap();
        for _ in 0..3 {
            bounded_sched
                .enqueue_event(SchedulerEvent::signal(ping.clone()))
                .unwrap();
        }
        let err = bounded_sched.step(dt, &clock).unwrap_err();
        assert!(matches!(
            err,
            SchedulerError::EventLimitExceeded { limit: 2, .. }
        ));
        bounded_sched.stop().unwrap();
    }

    #[test]
    fn ac3_lifecycle_start_and_stop_allocate_and_release_resources_and_reject_invalid_step() {
        let tracker = LifecycleTracker::new();
        let clock = VirtualMonotonicClock::new();
        let dt = StepDt::from_millis_f64(5.0).unwrap();
        let evt = EventKind::new("tick_emitted").unwrap();

        let mut scheduler = LocalScheduler::new(SchedulerConfig::default()).unwrap();
        assert_eq!(scheduler.lifecycle(), SchedulerLifecycle::Created);
        assert!(matches!(
            scheduler.step(dt, &clock),
            Err(SchedulerError::InvalidLifecycleState {
                operation: "step",
                expected: SchedulerLifecycle::Running,
                actual: SchedulerLifecycle::Created,
            })
        ));

        scheduler
            .register_task(
                ScheduledTask::new(
                    "emitter",
                    ScheduleTrigger::Periodic {
                        every_ticks: NonZeroU64::new(1).unwrap(),
                    },
                    EmitterTask {
                        emit_kind: evt,
                        emits_remaining: 1,
                        tracker: tracker.clone(),
                        resource: None,
                    },
                )
                .unwrap(),
            )
            .unwrap();

        scheduler.prepare().unwrap();
        assert_eq!(scheduler.lifecycle(), SchedulerLifecycle::Prepared);
        assert_eq!(tracker.active_count(), 0);
        assert!(matches!(
            scheduler.step(dt, &clock),
            Err(SchedulerError::InvalidLifecycleState {
                operation: "step",
                expected: SchedulerLifecycle::Running,
                actual: SchedulerLifecycle::Prepared,
            })
        ));

        scheduler.start().unwrap();
        assert_eq!(scheduler.lifecycle(), SchedulerLifecycle::Running);
        assert_eq!(tracker.starts(), 1);
        assert_eq!(tracker.active_count(), 1);

        let report = scheduler.step(dt, &clock).unwrap();
        assert_eq!(report.tick, Tick::new(1));
        assert_eq!(report.events_processed, 1);

        scheduler.stop().unwrap();
        assert_eq!(scheduler.lifecycle(), SchedulerLifecycle::Stopped);
        assert_eq!(tracker.stops(), 1);
        assert_eq!(tracker.active_count(), 0);
        assert!(matches!(
            scheduler.step(dt, &clock),
            Err(SchedulerError::InvalidLifecycleState {
                operation: "step",
                expected: SchedulerLifecycle::Running,
                actual: SchedulerLifecycle::Stopped,
            })
        ));
    }

    #[test]
    fn gog129_precomputed_order_matches_priority_then_registration() {
        struct Noop;
        impl TaskHandler for Noop {
            fn execute(
                &mut self,
                _ctx: &TaskContext<'_>,
            ) -> Result<Vec<SchedulerEvent>, SchedulerError> {
                Ok(Vec::new())
            }
        }
        let ping = EventKind::new("ping").unwrap();
        let pong = EventKind::new("pong").unwrap();
        let every1 = ScheduleTrigger::Periodic {
            every_ticks: NonZeroU64::new(1).unwrap(),
        };
        let every2 = ScheduleTrigger::Periodic {
            every_ticks: NonZeroU64::new(2).unwrap(),
        };
        let mut s = LocalScheduler::new(SchedulerConfig::default()).unwrap();
        let specs: Vec<(&str, ScheduleTrigger, i32)> = vec![
            ("p_lo", every1.clone(), 0),
            ("ping_lo", ScheduleTrigger::OnEvent(ping.clone()), 0),
            ("p_hi_a", every1.clone(), 5),
            ("pong_hi", ScheduleTrigger::OnEvent(pong.clone()), 9),
            ("ping_hi_a", ScheduleTrigger::OnEvent(ping.clone()), 3),
            ("p_even_hi", every2, 5),
            ("p_hi_b", every1, 5),
            ("ping_hi_b", ScheduleTrigger::OnEvent(ping.clone()), 3),
        ];
        for (name, trig, prio) in specs {
            s.register_task(
                ScheduledTask::new(name, trig, Noop)
                    .unwrap()
                    .with_priority(prio),
            )
            .unwrap();
        }
        s.prepare().unwrap();
        s.start().unwrap();
        let clock = VirtualMonotonicClock::new();
        let dt = StepDt::from_millis_f64(10.0).unwrap();
        s.enqueue_event(SchedulerEvent::signal(ping.clone()))
            .unwrap();
        s.enqueue_event(SchedulerEvent::signal(pong)).unwrap();
        s.enqueue_event(SchedulerEvent::signal(ping)).unwrap();

        let r1 = s.step(dt, &clock).unwrap();
        let names1: Vec<&str> = r1.executions.iter().map(|e| e.task_name.as_str()).collect();
        assert_eq!(
            names1,
            vec![
                "p_hi_a",
                "p_hi_b",
                "p_lo", // tick 1 periodic
                "ping_hi_a",
                "ping_hi_b",
                "ping_lo", // first ping
                "pong_hi", // pong
                "ping_hi_a",
                "ping_hi_b",
                "ping_lo", // second ping
            ]
        );
        let r2 = s.step(dt, &clock).unwrap();
        let names2: Vec<&str> = r2.executions.iter().map(|e| e.task_name.as_str()).collect();
        assert_eq!(names2, vec!["p_hi_a", "p_even_hi", "p_hi_b", "p_lo"]);
    }
}
