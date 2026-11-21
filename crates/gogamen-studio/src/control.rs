//! Controles de sesión simulada (GOG-90): `ControlPort` (única vía mutante) separado de
//! `ObservationPort` (snapshots inmutables). Determinista: sin reloj de pared.
//!
//! El tiempo virtual se expresa en [`Tick`] de `gogamen-core`. La UI sólo recibe puertos; el
//! estado mutable vive en [`SessionHost`] (lado motor) tras un `Mutex` privado.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard};

use gogamen_core::Tick;

use crate::params::{
    ParamAck, ParamCommand, ParamControlPort, ParamEvent, ParamId, ParamOutcome, ParamRejectReason,
    ParamRejectionRecord, ParamSchema, ParamState, ParamValue,
};

/// Velocidad virtual máxima aceptada (multiplicador de ticks por unidad de `pump`).
pub const MAX_SPEED: f64 = 1000.0;
/// Capacidad del log de rechazos visible en la observación (los más antiguos se descartan).
pub const REJECTION_LOG_CAP: usize = 64;
/// Versión (aditiva) del contrato de `pump`: 1 = sin tope; 2 (GOG-148) = ticks por llamada
/// acotados con backlog (`PumpOutcome`). `pump` conserva su firma.
pub const PUMP_CONTRACT_VERSION: u32 = 2;
/// Tope por defecto de ticks ejecutados por llamada a `pump` (bajo el lock).
pub const DEFAULT_MAX_TICKS_PER_PUMP: u64 = 100_000;
/// Backlog máximo retenido; el exceso se descarta (se reporta en `dropped`).
pub const MAX_PUMP_BACKLOG: u64 = 1 << 32;

/// Resultado de [`SessionHost::pump_bounded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PumpOutcome {
    /// Ticks realmente avanzados en esta llamada.
    pub executed: u64,
    /// Ticks debidos aún no ejecutados (backlog); el cliente continúa con más `pump`.
    pub pending: u64,
    /// `true` si el tope por llamada impidió ejecutar todo lo debido.
    pub truncated: bool,
    /// Ticks descartados por superar `MAX_PUMP_BACKLOG` (peticiones absurdas).
    pub dropped: u64,
}

/// Capacidad por defecto de la ventana de deduplicación de `CommandId` (GOG-141).
pub const DEFAULT_DEDUP_WINDOW: usize = 4096;

/// Resultado de registrar un id en la ventana.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DedupVerdict {
    New,
    Duplicate,
    Expired,
}

/// Ventana deslizante acotada de ids vistos. Semántica: un id dentro de la ventana es
/// `Duplicate`; un id fuera de ella con valor <= al mayor id expulsado es `Expired` (se
/// rechaza, pues podría ser un duplicado ya olvidado); cualquier otro es `New`.
/// Memoria: `len() <= capacity` siempre.
struct DedupWindow {
    capacity: usize,
    order: VecDeque<CommandId>,
    set: HashSet<CommandId>,
    max_evicted: Option<CommandId>,
}

impl DedupWindow {
    fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            order: VecDeque::new(),
            set: HashSet::new(),
            max_evicted: None,
        }
    }

    fn len(&self) -> usize {
        self.set.len()
    }

    fn clear(&mut self) {
        self.order.clear();
        self.set.clear();
        self.max_evicted = None;
    }

    fn evict_to(&mut self, cap: usize) {
        while self.order.len() > cap {
            if let Some(old) = self.order.pop_front() {
                self.set.remove(&old);
                self.max_evicted = Some(self.max_evicted.map_or(old, |m| m.max(old)));
            }
        }
    }

    fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity.max(1);
        self.evict_to(self.capacity);
    }

    fn check_and_insert(&mut self, id: CommandId) -> DedupVerdict {
        if self.set.contains(&id) {
            return DedupVerdict::Duplicate;
        }
        if self.max_evicted.is_some_and(|m| id <= m) {
            return DedupVerdict::Expired;
        }
        self.evict_to(self.capacity - 1);
        self.order.push_back(id);
        self.set.insert(id);
        DedupVerdict::New
    }
}

/// Identificador de comando, único por sesión (reutilizarlo se rechaza como duplicado).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CommandId(pub u64);

/// Identidad de una sesión: `epoch` cambia al cambiar de lab, `generation` en cada `Reset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId {
    pub epoch: u64,
    pub generation: u64,
}

/// Tipo de sesión. Sólo `Simulated` tiene control por defecto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    Simulated,
    Physical,
    Remote,
}

/// Estado de ejecución del motor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    Paused,
    Running,
    /// Motor detenido (sólo por cierre de visor con `stop_on_close`). Terminal.
    Stopped,
}

/// Órdenes de control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlAction {
    Pause,
    Resume,
    /// Avanza exactamente un tick virtual. Sólo válido con el motor en pausa.
    Step,
    Reset,
    SetSpeed(f64),
}

/// Comando identificado y dirigido a una sesión concreta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlCommand {
    pub id: CommandId,
    pub session: SessionId,
    pub action: ControlAction,
}

/// Motivo tipado de rechazo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectReason {
    InvalidSpeed,
    DuplicateCommandId,
    /// Id fuera de la ventana de deduplicación y no mayor que el mayor id expulsado: no se
    /// puede descartar que sea un duplicado, así que se rechaza (opción segura, GOG-141).
    ExpiredCommandId,
    StaleSession {
        current: SessionId,
    },
    /// `Step` con el motor en `Running` (decisión: se rechaza, no es no-op).
    StepWhileRunning,
    EngineStopped,
    TickOverflow,
}

/// Resultado de un comando.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AckOutcome {
    Applied,
    Rejected { reason: RejectReason },
}

/// ACK: siempre lleva `revision` y `tick` tras procesar el comando (sin cambios si se rechaza).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ack {
    pub id: CommandId,
    pub outcome: AckOutcome,
    pub revision: u64,
    pub tick: Tick,
}

/// Entrada del log de rechazos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RejectionRecord {
    pub id: CommandId,
    pub reason: RejectReason,
}

/// Snapshot inmutable del estado de la sesión.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub session: SessionId,
    pub kind: SessionKind,
    pub revision: u64,
    pub tick: Tick,
    pub run_state: RunState,
    pub speed: f64,
    pub viewer_attached: bool,
    pub controllable: bool,
    /// Total histórico de rechazos (no se trunca).
    pub rejection_count: u64,
    /// Últimos rechazos (máx. [`REJECTION_LOG_CAP`]), del más antiguo al más reciente.
    pub recent_rejections: Vec<RejectionRecord>,
    /// Valores vigentes de los parámetros dinámicos (orden del esquema).
    pub params: Vec<(ParamId, ParamValue)>,
    /// Última revisión de parámetros ya aplicada.
    pub param_revision: u64,
    /// Comandos de parámetros encolados pendientes de la próxima frontera de tick.
    pub pending_param_commands: usize,
    /// Total histórico de rechazos de comandos de parámetros.
    pub param_rejection_count: u64,
    /// Últimos rechazos de parámetros (máx. [`crate::params::PARAM_REJECTION_LOG_CAP`]).
    pub recent_param_rejections: Vec<ParamRejectionRecord>,
}

/// Único punto de entrada mutante.
pub trait ControlPort {
    fn submit(&self, command: ControlCommand) -> Ack;
}

/// Sólo lectura: devuelve valores propios, nunca referencias al estado.
pub trait ObservationPort {
    fn observe(&self) -> Observation;
}

/// Error tipado: la sesión se ofrece en sólo lectura.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlUnavailable {
    ReadOnlySession { kind: SessionKind },
}

impl fmt::Display for ControlUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadOnlySession { kind } => {
                write!(f, "{kind:?} session has no control capability (read-only)")
            }
        }
    }
}

impl std::error::Error for ControlUnavailable {}

/// Opciones de sesión.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionOptions {
    pub kind: SessionKind,
    /// `None` = por defecto del tipo (`Simulated` controlable; resto sólo lectura).
    pub control_capability: Option<bool>,
    /// Opt-in: cerrar el visor detiene el motor. Por defecto `false`.
    pub stop_on_close: bool,
}

impl SessionOptions {
    pub const fn simulated() -> Self {
        Self {
            kind: SessionKind::Simulated,
            control_capability: None,
            stop_on_close: false,
        }
    }
    pub const fn of_kind(kind: SessionKind) -> Self {
        Self {
            kind,
            control_capability: None,
            stop_on_close: false,
        }
    }
    pub const fn controllable(&self) -> bool {
        match self.control_capability {
            Some(c) => c,
            None => matches!(self.kind, SessionKind::Simulated),
        }
    }
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self::simulated()
    }
}

struct Inner {
    session: SessionId,
    options: SessionOptions,
    revision: u64,
    tick: Tick,
    run_state: RunState,
    speed: f64,
    /// Ticks fraccionarios acumulados por `pump` con velocidad no entera.
    carry: f64,
    /// Ticks debidos y no ejecutados por el tope de `pump` (GOG-148).
    backlog: u64,
    max_ticks_per_pump: u64,
    viewer_attached: bool,
    seen: DedupWindow,
    rejection_count: u64,
    rejections: VecDeque<RejectionRecord>,
    params: ParamState,
}

impl Inner {
    fn ack(&self, id: CommandId, outcome: AckOutcome) -> Ack {
        Ack {
            id,
            outcome,
            revision: self.revision,
            tick: self.tick,
        }
    }

    fn reject(&mut self, id: CommandId, reason: RejectReason) -> Ack {
        self.rejection_count += 1;
        if self.rejections.len() == REJECTION_LOG_CAP {
            self.rejections.pop_front();
        }
        self.rejections.push_back(RejectionRecord { id, reason });
        self.ack(id, AckOutcome::Rejected { reason })
    }

    fn apply(&mut self, c: ControlCommand) -> Ack {
        // Orden: sesión obsoleta, duplicado, motor detenido, validez. Los obsoletos no consumen id.
        if c.session != self.session {
            return self.reject(
                c.id,
                RejectReason::StaleSession {
                    current: self.session,
                },
            );
        }
        match self.seen.check_and_insert(c.id) {
            DedupVerdict::New => {}
            DedupVerdict::Duplicate => {
                return self.reject(c.id, RejectReason::DuplicateCommandId);
            }
            DedupVerdict::Expired => return self.reject(c.id, RejectReason::ExpiredCommandId),
        }
        if self.run_state == RunState::Stopped {
            return self.reject(c.id, RejectReason::EngineStopped);
        }
        match c.action {
            ControlAction::Pause => self.run_state = RunState::Paused,
            ControlAction::Resume => self.run_state = RunState::Running,
            ControlAction::Step => {
                if self.run_state == RunState::Running {
                    return self.reject(c.id, RejectReason::StepWhileRunning);
                }
                match self.tick.next() {
                    Ok(t) => {
                        self.tick = t;
                        self.params.apply_boundary(t);
                    }
                    Err(_) => return self.reject(c.id, RejectReason::TickOverflow),
                }
            }
            ControlAction::Reset => {
                self.tick = Tick::ZERO;
                self.run_state = RunState::Paused;
                self.carry = 0.0;
                self.backlog = 0;
                self.session.generation += 1;
                self.seen.clear();
                self.params.reset();
            }
            ControlAction::SetSpeed(s) => {
                if !s.is_finite() || s <= 0.0 || s > MAX_SPEED {
                    return self.reject(c.id, RejectReason::InvalidSpeed);
                }
                self.speed = s;
            }
        }
        self.revision += 1;
        self.ack(c.id, AckOutcome::Applied)
    }

    fn param_ack(&self, id: CommandId, outcome: ParamOutcome) -> ParamAck {
        ParamAck {
            id,
            outcome,
            param_revision: self.params.applied_revision(),
            tick: self.tick,
        }
    }

    fn param_reject(&mut self, id: CommandId, reason: ParamRejectReason) -> ParamAck {
        self.params.reject(id, reason.clone());
        self.param_ack(id, ParamOutcome::Rejected { reason })
    }

    /// Valida todo antes de encolar (atómico); se aplica en la siguiente frontera de tick.
    fn apply_params(&mut self, c: ParamCommand) -> ParamAck {
        if c.session != self.session {
            let r = ParamRejectReason::Session(RejectReason::StaleSession {
                current: self.session,
            });
            return self.param_reject(c.id, r);
        }
        if !self.options.controllable() {
            return self.param_reject(c.id, ParamRejectReason::ReadOnlySession);
        }
        match self.seen.check_and_insert(c.id) {
            DedupVerdict::New => {}
            DedupVerdict::Duplicate => {
                let r = ParamRejectReason::Session(RejectReason::DuplicateCommandId);
                return self.param_reject(c.id, r);
            }
            DedupVerdict::Expired => {
                let r = ParamRejectReason::Session(RejectReason::ExpiredCommandId);
                return self.param_reject(c.id, r);
            }
        }
        if self.run_state == RunState::Stopped {
            let r = ParamRejectReason::Session(RejectReason::EngineStopped);
            return self.param_reject(c.id, r);
        }
        let entries = match self.params.validate(&c.request) {
            Ok(e) => e,
            Err(r) => return self.param_reject(c.id, r),
        };
        let applies_at = match self.tick.next() {
            Ok(t) => t,
            Err(_) => {
                let r = ParamRejectReason::Session(RejectReason::TickOverflow);
                return self.param_reject(c.id, r);
            }
        };
        let revision = self.params.enqueue(c.id, entries);
        self.param_ack(
            c.id,
            ParamOutcome::Queued {
                revision,
                applies_at,
            },
        )
    }
}

/// Handle del lado motor: avanza el tiempo y gestiona el ciclo de vida. Clonable (comparte
/// estado). La UI nunca lo recibe, sólo los puertos.
#[derive(Clone)]
pub struct SessionHost {
    inner: Arc<Mutex<Inner>>,
}

impl SessionHost {
    pub fn new(epoch: u64, options: SessionOptions) -> Self {
        Self::with_params(epoch, options, ParamSchema::default())
    }

    /// Sesión con esquema de parámetros dinámicos (valores iniciales = `default` de cada uno).
    pub fn with_params(epoch: u64, options: SessionOptions, schema: ParamSchema) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                session: SessionId {
                    epoch,
                    generation: 0,
                },
                options,
                revision: 0,
                tick: Tick::ZERO,
                run_state: RunState::Paused,
                speed: 1.0,
                carry: 0.0,
                backlog: 0,
                max_ticks_per_pump: DEFAULT_MAX_TICKS_PER_PUMP,
                viewer_attached: true,
                seen: DedupWindow::new(DEFAULT_DEDUP_WINDOW),
                rejection_count: 0,
                rejections: VecDeque::new(),
                params: ParamState::new(schema),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Cambia el tope de ticks ejecutados por llamada a `pump` (mínimo 1).
    /// Capacidad de la ventana de deduplicación (mín. 1); reduce expulsando los más antiguos.
    pub fn set_dedup_window(&self, capacity: usize) {
        self.lock().seen.set_capacity(capacity);
    }

    /// Ids actualmente recordados por la ventana (siempre <= capacidad).
    pub fn dedup_len(&self) -> usize {
        self.lock().seen.len()
    }

    pub fn set_max_ticks_per_pump(&self, max: u64) {
        self.lock().max_ticks_per_pump = max.max(1);
    }

    /// Avanza `units` unidades de tiempo virtual: si está `Running` suma `floor(carry + speed *
    /// units)` ticks (fracción acumulada). Devuelve los ticks avanzados. No cambia `revision`.
    ///
    /// Semántica (contrato v2, GOG-148): se ejecutan como máximo `max_ticks_per_pump` ticks por
    /// llamada (defecto [`DEFAULT_MAX_TICKS_PER_PUMP`]) para no congelar el mutex. El resto queda
    /// como backlog (no se pierde ni se rechaza) y se drena en llamadas posteriores, también con
    /// `units == 0`; use [`Self::pump_bounded`] para ver `pending`/`truncated`. Con valores
    /// normales (por debajo del tope) el comportamiento es idéntico al anterior.
    pub fn pump(&self, units: u64) -> u64 {
        self.pump_bounded(units).executed
    }

    /// Como [`Self::pump`] pero informa de `executed`, `pending`, `truncated` y `dropped`.
    pub fn pump_bounded(&self, units: u64) -> PumpOutcome {
        let mut g = self.lock();
        if g.run_state != RunState::Running {
            return PumpOutcome {
                executed: 0,
                pending: g.backlog,
                truncated: false,
                dropped: 0,
            };
        }
        let total = g.carry + g.speed * units as f64;
        let whole = total.floor();
        g.carry = total - whole;
        let due = g.backlog.saturating_add(whole as u64);
        let dropped = due.saturating_sub(MAX_PUMP_BACKLOG);
        let due = due - dropped;
        let run = due.min(g.max_ticks_per_pump);
        let mut advanced = 0u64;
        let mut overflow = false;
        for _ in 0..run {
            match g.tick.next() {
                Ok(t) => {
                    g.tick = t;
                    g.params.apply_boundary(t);
                    advanced += 1;
                }
                Err(_) => {
                    overflow = true;
                    break;
                }
            }
        }
        g.backlog = if overflow { 0 } else { due - advanced };
        PumpOutcome {
            executed: advanced,
            pending: g.backlog,
            truncated: g.backlog > 0,
            dropped,
        }
    }

    /// El visor se cierra. El motor sigue como estaba salvo `stop_on_close` (opt-in).
    pub fn close_viewer(&self) {
        let mut g = self.lock();
        g.viewer_attached = false;
        if g.options.stop_on_close && g.run_state != RunState::Stopped {
            g.run_state = RunState::Stopped;
            g.revision += 1;
        }
    }

    pub fn observation_port(&self) -> ObservationHandle {
        ObservationHandle {
            inner: self.inner.clone(),
        }
    }

    /// Puerto de parámetros: siempre disponible; en sesiones de sólo lectura rechaza con
    /// `ReadOnlySession` (visible en la observación).
    pub fn param_port(&self) -> ParamHandle {
        ParamHandle {
            inner: self.inner.clone(),
        }
    }

    /// `Err` si la sesión no tiene capacidad de control (se ofrece en sólo lectura).
    pub fn control_port(&self) -> Result<ControlHandle, ControlUnavailable> {
        let g = self.lock();
        if g.options.controllable() {
            Ok(ControlHandle {
                inner: self.inner.clone(),
            })
        } else {
            Err(ControlUnavailable::ReadOnlySession {
                kind: g.options.kind,
            })
        }
    }
}

/// Implementación de [`ControlPort`].
#[derive(Clone)]
pub struct ControlHandle {
    inner: Arc<Mutex<Inner>>,
}

impl ControlPort for ControlHandle {
    fn submit(&self, command: ControlCommand) -> Ack {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .apply(command)
    }
}

/// Implementación de [`ParamControlPort`].
#[derive(Clone)]
pub struct ParamHandle {
    inner: Arc<Mutex<Inner>>,
}

impl ParamControlPort for ParamHandle {
    fn submit_params(&self, command: ParamCommand) -> ParamAck {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .apply_params(command)
    }
}

/// Implementación de [`ObservationPort`].
#[derive(Clone)]
pub struct ObservationHandle {
    inner: Arc<Mutex<Inner>>,
}

impl ObservationHandle {
    /// Esquema publicado de parámetros.
    pub fn param_schema(&self) -> ParamSchema {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .params
            .schema()
    }

    /// Log determinista de eventos de parámetros (de la generación actual).
    pub fn param_events(&self) -> Vec<ParamEvent> {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .params
            .events()
    }
}

impl ObservationPort for ObservationHandle {
    fn observe(&self) -> Observation {
        let g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        Observation {
            session: g.session,
            kind: g.options.kind,
            revision: g.revision,
            tick: g.tick,
            run_state: g.run_state,
            speed: g.speed,
            viewer_attached: g.viewer_attached,
            controllable: g.options.controllable(),
            rejection_count: g.rejection_count,
            recent_rejections: g.rejections.iter().copied().collect(),
            params: g.params.values(),
            param_revision: g.params.applied_revision(),
            pending_param_commands: g.params.pending_len(),
            param_rejection_count: g.params.rejection_count(),
            recent_param_rejections: g.params.rejections(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(id: u64, session: SessionId, action: ControlAction) -> ControlCommand {
        ControlCommand {
            id: CommandId(id),
            session,
            action,
        }
    }

    fn setup() -> (SessionHost, ControlHandle, ObservationHandle) {
        let h = SessionHost::new(1, SessionOptions::simulated());
        let c = h.control_port().unwrap();
        let o = h.observation_port();
        (h, c, o)
    }

    #[test]
    fn step_advances_exactly_one_tick_when_paused_and_acks_revision_and_tick() {
        let (_h, c, o) = setup();
        let s = o.observe().session;
        let a1 = c.submit(cmd(1, s, ControlAction::Step));
        assert_eq!(a1.outcome, AckOutcome::Applied);
        assert_eq!((a1.tick, a1.revision), (Tick::new(1), 1));
        let a2 = c.submit(cmd(2, s, ControlAction::Step));
        assert_eq!((a2.tick, a2.revision), (Tick::new(2), 2));
        let obs = o.observe();
        assert_eq!(obs.tick, Tick::new(2));
        assert_eq!(obs.run_state, RunState::Paused);
    }

    #[test]
    fn pause_resume_ack_and_running_advances_by_speed() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        let a = c.submit(cmd(1, s, ControlAction::Resume));
        assert_eq!((a.outcome, a.revision), (AckOutcome::Applied, 1));
        assert_eq!(
            c.submit(cmd(2, s, ControlAction::SetSpeed(2.5))).revision,
            2
        );
        assert_eq!(h.pump(2), 5);
        assert_eq!(h.pump(1), 2); // 2.5 -> 2 + carry .5
        assert_eq!(h.pump(1), 3); // .5 + 2.5 = 3
        let p = c.submit(cmd(3, s, ControlAction::Pause));
        assert_eq!((p.tick, p.revision), (Tick::new(10), 3));
        assert_eq!(h.pump(100), 0, "paused: no advance");
        assert_eq!(o.observe().tick, Tick::new(10));
    }

    #[test]
    fn step_while_running_is_rejected_without_advancing() {
        let (_h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        let before = o.observe();
        let a = c.submit(cmd(2, s, ControlAction::Step));
        assert_eq!(
            a.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::StepWhileRunning
            }
        );
        let after = o.observe();
        assert_eq!((after.tick, after.revision), (before.tick, before.revision));
    }

    #[test]
    fn reset_zeroes_tick_pauses_and_bumps_generation() {
        let (_h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Step));
        c.submit(cmd(2, s, ControlAction::Resume));
        let a = c.submit(cmd(3, s, ControlAction::Reset));
        assert_eq!((a.outcome, a.tick), (AckOutcome::Applied, Tick::ZERO));
        let obs = o.observe();
        assert_eq!(obs.run_state, RunState::Paused);
        assert_eq!(obs.session.generation, s.generation + 1);
        assert_eq!(obs.session.epoch, s.epoch);
        // comando con la generación anterior queda obsoleto
        let stale = c.submit(cmd(4, s, ControlAction::Step));
        assert_eq!(
            stale.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::StaleSession {
                    current: obs.session
                }
            }
        );
        assert_eq!(o.observe().tick, Tick::ZERO);
        // ids se pueden reutilizar tras reset con la nueva generación
        let ok = c.submit(cmd(1, obs.session, ControlAction::Step));
        assert_eq!(ok.outcome, AckOutcome::Applied);
    }

    #[test]
    fn invalid_duplicate_and_stale_do_not_alter_state_and_are_visible() {
        let (_h, c, o) = setup();
        let s = o.observe().session;
        assert_eq!(
            c.submit(cmd(1, s, ControlAction::Step)).outcome,
            AckOutcome::Applied
        );
        let base = o.observe();
        let bad_speeds = [f64::NAN, f64::INFINITY, 0.0, -1.0, MAX_SPEED + 1.0];
        let mut expected = Vec::new();
        for (i, sp) in bad_speeds.iter().enumerate() {
            let id = 10 + i as u64;
            let a = c.submit(cmd(id, s, ControlAction::SetSpeed(*sp)));
            assert_eq!(
                a.outcome,
                AckOutcome::Rejected {
                    reason: RejectReason::InvalidSpeed
                }
            );
            assert_eq!((a.revision, a.tick), (base.revision, base.tick));
            expected.push(RejectionRecord {
                id: CommandId(id),
                reason: RejectReason::InvalidSpeed,
            });
        }
        let dup = c.submit(cmd(1, s, ControlAction::Step));
        assert_eq!(
            dup.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::DuplicateCommandId
            }
        );
        expected.push(RejectionRecord {
            id: CommandId(1),
            reason: RejectReason::DuplicateCommandId,
        });
        let other = SessionId {
            epoch: 99,
            generation: 0,
        };
        let st = c.submit(cmd(50, other, ControlAction::Reset));
        assert!(matches!(
            st.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::StaleSession { .. }
            }
        ));
        expected.push(RejectionRecord {
            id: CommandId(50),
            reason: RejectReason::StaleSession { current: s },
        });
        let after = o.observe();
        assert_eq!(
            (
                after.tick,
                after.revision,
                after.run_state,
                after.speed,
                after.session
            ),
            (
                base.tick,
                base.revision,
                base.run_state,
                base.speed,
                base.session
            )
        );
        assert_eq!(after.rejection_count, 7);
        assert_eq!(after.recent_rejections, expected);
    }

    #[test]
    fn rejection_log_is_bounded_but_count_is_total() {
        let (_h, c, o) = setup();
        let s = o.observe().session;
        for i in 0..(REJECTION_LOG_CAP as u64 + 10) {
            c.submit(cmd(i, s, ControlAction::SetSpeed(-1.0)));
        }
        let obs = o.observe();
        assert_eq!(obs.rejection_count, REJECTION_LOG_CAP as u64 + 10);
        assert_eq!(obs.recent_rejections.len(), REJECTION_LOG_CAP);
        assert_eq!(obs.recent_rejections[0].id, CommandId(10));
    }

    #[test]
    fn physical_and_remote_sessions_are_read_only_but_observable() {
        for kind in [SessionKind::Physical, SessionKind::Remote] {
            let h = SessionHost::new(7, SessionOptions::of_kind(kind));
            assert!(matches!(
                h.control_port(),
                Err(ControlUnavailable::ReadOnlySession { kind: k }) if k == kind
            ));
            let obs = h.observation_port().observe();
            assert_eq!(obs.kind, kind);
            assert!(!obs.controllable);
        }
        // capacidad explícita habilita el control
        let mut o = SessionOptions::of_kind(SessionKind::Remote);
        o.control_capability = Some(true);
        assert!(SessionHost::new(8, o).control_port().is_ok());
    }

    #[test]
    fn closing_viewer_keeps_engine_running_by_default() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        h.close_viewer();
        let obs = o.observe();
        assert!(!obs.viewer_attached);
        assert_eq!(obs.run_state, RunState::Running);
        assert_eq!(h.pump(3), 3, "engine keeps advancing");
    }

    #[test]
    fn stop_on_close_opt_in_stops_engine_and_rejects_further_commands() {
        let mut opts = SessionOptions::simulated();
        opts.stop_on_close = true;
        let h = SessionHost::new(1, opts);
        let c = h.control_port().unwrap();
        let o = h.observation_port();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        h.close_viewer();
        assert_eq!(o.observe().run_state, RunState::Stopped);
        assert_eq!(h.pump(5), 0);
        let a = c.submit(cmd(2, s, ControlAction::Step));
        assert_eq!(
            a.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::EngineStopped
            }
        );
        assert_eq!(o.observe().tick, Tick::ZERO);
    }

    #[test]
    fn gog_148_extreme_units_is_bounded_and_leaves_backlog() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        h.set_max_ticks_per_pump(1000);
        let out = h.pump_bounded(u64::MAX);
        assert_eq!(out.executed, 1000);
        assert!(out.truncated);
        assert!(out.pending > 0);
        assert_eq!(o.observe().tick, Tick::new(1000));
        assert_eq!(h.pump(0), 1000, "backlog drains on later calls");
        assert_eq!(PUMP_CONTRACT_VERSION, 2);
    }

    #[test]
    fn gog_148_normal_units_unchanged_and_backlog_conserved() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        h.set_max_ticks_per_pump(10);
        let out = h.pump_bounded(25);
        assert_eq!((out.executed, out.pending, out.truncated), (10, 15, true));
        assert_eq!(h.pump(0), 10);
        let out = h.pump_bounded(0);
        assert_eq!((out.executed, out.pending, out.truncated), (5, 0, false));
        c.submit(cmd(2, s, ControlAction::Reset));
        assert_eq!(h.pump_bounded(0).pending, 0);
    }

    #[test]
    fn gog_148_observe_and_control_progress_during_huge_pump() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};
        let (h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Resume));
        let done = Arc::new(AtomicBool::new(false));
        let d2 = done.clone();
        let h2 = h.clone();
        let t0 = Instant::now();
        let pumper = std::thread::spawn(move || {
            for _ in 0..50 {
                h2.pump_bounded(u64::MAX);
            }
            d2.store(true, Ordering::SeqCst);
        });
        let mut rounds = 0u64;
        let mut next_id = 100;
        while !done.load(Ordering::SeqCst) {
            let _ = o.observe();
            let _ = c.submit(cmd(next_id, s, ControlAction::SetSpeed(1.0)));
            next_id += 1;
            rounds += 1;
            assert!(t0.elapsed() < Duration::from_secs(60), "host frozen");
        }
        pumper.join().unwrap();
        assert!(rounds >= 1);
        assert!(t0.elapsed() < Duration::from_secs(60));
    }

    #[test]
    fn gog_141_dedup_window_stays_bounded_over_100k_commands() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        h.set_dedup_window(256);
        for i in 0..100_500u64 {
            c.submit(cmd(i, s, ControlAction::SetSpeed(1.0)));
            assert!(h.dedup_len() <= 256);
        }
        assert_eq!(h.dedup_len(), 256);
    }

    #[test]
    fn gog_141_duplicate_in_window_rejected_and_old_id_expired() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        h.set_dedup_window(4);
        for i in 1..=10u64 {
            c.submit(cmd(i, s, ControlAction::SetSpeed(1.0)));
        }
        let dup = c.submit(cmd(10, s, ControlAction::SetSpeed(1.0)));
        assert_eq!(
            dup.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::DuplicateCommandId
            }
        );
        let old = c.submit(cmd(1, s, ControlAction::SetSpeed(1.0)));
        assert_eq!(
            old.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::ExpiredCommandId
            }
        );
        let fresh = c.submit(cmd(11, s, ControlAction::SetSpeed(1.0)));
        assert_eq!(fresh.outcome, AckOutcome::Applied);
    }

    #[test]
    fn gog_141_reset_clears_window_and_rejects_previous_generation() {
        let (h, c, o) = setup();
        let s = o.observe().session;
        c.submit(cmd(1, s, ControlAction::Step));
        c.submit(cmd(2, s, ControlAction::Reset));
        assert_eq!(h.dedup_len(), 0);
        let stale = c.submit(cmd(3, s, ControlAction::Step));
        assert!(matches!(
            stale.outcome,
            AckOutcome::Rejected {
                reason: RejectReason::StaleSession { .. }
            }
        ));
        let s2 = o.observe().session;
        assert_eq!(s2.generation, s.generation + 1);
        assert_eq!(
            c.submit(cmd(1, s2, ControlAction::Step)).outcome,
            AckOutcome::Applied
        );
        assert_eq!(
            c.submit(cmd(1, s2, ControlAction::Step)).outcome,
            AckOutcome::Rejected {
                reason: RejectReason::DuplicateCommandId
            }
        );
    }
}
