//! Servidor de protocolo residente JSON-Lines para el SDK Python de Gogamen.
//!
//! Mantiene las sesiones [`GridWorldEnv`] residentes en Rust y expone operaciones
//! ligeras (`create_session`, `step`, `events`, `reset`, `close_session`) por `stdin`/`stdout`
//! sin transferir snapshots completos por paso.
//!
//! Cuotas (GOG-139, [`ProtocolQuotas`]): sesiones max 64 (`QuotaExceeded` al crear), eventos
//! retenidos por sesion max 1024 / 256 KiB (ring buffer, `seq` monotonico). `events` acepta
//! `max_count` (>=1, tope 1000 por respuesta) y devuelve ademas `next_seq`, `first_seq`,
//! `gap`, `dropped` y `has_more` (campos aditivos).

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::io::{self, BufRead, Read, Write};

use gogamen_core::{
    Action, Actuator, DiscreteAction, DomainError, EnvironmentError, EnvironmentObserver,
    LiveEnvironment, Terminal, TerminalReason,
};
use gogamen_lab_grid::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldEnv};
use gogamen_spec::{
    JsonLimits, JsonNumber, JsonValue, parse_with_limits, write_canonical_f64, write_json_string,
};

/// Extensiones de acceso y serialización sobre el `JsonValue` canónico de `gogamen-spec`.
trait JsonValueExt {
    fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>>;
    fn as_str(&self) -> Option<&str>;
    fn as_bool(&self) -> Option<bool>;
    fn as_i64(&self) -> Option<i64>;
    fn to_json_string(&self) -> String;
    fn write_to(&self, out: &mut String);
}

impl JsonValueExt for JsonValue {
    fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            Self::Object(map) => Some(map),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Number(JsonNumber::I64(v)) => Some(*v),
            Self::Number(JsonNumber::U64(v)) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    fn to_json_string(&self) -> String {
        let mut out = String::new();
        self.write_to(&mut out);
        out
    }

    fn write_to(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(JsonNumber::I64(v)) => {
                let _ = write!(out, "{v}");
            }
            Self::Number(JsonNumber::U64(v)) => {
                let _ = write!(out, "{v}");
            }
            Self::Number(JsonNumber::F64(v)) => {
                // Un float no finito no es JSON válido: se emite `null` en vez de `NaN`/`inf`.
                if write_canonical_f64(out, *v).is_err() {
                    out.push_str("null");
                }
            }
            Self::String(s) => write_json_string(out, s),
            Self::Array(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_to(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    write_json_string(out, k);
                    out.push(':');
                    v.write_to(out);
                }
                out.push('}');
            }
        }
    }
}

#[cfg(test)]
fn parse_json_line(input: &str) -> Result<JsonValue, String> {
    parse_with_limits(input, JsonLimits::default()).map_err(|err| err.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolErrorCode {
    InvalidRequest,
    UnknownSession,
    DomainError,
    Terminated,
    QuotaExceeded,
    InternalError,
}

impl ProtocolErrorCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "InvalidRequest",
            Self::UnknownSession => "UnknownSession",
            Self::DomainError => "DomainError",
            Self::Terminated => "Terminated",
            Self::QuotaExceeded => "QuotaExceeded",
            Self::InternalError => "InternalError",
        }
    }
}

#[derive(Debug, Clone)]
struct ProtocolEvent {
    seq: u64,
    tick: u64,
    kind: &'static str,
    detail: String,
}

impl ProtocolEvent {
    fn to_json(&self) -> JsonValue {
        let mut map = BTreeMap::new();
        map.insert(
            "seq".to_owned(),
            JsonValue::Number(JsonNumber::I64(self.seq as i64)),
        );
        map.insert(
            "tick".to_owned(),
            JsonValue::Number(JsonNumber::I64(self.tick as i64)),
        );
        map.insert("kind".to_owned(), JsonValue::String(self.kind.to_owned()));
        map.insert("detail".to_owned(), JsonValue::String(self.detail.clone()));
        JsonValue::Object(map)
    }
}

/// Sobrecarga fija estimada por evento retenido (campos numericos, estructura).
const EVENT_OVERHEAD_BYTES: usize = 32;

/// Cuotas de recursos del servidor de protocolo (GOG-139).
///
/// - `max_sessions`: sesiones simultaneas; `create_session` por encima devuelve `QuotaExceeded`.
/// - `max_events_per_session`: eventos retenidos por sesion (ring buffer; se descartan los mas antiguos).
/// - `max_event_bytes_per_session`: bytes estimados de eventos retenidos por sesion.
/// - `max_events_per_response`: tope de eventos por respuesta de `events`.
///
/// Un valor `0` en cualquier campo se normaliza a `1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolQuotas {
    pub max_sessions: usize,
    pub max_events_per_session: usize,
    pub max_event_bytes_per_session: usize,
    pub max_events_per_response: usize,
}

impl Default for ProtocolQuotas {
    fn default() -> Self {
        Self {
            max_sessions: 64,
            max_events_per_session: 1024,
            max_event_bytes_per_session: 256 * 1024,
            max_events_per_response: 1000,
        }
    }
}

impl ProtocolQuotas {
    fn normalized(self) -> Self {
        Self {
            max_sessions: self.max_sessions.max(1),
            max_events_per_session: self.max_events_per_session.max(1),
            max_event_bytes_per_session: self.max_event_bytes_per_session.max(1),
            max_events_per_response: self.max_events_per_response.max(1),
        }
    }
}

impl ProtocolEvent {
    fn approx_bytes(&self) -> usize {
        EVENT_OVERHEAD_BYTES + self.kind.len() + self.detail.len()
    }
}

struct GridSession {
    env: GridWorldEnv,
    next_event_seq: u64,
    events: VecDeque<ProtocolEvent>,
    event_bytes: usize,
}

impl GridSession {
    fn new(env: GridWorldEnv, quotas: &ProtocolQuotas) -> Self {
        let mut session = Self {
            env,
            next_event_seq: 1,
            events: VecDeque::new(),
            event_bytes: 0,
        };
        session.push_event(
            quotas,
            0,
            "session_created",
            "grid session initialized".to_owned(),
        );
        session
    }

    /// Primer `seq` aun disponible (o el siguiente a asignar si no hay eventos retenidos).
    fn first_seq(&self) -> u64 {
        self.events.front().map_or(self.next_event_seq, |e| e.seq)
    }

    fn push_event(
        &mut self,
        quotas: &ProtocolQuotas,
        tick: u64,
        kind: &'static str,
        detail: String,
    ) -> ProtocolEvent {
        let ev = ProtocolEvent {
            seq: self.next_event_seq,
            tick,
            kind,
            detail,
        };
        self.next_event_seq += 1;
        self.event_bytes += ev.approx_bytes();
        self.events.push_back(ev.clone());
        // Ring buffer: descarta los mas antiguos (siempre conserva el ultimo evento).
        while self.events.len() > 1
            && (self.events.len() > quotas.max_events_per_session
                || self.event_bytes > quotas.max_event_bytes_per_session)
        {
            if let Some(old) = self.events.pop_front() {
                self.event_bytes -= old.approx_bytes();
            }
        }
        ev
    }
}

/// Servidor residente del protocolo JSON-Lines.
#[derive(Default)]
pub struct ProtocolServer {
    next_session_num: u64,
    sessions: BTreeMap<String, GridSession>,
    limits: JsonLimits,
    quotas: ProtocolQuotas,
}

/// Resultado de leer una línea acotada del flujo de entrada.
enum BoundedLine {
    Eof,
    Line(String),
    TooLong,
}

/// Lee una línea de como máximo `max_bytes` (sin contar el salto de línea) sin asignar más:
/// si se excede, descarta el resto de la línea en trozos del buffer y devuelve `TooLong`.
fn read_bounded_line<R: BufRead>(reader: &mut R, max_bytes: usize) -> io::Result<BoundedLine> {
    let mut buf = Vec::new();
    let limit = (max_bytes as u64).saturating_add(2); // +1 de '\n', +1 para detectar exceso
    let n = Read::by_ref(reader)
        .take(limit)
        .read_until(b'\n', &mut buf)?;
    if n == 0 {
        return Ok(BoundedLine::Eof);
    }
    let terminated = buf.last() == Some(&b'\n');
    if terminated {
        buf.pop();
    }
    if buf.len() > max_bytes {
        let mut done = terminated;
        while !done {
            let chunk = reader.fill_buf()?;
            if chunk.is_empty() {
                break;
            }
            match chunk.iter().position(|b| *b == b'\n') {
                Some(i) => {
                    reader.consume(i + 1);
                    done = true;
                }
                None => {
                    let len = chunk.len();
                    reader.consume(len);
                }
            }
        }
        return Ok(BoundedLine::TooLong);
    }
    Ok(BoundedLine::Line(
        String::from_utf8_lossy(&buf).into_owned(),
    ))
}

impl ProtocolServer {
    /// Crea un nuevo servidor de protocolo sin sesiones activas.
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Crea un servidor con límites JSON explícitos (el tope de bytes aplica por línea).
    pub fn with_limits(limits: JsonLimits) -> Self {
        Self {
            limits,
            ..Self::default()
        }
    }

    /// Crea un servidor con cuotas de sesiones/eventos explicitas (limites JSON por defecto).
    #[allow(dead_code)]
    pub fn with_quotas(quotas: ProtocolQuotas) -> Self {
        Self {
            quotas: quotas.normalized(),
            ..Self::default()
        }
    }

    /// Ejecuta el bucle de lectura por líneas hasta alcanzar EOF.
    ///
    /// Cada línea se lee con tope `limits.max_bytes`; una línea mayor se rechaza con
    /// `invalid_request` sin acumularla en memoria.
    pub fn serve<R: BufRead, W: Write>(&mut self, mut reader: R, mut writer: W) -> io::Result<()> {
        loop {
            let line = match read_bounded_line(&mut reader, self.limits.max_bytes)? {
                BoundedLine::Eof => break,
                BoundedLine::TooLong => {
                    let msg = format!(
                        "request line exceeds maximum of {} bytes",
                        self.limits.max_bytes
                    );
                    let response =
                        error_response(JsonValue::Null, ProtocolErrorCode::InvalidRequest, &msg);
                    writeln!(writer, "{response}")?;
                    writer.flush()?;
                    continue;
                }
                BoundedLine::Line(line) => line,
            };
            if line.trim().is_empty() {
                continue;
            }
            let response = self.handle_line(&line);
            writeln!(writer, "{response}")?;
            writer.flush()?;
        }
        Ok(())
    }

    /// Procesa una línea JSON de petición y devuelve la línea JSON de respuesta.
    pub fn handle_line(&mut self, line: &str) -> String {
        let parsed = match parse_with_limits(line, self.limits).map_err(|e| e.to_string()) {
            Ok(val) => val,
            Err(err) => {
                return error_response(
                    JsonValue::Null,
                    ProtocolErrorCode::InvalidRequest,
                    &format!("malformed JSON request: {err}"),
                );
            }
        };

        let Some(req) = parsed.as_object() else {
            return error_response(
                JsonValue::Null,
                ProtocolErrorCode::InvalidRequest,
                "request root must be a JSON object",
            );
        };

        let req_id = req.get("id").cloned().unwrap_or(JsonValue::Null);
        let Some(op) = req.get("op").and_then(JsonValue::as_str) else {
            return error_response(
                req_id,
                ProtocolErrorCode::InvalidRequest,
                "missing or non-string 'op' field",
            );
        };

        match self.dispatch_op(op, req) {
            Ok(mut payload) => {
                payload.insert("ok".to_owned(), JsonValue::Bool(true));
                payload.insert("id".to_owned(), req_id);
                JsonValue::Object(payload).to_json_string()
            }
            Err((code, msg)) => error_response(req_id, code, &msg),
        }
    }

    fn dispatch_op(
        &mut self,
        op: &str,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        match op {
            "create_session" => self.op_create_session(req),
            "step" => self.op_step(req),
            "events" => self.op_events(req),
            "reset" => self.op_reset(req),
            "close_session" => self.op_close_session(req),
            other => Err((
                ProtocolErrorCode::InvalidRequest,
                format!("unknown protocol op '{other}'"),
            )),
        }
    }

    fn op_create_session(
        &mut self,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        let world = match req.get("world") {
            None => "grid",
            Some(v) => v.as_str().ok_or((
                ProtocolErrorCode::InvalidRequest,
                "'world' must be a string".to_owned(),
            ))?,
        };
        if world != "grid" {
            return Err((
                ProtocolErrorCode::InvalidRequest,
                format!("unsupported world '{world}'; only 'grid' is supported"),
            ));
        }

        let partial_obs = match req.get("partial_obs") {
            None => false,
            Some(v) => v.as_bool().ok_or((
                ProtocolErrorCode::InvalidRequest,
                "'partial_obs' must be a boolean".to_owned(),
            ))?,
        };

        let partial_radius = match req.get("partial_radius") {
            None => 2usize,
            Some(v) => {
                let raw = v.as_i64().ok_or((
                    ProtocolErrorCode::InvalidRequest,
                    "'partial_radius' must be a non-negative integer".to_owned(),
                ))?;
                if raw < 0 {
                    return Err((
                        ProtocolErrorCode::InvalidRequest,
                        "'partial_radius' must be >= 0".to_owned(),
                    ));
                }
                raw as usize
            }
        };

        let config = GridWorldConfig::default().with_partial_observation(if partial_obs {
            Some(partial_radius)
        } else {
            None
        });
        let env = GridWorldEnv::new(config).map_err(map_domain_err)?;
        let obs = EnvironmentObserver::observe(&env).map_err(|e| {
            (
                ProtocolErrorCode::InternalError,
                format!("failed to observe initial grid state: {e}"),
            )
        })?;

        if self.sessions.len() >= self.quotas.max_sessions {
            return Err((
                ProtocolErrorCode::QuotaExceeded,
                format!(
                    "session quota exceeded (max_sessions={})",
                    self.quotas.max_sessions
                ),
            ));
        }
        self.next_session_num += 1;
        let session_id = format!("sess-{}", self.next_session_num);
        let session = GridSession::new(env, &self.quotas);
        let events: Vec<JsonValue> = session.events.iter().map(ProtocolEvent::to_json).collect();
        self.sessions.insert(session_id.clone(), session);

        let mut out = BTreeMap::new();
        out.insert("session_id".to_owned(), JsonValue::String(session_id));
        out.insert("world".to_owned(), JsonValue::String("grid".to_owned()));
        out.insert(
            "action_cardinality".to_owned(),
            JsonValue::Number(JsonNumber::I64(GRID_ACTION_CARDINALITY as i64)),
        );
        out.insert("tick".to_owned(), JsonValue::Number(JsonNumber::I64(0)));
        out.insert(
            "elapsed_s".to_owned(),
            JsonValue::Number(JsonNumber::F64(0.0)),
        );
        out.insert("observation".to_owned(), features_to_json(obs.features()));
        out.insert("terminal".to_owned(), terminal_to_json(Terminal::Ongoing));
        out.insert("events".to_owned(), JsonValue::Array(events));
        Ok(out)
    }

    fn op_step(
        &mut self,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        let session_id = require_session_id(req)?;
        let action_raw = req.get("action").and_then(JsonValue::as_i64).ok_or((
            ProtocolErrorCode::InvalidRequest,
            "missing or non-integer 'action' field".to_owned(),
        ))?;

        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            (
                ProtocolErrorCode::UnknownSession,
                format!("unknown session '{session_id}'"),
            )
        })?;

        if session.env.terminal().is_terminal() {
            return Err((
                ProtocolErrorCode::Terminated,
                format!("session '{session_id}' is already terminal"),
            ));
        }

        if action_raw < 0 {
            return Err((
                ProtocolErrorCode::DomainError,
                format!("action index {action_raw} must be >= 0"),
            ));
        }
        let action = Action::Discrete(
            DiscreteAction::new(action_raw as usize, GRID_ACTION_CARDINALITY)
                .map_err(map_domain_err)?,
        );

        let prev_pos = session.env.position();
        let prev_collisions = session.env.collisions();
        let outcome = session.env.apply(&action).map_err(|err| match err {
            EnvironmentError::AlreadyTerminal => (
                ProtocolErrorCode::Terminated,
                format!("session '{session_id}' is already terminal"),
            ),
            EnvironmentError::InvalidAction(d) => map_domain_err(d),
            other => (
                ProtocolErrorCode::InternalError,
                format!("environment apply failed: {other}"),
            ),
        })?;

        let new_pos = session.env.position();
        let dx = new_pos.x as i64 - prev_pos.x as i64;
        let dy = new_pos.y as i64 - prev_pos.y as i64;
        let collided = session.env.collisions() > prev_collisions;
        let tick = session.env.tick().as_u64();
        let reward = session.env.last_reward();
        let total_reward = session.env.total_reward();

        let mut step_events = Vec::new();
        if collided {
            step_events.push(
                session
                    .push_event(
                        &self.quotas,
                        tick,
                        "wall_collision",
                        format!("collision at ({}, {})", new_pos.x, new_pos.y),
                    )
                    .to_json(),
            );
        } else {
            step_events.push(
                session
                    .push_event(
                        &self.quotas,
                        tick,
                        "step_applied",
                        format!("action={action_raw} dx={dx} dy={dy}"),
                    )
                    .to_json(),
            );
        }

        if let Some(reason) = outcome.terminal().reason() {
            step_events.push(
                session
                    .push_event(
                        &self.quotas,
                        tick,
                        "terminal_reached",
                        format!("reason={}", terminal_reason_str(reason)),
                    )
                    .to_json(),
            );
        }

        let mut delta = BTreeMap::new();
        delta.insert("dx".to_owned(), JsonValue::Number(JsonNumber::I64(dx)));
        delta.insert("dy".to_owned(), JsonValue::Number(JsonNumber::I64(dy)));
        delta.insert("collided".to_owned(), JsonValue::Bool(collided));

        let mut out = BTreeMap::new();
        out.insert(
            "session_id".to_owned(),
            JsonValue::String(session_id.to_owned()),
        );
        out.insert(
            "tick".to_owned(),
            JsonValue::Number(JsonNumber::I64(tick as i64)),
        );
        out.insert(
            "elapsed_s".to_owned(),
            JsonValue::Number(JsonNumber::F64(
                outcome.observation().elapsed().as_secs_f64(),
            )),
        );
        out.insert(
            "reward".to_owned(),
            JsonValue::Number(JsonNumber::F64(reward)),
        );
        out.insert(
            "total_reward".to_owned(),
            JsonValue::Number(JsonNumber::F64(total_reward)),
        );
        out.insert("delta".to_owned(), JsonValue::Object(delta));
        out.insert(
            "observation".to_owned(),
            features_to_json(outcome.observation().features()),
        );
        out.insert("terminal".to_owned(), terminal_to_json(outcome.terminal()));
        out.insert("events".to_owned(), JsonValue::Array(step_events));
        Ok(out)
    }

    fn op_events(
        &mut self,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        let session_id = require_session_id(req)?;
        let since_seq = match req.get("since_seq") {
            None => 0u64,
            Some(v) => {
                let raw = v.as_i64().ok_or((
                    ProtocolErrorCode::InvalidRequest,
                    "'since_seq' must be a non-negative integer".to_owned(),
                ))?;
                if raw < 0 {
                    return Err((
                        ProtocolErrorCode::InvalidRequest,
                        "'since_seq' must be >= 0".to_owned(),
                    ));
                }
                raw as u64
            }
        };
        let cap = self.quotas.max_events_per_response;
        let drain = match req.get("drain") {
            None => false,
            Some(v) => v.as_bool().ok_or((
                ProtocolErrorCode::InvalidRequest,
                "'drain' must be a boolean".to_owned(),
            ))?,
        };

        let max_count = match req.get("max_count") {
            None => cap,
            Some(v) => {
                let raw = v.as_i64().ok_or((
                    ProtocolErrorCode::InvalidRequest,
                    "'max_count' must be a positive integer".to_owned(),
                ))?;
                if raw < 1 {
                    return Err((
                        ProtocolErrorCode::InvalidRequest,
                        "'max_count' must be >= 1".to_owned(),
                    ));
                }
                (raw as u64).min(cap as u64) as usize
            }
        };

        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            (
                ProtocolErrorCode::UnknownSession,
                format!("unknown session '{session_id}'"),
            )
        })?;

        let first_seq = session.first_seq();
        let last_assigned = session.next_event_seq - 1;
        // GAP: hay eventos con seq > since_seq que ya fueron descartados por la retencion.
        let dropped = first_seq.saturating_sub(since_seq.saturating_add(1));
        let gap = dropped > 0;

        let selected: Vec<&ProtocolEvent> = session
            .events
            .iter()
            .filter(|e| e.seq > since_seq)
            .take(max_count)
            .collect();
        let next_seq = selected
            .last()
            .map_or(since_seq.max(last_assigned), |e| e.seq);
        let has_more = next_seq < last_assigned;
        let matched: Vec<JsonValue> = selected.into_iter().map(ProtocolEvent::to_json).collect();

        if drain {
            session
                .events
                .retain(|e| e.seq <= since_seq || e.seq > next_seq);
            session.event_bytes = session.events.iter().map(ProtocolEvent::approx_bytes).sum();
        }

        let to_num = |n: u64| JsonValue::Number(JsonNumber::I64(n as i64));
        let mut out = BTreeMap::new();
        out.insert(
            "session_id".to_owned(),
            JsonValue::String(session_id.to_owned()),
        );
        out.insert("events".to_owned(), JsonValue::Array(matched));
        out.insert("next_seq".to_owned(), to_num(next_seq));
        out.insert("first_seq".to_owned(), to_num(first_seq));
        out.insert("gap".to_owned(), JsonValue::Bool(gap));
        out.insert("dropped".to_owned(), to_num(dropped));
        out.insert("has_more".to_owned(), JsonValue::Bool(has_more));
        Ok(out)
    }

    fn op_reset(
        &mut self,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        let session_id = require_session_id(req)?;
        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            (
                ProtocolErrorCode::UnknownSession,
                format!("unknown session '{session_id}'"),
            )
        })?;

        let obs = LiveEnvironment::reset(&mut session.env).map_err(|e| {
            (
                ProtocolErrorCode::InternalError,
                format!("failed to reset grid session: {e}"),
            )
        })?;
        let ev = session.push_event(
            &self.quotas,
            0,
            "session_reset",
            "grid session reset".to_owned(),
        );

        let mut out = BTreeMap::new();
        out.insert(
            "session_id".to_owned(),
            JsonValue::String(session_id.to_owned()),
        );
        out.insert("tick".to_owned(), JsonValue::Number(JsonNumber::I64(0)));
        out.insert(
            "elapsed_s".to_owned(),
            JsonValue::Number(JsonNumber::F64(0.0)),
        );
        out.insert("observation".to_owned(), features_to_json(obs.features()));
        out.insert("terminal".to_owned(), terminal_to_json(Terminal::Ongoing));
        out.insert("events".to_owned(), JsonValue::Array(vec![ev.to_json()]));
        Ok(out)
    }

    fn op_close_session(
        &mut self,
        req: &BTreeMap<String, JsonValue>,
    ) -> Result<BTreeMap<String, JsonValue>, (ProtocolErrorCode, String)> {
        let session_id = require_session_id(req)?;
        if self.sessions.remove(session_id).is_none() {
            return Err((
                ProtocolErrorCode::UnknownSession,
                format!("unknown session '{session_id}'"),
            ));
        }

        let mut out = BTreeMap::new();
        out.insert(
            "session_id".to_owned(),
            JsonValue::String(session_id.to_owned()),
        );
        out.insert("closed".to_owned(), JsonValue::Bool(true));
        Ok(out)
    }
}

/// Ejecuta el servidor de protocolo JSON-Lines sobre `stdin` y `stdout`.
pub fn serve_stdio() -> Result<(), String> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut server = ProtocolServer::with_limits(JsonLimits::default());
    server
        .serve(stdin.lock(), stdout.lock())
        .map_err(|err| format!("error de E/S en serve-protocol: {err}"))
}

fn require_session_id(
    req: &BTreeMap<String, JsonValue>,
) -> Result<&str, (ProtocolErrorCode, String)> {
    req.get("session_id")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            (
                ProtocolErrorCode::InvalidRequest,
                "missing or non-string 'session_id' field".to_owned(),
            )
        })
}

fn map_domain_err(err: DomainError) -> (ProtocolErrorCode, String) {
    match err {
        DomainError::AlreadyTerminal => (ProtocolErrorCode::Terminated, err.to_string()),
        other => (ProtocolErrorCode::DomainError, other.to_string()),
    }
}

fn features_to_json(features: &[f64]) -> JsonValue {
    JsonValue::Array(
        features
            .iter()
            .map(|&v| JsonValue::Number(JsonNumber::F64(v)))
            .collect(),
    )
}

fn terminal_reason_str(reason: TerminalReason) -> &'static str {
    match reason {
        TerminalReason::GoalReached => "GoalReached",
        TerminalReason::Failure => "Failure",
        TerminalReason::HorizonReached => "HorizonReached",
        TerminalReason::Cancelled => "Cancelled",
        TerminalReason::InvariantViolated => "InvariantViolated",
    }
}

fn terminal_to_json(terminal: Terminal) -> JsonValue {
    let mut map = BTreeMap::new();
    map.insert(
        "is_terminal".to_owned(),
        JsonValue::Bool(terminal.is_terminal()),
    );
    let reason = match terminal.reason() {
        Some(r) => JsonValue::String(terminal_reason_str(r).to_owned()),
        None => JsonValue::Null,
    };
    map.insert("reason".to_owned(), reason);
    JsonValue::Object(map)
}

fn error_response(id: JsonValue, code: ProtocolErrorCode, message: &str) -> String {
    let mut err_obj = BTreeMap::new();
    err_obj.insert(
        "code".to_owned(),
        JsonValue::String(code.as_str().to_owned()),
    );
    err_obj.insert("message".to_owned(), JsonValue::String(message.to_owned()));

    let mut root = BTreeMap::new();
    root.insert("ok".to_owned(), JsonValue::Bool(false));
    root.insert("id".to_owned(), id);
    root.insert("error".to_owned(), JsonValue::Object(err_obj));
    JsonValue::Object(root).to_json_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_lab_grid::{ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_STAY, ACTION_UP};

    #[test]
    fn gog_138_serve_rejects_oversized_line_and_continues() {
        let limits = JsonLimits {
            max_bytes: 64,
            max_elements: 100,
            max_string_bytes: 32,
        };
        let mut server = ProtocolServer::with_limits(limits);
        let big = "x".repeat(10_000);
        let input = format!(
            "{{\"op\":\"create_session\",\"pad\":\"{big}\"}}\n{{\"id\":1,\"op\":\"create_session\"}}\n"
        );
        let mut out = Vec::new();
        server
            .serve(io::Cursor::new(input.into_bytes()), &mut out)
            .unwrap();
        let text = String::from_utf8(out).unwrap();
        let mut lines = text.lines();
        let first = lines.next().unwrap();
        assert!(
            first.contains("\"ok\":false") && first.contains("exceeds maximum"),
            "{first}"
        );
        let second = lines.next().unwrap();
        assert!(second.contains("\"ok\":true"), "{second}");
    }

    #[test]
    fn gog_138_read_bounded_line_exact_limit_and_unterminated_overflow() {
        let mut r = io::Cursor::new(b"12345\nabc".to_vec());
        assert!(
            matches!(read_bounded_line(&mut r, 5).unwrap(), BoundedLine::Line(l) if l == "12345")
        );
        assert!(
            matches!(read_bounded_line(&mut r, 5).unwrap(), BoundedLine::Line(l) if l == "abc")
        );
        assert!(matches!(
            read_bounded_line(&mut r, 5).unwrap(),
            BoundedLine::Eof
        ));
        let mut r = io::Cursor::new(b"123456789012\nok\n".to_vec());
        assert!(matches!(
            read_bounded_line(&mut r, 5).unwrap(),
            BoundedLine::TooLong
        ));
        assert!(matches!(read_bounded_line(&mut r, 5).unwrap(), BoundedLine::Line(l) if l == "ok"));
    }

    #[test]
    fn protocol_session_lifecycle_and_no_snapshot_per_step() {
        let mut server = ProtocolServer::new();

        let create_resp =
            parse_json_line(&server.handle_line(r#"{"id":1,"op":"create_session"}"#)).unwrap();
        let create_obj = create_resp.as_object().unwrap();
        assert_eq!(create_obj.get("ok"), Some(&JsonValue::Bool(true)));
        let sid = create_obj
            .get("session_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(
            create_obj.get("tick"),
            Some(&JsonValue::Number(JsonNumber::I64(0)))
        );

        let step_line = server.handle_line(&format!(
            r#"{{"id":2,"op":"step","session_id":"{sid}","action":{ACTION_RIGHT}}}"#
        ));
        let step_resp = parse_json_line(&step_line).unwrap();
        let step_obj = step_resp.as_object().unwrap();
        assert_eq!(step_obj.get("ok"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            step_obj.get("tick"),
            Some(&JsonValue::Number(JsonNumber::I64(1)))
        );
        assert!(!step_obj.contains_key("snapshot"));
        assert!(!step_obj.contains_key("map"));
        assert!(!step_obj.contains_key("config"));
        assert!(!step_obj.contains_key("cells"));
        assert!(step_obj.contains_key("delta"));
        assert!(step_obj.contains_key("observation"));
        assert!(step_obj.contains_key("reward"));
        assert!(step_obj.contains_key("terminal"));
        assert!(step_obj.contains_key("events"));

        let events_resp = parse_json_line(&server.handle_line(&format!(
            r#"{{"id":3,"op":"events","session_id":"{sid}","since_seq":0}}"#
        )))
        .unwrap();
        let events_arr = match events_resp.as_object().unwrap().get("events").unwrap() {
            JsonValue::Array(a) => a,
            _ => panic!("expected events array"),
        };
        assert_eq!(events_arr.len(), 2);

        let reset_resp = parse_json_line(
            &server.handle_line(&format!(r#"{{"id":4,"op":"reset","session_id":"{sid}"}}"#)),
        )
        .unwrap();
        assert_eq!(
            reset_resp.as_object().unwrap().get("tick"),
            Some(&JsonValue::Number(JsonNumber::I64(0)))
        );

        let close_resp = parse_json_line(&server.handle_line(&format!(
            r#"{{"id":5,"op":"close_session","session_id":"{sid}"}}"#
        )))
        .unwrap();
        assert_eq!(
            close_resp.as_object().unwrap().get("closed"),
            Some(&JsonValue::Bool(true))
        );

        let after_close = parse_json_line(&server.handle_line(&format!(
            r#"{{"id":6,"op":"step","session_id":"{sid}","action":{ACTION_STAY}}}"#
        )))
        .unwrap();
        let err = after_close
            .as_object()
            .unwrap()
            .get("error")
            .unwrap()
            .as_object()
            .unwrap();
        assert_eq!(
            err.get("code"),
            Some(&JsonValue::String("UnknownSession".to_owned()))
        );
    }

    #[test]
    fn protocol_returns_typed_errors() {
        let mut server = ProtocolServer::new();

        let bad_json = parse_json_line(&server.handle_line("{not valid")).unwrap();
        assert_eq!(
            bad_json
                .as_object()
                .unwrap()
                .get("error")
                .unwrap()
                .as_object()
                .unwrap()
                .get("code"),
            Some(&JsonValue::String("InvalidRequest".to_owned()))
        );

        let create = parse_json_line(&server.handle_line(r#"{"op":"create_session"}"#)).unwrap();
        let sid = create
            .as_object()
            .unwrap()
            .get("session_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();

        let bad_action = parse_json_line(&server.handle_line(&format!(
            r#"{{"op":"step","session_id":"{sid}","action":99}}"#
        )))
        .unwrap();
        assert_eq!(
            bad_action
                .as_object()
                .unwrap()
                .get("error")
                .unwrap()
                .as_object()
                .unwrap()
                .get("code"),
            Some(&JsonValue::String("DomainError".to_owned()))
        );

        // Move down from S=(3,3) into wall at (3,4) -> wall_collision event.
        let wall_step = parse_json_line(&server.handle_line(&format!(
            r#"{{"op":"step","session_id":"{sid}","action":{ACTION_DOWN}}}"#
        )))
        .unwrap();
        let delta = wall_step
            .as_object()
            .unwrap()
            .get("delta")
            .unwrap()
            .as_object()
            .unwrap();
        assert_eq!(delta.get("collided"), Some(&JsonValue::Bool(true)));
        assert_eq!(
            delta.get("dx"),
            Some(&JsonValue::Number(JsonNumber::I64(0)))
        );
        assert_eq!(
            delta.get("dy"),
            Some(&JsonValue::Number(JsonNumber::I64(0)))
        );

        // Reach Goal G=(7,1) and verify Terminated error on subsequent step.
        let path_to_goal = [
            ACTION_UP,
            ACTION_UP,
            ACTION_LEFT,
            ACTION_LEFT,
            ACTION_DOWN,
            ACTION_DOWN,
            ACTION_DOWN,
            ACTION_DOWN,
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_UP,
            ACTION_UP,
            ACTION_UP,
            ACTION_UP,
        ];
        let mut last_terminal = JsonValue::Null;
        for act in path_to_goal {
            let step = parse_json_line(&server.handle_line(&format!(
                r#"{{"op":"step","session_id":"{sid}","action":{act}}}"#
            )))
            .unwrap();
            last_terminal = step.as_object().unwrap().get("terminal").unwrap().clone();
        }
        assert_eq!(
            last_terminal.as_object().unwrap().get("is_terminal"),
            Some(&JsonValue::Bool(true))
        );
        assert_eq!(
            last_terminal.as_object().unwrap().get("reason"),
            Some(&JsonValue::String("GoalReached".to_owned()))
        );

        let after_goal = parse_json_line(&server.handle_line(&format!(
            r#"{{"op":"step","session_id":"{sid}","action":{ACTION_STAY}}}"#
        )))
        .unwrap();
        assert_eq!(
            after_goal
                .as_object()
                .unwrap()
                .get("error")
                .unwrap()
                .as_object()
                .unwrap()
                .get("code"),
            Some(&JsonValue::String("Terminated".to_owned()))
        );
    }

    #[test]
    fn protocol_matches_direct_rust_grid_env() {
        let mut direct = GridWorldEnv::new(GridWorldConfig::default()).unwrap();
        let mut server = ProtocolServer::new();
        let create = parse_json_line(&server.handle_line(r#"{"op":"create_session"}"#)).unwrap();
        let sid = create
            .as_object()
            .unwrap()
            .get("session_id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();

        let actions = [
            ACTION_RIGHT,
            ACTION_RIGHT,
            ACTION_DOWN,
            ACTION_UP,
            ACTION_STAY,
        ];
        for &act_idx in &actions {
            let act =
                Action::Discrete(DiscreteAction::new(act_idx, GRID_ACTION_CARDINALITY).unwrap());
            let direct_outcome = direct.apply(&act).unwrap();

            let resp = parse_json_line(&server.handle_line(&format!(
                r#"{{"op":"step","session_id":"{sid}","action":{act_idx}}}"#
            )))
            .unwrap();
            let obj = resp.as_object().unwrap();
            assert_eq!(
                obj.get("tick"),
                Some(&JsonValue::Number(JsonNumber::I64(
                    direct.tick().as_u64() as i64
                )))
            );
            assert_eq!(
                obj.get("reward"),
                Some(&JsonValue::Number(JsonNumber::F64(direct.last_reward())))
            );
            assert_eq!(
                obj.get("total_reward"),
                Some(&JsonValue::Number(JsonNumber::F64(direct.total_reward())))
            );
            assert_eq!(
                obj.get("observation"),
                Some(&features_to_json(direct_outcome.observation().features()))
            );
        }
    }

    fn gog_139_call(server: &mut ProtocolServer, line: &str) -> BTreeMap<String, JsonValue> {
        parse_json_line(&server.handle_line(line))
            .unwrap()
            .as_object()
            .unwrap()
            .clone()
    }

    fn gog_139_session(server: &mut ProtocolServer) -> String {
        let r = gog_139_call(server, r#"{"id":1,"op":"create_session"}"#);
        r.get("session_id").unwrap().as_str().unwrap().to_owned()
    }

    fn gog_139_i64(m: &BTreeMap<String, JsonValue>, k: &str) -> i64 {
        m.get(k).unwrap().as_i64().unwrap()
    }

    fn gog_139_seqs(m: &BTreeMap<String, JsonValue>) -> Vec<i64> {
        match m.get("events").unwrap() {
            JsonValue::Array(a) => a
                .iter()
                .map(|e| gog_139_i64(e.as_object().unwrap(), "seq"))
                .collect(),
            _ => panic!("events not array"),
        }
    }

    fn gog_139_steps(server: &mut ProtocolServer, sid: &str, n: usize) {
        for _ in 0..n {
            let r = gog_139_call(
                server,
                &format!(r#"{{"id":2,"op":"step","session_id":"{sid}","action":4}}"#),
            );
            assert!(r.contains_key("events"), "step failed: {r:?}");
        }
    }

    #[test]
    fn gog_139_fast_producer_bounds_retention_and_signals_gap() {
        let q = ProtocolQuotas {
            max_events_per_session: 10,
            ..ProtocolQuotas::default()
        };
        let mut server = ProtocolServer::with_quotas(q);
        let sid = gog_139_session(&mut server);
        gog_139_steps(&mut server, &sid, 100);
        let r = gog_139_call(
            &mut server,
            &format!(r#"{{"id":3,"op":"events","session_id":"{sid}","since_seq":0}}"#),
        );
        let seqs = gog_139_seqs(&r);
        assert_eq!(seqs.len(), 10);
        assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1));
        assert_eq!(r.get("gap").unwrap().as_bool(), Some(true));
        let first = gog_139_i64(&r, "first_seq");
        assert_eq!(first, seqs[0]);
        assert_eq!(gog_139_i64(&r, "dropped"), first - 1);
        assert_eq!(gog_139_i64(&r, "next_seq"), *seqs.last().unwrap());
        assert!(*seqs.last().unwrap() >= 101, "seq must stay monotonic");
        assert!(server.sessions[&sid].events.len() <= 10);
    }

    #[test]
    fn gog_139_byte_quota_discards_oldest() {
        let q = ProtocolQuotas {
            max_event_bytes_per_session: 300,
            ..ProtocolQuotas::default()
        };
        let mut server = ProtocolServer::with_quotas(q);
        let sid = gog_139_session(&mut server);
        gog_139_steps(&mut server, &sid, 50);
        let s = &server.sessions[&sid];
        assert!(s.event_bytes <= 300);
        assert!(s.events.len() < 10);
        let real: usize = s.events.iter().map(ProtocolEvent::approx_bytes).sum();
        assert_eq!(real, s.event_bytes);
    }

    #[test]
    fn gog_139_slow_reader_paginates_without_duplicates() {
        let mut server = ProtocolServer::new();
        let sid = gog_139_session(&mut server);
        gog_139_steps(&mut server, &sid, 25);
        let mut all = Vec::new();
        let mut cursor = 0i64;
        loop {
            let r = gog_139_call(
                &mut server,
                &format!(
                    r#"{{"id":3,"op":"events","session_id":"{sid}","since_seq":{cursor},"max_count":7}}"#
                ),
            );
            let seqs = gog_139_seqs(&r);
            assert!(seqs.len() <= 7);
            assert_eq!(r.get("gap").unwrap().as_bool(), Some(false));
            all.extend(seqs);
            cursor = gog_139_i64(&r, "next_seq");
            if r.get("has_more").unwrap().as_bool() == Some(false) {
                break;
            }
        }
        let expected: Vec<i64> = (1..=all.len() as i64).collect();
        assert_eq!(all, expected);
        assert!(all.len() >= 26);
    }

    #[test]
    fn gog_139_events_default_cap_and_invalid_max_count() {
        let q = ProtocolQuotas {
            max_events_per_response: 5,
            ..ProtocolQuotas::default()
        };
        let mut server = ProtocolServer::with_quotas(q);
        let sid = gog_139_session(&mut server);
        gog_139_steps(&mut server, &sid, 20);
        let r = gog_139_call(
            &mut server,
            &format!(r#"{{"id":3,"op":"events","session_id":"{sid}"}}"#),
        );
        assert_eq!(gog_139_seqs(&r).len(), 5);
        assert_eq!(r.get("has_more").unwrap().as_bool(), Some(true));
        let r = gog_139_call(
            &mut server,
            &format!(r#"{{"id":3,"op":"events","session_id":"{sid}","max_count":0}}"#),
        );
        assert_eq!(r.get("ok").and_then(JsonValue::as_bool), Some(false));
        let r = gog_139_call(
            &mut server,
            &format!(
                r#"{{"id":3,"op":"events","session_id":"{sid}","max_count":2,"since_seq":1000}}"#
            ),
        );
        assert!(gog_139_seqs(&r).is_empty());
        assert_eq!(r.get("gap").unwrap().as_bool(), Some(false));
        assert_eq!(gog_139_i64(&r, "next_seq"), 1000);
    }

    #[test]
    fn gog_139_session_quota_rejects_and_close_frees() {
        let q = ProtocolQuotas {
            max_sessions: 2,
            ..ProtocolQuotas::default()
        };
        let mut server = ProtocolServer::with_quotas(q);
        let a = gog_139_session(&mut server);
        let _b = gog_139_session(&mut server);
        let raw = server.handle_line(r#"{"id":9,"op":"create_session"}"#);
        assert!(raw.contains("QuotaExceeded"), "{raw}");
        assert_eq!(server.sessions.len(), 2);
        let closed = server.handle_line(&format!(
            r#"{{"id":10,"op":"close_session","session_id":"{a}"}}"#
        ));
        assert!(closed.contains("\"closed\":true"), "{closed}");
        let _c = gog_139_session(&mut server);
        assert_eq!(server.sessions.len(), 2);
    }

    #[test]
    fn gog_139_memory_bounded_under_load() {
        let q = ProtocolQuotas {
            max_sessions: 3,
            max_events_per_session: 16,
            max_event_bytes_per_session: 4096,
            max_events_per_response: 8,
        };
        let mut server = ProtocolServer::with_quotas(q);
        for _ in 0..5 {
            let _ = server.handle_line(r#"{"id":1,"op":"create_session"}"#);
        }
        let ids: Vec<String> = server.sessions.keys().cloned().collect();
        for sid in &ids {
            gog_139_steps(&mut server, sid, 200);
            let _ = server.handle_line(&format!(r#"{{"id":4,"op":"reset","session_id":"{sid}"}}"#));
        }
        assert!(server.sessions.len() <= 3);
        for s in server.sessions.values() {
            assert!(s.events.len() <= 16);
            assert!(s.event_bytes <= 4096);
        }
    }
}
