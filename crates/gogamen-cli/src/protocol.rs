//! Servidor de protocolo residente JSON-Lines para el SDK Python de Gogamen.
//!
//! Mantiene las sesiones [`GridWorldEnv`] residentes en Rust y expone operaciones
//! ligeras (`create_session`, `step`, `events`, `reset`, `close_session`) por `stdin`/`stdout`
//! sin transferir snapshots completos por paso.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::{self, BufRead, Write};

use gogamen_core::{
    Action, Actuator, DiscreteAction, DomainError, EnvironmentError, EnvironmentObserver,
    LiveEnvironment, Terminal, TerminalReason,
};
use gogamen_lab_grid::{GRID_ACTION_CARDINALITY, GridWorldConfig, GridWorldEnv};
use gogamen_spec::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};

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

fn parse_json_line(input: &str) -> Result<JsonValue, String> {
    parse_json(input).map_err(|err| err.to_string())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProtocolErrorCode {
    InvalidRequest,
    UnknownSession,
    DomainError,
    Terminated,
    InternalError,
}

impl ProtocolErrorCode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "InvalidRequest",
            Self::UnknownSession => "UnknownSession",
            Self::DomainError => "DomainError",
            Self::Terminated => "Terminated",
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

struct GridSession {
    env: GridWorldEnv,
    next_event_seq: u64,
    events: Vec<ProtocolEvent>,
}

impl GridSession {
    fn new(env: GridWorldEnv) -> Self {
        let mut session = Self {
            env,
            next_event_seq: 1,
            events: Vec::new(),
        };
        session.push_event(0, "session_created", "grid session initialized".to_owned());
        session
    }

    fn push_event(&mut self, tick: u64, kind: &'static str, detail: String) -> ProtocolEvent {
        let ev = ProtocolEvent {
            seq: self.next_event_seq,
            tick,
            kind,
            detail,
        };
        self.next_event_seq += 1;
        self.events.push(ev.clone());
        ev
    }
}

/// Servidor residente del protocolo JSON-Lines.
#[derive(Default)]
pub struct ProtocolServer {
    next_session_num: u64,
    sessions: BTreeMap<String, GridSession>,
}

impl ProtocolServer {
    /// Crea un nuevo servidor de protocolo sin sesiones activas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Ejecuta el bucle de lectura por líneas hasta alcanzar EOF.
    pub fn serve<R: BufRead, W: Write>(&mut self, reader: R, mut writer: W) -> io::Result<()> {
        for line_res in reader.lines() {
            let line = line_res?;
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
        let parsed = match parse_json_line(line) {
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

        self.next_session_num += 1;
        let session_id = format!("sess-{}", self.next_session_num);
        let session = GridSession::new(env);
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
        let drain = match req.get("drain") {
            None => false,
            Some(v) => v.as_bool().ok_or((
                ProtocolErrorCode::InvalidRequest,
                "'drain' must be a boolean".to_owned(),
            ))?,
        };

        let session = self.sessions.get_mut(session_id).ok_or_else(|| {
            (
                ProtocolErrorCode::UnknownSession,
                format!("unknown session '{session_id}'"),
            )
        })?;

        let matched: Vec<JsonValue> = session
            .events
            .iter()
            .filter(|e| e.seq > since_seq)
            .map(ProtocolEvent::to_json)
            .collect();

        if drain {
            session.events.retain(|e| e.seq <= since_seq);
        }

        let mut out = BTreeMap::new();
        out.insert(
            "session_id".to_owned(),
            JsonValue::String(session_id.to_owned()),
        );
        out.insert("events".to_owned(), JsonValue::Array(matched));
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
        let ev = session.push_event(0, "session_reset", "grid session reset".to_owned());

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
    let mut server = ProtocolServer::new();
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
}
