//! Parámetros dinámicos y comandos tipados (GOG-95), sobre los controles de sesión de GOG-90.
//!
//! - El esquema ([`ParamSchema`]) publica por parámetro tipo, rango, unidad y mutabilidad.
//! - Un cambio autorizado se valida entera y atómicamente, se **encola** y se aplica en la
//!   siguiente frontera de tick (nunca a mitad de tick). El ACK lleva la revisión de parámetros
//!   y el tick de aplicación; cada cambio aplicado añade un [`ParamEvent`] al log determinista.
//! - Los cambios estructurales ([`ParamRequest::Structural`]) se rechazan: requieren compilación
//!   o sustitución explícita, fuera del alcance de este canal.
//!
//! Sin reloj de pared. Los tipos de `gogamen-core`/`gogamen-spec` no definen parámetros/unidades
//! (sólo capacidades de módulos), así que se añaden los mínimos aquí.

use std::collections::VecDeque;
use std::sync::Arc;

use gogamen_core::Tick;

use crate::control::{CommandId, RejectReason, SessionId};

/// Capacidad del log de rechazos de parámetros visible en la observación.
pub const PARAM_REJECTION_LOG_CAP: usize = 64;

/// Identificador estable de parámetro.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(pub String);

impl ParamId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
}

/// Unidad física/lógica del parámetro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamUnit {
    None,
    Seconds,
    Meters,
    Radians,
    Degrees,
    Hertz,
    Newtons,
    Custom(&'static str),
}

/// Valor de un parámetro.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    Enum(String),
}

/// Tipo y rango declarados.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    Float { min: f64, max: f64, step: f64 },
    Int { min: i64, max: i64, step: i64 },
    Bool,
    Enum { variants: Vec<String> },
}

/// Mutabilidad en caliente.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mutability {
    Mutable,
    Immutable,
}

/// Descriptor publicado de un parámetro.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDescriptor {
    pub id: ParamId,
    pub label: String,
    pub kind: ParamKind,
    pub unit: ParamUnit,
    pub mutability: Mutability,
    pub default: ParamValue,
}

/// Error de construcción del esquema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaError {
    DuplicateId(ParamId),
    InvalidDescriptor(ParamId),
}

/// Esquema validado: ids únicos, rangos coherentes y valores por defecto válidos.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParamSchema {
    descriptors: Vec<ParamDescriptor>,
}

fn check_value(kind: &ParamKind, v: &ParamValue) -> Result<(), CheckError> {
    match (kind, v) {
        (ParamKind::Float { min, max, .. }, ParamValue::Float(x)) => {
            if x.is_finite() && x >= min && x <= max {
                Ok(())
            } else {
                Err(CheckError::OutOfRange)
            }
        }
        (ParamKind::Int { min, max, .. }, ParamValue::Int(x)) => {
            if x >= min && x <= max {
                Ok(())
            } else {
                Err(CheckError::OutOfRange)
            }
        }
        (ParamKind::Bool, ParamValue::Bool(_)) => Ok(()),
        (ParamKind::Enum { variants }, ParamValue::Enum(s)) => {
            if variants.contains(s) {
                Ok(())
            } else {
                Err(CheckError::OutOfRange)
            }
        }
        _ => Err(CheckError::WrongType),
    }
}

enum CheckError {
    OutOfRange,
    WrongType,
}

impl ParamSchema {
    pub fn new(descriptors: Vec<ParamDescriptor>) -> Result<Self, SchemaError> {
        for (i, d) in descriptors.iter().enumerate() {
            if descriptors[..i].iter().any(|o| o.id == d.id) {
                return Err(SchemaError::DuplicateId(d.id.clone()));
            }
            let sane = match &d.kind {
                ParamKind::Float { min, max, step } => {
                    min.is_finite() && max.is_finite() && min <= max && *step > 0.0
                }
                ParamKind::Int { min, max, step } => min <= max && *step > 0,
                ParamKind::Bool => true,
                ParamKind::Enum { variants } => !variants.is_empty(),
            };
            if !sane || check_value(&d.kind, &d.default).is_err() {
                return Err(SchemaError::InvalidDescriptor(d.id.clone()));
            }
        }
        Ok(Self { descriptors })
    }

    pub fn descriptors(&self) -> &[ParamDescriptor] {
        &self.descriptors
    }

    fn index_of(&self, id: &ParamId) -> Option<usize> {
        self.descriptors.iter().position(|d| &d.id == id)
    }
}

/// Un cambio individual.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamChange {
    pub id: ParamId,
    pub value: ParamValue,
}

/// Petición tipada.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamRequest {
    /// Conjunto atómico: si una entrada es inválida no se aplica ninguna.
    Set(Vec<ParamChange>),
    /// Cambio estructural (topología/spec): requiere compilación/sustitución explícita.
    Structural { description: String },
}

/// Comando de parámetros, identificado y dirigido a una sesión.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamCommand {
    pub id: CommandId,
    pub session: SessionId,
    pub request: ParamRequest,
}

/// Motivo tipado de rechazo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamRejectReason {
    UnknownParam(ParamId),
    WrongType(ParamId),
    OutOfRange(ParamId),
    /// El parámetro es inmutable en caliente.
    Immutable(ParamId),
    /// Sesión de sólo lectura (permiso denegado).
    ReadOnlySession,
    /// Requiere compilación/sustitución explícita.
    StructuralChange,
    EmptyCommand,
    DuplicateParamInCommand(ParamId),
    /// Rechazo de sesión compartido con GOG-90 (obsoleta, duplicado, detenido, overflow).
    Session(RejectReason),
}

/// Resultado de un comando de parámetros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParamOutcome {
    /// Encolado: se aplicará en `applies_at` con la revisión `revision`.
    Queued {
        revision: u64,
        applies_at: Tick,
    },
    Rejected {
        reason: ParamRejectReason,
    },
}

/// ACK: `tick` es el tick actual al procesar; `param_revision` la última revisión ya aplicada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamAck {
    pub id: CommandId,
    pub outcome: ParamOutcome,
    pub param_revision: u64,
    pub tick: Tick,
}

/// Evento reproducible de un cambio aplicado.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamEvent {
    pub param: ParamId,
    pub old: ParamValue,
    pub new: ParamValue,
    pub revision: u64,
    pub tick: Tick,
    pub command: CommandId,
}

/// Entrada del log de rechazos de parámetros.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamRejectionRecord {
    pub id: CommandId,
    pub reason: ParamRejectReason,
}

/// Único punto de entrada mutante para parámetros (la UI sólo ve este puerto).
pub trait ParamControlPort {
    fn submit_params(&self, command: ParamCommand) -> ParamAck;
}

struct Pending {
    command: CommandId,
    revision: u64,
    entries: Vec<(usize, ParamValue)>,
}

/// Estado de parámetros del lado motor (dentro del `Mutex` de la sesión).
pub(crate) struct ParamState {
    schema: Arc<ParamSchema>,
    values: Vec<ParamValue>,
    pending: VecDeque<Pending>,
    applied_revision: u64,
    queued_revision: u64,
    events: Vec<ParamEvent>,
    rejection_count: u64,
    rejections: VecDeque<ParamRejectionRecord>,
}

impl ParamState {
    pub(crate) fn new(schema: ParamSchema) -> Self {
        let values = schema
            .descriptors
            .iter()
            .map(|d| d.default.clone())
            .collect();
        Self {
            schema: Arc::new(schema),
            values,
            pending: VecDeque::new(),
            applied_revision: 0,
            queued_revision: 0,
            events: Vec::new(),
            rejection_count: 0,
            rejections: VecDeque::new(),
        }
    }

    pub(crate) fn schema(&self) -> ParamSchema {
        (*self.schema).clone()
    }

    pub(crate) fn values(&self) -> Vec<(ParamId, ParamValue)> {
        self.schema
            .descriptors
            .iter()
            .zip(&self.values)
            .map(|(d, v)| (d.id.clone(), v.clone()))
            .collect()
    }

    pub(crate) fn applied_revision(&self) -> u64 {
        self.applied_revision
    }
    pub(crate) fn pending_len(&self) -> usize {
        self.pending.len()
    }
    pub(crate) fn events(&self) -> Vec<ParamEvent> {
        self.events.clone()
    }
    pub(crate) fn rejection_count(&self) -> u64 {
        self.rejection_count
    }
    pub(crate) fn rejections(&self) -> Vec<ParamRejectionRecord> {
        self.rejections.iter().cloned().collect()
    }

    /// `Reset`: valores por defecto, cola y log de eventos vacíos, revisión a 0 (nueva generación).
    pub(crate) fn reset(&mut self) {
        self.values = self
            .schema
            .descriptors
            .iter()
            .map(|d| d.default.clone())
            .collect();
        self.pending.clear();
        self.applied_revision = 0;
        self.queued_revision = 0;
        self.events.clear();
    }

    pub(crate) fn reject(&mut self, id: CommandId, reason: ParamRejectReason) {
        self.rejection_count += 1;
        if self.rejections.len() == PARAM_REJECTION_LOG_CAP {
            self.rejections.pop_front();
        }
        self.rejections
            .push_back(ParamRejectionRecord { id, reason });
    }

    /// Valida sin mutar nada. Devuelve las entradas resueltas por índice.
    pub(crate) fn validate(
        &self,
        request: &ParamRequest,
    ) -> Result<Vec<(usize, ParamValue)>, ParamRejectReason> {
        let changes = match request {
            ParamRequest::Structural { .. } => return Err(ParamRejectReason::StructuralChange),
            ParamRequest::Set(c) => c,
        };
        if changes.is_empty() {
            return Err(ParamRejectReason::EmptyCommand);
        }
        let mut out: Vec<(usize, ParamValue)> = Vec::with_capacity(changes.len());
        for ch in changes {
            let idx = self
                .schema
                .index_of(&ch.id)
                .ok_or_else(|| ParamRejectReason::UnknownParam(ch.id.clone()))?;
            if out.iter().any(|(i, _)| *i == idx) {
                return Err(ParamRejectReason::DuplicateParamInCommand(ch.id.clone()));
            }
            let d = &self.schema.descriptors[idx];
            if d.mutability == Mutability::Immutable {
                return Err(ParamRejectReason::Immutable(ch.id.clone()));
            }
            check_value(&d.kind, &ch.value).map_err(|e| match e {
                CheckError::OutOfRange => ParamRejectReason::OutOfRange(ch.id.clone()),
                CheckError::WrongType => ParamRejectReason::WrongType(ch.id.clone()),
            })?;
            out.push((idx, ch.value.clone()));
        }
        Ok(out)
    }

    /// Encola una petición ya validada y devuelve su revisión.
    pub(crate) fn enqueue(&mut self, command: CommandId, entries: Vec<(usize, ParamValue)>) -> u64 {
        self.queued_revision += 1;
        let revision = self.queued_revision;
        self.pending.push_back(Pending {
            command,
            revision,
            entries,
        });
        revision
    }

    /// Frontera de tick: aplica en orden FIFO todo lo encolado, con eventos en `tick`.
    pub(crate) fn apply_boundary(&mut self, tick: Tick) {
        while let Some(p) = self.pending.pop_front() {
            for (idx, new) in p.entries {
                let old = std::mem::replace(&mut self.values[idx], new.clone());
                self.events.push(ParamEvent {
                    param: self.schema.descriptors[idx].id.clone(),
                    old,
                    new,
                    revision: p.revision,
                    tick,
                    command: p.command,
                });
            }
            self.applied_revision = p.revision;
        }
    }
}

#[cfg(test)]
pub(crate) fn demo_schema() -> ParamSchema {
    ParamSchema::new(vec![
        ParamDescriptor {
            id: ParamId::new("gain"),
            label: "Gain".into(),
            kind: ParamKind::Float {
                min: 0.0,
                max: 10.0,
                step: 0.5,
            },
            unit: ParamUnit::Newtons,
            mutability: Mutability::Mutable,
            default: ParamValue::Float(1.0),
        },
        ParamDescriptor {
            id: ParamId::new("count"),
            label: "Count".into(),
            kind: ParamKind::Int {
                min: 1,
                max: 8,
                step: 1,
            },
            unit: ParamUnit::None,
            mutability: Mutability::Mutable,
            default: ParamValue::Int(2),
        },
        ParamDescriptor {
            id: ParamId::new("enabled"),
            label: "Enabled".into(),
            kind: ParamKind::Bool,
            unit: ParamUnit::None,
            mutability: Mutability::Mutable,
            default: ParamValue::Bool(false),
        },
        ParamDescriptor {
            id: ParamId::new("mode"),
            label: "Mode".into(),
            kind: ParamKind::Enum {
                variants: vec!["eco".into(), "sport".into()],
            },
            unit: ParamUnit::None,
            mutability: Mutability::Mutable,
            default: ParamValue::Enum("eco".into()),
        },
        ParamDescriptor {
            id: ParamId::new("mass"),
            label: "Mass".into(),
            kind: ParamKind::Float {
                min: 0.1,
                max: 100.0,
                step: 0.1,
            },
            unit: ParamUnit::Custom("kg"),
            mutability: Mutability::Immutable,
            default: ParamValue::Float(5.0),
        },
    ])
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        ControlAction, ControlCommand, ControlPort, ObservationPort, SessionHost, SessionKind,
        SessionOptions,
    };

    fn set(id: u64, s: SessionId, changes: &[(&str, ParamValue)]) -> ParamCommand {
        ParamCommand {
            id: CommandId(id),
            session: s,
            request: ParamRequest::Set(
                changes
                    .iter()
                    .map(|(n, v)| ParamChange {
                        id: ParamId::new(*n),
                        value: v.clone(),
                    })
                    .collect(),
            ),
        }
    }

    fn step(id: u64, s: SessionId) -> ControlCommand {
        ControlCommand {
            id: CommandId(id),
            session: s,
            action: ControlAction::Step,
        }
    }

    fn val(o: &crate::control::Observation, n: &str) -> ParamValue {
        o.params
            .iter()
            .find(|(i, _)| i.0 == n)
            .map(|(_, v)| v.clone())
            .unwrap()
    }

    fn scenario() -> (Vec<ParamAck>, Vec<ParamEvent>, Vec<(ParamId, ParamValue)>) {
        let h = SessionHost::with_params(1, SessionOptions::simulated(), demo_schema());
        let c = h.control_port().unwrap();
        let p = h.param_port();
        let o = h.observation_port();
        let s = o.observe().session;
        let mut acks = Vec::new();
        acks.push(p.submit_params(set(1, s, &[("gain", ParamValue::Float(3.0))])));
        c.submit(step(2, s));
        acks.push(p.submit_params(set(
            3,
            s,
            &[
                ("gain", ParamValue::Float(4.5)),
                ("mode", ParamValue::Enum("sport".into())),
            ],
        )));
        acks.push(p.submit_params(set(4, s, &[("count", ParamValue::Int(5))])));
        h.control_port().unwrap().submit(ControlCommand {
            id: CommandId(5),
            session: s,
            action: ControlAction::Resume,
        });
        h.pump(3);
        (acks, o.param_events(), o.observe().params)
    }

    #[test]
    fn ac1_change_is_queued_applied_at_tick_boundary_with_revision_and_event() {
        let h = SessionHost::with_params(1, SessionOptions::simulated(), demo_schema());
        let c = h.control_port().unwrap();
        let p = h.param_port();
        let o = h.observation_port();
        let s = o.observe().session;
        let ack = p.submit_params(set(1, s, &[("gain", ParamValue::Float(3.0))]));
        assert_eq!(
            ack.outcome,
            ParamOutcome::Queued {
                revision: 1,
                applies_at: Tick::new(1)
            }
        );
        assert_eq!(ack.tick, Tick::ZERO);
        // mid-tick: nothing applied
        let mid = o.observe();
        assert_eq!(val(&mid, "gain"), ParamValue::Float(1.0));
        assert_eq!((mid.param_revision, mid.pending_param_commands), (0, 1));
        assert!(o.param_events().is_empty());
        // boundary
        c.submit(step(2, s));
        let after = o.observe();
        assert_eq!(val(&after, "gain"), ParamValue::Float(3.0));
        assert_eq!((after.param_revision, after.pending_param_commands), (1, 0));
        assert_eq!(
            o.param_events(),
            vec![ParamEvent {
                param: ParamId::new("gain"),
                old: ParamValue::Float(1.0),
                new: ParamValue::Float(3.0),
                revision: 1,
                tick: Tick::new(1),
                command: CommandId(1),
            }]
        );
    }

    #[test]
    fn ac1_pump_applies_on_first_boundary_and_replay_is_identical() {
        let a = scenario();
        let b = scenario();
        assert_eq!(a, b, "same sequence => same acks, events and values");
        let (acks, events, values) = a;
        assert_eq!(
            acks[1].outcome,
            ParamOutcome::Queued {
                revision: 2,
                applies_at: Tick::new(2)
            }
        );
        // multi-param command: one event per entry sharing revision; queued after: rev 3 at tick 2
        let revs: Vec<(u64, Tick)> = events.iter().map(|e| (e.revision, e.tick)).collect();
        let (t1, t2) = (Tick::new(1), Tick::new(2));
        assert_eq!(revs, vec![(1, t1), (2, t2), (2, t2), (3, t2)]);
        assert!(values.contains(&(ParamId::new("count"), ParamValue::Int(5))));
        assert!(values.contains(&(ParamId::new("mode"), ParamValue::Enum("sport".into()))));
    }

    #[test]
    fn ac2_invalid_requests_are_rejected_typed_without_partial_mutation() {
        let h = SessionHost::with_params(1, SessionOptions::simulated(), demo_schema());
        let c = h.control_port().unwrap();
        let p = h.param_port();
        let o = h.observation_port();
        let s = o.observe().session;
        let base = o.observe().params;
        let cases: Vec<(ParamCommand, ParamRejectReason)> = vec![
            (
                set(1, s, &[("gain", ParamValue::Float(10.5))]),
                ParamRejectReason::OutOfRange(ParamId::new("gain")),
            ),
            (
                set(2, s, &[("gain", ParamValue::Float(f64::NAN))]),
                ParamRejectReason::OutOfRange(ParamId::new("gain")),
            ),
            (
                set(3, s, &[("count", ParamValue::Float(2.0))]),
                ParamRejectReason::WrongType(ParamId::new("count")),
            ),
            (
                set(4, s, &[("mode", ParamValue::Enum("warp".into()))]),
                ParamRejectReason::OutOfRange(ParamId::new("mode")),
            ),
            (
                set(5, s, &[("mass", ParamValue::Float(6.0))]),
                ParamRejectReason::Immutable(ParamId::new("mass")),
            ),
            (
                set(6, s, &[("nope", ParamValue::Bool(true))]),
                ParamRejectReason::UnknownParam(ParamId::new("nope")),
            ),
            (
                // atomic: valid first entry must not be applied
                set(
                    7,
                    s,
                    &[
                        ("gain", ParamValue::Float(2.0)),
                        ("count", ParamValue::Int(99)),
                    ],
                ),
                ParamRejectReason::OutOfRange(ParamId::new("count")),
            ),
            (
                set(
                    8,
                    s,
                    &[
                        ("gain", ParamValue::Float(2.0)),
                        ("gain", ParamValue::Float(3.0)),
                    ],
                ),
                ParamRejectReason::DuplicateParamInCommand(ParamId::new("gain")),
            ),
            (set(9, s, &[]), ParamRejectReason::EmptyCommand),
            (
                ParamCommand {
                    id: CommandId(10),
                    session: s,
                    request: ParamRequest::Structural {
                        description: "add node".into(),
                    },
                },
                ParamRejectReason::StructuralChange,
            ),
        ];
        let mut expected = Vec::new();
        for (cmd, reason) in cases {
            let id = cmd.id;
            let ack = p.submit_params(cmd);
            assert_eq!(
                ack.outcome,
                ParamOutcome::Rejected {
                    reason: reason.clone()
                }
            );
            expected.push(ParamRejectionRecord { id, reason });
        }
        c.submit(step(100, s)); // boundary: nothing must have been queued
        let obs = o.observe();
        assert_eq!(obs.params, base);
        assert_eq!((obs.param_revision, obs.pending_param_commands), (0, 0));
        assert!(o.param_events().is_empty());
        assert_eq!(obs.param_rejection_count, 10);
        assert_eq!(obs.recent_param_rejections, expected);
    }

    #[test]
    fn ac2_read_only_session_denies_and_logs_without_mutation() {
        let h = SessionHost::with_params(
            1,
            SessionOptions::of_kind(SessionKind::Remote),
            demo_schema(),
        );
        assert!(h.control_port().is_err());
        let o = h.observation_port();
        let s = o.observe().session;
        let ack = h
            .param_port()
            .submit_params(set(1, s, &[("gain", ParamValue::Float(3.0))]));
        assert_eq!(
            ack.outcome,
            ParamOutcome::Rejected {
                reason: ParamRejectReason::ReadOnlySession
            }
        );
        let obs = o.observe();
        assert_eq!(val(&obs, "gain"), ParamValue::Float(1.0));
        assert_eq!(obs.pending_param_commands, 0);
        assert_eq!(obs.param_rejection_count, 1);
    }

    #[test]
    fn stale_and_duplicate_commands_are_rejected_and_reset_restores_defaults() {
        let h = SessionHost::with_params(1, SessionOptions::simulated(), demo_schema());
        let c = h.control_port().unwrap();
        let p = h.param_port();
        let o = h.observation_port();
        let s = o.observe().session;
        p.submit_params(set(1, s, &[("count", ParamValue::Int(4))]));
        let dup = p.submit_params(set(1, s, &[("count", ParamValue::Int(5))]));
        assert_eq!(
            dup.outcome,
            ParamOutcome::Rejected {
                reason: ParamRejectReason::Session(RejectReason::DuplicateCommandId)
            }
        );
        c.submit(step(2, s));
        assert_eq!(val(&o.observe(), "count"), ParamValue::Int(4));
        c.submit(ControlCommand {
            id: CommandId(3),
            session: s,
            action: ControlAction::Reset,
        });
        let obs = o.observe();
        assert_eq!(val(&obs, "count"), ParamValue::Int(2));
        assert_eq!(obs.param_revision, 0);
        let stale = p.submit_params(set(9, s, &[("count", ParamValue::Int(3))]));
        assert!(matches!(
            stale.outcome,
            ParamOutcome::Rejected {
                reason: ParamRejectReason::Session(RejectReason::StaleSession { .. })
            }
        ));
    }

    #[test]
    fn schema_validation_rejects_duplicates_and_bad_defaults() {
        let mut d = demo_schema().descriptors().to_vec();
        d.push(d[0].clone());
        assert!(matches!(
            ParamSchema::new(d),
            Err(SchemaError::DuplicateId(_))
        ));
        let mut d = demo_schema().descriptors().to_vec();
        d[0].default = ParamValue::Float(99.0);
        assert!(matches!(
            ParamSchema::new(d),
            Err(SchemaError::InvalidDescriptor(_))
        ));
    }
}
