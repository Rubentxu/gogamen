//! Proyección pura (sin Rerun) de una traza hacia una lista ordenada de [`Item`]s.
//!
//! El [`ExportPlan`] es transitorio: se construye desde una referencia inmutable a la traza
//! fuente, se consume al emitir y no se persiste. Aquí se decide, de forma testeable sin Rerun:
//! qué se exporta, con qué tiempos/unidades, cómo se marcan huecos y redacciones, y qué se
//! deniega por falta de permiso.

use gogamen_core::Action;
use gogamen_lab_grid::CellKind;
use gogamen_runtime::scientific_trace::{
    RecordedGuardDecision, ScientificRecord, ScientificTraceArchive, SensitiveFieldMode,
    SensitiveFieldPolicy, TraceGapReason,
};
use gogamen_studio::replay::{Recording, TICKS_PER_SECOND_1X};

/// Texto del marcador de redacción (idéntico al de la traza científica).
pub const REDACTED_MARKER: &str = "[REDACTED]";
/// Texto del marcador de contenido retenido por falta de permiso.
pub const WITHHELD_MARKER: &str = "[WITHHELD: export permission not granted]";

/// Nombre de la línea de tiempo discreta de ticks (sin unidad).
pub const TIMELINE_TICK: &str = "tick";
/// Nombre de la línea de tiempo de tiempo simulado (duración en **segundos**).
pub const TIMELINE_SIM_TIME: &str = "sim_time";
/// Nombre de la línea de tiempo de revisiones de la traza científica (sin unidad).
pub const TIMELINE_REVISION: &str = "revision";

/// Permisos explícitos de exportación. **Todo denegado por defecto.**
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExportPermissions {
    /// Permite exportar texto libre (anotaciones, motivos de rechazo, detalles de fallo).
    pub messages: bool,
    /// Permite exportar valores que la política sensible clasifica como secretos y que aún
    /// están en claro en la fuente. Un valor ya redactado en la fuente nunca se "des-redacta".
    pub secrets: bool,
}

/// Opciones de exportación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportOptions {
    /// Identificador de aplicación del recording Rerun.
    pub application_id: String,
    /// Permisos (por defecto, todo denegado).
    pub permissions: ExportPermissions,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            application_id: "gogamen".to_owned(),
            permissions: ExportPermissions::default(),
        }
    }
}

/// Instante de un [`Item`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ItemTime {
    /// Dato estático (sin tiempo): mapa, metadatos, unidades.
    Static,
    /// Dato en un instante; cada componente ausente se deshabilita en su timeline.
    At {
        /// Tick discreto.
        tick: Option<u64>,
        /// Revisión de la traza científica.
        revision: Option<u64>,
        /// Tiempo simulado en segundos.
        seconds: Option<f64>,
    },
}

/// Tipo de celda del mapa (vista de escena).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapCell {
    /// Muro.
    Wall,
    /// Objetivo.
    Goal,
    /// Peligro.
    Hazard,
    /// Celda inicial.
    Start,
}

/// Contenido de un [`Item`].
#[derive(Debug, Clone, PartialEq)]
pub enum Payload {
    /// Metadatos de una serie escalar: nombre legible y **unidad** explícita.
    SeriesMeta {
        /// Nombre de la serie.
        name: String,
        /// Unidad (`"count"`, `"reward"`, `"index"`, ...).
        unit: String,
    },
    /// Muestra escalar.
    Scalar(f64),
    /// Muestra escalar **ausente** (hueco explícito; nunca se rellena con cero).
    GapScalar,
    /// La entidad no existe en este instante (hueco explícito de una vista de escena).
    GapEntity,
    /// Etiqueta de estado de vocabulario cerrado (no es texto libre).
    Label(String),
    /// Texto libre exportado **con permiso**.
    Message(String),
    /// Valor redactado: se conserva el marcador, nunca el contenido.
    Redacted,
    /// Texto libre existente pero retenido por falta de permiso.
    Withheld,
    /// Punto 2D de escena (coordenadas de celda, centro).
    Point2D {
        /// Coordenada x (columna + 0.5).
        x: f32,
        /// Coordenada y (fila + 0.5).
        y: f32,
    },
    /// Celda de mapa estática.
    Cell {
        /// Tipo de celda.
        kind: MapCell,
        /// Columna.
        x: usize,
        /// Fila.
        y: usize,
    },
}

/// Un dato proyectado: entidad + instante + contenido.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    /// Instante.
    pub time: ItemTime,
    /// Ruta de entidad Rerun (p. ej. `gogamen/series/total_reward`).
    pub entity: String,
    /// Contenido.
    pub payload: Payload,
}

/// Plan de exportación ordenado.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportPlan {
    /// Identificador de aplicación.
    pub application_id: String,
    /// Elementos en orden de emisión.
    pub items: Vec<Item>,
}

impl ExportPlan {
    fn new(options: &ExportOptions) -> Self {
        Self {
            application_id: options.application_id.clone(),
            items: Vec::new(),
        }
    }

    fn push(&mut self, time: ItemTime, entity: impl Into<String>, payload: Payload) {
        self.items.push(Item {
            time,
            entity: entity.into(),
            payload,
        });
    }

    fn series_meta(&mut self, entity: &str, name: &str, unit: &str) {
        self.push(
            ItemTime::Static,
            entity,
            Payload::SeriesMeta {
                name: name.to_owned(),
                unit: unit.to_owned(),
            },
        );
    }

    /// Entidades distintas, en orden de aparición.
    pub fn entities(&self) -> Vec<&str> {
        let mut seen: Vec<&str> = Vec::new();
        for it in &self.items {
            if !seen.contains(&it.entity.as_str()) {
                seen.push(&it.entity);
            }
        }
        seen
    }

    /// Cadena con todo el texto exportado (para auditorías de no-fuga en tests).
    pub fn exported_text(&self) -> String {
        let mut out = String::new();
        for it in &self.items {
            out.push_str(&it.entity);
            out.push('\n');
            match &it.payload {
                Payload::Label(s) | Payload::Message(s) => out.push_str(s),
                Payload::SeriesMeta { name, unit } => {
                    out.push_str(name);
                    out.push_str(unit);
                }
                _ => {}
            }
            out.push('\n');
        }
        out
    }
}

/// Clasificación de un texto libre con la política sensible de la traza científica.
fn classify_text(text: &str, key: &str, options: &ExportOptions) -> Payload {
    let policy = SensitiveFieldPolicy::new(SensitiveFieldMode::Redact);
    if text == REDACTED_MARKER {
        return Payload::Redacted;
    }
    if policy.is_sensitive_key(key) || policy.is_sensitive_value(text) {
        return if options.permissions.secrets && options.permissions.messages {
            Payload::Message(text.to_owned())
        } else {
            Payload::Redacted
        };
    }
    if options.permissions.messages {
        Payload::Message(text.to_owned())
    } else {
        Payload::Withheld
    }
}

/// Identificador no-mensaje (spec/model): sólo se redacta si es sensible.
fn classify_identifier(key: &str, value: &str, options: &ExportOptions) -> Payload {
    match classify_text(value, key, options) {
        Payload::Withheld | Payload::Message(_) => Payload::Label(value.to_owned()),
        other => other,
    }
}

// ---------------------------------------------------------------------------------------------
// Recording (GOG-94)
// ---------------------------------------------------------------------------------------------

/// Proyecta un [`Recording`] Grid: tiempos, entidades, series con unidad, vista de mapa y
/// agente, y huecos explícitos. El tiempo simulado es `tick / 10` s (dt del Grid = 100 ms,
/// ver [`TICKS_PER_SECOND_1X`]).
pub fn plan_from_recording(rec: &Recording, options: &ExportOptions) -> ExportPlan {
    let mut plan = ExportPlan::new(options);
    let s = ItemTime::Static;

    plan.push(
        s,
        "gogamen/meta/source",
        Payload::Label("grid-recording".into()),
    );
    plan.push(
        s,
        "gogamen/meta/seed",
        Payload::Label(rec.seed().to_string()),
    );
    plan.push(
        s,
        "gogamen/meta/time_units",
        Payload::Label("sim_time: seconds (tick / 10); tick: dimensionless step index".into()),
    );

    plan.series_meta("gogamen/series/total_reward", "total_reward", "reward");
    plan.series_meta("gogamen/series/last_reward", "last_reward", "reward");
    plan.series_meta("gogamen/series/collisions", "collisions", "count");

    let map = rec.map();
    for y in 0..map.height() {
        for x in 0..map.width() {
            let pos = gogamen_lab_grid::GridPos::new(x, y);
            let kind = match map.cell(pos).unwrap_or(CellKind::Empty) {
                CellKind::Wall => Some(MapCell::Wall),
                CellKind::Goal => Some(MapCell::Goal),
                CellKind::Hazard => Some(MapCell::Hazard),
                CellKind::Empty if pos == map.start() => Some(MapCell::Start),
                CellKind::Empty => None,
            };
            if let Some(kind) = kind {
                let entity = match kind {
                    MapCell::Wall => "gogamen/scene/map/wall",
                    MapCell::Goal => "gogamen/scene/map/goal",
                    MapCell::Hazard => "gogamen/scene/map/hazard",
                    MapCell::Start => "gogamen/scene/map/start",
                };
                plan.push(s, entity, Payload::Cell { kind, x, y });
            }
        }
    }

    let time_of = |tick: u64| ItemTime::At {
        tick: Some(tick),
        revision: None,
        seconds: Some(tick as f64 / TICKS_PER_SECOND_1X),
    };
    let mut next_expected = 0u64;
    for frame in rec.frames() {
        // Ticks ausentes (hueco) antes de este frame: marcados, jamás interpolados.
        for tick in next_expected..frame.tick {
            push_gap_tick(&mut plan, time_of(tick), tick);
        }
        let t = time_of(frame.tick);
        plan.push(
            t,
            "gogamen/series/total_reward",
            Payload::Scalar(frame.total_reward),
        );
        plan.push(
            t,
            "gogamen/series/last_reward",
            Payload::Scalar(frame.last_reward),
        );
        plan.push(
            t,
            "gogamen/series/collisions",
            Payload::Scalar(frame.collisions as f64),
        );
        plan.push(
            t,
            "gogamen/scene/agent",
            Payload::Point2D {
                x: frame.agent.x as f32 + 0.5,
                y: frame.agent.y as f32 + 0.5,
            },
        );
        plan.push(
            t,
            "gogamen/status/terminal",
            Payload::Label(format!("{:?}", frame.terminal)),
        );
        next_expected = frame.tick + 1;
    }
    // Hueco de cola hasta el último tick declarado.
    for tick in next_expected..=rec.last_tick() {
        push_gap_tick(&mut plan, time_of(tick), tick);
    }
    plan
}

fn push_gap_tick(plan: &mut ExportPlan, t: ItemTime, tick: u64) {
    for series in ["total_reward", "last_reward", "collisions"] {
        plan.push(t, format!("gogamen/series/{series}"), Payload::GapScalar);
    }
    plan.push(t, "gogamen/scene/agent", Payload::GapEntity);
    plan.push(
        t,
        "gogamen/status/missing",
        Payload::Label(format!("missing frame at tick {tick}")),
    );
}

// ---------------------------------------------------------------------------------------------
// ScientificTraceArchive (GOG-93)
// ---------------------------------------------------------------------------------------------

/// Proyecta un [`ScientificTraceArchive`]: series de observación/acción con tiempo simulado,
/// decisiones, anotaciones de checkpoint (texto libre: sujetas a permisos y redacción) y
/// huecos del manifiesto.
pub fn plan_from_archive(archive: &ScientificTraceArchive, options: &ExportOptions) -> ExportPlan {
    let mut plan = ExportPlan::new(options);
    let s = ItemTime::Static;
    let m = &archive.manifest;

    plan.push(
        s,
        "gogamen/meta/source",
        Payload::Label("scientific-trace".into()),
    );
    plan.push(
        s,
        "gogamen/meta/run_id",
        Payload::Label(m.run_id.as_u64().to_string()),
    );
    plan.push(
        s,
        "gogamen/meta/seed",
        Payload::Label(m.seed.as_u64().to_string()),
    );
    plan.push(
        s,
        "gogamen/meta/spec_id",
        classify_identifier("spec_id", &m.spec_id, options),
    );
    plan.push(
        s,
        "gogamen/meta/model_id",
        classify_identifier("model_id", &m.model_id, options),
    );
    plan.push(
        s,
        "gogamen/meta/is_complete",
        Payload::Label(m.is_complete.to_string()),
    );
    plan.push(
        s,
        "gogamen/meta/time_units",
        Payload::Label("sim_time: seconds; tick, revision: dimensionless".into()),
    );

    let dim = archive
        .records
        .first()
        .map_or(0, |r| r.inputs.observation.dimension());
    for i in 0..dim {
        plan.series_meta(
            &format!("gogamen/obs/feature_{i}"),
            &format!("obs_feature_{i}"),
            "unspecified",
        );
    }

    for rec in &archive.records {
        plan_record(&mut plan, rec, options);
    }

    for gap in &m.gaps {
        let t = ItemTime::At {
            tick: None,
            revision: Some(gap.from_revision.as_u64()),
            seconds: None,
        };
        for i in 0..dim {
            plan.push(t, format!("gogamen/obs/feature_{i}"), Payload::GapScalar);
        }
        let reason = match &gap.reason {
            TraceGapReason::QueueOverflow => "queue overflow".to_owned(),
            TraceGapReason::QuotaExceeded => "quota exceeded".to_owned(),
            // El detalle del fallo es texto libre: sólo con permiso.
            TraceGapReason::DiskFailure(detail) if options.permissions.messages => {
                format!("disk failure: {detail}")
            }
            TraceGapReason::DiskFailure(_) => "disk failure".to_owned(),
        };
        plan.push(
            t,
            "gogamen/status/missing",
            Payload::Label(format!(
                "missing {} record(s), revisions {}..={}: {reason}",
                gap.dropped_count,
                gap.from_revision.as_u64(),
                gap.to_revision.as_u64()
            )),
        );
    }
    plan
}

fn plan_record(plan: &mut ExportPlan, rec: &ScientificRecord, options: &ExportOptions) {
    let obs = &rec.inputs.observation;
    let t = ItemTime::At {
        tick: Some(rec.inputs.tick.as_u64()),
        revision: Some(rec.revision.as_u64()),
        seconds: Some(obs.elapsed().as_secs_f64()),
    };
    for (i, v) in obs.features().iter().enumerate() {
        plan.push(t, format!("gogamen/obs/feature_{i}"), Payload::Scalar(*v));
    }

    let d = &rec.decision;
    plan.push(
        t,
        "gogamen/decision/plan_status",
        Payload::Label(format!("{:?}", d.plan_status)),
    );
    let (guard, message) = match &d.guard_decision {
        RecordedGuardDecision::Allow(_) => ("Allow", None),
        RecordedGuardDecision::Fallback(_) => ("Fallback", None),
        RecordedGuardDecision::NoOp => ("NoOp", None),
        RecordedGuardDecision::Reject(msg) => ("Reject", Some(msg.as_str())),
    };
    plan.push(t, "gogamen/decision/guard", Payload::Label(guard.into()));
    if let Some(msg) = message {
        plan.push(
            t,
            "gogamen/decision/guard_message",
            classify_text(msg, "guard_message", options),
        );
    }
    plan.push(
        t,
        "gogamen/decision/used_fallback",
        Payload::Scalar(f64::from(u8::from(d.used_fallback))),
    );
    match &d.applied_action {
        Some(Action::Discrete(a)) => {
            plan.push(
                t,
                "gogamen/action/discrete",
                Payload::Scalar(a.index() as f64),
            );
        }
        Some(Action::Continuous(values)) => {
            for (i, v) in values.iter().enumerate() {
                plan.push(
                    t,
                    format!("gogamen/action/continuous_{i}"),
                    Payload::Scalar(*v),
                );
            }
        }
        // `None` es una decisión (NoOp/Reject), no un dato perdido.
        None => plan.push(t, "gogamen/action/applied", Payload::Label("none".into())),
    }

    if let Some(cp) = &rec.checkpoint {
        for (key, value) in &cp.annotations {
            plan.push(
                t,
                format!("gogamen/checkpoint/annotations/{}", entity_segment(key)),
                classify_text(value, key, options),
            );
        }
    }
}

/// Segmento de entidad seguro (`[A-Za-z0-9_-]`; el resto se sustituye por `_`).
fn entity_segment(raw: &str) -> String {
    let s: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() { "_".to_owned() } else { s }
}
