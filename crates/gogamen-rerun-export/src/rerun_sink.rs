//! Emisor Rerun (feature `rerun`): consume un [`ExportPlan`] y escribe un `.rrd`.
//!
//! La escritura es **descartable**: se emite a `<destino>.partial` y sólo se renombra al
//! destino final si todo salió bien; ante cualquier error el parcial se elimina. La traza
//! fuente sólo se recibe por referencia inmutable, de modo que exportar (o fallar al
//! exportar) no puede alterar el replay.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use gogamen_runtime::ScientificTraceArchive;
use gogamen_studio::replay::Recording;
use rerun::blueprint::{
    Blueprint, BlueprintOpts, Horizontal, Spatial2DView, TextLogView, TimeSeriesView, Vertical,
};

use crate::plan::{
    ExportOptions, ExportPlan, Item, ItemTime, MapCell, Payload, REDACTED_MARKER,
    TIMELINE_REVISION, TIMELINE_SIM_TIME, TIMELINE_TICK, WITHHELD_MARKER, plan_from_archive,
    plan_from_recording,
};

/// Error de exportación (la fuente nunca resulta afectada).
#[derive(Debug)]
pub enum ExportError {
    /// Error del SDK de Rerun.
    Rerun(String),
    /// Error de E/S al finalizar el archivo.
    Io(std::io::Error),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rerun(m) => write!(f, "rerun export failed: {m}"),
            Self::Io(e) => write!(f, "rerun export I/O error: {e}"),
        }
    }
}

impl std::error::Error for ExportError {}

fn rr<E: fmt::Display>(e: E) -> ExportError {
    ExportError::Rerun(e.to_string())
}

/// Exporta un [`Recording`] Grid a `path` (`.rrd`).
pub fn export_recording(
    rec: &Recording,
    options: &ExportOptions,
    path: &Path,
) -> Result<(), ExportError> {
    write_plan(&plan_from_recording(rec, options), path)
}

/// Exporta un [`ScientificTraceArchive`] a `path` (`.rrd`).
pub fn export_archive(
    archive: &ScientificTraceArchive,
    options: &ExportOptions,
    path: &Path,
) -> Result<(), ExportError> {
    write_plan(&plan_from_archive(archive, options), path)
}

fn partial_path(path: &Path) -> PathBuf {
    let mut os = path.as_os_str().to_owned();
    os.push(".partial");
    PathBuf::from(os)
}

/// Escribe el plan de forma atómica: parcial + rename; el parcial se descarta si falla.
pub fn write_plan(plan: &ExportPlan, path: &Path) -> Result<(), ExportError> {
    let partial = partial_path(path);
    let result = emit(plan, &partial)
        .and_then(|()| std::fs::rename(&partial, path).map_err(ExportError::Io));
    if result.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    result
}

fn emit(plan: &ExportPlan, out: &Path) -> Result<(), ExportError> {
    let rec = rerun::RecordingStreamBuilder::new(
        rerun::ApplicationId::try_from(plan.application_id.clone()).map_err(rr)?,
    )
    .save(out)
    .map_err(rr)?;

    send_blueprint(&rec, plan)?;

    // Celdas de mapa estáticas agrupadas por entidad (una caja por celda).
    let mut cells: BTreeMap<&str, Vec<(MapCell, usize, usize)>> = BTreeMap::new();
    for item in &plan.items {
        if let Payload::Cell { kind, x, y } = &item.payload {
            cells
                .entry(item.entity.as_str())
                .or_default()
                .push((*kind, *x, *y));
        }
    }
    for (entity, list) in &cells {
        let centers: Vec<(f32, f32)> = list
            .iter()
            .map(|(_, x, y)| (*x as f32 + 0.5, *y as f32 + 0.5))
            .collect();
        let kind = list[0].0;
        let (color, label) = match kind {
            MapCell::Wall => ((110, 110, 120), "wall"),
            MapCell::Goal => ((60, 200, 90), "goal"),
            MapCell::Hazard => ((220, 70, 60), "hazard"),
            MapCell::Start => ((70, 120, 230), "start"),
        };
        rec.log_static(
            *entity,
            &rerun::Boxes2D::from_centers_and_half_sizes(centers, vec![(0.5, 0.5); list.len()])
                .with_colors(vec![
                    rerun::Color::from_rgb(color.0, color.1, color.2);
                    list.len()
                ])
                .with_labels(vec![label; list.len()]),
        )
        .map_err(rr)?;
    }

    for item in &plan.items {
        emit_item(&rec, item)?;
    }
    rec.flush_blocking().map_err(rr)?;
    drop(rec);
    Ok(())
}

fn send_blueprint(rec: &rerun::RecordingStream, plan: &ExportPlan) -> Result<(), ExportError> {
    let has = |prefix: &str| plan.items.iter().any(|i| i.entity.starts_with(prefix));
    let mut left = Vec::new();
    if has("gogamen/scene/") {
        left.push(
            Spatial2DView::new("Scene")
                .with_origin("gogamen/scene")
                .into(),
        );
    }
    let mut series_contents = Vec::new();
    for p in ["series", "obs", "action"] {
        if has(&format!("gogamen/{p}/")) {
            series_contents.push(format!("/gogamen/{p}/**"));
        }
    }
    let mut right = Vec::new();
    if !series_contents.is_empty() {
        right.push(
            TimeSeriesView::new("Series")
                .with_origin("/")
                .with_contents(series_contents)
                .into(),
        );
    }
    right.push(
        TextLogView::new("Status, decisions and gaps")
            .with_origin("/")
            .with_contents([
                "/gogamen/status/**",
                "/gogamen/decision/**",
                "/gogamen/checkpoint/**",
            ])
            .into(),
    );
    let mut cols = Vec::new();
    if !left.is_empty() {
        cols.push(Vertical::new(left).into());
    }
    cols.push(Vertical::new(right).into());
    let blueprint = Blueprint::new(Horizontal::new(cols));
    rec.send_blueprint_opts(&BlueprintOpts {
        blueprint,
        activation: Default::default(),
    })
    .map_err(rr)
}

fn set_times(rec: &rerun::RecordingStream, time: &ItemTime) {
    if let ItemTime::At {
        tick,
        revision,
        seconds,
    } = time
    {
        match tick {
            Some(t) => rec.set_time(TIMELINE_TICK, rerun::TimeCell::from_sequence(*t as i64)),
            None => rec.disable_timeline(TIMELINE_TICK),
        }
        match revision {
            Some(r) => rec.set_time(TIMELINE_REVISION, rerun::TimeCell::from_sequence(*r as i64)),
            None => rec.disable_timeline(TIMELINE_REVISION),
        }
        match seconds {
            Some(s) => rec.set_time(
                TIMELINE_SIM_TIME,
                rerun::TimeCell::from_duration_nanos((*s * 1e9).round() as i64),
            ),
            None => rec.disable_timeline(TIMELINE_SIM_TIME),
        }
    }
}

fn emit_item(rec: &rerun::RecordingStream, item: &Item) -> Result<(), ExportError> {
    let path = item.entity.as_str();
    set_times(rec, &item.time);
    let is_static = matches!(item.time, ItemTime::Static);
    match &item.payload {
        Payload::Cell { .. } => Ok(()), // ya emitido agrupado
        Payload::SeriesMeta { name, unit } => rec
            .log_static(
                path,
                &rerun::SeriesLines::new().with_names([format!("{name} [{unit}]")]),
            )
            .map_err(rr),
        Payload::Scalar(v) => rec.log(path, &rerun::Scalars::single(*v)).map_err(rr),
        // Hueco explícito: NaN corta la línea del gráfico; jamás se rellena con 0.
        Payload::GapScalar => rec.log(path, &rerun::Scalars::single(f64::NAN)).map_err(rr),
        Payload::GapEntity => rec.log(path, &rerun::Clear::flat()).map_err(rr),
        Payload::Point2D { x, y } => rec
            .log(
                path,
                &rerun::Points2D::new([(*x, *y)])
                    .with_radii([0.3])
                    .with_colors([rerun::Color::from_rgb(250, 200, 40)])
                    .with_labels(["agent"]),
            )
            .map_err(rr),
        Payload::Label(s) | Payload::Message(s) => text(rec, path, is_static, s, "INFO"),
        Payload::Redacted => text(rec, path, is_static, REDACTED_MARKER, "WARN"),
        Payload::Withheld => text(rec, path, is_static, WITHHELD_MARKER, "WARN"),
    }
}

fn text(
    rec: &rerun::RecordingStream,
    path: &str,
    is_static: bool,
    body: &str,
    level: &str,
) -> Result<(), ExportError> {
    if is_static {
        rec.log_static(path, &rerun::TextDocument::new(body))
            .map_err(rr)
    } else {
        rec.log(path, &rerun::TextLog::new(body).with_level(level))
            .map_err(rr)
    }
}
