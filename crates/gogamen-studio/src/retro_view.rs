//! Vista retro: framebuffer + overlays en coordenadas de imagen (GOG-111).
//!
//! # Datos, no emulador
//! Este módulo es una proyección pura `RetroFrameView -> Scene2D / SVG`. No ejecuta ni conoce
//! ninguna dinámica de emulador (test textual de guarda): recibe un frame ya capturado (bytes
//! sin tocar), items de overlay ya extraídos y contadores ya medidos.
//!
//! # Percepción
//! Cada [`OverlayItem`] lleva su [`PerceptionSource`]:
//! - `Privileged`: leído del estado interno del emulador (RAM). Se dibuja discontinuo, verde, con
//!   prefijo `[RAM]`, en la capa personalizada [`PRIVILEGED_LAYER`], epistemología `Observed`.
//! - `PixelOnly`: salida de un detector que sólo vio píxeles. Se dibuja continuo, naranja, con
//!   prefijo `[PX]`, en la capa `Sensors`, epistemología `Inferred`.
//!
//! En modo [`PerceptionMode::PixelOnly`] sólo se muestran detecciones `PixelOnly` aportadas
//! (`RetroFrameView::detections`); sin salida de detector no se muestra **nada**: nunca se
//! sintetizan detecciones ni se filtran datos privilegiados.
//!
//! # Relación píxel/pantalla
//! La escena usa coordenadas de imagen (px, eje Y hacia abajo, origen arriba-izquierda). El SVG usa
//! [`ImageViewport`] (escala entera + letterbox entero; ver `gogamen_projection::image`).

use std::fmt;
use std::fmt::Write;

use gogamen_core::{EpistemicKind, Tick};
use gogamen_projection::{
    CoordinateSpace, ImageViewport, LayerId, Point2, ProjectionError, ScaleUnits, Scene2D, SceneId,
    SceneLayer, SceneNode, SceneNodeId, ScenePrimitive, Texture,
};

use crate::viewport::{escape, num, write_texture_runs};

/// Capa de la escena para overlays privilegiados.
pub const PRIVILEGED_LAYER: LayerId = LayerId::Custom(1);

/// Origen de un item de overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionSource {
    /// Estado interno del emulador (RAM): no disponible para un agente que sólo ve píxeles.
    Privileged,
    /// Salida de un detector sobre píxeles.
    PixelOnly,
}

impl PerceptionSource {
    /// Prefijo visible de la etiqueta.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Privileged => "[RAM]",
            Self::PixelOnly => "[PX]",
        }
    }
}

/// Qué percepción muestra la vista.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PerceptionMode {
    /// Privilegiada + detecciones de píxeles (cada una con su etiqueta).
    Privileged,
    /// Sólo detecciones de píxeles aportadas por un detector.
    PixelOnly,
}

/// Caja en coordenadas de imagen (píxeles enteros).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OverlayItem {
    pub source: PerceptionSource,
    pub label: String,
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Contadores de frames (separados de los comandos de emulador).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameCounters {
    pub produced: u64,
    pub published: u64,
    pub consumed: u64,
    pub dropped_slow: u64,
    pub dropped_disconnected: u64,
}

/// Contadores de comandos de control de emulador.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CommandCounters {
    pub received: u64,
    pub applied: u64,
    pub rejected: u64,
}

/// Errores de construcción de la vista.
#[derive(Debug, Clone, PartialEq)]
pub enum RetroViewError {
    /// Un item no coincide con la lista donde se aportó.
    WrongSource,
    /// Caja fuera de la imagen o vacía.
    ItemOutOfBounds,
    /// La licencia/atribución es obligatoria.
    MissingLicense,
    /// Error de proyección.
    Projection(ProjectionError),
}

impl fmt::Display for RetroViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "retro view error: {self:?}")
    }
}

impl std::error::Error for RetroViewError {}

impl From<ProjectionError> for RetroViewError {
    fn from(e: ProjectionError) -> Self {
        Self::Projection(e)
    }
}

/// Frame capturado + overlays + metadatos para mostrar.
#[derive(Debug, Clone, PartialEq)]
pub struct RetroFrameView {
    frame_index: u64,
    texture: Texture,
    privileged: Vec<OverlayItem>,
    detections: Vec<OverlayItem>,
    license: String,
    frames: FrameCounters,
    commands: CommandCounters,
}

impl RetroFrameView {
    /// Valida y construye la vista. `privileged` sólo admite items `Privileged`; `detections`
    /// sólo `PixelOnly`; las cajas deben caber en la imagen; la licencia no puede ser vacía.
    pub fn new(
        frame_index: u64,
        texture: Texture,
        privileged: Vec<OverlayItem>,
        detections: Vec<OverlayItem>,
        license: impl Into<String>,
        frames: FrameCounters,
        commands: CommandCounters,
    ) -> Result<Self, RetroViewError> {
        let license = license.into();
        if license.trim().is_empty() {
            return Err(RetroViewError::MissingLicense);
        }
        let check = |items: &[OverlayItem], want: PerceptionSource| {
            for i in items {
                if i.source != want {
                    return Err(RetroViewError::WrongSource);
                }
                let fits = i.w > 0
                    && i.h > 0
                    && u64::from(i.x) + u64::from(i.w) <= u64::from(texture.width())
                    && u64::from(i.y) + u64::from(i.h) <= u64::from(texture.height());
                if !fits {
                    return Err(RetroViewError::ItemOutOfBounds);
                }
            }
            Ok(())
        };
        check(&privileged, PerceptionSource::Privileged)?;
        check(&detections, PerceptionSource::PixelOnly)?;
        Ok(Self {
            frame_index,
            texture,
            privileged,
            detections,
            license,
            frames,
            commands,
        })
    }

    pub fn texture(&self) -> &Texture {
        &self.texture
    }

    pub fn license(&self) -> &str {
        &self.license
    }

    pub fn frame_index(&self) -> u64 {
        self.frame_index
    }

    pub fn frame_counters(&self) -> FrameCounters {
        self.frames
    }

    pub fn command_counters(&self) -> CommandCounters {
        self.commands
    }

    /// Items visibles en `mode`, en orden estable (privilegiados primero).
    pub fn visible_items(&self, mode: PerceptionMode) -> Vec<&OverlayItem> {
        match mode {
            PerceptionMode::Privileged => self
                .privileged
                .iter()
                .chain(self.detections.iter())
                .collect(),
            PerceptionMode::PixelOnly => self.detections.iter().collect(),
        }
    }
}

fn pt(x: f64, y: f64) -> Result<Point2, ProjectionError> {
    Point2::new(x, y)
}

/// Proyecta la vista a una [`Scene2D`] en coordenadas de imagen (px, Y hacia abajo).
pub fn project_retro_scene(
    view: &RetroFrameView,
    mode: PerceptionMode,
) -> Result<Scene2D, ProjectionError> {
    let space = CoordinateSpace::new(ScaleUnits::Custom, 1.0, pt(0.0, 0.0)?, false)?;
    let mut scene = Scene2D::new(
        SceneId::new(view.frame_index),
        Tick::new(view.frame_index),
        EpistemicKind::Observed,
        space,
    );
    scene.add_layer(SceneLayer::new(LayerId::Environment))?;
    scene.add_layer(SceneLayer::new(LayerId::Sensors))?;
    scene.add_layer(SceneLayer::new(PRIVILEGED_LAYER))?;
    scene.add_layer(SceneLayer::new(LayerId::Labels))?;
    let (w, h) = (
        f64::from(view.texture.width()),
        f64::from(view.texture.height()),
    );
    let mut next = 1u64;
    let mut id = || {
        let n = SceneNodeId::new(next);
        next += 1;
        n
    };
    scene.add_node(
        LayerId::Environment,
        SceneNode::new(
            id(),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::image(pt(0.0, 0.0)?, w, h, view.texture.clone())?,
            false,
        ),
    )?;
    for item in view.visible_items(mode) {
        let (layer, kind) = match item.source {
            PerceptionSource::Privileged => (PRIVILEGED_LAYER, EpistemicKind::Observed),
            PerceptionSource::PixelOnly => (LayerId::Sensors, EpistemicKind::Inferred),
        };
        let (x, y) = (f64::from(item.x), f64::from(item.y));
        scene.add_node(
            layer,
            SceneNode::new(
                id(),
                None,
                kind,
                ScenePrimitive::rect(pt(x, y)?, f64::from(item.w), f64::from(item.h))?,
                true,
            ),
        )?;
        scene.add_node(
            LayerId::Labels,
            SceneNode::new(
                id(),
                None,
                kind,
                ScenePrimitive::label(pt(x, y)?, format!("{} {}", item.source.tag(), item.label)),
                false,
            ),
        )?;
    }
    Ok(scene)
}

/// Renderiza la vista a SVG determinista en una ventana `view_w x view_h`.
pub fn render_retro_svg(
    view: &RetroFrameView,
    mode: PerceptionMode,
    view_w: u32,
    view_h: u32,
) -> Result<String, RetroViewError> {
    let scene = project_retro_scene(view, mode)?;
    let vp = ImageViewport::fit(view.texture.width(), view.texture.height(), view_w, view_h)?;
    let mut out = String::new();
    let _ = writeln!(
        out,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{view_w}\" height=\"{view_h}\" viewBox=\"0 0 {view_w} {view_h}\">"
    );
    let _ = writeln!(out, "<title>{}</title>", escape(&view.license));
    let _ = writeln!(
        out,
        "<rect width=\"100%\" height=\"100%\" fill=\"#202020\"/>"
    );
    let mut layers: Vec<_> = scene.layers().iter().filter(|l| l.is_visible()).collect();
    layers.sort_by_key(|l| (l.z_order(), l.id()));
    for layer in layers {
        let _ = writeln!(out, "<g id=\"{}\">", escape(&layer.id().canonical_name()));
        let privileged = layer.id() == PRIVILEGED_LAYER;
        let (stroke, dash) = if privileged {
            ("#00c853", " stroke-dasharray=\"4 2\"")
        } else {
            ("#ff6d00", "")
        };
        for node in layer.nodes() {
            match node.primitive() {
                ScenePrimitive::Image { texture, .. } => {
                    let (x, y) = vp.image_to_screen(0.0, 0.0);
                    write_texture_runs(&mut out, texture, (x, y), (vp.scale(), vp.scale()));
                }
                ScenePrimitive::Rect { min, width, height } => {
                    let (x, y, w, h) = vp.image_rect_to_screen(min.x(), min.y(), *width, *height);
                    let _ = writeln!(
                        out,
                        "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{stroke}\"{dash}/>",
                        num(x),
                        num(y),
                        num(w),
                        num(h)
                    );
                }
                ScenePrimitive::Label { position, text } => {
                    let (x, y) = vp.image_to_screen(position.x(), position.y());
                    let color = if text.starts_with(PerceptionSource::Privileged.tag()) {
                        "#00c853"
                    } else {
                        "#ff6d00"
                    };
                    let _ = writeln!(
                        out,
                        "<text x=\"{}\" y=\"{}\" fill=\"{color}\" font-size=\"10\">{}</text>",
                        num(x),
                        num((y - 2.0).max(10.0)),
                        escape(text)
                    );
                }
                ScenePrimitive::Circle { .. } | ScenePrimitive::Polyline { .. } => {}
            }
        }
        out.push_str("</g>\n");
    }
    let f = view.frames;
    let c = view.commands;
    let _ = writeln!(
        out,
        "<text x=\"4\" y=\"{}\" fill=\"#ffffff\" font-size=\"10\">frames produced={} published={} consumed={} dropped_slow={} dropped_disconnected={} | commands received={} applied={} rejected={}</text>",
        view_h.saturating_sub(14),
        f.produced,
        f.published,
        f.consumed,
        f.dropped_slow,
        f.dropped_disconnected,
        c.received,
        c.applied,
        c.rejected
    );
    let _ = writeln!(
        out,
        "<text x=\"4\" y=\"{}\" fill=\"#cccccc\" font-size=\"9\">license: {}</text>",
        view_h.saturating_sub(3),
        escape(&view.license)
    );
    out.push_str("</svg>\n");
    Ok(out)
}

#[cfg(test)]
mod guard {
    /// Guarda textual: la proyección/render no contiene dinámica de emulador.
    #[test]
    fn projection_code_has_no_emulator_dynamics() {
        let src = include_str!("retro_view.rs");
        let code = src.split("#[cfg(test)]").next().unwrap();
        for banned in [
            "gogamen_lab_retro",
            "FreePlayCore",
            "LibretroCorePort",
            "Adapter",
            "run_frame",
            ".step(",
            "retro_session",
            "LiveEnvironment",
            "Actuator",
        ] {
            assert!(!code.contains(banned), "retro_view references {banned}");
        }
    }
}
