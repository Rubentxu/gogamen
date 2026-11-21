//! Frontera de backend de viewport y backend de software determinista (SVG).

use std::fmt::Write;

use gogamen_core::EpistemicKind;
use gogamen_projection::{Scene2D, ScenePrimitive};

use crate::camera::Camera2D;

/// Tipo de backend de render. Solo `Window` cuenta como render gráfico real en ventana.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Render por software sin ventana (SVG / framebuffer).
    Software,
    /// Ventana gráfica real (Bevy/egui; diferido, no implementado).
    Window,
}

/// Frontera donde se enchufará un backend Bevy/egui.
pub trait ViewportBackend {
    fn kind(&self) -> BackendKind;
    fn name(&self) -> &'static str;
    /// Renderiza la escena a través de la cámara.
    fn render(&self, scene: &Scene2D, camera: &Camera2D) -> Vec<u8>;
}

/// Backend de software: serializa la escena a SVG de forma byte-determinista.
#[derive(Debug, Clone, Copy, Default)]
pub struct SoftwareSvgBackend;

impl SoftwareSvgBackend {
    /// Renderiza a SVG (texto).
    pub fn render_svg(&self, scene: &Scene2D, camera: &Camera2D) -> String {
        let (w, h) = camera.window();
        let mut out = String::new();
        let _ = writeln!(
            out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">",
            num(w),
            num(h),
            num(w),
            num(h)
        );
        let _ = writeln!(
            out,
            "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
        );
        let mut layers: Vec<_> = scene.layers().iter().filter(|l| l.is_visible()).collect();
        layers.sort_by_key(|l| (l.z_order(), l.id()));
        for layer in layers {
            let _ = writeln!(out, "<g id=\"{}\">", escape(&layer.id().canonical_name()));
            for node in layer.nodes() {
                let color = node_color(node.color(), node.epistemic_kind());
                match node.primitive() {
                    ScenePrimitive::Circle { center, radius } => {
                        let (cx, cy) = camera.world_to_screen(center.x(), center.y());
                        let _ = writeln!(
                            out,
                            "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" fill=\"none\" stroke=\"{color}\"/>",
                            num(cx),
                            num(cy),
                            num(radius * camera.zoom())
                        );
                    }
                    ScenePrimitive::Polyline { points, closed } => {
                        let pts: Vec<String> = points
                            .iter()
                            .map(|p| {
                                let (x, y) = camera.world_to_screen(p.x(), p.y());
                                format!("{},{}", num(x), num(y))
                            })
                            .collect();
                        let tag = if *closed { "polygon" } else { "polyline" };
                        let _ = writeln!(
                            out,
                            "<{tag} points=\"{}\" fill=\"none\" stroke=\"{color}\"/>",
                            pts.join(" ")
                        );
                    }
                    ScenePrimitive::Rect { min, width, height } => {
                        let (ax, ay) = camera.world_to_screen(min.x(), min.y());
                        let (bx, by) = camera.world_to_screen(min.x() + width, min.y() + height);
                        let _ = writeln!(
                            out,
                            "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"none\" stroke=\"{color}\"/>",
                            num(ax.min(bx)),
                            num(ay.min(by)),
                            num((bx - ax).abs()),
                            num((by - ay).abs())
                        );
                    }
                    ScenePrimitive::Image {
                        min,
                        width,
                        height,
                        texture,
                    } => {
                        let (ax, ay) = camera.world_to_screen(min.x(), min.y());
                        let (bx, by) = camera.world_to_screen(min.x() + width, min.y() + height);
                        write_texture_runs(
                            &mut out,
                            texture,
                            (ax.min(bx), ay.min(by)),
                            (
                                (bx - ax).abs() / f64::from(texture.width()),
                                (by - ay).abs() / f64::from(texture.height()),
                            ),
                        );
                    }
                    ScenePrimitive::Label { position, text } => {
                        let (x, y) = camera.world_to_screen(position.x(), position.y());
                        let _ = writeln!(
                            out,
                            "<text x=\"{}\" y=\"{}\" fill=\"{color}\">{}</text>",
                            num(x),
                            num(y),
                            escape(text)
                        );
                    }
                }
            }
            out.push_str("</g>\n");
        }
        out.push_str("</svg>\n");
        out
    }
}

impl ViewportBackend for SoftwareSvgBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Software
    }
    fn name(&self) -> &'static str {
        "software-svg"
    }
    fn render(&self, scene: &Scene2D, camera: &Camera2D) -> Vec<u8> {
        self.render_svg(scene, camera).into_bytes()
    }
}

/// Dibuja una textura como rectángulos (rachas horizontales de color igual, alfa > 0), origen
/// `(x0, y0)` en pantalla y tamaño de píxel `(pw, ph)`. Determinista y sin dependencias.
pub(crate) fn write_texture_runs(
    out: &mut String,
    texture: &gogamen_projection::Texture,
    (x0, y0): (f64, f64),
    (pw, ph): (f64, f64),
) {
    for y in 0..texture.height() {
        let mut x = 0;
        while x < texture.width() {
            let c = texture.rgba_at(x, y).unwrap_or([0; 4]);
            let mut end = x + 1;
            while end < texture.width() && texture.rgba_at(end, y) == Some(c) {
                end += 1;
            }
            if c[3] > 0 {
                let _ = writeln!(
                    out,
                    "<rect x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" fill=\"#{:02x}{:02x}{:02x}\" shape-rendering=\"crispEdges\"/>",
                    num(x0 + f64::from(x) * pw),
                    num(y0 + f64::from(y) * ph),
                    num(f64::from(end - x) * pw),
                    num(ph),
                    c[0],
                    c[1],
                    c[2]
                );
            }
            x = end;
        }
    }
}

fn node_color(over: Option<[u8; 3]>, kind: EpistemicKind) -> String {
    match over {
        Some([r, g, b]) => format!("#{r:02x}{g:02x}{b:02x}"),
        None => color(kind).to_owned(),
    }
}

fn color(kind: EpistemicKind) -> &'static str {
    match kind {
        EpistemicKind::Observed => "#1f77b4",
        EpistemicKind::Inferred => "#ff7f0e",
        EpistemicKind::Hypothetical => "#9467bd",
    }
}

/// Formato numérico fijo (3 decimales, sin `-0.000`).
pub(crate) fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    if s == "-0.000" { "0.000".to_owned() } else { s }
}

pub(crate) fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => o.push_str("&amp;"),
            '<' => o.push_str("&lt;"),
            '>' => o.push_str("&gt;"),
            '"' => o.push_str("&quot;"),
            _ => o.push(c),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::scene_bounds;
    use crate::fixtures::scene_alpha;

    #[test]
    fn render_is_byte_identical_and_matches_golden() {
        let scene = scene_alpha();
        let mut cam = Camera2D::new(200.0, 100.0, true);
        cam.fit_to_window(200.0, 100.0, &scene_bounds(&scene).unwrap(), 10.0);
        let b = SoftwareSvgBackend;
        let a1 = b.render(&scene, &cam);
        let a2 = b.render(&scene.clone(), &cam.clone());
        assert_eq!(a1, a2);
        let svg = String::from_utf8(a1).unwrap();
        let golden = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"200.000\" height=\"100.000\" viewBox=\"0 0 200.000 100.000\">\n\
<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n\
<g id=\"environment\">\n\
<circle cx=\"100.000\" cy=\"50.000\" r=\"40.000\" fill=\"none\" stroke=\"#1f77b4\"/>\n\
<text x=\"140.000\" y=\"10.000\" fill=\"#1f77b4\">a&amp;b</text>\n\
</g>\n</svg>\n";
        assert_eq!(svg, golden);
        assert_eq!(b.kind(), BackendKind::Software);
    }

    #[test]
    fn camera_changes_output() {
        let scene = scene_alpha();
        let mut cam = Camera2D::new(200.0, 100.0, true);
        let b = SoftwareSvgBackend;
        let before = b.render_svg(&scene, &cam);
        cam.pan(5.0, 5.0);
        assert_ne!(before, b.render_svg(&scene, &cam));
    }
}
