//! Cámara 2D determinista: pan, zoom con ancla y ajuste a ventana.

use gogamen_projection::{Scene2D, ScenePrimitive};

/// Zoom mínimo (píxeles por unidad de escena).
pub const MIN_ZOOM: f64 = 1e-3;
/// Zoom máximo (píxeles por unidad de escena).
pub const MAX_ZOOM: f64 = 1e4;

/// Caja delimitadora en coordenadas de escena.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
    fn include(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }
}

/// Límites de todos los nodos de la escena; `None` si está vacía.
pub fn scene_bounds(scene: &Scene2D) -> Option<Bounds> {
    let mut b: Option<Bounds> = None;
    let mut add = |x: f64, y: f64| match &mut b {
        Some(bb) => bb.include(x, y),
        None => {
            b = Some(Bounds {
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
            })
        }
    };
    for layer in scene.layers() {
        for node in layer.nodes() {
            match node.primitive() {
                ScenePrimitive::Circle { center, radius } => {
                    add(center.x() - radius, center.y() - radius);
                    add(center.x() + radius, center.y() + radius);
                }
                ScenePrimitive::Polyline { points, .. } => {
                    for p in points {
                        add(p.x(), p.y());
                    }
                }
                ScenePrimitive::Rect { min, width, height }
                | ScenePrimitive::Image {
                    min, width, height, ..
                } => {
                    add(min.x(), min.y());
                    add(min.x() + width, min.y() + height);
                }
                ScenePrimitive::Label { position, .. } => add(position.x(), position.y()),
            }
        }
    }
    b
}

/// Cámara 2D: centro en coordenadas de escena, zoom en px/unidad y tamaño de ventana.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2D {
    center: (f64, f64),
    zoom: f64,
    window: (f64, f64),
    y_up: bool,
}

impl Camera2D {
    /// Crea una cámara centrada en el origen con zoom 1. `y_up` refleja el eje Y de la escena.
    pub fn new(window_w: f64, window_h: f64, y_up: bool) -> Self {
        Self {
            center: (0.0, 0.0),
            zoom: 1.0,
            window: (sanitize_size(window_w), sanitize_size(window_h)),
            y_up,
        }
    }

    /// Identity camera for scenes that are already in pixel space (origin top-left, `y` down):
    /// `world_to_screen` returns its input.
    pub fn screen_space(window_w: f64, window_h: f64) -> Self {
        let (w, h) = (sanitize_size(window_w), sanitize_size(window_h));
        Self {
            center: (w / 2.0, h / 2.0),
            zoom: 1.0,
            window: (w, h),
            y_up: false,
        }
    }

    pub fn center(&self) -> (f64, f64) {
        self.center
    }
    pub fn zoom(&self) -> f64 {
        self.zoom
    }
    pub fn window(&self) -> (f64, f64) {
        self.window
    }
    pub fn y_up(&self) -> bool {
        self.y_up
    }

    /// Cambia el tamaño de ventana manteniendo centro y zoom.
    pub fn resize(&mut self, w: f64, h: f64) {
        self.window = (sanitize_size(w), sanitize_size(h));
    }

    /// Escena -> pantalla (píxeles, origen arriba-izquierda).
    pub fn world_to_screen(&self, x: f64, y: f64) -> (f64, f64) {
        let sx = (x - self.center.0) * self.zoom + self.window.0 / 2.0;
        let dy = (y - self.center.1) * self.zoom;
        let sy = if self.y_up {
            self.window.1 / 2.0 - dy
        } else {
            self.window.1 / 2.0 + dy
        };
        (sx, sy)
    }

    /// Pantalla -> escena.
    pub fn screen_to_world(&self, sx: f64, sy: f64) -> (f64, f64) {
        let x = (sx - self.window.0 / 2.0) / self.zoom + self.center.0;
        let dy = (sy - self.window.1 / 2.0) / self.zoom;
        let y = if self.y_up {
            self.center.1 - dy
        } else {
            self.center.1 + dy
        };
        (x, y)
    }

    /// Arrastra el contenido `dx, dy` píxeles (el contenido sigue al cursor).
    pub fn pan(&mut self, dx_px: f64, dy_px: f64) {
        if !dx_px.is_finite() || !dy_px.is_finite() {
            return;
        }
        self.center.0 -= dx_px / self.zoom;
        if self.y_up {
            self.center.1 += dy_px / self.zoom;
        } else {
            self.center.1 -= dy_px / self.zoom;
        }
    }

    /// Multiplica el zoom por `factor` (acotado) dejando fijo el punto de escena bajo `anchor`.
    pub fn zoom_at(&mut self, factor: f64, anchor_x: f64, anchor_y: f64) {
        if !(factor.is_finite() && factor > 0.0 && anchor_x.is_finite() && anchor_y.is_finite()) {
            return;
        }
        let (wx, wy) = self.screen_to_world(anchor_x, anchor_y);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        self.center.0 = wx - (anchor_x - self.window.0 / 2.0) / self.zoom;
        let dy = (anchor_y - self.window.1 / 2.0) / self.zoom;
        self.center.1 = if self.y_up { wy + dy } else { wy - dy };
    }

    /// Ajusta la cámara para que `bounds` quepa en la ventana con `padding_px` por lado,
    /// preservando la relación de aspecto (zoom uniforme).
    pub fn fit_to_window(
        &mut self,
        window_w: f64,
        window_h: f64,
        bounds: &Bounds,
        padding_px: f64,
    ) {
        self.resize(window_w, window_h);
        let pad = if padding_px.is_finite() {
            padding_px.max(0.0)
        } else {
            0.0
        };
        let avail_w = (self.window.0 - 2.0 * pad).max(1.0);
        let avail_h = (self.window.1 - 2.0 * pad).max(1.0);
        let (bw, bh) = (bounds.width(), bounds.height());
        let zx = if bw > 0.0 {
            avail_w / bw
        } else {
            f64::INFINITY
        };
        let zy = if bh > 0.0 {
            avail_h / bh
        } else {
            f64::INFINITY
        };
        let z = zx.min(zy);
        self.zoom = if z.is_finite() {
            z.clamp(MIN_ZOOM, MAX_ZOOM)
        } else {
            1.0
        };
        self.center = bounds.center();
    }
}

fn sanitize_size(v: f64) -> f64 {
    if v.is_finite() && v >= 1.0 { v } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b() -> Bounds {
        Bounds {
            min_x: -10.0,
            min_y: -5.0,
            max_x: 30.0,
            max_y: 15.0,
        }
    }

    #[test]
    fn fit_keeps_scene_inside_window_and_preserves_aspect() {
        for y_up in [false, true] {
            let mut c = Camera2D::new(100.0, 100.0, y_up);
            c.fit_to_window(800.0, 600.0, &b(), 20.0);
            let (x0, y0) = c.world_to_screen(-10.0, -5.0);
            let (x1, y1) = c.world_to_screen(30.0, 15.0);
            for v in [x0, x1] {
                assert!((20.0 - 1e-9..=780.0 + 1e-9).contains(&v), "x {v}");
            }
            for v in [y0, y1] {
                assert!((20.0 - 1e-9..=580.0 + 1e-9).contains(&v), "y {v}");
            }
            // aspecto: ancho/alto en pantalla == ancho/alto en escena (zoom uniforme)
            let ratio = (x1 - x0).abs() / (y1 - y0).abs();
            assert!((ratio - 40.0 / 20.0).abs() < 1e-9);
            // limitado por ancho: 760/40 = 19
            assert!((c.zoom() - 19.0).abs() < 1e-9);
            let (cx, cy) = c.world_to_screen(10.0, 5.0);
            assert!((cx - 400.0).abs() < 1e-9 && (cy - 300.0).abs() < 1e-9);
        }
    }

    #[test]
    fn fit_degenerate_bounds_is_finite() {
        let mut c = Camera2D::new(100.0, 100.0, false);
        let pt = Bounds {
            min_x: 3.0,
            min_y: 3.0,
            max_x: 3.0,
            max_y: 3.0,
        };
        c.fit_to_window(200.0, 100.0, &pt, 10.0);
        assert_eq!(c.zoom(), 1.0);
        assert_eq!(c.center(), (3.0, 3.0));
    }

    #[test]
    fn zoom_anchor_invariant() {
        for y_up in [false, true] {
            let mut c = Camera2D::new(640.0, 480.0, y_up);
            c.pan(13.0, -7.0);
            let anchor = (123.0, 321.0);
            let before = c.screen_to_world(anchor.0, anchor.1);
            for f in [1.5, 0.3, 2.0, 0.9] {
                c.zoom_at(f, anchor.0, anchor.1);
                let after = c.screen_to_world(anchor.0, anchor.1);
                assert!((after.0 - before.0).abs() < 1e-9);
                assert!((after.1 - before.1).abs() < 1e-9);
                let s = c.world_to_screen(before.0, before.1);
                assert!((s.0 - anchor.0).abs() < 1e-9 && (s.1 - anchor.1).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn zoom_is_clamped() {
        let mut c = Camera2D::new(100.0, 100.0, false);
        c.zoom_at(1e12, 50.0, 50.0);
        assert_eq!(c.zoom(), MAX_ZOOM);
        c.zoom_at(1e-30, 10.0, 10.0);
        assert_eq!(c.zoom(), MIN_ZOOM);
        let snapshot = c;
        c.zoom_at(f64::NAN, 1.0, 1.0);
        c.zoom_at(-2.0, 1.0, 1.0);
        c.pan(f64::INFINITY, 0.0);
        assert_eq!(c, snapshot);
    }

    #[test]
    fn pan_moves_content_with_cursor() {
        for y_up in [false, true] {
            let mut c = Camera2D::new(200.0, 200.0, y_up);
            c.zoom_at(2.0, 100.0, 100.0);
            let before = c.world_to_screen(5.0, 5.0);
            c.pan(30.0, -12.0);
            let after = c.world_to_screen(5.0, 5.0);
            assert!((after.0 - before.0 - 30.0).abs() < 1e-9);
            assert!((after.1 - before.1 + 12.0).abs() < 1e-9);
        }
    }

    #[test]
    fn resize_keeps_center_and_zoom_and_sanitizes() {
        let mut c = Camera2D::new(100.0, 100.0, false);
        c.fit_to_window(400.0, 300.0, &b(), 0.0);
        let (z, ctr) = (c.zoom(), c.center());
        c.resize(800.0, 600.0);
        assert_eq!((c.zoom(), c.center()), (z, ctr));
        // el centro de la escena sigue en el centro de la nueva ventana
        let s = c.world_to_screen(ctr.0, ctr.1);
        assert_eq!(s, (400.0, 300.0));
        c.resize(0.0, f64::NAN);
        assert_eq!(c.window(), (1.0, 1.0));
        // refit tras resize mantiene la escena dentro
        c.fit_to_window(200.0, 900.0, &b(), 0.0);
        let (x0, _) = c.world_to_screen(-10.0, 0.0);
        let (x1, _) = c.world_to_screen(30.0, 0.0);
        assert!(x0 >= -1e-9 && x1 <= 200.0 + 1e-9);
    }

    #[test]
    fn world_screen_roundtrip() {
        let mut c = Camera2D::new(321.0, 123.0, true);
        c.pan(5.0, 9.0);
        c.zoom_at(3.0, 10.0, 20.0);
        let s = c.world_to_screen(1.25, -4.5);
        let w = c.screen_to_world(s.0, s.1);
        assert!((w.0 - 1.25).abs() < 1e-9 && (w.1 + 4.5).abs() < 1e-9);
    }
}
