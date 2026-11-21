//! Lightweight, deterministic 3D scene description and orbit-camera projection to [`Scene2D`]
//! (`std`-only, no GPU, no physics).
//!
//! # Coordinate system (documented contract)
//! - World frame: **right-handed**, `x` forward/east, `y` left/north, `z` up. Units: **metres**.
//! - Angles are **radians**. `yaw` is a counter-clockwise rotation about `+z` (so yaw `0` points
//!   along `+x`, yaw `pi/2` along `+y`).
//! - Screen (projection output): pixels, origin top-left, `+x` right, `+y` down.
//!
//! # Camera
//! [`OrbitCamera`] orbits a `target`: `azimuth` is the angle of the eye around `+z` measured from
//! `+x`, `elevation` is the angle of the eye above the `xy` plane, `distance` the eye-target
//! distance. The camera is pure *view state*: it is never consulted by any simulation.
//!
//! The projection output is a [`Scene2D`] in pixel space, so any 2D backend (software SVG, egui
//! window) renders it unchanged. Output is a pure function of `(scene, camera)`.

use std::collections::BTreeMap;
use std::ops::{Add, Sub};

use gogamen_core::{EpistemicKind, Tick};

use crate::coords::{CoordinateSpace, Point2, ScaleUnits};
use crate::scene::{LayerId, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId, ScenePrimitive};

/// 3D vector / point (metres in world space).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// Forward/east component.
    pub x: f64,
    /// Left/north component.
    pub y: f64,
    /// Up component.
    pub z: f64,
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Vec3 {
    /// Origin.
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    /// World `+x`.
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    /// World `+y`.
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    /// World `+z` (up).
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    /// Builds a vector.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
    /// `self * k`.
    pub fn scale(self, k: f64) -> Self {
        Self::new(self.x * k, self.y * k, self.z * k)
    }
    /// Dot product.
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }
    /// Cross product (right-handed).
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }
    /// Euclidean norm.
    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }
    /// Unit vector (the input itself if its norm is zero).
    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 0.0 { self.scale(1.0 / n) } else { self }
    }
    /// `true` if all components are finite.
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

/// Projection model of the camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Projection {
    /// Pinhole perspective.
    Perspective,
    /// Orthographic; the scale equals the perspective scale at the target depth.
    Orthographic,
}

/// A projected point: pixel coordinates plus camera-space depth (metres along the view axis).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScreenPoint {
    /// Pixel x (right).
    pub x: f64,
    /// Pixel y (down).
    pub y: f64,
    /// Depth along the forward axis (`> NEAR` when visible).
    pub depth: f64,
}

/// Near clipping distance (m).
pub const NEAR: f64 = 0.05;
/// Minimum eye-target distance (m).
pub const MIN_DISTANCE: f64 = 0.25;
/// Maximum eye-target distance (m).
pub const MAX_DISTANCE: f64 = 500.0;
/// Elevation clamp (rad): just under 90 degrees so the basis never degenerates.
pub const MAX_ELEVATION: f64 = 1.553_343; // 89 degrees

/// Orbit camera: pure view state (never part of a physics snapshot).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    target: Vec3,
    azimuth: f64,
    elevation: f64,
    distance: f64,
    fov_y: f64,
    projection: Projection,
    viewport: (f64, f64),
}

impl OrbitCamera {
    /// New perspective camera (60 degree vertical field of view).
    pub fn new(
        target: Vec3,
        azimuth: f64,
        elevation: f64,
        distance: f64,
        viewport: (f64, f64),
    ) -> Self {
        let mut c = Self {
            target: if target.is_finite() {
                target
            } else {
                Vec3::ZERO
            },
            azimuth: 0.0,
            elevation: 0.0,
            distance: 1.0,
            fov_y: 60.0_f64.to_radians(),
            projection: Projection::Perspective,
            viewport: (1.0, 1.0),
        };
        c.azimuth = if azimuth.is_finite() { azimuth } else { 0.0 };
        c.elevation = clamp_elev(elevation);
        c.distance = clamp_dist(distance);
        c.set_viewport(viewport.0, viewport.1);
        c
    }

    /// Orbit target (look-at point).
    pub fn target(&self) -> Vec3 {
        self.target
    }
    /// Azimuth (rad).
    pub fn azimuth(&self) -> f64 {
        self.azimuth
    }
    /// Elevation (rad).
    pub fn elevation(&self) -> f64 {
        self.elevation
    }
    /// Eye-target distance (m).
    pub fn distance(&self) -> f64 {
        self.distance
    }
    /// Projection model.
    pub fn projection(&self) -> Projection {
        self.projection
    }
    /// Viewport size in pixels.
    pub fn viewport(&self) -> (f64, f64) {
        self.viewport
    }

    /// Sets the viewport (pixels); non-finite or tiny values are clamped to 1.
    pub fn set_viewport(&mut self, w: f64, h: f64) {
        let s = |v: f64| if v.is_finite() && v >= 1.0 { v } else { 1.0 };
        self.viewport = (s(w), s(h));
    }

    /// Switches the projection model.
    pub fn set_projection(&mut self, p: Projection) {
        self.projection = p;
    }

    /// Orbits by `(d_azimuth, d_elevation)` radians (elevation clamped; non-finite ignored).
    pub fn orbit(&mut self, d_azimuth: f64, d_elevation: f64) {
        if d_azimuth.is_finite() && d_elevation.is_finite() {
            self.azimuth += d_azimuth;
            self.elevation = clamp_elev(self.elevation + d_elevation);
        }
    }

    /// Dolly zoom: multiplies the magnification by `factor` (`distance /= factor`, clamped).
    pub fn zoom(&mut self, factor: f64) {
        if factor.is_finite() && factor > 0.0 {
            self.distance = clamp_dist(self.distance / factor);
        }
    }

    /// Pans the target in the camera's screen plane projected on the ground: dragging by
    /// `(dx_px, dy_px)` makes the scene follow the cursor horizontally and moves it along the
    /// ground-forward direction vertically.
    pub fn pan_pixels(&mut self, dx_px: f64, dy_px: f64) {
        if !(dx_px.is_finite() && dy_px.is_finite()) {
            return;
        }
        let (right, _up, fwd) = self.basis();
        let flat = Vec3::new(fwd.x, fwd.y, 0.0).normalized();
        let m_per_px = self.distance / self.focal_px();
        self.target = self
            .target
            .add(right.scale(-dx_px * m_per_px))
            .add(flat.scale(dy_px * m_per_px));
    }

    /// Camera position in world space.
    pub fn eye(&self) -> Vec3 {
        let (ce, se) = (self.elevation.cos(), self.elevation.sin());
        self.target.add(
            Vec3::new(ce * self.azimuth.cos(), ce * self.azimuth.sin(), se).scale(self.distance),
        )
    }

    /// Orthonormal camera basis `(right, up, forward)` in world space.
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = self.target.sub(self.eye()).normalized();
        let right = forward.cross(Vec3::Z).normalized();
        let up = right.cross(forward);
        (right, up, forward)
    }

    /// Focal length in pixels (`(h/2) / tan(fov_y/2)`).
    pub fn focal_px(&self) -> f64 {
        (self.viewport.1 / 2.0) / (self.fov_y / 2.0).tan()
    }

    /// World point to camera space `(right, up, depth)`.
    pub fn to_camera_space(&self, p: Vec3) -> Vec3 {
        let (right, up, fwd) = self.basis();
        let d = p.sub(self.eye());
        Vec3::new(d.dot(right), d.dot(up), d.dot(fwd))
    }

    fn screen_of(&self, c: Vec3) -> ScreenPoint {
        let f = self.focal_px();
        let k = match self.projection {
            Projection::Perspective => f / c.z,
            Projection::Orthographic => f / self.distance,
        };
        ScreenPoint {
            x: self.viewport.0 / 2.0 + c.x * k,
            y: self.viewport.1 / 2.0 - c.y * k,
            depth: c.z,
        }
    }

    /// Projects a world point; `None` if it is behind the near plane or non-finite.
    pub fn project(&self, p: Vec3) -> Option<ScreenPoint> {
        if !p.is_finite() {
            return None;
        }
        let c = self.to_camera_space(p);
        (c.z >= NEAR).then(|| self.screen_of(c))
    }

    /// Projects a segment, clipping it against the near plane; `None` if fully clipped.
    pub fn project_segment(&self, a: Vec3, b: Vec3) -> Option<(ScreenPoint, ScreenPoint)> {
        if !(a.is_finite() && b.is_finite()) {
            return None;
        }
        let (mut ca, mut cb) = (self.to_camera_space(a), self.to_camera_space(b));
        if ca.z < NEAR && cb.z < NEAR {
            return None;
        }
        let lerp = |p: Vec3, q: Vec3| {
            let t = (NEAR - p.z) / (q.z - p.z);
            p.add(q.sub(p).scale(t))
        };
        if ca.z < NEAR {
            ca = lerp(ca, cb);
        } else if cb.z < NEAR {
            cb = lerp(cb, ca);
        }
        Some((self.screen_of(ca), self.screen_of(cb)))
    }
}

fn clamp_elev(e: f64) -> f64 {
    if e.is_finite() {
        e.clamp(-MAX_ELEVATION, MAX_ELEVATION)
    } else {
        0.0
    }
}

fn clamp_dist(d: f64) -> f64 {
    if d.is_finite() {
        d.clamp(MIN_DISTANCE, MAX_DISTANCE)
    } else {
        1.0
    }
}

/// 3D primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum Primitive3D {
    /// Polyline (optionally closed) through 3D points.
    Polyline {
        /// Vertices (m).
        points: Vec<Vec3>,
        /// Connect the last vertex to the first.
        closed: bool,
    },
    /// Text anchored at a 3D point.
    Label {
        /// Anchor (m).
        position: Vec3,
        /// Text.
        text: String,
    },
    /// Text at a fixed pixel position (HUD / legend); independent of the camera.
    ScreenLabel {
        /// Pixel x.
        x: f64,
        /// Pixel y.
        y: f64,
        /// Text.
        text: String,
    },
}

/// A node of a [`Scene3D`].
#[derive(Debug, Clone, PartialEq)]
pub struct Node3D {
    /// 2D layer it is projected into (z-ordering and visibility).
    pub layer: LayerId,
    /// Epistemic classification (drives the default stroke colour).
    pub kind: EpistemicKind,
    /// Optional colour override `[r, g, b]`.
    pub color: Option<[u8; 3]>,
    /// Optional pick tag (copied to the projected node's `state_code`).
    pub tag: Option<u32>,
    /// Geometry.
    pub primitive: Primitive3D,
}

impl Node3D {
    /// Polyline node.
    pub fn polyline(layer: LayerId, kind: EpistemicKind, points: Vec<Vec3>, closed: bool) -> Self {
        Self {
            layer,
            kind,
            color: None,
            tag: None,
            primitive: Primitive3D::Polyline { points, closed },
        }
    }
    /// Label node.
    pub fn label(
        layer: LayerId,
        kind: EpistemicKind,
        position: Vec3,
        text: impl Into<String>,
    ) -> Self {
        Self {
            layer,
            kind,
            color: None,
            tag: None,
            primitive: Primitive3D::Label {
                position,
                text: text.into(),
            },
        }
    }
    /// Screen-space (HUD) label at pixel `(x, y)`; ignores the camera.
    pub fn screen_label(
        layer: LayerId,
        kind: EpistemicKind,
        x: f64,
        y: f64,
        text: impl Into<String>,
    ) -> Self {
        Self {
            layer,
            kind,
            color: None,
            tag: None,
            primitive: Primitive3D::ScreenLabel {
                x,
                y,
                text: text.into(),
            },
        }
    }
    /// Sets the colour override.
    pub fn with_color(mut self, rgb: [u8; 3]) -> Self {
        self.color = Some(rgb);
        self
    }
    /// Sets the pick tag.
    pub fn with_tag(mut self, tag: u32) -> Self {
        self.tag = Some(tag);
        self
    }
}

/// Deterministic 3D scene: an ordered list of nodes.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene3D {
    scene_id: SceneId,
    tick: Tick,
    nodes: Vec<Node3D>,
}

impl Scene3D {
    /// Empty scene.
    pub fn new(scene_id: SceneId, tick: Tick) -> Self {
        Self {
            scene_id,
            tick,
            nodes: Vec::new(),
        }
    }
    /// Appends a node.
    pub fn push(&mut self, node: Node3D) {
        self.nodes.push(node);
    }
    /// Nodes in insertion order.
    pub fn nodes(&self) -> &[Node3D] {
        &self.nodes
    }
    /// Scene tick.
    pub fn tick(&self) -> Tick {
        self.tick
    }

    /// Projects through `camera` into a pixel-space [`Scene2D`] (units `Custom`, `y` down).
    ///
    /// Polylines are clipped at the near plane (a clipped polyline may become several 2D
    /// polylines). Within a layer, nodes are painted far to near (stable for ties). Node ids are
    /// sequential from 1 in paint order of the final scene; the 3D pick tag is copied to
    /// `state_code`.
    pub fn project(&self, camera: &OrbitCamera) -> Scene2D {
        let space = CoordinateSpace::new(ScaleUnits::Custom, 1.0, Point2::ZERO, false)
            .expect("unit scale is valid");
        let mut scene = Scene2D::new(self.scene_id, self.tick, EpistemicKind::Observed, space);
        // layer -> [(depth, seq, kind, color, tag, primitive)]
        type Item = (
            f64,
            usize,
            EpistemicKind,
            Option<[u8; 3]>,
            Option<u32>,
            ScenePrimitive,
        );
        let mut per_layer: BTreeMap<LayerId, Vec<Item>> = BTreeMap::new();
        let mut seq = 0usize;
        for n in &self.nodes {
            let mut out: Vec<(f64, ScenePrimitive)> = Vec::new();
            match &n.primitive {
                Primitive3D::Label { position, text } => {
                    if let Some(sp) = camera.project(*position)
                        && let Ok(p) = Point2::new(sp.x, sp.y)
                    {
                        out.push((sp.depth, ScenePrimitive::label(p, text.clone())));
                    }
                }
                Primitive3D::ScreenLabel { x, y, text } => {
                    if let Ok(p) = Point2::new(*x, *y) {
                        out.push((0.0, ScenePrimitive::label(p, text.clone())));
                    }
                }
                Primitive3D::Polyline { points, closed } => {
                    let mut pts = points.clone();
                    if *closed && let Some(f) = points.first() {
                        pts.push(*f);
                    }
                    let mut run: Vec<Point2> = Vec::new();
                    let mut depth_sum = 0.0;
                    let mut depth_n = 0usize;
                    let flush =
                        |run: &mut Vec<Point2>,
                         ds: f64,
                         dn: usize,
                         out: &mut Vec<(f64, ScenePrimitive)>| {
                            if run.len() >= 2
                                && let Ok(p) = ScenePrimitive::polyline(std::mem::take(run), false)
                            {
                                out.push((ds / dn.max(1) as f64, p));
                            }
                            run.clear();
                        };
                    for w in pts.windows(2) {
                        match camera.project_segment(w[0], w[1]) {
                            Some((a, b)) => {
                                let (Ok(pa), Ok(pb)) =
                                    (Point2::new(a.x, a.y), Point2::new(b.x, b.y))
                                else {
                                    flush(&mut run, depth_sum, depth_n, &mut out);
                                    depth_sum = 0.0;
                                    depth_n = 0;
                                    continue;
                                };
                                if run.last() != Some(&pa) {
                                    flush(&mut run, depth_sum, depth_n, &mut out);
                                    depth_sum = 0.0;
                                    depth_n = 0;
                                    run.push(pa);
                                    depth_sum += a.depth;
                                    depth_n += 1;
                                }
                                run.push(pb);
                                depth_sum += b.depth;
                                depth_n += 1;
                            }
                            None => {
                                flush(&mut run, depth_sum, depth_n, &mut out);
                                depth_sum = 0.0;
                                depth_n = 0;
                            }
                        }
                    }
                    flush(&mut run, depth_sum, depth_n, &mut out);
                }
            }
            for (depth, prim) in out {
                per_layer
                    .entry(n.layer)
                    .or_default()
                    .push((depth, seq, n.kind, n.color, n.tag, prim));
                seq += 1;
            }
        }
        let mut next_id = 1u64;
        for (layer, mut items) in per_layer {
            // Far to near; ties keep insertion order.
            items.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let mut l = SceneLayer::new(layer);
            for (_, _, kind, color, tag, prim) in items {
                let selectable = tag.is_some();
                let mut node =
                    SceneNode::new(SceneNodeId::new(next_id), None, kind, prim, selectable);
                if let Some(c) = color {
                    node = node.with_color(c);
                }
                if let Some(t) = tag {
                    node = node.with_state_code(t);
                }
                next_id += 1;
                l.add_node(node).expect("sequential ids are unique");
            }
            scene.add_layer(l).expect("one layer per id");
        }
        scene
    }

    /// Picks the tagged polyline whose projection passes within `tolerance_px` of `(x, y)`;
    /// ties resolved by smaller distance, then smaller tag. Pure function of `(scene, camera)`.
    pub fn pick(&self, camera: &OrbitCamera, x: f64, y: f64, tolerance_px: f64) -> Option<u32> {
        let mut best: Option<(f64, u32)> = None;
        for n in &self.nodes {
            let (Some(tag), Primitive3D::Polyline { points, closed }) = (n.tag, &n.primitive)
            else {
                continue;
            };
            let mut pts = points.clone();
            if *closed && let Some(f) = points.first() {
                pts.push(*f);
            }
            for w in pts.windows(2) {
                if let Some((a, b)) = camera.project_segment(w[0], w[1]) {
                    let d = dist_point_segment((x, y), (a.x, a.y), (b.x, b.y));
                    if d <= tolerance_px && best.is_none_or(|(bd, bt)| (d, tag) < (bd, bt)) {
                        best = Some((d, tag));
                    }
                }
            }
        }
        best.map(|(_, t)| t)
    }
}

fn dist_point_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam() -> OrbitCamera {
        // Eye on +x looking at the origin: right = +y, up = +z.
        OrbitCamera::new(Vec3::ZERO, 0.0, 0.0, 10.0, (800.0, 600.0))
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn known_points_project_to_expected_pixels() {
        let c = cam();
        let f = 300.0 / 30.0_f64.to_radians().tan(); // 519.615...
        let o = c.project(Vec3::ZERO).unwrap();
        assert!(close(o.x, 400.0) && close(o.y, 300.0) && close(o.depth, 10.0));
        let py = c.project(Vec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(close(py.x, 400.0 + f / 10.0) && close(py.y, 300.0)); // +y is screen-right
        let pz = c.project(Vec3::new(0.0, 0.0, 1.0)).unwrap();
        assert!(close(pz.x, 400.0) && close(pz.y, 300.0 - f / 10.0)); // +z is screen-up
        // Perspective: a point 5 m closer is magnified 2x.
        let near = c.project(Vec3::new(5.0, 1.0, 0.0)).unwrap();
        assert!(close(near.x, 400.0 + f / 5.0) && close(near.depth, 5.0));
    }

    #[test]
    fn orthographic_matches_perspective_at_target_depth_and_ignores_depth() {
        let mut c = cam();
        c.set_projection(Projection::Orthographic);
        let f = 300.0 / 30.0_f64.to_radians().tan();
        let a = c.project(Vec3::new(0.0, 1.0, 0.0)).unwrap();
        let b = c.project(Vec3::new(5.0, 1.0, 0.0)).unwrap();
        assert!(close(a.x, 400.0 + f / 10.0));
        assert!(close(a.x, b.x) && close(a.y, b.y));
    }

    #[test]
    fn right_handed_basis_and_top_down_view() {
        let c = cam();
        let (r, u, f) = c.basis();
        assert!(
            close(r.dot(Vec3::Y), 1.0) && close(u.dot(Vec3::Z), 1.0) && close(f.dot(Vec3::X), -1.0)
        );
        // right x up == -forward (right-handed screen frame looking down -forward).
        let ru = r.cross(u);
        assert!(close(ru.dot(f), -1.0));
        // Looking from the front-left (azimuth -90 deg: eye on -y): +x is screen-right.
        let c2 = OrbitCamera::new(
            Vec3::ZERO,
            -std::f64::consts::FRAC_PI_2,
            0.0,
            10.0,
            (800.0, 600.0),
        );
        let p = c2.project(Vec3::X).unwrap();
        assert!(p.x > 400.0 && close(p.y, 300.0));
    }

    #[test]
    fn points_behind_the_camera_and_segments_are_clipped() {
        let c = cam();
        assert!(c.project(Vec3::new(20.0, 0.0, 0.0)).is_none());
        let (a, b) = c
            .project_segment(Vec3::ZERO, Vec3::new(20.0, 0.0, 0.0))
            .unwrap();
        assert!(close(a.depth, 10.0) && close(b.depth, NEAR));
        assert!(
            c.project_segment(Vec3::new(11.0, 0.0, 0.0), Vec3::new(20.0, 0.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn orbit_zoom_pan_are_clamped_and_finite_only() {
        let mut c = cam();
        c.orbit(0.5, 10.0);
        assert!(close(c.azimuth(), 0.5) && close(c.elevation(), MAX_ELEVATION));
        c.orbit(f64::NAN, 1.0);
        assert!(close(c.azimuth(), 0.5));
        c.zoom(1e9);
        assert!(close(c.distance(), MIN_DISTANCE));
        c.zoom(0.0);
        assert!(close(c.distance(), MIN_DISTANCE));
        let t = c.target();
        c.pan_pixels(10.0, -4.0);
        assert!(c.target() != t && c.target().is_finite());
    }

    #[test]
    fn projection_is_deterministic_and_camera_changes_output() {
        let mut s = Scene3D::new(SceneId::new(1), Tick::new(0));
        s.push(Node3D::polyline(
            LayerId::Environment,
            EpistemicKind::Observed,
            vec![Vec3::ZERO, Vec3::X, Vec3::Y],
            true,
        ));
        s.push(Node3D::label(
            LayerId::Labels,
            EpistemicKind::Observed,
            Vec3::Z,
            "z",
        ));
        let c = cam();
        assert_eq!(s.project(&c), s.project(&c.clone()));
        let mut c2 = c;
        c2.orbit(0.3, 0.1);
        assert_ne!(s.project(&c), s.project(&c2));
        assert_eq!(s.project(&c).layers().len(), 2);
    }

    #[test]
    fn colour_override_is_kept_and_serialised_only_when_present() {
        let mut s = Scene3D::new(SceneId::new(1), Tick::new(0));
        s.push(
            Node3D::polyline(
                LayerId::Environment,
                EpistemicKind::Observed,
                vec![Vec3::ZERO, Vec3::Y],
                false,
            )
            .with_color([1, 2, 3])
            .with_tag(7),
        );
        let p = s.project(&cam());
        let n = &p.layers()[0].nodes()[0];
        assert_eq!(n.color(), Some([1, 2, 3]));
        assert_eq!(n.state_code(), Some(7));
        assert!(n.is_selectable());
        assert!(p.to_canonical_json().contains("\"color\":\"#010203\""));
    }

    #[test]
    fn pick_returns_nearest_tagged_polyline() {
        let mut s = Scene3D::new(SceneId::new(1), Tick::new(0));
        s.push(
            Node3D::polyline(
                LayerId::Environment,
                EpistemicKind::Observed,
                vec![Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
                false,
            )
            .with_tag(9),
        );
        let c = cam();
        assert_eq!(s.pick(&c, 400.0, 302.0, 4.0), Some(9));
        assert_eq!(s.pick(&c, 400.0, 400.0, 4.0), None);
    }
}
