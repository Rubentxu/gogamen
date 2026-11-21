//! `lab.robot3d` (GOG-105): 3D view of the robot pose / environment of the navigation simulator.
//!
//! The view is a deterministic wireframe projection ([`Scene3D`] through an [`OrbitCamera`] into
//! a pixel-space `Scene2D`), drawn by the same backends as every other lab (software SVG and the
//! egui window). There is no physics engine and no asset pipeline here: the physical truth comes
//! from the existing `gogamen-lab-mujoco` navigation [`Sim`].
//!
//! # Coordinate system and units
//! - World frame: **right-handed**, `x` forward/east, `y` left/north, `z` up. **Metres** and
//!   **radians**; `yaw` is counter-clockwise about `+z` (yaw 0 faces `+x`, yaw `pi/2` faces `+y`).
//! - Axes drawn at the origin, 1 m long: **x red**, **y green**, **z blue**, each labelled.
//! - Pixels: origin top-left, `y` down (output of the projection only).
//!
//! # Graphic state vs physical state (AC3)
//! - [`ViewState`] (camera, selection, projection mode) is **graphic** state. It has no field of
//!   the simulation.
//! - [`Robot3dPhysics`] wraps the simulation; [`Robot3dSnapshot`] is its integration snapshot.
//!   Neither has a camera or selection. Camera input can therefore never change a physics trace
//!   hash (see the tests).
//!
//! # Epistemic styling
//! - Observed (blue): true pose, environment (the physical truth from the simulator).
//! - Inferred (orange): fused estimate and range rays.
//! - Hypothetical (purple) **or red**: STALE data (scan age above the safety threshold) and the
//!   uncertainty ellipse, always with a text label.
//!
//! The uncertainty is a **heuristic drift interval** (it grows linearly with the scan age), not a
//! filter covariance; it is labelled as such.

use gogamen_core::{EpistemicKind, Tick};
use gogamen_lab_mujoco::ContinuousPhysicsEnginePort;
use gogamen_lab_mujoco::nav::{
    GridPlanner, ScenarioConfig, Shape, Sim, SimSnapshot, TraceRow, trace_hash,
};
use gogamen_projection::{
    LayerId, Node3D, OrbitCamera, Projection, Scene2D, Scene3D, SceneId, Vec3,
};

use crate::control::Observation;
use crate::shell::{LabEntry, LabSession, ViewInput};
use gogamen_sdk::extensions::ExtensionId;
use std::sync::Arc;

/// Registry id of the lab.
pub const ROBOT3D_ID: &str = "lab.robot3d";
/// One control tick (the shell's `Step`) advances this many simulation ticks (`dt` = 0.01 s).
pub const SIM_TICKS_PER_CONTROL_TICK: u64 = 10;
/// Seed of the demo scenario.
pub const ROBOT3D_SEED: u64 = 42;
/// Sim tick at which the demo scanner stops delivering (sensor dropout).
pub const DEMO_DROPOUT_TICK: u64 = 100;
/// Drift-interval growth (m per second of scan age); heuristic, see the module docs.
pub const DRIFT_M_PER_S: f64 = 0.20;
/// Drift-interval floor (m).
pub const DRIFT_FLOOR_M: f64 = 0.05;

/// Tags used for picking.
pub mod tag {
    /// True robot.
    pub const ROBOT: u32 = 1;
    /// Estimated robot.
    pub const ESTIMATE: u32 = 2;
    /// Goal marker.
    pub const GOAL: u32 = 3;
}

const RED: [u8; 3] = [0xd6, 0x27, 0x28];
const GREEN: [u8; 3] = [0x2c, 0xa0, 0x2c];
const BLUE: [u8; 3] = [0x1f, 0x3f, 0xd6];
const GRID_GREY: [u8; 3] = [0xd0, 0xd0, 0xd0];
const WALL_GREY: [u8; 3] = [0x55, 0x55, 0x55];

// ------------------------------------------------------------------------------------------
// Physical side
// ------------------------------------------------------------------------------------------

/// Scan freshness as seen by the safety logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanStatus {
    /// No scan has been delivered yet.
    NoneYet,
    /// Last scan is `age` sim ticks old (`age <= threshold`).
    Fresh {
        /// Age in sim ticks.
        age: u64,
    },
    /// Last scan is older than the threshold.
    Stale {
        /// Age in sim ticks.
        age: u64,
    },
}

/// Read-only physical facts used to build the scene. Plain data: no camera, no selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Robot3dFrame {
    /// Sim tick.
    pub sim_tick: u64,
    /// Seconds per sim tick.
    pub dt: f64,
    /// True pose `[x, y, yaw]` (m, m, rad).
    pub true_pose: [f64; 3],
    /// Fused pose estimate `[x, y, yaw]`.
    pub est_pose: [f64; 3],
    /// Scan freshness.
    pub scan: ScanStatus,
    /// Stale threshold in sim ticks.
    pub stale_ticks: u64,
    /// Range readings (m) and their bearings (rad, relative to the heading).
    pub ranges: Vec<f64>,
    /// Ray bearings.
    pub angles: Vec<f64>,
    /// Maximum range (m).
    pub max_range: f64,
    /// Static obstacles.
    pub statics: Vec<Shape>,
    /// Moving obstacles now: centre and radius.
    pub movers: Vec<([f64; 2], f64)>,
    /// Goal `[x, y]`.
    pub goal: [f64; 2],
    /// Robot radius (m).
    pub robot_radius: f64,
    /// True trail (decimated).
    pub true_trail: Vec<[f64; 2]>,
    /// Estimated trail (decimated).
    pub est_trail: Vec<[f64; 2]>,
}

impl Robot3dFrame {
    /// `true` if the data are stale (or never arrived).
    pub fn is_stale(&self) -> bool {
        matches!(self.scan, ScanStatus::Stale { .. } | ScanStatus::NoneYet)
    }
    /// Scan age in seconds, if any scan was delivered.
    pub fn scan_age_s(&self) -> Option<f64> {
        match self.scan {
            ScanStatus::NoneYet => None,
            ScanStatus::Fresh { age } | ScanStatus::Stale { age } => Some(age as f64 * self.dt),
        }
    }
    /// Heuristic drift radius (m) of the estimate (grows with the scan age).
    pub fn drift_radius_m(&self) -> f64 {
        DRIFT_FLOOR_M + DRIFT_M_PER_S * self.scan_age_s().unwrap_or(0.0)
    }
}

/// Integration snapshot of the simulation (no camera, no selection).
#[derive(Debug, Clone)]
pub struct Robot3dSnapshot {
    inner: SimSnapshot<GridPlanner>,
}

/// Physical simulation wrapper: the corridor scenario with a sensor dropout.
#[derive(Debug, Clone)]
pub struct Robot3dPhysics {
    cfg: ScenarioConfig,
    sim: Sim<GridPlanner>,
}

/// Demo scenario: corridor with a scanner dropout at [`DEMO_DROPOUT_TICK`].
pub fn demo_config() -> ScenarioConfig {
    let mut cfg = ScenarioConfig::corridor(ROBOT3D_SEED);
    cfg.range.dropout_from_tick = Some(DEMO_DROPOUT_TICK);
    cfg
}

impl Robot3dPhysics {
    /// New simulation at tick 0.
    pub fn new(cfg: ScenarioConfig) -> Self {
        let planner = GridPlanner::new(cfg.limits);
        Self {
            sim: Sim::new(cfg.clone(), planner),
            cfg,
        }
    }

    /// Demo scenario.
    pub fn demo() -> Self {
        Self::new(demo_config())
    }

    /// Sim ticks executed.
    pub fn sim_tick(&self) -> u64 {
        self.sim.tick()
    }

    /// Advances to `target` sim ticks (never goes back; the episode may end earlier).
    pub fn advance_to(&mut self, target: u64) {
        while self.sim.tick() < target {
            let before = self.sim.tick();
            if self.sim.step_tick().is_err() || self.sim.tick() == before {
                break;
            }
        }
    }

    /// Hash of the full trace (FNV-1a over the bit patterns): the physics fingerprint.
    pub fn trace_hash(&self) -> u64 {
        trace_hash(self.sim.trace())
    }

    /// Trace rows so far.
    pub fn trace(&self) -> &[TraceRow] {
        self.sim.trace()
    }

    /// Takes an integration snapshot.
    pub fn snapshot(&mut self) -> Option<Robot3dSnapshot> {
        self.sim
            .snapshot()
            .ok()
            .map(|inner| Robot3dSnapshot { inner })
    }

    /// Restores a snapshot.
    pub fn restore(&mut self, s: &Robot3dSnapshot) -> bool {
        self.sim.restore(&s.inner).is_ok()
    }

    /// Physical facts for drawing.
    pub fn frame(&self) -> Robot3dFrame {
        let tick = self.sim.tick();
        let dt = self.sim.engine().dt();
        let stale_ticks = self.cfg.safety.stale_ticks;
        let fusion = self.sim.fusion();
        let scan = match fusion.scan_age(tick) {
            None => ScanStatus::NoneYet,
            Some(age) if age > stale_ticks => ScanStatus::Stale { age },
            Some(age) => ScanStatus::Fresh { age },
        };
        let trace = self.sim.trace();
        let true_pose = trace.last().map_or(self.cfg.start, |r| r.true_pose);
        let t = tick as f64 * dt;
        let decimate = |f: fn(&TraceRow) -> [f64; 3]| -> Vec<[f64; 2]> {
            let mut v = vec![[self.cfg.start[0], self.cfg.start[1]]];
            v.extend(trace.iter().step_by(5).map(|r| {
                let p = f(r);
                [p[0], p[1]]
            }));
            v
        };
        Robot3dFrame {
            sim_tick: tick,
            dt,
            true_pose,
            est_pose: fusion.pose(),
            scan,
            stale_ticks,
            ranges: fusion.ranges().to_vec(),
            angles: fusion.angles().to_vec(),
            max_range: fusion.max_range(),
            statics: self.cfg.world.statics.clone(),
            movers: self
                .cfg
                .world
                .movers
                .iter()
                .map(|m| (m.center_at(t), m.radius))
                .collect(),
            goal: self.cfg.goal,
            robot_radius: self.cfg.robot_radius,
            true_trail: decimate(|r| r.true_pose),
            est_trail: decimate(|r| r.est_pose),
        }
    }
}

// ------------------------------------------------------------------------------------------
// Graphic side
// ------------------------------------------------------------------------------------------

/// What the user selected (graphic state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Selection {
    /// True robot.
    Robot,
    /// Estimated robot.
    Estimate,
    /// Goal marker.
    Goal,
}

impl Selection {
    fn from_tag(t: u32) -> Option<Self> {
        match t {
            tag::ROBOT => Some(Self::Robot),
            tag::ESTIMATE => Some(Self::Estimate),
            tag::GOAL => Some(Self::Goal),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Robot => "true robot (observed)",
            Self::Estimate => "fused estimate (inferred)",
            Self::Goal => "goal",
        }
    }
}

/// Graphic state: camera and selection. Contains nothing of the simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewState {
    /// Orbit camera.
    pub camera: OrbitCamera,
    /// Selected entity.
    pub selection: Option<Selection>,
}

/// Default camera for a viewport (behind-right of the corridor, looking toward `+x`).
pub fn default_camera(w: f64, h: f64) -> OrbitCamera {
    OrbitCamera::new(Vec3::new(2.6, 0.0, 0.0), -2.2, 0.62, 9.0, (w, h))
}

impl ViewState {
    /// Fresh view for a viewport.
    pub fn new(w: f64, h: f64) -> Self {
        Self {
            camera: default_camera(w, h),
            selection: None,
        }
    }
}

fn v3(p: [f64; 2], z: f64) -> Vec3 {
    Vec3::new(p[0], p[1], z)
}

/// Wireframe prism: bottom ring, top ring (closed) and vertical edges (open).
fn prism(corners: &[[f64; 2]], z0: f64, z1: f64) -> Vec<(Vec<Vec3>, bool)> {
    let ring = |z: f64| corners.iter().map(|c| v3(*c, z)).collect::<Vec<_>>();
    let mut out = vec![(ring(z0), true), (ring(z1), true)];
    for c in corners {
        out.push((vec![v3(*c, z0), v3(*c, z1)], false));
    }
    out
}

fn circle_pts(c: [f64; 2], r: f64, n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|i| {
            let a = std::f64::consts::TAU * i as f64 / n as f64;
            [c[0] + r * a.cos(), c[1] + r * a.sin()]
        })
        .collect()
}

fn oriented_box(pose: [f64; 3], len: f64, wid: f64) -> Vec<[f64; 2]> {
    let (s, c) = pose[2].sin_cos();
    [
        (len / 2.0, wid / 2.0),
        (len / 2.0, -wid / 2.0),
        (-len / 2.0, -wid / 2.0),
        (-len / 2.0, wid / 2.0),
    ]
    .iter()
    .map(|(l, w)| [pose[0] + l * c - w * s, pose[1] + l * s + w * c])
    .collect()
}

struct Builder {
    s: Scene3D,
}

impl Builder {
    fn poly(
        &mut self,
        layer: LayerId,
        kind: EpistemicKind,
        pts: Vec<Vec3>,
        closed: bool,
        color: Option<[u8; 3]>,
        tag: Option<u32>,
    ) {
        let mut n = Node3D::polyline(layer, kind, pts, closed);
        if let Some(c) = color {
            n = n.with_color(c);
        }
        if let Some(t) = tag {
            n = n.with_tag(t);
        }
        self.s.push(n);
    }

    fn prism(
        &mut self,
        layer: LayerId,
        kind: EpistemicKind,
        corners: &[[f64; 2]],
        z: (f64, f64),
        color: Option<[u8; 3]>,
        tag: Option<u32>,
    ) {
        for (pts, closed) in prism(corners, z.0, z.1) {
            self.poly(layer, kind, pts, closed, color, tag);
        }
    }

    fn label(&mut self, kind: EpistemicKind, at: Vec3, text: &str, color: Option<[u8; 3]>) {
        let mut n = Node3D::label(LayerId::Labels, kind, at, text);
        if let Some(c) = color {
            n = n.with_color(c);
        }
        self.s.push(n);
    }

    fn hud(&mut self, row: usize, text: String, color: Option<[u8; 3]>) {
        let mut n = Node3D::screen_label(
            LayerId::Labels,
            EpistemicKind::Observed,
            10.0,
            20.0 + 18.0 * row as f64,
            text,
        );
        if let Some(c) = color {
            n = n.with_color(c);
        }
        self.s.push(n);
    }
}

/// Builds the 3D scene for a physical frame and the graphic selection. The camera is **not** an
/// input: the scene is a pure function of the physical frame and the selection.
pub fn build_scene(frame: &Robot3dFrame, selection: Option<Selection>) -> Scene3D {
    use EpistemicKind::{Hypothetical, Inferred, Observed};
    let mut b = Builder {
        s: Scene3D::new(SceneId::new(105), Tick::new(frame.sim_tick)),
    };
    let stale = frame.is_stale();
    let env = LayerId::Environment;

    // Ground grid (1 m) over the corridor.
    for i in -2..=2 {
        let y = f64::from(i);
        b.poly(
            env,
            Observed,
            vec![Vec3::new(-1.0, y, 0.0), Vec3::new(7.0, y, 0.0)],
            false,
            Some(GRID_GREY),
            None,
        );
    }
    for i in -1..=7 {
        let x = f64::from(i);
        b.poly(
            env,
            Observed,
            vec![Vec3::new(x, -2.0, 0.0), Vec3::new(x, 2.0, 0.0)],
            false,
            Some(GRID_GREY),
            None,
        );
    }

    // Static obstacles as wireframe prisms, movers as cylinders (true environment).
    for shape in &frame.statics {
        let corners = match *shape {
            Shape::Aabb { min, max } => {
                vec![min, [max[0], min[1]], max, [min[0], max[1]]]
            }
            Shape::Circle { c, r } => circle_pts(c, r, 16),
        };
        b.prism(env, Observed, &corners, (0.0, 0.5), Some(WALL_GREY), None);
    }
    for (c, r) in &frame.movers {
        b.prism(
            env,
            Observed,
            &circle_pts(*c, *r, 12),
            (0.0, 0.4),
            None,
            None,
        );
    }

    // World axes (x red, y green, z blue), 1 m, labelled.
    let o = Vec3::ZERO;
    for (end, color, name) in [
        (Vec3::X, RED, "x (forward)"),
        (Vec3::Y, GREEN, "y (left)"),
        (Vec3::Z, BLUE, "z (up)"),
    ] {
        b.poly(env, Observed, vec![o, end], false, Some(color), None);
        b.label(Observed, end.scale(1.08), name, Some(color));
    }

    // Goal marker (tagged, pickable).
    let g = frame.goal;
    b.prism(
        env,
        Observed,
        &circle_pts(g, 0.15, 12),
        (0.0, 0.8),
        Some([0x2c, 0xa0, 0x2c]),
        Some(tag::GOAL),
    );
    b.label(Observed, v3(g, 0.9), "goal", None);

    // True trail and true robot (observed truth).
    let trail = |pts: &[[f64; 2]], z: f64| pts.iter().map(|p| v3(*p, z)).collect::<Vec<_>>();
    if frame.true_trail.len() >= 2 {
        b.poly(
            LayerId::Decisions,
            Observed,
            trail(&frame.true_trail, 0.01),
            false,
            None,
            None,
        );
    }
    let tp = frame.true_pose;
    b.prism(
        LayerId::Decisions,
        Observed,
        &oriented_box(tp, 0.40, 0.30),
        (0.0, 0.20),
        None,
        Some(tag::ROBOT),
    );
    let (sy, cy) = tp[2].sin_cos();
    b.poly(
        LayerId::Decisions,
        Observed,
        vec![
            Vec3::new(tp[0], tp[1], 0.10),
            Vec3::new(tp[0] + 0.55 * cy, tp[1] + 0.55 * sy, 0.10),
        ],
        false,
        None,
        Some(tag::ROBOT),
    );
    b.label(Observed, Vec3::new(tp[0], tp[1], 0.32), "true pose", None);

    // Estimate, rays, staleness and uncertainty.
    let ep = frame.est_pose;
    let (kind, color) = if stale {
        (Hypothetical, Some(RED))
    } else {
        (Inferred, None)
    };
    if frame.est_trail.len() >= 2 {
        b.poly(
            LayerId::Sensors,
            Inferred,
            trail(&frame.est_trail, 0.02),
            false,
            None,
            None,
        );
    }
    for (r, a) in frame.ranges.iter().zip(&frame.angles) {
        let ang = ep[2] + a;
        let end = Vec3::new(ep[0] + r * ang.cos(), ep[1] + r * ang.sin(), 0.10);
        b.poly(
            LayerId::Sensors,
            kind,
            vec![Vec3::new(ep[0], ep[1], 0.10), end],
            false,
            color,
            None,
        );
    }
    b.prism(
        LayerId::Sensors,
        kind,
        &oriented_box(ep, 0.30, 0.22),
        (0.0, 0.14),
        color,
        Some(tag::ESTIMATE),
    );
    // Uncertainty: heuristic drift ellipse (a along heading, b = a / 2) on the ground.
    let a = frame.drift_radius_m();
    let (se, ce) = ep[2].sin_cos();
    let ell: Vec<Vec3> = (0..32)
        .map(|i| {
            let t = std::f64::consts::TAU * f64::from(i) / 32.0;
            let (lx, ly) = (a * t.cos(), a * 0.5 * t.sin());
            Vec3::new(ep[0] + lx * ce - ly * se, ep[1] + lx * se + ly * ce, 0.03)
        })
        .collect();
    b.poly(
        LayerId::Imagination,
        Hypothetical,
        ell,
        true,
        if stale { Some(RED) } else { None },
        None,
    );
    b.label(
        Hypothetical,
        Vec3::new(ep[0], ep[1] - 0.45, 0.03),
        &format!("uncertainty (heuristic drift) +-{a:.2} m"),
        if stale { Some(RED) } else { None },
    );
    let (status, status_color) = match frame.scan {
        ScanStatus::NoneYet => ("NO SCAN YET (treated as STALE)".to_owned(), Some(RED)),
        ScanStatus::Fresh { age } => (
            format!(
                "fresh: scan age {:.2} s <= {:.2} s",
                age as f64 * frame.dt,
                frame.stale_ticks as f64 * frame.dt
            ),
            None,
        ),
        ScanStatus::Stale { age } => (
            format!(
                "STALE: scan age {:.2} s > {:.2} s",
                age as f64 * frame.dt,
                frame.stale_ticks as f64 * frame.dt
            ),
            Some(RED),
        ),
    };
    b.label(kind, Vec3::new(ep[0], ep[1], 0.55), &status, status_color);
    if stale {
        // Red cross over the estimate: STALE marker.
        for (dx, dy) in [(0.18, 0.18), (0.18, -0.18)] {
            b.poly(
                LayerId::Decisions,
                Hypothetical,
                vec![
                    Vec3::new(ep[0] - dx, ep[1] - dy, 0.45),
                    Vec3::new(ep[0] + dx, ep[1] + dy, 0.45),
                ],
                false,
                Some(RED),
                None,
            );
        }
    }

    // Selection highlight (graphic state only).
    if let Some(sel) = selection {
        let (c, r) = match sel {
            Selection::Robot => ([tp[0], tp[1]], 0.45),
            Selection::Estimate => ([ep[0], ep[1]], 0.45),
            Selection::Goal => (g, 0.35),
        };
        b.poly(
            LayerId::Decisions,
            Observed,
            circle_pts(c, r, 24).iter().map(|p| v3(*p, 0.04)).collect(),
            true,
            Some([0xff, 0xc1, 0x07]),
            None,
        );
    }

    // HUD (screen space).
    b.hud(
        0,
        format!(
            "lab.robot3d  sim tick {}  t={:.2} s",
            frame.sim_tick,
            frame.sim_tick as f64 * frame.dt
        ),
        None,
    );
    b.hud(
        1,
        "units: metres, radians | right-handed, z up, yaw about +z".to_owned(),
        None,
    );
    b.hud(2, status, status_color);
    b.hud(
        3,
        "blue = true (observed) | orange = estimate/rays (inferred) | purple/red = stale / drift"
            .to_owned(),
        None,
    );
    if let Some(sel) = selection {
        b.hud(4, format!("selected: {}", sel.name()), None);
    }
    b.s
}

// ------------------------------------------------------------------------------------------
// Session
// ------------------------------------------------------------------------------------------

/// Pixels of pick tolerance.
const PICK_TOLERANCE_PX: f64 = 8.0;

/// Shell session: physics + view state + cached projection.
pub struct Robot3dSession {
    physics: Robot3dPhysics,
    view: ViewState,
    frame: Robot3dFrame,
    scene3d: Scene3D,
    scene: Scene2D,
    generation: u64,
    control_tick: u64,
}

impl Robot3dSession {
    /// New session for a viewport.
    pub fn new(w: f64, h: f64) -> Self {
        Self::with_physics(Robot3dPhysics::demo(), w, h)
    }

    /// Session over an explicit physical simulation.
    pub fn with_physics(physics: Robot3dPhysics, w: f64, h: f64) -> Self {
        let view = ViewState::new(w, h);
        let frame = physics.frame();
        let scene3d = build_scene(&frame, view.selection);
        let scene = scene3d.project(&view.camera);
        Self {
            physics,
            view,
            frame,
            scene3d,
            scene,
            generation: 0,
            control_tick: 0,
        }
    }

    /// Graphic state.
    pub fn view(&self) -> &ViewState {
        &self.view
    }
    /// Physical simulation.
    pub fn physics(&self) -> &Robot3dPhysics {
        &self.physics
    }
    /// Current physical frame.
    pub fn frame(&self) -> &Robot3dFrame {
        &self.frame
    }
    /// 3D scene (before projection).
    pub fn scene3d(&self) -> &Scene3D {
        &self.scene3d
    }
    /// Control ticks applied.
    pub fn control_tick(&self) -> u64 {
        self.control_tick
    }

    /// Aligns the physics with `control_tick` (rebuilds the simulation if time went backwards).
    pub fn set_control_tick(&mut self, control_tick: u64) {
        if control_tick < self.control_tick {
            self.physics = Robot3dPhysics::demo();
        }
        self.control_tick = control_tick;
        self.physics
            .advance_to(control_tick.saturating_mul(SIM_TICKS_PER_CONTROL_TICK));
        self.frame = self.physics.frame();
        self.rebuild();
    }

    fn rebuild(&mut self) {
        self.scene3d = build_scene(&self.frame, self.view.selection);
        self.scene = self.scene3d.project(&self.view.camera);
    }

    /// Applies a graphic input. Never touches the physics.
    pub fn apply_view_input(&mut self, input: &ViewInput) {
        let (w, h) = self.view.camera.viewport();
        match *input {
            ViewInput::Orbit { dx_px, dy_px } => {
                self.view.camera.orbit(-dx_px * 0.01, dy_px * 0.01);
            }
            ViewInput::Pan { dx_px, dy_px } => self.view.camera.pan_pixels(dx_px, dy_px),
            ViewInput::Zoom { factor } => self.view.camera.zoom(factor),
            ViewInput::Click { x, y } => {
                self.view.selection = self
                    .scene3d
                    .pick(&self.view.camera, x, y, PICK_TOLERANCE_PX)
                    .and_then(Selection::from_tag);
            }
            ViewInput::Reset => {
                self.view = ViewState::new(w, h);
            }
            ViewInput::ToggleProjection => {
                let p = match self.view.camera.projection() {
                    Projection::Perspective => Projection::Orthographic,
                    Projection::Orthographic => Projection::Perspective,
                };
                self.view.camera.set_projection(p);
            }
        }
        self.rebuild();
    }
}

impl LabSession for Robot3dSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, o: &Observation) {
        if o.session.generation != self.generation {
            self.generation = o.session.generation;
            self.physics = Robot3dPhysics::demo();
            self.control_tick = 0;
        }
        self.set_control_tick(o.tick.as_u64());
    }

    fn screen_space(&self) -> bool {
        true
    }

    fn resize(&mut self, w: f64, h: f64) {
        self.view.camera.set_viewport(w, h);
        self.rebuild();
    }

    fn view_input(&mut self, input: &ViewInput) {
        self.apply_view_input(input);
    }
}

/// Registry entry of the lab.
pub fn robot3d_lab_entry() -> LabEntry {
    LabEntry::new(
        ExtensionId::new(ROBOT3D_ID),
        "Robot 3D",
        Arc::new(|| Box::new(Robot3dSession::new(900.0, 600.0))),
    )
}

/// Number of scripted 3D captures driven by [`drive_shot`] (window feature).
pub const SHOT_COUNT: usize = 4;

/// Drives the shell into the state of scripted capture `idx` (0-based):
/// 0 initial (fresh scan), 1 orbited + zoomed, 2 panned + orthographic, 3 stale/uncertainty
/// (perspective again, stepped past the dropout). Camera moves go through `view_input` only;
/// time advances only through the control port.
pub fn drive_shot(
    shell: &mut crate::shell::StudioShell,
    idx: usize,
    next_command: &mut u64,
) -> Result<(), String> {
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    if idx == 0 {
        shell.select_lab(ROBOT3D_ID).map_err(|e| e.to_string())?;
    }
    let target_tick: Option<u64> = match idx {
        0 => Some(6),
        3 => Some(30),
        _ => None,
    };
    match idx {
        1 => {
            shell.view_input(ViewInput::Orbit {
                dx_px: 180.0,
                dy_px: -30.0,
            });
            shell.view_input(ViewInput::Zoom { factor: 1.25 });
        }
        2 => {
            shell.view_input(ViewInput::Pan {
                dx_px: 60.0,
                dy_px: 20.0,
            });
            shell.view_input(ViewInput::ToggleProjection);
        }
        3 => {
            shell.view_input(ViewInput::ToggleProjection);
            shell.view_input(ViewInput::Reset);
        }
        _ => {}
    }
    if let Some(t) = target_tick {
        let control = shell
            .control_port()
            .ok_or("no active session")?
            .map_err(|e| format!("{e:?}"))?;
        let observe = shell.observation_port().ok_or("no active session")?;
        while observe.observe().tick.as_u64() < t {
            *next_command += 1;
            let ack = control.submit(ControlCommand {
                id: CommandId(*next_command),
                session: observe.observe().session,
                action: ControlAction::Step,
            });
            if !matches!(ack.outcome, AckOutcome::Applied) {
                return Err(format!("step rejected: {ack:?}"));
            }
            shell.sync_active();
        }
    }
    shell.sync_active();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_projection::ScenePrimitive;

    fn labels(s: &Scene2D) -> Vec<String> {
        s.layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter_map(|n| match n.primitive() {
                ScenePrimitive::Label { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    fn session_at(control_ticks: u64) -> Robot3dSession {
        let mut s = Robot3dSession::new(900.0, 600.0);
        s.set_control_tick(control_ticks);
        s
    }

    // ---- AC1: coordinates, axes, yaw -------------------------------------------------------

    #[test]
    fn axes_are_coloured_and_labelled_with_documented_directions() {
        let s = session_at(3);
        let two = s.scene.clone();
        let ls = labels(&two);
        for needle in ["x (forward)", "y (left)", "z (up)"] {
            assert!(
                ls.iter().any(|l| l == needle),
                "missing axis label {needle}"
            );
        }
        let colours: Vec<[u8; 3]> = two
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter_map(|n| n.color())
            .collect();
        for c in [RED, GREEN, BLUE] {
            assert!(colours.contains(&c), "missing axis colour {c:?}");
        }
        assert!(
            ls.iter()
                .any(|l| l.contains("right-handed, z up, yaw about +z"))
        );
    }

    #[test]
    fn default_camera_projects_axes_to_expected_screen_directions() {
        let cam = default_camera(900.0, 600.0);
        let o = cam.project(Vec3::ZERO).unwrap();
        let x = cam.project(Vec3::X).unwrap();
        let y = cam.project(Vec3::Y).unwrap();
        let z = cam.project(Vec3::Z).unwrap();
        // Looking toward +x from behind-right: +x goes away (up/right on screen), +y goes left,
        // +z goes up the screen (pixel y decreases).
        assert!(z.y < o.y, "z must go up");
        assert!(y.x < o.x, "+y (left of the robot) is on the screen-left");
        assert!(x.x > o.x || x.y < o.y, "+x recedes into the scene");
        // Handedness: (x cross y) == z in world space.
        assert_eq!(Vec3::X.cross(Vec3::Y), Vec3::Z);
    }

    #[test]
    fn yaw_is_counter_clockwise_about_z_and_heading_follows_it() {
        // Robot box heading: yaw 0 -> +x, yaw pi/2 -> +y.
        let b0 = oriented_box([0.0, 0.0, 0.0], 2.0, 1.0);
        assert!((b0[0][0] - 1.0).abs() < 1e-12 && (b0[0][1] - 0.5).abs() < 1e-12);
        let b90 = oriented_box([0.0, 0.0, std::f64::consts::FRAC_PI_2], 2.0, 1.0);
        // front-left corner (l=+1, w=+0.5) maps to (-0.5, +1).
        assert!((b90[0][0] + 0.5).abs() < 1e-12 && (b90[0][1] - 1.0).abs() < 1e-12);
        // On screen with the top-down-ish default camera, a robot facing +y points to screen-left
        // of one facing +x (the +y axis projects to the left of +x).
        let cam = default_camera(900.0, 600.0);
        let o = cam.project(Vec3::ZERO).unwrap();
        let fx = cam.project(Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let fy = cam.project(Vec3::new(0.0, 1.0, 0.0)).unwrap();
        assert!(fy.x < fx.x, "heading +y is to the left of heading +x");
        assert!(o.depth > 0.0);
    }

    #[test]
    fn true_pose_follows_the_simulator_pose() {
        let s = session_at(8);
        let last = s.physics.trace().last().unwrap().true_pose;
        assert_eq!(s.frame.true_pose, last);
        assert!(s.frame.true_pose[0] > 0.0, "robot moved along +x");
    }

    // ---- AC2: camera never changes physics; stale + uncertainty visible ---------------------

    #[test]
    fn camera_inputs_do_not_change_the_physics_trace_hash() {
        let mut plain = session_at(20);
        let reference = plain.physics.trace_hash();
        let mut moved = Robot3dSession::new(900.0, 600.0);
        for t in 1..=20 {
            moved.apply_view_input(&ViewInput::Orbit {
                dx_px: 37.0,
                dy_px: -12.0,
            });
            moved.apply_view_input(&ViewInput::Zoom { factor: 1.3 });
            moved.apply_view_input(&ViewInput::Pan {
                dx_px: 5.0,
                dy_px: 9.0,
            });
            moved.apply_view_input(&ViewInput::Click { x: 450.0, y: 300.0 });
            if t % 7 == 0 {
                moved.apply_view_input(&ViewInput::ToggleProjection);
            }
            moved.set_control_tick(t);
        }
        moved.apply_view_input(&ViewInput::Reset);
        assert_eq!(moved.physics.trace_hash(), reference);
        assert_eq!(moved.frame, plain.frame);
        assert_eq!(moved.physics.trace(), plain.physics.trace());
        // And the hash is not trivially constant.
        plain.set_control_tick(21);
        assert_ne!(plain.physics.trace_hash(), reference);
    }

    #[test]
    fn camera_change_changes_projection_but_not_the_3d_scene() {
        let mut s = session_at(5);
        let scene3d = s.scene3d().clone();
        let before = s.scene.clone();
        s.apply_view_input(&ViewInput::Orbit {
            dx_px: 80.0,
            dy_px: 20.0,
        });
        assert_eq!(s.scene3d(), &scene3d, "3D scene is camera independent");
        assert_ne!(s.scene, before, "projection follows the camera");
    }

    #[test]
    fn fresh_data_has_no_stale_marker_and_stale_data_is_distinct_and_labelled() {
        let fresh = session_at(6);
        assert!(!fresh.frame.is_stale(), "{:?}", fresh.frame.scan);
        assert!(!labels(&fresh.scene).iter().any(|l| l.contains("STALE")));
        assert!(
            labels(&fresh.scene)
                .iter()
                .any(|l| l.starts_with("fresh: scan age"))
        );

        let stale = session_at(30);
        assert!(stale.frame.is_stale(), "{:?}", stale.frame.scan);
        let ls = labels(&stale.scene);
        assert!(
            ls.iter()
                .any(|l| l.starts_with("STALE: scan age") && l.contains("> 0.40 s")),
            "{ls:?}"
        );
        // Stale nodes use the Hypothetical kind and the red override (distinct from fresh).
        let stale_nodes = stale
            .scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter(|n| n.color() == Some(RED) && n.epistemic_kind() == EpistemicKind::Hypothetical)
            .count();
        assert!(
            stale_nodes > 21,
            "rays + estimate + ellipse + cross, got {stale_nodes}"
        );
        let fresh_red = fresh
            .scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter(|n| n.color() == Some(RED) && n.epistemic_kind() == EpistemicKind::Hypothetical)
            .count();
        // Only the x axis (red, Observed) shares the colour; no Hypothetical red when fresh.
        assert_eq!(fresh_red, 0);
    }

    #[test]
    fn uncertainty_ellipse_is_projected_and_grows_with_scan_age() {
        let fresh = session_at(6);
        let stale = session_at(30);
        assert!(stale.frame.drift_radius_m() > fresh.frame.drift_radius_m());
        for s in [&fresh, &stale] {
            let ell = s
                .scene
                .layers()
                .iter()
                .find(|l| l.id() == LayerId::Imagination)
                .expect("imagination layer carries the uncertainty ellipse");
            assert!(ell.nodes().iter().any(|n| matches!(
                n.primitive(),
                // A closed 3D loop is projected as an open polyline repeating the first point.
                ScenePrimitive::Polyline { points, .. } if points.len() >= 32
            ) && n.epistemic_kind()
                == EpistemicKind::Hypothetical));
            assert!(
                labels(&s.scene)
                    .iter()
                    .any(|l| l.contains("uncertainty (heuristic drift)"))
            );
        }
        // Wider on screen after the dropout (same default camera).
        let width = |s: &Robot3dSession| {
            let l = s
                .scene
                .layers()
                .iter()
                .find(|l| l.id() == LayerId::Imagination)
                .unwrap();
            let pts = l
                .nodes()
                .iter()
                .find_map(|n| match n.primitive() {
                    ScenePrimitive::Polyline { points, .. } if points.len() >= 32 => {
                        Some(points.clone())
                    }
                    _ => None,
                })
                .unwrap();
            let (mn, mx) = pts.iter().fold((f64::MAX, f64::MIN), |(a, b), p| {
                (a.min(p.x()), b.max(p.x()))
            });
            mx - mn
        };
        assert!(width(&stale) > width(&fresh));
    }

    // ---- AC3: graphic vs physical state ------------------------------------------------------

    fn struct_body<'a>(src: &'a str, name: &str) -> &'a str {
        let start = src
            .find(&format!("pub struct {name}"))
            .expect("struct present");
        let rest = &src[start..];
        let end = rest.find("\n}\n").expect("struct end");
        &rest[..end]
    }

    #[test]
    fn snapshot_and_physics_have_no_graphic_fields_and_view_has_no_physics_fields() {
        let src = include_str!("robot3d.rs");
        for physical in ["Robot3dSnapshot", "Robot3dPhysics", "Robot3dFrame"] {
            let body = struct_body(src, physical);
            for graphic in [
                "OrbitCamera",
                "ViewState",
                "Selection",
                "camera",
                "Projection",
            ] {
                assert!(!body.contains(graphic), "{physical} contains `{graphic}`");
            }
        }
        let view = struct_body(src, "ViewState");
        for physical in ["Sim<", "Snapshot", "Robot3dPhysics", "trace", "engine"] {
            assert!(!view.contains(physical), "ViewState contains `{physical}`");
        }
        // The scene builder does not take a camera at all.
        let sig = src.find("pub fn build_scene(").unwrap();
        let sig_end = src[sig..].find('{').unwrap();
        assert!(!src[sig..sig + sig_end].contains("OrbitCamera"));
    }

    #[test]
    fn restoring_a_physics_snapshot_leaves_the_view_untouched_and_replays_identically() {
        let mut s = session_at(10);
        let snap = s.physics.snapshot().unwrap();
        s.apply_view_input(&ViewInput::Orbit {
            dx_px: 100.0,
            dy_px: 40.0,
        });
        s.apply_view_input(&ViewInput::Click { x: 450.0, y: 300.0 });
        let view_before = s.view.clone();
        let hash10 = s.physics.trace_hash();
        s.set_control_tick(25);
        assert_ne!(s.physics.trace_hash(), hash10);
        assert!(s.physics.restore(&snap));
        assert_eq!(
            s.view, view_before,
            "restoring physics must not touch the view"
        );
        assert_eq!(s.physics.trace_hash(), hash10);
        s.physics.advance_to(250);
        let replay = s.physics.trace_hash();
        let direct = session_at(25).physics.trace_hash();
        assert_eq!(
            replay, direct,
            "replay from the snapshot equals the direct run"
        );
    }

    #[test]
    fn selection_is_graphic_state_picked_through_the_projection() {
        let mut s = session_at(3);
        let cam = s.view.camera;
        let tp = s.frame.true_pose;
        let p = cam.project(Vec3::new(tp[0], tp[1], 0.10)).unwrap();
        s.apply_view_input(&ViewInput::Click { x: p.x, y: p.y });
        assert_eq!(s.view.selection, Some(Selection::Robot));
        assert!(
            labels(&s.scene)
                .iter()
                .any(|l| l.starts_with("selected: true robot"))
        );
        s.apply_view_input(&ViewInput::Click { x: 3.0, y: 590.0 });
        assert_eq!(s.view.selection, None);
    }

    #[test]
    fn session_is_deterministic_and_registers_as_a_screen_space_lab() {
        let a = session_at(12);
        let b = session_at(12);
        assert_eq!(a.scene, b.scene);
        assert!(a.screen_space());
        assert_eq!(robot3d_lab_entry().id().as_str(), ROBOT3D_ID);
    }

    #[test]
    fn time_going_backwards_rebuilds_the_simulation() {
        let mut s = session_at(12);
        let h12 = s.physics.trace_hash();
        s.set_control_tick(4);
        assert_eq!(s.physics.sim_tick(), 40);
        s.set_control_tick(12);
        assert_eq!(s.physics.trace_hash(), h12);
    }
}
