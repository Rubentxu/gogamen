//! Real-window viewport backend (feature `window`, eframe/egui 0.37 with the glow renderer).
//!
//! The window reuses the existing [`StudioShell`] (lab selector, single active session) and
//! [`Camera2D`] (pan / zoom-at-cursor / fit-to-window). Nothing here is compiled without the
//! `window` feature, so the default build stays `std`-only.
//!
//! Two entry points:
//! - [`run_interactive`]: a normal window (drag = pan, wheel = zoom, resize = re-fit).
//! - [`run_script`]: a deterministic scripted session that applies zoom/pan/resize/lab-switch
//!   and captures the rendered framebuffer after each step (see [`ScriptReport`]).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use eframe::egui;
use eframe::glow::HasContext;
use gogamen_core::EpistemicKind;
use gogamen_projection::{Scene2D, ScenePrimitive};

use crate::camera::Camera2D;
use crate::life_lab::LifeProbe;
use crate::shell::StudioShell;
use crate::status::{RenderTestReport, RenderTestStatus};
use crate::tour::{StepFacts, TOUR_PLAN, TourStep, drive_step, tour_shell};
use crate::viewport::{BackendKind, ViewportBackend, escape, num};

/// One primitive submitted to the window painter, in screen points relative to the viewport.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawCmd {
    Circle {
        cx: f64,
        cy: f64,
        r: f64,
        kind: EpistemicKind,
        /// Per-node colour override (`None` = epistemic palette).
        color: Option<[u8; 3]>,
    },
    Lines {
        points: Vec<(f64, f64)>,
        closed: bool,
        kind: EpistemicKind,
        color: Option<[u8; 3]>,
    },
    Rect {
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        kind: EpistemicKind,
        color: Option<[u8; 3]>,
    },
    Image {
        x0: f64,
        y0: f64,
        x1: f64,
        y1: f64,
        texture_w: u32,
        texture_h: u32,
    },
    Label {
        x: f64,
        y: f64,
        text: String,
        kind: EpistemicKind,
        color: Option<[u8; 3]>,
    },
}

/// Pure projection of a scene through a camera into window draw commands (z-ordered).
pub fn draw_list(scene: &Scene2D, camera: &Camera2D) -> Vec<DrawCmd> {
    let mut out = Vec::new();
    let mut layers: Vec<_> = scene.layers().iter().filter(|l| l.is_visible()).collect();
    layers.sort_by_key(|l| (l.z_order(), l.id()));
    for layer in layers {
        for node in layer.nodes() {
            let kind = node.epistemic_kind();
            let color = node.color();
            match node.primitive() {
                ScenePrimitive::Circle { center, radius } => {
                    let (cx, cy) = camera.world_to_screen(center.x(), center.y());
                    out.push(DrawCmd::Circle {
                        cx,
                        cy,
                        r: radius * camera.zoom(),
                        kind,
                        color,
                    });
                }
                ScenePrimitive::Polyline { points, closed } => out.push(DrawCmd::Lines {
                    points: points
                        .iter()
                        .map(|p| camera.world_to_screen(p.x(), p.y()))
                        .collect(),
                    closed: *closed,
                    kind,
                    color,
                }),
                ScenePrimitive::Rect { min, width, height } => {
                    let (ax, ay) = camera.world_to_screen(min.x(), min.y());
                    let (bx, by) = camera.world_to_screen(min.x() + width, min.y() + height);
                    out.push(DrawCmd::Rect {
                        x0: ax.min(bx),
                        y0: ay.min(by),
                        x1: ax.max(bx),
                        y1: ay.max(by),
                        kind,
                        color,
                    });
                }
                ScenePrimitive::Image {
                    min,
                    width,
                    height,
                    texture,
                } => {
                    let (ax, ay) = camera.world_to_screen(min.x(), min.y());
                    let (bx, by) = camera.world_to_screen(min.x() + width, min.y() + height);
                    out.push(DrawCmd::Image {
                        x0: ax.min(bx),
                        y0: ay.min(by),
                        x1: ax.max(bx),
                        y1: ay.max(by),
                        texture_w: texture.width(),
                        texture_h: texture.height(),
                    });
                }
                ScenePrimitive::Label { position, text } => {
                    let (x, y) = camera.world_to_screen(position.x(), position.y());
                    out.push(DrawCmd::Label {
                        x,
                        y,
                        text: text.clone(),
                        kind,
                        color,
                    });
                }
            }
        }
    }
    out
}

fn stroke_color(kind: EpistemicKind, over: Option<[u8; 3]>) -> egui::Color32 {
    if let Some([r, g, b]) = over {
        return egui::Color32::from_rgb(r, g, b);
    }
    match kind {
        EpistemicKind::Observed => egui::Color32::from_rgb(0x1f, 0x77, 0xb4),
        EpistemicKind::Inferred => egui::Color32::from_rgb(0xff, 0x7f, 0x0e),
        EpistemicKind::Hypothetical => egui::Color32::from_rgb(0x94, 0x67, 0xbd),
    }
}

/// Palette colors used to decide whether the scene is visible in a capture.
const SCENE_COLORS: [[u8; 3]; 4] = [
    [0x1f, 0x77, 0xb4],
    [0xff, 0x7f, 0x0e],
    [0x94, 0x67, 0xbd],
    // GOG-105 STALE colour override (red).
    [0xd6, 0x27, 0x28],
];

fn paint_scene(painter: &egui::Painter, origin: egui::Pos2, cmds: &[DrawCmd], scene: &Scene2D) {
    let pt = |x: f64, y: f64| egui::pos2(origin.x + x as f32, origin.y + y as f32);
    let mut textures = scene
        .layers()
        .iter()
        .flat_map(|l| l.nodes())
        .filter_map(|n| match n.primitive() {
            ScenePrimitive::Image { texture, .. } => Some(texture),
            _ => None,
        });
    for cmd in cmds {
        match cmd {
            DrawCmd::Circle {
                cx,
                cy,
                r,
                kind,
                color,
            } => {
                painter.circle_stroke(
                    pt(*cx, *cy),
                    *r as f32,
                    egui::Stroke::new(2.0, stroke_color(*kind, *color)),
                );
            }
            DrawCmd::Lines {
                points,
                closed,
                kind,
                color,
            } => {
                let mut pts: Vec<_> = points.iter().map(|p| pt(p.0, p.1)).collect();
                if *closed && let Some(first) = pts.first().copied() {
                    pts.push(first);
                }
                painter.add(egui::Shape::line(
                    pts,
                    egui::Stroke::new(2.0, stroke_color(*kind, *color)),
                ));
            }
            DrawCmd::Rect {
                x0,
                y0,
                x1,
                y1,
                kind,
                color,
            } => {
                painter.rect_stroke(
                    egui::Rect::from_two_pos(pt(*x0, *y0), pt(*x1, *y1)),
                    0.0,
                    egui::Stroke::new(2.0, stroke_color(*kind, *color)),
                    egui::StrokeKind::Middle,
                );
            }
            DrawCmd::Image { x0, y0, x1, y1, .. } => {
                // Draw the texture as horizontal runs of equal colour (no GPU texture upload).
                let Some(tex) = textures.next() else { continue };
                let (pw, ph) = (
                    (x1 - x0) / f64::from(tex.width()),
                    (y1 - y0) / f64::from(tex.height()),
                );
                for y in 0..tex.height() {
                    for x in 0..tex.width() {
                        let c = tex.rgba_at(x, y).unwrap_or([0; 4]);
                        if c[3] == 0 {
                            continue;
                        }
                        let r = egui::Rect::from_min_max(
                            pt(x0 + f64::from(x) * pw, y0 + f64::from(y) * ph),
                            pt(x0 + f64::from(x + 1) * pw, y0 + f64::from(y + 1) * ph),
                        );
                        painter.rect_filled(r, 0.0, egui::Color32::from_rgb(c[0], c[1], c[2]));
                    }
                }
            }
            DrawCmd::Label {
                x,
                y,
                text,
                kind,
                color,
            } => {
                painter.text(
                    pt(*x, *y),
                    egui::Align2::LEFT_BOTTOM,
                    text,
                    egui::FontId::proportional(16.0),
                    stroke_color(*kind, *color),
                );
            }
        }
    }
}

/// Window backend marker. `render` serialises the draw list that the window painter receives
/// (deterministic text); the pixels themselves only exist inside a running window, so a PASS
/// must come from [`run_script`] / [`run_interactive`] evidence, never from `render` alone.
#[derive(Debug, Clone, Copy, Default)]
pub struct WindowBackend;

impl ViewportBackend for WindowBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Window
    }
    fn name(&self) -> &'static str {
        "window-eframe-glow"
    }
    fn render(&self, scene: &Scene2D, camera: &Camera2D) -> Vec<u8> {
        let mut s = String::new();
        for c in draw_list(scene, camera) {
            let line = match c {
                DrawCmd::Circle { cx, cy, r, .. } => {
                    format!("circle {} {} {}", num(cx), num(cy), num(r))
                }
                DrawCmd::Lines { points, closed, .. } => {
                    let p: Vec<_> = points
                        .iter()
                        .map(|p| format!("{},{}", num(p.0), num(p.1)))
                        .collect();
                    format!("lines closed={closed} {}", p.join(" "))
                }
                DrawCmd::Rect { x0, y0, x1, y1, .. } => {
                    format!("rect {} {} {} {}", num(x0), num(y0), num(x1), num(y1))
                }
                DrawCmd::Image { x0, y0, x1, y1, .. } => {
                    format!("image {} {} {} {}", num(x0), num(y0), num(x1), num(y1))
                }
                DrawCmd::Label { x, y, text, .. } => {
                    format!("label {} {} {}", num(x), num(y), escape(&text))
                }
            };
            s.push_str(&line);
            s.push('\n');
        }
        s.into_bytes()
    }
}

/// Facts about one captured framebuffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShotReport {
    pub name: String,
    pub size: (usize, usize),
    /// FNV-1a 64 over the RGBA pixels.
    pub hash: u64,
    /// Pixels within a small distance of the epistemic palette colours (scene strokes).
    pub scene_pixels: usize,
    pub distinct_colors: usize,
    pub path: Option<PathBuf>,
    /// `true` if the step that produced this shot was driven by the script, not by user input.
    pub programmatic: bool,
    pub lab: String,
    pub camera_zoom: u64,
}

/// Result of a window session.
#[derive(Debug, Clone, Default)]
pub struct ScriptReport {
    pub gl_renderer: String,
    pub gl_version: String,
    pub shots: Vec<ShotReport>,
    /// Headless-computed facts of each tour step (empty outside [`run_tour`]).
    pub tour: Vec<StepFacts>,
}

impl ScriptReport {
    /// All shots differ pairwise by hash and show the scene.
    pub fn all_distinct_and_visible(&self) -> bool {
        let n = self.shots.len();
        n > 0
            && self.shots.iter().all(|s| s.scene_pixels > 0)
            && (0..n).all(|i| (i + 1..n).all(|j| self.shots[i].hash != self.shots[j].hash))
    }

    /// Maps the evidence to a [`RenderTestReport`]: `Pass` only with a window backend and
    /// visible, pairwise-distinct captures.
    pub fn to_render_test_report(&self) -> RenderTestReport {
        RenderTestReport {
            backend: Some(BackendKind::Window),
            status: if self.all_distinct_and_visible() {
                RenderTestStatus::Pass
            } else {
                RenderTestStatus::Fail
            },
        }
    }
}

fn analyze(name: &str, img: &egui::ColorImage) -> ShotReport {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut scene_pixels = 0;
    let mut colors = std::collections::HashSet::new();
    for p in &img.pixels {
        let c = p.to_array();
        for b in c {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        colors.insert(c);
        let near = |t: [u8; 3]| {
            (0..3)
                .map(|i| i32::from(c[i]) - i32::from(t[i]))
                .map(i32::abs)
                .sum::<i32>()
                < 40
        };
        if SCENE_COLORS.iter().any(|t| near(*t)) {
            scene_pixels += 1;
        }
    }
    ShotReport {
        name: name.to_owned(),
        size: (img.size[0], img.size[1]),
        hash,
        scene_pixels,
        distinct_colors: colors.len(),
        path: None,
        programmatic: true,
        lab: String::new(),
        camera_zoom: 0,
    }
}

/// Minimal PNG writer (stored deflate blocks, no compression, no extra dependency).
fn write_png(path: &Path, img: &egui::ColorImage) -> std::io::Result<()> {
    fn crc(data: &[u8]) -> u32 {
        let mut c = 0xffff_ffffu32;
        for &b in data {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    0xedb8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        !c
    }
    fn chunk(out: &mut Vec<u8>, ty: &[u8; 4], data: &[u8]) {
        out.extend((data.len() as u32).to_be_bytes());
        let mut body = ty.to_vec();
        body.extend(data);
        out.extend(&body);
        out.extend(crc(&body).to_be_bytes());
    }
    let (w, h) = (img.size[0], img.size[1]);
    let mut raw = Vec::with_capacity((w * 4 + 1) * h);
    for y in 0..h {
        raw.push(0);
        for x in 0..w {
            let c = img.pixels[y * w + x].to_array();
            raw.extend([c[0], c[1], c[2], 255]);
        }
    }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(u8::from(chunks.peek().is_none()));
        z.extend((c.len() as u16).to_le_bytes());
        z.extend((!(c.len() as u16)).to_le_bytes());
        z.extend(c);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &v in &raw {
        a = (a + u32::from(v)) % 65521;
        b = (b + a) % 65521;
    }
    z.extend(((b << 16) | a).to_be_bytes());
    let mut ihdr = Vec::new();
    ihdr.extend((w as u32).to_be_bytes());
    ihdr.extend((h as u32).to_be_bytes());
    ihdr.extend([8, 6, 0, 0, 0]);
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    std::fs::write(path, out)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    WarmUp,
    ShotInitial,
    SettleZoomPan,
    ShotZoomPan,
    Resize,
    ShotResized,
    SwitchLab,
    ShotLab,
    Done,
}

struct Script {
    out_dir: PathBuf,
    step: Step,
    frames_in_step: u32,
    pending: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    waiting: bool,
    size_before_resize: egui::Vec2,
    shots: Vec<ShotReport>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TourPhase {
    WarmUp,
    Drive,
    Settle,
    Shot,
    Done,
}

/// State of the multi-lab tour (GOG-116): select each lab, advance it deterministically, capture.
struct TourRun {
    out_dir: PathBuf,
    idx: usize,
    phase: TourPhase,
    frames: u32,
    waiting: bool,
    pending: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    shots: Vec<ShotReport>,
    facts: Vec<StepFacts>,
    life: LifeProbe,
    next_cmd: u64,
}

/// Scripted 3D robot captures (GOG-105): initial, orbited, pan + orthographic, stale/uncertainty.
struct Robot3dRun {
    out_dir: PathBuf,
    idx: usize,
    phase: TourPhase,
    frames: u32,
    waiting: bool,
    pending: Arc<Mutex<Option<Arc<egui::ColorImage>>>>,
    shots: Vec<ShotReport>,
    next_cmd: u64,
}

/// Names of the scripted 3D captures, in order.
pub const ROBOT3D_SHOTS: [&str; 4] = [
    "1_initial",
    "2_orbited_zoomed",
    "3_pan_orthographic",
    "4_stale_uncertainty",
];

struct StudioApp {
    shell: StudioShell,
    last_rect_size: Option<(f64, f64)>,
    script: Option<Script>,
    tour: Option<TourRun>,
    robot3d: Option<Robot3dRun>,
    next_cmd: u64,
    report: Arc<Mutex<ScriptReport>>,
    gl_info_done: bool,
}

impl StudioApp {
    fn viewport_ui(&mut self, ui: &mut egui::Ui) {
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let size = (f64::from(rect.width()), f64::from(rect.height()));
        // Re-fit whenever the viewport size changes (and on first frame).
        if self.last_rect_size != Some(size) {
            self.last_rect_size = Some(size);
            self.fit(size);
        }
        if self.shell.active_is_screen_space() {
            self.view_input_ui(ui, &resp, rect);
        } else if resp.double_clicked() {
            self.fit(size);
        }
        if self.shell.active_is_screen_space() {
            // Screen-space (3D) session: input was forwarded above; no 2D pan/zoom.
        } else if resp.dragged() {
            let d = resp.drag_delta();
            self.shell.camera_mut().pan(f64::from(d.x), f64::from(d.y));
        }
        if resp.hovered() && !self.shell.active_is_screen_space() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0
                && let Some(p) = ui.input(|i| i.pointer.hover_pos())
            {
                let f = (f64::from(scroll) * 0.005).exp();
                self.shell.camera_mut().zoom_at(
                    f,
                    f64::from(p.x - rect.min.x),
                    f64::from(p.y - rect.min.y),
                );
            }
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
        if let Some(scene) = self.shell.scene() {
            let cmds = draw_list(scene, self.shell.camera());
            paint_scene(&painter, rect.min, &cmds, scene);
        }
    }

    /// Pointer input for a screen-space (3D) session: left-drag orbit, right/shift-drag pan,
    /// wheel zoom, click select, double-click reset. Graphic input only; never reaches physics.
    fn view_input_ui(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: egui::Rect) {
        use crate::shell::ViewInput;
        if resp.dragged() {
            let d = resp.drag_delta();
            let pan =
                resp.dragged_by(egui::PointerButton::Secondary) || ui.input(|i| i.modifiers.shift);
            let (dx_px, dy_px) = (f64::from(d.x), f64::from(d.y));
            self.shell.view_input(if pan {
                ViewInput::Pan { dx_px, dy_px }
            } else {
                ViewInput::Orbit { dx_px, dy_px }
            });
        }
        if resp.hovered() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 {
                self.shell.view_input(ViewInput::Zoom {
                    factor: (f64::from(scroll) * 0.005).exp(),
                });
            }
        }
        if resp.double_clicked() {
            self.shell.view_input(ViewInput::Reset);
        } else if resp.clicked()
            && let Some(p) = resp.interact_pointer_pos()
        {
            self.shell.view_input(ViewInput::Click {
                x: f64::from(p.x - rect.min.x),
                y: f64::from(p.y - rect.min.y),
            });
        }
    }

    fn drive_robot3d(&mut self, ctx: &egui::Context) {
        let Some(t) = self.robot3d.as_mut() else {
            return;
        };
        t.frames += 1;
        ctx.request_repaint();
        match t.phase {
            TourPhase::WarmUp => {
                if t.frames > 8 {
                    t.phase = TourPhase::Drive;
                    t.frames = 0;
                }
            }
            TourPhase::Drive => {
                if let Err(e) = crate::robot3d::drive_shot(&mut self.shell, t.idx, &mut t.next_cmd)
                {
                    eprintln!("robot3d step {} failed: {e}", t.idx);
                    t.phase = TourPhase::Done;
                    return;
                }
                self.last_rect_size = None;
                t.phase = TourPhase::Settle;
                t.frames = 0;
            }
            TourPhase::Settle => {
                if t.frames > 6 {
                    t.phase = TourPhase::Shot;
                    t.frames = 0;
                }
            }
            TourPhase::Shot => {
                if !t.waiting {
                    let slot = Arc::clone(&t.pending);
                    ctx.request_screenshot(move |img| {
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(img);
                        }
                    });
                    t.waiting = true;
                } else if let Some(img) = t.pending.lock().ok().and_then(|mut g| g.take()) {
                    let name = ROBOT3D_SHOTS[t.idx];
                    let mut r = analyze(name, &img);
                    let p = t.out_dir.join(format!("{name}.png"));
                    if write_png(&p, &img).is_ok() {
                        r.path = Some(p);
                    }
                    r.lab = self.shell.active_lab_id().unwrap_or("").to_owned();
                    t.shots.push(r);
                    t.waiting = false;
                    t.frames = 0;
                    t.idx += 1;
                    t.phase = if t.idx >= ROBOT3D_SHOTS.len() {
                        TourPhase::Done
                    } else {
                        TourPhase::Drive
                    };
                }
            }
            TourPhase::Done => {
                if !t.shots.is_empty()
                    && let Ok(mut r) = self.report.lock()
                {
                    r.shots = std::mem::take(&mut t.shots);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    fn fit(&mut self, size: (f64, f64)) {
        self.shell.resize(size.0, size.1);
    }

    fn drive_tour(&mut self, ctx: &egui::Context) {
        let Some(t) = self.tour.as_mut() else {
            return;
        };
        t.frames += 1;
        ctx.request_repaint();
        match t.phase {
            TourPhase::WarmUp => {
                if t.frames > 8 {
                    t.phase = TourPhase::Drive;
                    t.frames = 0;
                }
            }
            TourPhase::Drive => {
                let step: TourStep = TOUR_PLAN[t.idx];
                match drive_step(&mut self.shell, &t.life, &step, &mut t.next_cmd) {
                    Ok(f) => t.facts.push(f),
                    Err(e) => {
                        eprintln!("tour step {} failed: {e}", step.lab);
                        t.phase = TourPhase::Done;
                        return;
                    }
                }
                self.last_rect_size = None; // re-fit the new scene
                t.phase = TourPhase::Settle;
                t.frames = 0;
            }
            TourPhase::Settle => {
                if t.frames > 6 {
                    t.phase = TourPhase::Shot;
                    t.frames = 0;
                }
            }
            TourPhase::Shot => {
                if !t.waiting {
                    let slot = Arc::clone(&t.pending);
                    ctx.request_screenshot(move |img| {
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(img);
                        }
                    });
                    t.waiting = true;
                } else if let Some(img) = t.pending.lock().ok().and_then(|mut g| g.take()) {
                    let lab = TOUR_PLAN[t.idx].lab;
                    let name = format!("{:02}_{}", t.idx + 1, lab.trim_start_matches("lab."));
                    let mut r = analyze(&name, &img);
                    let p = t.out_dir.join(format!("{name}.png"));
                    if write_png(&p, &img).is_ok() {
                        r.path = Some(p);
                    }
                    r.lab = self.shell.active_lab_id().unwrap_or("").to_owned();
                    r.camera_zoom = self.shell.camera().zoom().to_bits();
                    t.shots.push(r);
                    t.waiting = false;
                    t.frames = 0;
                    t.idx += 1;
                    t.phase = if t.idx >= TOUR_PLAN.len() {
                        TourPhase::Done
                    } else {
                        TourPhase::Drive
                    };
                }
            }
            TourPhase::Done => {
                // Close is asynchronous: later Done frames must not overwrite the report.
                if !t.shots.is_empty()
                    && let Ok(mut r) = self.report.lock()
                {
                    r.shots = std::mem::take(&mut t.shots);
                    r.tour = std::mem::take(&mut t.facts);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    /// Interactive control through the control port: `Step` or `Reset` on the active lab.
    fn submit_control(&mut self, action: crate::control::ControlAction) {
        use crate::control::{CommandId, ControlCommand, ControlPort, ObservationPort};
        if let (Some(Ok(control)), Some(obs)) =
            (self.shell.control_port(), self.shell.observation_port())
        {
            self.next_cmd += 1;
            control.submit(ControlCommand {
                id: CommandId(self.next_cmd),
                session: obs.observe().session,
                action,
            });
            self.shell.sync_active();
        }
    }

    fn drive_script(&mut self, ctx: &egui::Context) {
        let Some(sc) = self.script.as_mut() else {
            return;
        };
        sc.frames_in_step += 1;
        ctx.request_repaint();
        let take = |sc: &mut Script| sc.pending.lock().ok().and_then(|mut g| g.take());
        match sc.step {
            Step::WarmUp => {
                if sc.frames_in_step > 8 {
                    sc.step = Step::ShotInitial;
                    sc.frames_in_step = 0;
                }
            }
            Step::ShotInitial | Step::ShotZoomPan | Step::ShotResized | Step::ShotLab => {
                if !sc.waiting {
                    let slot = Arc::clone(&sc.pending);
                    ctx.request_screenshot(move |img| {
                        if let Ok(mut g) = slot.lock() {
                            *g = Some(img);
                        }
                    });
                    sc.waiting = true;
                } else if let Some(img) = take(sc) {
                    let name = match sc.step {
                        Step::ShotInitial => "1_initial_fit",
                        Step::ShotZoomPan => "2_zoom_pan",
                        Step::ShotResized => "3_resized_refit",
                        _ => "4_lab_beta",
                    };
                    let mut r = analyze(name, &img);
                    let p = sc.out_dir.join(format!("{name}.png"));
                    if write_png(&p, &img).is_ok() {
                        r.path = Some(p);
                    }
                    r.lab = self.shell.active_lab_id().unwrap_or("").to_owned();
                    r.camera_zoom = self.shell.camera().zoom().to_bits();
                    sc.shots.push(r);
                    sc.waiting = false;
                    sc.frames_in_step = 0;
                    sc.step = match sc.step {
                        Step::ShotInitial => {
                            // Programmatic input: zoom x1.6 about the viewport centre, then pan.
                            let (w, h) = self.shell.camera().window();
                            let cam = self.shell.camera_mut();
                            cam.zoom_at(1.6, w / 2.0, h / 2.0);
                            cam.pan(40.0, -25.0);
                            Step::SettleZoomPan
                        }
                        Step::ShotZoomPan => Step::Resize,
                        Step::ShotResized => Step::SwitchLab,
                        _ => Step::Done,
                    };
                }
            }
            Step::SettleZoomPan => {
                if sc.frames_in_step > 4 {
                    sc.step = Step::ShotZoomPan;
                    sc.frames_in_step = 0;
                }
            }
            Step::Resize => {
                if sc.frames_in_step == 1 {
                    sc.size_before_resize = ctx.content_rect().size();
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        720.0, 380.0,
                    )));
                } else if (ctx.content_rect().size() != sc.size_before_resize
                    && sc.frames_in_step > 6)
                    || sc.frames_in_step > 180
                {
                    sc.step = Step::ShotResized;
                    sc.frames_in_step = 0;
                }
            }
            Step::SwitchLab => {
                let _ = self.shell.select_lab("fixture.beta");
                self.last_rect_size = None; // force fit of the new scene
                sc.step = Step::ShotLab;
                sc.frames_in_step = 0;
            }
            Step::Done => {
                if !sc.shots.is_empty()
                    && let Ok(mut r) = self.report.lock()
                {
                    r.shots = std::mem::take(&mut sc.shots);
                }
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }
}

impl eframe::App for StudioApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if !self.gl_info_done
            && let Some(gl) = frame.gl()
        {
            // SAFETY: plain string queries on the current GL context owned by eframe.
            let (r, v) = unsafe {
                (
                    gl.get_parameter_string(eframe::glow::RENDERER),
                    gl.get_parameter_string(eframe::glow::VERSION),
                )
            };
            if let Ok(mut rep) = self.report.lock() {
                rep.gl_renderer = r;
                rep.gl_version = v;
            }
            self.gl_info_done = true;
        }
        let ctx = ui.ctx().clone();
        ctx.set_visuals(egui::Visuals::light());
        egui::Panel::top("labs").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label("Lab:");
                let ids: Vec<String> = self
                    .shell
                    .lab_ids()
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect();
                let active = self.shell.active_lab_id().map(str::to_owned);
                for id in ids {
                    if ui
                        .selectable_label(active.as_deref() == Some(id.as_str()), &id)
                        .clicked()
                    {
                        let _ = self.shell.select_lab(&id);
                        self.last_rect_size = None;
                    }
                }
                if ui.button("Fit").clicked() {
                    self.last_rect_size = None;
                }
                if ui.button("Step").clicked() {
                    self.submit_control(crate::control::ControlAction::Step);
                }
                if ui.button("Reset").clicked() {
                    self.submit_control(crate::control::ControlAction::Reset);
                }
                if self.shell.active_is_screen_space() {
                    if ui.button("Ortho/Persp").clicked() {
                        self.shell
                            .view_input(crate::shell::ViewInput::ToggleProjection);
                    }
                    if ui.button("Reset view").clicked() {
                        self.shell.view_input(crate::shell::ViewInput::Reset);
                    }
                } else {
                    ui.label(format!("zoom {:.2}", self.shell.camera().zoom()));
                }
            });
        });
        egui::CentralPanel::default()
            .frame(egui::Frame::NONE)
            .show(ui, |ui| self.viewport_ui(ui));
        self.drive_script(&ctx);
        self.drive_tour(&ctx);
        self.drive_robot3d(&ctx);
    }
}

fn run(
    shell: StudioShell,
    script: Option<PathBuf>,
    tour: Option<(PathBuf, LifeProbe)>,
    robot3d: Option<PathBuf>,
    initial_size: [f32; 2],
) -> Result<ScriptReport, String> {
    let report = Arc::new(Mutex::new(ScriptReport::default()));
    let app = StudioApp {
        shell,
        last_rect_size: None,
        tour: tour.map(|(out_dir, life)| TourRun {
            out_dir,
            idx: 0,
            phase: TourPhase::WarmUp,
            frames: 0,
            waiting: false,
            pending: Arc::new(Mutex::new(None)),
            shots: Vec::new(),
            facts: Vec::new(),
            life,
            next_cmd: 0,
        }),
        robot3d: robot3d.map(|out_dir| Robot3dRun {
            out_dir,
            idx: 0,
            phase: TourPhase::WarmUp,
            frames: 0,
            waiting: false,
            pending: Arc::new(Mutex::new(None)),
            shots: Vec::new(),
            next_cmd: 0,
        }),
        next_cmd: 0,
        script: script.map(|out_dir| Script {
            out_dir,
            step: Step::WarmUp,
            frames_in_step: 0,
            pending: Arc::new(Mutex::new(None)),
            waiting: false,
            size_before_resize: egui::Vec2::ZERO,
            shots: Vec::new(),
        }),
        report: Arc::clone(&report),
        gl_info_done: false,
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Gogamen Studio")
            .with_inner_size(initial_size),
        ..Default::default()
    };
    eframe::run_native(
        "gogamen-studio",
        options,
        Box::new(move |_cc| Ok(Box::new(app))),
    )
    .map_err(|e| e.to_string())?;
    let out = report.lock().map_err(|e| e.to_string())?.clone();
    Ok(out)
}

/// Opens a normal interactive window on `shell` (blocks until it is closed).
pub fn run_interactive(shell: StudioShell) -> Result<(), String> {
    shell_with_default_lab(shell).and_then(|s| run(s, None, None, None, [900.0, 620.0]).map(|_| ()))
}

/// Runs the deterministic scripted session (zoom+pan, window resize, lab switch) and writes a
/// PNG capture of every step under `out_dir`. Requires a real display.
pub fn run_script(shell: StudioShell, out_dir: &Path) -> Result<ScriptReport, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let shell = shell_with_default_lab(shell)?;
    run(shell, Some(out_dir.to_owned()), None, None, [900.0, 620.0])
}

/// Runs the multi-lab tour ([`TOUR_PLAN`]) in a real window and captures one PNG per lab under
/// `out_dir` (`NN_<lab>.png`). Requires a real display.
pub fn run_tour(out_dir: &Path) -> Result<ScriptReport, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let (shell, life) = tour_shell(900.0, 620.0);
    run(
        shell,
        None,
        Some((out_dir.to_owned(), life)),
        None,
        [900.0, 620.0],
    )
}

/// Interactive window over the seven tour labs (Step/Reset buttons drive the control port).
pub fn run_tour_interactive() -> Result<(), String> {
    let (shell, _life) = tour_shell(900.0, 620.0);
    shell_with_default_lab(shell).and_then(|s| run(s, None, None, None, [900.0, 620.0]).map(|_| ()))
}

/// Interactive window on the tour shell with `lab.robot3d` pre-selected.
pub fn run_robot3d_interactive() -> Result<(), String> {
    let (mut shell, _life) = tour_shell(900.0, 620.0);
    shell
        .select_lab(crate::robot3d::ROBOT3D_ID)
        .map_err(|e| e.to_string())?;
    run(shell, None, None, None, [900.0, 620.0]).map(|_| ())
}

/// Runs the scripted 3D captures ([`ROBOT3D_SHOTS`]) in a real window and writes one PNG per
/// state under `out_dir`. Requires a real display.
pub fn run_robot3d_shots(out_dir: &Path) -> Result<ScriptReport, String> {
    std::fs::create_dir_all(out_dir).map_err(|e| e.to_string())?;
    let (shell, _life) = tour_shell(900.0, 620.0);
    run(shell, None, None, Some(out_dir.to_owned()), [900.0, 620.0])
}

fn shell_with_default_lab(mut shell: StudioShell) -> Result<StudioShell, String> {
    if shell.active_lab_id().is_none() {
        let first = shell
            .lab_ids()
            .first()
            .map(|s| (*s).to_owned())
            .ok_or("shell has no labs")?;
        shell.select_lab(&first).map_err(|e| e.to_string())?;
    }
    Ok(shell)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::scene_alpha;

    #[test]
    fn draw_list_follows_camera_and_backend_is_window_kind() {
        let scene = scene_alpha();
        let mut cam = Camera2D::new(200.0, 100.0, true);
        let before = WindowBackend.render(&scene, &cam);
        assert_eq!(before, WindowBackend.render(&scene, &cam));
        assert!(matches!(draw_list(&scene, &cam)[0], DrawCmd::Circle { .. }));
        cam.pan(5.0, 5.0);
        assert_ne!(before, WindowBackend.render(&scene, &cam));
        assert_eq!(WindowBackend.kind(), BackendKind::Window);
    }

    #[test]
    fn empty_report_is_never_a_pass() {
        let r = ScriptReport::default().to_render_test_report();
        assert_eq!(r.status, RenderTestStatus::Fail);
    }
}
