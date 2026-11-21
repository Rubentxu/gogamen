//! Arena: vistas sincronizadas (GOG-101), headless y sin Bevy/egui.
//!
//! Compara dos [`Recording`] Grid (GOG-94) en un viewport compartido con **un solo cursor de tick
//! lógico**. Sólo recibe datos: no ejecuta planificadores ni dinámica.
//!
//! - Alineación únicamente por tick lógico: cada panel muestra el frame grabado *exacto* de ese
//!   tick. Si el tick cae en un hueco se muestra [`PanelSlot::Gap`]; si supera el final de esa
//!   grabación, [`PanelSlot::Ended`]. Nunca se repite/extrapola el último frame ni se interpola.
//! - Las métricas **no se recalculan**: provienen de [`PairedSeedOutcome`] (`gogamen-arena`,
//!   GOG-100). Sin informe pareado, con semillas/mapas que no corresponden, el veredicto es
//!   [`Verdict::NotComparable`] y no se declara ganador.
//! - Cada panel proyecta con el mismo [`project_grid_scene`] que el modo en vivo.

use gogamen_arena::{PairedSeedOutcome, PairedWinner, PlannerRunResult};
use gogamen_core::{EpistemicKind, Tick};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};

use crate::camera::{Camera2D, scene_bounds};
use crate::counterfactual::RecordingId;
use crate::grid_lab::{GridWorldView, project_grid_scene};
use crate::replay::{Frame, Gap, Recording};
use crate::viewport::SoftwareSvgBackend;

/// Lado de un panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    fn tag(self) -> &'static str {
        match self {
            Self::Left => "A",
            Self::Right => "B",
        }
    }
}

/// Qué hay en un panel al tick común.
#[derive(Debug, Clone, PartialEq)]
pub enum PanelSlot {
    /// Frame grabado exacto de ese tick.
    Frame(Frame),
    /// Tick sin dato grabado dentro del rango de la grabación.
    Gap(Gap),
    /// El tick supera el final de esta grabación (sin dato; no se repite el último frame).
    Ended { last_tick: u64 },
}

/// Métricas del run tal como las calculó el runner de la arena (no se recalculan aquí).
#[derive(Debug, Clone, PartialEq)]
pub struct PanelMetrics {
    pub planner_id: String,
    pub status: &'static str,
    pub steps: u64,
    pub transitions: u64,
    pub total_reward: f64,
}

impl PanelMetrics {
    fn of(r: &PlannerRunResult) -> Self {
        Self {
            planner_id: r.planner_id.clone(),
            status: r.status.label(),
            steps: r.steps_executed,
            transitions: r.transitions_evaluated,
            total_reward: r.total_reward,
        }
    }
}

/// Datos de un panel (propios, sin referencias a motor).
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaPanel {
    pub side: Side,
    /// Id distintivo: lado + hash FNV-1a de la grabación.
    pub run_id: String,
    pub seed: u64,
    pub map_size: (usize, usize),
    pub last_tick: u64,
    pub frame_count: usize,
    pub gap_count: usize,
    pub slot: PanelSlot,
    /// Vista del mundo para el proyector Grid (sólo si hay frame).
    pub world: Option<GridWorldView>,
    pub metrics: Option<PanelMetrics>,
}

/// Veredicto de la comparación.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// Métricas comparables: el ganador es el del informe pareado de la arena.
    Comparable { winner: Option<Side> },
    /// No comparable: no hay ganador.
    NotComparable { reason: String },
}

/// Vista completa de la arena a un tick lógico común.
#[derive(Debug, Clone, PartialEq)]
pub struct ArenaView {
    pub tick: u64,
    /// Longitud de la línea de tiempo común (máximo de los `last_tick`).
    pub timeline_len: u64,
    pub left: ArenaPanel,
    pub right: ArenaPanel,
    pub verdict: Verdict,
}

fn panel(
    side: Side,
    rec: &Recording,
    tick: u64,
    metrics: Option<PanelMetrics>,
    run_id: String,
) -> ArenaPanel {
    let slot = if tick > rec.last_tick() {
        PanelSlot::Ended {
            last_tick: rec.last_tick(),
        }
    } else if let Some(f) = rec.frame_at(tick) {
        PanelSlot::Frame(f.clone())
    } else {
        PanelSlot::Gap(
            rec.gaps()
                .iter()
                .copied()
                .find(|g| (g.from..=g.to).contains(&tick))
                .unwrap_or(Gap {
                    from: tick,
                    to: tick,
                }),
        )
    };
    let world = match &slot {
        PanelSlot::Frame(f) => Some(rec.world_view(f)),
        _ => None,
    };
    ArenaPanel {
        side,
        run_id,
        seed: rec.seed(),
        map_size: (rec.map().width(), rec.map().height()),
        last_tick: rec.last_tick(),
        frame_count: rec.frames().len(),
        gap_count: rec.gaps().len(),
        slot,
        world,
        metrics,
    }
}

fn verdict(
    left: &Recording,
    right: &Recording,
    paired: Option<&PairedSeedOutcome>,
) -> (Verdict, bool) {
    let nc = |r: &str| {
        (
            Verdict::NotComparable {
                reason: r.to_string(),
            },
            false,
        )
    };
    let Some(p) = paired else {
        return nc("no arena report for these runs");
    };
    if left.seed() != p.seed.as_u64() || right.seed() != p.seed.as_u64() {
        return nc("recording seed differs from the arena report seed");
    }
    if left.map() != right.map() {
        return nc("runs use different maps");
    }
    let winner = match p.outcome_winner {
        PairedWinner::Left => Some(Side::Left),
        PairedWinner::Right => Some(Side::Right),
        PairedWinner::Tie => None,
    };
    (Verdict::Comparable { winner }, true)
}

impl ArenaView {
    /// Construye la vista a `tick` (acotado a la línea de tiempo común). `paired` es el resultado
    /// pareado del runner de la arena; sin él (o si no corresponde) no es comparable.
    pub fn new(
        tick: u64,
        left: &Recording,
        right: &Recording,
        paired: Option<&PairedSeedOutcome>,
    ) -> Self {
        let timeline_len = left.last_tick().max(right.last_tick());
        let tick = tick.min(timeline_len);
        let (verdict, comparable) = verdict(left, right, paired);
        let m = |f: fn(&PairedSeedOutcome) -> &PlannerRunResult| {
            paired
                .filter(|_| comparable)
                .map(|p| PanelMetrics::of(f(p)))
        };
        let (lid, rid) = (RecordingId::of(left).0, RecordingId::of(right).0);
        Self {
            tick,
            timeline_len,
            left: panel(
                Side::Left,
                left,
                tick,
                m(|p| &p.left),
                format!("A-{lid:016x}"),
            ),
            right: panel(
                Side::Right,
                right,
                tick,
                m(|p| &p.right),
                format!("B-{rid:016x}"),
            ),
            verdict,
        }
    }
}

/// Cursor lógico único compartido por ambos paneles.
pub struct ArenaSession<'a> {
    left: &'a Recording,
    right: &'a Recording,
    paired: Option<&'a PairedSeedOutcome>,
    tick: u64,
}

impl<'a> ArenaSession<'a> {
    pub fn new(
        left: &'a Recording,
        right: &'a Recording,
        paired: Option<&'a PairedSeedOutcome>,
    ) -> Self {
        Self {
            left,
            right,
            paired,
            tick: 0,
        }
    }
    pub fn timeline_len(&self) -> u64 {
        self.left.last_tick().max(self.right.last_tick())
    }
    pub fn tick(&self) -> u64 {
        self.tick
    }
    /// Salta ambos paneles al mismo tick (acotado a la línea de tiempo común).
    pub fn seek(&mut self, tick: u64) {
        self.tick = tick.min(self.timeline_len());
    }
    pub fn step_forward(&mut self) {
        self.seek(self.tick.saturating_add(1));
    }
    pub fn step_back(&mut self) {
        self.seek(self.tick.saturating_sub(1));
    }
    pub fn view(&self) -> ArenaView {
        ArenaView::new(self.tick, self.left, self.right, self.paired)
    }
}

// ===== RENDER SECTION: proyector puro; nada de dinámica, planificadores ni motor desde aquí =====

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite coordinate")
}

/// Líneas de texto de un panel.
pub fn panel_lines(p: &ArenaPanel, tick: u64) -> Vec<String> {
    let mut l = vec![
        format!("run {} seed={}", p.run_id, p.seed),
        format!(
            "tick {tick} | recorded ticks 0..={} | frames {} | gaps {}",
            p.last_tick, p.frame_count, p.gap_count
        ),
    ];
    l.push(match &p.slot {
        PanelSlot::Frame(f) => format!(
            "agent ({},{}) reward={:.2} terminal={:?}",
            f.agent.x, f.agent.y, f.total_reward, f.terminal
        ),
        PanelSlot::Gap(g) => format!("GAP ticks {}..={}: no data", g.from, g.to),
        PanelSlot::Ended { last_tick } => {
            format!("ENDED at tick {last_tick}: no data past end")
        }
    });
    l.push(match &p.metrics {
        Some(m) => format!(
            "arena metrics: {} {} steps={} transitions={} reward={:.2}",
            m.planner_id, m.status, m.steps, m.transitions, m.total_reward
        ),
        None => "arena metrics: not available".to_string(),
    });
    l
}

/// Línea de veredicto.
pub fn verdict_line(v: &Verdict) -> String {
    match v {
        Verdict::Comparable { winner: Some(s) } => {
            format!("verdict: {} wins (arena paired outcome)", s.tag())
        }
        Verdict::Comparable { winner: None } => "verdict: tie (arena paired outcome)".to_string(),
        Verdict::NotComparable { reason } => format!("verdict: not comparable ({reason})"),
    }
}

/// Proyección pura. Los dos paneles comparten viewport (offset en X) y un cursor de tick.
/// Capas: `Environment`/`Labels`/`Decisions` (proyector Grid reutilizado, desplazado),
/// `Custom(1)` para la línea de tiempo común y marcadores de fin/hueco.
pub fn project_arena_scene(view: &ArenaView) -> Scene2D {
    let mut scene = Scene2D::new(
        SceneId::new(101),
        Tick::new(view.tick),
        EpistemicKind::Observed,
        CoordinateSpace::meters(),
    );
    for layer in [
        LayerId::Environment,
        LayerId::Labels,
        LayerId::Decisions,
        LayerId::Custom(1),
    ] {
        scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }
    let mut next = 1u64;
    let mut id = || {
        let n = next;
        next += 1;
        SceneNodeId::new(n)
    };
    let (w, h) = (
        view.left.map_size.0.max(view.right.map_size.0) as f64,
        view.left.map_size.1.max(view.right.map_size.1) as f64,
    );
    let pitch = w + 2.0;
    for (i, p) in [&view.left, &view.right].into_iter().enumerate() {
        let dx = i as f64 * pitch;
        if let Some(world) = &p.world {
            let sub = project_grid_scene(world);
            for layer in sub.layers() {
                for n in layer.nodes() {
                    let prim = translate(n.primitive(), dx);
                    let mut node =
                        SceneNode::new(id(), n.entity_id(), n.epistemic_kind(), prim, false);
                    if let Some(c) = n.state_code() {
                        node = node.with_state_code(c);
                    }
                    scene.add_node(layer.id(), node).expect("node");
                }
            }
        } else {
            // Sin dato: sólo el contorno del panel; nunca el último frame.
            let rect = ScenePrimitive::rect(pt(dx, -h), w, h).expect("panel rect");
            scene
                .add_node(
                    LayerId::Custom(1),
                    SceneNode::new(id(), None, EpistemicKind::Observed, rect, false),
                )
                .expect("node");
        }
        for (j, line) in panel_lines(p, view.tick).into_iter().enumerate() {
            let lab = ScenePrimitive::label(pt(dx, -h - 0.6 - j as f64 * 0.6), line);
            scene
                .add_node(
                    LayerId::Labels,
                    SceneNode::new(id(), None, EpistemicKind::Observed, lab, false),
                )
                .expect("node");
        }
    }
    // Línea de tiempo común sobre ambos paneles + cursor.
    let total = 2.0 * pitch - 2.0;
    let line = ScenePrimitive::polyline(vec![pt(0.0, 0.8), pt(total, 0.8)], false).expect("line");
    scene
        .add_node(
            LayerId::Custom(1),
            SceneNode::new(id(), None, EpistemicKind::Observed, line, false),
        )
        .expect("node");
    let frac = if view.timeline_len == 0 {
        0.0
    } else {
        view.tick as f64 / view.timeline_len as f64
    };
    let cursor = ScenePrimitive::circle(pt(frac * total, 0.8), 0.15).expect("cursor");
    scene
        .add_node(
            LayerId::Decisions,
            SceneNode::new(id(), None, EpistemicKind::Observed, cursor, false),
        )
        .expect("node");
    let t = ScenePrimitive::label(
        pt(0.0, 1.2),
        format!("shared tick {}/{}", view.tick, view.timeline_len),
    );
    scene
        .add_node(
            LayerId::Labels,
            SceneNode::new(id(), None, EpistemicKind::Observed, t, false),
        )
        .expect("node");
    let v = ScenePrimitive::label(pt(0.0, -h - 0.6 - 4.0 * 0.6), verdict_line(&view.verdict));
    scene
        .add_node(
            LayerId::Labels,
            SceneNode::new(id(), None, EpistemicKind::Observed, v, false),
        )
        .expect("node");
    scene
}

fn translate(p: &ScenePrimitive, dx: f64) -> ScenePrimitive {
    match p {
        ScenePrimitive::Circle { center, radius } => ScenePrimitive::Circle {
            center: pt(center.x() + dx, center.y()),
            radius: *radius,
        },
        ScenePrimitive::Polyline { points, closed } => ScenePrimitive::Polyline {
            points: points.iter().map(|q| pt(q.x() + dx, q.y())).collect(),
            closed: *closed,
        },
        ScenePrimitive::Rect { min, width, height } => ScenePrimitive::Rect {
            min: pt(min.x() + dx, min.y()),
            width: *width,
            height: *height,
        },
        ScenePrimitive::Label { position, text } => {
            ScenePrimitive::label(pt(position.x() + dx, position.y()), text.clone())
        }
        ScenePrimitive::Image {
            min,
            width,
            height,
            texture,
        } => ScenePrimitive::Image {
            min: pt(min.x() + dx, min.y()),
            width: *width,
            height: *height,
            texture: texture.clone(),
        },
    }
}

/// SVG determinista de la vista.
pub fn render_arena_svg(view: &ArenaView, width: f64, height: f64) -> String {
    let scene = project_arena_scene(view);
    let mut cam = Camera2D::new(width, height, true);
    if let Some(bounds) = scene_bounds(&scene) {
        cam.fit_to_window(width, height, &bounds, 8.0);
    }
    SoftwareSvgBackend.render_svg(&scene, &cam)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::record_grid;
    use gogamen_arena::{
        ArenaBudget, ArenaComparisonConfig, ArenaPlanner, ArenaScenario, ArenaSeedSet,
        GridArenaConfig, run_arena_comparison,
    };
    use gogamen_core::MasterSeed;
    use gogamen_lab_grid::{GridMap, GridWorldConfig};

    fn rec(seed: u64, ticks: u64) -> Recording {
        record_grid(seed, GridMap::cul_de_sac_map(), ticks).unwrap()
    }

    fn texts(scene: &Scene2D) -> Vec<String> {
        scene
            .layers()
            .iter()
            .flat_map(|l| l.nodes())
            .filter_map(|n| match n.primitive() {
                ScenePrimitive::Label { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    fn paired(seed: u64) -> PairedSeedOutcome {
        let map = GridMap::from_ascii(&["#######", "#S...G#", "#######"]).unwrap();
        let cfg = ArenaComparisonConfig {
            spec_id: "arena-view".to_owned(),
            scenario: ArenaScenario::Grid(
                GridArenaConfig::new("corridor", map, GridWorldConfig::default()).unwrap(),
            ),
            budget: ArenaBudget::new(20, 4_000, 128),
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(seed)],
                vec![MasterSeed::new(seed + 1000)],
            )
            .unwrap(),
            left_planner: ArenaPlanner::GridRandomSeeded,
            right_planner: ArenaPlanner::GridPrivilegedBfs,
        };
        run_arena_comparison(&cfg)
            .unwrap()
            .paired_outcomes
            .into_iter()
            .find(|p| p.seed.as_u64() == seed)
            .unwrap()
    }

    fn agent_x(scene: &Scene2D) -> Vec<f64> {
        scene
            .layers()
            .iter()
            .find(|l| l.id() == LayerId::Decisions)
            .unwrap()
            .nodes()
            .iter()
            .filter_map(|n| match n.primitive() {
                ScenePrimitive::Circle { center, radius } if *radius > 0.3 => Some(center.x()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn ac1_two_panels_share_one_tick_with_distinct_ids_and_seeds() {
        let (a, b) = (rec(42, 12), rec(7, 12));
        let mut s = ArenaSession::new(&a, &b, None);
        s.seek(5);
        let v = s.view();
        assert_eq!(v.tick, 5);
        assert_eq!(v.left.world.as_ref().unwrap().tick.as_u64(), 5);
        assert_eq!(v.right.world.as_ref().unwrap().tick.as_u64(), 5);
        assert_ne!(v.left.run_id, v.right.run_id);
        let scene = project_arena_scene(&v);
        let t = texts(&scene);
        assert!(
            t.iter()
                .any(|x| x.contains(&v.left.run_id) && x.contains("seed=42"))
        );
        assert!(
            t.iter()
                .any(|x| x.contains(&v.right.run_id) && x.contains("seed=7"))
        );
        assert!(t.iter().any(|x| x == "shared tick 5/12"));
        // un agente por panel, el segundo desplazado a la derecha
        let xs = agent_x(&scene);
        assert_eq!(xs.len(), 2);
        assert!(xs[1] > xs[0] + 5.0);
        // scrub: ambos siguen al mismo cursor
        for t in [0, 3, 12, 99] {
            s.seek(t);
            let v = s.view();
            let want = t.min(12);
            assert_eq!(v.tick, want);
            assert_eq!(v.left.world.as_ref().unwrap().tick.as_u64(), want);
            assert_eq!(v.right.world.as_ref().unwrap().tick.as_u64(), want);
        }
        s.step_back();
        assert_eq!(s.tick(), 11);
    }

    #[test]
    fn ac2_shorter_run_shows_ended_marker_not_last_frame() {
        let (long, short) = (rec(42, 12), rec(42, 4));
        assert!(short.last_tick() < long.last_tick());
        let v = ArenaView::new(9, &long, &short, None);
        assert_eq!(v.tick, 9);
        assert!(matches!(v.left.slot, PanelSlot::Frame(_)));
        assert_eq!(
            v.right.slot,
            PanelSlot::Ended {
                last_tick: short.last_tick()
            }
        );
        assert!(v.right.world.is_none());
        let scene = project_arena_scene(&v);
        assert!(
            texts(&scene)
                .iter()
                .any(|x| x.contains("ENDED") && x.contains("no data"))
        );
        // sólo el agente del panel largo; el corto no repite su último frame
        assert_eq!(agent_x(&scene).len(), 1);
        // el último tick propio sí muestra frame
        let at_end = ArenaView::new(short.last_tick(), &long, &short, None);
        assert!(matches!(at_end.right.slot, PanelSlot::Frame(_)));
    }

    #[test]
    fn ac2_gap_in_one_run_is_explicit_and_other_unaffected() {
        let full = rec(42, 10);
        let frames: Vec<Frame> = full
            .frames()
            .iter()
            .filter(|f| f.tick != 3 && f.tick != 4)
            .cloned()
            .collect();
        let rows = crate::replay::map_to_rows(full.map());
        let gappy = Recording::new(42, rows, 10, frames).unwrap();
        let v = ArenaView::new(3, &full, &gappy, None);
        assert_eq!(v.right.slot, PanelSlot::Gap(Gap { from: 3, to: 4 }));
        assert!(matches!(v.left.slot, PanelSlot::Frame(_)));
        let scene = project_arena_scene(&v);
        assert!(texts(&scene).iter().any(|x| x.contains("GAP ticks 3..=4")));
        assert_eq!(agent_x(&scene).len(), 1);
        // el panel sin hueco es idéntico con o sin hueco en el otro
        let other = ArenaView::new(3, &full, &full, None);
        assert_eq!(v.left.slot, other.left.slot);
        assert_eq!(v.left.world, other.left.world);
    }

    #[test]
    fn ac2_different_pace_aligned_by_logical_tick_without_interpolation() {
        let dense = rec(42, 10);
        let rows = crate::replay::map_to_rows(dense.map());
        // ritmo mitad: sólo ticks pares
        let sparse_frames: Vec<Frame> = dense
            .frames()
            .iter()
            .filter(|f| f.tick % 2 == 0)
            .cloned()
            .collect();
        let sparse = Recording::new(42, rows, 10, sparse_frames).unwrap();
        for t in 0..=10u64 {
            let v = ArenaView::new(t, &dense, &sparse, None);
            assert_eq!(v.left.world.as_ref().unwrap().tick.as_u64(), t);
            match &v.right.slot {
                PanelSlot::Frame(f) => {
                    assert_eq!(f.tick, t);
                    assert_eq!(t % 2, 0);
                    assert_eq!(Some(f), dense.frame_at(t));
                }
                PanelSlot::Gap(g) => {
                    assert_eq!(t % 2, 1);
                    assert_eq!((g.from, g.to), (t, t));
                    assert!(v.right.world.is_none());
                }
                PanelSlot::Ended { .. } => panic!("not ended"),
            }
        }
    }

    #[test]
    fn ac2_no_winner_when_not_comparable_and_metrics_withheld() {
        let (a, b) = (rec(42, 8), rec(7, 8));
        let p = paired(42);
        // sin informe
        let v = ArenaView::new(3, &a, &b, None);
        assert!(matches!(v.verdict, Verdict::NotComparable { .. }));
        assert!(v.left.metrics.is_none() && v.right.metrics.is_none());
        let t = texts(&project_arena_scene(&v));
        assert!(t.iter().any(|x| x.contains("not comparable")));
        assert!(!t.iter().any(|x| x.contains("wins") || x.contains("tie")));
        // semilla de la grabación B no coincide con el informe
        let v = ArenaView::new(3, &a, &b, Some(&p));
        assert!(matches!(v.verdict, Verdict::NotComparable { .. }));
        assert!(v.left.metrics.is_none());
    }

    #[test]
    fn ac3_comparable_metrics_come_from_arena_report() {
        let (a, b) = (rec(42, 12), rec(42, 6));
        let p = paired(42);
        let v = ArenaView::new(2, &a, &b, Some(&p));
        let want = match p.outcome_winner {
            PairedWinner::Left => Some(Side::Left),
            PairedWinner::Right => Some(Side::Right),
            PairedWinner::Tie => None,
        };
        assert_eq!(v.verdict, Verdict::Comparable { winner: want });
        let (lm, rm) = (
            v.left.metrics.as_ref().unwrap(),
            v.right.metrics.as_ref().unwrap(),
        );
        assert_eq!(lm.planner_id, p.left.planner_id);
        assert_eq!(lm.steps, p.left.steps_executed);
        assert_eq!(rm.total_reward, p.right.total_reward);
        assert_eq!(rm.transitions, p.right.transitions_evaluated);
        let t = texts(&project_arena_scene(&v));
        assert!(
            t.iter()
                .any(|x| x.contains(&lm.planner_id) && x.contains("steps="))
        );
        assert!(
            t.iter()
                .any(|x| x.starts_with("verdict:") && !x.contains("not comparable"))
        );
    }

    #[test]
    fn ac3_panels_reuse_grid_projector_and_output_is_deterministic() {
        let (a, b) = (rec(42, 12), rec(7, 12));
        let v = ArenaView::new(4, &a, &b, None);
        let single = project_grid_scene(v.left.world.as_ref().unwrap());
        let merged = project_arena_scene(&v);
        let count = |s: &Scene2D, l: LayerId| {
            s.layers()
                .iter()
                .find(|x| x.id() == l)
                .unwrap()
                .nodes()
                .len()
        };
        let single_env = count(&single, LayerId::Environment);
        assert_eq!(count(&merged, LayerId::Environment), 2 * single_env);
        let svg = render_arena_svg(&v, 800.0, 400.0);
        assert_eq!(svg, render_arena_svg(&v.clone(), 800.0, 400.0));
        assert_eq!(project_arena_scene(&v), project_arena_scene(&v));
        assert!(svg.starts_with("<svg"));
        // la vista no altera las grabaciones
        let before = (a.to_bytes(), b.to_bytes());
        let _ = ArenaSession::new(&a, &b, None).view();
        assert_eq!(before, (a.to_bytes(), b.to_bytes()));
    }

    #[test]
    fn render_section_has_no_dynamics_planner_or_engine() {
        let src = include_str!("arena_view.rs");
        let start = src.find("// ===== RENDER SECTION").unwrap();
        let end = src.find("#[cfg(test)]").unwrap();
        let render: String = src[start..end]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "GridWorldModel",
            "GridWorldEnv",
            "GridRollout",
            "BranchableSnapshot",
            "ActionGuard",
            "step_in_place",
            "restore(",
            "run_arena_comparison",
            "ArenaPlanner",
            "ArenaComparisonConfig",
            "Planner::",
            "record_grid",
            "Actuator",
            "ControlPort",
            "SessionHost",
            "std::net",
            "std::fs",
            "gogamen_lab_",
            "RngSource",
            "next_u64",
            ".step(",
            "summarize_",
            "DispersionStats",
        ] {
            assert!(
                !render.contains(banned),
                "render section references {banned}"
            );
        }
    }
}
