//! Studio timeline widget: control bits and decisive moments (GOG-202).
//!
//! Consumes the data layer of [`crate::decisive_timeline`] (`timeline_rows`,
//! `timeline_lines`, `open_counterfactual_at`) and adds the widget itself:
//!
//! - [`TimelineLayout`]: a normalized geometry (rows, control-bit bars, decisive markers,
//!   tick axis) derived only from the [`TimelineRow`]s;
//! - [`TimelineWidget`]: selection state machine. A decisive row opens the counterfactual
//!   debugger through the existing `open_counterfactual_at` (no duplicated logic) and keeps
//!   the resulting [`CounterfactualView`]; a non-decisive row never opens it;
//! - with the optional `window` feature, [`timeline_ui`] paints that geometry with egui and
//!   forwards a click on a row to the state machine;
//! - [`TimelineLayout::of_marks`], [`row_tick_at`] and the `window` painter `paint_timeline` are
//!   shared by every tick-row panel; the emergence panel ([`crate::emergence_ui`], GOG-205)
//!   reuses them instead of duplicating the geometry or the renderer.
//!
//! The [`Recording`] is only borrowed: opening the debugger rebuilds a checkpoint in isolated
//! branches and never writes to the recording (the widget tests assert an identical hash).
//! Without the `window` feature everything except the painter is compiled and tested, so the
//! state machine and the layout stay verifiable on a headless host.

use gogamen_core::{Action, DecisiveEvent};

use crate::counterfactual::{CounterfactualView, counterfactual_lines};
use crate::decisive_timeline::{
    OpenAtError, TimelineRow, open_counterfactual_at, timeline_lines, timeline_rows,
};
use crate::replay::Recording;

/// Normalized band `(top, bottom)` of the timeline rows inside the widget.
pub const TIMELINE_BAND: (f64, f64) = (0.04, 0.62);
/// Normalized x where control-bit bars start (left of it: row labels).
pub const BAR_X0: f64 = 0.34;
/// Normalized x where a full-scale control-bit bar ends.
pub const BAR_X1: f64 = 0.84;
/// Normalized x of the decisive-moment marker.
pub const MARKER_X: f64 = 0.92;

/// What the widget runs when a decisive moment is selected: the alternative root action and
/// the horizon (ticks after the checkpoint). Supplied by the caller, never guessed here.
#[derive(Debug, Clone, PartialEq)]
pub struct CounterfactualChoice {
    /// Alternative action for the first branch step.
    pub alternative: Action,
    /// Branch horizon in ticks.
    pub horizon: u64,
}

/// Geometry of one timeline row, in normalized widget coordinates (`0..=1`, y down).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RowGeometry {
    /// Tick of the row.
    pub tick: u64,
    /// Decisive mark of the row.
    pub decisive: bool,
    /// Row band.
    pub y0: f64,
    /// Row band end.
    pub y1: f64,
    /// Control-bit bar `(x0, x1)`; `None` when the event carries no control bits.
    pub bar: Option<(f64, f64)>,
    /// Decisive marker centre, if the row is decisive.
    pub marker: Option<(f64, f64)>,
}

/// Normalized layout of the whole timeline panel.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineLayout {
    /// Largest positive bar magnitude (bar full scale; `1.0` when there is none). It is the
    /// control bits for the timeline and the hypothesis evidence for the emergence panel.
    pub max_bits: f64,
    /// One entry per row, in tick order.
    pub rows: Vec<RowGeometry>,
    /// Tick axis: `(tick, normalized x)`.
    pub axis: Vec<(u64, f64)>,
}

fn bar_x(bits: f64, max_bits: f64) -> f64 {
    let ratio = if max_bits > 0.0 {
        (bits / max_bits).clamp(0.0, 1.0)
    } else {
        0.0
    };
    BAR_X0 + ratio * (BAR_X1 - BAR_X0)
}

impl TimelineLayout {
    /// Lays out the rows evenly inside [`TIMELINE_BAND`]; bars are proportional to control bits.
    pub fn of(rows: &[TimelineRow]) -> Self {
        Self::of_marks(rows.iter().map(|r| (r.tick, r.control_bits, r.decisive)))
    }

    /// Lays out generic `(tick, magnitude, marked)` entries with the same geometry as [`Self::of`].
    ///
    /// This is the shared layout of every tick-row panel: the timeline passes control bits and
    /// decisive marks, the emergence panel ([`crate::emergence_ui`]) passes hypothesis evidence
    /// and hypothesis marks. `max_bits` is the full-scale magnitude of both.
    pub fn of_marks(entries: impl Iterator<Item = (u64, Option<f64>, bool)>) -> Self {
        let entries: Vec<(u64, Option<f64>, bool)> = entries.collect();
        let max_bits = entries
            .iter()
            .filter_map(|(_, b, _)| *b)
            .filter(|b| b.is_finite() && *b > 0.0)
            .fold(0.0f64, f64::max);
        let max_bits = if max_bits > 0.0 { max_bits } else { 1.0 };
        let n = entries.len();
        let band = TIMELINE_BAND.1 - TIMELINE_BAND.0;
        let (first, last) = match (entries.first(), entries.last()) {
            (Some(a), Some(b)) => (a.0, b.0),
            _ => (0, 0),
        };
        let span = last.saturating_sub(first);
        let axis = entries
            .iter()
            .map(|(tick, _, _)| {
                let x = if span == 0 {
                    (BAR_X0 + BAR_X1) / 2.0
                } else {
                    BAR_X0 + ((tick - first) as f64 / span as f64) * (BAR_X1 - BAR_X0)
                };
                (*tick, x)
            })
            .collect();
        let rows = entries
            .iter()
            .enumerate()
            .map(|(i, (tick, bits, mark))| {
                let y0 = TIMELINE_BAND.0 + band * (i as f64 / n as f64);
                let y1 = TIMELINE_BAND.0 + band * ((i + 1) as f64 / n as f64);
                let mid = f64::midpoint(y0, y1);
                RowGeometry {
                    tick: *tick,
                    decisive: *mark,
                    y0,
                    y1,
                    bar: bits
                        .filter(|b| b.is_finite())
                        .map(|b| (BAR_X0, bar_x(b, max_bits))),
                    marker: mark.then_some((MARKER_X, mid)),
                }
            })
            .collect();
        Self {
            max_bits,
            rows,
            axis,
        }
    }
}

/// Row tick under normalized widget coordinates `(x, y)`. Only the timeline band is clickable.
///
/// Shared hit test of the tick-row panels (timeline and emergence).
pub fn row_tick_at(layout: &TimelineLayout, x: f64, y: f64) -> Option<u64> {
    if !(0.0..=1.0).contains(&x) || !(TIMELINE_BAND.0..=TIMELINE_BAND.1).contains(&y) {
        return None;
    }
    let n = layout.rows.len();
    if n == 0 {
        return None;
    }
    let band = TIMELINE_BAND.1 - TIMELINE_BAND.0;
    let i = (((y - TIMELINE_BAND.0) / band) * n as f64).floor();
    let i = (i as usize).min(n - 1);
    Some(layout.rows[i].tick)
}

/// One drawable mark of the timeline (pure; the egui painter maps these to shapes).
#[derive(Debug, Clone, PartialEq)]
pub enum TimelineMark {
    /// Selection highlight over a row band.
    Selection {
        /// Left edge.
        x0: f64,
        /// Top edge.
        y0: f64,
        /// Right edge.
        x1: f64,
        /// Bottom edge.
        y1: f64,
    },
    /// Tick-axis graduation.
    AxisTick {
        /// Tick value.
        tick: u64,
        /// Normalized x.
        x: f64,
        /// Top of the graduation.
        y0: f64,
        /// Bottom of the graduation.
        y1: f64,
    },
    /// Control-bit magnitude bar of one row.
    ControlBar {
        /// Row tick.
        tick: u64,
        /// Bar start (normalized x).
        x0: f64,
        /// Bar end (normalized x).
        x1: f64,
        /// Bar centre (normalized y).
        y: f64,
        /// Bar height (normalized).
        height: f64,
    },
    /// Decisive-moment marker.
    DecisiveMarker {
        /// Row tick.
        tick: u64,
        /// Centre x.
        x: f64,
        /// Centre y.
        y: f64,
    },
    /// Row text (same text as [`crate::decisive_timeline::timeline_lines`]).
    RowLabel {
        /// Row tick.
        tick: u64,
        /// Normalized x.
        x: f64,
        /// Normalized y.
        y: f64,
        /// Label text.
        text: String,
    },
}

/// Result of selecting a row.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionOutcome {
    /// Decisive row; the counterfactual debugger opened at `tick`.
    Opened {
        /// Selected tick.
        tick: u64,
    },
    /// Non-decisive row: selected, but the debugger is not opened.
    NotDecisive {
        /// Selected tick.
        tick: u64,
    },
    /// The tick is not part of the timeline.
    UnknownTick {
        /// Requested tick.
        tick: u64,
    },
    /// Decisive row, but the checkpoint was insufficient; the typed error is stored.
    Rejected {
        /// Selected tick.
        tick: u64,
        /// Why the debugger could not open.
        error: OpenAtError,
    },
}

/// Timeline widget state: rows, layout, selection and the opened counterfactual view.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineWidget {
    events: Vec<DecisiveEvent>,
    rows: Vec<TimelineRow>,
    layout: TimelineLayout,
    choice: CounterfactualChoice,
    selected: Option<u64>,
    view: Option<CounterfactualView>,
    error: Option<OpenAtError>,
}

impl TimelineWidget {
    /// Builds the widget from the recording (read-only) and the decisive events.
    pub fn new(rec: &Recording, events: Vec<DecisiveEvent>, choice: CounterfactualChoice) -> Self {
        let rows = timeline_rows(rec, &events);
        let layout = TimelineLayout::of(&rows);
        Self {
            events,
            rows,
            layout,
            choice,
            selected: None,
            view: None,
            error: None,
        }
    }

    /// Rows aligned with the recording, in tick order.
    pub fn rows(&self) -> &[TimelineRow] {
        &self.rows
    }

    /// Normalized layout consumed by the painter.
    pub const fn layout(&self) -> &TimelineLayout {
        &self.layout
    }

    /// Counterfactual choice used when a decisive row is selected.
    pub const fn choice(&self) -> &CounterfactualChoice {
        &self.choice
    }

    /// Currently selected tick, if any.
    pub const fn selected(&self) -> Option<u64> {
        self.selected
    }

    /// Opened counterfactual view, if the last selection was decisive and the checkpoint
    /// accepted it.
    pub const fn view(&self) -> Option<&CounterfactualView> {
        self.view.as_ref()
    }

    /// Typed error of the last decisive selection, if the checkpoint was insufficient.
    pub const fn error(&self) -> Option<&OpenAtError> {
        self.error.as_ref()
    }

    /// Row hit-testing in normalized widget coordinates. Only the timeline band is clickable.
    pub fn tick_at(&self, x: f64, y: f64) -> Option<u64> {
        row_tick_at(&self.layout, x, y)
    }

    /// Selects `tick`. Decisive rows call `open_counterfactual_at`; non-decisive rows only
    /// move the selection. The recording is never written.
    pub fn select_tick(&mut self, rec: &Recording, tick: u64) -> SelectionOutcome {
        let Some(row) = self.rows.iter().find(|r| r.tick == tick).cloned() else {
            self.selected = None;
            self.view = None;
            self.error = None;
            return SelectionOutcome::UnknownTick { tick };
        };
        self.selected = Some(tick);
        self.view = None;
        self.error = None;
        if !row.decisive {
            return SelectionOutcome::NotDecisive { tick };
        }
        let Some(event) = self.events.iter().find(|e| e.tick == tick).cloned() else {
            return SelectionOutcome::UnknownTick { tick };
        };
        match open_counterfactual_at(
            rec,
            &event,
            self.choice.alternative.clone(),
            self.choice.horizon,
        ) {
            Ok(view) => {
                self.view = Some(view);
                SelectionOutcome::Opened { tick }
            }
            Err(error) => {
                self.error = Some(error.clone());
                SelectionOutcome::Rejected { tick, error }
            }
        }
    }

    /// Draw marks of the current state (selection highlight, axis, bars, markers, labels).
    pub fn marks(&self) -> Vec<TimelineMark> {
        let lines = timeline_lines(&self.rows);
        let mut out = Vec::new();
        for (i, g) in self.layout.rows.iter().enumerate() {
            if self.selected == Some(g.tick) {
                out.push(TimelineMark::Selection {
                    x0: 0.01,
                    y0: g.y0,
                    x1: 0.99,
                    y1: g.y1,
                });
            }
            if let Some((x0, x1)) = g.bar {
                out.push(TimelineMark::ControlBar {
                    tick: g.tick,
                    x0,
                    x1,
                    y: f64::midpoint(g.y0, g.y1),
                    height: (g.y1 - g.y0) * 0.55,
                });
            }
            if let Some((x, y)) = g.marker {
                out.push(TimelineMark::DecisiveMarker { tick: g.tick, x, y });
            }
            out.push(TimelineMark::RowLabel {
                tick: g.tick,
                x: 0.02,
                y: f64::midpoint(g.y0, g.y1),
                text: lines[i].clone(),
            });
        }
        for (tick, x) in &self.layout.axis {
            out.push(TimelineMark::AxisTick {
                tick: *tick,
                x: *x,
                y0: TIMELINE_BAND.1,
                y1: TIMELINE_BAND.1 + 0.015,
            });
        }
        out
    }

    /// Text of the detail area: row summary, counterfactual view (when opened) or the error.
    /// Non-decisive selections never show a counterfactual view.
    pub fn detail_lines(&self) -> Vec<String> {
        let decisive = self.rows.iter().filter(|r| r.decisive).count();
        let mut lines = vec![format!(
            "timeline: {} events, {} decisive, max {:.2} control bits",
            self.rows.len(),
            decisive,
            self.layout.max_bits
        )];
        if let Some(view) = &self.view {
            lines.extend(counterfactual_lines(view));
        }
        if let Some(error) = &self.error {
            lines.push(format!("counterfactual not opened: {error}"));
        }
        lines
    }
}

#[cfg(feature = "window")]
mod egui_impl {
    use eframe::egui;

    use super::{SelectionOutcome, TIMELINE_BAND, TimelineMark, TimelineWidget};
    use crate::replay::Recording;

    fn pos(rect: egui::Rect, x: f64, y: f64) -> egui::Pos2 {
        egui::pos2(
            rect.min.x + x as f32 * rect.width(),
            rect.min.y + y as f32 * rect.height(),
        )
    }

    /// Paints `marks` and the `detail` text into `rect`. This is the single renderer of the
    /// tick-row panels: the timeline paints its control bits and decisive marks through it and
    /// the emergence panel ([`crate::emergence_ui`]) paints hypothesis marks and evidence
    /// through it, so neither duplicates the painter.
    pub fn paint_timeline(
        painter: &egui::Painter,
        rect: egui::Rect,
        marks: &[TimelineMark],
        detail: &[String],
    ) {
        painter.rect_filled(rect, 0.0, egui::Color32::WHITE);
        let band = egui::Rect::from_min_max(
            pos(rect, 0.0, TIMELINE_BAND.0),
            pos(rect, 1.0, TIMELINE_BAND.1),
        );
        painter.rect_filled(band, 0.0, egui::Color32::from_rgb(0xf7, 0xf7, 0xf7));
        let blue = egui::Color32::from_rgb(0x1f, 0x77, 0xb4);
        let orange = egui::Color32::from_rgb(0xff, 0x7f, 0x0e);
        let grey = egui::Color32::from_rgb(0x66, 0x66, 0x66);
        for mark in marks {
            match mark {
                TimelineMark::Selection { x0, y0, x1, y1 } => {
                    painter.rect_filled(
                        egui::Rect::from_min_max(pos(rect, *x0, *y0), pos(rect, *x1, *y1)),
                        2.0,
                        egui::Color32::from_rgb(0xff, 0xf3, 0xcd),
                    );
                }
                TimelineMark::AxisTick { x, y0, y1, .. } => {
                    painter.line_segment(
                        [pos(rect, *x, *y0), pos(rect, *x, *y1)],
                        egui::Stroke::new(1.0, grey),
                    );
                }
                TimelineMark::ControlBar {
                    x0, x1, y, height, ..
                } => {
                    painter.rect_filled(
                        egui::Rect::from_min_max(
                            pos(rect, *x0, *y - height / 2.0),
                            pos(rect, *x1, *y + height / 2.0),
                        ),
                        1.0,
                        blue,
                    );
                }
                TimelineMark::DecisiveMarker { x, y, .. } => {
                    painter.circle_filled(pos(rect, *x, *y), 4.0, orange);
                }
                TimelineMark::RowLabel { x, y, text, .. } => {
                    painter.text(
                        pos(rect, *x, *y),
                        egui::Align2::LEFT_CENTER,
                        text,
                        egui::FontId::monospace(12.0),
                        egui::Color32::BLACK,
                    );
                }
            }
        }
        // Detail area: the caller's lines (counterfactual view / decision and comparison lines).
        let base = rect.min.y + TIMELINE_BAND.1 as f32 * rect.height() + 8.0;
        for (i, line) in detail.iter().enumerate() {
            painter.text(
                egui::pos2(rect.min.x + 6.0, base + i as f32 * 15.0),
                egui::Align2::LEFT_TOP,
                line,
                egui::FontId::monospace(11.0),
                egui::Color32::from_rgb(0x22, 0x22, 0x22),
            );
        }
    }

    /// Paints the timeline into `ui` (feature `window`) and applies a click on a row to the
    /// widget state machine. Returns the allocated response and the selection outcome, if the
    /// frame contained a click on a row.
    ///
    /// Headless-safe: it only uses [`egui::Painter`] and input events, so it can be driven by
    /// `egui::Context::run_ui` without a window or a GPU.
    pub fn timeline_ui(
        ui: &mut egui::Ui,
        widget: &mut TimelineWidget,
        rec: &Recording,
    ) -> (egui::Response, Option<SelectionOutcome>) {
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), egui::Sense::click());
        let painter = ui.painter_at(rect);
        paint_timeline(&painter, rect, &widget.marks(), &widget.detail_lines());
        let mut outcome = None;
        if response.clicked()
            && let Some(p) = response.interact_pointer_pos()
        {
            let nx = f64::from((p.x - rect.min.x) / rect.width());
            let ny = f64::from((p.y - rect.min.y) / rect.height());
            if let Some(tick) = widget.tick_at(nx, ny) {
                outcome = Some(widget.select_tick(rec, tick));
            }
        }
        (response, outcome)
    }
}

#[cfg(feature = "window")]
pub use egui_impl::{paint_timeline, timeline_ui};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual::RecordingId;
    use crate::recorder::record_grid;
    use gogamen_core::{
        BinSpec, BinningVersion, ControlBitsConfig, DecisionInput, DecisiveTimeline,
        DeterministicRng, DiscreteAction, FmcPlanner, FmcPlannerConfig, FmcScoringConfig,
        GenerativeModel, MasterSeed, PlanBudget, PlanRequest, PlanningContext, RngSource,
        StreamKey, control_bits_from_walkers,
    };
    use gogamen_lab_grid::world::GRID_ACTION_CARDINALITY;
    use gogamen_lab_grid::{GridMap, GridWorldConfig, GridWorldModel};

    const SEED: u64 = 42;
    const TICKS: u64 = 16;
    const TOP_QUANTILE: f64 = 0.25;

    /// World used by the real FMC run: a corridor with one hazard, so some root actions are
    /// unviable and the viability spread is not identically zero.
    fn map() -> GridMap {
        GridMap::from_ascii(&[
            "#########",
            "#.......#",
            "#..H....#",
            "#.......#",
            "#S.....G#",
            "#.......#",
            "#########",
        ])
        .expect("fixture map")
    }

    /// Real simulation fixture: records a Grid rollout with the engine policy, then replays
    /// the same policy on the real `GridWorldModel` to recover the recorded states and runs a
    /// real `FmcPlanner` at every one of them. The returned decisions are therefore genuine
    /// FMC decisions taken at the recording's ticks (no hand-made spreads or control bits).
    fn fmc_decisions(rec: &Recording) -> Vec<DecisionInput> {
        let config = GridWorldConfig {
            map: rec.map().clone(),
            ..GridWorldConfig::default()
        };
        let mut model =
            GridWorldModel::new(config.clone(), MasterSeed::new(rec.seed())).expect("grid model");
        let mut policy = MasterSeed::new(rec.seed()).derive(StreamKey::Episode(0));
        let planner = FmcPlanner::new(
            FmcPlannerConfig::new(
                48,
                8,
                config.dt,
                FmcScoringConfig::default(),
                MasterSeed::new(rec.seed() ^ 0xF3C),
            )
            .expect("fmc config"),
        )
        .with_walker_capture(true);
        let scale = BinSpec::uniform(
            "gog202-grid-timeline",
            BinningVersion::new(1, 0),
            vec![
                (0.0, rec.map().width() as f64, rec.map().width()),
                (0.0, rec.map().height() as f64, rec.map().height()),
                (0.0, 20.0, 20),
                (-1.0, 5.0, 6),
                (0.0, 5.0, 5),
            ],
        )
        .expect("bin spec");
        let cb_cfg = ControlBitsConfig {
            resamples: 20,
            ..ControlBitsConfig::default()
        };
        let mut rng = DeterministicRng::new(rec.seed() ^ 0x5EED);
        let mut out = Vec::new();
        let last = rec.last_tick();
        for tick in 0..=last {
            let state = model.state().clone();
            let request = PlanRequest::from_state(
                state.clone(),
                model.action_space().clone(),
                PlanBudget::with_transitions(100_000),
            );
            let (result, trace) = {
                let mut ctx = PlanningContext::new(&model, &mut rng);
                planner
                    .plan_with_context(&mut ctx, &request)
                    .expect("fmc plan")
            };
            let chosen = result
                .authorized_action()
                .and_then(Action::as_discrete)
                .map(DiscreteAction::index);
            let viability = trace.decision_viability().cloned().expect("viability");
            let control_bits = trace.final_walkers().and_then(|w| {
                let (a, s): (Vec<_>, Vec<_>) = w.iter().cloned().unzip();
                control_bits_from_walkers(&a, &s, &scale, &cb_cfg)
                    .ok()
                    .map(|e| e.control_corrected)
            });
            if let Some(chosen) = chosen {
                out.push(DecisionInput {
                    tick,
                    state: state.values().to_vec(),
                    chosen,
                    viability,
                    control_bits,
                });
            }
            if tick == last {
                break;
            }
            let idx = (policy.next_u64() % GRID_ACTION_CARDINALITY as u64) as usize;
            let action = Action::Discrete(
                DiscreteAction::new(idx, GRID_ACTION_CARDINALITY).expect("policy action"),
            );
            if model.step_in_place(&action).is_err() {
                break;
            }
        }
        // The replayed model must be exactly the recorded simulation, otherwise the FMC
        // decisions would not belong to the ticks they are placed at.
        let end = rec.frame_at(rec.last_tick()).expect("last frame");
        assert_eq!(model.position(), end.agent, "fixture replay diverged");
        assert_eq!(model.total_reward(), end.total_reward);
        assert!(!out.is_empty());
        out
    }

    struct Fixture {
        rec: Recording,
        events: Vec<DecisiveEvent>,
        decisive_ticks: Vec<u64>,
        non_decisive_ticks: Vec<u64>,
    }

    fn fixture() -> Fixture {
        let rec = record_grid(SEED, map(), TICKS).expect("record");
        let timeline =
            DecisiveTimeline::build(fmc_decisions(&rec), TOP_QUANTILE).expect("decisive timeline");
        let events = timeline.events();
        let decisive_ticks = timeline
            .decisive_samples()
            .map(|s| s.input.tick)
            .collect::<Vec<_>>();
        let non_decisive_ticks = events
            .iter()
            .filter(|e| !e.decisive)
            .map(|e| e.tick)
            .collect();
        Fixture {
            rec,
            events,
            decisive_ticks,
            non_decisive_ticks,
        }
    }

    fn choice() -> CounterfactualChoice {
        CounterfactualChoice {
            alternative: Action::Discrete(
                DiscreteAction::new(1, GRID_ACTION_CARDINALITY).expect("alt action"),
            ),
            horizon: 4,
        }
    }

    #[test]
    fn timeline_widget_layout_draws_control_bits_and_decisive_moments() {
        let f = fixture();
        assert!(
            !f.decisive_ticks.is_empty(),
            "the real FMC run must produce decisive moments"
        );
        assert!(
            f.events.iter().any(|e| e.control_bits.is_some()),
            "the real FMC run must produce control bits"
        );

        let widget = TimelineWidget::new(&f.rec, f.events.clone(), choice());
        eprintln!(
            "GOG-202 fixture: rows={} decisive={:?} events={:?}",
            widget.rows().len(),
            f.decisive_ticks,
            f.events
        );
        assert_eq!(widget.rows().len(), f.events.len());
        assert_eq!(widget.rows().len(), widget.layout().rows.len());
        assert_eq!(widget.layout().axis.len(), widget.rows().len());

        let marks = widget.marks();
        let bars = marks
            .iter()
            .filter(|m| matches!(m, TimelineMark::ControlBar { .. }))
            .count();
        let expected_bars = f.events.iter().filter(|e| e.control_bits.is_some()).count();
        assert_eq!(bars, expected_bars, "one bar per control-bits event");
        let markers = marks
            .iter()
            .filter(|m| matches!(m, TimelineMark::DecisiveMarker { .. }))
            .count();
        assert_eq!(
            markers,
            f.decisive_ticks.len(),
            "one marker per decisive event"
        );
        assert_eq!(
            marks
                .iter()
                .filter(|m| matches!(m, TimelineMark::AxisTick { .. }))
                .count(),
            widget.rows().len(),
            "the tick axis covers every row"
        );
        // Labels come from the data layer, so the decisive text is drawn too.
        let labels: Vec<&str> = marks
            .iter()
            .filter_map(|m| match m {
                TimelineMark::RowLabel { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(labels, timeline_lines(widget.rows()));
        assert!(
            labels.iter().any(|l| l.ends_with("DECISIVE")),
            "no decisive label among {labels:?}"
        );

        // Bars are proportional to the control bits of their row.
        let bar_of = |tick: u64| {
            marks.iter().find_map(|m| match m {
                TimelineMark::ControlBar {
                    tick: t, x0, x1, ..
                } if *t == tick => Some((*x0, *x1)),
                _ => None,
            })
        };
        let with_bits: Vec<(u64, f64)> = widget
            .rows()
            .iter()
            .filter_map(|r| r.control_bits.map(|b| (r.tick, b)))
            .collect();
        for (tick, bits) in &with_bits {
            let (x0, x1) = bar_of(*tick).expect("bar");
            let expected = BAR_X0 + (bits / widget.layout().max_bits) * (BAR_X1 - BAR_X0);
            assert!((x0 - BAR_X0).abs() < f64::EPSILON);
            assert!(
                (x1 - expected).abs() < 1e-9,
                "tick {tick} bar {x1} != {expected}"
            );
        }
    }

    #[test]
    fn timeline_widget_selects_decisive_and_opens_counterfactual_without_altering_recording() {
        let f = fixture();
        let tick = f.decisive_ticks[0];
        let non_decisive = *f.non_decisive_ticks.first().expect("non-decisive row");
        let bytes = f.rec.to_bytes();
        let id = RecordingId::of(&f.rec);

        let mut widget = TimelineWidget::new(&f.rec, f.events.clone(), choice());
        assert_eq!(widget.selected(), None);
        assert!(widget.view().is_none());

        // A non-decisive row is selected but never opens the debugger.
        assert_eq!(
            widget.select_tick(&f.rec, non_decisive),
            SelectionOutcome::NotDecisive { tick: non_decisive }
        );
        assert_eq!(widget.selected(), Some(non_decisive));
        assert!(widget.view().is_none());
        assert!(
            !widget
                .detail_lines()
                .iter()
                .any(|l| l.contains("PREDICTED"))
        );

        // The decisive row opens the existing counterfactual debugger.
        assert_eq!(
            widget.select_tick(&f.rec, tick),
            SelectionOutcome::Opened { tick }
        );
        assert_eq!(widget.selected(), Some(tick));
        let view = widget.view().expect("counterfactual view");
        assert_eq!(view.tick, tick);
        assert_eq!(view.alternative.header.tick, tick);
        assert!(widget.error().is_none());
        let details = widget.detail_lines();
        for needle in ["PREDICTED alternative", "OBSERVED (recording)"] {
            assert!(
                details.iter().any(|l| l.contains(needle)),
                "missing {needle}: {details:?}"
            );
        }

        // Unknown tick resets the selection.
        assert_eq!(
            widget.select_tick(&f.rec, 9_999),
            SelectionOutcome::UnknownTick { tick: 9_999 }
        );
        assert!(widget.selected().is_none() && widget.view().is_none());

        // Reading and opening left the simulation bit-identical.
        assert_eq!(f.rec.to_bytes(), bytes, "recording bytes changed");
        assert_eq!(RecordingId::of(&f.rec), id, "recording hash changed");
    }

    #[cfg(feature = "window")]
    #[test]
    fn timeline_widget_egui_headless_paints_and_opens_on_click() {
        use eframe::egui;

        let f = fixture();
        let tick = f.decisive_ticks[0];
        let bytes = f.rec.to_bytes();
        let id = RecordingId::of(&f.rec);
        let mut widget = TimelineWidget::new(&f.rec, f.events.clone(), choice());
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(720.0, 420.0));
        let input = |events| egui::RawInput {
            screen_rect: Some(screen),
            events,
            ..Default::default()
        };
        let run = |events: Vec<egui::Event>,
                   widget: &mut TimelineWidget|
         -> (
            egui::Rect,
            Vec<String>,
            usize,
            usize,
            Option<SelectionOutcome>,
        ) {
            let mut seen = None;
            let mut allocated = None;
            let out = ctx.run_ui(input(events), |ui| {
                let (resp, outcome) = timeline_ui(ui, widget, &f.rec);
                allocated = Some(resp.rect);
                seen = outcome;
            });
            let mut texts = Vec::new();
            let mut filled = 0usize;
            let mut circles = 0usize;
            for clipped in &out.shapes {
                match &clipped.shape {
                    egui::Shape::Text(t) => texts.push(t.galley.text().to_owned()),
                    egui::Shape::Rect(r) if r.fill != egui::Color32::TRANSPARENT => filled += 1,
                    egui::Shape::Circle(_) => circles += 1,
                    _ => {}
                }
            }
            out.drop_without_applying_deltas();
            (allocated.expect("allocation"), texts, filled, circles, seen)
        };
        // One paint frame: shapes for the rows, bars, markers, labels and details.
        let (rect, texts, filled, circles, outcome) = run(Vec::new(), &mut widget);
        assert!(outcome.is_none());
        assert!(filled >= widget.rows().len(), "no bars painted");
        assert_eq!(
            circles,
            f.decisive_ticks.len(),
            "one painted marker per decisive moment"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("t=") && t.ends_with("DECISIVE")),
            "no decisive row label: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.starts_with("timeline: ")),
            "no timeline summary: {texts:?}"
        );

        // Click the decisive row: pointer move, press and release over the row band.
        let row = widget
            .layout()
            .rows
            .iter()
            .find(|r| r.tick == tick)
            .copied()
            .expect("decisive row geometry");
        let p = egui::pos2(
            rect.min.x + 0.5 * rect.width(),
            rect.min.y + (f64::midpoint(row.y0, row.y1) as f32) * rect.height(),
        );
        let button = |pressed| egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::default(),
        };
        assert!(
            run(vec![egui::Event::PointerMoved(p)], &mut widget)
                .4
                .is_none()
        );
        assert!(run(vec![button(true)], &mut widget).4.is_none());
        let outcome = run(vec![button(false)], &mut widget).4;
        assert_eq!(outcome, Some(SelectionOutcome::Opened { tick }));
        assert_eq!(
            widget.selected(),
            Some(tick),
            "click did not select the row"
        );
        assert!(widget.view().is_some(), "click did not open the debugger");
        // The next frame paints the opened counterfactual view.
        let (_, texts, _, _, outcome) = run(Vec::new(), &mut widget);
        assert!(outcome.is_none());
        assert!(
            texts.iter().any(|t| t.contains("PREDICTED alternative")),
            "counterfactual view not painted: {texts:?}"
        );
        // ... and the recording is untouched.
        assert_eq!(f.rec.to_bytes(), bytes);
        assert_eq!(RecordingId::of(&f.rec), id);
    }
}
