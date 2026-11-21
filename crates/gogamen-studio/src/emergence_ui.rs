//! Emergence panel widget (GOG-205): draws [`DecisionRecord`] lanes as a tick-row panel.
//!
//! This module adds no renderer and no second data layer:
//!
//! - the data comes from [`crate::emergence_view`] (`DecisionView::lines`,
//!   `compare_policies` / `PolicyComparison::lines`);
//! - the geometry is the shared [`TimelineLayout`] of [`crate::timeline_widget`]
//!   (`of_marks`: one row per decision, evidence bars, hypothesis markers, tick axis) and the
//!   hit test is [`row_tick_at`];
//! - the drawing is [`crate::timeline_widget::paint_timeline`], the same painter the timeline
//!   uses, fed with [`TimelineMark`]s and the detail lines.
//!
//! Every row is one [`DecisionRecord`] (its policy, tick and evidence). Selecting a row rebuilds
//! that decision's [`DecisionView`] against the borrowed [`Recording`] (real lane up to the
//! decision tick, hypothetical lane from the record) and paints its lines plus the comparison
//! lines when exactly two policies are loaded. The recording is never written.
//!
//! Without the `window` feature the layout, marks and selection state machine are still
//! compiled and tested; [`emergence_ui`] is the feature-gated painter.

use crate::cognitive::Known;
use crate::emergence_view::{
    DecisionRecord, DecisionView, PolicyComparison, compare_policies, decision_view,
};
use crate::replay::Recording;
use crate::timeline_widget::{TIMELINE_BAND, TimelineLayout, TimelineMark, row_tick_at};

/// Result of selecting a row of the emergence panel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergenceSelection {
    /// The decision at `tick` was selected; its view was rebuilt from the recording.
    Opened {
        /// Selected tick.
        tick: u64,
    },
    /// No decision at that tick; the selection was cleared.
    UnknownTick {
        /// Requested tick.
        tick: u64,
    },
}

/// One row per decision record; panel state is the selection plus the opened view.
#[derive(Debug, Clone, PartialEq)]
pub struct EmergenceWidget {
    /// Records sorted by tick (the layout has one row per record, in this order).
    records: Vec<DecisionRecord>,
    layout: TimelineLayout,
    selected: Option<u64>,
    view: Option<DecisionView>,
    comparison: Option<PolicyComparison>,
}

/// Bar magnitude of a record: the evidence count of its hypothesis, if recorded.
fn evidence(record: &DecisionRecord) -> Option<f64> {
    match &record.hypothesis {
        Known::Known(h) => match h.evidence_count {
            Known::Known(n) => Some(n as f64),
            Known::Unknown => None,
        },
        Known::Unknown => None,
    }
}

impl EmergenceWidget {
    /// Builds the panel over `records` (sorted by tick, stable). When exactly two policies are
    /// loaded, their decisions are also compared field by field through `compare_policies`.
    pub fn new(mut records: Vec<DecisionRecord>) -> Self {
        records.sort_by_key(|r| r.tick);
        let layout = TimelineLayout::of_marks(
            records
                .iter()
                .map(|r| (r.tick, evidence(r), matches!(r.hypothesis, Known::Known(_)))),
        );
        let comparison = match records.as_slice() {
            [a, b] => Some(compare_policies(a, b)),
            _ => None,
        };
        Self {
            records,
            layout,
            selected: None,
            view: None,
            comparison,
        }
    }

    /// Records in drawing order.
    pub fn records(&self) -> &[DecisionRecord] {
        &self.records
    }

    /// Shared tick-row layout (rows, evidence bars, hypothesis markers, tick axis).
    pub const fn layout(&self) -> &TimelineLayout {
        &self.layout
    }

    /// Comparison of the two loaded policies, if there are exactly two.
    pub const fn comparison(&self) -> Option<&PolicyComparison> {
        self.comparison.as_ref()
    }

    /// Currently selected tick, if any.
    pub const fn selected(&self) -> Option<u64> {
        self.selected
    }

    /// Opened decision view of the selected tick, if any.
    pub const fn view(&self) -> Option<&DecisionView> {
        self.view.as_ref()
    }

    /// Row hit-testing in normalized widget coordinates (shared band of the tick-row panels).
    pub fn tick_at(&self, x: f64, y: f64) -> Option<u64> {
        row_tick_at(&self.layout, x, y)
    }

    /// Selects `tick` and rebuilds its [`DecisionView`] from the recording (read-only).
    /// Selecting a tick without a decision clears the selection.
    pub fn select_tick(&mut self, rec: &Recording, tick: u64) -> EmergenceSelection {
        match self.records.iter().find(|r| r.tick == tick) {
            Some(record) => {
                self.selected = Some(tick);
                self.view = Some(decision_view(rec, record));
                EmergenceSelection::Opened { tick }
            }
            None => {
                self.selected = None;
                self.view = None;
                EmergenceSelection::UnknownTick { tick }
            }
        }
    }

    /// Row label of record `index`: `t={tick} {policy} hypothesis|no-hypothesis predicates=N`.
    fn row_label(&self, index: usize) -> String {
        let record = &self.records[index];
        let hypothesis = if matches!(record.hypothesis, Known::Known(_)) {
            "hypothesis"
        } else {
            "no-hypothesis"
        };
        format!(
            "t={} {} {} predicates={}",
            record.tick,
            record.policy,
            hypothesis,
            record.predicates.len()
        )
    }

    /// Draw marks: selection highlight, evidence bars, hypothesis markers, row labels and axis.
    pub fn marks(&self) -> Vec<TimelineMark> {
        let mut out = Vec::new();
        for (index, g) in self.layout.rows.iter().enumerate() {
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
                text: self.row_label(index),
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

    /// Detail text: the panel summary, then `DecisionView::lines` of the selection and the
    /// `PolicyComparison::lines` of the two policies (when present).
    pub fn detail_lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "emergence: {} decisions, {} with hypothesis, {} declared predicates",
            self.records.len(),
            self.records
                .iter()
                .filter(|r| matches!(r.hypothesis, Known::Known(_)))
                .count(),
            self.records
                .iter()
                .map(|r| r.predicates.len())
                .sum::<usize>()
        )];
        if let Some(view) = &self.view {
            out.extend(view.lines());
        }
        if let Some(comparison) = &self.comparison {
            out.extend(comparison.lines());
        }
        out
    }
}

#[cfg(feature = "window")]
mod egui_impl {
    use eframe::egui;

    use super::{EmergenceSelection, EmergenceWidget};
    use crate::replay::Recording;
    use crate::timeline_widget::paint_timeline;

    /// Paints the emergence panel into `ui` (feature `window`) through the shared
    /// [`paint_timeline`] renderer and forwards a click on a row to the selection state machine.
    ///
    /// Headless-safe: it only uses [`egui::Painter`] and input events, so it can be driven by
    /// `egui::Context::run_ui` without a window or a GPU.
    pub fn emergence_ui(
        ui: &mut egui::Ui,
        widget: &mut EmergenceWidget,
        rec: &Recording,
    ) -> (egui::Response, Option<EmergenceSelection>) {
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
pub use egui_impl::emergence_ui;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual::RecordingId;
    use crate::recorder::record_grid;
    use gogamen_core::causal_revision::{CausalRevisor, RevisionConfig};
    use gogamen_core::conditional_effects::ConditionalObservation;
    use gogamen_core::predicate_invention::{InventionConfig, PredicateInventor};
    use gogamen_core::{EvidenceRef, ObservationId};
    use gogamen_lab_grid::GridMap;

    use crate::timeline_widget::{BAR_X0, BAR_X1, MARKER_X};

    const SEED: u64 = 42;
    const TICKS: u64 = 8;

    fn recording() -> Recording {
        record_grid(SEED, GridMap::cul_de_sac_map(), TICKS).expect("record")
    }

    fn obs(id: u64, state: [bool; 3], moved: bool) -> ConditionalObservation {
        ConditionalObservation {
            id: ObservationId::new(id),
            evidence: EvidenceRef::new(9000 + id),
            state: state.to_vec(),
            action: 0,
            next: vec![moved, false, false],
        }
    }

    /// Real causal fixture: bit1 is the "blocked" context (mirrors the core crate fixture).
    fn causal_records(tick: u64) -> DecisionRecord {
        let train: Vec<_> = (0..60u64)
            .map(|i| obs(1 + i, [(i / 2) % 2 == 1, i % 2 == 1, false], i % 2 == 0))
            .collect();
        let hold: Vec<_> = (0..40u64)
            .map(|i| obs(1000 + i, [false, i % 2 == 1, false], i % 2 == 0))
            .collect();
        let mut revisor = CausalRevisor::new(3, 1, RevisionConfig::default()).unwrap();
        revisor.revise(0, 0, &train, &hold).unwrap();
        DecisionRecord::from_causal_revision("causal", tick, &revisor, (0, 0))
    }

    /// Deterministic LCG bit source (same construction as the invention crate fixtures).
    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    fn invention_data(start: u64, n: u64, mut seed: u64) -> Vec<ConditionalObservation> {
        (0..n)
            .map(|i| {
                let r = lcg(&mut seed);
                let state: Vec<bool> = (0..4).map(|k| (r >> k) & 1 == 1).collect();
                let y = state[0] ^ state[1];
                ConditionalObservation {
                    id: ObservationId::new(start + i),
                    evidence: EvidenceRef::new(7000 + start + i),
                    state,
                    action: 0,
                    next: vec![y, false, false, false],
                }
            })
            .collect()
    }

    /// Real invention fixture: hidden XOR context, accepted and adopted by the inventor.
    fn invention_record(tick: u64) -> DecisionRecord {
        let outcome =
            PredicateInventor::new(4, 1, InventionConfig::default(), RevisionConfig::default())
                .unwrap()
                .invent(
                    0,
                    0,
                    &invention_data(1, 240, 1),
                    &invention_data(10_000, 120, 2),
                )
                .unwrap();
        assert!(outcome.report.adopted, "{}", outcome.report.render());
        DecisionRecord::from_predicate_invention("invention", tick, &outcome, (0, 0))
    }

    fn records() -> Vec<DecisionRecord> {
        vec![causal_records(2), invention_record(5)]
    }

    #[test]
    fn emergence_ui_layout_draws_hypothesis_rows_evidence_and_ticks() {
        let records = records();
        let widget = EmergenceWidget::new(records.clone());
        assert_eq!(widget.records(), records.as_slice());
        assert_eq!(widget.layout().rows.len(), 2);
        assert_eq!(widget.layout().axis.len(), 2);
        // Evidence bars come from the records' hypothesis evidence (60 and 240 samples).
        assert_eq!(widget.layout().max_bits, 240.0);
        assert_eq!(
            widget.layout().rows[0].bar,
            Some((BAR_X0, BAR_X0 + (60.0 / 240.0) * (BAR_X1 - BAR_X0)))
        );
        assert_eq!(widget.layout().rows[1].bar, Some((BAR_X0, BAR_X1)));
        // Both records carry a hypothesis, so both rows are marked.
        assert!(widget.layout().rows.iter().all(|r| r.marker.is_some()));
        assert_eq!(widget.layout().rows[0].marker.map(|m| m.0), Some(MARKER_X));

        let marks = widget.marks();
        let labels: Vec<&str> = marks
            .iter()
            .filter_map(|m| match m {
                TimelineMark::RowLabel { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(labels.len(), 2);
        assert!(labels[0].starts_with("t=2 causal hypothesis predicates="));
        assert!(labels[1].starts_with("t=5 invention hypothesis predicates=1"));
        assert_eq!(
            marks
                .iter()
                .filter(|m| matches!(m, TimelineMark::AxisTick { .. }))
                .count(),
            2
        );
        // Comparison of the two policies is available and honest about its scope.
        let comparison = widget.comparison().expect("two policies are compared");
        assert_eq!(
            (comparison.policy_a.as_str(), comparison.policy_b.as_str()),
            ("causal", "invention")
        );
        assert!(comparison.lines()[0].contains("no intentions inferred"));
        assert!(widget.selected().is_none() && widget.view().is_none());
        // Panel summary is the first detail line; no view lines before a selection.
        let details = widget.detail_lines();
        assert_eq!(
            details[0],
            "emergence: 2 decisions, 2 with hypothesis, 3 declared predicates"
        );
        assert!(!details.iter().any(|l| l.starts_with("policy=")));
    }

    #[test]
    fn emergence_ui_selects_record_and_opens_its_view_without_altering_recording() {
        let rec = recording();
        let bytes = rec.to_bytes();
        let id = RecordingId::of(&rec);
        let mut widget = EmergenceWidget::new(records());

        assert_eq!(
            widget.select_tick(&rec, 2),
            EmergenceSelection::Opened { tick: 2 }
        );
        assert_eq!(widget.selected(), Some(2));
        let view = widget.view().expect("decision view");
        assert_eq!(view.record.policy, "causal");
        assert!(matches!(view.record.scale_label, Known::Unknown));
        assert!(!view.real_lane.is_empty());
        assert!(view.real_lane.iter().all(|s| s.tick <= 2));

        let details = widget.detail_lines();
        assert!(
            details.iter().any(|l| l == "policy=causal tick=2"),
            "{details:?}"
        );
        assert!(
            details
                .iter()
                .any(|l| l.starts_with("hypothesis=causal rule next[b0]|a0:")),
            "{details:?}"
        );
        assert!(details.iter().any(|l| l == "predicates=b1=false; b1=true"
            || l == "predicates=b1=true; b1=false"));
        assert!(
            details
                .iter()
                .any(|l| l.starts_with("compare causal vs invention")),
            "{details:?}"
        );

        // Selecting the invention row replaces the view with its own predicates/model.
        assert_eq!(
            widget.select_tick(&rec, 5),
            EmergenceSelection::Opened { tick: 5 }
        );
        let details = widget.detail_lines();
        assert!(details.iter().any(|l| l == "policy=invention tick=5"));
        assert!(details.iter().any(|l| l == "predicates=Xor(b0,b1)"));
        assert!(
            details
                .iter()
                .any(|l| l.contains("model=rev") && l.contains("unknown"))
        );

        // Unknown tick clears the selection and the view.
        assert_eq!(
            widget.select_tick(&rec, 7),
            EmergenceSelection::UnknownTick { tick: 7 }
        );
        assert!(widget.selected().is_none() && widget.view().is_none());

        // Viewing only borrowed the recording.
        assert_eq!(rec.to_bytes(), bytes, "recording bytes changed");
        assert_eq!(RecordingId::of(&rec), id, "recording hash changed");
    }

    /// Row hit-testing maps the band to the record ticks (outside the band: nothing).
    #[test]
    fn emergence_ui_hit_tests_rows_inside_the_shared_band() {
        let widget = EmergenceWidget::new(records());
        let mid = |y: f64| {
            f64::midpoint(
                widget.layout().rows[y as usize].y0,
                widget.layout().rows[y as usize].y1,
            )
        };
        assert_eq!(widget.tick_at(0.5, mid(0.0)), Some(2));
        assert_eq!(widget.tick_at(0.5, mid(1.0)), Some(5));
        assert_eq!(widget.tick_at(0.5, 0.0), None);
        assert_eq!(widget.tick_at(1.5, mid(0.0)), None);
    }

    #[cfg(feature = "window")]
    #[test]
    fn emergence_ui_egui_headless_paints_lines_and_selects_on_click() {
        use eframe::egui;

        let rec = recording();
        let bytes = rec.to_bytes();
        let id = RecordingId::of(&rec);
        let mut widget = EmergenceWidget::new(records());
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(720.0, 420.0));
        let run = |events: Vec<egui::Event>,
                   widget: &mut EmergenceWidget|
         -> (
            egui::Rect,
            Vec<String>,
            usize,
            usize,
            Option<EmergenceSelection>,
        ) {
            let mut seen = None;
            let mut allocated = None;
            let out = ctx.run_ui(
                egui::RawInput {
                    screen_rect: Some(screen),
                    events,
                    ..Default::default()
                },
                |ui| {
                    let (response, outcome) = emergence_ui(ui, widget, &rec);
                    allocated = Some(response.rect);
                    seen = outcome;
                },
            );
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

        // One paint frame: summary, comparison and row labels are drawn through the shared painter.
        let (rect, texts, filled, circles, outcome) = run(Vec::new(), &mut widget);
        assert!(outcome.is_none());
        assert!(
            filled >= 4,
            "background, band and two evidence bars expected: {filled}"
        );
        assert_eq!(circles, 2, "one marker per hypothesis row");
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("emergence: 2 decisions"))
        );
        assert!(texts.iter().any(|t| t.starts_with("t=2 causal hypothesis")));
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("compare causal vs invention")),
            "comparison lines missing: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.starts_with("policy=causal")),
            "no view before selection"
        );

        // Click the causal row (row 0): pointer move, press and release over the band.
        let row = widget.layout().rows[0];
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
        assert_eq!(
            run(vec![button(false)], &mut widget).4,
            Some(EmergenceSelection::Opened { tick: 2 })
        );
        assert_eq!(widget.selected(), Some(2));

        // The next frame paints the opened `DecisionView::lines` of the selected record.
        let (_, texts, _, _, outcome) = run(Vec::new(), &mut widget);
        assert!(outcome.is_none());
        assert!(
            texts.iter().any(|t| t == "policy=causal tick=2"),
            "{texts:?}"
        );
        assert!(
            texts
                .iter()
                .any(|t| t.starts_with("hypothesis=causal rule next[b0]|a0:")),
            "{texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.starts_with("[real] t=0")),
            "{texts:?}"
        );
        // ... and the recording is untouched.
        assert_eq!(rec.to_bytes(), bytes);
        assert_eq!(RecordingId::of(&rec), id);
    }
}
