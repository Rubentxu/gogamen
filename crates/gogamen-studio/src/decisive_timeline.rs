//! Timeline rows for control bits and decisive moments (GOG-181, data slice).
//!
//! Studio reads [`DecisiveEvent`]s exported by `gogamen_core::DecisiveTimeline` and lines them
//! up with a disconnected [`Recording`]. Everything here is read-only: the recording is only
//! borrowed, and opening the counterfactual debugger at a decisive tick reuses
//! [`CounterfactualDebugger::compare`], which runs isolated branches from a checkpoint.
//!
//! The visual widget (drawing these rows in an egui timeline) is NOT implemented here; this
//! module is the data/logic layer such a widget consumes and it is testable without a GPU.

use gogamen_core::{Action, DecisiveEvent};

use crate::counterfactual::{CheckpointError, CounterfactualDebugger, CounterfactualView};
use crate::replay::Recording;

/// Why a decisive event cannot open the debugger.
#[derive(Debug, Clone, PartialEq)]
pub enum OpenAtError {
    /// The event is not marked decisive.
    NotDecisive { tick: u64 },
    /// The checkpoint could not be reconstructed.
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for OpenAtError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotDecisive { tick } => write!(f, "event at tick {tick} is not decisive"),
            Self::Checkpoint(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OpenAtError {}

/// One timeline row: an event aligned with the recording.
#[derive(Debug, Clone, PartialEq)]
pub struct TimelineRow {
    pub tick: u64,
    pub spread: f64,
    pub decisive: bool,
    pub control_bits: Option<f64>,
    /// `false` when the tick lies beyond the recording (the row is kept, never invented away).
    pub within_recording: bool,
}

/// Aligns `events` with `rec` (sorted by tick, stable). Read-only over the recording.
pub fn timeline_rows(rec: &Recording, events: &[DecisiveEvent]) -> Vec<TimelineRow> {
    let mut rows: Vec<TimelineRow> = events
        .iter()
        .map(|e| TimelineRow {
            tick: e.tick,
            spread: e.spread,
            decisive: e.decisive,
            control_bits: e.control_bits,
            within_recording: e.tick <= rec.last_tick(),
        })
        .collect();
    rows.sort_by_key(|r| r.tick);
    rows
}

/// Text lines for the timeline panel, e.g. `t=3 control=0.42 bits spread=1.00 DECISIVE`.
pub fn timeline_lines(rows: &[TimelineRow]) -> Vec<String> {
    rows.iter()
        .map(|r| {
            let cb = r
                .control_bits
                .map_or_else(|| "n/a".to_owned(), |b| format!("{b:.2} bits"));
            let mark = if r.decisive { " DECISIVE" } else { "" };
            let outside = if r.within_recording {
                ""
            } else {
                " (beyond recording)"
            };
            format!(
                "t={} control={cb} spread={:.2}{mark}{outside}",
                r.tick, r.spread
            )
        })
        .collect()
}

/// Opens the counterfactual debugger at a decisive event. Does not alter the recording.
pub fn open_counterfactual_at(
    rec: &Recording,
    event: &DecisiveEvent,
    alternative: Action,
    horizon: u64,
) -> Result<CounterfactualView, OpenAtError> {
    if !event.decisive {
        return Err(OpenAtError::NotDecisive { tick: event.tick });
    }
    CounterfactualDebugger::compare(rec, event.tick, alternative, horizon)
        .map_err(OpenAtError::Checkpoint)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual::RecordingId;
    use crate::recorder::record_grid;
    use gogamen_core::DiscreteAction;
    use gogamen_lab_grid::GridMap;
    use gogamen_lab_grid::world::GRID_ACTION_CARDINALITY;

    fn events() -> Vec<DecisiveEvent> {
        vec![
            DecisiveEvent {
                tick: 6,
                spread: 0.2,
                decisive: false,
                control_bits: None,
            },
            DecisiveEvent {
                tick: 3,
                spread: 1.0,
                decisive: true,
                control_bits: Some(0.42),
            },
            DecisiveEvent {
                tick: 99,
                spread: 0.9,
                decisive: true,
                control_bits: Some(0.1),
            },
        ]
    }

    #[test]
    fn decisive_moments_timeline_rows_and_open_do_not_alter_recording() {
        let rec = record_grid(42, GridMap::cul_de_sac_map(), 12).unwrap();
        let (bytes, id) = (rec.to_bytes(), RecordingId::of(&rec));

        let rows = timeline_rows(&rec, &events());
        assert_eq!(rows.iter().map(|r| r.tick).collect::<Vec<_>>(), [3, 6, 99]);
        assert!(rows[0].decisive && rows[0].within_recording && !rows[2].within_recording);
        let lines = timeline_lines(&rows);
        assert_eq!(lines[0], "t=3 control=0.42 bits spread=1.00 DECISIVE");
        assert_eq!(lines[1], "t=6 control=n/a spread=0.20");
        assert!(lines[2].ends_with("(beyond recording)"));

        let alt = Action::Discrete(DiscreteAction::new(1, GRID_ACTION_CARDINALITY).unwrap());
        let ev = events();
        let v1 = open_counterfactual_at(&rec, &ev[1], alt.clone(), 6).expect("opens at decisive");
        let v2 = open_counterfactual_at(&rec, &ev[1], alt.clone(), 6).unwrap();
        assert_eq!(v1, v2, "opening is repeatable");
        assert_eq!(v1.tick, 3);
        assert_eq!(
            open_counterfactual_at(&rec, &ev[0], alt.clone(), 6),
            Err(OpenAtError::NotDecisive { tick: 6 })
        );
        assert!(matches!(
            open_counterfactual_at(&rec, &ev[2], alt, 6),
            Err(OpenAtError::Checkpoint(_))
        ));

        // Reading and opening left the simulation record bit-identical.
        assert_eq!(rec.to_bytes(), bytes);
        assert_eq!(RecordingId::of(&rec), id);
    }
}
