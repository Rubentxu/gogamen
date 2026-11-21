//! Fixtures deterministas para ejemplos y tests (no son un formato fuente: sólo construyen
//! las trazas existentes `Recording` / `ScientificTraceArchive`).

use std::collections::BTreeMap;

use gogamen_core::{
    Action, DeterministicRng, DiscreteAction, MasterSeed, Observation, PlanStatus, RunId,
    SimDuration, SimulationSnapshot, State, StreamRevision, Terminal, Tick,
};
use gogamen_lab_grid::GridMap;
use gogamen_runtime::scientific_trace::{
    CheckpointRef, DecisionAction, RecordedGuardDecision, ScientificRecord, ScientificTraceArchive,
    StreamSeeds, TickInputs, TraceGapReason, TraceManifest, TraceManifestConfig,
};
use gogamen_studio::recorder::record_grid;
use gogamen_studio::replay::{Frame, Recording};

/// Grabación Grid real (callejón sin salida) con un hueco de ticks `gap_from..=gap_to`
/// eliminado de los frames, para ejercitar huecos explícitos.
pub fn grid_recording_with_gap(ticks: u64, gap_from: u64, gap_to: u64) -> Recording {
    let full = record_grid(7, GridMap::cul_de_sac_map(), ticks).expect("grid fixture");
    let frames: Vec<Frame> = full
        .frames()
        .iter()
        .filter(|f| f.tick < gap_from || f.tick > gap_to)
        .cloned()
        .collect();
    let rows = gogamen_studio::replay::map_to_rows(full.map());
    Recording::new(full.seed(), rows, full.last_tick(), frames).expect("gapped recording")
}

/// Contenido en claro del secreto sembrado en el fixture científico (para tests de no-fuga).
pub const FIXTURE_SECRET: &str = "sk-live-0123456789";
/// Mensaje libre sembrado en el fixture científico.
pub const FIXTURE_MESSAGE: &str = "operator note: tuning gain by hand";
/// Motivo de rechazo sembrado (texto libre).
pub const FIXTURE_REJECT: &str = "velocity limit exceeded by operator override";

/// Archivo científico sintético con: anotaciones (mensaje, secreto, valor ya redactado),
/// un rechazo con motivo y un hueco de manifiesto (revisiones 3..=4, fallo de disco).
pub fn scientific_archive() -> ScientificTraceArchive {
    let seed = MasterSeed::new(11);
    let mut manifest = TraceManifest::new(TraceManifestConfig::new(
        RunId::new(1),
        "spec-demo",
        "model-demo",
        seed,
    ));
    manifest.record_gap(
        StreamRevision::new(3),
        StreamRevision::new(4),
        2,
        TraceGapReason::DiskFailure("EIO on /secret/path".into()),
    );
    manifest.total_records = 4;

    let mut records = Vec::new();
    for (i, rev) in [0u64, 1, 2, 5].into_iter().enumerate() {
        let tick = Tick::new(i as u64);
        let elapsed = SimDuration::from_secs_f64(i as f64 * 0.5).unwrap();
        let obs = Observation::new(tick, elapsed, vec![i as f64, 10.0 - i as f64]).unwrap();
        let guard = if i == 2 {
            RecordedGuardDecision::Reject(FIXTURE_REJECT.into())
        } else {
            RecordedGuardDecision::Allow(Action::Discrete(DiscreteAction::new(1, 3).unwrap()))
        };
        let checkpoint = (i == 1).then(|| {
            let state = State::new(tick, elapsed, vec![1.0, 2.0]).unwrap();
            let snap = SimulationSnapshot::new(
                state,
                &DeterministicRng::new(seed.as_u64()),
                Terminal::Ongoing,
            );
            let mut ann = BTreeMap::new();
            ann.insert("note".to_owned(), FIXTURE_MESSAGE.to_owned());
            ann.insert("api_token".to_owned(), FIXTURE_SECRET.to_owned());
            ann.insert("already".to_owned(), "[REDACTED]".to_owned());
            CheckpointRef::with_annotations(snap, ann)
        });
        records.push(ScientificRecord {
            revision: StreamRevision::new(rev),
            inputs: TickInputs::new(tick, obs),
            decision: DecisionAction {
                plan_status: PlanStatus::Completed,
                proposed_action: None,
                applied_action: matches!(guard, RecordedGuardDecision::Allow(_))
                    .then(|| Action::Discrete(DiscreteAction::new(1, 3).unwrap())),
                guard_decision: guard,
                used_fallback: false,
            },
            stream_seeds: StreamSeeds::new(seed),
            checkpoint,
        });
    }
    ScientificTraceArchive { manifest, records }
}
