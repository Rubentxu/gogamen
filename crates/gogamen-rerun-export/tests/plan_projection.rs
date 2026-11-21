//! Proyección pura (sin Rerun): AC2 (huecos, redacciones, permisos) y AC3 (descartable).

use gogamen_rerun_export::fixtures::{
    FIXTURE_MESSAGE, FIXTURE_REJECT, FIXTURE_SECRET, grid_recording_with_gap, scientific_archive,
};
use gogamen_rerun_export::plan::{REDACTED_MARKER, WITHHELD_MARKER};
use gogamen_rerun_export::{
    ExportOptions, ExportPermissions, ItemTime, Payload, plan_from_archive, plan_from_recording,
};
use gogamen_runtime::scientific_trace::{SensitiveFieldMode, SensitiveFieldPolicy};

fn find<'a>(
    plan: &'a gogamen_rerun_export::ExportPlan,
    entity: &str,
) -> Vec<&'a gogamen_rerun_export::Item> {
    plan.items.iter().filter(|i| i.entity == entity).collect()
}

fn is_tick(it: &gogamen_rerun_export::Item, tick: u64) -> bool {
    matches!(it.time, ItemTime::At { tick: Some(t), .. } if t == tick)
}

#[test]
fn permissions_default_to_deny() {
    let p = ExportPermissions::default();
    assert!(!p.messages && !p.secrets);
    assert_eq!(ExportOptions::default().permissions, p);
}

#[test]
fn recording_projects_time_units_entities_and_scene() {
    let rec = grid_recording_with_gap(20, 100, 100);
    let plan = plan_from_recording(&rec, &ExportOptions::default());
    // Unidades explícitas.
    let metas: Vec<_> = plan
        .items
        .iter()
        .filter_map(|i| match &i.payload {
            Payload::SeriesMeta { unit, .. } => Some(unit.as_str()),
            _ => None,
        })
        .collect();
    assert!(metas.contains(&"reward") && metas.contains(&"count"));
    // Tiempo simulado = tick / 10 s.
    let rewards = find(&plan, "gogamen/series/total_reward");
    for it in rewards {
        if let ItemTime::At {
            tick: Some(t),
            seconds: Some(s),
            ..
        } = it.time
        {
            assert!((s - t as f64 / 10.0).abs() < 1e-12);
        }
    }
    // Vista de escena: mapa estático + agente.
    assert!(plan.entities().contains(&"gogamen/scene/map/wall"));
    assert!(plan.entities().contains(&"gogamen/scene/agent"));
}

#[test]
fn recording_gaps_are_explicit_never_zero_filled() {
    let rec = grid_recording_with_gap(30, 5, 8);
    assert!(!rec.gaps().is_empty(), "fixture must contain a gap");
    let plan = plan_from_recording(&rec, &ExportOptions::default());
    for tick in 5..=8u64 {
        let at = |it: &&gogamen_rerun_export::Item| is_tick(it, tick);
        let series = find(&plan, "gogamen/series/total_reward");
        let hit: Vec<_> = series.iter().filter(|i| at(i)).collect();
        assert_eq!(hit.len(), 1);
        assert_eq!(hit[0].payload, Payload::GapScalar, "tick {tick}");
        let agent = find(&plan, "gogamen/scene/agent");
        assert!(
            agent
                .iter()
                .filter(|i| at(i))
                .all(|i| i.payload == Payload::GapEntity)
        );
        let missing = find(&plan, "gogamen/status/missing");
        assert!(missing.iter().any(at));
    }
    // Ningún Scalar(0.0) inventado en los ticks del hueco.
    assert!(!find(&plan, "gogamen/series/collisions").iter().any(|i| {
        matches!(i.time, ItemTime::At { tick: Some(t), .. } if (5..=8).contains(&t))
            && matches!(i.payload, Payload::Scalar(_))
    }));
}

#[test]
fn archive_denies_messages_and_secrets_by_default() {
    let archive = scientific_archive();
    let plan = plan_from_archive(&archive, &ExportOptions::default());
    let text = plan.exported_text();
    for leaked in [
        FIXTURE_SECRET,
        FIXTURE_MESSAGE,
        FIXTURE_REJECT,
        "EIO",
        "/secret/path",
    ] {
        assert!(!text.contains(leaked), "leaked `{leaked}`");
    }
    // Redactado y retenido se conservan como marcadores.
    let token = find(&plan, "gogamen/checkpoint/annotations/api_token");
    assert_eq!(token[0].payload, Payload::Redacted);
    let already = find(&plan, "gogamen/checkpoint/annotations/already");
    assert_eq!(already[0].payload, Payload::Redacted);
    let note = find(&plan, "gogamen/checkpoint/annotations/note");
    assert_eq!(note[0].payload, Payload::Withheld);
    let rej = find(&plan, "gogamen/decision/guard_message");
    assert_eq!(rej[0].payload, Payload::Withheld);
    assert!(!REDACTED_MARKER.is_empty() && !WITHHELD_MARKER.is_empty());
}

#[test]
fn archive_messages_permission_does_not_unlock_secrets() {
    let options = ExportOptions {
        permissions: ExportPermissions {
            messages: true,
            secrets: false,
        },
        ..ExportOptions::default()
    };
    let plan = plan_from_archive(&scientific_archive(), &options);
    let text = plan.exported_text();
    assert!(text.contains(FIXTURE_MESSAGE));
    assert!(text.contains(FIXTURE_REJECT));
    assert!(!text.contains(FIXTURE_SECRET));
    assert_eq!(
        find(&plan, "gogamen/checkpoint/annotations/api_token")[0].payload,
        Payload::Redacted
    );
}

#[test]
fn archive_missing_data_is_preserved_as_gaps() {
    let plan = plan_from_archive(&scientific_archive(), &ExportOptions::default());
    let gaps: Vec<_> = find(&plan, "gogamen/status/missing");
    assert_eq!(gaps.len(), 1);
    match &gaps[0].payload {
        Payload::Label(s) => {
            assert!(s.contains("3..=4") && s.contains("disk failure"), "{s}");
        }
        other => panic!("unexpected {other:?}"),
    }
    // Hueco escalar explícito por feature, en la revisión del hueco.
    for i in 0..2 {
        let f = find(&plan, &format!("gogamen/obs/feature_{i}"));
        assert!(f.iter().any(|it| it.payload == Payload::GapScalar
            && matches!(
                it.time,
                ItemTime::At {
                    revision: Some(3),
                    tick: None,
                    ..
                }
            )));
    }
    let complete = find(&plan, "gogamen/meta/is_complete");
    assert_eq!(complete[0].payload, Payload::Label("false".into()));
}

#[test]
fn export_projection_leaves_sources_byte_identical_and_is_deterministic() {
    let rec = grid_recording_with_gap(25, 3, 4);
    let archive = scientific_archive();
    let policy = SensitiveFieldPolicy::new(SensitiveFieldMode::Redact);
    let rec_before = rec.to_bytes();
    let arc_before = archive.to_bytes(&policy).unwrap();

    let opts = ExportOptions::default();
    let p1 = plan_from_recording(&rec, &opts);
    let _ = plan_from_archive(&archive, &opts);
    drop(p1); // descartar el plan no tiene efecto sobre el replay

    assert_eq!(rec.to_bytes(), rec_before);
    assert_eq!(archive.to_bytes(&policy).unwrap(), arc_before);
    assert_eq!(
        plan_from_recording(&rec, &opts),
        plan_from_recording(&rec, &opts)
    );
}
