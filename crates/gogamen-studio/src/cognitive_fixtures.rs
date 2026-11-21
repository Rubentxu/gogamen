//! Fixtures de [`CognitiveView`] (GOG-96) construidos con tipos reales de `gogamen-core`:
//! las creencias se actualizan con [`BinaryBelief::update_with_observation`] y los mensajes
//! son [`SituatedMessage`]. Ambos se proyectan con la **misma** función
//! [`crate::cognitive::project_cognitive_scene`].

use std::collections::BTreeMap;
use std::time::Duration;

use gogamen_core::{
    BinaryBelief, CommCost, CommPayload, Confidence, ConversationScope, EntityId, EpisodeId,
    EpisodeProvenance, EpisodicRecord, EvidenceDeduplicator, EvidenceRef, FrameId, InterlocutorId,
    InterlocutorKind, LikelihoodModel, MessageId, MonotonicInstant, ObservationEnvelope,
    ObservationKind, OwnerId, ProvenanceMetadata, SensorId, SituatedMessage, SummaryCode, Tick,
    UnitTag,
};

use crate::cognitive::{
    Authorization, CausalLink, CogRef, CognitiveView, EvidenceSource, EvidenceView, HypothesisView,
    IntentionView, Known, MessageView,
};

fn envelope(
    msg: u64,
    sensor: u64,
    ev: u64,
    tick: u64,
    kind: ObservationKind,
    c: f64,
    value: bool,
) -> ObservationEnvelope<bool> {
    let at = MonotonicInstant::from_nanos(u128::from(tick) * 1_000);
    ObservationEnvelope::new(
        ProvenanceMetadata {
            message_id: MessageId::new(msg),
            observed_at: at,
            received_at: at,
            sensor_id: SensorId::new(sensor),
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence: Confidence::new(c).expect("confidence"),
            revision: 1,
            evidence_ref: EvidenceRef::new(ev),
            claim_kind: kind,
        },
        value,
    )
}

/// Aplica una observación a una creencia (única vía de actualización: la de core).
fn observe(belief: &mut BinaryBelief, env: &ObservationEnvelope<bool>, sens: f64, spec: f64) {
    let model = LikelihoodModel::from_sensitivity_specificity(sens, spec).expect("model");
    let mut dedup = EvidenceDeduplicator::new();
    let mut wm = None;
    belief
        .update_with_observation(env, &model, &mut dedup, &mut wm)
        .expect("update");
}

fn conf(v: f64) -> Known<Confidence> {
    Known::Known(Confidence::new(v).expect("confidence"))
}

fn fact_message(id: u64, from: &str, to: &str, key: &str, value: &str, at: u64) -> SituatedMessage {
    SituatedMessage::new(
        MessageId::new(id),
        InterlocutorId::new(from).expect("sender"),
        InterlocutorId::new(to).expect("recipient"),
        ConversationScope::Direct,
        MonotonicInstant::from_nanos(u128::from(at) * 1_000),
        CommCost::new(1, 0, Duration::ZERO),
        CommPayload::structured_fact(key, value, BTreeMap::new(), None, None).expect("fact"),
    )
    .expect("message")
}

fn episode(id: u64, tick: u64, ev: u64, msg: u64, kind: ObservationKind) -> EpisodicRecord {
    EpisodicRecord {
        episode_id: EpisodeId::new(id),
        owner: OwnerId::new(1),
        entity: EntityId::new(7),
        tick: Tick::new(tick),
        observed_at: MonotonicInstant::from_nanos(u128::from(tick) * 1_000),
        revision: 1,
        provenance: EpisodeProvenance::new(
            EvidenceRef::new(ev),
            MessageId::new(msg),
            SensorId::new(1),
            kind,
            Confidence::new(0.9).expect("confidence"),
        ),
        summary_code: SummaryCode::new(42),
        numeric_features: vec![3.0, 2.0],
        raw_text_redacted: true,
    }
}

/// Fixture del mundo de lenguaje de Grid: dos hipótesis contradictorias sobre la celda (3,2).
///
/// `updated = false`: sólo hay una afirmación humana (celda libre) en el tick 4.
/// `updated = true`: además un sensor directo (tick 9) dice "bloqueada" y ambas creencias se
/// actualizan con el modelo de verosimilitud de core.
pub fn grid_language_fixture(updated: bool, authorization: Authorization) -> CognitiveView {
    let subject = "cell(3,2)";
    // H1 = "bloqueada" y H2 = "libre" (la observación `false` apoya H2 y refuta H1)
    let mut blocked = BinaryBelief::new(0.5).expect("prior");
    let mut free = BinaryBelief::new(0.5).expect("prior");
    let human = envelope(10, 1, 100, 4, ObservationKind::Assertion, 0.9, false);
    observe(&mut blocked, &human, 0.7, 0.7);
    observe(
        &mut free,
        &envelope(10, 1, 100, 4, ObservationKind::Assertion, 0.9, true),
        0.7,
        0.7,
    );
    let human_ev = |says| EvidenceView {
        evidence: EvidenceRef::new(100),
        source: EvidenceSource::Human,
        claim_kind: ObservationKind::Assertion,
        says: Known::Known(says),
        observed_tick: Known::Known(4),
        reported_confidence: conf(0.9),
        source_reliability: conf(0.7),
    };
    let mut h1_ev = vec![human_ev(false)];
    let mut h2_ev = vec![human_ev(true)];
    let mut last = 4;
    if updated {
        let sensor = |v| envelope(11, 2, 101, 9, ObservationKind::Measurement, 0.6, v);
        observe(&mut blocked, &sensor(true), 0.9, 0.9);
        observe(&mut free, &sensor(false), 0.9, 0.9);
        let sensor_ev = |says| EvidenceView {
            evidence: EvidenceRef::new(101),
            source: EvidenceSource::DirectSensor,
            claim_kind: ObservationKind::Measurement,
            says: Known::Known(says),
            observed_tick: Known::Known(9),
            reported_confidence: conf(0.6),
            source_reliability: conf(0.9),
        };
        h1_ev.push(sensor_ev(true));
        h2_ev.push(sensor_ev(false));
        last = 9;
    }
    let mut view = CognitiveView {
        title: "grid language-world".into(),
        tick: 12,
        authorization,
        hypotheses: vec![
            HypothesisView {
                id: 1,
                subject: subject.into(),
                statement: "blocked".into(),
                belief: Known::Known(blocked),
                last_update_tick: Known::Known(last),
                evidence: h1_ev,
            },
            HypothesisView {
                id: 2,
                subject: subject.into(),
                statement: "free".into(),
                belief: Known::Known(free),
                last_update_tick: Known::Known(last),
                evidence: h2_ev,
            },
        ],
        episodes: vec![episode(1, 4, 100, 10, ObservationKind::Assertion)],
        intentions: vec![IntentionView {
            id: 1,
            label: "route-around cell(3,2)".into(),
            since_tick: Known::Known(10),
        }],
        messages: vec![MessageView {
            message: fact_message(10, "operator", "agent-1", "cell(3,2)", "free", 4),
            sender_kind: Known::Known(InterlocutorKind::Human),
            sent_tick: Known::Known(4),
        }],
        links: vec![CausalLink {
            from: CogRef::Message(MessageId::new(10)),
            to: CogRef::Evidence(EvidenceRef::new(100)),
        }],
    };
    if updated {
        view.links.push(CausalLink {
            from: CogRef::Evidence(EvidenceRef::new(101)),
            to: CogRef::Intention(1),
        });
    }
    view
}

/// Fixture de Life: señal simbólica entre organismos con campos desconocidos explícitos.
pub fn life_signal_fixture(authorization: Authorization) -> CognitiveView {
    let mut food = BinaryBelief::new(0.5).expect("prior");
    observe(
        &mut food,
        &envelope(20, 3, 200, 30, ObservationKind::Assertion, 0.5, true),
        0.6,
        0.6,
    );
    CognitiveView {
        title: "life signal channel".into(),
        tick: 33,
        authorization,
        hypotheses: vec![HypothesisView {
            id: 1,
            subject: "food-near-organism-5".into(),
            statement: "food present".into(),
            belief: Known::Known(food),
            last_update_tick: Known::Known(30),
            evidence: vec![EvidenceView {
                evidence: EvidenceRef::new(200),
                source: EvidenceSource::Agent,
                claim_kind: ObservationKind::Assertion,
                says: Known::Known(true),
                observed_tick: Known::Known(30),
                reported_confidence: conf(0.5),
                source_reliability: Known::Unknown,
            }],
        }],
        episodes: vec![episode(3, 30, 200, 20, ObservationKind::Assertion)],
        intentions: vec![],
        messages: vec![MessageView {
            message: fact_message(20, "organism-9", "organism-5", "signal", "S3", 30),
            sender_kind: Known::Unknown,
            sent_tick: Known::Unknown,
        }],
        links: vec![],
    }
}
