use std::time::Duration;

use gogamen_core::{
    ActionGuard, ActionProposal, ActionSpace, Confidence, EvidenceRef, FixtureBehavior, FrameId,
    GuardDecision, IdempotencyKey, MessageId, MonotonicDeadline, MonotonicInstant, ObservationKind,
    OptionId, ProvenanceMetadata, QueryId, ReplaySystemOneProvider, SensorId, SystemOneFixture,
    Tick, UnitTag, VirtualMonotonicClock, WorkflowStepError, WorkflowStepStatus,
};
use gogamen_lab_automation::{
    COGNITIVE_DECISION_ESCALATE, EventClassification, FakeServiceActuator, QUERY_INCIDENT_CLASSIFY,
    RemediationKind, STATE_ERROR, STATE_MONITORING, STATE_REMEDIATE, STATE_RESOLVED, STATE_TIMEOUT,
    ServiceSupervisor, ServiceSupervisorConfig, ServiceThresholds, SyntheticServiceEvent,
    compile_supervisor_spec,
};
use gogamen_spec::SystemSpec;

fn sample_provenance() -> ProvenanceMetadata {
    ProvenanceMetadata {
        message_id: MessageId::new(1),
        observed_at: MonotonicInstant::ZERO,
        received_at: MonotonicInstant::ZERO,
        sensor_id: SensorId::new(42),
        frame_id: FrameId::world(),
        units: UnitTag::dimensionless(),
        confidence: Confidence::new(0.88).expect("valid confidence"),
        revision: 1,
        evidence_ref: EvidenceRef::new(9001),
        claim_kind: ObservationKind::Proposal,
    }
}

#[test]
fn deterministic_mode_executes_alert_diagnose_remediate_resolved_and_replays_identically() {
    let clock = VirtualMonotonicClock::new();
    let config = ServiceSupervisorConfig::default();

    let script = vec![
        (
            Tick::new(1),
            SyntheticServiceEvent::LatencySpike { p99_ms: 80 },
        ),
        (
            Tick::new(2),
            SyntheticServiceEvent::QueueBacklog { depth: 120 },
        ),
        (
            Tick::new(3),
            SyntheticServiceEvent::ProbePassed { consecutive_ok: 2 },
        ),
    ];

    let mut supervisor = ServiceSupervisor::new(config.clone()).expect("create supervisor");
    assert_eq!(supervisor.current_state().as_str(), STATE_MONITORING);

    // Tick 1: por debajo del umbral -> permanece en Monitoring.
    let step1 = supervisor
        .process_event(
            &script[0].1,
            Some(IdempotencyKey::new("ev-1").unwrap()),
            script[0].0,
            &clock,
            None,
        )
        .unwrap();
    assert_eq!(step1.classification, EventClassification::BelowThreshold);
    assert_eq!(step1.current_state.as_str(), STATE_MONITORING);
    assert!(supervisor.actuator().executed_actions().is_empty());

    // Tick 2: QueueBacklog >= 100 -> Alert -> Diagnose -> Remediate + FakeServiceActuator.
    let step2 = supervisor
        .process_event(
            &script[1].1,
            Some(IdempotencyKey::new("ev-2").unwrap()),
            script[1].0,
            &clock,
            None,
        )
        .unwrap();
    assert!(matches!(
        step2.classification,
        EventClassification::Breach {
            kind: "queue_backlog",
            ..
        }
    ));
    assert_eq!(step2.workflow_outcomes.len(), 3);
    assert_eq!(step2.workflow_outcomes[0].current_state.as_str(), "Alert");
    assert_eq!(
        step2.workflow_outcomes[1].current_state.as_str(),
        "Diagnose"
    );
    assert_eq!(
        step2.workflow_outcomes[2].current_state.as_str(),
        STATE_REMEDIATE
    );
    assert!(
        step2
            .guard_decision
            .as_ref()
            .is_some_and(GuardDecision::is_allowed)
    );
    assert_eq!(supervisor.actuator().executed_actions().len(), 1);
    assert_eq!(
        supervisor.actuator().executed_actions()[0].action_name,
        "restart_worker"
    );

    // Tick 3: ProbePassed >= 2 -> Resolved (terminal).
    let step3 = supervisor
        .process_event(
            &script[2].1,
            Some(IdempotencyKey::new("ev-3").unwrap()),
            script[2].0,
            &clock,
            None,
        )
        .unwrap();
    assert_eq!(step3.current_state.as_str(), STATE_RESOLVED);
    assert!(supervisor.is_terminal());

    // Replay determinista puro sin IA.
    let replay_a = ServiceSupervisor::replay(config.clone(), &script, &clock, None).unwrap();
    let replay_b = ServiceSupervisor::replay(config, &script, &clock, None).unwrap();

    assert_eq!(replay_a.final_state.as_str(), STATE_RESOLVED);
    assert_eq!(replay_a.trace_hash, replay_b.trace_hash);
    assert_eq!(replay_a.trace, replay_b.trace);
    assert_eq!(replay_a.executed_actions, replay_b.executed_actions);
    assert!(replay_a.cognitive_events.is_empty());
}

#[test]
fn cognitive_mode_classifies_ambiguous_fixture_without_changing_core_workflow() {
    let clock = VirtualMonotonicClock::new();
    let config = ServiceSupervisorConfig::default();

    // Latencia 180ms está en la banda ambigua [150, 250).
    let ambiguous_event = SyntheticServiceEvent::LatencySpike { p99_ms: 180 };

    // Sin proveedor System One, el evento ambiguo no dispara alerta.
    let mut no_ai = ServiceSupervisor::new(config.clone()).unwrap();
    let no_ai_step = no_ai
        .process_event(
            &ambiguous_event,
            Some(IdempotencyKey::new("amb-1").unwrap()),
            Tick::new(1),
            &clock,
            None,
        )
        .unwrap();
    assert!(matches!(
        no_ai_step.classification,
        EventClassification::Ambiguous { .. }
    ));
    assert_eq!(no_ai_step.current_state.as_str(), STATE_MONITORING);
    assert!(no_ai_step.cognitive_event.is_none());

    // Con ReplaySystemOneProvider clasificando el fixture como "escalate_alert":
    let escalate_opt = OptionId::new(COGNITIVE_DECISION_ESCALATE).unwrap();
    let noise_opt = OptionId::new("transient_noise").unwrap();
    let fixture = SystemOneFixture::new(
        FixtureBehavior::Choice {
            selected: escalate_opt.clone(),
            raw_scores: vec![(escalate_opt, 3.2), (noise_opt, 0.4)],
            validated_distribution: None,
        },
        sample_provenance(),
    );
    let provider = ReplaySystemOneProvider::new("fixture-classifier")
        .with_fixture(QueryId::new(QUERY_INCIDENT_CLASSIFY).unwrap(), fixture);

    let script = vec![
        (Tick::new(1), ambiguous_event),
        (
            Tick::new(2),
            SyntheticServiceEvent::RecoveryConfirmed { healthy_ticks: 3 },
        ),
    ];

    let replay_1 =
        ServiceSupervisor::replay(config.clone(), &script, &clock, Some(&provider)).unwrap();
    let replay_2 = ServiceSupervisor::replay(config, &script, &clock, Some(&provider)).unwrap();

    assert_eq!(replay_1.final_state.as_str(), STATE_RESOLVED);
    assert_eq!(replay_1.cognitive_events.len(), 1);
    assert_eq!(
        replay_1.cognitive_events[0].decision_code(),
        COGNITIVE_DECISION_ESCALATE
    );
    assert_eq!(replay_1.executed_actions.len(), 1);
    assert_eq!(replay_1.trace_hash, replay_2.trace_hash);
    assert_eq!(replay_1.cognitive_events, replay_2.cognitive_events);
}

#[test]
fn third_party_can_configure_and_compile_custom_service_via_system_spec_json() {
    let custom_config = ServiceSupervisorConfig {
        service_name: "billing-worker".to_owned(),
        thresholds: ServiceThresholds {
            max_failed_checks: 2,
            latency_threshold_ms: 80,
            latency_ambiguity_ms: Some(50),
            queue_backlog_threshold: 25,
            required_probe_passes: 1,
            required_recovery_ticks: 1,
        },
        remediation_action: RemediationKind::DrainQueue,
        max_retries: 1,
        action_space_size: 8,
    };

    let json = custom_config
        .to_system_spec()
        .to_canonical_json()
        .expect("serialize spec");
    let parsed_spec = SystemSpec::from_json(&json).expect("parse third-party json");

    let (compiled_cfg, plan) =
        compile_supervisor_spec(&parsed_spec).expect("compile third-party spec");
    assert_eq!(compiled_cfg, custom_config);
    assert_ne!(plan.semantic_fingerprint(), 0);

    let clock = VirtualMonotonicClock::new();
    let mut supervisor = ServiceSupervisor::from_spec_json(&json).unwrap();

    // 85ms supera el umbral custom de 80ms (aunque no superaría el default de 250ms).
    let step = supervisor
        .process_event(
            &SyntheticServiceEvent::LatencySpike { p99_ms: 85 },
            None,
            Tick::new(1),
            &clock,
            None,
        )
        .unwrap();
    assert_eq!(step.current_state.as_str(), STATE_REMEDIATE);
    assert_eq!(supervisor.actuator().executed_actions().len(), 1);
    assert_eq!(
        supervisor.actuator().executed_actions()[0].action_name,
        "drain_queue"
    );
    assert_eq!(
        supervisor.actuator().executed_actions()[0].discrete_index,
        2
    );

    let resolved = supervisor
        .process_event(
            &SyntheticServiceEvent::RecoveryConfirmed { healthy_ticks: 1 },
            None,
            Tick::new(2),
            &clock,
            None,
        )
        .unwrap();
    assert_eq!(resolved.current_state.as_str(), STATE_RESOLVED);
    assert!(supervisor.is_terminal());
}

#[test]
fn error_and_timeout_branches_and_unauthorized_actuator_are_enforced() {
    let clock = VirtualMonotonicClock::new();
    let config = ServiceSupervisorConfig {
        max_retries: 1,
        ..ServiceSupervisorConfig::default()
    };

    // 1. Rama OnError tras agotar RetryPolicy (max_retries = 1).
    let mut err_sup = ServiceSupervisor::new(config.clone()).unwrap();
    let err = WorkflowStepError::new("E_IO", "synthetic failure").unwrap();
    let first = err_sup
        .report_error(&err, None, Tick::new(1), &clock)
        .unwrap();
    assert!(matches!(
        first.status,
        WorkflowStepStatus::Retrying {
            attempt: 1,
            max_retries: 1
        }
    ));
    let second = err_sup
        .report_error(&err, None, Tick::new(2), &clock)
        .unwrap();
    assert_eq!(second.status, WorkflowStepStatus::ErrorBranchTaken);
    assert_eq!(err_sup.current_state().as_str(), STATE_ERROR);
    assert!(err_sup.is_terminal());

    // 2. Rama OnTimeout cuando expira MonotonicDeadline en VirtualMonotonicClock.
    let mut to_sup = ServiceSupervisor::new(config).unwrap();
    let deadline = MonotonicDeadline::from_now(&clock, Duration::from_millis(50));
    to_sup.set_deadline(Some(deadline));
    clock.advance(Duration::from_millis(60));
    assert!(deadline.is_expired(&clock));

    let to_step = to_sup
        .process_event(
            &SyntheticServiceEvent::HealthDegraded { failed_checks: 1 },
            None,
            Tick::new(1),
            &clock,
            None,
        )
        .unwrap();
    assert_eq!(to_step.current_state.as_str(), STATE_TIMEOUT);
    assert!(to_sup.is_terminal());

    // 3. ActionGuard rechaza propuestas sin autorización explícita en FakeServiceActuator.
    let mut actuator = FakeServiceActuator::new();
    let guard = ActionGuard::strict();
    let space = ActionSpace::discrete(4).unwrap();
    let unauthorized = ActionProposal::unauthorized(Tick::new(1));
    let decision = actuator.evaluate_and_execute(
        &guard,
        &unauthorized,
        &space,
        Tick::new(1),
        "auth",
        "restart_worker",
    );
    assert!(decision.is_rejected());
    assert!(actuator.executed_actions().is_empty());
    assert_eq!(actuator.rejected_count(), 1);
}
