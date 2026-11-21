use std::time::Duration;

use gogamen_core::{
    Action, ActionGuard, ActionSpace, CapabilityCatalog, CapabilityVersion, EvidenceRef,
    ModuleContractError, ModuleManifest, ModuleOrigin, ObservationKind, OperationBinding,
    OperationSpec, Permission, SafeFallbackPolicy, SensorId, Terminal, TerminalReason, Tick,
};
use gogamen_experimental::mock::{
    MOCK_GENERATIVE_SENSOR, MockGenerative, MockVisual, wrap_generated,
};
use gogamen_experimental::{
    AbstainReason, Availability, CandidateBody, CandidateRejection, CandidateValidator, Claim,
    DetectionRejection, ExperimentalCapability as Cap, ExperimentalConfig, ExperimentalHub,
    GenerativeCandidate, Outcome, ProposalRequest, ValidatedCandidate, ValidationContext,
    VisualDetection, VisualFrame,
};

const V1: CapabilityVersion = CapabilityVersion::new(1, 0, 0);

fn catalog() -> CapabilityCatalog {
    let mut c = CapabilityCatalog::new();
    c.register_module(
        ModuleManifest::new("nav", ModuleOrigin::Builtin, V1)
            .with_granted_permission(Permission::state_write())
            .with_operation(
                OperationSpec::new("move", V1).with_required_permission(Permission::state_write()),
            )
            .with_operation(
                OperationSpec::new("fs-write", V1)
                    .with_required_permission(Permission::filesystem()),
            ),
    )
    .expect("register");
    c
}

fn act(
    module: &str,
    op: &str,
    v: CapabilityVersion,
    idx: usize,
    space: &ActionSpace,
) -> CandidateBody {
    CandidateBody::Action {
        binding: OperationBinding::new(module, op, v),
        action: Action::Discrete(
            // Build via the space when valid, otherwise through an oversized space (hallucinated index).
            match Action::discrete_in_space(idx, space) {
                Ok(Action::Discrete(d)) => d,
                _ => match Action::discrete_in_space(idx, &ActionSpace::discrete(idx + 1).unwrap())
                {
                    Ok(Action::Discrete(d)) => d,
                    _ => unreachable!(),
                },
            },
        ),
    }
}

fn cand(body: CandidateBody, evidence: Vec<u64>) -> GenerativeCandidate {
    GenerativeCandidate {
        id: "c".into(),
        body,
        claims: vec![Claim {
            statement: "s".into(),
            evidence: evidence.into_iter().map(EvidenceRef::new).collect(),
        }],
    }
}

fn env(c: GenerativeCandidate) -> gogamen_core::ObservationEnvelope<GenerativeCandidate> {
    wrap_generated(1, c, ObservationKind::Proposal, MOCK_GENERATIVE_SENSOR)
}

struct Fx {
    catalog: CapabilityCatalog,
    guard: ActionGuard,
    space: ActionSpace,
}

fn fx() -> Fx {
    Fx {
        catalog: catalog(),
        guard: ActionGuard::strict(),
        space: ActionSpace::discrete(3).unwrap(),
    }
}

fn run(
    f: &Fx,
    e: &gogamen_core::ObservationEnvelope<GenerativeCandidate>,
    terminal: Terminal,
) -> Result<ValidatedCandidate, CandidateRejection> {
    let v = CandidateValidator::new(&f.catalog, &f.guard)
        .with_generative_sensor(MOCK_GENERATIVE_SENSOR)
        .with_evidence([EvidenceRef::new(1), EvidenceRef::new(2)])
        .with_goals(["explore"]);
    v.validate(
        e,
        &ValidationContext {
            action_space: &f.space,
            tick: Tick::ZERO,
            terminal,
        },
    )
}

// ---------------- AC1: opt-in, abstention, discovery ----------------

fn req() -> ProposalRequest {
    ProposalRequest {
        key: "k".into(),
        prompt: "p".into(),
    }
}

#[test]
fn default_config_abstains_and_never_calls_provider() {
    let mock = MockGenerative::new(Duration::from_millis(1), 1).script("k", vec![]);
    let hub = ExperimentalHub::new(ExperimentalConfig::default()).with_generative(Box::new(mock));
    assert_eq!(
        hub.propose(&req()).abstain_reason(),
        Some(&AbstainReason::DisabledByConfig(Cap::GenerativeSystemTwo))
    );
    let frame = VisualFrame {
        frame_id: "f".into(),
        bytes: vec![],
    };
    assert_eq!(
        hub.perceive(&frame).abstain_reason(),
        Some(&AbstainReason::DisabledByConfig(Cap::VisualBox))
    );
    // A bare default hub abstains too.
    assert!(ExperimentalHub::default().propose(&req()).is_abstain());
}

#[test]
fn enabled_without_provider_or_unavailable_or_failing_abstains_typed() {
    let cfg = ExperimentalConfig::default()
        .enable(Cap::GenerativeSystemTwo)
        .enable(Cap::VisualBox);
    let hub = ExperimentalHub::new(cfg);
    assert_eq!(
        hub.propose(&req()).abstain_reason(),
        Some(&AbstainReason::NoProvider(Cap::GenerativeSystemTwo))
    );

    let hub = ExperimentalHub::new(cfg)
        .with_generative(Box::new(MockGenerative::default().unavailable("no model")))
        .with_visual(Box::new(MockVisual::default().unavailable("no box")));
    assert!(matches!(
        hub.propose(&req()).abstain_reason(),
        Some(AbstainReason::ProviderUnavailable {
            reason: "no model",
            ..
        })
    ));
    let frame = VisualFrame {
        frame_id: "f".into(),
        bytes: vec![],
    };
    assert!(matches!(
        hub.perceive(&frame).abstain_reason(),
        Some(AbstainReason::ProviderUnavailable {
            reason: "no box",
            ..
        })
    ));

    // Provider error (no fixture for key) -> ProviderFailed, not a panic.
    let hub = ExperimentalHub::new(cfg).with_generative(Box::new(MockGenerative::default()));
    assert!(matches!(
        hub.propose(&req()).abstain_reason(),
        Some(AbstainReason::ProviderFailed { .. })
    ));
}

#[test]
fn enabled_and_available_yields_value_and_discovery_reports_it() {
    let cfg = ExperimentalConfig::default().enable(Cap::GenerativeSystemTwo);
    let mock = MockGenerative::new(Duration::from_millis(5), 7).script("k", vec![]);
    let hub = ExperimentalHub::new(cfg).with_generative(Box::new(mock));
    assert!(matches!(hub.propose(&req()), Outcome::Value(_)));
    let report = hub.discover();
    assert!(report.is_available(Cap::GenerativeSystemTwo));
    assert!(!report.is_available(Cap::VisualBox));
    let visual = &report.statuses[0];
    assert!(!visual.enabled && !visual.provider_registered);

    // Default hub: nothing available, and discovery says why.
    let none = ExperimentalHub::default().discover();
    assert!(!none.is_available(Cap::GenerativeSystemTwo) && !none.is_available(Cap::VisualBox));
    assert_eq!(
        none.statuses[1].availability,
        Availability::Unavailable("disabled by config")
    );
}

// ---------------- AC2: validation paths ----------------

#[test]
fn rejects_unlabelled_origin_and_wrong_claim_kind() {
    let f = fx();
    let c = cand(
        CandidateBody::Goal {
            name: "explore".into(),
        },
        vec![1],
    );
    let e = wrap_generated(1, c.clone(), ObservationKind::Proposal, SensorId::new(1));
    assert!(matches!(
        run(&f, &e, Terminal::Ongoing),
        Err(CandidateRejection::UnlabelledOrigin { .. })
    ));
    let e = wrap_generated(1, c, ObservationKind::Measurement, MOCK_GENERATIVE_SENSOR);
    assert_eq!(
        run(&f, &e, Terminal::Ongoing),
        Err(CandidateRejection::WrongClaimKind {
            found: ObservationKind::Measurement
        })
    );
}

#[test]
fn rejects_ungrounded_claims() {
    let f = fx();
    let goal = || CandidateBody::Goal {
        name: "explore".into(),
    };
    // Unknown evidence (hallucinated).
    assert_eq!(
        run(&f, &env(cand(goal(), vec![777])), Terminal::Ongoing),
        Err(CandidateRejection::UngroundedClaim { claim_index: 0 })
    );
    // No evidence at all.
    assert_eq!(
        run(&f, &env(cand(goal(), vec![])), Terminal::Ongoing),
        Err(CandidateRejection::UngroundedClaim { claim_index: 0 })
    );
    // Partly grounded still rejected.
    assert_eq!(
        run(&f, &env(cand(goal(), vec![1, 777])), Terminal::Ongoing),
        Err(CandidateRejection::UngroundedClaim { claim_index: 0 })
    );
    // Self-citation of the generator's own evidence ref does not ground anything.
    let own = env(cand(goal(), vec![1_000_001]));
    assert!(own.evidence_ref().as_u64() == 1_000_001);
    assert_eq!(
        run(&f, &own, Terminal::Ongoing),
        Err(CandidateRejection::UngroundedClaim { claim_index: 0 })
    );
    // Notice with no claims is an assertion without evidence.
    let mut n = cand(
        CandidateBody::Notice {
            text: "door open".into(),
        },
        vec![1],
    );
    n.claims.clear();
    assert_eq!(
        run(&f, &env(n), Terminal::Ongoing),
        Err(CandidateRejection::NoticeWithoutClaims)
    );
    let e = cand(CandidateBody::Notice { text: "  ".into() }, vec![1]);
    assert_eq!(
        run(&f, &env(e), Terminal::Ongoing),
        Err(CandidateRejection::EmptyNotice)
    );
}

#[test]
fn rejects_out_of_catalog_goals_and_operations() {
    let f = fx();
    assert!(matches!(
        run(
            &f,
            &env(cand(act("ghost", "move", V1, 0, &f.space), vec![1])),
            Terminal::Ongoing
        ),
        Err(CandidateRejection::OutOfCatalog(
            ModuleContractError::UnregisteredModule { .. }
        ))
    ));
    assert!(matches!(
        run(
            &f,
            &env(cand(act("nav", "teleport", V1, 0, &f.space), vec![1])),
            Terminal::Ongoing
        ),
        Err(CandidateRejection::OutOfCatalog(
            ModuleContractError::UnregisteredOperation { .. }
        ))
    ));
    assert!(matches!(
        run(
            &f,
            &env(cand(
                act("nav", "move", CapabilityVersion::new(2, 0, 0), 0, &f.space),
                vec![1]
            )),
            Terminal::Ongoing
        ),
        Err(CandidateRejection::OutOfCatalog(
            ModuleContractError::IncompatibleVersion { .. }
        ))
    ));
    assert!(matches!(
        run(
            &f,
            &env(cand(act("nav", "fs-write", V1, 0, &f.space), vec![1])),
            Terminal::Ongoing
        ),
        Err(CandidateRejection::OutOfCatalog(
            ModuleContractError::PermissionDenied { .. }
        ))
    ));
    assert_eq!(
        run(
            &f,
            &env(cand(
                CandidateBody::Goal {
                    name: "conquer".into()
                },
                vec![1]
            )),
            Terminal::Ongoing
        ),
        Err(CandidateRejection::UnknownGoal {
            name: "conquer".into()
        })
    );
}

#[test]
fn guard_rejects_invalid_actions_and_terminal_state() {
    let f = fx();
    // Index 5 is outside the 3-action space.
    let r = run(
        &f,
        &env(cand(act("nav", "move", V1, 5, &f.space), vec![1])),
        Terminal::Ongoing,
    );
    assert!(
        matches!(r, Err(CandidateRejection::GuardDenied { reason: Some(_) })),
        "{r:?}"
    );
    let r = run(
        &f,
        &env(cand(act("nav", "move", V1, 1, &f.space), vec![1])),
        Terminal::Terminated(TerminalReason::Failure),
    );
    assert!(
        matches!(r, Err(CandidateRejection::GuardDenied { reason: Some(_) })),
        "{r:?}"
    );
}

#[test]
fn guard_fallback_is_not_a_usable_generated_action() {
    let mut f = fx();
    f.guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(
        Action::discrete_in_space(0, &f.space).unwrap(),
    ));
    let r = run(
        &f,
        &env(cand(act("nav", "move", V1, 5, &f.space), vec![1])),
        Terminal::Ongoing,
    );
    // The fallback replaces the candidate; the generated candidate must not be reported as usable.
    assert!(
        matches!(r, Err(CandidateRejection::GuardDenied { reason: None })),
        "{r:?}"
    );
}

#[test]
fn valid_candidates_pass_and_actions_only_come_guarded() {
    let f = fx();
    match run(
        &f,
        &env(cand(act("nav", "move", V1, 2, &f.space), vec![1, 2])),
        Terminal::Ongoing,
    ) {
        Ok(ValidatedCandidate::Action(g)) => {
            assert!(f.space.validate(g.action()).is_ok());
            assert_eq!(g.binding().operation_id.as_str(), "move");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        run(
            &f,
            &env(cand(
                CandidateBody::Goal {
                    name: "explore".into()
                },
                vec![2]
            )),
            Terminal::Ongoing
        ),
        Ok(ValidatedCandidate::Goal("explore".into()))
    );
    assert_eq!(
        run(
            &f,
            &env(cand(
                CandidateBody::Notice {
                    text: "door open".into()
                },
                vec![1]
            )),
            Terminal::Ongoing
        ),
        Ok(ValidatedCandidate::Notice("door open".into()))
    );
}

#[test]
fn no_action_without_guard_across_a_sweep() {
    // Sweep many candidates (valid and hallucinated indices, with and without grounding): every
    // Ok(Action) must satisfy the action space, and any out-of-space or terminal case never yields one.
    let f = fx();
    for idx in 0..8 {
        for ev in [vec![1], vec![999], vec![]] {
            for term in [
                Terminal::Ongoing,
                Terminal::Terminated(TerminalReason::Cancelled),
            ] {
                let r = run(
                    &f,
                    &env(cand(act("nav", "move", V1, idx, &f.space), ev.clone())),
                    term,
                );
                if let Ok(ValidatedCandidate::Action(g)) = r {
                    assert!(idx < 3 && ev == vec![1] && term == Terminal::Ongoing);
                    assert!(f.space.validate(g.action()).is_ok());
                }
            }
        }
    }
}

#[test]
fn detections_validated_by_label_catalog_and_evidence() {
    let f = fx();
    let v = CandidateValidator::new(&f.catalog, &f.guard)
        .with_labels(["door"])
        .with_evidence([EvidenceRef::new(1)]);
    let mk = |l: &str, e| VisualDetection {
        label: l.into(),
        confidence: gogamen_core::Confidence::new(0.9).unwrap(),
        evidence: EvidenceRef::new(e),
    };
    assert_eq!(v.validate_detection(&mk("door", 1)), Ok(()));
    assert_eq!(
        v.validate_detection(&mk("unicorn", 1)),
        Err(DetectionRejection::UnknownLabel("unicorn".into()))
    );
    assert_eq!(
        v.validate_detection(&mk("door", 5)),
        Err(DetectionRejection::UngroundedDetection)
    );
}
