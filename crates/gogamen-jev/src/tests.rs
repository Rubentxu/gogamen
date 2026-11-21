//! Unit tests over a scripted in-memory transport (no network code involved).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gogamen_core::{
    Action, ActionGuard, ActionProposal, ActionSpace, FixtureBehavior, GuardContext, GuardDecision,
    GuardRejectionReason, MonotonicDeadline, MonotonicInstant, OptionId, PlanResult,
    ProvenanceMetadata, QueryId, ReplaySystemOneProvider, SystemOneError, SystemOneFixture,
    SystemOneProvider, Terminal, Tick, TypedQuery, VirtualMonotonicClock,
};
use gogamen_core::{
    Confidence, EvidenceRef, FrameId, MessageId, ObservationKind, SensorId, UnitTag,
};

use crate::*;

const KEY: &str = "sk-test-SECRET-123";
const HUMAN: &str = "HUMAN_TEXT_MARKER pick up the red cup";

#[derive(Clone, Default)]
struct Fake {
    script: Arc<Mutex<VecDeque<Result<HttpResponse, TransportError>>>>,
    seen: Arc<Mutex<Vec<HttpRequest>>>,
}

impl Fake {
    fn push(&self, r: Result<HttpResponse, TransportError>) -> &Self {
        self.script.lock().unwrap().push_back(r);
        self
    }
    fn calls(&self) -> usize {
        self.seen.lock().unwrap().len()
    }
}

impl JevTransport for Fake {
    fn send(&self, req: &HttpRequest, _t: Duration) -> Result<HttpResponse, TransportError> {
        self.seen.lock().unwrap().push(req.clone());
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .expect("unscripted call")
    }
}

fn resp(status: u16, body: &str) -> Result<HttpResponse, TransportError> {
    Ok(HttpResponse {
        status,
        retry_after: None,
        body: body.as_bytes().to_vec(),
    })
}

const CAPS: &str = r#"{"contract_version":"gogamen-jev/1","models":["jev-m1"],"query_kinds":["choice","noul","score"]}"#;

fn choice_ok(id: &str) -> String {
    format!(
        r#"{{"contract_version":"gogamen-jev/1","model":"jev-m1","query_id":"{id}","status":"answered","selected":"a","scores":{{"a":2.0,"b":1.0}},"distribution":{{"a":0.75,"b":0.25}}}}"#
    )
}

fn q(id: &str) -> TypedQuery {
    TypedQuery::choice(
        QueryId::new(id).unwrap(),
        vec![OptionId::new("a").unwrap(), OptionId::new("b").unwrap()],
        None,
    )
    .unwrap()
}

fn replay_for(id: &str) -> ReplaySystemOneProvider {
    let prov = ProvenanceMetadata {
        message_id: MessageId::new(9),
        observed_at: MonotonicInstant::ZERO,
        received_at: MonotonicInstant::ZERO,
        sensor_id: SensorId::new(99),
        frame_id: FrameId::world(),
        units: UnitTag::dimensionless(),
        confidence: Confidence::FULL,
        revision: 0,
        evidence_ref: EvidenceRef::new(1),
        claim_kind: ObservationKind::Proposal,
    };
    ReplaySystemOneProvider::new("replay").with_fixture(
        QueryId::new(id).unwrap(),
        SystemOneFixture::new(
            FixtureBehavior::Choice {
                selected: OptionId::new("b").unwrap(),
                raw_scores: vec![
                    (OptionId::new("a").unwrap(), 0.0),
                    (OptionId::new("b").unwrap(), 1.0),
                ],
                validated_distribution: None,
            },
            prov,
        ),
    )
}

fn provider(
    fake: &Fake,
    key: Option<&str>,
    replay: ReplaySystemOneProvider,
) -> JevHttpProvider<Fake> {
    let mut cfg = JevConfig::new("http://jev.invalid/", "jev-m1");
    cfg.breaker.failure_threshold = 2;
    cfg.breaker.cooldown = Duration::from_secs(10);
    let secret: Box<dyn SecretProvider> = match key {
        Some(k) => Box::new(StaticSecret::some(k)),
        None => Box::new(StaticSecret::none()),
    };
    JevHttpProvider::new(cfg, fake.clone(), secret, replay)
}

#[test]
fn success_discovers_then_queries_and_wraps_envelope() {
    let fake = Fake::default();
    fake.push(resp(200, CAPS)).push(resp(200, &choice_ok("q1")));
    let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
    let clock = VirtualMonotonicClock::new();
    let env = p.query_envelope(&q("q1"), &clock).unwrap();
    assert_eq!(env.payload().selected_option().unwrap().as_str(), "a");
    assert_eq!(env.sensor_id(), SensorId::new(1));
    assert_eq!(env.claim_kind(), ObservationKind::Assertion);
    assert!((env.confidence().as_f64() - 0.75).abs() < 1e-12);
    let seen = fake.seen.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].url, "http://jev.invalid/v1/capabilities");
    assert_eq!(seen[1].url, "http://jev.invalid/v1/system-one/query");
    let auth = seen[1]
        .headers
        .iter()
        .find(|(n, _)| n == "Authorization")
        .unwrap();
    assert_eq!(auth.1, format!("Bearer {KEY}"));
    assert!(
        seen[1]
            .headers
            .iter()
            .any(|(n, v)| n == "X-Gogamen-Contract" && v == "gogamen-jev/1")
    );
    // Request body carries typed ids only and the pinned model.
    let body = String::from_utf8(seen[1].body.clone()).unwrap();
    assert!(body.contains(r#""model":"jev-m1""#) && !body.contains(KEY));
    // Debug of the request never prints header values.
    assert!(!format!("{:?}", seen[1]).contains(KEY));
    drop(seen);
    // Discovery is cached: second query is a single POST.
    fake.push(resp(200, &choice_ok("q2")));
    p.query(&q("q2"), &clock).unwrap();
    assert_eq!(fake.calls(), 3);
}

#[test]
fn auth_failure_falls_back_opens_breaker_then_half_opens() {
    let fake = Fake::default();
    fake.push(resp(401, "denied"));
    let p = provider(&fake, Some(KEY), replay_for("q1"));
    let clock = VirtualMonotonicClock::new();
    let out = p.query_detailed(&q("q1"), &clock);
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::Auth { status: 401 })
    );
    assert_eq!(out.result.unwrap().selected_option().unwrap().as_str(), "b");
    // Open: no further transport calls.
    let out = p.query_detailed(&q("q1"), &clock);
    assert!(matches!(
        out.source,
        QuerySource::Fallback(JevError::CircuitOpen { .. })
    ));
    assert_eq!(fake.calls(), 1);
    // After the cool-down one trial is allowed again.
    clock.advance(Duration::from_secs(11));
    fake.push(resp(200, CAPS)).push(resp(200, &choice_ok("q1")));
    assert_eq!(p.query_detailed(&q("q1"), &clock).source, QuerySource::Live);
    assert_eq!(fake.calls(), 3);
}

#[test]
fn rate_limit_honours_retry_after() {
    let fake = Fake::default();
    fake.push(Ok(HttpResponse {
        status: 429,
        retry_after: Some("7".into()),
        body: vec![],
    }));
    let p = provider(&fake, Some(KEY), replay_for("q1"));
    let clock = VirtualMonotonicClock::new();
    let out = p.query_detailed(&q("q1"), &clock);
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::RateLimited {
            retry_after: Some(Duration::from_secs(7))
        })
    );
    clock.advance(Duration::from_secs(6));
    assert!(matches!(
        p.query_detailed(&q("q1"), &clock).source,
        QuerySource::Fallback(JevError::CircuitOpen { .. })
    ));
    assert_eq!(fake.calls(), 1);
}

#[test]
fn breaker_opens_after_threshold_of_server_errors() {
    let fake = Fake::default();
    fake.push(resp(503, "")).push(resp(503, ""));
    let p = provider(&fake, Some(KEY), replay_for("q1"));
    let clock = VirtualMonotonicClock::new();
    for _ in 0..2 {
        assert_eq!(
            p.query_detailed(&q("q1"), &clock).source,
            QuerySource::Fallback(JevError::ServerError { status: 503 })
        );
    }
    assert!(matches!(
        p.query_detailed(&q("q1"), &clock).source,
        QuerySource::Fallback(JevError::CircuitOpen { .. })
    ));
}

#[test]
fn malformed_and_schema_violations_are_typed_and_redacted() {
    let cases: [(&str, JevError); 6] = [
        (
            "{not json HUMAN_TEXT_MARKER sk-test-SECRET-123",
            JevError::MalformedJson,
        ),
        (
            "[]",
            JevError::Schema {
                reason: "jev: response is not an object",
            },
        ),
        (
            r#"{"contract_version":"gogamen-jev/9","model":"jev-m1"}"#,
            JevError::ContractVersionMismatch,
        ),
        (
            r#"{"contract_version":"gogamen-jev/1","model":"other","query_id":"q1","status":"answered"}"#,
            JevError::ModelMismatch,
        ),
        (
            r#"{"contract_version":"gogamen-jev/1","model":"jev-m1","query_id":"q1","status":"maybe"}"#,
            JevError::Schema {
                reason: "jev: missing or invalid status",
            },
        ),
        // Answered choice selecting an option that was never offered: core validation rejects it.
        (
            r#"{"contract_version":"gogamen-jev/1","model":"jev-m1","query_id":"q1","status":"answered","selected":"zzz","distribution":{"a":0.5,"b":0.5}}"#,
            JevError::Schema {
                reason: "jev: response violates system one validation",
            },
        ),
    ];
    for (body, expected) in cases {
        let fake = Fake::default();
        fake.push(resp(200, CAPS)).push(resp(200, body));
        let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
        let clock = VirtualMonotonicClock::new();
        let out = p.query_detailed(&q("q1"), &clock);
        assert_eq!(out.source, QuerySource::Fallback(expected.clone()));
        let rendered = format!("{out:?}|{p:?}|{expected}|{:?}", p.events());
        assert!(!rendered.contains(KEY), "key leaked: {rendered}");
        assert!(
            !rendered.contains("HUMAN_TEXT_MARKER"),
            "raw text leaked: {rendered}"
        );
        // No replay fixture: degraded to a typed error, never a fabricated answer.
        assert!(matches!(
            out.result,
            Err(SystemOneError::InvalidResponse { .. })
        ));
    }
}

#[test]
fn missing_credentials_makes_no_call_and_degrades() {
    let fake = Fake::default();
    let p = provider(&fake, None, replay_for("q1"));
    let clock = VirtualMonotonicClock::new();
    let out = p.query_detailed(&q("q1"), &clock);
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::MissingCredentials)
    );
    assert!(out.result.is_ok());
    assert_eq!(fake.calls(), 0);
}

#[test]
fn unadvertised_model_is_a_capability_mismatch() {
    let fake = Fake::default();
    fake.push(resp(
        200,
        r#"{"contract_version":"gogamen-jev/1","models":["other"],"query_kinds":["choice"]}"#,
    ));
    let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
    let out = p.query_detailed(&q("q1"), &VirtualMonotonicClock::new());
    assert!(matches!(
        out.source,
        QuerySource::Fallback(JevError::CapabilityMismatch { .. })
    ));
}

#[test]
fn expired_deadline_never_calls_and_yields_timed_out() {
    let fake = Fake::default();
    let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
    let clock = VirtualMonotonicClock::new();
    let query = q("q1").with_deadline(MonotonicDeadline::from_now(&clock, Duration::ZERO));
    let out = p.query_detailed(&query, &clock);
    assert_eq!(out.source, QuerySource::Fallback(JevError::Timeout));
    assert!(out.result.unwrap().status().is_timed_out());
    assert_eq!(fake.calls(), 0);
}

#[test]
fn transport_timeout_maps_to_timeout() {
    let fake = Fake::default();
    fake.push(Err(TransportError::Timeout));
    let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
    let out = p.query_detailed(&q("q1"), &VirtualMonotonicClock::new());
    assert_eq!(out.source, QuerySource::Fallback(JevError::Timeout));
    assert!(out.result.unwrap().status().is_timed_out());
}

#[test]
fn env_secret_debug_shows_only_variable_name() {
    let s = EnvSecret::new("GOGAMEN_JEV_KEY_UNSET_FOR_TEST");
    assert!(s.secret().is_none());
    assert!(format!("{s:?}").contains("GOGAMEN_JEV_KEY_UNSET_FOR_TEST"));
    assert_eq!(format!("{:?}", Secret::new(KEY)), "Secret([REDACTED])");
    assert_eq!(redact(&format!("x {KEY} y"), KEY), "x [REDACTED] y");
    let _ = HUMAN;
}

#[test]
fn jev_envelope_cannot_authorize_an_action_guard_still_decides() {
    let fake = Fake::default();
    fake.push(resp(200, CAPS)).push(resp(200, &choice_ok("q1")));
    let p = provider(&fake, Some(KEY), ReplaySystemOneProvider::default());
    let env = p
        .query_envelope(&q("q1"), &VirtualMonotonicClock::new())
        .unwrap();
    assert!(env.payload().status().is_answered());
    // The adapter exposes no way to mint an authorized proposal; a proposal derived from "no
    // valid plan" is rejected by the independent guard regardless of what Jev answered.
    let space = ActionSpace::discrete(2).unwrap();
    let ctx = GuardContext::new(
        &space,
        Tick::new(1),
        std::time::Instant::now(),
        Terminal::Ongoing,
    );
    let proposal = ActionProposal::from_plan_result(&PlanResult::no_valid_action(0), Tick::new(1));
    assert_eq!(
        ActionGuard::strict().evaluate(&proposal, &ctx),
        GuardDecision::Reject(GuardRejectionReason::Unauthorized)
    );
    let _ = Action::discrete_in_space(0, &space).unwrap();
}
