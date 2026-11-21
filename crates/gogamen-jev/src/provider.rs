//! `JevHttpProvider`: [`SystemOneProvider`] over the `gogamen-jev/1` HTTP contract.

use std::sync::Mutex;
use std::time::Duration;

use gogamen_core::{
    Confidence, EvidenceRef, FrameId, MessageId, MonotonicClock, MonotonicInstant, NoulVerdict,
    ObservationEnvelope, ObservationKind, OptionId, ProvenanceMetadata, RawScores,
    ReplaySystemOneProvider, ResponseStatus, SensorId, SystemOneError, SystemOnePayload,
    SystemOneProvider, SystemOneResponse, TypedQuery, UnitTag, ValidatedDistribution,
};

use crate::breaker::{BreakerConfig, CircuitBreaker};
use crate::error::{JevError, TransportError};
use crate::secret::{Secret, SecretProvider};
use crate::transport::{HttpRequest, HttpResponse, JevTransport, Method};
use crate::wire::{
    self, CAPABILITIES_PATH, CONTRACT_VERSION, Capabilities, ParsedAnswer, QUERY_PATH,
};

/// Adapter configuration. Contains no secret (the key comes from a [`SecretProvider`]).
#[derive(Debug, Clone)]
pub struct JevConfig {
    /// Base URL without trailing slash, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// Pinned model identifier; discovery must advertise it and every answer must echo it.
    pub model: String,
    /// Header carrying the credential (default `Authorization`).
    pub auth_header: String,
    /// Prefix placed before the key in that header (default `Bearer `).
    pub auth_scheme: String,
    /// Upper bound for one query, including discovery. Shrunk by the query deadline.
    pub timeout: Duration,
    /// `SensorId` stamped on live provenance.
    pub sensor_id: u64,
    /// Circuit breaker tuning.
    pub breaker: BreakerConfig,
}

impl JevConfig {
    /// Defaults: `Authorization: Bearer <key>`, 2 s budget, sensor id 1.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            auth_header: "Authorization".to_owned(),
            auth_scheme: "Bearer ".to_owned(),
            timeout: Duration::from_secs(2),
            sensor_id: 1,
            breaker: BreakerConfig::default(),
        }
    }
}

/// Where an answer came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuerySource {
    /// Served by the live service.
    Live,
    /// Served by the Replay fallback because of `cause`.
    Fallback(JevError),
}

/// Result of [`JevHttpProvider::query_detailed`].
#[derive(Debug, Clone)]
pub struct JevOutcome {
    /// Typed answer (live or fallback).
    pub result: Result<SystemOneResponse, SystemOneError>,
    /// Origin of the answer.
    pub source: QuerySource,
}

/// Log-safe event: static labels only (no ids, bodies or keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JevEvent {
    /// Monotonic sequence number.
    pub seq: u64,
    /// `"live"` or `"fallback"`.
    pub source: &'static str,
    /// Error class when fallback, `"ok"` otherwise.
    pub class: &'static str,
}

const MAX_EVENTS: usize = 256;

#[derive(Debug)]
struct State {
    breaker: CircuitBreaker,
    capabilities: Option<Capabilities>,
    events: Vec<JevEvent>,
    seq: u64,
}

/// Jev provider with strict timeout, circuit breaker and automatic Replay fallback.
pub struct JevHttpProvider<T: JevTransport> {
    config: JevConfig,
    transport: T,
    secret: Box<dyn SecretProvider>,
    fallback: ReplaySystemOneProvider,
    state: Mutex<State>,
}

impl<T: JevTransport> std::fmt::Debug for JevHttpProvider<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevHttpProvider")
            .field("config", &self.config)
            .field("secret", &self.secret)
            .field("fallback_fixtures", &self.fallback.fixture_count())
            .finish_non_exhaustive()
    }
}

impl<T: JevTransport> JevHttpProvider<T> {
    /// Builds the provider.
    pub fn new(
        config: JevConfig,
        transport: T,
        secret: Box<dyn SecretProvider>,
        fallback: ReplaySystemOneProvider,
    ) -> Self {
        let breaker = CircuitBreaker::new(config.breaker);
        Self {
            config,
            transport,
            secret,
            fallback,
            state: Mutex::new(State {
                breaker,
                capabilities: None,
                events: Vec::new(),
                seq: 0,
            }),
        }
    }

    /// Log-safe event history (bounded).
    pub fn events(&self) -> Vec<JevEvent> {
        self.lock().events.clone()
    }

    /// Capabilities cached by the last successful discovery.
    pub fn capabilities(&self) -> Option<Capabilities> {
        self.lock().capabilities.clone()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Like [`SystemOneProvider::query`] but reports the origin of the answer.
    pub fn query_detailed(&self, query: &TypedQuery, clock: &dyn MonotonicClock) -> JevOutcome {
        if let Err(e) = query.validate() {
            return JevOutcome {
                result: Err(e),
                source: QuerySource::Live,
            };
        }
        let start = clock.now();
        let mut budget = self.config.timeout;
        if let Some(deadline) = query.deadline() {
            budget = budget.min(deadline.remaining_at(start));
        }
        if budget.is_zero() {
            // Strict: never start a call that cannot finish inside its monotonic budget.
            return self.degrade(query, clock, JevError::Timeout);
        }
        let allowed = self.lock().breaker.allow(start);
        if let Err(retry_in) = allowed {
            return self.degrade(query, clock, JevError::CircuitOpen { retry_in });
        }
        match self.live(query, clock, start, budget) {
            Ok(resp) => {
                let mut st = self.lock();
                st.breaker.on_success();
                Self::push_event(&mut st, "live", "ok");
                JevOutcome {
                    result: Ok(resp),
                    source: QuerySource::Live,
                }
            }
            Err(err) => {
                let now = clock.now();
                {
                    let mut st = self.lock();
                    match &err {
                        JevError::MissingCredentials => {}
                        JevError::RateLimited { retry_after } => st.breaker.on_failure(
                            now,
                            Some(retry_after.unwrap_or(self.config.breaker.cooldown)),
                        ),
                        JevError::Auth { .. } | JevError::CapabilityMismatch { .. } => st
                            .breaker
                            .on_failure(now, Some(self.config.breaker.cooldown)),
                        _ => st.breaker.on_failure(now, None),
                    }
                }
                self.degrade(query, clock, err)
            }
        }
    }

    /// Answer wrapped as an [`ObservationEnvelope`] carrying its provenance. The envelope is an
    /// observation (assertion/proposal), never an authorization: it cannot construct an
    /// `ActionProposal`, so `ActionGuard` stays the only path to actuation.
    pub fn query_envelope(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
    ) -> Result<ObservationEnvelope<SystemOneResponse>, SystemOneError> {
        let response = self.query_detailed(query, clock).result?;
        Ok(ObservationEnvelope::new(
            response.provenance().clone(),
            response,
        ))
    }

    fn push_event(st: &mut State, source: &'static str, class: &'static str) {
        st.seq += 1;
        if st.events.len() == MAX_EVENTS {
            st.events.remove(0);
        }
        let seq = st.seq;
        st.events.push(JevEvent { seq, source, class });
    }

    fn degrade(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
        cause: JevError,
    ) -> JevOutcome {
        Self::push_event(&mut self.lock(), "fallback", cause.class());
        let result = match self.fallback.query(query, clock) {
            Ok(r) => Ok(r),
            Err(SystemOneError::MissingFixture { .. }) => {
                if matches!(cause, JevError::Timeout) {
                    self.synth_timed_out(query, clock)
                } else {
                    Err(SystemOneError::InvalidResponse {
                        reason: cause.reason(),
                    })
                }
            }
            Err(other) => Err(other),
        };
        JevOutcome {
            result,
            source: QuerySource::Fallback(cause),
        }
    }

    fn synth_timed_out(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
    ) -> Result<SystemOneResponse, SystemOneError> {
        let now = clock.now();
        let prov = self.provenance(query, now, now, Confidence::ZERO, ObservationKind::Unknown);
        let payload = match query {
            TypedQuery::Choice { .. } => SystemOnePayload::Choice { selected: None },
            TypedQuery::Score { .. } => SystemOnePayload::Score,
            TypedQuery::Noul { hypothesis_id, .. } => SystemOnePayload::Noul {
                hypothesis_id: hypothesis_id.clone(),
                verdict: NoulVerdict::Unknown,
            },
        };
        SystemOneResponse::new_validated(
            query,
            ResponseStatus::TimedOut,
            payload,
            prov,
            None,
            RawScores::empty(),
        )
    }

    fn provenance(
        &self,
        query: &TypedQuery,
        observed_at: MonotonicInstant,
        received_at: MonotonicInstant,
        confidence: Confidence,
        claim_kind: ObservationKind,
    ) -> ProvenanceMetadata {
        let seq = self.lock().seq;
        ProvenanceMetadata {
            message_id: MessageId::new(seq + 1),
            observed_at,
            received_at,
            sensor_id: SensorId::new(self.config.sensor_id),
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence,
            revision: 0,
            evidence_ref: EvidenceRef::new(fnv1a(&[
                self.config.model.as_bytes(),
                query.query_id().as_str().as_bytes(),
                &(seq + 1).to_le_bytes(),
            ])),
            claim_kind,
        }
    }

    fn live(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
        start: MonotonicInstant,
        budget: Duration,
    ) -> Result<SystemOneResponse, JevError> {
        let secret = self.secret.secret().ok_or(JevError::MissingCredentials)?;
        let deadline_at = start.saturating_add(budget);

        let cached = self.lock().capabilities.clone();
        let caps = match cached {
            Some(c) => c,
            None => {
                let body = self.call(
                    Method::Get,
                    CAPABILITIES_PATH,
                    Vec::new(),
                    &secret,
                    clock,
                    deadline_at,
                )?;
                let c = wire::parse_capabilities(&body)?;
                self.lock().capabilities = Some(c.clone());
                c
            }
        };
        if !caps.models.iter().any(|m| m == &self.config.model) {
            // Re-discover next time: the service may have rolled its catalogue.
            self.lock().capabilities = None;
            return Err(JevError::CapabilityMismatch {
                reason: "jev: pinned model not advertised",
            });
        }
        if !caps
            .query_kinds
            .iter()
            .any(|k| k == wire::kind_label(query))
        {
            return Err(JevError::CapabilityMismatch {
                reason: "jev: query kind not advertised",
            });
        }

        let remaining = deadline_at.saturating_duration_since(clock.now());
        let body = wire::build_query_body(&self.config.model, query, remaining.as_millis());
        let raw = self.call(Method::Post, QUERY_PATH, body, &secret, clock, deadline_at)?;
        let parsed = wire::parse_query_response(&raw, query, &self.config.model)?;
        self.assemble(query, parsed, start, clock.now())
    }

    fn call(
        &self,
        method: Method,
        path: &str,
        body: Vec<u8>,
        secret: &Secret,
        clock: &dyn MonotonicClock,
        deadline_at: MonotonicInstant,
    ) -> Result<Vec<u8>, JevError> {
        let remaining = deadline_at.saturating_duration_since(clock.now());
        if remaining.is_zero() {
            return Err(JevError::Timeout);
        }
        let mut headers = vec![
            (
                self.config.auth_header.clone(),
                format!("{}{}", self.config.auth_scheme, secret.expose()),
            ),
            ("Accept".to_owned(), "application/json".to_owned()),
            ("X-Gogamen-Contract".to_owned(), CONTRACT_VERSION.to_owned()),
        ];
        if method == Method::Post {
            headers.push(("Content-Type".to_owned(), "application/json".to_owned()));
        }
        let request = HttpRequest {
            method,
            url: format!("{}{path}", self.config.base_url),
            headers,
            body,
        };
        let response: HttpResponse =
            self.transport
                .send(&request, remaining)
                .map_err(|e| match e {
                    TransportError::Timeout => JevError::Timeout,
                    TransportError::Other => JevError::Transport,
                })?;
        // A late answer is discarded even if the transport let it through.
        if clock.now() > deadline_at {
            return Err(JevError::Timeout);
        }
        match response.status {
            200..=299 => Ok(response.body),
            401 | 403 => Err(JevError::Auth {
                status: response.status,
            }),
            429 => Err(JevError::RateLimited {
                retry_after: response
                    .retry_after
                    .as_deref()
                    .and_then(wire::parse_retry_after),
            }),
            500..=599 => Err(JevError::ServerError {
                status: response.status,
            }),
            status => Err(JevError::UnexpectedStatus { status }),
        }
    }

    fn assemble(
        &self,
        query: &TypedQuery,
        parsed: ParsedAnswer,
        observed_at: MonotonicInstant,
        received_at: MonotonicInstant,
    ) -> Result<SystemOneResponse, JevError> {
        const INVALID: JevError = JevError::Schema {
            reason: "jev: response violates system one validation",
        };
        let raw = RawScores::new(parsed.scores).map_err(|_| INVALID)?;
        let distribution = parsed
            .distribution
            .map(ValidatedDistribution::new)
            .transpose()
            .map_err(|_| INVALID)?;
        let payload = match query {
            TypedQuery::Choice { .. } => SystemOnePayload::Choice {
                selected: parsed.selected,
            },
            TypedQuery::Score { .. } => SystemOnePayload::Score,
            TypedQuery::Noul { hypothesis_id, .. } => SystemOnePayload::Noul {
                hypothesis_id: hypothesis_id.clone(),
                verdict: parsed.verdict.unwrap_or(NoulVerdict::Unknown),
            },
        };
        let top: f64 = distribution
            .as_ref()
            .map(|d| {
                d.entries()
                    .iter()
                    .map(|(_, p): &(OptionId, f64)| *p)
                    .fold(0.0, f64::max)
            })
            .unwrap_or(0.0);
        let confidence = Confidence::new(top.clamp(0.0, 1.0)).unwrap_or(Confidence::ZERO);
        let kind = if parsed.status.is_answered() {
            ObservationKind::Assertion
        } else {
            ObservationKind::Unknown
        };
        let prov = self.provenance(query, observed_at, received_at, confidence, kind);
        SystemOneResponse::new_validated(query, parsed.status, payload, prov, distribution, raw)
            .map_err(|_| INVALID)
    }
}

impl<T: JevTransport> SystemOneProvider for JevHttpProvider<T> {
    fn provider_id(&self) -> &str {
        "jev-http"
    }

    fn query(
        &self,
        query: &TypedQuery,
        clock: &dyn MonotonicClock,
    ) -> Result<SystemOneResponse, SystemOneError> {
        self.query_detailed(query, clock).result
    }
}

fn fnv1a(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for part in parts {
        for b in *part {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    h
}
