# gogamen-jev

Opt-in adapter that implements the core `SystemOneProvider` port over HTTP (GOG-37).

## Status and honesty statement

**No verified Jev documentation, endpoint or credentials were available.** The HTTP contract
below (`gogamen-jev/1`) is a **hypothesis designed in this repository**. It is *not* known to
match the real Jev service. Treat every path, header, field name and model id as provisional
until checked against real documentation; the real smoke (below) is the intended check and is
`NOT_RUN` unless you provide an endpoint.

## Layout and features

| Build | Contents |
|---|---|
| default | Contract logic, circuit breaker, redaction, Replay fallback, `JevTransport` trait. **No network code**, no HTTP client. |
| `--features http` | `UreqTransport` (`ureq` 3, no TLS: only loopback `http://` with opt-in is accepted). |
| `--features https` | Adds rustls TLS to `ureq`; required for any `https://` endpoint. Not exercised by the offline tests. |

**Endpoint policy (GOG-140):** remote `http://` is always refused; loopback `http://`
(`localhost`, `127.0.0.0/8`, `[::1]`) needs `JevConfig::allow_insecure_loopback_http = true`
(default `false`, even without a key); `https://` needs feature `https`; userinfo in the URL is
rejected. A refused endpoint yields `JevError::InsecureEndpoint` before the key is read and the
Replay fallback answers.

`gogamen-core` and `gogamen-cli` do not depend on this crate; their dependency trees stay free
of network crates.

## Contract `gogamen-jev/1` (hypothesis)

* Auth: header `Authorization: Bearer <key>` (header name and scheme configurable). The key is
  read lazily from an environment variable (`EnvSecret`) or any `SecretProvider`; it is never
  stored in config, `Debug` output, errors, events or request bodies.
* Every request also sends `X-Gogamen-Contract: gogamen-jev/1`.
* **Capability discovery**: `GET {base}/v1/capabilities` ->
  `{"contract_version":"gogamen-jev/1","models":["..."],"query_kinds":["choice","noul","score"]}`.
  Cached; the adapter refuses to query unless the **pinned model** and the query kind are
  advertised.
* **Query**: `POST {base}/v1/system-one/query` with
  `{"contract_version","model","timeout_ms","query":{"query_id","kind","options"|"hypothesis_id"}}`.
  Only typed identifiers are sent; a `TypedQuery` carries no free-form human text.
* **Answer**: `{"contract_version","model","query_id","status":"answered|unknown|timed_out",
  "selected"?,"verdict"?,"scores"?{opt:number},"distribution"?{opt:prob}}`. The model echoed must
  equal the pin. The result is validated by core (`SystemOneResponse::new_validated`).

## Behaviour

* **Timeout**: budget = `min(config.timeout, query deadline)` on the injected monotonic clock;
  a call is never started with an empty budget, the transport gets the remaining budget, and a
  late answer is discarded.
* **Typed errors** (`JevError`): `Timeout`, `Auth{401|403}`, `RateLimited{retry_after}` (delta
  seconds only), `ServerError`, `UnexpectedStatus`, `Transport`, `MalformedJson`, `Schema`,
  `ContractVersionMismatch`, `CapabilityMismatch`, `ModelMismatch`, `MissingCredentials`,
  `CircuitOpen`. They hold only static labels, codes and durations.
* **Circuit breaker**: opens after N consecutive failures, immediately on auth/capability
  errors and on 429 (for `Retry-After`, capped); one half-open trial after the cool-down.
* **Fallback**: every failure degrades automatically to the injected `ReplaySystemOneProvider`;
  with no fixture, a timeout yields a `TimedOut` response and other causes a typed
  `SystemOneError::InvalidResponse` (never a fabricated answer). `query_detailed` reports
  `QuerySource::{Live, Fallback(cause)}`.
* **Provenance / guard**: `query_envelope` returns `ObservationEnvelope<SystemOneResponse>`
  (live answers are `Assertion` with confidence = top probability). The adapter has no way to
  build an authorized `ActionProposal`; `ActionGuard` remains the only path to actuation.
* **Redaction**: bodies are never copied into errors/logs; `JevHttpProvider`, `HttpRequest`,
  `Secret`, `EnvSecret` have redacting `Debug` impls (tests assert key and raw text absence).

## Tests

```
cargo test -p gogamen-jev                  # unit tests, scripted transport, no network code
cargo test -p gogamen-jev --features http  # + contract tests vs. a mock server on 127.0.0.1
cargo run  -p gogamen-jev --features http --example jev_smoke
```

The smoke example needs `GOGAMEN_JEV_URL` and `GOGAMEN_JEV_KEY` (optional
`GOGAMEN_JEV_MODEL`), sends one synthetic query, and otherwise prints `NOT_RUN` (not a pass).

## Limitations

* Contract unverified against the real service (see above). Visual/Box input is out of scope.
* Plain HTTP only unless `https` is enabled (compiled but not tested here).
* `Retry-After` HTTP-date form is not parsed. One connection per request (no pooling).
* A new `ureq::Agent` is built per call; cheap but not optimized.
