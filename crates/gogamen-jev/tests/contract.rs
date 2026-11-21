//! Contract tests against a local mock server on 127.0.0.1 (no external network).
//! Run with `cargo test -p gogamen-jev --features http`.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use gogamen_core::{
    OptionId, QueryId, ReplaySystemOneProvider, SystemMonotonicClock, SystemOneError, TypedQuery,
};
use gogamen_jev::{JevConfig, JevError, JevHttpProvider, QuerySource, StaticSecret, UreqTransport};

const KEY: &str = "sk-mock-SECRET-456";
const CAPS: &str = r#"{"contract_version":"gogamen-jev/1","models":["jev-m1"],"query_kinds":["choice","noul","score"]}"#;

#[derive(Clone)]
struct Reply {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: String,
    delay: Duration,
}

fn ok(body: &str) -> Reply {
    Reply {
        status: 200,
        headers: vec![],
        body: body.to_owned(),
        delay: Duration::ZERO,
    }
}

fn status(code: u16, body: &str) -> Reply {
    Reply {
        status: code,
        ..ok(body)
    }
}

#[derive(Debug, Clone)]
struct Seen {
    request_line: String,
    headers: Vec<(String, String)>,
    body: String,
}

struct Mock {
    base: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

fn handle(mut s: TcpStream, replies: &Mutex<Vec<Reply>>, seen: &Mutex<Vec<Seen>>) {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    let head_end = loop {
        let n = s.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            return;
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("").to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| {
            l.split_once(':')
                .map(|(n, v)| (n.trim().to_lowercase(), v.trim().to_owned()))
        })
        .collect();
    let len = headers
        .iter()
        .find(|(n, _)| n == "content-length")
        .and_then(|(_, v)| v.parse::<usize>().ok())
        .unwrap_or(0);
    while buf.len() < head_end + len {
        let n = s.read(&mut tmp).unwrap_or(0);
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
    seen.lock().unwrap().push(Seen {
        request_line,
        headers,
        body,
    });
    let reply = {
        let mut r = replies.lock().unwrap();
        if r.is_empty() {
            status(500, "script exhausted")
        } else {
            r.remove(0)
        }
    };
    thread::sleep(reply.delay);
    let mut out = format!(
        "HTTP/1.1 {} X\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/json\r\n",
        reply.status,
        reply.body.len()
    );
    for (n, v) in &reply.headers {
        out.push_str(&format!("{n}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(&reply.body);
    let _ = s.write_all(out.as_bytes());
}

/// Serves the scripted replies in order, one connection each, from a detached thread.
fn mock(replies: Vec<Reply>) -> Mock {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (seen2, replies) = (Arc::clone(&seen), Arc::new(Mutex::new(replies)));
    thread::spawn(move || {
        for conn in listener.incoming().flatten() {
            let (r, s) = (Arc::clone(&replies), Arc::clone(&seen2));
            thread::spawn(move || handle(conn, &r, &s));
        }
    });
    Mock { base, seen }
}

fn query() -> TypedQuery {
    TypedQuery::choice(
        QueryId::new("q1").unwrap(),
        vec![OptionId::new("a").unwrap(), OptionId::new("b").unwrap()],
        None,
    )
    .unwrap()
}

fn provider(base: &str, timeout: Duration) -> JevHttpProvider<UreqTransport> {
    let mut cfg = JevConfig::new(base, "jev-m1");
    cfg.timeout = timeout;
    JevHttpProvider::new(
        cfg,
        UreqTransport,
        Box::new(StaticSecret::some(KEY)),
        ReplaySystemOneProvider::default(),
    )
}

fn assert_clean(rendered: &str) {
    assert!(!rendered.contains(KEY), "key leaked: {rendered}");
    assert!(
        !rendered.contains("HUMAN_TEXT_MARKER"),
        "raw text leaked: {rendered}"
    );
}

#[test]
fn success_over_real_http() {
    let answer = r#"{"contract_version":"gogamen-jev/1","model":"jev-m1","query_id":"q1","status":"answered","selected":"b","scores":{"a":0.0,"b":1.0},"distribution":{"a":0.25,"b":0.75}}"#;
    let m = mock(vec![ok(CAPS), ok(answer)]);
    let p = provider(&m.base, Duration::from_secs(5));
    let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
    assert_eq!(out.source, QuerySource::Live);
    assert_eq!(out.result.unwrap().selected_option().unwrap().as_str(), "b");
    let seen = m.seen.lock().unwrap();
    assert!(seen[0].request_line.starts_with("GET /v1/capabilities"));
    assert!(
        seen[1]
            .request_line
            .starts_with("POST /v1/system-one/query")
    );
    let h = |n: &str| {
        seen[1]
            .headers
            .iter()
            .find(|(k, _)| k == n)
            .map(|(_, v)| v.clone())
    };
    assert_eq!(h("authorization").unwrap(), format!("Bearer {KEY}"));
    assert_eq!(h("x-gogamen-contract").unwrap(), "gogamen-jev/1");
    assert!(seen[1].body.contains(r#""model":"jev-m1""#));
    assert!(!seen[1].body.contains(KEY));
}

#[test]
fn timeout_is_strict_and_degrades() {
    let slow = Reply {
        delay: Duration::from_millis(1500),
        ..ok(CAPS)
    };
    let m = mock(vec![slow]);
    let p = provider(&m.base, Duration::from_millis(250));
    let t0 = Instant::now();
    let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
    let elapsed = t0.elapsed();
    assert_eq!(out.source, QuerySource::Fallback(JevError::Timeout));
    assert!(out.result.unwrap().status().is_timed_out());
    assert!(
        elapsed < Duration::from_millis(1200),
        "timeout not enforced: {elapsed:?}"
    );
}

#[test]
fn auth_errors_map_to_auth() {
    for code in [401u16, 403] {
        let m = mock(vec![status(code, "HUMAN_TEXT_MARKER nope")]);
        let p = provider(&m.base, Duration::from_secs(5));
        let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
        assert_eq!(
            out.source,
            QuerySource::Fallback(JevError::Auth { status: code })
        );
        assert!(matches!(
            out.result,
            Err(SystemOneError::InvalidResponse { .. })
        ));
        assert_clean(&format!("{out:?}{p:?}{:?}", p.events()));
    }
}

#[test]
fn rate_limit_carries_retry_after_and_opens_breaker() {
    let limited = Reply {
        headers: vec![("Retry-After", "12".into())],
        ..status(429, "slow down")
    };
    let m = mock(vec![ok(CAPS), limited]);
    let p = provider(&m.base, Duration::from_secs(5));
    let clock = SystemMonotonicClock::new();
    let out = p.query_detailed(&query(), &clock);
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::RateLimited {
            retry_after: Some(Duration::from_secs(12))
        })
    );
    // Breaker is open: a second query makes no further HTTP call.
    let calls = m.seen.lock().unwrap().len();
    let out = p.query_detailed(&query(), &clock);
    assert!(matches!(
        out.source,
        QuerySource::Fallback(JevError::CircuitOpen { .. })
    ));
    assert_eq!(m.seen.lock().unwrap().len(), calls);
}

#[test]
fn malformed_bodies_map_to_typed_errors_without_leaking() {
    let leaky = format!("{{\"echo\": \"HUMAN_TEXT_MARKER {KEY}\"");
    let cases = [
        (leaky.as_str(), JevError::MalformedJson),
        (
            r#"{"contract_version":"gogamen-jev/1","model":"jev-m1","query_id":"q1","status":"answered","selected":"a"}"#,
            JevError::Schema {
                reason: "jev: response violates system one validation",
            },
        ),
    ];
    for (body, expected) in cases {
        let m = mock(vec![ok(CAPS), ok(body)]);
        let p = provider(&m.base, Duration::from_secs(5));
        let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
        assert_eq!(out.source, QuerySource::Fallback(expected));
        assert_clean(&format!("{out:?}{p:?}{:?}", p.events()));
    }
}

#[test]
fn server_error_and_refused_connection_degrade_to_replay() {
    let m = mock(vec![status(503, "")]);
    let p = provider(&m.base, Duration::from_secs(5));
    let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::ServerError { status: 503 })
    );

    // Port that was bound and released: connection refused, no network involved.
    let dead = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        format!("http://{}", l.local_addr().unwrap())
    };
    let p = provider(&dead, Duration::from_secs(2));
    let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
    assert!(matches!(
        out.source,
        QuerySource::Fallback(JevError::Transport | JevError::Timeout)
    ));
}

#[test]
fn without_key_the_adapter_makes_no_request() {
    let m = mock(vec![]);
    let cfg = JevConfig::new(&m.base, "jev-m1");
    let p = JevHttpProvider::new(
        cfg,
        UreqTransport,
        Box::new(StaticSecret::none()),
        ReplaySystemOneProvider::default(),
    );
    let out = p.query_detailed(&query(), &SystemMonotonicClock::new());
    assert_eq!(
        out.source,
        QuerySource::Fallback(JevError::MissingCredentials)
    );
    assert!(m.seen.lock().unwrap().is_empty());
}
