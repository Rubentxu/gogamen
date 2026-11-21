//! Wire format of the (hypothetical) `gogamen-jev/1` contract. See the crate README.

use gogamen_core::{NoulVerdict, OptionId, ResponseStatus, TypedQuery};
use serde_json::{Map, Value, json};

use crate::error::JevError;

/// Contract identifier sent in `X-Gogamen-Contract` and in every body.
pub const CONTRACT_VERSION: &str = "gogamen-jev/1";
/// Capability discovery path (GET).
pub const CAPABILITIES_PATH: &str = "/v1/capabilities";
/// Query path (POST).
pub const QUERY_PATH: &str = "/v1/system-one/query";

/// Result of capability discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capabilities {
    /// Models the service offers.
    pub models: Vec<String>,
    /// Supported query kinds (`choice`, `noul`, `score`).
    pub query_kinds: Vec<String>,
}

/// Query kind label used on the wire.
pub const fn kind_label(query: &TypedQuery) -> &'static str {
    match query {
        TypedQuery::Choice { .. } => "choice",
        TypedQuery::Noul { .. } => "noul",
        TypedQuery::Score { .. } => "score",
    }
}

/// Builds the request body. Only typed identifiers leave the process: no free-form human text
/// exists in a [`TypedQuery`], so none can be transmitted.
pub fn build_query_body(model: &str, query: &TypedQuery, timeout_ms: u128) -> Vec<u8> {
    let mut q = Map::new();
    q.insert("query_id".into(), json!(query.query_id().as_str()));
    q.insert("kind".into(), json!(kind_label(query)));
    match query {
        TypedQuery::Choice { options, .. } | TypedQuery::Score { options, .. } => {
            let ids: Vec<&str> = options.iter().map(OptionId::as_str).collect();
            q.insert("options".into(), json!(ids));
        }
        TypedQuery::Noul { hypothesis_id, .. } => {
            q.insert("hypothesis_id".into(), json!(hypothesis_id.as_str()));
        }
    }
    let body = json!({
        "contract_version": CONTRACT_VERSION,
        "model": model,
        "timeout_ms": u64::try_from(timeout_ms).unwrap_or(u64::MAX),
        "query": Value::Object(q),
    });
    body.to_string().into_bytes()
}

fn parse_json(body: &[u8]) -> Result<Value, JevError> {
    serde_json::from_slice(body).map_err(|_| JevError::MalformedJson)
}

const fn schema(reason: &'static str) -> JevError {
    JevError::Schema { reason }
}

fn check_version(v: &Value) -> Result<(), JevError> {
    match v.get("contract_version").and_then(Value::as_str) {
        Some(CONTRACT_VERSION) => Ok(()),
        Some(_) => Err(JevError::ContractVersionMismatch),
        None => Err(schema("jev: missing contract_version")),
    }
}

fn string_list(v: &Value, key: &'static str) -> Result<Vec<String>, JevError> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or(schema("jev: capabilities field missing or not an array"))?
        .iter()
        .map(|e| {
            e.as_str()
                .map(str::to_owned)
                .ok_or(schema("jev: capabilities entry is not a string"))
        })
        .collect()
}

/// Parses a capabilities document.
pub fn parse_capabilities(body: &[u8]) -> Result<Capabilities, JevError> {
    let v = parse_json(body)?;
    check_version(&v)?;
    Ok(Capabilities {
        models: string_list(&v, "models")?,
        query_kinds: string_list(&v, "query_kinds")?,
    })
}

/// Parsed (not yet core-validated) answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedAnswer {
    /// Resolution status.
    pub status: ResponseStatus,
    /// Selected option (`choice`).
    pub selected: Option<OptionId>,
    /// Verdict (`noul`).
    pub verdict: Option<NoulVerdict>,
    /// Raw scores in key order.
    pub scores: Vec<(OptionId, f64)>,
    /// Normalized distribution in key order.
    pub distribution: Option<Vec<(OptionId, f64)>>,
}

fn option(s: &str) -> Result<OptionId, JevError> {
    OptionId::new(s).map_err(|_| schema("jev: empty option id"))
}

fn number_map(v: &Value, key: &'static str) -> Result<Option<Vec<(OptionId, f64)>>, JevError> {
    let Some(field) = v.get(key) else {
        return Ok(None);
    };
    if field.is_null() {
        return Ok(None);
    }
    let map = field
        .as_object()
        .ok_or(schema("jev: scores/distribution is not an object"))?;
    map.iter()
        .map(|(k, n)| {
            let n = n
                .as_f64()
                .ok_or(schema("jev: score/probability is not a number"))?;
            Ok((option(k)?, n))
        })
        .collect::<Result<Vec<_>, JevError>>()
        .map(Some)
}

/// Parses a query response for `query`, requiring the pinned `model`.
pub fn parse_query_response(
    body: &[u8],
    query: &TypedQuery,
    model: &str,
) -> Result<ParsedAnswer, JevError> {
    let v = parse_json(body)?;
    if !v.is_object() {
        return Err(schema("jev: response is not an object"));
    }
    check_version(&v)?;
    match v.get("model").and_then(Value::as_str) {
        Some(m) if m == model => {}
        Some(_) => return Err(JevError::ModelMismatch),
        None => return Err(schema("jev: missing model")),
    }
    if v.get("query_id").and_then(Value::as_str) != Some(query.query_id().as_str()) {
        return Err(schema("jev: query_id mismatch"));
    }
    let status = match v.get("status").and_then(Value::as_str) {
        Some("answered") => ResponseStatus::Answered,
        Some("unknown") => ResponseStatus::Unknown,
        Some("timed_out") => ResponseStatus::TimedOut,
        _ => return Err(schema("jev: missing or invalid status")),
    };
    let selected = match v.get("selected") {
        Some(Value::String(s)) => Some(option(s)?),
        Some(Value::Null) | None => None,
        Some(_) => return Err(schema("jev: selected is not a string")),
    };
    let verdict = match v.get("verdict") {
        Some(Value::String(s)) => Some(match s.as_str() {
            "true" => NoulVerdict::True,
            "false" => NoulVerdict::False,
            "unknown" => NoulVerdict::Unknown,
            _ => return Err(schema("jev: invalid verdict")),
        }),
        Some(Value::Null) | None => None,
        Some(_) => return Err(schema("jev: verdict is not a string")),
    };
    Ok(ParsedAnswer {
        status,
        selected,
        verdict,
        scores: number_map(&v, "scores")?.unwrap_or_default(),
        distribution: number_map(&v, "distribution")?,
    })
}

/// Parses a `Retry-After` header in delta-seconds form only (HTTP-date is unsupported).
pub fn parse_retry_after(value: &str) -> Option<std::time::Duration> {
    value
        .trim()
        .parse::<u64>()
        .ok()
        .map(std::time::Duration::from_secs)
}
