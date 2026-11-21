//! Typed, redacted error taxonomy for the Jev adapter.
//!
//! Errors carry only static labels, status codes and durations. They never embed response
//! bodies, request text or credentials, so `Debug`/`Display` are safe to log.

use std::fmt;
use std::time::Duration;

/// Failure class of one live Jev call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JevError {
    /// No credential could be resolved from the configured secret source (no call was made).
    MissingCredentials,
    /// The circuit breaker is open; no call was made.
    CircuitOpen {
        /// Remaining cool-down on the injected monotonic clock.
        retry_in: Duration,
    },
    /// Strict monotonic budget exhausted (transport timeout or late answer discarded).
    Timeout,
    /// HTTP 401 or 403.
    Auth {
        /// Status code received.
        status: u16,
    },
    /// HTTP 429, with the parsed `Retry-After` (delta-seconds form only).
    RateLimited {
        /// Parsed `Retry-After`, if present and numeric.
        retry_after: Option<Duration>,
    },
    /// HTTP 5xx.
    ServerError {
        /// Status code received.
        status: u16,
    },
    /// Any other non-2xx status.
    UnexpectedStatus {
        /// Status code received.
        status: u16,
    },
    /// Connection-level failure (refused, reset, DNS, TLS...).
    Transport,
    /// Body is not valid JSON.
    MalformedJson,
    /// Valid JSON that violates the contract schema or the typed query.
    Schema {
        /// Static description of the violated rule.
        reason: &'static str,
    },
    /// The service answered with a different contract version.
    ContractVersionMismatch,
    /// The service does not advertise the pinned model or the query kind.
    CapabilityMismatch {
        /// Static description of the missing capability.
        reason: &'static str,
    },
    /// The response names a different model than the pinned one.
    ModelMismatch,
}

impl JevError {
    /// Static, log-safe description (also used as the `SystemOneError::InvalidResponse` reason).
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::MissingCredentials => "jev: no credential available",
            Self::CircuitOpen { .. } => "jev: circuit breaker open",
            Self::Timeout => "jev: monotonic timeout",
            Self::Auth { .. } => "jev: authentication rejected",
            Self::RateLimited { .. } => "jev: rate limited",
            Self::ServerError { .. } => "jev: server error",
            Self::UnexpectedStatus { .. } => "jev: unexpected http status",
            Self::Transport => "jev: transport failure",
            Self::MalformedJson => "jev: malformed json",
            Self::Schema { reason } | Self::CapabilityMismatch { reason } => reason,
            Self::ContractVersionMismatch => "jev: contract version mismatch",
            Self::ModelMismatch => "jev: model mismatch",
        }
    }

    /// Short stable class label for event logs.
    pub const fn class(&self) -> &'static str {
        match self {
            Self::MissingCredentials => "missing_credentials",
            Self::CircuitOpen { .. } => "circuit_open",
            Self::Timeout => "timeout",
            Self::Auth { .. } => "auth",
            Self::RateLimited { .. } => "rate_limited",
            Self::ServerError { .. } => "server_error",
            Self::UnexpectedStatus { .. } => "unexpected_status",
            Self::Transport => "transport",
            Self::MalformedJson => "malformed_json",
            Self::Schema { .. } => "schema",
            Self::ContractVersionMismatch => "contract_version",
            Self::CapabilityMismatch { .. } => "capability",
            Self::ModelMismatch => "model_mismatch",
        }
    }
}

impl fmt::Display for JevError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auth { status }
            | Self::ServerError { status }
            | Self::UnexpectedStatus { status } => {
                write!(f, "{} (status {status})", self.reason())
            }
            Self::RateLimited {
                retry_after: Some(d),
            } => {
                write!(f, "{} (retry after {}s)", self.reason(), d.as_secs())
            }
            Self::CircuitOpen { retry_in } => {
                write!(f, "{} (retry in {}ms)", self.reason(), retry_in.as_millis())
            }
            _ => f.write_str(self.reason()),
        }
    }
}

impl std::error::Error for JevError {}

/// Failure reported by a [`crate::JevTransport`] before any HTTP status exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportError {
    /// The call exceeded the timeout handed to the transport.
    Timeout,
    /// Connection-level failure.
    Other,
}
