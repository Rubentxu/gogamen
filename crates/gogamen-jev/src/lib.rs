//! Decoupled, opt-in Jev adapter for the core `SystemOneProvider` port (GOG-37).
//!
//! * Default build: contract logic, breaker, redaction, Replay fallback and a transport trait,
//!   with **no** network code. `gogamen-core` and `gogamen-cli` do not depend on this crate.
//! * Feature `http`: plain-HTTP transport on `ureq`. Feature `https`: adds TLS.
//!
//! The HTTP contract (`gogamen-jev/1`) is a **hypothesis** defined by this repo; no verified
//! Jev documentation was available. See the crate README.

pub mod breaker;
pub mod error;
pub mod provider;
pub mod secret;
pub mod transport;
pub mod wire;

pub use breaker::{BreakerConfig, CircuitBreaker};
pub use error::{JevError, TransportError};
pub use provider::{JevConfig, JevEvent, JevHttpProvider, JevOutcome, QuerySource};
pub use secret::{EnvSecret, Secret, SecretProvider, StaticSecret, redact};
#[cfg(feature = "http")]
pub use transport::http::UreqTransport;
pub use transport::{HttpRequest, HttpResponse, JevTransport, Method};
pub use wire::{CONTRACT_VERSION, Capabilities};

#[cfg(test)]
mod tests;
