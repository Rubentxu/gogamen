//! Secret handling: credentials are resolved lazily from the environment (or an injected
//! provider) and are never printed.

use std::fmt;

/// An opaque credential. `Debug` is always redacted and there is no `Display`.
#[derive(Clone)]
pub struct Secret(String);

impl Secret {
    /// Wraps a credential value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Exposes the raw value. Only the transport layer should call this.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret([REDACTED])")
    }
}

/// Source of the API key (environment variable, vault client, test double...).
pub trait SecretProvider: Send + Sync + fmt::Debug {
    /// Returns the current credential, if any. Empty values count as absent.
    fn secret(&self) -> Option<Secret>;
}

/// Reads the credential from a named environment variable at call time.
#[derive(Debug, Clone)]
pub struct EnvSecret {
    var: String,
}

impl EnvSecret {
    /// Creates a provider reading `var` (the variable *name* is not secret).
    pub fn new(var: impl Into<String>) -> Self {
        Self { var: var.into() }
    }
}

impl SecretProvider for EnvSecret {
    fn secret(&self) -> Option<Secret> {
        std::env::var(&self.var)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(Secret::new)
    }
}

/// Fixed in-memory credential (tests and injected vault clients).
#[derive(Debug, Clone)]
pub struct StaticSecret(Option<Secret>);

impl StaticSecret {
    /// Provider that always yields `value`.
    pub fn some(value: impl Into<String>) -> Self {
        Self(Some(Secret::new(value)))
    }

    /// Provider that never yields a credential.
    pub const fn none() -> Self {
        Self(None)
    }
}

impl SecretProvider for StaticSecret {
    fn secret(&self) -> Option<Secret> {
        self.0.clone()
    }
}

/// Replaces every occurrence of `secret` in `text` with `[REDACTED]`.
///
/// Defensive helper for free-form diagnostics; the adapter itself never builds messages from
/// response bodies.
pub fn redact(text: &str, secret: &str) -> String {
    if secret.is_empty() {
        return text.to_owned();
    }
    text.replace(secret, "[REDACTED]")
}
