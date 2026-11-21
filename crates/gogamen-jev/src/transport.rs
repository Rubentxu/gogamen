//! Transport port. The provider is generic over it so the contract logic is testable with no
//! network code compiled in.

use std::fmt;
use std::time::Duration;

use crate::error::TransportError;

/// HTTP verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// `GET`
    Get,
    /// `POST`
    Post,
}

/// Outgoing request. `Debug` prints header *names* only (values may hold the API key).
#[derive(Clone)]
pub struct HttpRequest {
    /// Verb.
    pub method: Method,
    /// Absolute URL.
    pub url: String,
    /// Header name/value pairs.
    pub headers: Vec<(String, String)>,
    /// Body bytes (empty for GET).
    pub body: Vec<u8>,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(n, _)| n.as_str()).collect();
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("header_names", &names)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// Incoming response (only what the contract needs).
#[derive(Clone)]
pub struct HttpResponse {
    /// Status code.
    pub status: u16,
    /// Raw `Retry-After` header value.
    pub retry_after: Option<String>,
    /// Body bytes. Never logged.
    pub body: Vec<u8>,
}

impl fmt::Debug for HttpResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpResponse")
            .field("status", &self.status)
            .field("body_len", &self.body.len())
            .finish()
    }
}

/// Blocking transport with a hard per-call timeout.
pub trait JevTransport: Send + Sync {
    /// Sends `request`, returning any HTTP status (including 4xx/5xx) as `Ok`.
    fn send(
        &self,
        request: &HttpRequest,
        timeout: Duration,
    ) -> Result<HttpResponse, TransportError>;
}

/// Plain-HTTP transport on `ureq` (feature `http`).
#[cfg(feature = "http")]
pub mod http {
    use super::{HttpRequest, HttpResponse, JevTransport, Method, TransportError};
    use std::io::Read;
    use std::time::Duration;

    /// `ureq`-backed transport. Redirects are disabled and 4xx/5xx are returned as responses.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct UreqTransport;

    impl JevTransport for UreqTransport {
        fn send(
            &self,
            request: &HttpRequest,
            timeout: Duration,
        ) -> Result<HttpResponse, TransportError> {
            let agent: ureq::Agent = ureq::Agent::config_builder()
                .timeout_global(Some(timeout))
                .http_status_as_error(false)
                .max_redirects(0)
                .build()
                .into();
            let result = match request.method {
                Method::Get => {
                    let mut r = agent.get(&request.url);
                    for (n, v) in &request.headers {
                        r = r.header(n, v);
                    }
                    r.call()
                }
                Method::Post => {
                    let mut r = agent.post(&request.url);
                    for (n, v) in &request.headers {
                        r = r.header(n, v);
                    }
                    r.send(&request.body[..])
                }
            };
            let mut response = result.map_err(|e| match e {
                ureq::Error::Timeout(_) => TransportError::Timeout,
                ureq::Error::Io(ref io)
                    if matches!(
                        io.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) =>
                {
                    TransportError::Timeout
                }
                _ => TransportError::Other,
            })?;
            let status = response.status().as_u16();
            let retry_after = response
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned);
            let mut body = Vec::new();
            response
                .body_mut()
                .as_reader()
                .take(1 << 20)
                .read_to_end(&mut body)
                .map_err(|e| {
                    if matches!(
                        e.kind(),
                        std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
                    ) {
                        TransportError::Timeout
                    } else {
                        TransportError::Other
                    }
                })?;
            Ok(HttpResponse {
                status,
                retry_after,
                body,
            })
        }
    }
}
