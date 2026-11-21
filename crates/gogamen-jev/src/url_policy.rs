//! Endpoint policy for `JevConfig::base_url` (GOG-140). Std-only URL inspection.
//!
//! The bearer key travels in a header, so a cleartext remote endpoint would leak it:
//!
//! * `https://` is accepted only when the crate is built with feature `https` (TLS).
//! * `http://` to a **non-loopback** host is always rejected (with or without credentials).
//! * `http://` to loopback (`localhost`, `127.0.0.0/8`, `[::1]`) requires the explicit opt-in
//!   `JevConfig::allow_insecure_loopback_http` (default `false`), even when no key is sent.
//! * userinfo (`user@host`) is rejected, so `http://127.0.0.1@evil.com` cannot masquerade.
//! * Any other scheme, empty host, bad port, whitespace or control char is rejected.
//!
//! Rejection reasons are static strings: the URL itself is never echoed.

use std::net::{Ipv4Addr, Ipv6Addr};

/// Checks `base_url` against the policy. `https_enabled` says whether TLS support is compiled in.
pub fn check_base_url(
    base_url: &str,
    allow_insecure_loopback_http: bool,
    https_enabled: bool,
) -> Result<(), &'static str> {
    if base_url
        .chars()
        .any(|c| c.is_whitespace() || c.is_control() || c == '\\')
    {
        return Err("jev: endpoint url contains forbidden characters");
    }
    let (scheme, rest) = base_url
        .split_once("://")
        .ok_or("jev: endpoint url has no scheme")?;
    let scheme = scheme.to_ascii_lowercase();
    if scheme != "http" && scheme != "https" {
        return Err("jev: endpoint scheme must be http or https");
    }
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains('@') {
        return Err("jev: endpoint url must not contain userinfo");
    }
    let host = host_of(authority)?;
    if scheme == "https" {
        return if https_enabled {
            Ok(())
        } else {
            Err("jev: https endpoint requires the `https` cargo feature")
        };
    }
    if !is_loopback(host) {
        return Err("jev: cleartext http to a remote host is refused (use https)");
    }
    if allow_insecure_loopback_http {
        Ok(())
    } else {
        Err("jev: loopback http requires allow_insecure_loopback_http")
    }
}

/// Extracts the host from `host[:port]` / `[v6][:port]`, validating the port.
fn host_of(authority: &str) -> Result<&str, &'static str> {
    const BAD: &str = "jev: endpoint url has an invalid authority";
    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        let (host, after) = inner.split_once(']').ok_or(BAD)?;
        let port = match after {
            "" => None,
            p => Some(p.strip_prefix(':').ok_or(BAD)?),
        };
        (host, port)
    } else {
        match authority.rsplit_once(':') {
            Some((h, p)) => (h, Some(p)),
            None => (authority, None),
        }
    };
    if host.is_empty() || host.contains([':', '[', ']']) && !authority.starts_with('[') {
        return Err(BAD);
    }
    if let Some(p) = port
        && (p.is_empty() || p.len() > 5 || !p.bytes().all(|b| b.is_ascii_digit()) || {
            p.parse::<u32>().map_or(true, |n| n > 65535)
        })
    {
        return Err(BAD);
    }
    Ok(host)
}

fn is_loopback(host: &str) -> bool {
    let host = host.to_ascii_lowercase();
    if host == "localhost" {
        return true;
    }
    if let Ok(v4) = host.parse::<Ipv4Addr>() {
        return v4.is_loopback();
    }
    host.parse::<Ipv6Addr>().is_ok_and(|v6| v6.is_loopback())
}
