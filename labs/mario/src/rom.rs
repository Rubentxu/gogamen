//! User ROM handling: SHA-256 hashing, known-hash list, typed NOT_RUN reasons.
//! No ROM is distributed; the known-hash list is user configuration.

use sha2::{Digest, Sha256};
use std::fmt;
use std::path::Path;

/// Lowercase hex SHA-256 of `data`.
pub fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Typed reasons for which a run is NOT_RUN (nothing was executed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotRunReason {
    /// No ROM path was supplied.
    NoRomProvided,
    /// ROM hash is in the known list but no emulator backend is compiled in.
    BackendUnavailable { label: String, sha256: String },
}

impl fmt::Display for NotRunReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRomProvided => write!(f, "no_rom_provided"),
            Self::BackendUnavailable { label, sha256 } => write!(
                f,
                "backend_unavailable (rom '{label}' sha256={sha256} is a known hash, but no emulator backend is compiled in)"
            ),
        }
    }
}

/// Typed ROM errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RomError {
    Io {
        path: String,
        reason: String,
    },
    EmptyFile,
    /// The hash is not in the known list (list size included for diagnosis).
    HashNotKnown {
        sha256: String,
        known_count: usize,
    },
    /// Malformed line in the known-hash file.
    BadConfig {
        line: usize,
        reason: &'static str,
    },
}

impl fmt::Display for RomError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, reason } => write!(f, "cannot read '{path}': {reason}"),
            Self::EmptyFile => write!(f, "ROM file is empty"),
            Self::HashNotKnown {
                sha256,
                known_count,
            } => write!(
                f,
                "hash mismatch: sha256={sha256} is not among the {known_count} known integration hash(es)"
            ),
            Self::BadConfig { line, reason } => {
                write!(f, "known-hash config line {line}: {reason}")
            }
        }
    }
}

impl std::error::Error for RomError {}

/// A ROM whose hash matched the known list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RomVerified {
    pub label: String,
    pub sha256: String,
}

/// Configurable list of known integration hashes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct KnownRoms {
    entries: Vec<(String, String)>, // (sha256 lowercase hex, label)
}

impl KnownRoms {
    pub fn parse(text: &str) -> Result<Self, RomError> {
        let mut entries = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut it = line.splitn(3, char::is_whitespace);
            if it.next() != Some("sha256") {
                return Err(RomError::BadConfig {
                    line: i + 1,
                    reason: "expected 'sha256 <hex> <label>'",
                });
            }
            let hex = it.next().unwrap_or("").to_ascii_lowercase();
            if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(RomError::BadConfig {
                    line: i + 1,
                    reason: "hash must be 64 hex chars",
                });
            }
            let label = it.next().unwrap_or("").trim().to_string();
            entries.push((hex, label));
        }
        Ok(Self { entries })
    }

    pub fn load(path: &Path) -> Result<Self, RomError> {
        let text = std::fs::read_to_string(path).map_err(|e| RomError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        Self::parse(&text)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hash `bytes` and check it against the list.
    pub fn verify_bytes(&self, bytes: &[u8]) -> Result<RomVerified, RomError> {
        if bytes.is_empty() {
            return Err(RomError::EmptyFile);
        }
        let sha256 = sha256_hex(bytes);
        match self.entries.iter().find(|(h, _)| *h == sha256) {
            Some((_, label)) => Ok(RomVerified {
                label: label.clone(),
                sha256,
            }),
            None => Err(RomError::HashNotKnown {
                sha256,
                known_count: self.entries.len(),
            }),
        }
    }

    pub fn verify_file(&self, path: &Path) -> Result<RomVerified, RomError> {
        let bytes = std::fs::read(path).map_err(|e| RomError::Io {
            path: path.display().to_string(),
            reason: e.to_string(),
        })?;
        self.verify_bytes(&bytes)
    }
}
