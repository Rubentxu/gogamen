//! GOG-159: shared corpus of the Kotlin cognitive DSL, checked with the Rust parser.
//!
//! Positive cases are canonical JSON emitted by the DSL; negative cases are the
//! rejections that the Rust parser can reproduce. Gates only the DSL can enforce
//! (pure leak, privileged + hidden, unit mismatch, matrix size) carry `-` as location.

use gogamen_spec::{
    AdaptiveExperimentSpec, SpecError, canonical_grid_cognitive_spec_experiment_fixture,
};
use std::fs;
use std::path::PathBuf;

fn corpus_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../spec/golden/cognitive_dsl")
}

fn read(rel: &str) -> String {
    fs::read_to_string(corpus_dir().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// FNV-1a 64 over the canonical bytes (local helper; the runtime spec hash lives elsewhere).
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn location(err: &SpecError) -> String {
    match err {
        SpecError::UnknownField { context, field } => format!("UnknownField:{context}.{field}"),
        SpecError::MissingField { context, field } => format!("MissingField:{context}.{field}"),
        SpecError::OverlappingSeeds { seed } => format!("OverlappingSeeds:{seed}"),
        other => format!("Other:{other:?}"),
    }
}

struct Case {
    id: String,
    dsl_code: String,
    dsl_path: String,
    rust_location: String,
}

fn cases() -> Vec<Case> {
    read("cases.tsv")
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            assert_eq!(c.len(), 5, "bad row: {l}");
            Case {
                id: c[0].into(),
                dsl_code: c[1].into(),
                dsl_path: c[2].into(),
                rust_location: c[4].into(),
            }
        })
        .collect()
}

#[test]
fn cognitive_dsl_corpus_h0_is_byte_identical_to_native_fixture() {
    let native = canonical_grid_cognitive_spec_experiment_fixture()
        .to_canonical_json()
        .unwrap();
    let dsl = read("positive/h0.json");
    assert_eq!(dsl, native);
    assert_eq!(fnv1a64(dsl.as_bytes()), fnv1a64(native.as_bytes()));
}

#[test]
fn cognitive_dsl_corpus_positives_parse_and_round_trip() {
    let mut count = 0;
    for i in 0..=7 {
        let json = read(&format!("positive/h{i}.json"));
        let spec = AdaptiveExperimentSpec::from_json(&json)
            .unwrap_or_else(|e| panic!("h{i} rejected by the Rust parser: {e}"));
        assert_eq!(
            spec.to_canonical_json().unwrap(),
            json,
            "h{i} not canonical"
        );
        count += 1;
    }
    assert!(count >= 8);
}

#[test]
fn cognitive_dsl_corpus_negatives_are_rejected_with_diagnostic_and_path() {
    let cases = cases();
    assert!(cases.len() >= 7, "need at least 7 negative cases");
    let mut rust_checked = 0;
    for c in &cases {
        assert!(c.dsl_code.starts_with("E_"), "{}: bad code", c.id);
        assert!(!c.dsl_path.is_empty(), "{}: missing path", c.id);
        if c.rust_location == "-" {
            continue;
        }
        let json = read(&format!("negative/{}.json", c.id));
        let err = AdaptiveExperimentSpec::from_json(&json)
            .expect_err(&format!("{} must be rejected by Rust", c.id));
        assert_eq!(location(&err), c.rust_location, "{}", c.id);
        rust_checked += 1;
    }
    assert!(rust_checked >= 6);
}
