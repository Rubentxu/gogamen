//! Equivalencia Rust <-> Kotlin DSL del Language World (`GOG-72`, AC #1).
//!
//! El fichero dorado `language_world_canonical_v0.json` lo emite el fixture Rust
//! (`canonical_language_world_fixture_json`) y el test Kotlin
//! (`EpistemicDslTest`) comprueba byte a byte que el DSL Kotlin produce el mismo JSON.
//! Aquí se comprueba que el fichero dorado es exactamente el fixture Rust y que su
//! ejecución en el runtime reproduce la traza del fixture por seed, sin JVM.

use gogamen_core::MasterSeed;
use gogamen_runtime::PlanRuntimeEngine;
use gogamen_spec::{
    SystemSpec, canonical_language_world_fixture, canonical_language_world_fixture_json,
};

const GOLDEN_JSON: &str =
    include_str!("../../gogamen-spec/fixtures/language_world_canonical_v0.json");

#[test]
fn golden_json_equals_rust_fixture_and_replays_same_trace_per_seed() {
    assert_eq!(GOLDEN_JSON, canonical_language_world_fixture_json());

    let golden_spec = SystemSpec::from_json(GOLDEN_JSON).expect("golden json must parse");
    assert_eq!(golden_spec, canonical_language_world_fixture());

    let engine = PlanRuntimeEngine::new();
    let mut hashes = Vec::new();
    for seed in [42_u64, 99, 2026] {
        let from_rust = engine
            .run_spec_with_seed(
                &canonical_language_world_fixture(),
                10,
                MasterSeed::new(seed),
            )
            .expect("rust fixture must run");
        let from_golden = engine
            .run_json_with_seed(GOLDEN_JSON, 10, MasterSeed::new(seed))
            .expect("golden json must run");
        assert_eq!(
            from_rust.semantic_fingerprint,
            from_golden.semantic_fingerprint
        );
        assert_eq!(from_rust.trace_hash, from_golden.trace_hash);
        assert_eq!(from_rust.final_outputs, from_golden.final_outputs);
        hashes.push(from_golden.trace_hash);
    }
    assert_ne!(hashes[0], hashes[1], "distinct seeds must change the trace");
}
