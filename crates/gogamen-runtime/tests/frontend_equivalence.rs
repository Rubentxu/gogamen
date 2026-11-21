use std::time::Instant;

use gogamen_compiler::{CompiledPlanArtifact, SpecCompiler};
use gogamen_core::{MasterSeed, TemporalProfile};
use gogamen_runtime::{PlanRuntimeEngine, builtin_plan_catalog};
use gogamen_spec::{
    SystemSpec, canonical_cognitive_agent_fixture, canonical_cognitive_agent_fixture_json,
    canonical_two_node_fixture, canonical_two_node_fixture_json,
};

const KOTLIN_TWO_NODE_CANONICAL_JSON: &str = r#"{"config":{"deterministic":true,"dt_secs":0.02},"connections":[{"delay_ticks":0,"from_node":"sensor","from_port":"observation","to_node":"controller","to_port":"observation"}],"nodes":[{"config":{"normalized":true,"rate_hz":50},"id":"sensor","kind":"observer","ports":[{"direction":"output","id":"observation","port_type":"f64"}]},{"config":{"gain":0.5,"mode":"greedy"},"id":"controller","kind":"policy","ports":[{"direction":"input","id":"observation","port_type":"f64"},{"direction":"output","id":"action","port_type":"i64"}]}],"schema_version":"gogamen.system_spec.v0","spec_id":"two_node_canonical_v0"}"#;

const KOTLIN_COGNITIVE_AGENT_CANONICAL_JSON: &str = r#"{"config":{"deterministic":true,"dt_secs":0.02},"connections":[{"delay_ticks":0,"from_node":"sensor","from_port":"observation","to_node":"belief","to_port":"observation"},{"delay_ticks":0,"from_node":"sensor","from_port":"energy","to_node":"belief","to_port":"energy"},{"delay_ticks":0,"from_node":"belief","from_port":"belief_score","to_node":"planner","to_port":"belief_score"},{"delay_ticks":0,"from_node":"belief","from_port":"should_explore","to_node":"planner","to_port":"should_explore"}],"nodes":[{"config":{"base_energy":1.0,"noise_scale":0.1},"id":"sensor","kind":"agent::sensor","ports":[{"direction":"output","id":"observation","port_type":"f64"},{"direction":"output","id":"energy","port_type":"f64"}]},{"config":{"curiosity":0.6,"expr_json":"{\"op\":\"gt\",\"lhs\":{\"op\":\"mul\",\"lhs\":{\"op\":\"var\",\"name\":\"energy\"},\"rhs\":{\"op\":\"var\",\"name\":\"curiosity\"}},\"rhs\":{\"op\":\"var\",\"name\":\"threshold\"}}","threshold":0.4},"id":"belief","kind":"agent::belief","ports":[{"direction":"input","id":"observation","port_type":"f64"},{"direction":"input","id":"energy","port_type":"f64"},{"direction":"output","id":"belief_score","port_type":"f64"},{"direction":"output","id":"should_explore","port_type":"bool"}]},{"config":{"exploit_gain":1.0,"explore_gain":2.0},"id":"planner","kind":"agent::planner","ports":[{"direction":"input","id":"belief_score","port_type":"f64"},{"direction":"input","id":"should_explore","port_type":"bool"},{"direction":"output","id":"action","port_type":"i64"}]}],"schema_version":"gogamen.system_spec.v0","spec_id":"cognitive_agent_canonical_v0"}"#;

const NON_ZERO_COST_NOTICE: &str = "Aviso: no se promete coste cero de abstracción; se miden por separado autoría, compilación, carga y ejecución.";

#[derive(Debug)]
struct StageTimingsNs {
    authoring_ns: u128,
    compile_ns: u128,
    load_ns: u128,
    run_ns: u128,
}

fn verify_frontend_equivalence(
    label: &str,
    rust_builder: fn() -> SystemSpec,
    direct_json: &str,
    kotlin_json: &str,
    seeds: &[u64],
    ticks: u64,
    expect_seed_sensitivity: bool,
) {
    let catalog = builtin_plan_catalog();
    let engine = PlanRuntimeEngine::new();

    // --- 1. Autoría (medida por separado) ---
    let t_auth_rust = Instant::now();
    let rust_spec = rust_builder();
    let rust_auth_ns = t_auth_rust.elapsed().as_nanos();

    let t_auth_json = Instant::now();
    let json_spec = SystemSpec::from_json(direct_json).expect("direct json must parse");
    let json_auth_ns = t_auth_json.elapsed().as_nanos();

    let t_auth_kt = Instant::now();
    let kotlin_spec = SystemSpec::from_json(kotlin_json).expect("kotlin json must parse");
    let kotlin_auth_ns = t_auth_kt.elapsed().as_nanos();

    assert_eq!(rust_spec, json_spec);
    assert_eq!(rust_spec, kotlin_spec);

    // --- 2. Compilación (SpecCompiler::compile + CompiledPlanArtifact::compile) ---
    let rust_plan_direct =
        SpecCompiler::compile(&rust_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("rust SpecCompiler::compile");
    let json_plan_direct =
        SpecCompiler::compile(&json_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("json SpecCompiler::compile");
    let kotlin_plan_direct =
        SpecCompiler::compile(&kotlin_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("kotlin SpecCompiler::compile");

    assert_eq!(
        rust_plan_direct.semantic_fingerprint(),
        json_plan_direct.semantic_fingerprint()
    );
    assert_eq!(
        rust_plan_direct.semantic_fingerprint(),
        kotlin_plan_direct.semantic_fingerprint()
    );

    let t_comp_rust = Instant::now();
    let rust_artifact =
        CompiledPlanArtifact::compile(&rust_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("rust compile artifact");
    let rust_compile_ns = t_comp_rust.elapsed().as_nanos();

    let t_comp_json = Instant::now();
    let json_artifact =
        CompiledPlanArtifact::compile(&json_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("json compile artifact");
    let json_compile_ns = t_comp_json.elapsed().as_nanos();

    let t_comp_kt = Instant::now();
    let kotlin_artifact =
        CompiledPlanArtifact::compile(&kotlin_spec, &catalog, TemporalProfile::BoundedTick)
            .expect("kotlin compile artifact");
    let kotlin_compile_ns = t_comp_kt.elapsed().as_nanos();

    // AC #1: exactamente el mismo artifact_hash y spec_hash (y semantic_fingerprint en planes compilados/cargados)
    assert_eq!(rust_artifact.spec_hash(), json_artifact.spec_hash());
    assert_eq!(rust_artifact.spec_hash(), kotlin_artifact.spec_hash());
    assert_eq!(rust_artifact.artifact_hash(), json_artifact.artifact_hash());
    assert_eq!(
        rust_artifact.artifact_hash(),
        kotlin_artifact.artifact_hash()
    );

    let rust_artifact_json = rust_artifact
        .to_canonical_json()
        .expect("rust artifact json");
    let json_artifact_json = json_artifact
        .to_canonical_json()
        .expect("json artifact json");
    let kotlin_artifact_json = kotlin_artifact
        .to_canonical_json()
        .expect("kotlin artifact json");
    assert_eq!(rust_artifact_json, json_artifact_json);
    assert_eq!(rust_artifact_json, kotlin_artifact_json);

    // --- 3. Carga (load_executable_plan) ---
    let t_load_rust = Instant::now();
    let rust_loaded = rust_artifact
        .load_executable_plan(&catalog)
        .expect("load rust executable plan");
    let rust_load_ns = t_load_rust.elapsed().as_nanos();

    let t_load_json = Instant::now();
    let json_loaded = json_artifact
        .load_executable_plan(&catalog)
        .expect("load json executable plan");
    let json_load_ns = t_load_json.elapsed().as_nanos();

    let t_load_kt = Instant::now();
    let kotlin_loaded = kotlin_artifact
        .load_executable_plan(&catalog)
        .expect("load kotlin executable plan");
    let kotlin_load_ns = t_load_kt.elapsed().as_nanos();

    assert_eq!(
        rust_loaded.semantic_fingerprint(),
        rust_plan_direct.semantic_fingerprint()
    );
    assert_eq!(
        rust_loaded.semantic_fingerprint(),
        json_loaded.semantic_fingerprint()
    );
    assert_eq!(
        rust_loaded.semantic_fingerprint(),
        kotlin_loaded.semantic_fingerprint()
    );

    // --- 4. Ejecución sin JVM por múltiples seeds (AC #2) ---
    let mut observed_hashes_by_seed = Vec::with_capacity(seeds.len());
    let mut rust_run_ns_total = 0u128;
    let mut json_run_ns_total = 0u128;
    let mut kotlin_run_ns_total = 0u128;

    for &seed_u64 in seeds {
        let seed = MasterSeed::new(seed_u64);

        let t_run_rust = Instant::now();
        let rust_summary = engine
            .run_plan_with_seed(&rust_loaded, ticks, seed)
            .expect("rust run_plan_with_seed");
        rust_run_ns_total += t_run_rust.elapsed().as_nanos();

        let t_run_json = Instant::now();
        let json_summary = engine
            .run_plan_with_seed(&json_loaded, ticks, seed)
            .expect("json run_plan_with_seed");
        json_run_ns_total += t_run_json.elapsed().as_nanos();

        let t_run_kt = Instant::now();
        let kotlin_summary = engine
            .run_plan_with_seed(&kotlin_loaded, ticks, seed)
            .expect("kotlin run_plan_with_seed");
        kotlin_run_ns_total += t_run_kt.elapsed().as_nanos();

        // También verificamos la ruta completa run_artifact_json_with_seed sin JVM.
        let artifact_run_rust = engine
            .run_artifact_json_with_seed(&rust_artifact_json, ticks, seed)
            .expect("run_artifact_json_with_seed rust");
        let artifact_run_kt = engine
            .run_artifact_json_with_seed(&kotlin_artifact_json, ticks, seed)
            .expect("run_artifact_json_with_seed kotlin");

        assert_eq!(rust_summary, json_summary);
        assert_eq!(rust_summary, kotlin_summary);
        assert_eq!(rust_summary, artifact_run_rust);
        assert_eq!(rust_summary, artifact_run_kt);

        observed_hashes_by_seed.push(rust_summary.trace_hash);
    }

    if expect_seed_sensitivity && seeds.len() >= 2 {
        assert_ne!(
            observed_hashes_by_seed[0], observed_hashes_by_seed[1],
            "{label}: different seeds should produce distinct deterministic traces"
        );
    }

    let rust_timings = StageTimingsNs {
        authoring_ns: rust_auth_ns,
        compile_ns: rust_compile_ns,
        load_ns: rust_load_ns,
        run_ns: rust_run_ns_total,
    };
    let json_timings = StageTimingsNs {
        authoring_ns: json_auth_ns,
        compile_ns: json_compile_ns,
        load_ns: json_load_ns,
        run_ns: json_run_ns_total,
    };
    let kotlin_timings = StageTimingsNs {
        authoring_ns: kotlin_auth_ns,
        compile_ns: kotlin_compile_ns,
        load_ns: kotlin_load_ns,
        run_ns: kotlin_run_ns_total,
    };

    // AC #3: reporte separado de tiempos + aviso explícito
    for (frontend, t) in [
        ("rust_api", &rust_timings),
        ("direct_json", &json_timings),
        ("kotlin_json", &kotlin_timings),
    ] {
        println!(
            "[{label}::{frontend}] authoring_ns={} compile_ns={} load_ns={} run_ns={} | {NON_ZERO_COST_NOTICE}",
            t.authoring_ns, t.compile_ns, t.load_ns, t.run_ns
        );
    }
    assert!(NON_ZERO_COST_NOTICE.contains("no se promete coste cero de abstracción"));
}

#[test]
fn rust_json_and_kotlin_frontends_are_equivalent_for_general_and_cognitive_systems() {
    let seeds = [42_u64, 7_u64, 2026_u64];

    verify_frontend_equivalence(
        "two_node_canonical_v0",
        canonical_two_node_fixture,
        &canonical_two_node_fixture_json(),
        KOTLIN_TWO_NODE_CANONICAL_JSON,
        &seeds,
        25,
        false,
    );

    verify_frontend_equivalence(
        "cognitive_agent_canonical_v0",
        canonical_cognitive_agent_fixture,
        &canonical_cognitive_agent_fixture_json(),
        KOTLIN_COGNITIVE_AGENT_CANONICAL_JSON,
        &seeds,
        25,
        true,
    );
}
