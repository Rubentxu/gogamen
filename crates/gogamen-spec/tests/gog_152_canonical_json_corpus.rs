//! GOG-152: corpus dorado compartido de JSON canonico (Rust <-> Kotlin).
//!
//! Norma: el codec Rust (`codec.rs`/`json.rs`) es la referencia. El MISMO corpus
//! `spec/golden/canonical_json/cases.json` lo consume el test Kotlin `Gog152CanonicalJsonCorpusTest`.

use std::path::PathBuf;

use gogamen_core::{PortSpec, PortType};
use gogamen_spec::{
    ExprAst, ExprUnit, JsonValue, SpecConfigValue, SystemNodeSpec, SystemSpec, parse_json,
};

fn load() -> Vec<JsonValue> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/golden/canonical_json/cases.json");
    let text = std::fs::read_to_string(&path).expect("corpus must exist");
    let JsonValue::Object(mut root) = parse_json(&text).expect("corpus must parse") else {
        panic!("corpus root must be an object")
    };
    assert_eq!(
        root.get("schema"),
        Some(&JsonValue::String(
            "gogamen.golden.canonical_json.v1".to_owned()
        ))
    );
    let Some(JsonValue::Array(cases)) = root.remove("cases") else {
        panic!("cases must be an array")
    };
    cases
}

fn obj(v: &JsonValue) -> &std::collections::BTreeMap<String, JsonValue> {
    match v {
        JsonValue::Object(o) => o,
        other => panic!("expected object, got {other:?}"),
    }
}

fn st(v: &JsonValue) -> &str {
    match v {
        JsonValue::String(s) => s,
        other => panic!("expected string, got {other:?}"),
    }
}

fn arr(v: &JsonValue) -> &[JsonValue] {
    match v {
        JsonValue::Array(a) => a,
        other => panic!("expected array, got {other:?}"),
    }
}

fn config_value(ty: &str, lit: &str) -> SpecConfigValue {
    match ty {
        "bool" => SpecConfigValue::Bool(lit == "true"),
        "i64" => SpecConfigValue::I64(lit.parse().unwrap()),
        "f64" => SpecConfigValue::f64(lit.parse().unwrap()).unwrap(),
        "str" => SpecConfigValue::String(lit.to_owned()),
        other => panic!("unknown config type {other}"),
    }
}

fn port_type(t: &str) -> PortType {
    match t.strip_prefix("custom:") {
        Some(name) => PortType::custom(name),
        None => match t {
            "bool" => PortType::Bool,
            "f64" => PortType::F64,
            other => panic!("unknown port type {other}"),
        },
    }
}

fn build_node(n: &JsonValue) -> SystemNodeSpec {
    let n = obj(n);
    let mut node = SystemNodeSpec::new(st(&n["id"]), st(&n["kind"]));
    for e in arr(&n["config"]) {
        let e = arr(e);
        node = node.with_config(st(&e[0]), config_value(st(&e[1]), st(&e[2])));
    }
    for p in arr(&n["ports"]) {
        let p = arr(p);
        let ty = port_type(st(&p[2]));
        node = node.with_port(match st(&p[1]) {
            "input" => PortSpec::input(st(&p[0]), ty),
            _ => PortSpec::output(st(&p[0]), ty),
        });
    }
    node
}

fn emit(case: &JsonValue) -> String {
    let c = obj(case);
    match st(&c["kind"]) {
        "system_spec" => {
            let mut spec = SystemSpec::new(st(&c["spec_id"]));
            for e in arr(&c["config"]) {
                let e = arr(e);
                spec = spec.with_config(st(&e[0]), config_value(st(&e[1]), st(&e[2])));
            }
            if let Some(nodes) = c.get("nodes") {
                for n in arr(nodes) {
                    spec = spec.with_node(build_node(n));
                }
            }
            spec.to_canonical_json().unwrap()
        }
        "expr_const_f64" => {
            assert_eq!(st(&c["unit"]), "dimensionless");
            ExprAst::const_f64_with_unit(st(&c["value"]).parse().unwrap(), ExprUnit::Dimensionless)
                .unwrap()
                .to_canonical_json()
                .unwrap()
        }
        "expr_nested_add" => ExprAst::add(
            ExprAst::const_f64(st(&c["value"]).parse().unwrap()).unwrap(),
            ExprAst::var("x\u{e9}").unwrap(),
        )
        .to_canonical_json()
        .unwrap(),
        other => panic!("unknown case kind {other}"),
    }
}

#[test]
fn gog_152_rust_emits_exact_golden_bytes() {
    let cases = load();
    assert!(cases.len() >= 40, "corpus unexpectedly small");
    for case in &cases {
        let name = st(&obj(case)["name"]);
        let expected = st(&obj(case)["expected"]);
        assert_eq!(emit(case), expected, "case {name}");
    }
}

#[test]
fn gog_152_golden_is_parse_and_reemit_stable_for_system_specs() {
    for case in load() {
        let c = obj(&case);
        if st(&c["kind"]) != "system_spec" || c.contains_key("nodes") {
            continue;
        }
        let expected = st(&c["expected"]);
        let parsed = SystemSpec::from_json(expected).unwrap();
        assert_eq!(
            parsed.to_canonical_json().unwrap(),
            expected,
            "{}",
            st(&c["name"])
        );
    }
}

#[test]
fn gog_152_corpus_covers_required_categories() {
    let names: Vec<String> = load()
        .iter()
        .map(|c| st(&obj(c)["name"]).to_owned())
        .collect();
    for needle in [
        "f64_neg_zero",
        "f64_1e21",
        "str_non_bmp_emoji",
        "str_control_u00xx",
        "field_order_config_keys",
        "key_order_unicode_codepoint",
        "empty_everything",
        "node_empty_ports_and_config",
    ] {
        assert!(names.iter().any(|n| n == needle), "missing case {needle}");
    }
}
