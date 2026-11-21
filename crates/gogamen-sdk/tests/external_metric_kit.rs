//! GOG-191: kit de invariantes para métricas de terceros (closures, proceso externo).

use std::time::Duration;

use gogamen_sdk::metric_kit::{
    INVARIANT_IDS, KitConfig, METRIC_KIT_VERSION, MetricInput, ProcessMetric, metric_kit_hash,
    run_metric_kit,
};

fn shannon(i: &MetricInput, _s: u64) -> Result<f64, String> {
    Ok(i.states
        .iter()
        .map(|(_, p)| if *p > 0.0 { -p * p.log2() } else { 0.0 })
        .sum())
}

fn label_binned(i: &MetricInput, _s: u64) -> Result<f64, String> {
    let mut b = std::collections::BTreeMap::new();
    for (l, p) in &i.states {
        *b.entry(l / 2).or_insert(0.0) += p;
    }
    Ok(b.values().map(|p: &f64| -p * p.log2()).sum())
}

#[test]
fn external_metric_kit_passes_valid_metric() {
    let r = run_metric_kit("shannon", &mut shannon, &KitConfig::default());
    assert!(r.passed(), "{}", r.render());
    assert_eq!(r.results.len(), INVARIANT_IDS.len());
    assert_eq!(r.kit_version, METRIC_KIT_VERSION);
}

#[test]
fn external_metric_kit_relabel_counterexample_is_minimal() {
    let r = run_metric_kit("binned", &mut label_binned, &KitConfig::default());
    assert!(!r.passed());
    let c = r
        .result("relabel_invariance")
        .unwrap()
        .counterexample
        .as_ref()
        .expect("fails");
    // mínimo: 2 estados, etiquetas canónicas 0 y 1, uniforme
    assert_eq!(c.input.states, vec![(0, 0.5), (1, 0.5)], "{}", r.render());
    // el resto de invariantes no se ven afectados por el artefacto de bins salvo la duplicación
    assert!(r.result("zero_on_deterministic").unwrap().passed());
}

#[test]
fn external_metric_kit_bias_breaks_zero_bounds_minimally() {
    let mut biased = |i: &MetricInput, s: u64| shannon(i, s).map(|v| v + 0.25);
    let r = run_metric_kit("biased", &mut biased, &KitConfig::default());
    let z = r.result("zero_on_deterministic").unwrap();
    assert_eq!(
        z.counterexample.as_ref().unwrap().input.states,
        vec![(0, 1.0)]
    );
    let b = r.result("bounds").unwrap();
    assert_eq!(b.counterexample.as_ref().unwrap().input.states.len(), 1);
}

#[test]
fn external_metric_kit_detects_nondeterminism_and_nonmonotone() {
    let mut n = 0u64;
    let mut flaky = |i: &MetricInput, s: u64| {
        n += 1;
        shannon(i, s).map(|v| v + (n as f64) * 1e-6)
    };
    let r = run_metric_kit("flaky", &mut flaky, &KitConfig::default());
    let c = r.result("seed_determinism").unwrap();
    assert_eq!(c.counterexample.as_ref().unwrap().input.states.len(), 1);
    // métrica decreciente con la mezcla: 2 - H
    let mut inv = |i: &MetricInput, s: u64| shannon(i, s).map(|v| 2.0 - v);
    let r = run_metric_kit("inverted", &mut inv, &KitConfig::default());
    assert!(!r.result("mixing_monotonicity").unwrap().passed());
}

#[test]
fn external_metric_kit_report_is_deterministic_with_version_and_hash() {
    let a = run_metric_kit("x", &mut label_binned, &KitConfig::default()).render();
    let b = run_metric_kit("x", &mut label_binned, &KitConfig::default()).render();
    assert_eq!(a, b);
    assert!(a.contains(METRIC_KIT_VERSION));
    assert!(a.contains(&format!("{:016x}", metric_kit_hash())));
    // hash estable (cambia solo si cambian versión/ids/constantes del kit)
    assert_eq!(metric_kit_hash(), metric_kit_hash());
}

fn python_script(name: &str) -> Option<String> {
    let path = format!(
        "{}/../../labs/third_party_example/examples/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::process::Command::new("python3")
        .arg("-I")
        .arg("--version")
        .output()
        .ok()?;
    Some(path)
}

#[test]
fn external_metric_kit_python_process_over_line_protocol() {
    let Some(script) = python_script("metric_stdio.py") else {
        eprintln!("NOT_RUN: python3 no disponible");
        return;
    };
    let cfg = KitConfig {
        cases: 40,
        ..KitConfig::default()
    };
    let spawn = |extra: &[&str]| {
        let mut args = vec!["-I".to_string(), script.clone()];
        args.extend(extra.iter().map(|s| s.to_string()));
        ProcessMetric::spawn("python3", &args, Duration::from_secs(10)).unwrap()
    };
    let r = run_metric_kit("py", &mut spawn(&[]), &cfg);
    assert!(r.passed(), "{}", r.render());
    let r = run_metric_kit("py-broken", &mut spawn(&["--broken"]), &cfg);
    let c = r.result("relabel_invariance").unwrap();
    assert_eq!(
        c.counterexample.as_ref().unwrap().input.states,
        vec![(0, 0.5), (1, 0.5)],
        "{}",
        r.render()
    );
}

/// Cobertura del selector de GOG-197 (`metric_invariants`): el kit acepta una métrica válida y
/// rechaza una rota con contraejemplo.
#[test]
fn metric_invariants_kit_accepts_valid_and_rejects_broken() {
    assert!(run_metric_kit("ok", &mut shannon, &KitConfig::default()).passed());
    assert!(!run_metric_kit("bad", &mut label_binned, &KitConfig::default()).passed());
}
