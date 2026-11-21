//! Métricas de ejemplo de un tercero para el kit de invariantes (GOG-191).
//!
//! Se definen FUERA de `gogamen-core`/`gogamen-sdk` y se someten con una llamada:
//! `gogamen_sdk::metric_kit::run_metric_kit("nombre", &mut shannon_bits, &KitConfig::default())`.

use gogamen_sdk::metric_kit::MetricInput;

/// Entropía de Shannon (bits) de la distribución. Debe PASAR todos los invariantes.
pub fn shannon_bits(input: &MetricInput, _seed: u64) -> Result<f64, String> {
    Ok(input
        .states
        .iter()
        .map(|(_, p)| if *p > 0.0 { -p * p.log2() } else { 0.0 })
        .sum())
}

/// Métrica deliberadamente rota: agrupa las etiquetas en bins de ancho 2 (`label / 2`), por lo
/// que depende del nombre de los estados. Debe FALLAR `relabel_invariance` (artefacto de bins).
pub fn label_binned_bits(input: &MetricInput, _seed: u64) -> Result<f64, String> {
    let mut bins: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
    for (l, p) in &input.states {
        *bins.entry(l / 2).or_insert(0.0) += p;
    }
    Ok(bins.values().map(|p| -p * p.log2()).sum())
}

/// Métrica rota: añade un sesgo constante (entropía + 0.25). Falla `zero_on_deterministic`
/// y `bounds`.
pub fn biased_bits(input: &MetricInput, seed: u64) -> Result<f64, String> {
    Ok(shannon_bits(input, seed)? + 0.25)
}
