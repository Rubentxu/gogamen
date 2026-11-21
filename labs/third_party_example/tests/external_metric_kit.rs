//! GOG-191: métricas del tercero sometidas al kit del SDK (sin tocar core).

use gogamen_example_third_party::metric::{biased_bits, label_binned_bits, shannon_bits};
use gogamen_sdk::metric_kit::{KitConfig, run_metric_kit};

#[test]
fn external_metric_kit_third_party_good_and_broken() {
    let cfg = KitConfig::default();
    assert!(run_metric_kit("shannon", &mut shannon_bits, &cfg).passed());
    let r = run_metric_kit("binned", &mut label_binned_bits, &cfg);
    let c = r.result("relabel_invariance").unwrap();
    assert_eq!(c.counterexample.as_ref().unwrap().input.states.len(), 2);
    assert!(!run_metric_kit("biased", &mut biased_bits, &cfg).passed());
}
