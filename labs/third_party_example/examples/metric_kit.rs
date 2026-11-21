//! Somete métricas de un tercero al kit de invariantes de entropía futura (GOG-191):
//! `cargo run -p gogamen-example-third-party --offline --example metric_kit`.

use gogamen_example_third_party::metric::{biased_bits, label_binned_bits, shannon_bits};
use gogamen_sdk::metric_kit::{KitConfig, run_metric_kit};

fn main() {
    let cfg = KitConfig::default();
    print!(
        "{}",
        run_metric_kit("shannon_bits", &mut shannon_bits, &cfg).render()
    );
    print!(
        "{}",
        run_metric_kit("label_binned_bits (broken)", &mut label_binned_bits, &cfg).render()
    );
    print!(
        "{}",
        run_metric_kit("biased_bits (broken)", &mut biased_bits, &cfg).render()
    );
}
