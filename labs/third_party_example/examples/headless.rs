//! Ejecuta el lab de ejemplo sin UI: `cargo run -p gogamen-example-third-party --example headless`.

use gogamen_example_third_party::{OrbitProjector, orbit_projection};
use gogamen_projection::run_headless;

fn main() {
    let scenes = run_headless(&mut OrbitProjector::default(), &orbit_projection(), 5)
        .expect("conformant lab");
    println!("{}", scenes.last().expect("scenes").to_canonical_json());
}
