//! Vista de grafo y eventos de automatización a SVG
//! (`cargo run -p gogamen-studio --example automation_svg -- [incident|retry|timeout]`). SVG por stdout.

use gogamen_studio::automation::{
    incident_scenario, render_automation_svg, retry_error_scenario, run_automation,
    timeout_scenario,
};

fn main() {
    let which = std::env::args().nth(1).unwrap_or_else(|| "incident".into());
    let scenario = match which.as_str() {
        "retry" => retry_error_scenario(),
        "timeout" => timeout_scenario(),
        _ => incident_scenario(),
    };
    let view = run_automation(&which, &scenario).expect("headless automation rollout");
    print!("{}", render_automation_svg(&view, 1100.0, 640.0));
}
