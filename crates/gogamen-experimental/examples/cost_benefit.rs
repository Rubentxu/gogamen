use gogamen_experimental::compare::run_comparison;

fn main() {
    let r = run_comparison();
    println!("SYNTHETIC FIXTURES ONLY - not evidence about real models.");
    for a in [&r.baseline, &r.generative] {
        println!(
            "{:<28} items={} produced={} valid_rate={:.2} accuracy={:.2} measured={:?} declared_latency={:?} declared_units={} rej_ungrounded={} rej_goal={}",
            a.name,
            a.items,
            a.produced,
            a.valid_rate(),
            a.accuracy(),
            a.measured_total,
            a.declared_latency_total,
            a.declared_units_total,
            a.rejected_ungrounded,
            a.rejected_unknown_goal
        );
    }
    println!(
        "disabled-by-default abstentions: {}/{}",
        r.disabled_abstentions, r.baseline.items
    );
}
