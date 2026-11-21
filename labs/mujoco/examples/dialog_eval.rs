//! Policy comparison over 16 annotated, synthetic dialog fixtures.
//! Run: `cargo run --release -p gogamen-lab-mujoco --example dialog_eval`
use gogamen_lab_mujoco::dialog::*;

fn main() {
    let fixtures = annotated_fixtures();
    let kinds = [
        PolicyKind::NoAskGuess,
        PolicyKind::NoAskAbstain,
        PolicyKind::EpistemicAsk,
        PolicyKind::AlwaysAsk,
    ];
    let p = DialogPolicy::new(PolicyKind::EpistemicAsk);
    println!(
        "fixtures={} oracle=Replay(latency 5 ticks) prior={} speaker_rel={} human_rel={} threshold={} reward={} penalty={} ask_cost={} p_resp={}",
        fixtures.len(),
        p.prior_p_left,
        p.speaker_reliability,
        p.human_reliability,
        p.act_threshold,
        p.reward,
        p.penalty,
        p.ask_cost_units,
        p.response_probability
    );
    println!(
        "{:<26} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8} {:>7} {:>6} {:>8} {:>7} {:>8}",
        "policy",
        "succ",
        "wrong",
        "abst",
        "coll",
        "other",
        "asks",
        "utility",
        "brier",
        "ece",
        "mean_cf",
        "map_acc",
        "ticks"
    );
    for (mode_name, mode) in [
        ("replay", OracleMode::Replay),
        ("jev-never", OracleMode::Never),
    ] {
        println!("-- oracle: {mode_name}");
        for k in kinds {
            let (r, _) = run_policy(&fixtures, DialogPolicy::new(k), mode);
            println!(
                "{:<26} {:>4} {:>5} {:>5} {:>5} {:>5} {:>5} {:>8.1} {:>7.4} {:>6.4} {:>8.4} {:>7.4} {:>8.1}",
                k.label(),
                r.successes,
                r.wrong_door,
                r.abstained,
                r.collisions,
                r.other_failures,
                r.queries,
                r.utility,
                r.brier,
                r.ece,
                r.mean_confidence,
                r.map_accuracy,
                r.mean_ticks
            );
        }
    }
    println!("reference: constant P(left)=0.5 has Brier 0.25000");
    println!("-- per-fixture (ask-when-worth-it, replay): id truth p_left asks verdict decided_at");
    let (_, rs) = run_policy(
        &fixtures,
        DialogPolicy::new(PolicyKind::EpistemicAsk),
        OracleMode::Replay,
    );
    for (f, r) in fixtures.iter().zip(&rs) {
        println!(
            "{:<24} truth={:<5} p_left={:.4} asks={} {:?} decided_at={:?} ticks={}",
            f.id,
            if f.truth_left { "left" } else { "right" },
            r.p_left,
            r.queries,
            r.verdict,
            r.decided_at,
            r.ticks
        );
    }
}
