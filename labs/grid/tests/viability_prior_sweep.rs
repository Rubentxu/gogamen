//! GOG-186: paired-seed sweep of the failure pseudocount (viability prior).

use gogamen_core::EndowmentProvenance;
use gogamen_lab_grid::viability_prior::{
    PriorSweepConfig, PriorSweepReport, Verdict, run_prior_sweep, spearman,
};

fn print(r: &PriorSweepReport) {
    println!("prior | deaths mean±sd [min,max] | explored mean±sd [min,max]");
    for row in &r.rows {
        println!(
            "{:>6} | {:.3}±{:.3} [{},{}] | {:.2}±{:.2} [{},{}]",
            row.prior,
            row.deaths.mean,
            row.deaths.std,
            row.deaths.min,
            row.deaths.max,
            row.explored.mean,
            row.explored.std,
            row.explored.min,
            row.explored.max
        );
    }
    println!(
        "rho_deaths={:.3} rho_explored={:.3} z_deaths={:.2} z_explored={:.2} verdict={:?}",
        r.rho_deaths, r.rho_explored, r.z_deaths, r.z_explored, r.verdict
    );
}

#[test]
fn viability_prior_sweep_paired_seeds_ledger_and_verdict() {
    let config = PriorSweepConfig::new();
    let report = run_prior_sweep(&config).unwrap();
    print(&report);

    // Paired design: identical seeds in every row.
    for row in &report.rows {
        let seeds: Vec<u64> = row.outcomes.iter().map(|o| o.seed).collect();
        assert_eq!(seeds, config.seeds);
        assert!(row.explored.min >= 1.0 && row.explored.max >= row.explored.min);
    }
    assert_eq!(report.rows.len(), config.priors.len());

    // Ledger: every prior recorded as given by design.
    assert_eq!(report.ledger.entries.len(), config.priors.len());
    for (e, p) in report.ledger.entries.iter().zip(&config.priors) {
        assert_eq!(e.name, "failure_pseudocount");
        assert_eq!(e.value, *p);
        assert_eq!(e.source, EndowmentProvenance::Given);
    }

    // Determinism of the whole sweep.
    assert_eq!(report, run_prior_sweep(&config).unwrap());

    // The verdict is whatever the preregistered rule says; Refuted is a valid outcome.
    assert!(matches!(
        report.verdict,
        Verdict::Confirmed | Verdict::Refuted
    ));
}

#[test]
fn viability_prior_sweep_spearman_sanity() {
    assert!((spearman(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0]) + 1.0).abs() < 1e-12);
    assert!((spearman(&[1.0, 2.0, 3.0], &[1.0, 5.0, 9.0]) - 1.0).abs() < 1e-12);
    assert_eq!(spearman(&[1.0, 2.0, 3.0], &[2.0, 2.0, 2.0]), 0.0);
}
