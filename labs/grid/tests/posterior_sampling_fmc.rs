//! GOG-195: posterior sampling in FMC vs planning on the mean, paired, reserve seeds.

use std::collections::BTreeSet;

use gogamen_lab_grid::posterior_sampling_fmc::*;

fn small() -> StudyConfig {
    StudyConfig {
        budget: 60,
        checkpoint_every: 30,
        ..StudyConfig::new()
    }
}

#[test]
fn posterior_sampling_fmc_samples_are_deterministic_and_recorded() {
    let cfg = small();
    let a = run_arm(&cfg, Arm::PosteriorSample, 1).unwrap();
    let b = run_arm(&cfg, Arm::PosteriorSample, 1).unwrap();
    assert_eq!(a, b, "same seed, same samples and same outcome");
    // Every decision records which sample it used.
    assert_eq!(a.samples.len(), cfg.budget);
    let ids: BTreeSet<u64> = a.samples.iter().map(|s| s.sample_id).collect();
    assert_eq!(ids.len(), cfg.budget, "one fresh sample per planning call");
    let hashes: BTreeSet<u64> = a.samples.iter().map(|s| s.table_hash).collect();
    assert!(hashes.len() > cfg.budget / 2, "distinct tables per call");
    assert!(a.samples.iter().all(|s| s.seed == a.samples[0].seed));
    // Learning advances the model revision between decisions.
    assert!(a.samples.last().unwrap().model_revision > a.samples[0].model_revision);
    // Another seed samples differently.
    let c = run_arm(&cfg, Arm::PosteriorSample, 2).unwrap();
    assert_ne!(a.samples[0].table_hash, c.samples[0].table_hash);
    // Non-sampling arms record no samples.
    assert!(run_arm(&cfg, Arm::MeanPlan, 1).unwrap().samples.is_empty());
}

#[test]
fn posterior_sampling_fmc_equal_budget_and_disjoint_seeds() {
    let cfg = StudyConfig::new();
    assert!(cfg.dev_seeds.iter().all(|d| !cfg.reserve_seeds.contains(d)));
    // Preregistered constants are fixed in code.
    assert_eq!((GAP_TOLERANCE, MIN_GAIN, MIN_Z), (1.0, 150.0, 2.0));
}

#[test]
fn posterior_sampling_fmc_study_on_reserve_seeds_reports_preregistered_verdicts() {
    let cfg = StudyConfig::new();
    let report = run_study(&cfg, &cfg.reserve_seeds).unwrap();
    println!(
        "reserve seeds = {:?}; budget = {} real transitions per arm; checkpoint every {}; censored value = {}",
        cfg.reserve_seeds,
        cfg.budget,
        cfg.checkpoint_every,
        cfg.budget + cfg.checkpoint_every
    );
    println!(
        "arm | experience-to-ceiling mean±sd [min,med,max] | censored | final gap mean±sd | pairs covered"
    );
    for a in &report.arms {
        println!(
            "{:>16} | {:.1}±{:.1} [{},{},{}] | {}/{} | {:.2}±{:.2} | {:.1}",
            a.arm.label(),
            a.etc.mean,
            a.etc.std,
            a.etc.min,
            a.etc.median,
            a.etc.max,
            a.censored,
            a.runs.len(),
            a.final_gap.mean,
            a.final_gap.std,
            a.pairs.mean
        );
    }
    for c in &report.comparisons {
        println!(
            "posterior vs {:>14}: gain {:+.1} z {:+.2} W/L/T {}/{}/{} => {:?}",
            c.versus.label(),
            c.mean_gain,
            c.z,
            c.wins,
            c.losses,
            c.ties,
            c.verdict
        );
    }
    for a in &report.arms {
        let seeds: Vec<u64> = a.runs.iter().map(|r| r.seed).collect();
        assert_eq!(seeds, cfg.reserve_seeds, "paired seeds in every arm");
    }
    // Paired ceiling: identical in every arm.
    let ceil = |a: &ArmReport| {
        a.runs
            .iter()
            .map(|r| r.ceiling.to_bits())
            .collect::<Vec<_>>()
    };
    assert!(report.arms.iter().all(|a| ceil(a) == ceil(&report.arms[0])));
    assert_eq!(report.comparisons.len(), Arm::ALL.len() - 1);
    // Verdicts are whatever the preregistered rule says; null and worse are valid.
    for c in &report.comparisons {
        assert!(matches!(
            c.verdict,
            Verdict::Better | Verdict::Null | Verdict::Worse
        ));
    }
}
