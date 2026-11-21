//! Arena headless de comparación científica pareada (`gogamen-arena`, `std`-only).
//!
//! Ejecuta dos planificadores existentes bajo exactamente el mismo escenario,
//! contrato de observación (`ObservationMode`), presupuesto (`ArenaBudget`) y
//! semillas deterministas separadas en `Dev` y `Holdout`:
//! - Sin UI ni dependencia de Bevy (base headless para `GOG-101`).
//! - Resultados pareados por semilla ([`PairedSeedOutcome`]), tasas y conteos de éxito,
//!   consumo de presupuesto y dispersión estadística ([`DispersionStats`]).
//! - Fallos, timeouts y agotamientos de presupuesto ([`RunOutcomeStatus`]) contabilizados
//!   explícitamente con denominador total intacto.
//! - Trazas científicas verificadas compatibles con `gogamen_runtime::scientific_trace`
//!   (`ScientificTraceArchive`, `TraceManifest`, `ScientificRecord`).

pub mod profiling;
pub mod runner;
pub mod stats;

pub use gogamen_lab_grid::{ObservationMode, SeedSplit};
pub use profiling::{
    BeforeAfterProfileComparison, BottleneckStage, ComparisonOutcome, Decision, ExecutionPolicy,
    NO_GPU_OR_PARALLELISM_WITHOUT_EVIDENCE_NOTICE, OptimizationAcceptancePolicy, ProfileReport,
    ProfileVariant, ProfilingConfig, StageBreakdown, StageLatencyStats, run_before_after_profiling,
    run_before_after_profiling_with_clock, run_profile_variant,
};
pub use runner::{
    ArenaBudget, ArenaComparisonConfig, ArenaComparisonReport, ArenaError, ArenaPlanner,
    ArenaScenario, ArenaSeedSet, CartPoleArenaConfig, GridArenaConfig, PairedSeedOutcome,
    PlannerRunResult, run_arena_comparison,
};
pub use stats::{
    DispersionStats, PairedComparisonSummary, PairedWinner, PlannerSummary, RunOutcomeStatus,
    summarize_paired_outcomes, summarize_planner_runs,
};

#[cfg(test)]
mod tests {
    use gogamen_core::{FmcScoringConfig, MasterSeed};
    use gogamen_lab_cartpole::{CartPoleConfig, CartPoleKinematics, LinearFeedbackController};
    use gogamen_lab_grid::{GridMap, GridWorldConfig};
    use gogamen_runtime::scientific_trace::{
        ScientificTraceArchive, SensitiveFieldPolicy, TraceManifest,
    };

    use super::*;

    #[test]
    fn ac1_paired_grid_comparison_is_deterministic_with_shared_observation_and_budget() {
        let map = GridMap::from_ascii(&["#######", "#S...G#", "#######"]).unwrap();
        let grid_cfg = GridArenaConfig::new("corridor", map, GridWorldConfig::default())
            .unwrap()
            .with_observation_mode(ObservationMode::Partial { radius: 1 });

        let config = ArenaComparisonConfig {
            spec_id: "arena-grid-ac1".to_owned(),
            scenario: ArenaScenario::Grid(grid_cfg),
            budget: ArenaBudget::new(20, 4_000, 128),
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(11), MasterSeed::new(12)],
                vec![MasterSeed::new(101), MasterSeed::new(102)],
            )
            .unwrap(),
            left_planner: ArenaPlanner::GridFmc {
                population_size: 16,
                horizon: 6,
                scoring: FmcScoringConfig::default(),
            },
            right_planner: ArenaPlanner::GridMonteCarlo {
                horizon: 6,
                rollouts: 16,
            },
        };

        let report_a = run_arena_comparison(&config).unwrap();
        let report_b = run_arena_comparison(&config).unwrap();

        assert_eq!(
            report_a.observation_contract,
            "GridObservation::Partial(radius=1)"
        );
        assert_eq!(report_a.paired_outcomes.len(), 4);
        assert_eq!(report_a.left_overall.total_runs, 4);
        assert_eq!(report_a.right_overall.total_runs, 4);
        assert_eq!(report_a.left_overall.succeeded_count, 4);
        assert_eq!(report_a.right_overall.succeeded_count, 4);
        assert_eq!(report_a.paired_overall.both_succeeded, 4);

        // Determinismo bit-a-bit en métricas y trazas científicas (excluyendo latencia de reloj).
        for (pa, pb) in report_a
            .paired_outcomes
            .iter()
            .zip(report_b.paired_outcomes.iter())
        {
            assert_eq!(pa.seed, pb.seed);
            assert_eq!(pa.split, pb.split);
            assert_eq!(pa.steps_delta, pb.steps_delta);
            assert_eq!(pa.transitions_delta, pb.transitions_delta);
            assert_eq!(pa.left.status, pb.left.status);
            assert_eq!(pa.right.status, pb.right.status);
            assert_eq!(pa.left.trace_archive, pb.left.trace_archive);
            assert_eq!(pa.right.trace_archive, pb.right.trace_archive);
        }

        // Dispersión y consumo de presupuesto presentes.
        assert!(report_a.left_overall.steps_dispersion.min > 0.0);
        assert!(report_a.left_overall.mean_budget_utilization_steps > 0.0);
        assert!(report_a.left_overall.mean_budget_utilization_transitions > 0.0);
        assert!(!report_a.format_summary().is_empty());
    }

    #[test]
    fn ac1_paired_cartpole_comparison_reuses_existing_controllers_headless() {
        let cp_cfg = CartPoleConfig::default()
            .with_initial_state(CartPoleKinematics::new(0.1, 0.0, 0.08, 0.0).unwrap());
        let scenario = ArenaScenario::CartPole(
            CartPoleArenaConfig::new(
                "cartpole-perturbed",
                cp_cfg,
                gogamen_core::StepDt::from_secs_f64(0.02).unwrap(),
            )
            .unwrap(),
        );

        let config = ArenaComparisonConfig {
            spec_id: "arena-cartpole-ac1".to_owned(),
            scenario,
            budget: ArenaBudget::new(120, 120, 1),
            seeds: ArenaSeedSet::new(vec![MasterSeed::new(7)], vec![MasterSeed::new(77)]).unwrap(),
            left_planner: ArenaPlanner::CartPoleLinearFeedback(
                LinearFeedbackController::default_stabilizing(),
            ),
            right_planner: ArenaPlanner::CartPoleEntropicProxy(Box::default()),
        };

        let report = run_arena_comparison(&config).unwrap();
        assert_eq!(
            report.observation_contract,
            "CartPoleObservation::FullKinematics"
        );
        assert_eq!(report.left_overall.total_runs, 2);
        assert_eq!(report.right_overall.total_runs, 2);
        assert_eq!(report.left_overall.succeeded_count, 2);
        assert_eq!(report.paired_outcomes.len(), 2);
        assert_eq!(
            report.paired_outcomes[0].left.trace_archive.records.len(),
            120
        );
    }

    #[test]
    fn ac2_failures_timeouts_and_budget_exhaustion_preserve_total_denominator() {
        // Caso 1: mapa cerrado -> BFS falla explícitamente (`no_path_to_goal`), Random agota ticks (`TimedOut`).
        let closed_map = GridMap::from_ascii(&["#####", "#S#G#", "#####"]).unwrap();
        let closed_cfg =
            GridArenaConfig::new("closed-map", closed_map, GridWorldConfig::default()).unwrap();
        let config_closed = ArenaComparisonConfig {
            spec_id: "arena-ac2-closed".to_owned(),
            scenario: ArenaScenario::Grid(closed_cfg),
            budget: ArenaBudget::new(10, 500, 16),
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(1), MasterSeed::new(2)],
                vec![MasterSeed::new(10), MasterSeed::new(20)],
            )
            .unwrap(),
            left_planner: ArenaPlanner::GridPrivilegedBfs,
            right_planner: ArenaPlanner::GridRandomSeeded,
        };

        let report_closed = run_arena_comparison(&config_closed).unwrap();
        assert_eq!(report_closed.left_overall.total_runs, 4);
        assert_eq!(report_closed.left_overall.succeeded_count, 0);
        assert_eq!(report_closed.left_overall.failed_count, 4);
        assert_eq!(report_closed.left_overall.success_rate, 0.0);
        assert_eq!(report_closed.left_overall.failure_notes.len(), 4);

        assert_eq!(report_closed.right_overall.total_runs, 4);
        assert_eq!(report_closed.right_overall.succeeded_count, 0);
        assert_eq!(report_closed.right_overall.timed_out_count, 4);
        assert_eq!(report_closed.right_overall.success_rate, 0.0);

        // Caso 2: presupuesto de transiciones muy estricto -> `BudgetExhausted` contabilizado en denominador.
        let open_map = GridMap::from_ascii(&["########", "#S....G#", "########"]).unwrap();
        let open_cfg =
            GridArenaConfig::new("budget-tight", open_map, GridWorldConfig::default()).unwrap();
        let config_tight = ArenaComparisonConfig {
            spec_id: "arena-ac2-budget".to_owned(),
            scenario: ArenaScenario::Grid(open_cfg),
            budget: ArenaBudget::new(50, 10, 8),
            seeds: ArenaSeedSet::new(vec![MasterSeed::new(1)], vec![MasterSeed::new(2)]).unwrap(),
            left_planner: ArenaPlanner::GridMonteCarlo {
                horizon: 5,
                rollouts: 8,
            },
            right_planner: ArenaPlanner::GridRandomSeeded,
        };

        let report_tight = run_arena_comparison(&config_tight).unwrap();
        assert_eq!(report_tight.left_overall.total_runs, 2);
        assert_eq!(report_tight.left_overall.succeeded_count, 0);
        assert_eq!(report_tight.left_overall.budget_exhausted_count, 2);
        assert_eq!(
            report_tight.left_overall.succeeded_count
                + report_tight.left_overall.failed_count
                + report_tight.left_overall.timed_out_count
                + report_tight.left_overall.budget_exhausted_count,
            report_tight.left_overall.total_runs
        );
    }

    #[test]
    fn ac2_seed_split_enforces_strict_disjoint_dev_and_holdout_sets() {
        let overlap_err = ArenaSeedSet::new(
            vec![MasterSeed::new(1), MasterSeed::new(42)],
            vec![MasterSeed::new(42), MasterSeed::new(99)],
        )
        .unwrap_err();
        assert_eq!(
            overlap_err,
            ArenaError::OverlappingHoldoutSeeds {
                seed: MasterSeed::new(42),
            }
        );

        let dup_err = ArenaSeedSet::new(
            vec![MasterSeed::new(5), MasterSeed::new(5)],
            vec![MasterSeed::new(10)],
        )
        .unwrap_err();
        assert_eq!(
            dup_err,
            ArenaError::DuplicateSeed {
                split: SeedSplit::Dev,
                seed: MasterSeed::new(5),
            }
        );

        let empty_err = ArenaSeedSet::new(vec![], vec![]).unwrap_err();
        assert_eq!(empty_err, ArenaError::EmptySeedSet);

        // Verificar desglose separado Dev vs Holdout en el reporte.
        let map = GridMap::from_ascii(&["#####", "#S.G#", "#####"]).unwrap();
        let cfg = ArenaComparisonConfig {
            spec_id: "arena-ac2-split".to_owned(),
            scenario: ArenaScenario::Grid(
                GridArenaConfig::new("split-check", map, GridWorldConfig::default()).unwrap(),
            ),
            budget: ArenaBudget::new(15, 2_000, 64),
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(1), MasterSeed::new(2)],
                vec![MasterSeed::new(101)],
            )
            .unwrap(),
            left_planner: ArenaPlanner::GridPrivilegedBfs,
            right_planner: ArenaPlanner::GridRandomSeeded,
        };
        let report = run_arena_comparison(&cfg).unwrap();
        let left_dev = report.left_dev.as_ref().unwrap();
        let left_holdout = report.left_holdout.as_ref().unwrap();
        assert_eq!(left_dev.split, Some(SeedSplit::Dev));
        assert_eq!(left_dev.total_runs, 2);
        assert_eq!(left_holdout.split, Some(SeedSplit::Holdout));
        assert_eq!(left_holdout.total_runs, 1);
        assert_eq!(
            report.paired_dev.as_ref().unwrap().total_pairs
                + report.paired_holdout.as_ref().unwrap().total_pairs,
            report.paired_overall.total_pairs
        );
    }

    #[test]
    fn ac3_produces_valid_scientific_trace_archives_roundtrippable_without_ui() {
        let map = GridMap::from_ascii(&["#####", "#S.G#", "#####"]).unwrap();
        let cfg = ArenaComparisonConfig {
            spec_id: "arena-ac3-trace".to_owned(),
            scenario: ArenaScenario::Grid(
                GridArenaConfig::new("trace-check", map, GridWorldConfig::default()).unwrap(),
            ),
            budget: ArenaBudget::new(10, 1_000, 64),
            seeds: ArenaSeedSet::new(vec![MasterSeed::new(7)], vec![MasterSeed::new(70)]).unwrap(),
            left_planner: ArenaPlanner::GridFmc {
                population_size: 8,
                horizon: 4,
                scoring: FmcScoringConfig::default(),
            },
            right_planner: ArenaPlanner::GridMonteCarlo {
                horizon: 4,
                rollouts: 8,
            },
        };

        let report = run_arena_comparison(&cfg).unwrap();
        for pair in &report.paired_outcomes {
            for run in [&pair.left, &pair.right] {
                run.trace_manifest.verify().unwrap();
                assert!(run.trace_manifest.is_complete);
                assert_eq!(run.trace_manifest.spec_id, "arena-ac3-trace");
                assert_eq!(run.trace_manifest.seed, pair.seed);
                assert_eq!(run.trace_manifest.total_records, run.steps_executed);
                assert_eq!(run.trace_archive.records.len() as u64, run.steps_executed);

                // Roundtrip binario completo de `ScientificTraceArchive` y `TraceManifest`.
                let archive_bytes = run
                    .trace_archive
                    .to_bytes(&SensitiveFieldPolicy::default())
                    .unwrap();
                let decoded_archive = ScientificTraceArchive::from_bytes(&archive_bytes).unwrap();
                assert_eq!(decoded_archive, run.trace_archive);

                let manifest_bytes = run.trace_manifest.to_bytes();
                let decoded_manifest = TraceManifest::from_bytes(&manifest_bytes).unwrap();
                assert_eq!(decoded_manifest, run.trace_manifest);
            }
        }
    }

    #[test]
    fn incompatible_planner_and_scenario_are_rejected() {
        let cfg = ArenaComparisonConfig {
            spec_id: "invalid-pair".to_owned(),
            scenario: ArenaScenario::Grid(GridArenaConfig::default()),
            budget: ArenaBudget::default(),
            seeds: ArenaSeedSet::default(),
            left_planner: ArenaPlanner::CartPoleLinearFeedback(
                LinearFeedbackController::default_stabilizing(),
            ),
            right_planner: ArenaPlanner::GridRandomSeeded,
        };
        let err = run_arena_comparison(&cfg).unwrap_err();
        assert!(matches!(
            err,
            ArenaError::IncompatiblePlanner { side: "left", .. }
        ));
    }
}
