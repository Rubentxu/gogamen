//! Pre-registered hypothesis verdicts, kept separate from the build/test status.
//!
//! The verdict (`SUPPORTED | REFUTED | INCONCLUSIVE`) answers "what did the experiment say
//! about the claim?", whereas [`BuildStatus`] (`PASS | FAIL | NOT_RUN`) answers "did the code
//! build and the checks run?". The two are reported side by side and never derived from each
//! other: a well-executed experiment that refutes its hypothesis is a green build.
//!
//! Rules enforced by [`evaluate_hypotheses`]:
//! - no pre-registered hash in the trace manifest -> `INCONCLUSIVE` (never `SUPPORTED`);
//! - hash of the current spec differs from the manifest hash (e.g. a threshold was edited after
//!   running) -> `INCONCLUSIVE` with basis `HASH_MISMATCH`, which flags the tampering;
//! - incomplete trace or missing/non-finite metric -> `INCONCLUSIVE`.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gogamen_runtime::scientific_trace::TraceManifest;
use gogamen_spec::{HypothesisSpec, SpecError, hypotheses_hash};

use crate::stats::PlannerSummary;

/// Hypothesis verdict, independent of the build status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisVerdict {
    /// The pre-registered support condition held on a verified, complete trace.
    Supported,
    /// The pre-registered refutation condition held on a verified, complete trace.
    Refuted,
    /// Not decidable (or not decidable honestly); never counted as supported.
    Inconclusive,
}

impl HypothesisVerdict {
    /// Stable uppercase label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Supported => "SUPPORTED",
            Self::Refuted => "REFUTED",
            Self::Inconclusive => "INCONCLUSIVE",
        }
    }
}

/// Build/test status, deliberately a different type from [`HypothesisVerdict`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildStatus {
    /// Checks ran and passed.
    Pass,
    /// Checks ran and failed.
    Fail,
    /// Checks were not run.
    NotRun,
}

impl BuildStatus {
    /// Stable uppercase label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Fail => "FAIL",
            Self::NotRun => "NOT_RUN",
        }
    }
}

/// Why a verdict was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerdictBasis {
    /// Support condition satisfied.
    SupportConditionMet,
    /// Refutation condition satisfied.
    RefutationConditionMet,
    /// Both conditions satisfied at once (contradictory registration).
    ConditionsOverlap,
    /// Neither condition satisfied.
    BetweenThresholds,
    /// No hypotheses hash was recorded in the manifest before execution.
    NotPreregistered,
    /// The spec's hypotheses hash differs from the one recorded before execution.
    HashMismatch,
    /// The trace is incomplete (gaps or drops).
    IncompleteTrace,
    /// The metric was not observed or is not finite.
    MetricUnavailable,
}

impl VerdictBasis {
    /// Stable uppercase label.
    pub const fn label(self) -> &'static str {
        match self {
            Self::SupportConditionMet => "SUPPORT_CONDITION_MET",
            Self::RefutationConditionMet => "REFUTATION_CONDITION_MET",
            Self::ConditionsOverlap => "CONDITIONS_OVERLAP",
            Self::BetweenThresholds => "BETWEEN_THRESHOLDS",
            Self::NotPreregistered => "NOT_PREREGISTERED",
            Self::HashMismatch => "HASH_MISMATCH",
            Self::IncompleteTrace => "INCOMPLETE_TRACE",
            Self::MetricUnavailable => "METRIC_UNAVAILABLE",
        }
    }
}

/// Evidence of one executed run against which hypotheses are adjudicated.
#[derive(Debug, Clone)]
pub struct HypothesisEvidence<'a> {
    /// Manifest of the run's trace (`None` when no trace exists).
    pub manifest: Option<&'a TraceManifest>,
    /// Reference (path/URI) to the archived trace or receipt; derived from the manifest if `None`.
    pub trace_ref: Option<String>,
    /// Observed metric values by metric name.
    pub observed: BTreeMap<String, f64>,
    /// Build/test status of the run; only displayed, never affects the verdict.
    pub build: BuildStatus,
}

/// One row of the verdict table.
#[derive(Debug, Clone, PartialEq)]
pub struct VerdictEntry {
    /// Hypothesis identifier.
    pub hypothesis_id: String,
    /// Claim text.
    pub claim: String,
    /// Metric evaluated.
    pub metric: String,
    /// Observed metric value, if any.
    pub observed: Option<f64>,
    /// Verdict.
    pub verdict: HypothesisVerdict,
    /// Reason for the verdict.
    pub basis: VerdictBasis,
    /// Hash recorded in the trace manifest before execution.
    pub registered_hash: Option<u64>,
    /// Hash of the hypotheses in the spec being adjudicated.
    pub spec_hash: Option<u64>,
    /// Link to the trace/receipt backing this row.
    pub evidence_ref: String,
    /// Build status (independent of the verdict).
    pub build: BuildStatus,
}

/// Separate counts per verdict; `INCONCLUSIVE` is never folded into `supported`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VerdictCounts {
    /// Number of `SUPPORTED`.
    pub supported: usize,
    /// Number of `REFUTED`.
    pub refuted: usize,
    /// Number of `INCONCLUSIVE`.
    pub inconclusive: usize,
}

/// Derives a stable reference to a trace from its manifest.
pub fn trace_ref_for_manifest(manifest: &TraceManifest) -> String {
    format!(
        "trace://run/{}/spec/{}/model/{}/checksum/{:016x}",
        manifest.run_id.as_u64(),
        manifest.spec_id,
        manifest.model_id,
        manifest.checksum
    )
}

/// Adjudicates every hypothesis of the spec against one run's evidence.
pub fn evaluate_hypotheses(
    hypotheses: &[HypothesisSpec],
    evidence: &HypothesisEvidence<'_>,
) -> Result<Vec<VerdictEntry>, SpecError> {
    let spec_hash = hypotheses_hash(hypotheses)?;
    let registered = evidence.manifest.and_then(|m| m.hypotheses_hash);
    let evidence_ref = evidence
        .trace_ref
        .clone()
        .or_else(|| evidence.manifest.map(trace_ref_for_manifest))
        .unwrap_or_else(|| "none".to_owned());
    let complete = evidence.manifest.is_some_and(|m| m.is_complete);

    let mut rows = Vec::with_capacity(hypotheses.len());
    for h in hypotheses {
        let observed = evidence
            .observed
            .get(&h.metric)
            .copied()
            .filter(|v| v.is_finite());
        let (verdict, basis) = if registered.is_none() {
            (
                HypothesisVerdict::Inconclusive,
                VerdictBasis::NotPreregistered,
            )
        } else if registered != spec_hash {
            (HypothesisVerdict::Inconclusive, VerdictBasis::HashMismatch)
        } else if !complete {
            (
                HypothesisVerdict::Inconclusive,
                VerdictBasis::IncompleteTrace,
            )
        } else if let Some(v) = observed {
            let supports = h.comparator.holds(v, h.threshold);
            let refutes = h.refutation_comparator.holds(v, h.refutation_threshold);
            match (supports, refutes) {
                (true, false) => (
                    HypothesisVerdict::Supported,
                    VerdictBasis::SupportConditionMet,
                ),
                (false, true) => (
                    HypothesisVerdict::Refuted,
                    VerdictBasis::RefutationConditionMet,
                ),
                (true, true) => (
                    HypothesisVerdict::Inconclusive,
                    VerdictBasis::ConditionsOverlap,
                ),
                (false, false) => (
                    HypothesisVerdict::Inconclusive,
                    VerdictBasis::BetweenThresholds,
                ),
            }
        } else {
            (
                HypothesisVerdict::Inconclusive,
                VerdictBasis::MetricUnavailable,
            )
        };
        rows.push(VerdictEntry {
            hypothesis_id: h.hypothesis_id.clone(),
            claim: h.claim.clone(),
            metric: h.metric.clone(),
            observed,
            verdict,
            basis,
            registered_hash: registered,
            spec_hash,
            evidence_ref: evidence_ref.clone(),
            build: evidence.build,
        });
    }
    Ok(rows)
}

/// Counts verdicts separately.
pub fn count_verdicts(entries: &[VerdictEntry]) -> VerdictCounts {
    let mut c = VerdictCounts::default();
    for e in entries {
        match e.verdict {
            HypothesisVerdict::Supported => c.supported += 1,
            HypothesisVerdict::Refuted => c.refuted += 1,
            HypothesisVerdict::Inconclusive => c.inconclusive += 1,
        }
    }
    c
}

/// Renders the verdict table (Markdown) with a link to each trace/receipt and separate counts.
pub fn render_verdict_table(entries: &[VerdictEntry]) -> String {
    let mut out = String::new();
    out.push_str(
        "| hypothesis | metric | observed | verdict | basis | build | evidence |\n\
         |---|---|---|---|---|---|---|\n",
    );
    for e in entries {
        let observed = e
            .observed
            .map_or_else(|| "n/a".to_owned(), |v| format!("{v}"));
        let _ = writeln!(
            out,
            "| {} | {} | {} | {} | {} | {} | {} |",
            e.hypothesis_id,
            e.metric,
            observed,
            e.verdict.label(),
            e.basis.label(),
            e.build.label(),
            e.evidence_ref
        );
    }
    let c = count_verdicts(entries);
    let _ = writeln!(
        out,
        "\nSUPPORTED={} REFUTED={} INCONCLUSIVE={} (INCONCLUSIVE is never counted as SUPPORTED)",
        c.supported, c.refuted, c.inconclusive
    );
    out
}

/// Extracts the metrics hypotheses can refer to from a planner summary.
pub fn planner_metrics(summary: &PlannerSummary) -> BTreeMap<String, f64> {
    BTreeMap::from([
        ("success_rate".to_owned(), summary.success_rate),
        (
            "mean_budget_utilization_steps".to_owned(),
            summary.mean_budget_utilization_steps,
        ),
        (
            "mean_budget_utilization_transitions".to_owned(),
            summary.mean_budget_utilization_transitions,
        ),
    ])
}

#[cfg(test)]
mod tests {
    use gogamen_core::MasterSeed;
    use gogamen_lab_grid::{GridMap, GridWorldConfig};
    use gogamen_spec::HypothesisComparator;

    use super::*;
    use crate::runner::{
        ArenaBudget, ArenaComparisonConfig, ArenaPlanner, ArenaScenario, ArenaSeedSet,
        GridArenaConfig, run_arena_comparison_preregistered,
    };

    fn hypothesis(support_threshold: f64) -> HypothesisSpec {
        HypothesisSpec::new(
            "h-random-solves-closed-map",
            "random planner solves the closed map at least half of the time",
            "success_rate",
            HypothesisComparator::Ge,
            support_threshold,
            HypothesisComparator::Lt,
            0.1,
        )
    }

    fn closed_map_config() -> ArenaComparisonConfig {
        let map = GridMap::from_ascii(&["#####", "#S#G#", "#####"]).unwrap();
        let cfg = GridArenaConfig::new("closed-map", map, GridWorldConfig::default()).unwrap();
        ArenaComparisonConfig {
            spec_id: "arena-hypothesis".to_owned(),
            scenario: ArenaScenario::Grid(cfg),
            budget: ArenaBudget::new(10, 500, 16),
            seeds: ArenaSeedSet::new(
                vec![MasterSeed::new(1), MasterSeed::new(2)],
                vec![MasterSeed::new(10), MasterSeed::new(20)],
            )
            .unwrap(),
            left_planner: ArenaPlanner::GridPrivilegedBfs,
            right_planner: ArenaPlanner::GridRandomSeeded,
        }
    }

    #[test]
    fn hypothesis_verdict_refuted_is_archived_and_does_not_break_the_build() {
        let hypotheses = vec![hypothesis(0.5)];
        let hash = hypotheses_hash(&hypotheses).unwrap();
        let report = run_arena_comparison_preregistered(&closed_map_config(), hash).unwrap();
        let run = &report.paired_outcomes[0].right;
        // Hash was stamped before execution and survives the archive roundtrip.
        assert_eq!(run.trace_manifest.hypotheses_hash, hash);
        assert_eq!(run.trace_archive.manifest.hypotheses_hash, hash);
        run.trace_archive.manifest.verify().unwrap();

        let evidence = HypothesisEvidence {
            manifest: Some(&run.trace_archive.manifest),
            trace_ref: None,
            observed: planner_metrics(&report.right_overall),
            build: BuildStatus::Pass,
        };
        let rows = evaluate_hypotheses(&hypotheses, &evidence).unwrap();
        // The experiment was well executed and refutes the claim, yet the test passes (green).
        assert_eq!(rows[0].verdict, HypothesisVerdict::Refuted);
        assert_eq!(rows[0].basis, VerdictBasis::RefutationConditionMet);
        assert_eq!(rows[0].build, BuildStatus::Pass);
        assert!(rows[0].evidence_ref.starts_with("trace://run/"));
        let table = render_verdict_table(&rows);
        assert!(table.contains("REFUTED"), "{table}");
        assert!(table.contains(&rows[0].evidence_ref));
    }

    #[test]
    fn hypothesis_verdict_without_registration_is_inconclusive_never_supported() {
        let hypotheses = vec![hypothesis(0.5)];
        // Run WITHOUT stamping the hash, with a metric that would satisfy support.
        let report = run_arena_comparison_preregistered(&closed_map_config(), None).unwrap();
        let manifest = &report.paired_outcomes[0].left.trace_archive.manifest;
        let mut observed = BTreeMap::new();
        observed.insert("success_rate".to_owned(), 1.0);
        let evidence = HypothesisEvidence {
            manifest: Some(manifest),
            trace_ref: None,
            observed: observed.clone(),
            build: BuildStatus::Pass,
        };
        let rows = evaluate_hypotheses(&hypotheses, &evidence).unwrap();
        assert_eq!(rows[0].verdict, HypothesisVerdict::Inconclusive);
        assert_eq!(rows[0].basis, VerdictBasis::NotPreregistered);

        // No manifest at all is also INCONCLUSIVE.
        let none = HypothesisEvidence {
            manifest: None,
            trace_ref: None,
            observed,
            build: BuildStatus::NotRun,
        };
        let rows = evaluate_hypotheses(&hypotheses, &none).unwrap();
        assert_eq!(rows[0].verdict, HypothesisVerdict::Inconclusive);
        assert_eq!(rows[0].build, BuildStatus::NotRun);
    }

    #[test]
    fn hypothesis_verdict_threshold_edited_after_run_is_flagged() {
        let original = vec![hypothesis(0.5)];
        let hash = hypotheses_hash(&original).unwrap();
        let report = run_arena_comparison_preregistered(&closed_map_config(), hash).unwrap();
        let manifest = &report.paired_outcomes[0].left.trace_archive.manifest;
        let mut observed = BTreeMap::new();
        // Metric that supports the (post-hoc) edited threshold but not the registered one.
        observed.insert("success_rate".to_owned(), 0.3);
        let evidence = HypothesisEvidence {
            manifest: Some(manifest),
            trace_ref: Some("receipts/run-1.gtrace".to_owned()),
            observed,
            build: BuildStatus::Pass,
        };
        let edited = vec![hypothesis(0.2)];
        assert_ne!(hypotheses_hash(&edited).unwrap(), hash);
        let rows = evaluate_hypotheses(&edited, &evidence).unwrap();
        assert_eq!(rows[0].verdict, HypothesisVerdict::Inconclusive);
        assert_eq!(rows[0].basis, VerdictBasis::HashMismatch);
        assert_eq!(rows[0].evidence_ref, "receipts/run-1.gtrace");
        // The original registration is still adjudicated normally.
        let rows = evaluate_hypotheses(&original, &evidence).unwrap();
        assert_eq!(rows[0].basis, VerdictBasis::BetweenThresholds);
    }

    #[test]
    fn hypothesis_verdict_table_counts_are_separate_and_link_evidence() {
        let hypotheses = vec![
            hypothesis(0.5),
            HypothesisSpec::new(
                "h-supported",
                "bfs budget use stays below 2",
                "mean_budget_utilization_steps",
                HypothesisComparator::Le,
                2.0,
                HypothesisComparator::Gt,
                5.0,
            ),
            HypothesisSpec::new(
                "h-missing",
                "unobserved metric",
                "never_measured",
                HypothesisComparator::Ge,
                1.0,
                HypothesisComparator::Lt,
                0.0,
            ),
        ];
        let hash = hypotheses_hash(&hypotheses).unwrap();
        let report = run_arena_comparison_preregistered(&closed_map_config(), hash).unwrap();
        let run = &report.paired_outcomes[0].right;
        let evidence = HypothesisEvidence {
            manifest: Some(&run.trace_archive.manifest),
            trace_ref: None,
            observed: planner_metrics(&report.right_overall),
            build: BuildStatus::Fail,
        };
        let rows = evaluate_hypotheses(&hypotheses, &evidence).unwrap();
        let counts = count_verdicts(&rows);
        assert_eq!(
            counts,
            VerdictCounts {
                supported: 1,
                refuted: 1,
                inconclusive: 1
            }
        );
        let table = render_verdict_table(&rows);
        assert!(
            table.contains("SUPPORTED=1 REFUTED=1 INCONCLUSIVE=1"),
            "{table}"
        );
        // Build FAIL does not alter verdicts.
        assert_eq!(rows[0].verdict, HypothesisVerdict::Refuted);
        assert!(rows.iter().all(|r| table.contains(&r.evidence_ref)));
    }
}
