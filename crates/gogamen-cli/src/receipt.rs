//! Experiment receipt: a self-describing, re-verifiable record of one experiment run.
//!
//! A receipt stores the experiment specification and its hash, the pre-registered hypotheses
//! hash, every seed, the gogamen version, the endowment ledger (what was handed to the agents and
//! what was derived or measured), per-run outcomes with their deterministic hashes, aggregate
//! curves with dispersion, and the hypothesis verdicts.
//!
//! `verify_receipt` re-executes the experiment from the specification stored in the receipt and
//! compares the full reproduced body against the stored one, so any edit (a threshold, a seed, a
//! verdict, a hash) is reported with the path of the first difference.
//!
//! Hashes are 64-bit FNV-1a: they detect accidental or casual edits and non-reproducible runs,
//! they are not cryptographic signatures. Someone who edits a receipt *and* recomputes its
//! hashes is still caught by re-execution, but only because the stored body is compared with the
//! reproduced one, not because the hash cannot be forged.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gogamen_arena::{ArenaSeedSet, VerdictEntry};
use gogamen_arena::{
    BuildStatus, HypothesisEvidence, KartArenaBudget, KartArenaConfig, KartArenaReport, KartRole,
    KartRoleSummary, KartRunResult, evaluate_hypotheses, run_kart_arena_preregistered,
};
use gogamen_core::{EndowmentProvenance, MasterSeed};
use gogamen_lab_racing::TrackMap;
use gogamen_spec::{
    HypothesisComparator, HypothesisSpec, JsonNumber, JsonValue, hypotheses_hash, parse_json,
    write_canonical_f64, write_json_string,
};

/// Receipt format identifier.
pub const RECEIPT_FORMAT: &str = "gogamen.receipt.v1";
/// Identifier of the first bundled example.
pub const KART_EXAMPLE_ID: &str = "kart-pure-vs-random-vs-ceiling";
/// Maximum accepted receipt size when verifying (bytes).
pub const MAX_RECEIPT_BYTES: usize = 8 * 1024 * 1024;

/// Specification of one Kart experiment (everything needed to reproduce it).
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSpec {
    /// Example identifier.
    pub example_id: String,
    /// Track name (`rectangular` or `chicane`).
    pub track: String,
    /// Maximum ticks per episode.
    pub max_ticks: usize,
    /// Planning horizon shared by the planning roles.
    pub horizon: usize,
    /// Simulated transitions per decision shared by the fair roles.
    pub transitions_per_decision: usize,
    /// Development seeds.
    pub dev_seeds: Vec<u64>,
    /// Held-out seeds.
    pub holdout_seeds: Vec<u64>,
    /// Hypotheses to pre-register (may be empty).
    pub hypotheses: Vec<HypothesisSpec>,
}

impl ExperimentSpec {
    /// Default spec: pure reward-free controller versus random, with the privileged FMC as ceiling.
    pub fn kart_default(track: &str) -> Self {
        let h = |id: &str,
                 claim: &str,
                 metric: &str,
                 c: HypothesisComparator,
                 t: f64,
                 rc: HypothesisComparator,
                 rt: f64| HypothesisSpec::new(id, claim, metric, c, t, rc, rt);
        Self {
            example_id: KART_EXAMPLE_ID.to_owned(),
            track: track.to_owned(),
            max_ticks: 120,
            horizon: 6,
            transitions_per_decision: 9 * 6 * 2,
            dev_seeds: vec![1, 2, 3, 4],
            holdout_seeds: vec![101, 102, 103, 104],
            hypotheses: vec![
                h(
                    "H1-pure-beats-random",
                    "The pure reward-free controller clears at least 0.10 more of the track than random",
                    "pure_minus_random.efficacy_gap",
                    HypothesisComparator::Ge,
                    0.10,
                    HypothesisComparator::Le,
                    0.0,
                ),
                h(
                    "H2-pure-is-viable",
                    "At least 75% of the pure controller's episodes end without a collision",
                    "pure.viable_trajectory_rate",
                    HypothesisComparator::Ge,
                    0.75,
                    HypothesisComparator::Lt,
                    0.25,
                ),
                h(
                    "H3-near-ceiling",
                    "The pure controller is within 0.10 efficacy of the privileged 4x-budget ceiling",
                    "ceiling_minus_pure.efficacy_gap",
                    HypothesisComparator::Le,
                    0.10,
                    HypothesisComparator::Gt,
                    0.30,
                ),
            ],
        }
    }

    /// Canonical JSON of the specification.
    pub fn to_json(&self) -> JsonValue {
        let seeds = |v: &[u64]| arr(v.iter().map(|s| num_u(*s)).collect());
        let hyps = arr(self
            .hypotheses
            .iter()
            .map(|h| {
                obj([
                    ("hypothesis_id", st(&h.hypothesis_id)),
                    ("claim", st(&h.claim)),
                    ("metric", st(&h.metric)),
                    ("comparator", st(h.comparator.as_str())),
                    ("threshold", JsonValue::Number(JsonNumber::F64(h.threshold))),
                    (
                        "refutation_comparator",
                        st(h.refutation_comparator.as_str()),
                    ),
                    (
                        "refutation_threshold",
                        JsonValue::Number(JsonNumber::F64(h.refutation_threshold)),
                    ),
                ])
            })
            .collect());
        obj([
            ("example_id", st(&self.example_id)),
            ("track", st(&self.track)),
            ("max_ticks", num_u(self.max_ticks as u64)),
            ("horizon", num_u(self.horizon as u64)),
            (
                "transitions_per_decision",
                num_u(self.transitions_per_decision as u64),
            ),
            ("dev_seeds", seeds(&self.dev_seeds)),
            ("holdout_seeds", seeds(&self.holdout_seeds)),
            ("hypotheses", hyps),
        ])
    }

    /// Parses a specification previously produced by [`Self::to_json`].
    pub fn from_json(v: &JsonValue) -> Result<Self, String> {
        let o = as_obj(v, "spec")?;
        let seeds = |key: &str| -> Result<Vec<u64>, String> {
            as_arr(get(o, key)?, key)?
                .iter()
                .map(|x| as_u64(x, key))
                .collect()
        };
        let mut hypotheses = Vec::new();
        for h in as_arr(get(o, "hypotheses")?, "hypotheses")? {
            let ho = as_obj(h, "hypothesis")?;
            let cmp = |key: &str| -> Result<HypothesisComparator, String> {
                let s = as_str(get(ho, key)?, key)?;
                HypothesisComparator::parse(s).ok_or_else(|| format!("unknown comparator '{s}'"))
            };
            hypotheses.push(HypothesisSpec::new(
                as_str(get(ho, "hypothesis_id")?, "hypothesis_id")?,
                as_str(get(ho, "claim")?, "claim")?,
                as_str(get(ho, "metric")?, "metric")?,
                cmp("comparator")?,
                as_f64(get(ho, "threshold")?, "threshold")?,
                cmp("refutation_comparator")?,
                as_f64(get(ho, "refutation_threshold")?, "refutation_threshold")?,
            ));
        }
        Ok(Self {
            example_id: as_str(get(o, "example_id")?, "example_id")?.to_owned(),
            track: as_str(get(o, "track")?, "track")?.to_owned(),
            max_ticks: as_u64(get(o, "max_ticks")?, "max_ticks")? as usize,
            horizon: as_u64(get(o, "horizon")?, "horizon")? as usize,
            transitions_per_decision: as_u64(
                get(o, "transitions_per_decision")?,
                "transitions_per_decision",
            )? as usize,
            dev_seeds: seeds("dev_seeds")?,
            holdout_seeds: seeds("holdout_seeds")?,
            hypotheses,
        })
    }

    /// Hash of the canonical specification JSON.
    pub fn spec_hash(&self) -> u64 {
        fnv1a(canonical_json(&self.to_json()).as_bytes())
    }

    fn arena_config(&self) -> Result<KartArenaConfig, String> {
        if self.example_id != KART_EXAMPLE_ID {
            return Err(format!("unknown example '{}'", self.example_id));
        }
        let track = match self.track.as_str() {
            "rectangular" => TrackMap::rectangular_circuit(),
            "chicane" => TrackMap::chicane_circuit(),
            other => return Err(format!("unknown track '{other}'")),
        };
        let budget =
            KartArenaBudget::new(self.max_ticks, self.horizon, self.transitions_per_decision)
                .map_err(|e| e.to_string())?;
        let seeds = ArenaSeedSet::new(
            self.dev_seeds
                .iter()
                .copied()
                .map(MasterSeed::new)
                .collect(),
            self.holdout_seeds
                .iter()
                .copied()
                .map(MasterSeed::new)
                .collect(),
        )
        .map_err(|e| e.to_string())?;
        Ok(KartArenaConfig::new(
            self.example_id.clone(),
            track,
            budget,
            seeds,
        ))
    }
}

/// One row of the endowment ledger.
#[derive(Debug, Clone, PartialEq)]
pub struct LedgerEntry {
    /// Quantity name.
    pub name: String,
    /// Human-readable value.
    pub value: String,
    /// Where the quantity comes from.
    pub provenance: EndowmentProvenance,
    /// Which roles receive it.
    pub scope: String,
}

/// Per-tick aggregate of one role across seeds.
#[derive(Debug, Clone, PartialEq)]
pub struct RoleCurve {
    /// Role label.
    pub role: String,
    /// `true` for the privileged ceiling.
    pub privileged: bool,
    /// Mean checkpoints cleared per tick.
    pub mean: Vec<f64>,
    /// Population standard deviation per tick.
    pub std_dev: Vec<f64>,
    /// Minimum per tick.
    pub min: Vec<f64>,
    /// Maximum per tick.
    pub max: Vec<f64>,
}

/// Everything an experiment produced (the receipt body plus the data the report needs).
#[derive(Debug, Clone)]
pub struct ExperimentOutcome {
    /// Specification as executed.
    pub spec: ExperimentSpec,
    /// `true` when the hypotheses hash was stamped before execution.
    pub preregistered: bool,
    /// Hash of the specification.
    pub spec_hash: u64,
    /// Pre-registered hypotheses hash, when stamped.
    pub registered_hypotheses_hash: Option<u64>,
    /// Hash of the full arena report (excludes wall-clock latency).
    pub report_hash: u64,
    /// Endowment ledger.
    pub ledger: Vec<LedgerEntry>,
    /// Per-role summaries (fair roles first, then the ceiling).
    pub summaries: Vec<KartRoleSummary>,
    /// Every episode, including failed ones.
    pub runs: Vec<RunRecord>,
    /// Curves with dispersion per role.
    pub curves: Vec<RoleCurve>,
    /// Observed metrics hypotheses refer to.
    pub metrics: BTreeMap<String, f64>,
    /// Hypothesis verdicts.
    pub verdicts: Vec<VerdictEntry>,
}

/// Compact record of one episode.
#[derive(Debug, Clone, PartialEq)]
pub struct RunRecord {
    /// Role label.
    pub role: String,
    /// `true` for the privileged ceiling.
    pub privileged: bool,
    /// Seed split label.
    pub split: String,
    /// Seed.
    pub seed: u64,
    /// Outcome label.
    pub status: String,
    /// Failure reason, if the episode did not reach the goal.
    pub reason: Option<String>,
    /// Ticks executed.
    pub steps: u64,
    /// Checkpoints cleared.
    pub checkpoints: u64,
    /// Collisions.
    pub collisions: u64,
    /// Deterministic hash of the controls and result.
    pub control_hash: u64,
    /// Checksum of the scientific trace archive.
    pub trace_checksum: u64,
}

fn split_label(split: gogamen_arena::SeedSplit) -> &'static str {
    match split {
        gogamen_arena::SeedSplit::Dev => "dev",
        gogamen_arena::SeedSplit::Holdout => "holdout",
    }
}

/// Runs the experiment. With `preregister = false` the hypotheses hash is not stamped into the
/// traces, so no hypothesis can come out `SUPPORTED`.
pub fn run_experiment(
    spec: &ExperimentSpec,
    preregister: bool,
) -> Result<ExperimentOutcome, String> {
    let config = spec.arena_config()?;
    let hash = hypotheses_hash(&spec.hypotheses).map_err(|e| format!("invalid hypotheses: {e}"))?;
    let stamped = if preregister { hash } else { None };
    let report = run_kart_arena_preregistered(&config, stamped).map_err(|e| e.to_string())?;

    let all_runs: Vec<&KartRunResult> = report
        .outcomes
        .iter()
        .flat_map(|o| o.fair_runs.iter().chain(&o.ceiling_runs))
        .collect();
    let runs: Vec<RunRecord> = all_runs
        .iter()
        .map(|r| RunRecord {
            role: r.role.label().to_owned(),
            privileged: r.is_privileged,
            split: split_label(r.split).to_owned(),
            seed: r.seed.as_u64(),
            status: r.status.label().to_owned(),
            reason: r.status.failure_reason().map(str::to_owned),
            steps: r.steps_executed,
            checkpoints: r.checkpoints_cleared as u64,
            collisions: r.collisions as u64,
            control_hash: r.control_hash,
            trace_checksum: r.trace_manifest.checksum,
        })
        .collect();

    let curves = build_curves(&report, &all_runs, spec.max_ticks);
    let summaries: Vec<KartRoleSummary> = report
        .fair_summaries
        .iter()
        .chain(&report.ceiling_summaries)
        .cloned()
        .collect();
    let metrics = build_metrics(&summaries);

    // Conservative evidence: use the first manifest that is incomplete or lacks the registered
    // hash if there is one, otherwise the first manifest.
    let manifest = all_runs
        .iter()
        .map(|r| &r.trace_manifest)
        .find(|m| !m.is_complete || m.hypotheses_hash != stamped)
        .or_else(|| all_runs.first().map(|r| &r.trace_manifest));
    let evidence = HypothesisEvidence {
        manifest,
        trace_ref: None,
        observed: metrics.clone(),
        build: BuildStatus::NotRun,
    };
    let verdicts = evaluate_hypotheses(&spec.hypotheses, &evidence)
        .map_err(|e| format!("invalid hypotheses: {e}"))?;

    Ok(ExperimentOutcome {
        spec: spec.clone(),
        preregistered: stamped.is_some(),
        spec_hash: spec.spec_hash(),
        registered_hypotheses_hash: stamped,
        report_hash: report.report_hash(),
        ledger: build_ledger(spec, &config, &all_runs),
        summaries,
        runs,
        curves,
        metrics,
        verdicts,
    })
}

fn build_curves(report: &KartArenaReport, runs: &[&KartRunResult], ticks: usize) -> Vec<RoleCurve> {
    let mut roles: Vec<KartRole> = report
        .fair_summaries
        .iter()
        .chain(&report.ceiling_summaries)
        .map(|s| s.role)
        .collect();
    roles.dedup();
    roles
        .into_iter()
        .map(|role| {
            // Per episode: checkpoints cleared at the start of each tick, held at the final value
            // once the episode ended (success, crash or timeout).
            let series: Vec<Vec<f64>> = runs
                .iter()
                .filter(|r| r.role == role)
                .map(|r| {
                    let mut s: Vec<f64> = r
                        .trace_archive
                        .records
                        .iter()
                        .map(|rec| {
                            rec.inputs
                                .observation
                                .features()
                                .get(4)
                                .copied()
                                .unwrap_or(0.0)
                        })
                        .collect();
                    s.truncate(ticks);
                    s.resize(ticks, r.checkpoints_cleared as f64);
                    s
                })
                .collect();
            let mut curve = RoleCurve {
                role: role.label().to_owned(),
                privileged: role.is_privileged(),
                mean: Vec::new(),
                std_dev: Vec::new(),
                min: Vec::new(),
                max: Vec::new(),
            };
            for t in 0..ticks {
                let col: Vec<f64> = series.iter().map(|s| s[t]).collect();
                let d = gogamen_arena::DispersionStats::compute(&col);
                curve.mean.push(d.mean);
                curve.std_dev.push(d.std_dev);
                curve.min.push(d.min);
                curve.max.push(d.max);
            }
            curve
        })
        .collect()
}

fn build_metrics(summaries: &[KartRoleSummary]) -> BTreeMap<String, f64> {
    let find = |role: KartRole| summaries.iter().find(|s| s.role == role);
    let mut m = BTreeMap::new();
    let eff = |role: KartRole| find(role).map(|s| s.mean_efficacy);
    for (name, role) in [
        ("pure", KartRole::PureEntropic),
        ("goal_fmc", KartRole::GoalOrientedFmc),
        ("random", KartRole::Random),
        ("geometric", KartRole::GeometricBaseline),
        ("ceiling", KartRole::PrivilegedFmcCeiling),
    ] {
        if let Some(s) = find(role) {
            m.insert(format!("{name}.mean_efficacy"), s.mean_efficacy);
            m.insert(
                format!("{name}.viable_trajectory_rate"),
                s.viable_trajectories as f64 / s.total_runs.max(1) as f64,
            );
        }
    }
    if let (Some(p), Some(r)) = (eff(KartRole::PureEntropic), eff(KartRole::Random)) {
        m.insert("pure_minus_random.efficacy_gap".to_owned(), p - r);
    }
    if let (Some(c), Some(p)) = (
        eff(KartRole::PrivilegedFmcCeiling),
        eff(KartRole::PureEntropic),
    ) {
        m.insert("ceiling_minus_pure.efficacy_gap".to_owned(), c - p);
    }
    m
}

fn build_ledger(
    spec: &ExperimentSpec,
    config: &KartArenaConfig,
    runs: &[&KartRunResult],
) -> Vec<LedgerEntry> {
    let e = |name: &str, value: String, provenance, scope: &str| LedgerEntry {
        name: name.to_owned(),
        value,
        provenance,
        scope: scope.to_owned(),
    };
    let given = EndowmentProvenance::Given;
    let viable: Vec<f64> = runs
        .iter()
        .filter(|r| r.role == KartRole::PureEntropic)
        .filter_map(|r| r.viable_future_fraction)
        .collect();
    let mean_viable = if viable.is_empty() {
        0.0
    } else {
        viable.iter().sum::<f64>() / viable.len() as f64
    };
    let w = &config.weights;
    vec![
        e("track", spec.track.clone(), given, "all roles"),
        e(
            "kinematics",
            "KinematicBicycleConfig::default".to_owned(),
            given,
            "all roles",
        ),
        e(
            "observation_contract",
            "KartObservation::FullKinematics+checkpoints_cleared".to_owned(),
            given,
            "all roles",
        ),
        e("max_ticks", spec.max_ticks.to_string(), given, "all roles"),
        e(
            "horizon",
            spec.horizon.to_string(),
            given,
            "pure, goal_fmc, ceiling",
        ),
        e(
            "transitions_per_decision",
            spec.transitions_per_decision.to_string(),
            given,
            "pure, goal_fmc (equal, enforced)",
        ),
        e(
            "ceiling_transitions_per_decision",
            (spec.transitions_per_decision * 4).to_string(),
            given,
            "privileged ceiling only (never in the fair comparison)",
        ),
        e(
            "reward_function",
            "none (reward-free)".to_owned(),
            given,
            "pure",
        ),
        e(
            "objective_weights",
            format!(
                "checkpoint_bonus={}, lap_goal_bonus={}, progress_weight={}, forward_speed_weight={}, collision_penalty={}",
                w.checkpoint_bonus,
                w.lap_goal_bonus,
                w.progress_weight,
                w.forward_speed_weight,
                w.collision_penalty
            ),
            given,
            "goal_fmc, ceiling",
        ),
        e(
            "seeds",
            format!("dev={:?} holdout={:?}", spec.dev_seeds, spec.holdout_seeds),
            given,
            "all roles (paired)",
        ),
        e(
            "viable_future_fraction_mean",
            format!("{mean_viable}"),
            EndowmentProvenance::Derived,
            "pure (computed from its own simulated futures)",
        ),
        e(
            "checkpoints_cleared / collisions",
            "per episode, see runs".to_owned(),
            EndowmentProvenance::Measured,
            "all roles",
        ),
        e(
            "learned quantities",
            "none in this example".to_owned(),
            EndowmentProvenance::Learned,
            "n/a",
        ),
    ]
}

// ---------------------------------------------------------------------------------------------
// Receipt (JSON) construction and verification
// ---------------------------------------------------------------------------------------------

pub(crate) fn hex(h: u64) -> JsonValue {
    JsonValue::String(format!("{h:016x}"))
}

impl ExperimentOutcome {
    /// Receipt body (everything covered by the receipt hash).
    pub fn body_json(&self) -> JsonValue {
        let f = |v: f64| JsonValue::Number(JsonNumber::F64(v));
        let fa = |v: &[f64]| arr(v.iter().map(|x| f(*x)).collect());
        let ledger = arr(self
            .ledger
            .iter()
            .map(|l| {
                obj([
                    ("name", st(&l.name)),
                    ("value", st(&l.value)),
                    ("provenance", st(l.provenance.label())),
                    ("scope", st(&l.scope)),
                ])
            })
            .collect());
        let runs = arr(self
            .runs
            .iter()
            .map(|r| {
                obj([
                    ("role", st(&r.role)),
                    ("privileged", JsonValue::Bool(r.privileged)),
                    ("split", st(&r.split)),
                    ("seed", num_u(r.seed)),
                    ("status", st(&r.status)),
                    ("reason", r.reason.as_deref().map_or(JsonValue::Null, st)),
                    ("steps", num_u(r.steps)),
                    ("checkpoints", num_u(r.checkpoints)),
                    ("collisions", num_u(r.collisions)),
                    ("control_hash", hex(r.control_hash)),
                    ("trace_checksum", hex(r.trace_checksum)),
                ])
            })
            .collect());
        let summaries = arr(self
            .summaries
            .iter()
            .map(|s| {
                let d = &s.efficacy_dispersion;
                obj([
                    ("role", st(s.role.label())),
                    ("privileged", JsonValue::Bool(s.is_privileged)),
                    ("total_runs", num_u(s.total_runs as u64)),
                    ("succeeded", num_u(s.succeeded_count as u64)),
                    ("failed", num_u(s.failed_count as u64)),
                    ("timed_out", num_u(s.timed_out_count as u64)),
                    ("efficacy_mean", f(d.mean)),
                    ("efficacy_std_dev", f(d.std_dev)),
                    ("efficacy_min", f(d.min)),
                    ("efficacy_q25", f(d.q25)),
                    ("efficacy_median", f(d.median)),
                    ("efficacy_q75", f(d.q75)),
                    ("efficacy_max", f(d.max)),
                ])
            })
            .collect());
        let curves = arr(self
            .curves
            .iter()
            .map(|c| {
                obj([
                    ("role", st(&c.role)),
                    ("privileged", JsonValue::Bool(c.privileged)),
                    ("mean", fa(&c.mean)),
                    ("std_dev", fa(&c.std_dev)),
                    ("min", fa(&c.min)),
                    ("max", fa(&c.max)),
                ])
            })
            .collect());
        let metrics = JsonValue::Object(
            self.metrics
                .iter()
                .map(|(k, v)| (k.clone(), f(*v)))
                .collect(),
        );
        let verdicts = arr(self
            .verdicts
            .iter()
            .map(|v| {
                obj([
                    ("hypothesis_id", st(&v.hypothesis_id)),
                    ("claim", st(&v.claim)),
                    ("metric", st(&v.metric)),
                    ("observed", v.observed.map_or(JsonValue::Null, f)),
                    ("verdict", st(v.verdict.label())),
                    ("basis", st(v.basis.label())),
                    (
                        "registered_hash",
                        v.registered_hash.map_or(JsonValue::Null, hex),
                    ),
                    ("spec_hash", v.spec_hash.map_or(JsonValue::Null, hex)),
                    ("evidence_ref", st(&v.evidence_ref)),
                    ("build", st(v.build.label())),
                ])
            })
            .collect());
        obj([
            ("format", st(RECEIPT_FORMAT)),
            ("gogamen_version", st(env!("CARGO_PKG_VERSION"))),
            ("spec", self.spec.to_json()),
            ("spec_hash", hex(self.spec_hash)),
            ("preregistered", JsonValue::Bool(self.preregistered)),
            (
                "registered_hypotheses_hash",
                self.registered_hypotheses_hash.map_or(JsonValue::Null, hex),
            ),
            ("endowment_ledger", ledger),
            ("report_hash", hex(self.report_hash)),
            ("runs", runs),
            ("summaries", summaries),
            ("curves", curves),
            ("metrics", metrics),
            ("verdicts", verdicts),
        ])
    }

    /// Receipt hash over the canonical body.
    pub fn receipt_hash(&self) -> u64 {
        fnv1a(canonical_json(&self.body_json()).as_bytes())
    }

    /// Serialized receipt: `{"body":...,"receipt_hash":"..."}` (canonical, trailing newline).
    pub fn receipt_text(&self) -> String {
        let doc = obj([
            ("body", self.body_json()),
            ("receipt_hash", hex(self.receipt_hash())),
        ]);
        let mut s = canonical_json(&doc);
        s.push('\n');
        s
    }
}

/// Successful verification summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verification {
    /// Hash of the receipt body.
    pub receipt_hash: u64,
    /// Hash of the specification.
    pub spec_hash: u64,
    /// Hash of the arena report.
    pub report_hash: u64,
    /// Number of episodes re-executed and matched.
    pub runs_reproduced: usize,
}

/// Verifies a receipt by re-executing it. Errors carry the first difference found.
pub fn verify_receipt(text: &str) -> Result<Verification, String> {
    let doc = parse_json(text).map_err(|e| format!("receipt is not valid JSON: {e}"))?;
    let top = as_obj(&doc, "receipt")?;
    let body = get(top, "body")?;
    let stored_hash = as_str(get(top, "receipt_hash")?, "receipt_hash")?;
    let computed = format!("{:016x}", fnv1a(canonical_json(body).as_bytes()));
    if stored_hash != computed {
        return Err(format!(
            "receipt hash mismatch: stored {stored_hash}, body hashes to {computed} (the receipt was edited)"
        ));
    }
    let b = as_obj(body, "body")?;
    if as_str(get(b, "format")?, "format")? != RECEIPT_FORMAT {
        return Err("unsupported receipt format".to_owned());
    }
    let spec = ExperimentSpec::from_json(get(b, "spec")?)?;
    let preregistered = matches!(get(b, "preregistered")?, JsonValue::Bool(true));
    let fresh = run_experiment(&spec, preregistered)?;
    let fresh_body = parse_json(&canonical_json(&fresh.body_json()))
        .map_err(|e| format!("internal error re-reading reproduced body: {e}"))?;
    if let Some(diff) = first_difference(body, &fresh_body, "body") {
        return Err(format!("reproduction differs from the receipt: {diff}"));
    }
    Ok(Verification {
        receipt_hash: fresh.receipt_hash(),
        spec_hash: fresh.spec_hash,
        report_hash: fresh.report_hash,
        runs_reproduced: fresh.runs.len(),
    })
}

pub(crate) fn first_difference(
    stored: &JsonValue,
    fresh: &JsonValue,
    path: &str,
) -> Option<String> {
    match (stored, fresh) {
        (JsonValue::Object(a), JsonValue::Object(b)) => {
            for k in a.keys().chain(b.keys()) {
                match (a.get(k), b.get(k)) {
                    (Some(x), Some(y)) => {
                        if let Some(d) = first_difference(x, y, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    _ => return Some(format!("{path}.{k}: field present on one side only")),
                }
            }
            None
        }
        (JsonValue::Array(a), JsonValue::Array(b)) => {
            if a.len() != b.len() {
                return Some(format!(
                    "{path}: stored length {}, reproduced length {}",
                    a.len(),
                    b.len()
                ));
            }
            a.iter()
                .zip(b)
                .enumerate()
                .find_map(|(i, (x, y))| first_difference(x, y, &format!("{path}[{i}]")))
        }
        _ if json_scalar_eq(stored, fresh) => None,
        _ => Some(format!(
            "{path}: stored {}, reproduced {}",
            canonical_json(stored),
            canonical_json(fresh)
        )),
    }
}

fn json_scalar_eq(a: &JsonValue, b: &JsonValue) -> bool {
    match (a, b) {
        (JsonValue::Number(x), JsonValue::Number(y)) => match (num_i128(x), num_i128(y)) {
            (Some(p), Some(q)) => p == q,
            _ => num_f64(x) == num_f64(y),
        },
        _ => a == b,
    }
}

fn num_i128(n: &JsonNumber) -> Option<i128> {
    match n {
        JsonNumber::I64(v) => Some(i128::from(*v)),
        JsonNumber::U64(v) => Some(i128::from(*v)),
        JsonNumber::F64(_) => None,
    }
}

fn num_f64(n: &JsonNumber) -> f64 {
    match n {
        JsonNumber::I64(v) => *v as f64,
        JsonNumber::U64(v) => *v as f64,
        JsonNumber::F64(v) => *v,
    }
}

// ---------------------------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------------------------

pub(crate) fn st(s: &str) -> JsonValue {
    JsonValue::String(s.to_owned())
}

pub(crate) fn num_u(v: u64) -> JsonValue {
    JsonValue::Number(JsonNumber::U64(v))
}

pub(crate) fn arr(v: Vec<JsonValue>) -> JsonValue {
    JsonValue::Array(v)
}

pub(crate) fn obj<const N: usize>(fields: [(&str, JsonValue); N]) -> JsonValue {
    JsonValue::Object(fields.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

type JsonObject = BTreeMap<String, JsonValue>;

pub(crate) fn as_obj<'a>(v: &'a JsonValue, what: &str) -> Result<&'a JsonObject, String> {
    match v {
        JsonValue::Object(o) => Ok(o),
        _ => Err(format!("{what}: expected an object")),
    }
}

fn as_arr<'a>(v: &'a JsonValue, what: &str) -> Result<&'a Vec<JsonValue>, String> {
    match v {
        JsonValue::Array(a) => Ok(a),
        _ => Err(format!("{what}: expected an array")),
    }
}

pub(crate) fn as_str<'a>(v: &'a JsonValue, what: &str) -> Result<&'a str, String> {
    match v {
        JsonValue::String(s) => Ok(s),
        _ => Err(format!("{what}: expected a string")),
    }
}

fn as_u64(v: &JsonValue, what: &str) -> Result<u64, String> {
    match v {
        JsonValue::Number(JsonNumber::U64(n)) => Ok(*n),
        JsonValue::Number(JsonNumber::I64(n)) if *n >= 0 => Ok(*n as u64),
        _ => Err(format!("{what}: expected a non-negative integer")),
    }
}

fn as_f64(v: &JsonValue, what: &str) -> Result<f64, String> {
    match v {
        JsonValue::Number(n) => Ok(num_f64(n)),
        _ => Err(format!("{what}: expected a number")),
    }
}

pub(crate) fn get<'a>(o: &'a JsonObject, key: &str) -> Result<&'a JsonValue, String> {
    o.get(key).ok_or_else(|| format!("missing field '{key}'"))
}

/// Canonical JSON text (sorted keys, no whitespace).
pub fn canonical_json(v: &JsonValue) -> String {
    let mut out = String::new();
    write_value(&mut out, v);
    out
}

fn write_value(out: &mut String, v: &JsonValue) {
    match v {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        JsonValue::Number(JsonNumber::I64(n)) => {
            let _ = write!(out, "{n}");
        }
        JsonValue::Number(JsonNumber::U64(n)) => {
            let _ = write!(out, "{n}");
        }
        JsonValue::Number(JsonNumber::F64(n)) => {
            if write_canonical_f64(out, *n).is_err() {
                out.push_str("null");
            }
        }
        JsonValue::String(s) => write_json_string(out, s),
        JsonValue::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(out, x);
            }
            out.push(']');
        }
        JsonValue::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_json_string(out, k);
                out.push(':');
                write_value(out, x);
            }
            out.push('}');
        }
    }
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_arena::HypothesisVerdict;

    /// Small spec so the tests stay fast.
    fn small_spec() -> ExperimentSpec {
        let mut s = ExperimentSpec::kart_default("rectangular");
        s.max_ticks = 30;
        s.dev_seeds = vec![1, 2];
        s.holdout_seeds = vec![101];
        s
    }

    #[test]
    fn receipt_roundtrip() {
        let spec = small_spec();
        let a = run_experiment(&spec, true).unwrap();
        let text = a.receipt_text();
        // Stable between executions (hash and text).
        let b = run_experiment(&spec, true).unwrap();
        assert_eq!(text, b.receipt_text());
        assert_eq!(a.receipt_hash(), b.receipt_hash());
        // Verifies by re-execution.
        let v = verify_receipt(&text).unwrap();
        assert_eq!(v.receipt_hash, a.receipt_hash());
        assert_eq!(v.runs_reproduced, 3 * 5);
        // Contains spec hash, seeds, version, ledger, verdicts.
        assert!(text.contains(&format!("{:016x}", a.spec_hash)));
        assert!(text.contains("gogamen_version"));
        assert!(text.contains("endowment_ledger"));
        assert!(text.contains("verdicts"));
    }

    #[test]
    fn receipt_tampering_is_detected_with_clear_error() {
        let spec = small_spec();
        let a = run_experiment(&spec, true).unwrap();
        let text = a.receipt_text();

        // 1. Edited field without fixing the receipt hash.
        let edited = text.replacen("\"seed\":1,", "\"seed\":9,", 1);
        assert_ne!(edited, text);
        let err = verify_receipt(&edited).unwrap_err();
        assert!(err.contains("receipt hash mismatch"), "{err}");

        // 2. Edited threshold AND recomputed receipt hash: caught by re-execution.
        let mut doc = parse_json(&text).unwrap();
        let JsonValue::Object(top) = &mut doc else {
            panic!()
        };
        let JsonValue::Object(body) = top.get_mut("body").unwrap() else {
            panic!()
        };
        let JsonValue::Array(verdicts) = body.get_mut("verdicts").unwrap() else {
            panic!()
        };
        let JsonValue::Object(v0) = &mut verdicts[0] else {
            panic!()
        };
        v0.insert("verdict".to_owned(), st("SUPPORTED"));
        v0.insert("basis".to_owned(), st("SUPPORT_CONDITION_MET"));
        let new_hash = format!(
            "{:016x}",
            fnv1a(canonical_json(top.get("body").unwrap()).as_bytes())
        );
        top.insert("receipt_hash".to_owned(), JsonValue::String(new_hash));
        let forged = canonical_json(&doc);
        let err = verify_receipt(&forged).unwrap_err();
        assert!(err.contains("reproduction differs"), "{err}");
        assert!(err.contains("verdicts[0]"), "{err}");

        // 3. Not JSON.
        assert!(
            verify_receipt("garbage")
                .unwrap_err()
                .contains("not valid JSON")
        );
    }

    #[test]
    fn no_hypothesis_is_supported_without_preregistration() {
        let spec = small_spec();
        let unregistered = run_experiment(&spec, false).unwrap();
        assert!(!unregistered.preregistered);
        assert_eq!(unregistered.registered_hypotheses_hash, None);
        assert!(!unregistered.verdicts.is_empty());
        assert!(unregistered.verdicts.iter().all(|v| {
            v.verdict == HypothesisVerdict::Inconclusive
                && v.basis == gogamen_arena::VerdictBasis::NotPreregistered
        }));
        // Even a hypothesis that trivially holds cannot be SUPPORTED without a registered hash.
        let mut trivial = small_spec();
        trivial.hypotheses = vec![HypothesisSpec::new(
            "H-trivial",
            "efficacy is non-negative",
            "pure.mean_efficacy",
            HypothesisComparator::Ge,
            0.0,
            HypothesisComparator::Lt,
            -1.0,
        )];
        let o = run_experiment(&trivial, false).unwrap();
        assert!(o.verdicts[0].observed.is_some());
        assert_eq!(o.verdicts[0].verdict, HypothesisVerdict::Inconclusive);
        // The same hypothesis, pre-registered, is SUPPORTED (the rule is about registration).
        let o = run_experiment(&trivial, true).unwrap();
        assert_eq!(o.verdicts[0].verdict, HypothesisVerdict::Supported);
        // A no-prereg receipt still verifies (it records preregistered=false).
        verify_receipt(&unregistered.receipt_text()).unwrap();
    }

    #[test]
    fn editing_hypotheses_after_the_run_is_flagged() {
        let spec = small_spec();
        let text = run_experiment(&spec, true).unwrap().receipt_text();
        // Change a threshold in the stored spec; hash covers it.
        let edited = text.replacen("\"threshold\":0.1}", "\"threshold\":0.01}", 1);
        assert_ne!(edited, text);
        assert!(verify_receipt(&edited).is_err());
    }
}
