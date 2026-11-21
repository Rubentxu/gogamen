//! Emergence view model (GOG-177, GOG-205, DEM-024): data slice over the existing Studio layers.
//!
//! Shows, for one decision, the scale that was chosen, the hypothesis/model version and evidence
//! that were used, the predicates, and the hypothetical futures that were considered, next to the
//! real-world trace from a [`Recording`]. It is a headless view model (like
//! `decisive_timeline`): it reuses [`Known`] from the cognitive inspector, [`Recording`] from the
//! replay layer and the GOG-175 [`NestedDecision`] record. The painter lives in
//! [`crate::emergence_ui`], which consumes [`DecisionView::lines`] / [`PolicyComparison::lines`].
//!
//! Since GOG-205, [`DecisionRecord`] also has adapters from the real hypothesis/model sources:
//! [`DecisionRecord::from_learned_model`], [`DecisionRecord::from_conditional_effects`],
//! [`DecisionRecord::from_causal_revision`] and [`DecisionRecord::from_predicate_invention`].
//! Each one copies the source's own revision, checksum (when it has one), evidence counts and
//! predicates; nothing is synthesized that the source does not carry.
//!
//! Honesty rules:
//! - A field that the record does not carry is [`Known::Unknown`] and is printed `unknown`.
//! - [`ModelRef::checksum`] is itself [`Known`]: sources such as `ModelRevision` version their
//!   rules but do not publish a checksum, so their records say `unknown` instead of borrowing a
//!   hash from somewhere else.
//! - Real-world steps ([`RealStep`], from the recording) and hypothetical futures
//!   ([`HypotheticalFuture`]) are different types, printed in separate lanes, never merged.
//! - Everything is read-only: the recording is only borrowed.
//! - The comparison only contrasts fields recorded by the model; it has no field for human
//!   intentions, goals or motives beyond the macro goal the model itself emitted.

use gogamen_core::causal_revision::CausalRevisor;
use gogamen_core::conditional_effects::{
    BitPrediction, ConditionalEffectsLearner, Literal, PredictionHealth,
};
use gogamen_core::learned_model::{LearnedStepReport, TabularLearnedModel};
use gogamen_core::predicate_invention::{CandidateVerdict, InventionOutcome, Predicate};
use gogamen_lab_grid::conditional_effects::{GatedChoice, gated_choice};
use gogamen_lab_grid::nested_scales::{Level, NestedDecision};

use crate::cognitive::Known;
use crate::replay::Recording;

/// Frozen model identity (e.g. `ModelVersion::revision()` / `checksum()`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelRef {
    pub revision: u64,
    /// Frozen checksum; [`Known::Unknown`] when the source type has no checksum.
    pub checksum: Known<u64>,
}

impl ModelRef {
    /// `rev{revision}/{checksum:016x}`, or `rev{revision}/unknown`.
    pub fn text(&self) -> String {
        format!(
            "rev{}/{}",
            self.revision,
            show(&self.checksum, |c| format!("{c:016x}"))
        )
    }
}

/// Hypothesis the policy declares to have used.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisUsed {
    pub description: String,
    pub model: Known<ModelRef>,
    pub evidence_count: Known<u64>,
}

/// A simulated (not executed) future considered by a policy.
#[derive(Debug, Clone, PartialEq)]
pub struct HypotheticalFuture {
    pub label: String,
    pub first_action: usize,
    pub horizon: usize,
    pub viable_fraction: Known<f64>,
}

/// Everything recorded about one decision of one policy.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub policy: String,
    pub tick: u64,
    pub action_index: Known<usize>,
    pub scale_label: Known<String>,
    pub scale_version: Known<(u32, u32)>,
    pub predictive_error: Known<f64>,
    /// Macro goal emitted by the model: `(scale label, macro id)`.
    pub macro_goal: Known<(String, usize)>,
    pub hypothesis: Known<HypothesisUsed>,
    pub predicates: Vec<String>,
    pub hypothetical: Vec<HypotheticalFuture>,
}

impl DecisionRecord {
    /// A record with only policy and tick; every other field is `unknown`/empty.
    pub fn bare(policy: &str, tick: u64) -> Self {
        Self {
            policy: policy.to_owned(),
            tick,
            action_index: Known::Unknown,
            scale_label: Known::Unknown,
            scale_version: Known::Unknown,
            predictive_error: Known::Unknown,
            macro_goal: Known::Unknown,
            hypothesis: Known::Unknown,
            predicates: Vec::new(),
            hypothetical: Vec::new(),
        }
    }

    /// Adapts a GOG-175 nested decision: the coarse level supplies scale and macro goal. The
    /// hypothesis, predicates and futures are not part of that record, so they stay unknown/empty.
    pub fn from_nested(policy: &str, tick: u64, d: &NestedDecision) -> Self {
        let mut r = Self::bare(policy, tick);
        r.action_index = Known::Known(d.action_index);
        if let Some(c) = d.levels.iter().find(|l| l.level == Level::Coarse) {
            r.scale_label = Known::Known(c.scale_label.clone());
            r.scale_version = Known::Known(c.scale_version);
            r.predictive_error = Known::Known(c.uncertainty.predictive_error);
            r.macro_goal = c.goal.as_ref().map_or(Known::Unknown, |g| {
                Known::Known((g.scale_label.clone(), g.macro_id))
            });
        }
        r
    }

    /// Adapts a frozen tabular learned model (`learned_model`) and the report of one real step:
    /// the typed scale identity, the frozen revision/checksum and the evidence behind the
    /// `(bucket, action)` pair come from the source. The action index is a parameter because
    /// [`LearnedStepReport`] stores the buckets, not the action it was sampled with.
    pub fn from_learned_model(
        policy: &str,
        tick: u64,
        model: &TabularLearnedModel,
        step: &LearnedStepReport,
        action_index: usize,
    ) -> Self {
        let mut r = Self::bare(policy, tick);
        let version = model.version();
        let scale = version.scale().as_scale();
        r.action_index = Known::Known(action_index);
        r.scale_label = Known::Known(scale.id().to_owned());
        r.scale_version = Known::Known((scale.version().major(), scale.version().minor()));
        r.hypothesis = Known::Known(HypothesisUsed {
            description: format!(
                "tabular learned model: bucket {} -> {} uncertain={}",
                step.state_bucket, step.next_bucket, step.uncertain
            ),
            model: Known::Known(ModelRef {
                revision: version.revision(),
                checksum: Known::Known(version.checksum()),
            }),
            evidence_count: Known::Known(step.evidence),
        });
        r
    }

    /// Adapts the conditional-effects learner (`conditional_effects`) at one observed state:
    /// the gate result ([`GatedChoice`]), the frozen revision/checksum, the clause of the target
    /// prediction and the number of applied observations become the record's hypothesis, model,
    /// evidence and predicates. A closed gate offers no action (`action` stays unknown) and the
    /// window Brier is reported as the predictive error.
    pub fn from_conditional_effects(
        policy: &str,
        tick: u64,
        learner: &ConditionalEffectsLearner,
        state: &[bool],
        candidate: usize,
        target: usize,
    ) -> Self {
        let mut r = Self::bare(policy, tick);
        let version = learner.freeze();
        let model = Known::Known(ModelRef {
            revision: version.revision(),
            checksum: Known::Known(version.checksum()),
        });
        let evidence_count = Known::Known(learner.log().len() as u64);
        r.predictive_error = match learner.health() {
            PredictionHealth::Warmup => Known::Unknown,
            PredictionHealth::Healthy { mean_brier }
            | PredictionHealth::PredictionFailure { mean_brier } => Known::Known(mean_brier),
        };
        let (description, action, clause) = match gated_choice(learner, candidate) {
            GatedChoice::Act(action) => {
                let prediction = version.predict(state, action).ok();
                let bit = prediction.as_ref().and_then(|b| b.get(target));
                match bit {
                    Some(BitPrediction {
                        probability,
                        informed,
                        clause,
                        evidence,
                    }) => (
                        format!(
                            "conditional effects: action {action} next[b{target}] p={probability:.3} informed={informed} clause_evidence={evidence}"
                        ),
                        Known::Known(action),
                        clause.clone(),
                    ),
                    None => (
                        format!(
                            "conditional effects: action {action} (no prediction for next[b{target}])"
                        ),
                        Known::Known(action),
                        None,
                    ),
                }
            }
            GatedChoice::PredictionFailure(error) => (
                format!("conditional effects: prediction failure ({error}); no action offered"),
                Known::Unknown,
                None,
            ),
        };
        r.action_index = action;
        r.hypothesis = Known::Known(HypothesisUsed {
            description,
            model,
            evidence_count,
        });
        r.predicates = clause
            .unwrap_or_default()
            .iter()
            .map(literal_text)
            .collect();
        r
    }

    /// Adapts one rule of the counterexample-driven revisor (`causal_revision`): the current
    /// revision, the regions of `key = (action, target bit)` and the samples they cover become
    /// the record's hypothesis, model, evidence and predicates (one per region context).
    /// Rejected candidates are counted with their last reason. `ModelRevision` publishes no
    /// checksum, so the model reference reports `unknown` there; revision `0` is the empty model
    /// and yields no model reference at all.
    pub fn from_causal_revision(
        policy: &str,
        tick: u64,
        revisor: &CausalRevisor,
        key: (usize, usize),
    ) -> Self {
        let mut r = Self::bare(policy, tick);
        r.action_index = Known::Known(key.0);
        let current = revisor.current();
        if revisor.revision() == 0 && current.rules.is_empty() {
            return r;
        }
        let rejected: Vec<_> = revisor.rejected().iter().filter(|c| c.key == key).collect();
        let (evidence, description) = match current.rules.get(&key) {
            Some(rule) => (
                rule.regions.iter().map(|g| g.n).sum::<u64>(),
                format!(
                    "causal rule next[b{}]|a{}: {} regions status={:?}{}",
                    key.1,
                    key.0,
                    rule.regions.len(),
                    rule.status,
                    rejected_note(&rejected)
                ),
            ),
            None => (
                0,
                format!(
                    "causal rule next[b{}]|a{}: absent in revision {}{}",
                    key.1,
                    key.0,
                    current.revision,
                    rejected_note(&rejected)
                ),
            ),
        };
        r.hypothesis = Known::Known(HypothesisUsed {
            description,
            model: Known::Known(ModelRef {
                revision: current.revision,
                checksum: Known::Unknown,
            }),
            evidence_count: Known::Known(evidence),
        });
        if let Some(rule) = current.rules.get(&key) {
            r.predicates = rule
                .regions
                .iter()
                .map(|g| context_text(&g.context))
                .collect();
        }
        r
    }

    /// Adapts one invention run (`predicate_invention`): the adopted predicates become the
    /// record's predicates, the report's holdout Brier becomes the predictive error and the
    /// training samples used become the evidence count. The model reference exists only when
    /// the invention was adopted; its revision comes from the underlying revisor (no checksum
    /// is published, so it is `unknown`).
    pub fn from_predicate_invention(
        policy: &str,
        tick: u64,
        outcome: &InventionOutcome,
        key: (usize, usize),
    ) -> Self {
        let mut r = Self::bare(policy, tick);
        let (action, target) = key;
        let report = &outcome.report;
        let accepted = report
            .candidates
            .iter()
            .filter(|c| matches!(c.verdict, CandidateVerdict::Accepted))
            .count();
        r.action_index = Known::Known(action);
        r.predictive_error = Known::Known(report.invented_brier);
        let (model, predicates) = match &outcome.model {
            Some(m) => (
                Known::Known(ModelRef {
                    revision: m.revisor().current().revision,
                    checksum: Known::Unknown,
                }),
                m.predicates.iter().map(Predicate::describe).collect(),
            ),
            None => (Known::Unknown, Vec::new()),
        };
        r.hypothesis = Known::Known(HypothesisUsed {
            description: format!(
                "predicate invention next[b{target}]|a{action}: adopted={} accepted={accepted}/{} baseline_brier={:.4} invented_brier={:.4} evals={} train_used={}",
                report.adopted,
                report.candidates.len(),
                report.baseline_brier,
                report.invented_brier,
                report.evals,
                report.train_used
            ),
            model,
            evidence_count: Known::Known(report.train_used as u64),
        });
        r.predicates = predicates;
        r
    }
}

fn literal_text(literal: &Literal) -> String {
    match *literal {
        Literal::Bit { index, value } => format!("b{index}={value}"),
        Literal::Action(action) => format!("a{action}"),
    }
}

fn context_text(context: &[(u16, bool)]) -> String {
    if context.is_empty() {
        return "true".to_owned();
    }
    context
        .iter()
        .map(|&(index, value)| format!("b{index}={value}"))
        .collect::<Vec<_>>()
        .join("&")
}

fn rejected_note(rejected: &[&gogamen_core::causal_revision::RejectedCandidate]) -> String {
    match rejected.last() {
        Some(c) => format!(
            " rejected={} last={:?}@{}{}",
            rejected.len(),
            c.reason,
            c.at_revision,
            c.candidate_brier
                .map_or_else(String::new, |b| format!(" candidate_brier={b:.4}"))
        ),
        None => " rejected=0".to_owned(),
    }
}

fn show<T>(k: &Known<T>, f: impl Fn(&T) -> String) -> String {
    match k {
        Known::Known(v) => f(v),
        Known::Unknown => "unknown".to_owned(),
    }
}

/// One step of the real world, taken from a recording.
#[derive(Debug, Clone, PartialEq)]
pub struct RealStep {
    pub tick: u64,
    pub agent: String,
    pub total_reward: f64,
}

/// Real-world trace up to `upto_tick` (inclusive). Read-only over the recording.
pub fn real_trace(rec: &Recording, upto_tick: u64) -> Vec<RealStep> {
    rec.frames()
        .iter()
        .filter(|f| f.tick <= upto_tick)
        .map(|f| RealStep {
            tick: f.tick,
            agent: format!("{:?}", f.agent),
            total_reward: f.total_reward,
        })
        .collect()
}

/// View of one decision with separate real and hypothetical lanes.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionView {
    pub record: DecisionRecord,
    /// Executed world, from the recording.
    pub real_lane: Vec<RealStep>,
    /// Simulated futures from the policy; never contains real steps.
    pub hypothetical_lane: Vec<HypotheticalFuture>,
}

/// Builds the view of `record` against the real trace of `rec` up to the decision tick.
pub fn decision_view(rec: &Recording, record: &DecisionRecord) -> DecisionView {
    DecisionView {
        real_lane: real_trace(rec, record.tick),
        hypothetical_lane: record.hypothetical.clone(),
        record: record.clone(),
    }
}

impl DecisionView {
    /// Text lines for the inspector panel.
    pub fn lines(&self) -> Vec<String> {
        let r = &self.record;
        let mut out = vec![
            format!("policy={} tick={}", r.policy, r.tick),
            format!("action={}", show(&r.action_index, |a| a.to_string())),
            format!(
                "scale={} version={} predictive_error={}",
                show(&r.scale_label, String::clone),
                show(&r.scale_version, |v| format!("{}.{}", v.0, v.1)),
                show(&r.predictive_error, |e| format!("{e:.4}"))
            ),
            format!(
                "macro_goal={}",
                show(&r.macro_goal, |g| format!("{}#{}", g.0, g.1))
            ),
        ];
        match &r.hypothesis {
            Known::Known(h) => out.push(format!(
                "hypothesis={} model={} evidence={}",
                h.description,
                show(&h.model, ModelRef::text),
                show(&h.evidence_count, |n| n.to_string())
            )),
            Known::Unknown => out.push("hypothesis=unknown model=unknown evidence=unknown".into()),
        }
        out.push(format!(
            "predicates={}",
            if r.predicates.is_empty() {
                "unknown".to_owned()
            } else {
                r.predicates.join("; ")
            }
        ));
        for s in &self.real_lane {
            out.push(format!(
                "[real] t={} agent={} reward={:.2}",
                s.tick, s.agent, s.total_reward
            ));
        }
        for h in &self.hypothetical_lane {
            out.push(format!(
                "[hypothetical] {} first_action={} horizon={} viable={}",
                h.label,
                h.first_action,
                h.horizon,
                show(&h.viable_fraction, |v| format!("{v:.2}"))
            ));
        }
        out
    }
}

/// One compared field of two policies' decisions.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonRow {
    pub field: &'static str,
    pub a: String,
    pub b: String,
    pub differs: bool,
}

/// Side-by-side comparison of two decisions, restricted to recorded model fields.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyComparison {
    pub policy_a: String,
    pub policy_b: String,
    pub rows: Vec<ComparisonRow>,
}

/// Compares two decisions field by field. `unknown` on both sides counts as "same unknown", not
/// as agreement about a value.
pub fn compare_policies(a: &DecisionRecord, b: &DecisionRecord) -> PolicyComparison {
    let va = decision_view_fields(a);
    let vb = decision_view_fields(b);
    let rows = va
        .into_iter()
        .zip(vb)
        .map(|((field, x), (_, y))| ComparisonRow {
            field,
            differs: x != y,
            a: x,
            b: y,
        })
        .collect();
    PolicyComparison {
        policy_a: a.policy.clone(),
        policy_b: b.policy.clone(),
        rows,
    }
}

fn decision_view_fields(r: &DecisionRecord) -> Vec<(&'static str, String)> {
    let hyp = show(&r.hypothesis, |h| {
        format!(
            "{} model={} evidence={}",
            h.description,
            show(&h.model, ModelRef::text),
            show(&h.evidence_count, |n| n.to_string())
        )
    });
    vec![
        ("action", show(&r.action_index, |a| a.to_string())),
        ("scale", show(&r.scale_label, String::clone)),
        (
            "scale_version",
            show(&r.scale_version, |v| format!("{}.{}", v.0, v.1)),
        ),
        (
            "predictive_error",
            show(&r.predictive_error, |e| format!("{e:.4}")),
        ),
        (
            "macro_goal",
            show(&r.macro_goal, |g| format!("{}#{}", g.0, g.1)),
        ),
        ("hypothesis", hyp),
        (
            "predicates",
            if r.predicates.is_empty() {
                "unknown".to_owned()
            } else {
                r.predicates.join("; ")
            },
        ),
        ("hypothetical_futures", r.hypothetical.len().to_string()),
    ]
}

impl PolicyComparison {
    /// Text lines; the header states that only recorded fields are compared.
    pub fn lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "compare {} vs {} (recorded model fields only; no intentions inferred)",
            self.policy_a, self.policy_b
        )];
        for r in &self.rows {
            out.push(format!(
                "{}: {} | {}{}",
                r.field,
                r.a,
                r.b,
                if r.differs { " (differs)" } else { "" }
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::counterfactual::RecordingId;
    use crate::recorder::record_grid;
    use gogamen_core::causal_revision::{CausalRevisor, RevisionConfig, RevisionOutcome};
    use gogamen_core::conditional_effects::ConditionalObservation;
    use gogamen_core::learned_model::{
        LearnedModelConfig, ModelVersion, TabularLearnedModel, TabularScale, UnseenPairPolicy,
    };
    use gogamen_core::predicate_invention::{InventionConfig, PredicateInventor};
    use gogamen_core::{
        Action, BinSpec, BinningVersion, DeterministicRng, DiscreteAction, EvidenceRef,
        LearningPolicy, ObservationId, SimDuration, State, StepDt, TabularTransitionLearner,
        Terminal, Tick, TransitionObservation,
    };
    use gogamen_lab_grid::GridMap;
    use gogamen_lab_grid::conditional_effects::{
        Dynamics, TRAIN_SEED, TRAIN_STEPS, generate_stream, new_lab_learner,
    };
    use gogamen_lab_grid::nested_scales::{
        BlockWorldKind, LevelParams, NestedAgent, NestedConfig, OneLevelAgent, block_world,
    };

    fn nested_record() -> DecisionRecord {
        let bw = block_world(11, BlockWorldKind::Lumpable).unwrap();
        let start = bw.starts[0];
        let agent = NestedAgent::new(
            &bw.world,
            &[bw.blocks.clone(), bw.identity.clone()],
            start,
            NestedConfig::preregistered(7),
        )
        .unwrap();
        let d = agent.decide(start, 2).unwrap();
        DecisionRecord::from_nested("nested", 2, &d)
    }

    #[test]
    fn emergence_view_shows_known_truthfully_and_unknown_when_missing() {
        let rec = record_grid(42, GridMap::cul_de_sac_map(), 8).unwrap();
        let r = nested_record();
        // Scale/version/goal come from the real GOG-175 record, not invented.
        assert!(matches!(&r.scale_label, Known::Known(l) if l == "blocks"));
        assert!(matches!(r.scale_version, Known::Known(_)));
        let lines = decision_view(&rec, &r).lines();
        assert!(lines[2].starts_with("scale=blocks version="), "{lines:?}");
        // The nested record carries no hypothesis/predicates: unknown, not guessed.
        assert!(lines.contains(&"hypothesis=unknown model=unknown evidence=unknown".to_owned()));
        assert!(lines.contains(&"predicates=unknown".to_owned()));

        // Fully bare record: everything unknown.
        let bare = decision_view(&rec, &DecisionRecord::bare("p", 1)).lines();
        assert_eq!(bare[1], "action=unknown");
        assert!(bare[2].contains("scale=unknown version=unknown predictive_error=unknown"));
        assert_eq!(bare[3], "macro_goal=unknown");

        // Present hypothesis shows exactly the given version/evidence; missing evidence is unknown.
        let mut r2 = DecisionRecord::bare("p", 1);
        r2.hypothesis = Known::Known(HypothesisUsed {
            description: "H1".into(),
            model: Known::Known(ModelRef {
                revision: 3,
                checksum: Known::Known(0xAB),
            }),
            evidence_count: Known::Unknown,
        });
        r2.predicates = vec!["x0&&x1".into()];
        let l2 = decision_view(&rec, &r2).lines();
        assert!(
            l2.contains(&"hypothesis=H1 model=rev3/00000000000000ab evidence=unknown".to_owned())
        );
        assert!(l2.contains(&"predicates=x0&&x1".to_owned()));
    }

    #[test]
    fn emergence_view_keeps_real_and_hypothetical_separate_and_read_only() {
        let rec = record_grid(42, GridMap::cul_de_sac_map(), 8).unwrap();
        let (bytes, id) = (rec.to_bytes(), RecordingId::of(&rec));
        let mut r = DecisionRecord::bare("p", 3);
        r.hypothetical = vec![HypotheticalFuture {
            label: "if-a1".into(),
            first_action: 1,
            horizon: 5,
            viable_fraction: Known::Unknown,
        }];
        let v1 = decision_view(&rec, &r);
        let v2 = decision_view(&rec, &r);
        assert_eq!(v1, v2, "replay is repeatable");
        assert!(v1.real_lane.iter().all(|s| s.tick <= 3) && !v1.real_lane.is_empty());
        assert_eq!(v1.hypothetical_lane.len(), 1);
        let lines = v1.lines();
        let real = lines.iter().filter(|l| l.starts_with("[real]")).count();
        let hyp = lines
            .iter()
            .filter(|l| l.starts_with("[hypothetical]"))
            .count();
        assert_eq!((real, hyp), (v1.real_lane.len(), 1));
        assert!(lines.iter().any(|l| l.ends_with("viable=unknown")));
        // The simulation record is bit-identical after viewing.
        assert_eq!(rec.to_bytes(), bytes);
        assert_eq!(RecordingId::of(&rec), id);
    }

    #[test]
    fn emergence_view_compares_two_policies_without_intentions() {
        let bw = block_world(11, BlockWorldKind::Lumpable).unwrap();
        let start = bw.starts[0];
        let cfg = NestedConfig::preregistered(7);
        let nested = NestedAgent::new(
            &bw.world,
            &[bw.blocks.clone(), bw.identity.clone()],
            start,
            cfg.clone(),
        )
        .unwrap();
        let a = DecisionRecord::from_nested("nested", 2, &nested.decide(start, 2).unwrap());
        let one = OneLevelAgent::new(
            &bw.world,
            &bw.identity,
            LevelParams {
                population: cfg.total_budget() / cfg.fine.horizon,
                horizon: cfg.fine.horizon,
            },
            7,
        )
        .unwrap();
        let (act, _) = one.decide(start, 2).unwrap();
        let mut b = DecisionRecord::bare("one-level", 2);
        b.action_index = Known::Known(act);

        let cmp = compare_policies(&a, &b);
        let row = |f: &str| cmp.rows.iter().find(|r| r.field == f).unwrap();
        assert!(row("scale").differs && row("scale").b == "unknown");
        assert_eq!(row("hypothesis").a, "unknown");
        assert!(
            !row("hypothesis").differs,
            "both unknown is not a recorded difference"
        );
        let lines = cmp.lines();
        assert!(lines[0].contains("no intentions inferred"));
        // Only recorded model fields: no intention/desire/belief rows exist.
        for r in &cmp.rows {
            for banned in ["intention", "desire", "motive", "belief", "wants"] {
                assert!(!r.field.contains(banned));
            }
        }
        assert_eq!(
            compare_policies(&a, &a)
                .rows
                .iter()
                .filter(|r| r.differs)
                .count(),
            0
        );
    }

    // ---------------------------------------------------------------------------------------
    // GOG-205: adapters from the real hypothesis/model sources.
    // ---------------------------------------------------------------------------------------

    /// Real tabular learned model: one bucket axis, two actions, five observations per
    /// `(bucket, action)` pair recorded through the typed `TabularScale`.
    fn learned_model() -> TabularLearnedModel {
        let spec = BinSpec::uniform(
            "gog205-scale",
            BinningVersion::new(2, 1),
            vec![(0.0, 4.0, 4)],
        )
        .unwrap();
        let scale = TabularScale::new(spec).unwrap();
        let mut learner =
            TabularTransitionLearner::new(scale.num_states(), 2, LearningPolicy::laplace())
                .unwrap();
        let mut id = 0u64;
        for bucket in 0..4usize {
            let x = bucket as f64 + 0.5;
            for _ in 0..5 {
                for action in 0..2usize {
                    id += 1;
                    let sample = scale
                        .sample_from_transition(&[x], action, &[x], Terminal::Ongoing)
                        .unwrap();
                    learner
                        .observe(&TransitionObservation::new(
                            ObservationId::new(id),
                            EvidenceRef::new(2000 + id),
                            sample.state_bucket(),
                            sample.action_idx(),
                            sample.next_state_bucket(),
                        ))
                        .unwrap();
                }
            }
        }
        let version = ModelVersion::freeze(&learner, scale).unwrap();
        TabularLearnedModel::new(
            version,
            LearnedModelConfig::new(0.5, 1, UnseenPairPolicy::SamplePrior).unwrap(),
        )
        .unwrap()
    }

    /// Real counterexample fixtures: bit1 is the "blocked" context, bit0 is noise and bit2 marks
    /// a single isolated outlier (same construction as the `causal_revision` crate tests).
    fn causal_obs(id: u64, state: [bool; 3], moved: bool) -> ConditionalObservation {
        ConditionalObservation {
            id: ObservationId::new(id),
            evidence: EvidenceRef::new(9000 + id),
            state: state.to_vec(),
            action: 0,
            next: vec![moved, false, false],
        }
    }

    fn causal_fixture(start: u64, n: u64) -> Vec<ConditionalObservation> {
        (0..n)
            .map(|i| {
                let blocked = i % 2 == 1;
                let b0 = (i / 2) % 2 == 1;
                causal_obs(start + i, [b0, blocked, false], !blocked)
            })
            .collect()
    }

    /// Deterministic LCG bit source for the invention fixtures.
    fn lcg(s: &mut u64) -> u64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    }

    /// Four uniform bits where `rule` decides `next[0]` (bits 2 and 3 are distractors).
    fn invention_data(
        seed: u64,
        start: u64,
        n: u64,
        mut rule: impl FnMut(&[bool]) -> bool,
    ) -> Vec<ConditionalObservation> {
        let mut s = seed;
        (0..n)
            .map(|i| {
                let r = lcg(&mut s);
                let state: Vec<bool> = (0..4).map(|k| (r >> k) & 1 == 1).collect();
                let y = rule(&state);
                ConditionalObservation {
                    id: ObservationId::new(start + i),
                    evidence: EvidenceRef::new(7000 + start + i),
                    state,
                    action: 0,
                    next: vec![y, false, false, false],
                }
            })
            .collect()
    }

    /// A real recording used only to prove the lines render inside the real lane.
    fn recording() -> Recording {
        record_grid(42, GridMap::cul_de_sac_map(), 8).unwrap()
    }

    #[test]
    fn emergence_view_learned_model_adapter_fills_model_scale_and_evidence() {
        let model = learned_model();
        let state = State::new(Tick::ZERO, SimDuration::ZERO, vec![1.5]).unwrap();
        let dt = StepDt::from_millis_f64(100.0).unwrap();
        let action = Action::Discrete(DiscreteAction::new(0, 2).unwrap());
        let (_, step) = model
            .step_with_report(&state, &action, dt, &mut DeterministicRng::new(7))
            .unwrap();

        let r = DecisionRecord::from_learned_model("learned", 4, &model, &step, 0);
        let version = model.version();
        let scale_id = version.scale().as_scale().id();
        assert_eq!(r.action_index, Known::Known(0));
        assert_eq!(r.scale_label, Known::Known(scale_id.to_owned()));
        assert_eq!(r.scale_version, Known::Known((2, 1)));
        let Known::Known(h) = &r.hypothesis else {
            panic!("learned model must fill the hypothesis");
        };
        assert_eq!(
            h.model,
            Known::Known(ModelRef {
                revision: version.revision(),
                checksum: Known::Known(version.checksum()),
            })
        );
        assert_eq!(h.evidence_count, Known::Known(step.evidence));
        assert!(
            matches!(h.evidence_count, Known::Known(n) if n > 0),
            "evidence must come from the trained pairs: {h:?}"
        );
        assert!(r.predicates.is_empty(), "a tabular model has no predicates");

        let lines = decision_view(&recording(), &r).lines();
        assert_eq!(
            lines[2],
            format!("scale={scale_id} version=2.1 predictive_error=unknown")
        );
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with("hypothesis=tabular learned model:")
                    && l.contains(&format!(
                        "model=rev{}/{:016x}",
                        version.revision(),
                        version.checksum()
                    ))
                    && l.contains(&format!("evidence={}", step.evidence))),
            "{lines:?}"
        );
    }

    #[test]
    fn emergence_view_conditional_effects_adapter_maps_gate_clause_and_evidence() {
        let mut learner = new_lab_learner().unwrap();
        let train = generate_stream(
            &GridMap::cul_de_sac_map(),
            Dynamics::Normal,
            TRAIN_SEED,
            TRAIN_STEPS,
            0,
        );
        for o in &train {
            learner.observe(o).unwrap();
        }
        assert_eq!(learner.log().len(), TRAIN_STEPS);

        let state = train[0].state.clone();
        let r = DecisionRecord::from_conditional_effects("conditional", 1, &learner, &state, 0, 0);
        let version = learner.freeze();
        let Known::Known(h) = &r.hypothesis else {
            panic!("conditional effects must fill the hypothesis");
        };
        assert_eq!(
            h.model,
            Known::Known(ModelRef {
                revision: version.revision(),
                checksum: Known::Known(version.checksum()),
            })
        );
        assert_eq!(h.evidence_count, Known::Known(TRAIN_STEPS as u64));
        // The record mirrors the learner's own gate: a closed gate offers no action.
        eprintln!(
            "GOG-205 conditional fixture: gate={:?} health={:?} version={} evidence={:?}",
            learner.gate(),
            learner.health(),
            version.revision(),
            h.evidence_count
        );
        match learner.gate() {
            Ok(()) => {
                assert_eq!(r.action_index, Known::Known(0));
                assert!(
                    h.description
                        .starts_with("conditional effects: action 0 next[b0] p="),
                    "{}",
                    h.description
                );
                let clause = version.predict(&state, 0).unwrap()[0].clause.clone();
                let expected: Vec<String> = clause
                    .unwrap_or_default()
                    .iter()
                    .map(literal_text)
                    .collect();
                assert_eq!(r.predicates, expected, "predicates are the clause literals");
                assert!(
                    !r.predicates.is_empty(),
                    "the informed target bit must expose its clause"
                );
            }
            Err(_) => {
                assert_eq!(r.action_index, Known::Unknown);
                assert!(h.description.contains("prediction failure"));
                assert!(r.predicates.is_empty());
            }
        }
        match learner.health() {
            PredictionHealth::Healthy { mean_brier }
            | PredictionHealth::PredictionFailure { mean_brier } => {
                assert_eq!(r.predictive_error, Known::Known(mean_brier));
            }
            PredictionHealth::Warmup => assert_eq!(r.predictive_error, Known::Unknown),
        }

        let lines = decision_view(&recording(), &r).lines();
        assert!(
            lines[4].starts_with("hypothesis=conditional effects:"),
            "{lines:?}"
        );
        assert!(lines.iter().any(|l| l.contains(&format!(
            "model=rev{}/{:016x}",
            version.revision(),
            version.checksum()
        ))));
    }

    #[test]
    fn emergence_view_causal_revision_adapter_maps_regions_and_evidence() {
        let mut revisor = CausalRevisor::new(3, 1, RevisionConfig::default()).unwrap();
        // Revision 0 is the empty model: no model reference and no hypothesis are claimed.
        let empty = DecisionRecord::from_causal_revision("causal", 2, &revisor, (0, 0));
        assert_eq!(empty.hypothesis, Known::Unknown);
        assert!(empty.predicates.is_empty());

        let train = causal_fixture(1, 60);
        let hold = causal_fixture(1000, 40);
        assert_eq!(
            revisor.revise(0, 0, &train, &hold).unwrap(),
            RevisionOutcome::Promoted { revision: 1 }
        );
        let rule = revisor.current().rules[&(0, 0)].clone();
        let r = DecisionRecord::from_causal_revision("causal", 2, &revisor, (0, 0));
        assert_eq!(r.action_index, Known::Known(0));
        let Known::Known(h) = &r.hypothesis else {
            panic!("causal revision must fill the hypothesis");
        };
        assert_eq!(
            h.model,
            Known::Known(ModelRef {
                revision: 1,
                checksum: Known::Unknown,
            }),
            "ModelRevision publishes no checksum"
        );
        assert_eq!(
            h.evidence_count,
            Known::Known(rule.regions.iter().map(|g| g.n).sum::<u64>())
        );
        assert!(matches!(h.evidence_count, Known::Known(n) if n == 60));
        let mut predicates = r.predicates.clone();
        predicates.sort();
        assert_eq!(
            predicates,
            vec!["b1=false".to_owned(), "b1=true".to_owned()]
        );
        assert!(
            h.description.contains("2 regions status=Active rejected=0"),
            "{}",
            h.description
        );

        // A rejected candidate is reported with its reason, never silently dropped.
        let small = causal_fixture(3000, 10);
        assert_eq!(
            revisor.revise(0, 0, &train, &small).unwrap(),
            RevisionOutcome::Rejected {
                reason: gogamen_core::causal_revision::RejectReason::InsufficientHoldout
            }
        );
        let r2 = DecisionRecord::from_causal_revision("causal", 3, &revisor, (0, 0));
        let Known::Known(h2) = &r2.hypothesis else {
            panic!();
        };
        assert!(
            h2.description
                .contains("rejected=1 last=InsufficientHoldout@1"),
            "{}",
            h2.description
        );

        let lines = decision_view(&recording(), &r).lines();
        assert!(
            lines.iter().any(|l| l.contains("model=rev1/unknown")),
            "{lines:?}"
        );
        assert!(lines.iter().any(|l| l.starts_with("predicates=b1=")));
        // The comparison reads the adapted fields, not a second data layer.
        let cmp = compare_policies(&r, &r2);
        let hyp = cmp.rows.iter().find(|r| r.field == "hypothesis").unwrap();
        assert!(
            hyp.differs,
            "the rejected candidate changed the description"
        );
        assert!(cmp.lines()[0].contains("no intentions inferred"));
    }

    #[test]
    fn emergence_view_predicate_invention_adapter_maps_predicates_and_evidence() {
        let xor = |s: &[bool]| s[0] ^ s[1];
        let train = invention_data(1, 1, 240, xor);
        let hold = invention_data(2, 10_000, 120, xor);
        let outcome =
            PredicateInventor::new(4, 1, InventionConfig::default(), RevisionConfig::default())
                .unwrap()
                .invent(0, 0, &train, &hold)
                .unwrap();
        assert!(outcome.report.adopted);
        let adopted_revision = outcome.model.as_ref().unwrap().revisor().current().revision;
        assert!(adopted_revision > 0);

        let r = DecisionRecord::from_predicate_invention("invention", 5, &outcome, (0, 0));
        assert_eq!(r.action_index, Known::Known(0));
        assert_eq!(r.predicates, vec!["Xor(b0,b1)".to_owned()]);
        assert_eq!(
            r.predictive_error,
            Known::Known(outcome.report.invented_brier)
        );
        let Known::Known(h) = &r.hypothesis else {
            panic!("invention must fill the hypothesis");
        };
        assert_eq!(
            h.model,
            Known::Known(ModelRef {
                revision: adopted_revision,
                checksum: Known::Unknown,
            })
        );
        assert_eq!(h.evidence_count, Known::Known(240));
        assert!(
            h.description.contains("adopted=true accepted=1/")
                && h.description.contains("train_used=240"),
            "{}",
            h.description
        );

        let lines = decision_view(&recording(), &r).lines();
        assert!(
            lines.contains(&"predicates=Xor(b0,b1)".to_owned()),
            "{lines:?}"
        );
        assert!(
            lines
                .iter()
                .any(|l| l.contains(&format!("model=rev{adopted_revision}/unknown")))
        );

        // A run that is not adopted claims no model and no predicates.
        let mut s1 = 99u64;
        let mut s2 = 77u64;
        let train = invention_data(1, 1, 240, |_| lcg(&mut s1) & 1 == 1);
        let hold = invention_data(2, 10_000, 120, |_| lcg(&mut s2) & 1 == 1);
        let rejected =
            PredicateInventor::new(4, 1, InventionConfig::default(), RevisionConfig::default())
                .unwrap()
                .invent(0, 0, &train, &hold)
                .unwrap();
        assert!(!rejected.report.adopted && rejected.model.is_none());
        let r2 = DecisionRecord::from_predicate_invention("invention", 5, &rejected, (0, 0));
        let Known::Known(h2) = &r2.hypothesis else {
            panic!();
        };
        assert_eq!(h2.model, Known::Unknown);
        assert!(r2.predicates.is_empty());
        assert_eq!(
            h2.evidence_count,
            Known::Known(rejected.report.train_used as u64)
        );

        // Comparing an adopted and a non-adopted invention shows the adoption difference.
        let cmp = compare_policies(&r, &r2);
        let model = cmp.rows.iter().find(|r| r.field == "hypothesis").unwrap();
        assert!(model.differs);
        assert!(cmp.lines().iter().any(|l| l.contains("(differs)")));
    }
}
