//! Cost/benefit harness: mock generative System Two vs the local keyword baseline, over the
//! synthetic annotated utterances of `gogamen-lab-mujoco::dialog`.
//!
//! **Honesty statement**: the generative side is a deterministic mock whose error pattern is
//! scripted by this file. Numbers produced here describe the harness and the validation
//! pipeline on synthetic fixtures. They are NOT evidence about any real model, real latency, or
//! real cost. The only genuinely measured quantity is the local CPU time of the baseline and of
//! the mock+validation path.

use std::time::{Duration, Instant};

use gogamen_core::{
    ActionGuard, ActionSpace, CapabilityCatalog, EvidenceRef, NoulVerdict, Terminal, Tick,
};
use gogamen_lab_mujoco::dialog::annotated_fixtures;

use crate::candidate::{
    CandidateBody, CandidateRejection, CandidateValidator, Claim, GenerativeCandidate,
    ValidatedCandidate, ValidationContext,
};
use crate::capability::{
    ExperimentalCapability, ExperimentalConfig, ExperimentalHub, Outcome, ProposalRequest,
};
use crate::mock::{MOCK_GENERATIVE_SENSOR, MockGenerative};

/// Goal proposed when an utterance means "left door".
pub const GOAL_LEFT: &str = "target-left";
/// Goal proposed when an utterance means "right door".
pub const GOAL_RIGHT: &str = "target-right";

/// Declared (simulated) latency of one mock generative call.
pub const DECLARED_GENERATIVE_LATENCY: Duration = Duration::from_millis(800);
/// Declared abstract cost units of one mock generative call.
pub const DECLARED_GENERATIVE_UNITS: u64 = 350;

/// Local keyword baseline: `Some(true)` = left, `Some(false)` = right, `None` = no usable claim.
pub fn local_detector(text: &str) -> Option<bool> {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    let left = words.contains(&"left");
    let right = words.contains(&"right");
    let negated = words.contains(&"not");
    match (left, right) {
        (true, false) => Some(!negated),
        (false, true) => Some(negated),
        _ => None,
    }
}

fn expected(v: NoulVerdict) -> Option<bool> {
    match v {
        NoulVerdict::True => Some(true),
        NoulVerdict::False => Some(false),
        _ => None,
    }
}

/// Metrics for one approach.
#[derive(Debug, Clone, PartialEq)]
pub struct ApproachReport {
    /// Label.
    pub name: &'static str,
    /// Utterances evaluated.
    pub items: usize,
    /// Utterances for which a candidate/decision was produced.
    pub produced: usize,
    /// Produced candidates that passed validation (baseline: all produced).
    pub valid: usize,
    /// Decisions equal to the recorded annotation (`None` expected counts as correct abstention).
    pub correct: usize,
    /// Rejections by typed reason name.
    pub rejected_ungrounded: usize,
    /// Rejections of unknown goals.
    pub rejected_unknown_goal: usize,
    /// Total measured wall time of the approach on this machine.
    pub measured_total: Duration,
    /// Total declared (simulated) latency, zero for the baseline.
    pub declared_latency_total: Duration,
    /// Total declared abstract cost units, zero for the baseline.
    pub declared_units_total: u64,
}

impl ApproachReport {
    /// Valid / produced (1.0 when nothing was produced).
    pub fn valid_rate(&self) -> f64 {
        if self.produced == 0 {
            1.0
        } else {
            self.valid as f64 / self.produced as f64
        }
    }

    /// Correct / items.
    pub fn accuracy(&self) -> f64 {
        self.correct as f64 / self.items.max(1) as f64
    }
}

/// Result of the comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonReport {
    /// Local baseline.
    pub baseline: ApproachReport,
    /// Mock generative + validation.
    pub generative: ApproachReport,
    /// With the capability disabled (default), the run produced zero candidates and no errors.
    pub disabled_abstentions: usize,
}

/// Runs the deterministic comparison.
pub fn run_comparison() -> ComparisonReport {
    let utterances: Vec<(String, Option<bool>)> = annotated_fixtures()
        .into_iter()
        .flat_map(|f| {
            f.claims
                .into_iter()
                .map(|c| (c.text.to_string(), expected(c.verdict)))
        })
        .collect();

    // Baseline.
    let t0 = Instant::now();
    let decisions: Vec<Option<bool>> = utterances.iter().map(|(t, _)| local_detector(t)).collect();
    let measured = t0.elapsed();
    let correct = decisions
        .iter()
        .zip(&utterances)
        .filter(|(d, (_, e))| *d == e)
        .count();
    let produced = decisions.iter().flatten().count();
    let baseline = ApproachReport {
        name: "local-keyword-detector",
        items: utterances.len(),
        produced,
        valid: produced,
        correct,
        rejected_ungrounded: 0,
        rejected_unknown_goal: 0,
        measured_total: measured,
        declared_latency_total: Duration::ZERO,
        declared_units_total: 0,
    };

    // Mock generative: scripted pattern of faults by index.
    let goal = |name: &str, ev: u64| GenerativeCandidate {
        id: format!("cand-{name}-{ev}"),
        body: CandidateBody::Goal { name: name.into() },
        claims: vec![Claim {
            statement: format!("utterance #{ev} implies {name}"),
            evidence: vec![EvidenceRef::new(ev)],
        }],
    };
    let mut mock = MockGenerative::new(DECLARED_GENERATIVE_LATENCY, DECLARED_GENERATIVE_UNITS);
    for (i, (_, exp)) in utterances.iter().enumerate() {
        let ev = i as u64;
        let list = match exp {
            None => vec![],
            Some(left) => {
                let truthful = if *left { GOAL_LEFT } else { GOAL_RIGHT };
                let flipped = if *left { GOAL_RIGHT } else { GOAL_LEFT };
                let mut c = match i % 6 {
                    // Valid but wrong interpretation.
                    3 => goal(flipped, ev),
                    // Hallucinated goal, out of the goal catalog.
                    5 => goal("target-center", ev),
                    _ => goal(truthful, ev),
                };
                if i % 6 == 4 {
                    // Hallucinated evidence: cites an observation that does not exist.
                    c.claims[0].evidence = vec![EvidenceRef::new(9_999 + ev)];
                }
                vec![c]
            }
        };
        mock = mock.script(format!("u{i}"), list);
    }

    let catalog = CapabilityCatalog::new();
    let guard = ActionGuard::strict();
    let space = ActionSpace::discrete(2).expect("static discrete space");
    let validator = CandidateValidator::new(&catalog, &guard)
        .with_generative_sensor(MOCK_GENERATIVE_SENSOR)
        .with_evidence((0..utterances.len() as u64).map(EvidenceRef::new))
        .with_goals([GOAL_LEFT, GOAL_RIGHT]);
    let ctx = ValidationContext {
        action_space: &space,
        tick: Tick::ZERO,
        terminal: Terminal::Ongoing,
    };

    // Disabled-by-default run: must abstain for every request.
    let disabled_hub =
        ExperimentalHub::new(ExperimentalConfig::default()).with_generative(Box::new(mock.clone()));
    let disabled_abstentions = (0..utterances.len())
        .filter(|i| {
            disabled_hub
                .propose(&ProposalRequest {
                    key: format!("u{i}"),
                    prompt: String::new(),
                })
                .is_abstain()
        })
        .count();

    let hub = ExperimentalHub::new(
        ExperimentalConfig::default().enable(ExperimentalCapability::GenerativeSystemTwo),
    )
    .with_generative(Box::new(mock));

    let mut rep = ApproachReport {
        name: "mock-generative-system-two",
        items: utterances.len(),
        produced: 0,
        valid: 0,
        correct: 0,
        rejected_ungrounded: 0,
        rejected_unknown_goal: 0,
        measured_total: Duration::ZERO,
        declared_latency_total: Duration::ZERO,
        declared_units_total: 0,
    };
    let t1 = Instant::now();
    for (i, (_, exp)) in utterances.iter().enumerate() {
        let out = hub.propose(&ProposalRequest {
            key: format!("u{i}"),
            prompt: String::new(),
        });
        let Outcome::Value(gen_out) = out else {
            continue;
        };
        rep.declared_latency_total += gen_out.cost.latency;
        rep.declared_units_total += gen_out.cost.units;
        let mut decision = None;
        for env in &gen_out.candidates {
            rep.produced += 1;
            match validator.validate(env, &ctx) {
                Ok(ValidatedCandidate::Goal(name)) => {
                    rep.valid += 1;
                    decision = Some(name == GOAL_LEFT);
                }
                Ok(_) => rep.valid += 1,
                Err(CandidateRejection::UngroundedClaim { .. }) => rep.rejected_ungrounded += 1,
                Err(CandidateRejection::UnknownGoal { .. }) => rep.rejected_unknown_goal += 1,
                Err(_) => {}
            }
        }
        if decision == *exp {
            rep.correct += 1;
        }
    }
    rep.measured_total = t1.elapsed();

    ComparisonReport {
        baseline,
        generative: rep,
        disabled_abstentions,
    }
}
