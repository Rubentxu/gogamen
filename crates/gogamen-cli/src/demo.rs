//! GOG-178 / DEM-025: offline transfer demo, from the cognitive DSL spec to a replayable receipt.
//!
//! Pipeline (all in-process, no UI, LLM, JVM, network):
//! 1. the cognitive experiment spec emitted by the Kotlin DSL (golden corpus, byte-identical to the
//!    Rust canonical form) is parsed by the Rust spec parser and hashed;
//! 2. a tabular learned model (cell abstraction, GOG-167) is trained from random experience on
//!    grid map A only;
//! 3. the FMC planner (GOG-168) uses that model on the unseen map B with holdout seeds, next to
//!    baselines (random, untrained prior, tabular greedy) and a privileged ceiling reported apart;
//! 4. a receipt (`gogamen.demo.v1`) stores spec hash, seeds, model versions, results, limits and
//!    per-decision traces; `verify-receipt` re-executes it and compares the whole body.
//!
//! The decision rule is fixed in code before running and the result may be negative. There is no
//! hidden reward: the planner only maximises the task goal declared in the grid task.

use gogamen_arena::emergence::{
    GRID_DEMO_HOLDOUT, GRID_DEMO_TRAIN, GridTransferDemo, run_grid_transfer_demo,
};
use gogamen_spec::{AdaptiveExperimentSpec, JsonValue, parse_json};

use crate::receipt::{
    arr, as_obj, as_str, canonical_json, first_difference, fnv1a, get, hex, num_u, obj, st,
};

/// Receipt format tag of the demo.
pub const DEMO_FORMAT: &str = "gogamen.demo.v1";
/// Stable example id.
pub const DEMO_EXAMPLE_ID: &str = "grid-transfer-learned-vs-baseline";
/// DSL spec shipped with the repo (golden corpus, emitted by the Kotlin DSL).
pub const DEFAULT_DSL_SPEC: &str =
    include_str!("../../../spec/golden/cognitive_dsl/positive/h0.json");
pub const DEFAULT_DSL_SOURCE: &str = "spec/golden/cognitive_dsl/positive/h0.json";

/// Pre-declared decision rule (kept as text inside the receipt).
const DECISION_RULE: &str = "TRANSFER_SUPPORTED iff learned(A->B) goals > random goals AND learned(A->B) goals > untrained-prior goals on the holdout seeds; otherwise TRANSFER_NOT_SUPPORTED (reported, not hidden). Ceiling is never compared.";

/// Outcome of one demo run.
pub struct DemoOutcome {
    body: JsonValue,
    /// Hash of the DSL spec canonical JSON.
    pub spec_hash: u64,
    /// `TRANSFER_SUPPORTED` or `TRANSFER_NOT_SUPPORTED`.
    pub verdict: &'static str,
    /// Human summary lines (arm: goals/episodes).
    pub summary: Vec<String>,
    /// Number of episodes executed.
    pub episodes: usize,
}

impl DemoOutcome {
    /// Hash over the canonical body.
    pub fn receipt_hash(&self) -> u64 {
        fnv1a(canonical_json(&self.body).as_bytes())
    }

    /// Serialized receipt.
    pub fn receipt_text(&self) -> String {
        let doc = obj([
            ("body", self.body.clone()),
            ("receipt_hash", hex(self.receipt_hash())),
        ]);
        let mut s = canonical_json(&doc);
        s.push('\n');
        s
    }

    /// Self-contained headless HTML view of the receipt (no scripts, no network).
    pub fn render_html(&self) -> String {
        let mut h = String::from(
            "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><title>gogamen demo report</title><style>body{font:14px system-ui;max-width:900px;margin:2em auto}td,th{border:1px solid #bbb;padding:2px 8px}</style></head><body>\n",
        );
        h.push_str(&format!(
            "<h1>Transfer demo: {DEMO_EXAMPLE_ID}</h1>\n<p>spec_hash <code>{:016x}</code>, receipt_hash <code>{:016x}</code>. Offline, no hidden reward, no emergence or AGI claim.</p>\n<p>Verdict: <b>{}</b><br>Rule: {}</p>\n<table><tr><th>arm (trained on, evaluated on {GRID_DEMO_HOLDOUT})</th><th>goals / episodes</th></tr>\n",
            self.spec_hash,
            self.receipt_hash(),
            self.verdict,
            esc(DECISION_RULE)
        ));
        for l in &self.summary {
            h.push_str(&format!("<tr><td>{}</td></tr>\n", esc(l)));
        }
        h.push_str("</table>\n<p>Full traces, model versions and limits: see receipt.json.</p></body></html>\n");
        h
    }
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;")
}

fn seeds_json(s: &[u64]) -> JsonValue {
    arr(s.iter().map(|x| num_u(*x)).collect())
}

/// Runs the demo from DSL spec text (`dsl_source` is a label, e.g. the path).
pub fn run_demo(dsl_text: &str, dsl_source: &str) -> Result<DemoOutcome, String> {
    let spec = AdaptiveExperimentSpec::from_json(dsl_text)
        .map_err(|e| format!("DSL spec rejected by the Rust parser: {e}"))?;
    let canonical = spec
        .to_canonical_json()
        .map_err(|e| format!("DSL spec has no canonical form: {e}"))?;
    let spec_hash = fnv1a(canonical.as_bytes());
    let demo = run_grid_transfer_demo().map_err(|e| format!("demo run failed: {e}"))?;
    let learned = demo.goals("planner_learned_fmc", GRID_DEMO_TRAIN);
    let prior = demo.goals("planner_prior_ablation", "none");
    let random = demo.goals("random", "none");
    let supported = learned.0 > random.0 && learned.0 > prior.0;
    let verdict = if supported {
        "TRANSFER_SUPPORTED"
    } else {
        "TRANSFER_NOT_SUPPORTED"
    };
    let summary = summarize(&demo);
    let body = obj([
        ("format", st(DEMO_FORMAT)),
        ("example_id", st(DEMO_EXAMPLE_ID)),
        ("gogamen_version", st(env!("CARGO_PKG_VERSION"))),
        (
            "spec",
            obj([
                ("source", st(dsl_source)),
                ("canonical_json", st(&canonical)),
                ("spec_hash", hex(spec_hash)),
                ("experiment_id", st(&spec.experiment_id)),
                ("spec_dev_seeds", seeds_json(&spec.seeds.dev_seeds)),
                ("spec_holdout_seeds", seeds_json(&spec.seeds.holdout_seeds)),
                (
                    "binding_note",
                    st(
                        "the spec is parsed, validated and hashed; the demo world, seeds and budget are fixed in the Arena protocol below and are not yet derived from the spec",
                    ),
                ),
            ]),
        ),
        (
            "protocol",
            obj([
                ("train_map", st(GRID_DEMO_TRAIN)),
                ("holdout_map", st(GRID_DEMO_HOLDOUT)),
                ("holdout_seeds", seeds_json(&demo.holdout_seeds)),
                ("observations_per_seed", num_u(demo.observations as u64)),
                ("max_ticks", num_u(demo.max_ticks)),
                (
                    "budget_transitions_per_decision",
                    num_u(demo.budget_transitions_per_decision as u64),
                ),
                (
                    "reward",
                    st("none beyond the declared grid task goal; no hidden shaping"),
                ),
                ("decision_rule", st(DECISION_RULE)),
            ]),
        ),
        (
            "runtime",
            obj([
                ("ui", JsonValue::Bool(false)),
                ("llm", JsonValue::Bool(false)),
                ("jvm", JsonValue::Bool(false)),
                ("network", JsonValue::Bool(false)),
            ]),
        ),
        (
            "episodes",
            arr(demo.episodes.iter().map(episode_json).collect()),
        ),
        ("summary", arr(summary.iter().map(|l| st(l)).collect())),
        (
            "verdict",
            obj([
                ("result", st(verdict)),
                ("learned_goals", num_u(learned.0 as u64)),
                ("prior_goals", num_u(prior.0 as u64)),
                ("random_goals", num_u(random.0 as u64)),
                ("episodes_per_arm", num_u(learned.1 as u64)),
            ]),
        ),
        (
            "limits",
            arr(vec![
                st(
                    "single grid pair (A->B), 6 holdout seeds: one episode = 1/6; differences of one episode are noise",
                ),
                st(
                    "no emergence, AGI or general-transfer claim; the verdict concerns only these maps and seeds",
                ),
                st("Kart variant not included in this slice"),
                st("decision traces record action index per decision, not full environment state"),
            ]),
        ),
    ]);
    Ok(DemoOutcome {
        body,
        spec_hash,
        verdict,
        summary,
        episodes: demo.episodes.len(),
    })
}

fn summarize(demo: &GridTransferDemo) -> Vec<String> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for e in &demo.episodes {
        let k = (e.arm.clone(), e.trained_on.clone());
        if !seen.contains(&k) {
            seen.push(k);
        }
    }
    seen.into_iter()
        .map(|(arm, tr)| {
            let (g, n) = demo.goals(&arm, &tr);
            let role = if arm == "privileged_fmc_ceiling" {
                " [privileged ceiling, not compared]"
            } else {
                ""
            };
            format!("{arm} (trained_on={tr}): {g}/{n} goals{role}")
        })
        .collect()
}

fn episode_json(e: &gogamen_arena::emergence::GridDemoEpisode) -> JsonValue {
    let opt = |v: Option<u64>| v.map_or(JsonValue::Null, hex);
    obj([
        ("arm", st(&e.arm)),
        ("trained_on", st(&e.trained_on)),
        ("evaluated_on", st(&e.evaluated_on)),
        ("seed", num_u(e.seed)),
        ("reached_goal", JsonValue::Bool(e.reached_goal)),
        ("failed", JsonValue::Bool(e.failed)),
        ("ticks", num_u(e.ticks)),
        ("planning_transitions", num_u(e.planning_transitions)),
        (
            "model_revision",
            e.model_revision.map_or(JsonValue::Null, num_u),
        ),
        ("model_checksum", opt(e.model_checksum)),
        ("fallback_decisions", num_u(e.fallback_decisions)),
        (
            "decision_trace",
            arr(e
                .decision_trace
                .iter()
                .map(|(tick, action, used, fb)| {
                    arr(vec![
                        num_u(*tick),
                        num_u(*action as u64),
                        num_u(*used as u64),
                        fb.map_or(JsonValue::Null, st),
                    ])
                })
                .collect()),
        ),
    ])
}

/// Result of verifying a demo receipt.
pub struct DemoVerification {
    /// Receipt hash.
    pub receipt_hash: u64,
    /// Spec hash.
    pub spec_hash: u64,
    /// Episodes reproduced.
    pub episodes: usize,
    /// Verdict reproduced.
    pub verdict: &'static str,
}

/// True when the receipt text declares the demo format.
pub fn is_demo_receipt(text: &str) -> bool {
    parse_json(text)
        .ok()
        .and_then(|d| match d {
            JsonValue::Object(o) => match o.get("body") {
                Some(JsonValue::Object(b)) => match b.get("format") {
                    Some(JsonValue::String(f)) => Some(f == DEMO_FORMAT),
                    _ => None,
                },
                _ => None,
            },
            _ => None,
        })
        .unwrap_or(false)
}

/// Re-executes the demo from the spec stored in the receipt and compares the full body.
pub fn verify_demo_receipt(text: &str) -> Result<DemoVerification, String> {
    let doc = parse_json(text).map_err(|e| format!("receipt is not valid JSON: {e}"))?;
    let top = as_obj(&doc, "receipt")?;
    let body = get(top, "body")?;
    let stored = as_str(get(top, "receipt_hash")?, "receipt_hash")?;
    let computed = format!("{:016x}", fnv1a(canonical_json(body).as_bytes()));
    if stored != computed {
        return Err(format!(
            "receipt hash mismatch: stored {stored}, body hashes to {computed} (the receipt was edited)"
        ));
    }
    let b = as_obj(body, "body")?;
    let spec = as_obj(get(b, "spec")?, "spec")?;
    let dsl = as_str(get(spec, "canonical_json")?, "canonical_json")?;
    let source = as_str(get(spec, "source")?, "source")?;
    let fresh = run_demo(dsl, source)?;
    let fresh_body = parse_json(&canonical_json(&fresh.body))
        .map_err(|e| format!("internal error re-reading reproduced body: {e}"))?;
    if let Some(d) = first_difference(body, &fresh_body, "body") {
        return Err(format!("reproduction differs from the receipt: {d}"));
    }
    Ok(DemoVerification {
        receipt_hash: fresh.receipt_hash(),
        spec_hash: fresh.spec_hash,
        episodes: fresh.episodes,
        verdict: fresh.verdict,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decisive demo test (GOG-178): one run emits spec hash, seeds, versioned models, results and
    /// traces; a learned model trained on map A is evaluated on holdout map B against baselines;
    /// the verdict is reported either way; the receipt re-executes identically; edits are caught.
    #[test]
    fn demo_transfer() {
        let a = run_demo(DEFAULT_DSL_SPEC, DEFAULT_DSL_SOURCE).unwrap();
        let b = run_demo(DEFAULT_DSL_SPEC, DEFAULT_DSL_SOURCE).unwrap();
        assert_eq!(a.receipt_text(), b.receipt_text(), "deterministic");
        assert!(a.verdict == "TRANSFER_SUPPORTED" || a.verdict == "TRANSFER_NOT_SUPPORTED");
        let text = a.receipt_text();
        for needle in [
            "spec_hash",
            "holdout_seeds",
            "model_checksum",
            "decision_trace",
            "limits",
            "planner_learned_fmc",
            "random",
            "\"jvm\":false",
        ] {
            assert!(text.contains(needle), "missing {needle}");
        }
        assert!(a.summary.iter().any(|l| l.contains("random")));
        assert!(is_demo_receipt(&text));
        let v = verify_demo_receipt(&text).unwrap();
        assert_eq!(v.receipt_hash, a.receipt_hash());
        assert_eq!(v.episodes, a.episodes);
        // Tampering with a result is detected.
        let tampered = text.replacen("\"reached_goal\":true", "\"reached_goal\":false", 1);
        if tampered != text {
            assert!(verify_demo_receipt(&tampered).is_err());
        }
        // An invalid DSL spec is rejected.
        assert!(run_demo("{}", "x").is_err());
    }
}
