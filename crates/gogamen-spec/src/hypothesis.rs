//! Pre-registered hypotheses (`HypothesisSpec`) and their canonical hash.
//!
//! A hypothesis states a claim about one metric, a support condition and a refutation
//! condition. The hash over the canonical form is recorded in the trace manifest *before*
//! execution, so any later edit of a threshold is detectable. The verdict
//! (`SUPPORTED | REFUTED | INCONCLUSIVE`) is evaluated elsewhere and is independent of the
//! build/test status.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::TraceHasher;

use crate::codec::{as_object, reject_unknown_fields, require_array, require_string};
use crate::error::{SpecError, UnknownFieldPolicy};
use crate::json::{JsonNumber, JsonValue, write_canonical_f64, write_json_string};
use crate::model::validate_non_empty;

/// Comparison operator between an observed metric and a threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypothesisComparator {
    /// `observed > threshold`.
    Gt,
    /// `observed >= threshold`.
    Ge,
    /// `observed < threshold`.
    Lt,
    /// `observed <= threshold`.
    Le,
    /// `observed == threshold` (exact).
    Eq,
}

impl HypothesisComparator {
    /// Stable textual name used in the canonical JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Gt => "gt",
            Self::Ge => "ge",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Eq => "eq",
        }
    }

    /// Parses the stable textual name.
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "gt" => Self::Gt,
            "ge" => Self::Ge,
            "lt" => Self::Lt,
            "le" => Self::Le,
            "eq" => Self::Eq,
            _ => return None,
        })
    }

    /// Applies the comparison `observed <op> threshold`.
    pub fn holds(self, observed: f64, threshold: f64) -> bool {
        match self {
            Self::Gt => observed > threshold,
            Self::Ge => observed >= threshold,
            Self::Lt => observed < threshold,
            Self::Le => observed <= threshold,
            Self::Eq => observed == threshold,
        }
    }
}

/// One pre-registered, falsifiable hypothesis.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisSpec {
    /// Unique identifier inside the experiment.
    pub hypothesis_id: String,
    /// Human-readable claim.
    pub claim: String,
    /// Name of the metric the hypothesis is evaluated on.
    pub metric: String,
    /// Support condition: `metric <comparator> threshold` supports the claim.
    pub comparator: HypothesisComparator,
    /// Support threshold.
    pub threshold: f64,
    /// Refutation condition: `metric <refutation_comparator> refutation_threshold` refutes it.
    pub refutation_comparator: HypothesisComparator,
    /// Refutation threshold.
    pub refutation_threshold: f64,
}

impl HypothesisSpec {
    /// Builds a hypothesis.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hypothesis_id: impl Into<String>,
        claim: impl Into<String>,
        metric: impl Into<String>,
        comparator: HypothesisComparator,
        threshold: f64,
        refutation_comparator: HypothesisComparator,
        refutation_threshold: f64,
    ) -> Self {
        Self {
            hypothesis_id: hypothesis_id.into(),
            claim: claim.into(),
            metric: metric.into(),
            comparator,
            threshold,
            refutation_comparator,
            refutation_threshold,
        }
    }

    /// Validates non-empty text fields and finite thresholds.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.hypothesis_id, "HypothesisSpec.hypothesis_id")?;
        validate_non_empty(&self.claim, "HypothesisSpec.claim")?;
        validate_non_empty(&self.metric, "HypothesisSpec.metric")?;
        for (v, name) in [
            (self.threshold, "HypothesisSpec.threshold"),
            (
                self.refutation_threshold,
                "HypothesisSpec.refutation_threshold",
            ),
        ] {
            if !v.is_finite() {
                return Err(SpecError::NonFiniteFloat {
                    context: name.to_owned(),
                });
            }
        }
        Ok(())
    }
}

/// Validates every hypothesis and identifier uniqueness.
pub fn validate_hypotheses(hypotheses: &[HypothesisSpec]) -> Result<(), SpecError> {
    let mut seen = BTreeSet::new();
    for h in hypotheses {
        h.validate()?;
        if !seen.insert(h.hypothesis_id.as_str()) {
            return Err(SpecError::DuplicateIdentifier {
                context: "AdaptiveExperimentSpec.hypotheses",
                id: h.hypothesis_id.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn write_hypothesis(out: &mut String, h: &HypothesisSpec) -> Result<(), SpecError> {
    out.push_str("{\"claim\":");
    write_json_string(out, &h.claim);
    let _ = write!(out, ",\"comparator\":\"{}\"", h.comparator.as_str());
    out.push_str(",\"hypothesis_id\":");
    write_json_string(out, &h.hypothesis_id);
    out.push_str(",\"metric\":");
    write_json_string(out, &h.metric);
    let _ = write!(
        out,
        ",\"refutation_comparator\":\"{}\",\"refutation_threshold\":",
        h.refutation_comparator.as_str()
    );
    write_canonical_f64(out, h.refutation_threshold)?;
    out.push_str(",\"threshold\":");
    write_canonical_f64(out, h.threshold)?;
    out.push('}');
    Ok(())
}

/// Canonical JSON array of the hypotheses, sorted by `hypothesis_id`.
pub fn canonical_hypotheses_json(hypotheses: &[HypothesisSpec]) -> Result<String, SpecError> {
    validate_hypotheses(hypotheses)?;
    let mut sorted: Vec<&HypothesisSpec> = hypotheses.iter().collect();
    sorted.sort_by(|a, b| a.hypothesis_id.cmp(&b.hypothesis_id));
    let mut out = String::from("[");
    for (i, h) in sorted.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_hypothesis(&mut out, h)?;
    }
    out.push(']');
    Ok(out)
}

/// Pre-registration hash (FNV-1a via `TraceHasher`) of the canonical hypotheses.
///
/// Returns `None` when no hypothesis is registered. Order-independent; changing any
/// claim, metric, comparator or threshold changes the hash.
pub fn hypotheses_hash(hypotheses: &[HypothesisSpec]) -> Result<Option<u64>, SpecError> {
    if hypotheses.is_empty() {
        return Ok(None);
    }
    let json = canonical_hypotheses_json(hypotheses)?;
    let mut hasher = TraceHasher::new();
    hasher.write_str("gogamen.hypotheses.v1");
    hasher.write_str(&json);
    Ok(Some(hasher.finish()))
}

fn num_f64(map: &BTreeMap<String, JsonValue>, field: &'static str) -> Result<f64, SpecError> {
    const CTX: &str = "HypothesisSpec";
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::F64(n))) => Ok(*n),
        Some(JsonValue::Number(JsonNumber::I64(n))) => Ok(*n as f64),
        Some(JsonValue::Number(JsonNumber::U64(n))) => Ok(*n as f64),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context: CTX,
            field,
            reason: "expected number".to_owned(),
        }),
        None => Err(SpecError::MissingField {
            context: CTX,
            field,
        }),
    }
}

fn comparator(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
) -> Result<HypothesisComparator, SpecError> {
    let s = require_string(map, field, "HypothesisSpec")?;
    HypothesisComparator::parse(s).ok_or_else(|| SpecError::InvalidFieldValue {
        context: "HypothesisSpec",
        field,
        reason: format!("unknown comparator `{s}`"),
    })
}

pub(crate) fn parse_hypotheses(
    map: &BTreeMap<String, JsonValue>,
    policy: UnknownFieldPolicy,
) -> Result<Vec<HypothesisSpec>, SpecError> {
    const CTX: &str = "HypothesisSpec";
    if !map.contains_key("hypotheses") {
        return Ok(Vec::new());
    }
    let items = require_array(map, "hypotheses", "AdaptiveExperimentSpec")?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let m = as_object(item, CTX)?;
        reject_unknown_fields(
            m,
            &[
                "hypothesis_id",
                "claim",
                "metric",
                "comparator",
                "threshold",
                "refutation_comparator",
                "refutation_threshold",
            ],
            CTX,
            policy,
        )?;
        out.push(HypothesisSpec {
            hypothesis_id: require_string(m, "hypothesis_id", CTX)?.to_owned(),
            claim: require_string(m, "claim", CTX)?.to_owned(),
            metric: require_string(m, "metric", CTX)?.to_owned(),
            comparator: comparator(m, "comparator")?,
            threshold: num_f64(m, "threshold")?,
            refutation_comparator: comparator(m, "refutation_comparator")?,
            refutation_threshold: num_f64(m, "refutation_threshold")?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::{AdaptiveExperimentSpec, canonical_grid_experiment_fixture};

    fn sample() -> HypothesisSpec {
        HypothesisSpec::new(
            "h1",
            "planner beats random",
            "success_rate",
            HypothesisComparator::Ge,
            0.8,
            HypothesisComparator::Lt,
            0.5,
        )
    }

    #[test]
    fn hypothesis_hash_none_when_empty_and_changes_with_threshold() {
        assert_eq!(hypotheses_hash(&[]).unwrap(), None);
        let a = hypotheses_hash(&[sample()]).unwrap().unwrap();
        let mut moved = sample();
        moved.threshold = 0.7;
        let b = hypotheses_hash(&[moved]).unwrap().unwrap();
        assert_ne!(a, b);
        assert_eq!(a, hypotheses_hash(&[sample()]).unwrap().unwrap());
    }

    #[test]
    fn hypothesis_hash_is_order_independent() {
        let mut other = sample();
        other.hypothesis_id = "h0".to_owned();
        let ab = hypotheses_hash(&[sample(), other.clone()]).unwrap();
        let ba = hypotheses_hash(&[other, sample()]).unwrap();
        assert_eq!(ab, ba);
    }

    #[test]
    fn hypothesis_validation_rejects_bad_input() {
        let mut bad = sample();
        bad.threshold = f64::NAN;
        assert!(bad.validate().is_err());
        assert!(validate_hypotheses(&[sample(), sample()]).is_err());
        let mut empty = sample();
        empty.metric = " ".to_owned();
        assert!(empty.validate().is_err());
    }

    #[test]
    fn hypothesis_experiment_json_roundtrip_and_backward_compat() {
        let base = canonical_grid_experiment_fixture();
        let base_json = base.to_canonical_json().unwrap();
        assert!(!base_json.contains("hypotheses"));

        let mut with = base.clone();
        with.hypotheses.push(sample());
        let json = with.to_canonical_json().unwrap();
        assert!(json.contains("\"hypotheses\":["));
        let parsed = AdaptiveExperimentSpec::from_json(&json).unwrap();
        assert_eq!(parsed.hypotheses, with.hypotheses);
        assert_eq!(parsed.to_canonical_json().unwrap(), json);
        // Old documents (without the field) still parse to an empty list.
        let old = AdaptiveExperimentSpec::from_json(&base_json).unwrap();
        assert!(old.hypotheses.is_empty());
    }
}
