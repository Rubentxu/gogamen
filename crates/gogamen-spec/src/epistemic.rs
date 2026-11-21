//! Especificaciones declarativas epistémicas (`Choice`/`Noul`/`Score`, `AbstentionPolicy`,
//! `Ask`/`Inspect`, `ProvenanceSpec` y `GuardPolicySpec`) para `SystemSpec` (`std`-only).

use std::collections::{BTreeMap, BTreeSet};

use crate::codec::{as_object, reject_unknown_fields, require_string};
use crate::error::{SpecError, UnknownFieldPolicy};
use crate::json::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};

/// Modelos offline disponibles por defecto para grounding/preguntas epistémicas.
pub const AVAILABLE_EPISTEMIC_MODELS: &[&str] = &[
    "offline_replay_v1",
    "system_one_choice_v1",
    "system_one_noul_v1",
    "system_one_score_v1",
];

/// Pregunta epistémica tipada (`Choice`, `Noul` o `Score`).
#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicQuestionSpec {
    Choice {
        query_id: String,
        prompt: String,
        options: Vec<String>,
        model_id: String,
        deadline_ticks: Option<u64>,
    },
    Noul {
        query_id: String,
        prompt: String,
        hypothesis_id: String,
        model_id: String,
        deadline_ticks: Option<u64>,
    },
    Score {
        query_id: String,
        prompt: String,
        options: Vec<String>,
        min_score: f64,
        max_score: f64,
        model_id: String,
        deadline_ticks: Option<u64>,
    },
}

impl EpistemicQuestionSpec {
    pub fn model_id(&self) -> &str {
        match self {
            Self::Choice { model_id, .. }
            | Self::Noul { model_id, .. }
            | Self::Score { model_id, .. } => model_id,
        }
    }

    pub fn validate(&self) -> Result<(), SpecError> {
        match self {
            Self::Choice {
                query_id,
                prompt,
                options,
                model_id,
                deadline_ticks,
            } => {
                require_non_empty(query_id, "EpistemicQuestion.query_id")?;
                require_non_empty(prompt, "EpistemicQuestion.prompt")?;
                require_non_empty(model_id, "EpistemicQuestion.model_id")?;
                validate_options(options, "EpistemicQuestion.options")?;
                validate_deadline(*deadline_ticks)?;
            }
            Self::Noul {
                query_id,
                prompt,
                hypothesis_id,
                model_id,
                deadline_ticks,
            } => {
                require_non_empty(query_id, "EpistemicQuestion.query_id")?;
                require_non_empty(prompt, "EpistemicQuestion.prompt")?;
                require_non_empty(hypothesis_id, "EpistemicQuestion.hypothesis_id")?;
                require_non_empty(model_id, "EpistemicQuestion.model_id")?;
                validate_deadline(*deadline_ticks)?;
            }
            Self::Score {
                query_id,
                prompt,
                options,
                min_score,
                max_score,
                model_id,
                deadline_ticks,
            } => {
                require_non_empty(query_id, "EpistemicQuestion.query_id")?;
                require_non_empty(prompt, "EpistemicQuestion.prompt")?;
                require_non_empty(model_id, "EpistemicQuestion.model_id")?;
                validate_options(options, "EpistemicQuestion.options")?;
                if !min_score.is_finite() || !max_score.is_finite() || min_score >= max_score {
                    return Err(SpecError::InvalidFieldValue {
                        context: "EpistemicQuestion.Score",
                        field: "min_score/max_score",
                        reason: format!(
                            "min_score ({min_score}) and max_score ({max_score}) must be finite with min_score < max_score"
                        ),
                    });
                }
                validate_deadline(*deadline_ticks)?;
            }
        }
        Ok(())
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_question_json(&mut out, self)?;
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let q = parse_question_json(&value)?;
        q.validate()?;
        Ok(q)
    }
}

/// Política de abstención explícita.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstentionPolicySpec {
    pub min_confidence: f64,
    pub fallback_action: String,
    pub abstain_on_timeout: bool,
    pub abstain_on_ambiguity: bool,
    pub ambiguity_margin: f64,
}

impl AbstentionPolicySpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_unit_interval(self.min_confidence, "AbstentionPolicy", "min_confidence")?;
        require_non_empty(&self.fallback_action, "AbstentionPolicy.fallback_action")?;
        validate_unit_interval(
            self.ambiguity_margin,
            "AbstentionPolicy",
            "ambiguity_margin",
        )?;
        Ok(())
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"abstain_on_ambiguity\":");
        out.push_str(if self.abstain_on_ambiguity {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"abstain_on_timeout\":");
        out.push_str(if self.abstain_on_timeout {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"ambiguity_margin\":");
        write_canonical_f64(&mut out, self.ambiguity_margin)?;
        out.push_str(",\"fallback_action\":");
        write_json_string(&mut out, &self.fallback_action);
        out.push_str(",\"min_confidence\":");
        write_canonical_f64(&mut out, self.min_confidence)?;
        out.push('}');
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let map = as_object(&value, "AbstentionPolicy")?;
        reject_unknown_fields(
            map,
            &[
                "abstain_on_ambiguity",
                "abstain_on_timeout",
                "ambiguity_margin",
                "fallback_action",
                "min_confidence",
            ],
            "AbstentionPolicy",
            UnknownFieldPolicy::Reject,
        )?;
        let spec = Self {
            abstain_on_ambiguity: require_bool(map, "abstain_on_ambiguity", "AbstentionPolicy")?,
            abstain_on_timeout: require_bool(map, "abstain_on_timeout", "AbstentionPolicy")?,
            ambiguity_margin: require_f64(map, "ambiguity_margin", "AbstentionPolicy")?,
            fallback_action: require_string(map, "fallback_action", "AbstentionPolicy")?.to_owned(),
            min_confidence: require_f64(map, "min_confidence", "AbstentionPolicy")?,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Especificación declarativa de acción epistémica `Ask`.
#[derive(Debug, Clone, PartialEq)]
pub struct AskActionSpec {
    pub recipient: String,
    pub scope: String,
    pub question: EpistemicQuestionSpec,
    pub comm_cost: f64,
    pub step_cost: f64,
    pub response_probability: f64,
    pub timeout_ticks: u64,
}

impl AskActionSpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        require_non_empty(&self.recipient, "AskAction.recipient")?;
        require_non_empty(&self.scope, "AskAction.scope")?;
        self.question.validate()?;
        validate_non_negative_finite(self.comm_cost, "AskAction", "comm_cost")?;
        validate_non_negative_finite(self.step_cost, "AskAction", "step_cost")?;
        validate_unit_interval(
            self.response_probability,
            "AskAction",
            "response_probability",
        )?;
        if self.timeout_ticks == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "AskAction",
                field: "timeout_ticks",
                reason: "timeout_ticks must be > 0".to_owned(),
            });
        }
        Ok(())
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"comm_cost\":");
        write_canonical_f64(&mut out, self.comm_cost)?;
        out.push_str(",\"question\":");
        write_question_json(&mut out, &self.question)?;
        out.push_str(",\"recipient\":");
        write_json_string(&mut out, &self.recipient);
        out.push_str(",\"response_probability\":");
        write_canonical_f64(&mut out, self.response_probability)?;
        out.push_str(",\"scope\":");
        write_json_string(&mut out, &self.scope);
        out.push_str(",\"step_cost\":");
        write_canonical_f64(&mut out, self.step_cost)?;
        out.push_str(&format!(",\"timeout_ticks\":{}}}", self.timeout_ticks));
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        Self::from_json_value(&value)
    }

    /// Parsea una acción `Ask` desde un [`JsonValue`] ya decodificado.
    pub(crate) fn from_json_value(value: &JsonValue) -> Result<Self, SpecError> {
        let map = as_object(value, "AskAction")?;
        reject_unknown_fields(
            map,
            &[
                "comm_cost",
                "question",
                "recipient",
                "response_probability",
                "scope",
                "step_cost",
                "timeout_ticks",
            ],
            "AskAction",
            UnknownFieldPolicy::Reject,
        )?;
        let q_val = map.get("question").ok_or(SpecError::MissingField {
            context: "AskAction",
            field: "question",
        })?;
        let spec = Self {
            comm_cost: require_f64(map, "comm_cost", "AskAction")?,
            question: parse_question_json(q_val)?,
            recipient: require_string(map, "recipient", "AskAction")?.to_owned(),
            response_probability: require_f64(map, "response_probability", "AskAction")?,
            scope: require_string(map, "scope", "AskAction")?.to_owned(),
            step_cost: require_f64(map, "step_cost", "AskAction")?,
            timeout_ticks: require_u64(map, "timeout_ticks", "AskAction")?,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Especificación declarativa de acción epistémica `Inspect`.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectActionSpec {
    pub sensor_id: String,
    pub inspect_cost: f64,
    pub target_entity: Option<String>,
}

impl InspectActionSpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        require_non_empty(&self.sensor_id, "InspectAction.sensor_id")?;
        validate_non_negative_finite(self.inspect_cost, "InspectAction", "inspect_cost")?;
        if let Some(target) = &self.target_entity {
            require_non_empty(target, "InspectAction.target_entity")?;
        }
        Ok(())
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"inspect_cost\":");
        write_canonical_f64(&mut out, self.inspect_cost)?;
        out.push_str(",\"sensor_id\":");
        write_json_string(&mut out, &self.sensor_id);
        if let Some(target) = &self.target_entity {
            out.push_str(",\"target_entity\":");
            write_json_string(&mut out, target);
        }
        out.push('}');
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let map = as_object(&value, "InspectAction")?;
        reject_unknown_fields(
            map,
            &["inspect_cost", "sensor_id", "target_entity"],
            "InspectAction",
            UnknownFieldPolicy::Reject,
        )?;
        let target_entity = match map.get("target_entity") {
            Some(JsonValue::String(s)) => Some(s.clone()),
            Some(_) => {
                return Err(SpecError::InvalidFieldValue {
                    context: "InspectAction",
                    field: "target_entity",
                    reason: "expected string".to_owned(),
                });
            }
            None => None,
        };
        let spec = Self {
            inspect_cost: require_f64(map, "inspect_cost", "InspectAction")?,
            sensor_id: require_string(map, "sensor_id", "InspectAction")?.to_owned(),
            target_entity,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Metadatos explícitos de procedencia (`source_id`, `sensor_id`, `evidence_ref`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProvenanceSpec {
    pub source_id: String,
    pub sensor_id: u64,
    pub evidence_ref: u64,
    pub epistemic_status: String,
}

impl ProvenanceSpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        require_non_empty(&self.source_id, "ProvenanceSpec.source_id")?;
        if self.sensor_id == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "ProvenanceSpec",
                field: "sensor_id",
                reason: "sensor_id must be > 0".to_owned(),
            });
        }
        if self.evidence_ref == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "ProvenanceSpec",
                field: "evidence_ref",
                reason: "evidence_ref must be > 0".to_owned(),
            });
        }
        match self.epistemic_status.as_str() {
            "observed" | "reported" | "inferred" | "hypothetical" => Ok(()),
            other => Err(SpecError::InvalidFieldValue {
                context: "ProvenanceSpec",
                field: "epistemic_status",
                reason: format!("unsupported epistemic_status '{other}'"),
            }),
        }
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"epistemic_status\":");
        write_json_string(&mut out, &self.epistemic_status);
        out.push_str(&format!(
            ",\"evidence_ref\":{},\"sensor_id\":{},\"source_id\":",
            self.evidence_ref, self.sensor_id
        ));
        write_json_string(&mut out, &self.source_id);
        out.push('}');
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let map = as_object(&value, "ProvenanceSpec")?;
        reject_unknown_fields(
            map,
            &["epistemic_status", "evidence_ref", "sensor_id", "source_id"],
            "ProvenanceSpec",
            UnknownFieldPolicy::Reject,
        )?;
        let spec = Self {
            epistemic_status: require_string(map, "epistemic_status", "ProvenanceSpec")?.to_owned(),
            evidence_ref: require_u64(map, "evidence_ref", "ProvenanceSpec")?,
            sensor_id: require_u64(map, "sensor_id", "ProvenanceSpec")?,
            source_id: require_string(map, "source_id", "ProvenanceSpec")?.to_owned(),
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Política inmutable de `ActionGuard` y aislamiento de permisos frente a mensajes humanos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardPolicySpec {
    pub guard_policy: String,
    pub allowed_actions: Vec<String>,
    pub immutable_permissions: bool,
    pub human_messages_cannot_mutate_permissions: bool,
}

impl GuardPolicySpec {
    pub fn validate(&self) -> Result<(), SpecError> {
        match self.guard_policy.as_str() {
            "reject_only" | "explicit_fallback" => {}
            other => {
                return Err(SpecError::InvalidFieldValue {
                    context: "GuardPolicySpec",
                    field: "guard_policy",
                    reason: format!("unsupported guard_policy '{other}'"),
                });
            }
        }
        if !self.immutable_permissions {
            return Err(SpecError::InvalidFieldValue {
                context: "GuardPolicySpec",
                field: "immutable_permissions",
                reason: "PermissionSet must remain immutable".to_owned(),
            });
        }
        if !self.human_messages_cannot_mutate_permissions {
            return Err(SpecError::InvalidFieldValue {
                context: "GuardPolicySpec",
                field: "human_messages_cannot_mutate_permissions",
                reason: "human messages/proposals cannot mutate ActionGuard or PermissionSet"
                    .to_owned(),
            });
        }
        let mut seen = BTreeSet::new();
        for action in &self.allowed_actions {
            require_non_empty(action, "GuardPolicySpec.allowed_actions")?;
            if !seen.insert(action.as_str()) {
                return Err(SpecError::InvalidFieldValue {
                    context: "GuardPolicySpec",
                    field: "allowed_actions",
                    reason: format!("duplicate allowed action '{action}'"),
                });
            }
        }
        Ok(())
    }

    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"allowed_actions\":[");
        for (idx, action) in self.allowed_actions.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            write_json_string(&mut out, action);
        }
        out.push_str("],\"guard_policy\":");
        write_json_string(&mut out, &self.guard_policy);
        out.push_str(",\"human_messages_cannot_mutate_permissions\":");
        out.push_str(if self.human_messages_cannot_mutate_permissions {
            "true"
        } else {
            "false"
        });
        out.push_str(",\"immutable_permissions\":");
        out.push_str(if self.immutable_permissions {
            "true"
        } else {
            "false"
        });
        out.push('}');
        Ok(out)
    }

    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        Self::from_json_value(&value)
    }

    /// Parsea una política de guardia desde un [`JsonValue`] ya decodificado.
    pub(crate) fn from_json_value(value: &JsonValue) -> Result<Self, SpecError> {
        let map = as_object(value, "GuardPolicySpec")?;
        reject_unknown_fields(
            map,
            &[
                "allowed_actions",
                "guard_policy",
                "human_messages_cannot_mutate_permissions",
                "immutable_permissions",
            ],
            "GuardPolicySpec",
            UnknownFieldPolicy::Reject,
        )?;
        let allowed_actions = require_string_array(map, "allowed_actions", "GuardPolicySpec")?;
        let spec = Self {
            allowed_actions,
            guard_policy: require_string(map, "guard_policy", "GuardPolicySpec")?.to_owned(),
            human_messages_cannot_mutate_permissions: require_bool(
                map,
                "human_messages_cannot_mutate_permissions",
                "GuardPolicySpec",
            )?,
            immutable_permissions: require_bool(map, "immutable_permissions", "GuardPolicySpec")?,
        };
        spec.validate()?;
        Ok(spec)
    }
}

/// Bloque epistémico tipado de un agente: pregunta de grounding, acción `Ask` y guardia.
///
/// Sustituye a las claves de configuración legadas `question_json`, `ask_json` y `guard_json`
/// que la fixture `language_world` transportaba como cadenas dentro de `config`. Los tres
/// campos son opcionales y se omiten del JSON canónico cuando no se declaran, de modo que un
/// agente sin bloque epistémico conserva sus bytes históricos.
#[derive(Debug, Clone, PartialEq)]
pub struct EpistemicSpec {
    /// Pregunta de grounding declarada.
    pub question: Option<EpistemicQuestionSpec>,
    /// Acción `Ask` declarada.
    pub ask: Option<AskActionSpec>,
    /// Política de guardia declarada.
    pub guard: Option<GuardPolicySpec>,
}

impl EpistemicSpec {
    /// Valida las tres piezas declaradas.
    pub fn validate(&self) -> Result<(), SpecError> {
        if let Some(question) = &self.question {
            question.validate()?;
        }
        if let Some(ask) = &self.ask {
            ask.validate()?;
        }
        if let Some(guard) = &self.guard {
            guard.validate()?;
        }
        Ok(())
    }

    /// Emite el bloque canónico (claves ordenadas, piezas ausentes omitidas).
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::from("{");
        let mut first = true;
        if let Some(ask) = &self.ask {
            out.push_str("\"ask\":");
            out.push_str(&ask.to_canonical_json()?);
            first = false;
        }
        if let Some(guard) = &self.guard {
            if !first {
                out.push(',');
            }
            out.push_str("\"guard\":");
            out.push_str(&guard.to_canonical_json()?);
            first = false;
        }
        if let Some(question) = &self.question {
            if !first {
                out.push(',');
            }
            out.push_str("\"question\":");
            write_question_json(&mut out, question)?;
        }
        out.push('}');
        Ok(out)
    }

    /// Parsea un bloque epistémico desde un [`JsonValue`] ya decodificado.
    pub(crate) fn from_json_value(
        value: &JsonValue,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let map = as_object(value, "EpistemicSpec")?;
        reject_unknown_fields(
            map,
            &["question", "ask", "guard"],
            "EpistemicSpec",
            policy,
        )?;
        let spec = Self {
            question: map.get("question").map(parse_question_json).transpose()?,
            ask: map
                .get("ask")
                .map(AskActionSpec::from_json_value)
                .transpose()?,
            guard: map
                .get("guard")
                .map(GuardPolicySpec::from_json_value)
                .transpose()?,
        };
        spec.validate()?;
        Ok(spec)
    }
}

fn require_non_empty(value: &str, context: &'static str) -> Result<(), SpecError> {
    if value.trim().is_empty() {
        Err(SpecError::EmptyIdentifier { context })
    } else {
        Ok(())
    }
}

fn validate_options(options: &[String], context: &'static str) -> Result<(), SpecError> {
    if options.is_empty() {
        return Err(SpecError::InvalidFieldValue {
            context,
            field: "options",
            reason: "options must not be empty".to_owned(),
        });
    }
    let mut seen = BTreeSet::new();
    for opt in options {
        require_non_empty(opt, context)?;
        if !seen.insert(opt.as_str()) {
            return Err(SpecError::InvalidFieldValue {
                context,
                field: "options",
                reason: format!("duplicate option '{opt}'"),
            });
        }
    }
    Ok(())
}

fn validate_deadline(deadline_ticks: Option<u64>) -> Result<(), SpecError> {
    if let Some(ticks) = deadline_ticks
        && ticks == 0
    {
        return Err(SpecError::InvalidFieldValue {
            context: "EpistemicQuestion",
            field: "deadline_ticks",
            reason: "deadline_ticks must be > 0".to_owned(),
        });
    }
    Ok(())
}

fn validate_unit_interval(
    value: f64,
    context: &'static str,
    field: &'static str,
) -> Result<(), SpecError> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: format!("{field} ({value}) must be finite and within [0.0, 1.0]"),
        })
    } else {
        Ok(())
    }
}

fn validate_non_negative_finite(
    value: f64,
    context: &'static str,
    field: &'static str,
) -> Result<(), SpecError> {
    if !value.is_finite() || value < 0.0 {
        Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: format!("{field} ({value}) must be finite and >= 0.0"),
        })
    } else {
        Ok(())
    }
}

fn require_bool(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<bool, SpecError> {
    match map.get(field) {
        Some(JsonValue::Bool(b)) => Ok(*b),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected bool".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

fn require_f64(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<f64, SpecError> {
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::F64(v))) => Ok(*v),
        Some(JsonValue::Number(JsonNumber::I64(v))) => Ok(*v as f64),
        Some(JsonValue::Number(JsonNumber::U64(v))) => Ok(*v as f64),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected number".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

fn require_u64(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<u64, SpecError> {
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::U64(v))) => Ok(*v),
        Some(JsonValue::Number(JsonNumber::I64(v))) if *v >= 0 => Ok(*v as u64),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected non-negative integer".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

fn require_string_array(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<Vec<String>, SpecError> {
    match map.get(field) {
        Some(JsonValue::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    JsonValue::String(s) => out.push(s.clone()),
                    _ => {
                        return Err(SpecError::InvalidFieldValue {
                            context,
                            field,
                            reason: "expected array of strings".to_owned(),
                        });
                    }
                }
            }
            Ok(out)
        }
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected array".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

fn write_question_json(out: &mut String, q: &EpistemicQuestionSpec) -> Result<(), SpecError> {
    out.push('{');
    match q {
        EpistemicQuestionSpec::Choice {
            query_id,
            prompt,
            options,
            model_id,
            deadline_ticks,
        } => {
            if let Some(d) = deadline_ticks {
                out.push_str(&format!("\"deadline_ticks\":{d},"));
            }
            out.push_str("\"kind\":\"choice\",\"model_id\":");
            write_json_string(out, model_id);
            out.push_str(",\"options\":[");
            for (idx, opt) in options.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json_string(out, opt);
            }
            out.push_str("],\"prompt\":");
            write_json_string(out, prompt);
            out.push_str(",\"query_id\":");
            write_json_string(out, query_id);
        }
        EpistemicQuestionSpec::Noul {
            query_id,
            prompt,
            hypothesis_id,
            model_id,
            deadline_ticks,
        } => {
            if let Some(d) = deadline_ticks {
                out.push_str(&format!("\"deadline_ticks\":{d},"));
            }
            out.push_str("\"hypothesis_id\":");
            write_json_string(out, hypothesis_id);
            out.push_str(",\"kind\":\"noul\",\"model_id\":");
            write_json_string(out, model_id);
            out.push_str(",\"prompt\":");
            write_json_string(out, prompt);
            out.push_str(",\"query_id\":");
            write_json_string(out, query_id);
        }
        EpistemicQuestionSpec::Score {
            query_id,
            prompt,
            options,
            min_score,
            max_score,
            model_id,
            deadline_ticks,
        } => {
            if let Some(d) = deadline_ticks {
                out.push_str(&format!("\"deadline_ticks\":{d},"));
            }
            out.push_str("\"kind\":\"score\",\"max_score\":");
            write_canonical_f64(out, *max_score)?;
            out.push_str(",\"min_score\":");
            write_canonical_f64(out, *min_score)?;
            out.push_str(",\"model_id\":");
            write_json_string(out, model_id);
            out.push_str(",\"options\":[");
            for (idx, opt) in options.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_json_string(out, opt);
            }
            out.push_str("],\"prompt\":");
            write_json_string(out, prompt);
            out.push_str(",\"query_id\":");
            write_json_string(out, query_id);
        }
    }
    out.push('}');
    Ok(())
}

fn parse_question_json(value: &JsonValue) -> Result<EpistemicQuestionSpec, SpecError> {
    let map = as_object(value, "EpistemicQuestion")?;
    let kind = require_string(map, "kind", "EpistemicQuestion")?;
    let deadline_ticks = match map.get("deadline_ticks") {
        Some(_) => Some(require_u64(map, "deadline_ticks", "EpistemicQuestion")?),
        None => None,
    };
    let model_id = require_string(map, "model_id", "EpistemicQuestion")?.to_owned();
    let prompt = require_string(map, "prompt", "EpistemicQuestion")?.to_owned();
    let query_id = require_string(map, "query_id", "EpistemicQuestion")?.to_owned();

    match kind {
        "choice" => {
            reject_unknown_fields(
                map,
                &[
                    "deadline_ticks",
                    "kind",
                    "model_id",
                    "options",
                    "prompt",
                    "query_id",
                ],
                "EpistemicQuestion.Choice",
                UnknownFieldPolicy::Reject,
            )?;
            let options = require_string_array(map, "options", "EpistemicQuestion.Choice")?;
            Ok(EpistemicQuestionSpec::Choice {
                query_id,
                prompt,
                options,
                model_id,
                deadline_ticks,
            })
        }
        "noul" => {
            reject_unknown_fields(
                map,
                &[
                    "deadline_ticks",
                    "hypothesis_id",
                    "kind",
                    "model_id",
                    "prompt",
                    "query_id",
                ],
                "EpistemicQuestion.Noul",
                UnknownFieldPolicy::Reject,
            )?;
            let hypothesis_id =
                require_string(map, "hypothesis_id", "EpistemicQuestion.Noul")?.to_owned();
            Ok(EpistemicQuestionSpec::Noul {
                query_id,
                prompt,
                hypothesis_id,
                model_id,
                deadline_ticks,
            })
        }
        "score" => {
            reject_unknown_fields(
                map,
                &[
                    "deadline_ticks",
                    "kind",
                    "max_score",
                    "min_score",
                    "model_id",
                    "options",
                    "prompt",
                    "query_id",
                ],
                "EpistemicQuestion.Score",
                UnknownFieldPolicy::Reject,
            )?;
            let options = require_string_array(map, "options", "EpistemicQuestion.Score")?;
            let min_score = require_f64(map, "min_score", "EpistemicQuestion.Score")?;
            let max_score = require_f64(map, "max_score", "EpistemicQuestion.Score")?;
            Ok(EpistemicQuestionSpec::Score {
                query_id,
                prompt,
                options,
                min_score,
                max_score,
                model_id,
                deadline_ticks,
            })
        }
        other => Err(SpecError::InvalidFieldValue {
            context: "EpistemicQuestion",
            field: "kind",
            reason: format!("unsupported question kind '{other}'"),
        }),
    }
}
