//! Serialización JSON canónica y deserialización estricta con política de campos desconocidos.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use gogamen_core::{CausalKind, NodeId, PortDirection, PortId, PortSpec, PortType};

use crate::error::{SpecError, UnknownFieldPolicy};
use crate::json::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};
use crate::model::{
    AGENT_SPEC_SCHEMA_VERSION, AgentSpec, EXPERIMENT_SPEC_SCHEMA_VERSION, ExperimentSpec,
    SYSTEM_SPEC_SCHEMA_VERSION, SpecConfigMap, SpecConfigValue, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec,
};

impl SystemSpec {
    /// Serializa la especificación a JSON canónico determinista tras validar el contrato.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_system_spec(&mut out, self)?;
        Ok(out)
    }

    /// Deserializa y valida estrictamente un [`SystemSpec`] rechazando campos desconocidos.
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa y valida un [`SystemSpec`] aplicando la política indicada para campos desconocidos.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let spec = parse_system_spec(&value, policy)?;
        spec.validate()?;
        Ok(spec)
    }
}

impl AgentSpec {
    /// Serializa el agente a JSON canónico determinista tras validar el contrato.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_agent_spec(&mut out, self)?;
        Ok(out)
    }

    /// Deserializa y valida estrictamente un [`AgentSpec`] rechazando campos desconocidos.
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa y valida un [`AgentSpec`] aplicando la política indicada para campos desconocidos.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let spec = parse_agent_spec(&value, policy)?;
        spec.validate()?;
        Ok(spec)
    }
}

impl ExperimentSpec {
    /// Serializa el experimento a JSON canónico determinista tras validar el contrato.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_experiment_spec(&mut out, self)?;
        Ok(out)
    }

    /// Deserializa y valida estrictamente un [`ExperimentSpec`] rechazando campos desconocidos.
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa y valida un [`ExperimentSpec`] aplicando la política indicada para campos desconocidos.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let spec = parse_experiment_spec(&value, policy)?;
        spec.validate()?;
        Ok(spec)
    }
}

/// Construye el fixture canónico portable de dos nodos y una conexión tipada.
pub fn canonical_two_node_fixture() -> SystemSpec {
    let sensor = SystemNodeSpec::new("sensor", "observer")
        .with_output("observation", PortType::F64)
        .with_config("rate_hz", SpecConfigValue::I64(50))
        .with_config("normalized", SpecConfigValue::Bool(true));

    let controller = SystemNodeSpec::new("controller", "policy")
        .with_input("observation", PortType::F64)
        .with_output("action", PortType::I64)
        .with_config("gain", SpecConfigValue::F64(0.5))
        .with_config("mode", SpecConfigValue::String("greedy".to_owned()));

    let link = SystemConnectionSpec::instant("sensor", "observation", "controller", "observation");

    SystemSpec::new("two_node_canonical_v0")
        .with_node(sensor)
        .with_node(controller)
        .with_connection(link)
        .with_config("dt_secs", SpecConfigValue::F64(0.02))
        .with_config("deterministic", SpecConfigValue::Bool(true))
}

/// Exporta el fixture canónico de dos nodos y una conexión en formato JSON canónico.
pub fn canonical_two_node_fixture_json() -> String {
    canonical_two_node_fixture()
        .to_canonical_json()
        .expect("canonical two-node fixture must serialize cleanly")
}

/// Construye el fixture canónico portable de agente cognitivo (`sensor` -> `belief` con `ExprAst` -> `planner`).
pub fn canonical_cognitive_agent_fixture() -> SystemSpec {
    let explore_expr = crate::expr::ExprAst::gt(
        crate::expr::ExprAst::mul(
            crate::expr::ExprAst::var("energy").expect("valid var"),
            crate::expr::ExprAst::var("curiosity").expect("valid var"),
        ),
        crate::expr::ExprAst::var("threshold").expect("valid var"),
    );
    let expr_json = explore_expr
        .to_canonical_json()
        .expect("explore_expr must serialize cleanly");

    let sensor = SystemNodeSpec::new("sensor", "agent::sensor")
        .with_output("observation", PortType::F64)
        .with_output("energy", PortType::F64)
        .with_config("base_energy", SpecConfigValue::F64(1.0))
        .with_config("noise_scale", SpecConfigValue::F64(0.1));

    let belief = SystemNodeSpec::new("belief", "agent::belief")
        .with_input("observation", PortType::F64)
        .with_input("energy", PortType::F64)
        .with_output("belief_score", PortType::F64)
        .with_output("should_explore", PortType::Bool)
        .with_config("curiosity", SpecConfigValue::F64(0.6))
        .with_config("expr_json", SpecConfigValue::String(expr_json))
        .with_config("threshold", SpecConfigValue::F64(0.4));

    let planner = SystemNodeSpec::new("planner", "agent::planner")
        .with_input("belief_score", PortType::F64)
        .with_input("should_explore", PortType::Bool)
        .with_output("action", PortType::I64)
        .with_config("exploit_gain", SpecConfigValue::F64(1.0))
        .with_config("explore_gain", SpecConfigValue::F64(2.0));

    SystemSpec::new("cognitive_agent_canonical_v0")
        .with_node(sensor)
        .with_node(belief)
        .with_node(planner)
        .with_connection(SystemConnectionSpec::instant(
            "sensor",
            "observation",
            "belief",
            "observation",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "sensor", "energy", "belief", "energy",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief",
            "belief_score",
            "planner",
            "belief_score",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief",
            "should_explore",
            "planner",
            "should_explore",
        ))
        .with_config("dt_secs", SpecConfigValue::F64(0.02))
        .with_config("deterministic", SpecConfigValue::Bool(true))
}

/// Exporta el fixture canónico de agente cognitivo en formato JSON canónico.
pub fn canonical_cognitive_agent_fixture_json() -> String {
    canonical_cognitive_agent_fixture()
        .to_canonical_json()
        .expect("canonical cognitive agent fixture must serialize cleanly")
}

/// Construye el fixture canónico portable de Language World (`human_channel` + `direct_sensor` -> `grounder` -> `belief` -> `guarded_planner`).
pub fn canonical_language_world_fixture() -> SystemSpec {
    use crate::epistemic::{
        AbstentionPolicySpec, AskActionSpec, EpistemicQuestionSpec, GuardPolicySpec,
        InspectActionSpec, ProvenanceSpec,
    };

    let human_provenance = ProvenanceSpec {
        source_id: "human_channel".to_owned(),
        sensor_id: 100,
        evidence_ref: 101,
        epistemic_status: "reported".to_owned(),
    }
    .to_canonical_json()
    .expect("human provenance must serialize");

    let direct_provenance = ProvenanceSpec {
        source_id: "direct_sensor".to_owned(),
        sensor_id: 200,
        evidence_ref: 500,
        epistemic_status: "observed".to_owned(),
    }
    .to_canonical_json()
    .expect("direct provenance must serialize");

    let grounding_question = EpistemicQuestionSpec::Choice {
        query_id: "lw-entity-notice-1".to_owned(),
        prompt: "Which registered cell does the human notice refer to?".to_owned(),
        options: vec!["cell_2_1".to_owned(), "cell_1_0".to_owned()],
        model_id: "offline_replay_v1".to_owned(),
        deadline_ticks: Some(5),
    };
    let grounding_question_json = grounding_question
        .to_canonical_json()
        .expect("grounding question must serialize");

    let abstention_json = AbstentionPolicySpec {
        min_confidence: 0.65,
        fallback_action: "abstain".to_owned(),
        abstain_on_timeout: true,
        abstain_on_ambiguity: true,
        ambiguity_margin: 0.05,
    }
    .to_canonical_json()
    .expect("abstention policy must serialize");

    let ask_json = AskActionSpec {
        recipient: "human_operator".to_owned(),
        scope: "language_world".to_owned(),
        question: EpistemicQuestionSpec::Noul {
            query_id: "lw-ask-obstacle-1".to_owned(),
            prompt: "Is cell (2,1) blocked by an obstacle?".to_owned(),
            hypothesis_id: "obstacle_at_2_1".to_owned(),
            model_id: "system_one_noul_v1".to_owned(),
            deadline_ticks: Some(3),
        },
        comm_cost: 0.5,
        step_cost: 0.1,
        response_probability: 0.85,
        timeout_ticks: 3,
    }
    .to_canonical_json()
    .expect("ask action must serialize");

    let inspect_json = InspectActionSpec {
        sensor_id: "direct_sensor".to_owned(),
        inspect_cost: 0.25,
        target_entity: Some("cell_2_1".to_owned()),
    }
    .to_canonical_json()
    .expect("inspect action must serialize");

    let guard_json = GuardPolicySpec {
        guard_policy: "reject_only".to_owned(),
        allowed_actions: vec![
            "stay".to_owned(),
            "right".to_owned(),
            "down".to_owned(),
            "left".to_owned(),
        ],
        immutable_permissions: true,
        human_messages_cannot_mutate_permissions: true,
    }
    .to_canonical_json()
    .expect("guard policy must serialize");

    let human_channel = SystemNodeSpec::new("human_channel", "language::human_channel")
        .with_output("observation", PortType::F64)
        .with_output("message_confidence", PortType::F64)
        .with_output("is_order", PortType::Bool)
        .with_config("evidence_ref", SpecConfigValue::I64(101))
        .with_config("provenance_json", SpecConfigValue::String(human_provenance))
        .with_config("sensor_id", SpecConfigValue::I64(100))
        .with_config(
            "source_id",
            SpecConfigValue::String("human_channel".to_owned()),
        )
        .with_config("speech_act", SpecConfigValue::String("notice".to_owned()));

    let direct_sensor = SystemNodeSpec::new("direct_sensor", "language::direct_sensor")
        .with_output("direct_obs", PortType::F64)
        .with_config("evidence_ref", SpecConfigValue::I64(500))
        .with_config(
            "provenance_json",
            SpecConfigValue::String(direct_provenance),
        )
        .with_config("sensor_id", SpecConfigValue::I64(200))
        .with_config(
            "source_id",
            SpecConfigValue::String("direct_sensor".to_owned()),
        );

    let grounder = SystemNodeSpec::new("grounder", "language::grounder")
        .with_input("observation", PortType::F64)
        .with_input("message_confidence", PortType::F64)
        .with_output("grounded_prob", PortType::F64)
        .with_output("abstained", PortType::Bool)
        .with_config("abstention_json", SpecConfigValue::String(abstention_json))
        .with_config(
            "model_id",
            SpecConfigValue::String("offline_replay_v1".to_owned()),
        )
        .with_config(
            "question_json",
            SpecConfigValue::String(grounding_question_json),
        )
        .with_config(
            "source_id",
            SpecConfigValue::String("human_channel".to_owned()),
        );

    let belief = SystemNodeSpec::new("belief", "language::belief")
        .with_input("grounded_prob", PortType::F64)
        .with_input("abstained", PortType::Bool)
        .with_input("direct_obs", PortType::F64)
        .with_output("belief_prob", PortType::F64)
        .with_output("should_ask", PortType::Bool)
        .with_output("should_inspect", PortType::Bool)
        .with_config("ask_json", SpecConfigValue::String(ask_json))
        .with_config("inspect_json", SpecConfigValue::String(inspect_json))
        .with_config("prior_obstacle_prob", SpecConfigValue::F64(0.5));

    let guarded_planner = SystemNodeSpec::new("guarded_planner", "language::guarded_planner")
        .with_input("belief_prob", PortType::F64)
        .with_input("should_ask", PortType::Bool)
        .with_input("should_inspect", PortType::Bool)
        .with_input("is_order", PortType::Bool)
        .with_output("action", PortType::I64)
        .with_output("guard_rejected", PortType::Bool)
        .with_config("guard_json", SpecConfigValue::String(guard_json))
        .with_config(
            "guard_policy",
            SpecConfigValue::String("reject_only".to_owned()),
        )
        .with_config("human_can_mutate_permissions", SpecConfigValue::Bool(false))
        .with_config("immutable_permissions", SpecConfigValue::Bool(true));

    SystemSpec::new("language_world_canonical_v0")
        .with_node(human_channel)
        .with_node(direct_sensor)
        .with_node(grounder)
        .with_node(belief)
        .with_node(guarded_planner)
        .with_connection(SystemConnectionSpec::instant(
            "human_channel",
            "observation",
            "grounder",
            "observation",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "human_channel",
            "message_confidence",
            "grounder",
            "message_confidence",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "grounder",
            "grounded_prob",
            "belief",
            "grounded_prob",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "grounder",
            "abstained",
            "belief",
            "abstained",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "direct_sensor",
            "direct_obs",
            "belief",
            "direct_obs",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief",
            "belief_prob",
            "guarded_planner",
            "belief_prob",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief",
            "should_ask",
            "guarded_planner",
            "should_ask",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief",
            "should_inspect",
            "guarded_planner",
            "should_inspect",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "human_channel",
            "is_order",
            "guarded_planner",
            "is_order",
        ))
        .with_config("deterministic", SpecConfigValue::Bool(true))
        .with_config("dt_secs", SpecConfigValue::F64(0.02))
}

/// Exporta el fixture canónico de Language World en formato JSON canónico.
pub fn canonical_language_world_fixture_json() -> String {
    canonical_language_world_fixture()
        .to_canonical_json()
        .expect("canonical language world fixture must serialize cleanly")
}

// ---------------------------------------------------------------------------
// Serialización canónica
// ---------------------------------------------------------------------------

pub(crate) fn write_system_spec(out: &mut String, spec: &SystemSpec) -> Result<(), SpecError> {
    out.push('{');
    out.push_str("\"config\":");
    write_config_map(out, &spec.config)?;
    out.push_str(",\"connections\":[");
    for (idx, conn) in spec.connections.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_connection_spec(out, conn);
    }
    out.push_str("],\"nodes\":[");
    for (idx, node) in spec.nodes.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_node_spec(out, node)?;
    }
    out.push_str("],\"schema_version\":");
    write_json_string(out, &spec.schema_version);
    out.push_str(",\"spec_id\":");
    write_json_string(out, &spec.spec_id);
    out.push('}');
    Ok(())
}

fn write_node_spec(out: &mut String, node: &SystemNodeSpec) -> Result<(), SpecError> {
    out.push('{');
    out.push_str("\"config\":");
    write_config_map(out, &node.config)?;
    out.push_str(",\"id\":");
    write_json_string(out, node.id.as_str());
    out.push_str(",\"kind\":");
    write_json_string(out, &node.kind);
    out.push_str(",\"ports\":[");
    for (idx, port) in node.ports.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_port_spec(out, port);
    }
    out.push_str("]}");
    Ok(())
}

fn write_port_spec(out: &mut String, port: &PortSpec) {
    out.push('{');
    out.push_str("\"direction\":");
    let dir = match port.direction {
        PortDirection::Input => "input",
        PortDirection::Output => "output",
    };
    write_json_string(out, dir);
    out.push_str(",\"id\":");
    write_json_string(out, port.id.as_str());
    out.push_str(",\"port_type\":");
    write_port_type(out, &port.port_type);
    out.push('}');
}

pub(crate) fn write_port_type(out: &mut String, port_type: &PortType) {
    match port_type {
        PortType::Bool => write_json_string(out, "bool"),
        PortType::I64 => write_json_string(out, "i64"),
        PortType::U64 => write_json_string(out, "u64"),
        PortType::F64 => write_json_string(out, "f64"),
        PortType::String => write_json_string(out, "string"),
        PortType::Unit => write_json_string(out, "unit"),
        PortType::Custom(name) => {
            out.push_str("{\"custom\":");
            write_json_string(out, name);
            out.push('}');
        }
    }
}

fn write_connection_spec(out: &mut String, conn: &SystemConnectionSpec) {
    out.push('{');
    let _ = write!(out, "\"delay_ticks\":{}", conn.causal.delay_ticks());
    out.push_str(",\"from_node\":");
    write_json_string(out, conn.from_node.as_str());
    out.push_str(",\"from_port\":");
    write_json_string(out, conn.from_port.as_str());
    out.push_str(",\"to_node\":");
    write_json_string(out, conn.to_node.as_str());
    out.push_str(",\"to_port\":");
    write_json_string(out, conn.to_port.as_str());
    out.push('}');
}

fn write_agent_spec(out: &mut String, agent: &AgentSpec) -> Result<(), SpecError> {
    out.push('{');
    out.push_str("\"agent_id\":");
    write_json_string(out, &agent.agent_id);
    out.push_str(",\"config\":");
    write_config_map(out, &agent.config)?;
    out.push_str(",\"schema_version\":");
    write_json_string(out, &agent.schema_version);
    out.push_str(",\"system\":");
    write_system_spec(out, &agent.system)?;
    out.push('}');
    Ok(())
}

fn write_experiment_spec(out: &mut String, exp: &ExperimentSpec) -> Result<(), SpecError> {
    out.push('{');
    out.push_str("\"agents\":[");
    for (idx, agent) in exp.agents.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_agent_spec(out, agent)?;
    }
    out.push_str("],\"config\":");
    write_config_map(out, &exp.config)?;
    out.push_str(",\"experiment_id\":");
    write_json_string(out, &exp.experiment_id);
    out.push_str(",\"schema_version\":");
    write_json_string(out, &exp.schema_version);
    let _ = write!(out, ",\"seed\":{}", exp.seed);
    out.push_str(",\"system\":");
    write_system_spec(out, &exp.system)?;
    out.push('}');
    Ok(())
}

pub(crate) fn write_config_map(out: &mut String, config: &SpecConfigMap) -> Result<(), SpecError> {
    out.push('{');
    for (idx, (key, value)) in config.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_json_string(out, key);
        out.push(':');
        match value {
            SpecConfigValue::Bool(b) => {
                out.push_str(if *b { "true" } else { "false" });
            }
            SpecConfigValue::I64(n) => {
                let _ = write!(out, "{n}");
            }
            SpecConfigValue::F64(v) => {
                write_canonical_f64(out, *v)?;
            }
            SpecConfigValue::String(s) => {
                write_json_string(out, s);
            }
        }
    }
    out.push('}');
    Ok(())
}

// ---------------------------------------------------------------------------
// Deserialización estricta
// ---------------------------------------------------------------------------

pub(crate) fn as_object<'a>(
    value: &'a JsonValue,
    context: &'static str,
) -> Result<&'a BTreeMap<String, JsonValue>, SpecError> {
    match value {
        JsonValue::Object(map) => Ok(map),
        _ => Err(SpecError::InvalidFieldValue {
            context,
            field: "<root>",
            reason: "expected JSON object".to_owned(),
        }),
    }
}

pub(crate) fn reject_unknown_fields(
    map: &BTreeMap<String, JsonValue>,
    known: &[&str],
    context: &'static str,
    policy: UnknownFieldPolicy,
) -> Result<(), SpecError> {
    if policy == UnknownFieldPolicy::AllowIgnored {
        return Ok(());
    }
    for key in map.keys() {
        if !known.contains(&key.as_str()) {
            return Err(SpecError::UnknownField {
                context,
                field: key.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn require_string<'a>(
    map: &'a BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<&'a str, SpecError> {
    match map.get(field) {
        Some(JsonValue::String(s)) => Ok(s.as_str()),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected string".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

pub(crate) fn require_schema_version(
    map: &BTreeMap<String, JsonValue>,
    expected: &'static str,
    context: &'static str,
) -> Result<String, SpecError> {
    let found = match map.get("schema_version") {
        Some(JsonValue::String(s)) => s.clone(),
        Some(_) => {
            return Err(SpecError::InvalidFieldValue {
                context,
                field: "schema_version",
                reason: "expected string".to_owned(),
            });
        }
        None => {
            return Err(SpecError::MissingField {
                context,
                field: "schema_version",
            });
        }
    };
    if found != expected {
        return Err(SpecError::IncompatibleSchemaVersion { expected, found });
    }
    Ok(found)
}

pub(crate) fn require_array<'a>(
    map: &'a BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<&'a Vec<JsonValue>, SpecError> {
    match map.get(field) {
        Some(JsonValue::Array(items)) => Ok(items),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected array".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

pub(crate) fn require_u64(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<u64, SpecError> {
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::I64(n))) if *n >= 0 => Ok(*n as u64),
        Some(JsonValue::Number(JsonNumber::U64(n))) => Ok(*n),
        Some(_) => Err(SpecError::InvalidFieldValue {
            context,
            field,
            reason: "expected non-negative u64 integer".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

pub(crate) fn parse_optional_config(
    map: &BTreeMap<String, JsonValue>,
    context: &'static str,
) -> Result<SpecConfigMap, SpecError> {
    let Some(raw) = map.get("config") else {
        return Ok(SpecConfigMap::new());
    };
    let JsonValue::Object(entries) = raw else {
        return Err(SpecError::InvalidFieldValue {
            context,
            field: "config",
            reason: "expected object".to_owned(),
        });
    };
    let mut config = SpecConfigMap::new();
    for (key, val) in entries {
        let parsed = match val {
            JsonValue::Bool(b) => SpecConfigValue::Bool(*b),
            JsonValue::Number(JsonNumber::I64(n)) => SpecConfigValue::I64(*n),
            JsonValue::Number(JsonNumber::F64(v)) => SpecConfigValue::f64(*v)?,
            JsonValue::Number(JsonNumber::U64(n)) => {
                return Err(SpecError::InvalidFieldValue {
                    context,
                    field: "config",
                    reason: format!("integer '{n}' for key '{key}' exceeds i64::MAX"),
                });
            }
            JsonValue::String(s) => SpecConfigValue::String(s.clone()),
            _ => {
                return Err(SpecError::InvalidFieldValue {
                    context,
                    field: "config",
                    reason: format!(
                        "value for key '{key}' must be scalar (bool, i64, f64, or string)"
                    ),
                });
            }
        };
        config.insert(key.clone(), parsed);
    }
    Ok(config)
}

pub(crate) fn parse_system_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<SystemSpec, SpecError> {
    const CTX: &str = "SystemSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "schema_version",
            "spec_id",
            "nodes",
            "connections",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_schema_version(map, SYSTEM_SPEC_SCHEMA_VERSION, CTX)?;
    let spec_id = require_string(map, "spec_id", CTX)?.to_owned();
    let raw_nodes = require_array(map, "nodes", CTX)?;
    let raw_conns = require_array(map, "connections", CTX)?;
    let config = parse_optional_config(map, CTX)?;

    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for raw_node in raw_nodes {
        nodes.push(parse_node_spec(raw_node, policy)?);
    }

    let mut connections = Vec::with_capacity(raw_conns.len());
    for raw_conn in raw_conns {
        connections.push(parse_connection_spec(raw_conn, policy)?);
    }

    Ok(SystemSpec {
        schema_version,
        spec_id,
        nodes,
        connections,
        config,
    })
}

fn parse_node_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<SystemNodeSpec, SpecError> {
    const CTX: &str = "SystemNodeSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["id", "kind", "ports", "config"], CTX, policy)?;

    let id = NodeId::new(require_string(map, "id", CTX)?);
    let kind = require_string(map, "kind", CTX)?.to_owned();
    let raw_ports = require_array(map, "ports", CTX)?;
    let config = parse_optional_config(map, CTX)?;

    let mut ports = Vec::with_capacity(raw_ports.len());
    for raw_port in raw_ports {
        ports.push(parse_port_spec(raw_port, policy)?);
    }

    Ok(SystemNodeSpec {
        id,
        kind,
        ports,
        config,
    })
}

fn parse_port_spec(value: &JsonValue, policy: UnknownFieldPolicy) -> Result<PortSpec, SpecError> {
    const CTX: &str = "PortSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["id", "direction", "port_type"], CTX, policy)?;

    let id = PortId::new(require_string(map, "id", CTX)?);
    let direction = match require_string(map, "direction", CTX)? {
        "input" => PortDirection::Input,
        "output" => PortDirection::Output,
        other => {
            return Err(SpecError::InvalidFieldValue {
                context: CTX,
                field: "direction",
                reason: format!("expected 'input' or 'output', found '{other}'"),
            });
        }
    };

    let raw_type = map.get("port_type").ok_or(SpecError::MissingField {
        context: CTX,
        field: "port_type",
    })?;
    let port_type = parse_port_type(raw_type, policy)?;

    Ok(PortSpec {
        id,
        direction,
        port_type,
    })
}

pub(crate) fn parse_port_type(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<PortType, SpecError> {
    const CTX: &str = "PortType";
    match value {
        JsonValue::String(s) => match s.as_str() {
            "bool" => Ok(PortType::Bool),
            "i64" => Ok(PortType::I64),
            "u64" => Ok(PortType::U64),
            "f64" => Ok(PortType::F64),
            "string" => Ok(PortType::String),
            "unit" => Ok(PortType::Unit),
            other => Err(SpecError::InvalidFieldValue {
                context: CTX,
                field: "port_type",
                reason: format!("unknown primitive port type '{other}'"),
            }),
        },
        JsonValue::Object(map) => {
            reject_unknown_fields(map, &["custom"], CTX, policy)?;
            let name = require_string(map, "custom", CTX)?;
            Ok(PortType::custom(name))
        }
        _ => Err(SpecError::InvalidFieldValue {
            context: CTX,
            field: "port_type",
            reason: "expected primitive type string or {\"custom\": \"...\"}".to_owned(),
        }),
    }
}

fn parse_connection_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<SystemConnectionSpec, SpecError> {
    const CTX: &str = "SystemConnectionSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "from_node",
            "from_port",
            "to_node",
            "to_port",
            "delay_ticks",
        ],
        CTX,
        policy,
    )?;

    let from_node = NodeId::new(require_string(map, "from_node", CTX)?);
    let from_port = PortId::new(require_string(map, "from_port", CTX)?);
    let to_node = NodeId::new(require_string(map, "to_node", CTX)?);
    let to_port = PortId::new(require_string(map, "to_port", CTX)?);
    let delay_ticks = require_u64(map, "delay_ticks", CTX)?;

    Ok(SystemConnectionSpec {
        from_node,
        from_port,
        to_node,
        to_port,
        causal: CausalKind::from_delay_ticks(delay_ticks),
    })
}

fn parse_agent_spec(value: &JsonValue, policy: UnknownFieldPolicy) -> Result<AgentSpec, SpecError> {
    const CTX: &str = "AgentSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &["schema_version", "agent_id", "system", "config"],
        CTX,
        policy,
    )?;

    let schema_version = require_schema_version(map, AGENT_SPEC_SCHEMA_VERSION, CTX)?;
    let agent_id = require_string(map, "agent_id", CTX)?.to_owned();
    let raw_system = map.get("system").ok_or(SpecError::MissingField {
        context: CTX,
        field: "system",
    })?;
    let system = parse_system_spec(raw_system, policy)?;
    let config = parse_optional_config(map, CTX)?;

    Ok(AgentSpec {
        schema_version,
        agent_id,
        system,
        config,
    })
}

pub(crate) fn parse_experiment_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<ExperimentSpec, SpecError> {
    const CTX: &str = "ExperimentSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "schema_version",
            "experiment_id",
            "seed",
            "system",
            "agents",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_schema_version(map, EXPERIMENT_SPEC_SCHEMA_VERSION, CTX)?;
    let experiment_id = require_string(map, "experiment_id", CTX)?.to_owned();
    let seed = require_u64(map, "seed", CTX)?;
    let raw_system = map.get("system").ok_or(SpecError::MissingField {
        context: CTX,
        field: "system",
    })?;
    let system = parse_system_spec(raw_system, policy)?;
    let raw_agents = require_array(map, "agents", CTX)?;
    let mut agents = Vec::with_capacity(raw_agents.len());
    for raw_agent in raw_agents {
        agents.push(parse_agent_spec(raw_agent, policy)?);
    }
    let config = parse_optional_config(map, CTX)?;

    Ok(ExperimentSpec {
        schema_version,
        experiment_id,
        seed,
        system,
        agents,
        config,
    })
}
