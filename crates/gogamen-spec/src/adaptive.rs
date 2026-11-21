//! Especificación adaptativa (`AdaptiveAgentSpec`) y `AdaptiveExperimentSpec` tipados.
//!
//! Este módulo amplía el contrato declarativo `std`-only de `gogamen-spec` componiendo
//! [`SystemSpec`] directamente (sin segunda IR), cubriendo mundo, sensores, modelo,
//! objetivos tipados ([`ExprAst`]), restricciones, presupuesto, semillas disjuntas,
//! catálogo explícito de acciones ([`CapabilityCatalog`] / [`ActionRegistry`]) y
//! migración versionada desde `gogamen.experiment_spec.v0` hacia `gogamen.experiment_spec.v0.1`.
//!
//! # Migración de versiones (`v0` -> `v0.1`)
//! La función [`migrate_experiment_spec_json`] soporta:
//! - `gogamen.experiment_spec.v0.1`: valida directamente como [`AdaptiveExperimentSpec`].
//! - `gogamen.experiment_spec.v0` (legacy [`ExperimentSpec`](crate::ExperimentSpec)):
//!   migra determinísticamente a [`AdaptiveExperimentSpec`] derivando [`WorldSpecRef`],
//!   [`BudgetSpec`] acotado por defecto, [`SeedSpec`] disjunto (`dev=[seed]`,
//!   `holdout=[seed.wrapping_add(1)]`) y [`ModelSpecRef`] por agente, registrando cada
//!   paso y diagnóstico en [`SchemaMigrationReport`].
//! - Cualquier otra versión o estructura inválida devuelve [`SpecMigrationError`] con
//!   [`SpecDiagnostic`] (ruta de campo y versión esperada vs recibida).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::{NodeId, PortId, PortType};

use crate::codec::{
    as_object, parse_experiment_spec, parse_optional_config, parse_port_type, parse_system_spec,
    reject_unknown_fields, require_array, require_schema_version, require_string, require_u64,
    write_config_map, write_port_type, write_system_spec,
};
use crate::error::{SpecDiagnostic, SpecError, SpecMigrationError, UnknownFieldPolicy};
use crate::expr::{ExprAst, parse_expr_json, write_expr_json};
use crate::json::{JsonNumber, JsonValue, parse_json, write_json_string};
use crate::model::{
    EXPERIMENT_SPEC_SCHEMA_VERSION, SpecConfigMap, SpecConfigValue, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec, validate_config_map, validate_non_empty,
};

/// Versión de esquema canónica para [`AdaptiveAgentSpec`].
pub const ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION: &str = "gogamen.adaptive_agent_spec.v0.1";
/// Versión de esquema canónica para [`AdaptiveExperimentSpec`].
pub const ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION: &str = "gogamen.experiment_spec.v0.1";

const FORBIDDEN_EVAL_KEYS: &[&str] = &[
    "eval",
    "exec",
    "script",
    "code",
    "inline_code",
    "lambda",
    "eval_code",
    "dsl_code",
];

const FORBIDDEN_EVAL_MARKERS: &[&str] = &[
    "eval(",
    "exec(",
    "javascript:",
    "python:",
    "kotlin:",
    "lambda ",
    "${",
];

/// Referencia declarativa al mundo/entorno de un experimento adaptativo.
#[derive(Debug, Clone, PartialEq)]
pub struct WorldSpecRef {
    /// Identificador único del mundo.
    pub world_id: String,
    /// Tipo de mundo (por ejemplo `"grid"`, `"grid_pomdp"`).
    pub kind: String,
    /// Parámetros escalares del mundo.
    pub config: SpecConfigMap,
}

impl WorldSpecRef {
    /// Crea una referencia declarativa de mundo.
    pub fn new(world_id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            world_id: world_id.into(),
            kind: kind.into(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar de configuración al mundo.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida identificadores, configuración y ausencia de código arbitrario.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.world_id, "WorldSpecRef.world_id")?;
        validate_non_empty(&self.kind, "WorldSpecRef.kind")?;
        reject_code_eval_str(&self.world_id, "WorldSpecRef", "world_id")?;
        reject_code_eval_str(&self.kind, "WorldSpecRef", "kind")?;
        validate_config_map(&self.config, "WorldSpecRef.config")?;
        reject_code_eval_in_config(&self.config, "WorldSpecRef.config")?;
        Ok(())
    }
}

/// Enlace declarativo de un sensor sobre un puerto tipado de [`SystemSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct SensorBindingSpec {
    /// Identificador único del sensor en el agente.
    pub sensor_id: String,
    /// Tipo o modalidad del sensor (por ejemplo `"position"`, `"partial_fov"`).
    pub kind: String,
    /// Nodo origen en [`SystemSpec`].
    pub source_node: NodeId,
    /// Puerto origen en el nodo.
    pub source_port: PortId,
    /// Tipo de dato transportado por el sensor.
    pub port_type: PortType,
    /// Parámetros escalares del sensor.
    pub config: SpecConfigMap,
}

impl SensorBindingSpec {
    /// Crea un enlace de sensor tipado.
    pub fn new(
        sensor_id: impl Into<String>,
        kind: impl Into<String>,
        source_node: impl Into<NodeId>,
        source_port: impl Into<PortId>,
        port_type: PortType,
    ) -> Self {
        Self {
            sensor_id: sensor_id.into(),
            kind: kind.into(),
            source_node: source_node.into(),
            source_port: source_port.into(),
            port_type,
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar al sensor.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida el enlace del sensor y su configuración.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.sensor_id, "SensorBindingSpec.sensor_id")?;
        validate_non_empty(&self.kind, "SensorBindingSpec.kind")?;
        validate_non_empty(self.source_node.as_str(), "SensorBindingSpec.source_node")?;
        validate_non_empty(self.source_port.as_str(), "SensorBindingSpec.source_port")?;
        if let PortType::Custom(name) = &self.port_type {
            validate_non_empty(name, "SensorBindingSpec.port_type")?;
        }
        reject_code_eval_str(&self.sensor_id, "SensorBindingSpec", "sensor_id")?;
        reject_code_eval_str(&self.kind, "SensorBindingSpec", "kind")?;
        validate_config_map(&self.config, "SensorBindingSpec.config")?;
        reject_code_eval_in_config(&self.config, "SensorBindingSpec.config")?;
        Ok(())
    }
}

/// Referencia declarativa al modelo interno/política de un agente adaptativo.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpecRef {
    /// Identificador único del modelo.
    pub model_id: String,
    /// Familia o arquitectura declarativa (por ejemplo `"tabular_policy"`, `"belief_state"`).
    pub kind: String,
    /// Parámetros escalares del modelo.
    pub config: SpecConfigMap,
}

impl ModelSpecRef {
    /// Crea una referencia declarativa a modelo.
    pub fn new(model_id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            kind: kind.into(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar al modelo.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida la referencia del modelo y su configuración.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.model_id, "ModelSpecRef.model_id")?;
        validate_non_empty(&self.kind, "ModelSpecRef.kind")?;
        reject_code_eval_str(&self.model_id, "ModelSpecRef", "model_id")?;
        reject_code_eval_str(&self.kind, "ModelSpecRef", "kind")?;
        validate_config_map(&self.config, "ModelSpecRef.config")?;
        reject_code_eval_in_config(&self.config, "ModelSpecRef.config")?;
        Ok(())
    }
}

/// Objetivo declarativo tipado expresado mediante [`ExprAst`].
#[derive(Debug, Clone, PartialEq)]
pub struct GoalSpec {
    /// Identificador único del objetivo.
    pub goal_id: String,
    /// Expresión simbólica acotada del objetivo.
    pub expr: ExprAst,
    /// Indica si el objetivo se maximiza (`true`) o minimiza (`false`).
    pub maximize: bool,
    /// Parámetros escalares asociados al objetivo.
    pub config: SpecConfigMap,
}

impl GoalSpec {
    /// Crea un objetivo declarativo con [`ExprAst`].
    pub fn new(goal_id: impl Into<String>, expr: ExprAst, maximize: bool) -> Self {
        Self {
            goal_id: goal_id.into(),
            expr,
            maximize,
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar al objetivo.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida el identificador, AST tipado y configuración del objetivo.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.goal_id, "GoalSpec.goal_id")?;
        reject_code_eval_str(&self.goal_id, "GoalSpec", "goal_id")?;
        self.expr.validate()?;
        validate_config_map(&self.config, "GoalSpec.config")?;
        reject_code_eval_in_config(&self.config, "GoalSpec.config")?;
        Ok(())
    }
}

/// Restricción declarativa tipada expresada mediante [`ExprAst`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintSpec {
    /// Identificador único de la restricción.
    pub constraint_id: String,
    /// Condición booleana/escalar acotada.
    pub condition: ExprAst,
    /// Indica si la restricción es dura (`true`) o blanda (`false`).
    pub hard: bool,
    /// Parámetros escalares asociados a la restricción.
    pub config: SpecConfigMap,
}

impl ConstraintSpec {
    /// Crea una restricción declarativa.
    pub fn new(constraint_id: impl Into<String>, condition: ExprAst, hard: bool) -> Self {
        Self {
            constraint_id: constraint_id.into(),
            condition,
            hard,
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar a la restricción.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida la restricción y su AST tipado.
    pub fn validate(&self) -> Result<(), SpecError> {
        validate_non_empty(&self.constraint_id, "ConstraintSpec.constraint_id")?;
        reject_code_eval_str(&self.constraint_id, "ConstraintSpec", "constraint_id")?;
        self.condition.validate()?;
        validate_config_map(&self.config, "ConstraintSpec.config")?;
        reject_code_eval_in_config(&self.config, "ConstraintSpec.config")?;
        Ok(())
    }
}

/// Presupuesto de ejecución acotado para un experimento adaptativo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSpec {
    /// Máximo número de ticks permitidos.
    pub max_ticks: u64,
    /// Máximo número de transiciones permitidas.
    pub max_transitions: u64,
    /// Tiempo límite en milisegundos.
    pub deadline_ms: u64,
}

impl BudgetSpec {
    /// Crea un presupuesto de ejecución acotado.
    pub const fn new(max_ticks: u64, max_transitions: u64, deadline_ms: u64) -> Self {
        Self {
            max_ticks,
            max_transitions,
            deadline_ms,
        }
    }

    /// Valida que todos los límites del presupuesto sean estrictamente positivos.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.max_ticks == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "BudgetSpec",
                field: "max_ticks",
                reason: "must be greater than 0".to_owned(),
            });
        }
        if self.max_transitions == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "BudgetSpec",
                field: "max_transitions",
                reason: "must be greater than 0".to_owned(),
            });
        }
        if self.deadline_ms == 0 {
            return Err(SpecError::InvalidFieldValue {
                context: "BudgetSpec",
                field: "deadline_ms",
                reason: "must be greater than 0".to_owned(),
            });
        }
        Ok(())
    }
}

/// Partición determinista de semillas `dev` y `holdout` estrictamente disjuntas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeedSpec {
    /// Semillas de desarrollo/entrenamiento.
    pub dev_seeds: Vec<u64>,
    /// Semillas de evaluación holdout (disjuntas respecto a `dev_seeds`).
    pub holdout_seeds: Vec<u64>,
}

impl SeedSpec {
    /// Crea una especificación de semillas `dev` y `holdout`.
    pub fn new(dev_seeds: Vec<u64>, holdout_seeds: Vec<u64>) -> Self {
        Self {
            dev_seeds,
            holdout_seeds,
        }
    }

    /// Indica si `dev_seeds` y `holdout_seeds` son disjuntos.
    pub fn is_disjoint(&self) -> bool {
        let dev: BTreeSet<u64> = self.dev_seeds.iter().copied().collect();
        self.holdout_seeds.iter().all(|s| !dev.contains(s))
    }

    /// Valida que ambos conjuntos no estén vacíos, no tengan duplicados y sean disjuntos.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.dev_seeds.is_empty() {
            return Err(SpecError::InvalidFieldValue {
                context: "SeedSpec",
                field: "dev_seeds",
                reason: "must contain at least one seed".to_owned(),
            });
        }
        if self.holdout_seeds.is_empty() {
            return Err(SpecError::InvalidFieldValue {
                context: "SeedSpec",
                field: "holdout_seeds",
                reason: "must contain at least one seed".to_owned(),
            });
        }

        let mut dev_set = BTreeSet::new();
        for &seed in &self.dev_seeds {
            if !dev_set.insert(seed) {
                return Err(SpecError::InvalidFieldValue {
                    context: "SeedSpec",
                    field: "dev_seeds",
                    reason: format!("duplicate seed {seed}"),
                });
            }
        }

        let mut holdout_set = BTreeSet::new();
        for &seed in &self.holdout_seeds {
            if !holdout_set.insert(seed) {
                return Err(SpecError::InvalidFieldValue {
                    context: "SeedSpec",
                    field: "holdout_seeds",
                    reason: format!("duplicate seed {seed}"),
                });
            }
            if dev_set.contains(&seed) {
                return Err(SpecError::OverlappingSeeds { seed });
            }
        }
        Ok(())
    }
}

/// Catálogo explícito de acciones y capacidades registradas permitidas en una spec adaptativa.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CapabilityCatalog {
    registered_actions: BTreeSet<String>,
}

/// Alias semántico para [`CapabilityCatalog`] como registro explícito de acciones.
pub type ActionRegistry = CapabilityCatalog;

impl CapabilityCatalog {
    /// Crea un catálogo vacío sin acciones registradas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra una acción en el catálogo de forma encadenable.
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.registered_actions.insert(action.into());
        self
    }

    /// Registra una acción validando que no sea vacía ni contenga evaluación de código.
    pub fn register(&mut self, action: impl Into<String>) -> Result<(), SpecError> {
        let action = action.into();
        validate_non_empty(&action, "CapabilityCatalog.action")?;
        reject_code_eval_str(&action, "CapabilityCatalog", "action")?;
        self.registered_actions.insert(action);
        Ok(())
    }

    /// Devuelve `true` si la acción está registrada en el catálogo.
    pub fn contains(&self, action: &str) -> bool {
        self.registered_actions.contains(action)
    }

    /// Devuelve el conjunto ordenado de acciones registradas.
    pub fn actions(&self) -> &BTreeSet<String> {
        &self.registered_actions
    }

    /// Valida que una acción sea segura y esté registrada en el catálogo.
    pub fn validate_action(&self, action: &str, context: &'static str) -> Result<(), SpecError> {
        validate_non_empty(action, context)?;
        reject_code_eval_str(action, context, "actions")?;
        if !self.contains(action) {
            return Err(SpecError::UnregisteredAction {
                context,
                action: action.to_owned(),
            });
        }
        Ok(())
    }

    /// Catálogo canónico por defecto para entornos Grid y cognitivos.
    pub fn canonical_grid_defaults() -> Self {
        Self::new()
            .with_action("move_north")
            .with_action("move_south")
            .with_action("move_east")
            .with_action("move_west")
            .with_action("stay")
            .with_action("observe")
            .with_action("sense_local")
    }
}

/// Especificación declarativa de un agente adaptativo compuesta sobre [`SystemSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveAgentSpec {
    /// Versión de esquema (`ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION`).
    pub schema_version: String,
    /// Identificador único del agente.
    pub agent_id: String,
    /// Sistema causal subyacente que implementa el agente.
    pub system: SystemSpec,
    /// Sensores declarados y enlazados al sistema.
    pub sensors: Vec<SensorBindingSpec>,
    /// Referencia al modelo/política interna del agente.
    pub model: ModelSpecRef,
    /// Acciones declaradas que el agente puede emitir.
    pub actions: Vec<String>,
    /// Objetivos declarativos del agente.
    pub goals: Vec<GoalSpec>,
    /// Restricciones declarativas del agente.
    pub constraints: Vec<ConstraintSpec>,
    /// Parámetros escalares adicionales del agente.
    pub config: SpecConfigMap,
}

impl AdaptiveAgentSpec {
    /// Crea una especificación de agente adaptativo compuesta sobre [`SystemSpec`].
    pub fn new(agent_id: impl Into<String>, system: SystemSpec, model: ModelSpecRef) -> Self {
        Self {
            schema_version: ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION.to_owned(),
            agent_id: agent_id.into(),
            system,
            sensors: Vec::new(),
            model,
            actions: Vec::new(),
            goals: Vec::new(),
            constraints: Vec::new(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un sensor al agente.
    pub fn with_sensor(mut self, sensor: SensorBindingSpec) -> Self {
        self.sensors.push(sensor);
        self
    }

    /// Añade una acción declarada al agente.
    pub fn with_action(mut self, action: impl Into<String>) -> Self {
        self.actions.push(action.into());
        self
    }

    /// Añade un objetivo tipado al agente.
    pub fn with_goal(mut self, goal: GoalSpec) -> Self {
        self.goals.push(goal);
        self
    }

    /// Añade una restricción tipada al agente.
    pub fn with_constraint(mut self, constraint: ConstraintSpec) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Añade un parámetro escalar de configuración al agente.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida estructural y semánticamente el agente adaptativo.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        validate_non_empty(&self.agent_id, "AdaptiveAgentSpec.agent_id")?;
        reject_code_eval_str(&self.agent_id, "AdaptiveAgentSpec", "agent_id")?;
        validate_config_map(&self.config, "AdaptiveAgentSpec.config")?;
        reject_code_eval_in_config(&self.config, "AdaptiveAgentSpec.config")?;

        self.system.validate()?;
        self.model.validate()?;

        let mut seen_sensors = BTreeSet::new();
        for sensor in &self.sensors {
            sensor.validate()?;
            if !seen_sensors.insert(sensor.sensor_id.clone()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "AdaptiveAgentSpec.sensors",
                    id: sensor.sensor_id.clone(),
                });
            }
        }

        let mut seen_actions = BTreeSet::new();
        for action in &self.actions {
            validate_non_empty(action, "AdaptiveAgentSpec.actions")?;
            reject_code_eval_str(action, "AdaptiveAgentSpec", "actions")?;
            if !seen_actions.insert(action.clone()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "AdaptiveAgentSpec.actions",
                    id: action.clone(),
                });
            }
        }

        validate_unique_goals(&self.goals, "AdaptiveAgentSpec.goals")?;
        validate_unique_constraints(&self.constraints, "AdaptiveAgentSpec.constraints")?;
        Ok(())
    }

    /// Valida el agente adaptativo y verifica que todas sus acciones estén registradas.
    pub fn validate_with_registry(&self, registry: &ActionRegistry) -> Result<(), SpecError> {
        self.validate()?;
        for action in &self.actions {
            registry.validate_action(action, "AdaptiveAgentSpec.actions")?;
        }
        Ok(())
    }

    /// Serializa el agente adaptativo a JSON canónico determinista.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_adaptive_agent_spec(&mut out, self)?;
        Ok(out)
    }

    /// Deserializa y valida estrictamente un [`AdaptiveAgentSpec`].
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa y valida un [`AdaptiveAgentSpec`] aplicando la política indicada.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let spec = parse_adaptive_agent_spec(&value, policy)?;
        spec.validate()?;
        Ok(spec)
    }
}

/// Especificación declarativa de un experimento adaptativo tipado compuesta sobre [`SystemSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveExperimentSpec {
    /// Versión de esquema (`ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION`).
    pub schema_version: String,
    /// Identificador único del experimento.
    pub experiment_id: String,
    /// Mundo/entorno objetivo del experimento.
    pub world: WorldSpecRef,
    /// Sistema causal base del experimento.
    pub system: SystemSpec,
    /// Agentes adaptativos participantes.
    pub agents: Vec<AdaptiveAgentSpec>,
    /// Objetivos globales del experimento.
    pub goals: Vec<GoalSpec>,
    /// Restricciones globales del experimento.
    pub constraints: Vec<ConstraintSpec>,
    /// Presupuesto de ejecución acotado.
    pub budget: BudgetSpec,
    /// Semillas `dev` y `holdout` disjuntas.
    pub seeds: SeedSpec,
    /// Parámetros escalares del experimento.
    pub config: SpecConfigMap,
}

impl AdaptiveExperimentSpec {
    /// Crea una especificación de experimento adaptativo tipado.
    pub fn new(
        experiment_id: impl Into<String>,
        world: WorldSpecRef,
        system: SystemSpec,
        budget: BudgetSpec,
        seeds: SeedSpec,
    ) -> Self {
        Self {
            schema_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION.to_owned(),
            experiment_id: experiment_id.into(),
            world,
            system,
            agents: Vec::new(),
            goals: Vec::new(),
            constraints: Vec::new(),
            budget,
            seeds,
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un agente adaptativo al experimento.
    pub fn with_agent(mut self, agent: AdaptiveAgentSpec) -> Self {
        self.agents.push(agent);
        self
    }

    /// Añade un objetivo global al experimento.
    pub fn with_goal(mut self, goal: GoalSpec) -> Self {
        self.goals.push(goal);
        self
    }

    /// Añade una restricción global al experimento.
    pub fn with_constraint(mut self, constraint: ConstraintSpec) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Añade un parámetro escalar de configuración al experimento.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida estructural y semánticamente todo el experimento adaptativo.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        validate_non_empty(&self.experiment_id, "AdaptiveExperimentSpec.experiment_id")?;
        reject_code_eval_str(
            &self.experiment_id,
            "AdaptiveExperimentSpec",
            "experiment_id",
        )?;
        validate_config_map(&self.config, "AdaptiveExperimentSpec.config")?;
        reject_code_eval_in_config(&self.config, "AdaptiveExperimentSpec.config")?;

        self.world.validate()?;
        self.system.validate()?;
        self.budget.validate()?;
        self.seeds.validate()?;

        let mut seen_agents = BTreeSet::new();
        for agent in &self.agents {
            agent.validate()?;
            if !seen_agents.insert(agent.agent_id.clone()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "AdaptiveExperimentSpec.agents",
                    id: agent.agent_id.clone(),
                });
            }
        }

        validate_unique_goals(&self.goals, "AdaptiveExperimentSpec.goals")?;
        validate_unique_constraints(&self.constraints, "AdaptiveExperimentSpec.constraints")?;
        Ok(())
    }

    /// Valida el experimento adaptativo y verifica todas las acciones contra [`ActionRegistry`].
    pub fn validate_with_registry(&self, registry: &ActionRegistry) -> Result<(), SpecError> {
        self.validate()?;
        for agent in &self.agents {
            agent.validate_with_registry(registry)?;
        }
        Ok(())
    }

    /// Serializa el experimento adaptativo a JSON canónico determinista.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        write_adaptive_experiment_spec(&mut out, self)?;
        Ok(out)
    }

    /// Deserializa y valida estrictamente un [`AdaptiveExperimentSpec`].
    pub fn from_json(json: &str) -> Result<Self, SpecError> {
        Self::from_json_with_policy(json, UnknownFieldPolicy::Reject)
    }

    /// Deserializa y valida un [`AdaptiveExperimentSpec`] aplicando la política indicada.
    pub fn from_json_with_policy(
        json: &str,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        let value = parse_json(json)?;
        let spec = parse_adaptive_experiment_spec(&value, policy)?;
        spec.validate()?;
        Ok(spec)
    }
}

/// Reporte estructurado de una migración de esquema de especificación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaMigrationReport {
    /// Versión de esquema de origen detectada.
    pub source_version: String,
    /// Versión de esquema de destino producida.
    pub target_version: String,
    /// Pasos aplicados durante la migración.
    pub applied_steps: Vec<String>,
    /// Diagnósticos informativos generados durante la migración.
    pub diagnostics: Vec<SpecDiagnostic>,
}

/// Resultado de migrar una especificación de experimento hacia [`AdaptiveExperimentSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct MigratedExperimentSpec {
    /// Especificación adaptativa tipada resultante.
    pub spec: AdaptiveExperimentSpec,
    /// Reporte detallado de la migración aplicada.
    pub report: SchemaMigrationReport,
}

/// Migra una especificación JSON de experimento (`v0` legacy o `v0.1`) hacia [`AdaptiveExperimentSpec`].
///
/// Devuelve [`SpecMigrationError`] con [`SpecDiagnostic`] detallando la ruta de campo y
/// versión esperada vs recibida si el esquema es incompatible o inválido.
pub fn migrate_experiment_spec_json(
    json: &str,
) -> Result<MigratedExperimentSpec, SpecMigrationError> {
    let value = parse_json(json).map_err(|err| {
        SpecMigrationError::new(
            SpecDiagnostic::new("$", "failed to parse experiment JSON"),
            Some(err),
        )
    })?;

    let root = as_object(&value, "ExperimentSpecMigration").map_err(|err| {
        SpecMigrationError::new(
            SpecDiagnostic::new("$", "experiment root must be a JSON object"),
            Some(err),
        )
    })?;

    let expected_versions =
        format!("{EXPERIMENT_SPEC_SCHEMA_VERSION} | {ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION}");
    let found_version = match root.get("schema_version") {
        Some(JsonValue::String(s)) => s.clone(),
        Some(_) => {
            return Err(SpecMigrationError::new(
                SpecDiagnostic::version_mismatch(
                    "$.schema_version",
                    expected_versions,
                    None,
                    "schema_version must be a JSON string",
                ),
                Some(SpecError::InvalidFieldValue {
                    context: "ExperimentSpecMigration",
                    field: "schema_version",
                    reason: "expected string".to_owned(),
                }),
            ));
        }
        None => {
            return Err(SpecMigrationError::new(
                SpecDiagnostic::version_mismatch(
                    "$.schema_version",
                    expected_versions,
                    None,
                    "missing required schema_version",
                ),
                Some(SpecError::MissingField {
                    context: "ExperimentSpecMigration",
                    field: "schema_version",
                }),
            ));
        }
    };

    if found_version == ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION {
        let spec =
            parse_adaptive_experiment_spec(&value, UnknownFieldPolicy::Reject).map_err(|err| {
                SpecMigrationError::new(
                    SpecDiagnostic::new("$", "invalid v0.1 adaptive experiment spec"),
                    Some(err),
                )
            })?;
        spec.validate().map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "adaptive experiment validation failed"),
                Some(err),
            )
        })?;
        return Ok(MigratedExperimentSpec {
            spec,
            report: SchemaMigrationReport {
                source_version: found_version,
                target_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION.to_owned(),
                applied_steps: vec!["already_at_target_version".to_owned()],
                diagnostics: Vec::new(),
            },
        });
    }

    if found_version == EXPERIMENT_SPEC_SCHEMA_VERSION {
        let legacy = parse_experiment_spec(&value, UnknownFieldPolicy::Reject).map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "invalid legacy v0 experiment spec"),
                Some(err),
            )
        })?;
        legacy.validate().map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "legacy v0 experiment validation failed"),
                Some(err),
            )
        })?;

        let mut diagnostics = Vec::new();
        let mut applied_steps = vec!["upgrade_schema_version:v0->v0.1".to_owned()];

        let world = WorldSpecRef::new(
            format!("{}_world", legacy.experiment_id),
            "legacy_system_world",
        );
        diagnostics.push(SpecDiagnostic::new(
            "$.world",
            "synthesized WorldSpecRef from legacy experiment_id",
        ));
        applied_steps.push("synthesize_world_ref".to_owned());

        let budget = BudgetSpec::new(1000, 1000, 10_000);
        diagnostics.push(SpecDiagnostic::new(
            "$.budget",
            "applied default bounded BudgetSpec (1000 ticks, 1000 transitions, 10000 ms)",
        ));
        applied_steps.push("apply_default_budget".to_owned());

        let holdout_seed = legacy.seed.wrapping_add(1);
        let seeds = SeedSpec::new(vec![legacy.seed], vec![holdout_seed]);
        diagnostics.push(SpecDiagnostic::new(
            "$.seeds",
            format!(
                "mapped legacy seed {} to dev_seeds and derived disjoint holdout seed {}",
                legacy.seed, holdout_seed
            ),
        ));
        applied_steps.push("partition_legacy_seed".to_owned());

        let mut agents = Vec::with_capacity(legacy.agents.len());
        for (idx, legacy_agent) in legacy.agents.into_iter().enumerate() {
            let model = ModelSpecRef::new(
                format!("{}_model", legacy_agent.agent_id),
                "legacy_system_policy",
            );
            let mut adaptive_agent =
                AdaptiveAgentSpec::new(legacy_agent.agent_id, legacy_agent.system, model)
                    .with_action("stay");
            adaptive_agent.config = legacy_agent.config;
            agents.push(adaptive_agent);
            diagnostics.push(SpecDiagnostic::new(
                format!("$.agents[{idx}]"),
                "migrated legacy AgentSpec to AdaptiveAgentSpec with default 'stay' action",
            ));
        }
        applied_steps.push("migrate_legacy_agents".to_owned());

        let mut spec =
            AdaptiveExperimentSpec::new(legacy.experiment_id, world, legacy.system, budget, seeds);
        spec.agents = agents;
        spec.config = legacy.config;
        spec.validate().map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "migrated experiment failed v0.1 validation"),
                Some(err),
            )
        })?;

        return Ok(MigratedExperimentSpec {
            spec,
            report: SchemaMigrationReport {
                source_version: found_version,
                target_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION.to_owned(),
                applied_steps,
                diagnostics,
            },
        });
    }

    Err(SpecMigrationError::new(
        SpecDiagnostic::version_mismatch(
            "$.schema_version",
            expected_versions,
            Some(found_version.clone()),
            format!("unsupported experiment schema_version '{found_version}'"),
        ),
        Some(SpecError::IncompatibleSchemaVersion {
            expected: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
            found: found_version,
        }),
    ))
}

/// Construye un fixture canónico determinista de experimento Grid 5x5 sin UI.
pub fn canonical_grid_experiment_fixture() -> AdaptiveExperimentSpec {
    let env_node = SystemNodeSpec::new("grid_env", "grid_world")
        .with_output("grid_obs", PortType::custom("GridObservation"))
        .with_input("action", PortType::String)
        .with_config("width", SpecConfigValue::I64(5))
        .with_config("height", SpecConfigValue::I64(5));

    let policy_node = SystemNodeSpec::new("agent_policy", "policy")
        .with_input("grid_obs", PortType::custom("GridObservation"))
        .with_output("action", PortType::String);

    let system = SystemSpec::new("grid_5x5_system")
        .with_node(env_node)
        .with_node(policy_node)
        .with_connection(SystemConnectionSpec::instant(
            "grid_env",
            "grid_obs",
            "agent_policy",
            "grid_obs",
        ))
        .with_connection(SystemConnectionSpec::delayed(
            "agent_policy",
            "action",
            "grid_env",
            "action",
            1,
        ));

    let sensor = SensorBindingSpec::new(
        "grid_pos_sensor",
        "position",
        "grid_env",
        "grid_obs",
        PortType::custom("GridObservation"),
    );

    let model = ModelSpecRef::new("tabular_q_v0", "tabular_policy")
        .with_config("alpha", SpecConfigValue::F64(0.1))
        .with_config("gamma", SpecConfigValue::F64(0.95));

    let reward_expr = ExprAst::gt(
        ExprAst::var("reward").expect("valid var"),
        ExprAst::const_f64(0.0).expect("finite"),
    );
    let energy_expr = ExprAst::ge(
        ExprAst::var("energy").expect("valid var"),
        ExprAst::const_f64(0.0).expect("finite"),
    );

    let agent = AdaptiveAgentSpec::new("grid_navigator", system.clone(), model)
        .with_sensor(sensor)
        .with_action("move_north")
        .with_action("move_south")
        .with_action("move_east")
        .with_action("move_west")
        .with_goal(GoalSpec::new("reach_goal", reward_expr.clone(), true))
        .with_constraint(ConstraintSpec::new(
            "non_negative_energy",
            energy_expr,
            true,
        ));

    let world = WorldSpecRef::new("grid_world_5x5", "grid")
        .with_config("width", SpecConfigValue::I64(5))
        .with_config("height", SpecConfigValue::I64(5));

    AdaptiveExperimentSpec::new(
        "grid_canonical_v0_1",
        world,
        system,
        BudgetSpec::new(500, 1000, 5000),
        SeedSpec::new(vec![101, 102, 103], vec![201, 202]),
    )
    .with_agent(agent)
    .with_goal(GoalSpec::new("maximize_return", reward_expr, true))
    .with_config("headless", SpecConfigValue::Bool(true))
}

/// Exporta el fixture canónico Grid 5x5 en JSON canónico determinista.
pub fn canonical_grid_experiment_fixture_json() -> String {
    canonical_grid_experiment_fixture()
        .to_canonical_json()
        .expect("canonical grid experiment fixture must serialize")
}

/// Construye un fixture canónico cognitivo (Grid parcialmente observable con creencias y objetivo) sin UI.
pub fn canonical_cognitive_grid_experiment_fixture() -> AdaptiveExperimentSpec {
    let env_node = SystemNodeSpec::new("pomdp_grid_env", "grid_pomdp")
        .with_output("local_fov", PortType::custom("LocalObservationGrid"))
        .with_input("action", PortType::String)
        .with_config("fov_radius", SpecConfigValue::I64(2))
        .with_config("partial_observability", SpecConfigValue::Bool(true));

    let belief_node = SystemNodeSpec::new("belief_updater", "belief_state")
        .with_input("local_fov", PortType::custom("LocalObservationGrid"))
        .with_output("belief_state", PortType::custom("BeliefDistribution"));

    let planner_node = SystemNodeSpec::new("cognitive_planner", "active_inference")
        .with_input("belief_state", PortType::custom("BeliefDistribution"))
        .with_output("action", PortType::String);

    let system = SystemSpec::new("cognitive_grid_pomdp_system")
        .with_node(env_node)
        .with_node(belief_node)
        .with_node(planner_node)
        .with_connection(SystemConnectionSpec::instant(
            "pomdp_grid_env",
            "local_fov",
            "belief_updater",
            "local_fov",
        ))
        .with_connection(SystemConnectionSpec::instant(
            "belief_updater",
            "belief_state",
            "cognitive_planner",
            "belief_state",
        ))
        .with_connection(SystemConnectionSpec::delayed(
            "cognitive_planner",
            "action",
            "pomdp_grid_env",
            "action",
            1,
        ));

    let fov_sensor = SensorBindingSpec::new(
        "local_fov_sensor",
        "partial_fov",
        "pomdp_grid_env",
        "local_fov",
        PortType::custom("LocalObservationGrid"),
    )
    .with_config("radius", SpecConfigValue::I64(2));

    let model = ModelSpecRef::new("pomdp_belief_v0", "belief_state")
        .with_config("belief_horizon", SpecConfigValue::I64(8))
        .with_config("entropy_weight", SpecConfigValue::F64(0.25));

    let epistemic_goal = ExprAst::add(
        ExprAst::var("reward").expect("valid var"),
        ExprAst::mul(
            ExprAst::const_f64(0.25).expect("finite"),
            ExprAst::var("info_gain").expect("valid var"),
        ),
    );
    let energy_guard = ExprAst::gt(
        ExprAst::var("energy").expect("valid var"),
        ExprAst::const_f64(0.0).expect("finite"),
    );

    let agent = AdaptiveAgentSpec::new("cognitive_explorer", system.clone(), model)
        .with_sensor(fov_sensor)
        .with_action("move_north")
        .with_action("move_south")
        .with_action("move_east")
        .with_action("move_west")
        .with_action("observe")
        .with_action("sense_local")
        .with_goal(GoalSpec::new(
            "epistemic_utility",
            epistemic_goal.clone(),
            true,
        ))
        .with_constraint(ConstraintSpec::new(
            "energy_guard",
            energy_guard.clone(),
            true,
        ));

    let world = WorldSpecRef::new("grid_pomdp_7x7", "grid_pomdp")
        .with_config("width", SpecConfigValue::I64(7))
        .with_config("height", SpecConfigValue::I64(7))
        .with_config("partial_observability", SpecConfigValue::Bool(true));

    AdaptiveExperimentSpec::new(
        "cognitive_grid_canonical_v0_1",
        world,
        system,
        BudgetSpec::new(1000, 2500, 10_000),
        SeedSpec::new(vec![11, 12, 13], vec![1001, 1002]),
    )
    .with_agent(agent)
    .with_goal(GoalSpec::new("epistemic_utility", epistemic_goal, true))
    .with_constraint(ConstraintSpec::new("energy_guard", energy_guard, true))
    .with_config("headless", SpecConfigValue::Bool(true))
}

/// Exporta el fixture canónico cognitivo en JSON canónico determinista.
pub fn canonical_cognitive_grid_experiment_fixture_json() -> String {
    canonical_cognitive_grid_experiment_fixture()
        .to_canonical_json()
        .expect("canonical cognitive grid experiment fixture must serialize")
}

// ---------------------------------------------------------------------------
// Validación auxiliar (seguridad sin eval + unicidad)
// ---------------------------------------------------------------------------

fn reject_code_eval_str(value: &str, context: &'static str, field: &str) -> Result<(), SpecError> {
    let lower = value.to_ascii_lowercase();
    for marker in FORBIDDEN_EVAL_MARKERS {
        if lower.contains(marker) {
            return Err(SpecError::CodeEvalRejected {
                context,
                field: field.to_owned(),
                reason: format!("forbidden code evaluation marker '{marker}' detected"),
            });
        }
    }
    Ok(())
}

fn reject_code_eval_in_config(
    config: &SpecConfigMap,
    context: &'static str,
) -> Result<(), SpecError> {
    for (key, value) in config {
        let lower_key = key.to_ascii_lowercase();
        if FORBIDDEN_EVAL_KEYS.contains(&lower_key.as_str()) {
            return Err(SpecError::CodeEvalRejected {
                context,
                field: key.clone(),
                reason: format!("config key '{key}' attempts arbitrary code evaluation"),
            });
        }
        if let SpecConfigValue::String(s) = value {
            reject_code_eval_str(s, context, key)?;
        }
    }
    Ok(())
}

fn validate_unique_goals(goals: &[GoalSpec], context: &'static str) -> Result<(), SpecError> {
    let mut seen = BTreeSet::new();
    for goal in goals {
        goal.validate()?;
        if !seen.insert(goal.goal_id.clone()) {
            return Err(SpecError::DuplicateIdentifier {
                context,
                id: goal.goal_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_unique_constraints(
    constraints: &[ConstraintSpec],
    context: &'static str,
) -> Result<(), SpecError> {
    let mut seen = BTreeSet::new();
    for constraint in constraints {
        constraint.validate()?;
        if !seen.insert(constraint.constraint_id.clone()) {
            return Err(SpecError::DuplicateIdentifier {
                context,
                id: constraint.constraint_id.clone(),
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Serialización JSON canónica
// ---------------------------------------------------------------------------

fn write_world_spec_ref(out: &mut String, world: &WorldSpecRef) -> Result<(), SpecError> {
    out.push_str("{\"config\":");
    write_config_map(out, &world.config)?;
    out.push_str(",\"kind\":");
    write_json_string(out, &world.kind);
    out.push_str(",\"world_id\":");
    write_json_string(out, &world.world_id);
    out.push('}');
    Ok(())
}

fn write_sensor_binding_spec(
    out: &mut String,
    sensor: &SensorBindingSpec,
) -> Result<(), SpecError> {
    out.push_str("{\"config\":");
    write_config_map(out, &sensor.config)?;
    out.push_str(",\"kind\":");
    write_json_string(out, &sensor.kind);
    out.push_str(",\"port_type\":");
    write_port_type(out, &sensor.port_type);
    out.push_str(",\"sensor_id\":");
    write_json_string(out, &sensor.sensor_id);
    out.push_str(",\"source_node\":");
    write_json_string(out, sensor.source_node.as_str());
    out.push_str(",\"source_port\":");
    write_json_string(out, sensor.source_port.as_str());
    out.push('}');
    Ok(())
}

fn write_model_spec_ref(out: &mut String, model: &ModelSpecRef) -> Result<(), SpecError> {
    out.push_str("{\"config\":");
    write_config_map(out, &model.config)?;
    out.push_str(",\"kind\":");
    write_json_string(out, &model.kind);
    out.push_str(",\"model_id\":");
    write_json_string(out, &model.model_id);
    out.push('}');
    Ok(())
}

fn write_goal_spec(out: &mut String, goal: &GoalSpec) -> Result<(), SpecError> {
    out.push_str("{\"config\":");
    write_config_map(out, &goal.config)?;
    out.push_str(",\"expr\":");
    write_expr_json(out, &goal.expr, 0)?;
    out.push_str(",\"goal_id\":");
    write_json_string(out, &goal.goal_id);
    out.push_str(",\"maximize\":");
    out.push_str(if goal.maximize { "true" } else { "false" });
    out.push('}');
    Ok(())
}

fn write_constraint_spec(out: &mut String, constraint: &ConstraintSpec) -> Result<(), SpecError> {
    out.push_str("{\"condition\":");
    write_expr_json(out, &constraint.condition, 0)?;
    out.push_str(",\"config\":");
    write_config_map(out, &constraint.config)?;
    out.push_str(",\"constraint_id\":");
    write_json_string(out, &constraint.constraint_id);
    out.push_str(",\"hard\":");
    out.push_str(if constraint.hard { "true" } else { "false" });
    out.push('}');
    Ok(())
}

fn write_budget_spec(out: &mut String, budget: &BudgetSpec) {
    let _ = write!(
        out,
        "{{\"deadline_ms\":{},\"max_ticks\":{},\"max_transitions\":{}}}",
        budget.deadline_ms, budget.max_ticks, budget.max_transitions
    );
}

fn write_seed_spec(out: &mut String, seeds: &SeedSpec) {
    out.push_str("{\"dev_seeds\":[");
    for (idx, seed) in seeds.dev_seeds.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{seed}");
    }
    out.push_str("],\"holdout_seeds\":[");
    for (idx, seed) in seeds.holdout_seeds.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{seed}");
    }
    out.push_str("]}");
}

fn write_adaptive_agent_spec(out: &mut String, agent: &AdaptiveAgentSpec) -> Result<(), SpecError> {
    out.push_str("{\"actions\":[");
    for (idx, action) in agent.actions.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_json_string(out, action);
    }
    out.push_str("],\"agent_id\":");
    write_json_string(out, &agent.agent_id);
    out.push_str(",\"config\":");
    write_config_map(out, &agent.config)?;
    out.push_str(",\"constraints\":[");
    for (idx, constraint) in agent.constraints.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_constraint_spec(out, constraint)?;
    }
    out.push_str("],\"goals\":[");
    for (idx, goal) in agent.goals.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_goal_spec(out, goal)?;
    }
    out.push_str("],\"model\":");
    write_model_spec_ref(out, &agent.model)?;
    out.push_str(",\"schema_version\":");
    write_json_string(out, &agent.schema_version);
    out.push_str(",\"sensors\":[");
    for (idx, sensor) in agent.sensors.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_sensor_binding_spec(out, sensor)?;
    }
    out.push_str("],\"system\":");
    write_system_spec(out, &agent.system)?;
    out.push('}');
    Ok(())
}

fn write_adaptive_experiment_spec(
    out: &mut String,
    exp: &AdaptiveExperimentSpec,
) -> Result<(), SpecError> {
    out.push_str("{\"agents\":[");
    for (idx, agent) in exp.agents.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_adaptive_agent_spec(out, agent)?;
    }
    out.push_str("],\"budget\":");
    write_budget_spec(out, &exp.budget);
    out.push_str(",\"config\":");
    write_config_map(out, &exp.config)?;
    out.push_str(",\"constraints\":[");
    for (idx, constraint) in exp.constraints.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_constraint_spec(out, constraint)?;
    }
    out.push_str("],\"experiment_id\":");
    write_json_string(out, &exp.experiment_id);
    out.push_str(",\"goals\":[");
    for (idx, goal) in exp.goals.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_goal_spec(out, goal)?;
    }
    out.push_str("],\"schema_version\":");
    write_json_string(out, &exp.schema_version);
    out.push_str(",\"seeds\":");
    write_seed_spec(out, &exp.seeds);
    out.push_str(",\"system\":");
    write_system_spec(out, &exp.system)?;
    out.push_str(",\"world\":");
    write_world_spec_ref(out, &exp.world)?;
    out.push('}');
    Ok(())
}

// ---------------------------------------------------------------------------
// Deserialización JSON estricta
// ---------------------------------------------------------------------------

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
            reason: "expected boolean".to_owned(),
        }),
        None => Err(SpecError::MissingField { context, field }),
    }
}

fn require_field<'a>(
    map: &'a BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<&'a JsonValue, SpecError> {
    map.get(field)
        .ok_or(SpecError::MissingField { context, field })
}

fn parse_world_spec_ref(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<WorldSpecRef, SpecError> {
    const CTX: &str = "WorldSpecRef";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["world_id", "kind", "config"], CTX, policy)?;
    Ok(WorldSpecRef {
        world_id: require_string(map, "world_id", CTX)?.to_owned(),
        kind: require_string(map, "kind", CTX)?.to_owned(),
        config: parse_optional_config(map, CTX)?,
    })
}

fn parse_sensor_binding_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<SensorBindingSpec, SpecError> {
    const CTX: &str = "SensorBindingSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "sensor_id",
            "kind",
            "source_node",
            "source_port",
            "port_type",
            "config",
        ],
        CTX,
        policy,
    )?;
    let port_type_val = require_field(map, "port_type", CTX)?;
    Ok(SensorBindingSpec {
        sensor_id: require_string(map, "sensor_id", CTX)?.to_owned(),
        kind: require_string(map, "kind", CTX)?.to_owned(),
        source_node: NodeId::new(require_string(map, "source_node", CTX)?),
        source_port: PortId::new(require_string(map, "source_port", CTX)?),
        port_type: parse_port_type(port_type_val, policy)?,
        config: parse_optional_config(map, CTX)?,
    })
}

fn parse_model_spec_ref(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<ModelSpecRef, SpecError> {
    const CTX: &str = "ModelSpecRef";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["model_id", "kind", "config"], CTX, policy)?;
    Ok(ModelSpecRef {
        model_id: require_string(map, "model_id", CTX)?.to_owned(),
        kind: require_string(map, "kind", CTX)?.to_owned(),
        config: parse_optional_config(map, CTX)?,
    })
}

fn parse_goal_spec(value: &JsonValue, policy: UnknownFieldPolicy) -> Result<GoalSpec, SpecError> {
    const CTX: &str = "GoalSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["goal_id", "expr", "maximize", "config"], CTX, policy)?;
    let expr_val = require_field(map, "expr", CTX)?;
    Ok(GoalSpec {
        goal_id: require_string(map, "goal_id", CTX)?.to_owned(),
        expr: parse_expr_json(expr_val, policy, 0)?,
        maximize: require_bool(map, "maximize", CTX)?,
        config: parse_optional_config(map, CTX)?,
    })
}

fn parse_constraint_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<ConstraintSpec, SpecError> {
    const CTX: &str = "ConstraintSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &["constraint_id", "condition", "hard", "config"],
        CTX,
        policy,
    )?;
    let cond_val = require_field(map, "condition", CTX)?;
    Ok(ConstraintSpec {
        constraint_id: require_string(map, "constraint_id", CTX)?.to_owned(),
        condition: parse_expr_json(cond_val, policy, 0)?,
        hard: require_bool(map, "hard", CTX)?,
        config: parse_optional_config(map, CTX)?,
    })
}

fn parse_budget_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<BudgetSpec, SpecError> {
    const CTX: &str = "BudgetSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &["max_ticks", "max_transitions", "deadline_ms"],
        CTX,
        policy,
    )?;
    Ok(BudgetSpec {
        max_ticks: require_u64(map, "max_ticks", CTX)?,
        max_transitions: require_u64(map, "max_transitions", CTX)?,
        deadline_ms: require_u64(map, "deadline_ms", CTX)?,
    })
}

fn parse_u64_array(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    context: &'static str,
) -> Result<Vec<u64>, SpecError> {
    let items = require_array(map, field, context)?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        match item {
            JsonValue::Number(JsonNumber::I64(n)) if *n >= 0 => out.push(*n as u64),
            JsonValue::Number(JsonNumber::U64(n)) => out.push(*n),
            _ => {
                return Err(SpecError::InvalidFieldValue {
                    context,
                    field,
                    reason: "expected array of non-negative u64 integers".to_owned(),
                });
            }
        }
    }
    Ok(out)
}

fn parse_seed_spec(value: &JsonValue, policy: UnknownFieldPolicy) -> Result<SeedSpec, SpecError> {
    const CTX: &str = "SeedSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(map, &["dev_seeds", "holdout_seeds"], CTX, policy)?;
    Ok(SeedSpec {
        dev_seeds: parse_u64_array(map, "dev_seeds", CTX)?,
        holdout_seeds: parse_u64_array(map, "holdout_seeds", CTX)?,
    })
}

fn parse_adaptive_agent_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<AdaptiveAgentSpec, SpecError> {
    const CTX: &str = "AdaptiveAgentSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "schema_version",
            "agent_id",
            "system",
            "sensors",
            "model",
            "actions",
            "goals",
            "constraints",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_schema_version(map, ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION, CTX)?;
    let agent_id = require_string(map, "agent_id", CTX)?.to_owned();
    let system = parse_system_spec(require_field(map, "system", CTX)?, policy)?;

    let raw_sensors = require_array(map, "sensors", CTX)?;
    let mut sensors = Vec::with_capacity(raw_sensors.len());
    for s in raw_sensors {
        sensors.push(parse_sensor_binding_spec(s, policy)?);
    }

    let model = parse_model_spec_ref(require_field(map, "model", CTX)?, policy)?;

    let raw_actions = require_array(map, "actions", CTX)?;
    let mut actions = Vec::with_capacity(raw_actions.len());
    for a in raw_actions {
        match a {
            JsonValue::String(s) => actions.push(s.clone()),
            _ => {
                return Err(SpecError::InvalidFieldValue {
                    context: CTX,
                    field: "actions",
                    reason: "expected array of action strings".to_owned(),
                });
            }
        }
    }

    let raw_goals = require_array(map, "goals", CTX)?;
    let mut goals = Vec::with_capacity(raw_goals.len());
    for g in raw_goals {
        goals.push(parse_goal_spec(g, policy)?);
    }

    let raw_constraints = require_array(map, "constraints", CTX)?;
    let mut constraints = Vec::with_capacity(raw_constraints.len());
    for c in raw_constraints {
        constraints.push(parse_constraint_spec(c, policy)?);
    }

    let config = parse_optional_config(map, CTX)?;

    Ok(AdaptiveAgentSpec {
        schema_version,
        agent_id,
        system,
        sensors,
        model,
        actions,
        goals,
        constraints,
        config,
    })
}

fn parse_adaptive_experiment_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
) -> Result<AdaptiveExperimentSpec, SpecError> {
    const CTX: &str = "AdaptiveExperimentSpec";
    let map = as_object(value, CTX)?;
    reject_unknown_fields(
        map,
        &[
            "schema_version",
            "experiment_id",
            "world",
            "system",
            "agents",
            "goals",
            "constraints",
            "budget",
            "seeds",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_schema_version(map, ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION, CTX)?;
    let experiment_id = require_string(map, "experiment_id", CTX)?.to_owned();
    let world = parse_world_spec_ref(require_field(map, "world", CTX)?, policy)?;
    let system = parse_system_spec(require_field(map, "system", CTX)?, policy)?;

    let raw_agents = require_array(map, "agents", CTX)?;
    let mut agents = Vec::with_capacity(raw_agents.len());
    for a in raw_agents {
        agents.push(parse_adaptive_agent_spec(a, policy)?);
    }

    let raw_goals = require_array(map, "goals", CTX)?;
    let mut goals = Vec::with_capacity(raw_goals.len());
    for g in raw_goals {
        goals.push(parse_goal_spec(g, policy)?);
    }

    let raw_constraints = require_array(map, "constraints", CTX)?;
    let mut constraints = Vec::with_capacity(raw_constraints.len());
    for c in raw_constraints {
        constraints.push(parse_constraint_spec(c, policy)?);
    }

    let budget = parse_budget_spec(require_field(map, "budget", CTX)?, policy)?;
    let seeds = parse_seed_spec(require_field(map, "seeds", CTX)?, policy)?;
    let config = parse_optional_config(map, CTX)?;

    Ok(AdaptiveExperimentSpec {
        schema_version,
        experiment_id,
        world,
        system,
        agents,
        goals,
        constraints,
        budget,
        seeds,
        config,
    })
}
