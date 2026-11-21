//! Especificación adaptativa (`AdaptiveAgentSpec`) y `AdaptiveExperimentSpec` tipados.
//!
//! Este módulo amplía el contrato declarativo `std`-only de `gogamen-spec` componiendo
//! [`SystemSpec`] directamente (sin segunda IR), cubriendo mundo, sensores, modelo,
//! objetivos tipados ([`ExprAst`]), restricciones, presupuesto, semillas disjuntas,
//! catálogo explícito de acciones ([`CapabilityCatalog`] / [`ActionRegistry`]) y
//! migración versionada desde `gogamen.experiment_spec.v0`/`v0.1` hacia el contrato tipado
//! `gogamen.experiment_spec.v0.2` (principio, escala, aprendizaje, fenómenos y `vary`).
//!
//! # Versionado
//! La v0.1 sigue siendo la versión por defecto de [`AdaptiveExperimentSpec::new`] para no
//! alterar los bytes de los corpus dorados compartidos con el DSL Kotlin/Python. La v0.2
//! ([`ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2`]) añade los campos tipados v2 (obligatorios si se
//! declaran) y es el destino de la migración.
//!
//! # Migración de versiones (`v0`/`v0.1` -> `v0.2`)
//! La función [`migrate_experiment_spec_json`] soporta:
//! - `gogamen.experiment_spec.v0.2`: valida directamente como [`AdaptiveExperimentSpec`] sin
//!   cambios (`already_at_target_version`).
//! - `gogamen.experiment_spec.v0.1`: eleva `phenomena_json`, `vary_json`, `mode=pure`,
//!   `privileged`, `observability=hidden` y `question_json`/`ask_json`/`guard_json` a campos
//!   tipados con [`SchemaMigrationReport`].
//! - `gogamen.experiment_spec.v0` (legacy [`ExperimentSpec`](crate::ExperimentSpec)):
//!   migra determinísticamente derivando [`WorldSpecRef`], [`BudgetSpec`] acotado por defecto,
//!   [`SeedSpec`] disjunto (`dev=[seed]`, `holdout=[seed.wrapping_add(1)]`) y
//!   [`ModelSpecRef`] por agente, y después encadena el ascenso a v0.2, registrando cada
//!   paso y diagnóstico en [`SchemaMigrationReport`].
//! - Cualquier otra versión o estructura inválida devuelve [`SpecMigrationError`] con
//!   [`SpecDiagnostic`] (ruta de campo y versión esperada vs recibida).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::{NodeId, PortId, PortType};

use crate::codec::{
    as_object, parse_experiment_spec, parse_optional_config, parse_port_type, parse_system_spec,
    reject_unknown_fields, require_array, require_string, require_u64,
    write_config_map, write_port_type, write_system_spec,
};
use crate::cognitive::{CognitiveRefs, CognitiveSpec, LearningSpec};
use crate::endowment::{
    EndowmentBook, ModelAccess, PhenomenonSpec, PrincipleSpec, ScaleAxisSpec, ScaleSpec,
    VaryAxisSpec, parse_learning_spec, parse_phenomena, parse_principle_spec, parse_scale_spec,
    parse_vary,
};
use crate::epistemic::{EpistemicQuestionSpec, EpistemicSpec};
use crate::error::{SpecDiagnostic, SpecError, SpecMigrationError, UnknownFieldPolicy};
use crate::expr::{ExprAst, parse_expr_json, write_expr_json};
use crate::hypothesis::{
    HypothesisComparator, HypothesisSpec, canonical_hypotheses_json, parse_hypotheses,
    validate_hypotheses,
};
use crate::json::{JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string};
use crate::model::{
    EXPERIMENT_SPEC_SCHEMA_VERSION, SpecConfigMap, SpecConfigValue, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec, validate_config_map, validate_non_empty,
};

/// Versión de esquema por defecto para [`AdaptiveAgentSpec`] (compatibilidad de bytes v0.1).
pub const ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION: &str = "gogamen.adaptive_agent_spec.v0.1";
/// Versión de esquema por defecto para [`AdaptiveExperimentSpec`] (compatibilidad de bytes v0.1).
pub const ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION: &str = "gogamen.experiment_spec.v0.1";
/// Versión v0.2 del agente: habilita principio, escala, aprendizaje, fenómenos y epistémico tipados.
pub const ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2: &str = "gogamen.adaptive_agent_spec.v0.2";
/// Versión v0.2 del experimento: habilita `vary` y agentes v0.2.
pub const ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2: &str = "gogamen.experiment_spec.v0.2";

const AGENT_SCHEMA_VERSIONS: &[&str] = &[
    ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION,
    ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
];
const EXPERIMENT_SCHEMA_VERSIONS: &[&str] = &[
    ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
    ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2,
];

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
    /// Modelo privilegiado del laboratorio: lee estado del mundo directamente.
    ///
    /// Se omite del JSON canónico cuando es `false`, así que las fixtures históricas conservan
    /// sus bytes. Es incompatible con `modelAccess: hidden`.
    pub privileged: bool,
}

impl ModelSpecRef {
    /// Crea una referencia declarativa a modelo (no privilegiado).
    pub fn new(model_id: impl Into<String>, kind: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            kind: kind.into(),
            config: SpecConfigMap::new(),
            privileged: false,
        }
    }

    /// Declara que el modelo es el modelo privilegiado del laboratorio.
    pub fn with_privileged(mut self, privileged: bool) -> Self {
        self.privileged = privileged;
        self
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
    /// Bloque cognitivo opcional (política entrópica, resolución, aprendizaje, memoria).
    pub cognitive: Option<CognitiveSpec>,
    /// Principio de selección tipado y observables que declara leer (v0.2).
    pub principle: Option<PrincipleSpec>,
    /// Escala compartida por aprendiz y planificador (v0.2).
    pub scale: Option<ScaleSpec>,
    /// Aprendizaje tipado de la v2 (v0.2; debe coincidir con `cognitive.learning` si existe).
    pub learn: Option<LearningSpec>,
    /// Fenómenos medidos desde fuera (v0.2).
    pub phenomena: Vec<PhenomenonSpec>,
    /// Acceso del agente al estado del mundo (`visible` por defecto, v0.2).
    pub model_access: ModelAccess,
    /// Bloque epistémico tipado que sustituye a `question_json`/`ask_json`/`guard_json` (v0.2).
    pub epistemic: Option<EpistemicSpec>,
    /// Parámetros escalares adicionales del agente.
    pub config: SpecConfigMap,
}

impl AdaptiveAgentSpec {
    /// Crea una especificación de agente adaptativo compuesta sobre [`SystemSpec`] (v0.1).
    pub fn new(agent_id: impl Into<String>, system: SystemSpec, model: ModelSpecRef) -> Self {
        Self::new_with_version(
            agent_id,
            system,
            model,
            ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION,
        )
    }

    /// Crea un agente con el contrato tipado v0.2.
    pub fn new_v2(agent_id: impl Into<String>, system: SystemSpec, model: ModelSpecRef) -> Self {
        Self::new_with_version(
            agent_id,
            system,
            model,
            ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
        )
    }

    fn new_with_version(
        agent_id: impl Into<String>,
        system: SystemSpec,
        model: ModelSpecRef,
        schema_version: &str,
    ) -> Self {
        Self {
            schema_version: schema_version.to_owned(),
            agent_id: agent_id.into(),
            system,
            sensors: Vec::new(),
            model,
            actions: Vec::new(),
            goals: Vec::new(),
            constraints: Vec::new(),
            cognitive: None,
            principle: None,
            scale: None,
            learn: None,
            phenomena: Vec::new(),
            model_access: ModelAccess::Visible,
            epistemic: None,
            config: SpecConfigMap::new(),
        }
    }

    /// Declara el principio de selección tipado.
    pub fn with_principle(mut self, principle: PrincipleSpec) -> Self {
        self.principle = Some(principle);
        self
    }

    /// Declara la escala compartida.
    pub fn with_scale(mut self, scale: ScaleSpec) -> Self {
        self.scale = Some(scale);
        self
    }

    /// Declara el aprendizaje tipado de la v2.
    pub fn with_learn(mut self, learning: LearningSpec) -> Self {
        self.learn = Some(learning);
        self
    }

    /// Añade un fenómeno medido desde fuera.
    pub fn with_phenomenon(mut self, phenomenon: PhenomenonSpec) -> Self {
        self.phenomena.push(phenomenon);
        self
    }

    /// Declara el modo de acceso al mundo.
    pub fn with_model_access(mut self, access: ModelAccess) -> Self {
        self.model_access = access;
        self
    }

    /// Declara el bloque epistémico tipado.
    pub fn with_epistemic(mut self, epistemic: EpistemicSpec) -> Self {
        self.epistemic = Some(epistemic);
        self
    }

    fn is_v2(&self) -> bool {
        self.schema_version == ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2
    }

    /// Declara el bloque cognitivo opcional del agente.
    pub fn with_cognitive(mut self, cognitive: CognitiveSpec) -> Self {
        self.cognitive = Some(cognitive);
        self
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
        if !AGENT_SCHEMA_VERSIONS.contains(&self.schema_version.as_str()) {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
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
        if let Some(cognitive) = &self.cognitive {
            let mut refs = CognitiveRefs::default().with_system(&self.system);
            refs.sensors = self.sensors.iter().map(|s| s.sensor_id.clone()).collect();
            refs.models.insert(self.model.model_id.clone());
            cognitive.validate(&refs)?;
        }
        self.validate_typed_fields()?;
        Ok(())
    }

    /// Valida los campos tipados v2 y su coherencia con el resto del agente.
    fn validate_typed_fields(&self) -> Result<(), SpecError> {
        let v2 = self.is_v2();
        ensure_v2(v2, self.principle.is_some(), "agent.principle")?;
        ensure_v2(v2, self.scale.is_some(), "agent.scale")?;
        ensure_v2(v2, self.learn.is_some(), "agent.learn")?;
        ensure_v2(v2, !self.phenomena.is_empty(), "agent.phenomena")?;
        ensure_v2(v2, self.epistemic.is_some(), "agent.epistemic")?;
        ensure_v2(
            v2,
            self.model_access != ModelAccess::Visible,
            "agent.model_access",
        )?;
        ensure_v2(v2, self.model.privileged, "agent.model.privileged")?;

        if let Some(principle) = &self.principle {
            crate::endowment::validate_principle(principle, "agent.principle")?;
        }
        if let Some(scale) = &self.scale {
            scale.to_scale("agent.scale")?;
        }
        if let Some(learning) = &self.learn {
            crate::endowment::validate_learning(learning, "agent.learn")?;
            if learning.model != self.model.model_id {
                return Err(SpecError::InvalidReference {
                    path: "agent.learn.model".to_owned(),
                    kind: "model",
                    id: learning.model.chars().take(64).collect(),
                });
            }
            if let Some(cognitive) = &self.cognitive {
                if cognitive.learning != *learning {
                    return Err(SpecError::InvalidFieldValue {
                        context: "AdaptiveAgentSpec",
                        field: "learn",
                        reason: "agent.learn: conflicts with cognitive.learning".to_owned(),
                    });
                }
            }
        }

        let mut seen_phenomena = BTreeSet::new();
        for (i, phenomenon) in self.phenomena.iter().enumerate() {
            phenomenon.validate(&format!("agent.phenomena[{i}]"))?;
            if !seen_phenomena.insert(phenomenon.phenomenon_id.as_str()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "AdaptiveAgentSpec.phenomena",
                    id: phenomenon.phenomenon_id.clone(),
                });
            }
        }

        // (a) Un principio puro no puede leer un observable que un fenómeno mide.
        if let Some(principle) = &self.principle {
            if principle.kind == gogamen_core::SelectionPrinciple::Pure {
                for (i, phenomenon) in self.phenomena.iter().enumerate() {
                    if principle
                        .observables
                        .iter()
                        .any(|declared| declared == &phenomenon.observable)
                    {
                        return Err(SpecError::PurePrincipleReadsPhenomenonObservable {
                            path: format!("agent.phenomena[{i}].observable"),
                            observable: phenomenon.observable.chars().take(64).collect(),
                        });
                    }
                }
            }
        }

        // (b) Un modelo privilegiado del laboratorio no puede convivir con acceso oculto.
        if self.model_access == ModelAccess::Hidden && self.model.privileged {
            return Err(SpecError::PrivilegedModelWithHiddenAccess {
                path: "agent.model_access".to_owned(),
                model_id: self.model.model_id.chars().take(64).collect(),
            });
        }

        if let Some(epistemic) = &self.epistemic {
            epistemic.validate()?;
        }
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
    /// Bloque cognitivo opcional por defecto del experimento.
    pub cognitive: Option<CognitiveSpec>,
    /// Hypotheses pre-registered before execution (omitted from JSON when empty).
    pub hypotheses: Vec<HypothesisSpec>,
    /// Ejes de ablación que expanden el experimento a una matriz (v0.2).
    pub vary: Vec<VaryAxisSpec>,
    /// Parámetros escalares del experimento.
    pub config: SpecConfigMap,
}

impl AdaptiveExperimentSpec {
    /// Crea una especificación de experimento adaptativo tipado (v0.1).
    pub fn new(
        experiment_id: impl Into<String>,
        world: WorldSpecRef,
        system: SystemSpec,
        budget: BudgetSpec,
        seeds: SeedSpec,
    ) -> Self {
        Self::new_with_version(
            experiment_id,
            world,
            system,
            budget,
            seeds,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION,
        )
    }

    /// Crea un experimento con el contrato tipado v0.2.
    pub fn new_v2(
        experiment_id: impl Into<String>,
        world: WorldSpecRef,
        system: SystemSpec,
        budget: BudgetSpec,
        seeds: SeedSpec,
    ) -> Self {
        Self::new_with_version(
            experiment_id,
            world,
            system,
            budget,
            seeds,
            ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2,
        )
    }

    fn new_with_version(
        experiment_id: impl Into<String>,
        world: WorldSpecRef,
        system: SystemSpec,
        budget: BudgetSpec,
        seeds: SeedSpec,
        schema_version: &str,
    ) -> Self {
        Self {
            schema_version: schema_version.to_owned(),
            experiment_id: experiment_id.into(),
            world,
            system,
            agents: Vec::new(),
            goals: Vec::new(),
            constraints: Vec::new(),
            budget,
            seeds,
            cognitive: None,
            hypotheses: Vec::new(),
            vary: Vec::new(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un eje de ablación.
    pub fn with_vary(mut self, axis: VaryAxisSpec) -> Self {
        self.vary.push(axis);
        self
    }

    fn is_v2(&self) -> bool {
        self.schema_version == ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2
    }

    /// Pre-registers a hypothesis for the experiment.
    pub fn with_hypothesis(mut self, hypothesis: HypothesisSpec) -> Self {
        self.hypotheses.push(hypothesis);
        self
    }

    /// Declara el bloque cognitivo opcional por defecto del experimento.
    pub fn with_cognitive(mut self, cognitive: CognitiveSpec) -> Self {
        self.cognitive = Some(cognitive);
        self
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
        if !EXPERIMENT_SCHEMA_VERSIONS.contains(&self.schema_version.as_str()) {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2,
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
        validate_hypotheses(&self.hypotheses)?;
        ensure_v2(self.is_v2(), !self.vary.is_empty(), "vary")?;
        let mut seen_vary = BTreeSet::new();
        for (i, axis) in self.vary.iter().enumerate() {
            axis.validate(&format!("vary[{i}]"))?;
            if !seen_vary.insert(axis.param.as_str()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "AdaptiveExperimentSpec.vary",
                    id: axis.param.clone(),
                });
            }
        }
        if let Some(cognitive) = &self.cognitive {
            // Referencias resueltas contra el sistema del experimento y la unión de sus agentes.
            let mut refs = CognitiveRefs::default().with_system(&self.system);
            for agent in &self.agents {
                refs.sensors
                    .extend(agent.sensors.iter().map(|s| s.sensor_id.clone()));
                refs.models.insert(agent.model.model_id.clone());
            }
            cognitive.validate(&refs)?;
        }
        Ok(())
    }

    /// Emite el libro de dotación canónico del experimento (valida antes de construirlo).
    pub fn endowment_book(&self) -> Result<EndowmentBook, SpecError> {
        self.validate()?;
        EndowmentBook::from_experiment(self)
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

/// Migra una especificación JSON de experimento (`v0` legacy o `v0.1`) hacia el contrato
/// tipado v0.2 ([`AdaptiveExperimentSpec`]).
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

    let expected_versions = format!(
        "{EXPERIMENT_SPEC_SCHEMA_VERSION} | {ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION} | {ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2}"
    );
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

    if found_version == ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2 {
        let spec =
            parse_adaptive_experiment_spec(&value, UnknownFieldPolicy::Reject).map_err(|err| {
                SpecMigrationError::new(
                    SpecDiagnostic::new("$", "invalid v0.2 adaptive experiment spec"),
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
                target_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned(),
                applied_steps: vec!["already_at_target_version".to_owned()],
                diagnostics: Vec::new(),
            },
        });
    }

    if found_version == ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION {
        let spec =
            parse_adaptive_experiment_spec(&value, UnknownFieldPolicy::Reject).map_err(|err| {
                SpecMigrationError::new(
                    SpecDiagnostic::new("$", "invalid v0.1 adaptive experiment spec"),
                    Some(err),
                )
            })?;
        let (spec, applied_steps, diagnostics) = lift_v01_to_v02(spec).map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "v0.1 -> v0.2 migration failed"),
                Some(err),
            )
        })?;
        spec.validate().map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "migrated experiment failed v0.2 validation"),
                Some(err),
            )
        })?;
        return Ok(MigratedExperimentSpec {
            spec,
            report: SchemaMigrationReport {
                source_version: found_version,
                target_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned(),
                applied_steps,
                diagnostics,
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

        // El contrato vigente es v0.2: se encadena el ascenso v0.1 -> v0.2 para que toda
        // migración termine en la misma versión objetivo.
        let (spec, lift_steps, lift_diagnostics) = lift_v01_to_v02(spec).map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "v0 -> v0.2 migration failed"),
                Some(err),
            )
        })?;
        applied_steps.extend(lift_steps);
        diagnostics.extend(lift_diagnostics);
        spec.validate().map_err(|err| {
            SpecMigrationError::new(
                SpecDiagnostic::new("$", "migrated experiment failed v0.2 validation"),
                Some(err),
            )
        })?;

        return Ok(MigratedExperimentSpec {
            spec,
            report: SchemaMigrationReport {
                source_version: found_version,
                target_version: ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned(),
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

/// Eleva una especificación v0.1 al contrato tipado v0.2.
///
/// Las construcciones que el DSL Kotlin/Python transportaba como cadenas en `config` se
/// convierten en campos tipados y se eliminan del mapa de configuración:
/// `mode = "pure"`, `privileged`, `observability = "hidden"`, `phenomena_json`, `vary_json` y
/// las claves epistémicas `question_json`/`ask_json`/`guard_json` de los nodos del sistema.
fn lift_v01_to_v02(
    mut experiment: AdaptiveExperimentSpec,
) -> Result<(AdaptiveExperimentSpec, Vec<String>, Vec<SpecDiagnostic>), SpecError> {
    let mut steps = vec!["upgrade_schema_version:v0.1->v0.2".to_owned()];
    let mut diagnostics = Vec::new();
    experiment.schema_version = ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned();

    let hidden_world = matches!(
        experiment.world.config.get("observability"),
        Some(SpecConfigValue::String(value)) if value == "hidden"
    );

    for (i, agent) in experiment.agents.iter_mut().enumerate() {
        agent.schema_version = ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2.to_owned();

        if matches!(
            agent.config.get("mode"),
            Some(SpecConfigValue::String(value)) if value == "pure"
        ) {
            agent.config.remove("mode");
            if agent.principle.is_none() {
                agent.principle = Some(PrincipleSpec {
                    kind: gogamen_core::SelectionPrinciple::Pure,
                    observables: Vec::new(),
                });
                diagnostics.push(SpecDiagnostic::new(
                    format!("$.agents[{i}].config.mode"),
                    "lifted legacy 'engine = pure' into agent.principle",
                ));
                steps.push("lift_agent_principle".to_owned());
            }
        }

        if matches!(
            agent.model.config.get("privileged"),
            Some(SpecConfigValue::Bool(true))
        ) {
            agent.model.config.remove("privileged");
            agent.model.privileged = true;
            diagnostics.push(SpecDiagnostic::new(
                format!("$.agents[{i}].model.config.privileged"),
                "lifted legacy privileged model flag into model.privileged",
            ));
            steps.push("lift_model_privilege".to_owned());
        }

        if hidden_world {
            agent.model_access = ModelAccess::Hidden;
        }

        let mut epistemic = agent.epistemic.clone().unwrap_or(EpistemicSpec {
            question: None,
            ask: None,
            guard: None,
        });
        let mut lifted_epistemic = false;
        for (j, node) in agent.system.nodes.iter_mut().enumerate() {
            for key in ["question_json", "ask_json", "guard_json"] {
                let Some(SpecConfigValue::String(raw)) = node.config.get(key).cloned() else {
                    continue;
                };
                let path = format!("$.agents[{i}].system.nodes[{j}].config.{key}");
                match key {
                    "question_json" => {
                        if epistemic.question.is_some() {
                            return Err(migration_invalid(&path, "duplicate question_json"));
                        }
                        epistemic.question = Some(EpistemicQuestionSpec::from_json(&raw)?);
                    }
                    "ask_json" => {
                        if epistemic.ask.is_some() {
                            return Err(migration_invalid(&path, "duplicate ask_json"));
                        }
                        epistemic.ask = Some(crate::epistemic::AskActionSpec::from_json(&raw)?);
                    }
                    "guard_json" => {
                        if epistemic.guard.is_some() {
                            return Err(migration_invalid(&path, "duplicate guard_json"));
                        }
                        epistemic.guard =
                            Some(crate::epistemic::GuardPolicySpec::from_json(&raw)?);
                    }
                    _ => unreachable!("epistemic config key set"),
                }
                node.config.remove(key);
                lifted_epistemic = true;
                diagnostics.push(SpecDiagnostic::new(
                    path,
                    format!("lifted legacy {key} into agent.epistemic"),
                ));
            }
        }
        if lifted_epistemic {
            agent.epistemic = Some(epistemic);
            steps.push("lift_epistemic_config".to_owned());
        }
    }

    if hidden_world {
        diagnostics.push(SpecDiagnostic::new(
            "$.world.config.observability",
            "mapped world observability 'hidden' to agent.model_access = hidden",
        ));
        steps.push("lift_world_model_access".to_owned());
    }

    if let Some(SpecConfigValue::String(raw)) = experiment.config.get("phenomena_json").cloned() {
        let phenomena = lift_phenomena_config(&raw, "$.config.phenomena_json")?;
        for agent in &mut experiment.agents {
            if agent.phenomena.is_empty() {
                agent.phenomena = phenomena.clone();
            }
        }
        experiment.config.remove("phenomena_json");
        diagnostics.push(SpecDiagnostic::new(
            "$.config.phenomena_json",
            format!(
                "lifted {} phenomena into agent.phenomena",
                phenomena.len()
            ),
        ));
        steps.push("lift_phenomena".to_owned());
    }

    if let Some(SpecConfigValue::String(raw)) = experiment.config.get("vary_json").cloned() {
        let value = parse_json(&raw)?;
        experiment.vary = parse_vary(&value, UnknownFieldPolicy::Reject, "vary")?;
        experiment.config.remove("vary_json");
        diagnostics.push(SpecDiagnostic::new(
            "$.config.vary_json",
            format!("lifted {} vary axes into experiment.vary", experiment.vary.len()),
        ));
        steps.push("lift_vary".to_owned());
    }

    if experiment.config.contains_key("readings_json") {
        diagnostics.push(SpecDiagnostic::new(
            "$.config.readings_json",
            "readings are not part of the typed contract yet; kept as legacy config",
        ));
    }

    Ok((experiment, steps, diagnostics))
}

fn migration_invalid(path: &str, reason: &str) -> SpecError {
    SpecError::InvalidFieldValue {
        context: "ExperimentSpecMigration",
        field: "<field>",
        reason: format!("{path}: {reason}"),
    }
}

/// Parsea un `phenomena_json` legado. El DSL Kotlin no declara `observable`; en ese caso se
/// deriva la primera variable de la expresión del predicado (o el id del fenómeno como último
/// recurso) para no inventar una medición inexistente.
fn lift_phenomena_config(raw: &str, path: &str) -> Result<Vec<PhenomenonSpec>, SpecError> {
    let mut value = parse_json(raw)?;
    if let JsonValue::Array(items) = &mut value {
        for (i, item) in items.iter_mut().enumerate() {
            let JsonValue::Object(map) = item else {
                return Err(migration_invalid(
                    &format!("{path}[{i}]"),
                    "expected JSON object",
                ));
            };
            if !map.contains_key("observable") {
                let id = match map.get("id") {
                    Some(JsonValue::String(id)) => id.clone(),
                    _ => String::new(),
                };
                let derived = ["expr", "trigger", "then"]
                    .iter()
                    .find_map(|key| map.get(*key).and_then(first_variable_json))
                    .unwrap_or(id);
                map.insert(
                    "observable".to_owned(),
                    JsonValue::String(derived),
                );
            }
        }
    }
    parse_phenomena(&value, UnknownFieldPolicy::Reject, path)
}

fn first_variable_json(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::Object(map) => {
            if matches!(
                map.get("op"),
                Some(JsonValue::String(op)) if op == "var"
            ) {
                if let Some(JsonValue::String(name)) = map.get("name") {
                    return Some(name.clone());
                }
            }
            map.values().find_map(first_variable_json)
        }
        JsonValue::Array(items) => items.iter().find_map(first_variable_json),
        _ => None,
    }
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

fn ensure_v2(v2: bool, present: bool, path: &str) -> Result<(), SpecError> {
    if present && !v2 {
        return Err(SpecError::TypedFieldRequiresV2 {
            path: path.to_owned(),
            expected: ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2,
        });
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
    if model.privileged {
        out.push_str(",\"privileged\":true");
    }
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
    if let Some(cognitive) = &agent.cognitive {
        out.push_str(",\"cognitive\":");
        cognitive.write_json(out)?;
    }
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
    out.push(']');
    if let Some(epistemic) = &agent.epistemic {
        out.push_str(",\"epistemic\":");
        out.push_str(&epistemic.to_canonical_json()?);
    }
    if let Some(learning) = &agent.learn {
        out.push_str(",\"learn\":{\"learning_rate\":");
        write_canonical_f64(out, learning.learning_rate)?;
        out.push_str(",\"model\":");
        write_json_string(out, &learning.model);
        out.push_str(",\"rule\":");
        write_json_string(out, learning.rule.as_str());
        let _ = write!(out, ",\"update_every_ticks\":{}}}", learning.update_every_ticks);
    }
    out.push_str(",\"model\":");
    write_model_spec_ref(out, &agent.model)?;
    if agent.model_access != ModelAccess::Visible {
        out.push_str(",\"model_access\":");
        write_json_string(out, agent.model_access.name());
    }
    if !agent.phenomena.is_empty() {
        out.push_str(",\"phenomena\":[");
        for (idx, phenomenon) in agent.phenomena.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            crate::endowment::write_phenomenon(out, phenomenon)?;
        }
        out.push(']');
    }
    if let Some(principle) = &agent.principle {
        out.push_str(",\"principle\":");
        crate::endowment::write_principle_spec(out, principle);
    }
    if let Some(scale) = &agent.scale {
        out.push_str(",\"scale\":");
        crate::endowment::write_scale_spec(out, scale);
    }
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
    if let Some(cognitive) = &exp.cognitive {
        out.push_str(",\"cognitive\":");
        cognitive.write_json(out)?;
    }
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
    out.push(']');
    if !exp.hypotheses.is_empty() {
        out.push_str(",\"hypotheses\":");
        out.push_str(&canonical_hypotheses_json(&exp.hypotheses)?);
    }
    out.push_str(",\"schema_version\":");
    write_json_string(out, &exp.schema_version);
    out.push_str(",\"seeds\":");
    write_seed_spec(out, &exp.seeds);
    out.push_str(",\"system\":");
    write_system_spec(out, &exp.system)?;
    if !exp.vary.is_empty() {
        out.push_str(",\"vary\":[");
        let mut axes: Vec<&VaryAxisSpec> = exp.vary.iter().collect();
        axes.sort_by(|a, b| a.param.cmp(&b.param));
        for (idx, axis) in axes.into_iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push_str("{\"param\":");
            write_json_string(out, &axis.param);
            out.push_str(",\"values\":[");
            for (j, value) in axis.values.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                write_canonical_f64(out, *value)?;
            }
            out.push_str("]}");
        }
        out.push(']');
    }
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
    reject_unknown_fields(
        map,
        &["model_id", "kind", "config", "privileged"],
        CTX,
        policy,
    )?;
    let privileged = match map.get("privileged") {
        Some(JsonValue::Bool(value)) => *value,
        Some(_) => {
            return Err(SpecError::InvalidFieldValue {
                context: CTX,
                field: "privileged",
                reason: "expected boolean".to_owned(),
            });
        }
        None => false,
    };
    Ok(ModelSpecRef {
        model_id: require_string(map, "model_id", CTX)?.to_owned(),
        kind: require_string(map, "kind", CTX)?.to_owned(),
        config: parse_optional_config(map, CTX)?,
        privileged,
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
            "cognitive",
            "principle",
            "scale",
            "learn",
            "phenomena",
            "model_access",
            "epistemic",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_any_schema_version(map, AGENT_SCHEMA_VERSIONS, CTX)?;
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

    let principle = map
        .get("principle")
        .map(|value| parse_principle_spec(value, policy, "agent.principle"))
        .transpose()?;
    let scale = map
        .get("scale")
        .map(|value| parse_scale_spec(value, policy, "agent.scale"))
        .transpose()?;
    let learn = map
        .get("learn")
        .map(|value| parse_learning_spec(value, policy, "agent.learn"))
        .transpose()?;
    let phenomena = match map.get("phenomena") {
        Some(value) => parse_phenomena(value, policy, "agent.phenomena")?,
        None => Vec::new(),
    };
    let model_access = match map.get("model_access") {
        Some(JsonValue::String(raw)) => ModelAccess::parse(raw).ok_or_else(|| {
            SpecError::UnknownVariant {
                path: "agent.model_access".to_owned(),
                value: raw.chars().take(64).collect(),
                expected: ModelAccess::EXPECTED,
            }
        })?,
        Some(_) => {
            return Err(SpecError::InvalidFieldValue {
                context: CTX,
                field: "model_access",
                reason: "expected string".to_owned(),
            });
        }
        None => ModelAccess::Visible,
    };
    let epistemic = map
        .get("epistemic")
        .map(|value| EpistemicSpec::from_json_value(value, policy))
        .transpose()?;

    Ok(AdaptiveAgentSpec {
        schema_version,
        agent_id,
        system,
        sensors,
        model,
        actions,
        goals,
        constraints,
        cognitive: parse_optional_cognitive(map, policy)?,
        principle,
        scale,
        learn,
        phenomena,
        model_access,
        epistemic,
        config,
    })
}

fn parse_optional_cognitive(
    map: &BTreeMap<String, JsonValue>,
    policy: UnknownFieldPolicy,
) -> Result<Option<CognitiveSpec>, SpecError> {
    map.get("cognitive")
        .map(|v| CognitiveSpec::from_json_value(v, policy))
        .transpose()
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
            "cognitive",
            "hypotheses",
            "vary",
            "config",
        ],
        CTX,
        policy,
    )?;

    let schema_version = require_any_schema_version(map, EXPERIMENT_SCHEMA_VERSIONS, CTX)?;
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
    let vary = match map.get("vary") {
        Some(value) => parse_vary(value, policy, "vary")?,
        None => Vec::new(),
    };

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
        cognitive: parse_optional_cognitive(map, policy)?,
        hypotheses: parse_hypotheses(map, policy)?,
        vary,
        config,
    })
}

fn require_any_schema_version(
    map: &BTreeMap<String, JsonValue>,
    allowed: &'static [&'static str],
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
    if !allowed.contains(&found.as_str()) {
        return Err(SpecError::IncompatibleSchemaVersion {
            expected: allowed.last().copied().unwrap_or(""),
            found,
        });
    }
    Ok(found)
}

/// Fixture Grid v0.2: principio dirigido, escala 7x7, aprendizaje, fenómeno e hipótesis.
///
/// Es el contrato versionado que ejercita el libro de dotación sobre el experimento Grid
/// canónico (mismo sistema y sensores que GOG-158, sin UI).
pub fn canonical_grid_endowment_experiment_fixture() -> AdaptiveExperimentSpec {
    let mut experiment = crate::cognitive::canonical_grid_cognitive_spec_experiment_fixture();
    experiment.experiment_id = "grid_endowment_v0_2".to_owned();
    experiment.schema_version = ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned();
    experiment.vary = vec![VaryAxisSpec {
        param: "cognitive.entropic.temperature".to_owned(),
        values: vec![0.25, 0.5],
    }];
    experiment.hypotheses = vec![HypothesisSpec::new(
        "h1_learning_improves",
        "planning on the learned model narrows the gap to the privileged ceiling",
        "success_rate",
        HypothesisComparator::Ge,
        0.8,
        HypothesisComparator::Lt,
        0.3,
    )];

    let agent = &mut experiment.agents[0];
    agent.schema_version = ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2.to_owned();
    agent.principle = Some(PrincipleSpec {
        kind: gogamen_core::SelectionPrinciple::Directed,
        observables: vec!["reward".to_owned()],
    });
    agent.scale = Some(ScaleSpec {
        scale_id: "grid.local_fov.25".to_owned(),
        binning_major: 1,
        binning_minor: 0,
        axes: vec![ScaleAxisSpec {
            min: 0.0,
            max: 6.0,
            bins: 25,
        }],
    });
    agent.learn = agent
        .cognitive
        .as_ref()
        .map(|cognitive| cognitive.learning.clone());
    agent.phenomena = vec![PhenomenonSpec {
        phenomenon_id: "goal_reached".to_owned(),
        observable: "reward".to_owned(),
        predicate: crate::endowment::PhenomenonPredicate::Eventually {
            expr: ExprAst::gt(
                ExprAst::var("reward").expect("valid var"),
                ExprAst::const_f64(0.0).expect("finite"),
            ),
            within_ticks: Some(200),
        },
    }];
    experiment
}

/// Exporta el fixture Grid v0.2 en JSON canónico determinista.
pub fn canonical_grid_endowment_experiment_fixture_json() -> String {
    canonical_grid_endowment_experiment_fixture()
        .to_canonical_json()
        .expect("canonical grid endowment fixture must serialize")
}

/// Fixture Kart v0.2: principio dirigido, escala de velocidad/dirección, fenómeno de rate.
pub fn canonical_kart_endowment_experiment_fixture() -> AdaptiveExperimentSpec {
    let mut experiment = crate::cognitive::canonical_kart_cognitive_spec_experiment_fixture();
    experiment.experiment_id = "kart_endowment_v0_2".to_owned();
    experiment.schema_version = ADAPTIVE_EXPERIMENT_SPEC_SCHEMA_VERSION_V2.to_owned();
    experiment.vary = vec![VaryAxisSpec {
        param: "cognitive.learning.learning_rate".to_owned(),
        values: vec![0.05, 0.1],
    }];
    experiment.hypotheses = vec![HypothesisSpec::new(
        "h2_collision_rate_bound",
        "the directed kart keeps collisions below the declared budget",
        "collision_rate",
        HypothesisComparator::Le,
        0.2,
        HypothesisComparator::Gt,
        0.5,
    )];

    let agent = &mut experiment.agents[0];
    agent.schema_version = ADAPTIVE_AGENT_SPEC_SCHEMA_VERSION_V2.to_owned();
    agent.principle = Some(PrincipleSpec {
        kind: gogamen_core::SelectionPrinciple::Directed,
        observables: vec!["collision".to_owned(), "progress".to_owned()],
    });
    agent.scale = Some(ScaleSpec {
        scale_id: "kart.speed_steering".to_owned(),
        binning_major: 1,
        binning_minor: 0,
        axes: vec![
            ScaleAxisSpec {
                min: 0.0,
                max: 20.0,
                bins: 16,
            },
            ScaleAxisSpec {
                min: -1.0,
                max: 1.0,
                bins: 8,
            },
        ],
    });
    agent.learn = agent
        .cognitive
        .as_ref()
        .map(|cognitive| cognitive.learning.clone());
    agent.phenomena = vec![PhenomenonSpec {
        phenomenon_id: "collision_free_lap".to_owned(),
        observable: "collision".to_owned(),
        predicate: crate::endowment::PhenomenonPredicate::Rate {
            expr: ExprAst::gt(
                ExprAst::var("collision").expect("valid var"),
                ExprAst::const_f64(0.0).expect("finite"),
            ),
            comparator: HypothesisComparator::Le,
            threshold: 0.1,
        },
    }];
    experiment
}

/// Exporta el fixture Kart v0.2 en JSON canónico determinista.
pub fn canonical_kart_endowment_experiment_fixture_json() -> String {
    canonical_kart_endowment_experiment_fixture()
        .to_canonical_json()
        .expect("canonical kart endowment fixture must serialize")
}
