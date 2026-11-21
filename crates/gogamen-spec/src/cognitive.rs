//! `CognitiveSpec`: política entrópica, resolución, aprendizaje y memoria como datos tipados.
//!
//! Extiende el contrato declarativo de `gogamen-spec` de forma **aditiva y opcional**
//! (campo `cognitive` en [`AdaptiveAgentSpec`](crate::AdaptiveAgentSpec) y
//! [`AdaptiveExperimentSpec`](crate::AdaptiveExperimentSpec)); no introduce una IR rival ni
//! ejecuta callbacks. Es solo *dato validado*: ningún campo contiene código ni expresiones
//! evaluables, y el JSON canónico de specs sin `cognitive` queda byte a byte idéntico.
//!
//! # Seguridad (entrada no confiable)
//! - Conjuntos cerrados de variantes (`objective`, `rule`, `kind`, permisos): una variante
//!   desconocida produce [`SpecError::UnknownVariant`] con la ruta del campo.
//! - Todos los flotantes deben ser finitos y estar en rango ([`SpecError::NonFiniteFloat`]).
//! - Colecciones, cadenas y magnitudes acotadas (`MAX_*`); se comprueban **antes** de
//!   iterar ([`SpecError::LimitExceeded`]), es decir, fail-closed.
//! - Las referencias (sensor, modelo, puerto de memoria) deben existir y tener el tipo
//!   esperado en el agente/sistema ([`SpecError::InvalidReference`]).
//!
//! Este módulo solo describe *qué* se pide; no demuestra ni afirma ningún fenómeno de
//! dinámica entrópica multiescala: la evaluación empírica es trabajo posterior.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::PortType;

use crate::adaptive::{
    AdaptiveAgentSpec, AdaptiveExperimentSpec, BudgetSpec, ModelSpecRef, SeedSpec,
    SensorBindingSpec, WorldSpecRef, canonical_cognitive_grid_experiment_fixture,
};
use crate::codec::{as_object, reject_unknown_fields, require_schema_version};
use crate::error::{SpecError, UnknownFieldPolicy};
use crate::json::{JsonNumber, JsonValue, write_canonical_f64, write_json_string};
use crate::model::{SpecConfigValue, SystemConnectionSpec, SystemNodeSpec, SystemSpec};

/// Versión de esquema canónica del bloque cognitivo opcional.
pub const COGNITIVE_SPEC_SCHEMA_VERSION: &str = "gogamen.cognitive_spec.v0";
/// Máximo de niveles de resolución por especificación.
pub const MAX_RESOLUTION_LEVELS: usize = 16;
/// Máximo de bins por nivel de resolución.
pub const MAX_RESOLUTION_BINS: u32 = 65_536;
/// Máximo del horizonte de planificación (ticks).
pub const MAX_ENTROPIC_HORIZON: u32 = 4_096;
/// Máxima capacidad declarable de memoria (registros).
pub const MAX_MEMORY_CAPACITY: u32 = 1_000_000;
/// Máximo de bytes UTF-8 de cualquier identificador del bloque cognitivo.
pub const MAX_COGNITIVE_ID_BYTES: usize = 128;
/// Nombre del tipo de puerto personalizado que debe tener el puerto de memoria enlazado.
pub const MEMORY_PORT_TYPE_NAME: &str = "MemoryPort";

/// Objetivo entrópico declarado (conjunto cerrado; reutiliza conceptos FMC/entropía/empowerment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropicObjective {
    /// Planificación FMC con diversidad de trayectorias.
    Fmc,
    /// Maximización de entropía de acciones/estados futuros.
    Entropy,
    /// Empowerment (información mutua acción-estado futuro).
    Empowerment,
    /// Ganancia de información epistémica.
    InfoGain,
}

impl EntropicObjective {
    const NAMES: &'static [&'static str] = &["fmc", "entropy", "empowerment", "info_gain"];
    fn as_str(self) -> &'static str {
        match self {
            Self::Fmc => "fmc",
            Self::Entropy => "entropy",
            Self::Empowerment => "empowerment",
            Self::InfoGain => "info_gain",
        }
    }
    fn parse(s: &str, path: &str) -> Result<Self, SpecError> {
        match s {
            "fmc" => Ok(Self::Fmc),
            "entropy" => Ok(Self::Entropy),
            "empowerment" => Ok(Self::Empowerment),
            "info_gain" => Ok(Self::InfoGain),
            _ => Err(unknown_variant(path, s, Self::NAMES)),
        }
    }
}

/// Política entrópica tipada.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicPolicySpec {
    /// Objetivo entrópico.
    pub objective: EntropicObjective,
    /// Temperatura estrictamente positiva y finita.
    pub temperature: f64,
    /// Peso del término entrópico (`>= 0`, finito).
    pub entropy_weight: f64,
    /// Horizonte en ticks (`1..=MAX_ENTROPIC_HORIZON`).
    pub horizon: u32,
}

/// Nivel de resolución sobre un sensor del agente.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionLevelSpec {
    /// Identificador único del nivel.
    pub level_id: String,
    /// `sensor_id` referenciado (debe existir en el agente).
    pub sensor: String,
    /// Número de bins (`1..=MAX_RESOLUTION_BINS`).
    pub bins: u32,
}

/// Resolución multiescala: niveles ordenados de grueso a fino (bins estrictamente crecientes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolutionSpec {
    /// Versión del binning (etiqueta no vacía, p. ej. `"v0"`).
    pub binning_version: String,
    /// Niveles (`1..=MAX_RESOLUTION_LEVELS`).
    pub levels: Vec<ResolutionLevelSpec>,
}

/// Regla de aprendizaje declarada (conjunto cerrado).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LearningRule {
    /// Sin aprendizaje (modelo fijo).
    None,
    /// Actualización bayesiana.
    Bayesian,
    /// Diferencia temporal.
    TemporalDifference,
    /// Regla hebbiana.
    Hebbian,
}

impl LearningRule {
    const NAMES: &'static [&'static str] = &["none", "bayesian", "td", "hebbian"];
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bayesian => "bayesian",
            Self::TemporalDifference => "td",
            Self::Hebbian => "hebbian",
        }
    }
    pub(crate) fn parse(s: &str, path: &str) -> Result<Self, SpecError> {
        match s {
            "none" => Ok(Self::None),
            "bayesian" => Ok(Self::Bayesian),
            "td" => Ok(Self::TemporalDifference),
            "hebbian" => Ok(Self::Hebbian),
            _ => Err(unknown_variant(path, s, Self::NAMES)),
        }
    }
}

/// Aprendizaje tipado.
#[derive(Debug, Clone, PartialEq)]
pub struct LearningSpec {
    /// Regla de aprendizaje.
    pub rule: LearningRule,
    /// `model_id` referenciado (debe ser el modelo del agente).
    pub model: String,
    /// Tasa en `(0, 1]`, finita.
    pub learning_rate: f64,
    /// Periodo de actualización en ticks (`>= 1`).
    pub update_every_ticks: u64,
}

/// Tipo de memoria declarada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryKind {
    /// Memoria de trabajo.
    Working,
    /// Memoria episódica.
    Episodic,
}

impl MemoryKind {
    const NAMES: &'static [&'static str] = &["working", "episodic"];
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Episodic => "episodic",
        }
    }
    fn parse(s: &str, path: &str) -> Result<Self, SpecError> {
        match s {
            "working" => Ok(Self::Working),
            "episodic" => Ok(Self::Episodic),
            _ => Err(unknown_variant(path, s, Self::NAMES)),
        }
    }
}

/// Permiso declarado sobre la memoria (conjunto cerrado; es una *declaración*, no concede nada).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPermission {
    /// Lectura.
    Read,
    /// Escritura.
    Write,
    /// Anexado.
    Append,
}

impl MemoryPermission {
    const NAMES: &'static [&'static str] = &["read", "write", "append"];
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Append => "append",
        }
    }
    fn parse(s: &str, path: &str) -> Result<Self, SpecError> {
        match s {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "append" => Ok(Self::Append),
            _ => Err(unknown_variant(path, s, Self::NAMES)),
        }
    }
}

/// Referencia a un puerto de nodo del sistema que actúa como `MemoryPort`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPortRef {
    /// Nodo propietario.
    pub node: String,
    /// Puerto (debe tener tipo `Custom("MemoryPort")`).
    pub port: String,
}

/// Memoria tipada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorySpec {
    /// Tipo de memoria.
    pub kind: MemoryKind,
    /// Capacidad en registros (`1..=MAX_MEMORY_CAPACITY`).
    pub capacity: u32,
    /// Puerto del sistema enlazado.
    pub port: MemoryPortRef,
    /// Permisos declarados (sin duplicados, no vacío).
    pub permissions: Vec<MemoryPermission>,
}

/// Contexto contra el que se resuelven las referencias de un [`CognitiveSpec`].
#[derive(Debug, Clone, Default)]
pub struct CognitiveRefs {
    /// Puertos disponibles `(nodo, puerto) -> tipo`.
    pub ports: BTreeMap<(String, String), PortType>,
    /// Identificadores de sensores.
    pub sensors: BTreeSet<String>,
    /// Identificadores de modelos.
    pub models: BTreeSet<String>,
}

impl CognitiveRefs {
    /// Recoge los puertos de un [`SystemSpec`].
    pub fn with_system(mut self, system: &SystemSpec) -> Self {
        for node in &system.nodes {
            for port in &node.ports {
                self.ports.insert(
                    (node.id.to_string(), port.id.to_string()),
                    port.port_type.clone(),
                );
            }
        }
        self
    }
}

/// Bloque cognitivo opcional y versionado.
#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveSpec {
    /// Versión de esquema ([`COGNITIVE_SPEC_SCHEMA_VERSION`]).
    pub schema_version: String,
    /// Política entrópica.
    pub entropic: EntropicPolicySpec,
    /// Resolución multiescala.
    pub resolution: ResolutionSpec,
    /// Aprendizaje.
    pub learning: LearningSpec,
    /// Memoria.
    pub memory: MemorySpec,
}

fn unknown_variant(path: &str, value: &str, expected: &'static [&'static str]) -> SpecError {
    SpecError::UnknownVariant {
        path: path.to_owned(),
        value: truncate(value),
        expected,
    }
}

fn truncate(s: &str) -> String {
    s.chars().take(64).collect()
}

fn check_id(value: &str, path: &str) -> Result<(), SpecError> {
    if value.trim().is_empty() {
        return Err(SpecError::InvalidFieldValue {
            context: "CognitiveSpec",
            field: "identifier",
            reason: format!("{path} must not be empty"),
        });
    }
    if value.len() > MAX_COGNITIVE_ID_BYTES {
        return Err(SpecError::LimitExceeded {
            path: path.to_owned(),
            limit: MAX_COGNITIVE_ID_BYTES,
            found: value.len(),
        });
    }
    Ok(())
}

fn check_f64(v: f64, path: &str) -> Result<(), SpecError> {
    if v.is_finite() {
        Ok(())
    } else {
        Err(SpecError::NonFiniteFloat {
            context: path.to_owned(),
        })
    }
}

fn out_of_range(field: &'static str, path: &str, reason: &str) -> SpecError {
    SpecError::InvalidFieldValue {
        context: "CognitiveSpec",
        field,
        reason: format!("{path}: {reason}"),
    }
}

fn invalid_ref(path: &str, kind: &'static str, id: &str) -> SpecError {
    SpecError::InvalidReference {
        path: path.to_owned(),
        kind,
        id: truncate(id),
    }
}

impl CognitiveSpec {
    /// Valida rangos, límites y tipos **sin** resolver referencias.
    pub fn validate_shape(&self) -> Result<(), SpecError> {
        if self.schema_version != COGNITIVE_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: COGNITIVE_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        let e = &self.entropic;
        check_f64(e.temperature, "cognitive.entropic.temperature")?;
        if e.temperature <= 0.0 {
            return Err(out_of_range(
                "temperature",
                "cognitive.entropic.temperature",
                "must be > 0",
            ));
        }
        check_f64(e.entropy_weight, "cognitive.entropic.entropy_weight")?;
        if e.entropy_weight < 0.0 {
            return Err(out_of_range(
                "entropy_weight",
                "cognitive.entropic.entropy_weight",
                "must be >= 0",
            ));
        }
        if e.horizon == 0 || e.horizon > MAX_ENTROPIC_HORIZON {
            return Err(out_of_range(
                "horizon",
                "cognitive.entropic.horizon",
                &format!("must be in 1..={MAX_ENTROPIC_HORIZON}"),
            ));
        }

        let r = &self.resolution;
        check_id(&r.binning_version, "cognitive.resolution.binning_version")?;
        if r.levels.is_empty() {
            return Err(out_of_range(
                "levels",
                "cognitive.resolution.levels",
                "at least one level required",
            ));
        }
        if r.levels.len() > MAX_RESOLUTION_LEVELS {
            return Err(SpecError::LimitExceeded {
                path: "cognitive.resolution.levels".to_owned(),
                limit: MAX_RESOLUTION_LEVELS,
                found: r.levels.len(),
            });
        }
        let mut seen = BTreeSet::new();
        let mut prev_bins = 0u32;
        for (i, level) in r.levels.iter().enumerate() {
            let base = format!("cognitive.resolution.levels[{i}]");
            check_id(&level.level_id, &format!("{base}.level_id"))?;
            check_id(&level.sensor, &format!("{base}.sensor"))?;
            if !seen.insert(level.level_id.clone()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "CognitiveSpec.resolution.levels",
                    id: truncate(&level.level_id),
                });
            }
            if level.bins == 0 || level.bins > MAX_RESOLUTION_BINS {
                return Err(out_of_range(
                    "bins",
                    &format!("{base}.bins"),
                    &format!("must be in 1..={MAX_RESOLUTION_BINS}"),
                ));
            }
            if level.bins <= prev_bins {
                return Err(out_of_range(
                    "bins",
                    &format!("{base}.bins"),
                    "levels must be ordered coarse to fine (strictly increasing bins)",
                ));
            }
            prev_bins = level.bins;
        }

        let l = &self.learning;
        check_id(&l.model, "cognitive.learning.model")?;
        check_f64(l.learning_rate, "cognitive.learning.learning_rate")?;
        if !(l.learning_rate > 0.0 && l.learning_rate <= 1.0) {
            return Err(out_of_range(
                "learning_rate",
                "cognitive.learning.learning_rate",
                "must be in (0, 1]",
            ));
        }
        if l.update_every_ticks == 0 {
            return Err(out_of_range(
                "update_every_ticks",
                "cognitive.learning.update_every_ticks",
                "must be >= 1",
            ));
        }

        let m = &self.memory;
        if m.capacity == 0 || m.capacity > MAX_MEMORY_CAPACITY {
            return Err(out_of_range(
                "capacity",
                "cognitive.memory.capacity",
                &format!("must be in 1..={MAX_MEMORY_CAPACITY}"),
            ));
        }
        check_id(&m.port.node, "cognitive.memory.port.node")?;
        check_id(&m.port.port, "cognitive.memory.port.port")?;
        if m.permissions.is_empty() {
            return Err(out_of_range(
                "permissions",
                "cognitive.memory.permissions",
                "at least one permission required",
            ));
        }
        if m.permissions.len() > MemoryPermission::NAMES.len() {
            return Err(SpecError::LimitExceeded {
                path: "cognitive.memory.permissions".to_owned(),
                limit: MemoryPermission::NAMES.len(),
                found: m.permissions.len(),
            });
        }
        let mut seen_perm = BTreeSet::new();
        for p in &m.permissions {
            if !seen_perm.insert(*p) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "CognitiveSpec.memory.permissions",
                    id: p.as_str().to_owned(),
                });
            }
        }
        Ok(())
    }

    /// Valida forma y resuelve referencias (sensores, modelo, puerto de memoria) contra `refs`.
    pub fn validate(&self, refs: &CognitiveRefs) -> Result<(), SpecError> {
        self.validate_shape()?;
        for (i, level) in self.resolution.levels.iter().enumerate() {
            if !refs.sensors.contains(&level.sensor) {
                return Err(invalid_ref(
                    &format!("cognitive.resolution.levels[{i}].sensor"),
                    "sensor",
                    &level.sensor,
                ));
            }
        }
        if !refs.models.contains(&self.learning.model) {
            return Err(invalid_ref(
                "cognitive.learning.model",
                "model",
                &self.learning.model,
            ));
        }
        let key = (self.memory.port.node.clone(), self.memory.port.port.clone());
        match refs.ports.get(&key) {
            None => Err(invalid_ref(
                "cognitive.memory.port",
                "port",
                &format!("{}/{}", key.0, key.1),
            )),
            Some(PortType::Custom(name)) if name == MEMORY_PORT_TYPE_NAME => Ok(()),
            Some(other) => Err(SpecError::InvalidFieldValue {
                context: "CognitiveSpec",
                field: "port",
                reason: format!(
                    "cognitive.memory.port must have type Custom(\"{MEMORY_PORT_TYPE_NAME}\"), found {other:?}"
                ),
            }),
        }
    }

    /// Escribe el JSON canónico (claves ordenadas) del bloque.
    pub(crate) fn write_json(&self, out: &mut String) -> Result<(), SpecError> {
        let e = &self.entropic;
        out.push_str("{\"entropic\":{\"entropy_weight\":");
        write_canonical_f64(out, e.entropy_weight)?;
        let _ = write!(out, ",\"horizon\":{},\"objective\":", e.horizon);
        write_json_string(out, e.objective.as_str());
        out.push_str(",\"temperature\":");
        write_canonical_f64(out, e.temperature)?;
        let l = &self.learning;
        out.push_str("},\"learning\":{\"learning_rate\":");
        write_canonical_f64(out, l.learning_rate)?;
        out.push_str(",\"model\":");
        write_json_string(out, &l.model);
        out.push_str(",\"rule\":");
        write_json_string(out, l.rule.as_str());
        let _ = write!(out, ",\"update_every_ticks\":{}}}", l.update_every_ticks);
        let m = &self.memory;
        let _ = write!(out, ",\"memory\":{{\"capacity\":{},\"kind\":", m.capacity);
        write_json_string(out, m.kind.as_str());
        out.push_str(",\"permissions\":[");
        for (i, p) in m.permissions.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json_string(out, p.as_str());
        }
        out.push_str("],\"port\":{\"node\":");
        write_json_string(out, &m.port.node);
        out.push_str(",\"port\":");
        write_json_string(out, &m.port.port);
        out.push_str("}},\"resolution\":{\"binning_version\":");
        write_json_string(out, &self.resolution.binning_version);
        out.push_str(",\"levels\":[");
        for (i, lv) in self.resolution.levels.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            let _ = write!(out, "{{\"bins\":{},\"level_id\":", lv.bins);
            write_json_string(out, &lv.level_id);
            out.push_str(",\"sensor\":");
            write_json_string(out, &lv.sensor);
            out.push('}');
        }
        out.push_str("]},\"schema_version\":");
        write_json_string(out, &self.schema_version);
        out.push('}');
        Ok(())
    }

    /// Parsea (sin resolver referencias) un bloque cognitivo desde JSON ya parseado.
    pub(crate) fn from_json_value(
        value: &JsonValue,
        policy: UnknownFieldPolicy,
    ) -> Result<Self, SpecError> {
        const CTX: &str = "CognitiveSpec";
        let map = as_object(value, CTX)?;
        reject_unknown_fields(
            map,
            &[
                "schema_version",
                "entropic",
                "resolution",
                "learning",
                "memory",
            ],
            CTX,
            policy,
        )?;
        let schema_version = require_schema_version(map, COGNITIVE_SPEC_SCHEMA_VERSION, CTX)?;

        let em = sub_object(
            map,
            "entropic",
            "CognitiveSpec.entropic",
            policy,
            &["objective", "temperature", "entropy_weight", "horizon"],
        )?;
        let entropic = EntropicPolicySpec {
            objective: EntropicObjective::parse(
                &req_str(em, "objective", "cognitive.entropic")?,
                "cognitive.entropic.objective",
            )?,
            temperature: req_f64(em, "temperature", "cognitive.entropic")?,
            entropy_weight: req_f64(em, "entropy_weight", "cognitive.entropic")?,
            horizon: req_u32(em, "horizon", "cognitive.entropic")?,
        };

        let rm = sub_object(
            map,
            "resolution",
            "CognitiveSpec.resolution",
            policy,
            &["binning_version", "levels"],
        )?;
        let raw_levels = match rm.get("levels") {
            Some(JsonValue::Array(a)) => a,
            Some(_) => return Err(type_err("levels", "cognitive.resolution.levels", "array")),
            None => return Err(missing("levels", "CognitiveSpec.resolution")),
        };
        if raw_levels.len() > MAX_RESOLUTION_LEVELS {
            return Err(SpecError::LimitExceeded {
                path: "cognitive.resolution.levels".to_owned(),
                limit: MAX_RESOLUTION_LEVELS,
                found: raw_levels.len(),
            });
        }
        let mut levels = Vec::with_capacity(raw_levels.len());
        for (i, raw) in raw_levels.iter().enumerate() {
            let base = format!("cognitive.resolution.levels[{i}]");
            let lm = as_object(raw, "CognitiveSpec.resolution.levels")?;
            reject_unknown_fields(
                lm,
                &["level_id", "sensor", "bins"],
                "CognitiveSpec.resolution.levels",
                policy,
            )?;
            levels.push(ResolutionLevelSpec {
                level_id: req_str(lm, "level_id", &base)?,
                sensor: req_str(lm, "sensor", &base)?,
                bins: req_u32(lm, "bins", &base)?,
            });
        }
        let resolution = ResolutionSpec {
            binning_version: req_str(rm, "binning_version", "cognitive.resolution")?,
            levels,
        };

        let lm = sub_object(
            map,
            "learning",
            "CognitiveSpec.learning",
            policy,
            &["rule", "model", "learning_rate", "update_every_ticks"],
        )?;
        let learning = LearningSpec {
            rule: LearningRule::parse(
                &req_str(lm, "rule", "cognitive.learning")?,
                "cognitive.learning.rule",
            )?,
            model: req_str(lm, "model", "cognitive.learning")?,
            learning_rate: req_f64(lm, "learning_rate", "cognitive.learning")?,
            update_every_ticks: req_u64(lm, "update_every_ticks", "cognitive.learning")?,
        };

        let mm = sub_object(
            map,
            "memory",
            "CognitiveSpec.memory",
            policy,
            &["kind", "capacity", "port", "permissions"],
        )?;
        let raw_perms = match mm.get("permissions") {
            Some(JsonValue::Array(a)) => a,
            Some(_) => {
                return Err(type_err(
                    "permissions",
                    "cognitive.memory.permissions",
                    "array",
                ));
            }
            None => return Err(missing("permissions", "CognitiveSpec.memory")),
        };
        if raw_perms.len() > MemoryPermission::NAMES.len() {
            return Err(SpecError::LimitExceeded {
                path: "cognitive.memory.permissions".to_owned(),
                limit: MemoryPermission::NAMES.len(),
                found: raw_perms.len(),
            });
        }
        let mut permissions = Vec::with_capacity(raw_perms.len());
        for (i, raw) in raw_perms.iter().enumerate() {
            let path = format!("cognitive.memory.permissions[{i}]");
            let JsonValue::String(s) = raw else {
                return Err(type_err("permissions", &path, "string"));
            };
            permissions.push(MemoryPermission::parse(s, &path)?);
        }
        let pm = sub_object(
            mm,
            "port",
            "CognitiveSpec.memory.port",
            policy,
            &["node", "port"],
        )?;
        let memory = MemorySpec {
            kind: MemoryKind::parse(
                &req_str(mm, "kind", "cognitive.memory")?,
                "cognitive.memory.kind",
            )?,
            capacity: req_u32(mm, "capacity", "cognitive.memory")?,
            port: MemoryPortRef {
                node: req_str(pm, "node", "cognitive.memory.port")?,
                port: req_str(pm, "port", "cognitive.memory.port")?,
            },
            permissions,
        };

        Ok(Self {
            schema_version,
            entropic,
            resolution,
            learning,
            memory,
        })
    }
}

type JsonMap = BTreeMap<String, JsonValue>;

fn missing(field: &'static str, context: &'static str) -> SpecError {
    SpecError::MissingField { context, field }
}

fn type_err(field: &'static str, path: &str, expected: &str) -> SpecError {
    SpecError::InvalidFieldValue {
        context: "CognitiveSpec",
        field,
        reason: format!("{path}: expected {expected}"),
    }
}

fn sub_object<'a>(
    map: &'a JsonMap,
    field: &'static str,
    context: &'static str,
    policy: UnknownFieldPolicy,
    known: &[&str],
) -> Result<&'a JsonMap, SpecError> {
    let raw = map.get(field).ok_or(SpecError::MissingField {
        context: "CognitiveSpec",
        field,
    })?;
    let obj = as_object(raw, context)?;
    reject_unknown_fields(obj, known, context, policy)?;
    Ok(obj)
}

fn req_str(map: &JsonMap, field: &'static str, base: &str) -> Result<String, SpecError> {
    match map.get(field) {
        Some(JsonValue::String(s)) => {
            if s.len() > MAX_COGNITIVE_ID_BYTES {
                return Err(SpecError::LimitExceeded {
                    path: format!("{base}.{field}"),
                    limit: MAX_COGNITIVE_ID_BYTES,
                    found: s.len(),
                });
            }
            Ok(s.clone())
        }
        Some(_) => Err(type_err(field, &format!("{base}.{field}"), "string")),
        None => Err(SpecError::MissingField {
            context: "CognitiveSpec",
            field,
        }),
    }
}

fn req_f64(map: &JsonMap, field: &'static str, base: &str) -> Result<f64, SpecError> {
    let path = format!("{base}.{field}");
    let v = match map.get(field) {
        Some(JsonValue::Number(JsonNumber::F64(v))) => *v,
        Some(JsonValue::Number(JsonNumber::I64(n))) => *n as f64,
        Some(JsonValue::Number(JsonNumber::U64(n))) => *n as f64,
        Some(_) => return Err(type_err(field, &path, "number")),
        None => {
            return Err(SpecError::MissingField {
                context: "CognitiveSpec",
                field,
            });
        }
    };
    check_f64(v, &path)?;
    Ok(v)
}

fn req_u64(map: &JsonMap, field: &'static str, base: &str) -> Result<u64, SpecError> {
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::I64(n))) if *n >= 0 => Ok(*n as u64),
        Some(JsonValue::Number(JsonNumber::U64(n))) => Ok(*n),
        Some(_) => Err(type_err(
            field,
            &format!("{base}.{field}"),
            "non-negative integer",
        )),
        None => Err(SpecError::MissingField {
            context: "CognitiveSpec",
            field,
        }),
    }
}

fn req_u32(map: &JsonMap, field: &'static str, base: &str) -> Result<u32, SpecError> {
    let n = req_u64(map, field, base)?;
    u32::try_from(n).map_err(|_| out_of_range(field, &format!("{base}.{field}"), "exceeds u32"))
}

// ---------------------------------------------------------------------------
// Fixtures canónicos (Grid y Kart), solo datos
// ---------------------------------------------------------------------------

fn memory_node() -> SystemNodeSpec {
    SystemNodeSpec::new("memory_store", "memory")
        .with_output("memory", PortType::custom(MEMORY_PORT_TYPE_NAME))
}

fn sample_cognitive(sensor: &str, model: &str, bins: [u32; 3]) -> CognitiveSpec {
    let levels = ["coarse", "mid", "fine"]
        .iter()
        .zip(bins)
        .map(|(id, bins)| ResolutionLevelSpec {
            level_id: (*id).to_owned(),
            sensor: sensor.to_owned(),
            bins,
        })
        .collect();
    CognitiveSpec {
        schema_version: COGNITIVE_SPEC_SCHEMA_VERSION.to_owned(),
        entropic: EntropicPolicySpec {
            objective: EntropicObjective::Fmc,
            temperature: 0.5,
            entropy_weight: 0.25,
            horizon: 16,
        },
        resolution: ResolutionSpec {
            binning_version: "v0".to_owned(),
            levels,
        },
        learning: LearningSpec {
            rule: LearningRule::Bayesian,
            model: model.to_owned(),
            learning_rate: 0.1,
            update_every_ticks: 4,
        },
        memory: MemorySpec {
            kind: MemoryKind::Episodic,
            capacity: 256,
            port: MemoryPortRef {
                node: "memory_store".to_owned(),
                port: "memory".to_owned(),
            },
            permissions: vec![MemoryPermission::Read, MemoryPermission::Append],
        },
    }
}

/// Fixture Grid: el experimento cognitivo canónico más un [`CognitiveSpec`] por agente.
pub fn canonical_grid_cognitive_spec_experiment_fixture() -> AdaptiveExperimentSpec {
    let mut exp = canonical_cognitive_grid_experiment_fixture();
    exp.experiment_id = "grid_cognitive_spec_v0".to_owned();
    exp.system.nodes.push(memory_node());
    for agent in &mut exp.agents {
        agent.system.nodes.push(memory_node());
        agent.cognitive = Some(sample_cognitive(
            "local_fov_sensor",
            "pomdp_belief_v0",
            [3, 9, 25],
        ));
    }
    exp.cognitive = exp.agents[0].cognitive.clone();
    exp
}

/// Fixture Kart: mundo `kart` con sensor de rango, acciones de conducción y [`CognitiveSpec`].
pub fn canonical_kart_cognitive_spec_experiment_fixture() -> AdaptiveExperimentSpec {
    let system = SystemSpec::new("kart_cognitive_system")
        .with_node(
            SystemNodeSpec::new("kart_env", "kart")
                .with_output("rays", PortType::custom("RangeScan"))
                .with_input("control", PortType::String),
        )
        .with_node(
            SystemNodeSpec::new("kart_planner", "fmc_planner")
                .with_input("rays", PortType::custom("RangeScan"))
                .with_output("control", PortType::String),
        )
        .with_node(memory_node())
        .with_connection(SystemConnectionSpec::instant(
            "kart_env",
            "rays",
            "kart_planner",
            "rays",
        ))
        .with_connection(SystemConnectionSpec::delayed(
            "kart_planner",
            "control",
            "kart_env",
            "control",
            1,
        ));
    let sensor = SensorBindingSpec::new(
        "range_sensor",
        "range_scan",
        "kart_env",
        "rays",
        PortType::custom("RangeScan"),
    );
    let agent = AdaptiveAgentSpec::new(
        "kart_driver",
        system.clone(),
        ModelSpecRef::new("kart_dynamics_v0", "dynamics"),
    )
    .with_sensor(sensor)
    .with_action("steer_left")
    .with_action("steer_right")
    .with_action("throttle")
    .with_action("brake")
    .with_cognitive(sample_cognitive(
        "range_sensor",
        "kart_dynamics_v0",
        [4, 16, 64],
    ));
    AdaptiveExperimentSpec::new(
        "kart_cognitive_spec_v0",
        WorldSpecRef::new("kart_track_0", "kart"),
        system,
        BudgetSpec::new(2000, 5000, 20_000),
        SeedSpec::new(vec![1, 2], vec![100]),
    )
    .with_agent(agent)
    .with_config("headless", SpecConfigValue::Bool(true))
}
