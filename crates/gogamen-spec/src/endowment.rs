//! Libro de dotación (`EndowmentBook`) y campos tipados v2 de un experimento.
//!
//! # Qué es
//! La *dotación* (`experiment.endowment`) es todo lo que el agente recibe sin haberlo
//! aprendido, clasificado por origen:
//! - **Entorno** (`world`): sentidos (sensores) y actuadores (acciones) dados por el mundo.
//! - **Diseño** (`agent`): principio de selección, escala, aprendizaje y memoria/permisos.
//! - **Medido desde fuera** (`phenomena`): predicados sobre observables que el evaluador mide.
//! - **Afirmaciones** (`hypotheses`): hipótesis pre-registradas con su refutación.
//! - **Familia** (`vary`): ejes de ablación que expanden el experimento a una matriz.
//!
//! [`EndowmentBook::from_experiment`] materializa ese libro desde un
//! [`AdaptiveExperimentSpec`](crate::AdaptiveExperimentSpec) validado;
//! [`EndowmentBook::to_canonical_json`] emite JSON canónico (claves ordenadas, arrays
//! ordenados) y [`EndowmentBook::hash`] produce un hash FNV-1a estable sobre esos bytes con
//! etiqueta de dominio `gogamen.endowment.v1`. El hash viaja en
//! `gogamen_runtime::scientific_trace::TraceManifest` como trailer opcional, de modo que un
//! manifiesto sin dotación conserva sus bytes y checksum históricos.
//!
//! # Reutilización
//! No hay tipos duplicados: el principio es [`SelectionPrinciple`] de `gogamen-core`, la escala
//! se construye con [`Scale`]/[`ScaleIdentity`] (identidad `(id, versión, hash)` calculada por
//! el contrato de GOG-200), el aprendizaje es [`LearningSpec`], la memoria es [`MemorySpec`]
//! con [`MemoryPermission`] y las hipótesis son [`HypothesisSpec`].
//!
//! # Frontera
//! El libro es **dato declarativo**, nunca estado ni procedencia de evidencia. No concede
//! permisos: [`MemoryPermission`] declara, no autoriza. Los errores de validación llevan ruta
//! precisa (`agent.phenomena[0].observable`, `vary[0].values[1]`, ...) para que el autor sepa
//! qué declaración corregir.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use gogamen_core::{BinningVersion, Scale, ScaleIdentity, SelectionPrinciple, TraceHasher};

use crate::cognitive::{LearningRule, LearningSpec, MemorySpec};
use crate::error::{SpecError, UnknownFieldPolicy};
use crate::expr::{ExprAst, parse_expr_json, write_expr_json};
use crate::hypothesis::{
    HypothesisComparator, HypothesisSpec, canonical_hypotheses_json, validate_hypotheses,
};
use crate::json::{JsonNumber, JsonValue, write_canonical_f64, write_json_string};

/// Versión del contrato del libro de dotación.
pub const ENDOWMENT_BOOK_SCHEMA_VERSION: &str = "gogamen.endowment_book.v1";

/// Etiqueta de dominio del hash FNV-1a del libro de dotación.
const ENDOWMENT_HASH_DOMAIN: &str = "gogamen.endowment.v1";

// ---------------------------------------------------------------------------
// Acceso al mundo (modelAccess)
// ---------------------------------------------------------------------------

/// Modo de acceso del agente al estado del mundo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelAccess {
    /// El agente solo ve lo que sus sensores exponen.
    #[default]
    Visible,
    /// El agente no puede leer el estado del mundo: un modelo privilegiado es incompatible.
    Hidden,
}

impl ModelAccess {
    /// Etiqueta estable usada en JSON canónico.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Visible => "visible",
            Self::Hidden => "hidden",
        }
    }

    /// Variantes admitidas (error tipado con la lista cerrada).
    pub const EXPECTED: &'static [&'static str] = &["visible", "hidden"];

    /// Parsea la etiqueta estable; devuelve `None` si no pertenece al conjunto cerrado.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "visible" => Some(Self::Visible),
            "hidden" => Some(Self::Hidden),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Principio de selección declarado
// ---------------------------------------------------------------------------

/// Principio de selección y observables que la puntuación declara leer.
///
/// `observables` es la lista explícita de observables que entran en la puntuación. Un
/// principio [`SelectionPrinciple::Pure`] no puede leer un observable que un fenómeno mida:
/// esa combinación se rechaza con ruta `agent.phenomena[i].observable`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrincipleSpec {
    /// Principio de selección (tipo compartido con el planificador FMC).
    pub kind: SelectionPrinciple,
    /// Observables que la puntuación declara leer (sin duplicados, sin vacíos).
    pub observables: Vec<String>,
}

impl PrincipleSpec {
    /// Principio dirigido por defecto, sin observables declarados explícitamente.
    pub fn directed_default() -> Self {
        Self {
            kind: SelectionPrinciple::Directed,
            observables: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Escala declarada
// ---------------------------------------------------------------------------

/// Eje uniforme de una escala: `[min, max)` en `bins` intervalos.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleAxisSpec {
    /// Límite inferior finito.
    pub min: f64,
    /// Límite superior finito (`> min`).
    pub max: f64,
    /// Número de bins (`>= 1`).
    pub bins: u32,
}

/// Declaración de escala compartida por aprendiz y planificador (GOG-200).
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleSpec {
    /// Identificador estable de la escala.
    pub scale_id: String,
    /// Componente mayor de la versión de binning.
    pub binning_major: u32,
    /// Componente menor de la versión de binning.
    pub binning_minor: u32,
    /// Ejes numéricos en orden de declaración.
    pub axes: Vec<ScaleAxisSpec>,
}

impl ScaleSpec {
    /// Construye la [`Scale`] canónica; la identidad y el hash los calcula `gogamen-core`.
    pub fn to_scale(&self, base: &str) -> Result<Scale, SpecError> {
        if self.scale_id.trim().is_empty() {
            return Err(invalid(base, "scale_id must not be empty"));
        }
        if self.axes.is_empty() {
            return Err(invalid(base, "at least one axis is required"));
        }
        let mut axes = Vec::with_capacity(self.axes.len());
        for (i, axis) in self.axes.iter().enumerate() {
            check_f64(axis.min, &format!("{base}.axes[{i}].min"))?;
            check_f64(axis.max, &format!("{base}.axes[{i}].max"))?;
            if axis.bins == 0 {
                return Err(invalid(
                    &format!("{base}.axes[{i}].bins"),
                    "bins must be >= 1",
                ));
            }
            axes.push((axis.min, axis.max, axis.bins as usize));
        }
        Scale::uniform_grid(
            self.scale_id.clone(),
            BinningVersion::new(self.binning_major, self.binning_minor),
            axes,
        )
        .map_err(|err| invalid(base, &format!("invalid scale declaration: {err}")))
    }
}

// ---------------------------------------------------------------------------
// Fenómenos
// ---------------------------------------------------------------------------

/// Predicado de traza de un fenómeno (cuatro operadores de la capa de preguntas).
#[derive(Debug, Clone, PartialEq)]
pub enum PhenomenonPredicate {
    /// El predicado se cumple en todo tick.
    Always {
        /// Predicado booleano.
        expr: ExprAst,
    },
    /// El predicado se cumple alguna vez (opcionalmente dentro de una ventana).
    Eventually {
        /// Predicado booleano.
        expr: ExprAst,
        /// Ventana máxima en ticks (`>= 1`) cuando se declara.
        within_ticks: Option<u64>,
    },
    /// Si `trigger` se cumple, `then` se cumple después.
    After {
        /// Disparador.
        trigger: ExprAst,
        /// Consecuente.
        then: ExprAst,
    },
    /// La fracción de ticks que cumple `expr` satisface `comparator`/`threshold`.
    Rate {
        /// Predicado booleano.
        expr: ExprAst,
        /// Comparador aplicado a la fracción observada.
        comparator: HypothesisComparator,
        /// Umbral en `[0, 1]`.
        threshold: f64,
    },
}

/// Fenómeno declarado: un observable medido desde fuera con un predicado de traza.
#[derive(Debug, Clone, PartialEq)]
pub struct PhenomenonSpec {
    /// Identificador único dentro del agente.
    pub phenomenon_id: String,
    /// Observable que el fenómeno mide (nombre estable del evaluador).
    pub observable: String,
    /// Predicado de traza.
    pub predicate: PhenomenonPredicate,
}

impl PhenomenonSpec {
    pub(crate) fn validate(&self, base: &str) -> Result<(), SpecError> {
        if self.phenomenon_id.trim().is_empty() {
            return Err(invalid(&format!("{base}.id"), "phenomenon id must not be empty"));
        }
        if self.observable.trim().is_empty() {
            return Err(invalid(
                &format!("{base}.observable"),
                "phenomenon observable must not be empty",
            ));
        }
        match &self.predicate {
            PhenomenonPredicate::Always { expr } | PhenomenonPredicate::Eventually { expr, .. } => {
                expr.validate()?;
            }
            PhenomenonPredicate::After { trigger, then } => {
                trigger.validate()?;
                then.validate()?;
            }
            PhenomenonPredicate::Rate {
                expr, threshold, ..
            } => {
                expr.validate()?;
                check_f64(*threshold, &format!("{base}.threshold"))?;
                if !(0.0..=1.0).contains(threshold) {
                    return Err(invalid(
                        &format!("{base}.threshold"),
                        "rate threshold must be within [0, 1]",
                    ));
                }
            }
        }
        if let PhenomenonPredicate::Eventually {
            within_ticks: Some(0),
            ..
        } = &self.predicate
        {
            return Err(invalid(
                &format!("{base}.within_ticks"),
                "within_ticks must be >= 1",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Ejes de ablación (vary)
// ---------------------------------------------------------------------------

/// Eje de `vary`: un parámetro y los valores finitos que expanden la matriz.
#[derive(Debug, Clone, PartialEq)]
pub struct VaryAxisSpec {
    /// Parámetro ablacionado (p. ej. `entropic.temperature`).
    pub param: String,
    /// Valores a barrer (no vacío, todos finitos).
    pub values: Vec<f64>,
}

impl VaryAxisSpec {
    /// Valida el eje con la ruta indicada (`vary[i]`).
    pub fn validate(&self, base: &str) -> Result<(), SpecError> {
        if self.param.trim().is_empty() {
            return Err(invalid(&format!("{base}.param"), "vary param must not be empty"));
        }
        if self.values.is_empty() {
            return Err(invalid(
                &format!("{base}.values"),
                "vary axis requires at least one value",
            ));
        }
        for (j, value) in self.values.iter().enumerate() {
            check_f64(*value, &format!("{base}.values[{j}]"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Libro de dotación
// ---------------------------------------------------------------------------

/// Entorno que da la información: sensores y actuadores declarados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndowmentWorld {
    /// Identificador del mundo.
    pub world_id: String,
    /// Tipo de mundo.
    pub kind: String,
    /// Sensores (sentidos) ordenados por identificador.
    pub sensors: Vec<String>,
    /// Acciones (actuadores) ordenadas por identificador.
    pub actions: Vec<String>,
}

/// Fenómeno con el agente que lo declara.
#[derive(Debug, Clone, PartialEq)]
pub struct EndowmentPhenomenon {
    /// Agente declarante.
    pub agent_id: String,
    /// Fenómeno declarado.
    pub phenomenon: PhenomenonSpec,
}

/// Dotación de un agente: principio, escala, aprendizaje y memoria.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentEndowment {
    /// Identificador del agente.
    pub agent_id: String,
    /// Principio de selección declarado (dirigido por defecto).
    pub principle: PrincipleSpec,
    /// Identidad de la escala compartida, si se declaró.
    pub scale: Option<ScaleIdentity>,
    /// Aprendizaje declarado (`agent.learn` o `cognitive.learning`).
    pub learning: Option<LearningSpec>,
    /// Memoria declarada (`cognitive.memory`).
    pub memory: Option<MemorySpec>,
}

/// Libro de dotación canónico de un experimento.
#[derive(Debug, Clone, PartialEq)]
pub struct EndowmentBook {
    /// Versión del contrato ([`ENDOWMENT_BOOK_SCHEMA_VERSION`]).
    pub schema_version: String,
    /// Identificador del experimento.
    pub experiment_id: String,
    /// Entorno (dado).
    pub world: EndowmentWorld,
    /// Agentes ordenados por identificador (diseño).
    pub agents: Vec<AgentEndowment>,
    /// Fenómenos medidos desde fuera.
    pub phenomena: Vec<EndowmentPhenomenon>,
    /// Hipótesis pre-registradas.
    pub hypotheses: Vec<HypothesisSpec>,
    /// Ejes de ablación ordenados por parámetro.
    pub vary: Vec<VaryAxisSpec>,
}

impl EndowmentBook {
    /// Construye el libro desde un experimento ya validado.
    ///
    /// El resultado es determinista: agentes y fenómenos se ordenan por identificador,
    /// fenómenos por `(agent_id, phenomenon_id)`, hipótesis por `hypothesis_id` y ejes de
    /// `vary` por `param`, así que dos ejecuciones del mismo contrato producen el mismo hash.
    pub fn from_experiment(
        experiment: &crate::adaptive::AdaptiveExperimentSpec,
    ) -> Result<Self, SpecError> {
        let mut agents = Vec::with_capacity(experiment.agents.len());
        let mut phenomena = Vec::new();
        let mut sensors = BTreeSet::new();
        let mut actions = BTreeSet::new();

        for agent in &experiment.agents {
            sensors.extend(agent.sensors.iter().map(|s| s.sensor_id.clone()));
            actions.extend(agent.actions.iter().cloned());

            let principle = agent
                .principle
                .clone()
                .unwrap_or_else(PrincipleSpec::directed_default);
            let scale = match &agent.scale {
                Some(scale) => Some(scale.to_scale("agent.scale")?.identity()),
                None => None,
            };
            let learning = agent.learn.clone().or_else(|| {
                agent
                    .cognitive
                    .as_ref()
                    .map(|cognitive| cognitive.learning.clone())
            });
            let memory = agent
                .cognitive
                .as_ref()
                .map(|cognitive| cognitive.memory.clone())
                .or_else(|| {
                    experiment
                        .cognitive
                        .as_ref()
                        .map(|cognitive| cognitive.memory.clone())
                });

            for phenomenon in &agent.phenomena {
                phenomena.push(EndowmentPhenomenon {
                    agent_id: agent.agent_id.clone(),
                    phenomenon: phenomenon.clone(),
                });
            }

            agents.push(AgentEndowment {
                agent_id: agent.agent_id.clone(),
                principle,
                scale,
                learning,
                memory,
            });
        }

        agents.sort_by(|a, b| a.agent_id.cmp(&b.agent_id));
        phenomena.sort_by(|a, b| {
            a.agent_id
                .cmp(&b.agent_id)
                .then_with(|| a.phenomenon.phenomenon_id.cmp(&b.phenomenon.phenomenon_id))
        });
        let mut vary = experiment.vary.clone();
        vary.sort_by(|a, b| a.param.cmp(&b.param));

        Ok(Self {
            schema_version: ENDOWMENT_BOOK_SCHEMA_VERSION.to_owned(),
            experiment_id: experiment.experiment_id.clone(),
            world: EndowmentWorld {
                world_id: experiment.world.world_id.clone(),
                kind: experiment.world.kind.clone(),
                sensors: sensors.into_iter().collect(),
                actions: actions.into_iter().collect(),
            },
            agents,
            phenomena,
            hypotheses: experiment.hypotheses.clone(),
            vary,
        })
    }

    /// Valida el libro antes de emitirlo.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != ENDOWMENT_BOOK_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: ENDOWMENT_BOOK_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        if self.experiment_id.trim().is_empty() {
            return Err(invalid(
                "$.experiment_id",
                "experiment id must not be empty",
            ));
        }
        if self.world.world_id.trim().is_empty() {
            return Err(invalid("$.world.world_id", "world id must not be empty"));
        }
        let mut seen_agents = BTreeSet::new();
        for (i, agent) in self.agents.iter().enumerate() {
            let base = format!("$.agents[{i}]");
            if agent.agent_id.trim().is_empty() {
                return Err(invalid(&format!("{base}.agent_id"), "agent id must not be empty"));
            }
            if !seen_agents.insert(agent.agent_id.as_str()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "EndowmentBook.agents",
                    id: agent.agent_id.clone(),
                });
            }
            validate_principle(&agent.principle, &format!("{base}.principle"))?;
            if let Some(learning) = &agent.learning {
                validate_learning(learning, &format!("{base}.learning"))?;
            }
            if let Some(memory) = &agent.memory {
                if memory.capacity == 0 {
                    return Err(invalid(
                        &format!("{base}.memory.capacity"),
                        "capacity must be >= 1",
                    ));
                }
                if memory.permissions.is_empty() {
                    return Err(invalid(
                        &format!("{base}.memory.permissions"),
                        "at least one permission required",
                    ));
                }
            }
        }
        let mut seen_phenomena = BTreeSet::new();
        for (i, entry) in self.phenomena.iter().enumerate() {
            let base = format!("$.phenomena[{i}]");
            entry
                .phenomenon
                .validate(&format!("{base}.phenomenon"))?;
            if !seen_phenomena.insert((
                entry.agent_id.clone(),
                entry.phenomenon.phenomenon_id.clone(),
            )) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "EndowmentBook.phenomena",
                    id: entry.phenomenon.phenomenon_id.clone(),
                });
            }
        }
        validate_hypotheses(&self.hypotheses)?;
        let mut seen_params = BTreeSet::new();
        for (i, axis) in self.vary.iter().enumerate() {
            axis.validate(&format!("vary[{i}]"))?;
            if !seen_params.insert(axis.param.as_str()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "EndowmentBook.vary",
                    id: axis.param.clone(),
                });
            }
        }
        Ok(())
    }

    /// Emite el libro como JSON canónico con claves ordenadas y arrays deterministas.
    pub fn to_canonical_json(&self) -> Result<String, SpecError> {
        self.validate()?;
        let mut out = String::new();
        out.push_str("{\"agents\":[");
        for (i, agent) in self.agents.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_agent_endowment(&mut out, agent)?;
        }
        out.push_str("],\"experiment_id\":");
        write_json_string(&mut out, &self.experiment_id);
        out.push_str(",\"hypotheses\":");
        out.push_str(&canonical_hypotheses_json(&self.hypotheses)?);
        out.push_str(",\"phenomena\":[");
        for (i, entry) in self.phenomena.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"agent_id\":");
            write_json_string(&mut out, &entry.agent_id);
            out.push_str(",\"phenomenon\":");
            write_phenomenon(&mut out, &entry.phenomenon)?;
            out.push('}');
        }
        out.push_str("],\"schema_version\":");
        write_json_string(&mut out, &self.schema_version);
        out.push_str(",\"vary\":[");
        for (i, axis) in self.vary.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"param\":");
            write_json_string(&mut out, &axis.param);
            out.push_str(",\"values\":[");
            for (j, value) in axis.values.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                write_canonical_f64(&mut out, *value)?;
            }
            out.push_str("]}");
        }
        out.push_str("],\"world\":{\"actions\":[");
        for (i, action) in self.world.actions.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json_string(&mut out, action);
        }
        out.push_str("],\"kind\":");
        write_json_string(&mut out, &self.world.kind);
        out.push_str(",\"sensors\":[");
        for (i, sensor) in self.world.sensors.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json_string(&mut out, sensor);
        }
        out.push_str("],\"world_id\":");
        write_json_string(&mut out, &self.world.world_id);
        out.push_str("}}");
        Ok(out)
    }

    /// Hash FNV-1a estable sobre el JSON canónico del libro.
    pub fn hash(&self) -> Result<u64, SpecError> {
        let json = self.to_canonical_json()?;
        let mut hasher = TraceHasher::new();
        hasher.write_str(ENDOWMENT_HASH_DOMAIN);
        hasher.write_str(&json);
        Ok(hasher.finish())
    }
}

// ---------------------------------------------------------------------------
// Validación compartida y serialización interna
// ---------------------------------------------------------------------------

fn invalid(path: &str, reason: &str) -> SpecError {
    SpecError::InvalidFieldValue {
        context: "EndowmentSpec",
        field: "<field>",
        reason: format!("{path}: {reason}"),
    }
}

fn check_f64(value: f64, path: &str) -> Result<(), SpecError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SpecError::NonFiniteFloat {
            context: path.to_owned(),
        })
    }
}

pub(crate) fn validate_principle(principle: &PrincipleSpec, base: &str) -> Result<(), SpecError> {
    principle.kind.validate().map_err(|err| {
        invalid(
            &format!("{base}.kind"),
            &format!("selection principle '{}' cannot be used: {err}", principle.kind.name()),
        )
    })?;
    let mut seen = BTreeSet::new();
    for (i, observable) in principle.observables.iter().enumerate() {
        if observable.trim().is_empty() {
            return Err(invalid(
                &format!("{base}.observables[{i}]"),
                "observable name must not be empty",
            ));
        }
        if !seen.insert(observable.as_str()) {
            return Err(SpecError::DuplicateIdentifier {
                context: "PrincipleSpec.observables",
                id: observable.clone(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_learning(learning: &LearningSpec, base: &str) -> Result<(), SpecError> {
    if learning.model.trim().is_empty() {
        return Err(invalid(&format!("{base}.model"), "learning model must not be empty"));
    }
    check_f64(learning.learning_rate, &format!("{base}.learning_rate"))?;
    if !(learning.learning_rate > 0.0 && learning.learning_rate <= 1.0) {
        return Err(invalid(
            &format!("{base}.learning_rate"),
            "learning_rate must be in (0, 1]",
        ));
    }
    if learning.update_every_ticks == 0 {
        return Err(invalid(
            &format!("{base}.update_every_ticks"),
            "update_every_ticks must be >= 1",
        ));
    }
    Ok(())
}

pub(crate) fn write_principle_spec(out: &mut String, principle: &PrincipleSpec) {
    out.push_str("{\"kind\":");
    write_json_string(out, principle.kind.name());
    out.push_str(",\"observables\":[");
    let mut observables: Vec<&String> = principle.observables.iter().collect();
    observables.sort();
    for (i, observable) in observables.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_string(out, observable);
    }
    out.push_str("]}");
}

pub(crate) fn write_scale_spec(out: &mut String, scale: &ScaleSpec) {
    out.push_str("{\"axes\":[");
    for (i, axis) in scale.axes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{{\"bins\":{},\"max\":", axis.bins);
        // La validación previa garantiza valores finitos.
        let _ = write_canonical_f64(out, axis.max);
        out.push_str(",\"min\":");
        let _ = write_canonical_f64(out, axis.min);
        out.push('}');
    }
    let _ = write!(
        out,
        "],\"binning_version\":{{\"major\":{},\"minor\":{}}},\"scale_id\":",
        scale.binning_major, scale.binning_minor
    );
    write_json_string(out, &scale.scale_id);
    out.push('}');
}

fn write_agent_endowment(out: &mut String, agent: &AgentEndowment) -> Result<(), SpecError> {
    out.push_str("{\"agent_id\":");
    write_json_string(out, &agent.agent_id);
    if let Some(learning) = &agent.learning {
        out.push_str(",\"learning\":{\"learning_rate\":");
        write_canonical_f64(out, learning.learning_rate)?;
        out.push_str(",\"model\":");
        write_json_string(out, &learning.model);
        out.push_str(",\"rule\":");
        write_json_string(out, learning.rule.as_str());
        let _ = write!(out, ",\"update_every_ticks\":{}}}", learning.update_every_ticks);
    }
    if let Some(memory) = &agent.memory {
        let _ = write!(out, ",\"memory\":{{\"capacity\":{},\"kind\":", memory.capacity);
        write_json_string(out, memory.kind.as_str());
        out.push_str(",\"permissions\":[");
        for (i, permission) in memory.permissions.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            write_json_string(out, permission.as_str());
        }
        out.push_str("],\"port\":{\"node\":");
        write_json_string(out, &memory.port.node);
        out.push_str(",\"port\":");
        write_json_string(out, &memory.port.port);
        out.push_str("}}");
    }
    out.push_str(",\"principle\":{\"kind\":");
    write_json_string(out, agent.principle.kind.name());
    out.push_str(",\"observables\":[");
    let mut observables: Vec<&String> = agent.principle.observables.iter().collect();
    observables.sort();
    for (i, observable) in observables.into_iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_string(out, observable);
    }
    out.push_str("]}");
    if let Some(scale) = &agent.scale {
        out.push_str(",\"scale\":{\"binning_version\":{\"major\":");
        let _ = write!(out, "{}", scale.version().major());
        let _ = write!(out, ",\"minor\":{}}}", scale.version().minor());
        out.push_str(",\"scale_id\":");
        write_json_string(out, scale.id());
        let _ = write!(out, ",\"scale_hash\":{}}}", scale.hash());
    }
    out.push('}');
    Ok(())
}

pub(crate) fn write_phenomenon(out: &mut String, phenomenon: &PhenomenonSpec) -> Result<(), SpecError> {
    match &phenomenon.predicate {
        PhenomenonPredicate::Always { expr } => {
            out.push_str("{\"expr\":");
            write_expr_json(out, expr, 0)?;
            out.push_str(",\"id\":");
            write_json_string(out, &phenomenon.phenomenon_id);
            out.push_str(",\"kind\":\"always\",\"observable\":");
            write_json_string(out, &phenomenon.observable);
            out.push('}');
        }
        PhenomenonPredicate::Eventually { expr, within_ticks } => {
            out.push_str("{\"expr\":");
            write_expr_json(out, expr, 0)?;
            out.push_str(",\"id\":");
            write_json_string(out, &phenomenon.phenomenon_id);
            out.push_str(",\"kind\":\"eventually\",\"observable\":");
            write_json_string(out, &phenomenon.observable);
            if let Some(within) = within_ticks {
                let _ = write!(out, ",\"within_ticks\":{within}");
            }
            out.push('}');
        }
        PhenomenonPredicate::After { trigger, then } => {
            out.push_str("{\"id\":");
            write_json_string(out, &phenomenon.phenomenon_id);
            out.push_str(",\"kind\":\"after\",\"observable\":");
            write_json_string(out, &phenomenon.observable);
            out.push_str(",\"then\":");
            write_expr_json(out, then, 0)?;
            out.push_str(",\"trigger\":");
            write_expr_json(out, trigger, 0)?;
            out.push('}');
        }
        PhenomenonPredicate::Rate {
            expr,
            comparator,
            threshold,
        } => {
            out.push_str("{\"comparator\":");
            write_json_string(out, comparator.as_str());
            out.push_str(",\"expr\":");
            write_expr_json(out, expr, 0)?;
            out.push_str(",\"id\":");
            write_json_string(out, &phenomenon.phenomenon_id);
            out.push_str(",\"kind\":\"rate\",\"observable\":");
            write_json_string(out, &phenomenon.observable);
            out.push_str(",\"threshold\":");
            write_canonical_f64(out, *threshold)?;
            out.push('}');
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Parsers de campos tipados (JSON canónico del agente/experimento)
// ---------------------------------------------------------------------------

fn json_obj<'a>(
    value: &'a JsonValue,
    base: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, SpecError> {
    match value {
        JsonValue::Object(map) => Ok(map),
        _ => Err(invalid(base, "expected JSON object")),
    }
}

fn req_string<'a>(
    map: &'a BTreeMap<String, JsonValue>,
    field: &'static str,
    base: &str,
) -> Result<&'a str, SpecError> {
    match map.get(field) {
        Some(JsonValue::String(s)) => Ok(s.as_str()),
        Some(_) => Err(invalid(&format!("{base}.{field}"), "expected string")),
        None => Err(invalid(&format!("{base}.{field}"), "missing required field")),
    }
}

fn req_f64(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    base: &str,
) -> Result<f64, SpecError> {
    let path = format!("{base}.{field}");
    let value = match map.get(field) {
        Some(JsonValue::Number(JsonNumber::F64(v))) => *v,
        Some(JsonValue::Number(JsonNumber::I64(v))) => *v as f64,
        Some(JsonValue::Number(JsonNumber::U64(v))) => *v as f64,
        Some(_) => return Err(invalid(&path, "expected number")),
        None => return Err(invalid(&path, "missing required field")),
    };
    check_f64(value, &path)?;
    Ok(value)
}

fn req_u64(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    base: &str,
) -> Result<u64, SpecError> {
    match map.get(field) {
        Some(JsonValue::Number(JsonNumber::U64(v))) => Ok(*v),
        Some(JsonValue::Number(JsonNumber::I64(v))) if *v >= 0 => Ok(*v as u64),
        Some(_) => Err(invalid(
            &format!("{base}.{field}"),
            "expected non-negative integer",
        )),
        None => Err(invalid(&format!("{base}.{field}"), "missing required field")),
    }
}

fn req_u32(
    map: &BTreeMap<String, JsonValue>,
    field: &'static str,
    base: &str,
) -> Result<u32, SpecError> {
    let value = req_u64(map, field, base)?;
    u32::try_from(value)
        .map_err(|_| invalid(&format!("{base}.{field}"), "value exceeds u32"))
}

fn req_array<'a>(
    map: &'a BTreeMap<String, JsonValue>,
    field: &'static str,
    base: &str,
) -> Result<&'a Vec<JsonValue>, SpecError> {
    match map.get(field) {
        Some(JsonValue::Array(items)) => Ok(items),
        Some(_) => Err(invalid(&format!("{base}.{field}"), "expected array")),
        None => Err(invalid(&format!("{base}.{field}"), "missing required field")),
    }
}

/// Parsea `principle` desde JSON canónico.
pub fn parse_principle_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<PrincipleSpec, SpecError> {
    let map = json_obj(value, base)?;
    crate::codec::reject_unknown_fields(map, &["kind", "observables"], "PrincipleSpec", policy)?;
    let raw_kind = req_string(map, "kind", base)?;
    let kind = parse_selection_principle(raw_kind).ok_or_else(|| SpecError::UnknownVariant {
        path: format!("{base}.kind"),
        value: raw_kind.chars().take(64).collect(),
        expected: &["pure", "situated", "directed"],
    })?;
    let observables = req_array(map, "observables", base)?
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
            JsonValue::String(s) => Ok(s.clone()),
            _ => Err(invalid(
                &format!("{base}.observables[{i}]"),
                "expected string",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(PrincipleSpec { kind, observables })
}

/// Parsea `scale` desde JSON canónico.
pub fn parse_scale_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<ScaleSpec, SpecError> {
    let map = json_obj(value, base)?;
    crate::codec::reject_unknown_fields(
        map,
        &["scale_id", "binning_version", "axes"],
        "ScaleSpec",
        policy,
    )?;
    let version_value = map
        .get("binning_version")
        .ok_or_else(|| invalid(&format!("{base}.binning_version"), "missing required field"))?;
    let version = json_obj(version_value, &format!("{base}.binning_version"))?;
    let axes_value = req_array(map, "axes", base)?;
    let mut axes = Vec::with_capacity(axes_value.len());
    for (i, item) in axes_value.iter().enumerate() {
        let axis_path = format!("{base}.axes[{i}]");
        let axis = json_obj(item, &axis_path)?;
        axes.push(ScaleAxisSpec {
            min: req_f64(axis, "min", &axis_path)?,
            max: req_f64(axis, "max", &axis_path)?,
            bins: req_u32(axis, "bins", &axis_path)?,
        });
    }
    Ok(ScaleSpec {
        scale_id: req_string(map, "scale_id", base)?.to_owned(),
        binning_major: req_u32(version, "major", &format!("{base}.binning_version"))?,
        binning_minor: req_u32(version, "minor", &format!("{base}.binning_version"))?,
        axes,
    })
}

/// Parsea `phenomena` desde JSON canónico.
pub fn parse_phenomena(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<Vec<PhenomenonSpec>, SpecError> {
    let items = match value {
        JsonValue::Array(items) => items,
        _ => return Err(invalid(base, "expected array")),
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        out.push(parse_phenomenon(item, policy, &format!("{base}[{i}]"))?);
    }
    Ok(out)
}

fn parse_phenomenon(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<PhenomenonSpec, SpecError> {
    let map = json_obj(value, base)?;
    let kind = req_string(map, "kind", base)?.to_owned();
    let phenomenon_id = req_string(map, "id", base)?.to_owned();
    let observable = req_string(map, "observable", base)?.to_owned();
    let expr = |field: &'static str| -> Result<ExprAst, SpecError> {
        let raw = map
            .get(field)
            .ok_or_else(|| invalid(&format!("{base}.{field}"), "missing required field"))?;
        parse_expr_json(raw, policy, 0)
    };
    let predicate = match kind.as_str() {
        "always" => {
            crate::codec::reject_unknown_fields(
                map,
                &["id", "kind", "observable", "expr"],
                "PhenomenonSpec",
                policy,
            )?;
            PhenomenonPredicate::Always { expr: expr("expr")? }
        }
        "eventually" => {
            crate::codec::reject_unknown_fields(
                map,
                &["id", "kind", "observable", "expr", "within_ticks"],
                "PhenomenonSpec",
                policy,
            )?;
            let within_ticks = match map.get("within_ticks") {
                Some(_) => Some(req_u64(map, "within_ticks", base)?),
                None => None,
            };
            PhenomenonPredicate::Eventually {
                expr: expr("expr")?,
                within_ticks,
            }
        }
        "after" => {
            crate::codec::reject_unknown_fields(
                map,
                &["id", "kind", "observable", "trigger", "then"],
                "PhenomenonSpec",
                policy,
            )?;
            PhenomenonPredicate::After {
                trigger: expr("trigger")?,
                then: expr("then")?,
            }
        }
        "rate" => {
            crate::codec::reject_unknown_fields(
                map,
                &[
                    "id",
                    "kind",
                    "observable",
                    "expr",
                    "comparator",
                    "threshold",
                ],
                "PhenomenonSpec",
                policy,
            )?;
            let raw_cmp = req_string(map, "comparator", base)?;
            let comparator = HypothesisComparator::parse(raw_cmp).ok_or_else(|| {
                SpecError::UnknownVariant {
                    path: format!("{base}.comparator"),
                    value: raw_cmp.chars().take(64).collect(),
                    expected: &["gt", "ge", "lt", "le", "eq"],
                }
            })?;
            PhenomenonPredicate::Rate {
                expr: expr("expr")?,
                comparator,
                threshold: req_f64(map, "threshold", base)?,
            }
        }
        other => {
            return Err(SpecError::UnknownVariant {
                path: format!("{base}.kind"),
                value: other.chars().take(64).collect(),
                expected: &["always", "eventually", "after", "rate"],
            });
        }
    };
    Ok(PhenomenonSpec {
        phenomenon_id,
        observable,
        predicate,
    })
}

/// Parsea `vary` desde JSON canónico.
pub fn parse_vary(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<Vec<VaryAxisSpec>, SpecError> {
    let items = match value {
        JsonValue::Array(items) => items,
        _ => return Err(invalid(base, "expected array")),
    };
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let axis_path = format!("{base}[{i}]");
        let map = json_obj(item, &axis_path)?;
        crate::codec::reject_unknown_fields(map, &["param", "values"], "VaryAxisSpec", policy)?;
        let values = req_array(map, "values", &axis_path)?
            .iter()
            .enumerate()
            .map(|(j, raw)| match raw {
                JsonValue::Number(JsonNumber::F64(v)) => Ok(*v),
                JsonValue::Number(JsonNumber::I64(v)) => Ok(*v as f64),
                JsonValue::Number(JsonNumber::U64(v)) => Ok(*v as f64),
                _ => Err(invalid(
                    &format!("{axis_path}.values[{j}]"),
                    "expected number",
                )),
            })
            .collect::<Result<Vec<f64>, SpecError>>()?;
        for (j, value) in values.iter().enumerate() {
            check_f64(*value, &format!("{axis_path}.values[{j}]"))?;
        }
        out.push(VaryAxisSpec {
            param: req_string(map, "param", &axis_path)?.to_owned(),
            values,
        });
    }
    Ok(out)
}

/// Parsea `learn` (mismo contrato tipado que `cognitive.learning`).
pub fn parse_learning_spec(
    value: &JsonValue,
    policy: UnknownFieldPolicy,
    base: &str,
) -> Result<LearningSpec, SpecError> {
    let map = json_obj(value, base)?;
    crate::codec::reject_unknown_fields(
        map,
        &["rule", "model", "learning_rate", "update_every_ticks"],
        "LearningSpec",
        policy,
    )?;
    let raw_rule = req_string(map, "rule", base)?;
    let rule = LearningRule::parse(raw_rule, &format!("{base}.rule"))?;
    Ok(LearningSpec {
        rule,
        model: req_string(map, "model", base)?.to_owned(),
        learning_rate: req_f64(map, "learning_rate", base)?,
        update_every_ticks: req_u64(map, "update_every_ticks", base)?,
    })
}

fn parse_selection_principle(raw: &str) -> Option<SelectionPrinciple> {
    match raw {
        "pure" => Some(SelectionPrinciple::Pure),
        "situated" => Some(SelectionPrinciple::Situated),
        "directed" => Some(SelectionPrinciple::Directed),
        _ => None,
    }
}
