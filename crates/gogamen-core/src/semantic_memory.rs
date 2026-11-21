//! Memoria semántica de conocimiento revisable y resolución de grounding (`std`-only).
//!
//! Este módulo provee un repositorio nativo en memoria sin dependencias externas (ni DB ni LLM)
//! para representar entidades semánticas, proposiciones revisables con procedencia auditable
//! y resolución determinista de candidatos de grounding:
//! - [`SemanticEntity`]: entidad identificada por [`EntityId`] con [`EntityKind`], etiquetas y
//!   atributos estructurados para grounding.
//! - [`PropositionKey`] y [`EpistemicStatus`]: distinguen explícitamente hechos propuestos
//!   ([`EpistemicStatus::ProposedFact`]), hipótesis ([`EpistemicStatus::Hypothesis`]),
//!   observaciones corroboradas ([`EpistemicStatus::CorroboratedObservation`]) y conocimiento
//!   en disputa o revisado ([`EpistemicStatus::Contested`], [`EpistemicStatus::Revised`]).
//! - [`PropositionRecord`]: mantiene creencia binaria calibrada ([`BinaryBelief`]), número de
//!   revisión monotónico e historial append-only de [`EvidenceEntry`] (`Supporting` o
//!   `Contradicting`), de modo que la evidencia contradictoria revisa la creencia **sin destruir
//!   la procedencia previa**.
//! - [`SemanticKnowledgeStore`]: operaciones `upsert_entity`, `record_proposition`,
//!   `add_evidence`, `query_by_entity` y `resolve_grounding_candidates`.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::belief::{BeliefError, BeliefUpdateStatus, BinaryBelief, LikelihoodModel};
use crate::clock::MonotonicInstant;
use crate::observability::EntityId;
use crate::provenance::{Confidence, EvidenceRef, ObservationKind, SensorId};

/// Errores explícitos de validación y operación en memoria semántica revisable.
#[derive(Debug, Clone, PartialEq)]
pub enum SemanticMemoryError {
    /// Un identificador textual estructurado está vacío o solo contiene espacios.
    EmptyIdentifier {
        /// Nombre del campo inválido.
        field: &'static str,
    },
    /// Un umbral o parámetro de probabilidad no es finito o está fuera de `[0.0, 1.0]`.
    InvalidProbability {
        /// Nombre del parámetro inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Ya existe una proposición registrada para la misma [`PropositionKey`];
    /// debe revisarse mediante `add_evidence` para preservar su procedencia.
    PropositionAlreadyExists,
    /// No existe ninguna proposición registrada para la [`PropositionKey`] consultada.
    PropositionNotFound,
    /// Error propagado desde el subsistema de creencias discretas ([`BeliefError`]).
    Belief(BeliefError),
}

impl fmt::Display for SemanticMemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { field } => {
                write!(f, "semantic memory identifier '{field}' cannot be empty")
            }
            Self::InvalidProbability { field, value } => {
                write!(
                    f,
                    "invalid semantic memory probability '{field}' ({value}): must be finite and within [0.0, 1.0]"
                )
            }
            Self::PropositionAlreadyExists => {
                write!(
                    f,
                    "proposition already exists; use add_evidence to revise without destroying provenance"
                )
            }
            Self::PropositionNotFound => {
                write!(f, "proposition not found in semantic knowledge store")
            }
            Self::Belief(err) => write!(f, "semantic memory belief error: {err}"),
        }
    }
}

impl Error for SemanticMemoryError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Belief(err) => Some(err),
            _ => None,
        }
    }
}

impl From<BeliefError> for SemanticMemoryError {
    fn from(err: BeliefError) -> Self {
        Self::Belief(err)
    }
}

fn validate_non_empty(
    field: &'static str,
    value: impl Into<String>,
) -> Result<String, SemanticMemoryError> {
    let s = value.into();
    if s.trim().is_empty() {
        return Err(SemanticMemoryError::EmptyIdentifier { field });
    }
    Ok(s)
}

/// Tipo o categoría estructurada de una [`SemanticEntity`] (por ejemplo `"door"`, `"agent"`, `"zone"`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityKind(String);

impl EntityKind {
    /// Construye un [`EntityKind`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(kind: impl Into<String>) -> Result<Self, SemanticMemoryError> {
        Ok(Self(validate_non_empty("entity_kind", kind)?))
    }

    /// Devuelve la representación textual de la categoría.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador estructurado de predicado semántico (por ejemplo `"is_open"`, `"located_in"`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PredicateId(String);

impl PredicateId {
    /// Construye un [`PredicateId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(predicate: impl Into<String>) -> Result<Self, SemanticMemoryError> {
        Ok(Self(validate_non_empty("predicate", predicate)?))
    }

    /// Devuelve la representación textual del predicado.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PredicateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Valor estructurado para objetos de proposiciones y atributos de entidades semánticas.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PropositionValue {
    /// Referencia relacional a otra entidad por [`EntityId`].
    Entity(EntityId),
    /// Valor booleano discreto.
    Bool(bool),
    /// Valor entero discreto.
    Int(i64),
    /// Símbolo o etiqueta estructurada no vacía.
    Symbol(String),
}

impl PropositionValue {
    /// Crea un valor simbólico validando que no esté vacío ni contenga solo espacios.
    pub fn symbol(value: impl Into<String>) -> Result<Self, SemanticMemoryError> {
        Ok(Self::Symbol(validate_non_empty("symbol", value)?))
    }
}

impl fmt::Display for PropositionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entity(id) => write!(f, "{id}"),
            Self::Bool(v) => write!(f, "{v}"),
            Self::Int(v) => write!(f, "{v}"),
            Self::Symbol(s) => f.write_str(s),
        }
    }
}

/// Entidad semántica registrada para grounding, con categoría, etiquetas y atributos estructurados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SemanticEntity {
    id: EntityId,
    kind: EntityKind,
    tags: BTreeSet<String>,
    attributes: BTreeMap<String, PropositionValue>,
}

impl SemanticEntity {
    /// Crea una nueva [`SemanticEntity`] sin etiquetas ni atributos iniciales.
    pub fn new(id: EntityId, kind: EntityKind) -> Self {
        Self {
            id,
            kind,
            tags: BTreeSet::new(),
            attributes: BTreeMap::new(),
        }
    }

    /// Añade una etiqueta estructurada no vacía a la entidad.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Result<Self, SemanticMemoryError> {
        let tag = validate_non_empty("tag", tag)?;
        self.tags.insert(tag);
        Ok(self)
    }

    /// Añade o actualiza un atributo estructurado `(clave, valor)` en la entidad.
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: PropositionValue,
    ) -> Result<Self, SemanticMemoryError> {
        let name = validate_non_empty("attribute_name", name)?;
        self.attributes.insert(name, value);
        Ok(self)
    }

    /// Devuelve el [`EntityId`] de la entidad.
    pub const fn id(&self) -> EntityId {
        self.id
    }

    /// Devuelve el [`EntityKind`] de la entidad.
    pub const fn kind(&self) -> &EntityKind {
        &self.kind
    }

    /// Devuelve el conjunto ordenado de etiquetas de la entidad.
    pub const fn tags(&self) -> &BTreeSet<String> {
        &self.tags
    }

    /// Indica si la entidad posee la etiqueta `tag`.
    pub fn has_tag(&self, tag: &str) -> bool {
        self.tags.contains(tag)
    }

    /// Devuelve el mapa ordenado de atributos estructurados de la entidad.
    pub const fn attributes(&self) -> &BTreeMap<String, PropositionValue> {
        &self.attributes
    }

    /// Obtiene el valor de un atributo estructurado por nombre, si existe.
    pub fn attribute(&self, name: &str) -> Option<&PropositionValue> {
        self.attributes.get(name)
    }
}

/// Clave canónica de una proposición semántica `(subject, predicate, object)`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropositionKey {
    /// Entidad sujeto de la proposición.
    pub subject: EntityId,
    /// Predicado semántico.
    pub predicate: PredicateId,
    /// Valor u objeto de la proposición.
    pub object: PropositionValue,
}

impl PropositionKey {
    /// Crea una nueva [`PropositionKey`].
    pub const fn new(subject: EntityId, predicate: PredicateId, object: PropositionValue) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

/// Estado epistémico explícito de una proposición en memoria semántica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EpistemicStatus {
    /// Hecho propuesto (aún no corroborado por observación independiente).
    ProposedFact,
    /// Hipótesis tentativa sujeta a confirmación o refutación.
    Hypothesis,
    /// Observación corroborada por evidencia de soporte.
    CorroboratedObservation,
    /// Proposición con evidencia contradictoria registrada, en disputa abierta.
    Contested,
    /// Proposición revisada/revertida tras evidencia contradictoria que redujo la creencia por debajo de `0.5`.
    Revised,
}

impl EpistemicStatus {
    /// Devuelve `true` si el estado es [`EpistemicStatus::ProposedFact`].
    pub const fn is_proposed_fact(self) -> bool {
        matches!(self, Self::ProposedFact)
    }

    /// Devuelve `true` si el estado es [`EpistemicStatus::Hypothesis`].
    pub const fn is_hypothesis(self) -> bool {
        matches!(self, Self::Hypothesis)
    }

    /// Devuelve `true` si el estado es [`EpistemicStatus::CorroboratedObservation`].
    pub const fn is_corroborated(self) -> bool {
        matches!(self, Self::CorroboratedObservation)
    }

    /// Devuelve `true` si el estado es [`EpistemicStatus::Contested`].
    pub const fn is_contested(self) -> bool {
        matches!(self, Self::Contested)
    }

    /// Devuelve `true` si el estado es [`EpistemicStatus::Revised`].
    pub const fn is_revised(self) -> bool {
        matches!(self, Self::Revised)
    }

    /// Devuelve `true` si existe contradicción registrada (`Contested` o `Revised`).
    pub const fn has_contradiction(self) -> bool {
        matches!(self, Self::Contested | Self::Revised)
    }
}

impl fmt::Display for EpistemicStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::ProposedFact => "proposed_fact",
            Self::Hypothesis => "hypothesis",
            Self::CorroboratedObservation => "corroborated_observation",
            Self::Contested => "contested",
            Self::Revised => "revised",
        };
        f.write_str(label)
    }
}

/// Polaridad de una evidencia respecto a una proposición: soporte o contradicción.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidencePolarity {
    /// Evidencia que apoya la veracidad de la proposición.
    Supporting,
    /// Evidencia que contradice la proposición.
    Contradicting,
}

impl EvidencePolarity {
    /// Devuelve `true` si la evidencia es de soporte.
    pub const fn is_supporting(self) -> bool {
        matches!(self, Self::Supporting)
    }

    /// Devuelve `true` si la evidencia es contradictoria.
    pub const fn is_contradicting(self) -> bool {
        matches!(self, Self::Contradicting)
    }
}

impl fmt::Display for EvidencePolarity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Supporting => "supporting",
            Self::Contradicting => "contradicting",
        };
        f.write_str(label)
    }
}

/// Entrada de evidencia proporcionada al registrar o revisar una proposición.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceInput {
    /// Referencia única a la evidencia de procedencia.
    pub evidence_ref: EvidenceRef,
    /// Instante monotónico en que se observó la evidencia.
    pub observed_at: MonotonicInstant,
    /// Identificador del sensor o fuente emisora.
    pub sensor_id: SensorId,
    /// Polaridad respecto a la proposición (`Supporting` o `Contradicting`).
    pub polarity: EvidencePolarity,
    /// Clasificación epistémica de la observación fuente.
    pub claim_kind: ObservationKind,
    /// Confianza escalar reportada por la fuente (`[0.0, 1.0]`).
    pub confidence: Confidence,
}

impl EvidenceInput {
    /// Construye una nueva [`EvidenceInput`].
    pub const fn new(
        evidence_ref: EvidenceRef,
        observed_at: MonotonicInstant,
        sensor_id: SensorId,
        polarity: EvidencePolarity,
        claim_kind: ObservationKind,
        confidence: Confidence,
    ) -> Self {
        Self {
            evidence_ref,
            observed_at,
            sensor_id,
            polarity,
            claim_kind,
            confidence,
        }
    }
}

/// Registro inmutable en el historial append-only de evidencia de una [`PropositionRecord`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceEntry {
    /// Referencia única a la evidencia de procedencia.
    pub evidence_ref: EvidenceRef,
    /// Instante monotónico en que se observó la evidencia.
    pub observed_at: MonotonicInstant,
    /// Identificador del sensor o fuente emisora.
    pub sensor_id: SensorId,
    /// Polaridad respecto a la proposición (`Supporting` o `Contradicting`).
    pub polarity: EvidencePolarity,
    /// Clasificación epistémica de la observación fuente.
    pub claim_kind: ObservationKind,
    /// Confianza escalar reportada por la fuente (`[0.0, 1.0]`).
    pub confidence: Confidence,
    /// Revisión monotónica de la proposición en la que se registró esta evidencia.
    pub revision: u64,
}

impl EvidenceEntry {
    const fn from_input(input: EvidenceInput, revision: u64) -> Self {
        Self {
            evidence_ref: input.evidence_ref,
            observed_at: input.observed_at,
            sensor_id: input.sensor_id,
            polarity: input.polarity,
            claim_kind: input.claim_kind,
            confidence: input.confidence,
            revision,
        }
    }
}

/// Resultado explícito de incorporar nueva evidencia a una proposición existente.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RevisionStatus {
    /// La evidencia se anexó al historial y revisó la creencia y/o estado epistémico.
    Updated,
    /// La referencia [`EvidenceRef`] ya existía en el historial y se ignoró sin mutar estado.
    DuplicateEvidenceIgnored,
    /// La evidencia se anexó al historial y marcó la proposición como en disputa, preservando
    /// la creencia previa al ser la verosimilitud marginal cero (`P(obs) == 0.0`).
    ConflictPreservedPriorBelief,
}

impl RevisionStatus {
    /// Devuelve `true` si la evidencia actualizó la creencia.
    pub const fn is_updated(self) -> bool {
        matches!(self, Self::Updated)
    }

    /// Devuelve `true` si la evidencia fue ignorada por duplicado.
    pub const fn is_duplicate(self) -> bool {
        matches!(self, Self::DuplicateEvidenceIgnored)
    }

    /// Devuelve `true` si hubo conflicto de verosimilitud marginal cero preservando el prior.
    pub const fn is_conflict(self) -> bool {
        matches!(self, Self::ConflictPreservedPriorBelief)
    }
}

/// Registro revisable de una proposición semántica con creencia calibrada y procedencia append-only.
#[derive(Debug, Clone, PartialEq)]
pub struct PropositionRecord {
    key: PropositionKey,
    status: EpistemicStatus,
    belief: BinaryBelief,
    revision: u64,
    evidence_history: Vec<EvidenceEntry>,
}

impl PropositionRecord {
    /// Devuelve la [`PropositionKey`] canónica del registro.
    pub const fn key(&self) -> &PropositionKey {
        &self.key
    }

    /// Devuelve el [`EpistemicStatus`] actual de la proposición.
    pub const fn status(&self) -> EpistemicStatus {
        self.status
    }

    /// Devuelve la [`BinaryBelief`] actual sobre la proposición.
    pub const fn belief(&self) -> &BinaryBelief {
        &self.belief
    }

    /// Devuelve la probabilidad actual `P(true)` de la proposición.
    pub fn p_true(&self) -> f64 {
        self.belief.p_true()
    }

    /// Devuelve el número de revisión monotónico actual (`>= 1`).
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Devuelve el historial append-only completo de evidencias registradas (soporte y contradicción).
    pub fn evidence_history(&self) -> &[EvidenceEntry] {
        &self.evidence_history
    }

    /// Cuenta cuántas entradas de soporte existen en el historial de evidencia.
    pub fn supporting_count(&self) -> usize {
        self.evidence_history
            .iter()
            .filter(|e| e.polarity.is_supporting())
            .count()
    }

    /// Cuenta cuántas entradas contradictorias existen en el historial de evidencia.
    pub fn contradicting_count(&self) -> usize {
        self.evidence_history
            .iter()
            .filter(|e| e.polarity.is_contradicting())
            .count()
    }

    /// Indica si `evidence_ref` ya se encuentra registrada en el historial.
    pub fn has_evidence(&self, evidence_ref: EvidenceRef) -> bool {
        self.evidence_history
            .iter()
            .any(|e| e.evidence_ref == evidence_ref)
    }

    /// Devuelve el instante monotónico de la evidencia más reciente en el historial.
    pub fn last_observed_at(&self) -> Option<MonotonicInstant> {
        self.evidence_history.iter().map(|e| e.observed_at).max()
    }
}

/// Filtro determinista para resolver candidatos de grounding en [`SemanticKnowledgeStore`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroundingFilter {
    /// Restringe la resolución a una entidad concreta, si se especifica.
    pub entity: Option<EntityId>,
    /// Restringe a entidades de una categoría [`EntityKind`] concreta.
    pub entity_kind: Option<EntityKind>,
    /// Etiquetas que la entidad debe poseer todas simultáneamente.
    pub required_tags: BTreeSet<String>,
    /// Atributos estructurados exactos que la entidad debe satisfacer.
    pub required_attributes: BTreeMap<String, PropositionValue>,
    /// Restringe las proposiciones emparejadas a un valor objeto concreto.
    pub object: Option<PropositionValue>,
    /// Probabilidad mínima `P(true)` requerida en las proposiciones emparejadas.
    pub min_p_true: Option<f64>,
    /// Conjunto de estados epistémicos permitidos para las proposiciones emparejadas.
    pub allowed_statuses: Option<BTreeSet<EpistemicStatus>>,
}

impl GroundingFilter {
    /// Crea un filtro vacío sin restricciones.
    pub fn new() -> Self {
        Self::default()
    }

    /// Restringe la búsqueda a una [`EntityId`] específica.
    pub const fn with_entity(mut self, entity: EntityId) -> Self {
        self.entity = Some(entity);
        self
    }

    /// Restringe la búsqueda a entidades de tipo `kind`.
    pub fn with_entity_kind(mut self, kind: EntityKind) -> Self {
        self.entity_kind = Some(kind);
        self
    }

    /// Exige que la entidad contenga la etiqueta `tag`.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Result<Self, SemanticMemoryError> {
        let tag = validate_non_empty("tag", tag)?;
        self.required_tags.insert(tag);
        Ok(self)
    }

    /// Exige que la entidad posea el atributo `name` con valor exacto `value`.
    pub fn with_attribute(
        mut self,
        name: impl Into<String>,
        value: PropositionValue,
    ) -> Result<Self, SemanticMemoryError> {
        let name = validate_non_empty("attribute_name", name)?;
        self.required_attributes.insert(name, value);
        Ok(self)
    }

    /// Restringe las proposiciones emparejadas a un objeto concreto.
    pub fn with_object(mut self, object: PropositionValue) -> Self {
        self.object = Some(object);
        self
    }

    /// Exige una probabilidad mínima `P(true)` finita en `[0.0, 1.0]`.
    pub fn with_min_p_true(mut self, min_p_true: f64) -> Result<Self, SemanticMemoryError> {
        if !min_p_true.is_finite() || !(0.0..=1.0).contains(&min_p_true) {
            return Err(SemanticMemoryError::InvalidProbability {
                field: "min_p_true",
                value: min_p_true,
            });
        }
        self.min_p_true = Some(if min_p_true == 0.0 { 0.0 } else { min_p_true });
        Ok(self)
    }

    /// Restringe las proposiciones emparejadas a los [`EpistemicStatus`] indicados.
    pub fn with_allowed_statuses(
        mut self,
        statuses: impl IntoIterator<Item = EpistemicStatus>,
    ) -> Self {
        self.allowed_statuses = Some(statuses.into_iter().collect());
        self
    }

    fn has_proposition_constraints(&self, predicate: Option<&PredicateId>) -> bool {
        predicate.is_some()
            || self.object.is_some()
            || self.min_p_true.is_some()
            || self.allowed_statuses.is_some()
    }

    fn matches_entity(&self, entity: &SemanticEntity) -> bool {
        if let Some(expected_id) = self.entity
            && entity.id() != expected_id
        {
            return false;
        }
        if let Some(expected_kind) = &self.entity_kind
            && entity.kind() != expected_kind
        {
            return false;
        }
        if !self.required_tags.iter().all(|tag| entity.has_tag(tag)) {
            return false;
        }
        for (name, expected_val) in &self.required_attributes {
            if entity.attribute(name) != Some(expected_val) {
                return false;
            }
        }
        true
    }

    fn matches_proposition(
        &self,
        predicate: Option<&PredicateId>,
        record: &PropositionRecord,
    ) -> bool {
        if let Some(expected_pred) = predicate
            && &record.key.predicate != expected_pred
        {
            return false;
        }
        if let Some(expected_obj) = &self.object
            && &record.key.object != expected_obj
        {
            return false;
        }
        if let Some(min_p) = self.min_p_true
            && record.p_true() < min_p
        {
            return false;
        }
        if let Some(allowed) = &self.allowed_statuses
            && !allowed.contains(&record.status)
        {
            return false;
        }
        true
    }
}

/// Candidato resuelto para grounding de una entidad junto con sus proposiciones emparejadas.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundingCandidate {
    /// Entidad semántica candidata.
    pub entity: SemanticEntity,
    /// Proposiciones de la entidad que satisfacen la consulta de grounding.
    pub matched_propositions: Vec<PropositionRecord>,
    /// Mayor probabilidad `P(true)` entre las proposiciones emparejadas (`0.0` si no hay ninguna).
    pub best_p_true: f64,
    /// Estado epistémico de la proposición de mayor probabilidad emparejada.
    pub best_status: Option<EpistemicStatus>,
}

/// Repositorio nativo de memoria semántica revisable en memoria (`std`-only, determinista).
#[derive(Debug, Clone, Default)]
pub struct SemanticKnowledgeStore {
    entities: BTreeMap<EntityId, SemanticEntity>,
    propositions: BTreeMap<PropositionKey, PropositionRecord>,
}

impl SemanticKnowledgeStore {
    /// Crea un repositorio semántico vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Devuelve el número de entidades registradas.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Devuelve el número de proposiciones registradas.
    pub fn proposition_count(&self) -> usize {
        self.propositions.len()
    }

    /// Inserta o actualiza una [`SemanticEntity`], devolviendo la versión previa si existía.
    pub fn upsert_entity(&mut self, entity: SemanticEntity) -> Option<SemanticEntity> {
        self.entities.insert(entity.id(), entity)
    }

    /// Obtiene una [`SemanticEntity`] por su [`EntityId`], si existe.
    pub fn get_entity(&self, id: EntityId) -> Option<&SemanticEntity> {
        self.entities.get(&id)
    }

    /// Registra una nueva proposición con su estado epistémico inicial, creencia previa y
    /// primera evidencia de procedencia.
    ///
    /// Si ya existe una proposición con la misma [`PropositionKey`], devuelve
    /// [`SemanticMemoryError::PropositionAlreadyExists`] para evitar sobrescribir o destruir
    /// su historial de procedencia.
    pub fn record_proposition(
        &mut self,
        key: PropositionKey,
        initial_status: EpistemicStatus,
        prior_belief: BinaryBelief,
        initial_evidence: EvidenceInput,
    ) -> Result<&PropositionRecord, SemanticMemoryError> {
        if self.propositions.contains_key(&key) {
            return Err(SemanticMemoryError::PropositionAlreadyExists);
        }

        let record = PropositionRecord {
            key: key.clone(),
            status: initial_status,
            belief: prior_belief,
            revision: 1,
            evidence_history: vec![EvidenceEntry::from_input(initial_evidence, 1)],
        };
        self.propositions.insert(key.clone(), record);
        Ok(self
            .propositions
            .get(&key)
            .expect("inserted proposition must exist"))
    }

    /// Obtiene un [`PropositionRecord`] por su [`PropositionKey`], si existe.
    pub fn get_proposition(&self, key: &PropositionKey) -> Option<&PropositionRecord> {
        self.propositions.get(key)
    }

    /// Añade nueva evidencia de soporte o contradictoria a una proposición existente,
    /// revisando su [`BinaryBelief`] mediante el [`LikelihoodModel`] calibrado y actualizando
    /// su [`EpistemicStatus`] **sin destruir ni modificar la procedencia previa**.
    pub fn add_evidence(
        &mut self,
        key: &PropositionKey,
        evidence: EvidenceInput,
        model: &LikelihoodModel,
    ) -> Result<RevisionStatus, SemanticMemoryError> {
        let Some(record) = self.propositions.get_mut(key) else {
            return Err(SemanticMemoryError::PropositionNotFound);
        };

        if record.has_evidence(evidence.evidence_ref) {
            return Ok(RevisionStatus::DuplicateEvidenceIgnored);
        }

        let outcome = match evidence.polarity {
            EvidencePolarity::Supporting => 1_usize,
            EvidencePolarity::Contradicting => 0_usize,
        };

        let mut next_discrete = record.belief.as_discrete().clone();
        let update_status = next_discrete.update_with_outcome(outcome, model)?;

        let next_revision = record.revision.saturating_add(1);
        record.revision = next_revision;
        record
            .evidence_history
            .push(EvidenceEntry::from_input(evidence, next_revision));

        let revision_status = match update_status {
            BeliefUpdateStatus::Updated => {
                record.belief = BinaryBelief::from_discrete(next_discrete)?;
                RevisionStatus::Updated
            }
            BeliefUpdateStatus::ConflictImpossibleObservation => {
                RevisionStatus::ConflictPreservedPriorBelief
            }
            BeliefUpdateStatus::DuplicateEvidenceIgnored
            | BeliefUpdateStatus::AbstainedStaleObservation
            | BeliefUpdateStatus::AbstainedUnsupportedKind => {
                unreachable!(
                    "update_with_outcome only returns Updated or ConflictImpossibleObservation"
                )
            }
        };

        let supporting_count = record.supporting_count();
        let contradicting_count = record.contradicting_count();

        record.status = match evidence.polarity {
            EvidencePolarity::Contradicting => {
                if revision_status.is_conflict() || record.p_true() >= 0.5 {
                    EpistemicStatus::Contested
                } else {
                    EpistemicStatus::Revised
                }
            }
            EvidencePolarity::Supporting => {
                if contradicting_count > 0 {
                    if record.p_true() >= 0.5 {
                        EpistemicStatus::Contested
                    } else {
                        EpistemicStatus::Revised
                    }
                } else if supporting_count >= 2
                    || (evidence.claim_kind.is_measurement() && record.p_true() >= 0.5)
                {
                    EpistemicStatus::CorroboratedObservation
                } else {
                    record.status
                }
            }
        };

        Ok(revision_status)
    }

    /// Consulta todas las proposiciones asociadas a una entidad sujeto en orden determinista.
    pub fn query_by_entity(&self, entity: EntityId) -> Vec<&PropositionRecord> {
        self.propositions
            .values()
            .filter(|record| record.key.subject == entity)
            .collect()
    }

    /// Resuelve candidatos de grounding deterministamente según un predicado opcional y un
    /// [`GroundingFilter`].
    ///
    /// Los candidatos se ordenan por mayor probabilidad `best_p_true` descendente y, en caso
    /// de empate, por [`EntityId`] ascendente.
    pub fn resolve_grounding_candidates(
        &self,
        predicate: Option<&PredicateId>,
        filter: &GroundingFilter,
    ) -> Vec<GroundingCandidate> {
        let require_proposition_match = filter.has_proposition_constraints(predicate);
        let mut candidates = Vec::new();

        for entity in self.entities.values() {
            if !filter.matches_entity(entity) {
                continue;
            }

            let matched_propositions: Vec<PropositionRecord> = self
                .query_by_entity(entity.id())
                .into_iter()
                .filter(|record| filter.matches_proposition(predicate, record))
                .cloned()
                .collect();

            if require_proposition_match && matched_propositions.is_empty() {
                continue;
            }

            let (best_p_true, best_status) =
                matched_propositions
                    .iter()
                    .fold((0.0_f64, None), |(acc_p, acc_status), record| {
                        let p = record.p_true();
                        if acc_status.is_none() || p > acc_p {
                            (p, Some(record.status()))
                        } else {
                            (acc_p, acc_status)
                        }
                    });

            candidates.push(GroundingCandidate {
                entity: entity.clone(),
                matched_propositions,
                best_p_true,
                best_status,
            });
        }

        candidates.sort_by(|a, b| {
            b.best_p_true
                .total_cmp(&a.best_p_true)
                .then_with(|| a.entity.id().cmp(&b.entity.id()))
        });
        candidates
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_evidence(
        evidence_ref: u64,
        observed_ns: u128,
        sensor_id: u64,
        polarity: EvidencePolarity,
        claim_kind: ObservationKind,
        confidence: f64,
    ) -> EvidenceInput {
        EvidenceInput::new(
            EvidenceRef::new(evidence_ref),
            MonotonicInstant::from_nanos(observed_ns),
            SensorId::new(sensor_id),
            polarity,
            claim_kind,
            Confidence::new(confidence).unwrap(),
        )
    }

    #[test]
    fn ac1_proposed_fact_hypothesis_and_corroborated_observation_are_distinguishable() {
        let mut store = SemanticKnowledgeStore::new();
        let door = EntityId::new(10);
        let pred_open = PredicateId::new("is_open").unwrap();
        let pred_locked = PredicateId::new("is_locked").unwrap();
        let pred_zone = PredicateId::new("located_in").unwrap();

        let key_proposed = PropositionKey::new(door, pred_open, PropositionValue::Bool(true));
        let key_hypothesis = PropositionKey::new(door, pred_locked, PropositionValue::Bool(true));
        let key_corroborated = PropositionKey::new(
            door,
            pred_zone,
            PropositionValue::symbol("sector_alpha").unwrap(),
        );

        let proposed = store
            .record_proposition(
                key_proposed.clone(),
                EpistemicStatus::ProposedFact,
                BinaryBelief::new(0.55).unwrap(),
                sample_evidence(
                    1,
                    100,
                    1,
                    EvidencePolarity::Supporting,
                    ObservationKind::Proposal,
                    0.6,
                ),
            )
            .unwrap()
            .clone();

        let hypothesis = store
            .record_proposition(
                key_hypothesis.clone(),
                EpistemicStatus::Hypothesis,
                BinaryBelief::new(0.4).unwrap(),
                sample_evidence(
                    2,
                    110,
                    2,
                    EvidencePolarity::Supporting,
                    ObservationKind::Unknown,
                    0.4,
                ),
            )
            .unwrap()
            .clone();

        let corroborated = store
            .record_proposition(
                key_corroborated.clone(),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.9).unwrap(),
                sample_evidence(
                    3,
                    120,
                    3,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.95,
                ),
            )
            .unwrap()
            .clone();

        assert!(proposed.status().is_proposed_fact());
        assert!(!proposed.status().is_hypothesis());
        assert!(!proposed.status().is_corroborated());

        assert!(hypothesis.status().is_hypothesis());
        assert!(!hypothesis.status().is_proposed_fact());
        assert!(!hypothesis.status().is_corroborated());

        assert!(corroborated.status().is_corroborated());
        assert!(!corroborated.status().is_proposed_fact());
        assert!(!corroborated.status().is_hypothesis());

        assert_ne!(proposed.status(), hypothesis.status());
        assert_ne!(proposed.status(), corroborated.status());
        assert_ne!(hypothesis.status(), corroborated.status());

        // Una hipótesis puede promoverse a observación corroborada al añadir evidencia de soporte.
        let model = LikelihoodModel::from_sensitivity_specificity(0.85, 0.85).unwrap();
        let rev = store
            .add_evidence(
                &key_hypothesis,
                sample_evidence(
                    4,
                    150,
                    4,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.9,
                ),
                &model,
            )
            .unwrap();
        assert_eq!(rev, RevisionStatus::Updated);

        let promoted = store.get_proposition(&key_hypothesis).unwrap();
        assert!(promoted.status().is_corroborated());
        assert!(promoted.p_true() > 0.5);
        assert_eq!(promoted.revision(), 2);
        assert_eq!(promoted.evidence_history().len(), 2);
    }

    #[test]
    fn ac2_contradicting_evidence_revises_belief_without_destroying_prior_provenance() {
        let mut store = SemanticKnowledgeStore::new();
        let valve = EntityId::new(42);
        let key = PropositionKey::new(
            valve,
            PredicateId::new("status").unwrap(),
            PropositionValue::symbol("open").unwrap(),
        );

        let initial_ev = sample_evidence(
            1001,
            1_000,
            10,
            EvidencePolarity::Supporting,
            ObservationKind::Measurement,
            0.9,
        );
        store
            .record_proposition(
                key.clone(),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.8).unwrap(),
                initial_ev,
            )
            .unwrap();

        // Sobrescribir con record_proposition está prohibido para proteger la procedencia previa.
        assert!(matches!(
            store.record_proposition(
                key.clone(),
                EpistemicStatus::ProposedFact,
                BinaryBelief::new(0.5).unwrap(),
                sample_evidence(
                    1002,
                    1_500,
                    11,
                    EvidencePolarity::Supporting,
                    ObservationKind::Assertion,
                    0.5,
                ),
            ),
            Err(SemanticMemoryError::PropositionAlreadyExists)
        ));

        let model = LikelihoodModel::from_sensitivity_specificity(0.8, 0.8).unwrap();

        // Primera evidencia contradictoria: con prior 0.8 y (0.8, 0.8), P(true | obs=false) = 0.5.
        let contra_1 = sample_evidence(
            2001,
            2_000,
            20,
            EvidencePolarity::Contradicting,
            ObservationKind::Measurement,
            0.85,
        );
        let status_1 = store.add_evidence(&key, contra_1, &model).unwrap();
        assert_eq!(status_1, RevisionStatus::Updated);

        let rec_after_1 = store.get_proposition(&key).unwrap();
        assert!((rec_after_1.p_true() - 0.5).abs() < 1e-12);
        assert_eq!(rec_after_1.status(), EpistemicStatus::Contested);
        assert!(rec_after_1.status().has_contradiction());
        assert_eq!(rec_after_1.revision(), 2);
        assert_eq!(rec_after_1.evidence_history().len(), 2);
        assert_eq!(
            rec_after_1.evidence_history()[0].evidence_ref,
            EvidenceRef::new(1001)
        );
        assert_eq!(
            rec_after_1.evidence_history()[0].sensor_id,
            SensorId::new(10)
        );
        assert_eq!(
            rec_after_1.evidence_history()[0].polarity,
            EvidencePolarity::Supporting
        );
        assert_eq!(
            rec_after_1.evidence_history()[1].evidence_ref,
            EvidenceRef::new(2001)
        );
        assert_eq!(
            rec_after_1.evidence_history()[1].polarity,
            EvidencePolarity::Contradicting
        );

        // Duplicado de la misma EvidenceRef no vuelve a descontar ni altera el historial.
        let dup_status = store.add_evidence(&key, contra_1, &model).unwrap();
        assert_eq!(dup_status, RevisionStatus::DuplicateEvidenceIgnored);
        assert_eq!(store.get_proposition(&key).unwrap().revision(), 2);
        assert_eq!(
            store
                .get_proposition(&key)
                .unwrap()
                .evidence_history()
                .len(),
            2
        );

        // Segunda evidencia contradictoria: baja P(true) de 0.5 a 0.2 y pasa a Revised.
        let contra_2 = sample_evidence(
            2002,
            3_000,
            21,
            EvidencePolarity::Contradicting,
            ObservationKind::Measurement,
            0.9,
        );
        let status_2 = store.add_evidence(&key, contra_2, &model).unwrap();
        assert_eq!(status_2, RevisionStatus::Updated);

        let rec_after_2 = store.get_proposition(&key).unwrap();
        assert!((rec_after_2.p_true() - 0.2).abs() < 1e-12);
        assert_eq!(rec_after_2.status(), EpistemicStatus::Revised);
        assert!(rec_after_2.status().is_revised());
        assert_eq!(rec_after_2.revision(), 3);
        assert_eq!(rec_after_2.supporting_count(), 1);
        assert_eq!(rec_after_2.contradicting_count(), 2);

        // Toda la procedencia previa sigue intacta y auditable en orden append-only.
        let refs: Vec<EvidenceRef> = rec_after_2
            .evidence_history()
            .iter()
            .map(|e| e.evidence_ref)
            .collect();
        assert_eq!(
            refs,
            vec![
                EvidenceRef::new(1001),
                EvidenceRef::new(2001),
                EvidenceRef::new(2002),
            ]
        );
        assert_eq!(
            rec_after_2.last_observed_at(),
            Some(MonotonicInstant::from_nanos(3_000))
        );
    }

    #[test]
    fn ac3_entity_query_and_grounding_candidates_resolve_deterministically_in_fixture() {
        let mut store = SemanticKnowledgeStore::new();

        let door_kind = EntityKind::new("door").unwrap();
        let beacon_kind = EntityKind::new("beacon").unwrap();
        let pred_passable = PredicateId::new("is_passable").unwrap();
        let pred_located = PredicateId::new("located_in").unwrap();

        let e_door_a = SemanticEntity::new(EntityId::new(1), door_kind.clone())
            .with_tag("north_wing")
            .unwrap()
            .with_tag("accessible")
            .unwrap()
            .with_attribute("material", PropositionValue::symbol("steel").unwrap())
            .unwrap();
        let e_door_b = SemanticEntity::new(EntityId::new(2), door_kind.clone())
            .with_tag("north_wing")
            .unwrap()
            .with_tag("accessible")
            .unwrap()
            .with_attribute("material", PropositionValue::symbol("wood").unwrap())
            .unwrap();
        let e_door_c = SemanticEntity::new(EntityId::new(3), door_kind.clone())
            .with_tag("south_wing")
            .unwrap()
            .with_tag("accessible")
            .unwrap();
        let e_beacon = SemanticEntity::new(EntityId::new(4), beacon_kind)
            .with_tag("north_wing")
            .unwrap();

        assert!(store.upsert_entity(e_door_a).is_none());
        assert!(store.upsert_entity(e_door_b).is_none());
        assert!(store.upsert_entity(e_door_c).is_none());
        assert!(store.upsert_entity(e_beacon).is_none());
        assert_eq!(store.entity_count(), 4);

        // Door 1: observación corroborada con alta creencia (0.92).
        store
            .record_proposition(
                PropositionKey::new(
                    EntityId::new(1),
                    pred_passable.clone(),
                    PropositionValue::Bool(true),
                ),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.92).unwrap(),
                sample_evidence(
                    10,
                    100,
                    1,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.95,
                ),
            )
            .unwrap();
        store
            .record_proposition(
                PropositionKey::new(
                    EntityId::new(1),
                    pred_located.clone(),
                    PropositionValue::Entity(EntityId::new(100)),
                ),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.95).unwrap(),
                sample_evidence(
                    11,
                    105,
                    1,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.95,
                ),
            )
            .unwrap();

        // Door 2: solo hipótesis con probabilidad moderada (0.60).
        store
            .record_proposition(
                PropositionKey::new(
                    EntityId::new(2),
                    pred_passable.clone(),
                    PropositionValue::Bool(true),
                ),
                EpistemicStatus::Hypothesis,
                BinaryBelief::new(0.60).unwrap(),
                sample_evidence(
                    12,
                    110,
                    2,
                    EvidencePolarity::Supporting,
                    ObservationKind::Unknown,
                    0.6,
                ),
            )
            .unwrap();

        // Door 3: observación corroborada (0.85), pero en south_wing.
        store
            .record_proposition(
                PropositionKey::new(
                    EntityId::new(3),
                    pred_passable.clone(),
                    PropositionValue::Bool(true),
                ),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(0.85).unwrap(),
                sample_evidence(
                    13,
                    115,
                    3,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.9,
                ),
            )
            .unwrap();

        // Consulta directa por entidad devuelve todas sus proposiciones en orden determinista.
        let door1_props = store.query_by_entity(EntityId::new(1));
        assert_eq!(door1_props.len(), 2);
        assert_eq!(door1_props[0].key().predicate, pred_passable);
        assert_eq!(door1_props[1].key().predicate, pred_located);

        // Resolución de grounding para puertas transitables en north_wing (ordenado por P(true) desc).
        let north_filter = GroundingFilter::new()
            .with_entity_kind(door_kind.clone())
            .with_tag("north_wing")
            .unwrap()
            .with_object(PropositionValue::Bool(true))
            .with_min_p_true(0.5)
            .unwrap();
        let candidates = store.resolve_grounding_candidates(Some(&pred_passable), &north_filter);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].entity.id(), EntityId::new(1));
        assert!((candidates[0].best_p_true - 0.92).abs() < 1e-12);
        assert_eq!(
            candidates[0].best_status,
            Some(EpistemicStatus::CorroboratedObservation)
        );
        assert_eq!(candidates[1].entity.id(), EntityId::new(2));
        assert!((candidates[1].best_p_true - 0.60).abs() < 1e-12);
        assert_eq!(candidates[1].best_status, Some(EpistemicStatus::Hypothesis));

        // Si restringimos a observaciones corroboradas, solo se resuelve Door 1.
        let corroborated_only =
            north_filter.with_allowed_statuses([EpistemicStatus::CorroboratedObservation]);
        let strict_candidates =
            store.resolve_grounding_candidates(Some(&pred_passable), &corroborated_only);
        assert_eq!(strict_candidates.len(), 1);
        assert_eq!(strict_candidates[0].entity.id(), EntityId::new(1));

        // Consulta de grounding enfocada a una entidad concreta por atributo.
        let specific_filter = GroundingFilter::new()
            .with_entity(EntityId::new(1))
            .with_attribute("material", PropositionValue::symbol("steel").unwrap())
            .unwrap();
        let specific = store.resolve_grounding_candidates(Some(&pred_located), &specific_filter);
        assert_eq!(specific.len(), 1);
        assert_eq!(specific[0].matched_propositions.len(), 1);
        assert_eq!(
            specific[0].matched_propositions[0].key().object,
            PropositionValue::Entity(EntityId::new(100))
        );
    }

    #[test]
    fn validates_identifiers_and_handles_conflict_preserving_provenance() {
        assert!(matches!(
            EntityKind::new("   "),
            Err(SemanticMemoryError::EmptyIdentifier {
                field: "entity_kind"
            })
        ));
        assert!(matches!(
            PredicateId::new(""),
            Err(SemanticMemoryError::EmptyIdentifier { field: "predicate" })
        ));
        assert!(matches!(
            PropositionValue::symbol(" "),
            Err(SemanticMemoryError::EmptyIdentifier { field: "symbol" })
        ));
        assert!(matches!(
            GroundingFilter::new().with_min_p_true(f64::NAN),
            Err(SemanticMemoryError::InvalidProbability {
                field: "min_p_true",
                ..
            })
        ));

        let mut store = SemanticKnowledgeStore::new();
        let key = PropositionKey::new(
            EntityId::new(99),
            PredicateId::new("is_present").unwrap(),
            PropositionValue::Bool(true),
        );
        assert!(matches!(
            store.add_evidence(
                &key,
                sample_evidence(
                    1,
                    10,
                    1,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    0.8,
                ),
                &LikelihoodModel::from_sensitivity_specificity(0.8, 0.8).unwrap(),
            ),
            Err(SemanticMemoryError::PropositionNotFound)
        ));

        // Con prior 1.0 y sensor ideal (1.0, 1.0), una evidencia contradictoria tiene P(obs) == 0.0:
        // se preserva la creencia sin NaN, se registra la evidencia y pasa a Contested.
        store
            .record_proposition(
                key.clone(),
                EpistemicStatus::CorroboratedObservation,
                BinaryBelief::new(1.0).unwrap(),
                sample_evidence(
                    10,
                    100,
                    1,
                    EvidencePolarity::Supporting,
                    ObservationKind::Measurement,
                    1.0,
                ),
            )
            .unwrap();
        let ideal_model = LikelihoodModel::from_sensitivity_specificity(1.0, 1.0).unwrap();
        let conflict = store
            .add_evidence(
                &key,
                sample_evidence(
                    11,
                    200,
                    2,
                    EvidencePolarity::Contradicting,
                    ObservationKind::Measurement,
                    1.0,
                ),
                &ideal_model,
            )
            .unwrap();
        assert_eq!(conflict, RevisionStatus::ConflictPreservedPriorBelief);
        let rec = store.get_proposition(&key).unwrap();
        assert_eq!(rec.status(), EpistemicStatus::Contested);
        assert_eq!(rec.p_true(), 1.0);
        assert_eq!(rec.evidence_history().len(), 2);
    }
}
