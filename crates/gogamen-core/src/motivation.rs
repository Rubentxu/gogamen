//! Módulo reutilizable de motivación, homeostasis y utilidad dinámica (`std`-only).
//!
//! Este módulo es completamente agnóstico del dominio (no presupone agentes ni
//! reglas de ningún entorno específico) y separa estrictamente:
//! - **Homeostasis ([`DriveSpec`], [`DriveReading`], [`HomeostaticState`]):**
//!   variables internas con rango `[min_bound, max_bound]`, objetivo `target`,
//!   peso base y ganancia por déficit. El déficit se normaliza y satura en `[0.0, 1.0]`.
//! - **Utilidad dinámica ([`ObjectiveTerm`], [`WeightedObjective`], [`DynamicUtility`]):**
//!   términos de objetivo cuya ponderación efectiva se modula según
//!   `base_weight + deficit_gain * deficit`.
//! - **Restricciones inviolables ([`HardConstraint`], [`ForbiddenActionRule`]):**
//!   barreras duras completamente independientes de los objetivos ponderados.
//!   [`UtilityEvaluator::evaluate_candidates`] puntúa las acciones candidatas por su
//!   utilidad dinámica, pero **nunca** permite seleccionar una acción prohibida por
//!   una [`HardConstraint`], sin importar cuán alta sea su utilidad ponderada.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::action::Action;

/// Errores explícitos de validación y evaluación en el módulo de motivación.
#[derive(Debug, Clone, PartialEq)]
pub enum MotivationError {
    /// Una entrada numérica es `NaN`, `+inf` o `-inf`.
    NonFiniteInput {
        /// Campo o contexto donde se detectó el valor no finito.
        field: &'static str,
        /// Valor inválido recibido.
        value: f64,
    },
    /// Los límites `[min_bound, max_bound]` o el `target` de un drive son inválidos.
    InvalidBounds {
        /// Límite inferior recibido.
        min_bound: f64,
        /// Límite superior recibido.
        max_bound: f64,
        /// Valor objetivo recibido.
        target: f64,
    },
    /// Un peso base o ganancia por déficit es negativo o inválido.
    InvalidWeight {
        /// Campo donde se detectó el peso inválido.
        field: &'static str,
        /// Valor inválido recibido.
        value: f64,
    },
    /// Un identificador requerido está vacío.
    EmptyIdentifier {
        /// Tipo de identificador vacío.
        kind: &'static str,
    },
    /// Se registró un [`DriveId`] duplicado en [`HomeostaticState`].
    DuplicateDrive {
        /// Identificador duplicado.
        drive_id: DriveId,
    },
    /// Se consultó o actualizó un [`DriveId`] no registrado en [`HomeostaticState`].
    UnknownDrive {
        /// Identificador desconocido.
        drive_id: DriveId,
    },
    /// Se registró un término de objetivo duplicado en [`DynamicUtility`].
    DuplicateObjective {
        /// Nombre del objetivo duplicado.
        name: String,
    },
}

impl fmt::Display for MotivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput { field, value } => {
                write!(f, "non-finite motivation input in '{field}': {value}")
            }
            Self::InvalidBounds {
                min_bound,
                max_bound,
                target,
            } => write!(
                f,
                "invalid drive bounds: min_bound={min_bound}, max_bound={max_bound}, target={target} (requires min_bound < max_bound and target in [min_bound, max_bound])"
            ),
            Self::InvalidWeight { field, value } => {
                write!(f, "invalid weight in '{field}': {value} (must be >= 0.0)")
            }
            Self::EmptyIdentifier { kind } => {
                write!(f, "identifier '{kind}' cannot be empty")
            }
            Self::DuplicateDrive { drive_id } => {
                write!(f, "duplicate drive id '{drive_id}'")
            }
            Self::UnknownDrive { drive_id } => {
                write!(f, "unknown drive id '{drive_id}'")
            }
            Self::DuplicateObjective { name } => {
                write!(f, "duplicate objective term '{name}'")
            }
        }
    }
}

impl Error for MotivationError {}

fn ensure_finite_input(field: &'static str, value: f64) -> Result<f64, MotivationError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MotivationError::NonFiniteInput { field, value })
    }
}

fn ensure_non_negative_weight(field: &'static str, value: f64) -> Result<f64, MotivationError> {
    ensure_finite_input(field, value)?;
    if value < 0.0 {
        return Err(MotivationError::InvalidWeight { field, value });
    }
    Ok(value)
}

/// Identificador canónico de un drive homeostático independiente del dominio.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DriveId(String);

impl DriveId {
    /// Construye un [`DriveId`] no vacío.
    pub fn new(id: impl Into<String>) -> Result<Self, MotivationError> {
        let raw = id.into();
        if raw.trim().is_empty() {
            return Err(MotivationError::EmptyIdentifier { kind: "drive_id" });
        }
        Ok(Self(raw))
    }

    /// Devuelve el identificador como `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DriveId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Especificación inmutable de un drive homeostático.
///
/// Define el punto de equilibrio `target` dentro de `[min_bound, max_bound]`,
/// un `base_weight` (`>= 0.0`) y una ganancia `deficit_gain` (`>= 0.0`) que
/// amplifica la prioridad del drive cuando su déficit aumenta.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveSpec {
    id: DriveId,
    target: f64,
    min_bound: f64,
    max_bound: f64,
    base_weight: f64,
    deficit_gain: f64,
}

impl DriveSpec {
    /// Crea una especificación de drive validando finitud, cotas y pesos no negativos.
    pub fn new(
        id: DriveId,
        target: f64,
        min_bound: f64,
        max_bound: f64,
        base_weight: f64,
        deficit_gain: f64,
    ) -> Result<Self, MotivationError> {
        ensure_finite_input("drive_spec.target", target)?;
        ensure_finite_input("drive_spec.min_bound", min_bound)?;
        ensure_finite_input("drive_spec.max_bound", max_bound)?;
        ensure_non_negative_weight("drive_spec.base_weight", base_weight)?;
        ensure_non_negative_weight("drive_spec.deficit_gain", deficit_gain)?;

        if min_bound >= max_bound || target < min_bound || target > max_bound {
            return Err(MotivationError::InvalidBounds {
                min_bound,
                max_bound,
                target,
            });
        }

        Ok(Self {
            id,
            target,
            min_bound,
            max_bound,
            base_weight,
            deficit_gain,
        })
    }

    /// Devuelve el [`DriveId`] asociado.
    pub const fn id(&self) -> &DriveId {
        &self.id
    }

    /// Devuelve el valor objetivo homeostático.
    pub const fn target(&self) -> f64 {
        self.target
    }

    /// Devuelve el límite inferior del rango nominal.
    pub const fn min_bound(&self) -> f64 {
        self.min_bound
    }

    /// Devuelve el límite superior del rango nominal.
    pub const fn max_bound(&self) -> f64 {
        self.max_bound
    }

    /// Devuelve el peso base del drive.
    pub const fn base_weight(&self) -> f64 {
        self.base_weight
    }

    /// Devuelve la ganancia aplicada al déficit normalizado.
    pub const fn deficit_gain(&self) -> f64 {
        self.deficit_gain
    }

    /// Calcula el déficit normalizado y saturado en `[0.0, 1.0]` para `value`.
    ///
    /// Rechaza explícitamente valores `NaN` o infinitos con [`MotivationError::NonFiniteInput`].
    /// Valores fuera de `[min_bound, max_bound]` se saturan al límite correspondiente.
    pub fn compute_deficit(&self, value: f64) -> Result<f64, MotivationError> {
        ensure_finite_input("drive.value", value)?;
        let clamped = value.clamp(self.min_bound, self.max_bound);

        let raw_deficit = if clamped < self.target {
            let span = self.target - self.min_bound;
            if span > 0.0 {
                (self.target - clamped) / span
            } else {
                0.0
            }
        } else if clamped > self.target {
            let span = self.max_bound - self.target;
            if span > 0.0 {
                (clamped - self.target) / span
            } else {
                0.0
            }
        } else {
            0.0
        };

        Ok(raw_deficit.clamp(0.0, 1.0))
    }

    /// Calcula el peso efectivo `base_weight + deficit_gain * deficit` para un déficit dado.
    pub fn effective_weight(&self, deficit: f64) -> Result<f64, MotivationError> {
        ensure_finite_input("drive.deficit", deficit)?;
        let saturated_deficit = deficit.clamp(0.0, 1.0);
        let weight = self.base_weight + self.deficit_gain * saturated_deficit;
        ensure_finite_input("drive.effective_weight", weight)?;
        Ok(weight)
    }

    /// Genera una lectura [`DriveReading`] validada a partir de un valor observado.
    pub fn reading(&self, value: f64) -> Result<DriveReading, MotivationError> {
        let deficit = self.compute_deficit(value)?;
        let effective_weight = self.effective_weight(deficit)?;
        Ok(DriveReading {
            drive_id: self.id.clone(),
            value,
            deficit,
            effective_weight,
        })
    }
}

/// Lectura puntual e inmutable de un drive con su déficit normalizado y peso efectivo.
#[derive(Debug, Clone, PartialEq)]
pub struct DriveReading {
    drive_id: DriveId,
    value: f64,
    deficit: f64,
    effective_weight: f64,
}

impl DriveReading {
    /// Identificador del drive.
    pub const fn drive_id(&self) -> &DriveId {
        &self.drive_id
    }

    /// Valor escalar observado (finito).
    pub const fn value(&self) -> f64 {
        self.value
    }

    /// Déficit normalizado y saturado en `[0.0, 1.0]`.
    pub const fn deficit(&self) -> f64 {
        self.deficit
    }

    /// Peso efectivo modulado (`base_weight + deficit_gain * deficit`).
    pub const fn effective_weight(&self) -> f64 {
        self.effective_weight
    }
}

/// Estado homeostático agregado sobre un conjunto de [`DriveSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct HomeostaticState {
    specs: BTreeMap<DriveId, DriveSpec>,
    readings: BTreeMap<DriveId, DriveReading>,
}

impl HomeostaticState {
    /// Construye un estado homeostático inicializando cada drive en su `target` (déficit `0.0`).
    pub fn new(specs: Vec<DriveSpec>) -> Result<Self, MotivationError> {
        let mut spec_map = BTreeMap::new();
        let mut reading_map = BTreeMap::new();

        for spec in specs {
            let id = spec.id().clone();
            if spec_map.contains_key(&id) {
                return Err(MotivationError::DuplicateDrive { drive_id: id });
            }
            let initial = spec.reading(spec.target())?;
            spec_map.insert(id.clone(), spec);
            reading_map.insert(id, initial);
        }

        Ok(Self {
            specs: spec_map,
            readings: reading_map,
        })
    }

    /// Actualiza la lectura de un drive registrado validando finitud y calculando su déficit.
    pub fn set_reading(
        &mut self,
        drive_id: &DriveId,
        value: f64,
    ) -> Result<&DriveReading, MotivationError> {
        let spec = self
            .specs
            .get(drive_id)
            .ok_or_else(|| MotivationError::UnknownDrive {
                drive_id: drive_id.clone(),
            })?;
        let reading = spec.reading(value)?;
        self.readings.insert(drive_id.clone(), reading);
        Ok(&self.readings[drive_id])
    }

    /// Actualiza múltiples lecturas de drives en lote.
    pub fn update<'a, I>(&mut self, updates: I) -> Result<(), MotivationError>
    where
        I: IntoIterator<Item = (&'a DriveId, f64)>,
    {
        for (drive_id, value) in updates {
            self.set_reading(drive_id, value)?;
        }
        Ok(())
    }

    /// Devuelve la especificación de un drive si está registrado.
    pub fn spec(&self, drive_id: &DriveId) -> Option<&DriveSpec> {
        self.specs.get(drive_id)
    }

    /// Devuelve la lectura actual de un drive si está registrado.
    pub fn reading(&self, drive_id: &DriveId) -> Option<&DriveReading> {
        self.readings.get(drive_id)
    }

    /// Devuelve el déficit normalizado `[0.0, 1.0]` del drive, si existe.
    pub fn deficit(&self, drive_id: &DriveId) -> Option<f64> {
        self.readings.get(drive_id).map(DriveReading::deficit)
    }

    /// Devuelve el peso efectivo del drive, si existe.
    pub fn effective_weight(&self, drive_id: &DriveId) -> Option<f64> {
        self.readings
            .get(drive_id)
            .map(DriveReading::effective_weight)
    }

    /// Devuelve la suma total de déficits actuales.
    pub fn total_deficit(&self) -> f64 {
        self.readings.values().map(DriveReading::deficit).sum()
    }
}

/// Término de objetivo ponderado perteneciente a una [`DynamicUtility`].
///
/// Puede tener un peso estático o modularse dinámicamente a partir del déficit de un
/// [`DriveId`] mediante `base_weight + deficit_gain * deficit`.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectiveTerm {
    name: String,
    drive_id: Option<DriveId>,
    base_weight: f64,
    deficit_gain: f64,
}

/// Alias semántico para un término de objetivo ponderado.
pub type WeightedObjective = ObjectiveTerm;

impl ObjectiveTerm {
    /// Crea un término de objetivo con peso estático independiente de drives.
    pub fn static_weight(
        name: impl Into<String>,
        base_weight: f64,
    ) -> Result<Self, MotivationError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(MotivationError::EmptyIdentifier {
                kind: "objective_name",
            });
        }
        ensure_non_negative_weight("objective.base_weight", base_weight)?;
        Ok(Self {
            name,
            drive_id: None,
            base_weight,
            deficit_gain: 0.0,
        })
    }

    /// Crea un término de objetivo modulado por el déficit de `drive_id`.
    pub fn modulated(
        name: impl Into<String>,
        drive_id: DriveId,
        base_weight: f64,
        deficit_gain: f64,
    ) -> Result<Self, MotivationError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(MotivationError::EmptyIdentifier {
                kind: "objective_name",
            });
        }
        ensure_non_negative_weight("objective.base_weight", base_weight)?;
        ensure_non_negative_weight("objective.deficit_gain", deficit_gain)?;
        Ok(Self {
            name,
            drive_id: Some(drive_id),
            base_weight,
            deficit_gain,
        })
    }

    /// Crea un término de objetivo heredando `base_weight` y `deficit_gain` de un [`DriveSpec`].
    pub fn from_drive(name: impl Into<String>, spec: &DriveSpec) -> Result<Self, MotivationError> {
        Self::modulated(
            name,
            spec.id().clone(),
            spec.base_weight(),
            spec.deficit_gain(),
        )
    }

    /// Nombre del término de objetivo.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Identificador del drive modulador, si aplica.
    pub const fn drive_id(&self) -> Option<&DriveId> {
        self.drive_id.as_ref()
    }

    /// Peso base del término.
    pub const fn base_weight(&self) -> f64 {
        self.base_weight
    }

    /// Ganancia aplicada según el déficit del drive asociado.
    pub const fn deficit_gain(&self) -> f64 {
        self.deficit_gain
    }
}

/// Desglose de la utilidad dinámica calculada para una acción candidata.
#[derive(Debug, Clone, PartialEq)]
pub struct UtilityBreakdown {
    total_utility: f64,
    term_contributions: BTreeMap<String, f64>,
}

impl UtilityBreakdown {
    /// Utilidad total agregada.
    pub const fn total_utility(&self) -> f64 {
        self.total_utility
    }

    /// Contribución individual por nombre de término de objetivo.
    pub const fn term_contributions(&self) -> &BTreeMap<String, f64> {
        &self.term_contributions
    }
}

/// Función de utilidad dinámica modulada por el estado homeostático.
#[derive(Debug, Clone, PartialEq)]
pub struct DynamicUtility {
    terms: Vec<WeightedObjective>,
}

impl DynamicUtility {
    /// Construye una utilidad dinámica validando que no existan nombres de término duplicados.
    pub fn new(terms: Vec<WeightedObjective>) -> Result<Self, MotivationError> {
        let mut seen = BTreeSet::new();
        for term in &terms {
            if !seen.insert(term.name().to_owned()) {
                return Err(MotivationError::DuplicateObjective {
                    name: term.name().to_owned(),
                });
            }
        }
        Ok(Self { terms })
    }

    /// Devuelve los términos configurados.
    pub fn terms(&self) -> &[WeightedObjective] {
        &self.terms
    }

    /// Calcula el peso efectivo de un término bajo el [`HomeostaticState`] dado.
    pub fn effective_weight(
        &self,
        term: &WeightedObjective,
        state: &HomeostaticState,
    ) -> Result<f64, MotivationError> {
        match term.drive_id() {
            Some(drive_id) => {
                let deficit =
                    state
                        .deficit(drive_id)
                        .ok_or_else(|| MotivationError::UnknownDrive {
                            drive_id: drive_id.clone(),
                        })?;
                let weight = term.base_weight() + term.deficit_gain() * deficit;
                ensure_finite_input("dynamic_utility.effective_weight", weight)?;
                Ok(weight)
            }
            None => Ok(term.base_weight()),
        }
    }

    /// Puntúa una [`ActionCandidate`] combinando los términos ponderados por el estado homeostático.
    pub fn score_candidate(
        &self,
        candidate: &ActionCandidate,
        state: &HomeostaticState,
    ) -> Result<UtilityBreakdown, MotivationError> {
        let mut total_utility = 0.0;
        let mut term_contributions = BTreeMap::new();

        for term in &self.terms {
            let weight = self.effective_weight(term, state)?;
            let raw_score = candidate.score_for(term.name()).unwrap_or(0.0);
            let contribution = weight * raw_score;
            ensure_finite_input("dynamic_utility.contribution", contribution)?;
            total_utility += contribution;
            ensure_finite_input("dynamic_utility.total_utility", total_utility)?;
            term_contributions.insert(term.name().to_owned(), contribution);
        }

        Ok(UtilityBreakdown {
            total_utility,
            term_contributions,
        })
    }
}

/// Acción candidata junto con sus puntuaciones finitas por objetivo.
#[derive(Debug, Clone, PartialEq)]
pub struct ActionCandidate {
    action: Action,
    objective_scores: BTreeMap<String, f64>,
}

impl ActionCandidate {
    /// Crea una candidata sin puntuaciones iniciales.
    pub const fn new(action: Action) -> Self {
        Self {
            action,
            objective_scores: BTreeMap::new(),
        }
    }

    /// Asocia una puntuación finita a un objetivo por nombre.
    pub fn with_score(
        mut self,
        objective_name: impl Into<String>,
        score: f64,
    ) -> Result<Self, MotivationError> {
        let name = objective_name.into();
        if name.trim().is_empty() {
            return Err(MotivationError::EmptyIdentifier {
                kind: "objective_name",
            });
        }
        ensure_finite_input("candidate.objective_score", score)?;
        self.objective_scores.insert(name, score);
        Ok(self)
    }

    /// Devuelve la [`Action`] candidata.
    pub const fn action(&self) -> &Action {
        &self.action
    }

    /// Devuelve la puntuación registrada para `objective_name`, si existe.
    pub fn score_for(&self, objective_name: &str) -> Option<f64> {
        self.objective_scores.get(objective_name).copied()
    }
}

/// Detalle de violación de una restricción inviolable ([`HardConstraint`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstraintViolation {
    constraint_name: String,
    reason: String,
}

impl ConstraintViolation {
    /// Crea una violación explícita de restricción inviolable.
    pub fn new(constraint_name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            constraint_name: constraint_name.into(),
            reason: reason.into(),
        }
    }

    /// Nombre de la restricción violada.
    pub fn constraint_name(&self) -> &str {
        &self.constraint_name
    }

    /// Motivo determinista del rechazo.
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Restricción inviolable independiente de los objetivos ponderados.
///
/// Cualquier acción que falle [`HardConstraint::check`] es descartada de forma
/// absoluta por [`UtilityEvaluator`], sin importar su utilidad dinámica.
pub trait HardConstraint {
    /// Nombre identificador de la restricción.
    fn name(&self) -> &str;

    /// Verifica si `action` cumple la restricción bajo el [`HomeostaticState`] actual.
    fn check(&self, action: &Action, state: &HomeostaticState) -> Result<(), ConstraintViolation>;
}

type ForbiddenPredicate = Box<dyn Fn(&Action, &HomeostaticState) -> bool + Send + Sync>;

/// Regla concreta de acción prohibida que implementa [`HardConstraint`].
pub struct ForbiddenActionRule {
    name: String,
    reason: String,
    predicate: ForbiddenPredicate,
}

impl ForbiddenActionRule {
    /// Crea una regla inviolable que prohíbe cualquier acción donde `is_forbidden(action, state)` sea `true`.
    pub fn new<F>(name: impl Into<String>, reason: impl Into<String>, is_forbidden: F) -> Self
    where
        F: Fn(&Action, &HomeostaticState) -> bool + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            reason: reason.into(),
            predicate: Box::new(is_forbidden),
        }
    }

    /// Crea una regla inviolable que prohíbe incondicionalmente un índice discreto concreto.
    pub fn discrete_index(
        name: impl Into<String>,
        forbidden_index: usize,
        reason: impl Into<String>,
    ) -> Self {
        Self::new(name, reason, move |action, _| {
            action
                .as_discrete()
                .is_some_and(|discrete| discrete.index() == forbidden_index)
        })
    }
}

impl HardConstraint for ForbiddenActionRule {
    fn name(&self) -> &str {
        &self.name
    }

    fn check(&self, action: &Action, state: &HomeostaticState) -> Result<(), ConstraintViolation> {
        if (self.predicate)(action, state) {
            Err(ConstraintViolation::new(
                self.name.clone(),
                self.reason.clone(),
            ))
        } else {
            Ok(())
        }
    }
}

/// Resultado individual de evaluación para una acción candidata.
#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluation {
    candidate_index: usize,
    action: Action,
    breakdown: UtilityBreakdown,
    violations: Vec<ConstraintViolation>,
}

impl CandidateEvaluation {
    /// Índice de la candidata en el slice de entrada.
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    /// Acción evaluada.
    pub const fn action(&self) -> &Action {
        &self.action
    }

    /// Utilidad dinámica calculada (independiente de si fue prohibida).
    pub const fn utility(&self) -> f64 {
        self.breakdown.total_utility()
    }

    /// Desglose completo por término de objetivo.
    pub const fn breakdown(&self) -> &UtilityBreakdown {
        &self.breakdown
    }

    /// Violaciones de [`HardConstraint`] detectadas (vacío si es permitida).
    pub fn violations(&self) -> &[ConstraintViolation] {
        &self.violations
    }

    /// Indica si la acción es elegible (no viola ninguna [`HardConstraint`]).
    pub fn is_allowed(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Acción seleccionada tras filtrar por [`HardConstraint`] y maximizar [`DynamicUtility`].
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedAction {
    candidate_index: usize,
    action: Action,
    breakdown: UtilityBreakdown,
}

impl SelectedAction {
    /// Índice de la candidata seleccionada en el slice original.
    pub const fn candidate_index(&self) -> usize {
        self.candidate_index
    }

    /// Acción seleccionada.
    pub const fn action(&self) -> &Action {
        &self.action
    }

    /// Utilidad dinámica obtenida.
    pub const fn utility(&self) -> f64 {
        self.breakdown.total_utility()
    }

    /// Desglose de utilidad de la acción seleccionada.
    pub const fn breakdown(&self) -> &UtilityBreakdown {
        &self.breakdown
    }
}

/// Resultado global de [`UtilityEvaluator::evaluate_candidates`].
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationOutcome {
    selected: Option<SelectedAction>,
    evaluations: Vec<CandidateEvaluation>,
}

impl EvaluationOutcome {
    /// Acción seleccionada de mayor utilidad entre las permitidas, o `None` si todas fueron prohibidas.
    pub const fn selected(&self) -> Option<&SelectedAction> {
        self.selected.as_ref()
    }

    /// Evaluaciones individuales de todas las candidatas en su orden original.
    pub fn evaluations(&self) -> &[CandidateEvaluation] {
        &self.evaluations
    }
}

/// Evaluador que combina [`DynamicUtility`] y restricciones inviolables [`HardConstraint`].
///
/// Garantiza por construcción que una acción prohibida por una [`HardConstraint`] nunca
/// sea seleccionada, con independencia de su utilidad ponderada.
pub struct UtilityEvaluator {
    utility: DynamicUtility,
    constraints: Vec<Box<dyn HardConstraint>>,
}

impl UtilityEvaluator {
    /// Crea un evaluador con la función de utilidad dinámica dada y sin restricciones iniciales.
    pub fn new(utility: DynamicUtility) -> Self {
        Self {
            utility,
            constraints: Vec::new(),
        }
    }

    /// Añade una restricción inviolable en estilo builder.
    pub fn with_constraint(mut self, constraint: impl HardConstraint + 'static) -> Self {
        self.constraints.push(Box::new(constraint));
        self
    }

    /// Añade una restricción inviolable al evaluador.
    pub fn add_constraint(&mut self, constraint: impl HardConstraint + 'static) {
        self.constraints.push(Box::new(constraint));
    }

    /// Evalúa las candidatas bajo `state`, puntuando con [`DynamicUtility`] y filtrando con [`HardConstraint`].
    pub fn evaluate_candidates(
        &self,
        candidates: &[ActionCandidate],
        state: &HomeostaticState,
    ) -> Result<EvaluationOutcome, MotivationError> {
        let mut evaluations = Vec::with_capacity(candidates.len());
        let mut best: Option<SelectedAction> = None;

        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let mut violations = Vec::new();
            for constraint in &self.constraints {
                if let Err(violation) = constraint.check(candidate.action(), state) {
                    violations.push(violation);
                }
            }

            let breakdown = self.utility.score_candidate(candidate, state)?;
            let is_allowed = violations.is_empty();

            if is_allowed {
                let candidate_utility = breakdown.total_utility();
                let should_replace = match &best {
                    None => true,
                    Some(current) => candidate_utility > current.utility(),
                };
                if should_replace {
                    best = Some(SelectedAction {
                        candidate_index,
                        action: candidate.action().clone(),
                        breakdown: breakdown.clone(),
                    });
                }
            }

            evaluations.push(CandidateEvaluation {
                candidate_index,
                action: candidate.action().clone(),
                breakdown,
                violations,
            });
        }

        Ok(EvaluationOutcome {
            selected: best,
            evaluations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionSpace;

    #[test]
    fn ac1_deficit_modifies_utility_in_two_distinct_fixture_domains() {
        // Dominio 1: Robot móvil con drives de batería y temperatura.
        let robot_space = ActionSpace::discrete(2).unwrap();
        let act_work = Action::discrete_in_space(0, &robot_space).unwrap();
        let act_recharge = Action::discrete_in_space(1, &robot_space).unwrap();

        let battery_id = DriveId::new("battery").unwrap();
        let temp_id = DriveId::new("temperature").unwrap();
        let battery_spec = DriveSpec::new(battery_id.clone(), 1.0, 0.0, 1.0, 1.0, 5.0).unwrap();
        let temp_spec = DriveSpec::new(temp_id.clone(), 40.0, 10.0, 90.0, 1.0, 2.0).unwrap();
        let mut robot_state = HomeostaticState::new(vec![battery_spec, temp_spec]).unwrap();

        let robot_utility = DynamicUtility::new(vec![
            WeightedObjective::modulated("productivity", temp_id.clone(), 2.0, 0.0).unwrap(),
            WeightedObjective::modulated("charge_gain", battery_id.clone(), 1.0, 8.0).unwrap(),
        ])
        .unwrap();
        let robot_eval = UtilityEvaluator::new(robot_utility);

        let robot_candidates = vec![
            ActionCandidate::new(act_work.clone())
                .with_score("productivity", 1.0)
                .unwrap()
                .with_score("charge_gain", 0.0)
                .unwrap(),
            ActionCandidate::new(act_recharge.clone())
                .with_score("productivity", 0.1)
                .unwrap()
                .with_score("charge_gain", 0.5)
                .unwrap(),
        ];

        // Con batería llena (deficit = 0.0), trabajar gana (2.0 > 0.2 + 0.5 = 0.7).
        let robot_full = robot_eval
            .evaluate_candidates(&robot_candidates, &robot_state)
            .unwrap();
        assert_eq!(robot_full.selected().unwrap().action(), &act_work);

        // Con batería baja (0.1 -> deficit = 0.9), la ganancia por déficit eleva recargar.
        robot_state.set_reading(&battery_id, 0.1).unwrap();
        let robot_low = robot_eval
            .evaluate_candidates(&robot_candidates, &robot_state)
            .unwrap();
        assert_eq!(robot_low.selected().unwrap().action(), &act_recharge);
        assert!(robot_low.selected().unwrap().utility() > robot_full.evaluations()[1].utility());

        // Dominio 2: Servicio de red con drives de cola y latencia (sin relación con robótica ni Life).
        let svc_space = ActionSpace::discrete(2).unwrap();
        let act_serve_normal = Action::discrete_in_space(0, &svc_space).unwrap();
        let act_shed_load = Action::discrete_in_space(1, &svc_space).unwrap();

        let queue_id = DriveId::new("queue_depth").unwrap();
        let latency_id = DriveId::new("p99_latency_ms").unwrap();
        let queue_spec = DriveSpec::new(queue_id.clone(), 0.0, 0.0, 100.0, 1.0, 6.0).unwrap();
        let latency_spec = DriveSpec::new(latency_id.clone(), 10.0, 0.0, 200.0, 1.0, 4.0).unwrap();
        let mut svc_state = HomeostaticState::new(vec![queue_spec, latency_spec]).unwrap();

        let svc_utility = DynamicUtility::new(vec![
            WeightedObjective::static_weight("throughput", 3.0).unwrap(),
            WeightedObjective::modulated("backlog_relief", queue_id.clone(), 0.5, 10.0).unwrap(),
        ])
        .unwrap();
        let svc_eval = UtilityEvaluator::new(svc_utility);

        let svc_candidates = vec![
            ActionCandidate::new(act_serve_normal.clone())
                .with_score("throughput", 1.0)
                .unwrap()
                .with_score("backlog_relief", 0.0)
                .unwrap(),
            ActionCandidate::new(act_shed_load.clone())
                .with_score("throughput", 0.2)
                .unwrap()
                .with_score("backlog_relief", 0.6)
                .unwrap(),
        ];

        // Con cola vacía (deficit = 0.0), servir normal gana (3.0 > 0.6 + 0.3 = 0.9).
        let svc_idle = svc_eval
            .evaluate_candidates(&svc_candidates, &svc_state)
            .unwrap();
        assert_eq!(svc_idle.selected().unwrap().action(), &act_serve_normal);

        // Con cola saturada (90.0 -> deficit = 0.9), aliviar backlog domina.
        svc_state.set_reading(&queue_id, 90.0).unwrap();
        let svc_overloaded = svc_eval
            .evaluate_candidates(&svc_candidates, &svc_state)
            .unwrap();
        assert_eq!(svc_overloaded.selected().unwrap().action(), &act_shed_load);
        assert!(svc_overloaded.selected().unwrap().utility() > svc_idle.evaluations()[1].utility());
    }

    #[test]
    fn ac2_normalization_saturation_and_non_finite_errors() {
        let drive_id = DriveId::new("thermal").unwrap();
        let spec = DriveSpec::new(drive_id.clone(), 50.0, 20.0, 90.0, 1.0, 2.0).unwrap();

        // Normalización exacta dentro de [min_bound, max_bound].
        assert!((spec.compute_deficit(50.0).unwrap() - 0.0).abs() < 1e-12);
        assert!((spec.compute_deficit(70.0).unwrap() - 0.5).abs() < 1e-12);
        assert!((spec.compute_deficit(35.0).unwrap() - 0.5).abs() < 1e-12);

        // Saturación estricta en [0.0, 1.0] fuera de límites.
        assert!((spec.compute_deficit(-100.0).unwrap() - 1.0).abs() < 1e-12);
        assert!((spec.compute_deficit(500.0).unwrap() - 1.0).abs() < 1e-12);

        // Entradas no finitas rechazadas explícitamente.
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                spec.compute_deficit(bad),
                Err(MotivationError::NonFiniteInput {
                    field: "drive.value",
                    ..
                })
            ));
            assert!(matches!(
                DriveSpec::new(drive_id.clone(), bad, 0.0, 1.0, 1.0, 1.0),
                Err(MotivationError::NonFiniteInput {
                    field: "drive_spec.target",
                    ..
                })
            ));
            assert!(matches!(
                DriveSpec::new(drive_id.clone(), 0.5, bad, 1.0, 1.0, 1.0),
                Err(MotivationError::NonFiniteInput {
                    field: "drive_spec.min_bound",
                    ..
                })
            ));
            assert!(matches!(
                DriveSpec::new(drive_id.clone(), 0.5, 0.0, bad, 1.0, 1.0),
                Err(MotivationError::NonFiniteInput {
                    field: "drive_spec.max_bound",
                    ..
                })
            ));
            assert!(matches!(
                DriveSpec::new(drive_id.clone(), 0.5, 0.0, 1.0, bad, 1.0),
                Err(MotivationError::NonFiniteInput {
                    field: "drive_spec.base_weight",
                    ..
                })
            ));
            assert!(matches!(
                DriveSpec::new(drive_id.clone(), 0.5, 0.0, 1.0, 1.0, bad),
                Err(MotivationError::NonFiniteInput {
                    field: "drive_spec.deficit_gain",
                    ..
                })
            ));

            let act = Action::discrete_in_space(0, &ActionSpace::discrete(1).unwrap()).unwrap();
            assert!(matches!(
                ActionCandidate::new(act).with_score("obj", bad),
                Err(MotivationError::NonFiniteInput {
                    field: "candidate.objective_score",
                    ..
                })
            ));
        }

        // Cotas inválidas y pesos negativos rechazados explícitamente.
        assert!(matches!(
            DriveSpec::new(drive_id.clone(), 0.5, 1.0, 0.0, 1.0, 1.0),
            Err(MotivationError::InvalidBounds { .. })
        ));
        assert!(matches!(
            DriveSpec::new(drive_id.clone(), 2.0, 0.0, 1.0, 1.0, 1.0),
            Err(MotivationError::InvalidBounds { .. })
        ));
        assert!(matches!(
            DriveSpec::new(drive_id.clone(), 0.5, 0.0, 1.0, -0.1, 1.0),
            Err(MotivationError::InvalidWeight {
                field: "drive_spec.base_weight",
                ..
            })
        ));
        assert!(matches!(
            DriveSpec::new(drive_id, 0.5, 0.0, 1.0, 1.0, -0.5),
            Err(MotivationError::InvalidWeight {
                field: "drive_spec.deficit_gain",
                ..
            })
        ));
    }

    #[test]
    fn ac3_high_utility_objective_never_overrides_hard_constraint() {
        let space = ActionSpace::discrete(2).unwrap();
        let forbidden_overdrive = Action::discrete_in_space(0, &space).unwrap();
        let safe_step = Action::discrete_in_space(1, &space).unwrap();

        let drive_id = DriveId::new("throughput_urge").unwrap();
        let spec = DriveSpec::new(drive_id.clone(), 100.0, 0.0, 100.0, 10.0, 100.0).unwrap();
        let mut state = HomeostaticState::new(vec![spec]).unwrap();
        // Déficit máximo (1.0) -> peso efectivo altísimo.
        state.set_reading(&drive_id, 0.0).unwrap();

        let utility = DynamicUtility::new(vec![
            WeightedObjective::modulated("mega_reward", drive_id, 1_000.0, 10_000.0).unwrap(),
        ])
        .unwrap();

        // Regla inviolable: prohíbe `forbidden_overdrive` (índice 0).
        let evaluator =
            UtilityEvaluator::new(utility).with_constraint(ForbiddenActionRule::discrete_index(
                "no_overdrive_in_safety_zone",
                0,
                "safety interlock forbids overdrive action",
            ));

        let candidates = vec![
            ActionCandidate::new(forbidden_overdrive.clone())
                .with_score("mega_reward", 1_000_000.0)
                .unwrap(),
            ActionCandidate::new(safe_step.clone())
                .with_score("mega_reward", 1.0)
                .unwrap(),
        ];

        let outcome = evaluator.evaluate_candidates(&candidates, &state).unwrap();

        // La acción prohibida tiene utilidad masiva, pero es rechazada por HardConstraint.
        assert!(!outcome.evaluations()[0].is_allowed());
        assert_eq!(outcome.evaluations()[0].violations().len(), 1);
        assert!(outcome.evaluations()[0].utility() > outcome.evaluations()[1].utility());

        // Se selecciona únicamente la acción segura permitida.
        assert_eq!(outcome.selected().unwrap().action(), &safe_step);

        // Si todas las candidatas están prohibidas, selected() es None (nunca fuerza una prohibida).
        let only_forbidden = vec![candidates[0].clone()];
        let all_blocked = evaluator
            .evaluate_candidates(&only_forbidden, &state)
            .unwrap();
        assert!(all_blocked.selected().is_none());
        assert!(!all_blocked.evaluations()[0].is_allowed());
    }
}
