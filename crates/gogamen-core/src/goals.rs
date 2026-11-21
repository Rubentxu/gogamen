//! Gestión dinámica de objetivos e intenciones con autorización y auditoría (`std`-only).
//!
//! Este módulo separa estrictamente:
//! - **Feedback semántico / observaciones** ([`ObservationEnvelope`], [`ObservationKind`],
//!   [`PendingOrRejectedObservation`]): las propuestas externas sin autorización explícita
//!   o expiradas se registran únicamente como observaciones descartadas y **nunca** mutan
//!   los objetivos activos ni disparan ejecución.
//! - **Permiso explícito de mutación** ([`GoalAuthority`] / [`GoalAuthorization`]): solo una
//!   [`GoalChangeProposal`] vigente (`now <= expires_at`), con autoridad válida para el
//!   [`GoalId`] objetivo y revisión estrictamente creciente (`revision > prev_revision`),
//!   actualiza [`GoalManager`].
//! - **Integración directa con utilidad dinámica** ([`GoalManager::to_weighted_objectives`],
//!   [`GoalManager::to_dynamic_utility`]): los cambios de objetivos y prioridades se reflejan
//!   de inmediato en [`DynamicUtility`] y [`crate::UtilityEvaluator`] sin recompilar ningún plan.
//! - **Auditoría y replay determinista** ([`GoalTransitionRecord`], [`GoalManager::replay`]):
//!   cada transición registra secuencia, `origin`, `revision`, `timestamp` y `outcome`
//!   ([`GoalTransitionOutcome`]).

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::clock::MonotonicInstant;
use crate::motivation::{DriveId, DynamicUtility, MotivationError, WeightedObjective};
use crate::provenance::{ObservationEnvelope, ObservationKind, SensorId};

/// Errores explícitos de validación o reconstrucción en la gestión de objetivos.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalError {
    /// Un identificador requerido está vacío o solo contiene espacios.
    EmptyIdentifier {
        /// Nombre del campo vacío.
        field: &'static str,
    },
    /// Un peso numérico es `NaN`, `+inf` o `-inf`.
    NonFiniteWeight {
        /// Campo donde se detectó el valor no finito.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Un peso numérico es negativo.
    NegativeWeight {
        /// Campo donde se detectó el valor negativo.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Error al convertir objetivos activos en utilidad dinámica.
    Motivation(MotivationError),
    /// El log de replay contiene una secuencia discontinua o incoherente.
    InvalidReplayRecord {
        /// Índice de secuencia esperado.
        expected_sequence: u64,
        /// Índice de secuencia encontrado.
        found_sequence: u64,
    },
}

impl fmt::Display for GoalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { field } => {
                write!(f, "goal identifier '{field}' cannot be empty")
            }
            Self::NonFiniteWeight { field, value } => {
                write!(f, "non-finite goal weight in '{field}': {value}")
            }
            Self::NegativeWeight { field, value } => {
                write!(
                    f,
                    "negative goal weight in '{field}': {value} (must be >= 0.0)"
                )
            }
            Self::Motivation(err) => write!(f, "motivation conversion error: {err}"),
            Self::InvalidReplayRecord {
                expected_sequence,
                found_sequence,
            } => write!(
                f,
                "invalid goal replay sequence: expected {expected_sequence}, found {found_sequence}"
            ),
        }
    }
}

impl Error for GoalError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Motivation(err) => Some(err),
            Self::EmptyIdentifier { .. }
            | Self::NonFiniteWeight { .. }
            | Self::NegativeWeight { .. }
            | Self::InvalidReplayRecord { .. } => None,
        }
    }
}

impl From<MotivationError> for GoalError {
    fn from(value: MotivationError) -> Self {
        Self::Motivation(value)
    }
}

fn validate_non_empty(field: &'static str, raw: impl Into<String>) -> Result<String, GoalError> {
    let value = raw.into();
    if value.trim().is_empty() {
        return Err(GoalError::EmptyIdentifier { field });
    }
    Ok(value)
}

fn validate_non_negative_weight(field: &'static str, value: f64) -> Result<f64, GoalError> {
    if !value.is_finite() {
        return Err(GoalError::NonFiniteWeight { field, value });
    }
    if value < 0.0 {
        return Err(GoalError::NegativeWeight { field, value });
    }
    Ok(value)
}

/// Identificador canónico y no vacío de un objetivo.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GoalId(String);

impl GoalId {
    /// Construye un [`GoalId`] validando que no esté vacío ni contenga solo espacios.
    pub fn new(id: impl Into<String>) -> Result<Self, GoalError> {
        Ok(Self(validate_non_empty("goal_id", id)?))
    }

    /// Devuelve el identificador como `&str`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for GoalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Origen estructurado de una propuesta o transición de objetivo.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GoalOrigin {
    /// Módulo interno del sistema o controlador.
    System(String),
    /// Operador humano externo.
    Human(String),
    /// Agente externo o par.
    Agent(String),
    /// Fuente sensorial identificada por [`SensorId`].
    Sensor(SensorId),
}

impl GoalOrigin {
    /// Crea un origen interno de sistema validando que no esté vacío.
    pub fn system(id: impl Into<String>) -> Result<Self, GoalError> {
        Ok(Self::System(validate_non_empty("origin.system", id)?))
    }

    /// Crea un origen de operador humano validando que no esté vacío.
    pub fn human(id: impl Into<String>) -> Result<Self, GoalError> {
        Ok(Self::Human(validate_non_empty("origin.human", id)?))
    }

    /// Crea un origen de agente externo validando que no esté vacío.
    pub fn agent(id: impl Into<String>) -> Result<Self, GoalError> {
        Ok(Self::Agent(validate_non_empty("origin.agent", id)?))
    }

    /// Crea un origen sensorial a partir de un [`SensorId`].
    pub const fn sensor(sensor_id: SensorId) -> Self {
        Self::Sensor(sensor_id)
    }
}

impl fmt::Display for GoalOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::System(id) => write!(f, "system:{id}"),
            Self::Human(id) => write!(f, "human:{id}"),
            Self::Agent(id) => write!(f, "agent:{id}"),
            Self::Sensor(sensor_id) => write!(f, "sensor:{}", sensor_id.as_u64()),
        }
    }
}

/// Alcance permitido por una autorización de mutación de objetivos.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GoalAuthorityScope {
    /// Autoriza mutar cualquier [`GoalId`].
    AnyGoal,
    /// Autoriza mutar únicamente el [`GoalId`] especificado.
    SpecificGoal(GoalId),
}

/// Credencial explícita de autorización para mutar objetivos en [`GoalManager`].
///
/// Nunca se infiere a partir de [`crate::Confidence`], [`ObservationKind`] ni texto libre.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GoalAuthority {
    issuer: String,
    scope: GoalAuthorityScope,
}

/// Alias semántico para [`GoalAuthority`].
pub type GoalAuthorization = GoalAuthority;

impl GoalAuthority {
    /// Crea una autorización explícita emitida por `issuer` con alcance sobre cualquier objetivo.
    pub fn for_any_goal(issuer: impl Into<String>) -> Result<Self, GoalError> {
        Ok(Self {
            issuer: validate_non_empty("authority.issuer", issuer)?,
            scope: GoalAuthorityScope::AnyGoal,
        })
    }

    /// Crea una autorización explícita emitida por `issuer` restringida a `goal_id`.
    pub fn for_goal(issuer: impl Into<String>, goal_id: GoalId) -> Result<Self, GoalError> {
        Ok(Self {
            issuer: validate_non_empty("authority.issuer", issuer)?,
            scope: GoalAuthorityScope::SpecificGoal(goal_id),
        })
    }

    /// Devuelve el emisor de la autorización.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// Devuelve el alcance de la autorización.
    pub const fn scope(&self) -> &GoalAuthorityScope {
        &self.scope
    }

    /// Indica si esta autorización permite mutar el `goal_id` dado.
    pub fn authorizes(&self, goal_id: &GoalId) -> bool {
        match &self.scope {
            GoalAuthorityScope::AnyGoal => true,
            GoalAuthorityScope::SpecificGoal(allowed) => allowed == goal_id,
        }
    }
}

/// Mutación solicitada sobre un objetivo.
#[derive(Debug, Clone, PartialEq)]
pub enum GoalMutation {
    /// Crea o actualiza un objetivo activo con prioridad y parámetros de utilidad.
    Upsert {
        /// Prioridad explícita (mayor valor implica mayor prioridad).
        priority: u32,
        /// Peso base no negativo en la función de utilidad.
        base_weight: f64,
        /// Ganancia adicional por déficit si el objetivo está modulado por un [`DriveId`].
        deficit_gain: f64,
        /// Drive homeostático modulador opcional.
        drive_id: Option<DriveId>,
    },
    /// Desactiva y retira un objetivo activo.
    Deactivate,
}

/// Propuesta externa o interna de cambio de objetivo sometida a [`GoalManager`].
#[derive(Debug, Clone, PartialEq)]
pub struct GoalChangeProposal {
    goal_id: GoalId,
    origin: GoalOrigin,
    revision: u64,
    expires_at: Option<MonotonicInstant>,
    authority: Option<GoalAuthority>,
    mutation: GoalMutation,
}

/// Alias semántico para [`GoalChangeProposal`].
pub type GoalProposal = GoalChangeProposal;

impl GoalChangeProposal {
    /// Construye una propuesta de creación/actualización de objetivo con peso estático.
    ///
    /// Por defecto carece de autorización (`authority: None`) y de caducidad (`expires_at: None`).
    pub fn upsert(
        goal_id: GoalId,
        origin: GoalOrigin,
        revision: u64,
        priority: u32,
        base_weight: f64,
    ) -> Result<Self, GoalError> {
        let base_weight = validate_non_negative_weight("proposal.base_weight", base_weight)?;
        Ok(Self {
            goal_id,
            origin,
            revision,
            expires_at: None,
            authority: None,
            mutation: GoalMutation::Upsert {
                priority,
                base_weight,
                deficit_gain: 0.0,
                drive_id: None,
            },
        })
    }

    /// Construye una propuesta de creación/actualización modulada por un [`DriveId`].
    pub fn upsert_modulated(
        goal_id: GoalId,
        origin: GoalOrigin,
        revision: u64,
        priority: u32,
        drive_id: DriveId,
        base_weight: f64,
        deficit_gain: f64,
    ) -> Result<Self, GoalError> {
        let base_weight = validate_non_negative_weight("proposal.base_weight", base_weight)?;
        let deficit_gain = validate_non_negative_weight("proposal.deficit_gain", deficit_gain)?;
        Ok(Self {
            goal_id,
            origin,
            revision,
            expires_at: None,
            authority: None,
            mutation: GoalMutation::Upsert {
                priority,
                base_weight,
                deficit_gain,
                drive_id: Some(drive_id),
            },
        })
    }

    /// Construye una propuesta de desactivación de un objetivo existente.
    pub const fn deactivate(goal_id: GoalId, origin: GoalOrigin, revision: u64) -> Self {
        Self {
            goal_id,
            origin,
            revision,
            expires_at: None,
            authority: None,
            mutation: GoalMutation::Deactivate,
        }
    }

    /// Asocia un instante monotónico límite de vigencia (`now <= expires_at`).
    pub const fn with_expires_at(mut self, expires_at: MonotonicInstant) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Adjunta una credencial explícita de autorización [`GoalAuthority`].
    pub fn with_authority(mut self, authority: GoalAuthority) -> Self {
        self.authority = Some(authority);
        self
    }

    /// Identificador del objetivo afectado.
    pub const fn goal_id(&self) -> &GoalId {
        &self.goal_id
    }

    /// Origen de la propuesta.
    pub const fn origin(&self) -> &GoalOrigin {
        &self.origin
    }

    /// Revisión monotónica declarada en la propuesta.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Instante límite de vigencia, si existe.
    pub const fn expires_at(&self) -> Option<MonotonicInstant> {
        self.expires_at
    }

    /// Autorización adjunta, si existe.
    pub const fn authority(&self) -> Option<&GoalAuthority> {
        self.authority.as_ref()
    }

    /// Mutación solicitada.
    pub const fn mutation(&self) -> &GoalMutation {
        &self.mutation
    }
}

/// Estado de un objetivo actualmente activo en [`GoalManager`].
#[derive(Debug, Clone, PartialEq)]
pub struct ActiveGoal {
    id: GoalId,
    priority: u32,
    revision: u64,
    base_weight: f64,
    deficit_gain: f64,
    drive_id: Option<DriveId>,
    origin: GoalOrigin,
    updated_at: MonotonicInstant,
    expires_at: Option<MonotonicInstant>,
}

impl ActiveGoal {
    /// Identificador del objetivo.
    pub const fn id(&self) -> &GoalId {
        &self.id
    }

    /// Prioridad explícita (mayor valor = mayor prioridad).
    pub const fn priority(&self) -> u32 {
        self.priority
    }

    /// Última revisión monotónica aplicada.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Peso base de utilidad.
    pub const fn base_weight(&self) -> f64 {
        self.base_weight
    }

    /// Ganancia por déficit de drive.
    pub const fn deficit_gain(&self) -> f64 {
        self.deficit_gain
    }

    /// Drive modulador asociado, si existe.
    pub const fn drive_id(&self) -> Option<&DriveId> {
        self.drive_id.as_ref()
    }

    /// Origen que estableció el estado actual del objetivo.
    pub const fn origin(&self) -> &GoalOrigin {
        &self.origin
    }

    /// Instante monotónico en que se aplicó la última mutación.
    pub const fn updated_at(&self) -> MonotonicInstant {
        self.updated_at
    }

    /// Instante límite de vigencia, si existe.
    pub const fn expires_at(&self) -> Option<MonotonicInstant> {
        self.expires_at
    }

    /// Indica si el objetivo sigue vigente en `now` (`now <= expires_at`).
    pub fn is_valid_at(&self, now: MonotonicInstant) -> bool {
        match self.expires_at {
            Some(limit) => now <= limit,
            None => true,
        }
    }

    /// Convierte el objetivo activo en un [`WeightedObjective`] integrable en [`DynamicUtility`].
    pub fn to_weighted_objective(&self) -> Result<WeightedObjective, MotivationError> {
        match &self.drive_id {
            Some(drive_id) => WeightedObjective::modulated(
                self.id.as_str(),
                drive_id.clone(),
                self.base_weight,
                self.deficit_gain,
            ),
            None => WeightedObjective::static_weight(self.id.as_str(), self.base_weight),
        }
    }
}

/// Resultado determinista de evaluar una propuesta de cambio de objetivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalTransitionOutcome {
    /// La propuesta estaba autorizada, vigente y con revisión estrictamente superior; se aplicó.
    Applied,
    /// La propuesta carecía de [`GoalAuthority`] válida para el objetivo; no mutó estado.
    RejectedUnauthorized,
    /// La propuesta superó su ventana de vigencia (`now > expires_at`); no mutó estado.
    RejectedExpired,
    /// La revisión de la propuesta no superó la última revisión registrada para ese [`GoalId`].
    RejectedStaleRevision,
}

/// Registro de observación o propuesta descartada sin mutar los objetivos activos.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingOrRejectedObservation {
    goal_id: GoalId,
    origin: GoalOrigin,
    revision: u64,
    observed_at: MonotonicInstant,
    expires_at: Option<MonotonicInstant>,
    outcome: GoalTransitionOutcome,
    mutation: GoalMutation,
}

impl PendingOrRejectedObservation {
    /// Identificador del objetivo al que hacía referencia la observación.
    pub const fn goal_id(&self) -> &GoalId {
        &self.goal_id
    }

    /// Origen de la observación o propuesta descartada.
    pub const fn origin(&self) -> &GoalOrigin {
        &self.origin
    }

    /// Revisión declarada en la observación.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Instante monotónico en que se evaluó la observación.
    pub const fn observed_at(&self) -> MonotonicInstant {
        self.observed_at
    }

    /// Caducidad declarada en la propuesta, si existía.
    pub const fn expires_at(&self) -> Option<MonotonicInstant> {
        self.expires_at
    }

    /// Motivo de rechazo o descarte.
    pub const fn outcome(&self) -> GoalTransitionOutcome {
        self.outcome
    }

    /// Carga de mutación solicitada (conservada solo como registro sin ejecución).
    pub const fn mutation(&self) -> &GoalMutation {
        &self.mutation
    }
}

/// Registro inmutable de auditoría y replay de una transición evaluada por [`GoalManager`].
#[derive(Debug, Clone, PartialEq)]
pub struct GoalTransitionRecord {
    sequence: u64,
    goal_id: GoalId,
    origin: GoalOrigin,
    revision: u64,
    timestamp: MonotonicInstant,
    outcome: GoalTransitionOutcome,
    mutation: GoalMutation,
    expires_at: Option<MonotonicInstant>,
    authority_issuer: Option<String>,
}

impl GoalTransitionRecord {
    /// Índice secuencial determinista de la transición.
    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Objetivo afectado por la propuesta.
    pub const fn goal_id(&self) -> &GoalId {
        &self.goal_id
    }

    /// Origen que propuso la transición.
    pub const fn origin(&self) -> &GoalOrigin {
        &self.origin
    }

    /// Revisión declarada en la propuesta.
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    /// Instante monotónico en que se evaluó la transición.
    pub const fn timestamp(&self) -> MonotonicInstant {
        self.timestamp
    }

    /// Resultado de la evaluación.
    pub const fn outcome(&self) -> GoalTransitionOutcome {
        self.outcome
    }

    /// Mutación solicitada.
    pub const fn mutation(&self) -> &GoalMutation {
        &self.mutation
    }

    /// Límite de vigencia declarado, si existía.
    pub const fn expires_at(&self) -> Option<MonotonicInstant> {
        self.expires_at
    }

    /// Emisor de la autoridad adjunta, si se presentó alguna.
    pub fn authority_issuer(&self) -> Option<&str> {
        self.authority_issuer.as_deref()
    }
}

/// Gestor de objetivos activos, propuestas externas, observaciones descartadas y replay.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GoalManager {
    active_goals: BTreeMap<GoalId, ActiveGoal>,
    revisions: BTreeMap<GoalId, u64>,
    rejected_observations: Vec<PendingOrRejectedObservation>,
    transition_log: Vec<GoalTransitionRecord>,
}

impl GoalManager {
    /// Crea un gestor vacío sin objetivos activos ni transiciones registradas.
    pub fn new() -> Self {
        Self::default()
    }

    /// Evalúa una [`GoalChangeProposal`] en el instante monotónico `now`.
    ///
    /// - Solo muta objetivos activos si tiene [`GoalAuthority`] válida para `goal_id`,
    ///   está vigente (`now <= expires_at`) y su revisión es estrictamente mayor que la
    ///   última revisión aplicada para ese `goal_id`.
    /// - Cualquier propuesta no autorizada, expirada o con revisión obsoleta se registra
    ///   en [`GoalManager::pending_or_rejected_observations`] sin mutar objetivos activos.
    /// - Todas las transiciones quedan registradas en [`GoalManager::transition_log`].
    pub fn apply_proposal(
        &mut self,
        proposal: GoalChangeProposal,
        now: MonotonicInstant,
    ) -> GoalTransitionOutcome {
        let authority_issuer = proposal.authority.as_ref().map(|a| a.issuer().to_owned());

        let is_authorized = proposal
            .authority
            .as_ref()
            .is_some_and(|auth| auth.authorizes(&proposal.goal_id));

        let outcome = if !is_authorized {
            GoalTransitionOutcome::RejectedUnauthorized
        } else if proposal
            .expires_at
            .is_some_and(|expires_at| now > expires_at)
        {
            GoalTransitionOutcome::RejectedExpired
        } else if self
            .revisions
            .get(&proposal.goal_id)
            .is_some_and(|&prev_rev| proposal.revision <= prev_rev)
        {
            GoalTransitionOutcome::RejectedStaleRevision
        } else {
            GoalTransitionOutcome::Applied
        };

        if outcome == GoalTransitionOutcome::Applied {
            self.apply_mutation_unchecked(
                &proposal.goal_id,
                &proposal.origin,
                proposal.revision,
                now,
                proposal.expires_at,
                &proposal.mutation,
            );
        } else {
            self.rejected_observations
                .push(PendingOrRejectedObservation {
                    goal_id: proposal.goal_id.clone(),
                    origin: proposal.origin.clone(),
                    revision: proposal.revision,
                    observed_at: now,
                    expires_at: proposal.expires_at,
                    outcome,
                    mutation: proposal.mutation.clone(),
                });
        }

        let sequence = self.transition_log.len() as u64;
        self.transition_log.push(GoalTransitionRecord {
            sequence,
            goal_id: proposal.goal_id,
            origin: proposal.origin,
            revision: proposal.revision,
            timestamp: now,
            outcome,
            mutation: proposal.mutation,
            expires_at: proposal.expires_at,
            authority_issuer,
        });

        outcome
    }

    /// Evalúa un [`ObservationEnvelope<GoalChangeProposal>`] manteniendo aislamiento epistémico.
    ///
    /// Si `envelope.claim_kind()` no es [`ObservationKind::Proposal`], la observación se
    /// rechaza como [`GoalTransitionOutcome::RejectedUnauthorized`] sin ejecutar mutación,
    /// incluso si tuviera alta confianza (`Confidence`). Si es `Proposal`, delega en
    /// [`GoalManager::apply_proposal`], donde sigue requiriéndose [`GoalAuthority`] explícita.
    pub fn apply_observation_envelope(
        &mut self,
        envelope: &ObservationEnvelope<GoalChangeProposal>,
        now: MonotonicInstant,
    ) -> GoalTransitionOutcome {
        let mut proposal = envelope.payload().clone();
        if envelope.claim_kind() != ObservationKind::Proposal {
            // El feedback semántico no-Proposal jamás puede portar autoridad de mutación.
            proposal.authority = None;
        }
        self.apply_proposal(proposal, now)
    }

    fn apply_mutation_unchecked(
        &mut self,
        goal_id: &GoalId,
        origin: &GoalOrigin,
        revision: u64,
        timestamp: MonotonicInstant,
        expires_at: Option<MonotonicInstant>,
        mutation: &GoalMutation,
    ) {
        self.revisions.insert(goal_id.clone(), revision);
        match mutation {
            GoalMutation::Upsert {
                priority,
                base_weight,
                deficit_gain,
                drive_id,
            } => {
                self.active_goals.insert(
                    goal_id.clone(),
                    ActiveGoal {
                        id: goal_id.clone(),
                        priority: *priority,
                        revision,
                        base_weight: *base_weight,
                        deficit_gain: *deficit_gain,
                        drive_id: drive_id.clone(),
                        origin: origin.clone(),
                        updated_at: timestamp,
                        expires_at,
                    },
                );
            }
            GoalMutation::Deactivate => {
                self.active_goals.remove(goal_id);
            }
        }
    }

    /// Reconstruye exactamente un [`GoalManager`] por replay determinista de un log de transiciones.
    pub fn replay(records: &[GoalTransitionRecord]) -> Result<Self, GoalError> {
        let mut manager = Self::new();
        for (idx, record) in records.iter().enumerate() {
            let expected_sequence = idx as u64;
            if record.sequence != expected_sequence {
                return Err(GoalError::InvalidReplayRecord {
                    expected_sequence,
                    found_sequence: record.sequence,
                });
            }

            if record.outcome == GoalTransitionOutcome::Applied {
                manager.apply_mutation_unchecked(
                    &record.goal_id,
                    &record.origin,
                    record.revision,
                    record.timestamp,
                    record.expires_at,
                    &record.mutation,
                );
            } else {
                manager
                    .rejected_observations
                    .push(PendingOrRejectedObservation {
                        goal_id: record.goal_id.clone(),
                        origin: record.origin.clone(),
                        revision: record.revision,
                        observed_at: record.timestamp,
                        expires_at: record.expires_at,
                        outcome: record.outcome,
                        mutation: record.mutation.clone(),
                    });
            }

            manager.transition_log.push(record.clone());
        }
        Ok(manager)
    }

    /// Elimina objetivos activos cuyo `expires_at` haya expirado en `now` (`now > expires_at`).
    ///
    /// Devuelve la cantidad de objetivos expirados retirados.
    pub fn prune_expired(&mut self, now: MonotonicInstant) -> usize {
        let before = self.active_goals.len();
        self.active_goals.retain(|_, goal| goal.is_valid_at(now));
        before - self.active_goals.len()
    }

    /// Devuelve un objetivo activo por [`GoalId`], si existe.
    pub fn get(&self, goal_id: &GoalId) -> Option<&ActiveGoal> {
        self.active_goals.get(goal_id)
    }

    /// Devuelve los objetivos activos ordenados deterministamente por prioridad descendente
    /// y [`GoalId`] ascendente en caso de empate.
    pub fn active_goals(&self) -> Vec<&ActiveGoal> {
        let mut goals: Vec<&ActiveGoal> = self.active_goals.values().collect();
        goals.sort_by(|a, b| b.priority.cmp(&a.priority).then_with(|| a.id.cmp(&b.id)));
        goals
    }

    /// Devuelve los objetivos activos y vigentes en `now` (`now <= expires_at`), ordenados
    /// por prioridad descendente y [`GoalId`] ascendente.
    pub fn active_goals_at(&self, now: MonotonicInstant) -> Vec<&ActiveGoal> {
        self.active_goals()
            .into_iter()
            .filter(|goal| goal.is_valid_at(now))
            .collect()
    }

    /// Proyecta los objetivos activos como [`WeightedObjective`] integrables directamente
    /// con [`DynamicUtility`] y [`crate::UtilityEvaluator`] sin recompilar ningún plan.
    pub fn to_weighted_objectives(&self) -> Result<Vec<WeightedObjective>, MotivationError> {
        self.active_goals()
            .into_iter()
            .map(ActiveGoal::to_weighted_objective)
            .collect()
    }

    /// Proyecta los objetivos activos vigentes en `now` como [`WeightedObjective`].
    pub fn to_weighted_objectives_at(
        &self,
        now: MonotonicInstant,
    ) -> Result<Vec<WeightedObjective>, MotivationError> {
        self.active_goals_at(now)
            .into_iter()
            .map(ActiveGoal::to_weighted_objective)
            .collect()
    }

    /// Construye una [`DynamicUtility`] directamente a partir de los objetivos activos.
    pub fn to_dynamic_utility(&self) -> Result<DynamicUtility, MotivationError> {
        DynamicUtility::new(self.to_weighted_objectives()?)
    }

    /// Construye una [`DynamicUtility`] directamente a partir de los objetivos vigentes en `now`.
    pub fn to_dynamic_utility_at(
        &self,
        now: MonotonicInstant,
    ) -> Result<DynamicUtility, MotivationError> {
        DynamicUtility::new(self.to_weighted_objectives_at(now)?)
    }

    /// Registro de observaciones/propuestas descartadas sin mutar objetivos activos.
    pub fn pending_or_rejected_observations(&self) -> &[PendingOrRejectedObservation] {
        &self.rejected_observations
    }

    /// Registro completo de auditoría y replay de transiciones evaluadas.
    pub fn transition_log(&self) -> &[GoalTransitionRecord] {
        &self.transition_log
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::action::{Action, ActionSpace};
    use crate::clock::{MonotonicClock, VirtualMonotonicClock};
    use crate::motivation::{ActionCandidate, DriveSpec, HomeostaticState, UtilityEvaluator};
    use crate::provenance::{
        Confidence, EvidenceRef, FrameId, MessageId, ProvenanceMetadata, UnitTag,
    };

    fn make_metadata(kind: ObservationKind, now: MonotonicInstant) -> ProvenanceMetadata {
        ProvenanceMetadata {
            message_id: MessageId::new(1),
            observed_at: now,
            received_at: now,
            sensor_id: SensorId::new(7),
            frame_id: FrameId::world(),
            units: UnitTag::new("ratio").unwrap(),
            confidence: Confidence::new(1.0).unwrap(),
            revision: 1,
            evidence_ref: EvidenceRef::new(99),
            claim_kind: kind,
        }
    }

    #[test]
    fn ac1_authorized_external_goal_modifies_utility_without_recompiling_plan() {
        let clock = VirtualMonotonicClock::new();
        let mut manager = GoalManager::new();

        let goal_explore = GoalId::new("explore").unwrap();
        let goal_recharge = GoalId::new("recharge").unwrap();
        let sys_origin = GoalOrigin::system("bootstrap").unwrap();
        let sys_auth = GoalAuthority::for_any_goal("kernel").unwrap();

        let initial_explore =
            GoalChangeProposal::upsert(goal_explore.clone(), sys_origin.clone(), 1, 10, 5.0)
                .unwrap()
                .with_authority(sys_auth.clone());
        let initial_recharge =
            GoalChangeProposal::upsert(goal_recharge.clone(), sys_origin, 1, 5, 1.0)
                .unwrap()
                .with_authority(sys_auth);

        assert_eq!(
            manager.apply_proposal(initial_explore, clock.now()),
            GoalTransitionOutcome::Applied
        );
        assert_eq!(
            manager.apply_proposal(initial_recharge, clock.now()),
            GoalTransitionOutcome::Applied
        );

        let space = ActionSpace::discrete(2).unwrap();
        let act_explore = Action::discrete_in_space(0, &space).unwrap();
        let act_recharge = Action::discrete_in_space(1, &space).unwrap();
        let candidates = vec![
            ActionCandidate::new(act_explore.clone())
                .with_score("explore", 1.0)
                .unwrap()
                .with_score("recharge", 0.0)
                .unwrap(),
            ActionCandidate::new(act_recharge.clone())
                .with_score("explore", 0.0)
                .unwrap()
                .with_score("recharge", 1.0)
                .unwrap(),
        ];

        let battery_id = DriveId::new("battery").unwrap();
        let state = HomeostaticState::new(vec![
            DriveSpec::new(battery_id.clone(), 1.0, 0.0, 1.0, 1.0, 4.0).unwrap(),
        ])
        .unwrap();

        // Evaluación inicial: gana explorar (utilidad 5.0 vs 1.0).
        let eval_before = UtilityEvaluator::new(manager.to_dynamic_utility().unwrap());
        let outcome_before = eval_before
            .evaluate_candidates(&candidates, &state)
            .unwrap();
        assert_eq!(outcome_before.selected().unwrap().action(), &act_explore);

        // Objetivo externo autorizado (vigente hasta t + 10s) cambia prioridad y peso de recharge.
        clock.advance(Duration::from_secs(2));
        let deadline = clock.now().checked_add(Duration::from_secs(10)).unwrap();
        let ext_origin = GoalOrigin::human("operator-1").unwrap();
        let ext_auth = GoalAuthority::for_goal("mission-control", goal_recharge.clone()).unwrap();
        let ext_proposal = GoalChangeProposal::upsert_modulated(
            goal_recharge.clone(),
            ext_origin,
            2,
            20,
            battery_id,
            8.0,
            2.0,
        )
        .unwrap()
        .with_expires_at(deadline)
        .with_authority(ext_auth);

        assert_eq!(
            manager.apply_proposal(ext_proposal, clock.now()),
            GoalTransitionOutcome::Applied
        );

        // Sin recompilar plan, to_weighted_objectives() / to_dynamic_utility() refleja el cambio.
        let weighted = manager.to_weighted_objectives().unwrap();
        assert_eq!(weighted[0].name(), "recharge");
        assert_eq!(weighted[1].name(), "explore");

        let eval_after = UtilityEvaluator::new(manager.to_dynamic_utility().unwrap());
        let outcome_after = eval_after.evaluate_candidates(&candidates, &state).unwrap();
        assert_eq!(outcome_after.selected().unwrap().action(), &act_recharge);
    }

    #[test]
    fn ac2_unauthorized_or_expired_proposal_is_recorded_as_observation_without_execution() {
        let clock = VirtualMonotonicClock::new();
        let mut manager = GoalManager::new();

        let goal_a = GoalId::new("goal-a").unwrap();
        let goal_b = GoalId::new("goal-b").unwrap();

        let initial = GoalChangeProposal::upsert(
            goal_a.clone(),
            GoalOrigin::system("init").unwrap(),
            1,
            10,
            2.0,
        )
        .unwrap()
        .with_authority(GoalAuthority::for_any_goal("kernel").unwrap());
        assert_eq!(
            manager.apply_proposal(initial, clock.now()),
            GoalTransitionOutcome::Applied
        );

        // 1) Propuesta sin autoridad.
        let no_auth = GoalChangeProposal::upsert(
            goal_a.clone(),
            GoalOrigin::agent("external-llm").unwrap(),
            2,
            99,
            100.0,
        )
        .unwrap();
        assert_eq!(
            manager.apply_proposal(no_auth, clock.now()),
            GoalTransitionOutcome::RejectedUnauthorized
        );

        // 2) Propuesta con autoridad emitida para otro GoalId.
        let wrong_scope = GoalChangeProposal::upsert(
            goal_a.clone(),
            GoalOrigin::agent("peer").unwrap(),
            3,
            99,
            100.0,
        )
        .unwrap()
        .with_authority(GoalAuthority::for_goal("peer-auth", goal_b.clone()).unwrap());
        assert_eq!(
            manager.apply_proposal(wrong_scope, clock.now()),
            GoalTransitionOutcome::RejectedUnauthorized
        );

        // 3) ObservationEnvelope con ObservationKind::Assertion aunque lleve autoridad adjunta y confidence 1.0.
        let semantic_payload = GoalChangeProposal::upsert(
            goal_a.clone(),
            GoalOrigin::sensor(SensorId::new(7)),
            4,
            99,
            100.0,
        )
        .unwrap()
        .with_authority(GoalAuthority::for_any_goal("spoofed").unwrap());
        let envelope = ObservationEnvelope::new(
            make_metadata(ObservationKind::Assertion, clock.now()),
            semantic_payload,
        );
        assert_eq!(
            manager.apply_observation_envelope(&envelope, clock.now()),
            GoalTransitionOutcome::RejectedUnauthorized
        );

        // 4) Propuesta autorizada pero expirada (now > expires_at).
        let expires_at = clock.now().checked_add(Duration::from_secs(5)).unwrap();
        clock.advance(Duration::from_secs(6));
        let expired = GoalChangeProposal::upsert(
            goal_a.clone(),
            GoalOrigin::human("operator").unwrap(),
            5,
            50,
            50.0,
        )
        .unwrap()
        .with_expires_at(expires_at)
        .with_authority(GoalAuthority::for_any_goal("mission-control").unwrap());
        assert_eq!(
            manager.apply_proposal(expired, clock.now()),
            GoalTransitionOutcome::RejectedExpired
        );

        // Ninguna de las propuestas rechazadas mutó el objetivo activo.
        let active = manager.get(&goal_a).unwrap();
        assert_eq!(active.priority(), 10);
        assert_eq!(active.base_weight(), 2.0);
        assert_eq!(active.revision(), 1);
        assert_eq!(manager.pending_or_rejected_observations().len(), 4);
        assert_eq!(
            manager.pending_or_rejected_observations()[3].outcome(),
            GoalTransitionOutcome::RejectedExpired
        );
    }

    #[test]
    fn ac3_replay_records_origin_revision_and_outcome_and_reconstructs_state() {
        let clock = VirtualMonotonicClock::new();
        let mut manager = GoalManager::new();

        let goal = GoalId::new("stabilize").unwrap();
        let auth = GoalAuthority::for_any_goal("supervisor").unwrap();

        let p1 = GoalChangeProposal::upsert(
            goal.clone(),
            GoalOrigin::system("boot").unwrap(),
            1,
            10,
            3.0,
        )
        .unwrap()
        .with_authority(auth.clone());
        assert_eq!(
            manager.apply_proposal(p1, clock.now()),
            GoalTransitionOutcome::Applied
        );

        clock.advance(Duration::from_millis(50));
        // Revisión obsoleta (1 <= 1).
        let p2_stale = GoalChangeProposal::upsert(
            goal.clone(),
            GoalOrigin::human("op-a").unwrap(),
            1,
            20,
            7.0,
        )
        .unwrap()
        .with_authority(auth.clone());
        assert_eq!(
            manager.apply_proposal(p2_stale, clock.now()),
            GoalTransitionOutcome::RejectedStaleRevision
        );

        clock.advance(Duration::from_millis(50));
        // No autorizada.
        let p3_unauth = GoalChangeProposal::upsert(
            goal.clone(),
            GoalOrigin::agent("rogue").unwrap(),
            2,
            99,
            99.0,
        )
        .unwrap();
        assert_eq!(
            manager.apply_proposal(p3_unauth, clock.now()),
            GoalTransitionOutcome::RejectedUnauthorized
        );

        clock.advance(Duration::from_millis(50));
        // Actualización válida en límite exacto de vigencia (now == expires_at).
        let exact_deadline = clock.now();
        let p4_valid = GoalChangeProposal::upsert(
            goal.clone(),
            GoalOrigin::human("op-b").unwrap(),
            2,
            30,
            9.0,
        )
        .unwrap()
        .with_expires_at(exact_deadline)
        .with_authority(auth);
        assert_eq!(
            manager.apply_proposal(p4_valid, clock.now()),
            GoalTransitionOutcome::Applied
        );

        let log = manager.transition_log();
        assert_eq!(log.len(), 4);
        assert_eq!(log[0].origin(), &GoalOrigin::system("boot").unwrap());
        assert_eq!(log[0].revision(), 1);
        assert_eq!(log[0].outcome(), GoalTransitionOutcome::Applied);

        assert_eq!(log[1].origin(), &GoalOrigin::human("op-a").unwrap());
        assert_eq!(log[1].revision(), 1);
        assert_eq!(
            log[1].outcome(),
            GoalTransitionOutcome::RejectedStaleRevision
        );

        assert_eq!(log[2].origin(), &GoalOrigin::agent("rogue").unwrap());
        assert_eq!(log[2].revision(), 2);
        assert_eq!(
            log[2].outcome(),
            GoalTransitionOutcome::RejectedUnauthorized
        );

        assert_eq!(log[3].origin(), &GoalOrigin::human("op-b").unwrap());
        assert_eq!(log[3].revision(), 2);
        assert_eq!(log[3].outcome(), GoalTransitionOutcome::Applied);
        assert_eq!(log[3].authority_issuer(), Some("supervisor"));

        // Replay reconstruye exactamente el estado completo de GoalManager.
        let replayed = GoalManager::replay(log).unwrap();
        assert_eq!(replayed, manager);
    }

    #[test]
    fn validates_identifiers_weights_deactivation_and_expiration_pruning() {
        assert!(matches!(
            GoalId::new("   "),
            Err(GoalError::EmptyIdentifier { field: "goal_id" })
        ));
        assert!(matches!(
            GoalAuthority::for_any_goal(""),
            Err(GoalError::EmptyIdentifier {
                field: "authority.issuer"
            })
        ));
        let gid = GoalId::new("g1").unwrap();
        let origin = GoalOrigin::system("s1").unwrap();
        assert!(matches!(
            GoalChangeProposal::upsert(gid.clone(), origin.clone(), 1, 1, f64::NAN),
            Err(GoalError::NonFiniteWeight { .. })
        ));
        assert!(matches!(
            GoalChangeProposal::upsert(gid.clone(), origin.clone(), 1, 1, -0.5),
            Err(GoalError::NegativeWeight { .. })
        ));

        let clock = VirtualMonotonicClock::new();
        let mut manager = GoalManager::new();
        let auth = GoalAuthority::for_any_goal("kernel").unwrap();
        let exp = clock.now().checked_add(Duration::from_secs(3)).unwrap();
        let p = GoalChangeProposal::upsert(gid.clone(), origin.clone(), 1, 5, 2.0)
            .unwrap()
            .with_expires_at(exp)
            .with_authority(auth.clone());
        assert_eq!(
            manager.apply_proposal(p, clock.now()),
            GoalTransitionOutcome::Applied
        );
        assert_eq!(manager.active_goals_at(clock.now()).len(), 1);

        clock.advance(Duration::from_secs(4));
        assert_eq!(manager.active_goals_at(clock.now()).len(), 0);
        assert_eq!(manager.prune_expired(clock.now()), 1);
        assert!(manager.get(&gid).is_none());

        // Deactivate también respeta revisión monotónica: revisión 1 sigue siendo obsoleta.
        let stale_deact =
            GoalChangeProposal::deactivate(gid.clone(), origin, 1).with_authority(auth);
        assert_eq!(
            manager.apply_proposal(stale_deact, clock.now()),
            GoalTransitionOutcome::RejectedStaleRevision
        );
    }
}
