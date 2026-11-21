//! Reglas causales como hipótesis contextuales con evidencia (`std`-only, GOG-163).
//!
//! Una [`CausalRule`] es **solo datos tipados**: condiciones ([`Condition`]), ámbito temporal
//! ([`TemporalScope`]), resultados probabilísticos ([`Outcome`]), procedencia
//! ([`EvidenceRef`]), versión y estado epistémico ([`EpistemicStatus`]). No contiene
//! expresiones evaluables, cadenas interpretadas ni closures: la evaluación es una función
//! pura y cerrada ([`CausalRule::evaluate`]) sobre un [`RuleContext`] inmutable.
//!
//! # Estados distintos
//! [`RuleEvaluation`] separa explícitamente: aplicabilidad ([`RuleEvaluation::Applies`]),
//! contradicción por observación conocida ([`RuleEvaluation::Contradicted`]), ausencia de
//! observación ([`RuleEvaluation::NotObserved`]), fuera de ámbito temporal, regla revisada
//! y permiso de acción ([`RuleEvaluation::ActionNotPermitted`]). Que una regla "aplique" nunca
//! concede permiso: el permiso se resuelve con [`ActionPermission`], derivado de una
//! [`GuardDecision`] ya emitida o de [`permission_from_guard`], que solo toma `&ActionGuard`.
//!
//! # Memoria semántica
//! No hay un almacén rival: [`record_rule_hypothesis`] y [`add_rule_evidence`] registran la
//! regla como proposición `causal_rule` en el [`SemanticKnowledgeStore`] existente, que
//! conserva historial y evidencia append-only. Este módulo no demuestra que una regla sea
//! cierta; solo representa y valida hipótesis.
//!
//! # Codificación estable
//! [`CausalRule::canonical_bytes`] produce una codificación little-endian con etiquetas,
//! prefijada por [`CAUSAL_RULE_ENCODING_VERSION`]; [`CausalRule::stable_hash`] es FNV-1a de
//! 64 bits sobre esos bytes. Sin reloj de pared ni aleatoriedad.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::action::Action;
use crate::belief::{BinaryBelief, LikelihoodModel};
use crate::guard::{ActionGuard, ActionProposal, DomainPrecondition, GuardContext, GuardDecision};
use crate::observability::EntityId;
use crate::provenance::EvidenceRef;
use crate::semantic_memory::{
    EpistemicStatus, EvidenceInput, PredicateId, PropositionKey, PropositionValue, RevisionStatus,
    SemanticKnowledgeStore, SemanticMemoryError,
};

/// Versión de la codificación canónica de reglas.
pub const CAUSAL_RULE_ENCODING_VERSION: u32 = 1;
/// Longitud máxima (bytes) de identificadores y nombres de variable/valor simbólico.
pub const MAX_IDENTIFIER_LEN: usize = 64;
/// Máximo de condiciones por regla.
pub const MAX_CONDITIONS: usize = 16;
/// Máximo de resultados probabilísticos por regla.
pub const MAX_OUTCOMES: usize = 16;
/// Máximo de efectos por resultado.
pub const MAX_EFFECTS: usize = 16;
/// Máximo de referencias de procedencia por regla.
pub const MAX_PROVENANCE: usize = 16;
/// Máximo de valores en una condición [`Condition::OneOf`].
pub const MAX_ONE_OF: usize = 16;
/// Máximo de reglas en un [`CausalRuleSet`].
pub const MAX_RULES: usize = 256;
/// Tolerancia absoluta para que la distribución de resultados sume 1.
pub const PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;
/// Predicado usado en la memoria semántica para reglas causales.
pub const CAUSAL_RULE_PREDICATE: &str = "causal_rule";

/// Errores de validación (fail-closed) de reglas causales.
#[derive(Debug, Clone, PartialEq)]
pub enum CausalRuleError {
    /// Identificador/nombre vacío, demasiado largo o con caracteres no permitidos.
    InvalidIdentifier {
        /// Campo inválido.
        field: &'static str,
    },
    /// Probabilidad no finita o fuera de `[0, 1]`.
    InvalidProbability {
        /// Valor recibido.
        value: f64,
    },
    /// La distribución de resultados no suma 1 dentro de la tolerancia (o está vacía).
    OutcomeDistributionNotNormalized {
        /// Suma observada.
        sum: f64,
    },
    /// Colección fuera de límites.
    TooMany {
        /// Colección afectada.
        field: &'static str,
        /// Límite máximo.
        max: usize,
    },
    /// Rango entero o ámbito temporal con `min > max`.
    InvalidRange,
    /// Versión cero (las versiones empiezan en 1).
    InvalidVersion,
    /// Identificador de regla duplicado en un conjunto.
    DuplicateRuleId,
    /// Etiqueta de resultado duplicada dentro de una regla.
    DuplicateOutcomeLabel,
    /// Error propagado de la memoria semántica.
    Memory(SemanticMemoryError),
}

impl fmt::Display for CausalRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidIdentifier { field } => write!(f, "invalid identifier '{field}'"),
            Self::InvalidProbability { value } => {
                write!(f, "probability {value} must be finite within [0, 1]")
            }
            Self::OutcomeDistributionNotNormalized { sum } => {
                write!(f, "outcome probabilities sum to {sum}, expected 1")
            }
            Self::TooMany { field, max } => write!(f, "too many '{field}' (max {max})"),
            Self::InvalidRange => f.write_str("invalid range: min > max"),
            Self::InvalidVersion => f.write_str("rule version must be >= 1"),
            Self::DuplicateRuleId => f.write_str("duplicate rule id"),
            Self::DuplicateOutcomeLabel => f.write_str("duplicate outcome label"),
            Self::Memory(e) => write!(f, "semantic memory error: {e}"),
        }
    }
}

impl Error for CausalRuleError {}

impl From<SemanticMemoryError> for CausalRuleError {
    fn from(e: SemanticMemoryError) -> Self {
        Self::Memory(e)
    }
}

fn validate_ident(field: &'static str, s: String) -> Result<String, CausalRuleError> {
    let ok = !s.is_empty()
        && s.len() <= MAX_IDENTIFIER_LEN
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'));
    if ok {
        Ok(s)
    } else {
        Err(CausalRuleError::InvalidIdentifier { field })
    }
}

macro_rules! ident_type {
    ($(#[$m:meta])* $name:ident, $field:literal) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);
        impl $name {
            /// Construye el identificador validando longitud y alfabeto (`[A-Za-z0-9_.:-]`).
            pub fn new(value: impl Into<String>) -> Result<Self, CausalRuleError> {
                validate_ident($field, value.into()).map(Self)
            }
            /// Devuelve el texto.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

ident_type!(
    /// Identificador único de regla.
    RuleId,
    "rule_id"
);
ident_type!(
    /// Nombre de variable observable (por ejemplo `"cell_north"`).
    VariableName,
    "variable"
);
ident_type!(
    /// Etiqueta de un resultado probabilístico.
    OutcomeLabel,
    "outcome_label"
);

/// Valor discreto comparable en condiciones y efectos.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RuleValue {
    /// Booleano.
    Bool(bool),
    /// Entero.
    Int(i64),
    /// Símbolo validado.
    Symbol(String),
}

impl RuleValue {
    /// Construye un símbolo validado.
    pub fn symbol(value: impl Into<String>) -> Result<Self, CausalRuleError> {
        validate_ident("symbol", value.into()).map(Self::Symbol)
    }
}

/// Condición tipada sobre una variable observable. Solo datos; sin expresiones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Condition {
    /// La variable es igual al valor.
    Equals(VariableName, RuleValue),
    /// La variable es distinta al valor.
    NotEquals(VariableName, RuleValue),
    /// La variable es un entero en `[min, max]`.
    IntInRange(VariableName, i64, i64),
    /// La variable es uno de los valores.
    OneOf(VariableName, Vec<RuleValue>),
}

impl Condition {
    /// Construye `IntInRange` validando `min <= max`.
    pub fn int_in_range(var: VariableName, min: i64, max: i64) -> Result<Self, CausalRuleError> {
        if min > max {
            return Err(CausalRuleError::InvalidRange);
        }
        Ok(Self::IntInRange(var, min, max))
    }

    /// Construye `OneOf` validando que sea no vacío y acotado.
    pub fn one_of(var: VariableName, values: Vec<RuleValue>) -> Result<Self, CausalRuleError> {
        if values.is_empty() || values.len() > MAX_ONE_OF {
            return Err(CausalRuleError::TooMany {
                field: "one_of",
                max: MAX_ONE_OF,
            });
        }
        Ok(Self::OneOf(var, values))
    }

    /// Variable referida por la condición.
    pub const fn variable(&self) -> &VariableName {
        match self {
            Self::Equals(v, _)
            | Self::NotEquals(v, _)
            | Self::IntInRange(v, _, _)
            | Self::OneOf(v, _) => v,
        }
    }

    fn holds(&self, observed: &RuleValue) -> bool {
        match self {
            Self::Equals(_, v) => observed == v,
            Self::NotEquals(_, v) => observed != v,
            Self::IntInRange(_, lo, hi) => {
                matches!(observed, RuleValue::Int(i) if i >= lo && i <= hi)
            }
            Self::OneOf(_, vs) => vs.contains(observed),
        }
    }
}

/// Ámbito temporal en ticks discretos, inclusivo; `until = None` es abierto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemporalScope {
    from_tick: u64,
    until_tick: Option<u64>,
}

impl TemporalScope {
    /// Siempre vigente.
    pub const ALWAYS: Self = Self {
        from_tick: 0,
        until_tick: None,
    };

    /// Construye el ámbito `[from, until]` validando orden.
    pub fn new(from_tick: u64, until_tick: Option<u64>) -> Result<Self, CausalRuleError> {
        if until_tick.is_some_and(|u| u < from_tick) {
            return Err(CausalRuleError::InvalidRange);
        }
        Ok(Self {
            from_tick,
            until_tick,
        })
    }

    /// Indica si `tick` está dentro del ámbito.
    pub fn contains(self, tick: u64) -> bool {
        tick >= self.from_tick && self.until_tick.is_none_or(|u| tick <= u)
    }
}

/// Efecto declarativo de un resultado: asigna un valor a una variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect {
    /// Variable afectada.
    pub variable: VariableName,
    /// Valor resultante.
    pub value: RuleValue,
}

/// Resultado probabilístico de aplicar la regla.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    label: OutcomeLabel,
    probability: f64,
    effects: Vec<Effect>,
}

impl Outcome {
    /// Construye un resultado validando probabilidad finita en `[0, 1]` y límite de efectos.
    pub fn new(
        label: OutcomeLabel,
        probability: f64,
        effects: Vec<Effect>,
    ) -> Result<Self, CausalRuleError> {
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Err(CausalRuleError::InvalidProbability { value: probability });
        }
        if effects.len() > MAX_EFFECTS {
            return Err(CausalRuleError::TooMany {
                field: "effects",
                max: MAX_EFFECTS,
            });
        }
        Ok(Self {
            label,
            probability,
            effects,
        })
    }

    /// Etiqueta.
    pub const fn label(&self) -> &OutcomeLabel {
        &self.label
    }
    /// Probabilidad.
    pub const fn probability(&self) -> f64 {
        self.probability
    }
    /// Efectos declarados.
    pub fn effects(&self) -> &[Effect] {
        &self.effects
    }
}

/// Regla causal validada: hipótesis contextual con evidencia. Inmutable tras construirse.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalRule {
    id: RuleId,
    version: u32,
    action: Option<usize>,
    conditions: Vec<Condition>,
    scope: TemporalScope,
    outcomes: Vec<Outcome>,
    provenance: Vec<EvidenceRef>,
    status: EpistemicStatus,
}

impl CausalRule {
    /// Construye y valida una regla (fail-closed). `action` es el índice discreto de la acción
    /// de intervención cuya ejecución requiere permiso externo; `None` = transición pasiva.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: RuleId,
        version: u32,
        action: Option<usize>,
        conditions: Vec<Condition>,
        scope: TemporalScope,
        outcomes: Vec<Outcome>,
        provenance: Vec<EvidenceRef>,
        status: EpistemicStatus,
    ) -> Result<Self, CausalRuleError> {
        if version == 0 {
            return Err(CausalRuleError::InvalidVersion);
        }
        if conditions.len() > MAX_CONDITIONS {
            return Err(CausalRuleError::TooMany {
                field: "conditions",
                max: MAX_CONDITIONS,
            });
        }
        for c in &conditions {
            match c {
                Condition::IntInRange(_, lo, hi) if lo > hi => {
                    return Err(CausalRuleError::InvalidRange);
                }
                Condition::OneOf(_, vs) if vs.is_empty() || vs.len() > MAX_ONE_OF => {
                    return Err(CausalRuleError::TooMany {
                        field: "one_of",
                        max: MAX_ONE_OF,
                    });
                }
                _ => {}
            }
        }
        if outcomes.is_empty() || outcomes.len() > MAX_OUTCOMES {
            return Err(CausalRuleError::TooMany {
                field: "outcomes",
                max: MAX_OUTCOMES,
            });
        }
        if provenance.len() > MAX_PROVENANCE {
            return Err(CausalRuleError::TooMany {
                field: "provenance",
                max: MAX_PROVENANCE,
            });
        }
        let labels: BTreeSet<&OutcomeLabel> = outcomes.iter().map(Outcome::label).collect();
        if labels.len() != outcomes.len() {
            return Err(CausalRuleError::DuplicateOutcomeLabel);
        }
        let sum: f64 = outcomes.iter().map(Outcome::probability).sum();
        if !sum.is_finite() || (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
            return Err(CausalRuleError::OutcomeDistributionNotNormalized { sum });
        }
        Ok(Self {
            id,
            version,
            action,
            conditions,
            scope,
            outcomes,
            provenance,
            status,
        })
    }

    /// Identificador.
    pub const fn id(&self) -> &RuleId {
        &self.id
    }
    /// Versión (>= 1).
    pub const fn version(&self) -> u32 {
        self.version
    }
    /// Índice de acción de intervención, si la hay.
    pub const fn action(&self) -> Option<usize> {
        self.action
    }
    /// Condiciones.
    pub fn conditions(&self) -> &[Condition] {
        &self.conditions
    }
    /// Ámbito temporal.
    pub const fn scope(&self) -> TemporalScope {
        self.scope
    }
    /// Resultados probabilísticos.
    pub fn outcomes(&self) -> &[Outcome] {
        &self.outcomes
    }
    /// Referencias de procedencia.
    pub fn provenance(&self) -> &[EvidenceRef] {
        &self.provenance
    }
    /// Estado epistémico declarado.
    pub const fn status(&self) -> EpistemicStatus {
        self.status
    }

    /// Evalúa la regla de forma pura. Orden de precedencia (fail-closed): permiso, regla
    /// revisada, ámbito temporal, contradicción conocida, ausencia de observación, aplica.
    pub fn evaluate(&self, ctx: &RuleContext, permission: ActionPermission) -> RuleEvaluation {
        if self.action.is_some() && permission != ActionPermission::Permitted {
            return RuleEvaluation::ActionNotPermitted;
        }
        if self.status.is_revised() {
            return RuleEvaluation::Revised;
        }
        if !self.scope.contains(ctx.tick) {
            return RuleEvaluation::OutOfScope;
        }
        let mut contradicted = Vec::new();
        let mut missing = Vec::new();
        for (i, c) in self.conditions.iter().enumerate() {
            match ctx.observations.get(c.variable()) {
                Some(Observed::Known(v)) => {
                    if !c.holds(v) {
                        contradicted.push(i);
                    }
                }
                Some(Observed::Unknown) | None => missing.push(i),
            }
        }
        if !contradicted.is_empty() {
            return RuleEvaluation::Contradicted {
                failed_conditions: contradicted,
            };
        }
        if !missing.is_empty() {
            return RuleEvaluation::NotObserved {
                missing_conditions: missing,
            };
        }
        RuleEvaluation::Applies {
            outcomes: self
                .outcomes
                .iter()
                .map(|o| (o.label.clone(), o.probability))
                .collect(),
        }
    }

    /// Codificación canónica estable (prefijada con [`CAUSAL_RULE_ENCODING_VERSION`]).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut w = Vec::new();
        w.extend_from_slice(&CAUSAL_RULE_ENCODING_VERSION.to_le_bytes());
        put_str(&mut w, self.id.as_str());
        w.extend_from_slice(&self.version.to_le_bytes());
        match self.action {
            None => w.push(0),
            Some(a) => {
                w.push(1);
                w.extend_from_slice(&(a as u64).to_le_bytes());
            }
        }
        w.push(status_tag(self.status));
        w.extend_from_slice(&self.scope.from_tick.to_le_bytes());
        match self.scope.until_tick {
            None => w.push(0),
            Some(u) => {
                w.push(1);
                w.extend_from_slice(&u.to_le_bytes());
            }
        }
        w.extend_from_slice(&(self.conditions.len() as u32).to_le_bytes());
        for c in &self.conditions {
            match c {
                Condition::Equals(v, x) => {
                    w.push(1);
                    put_str(&mut w, v.as_str());
                    put_value(&mut w, x);
                }
                Condition::NotEquals(v, x) => {
                    w.push(2);
                    put_str(&mut w, v.as_str());
                    put_value(&mut w, x);
                }
                Condition::IntInRange(v, lo, hi) => {
                    w.push(3);
                    put_str(&mut w, v.as_str());
                    w.extend_from_slice(&lo.to_le_bytes());
                    w.extend_from_slice(&hi.to_le_bytes());
                }
                Condition::OneOf(v, xs) => {
                    w.push(4);
                    put_str(&mut w, v.as_str());
                    w.extend_from_slice(&(xs.len() as u32).to_le_bytes());
                    for x in xs {
                        put_value(&mut w, x);
                    }
                }
            }
        }
        w.extend_from_slice(&(self.outcomes.len() as u32).to_le_bytes());
        for o in &self.outcomes {
            put_str(&mut w, o.label.as_str());
            let p = if o.probability == 0.0 {
                0.0
            } else {
                o.probability
            };
            w.extend_from_slice(&p.to_bits().to_le_bytes());
            w.extend_from_slice(&(o.effects.len() as u32).to_le_bytes());
            for e in &o.effects {
                put_str(&mut w, e.variable.as_str());
                put_value(&mut w, &e.value);
            }
        }
        w.extend_from_slice(&(self.provenance.len() as u32).to_le_bytes());
        for r in &self.provenance {
            w.extend_from_slice(&r.as_u64().to_le_bytes());
        }
        w
    }

    /// Hash FNV-1a de 64 bits de [`Self::canonical_bytes`].
    pub fn stable_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.canonical_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}

fn put_str(w: &mut Vec<u8>, s: &str) {
    w.extend_from_slice(&(s.len() as u32).to_le_bytes());
    w.extend_from_slice(s.as_bytes());
}

fn put_value(w: &mut Vec<u8>, v: &RuleValue) {
    match v {
        RuleValue::Bool(b) => {
            w.push(1);
            w.push(u8::from(*b));
        }
        RuleValue::Int(i) => {
            w.push(2);
            w.extend_from_slice(&i.to_le_bytes());
        }
        RuleValue::Symbol(s) => {
            w.push(3);
            put_str(w, s);
        }
    }
}

const fn status_tag(s: EpistemicStatus) -> u8 {
    match s {
        EpistemicStatus::ProposedFact => 1,
        EpistemicStatus::Hypothesis => 2,
        EpistemicStatus::CorroboratedObservation => 3,
        EpistemicStatus::Contested => 4,
        EpistemicStatus::Revised => 5,
    }
}

/// Observación de una variable: conocida o explícitamente desconocida.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Observed {
    /// Valor observado.
    Known(RuleValue),
    /// Sin observación (equivale a variable ausente).
    Unknown,
}

/// Contexto inmutable de evaluación: tick discreto y observaciones.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RuleContext {
    tick: u64,
    observations: BTreeMap<VariableName, Observed>,
}

impl RuleContext {
    /// Crea un contexto vacío en `tick`.
    pub fn new(tick: u64) -> Self {
        Self {
            tick,
            observations: BTreeMap::new(),
        }
    }

    /// Añade/reemplaza una observación.
    #[must_use]
    pub fn with(mut self, var: VariableName, obs: Observed) -> Self {
        self.observations.insert(var, obs);
        self
    }
}

/// Permiso de acción resuelto externamente (nunca concedido por la regla).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionPermission {
    /// La guardia autorizó exactamente la acción de la regla.
    Permitted,
    /// No autorizada, o no consultada: se trata como denegada.
    NotPermitted,
}

impl ActionPermission {
    /// Deriva el permiso de una decisión ya emitida: solo `Allow` de la misma acción discreta.
    pub fn from_decision(decision: &GuardDecision, rule_action: usize) -> Self {
        match decision {
            GuardDecision::Allow(Action::Discrete(d)) if d.index() == rule_action => {
                Self::Permitted
            }
            _ => Self::NotPermitted,
        }
    }
}

/// Consulta a la guardia por referencia compartida (`&ActionGuard`): no la muta ni la
/// reconfigura. Devuelve [`ActionPermission::Permitted`] solo si `Allow` de la acción de la regla.
pub fn permission_from_guard<P: DomainPrecondition>(
    guard: &ActionGuard<P>,
    proposal: &ActionProposal,
    ctx: &GuardContext<'_>,
    rule_action: usize,
) -> ActionPermission {
    ActionPermission::from_decision(&guard.evaluate(proposal, ctx), rule_action)
}

/// Resultado de evaluar una regla: estados mutuamente distintos.
#[derive(Debug, Clone, PartialEq)]
pub enum RuleEvaluation {
    /// Todas las condiciones se cumplen; distribución de resultados (etiqueta, probabilidad).
    /// No implica permiso de actuar si la regla tiene acción (ese caso ya fue filtrado).
    Applies {
        /// Resultados probabilísticos.
        outcomes: Vec<(OutcomeLabel, f64)>,
    },
    /// Alguna condición se evaluó con valor conocido y es falsa.
    Contradicted {
        /// Índices de condiciones violadas.
        failed_conditions: Vec<usize>,
    },
    /// Sin contradicción conocida, pero faltan observaciones.
    NotObserved {
        /// Índices de condiciones sin observación.
        missing_conditions: Vec<usize>,
    },
    /// La regla requiere una acción que no está permitida.
    ActionNotPermitted,
    /// El tick actual está fuera del ámbito temporal.
    OutOfScope,
    /// La regla fue revisada/refutada en memoria epistémica y no se usa.
    Revised,
}

/// Conjunto de reglas sin identificadores duplicados y acotado.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CausalRuleSet {
    rules: BTreeMap<RuleId, CausalRule>,
}

impl CausalRuleSet {
    /// Conjunto vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Inserta una regla; rechaza id duplicado o exceso de reglas.
    pub fn insert(&mut self, rule: CausalRule) -> Result<(), CausalRuleError> {
        if self.rules.contains_key(rule.id()) {
            return Err(CausalRuleError::DuplicateRuleId);
        }
        if self.rules.len() >= MAX_RULES {
            return Err(CausalRuleError::TooMany {
                field: "rules",
                max: MAX_RULES,
            });
        }
        self.rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Número de reglas.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// `true` si no hay reglas.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Obtiene una regla.
    pub fn get(&self, id: &RuleId) -> Option<&CausalRule> {
        self.rules.get(id)
    }

    /// Hash estable del conjunto (orden por id).
    pub fn stable_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for r in self.rules.values() {
            for b in r.stable_hash().to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }
}

/// Clave de proposición de una regla (id, versión y hash estable) para un sujeto dado.
pub fn rule_proposition_key(
    subject: EntityId,
    rule: &CausalRule,
) -> Result<PropositionKey, CausalRuleError> {
    let object = PropositionValue::symbol(format!(
        "{}@v{}#{:016x}",
        rule.id().as_str(),
        rule.version(),
        rule.stable_hash()
    ))?;
    Ok(PropositionKey::new(
        subject,
        PredicateId::new(CAUSAL_RULE_PREDICATE)?,
        object,
    ))
}

/// Registra la regla como hipótesis (`Hypothesis`, prior 0.5) en el almacén semántico
/// existente, con su primera evidencia. Devuelve la clave de proposición.
pub fn record_rule_hypothesis(
    store: &mut SemanticKnowledgeStore,
    subject: EntityId,
    rule: &CausalRule,
    initial_evidence: EvidenceInput,
) -> Result<PropositionKey, CausalRuleError> {
    let key = rule_proposition_key(subject, rule)?;
    let prior = BinaryBelief::new(0.5).map_err(SemanticMemoryError::from)?;
    store.record_proposition(
        key.clone(),
        EpistemicStatus::Hypothesis,
        prior,
        initial_evidence,
    )?;
    Ok(key)
}

/// Añade evidencia (soporte o contradicción) a la hipótesis de la regla.
pub fn add_rule_evidence(
    store: &mut SemanticKnowledgeStore,
    subject: EntityId,
    rule: &CausalRule,
    evidence: EvidenceInput,
    model: &LikelihoodModel,
) -> Result<RevisionStatus, CausalRuleError> {
    let key = rule_proposition_key(subject, rule)?;
    Ok(store.add_evidence(&key, evidence, model)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionSpace, DiscreteAction};
    use crate::clock::MonotonicInstant;
    use crate::domain::Terminal;
    use crate::guard::SafeFallbackPolicy;
    use crate::provenance::{Confidence, ObservationKind, SensorId};
    use crate::semantic_memory::EvidencePolarity;
    use crate::units::Tick;

    fn vn(s: &str) -> VariableName {
        VariableName::new(s).unwrap()
    }
    fn sym(s: &str) -> RuleValue {
        RuleValue::symbol(s).unwrap()
    }
    fn outcome(l: &str, p: f64, var: &str, val: &str) -> Outcome {
        Outcome::new(
            OutcomeLabel::new(l).unwrap(),
            p,
            vec![Effect {
                variable: vn(var),
                value: sym(val),
            }],
        )
        .unwrap()
    }

    /// Fixture Grid a partir de datos planos: empujar una caja hacia el norte.
    fn push_rule(version: u32, action: Option<usize>) -> CausalRule {
        CausalRule::new(
            RuleId::new("grid.push_north").unwrap(),
            version,
            action,
            vec![
                Condition::Equals(vn("north_cell"), sym("box")),
                Condition::NotEquals(vn("beyond_north"), sym("wall")),
            ],
            TemporalScope::new(0, Some(100)).unwrap(),
            vec![
                outcome("moved", 0.9, "north_cell", "empty"),
                outcome("stuck", 0.1, "north_cell", "box"),
            ],
            vec![EvidenceRef::new(1)],
            EpistemicStatus::Hypothesis,
        )
        .unwrap()
    }

    /// Segunda regla: puerta con llave, dependiente de contador entero.
    fn door_rule() -> CausalRule {
        CausalRule::new(
            RuleId::new("grid.open_door").unwrap(),
            1,
            None,
            vec![
                Condition::Equals(vn("has_key"), RuleValue::Bool(true)),
                Condition::int_in_range(vn("steps_left"), 1, 50).unwrap(),
                Condition::one_of(vn("facing"), vec![sym("north"), sym("east")]).unwrap(),
            ],
            TemporalScope::ALWAYS,
            vec![outcome("opened", 1.0, "door", "open")],
            vec![],
            EpistemicStatus::ProposedFact,
        )
        .unwrap()
    }

    fn full_ctx(tick: u64) -> RuleContext {
        RuleContext::new(tick)
            .with(vn("north_cell"), Observed::Known(sym("box")))
            .with(vn("beyond_north"), Observed::Known(sym("floor")))
    }

    fn ev(r: u64, pol: EvidencePolarity) -> EvidenceInput {
        EvidenceInput::new(
            EvidenceRef::new(r),
            MonotonicInstant::from_nanos(u128::from(r)),
            SensorId::new(1),
            pol,
            ObservationKind::Measurement,
            Confidence::new(0.9).unwrap(),
        )
    }

    #[test]
    fn gog_163_two_rules_declared_validated_and_applied() {
        let mut set = CausalRuleSet::new();
        set.insert(push_rule(1, None)).unwrap();
        set.insert(door_rule()).unwrap();
        assert_eq!(set.len(), 2);

        let r = set.get(&RuleId::new("grid.push_north").unwrap()).unwrap();
        match r.evaluate(&full_ctx(5), ActionPermission::NotPermitted) {
            RuleEvaluation::Applies { outcomes } => {
                assert_eq!(outcomes.len(), 2);
                let s: f64 = outcomes.iter().map(|(_, p)| p).sum();
                assert!((s - 1.0).abs() < 1e-12);
            }
            other => panic!("unexpected {other:?}"),
        }
        let d = set.get(&RuleId::new("grid.open_door").unwrap()).unwrap();
        let ctx = RuleContext::new(0)
            .with(vn("has_key"), Observed::Known(RuleValue::Bool(true)))
            .with(vn("steps_left"), Observed::Known(RuleValue::Int(7)))
            .with(vn("facing"), Observed::Known(sym("east")));
        assert!(matches!(
            d.evaluate(&ctx, ActionPermission::NotPermitted),
            RuleEvaluation::Applies { .. }
        ));
    }

    #[test]
    fn gog_163_contradiction_absence_permission_scope_are_distinct() {
        let passive = push_rule(1, None);
        // Contradicción: valor conocido incompatible.
        let c = RuleContext::new(5)
            .with(vn("north_cell"), Observed::Known(sym("empty")))
            .with(vn("beyond_north"), Observed::Known(sym("floor")));
        assert_eq!(
            passive.evaluate(&c, ActionPermission::Permitted),
            RuleEvaluation::Contradicted {
                failed_conditions: vec![0]
            }
        );
        // Ausencia: variable ausente y variable Unknown.
        let n = RuleContext::new(5)
            .with(vn("north_cell"), Observed::Known(sym("box")))
            .with(vn("beyond_north"), Observed::Unknown);
        assert_eq!(
            passive.evaluate(&n, ActionPermission::Permitted),
            RuleEvaluation::NotObserved {
                missing_conditions: vec![1]
            }
        );
        let absent = RuleContext::new(5);
        assert_eq!(
            passive.evaluate(&absent, ActionPermission::Permitted),
            RuleEvaluation::NotObserved {
                missing_conditions: vec![0, 1]
            }
        );
        // Contradicción conocida prevalece sobre ausencia.
        let mixed = RuleContext::new(5).with(vn("north_cell"), Observed::Known(sym("wall")));
        assert!(matches!(
            passive.evaluate(&mixed, ActionPermission::Permitted),
            RuleEvaluation::Contradicted { .. }
        ));
        // Permiso: regla con acción y contexto que cumple todo.
        let active = push_rule(1, Some(2));
        assert_eq!(
            active.evaluate(&full_ctx(5), ActionPermission::NotPermitted),
            RuleEvaluation::ActionNotPermitted
        );
        assert!(matches!(
            active.evaluate(&full_ctx(5), ActionPermission::Permitted),
            RuleEvaluation::Applies { .. }
        ));
        // Ámbito temporal.
        assert_eq!(
            passive.evaluate(&full_ctx(101), ActionPermission::Permitted),
            RuleEvaluation::OutOfScope
        );
    }

    #[test]
    fn gog_163_validation_fails_closed() {
        let mk = |ps: &[f64]| {
            let outs = ps
                .iter()
                .enumerate()
                .map(|(i, p)| Outcome::new(OutcomeLabel::new(format!("o{i}")).unwrap(), *p, vec![]))
                .collect::<Result<Vec<_>, _>>()?;
            CausalRule::new(
                RuleId::new("r").unwrap(),
                1,
                None,
                vec![],
                TemporalScope::ALWAYS,
                outs,
                vec![],
                EpistemicStatus::Hypothesis,
            )
        };
        assert!(mk(&[0.5, 0.5]).is_ok());
        assert!(matches!(
            mk(&[f64::NAN]),
            Err(CausalRuleError::InvalidProbability { .. })
        ));
        assert!(matches!(
            mk(&[1.5]),
            Err(CausalRuleError::InvalidProbability { .. })
        ));
        assert!(matches!(
            mk(&[-0.1, 1.1]),
            Err(CausalRuleError::InvalidProbability { .. })
        ));
        assert!(matches!(
            mk(&[0.5, 0.4]),
            Err(CausalRuleError::OutcomeDistributionNotNormalized { .. })
        ));
        assert!(matches!(
            mk(&[]),
            Err(CausalRuleError::TooMany {
                field: "outcomes",
                ..
            })
        ));
        assert!(RuleId::new("").is_err());
        assert!(RuleId::new("x".repeat(MAX_IDENTIFIER_LEN + 1)).is_err());
        assert!(RuleId::new("bad id").is_err());
        assert!(RuleId::new("1 + 1").is_err());
        assert_eq!(
            Condition::int_in_range(vn("a"), 3, 1),
            Err(CausalRuleError::InvalidRange)
        );
        assert!(TemporalScope::new(5, Some(4)).is_err());

        let many = (0..=MAX_CONDITIONS)
            .map(|_| Condition::Equals(vn("a"), RuleValue::Int(1)))
            .collect();
        let r = CausalRule::new(
            RuleId::new("r").unwrap(),
            1,
            None,
            many,
            TemporalScope::ALWAYS,
            vec![outcome("a", 1.0, "x", "y")],
            vec![],
            EpistemicStatus::Hypothesis,
        );
        assert!(matches!(r, Err(CausalRuleError::TooMany { .. })));
        assert_eq!(
            CausalRule::new(
                RuleId::new("r").unwrap(),
                0,
                None,
                vec![],
                TemporalScope::ALWAYS,
                vec![outcome("a", 1.0, "x", "y")],
                vec![],
                EpistemicStatus::Hypothesis
            ),
            Err(CausalRuleError::InvalidVersion)
        );
        let dup_label = CausalRule::new(
            RuleId::new("r").unwrap(),
            1,
            None,
            vec![],
            TemporalScope::ALWAYS,
            vec![outcome("a", 0.5, "x", "y"), outcome("a", 0.5, "x", "z")],
            vec![],
            EpistemicStatus::Hypothesis,
        );
        assert_eq!(dup_label, Err(CausalRuleError::DuplicateOutcomeLabel));

        let mut set = CausalRuleSet::new();
        set.insert(push_rule(1, None)).unwrap();
        assert_eq!(
            set.insert(push_rule(2, None)),
            Err(CausalRuleError::DuplicateRuleId)
        );
    }

    #[test]
    fn gog_163_canonical_encoding_is_stable_and_versioned() {
        let a = push_rule(1, None);
        let b = push_rule(1, None);
        assert_eq!(a.canonical_bytes(), b.canonical_bytes());
        assert_eq!(a.stable_hash(), b.stable_hash());
        assert_eq!(
            &a.canonical_bytes()[..4],
            &CAUSAL_RULE_ENCODING_VERSION.to_le_bytes()
        );
        assert_ne!(a.stable_hash(), push_rule(2, None).stable_hash());
        assert_ne!(a.stable_hash(), push_rule(1, Some(0)).stable_hash());
        // Valor fijado: solo cambia si cambia la codificación (entonces subir la versión).
        assert_eq!(a.stable_hash(), 11_941_510_787_335_962_617_u64);
        let mut s1 = CausalRuleSet::new();
        s1.insert(a).unwrap();
        s1.insert(door_rule()).unwrap();
        let mut s2 = CausalRuleSet::new();
        s2.insert(door_rule()).unwrap();
        s2.insert(b).unwrap();
        assert_eq!(s1.stable_hash(), s2.stable_hash());
    }

    #[test]
    fn gog_163_store_keeps_history_and_evidence_for_rule() {
        let mut store = SemanticKnowledgeStore::new();
        let subject = EntityId::new(900);
        let rule = push_rule(1, None);
        let key = record_rule_hypothesis(
            &mut store,
            subject,
            &rule,
            ev(10, EvidencePolarity::Supporting),
        )
        .unwrap();
        assert_eq!(
            store.get_proposition(&key).unwrap().status(),
            EpistemicStatus::Hypothesis
        );
        // Duplicado de la misma regla/versión: no sobrescribe.
        assert!(matches!(
            record_rule_hypothesis(
                &mut store,
                subject,
                &rule,
                ev(11, EvidencePolarity::Supporting)
            ),
            Err(CausalRuleError::Memory(
                SemanticMemoryError::PropositionAlreadyExists
            ))
        ));
        let model = LikelihoodModel::from_sensitivity_specificity(0.9, 0.9).unwrap();
        add_rule_evidence(
            &mut store,
            subject,
            &rule,
            ev(12, EvidencePolarity::Contradicting),
            &model,
        )
        .unwrap();
        let rec = store.get_proposition(&key).unwrap();
        assert_eq!(rec.evidence_history().len(), 2);
        assert_eq!(rec.supporting_count(), 1);
        assert_eq!(rec.contradicting_count(), 1);
        assert!(rec.status().has_contradiction());
        // Otra versión es una proposición distinta: el historial de la anterior se conserva.
        let key2 = record_rule_hypothesis(
            &mut store,
            subject,
            &push_rule(2, None),
            ev(13, EvidencePolarity::Supporting),
        )
        .unwrap();
        assert_ne!(key, key2);
        assert_eq!(store.proposition_count(), 2);
        assert_eq!(
            store
                .get_proposition(&key)
                .unwrap()
                .evidence_history()
                .len(),
            2
        );
    }

    #[test]
    fn gog_163_permission_from_guard_takes_shared_ref_and_never_grants_by_default() {
        let guard = ActionGuard::strict();
        let space = ActionSpace::Discrete { cardinality: 4 };
        let ctx = GuardContext::now(&space, Tick::new(0), Terminal::Ongoing);
        let _ = (&guard, &ctx, SafeFallbackPolicy::RejectOnly);
        let allow = GuardDecision::Allow(Action::Discrete(DiscreteAction::new(2, 4).unwrap()));
        assert_eq!(
            ActionPermission::from_decision(&allow, 2),
            ActionPermission::Permitted
        );
        assert_eq!(
            ActionPermission::from_decision(&allow, 1),
            ActionPermission::NotPermitted
        );
        assert_eq!(
            ActionPermission::from_decision(&GuardDecision::NoOp, 2),
            ActionPermission::NotPermitted
        );
        let active = push_rule(1, Some(2));
        let p = ActionPermission::from_decision(&GuardDecision::NoOp, 2);
        assert_eq!(
            active.evaluate(&full_ctx(1), p),
            RuleEvaluation::ActionNotPermitted
        );
    }

    #[test]
    fn gog_163_revised_rule_is_not_used() {
        let r = CausalRule::new(
            RuleId::new("r").unwrap(),
            1,
            None,
            vec![],
            TemporalScope::ALWAYS,
            vec![outcome("a", 1.0, "x", "y")],
            vec![],
            EpistemicStatus::Revised,
        )
        .unwrap();
        assert_eq!(
            r.evaluate(&RuleContext::new(0), ActionPermission::Permitted),
            RuleEvaluation::Revised
        );
    }
}
