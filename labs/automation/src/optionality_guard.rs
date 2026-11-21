//! Guardia por opcionalidad (GOG-193, DEM-040): veta acciones que dejan menos de `N` caminos de
//! recuperación.
//!
//! El principio entrópico actúa aquí **solo como guardia** (una [`DomainPrecondition`] que puede
//! rechazar), nunca como motor que elija acciones.
//!
//! # Medidas
//! - **Viabilidad**: número de secuencias de acciones de longitud exactamente `horizon` que, desde
//!   el estado sucesor, terminan en un estado viable de un [`TransitionGraph`] declarado
//!   (recuento exacto por programación dinámica, mismo espíritu que el oráculo de futuros viables
//!   de GOG-154, sin dependencia de `labs/grid`).
//! - **Bits de control**: `I(A0; Z)` sobre la enumeración exhaustiva de esas secuencias, usando
//!   [`gogamen_core::control_bits_from_pairs`] (GOG-180). Se informa siempre; solo veta si la
//!   política declara un suelo opcional.
//!
//! # Umbral declarado
//! [`OptionalityPolicy`] es inmutable tras construirla (campos privados, sin setters). Un valor
//! "aprendido" solo puede **endurecer** el umbral: `effective = max(declared, learned)`.
//!
//! # Límites
//! El grafo es un modelo declarado a mano; el recuento depende de `horizon`, así que una
//! recuperación más larga que el horizonte cuenta como 0 caminos (falso rechazo posible).

use std::cell::Cell;
use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, BinningVersion, ControlBitsConfig, DomainPrecondition, GuardContext,
    control_bits_from_pairs,
};

/// Error al construir un [`TransitionGraph`] o una [`OptionalityPolicy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    /// El grafo no tiene estados o acciones.
    EmptyGraph,
    /// Una fila de transiciones no tiene `n_actions` entradas o apunta fuera de rango.
    MalformedTransitions,
    /// `viable` no tiene un valor por estado.
    MalformedViableSet,
    /// `min_recovery_paths` o `horizon` igual a cero.
    ZeroParameter,
    /// El suelo de bits de control no es finito.
    NonFiniteFloor,
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Self::EmptyGraph => "transition graph has no states or no actions",
            Self::MalformedTransitions => "transition table is malformed or out of range",
            Self::MalformedViableSet => "viable set must have one flag per state",
            Self::ZeroParameter => "min_recovery_paths and horizon must be >= 1",
            Self::NonFiniteFloor => "control bits floor must be finite",
        };
        f.write_str(msg)
    }
}

impl Error for PolicyError {}

/// Grafo de transiciones determinista declarado (estado, acción) -> estado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionGraph {
    next: Vec<Vec<usize>>,
    viable: Vec<bool>,
}

impl TransitionGraph {
    /// Construye el grafo validando rangos. `next[s][a]` es el sucesor de `s` bajo la acción `a`.
    pub fn new(next: Vec<Vec<usize>>, viable: Vec<bool>) -> Result<Self, PolicyError> {
        let n = next.len();
        if n == 0 || next[0].is_empty() {
            return Err(PolicyError::EmptyGraph);
        }
        let m = next[0].len();
        if next
            .iter()
            .any(|row| row.len() != m || row.iter().any(|&t| t >= n))
        {
            return Err(PolicyError::MalformedTransitions);
        }
        if viable.len() != n {
            return Err(PolicyError::MalformedViableSet);
        }
        Ok(Self { next, viable })
    }

    /// Número de estados.
    pub fn state_count(&self) -> usize {
        self.next.len()
    }

    /// Número de acciones.
    pub fn action_count(&self) -> usize {
        self.next[0].len()
    }

    /// Sucesor de `state` bajo `action`, o `None` fuera de rango.
    pub fn step(&self, state: usize, action: usize) -> Option<usize> {
        self.next.get(state)?.get(action).copied()
    }

    /// `true` si `state` es viable.
    pub fn is_viable(&self, state: usize) -> bool {
        self.viable.get(state).copied().unwrap_or(false)
    }

    /// Recuento exacto de secuencias de longitud `horizon` desde `state` que acaban en viable.
    pub fn recovery_paths(&self, state: usize, horizon: u32) -> u64 {
        let n = self.state_count();
        let mut counts: Vec<u64> = (0..n).map(|s| u64::from(self.viable[s])).collect();
        for _ in 0..horizon {
            counts = (0..n)
                .map(|s| {
                    self.next[s]
                        .iter()
                        .fold(0u64, |acc, &t| acc.saturating_add(counts[t]))
                })
                .collect();
        }
        counts.get(state).copied().unwrap_or(0)
    }

    /// Alcanzabilidad sin límite de horizonte (verdad de referencia para las pruebas).
    pub fn can_ever_recover(&self, state: usize) -> bool {
        let mut seen = vec![false; self.state_count()];
        let mut stack = vec![state];
        while let Some(s) = stack.pop() {
            if s >= seen.len() || seen[s] {
                continue;
            }
            seen[s] = true;
            if self.viable[s] {
                return true;
            }
            stack.extend(self.next[s].iter().copied());
        }
        false
    }

    /// Bits de control corregidos `I(A0; Z)` desde `state` sobre todas las secuencias de `horizon`.
    ///
    /// Devuelve `(plug-in, corregido)`; `(0.0, 0.0)` si la estimación falla.
    pub fn control_bits(&self, state: usize, horizon: u32) -> (f64, f64) {
        let mut pairs = Vec::new();
        let m = self.action_count();
        for first in 0..m {
            let mut frontier = vec![self.next[state][first]];
            for _ in 1..horizon {
                frontier = frontier
                    .iter()
                    .flat_map(|&s| self.next[s].iter().copied())
                    .collect();
            }
            pairs.extend(frontier.into_iter().map(|z| (first, z)));
        }
        let cfg = ControlBitsConfig {
            resamples: 50,
            confidence: 0.95,
            seed: 193,
        };
        control_bits_from_pairs(&pairs, "service_recovery_graph", BinningVersion::V1_0, &cfg)
            .map(|e| (e.control_plugin, e.control_corrected))
            .unwrap_or((0.0, 0.0))
    }
}

/// Política declarada: umbral mínimo de caminos de recuperación. Inmutable tras construirse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionalityPolicy {
    min_recovery_paths: u64,
    horizon: u32,
    min_control_bits: Option<f64>,
}

impl OptionalityPolicy {
    /// Declara el umbral `min_recovery_paths` (>= 1) y el `horizon` (>= 1).
    pub fn new(min_recovery_paths: u64, horizon: u32) -> Result<Self, PolicyError> {
        if min_recovery_paths == 0 || horizon == 0 {
            return Err(PolicyError::ZeroParameter);
        }
        Ok(Self {
            min_recovery_paths,
            horizon,
            min_control_bits: None,
        })
    }

    /// Declara además un suelo de bits de control corregidos (constructor consumidor).
    pub fn with_min_control_bits(mut self, floor: f64) -> Result<Self, PolicyError> {
        if !floor.is_finite() {
            return Err(PolicyError::NonFiniteFloor);
        }
        self.min_control_bits = Some(floor);
        Ok(self)
    }

    /// Umbral declarado (el que fija la política, antes de cualquier ajuste aprendido).
    pub const fn declared_min_recovery_paths(&self) -> u64 {
        self.min_recovery_paths
    }

    /// Horizonte declarado.
    pub const fn horizon(&self) -> u32 {
        self.horizon
    }

    /// Suelo de bits de control declarado, si existe.
    pub const fn min_control_bits(&self) -> Option<f64> {
        self.min_control_bits
    }

    /// Umbral efectivo dado un valor aprendido: solo puede endurecer, nunca relajar.
    pub const fn effective_min_paths(&self, learned: Option<u64>) -> u64 {
        match learned {
            Some(l) if l > self.min_recovery_paths => l,
            _ => self.min_recovery_paths,
        }
    }
}

/// Motivo tipado de rechazo por opcionalidad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionalityRejection {
    /// La acción deja menos caminos de recuperación que el umbral efectivo.
    InsufficientRecoveryPaths {
        /// Estado sucesor evaluado.
        successor: usize,
        /// Caminos encontrados.
        found: u64,
        /// Umbral efectivo exigido.
        required: u64,
    },
    /// Los bits de control del sucesor están por debajo del suelo declarado.
    ControlBelowFloor {
        /// Estado sucesor evaluado.
        successor: usize,
        /// Bits de control corregidos.
        control_bits: f64,
        /// Suelo declarado.
        floor: f64,
    },
    /// La acción o el estado actual están fuera del grafo declarado (fail-closed).
    OutsideDeclaredGraph,
}

impl OptionalityRejection {
    /// Código estático estable (lo usa el `DomainPrecondition`, que exige `&'static str`).
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InsufficientRecoveryPaths { .. } => "optionality:insufficient_recovery_paths",
            Self::ControlBelowFloor { .. } => "optionality:control_below_floor",
            Self::OutsideDeclaredGraph => "optionality:outside_declared_graph",
        }
    }
}

impl fmt::Display for OptionalityRejection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientRecoveryPaths {
                successor,
                found,
                required,
            } => write!(
                f,
                "successor state {successor} has {found} recovery paths, {required} required"
            ),
            Self::ControlBelowFloor {
                successor,
                control_bits,
                floor,
            } => write!(
                f,
                "successor state {successor} has {control_bits:.3} control bits, floor {floor:.3}"
            ),
            Self::OutsideDeclaredGraph => f.write_str("action or state outside declared graph"),
        }
    }
}

impl Error for OptionalityRejection {}

/// Evidencia de una autorización.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionalityVerdict {
    /// Estado sucesor evaluado.
    pub successor: usize,
    /// Caminos de recuperación del sucesor.
    pub recovery_paths: u64,
    /// Bits de control corregidos del sucesor.
    pub control_bits: f64,
}

/// Precondición de dominio por opcionalidad sobre un [`TransitionGraph`] declarado.
///
/// El estado actual del servicio lo fija el llamador con [`OptionalityGuard::observe_state`]
/// (el [`GuardContext`] no transporta estado de dominio discreto).
#[derive(Debug)]
pub struct OptionalityGuard {
    graph: TransitionGraph,
    policy: OptionalityPolicy,
    learned_hint: Option<u64>,
    current_state: Cell<usize>,
    last_rejection: Cell<Option<OptionalityRejection>>,
}

impl OptionalityGuard {
    /// Crea la guardia en `initial_state`.
    pub fn new(graph: TransitionGraph, policy: OptionalityPolicy, initial_state: usize) -> Self {
        Self {
            graph,
            policy,
            learned_hint: None,
            current_state: Cell::new(initial_state),
            last_rejection: Cell::new(None),
        }
    }

    /// Registra un umbral sugerido por aprendizaje. Se acota: nunca baja del umbral declarado.
    pub fn with_learned_hint(mut self, hint: u64) -> Self {
        self.learned_hint = Some(hint);
        self
    }

    /// Umbral efectivo vigente (`max(declarado, aprendido)`).
    pub fn effective_min_paths(&self) -> u64 {
        self.policy.effective_min_paths(self.learned_hint)
    }

    /// Política declarada.
    pub const fn policy(&self) -> &OptionalityPolicy {
        &self.policy
    }

    /// Fija el estado actual observado del servicio.
    pub fn observe_state(&self, state: usize) {
        self.current_state.set(state);
    }

    /// Último rechazo tipado emitido vía [`DomainPrecondition::check`].
    pub fn last_rejection(&self) -> Option<OptionalityRejection> {
        self.last_rejection.get()
    }

    /// Evalúa `action_index` desde el estado actual con motivo tipado.
    pub fn evaluate(
        &self,
        action_index: usize,
    ) -> Result<OptionalityVerdict, OptionalityRejection> {
        let Some(successor) = self.graph.step(self.current_state.get(), action_index) else {
            return Err(OptionalityRejection::OutsideDeclaredGraph);
        };
        let required = self.effective_min_paths();
        let found = self.graph.recovery_paths(successor, self.policy.horizon());
        if found < required {
            return Err(OptionalityRejection::InsufficientRecoveryPaths {
                successor,
                found,
                required,
            });
        }
        let (_, control_bits) = self.graph.control_bits(successor, self.policy.horizon());
        if let Some(floor) = self.policy.min_control_bits()
            && control_bits < floor
        {
            return Err(OptionalityRejection::ControlBelowFloor {
                successor,
                control_bits,
                floor,
            });
        }
        Ok(OptionalityVerdict {
            successor,
            recovery_paths: found,
            control_bits,
        })
    }
}

impl DomainPrecondition for OptionalityGuard {
    fn check(&self, action: &Action, _ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        let result = match action {
            Action::Discrete(d) => self.evaluate(d.index()),
            _ => Err(OptionalityRejection::OutsideDeclaredGraph),
        };
        match result {
            Ok(_) => {
                self.last_rejection.set(None);
                Ok(())
            }
            Err(rej) => {
                self.last_rejection.set(Some(rej));
                Err(rej.code())
            }
        }
    }
}

impl DomainPrecondition for &OptionalityGuard {
    fn check(&self, action: &Action, ctx: &GuardContext<'_>) -> Result<(), &'static str> {
        (**self).check(action, ctx)
    }
}

/// Estados del grafo de recuperación del servicio local (índices).
pub mod svc {
    /// Servicio sano (viable).
    pub const HEALTHY: usize = 0;
    /// Degradado.
    pub const DEGRADED: usize = 1;
    /// Sobrecargado.
    pub const OVERLOADED: usize = 2;
    /// Worker caído: solo `restart_worker` vuelve.
    pub const WORKER_DOWN: usize = 3;
    /// Fallo profundo, 4 reinicios hasta sano (más largo que horizonte 3).
    pub const DEEP_1: usize = 4;
    /// Fallo profundo, 3 reinicios.
    pub const DEEP_2: usize = 5;
    /// Fallo profundo, 2 reinicios.
    pub const DEEP_3: usize = 6;
    /// Fallo profundo, 1 reinicio.
    pub const DEEP_4: usize = 7;
    /// Datos purgados (absorbente, no viable).
    pub const PURGED: usize = 8;
    /// Servicio retirado (absorbente, no viable).
    pub const DECOMMISSIONED: usize = 9;
    /// Número de estados.
    pub const STATES: usize = 10;
    /// Acciones: 0 no_op, 1 restart_worker, 2 drain_queue, 3 shed_load, 4 purge_state (irreversible),
    /// 5 decommission (irreversible).
    pub const ACTIONS: usize = 6;
    /// Índice de `purge_state`.
    pub const PURGE: usize = 4;
    /// Índice de `decommission`.
    pub const DECOMMISSION: usize = 5;
}

/// Grafo declarado del supervisor de servicio local (modelo de laboratorio, no medido).
pub fn service_recovery_graph() -> TransitionGraph {
    use svc::*;
    // Acción por defecto no listada: el estado no cambia.
    let mut next = vec![vec![0usize; ACTIONS]; STATES];
    for (s, row) in next.iter_mut().enumerate() {
        row.fill(s);
        row[PURGE] = PURGED;
        row[DECOMMISSION] = DECOMMISSIONED;
    }
    // Healthy: restart/drain mantienen sano, shed degrada.
    next[HEALTHY][3] = DEGRADED;
    // Degraded: restart y drain recuperan.
    next[DEGRADED][1] = HEALTHY;
    next[DEGRADED][2] = HEALTHY;
    // Overloaded: drain/shed -> degradado; restart -> worker caído.
    next[OVERLOADED][1] = WORKER_DOWN;
    next[OVERLOADED][2] = DEGRADED;
    next[OVERLOADED][3] = DEGRADED;
    // Worker caído: solo restart vuelve.
    next[WORKER_DOWN][1] = HEALTHY;
    // Fallo profundo: cadena de reinicios.
    next[DEEP_1][1] = DEEP_2;
    next[DEEP_2][1] = DEEP_3;
    next[DEEP_3][1] = DEEP_4;
    next[DEEP_4][1] = HEALTHY;
    // Absorbentes: ya fijados por `row.fill(s)` pero purge/decommission no deben salir.
    for s in [PURGED, DECOMMISSIONED] {
        next[s].fill(s);
    }
    let mut viable = vec![false; STATES];
    viable[HEALTHY] = true;
    TransitionGraph::new(next, viable).expect("declared service graph is well-formed")
}
