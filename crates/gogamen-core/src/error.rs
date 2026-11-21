//! Errores explícitos de dominio y validación en `gogamen-core`.

use std::error::Error;
use std::fmt;

/// Error de validación o invariante de dominio en contratos de Gogamen.
#[derive(Debug, Clone, PartialEq)]
pub enum DomainError {
    /// Un valor escalar o continuo no es finito (`NaN`, `+inf` o `-inf`).
    NonFiniteValue {
        /// Campo o contexto donde se detectó el valor no finito.
        field: &'static str,
        /// Valor inválido recibido.
        value: f64,
    },
    /// El paso temporal `dt` no es válido (debe ser finito y estrictamente `> 0.0` segundos).
    InvalidStepDt {
        /// Valor en segundos recibido.
        value: f64,
    },
    /// Una duración de simulación es negativa o no finita.
    InvalidDuration {
        /// Valor en segundos recibido.
        value: f64,
    },
    /// Desbordamiento al avanzar `Tick` o acumular `SimDuration`.
    TimeOverflow {
        /// Descripción de la operación que desbordó.
        context: &'static str,
    },
    /// El espacio de acciones está vacío (por ejemplo cardinalidad 0 o vector de límites vacío).
    EmptyActionSpace,
    /// Los límites `[min, max]` de una dimensión continua son inválidos (`min > max` o no finitos).
    InvalidActionBounds {
        /// Límite inferior recibido.
        min: f64,
        /// Límite superior recibido.
        max: f64,
    },
    /// Una acción discreta tiene un índice fuera del rango `[0, cardinality)`.
    DiscreteActionOutOfRange {
        /// Índice recibido.
        index: usize,
        /// Cardinalidad válida del espacio discreto.
        cardinality: usize,
    },
    /// Una acción continua tiene un componente fuera de sus límites `[min, max]`.
    ContinuousActionOutOfRange {
        /// Índice de la dimensión dentro del vector de acción.
        dimension: usize,
        /// Valor recibido.
        value: f64,
        /// Límite inferior permitido.
        min: f64,
        /// Límite superior permitido.
        max: f64,
    },
    /// La dimensionalidad de la acción no coincide con la definida por el espacio de acciones.
    ActionDimensionMismatch {
        /// Dimensionalidad esperada por el espacio de acciones.
        expected: usize,
        /// Dimensionalidad recibida en la acción.
        actual: usize,
    },
    /// La variante de acción (discreta vs continua) no es compatible con el espacio de acciones.
    IncompatibleActionKind {
        /// Tipo esperado según el espacio de acciones.
        expected: &'static str,
        /// Tipo recibido.
        actual: &'static str,
    },
    /// Inconsistencia temporal o de invariante al construir una [`crate::Transition`].
    InvalidTransition {
        /// Detalle de la inconsistencia detectada.
        reason: &'static str,
    },
    /// Se intentó ejecutar un paso sobre un mundo que ya alcanzó un estado terminal.
    AlreadyTerminal,
}

/// Alias semántico para errores de validación de constructores.
pub type ValidationError = DomainError;

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue { field, value } => {
                write!(f, "non-finite value in '{field}': {value}")
            }
            Self::InvalidStepDt { value } => {
                write!(
                    f,
                    "invalid step dt ({value} s): must be finite and strictly positive (> 0.0)"
                )
            }
            Self::InvalidDuration { value } => {
                write!(
                    f,
                    "invalid simulation duration ({value} s): must be finite and >= 0.0"
                )
            }
            Self::TimeOverflow { context } => {
                write!(f, "time arithmetic overflow in {context}")
            }
            Self::EmptyActionSpace => {
                write!(f, "action space cannot be empty")
            }
            Self::InvalidActionBounds { min, max } => {
                write!(
                    f,
                    "invalid action bounds [{min}, {max}]: bounds must be finite and min <= max"
                )
            }
            Self::DiscreteActionOutOfRange { index, cardinality } => {
                write!(
                    f,
                    "discrete action index {index} out of range [0, {cardinality})"
                )
            }
            Self::ContinuousActionOutOfRange {
                dimension,
                value,
                min,
                max,
            } => {
                write!(
                    f,
                    "continuous action dimension {dimension} value {value} out of bounds [{min}, {max}]"
                )
            }
            Self::ActionDimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "action dimension mismatch: expected {expected}, got {actual}"
                )
            }
            Self::IncompatibleActionKind { expected, actual } => {
                write!(
                    f,
                    "incompatible action kind: expected {expected}, got {actual}"
                )
            }
            Self::InvalidTransition { reason } => {
                write!(f, "invalid transition: {reason}")
            }
            Self::AlreadyTerminal => {
                write!(f, "cannot step a world that is already in a terminal state")
            }
        }
    }
}

impl Error for DomainError {}

/// Error de transición o evaluación en un modelo generativo simulado ([`crate::GenerativeModel`]).
///
/// Es deliberadamente un tipo distinto de [`EnvironmentError`] para separar fallos de
/// simulación pura respecto a fallos de observación o actuación en el entorno vivo.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    /// La acción proporcionada no es válida para el espacio de acciones del modelo.
    InvalidAction(DomainError),
    /// El estado de origen no es compatible con la dinámica del modelo.
    InvalidState {
        /// Motivo de incompatibilidad del estado.
        reason: &'static str,
    },
    /// El estado de origen ya es terminal y no admite nuevas transiciones simuladas.
    TerminalState,
    /// Error de dominio al construir el estado sucesor, observación o transición.
    TransitionFailed(DomainError),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAction(err) => write!(f, "model invalid action: {err}"),
            Self::InvalidState { reason } => write!(f, "model invalid state: {reason}"),
            Self::TerminalState => {
                write!(f, "model cannot simulate step from a terminal state")
            }
            Self::TransitionFailed(err) => write!(f, "model transition failed: {err}"),
        }
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAction(err) | Self::TransitionFailed(err) => Some(err),
            Self::InvalidState { .. } | Self::TerminalState => None,
        }
    }
}

/// Error de observación o actuación sobre un entorno vivo ([`crate::LiveEnvironment`]).
///
/// Es deliberadamente un tipo distinto de [`ModelError`] para distinguir fallos de
/// sensores/actuadores reales respecto a errores de simulación en un modelo generativo.
#[derive(Debug, Clone, PartialEq)]
pub enum EnvironmentError {
    /// La acción solicitada no es válida para el actuador del entorno vivo.
    InvalidAction(DomainError),
    /// Fallo al leer sensores o construir una [`crate::Observation`] del entorno vivo.
    ObservationFailed {
        /// Detalle del fallo de observación.
        reason: &'static str,
    },
    /// Fallo al ejecutar una acción sobre el [`crate::Actuator`] del entorno vivo.
    ActuationFailed {
        /// Detalle del fallo de actuación.
        reason: &'static str,
    },
    /// El entorno vivo alcanzó un estado terminal y requiere reinicio antes de actuar.
    AlreadyTerminal,
    /// El entorno vivo no está disponible o perdió conectividad.
    Unavailable {
        /// Detalle de indisponibilidad.
        reason: &'static str,
    },
}

impl fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAction(err) => write!(f, "environment invalid action: {err}"),
            Self::ObservationFailed { reason } => {
                write!(f, "environment observation failed: {reason}")
            }
            Self::ActuationFailed { reason } => {
                write!(f, "environment actuation failed: {reason}")
            }
            Self::AlreadyTerminal => {
                write!(f, "environment is already in a terminal state")
            }
            Self::Unavailable { reason } => {
                write!(f, "environment unavailable: {reason}")
            }
        }
    }
}

impl Error for EnvironmentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidAction(err) => Some(err),
            Self::ObservationFailed { .. }
            | Self::ActuationFailed { .. }
            | Self::AlreadyTerminal
            | Self::Unavailable { .. } => None,
        }
    }
}

/// Error al capturar, validar o restaurar un snapshot ramificable de simulación.
#[derive(Debug, Clone, PartialEq)]
pub enum SnapshotError {
    /// El snapshot o sobre (`SnapshotEnvelope`) está incompleto y carece de un campo obligatorio.
    IncompleteSnapshot {
        /// Nombre del campo obligatorio ausente.
        missing_field: &'static str,
    },
    /// El checksum de integridad del snapshot no coincide con su contenido.
    CorruptedChecksum {
        /// Checksum esperado registrado en el snapshot.
        expected: u64,
        /// Checksum calculado sobre el contenido actual.
        actual: u64,
    },
    /// La versión de esquema del snapshot no es compatible con la versión actual.
    UnsupportedVersion {
        /// Versión esperada por el runtime actual.
        expected: u32,
        /// Versión encontrada en el snapshot.
        actual: u32,
    },
    /// El tick del estado capturado no coincide con el tick del reloj del snapshot.
    InconsistentClock {
        /// Tick registrado en el [`crate::State`].
        state_tick: u64,
        /// Tick registrado en el snapshot.
        snapshot_tick: u64,
    },
    /// Error de dominio subyacente durante la validación o restauración del snapshot.
    Domain(DomainError),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompleteSnapshot { missing_field } => {
                write!(
                    f,
                    "incomplete snapshot: missing required field '{missing_field}'"
                )
            }
            Self::CorruptedChecksum { expected, actual } => {
                write!(
                    f,
                    "corrupted snapshot checksum: expected 0x{expected:016x}, got 0x{actual:016x}"
                )
            }
            Self::UnsupportedVersion { expected, actual } => {
                write!(
                    f,
                    "unsupported snapshot version: expected {expected}, got {actual}"
                )
            }
            Self::InconsistentClock {
                state_tick,
                snapshot_tick,
            } => {
                write!(
                    f,
                    "inconsistent snapshot clock: state tick {state_tick} != snapshot tick {snapshot_tick}"
                )
            }
            Self::Domain(err) => write!(f, "snapshot domain error: {err}"),
        }
    }
}

impl Error for SnapshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(err) => Some(err),
            Self::IncompleteSnapshot { .. }
            | Self::CorruptedChecksum { .. }
            | Self::UnsupportedVersion { .. }
            | Self::InconsistentClock { .. } => None,
        }
    }
}

impl From<DomainError> for SnapshotError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

/// Valida que un escalar `f64` sea finito (`!is_nan() && !is_infinite()`).
pub(crate) fn ensure_finite(field: &'static str, value: f64) -> Result<f64, DomainError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(DomainError::NonFiniteValue { field, value })
    }
}

/// Valida que todos los elementos de un slice `f64` sean finitos.
pub(crate) fn ensure_all_finite(field: &'static str, values: &[f64]) -> Result<(), DomainError> {
    for &value in values {
        ensure_finite(field, value)?;
    }
    Ok(())
}
