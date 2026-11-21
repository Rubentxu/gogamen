//! Errores explícitos de proyección 2D independiente del renderer (`std`-only).

use std::error::Error;
use std::fmt;

use gogamen_core::{ObservabilityError, SnapshotError};

use crate::scene::LayerId;

/// Errores producidos al construir coordenadas, escenas 2D o proyectar estados/frames.
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionError {
    /// Se detectó un valor numérico no finito (`NaN` o infinito).
    NonFiniteValue {
        /// Campo donde se detectó el valor inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Se proporcionó un parámetro de escala o dimensión inválido (`<= 0.0` o `< 0.0`).
    InvalidDimension {
        /// Campo donde se detectó la dimensión inválida.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// Una polilínea requiere al menos dos puntos.
    InsufficientPolylinePoints {
        /// Número de puntos recibidos.
        count: usize,
    },
    /// Se intentó registrar una capa duplicada en la escena.
    DuplicateLayerId {
        /// Identificador de la capa duplicada.
        layer_id: LayerId,
    },
    /// Se intentó registrar un nodo con un identificador ya existente en la escena.
    DuplicateNodeId {
        /// Identificador numérico del nodo duplicado.
        node_id: u64,
    },
    /// Error propagado desde la verificación de un [`gogamen_core::SimulationSnapshot`].
    Snapshot(SnapshotError),
    /// Error propagado desde la validación de un [`gogamen_core::VisualFrame`].
    Observability(ObservabilityError),
    /// Parámetro de muestreo o presupuesto de overlay inválido.
    InvalidSamplingConfig {
        /// Campo donde se detectó el parámetro inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// El presupuesto máximo de bytes por frame no alcanza ni siquiera para la cabecera mínima de escena.
    OverlayBudgetTooSmall {
        /// Bytes mínimos requeridos para la escena vacía.
        required_bytes: usize,
        /// Límite configurado en `max_bytes_per_frame`.
        max_bytes: usize,
    },
    /// Textura de imagen inválida (dimensiones, longitud de buffer o paleta).
    InvalidTexture {
        /// Motivo estático del rechazo.
        reason: &'static str,
    },
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue { field, value } => {
                write!(f, "non-finite value in {field}: {value}")
            }
            Self::InvalidDimension { field, value } => {
                write!(f, "invalid dimension in {field}: {value}")
            }
            Self::InsufficientPolylinePoints { count } => {
                write!(f, "polyline requires at least 2 points, got {count}")
            }
            Self::DuplicateLayerId { layer_id } => {
                write!(f, "duplicate layer id in scene: {layer_id}")
            }
            Self::DuplicateNodeId { node_id } => {
                write!(f, "duplicate node id in scene: {node_id}")
            }
            Self::Snapshot(err) => write!(f, "snapshot error: {err}"),
            Self::Observability(err) => write!(f, "observability error: {err}"),
            Self::InvalidSamplingConfig { field, value } => {
                write!(f, "invalid sampling config in {field}: {value}")
            }
            Self::InvalidTexture { reason } => write!(f, "invalid texture: {reason}"),
            Self::OverlayBudgetTooSmall {
                required_bytes,
                max_bytes,
            } => {
                write!(
                    f,
                    "overlay max_bytes_per_frame ({max_bytes}) is smaller than minimum scene size ({required_bytes})"
                )
            }
        }
    }
}

impl Error for ProjectionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Snapshot(err) => Some(err),
            Self::Observability(err) => Some(err),
            Self::NonFiniteValue { .. }
            | Self::InvalidDimension { .. }
            | Self::InsufficientPolylinePoints { .. }
            | Self::DuplicateLayerId { .. }
            | Self::DuplicateNodeId { .. }
            | Self::InvalidSamplingConfig { .. }
            | Self::OverlayBudgetTooSmall { .. }
            | Self::InvalidTexture { .. } => None,
        }
    }
}

impl From<SnapshotError> for ProjectionError {
    fn from(err: SnapshotError) -> Self {
        Self::Snapshot(err)
    }
}

impl From<ObservabilityError> for ProjectionError {
    fn from(err: ObservabilityError) -> Self {
        Self::Observability(err)
    }
}

/// Valida que un escalar `f64` sea finito.
pub(crate) fn ensure_finite(field: &'static str, value: f64) -> Result<f64, ProjectionError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ProjectionError::NonFiniteValue { field, value })
    }
}

/// Valida que un escalar `f64` sea finito y estrictamente positivo (`> 0.0`).
pub(crate) fn ensure_positive(field: &'static str, value: f64) -> Result<f64, ProjectionError> {
    ensure_finite(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ProjectionError::InvalidDimension { field, value })
    }
}

/// Valida que un escalar `f64` sea finito y no negativo (`>= 0.0`).
pub(crate) fn ensure_non_negative(field: &'static str, value: f64) -> Result<f64, ProjectionError> {
    ensure_finite(field, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ProjectionError::InvalidDimension { field, value })
    }
}
