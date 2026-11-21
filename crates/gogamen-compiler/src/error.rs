//! Errores explícitos de compilación y validación previa a la ejecución.

use std::error::Error;
use std::fmt;

use gogamen_core::{ConnectionError, ModuleContractError, NodeId, PortId};
use gogamen_spec::SpecError;

/// Error producido durante la compilación de un [`gogamen_spec::SystemSpec`]
/// a un [`crate::ExecutablePlan`].
///
/// Cualquier fallo detiene la compilación antes de inicializar o asignar
/// recursos, fábricas o actuadores.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// Error en la especificación declarativa o su esquema.
    Spec(SpecError),
    /// Error estructural o causal en el grafo de conexiones.
    Graph(ConnectionError),
    /// Error de contrato de módulo, capacidad, permisos, versión o perfil temporal.
    Contract(ModuleContractError),
    /// Configuración declarativa inválida en un nodo (por ejemplo versión o módulo malformados).
    InvalidNodeConfig {
        /// Nodo donde ocurrió el error.
        node_id: NodeId,
        /// Clave de configuración o atributo afectado.
        key: String,
        /// Motivo del rechazo.
        reason: String,
    },
    /// Incompatibilidad entre los puertos declarados en el nodo y el contrato de la operación.
    PortContractMismatch {
        /// Nodo donde ocurrió la discrepancia.
        node_id: NodeId,
        /// Puerto afectado.
        port_id: PortId,
        /// Motivo de la incompatibilidad.
        reason: String,
    },
}

impl From<SpecError> for CompileError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

impl From<ConnectionError> for CompileError {
    fn from(value: ConnectionError) -> Self {
        Self::Graph(value)
    }
}

impl From<ModuleContractError> for CompileError {
    fn from(value: ModuleContractError) -> Self {
        Self::Contract(value)
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(err) => write!(f, "spec validation error: {err}"),
            Self::Graph(err) => write!(f, "connection graph compilation error: {err}"),
            Self::Contract(err) => write!(f, "capability contract error: {err}"),
            Self::InvalidNodeConfig {
                node_id,
                key,
                reason,
            } => write!(
                f,
                "invalid configuration for node '{node_id}' key '{key}': {reason}"
            ),
            Self::PortContractMismatch {
                node_id,
                port_id,
                reason,
            } => write!(
                f,
                "port contract mismatch on node '{node_id}' port '{port_id}': {reason}"
            ),
        }
    }
}

impl Error for CompileError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Spec(err) => Some(err),
            Self::Graph(err) => Some(err),
            Self::Contract(err) => Some(err),
            Self::InvalidNodeConfig { .. } | Self::PortContractMismatch { .. } => None,
        }
    }
}
