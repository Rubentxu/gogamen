//! Tipos de error y política de validación para especificaciones portables de Gogamen.

use std::error::Error;
use std::fmt;

use gogamen_core::ConnectionError;

/// Política para campos desconocidos al deserializar especificaciones JSON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UnknownFieldPolicy {
    /// Rechaza cualquier campo no reconocido en el esquema (comportamiento estricto por defecto).
    #[default]
    Reject,
    /// Ignora campos no reconocidos manteniendo la validación estricta de los conocidos.
    AllowIgnored,
}

/// Errores de parseo, validación y conversión de especificaciones portables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecError {
    /// Error sintáctico al parsear JSON.
    JsonSyntax {
        /// Desplazamiento en bytes donde falló el parser.
        offset: usize,
        /// Descripción del error sintáctico.
        message: String,
    },
    /// Clave duplicada dentro de un objeto JSON.
    DuplicateJsonKey {
        /// Clave duplicada.
        key: String,
    },
    /// Versión de esquema ausente o incompatible con la esperada.
    IncompatibleSchemaVersion {
        /// Versión esperada.
        expected: &'static str,
        /// Versión encontrada.
        found: String,
    },
    /// Falta un campo obligatorio en un objeto JSON.
    MissingField {
        /// Contexto del objeto (`SystemSpec`, `SystemNodeSpec`, etc.).
        context: &'static str,
        /// Nombre del campo faltante.
        field: &'static str,
    },
    /// Campo desconocido encontrado bajo [`UnknownFieldPolicy::Reject`].
    UnknownField {
        /// Contexto del objeto.
        context: &'static str,
        /// Nombre del campo desconocido.
        field: String,
    },
    /// Tipo o valor inválido para un campo esperado.
    InvalidFieldValue {
        /// Contexto del objeto.
        context: &'static str,
        /// Nombre del campo.
        field: &'static str,
        /// Motivo del rechazo.
        reason: String,
    },
    /// Identificador vacío o compuesto únicamente por espacios.
    EmptyIdentifier {
        /// Contexto donde apareció el identificador vacío.
        context: &'static str,
    },
    /// Identificador duplicado en una colección (`agent_id`, etc.).
    DuplicateIdentifier {
        /// Contexto de la colección.
        context: &'static str,
        /// Identificador duplicado.
        id: String,
    },
    /// Valor flotante no finito (`NaN` o infinito).
    NonFiniteFloat {
        /// Contexto donde apareció el valor.
        context: String,
    },
    /// Acción u operación no registrada en el catálogo explícito de capacidades.
    UnregisteredAction {
        /// Contexto donde se declaró la acción.
        context: &'static str,
        /// Identificador de la acción no registrada.
        action: String,
    },
    /// Intento de evaluación o incrustación de código arbitrario rechazado.
    CodeEvalRejected {
        /// Contexto donde se detectó el intento.
        context: &'static str,
        /// Campo o ruta local donde apareció.
        field: String,
        /// Motivo del rechazo.
        reason: String,
    },
    /// Conjunto de semillas `dev_seeds` y `holdout_seeds` no disjunto.
    OverlappingSeeds {
        /// Semilla presente en ambos conjuntos.
        seed: u64,
    },
    /// Error estructural o causal al construir o compilar [`gogamen_core::ConnectionGraph`].
    Graph(ConnectionError),
}

impl From<ConnectionError> for SpecError {
    fn from(value: ConnectionError) -> Self {
        Self::Graph(value)
    }
}

impl fmt::Display for SpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonSyntax { offset, message } => {
                write!(f, "JSON syntax error at byte {offset}: {message}")
            }
            Self::DuplicateJsonKey { key } => {
                write!(f, "duplicate JSON key '{key}'")
            }
            Self::IncompatibleSchemaVersion { expected, found } => {
                write!(
                    f,
                    "incompatible schema_version: expected '{expected}', found '{found}'"
                )
            }
            Self::MissingField { context, field } => {
                write!(f, "missing required field '{field}' in {context}")
            }
            Self::UnknownField { context, field } => {
                write!(f, "unknown field '{field}' in {context}")
            }
            Self::InvalidFieldValue {
                context,
                field,
                reason,
            } => {
                write!(f, "invalid field '{field}' in {context}: {reason}")
            }
            Self::EmptyIdentifier { context } => {
                write!(f, "empty identifier in {context}")
            }
            Self::DuplicateIdentifier { context, id } => {
                write!(f, "duplicate identifier '{id}' in {context}")
            }
            Self::NonFiniteFloat { context } => {
                write!(f, "non-finite float value in {context}")
            }
            Self::UnregisteredAction { context, action } => {
                write!(f, "unregistered action '{action}' in {context}")
            }
            Self::CodeEvalRejected {
                context,
                field,
                reason,
            } => {
                write!(
                    f,
                    "arbitrary code evaluation rejected in {context}.{field}: {reason}"
                )
            }
            Self::OverlappingSeeds { seed } => {
                write!(
                    f,
                    "dev_seeds and holdout_seeds must be disjoint (overlapping seed {seed})"
                )
            }
            Self::Graph(err) => write!(f, "connection graph validation error: {err}"),
        }
    }
}

impl Error for SpecError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Graph(err) => Some(err),
            _ => None,
        }
    }
}

/// Diagnóstico estructurado con ruta de campo y detalle de versión o validación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecDiagnostic {
    /// Ruta del campo afectado (por ejemplo `$.schema_version` o `$.seeds.holdout_seeds`).
    pub field_path: String,
    /// Versión esperada del esquema, cuando aplica.
    pub expected_version: Option<String>,
    /// Versión recibida del esquema, cuando aplica.
    pub found_version: Option<String>,
    /// Mensaje descriptivo del diagnóstico.
    pub message: String,
}

impl SpecDiagnostic {
    /// Crea un diagnóstico sin metadatos de versión.
    pub fn new(field_path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field_path: field_path.into(),
            expected_version: None,
            found_version: None,
            message: message.into(),
        }
    }

    /// Crea un diagnóstico de incompatibilidad o migración de versión.
    pub fn version_mismatch(
        field_path: impl Into<String>,
        expected_version: impl Into<String>,
        found_version: Option<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            field_path: field_path.into(),
            expected_version: Some(expected_version.into()),
            found_version,
            message: message.into(),
        }
    }
}

impl fmt::Display for SpecDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.field_path, self.message)?;
        if let Some(expected) = &self.expected_version {
            write!(f, " (expected: {expected}")?;
            match &self.found_version {
                Some(found) => write!(f, ", found: {found})")?,
                None => write!(f, ", found: <missing>)")?,
            }
        }
        Ok(())
    }
}

/// Error tipado al migrar una especificación JSON entre versiones de esquema.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecMigrationError {
    /// Diagnóstico estructurado con ruta de campo y versión esperada vs recibida.
    pub diagnostic: SpecDiagnostic,
    /// Error subyacente de especificación, si existe.
    pub source: Option<Box<SpecError>>,
}

impl SpecMigrationError {
    /// Construye un error de migración a partir de un diagnóstico y una causa opcional.
    pub fn new(diagnostic: SpecDiagnostic, source: Option<SpecError>) -> Self {
        Self {
            diagnostic,
            source: source.map(Box::new),
        }
    }
}

impl fmt::Display for SpecMigrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "spec migration error: {}", self.diagnostic)?;
        if let Some(src) = &self.source {
            write!(f, ": {src}")?;
        }
        Ok(())
    }
}

impl Error for SpecMigrationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref().map(|e| e as &(dyn Error + 'static))
    }
}
