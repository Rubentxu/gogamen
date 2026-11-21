//! Contrato unificado de capacidades, efectos y perfil temporal de módulos (`std`-only).
//!
//! Este módulo define el catálogo y las reglas de validación estática/previa a la ejecución
//! para módulos integrados (`Builtin`) y de terceros (`ThirdParty`) bajo **exactamente el
//! mismo contrato**, sin elevación implícita de permisos:
//! - [`ModuleManifest`]: manifiesto declarativo de un módulo con su [`ModuleOrigin`],
//!   [`CapabilityVersion`], [`PermissionSet`] otorgado y operaciones registradas.
//! - [`OperationSpec`]: especificación de una operación con sus puertos tipados ([`PortSpec`]),
//!   permisos requeridos, efectos declarados ([`EffectKind`]) y [`TemporalProfile`].
//! - [`DeclaredProperty`] vs [`ObservedEvidence`]: separación explícita entre lo que un módulo
//!   declara en su contrato y la evidencia observada durante inspección o ejecución.
//! - [`CapabilityCatalog`]: catálogo unificado que registra manifiestos y valida versiones,
//!   permisos y compatibilidad temporal de rutas/grafos (rechazando `BlockingIo` en rutas
//!   acotadas como [`TemporalProfile::BoundedTick`] o [`TemporalProfile::BoundedRt`]).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::connection::{PortId, PortSpec};

/// Identificador lógico de un módulo registrado en [`CapabilityCatalog`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModuleId(String);

impl ModuleId {
    /// Crea un identificador de módulo a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ModuleId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ModuleId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for ModuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador lógico de una operación expuesta por un módulo.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperationId(String);

impl OperationId {
    /// Crea un identificador de operación a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for OperationId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for OperationId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Origen de un módulo dentro del catálogo unificado.
///
/// Tanto [`ModuleOrigin::Builtin`] como [`ModuleOrigin::ThirdParty`] se someten al mismo
/// contrato de capacidades, efectos, versiones y permisos explícitos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModuleOrigin {
    /// Módulo integrado en la distribución base de Gogamen.
    Builtin,
    /// Módulo externo o de terceros.
    ThirdParty,
}

impl fmt::Display for ModuleOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Builtin => f.write_str("Builtin"),
            Self::ThirdParty => f.write_str("ThirdParty"),
        }
    }
}

/// Versión semántica de una capacidad, módulo u operación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityVersion {
    /// Versión mayor; cambios incompatibles incrementan `major`.
    pub major: u32,
    /// Versión menor; ampliaciones compatibles hacia atrás incrementan `minor`.
    pub minor: u32,
    /// Versión de parche; correcciones compatibles incrementan `patch`.
    pub patch: u32,
}

impl CapabilityVersion {
    /// Crea una versión semántica explícita.
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Indica si esta versión disponible satisface la versión `required`.
    ///
    /// Es compatible únicamente si `self.major == required.major` y
    /// `(self.minor, self.patch) >= (required.minor, required.patch)`.
    pub const fn is_compatible_with(&self, required: &Self) -> bool {
        if self.major != required.major {
            return false;
        }
        if self.minor != required.minor {
            return self.minor > required.minor;
        }
        self.patch >= required.patch
    }
}

impl fmt::Display for CapabilityVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Identificador explícito de un permiso de capacidad.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Permission(String);

impl Permission {
    /// Crea un permiso nominal a partir de cualquier cadena convertible.
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Devuelve la representación textual del permiso.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Permiso estándar de lectura de estado.
    pub fn state_read() -> Self {
        Self("state:read".to_owned())
    }

    /// Permiso estándar de mutación de estado.
    pub fn state_write() -> Self {
        Self("state:write".to_owned())
    }

    /// Permiso estándar de acceso a sistema de archivos.
    pub fn filesystem() -> Self {
        Self("io:filesystem".to_owned())
    }

    /// Permiso estándar de acceso a red.
    pub fn network() -> Self {
        Self("io:network".to_owned())
    }

    /// Permiso estándar de operaciones de E/S bloqueante.
    pub fn blocking_io() -> Self {
        Self("io:blocking".to_owned())
    }
}

impl From<&str> for Permission {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for Permission {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Conjunto ordenado y determinista de permisos requeridos u otorgados.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PermissionSet {
    permissions: BTreeSet<Permission>,
}

impl PermissionSet {
    /// Crea un conjunto vacío de permisos.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Crea un conjunto a partir de un iterador de permisos.
    pub fn from_permissions<I, P>(iter: I) -> Self
    where
        I: IntoIterator<Item = P>,
        P: Into<Permission>,
    {
        Self {
            permissions: iter.into_iter().map(Into::into).collect(),
        }
    }

    /// Añade un permiso al conjunto (estilo builder).
    pub fn with_permission(mut self, permission: impl Into<Permission>) -> Self {
        self.permissions.insert(permission.into());
        self
    }

    /// Inserta un permiso en el conjunto.
    pub fn insert(&mut self, permission: impl Into<Permission>) -> bool {
        self.permissions.insert(permission.into())
    }

    /// Indica si el conjunto contiene el permiso consultado.
    pub fn contains(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Indica si todos los permisos de `self` están incluidos en `granted`.
    pub fn is_subset_of(&self, granted: &Self) -> bool {
        self.permissions.is_subset(&granted.permissions)
    }

    /// Devuelve el subconjunto de permisos presentes en `self` que faltan en `granted`.
    pub fn missing_from(&self, granted: &Self) -> Self {
        Self {
            permissions: self
                .permissions
                .difference(&granted.permissions)
                .cloned()
                .collect(),
        }
    }

    /// Número de permisos en el conjunto.
    pub fn len(&self) -> usize {
        self.permissions.len()
    }

    /// Indica si el conjunto no contiene permisos.
    pub fn is_empty(&self) -> bool {
        self.permissions.is_empty()
    }

    /// Iterador determinista sobre los permisos del conjunto.
    pub fn iter(&self) -> impl Iterator<Item = &Permission> {
        self.permissions.iter()
    }
}

impl fmt::Display for PermissionSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        for (idx, perm) in self.permissions.iter().enumerate() {
            if idx > 0 {
                f.write_str(", ")?;
            }
            fmt::Display::fmt(perm, f)?;
        }
        f.write_str("}")
    }
}

/// Clasificación explícita de efectos declarados por una operación de módulo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EffectKind {
    /// Cálculo puro sin efectos secundarios observables.
    Pure,
    /// Mutación acotada de estado interno o de dominio.
    StateMutation,
    /// Operación de E/S bloqueante (prohibida en rutas temporales acotadas).
    BlockingIo,
    /// Comunicación por red.
    Network,
    /// Lectura o escritura sobre sistema de archivos.
    Filesystem,
}

impl EffectKind {
    /// Indica si el efecto implica E/S bloqueante incompatible con rutas acotadas.
    pub const fn is_blocking(&self) -> bool {
        matches!(self, Self::BlockingIo)
    }
}

impl fmt::Display for EffectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pure => f.write_str("Pure"),
            Self::StateMutation => f.write_str("StateMutation"),
            Self::BlockingIo => f.write_str("BlockingIo"),
            Self::Network => f.write_str("Network"),
            Self::Filesystem => f.write_str("Filesystem"),
        }
    }
}

/// Perfil temporal declarado para una operación o una ruta de ejecución.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TemporalProfile {
    /// Ruta o módulo acotado al presupuesto determinista de cada tick.
    BoundedTick,
    /// Ruta o módulo acotado de tiempo real estricto.
    BoundedRt,
    /// Ejecución de mejor esfuerzo sin garantía estricta de latencia.
    BestEffort,
    /// Ejecución en segundo plano no acotada temporalmente.
    UnboundedBackground,
}

impl TemporalProfile {
    /// Indica si el perfil exige ejecución temporalmente acotada (`BoundedTick` o `BoundedRt`).
    pub const fn is_bounded(&self) -> bool {
        matches!(self, Self::BoundedTick | Self::BoundedRt)
    }
}

impl fmt::Display for TemporalProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BoundedTick => f.write_str("BoundedTick"),
            Self::BoundedRt => f.write_str("BoundedRt"),
            Self::BestEffort => f.write_str("BestEffort"),
            Self::UnboundedBackground => f.write_str("UnboundedBackground"),
        }
    }
}

/// Propiedad declarada por el manifiesto de un módulo u operación.
///
/// Representa una afirmación contractual previa a la ejecución y **no** constituye
/// evidencia verificada en tiempo de ejecución ([`ObservedEvidence`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeclaredProperty {
    /// Versión declarada de la operación.
    Version(CapabilityVersion),
    /// Puertos tipados declarados.
    Ports(Vec<PortSpec>),
    /// Permisos requeridos declarados.
    RequiredPermissions(PermissionSet),
    /// Efectos declarados.
    Effects(BTreeSet<EffectKind>),
    /// Perfil temporal declarado.
    TemporalProfile(TemporalProfile),
}

/// Evidencia observada empíricamente durante inspección o ejecución de una operación.
///
/// Se mantiene separada de [`DeclaredProperty`] para evitar confundir promesas
/// declarativas del manifiesto con comportamiento real observado.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservedEvidence {
    /// Efecto real observado durante la ejecución de la operación.
    EffectObserved {
        /// Módulo observado.
        module_id: ModuleId,
        /// Operación observada.
        operation_id: OperationId,
        /// Efecto detectado.
        effect: EffectKind,
    },
    /// Permiso ejercido por la operación durante la ejecución.
    PermissionExercised {
        /// Módulo observado.
        module_id: ModuleId,
        /// Operación observada.
        operation_id: OperationId,
        /// Permiso utilizado.
        permission: Permission,
    },
    /// Medición de tiempo o trabajo consumido observado para una invocación.
    ExecutionMeasured {
        /// Módulo observado.
        module_id: ModuleId,
        /// Operación observada.
        operation_id: OperationId,
        /// Duración observada en nanosegundos.
        duration_ns: u64,
    },
}

/// Especificación declarativa de una operación expuesta por un módulo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSpec {
    /// Identificador único de la operación dentro del módulo.
    pub id: OperationId,
    /// Versión semántica de la operación.
    pub version: CapabilityVersion,
    /// Puertos tipados de entrada y salida de la operación.
    pub ports: Vec<PortSpec>,
    /// Permisos mínimos que requiere la operación para ejecutarse.
    pub required_permissions: PermissionSet,
    /// Conjunto de efectos declarados por la operación.
    pub declared_effects: BTreeSet<EffectKind>,
    /// Perfil temporal declarado por la operación.
    pub temporal_profile: TemporalProfile,
}

impl OperationSpec {
    /// Crea una operación con perfil [`TemporalProfile::BoundedTick`] y sin efectos ni permisos.
    pub fn new(id: impl Into<OperationId>, version: CapabilityVersion) -> Self {
        Self {
            id: id.into(),
            version,
            ports: Vec::new(),
            required_permissions: PermissionSet::empty(),
            declared_effects: BTreeSet::new(),
            temporal_profile: TemporalProfile::BoundedTick,
        }
    }

    /// Añade un puerto tipado a la operación (estilo builder).
    pub fn with_port(mut self, port: PortSpec) -> Self {
        self.ports.push(port);
        self
    }

    /// Añade un puerto de entrada tipado (estilo builder).
    pub fn with_input(
        mut self,
        id: impl Into<PortId>,
        port_type: crate::connection::PortType,
    ) -> Self {
        self.ports.push(PortSpec::input(id, port_type));
        self
    }

    /// Añade un puerto de salida tipado (estilo builder).
    pub fn with_output(
        mut self,
        id: impl Into<PortId>,
        port_type: crate::connection::PortType,
    ) -> Self {
        self.ports.push(PortSpec::output(id, port_type));
        self
    }

    /// Añade un permiso requerido por la operación (estilo builder).
    pub fn with_required_permission(mut self, permission: impl Into<Permission>) -> Self {
        self.required_permissions.insert(permission);
        self
    }

    /// Reemplaza el conjunto de permisos requeridos (estilo builder).
    pub fn with_required_permissions(mut self, permissions: PermissionSet) -> Self {
        self.required_permissions = permissions;
        self
    }

    /// Añade un efecto declarado por la operación (estilo builder).
    pub fn with_effect(mut self, effect: EffectKind) -> Self {
        self.declared_effects.insert(effect);
        self
    }

    /// Añade varios efectos declarados por la operación (estilo builder).
    pub fn with_effects<I>(mut self, effects: I) -> Self
    where
        I: IntoIterator<Item = EffectKind>,
    {
        self.declared_effects.extend(effects);
        self
    }

    /// Establece el perfil temporal declarado (estilo builder).
    pub fn with_temporal_profile(mut self, temporal_profile: TemporalProfile) -> Self {
        self.temporal_profile = temporal_profile;
        self
    }

    /// Indica si la operación declara algún efecto de E/S bloqueante.
    pub fn declares_blocking_io(&self) -> bool {
        self.declared_effects.iter().any(EffectKind::is_blocking)
    }

    /// Devuelve la lista explícita de [`DeclaredProperty`] asociadas a esta operación.
    pub fn declared_properties(&self) -> Vec<DeclaredProperty> {
        vec![
            DeclaredProperty::Version(self.version),
            DeclaredProperty::Ports(self.ports.clone()),
            DeclaredProperty::RequiredPermissions(self.required_permissions.clone()),
            DeclaredProperty::Effects(self.declared_effects.clone()),
            DeclaredProperty::TemporalProfile(self.temporal_profile),
        ]
    }
}

/// Manifiesto declarativo de un módulo registrado en [`CapabilityCatalog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModuleManifest {
    /// Identificador único del módulo.
    pub id: ModuleId,
    /// Origen del módulo (`Builtin` o `ThirdParty`).
    pub origin: ModuleOrigin,
    /// Versión del manifiesto del módulo.
    pub version: CapabilityVersion,
    /// Conjunto explícito de permisos otorgados al módulo.
    pub granted_permissions: PermissionSet,
    /// Operaciones declaradas por el módulo.
    pub operations: Vec<OperationSpec>,
}

impl ModuleManifest {
    /// Crea un manifiesto sin permisos otorgados ni operaciones todavía.
    pub fn new(id: impl Into<ModuleId>, origin: ModuleOrigin, version: CapabilityVersion) -> Self {
        Self {
            id: id.into(),
            origin,
            version,
            granted_permissions: PermissionSet::empty(),
            operations: Vec::new(),
        }
    }

    /// Otorga un permiso explícito al módulo (estilo builder).
    pub fn with_granted_permission(mut self, permission: impl Into<Permission>) -> Self {
        self.granted_permissions.insert(permission);
        self
    }

    /// Establece el conjunto explícito de permisos otorgados (estilo builder).
    pub fn with_granted_permissions(mut self, permissions: PermissionSet) -> Self {
        self.granted_permissions = permissions;
        self
    }

    /// Registra una operación dentro del manifiesto (estilo builder).
    pub fn with_operation(mut self, operation: OperationSpec) -> Self {
        self.operations.push(operation);
        self
    }
}

/// Solicitud de validación o enlace de una operación dentro de una ruta temporal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationBinding {
    /// Módulo proveedor de la operación.
    pub module_id: ModuleId,
    /// Operación solicitada.
    pub operation_id: OperationId,
    /// Versión mínima compatible requerida por el llamador o grafo.
    pub required_version: CapabilityVersion,
}

impl OperationBinding {
    /// Crea una solicitud de enlace hacia `module_id::operation_id` con `required_version`.
    pub fn new(
        module_id: impl Into<ModuleId>,
        operation_id: impl Into<OperationId>,
        required_version: CapabilityVersion,
    ) -> Self {
        Self {
            module_id: module_id.into(),
            operation_id: operation_id.into(),
            required_version,
        }
    }
}

/// Operación validada contra [`CapabilityCatalog`], conservando su origen y propiedades declaradas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedOperation {
    /// Identificador del módulo proveedor.
    pub module_id: ModuleId,
    /// Origen del módulo (`Builtin` o `ThirdParty`).
    pub origin: ModuleOrigin,
    /// Especificación completa de la operación validada.
    pub operation: OperationSpec,
    /// Propiedades declaradas por la operación (distintas de [`ObservedEvidence`]).
    pub declared_properties: Vec<DeclaredProperty>,
}

/// Errores explícitos de validación del contrato de módulos, capacidades y efectos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModuleContractError {
    /// Intento de registrar dos módulos con el mismo identificador.
    DuplicateModule {
        /// Identificador duplicado.
        module_id: ModuleId,
    },
    /// Intento de registrar dos operaciones con el mismo identificador en un módulo.
    DuplicateOperation {
        /// Módulo propietario.
        module_id: ModuleId,
        /// Operación duplicada.
        operation_id: OperationId,
    },
    /// Intento de registrar dos puertos con el mismo identificador en una operación.
    DuplicatePort {
        /// Módulo propietario.
        module_id: ModuleId,
        /// Operación propietaria.
        operation_id: OperationId,
        /// Puerto duplicado.
        port_id: PortId,
    },
    /// El módulo solicitado no existe en el catálogo.
    UnregisteredModule {
        /// Módulo ausente.
        module_id: ModuleId,
    },
    /// La operación solicitada no está registrada en el módulo.
    UnregisteredOperation {
        /// Módulo consultado.
        module_id: ModuleId,
        /// Operación ausente.
        operation_id: OperationId,
    },
    /// La versión disponible de la operación es incompatible con la requerida.
    IncompatibleVersion {
        /// Módulo consultado.
        module_id: ModuleId,
        /// Operación consultada.
        operation_id: OperationId,
        /// Versión mínima requerida.
        required: CapabilityVersion,
        /// Versión disponible en el catálogo.
        available: CapabilityVersion,
    },
    /// La operación requiere permisos que no están otorgados en el [`PermissionSet`] del módulo.
    PermissionDenied {
        /// Módulo evaluado.
        module_id: ModuleId,
        /// Operación evaluada.
        operation_id: OperationId,
        /// Origen del módulo (`Builtin` o `ThirdParty`).
        origin: ModuleOrigin,
        /// Permisos faltantes respecto a los otorgados.
        missing: PermissionSet,
    },
    /// Una ruta temporal acotada rechaza una operación que declara [`EffectKind::BlockingIo`].
    BlockingIoInBoundedPath {
        /// Módulo rechazado.
        module_id: ModuleId,
        /// Operación rechazada.
        operation_id: OperationId,
        /// Perfil temporal de la ruta o grafo.
        path_profile: TemporalProfile,
        /// Perfil temporal declarado por la operación.
        operation_profile: TemporalProfile,
    },
    /// El perfil temporal de la operación no es compatible con una ruta temporal acotada.
    TemporalProfileMismatch {
        /// Módulo rechazado.
        module_id: ModuleId,
        /// Operación rechazada.
        operation_id: OperationId,
        /// Perfil temporal exigido por la ruta.
        path_profile: TemporalProfile,
        /// Perfil temporal declarado por la operación.
        operation_profile: TemporalProfile,
    },
    /// La evidencia observada contradice las propiedades declaradas por la operación.
    UndeclaredObservedEffect {
        /// Módulo auditado.
        module_id: ModuleId,
        /// Operación auditada.
        operation_id: OperationId,
        /// Efecto observado no declarado en el manifiesto.
        observed: EffectKind,
    },
}

impl fmt::Display for ModuleContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateModule { module_id } => {
                write!(f, "módulo duplicado en el catálogo: '{module_id}'")
            }
            Self::DuplicateOperation {
                module_id,
                operation_id,
            } => write!(
                f,
                "operación duplicada '{operation_id}' en el módulo '{module_id}'"
            ),
            Self::DuplicatePort {
                module_id,
                operation_id,
                port_id,
            } => write!(
                f,
                "puerto duplicado '{port_id}' en '{module_id}::{operation_id}'"
            ),
            Self::UnregisteredModule { module_id } => {
                write!(f, "módulo no registrado: '{module_id}'")
            }
            Self::UnregisteredOperation {
                module_id,
                operation_id,
            } => write!(f, "operación no registrada: '{module_id}::{operation_id}'"),
            Self::IncompatibleVersion {
                module_id,
                operation_id,
                required,
                available,
            } => write!(
                f,
                "versión incompatible para '{module_id}::{operation_id}': requerida {required}, disponible {available}"
            ),
            Self::PermissionDenied {
                module_id,
                operation_id,
                origin,
                missing,
            } => write!(
                f,
                "permiso denegado para '{module_id}::{operation_id}' ({origin}): faltan {missing}"
            ),
            Self::BlockingIoInBoundedPath {
                module_id,
                operation_id,
                path_profile,
                operation_profile,
            } => write!(
                f,
                "operación '{module_id}::{operation_id}' rechazada: declara BlockingIo en ruta acotada {path_profile} (perfil operación: {operation_profile})"
            ),
            Self::TemporalProfileMismatch {
                module_id,
                operation_id,
                path_profile,
                operation_profile,
            } => write!(
                f,
                "perfil temporal incompatible para '{module_id}::{operation_id}': ruta {path_profile}, operación {operation_profile}"
            ),
            Self::UndeclaredObservedEffect {
                module_id,
                operation_id,
                observed,
            } => write!(
                f,
                "evidencia observada no declarada en '{module_id}::{operation_id}': efecto {observed}"
            ),
        }
    }
}

impl Error for ModuleContractError {}

#[derive(Debug, Clone)]
struct RegisteredModule {
    manifest: ModuleManifest,
    operations: BTreeMap<OperationId, OperationSpec>,
}

/// Catálogo unificado de módulos y operaciones bajo un contrato único.
///
/// Tanto [`ModuleOrigin::Builtin`] como [`ModuleOrigin::ThirdParty`] se registran y validan
/// con exactamente las mismas reglas de versión, permisos explícitos, efectos y perfil temporal.
#[derive(Debug, Clone, Default)]
pub struct CapabilityCatalog {
    modules: BTreeMap<ModuleId, RegisteredModule>,
}

impl CapabilityCatalog {
    /// Crea un catálogo vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra un [`ModuleManifest`] verificando unicidad de módulo, operaciones y puertos.
    pub fn register_module(&mut self, manifest: ModuleManifest) -> Result<(), ModuleContractError> {
        if self.modules.contains_key(&manifest.id) {
            return Err(ModuleContractError::DuplicateModule {
                module_id: manifest.id,
            });
        }

        let mut operations = BTreeMap::new();
        for op in &manifest.operations {
            if operations.contains_key(&op.id) {
                return Err(ModuleContractError::DuplicateOperation {
                    module_id: manifest.id.clone(),
                    operation_id: op.id.clone(),
                });
            }

            let mut seen_ports = BTreeSet::new();
            for port in &op.ports {
                if !seen_ports.insert(&port.id) {
                    return Err(ModuleContractError::DuplicatePort {
                        module_id: manifest.id.clone(),
                        operation_id: op.id.clone(),
                        port_id: port.id.clone(),
                    });
                }
            }

            operations.insert(op.id.clone(), op.clone());
        }

        self.modules.insert(
            manifest.id.clone(),
            RegisteredModule {
                manifest,
                operations,
            },
        );
        Ok(())
    }

    /// Devuelve el manifiesto de un módulo registrado si existe.
    pub fn module(&self, module_id: &ModuleId) -> Option<&ModuleManifest> {
        self.modules.get(module_id).map(|entry| &entry.manifest)
    }

    /// Itera sobre todos los manifiestos de módulos registrados en orden determinista.
    pub fn modules(&self) -> impl Iterator<Item = &ModuleManifest> {
        self.modules.values().map(|entry| &entry.manifest)
    }

    /// Devuelve los identificadores de todas las operaciones registradas en orden determinista.
    pub fn operation_ids(&self) -> Vec<OperationId> {
        let mut ids = BTreeSet::new();
        for entry in self.modules.values() {
            for op_id in entry.operations.keys() {
                ids.insert(op_id.clone());
            }
        }
        ids.into_iter().collect()
    }

    /// Indica si existe alguna operación registrada con el [`OperationId`] indicado.
    pub fn contains_operation(&self, operation_id: &OperationId) -> bool {
        self.modules
            .values()
            .any(|entry| entry.operations.contains_key(operation_id))
    }

    /// Valida y resuelve una operación individual bajo un perfil temporal objetivo.
    ///
    /// Aplica exactamente las mismas comprobaciones a módulos `Builtin` y `ThirdParty`:
    /// 1. Existencia de módulo y operación.
    /// 2. Compatibilidad de [`CapabilityVersion`].
    /// 3. Subconjunto estricto de permisos (`required_permissions ⊆ granted_permissions`),
    ///    sin elevar permisos implícitamente por origen.
    /// 4. Rechazo de [`EffectKind::BlockingIo`] y perfiles no acotados cuando `path_profile`
    ///    es acotado (`BoundedTick` / `BoundedRt`).
    pub fn validate_operation(
        &self,
        binding: &OperationBinding,
        path_profile: TemporalProfile,
    ) -> Result<ValidatedOperation, ModuleContractError> {
        let entry = self.modules.get(&binding.module_id).ok_or_else(|| {
            ModuleContractError::UnregisteredModule {
                module_id: binding.module_id.clone(),
            }
        })?;

        let op = entry.operations.get(&binding.operation_id).ok_or_else(|| {
            ModuleContractError::UnregisteredOperation {
                module_id: binding.module_id.clone(),
                operation_id: binding.operation_id.clone(),
            }
        })?;

        if !op.version.is_compatible_with(&binding.required_version) {
            return Err(ModuleContractError::IncompatibleVersion {
                module_id: binding.module_id.clone(),
                operation_id: binding.operation_id.clone(),
                required: binding.required_version,
                available: op.version,
            });
        }

        if !op
            .required_permissions
            .is_subset_of(&entry.manifest.granted_permissions)
        {
            let missing = op
                .required_permissions
                .missing_from(&entry.manifest.granted_permissions);
            return Err(ModuleContractError::PermissionDenied {
                module_id: binding.module_id.clone(),
                operation_id: binding.operation_id.clone(),
                origin: entry.manifest.origin,
                missing,
            });
        }

        if path_profile.is_bounded() {
            if op.declares_blocking_io() {
                return Err(ModuleContractError::BlockingIoInBoundedPath {
                    module_id: binding.module_id.clone(),
                    operation_id: binding.operation_id.clone(),
                    path_profile,
                    operation_profile: op.temporal_profile,
                });
            }
            if !op.temporal_profile.is_bounded() {
                return Err(ModuleContractError::TemporalProfileMismatch {
                    module_id: binding.module_id.clone(),
                    operation_id: binding.operation_id.clone(),
                    path_profile,
                    operation_profile: op.temporal_profile,
                });
            }
        }

        Ok(ValidatedOperation {
            module_id: binding.module_id.clone(),
            origin: entry.manifest.origin,
            declared_properties: op.declared_properties(),
            operation: op.clone(),
        })
    }

    /// Valida una ruta o grafo completo de operaciones bajo un [`TemporalProfile`] común.
    pub fn validate_temporal_path(
        &self,
        path_profile: TemporalProfile,
        bindings: &[OperationBinding],
    ) -> Result<Vec<ValidatedOperation>, ModuleContractError> {
        let mut validated = Vec::with_capacity(bindings.len());
        for binding in bindings {
            validated.push(self.validate_operation(binding, path_profile)?);
        }
        Ok(validated)
    }

    /// Audita una pieza de [`ObservedEvidence`] frente a las propiedades declaradas del catálogo.
    pub fn audit_observed_evidence(
        &self,
        evidence: &ObservedEvidence,
    ) -> Result<(), ModuleContractError> {
        match evidence {
            ObservedEvidence::EffectObserved {
                module_id,
                operation_id,
                effect,
            } => {
                let entry = self.modules.get(module_id).ok_or_else(|| {
                    ModuleContractError::UnregisteredModule {
                        module_id: module_id.clone(),
                    }
                })?;
                let op = entry.operations.get(operation_id).ok_or_else(|| {
                    ModuleContractError::UnregisteredOperation {
                        module_id: module_id.clone(),
                        operation_id: operation_id.clone(),
                    }
                })?;
                if !op.declared_effects.contains(effect) {
                    return Err(ModuleContractError::UndeclaredObservedEffect {
                        module_id: module_id.clone(),
                        operation_id: operation_id.clone(),
                        observed: *effect,
                    });
                }
                Ok(())
            }
            ObservedEvidence::PermissionExercised {
                module_id,
                operation_id,
                permission,
            } => {
                let entry = self.modules.get(module_id).ok_or_else(|| {
                    ModuleContractError::UnregisteredModule {
                        module_id: module_id.clone(),
                    }
                })?;
                let _op = entry.operations.get(operation_id).ok_or_else(|| {
                    ModuleContractError::UnregisteredOperation {
                        module_id: module_id.clone(),
                        operation_id: operation_id.clone(),
                    }
                })?;
                if !entry.manifest.granted_permissions.contains(permission) {
                    let missing = PermissionSet::empty().with_permission(permission.clone());
                    return Err(ModuleContractError::PermissionDenied {
                        module_id: module_id.clone(),
                        operation_id: operation_id.clone(),
                        origin: entry.manifest.origin,
                        missing,
                    });
                }
                Ok(())
            }
            ObservedEvidence::ExecutionMeasured {
                module_id,
                operation_id,
                duration_ns: _,
            } => {
                let entry = self.modules.get(module_id).ok_or_else(|| {
                    ModuleContractError::UnregisteredModule {
                        module_id: module_id.clone(),
                    }
                })?;
                if !entry.operations.contains_key(operation_id) {
                    return Err(ModuleContractError::UnregisteredOperation {
                        module_id: module_id.clone(),
                        operation_id: operation_id.clone(),
                    });
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::connection::PortType;

    fn sample_builtin_manifest() -> ModuleManifest {
        ModuleManifest::new(
            "core.math",
            ModuleOrigin::Builtin,
            CapabilityVersion::new(1, 2, 0),
        )
        .with_granted_permission(Permission::state_read())
        .with_operation(
            OperationSpec::new("integrate", CapabilityVersion::new(1, 2, 0))
                .with_input("dt", PortType::F64)
                .with_output("value", PortType::F64)
                .with_required_permission(Permission::state_read())
                .with_effect(EffectKind::Pure)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        )
    }

    #[test]
    fn unregistered_module_or_operation_returns_explicit_diagnostic() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register_module(sample_builtin_manifest())
            .expect("register builtin");

        let missing_module = catalog.validate_operation(
            &OperationBinding::new("unknown.mod", "op", CapabilityVersion::new(1, 0, 0)),
            TemporalProfile::BoundedTick,
        );
        assert_eq!(
            missing_module,
            Err(ModuleContractError::UnregisteredModule {
                module_id: ModuleId::new("unknown.mod"),
            })
        );

        let missing_op = catalog.validate_operation(
            &OperationBinding::new("core.math", "missing_op", CapabilityVersion::new(1, 0, 0)),
            TemporalProfile::BoundedTick,
        );
        assert_eq!(
            missing_op,
            Err(ModuleContractError::UnregisteredOperation {
                module_id: ModuleId::new("core.math"),
                operation_id: OperationId::new("missing_op"),
            })
        );
    }

    #[test]
    fn incompatible_major_or_lower_minor_version_returns_diagnostic() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register_module(sample_builtin_manifest())
            .expect("register builtin");

        // Major distinto (2.0.0 requerido frente a 1.2.0 disponible).
        let wrong_major = catalog.validate_operation(
            &OperationBinding::new("core.math", "integrate", CapabilityVersion::new(2, 0, 0)),
            TemporalProfile::BoundedTick,
        );
        assert_eq!(
            wrong_major,
            Err(ModuleContractError::IncompatibleVersion {
                module_id: ModuleId::new("core.math"),
                operation_id: OperationId::new("integrate"),
                required: CapabilityVersion::new(2, 0, 0),
                available: CapabilityVersion::new(1, 2, 0),
            })
        );

        // Minor menor al requerido (1.3.0 requerido frente a 1.2.0 disponible).
        let lower_minor = catalog.validate_operation(
            &OperationBinding::new("core.math", "integrate", CapabilityVersion::new(1, 3, 0)),
            TemporalProfile::BoundedTick,
        );
        assert_eq!(
            lower_minor,
            Err(ModuleContractError::IncompatibleVersion {
                module_id: ModuleId::new("core.math"),
                operation_id: OperationId::new("integrate"),
                required: CapabilityVersion::new(1, 3, 0),
                available: CapabilityVersion::new(1, 2, 0),
            })
        );

        // Minor mayor o igual y mismo major (1.1.0 requerido frente a 1.2.0 disponible) es válido.
        assert!(
            catalog
                .validate_operation(
                    &OperationBinding::new(
                        "core.math",
                        "integrate",
                        CapabilityVersion::new(1, 1, 0)
                    ),
                    TemporalProfile::BoundedTick,
                )
                .is_ok()
        );
    }

    #[test]
    fn bounded_temporal_path_rejects_declared_blocking_io() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register_module(sample_builtin_manifest())
            .expect("register builtin");

        let io_module = ModuleManifest::new(
            "ext.storage",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permission(Permission::filesystem())
        .with_granted_permission(Permission::blocking_io())
        .with_operation(
            OperationSpec::new("flush_disk", CapabilityVersion::new(1, 0, 0))
                .with_input("payload", PortType::String)
                .with_required_permission(Permission::filesystem())
                .with_required_permission(Permission::blocking_io())
                .with_effects([EffectKind::Filesystem, EffectKind::BlockingIo])
                .with_temporal_profile(TemporalProfile::UnboundedBackground),
        );
        catalog.register_module(io_module).expect("register io");

        let path = [
            OperationBinding::new("core.math", "integrate", CapabilityVersion::new(1, 0, 0)),
            OperationBinding::new("ext.storage", "flush_disk", CapabilityVersion::new(1, 0, 0)),
        ];

        for bounded_profile in [TemporalProfile::BoundedTick, TemporalProfile::BoundedRt] {
            let err = catalog
                .validate_temporal_path(bounded_profile, &path)
                .expect_err("bounded path must reject BlockingIo");
            assert_eq!(
                err,
                ModuleContractError::BlockingIoInBoundedPath {
                    module_id: ModuleId::new("ext.storage"),
                    operation_id: OperationId::new("flush_disk"),
                    path_profile: bounded_profile,
                    operation_profile: TemporalProfile::UnboundedBackground,
                }
            );
        }

        // En ruta no acotada sí se permite.
        let ok = catalog.validate_temporal_path(TemporalProfile::UnboundedBackground, &path);
        assert!(ok.is_ok());
    }

    #[test]
    fn builtin_and_third_party_share_exact_contract_without_implicit_privilege_escalation() {
        let mut catalog = CapabilityCatalog::new();

        // Builtin con operación que requiere `state:write`, pero el manifiesto solo tiene `state:read`.
        let builtin = ModuleManifest::new(
            "builtin.writer",
            ModuleOrigin::Builtin,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permission(Permission::state_read())
        .with_operation(
            OperationSpec::new("mutate", CapabilityVersion::new(1, 0, 0))
                .with_required_permission(Permission::state_write())
                .with_effect(EffectKind::StateMutation)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        );

        // ThirdParty con exactamente el mismo déficit de permisos.
        let third_party = ModuleManifest::new(
            "plugin.writer",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permission(Permission::state_read())
        .with_operation(
            OperationSpec::new("mutate", CapabilityVersion::new(1, 0, 0))
                .with_required_permission(Permission::state_write())
                .with_effect(EffectKind::StateMutation)
                .with_temporal_profile(TemporalProfile::BoundedTick),
        );

        catalog.register_module(builtin).expect("register builtin");
        catalog
            .register_module(third_party)
            .expect("register third_party");

        let builtin_err = catalog
            .validate_operation(
                &OperationBinding::new("builtin.writer", "mutate", CapabilityVersion::new(1, 0, 0)),
                TemporalProfile::BoundedTick,
            )
            .expect_err("builtin must not elevate permissions implicitly");
        assert_eq!(
            builtin_err,
            ModuleContractError::PermissionDenied {
                module_id: ModuleId::new("builtin.writer"),
                operation_id: OperationId::new("mutate"),
                origin: ModuleOrigin::Builtin,
                missing: PermissionSet::empty().with_permission(Permission::state_write()),
            }
        );

        let third_party_err = catalog
            .validate_operation(
                &OperationBinding::new("plugin.writer", "mutate", CapabilityVersion::new(1, 0, 0)),
                TemporalProfile::BoundedTick,
            )
            .expect_err("third_party must not elevate permissions implicitly");
        assert_eq!(
            third_party_err,
            ModuleContractError::PermissionDenied {
                module_id: ModuleId::new("plugin.writer"),
                operation_id: OperationId::new("mutate"),
                origin: ModuleOrigin::ThirdParty,
                missing: PermissionSet::empty().with_permission(Permission::state_write()),
            }
        );

        // Cuando ThirdParty sí tiene el permiso explícitamente otorgado, valida igual que Builtin.
        let authorized_third_party = ModuleManifest::new(
            "plugin.authorized",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(1, 0, 0),
        )
        .with_granted_permission(Permission::state_write())
        .with_operation(
            OperationSpec::new("mutate", CapabilityVersion::new(1, 0, 0))
                .with_required_permission(Permission::state_write())
                .with_effect(EffectKind::StateMutation)
                .with_temporal_profile(TemporalProfile::BoundedRt),
        );
        catalog
            .register_module(authorized_third_party)
            .expect("register authorized third_party");

        let validated = catalog
            .validate_operation(
                &OperationBinding::new(
                    "plugin.authorized",
                    "mutate",
                    CapabilityVersion::new(1, 0, 0),
                ),
                TemporalProfile::BoundedRt,
            )
            .expect("authorized third_party should validate");
        assert_eq!(validated.origin, ModuleOrigin::ThirdParty);
        assert!(!validated.declared_properties.is_empty());
    }

    #[test]
    fn declared_property_is_distinguished_from_observed_evidence() {
        let mut catalog = CapabilityCatalog::new();
        catalog
            .register_module(sample_builtin_manifest())
            .expect("register builtin");

        let undeclared = ObservedEvidence::EffectObserved {
            module_id: ModuleId::new("core.math"),
            operation_id: OperationId::new("integrate"),
            effect: EffectKind::Network,
        };
        assert_eq!(
            catalog.audit_observed_evidence(&undeclared),
            Err(ModuleContractError::UndeclaredObservedEffect {
                module_id: ModuleId::new("core.math"),
                operation_id: OperationId::new("integrate"),
                observed: EffectKind::Network,
            })
        );

        let declared = ObservedEvidence::EffectObserved {
            module_id: ModuleId::new("core.math"),
            operation_id: OperationId::new("integrate"),
            effect: EffectKind::Pure,
        };
        assert!(catalog.audit_observed_evidence(&declared).is_ok());
    }
}
