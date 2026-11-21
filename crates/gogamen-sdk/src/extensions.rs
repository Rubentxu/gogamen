//! Registro estático de extensiones por capacidad (`std`-only offline).
//!
//! Este módulo permite a consumidores externos e internos registrar **adaptadores**
//! ([`LiveEnvironment`]), **planificadores** ([`DynPlanner`] / [`Planner`]) y **sensores**
//! ([`EnvironmentObserver`]) y **kernels de operación** para `PlanRuntimeEngine` (`register_plan_kernel` + `build_kernel_registry`) en tiempo de compilación/construcción del binario mediante
//! factories tipadas, sin ABI dinámica de Rust, sin `dlopen` y sin descarga o ejecución
//! de plugins remotos.
//!
//! # Garantías de seguridad y contrato
//! - **Validación previa al registro:** [`StaticExtensionRegistry`] verifica esquema de
//!   contrato, compatibilidad SemVer ([`CapabilityVersion`]), capacidades declaradas y
//!   origen estático antes de aceptar cualquier [`ExtensionManifest`].
//! - **Sin descarga ni carga dinámica:** cualquier origen remoto (`http://`, `https://`,
//!   `git://`, etc.) o biblioteca dinámica (`.so`, `.dll`, `.dylib`, `dlopen`) se rechaza
//!   con [`ExtensionError::RemoteSourceRejected`] o [`ExtensionError::DynamicLoadingRejected`].
//! - **Sin elevación implícita de permisos:** declarar capacidades en [`ExtensionManifest`]
//!   no otorga permisos automáticamente. La instanciación exige un [`ExtensionActivationContext`]
//!   con [`PermissionSet`] explícito (y política [`SafeFallbackPolicy`] para [`ActionGuard`]).

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

use gogamen_core::{
    ActionGuard, CapabilityVersion, EnvironmentObserver, GenerativeModel, LiveEnvironment,
    ModelError, ModuleManifest, ModuleOrigin, NoPrecondition, PermissionSet, PlanRequest,
    PlanResult, Planner, PlanningContext, SafeFallbackPolicy,
};
use gogamen_runtime::{ExternalKernel, KernelRegistry};

/// Esquema de contrato por defecto para extensiones estáticas de `gogamen-sdk`.
pub const EXTENSION_CONTRACT_SCHEMA_V1: &str = "gogamen.ext.v1";

/// Versión SemVer de contrato de extensiones soportada por defecto (`0.1.0`).
pub const EXTENSION_HOST_VERSION_V1: CapabilityVersion = CapabilityVersion::new(0, 1, 0);

/// Identificador único de una extensión registrada estáticamente.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExtensionId(String);

impl ExtensionId {
    /// Crea un identificador de extensión a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for ExtensionId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for ExtensionId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for ExtensionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador lógico de una capacidad declarada o requerida por una extensión.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CapabilityId(String);

impl CapabilityId {
    /// Crea un identificador de capacidad a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for CapabilityId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for CapabilityId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for CapabilityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Requisito de capacidad y versión mínima compatible declarada por una extensión.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityRequirement {
    /// Identificador de la capacidad requerida.
    pub id: CapabilityId,
    /// Versión mínima compatible requerida.
    pub min_version: CapabilityVersion,
}

impl CapabilityRequirement {
    /// Crea un requisito de capacidad con su versión mínima compatible.
    pub fn new(id: impl Into<CapabilityId>, min_version: CapabilityVersion) -> Self {
        Self {
            id: id.into(),
            min_version,
        }
    }
}

/// Categoría tipada de extensión soportada por [`StaticExtensionRegistry`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExtensionKind {
    /// Adaptador de entorno vivo ([`LiveEnvironment`]).
    Adapter,
    /// Planificador aislado ([`DynPlanner`] / [`Planner`]).
    Planner,
    /// Sensor de observación de solo lectura ([`EnvironmentObserver`]).
    Sensor,
    /// Kernel de operación para `PlanRuntimeEngine` ([`ExternalKernel`]).
    PlanKernel,
}

impl fmt::Display for ExtensionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adapter => f.write_str("Adapter"),
            Self::Planner => f.write_str("Planner"),
            Self::Sensor => f.write_str("Sensor"),
            Self::PlanKernel => f.write_str("PlanKernel"),
        }
    }
}

/// Origen de enlace de una extensión.
///
/// Únicamente [`ExtensionSource::StaticLinked`] es aceptado por [`StaticExtensionRegistry`];
/// cualquier origen remoto o biblioteca dinámica es rechazado explícitamente al registrar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ExtensionSource {
    /// Extensión compilada y enlazada estáticamente en el binario anfitrión.
    StaticLinked {
        /// Nombre del crate o módulo Rust enlazado estáticamente.
        crate_name: String,
    },
    /// Origen remoto explícitamente prohibido (URL `http://`, `https://`, `git://`, etc.).
    RemoteUri(String),
    /// Biblioteca dinámica explícitamente prohibida (`.so`, `.dll`, `.dylib`, `dlopen`).
    DynamicLibrary(String),
}

impl ExtensionSource {
    /// Crea un origen estático enlazado en el binario.
    pub fn static_linked(crate_name: impl Into<String>) -> Self {
        Self::StaticLinked {
            crate_name: crate_name.into(),
        }
    }

    /// Clasifica una cadena de origen/localizador detectando URLs remotas o carga dinámica.
    pub fn from_locator(locator: impl Into<String>) -> Self {
        let raw = locator.into();
        if looks_like_remote_uri(&raw) {
            Self::RemoteUri(raw)
        } else if looks_like_dynamic_library(&raw) {
            Self::DynamicLibrary(raw)
        } else {
            Self::StaticLinked { crate_name: raw }
        }
    }
}

fn looks_like_remote_uri(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    const REMOTE_PREFIXES: [&str; 9] = [
        "http://", "https://", "ftp://", "ftps://", "ssh://", "git://", "ws://", "wss://",
        "file://",
    ];
    REMOTE_PREFIXES
        .iter()
        .any(|prefix| lower.starts_with(prefix))
        || lower.contains("://")
}

fn looks_like_dynamic_library(value: &str) -> bool {
    let lower = value.trim().to_ascii_lowercase();
    lower.ends_with(".so")
        || lower.ends_with(".dll")
        || lower.ends_with(".dylib")
        || lower.contains(".so.")
        || lower.contains("dlopen")
}

/// Manifiesto declarativo de una extensión estática.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtensionManifest {
    /// Identificador único de la extensión.
    pub id: ExtensionId,
    /// Categoría de extensión (`Adapter`, `Planner` o `Sensor`).
    pub kind: ExtensionKind,
    /// Origen del módulo (`Builtin` o `ThirdParty`).
    pub origin: ModuleOrigin,
    /// Origen de enlace (solo [`ExtensionSource::StaticLinked`] es válido).
    pub source: ExtensionSource,
    /// Versión propia de la extensión.
    pub version: CapabilityVersion,
    /// Versión mínima de contrato SDK requerida por la extensión.
    pub required_sdk_version: CapabilityVersion,
    /// Identificador del esquema de contrato requerido.
    pub contract_schema: String,
    /// Capacidades declaradas/requeridas que el anfitrión debe soportar.
    pub declared_capabilities: Vec<CapabilityRequirement>,
    /// Permisos explícitos requeridos al instanciar (declararlos aquí no los otorga).
    pub required_permissions: PermissionSet,
}

impl ExtensionManifest {
    /// Crea un manifiesto base para una extensión enlazada estáticamente, sin permisos
    /// implícitos ni capacidades declaradas todavía.
    pub fn new(
        id: impl Into<ExtensionId>,
        kind: ExtensionKind,
        origin: ModuleOrigin,
        crate_name: impl Into<String>,
        version: CapabilityVersion,
    ) -> Self {
        Self {
            id: id.into(),
            kind,
            origin,
            source: ExtensionSource::static_linked(crate_name),
            version,
            required_sdk_version: EXTENSION_HOST_VERSION_V1,
            contract_schema: EXTENSION_CONTRACT_SCHEMA_V1.to_owned(),
            declared_capabilities: Vec::new(),
            required_permissions: PermissionSet::empty(),
        }
    }

    /// Establece explícitamente el origen de enlace (estilo builder).
    pub fn with_source(mut self, source: ExtensionSource) -> Self {
        self.source = source;
        self
    }

    /// Establece la versión mínima de contrato SDK requerida (estilo builder).
    pub fn with_required_sdk_version(mut self, version: CapabilityVersion) -> Self {
        self.required_sdk_version = version;
        self
    }

    /// Establece el esquema de contrato requerido (estilo builder).
    pub fn with_contract_schema(mut self, schema: impl Into<String>) -> Self {
        self.contract_schema = schema.into();
        self
    }

    /// Añade un requisito de capacidad declarada (estilo builder).
    pub fn with_capability(
        mut self,
        capability_id: impl Into<CapabilityId>,
        min_version: CapabilityVersion,
    ) -> Self {
        self.declared_capabilities
            .push(CapabilityRequirement::new(capability_id, min_version));
        self
    }

    /// Añade un permiso requerido para activar la extensión (estilo builder).
    pub fn with_required_permission(
        mut self,
        permission: impl Into<gogamen_core::Permission>,
    ) -> Self {
        self.required_permissions.insert(permission);
        self
    }

    /// Establece el conjunto de permisos requeridos para activar la extensión (estilo builder).
    pub fn with_required_permissions(mut self, permissions: PermissionSet) -> Self {
        self.required_permissions = permissions;
        self
    }
}

/// Trait objetizable para ejecutar cualquier [`Planner`] instanciado desde una factory estática.
pub trait DynPlanner {
    /// Planifica una acción para `request` operando exclusivamente sobre `ctx`.
    fn plan_dyn(
        &mut self,
        ctx: &mut PlanningContext<'_, dyn GenerativeModel>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError>;
}

impl<P: Planner> DynPlanner for P {
    fn plan_dyn(
        &mut self,
        ctx: &mut PlanningContext<'_, dyn GenerativeModel>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        self.plan(ctx, request)
    }
}

/// Contexto explícito de activación para instanciar una extensión registrada.
///
/// Separa la declaración de capacidades del otorgamiento real de permisos y de la
/// política de [`ActionGuard`], impidiendo cualquier elevación implícita de privilegios.
#[derive(Debug, Clone)]
pub struct ExtensionActivationContext {
    /// Permisos explícitamente otorgados por el anfitrión para esta instanciación.
    pub granted_permissions: PermissionSet,
    /// Política de respaldo explícita para construir un [`ActionGuard`].
    pub guard_policy: SafeFallbackPolicy,
}

impl ExtensionActivationContext {
    /// Crea un contexto de activación con permisos vacíos y guardia estricta (`RejectOnly`).
    pub fn strict_empty() -> Self {
        Self {
            granted_permissions: PermissionSet::empty(),
            guard_policy: SafeFallbackPolicy::RejectOnly,
        }
    }

    /// Crea un contexto de activación con los permisos explícitos indicados y guardia estricta.
    pub fn with_permissions(granted_permissions: PermissionSet) -> Self {
        Self {
            granted_permissions,
            guard_policy: SafeFallbackPolicy::RejectOnly,
        }
    }

    /// Establece la política de respaldo para [`ActionGuard`] (estilo builder).
    pub fn with_guard_policy(mut self, guard_policy: SafeFallbackPolicy) -> Self {
        self.guard_policy = guard_policy;
        self
    }

    /// Construye un [`ActionGuard`] con la política configurada en este contexto.
    pub fn action_guard(&self) -> ActionGuard<NoPrecondition> {
        ActionGuard::new(self.guard_policy.clone())
    }
}

/// Factory tipada para instanciar un adaptador [`LiveEnvironment`].
pub type AdapterFactory = Arc<
    dyn Fn(&ExtensionActivationContext) -> Result<Box<dyn LiveEnvironment>, ExtensionError>
        + Send
        + Sync,
>;

/// Factory tipada para instanciar un planificador [`DynPlanner`].
pub type PlannerFactory = Arc<
    dyn Fn(&ExtensionActivationContext) -> Result<Box<dyn DynPlanner>, ExtensionError>
        + Send
        + Sync,
>;

/// Factory tipada para instanciar un sensor [`EnvironmentObserver`].
pub type SensorFactory = Arc<
    dyn Fn(&ExtensionActivationContext) -> Result<Box<dyn EnvironmentObserver>, ExtensionError>
        + Send
        + Sync,
>;

/// Factory tipada para instanciar un [`ExternalKernel`] de `PlanRuntimeEngine`.
pub type PlanKernelFactory = Arc<
    dyn Fn(&ExtensionActivationContext) -> Result<Arc<dyn ExternalKernel>, ExtensionError>
        + Send
        + Sync,
>;

/// Errores tipados del registro estático de extensiones por capacidad.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExtensionError {
    /// El manifiesto contiene campos vacíos o estructuralmente inválidos.
    InvalidManifest {
        /// Identificador de la extensión afectada.
        id: ExtensionId,
        /// Motivo de invalidez.
        reason: String,
    },
    /// Ya existe una extensión registrada con el mismo identificador.
    DuplicateExtensionId {
        /// Identificador duplicado.
        id: ExtensionId,
    },
    /// La extensión solicitada no existe en el registro.
    ExtensionNotFound {
        /// Identificador ausente.
        id: ExtensionId,
    },
    /// Se rechazó un origen remoto o intento de descarga de plugin.
    RemoteSourceRejected {
        /// Identificador de la extensión rechazada.
        id: ExtensionId,
        /// Origen remoto detectado.
        source: String,
    },
    /// Se rechazó un intento de carga dinámica (`dlopen`, `.so`, `.dll`, `.dylib`).
    DynamicLoadingRejected {
        /// Identificador de la extensión rechazada.
        id: ExtensionId,
        /// Ruta o referencia dinámica detectada.
        source: String,
    },
    /// El esquema de contrato declarado es incompatible con el anfitrión.
    IncompatibleSchema {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Esquema esperado por el anfitrión.
        expected: String,
        /// Esquema declarado por la extensión.
        actual: String,
    },
    /// Una versión de capacidad o de contrato SDK no es compatible.
    IncompatibleVersion {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Capacidad o contrato en conflicto.
        capability: CapabilityId,
        /// Versión mínima requerida por la extensión.
        required: CapabilityVersion,
        /// Versión disponible en el anfitrión.
        available: CapabilityVersion,
    },
    /// La extensión declara una capacidad no soportada por el registro anfitrión.
    UnsupportedCapability {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Capacidad no soportada.
        capability: CapabilityId,
    },
    /// La categoría del manifiesto no coincide con la factory registrada o solicitada.
    KindMismatch {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Categoría esperada.
        expected: ExtensionKind,
        /// Categoría encontrada.
        actual: ExtensionKind,
    },
    /// Faltan permisos explícitos en [`ExtensionActivationContext`] para instanciar.
    InsufficientPermissions {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Subconjunto de permisos faltantes.
        missing: PermissionSet,
    },
    /// La factory de la extensión falló durante la instanciación.
    InstantiationFailed {
        /// Identificador de la extensión.
        id: ExtensionId,
        /// Detalle del fallo.
        reason: String,
    },
}

impl fmt::Display for ExtensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidManifest { id, reason } => {
                write!(f, "invalid extension manifest `{id}`: {reason}")
            }
            Self::DuplicateExtensionId { id } => {
                write!(f, "duplicate extension id `{id}`")
            }
            Self::ExtensionNotFound { id } => {
                write!(f, "extension `{id}` is not registered")
            }
            Self::RemoteSourceRejected { id, source } => {
                write!(
                    f,
                    "remote extension source rejected for `{id}` (`{source}`): remote plugins are forbidden"
                )
            }
            Self::DynamicLoadingRejected { id, source } => {
                write!(
                    f,
                    "dynamic loading rejected for `{id}` (`{source}`): dlopen/dynamic plugins are forbidden"
                )
            }
            Self::IncompatibleSchema {
                id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "incompatible contract schema for `{id}`: expected `{expected}`, got `{actual}`"
                )
            }
            Self::IncompatibleVersion {
                id,
                capability,
                required,
                available,
            } => {
                write!(
                    f,
                    "incompatible version for `{id}` on `{capability}`: requires {required}, host provides {available}"
                )
            }
            Self::UnsupportedCapability { id, capability } => {
                write!(
                    f,
                    "unsupported capability `{capability}` declared by extension `{id}`"
                )
            }
            Self::KindMismatch {
                id,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "extension kind mismatch for `{id}`: expected {expected}, got {actual}"
                )
            }
            Self::InsufficientPermissions { id, missing } => {
                write!(
                    f,
                    "insufficient explicit permissions to instantiate `{id}`: missing {missing}"
                )
            }
            Self::InstantiationFailed { id, reason } => {
                write!(f, "failed to instantiate extension `{id}`: {reason}")
            }
        }
    }
}

impl Error for ExtensionError {}

#[derive(Clone)]
enum RegisteredFactory {
    Adapter(AdapterFactory),
    Planner(PlannerFactory),
    Sensor(SensorFactory),
    PlanKernel {
        module: Box<ModuleManifest>,
        operation_id: String,
        factory: PlanKernelFactory,
    },
}

impl RegisteredFactory {
    fn kind(&self) -> ExtensionKind {
        match self {
            Self::Adapter(_) => ExtensionKind::Adapter,
            Self::Planner(_) => ExtensionKind::Planner,
            Self::Sensor(_) => ExtensionKind::Sensor,
            Self::PlanKernel { .. } => ExtensionKind::PlanKernel,
        }
    }
}

#[derive(Clone)]
struct RegisteredExtension {
    manifest: ExtensionManifest,
    factory: RegisteredFactory,
}

/// Registro estático de extensiones por capacidad para adaptadores, planificadores y sensores.
///
/// Todas las extensiones se enlazan estáticamente al construir el binario anfitrión y se
/// validan antes de quedar disponibles para instanciación.
#[derive(Clone)]
pub struct StaticExtensionRegistry {
    contract_schema: String,
    host_version: CapabilityVersion,
    supported_capabilities: BTreeMap<CapabilityId, CapabilityVersion>,
    extensions: BTreeMap<ExtensionId, RegisteredExtension>,
}

impl fmt::Debug for StaticExtensionRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticExtensionRegistry")
            .field("contract_schema", &self.contract_schema)
            .field("host_version", &self.host_version)
            .field("supported_capabilities", &self.supported_capabilities)
            .field(
                "registered_extensions",
                &self.extensions.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for StaticExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticExtensionRegistry {
    /// Crea un registro estático con el esquema y versión por defecto (`gogamen.ext.v1`, `0.1.0`).
    pub fn new() -> Self {
        Self::with_host_contract(EXTENSION_CONTRACT_SCHEMA_V1, EXTENSION_HOST_VERSION_V1)
    }

    /// Crea un registro estático especificando el esquema y versión de contrato del anfitrión.
    pub fn with_host_contract(
        contract_schema: impl Into<String>,
        host_version: CapabilityVersion,
    ) -> Self {
        Self {
            contract_schema: contract_schema.into(),
            host_version,
            supported_capabilities: BTreeMap::new(),
            extensions: BTreeMap::new(),
        }
    }

    /// Declara una capacidad soportada por el anfitrión junto con su versión disponible.
    pub fn with_supported_capability(
        mut self,
        capability_id: impl Into<CapabilityId>,
        version: CapabilityVersion,
    ) -> Self {
        self.supported_capabilities
            .insert(capability_id.into(), version);
        self
    }

    /// Registra o actualiza una capacidad soportada por el anfitrión.
    pub fn register_supported_capability(
        &mut self,
        capability_id: impl Into<CapabilityId>,
        version: CapabilityVersion,
    ) {
        self.supported_capabilities
            .insert(capability_id.into(), version);
    }

    /// Devuelve el esquema de contrato configurado en este registro.
    pub fn contract_schema(&self) -> &str {
        &self.contract_schema
    }

    /// Devuelve la versión de contrato SDK del anfitrión.
    pub const fn host_version(&self) -> CapabilityVersion {
        self.host_version
    }

    /// Devuelve el mapa ordenado de capacidades soportadas por el anfitrión.
    pub fn supported_capabilities(&self) -> &BTreeMap<CapabilityId, CapabilityVersion> {
        &self.supported_capabilities
    }

    /// Valida compatibilidad, origen estático y unicidad de un [`ExtensionManifest`] antes
    /// de registrarlo.
    pub fn validate_manifest(
        &self,
        manifest: &ExtensionManifest,
        expected_kind: ExtensionKind,
    ) -> Result<(), ExtensionError> {
        if manifest.id.as_str().trim().is_empty() {
            return Err(ExtensionError::InvalidManifest {
                id: manifest.id.clone(),
                reason: "extension id must not be empty".to_owned(),
            });
        }

        if self.extensions.contains_key(&manifest.id) {
            return Err(ExtensionError::DuplicateExtensionId {
                id: manifest.id.clone(),
            });
        }

        if manifest.kind != expected_kind {
            return Err(ExtensionError::KindMismatch {
                id: manifest.id.clone(),
                expected: expected_kind,
                actual: manifest.kind,
            });
        }

        match &manifest.source {
            ExtensionSource::RemoteUri(uri) => {
                return Err(ExtensionError::RemoteSourceRejected {
                    id: manifest.id.clone(),
                    source: uri.clone(),
                });
            }
            ExtensionSource::DynamicLibrary(path) => {
                return Err(ExtensionError::DynamicLoadingRejected {
                    id: manifest.id.clone(),
                    source: path.clone(),
                });
            }
            ExtensionSource::StaticLinked { crate_name } => {
                if crate_name.trim().is_empty() {
                    return Err(ExtensionError::InvalidManifest {
                        id: manifest.id.clone(),
                        reason: "static crate_name must not be empty".to_owned(),
                    });
                }
                if looks_like_remote_uri(crate_name) {
                    return Err(ExtensionError::RemoteSourceRejected {
                        id: manifest.id.clone(),
                        source: crate_name.clone(),
                    });
                }
                if looks_like_dynamic_library(crate_name) {
                    return Err(ExtensionError::DynamicLoadingRejected {
                        id: manifest.id.clone(),
                        source: crate_name.clone(),
                    });
                }
            }
        }

        if manifest.contract_schema != self.contract_schema {
            return Err(ExtensionError::IncompatibleSchema {
                id: manifest.id.clone(),
                expected: self.contract_schema.clone(),
                actual: manifest.contract_schema.clone(),
            });
        }

        if !self
            .host_version
            .is_compatible_with(&manifest.required_sdk_version)
        {
            return Err(ExtensionError::IncompatibleVersion {
                id: manifest.id.clone(),
                capability: CapabilityId::new("sdk.contract"),
                required: manifest.required_sdk_version,
                available: self.host_version,
            });
        }

        for req in &manifest.declared_capabilities {
            let Some(&available) = self.supported_capabilities.get(&req.id) else {
                return Err(ExtensionError::UnsupportedCapability {
                    id: manifest.id.clone(),
                    capability: req.id.clone(),
                });
            };
            if !available.is_compatible_with(&req.min_version) {
                return Err(ExtensionError::IncompatibleVersion {
                    id: manifest.id.clone(),
                    capability: req.id.clone(),
                    required: req.min_version,
                    available,
                });
            }
        }

        Ok(())
    }

    /// Registra un adaptador [`LiveEnvironment`] estático tras validar su manifiesto.
    pub fn register_adapter<F>(
        &mut self,
        manifest: ExtensionManifest,
        factory: F,
    ) -> Result<(), ExtensionError>
    where
        F: Fn(&ExtensionActivationContext) -> Result<Box<dyn LiveEnvironment>, ExtensionError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_manifest(&manifest, ExtensionKind::Adapter)?;
        let id = manifest.id.clone();
        self.extensions.insert(
            id,
            RegisteredExtension {
                manifest,
                factory: RegisteredFactory::Adapter(Arc::new(factory)),
            },
        );
        Ok(())
    }

    /// Registra un planificador [`DynPlanner`] estático tras validar su manifiesto.
    pub fn register_planner<F>(
        &mut self,
        manifest: ExtensionManifest,
        factory: F,
    ) -> Result<(), ExtensionError>
    where
        F: Fn(&ExtensionActivationContext) -> Result<Box<dyn DynPlanner>, ExtensionError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_manifest(&manifest, ExtensionKind::Planner)?;
        let id = manifest.id.clone();
        self.extensions.insert(
            id,
            RegisteredExtension {
                manifest,
                factory: RegisteredFactory::Planner(Arc::new(factory)),
            },
        );
        Ok(())
    }

    /// Registra un sensor [`EnvironmentObserver`] estático tras validar su manifiesto.
    pub fn register_sensor<F>(
        &mut self,
        manifest: ExtensionManifest,
        factory: F,
    ) -> Result<(), ExtensionError>
    where
        F: Fn(&ExtensionActivationContext) -> Result<Box<dyn EnvironmentObserver>, ExtensionError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_manifest(&manifest, ExtensionKind::Sensor)?;
        let id = manifest.id.clone();
        self.extensions.insert(
            id,
            RegisteredExtension {
                manifest,
                factory: RegisteredFactory::Sensor(Arc::new(factory)),
            },
        );
        Ok(())
    }

    /// Registra un kernel de operación para `PlanRuntimeEngine`.
    ///
    /// `module` es el [`ModuleManifest`] (contrato del catálogo de capacidades) que declara
    /// `operation_id`. Varias extensiones pueden aportar operaciones del mismo módulo.
    pub fn register_plan_kernel<F>(
        &mut self,
        manifest: ExtensionManifest,
        module: ModuleManifest,
        operation_id: impl Into<String>,
        factory: F,
    ) -> Result<(), ExtensionError>
    where
        F: Fn(&ExtensionActivationContext) -> Result<Arc<dyn ExternalKernel>, ExtensionError>
            + Send
            + Sync
            + 'static,
    {
        self.validate_manifest(&manifest, ExtensionKind::PlanKernel)?;
        let operation_id = operation_id.into();
        if !module
            .operations
            .iter()
            .any(|op| op.id.as_str() == operation_id)
        {
            return Err(ExtensionError::InvalidManifest {
                id: manifest.id.clone(),
                reason: format!(
                    "module '{}' does not declare operation '{operation_id}'",
                    module.id.as_str()
                ),
            });
        }
        let id = manifest.id.clone();
        self.extensions.insert(
            id,
            RegisteredExtension {
                manifest,
                factory: RegisteredFactory::PlanKernel {
                    module: Box::new(module),
                    operation_id,
                    factory: Arc::new(factory),
                },
            },
        );
        Ok(())
    }

    /// Construye el [`KernelRegistry`] de `PlanRuntimeEngine` a partir de todas las
    /// extensiones [`ExtensionKind::PlanKernel`] registradas (orden determinista por id),
    /// validando permisos explícitos de `ctx` para cada una.
    pub fn build_kernel_registry(
        &self,
        ctx: &ExtensionActivationContext,
    ) -> Result<KernelRegistry, ExtensionError> {
        let mut modules: BTreeMap<String, ModuleManifest> = BTreeMap::new();
        let mut bindings: Vec<(ExtensionId, String, String, Arc<dyn ExternalKernel>)> = Vec::new();

        for id in self.registered_ids_by_kind(ExtensionKind::PlanKernel) {
            let entry = self.lookup_and_authorize(&id, ExtensionKind::PlanKernel, ctx)?;
            let RegisteredFactory::PlanKernel {
                module,
                operation_id,
                factory,
            } = &entry.factory
            else {
                continue;
            };
            let kernel = factory(ctx)?;
            let module_id = module.id.as_str().to_owned();
            match modules.get_mut(&module_id) {
                Some(existing) => {
                    for op in &module.operations {
                        if !existing.operations.iter().any(|o| o.id == op.id) {
                            existing.operations.push(op.clone());
                        }
                    }
                }
                None => {
                    modules.insert(module_id.clone(), (**module).clone());
                }
            }
            bindings.push((id, module_id, operation_id.clone(), kernel));
        }

        let mut registry = KernelRegistry::new();
        for manifest in modules.into_values() {
            let id = bindings
                .iter()
                .find(|(_, m, _, _)| m == manifest.id.as_str())
                .map(|(id, ..)| id.clone())
                .expect("module originates from a binding");
            registry.register_module(manifest).map_err(|err| {
                ExtensionError::InstantiationFailed {
                    id,
                    reason: err.to_string(),
                }
            })?;
        }
        for (id, module_id, operation_id, kernel) in bindings {
            registry
                .bind_kernel(&module_id, &operation_id, kernel)
                .map_err(|err| ExtensionError::InstantiationFailed {
                    id,
                    reason: err.to_string(),
                })?;
        }
        Ok(registry)
    }

    /// Devuelve el manifiesto de una extensión registrada si existe.
    pub fn manifest(&self, id: &ExtensionId) -> Option<&ExtensionManifest> {
        self.extensions.get(id).map(|entry| &entry.manifest)
    }

    /// Lista los identificadores de todas las extensiones registradas en orden determinista.
    pub fn registered_ids(&self) -> Vec<ExtensionId> {
        self.extensions.keys().cloned().collect()
    }

    /// Lista los identificadores de extensiones registradas de una categoría concreta.
    pub fn registered_ids_by_kind(&self, kind: ExtensionKind) -> Vec<ExtensionId> {
        self.extensions
            .iter()
            .filter(|(_, entry)| entry.factory.kind() == kind)
            .map(|(id, _)| id.clone())
            .collect()
    }

    fn lookup_and_authorize(
        &self,
        id: &ExtensionId,
        expected_kind: ExtensionKind,
        ctx: &ExtensionActivationContext,
    ) -> Result<&RegisteredExtension, ExtensionError> {
        let entry = self
            .extensions
            .get(id)
            .ok_or_else(|| ExtensionError::ExtensionNotFound { id: id.clone() })?;

        let actual_kind = entry.factory.kind();
        if actual_kind != expected_kind {
            return Err(ExtensionError::KindMismatch {
                id: id.clone(),
                expected: expected_kind,
                actual: actual_kind,
            });
        }

        if !entry
            .manifest
            .required_permissions
            .is_subset_of(&ctx.granted_permissions)
        {
            let missing = entry
                .manifest
                .required_permissions
                .missing_from(&ctx.granted_permissions);
            return Err(ExtensionError::InsufficientPermissions {
                id: id.clone(),
                missing,
            });
        }

        Ok(entry)
    }

    /// Instancia un adaptador [`LiveEnvironment`] validando permisos explícitos en `ctx`.
    pub fn instantiate_adapter(
        &self,
        id: &ExtensionId,
        ctx: &ExtensionActivationContext,
    ) -> Result<Box<dyn LiveEnvironment>, ExtensionError> {
        let entry = self.lookup_and_authorize(id, ExtensionKind::Adapter, ctx)?;
        match &entry.factory {
            RegisteredFactory::Adapter(factory) => factory(ctx),
            other => Err(ExtensionError::KindMismatch {
                id: id.clone(),
                expected: ExtensionKind::Adapter,
                actual: other.kind(),
            }),
        }
    }

    /// Instancia un planificador [`DynPlanner`] validando permisos explícitos en `ctx`.
    pub fn instantiate_planner(
        &self,
        id: &ExtensionId,
        ctx: &ExtensionActivationContext,
    ) -> Result<Box<dyn DynPlanner>, ExtensionError> {
        let entry = self.lookup_and_authorize(id, ExtensionKind::Planner, ctx)?;
        match &entry.factory {
            RegisteredFactory::Planner(factory) => factory(ctx),
            other => Err(ExtensionError::KindMismatch {
                id: id.clone(),
                expected: ExtensionKind::Planner,
                actual: other.kind(),
            }),
        }
    }

    /// Instancia un sensor [`EnvironmentObserver`] validando permisos explícitos en `ctx`.
    pub fn instantiate_sensor(
        &self,
        id: &ExtensionId,
        ctx: &ExtensionActivationContext,
    ) -> Result<Box<dyn EnvironmentObserver>, ExtensionError> {
        let entry = self.lookup_and_authorize(id, ExtensionKind::Sensor, ctx)?;
        match &entry.factory {
            RegisteredFactory::Sensor(factory) => factory(ctx),
            other => Err(ExtensionError::KindMismatch {
                id: id.clone(),
                expected: ExtensionKind::Sensor,
                actual: other.kind(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::{
        Action, ActionSpace, Actuator, ApplyOutcome, EnvironmentError, Observation, Permission,
        SimDuration, StepDt, Terminal, Tick,
    };

    struct DummySensor;
    impl EnvironmentObserver for DummySensor {
        fn observe(&self) -> Result<Observation, EnvironmentError> {
            Observation::new(Tick::ZERO, SimDuration::ZERO, vec![1.0]).map_err(|_| {
                EnvironmentError::ObservationFailed {
                    reason: "dummy sensor failed",
                }
            })
        }
    }

    struct DummyAdapter {
        space: ActionSpace,
    }
    impl EnvironmentObserver for DummyAdapter {
        fn observe(&self) -> Result<Observation, EnvironmentError> {
            Observation::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).map_err(|_| {
                EnvironmentError::ObservationFailed {
                    reason: "dummy adapter observe failed",
                }
            })
        }
    }
    impl Actuator for DummyAdapter {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
            self.space
                .validate(action)
                .map_err(EnvironmentError::InvalidAction)?;
            let obs = self.observe()?;
            Ok(ApplyOutcome::new(obs, Terminal::Ongoing))
        }
    }
    impl LiveEnvironment for DummyAdapter {
        fn reset(&mut self) -> Result<Observation, EnvironmentError> {
            self.observe()
        }
    }

    struct DummyPlanner;
    impl Planner for DummyPlanner {
        fn plan<M: GenerativeModel + ?Sized>(
            &mut self,
            ctx: &mut PlanningContext<'_, M>,
            _request: &PlanRequest,
        ) -> Result<PlanResult, ModelError> {
            let action = Action::discrete_in_space(0, ctx.model().action_space())
                .map_err(ModelError::TransitionFailed)?;
            Ok(PlanResult::completed(action, 1))
        }
    }

    fn base_registry() -> StaticExtensionRegistry {
        StaticExtensionRegistry::new()
            .with_supported_capability("sensor.scalar", CapabilityVersion::new(0, 1, 0))
            .with_supported_capability("planner.discrete", CapabilityVersion::new(0, 1, 0))
            .with_supported_capability("adapter.live", CapabilityVersion::new(0, 1, 0))
    }

    #[test]
    fn rejects_incompatible_version_schema_unsupported_capability_and_duplicates() {
        let mut registry = base_registry();

        let bad_sdk_version = ExtensionManifest::new(
            "ext.bad_sdk",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "third_party_crate",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_required_sdk_version(CapabilityVersion::new(0, 2, 0));

        assert_eq!(
            registry.register_sensor(bad_sdk_version, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::IncompatibleVersion {
                id: ExtensionId::new("ext.bad_sdk"),
                capability: CapabilityId::new("sdk.contract"),
                required: CapabilityVersion::new(0, 2, 0),
                available: EXTENSION_HOST_VERSION_V1,
            })
        );

        let bad_cap_version = ExtensionManifest::new(
            "ext.bad_cap_ver",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "third_party_crate",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_capability("sensor.scalar", CapabilityVersion::new(1, 0, 0));

        assert_eq!(
            registry.register_sensor(bad_cap_version, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::IncompatibleVersion {
                id: ExtensionId::new("ext.bad_cap_ver"),
                capability: CapabilityId::new("sensor.scalar"),
                required: CapabilityVersion::new(1, 0, 0),
                available: CapabilityVersion::new(0, 1, 0),
            })
        );

        let unsupported_cap = ExtensionManifest::new(
            "ext.unsupported",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "third_party_crate",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_capability("sensor.quantum", CapabilityVersion::new(0, 1, 0));

        assert_eq!(
            registry.register_sensor(unsupported_cap, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::UnsupportedCapability {
                id: ExtensionId::new("ext.unsupported"),
                capability: CapabilityId::new("sensor.quantum"),
            })
        );

        let bad_schema = ExtensionManifest::new(
            "ext.bad_schema",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "third_party_crate",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_contract_schema("gogamen.ext.v99");

        assert_eq!(
            registry.register_sensor(bad_schema, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::IncompatibleSchema {
                id: ExtensionId::new("ext.bad_schema"),
                expected: EXTENSION_CONTRACT_SCHEMA_V1.to_owned(),
                actual: "gogamen.ext.v99".to_owned(),
            })
        );

        let valid = ExtensionManifest::new(
            "ext.sensor_ok",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "third_party_crate",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_capability("sensor.scalar", CapabilityVersion::new(0, 1, 0));

        assert!(
            registry
                .register_sensor(valid.clone(), |_| Ok(Box::new(DummySensor)))
                .is_ok()
        );
        assert_eq!(
            registry.register_sensor(valid, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::DuplicateExtensionId {
                id: ExtensionId::new("ext.sensor_ok"),
            })
        );
    }

    #[test]
    fn rejects_remote_sources_and_dynamic_loading() {
        let mut registry = base_registry();

        let remote_manifest = ExtensionManifest::new(
            "ext.remote",
            ExtensionKind::Planner,
            ModuleOrigin::ThirdParty,
            "unused",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_source(ExtensionSource::from_locator(
            "https://plugins.example.com/p.wasm",
        ));

        assert_eq!(
            registry.register_planner(remote_manifest, |_| Ok(Box::new(DummyPlanner))),
            Err(ExtensionError::RemoteSourceRejected {
                id: ExtensionId::new("ext.remote"),
                source: "https://plugins.example.com/p.wasm".to_owned(),
            })
        );

        let dlopen_manifest = ExtensionManifest::new(
            "ext.dlopen",
            ExtensionKind::Adapter,
            ModuleOrigin::ThirdParty,
            "unused",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_source(ExtensionSource::from_locator("/tmp/libplugin.so"));

        assert_eq!(
            registry.register_adapter(dlopen_manifest, |_| {
                Ok(Box::new(DummyAdapter {
                    space: ActionSpace::discrete(2).expect("valid"),
                }))
            }),
            Err(ExtensionError::DynamicLoadingRejected {
                id: ExtensionId::new("ext.dlopen"),
                source: "/tmp/libplugin.so".to_owned(),
            })
        );

        let disguised_remote = ExtensionManifest::new(
            "ext.disguised",
            ExtensionKind::Sensor,
            ModuleOrigin::ThirdParty,
            "git://example.com/plugin.git",
            CapabilityVersion::new(0, 1, 0),
        );

        assert!(matches!(
            registry.register_sensor(disguised_remote, |_| Ok(Box::new(DummySensor))),
            Err(ExtensionError::RemoteSourceRejected { .. })
        ));
    }

    #[test]
    fn declared_capabilities_do_not_grant_implicit_permissions() {
        let mut registry = base_registry();

        let manifest = ExtensionManifest::new(
            "ext.guarded_adapter",
            ExtensionKind::Adapter,
            ModuleOrigin::ThirdParty,
            "third_party_adapter",
            CapabilityVersion::new(0, 1, 0),
        )
        .with_capability("adapter.live", CapabilityVersion::new(0, 1, 0))
        .with_required_permission(Permission::state_read())
        .with_required_permission(Permission::state_write());

        registry
            .register_adapter(manifest, |_| {
                Ok(Box::new(DummyAdapter {
                    space: ActionSpace::discrete(2).expect("valid"),
                }))
            })
            .expect("valid static adapter");

        let empty_ctx = ExtensionActivationContext::strict_empty();
        let err = registry
            .instantiate_adapter(&ExtensionId::new("ext.guarded_adapter"), &empty_ctx)
            .err()
            .expect("empty permissions must be rejected");

        assert_eq!(
            err,
            ExtensionError::InsufficientPermissions {
                id: ExtensionId::new("ext.guarded_adapter"),
                missing: PermissionSet::from_permissions([
                    Permission::state_read(),
                    Permission::state_write(),
                ]),
            }
        );

        let granted_ctx =
            ExtensionActivationContext::with_permissions(PermissionSet::from_permissions([
                Permission::state_read(),
                Permission::state_write(),
            ]));
        let mut adapter = registry
            .instantiate_adapter(&ExtensionId::new("ext.guarded_adapter"), &granted_ctx)
            .expect("explicitly granted permissions allow instantiation");
        let obs = adapter.reset().expect("reset succeeds");
        assert_eq!(obs.features(), &[0.0]);
        let _dt = StepDt::from_secs_f64(0.1).expect("valid dt");
    }
}
