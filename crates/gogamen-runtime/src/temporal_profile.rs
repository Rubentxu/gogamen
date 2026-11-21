//! Validación de perfil temporal previa a `start` y buffers preasignados para ruta crítica (`std`-only).
//!
//! Este módulo enlaza los perfiles de ejecución del runtime ([`RuntimeTemporalProfile`])
//! con el contrato de capacidades de [`gogamen_core::capability`] y separa de forma explícita:
//! - **Propiedades declaradas** ([`DeclaredCapabilityClaim`]): afirmaciones previas a la
//!   ejecución sobre efectos bloqueantes, bloqueos de sincronización y acotamiento de memoria.
//! - **Evidencia observada** ([`ObservedAllocationProof`]): mediciones reales de asignaciones
//!   en heap durante `step` tras warmup con límites y capacidades registradas.
//!
//! # Advertencia de no certificación en tiempo real estricto
//!
//! Ninguna validación de perfil temporal ni prueba de cero asignaciones en heap en este
//! módulo certifica tiempo real estricto (hard real-time). Consulte
//! [`STRICT_REALTIME_NON_CERTIFICATION_NOTICE`].

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::num::NonZeroUsize;

use gogamen_core::{
    CapabilityCatalog, CapabilityVersion, DeclaredProperty, EffectKind, ModuleContractError,
    ModuleId, OperationBinding, OperationId, PermissionSet, TemporalProfile, ValidatedOperation,
};

/// Aviso explícito de que la validación temporal y las pruebas de cero asignaciones
/// no certifican tiempo real estricto ni acotan jitter del sistema operativo o hardware.
pub const STRICT_REALTIME_NON_CERTIFICATION_NOTICE: &str = "La validación de perfil temporal y las pruebas de cero asignaciones en heap no certifican tiempo real estricto ni acotan jitter del sistema operativo o hardware.";

/// Perfil temporal de ejecución en [`crate`] enlazado con [`gogamen_core::TemporalProfile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RuntimeTemporalProfile {
    /// Ejecución general de mejor esfuerzo; permite E/S bloqueante y memoria dinámica.
    General,
    /// Ruta crítica interactiva acotada al presupuesto del tick (`BoundedTick`).
    Interactive,
    /// Ruta crítica consciente de plazos (`BoundedRt`), sin certificar tiempo real estricto.
    DeadlineAware,
}

impl RuntimeTemporalProfile {
    /// Convierte este perfil al [`gogamen_core::TemporalProfile`] equivalente.
    pub const fn to_core_temporal_profile(self) -> TemporalProfile {
        match self {
            Self::General => TemporalProfile::BestEffort,
            Self::Interactive => TemporalProfile::BoundedTick,
            Self::DeadlineAware => TemporalProfile::BoundedRt,
        }
    }

    /// Intenta construir un [`RuntimeTemporalProfile`] a partir de [`gogamen_core::TemporalProfile`].
    pub const fn from_core_temporal_profile(profile: TemporalProfile) -> Self {
        match profile {
            TemporalProfile::BestEffort | TemporalProfile::UnboundedBackground => Self::General,
            TemporalProfile::BoundedTick => Self::Interactive,
            TemporalProfile::BoundedRt => Self::DeadlineAware,
        }
    }

    /// Indica si el perfil corresponde a una ruta crítica acotada (`Interactive` o `DeadlineAware`).
    pub const fn is_critical_path(self) -> bool {
        matches!(self, Self::Interactive | Self::DeadlineAware)
    }

    /// Indica si el perfil declarado por una operación es compatible con este perfil de ruta.
    pub const fn accepts_declared_profile(self, declared: Self) -> bool {
        match self {
            Self::General => true,
            Self::Interactive => matches!(declared, Self::Interactive | Self::DeadlineAware),
            Self::DeadlineAware => matches!(declared, Self::DeadlineAware),
        }
    }
}

impl From<RuntimeTemporalProfile> for TemporalProfile {
    fn from(value: RuntimeTemporalProfile) -> Self {
        value.to_core_temporal_profile()
    }
}

impl From<TemporalProfile> for RuntimeTemporalProfile {
    fn from(value: TemporalProfile) -> Self {
        Self::from_core_temporal_profile(value)
    }
}

impl fmt::Display for RuntimeTemporalProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::General => f.write_str("General"),
            Self::Interactive => f.write_str("Interactive"),
            Self::DeadlineAware => f.write_str("DeadlineAware"),
        }
    }
}

/// Declaración contractual sobre el acotamiento de memoria de una operación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryBoundClaim {
    /// Memoria acotada explícitamente en bytes por invocación o estado.
    Bounded {
        /// Límite superior declarado en bytes.
        max_bytes: usize,
    },
    /// Cero asignaciones en heap tras la fase de preparación (`prepare`).
    ZeroHeapAfterPrepare {
        /// Bytes preasignados durante `prepare`.
        preallocated_bytes: usize,
    },
    /// Memoria no acotada o crecimiento dinámico arbitrario en ruta de ejecución.
    Unbounded,
}

impl MemoryBoundClaim {
    /// Indica si la declaración corresponde a memoria acotada.
    pub const fn is_bounded(self) -> bool {
        !matches!(self, Self::Unbounded)
    }
}

/// Propiedades declaradas por una operación respecto a bloqueos, memoria y perfil temporal.
///
/// Representa una afirmación contractual previa a la ejecución y **no** constituye
/// evidencia verificada en tiempo de ejecución ([`ObservedAllocationProof`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredCapabilityClaim {
    /// Perfil temporal declarado para la operación.
    pub profile: RuntimeTemporalProfile,
    /// Indica si la operación declara E/S bloqueante.
    pub declares_blocking_io: bool,
    /// Indica si la operación declara bloqueos de sincronización bloqueantes (p. ej. mutex bloqueante).
    pub declares_blocking_lock: bool,
    /// Declaración sobre acotamiento de memoria.
    pub memory_bound: MemoryBoundClaim,
    /// Efectos declarados compatibles con [`gogamen_core::EffectKind`].
    pub declared_effects: BTreeSet<EffectKind>,
    /// Permisos requeridos declarados.
    pub required_permissions: PermissionSet,
}

impl DeclaredCapabilityClaim {
    /// Construye una declaración no bloqueante y con cero asignaciones tras `prepare`.
    pub fn critical_non_blocking(
        profile: RuntimeTemporalProfile,
        preallocated_bytes: usize,
    ) -> Self {
        let mut declared_effects = BTreeSet::new();
        declared_effects.insert(EffectKind::StateMutation);
        Self {
            profile,
            declares_blocking_io: false,
            declares_blocking_lock: false,
            memory_bound: MemoryBoundClaim::ZeroHeapAfterPrepare { preallocated_bytes },
            declared_effects,
            required_permissions: PermissionSet::empty(),
        }
    }

    /// Construye una declaración de propósito general (permite bloqueos y memoria dinámica).
    pub fn general_unbounded() -> Self {
        Self {
            profile: RuntimeTemporalProfile::General,
            declares_blocking_io: false,
            declares_blocking_lock: false,
            memory_bound: MemoryBoundClaim::Unbounded,
            declared_effects: BTreeSet::new(),
            required_permissions: PermissionSet::empty(),
        }
    }

    /// Marca si la operación declara E/S bloqueante (estilo builder).
    pub fn with_blocking_io(mut self, blocking_io: bool) -> Self {
        self.declares_blocking_io = blocking_io;
        if blocking_io {
            self.declared_effects.insert(EffectKind::BlockingIo);
        }
        self
    }

    /// Marca si la operación declara bloqueos de sincronización bloqueantes (estilo builder).
    pub const fn with_blocking_lock(mut self, blocking_lock: bool) -> Self {
        self.declares_blocking_lock = blocking_lock;
        self
    }

    /// Establece la declaración de acotamiento de memoria (estilo builder).
    pub const fn with_memory_bound(mut self, memory_bound: MemoryBoundClaim) -> Self {
        self.memory_bound = memory_bound;
        self
    }

    /// Añade un efecto declarado (estilo builder).
    pub fn with_effect(mut self, effect: EffectKind) -> Self {
        if effect.is_blocking() {
            self.declares_blocking_io = true;
        }
        self.declared_effects.insert(effect);
        self
    }

    /// Indica si la operación declara E/S bloqueante por bandera o por [`EffectKind::BlockingIo`].
    pub fn has_blocking_io(&self) -> bool {
        self.declares_blocking_io
            || self
                .declared_effects
                .iter()
                .any(|effect| effect.is_blocking())
    }

    /// Convierte la declaración en propiedades compatibles con [`gogamen_core::DeclaredProperty`].
    pub fn to_core_declared_properties(&self) -> Vec<DeclaredProperty> {
        vec![
            DeclaredProperty::RequiredPermissions(self.required_permissions.clone()),
            DeclaredProperty::Effects(self.declared_effects.clone()),
            DeclaredProperty::TemporalProfile(self.profile.to_core_temporal_profile()),
        ]
    }
}

/// Descriptor de una operación participante en un plan de ruta crítica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalOperationDescriptor {
    /// Módulo propietario de la operación.
    pub module_id: ModuleId,
    /// Identificador único de la operación.
    pub operation_id: OperationId,
    /// Versión semántica requerida al catálogo, si se valida contra [`CapabilityCatalog`].
    pub required_version: CapabilityVersion,
    /// Propiedades declaradas por la operación.
    pub claim: DeclaredCapabilityClaim,
}

impl CriticalOperationDescriptor {
    /// Crea un descriptor de operación con versión `1.0.0`.
    pub fn new(
        module_id: impl Into<ModuleId>,
        operation_id: impl Into<OperationId>,
        claim: DeclaredCapabilityClaim,
    ) -> Self {
        Self {
            module_id: module_id.into(),
            operation_id: operation_id.into(),
            required_version: CapabilityVersion::new(1, 0, 0),
            claim,
        }
    }

    /// Establece la versión requerida para validación en [`CapabilityCatalog`].
    pub const fn with_required_version(mut self, version: CapabilityVersion) -> Self {
        self.required_version = version;
        self
    }

    /// Construye el [`OperationBinding`] equivalente para [`CapabilityCatalog`].
    pub fn to_binding(&self) -> OperationBinding {
        OperationBinding::new(
            self.module_id.clone(),
            self.operation_id.clone(),
            self.required_version,
        )
    }
}

/// Configuración de capacidad fija para los buffers preasignados de ruta crítica.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CriticalBufferConfig {
    /// Capacidad máxima del buffer circular preasignado (en número de muestras).
    pub ring_capacity: NonZeroUsize,
    /// Tamaño fijo en bytes del buffer scratch preasignado.
    pub scratch_capacity_bytes: usize,
}

impl CriticalBufferConfig {
    /// Crea una configuración de buffers preasignados validando que `ring_capacity > 0`.
    pub fn try_new(
        ring_capacity: usize,
        scratch_capacity_bytes: usize,
    ) -> Result<Self, TemporalValidationError> {
        let ring_capacity = NonZeroUsize::new(ring_capacity).ok_or(
            TemporalValidationError::InvalidBufferConfig("ring_capacity must be greater than zero"),
        )?;
        Ok(Self {
            ring_capacity,
            scratch_capacity_bytes,
        })
    }
}

/// Plan declarativo de ruta crítica a validar antes de `start` y preparar con buffers fijos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CriticalPathPlan {
    /// Perfil temporal objetivo del plan.
    pub profile: RuntimeTemporalProfile,
    /// Operaciones que integran la ruta de ejecución.
    pub operations: Vec<CriticalOperationDescriptor>,
    /// Configuración de buffers fijos a preasignar en `prepare`.
    pub buffer_config: CriticalBufferConfig,
}

impl CriticalPathPlan {
    /// Crea un plan con perfil objetivo, operaciones y configuración de buffers.
    pub fn new(
        profile: RuntimeTemporalProfile,
        operations: Vec<CriticalOperationDescriptor>,
        buffer_config: CriticalBufferConfig,
    ) -> Self {
        Self {
            profile,
            operations,
            buffer_config,
        }
    }
}

/// Límites y capacidades exactas preasignadas para [`CriticalPathBuffers`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CriticalBufferLimits {
    /// Capacidad fija del [`PreallocatedRingBuffer`] en número de elementos.
    pub ring_capacity: usize,
    /// Capacidad fija del buffer scratch en bytes.
    pub scratch_capacity_bytes: usize,
    /// Tamaño en bytes de cada ranura del ring buffer (`size_of::<Option<T>>()`).
    pub slot_size_bytes: usize,
    /// Total aproximado de bytes preasignados en heap durante `prepare`.
    pub total_preallocated_bytes: usize,
}

/// Buffer circular de capacidad fija preasignada en `prepare` y sin asignaciones en `step`.
#[derive(Debug, Clone)]
pub struct PreallocatedRingBuffer<T> {
    slots: Box<[Option<T>]>,
    head: usize,
    len: usize,
    overwrites: u64,
}

impl<T> PreallocatedRingBuffer<T> {
    /// Asigna en heap exactamente `capacity` ranuras vacías por única vez.
    pub fn with_capacity(capacity: NonZeroUsize) -> Self {
        let mut vec = Vec::with_capacity(capacity.get());
        vec.resize_with(capacity.get(), || None);
        Self {
            slots: vec.into_boxed_slice(),
            head: 0,
            len: 0,
            overwrites: 0,
        }
    }

    /// Capacidad máxima fija del buffer.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Número de elementos actualmente almacenados.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Indica si el buffer está vacío.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Indica si el buffer alcanzó su capacidad fija.
    pub fn is_full(&self) -> bool {
        self.len == self.slots.len()
    }

    /// Número total de sobrescrituras producidas por `push_overwrite`.
    pub const fn overwrites(&self) -> u64 {
        self.overwrites
    }

    /// Inserta un elemento en O(1) sin asignar memoria en heap.
    ///
    /// Si el buffer está lleno, sobrescribe y devuelve el elemento más antiguo.
    pub fn push_overwrite(&mut self, value: T) -> Option<T> {
        let cap = self.slots.len();
        if self.len == cap {
            let evicted = self.slots[self.head].replace(value);
            self.head = (self.head + 1) % cap;
            self.overwrites = self.overwrites.saturating_add(1);
            evicted
        } else {
            let tail = (self.head + self.len) % cap;
            self.slots[tail] = Some(value);
            self.len += 1;
            None
        }
    }

    /// Intenta insertar un elemento en O(1) sin sobrescribir ni asignar en heap.
    pub fn try_push(&mut self, value: T) -> Result<(), T> {
        let cap = self.slots.len();
        if self.len == cap {
            return Err(value);
        }
        let tail = (self.head + self.len) % cap;
        self.slots[tail] = Some(value);
        self.len += 1;
        Ok(())
    }

    /// Extrae el elemento más antiguo en O(1) sin liberar memoria del backing slice.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let item = self.slots[self.head].take();
        self.head = (self.head + 1) % self.slots.len();
        self.len -= 1;
        item
    }

    /// Devuelve una referencia al elemento más reciente sin extraerlo.
    pub fn latest(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        let idx = (self.head + self.len - 1) % self.slots.len();
        self.slots[idx].as_ref()
    }

    /// Itera sobre los elementos almacenados del más antiguo al más reciente.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let cap = self.slots.len();
        (0..self.len).filter_map(move |offset| {
            let idx = (self.head + offset) % cap;
            self.slots[idx].as_ref()
        })
    }

    /// Limpia todas las ranuras en sitio sin liberar la capacidad preasignada.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }
}

/// Conjunto de buffers preasignados para ejecutar pasos críticos con 0 asignaciones en heap.
#[derive(Debug, Clone)]
pub struct CriticalPathBuffers<T> {
    ring: PreallocatedRingBuffer<T>,
    scratch: Box<[u8]>,
    limits: CriticalBufferLimits,
}

impl<T> CriticalPathBuffers<T> {
    /// Preasigna todos los buffers según `config` antes de `start`.
    pub fn prepare(config: CriticalBufferConfig) -> Self {
        let ring = PreallocatedRingBuffer::with_capacity(config.ring_capacity);
        let scratch = vec![0_u8; config.scratch_capacity_bytes].into_boxed_slice();
        let slot_size_bytes = size_of::<Option<T>>();
        let ring_bytes = config.ring_capacity.get().saturating_mul(slot_size_bytes);
        let total_preallocated_bytes = ring_bytes.saturating_add(config.scratch_capacity_bytes);
        let limits = CriticalBufferLimits {
            ring_capacity: config.ring_capacity.get(),
            scratch_capacity_bytes: config.scratch_capacity_bytes,
            slot_size_bytes,
            total_preallocated_bytes,
        };
        Self {
            ring,
            scratch,
            limits,
        }
    }

    /// Límites y capacidades fijas preasignadas.
    pub const fn limits(&self) -> CriticalBufferLimits {
        self.limits
    }

    /// Inserta una muestra en el buffer circular sin asignar en heap.
    pub fn push_sample(&mut self, value: T) -> Option<T> {
        self.ring.push_overwrite(value)
    }

    /// Extrae la muestra más antigua del buffer circular sin asignar en heap.
    pub fn pop_sample(&mut self) -> Option<T> {
        self.ring.pop()
    }

    /// Acceso inmutable al buffer circular preasignado.
    pub const fn ring(&self) -> &PreallocatedRingBuffer<T> {
        &self.ring
    }

    /// Acceso mutable al buffer circular preasignado.
    pub fn ring_mut(&mut self) -> &mut PreallocatedRingBuffer<T> {
        &mut self.ring
    }

    /// Acceso de solo lectura al buffer scratch preasignado.
    pub fn scratch(&self) -> &[u8] {
        &self.scratch
    }

    /// Acceso mutable al buffer scratch preasignado.
    pub fn scratch_mut(&mut self) -> &mut [u8] {
        &mut self.scratch
    }
}

/// Errores explícitos de validación temporal previa a `start` o ciclo de vida.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalValidationError {
    /// Se rechazó una operación con E/S bloqueante en ruta crítica.
    BlockingIoRejected {
        /// Operación infractora.
        operation_id: OperationId,
        /// Perfil de ruta crítica que la rechazó.
        profile: RuntimeTemporalProfile,
    },
    /// Se rechazó una operación con bloqueo de sincronización en ruta crítica.
    BlockingLockRejected {
        /// Operación infractora.
        operation_id: OperationId,
        /// Perfil de ruta crítica que la rechazó.
        profile: RuntimeTemporalProfile,
    },
    /// Se rechazó una operación con memoria no acotada declarada en ruta crítica.
    UnboundedMemoryRejected {
        /// Operación infractora.
        operation_id: OperationId,
        /// Perfil de ruta crítica que la rechazó.
        profile: RuntimeTemporalProfile,
    },
    /// El perfil temporal declarado por la operación es incompatible con el perfil de la ruta.
    IncompatibleTemporalProfile {
        /// Operación infractora.
        operation_id: OperationId,
        /// Perfil requerido por la ruta.
        expected: RuntimeTemporalProfile,
        /// Perfil declarado por la operación.
        declared: RuntimeTemporalProfile,
    },
    /// Error devuelto por [`CapabilityCatalog::validate_temporal_path`].
    CatalogContract(ModuleContractError),
    /// Configuración inválida de buffers preasignados.
    InvalidBufferConfig(&'static str),
    /// Transición inválida del ciclo de vida `prepare -> start -> step -> stop`.
    LifecycleViolation(&'static str),
}

impl From<ModuleContractError> for TemporalValidationError {
    fn from(value: ModuleContractError) -> Self {
        Self::CatalogContract(value)
    }
}

impl fmt::Display for TemporalValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockingIoRejected {
                operation_id,
                profile,
            } => write!(
                f,
                "operation '{operation_id}' declares blocking IO and is rejected in {profile} critical path"
            ),
            Self::BlockingLockRejected {
                operation_id,
                profile,
            } => write!(
                f,
                "operation '{operation_id}' declares blocking lock and is rejected in {profile} critical path"
            ),
            Self::UnboundedMemoryRejected {
                operation_id,
                profile,
            } => write!(
                f,
                "operation '{operation_id}' declares unbounded memory and is rejected in {profile} critical path"
            ),
            Self::IncompatibleTemporalProfile {
                operation_id,
                expected,
                declared,
            } => write!(
                f,
                "operation '{operation_id}' declares profile {declared}, incompatible with {expected}"
            ),
            Self::CatalogContract(err) => write!(f, "capability catalog contract error: {err}"),
            Self::InvalidBufferConfig(msg) => write!(f, "invalid critical buffer config: {msg}"),
            Self::LifecycleViolation(msg) => write!(f, "critical path lifecycle violation: {msg}"),
        }
    }
}

impl Error for TemporalValidationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::CatalogContract(err) => Some(err),
            _ => None,
        }
    }
}

/// Resultado de validar un [`CriticalPathPlan`] antes de `start`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedCriticalPath {
    /// Perfil temporal validado.
    pub profile: RuntimeTemporalProfile,
    /// Perfil equivalente en [`gogamen_core::TemporalProfile`].
    pub core_profile: TemporalProfile,
    /// Declaraciones contractuales validadas por operación.
    pub declared_claims: Vec<(OperationId, DeclaredCapabilityClaim)>,
    /// Operaciones validadas contra [`CapabilityCatalog`], si se proporcionó catálogo.
    pub catalog_validated: Vec<ValidatedOperation>,
    /// Aviso explícito de no certificación en tiempo real estricto.
    pub non_certification_notice: &'static str,
}

/// Evidencia empírica observada sobre asignaciones en heap durante `step` tras warmup.
///
/// Se mantiene estrictamente separada de [`DeclaredCapabilityClaim`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObservedAllocationProof {
    /// Pasos de calentamiento ejecutados antes de iniciar la medición.
    pub warmup_steps: u64,
    /// Pasos medidos tras el calentamiento.
    pub measured_steps: u64,
    /// Número de llamadas `alloc` observadas durante los pasos medidos.
    pub heap_allocations_during_step: usize,
    /// Número de llamadas `realloc` observadas durante los pasos medidos.
    pub heap_reallocations_during_step: usize,
    /// Total de bytes solicitados al asignador durante los pasos medidos.
    pub heap_bytes_allocated_during_step: usize,
    /// Capacidades y límites de buffers fijos bajo los que se obtuvo la prueba.
    pub buffer_limits: CriticalBufferLimits,
    /// Aviso explícito de que esta prueba no certifica tiempo real estricto.
    pub non_certification_notice: &'static str,
}

impl ObservedAllocationProof {
    /// Construye una prueba observada adjuntando [`STRICT_REALTIME_NON_CERTIFICATION_NOTICE`].
    pub const fn new(
        warmup_steps: u64,
        measured_steps: u64,
        heap_allocations_during_step: usize,
        heap_reallocations_during_step: usize,
        heap_bytes_allocated_during_step: usize,
        buffer_limits: CriticalBufferLimits,
    ) -> Self {
        Self {
            warmup_steps,
            measured_steps,
            heap_allocations_during_step,
            heap_reallocations_during_step,
            heap_bytes_allocated_during_step,
            buffer_limits,
            non_certification_notice: STRICT_REALTIME_NON_CERTIFICATION_NOTICE,
        }
    }

    /// Indica si se ejecutó al menos un paso medido con cero asignaciones y cero reasignaciones.
    pub const fn zero_alloc_verified(&self) -> bool {
        self.measured_steps > 0
            && self.heap_allocations_during_step == 0
            && self.heap_reallocations_during_step == 0
    }
}

/// Reporte consolidado que distingue propiedades declaradas de evidencia observada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalProfileReport {
    /// Perfil temporal del runtime.
    pub profile: RuntimeTemporalProfile,
    /// Perfil temporal equivalente en [`gogamen_core::TemporalProfile`].
    pub core_profile: TemporalProfile,
    /// Propiedades declaradas por cada operación antes de `start`.
    pub declared_claims: Vec<(OperationId, DeclaredCapabilityClaim)>,
    /// Evidencia observada de asignaciones en heap durante `step`, si fue registrada.
    pub observed_proof: Option<ObservedAllocationProof>,
    /// Límites de buffers preasignados.
    pub buffer_limits: CriticalBufferLimits,
    /// Aviso explícito de no certificación en tiempo real estricto.
    pub non_certification_notice: &'static str,
}

/// Estado de ciclo de vida de una ruta crítica preparada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CriticalPathLifecycle {
    /// Validada y con buffers preasignados; aún no iniciada.
    Prepared,
    /// En ejecución; `step` permitido sin asignaciones en heap.
    Started,
    /// Detenida explícitamente.
    Stopped,
}

/// Ruta crítica validada antes de `start` con buffers preasignados en `prepare`.
#[derive(Debug, Clone)]
pub struct PreparedCriticalPath<T> {
    validation: ValidatedCriticalPath,
    buffers: CriticalPathBuffers<T>,
    lifecycle: CriticalPathLifecycle,
    steps_executed: u64,
    observed_proof: Option<ObservedAllocationProof>,
}

/// Valida todas las operaciones de `plan` antes de `start`.
///
/// En perfiles críticos (`Interactive` y `DeadlineAware`), rechaza explícitamente
/// cualquier operación que declare E/S bloqueante, bloqueos de sincronización bloqueantes,
/// memoria no acotada o un perfil temporal incompatible. Si se proporciona `catalog`,
/// delega además la validación de contrato y permisos a [`CapabilityCatalog::validate_temporal_path`].
pub fn validate_before_start(
    plan: &CriticalPathPlan,
    catalog: Option<&CapabilityCatalog>,
) -> Result<ValidatedCriticalPath, TemporalValidationError> {
    let mut declared_claims = Vec::with_capacity(plan.operations.len());
    let mut bindings = Vec::with_capacity(plan.operations.len());

    for op in &plan.operations {
        if plan.profile.is_critical_path() {
            if op.claim.has_blocking_io() {
                return Err(TemporalValidationError::BlockingIoRejected {
                    operation_id: op.operation_id.clone(),
                    profile: plan.profile,
                });
            }
            if op.claim.declares_blocking_lock {
                return Err(TemporalValidationError::BlockingLockRejected {
                    operation_id: op.operation_id.clone(),
                    profile: plan.profile,
                });
            }
            if !op.claim.memory_bound.is_bounded() {
                return Err(TemporalValidationError::UnboundedMemoryRejected {
                    operation_id: op.operation_id.clone(),
                    profile: plan.profile,
                });
            }
            if !plan.profile.accepts_declared_profile(op.claim.profile) {
                return Err(TemporalValidationError::IncompatibleTemporalProfile {
                    operation_id: op.operation_id.clone(),
                    expected: plan.profile,
                    declared: op.claim.profile,
                });
            }
        }

        declared_claims.push((op.operation_id.clone(), op.claim.clone()));
        bindings.push(op.to_binding());
    }

    let core_profile = plan.profile.to_core_temporal_profile();
    let catalog_validated = match catalog {
        Some(cat) => cat.validate_temporal_path(core_profile, &bindings)?,
        None => Vec::new(),
    };

    Ok(ValidatedCriticalPath {
        profile: plan.profile,
        core_profile,
        declared_claims,
        catalog_validated,
        non_certification_notice: STRICT_REALTIME_NON_CERTIFICATION_NOTICE,
    })
}

/// Valida el plan antes de `start` y preasigna [`CriticalPathBuffers`] de capacidad fija.
pub fn prepare_critical_path<T>(
    plan: &CriticalPathPlan,
    catalog: Option<&CapabilityCatalog>,
) -> Result<PreparedCriticalPath<T>, TemporalValidationError> {
    let validation = validate_before_start(plan, catalog)?;
    let buffers = CriticalPathBuffers::prepare(plan.buffer_config);
    Ok(PreparedCriticalPath {
        validation,
        buffers,
        lifecycle: CriticalPathLifecycle::Prepared,
        steps_executed: 0,
        observed_proof: None,
    })
}

impl<T> PreparedCriticalPath<T> {
    /// Inicia la ruta crítica previamente validada y preasignada.
    pub fn start(&mut self) -> Result<(), TemporalValidationError> {
        match self.lifecycle {
            CriticalPathLifecycle::Prepared => {
                self.lifecycle = CriticalPathLifecycle::Started;
                Ok(())
            }
            CriticalPathLifecycle::Started => Err(TemporalValidationError::LifecycleViolation(
                "critical path is already started",
            )),
            CriticalPathLifecycle::Stopped => Err(TemporalValidationError::LifecycleViolation(
                "cannot start a stopped critical path without re-preparing",
            )),
        }
    }

    /// Ejecuta un paso crítico usando exclusivamente los buffers preasignados (0 asignaciones en heap).
    pub fn step<F>(&mut self, sample: T, f: F) -> Result<Option<T>, TemporalValidationError>
    where
        F: FnOnce(&mut CriticalPathBuffers<T>),
    {
        if self.lifecycle != CriticalPathLifecycle::Started {
            return Err(TemporalValidationError::LifecycleViolation(
                "step requires lifecycle state Started",
            ));
        }
        let evicted = self.buffers.push_sample(sample);
        f(&mut self.buffers);
        self.steps_executed = self.steps_executed.saturating_add(1);
        Ok(evicted)
    }

    /// Detiene explícitamente la ruta crítica.
    pub fn stop(&mut self) -> Result<(), TemporalValidationError> {
        if self.lifecycle != CriticalPathLifecycle::Started {
            return Err(TemporalValidationError::LifecycleViolation(
                "stop requires lifecycle state Started",
            ));
        }
        self.lifecycle = CriticalPathLifecycle::Stopped;
        Ok(())
    }

    /// Estado actual del ciclo de vida.
    pub const fn lifecycle(&self) -> CriticalPathLifecycle {
        self.lifecycle
    }

    /// Número de pasos ejecutados desde `start`.
    pub const fn steps_executed(&self) -> u64 {
        self.steps_executed
    }

    /// Límites de los buffers preasignados.
    pub const fn buffer_limits(&self) -> CriticalBufferLimits {
        self.buffers.limits()
    }

    /// Acceso de solo lectura a los buffers preasignados.
    pub const fn buffers(&self) -> &CriticalPathBuffers<T> {
        &self.buffers
    }

    /// Registra la evidencia empírica observada de asignaciones en heap durante `step`.
    pub fn record_observed_proof(&mut self, proof: ObservedAllocationProof) {
        self.observed_proof = Some(proof);
    }

    /// Genera un [`TemporalProfileReport`] separando propiedades declaradas de evidencia observada.
    pub fn report(&self) -> TemporalProfileReport {
        TemporalProfileReport {
            profile: self.validation.profile,
            core_profile: self.validation.core_profile,
            declared_claims: self.validation.declared_claims.clone(),
            observed_proof: self.observed_proof,
            buffer_limits: self.buffers.limits(),
            non_certification_notice: STRICT_REALTIME_NON_CERTIFICATION_NOTICE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::{ModuleManifest, ModuleOrigin, OperationSpec};

    fn non_blocking_op(id: &str, profile: RuntimeTemporalProfile) -> CriticalOperationDescriptor {
        CriticalOperationDescriptor::new(
            "control",
            id,
            DeclaredCapabilityClaim::critical_non_blocking(profile, 256),
        )
    }

    fn buffer_cfg() -> CriticalBufferConfig {
        CriticalBufferConfig::try_new(8, 64).unwrap()
    }

    #[test]
    fn ac1_rejects_blocking_io_blocking_lock_and_unbounded_memory_before_start_in_critical_profiles()
     {
        for profile in [
            RuntimeTemporalProfile::Interactive,
            RuntimeTemporalProfile::DeadlineAware,
        ] {
            let blocking_io_plan = CriticalPathPlan::new(
                profile,
                vec![CriticalOperationDescriptor::new(
                    "io",
                    "disk_write",
                    DeclaredCapabilityClaim::critical_non_blocking(profile, 64)
                        .with_blocking_io(true),
                )],
                buffer_cfg(),
            );
            assert!(matches!(
                validate_before_start(&blocking_io_plan, None),
                Err(TemporalValidationError::BlockingIoRejected { .. })
            ));

            let blocking_lock_plan = CriticalPathPlan::new(
                profile,
                vec![CriticalOperationDescriptor::new(
                    "sync",
                    "std_mutex_lock",
                    DeclaredCapabilityClaim::critical_non_blocking(profile, 64)
                        .with_blocking_lock(true),
                )],
                buffer_cfg(),
            );
            assert!(matches!(
                validate_before_start(&blocking_lock_plan, None),
                Err(TemporalValidationError::BlockingLockRejected { .. })
            ));

            let unbounded_mem_plan = CriticalPathPlan::new(
                profile,
                vec![CriticalOperationDescriptor::new(
                    "alloc",
                    "grow_vec",
                    DeclaredCapabilityClaim::critical_non_blocking(profile, 64)
                        .with_memory_bound(MemoryBoundClaim::Unbounded),
                )],
                buffer_cfg(),
            );
            assert!(matches!(
                validate_before_start(&unbounded_mem_plan, None),
                Err(TemporalValidationError::UnboundedMemoryRejected { .. })
            ));
        }
    }

    #[test]
    fn ac1_general_profile_allows_blocking_and_unbounded_while_catalog_validation_enforces_core_contract()
     {
        let general_plan = CriticalPathPlan::new(
            RuntimeTemporalProfile::General,
            vec![CriticalOperationDescriptor::new(
                "worker",
                "slow_io",
                DeclaredCapabilityClaim::general_unbounded()
                    .with_blocking_io(true)
                    .with_blocking_lock(true),
            )],
            buffer_cfg(),
        );
        let validated = validate_before_start(&general_plan, None).expect("general plan allowed");
        assert_eq!(validated.core_profile, TemporalProfile::BestEffort);

        let mut catalog = CapabilityCatalog::new();
        catalog
            .register_module(
                ModuleManifest::new(
                    "control",
                    ModuleOrigin::Builtin,
                    CapabilityVersion::new(1, 0, 0),
                )
                .with_operation(
                    OperationSpec::new("step_pid", CapabilityVersion::new(1, 0, 0))
                        .with_effect(EffectKind::StateMutation)
                        .with_temporal_profile(TemporalProfile::BoundedRt),
                ),
            )
            .unwrap();

        let rt_plan = CriticalPathPlan::new(
            RuntimeTemporalProfile::DeadlineAware,
            vec![non_blocking_op(
                "step_pid",
                RuntimeTemporalProfile::DeadlineAware,
            )],
            buffer_cfg(),
        );
        let validated_rt =
            validate_before_start(&rt_plan, Some(&catalog)).expect("catalog validates BoundedRt");
        assert_eq!(validated_rt.core_profile, TemporalProfile::BoundedRt);
        assert_eq!(validated_rt.catalog_validated.len(), 1);
    }

    #[test]
    fn ac2_preallocated_ring_buffer_and_lifecycle_work_without_resizing() {
        let plan = CriticalPathPlan::new(
            RuntimeTemporalProfile::Interactive,
            vec![non_blocking_op("tick", RuntimeTemporalProfile::Interactive)],
            CriticalBufferConfig::try_new(4, 16).unwrap(),
        );
        let mut prepared = prepare_critical_path::<u64>(&plan, None).unwrap();
        assert_eq!(prepared.lifecycle(), CriticalPathLifecycle::Prepared);

        // step antes de start falla explícitamente.
        assert!(matches!(
            prepared.step(1, |_| {}),
            Err(TemporalValidationError::LifecycleViolation(_))
        ));

        prepared.start().unwrap();
        for i in 1..=6_u64 {
            let evicted = prepared
                .step(i, |bufs| {
                    bufs.scratch_mut()[0] = (i & 0xff) as u8;
                })
                .unwrap();
            if i <= 4 {
                assert_eq!(evicted, None);
            } else {
                assert_eq!(evicted, Some(i - 4));
            }
        }
        assert_eq!(prepared.buffers().ring().len(), 4);
        assert_eq!(prepared.buffers().ring().overwrites(), 2);
        assert_eq!(prepared.buffers().ring().latest(), Some(&6));
        prepared.stop().unwrap();
    }

    #[test]
    fn ac3_report_separates_declared_claims_from_observed_proof_and_includes_notice() {
        let plan = CriticalPathPlan::new(
            RuntimeTemporalProfile::DeadlineAware,
            vec![non_blocking_op(
                "tick",
                RuntimeTemporalProfile::DeadlineAware,
            )],
            buffer_cfg(),
        );
        let mut prepared = prepare_critical_path::<u32>(&plan, None).unwrap();
        let initial_report = prepared.report();
        assert!(initial_report.observed_proof.is_none());
        assert_eq!(initial_report.declared_claims.len(), 1);
        assert_eq!(
            initial_report.non_certification_notice,
            STRICT_REALTIME_NON_CERTIFICATION_NOTICE
        );

        let proof = ObservedAllocationProof::new(10, 100, 0, 0, 0, prepared.buffer_limits());
        assert!(proof.zero_alloc_verified());
        prepared.record_observed_proof(proof);

        let final_report = prepared.report();
        assert!(final_report.observed_proof.unwrap().zero_alloc_verified());
        assert_eq!(
            final_report
                .observed_proof
                .unwrap()
                .non_certification_notice,
            STRICT_REALTIME_NON_CERTIFICATION_NOTICE
        );
    }
}
