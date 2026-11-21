//! Extensiones WASM opcionales con presupuesto (feature `wasm`, GOG-86).
//!
//! Sandbox para **efectos puros no críticos** basado en Wasmtime:
//! - **Fuel** ([`WasmSandboxConfig::fuel`]): presupuesto *determinista* de instrucciones,
//!   no de tiempo de pared. Al agotarse produce [`WasmError::OutOfFuel`].
//! - **Memoria** ([`WasmSandboxConfig::max_memory_bytes`]): tope duro mediante
//!   `ResourceLimiter`; memorias declaradas o crecidas por encima se rechazan con
//!   [`WasmError::MemoryLimitExceeded`].
//! - **Imports**: se validan contra una allowlist cerrada ([`WasmHostImport`]) *antes* de
//!   instanciar ([`WasmError::UnauthorizedImport`]). Por defecto la allowlist está vacía:
//!   el módulo no recibe actuadores, capacidades, reloj, WASI ni red.
//!
//! - **Carga/compilación** (GOG-150): `bytes` por encima de
//!   [`WasmSandboxConfig::max_module_bytes`] (default 4 MiB) se rechazan *antes* de
//!   `Module::new` ([`WasmError::ModuleTooLarge`]). La compilación corre en un hilo aparte
//!   con timeout ([`WasmSandboxConfig::compile_timeout`], default 5 s): si vence, `load`
//!   devuelve [`WasmError::CompileTimeout`] sin bloquear al llamador. Wasmtime no ofrece
//!   cancelación de la compilación (ni fuel ni epoch la cubren), así que el hilo huérfano
//!   termina por su cuenta y descarta el resultado; durante ese tiempo puede seguir
//!   consumiendo una CPU y memoria acotadas por el tope de bytes. No hay validación
//!   estructural previa adicional (no se añade dependencia `wasmparser`).
//!
//! # Sin garantías de tiempo real duro
//! El fuel es un presupuesto determinista de instrucciones Wasm; **no** acota la latencia
//! de pared ni ofrece garantías hard real-time. Compilar y instanciar tiene coste variable.
//!
//! # ABI de efecto
//! El módulo exporta `run: (i64) -> i64` (función pura). Cada invocación usa un `Store`
//! nuevo (sin estado entre llamadas) con el fuel completo.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::sync::mpsc;
use std::time::Duration;

use wasmtime::{
    Config, Engine, ExternType, Instance, InstancePre, Linker, Module, ResourceLimiter, Store, Trap,
};

use crate::extensions::{ExtensionError, ExtensionManifest, StaticExtensionRegistry};

/// Nombre del módulo de imports permitidos.
pub const WASM_HOST_MODULE: &str = "gogamen";
/// Nombre de la función exportada por el efecto.
pub const WASM_ENTRY_EXPORT: &str = "run";

/// Funciones puras que el anfitrión puede ofrecer a un módulo (única allowlist posible).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum WasmHostImport {
    /// `gogamen.min_i64(i64, i64) -> i64`.
    MinI64,
    /// `gogamen.max_i64(i64, i64) -> i64`.
    MaxI64,
}

impl WasmHostImport {
    fn name(self) -> &'static str {
        match self {
            Self::MinI64 => "min_i64",
            Self::MaxI64 => "max_i64",
        }
    }
    fn matches(self, module: &str, name: &str) -> bool {
        module == WASM_HOST_MODULE && name == self.name()
    }
}

/// Configuración del sandbox WASM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmSandboxConfig {
    /// Fuel (instrucciones Wasm aproximadas) por invocación.
    pub fuel: u64,
    /// Máximo de bytes de memoria lineal (múltiplo de 64 KiB en la práctica).
    pub max_memory_bytes: usize,
    /// Imports permitidos; vacío por defecto (efecto puro sin anfitrión).
    pub allowed_imports: BTreeSet<WasmHostImport>,
    /// Tamaño máximo del módulo de entrada en bytes, comprobado antes de compilar.
    pub max_module_bytes: usize,
    /// Tiempo máximo de espera de la compilación (el hilo no es cancelable).
    pub compile_timeout: Duration,
}

/// Tope por defecto de bytes del módulo (4 MiB).
pub const DEFAULT_MAX_MODULE_BYTES: usize = 4 * 1024 * 1024;
/// Timeout de compilación por defecto (5 s).
pub const DEFAULT_COMPILE_TIMEOUT: Duration = Duration::from_secs(5);

impl WasmSandboxConfig {
    /// Configuración estricta: fuel dado, 1 página (64 KiB) y sin imports.
    pub fn strict(fuel: u64) -> Self {
        Self {
            fuel,
            max_memory_bytes: 64 * 1024,
            allowed_imports: BTreeSet::new(),
            max_module_bytes: DEFAULT_MAX_MODULE_BYTES,
            compile_timeout: DEFAULT_COMPILE_TIMEOUT,
        }
    }
    /// Cambia el tope de bytes del módulo (rechazo previo a compilar).
    pub fn with_max_module_bytes(mut self, bytes: usize) -> Self {
        self.max_module_bytes = bytes;
        self
    }
    /// Cambia el timeout de compilación.
    pub fn with_compile_timeout(mut self, timeout: Duration) -> Self {
        self.compile_timeout = timeout;
        self
    }
    /// Cambia el tope de memoria en bytes.
    pub fn with_max_memory_bytes(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = bytes;
        self
    }
    /// Permite un import puro del anfitrión.
    pub fn allow_import(mut self, import: WasmHostImport) -> Self {
        self.allowed_imports.insert(import);
        self
    }
}

/// Errores tipados del sandbox WASM.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmError {
    /// El módulo no compila/valida.
    InvalidModule(String),
    /// Import fuera de la allowlist (rechazado antes de instanciar).
    UnauthorizedImport {
        /// Módulo importado.
        module: String,
        /// Nombre importado.
        name: String,
    },
    /// Memoria declarada o crecida por encima del tope.
    MemoryLimitExceeded {
        /// Tope configurado en bytes.
        limit_bytes: usize,
    },
    /// El presupuesto de fuel se agotó (fallo controlado).
    OutOfFuel {
        /// Fuel configurado.
        budget: u64,
    },
    /// El módulo no exporta `run: (i64) -> i64`.
    MissingEntry,
    /// Otro trap de ejecución (p. ej. `unreachable`).
    Trap(String),
    /// El módulo supera el tope de bytes (rechazado antes de compilar).
    ModuleTooLarge {
        /// Tamaño recibido.
        size: usize,
        /// Tope configurado.
        limit: usize,
    },
    /// La compilación no terminó en el tiempo asignado (el hilo queda huérfano).
    CompileTimeout {
        /// Timeout configurado en milisegundos.
        timeout_ms: u64,
    },
    /// Error interno de configuración del motor.
    Engine(String),
    /// El manifiesto viola el contrato de pureza (permisos, origen remoto, etc.).
    Manifest(String),
}

impl fmt::Display for WasmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModule(m) => write!(f, "invalid wasm module: {m}"),
            Self::UnauthorizedImport { module, name } => {
                write!(f, "unauthorized wasm import `{module}.{name}`")
            }
            Self::MemoryLimitExceeded { limit_bytes } => {
                write!(f, "wasm memory exceeds limit of {limit_bytes} bytes")
            }
            Self::OutOfFuel { budget } => write!(f, "wasm fuel budget {budget} exhausted"),
            Self::MissingEntry => write!(f, "wasm module lacks `run: (i64) -> i64` export"),
            Self::Trap(m) => write!(f, "wasm trap: {m}"),
            Self::ModuleTooLarge { size, limit } => {
                write!(f, "wasm module of {size} bytes exceeds limit of {limit}")
            }
            Self::CompileTimeout { timeout_ms } => {
                write!(f, "wasm compilation exceeded {timeout_ms} ms")
            }
            Self::Engine(m) => write!(f, "wasm engine error: {m}"),
            Self::Manifest(m) => write!(f, "wasm extension manifest rejected: {m}"),
        }
    }
}

impl Error for WasmError {}

struct Limiter {
    max_memory: usize,
    denied: bool,
}

impl ResourceLimiter for Limiter {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        if desired > self.max_memory {
            self.denied = true;
            return Ok(false);
        }
        Ok(true)
    }
    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> wasmtime::Result<bool> {
        Ok(desired <= 1024)
    }
    fn instances(&self) -> usize {
        1
    }
    fn memories(&self) -> usize {
        1
    }
    fn tables(&self) -> usize {
        1
    }
}

/// Compila en un hilo aparte y espera como mucho `timeout`. Si vence, el hilo queda
/// huérfano (la compilación de Wasmtime no es cancelable) y termina por su cuenta.
fn compile_with_timeout(
    engine: &Engine,
    bytes: &[u8],
    timeout: Duration,
) -> Result<Module, WasmError> {
    let (tx, rx) = mpsc::channel();
    let engine = engine.clone();
    let owned = bytes.to_vec();
    std::thread::Builder::new()
        .name("gogamen-wasm-compile".to_owned())
        .spawn(move || {
            let r = Module::new(&engine, &owned).map_err(|e| format!("{e}"));
            let _ = tx.send(r);
        })
        .map_err(|e| WasmError::Engine(format!("spawn compile thread: {e}")))?;
    match rx.recv_timeout(timeout) {
        Ok(Ok(m)) => Ok(m),
        Ok(Err(e)) => Err(WasmError::InvalidModule(e)),
        Err(_) => Err(WasmError::CompileTimeout {
            timeout_ms: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
        }),
    }
}

/// Efecto WASM puro, validado y precompilado, invocable bajo presupuesto.
pub struct WasmEffect {
    engine: Engine,
    pre: InstancePre<Limiter>,
    config: WasmSandboxConfig,
}

impl fmt::Debug for WasmEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WasmEffect")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl WasmEffect {
    /// Compila `bytes` (binario Wasm o texto WAT), valida imports contra la allowlist y
    /// deja el módulo listo para instanciar. No ejecuta código del módulo.
    pub fn load(bytes: &[u8], config: WasmSandboxConfig) -> Result<Self, WasmError> {
        if bytes.len() > config.max_module_bytes {
            return Err(WasmError::ModuleTooLarge {
                size: bytes.len(),
                limit: config.max_module_bytes,
            });
        }
        let mut cfg = Config::new();
        cfg.consume_fuel(true);
        let engine = Engine::new(&cfg).map_err(|e| WasmError::Engine(e.to_string()))?;
        let module = compile_with_timeout(&engine, bytes, config.compile_timeout)?;

        let mut linker: Linker<Limiter> = Linker::new(&engine);
        for import in module.imports() {
            let allowed = matches!(import.ty(), ExternType::Func(_))
                && config
                    .allowed_imports
                    .iter()
                    .any(|a| a.matches(import.module(), import.name()));
            if !allowed {
                return Err(WasmError::UnauthorizedImport {
                    module: import.module().to_owned(),
                    name: import.name().to_owned(),
                });
            }
        }
        for imp in &config.allowed_imports {
            let result = match imp {
                WasmHostImport::MinI64 => {
                    linker.func_wrap(WASM_HOST_MODULE, imp.name(), |a: i64, b: i64| a.min(b))
                }
                WasmHostImport::MaxI64 => {
                    linker.func_wrap(WASM_HOST_MODULE, imp.name(), |a: i64, b: i64| a.max(b))
                }
            };
            result.map_err(|e| WasmError::Engine(e.to_string()))?;
        }
        let pre = linker
            .instantiate_pre(&module)
            .map_err(|e| WasmError::Engine(e.to_string()))?;
        match module.get_export(WASM_ENTRY_EXPORT) {
            Some(ExternType::Func(ty))
                if ty.params().len() == 1
                    && ty.results().len() == 1
                    && ty
                        .params()
                        .next()
                        .is_some_and(|t| t.matches(&wasmtime::ValType::I64))
                    && ty
                        .results()
                        .next()
                        .is_some_and(|t| t.matches(&wasmtime::ValType::I64)) => {}
            _ => return Err(WasmError::MissingEntry),
        }
        Ok(Self {
            engine,
            pre,
            config,
        })
    }

    /// Configuración efectiva.
    pub fn config(&self) -> &WasmSandboxConfig {
        &self.config
    }

    /// Invoca el efecto con un `Store` nuevo y el fuel completo. Devuelve el resultado y el
    /// fuel consumido.
    pub fn invoke_with_fuel(&self, input: i64) -> Result<(i64, u64), WasmError> {
        let limit = self.config.max_memory_bytes;
        let mut store = Store::new(
            &self.engine,
            Limiter {
                max_memory: limit,
                denied: false,
            },
        );
        store.limiter(|l| l);
        store
            .set_fuel(self.config.fuel)
            .map_err(|e| WasmError::Engine(e.to_string()))?;

        let mem_err = |store: &Store<Limiter>| {
            store
                .data()
                .denied
                .then_some(WasmError::MemoryLimitExceeded { limit_bytes: limit })
        };
        let instance: Instance = match self.pre.instantiate(&mut store) {
            Ok(i) => i,
            Err(e) => return Err(self.classify(&store, e, limit)),
        };
        let run = instance
            .get_typed_func::<i64, i64>(&mut store, WASM_ENTRY_EXPORT)
            .map_err(|_| WasmError::MissingEntry)?;
        match run.call(&mut store, input) {
            Ok(v) => {
                // `memory.grow` denegado devuelve -1 al módulo; el sandbox lo trata como
                // violación (el efecto no puede degradar silenciosamente el tope).
                if let Some(e) = mem_err(&store) {
                    return Err(e);
                }
                let left = store.get_fuel().unwrap_or(0);
                Ok((v, self.config.fuel - left))
            }
            Err(e) => Err(mem_err(&store).unwrap_or_else(|| self.classify(&store, e, limit))),
        }
    }

    /// Invoca el efecto (`run`) bajo presupuesto.
    pub fn invoke(&self, input: i64) -> Result<i64, WasmError> {
        self.invoke_with_fuel(input).map(|(v, _)| v)
    }

    fn classify(&self, store: &Store<Limiter>, e: wasmtime::Error, limit: usize) -> WasmError {
        if let Some(trap) = e.downcast_ref::<Trap>()
            && *trap == Trap::OutOfFuel
        {
            return WasmError::OutOfFuel {
                budget: self.config.fuel,
            };
        }
        if store.data().denied {
            return WasmError::MemoryLimitExceeded { limit_bytes: limit };
        }
        let msg = format!("{e}");
        if msg.contains("memory minimum size") || msg.contains("exceeds memory limits") {
            return WasmError::MemoryLimitExceeded { limit_bytes: limit };
        }
        WasmError::Trap(msg)
    }
}

/// Extensión WASM con manifiesto: exige pureza (sin permisos requeridos, origen estático).
///
/// No es una nueva [`crate::ExtensionKind`]: un efecto WASM nunca recibe actuadores ni
/// permisos; solo `i64 -> i64` y la allowlist de imports puros.
#[derive(Debug)]
pub struct WasmEffectExtension {
    manifest: ExtensionManifest,
    effect: WasmEffect,
}

impl WasmEffectExtension {
    /// Valida el manifiesto (contrato estático, sin permisos) y carga el módulo.
    pub fn new(
        manifest: ExtensionManifest,
        bytes: &[u8],
        config: WasmSandboxConfig,
    ) -> Result<Self, WasmError> {
        if manifest.required_permissions.iter().next().is_some() {
            return Err(WasmError::Manifest(
                "wasm effects are pure and must not require permissions".to_owned(),
            ));
        }
        StaticExtensionRegistry::new()
            .validate_manifest(&manifest, manifest.kind)
            .map_err(|e: ExtensionError| WasmError::Manifest(e.to_string()))?;
        let effect = WasmEffect::load(bytes, config)?;
        Ok(Self { manifest, effect })
    }
    /// Manifiesto declarado.
    pub fn manifest(&self) -> &ExtensionManifest {
        &self.manifest
    }
    /// Efecto cargado.
    pub fn effect(&self) -> &WasmEffect {
        &self.effect
    }
}
