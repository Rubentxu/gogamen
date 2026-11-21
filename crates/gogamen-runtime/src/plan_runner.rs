//! Ejecución headless de [`ExecutablePlan`] sin frontend ni DSL (`std`-only).
//!
//! Separa estrictamente la definición compilada inmutable ([`ExecutablePlan`]) de
//! cada instancia mutable de ejecución ([`PlanInstance`]): múltiples instancias
//! pueden compartir el mismo `&ExecutablePlan` o `Arc<ExecutablePlan>` sin
//! compartir buffers de puertos, estado de nodos, reloj lógico ni `TraceHasher`.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

use gogamen_compiler::{
    ArtifactError, BoundedExprEvaluator, CompileError, CompiledExpr, CompiledPlanArtifact,
    ConstantValue, ExecutablePlan, ExprSchema, ExprValue, SpecCompiler, VarIndex,
    load_plan_from_artifact_json,
};
use gogamen_core::{
    CapabilityCatalog, CapabilityVersion, DomainError, EffectKind, LogicalClock, MasterSeed,
    ModuleContractError, ModuleManifest, ModuleOrigin, NodeIndex, OperationSpec, PlanRuntime,
    PortIndex, PortType, StepDt, TemporalProfile, Tick, TraceHasher,
};
use gogamen_spec::{
    AVAILABLE_EPISTEMIC_MODELS, AbstentionPolicySpec, AskActionSpec, EpistemicQuestionSpec,
    ExprAst, ExprType, ExprUnit, GuardPolicySpec, InspectActionSpec, ProvenanceSpec, SpecError,
    SystemSpec, UnknownFieldPolicy,
};

const DEFAULT_DT_SECS: f64 = 0.02;
const DEFAULT_SEED: u64 = 42;

/// Valor tipado que fluye por los puertos de un [`ExecutablePlan`] en tiempo de ejecución.
#[derive(Debug, Clone, PartialEq)]
pub enum PortValue {
    /// Booleano.
    Bool(bool),
    /// Entero con signo de 64 bits.
    I64(i64),
    /// Entero sin signo de 64 bits.
    U64(u64),
    /// Punto flotante de 64 bits.
    F64(f64),
    /// Cadena UTF-8.
    String(String),
    /// Valor unitario.
    Unit,
}

impl PortValue {
    fn write_to_hasher(&self, hasher: &mut TraceHasher) {
        match self {
            Self::Bool(b) => {
                hasher.write_u8(1);
                hasher.write_u8(u8::from(*b));
            }
            Self::I64(v) => {
                hasher.write_u8(2);
                hasher.write_u64(*v as u64);
            }
            Self::U64(v) => {
                hasher.write_u8(3);
                hasher.write_u64(*v);
            }
            Self::F64(v) => {
                hasher.write_u8(4);
                hasher.write_f64(*v);
            }
            Self::String(s) => {
                hasher.write_u8(5);
                hasher.write_str(s);
            }
            Self::Unit => {
                hasher.write_u8(6);
            }
        }
    }
}

/// Errores al compilar, instanciar o ejecutar un [`ExecutablePlan`].
#[derive(Debug, Clone, PartialEq)]
pub enum PlanRuntimeError {
    /// Error parseando o validando la especificación declarativa.
    Spec(SpecError),
    /// Error compilando la especificación contra el catálogo de capacidades.
    Compile(CompileError),
    /// Error verificando, cargando o compilando un artefacto versionado.
    Artifact(ArtifactError),
    /// Error de dominio (por ejemplo `StepDt` inválido o desbordamiento de reloj).
    Domain(DomainError),
    /// Operación no soportada por el runtime integrado.
    UnsupportedOperation {
        /// Identificador de módulo.
        module_id: String,
        /// Identificador de operación.
        operation_id: String,
    },
    /// Configuración de sistema inválida para instanciar el plan.
    InvalidSystemConfig {
        /// Clave de configuración.
        key: &'static str,
        /// Motivo del rechazo.
        reason: String,
    },
    /// Configuración de nodo inválida para instanciar o ejecutar el plan.
    InvalidNodeConfig {
        /// Clave de configuración.
        key: &'static str,
        /// Motivo del rechazo.
        reason: String,
    },
    /// El plan pasado a `step` no coincide con el plan usado al instanciar.
    PlanMismatch {
        /// Huella semántica esperada por la instancia.
        expected_fingerprint: u64,
        /// Huella semántica del plan recibido.
        found_fingerprint: u64,
    },
}

impl fmt::Display for PlanRuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(err) => write!(f, "spec error: {err}"),
            Self::Compile(err) => write!(f, "compile error: {err}"),
            Self::Artifact(err) => write!(f, "artifact error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::UnsupportedOperation {
                module_id,
                operation_id,
            } => write!(
                f,
                "unsupported builtin operation '{module_id}::{operation_id}'"
            ),
            Self::InvalidSystemConfig { key, reason } => {
                write!(f, "invalid system config '{key}': {reason}")
            }
            Self::InvalidNodeConfig { key, reason } => {
                write!(f, "invalid node config '{key}': {reason}")
            }
            Self::PlanMismatch {
                expected_fingerprint,
                found_fingerprint,
            } => write!(
                f,
                "plan mismatch: expected fingerprint {expected_fingerprint:#018x}, found {found_fingerprint:#018x}"
            ),
        }
    }
}

impl std::error::Error for PlanRuntimeError {}

impl From<SpecError> for PlanRuntimeError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

impl From<CompileError> for PlanRuntimeError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

impl From<ArtifactError> for PlanRuntimeError {
    fn from(value: ArtifactError) -> Self {
        Self::Artifact(value)
    }
}

impl From<DomainError> for PlanRuntimeError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

/// Construye el catálogo de capacidades integrado para sistemas convencionales sin frontend.
pub fn builtin_plan_catalog() -> CapabilityCatalog {
    let mut catalog = CapabilityCatalog::new();

    let observer = ModuleManifest::new(
        "observer",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("observer", CapabilityVersion::new(1, 0, 0))
            .with_output("observation", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    let policy = ModuleManifest::new(
        "policy",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("policy", CapabilityVersion::new(1, 0, 0))
            .with_input("observation", PortType::F64)
            .with_output("action", PortType::I64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    let math = ModuleManifest::new(
        "core.math",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("source", CapabilityVersion::new(1, 0, 0))
            .with_output("out", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("gain", CapabilityVersion::new(1, 0, 0))
            .with_input("in", PortType::F64)
            .with_output("out", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("accumulate", CapabilityVersion::new(1, 0, 0))
            .with_input("in", PortType::F64)
            .with_output("out", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("counter", CapabilityVersion::new(1, 0, 0))
            .with_output("count", PortType::I64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    let agent = ModuleManifest::new(
        "agent",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("sensor", CapabilityVersion::new(1, 0, 0))
            .with_output("observation", PortType::F64)
            .with_output("energy", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("belief", CapabilityVersion::new(1, 0, 0))
            .with_input("observation", PortType::F64)
            .with_input("energy", PortType::F64)
            .with_output("belief_score", PortType::F64)
            .with_output("should_explore", PortType::Bool)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("planner", CapabilityVersion::new(1, 0, 0))
            .with_input("belief_score", PortType::F64)
            .with_input("should_explore", PortType::Bool)
            .with_output("action", PortType::I64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    let language = ModuleManifest::new(
        "language",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("human_channel", CapabilityVersion::new(1, 0, 0))
            .with_output("observation", PortType::F64)
            .with_output("message_confidence", PortType::F64)
            .with_output("is_order", PortType::Bool)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("direct_sensor", CapabilityVersion::new(1, 0, 0))
            .with_output("direct_obs", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("grounder", CapabilityVersion::new(1, 0, 0))
            .with_input("observation", PortType::F64)
            .with_input("message_confidence", PortType::F64)
            .with_output("grounded_prob", PortType::F64)
            .with_output("abstained", PortType::Bool)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("belief", CapabilityVersion::new(1, 0, 0))
            .with_input("grounded_prob", PortType::F64)
            .with_input("abstained", PortType::Bool)
            .with_input("direct_obs", PortType::F64)
            .with_output("belief_prob", PortType::F64)
            .with_output("should_ask", PortType::Bool)
            .with_output("should_inspect", PortType::Bool)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("guarded_planner", CapabilityVersion::new(1, 0, 0))
            .with_input("belief_prob", PortType::F64)
            .with_input("should_ask", PortType::Bool)
            .with_input("should_inspect", PortType::Bool)
            .with_input("is_order", PortType::Bool)
            .with_output("action", PortType::I64)
            .with_output("guard_rejected", PortType::Bool)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    catalog
        .register_module(observer)
        .expect("builtin observer module must register");
    catalog
        .register_module(policy)
        .expect("builtin policy module must register");
    catalog
        .register_module(math)
        .expect("builtin core.math module must register");
    catalog
        .register_module(agent)
        .expect("builtin agent module must register");
    catalog
        .register_module(language)
        .expect("builtin language module must register");
    catalog
}

/// Contexto de solo lectura entregado a un [`ExternalKernel`] al instanciar un nodo.
///
/// Expone la configuración constante del nodo resuelta por el compilador y la semilla
/// maestra, de modo que la inicialización sea determinista.
#[derive(Debug, Clone, Copy)]
pub struct ExternalNodeContext<'a> {
    plan: &'a ExecutablePlan,
    node: NodeIndex,
    seed: MasterSeed,
}

impl<'a> ExternalNodeContext<'a> {
    /// Índice del nodo que se está instanciando.
    pub fn node(&self) -> NodeIndex {
        self.node
    }

    /// Semilla maestra de la instancia.
    pub fn seed(&self) -> MasterSeed {
        self.seed
    }

    /// Constante `f64` (o `i64` promovida) del nodo.
    pub fn config_f64(&self, key: &str) -> Option<f64> {
        node_f64(self.plan, self.node, key)
    }

    /// Constante `i64` del nodo.
    pub fn config_i64(&self, key: &str) -> Option<i64> {
        node_i64(self.plan, self.node, key)
    }

    /// Constante booleana del nodo.
    pub fn config_bool(&self, key: &str) -> Option<bool> {
        node_bool(self.plan, self.node, key)
    }

    /// Constante de texto del nodo.
    pub fn config_str(&self, key: &str) -> Option<&'a str> {
        node_str(self.plan, self.node, key)
    }
}

/// Kernel de operación registrado fuera de los kernels integrados del runtime.
///
/// Debe ser determinista: la salida depende únicamente del estado, las entradas y la
/// configuración del nodo. El estado por nodo es un `Vec<PortValue>` propiedad de la
/// instancia, por lo que [`PlanInstance`] sigue siendo clonable y aislada.
pub trait ExternalKernel: fmt::Debug + Send + Sync {
    /// Construye el estado inicial de un nodo (por defecto, vacío).
    fn init_state(
        &self,
        _node: &ExternalNodeContext<'_>,
    ) -> Result<Vec<PortValue>, PlanRuntimeError> {
        Ok(Vec::new())
    }

    /// Evalúa un tick: lee `inputs`, actualiza `state` y escribe `outputs`.
    fn eval(
        &self,
        state: &mut Vec<PortValue>,
        inputs: &[Option<PortValue>],
        outputs: &mut [Option<PortValue>],
    );
}

/// Error al construir un [`KernelRegistry`].
#[derive(Debug, Clone, PartialEq)]
pub enum KernelRegistryError {
    /// El módulo externo colisiona con un módulo integrado o con otro ya registrado.
    DuplicateModule {
        /// Módulo duplicado.
        module_id: String,
    },
    /// El kernel referencia una operación no declarada en el manifiesto del módulo.
    UnknownOperation {
        /// Módulo consultado.
        module_id: String,
        /// Operación inexistente.
        operation_id: String,
    },
    /// Ya existe un kernel enlazado para `(module_id, operation_id)`.
    DuplicateKernel {
        /// Módulo propietario.
        module_id: String,
        /// Operación duplicada.
        operation_id: String,
    },
    /// El manifiesto viola el contrato del catálogo de capacidades.
    Contract(ModuleContractError),
}

impl fmt::Display for KernelRegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateModule { module_id } => {
                write!(f, "module '{module_id}' is already registered or builtin")
            }
            Self::UnknownOperation {
                module_id,
                operation_id,
            } => write!(
                f,
                "operation '{module_id}::{operation_id}' is not declared by its module manifest"
            ),
            Self::DuplicateKernel {
                module_id,
                operation_id,
            } => write!(
                f,
                "kernel for '{module_id}::{operation_id}' is already bound"
            ),
            Self::Contract(err) => write!(f, "module contract error: {err:?}"),
        }
    }
}

impl std::error::Error for KernelRegistryError {}

/// Registro determinista de kernels externos indexados por `(module_id, operation_id)`.
///
/// Cada módulo externo se declara con su [`ModuleManifest`] (contrato del catálogo de
/// capacidades) y sus operaciones se enlazan a un [`ExternalKernel`]. El registro produce
/// el [`CapabilityCatalog`] ampliado (integrado + externo) que debe usarse tanto para
/// compilar como para cargar artefactos.
#[derive(Debug, Clone, Default)]
pub struct KernelRegistry {
    modules: BTreeMap<String, ModuleManifest>,
    kernels: BTreeMap<(String, String), Arc<dyn ExternalKernel>>,
}

impl KernelRegistry {
    /// Crea un registro vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declara un módulo externo. Rechaza colisiones con módulos integrados o duplicados.
    pub fn register_module(&mut self, manifest: ModuleManifest) -> Result<(), KernelRegistryError> {
        let id = manifest.id.as_str().to_owned();
        let mut probe = builtin_plan_catalog();
        for existing in self.modules.values() {
            probe
                .register_module(existing.clone())
                .map_err(KernelRegistryError::Contract)?;
        }
        if probe.module(&manifest.id).is_some() {
            return Err(KernelRegistryError::DuplicateModule { module_id: id });
        }
        probe
            .register_module(manifest.clone())
            .map_err(KernelRegistryError::Contract)?;
        self.modules.insert(id, manifest);
        Ok(())
    }

    /// Enlaza un kernel a una operación declarada por un módulo ya registrado.
    pub fn bind_kernel(
        &mut self,
        module_id: &str,
        operation_id: &str,
        kernel: Arc<dyn ExternalKernel>,
    ) -> Result<(), KernelRegistryError> {
        let declared = self
            .modules
            .get(module_id)
            .is_some_and(|m| m.operations.iter().any(|op| op.id.as_str() == operation_id));
        if !declared {
            return Err(KernelRegistryError::UnknownOperation {
                module_id: module_id.to_owned(),
                operation_id: operation_id.to_owned(),
            });
        }
        let key = (module_id.to_owned(), operation_id.to_owned());
        if self.kernels.contains_key(&key) {
            return Err(KernelRegistryError::DuplicateKernel {
                module_id: key.0,
                operation_id: key.1,
            });
        }
        self.kernels.insert(key, kernel);
        Ok(())
    }

    /// Devuelve el kernel enlazado a `(module_id, operation_id)`, si existe.
    pub fn kernel(&self, module_id: &str, operation_id: &str) -> Option<&Arc<dyn ExternalKernel>> {
        self.kernels
            .get(&(module_id.to_owned(), operation_id.to_owned()))
    }

    /// Número de kernels enlazados.
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Indica si no hay kernels enlazados.
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// Catálogo ampliado: [`builtin_plan_catalog`] más los módulos externos (orden determinista).
    pub fn catalog(&self) -> CapabilityCatalog {
        let mut catalog = builtin_plan_catalog();
        for manifest in self.modules.values() {
            // `register_module` ya validó el contrato; un fallo aquí sería inalcanzable.
            let _ = catalog.register_module(manifest.clone());
        }
        catalog
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
enum BuiltinKernel {
    External {
        kernel: Arc<dyn ExternalKernel>,
    },
    Observer {
        rate_hz: i64,
        normalized: bool,
    },
    Policy {
        gain: f64,
    },
    MathSource {
        step: f64,
    },
    MathGain {
        factor: f64,
    },
    MathAccumulate {
        bias: f64,
    },
    MathCounter {
        step: i64,
    },
    AgentSensor {
        base_energy: f64,
        noise_scale: f64,
    },
    AgentBelief {
        curiosity: f64,
        threshold: f64,
        compiled_expr: Option<CompiledExpr>,
        energy_var: VarIndex,
        curiosity_var: VarIndex,
        threshold_var: VarIndex,
    },
    AgentPlanner {
        explore_gain: f64,
        exploit_gain: f64,
        seed_mix: u64,
    },
    LanguageHumanChannel {
        is_order: bool,
        seed_mix: u64,
    },
    LanguageDirectSensor {
        seed_mix: u64,
    },
    LanguageGrounder {
        min_confidence: f64,
    },
    LanguageBelief {
        ask_cost: f64,
        response_probability: f64,
        inspect_cost: f64,
    },
    LanguageGuardedPlanner {
        seed_mix: u64,
    },
}

#[derive(Debug, Clone)]
enum NodeExecState {
    External { state: Vec<PortValue> },
    Observer { counter: u64 },
    Policy { accumulator: f64 },
    MathSource { value: f64 },
    Stateless,
    MathAccumulate { sum: f64 },
    MathCounter { count: i64 },
    AgentSensor { rng_state: u64, tick: u64 },
    AgentBelief { belief_ema: f64 },
    AgentPlanner { step_count: u64 },
    LanguageChannel { tick: u64 },
    LanguageBelief { belief_prob: f64 },
    LanguagePlanner { step_count: u64 },
}

impl BuiltinKernel {
    fn resolve(
        plan: &ExecutablePlan,
        node_idx: NodeIndex,
        seed: MasterSeed,
        external: &KernelRegistry,
    ) -> Result<(Self, NodeExecState), PlanRuntimeError> {
        let op = plan
            .node_operation(node_idx)
            .expect("plan must contain operation for every node");
        let factory = plan
            .factory(op.factory)
            .expect("plan must contain factory for every operation");
        let module = factory.module_id.as_str();
        let operation = factory.operation_id.as_str();

        match (module, operation) {
            ("observer", "observer") => {
                let rate_hz = node_i64(plan, node_idx, "rate_hz").unwrap_or(50).max(1);
                let normalized = node_bool(plan, node_idx, "normalized").unwrap_or(true);
                Ok((
                    Self::Observer {
                        rate_hz,
                        normalized,
                    },
                    NodeExecState::Observer { counter: 0 },
                ))
            }
            ("policy", "policy") => {
                let gain = node_f64(plan, node_idx, "gain").unwrap_or(1.0);
                Ok((
                    Self::Policy { gain },
                    NodeExecState::Policy { accumulator: 0.0 },
                ))
            }
            ("core.math", "source") => {
                let initial = node_f64(plan, node_idx, "initial").unwrap_or(0.0);
                let step = node_f64(plan, node_idx, "step").unwrap_or(1.0);
                Ok((
                    Self::MathSource { step },
                    NodeExecState::MathSource { value: initial },
                ))
            }
            ("core.math", "gain") => {
                let factor = node_f64(plan, node_idx, "factor").unwrap_or(1.0);
                Ok((Self::MathGain { factor }, NodeExecState::Stateless))
            }
            ("core.math", "accumulate") => {
                let bias = node_f64(plan, node_idx, "bias").unwrap_or(0.0);
                Ok((
                    Self::MathAccumulate { bias },
                    NodeExecState::MathAccumulate { sum: 0.0 },
                ))
            }
            ("core.math", "counter") => {
                let start = node_i64(plan, node_idx, "start").unwrap_or(0);
                let step = node_i64(plan, node_idx, "step").unwrap_or(1);
                Ok((
                    Self::MathCounter { step },
                    NodeExecState::MathCounter { count: start },
                ))
            }
            ("agent", "sensor") => {
                let base_energy = node_f64(plan, node_idx, "base_energy").unwrap_or(1.0);
                let noise_scale = node_f64(plan, node_idx, "noise_scale").unwrap_or(0.1);
                let initial_state = splitmix64(seed.as_u64() ^ (node_idx.0 as u64).wrapping_add(1));
                Ok((
                    Self::AgentSensor {
                        base_energy,
                        noise_scale,
                    },
                    NodeExecState::AgentSensor {
                        rng_state: initial_state,
                        tick: 0,
                    },
                ))
            }
            ("agent", "belief") => {
                let curiosity = node_f64(plan, node_idx, "curiosity").unwrap_or(0.6);
                let threshold = node_f64(plan, node_idx, "threshold").unwrap_or(0.4);
                let mut schema = ExprSchema::new();
                let energy_var = schema
                    .register("energy", ExprType::Scalar(ExprUnit::Energy))
                    .map_err(|err| PlanRuntimeError::InvalidNodeConfig {
                        key: "expr_json",
                        reason: err.to_string(),
                    })?;
                let curiosity_var = schema
                    .register("curiosity", ExprType::Scalar(ExprUnit::Ratio))
                    .map_err(|err| PlanRuntimeError::InvalidNodeConfig {
                        key: "expr_json",
                        reason: err.to_string(),
                    })?;
                let threshold_var = schema
                    .register("threshold", ExprType::Scalar(ExprUnit::Energy))
                    .map_err(|err| PlanRuntimeError::InvalidNodeConfig {
                        key: "expr_json",
                        reason: err.to_string(),
                    })?;

                let compiled_expr = if let Some(expr_json) = node_str(plan, node_idx, "expr_json") {
                    let ast =
                        ExprAst::from_json_with_policy(expr_json, UnknownFieldPolicy::Reject)?;
                    Some(CompiledExpr::compile(&ast, &schema).map_err(|err| {
                        PlanRuntimeError::InvalidNodeConfig {
                            key: "expr_json",
                            reason: err.to_string(),
                        }
                    })?)
                } else {
                    None
                };

                Ok((
                    Self::AgentBelief {
                        curiosity,
                        threshold,
                        compiled_expr,
                        energy_var,
                        curiosity_var,
                        threshold_var,
                    },
                    NodeExecState::AgentBelief { belief_ema: 0.0 },
                ))
            }
            ("agent", "planner") => {
                let explore_gain = node_f64(plan, node_idx, "explore_gain").unwrap_or(2.0);
                let exploit_gain = node_f64(plan, node_idx, "exploit_gain").unwrap_or(1.0);
                let seed_mix = splitmix64(seed.as_u64() ^ 0x517c_c1b7_2722_0a95);
                Ok((
                    Self::AgentPlanner {
                        explore_gain,
                        exploit_gain,
                        seed_mix,
                    },
                    NodeExecState::AgentPlanner { step_count: 0 },
                ))
            }
            ("language", "human_channel") => {
                validate_provenance_node(plan, node_idx)?;
                let speech_act = node_str(plan, node_idx, "speech_act").unwrap_or("notice");
                let is_order = speech_act == "order";
                let seed_mix = splitmix64(seed.as_u64() ^ 0x1a2b_3c4d_5e6f_7788);
                Ok((
                    Self::LanguageHumanChannel { is_order, seed_mix },
                    NodeExecState::LanguageChannel { tick: 0 },
                ))
            }
            ("language", "direct_sensor") => {
                validate_provenance_node(plan, node_idx)?;
                let seed_mix = splitmix64(seed.as_u64() ^ 0x9988_7766_5544_3322);
                Ok((
                    Self::LanguageDirectSensor { seed_mix },
                    NodeExecState::LanguageChannel { tick: 0 },
                ))
            }
            ("language", "grounder") => {
                let model_id = node_str(plan, node_idx, "model_id").ok_or(
                    PlanRuntimeError::InvalidNodeConfig {
                        key: "model_id",
                        reason: "missing required model_id on language::grounder".to_owned(),
                    },
                )?;
                validate_available_model(model_id, "model_id")?;

                let source_id = node_str(plan, node_idx, "source_id").ok_or(
                    PlanRuntimeError::InvalidNodeConfig {
                        key: "source_id",
                        reason: "missing required source_id on language::grounder".to_owned(),
                    },
                )?;
                let registered_sources = collect_registered_sources(plan);
                if !registered_sources.contains(&source_id) {
                    return Err(PlanRuntimeError::InvalidNodeConfig {
                        key: "source_id",
                        reason: format!(
                            "incompatible source reference '{source_id}': not registered by any sensor or channel node"
                        ),
                    });
                }

                if let Some(question_json) = node_str(plan, node_idx, "question_json") {
                    let q = EpistemicQuestionSpec::from_json(question_json).map_err(|err| {
                        PlanRuntimeError::InvalidNodeConfig {
                            key: "question_json",
                            reason: err.to_string(),
                        }
                    })?;
                    validate_available_model(q.model_id(), "question_json")?;
                }

                let min_confidence =
                    if let Some(abstention_json) = node_str(plan, node_idx, "abstention_json") {
                        let policy =
                            AbstentionPolicySpec::from_json(abstention_json).map_err(|err| {
                                PlanRuntimeError::InvalidNodeConfig {
                                    key: "abstention_json",
                                    reason: err.to_string(),
                                }
                            })?;
                        policy.min_confidence
                    } else {
                        0.65
                    };

                Ok((
                    Self::LanguageGrounder { min_confidence },
                    NodeExecState::Stateless,
                ))
            }
            ("language", "belief") => {
                let prior_obstacle_prob =
                    node_f64(plan, node_idx, "prior_obstacle_prob").unwrap_or(0.5);
                if !prior_obstacle_prob.is_finite() || !(0.0..=1.0).contains(&prior_obstacle_prob) {
                    return Err(PlanRuntimeError::InvalidNodeConfig {
                        key: "prior_obstacle_prob",
                        reason: format!(
                            "prior_obstacle_prob ({prior_obstacle_prob}) must be finite and within [0.0, 1.0]"
                        ),
                    });
                }

                let (ask_cost, response_probability) =
                    if let Some(ask_json) = node_str(plan, node_idx, "ask_json") {
                        let ask = AskActionSpec::from_json(ask_json).map_err(|err| {
                            PlanRuntimeError::InvalidNodeConfig {
                                key: "ask_json",
                                reason: err.to_string(),
                            }
                        })?;
                        validate_available_model(ask.question.model_id(), "ask_json")?;
                        (ask.comm_cost + ask.step_cost, ask.response_probability)
                    } else {
                        (0.5, 0.85)
                    };

                let inspect_cost = if let Some(inspect_json) =
                    node_str(plan, node_idx, "inspect_json")
                {
                    let inspect = InspectActionSpec::from_json(inspect_json).map_err(|err| {
                        PlanRuntimeError::InvalidNodeConfig {
                            key: "inspect_json",
                            reason: err.to_string(),
                        }
                    })?;
                    let registered_sources = collect_registered_sources(plan);
                    if !registered_sources.contains(&inspect.sensor_id.as_str()) {
                        return Err(PlanRuntimeError::InvalidNodeConfig {
                            key: "inspect_json",
                            reason: format!(
                                "incompatible sensor_id reference '{}': not registered by any sensor or channel node",
                                inspect.sensor_id
                            ),
                        });
                    }
                    inspect.inspect_cost
                } else {
                    0.25
                };

                Ok((
                    Self::LanguageBelief {
                        ask_cost,
                        response_probability,
                        inspect_cost,
                    },
                    NodeExecState::LanguageBelief {
                        belief_prob: prior_obstacle_prob,
                    },
                ))
            }
            ("language", "guarded_planner") => {
                let guard_policy =
                    node_str(plan, node_idx, "guard_policy").unwrap_or("reject_only");
                if guard_policy != "reject_only" && guard_policy != "explicit_fallback" {
                    return Err(PlanRuntimeError::InvalidNodeConfig {
                        key: "guard_policy",
                        reason: format!("unsupported guard_policy '{guard_policy}'"),
                    });
                }
                if node_bool(plan, node_idx, "human_can_mutate_permissions") == Some(true) {
                    return Err(PlanRuntimeError::InvalidNodeConfig {
                        key: "human_can_mutate_permissions",
                        reason: "human messages cannot mutate ActionGuard or PermissionSet"
                            .to_owned(),
                    });
                }
                if node_bool(plan, node_idx, "immutable_permissions") == Some(false) {
                    return Err(PlanRuntimeError::InvalidNodeConfig {
                        key: "immutable_permissions",
                        reason: "PermissionSet must remain immutable".to_owned(),
                    });
                }
                if let Some(guard_json) = node_str(plan, node_idx, "guard_json") {
                    GuardPolicySpec::from_json(guard_json).map_err(|err| {
                        PlanRuntimeError::InvalidNodeConfig {
                            key: "guard_json",
                            reason: err.to_string(),
                        }
                    })?;
                }
                let seed_mix = splitmix64(seed.as_u64() ^ 0xdead_beef_cafe_0001);
                Ok((
                    Self::LanguageGuardedPlanner { seed_mix },
                    NodeExecState::LanguagePlanner { step_count: 0 },
                ))
            }
            _ => match external.kernel(module, operation) {
                Some(kernel) => {
                    let state = kernel.init_state(&ExternalNodeContext {
                        plan,
                        node: node_idx,
                        seed,
                    })?;
                    Ok((
                        Self::External {
                            kernel: Arc::clone(kernel),
                        },
                        NodeExecState::External { state },
                    ))
                }
                None => Err(PlanRuntimeError::UnsupportedOperation {
                    module_id: module.to_owned(),
                    operation_id: operation.to_owned(),
                }),
            },
        }
    }

    fn eval(
        &self,
        state: &mut NodeExecState,
        inputs: &[Option<PortValue>],
        outputs: &mut [Option<PortValue>],
    ) {
        match (self, state) {
            (Self::External { kernel }, NodeExecState::External { state }) => {
                kernel.eval(state, inputs, outputs);
            }
            (
                Self::Observer {
                    rate_hz,
                    normalized,
                },
                NodeExecState::Observer { counter },
            ) => {
                *counter += 1;
                let value = if *normalized {
                    ((*counter % (*rate_hz as u64)) as f64) / (*rate_hz as f64)
                } else {
                    *counter as f64
                };
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(value));
                }
            }
            (Self::Policy { gain }, NodeExecState::Policy { accumulator }) => {
                let obs = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                *accumulator += obs * *gain;
                let action = (*accumulator * 100.0).round() as i64;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::I64(action));
                }
            }
            (Self::MathSource { step }, NodeExecState::MathSource { value }) => {
                let emitted = *value;
                *value += *step;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(emitted));
                }
            }
            (Self::MathGain { factor }, NodeExecState::Stateless) => {
                let input = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(input * *factor));
                }
            }
            (Self::MathAccumulate { bias }, NodeExecState::MathAccumulate { sum }) => {
                let input = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                *sum += input + *bias;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(*sum));
                }
            }
            (Self::MathCounter { step }, NodeExecState::MathCounter { count }) => {
                let emitted = *count;
                *count += *step;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::I64(emitted));
                }
            }
            (
                Self::AgentSensor {
                    base_energy,
                    noise_scale,
                },
                NodeExecState::AgentSensor { rng_state, tick },
            ) => {
                *tick += 1;
                *rng_state = splitmix64(*rng_state ^ *tick);
                let unit_obs = ((*rng_state & 0xFFFF) as f64) / 65535.0;
                let unit_energy = (((*rng_state >> 16) & 0xFFFF) as f64) / 65535.0;
                let observation = unit_obs * (1.0 + *noise_scale);
                let energy = (*base_energy * (0.5 + 0.5 * unit_energy)).max(0.0);
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(observation));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::F64(energy));
                }
            }
            (
                Self::AgentBelief {
                    curiosity,
                    threshold,
                    compiled_expr,
                    energy_var,
                    curiosity_var,
                    threshold_var,
                },
                NodeExecState::AgentBelief { belief_ema },
            ) => {
                let observation = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                let energy = match inputs.get(1).and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                *belief_ema = 0.5 * *belief_ema + 0.5 * (observation * *curiosity + energy);
                let should_explore = if let Some(expr) = compiled_expr {
                    let mut values = [ExprValue::Scalar(0.0); 3];
                    values[energy_var.as_usize()] = ExprValue::Scalar(energy);
                    values[curiosity_var.as_usize()] = ExprValue::Scalar(*curiosity);
                    values[threshold_var.as_usize()] = ExprValue::Scalar(*threshold);
                    BoundedExprEvaluator::eval(expr, &values, 64)
                        .ok()
                        .and_then(ExprValue::as_bool)
                        .unwrap_or(false)
                } else {
                    (energy * *curiosity) > *threshold
                };
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(*belief_ema));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::Bool(should_explore));
                }
            }
            (
                Self::AgentPlanner {
                    explore_gain,
                    exploit_gain,
                    seed_mix,
                },
                NodeExecState::AgentPlanner { step_count },
            ) => {
                *step_count += 1;
                let belief_score = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                let should_explore = match inputs.get(1).and_then(Option::as_ref) {
                    Some(PortValue::Bool(b)) => *b,
                    _ => false,
                };
                let gain = if should_explore {
                    *explore_gain
                } else {
                    *exploit_gain
                };
                let tie_break = ((splitmix64(*seed_mix ^ *step_count) % 5) as i64) - 2;
                let action = ((belief_score * gain * 100.0).round() as i64) + tie_break;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::I64(action));
                }
            }
            (
                Self::LanguageHumanChannel { is_order, seed_mix },
                NodeExecState::LanguageChannel { tick },
            ) => {
                *tick += 1;
                let unit = ((splitmix64(*seed_mix ^ *tick) & 0xFFFF) as f64) / 65535.0;
                let obs = 0.25 + 0.7 * unit;
                let conf = 0.55 + 0.4 * ((unit * 1.3) % 1.0);
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(obs));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::F64(conf));
                }
                if let Some(slot) = outputs.get_mut(2) {
                    *slot = Some(PortValue::Bool(*is_order));
                }
            }
            (Self::LanguageDirectSensor { seed_mix }, NodeExecState::LanguageChannel { tick }) => {
                *tick += 1;
                let unit = ((splitmix64(*seed_mix ^ *tick) & 0xFFFF) as f64) / 65535.0;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(unit));
                }
            }
            (Self::LanguageGrounder { min_confidence }, NodeExecState::Stateless) => {
                let obs = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                let conf = match inputs.get(1).and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.0,
                };
                let abstained = conf < *min_confidence;
                let grounded_prob = if abstained { 0.5 } else { obs.clamp(0.0, 1.0) };
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(grounded_prob));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::Bool(abstained));
                }
            }
            (
                Self::LanguageBelief {
                    ask_cost,
                    response_probability,
                    inspect_cost,
                },
                NodeExecState::LanguageBelief { belief_prob },
            ) => {
                let grounded_prob = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.5,
                };
                let abstained = match inputs.get(1).and_then(Option::as_ref) {
                    Some(PortValue::Bool(b)) => *b,
                    _ => false,
                };
                let direct_obs = match inputs.get(2).and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.5,
                };
                *belief_prob = if abstained {
                    0.5 * *belief_prob + 0.5 * direct_obs
                } else {
                    0.4 * *belief_prob + 0.3 * grounded_prob + 0.3 * direct_obs
                };
                let uncertainty = 1.0 - (2.0 * (*belief_prob - 0.5)).abs();
                let ask_net_value = uncertainty * *response_probability - *ask_cost;
                let inspect_net_value = uncertainty - *inspect_cost;
                let should_inspect = inspect_net_value > 0.0 && inspect_net_value >= ask_net_value;
                let should_ask = !should_inspect && ask_net_value > 0.0;
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::F64(*belief_prob));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::Bool(should_ask));
                }
                if let Some(slot) = outputs.get_mut(2) {
                    *slot = Some(PortValue::Bool(should_inspect));
                }
            }
            (
                Self::LanguageGuardedPlanner { seed_mix },
                NodeExecState::LanguagePlanner { step_count },
            ) => {
                *step_count += 1;
                let belief_prob = match inputs.first().and_then(Option::as_ref) {
                    Some(PortValue::F64(v)) => *v,
                    _ => 0.5,
                };
                let should_ask = match inputs.get(1).and_then(Option::as_ref) {
                    Some(PortValue::Bool(b)) => *b,
                    _ => false,
                };
                let should_inspect = match inputs.get(2).and_then(Option::as_ref) {
                    Some(PortValue::Bool(b)) => *b,
                    _ => false,
                };
                let is_order = match inputs.get(3).and_then(Option::as_ref) {
                    Some(PortValue::Bool(b)) => *b,
                    _ => false,
                };
                let guard_rejected = is_order;
                let action = if guard_rejected {
                    0
                } else if should_inspect {
                    1
                } else if should_ask {
                    2
                } else if belief_prob > 0.5 {
                    3
                } else {
                    ((splitmix64(*seed_mix ^ *step_count) % 3) as i64) + 1
                };
                if let Some(slot) = outputs.get_mut(0) {
                    *slot = Some(PortValue::I64(action));
                }
                if let Some(slot) = outputs.get_mut(1) {
                    *slot = Some(PortValue::Bool(guard_rejected));
                }
            }
            _ => {}
        }
    }
}

fn validate_available_model(model_id: &str, key: &'static str) -> Result<(), PlanRuntimeError> {
    if !AVAILABLE_EPISTEMIC_MODELS.contains(&model_id) {
        return Err(PlanRuntimeError::InvalidNodeConfig {
            key,
            reason: format!(
                "unavailable model '{model_id}'; available models: {:?}",
                AVAILABLE_EPISTEMIC_MODELS
            ),
        });
    }
    Ok(())
}

fn validate_provenance_node(
    plan: &ExecutablePlan,
    node_idx: NodeIndex,
) -> Result<(), PlanRuntimeError> {
    let source_id =
        node_str(plan, node_idx, "source_id").ok_or(PlanRuntimeError::InvalidNodeConfig {
            key: "source_id",
            reason: "missing required source_id".to_owned(),
        })?;
    if source_id.trim().is_empty() {
        return Err(PlanRuntimeError::InvalidNodeConfig {
            key: "source_id",
            reason: "source_id cannot be empty".to_owned(),
        });
    }
    let sensor_id =
        node_i64(plan, node_idx, "sensor_id").ok_or(PlanRuntimeError::InvalidNodeConfig {
            key: "sensor_id",
            reason: "missing required sensor_id".to_owned(),
        })?;
    if sensor_id <= 0 {
        return Err(PlanRuntimeError::InvalidNodeConfig {
            key: "sensor_id",
            reason: format!("sensor_id ({sensor_id}) must be > 0"),
        });
    }
    let evidence_ref =
        node_i64(plan, node_idx, "evidence_ref").ok_or(PlanRuntimeError::InvalidNodeConfig {
            key: "evidence_ref",
            reason: "missing required evidence_ref".to_owned(),
        })?;
    if evidence_ref <= 0 {
        return Err(PlanRuntimeError::InvalidNodeConfig {
            key: "evidence_ref",
            reason: format!("evidence_ref ({evidence_ref}) must be > 0"),
        });
    }
    if let Some(provenance_json) = node_str(plan, node_idx, "provenance_json") {
        ProvenanceSpec::from_json(provenance_json).map_err(|err| {
            PlanRuntimeError::InvalidNodeConfig {
                key: "provenance_json",
                reason: err.to_string(),
            }
        })?;
    }
    Ok(())
}

fn collect_registered_sources(plan: &ExecutablePlan) -> Vec<&str> {
    let mut sources = Vec::new();
    for idx in 0..plan.node_count() {
        let node_idx = NodeIndex(idx);
        if let Some(op) = plan.node_operation(node_idx)
            && let Some(factory) = plan.factory(op.factory)
            && factory.module_id.as_str() == "language"
            && matches!(
                factory.operation_id.as_str(),
                "human_channel" | "direct_sensor"
            )
            && let Some(source_id) = node_str(plan, node_idx, "source_id")
        {
            sources.push(source_id);
        }
    }
    sources
}

fn node_f64(plan: &ExecutablePlan, node: NodeIndex, key: &str) -> Option<f64> {
    let idx = plan.node_constant_index(node, key)?;
    match plan.constant(idx)? {
        ConstantValue::F64(v) => Some(*v),
        ConstantValue::I64(v) => Some(*v as f64),
        _ => None,
    }
}

fn node_i64(plan: &ExecutablePlan, node: NodeIndex, key: &str) -> Option<i64> {
    let idx = plan.node_constant_index(node, key)?;
    match plan.constant(idx)? {
        ConstantValue::I64(v) => Some(*v),
        _ => None,
    }
}

fn node_bool(plan: &ExecutablePlan, node: NodeIndex, key: &str) -> Option<bool> {
    let idx = plan.node_constant_index(node, key)?;
    match plan.constant(idx)? {
        ConstantValue::Bool(v) => Some(*v),
        _ => None,
    }
}

fn node_str<'a>(plan: &'a ExecutablePlan, node: NodeIndex, key: &str) -> Option<&'a str> {
    let idx = plan.node_constant_index(node, key)?;
    match plan.constant(idx)? {
        ConstantValue::String(s) => Some(s.as_str()),
        _ => None,
    }
}

/// Resumen determinista de una ejecución de `N` ticks sobre [`PlanInstance`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlanRunSummary {
    /// Identificador de la especificación ejecutada.
    pub spec_id: String,
    /// Huella semántica canónica del [`ExecutablePlan`].
    pub semantic_fingerprint: u64,
    /// Número total de ticks ejecutados en la instancia.
    pub ticks_executed: u64,
    /// Hash determinista de la traza de ejecución.
    pub trace_hash: u64,
    /// Representación hexadecimal del hash de traza.
    pub trace_hash_hex: String,
    /// Salidas finales de cada nodo indexadas por [`NodeIndex`] y [`PortIndex`].
    pub final_outputs: Vec<Vec<Option<PortValue>>>,
}

/// Instancia mutable e independiente de un [`ExecutablePlan`] inmutable.
///
/// Cada [`PlanInstance`] posee sus propios buffers de puertos, retardos, estados
/// internos de nodo, reloj lógico y [`TraceHasher`]. Dos instancias que consumen
/// el mismo `&ExecutablePlan` (o `Arc<ExecutablePlan>`) nunca comparten estado mutable.
#[derive(Debug, Clone)]
pub struct PlanInstance {
    plan_fingerprint: u64,
    dt: StepDt,
    clock: LogicalClock,
    graph_runtime: PlanRuntime<PortValue>,
    kernels: Vec<BuiltinKernel>,
    node_states: Vec<NodeExecState>,
    hasher: TraceHasher,
}

impl PlanInstance {
    /// Crea una nueva instancia mutable a partir de un [`ExecutablePlan`] inmutable
    /// usando una semilla explícita [`MasterSeed`].
    pub fn with_seed(plan: &ExecutablePlan, seed: MasterSeed) -> Result<Self, PlanRuntimeError> {
        let dt_secs = Self::dt_secs_of(plan)?;
        let dt = StepDt::from_secs_f64(dt_secs)?;
        Self::with_dt_and_seed(plan, dt, seed)
    }

    fn dt_secs_of(plan: &ExecutablePlan) -> Result<f64, PlanRuntimeError> {
        Ok(match plan.system_constant_index("dt_secs") {
            Some(idx) => match plan.constant(idx) {
                Some(ConstantValue::F64(v)) => *v,
                Some(_) => {
                    return Err(PlanRuntimeError::InvalidSystemConfig {
                        key: "dt_secs",
                        reason: "expected finite f64".to_owned(),
                    });
                }
                None => DEFAULT_DT_SECS,
            },
            None => DEFAULT_DT_SECS,
        })
    }

    /// Crea una nueva instancia mutable a partir de un [`ExecutablePlan`] inmutable.
    ///
    /// Lee `dt_secs` de la configuración del sistema si está presente; de lo contrario
    /// usa `0.02` segundos.
    pub fn new(plan: &ExecutablePlan) -> Result<Self, PlanRuntimeError> {
        Self::with_seed(plan, MasterSeed::new(DEFAULT_SEED))
    }

    /// Crea una nueva instancia mutable con un [`StepDt`] explícito.
    pub fn with_dt(plan: &ExecutablePlan, dt: StepDt) -> Result<Self, PlanRuntimeError> {
        Self::with_dt_and_seed(plan, dt, MasterSeed::new(DEFAULT_SEED))
    }

    /// Crea una nueva instancia mutable con un [`StepDt`] y [`MasterSeed`] explícitos.
    pub fn with_dt_and_seed(
        plan: &ExecutablePlan,
        dt: StepDt,
        seed: MasterSeed,
    ) -> Result<Self, PlanRuntimeError> {
        Self::with_kernels(plan, dt, seed, &KernelRegistry::new())
    }

    /// Crea una instancia resolviendo operaciones no integradas mediante un [`KernelRegistry`].
    ///
    /// Una operación sin kernel integrado ni externo produce
    /// [`PlanRuntimeError::UnsupportedOperation`].
    pub fn with_kernels(
        plan: &ExecutablePlan,
        dt: StepDt,
        seed: MasterSeed,
        external: &KernelRegistry,
    ) -> Result<Self, PlanRuntimeError> {
        let node_count = plan.node_count();
        let mut kernels = Vec::with_capacity(node_count);
        let mut node_states = Vec::with_capacity(node_count);

        for idx in 0..node_count {
            let (kernel, state) = BuiltinKernel::resolve(plan, NodeIndex(idx), seed, external)?;
            kernels.push(kernel);
            node_states.push(state);
        }

        let mut hasher = TraceHasher::new();
        hasher.write_u64(plan.semantic_fingerprint());
        hasher.write_step_dt(dt);

        Ok(Self {
            plan_fingerprint: plan.semantic_fingerprint(),
            dt,
            clock: LogicalClock::new(dt),
            graph_runtime: PlanRuntime::new(plan.graph_plan()),
            kernels,
            node_states,
            hasher,
        })
    }

    /// Devuelve el paso temporal configurado para esta instancia.
    pub fn dt(&self) -> StepDt {
        self.dt
    }

    /// Devuelve el tick lógico actual de esta instancia.
    pub fn tick(&self) -> Tick {
        self.clock.tick()
    }

    /// Devuelve el hash de traza acumulado hasta el tick actual.
    pub fn trace_hash(&self) -> u64 {
        self.hasher.clone().finish()
    }

    /// Devuelve el hash de traza en hexadecimal de 16 dígitos.
    pub fn trace_hash_hex(&self) -> String {
        self.hasher.clone().finish_hex()
    }

    /// Inyecta una entrada externa en un puerto concreto por índice directo.
    pub fn set_input(&mut self, node: NodeIndex, port: PortIndex, value: Option<PortValue>) {
        self.graph_runtime.set_input(node, port, value);
    }

    /// Lee la salida actual de un puerto por índice directo.
    pub fn output(&self, node: NodeIndex, port: PortIndex) -> Option<&PortValue> {
        self.graph_runtime.output(node, port)
    }

    /// Ejecuta un único tick determinista usando únicamente índices resueltos del plan.
    pub fn step(&mut self, plan: &ExecutablePlan) -> Result<Tick, PlanRuntimeError> {
        if plan.semantic_fingerprint() != self.plan_fingerprint
            || plan.node_count() != self.node_states.len()
        {
            return Err(PlanRuntimeError::PlanMismatch {
                expected_fingerprint: self.plan_fingerprint,
                found_fingerprint: plan.semantic_fingerprint(),
            });
        }

        let (tick, elapsed) = self.clock.advance()?;
        let kernels = &self.kernels;
        let node_states = &mut self.node_states;

        self.graph_runtime.execute_tick(
            plan.graph_plan(),
            tick,
            |node_idx, _tick, inputs, outputs| {
                kernels[node_idx.0].eval(&mut node_states[node_idx.0], inputs, outputs);
            },
        );

        self.hasher.write_tick(tick);
        self.hasher.write_duration(elapsed);
        for &node_idx in plan.execution_order() {
            self.hasher.write_u64(node_idx.0 as u64);
            let output_count = plan
                .node_operation(node_idx)
                .map(|op| op.output_count)
                .unwrap_or(0);
            self.hasher.write_u64(output_count as u64);
            for port_idx in 0..output_count {
                match self.graph_runtime.output(node_idx, PortIndex(port_idx)) {
                    Some(val) => {
                        self.hasher.write_u8(1);
                        val.write_to_hasher(&mut self.hasher);
                    }
                    None => {
                        self.hasher.write_u8(0);
                    }
                }
            }
        }

        Ok(tick)
    }

    /// Ejecuta `ticks` pasos consecutivos y devuelve el resumen determinista de la ejecución.
    pub fn run_ticks(
        &mut self,
        plan: &ExecutablePlan,
        ticks: u64,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        for _ in 0..ticks {
            self.step(plan)?;
        }

        let mut final_outputs = Vec::with_capacity(plan.node_count());
        for idx in 0..plan.node_count() {
            let node_idx = NodeIndex(idx);
            let output_count = plan
                .node_operation(node_idx)
                .map(|op| op.output_count)
                .unwrap_or(0);
            let mut ports = Vec::with_capacity(output_count);
            for port_idx in 0..output_count {
                ports.push(
                    self.graph_runtime
                        .output(node_idx, PortIndex(port_idx))
                        .cloned(),
                );
            }
            final_outputs.push(ports);
        }

        Ok(PlanRunSummary {
            spec_id: plan.spec_id().to_owned(),
            semantic_fingerprint: plan.semantic_fingerprint(),
            ticks_executed: self.clock.tick().as_u64(),
            trace_hash: self.trace_hash(),
            trace_hash_hex: self.trace_hash_hex(),
            final_outputs,
        })
    }
}

/// Motor sin frontend para compilar y ejecutar [`SystemSpec`] sobre [`ExecutablePlan`].
///
/// Por defecto solo conoce los kernels integrados; [`PlanRuntimeEngine::with_external_kernels`]
/// añade operaciones personalizadas registradas en un [`KernelRegistry`].
#[derive(Debug, Default, Clone)]
pub struct PlanRuntimeEngine {
    kernels: KernelRegistry,
}

impl PlanRuntimeEngine {
    /// Crea una nueva instancia del motor de ejecución de planes.
    pub fn new() -> Self {
        Self::default()
    }

    /// Crea un motor que además resuelve las operaciones externas de `kernels`.
    pub fn with_external_kernels(kernels: KernelRegistry) -> Self {
        Self { kernels }
    }

    /// Registro de kernels externos de este motor.
    pub fn kernel_registry(&self) -> &KernelRegistry {
        &self.kernels
    }

    /// Catálogo efectivo (integrado + externo) usado para compilar y cargar artefactos.
    pub fn catalog(&self) -> CapabilityCatalog {
        self.kernels.catalog()
    }

    /// Carga un [`ExecutablePlan`] desde un [`CompiledPlanArtifact`] ya parseado,
    /// verificando el catálogo efectivo del motor.
    pub fn load_artifact(
        &self,
        artifact: &CompiledPlanArtifact,
    ) -> Result<ExecutablePlan, PlanRuntimeError> {
        Ok(artifact.load_executable_plan(&self.catalog())?)
    }

    /// Carga un artefacto y ejecuta `ticks` pasos sobre una instancia nueva.
    pub fn run_artifact(
        &self,
        artifact: &CompiledPlanArtifact,
        ticks: u64,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.load_artifact(artifact)?;
        self.run_plan(&plan, ticks)
    }

    /// Compila una [`SystemSpec`] construida vía API Rust contra el catálogo integrado.
    pub fn compile_spec(&self, spec: &SystemSpec) -> Result<ExecutablePlan, PlanRuntimeError> {
        let catalog = self.catalog();
        Ok(SpecCompiler::compile(
            spec,
            &catalog,
            TemporalProfile::BoundedTick,
        )?)
    }

    /// Parsea y compila una especificación JSON contra el catálogo integrado.
    pub fn compile_json(&self, json: &str) -> Result<ExecutablePlan, PlanRuntimeError> {
        let spec = SystemSpec::from_json(json)?;
        self.compile_spec(&spec)
    }

    /// Instancia un [`PlanInstance`] aislado sobre un `&ExecutablePlan` compartido.
    pub fn instantiate(&self, plan: &ExecutablePlan) -> Result<PlanInstance, PlanRuntimeError> {
        self.instantiate_with_seed(plan, MasterSeed::new(DEFAULT_SEED))
    }

    /// Instancia un [`PlanInstance`] aislado con una semilla [`MasterSeed`] explícita.
    pub fn instantiate_with_seed(
        &self,
        plan: &ExecutablePlan,
        seed: MasterSeed,
    ) -> Result<PlanInstance, PlanRuntimeError> {
        let dt_secs = PlanInstance::dt_secs_of(plan)?;
        let dt = StepDt::from_secs_f64(dt_secs)?;
        PlanInstance::with_kernels(plan, dt, seed, &self.kernels)
    }

    /// Ejecuta `ticks` sobre una nueva instancia del plan dado.
    pub fn run_plan(
        &self,
        plan: &ExecutablePlan,
        ticks: u64,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let mut instance = self.instantiate(plan)?;
        instance.run_ticks(plan, ticks)
    }

    /// Ejecuta `ticks` sobre una nueva instancia del plan dado con una semilla [`MasterSeed`] explícita.
    pub fn run_plan_with_seed(
        &self,
        plan: &ExecutablePlan,
        ticks: u64,
        seed: MasterSeed,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let mut instance = self.instantiate_with_seed(plan, seed)?;
        instance.run_ticks(plan, ticks)
    }

    /// Compila y ejecuta una [`SystemSpec`] construida en Rust durante `ticks` pasos.
    pub fn run_spec(
        &self,
        spec: &SystemSpec,
        ticks: u64,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.compile_spec(spec)?;
        self.run_plan(&plan, ticks)
    }

    /// Compila y ejecuta una [`SystemSpec`] construida en Rust durante `ticks` pasos con semilla explícita.
    pub fn run_spec_with_seed(
        &self,
        spec: &SystemSpec,
        ticks: u64,
        seed: MasterSeed,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.compile_spec(spec)?;
        self.run_plan_with_seed(&plan, ticks, seed)
    }

    /// Parsea, compila y ejecuta una especificación JSON durante `ticks` pasos.
    pub fn run_json(&self, json: &str, ticks: u64) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.compile_json(json)?;
        self.run_plan(&plan, ticks)
    }

    /// Parsea, compila y ejecuta una especificación JSON durante `ticks` pasos con semilla explícita.
    pub fn run_json_with_seed(
        &self,
        json: &str,
        ticks: u64,
        seed: MasterSeed,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.compile_json(json)?;
        self.run_plan_with_seed(&plan, ticks, seed)
    }

    /// Compila una [`SystemSpec`] a un [`CompiledPlanArtifact`] portable contra el catálogo integrado.
    pub fn compile_artifact(
        &self,
        spec: &SystemSpec,
    ) -> Result<CompiledPlanArtifact, PlanRuntimeError> {
        let catalog = self.catalog();
        Ok(CompiledPlanArtifact::compile(
            spec,
            &catalog,
            TemporalProfile::BoundedTick,
        )?)
    }

    /// Verifica integridad/versión/catálogo y carga un [`ExecutablePlan`] desde JSON de artefacto.
    pub fn load_artifact_json(&self, json: &str) -> Result<ExecutablePlan, PlanRuntimeError> {
        let catalog = self.catalog();
        Ok(load_plan_from_artifact_json(json, &catalog)?)
    }

    /// Carga un artefacto JSON compilado y lo ejecuta durante `ticks` pasos sin JVM.
    pub fn run_artifact_json(
        &self,
        json: &str,
        ticks: u64,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.load_artifact_json(json)?;
        self.run_plan(&plan, ticks)
    }

    /// Carga un artefacto JSON compilado y lo ejecuta durante `ticks` pasos con semilla explícita sin JVM.
    pub fn run_artifact_json_with_seed(
        &self,
        json: &str,
        ticks: u64,
        seed: MasterSeed,
    ) -> Result<PlanRunSummary, PlanRuntimeError> {
        let plan = self.load_artifact_json(json)?;
        self.run_plan_with_seed(&plan, ticks, seed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use gogamen_core::NodeId;
    use gogamen_spec::{
        SpecConfigValue, SystemConnectionSpec, SystemNodeSpec, canonical_two_node_fixture,
        canonical_two_node_fixture_json,
    };

    use super::*;

    #[test]
    fn shared_executable_plan_runs_multiple_instances_without_mutable_state_leakage() {
        let engine = PlanRuntimeEngine::new();
        let plan = Arc::new(engine.compile_spec(&canonical_two_node_fixture()).unwrap());

        let mut instance_a = PlanInstance::new(&plan).unwrap();
        let mut instance_b = PlanInstance::new(&plan).unwrap();

        // Avanzamos solo instance_a 5 ticks; instance_b debe permanecer intacta en tick 0.
        let summary_a_5 = instance_a.run_ticks(&plan, 5).unwrap();
        assert_eq!(summary_a_5.ticks_executed, 5);
        assert_eq!(instance_b.tick(), Tick::ZERO);
        assert_ne!(instance_a.trace_hash(), instance_b.trace_hash());

        // Avanzamos instance_b 5 ticks y debe coincidir exactamente con summary_a_5.
        let summary_b_5 = instance_b.run_ticks(&plan, 5).unwrap();
        assert_eq!(summary_a_5.trace_hash, summary_b_5.trace_hash);
        assert_eq!(summary_a_5.final_outputs, summary_b_5.final_outputs);

        // Avanzamos solo instance_a 3 ticks más: instance_b no se ve afectada.
        let summary_a_8 = instance_a.run_ticks(&plan, 3).unwrap();
        assert_eq!(summary_a_8.ticks_executed, 8);
        assert_eq!(instance_b.tick(), Tick::new(5));
        assert_ne!(summary_a_8.trace_hash, summary_b_5.trace_hash);
    }

    #[test]
    fn json_and_rust_api_produce_identical_fingerprint_and_trace_hash() {
        let engine = PlanRuntimeEngine::new();
        let rust_spec = canonical_two_node_fixture();
        let json_str = canonical_two_node_fixture_json();

        let rust_summary = engine.run_spec(&rust_spec, 10).unwrap();
        let json_summary = engine.run_json(&json_str, 10).unwrap();

        assert_eq!(
            rust_summary.semantic_fingerprint,
            json_summary.semantic_fingerprint
        );
        assert_eq!(rust_summary.trace_hash, json_summary.trace_hash);
        assert_eq!(rust_summary.final_outputs, json_summary.final_outputs);
    }

    #[test]
    fn conventional_source_gain_accumulate_pipeline_runs_without_agents_or_ai() {
        let spec = SystemSpec::new("conventional_pipeline_v0")
            .with_config("dt_secs", SpecConfigValue::F64(0.1))
            .with_node(
                SystemNodeSpec::new("src", "core.math::source")
                    .with_output("out", PortType::F64)
                    .with_config("initial", SpecConfigValue::F64(1.0))
                    .with_config("step", SpecConfigValue::F64(1.0)),
            )
            .with_node(
                SystemNodeSpec::new("scale", "core.math::gain")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64)
                    .with_config("factor", SpecConfigValue::F64(2.0)),
            )
            .with_node(
                SystemNodeSpec::new("acc", "core.math::accumulate")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64)
                    .with_config("bias", SpecConfigValue::F64(0.5)),
            )
            .with_connection(SystemConnectionSpec::instant("src", "out", "scale", "in"))
            .with_connection(SystemConnectionSpec::instant("scale", "out", "acc", "in"));

        let engine = PlanRuntimeEngine::new();
        let plan = engine.compile_spec(&spec).unwrap();
        let summary = engine.run_plan(&plan, 3).unwrap();

        // Tick 1: src=1.0 -> scale=2.0 -> acc=2.5
        // Tick 2: src=2.0 -> scale=4.0 -> acc=2.5+4.5=7.0
        // Tick 3: src=3.0 -> scale=6.0 -> acc=7.0+6.5=13.5
        let acc_idx = plan.node_index(&NodeId::new("acc")).unwrap();
        assert_eq!(
            summary.final_outputs[acc_idx.0][0],
            Some(PortValue::F64(13.5))
        );
    }

    #[test]
    fn compiled_artifact_executes_identically_to_direct_spec_without_jvm() {
        let engine = PlanRuntimeEngine::new();
        let spec = gogamen_spec::canonical_two_node_fixture();

        let direct_summary = engine.run_spec(&spec, 25).expect("direct run_spec");
        let artifact = engine.compile_artifact(&spec).expect("compile_artifact");
        let artifact_json = artifact.to_canonical_json().expect("canonical json");
        let artifact_summary = engine
            .run_artifact_json(&artifact_json, 25)
            .expect("run_artifact_json");

        assert_eq!(artifact_summary, direct_summary);
    }

    #[derive(Debug)]
    struct ScaleBiasKernel;

    impl ExternalKernel for ScaleBiasKernel {
        fn init_state(
            &self,
            node: &ExternalNodeContext<'_>,
        ) -> Result<Vec<PortValue>, PlanRuntimeError> {
            Ok(vec![PortValue::F64(node.config_f64("bias").unwrap_or(0.0))])
        }

        fn eval(
            &self,
            state: &mut Vec<PortValue>,
            inputs: &[Option<PortValue>],
            outputs: &mut [Option<PortValue>],
        ) {
            let bias = match state.first() {
                Some(PortValue::F64(b)) => *b,
                _ => 0.0,
            };
            let x = match inputs.first().and_then(Option::as_ref) {
                Some(PortValue::F64(v)) => *v,
                _ => 0.0,
            };
            outputs[0] = Some(PortValue::F64(x * 3.0 + bias));
        }
    }

    fn custom_registry() -> KernelRegistry {
        let mut registry = KernelRegistry::new();
        registry
            .register_module(
                ModuleManifest::new(
                    "ext.demo",
                    ModuleOrigin::ThirdParty,
                    CapabilityVersion::new(1, 0, 0),
                )
                .with_operation(
                    OperationSpec::new("scale_bias", CapabilityVersion::new(1, 0, 0))
                        .with_input("in", PortType::F64)
                        .with_output("out", PortType::F64)
                        .with_effect(EffectKind::Pure)
                        .with_temporal_profile(TemporalProfile::BoundedTick),
                ),
            )
            .unwrap();
        registry
            .bind_kernel("ext.demo", "scale_bias", Arc::new(ScaleBiasKernel))
            .unwrap();
        registry
    }

    fn custom_spec() -> SystemSpec {
        SystemSpec::new("custom_op_v0")
            .with_config("dt_secs", SpecConfigValue::F64(0.1))
            .with_node(
                SystemNodeSpec::new("src", "core.math::source")
                    .with_output("out", PortType::F64)
                    .with_config("initial", SpecConfigValue::F64(1.0))
                    .with_config("step", SpecConfigValue::F64(1.0)),
            )
            .with_node(
                SystemNodeSpec::new("custom", "ext.demo::scale_bias")
                    .with_input("in", PortType::F64)
                    .with_output("out", PortType::F64)
                    .with_config("bias", SpecConfigValue::F64(0.5)),
            )
            .with_connection(SystemConnectionSpec::instant("src", "out", "custom", "in"))
    }

    #[test]
    fn compiled_artifact_with_external_kernel_executes_end_to_end() {
        let engine = PlanRuntimeEngine::with_external_kernels(custom_registry());
        let spec = custom_spec();
        let artifact = engine.compile_artifact(&spec).expect("compile custom op");
        let json = artifact.to_canonical_json().expect("canonical json");

        // Ruta JSON y ruta de artefacto en memoria producen el mismo resultado.
        let from_json = engine.run_artifact_json(&json, 3).expect("run json");
        let from_artifact = engine.run_artifact(&artifact, 3).expect("run artifact");
        assert_eq!(from_json, from_artifact);

        // Tick 3: src=3.0 -> 3.0 * 3 + 0.5 = 9.5
        let plan = engine.load_artifact(&artifact).unwrap();
        let idx = plan.node_index(&NodeId::new("custom")).unwrap();
        assert_eq!(
            from_artifact.final_outputs[idx.0][0],
            Some(PortValue::F64(9.5))
        );

        // Determinismo entre instancias.
        let again = engine.run_artifact(&artifact, 3).unwrap();
        assert_eq!(again.trace_hash, from_artifact.trace_hash);
    }

    #[test]
    fn unregistered_custom_operation_yields_typed_unsupported_error() {
        let engine = PlanRuntimeEngine::with_external_kernels(custom_registry());
        let plan = engine.compile_spec(&custom_spec()).unwrap();

        // Motor sin kernels externos: el plan existe, pero no se puede instanciar.
        let bare = PlanRuntimeEngine::new();
        let err = bare.instantiate(&plan).unwrap_err();
        assert_eq!(
            err,
            PlanRuntimeError::UnsupportedOperation {
                module_id: "ext.demo".to_owned(),
                operation_id: "scale_bias".to_owned(),
            }
        );

        // Módulo declarado en catálogo pero sin kernel enlazado: mismo error tipado.
        let mut declared_only = KernelRegistry::new();
        declared_only
            .register_module(
                custom_registry()
                    .catalog()
                    .module(&gogamen_core::ModuleId::new("ext.demo"))
                    .unwrap()
                    .clone(),
            )
            .unwrap();
        let engine = PlanRuntimeEngine::with_external_kernels(declared_only);
        assert!(matches!(
            engine.instantiate(&plan),
            Err(PlanRuntimeError::UnsupportedOperation { .. })
        ));
    }

    #[test]
    fn kernel_registry_rejects_builtin_collisions_unknown_and_duplicate_bindings() {
        let mut registry = custom_registry();
        let builtin = ModuleManifest::new(
            "core.math",
            ModuleOrigin::ThirdParty,
            CapabilityVersion::new(1, 0, 0),
        );
        assert!(matches!(
            registry.register_module(builtin),
            Err(KernelRegistryError::DuplicateModule { .. })
        ));
        assert!(matches!(
            registry.bind_kernel("ext.demo", "nope", Arc::new(ScaleBiasKernel)),
            Err(KernelRegistryError::UnknownOperation { .. })
        ));
        assert!(matches!(
            registry.bind_kernel("ext.demo", "scale_bias", Arc::new(ScaleBiasKernel)),
            Err(KernelRegistryError::DuplicateKernel { .. })
        ));
        assert_eq!(registry.len(), 1);
    }
}
