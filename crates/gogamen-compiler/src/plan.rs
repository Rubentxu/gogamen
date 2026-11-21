//! Plan ejecutable inmutable con índices resueltos y huella semántica canónica.

use std::collections::BTreeMap;

use gogamen_core::{
    CapabilityVersion, CompiledDelayedTransfer, CompiledPlan, CompiledTransfer, ModuleId,
    ModuleOrigin, NodeId, NodeIndex, OperationId, PortDirection, PortId, PortIndex,
    TemporalProfile,
};

/// Versión fija del formato de [`ExecutablePlan`] emitida por el compilador.
pub const COMPILER_PLAN_VERSION: &str = "gogamen.executable_plan.v0";

/// Índice numérico directo de una operación resuelta en [`ExecutablePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperationIndex(pub usize);

impl OperationIndex {
    /// Crea un índice de operación.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Devuelve el índice subyacente.
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// Índice numérico directo de una fábrica de módulo/operación resuelta en [`ExecutablePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FactoryIndex(pub usize);

impl FactoryIndex {
    /// Crea un índice de fábrica.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Devuelve el índice subyacente.
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// Índice numérico directo de una constante escalar resuelta en [`ExecutablePlan`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantIndex(pub usize);

impl ConstantIndex {
    /// Crea un índice de constante.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Devuelve el índice subyacente.
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

/// Valor constante escalar pre-validado e inmutable dentro de [`ExecutablePlan`].
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    /// Valor booleano.
    Bool(bool),
    /// Entero con signo de 64 bits.
    I64(i64),
    /// Punto flotante finito normalizado (`-0.0` normalizado a `+0.0`).
    F64(f64),
    /// Cadena de texto literal.
    String(String),
}

/// Fábrica resuelta y validada contra [`gogamen_core::CapabilityCatalog`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedFactory {
    /// Índice único de la fábrica dentro del plan.
    pub index: FactoryIndex,
    /// Identificador del módulo proveedor.
    pub module_id: ModuleId,
    /// Identificador de la operación dentro del módulo.
    pub operation_id: OperationId,
    /// Versión resuelta de la operación.
    pub version: CapabilityVersion,
    /// Origen del módulo (`Builtin` o `ThirdParty`).
    pub origin: ModuleOrigin,
    /// Perfil temporal declarado por la operación.
    pub temporal_profile: TemporalProfile,
}

/// Operación resuelta para un nodo concreto dentro de [`ExecutablePlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedOperation {
    /// Índice de esta operación en el plan.
    pub index: OperationIndex,
    /// Índice del nodo en [`CompiledPlan`].
    pub node_index: NodeIndex,
    /// Índice de la fábrica que instancia esta operación.
    pub factory: FactoryIndex,
    /// Perfil temporal efectivo bajo el cual se validó la operación.
    pub temporal_profile: TemporalProfile,
    /// Número de puertos de entrada del nodo.
    pub input_count: usize,
    /// Número de puertos de salida del nodo.
    pub output_count: usize,
    /// Índices de constantes asociadas al nodo en orden canónico de clave.
    pub constants: Vec<ConstantIndex>,
}

/// Contrato opcional de inicialización de recursos, fábricas y actuadores a partir
/// de un [`ExecutablePlan`] ya validado.
///
/// Su separación respecto a [`crate::SpecCompiler::compile`] garantiza que ninguna
/// especificación inválida llegue a inicializar recursos ni actuadores.
pub trait ResourceInitializer {
    /// Tipo de recurso o fábrica instanciada.
    type Resource;
    /// Tipo de actuador o manejador de operación instanciado.
    type Actuator;
    /// Error de inicialización del backend concreto.
    type Error;

    /// Inicializa una fábrica validada del plan.
    fn init_factory(&mut self, factory: &ResolvedFactory) -> Result<Self::Resource, Self::Error>;

    /// Inicializa un actuador/operación validada del plan.
    fn init_actuator(
        &mut self,
        operation: &ResolvedOperation,
        factory: &ResolvedFactory,
    ) -> Result<Self::Actuator, Self::Error>;
}

/// Conjunto de recursos y actuadores instanciados fuera del [`ExecutablePlan`] inmutable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstantiatedPlan<R, A> {
    /// Recursos inicializados indexados por [`FactoryIndex`].
    pub resources: Vec<R>,
    /// Actuadores inicializados indexados por [`OperationIndex`].
    pub actuators: Vec<A>,
}

/// Plan ejecutable inmutable compilado desde [`gogamen_spec::SystemSpec`].
///
/// Contiene exclusivamente índices resueltos, operaciones validadas y constantes
/// normalizadas, separado de cualquier estado mutable de ejecución (`step` no
/// requiere parseo JSON ni resolución de nombres por cadena).
#[derive(Debug, Clone)]
pub struct ExecutablePlan {
    pub(crate) schema_version: String,
    pub(crate) compiler_version: String,
    pub(crate) spec_id: String,
    pub(crate) temporal_profile: TemporalProfile,
    pub(crate) semantic_fingerprint: u64,
    pub(crate) graph_plan: CompiledPlan,
    pub(crate) factories: Vec<ResolvedFactory>,
    pub(crate) operations: Vec<ResolvedOperation>,
    pub(crate) node_to_operation: Vec<OperationIndex>,
    pub(crate) constants: Vec<ConstantValue>,
    pub(crate) system_constants: Vec<ConstantIndex>,
    pub(crate) system_constant_lookup: BTreeMap<String, ConstantIndex>,
    pub(crate) node_constant_lookups: Vec<BTreeMap<String, ConstantIndex>>,
}

impl ExecutablePlan {
    /// Versión de esquema de la especificación origen.
    pub fn schema_version(&self) -> &str {
        &self.schema_version
    }

    /// Versión fija del compilador/formato de plan.
    pub fn compiler_version(&self) -> &str {
        &self.compiler_version
    }

    /// Identificador del sistema compilado.
    pub fn spec_id(&self) -> &str {
        &self.spec_id
    }

    /// Perfil temporal objetivo validado para todo el plan.
    pub fn temporal_profile(&self) -> TemporalProfile {
        self.temporal_profile
    }

    /// Huella semántica canónica (`u64`) calculada con [`gogamen_core::TraceHasher`].
    ///
    /// Dos escrituras canónicas equivalentes (por ejemplo distinto orden de nodos,
    /// puertos, conexiones o claves de configuración) producen exactamente el mismo valor.
    pub fn semantic_fingerprint(&self) -> u64 {
        self.semantic_fingerprint
    }

    /// Plan topológico compilado de conexiones e índices de nodos/puertos.
    pub fn graph_plan(&self) -> &CompiledPlan {
        &self.graph_plan
    }

    /// Orden topológico determinista de ejecución por [`NodeIndex`].
    pub fn execution_order(&self) -> &[NodeIndex] {
        self.graph_plan.execution_order()
    }

    /// Número total de nodos compilados en el plan.
    pub fn node_count(&self) -> usize {
        self.graph_plan.node_count()
    }

    /// Fábricas únicas resueltas en orden canónico.
    pub fn factories(&self) -> &[ResolvedFactory] {
        &self.factories
    }

    /// Obtiene una fábrica por su [`FactoryIndex`] directo.
    pub fn factory(&self, index: FactoryIndex) -> Option<&ResolvedFactory> {
        self.factories.get(index.0)
    }

    /// Operaciones resueltas en el plan.
    pub fn operations(&self) -> &[ResolvedOperation] {
        &self.operations
    }

    /// Obtiene una operación por su [`OperationIndex`] directo.
    pub fn operation(&self, index: OperationIndex) -> Option<&ResolvedOperation> {
        self.operations.get(index.0)
    }

    /// Obtiene la operación resuelta para un [`NodeIndex`] en `O(1)` sin búsquedas por cadena.
    pub fn node_operation(&self, node: NodeIndex) -> Option<&ResolvedOperation> {
        let op_idx = *self.node_to_operation.get(node.0)?;
        self.operations.get(op_idx.0)
    }

    /// Constantes escalares pre-resueltas del plan.
    pub fn constants(&self) -> &[ConstantValue] {
        &self.constants
    }

    /// Obtiene una constante por su [`ConstantIndex`] directo en `O(1)`.
    pub fn constant(&self, index: ConstantIndex) -> Option<&ConstantValue> {
        self.constants.get(index.0)
    }

    /// Índices de constantes de nivel sistema en orden canónico de clave.
    pub fn system_constant_indices(&self) -> &[ConstantIndex] {
        &self.system_constants
    }

    /// Transferencias instantáneas salientes de un nodo por [`NodeIndex`] directo.
    pub fn instant_outgoing(&self, node: NodeIndex) -> &[CompiledTransfer] {
        self.graph_plan.node_instant_outgoing(node)
    }

    /// Transferencias retardadas del plan con sus [`gogamen_core::BufferSlot`] asignados.
    pub fn delayed_transfers(&self) -> &[CompiledDelayedTransfer] {
        self.graph_plan.delayed_transfers()
    }

    /// Resuelve un [`NodeId`] a [`NodeIndex`] durante la fase de configuración (fuera de `step`).
    pub fn node_index(&self, id: &NodeId) -> Option<NodeIndex> {
        self.graph_plan.node_index(id)
    }

    /// Resuelve un [`PortId`] a `(PortDirection, PortIndex)` durante la fase de configuración.
    pub fn port_index(&self, node: NodeIndex, port: &PortId) -> Option<(PortDirection, PortIndex)> {
        self.graph_plan.port_index(node, port)
    }

    /// Resuelve una clave de constante de sistema a [`ConstantIndex`] (fuera de `step`).
    pub fn system_constant_index(&self, key: &str) -> Option<ConstantIndex> {
        self.system_constant_lookup.get(key).copied()
    }

    /// Resuelve una clave de constante de nodo a [`ConstantIndex`] (fuera de `step`).
    pub fn node_constant_index(&self, node: NodeIndex, key: &str) -> Option<ConstantIndex> {
        self.node_constant_lookups
            .get(node.0)
            .and_then(|map| map.get(key).copied())
    }

    /// Instancia recursos y actuadores usando únicamente las fábricas y operaciones ya validadas.
    pub fn instantiate<I: ResourceInitializer>(
        &self,
        initializer: &mut I,
    ) -> Result<InstantiatedPlan<I::Resource, I::Actuator>, I::Error> {
        let mut resources = Vec::with_capacity(self.factories.len());
        for factory in &self.factories {
            resources.push(initializer.init_factory(factory)?);
        }

        let mut actuators = Vec::with_capacity(self.operations.len());
        for operation in &self.operations {
            let factory = &self.factories[operation.factory.0];
            actuators.push(initializer.init_actuator(operation, factory)?);
        }

        Ok(InstantiatedPlan {
            resources,
            actuators,
        })
    }
}
