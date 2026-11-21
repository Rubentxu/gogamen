//! Grafo de puertos tipados y plan de conexiones con causalidad (`std`-only).
//!
//! Este módulo define la topología estática de nodos con puertos de entrada/salida tipados,
//! validación de compatibilidad de tipos, análisis de causalidad (`Instant` vs `Delayed(u64)`)
//! y compilación a un plan indexado determinista sin búsquedas por nombre en ejecución de tick:
//! - [`ConnectionGraph`]: constructor declarativo de nodos, puertos y conexiones.
//! - [`CompiledPlan`]: plan preparado con orden topológico estable e índices resueltos
//!   ([`NodeIndex`], [`PortIndex`], [`BufferSlot`]).
//! - [`PlanRuntime`]: ejecutor por tick que opera exclusivamente por índices directos sobre
//!   buffers preasignados sin búsquedas en `HashMap` ni comparaciones de `String`.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fmt;

use crate::units::Tick;

/// Identificador lógico de un nodo dentro de un [`ConnectionGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(String);

impl NodeId {
    /// Crea un identificador de nodo a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for NodeId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for NodeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identificador lógico de un puerto dentro de un nodo.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PortId(String);

impl PortId {
    /// Crea un identificador de puerto a partir de cualquier cadena convertible.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Devuelve la representación textual del identificador.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for PortId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl From<String> for PortId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Dirección de un puerto en un nodo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortDirection {
    /// Puerto de entrada que consume datos producidos por otro nodo o por un buffer retardado.
    Input,
    /// Puerto de salida que produce datos hacia conexiones instantáneas o retardadas.
    Output,
}

impl fmt::Display for PortDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input => f.write_str("Input"),
            Self::Output => f.write_str("Output"),
        }
    }
}

/// Descriptor explícito de tipo para puertos de [`ConnectionGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PortType {
    /// Booleano (`bool`).
    Bool,
    /// Entero con signo de 64 bits (`i64`).
    I64,
    /// Entero sin signo de 64 bits (`u64`).
    U64,
    /// Punto flotante de 64 bits (`f64`).
    F64,
    /// Cadena de texto (`String`).
    String,
    /// Señal sin carga útil (`()`).
    Unit,
    /// Tipo nominal personalizado explícito.
    Custom(String),
}

impl PortType {
    /// Construye un tipo nominal personalizado.
    pub fn custom(name: impl Into<String>) -> Self {
        Self::Custom(name.into())
    }

    /// Indica si este tipo es compatible como origen para un puerto destino `expected`.
    pub fn is_compatible_with(&self, expected: &Self) -> bool {
        self == expected
    }
}

impl fmt::Display for PortType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool => f.write_str("Bool"),
            Self::I64 => f.write_str("I64"),
            Self::U64 => f.write_str("U64"),
            Self::F64 => f.write_str("F64"),
            Self::String => f.write_str("String"),
            Self::Unit => f.write_str("Unit"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

/// Especificación declarativa de un puerto en un nodo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortSpec {
    /// Identificador del puerto dentro del nodo.
    pub id: PortId,
    /// Dirección del puerto (`Input` o `Output`).
    pub direction: PortDirection,
    /// Tipo de dato transportado por el puerto.
    pub port_type: PortType,
}

impl PortSpec {
    /// Crea un puerto de entrada tipado.
    pub fn input(id: impl Into<PortId>, port_type: PortType) -> Self {
        Self {
            id: id.into(),
            direction: PortDirection::Input,
            port_type,
        }
    }

    /// Crea un puerto de salida tipado.
    pub fn output(id: impl Into<PortId>, port_type: PortType) -> Self {
        Self {
            id: id.into(),
            direction: PortDirection::Output,
            port_type,
        }
    }
}

/// Especificación declarativa de un nodo y sus puertos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeSpec {
    /// Identificador único del nodo en el grafo.
    pub id: NodeId,
    /// Lista ordenada de puertos del nodo.
    pub ports: Vec<PortSpec>,
}

impl NodeSpec {
    /// Crea un nodo sin puertos todavía.
    pub fn new(id: impl Into<NodeId>) -> Self {
        Self {
            id: id.into(),
            ports: Vec::new(),
        }
    }

    /// Añade un puerto genérico al nodo (estilo builder).
    pub fn with_port(mut self, port: PortSpec) -> Self {
        self.ports.push(port);
        self
    }

    /// Añade un puerto de entrada tipado al nodo (estilo builder).
    pub fn with_input(mut self, id: impl Into<PortId>, port_type: PortType) -> Self {
        self.ports.push(PortSpec::input(id, port_type));
        self
    }

    /// Añade un puerto de salida tipado al nodo (estilo builder).
    pub fn with_output(mut self, id: impl Into<PortId>, port_type: PortType) -> Self {
        self.ports.push(PortSpec::output(id, port_type));
        self
    }
}

/// Semántica causal de una conexión entre dos puertos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalKind {
    /// Dependencia instantánea dentro del mismo tick (`delay == 0`).
    Instant,
    /// Dependencia retardada por `delay >= 1` ticks mediante buffer de estado.
    Delayed(u64),
}

impl CausalKind {
    /// Construye la semántica causal a partir de un retardo en ticks (`0 => Instant`).
    pub const fn from_delay_ticks(delay_ticks: u64) -> Self {
        if delay_ticks == 0 {
            Self::Instant
        } else {
            Self::Delayed(delay_ticks)
        }
    }

    /// Devuelve el retardo en ticks (`0` si es instantánea).
    pub const fn delay_ticks(self) -> u64 {
        match self {
            Self::Instant => 0,
            Self::Delayed(d) => d,
        }
    }

    /// Indica si la conexión es instantánea (`delay == 0`).
    pub const fn is_instant(self) -> bool {
        self.delay_ticks() == 0
    }
}

/// Especificación declarativa de una conexión dirigida entre un puerto de salida y uno de entrada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionSpec {
    /// Nodo origen (productor).
    pub from_node: NodeId,
    /// Puerto de salida en el nodo origen.
    pub from_port: PortId,
    /// Nodo destino (consumidor).
    pub to_node: NodeId,
    /// Puerto de entrada en el nodo destino.
    pub to_port: PortId,
    /// Semántica causal (`Instant` o `Delayed(ticks)`).
    pub causal: CausalKind,
}

impl ConnectionSpec {
    /// Crea una conexión instantánea (`delay == 0`) entre salida y entrada.
    pub fn instant(
        from_node: impl Into<NodeId>,
        from_port: impl Into<PortId>,
        to_node: impl Into<NodeId>,
        to_port: impl Into<PortId>,
    ) -> Self {
        Self {
            from_node: from_node.into(),
            from_port: from_port.into(),
            to_node: to_node.into(),
            to_port: to_port.into(),
            causal: CausalKind::Instant,
        }
    }

    /// Crea una conexión con retardo explícito en ticks (`0` equivale a [`CausalKind::Instant`]).
    pub fn delayed(
        from_node: impl Into<NodeId>,
        from_port: impl Into<PortId>,
        to_node: impl Into<NodeId>,
        to_port: impl Into<PortId>,
        delay_ticks: u64,
    ) -> Self {
        Self {
            from_node: from_node.into(),
            from_port: from_port.into(),
            to_node: to_node.into(),
            to_port: to_port.into(),
            causal: CausalKind::from_delay_ticks(delay_ticks),
        }
    }
}

/// Índice numérico denso de un nodo resuelto al compilar (`0..node_count`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIndex(pub usize);

/// Índice numérico denso de un puerto dentro de su dirección en un nodo (`0..dir_port_count`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PortIndex(pub usize);

/// Índice numérico denso de un buffer de retardo resuelto al compilar (`0..delayed_count`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferSlot(pub usize);

/// Índice numérico de una conexión registrada en [`ConnectionGraph`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConnectionIndex(pub usize);

/// Detalle de incompatibilidad de tipos entre puerto origen y puerto destino.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeMismatchError {
    /// Nodo origen.
    pub from_node: NodeId,
    /// Puerto origen.
    pub from_port: PortId,
    /// Tipo encontrado en el puerto origen.
    pub found_type: PortType,
    /// Nodo destino.
    pub to_node: NodeId,
    /// Puerto destino.
    pub to_port: PortId,
    /// Tipo esperado por el puerto destino.
    pub expected_type: PortType,
}

/// Errores de validación y compilación de [`ConnectionGraph`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionError {
    /// Identificador vacío en nodo o puerto.
    EmptyIdentifier {
        /// Contexto donde apareció el identificador vacío.
        context: &'static str,
    },
    /// Se intentó registrar un nodo con un [`NodeId`] ya existente.
    DuplicateNode {
        /// Identificador duplicado.
        node: NodeId,
    },
    /// Se intentó registrar un puerto con [`PortId`] duplicado dentro del mismo nodo.
    DuplicatePort {
        /// Nodo contenedor.
        node: NodeId,
        /// Puerto duplicado.
        port: PortId,
    },
    /// Referencia a un nodo inexistente al conectar.
    UnknownNode {
        /// Identificador del nodo faltante.
        node: NodeId,
        /// Rol en la conexión (`"source"` o `"target"`).
        role: &'static str,
    },
    /// Referencia a un puerto inexistente en un nodo existente.
    UnknownPort {
        /// Nodo consultado.
        node: NodeId,
        /// Puerto faltante.
        port: PortId,
        /// Rol en la conexión (`"source"` o `"target"`).
        role: &'static str,
    },
    /// Dirección de puerto inválida para el extremo de la conexión.
    InvalidPortDirection {
        /// Nodo del puerto.
        node: NodeId,
        /// Puerto con dirección incompatible.
        port: PortId,
        /// Dirección esperada.
        expected: PortDirection,
        /// Dirección encontrada.
        found: PortDirection,
    },
    /// El puerto de entrada destino ya posee una conexión entrante asignada.
    InputPortAlreadyConnected {
        /// Nodo destino.
        to_node: NodeId,
        /// Puerto de entrada ya conectado.
        to_port: PortId,
    },
    /// Conexión entre puertos de tipos incompatibles con ruta de diagnóstico completa.
    TypeMismatch(Box<TypeMismatchError>),
    /// Se detectó un ciclo de dependencias instantáneas (`delay == 0`).
    InstantCycleDetected {
        /// Secuencia de nodos que forma el ciclo (cerrada con el nodo inicial al final).
        cycle: Vec<NodeId>,
    },
}

impl fmt::Display for ConnectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier { context } => {
                write!(f, "empty identifier in {context}")
            }
            Self::DuplicateNode { node } => {
                write!(f, "duplicate node id '{node}'")
            }
            Self::DuplicatePort { node, port } => {
                write!(f, "duplicate port id '{port}' on node '{node}'")
            }
            Self::UnknownNode { node, role } => {
                write!(f, "unknown {role} node '{node}'")
            }
            Self::UnknownPort { node, port, role } => {
                write!(f, "unknown {role} port '{port}' on node '{node}'")
            }
            Self::InvalidPortDirection {
                node,
                port,
                expected,
                found,
            } => write!(
                f,
                "invalid port direction on '{node}.{port}': expected {expected}, found {found}"
            ),
            Self::InputPortAlreadyConnected { to_node, to_port } => write!(
                f,
                "input port '{to_node}.{to_port}' already has an incoming connection"
            ),
            Self::TypeMismatch(detail) => write!(
                f,
                "type mismatch connecting '{}.{}' (found {}) -> '{}.{}' (expected {})",
                detail.from_node,
                detail.from_port,
                detail.found_type,
                detail.to_node,
                detail.to_port,
                detail.expected_type,
            ),
            Self::InstantCycleDetected { cycle } => {
                let path: Vec<&str> = cycle.iter().map(NodeId::as_str).collect();
                write!(f, "instantaneous cycle detected: {}", path.join(" -> "))
            }
        }
    }
}

impl Error for ConnectionError {}

#[derive(Debug, Clone)]
struct PortEntry {
    id: PortId,
    direction: PortDirection,
    port_type: PortType,
    dir_index: PortIndex,
}

#[derive(Debug, Clone)]
struct NodeEntry {
    id: NodeId,
    ports: Vec<PortEntry>,
    port_lookup: HashMap<PortId, usize>,
    input_count: usize,
    output_count: usize,
}

#[derive(Debug, Clone)]
struct ConnectionEntry {
    from_node: NodeIndex,
    from_port: PortIndex,
    to_node: NodeIndex,
    to_port: PortIndex,
    causal: CausalKind,
}

/// Grafo declarativo de nodos con puertos tipados y conexiones causales.
#[derive(Debug, Clone, Default)]
pub struct ConnectionGraph {
    nodes: Vec<NodeEntry>,
    node_lookup: HashMap<NodeId, NodeIndex>,
    connections: Vec<ConnectionEntry>,
}

impl ConnectionGraph {
    /// Crea un grafo de conexiones vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra un nodo y sus puertos, validando unicidad de identificadores.
    pub fn add_node(&mut self, spec: NodeSpec) -> Result<NodeIndex, ConnectionError> {
        if spec.id.as_str().trim().is_empty() {
            return Err(ConnectionError::EmptyIdentifier {
                context: "NodeSpec",
            });
        }
        if self.node_lookup.contains_key(&spec.id) {
            return Err(ConnectionError::DuplicateNode { node: spec.id });
        }

        let mut ports = Vec::with_capacity(spec.ports.len());
        let mut port_lookup = HashMap::with_capacity(spec.ports.len());
        let mut input_count = 0usize;
        let mut output_count = 0usize;

        for port in spec.ports {
            if port.id.as_str().trim().is_empty() {
                return Err(ConnectionError::EmptyIdentifier {
                    context: "PortSpec",
                });
            }
            if port_lookup.contains_key(&port.id) {
                return Err(ConnectionError::DuplicatePort {
                    node: spec.id,
                    port: port.id,
                });
            }

            let dir_index = match port.direction {
                PortDirection::Input => {
                    let idx = PortIndex(input_count);
                    input_count += 1;
                    idx
                }
                PortDirection::Output => {
                    let idx = PortIndex(output_count);
                    output_count += 1;
                    idx
                }
            };

            let vec_idx = ports.len();
            port_lookup.insert(port.id.clone(), vec_idx);
            ports.push(PortEntry {
                id: port.id,
                direction: port.direction,
                port_type: port.port_type,
                dir_index,
            });
        }

        let node_index = NodeIndex(self.nodes.len());
        self.node_lookup.insert(spec.id.clone(), node_index);
        self.nodes.push(NodeEntry {
            id: spec.id,
            ports,
            port_lookup,
            input_count,
            output_count,
        });

        Ok(node_index)
    }

    /// Registra una conexión dirigida validando existencia, dirección, unicidad de entrada y tipos.
    pub fn connect(&mut self, spec: ConnectionSpec) -> Result<ConnectionIndex, ConnectionError> {
        let from_node_idx =
            *self
                .node_lookup
                .get(&spec.from_node)
                .ok_or_else(|| ConnectionError::UnknownNode {
                    node: spec.from_node.clone(),
                    role: "source",
                })?;

        let to_node_idx =
            *self
                .node_lookup
                .get(&spec.to_node)
                .ok_or_else(|| ConnectionError::UnknownNode {
                    node: spec.to_node.clone(),
                    role: "target",
                })?;

        let from_port_entry = {
            let node = &self.nodes[from_node_idx.0];
            let port_vec_idx = *node.port_lookup.get(&spec.from_port).ok_or_else(|| {
                ConnectionError::UnknownPort {
                    node: spec.from_node.clone(),
                    port: spec.from_port.clone(),
                    role: "source",
                }
            })?;
            node.ports[port_vec_idx].clone()
        };

        if from_port_entry.direction != PortDirection::Output {
            return Err(ConnectionError::InvalidPortDirection {
                node: spec.from_node,
                port: spec.from_port,
                expected: PortDirection::Output,
                found: from_port_entry.direction,
            });
        }

        let to_port_entry = {
            let node = &self.nodes[to_node_idx.0];
            let port_vec_idx = *node.port_lookup.get(&spec.to_port).ok_or_else(|| {
                ConnectionError::UnknownPort {
                    node: spec.to_node.clone(),
                    port: spec.to_port.clone(),
                    role: "target",
                }
            })?;
            node.ports[port_vec_idx].clone()
        };

        if to_port_entry.direction != PortDirection::Input {
            return Err(ConnectionError::InvalidPortDirection {
                node: spec.to_node,
                port: spec.to_port,
                expected: PortDirection::Input,
                found: to_port_entry.direction,
            });
        }

        if !from_port_entry
            .port_type
            .is_compatible_with(&to_port_entry.port_type)
        {
            return Err(ConnectionError::TypeMismatch(Box::new(TypeMismatchError {
                from_node: spec.from_node,
                from_port: spec.from_port,
                found_type: from_port_entry.port_type,
                to_node: spec.to_node,
                to_port: spec.to_port,
                expected_type: to_port_entry.port_type,
            })));
        }

        let already_connected = self
            .connections
            .iter()
            .any(|conn| conn.to_node == to_node_idx && conn.to_port == to_port_entry.dir_index);
        if already_connected {
            return Err(ConnectionError::InputPortAlreadyConnected {
                to_node: spec.to_node,
                to_port: spec.to_port,
            });
        }

        let conn_index = ConnectionIndex(self.connections.len());
        self.connections.push(ConnectionEntry {
            from_node: from_node_idx,
            from_port: from_port_entry.dir_index,
            to_node: to_node_idx,
            to_port: to_port_entry.dir_index,
            causal: spec.causal,
        });

        Ok(conn_index)
    }

    /// Número de nodos registrados.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Número de conexiones registradas.
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }

    /// Prepara y compila el grafo validando el DAG instantáneo y resolviendo todos los índices.
    pub fn prepare(&self) -> Result<CompiledPlan, ConnectionError> {
        let node_count = self.nodes.len();
        let mut instant_adj: Vec<Vec<NodeIndex>> = vec![Vec::new(); node_count];
        let mut in_degree: Vec<usize> = vec![0; node_count];
        let mut instant_outgoing: Vec<Vec<CompiledTransfer>> = vec![Vec::new(); node_count];
        let mut delayed_transfers = Vec::new();

        for conn in &self.connections {
            if conn.causal.is_instant() {
                instant_adj[conn.from_node.0].push(conn.to_node);
                in_degree[conn.to_node.0] += 1;
                instant_outgoing[conn.from_node.0].push(CompiledTransfer {
                    from_node: conn.from_node,
                    from_port: conn.from_port,
                    to_node: conn.to_node,
                    to_port: conn.to_port,
                });
            } else {
                let slot = BufferSlot(delayed_transfers.len());
                delayed_transfers.push(CompiledDelayedTransfer {
                    from_node: conn.from_node,
                    from_port: conn.from_port,
                    to_node: conn.to_node,
                    to_port: conn.to_port,
                    delay_ticks: conn.causal.delay_ticks(),
                    buffer_slot: slot,
                });
            }
        }

        // Ordenación topológica estable (Kahn con min-heap por NodeIndex).
        let mut ready = BinaryHeap::new();
        for (idx, &deg) in in_degree.iter().enumerate() {
            if deg == 0 {
                ready.push(Reverse(NodeIndex(idx)));
            }
        }

        let mut execution_order = Vec::with_capacity(node_count);
        let mut remaining_in_degree = in_degree;

        while let Some(Reverse(node)) = ready.pop() {
            execution_order.push(node);
            for &next in &instant_adj[node.0] {
                remaining_in_degree[next.0] -= 1;
                if remaining_in_degree[next.0] == 0 {
                    ready.push(Reverse(next));
                }
            }
        }

        if execution_order.len() != node_count {
            let cycle_indices = find_deterministic_cycle(&instant_adj, &remaining_in_degree);
            let cycle = cycle_indices
                .into_iter()
                .map(|idx| self.nodes[idx.0].id.clone())
                .collect();
            return Err(ConnectionError::InstantCycleDetected { cycle });
        }

        let nodes = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, entry)| {
                let ports = entry
                    .ports
                    .iter()
                    .map(|p| CompiledPortMeta {
                        id: p.id.clone(),
                        direction: p.direction,
                        port_type: p.port_type.clone(),
                        index: p.dir_index,
                    })
                    .collect();
                CompiledNodeMeta {
                    id: entry.id.clone(),
                    index: NodeIndex(idx),
                    input_count: entry.input_count,
                    output_count: entry.output_count,
                    ports,
                }
            })
            .collect();

        Ok(CompiledPlan {
            nodes,
            node_lookup: self.node_lookup.clone(),
            execution_order,
            instant_outgoing,
            delayed_transfers,
        })
    }

    /// Alias de [`ConnectionGraph::prepare`].
    pub fn compile(&self) -> Result<CompiledPlan, ConnectionError> {
        self.prepare()
    }
}

fn find_deterministic_cycle(
    adj: &[Vec<NodeIndex>],
    remaining_in_degree: &[usize],
) -> Vec<NodeIndex> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum State {
        Unvisited,
        Visiting,
        Done,
    }

    fn dfs(
        node: NodeIndex,
        adj: &[Vec<NodeIndex>],
        remaining_in_degree: &[usize],
        state: &mut [State],
        stack: &mut Vec<NodeIndex>,
    ) -> Option<Vec<NodeIndex>> {
        state[node.0] = State::Visiting;
        stack.push(node);

        let mut neighbors = adj[node.0].clone();
        neighbors.sort();

        for next in neighbors {
            if remaining_in_degree[next.0] == 0 {
                continue;
            }
            match state[next.0] {
                State::Visiting => {
                    if let Some(pos) = stack.iter().position(|&n| n == next) {
                        let mut cycle = stack[pos..].to_vec();
                        cycle.push(next);
                        return Some(cycle);
                    }
                }
                State::Unvisited => {
                    if let Some(cycle) = dfs(next, adj, remaining_in_degree, state, stack) {
                        return Some(cycle);
                    }
                }
                State::Done => {}
            }
        }

        stack.pop();
        state[node.0] = State::Done;
        None
    }

    let mut state = vec![State::Unvisited; adj.len()];
    let mut stack = Vec::new();
    for (idx, &deg) in remaining_in_degree.iter().enumerate() {
        if deg > 0
            && state[idx] == State::Unvisited
            && let Some(cycle) = dfs(
                NodeIndex(idx),
                adj,
                remaining_in_degree,
                &mut state,
                &mut stack,
            )
        {
            return cycle;
        }
    }
    Vec::new()
}

/// Metadatos compilados de un puerto (usados solo en inspección o resolución inicial).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPortMeta {
    /// Identificador lógico del puerto.
    pub id: PortId,
    /// Dirección del puerto.
    pub direction: PortDirection,
    /// Tipo del puerto.
    pub port_type: PortType,
    /// Índice resuelto dentro de su dirección en el nodo.
    pub index: PortIndex,
}

/// Metadatos compilados de un nodo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledNodeMeta {
    /// Identificador lógico del nodo.
    pub id: NodeId,
    /// Índice resuelto del nodo.
    pub index: NodeIndex,
    /// Cantidad de puertos de entrada.
    pub input_count: usize,
    /// Cantidad de puertos de salida.
    pub output_count: usize,
    /// Metadatos de puertos del nodo.
    pub ports: Vec<CompiledPortMeta>,
}

/// Transferencia instantánea pre-resuelta por índices numéricos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledTransfer {
    /// Índice del nodo productor.
    pub from_node: NodeIndex,
    /// Índice del puerto de salida productor.
    pub from_port: PortIndex,
    /// Índice del nodo consumidor.
    pub to_node: NodeIndex,
    /// Índice del puerto de entrada consumidor.
    pub to_port: PortIndex,
}

/// Transferencia retardada pre-resuelta con su slot de buffer de estado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledDelayedTransfer {
    /// Índice del nodo productor.
    pub from_node: NodeIndex,
    /// Índice del puerto de salida productor.
    pub from_port: PortIndex,
    /// Índice del nodo consumidor.
    pub to_node: NodeIndex,
    /// Índice del puerto de entrada consumidor.
    pub to_port: PortIndex,
    /// Retardo en ticks (`>= 1`).
    pub delay_ticks: u64,
    /// Slot asignado al buffer circular de estado.
    pub buffer_slot: BufferSlot,
}

/// Plan de conexiones compilado con orden topológico determinista e índices directos.
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    nodes: Vec<CompiledNodeMeta>,
    node_lookup: HashMap<NodeId, NodeIndex>,
    execution_order: Vec<NodeIndex>,
    instant_outgoing: Vec<Vec<CompiledTransfer>>,
    delayed_transfers: Vec<CompiledDelayedTransfer>,
}

impl CompiledPlan {
    /// Número total de nodos en el plan.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Orden topológico determinista de ejecución de nodos por [`NodeIndex`].
    pub fn execution_order(&self) -> &[NodeIndex] {
        &self.execution_order
    }

    /// Devuelve el [`NodeId`] correspondiente a un [`NodeIndex`].
    pub fn node_id(&self, index: NodeIndex) -> Option<&NodeId> {
        self.nodes.get(index.0).map(|n| &n.id)
    }

    /// Resuelve un [`NodeId`] a su [`NodeIndex`] (para fase de configuración, fuera del tick).
    pub fn node_index(&self, id: &NodeId) -> Option<NodeIndex> {
        self.node_lookup.get(id).copied()
    }

    /// Resuelve un [`PortId`] dentro de un nodo a `(PortDirection, PortIndex)` (fuera del tick).
    pub fn port_index(&self, node: NodeIndex, port: &PortId) -> Option<(PortDirection, PortIndex)> {
        self.nodes.get(node.0).and_then(|meta| {
            meta.ports
                .iter()
                .find(|p| &p.id == port)
                .map(|p| (p.direction, p.index))
        })
    }

    /// Metadatos de un nodo por índice directo.
    pub fn node_meta(&self, index: NodeIndex) -> Option<&CompiledNodeMeta> {
        self.nodes.get(index.0)
    }

    /// Transferencias instantáneas salientes de un nodo por índice directo.
    pub fn node_instant_outgoing(&self, node: NodeIndex) -> &[CompiledTransfer] {
        self.instant_outgoing
            .get(node.0)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Todas las transferencias retardadas y sus [`BufferSlot`] asignados.
    pub fn delayed_transfers(&self) -> &[CompiledDelayedTransfer] {
        &self.delayed_transfers
    }

    /// Metadatos de todos los nodos compilados indexados por [`NodeIndex`].
    pub fn nodes(&self) -> &[CompiledNodeMeta] {
        &self.nodes
    }

    /// Matriz completa de transferencias instantáneas salientes indexada por [`NodeIndex`].
    pub fn instant_outgoing_all(&self) -> &[Vec<CompiledTransfer>] {
        &self.instant_outgoing
    }

    /// Reconstruye un [`CompiledPlan`] a partir de sus partes pre-resueltas validando
    /// invariantes estructurales de índices y topología sin ejecutar código externo.
    pub fn from_parts(
        nodes: Vec<CompiledNodeMeta>,
        execution_order: Vec<NodeIndex>,
        instant_outgoing: Vec<Vec<CompiledTransfer>>,
        delayed_transfers: Vec<CompiledDelayedTransfer>,
    ) -> Result<Self, String> {
        let node_count = nodes.len();
        if instant_outgoing.len() != node_count {
            return Err(format!(
                "instant_outgoing length {} does not match node count {}",
                instant_outgoing.len(),
                node_count
            ));
        }

        let mut node_lookup = HashMap::with_capacity(node_count);
        for (idx, meta) in nodes.iter().enumerate() {
            if meta.index.0 != idx {
                return Err(format!(
                    "node '{}' has non-canonical index {} at slot {}",
                    meta.id, meta.index.0, idx
                ));
            }
            if node_lookup.insert(meta.id.clone(), meta.index).is_some() {
                return Err(format!("duplicate node id '{}' in compiled plan", meta.id));
            }
        }

        if execution_order.len() != node_count {
            return Err(format!(
                "execution_order length {} does not match node count {}",
                execution_order.len(),
                node_count
            ));
        }
        let mut seen_order = vec![false; node_count];
        for &node_idx in &execution_order {
            let Some(slot) = seen_order.get_mut(node_idx.0) else {
                return Err(format!(
                    "execution_order index {} out of bounds",
                    node_idx.0
                ));
            };
            if *slot {
                return Err(format!(
                    "duplicate node index {} in execution_order",
                    node_idx.0
                ));
            }
            *slot = true;
        }

        for (src_slot, transfers) in instant_outgoing.iter().enumerate() {
            for t in transfers {
                if t.from_node.0 != src_slot {
                    return Err(format!(
                        "instant transfer from_node {} does not match slot {}",
                        t.from_node.0, src_slot
                    ));
                }
                let Some(from_meta) = nodes.get(t.from_node.0) else {
                    return Err(format!(
                        "transfer from_node {} out of bounds",
                        t.from_node.0
                    ));
                };
                let Some(to_meta) = nodes.get(t.to_node.0) else {
                    return Err(format!("transfer to_node {} out of bounds", t.to_node.0));
                };
                if t.from_port.0 >= from_meta.output_count {
                    return Err(format!(
                        "transfer from_port {} out of bounds for node {}",
                        t.from_port.0, t.from_node.0
                    ));
                }
                if t.to_port.0 >= to_meta.input_count {
                    return Err(format!(
                        "transfer to_port {} out of bounds for node {}",
                        t.to_port.0, t.to_node.0
                    ));
                }
            }
        }

        for (slot_idx, dt) in delayed_transfers.iter().enumerate() {
            let Some(from_meta) = nodes.get(dt.from_node.0) else {
                return Err(format!(
                    "delayed transfer from_node {} out of bounds",
                    dt.from_node.0
                ));
            };
            let Some(to_meta) = nodes.get(dt.to_node.0) else {
                return Err(format!(
                    "delayed transfer to_node {} out of bounds",
                    dt.to_node.0
                ));
            };
            if dt.from_port.0 >= from_meta.output_count {
                return Err(format!(
                    "delayed transfer from_port {} out of bounds for node {}",
                    dt.from_port.0, dt.from_node.0
                ));
            }
            if dt.to_port.0 >= to_meta.input_count {
                return Err(format!(
                    "delayed transfer to_port {} out of bounds for node {}",
                    dt.to_port.0, dt.to_node.0
                ));
            }
            if dt.delay_ticks == 0 {
                return Err("delayed transfer delay_ticks must be >= 1".to_owned());
            }
            if dt.buffer_slot.0 != slot_idx {
                return Err(format!(
                    "delayed transfer buffer_slot {} does not match slot {}",
                    dt.buffer_slot.0, slot_idx
                ));
            }
        }

        Ok(Self {
            nodes,
            node_lookup,
            execution_order,
            instant_outgoing,
            delayed_transfers,
        })
    }
}

#[derive(Debug, Clone)]
struct DelayBuffer<V> {
    slots: Vec<Option<V>>,
    cursor: usize,
}

impl<V: Clone> DelayBuffer<V> {
    fn new(delay_ticks: u64) -> Self {
        let len = usize::try_from(delay_ticks).expect("delay_ticks must fit in usize");
        Self {
            slots: vec![None; len],
            cursor: 0,
        }
    }

    fn peek(&self) -> Option<&V> {
        self.slots.get(self.cursor).and_then(Option::as_ref)
    }

    fn push_and_advance(&mut self, value: Option<V>) {
        if !self.slots.is_empty() {
            self.slots[self.cursor] = value;
            self.cursor = (self.cursor + 1) % self.slots.len();
        }
    }

    fn fill(&mut self, value: V) {
        for slot in &mut self.slots {
            *slot = Some(value.clone());
        }
    }
}

/// Estado de ejecución por índices directos para un [`CompiledPlan`].
///
/// Toda la memoria de entradas, salidas y buffers de retardo se preasigna en
/// [`PlanRuntime::new`]; [`PlanRuntime::execute_tick`] opera exclusivamente mediante
/// indexación numérica directa sin búsquedas por nombre.
#[derive(Debug, Clone)]
pub struct PlanRuntime<V> {
    /// Entradas de todos los nodos en un único bloque contiguo.
    flat_inputs: Vec<Option<V>>,
    /// Salidas de todos los nodos en un único bloque contiguo.
    flat_outputs: Vec<Option<V>>,
    /// `(offset, len)` de las entradas de cada nodo dentro de `flat_inputs`.
    node_input_spans: Vec<(usize, usize)>,
    /// `(offset, len)` de las salidas de cada nodo dentro de `flat_outputs`.
    node_output_spans: Vec<(usize, usize)>,
    delay_buffers: Vec<DelayBuffer<V>>,
}

impl<V: Clone> PlanRuntime<V> {
    /// Crea un runtime preasignado para el plan compilado.
    pub fn new(plan: &CompiledPlan) -> Self {
        let mut node_input_spans = Vec::with_capacity(plan.node_count());
        let mut node_output_spans = Vec::with_capacity(plan.node_count());
        let mut input_total = 0usize;
        let mut output_total = 0usize;

        for idx in 0..plan.node_count() {
            let meta = plan
                .node_meta(NodeIndex(idx))
                .expect("compiled plan node index must exist");
            node_input_spans.push((input_total, meta.input_count));
            node_output_spans.push((output_total, meta.output_count));
            input_total += meta.input_count;
            output_total += meta.output_count;
        }

        let delay_buffers = plan
            .delayed_transfers()
            .iter()
            .map(|d| DelayBuffer::new(d.delay_ticks))
            .collect();

        Self {
            flat_inputs: vec![None; input_total],
            flat_outputs: vec![None; output_total],
            node_input_spans,
            node_output_spans,
            delay_buffers,
        }
    }

    #[inline]
    fn input_index(&self, node: usize, port: usize) -> Option<usize> {
        let &(offset, len) = self.node_input_spans.get(node)?;
        (port < len).then_some(offset + port)
    }

    #[inline]
    fn output_index(&self, node: usize, port: usize) -> Option<usize> {
        let &(offset, len) = self.node_output_spans.get(node)?;
        (port < len).then_some(offset + port)
    }

    /// Inicializa un buffer de retardo con un valor semilla para los primeros ticks.
    pub fn seed_delay(&mut self, slot: BufferSlot, value: V) {
        if let Some(buf) = self.delay_buffers.get_mut(slot.0) {
            buf.fill(value);
        }
    }

    /// Escribe una entrada externa en un nodo/puerto por índice directo.
    pub fn set_input(&mut self, node: NodeIndex, port: PortIndex, value: Option<V>) {
        if let Some(i) = self.input_index(node.0, port.0) {
            self.flat_inputs[i] = value;
        }
    }

    /// Lee la salida producida por un nodo/puerto por índice directo.
    pub fn output(&self, node: NodeIndex, port: PortIndex) -> Option<&V> {
        let i = self.output_index(node.0, port.0)?;
        self.flat_outputs[i].as_ref()
    }

    /// Ejecuta un tick completo siguiendo el orden topológico compilado, sin búsquedas por nombre.
    pub fn execute_tick<F>(&mut self, plan: &CompiledPlan, tick: Tick, mut eval_node: F)
    where
        F: FnMut(NodeIndex, Tick, &[Option<V>], &mut [Option<V>]),
    {
        // Fase 1: cargar entradas retardadas desde los buffers de estado antes de ejecutar nodos.
        for delayed in plan.delayed_transfers() {
            let val = self.delay_buffers[delayed.buffer_slot.0].peek().cloned();
            let i = self
                .input_index(delayed.to_node.0, delayed.to_port.0)
                .unwrap();
            self.flat_inputs[i] = val;
        }

        // Fase 2: evaluar nodos en orden topológico estable y propagar conexiones instantáneas.
        for &node_idx in plan.execution_order() {
            let (in_off, in_len) = self.node_input_spans[node_idx.0];
            let (out_off, out_len) = self.node_output_spans[node_idx.0];
            self.flat_outputs[out_off..out_off + out_len].fill(None);

            {
                let inputs = &self.flat_inputs[in_off..in_off + in_len];
                let outputs = &mut self.flat_outputs[out_off..out_off + out_len];
                eval_node(node_idx, tick, inputs, outputs);
            }

            for transfer in plan.node_instant_outgoing(node_idx) {
                let from = self
                    .output_index(transfer.from_node.0, transfer.from_port.0)
                    .unwrap();
                let to = self
                    .input_index(transfer.to_node.0, transfer.to_port.0)
                    .unwrap();
                self.flat_inputs[to] = self.flat_outputs[from].clone();
            }
        }

        // Fase 3: comprometer las salidas hacia los buffers de retardo al final del tick.
        for delayed in plan.delayed_transfers() {
            let from = self
                .output_index(delayed.from_node.0, delayed.from_port.0)
                .unwrap();
            let val = self.flat_outputs[from].clone();
            self.delay_buffers[delayed.buffer_slot.0].push_and_advance(val);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incompatible_port_types_fail_with_full_diagnostic_path() {
        let mut graph = ConnectionGraph::new();
        graph
            .add_node(NodeSpec::new("sensor").with_output("reading", PortType::F64))
            .expect("sensor node");
        graph
            .add_node(NodeSpec::new("controller").with_input("flag", PortType::Bool))
            .expect("controller node");

        let err = graph
            .connect(ConnectionSpec::instant(
                "sensor",
                "reading",
                "controller",
                "flag",
            ))
            .expect_err("mismatched types must fail");

        assert_eq!(
            err,
            ConnectionError::TypeMismatch(Box::new(TypeMismatchError {
                from_node: NodeId::new("sensor"),
                from_port: PortId::new("reading"),
                found_type: PortType::F64,
                to_node: NodeId::new("controller"),
                to_port: PortId::new("flag"),
                expected_type: PortType::Bool,
            }))
        );

        let rendered = err.to_string();
        assert!(rendered.contains("sensor.reading"));
        assert!(rendered.contains("F64"));
        assert!(rendered.contains("controller.flag"));
        assert!(rendered.contains("Bool"));
    }

    #[test]
    fn instantaneous_cycle_is_rejected_with_cycle_path() {
        let mut graph = ConnectionGraph::new();
        graph
            .add_node(
                NodeSpec::new("A")
                    .with_input("in", PortType::I64)
                    .with_output("out", PortType::I64),
            )
            .expect("A");
        graph
            .add_node(
                NodeSpec::new("B")
                    .with_input("in", PortType::I64)
                    .with_output("out", PortType::I64),
            )
            .expect("B");
        graph
            .add_node(
                NodeSpec::new("C")
                    .with_input("in", PortType::I64)
                    .with_output("out", PortType::I64),
            )
            .expect("C");

        graph
            .connect(ConnectionSpec::instant("A", "out", "B", "in"))
            .expect("A->B");
        graph
            .connect(ConnectionSpec::instant("B", "out", "C", "in"))
            .expect("B->C");
        graph
            .connect(ConnectionSpec::instant("C", "out", "A", "in"))
            .expect("C->A");

        let err = graph.prepare().expect_err("instant cycle must be rejected");
        assert_eq!(
            err,
            ConnectionError::InstantCycleDetected {
                cycle: vec![
                    NodeId::new("A"),
                    NodeId::new("B"),
                    NodeId::new("C"),
                    NodeId::new("A"),
                ],
            }
        );
    }

    #[test]
    fn delayed_feedback_cycle_is_allowed_and_reproduces_expected_sequence() {
        // Feedback loop:
        //   accum.out -> delay_tap.in (instant)
        //   delay_tap.out -> accum.in (delayed 1 tick)
        //
        // accum(x) = x.unwrap_or(0) + 1
        // delay_tap(x) = x
        // Tick 0: accum reads None -> outputs 1 -> delay_tap outputs 1 -> buffer stores 1
        // Tick 1: accum reads 1    -> outputs 2 -> delay_tap outputs 2 -> buffer stores 2
        // Tick 2: accum reads 2    -> outputs 3 -> delay_tap outputs 3 -> buffer stores 3
        let mut graph = ConnectionGraph::new();
        let accum_idx = graph
            .add_node(
                NodeSpec::new("accum")
                    .with_input("feedback", PortType::I64)
                    .with_output("sum", PortType::I64),
            )
            .expect("accum");
        let tap_idx = graph
            .add_node(
                NodeSpec::new("tap")
                    .with_input("val", PortType::I64)
                    .with_output("out", PortType::I64),
            )
            .expect("tap");

        graph
            .connect(ConnectionSpec::instant("accum", "sum", "tap", "val"))
            .expect("instant forward");
        graph
            .connect(ConnectionSpec::delayed(
                "tap", "out", "accum", "feedback", 1,
            ))
            .expect("delayed feedback");

        let plan = graph.prepare().expect("delayed cycle must compile");
        assert_eq!(plan.execution_order(), &[accum_idx, tap_idx]);
        assert_eq!(plan.delayed_transfers().len(), 1);
        assert_eq!(plan.delayed_transfers()[0].buffer_slot, BufferSlot(0));

        let mut runtime = PlanRuntime::<i64>::new(&plan);
        let mut observed = Vec::new();

        for t in 0..5u64 {
            runtime.execute_tick(&plan, Tick::new(t), |node, _tick, inputs, outputs| {
                if node == accum_idx {
                    let prev = inputs[0].unwrap_or(0);
                    outputs[0] = Some(prev + 1);
                } else if node == tap_idx {
                    outputs[0] = inputs[0];
                }
            });
            observed.push(
                runtime
                    .output(accum_idx, PortIndex(0))
                    .copied()
                    .expect("accum output"),
            );
        }

        assert_eq!(observed, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn multi_tick_delay_reproduces_exact_lag_sequence() {
        let mut graph = ConnectionGraph::new();
        let src_idx = graph
            .add_node(NodeSpec::new("src").with_output("out", PortType::U64))
            .expect("src");
        let dst_idx = graph
            .add_node(
                NodeSpec::new("dst")
                    .with_input("in", PortType::U64)
                    .with_output("seen", PortType::U64),
            )
            .expect("dst");

        graph
            .connect(ConnectionSpec::delayed("src", "out", "dst", "in", 2))
            .expect("delay 2");

        let plan = graph.prepare().expect("compile");
        let mut runtime = PlanRuntime::<u64>::new(&plan);
        let mut seen = Vec::new();

        for t in 0..6u64 {
            runtime.execute_tick(&plan, Tick::new(t), |node, tick, inputs, outputs| {
                if node == src_idx {
                    outputs[0] = Some(tick.as_u64() + 10);
                } else if node == dst_idx {
                    outputs[0] = inputs[0];
                }
            });
            seen.push(runtime.output(dst_idx, PortIndex(0)).copied());
        }

        // Con delay 2: ticks 0 y 1 ven None; ticks 2..5 ven valores de ticks 0..3 (10..13).
        assert_eq!(
            seen,
            vec![None, None, Some(10), Some(11), Some(12), Some(13)]
        );
    }

    #[test]
    fn topological_order_is_stable_and_resolves_indices_at_prepare() {
        // Nodos independientes: n0, n1, n2, n3 con dependencia n2 -> n1.
        // Orden de inserción: n0 (0), n1 (1), n2 (2), n3 (3).
        // Al iniciar: n0, n2, n3 tienen in-degree 0 -> n0 va primero; luego n2 desbloquea n1;
        // entre n1 (1) y n3 (3) disponibles, desempate estable elige n1 antes que n3.
        // Orden esperado: [n0, n2, n1, n3].
        let mut graph = ConnectionGraph::new();
        let n0 = graph
            .add_node(NodeSpec::new("n0").with_output("o", PortType::Unit))
            .expect("n0");
        let n1 = graph
            .add_node(NodeSpec::new("n1").with_input("i", PortType::Unit))
            .expect("n1");
        let n2 = graph
            .add_node(NodeSpec::new("n2").with_output("o", PortType::Unit))
            .expect("n2");
        let n3 = graph
            .add_node(NodeSpec::new("n3").with_output("o", PortType::Unit))
            .expect("n3");

        graph
            .connect(ConnectionSpec::instant("n2", "o", "n1", "i"))
            .expect("n2->n1");

        let plan_a = graph.prepare().expect("prepare a");
        let plan_b = graph.prepare().expect("prepare b");

        assert_eq!(plan_a.execution_order(), &[n0, n2, n1, n3]);
        assert_eq!(plan_a.execution_order(), plan_b.execution_order());

        // Resolución de índices previa al tick.
        assert_eq!(plan_a.node_index(&NodeId::new("n2")), Some(n2));
        assert_eq!(
            plan_a.port_index(n2, &PortId::new("o")),
            Some((PortDirection::Output, PortIndex(0)))
        );
        assert_eq!(
            plan_a.port_index(n1, &PortId::new("i")),
            Some((PortDirection::Input, PortIndex(0)))
        );
        assert_eq!(
            plan_a.node_instant_outgoing(n2),
            &[CompiledTransfer {
                from_node: n2,
                from_port: PortIndex(0),
                to_node: n1,
                to_port: PortIndex(0),
            }]
        );
    }

    #[test]
    fn gog129_flat_ports_isolate_nodes_with_differing_port_counts() {
        let mut graph = ConnectionGraph::new();
        let a = graph
            .add_node(
                NodeSpec::new("a")
                    .with_input("i0", PortType::I64)
                    .with_output("o0", PortType::I64)
                    .with_output("o1", PortType::I64)
                    .with_output("o2", PortType::I64),
            )
            .unwrap();
        let b = graph.add_node(NodeSpec::new("b")).unwrap();
        let c = graph
            .add_node(
                NodeSpec::new("c")
                    .with_input("i0", PortType::I64)
                    .with_input("i1", PortType::I64)
                    .with_output("o0", PortType::I64),
            )
            .unwrap();
        graph
            .connect(ConnectionSpec::instant("a", "o2", "c", "i1"))
            .unwrap();
        graph
            .connect(ConnectionSpec::instant("a", "o0", "c", "i0"))
            .unwrap();
        let plan = graph.prepare().unwrap();
        let mut rt = PlanRuntime::<i64>::new(&plan);
        rt.set_input(a, PortIndex(0), Some(7));
        // Fuera de rango: ignorado, sin invadir nodos vecinos.
        rt.set_input(a, PortIndex(1), Some(99));
        rt.set_input(b, PortIndex(0), Some(99));
        rt.set_input(NodeIndex(42), PortIndex(0), Some(99));

        rt.execute_tick(&plan, Tick::new(0), |node, _t, inputs, outputs| {
            if node == a {
                assert_eq!(inputs.len(), 1);
                assert_eq!(outputs.len(), 3);
                let x = inputs[0].unwrap();
                outputs[0] = Some(x);
                outputs[1] = Some(x + 1);
                outputs[2] = Some(x + 2);
            } else if node == b {
                assert!(inputs.is_empty() && outputs.is_empty());
            } else if node == c {
                assert_eq!(inputs, &[Some(7), Some(9)]);
                outputs[0] = Some(inputs[0].unwrap() + inputs[1].unwrap());
            }
        });

        assert_eq!(rt.output(a, PortIndex(1)), Some(&8));
        assert_eq!(rt.output(a, PortIndex(3)), None);
        assert_eq!(rt.output(b, PortIndex(0)), None);
        assert_eq!(rt.output(c, PortIndex(0)), Some(&16));
        assert_eq!(rt.output(NodeIndex(42), PortIndex(0)), None);
    }
}
