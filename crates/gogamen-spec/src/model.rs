//! Modelo declarativo portable para `SystemSpec`, `AgentSpec` y `ExperimentSpec`.

use std::collections::{BTreeMap, BTreeSet};

use gogamen_core::{
    CausalKind, ConnectionGraph, ConnectionSpec, NodeId, NodeSpec, PortId, PortSpec, PortType,
};

use crate::error::SpecError;

/// Versión de esquema canónica para [`SystemSpec`].
pub const SYSTEM_SPEC_SCHEMA_VERSION: &str = "gogamen.system_spec.v0";
/// Versión de esquema canónica para [`AgentSpec`].
pub const AGENT_SPEC_SCHEMA_VERSION: &str = "gogamen.agent_spec.v0";
/// Versión de esquema canónica para [`ExperimentSpec`].
pub const EXPERIMENT_SPEC_SCHEMA_VERSION: &str = "gogamen.experiment_spec.v0";

/// Valor de configuración tipado escalar (sin DSL ni código arbitrario).
#[derive(Debug, Clone, PartialEq)]
pub enum SpecConfigValue {
    /// Valor booleano.
    Bool(bool),
    /// Entero con signo de 64 bits.
    I64(i64),
    /// Punto flotante finito de 64 bits (`-0.0` normalizado a `0.0`).
    F64(f64),
    /// Cadena de texto literal.
    String(String),
}

impl SpecConfigValue {
    /// Construye un valor `F64` validando finitud y normalizando `-0.0` a `0.0`.
    pub fn f64(value: f64) -> Result<Self, SpecError> {
        if !value.is_finite() {
            return Err(SpecError::NonFiniteFloat {
                context: "SpecConfigValue::f64".to_owned(),
            });
        }
        Ok(Self::F64(if value == 0.0 { 0.0 } else { value }))
    }

    pub(crate) fn validate(&self, key: &str) -> Result<(), SpecError> {
        if let Self::F64(v) = self
            && !v.is_finite()
        {
            return Err(SpecError::NonFiniteFloat {
                context: format!("config['{key}']"),
            });
        }
        Ok(())
    }
}

/// Mapa ordenado de parámetros escalares de configuración.
pub type SpecConfigMap = BTreeMap<String, SpecConfigValue>;

/// Especificación declarativa de un nodo dentro de [`SystemSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct SystemNodeSpec {
    /// Identificador único del nodo en el sistema.
    pub id: NodeId,
    /// Tipo o rol declarativo del nodo (por ejemplo `"world"`, `"planner"`).
    pub kind: String,
    /// Puertos tipados expuestos por el nodo.
    pub ports: Vec<PortSpec>,
    /// Parámetros de configuración escalares del nodo.
    pub config: SpecConfigMap,
}

impl SystemNodeSpec {
    /// Crea un nodo sin puertos ni configuración adicional.
    pub fn new(id: impl Into<NodeId>, kind: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind: kind.into(),
            ports: Vec::new(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un puerto al nodo.
    pub fn with_port(mut self, port: PortSpec) -> Self {
        self.ports.push(port);
        self
    }

    /// Añade un puerto de entrada tipado.
    pub fn with_input(mut self, id: impl Into<PortId>, port_type: PortType) -> Self {
        self.ports.push(PortSpec::input(id, port_type));
        self
    }

    /// Añade un puerto de salida tipado.
    pub fn with_output(mut self, id: impl Into<PortId>, port_type: PortType) -> Self {
        self.ports.push(PortSpec::output(id, port_type));
        self
    }

    /// Añade una entrada al mapa de configuración del nodo.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Convierte la definición estructural del nodo al contrato [`NodeSpec`] de `gogamen-core`.
    pub fn to_core_node_spec(&self) -> NodeSpec {
        NodeSpec {
            id: self.id.clone(),
            ports: self.ports.clone(),
        }
    }
}

/// Especificación declarativa de una conexión dirigida entre dos puertos de nodos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemConnectionSpec {
    /// Nodo productor.
    pub from_node: NodeId,
    /// Puerto de salida en el nodo productor.
    pub from_port: PortId,
    /// Nodo consumidor.
    pub to_node: NodeId,
    /// Puerto de entrada en el nodo consumidor.
    pub to_port: PortId,
    /// Semántica causal (`Instant` o `Delayed(ticks)`).
    pub causal: CausalKind,
}

impl SystemConnectionSpec {
    /// Crea una conexión instantánea (`delay_ticks == 0`).
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

    /// Crea una conexión con retardo en ticks (`0` equivale a instantánea).
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

    /// Convierte la conexión al contrato [`ConnectionSpec`] de `gogamen-core`.
    pub fn to_core_connection_spec(&self) -> ConnectionSpec {
        ConnectionSpec {
            from_node: self.from_node.clone(),
            from_port: self.from_port.clone(),
            to_node: self.to_node.clone(),
            to_port: self.to_port.clone(),
            causal: self.causal,
        }
    }
}

impl From<&SystemConnectionSpec> for ConnectionSpec {
    fn from(value: &SystemConnectionSpec) -> Self {
        value.to_core_connection_spec()
    }
}

impl From<SystemConnectionSpec> for ConnectionSpec {
    fn from(value: SystemConnectionSpec) -> Self {
        value.to_core_connection_spec()
    }
}

/// Especificación declarativa portable de un sistema compuesto de nodos y conexiones tipadas.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemSpec {
    /// Versión de esquema (`SYSTEM_SPEC_SCHEMA_VERSION`).
    pub schema_version: String,
    /// Identificador único del sistema.
    pub spec_id: String,
    /// Nodos declarados en el sistema.
    pub nodes: Vec<SystemNodeSpec>,
    /// Conexiones causales tipadas entre puertos.
    pub connections: Vec<SystemConnectionSpec>,
    /// Parámetros de configuración escalares del sistema.
    pub config: SpecConfigMap,
}

impl SystemSpec {
    /// Crea una especificación vacía con [`SYSTEM_SPEC_SCHEMA_VERSION`].
    pub fn new(spec_id: impl Into<String>) -> Self {
        Self {
            schema_version: SYSTEM_SPEC_SCHEMA_VERSION.to_owned(),
            spec_id: spec_id.into(),
            nodes: Vec::new(),
            connections: Vec::new(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un nodo a la especificación.
    pub fn with_node(mut self, node: SystemNodeSpec) -> Self {
        self.nodes.push(node);
        self
    }

    /// Añade una conexión a la especificación.
    pub fn with_connection(mut self, connection: SystemConnectionSpec) -> Self {
        self.connections.push(connection);
        self
    }

    /// Añade un parámetro escalar al mapa de configuración del sistema.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida el esquema, identificadores, configuración y topología causal completa.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != SYSTEM_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: SYSTEM_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        validate_non_empty(&self.spec_id, "SystemSpec.spec_id")?;
        validate_config_map(&self.config, "SystemSpec.config")?;

        for node in &self.nodes {
            validate_non_empty(&node.kind, "SystemNodeSpec.kind")?;
            validate_config_map(&node.config, "SystemNodeSpec.config")?;
            for port in &node.ports {
                if let PortType::Custom(name) = &port.port_type {
                    validate_non_empty(name, "PortType::Custom")?;
                }
            }
        }

        let _ = self.to_connection_graph()?;
        Ok(())
    }

    /// Convierte y compila la topología en un [`ConnectionGraph`] de `gogamen-core`.
    pub fn to_connection_graph(&self) -> Result<ConnectionGraph, SpecError> {
        let mut graph = ConnectionGraph::new();
        for node in &self.nodes {
            graph.add_node(node.to_core_node_spec())?;
        }
        for conn in &self.connections {
            graph.connect(conn.to_core_connection_spec())?;
        }
        let _ = graph.compile()?;
        Ok(graph)
    }
}

/// Especificación declarativa de un agente compuesta sobre [`SystemSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct AgentSpec {
    /// Versión de esquema (`AGENT_SPEC_SCHEMA_VERSION`).
    pub schema_version: String,
    /// Identificador del agente.
    pub agent_id: String,
    /// Sistema interno que implementa el agente.
    pub system: SystemSpec,
    /// Parámetros escalares específicos del agente.
    pub config: SpecConfigMap,
}

impl AgentSpec {
    /// Crea una especificación de agente compuesta sobre un [`SystemSpec`].
    pub fn new(agent_id: impl Into<String>, system: SystemSpec) -> Self {
        Self {
            schema_version: AGENT_SPEC_SCHEMA_VERSION.to_owned(),
            agent_id: agent_id.into(),
            system,
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un parámetro escalar al mapa de configuración del agente.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida el agente y su [`SystemSpec`] subyacente.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != AGENT_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: AGENT_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        validate_non_empty(&self.agent_id, "AgentSpec.agent_id")?;
        validate_config_map(&self.config, "AgentSpec.config")?;
        self.system.validate()?;
        Ok(())
    }
}

/// Especificación declarativa de un experimento compuesta sobre [`SystemSpec`] y [`AgentSpec`].
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentSpec {
    /// Versión de esquema (`EXPERIMENT_SPEC_SCHEMA_VERSION`).
    pub schema_version: String,
    /// Identificador del experimento.
    pub experiment_id: String,
    /// Semilla determinista maestra del experimento.
    pub seed: u64,
    /// Sistema de entorno / simulación base.
    pub system: SystemSpec,
    /// Agentes participantes compuestos sobre [`SystemSpec`].
    pub agents: Vec<AgentSpec>,
    /// Parámetros escalares del experimento.
    pub config: SpecConfigMap,
}

impl ExperimentSpec {
    /// Crea una especificación de experimento compuesta sobre un [`SystemSpec`].
    pub fn new(experiment_id: impl Into<String>, seed: u64, system: SystemSpec) -> Self {
        Self {
            schema_version: EXPERIMENT_SPEC_SCHEMA_VERSION.to_owned(),
            experiment_id: experiment_id.into(),
            seed,
            system,
            agents: Vec::new(),
            config: SpecConfigMap::new(),
        }
    }

    /// Añade un agente participante al experimento.
    pub fn with_agent(mut self, agent: AgentSpec) -> Self {
        self.agents.push(agent);
        self
    }

    /// Añade un parámetro escalar al mapa de configuración del experimento.
    pub fn with_config(mut self, key: impl Into<String>, value: SpecConfigValue) -> Self {
        self.config.insert(key.into(), value);
        self
    }

    /// Valida el experimento, su sistema base y todos sus agentes sin duplicados.
    pub fn validate(&self) -> Result<(), SpecError> {
        if self.schema_version != EXPERIMENT_SPEC_SCHEMA_VERSION {
            return Err(SpecError::IncompatibleSchemaVersion {
                expected: EXPERIMENT_SPEC_SCHEMA_VERSION,
                found: self.schema_version.clone(),
            });
        }
        validate_non_empty(&self.experiment_id, "ExperimentSpec.experiment_id")?;
        validate_config_map(&self.config, "ExperimentSpec.config")?;
        self.system.validate()?;

        let mut seen_agents = BTreeSet::new();
        for agent in &self.agents {
            agent.validate()?;
            if !seen_agents.insert(agent.agent_id.clone()) {
                return Err(SpecError::DuplicateIdentifier {
                    context: "ExperimentSpec.agents",
                    id: agent.agent_id.clone(),
                });
            }
        }
        Ok(())
    }
}

pub(crate) fn validate_non_empty(value: &str, context: &'static str) -> Result<(), SpecError> {
    if value.trim().is_empty() {
        return Err(SpecError::EmptyIdentifier { context });
    }
    Ok(())
}

pub(crate) fn validate_config_map(
    config: &SpecConfigMap,
    context: &'static str,
) -> Result<(), SpecError> {
    for (key, value) in config {
        validate_non_empty(key, context)?;
        value.validate(key)?;
    }
    Ok(())
}
