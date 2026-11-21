//! Compilador declarativo de [`SystemSpec`] a [`ExecutablePlan`].

use std::collections::BTreeMap;

use gogamen_core::{
    CapabilityCatalog, CapabilityVersion, ModuleId, OperationBinding, OperationId, PortDirection,
    PortType, TemporalProfile, TraceHasher, ValidatedOperation,
};
use gogamen_spec::{
    SpecConfigMap, SpecConfigValue, SystemConnectionSpec, SystemNodeSpec, SystemSpec,
};

use crate::error::CompileError;
use crate::plan::{
    COMPILER_PLAN_VERSION, ConstantIndex, ConstantValue, ExecutablePlan, FactoryIndex,
    InstantiatedPlan, OperationIndex, ResolvedFactory, ResolvedOperation, ResourceInitializer,
};

/// Error unificado para [`SpecCompiler::compile_and_instantiate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileAndInitError<E> {
    /// Fallo durante la validación o compilación (ningún recurso fue inicializado).
    Compile(CompileError),
    /// Fallo durante la instanciación de recursos o actuadores tras compilar con éxito.
    Init(E),
}

/// Resultado de [`SpecCompiler::compile_and_instantiate`].
pub type CompileAndInitResult<R, A, E> =
    Result<(ExecutablePlan, InstantiatedPlan<R, A>), CompileAndInitError<E>>;

/// Compilador estático de especificaciones declarativas [`SystemSpec`] hacia
/// un [`ExecutablePlan`] inmutable y libre de búsquedas por cadena en `step`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecCompiler;

impl SpecCompiler {
    /// Valida y compila una [`SystemSpec`] contra un [`CapabilityCatalog`] bajo
    /// un [`TemporalProfile`] objetivo.
    ///
    /// Si la especificación es inválida (topología causal, capacidades, permisos,
    /// versiones, `BlockingIo` en ruta acotada o configuración), falla con [`CompileError`]
    /// antes de inicializar o asignar ningún recurso, fábrica ni actuador.
    pub fn compile(
        spec: &SystemSpec,
        catalog: &CapabilityCatalog,
        temporal_profile: TemporalProfile,
    ) -> Result<ExecutablePlan, CompileError> {
        spec.validate()?;
        let graph = spec.to_connection_graph()?;
        let graph_plan = graph.compile()?;

        let mut validated_nodes = Vec::with_capacity(spec.nodes.len());
        for node in &spec.nodes {
            let binding = resolve_operation_binding(node, catalog)?;
            let validated = catalog.validate_operation(&binding, temporal_profile)?;
            validate_node_ports(node, &validated)?;
            validated_nodes.push((node, validated));
        }

        let mut factory_map: BTreeMap<(ModuleId, OperationId, CapabilityVersion), ResolvedFactory> =
            BTreeMap::new();
        for (_, validated) in &validated_nodes {
            let key = (
                validated.module_id.clone(),
                validated.operation.id.clone(),
                validated.operation.version,
            );
            factory_map.entry(key).or_insert_with(|| ResolvedFactory {
                index: FactoryIndex::new(0),
                module_id: validated.module_id.clone(),
                operation_id: validated.operation.id.clone(),
                version: validated.operation.version,
                origin: validated.origin,
                temporal_profile: validated.operation.temporal_profile,
            });
        }

        let mut factories = Vec::with_capacity(factory_map.len());
        let mut factory_indices: BTreeMap<
            (ModuleId, OperationId, CapabilityVersion),
            FactoryIndex,
        > = BTreeMap::new();
        for (idx, (key, mut factory)) in factory_map.into_iter().enumerate() {
            let factory_index = FactoryIndex::new(idx);
            factory.index = factory_index;
            factory_indices.insert(key, factory_index);
            factories.push(factory);
        }

        let mut constants = Vec::new();
        let (system_constants, system_constant_lookup) =
            intern_config_map(&spec.config, &mut constants);

        let node_count = graph_plan.node_count();
        let mut operations_by_node: Vec<Option<ResolvedOperation>> = vec![None; node_count];
        let mut node_constant_lookups: Vec<BTreeMap<String, ConstantIndex>> =
            vec![BTreeMap::new(); node_count];

        for (node, validated) in &validated_nodes {
            let node_index = graph_plan
                .node_index(&node.id)
                .expect("validated node must exist in compiled graph");
            let factory_key = (
                validated.module_id.clone(),
                validated.operation.id.clone(),
                validated.operation.version,
            );
            let factory_index = factory_indices[&factory_key];
            let (node_const_indices, node_const_lookup) =
                intern_config_map(&node.config, &mut constants);
            let meta = graph_plan
                .node_meta(node_index)
                .expect("compiled graph must contain node metadata");

            let op_index = OperationIndex::new(node_index.0);
            operations_by_node[node_index.0] = Some(ResolvedOperation {
                index: op_index,
                node_index,
                factory: factory_index,
                temporal_profile,
                input_count: meta.input_count,
                output_count: meta.output_count,
                constants: node_const_indices,
            });
            node_constant_lookups[node_index.0] = node_const_lookup;
        }

        let operations: Vec<ResolvedOperation> = operations_by_node
            .into_iter()
            .map(|entry| entry.expect("every node index must have a resolved operation"))
            .collect();
        let node_to_operation: Vec<OperationIndex> =
            (0..node_count).map(OperationIndex::new).collect();

        let semantic_fingerprint =
            compute_semantic_fingerprint(spec, temporal_profile, &validated_nodes);

        Ok(ExecutablePlan {
            schema_version: spec.schema_version.clone(),
            compiler_version: COMPILER_PLAN_VERSION.to_owned(),
            spec_id: spec.spec_id.clone(),
            temporal_profile,
            semantic_fingerprint,
            graph_plan,
            factories,
            operations,
            node_to_operation,
            constants,
            system_constants,
            system_constant_lookup,
            node_constant_lookups,
        })
    }

    /// Compila una [`SystemSpec`] y, **solo si la compilación tiene éxito**, inicializa
    /// sus fábricas y actuadores mediante `initializer`.
    pub fn compile_and_instantiate<I: ResourceInitializer>(
        spec: &SystemSpec,
        catalog: &CapabilityCatalog,
        temporal_profile: TemporalProfile,
        initializer: &mut I,
    ) -> CompileAndInitResult<I::Resource, I::Actuator, I::Error> {
        let plan =
            Self::compile(spec, catalog, temporal_profile).map_err(CompileAndInitError::Compile)?;
        let instantiated = plan
            .instantiate(initializer)
            .map_err(CompileAndInitError::Init)?;
        Ok((plan, instantiated))
    }
}

fn resolve_operation_binding(
    node: &SystemNodeSpec,
    catalog: &CapabilityCatalog,
) -> Result<OperationBinding, CompileError> {
    let (default_module, default_op) = if let Some((module, op)) = node.kind.split_once("::") {
        if module.trim().is_empty() || op.trim().is_empty() {
            return Err(CompileError::InvalidNodeConfig {
                node_id: node.id.clone(),
                key: "kind".to_owned(),
                reason: format!("malformed module::operation kind '{}'", node.kind),
            });
        }
        (module, op)
    } else {
        (node.kind.as_str(), node.kind.as_str())
    };

    let module_name = optional_string_config(node, "module")?.unwrap_or(default_module);
    let operation_name = optional_string_config(node, "operation")?.unwrap_or(default_op);

    let module_id = ModuleId::new(module_name);
    let operation_id = OperationId::new(operation_name);

    let required_version = if let Some(version_str) = optional_string_config(node, "version")? {
        parse_capability_version(node, version_str)?
    } else {
        catalog
            .module(&module_id)
            .and_then(|manifest| {
                manifest
                    .operations
                    .iter()
                    .find(|op| op.id == operation_id)
                    .map(|op| op.version)
            })
            .unwrap_or_else(|| CapabilityVersion::new(0, 0, 0))
    };

    Ok(OperationBinding::new(
        module_id,
        operation_id,
        required_version,
    ))
}

fn optional_string_config<'a>(
    node: &'a SystemNodeSpec,
    key: &str,
) -> Result<Option<&'a str>, CompileError> {
    match node.config.get(key) {
        None => Ok(None),
        Some(SpecConfigValue::String(value)) => {
            if value.trim().is_empty() {
                return Err(CompileError::InvalidNodeConfig {
                    node_id: node.id.clone(),
                    key: key.to_owned(),
                    reason: "value must not be empty".to_owned(),
                });
            }
            Ok(Some(value.as_str()))
        }
        Some(_) => Err(CompileError::InvalidNodeConfig {
            node_id: node.id.clone(),
            key: key.to_owned(),
            reason: "expected string value".to_owned(),
        }),
    }
}

fn parse_capability_version(
    node: &SystemNodeSpec,
    raw: &str,
) -> Result<CapabilityVersion, CompileError> {
    let parts: Vec<&str> = raw.split('.').collect();
    if parts.is_empty() || parts.len() > 3 {
        return Err(CompileError::InvalidNodeConfig {
            node_id: node.id.clone(),
            key: "version".to_owned(),
            reason: format!("invalid semantic version '{raw}'"),
        });
    }

    let parse_part = |part: &str| -> Result<u32, CompileError> {
        part.parse::<u32>()
            .map_err(|_| CompileError::InvalidNodeConfig {
                node_id: node.id.clone(),
                key: "version".to_owned(),
                reason: format!("invalid numeric component '{part}' in version '{raw}'"),
            })
    };

    let major = parse_part(parts[0])?;
    let minor = if parts.len() >= 2 {
        parse_part(parts[1])?
    } else {
        0
    };
    let patch = if parts.len() == 3 {
        parse_part(parts[2])?
    } else {
        0
    };

    Ok(CapabilityVersion::new(major, minor, patch))
}

fn validate_node_ports(
    node: &SystemNodeSpec,
    validated: &ValidatedOperation,
) -> Result<(), CompileError> {
    if validated.operation.ports.is_empty() {
        return Ok(());
    }

    for node_port in &node.ports {
        let Some(contract_port) = validated
            .operation
            .ports
            .iter()
            .find(|p| p.id == node_port.id)
        else {
            return Err(CompileError::PortContractMismatch {
                node_id: node.id.clone(),
                port_id: node_port.id.clone(),
                reason: format!(
                    "port not declared by operation '{}::{}'",
                    validated.module_id, validated.operation.id
                ),
            });
        };

        if contract_port.direction != node_port.direction {
            return Err(CompileError::PortContractMismatch {
                node_id: node.id.clone(),
                port_id: node_port.id.clone(),
                reason: format!(
                    "direction mismatch: expected {}, found {}",
                    contract_port.direction, node_port.direction
                ),
            });
        }

        if contract_port.port_type != node_port.port_type {
            return Err(CompileError::PortContractMismatch {
                node_id: node.id.clone(),
                port_id: node_port.id.clone(),
                reason: format!(
                    "type mismatch: expected {}, found {}",
                    contract_port.port_type, node_port.port_type
                ),
            });
        }
    }

    Ok(())
}

fn intern_config_map(
    config: &SpecConfigMap,
    constants: &mut Vec<ConstantValue>,
) -> (Vec<ConstantIndex>, BTreeMap<String, ConstantIndex>) {
    let mut indices = Vec::with_capacity(config.len());
    let mut lookup = BTreeMap::new();

    for (key, value) in config {
        let const_index = ConstantIndex::new(constants.len());
        constants.push(to_constant_value(value));
        indices.push(const_index);
        lookup.insert(key.clone(), const_index);
    }

    (indices, lookup)
}

fn to_constant_value(value: &SpecConfigValue) -> ConstantValue {
    match value {
        SpecConfigValue::Bool(b) => ConstantValue::Bool(*b),
        SpecConfigValue::I64(i) => ConstantValue::I64(*i),
        SpecConfigValue::F64(f) => ConstantValue::F64(if *f == 0.0 { 0.0 } else { *f }),
        SpecConfigValue::String(s) => ConstantValue::String(s.clone()),
    }
}

fn compute_semantic_fingerprint(
    spec: &SystemSpec,
    temporal_profile: TemporalProfile,
    validated_nodes: &[(&SystemNodeSpec, ValidatedOperation)],
) -> u64 {
    let mut hasher = TraceHasher::new();

    hasher.write_str(COMPILER_PLAN_VERSION);
    hasher.write_str(&spec.schema_version);
    hasher.write_str(&spec.spec_id);
    hasher.write_u8(temporal_profile_tag(temporal_profile));

    hash_config_map(&mut hasher, &spec.config);

    let mut sorted_nodes: Vec<(&SystemNodeSpec, &ValidatedOperation)> = validated_nodes
        .iter()
        .map(|(node, validated)| (*node, validated))
        .collect();
    sorted_nodes.sort_by(|(a, _), (b, _)| a.id.cmp(&b.id));

    hasher.write_u64(sorted_nodes.len() as u64);
    for (node, validated) in sorted_nodes {
        hasher.write_str(node.id.as_str());
        hasher.write_str(&node.kind);
        hasher.write_str(validated.module_id.as_str());
        hasher.write_str(validated.operation.id.as_str());
        hasher.write_u64(u64::from(validated.operation.version.major));
        hasher.write_u64(u64::from(validated.operation.version.minor));
        hasher.write_u64(u64::from(validated.operation.version.patch));

        let mut sorted_ports: Vec<_> = node.ports.iter().collect();
        sorted_ports.sort_by(|a, b| {
            a.id.cmp(&b.id)
                .then_with(|| port_direction_tag(a.direction).cmp(&port_direction_tag(b.direction)))
                .then_with(|| a.port_type.cmp(&b.port_type))
        });

        hasher.write_u64(sorted_ports.len() as u64);
        for port in sorted_ports {
            hasher.write_str(port.id.as_str());
            hasher.write_u8(port_direction_tag(port.direction));
            hash_port_type(&mut hasher, &port.port_type);
        }

        hash_config_map(&mut hasher, &node.config);
    }

    let mut sorted_connections: Vec<&SystemConnectionSpec> = spec.connections.iter().collect();
    sorted_connections.sort_by(|a, b| {
        a.from_node
            .cmp(&b.from_node)
            .then_with(|| a.from_port.cmp(&b.from_port))
            .then_with(|| a.to_node.cmp(&b.to_node))
            .then_with(|| a.to_port.cmp(&b.to_port))
            .then_with(|| a.causal.delay_ticks().cmp(&b.causal.delay_ticks()))
    });

    hasher.write_u64(sorted_connections.len() as u64);
    for conn in sorted_connections {
        hasher.write_str(conn.from_node.as_str());
        hasher.write_str(conn.from_port.as_str());
        hasher.write_str(conn.to_node.as_str());
        hasher.write_str(conn.to_port.as_str());
        hasher.write_u64(conn.causal.delay_ticks());
    }

    hasher.finish()
}

fn hash_config_map(hasher: &mut TraceHasher, config: &SpecConfigMap) {
    hasher.write_u64(config.len() as u64);
    for (key, value) in config {
        hasher.write_str(key);
        match value {
            SpecConfigValue::Bool(b) => {
                hasher.write_u8(0);
                hasher.write_u8(u8::from(*b));
            }
            SpecConfigValue::I64(i) => {
                hasher.write_u8(1);
                hasher.write_u64(*i as u64);
            }
            SpecConfigValue::F64(f) => {
                hasher.write_u8(2);
                hasher.write_f64(*f);
            }
            SpecConfigValue::String(s) => {
                hasher.write_u8(3);
                hasher.write_str(s);
            }
        }
    }
}

fn hash_port_type(hasher: &mut TraceHasher, port_type: &PortType) {
    match port_type {
        PortType::Bool => hasher.write_u8(0),
        PortType::I64 => hasher.write_u8(1),
        PortType::U64 => hasher.write_u8(2),
        PortType::F64 => hasher.write_u8(3),
        PortType::String => hasher.write_u8(4),
        PortType::Unit => hasher.write_u8(5),
        PortType::Custom(name) => {
            hasher.write_u8(6);
            hasher.write_str(name);
        }
    }
}

const fn port_direction_tag(direction: PortDirection) -> u8 {
    match direction {
        PortDirection::Input => 0,
        PortDirection::Output => 1,
    }
}

const fn temporal_profile_tag(profile: TemporalProfile) -> u8 {
    match profile {
        TemporalProfile::BoundedTick => 0,
        TemporalProfile::BoundedRt => 1,
        TemporalProfile::BestEffort => 2,
        TemporalProfile::UnboundedBackground => 3,
    }
}
