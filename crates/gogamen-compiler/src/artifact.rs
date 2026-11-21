//! Artefacto de plan compilado versionado (`std`-only, sin `serde` ni JVM).
//!
//! Permite persistir un [`ExecutablePlan`] compilado junto con la especificación
//! normalizada, índices pre-resueltos (`OperationIndex`, `FactoryIndex`,
//! `ConstantIndex`, topología compilada), metadatos de versión/runtime/catálogo y
//! checksums SHA-256 deterministas (`spec_hash`, `artifact_hash`).
//!
//! Cualquier artefacto corrupto o incompatible se rechaza **antes** de instanciar
//! ningún recurso o actuador, exigiendo recompilación explícita mediante
//! [`CompiledPlanArtifact::recompile_from_embedded_spec`].

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Write as _};

use gogamen_core::{
    BufferSlot, CapabilityCatalog, CapabilityVersion, CompiledDelayedTransfer, CompiledNodeMeta,
    CompiledPlan, CompiledPortMeta, CompiledTransfer, ModuleId, ModuleOrigin, NodeId, NodeIndex,
    OperationBinding, OperationId, PortDirection, PortId, PortIndex, PortType, TemporalProfile,
};
use gogamen_spec::{
    JsonNumber, JsonValue, SpecError, SystemSpec, parse_json as spec_parse_json,
    write_canonical_f64 as spec_write_canonical_f64, write_json_string,
};

use crate::compiler::SpecCompiler;
use crate::error::CompileError;
use crate::plan::{
    COMPILER_PLAN_VERSION, ConstantIndex, ConstantValue, ExecutablePlan, FactoryIndex,
    InstantiatedPlan, OperationIndex, ResolvedFactory, ResolvedOperation, ResourceInitializer,
};

/// Versión canónica del formato de artefacto compilado portable.
pub const COMPILED_ARTIFACT_VERSION: &str = "gogamen.compiled_artifact.v1";

/// Versión del runtime contra la que se compilan y validan los artefactos.
pub const COMPILED_ARTIFACT_RUNTIME_VERSION: &str = gogamen_core::core_version();

/// Error tipado al compilar, parsear, verificar o cargar un [`CompiledPlanArtifact`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactError {
    /// El JSON del artefacto está truncado, malformado o viola la estructura esperada.
    MalformedArtifact {
        /// Motivo del rechazo estructural.
        reason: String,
    },
    /// El checksum criptográfico SHA-256 del artefacto o de la spec embebida no coincide.
    ChecksumMismatch {
        /// Campo de checksum afectado (`"artifact_hash"` o `"spec_hash"`).
        field: &'static str,
        /// Hash esperado calculado a partir del contenido.
        expected: String,
        /// Hash encontrado en el artefacto.
        found: String,
    },
    /// La versión del formato de artefacto o del compilador es incompatible.
    IncompatibleArtifactVersion {
        /// Versión esperada.
        expected: String,
        /// Versión encontrada en el artefacto.
        found: String,
    },
    /// La versión del runtime es incompatible con la del artefacto.
    IncompatibleRuntimeVersion {
        /// Versión esperada.
        expected: String,
        /// Versión encontrada en el artefacto.
        found: String,
    },
    /// El catálogo de capacidades no coincide o carece de operaciones compatibles.
    IncompatibleCatalog {
        /// Motivo de la incompatibilidad con el catálogo.
        reason: String,
    },
    /// Error de compilación al compilar o recompilar explícitamente la spec.
    Compile(CompileError),
    /// Error al parsear o serializar la especificación declarativa embebida.
    Spec(SpecError),
}

impl From<CompileError> for ArtifactError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

impl From<SpecError> for ArtifactError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MalformedArtifact { reason } => write!(f, "malformed artifact: {reason}"),
            Self::ChecksumMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "artifact checksum mismatch on '{field}': expected {expected}, found {found}"
            ),
            Self::IncompatibleArtifactVersion { expected, found } => write!(
                f,
                "incompatible artifact version: expected '{expected}', found '{found}'"
            ),
            Self::IncompatibleRuntimeVersion { expected, found } => write!(
                f,
                "incompatible runtime version: expected '{expected}', found '{found}'"
            ),
            Self::IncompatibleCatalog { reason } => write!(f, "incompatible catalog: {reason}"),
            Self::Compile(err) => write!(f, "compile error: {err}"),
            Self::Spec(err) => write!(f, "spec error: {err}"),
        }
    }
}

impl Error for ArtifactError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Compile(err) => Some(err),
            Self::Spec(err) => Some(err),
            _ => None,
        }
    }
}

/// Error unificado para [`CompiledPlanArtifact::load_and_instantiate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactLoadAndInitError<E> {
    /// Fallo al verificar o cargar el artefacto (ningún recurso fue instanciado).
    Artifact(ArtifactError),
    /// Fallo al instanciar recursos o actuadores tras validar y cargar el artefacto.
    Init(E),
}

/// Resultado de [`CompiledPlanArtifact::load_and_instantiate`].
pub type ArtifactLoadAndInitResult<R, A, E> =
    Result<(ExecutablePlan, InstantiatedPlan<R, A>), ArtifactLoadAndInitError<E>>;

/// Resumen inspeccionable de un [`CompiledPlanArtifact`] sin instanciar recursos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactInspection {
    /// Versión del formato de artefacto.
    pub artifact_version: String,
    /// Versión del runtime objetivo.
    pub runtime_version: String,
    /// Versión del compilador de planes.
    pub compiler_version: String,
    /// Huella SHA-256 del catálogo de capacidades usado al compilar.
    pub catalog_fingerprint: String,
    /// Hash SHA-256 de la especificación canónica embebida.
    pub spec_hash: String,
    /// Hash SHA-256 del artefacto completo.
    pub artifact_hash: String,
    /// Versión de esquema de la spec origen.
    pub schema_version: String,
    /// Identificador del sistema compilado.
    pub spec_id: String,
    /// Perfil temporal objetivo validado.
    pub temporal_profile: TemporalProfile,
    /// Huella semántica canónica del plan.
    pub semantic_fingerprint: u64,
    /// Número de nodos en la topología compilada.
    pub node_count: usize,
    /// Número de fábricas pre-resueltas.
    pub factory_count: usize,
    /// Número de operaciones pre-resueltas.
    pub operation_count: usize,
    /// Número de constantes escalares internadas.
    pub constant_count: usize,
    /// Número de transferencias retardadas.
    pub delayed_transfer_count: usize,
    /// Fábricas resueltas incluidas en el artefacto.
    pub factories: Vec<ResolvedFactory>,
    /// Orden topológico determinista de ejecución.
    pub execution_order: Vec<NodeIndex>,
}

/// Artefacto de plan compilado portable y versionado (`std`-only).
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledPlanArtifact {
    artifact_version: String,
    runtime_version: String,
    compiler_version: String,
    catalog_fingerprint: String,
    spec_hash: String,
    artifact_hash: String,
    spec_canonical_json: String,
    schema_version: String,
    spec_id: String,
    temporal_profile: TemporalProfile,
    semantic_fingerprint: u64,
    factories: Vec<ResolvedFactory>,
    operations: Vec<ResolvedOperation>,
    node_to_operation: Vec<OperationIndex>,
    constants: Vec<ConstantValue>,
    system_constants: Vec<ConstantIndex>,
    system_constant_lookup: BTreeMap<String, ConstantIndex>,
    node_constant_lookups: Vec<BTreeMap<String, ConstantIndex>>,
    nodes: Vec<CompiledNodeMeta>,
    execution_order: Vec<NodeIndex>,
    instant_outgoing: Vec<Vec<CompiledTransfer>>,
    delayed_transfers: Vec<CompiledDelayedTransfer>,
}

impl CompiledPlanArtifact {
    /// Compila una [`SystemSpec`] contra `catalog` bajo `temporal_profile` y construye
    /// un artefacto portable con checksums SHA-256.
    pub fn compile(
        spec: &SystemSpec,
        catalog: &CapabilityCatalog,
        temporal_profile: TemporalProfile,
    ) -> Result<Self, ArtifactError> {
        let plan = SpecCompiler::compile(spec, catalog, temporal_profile)?;
        let spec_canonical_json = spec.to_canonical_json()?;
        let spec_hash = sha256_hex(spec_canonical_json.as_bytes());
        let catalog_fp = catalog_fingerprint(catalog);

        let mut artifact = Self {
            artifact_version: COMPILED_ARTIFACT_VERSION.to_owned(),
            runtime_version: COMPILED_ARTIFACT_RUNTIME_VERSION.to_owned(),
            compiler_version: plan.compiler_version.clone(),
            catalog_fingerprint: catalog_fp,
            spec_hash,
            artifact_hash: String::new(),
            spec_canonical_json,
            schema_version: plan.schema_version.clone(),
            spec_id: plan.spec_id.clone(),
            temporal_profile: plan.temporal_profile,
            semantic_fingerprint: plan.semantic_fingerprint,
            factories: plan.factories.clone(),
            operations: plan.operations.clone(),
            node_to_operation: plan.node_to_operation.clone(),
            constants: plan.constants.clone(),
            system_constants: plan.system_constants.clone(),
            system_constant_lookup: plan.system_constant_lookup.clone(),
            node_constant_lookups: plan.node_constant_lookups.clone(),
            nodes: plan.graph_plan.nodes().to_vec(),
            execution_order: plan.graph_plan.execution_order().to_vec(),
            instant_outgoing: plan.graph_plan.instant_outgoing_all().to_vec(),
            delayed_transfers: plan.graph_plan.delayed_transfers().to_vec(),
        };

        let payload = artifact.canonical_payload_json(false)?;
        artifact.artifact_hash = sha256_hex(payload.as_bytes());
        Ok(artifact)
    }

    /// Serializa el artefacto completo en formato JSON canónico determinista.
    pub fn to_canonical_json(&self) -> Result<String, ArtifactError> {
        self.canonical_payload_json(true)
    }

    /// Parsea un artefacto desde JSON verificando estrictamente su estructura y
    /// checksums SHA-256 (`artifact_hash` y `spec_hash`).
    pub fn from_json(json: &str) -> Result<Self, ArtifactError> {
        let root = parse_json(json)?;
        let obj = expect_object(&root, "artifact root")?;

        let artifact_version = expect_string_field(obj, "artifact_version")?;
        let runtime_version = expect_string_field(obj, "runtime_version")?;
        let compiler_version = expect_string_field(obj, "compiler_version")?;
        let catalog_fp = expect_string_field(obj, "catalog_fingerprint")?;
        let spec_hash = expect_string_field(obj, "spec_hash")?;
        let artifact_hash = expect_string_field(obj, "artifact_hash")?;
        let spec_canonical_json = expect_string_field(obj, "spec_canonical_json")?;
        let schema_version = expect_string_field(obj, "schema_version")?;
        let spec_id = expect_string_field(obj, "spec_id")?;
        let temporal_profile =
            parse_temporal_profile(&expect_string_field(obj, "temporal_profile")?)?;
        let semantic_fingerprint =
            parse_u64_hex(&expect_string_field(obj, "semantic_fingerprint")?)?;

        let factories = parse_factories(expect_array_field(obj, "factories")?)?;
        let operations = parse_operations(expect_array_field(obj, "operations")?)?;
        let node_to_operation = parse_operation_indices(
            expect_array_field(obj, "node_to_operation")?,
            "node_to_operation",
        )?;
        let constants = parse_constants(expect_array_field(obj, "constants")?)?;
        let system_constants = parse_constant_indices(
            expect_array_field(obj, "system_constants")?,
            "system_constants",
        )?;
        let system_constant_lookup =
            parse_constant_lookup(expect_object_field(obj, "system_constant_lookup")?)?;
        let node_constant_lookups =
            parse_node_constant_lookups(expect_array_field(obj, "node_constant_lookups")?)?;
        let nodes = parse_compiled_nodes(expect_array_field(obj, "nodes")?)?;
        let execution_order = parse_node_indices(
            expect_array_field(obj, "execution_order")?,
            "execution_order",
        )?;
        let instant_outgoing =
            parse_instant_outgoing(expect_array_field(obj, "instant_outgoing")?)?;
        let delayed_transfers =
            parse_delayed_transfers(expect_array_field(obj, "delayed_transfers")?)?;

        let artifact = Self {
            artifact_version,
            runtime_version,
            compiler_version,
            catalog_fingerprint: catalog_fp,
            spec_hash,
            artifact_hash,
            spec_canonical_json,
            schema_version,
            spec_id,
            temporal_profile,
            semantic_fingerprint,
            factories,
            operations,
            node_to_operation,
            constants,
            system_constants,
            system_constant_lookup,
            node_constant_lookups,
            nodes,
            execution_order,
            instant_outgoing,
            delayed_transfers,
        };

        let expected_artifact_payload = artifact.canonical_payload_json(false)?;
        let expected_artifact_hash = sha256_hex(expected_artifact_payload.as_bytes());
        if artifact.artifact_hash != expected_artifact_hash {
            return Err(ArtifactError::ChecksumMismatch {
                field: "artifact_hash",
                expected: expected_artifact_hash,
                found: artifact.artifact_hash,
            });
        }

        let expected_spec_hash = sha256_hex(artifact.spec_canonical_json.as_bytes());
        if artifact.spec_hash != expected_spec_hash {
            return Err(ArtifactError::ChecksumMismatch {
                field: "spec_hash",
                expected: expected_spec_hash,
                found: artifact.spec_hash,
            });
        }

        Ok(artifact)
    }

    /// Devuelve un resumen inspeccionable del artefacto sin instanciar recursos.
    pub fn inspect(&self) -> ArtifactInspection {
        ArtifactInspection {
            artifact_version: self.artifact_version.clone(),
            runtime_version: self.runtime_version.clone(),
            compiler_version: self.compiler_version.clone(),
            catalog_fingerprint: self.catalog_fingerprint.clone(),
            spec_hash: self.spec_hash.clone(),
            artifact_hash: self.artifact_hash.clone(),
            schema_version: self.schema_version.clone(),
            spec_id: self.spec_id.clone(),
            temporal_profile: self.temporal_profile,
            semantic_fingerprint: self.semantic_fingerprint,
            node_count: self.nodes.len(),
            factory_count: self.factories.len(),
            operation_count: self.operations.len(),
            constant_count: self.constants.len(),
            delayed_transfer_count: self.delayed_transfers.len(),
            factories: self.factories.clone(),
            execution_order: self.execution_order.clone(),
        }
    }

    /// Parsea e inspecciona un artefacto JSON verificando su integridad.
    pub fn inspect_json(json: &str) -> Result<ArtifactInspection, ArtifactError> {
        let artifact = Self::from_json(json)?;
        Ok(artifact.inspect())
    }

    /// Versión del formato de artefacto.
    pub fn artifact_version(&self) -> &str {
        &self.artifact_version
    }

    /// Versión del runtime asociada al artefacto.
    pub fn runtime_version(&self) -> &str {
        &self.runtime_version
    }

    /// Huella SHA-256 del catálogo de capacidades.
    pub fn catalog_fingerprint(&self) -> &str {
        &self.catalog_fingerprint
    }

    /// Hash SHA-256 de la especificación canónica embebida.
    pub fn spec_hash(&self) -> &str {
        &self.spec_hash
    }

    /// Hash SHA-256 del artefacto completo.
    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    /// Especificación canónica JSON embebida en el artefacto.
    pub fn spec_canonical_json(&self) -> &str {
        &self.spec_canonical_json
    }

    /// Valida compatibilidad de versión, runtime y catálogo, verifica índices
    /// pre-resueltos y reconstruye un [`ExecutablePlan`] sin ejecutar código externo
    /// ni recompilar implícitamente.
    pub fn load_executable_plan(
        &self,
        catalog: &CapabilityCatalog,
    ) -> Result<ExecutablePlan, ArtifactError> {
        if self.artifact_version != COMPILED_ARTIFACT_VERSION {
            return Err(ArtifactError::IncompatibleArtifactVersion {
                expected: COMPILED_ARTIFACT_VERSION.to_owned(),
                found: self.artifact_version.clone(),
            });
        }
        if self.compiler_version != COMPILER_PLAN_VERSION {
            return Err(ArtifactError::IncompatibleArtifactVersion {
                expected: COMPILER_PLAN_VERSION.to_owned(),
                found: self.compiler_version.clone(),
            });
        }
        if self.runtime_version != COMPILED_ARTIFACT_RUNTIME_VERSION {
            return Err(ArtifactError::IncompatibleRuntimeVersion {
                expected: COMPILED_ARTIFACT_RUNTIME_VERSION.to_owned(),
                found: self.runtime_version.clone(),
            });
        }

        let current_catalog_fp = catalog_fingerprint(catalog);
        if self.catalog_fingerprint != current_catalog_fp {
            return Err(ArtifactError::IncompatibleCatalog {
                reason: format!(
                    "catalog fingerprint mismatch: expected {current_catalog_fp}, found {}",
                    self.catalog_fingerprint
                ),
            });
        }

        for (idx, factory) in self.factories.iter().enumerate() {
            if factory.index.as_usize() != idx {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "factory index {} does not match slot {}",
                        factory.index.as_usize(),
                        idx
                    ),
                });
            }
            let binding = OperationBinding::new(
                factory.module_id.clone(),
                factory.operation_id.clone(),
                factory.version,
            );
            let validated = catalog
                .validate_operation(&binding, self.temporal_profile)
                .map_err(|err| ArtifactError::IncompatibleCatalog {
                    reason: format!(
                        "factory '{}' ('{}::{}') failed catalog validation: {err}",
                        factory.index.as_usize(),
                        factory.module_id,
                        factory.operation_id
                    ),
                })?;
            if validated.origin != factory.origin
                || validated.operation.temporal_profile != factory.temporal_profile
            {
                return Err(ArtifactError::IncompatibleCatalog {
                    reason: format!(
                        "factory '{}::{}' metadata mismatch against registered catalog",
                        factory.module_id, factory.operation_id
                    ),
                });
            }
        }

        let node_count = self.nodes.len();
        if self.operations.len() != node_count || self.node_to_operation.len() != node_count {
            return Err(ArtifactError::MalformedArtifact {
                reason: "operations and node_to_operation lengths must match node_count".to_owned(),
            });
        }
        if self.node_constant_lookups.len() != node_count {
            return Err(ArtifactError::MalformedArtifact {
                reason: "node_constant_lookups length must match node_count".to_owned(),
            });
        }

        for (idx, op) in self.operations.iter().enumerate() {
            if op.index.as_usize() != idx {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "operation index {} does not match slot {}",
                        op.index.as_usize(),
                        idx
                    ),
                });
            }
            if op.node_index.0 >= node_count {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!("operation node_index {} out of bounds", op.node_index.0),
                });
            }
            if op.factory.as_usize() >= self.factories.len() {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "operation factory index {} out of bounds",
                        op.factory.as_usize()
                    ),
                });
            }
            let node_meta = &self.nodes[op.node_index.0];
            if op.input_count != node_meta.input_count || op.output_count != node_meta.output_count
            {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "operation {} port counts do not match node {}",
                        op.index.as_usize(),
                        op.node_index.0
                    ),
                });
            }
            for &c_idx in &op.constants {
                if c_idx.as_usize() >= self.constants.len() {
                    return Err(ArtifactError::MalformedArtifact {
                        reason: format!("constant index {} out of bounds", c_idx.as_usize()),
                    });
                }
            }
        }

        for &op_idx in &self.node_to_operation {
            if op_idx.as_usize() >= self.operations.len() {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "node_to_operation index {} out of bounds",
                        op_idx.as_usize()
                    ),
                });
            }
        }

        for &c_idx in &self.system_constants {
            if c_idx.as_usize() >= self.constants.len() {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!("system constant index {} out of bounds", c_idx.as_usize()),
                });
            }
        }
        for &c_idx in self.system_constant_lookup.values() {
            if c_idx.as_usize() >= self.constants.len() {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!(
                        "system constant lookup index {} out of bounds",
                        c_idx.as_usize()
                    ),
                });
            }
        }
        for lookup in &self.node_constant_lookups {
            for &c_idx in lookup.values() {
                if c_idx.as_usize() >= self.constants.len() {
                    return Err(ArtifactError::MalformedArtifact {
                        reason: format!(
                            "node constant lookup index {} out of bounds",
                            c_idx.as_usize()
                        ),
                    });
                }
            }
        }

        let graph_plan = CompiledPlan::from_parts(
            self.nodes.clone(),
            self.execution_order.clone(),
            self.instant_outgoing.clone(),
            self.delayed_transfers.clone(),
        )
        .map_err(|reason| ArtifactError::MalformedArtifact { reason })?;

        Ok(ExecutablePlan {
            schema_version: self.schema_version.clone(),
            compiler_version: self.compiler_version.clone(),
            spec_id: self.spec_id.clone(),
            temporal_profile: self.temporal_profile,
            semantic_fingerprint: self.semantic_fingerprint,
            graph_plan,
            factories: self.factories.clone(),
            operations: self.operations.clone(),
            node_to_operation: self.node_to_operation.clone(),
            constants: self.constants.clone(),
            system_constants: self.system_constants.clone(),
            system_constant_lookup: self.system_constant_lookup.clone(),
            node_constant_lookups: self.node_constant_lookups.clone(),
        })
    }

    /// Carga un [`ExecutablePlan`] validando integridad/compatibilidad y **solo si tiene éxito**
    /// instancia recursos y actuadores mediante `initializer`.
    pub fn load_and_instantiate<I: ResourceInitializer>(
        &self,
        catalog: &CapabilityCatalog,
        initializer: &mut I,
    ) -> ArtifactLoadAndInitResult<I::Resource, I::Actuator, I::Error> {
        let plan = self
            .load_executable_plan(catalog)
            .map_err(ArtifactLoadAndInitError::Artifact)?;
        let instantiated = plan
            .instantiate(initializer)
            .map_err(ArtifactLoadAndInitError::Init)?;
        Ok((plan, instantiated))
    }

    /// Recompila explícitamente un artefacto a partir de su `spec_canonical_json`
    /// embebida tras verificar su checksum `spec_hash`.
    pub fn recompile_from_embedded_spec(
        &self,
        catalog: &CapabilityCatalog,
        temporal_profile: TemporalProfile,
    ) -> Result<Self, ArtifactError> {
        let expected_spec_hash = sha256_hex(self.spec_canonical_json.as_bytes());
        if self.spec_hash != expected_spec_hash {
            return Err(ArtifactError::ChecksumMismatch {
                field: "spec_hash",
                expected: expected_spec_hash,
                found: self.spec_hash.clone(),
            });
        }
        let spec = SystemSpec::from_json(&self.spec_canonical_json)?;
        Self::compile(&spec, catalog, temporal_profile)
    }

    fn canonical_payload_json(&self, include_artifact_hash: bool) -> Result<String, ArtifactError> {
        let mut out = String::with_capacity(2048);
        out.push('{');

        if include_artifact_hash {
            write_kv_str(&mut out, "artifact_hash", &self.artifact_hash, true);
            out.push(',');
        }
        write_kv_str(&mut out, "artifact_version", &self.artifact_version, false);
        out.push(',');
        write_kv_str(
            &mut out,
            "catalog_fingerprint",
            &self.catalog_fingerprint,
            false,
        );
        out.push(',');
        write_kv_str(&mut out, "compiler_version", &self.compiler_version, false);
        out.push(',');
        write_json_string(&mut out, "constants");
        out.push(':');
        write_constants(&mut out, &self.constants)?;
        out.push(',');
        write_json_string(&mut out, "delayed_transfers");
        out.push(':');
        write_delayed_transfers(&mut out, &self.delayed_transfers);
        out.push(',');
        write_json_string(&mut out, "execution_order");
        out.push(':');
        write_node_indices(&mut out, &self.execution_order);
        out.push(',');
        write_json_string(&mut out, "factories");
        out.push(':');
        write_factories(&mut out, &self.factories);
        out.push(',');
        write_json_string(&mut out, "instant_outgoing");
        out.push(':');
        write_instant_outgoing(&mut out, &self.instant_outgoing);
        out.push(',');
        write_json_string(&mut out, "node_constant_lookups");
        out.push(':');
        write_node_constant_lookups(&mut out, &self.node_constant_lookups);
        out.push(',');
        write_json_string(&mut out, "node_to_operation");
        out.push(':');
        write_operation_indices(&mut out, &self.node_to_operation);
        out.push(',');
        write_json_string(&mut out, "nodes");
        out.push(':');
        write_compiled_nodes(&mut out, &self.nodes);
        out.push(',');
        write_json_string(&mut out, "operations");
        out.push(':');
        write_operations(&mut out, &self.operations);
        out.push(',');
        write_kv_str(&mut out, "runtime_version", &self.runtime_version, false);
        out.push(',');
        write_kv_str(&mut out, "schema_version", &self.schema_version, false);
        out.push(',');
        write_kv_str(
            &mut out,
            "semantic_fingerprint",
            &format!("{:016x}", self.semantic_fingerprint),
            false,
        );
        out.push(',');
        write_kv_str(
            &mut out,
            "spec_canonical_json",
            &self.spec_canonical_json,
            false,
        );
        out.push(',');
        write_kv_str(&mut out, "spec_hash", &self.spec_hash, false);
        out.push(',');
        write_kv_str(&mut out, "spec_id", &self.spec_id, false);
        out.push(',');
        write_json_string(&mut out, "system_constant_lookup");
        out.push(':');
        write_constant_lookup(&mut out, &self.system_constant_lookup);
        out.push(',');
        write_json_string(&mut out, "system_constants");
        out.push(':');
        write_constant_indices(&mut out, &self.system_constants);
        out.push(',');
        write_kv_str(
            &mut out,
            "temporal_profile",
            temporal_profile_str(self.temporal_profile),
            false,
        );

        out.push('}');
        Ok(out)
    }
}

/// Parsea un artefacto JSON, valida integridad/versión/catálogo y carga el [`ExecutablePlan`].
pub fn load_plan_from_artifact_json(
    json: &str,
    catalog: &CapabilityCatalog,
) -> Result<ExecutablePlan, ArtifactError> {
    let artifact = CompiledPlanArtifact::from_json(json)?;
    artifact.load_executable_plan(catalog)
}

/// Calcula una huella SHA-256 determinista de un [`CapabilityCatalog`].
pub fn catalog_fingerprint(catalog: &CapabilityCatalog) -> String {
    let mut canonical = String::new();
    for manifest in catalog.modules() {
        let _ = write!(
            canonical,
            "module:{}:{:?}:{};",
            manifest.id.as_str(),
            manifest.origin,
            manifest.version
        );
        for perm in manifest.granted_permissions.iter() {
            let _ = write!(canonical, "grant:{};", perm.as_str());
        }
        let mut ops: Vec<_> = manifest.operations.iter().collect();
        ops.sort_by(|a, b| a.id.cmp(&b.id));
        for op in ops {
            let _ = write!(
                canonical,
                "op:{}:{}:{:?};",
                op.id.as_str(),
                op.version,
                op.temporal_profile
            );
            for perm in op.required_permissions.iter() {
                let _ = write!(canonical, "req:{};", perm.as_str());
            }
            for effect in &op.declared_effects {
                let _ = write!(canonical, "eff:{effect:?};");
            }
            for port in &op.ports {
                let _ = write!(
                    canonical,
                    "port:{}:{:?}:{:?};",
                    port.id.as_str(),
                    port.direction,
                    port.port_type
                );
            }
        }
    }
    sha256_hex(canonical.as_bytes())
}

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4) en std puro
// ---------------------------------------------------------------------------

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_hex(data: &[u8]) -> String {
    let mut state: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((data.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.as_chunks::<64>().0 {
        let mut w = [0u32; 64];
        for (i, word) in w.iter_mut().take(16).enumerate() {
            let base = i * 4;
            *word = u32::from_be_bytes([
                chunk[base],
                chunk[base + 1],
                chunk[base + 2],
                chunk[base + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        state[0] = state[0].wrapping_add(a);
        state[1] = state[1].wrapping_add(b);
        state[2] = state[2].wrapping_add(c);
        state[3] = state[3].wrapping_add(d);
        state[4] = state[4].wrapping_add(e);
        state[5] = state[5].wrapping_add(f);
        state[6] = state[6].wrapping_add(g);
        state[7] = state[7].wrapping_add(h);
    }

    let mut hex = String::with_capacity(64);
    for word in state {
        let _ = write!(hex, "{word:08x}");
    }
    hex
}

// ---------------------------------------------------------------------------
// Serialización JSON canónica en std puro
// ---------------------------------------------------------------------------

fn write_kv_str(out: &mut String, key: &str, value: &str, _first: bool) {
    write_json_string(out, key);
    out.push(':');
    write_json_string(out, value);
}

fn write_canonical_f64(out: &mut String, value: f64) -> Result<(), ArtifactError> {
    spec_write_canonical_f64(out, value).map_err(|_| ArtifactError::MalformedArtifact {
        reason: "non-finite float in constant".to_owned(),
    })
}

fn temporal_profile_str(profile: TemporalProfile) -> &'static str {
    match profile {
        TemporalProfile::BoundedTick => "bounded_tick",
        TemporalProfile::BoundedRt => "bounded_rt",
        TemporalProfile::BestEffort => "best_effort",
        TemporalProfile::UnboundedBackground => "unbounded_background",
    }
}

fn module_origin_str(origin: ModuleOrigin) -> &'static str {
    match origin {
        ModuleOrigin::Builtin => "builtin",
        ModuleOrigin::ThirdParty => "third_party",
    }
}

fn port_direction_str(dir: PortDirection) -> &'static str {
    match dir {
        PortDirection::Input => "input",
        PortDirection::Output => "output",
    }
}

fn write_port_type(out: &mut String, port_type: &PortType) {
    match port_type {
        PortType::Bool => write_json_string(out, "bool"),
        PortType::I64 => write_json_string(out, "i64"),
        PortType::U64 => write_json_string(out, "u64"),
        PortType::F64 => write_json_string(out, "f64"),
        PortType::String => write_json_string(out, "string"),
        PortType::Unit => write_json_string(out, "unit"),
        PortType::Custom(name) => {
            out.push_str("{\"custom\":");
            write_json_string(out, name);
            out.push('}');
        }
    }
}

fn write_constants(out: &mut String, constants: &[ConstantValue]) -> Result<(), ArtifactError> {
    out.push('[');
    for (idx, c) in constants.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        match c {
            ConstantValue::Bool(v) => {
                let _ = write!(out, "{{\"type\":\"bool\",\"value\":{v}}}");
            }
            ConstantValue::I64(v) => {
                let _ = write!(out, "{{\"type\":\"i64\",\"value\":{v}}}");
            }
            ConstantValue::F64(v) => {
                out.push_str("{\"type\":\"f64\",\"value\":");
                write_canonical_f64(out, *v)?;
                out.push('}');
            }
            ConstantValue::String(s) => {
                out.push_str("{\"type\":\"string\",\"value\":");
                write_json_string(out, s);
                out.push('}');
            }
        }
    }
    out.push(']');
    Ok(())
}

fn write_constant_indices(out: &mut String, indices: &[ConstantIndex]) {
    out.push('[');
    for (idx, val) in indices.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{}", val.as_usize());
    }
    out.push(']');
}

fn write_operation_indices(out: &mut String, indices: &[OperationIndex]) {
    out.push('[');
    for (idx, val) in indices.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{}", val.as_usize());
    }
    out.push(']');
}

fn write_node_indices(out: &mut String, indices: &[NodeIndex]) {
    out.push('[');
    for (idx, val) in indices.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(out, "{}", val.0);
    }
    out.push(']');
}

fn write_constant_lookup(out: &mut String, map: &BTreeMap<String, ConstantIndex>) {
    out.push('{');
    for (idx, (k, v)) in map.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_json_string(out, k);
        out.push(':');
        let _ = write!(out, "{}", v.as_usize());
    }
    out.push('}');
}

fn write_node_constant_lookups(out: &mut String, lookups: &[BTreeMap<String, ConstantIndex>]) {
    out.push('[');
    for (idx, map) in lookups.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_constant_lookup(out, map);
    }
    out.push(']');
}

fn write_factories(out: &mut String, factories: &[ResolvedFactory]) {
    out.push('[');
    for (idx, f) in factories.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        let _ = write!(out, "\"index\":{},", f.index.as_usize());
        write_kv_str(out, "module_id", f.module_id.as_str(), false);
        out.push(',');
        write_kv_str(out, "operation_id", f.operation_id.as_str(), false);
        out.push(',');
        write_kv_str(out, "origin", module_origin_str(f.origin), false);
        out.push(',');
        write_kv_str(
            out,
            "temporal_profile",
            temporal_profile_str(f.temporal_profile),
            false,
        );
        out.push(',');
        write_kv_str(out, "version", &f.version.to_string(), false);
        out.push('}');
    }
    out.push(']');
}

fn write_operations(out: &mut String, operations: &[ResolvedOperation]) {
    out.push('[');
    for (idx, op) in operations.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        out.push_str("\"constants\":");
        write_constant_indices(out, &op.constants);
        let _ = write!(
            out,
            ",\"factory\":{},\"index\":{},\"input_count\":{},\"node_index\":{},\"output_count\":{},",
            op.factory.as_usize(),
            op.index.as_usize(),
            op.input_count,
            op.node_index.0,
            op.output_count
        );
        write_kv_str(
            out,
            "temporal_profile",
            temporal_profile_str(op.temporal_profile),
            false,
        );
        out.push('}');
    }
    out.push(']');
}

fn write_compiled_nodes(out: &mut String, nodes: &[CompiledNodeMeta]) {
    out.push('[');
    for (idx, node) in nodes.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        out.push('{');
        write_kv_str(out, "id", node.id.as_str(), false);
        let _ = write!(
            out,
            ",\"index\":{},\"input_count\":{},\"output_count\":{},\"ports\":[",
            node.index.0, node.input_count, node.output_count
        );
        for (p_idx, port) in node.ports.iter().enumerate() {
            if p_idx > 0 {
                out.push(',');
            }
            out.push('{');
            write_kv_str(out, "direction", port_direction_str(port.direction), false);
            out.push(',');
            write_kv_str(out, "id", port.id.as_str(), false);
            let _ = write!(out, ",\"index\":{},\"port_type\":", port.index.0);
            write_port_type(out, &port.port_type);
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push(']');
}

fn write_instant_outgoing(out: &mut String, matrix: &[Vec<CompiledTransfer>]) {
    out.push('[');
    for (i, transfers) in matrix.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push('[');
        for (j, t) in transfers.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"from_node\":{},\"from_port\":{},\"to_node\":{},\"to_port\":{}}}",
                t.from_node.0, t.from_port.0, t.to_node.0, t.to_port.0
            );
        }
        out.push(']');
    }
    out.push(']');
}

fn write_delayed_transfers(out: &mut String, delayed: &[CompiledDelayedTransfer]) {
    out.push('[');
    for (idx, d) in delayed.iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{{\"buffer_slot\":{},\"delay_ticks\":{},\"from_node\":{},\"from_port\":{},\"to_node\":{},\"to_port\":{}}}",
            d.buffer_slot.0, d.delay_ticks, d.from_node.0, d.from_port.0, d.to_node.0, d.to_port.0
        );
    }
    out.push(']');
}

// ---------------------------------------------------------------------------
// Parser JSON estricto en std puro + deserializadores tipados
// ---------------------------------------------------------------------------

fn parse_json(input: &str) -> Result<JsonValue, ArtifactError> {
    spec_parse_json(input).map_err(|err| ArtifactError::MalformedArtifact {
        reason: err.to_string(),
    })
}

fn expect_object<'a>(
    val: &'a JsonValue,
    ctx: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, ArtifactError> {
    match val {
        JsonValue::Object(map) => Ok(map),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected object for {ctx}"),
        }),
    }
}

fn expect_field<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<&'a JsonValue, ArtifactError> {
    obj.get(field)
        .ok_or_else(|| ArtifactError::MalformedArtifact {
            reason: format!("missing required field '{field}'"),
        })
}

fn expect_string_field(
    obj: &BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<String, ArtifactError> {
    match expect_field(obj, field)? {
        JsonValue::String(s) => Ok(s.clone()),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected string for field '{field}'"),
        }),
    }
}

fn expect_array_field<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<&'a [JsonValue], ArtifactError> {
    match expect_field(obj, field)? {
        JsonValue::Array(arr) => Ok(arr.as_slice()),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected array for field '{field}'"),
        }),
    }
}

fn expect_object_field<'a>(
    obj: &'a BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<&'a BTreeMap<String, JsonValue>, ArtifactError> {
    match expect_field(obj, field)? {
        JsonValue::Object(map) => Ok(map),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected object for field '{field}'"),
        }),
    }
}

fn expect_usize(val: &JsonValue, ctx: &str) -> Result<usize, ArtifactError> {
    match val {
        JsonValue::Number(JsonNumber::U64(n)) => {
            usize::try_from(*n).map_err(|_| ArtifactError::MalformedArtifact {
                reason: format!("usize overflow in {ctx}"),
            })
        }
        JsonValue::Number(JsonNumber::I64(n)) if *n >= 0 => {
            usize::try_from(*n as u64).map_err(|_| ArtifactError::MalformedArtifact {
                reason: format!("usize overflow in {ctx}"),
            })
        }
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected non-negative integer for {ctx}"),
        }),
    }
}

fn expect_u64(val: &JsonValue, ctx: &str) -> Result<u64, ArtifactError> {
    match val {
        JsonValue::Number(JsonNumber::U64(n)) => Ok(*n),
        JsonValue::Number(JsonNumber::I64(n)) if *n >= 0 => Ok(*n as u64),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("expected u64 for {ctx}"),
        }),
    }
}

fn expect_usize_field(
    obj: &BTreeMap<String, JsonValue>,
    field: &str,
) -> Result<usize, ArtifactError> {
    expect_usize(expect_field(obj, field)?, field)
}

fn expect_u64_field(obj: &BTreeMap<String, JsonValue>, field: &str) -> Result<u64, ArtifactError> {
    expect_u64(expect_field(obj, field)?, field)
}

fn parse_temporal_profile(raw: &str) -> Result<TemporalProfile, ArtifactError> {
    match raw {
        "bounded_tick" => Ok(TemporalProfile::BoundedTick),
        "bounded_rt" => Ok(TemporalProfile::BoundedRt),
        "best_effort" => Ok(TemporalProfile::BestEffort),
        "unbounded_background" => Ok(TemporalProfile::UnboundedBackground),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("unknown temporal_profile '{raw}'"),
        }),
    }
}

fn parse_module_origin(raw: &str) -> Result<ModuleOrigin, ArtifactError> {
    match raw {
        "builtin" => Ok(ModuleOrigin::Builtin),
        "third_party" => Ok(ModuleOrigin::ThirdParty),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("unknown module origin '{raw}'"),
        }),
    }
}

fn parse_port_direction(raw: &str) -> Result<PortDirection, ArtifactError> {
    match raw {
        "input" => Ok(PortDirection::Input),
        "output" => Ok(PortDirection::Output),
        _ => Err(ArtifactError::MalformedArtifact {
            reason: format!("unknown port direction '{raw}'"),
        }),
    }
}

fn parse_port_type(val: &JsonValue) -> Result<PortType, ArtifactError> {
    match val {
        JsonValue::String(s) => match s.as_str() {
            "bool" => Ok(PortType::Bool),
            "i64" => Ok(PortType::I64),
            "u64" => Ok(PortType::U64),
            "f64" => Ok(PortType::F64),
            "string" => Ok(PortType::String),
            "unit" => Ok(PortType::Unit),
            _ => Err(ArtifactError::MalformedArtifact {
                reason: format!("unknown port_type '{s}'"),
            }),
        },
        JsonValue::Object(map) => {
            let name = expect_string_field(map, "custom")?;
            Ok(PortType::custom(name))
        }
        _ => Err(ArtifactError::MalformedArtifact {
            reason: "invalid port_type".to_owned(),
        }),
    }
}

fn parse_capability_version(raw: &str) -> Result<CapabilityVersion, ArtifactError> {
    let mut parts = raw.split('.');
    let major = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| ArtifactError::MalformedArtifact {
            reason: format!("invalid capability version '{raw}'"),
        })?;
    let minor = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| ArtifactError::MalformedArtifact {
            reason: format!("invalid capability version '{raw}'"),
        })?;
    let patch = parts
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .ok_or_else(|| ArtifactError::MalformedArtifact {
            reason: format!("invalid capability version '{raw}'"),
        })?;
    if parts.next().is_some() {
        return Err(ArtifactError::MalformedArtifact {
            reason: format!("invalid capability version '{raw}'"),
        });
    }
    Ok(CapabilityVersion::new(major, minor, patch))
}

fn parse_u64_hex(raw: &str) -> Result<u64, ArtifactError> {
    u64::from_str_radix(raw, 16).map_err(|_| ArtifactError::MalformedArtifact {
        reason: format!("invalid hex u64 '{raw}'"),
    })
}

fn parse_factories(arr: &[JsonValue]) -> Result<Vec<ResolvedFactory>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = expect_object(item, "factory")?;
        out.push(ResolvedFactory {
            index: FactoryIndex::new(expect_usize_field(obj, "index")?),
            module_id: ModuleId::new(expect_string_field(obj, "module_id")?),
            operation_id: OperationId::new(expect_string_field(obj, "operation_id")?),
            version: parse_capability_version(&expect_string_field(obj, "version")?)?,
            origin: parse_module_origin(&expect_string_field(obj, "origin")?)?,
            temporal_profile: parse_temporal_profile(&expect_string_field(
                obj,
                "temporal_profile",
            )?)?,
        });
    }
    Ok(out)
}

fn parse_operations(arr: &[JsonValue]) -> Result<Vec<ResolvedOperation>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = expect_object(item, "operation")?;
        out.push(ResolvedOperation {
            index: OperationIndex::new(expect_usize_field(obj, "index")?),
            node_index: NodeIndex(expect_usize_field(obj, "node_index")?),
            factory: FactoryIndex::new(expect_usize_field(obj, "factory")?),
            temporal_profile: parse_temporal_profile(&expect_string_field(
                obj,
                "temporal_profile",
            )?)?,
            input_count: expect_usize_field(obj, "input_count")?,
            output_count: expect_usize_field(obj, "output_count")?,
            constants: parse_constant_indices(
                expect_array_field(obj, "constants")?,
                "operation.constants",
            )?,
        });
    }
    Ok(out)
}

fn parse_operation_indices(
    arr: &[JsonValue],
    ctx: &str,
) -> Result<Vec<OperationIndex>, ArtifactError> {
    arr.iter()
        .map(|v| Ok(OperationIndex::new(expect_usize(v, ctx)?)))
        .collect()
}

fn parse_constant_indices(
    arr: &[JsonValue],
    ctx: &str,
) -> Result<Vec<ConstantIndex>, ArtifactError> {
    arr.iter()
        .map(|v| Ok(ConstantIndex::new(expect_usize(v, ctx)?)))
        .collect()
}

fn parse_node_indices(arr: &[JsonValue], ctx: &str) -> Result<Vec<NodeIndex>, ArtifactError> {
    arr.iter()
        .map(|v| Ok(NodeIndex(expect_usize(v, ctx)?)))
        .collect()
}

fn parse_constants(arr: &[JsonValue]) -> Result<Vec<ConstantValue>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = expect_object(item, "constant")?;
        let ty = expect_string_field(obj, "type")?;
        let val = expect_field(obj, "value")?;
        let parsed = match (ty.as_str(), val) {
            ("bool", JsonValue::Bool(b)) => ConstantValue::Bool(*b),
            ("i64", JsonValue::Number(JsonNumber::I64(n))) => ConstantValue::I64(*n),
            ("i64", JsonValue::Number(JsonNumber::U64(n))) => {
                let signed = i64::try_from(*n).map_err(|_| ArtifactError::MalformedArtifact {
                    reason: "i64 constant out of range".to_owned(),
                })?;
                ConstantValue::I64(signed)
            }
            ("f64", JsonValue::Number(JsonNumber::F64(f))) => ConstantValue::F64(*f),
            ("f64", JsonValue::Number(JsonNumber::I64(n))) => ConstantValue::F64(*n as f64),
            ("f64", JsonValue::Number(JsonNumber::U64(n))) => ConstantValue::F64(*n as f64),
            ("string", JsonValue::String(s)) => ConstantValue::String(s.clone()),
            _ => {
                return Err(ArtifactError::MalformedArtifact {
                    reason: format!("invalid constant entry for type '{ty}'"),
                });
            }
        };
        out.push(parsed);
    }
    Ok(out)
}

fn parse_constant_lookup(
    map: &BTreeMap<String, JsonValue>,
) -> Result<BTreeMap<String, ConstantIndex>, ArtifactError> {
    let mut out = BTreeMap::new();
    for (k, v) in map {
        out.insert(k.clone(), ConstantIndex::new(expect_usize(v, k)?));
    }
    Ok(out)
}

fn parse_node_constant_lookups(
    arr: &[JsonValue],
) -> Result<Vec<BTreeMap<String, ConstantIndex>>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let map = expect_object(item, "node_constant_lookup")?;
        out.push(parse_constant_lookup(map)?);
    }
    Ok(out)
}

fn parse_compiled_nodes(arr: &[JsonValue]) -> Result<Vec<CompiledNodeMeta>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = expect_object(item, "compiled_node")?;
        let ports_arr = expect_array_field(obj, "ports")?;
        let mut ports = Vec::with_capacity(ports_arr.len());
        for p in ports_arr {
            let p_obj = expect_object(p, "compiled_port")?;
            ports.push(CompiledPortMeta {
                id: PortId::new(expect_string_field(p_obj, "id")?),
                direction: parse_port_direction(&expect_string_field(p_obj, "direction")?)?,
                port_type: parse_port_type(expect_field(p_obj, "port_type")?)?,
                index: PortIndex(expect_usize_field(p_obj, "index")?),
            });
        }
        out.push(CompiledNodeMeta {
            id: NodeId::new(expect_string_field(obj, "id")?),
            index: NodeIndex(expect_usize_field(obj, "index")?),
            input_count: expect_usize_field(obj, "input_count")?,
            output_count: expect_usize_field(obj, "output_count")?,
            ports,
        });
    }
    Ok(out)
}

fn parse_instant_outgoing(arr: &[JsonValue]) -> Result<Vec<Vec<CompiledTransfer>>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let JsonValue::Array(inner) = item else {
            return Err(ArtifactError::MalformedArtifact {
                reason: "expected array in instant_outgoing".to_owned(),
            });
        };
        let mut transfers = Vec::with_capacity(inner.len());
        for t in inner {
            let obj = expect_object(t, "instant_transfer")?;
            transfers.push(CompiledTransfer {
                from_node: NodeIndex(expect_usize_field(obj, "from_node")?),
                from_port: PortIndex(expect_usize_field(obj, "from_port")?),
                to_node: NodeIndex(expect_usize_field(obj, "to_node")?),
                to_port: PortIndex(expect_usize_field(obj, "to_port")?),
            });
        }
        out.push(transfers);
    }
    Ok(out)
}

fn parse_delayed_transfers(
    arr: &[JsonValue],
) -> Result<Vec<CompiledDelayedTransfer>, ArtifactError> {
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let obj = expect_object(item, "delayed_transfer")?;
        out.push(CompiledDelayedTransfer {
            from_node: NodeIndex(expect_usize_field(obj, "from_node")?),
            from_port: PortIndex(expect_usize_field(obj, "from_port")?),
            to_node: NodeIndex(expect_usize_field(obj, "to_node")?),
            to_port: PortIndex(expect_usize_field(obj, "to_port")?),
            delay_ticks: expect_u64_field(obj, "delay_ticks")?,
            buffer_slot: BufferSlot(expect_usize_field(obj, "buffer_slot")?),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gogamen_core::{
        CapabilityCatalog, CapabilityVersion, EffectKind, ModuleManifest, ModuleOrigin,
        OperationSpec, PortType, TemporalProfile,
    };
    use gogamen_spec::{
        SpecConfigValue, SystemConnectionSpec, SystemNodeSpec, SystemSpec,
        canonical_two_node_fixture,
    };

    #[derive(Default)]
    struct SpyInitializer {
        factory_inits: usize,
        actuator_inits: usize,
        bound_factories: Vec<(FactoryIndex, String, String)>,
        bound_operations: Vec<(OperationIndex, FactoryIndex)>,
    }

    impl ResourceInitializer for SpyInitializer {
        type Resource = String;
        type Actuator = String;
        type Error = &'static str;

        fn init_factory(
            &mut self,
            factory: &ResolvedFactory,
        ) -> Result<Self::Resource, Self::Error> {
            self.factory_inits += 1;
            self.bound_factories.push((
                factory.index,
                factory.module_id.as_str().to_owned(),
                factory.operation_id.as_str().to_owned(),
            ));
            Ok(format!("{}::{}", factory.module_id, factory.operation_id))
        }

        fn init_actuator(
            &mut self,
            operation: &ResolvedOperation,
            factory: &ResolvedFactory,
        ) -> Result<Self::Actuator, Self::Error> {
            self.actuator_inits += 1;
            self.bound_operations.push((operation.index, factory.index));
            Ok(format!(
                "op#{}:factory#{}",
                operation.index.as_usize(),
                factory.index.as_usize()
            ))
        }
    }

    fn fixture_catalog() -> CapabilityCatalog {
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

        catalog
            .register_module(observer)
            .expect("register observer");
        catalog.register_module(policy).expect("register policy");
        catalog
    }

    fn recompute_hash(artifact: &mut CompiledPlanArtifact) {
        let payload = artifact
            .canonical_payload_json(false)
            .expect("canonical payload");
        artifact.artifact_hash = sha256_hex(payload.as_bytes());
    }

    #[test]
    fn sha256_matches_nist_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn ac1_compile_persist_inspect_and_load_canonical_fixture_without_jvm() {
        let spec = canonical_two_node_fixture();
        let catalog = fixture_catalog();

        let artifact = CompiledPlanArtifact::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("compile canonical fixture to artifact");
        let json = artifact
            .to_canonical_json()
            .expect("serialize artifact to canonical JSON");

        let inspection = CompiledPlanArtifact::inspect_json(&json).expect("inspect artifact");
        assert_eq!(inspection.artifact_version, COMPILED_ARTIFACT_VERSION);
        assert_eq!(
            inspection.runtime_version,
            COMPILED_ARTIFACT_RUNTIME_VERSION
        );
        assert_eq!(inspection.compiler_version, COMPILER_PLAN_VERSION);
        assert_eq!(inspection.spec_id, spec.spec_id);
        assert_eq!(inspection.node_count, 2);
        assert_eq!(inspection.factory_count, 2);
        assert_eq!(inspection.operation_count, 2);
        assert_eq!(
            inspection.catalog_fingerprint,
            catalog_fingerprint(&catalog)
        );
        assert_eq!(inspection.artifact_hash.len(), 64);
        assert_eq!(inspection.spec_hash.len(), 64);

        let loaded = load_plan_from_artifact_json(&json, &catalog).expect("load artifact");
        let direct = SpecCompiler::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("direct compile");

        assert_eq!(loaded.semantic_fingerprint(), direct.semantic_fingerprint());
        assert_eq!(loaded.execution_order(), direct.execution_order());
        assert_eq!(loaded.factories(), direct.factories());
        assert_eq!(loaded.operations(), direct.operations());
        assert_eq!(loaded.constants(), direct.constants());
        assert_eq!(loaded.node_count(), direct.node_count());
    }

    #[test]
    fn ac1_preserves_delayed_topology_and_constants_across_round_trip() {
        let spec = SystemSpec::new("delayed-loop")
            .with_config("dt_secs", SpecConfigValue::F64(0.01))
            .with_node(
                SystemNodeSpec::new("obs", "observer::observer")
                    .with_output("observation", PortType::F64),
            )
            .with_node(
                SystemNodeSpec::new("pol", "policy::policy")
                    .with_input("observation", PortType::F64)
                    .with_output("action", PortType::I64)
                    .with_config("gain", SpecConfigValue::F64(2.5)),
            )
            .with_connection(SystemConnectionSpec::delayed(
                "obs",
                "observation",
                "pol",
                "observation",
                2,
            ));
        let catalog = fixture_catalog();

        let artifact = CompiledPlanArtifact::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("compile delayed spec");
        let json = artifact.to_canonical_json().expect("canonical json");
        let loaded = load_plan_from_artifact_json(&json, &catalog).expect("load delayed plan");

        assert_eq!(loaded.delayed_transfers().len(), 1);
        assert_eq!(loaded.delayed_transfers()[0].delay_ticks, 2);
        let dt_idx = loaded
            .system_constant_index("dt_secs")
            .expect("dt_secs constant");
        assert_eq!(loaded.constant(dt_idx), Some(&ConstantValue::F64(0.01)));
    }

    #[test]
    fn ac2_corrupt_or_tampered_artifact_fails_before_instantiation() {
        let spec = canonical_two_node_fixture();
        let catalog = fixture_catalog();
        let artifact = CompiledPlanArtifact::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("compile");
        let json = artifact.to_canonical_json().expect("json");

        // 1. Truncated JSON
        let truncated = &json[..json.len() / 2];
        let err = CompiledPlanArtifact::from_json(truncated).unwrap_err();
        assert!(matches!(err, ArtifactError::MalformedArtifact { .. }));

        // 2. Tampered JSON payload without updating artifact_hash
        let tampered = json.replace(&spec.spec_id, "tampered-sys");
        let err = CompiledPlanArtifact::from_json(&tampered).unwrap_err();
        assert!(matches!(
            err,
            ArtifactError::ChecksumMismatch {
                field: "artifact_hash",
                ..
            }
        ));

        // 3. Tampered spec_canonical_json with recomputed artifact_hash -> spec_hash mismatch
        let mut bad_spec_artifact = artifact.clone();
        bad_spec_artifact.spec_canonical_json = "{}".to_owned();
        recompute_hash(&mut bad_spec_artifact);
        let bad_spec_json = bad_spec_artifact.to_canonical_json().unwrap();
        let err = CompiledPlanArtifact::from_json(&bad_spec_json).unwrap_err();
        assert!(matches!(
            err,
            ArtifactError::ChecksumMismatch {
                field: "spec_hash",
                ..
            }
        ));
    }

    #[test]
    fn ac2_incompatible_version_runtime_or_catalog_fails_before_instantiation_and_supports_explicit_recompile()
     {
        let spec = canonical_two_node_fixture();
        let catalog = fixture_catalog();
        let artifact = CompiledPlanArtifact::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("compile");

        let mut spy = SpyInitializer::default();

        // Incompatible artifact version
        let mut bad_ver = artifact.clone();
        bad_ver.artifact_version = "gogamen.compiled_artifact.v999".to_owned();
        recompute_hash(&mut bad_ver);
        let err = bad_ver
            .load_and_instantiate(&catalog, &mut spy)
            .unwrap_err();
        assert!(matches!(
            err,
            ArtifactLoadAndInitError::Artifact(ArtifactError::IncompatibleArtifactVersion { .. })
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // Incompatible runtime version
        let mut bad_rt = artifact.clone();
        bad_rt.runtime_version = "99.0.0".to_owned();
        recompute_hash(&mut bad_rt);
        let err = bad_rt.load_and_instantiate(&catalog, &mut spy).unwrap_err();
        assert!(matches!(
            err,
            ArtifactLoadAndInitError::Artifact(ArtifactError::IncompatibleRuntimeVersion { .. })
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // Incompatible catalog (missing operation / upgraded version)
        let mut partial_catalog = CapabilityCatalog::new();
        let observer_only = ModuleManifest::new(
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
        partial_catalog
            .register_module(observer_only)
            .expect("register observer_only");

        let err = artifact
            .load_and_instantiate(&partial_catalog, &mut spy)
            .unwrap_err();
        assert!(matches!(
            err,
            ArtifactLoadAndInitError::Artifact(ArtifactError::IncompatibleCatalog { .. })
        ));
        assert_eq!(spy.factory_inits, 0);
        assert_eq!(spy.actuator_inits, 0);

        // Compatible evolved catalog (e.g. minor version bump) requires explicit recompile
        let mut evolved_catalog = fixture_catalog();
        let extra = ModuleManifest::new(
            "core.extra",
            ModuleOrigin::Builtin,
            CapabilityVersion::new(1, 0, 0),
        );
        evolved_catalog
            .register_module(extra)
            .expect("register extra");

        let err = artifact
            .load_and_instantiate(&evolved_catalog, &mut spy)
            .unwrap_err();
        assert!(matches!(
            err,
            ArtifactLoadAndInitError::Artifact(ArtifactError::IncompatibleCatalog { .. })
        ));
        assert_eq!(spy.factory_inits, 0);

        let recompiled = artifact
            .recompile_from_embedded_spec(&evolved_catalog, TemporalProfile::BoundedTick)
            .expect("explicit recompile succeeds");
        let (_plan, instantiated) = recompiled
            .load_and_instantiate(&evolved_catalog, &mut spy)
            .expect("load recompiled artifact");
        assert_eq!(instantiated.resources.len(), 2);
        assert_eq!(instantiated.actuators.len(), 2);
        assert_eq!(spy.factory_inits, 2);
        assert_eq!(spy.actuator_inits, 2);
    }

    #[test]
    fn ac3_binds_registered_operations_by_pre_resolved_indices_without_pointers_or_arbitrary_code()
    {
        let spec = canonical_two_node_fixture();
        let catalog = fixture_catalog();
        let artifact = CompiledPlanArtifact::compile(&spec, &catalog, TemporalProfile::BoundedTick)
            .expect("compile");
        let json = artifact.to_canonical_json().expect("json");

        // Artifact must not contain raw pointers or dynamic script/bytecode markers
        assert!(!json.contains("0x"));
        assert!(!json.contains("fn_ptr"));
        assert!(!json.contains("eval("));

        let mut spy = SpyInitializer::default();
        let (plan, instantiated) = artifact
            .load_and_instantiate(&catalog, &mut spy)
            .expect("load and instantiate");

        // Bound via FactoryIndex and OperationIndex directly
        assert_eq!(spy.bound_factories.len(), 2);
        assert_eq!(spy.bound_operations.len(), 2);
        for node_idx in plan.execution_order() {
            let op = plan.node_operation(*node_idx).expect("op by NodeIndex");
            let _resource = &instantiated.resources[op.factory.as_usize()];
            let _actuator = &instantiated.actuators[op.index.as_usize()];
        }

        // Tampering with an operation index out of bounds fails before instantiation
        let mut bad_idx = artifact.clone();
        bad_idx.operations[0].factory = FactoryIndex::new(99);
        recompute_hash(&mut bad_idx);
        let mut spy2 = SpyInitializer::default();
        let err = bad_idx
            .load_and_instantiate(&catalog, &mut spy2)
            .unwrap_err();
        assert!(matches!(
            err,
            ArtifactLoadAndInitError::Artifact(ArtifactError::MalformedArtifact { .. })
        ));
        assert_eq!(spy2.factory_inits, 0);
        assert_eq!(spy2.actuator_inits, 0);
    }
}
