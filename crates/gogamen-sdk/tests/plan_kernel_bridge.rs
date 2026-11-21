//! Puente `StaticExtensionRegistry` -> `PlanRuntimeEngine` (GOG-132).

use std::sync::Arc;

use gogamen_core::{
    CapabilityVersion, EffectKind, ModuleManifest, ModuleOrigin, OperationSpec, Permission,
    PermissionSet, PortType, TemporalProfile,
};
use gogamen_sdk::{
    ExtensionActivationContext, ExtensionError, ExtensionKind, ExtensionManifest, ExternalKernel,
    PlanRuntimeEngine, PlanRuntimeError, PortValue, StaticExtensionRegistry, SystemConnectionSpec,
    SystemNodeSpec, SystemSpec,
};

#[derive(Debug)]
struct Negate;

impl ExternalKernel for Negate {
    fn eval(
        &self,
        _state: &mut Vec<PortValue>,
        inputs: &[Option<PortValue>],
        outputs: &mut [Option<PortValue>],
    ) {
        if let Some(Some(PortValue::F64(v))) = inputs.first() {
            outputs[0] = Some(PortValue::F64(-*v));
        }
    }
}

fn module() -> ModuleManifest {
    ModuleManifest::new(
        "ext.bridge",
        ModuleOrigin::ThirdParty,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("negate", CapabilityVersion::new(1, 0, 0))
            .with_input("in", PortType::F64)
            .with_output("out", PortType::F64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
}

fn manifest(perms: PermissionSet) -> ExtensionManifest {
    ExtensionManifest::new(
        "ext.bridge.negate",
        ExtensionKind::PlanKernel,
        ModuleOrigin::ThirdParty,
        "bridge_crate",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_required_permissions(perms)
}

fn spec() -> SystemSpec {
    SystemSpec::new("bridge_v0")
        .with_node(
            SystemNodeSpec::new("src", "core.math::source").with_output("out", PortType::F64),
        )
        .with_node(
            SystemNodeSpec::new("neg", "ext.bridge::negate")
                .with_input("in", PortType::F64)
                .with_output("out", PortType::F64),
        )
        .with_connection(SystemConnectionSpec::instant("src", "out", "neg", "in"))
}

#[test]
fn static_registry_bridges_to_plan_runtime_engine() {
    let mut registry = StaticExtensionRegistry::new();
    registry
        .register_plan_kernel(manifest(PermissionSet::empty()), module(), "negate", |_| {
            Ok(Arc::new(Negate))
        })
        .unwrap();

    let kernels = registry
        .build_kernel_registry(&ExtensionActivationContext::strict_empty())
        .unwrap();
    assert_eq!(kernels.len(), 1);

    let engine = PlanRuntimeEngine::with_external_kernels(kernels);
    let artifact = engine.compile_artifact(&spec()).unwrap();
    let summary = engine.run_artifact(&artifact, 2).unwrap();
    let plan = engine.load_artifact(&artifact).unwrap();
    let neg = plan.node_index(&gogamen_core::NodeId::new("neg")).unwrap();
    // core.math::source (initial 0.0, step 1.0) emite 1.0 en el tick 2 -> -1.0.
    assert_eq!(summary.final_outputs[neg.0][0], Some(PortValue::F64(-1.0)));

    // Sin el puente, el mismo plan no se instancia: error tipado.
    assert!(matches!(
        PlanRuntimeEngine::new().instantiate(&plan),
        Err(PlanRuntimeError::UnsupportedOperation { .. })
    ));
}

#[test]
fn bridge_requires_explicit_permissions_and_declared_operation() {
    let mut registry = StaticExtensionRegistry::new();
    let perms = PermissionSet::empty().with_permission(Permission::state_read());
    registry
        .register_plan_kernel(manifest(perms), module(), "negate", |_| {
            Ok(Arc::new(Negate))
        })
        .unwrap();
    assert!(matches!(
        registry.build_kernel_registry(&ExtensionActivationContext::strict_empty()),
        Err(ExtensionError::InsufficientPermissions { .. })
    ));

    let mut other = StaticExtensionRegistry::new();
    let bad = ExtensionManifest::new(
        "ext.bridge.bad",
        ExtensionKind::PlanKernel,
        ModuleOrigin::ThirdParty,
        "bridge_crate",
        CapabilityVersion::new(0, 1, 0),
    );
    assert!(matches!(
        other.register_plan_kernel(bad, module(), "missing", |_| Ok(Arc::new(Negate))),
        Err(ExtensionError::InvalidManifest { .. })
    ));
}
