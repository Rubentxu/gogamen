//! Configuración extensible mediante [`SystemSpec`] y compilación con [`SpecCompiler`].

use std::error::Error;
use std::fmt;

use gogamen_compiler::{CompileError, ExecutablePlan, SpecCompiler};
use gogamen_core::{
    CapabilityCatalog, CapabilityVersion, DiscreteAction, DomainError, EffectKind,
    ModuleContractError, ModuleManifest, ModuleOrigin, OperationSpec, PortType, SchedulerError,
    SystemOneError, TemporalProfile, WorkflowError,
};
use gogamen_spec::{SpecConfigValue, SpecError, SystemConnectionSpec, SystemNodeSpec, SystemSpec};

use crate::events::ServiceThresholds;

/// Error unificado del laboratorio de automatización.
#[derive(Debug)]
pub enum AutomationError {
    /// Configuración inválida del supervisor o sus umbrales.
    InvalidConfig(String),
    /// Error en la especificación declarativa [`SystemSpec`].
    Spec(SpecError),
    /// Error de compilación en [`SpecCompiler`].
    Compile(CompileError),
    /// Error en contratos de capacidades.
    Capability(ModuleContractError),
    /// Error del motor de workflows.
    Workflow(WorkflowError),
    /// Error del scheduler de eventos.
    Scheduler(SchedulerError),
    /// Error de dominio.
    Domain(DomainError),
    /// Error al consultar [`gogamen_core::SystemOneProvider`].
    SystemOne(SystemOneError),
}

impl fmt::Display for AutomationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid automation config: {msg}"),
            Self::Spec(err) => write!(f, "spec error: {err}"),
            Self::Compile(err) => write!(f, "compile error: {err}"),
            Self::Capability(err) => write!(f, "capability error: {err}"),
            Self::Workflow(err) => write!(f, "workflow error: {err}"),
            Self::Scheduler(err) => write!(f, "scheduler error: {err}"),
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::SystemOne(err) => write!(f, "system one error: {err}"),
        }
    }
}

impl Error for AutomationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(_) => None,
            Self::Spec(err) => Some(err),
            Self::Compile(err) => Some(err),
            Self::Capability(err) => Some(err),
            Self::Workflow(err) => Some(err),
            Self::Scheduler(err) => Some(err),
            Self::Domain(err) => Some(err),
            Self::SystemOne(err) => Some(err),
        }
    }
}

impl From<SpecError> for AutomationError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

impl From<CompileError> for AutomationError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

impl From<ModuleContractError> for AutomationError {
    fn from(value: ModuleContractError) -> Self {
        Self::Capability(value)
    }
}

impl From<WorkflowError> for AutomationError {
    fn from(value: WorkflowError) -> Self {
        Self::Workflow(value)
    }
}

impl From<SchedulerError> for AutomationError {
    fn from(value: SchedulerError) -> Self {
        Self::Scheduler(value)
    }
}

impl From<DomainError> for AutomationError {
    fn from(value: DomainError) -> Self {
        Self::Domain(value)
    }
}

impl From<SystemOneError> for AutomationError {
    fn from(value: SystemOneError) -> Self {
        Self::SystemOne(value)
    }
}

/// Acción de remediación sintética autorizable por [`gogamen_core::ActionGuard`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RemediationKind {
    /// Sin operación (`discrete(0)`).
    NoOp,
    /// Reinicio sintético del worker (`discrete(1)`).
    RestartWorker,
    /// Drenaje sintético de cola (`discrete(2)`).
    DrainQueue,
    /// Descarte controlado de carga (`discrete(3)`).
    ShedLoad,
    /// Acción personalizada declarada por un tercero mediante Spec.
    Custom {
        /// Nombre lógico de la acción.
        name: String,
        /// Índice discreto dentro de [`gogamen_core::ActionSpace`].
        action_id: usize,
    },
}

impl RemediationKind {
    /// Nombre lógico de la acción de remediación.
    pub fn action_name(&self) -> &str {
        match self {
            Self::NoOp => "no_op",
            Self::RestartWorker => "restart_worker",
            Self::DrainQueue => "drain_queue",
            Self::ShedLoad => "shed_load",
            Self::Custom { name, .. } => name.as_str(),
        }
    }

    /// Índice discreto asociado para evaluación en [`gogamen_core::ActionGuard`].
    pub const fn action_index(&self) -> usize {
        match self {
            Self::NoOp => 0,
            Self::RestartWorker => 1,
            Self::DrainQueue => 2,
            Self::ShedLoad => 3,
            Self::Custom { action_id, .. } => *action_id,
        }
    }

    /// Convierte la remediación en [`DiscreteAction`] acotada por `action_space_size`.
    pub fn discrete_action(
        &self,
        action_space_size: usize,
    ) -> Result<DiscreteAction, AutomationError> {
        Ok(DiscreteAction::new(self.action_index(), action_space_size)?)
    }

    /// Resuelve una remediación desde su nombre textual en `SystemSpec`.
    pub fn from_name_and_id(name: &str, action_id: Option<usize>) -> Result<Self, AutomationError> {
        let trimmed = name.trim();
        if trimmed.is_empty() {
            return Err(AutomationError::InvalidConfig(
                "remediation_action must not be empty".to_owned(),
            ));
        }
        Ok(match trimmed {
            "no_op" => Self::NoOp,
            "restart_worker" => Self::RestartWorker,
            "drain_queue" => Self::DrainQueue,
            "shed_load" => Self::ShedLoad,
            other => Self::Custom {
                name: other.to_owned(),
                action_id: action_id.unwrap_or(4),
            },
        })
    }
}

/// Configuración completa del supervisor de servicio local.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceSupervisorConfig {
    /// Nombre lógico del servicio supervisado.
    pub service_name: String,
    /// Umbrales configurables de eventos.
    pub thresholds: ServiceThresholds,
    /// Acción de remediación por defecto ante diagnóstico.
    pub remediation_action: RemediationKind,
    /// Número máximo de reintentos antes de tomar la rama `Error`.
    pub max_retries: u32,
    /// Tamaño del [`gogamen_core::ActionSpace`] discreto del actuador fake.
    pub action_space_size: usize,
}

impl Default for ServiceSupervisorConfig {
    fn default() -> Self {
        Self {
            service_name: "local-auth-service".to_owned(),
            thresholds: ServiceThresholds::default(),
            remediation_action: RemediationKind::RestartWorker,
            max_retries: 2,
            action_space_size: 16,
        }
    }
}

impl ServiceSupervisorConfig {
    /// Valida los invariantes de configuración.
    pub fn validate(&self) -> Result<(), AutomationError> {
        if self.service_name.trim().is_empty() {
            return Err(AutomationError::InvalidConfig(
                "service_name must not be empty".to_owned(),
            ));
        }
        self.thresholds.validate()?;
        if self.action_space_size == 0
            || self.remediation_action.action_index() >= self.action_space_size
        {
            return Err(AutomationError::InvalidConfig(format!(
                "remediation action index {} out of bounds for action_space_size {}",
                self.remediation_action.action_index(),
                self.action_space_size
            )));
        }
        Ok(())
    }

    /// Exporta la configuración como [`SystemSpec`] portable y compilable.
    pub fn to_system_spec(&self) -> SystemSpec {
        let mut supervisor_node =
            SystemNodeSpec::new("supervisor", "automation.supervisor::monitor")
                .with_output("remediation", PortType::I64)
                .with_config(
                    "service_name",
                    SpecConfigValue::String(self.service_name.clone()),
                )
                .with_config(
                    "max_failed_checks",
                    SpecConfigValue::I64(i64::from(self.thresholds.max_failed_checks)),
                )
                .with_config(
                    "latency_threshold_ms",
                    SpecConfigValue::I64(self.thresholds.latency_threshold_ms as i64),
                )
                .with_config(
                    "queue_backlog_threshold",
                    SpecConfigValue::I64(self.thresholds.queue_backlog_threshold as i64),
                )
                .with_config(
                    "required_probe_passes",
                    SpecConfigValue::I64(i64::from(self.thresholds.required_probe_passes)),
                )
                .with_config(
                    "required_recovery_ticks",
                    SpecConfigValue::I64(i64::from(self.thresholds.required_recovery_ticks)),
                )
                .with_config(
                    "max_retries",
                    SpecConfigValue::I64(i64::from(self.max_retries)),
                );

        if let Some(amb) = self.thresholds.latency_ambiguity_ms {
            supervisor_node = supervisor_node
                .with_config("latency_ambiguity_ms", SpecConfigValue::I64(amb as i64));
        }

        let actuator_node = SystemNodeSpec::new("actuator", "automation.actuator::remediate")
            .with_input("remediation", PortType::I64)
            .with_config(
                "remediation_action",
                SpecConfigValue::String(self.remediation_action.action_name().to_owned()),
            )
            .with_config(
                "remediation_action_id",
                SpecConfigValue::I64(self.remediation_action.action_index() as i64),
            )
            .with_config(
                "action_space_size",
                SpecConfigValue::I64(self.action_space_size as i64),
            );

        SystemSpec::new(format!("service-supervisor-{}", self.service_name))
            .with_node(supervisor_node)
            .with_node(actuator_node)
            .with_connection(SystemConnectionSpec::instant(
                "supervisor",
                "remediation",
                "actuator",
                "remediation",
            ))
    }

    /// Construye una [`ServiceSupervisorConfig`] desde una [`SystemSpec`] declarativa.
    ///
    /// Busca parámetros tanto en `spec.config` como en los nodos (`supervisor` / `actuator`),
    /// aplicando valores por defecto cuando algún parámetro opcional no se especifica.
    pub fn from_system_spec(spec: &SystemSpec) -> Result<Self, AutomationError> {
        spec.validate()?;

        let mut config = Self::default();
        if let Some(service_name) = find_string(spec, "service_name") {
            config.service_name = service_name.to_owned();
        }
        if let Some(v) = find_positive_u32(spec, "max_failed_checks")? {
            config.thresholds.max_failed_checks = v;
        }
        if let Some(v) = find_positive_u64(spec, "latency_threshold_ms")? {
            config.thresholds.latency_threshold_ms = v;
        }
        if let Some(v) = find_optional_u64(spec, "latency_ambiguity_ms")? {
            config.thresholds.latency_ambiguity_ms = if v == 0 { None } else { Some(v) };
        }
        if let Some(v) = find_positive_u64(spec, "queue_backlog_threshold")? {
            config.thresholds.queue_backlog_threshold = v;
        }
        if let Some(v) = find_positive_u32(spec, "required_probe_passes")? {
            config.thresholds.required_probe_passes = v;
        }
        if let Some(v) = find_positive_u32(spec, "required_recovery_ticks")? {
            config.thresholds.required_recovery_ticks = v;
        }
        if let Some(v) = find_non_negative_u32(spec, "max_retries")? {
            config.max_retries = v;
        }
        if let Some(v) = find_positive_u64(spec, "action_space_size")? {
            config.action_space_size = v as usize;
        }

        let action_id = find_non_negative_u32(spec, "remediation_action_id")?.map(|v| v as usize);
        if let Some(action_name) = find_string(spec, "remediation_action") {
            config.remediation_action = RemediationKind::from_name_and_id(action_name, action_id)?;
        }

        config.validate()?;
        Ok(config)
    }

    /// Parsea un JSON de [`SystemSpec`] y construye la configuración del supervisor.
    pub fn from_spec_json(json: &str) -> Result<Self, AutomationError> {
        let spec = SystemSpec::from_json(json)?;
        Self::from_system_spec(&spec)
    }
}

/// Catálogo de capacidades estándar para compilar specs de supervisión local.
pub fn automation_capability_catalog() -> Result<CapabilityCatalog, AutomationError> {
    let mut catalog = CapabilityCatalog::new();

    let supervisor_mod = ModuleManifest::new(
        "automation.supervisor",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("monitor", CapabilityVersion::new(1, 0, 0))
            .with_output("remediation", PortType::I64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    let actuator_mod = ModuleManifest::new(
        "automation.actuator",
        ModuleOrigin::Builtin,
        CapabilityVersion::new(1, 0, 0),
    )
    .with_operation(
        OperationSpec::new("remediate", CapabilityVersion::new(1, 0, 0))
            .with_input("remediation", PortType::I64)
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    );

    catalog.register_module(supervisor_mod)?;
    catalog.register_module(actuator_mod)?;
    Ok(catalog)
}

/// Valida y compila una [`SystemSpec`] de automatización en [`ServiceSupervisorConfig`] y [`ExecutablePlan`].
pub fn compile_supervisor_spec(
    spec: &SystemSpec,
) -> Result<(ServiceSupervisorConfig, ExecutablePlan), AutomationError> {
    let catalog = automation_capability_catalog()?;
    let plan = SpecCompiler::compile(spec, &catalog, TemporalProfile::BoundedTick)?;
    let config = ServiceSupervisorConfig::from_system_spec(spec)?;
    Ok((config, plan))
}

fn find_value<'a>(spec: &'a SystemSpec, key: &str) -> Option<&'a SpecConfigValue> {
    if let Some(val) = spec.config.get(key) {
        return Some(val);
    }
    for node in &spec.nodes {
        if let Some(val) = node.config.get(key) {
            return Some(val);
        }
    }
    None
}

fn find_string<'a>(spec: &'a SystemSpec, key: &str) -> Option<&'a str> {
    match find_value(spec, key) {
        Some(SpecConfigValue::String(s)) => Some(s.as_str()),
        _ => None,
    }
}

fn find_non_negative_u32(spec: &SystemSpec, key: &str) -> Result<Option<u32>, AutomationError> {
    match find_value(spec, key) {
        None => Ok(None),
        Some(SpecConfigValue::I64(v)) if *v >= 0 && *v <= i64::from(u32::MAX) => {
            Ok(Some(*v as u32))
        }
        Some(_) => Err(AutomationError::InvalidConfig(format!(
            "config '{key}' must be a non-negative integer within u32"
        ))),
    }
}

fn find_positive_u32(spec: &SystemSpec, key: &str) -> Result<Option<u32>, AutomationError> {
    match find_non_negative_u32(spec, key)? {
        Some(0) => Err(AutomationError::InvalidConfig(format!(
            "config '{key}' must be >= 1"
        ))),
        other => Ok(other),
    }
}

fn find_optional_u64(spec: &SystemSpec, key: &str) -> Result<Option<u64>, AutomationError> {
    match find_value(spec, key) {
        None => Ok(None),
        Some(SpecConfigValue::I64(v)) if *v >= 0 => Ok(Some(*v as u64)),
        Some(_) => Err(AutomationError::InvalidConfig(format!(
            "config '{key}' must be a non-negative integer"
        ))),
    }
}

fn find_positive_u64(spec: &SystemSpec, key: &str) -> Result<Option<u64>, AutomationError> {
    match find_optional_u64(spec, key)? {
        Some(0) => Err(AutomationError::InvalidConfig(format!(
            "config '{key}' must be >= 1"
        ))),
        other => Ok(other),
    }
}
