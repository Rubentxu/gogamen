//! Laboratorio de aplicación convencional: supervisor de servicio local sintético (`std`-only).
//!
//! Demuestra la composición de [`gogamen_core::StateMachineWorkflow`],
//! [`gogamen_core::ActionGuard`], [`gogamen_core::SystemOneProvider`] opcional y
//! extensibilidad declarativa mediante [`gogamen_spec::SystemSpec`] y
//! [`gogamen_compiler::SpecCompiler`] sin red ni procesos reales del sistema operativo.

pub mod events;
pub mod spec;
pub mod supervisor;

pub use events::{EventClassification, ServiceThresholds, SyntheticServiceEvent};
pub use spec::{
    AutomationError, RemediationKind, ServiceSupervisorConfig, automation_capability_catalog,
    compile_supervisor_spec,
};
pub use supervisor::{
    COGNITIVE_DECISION_ESCALATE, ExecutedFakeAction, FakeServiceActuator, QUERY_INCIDENT_CLASSIFY,
    ReplayReport, STATE_ALERT, STATE_DIAGNOSE, STATE_ERROR, STATE_MONITORING, STATE_REMEDIATE,
    STATE_RESOLVED, STATE_TIMEOUT, ServiceSupervisor, SupervisorStepReport, SupervisorTraceRecord,
    TraceRecordKind,
};
