//! Superficie pública del runtime headless y ejecución de planes compilados.
//!
//! # Estabilidad SemVer pre-1.0
//! Este módulo pertenece a `gogamen-sdk` (`0.1.0-alpha`, [`crate::SDK_SEMVER`]).
//! Sus tipos y métodos están sujetos a evolución en `0.x` hasta la estabilización `1.0.0`
//! ([`crate::API_STABILITY_POLICY`]).
//!
//! # Unidades
//! - **`dt` ([`gogamen_core::StepDt`]):** paso temporal físico en **segundos** (`f64 > 0.0`).
//! - **`planning_timeout` (`std::time::Duration`):** tiempo monotónico máximo por tick.
//! - **`tick` ([`gogamen_core::Tick`]):** contador discreto adimensional (`u64`).
//!
//! # Errores
//! - [`RuntimeError`]: errores del ciclo `observe -> plan -> guard -> act`
//!   (entorno vivo, modelo generativo, reloj o planificación).
//! - [`PlanRuntimeError`]: errores al instanciar o ejecutar un [`gogamen_compiler::ExecutablePlan`].

pub use gogamen_runtime::{
    ExecutionTrace, ExternalKernel, ExternalNodeContext, HeadlessRuntime, KernelRegistry,
    KernelRegistryError, PlanInstance, PlanRunSummary, PlanRuntimeEngine, PlanRuntimeError,
    PortValue, ProposalMutator, RejectionCause, Run1DSummary, RuntimeConfig, RuntimeError,
    TickTraceEntry, builtin_plan_catalog, run_1d_episode,
};
