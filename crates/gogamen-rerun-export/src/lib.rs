//! Exportador Rerun **opcional** (GOG-108): proyección de una sola dirección desde las trazas
//! existentes de Gogamen hacia un recording Rerun (`.rrd`).
//!
//! # Principios
//! - **Una sola fuente**: el exportador lee [`gogamen_studio::replay::Recording`] (GOG-94) o
//!   [`gogamen_runtime::ScientificTraceArchive`] (GOG-93) por referencia inmutable. No define
//!   un formato fuente nuevo ni persiste una copia del registro científico: el `.rrd` es un
//!   artefacto derivado y descartable.
//! - **Núcleo sin Rerun**: la proyección pura ([`plan`]) compila sin dependencias de Rerun; el
//!   emisor (`rerun_sink`) sólo existe con la feature `rerun`.
//! - **Huecos y redacciones se conservan** (ver [`plan::Payload`]); mensajes y secretos se
//!   deniegan por defecto ([`plan::ExportPermissions`]).

pub mod fixtures;
pub mod plan;

#[cfg(feature = "rerun")]
pub mod rerun_sink;

pub use plan::{
    ExportOptions, ExportPermissions, ExportPlan, Item, ItemTime, Payload, plan_from_archive,
    plan_from_recording,
};
