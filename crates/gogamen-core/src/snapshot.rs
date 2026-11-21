//! Contratos de snapshots ramificables (`snapshot` / `fork` / `restore`) e integridad (`std`-only).
//!
//! Permite capturar el estado completo de una simulación ([`State`], estado de
//! [`DeterministicRng`], [`Terminal`] y reloj [`Tick`]/[`SimDuration`]) con verificación
//! de integridad por checksum y versión de esquema, evitando contaminación entre ramas
//! (`fork`) y detectando snapshots incompletos o corruptos.

use std::fmt;
use std::mem::{size_of, size_of_val};
use std::time::Instant;

use crate::determinism::TraceHasher;
use crate::domain::{State, Terminal};
use crate::error::SnapshotError;
use crate::model::DeterministicRng;
use crate::units::{SimDuration, Tick};

/// Versión de esquema actual para [`SimulationSnapshot`].
pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;

/// Snapshot completo e íntegro de un estado de simulación ramificable.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulationSnapshot {
    version: u32,
    state: State,
    rng_state: u64,
    terminal: Terminal,
    tick: Tick,
    elapsed: SimDuration,
    checksum: u64,
}

/// Alias semántico para snapshots de mundo.
pub type WorldSnapshot = SimulationSnapshot;

impl SimulationSnapshot {
    /// Crea un nuevo [`SimulationSnapshot`] validado a partir de `state`, `rng` y `terminal`.
    ///
    /// El `tick` y `elapsed` del snapshot se toman directamente de `state`.
    pub fn new(state: State, rng: &DeterministicRng, terminal: Terminal) -> Self {
        let tick = state.tick();
        let elapsed = state.elapsed();
        let mut snap = Self {
            version: SNAPSHOT_SCHEMA_VERSION,
            state,
            rng_state: rng.state(),
            terminal,
            tick,
            elapsed,
            checksum: 0,
        };
        snap.checksum = snap.compute_checksum();
        snap
    }

    /// Construye un [`SimulationSnapshot`] a partir de sus componentes explícitos,
    /// verificando la coherencia entre `state.tick()` y `tick`.
    pub fn from_parts(
        state: State,
        rng_state: u64,
        terminal: Terminal,
        tick: Tick,
        elapsed: SimDuration,
    ) -> Result<Self, SnapshotError> {
        if state.tick() != tick {
            return Err(SnapshotError::InconsistentClock {
                state_tick: state.tick().as_u64(),
                snapshot_tick: tick.as_u64(),
            });
        }
        let mut snap = Self {
            version: SNAPSHOT_SCHEMA_VERSION,
            state,
            rng_state,
            terminal,
            tick,
            elapsed,
            checksum: 0,
        };
        snap.checksum = snap.compute_checksum();
        Ok(snap)
    }

    /// Devuelve la versión de esquema del snapshot.
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Devuelve el [`State`] capturado.
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// Devuelve el estado interno crudo (`u64`) del [`DeterministicRng`].
    pub const fn rng_state(&self) -> u64 {
        self.rng_state
    }

    /// Reconstruye un [`DeterministicRng`] posicionado en el estado capturado.
    pub const fn rng(&self) -> DeterministicRng {
        DeterministicRng::new(self.rng_state)
    }

    /// Devuelve el estado [`Terminal`] capturado.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Devuelve el [`Tick`] capturado.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve la [`SimDuration`] acumulada capturada.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Devuelve el checksum de integridad almacenado en el snapshot.
    pub const fn checksum(&self) -> u64 {
        self.checksum
    }

    /// Recalcula el checksum FNV-1a canónico sobre los campos del snapshot.
    pub fn compute_checksum(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_u64(u64::from(self.version));
        hasher.write_state(&self.state);
        hasher.write_u64(self.rng_state);
        hasher.write_terminal(self.terminal);
        hasher.write_tick(self.tick);
        hasher.write_duration(self.elapsed);
        hasher.finish()
    }

    /// Verifica versión de esquema, coherencia de reloj y checksum de integridad.
    pub fn verify(&self) -> Result<(), SnapshotError> {
        if self.version != SNAPSHOT_SCHEMA_VERSION {
            return Err(SnapshotError::UnsupportedVersion {
                expected: SNAPSHOT_SCHEMA_VERSION,
                actual: self.version,
            });
        }
        if self.state.tick() != self.tick {
            return Err(SnapshotError::InconsistentClock {
                state_tick: self.state.tick().as_u64(),
                snapshot_tick: self.tick.as_u64(),
            });
        }
        let actual = self.compute_checksum();
        if self.checksum != actual {
            return Err(SnapshotError::CorruptedChecksum {
                expected: self.checksum,
                actual,
            });
        }
        Ok(())
    }

    /// Estima el tamaño en bytes del snapshot (estructura base más buffer de valores).
    pub fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + size_of_val(self.state.values())
    }

    /// Convierte este snapshot en un [`SnapshotEnvelope`] editable o serializable.
    pub fn to_envelope(&self) -> SnapshotEnvelope {
        SnapshotEnvelope {
            version: Some(self.version),
            state: Some(self.state.clone()),
            rng_state: Some(self.rng_state),
            terminal: Some(self.terminal),
            tick: Some(self.tick),
            elapsed: Some(self.elapsed),
            checksum: Some(self.checksum),
        }
    }
}

/// Representación sin validar de un snapshot con campos opcionales.
///
/// Útil para deserialización, transporte o pruebas de detección de snapshots
/// incompletos o corruptos antes de convertir a [`SimulationSnapshot`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SnapshotEnvelope {
    /// Versión de esquema del snapshot.
    pub version: Option<u32>,
    /// Estado de simulación capturado.
    pub state: Option<State>,
    /// Estado interno del RNG determinista.
    pub rng_state: Option<u64>,
    /// Estado terminal del episodio.
    pub terminal: Option<Terminal>,
    /// Tick lógico del snapshot.
    pub tick: Option<Tick>,
    /// Tiempo acumulado de simulación.
    pub elapsed: Option<SimDuration>,
    /// Checksum de integridad del snapshot.
    pub checksum: Option<u64>,
}

impl SnapshotEnvelope {
    /// Valida que todos los campos obligatorios estén presentes y que el snapshot sea íntegro.
    pub fn into_snapshot(self) -> Result<SimulationSnapshot, SnapshotError> {
        let version = self.version.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "version",
        })?;
        let state = self.state.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "state",
        })?;
        let rng_state = self.rng_state.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "rng_state",
        })?;
        let terminal = self.terminal.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "terminal",
        })?;
        let tick = self.tick.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "tick",
        })?;
        let elapsed = self.elapsed.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "elapsed",
        })?;
        let checksum = self.checksum.ok_or(SnapshotError::IncompleteSnapshot {
            missing_field: "checksum",
        })?;

        let snapshot = SimulationSnapshot {
            version,
            state,
            rng_state,
            terminal,
            tick,
            elapsed,
            checksum,
        };
        snapshot.verify()?;
        Ok(snapshot)
    }
}

/// Métrica ligera de coste de memoria y latencia de ramificación de snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotMetrics {
    bytes_per_snapshot: usize,
    fork_latency_ns: u128,
}

impl SnapshotMetrics {
    /// Construye una métrica de snapshot con tamaño en bytes y latencia en nanosegundos.
    pub const fn new(bytes_per_snapshot: usize, fork_latency_ns: u128) -> Self {
        Self {
            bytes_per_snapshot,
            fork_latency_ns,
        }
    }

    /// Devuelve el tamaño estimado en bytes por snapshot.
    pub const fn bytes_per_snapshot(&self) -> usize {
        self.bytes_per_snapshot
    }

    /// Devuelve la latencia medida del `fork` en nanosegundos.
    pub const fn fork_latency_ns(&self) -> u128 {
        self.fork_latency_ns
    }
}

impl fmt::Display for SnapshotMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "bytes_per_snapshot={}, fork_latency_ns={}",
            self.bytes_per_snapshot, self.fork_latency_ns
        )
    }
}

/// Contrato de simulación ramificable con captura, bifurcación y restauración exacta.
///
/// Las implementaciones deben garantizar que:
/// - [`BranchableSnapshot::fork`] produce una copia aislada cuyo avance no contamina a `self`.
/// - [`BranchableSnapshot::restore`] valida la integridad del snapshot antes de mutar `self`,
///   reproduciendo exactamente la siguiente transición determinista.
pub trait BranchableSnapshot: Sized {
    /// Captura un [`SimulationSnapshot`] completo e íntegro del estado actual.
    fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError>;

    /// Crea una rama independiente (`fork`) a partir del estado actual.
    fn fork(&self) -> Result<Self, SnapshotError>;

    /// Restaura el estado interno de `self` al representado por `snapshot`.
    fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError>;

    /// Ejecuta [`BranchableSnapshot::fork`] midiendo [`SnapshotMetrics`] (`bytes_per_snapshot` y `fork_latency_ns`).
    fn fork_with_metrics(&self) -> Result<(Self, SnapshotMetrics), SnapshotError> {
        let snap = self.snapshot()?;
        let bytes_per_snapshot = snap.estimated_bytes();
        let start = Instant::now();
        let forked = self.fork()?;
        let fork_latency_ns = start.elapsed().as_nanos();
        Ok((
            forked,
            SnapshotMetrics::new(bytes_per_snapshot, fork_latency_ns),
        ))
    }

    /// Valida un [`SnapshotEnvelope`] y restaura su contenido en `self`.
    fn restore_from_envelope(&mut self, envelope: SnapshotEnvelope) -> Result<(), SnapshotError> {
        let snapshot = envelope.into_snapshot()?;
        self.restore(&snapshot)
    }
}

/// Alias de trait para capacidades ramificables mediante snapshots.
pub trait Snapshotable: BranchableSnapshot {}

impl<T: BranchableSnapshot> Snapshotable for T {}
