//! Contratos centrales de estado, observación, terminación y transición de mundo.
//!
//! Todos los tipos definidos aquí son **inmutables** (value types) y validan sus
//! invariantes en construcción:
//! - Ningún vector escalar de [`State`] u [`Observation`] admite `NaN` o infinitos.
//! - El tiempo acumulado usa [`SimDuration`] (segundos finitos `>= 0.0`) y los pasos usan [`StepDt`] (segundos finitos `> 0.0`).
//! - [`Terminal`] distingue explícitamente entre ejecución activa ([`Terminal::Ongoing`]) y
//!   finalización definitiva ([`Terminal::Terminated`]).

use crate::error::{DomainError, ensure_all_finite};
use crate::units::{SimDuration, StepDt, Tick};

/// Instantánea inmutable del estado de un mundo en un [`Tick`] y [`SimDuration`] concretos.
///
/// # Mutabilidad
/// `State` es completamente inmutable tras su creación. Para evolucionar un estado en el
/// tiempo sin mutar el original, utilice [`State::advance`], que produce una nueva instancia.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    tick: Tick,
    elapsed: SimDuration,
    values: Vec<f64>,
}

impl State {
    /// Construye un [`State`] validando que todos sus valores escalares sean finitos.
    pub fn new(tick: Tick, elapsed: SimDuration, values: Vec<f64>) -> Result<Self, DomainError> {
        ensure_all_finite("state.values", &values)?;
        Ok(Self {
            tick,
            elapsed,
            values,
        })
    }

    /// Devuelve el tick discreto en el que se capturó este estado.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve el tiempo de simulación acumulado en **segundos** (`>= 0.0`).
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Devuelve la vista inmutable de los valores escalares del estado.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Devuelve la dimensionalidad del vector de estado.
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// Genera un nuevo [`State`] avanzando un [`StepDt`] desde `self` sin mutar `self`.
    pub fn advance(&self, dt: StepDt, next_values: Vec<f64>) -> Result<Self, DomainError> {
        let next_tick = self.tick.next()?;
        let next_elapsed = self.elapsed.checked_add_dt(dt)?;
        Self::new(next_tick, next_elapsed, next_values)
    }
}

/// Observación inmutable emitida por un mundo en un [`Tick`] y [`SimDuration`] concretos.
///
/// # Mutabilidad
/// `Observation` es inmutable; cada paso del mundo produce una nueva observación.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    tick: Tick,
    elapsed: SimDuration,
    features: Vec<f64>,
}

impl Observation {
    /// Construye una [`Observation`] validando que todas sus características sean finitas.
    pub fn new(tick: Tick, elapsed: SimDuration, features: Vec<f64>) -> Result<Self, DomainError> {
        ensure_all_finite("observation.features", &features)?;
        Ok(Self {
            tick,
            elapsed,
            features,
        })
    }

    /// Devuelve el tick discreto asociado a la observación.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve el tiempo de simulación acumulado en **segundos** (`>= 0.0`).
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Devuelve la vista inmutable del vector de características observadas.
    pub fn features(&self) -> &[f64] {
        &self.features
    }

    /// Devuelve la dimensionalidad del vector de características.
    pub fn dimension(&self) -> usize {
        self.features.len()
    }
}

/// Motivo explícito por el cual un episodio o simulación ha alcanzado un estado terminal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TerminalReason {
    /// El objetivo del mundo o tarea se completó satisfactoriamente.
    GoalReached,
    /// Se produjo una condición de fallo terminal en la dinámica del mundo.
    Failure,
    /// Se alcanzó el límite máximo de ticks o de [`SimDuration`] configurado.
    HorizonReached,
    /// La ejecución fue cancelada externamente.
    Cancelled,
    /// Se detectó una violación de invariante físico o de dominio.
    InvariantViolated,
}

/// Estado de terminación de un paso o episodio.
///
/// # Semántica de terminal
/// - [`Terminal::Ongoing`]: el mundo sigue activo y puede recibir más acciones mediante pasos sucesivos.
/// - [`Terminal::Terminated`]: el episodio ha finalizado de forma definitiva; no deben ejecutarse
///   nuevos pasos sobre el mismo episodio hasta invocar una operación de reinicio (`reset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Terminal {
    /// El episodio continúa activo.
    #[default]
    Ongoing,
    /// El episodio ha terminado por un motivo explícito.
    Terminated(TerminalReason),
}

impl Terminal {
    /// Devuelve `true` si el estado es terminal (`Terminated(_)`).
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Terminated(_))
    }

    /// Devuelve `true` si el episodio continúa activo (`Ongoing`).
    pub const fn is_ongoing(self) -> bool {
        matches!(self, Self::Ongoing)
    }

    /// Devuelve el motivo de terminación si el estado es terminal.
    pub const fn reason(self) -> Option<TerminalReason> {
        match self {
            Self::Ongoing => None,
            Self::Terminated(reason) => Some(reason),
        }
    }
}

/// Resultado inmutable de aplicar una acción durante un paso [`StepDt`] sobre un mundo.
///
/// # Invariantes
/// - `next_state.tick() == prev_tick + 1`
/// - `observation.tick() == next_state.tick()`
/// - `dt` es estrictamente finito y positivo en segundos (`> 0.0`).
#[derive(Debug, Clone, PartialEq)]
pub struct Transition {
    prev_tick: Tick,
    dt: StepDt,
    next_state: State,
    observation: Observation,
    terminal: Terminal,
}

impl Transition {
    /// Construye una [`Transition`] verificando coherencia temporal entre `prev_tick`, `next_state` y `observation`.
    pub fn new(
        prev_tick: Tick,
        dt: StepDt,
        next_state: State,
        observation: Observation,
        terminal: Terminal,
    ) -> Result<Self, DomainError> {
        let expected_tick = prev_tick.next()?;
        if next_state.tick() != expected_tick {
            return Err(DomainError::InvalidTransition {
                reason: "next_state.tick must equal prev_tick.next()",
            });
        }
        if observation.tick() != next_state.tick() {
            return Err(DomainError::InvalidTransition {
                reason: "observation.tick must equal next_state.tick()",
            });
        }
        Ok(Self {
            prev_tick,
            dt,
            next_state,
            observation,
            terminal,
        })
    }

    /// Devuelve el tick previo a la transición.
    pub const fn prev_tick(&self) -> Tick {
        self.prev_tick
    }

    /// Devuelve el paso temporal `dt` en segundos aplicado en esta transición.
    pub const fn dt(&self) -> StepDt {
        self.dt
    }

    /// Devuelve el nuevo estado resultante tras el paso.
    pub const fn next_state(&self) -> &State {
        &self.next_state
    }

    /// Consume la transición y devuelve el nuevo estado sin clonarlo.
    pub fn into_next_state(self) -> State {
        self.next_state
    }

    /// Devuelve la observación correspondiente al nuevo estado.
    pub const fn observation(&self) -> &Observation {
        &self.observation
    }

    /// Devuelve la condición de terminación tras ejecutar este paso.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }
}
