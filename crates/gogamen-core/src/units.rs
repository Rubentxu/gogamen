//! Tipos temporales y de conteo con unidades explícitas.
//!
//! Todos los tipos de este módulo son valores inmutables (`Copy`) que validan sus
//! invariantes en construcción:
//! - [`Tick`]: conteo discreto adimensional de pasos de simulación (`u64`).
//! - [`StepDt`]: paso temporal físico en **segundos** (`f64`), estrictamente finito y positivo (`> 0.0`).
//! - [`SimDuration`] (alias [`Duration`]): tiempo acumulado de simulación en **segundos** (`f64`), finito y no negativo (`>= 0.0`).

use std::fmt;
use std::time::Duration as StdDuration;

use crate::error::DomainError;

const NANOS_PER_SEC: f64 = 1_000_000_000.0;
const MILLIS_PER_SEC: f64 = 1_000.0;

/// Contador discreto adimensional de pasos de simulación.
///
/// Tipo de valor inmutable: avanzar un tick devuelve un nuevo [`Tick`] sin mutar el original.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Tick(u64);

impl Tick {
    /// Tick inicial de un episodio (`0`).
    pub const ZERO: Self = Self(0);

    /// Crea un [`Tick`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente del tick.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Devuelve el siguiente tick (`self + 1`), fallando si hay desbordamiento de `u64`.
    pub fn next(self) -> Result<Self, DomainError> {
        self.0
            .checked_add(1)
            .map(Self)
            .ok_or(DomainError::TimeOverflow {
                context: "Tick::next",
            })
    }

    /// Suma `delta` ticks de forma verificada.
    pub fn checked_add(self, delta: u64) -> Result<Self, DomainError> {
        self.0
            .checked_add(delta)
            .map(Self)
            .ok_or(DomainError::TimeOverflow {
                context: "Tick::checked_add",
            })
    }
}

impl fmt::Display for Tick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tick({})", self.0)
    }
}

/// Paso temporal de simulación (`dt`) expresado explícitamente en **segundos** (`f64`).
///
/// Invariante: el valor almacenado siempre es finito y estrictamente positivo (`dt > 0.0`).
/// Tipo de valor inmutable.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct StepDt(f64);

impl StepDt {
    /// Construye un [`StepDt`] a partir de segundos (`f64`).
    ///
    /// Rechaza `NaN`, `+inf`, `-inf`, `0.0`, `-0.0` y cualquier valor `<= 0.0`.
    pub fn from_secs_f64(secs: f64) -> Result<Self, DomainError> {
        if secs.is_finite() && secs > 0.0 {
            Ok(Self(secs))
        } else {
            Err(DomainError::InvalidStepDt { value: secs })
        }
    }

    /// Construye un [`StepDt`] a partir de milisegundos (`f64`), convirtiéndolo a segundos.
    ///
    /// Rechaza `NaN`, infinitos y valores `<= 0.0` (o que al dividir resulten `<= 0.0`).
    pub fn from_millis_f64(millis: f64) -> Result<Self, DomainError> {
        if !millis.is_finite() || millis <= 0.0 {
            return Err(DomainError::InvalidStepDt {
                value: millis / MILLIS_PER_SEC,
            });
        }
        Self::from_secs_f64(millis / MILLIS_PER_SEC)
    }

    /// Construye un [`StepDt`] a partir de nanosegundos enteros (`> 0`).
    pub fn from_nanos(nanos: u64) -> Result<Self, DomainError> {
        if nanos == 0 {
            return Err(DomainError::InvalidStepDt { value: 0.0 });
        }
        Self::from_secs_f64((nanos as f64) / NANOS_PER_SEC)
    }

    /// Devuelve el paso temporal en segundos.
    pub const fn as_secs_f64(self) -> f64 {
        self.0
    }

    /// Devuelve el paso temporal aproximado en nanosegundos (`u128`).
    pub fn as_nanos(self) -> u128 {
        (self.0 * NANOS_PER_SEC).round() as u128
    }

    /// Convierte el paso temporal a [`std::time::Duration`], si es representable.
    pub fn try_into_std(self) -> Result<StdDuration, DomainError> {
        StdDuration::try_from_secs_f64(self.0).map_err(|_| DomainError::TimeOverflow {
            context: "StepDt::try_into_std",
        })
    }
}

impl fmt::Display for StepDt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} s", self.0)
    }
}

/// Duración acumulada de simulación expresada explícitamente en **segundos** (`f64`).
///
/// Invariante: el valor almacenado siempre es finito y no negativo (`>= 0.0`).
/// Tipo de valor inmutable.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default)]
pub struct SimDuration(f64);

/// Alias conveniente para [`SimDuration`].
pub type Duration = SimDuration;

impl SimDuration {
    /// Duración nula (`0.0` segundos).
    pub const ZERO: Self = Self(0.0);

    /// Construye una [`SimDuration`] a partir de segundos (`f64`).
    ///
    /// Rechaza `NaN`, infinitos y valores negativos (`< 0.0`). Normaliza `-0.0` a `0.0`.
    pub fn from_secs_f64(secs: f64) -> Result<Self, DomainError> {
        if secs.is_finite() && secs >= 0.0 {
            let normalized = if secs == 0.0 { 0.0 } else { secs };
            Ok(Self(normalized))
        } else {
            Err(DomainError::InvalidDuration { value: secs })
        }
    }

    /// Construye una [`SimDuration`] a partir de nanosegundos enteros.
    pub fn from_nanos(nanos: u64) -> Self {
        Self((nanos as f64) / NANOS_PER_SEC)
    }

    /// Devuelve la duración en segundos.
    pub const fn as_secs_f64(self) -> f64 {
        self.0
    }

    /// Devuelve la duración aproximada en nanosegundos (`u128`).
    pub fn as_nanos(self) -> u128 {
        (self.0 * NANOS_PER_SEC).round() as u128
    }

    /// Suma un [`StepDt`] devolviendo una nueva [`SimDuration`] sin mutar `self`.
    pub fn checked_add_dt(self, dt: StepDt) -> Result<Self, DomainError> {
        let sum = self.0 + dt.as_secs_f64();
        if sum.is_finite() {
            Ok(Self(sum))
        } else {
            Err(DomainError::TimeOverflow {
                context: "SimDuration::checked_add_dt",
            })
        }
    }

    /// Convierte la duración a [`std::time::Duration`], si es representable.
    pub fn try_into_std(self) -> Result<StdDuration, DomainError> {
        StdDuration::try_from_secs_f64(self.0).map_err(|_| DomainError::TimeOverflow {
            context: "SimDuration::try_into_std",
        })
    }
}

impl fmt::Display for SimDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} s", self.0)
    }
}
