//! Contratos de acciones y espacios de acción con validación estricta de rango y finitud.
//!
//! Todas las estructuras de este módulo son inmutables tras su construcción y garantizan
//! invariantes de dominio:
//! - Ningún límite ni componente continuo puede ser `NaN` o infinito.
//! - Ninguna acción discreta validada excede `[0, cardinality)`.
//! - Ninguna acción continua validada contra un [`ActionSpace`] excede sus [`ActionBounds`].

use crate::error::{DomainError, ensure_all_finite, ensure_finite};

/// Límites inclusivos `[min, max]` para una dimensión continua de acción.
///
/// Las unidades dependen del mundo concreto (por ejemplo metros/segundo o radianes),
/// pero los valores deben ser siempre finitos y cumplir `min <= max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionBounds {
    min: f64,
    max: f64,
}

impl ActionBounds {
    /// Construye límites inclusivos `[min, max]` validando finitud y orden.
    pub fn new(min: f64, max: f64) -> Result<Self, DomainError> {
        ensure_finite("action_bounds.min", min)?;
        ensure_finite("action_bounds.max", max)?;
        if min > max {
            return Err(DomainError::InvalidActionBounds { min, max });
        }
        Ok(Self { min, max })
    }

    /// Devuelve el límite inferior inclusivo.
    pub const fn min(self) -> f64 {
        self.min
    }

    /// Devuelve el límite superior inclusivo.
    pub const fn max(self) -> f64 {
        self.max
    }

    /// Indica si `value` es finito y pertenece al intervalo `[min, max]`.
    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }

    /// Valida un componente continuo para la dimensión `dimension`.
    pub fn validate(self, dimension: usize, value: f64) -> Result<f64, DomainError> {
        ensure_finite("action.continuous", value)?;
        if value < self.min || value > self.max {
            return Err(DomainError::ContinuousActionOutOfRange {
                dimension,
                value,
                min: self.min,
                max: self.max,
            });
        }
        Ok(value)
    }
}

/// Definición explícita del espacio de acciones admitido por un mundo.
#[derive(Debug, Clone, PartialEq)]
pub enum ActionSpace {
    /// Espacio discreto con índices válidos en `[0, cardinality)`.
    Discrete {
        /// Número total de acciones discretas (`>= 1`).
        cardinality: usize,
    },
    /// Espacio continuo de dimensión fija con límites por componente.
    Continuous {
        /// Límites `[min, max]` por cada dimensión (`!bounds.is_empty()`).
        bounds: Vec<ActionBounds>,
    },
}

impl ActionSpace {
    /// Crea un espacio de acciones discreto con `cardinality >= 1`.
    pub fn discrete(cardinality: usize) -> Result<Self, DomainError> {
        if cardinality == 0 {
            return Err(DomainError::EmptyActionSpace);
        }
        Ok(Self::Discrete { cardinality })
    }

    /// Crea un espacio de acciones continuo con al menos una dimensión acotada.
    pub fn continuous(bounds: Vec<ActionBounds>) -> Result<Self, DomainError> {
        if bounds.is_empty() {
            return Err(DomainError::EmptyActionSpace);
        }
        Ok(Self::Continuous { bounds })
    }

    /// Valida que `action` pertenezca a este espacio de acciones.
    pub fn validate(&self, action: &Action) -> Result<(), DomainError> {
        match (self, action) {
            (Self::Discrete { cardinality }, Action::Discrete(discrete)) => {
                if discrete.index() >= *cardinality {
                    return Err(DomainError::DiscreteActionOutOfRange {
                        index: discrete.index(),
                        cardinality: *cardinality,
                    });
                }
                Ok(())
            }
            (Self::Continuous { bounds }, Action::Continuous(values)) => {
                if values.len() != bounds.len() {
                    return Err(DomainError::ActionDimensionMismatch {
                        expected: bounds.len(),
                        actual: values.len(),
                    });
                }
                for (dim, (bound, &value)) in bounds.iter().zip(values.iter()).enumerate() {
                    bound.validate(dim, value)?;
                }
                Ok(())
            }
            (Self::Discrete { .. }, Action::Continuous(_)) => {
                Err(DomainError::IncompatibleActionKind {
                    expected: "discrete",
                    actual: "continuous",
                })
            }
            (Self::Continuous { .. }, Action::Discrete(_)) => {
                Err(DomainError::IncompatibleActionKind {
                    expected: "continuous",
                    actual: "discrete",
                })
            }
        }
    }
}

/// Acción discreta identificada por un índice entero no negativo.
///
/// Tipo de valor inmutable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DiscreteAction(usize);

impl DiscreteAction {
    /// Construye una acción discreta verificando que `index < cardinality`.
    pub fn new(index: usize, cardinality: usize) -> Result<Self, DomainError> {
        if cardinality == 0 {
            return Err(DomainError::EmptyActionSpace);
        }
        if index >= cardinality {
            return Err(DomainError::DiscreteActionOutOfRange { index, cardinality });
        }
        Ok(Self(index))
    }

    /// Devuelve el índice discreto de la acción.
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Acción aplicable a un mundo (sin presuponer un agente ni un sistema externo).
///
/// Tipo de valor inmutable.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Acción discreta indexada.
    Discrete(DiscreteAction),
    /// Vector continuo de comandos escalares finitos.
    Continuous(Vec<f64>),
}

impl Action {
    /// Construye una acción discreta validada contra un [`ActionSpace`].
    pub fn discrete_in_space(index: usize, space: &ActionSpace) -> Result<Self, DomainError> {
        match space {
            ActionSpace::Discrete { cardinality } => {
                Ok(Self::Discrete(DiscreteAction::new(index, *cardinality)?))
            }
            ActionSpace::Continuous { .. } => Err(DomainError::IncompatibleActionKind {
                expected: "continuous",
                actual: "discrete",
            }),
        }
    }

    /// Construye una acción continua verificando finitud de todos sus componentes.
    pub fn continuous(values: Vec<f64>) -> Result<Self, DomainError> {
        ensure_all_finite("action.continuous", &values)?;
        Ok(Self::Continuous(values))
    }

    /// Construye una acción continua validada contra un [`ActionSpace`].
    pub fn continuous_in_space(values: Vec<f64>, space: &ActionSpace) -> Result<Self, DomainError> {
        let action = Self::continuous(values)?;
        space.validate(&action)?;
        Ok(action)
    }

    /// Valida esta acción frente a `space`.
    pub fn validate_against(&self, space: &ActionSpace) -> Result<(), DomainError> {
        space.validate(self)
    }

    /// Devuelve el índice si es una acción discreta.
    pub const fn as_discrete(&self) -> Option<DiscreteAction> {
        match self {
            Self::Discrete(d) => Some(*d),
            Self::Continuous(_) => None,
        }
    }

    /// Devuelve el slice de valores si es una acción continua.
    pub fn as_continuous(&self) -> Option<&[f64]> {
        match self {
            Self::Discrete(_) => None,
            Self::Continuous(v) => Some(v.as_slice()),
        }
    }
}
