//! Contratos de entorno vivo y actuación (`std`-only), separados del modelo simulado.
//!
//! Este módulo distingue explícitamente:
//! - [`EnvironmentObserver`]: capacidad de solo lectura para observar el entorno vivo.
//! - [`Actuator`]: capacidad exclusiva de efecto para aplicar acciones sobre el entorno vivo.
//! - [`LiveEnvironment`]: entorno vivo completo que combina observación, actuación y reinicio.
//!
//! Ninguno de estos rasgos es requerido ni accesible desde [`crate::GenerativeModel`] o
//! [`crate::PlanningContext`].

use crate::action::{Action, ActionSpace};
use crate::domain::{Observation, Terminal};
use crate::error::EnvironmentError;

/// Resultado inmutable de aplicar una acción sobre un [`Actuator`] del entorno vivo.
#[derive(Debug, Clone, PartialEq)]
pub struct ApplyOutcome {
    observation: Observation,
    terminal: Terminal,
}

impl ApplyOutcome {
    /// Construye un nuevo [`ApplyOutcome`] con la observación y estado terminal resultantes.
    pub const fn new(observation: Observation, terminal: Terminal) -> Self {
        Self {
            observation,
            terminal,
        }
    }

    /// Devuelve la [`Observation`] emitida tras aplicar la acción.
    pub const fn observation(&self) -> &Observation {
        &self.observation
    }

    /// Devuelve el estado de terminación tras aplicar la acción.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Consume el resultado devolviendo la [`Observation`] interna.
    pub fn into_observation(self) -> Observation {
        self.observation
    }
}

/// Capacidad de solo lectura para observar un entorno vivo sin actuar sobre él.
pub trait EnvironmentObserver {
    /// Captura la observación actual del entorno vivo sin mutarlo.
    fn observe(&self) -> Result<Observation, EnvironmentError>;
}

/// Capacidad de actuación sobre un entorno vivo.
///
/// # Separación respecto a la planificación
/// Esta capacidad no debe exponerse al planificador ni al [`crate::GenerativeModel`].
pub trait Actuator {
    /// Devuelve el espacio de acciones válido aceptado por este actuador.
    fn action_space(&self) -> &ActionSpace;

    /// Aplica `action` sobre el entorno vivo y devuelve el [`ApplyOutcome`] resultante.
    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError>;
}

/// Contrato de un entorno vivo completo que soporta observación, actuación y reinicio.
///
/// Un [`LiveEnvironment`] no implementa [`crate::GenerativeModel`] y sus errores usan
/// [`EnvironmentError`] en lugar de [`crate::ModelError`].
///
/// # Ejemplo de aislamiento: un contexto de planificación no puede actuar
///
/// ```compile_fail
/// use gogamen_core::{
///     Action, ActionSpace, DeterministicRng, GenerativeModel, ModelError, Observation,
///     PlanningContext, RngSource, SimDuration, State, StepDt, Terminal, Transition,
/// };
///
/// struct DummyModel {
///     space: ActionSpace,
/// }
///
/// impl GenerativeModel for DummyModel {
///     fn action_space(&self) -> &ActionSpace {
///         &self.space
///     }
///
///     fn step(
///         &self,
///         state: &State,
///         _action: &Action,
///         dt: StepDt,
///         _rng: &mut dyn RngSource,
///     ) -> Result<Transition, ModelError> {
///         let next_state = state
///             .advance(dt, state.values().to_vec())
///             .map_err(ModelError::TransitionFailed)?;
///         let obs = Observation::new(
///             next_state.tick(),
///             next_state.elapsed(),
///             next_state.values().to_vec(),
///         )
///         .map_err(ModelError::TransitionFailed)?;
///         Transition::new(state.tick(), dt, next_state, obs, Terminal::Ongoing)
///             .map_err(ModelError::TransitionFailed)
///     }
/// }
///
/// let model = DummyModel {
///     space: ActionSpace::discrete(2).unwrap(),
/// };
/// let mut rng = DeterministicRng::new(1);
/// let mut ctx = PlanningContext::new(&model, &mut rng);
/// let action = Action::discrete_in_space(0, model.action_space()).unwrap();
/// // Error de compilación: `PlanningContext` y `GenerativeModel` no exponen `apply` ni `Actuator`.
/// let _ = ctx.apply(&action);
/// ```
pub trait LiveEnvironment: EnvironmentObserver + Actuator {
    /// Reinicia el entorno vivo a un nuevo episodio y devuelve la observación inicial.
    fn reset(&mut self) -> Result<Observation, EnvironmentError>;
}
