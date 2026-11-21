//! Controladores clásicos de referencia para estabilización de CartPole.

use gogamen_core::{Action, ActionSpace, DomainError, Observation};

use crate::physics::CartPoleKinematics;

/// Política de control para seleccionar acciones sobre un entorno o modelo CartPole.
pub trait CartPolePolicy {
    /// Selecciona una [`Action`] válida en `action_space` dada la observación actual.
    fn select_action(
        &mut self,
        observation: &Observation,
        action_space: &ActionSpace,
    ) -> Result<Action, DomainError>;
}

/// Controlador clásico lineal de realimentación de estado (PD/LQR) para CartPole.
///
/// Calcula la fuerza continua:
/// `F = k_x * x + k_x_dot * x_dot + k_theta * theta + k_theta_dot * theta_dot`
/// acotada al rango válido del [`ActionSpace`] continuo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFeedbackController {
    /// Ganancia proporcional sobre la posición del carro `x`.
    pub k_x: f64,
    /// Ganancia derivativa sobre la velocidad del carro `x_dot`.
    pub k_x_dot: f64,
    /// Ganancia proporcional sobre el ángulo del péndulo `theta`.
    pub k_theta: f64,
    /// Ganancia derivativa sobre la velocidad angular `theta_dot`.
    pub k_theta_dot: f64,
}

impl LinearFeedbackController {
    /// Crea un controlador lineal con ganancias explícitas.
    pub fn new(
        k_x: f64,
        k_x_dot: f64,
        k_theta: f64,
        k_theta_dot: f64,
    ) -> Result<Self, DomainError> {
        for (field, value) in [
            ("controller.k_x", k_x),
            ("controller.k_x_dot", k_x_dot),
            ("controller.k_theta", k_theta),
            ("controller.k_theta_dot", k_theta_dot),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFiniteValue { field, value });
            }
        }
        Ok(Self {
            k_x,
            k_x_dot,
            k_theta,
            k_theta_dot,
        })
    }

    /// Controlador PD/LQR de referencia precalibrado para los parámetros estándar de CartPole.
    pub const fn default_stabilizing() -> Self {
        Self {
            k_x: 1.5,
            k_x_dot: 2.5,
            k_theta: 40.0,
            k_theta_dot: 10.0,
        }
    }

    /// Calcula la fuerza sin acotar a partir de la cinemática.
    pub fn compute_force(&self, kinematics: CartPoleKinematics) -> f64 {
        self.k_x * kinematics.x
            + self.k_x_dot * kinematics.x_dot
            + self.k_theta * kinematics.theta
            + self.k_theta_dot * kinematics.theta_dot
    }
}

impl Default for LinearFeedbackController {
    fn default() -> Self {
        Self::default_stabilizing()
    }
}

impl CartPolePolicy for LinearFeedbackController {
    fn select_action(
        &mut self,
        observation: &Observation,
        action_space: &ActionSpace,
    ) -> Result<Action, DomainError> {
        let kinematics = CartPoleKinematics::from_slice(observation.features())?;
        let raw_force = self.compute_force(kinematics);
        match action_space {
            ActionSpace::Continuous { bounds } => {
                if bounds.len() != 1 {
                    return Err(DomainError::ActionDimensionMismatch {
                        expected: 1,
                        actual: bounds.len(),
                    });
                }
                let bound = bounds[0];
                let clamped = raw_force.clamp(bound.min(), bound.max());
                Action::continuous_in_space(vec![clamped], action_space)
            }
            ActionSpace::Discrete { .. } => Err(DomainError::IncompatibleActionKind {
                expected: "continuous",
                actual: "discrete",
            }),
        }
    }
}
