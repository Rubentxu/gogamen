//! Dinámica física clásica de péndulo invertido sobre carro (Barto/Sutton CartPole).

use gogamen_core::{DomainError, StepDt, Terminal, TerminalReason};

/// Gravedad estándar por defecto (m/s²).
pub const DEFAULT_GRAVITY: f64 = 9.8;
/// Masa estándar del carro por defecto (kg).
pub const DEFAULT_MASS_CART: f64 = 1.0;
/// Masa estándar del péndulo por defecto (kg).
pub const DEFAULT_MASS_POLE: f64 = 0.1;
/// Semilongitud estándar del péndulo por defecto (m).
pub const DEFAULT_HALF_POLE_LENGTH: f64 = 0.5;
/// Magnitud máxima de fuerza continua por defecto (N).
pub const DEFAULT_MAX_FORCE: f64 = 10.0;
/// Límite horizontal estándar del carril (m).
pub const DEFAULT_X_THRESHOLD: f64 = 2.4;
/// Límite angular estándar del péndulo (12 grados en radianes).
pub const DEFAULT_THETA_THRESHOLD_RADIANS: f64 = 12.0 * std::f64::consts::PI / 180.0;
/// Paso temporal estándar por defecto (20 ms = 50 Hz).
pub const DEFAULT_DT_SECS: f64 = 0.02;

/// Esquema de integración numérica para la dinámica de CartPole.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum IntegratorKind {
    /// Euler explícito clásico (`x_{n+1} = x_n + dt * v_n`, `v_{n+1} = v_n + dt * a_n`).
    Euler,
    /// Euler semi-implícito / simpléctico (`v_{n+1} = v_n + dt * a_n`, `x_{n+1} = x_n + dt * v_{n+1}`).
    #[default]
    SemiImplicitEuler,
    /// Runge-Kutta de orden 4 (RK4).
    Rk4,
}

/// Estado cinemático instantáneo del sistema CartPole (`[x, x_dot, theta, theta_dot]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartPoleKinematics {
    /// Posición horizontal del carro (m).
    pub x: f64,
    /// Velocidad horizontal del carro (m/s).
    pub x_dot: f64,
    /// Ángulo del péndulo respecto a la vertical hacia arriba (rad).
    pub theta: f64,
    /// Velocidad angular del péndulo (rad/s).
    pub theta_dot: f64,
}

impl CartPoleKinematics {
    /// Construye un estado cinemático validando que todas sus componentes sean finitas.
    pub fn new(x: f64, x_dot: f64, theta: f64, theta_dot: f64) -> Result<Self, DomainError> {
        let state = Self {
            x,
            x_dot,
            theta,
            theta_dot,
        };
        state.validate()?;
        Ok(state)
    }

    /// Valida que todas las variables cinemáticas sean finitas.
    pub fn validate(&self) -> Result<(), DomainError> {
        for (field, value) in [
            ("cartpole.x", self.x),
            ("cartpole.x_dot", self.x_dot),
            ("cartpole.theta", self.theta),
            ("cartpole.theta_dot", self.theta_dot),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFiniteValue { field, value });
            }
        }
        Ok(())
    }

    /// Serializa la cinemática como vector `[x, x_dot, theta, theta_dot]`.
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Reconstruye [`CartPoleKinematics`] desde un slice `[x, x_dot, theta, theta_dot]`.
    pub fn from_slice(values: &[f64]) -> Result<Self, DomainError> {
        if values.len() != 4 {
            return Err(DomainError::InvalidTransition {
                reason: "cartpole state must have 4 elements [x, x_dot, theta, theta_dot]",
            });
        }
        Self::new(values[0], values[1], values[2], values[3])
    }

    /// Evalúa si la cinemática viola los límites terminales de posición o ángulo.
    pub fn terminal_status(&self, config: &CartPoleConfig) -> Terminal {
        if self.x.abs() > config.x_threshold || self.theta.abs() > config.theta_threshold_radians {
            Terminal::Terminated(TerminalReason::Failure)
        } else {
            Terminal::Ongoing
        }
    }
}

impl Default for CartPoleKinematics {
    fn default() -> Self {
        Self {
            x: 0.0,
            x_dot: 0.0,
            theta: 0.05,
            theta_dot: 0.0,
        }
    }
}

/// Parámetros físicos y límites terminales de CartPole.
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleConfig {
    /// Aceleración gravitatoria positiva (m/s²).
    pub gravity: f64,
    /// Masa del carro (kg, `> 0.0`).
    pub mass_cart: f64,
    /// Masa del péndulo (kg, `> 0.0`).
    pub mass_pole: f64,
    /// Semilongitud del péndulo (m, `> 0.0`).
    pub half_pole_length: f64,
    /// Magnitud máxima de la fuerza horizontal aplicable (N, `> 0.0`).
    pub max_force: f64,
    /// Límite absoluto de posición del carro (`|x| > x_threshold` termina con `Failure`).
    pub x_threshold: f64,
    /// Límite absoluto angular del péndulo (`|theta| > theta_threshold_radians` termina con `Failure`).
    pub theta_threshold_radians: f64,
    /// Paso temporal por defecto para `LiveEnvironment::apply` y ejecución headless.
    pub dt: StepDt,
    /// Esquema de integración numérica.
    pub integrator: IntegratorKind,
    /// Estado cinemático inicial tras `reset`.
    pub initial_state: CartPoleKinematics,
}

impl Default for CartPoleConfig {
    fn default() -> Self {
        Self {
            gravity: DEFAULT_GRAVITY,
            mass_cart: DEFAULT_MASS_CART,
            mass_pole: DEFAULT_MASS_POLE,
            half_pole_length: DEFAULT_HALF_POLE_LENGTH,
            max_force: DEFAULT_MAX_FORCE,
            x_threshold: DEFAULT_X_THRESHOLD,
            theta_threshold_radians: DEFAULT_THETA_THRESHOLD_RADIANS,
            dt: StepDt::from_secs_f64(DEFAULT_DT_SECS).expect("0.02 s must be valid StepDt"),
            integrator: IntegratorKind::SemiImplicitEuler,
            initial_state: CartPoleKinematics::default(),
        }
    }
}

impl CartPoleConfig {
    /// Configura el estado inicial tras `reset`.
    pub fn with_initial_state(mut self, initial_state: CartPoleKinematics) -> Self {
        self.initial_state = initial_state;
        self
    }

    /// Configura el esquema de integración numérica.
    pub fn with_integrator(mut self, integrator: IntegratorKind) -> Self {
        self.integrator = integrator;
        self
    }

    /// Configura el paso temporal por defecto.
    pub fn with_dt(mut self, dt: StepDt) -> Self {
        self.dt = dt;
        self
    }

    /// Valida invariantes físicos de la configuración.
    pub fn validate(&self) -> Result<(), DomainError> {
        for (field, value) in [
            ("config.gravity", self.gravity),
            ("config.mass_cart", self.mass_cart),
            ("config.mass_pole", self.mass_pole),
            ("config.half_pole_length", self.half_pole_length),
            ("config.max_force", self.max_force),
            ("config.x_threshold", self.x_threshold),
            (
                "config.theta_threshold_radians",
                self.theta_threshold_radians,
            ),
        ] {
            if !value.is_finite() {
                return Err(DomainError::NonFiniteValue { field, value });
            }
            if value <= 0.0 {
                return Err(DomainError::InvalidTransition {
                    reason: "cartpole physical parameters and thresholds must be strictly positive",
                });
            }
        }
        self.initial_state.validate()?;
        Ok(())
    }
}

/// Derivada temporal `(x_dot, x_acc, theta_dot, theta_acc)` de la dinámica CartPole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartPoleDerivatives {
    /// Velocidad del carro `dx/dt`.
    pub x_dot: f64,
    /// Aceleración del carro `d(x_dot)/dt`.
    pub x_acc: f64,
    /// Velocidad angular del péndulo `d(theta)/dt`.
    pub theta_dot: f64,
    /// Aceleración angular del péndulo `d(theta_dot)/dt`.
    pub theta_acc: f64,
}

/// Calcula las aceleraciones instantáneas de Barto/Sutton CartPole dada una fuerza `force` (N).
pub fn compute_derivatives(
    config: &CartPoleConfig,
    state: CartPoleKinematics,
    force: f64,
) -> CartPoleDerivatives {
    let total_mass = config.mass_cart + config.mass_pole;
    let polemass_length = config.mass_pole * config.half_pole_length;
    let cos_theta = state.theta.cos();
    let sin_theta = state.theta.sin();

    let temp =
        (force + polemass_length * state.theta_dot * state.theta_dot * sin_theta) / total_mass;
    let denom = config.half_pole_length
        * (4.0 / 3.0 - (config.mass_pole * cos_theta * cos_theta) / total_mass);
    let theta_acc = (config.gravity * sin_theta - cos_theta * temp) / denom;
    let x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass;

    CartPoleDerivatives {
        x_dot: state.x_dot,
        x_acc,
        theta_dot: state.theta_dot,
        theta_acc,
    }
}

/// Paso temporal máximo (s) que se integra con un único RK4 sin sub-pasos.
pub const RK4_SUBSTEP_DT_SECS: f64 = 0.02;
/// Velocidad angular (rad/s) por encima de la cual se activa el sub-stepping.
pub const RK4_SUBSTEP_THETA_DOT: f64 = 5.0;
/// Giro angular máximo objetivo por sub-paso (rad) cuando `|theta_dot|` es alto.
pub const RK4_SUBSTEP_MAX_ANGLE: f64 = 0.1;
/// Máximo número de sub-pasos RK4 por llamada a [`integrate_step`] (acota el coste).
pub const MAX_RK4_SUBSTEPS: usize = 64;

/// Número determinista de sub-pasos RK4: `1` si `h <= 0.02` y `|theta_dot| <= 5`; si no,
/// `max(ceil(h / 0.02), ceil(|theta_dot| * h / 0.1))` acotado a `[1, MAX_RK4_SUBSTEPS]`.
pub fn rk4_substep_count(state: CartPoleKinematics, h: f64) -> usize {
    if h <= RK4_SUBSTEP_DT_SECS && state.theta_dot.abs() <= RK4_SUBSTEP_THETA_DOT {
        return 1;
    }
    let by_dt = (h / RK4_SUBSTEP_DT_SECS - 1e-9).ceil();
    let by_rate = (state.theta_dot.abs() * h / RK4_SUBSTEP_MAX_ANGLE - 1e-9).ceil();
    let n = by_dt.max(by_rate);
    if !n.is_finite() {
        return MAX_RK4_SUBSTEPS;
    }
    (n as usize).clamp(1, MAX_RK4_SUBSTEPS)
}

/// Un único paso RK4 clásico de tamaño `h` (sin validar).
fn rk4_single_step(
    config: &CartPoleConfig,
    state: CartPoleKinematics,
    force: f64,
    h: f64,
) -> CartPoleKinematics {
    let k1 = compute_derivatives(config, state, force);
    let s2 = CartPoleKinematics {
        x: state.x + 0.5 * h * k1.x_dot,
        x_dot: state.x_dot + 0.5 * h * k1.x_acc,
        theta: state.theta + 0.5 * h * k1.theta_dot,
        theta_dot: state.theta_dot + 0.5 * h * k1.theta_acc,
    };
    let k2 = compute_derivatives(config, s2, force);
    let s3 = CartPoleKinematics {
        x: state.x + 0.5 * h * k2.x_dot,
        x_dot: state.x_dot + 0.5 * h * k2.x_acc,
        theta: state.theta + 0.5 * h * k2.theta_dot,
        theta_dot: state.theta_dot + 0.5 * h * k2.theta_acc,
    };
    let k3 = compute_derivatives(config, s3, force);
    let s4 = CartPoleKinematics {
        x: state.x + h * k3.x_dot,
        x_dot: state.x_dot + h * k3.x_acc,
        theta: state.theta + h * k3.theta_dot,
        theta_dot: state.theta_dot + h * k3.theta_acc,
    };
    let k4 = compute_derivatives(config, s4, force);

    let sixth = h / 6.0;
    CartPoleKinematics {
        x: state.x + sixth * (k1.x_dot + 2.0 * k2.x_dot + 2.0 * k3.x_dot + k4.x_dot),
        x_dot: state.x_dot + sixth * (k1.x_acc + 2.0 * k2.x_acc + 2.0 * k3.x_acc + k4.x_acc),
        theta: state.theta
            + sixth * (k1.theta_dot + 2.0 * k2.theta_dot + 2.0 * k3.theta_dot + k4.theta_dot),
        theta_dot: state.theta_dot
            + sixth * (k1.theta_acc + 2.0 * k2.theta_acc + 2.0 * k3.theta_acc + k4.theta_acc),
    }
}

/// Integra un paso físico `dt` aplicando `force` sobre `state` según `config.integrator`.
pub fn integrate_step(
    config: &CartPoleConfig,
    state: CartPoleKinematics,
    force: f64,
    dt: StepDt,
) -> Result<CartPoleKinematics, DomainError> {
    let h = dt.as_secs_f64();
    let next = match config.integrator {
        IntegratorKind::Euler => {
            let d = compute_derivatives(config, state, force);
            CartPoleKinematics {
                x: state.x + h * d.x_dot,
                x_dot: state.x_dot + h * d.x_acc,
                theta: state.theta + h * d.theta_dot,
                theta_dot: state.theta_dot + h * d.theta_acc,
            }
        }
        IntegratorKind::SemiImplicitEuler => {
            let d = compute_derivatives(config, state, force);
            let next_x_dot = state.x_dot + h * d.x_acc;
            let next_theta_dot = state.theta_dot + h * d.theta_acc;
            CartPoleKinematics {
                x: state.x + h * next_x_dot,
                x_dot: next_x_dot,
                theta: state.theta + h * next_theta_dot,
                theta_dot: next_theta_dot,
            }
        }
        IntegratorKind::Rk4 => {
            let n = rk4_substep_count(state, h);
            if n == 1 {
                rk4_single_step(config, state, force, h)
            } else {
                let sub_h = h / n as f64;
                let mut cur = state;
                for _ in 0..n {
                    cur = rk4_single_step(config, cur, force, sub_h);
                }
                cur
            }
        }
    };
    next.validate()?;
    Ok(next)
}

#[cfg(test)]
mod substep_tests {
    use super::*;

    fn energy(c: &CartPoleConfig, s: CartPoleKinematics) -> f64 {
        let (m, mp, l) = (c.mass_cart, c.mass_pole, c.half_pole_length);
        0.5 * (m + mp) * s.x_dot * s.x_dot
            + mp * l * s.x_dot * s.theta_dot * s.theta.cos()
            + 0.5 * mp * l * l * (4.0 / 3.0) * s.theta_dot * s.theta_dot
            + mp * c.gravity * l * s.theta.cos()
    }

    fn max_drift(substep: bool, dt: f64, steps: usize) -> f64 {
        let c = CartPoleConfig::default().with_integrator(IntegratorKind::Rk4);
        let mut s = CartPoleKinematics::new(0.0, 0.0, 2.5, 8.0).unwrap();
        let e0 = energy(&c, s);
        let h = StepDt::from_secs_f64(dt).unwrap();
        let mut worst = 0.0_f64;
        for _ in 0..steps {
            s = if substep {
                integrate_step(&c, s, 0.0, h).unwrap()
            } else {
                rk4_single_step(&c, s, 0.0, dt)
            };
            worst = worst.max((energy(&c, s) - e0).abs());
        }
        worst
    }

    #[test]
    fn ac2_substepping_bounds_energy_drift_below_single_step_rk4() {
        let (dt, steps) = (0.1, 50);
        let single = max_drift(false, dt, steps);
        let sub = max_drift(true, dt, steps);
        eprintln!("single={single:e} sub={sub:e}");
        assert!(sub < 1e-5, "sub-stepped drift too large: {sub}");
        assert!(sub * 1000.0 < single, "sub={sub} single={single}");
    }

    #[test]
    fn small_dt_low_rate_is_bit_identical_to_single_rk4() {
        let c = CartPoleConfig::default().with_integrator(IntegratorKind::Rk4);
        let s = CartPoleKinematics::new(0.1, 0.2, 0.05, 1.0).unwrap();
        assert_eq!(rk4_substep_count(s, 0.02), 1);
        let h = StepDt::from_secs_f64(0.02).unwrap();
        assert_eq!(
            integrate_step(&c, s, 3.0, h).unwrap(),
            rk4_single_step(&c, s, 3.0, 0.02)
        );
    }

    #[test]
    fn substep_count_rules_and_clamp() {
        let low = CartPoleKinematics::new(0.0, 0.0, 0.0, 1.0).unwrap();
        let fast = CartPoleKinematics::new(0.0, 0.0, 0.0, 10.0).unwrap();
        assert_eq!(rk4_substep_count(low, 0.06), 3);
        assert_eq!(rk4_substep_count(low, 0.05), 3);
        assert!(rk4_substep_count(fast, 0.02) >= 2);
        assert_eq!(rk4_substep_count(fast, 100.0), MAX_RK4_SUBSTEPS);
    }

    #[test]
    fn ac3_substepped_trajectories_are_deterministic() {
        let run = || {
            let c = CartPoleConfig::default().with_integrator(IntegratorKind::Rk4);
            let mut s = CartPoleKinematics::new(0.0, 0.0, 2.5, 8.0).unwrap();
            let h = StepDt::from_secs_f64(0.1).unwrap();
            (0..40)
                .map(|i| {
                    s = integrate_step(&c, s, (i % 3) as f64, h).unwrap();
                    s.to_vec()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(), run());
    }
}
