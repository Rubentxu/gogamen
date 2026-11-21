//! Modelo cinemático de bicicleta 2D para el laboratorio Kart (`gogamen-lab-racing`).
//!
//! # Aproximaciones declaradas del modelo
//! Ver [`MODEL_APPROXIMATIONS_NOTE`]:
//! - Modelo cinemático de bicicleta 2D con referencia en el eje trasero.
//! - **Sin deslizamiento lateral de neumáticos** (no hay ángulo de deriva ni dinámica lateral).
//! - **Sin transferencia de pesos** longitudinal ni lateral.
//! - **Sin fricción compleja de neumáticos** (no Pacejka, no saturación de agarre, no suspensión ni aerodinámica).
//! - Solo resistencia lineal simple (`linear_drag * speed`) y límites acotados de velocidad y dirección.

use std::f64::consts::PI;

use gogamen_core::{DomainError, StepDt};

/// Declaración explícita de las aproximaciones del modelo cinemático 2D de kart.
pub const MODEL_APPROXIMATIONS_NOTE: &str = "Modelo cinemático de bicicleta 2D (eje trasero): \
sin deslizamiento lateral de neumáticos, sin transferencia de pesos, \
sin fricción compleja de neumáticos ni suspensión/aerodinámica; \
usa únicamente arrastre lineal simple y límites acotados de velocidad y dirección.";

/// Distancia entre ejes por defecto en metros.
pub const DEFAULT_WHEELBASE: f64 = 1.2;
/// Ángulo máximo de dirección por defecto en radianes (~0.6 rad ≈ 34.4°).
pub const DEFAULT_MAX_STEERING_ANGLE: f64 = 0.6;
/// Aceleración mínima (frenado/marcha atrás) por defecto en m/s².
pub const DEFAULT_MIN_ACCEL: f64 = -8.0;
/// Aceleración máxima por defecto en m/s².
pub const DEFAULT_MAX_ACCEL: f64 = 6.0;
/// Velocidad mínima por defecto en m/s.
pub const DEFAULT_MIN_SPEED: f64 = -3.0;
/// Velocidad máxima por defecto en m/s.
pub const DEFAULT_MAX_SPEED: f64 = 18.0;
/// Coeficiente de arrastre lineal por defecto (1/s).
pub const DEFAULT_LINEAR_DRAG: f64 = 0.05;
/// Radio efectivo del kart para detección de colisión contra muros/límites (m).
pub const DEFAULT_KART_RADIUS: f64 = 0.45;
/// Paso de integración fijo por defecto (s, 20 Hz).
pub const DEFAULT_KART_DT_SECS: f64 = 0.05;

/// Normaliza un ángulo al rango `(-PI, PI]`.
pub fn wrap_angle(angle: f64) -> f64 {
    let mut a = (angle + PI).rem_euclid(2.0 * PI) - PI;
    if a <= -PI {
        a += 2.0 * PI;
    }
    a
}

/// Configuración física del modelo cinemático de bicicleta 2D.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicBicycleConfig {
    /// Distancia entre ejes (m), debe ser finita y `> 0.0`.
    pub wheelbase: f64,
    /// Límite simétrico del ángulo de dirección `[-max_steering_angle, +max_steering_angle]` (rad).
    pub max_steering_angle: f64,
    /// Aceleración mínima permitida (m/s²).
    pub min_accel: f64,
    /// Aceleración máxima permitida (m/s²).
    pub max_accel: f64,
    /// Velocidad mínima permitida (m/s).
    pub min_speed: f64,
    /// Velocidad máxima permitida (m/s).
    pub max_speed: f64,
    /// Coeficiente de arrastre lineal (`>= 0.0`, 1/s).
    pub linear_drag: f64,
    /// Radio de colisión del kart (m), debe ser finito y `> 0.0`.
    pub kart_radius: f64,
    /// Paso temporal fijo de integración.
    pub dt: StepDt,
}

impl Default for KinematicBicycleConfig {
    fn default() -> Self {
        Self {
            wheelbase: DEFAULT_WHEELBASE,
            max_steering_angle: DEFAULT_MAX_STEERING_ANGLE,
            min_accel: DEFAULT_MIN_ACCEL,
            max_accel: DEFAULT_MAX_ACCEL,
            min_speed: DEFAULT_MIN_SPEED,
            max_speed: DEFAULT_MAX_SPEED,
            linear_drag: DEFAULT_LINEAR_DRAG,
            kart_radius: DEFAULT_KART_RADIUS,
            dt: StepDt::from_secs_f64(DEFAULT_KART_DT_SECS).expect("default kart dt is valid"),
        }
    }
}

impl KinematicBicycleConfig {
    /// Devuelve la declaración explícita de aproximaciones del modelo.
    pub const fn approximations_note(&self) -> &'static str {
        MODEL_APPROXIMATIONS_NOTE
    }

    /// Ajusta el paso temporal fijo de integración.
    pub const fn with_dt(mut self, dt: StepDt) -> Self {
        self.dt = dt;
        self
    }

    /// Ajusta el radio de colisión del kart.
    pub const fn with_kart_radius(mut self, kart_radius: f64) -> Self {
        self.kart_radius = kart_radius;
        self
    }

    /// Valida todos los invariantes físicos de la configuración.
    pub fn validate(&self) -> Result<(), DomainError> {
        if !self.wheelbase.is_finite() || self.wheelbase <= 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "wheelbase must be finite and > 0",
            });
        }
        if !self.max_steering_angle.is_finite()
            || self.max_steering_angle <= 0.0
            || self.max_steering_angle >= std::f64::consts::FRAC_PI_2
        {
            return Err(DomainError::InvalidTransition {
                reason: "max_steering_angle must be finite and in (0, pi/2)",
            });
        }
        if !self.min_accel.is_finite()
            || !self.max_accel.is_finite()
            || self.min_accel >= self.max_accel
        {
            return Err(DomainError::InvalidTransition {
                reason: "accel bounds must be finite with min_accel < max_accel",
            });
        }
        if !self.min_speed.is_finite()
            || !self.max_speed.is_finite()
            || self.min_speed >= self.max_speed
        {
            return Err(DomainError::InvalidTransition {
                reason: "speed bounds must be finite with min_speed < max_speed",
            });
        }
        if !self.linear_drag.is_finite() || self.linear_drag < 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "linear_drag must be finite and >= 0",
            });
        }
        if !self.kart_radius.is_finite() || self.kart_radius <= 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "kart_radius must be finite and > 0",
            });
        }
        Ok(())
    }
}

/// Estado cinemático 2D del kart: `(x, y, yaw, speed)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartState {
    /// Posición X en metros.
    pub x: f64,
    /// Posición Y en metros.
    pub y: f64,
    /// Orientación (heading) en radianes, normalizada en `(-PI, PI]`.
    pub yaw: f64,
    /// Velocidad longitudinal en m/s.
    pub speed: f64,
}

impl KartState {
    /// Construye un [`KartState`] validando que todos sus componentes sean finitos.
    pub fn new(x: f64, y: f64, yaw: f64, speed: f64) -> Result<Self, DomainError> {
        if !x.is_finite() || !y.is_finite() || !yaw.is_finite() || !speed.is_finite() {
            return Err(DomainError::InvalidTransition {
                reason: "kart state components must all be finite",
            });
        }
        Ok(Self {
            x,
            y,
            yaw: wrap_angle(yaw),
            speed,
        })
    }

    /// Serializa el estado cinemático en un vector `[x, y, yaw, speed]`.
    pub fn to_vec(self) -> Vec<f64> {
        vec![self.x, self.y, self.yaw, self.speed]
    }

    /// Decodifica un [`KartState`] desde los primeros 4 elementos de un slice.
    pub fn from_slice(values: &[f64]) -> Result<Self, DomainError> {
        if values.len() < 4 {
            return Err(DomainError::InvalidTransition {
                reason: "kart state requires at least 4 elements [x, y, yaw, speed]",
            });
        }
        Self::new(values[0], values[1], values[2], values[3])
    }
}

/// Control continuo acotado del kart: `[steering_angle, throttle_accel]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartControl {
    /// Ángulo de dirección en radianes.
    pub steering_angle: f64,
    /// Aceleración longitudinal comandada (throttle/freno) en m/s².
    pub throttle_accel: f64,
}

/// Derivadas instantáneas del modelo cinemático de bicicleta 2D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartDerivatives {
    /// Velocidad en X (`m/s`).
    pub x_dot: f64,
    /// Velocidad en Y (`m/s`).
    pub y_dot: f64,
    /// Tasa de giro (`rad/s`).
    pub yaw_dot: f64,
    /// Aceleración longitudinal neta (`m/s²`).
    pub speed_dot: f64,
}

/// Calcula las derivadas del modelo cinemático de bicicleta 2D (sin deslizamiento lateral).
pub fn compute_bicycle_derivatives(
    config: &KinematicBicycleConfig,
    state: KartState,
    control: KartControl,
) -> KartDerivatives {
    let steering = control
        .steering_angle
        .clamp(-config.max_steering_angle, config.max_steering_angle);
    let accel = control
        .throttle_accel
        .clamp(config.min_accel, config.max_accel);

    KartDerivatives {
        x_dot: state.speed * state.yaw.cos(),
        y_dot: state.speed * state.yaw.sin(),
        yaw_dot: (state.speed / config.wheelbase) * steering.tan(),
        speed_dot: accel - config.linear_drag * state.speed,
    }
}

fn add_scaled(state: KartState, deriv: KartDerivatives, scale: f64) -> KartState {
    KartState {
        x: state.x + scale * deriv.x_dot,
        y: state.y + scale * deriv.y_dot,
        yaw: state.yaw + scale * deriv.yaw_dot,
        speed: state.speed + scale * deriv.speed_dot,
    }
}

/// Integra un paso `dt` del modelo cinemático de bicicleta 2D usando RK4.
pub fn integrate_bicycle_step(
    config: &KinematicBicycleConfig,
    state: KartState,
    control: KartControl,
    dt: StepDt,
) -> Result<KartState, DomainError> {
    let h = dt.as_secs_f64();
    let k1 = compute_bicycle_derivatives(config, state, control);
    let k2 = compute_bicycle_derivatives(config, add_scaled(state, k1, 0.5 * h), control);
    let k3 = compute_bicycle_derivatives(config, add_scaled(state, k2, 0.5 * h), control);
    let k4 = compute_bicycle_derivatives(config, add_scaled(state, k3, h), control);

    let next_x = state.x + (h / 6.0) * (k1.x_dot + 2.0 * k2.x_dot + 2.0 * k3.x_dot + k4.x_dot);
    let next_y = state.y + (h / 6.0) * (k1.y_dot + 2.0 * k2.y_dot + 2.0 * k3.y_dot + k4.y_dot);
    let next_yaw =
        state.yaw + (h / 6.0) * (k1.yaw_dot + 2.0 * k2.yaw_dot + 2.0 * k3.yaw_dot + k4.yaw_dot);
    let next_speed = (state.speed
        + (h / 6.0) * (k1.speed_dot + 2.0 * k2.speed_dot + 2.0 * k3.speed_dot + k4.speed_dot))
        .clamp(config.min_speed, config.max_speed);

    KartState::new(next_x, next_y, next_yaw, next_speed)
}
