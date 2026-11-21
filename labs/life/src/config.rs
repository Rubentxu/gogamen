//! Configuración y errores del laboratorio de vida discreta (`gogamen-lab-life`).

use std::fmt;

/// Error de configuración o ejecución en [`crate::LifeWorld`].
#[derive(Debug, Clone, PartialEq)]
pub enum LifeError {
    /// Parámetro de configuración inválido o no finito.
    InvalidConfig(&'static str),
    /// Se intentó superar el límite máximo de población viva (`max_population`).
    PopulationLimitReached {
        /// Límite máximo configurado.
        max_population: usize,
    },
    /// Coordenadas fuera de los límites de la grilla.
    OutOfBounds {
        /// Coordenada X solicitada.
        x: usize,
        /// Coordenada Y solicitada.
        y: usize,
        /// Ancho de la grilla.
        width: usize,
        /// Alto de la grilla.
        height: usize,
    },
    /// Cantidad de energía inicial inválida para un organismo.
    InvalidEnergy {
        /// Valor de energía recibido.
        energy: f64,
    },
    /// El vector de acciones no coincide con la cantidad de organismos en el mundo.
    ActionCountMismatch {
        /// Cantidad esperada de acciones.
        expected: usize,
        /// Cantidad recibida.
        actual: usize,
    },
    /// Valor de gen inválido o fuera de los límites permitidos.
    InvalidGene {
        /// Nombre del gen.
        gene: &'static str,
        /// Valor recibido.
        value: f64,
    },
    /// El organismo no cumple las condiciones de energía o vida para reproducirse.
    CannotReproduce {
        /// Motivo del rechazo.
        reason: &'static str,
    },
}

impl fmt::Display for LifeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid LifeWorldConfig: {msg}"),
            Self::PopulationLimitReached { max_population } => {
                write!(
                    f,
                    "population limit reached (max_population={max_population})"
                )
            }
            Self::OutOfBounds {
                x,
                y,
                width,
                height,
            } => write!(f, "position ({x}, {y}) out of bounds ({width}x{height})"),
            Self::InvalidEnergy { energy } => write!(f, "invalid organism energy: {energy}"),
            Self::ActionCountMismatch { expected, actual } => {
                write!(
                    f,
                    "action count mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidGene { gene, value } => {
                write!(f, "invalid gene '{gene}': {value}")
            }
            Self::CannotReproduce { reason } => {
                write!(f, "cannot reproduce: {reason}")
            }
        }
    }
}

impl std::error::Error for LifeError {}

/// Configuración determinista de [`crate::LifeWorld`].
#[derive(Debug, Clone, PartialEq)]
pub struct LifeWorldConfig {
    /// Ancho de la grilla 2D discreta (`>= 1`).
    pub width: usize,
    /// Alto de la grilla 2D discreta (`>= 1`).
    pub height: usize,
    /// Población inicial de organismos sembrados al construir el mundo.
    pub initial_population: usize,
    /// Límite estricto de población viva simultánea (`>= 1`).
    pub max_population: usize,
    /// Energía inicial de cada organismo sembrado al inicio (`> 0.0` y `<= max_energy`).
    pub initial_energy: f64,
    /// Capacidad máxima de energía de un organismo (`> 0.0`).
    pub max_energy: f64,
    /// Coste metabólico basal por tick (`>= 0.0`).
    pub basal_cost: f64,
    /// Coste metabólico adicional por desplazarse una celda (`>= 0.0`).
    pub move_cost: f64,
    /// Máximo de alimento que un organismo puede consumir en un solo tick (`>= 0.0`).
    pub max_eat_per_tick: f64,
    /// Alimento máximo acumulable por celda en [`crate::FoodGrid`] (`>= 0.0`).
    pub max_cell_food: f64,
    /// Alimento inicial máximo sembrado por celda al crear el mundo (`>= 0.0` y `<= max_cell_food`).
    pub initial_cell_food: f64,
    /// Cantidad de alimento regenerada por celda activada en cada tick (`>= 0.0`).
    pub food_regen_per_cell: f64,
    /// Probabilidad `[0.0, 1.0]` de que una celda regenere alimento en un tick dado.
    pub food_regen_prob: f64,
    /// Radio de visión local (Chebyshev) expuesto en [`crate::LocalOrganismObservation`].
    pub vision_radius: usize,
    /// Energía mínima requerida en el progenitor para poder reproducirse (`>= reproduction_cost`).
    pub reproduction_threshold: f64,
    /// Energía descontada al progenitor al reproducirse (`> 0.0`).
    pub reproduction_cost: f64,
    /// Energía disipada como coste de reproducción (`>= 0.0` y `< reproduction_cost`).
    pub reproduction_overhead: f64,
    /// Probabilidad `[0.0, 1.0]` de mutar cada gen al reproducirse.
    pub mutation_rate: f64,
    /// Magnitud relativa de perturbación por mutación (`>= 0.0`).
    pub mutation_scale: f64,
}

impl Default for LifeWorldConfig {
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            initial_population: 8,
            max_population: 32,
            initial_energy: 10.0,
            max_energy: 20.0,
            basal_cost: 0.5,
            move_cost: 0.25,
            max_eat_per_tick: 4.0,
            max_cell_food: 10.0,
            initial_cell_food: 5.0,
            food_regen_per_cell: 0.5,
            food_regen_prob: 0.25,
            vision_radius: 2,
            reproduction_threshold: 12.0,
            reproduction_cost: 6.0,
            reproduction_overhead: 0.5,
            mutation_rate: 0.15,
            mutation_scale: 0.1,
        }
    }
}

impl LifeWorldConfig {
    /// Valida invariantes numéricos y de límites de la configuración.
    pub fn validate(&self) -> Result<(), LifeError> {
        if self.width == 0 || self.height == 0 {
            return Err(LifeError::InvalidConfig("grid dimensions must be >= 1"));
        }
        if self.max_population == 0 {
            return Err(LifeError::InvalidConfig("max_population must be >= 1"));
        }
        if self.initial_population > self.max_population {
            return Err(LifeError::InvalidConfig(
                "initial_population cannot exceed max_population",
            ));
        }
        if !self.max_energy.is_finite() || self.max_energy <= 0.0 {
            return Err(LifeError::InvalidConfig(
                "max_energy must be finite and > 0.0",
            ));
        }
        if !self.initial_energy.is_finite()
            || self.initial_energy <= 0.0
            || self.initial_energy > self.max_energy
        {
            return Err(LifeError::InvalidConfig(
                "initial_energy must be finite in (0.0, max_energy]",
            ));
        }
        for (val, msg) in [
            (self.basal_cost, "basal_cost must be finite and >= 0.0"),
            (self.move_cost, "move_cost must be finite and >= 0.0"),
            (
                self.max_eat_per_tick,
                "max_eat_per_tick must be finite and >= 0.0",
            ),
            (
                self.max_cell_food,
                "max_cell_food must be finite and >= 0.0",
            ),
            (
                self.food_regen_per_cell,
                "food_regen_per_cell must be finite and >= 0.0",
            ),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(LifeError::InvalidConfig(msg));
            }
        }
        if !self.initial_cell_food.is_finite()
            || self.initial_cell_food < 0.0
            || self.initial_cell_food > self.max_cell_food
        {
            return Err(LifeError::InvalidConfig(
                "initial_cell_food must be finite in [0.0, max_cell_food]",
            ));
        }
        if !self.food_regen_prob.is_finite()
            || self.food_regen_prob < 0.0
            || self.food_regen_prob > 1.0
        {
            return Err(LifeError::InvalidConfig(
                "food_regen_prob must be finite in [0.0, 1.0]",
            ));
        }
        if !self.reproduction_cost.is_finite() || self.reproduction_cost <= 0.0 {
            return Err(LifeError::InvalidConfig(
                "reproduction_cost must be finite and > 0.0",
            ));
        }
        if !self.reproduction_overhead.is_finite()
            || self.reproduction_overhead < 0.0
            || self.reproduction_overhead >= self.reproduction_cost
        {
            return Err(LifeError::InvalidConfig(
                "reproduction_overhead must be finite in [0.0, reproduction_cost)",
            ));
        }
        let child_energy = self.reproduction_cost - self.reproduction_overhead;
        if child_energy > self.max_energy {
            return Err(LifeError::InvalidConfig(
                "child_energy (reproduction_cost - reproduction_overhead) cannot exceed max_energy",
            ));
        }
        if !self.reproduction_threshold.is_finite()
            || self.reproduction_threshold < self.reproduction_cost
        {
            return Err(LifeError::InvalidConfig(
                "reproduction_threshold must be finite and >= reproduction_cost",
            ));
        }
        if !self.mutation_rate.is_finite() || self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err(LifeError::InvalidConfig(
                "mutation_rate must be finite in [0.0, 1.0]",
            ));
        }
        if !self.mutation_scale.is_finite() || self.mutation_scale < 0.0 {
            return Err(LifeError::InvalidConfig(
                "mutation_scale must be finite and >= 0.0",
            ));
        }
        Ok(())
    }
}
