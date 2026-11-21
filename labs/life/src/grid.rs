//! Grilla 2D discreta de recursos alimenticios (`FoodGrid`) con regeneración acotada.

use gogamen_core::{DeterministicRng, RngSource, TraceHasher};

use crate::config::LifeError;

/// Grilla 2D discreta que almacena la energía de alimento disponible por celda.
#[derive(Debug, Clone, PartialEq)]
pub struct FoodGrid {
    width: usize,
    height: usize,
    max_cell_food: f64,
    cells: Vec<f64>,
}

impl FoodGrid {
    /// Crea una grilla vacía (`0.0` en todas las celdas) con capacidad acotada `max_cell_food`.
    pub fn new(width: usize, height: usize, max_cell_food: f64) -> Result<Self, LifeError> {
        if width == 0 || height == 0 {
            return Err(LifeError::InvalidConfig("grid dimensions must be >= 1"));
        }
        if !max_cell_food.is_finite() || max_cell_food < 0.0 {
            return Err(LifeError::InvalidConfig(
                "max_cell_food must be finite and >= 0.0",
            ));
        }
        Ok(Self {
            width,
            height,
            max_cell_food,
            cells: vec![0.0; width * height],
        })
    }

    /// Crea una grilla sembrada de forma determinista con `rng` en `[0.0, initial_cell_food]`.
    pub fn seeded(
        width: usize,
        height: usize,
        max_cell_food: f64,
        initial_cell_food: f64,
        rng: &mut DeterministicRng,
    ) -> Result<Self, LifeError> {
        let mut grid = Self::new(width, height, max_cell_food)?;
        let cap = initial_cell_food.clamp(0.0, max_cell_food);
        for cell in &mut grid.cells {
            *cell = rng.next_f64() * cap;
        }
        Ok(grid)
    }

    /// Ancho de la grilla en celdas.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Alto de la grilla en celdas.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Límite máximo de alimento por celda.
    pub const fn max_cell_food(&self) -> f64 {
        self.max_cell_food
    }

    /// Vista de solo lectura de todas las celdas en orden fila-mayor (`y * width + x`).
    pub fn cells(&self) -> &[f64] {
        &self.cells
    }

    /// Devuelve `true` si `(x, y)` está dentro de los límites de la grilla.
    pub const fn in_bounds(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height
    }

    /// Lee el alimento disponible en `(x, y)` si está dentro de los límites.
    pub fn get(&self, x: usize, y: usize) -> Option<f64> {
        if self.in_bounds(x, y) {
            Some(self.cells[y * self.width + x])
        } else {
            None
        }
    }

    /// Fija el alimento en `(x, y)` acotado a `[0.0, max_cell_food]`.
    pub fn set(&mut self, x: usize, y: usize, amount: f64) -> Result<f64, LifeError> {
        if !self.in_bounds(x, y) {
            return Err(LifeError::OutOfBounds {
                x,
                y,
                width: self.width,
                height: self.height,
            });
        }
        if !amount.is_finite() || amount < 0.0 {
            return Err(LifeError::InvalidEnergy { energy: amount });
        }
        let clamped = amount.min(self.max_cell_food);
        self.cells[y * self.width + x] = clamped;
        Ok(clamped)
    }

    /// Consume hasta `requested` unidades de alimento en `(x, y)` y devuelve lo consumido.
    pub fn take(&mut self, x: usize, y: usize, requested: f64) -> f64 {
        if !self.in_bounds(x, y) || !requested.is_finite() || requested <= 0.0 {
            return 0.0;
        }
        let idx = y * self.width + x;
        let available = self.cells[idx];
        let taken = available.min(requested);
        self.cells[idx] = available - taken;
        taken
    }

    /// Regenera alimento de forma determinista y acotada a `max_cell_food`.
    ///
    /// Devuelve la energía total efectivamente añadida a la grilla.
    pub fn regenerate(
        &mut self,
        amount_per_cell: f64,
        prob: f64,
        rng: &mut DeterministicRng,
    ) -> f64 {
        if amount_per_cell <= 0.0 || prob <= 0.0 {
            return 0.0;
        }
        let mut added_total = 0.0;
        for cell in &mut self.cells {
            if prob >= 1.0 || rng.next_f64() < prob {
                let headroom = (self.max_cell_food - *cell).max(0.0);
                let added = headroom.min(amount_per_cell);
                *cell += added;
                added_total += added;
            }
        }
        added_total
    }

    /// Suma total de alimento presente actualmente en la grilla.
    pub fn total_food(&self) -> f64 {
        self.cells.iter().sum()
    }

    /// Incorpora el estado completo de la grilla al [`TraceHasher`].
    pub fn write_to_hasher(&self, hasher: &mut TraceHasher) {
        hasher.write_u64(self.width as u64);
        hasher.write_u64(self.height as u64);
        hasher.write_f64(self.max_cell_food);
        for &cell in &self.cells {
            hasher.write_f64(cell);
        }
    }
}
