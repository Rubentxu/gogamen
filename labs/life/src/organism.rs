//! Almacenamiento SoA de organismos (`OrganismSoA`), acciones y observación local aislada.

use gogamen_core::{Tick, TraceHasher};

use crate::genetics::Genome;

/// Acción discreta de un organismo en un tick de [`crate::LifeWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LifeAction {
    /// Permanece en la celda actual (sin coste extra de movimiento).
    #[default]
    Stay,
    /// Intenta desplazarse una celda hacia arriba (`y - 1`).
    MoveUp,
    /// Intenta desplazarse una celda hacia abajo (`y + 1`).
    MoveDown,
    /// Intenta desplazarse una celda hacia la izquierda (`x - 1`).
    MoveLeft,
    /// Intenta desplazarse una celda hacia la derecha (`x + 1`).
    MoveRight,
    /// Intenta reproducirse asexualmente si cumple umbral energético y cupo poblacional.
    Reproduce,
}

impl LifeAction {
    /// Desplazamiento relativo `(dx, dy)` asociado a la acción.
    pub const fn delta(self) -> (isize, isize) {
        match self {
            Self::Stay | Self::Reproduce => (0, 0),
            Self::MoveUp => (0, -1),
            Self::MoveDown => (0, 1),
            Self::MoveLeft => (-1, 0),
            Self::MoveRight => (1, 0),
        }
    }

    /// Devuelve `true` si la acción implica intento de movimiento.
    pub const fn is_move(self) -> bool {
        matches!(
            self,
            Self::MoveUp | Self::MoveDown | Self::MoveLeft | Self::MoveRight
        )
    }
}

/// Información de una celda relativa dentro del vecindario local de un organismo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalCellInfo {
    /// Desplazamiento relativo en X respecto al organismo (`-radius..=radius`).
    pub dx: isize,
    /// Desplazamiento relativo en Y respecto al organismo (`-radius..=radius`).
    pub dy: isize,
    /// Indica si la celda cae dentro de los límites de la grilla.
    pub in_bounds: bool,
    /// Alimento visible en la celda (`0.0` si está fuera de límites).
    pub food: f64,
    /// Indica si hay otro organismo vivo en esa celda (`false` si está fuera de límites).
    pub occupied_by_other: bool,
}

/// Observación local aislada entregada a un organismo.
///
/// Estructuralmente omite coordenadas globales, dimensiones del mundo, población total,
/// energía de otros organismos y semilla maestra, garantizando aislamiento de observación.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalOrganismObservation {
    tick: Tick,
    organism_id: u64,
    energy: f64,
    age: u64,
    genome: Genome,
    vision_radius: usize,
    local_cells: Vec<LocalCellInfo>,
}

impl LocalOrganismObservation {
    /// Construye una observación local para un organismo concreto.
    pub(crate) fn new(
        tick: Tick,
        organism_id: u64,
        energy: f64,
        age: u64,
        genome: Genome,
        vision_radius: usize,
        local_cells: Vec<LocalCellInfo>,
    ) -> Self {
        Self {
            tick,
            organism_id,
            energy,
            age,
            genome,
            vision_radius,
            local_cells,
        }
    }

    /// Tick actual de la observación.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Identificador del organismo observador.
    pub const fn organism_id(&self) -> u64 {
        self.organism_id
    }

    /// Energía propia del organismo observador.
    pub const fn energy(&self) -> f64 {
        self.energy
    }

    /// Edad propia del organismo en ticks.
    pub const fn age(&self) -> u64 {
        self.age
    }

    /// Genoma propio del organismo observador.
    pub const fn genome(&self) -> Genome {
        self.genome
    }

    /// Radio de visión local (Chebyshev).
    pub const fn vision_radius(&self) -> usize {
        self.vision_radius
    }

    /// Celdas visibles en el vecindario local `[-r..=r] x [-r..=r]`.
    pub fn local_cells(&self) -> &[LocalCellInfo] {
        &self.local_cells
    }

    /// Busca la celda relativa `(dx, dy)` dentro del vecindario observado.
    pub fn cell_at(&self, dx: isize, dy: isize) -> Option<&LocalCellInfo> {
        self.local_cells
            .iter()
            .find(|cell| cell.dx == dx && cell.dy == dy)
    }
}

/// Almacenamiento Structure-of-Arrays (SoA) para la población de organismos.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OrganismSoA {
    ids: Vec<u64>,
    pos_x: Vec<usize>,
    pos_y: Vec<usize>,
    energy: Vec<f64>,
    alive: Vec<bool>,
    age: Vec<u64>,
    genomes: Vec<Genome>,
}

impl OrganismSoA {
    /// Crea un contenedor SoA vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Número total de registros (vivos y muertos) en el SoA.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Indica si no hay registros en el SoA.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Cantidad de organismos actualmente vivos (`alive == true`).
    pub fn alive_count(&self) -> usize {
        self.alive.iter().filter(|&&a| a).count()
    }

    /// Suma de energía de todos los organismos vivos.
    pub fn total_alive_energy(&self) -> f64 {
        self.energy
            .iter()
            .zip(self.alive.iter())
            .filter_map(|(&e, &alive)| if alive { Some(e) } else { None })
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.ids.capacity().min(self.genomes.capacity())
    }

    /// Slice de identificadores únicos de organismos.
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }

    /// Slice de coordenadas X.
    pub fn pos_x(&self) -> &[usize] {
        &self.pos_x
    }

    /// Slice de coordenadas Y.
    pub fn pos_y(&self) -> &[usize] {
        &self.pos_y
    }

    /// Slice de energías actuales.
    pub fn energy(&self) -> &[f64] {
        &self.energy
    }

    /// Slice de banderas de vida (`true` = vivo).
    pub fn alive(&self) -> &[bool] {
        &self.alive
    }

    /// Slice de edades en ticks.
    pub fn age(&self) -> &[u64] {
        &self.age
    }

    /// Slice de genomas de los organismos.
    pub fn genomes(&self) -> &[Genome] {
        &self.genomes
    }

    pub(crate) fn push(&mut self, id: u64, x: usize, y: usize, energy: f64, genome: Genome) {
        self.ids.push(id);
        self.pos_x.push(x);
        self.pos_y.push(y);
        self.energy.push(energy);
        self.alive.push(true);
        self.age.push(0);
        self.genomes.push(genome);
    }

    pub(crate) fn set_pos(&mut self, idx: usize, x: usize, y: usize) {
        self.pos_x[idx] = x;
        self.pos_y[idx] = y;
    }

    pub(crate) fn set_energy(&mut self, idx: usize, energy: f64) {
        self.energy[idx] = energy;
    }

    pub(crate) fn kill(&mut self, idx: usize) {
        self.alive[idx] = false;
        self.energy[idx] = 0.0;
    }

    pub(crate) fn increment_age(&mut self, idx: usize) {
        self.age[idx] = self.age[idx].saturating_add(1);
    }

    /// Elimina físicamente los registros muertos (`alive == false`) preservando el orden relativo.
    pub fn prune_dead(&mut self) -> usize {
        let before = self.ids.len();
        let mut write_idx = 0;
        for read_idx in 0..before {
            if self.alive[read_idx] {
                if write_idx != read_idx {
                    self.ids[write_idx] = self.ids[read_idx];
                    self.pos_x[write_idx] = self.pos_x[read_idx];
                    self.pos_y[write_idx] = self.pos_y[read_idx];
                    self.energy[write_idx] = self.energy[read_idx];
                    self.alive[write_idx] = self.alive[read_idx];
                    self.age[write_idx] = self.age[read_idx];
                    self.genomes[write_idx] = self.genomes[read_idx];
                }
                write_idx += 1;
            }
        }
        self.ids.truncate(write_idx);
        self.pos_x.truncate(write_idx);
        self.pos_y.truncate(write_idx);
        self.energy.truncate(write_idx);
        self.alive.truncate(write_idx);
        self.age.truncate(write_idx);
        self.genomes.truncate(write_idx);
        before - write_idx
    }

    /// Incorpora el estado completo de los organismos al [`TraceHasher`].
    pub fn write_to_hasher(&self, hasher: &mut TraceHasher) {
        hasher.write_u64(self.ids.len() as u64);
        for i in 0..self.ids.len() {
            hasher.write_u64(self.ids[i]);
            hasher.write_u64(self.pos_x[i] as u64);
            hasher.write_u64(self.pos_y[i] as u64);
            hasher.write_f64(self.energy[i]);
            hasher.write_u8(u8::from(self.alive[i]));
            hasher.write_u64(self.age[i]);
            self.genomes[i].write_to_hasher(hasher);
        }
    }
}
