//! Mundo discreto `LifeWorld`, balance energético, observador privilegiado y `trace_hash`.

use gogamen_core::{DeterministicRng, MasterSeed, RngSource, StreamKey, Tick, TraceHasher};

use crate::config::{LifeError, LifeWorldConfig};
use crate::genetics::{Genome, LineageRecord};
use crate::grid::FoodGrid;
use crate::organism::{LifeAction, LocalCellInfo, LocalOrganismObservation, OrganismSoA};

/// Balance verificable de energía del mundo (`organismos + alimento`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyBalance {
    /// Energía total inicial al construir el mundo.
    pub initial_total: f64,
    /// Energía actual acumulada en organismos vivos.
    pub current_organisms: f64,
    /// Energía actual almacenada en [`FoodGrid`].
    pub current_food: f64,
    /// Energía inyectada mediante organismos sembrados manualmente tras el inicio.
    pub total_spawned_organisms: f64,
    /// Energía inyectada manualmente en celdas de alimento tras el inicio.
    pub total_injected_food: f64,
    /// Energía total añadida por regeneración acotada de alimento.
    pub total_regenerated: f64,
    /// Energía total gastada en metabolismo basal y movimiento.
    pub total_metabolic_spent: f64,
    /// Energía total disipada como coste de reproducción (`reproduction_overhead`).
    pub total_reproduction_overhead: f64,
    /// Energía total transferida desde [`FoodGrid`] hacia organismos.
    pub total_consumed: f64,
}

impl EnergyBalance {
    /// Energía total esperada según entradas y salidas contabilizadas.
    pub fn expected_total(&self) -> f64 {
        self.initial_total
            + self.total_spawned_organisms
            + self.total_injected_food
            + self.total_regenerated
            - self.total_metabolic_spent
            - self.total_reproduction_overhead
    }

    /// Energía total observada actualmente en el mundo (`organismos + grilla`).
    pub fn actual_total(&self) -> f64 {
        self.current_organisms + self.current_food
    }

    /// Discrepancia absoluta entre energía esperada y energía actual.
    pub fn discrepancy(&self) -> f64 {
        (self.actual_total() - self.expected_total()).abs()
    }
}

/// Resumen de un tick discreto ejecutado en [`LifeWorld`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifeTickReport {
    /// Tick recién completado.
    pub tick_completed: Tick,
    /// Organismos vivos al inicio del tick.
    pub alive_before: usize,
    /// Organismos vivos al finalizar el tick.
    pub alive_after: usize,
    /// Muertes ocurridas durante este tick.
    pub deaths: usize,
    /// Nacimientos ocurridos por reproducción durante este tick.
    pub births: usize,
    /// Energía gastada en metabolismo (basal + movimiento) en este tick.
    pub metabolic_spent: f64,
    /// Energía disipada por overhead reproductivo en este tick.
    pub reproduction_overhead: f64,
    /// Alimento consumido de la grilla en este tick.
    pub food_consumed: f64,
    /// Alimento regenerado en la grilla en este tick.
    pub food_regenerated: f64,
    /// Hash determinista del estado global tras completar el tick.
    pub trace_hash: u64,
}

/// Vista global privilegiada explícita para el observador o harness de experimento.
///
/// Separada estructuralmente de [`LocalOrganismObservation`]; los organismos nunca
/// reciben esta vista durante [`LifeWorld::step_tick_with`].
#[derive(Debug, Clone, Copy)]
pub struct PrivilegedGlobalView<'a> {
    tick: Tick,
    width: usize,
    height: usize,
    food_grid: &'a FoodGrid,
    organisms: &'a OrganismSoA,
    lineage: &'a [LineageRecord],
    energy_balance: EnergyBalance,
    trace_hash: u64,
}

impl<'a> PrivilegedGlobalView<'a> {
    /// Tick actual del mundo.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Ancho global de la grilla.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Alto global de la grilla.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Referencia de solo lectura a la grilla global de alimento.
    pub const fn food_grid(&self) -> &'a FoodGrid {
        self.food_grid
    }

    /// Referencia de solo lectura a toda la población SoA.
    pub const fn organisms(&self) -> &'a OrganismSoA {
        self.organisms
    }

    /// Historial completo de linaje de todos los organismos nacidos en el mundo.
    pub const fn lineage(&self) -> &'a [LineageRecord] {
        self.lineage
    }

    /// Balance global de energía del mundo.
    pub const fn energy_balance(&self) -> EnergyBalance {
        self.energy_balance
    }

    /// Hash determinista actual del estado global.
    pub const fn trace_hash(&self) -> u64 {
        self.trace_hash
    }
}

/// Mundo 2D discreto de organismos (SoA) y recursos (`FoodGrid`) determinista por [`MasterSeed`].
#[derive(Debug, Clone, PartialEq)]
pub struct LifeWorld {
    config: LifeWorldConfig,
    tick: Tick,
    seed: MasterSeed,
    rng: DeterministicRng,
    food: FoodGrid,
    organisms: OrganismSoA,
    /// Number of live organisms per cell (`width * height`, row-major). Derived index.
    cell_occupancy_count: Vec<u16>,
    /// Per-cell occupant key: the SoA index of the sole occupant when the count is 1,
    /// otherwise an XOR accumulator of occupant indices (meaningless for count != 1).
    /// This keeps removal O(1) without rescanning when the first occupant leaves.
    cell_first_occupant: Vec<u32>,
    /// When true, dead records are compacted at the end of every tick (opt-in).
    auto_compact: bool,
    lineage: Vec<LineageRecord>,
    next_organism_id: u64,
    initial_total_energy: f64,
    total_spawned_organisms: f64,
    total_injected_food: f64,
    total_regenerated: f64,
    total_metabolic_spent: f64,
    total_reproduction_overhead: f64,
    total_consumed: f64,
}

impl LifeWorld {
    /// Crea un mundo sembrado de forma determinista con `seed` y `config`.
    pub fn new(config: LifeWorldConfig, seed: MasterSeed) -> Result<Self, LifeError> {
        config.validate()?;
        let mut rng = seed.derive(StreamKey::Episode(0));
        let food = FoodGrid::seeded(
            config.width,
            config.height,
            config.max_cell_food,
            config.initial_cell_food,
            &mut rng,
        )?;
        let mut organisms = OrganismSoA::new();
        let mut lineage = Vec::with_capacity(config.initial_population);
        let mut next_organism_id = 1u64;

        for _ in 0..config.initial_population {
            let x = (rng.next_u64() as usize) % config.width;
            let y = (rng.next_u64() as usize) % config.height;
            organisms.push(
                next_organism_id,
                x,
                y,
                config.initial_energy,
                Genome::default(),
            );
            lineage.push(LineageRecord::new(next_organism_id, None, 0, Tick::ZERO));
            next_organism_id += 1;
        }

        let initial_total_energy = food.total_food() + organisms.total_alive_energy();
        let cells = config.width * config.height;
        let mut world = Self {
            config,
            tick: Tick::ZERO,
            seed,
            rng,
            food,
            organisms,
            cell_occupancy_count: vec![0; cells],
            cell_first_occupant: vec![0; cells],
            auto_compact: false,
            lineage,
            next_organism_id,
            initial_total_energy,
            total_spawned_organisms: 0.0,
            total_injected_food: 0.0,
            total_regenerated: 0.0,
            total_metabolic_spent: 0.0,
            total_reproduction_overhead: 0.0,
            total_consumed: 0.0,
        };
        world.rebuild_occupancy();
        Ok(world)
    }

    /// Crea un mundo vacío (sin alimento inicial ni organismos) para fixtures controlados.
    pub fn empty(config: LifeWorldConfig, seed: MasterSeed) -> Result<Self, LifeError> {
        config.validate()?;
        let rng = seed.derive(StreamKey::Episode(0));
        let food = FoodGrid::new(config.width, config.height, config.max_cell_food)?;
        let cells = config.width * config.height;
        Ok(Self {
            config,
            tick: Tick::ZERO,
            seed,
            rng,
            food,
            organisms: OrganismSoA::new(),
            cell_occupancy_count: vec![0; cells],
            cell_first_occupant: vec![0; cells],
            auto_compact: false,
            lineage: Vec::new(),
            next_organism_id: 1,
            initial_total_energy: 0.0,
            total_spawned_organisms: 0.0,
            total_injected_food: 0.0,
            total_regenerated: 0.0,
            total_metabolic_spent: 0.0,
            total_reproduction_overhead: 0.0,
            total_consumed: 0.0,
        })
    }

    /// Enables or disables end-of-tick compaction of dead organism records.
    ///
    /// Disabled by default so that SoA indices, `step_tick` action alignment and
    /// `trace_hash` (which hashes dead records) keep their historical behavior.
    /// When enabled, dead records are removed (see [`LifeWorld::prune_dead`]) at the end of
    /// every tick, so SoA length and per-tick cost are bounded by the live population
    /// (`<= max_population`) instead of the birth history. Organism ids are never reused and
    /// the relative order of survivors is preserved, but SoA indices shift between ticks and
    /// `trace_hash` differs from a run without compaction.
    pub fn set_auto_compact(&mut self, enabled: bool) {
        self.auto_compact = enabled;
    }

    /// Whether end-of-tick compaction is enabled.
    pub const fn auto_compact(&self) -> bool {
        self.auto_compact
    }

    fn cell_index(&self, x: usize, y: usize) -> usize {
        y * self.config.width + x
    }

    /// Registers the live organism `idx` at `(x, y)` in the occupancy index in O(1).
    fn occupancy_add(&mut self, idx: usize, x: usize, y: usize) {
        let c = self.cell_index(x, y);
        debug_assert!(self.cell_occupancy_count[c] < u16::MAX);
        self.cell_occupancy_count[c] += 1;
        self.cell_first_occupant[c] ^= idx as u32;
    }

    /// Removes organism `idx` from `(x, y)` in the occupancy index in O(1).
    fn occupancy_remove(&mut self, idx: usize, x: usize, y: usize) {
        let c = self.cell_index(x, y);
        debug_assert!(self.cell_occupancy_count[c] > 0);
        self.cell_occupancy_count[c] -= 1;
        self.cell_first_occupant[c] ^= idx as u32;
    }

    /// Rebuilds the occupancy index from the SoA (O(N)); used after index-shifting compaction.
    fn rebuild_occupancy(&mut self) {
        self.cell_occupancy_count.fill(0);
        self.cell_first_occupant.fill(0);
        for i in 0..self.organisms.len() {
            if self.organisms.alive()[i] {
                let (x, y) = (self.organisms.pos_x()[i], self.organisms.pos_y()[i]);
                self.occupancy_add(i, x, y);
            }
        }
    }

    /// Configuración del mundo.
    pub const fn config(&self) -> &LifeWorldConfig {
        &self.config
    }

    /// Cambia la tasa de mutación de reproducciones futuras (`[0.0, 1.0]`, finita).
    ///
    /// No altera organismos existentes; no entra en [`LifeWorld::trace_hash`] (los hijos que
    /// nazcan con la nueva tasa sí cambian el hash).
    pub fn set_mutation_rate(&mut self, rate: f64) -> Result<(), LifeError> {
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
            return Err(LifeError::InvalidConfig(
                "mutation_rate must be finite in [0.0, 1.0]",
            ));
        }
        self.config.mutation_rate = rate;
        Ok(())
    }

    /// Tick lógico actual.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Semilla maestra del mundo.
    pub const fn seed(&self) -> MasterSeed {
        self.seed
    }

    /// Cantidad de organismos vivos actualmente.
    pub fn alive_count(&self) -> usize {
        self.organisms.alive_count()
    }

    /// Inserta un nuevo organismo con [`Genome::default()`] respetando `max_population` y `max_energy`.
    pub fn spawn_organism(&mut self, x: usize, y: usize, energy: f64) -> Result<u64, LifeError> {
        self.spawn_organism_with_genome(x, y, energy, Genome::default())
    }

    /// Inserta un nuevo organismo con `genome` explícito respetando `max_population`, `max_energy`
    /// y los límites de cada gen.
    pub fn spawn_organism_with_genome(
        &mut self,
        x: usize,
        y: usize,
        energy: f64,
        genome: Genome,
    ) -> Result<u64, LifeError> {
        if self.alive_count() >= self.config.max_population {
            return Err(LifeError::PopulationLimitReached {
                max_population: self.config.max_population,
            });
        }
        if !self.food.in_bounds(x, y) {
            return Err(LifeError::OutOfBounds {
                x,
                y,
                width: self.config.width,
                height: self.config.height,
            });
        }
        if !energy.is_finite() || energy <= 0.0 || energy > self.config.max_energy {
            return Err(LifeError::InvalidEnergy { energy });
        }
        if !genome.is_valid() {
            return Err(LifeError::InvalidConfig("organism genome out of bounds"));
        }
        let id = self.next_organism_id;
        self.next_organism_id += 1;
        self.organisms.push(id, x, y, energy, genome);
        self.occupancy_add(self.organisms.len() - 1, x, y);
        self.lineage
            .push(LineageRecord::new(id, None, 0, self.tick));
        self.total_spawned_organisms += energy;
        Ok(id)
    }

    /// Historial completo de linaje de todos los organismos nacidos en el mundo.
    pub fn lineage(&self) -> &[LineageRecord] {
        &self.lineage
    }

    /// Busca el registro de linaje de `organism_id`.
    pub fn lineage_of(&self, organism_id: u64) -> Option<&LineageRecord> {
        let idx = organism_id.checked_sub(1)? as usize;
        self.lineage
            .get(idx)
            .filter(|rec| rec.organism_id == organism_id)
    }

    /// Reproduce explícitamente al organismo en el índice `parent_idx` si cumple energía y cupo.
    pub fn reproduce_organism(&mut self, parent_idx: usize) -> Result<u64, LifeError> {
        let (child_id, _) = self.try_reproduce(parent_idx, self.tick)?;
        Ok(child_id)
    }

    fn mark_lineage_dead(&mut self, organism_id: u64, death_tick: Tick) {
        if let Some(idx) = organism_id.checked_sub(1).map(|v| v as usize)
            && let Some(rec) = self.lineage.get_mut(idx)
            && rec.organism_id == organism_id
        {
            rec.death_tick = Some(death_tick);
        }
    }

    fn try_reproduce(
        &mut self,
        parent_idx: usize,
        event_tick: Tick,
    ) -> Result<(u64, bool), LifeError> {
        if parent_idx >= self.organisms.len() || !self.organisms.alive()[parent_idx] {
            return Err(LifeError::CannotReproduce {
                reason: "parent organism is not alive",
            });
        }
        if self.alive_count() >= self.config.max_population {
            return Err(LifeError::PopulationLimitReached {
                max_population: self.config.max_population,
            });
        }
        let parent_energy = self.organisms.energy()[parent_idx];
        if parent_energy < self.config.reproduction_threshold
            || parent_energy < self.config.reproduction_cost
        {
            return Err(LifeError::CannotReproduce {
                reason: "insufficient energy for reproduction",
            });
        }

        let parent_id = self.organisms.ids()[parent_idx];
        let parent_gen = self.lineage_of(parent_id).map_or(0, |rec| rec.generation);
        let px = self.organisms.pos_x()[parent_idx];
        let py = self.organisms.pos_y()[parent_idx];
        let parent_genome = self.organisms.genomes()[parent_idx];

        let child_id = self.next_organism_id;
        self.next_organism_id += 1;

        let mut child_rng = self.seed.derive(StreamKey::Agent {
            episode: 0,
            agent: child_id,
        });
        let child_genome = parent_genome.mutate(
            self.config.mutation_rate,
            self.config.mutation_scale,
            &mut child_rng,
        );

        let parent_energy_after = parent_energy - self.config.reproduction_cost;
        let child_energy = self.config.reproduction_cost - self.config.reproduction_overhead;

        let parent_died = if parent_energy_after <= 0.0 {
            self.occupancy_remove(parent_idx, px, py);
            self.organisms.kill(parent_idx);
            self.mark_lineage_dead(parent_id, event_tick);
            true
        } else {
            self.organisms.set_energy(parent_idx, parent_energy_after);
            false
        };

        self.organisms
            .push(child_id, px, py, child_energy, child_genome);
        self.occupancy_add(self.organisms.len() - 1, px, py);
        self.lineage.push(LineageRecord::new(
            child_id,
            Some(parent_id),
            parent_gen.saturating_add(1),
            event_tick,
        ));
        self.total_reproduction_overhead += self.config.reproduction_overhead;

        Ok((child_id, parent_died))
    }

    /// Fija el alimento de la celda `(x, y)` contabilizando el delta en el balance energético.
    pub fn set_cell_food(&mut self, x: usize, y: usize, amount: f64) -> Result<f64, LifeError> {
        let old = self.food.get(x, y).unwrap_or(0.0);
        let new = self.food.set(x, y, amount)?;
        self.total_injected_food += new - old;
        Ok(new)
    }

    /// Construye la observación local aislada para el organismo en el índice `idx`.
    pub fn observe_organism(&self, idx: usize) -> Option<LocalOrganismObservation> {
        if idx >= self.organisms.len() || !self.organisms.alive()[idx] {
            return None;
        }
        let ox = self.organisms.pos_x()[idx] as isize;
        let oy = self.organisms.pos_y()[idx] as isize;
        let r = self.config.vision_radius as isize;
        let side = (2 * r + 1) as usize;
        let mut local_cells = Vec::with_capacity(side * side);

        for dy in -r..=r {
            for dx in -r..=r {
                let gx = ox + dx;
                let gy = oy + dy;
                if gx < 0
                    || gy < 0
                    || (gx as usize) >= self.config.width
                    || (gy as usize) >= self.config.height
                {
                    local_cells.push(LocalCellInfo {
                        dx,
                        dy,
                        in_bounds: false,
                        food: 0.0,
                        occupied_by_other: false,
                    });
                } else {
                    let ux = gx as usize;
                    let uy = gy as usize;
                    let food = self.food.get(ux, uy).unwrap_or(0.0);
                    let occupied_by_other = self.is_cell_occupied_by_other(ux, uy, idx);
                    local_cells.push(LocalCellInfo {
                        dx,
                        dy,
                        in_bounds: true,
                        food,
                        occupied_by_other,
                    });
                }
            }
        }

        Some(LocalOrganismObservation::new(
            self.tick,
            self.organisms.ids()[idx],
            self.organisms.energy()[idx],
            self.organisms.age()[idx],
            self.organisms.genomes()[idx],
            self.config.vision_radius,
            local_cells,
        ))
    }

    /// O(1): true when a live organism other than `self_idx` is at `(x, y)`.
    ///
    /// Equivalent to a linear scan over live organisms (`i != self_idx && alive[i] && pos == (x, y)`).
    fn is_cell_occupied_by_other(&self, x: usize, y: usize, self_idx: usize) -> bool {
        let c = self.cell_index(x, y);
        let count = self.cell_occupancy_count[c];
        count > 1 || (count == 1 && self.cell_first_occupant[c] as usize != self_idx)
    }

    /// Reference O(N) implementation kept as a test oracle for the occupancy index.
    #[cfg(test)]
    fn is_cell_occupied_by_other_linear(&self, x: usize, y: usize, self_idx: usize) -> bool {
        let len = self.organisms.len();
        for i in 0..len {
            if i != self_idx
                && self.organisms.alive()[i]
                && self.organisms.pos_x()[i] == x
                && self.organisms.pos_y()[i] == y
            {
                return true;
            }
        }
        false
    }

    /// Avanza un tick discreto consultando a cada organismo vivo únicamente mediante
    /// [`LocalOrganismObservation`] (sin acceso a la verdad global).
    pub fn step_tick_with<F>(&mut self, mut decide: F) -> LifeTickReport
    where
        F: FnMut(&LocalOrganismObservation) -> LifeAction,
    {
        let len = self.organisms.len();
        let mut actions = Vec::with_capacity(len);
        for i in 0..len {
            if let Some(obs) = self.observe_organism(i) {
                actions.push(decide(&obs));
            } else {
                actions.push(LifeAction::Stay);
            }
        }
        self.apply_tick_actions(&actions)
    }

    /// Avanza un tick discreto aplicando un slice de acciones alineado con `OrganismSoA`.
    pub fn step_tick(&mut self, actions: &[LifeAction]) -> Result<LifeTickReport, LifeError> {
        if actions.len() != self.organisms.len() {
            return Err(LifeError::ActionCountMismatch {
                expected: self.organisms.len(),
                actual: actions.len(),
            });
        }
        Ok(self.apply_tick_actions(actions))
    }

    fn apply_tick_actions(&mut self, actions: &[LifeAction]) -> LifeTickReport {
        let alive_before = self.alive_count();
        let next_tick = self.tick.next().unwrap_or(self.tick);
        let mut deaths = 0usize;
        let mut births = 0usize;
        let mut tick_metabolic_spent = 0.0;
        let mut tick_reproduction_overhead = 0.0;
        let mut tick_food_consumed = 0.0;

        for (i, &action) in actions.iter().enumerate() {
            if !self.organisms.alive()[i] {
                continue;
            }

            let mut x = self.organisms.pos_x()[i];
            let mut y = self.organisms.pos_y()[i];
            let mut moved = false;

            if action.is_move() {
                let (dx, dy) = action.delta();
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0
                    && ny >= 0
                    && (nx as usize) < self.config.width
                    && (ny as usize) < self.config.height
                {
                    x = nx as usize;
                    y = ny as usize;
                    let (ox, oy) = (self.organisms.pos_x()[i], self.organisms.pos_y()[i]);
                    self.occupancy_remove(i, ox, oy);
                    self.organisms.set_pos(i, x, y);
                    self.occupancy_add(i, x, y);
                    moved = true;
                }
            }

            let cost = self.config.basal_cost + if moved { self.config.move_cost } else { 0.0 };
            let cur_energy = self.organisms.energy()[i];

            if cur_energy <= cost {
                let dead_id = self.organisms.ids()[i];
                tick_metabolic_spent += cur_energy;
                self.occupancy_remove(i, x, y);
                self.organisms.kill(i);
                self.mark_lineage_dead(dead_id, next_tick);
                deaths += 1;
                continue;
            }

            let mut next_energy = cur_energy - cost;
            tick_metabolic_spent += cost;

            let headroom = (self.config.max_energy - next_energy).max(0.0);
            let max_request = headroom.min(self.config.max_eat_per_tick);
            let consumed = self.food.take(x, y, max_request);
            next_energy += consumed;
            tick_food_consumed += consumed;

            self.organisms.set_energy(i, next_energy);
            self.organisms.increment_age(i);

            if action == LifeAction::Reproduce
                && let Ok((_, parent_died)) = self.try_reproduce(i, next_tick)
            {
                births += 1;
                tick_reproduction_overhead += self.config.reproduction_overhead;
                if parent_died {
                    deaths += 1;
                }
            }
        }

        let food_regenerated = self.food.regenerate(
            self.config.food_regen_per_cell,
            self.config.food_regen_prob,
            &mut self.rng,
        );

        self.total_metabolic_spent += tick_metabolic_spent;
        self.total_consumed += tick_food_consumed;
        self.total_regenerated += food_regenerated;
        self.tick = next_tick;
        if self.auto_compact {
            self.prune_dead();
        }

        let alive_after = self.alive_count();
        let trace_hash = self.trace_hash();

        LifeTickReport {
            tick_completed: self.tick,
            alive_before,
            alive_after,
            deaths,
            births,
            metabolic_spent: tick_metabolic_spent,
            reproduction_overhead: tick_reproduction_overhead,
            food_consumed: tick_food_consumed,
            food_regenerated,
            trace_hash,
        }
    }

    /// Elimina físicamente los registros muertos del contenedor SoA.
    pub fn prune_dead(&mut self) -> usize {
        let removed = self.organisms.prune_dead();
        if removed > 0 {
            self.rebuild_occupancy();
        }
        removed
    }

    /// Devuelve el balance de energía actual y acumulado para verificación.
    pub fn energy_balance(&self) -> EnergyBalance {
        EnergyBalance {
            initial_total: self.initial_total_energy,
            current_organisms: self.organisms.total_alive_energy(),
            current_food: self.food.total_food(),
            total_spawned_organisms: self.total_spawned_organisms,
            total_injected_food: self.total_injected_food,
            total_regenerated: self.total_regenerated,
            total_metabolic_spent: self.total_metabolic_spent,
            total_reproduction_overhead: self.total_reproduction_overhead,
            total_consumed: self.total_consumed,
        }
    }

    /// Calcula el hash determinista de 64 bits sobre el estado global completo del mundo.
    pub fn trace_hash(&self) -> u64 {
        let mut hasher = TraceHasher::new();
        hasher.write_u64(self.seed.as_u64());
        hasher.write_tick(self.tick);
        hasher.write_u64(self.rng.state());
        hasher.write_u64(self.next_organism_id);
        self.food.write_to_hasher(&mut hasher);
        self.organisms.write_to_hasher(&mut hasher);
        hasher.write_u64(self.lineage.len() as u64);
        for rec in &self.lineage {
            rec.write_to_hasher(&mut hasher);
        }
        hasher.write_f64(self.initial_total_energy);
        hasher.write_f64(self.total_spawned_organisms);
        hasher.write_f64(self.total_injected_food);
        hasher.write_f64(self.total_regenerated);
        hasher.write_f64(self.total_metabolic_spent);
        hasher.write_f64(self.total_reproduction_overhead);
        hasher.write_f64(self.total_consumed);
        hasher.finish()
    }

    /// Modalidad privilegiada explícita para que el observador externo inspeccione la verdad global.
    pub fn privileged_global_view(&self) -> PrivilegedGlobalView<'_> {
        PrivilegedGlobalView {
            tick: self.tick,
            width: self.config.width,
            height: self.config.height,
            food_grid: &self.food,
            organisms: &self.organisms,
            lineage: &self.lineage,
            energy_balance: self.energy_balance(),
            trace_hash: self.trace_hash(),
        }
    }
}

#[cfg(test)]
mod occupancy_tests {
    use super::*;

    fn churn_config() -> LifeWorldConfig {
        LifeWorldConfig {
            width: 6,
            height: 6,
            initial_population: 10,
            max_population: 20,
            initial_energy: 6.0,
            max_energy: 12.0,
            basal_cost: 1.0,
            move_cost: 0.5,
            max_eat_per_tick: 3.0,
            max_cell_food: 4.0,
            initial_cell_food: 2.0,
            food_regen_per_cell: 1.0,
            food_regen_prob: 0.5,
            vision_radius: 2,
            reproduction_threshold: 3.0,
            reproduction_cost: 2.5,
            reproduction_overhead: 0.5,
            mutation_rate: 0.1,
            mutation_scale: 0.1,
        }
    }

    fn pick(state: &mut u64) -> LifeAction {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        match (*state >> 33) % 7 {
            0 => LifeAction::Stay,
            1 => LifeAction::MoveUp,
            2 => LifeAction::MoveDown,
            3 => LifeAction::MoveLeft,
            4 | 5 => LifeAction::MoveRight,
            _ => LifeAction::Reproduce,
        }
    }

    /// Compares grid-based observations and the raw index against the linear-scan oracle.
    fn assert_matches_oracle(world: &LifeWorld) {
        let w = world.config.width;
        let h = world.config.height;
        for y in 0..h {
            for x in 0..w {
                let expected = (0..world.organisms.len())
                    .filter(|&i| {
                        world.organisms.alive()[i]
                            && world.organisms.pos_x()[i] == x
                            && world.organisms.pos_y()[i] == y
                    })
                    .count();
                assert_eq!(world.cell_occupancy_count[y * w + x] as usize, expected);
            }
        }
        for idx in 0..world.organisms.len() {
            let Some(obs) = world.observe_organism(idx) else {
                continue;
            };
            let ox = world.organisms.pos_x()[idx] as isize;
            let oy = world.organisms.pos_y()[idx] as isize;
            for cell in obs.local_cells() {
                if !cell.in_bounds {
                    assert!(!cell.occupied_by_other);
                    continue;
                }
                let ux = (ox + cell.dx) as usize;
                let uy = (oy + cell.dy) as usize;
                assert_eq!(
                    cell.occupied_by_other,
                    world.is_cell_occupied_by_other_linear(ux, uy, idx)
                );
            }
        }
    }

    fn run(seed: u64, ticks: u64, compact: bool) -> LifeWorld {
        let mut world = LifeWorld::new(churn_config(), MasterSeed::new(seed)).unwrap();
        world.set_auto_compact(compact);
        let mut st = seed ^ 0x9e37_79b9;
        for _ in 0..ticks {
            assert_matches_oracle(&world);
            world.step_tick_with(|_| pick(&mut st));
        }
        assert_matches_oracle(&world);
        world
    }

    #[test]
    fn grid_matches_linear_scan_across_seeds_moves_deaths_and_births() {
        for seed in [1u64, 7, 42, 1234] {
            let world = run(seed, 60, false);
            assert!(
                world.lineage().len() > 10,
                "expected births for seed {seed}"
            );
            assert!(world.organisms.alive_count() < world.organisms.len());
        }
    }

    #[test]
    fn grid_stays_correct_with_manual_spawn_shared_cells_and_prune() {
        let mut world = LifeWorld::empty(churn_config(), MasterSeed::new(3)).unwrap();
        world.spawn_organism(2, 2, 5.0).unwrap();
        world.spawn_organism(2, 2, 5.0).unwrap();
        world.spawn_organism(2, 2, 1.0).unwrap();
        assert_matches_oracle(&world);
        assert!(world.is_cell_occupied_by_other(2, 2, 0));
        world
            .step_tick(&[LifeAction::MoveUp, LifeAction::Stay, LifeAction::Stay])
            .unwrap();
        assert_matches_oracle(&world);
        // Slot 2 died; first occupant (slot 0) moved away: sole occupant must be slot 1.
        assert!(!world.is_cell_occupied_by_other(2, 2, 1));
        assert!(world.is_cell_occupied_by_other(2, 2, 0));
        world.prune_dead();
        assert_matches_oracle(&world);
    }

    #[test]
    fn auto_compact_bounds_memory_by_max_population_not_birth_history() {
        let max_pop = churn_config().max_population;
        let mut total_births = 0usize;
        let mut world = LifeWorld::new(churn_config(), MasterSeed::new(11)).unwrap();
        world.set_auto_compact(true);
        let mut st = 99u64;
        for _ in 0..400 {
            let r = world.step_tick_with(|_| pick(&mut st));
            total_births += r.births;
            assert!(world.organisms.len() <= max_pop);
            assert_eq!(world.organisms.len(), world.organisms.alive_count());
            assert!(world.organisms.capacity() <= 4 * max_pop);
        }
        assert_matches_oracle(&world);
        assert!(total_births > 4 * max_pop, "births={total_births}");
        // Ids stay unique and monotonic; lineage keeps full history.
        assert_eq!(world.lineage().len(), 10 + total_births);
        let ids = world.organisms.ids();
        assert!(ids.windows(2).all(|p| p[0] < p[1]));
    }

    #[test]
    fn auto_compact_is_deterministic_and_preserves_energy_balance() {
        let a = run(5, 120, true);
        let b = run(5, 120, true);
        assert_eq!(a.trace_hash(), b.trace_hash());
        assert!(a.energy_balance().discrepancy() < 1e-9);
    }

    #[test]
    fn default_mode_trace_hash_is_independent_of_occupancy_index() {
        // Same scenario twice without compaction: identical hashes and dead records retained.
        let a = run(21, 80, false);
        let b = run(21, 80, false);
        assert_eq!(a.trace_hash(), b.trace_hash());
        assert!(!a.auto_compact());
    }
}
