//! Genética acotada, mutación determinista y registro de linaje para `gogamen-lab-life`.

use gogamen_core::{DeterministicRng, RngSource, Tick, TraceHasher};

use crate::config::LifeError;

/// Límites inclusivos `[min, max]` para un gen numérico finito.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeneBounds {
    /// Valor mínimo permitido del gen.
    pub min: f64,
    /// Valor máximo permitido del gen.
    pub max: f64,
}

impl GeneBounds {
    /// Crea un rango `[min, max]` en tiempo de compilación.
    pub const fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Devuelve `true` si `value` es finito y pertenece a `[min, max]`.
    pub fn contains(self, value: f64) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }

    /// Acota `value` al intervalo `[min, max]` (si no es finito, devuelve `min`).
    pub fn clamp(self, value: f64) -> f64 {
        if !value.is_finite() {
            return self.min;
        }
        value.clamp(self.min, self.max)
    }

    /// Amplitud del rango (`max - min`).
    pub fn span(self) -> f64 {
        self.max - self.min
    }

    /// Muestrea un valor uniforme dentro de `[min, max]` usando `rng`.
    pub fn sample(self, rng: &mut DeterministicRng) -> f64 {
        let u = rng.next_f64().clamp(0.0, 1.0);
        self.clamp(self.min + u * self.span())
    }

    /// Aplica una perturbación simétrica acotada en `[-scale * span, +scale * span]`
    /// con probabilidad `rate`, garantizando salida en `[min, max]`.
    pub fn mutate_value(
        self,
        value: f64,
        rate: f64,
        scale: f64,
        rng: &mut DeterministicRng,
    ) -> f64 {
        let base = self.clamp(value);
        if rate <= 0.0 || scale <= 0.0 {
            return base;
        }
        if rate >= 1.0 || rng.next_f64() < rate {
            let signed_unit = rng.next_f64() * 2.0 - 1.0;
            let delta = signed_unit * scale * self.span();
            self.clamp(base + delta)
        } else {
            base
        }
    }
}

/// Límites del gen `speed_gene` (`[0.5, 2.0]`).
pub const SPEED_GENE_BOUNDS: GeneBounds = GeneBounds::new(0.5, 2.0);
/// Límites del gen `metabolism_efficiency` (`[0.5, 1.5]`).
pub const METABOLISM_EFFICIENCY_BOUNDS: GeneBounds = GeneBounds::new(0.5, 1.5);
/// Límites del gen `sensing_bias` (`[-1.0, 1.0]`).
pub const SENSING_BIAS_BOUNDS: GeneBounds = GeneBounds::new(-1.0, 1.0);

/// Número de clases de especie distintas devueltas por [`Genome::species_id`].
pub const SPECIES_COUNT: u32 = 6;

/// Genoma numérico acotado de un organismo en [`crate::LifeWorld`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Genome {
    /// Factor genético de velocidad en `[0.5, 2.0]` (neutro: `1.0`).
    pub speed_gene: f64,
    /// Eficiencia metabólica en `[0.5, 1.5]` (neutro: `1.0`).
    pub metabolism_efficiency: f64,
    /// Sesgo sensorial en `[-1.0, 1.0]` (neutro: `0.0`).
    pub sensing_bias: f64,
}

impl Default for Genome {
    fn default() -> Self {
        Self {
            speed_gene: 1.0,
            metabolism_efficiency: 1.0,
            sensing_bias: 0.0,
        }
    }
}

impl Genome {
    /// Construye un [`Genome`] validando que todos sus genes sean finitos y estén acotados.
    pub fn new(
        speed_gene: f64,
        metabolism_efficiency: f64,
        sensing_bias: f64,
    ) -> Result<Self, LifeError> {
        if !SPEED_GENE_BOUNDS.contains(speed_gene) {
            return Err(LifeError::InvalidGene {
                gene: "speed_gene",
                value: speed_gene,
            });
        }
        if !METABOLISM_EFFICIENCY_BOUNDS.contains(metabolism_efficiency) {
            return Err(LifeError::InvalidGene {
                gene: "metabolism_efficiency",
                value: metabolism_efficiency,
            });
        }
        if !SENSING_BIAS_BOUNDS.contains(sensing_bias) {
            return Err(LifeError::InvalidGene {
                gene: "sensing_bias",
                value: sensing_bias,
            });
        }
        Ok(Self {
            speed_gene,
            metabolism_efficiency,
            sensing_bias,
        })
    }

    /// Genera un genoma aleatorio determinista dentro de los límites de cada gen.
    pub fn random(rng: &mut DeterministicRng) -> Self {
        Self {
            speed_gene: SPEED_GENE_BOUNDS.sample(rng),
            metabolism_efficiency: METABOLISM_EFFICIENCY_BOUNDS.sample(rng),
            sensing_bias: SENSING_BIAS_BOUNDS.sample(rng),
        }
    }

    /// Devuelve `true` si todos los genes son finitos y están dentro de sus límites.
    pub fn is_valid(&self) -> bool {
        SPEED_GENE_BOUNDS.contains(self.speed_gene)
            && METABOLISM_EFFICIENCY_BOUNDS.contains(self.metabolism_efficiency)
            && SENSING_BIAS_BOUNDS.contains(self.sensing_bias)
    }

    /// Produce un genoma hijo mutado de forma reproducible y estrictamente acotada.
    pub fn mutate(&self, rate: f64, scale: f64, rng: &mut DeterministicRng) -> Self {
        Self {
            speed_gene: SPEED_GENE_BOUNDS.mutate_value(self.speed_gene, rate, scale, rng),
            metabolism_efficiency: METABOLISM_EFFICIENCY_BOUNDS.mutate_value(
                self.metabolism_efficiency,
                rate,
                scale,
                rng,
            ),
            sensing_bias: SENSING_BIAS_BOUNDS.mutate_value(self.sensing_bias, rate, scale, rng),
        }
    }

    /// Clase de especie del genoma: bin de genotipo discreto y determinista
    /// (`speed_gene` en 3 tramos x `metabolism_efficiency` en 2 tramos = [`SPECIES_COUNT`] clases).
    pub fn species_id(&self) -> u32 {
        let speed = SPEED_GENE_BOUNDS.clamp(self.speed_gene);
        let meta = METABOLISM_EFFICIENCY_BOUNDS.clamp(self.metabolism_efficiency);
        let s = (((speed - SPEED_GENE_BOUNDS.min) / SPEED_GENE_BOUNDS.span()) * 3.0).floor() as u32;
        let m = (((meta - METABOLISM_EFFICIENCY_BOUNDS.min) / METABOLISM_EFFICIENCY_BOUNDS.span())
            * 2.0)
            .floor() as u32;
        s.min(2) * 2 + m.min(1)
    }

    /// Incorpora el genoma al [`TraceHasher`].
    pub fn write_to_hasher(&self, hasher: &mut TraceHasher) {
        hasher.write_f64(self.speed_gene);
        hasher.write_f64(self.metabolism_efficiency);
        hasher.write_f64(self.sensing_bias);
    }
}

/// Registro histórico de linaje de un organismo en [`crate::LifeWorld`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineageRecord {
    /// Identificador único del organismo.
    pub organism_id: u64,
    /// Identificador del progenitor (`None` para organismos fundadores).
    pub parent_id: Option<u64>,
    /// Generación del organismo (`0` para fundadores, `parent_gen + 1` para descendientes).
    pub generation: u64,
    /// Tick lógico de nacimiento.
    pub birth_tick: Tick,
    /// Tick lógico de muerte (`None` mientras siga vivo).
    pub death_tick: Option<Tick>,
}

impl LineageRecord {
    /// Crea un nuevo registro de linaje vivo en `birth_tick`.
    pub const fn new(
        organism_id: u64,
        parent_id: Option<u64>,
        generation: u64,
        birth_tick: Tick,
    ) -> Self {
        Self {
            organism_id,
            parent_id,
            generation,
            birth_tick,
            death_tick: None,
        }
    }

    /// Incorpora el registro de linaje al [`TraceHasher`].
    pub fn write_to_hasher(&self, hasher: &mut TraceHasher) {
        hasher.write_u64(self.organism_id);
        match self.parent_id {
            Some(pid) => {
                hasher.write_u8(1);
                hasher.write_u64(pid);
            }
            None => hasher.write_u8(0),
        }
        hasher.write_u64(self.generation);
        hasher.write_tick(self.birth_tick);
        match self.death_tick {
            Some(dt) => {
                hasher.write_u8(1);
                hasher.write_tick(dt);
            }
            None => hasher.write_u8(0),
        }
    }
}
