//! Métricas de población del lab Life (NOE-104): única fuente de conteos, diversidad y
//! distribución genética que consumen los visores (Studio no las recalcula).

use crate::config::LifeError;
use crate::evolution_experiment::GenomeDistributionStats;
use crate::world::PrivilegedGlobalView;

/// Número de organismos vivos de una especie ([`crate::Genome::species_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeciesCount {
    pub species: u32,
    pub count: usize,
}

/// Métricas de población en un instante, calculadas sobre organismos vivos.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationMetrics {
    pub tick: u64,
    /// Organismos vivos (denominador de todas las medias).
    pub alive: usize,
    /// Nacimientos totales históricos, fundadores incluidos (`lineage().len()`).
    pub total_born: usize,
    /// Muertes totales históricas.
    pub total_died: usize,
    /// Energía media de los vivos (`0.0` si no hay).
    pub mean_energy: f64,
    /// Generación máxima entre los vivos (`0` si no hay).
    pub max_generation: u64,
    /// Conteos por especie viva, ordenados por id de especie (sólo `count > 0`).
    pub species: Vec<SpeciesCount>,
    /// Número de especies vivas.
    pub species_richness: usize,
    /// Índice de Shannon (nats) sobre las especies vivas (`0.0` si `alive <= 1`).
    pub shannon_diversity: f64,
    /// Distribución genética de los vivos.
    pub genome: GenomeDistributionStats,
}

impl PopulationMetrics {
    /// Calcula las métricas desde la vista privilegiada global.
    pub fn from_view(view: &PrivilegedGlobalView<'_>) -> Result<Self, LifeError> {
        let orgs = view.organisms();
        let mut genomes = Vec::new();
        let mut counts = [0usize; crate::genetics::SPECIES_COUNT as usize];
        let mut energy = 0.0;
        let mut max_generation = 0;
        for i in 0..orgs.len() {
            if !orgs.alive()[i] {
                continue;
            }
            let g = orgs.genomes()[i];
            counts[g.species_id() as usize] += 1;
            genomes.push(g);
            energy += orgs.energy()[i];
            let generation = view
                .lineage()
                .iter()
                .find(|r| r.organism_id == orgs.ids()[i])
                .map_or(0, |r| r.generation);
            max_generation = max_generation.max(generation);
        }
        let alive = genomes.len();
        let species: Vec<SpeciesCount> = counts
            .iter()
            .enumerate()
            .filter(|(_, c)| **c > 0)
            .map(|(s, c)| SpeciesCount {
                species: s as u32,
                count: *c,
            })
            .collect();
        let shannon_diversity = if alive > 1 {
            species
                .iter()
                .map(|s| {
                    let p = s.count as f64 / alive as f64;
                    -p * p.ln()
                })
                .sum()
        } else {
            0.0
        };
        let lineage = view.lineage();
        Ok(Self {
            tick: view.tick().as_u64(),
            alive,
            total_born: lineage.len(),
            total_died: lineage.iter().filter(|r| r.death_tick.is_some()).count(),
            mean_energy: if alive > 0 {
                energy / alive as f64
            } else {
                0.0
            },
            max_generation,
            species_richness: species.len(),
            species,
            shannon_diversity,
            genome: GenomeDistributionStats::from_genomes(&genomes)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use gogamen_core::MasterSeed;

    use super::*;
    use crate::{Genome, LifeWorld, LifeWorldConfig};

    fn world() -> LifeWorld {
        let cfg = LifeWorldConfig {
            initial_population: 0,
            ..LifeWorldConfig::default()
        };
        LifeWorld::empty(cfg, MasterSeed::new(1)).unwrap()
    }

    #[test]
    fn species_ids_are_in_range_and_bins_distinguish_genes() {
        let lo = Genome::new(0.5, 0.5, 0.0).unwrap();
        let hi = Genome::new(2.0, 1.5, 0.0).unwrap();
        assert_eq!(lo.species_id(), 0);
        assert_eq!(hi.species_id(), 5);
        assert!(Genome::default().species_id() < crate::SPECIES_COUNT);
    }

    #[test]
    fn metrics_count_species_diversity_and_lineage() {
        let mut w = world();
        let a = Genome::new(0.5, 0.5, 0.0).unwrap();
        let b = Genome::new(2.0, 1.5, 0.0).unwrap();
        w.spawn_organism_with_genome(0, 0, 5.0, a).unwrap();
        w.spawn_organism_with_genome(1, 0, 7.0, a).unwrap();
        w.spawn_organism_with_genome(2, 0, 9.0, b).unwrap();
        let m = PopulationMetrics::from_view(&w.privileged_global_view()).unwrap();
        assert_eq!((m.alive, m.total_born, m.total_died), (3, 3, 0));
        assert_eq!(m.species_richness, 2);
        assert_eq!(
            m.species,
            vec![
                SpeciesCount {
                    species: 0,
                    count: 2
                },
                SpeciesCount {
                    species: 5,
                    count: 1
                }
            ]
        );
        assert!((m.mean_energy - 7.0).abs() < 1e-12);
        let expect = -(2.0f64 / 3.0) * (2.0f64 / 3.0).ln() - (1.0f64 / 3.0) * (1.0f64 / 3.0).ln();
        assert!((m.shannon_diversity - expect).abs() < 1e-12);
        assert_eq!(m.genome.sample_count, 3);
    }

    #[test]
    fn empty_population_has_zero_metrics() {
        let m = PopulationMetrics::from_view(&world().privileged_global_view()).unwrap();
        assert_eq!((m.alive, m.species_richness), (0, 0));
        assert_eq!(m.shannon_diversity, 0.0);
        assert_eq!(m.genome.sample_count, 0);
    }

    #[test]
    fn set_mutation_rate_validates() {
        let mut w = world();
        w.set_mutation_rate(0.5).unwrap();
        assert_eq!(w.config().mutation_rate, 0.5);
        assert!(w.set_mutation_rate(1.5).is_err());
        assert!(w.set_mutation_rate(f64::NAN).is_err());
        assert_eq!(w.config().mutation_rate, 0.5);
    }
}
