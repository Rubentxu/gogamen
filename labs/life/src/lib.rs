//! Laboratorio de vida discreta con metabolismo, recursos y aislamiento de observación
//! (`gogamen-lab-life`, `std`-only + `gogamen-core`).
//!
//! Implementa sin dependencias externas ni UI:
//! - Almacenamiento SoA ([`OrganismSoA`]) y grilla 2D de recursos ([`FoodGrid`]) en [`LifeWorld`].
//! - Ticks discretos (`step_tick` / `step_tick_with`) con metabolismo basal, coste por movimiento,
//!   consumo acotado a `max_energy`, muerte automática (`energy <= 0.0`), límite estricto
//!   `max_population` y regeneración acotada de alimento con balance energético verificable.
//! - Aislamiento de observación: los organismos solo reciben [`LocalOrganismObservation`];
//!   la verdad global solo es accesible mediante [`PrivilegedGlobalView`].
//! - Reproducibilidad determinista por [`gogamen_core::MasterSeed`] y [`LifeWorld::trace_hash`].

pub mod adaptive_agent;
pub mod communication;
pub mod config;
pub mod evolution_experiment;
pub mod genetics;
pub mod grid;
pub mod organism;
pub mod population;
pub mod strategy_learning;
pub mod world;

pub use adaptive_agent::{
    AgentDecision, AgentRunMetrics, HeuristicLifeBaseline, HomeostaticFmcAgent, LOCAL_LIFE_ACTIONS,
    LifeAgentError, LifeCandidateFuture, LifeDriveMapping, LocalLifeModel, LocalLifePhysics,
    SharedWorldComparison, run_shared_world_comparison,
};
pub use communication::{
    CausalInterventionKind, CausalUtilityReport, CommunicationExperimentReport,
    EpistemicChannelAssessment, ForageCue, LifeCommConfig, LifeCommError, SignalCodebook,
    SignalMode, Symbol, VariantMetrics, run_communication_experiment,
};
pub use config::{LifeError, LifeWorldConfig};
pub use evolution_experiment::{
    BaselineComparison, EVOLUTION_EXPERIMENT_SCHEMA_VERSION, EvolutionExperimentReport,
    EvolutionExperimentSpec, EvolutionTrace, EvolutionVariantKind, GeneDistributionStats,
    GenomeDistributionStats, MutationControl, NO_EMERGENCE_WITHOUT_BASELINES_NOTICE, ScalarStats,
    SeedEvolutionTrace, VariantSummary, run_evolution_experiment, run_evolution_variant,
};
pub use genetics::{
    GeneBounds, Genome, LineageRecord, METABOLISM_EFFICIENCY_BOUNDS, SENSING_BIAS_BOUNDS,
    SPECIES_COUNT, SPEED_GENE_BOUNDS,
};
pub use grid::FoodGrid;
pub use organism::{LifeAction, LocalCellInfo, LocalOrganismObservation, OrganismSoA};
pub use population::{PopulationMetrics, SpeciesCount};
pub use strategy_learning::{
    ChannelAblationKind, ChannelAblationResult, ChannelAblationVerdict, LineageId,
    SAFETY_IMMUTABLE_AND_NO_CONSCIOUSNESS_NOTICE, STRATEGY_LEARNING_SCHEMA_VERSION,
    SafetyInvariantCheck, SafetyPolicySpec, StrategyBudget, StrategyComparisonConfig,
    StrategyComparisonReport, StrategyHoldoutMetrics, StrategyKind, StrategyLearningError,
    StrategyLineageRecord, run_strategy_comparison,
};
pub use world::{EnergyBalance, LifeTickReport, LifeWorld, PrivilegedGlobalView};

#[cfg(test)]
mod tests {
    use gogamen_core::MasterSeed;

    use super::*;

    fn greedy_local_forage(obs: &LocalOrganismObservation) -> LifeAction {
        let current_food = obs.cell_at(0, 0).map_or(0.0, |c| c.food);
        if current_food > 0.0 {
            return LifeAction::Stay;
        }
        let candidates = [
            (LifeAction::MoveUp, 0, -1),
            (LifeAction::MoveDown, 0, 1),
            (LifeAction::MoveLeft, -1, 0),
            (LifeAction::MoveRight, 1, 0),
        ];
        let mut best_action = LifeAction::Stay;
        let mut best_food = 0.0;
        for (action, dx, dy) in candidates {
            if let Some(cell) = obs.cell_at(dx, dy)
                && cell.in_bounds
                && cell.food > best_food
            {
                best_food = cell.food;
                best_action = action;
            }
        }
        best_action
    }

    #[test]
    fn ac1_energy_balance_consumption_and_max_energy_cap() {
        let config = LifeWorldConfig {
            width: 5,
            height: 5,
            initial_population: 0,
            max_population: 4,
            initial_energy: 5.0,
            max_energy: 10.0,
            basal_cost: 1.0,
            move_cost: 0.5,
            max_eat_per_tick: 6.0,
            max_cell_food: 12.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.25,
            food_regen_prob: 1.0,
            vision_radius: 1,
            ..LifeWorldConfig::default()
        };
        let mut world = LifeWorld::empty(config, MasterSeed::new(7)).unwrap();
        world.spawn_organism(1, 1, 8.0).unwrap();
        world.set_cell_food(2, 1, 10.0).unwrap();

        // Movimiento a (2, 1): coste = 1.0 (basal) + 0.5 (movimiento) = 1.5 -> energía 6.5.
        // Headroom hasta max_energy (10.0) = 3.5 (< max_eat_per_tick 6.0), por lo que solo consume 3.5.
        let report = world.step_tick(&[LifeAction::MoveRight]).unwrap();
        assert_eq!(report.tick_completed.as_u64(), 1);
        assert_eq!(report.deaths, 0);
        assert!((report.metabolic_spent - 1.5).abs() < 1e-12);
        assert!((report.food_consumed - 3.5).abs() < 1e-12);
        // 25 celdas * 0.25 = 6.25 regenerado.
        assert!((report.food_regenerated - 6.25).abs() < 1e-12);

        let view = world.privileged_global_view();
        assert!((view.organisms().energy()[0] - 10.0).abs() < 1e-12);
        assert_eq!(view.organisms().pos_x()[0], 2);
        assert_eq!(view.organisms().pos_y()[0], 1);
        assert_eq!(view.organisms().age()[0], 1);
        // Celda (2,1) tenía 10.0, consumió 3.5 -> 6.5, regeneró 0.25 -> 6.75.
        assert!((view.food_grid().get(2, 1).unwrap() - 6.75).abs() < 1e-12);
        assert!(view.energy_balance().discrepancy() < 1e-9);
    }

    #[test]
    fn ac1_death_and_population_limits_enforced() {
        let config = LifeWorldConfig {
            width: 4,
            height: 4,
            initial_population: 0,
            max_population: 2,
            initial_energy: 2.0,
            max_energy: 10.0,
            basal_cost: 1.0,
            move_cost: 0.5,
            max_eat_per_tick: 2.0,
            max_cell_food: 5.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            ..LifeWorldConfig::default()
        };
        let mut world = LifeWorld::empty(config.clone(), MasterSeed::new(11)).unwrap();

        world.spawn_organism(0, 0, 1.2).unwrap();
        world.spawn_organism(1, 0, 5.0).unwrap();

        // Intentar superar max_population = 2 falla explícitamente.
        let err = world.spawn_organism(2, 0, 5.0).unwrap_err();
        assert_eq!(err, LifeError::PopulationLimitReached { max_population: 2 });

        // Tick 1: organismo 0 intenta moverse (coste 1.5 >= energía 1.2) -> muere y queda en 0.0.
        let r1 = world
            .step_tick(&[LifeAction::MoveRight, LifeAction::Stay])
            .unwrap();
        assert_eq!(r1.alive_before, 2);
        assert_eq!(r1.alive_after, 1);
        assert_eq!(r1.deaths, 1);
        assert!((r1.metabolic_spent - 2.2).abs() < 1e-12);

        let view = world.privileged_global_view();
        assert!(!view.organisms().alive()[0]);
        assert!((view.organisms().energy()[0] - 0.0).abs() < 1e-12);
        assert!(view.organisms().alive()[1]);
        assert!((view.organisms().energy()[1] - 4.0).abs() < 1e-12);
        assert!(view.energy_balance().discrepancy() < 1e-9);

        // Tras morir uno, hay hueco para sembrar otro sin superar max_population.
        world.spawn_organism(3, 3, 3.0).unwrap();
        assert_eq!(world.alive_count(), 2);

        // También valida que initial_population > max_population sea rechazado.
        let bad_cfg = LifeWorldConfig {
            initial_population: 3,
            max_population: 2,
            ..config
        };
        assert!(matches!(
            bad_cfg.validate(),
            Err(LifeError::InvalidConfig(_))
        ));
    }

    #[test]
    fn ac1_multi_tick_bounded_regen_preserves_energy_balance() {
        let config = LifeWorldConfig {
            width: 10,
            height: 10,
            initial_population: 12,
            max_population: 16,
            initial_energy: 8.0,
            max_energy: 15.0,
            basal_cost: 0.6,
            move_cost: 0.3,
            max_eat_per_tick: 2.5,
            max_cell_food: 6.0,
            initial_cell_food: 4.0,
            food_regen_per_cell: 0.4,
            food_regen_prob: 0.35,
            vision_radius: 2,
            ..LifeWorldConfig::default()
        };
        let mut world = LifeWorld::new(config.clone(), MasterSeed::new(99)).unwrap();

        for _ in 0..40 {
            let report = world.step_tick_with(greedy_local_forage);
            assert!(report.alive_after <= config.max_population);
            let balance = world.energy_balance();
            assert!(
                balance.discrepancy() < 1e-9,
                "energy discrepancy {} too large at tick {}",
                balance.discrepancy(),
                report.tick_completed
            );
        }

        let view = world.privileged_global_view();
        for &cell in view.food_grid().cells() {
            assert!(cell >= 0.0 && cell <= config.max_cell_food + 1e-12);
        }
    }

    #[test]
    fn ac2_master_seed_reproduces_dynamics_and_global_state() {
        let config = LifeWorldConfig::default();
        let seed_a = MasterSeed::new(0x1234_5678_9abc_def0);
        let seed_b = MasterSeed::new(0x0fed_cba9_8765_4321);

        let mut world_a1 = LifeWorld::new(config.clone(), seed_a).unwrap();
        let mut world_a2 = LifeWorld::new(config.clone(), seed_a).unwrap();
        let mut world_b = LifeWorld::new(config, seed_b).unwrap();

        assert_eq!(world_a1.trace_hash(), world_a2.trace_hash());
        assert_ne!(world_a1.trace_hash(), world_b.trace_hash());

        for _ in 0..25 {
            let r1 = world_a1.step_tick_with(greedy_local_forage);
            let r2 = world_a2.step_tick_with(greedy_local_forage);
            let rb = world_b.step_tick_with(greedy_local_forage);
            assert_eq!(r1.trace_hash, r2.trace_hash);
            assert_eq!(r1.alive_after, r2.alive_after);
            assert!((r1.food_consumed - r2.food_consumed).abs() < 1e-12);
            assert_ne!(r1.trace_hash, rb.trace_hash);
        }

        assert_eq!(world_a1, world_a2);
    }

    #[test]
    fn ac3_observation_isolation_and_explicit_privileged_view() {
        let config = LifeWorldConfig {
            width: 6,
            height: 6,
            initial_population: 0,
            max_population: 4,
            initial_energy: 5.0,
            max_energy: 10.0,
            basal_cost: 0.5,
            move_cost: 0.2,
            max_eat_per_tick: 2.0,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            ..LifeWorldConfig::default()
        };
        let mut world = LifeWorld::empty(config, MasterSeed::new(42)).unwrap();
        // Organismo en esquina (0, 0) con radio 1.
        let id0 = world.spawn_organism(0, 0, 5.0).unwrap();
        // Otro organismo justo al lado en (1, 0) y comida en (1, 1).
        world.spawn_organism(1, 0, 7.0).unwrap();
        world.set_cell_food(1, 1, 4.5).unwrap();
        // Comida lejana en (5, 5) fuera de radio 1.
        world.set_cell_food(5, 5, 9.0).unwrap();

        let mut observed_food_sum = 0.0;
        let mut saw_out_of_bounds = false;
        let mut saw_neighbor_occupied = false;

        world.step_tick_with(|obs: &LocalOrganismObservation| {
            if obs.organism_id() == id0 {
                // Solo vecindario (2*1+1)^2 = 9 celdas relativas.
                assert_eq!(obs.local_cells().len(), 9);
                assert_eq!(obs.vision_radius(), 1);
                assert!((obs.energy() - 5.0).abs() < 1e-12);
                assert!(!obs.cell_at(-1, 0).unwrap().in_bounds);
                assert!(!obs.cell_at(0, -1).unwrap().in_bounds);
                assert!(obs.cell_at(1, 0).unwrap().occupied_by_other);
                assert!((obs.cell_at(1, 1).unwrap().food - 4.5).abs() < 1e-12);
                saw_out_of_bounds = !obs.cell_at(-1, -1).unwrap().in_bounds;
                saw_neighbor_occupied = obs.cell_at(1, 0).unwrap().occupied_by_other;
                observed_food_sum = obs.local_cells().iter().map(|c| c.food).sum();
            }
            LifeAction::Stay
        });

        assert!(saw_out_of_bounds);
        assert!(saw_neighbor_occupied);
        // No ve el 9.0 de (5, 5).
        assert!((observed_food_sum - 4.5).abs() < 1e-12);

        // La verdad global solo está disponible mediante PrivilegedGlobalView explícita.
        let global: PrivilegedGlobalView<'_> = world.privileged_global_view();
        assert_eq!(global.width(), 6);
        assert_eq!(global.height(), 6);
        assert!((global.food_grid().get(5, 5).unwrap() - 9.0).abs() < 1e-12);
        assert_eq!(global.organisms().len(), 2);
    }

    #[test]
    fn ac1_reproduction_conserves_energy_and_bounds_mutations() {
        let config = LifeWorldConfig {
            width: 6,
            height: 6,
            initial_population: 0,
            max_population: 8,
            initial_energy: 10.0,
            max_energy: 20.0,
            basal_cost: 1.0,
            move_cost: 0.5,
            max_eat_per_tick: 0.0,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            reproduction_threshold: 10.0,
            reproduction_cost: 6.0,
            reproduction_overhead: 1.25,
            mutation_rate: 1.0,
            mutation_scale: 2.0,
        };
        let seed = MasterSeed::new(0xdead_beef_cafe_babe);
        let mut world_a = LifeWorld::empty(config.clone(), seed).unwrap();
        let mut world_b = LifeWorld::empty(config, seed).unwrap();

        let founder_genome = Genome::new(1.95, 0.55, -0.95).unwrap();
        let parent_a = world_a
            .spawn_organism_with_genome(2, 2, 15.0, founder_genome)
            .unwrap();
        let _parent_b = world_b
            .spawn_organism_with_genome(2, 2, 15.0, founder_genome)
            .unwrap();

        // Tick 1: basal_cost = 1.0 -> 14.0 >= threshold 10.0 -> reproduce!
        // parent_energy_after = 14.0 - 6.0 = 8.0
        // child_energy = 6.0 - 1.25 = 4.75
        let r_a = world_a.step_tick(&[LifeAction::Reproduce]).unwrap();
        let r_b = world_b.step_tick(&[LifeAction::Reproduce]).unwrap();
        assert_eq!(r_a, r_b);
        assert_eq!(r_a.births, 1);
        assert_eq!(r_a.alive_after, 2);
        assert!((r_a.reproduction_overhead - 1.25).abs() < 1e-12);

        let view = world_a.privileged_global_view();
        assert_eq!(view.organisms().ids()[0], parent_a);
        assert!((view.organisms().energy()[0] - 8.0).abs() < 1e-12);
        assert!((view.organisms().energy()[1] - 4.75).abs() < 1e-12);
        assert!(view.energy_balance().discrepancy() < 1e-12);

        let child_genome = view.organisms().genomes()[1];
        assert!(child_genome.is_valid());
        assert!(SPEED_GENE_BOUNDS.contains(child_genome.speed_gene));
        assert!(METABOLISM_EFFICIENCY_BOUNDS.contains(child_genome.metabolism_efficiency));
        assert!(SENSING_BIAS_BOUNDS.contains(child_genome.sensing_bias));
        assert_eq!(
            child_genome,
            world_b.privileged_global_view().organisms().genomes()[1]
        );

        // Un organismo por debajo del umbral no se reproduce ni pierde energía de reproducción.
        let r2 = world_a
            .step_tick(&[LifeAction::Reproduce, LifeAction::Reproduce])
            .unwrap();
        assert_eq!(r2.births, 0);
        assert_eq!(r2.alive_after, 2);
        assert!((r2.reproduction_overhead - 0.0).abs() < 1e-12);
        assert!(world_a.energy_balance().discrepancy() < 1e-12);
    }

    #[test]
    fn ac2_lineage_tracks_generations_parents_and_death_tick() {
        let config = LifeWorldConfig {
            width: 4,
            height: 4,
            initial_population: 0,
            max_population: 6,
            initial_energy: 10.0,
            max_energy: 20.0,
            basal_cost: 1.0,
            move_cost: 0.5,
            max_eat_per_tick: 0.0,
            max_cell_food: 10.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            reproduction_threshold: 5.0,
            reproduction_cost: 4.0,
            reproduction_overhead: 0.0,
            mutation_rate: 0.25,
            mutation_scale: 0.1,
        };
        let mut world = LifeWorld::empty(config, MasterSeed::new(77)).unwrap();
        let g0_id = world.spawn_organism(1, 1, 12.0).unwrap();

        // Tick 1: fundador (gen 0) se reproduce -> hijo gen 1.
        let r1 = world.step_tick(&[LifeAction::Reproduce]).unwrap();
        assert_eq!(r1.births, 1);
        let g1_id = world.privileged_global_view().organisms().ids()[1];

        // Tick 2: fundador (12 - 1 - 4 = 7 -> 6 >= 5) vuelve a reproducirse,
        // pero también probamos que hijo (4 - 1 = 3 < 5) no puede; luego recargamos o
        // hacemos que g0 tenga nieto en otro fixture. Aquí g0 tiene 6 -> 2, nace g1_b.
        // Mejor: reproducimos explícitamente al hijo tras darle energía o con acciones:
        // En Tick 2, g0 Stay (7-1=6), g1 Stay (4-1=3).
        // En su lugar, probemos muerte de g1 por inanición:
        // Tick 2: g0 Stay (7->6), g1 Stay (4->3)
        // Tick 3: g0 Stay (6->5), g1 Stay (3->2)
        // Tick 4: g0 Stay (5->4), g1 Stay (2->1)
        // Tick 5: g0 Stay (4->3), g1 Stay (1<=1 -> muere en tick 5!)
        for _ in 0..3 {
            world
                .step_tick(&[LifeAction::Stay, LifeAction::Stay])
                .unwrap();
        }
        let r5 = world
            .step_tick(&[LifeAction::Stay, LifeAction::Stay])
            .unwrap();
        assert_eq!(r5.tick_completed.as_u64(), 5);
        assert_eq!(r5.deaths, 1);
        assert_eq!(r5.alive_after, 1);

        let rec0 = world.lineage_of(g0_id).unwrap();
        assert_eq!(rec0.parent_id, None);
        assert_eq!(rec0.generation, 0);
        assert_eq!(rec0.birth_tick.as_u64(), 0);
        assert_eq!(rec0.death_tick, None);

        let rec1 = world.lineage_of(g1_id).unwrap();
        assert_eq!(rec1.parent_id, Some(g0_id));
        assert_eq!(rec1.generation, 1);
        assert_eq!(rec1.birth_tick.as_u64(), 1);
        assert_eq!(rec1.death_tick.map(|t| t.as_u64()), Some(5));

        // El actor muerto queda marcado alive=false y prune_dead lo elimina físicamente.
        assert!(!world.privileged_global_view().organisms().alive()[1]);
        assert_eq!(world.prune_dead(), 1);
        assert_eq!(world.privileged_global_view().organisms().len(), 1);
        // El registro histórico de linaje se conserva tras prune_dead.
        assert_eq!(world.lineage().len(), 2);

        // Generación 2 mediante reproducción encadenada:
        let g0_b = world.spawn_organism(0, 0, 15.0).unwrap();
        let g1_b = world.reproduce_organism(1).unwrap(); // birth_tick = 5, gen = 1
        assert_eq!(world.lineage_of(g1_b).unwrap().generation, 1);
        assert_eq!(world.lineage_of(g1_b).unwrap().parent_id, Some(g0_b));

        // Damos energía suficiente al hijo vía comida en tick 6? No, reproduce_organism requiere >= 5.0;
        // aquí child_energy = 4.0, así que con 1 tick de comida (si hubiera) o comprobando gen 1->2
        // con un config de child_energy >= threshold:
        let mut world2 = LifeWorld::empty(
            LifeWorldConfig {
                reproduction_threshold: 5.0,
                reproduction_cost: 5.0,
                reproduction_overhead: 0.0,
                ..world.config().clone()
            },
            MasterSeed::new(88),
        )
        .unwrap();
        let a0 = world2.spawn_organism(0, 0, 12.0).unwrap();
        let a1 = world2.reproduce_organism(0).unwrap();
        let a2 = world2.reproduce_organism(1).unwrap();
        assert_eq!(world2.lineage_of(a0).unwrap().generation, 0);
        assert_eq!(world2.lineage_of(a1).unwrap().generation, 1);
        assert_eq!(world2.lineage_of(a1).unwrap().parent_id, Some(a0));
        assert_eq!(world2.lineage_of(a2).unwrap().generation, 2);
        assert_eq!(world2.lineage_of(a2).unwrap().parent_id, Some(a1));
        assert!(world2.energy_balance().discrepancy() < 1e-12);
    }

    #[test]
    fn ac3_population_budget_prevents_unbounded_growth() {
        let config = LifeWorldConfig {
            width: 4,
            height: 4,
            initial_population: 0,
            max_population: 4,
            initial_energy: 18.0,
            max_energy: 20.0,
            basal_cost: 0.2,
            move_cost: 0.1,
            max_eat_per_tick: 6.0,
            max_cell_food: 20.0,
            initial_cell_food: 20.0,
            food_regen_per_cell: 5.0,
            food_regen_prob: 1.0,
            vision_radius: 1,
            reproduction_threshold: 8.0,
            reproduction_cost: 5.0,
            reproduction_overhead: 0.5,
            mutation_rate: 0.5,
            mutation_scale: 0.25,
        };
        let mut world = LifeWorld::empty(config.clone(), MasterSeed::new(303)).unwrap();
        world.spawn_organism(1, 1, 18.0).unwrap();
        world.set_cell_food(1, 1, 20.0).unwrap();

        for _ in 0..25 {
            let report = world.step_tick_with(|_| LifeAction::Reproduce);
            assert!(report.alive_after <= config.max_population);
            assert!(world.alive_count() <= config.max_population);
            assert!(world.privileged_global_view().organisms().len() <= config.max_population);
            assert!(world.energy_balance().discrepancy() < 1e-9);
        }

        assert_eq!(world.alive_count(), config.max_population);
        assert_eq!(
            world.reproduce_organism(0).unwrap_err(),
            LifeError::PopulationLimitReached {
                max_population: config.max_population
            }
        );
    }
}
