//! Runner headless de experimentos evolutivos para `gogamen-lab-life` (`std`-only).
//!
//! Ejecuta experimentos reproducibles sobre múltiples [`MasterSeed`]s sin UI ni renderer,
//! registrando explícitamente:
//! - Controles de mutación (`mutation_enabled`, `mutation_rate`, `mutation_scale`) y semillas
//!   tanto en [`EvolutionExperimentSpec`] como en [`EvolutionTrace`].
//! - Todos los casos de extinción en el denominador y en la traza (nunca se descartan ni filtran).
//! - Denominadores explícitos (`total_seeds`, `extinct_seeds`, `surviving_seeds`,
//!   `total_births`, `total_deaths`), tasas de supervivencia/extinción, métricas de energía
//!   (media/min/max/dispersión) y distribución de genes por alelo.
//! - Comparación frente a baselines explícitos (`NoMutation`, `RandomAgent`, `FixedStay`)
//!   con la nota epistémica [`NO_EMERGENCE_WITHOUT_BASELINES_NOTICE`] y exportación a JSON
//!   canónico (`to_canonical_json`).

use std::fmt::{self, Write as _};

use gogamen_core::{MasterSeed, TraceHasher};

use crate::config::{LifeError, LifeWorldConfig};
use crate::genetics::{METABOLISM_EFFICIENCY_BOUNDS, SENSING_BIAS_BOUNDS, SPEED_GENE_BOUNDS};
use crate::organism::{LifeAction, LocalOrganismObservation};
use crate::world::LifeWorld;

/// Nota epistémica obligatoria: no se afirma emergencia sin comparar contra baselines explícitos.
pub const NO_EMERGENCE_WITHOUT_BASELINES_NOTICE: &str =
    "No emergence is claimed without explicit comparison against baselines.";

/// Versión de esquema para la exportación JSON canónica del experimento evolutivo.
pub const EVOLUTION_EXPERIMENT_SCHEMA_VERSION: &str = "gogamen.life.evolution_experiment.v1";

/// Control explícito de mutación para un experimento o variante evolutiva.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MutationControl {
    /// Indica si la mutación está habilitada (`true`) o deshabilitada (`false`, modo sin mutación).
    pub mutation_enabled: bool,
    /// Probabilidad `[0.0, 1.0]` de mutar cada gen al reproducirse cuando `mutation_enabled == true`.
    pub mutation_rate: f64,
    /// Amplitud relativa (`>= 0.0`) de perturbación genética cuando `mutation_enabled == true`.
    pub mutation_scale: f64,
}

impl MutationControl {
    /// Crea un control de mutación habilitado con tasa y amplitud explícitas.
    pub const fn enabled(mutation_rate: f64, mutation_scale: f64) -> Self {
        Self {
            mutation_enabled: true,
            mutation_rate,
            mutation_scale,
        }
    }

    /// Crea un control sin mutación (`mutation_enabled = false`, tasa y amplitud efectivas `0.0`).
    pub const fn disabled() -> Self {
        Self {
            mutation_enabled: false,
            mutation_rate: 0.0,
            mutation_scale: 0.0,
        }
    }

    /// Valida que la tasa y amplitud de mutación sean finitas y estén en rango válido.
    pub fn validate(&self) -> Result<(), LifeError> {
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

    /// Devuelve `(effective_rate, effective_scale)` aplicando `mutation_enabled`.
    pub const fn effective_rate_and_scale(&self) -> (f64, f64) {
        if self.mutation_enabled {
            (self.mutation_rate, self.mutation_scale)
        } else {
            (0.0, 0.0)
        }
    }
}

impl Default for MutationControl {
    fn default() -> Self {
        Self::enabled(0.15, 0.1)
    }
}

/// Variante experimental o baseline ejecutada dentro del estudio evolutivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvolutionVariantKind {
    /// Agente local adaptativo con el control de mutación configurado en la especificación.
    Evolving,
    /// Baseline sin mutación (`mutation_enabled = false`, misma política local).
    NoMutation,
    /// Baseline de agente aleatorio determinista (sin mutación).
    RandomAgent,
    /// Baseline de agente fijo (`LifeAction::Stay`, sin mutación).
    FixedStay,
}

impl EvolutionVariantKind {
    /// Identificador textual canónico de la variante.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Evolving => "evolving",
            Self::NoMutation => "baseline_no_mutation",
            Self::RandomAgent => "baseline_random_agent",
            Self::FixedStay => "baseline_fixed_stay",
        }
    }

    /// Indica si la variante corresponde a un baseline de comparación.
    pub const fn is_baseline(self) -> bool {
        !matches!(self, Self::Evolving)
    }
}

impl fmt::Display for EvolutionVariantKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Especificación reproducible de un experimento evolutivo headless sobre múltiples semillas.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionExperimentSpec {
    /// Identificador del experimento.
    pub experiment_id: String,
    /// Configuración base de [`LifeWorld`].
    pub world_config: LifeWorldConfig,
    /// Número de ticks discretos ejecutados por cada semilla (`>= 1`).
    pub ticks_per_seed: u64,
    /// Lista explícita de semillas maestras evaluadas (no vacía).
    pub seeds: Vec<MasterSeed>,
    /// Control explícito de mutación para la variante principal.
    pub mutation_control: MutationControl,
    /// Baselines explícitos frente a los que se compara la variante principal.
    pub baselines: Vec<EvolutionVariantKind>,
}

impl EvolutionExperimentSpec {
    /// Construye una especificación por defecto con múltiples semillas y baselines explícitos.
    pub fn new(
        experiment_id: impl Into<String>,
        world_config: LifeWorldConfig,
        ticks_per_seed: u64,
        seeds: Vec<MasterSeed>,
        mutation_control: MutationControl,
    ) -> Self {
        Self {
            experiment_id: experiment_id.into(),
            world_config,
            ticks_per_seed,
            seeds,
            mutation_control,
            baselines: vec![
                EvolutionVariantKind::NoMutation,
                EvolutionVariantKind::RandomAgent,
                EvolutionVariantKind::FixedStay,
            ],
        }
    }

    /// Valida invariantes de la especificación antes de ejecutar el experimento.
    pub fn validate(&self) -> Result<(), LifeError> {
        if self.experiment_id.trim().is_empty() {
            return Err(LifeError::InvalidConfig(
                "experiment_id must be a non-empty string",
            ));
        }
        if self.ticks_per_seed == 0 {
            return Err(LifeError::InvalidConfig("ticks_per_seed must be >= 1"));
        }
        if self.seeds.is_empty() {
            return Err(LifeError::InvalidConfig("seeds must not be empty"));
        }
        self.world_config.validate()?;
        self.mutation_control.validate()?;
        Ok(())
    }

    /// Exporta la especificación en JSON canónico sin dependencias externas.
    pub fn to_canonical_json(&self) -> Result<String, LifeError> {
        self.validate()?;
        let mut out = String::new();
        write_spec_json(&mut out, self)?;
        Ok(out)
    }
}

/// Estadísticas descriptivas escalares con denominador explícito (`count`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarStats {
    /// Denominador explícito de muestras incluidas en el cálculo.
    pub count: usize,
    /// Media aritmética (`0.0` si `count == 0`).
    pub mean: f64,
    /// Valor mínimo observado (`0.0` si `count == 0`).
    pub min: f64,
    /// Valor máximo observado (`0.0` si `count == 0`).
    pub max: f64,
    /// Varianza poblacional (`0.0` si `count == 0`).
    pub variance: f64,
    /// Desviación estándar poblacional (`0.0` si `count == 0`).
    pub std_dev: f64,
}

impl ScalarStats {
    /// Calcula media, mínimo, máximo, varianza y desviación estándar sobre `samples`.
    pub fn from_samples(samples: &[f64]) -> Result<Self, LifeError> {
        if samples.is_empty() {
            return Ok(Self {
                count: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                variance: 0.0,
                std_dev: 0.0,
            });
        }
        let mut sum = 0.0;
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for &v in samples {
            if !v.is_finite() {
                return Err(LifeError::InvalidConfig(
                    "non-finite sample encountered while computing statistics",
                ));
            }
            sum += v;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        let n = samples.len() as f64;
        let mean = sum / n;
        let mut sq_diff_sum = 0.0;
        for &v in samples {
            let diff = v - mean;
            sq_diff_sum += diff * diff;
        }
        let variance = sq_diff_sum / n;
        let std_dev = variance.sqrt();
        Ok(Self {
            count: samples.len(),
            mean,
            min,
            max,
            variance,
            std_dev,
        })
    }
}

/// Estadísticas de distribución para un gen acotado concreto.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneDistributionStats {
    /// Nombre del gen (`"speed_gene"`, `"metabolism_efficiency"`, `"sensing_bias"`).
    pub gene_name: String,
    /// Límite inferior permitido del gen.
    pub bounds_min: f64,
    /// Límite superior permitido del gen.
    pub bounds_max: f64,
    /// Estadísticas descriptivas sobre los alelos observados (`stats.count` = denominador).
    pub stats: ScalarStats,
}

/// Distribución conjunta de los tres genes del [`crate::Genome`] sobre una población.
#[derive(Debug, Clone, PartialEq)]
pub struct GenomeDistributionStats {
    /// Denominador explícito de organismos incluidos en la distribución.
    pub sample_count: usize,
    /// Distribución del gen `speed_gene`.
    pub speed_gene: GeneDistributionStats,
    /// Distribución del gen `metabolism_efficiency`.
    pub metabolism_efficiency: GeneDistributionStats,
    /// Distribución del gen `sensing_bias`.
    pub sensing_bias: GeneDistributionStats,
}

impl GenomeDistributionStats {
    /// Calcula la distribución genética sobre un conjunto de genomas.
    pub fn from_genomes(genomes: &[crate::genetics::Genome]) -> Result<Self, LifeError> {
        let mut speeds = Vec::with_capacity(genomes.len());
        let mut metas = Vec::with_capacity(genomes.len());
        let mut biases = Vec::with_capacity(genomes.len());
        for g in genomes {
            speeds.push(g.speed_gene);
            metas.push(g.metabolism_efficiency);
            biases.push(g.sensing_bias);
        }
        Ok(Self {
            sample_count: genomes.len(),
            speed_gene: GeneDistributionStats {
                gene_name: "speed_gene".to_owned(),
                bounds_min: SPEED_GENE_BOUNDS.min,
                bounds_max: SPEED_GENE_BOUNDS.max,
                stats: ScalarStats::from_samples(&speeds)?,
            },
            metabolism_efficiency: GeneDistributionStats {
                gene_name: "metabolism_efficiency".to_owned(),
                bounds_min: METABOLISM_EFFICIENCY_BOUNDS.min,
                bounds_max: METABOLISM_EFFICIENCY_BOUNDS.max,
                stats: ScalarStats::from_samples(&metas)?,
            },
            sensing_bias: GeneDistributionStats {
                gene_name: "sensing_bias".to_owned(),
                bounds_min: SENSING_BIAS_BOUNDS.min,
                bounds_max: SENSING_BIAS_BOUNDS.max,
                stats: ScalarStats::from_samples(&biases)?,
            },
        })
    }
}

/// Traza completa de una semilla individual dentro de una variante (incluye extinciones).
#[derive(Debug, Clone, PartialEq)]
pub struct SeedEvolutionTrace {
    /// Semilla maestra ejecutada.
    pub seed: MasterSeed,
    /// Variante evaluada.
    pub variant: EvolutionVariantKind,
    /// Si la mutación estuvo habilitada en esta ejecución.
    pub mutation_enabled: bool,
    /// Tasa efectiva de mutación aplicada.
    pub effective_mutation_rate: f64,
    /// Amplitud efectiva de mutación aplicada.
    pub effective_mutation_scale: f64,
    /// Número total de ticks ejecutados.
    pub ticks_executed: u64,
    /// Población inicial sembrada.
    pub initial_population: usize,
    /// Población viva al finalizar la ejecución (`0` si hubo extinción).
    pub final_alive: usize,
    /// `true` si la población se extinguió (`final_alive == 0`).
    pub extinct: bool,
    /// Primer tick en el que `alive_after == 0`, si ocurrió extinción.
    pub extinction_tick: Option<u64>,
    /// Total de nacimientos acumulados durante la ejecución.
    pub total_births: usize,
    /// Total de muertes acumuladas durante la ejecución.
    pub total_deaths: usize,
    /// Energía total de los organismos vivos al finalizar (`0.0` si extinto).
    pub final_total_energy: f64,
    /// Estadísticas de energía por organismo vivo al finalizar (`count = final_alive`).
    pub final_organism_energy: ScalarStats,
    /// Máxima generación alcanzada en el linaje de esta semilla.
    pub max_generation_reached: u64,
    /// Hash determinista final del estado global de [`LifeWorld`].
    pub final_trace_hash: u64,
    /// Discrepancia absoluta del balance energético global al finalizar.
    pub energy_balance_discrepancy: f64,
    /// Distribución de genes de los organismos vivos al finalizar (`sample_count = final_alive`).
    pub final_genome_distribution: GenomeDistributionStats,
    /// Trayectoria de organismos vivos al final de cada tick.
    pub tick_alive_counts: Vec<usize>,
}

/// Resumen agregado de una variante sobre todas las semillas (sin descartar extinciones).
#[derive(Debug, Clone, PartialEq)]
pub struct VariantSummary {
    /// Variante resumida.
    pub variant: EvolutionVariantKind,
    /// Si la mutación estuvo habilitada en esta variante.
    pub mutation_enabled: bool,
    /// Tasa efectiva de mutación aplicada.
    pub effective_mutation_rate: f64,
    /// Amplitud efectiva de mutación aplicada.
    pub effective_mutation_scale: f64,
    /// Denominador explícito: total de semillas ejecutadas.
    pub total_seeds: usize,
    /// Denominador explícito: semillas con al menos 1 organismo vivo al final.
    pub surviving_seeds: usize,
    /// Denominador explícito: semillas extintas al final (`final_alive == 0`).
    pub extinct_seeds: usize,
    /// Tasa de supervivencia (`surviving_seeds / total_seeds`).
    pub survival_rate: f64,
    /// Tasa de extinción (`extinct_seeds / total_seeds`).
    pub extinction_rate: f64,
    /// Denominador explícito: suma total de nacimientos en todas las semillas.
    pub total_births: usize,
    /// Denominador explícito: suma total de muertes en todas las semillas.
    pub total_deaths: usize,
    /// Denominador explícito: total de organismos vivos al final entre todas las semillas.
    pub total_surviving_organisms: usize,
    /// Estadísticas de nacimientos por semilla (sobre `total_seeds`).
    pub births_per_seed: ScalarStats,
    /// Estadísticas de muertes por semilla (sobre `total_seeds`).
    pub deaths_per_seed: ScalarStats,
    /// Estadísticas de población final por semilla (sobre `total_seeds`).
    pub final_alive_per_seed: ScalarStats,
    /// Estadísticas de energía total final por semilla (sobre `total_seeds`, incluye `0.0` en extintas).
    pub final_total_energy_per_seed: ScalarStats,
    /// Estadísticas de energía media por organismo en cada semilla (sobre `total_seeds`).
    pub final_mean_organism_energy_per_seed: ScalarStats,
    /// Distribución de genes agregada sobre todos los organismos vivos finales (`sample_count = total_surviving_organisms`).
    pub gene_distribution: GenomeDistributionStats,
}

/// Traza completa del experimento evolutivo con metadatos, controles y trazas por semilla.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionTrace {
    /// Especificación completa con semillas y controles de mutación.
    pub spec: EvolutionExperimentSpec,
    /// Resúmenes por variante (principal + baselines).
    pub variants: Vec<VariantSummary>,
    /// Trazas individuales por `(variante, semilla)` sin filtrar extinciones.
    pub seed_traces: Vec<SeedEvolutionTrace>,
}

/// Comparación cuantitativa entre la variante evolutiva principal y un baseline explícito.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineComparison {
    /// Variante baseline comparada.
    pub baseline_variant: EvolutionVariantKind,
    /// Diferencia en tasa de supervivencia (`evolving - baseline`).
    pub survival_rate_delta: f64,
    /// Diferencia en nacimientos totales (`evolving - baseline`).
    pub total_births_delta: isize,
    /// Diferencia en energía total media final por semilla (`evolving - baseline`).
    pub mean_final_energy_delta: f64,
    /// Indica si la variante evolutiva iguala o supera en supervivencia y supera en energía o nacimientos.
    pub outperforms_baseline: bool,
}

/// Reporte completo del experimento evolutivo headless con comparación frente a baselines.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionExperimentReport {
    /// Versión de esquema del reporte.
    pub schema_version: &'static str,
    /// Nota epistémica obligatoria sobre no afirmar emergencia sin baselines.
    pub emergence_notice: String,
    /// `true` cuando existen comparaciones explícitas contra baselines.
    pub baselines_compared: bool,
    /// Especificación del experimento.
    pub spec: EvolutionExperimentSpec,
    /// Resumen de la variante principal (`Evolving`).
    pub primary_summary: VariantSummary,
    /// Resúmenes de cada baseline ejecutado.
    pub baseline_summaries: Vec<VariantSummary>,
    /// Comparaciones explícitas de la variante principal frente a cada baseline.
    pub baseline_comparisons: Vec<BaselineComparison>,
    /// Traza científica completa del experimento.
    pub trace: EvolutionTrace,
}

impl EvolutionExperimentReport {
    /// Exporta el reporte completo a JSON canónico sin requerir renderer ni Bevy.
    pub fn to_canonical_json(&self) -> Result<String, LifeError> {
        let mut out = String::new();
        write_report_json(&mut out, self)?;
        Ok(out)
    }

    /// Renderiza un resumen legible en texto plano para inspección headless.
    pub fn render_summary(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "=== Gogamen Life Evolution Experiment ({}) ===",
            self.spec.experiment_id
        );
        let _ = writeln!(
            out,
            "seeds={} ticks_per_seed={} mutation_enabled={} rate={:.4} scale={:.4}",
            self.primary_summary.total_seeds,
            self.spec.ticks_per_seed,
            self.primary_summary.mutation_enabled,
            self.primary_summary.effective_mutation_rate,
            self.primary_summary.effective_mutation_scale
        );
        let _ = writeln!(
            out,
            "primary [{}]: surviving={}/{} ({:.3}), extinct={}/{}, births={}, deaths={}, mean_energy={:.3}",
            self.primary_summary.variant,
            self.primary_summary.surviving_seeds,
            self.primary_summary.total_seeds,
            self.primary_summary.survival_rate,
            self.primary_summary.extinct_seeds,
            self.primary_summary.total_seeds,
            self.primary_summary.total_births,
            self.primary_summary.total_deaths,
            self.primary_summary.final_total_energy_per_seed.mean
        );
        for b in &self.baseline_summaries {
            let _ = writeln!(
                out,
                "baseline [{}]: surviving={}/{} ({:.3}), extinct={}/{}, births={}, deaths={}, mean_energy={:.3}",
                b.variant,
                b.surviving_seeds,
                b.total_seeds,
                b.survival_rate,
                b.extinct_seeds,
                b.total_seeds,
                b.total_births,
                b.total_deaths,
                b.final_total_energy_per_seed.mean
            );
        }
        let _ = writeln!(out, "notice: {}", self.emergence_notice);
        out
    }
}

/// Ejecuta el experimento evolutivo completo (variante principal + baselines) de forma determinista.
pub fn run_evolution_experiment(
    spec: &EvolutionExperimentSpec,
) -> Result<EvolutionExperimentReport, LifeError> {
    spec.validate()?;

    let mut all_seed_traces = Vec::new();
    let (primary_summary, primary_traces) = run_variant(spec, EvolutionVariantKind::Evolving)?;
    all_seed_traces.extend(primary_traces);

    let mut baseline_summaries = Vec::with_capacity(spec.baselines.len());
    let mut baseline_comparisons = Vec::with_capacity(spec.baselines.len());
    for &baseline_kind in &spec.baselines {
        let (summary, traces) = run_variant(spec, baseline_kind)?;
        let comparison = compare_against_baseline(&primary_summary, &summary);
        baseline_summaries.push(summary);
        baseline_comparisons.push(comparison);
        all_seed_traces.extend(traces);
    }

    let mut variants = Vec::with_capacity(1 + baseline_summaries.len());
    variants.push(primary_summary.clone());
    variants.extend(baseline_summaries.iter().cloned());

    let trace = EvolutionTrace {
        spec: spec.clone(),
        variants,
        seed_traces: all_seed_traces,
    };

    Ok(EvolutionExperimentReport {
        schema_version: EVOLUTION_EXPERIMENT_SCHEMA_VERSION,
        emergence_notice: NO_EMERGENCE_WITHOUT_BASELINES_NOTICE.to_owned(),
        baselines_compared: !baseline_comparisons.is_empty(),
        spec: spec.clone(),
        primary_summary,
        baseline_summaries,
        baseline_comparisons,
        trace,
    })
}

/// Ejecuta una única variante sobre todas las semillas de `spec` sin descartar extinciones.
pub fn run_evolution_variant(
    spec: &EvolutionExperimentSpec,
    variant: EvolutionVariantKind,
) -> Result<(VariantSummary, Vec<SeedEvolutionTrace>), LifeError> {
    spec.validate()?;
    run_variant(spec, variant)
}

fn run_variant(
    spec: &EvolutionExperimentSpec,
    variant: EvolutionVariantKind,
) -> Result<(VariantSummary, Vec<SeedEvolutionTrace>), LifeError> {
    let (mutation_enabled, effective_rate, effective_scale) = match variant {
        EvolutionVariantKind::Evolving => {
            let (r, s) = spec.mutation_control.effective_rate_and_scale();
            (spec.mutation_control.mutation_enabled, r, s)
        }
        EvolutionVariantKind::NoMutation
        | EvolutionVariantKind::RandomAgent
        | EvolutionVariantKind::FixedStay => (false, 0.0, 0.0),
    };

    let mut world_cfg = spec.world_config.clone();
    world_cfg.mutation_rate = effective_rate;
    world_cfg.mutation_scale = effective_scale;
    world_cfg.validate()?;

    let mut seed_traces = Vec::with_capacity(spec.seeds.len());
    let mut surviving_genomes = Vec::new();

    for &seed in &spec.seeds {
        let (trace, genomes) = execute_single_seed(
            &world_cfg,
            seed,
            spec.ticks_per_seed,
            variant,
            mutation_enabled,
            effective_rate,
            effective_scale,
        )?;
        surviving_genomes.extend(genomes);
        seed_traces.push(trace);
    }

    let summary = summarize_variant(
        variant,
        mutation_enabled,
        effective_rate,
        effective_scale,
        &seed_traces,
        &surviving_genomes,
    )?;
    Ok((summary, seed_traces))
}

fn execute_single_seed(
    world_cfg: &LifeWorldConfig,
    seed: MasterSeed,
    ticks_per_seed: u64,
    variant: EvolutionVariantKind,
    mutation_enabled: bool,
    effective_mutation_rate: f64,
    effective_mutation_scale: f64,
) -> Result<(SeedEvolutionTrace, Vec<crate::genetics::Genome>), LifeError> {
    let (trace, genomes, _world) = execute_single_seed_with_world(
        world_cfg,
        seed,
        ticks_per_seed,
        variant,
        mutation_enabled,
        effective_mutation_rate,
        effective_mutation_scale,
    )?;
    Ok((trace, genomes))
}

/// Runs one seed headless and also returns the final world (used by tests to inspect SoA bounds).
///
/// Dead organism records are compacted at the end of every tick so SoA memory and per-tick
/// cost are bounded by `max_population` rather than by the birth history. This changes
/// `final_trace_hash` relative to a run that keeps dead records, but stays deterministic.
fn execute_single_seed_with_world(
    world_cfg: &LifeWorldConfig,
    seed: MasterSeed,
    ticks_per_seed: u64,
    variant: EvolutionVariantKind,
    mutation_enabled: bool,
    effective_mutation_rate: f64,
    effective_mutation_scale: f64,
) -> Result<(SeedEvolutionTrace, Vec<crate::genetics::Genome>, LifeWorld), LifeError> {
    let mut world = LifeWorld::new(world_cfg.clone(), seed)?;
    world.set_auto_compact(true);
    let initial_population = world.alive_count();
    let mut total_births = 0usize;
    let mut total_deaths = 0usize;
    let mut extinction_tick = if initial_population == 0 {
        Some(0)
    } else {
        None
    };
    let mut tick_alive_counts = Vec::with_capacity(ticks_per_seed as usize);

    for _ in 0..ticks_per_seed {
        let report = world.step_tick_with(|obs| decide_action(obs, world_cfg, seed, variant));
        total_births += report.births;
        total_deaths += report.deaths;
        tick_alive_counts.push(report.alive_after);
        if report.alive_after == 0 && extinction_tick.is_none() {
            extinction_tick = Some(report.tick_completed.as_u64());
        }
    }

    let view = world.privileged_global_view();
    let orgs = view.organisms();
    let mut alive_energies = Vec::new();
    let mut alive_genomes = Vec::new();
    for i in 0..orgs.len() {
        if orgs.alive()[i] {
            alive_energies.push(orgs.energy()[i]);
            alive_genomes.push(orgs.genomes()[i]);
        }
    }

    let final_alive = alive_energies.len();
    let extinct = final_alive == 0;
    let final_total_energy: f64 = alive_energies.iter().sum();
    let final_organism_energy = ScalarStats::from_samples(&alive_energies)?;
    let final_genome_distribution = GenomeDistributionStats::from_genomes(&alive_genomes)?;
    let max_generation_reached = view
        .lineage()
        .iter()
        .map(|rec| rec.generation)
        .max()
        .unwrap_or(0);

    let trace = SeedEvolutionTrace {
        seed,
        variant,
        mutation_enabled,
        effective_mutation_rate,
        effective_mutation_scale,
        ticks_executed: ticks_per_seed,
        initial_population,
        final_alive,
        extinct,
        extinction_tick,
        total_births,
        total_deaths,
        final_total_energy,
        final_organism_energy,
        max_generation_reached,
        final_trace_hash: view.trace_hash(),
        energy_balance_discrepancy: view.energy_balance().discrepancy(),
        final_genome_distribution,
        tick_alive_counts,
    };

    Ok((trace, alive_genomes, world))
}

fn decide_action(
    obs: &LocalOrganismObservation,
    cfg: &LifeWorldConfig,
    seed: MasterSeed,
    variant: EvolutionVariantKind,
) -> LifeAction {
    match variant {
        EvolutionVariantKind::FixedStay => LifeAction::Stay,
        EvolutionVariantKind::RandomAgent => {
            let mut hasher = TraceHasher::new();
            hasher.write_u64(seed.as_u64());
            hasher.write_tick(obs.tick());
            hasher.write_u64(obs.organism_id());
            let h = hasher.finish();
            match h % 6 {
                0 => LifeAction::Stay,
                1 => LifeAction::MoveUp,
                2 => LifeAction::MoveDown,
                3 => LifeAction::MoveLeft,
                4 => LifeAction::MoveRight,
                _ => LifeAction::Reproduce,
            }
        }
        EvolutionVariantKind::Evolving | EvolutionVariantKind::NoMutation => {
            if obs.energy() >= cfg.reproduction_threshold
                && obs.energy() > cfg.reproduction_cost + cfg.basal_cost
            {
                return LifeAction::Reproduce;
            }
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
            let mut best_action = None;
            let mut best_score = 0.0;
            let bias = obs.genome().sensing_bias;
            for (idx, (action, dx, dy)) in candidates.iter().enumerate() {
                if let Some(cell) = obs.cell_at(*dx, *dy)
                    && cell.in_bounds
                    && cell.food > 0.0
                {
                    let tie_breaker = ((idx as f64) * 1e-6) * (1.0 + bias * 0.5);
                    let score = cell.food + tie_breaker;
                    if score > best_score {
                        best_score = score;
                        best_action = Some(*action);
                    }
                }
            }
            if let Some(action) = best_action {
                return action;
            }
            // Exploración determinista cuando no hay alimento adyacente visible.
            let mut hasher = TraceHasher::new();
            hasher.write_u64(seed.as_u64());
            hasher.write_tick(obs.tick());
            hasher.write_u64(obs.organism_id());
            hasher.write_f64(obs.genome().speed_gene);
            match hasher.finish() % 4 {
                0 => LifeAction::MoveUp,
                1 => LifeAction::MoveDown,
                2 => LifeAction::MoveLeft,
                _ => LifeAction::MoveRight,
            }
        }
    }
}

fn summarize_variant(
    variant: EvolutionVariantKind,
    mutation_enabled: bool,
    effective_mutation_rate: f64,
    effective_mutation_scale: f64,
    seed_traces: &[SeedEvolutionTrace],
    surviving_genomes: &[crate::genetics::Genome],
) -> Result<VariantSummary, LifeError> {
    let total_seeds = seed_traces.len();
    let extinct_seeds = seed_traces.iter().filter(|t| t.extinct).count();
    let surviving_seeds = total_seeds - extinct_seeds;
    let total_births: usize = seed_traces.iter().map(|t| t.total_births).sum();
    let total_deaths: usize = seed_traces.iter().map(|t| t.total_deaths).sum();
    let total_surviving_organisms: usize = seed_traces.iter().map(|t| t.final_alive).sum();

    let survival_rate = if total_seeds == 0 {
        0.0
    } else {
        surviving_seeds as f64 / total_seeds as f64
    };
    let extinction_rate = if total_seeds == 0 {
        0.0
    } else {
        extinct_seeds as f64 / total_seeds as f64
    };

    let births_samples: Vec<f64> = seed_traces.iter().map(|t| t.total_births as f64).collect();
    let deaths_samples: Vec<f64> = seed_traces.iter().map(|t| t.total_deaths as f64).collect();
    let alive_samples: Vec<f64> = seed_traces.iter().map(|t| t.final_alive as f64).collect();
    let total_energy_samples: Vec<f64> = seed_traces.iter().map(|t| t.final_total_energy).collect();
    let mean_energy_samples: Vec<f64> = seed_traces
        .iter()
        .map(|t| t.final_organism_energy.mean)
        .collect();

    Ok(VariantSummary {
        variant,
        mutation_enabled,
        effective_mutation_rate,
        effective_mutation_scale,
        total_seeds,
        surviving_seeds,
        extinct_seeds,
        survival_rate,
        extinction_rate,
        total_births,
        total_deaths,
        total_surviving_organisms,
        births_per_seed: ScalarStats::from_samples(&births_samples)?,
        deaths_per_seed: ScalarStats::from_samples(&deaths_samples)?,
        final_alive_per_seed: ScalarStats::from_samples(&alive_samples)?,
        final_total_energy_per_seed: ScalarStats::from_samples(&total_energy_samples)?,
        final_mean_organism_energy_per_seed: ScalarStats::from_samples(&mean_energy_samples)?,
        gene_distribution: GenomeDistributionStats::from_genomes(surviving_genomes)?,
    })
}

fn compare_against_baseline(
    primary: &VariantSummary,
    baseline: &VariantSummary,
) -> BaselineComparison {
    let survival_rate_delta = primary.survival_rate - baseline.survival_rate;
    let total_births_delta = primary.total_births as isize - baseline.total_births as isize;
    let mean_final_energy_delta =
        primary.final_total_energy_per_seed.mean - baseline.final_total_energy_per_seed.mean;
    let outperforms_baseline = primary.survival_rate >= baseline.survival_rate
        && (mean_final_energy_delta > 0.0 || total_births_delta > 0 || survival_rate_delta > 0.0);

    BaselineComparison {
        baseline_variant: baseline.variant,
        survival_rate_delta,
        total_births_delta,
        mean_final_energy_delta,
        outperforms_baseline,
    }
}

// ---------------------------------------------------------------------------
// Serialización JSON canónica en std puro (orden de claves fijo y determinista)
// ---------------------------------------------------------------------------

fn write_json_str(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_json_f64(out: &mut String, value: f64) -> Result<(), LifeError> {
    if !value.is_finite() {
        return Err(LifeError::InvalidConfig(
            "cannot serialize non-finite float to canonical JSON",
        ));
    }
    let normalized = if value == 0.0 { 0.0 } else { value };
    let formatted = format!("{normalized}");
    out.push_str(&formatted);
    if !formatted.contains('.') && !formatted.contains('e') && !formatted.contains('E') {
        out.push_str(".0");
    }
    Ok(())
}

fn write_spec_json(out: &mut String, spec: &EvolutionExperimentSpec) -> Result<(), LifeError> {
    out.push('{');
    out.push_str("\"baselines\":[");
    for (i, b) in spec.baselines.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_json_str(out, b.as_str());
    }
    out.push_str("],\"experiment_id\":");
    write_json_str(out, &spec.experiment_id);
    out.push_str(",\"mutation_control\":");
    write_mutation_control_json(out, &spec.mutation_control)?;
    out.push_str(",\"seeds\":[");
    for (i, s) in spec.seeds.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{}", s.as_u64());
    }
    let _ = write!(
        out,
        "],\"ticks_per_seed\":{},\"world_config\":",
        spec.ticks_per_seed
    );
    write_world_config_json(out, &spec.world_config)?;
    out.push('}');
    Ok(())
}

fn write_mutation_control_json(out: &mut String, mc: &MutationControl) -> Result<(), LifeError> {
    let (eff_rate, eff_scale) = mc.effective_rate_and_scale();
    out.push_str("{\"effective_mutation_rate\":");
    write_json_f64(out, eff_rate)?;
    out.push_str(",\"effective_mutation_scale\":");
    write_json_f64(out, eff_scale)?;
    let _ = write!(out, ",\"mutation_enabled\":{}", mc.mutation_enabled);
    out.push_str(",\"mutation_rate\":");
    write_json_f64(out, mc.mutation_rate)?;
    out.push_str(",\"mutation_scale\":");
    write_json_f64(out, mc.mutation_scale)?;
    out.push('}');
    Ok(())
}

fn write_world_config_json(out: &mut String, cfg: &LifeWorldConfig) -> Result<(), LifeError> {
    out.push_str("{\"basal_cost\":");
    write_json_f64(out, cfg.basal_cost)?;
    out.push_str(",\"food_regen_per_cell\":");
    write_json_f64(out, cfg.food_regen_per_cell)?;
    out.push_str(",\"food_regen_prob\":");
    write_json_f64(out, cfg.food_regen_prob)?;
    let _ = write!(out, ",\"height\":{}", cfg.height);
    out.push_str(",\"initial_cell_food\":");
    write_json_f64(out, cfg.initial_cell_food)?;
    out.push_str(",\"initial_energy\":");
    write_json_f64(out, cfg.initial_energy)?;
    let _ = write!(out, ",\"initial_population\":{}", cfg.initial_population);
    out.push_str(",\"max_cell_food\":");
    write_json_f64(out, cfg.max_cell_food)?;
    out.push_str(",\"max_eat_per_tick\":");
    write_json_f64(out, cfg.max_eat_per_tick)?;
    out.push_str(",\"max_energy\":");
    write_json_f64(out, cfg.max_energy)?;
    let _ = write!(out, ",\"max_population\":{}", cfg.max_population);
    out.push_str(",\"move_cost\":");
    write_json_f64(out, cfg.move_cost)?;
    out.push_str(",\"mutation_rate\":");
    write_json_f64(out, cfg.mutation_rate)?;
    out.push_str(",\"mutation_scale\":");
    write_json_f64(out, cfg.mutation_scale)?;
    out.push_str(",\"reproduction_cost\":");
    write_json_f64(out, cfg.reproduction_cost)?;
    out.push_str(",\"reproduction_overhead\":");
    write_json_f64(out, cfg.reproduction_overhead)?;
    out.push_str(",\"reproduction_threshold\":");
    write_json_f64(out, cfg.reproduction_threshold)?;
    let _ = write!(
        out,
        ",\"vision_radius\":{},\"width\":{}}}",
        cfg.vision_radius, cfg.width
    );
    Ok(())
}

fn write_scalar_stats_json(out: &mut String, stats: &ScalarStats) -> Result<(), LifeError> {
    let _ = write!(out, "{{\"count\":{},\"max\":", stats.count);
    write_json_f64(out, stats.max)?;
    out.push_str(",\"mean\":");
    write_json_f64(out, stats.mean)?;
    out.push_str(",\"min\":");
    write_json_f64(out, stats.min)?;
    out.push_str(",\"std_dev\":");
    write_json_f64(out, stats.std_dev)?;
    out.push_str(",\"variance\":");
    write_json_f64(out, stats.variance)?;
    out.push('}');
    Ok(())
}

fn write_gene_dist_json(out: &mut String, gd: &GeneDistributionStats) -> Result<(), LifeError> {
    out.push_str("{\"bounds_max\":");
    write_json_f64(out, gd.bounds_max)?;
    out.push_str(",\"bounds_min\":");
    write_json_f64(out, gd.bounds_min)?;
    out.push_str(",\"gene_name\":");
    write_json_str(out, &gd.gene_name);
    out.push_str(",\"stats\":");
    write_scalar_stats_json(out, &gd.stats)?;
    out.push('}');
    Ok(())
}

fn write_genome_dist_json(out: &mut String, g: &GenomeDistributionStats) -> Result<(), LifeError> {
    out.push_str("{\"metabolism_efficiency\":");
    write_gene_dist_json(out, &g.metabolism_efficiency)?;
    let _ = write!(
        out,
        ",\"sample_count\":{},\"sensing_bias\":",
        g.sample_count
    );
    write_gene_dist_json(out, &g.sensing_bias)?;
    out.push_str(",\"speed_gene\":");
    write_gene_dist_json(out, &g.speed_gene)?;
    out.push('}');
    Ok(())
}

fn write_variant_summary_json(out: &mut String, v: &VariantSummary) -> Result<(), LifeError> {
    out.push_str("{\"births_per_seed\":");
    write_scalar_stats_json(out, &v.births_per_seed)?;
    out.push_str(",\"deaths_per_seed\":");
    write_scalar_stats_json(out, &v.deaths_per_seed)?;
    out.push_str(",\"effective_mutation_rate\":");
    write_json_f64(out, v.effective_mutation_rate)?;
    out.push_str(",\"effective_mutation_scale\":");
    write_json_f64(out, v.effective_mutation_scale)?;
    let _ = write!(
        out,
        ",\"extinct_seeds\":{},\"extinction_rate\":",
        v.extinct_seeds
    );
    write_json_f64(out, v.extinction_rate)?;
    out.push_str(",\"final_alive_per_seed\":");
    write_scalar_stats_json(out, &v.final_alive_per_seed)?;
    out.push_str(",\"final_mean_organism_energy_per_seed\":");
    write_scalar_stats_json(out, &v.final_mean_organism_energy_per_seed)?;
    out.push_str(",\"final_total_energy_per_seed\":");
    write_scalar_stats_json(out, &v.final_total_energy_per_seed)?;
    out.push_str(",\"gene_distribution\":");
    write_genome_dist_json(out, &v.gene_distribution)?;
    let _ = write!(
        out,
        ",\"mutation_enabled\":{},\"survival_rate\":",
        v.mutation_enabled
    );
    write_json_f64(out, v.survival_rate)?;
    let _ = write!(
        out,
        ",\"surviving_seeds\":{},\"total_births\":{},\"total_deaths\":{},\"total_seeds\":{},\"total_surviving_organisms\":{},\"variant\":",
        v.surviving_seeds,
        v.total_births,
        v.total_deaths,
        v.total_seeds,
        v.total_surviving_organisms
    );
    write_json_str(out, v.variant.as_str());
    out.push('}');
    Ok(())
}

fn write_seed_trace_json(out: &mut String, t: &SeedEvolutionTrace) -> Result<(), LifeError> {
    out.push_str("{\"effective_mutation_rate\":");
    write_json_f64(out, t.effective_mutation_rate)?;
    out.push_str(",\"effective_mutation_scale\":");
    write_json_f64(out, t.effective_mutation_scale)?;
    out.push_str(",\"energy_balance_discrepancy\":");
    write_json_f64(out, t.energy_balance_discrepancy)?;
    let _ = write!(out, ",\"extinct\":{},\"extinction_tick\":", t.extinct);
    match t.extinction_tick {
        Some(tick) => {
            let _ = write!(out, "{tick}");
        }
        None => out.push_str("null"),
    }
    let _ = write!(
        out,
        ",\"final_alive\":{},\"final_genome_distribution\":",
        t.final_alive
    );
    write_genome_dist_json(out, &t.final_genome_distribution)?;
    out.push_str(",\"final_organism_energy\":");
    write_scalar_stats_json(out, &t.final_organism_energy)?;
    out.push_str(",\"final_total_energy\":");
    write_json_f64(out, t.final_total_energy)?;
    let _ = write!(
        out,
        ",\"final_trace_hash\":{},\"initial_population\":{},\"max_generation_reached\":{},\"mutation_enabled\":{},\"seed\":{},\"tick_alive_counts\":[",
        t.final_trace_hash,
        t.initial_population,
        t.max_generation_reached,
        t.mutation_enabled,
        t.seed.as_u64()
    );
    for (i, alive) in t.tick_alive_counts.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(out, "{alive}");
    }
    let _ = write!(
        out,
        "],\"ticks_executed\":{},\"total_births\":{},\"total_deaths\":{},\"variant\":",
        t.ticks_executed, t.total_births, t.total_deaths
    );
    write_json_str(out, t.variant.as_str());
    out.push('}');
    Ok(())
}

fn write_baseline_comparison_json(
    out: &mut String,
    c: &BaselineComparison,
) -> Result<(), LifeError> {
    out.push_str("{\"baseline_variant\":");
    write_json_str(out, c.baseline_variant.as_str());
    out.push_str(",\"mean_final_energy_delta\":");
    write_json_f64(out, c.mean_final_energy_delta)?;
    let _ = write!(
        out,
        ",\"outperforms_baseline\":{},\"survival_rate_delta\":",
        c.outperforms_baseline
    );
    write_json_f64(out, c.survival_rate_delta)?;
    let _ = write!(out, ",\"total_births_delta\":{}}}", c.total_births_delta);
    Ok(())
}

fn write_report_json(
    out: &mut String,
    report: &EvolutionExperimentReport,
) -> Result<(), LifeError> {
    out.push_str("{\"baseline_comparisons\":[");
    for (i, c) in report.baseline_comparisons.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_baseline_comparison_json(out, c)?;
    }
    out.push_str("],\"baseline_summaries\":[");
    for (i, b) in report.baseline_summaries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_variant_summary_json(out, b)?;
    }
    let _ = write!(
        out,
        "],\"baselines_compared\":{},\"emergence_notice\":",
        report.baselines_compared
    );
    write_json_str(out, &report.emergence_notice);
    out.push_str(",\"primary_summary\":");
    write_variant_summary_json(out, &report.primary_summary)?;
    out.push_str(",\"schema_version\":");
    write_json_str(out, report.schema_version);
    out.push_str(",\"spec\":");
    write_spec_json(out, &report.spec)?;
    out.push_str(",\"trace\":{\"seed_traces\":[");
    for (i, st) in report.trace.seed_traces.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_seed_trace_json(out, st)?;
    }
    out.push_str("],\"spec\":");
    write_spec_json(out, &report.trace.spec)?;
    out.push_str(",\"variants\":[");
    for (i, v) in report.trace.variants.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_variant_summary_json(out, v)?;
    }
    out.push_str("]}}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_world_config() -> LifeWorldConfig {
        LifeWorldConfig {
            width: 12,
            height: 12,
            initial_population: 6,
            max_population: 24,
            initial_energy: 10.0,
            max_energy: 20.0,
            basal_cost: 0.45,
            move_cost: 0.2,
            max_eat_per_tick: 4.0,
            max_cell_food: 10.0,
            initial_cell_food: 6.0,
            food_regen_per_cell: 0.6,
            food_regen_prob: 0.35,
            vision_radius: 2,
            reproduction_threshold: 11.0,
            reproduction_cost: 5.5,
            reproduction_overhead: 0.5,
            mutation_rate: 0.35,
            mutation_scale: 0.2,
        }
    }

    #[test]
    fn long_experiment_bounds_soa_by_max_population_and_is_deterministic() {
        let mut cfg = test_world_config();
        cfg.basal_cost = 1.0;
        cfg.reproduction_threshold = 3.0;
        cfg.reproduction_cost = 2.5;
        cfg.initial_energy = 6.0;
        cfg.max_population = 16;
        for variant in [
            EvolutionVariantKind::RandomAgent,
            EvolutionVariantKind::FixedStay,
        ] {
            for seed in [1u64, 2, 3] {
                let run = || {
                    execute_single_seed_with_world(
                        &cfg,
                        MasterSeed::new(seed),
                        300,
                        variant,
                        true,
                        0.35,
                        0.2,
                    )
                    .unwrap()
                };
                let (trace_a, _, world_a) = run();
                let (trace_b, _, _) = run();
                assert_eq!(trace_a.final_trace_hash, trace_b.final_trace_hash);
                assert_eq!(trace_a, trace_b);
                assert!(trace_a.tick_alive_counts.iter().all(|&n| n <= 16));
                let soa = world_a.privileged_global_view().organisms();
                assert!(soa.len() <= cfg.max_population);
                assert!(soa.capacity() <= 4 * cfg.max_population);
                if variant == EvolutionVariantKind::RandomAgent {
                    // High turnover: far more births than the population bound.
                    assert!(world_a.lineage().len() > 2 * cfg.max_population);
                }
            }
        }
    }

    #[test]
    fn ac1_multi_seed_reproducible_with_explicit_denominators_and_gene_distribution() {
        let seeds = vec![
            MasterSeed::new(101),
            MasterSeed::new(202),
            MasterSeed::new(303),
            MasterSeed::new(404),
        ];
        let spec = EvolutionExperimentSpec::new(
            "life_evo_multi_seed_v1",
            test_world_config(),
            40,
            seeds.clone(),
            MutationControl::enabled(0.35, 0.2),
        );

        let report_a = run_evolution_experiment(&spec).unwrap();
        let report_b = run_evolution_experiment(&spec).unwrap();
        assert_eq!(report_a, report_b);

        let p = &report_a.primary_summary;
        assert_eq!(p.total_seeds, seeds.len());
        assert_eq!(p.surviving_seeds + p.extinct_seeds, p.total_seeds);
        assert!(
            (p.survival_rate + p.extinction_rate - 1.0).abs() < 1e-12,
            "rates must sum to 1.0"
        );
        assert_eq!(p.births_per_seed.count, p.total_seeds);
        assert_eq!(p.deaths_per_seed.count, p.total_seeds);
        assert_eq!(p.final_alive_per_seed.count, p.total_seeds);
        assert_eq!(p.final_total_energy_per_seed.count, p.total_seeds);
        assert_eq!(p.final_mean_organism_energy_per_seed.count, p.total_seeds);
        assert!(p.total_births > 0);

        // Distribución de genes con denominador explícito = organismos supervivientes totales.
        assert_eq!(
            p.gene_distribution.sample_count,
            p.total_surviving_organisms
        );
        assert_eq!(
            p.gene_distribution.speed_gene.stats.count,
            p.total_surviving_organisms
        );
        assert_eq!(
            p.gene_distribution.metabolism_efficiency.stats.count,
            p.total_surviving_organisms
        );
        assert_eq!(
            p.gene_distribution.sensing_bias.stats.count,
            p.total_surviving_organisms
        );
        assert!(p.gene_distribution.speed_gene.stats.std_dev >= 0.0);
        assert!(
            p.gene_distribution.speed_gene.stats.min >= SPEED_GENE_BOUNDS.min
                && p.gene_distribution.speed_gene.stats.max <= SPEED_GENE_BOUNDS.max
        );
    }

    #[test]
    fn ac2_mutation_controls_and_seeds_recorded_without_dropping_extinctions() {
        // Configuración austera para garantizar extinción en todas las semillas.
        let harsh_cfg = LifeWorldConfig {
            width: 8,
            height: 8,
            initial_population: 4,
            max_population: 12,
            initial_energy: 3.0,
            max_energy: 10.0,
            basal_cost: 1.5,
            move_cost: 0.5,
            max_eat_per_tick: 0.0,
            max_cell_food: 0.0,
            initial_cell_food: 0.0,
            food_regen_per_cell: 0.0,
            food_regen_prob: 0.0,
            vision_radius: 1,
            reproduction_threshold: 8.0,
            reproduction_cost: 4.0,
            reproduction_overhead: 0.5,
            mutation_rate: 0.5,
            mutation_scale: 0.25,
        };
        let seeds = vec![
            MasterSeed::new(11),
            MasterSeed::new(22),
            MasterSeed::new(33),
        ];
        let spec = EvolutionExperimentSpec::new(
            "life_evo_extinction_retention",
            harsh_cfg,
            10,
            seeds.clone(),
            MutationControl::disabled(),
        );

        let report = run_evolution_experiment(&spec).unwrap();
        assert_eq!(report.trace.spec.seeds, seeds);
        assert!(!report.trace.spec.mutation_control.mutation_enabled);
        assert_eq!(
            report
                .trace
                .spec
                .mutation_control
                .effective_rate_and_scale(),
            (0.0, 0.0)
        );

        let p = &report.primary_summary;
        assert!(!p.mutation_enabled);
        assert_eq!(p.effective_mutation_rate, 0.0);
        assert_eq!(p.effective_mutation_scale, 0.0);
        // Ninguna semilla extinta se descarta del denominador ni de la traza.
        assert_eq!(p.total_seeds, 3);
        assert_eq!(p.extinct_seeds, 3);
        assert_eq!(p.surviving_seeds, 0);
        assert!((p.extinction_rate - 1.0).abs() < 1e-12);
        assert!((p.survival_rate - 0.0).abs() < 1e-12);
        assert_eq!(p.births_per_seed.count, 3);
        assert_eq!(p.final_total_energy_per_seed.count, 3);
        assert_eq!(p.total_surviving_organisms, 0);
        assert_eq!(p.gene_distribution.sample_count, 0);

        let primary_traces: Vec<&SeedEvolutionTrace> = report
            .trace
            .seed_traces
            .iter()
            .filter(|t| t.variant == EvolutionVariantKind::Evolving)
            .collect();
        assert_eq!(primary_traces.len(), 3);
        for t in primary_traces {
            assert!(t.extinct);
            assert!(t.extinction_tick.is_some());
            assert_eq!(t.final_alive, 0);
            assert!(!t.mutation_enabled);
            assert!(t.energy_balance_discrepancy < 1e-9);
        }
    }

    #[test]
    fn ac3_compares_against_baselines_includes_notice_and_exports_canonical_json() {
        let seeds = vec![MasterSeed::new(7), MasterSeed::new(8), MasterSeed::new(9)];
        let spec = EvolutionExperimentSpec::new(
            "life_evo_baselines_json",
            test_world_config(),
            35,
            seeds,
            MutationControl::enabled(0.4, 0.25),
        );

        let report = run_evolution_experiment(&spec).unwrap();
        assert!(report.baselines_compared);
        assert_eq!(
            report.emergence_notice,
            NO_EMERGENCE_WITHOUT_BASELINES_NOTICE
        );
        assert_eq!(report.baseline_summaries.len(), 3);
        assert_eq!(report.baseline_comparisons.len(), 3);

        // El baseline NoMutation mantiene varianza genética 0 si los fundadores son idénticos.
        let no_mut_summary = report
            .baseline_summaries
            .iter()
            .find(|b| b.variant == EvolutionVariantKind::NoMutation)
            .expect("NoMutation baseline must be present");
        assert!(!no_mut_summary.mutation_enabled);
        assert!((no_mut_summary.gene_distribution.speed_gene.stats.std_dev - 0.0).abs() < 1e-12);

        // Exportación JSON canónica determinista sin renderer ni Bevy.
        let json_a = report.to_canonical_json().unwrap();
        let json_b = report.to_canonical_json().unwrap();
        assert_eq!(json_a, json_b);
        assert!(json_a.contains(NO_EMERGENCE_WITHOUT_BASELINES_NOTICE));
        assert!(json_a.contains("\"total_seeds\":3"));
        assert!(json_a.contains("\"extinct_seeds\":"));
        assert!(json_a.contains("\"surviving_seeds\":"));
        assert!(json_a.contains("\"baseline_no_mutation\""));
        assert!(json_a.contains("\"baseline_random_agent\""));
        assert!(json_a.contains("\"baseline_fixed_stay\""));
        assert!(json_a.contains("\"mutation_enabled\":true"));

        let summary_text = report.render_summary();
        assert!(summary_text.contains(NO_EMERGENCE_WITHOUT_BASELINES_NOTICE));
    }
}
