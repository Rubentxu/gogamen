//! Estadísticas de dispersión, clasificación explícita de fallos y agregados pareados.

use gogamen_lab_grid::SeedSplit;

use crate::runner::{PairedSeedOutcome, PlannerRunResult};

/// Estado explícito de finalización de una ejecución individual en la arena.
///
/// Todos los estados no exitosos ([`RunOutcomeStatus::Failed`],
/// [`RunOutcomeStatus::TimedOut`] y [`RunOutcomeStatus::BudgetExhausted`])
/// permanecen contabilizados en el denominador total sin filtrarse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunOutcomeStatus {
    /// La ejecución alcanzó el objetivo del escenario dentro del presupuesto.
    Succeeded,
    /// La ejecución terminó en fallo explícito (colisión fatal, violación de invariante o error).
    Failed {
        /// Motivo legible del fallo.
        reason: String,
    },
    /// La ejecución agotó el límite de pasos o el timeout de pared configurado sin éxito.
    TimedOut {
        /// Motivo legible del timeout.
        reason: String,
    },
    /// La ejecución agotó el presupuesto de transiciones/simulaciones sin éxito.
    BudgetExhausted {
        /// Motivo legible del agotamiento de presupuesto.
        reason: String,
    },
}

impl RunOutcomeStatus {
    /// Indica si la ejecución fue exitosa.
    pub const fn is_succeeded(&self) -> bool {
        matches!(self, Self::Succeeded)
    }

    /// Indica si la ejecución terminó en [`RunOutcomeStatus::Failed`].
    pub const fn is_failed(&self) -> bool {
        matches!(self, Self::Failed { .. })
    }

    /// Indica si la ejecución terminó en [`RunOutcomeStatus::TimedOut`].
    pub const fn is_timed_out(&self) -> bool {
        matches!(self, Self::TimedOut { .. })
    }

    /// Indica si la ejecución terminó en [`RunOutcomeStatus::BudgetExhausted`].
    pub const fn is_budget_exhausted(&self) -> bool {
        matches!(self, Self::BudgetExhausted { .. })
    }

    /// Devuelve el motivo explícito cuando la ejecución no tuvo éxito.
    pub fn failure_reason(&self) -> Option<&str> {
        match self {
            Self::Succeeded => None,
            Self::Failed { reason }
            | Self::TimedOut { reason }
            | Self::BudgetExhausted { reason } => Some(reason.as_str()),
        }
    }

    /// Etiqueta corta del estado para reportes y trazas.
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Succeeded => "Succeeded",
            Self::Failed { .. } => "Failed",
            Self::TimedOut { .. } => "TimedOut",
            Self::BudgetExhausted { .. } => "BudgetExhausted",
        }
    }
}

/// Ganador de una comparación pareada sobre una misma semilla.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairedWinner {
    /// El planificador izquierdo obtuvo mejor resultado pareado.
    Left,
    /// El planificador derecho obtuvo mejor resultado pareado.
    Right,
    /// Empate técnico entre ambos planificadores.
    Tie,
}

/// Métricas de dispersión estadística sobre un conjunto completo de ejecuciones.
#[derive(Debug, Clone, PartialEq)]
pub struct DispersionStats {
    /// Número total de muestras incluidas (denominador intacto).
    pub count: usize,
    /// Media aritmética.
    pub mean: f64,
    /// Desviación estándar poblacional (`0.0` si `count == 0`).
    pub std_dev: f64,
    /// Valor mínimo (`0.0` si `count == 0`).
    pub min: f64,
    /// Cuartil 25 (`Q25`).
    pub q25: f64,
    /// Mediana (`Q50`).
    pub median: f64,
    /// Cuartil 75 (`Q75`).
    pub q75: f64,
    /// Rango intercuartílico (`Q75 - Q25`).
    pub iqr: f64,
    /// Valor máximo (`0.0` si `count == 0`).
    pub max: f64,
}

impl DispersionStats {
    /// Calcula media, desviación estándar, cuartiles, IQR y extremos de `values`.
    pub fn compute(values: &[f64]) -> Self {
        let count = values.len();
        if count == 0 {
            return Self {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                q25: 0.0,
                median: 0.0,
                q75: 0.0,
                iqr: 0.0,
                max: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);

        let mean = sorted.iter().sum::<f64>() / (count as f64);
        let variance = sorted
            .iter()
            .map(|v| {
                let d = *v - mean;
                d * d
            })
            .sum::<f64>()
            / (count as f64);
        let std_dev = variance.sqrt();

        let min = sorted[0];
        let max = sorted[count - 1];
        let q25 = percentile_sorted(&sorted, 0.25);
        let median = percentile_sorted(&sorted, 0.50);
        let q75 = percentile_sorted(&sorted, 0.75);
        let iqr = q75 - q25;

        Self {
            count,
            mean,
            std_dev,
            min,
            q25,
            median,
            q75,
            iqr,
            max,
        }
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p * ((sorted.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Resumen agregado de un planificador sobre un conjunto de semillas (global, `Dev` o `Holdout`).
#[derive(Debug, Clone, PartialEq)]
pub struct PlannerSummary {
    /// Identificador del planificador.
    pub planner_id: String,
    /// Marca explícita de acceso privilegiado completo.
    pub is_privileged_full_info: bool,
    /// `None` para resumen global; `Some(split)` para `Dev` o `Holdout`.
    pub split: Option<SeedSplit>,
    /// Denominador total intacto (`succeeded + failed + timed_out + budget_exhausted`).
    pub total_runs: usize,
    /// Conteo de ejecuciones exitosas.
    pub succeeded_count: usize,
    /// Conteo de ejecuciones en estado `Failed`.
    pub failed_count: usize,
    /// Conteo de ejecuciones en estado `TimedOut`.
    pub timed_out_count: usize,
    /// Conteo de ejecuciones en estado `BudgetExhausted`.
    pub budget_exhausted_count: usize,
    /// Tasa de éxito sobre el denominador total intacto (`succeeded_count / total_runs`).
    pub success_rate: f64,
    /// Dispersión de pasos ejecutados.
    pub steps_dispersion: DispersionStats,
    /// Dispersión de transiciones evaluadas.
    pub transitions_dispersion: DispersionStats,
    /// Dispersión de recompensa total acumulada.
    pub reward_dispersion: DispersionStats,
    /// Utilización media de presupuesto de pasos en `[0.0, 1.0+]`.
    pub mean_budget_utilization_steps: f64,
    /// Utilización media de presupuesto de transiciones en `[0.0, 1.0+]`.
    pub mean_budget_utilization_transitions: f64,
    /// Latencia media de ejecución en nanosegundos.
    pub mean_latency_ns: f64,
    /// Registro explícito de cada fallo, timeout o agotamiento de presupuesto observado.
    pub failure_notes: Vec<String>,
}

/// Resumen agregado de diferencias pareadas entre ambos planificadores.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedComparisonSummary {
    /// `None` para resumen global; `Some(split)` para `Dev` o `Holdout`.
    pub split: Option<SeedSplit>,
    /// Total de pares evaluados (denominador intacto).
    pub total_pairs: usize,
    /// Pares donde ganó el planificador izquierdo.
    pub left_wins: usize,
    /// Pares donde ganó el planificador derecho.
    pub right_wins: usize,
    /// Pares empatados.
    pub ties: usize,
    /// Pares donde ambos planificadores tuvieron éxito.
    pub both_succeeded: usize,
    /// Pares donde solo el izquierdo tuvo éxito.
    pub left_only_succeeded: usize,
    /// Pares donde solo el derecho tuvo éxito.
    pub right_only_succeeded: usize,
    /// Pares donde ninguno tuvo éxito.
    pub both_non_success: usize,
    /// Dispersión de `left.steps - right.steps`.
    pub steps_delta_dispersion: DispersionStats,
    /// Dispersión de `left.transitions - right.transitions`.
    pub transitions_delta_dispersion: DispersionStats,
    /// Dispersión de `left.reward - right.reward`.
    pub reward_delta_dispersion: DispersionStats,
}

/// Construye el resumen agregado de un planificador preservando siempre el denominador total.
pub fn summarize_planner_runs(
    planner_id: &str,
    is_privileged_full_info: bool,
    split: Option<SeedSplit>,
    runs: &[&PlannerRunResult],
) -> PlannerSummary {
    let total_runs = runs.len();
    let succeeded_count = runs.iter().filter(|r| r.status.is_succeeded()).count();
    let failed_count = runs.iter().filter(|r| r.status.is_failed()).count();
    let timed_out_count = runs.iter().filter(|r| r.status.is_timed_out()).count();
    let budget_exhausted_count = runs
        .iter()
        .filter(|r| r.status.is_budget_exhausted())
        .count();

    let success_rate = if total_runs == 0 {
        0.0
    } else {
        (succeeded_count as f64) / (total_runs as f64)
    };

    let steps: Vec<f64> = runs.iter().map(|r| r.steps_executed as f64).collect();
    let transitions: Vec<f64> = runs
        .iter()
        .map(|r| r.transitions_evaluated as f64)
        .collect();
    let rewards: Vec<f64> = runs.iter().map(|r| r.total_reward).collect();

    let mean_budget_utilization_steps = if total_runs == 0 {
        0.0
    } else {
        runs.iter().map(|r| r.budget_utilization_steps).sum::<f64>() / (total_runs as f64)
    };
    let mean_budget_utilization_transitions = if total_runs == 0 {
        0.0
    } else {
        runs.iter()
            .map(|r| r.budget_utilization_transitions)
            .sum::<f64>()
            / (total_runs as f64)
    };
    let mean_latency_ns = if total_runs == 0 {
        0.0
    } else {
        (runs.iter().map(|r| r.latency_ns).sum::<u128>() as f64) / (total_runs as f64)
    };

    let failure_notes = runs
        .iter()
        .filter_map(|r| {
            r.status.failure_reason().map(|reason| {
                format!(
                    "{} {:?} seed={} [{}]: {} (steps={}, transitions={})",
                    planner_id,
                    r.split,
                    r.seed.as_u64(),
                    r.status.label(),
                    reason,
                    r.steps_executed,
                    r.transitions_evaluated,
                )
            })
        })
        .collect();

    PlannerSummary {
        planner_id: planner_id.to_owned(),
        is_privileged_full_info,
        split,
        total_runs,
        succeeded_count,
        failed_count,
        timed_out_count,
        budget_exhausted_count,
        success_rate,
        steps_dispersion: DispersionStats::compute(&steps),
        transitions_dispersion: DispersionStats::compute(&transitions),
        reward_dispersion: DispersionStats::compute(&rewards),
        mean_budget_utilization_steps,
        mean_budget_utilization_transitions,
        mean_latency_ns,
        failure_notes,
    }
}

/// Construye el resumen de diferencias pareadas para un conjunto de [`PairedSeedOutcome`].
pub fn summarize_paired_outcomes(
    split: Option<SeedSplit>,
    pairs: &[&PairedSeedOutcome],
) -> PairedComparisonSummary {
    let total_pairs = pairs.len();
    let mut left_wins = 0_usize;
    let mut right_wins = 0_usize;
    let mut ties = 0_usize;
    let mut both_succeeded = 0_usize;
    let mut left_only_succeeded = 0_usize;
    let mut right_only_succeeded = 0_usize;
    let mut both_non_success = 0_usize;

    let mut steps_deltas = Vec::with_capacity(total_pairs);
    let mut transitions_deltas = Vec::with_capacity(total_pairs);
    let mut reward_deltas = Vec::with_capacity(total_pairs);

    for pair in pairs {
        match pair.outcome_winner {
            PairedWinner::Left => left_wins += 1,
            PairedWinner::Right => right_wins += 1,
            PairedWinner::Tie => ties += 1,
        }
        match (pair.left.succeeded, pair.right.succeeded) {
            (true, true) => both_succeeded += 1,
            (true, false) => left_only_succeeded += 1,
            (false, true) => right_only_succeeded += 1,
            (false, false) => both_non_success += 1,
        }
        steps_deltas.push(pair.steps_delta as f64);
        transitions_deltas.push(pair.transitions_delta as f64);
        reward_deltas.push(pair.reward_delta);
    }

    PairedComparisonSummary {
        split,
        total_pairs,
        left_wins,
        right_wins,
        ties,
        both_succeeded,
        left_only_succeeded,
        right_only_succeeded,
        both_non_success,
        steps_delta_dispersion: DispersionStats::compute(&steps_deltas),
        transitions_delta_dispersion: DispersionStats::compute(&transitions_deltas),
        reward_delta_dispersion: DispersionStats::compute(&reward_deltas),
    }
}
