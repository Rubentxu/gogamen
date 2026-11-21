//! Oráculos de información de Shannon y estimadores empíricos de entropía sobre
//! trayectorias discretizadas por bins (`std`-only).
//!
//! # Alcance epistemológico
//! - [`shannon_entropy_bits`] implementa la entropía discreta exacta de Shannon
//!   `H(P) = -sum_i p_i log2(p_i)` en **bits**, validando finitud, no-negatividad y
//!   normalización de la distribución.
//! - [`estimate_trajectory_entropy_proxy`] (junto con [`EntropyProxyKind`] y
//!   [`TrajectoryEntropyEstimate`]) calcula un **proxy empírico de entropía**
//!   sobre trayectorias o estados finales agrupados en bins explícitos ([`BinSpec`],
//!   [`BinningVersion`]).
//! - Este estimador **no** pretende medir entropía causal física ni entropía
//!   diferencial continua exacta: cualquier interpretación más allá de un proxy
//!   empírico discretizado por bins requiere demostrar formalmente la equivalencia
//!   bajo el esquema de discretización elegido.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::domain::State;
use crate::units::Tick;

/// Tolerancia por defecto para validar que una distribución de probabilidad suma `1.0`.
pub const DEFAULT_PROBABILITY_SUM_TOLERANCE: f64 = 1e-9;

/// Errores explícitos al validar distribuciones, especificaciones de bins o estimar entropía.
#[derive(Debug, Clone, PartialEq)]
pub enum EntropyError {
    /// La distribución de probabilidad está vacía.
    EmptyDistribution,
    /// Una probabilidad no es finita (`NaN`, `+inf` o `-inf`).
    NonFiniteProbability {
        /// Índice de la entrada inválida en el slice.
        index: usize,
        /// Valor no finito recibido.
        value: f64,
    },
    /// Una probabilidad es estrictamente negativa (`< 0.0`).
    NegativeProbability {
        /// Índice de la entrada negativa en el slice.
        index: usize,
        /// Valor negativo recibido.
        value: f64,
    },
    /// La suma de probabilidades se desvía de `1.0` más allá de la tolerancia permitida.
    ProbabilitySumMismatch {
        /// Suma observada de las probabilidades.
        sum: f64,
        /// Tolerancia máxima permitida respecto a `1.0`.
        tolerance: f64,
    },
    /// La tolerancia proporcionada es negativa o no finita.
    InvalidTolerance {
        /// Valor de tolerancia recibido.
        value: f64,
    },
    /// La especificación de bins es inválida.
    InvalidBinSpec {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// No se proporcionaron trayectorias para estimar la distribución empírica.
    EmptyTrajectories,
    /// Una trayectoria no contiene ningún paso.
    EmptyTrajectory {
        /// Índice de la trayectoria vacía.
        trajectory_index: usize,
    },
    /// Un valor de estado utilizado para discretización no es finito.
    NonFiniteStateValue {
        /// Campo o contexto donde se detectó el valor no finito.
        field: &'static str,
        /// Valor inválido recibido.
        value: f64,
    },
    /// La dimensionalidad de un estado no coincide con la de [`BinSpec`].
    DimensionMismatch {
        /// Dimensionalidad esperada por [`BinSpec`].
        expected: usize,
        /// Dimensionalidad observada en el estado.
        actual: usize,
    },
}

impl fmt::Display for EntropyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyDistribution => write!(f, "probability distribution cannot be empty"),
            Self::NonFiniteProbability { index, value } => {
                write!(f, "non-finite probability at index {index}: {value}")
            }
            Self::NegativeProbability { index, value } => {
                write!(f, "negative probability at index {index}: {value}")
            }
            Self::ProbabilitySumMismatch { sum, tolerance } => {
                write!(
                    f,
                    "probability sum mismatch: sum {sum} differs from 1.0 by more than {tolerance}"
                )
            }
            Self::InvalidTolerance { value } => {
                write!(
                    f,
                    "invalid probability sum tolerance ({value}): must be finite and >= 0.0"
                )
            }
            Self::InvalidBinSpec { reason } => write!(f, "invalid bin specification: {reason}"),
            Self::EmptyTrajectories => {
                write!(
                    f,
                    "cannot estimate empirical entropy from an empty trajectory set"
                )
            }
            Self::EmptyTrajectory { trajectory_index } => {
                write!(f, "trajectory at index {trajectory_index} has no steps")
            }
            Self::NonFiniteStateValue { field, value } => {
                write!(f, "non-finite state value in '{field}': {value}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "state dimension mismatch for bin specification: expected {expected}, got {actual}"
                )
            }
        }
    }
}

impl Error for EntropyError {}

/// Calcula la entropía discreta de Shannon `H(P) = -sum_i p_i * log2(p_i)` en **bits**
/// usando [`DEFAULT_PROBABILITY_SUM_TOLERANCE`].
///
/// Aplica la convención estándar `0 * log2(0) = 0`.
pub fn shannon_entropy_bits(probs: &[f64]) -> Result<f64, EntropyError> {
    shannon_entropy_bits_with_tolerance(probs, DEFAULT_PROBABILITY_SUM_TOLERANCE)
}

/// Calcula la entropía discreta de Shannon en **bits** validando la suma con una tolerancia
/// explícita.
pub fn shannon_entropy_bits_with_tolerance(
    probs: &[f64],
    tolerance: f64,
) -> Result<f64, EntropyError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(EntropyError::InvalidTolerance { value: tolerance });
    }
    if probs.is_empty() {
        return Err(EntropyError::EmptyDistribution);
    }

    let mut sum = 0.0_f64;
    let mut entropy = 0.0_f64;

    for (index, &p) in probs.iter().enumerate() {
        if !p.is_finite() {
            return Err(EntropyError::NonFiniteProbability { index, value: p });
        }
        if p < 0.0 {
            return Err(EntropyError::NegativeProbability { index, value: p });
        }
        sum += p;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    if (sum - 1.0).abs() > tolerance {
        return Err(EntropyError::ProbabilitySumMismatch { sum, tolerance });
    }

    // Evita devolver `-0.0` en distribuciones deterministas `[1.0, 0.0, ...]`.
    if entropy == 0.0 { Ok(0.0) } else { Ok(entropy) }
}

/// Versión explícita del esquema de discretización por bins registrado en cada estimación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BinningVersion {
    major: u32,
    minor: u32,
}

impl BinningVersion {
    /// Versión inicial estándar `v1.0`.
    pub const V1_0: Self = Self::new(1, 0);

    /// Construye una versión de esquema de bins `major.minor`.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Devuelve el componente mayor de la versión.
    pub const fn major(self) -> u32 {
        self.major
    }

    /// Devuelve el componente menor de la versión.
    pub const fn minor(self) -> u32 {
        self.minor
    }
}

impl fmt::Display for BinningVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

/// Especificación de discretización por bins para un eje escalar del estado.
///
/// Los valores por debajo del límite inferior se asignan al primer bin (`0`) y los
/// valores por encima o iguales al límite superior se asignan al último bin (`num_bins - 1`).
#[derive(Debug, Clone, PartialEq)]
pub enum BinAxisSpec {
    /// Partición uniforme del intervalo `[min, max]` en `num_bins` intervalos de igual ancho.
    Uniform {
        /// Límite inferior del rango cubierto.
        min: f64,
        /// Límite superior del rango cubierto (`> min`).
        max: f64,
        /// Número de bins (`>= 1`).
        num_bins: usize,
    },
    /// Partición definida por fronteras explícitas estrictamente crecientes `[e_0, e_1, ..., e_k]`.
    ExplicitEdges {
        /// Fronteras de los `k` bins (longitud `k + 1 >= 2`).
        edges: Vec<f64>,
    },
}

impl BinAxisSpec {
    /// Construye un eje uniforme validando finitud, orden `min < max` y `num_bins >= 1`.
    pub fn uniform(min: f64, max: f64, num_bins: usize) -> Result<Self, EntropyError> {
        if !min.is_finite() || !max.is_finite() {
            return Err(EntropyError::InvalidBinSpec {
                reason: "uniform axis bounds must be finite",
            });
        }
        if min >= max {
            return Err(EntropyError::InvalidBinSpec {
                reason: "uniform axis requires min < max",
            });
        }
        if num_bins == 0 {
            return Err(EntropyError::InvalidBinSpec {
                reason: "uniform axis requires num_bins >= 1",
            });
        }
        Ok(Self::Uniform { min, max, num_bins })
    }

    /// Construye un eje con fronteras explícitas estrictamente crecientes.
    pub fn explicit_edges(edges: Vec<f64>) -> Result<Self, EntropyError> {
        if edges.len() < 2 {
            return Err(EntropyError::InvalidBinSpec {
                reason: "explicit edges require at least 2 boundary points",
            });
        }
        for window in edges.windows(2) {
            if !window[0].is_finite() || !window[1].is_finite() {
                return Err(EntropyError::InvalidBinSpec {
                    reason: "explicit bin edges must be finite",
                });
            }
            if window[0] >= window[1] {
                return Err(EntropyError::InvalidBinSpec {
                    reason: "explicit bin edges must be strictly increasing",
                });
            }
        }
        Ok(Self::ExplicitEdges { edges })
    }

    /// Devuelve el número total de bins en este eje.
    pub fn num_bins(&self) -> usize {
        match self {
            Self::Uniform { num_bins, .. } => *num_bins,
            Self::ExplicitEdges { edges } => edges.len() - 1,
        }
    }

    /// Devuelve el ancho uniforme si el eje es [`BinAxisSpec::Uniform`].
    pub fn uniform_width(&self) -> Option<f64> {
        match self {
            Self::Uniform { min, max, num_bins } => Some((*max - *min) / (*num_bins as f64)),
            Self::ExplicitEdges { .. } => None,
        }
    }

    /// Devuelve el índice de bin correspondiente a `value`.
    pub fn bin_of(&self, value: f64) -> Result<usize, EntropyError> {
        if !value.is_finite() {
            return Err(EntropyError::NonFiniteStateValue {
                field: "trajectory_step.values",
                value,
            });
        }
        match self {
            Self::Uniform { min, max, num_bins } => {
                if value <= *min {
                    return Ok(0);
                }
                if value >= *max {
                    return Ok(*num_bins - 1);
                }
                let ratio = (value - *min) / (*max - *min);
                let idx = (ratio * (*num_bins as f64)).floor() as usize;
                Ok(idx.min(*num_bins - 1))
            }
            Self::ExplicitEdges { edges } => {
                let last_bin = edges.len() - 2;
                if value < edges[0] {
                    return Ok(0);
                }
                if value >= edges[edges.len() - 1] {
                    return Ok(last_bin);
                }
                // `partition_point(|&e| e <= value)` devuelve el primer índice `i` con `edges[i] > value`.
                let upper = edges.partition_point(|&edge| edge <= value);
                Ok(upper.saturating_sub(1).min(last_bin))
            }
        }
    }
}

/// Especificación completa e inmutable de discretización por bins para vectores de estado.
#[derive(Debug, Clone, PartialEq)]
pub struct BinSpec {
    label: String,
    version: BinningVersion,
    axes: Vec<BinAxisSpec>,
}

impl BinSpec {
    /// Construye una especificación [`BinSpec`] a partir de una etiqueta, versión y ejes validados.
    pub fn from_axes(
        label: impl Into<String>,
        version: BinningVersion,
        axes: Vec<BinAxisSpec>,
    ) -> Result<Self, EntropyError> {
        let label = label.into();
        if label.trim().is_empty() {
            return Err(EntropyError::InvalidBinSpec {
                reason: "bin spec label cannot be empty",
            });
        }
        if axes.is_empty() {
            return Err(EntropyError::InvalidBinSpec {
                reason: "bin spec must define at least one axis",
            });
        }
        Ok(Self {
            label,
            version,
            axes,
        })
    }

    /// Atajo para construir una especificación uniforme a partir de tuplas `(min, max, num_bins)`.
    pub fn uniform(
        label: impl Into<String>,
        version: BinningVersion,
        axis_defs: Vec<(f64, f64, usize)>,
    ) -> Result<Self, EntropyError> {
        let mut axes = Vec::with_capacity(axis_defs.len());
        for (min, max, num_bins) in axis_defs {
            axes.push(BinAxisSpec::uniform(min, max, num_bins)?);
        }
        Self::from_axes(label, version, axes)
    }

    /// Etiqueta descriptiva del esquema de bins.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Versión explícita del esquema de discretización.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Ejes de discretización configurados.
    pub fn axes(&self) -> &[BinAxisSpec] {
        &self.axes
    }

    /// Dimensionalidad de estado esperada por esta especificación.
    pub fn dimension(&self) -> usize {
        self.axes.len()
    }

    /// Discretiza un vector de estado en sus coordenadas de bin por dimensión.
    pub fn bin_coords(&self, values: &[f64]) -> Result<Vec<usize>, EntropyError> {
        if values.len() != self.axes.len() {
            return Err(EntropyError::DimensionMismatch {
                expected: self.axes.len(),
                actual: values.len(),
            });
        }
        let mut coords = Vec::with_capacity(values.len());
        for (axis, &value) in self.axes.iter().zip(values.iter()) {
            coords.push(axis.bin_of(value)?);
        }
        Ok(coords)
    }
}

/// Paso discreto inmutable dentro de una trayectoria muestreada.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryStep {
    tick: Tick,
    values: Vec<f64>,
}

impl TrajectoryStep {
    /// Construye un [`TrajectoryStep`] validando que todos sus valores sean finitos.
    pub fn new(tick: Tick, values: Vec<f64>) -> Result<Self, EntropyError> {
        for &value in &values {
            if !value.is_finite() {
                return Err(EntropyError::NonFiniteStateValue {
                    field: "trajectory_step.values",
                    value,
                });
            }
        }
        Ok(Self { tick, values })
    }

    /// Construye un [`TrajectoryStep`] a partir de un [`State`] de dominio.
    pub fn from_state(state: &State) -> Self {
        Self {
            tick: state.tick(),
            values: state.values().to_vec(),
        }
    }

    /// Devuelve el tick discreto del paso.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve el vector de valores escalares del estado en este paso.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// Resumen inmutable y no vacío de una trayectoria muestreada.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectorySummary {
    steps: Vec<TrajectoryStep>,
}

impl TrajectorySummary {
    /// Construye un [`TrajectorySummary`] validando que contenga al menos un paso.
    pub fn new(steps: Vec<TrajectoryStep>) -> Result<Self, EntropyError> {
        if steps.is_empty() {
            return Err(EntropyError::EmptyTrajectory {
                trajectory_index: 0,
            });
        }
        Ok(Self { steps })
    }

    /// Construye un [`TrajectorySummary`] desde una secuencia de [`State`].
    pub fn from_states(states: &[State]) -> Result<Self, EntropyError> {
        if states.is_empty() {
            return Err(EntropyError::EmptyTrajectory {
                trajectory_index: 0,
            });
        }
        let steps = states.iter().map(TrajectoryStep::from_state).collect();
        Ok(Self { steps })
    }

    /// Devuelve la secuencia completa de pasos de la trayectoria.
    pub fn steps(&self) -> &[TrajectoryStep] {
        &self.steps
    }

    /// Devuelve el paso final de la trayectoria.
    pub fn final_step(&self) -> &TrajectoryStep {
        // Invariante de construcción: `steps` nunca está vacío.
        &self.steps[self.steps.len() - 1]
    }

    /// Devuelve la longitud (horizonte en pasos) de la trayectoria.
    pub fn horizon(&self) -> usize {
        self.steps.len()
    }
}

/// Tipo de proxy empírico calculado sobre el conjunto de trayectorias discretizadas por bins.
///
/// Ambos modos producen **proxies empíricos discretizados**, no entropía causal física.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntropyProxyKind {
    /// Proxy empírico sobre la secuencia completa de bins visitados en cada trayectoria futura.
    FutureTrajectoryProxy,
    /// Proxy empírico únicamente sobre el bin del estado final de cada trayectoria.
    FinalStateProxy,
}

/// Distribución empírica determinista agrupada por claves de bins discretos.
///
/// Cada clave es una secuencia de coordenadas de bin (`Vec<Vec<usize>>`):
/// - Para [`EntropyProxyKind::FinalStateProxy`], la clave contiene un único vector de
///   coordenadas correspondiente al estado final.
/// - Para [`EntropyProxyKind::FutureTrajectoryProxy`], la clave contiene la secuencia completa
///   de coordenadas de bin paso a paso de la trayectoria.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpiricalDistribution {
    total_samples: usize,
    counts: BTreeMap<Vec<Vec<usize>>, usize>,
    probabilities: Vec<f64>,
}

impl EmpiricalDistribution {
    /// Construye una distribución empírica determinista a partir de claves de bins observadas.
    pub fn from_keys(
        keys: impl IntoIterator<Item = Vec<Vec<usize>>>,
    ) -> Result<Self, EntropyError> {
        let mut counts = BTreeMap::new();
        let mut total_samples = 0_usize;

        for key in keys {
            *counts.entry(key).or_insert(0) += 1;
            total_samples += 1;
        }

        if total_samples == 0 {
            return Err(EntropyError::EmptyTrajectories);
        }

        let total_f64 = total_samples as f64;
        let probabilities: Vec<f64> = counts
            .values()
            .map(|&count| (count as f64) / total_f64)
            .collect();

        Ok(Self {
            total_samples,
            counts,
            probabilities,
        })
    }

    /// Número total de muestras (trayectorias) contabilizadas.
    pub const fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Número de bins o trayectorias discretizadas distintas observadas.
    pub fn occupied_bins(&self) -> usize {
        self.counts.len()
    }

    /// Mapa ordenado de conteos por clave discretizada.
    pub const fn counts(&self) -> &BTreeMap<Vec<Vec<usize>>, usize> {
        &self.counts
    }

    /// Vector de probabilidades empíricas (ordenadas por clave, suma `1.0`).
    pub fn probabilities(&self) -> &[f64] {
        &self.probabilities
    }

    /// Evalúa la entropía de Shannon en bits sobre esta distribución empírica.
    pub fn shannon_entropy_bits(&self) -> Result<f64, EntropyError> {
        shannon_entropy_bits(&self.probabilities)
    }
}

/// Resultado de estimar un proxy empírico de entropía sobre trayectorias discretizadas por bins.
///
/// Registra explícitamente el [`EntropyProxyKind`], la [`BinSpec`] y la [`BinningVersion`]
/// empleadas, dejando explícito que se trata de un proxy empírico agrupado por bins.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryEntropyEstimate {
    proxy_kind: EntropyProxyKind,
    entropy_bits: f64,
    bin_spec: BinSpec,
    binning_version: BinningVersion,
    distribution: EmpiricalDistribution,
}

impl TrajectoryEntropyEstimate {
    /// Tipo de proxy empírico estimado ([`EntropyProxyKind::FutureTrajectoryProxy`] o
    /// [`EntropyProxyKind::FinalStateProxy`]).
    pub const fn proxy_kind(&self) -> EntropyProxyKind {
        self.proxy_kind
    }

    /// Proxy empírico de entropía en **bits** calculado sobre la distribución agrupada por bins.
    pub const fn entropy_bits(&self) -> f64 {
        self.entropy_bits
    }

    /// Número de trayectorias muestreadas.
    pub const fn sample_count(&self) -> usize {
        self.distribution.total_samples()
    }

    /// Número de bins o trayectorias discretizadas distintas observadas.
    pub fn occupied_bins(&self) -> usize {
        self.distribution.occupied_bins()
    }

    /// Especificación de bins empleada en la discretización.
    pub const fn bin_spec(&self) -> &BinSpec {
        &self.bin_spec
    }

    /// Versión del esquema de bins registrada.
    pub const fn binning_version(&self) -> BinningVersion {
        self.binning_version
    }

    /// Distribución empírica subyacente agrupada por bins.
    pub const fn distribution(&self) -> &EmpiricalDistribution {
        &self.distribution
    }
}

/// Estima un proxy empírico de entropía en bits sobre un conjunto de trayectorias agrupadas por
/// [`BinSpec`], distinguiendo entre futuros completos ([`EntropyProxyKind::FutureTrajectoryProxy`])
/// y estados finales únicamente ([`EntropyProxyKind::FinalStateProxy`]).
pub fn estimate_trajectory_entropy_proxy(
    trajectories: &[TrajectorySummary],
    bin_spec: &BinSpec,
    proxy_kind: EntropyProxyKind,
) -> Result<TrajectoryEntropyEstimate, EntropyError> {
    if trajectories.is_empty() {
        return Err(EntropyError::EmptyTrajectories);
    }

    let mut keys = Vec::with_capacity(trajectories.len());
    for (idx, trajectory) in trajectories.iter().enumerate() {
        if trajectory.steps().is_empty() {
            return Err(EntropyError::EmptyTrajectory {
                trajectory_index: idx,
            });
        }
        let key = match proxy_kind {
            EntropyProxyKind::FutureTrajectoryProxy => {
                let mut seq = Vec::with_capacity(trajectory.horizon());
                for step in trajectory.steps() {
                    seq.push(bin_spec.bin_coords(step.values())?);
                }
                seq
            }
            EntropyProxyKind::FinalStateProxy => {
                vec![bin_spec.bin_coords(trajectory.final_step().values())?]
            }
        };
        keys.push(key);
    }

    let distribution = EmpiricalDistribution::from_keys(keys)?;
    let entropy_bits = distribution.shannon_entropy_bits()?;

    Ok(TrajectoryEntropyEstimate {
        proxy_kind,
        entropy_bits,
        bin_spec: bin_spec.clone(),
        binning_version: bin_spec.version(),
        distribution,
    })
}

/// Atajo para estimar el proxy empírico de entropía sobre trayectorias futuras completas.
pub fn estimate_future_trajectory_entropy_proxy(
    trajectories: &[TrajectorySummary],
    bin_spec: &BinSpec,
) -> Result<TrajectoryEntropyEstimate, EntropyError> {
    estimate_trajectory_entropy_proxy(
        trajectories,
        bin_spec,
        EntropyProxyKind::FutureTrajectoryProxy,
    )
}

/// Atajo para estimar el proxy empírico de entropía únicamente sobre estados finales.
pub fn estimate_final_state_entropy_proxy(
    trajectories: &[TrajectorySummary],
    bin_spec: &BinSpec,
) -> Result<TrajectoryEntropyEstimate, EntropyError> {
    estimate_trajectory_entropy_proxy(trajectories, bin_spec, EntropyProxyKind::FinalStateProxy)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shannon_entropy_computes_exact_uniform_and_deterministic_bits() {
        let uniform_4 = shannon_entropy_bits(&[0.25, 0.25, 0.25, 0.25]).unwrap();
        assert!((uniform_4 - 2.0).abs() < 1e-12);

        let deterministic = shannon_entropy_bits(&[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(deterministic, 0.0);
        assert!(deterministic.is_sign_positive());

        let single = shannon_entropy_bits(&[1.0]).unwrap();
        assert_eq!(single, 0.0);
    }

    #[test]
    fn shannon_entropy_rejects_invalid_distributions() {
        assert!(matches!(
            shannon_entropy_bits(&[]),
            Err(EntropyError::EmptyDistribution)
        ));
        assert!(matches!(
            shannon_entropy_bits(&[0.5, f64::NAN]),
            Err(EntropyError::NonFiniteProbability { index: 1, .. })
        ));
        assert!(matches!(
            shannon_entropy_bits(&[f64::INFINITY, 0.0]),
            Err(EntropyError::NonFiniteProbability { index: 0, .. })
        ));
        assert!(matches!(
            shannon_entropy_bits(&[1.2, -0.2]),
            Err(EntropyError::NegativeProbability { index: 1, .. })
        ));
        assert!(matches!(
            shannon_entropy_bits(&[0.4, 0.4]),
            Err(EntropyError::ProbabilitySumMismatch { .. })
        ));
        assert!(matches!(
            shannon_entropy_bits(&[0.6, 0.6]),
            Err(EntropyError::ProbabilitySumMismatch { .. })
        ));
        assert!(matches!(
            shannon_entropy_bits_with_tolerance(&[1.0], -1e-6),
            Err(EntropyError::InvalidTolerance { .. })
        ));
    }

    #[test]
    fn empirical_proxy_distinguishes_future_trajectories_from_final_states_and_records_bins() {
        let version = BinningVersion::new(1, 2);
        let bin_spec = BinSpec::uniform("1d-pos-grid", version, vec![(-1.0, 3.0, 4)]).unwrap();
        // Ancho esperado por bin: (3.0 - (-1.0)) / 4 = 1.0
        // Bins: [-1, 0) -> 0, [0, 1) -> 1, [1, 2) -> 2, [2, 3] -> 3
        assert_eq!(bin_spec.axes()[0].uniform_width(), Some(1.0));

        // 4 trayectorias que convergen al MISMO estado final (x = 1.5 -> bin 2),
        // pero recorren futuros intermedios distintos en t=1.
        let trajectories = vec![
            TrajectorySummary::new(vec![
                TrajectoryStep::new(Tick::new(0), vec![0.2]).unwrap(),
                TrajectoryStep::new(Tick::new(1), vec![-0.5]).unwrap(), // bin 0
                TrajectoryStep::new(Tick::new(2), vec![1.5]).unwrap(),  // bin 2
            ])
            .unwrap(),
            TrajectorySummary::new(vec![
                TrajectoryStep::new(Tick::new(0), vec![0.2]).unwrap(),
                TrajectoryStep::new(Tick::new(1), vec![0.5]).unwrap(), // bin 1
                TrajectoryStep::new(Tick::new(2), vec![1.5]).unwrap(), // bin 2
            ])
            .unwrap(),
            TrajectorySummary::new(vec![
                TrajectoryStep::new(Tick::new(0), vec![0.2]).unwrap(),
                TrajectoryStep::new(Tick::new(1), vec![1.5]).unwrap(), // bin 2
                TrajectoryStep::new(Tick::new(2), vec![1.5]).unwrap(), // bin 2
            ])
            .unwrap(),
            TrajectorySummary::new(vec![
                TrajectoryStep::new(Tick::new(0), vec![0.2]).unwrap(),
                TrajectoryStep::new(Tick::new(1), vec![2.5]).unwrap(), // bin 3
                TrajectoryStep::new(Tick::new(2), vec![1.5]).unwrap(), // bin 2
            ])
            .unwrap(),
        ];

        let future_est =
            estimate_future_trajectory_entropy_proxy(&trajectories, &bin_spec).unwrap();
        let final_est = estimate_final_state_entropy_proxy(&trajectories, &bin_spec).unwrap();

        assert_eq!(
            future_est.proxy_kind(),
            EntropyProxyKind::FutureTrajectoryProxy
        );
        assert_eq!(final_est.proxy_kind(), EntropyProxyKind::FinalStateProxy);

        // Futuros completos: 4 trayectorias equiprobables distintas -> log2(4) = 2.0 bits.
        assert!((future_est.entropy_bits() - 2.0).abs() < 1e-12);
        assert_eq!(future_est.occupied_bins(), 4);
        assert_eq!(future_est.sample_count(), 4);

        // Estados finales únicamente: todas convergen al bin 2 -> 0.0 bits.
        assert_eq!(final_est.entropy_bits(), 0.0);
        assert_eq!(final_est.occupied_bins(), 1);
        assert_eq!(final_est.sample_count(), 4);

        // Metadatos de bins y versión explícitamente registrados.
        assert_eq!(future_est.bin_spec().label(), "1d-pos-grid");
        assert_eq!(future_est.binning_version(), version);
        assert_eq!(final_est.bin_spec(), &bin_spec);
        assert_eq!(final_est.binning_version(), version);
    }

    #[test]
    fn explicit_edges_and_validation_reject_bad_inputs() {
        let axis = BinAxisSpec::explicit_edges(vec![0.0, 0.5, 2.0]).unwrap();
        assert_eq!(axis.num_bins(), 2);
        assert_eq!(axis.uniform_width(), None);
        assert_eq!(axis.bin_of(-1.0).unwrap(), 0);
        assert_eq!(axis.bin_of(0.25).unwrap(), 0);
        assert_eq!(axis.bin_of(0.5).unwrap(), 1);
        assert_eq!(axis.bin_of(3.0).unwrap(), 1);

        assert!(matches!(
            BinAxisSpec::explicit_edges(vec![1.0, 1.0]),
            Err(EntropyError::InvalidBinSpec { .. })
        ));
        assert!(matches!(
            BinSpec::uniform(" ", BinningVersion::V1_0, vec![(0.0, 1.0, 2)]),
            Err(EntropyError::InvalidBinSpec { .. })
        ));

        let spec = BinSpec::from_axes("edges", BinningVersion::V1_0, vec![axis]).unwrap();
        let traj = TrajectorySummary::new(vec![
            TrajectoryStep::new(Tick::ZERO, vec![0.1, 0.2]).unwrap(),
        ])
        .unwrap();
        assert!(matches!(
            estimate_final_state_entropy_proxy(&[traj], &spec),
            Err(EntropyError::DimensionMismatch {
                expected: 1,
                actual: 2
            })
        ));
        assert!(matches!(
            estimate_final_state_entropy_proxy(&[], &spec),
            Err(EntropyError::EmptyTrajectories)
        ));
    }
}
