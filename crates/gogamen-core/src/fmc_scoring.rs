//! Valor virtual (`VR`), relativización universal, selección de compañeros y clonación FMC (`std`-only).
//!
//! Implementa la variante `FMC-v1-gogamen`:
//! - [`FmcScoringConfig`]: hiperparámetros `alpha`, `beta`, `epsilon` y `z_clip`.
//! - [`relativize`]: normalización z-score acotada con `exp(clip(z, -z_clip, z_clip))`
//!   y retorno constante `1.0` cuando `std <= epsilon`.
//! - [`relativize_viable`] / [`pair_companions_viable`]: variantes enmascaradas usadas por
//!   [`score_population`]; media, `std` y compañeros se restringen al subconjunto viable
//!   `V = { i | !is_failure_terminal(terminal_i) }`.
//! - [`pair_companions`]: muestreo determinista de compañero `c != i` y distancia L2
//!   `||phi(s_i) - phi(s_c)||_2` (caso nulo explícito para `N < 2`).
//! - [`compute_virtual_rewards`]: `VR_i = D_i^alpha * R_i^beta`, con centinela `0.0`
//!   para walkers en fallo terminal ([`TerminalReason::Failure`] o [`TerminalReason::InvariantViolated`]).
//! - [`cloning_probability`] y [`resample_population`]: decisión estocástica reproducible
//!   sobre [`WalkerPopulation::clone_walkers_synchronously`], impidiendo clonar desde estados fallidos.

use std::error::Error;
use std::fmt;

use crate::domain::{State, Terminal, TerminalReason};
use crate::fmc_walkers::{PopulationError, WalkerId, WalkerPopulation};
use crate::model::RngSource;

/// Parámetros de configuración para el scoring de valor virtual `FMC-v1-gogamen`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmcScoringConfig {
    alpha: f64,
    beta: f64,
    epsilon: f64,
    z_clip: f64,
    unbiased_sampling: bool,
}

impl FmcScoringConfig {
    /// Crea y valida una configuración de scoring FMC.
    pub fn new(alpha: f64, beta: f64, epsilon: f64, z_clip: f64) -> Result<Self, FmcScoringError> {
        if !alpha.is_finite() || alpha < 0.0 {
            return Err(FmcScoringError::InvalidConfig {
                field: "alpha",
                value: alpha,
                reason: "must be finite and >= 0.0",
            });
        }
        if !beta.is_finite() || beta < 0.0 {
            return Err(FmcScoringError::InvalidConfig {
                field: "beta",
                value: beta,
                reason: "must be finite and >= 0.0",
            });
        }
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(FmcScoringError::InvalidConfig {
                field: "epsilon",
                value: epsilon,
                reason: "must be finite and >= 0.0",
            });
        }
        if !z_clip.is_finite() || z_clip <= 0.0 {
            return Err(FmcScoringError::InvalidConfig {
                field: "z_clip",
                value: z_clip,
                reason: "must be finite and > 0.0",
            });
        }

        Ok(Self {
            alpha,
            beta,
            epsilon,
            z_clip,
            unbiased_sampling: false,
        })
    }

    /// Exponente aplicado a la distancia relativizada `D_i`.
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Exponente aplicado a la recompensa relativizada `R_i`.
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Umbral mínimo de desviación estándar para evitar división por cero en [`relativize`].
    pub const fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Límite simétrico de recorte para z-scores `[-z_clip, z_clip]`.
    pub const fn z_clip(&self) -> f64 {
        self.z_clip
    }

    /// Opt-in: usa [`RngSource::next_usize_unbiased`] (rejection sampling) en vez de `next_u64() % span`.
    ///
    /// Default `false` (legacy modulo path) to keep seeded oracles bit-stable. When `true` it
    /// affects companion pairing in [`score_population`] / [`score_population_in_place`] and the
    /// planners' initial action sampling; it consumes a different RNG sequence.
    /// [`resample_population`] has no config and always keeps the legacy path.
    #[must_use]
    pub const fn with_unbiased_sampling(mut self, enabled: bool) -> Self {
        self.unbiased_sampling = enabled;
        self
    }

    /// Whether unbiased (rejection) sampling is enabled; see [`Self::with_unbiased_sampling`].
    pub const fn unbiased_sampling(&self) -> bool {
        self.unbiased_sampling
    }
}

impl Default for FmcScoringConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            epsilon: 1e-8,
            z_clip: 5.0,
            unbiased_sampling: false,
        }
    }
}

/// Errores explícitos de relativización, emparejamiento, scoring y resampling FMC.
#[derive(Debug, Clone, PartialEq)]
pub enum FmcScoringError {
    /// La población o vector de entrada está vacío (`N == 0`).
    EmptyPopulation,
    /// Un parámetro de [`FmcScoringConfig`] es inválido.
    InvalidConfig {
        /// Campo de configuración inválido.
        field: &'static str,
        /// Valor recibido.
        value: f64,
        /// Motivo de invalidez.
        reason: &'static str,
    },
    /// Se detectó un valor no finito (`NaN` o infinito).
    NonFiniteValue {
        /// Contexto donde apareció el valor no finito.
        context: &'static str,
        /// Índice del elemento afectado.
        index: usize,
        /// Valor no finito recibido.
        value: f64,
    },
    /// Se recibió un valor virtual negativo (`VR < 0.0`).
    NegativeVirtualReward {
        /// Índice del walker afectado.
        index: usize,
        /// Valor negativo recibido.
        value: f64,
    },
    /// Las proyecciones `phi(state)` tienen dimensionalidad heterogénea.
    FeatureDimensionMismatch {
        /// Walker cuya proyección difiere.
        walker: WalkerId,
        /// Dimensión esperada según el primer walker.
        expected: usize,
        /// Dimensión devuelta.
        actual: usize,
    },
    /// Dos slices de entrada no tienen la misma longitud que la población.
    LengthMismatch {
        /// Contexto del desajuste.
        context: &'static str,
        /// Longitud esperada.
        expected: usize,
        /// Longitud recibida.
        actual: usize,
    },
    /// Error propagado desde [`WalkerPopulation`].
    Population(PopulationError),
}

impl fmt::Display for FmcScoringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyPopulation => write!(f, "FMC scoring requires a non-empty population"),
            Self::InvalidConfig {
                field,
                value,
                reason,
            } => write!(f, "invalid FmcScoringConfig.{field} ({value}): {reason}"),
            Self::NonFiniteValue {
                context,
                index,
                value,
            } => write!(f, "non-finite value ({value}) at {context}[{index}]"),
            Self::NegativeVirtualReward { index, value } => {
                write!(f, "negative virtual reward ({value}) at index {index}")
            }
            Self::FeatureDimensionMismatch {
                walker,
                expected,
                actual,
            } => write!(
                f,
                "feature dimension mismatch for {walker}: expected {expected}, got {actual}"
            ),
            Self::LengthMismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "length mismatch in {context}: expected {expected}, got {actual}"
            ),
            Self::Population(err) => write!(f, "population error during FMC resampling: {err}"),
        }
    }
}

impl Error for FmcScoringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Population(err) => Some(err),
            _ => None,
        }
    }
}

impl From<PopulationError> for FmcScoringError {
    fn from(err: PopulationError) -> Self {
        Self::Population(err)
    }
}

/// Resultado del emparejamiento de un walker con un compañero aleatorio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CompanionPair {
    walker: WalkerId,
    companion: Option<WalkerId>,
    distance: f64,
}

impl CompanionPair {
    /// Identificador del walker evaluado.
    pub const fn walker(self) -> WalkerId {
        self.walker
    }

    /// Compañero muestreado (`None` si `N < 2`).
    pub const fn companion(self) -> Option<WalkerId> {
        self.companion
    }

    /// Distancia L2 `||phi(s_i) - phi(s_c)||_2` (`0.0` si `N < 2`).
    pub const fn distance(self) -> f64 {
        self.distance
    }
}

/// Resultado completo de scoring para una población de walkers.
#[derive(Debug, Clone, PartialEq)]
pub struct PopulationScore {
    companions: Vec<CompanionPair>,
    relativized_distances: Vec<f64>,
    relativized_rewards: Vec<f64>,
    virtual_rewards: Vec<f64>,
}

impl PopulationScore {
    /// Pares `(walker, companion, distance)` generados durante la evaluación.
    pub fn companions(&self) -> &[CompanionPair] {
        &self.companions
    }

    /// Distancias relativizadas `D_i` por walker.
    pub fn relativized_distances(&self) -> &[f64] {
        &self.relativized_distances
    }

    /// Recompensas relativizadas `R_i` por walker.
    pub fn relativized_rewards(&self) -> &[f64] {
        &self.relativized_rewards
    }

    /// Valores virtuales `VR_i` por walker (`0.0` para fallos terminales).
    pub fn virtual_rewards(&self) -> &[f64] {
        &self.virtual_rewards
    }
}

/// Decisión individual de clonación para un walker durante el resampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResampleDecision {
    walker: WalkerId,
    companion: Option<WalkerId>,
    probability: f64,
    cloned: bool,
}

impl ResampleDecision {
    /// Walker destino evaluado para clonación.
    pub const fn walker(self) -> WalkerId {
        self.walker
    }

    /// Compañero candidato seleccionado como fuente (`None` si `N < 2`).
    pub const fn companion(self) -> Option<WalkerId> {
        self.companion
    }

    /// Probabilidad de clonación calculada `p in [0.0, 1.0]`.
    pub const fn probability(self) -> f64 {
        self.probability
    }

    /// Indica si finalmente se ejecutó la clonación `companion -> walker`.
    pub const fn cloned(self) -> bool {
        self.cloned
    }
}

/// Resumen de ejecución de [`resample_population`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResampleReport {
    cloned_count: usize,
    skipped_failed_companion: usize,
    decisions: Vec<ResampleDecision>,
}

impl ResampleReport {
    /// Número total de clonaciones ejecutadas.
    pub const fn cloned_count(&self) -> usize {
        self.cloned_count
    }

    /// Número de decisiones omitidas porque el compañero estaba en fallo terminal.
    pub const fn skipped_failed_companion(&self) -> usize {
        self.skipped_failed_companion
    }

    /// Trazado por walker de las decisiones tomadas en el resampling.
    pub fn decisions(&self) -> &[ResampleDecision] {
        &self.decisions
    }
}

/// Devuelve `true` si el estado terminal corresponde a un fallo (`Failure` o `InvariantViolated`).
pub const fn is_failure_terminal(terminal: Terminal) -> bool {
    matches!(
        terminal,
        Terminal::Terminated(TerminalReason::Failure | TerminalReason::InvariantViolated)
    )
}

/// Relativiza un slice de valores escalares mediante `exp(clip((x - mean) / std, -z_clip, z_clip))`.
///
/// - Si `std <= config.epsilon()` (valores constantes o `N == 1`), devuelve `1.0` para todos.
/// - Rechaza entradas vacías o con valores no finitos (`NaN`, `±Inf`).
pub fn relativize(values: &[f64], config: &FmcScoringConfig) -> Result<Vec<f64>, FmcScoringError> {
    if values.is_empty() {
        return Err(FmcScoringError::EmptyPopulation);
    }

    let mut sum = 0.0;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "relativize.values",
                index,
                value,
            });
        }
        sum += value;
    }

    let n = values.len() as f64;
    let mean = sum / n;
    let mut sq_diff_sum = 0.0;
    for &value in values {
        let diff = value - mean;
        sq_diff_sum += diff * diff;
    }
    let std = (sq_diff_sum / n).sqrt();

    if !std.is_finite() {
        return Err(FmcScoringError::NonFiniteValue {
            context: "relativize.std",
            index: 0,
            value: std,
        });
    }

    if std <= config.epsilon() {
        return Ok(vec![1.0; values.len()]);
    }

    let mut out = Vec::with_capacity(values.len());
    let z_clip = config.z_clip();
    for (index, &value) in values.iter().enumerate() {
        let z = ((value - mean) / std).clamp(-z_clip, z_clip);
        let rel = z.exp();
        if !rel.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "relativize.output",
                index,
                value: rel,
            });
        }
        out.push(rel);
    }

    Ok(out)
}

/// Relativiza `values` usando únicamente el subconjunto viable `V = { i | !is_failure_terminal(terminals[i]) }`.
///
/// La media y la desviación estándar se calculan solo sobre `V`, de modo que la penalización de los
/// walkers en fallo terminal no infla `std` ni comprime los z-scores de los viables.
///
/// - Para `i` en `V` aplica la misma fórmula que [`relativize`] restringida a `V`.
/// - Si `|V| <= 1` o `std_V <= config.epsilon()`, devuelve `1.0` para `i` en `V`.
/// - Los walkers fuera de `V` reciben siempre `0.0` y sus valores no se validan.
/// - Rechaza longitudes inconsistentes y valores no finitos dentro de `V`.
pub fn relativize_viable(
    values: &[f64],
    terminals: &[Terminal],
    config: &FmcScoringConfig,
) -> Result<Vec<f64>, FmcScoringError> {
    let mut out = Vec::new();
    relativize_viable_into(values, terminals, config, &mut out)?;
    Ok(out)
}

/// Buffer-reusing variant of [`relativize_viable`]: writes the result into `out` (cleared first)
/// without allocating a viable-index or subset vector. Bit-identical to [`relativize_viable`].
pub fn relativize_viable_into(
    values: &[f64],
    terminals: &[Terminal],
    config: &FmcScoringConfig,
    out: &mut Vec<f64>,
) -> Result<(), FmcScoringError> {
    if values.is_empty() {
        return Err(FmcScoringError::EmptyPopulation);
    }
    if terminals.len() != values.len() {
        return Err(FmcScoringError::LengthMismatch {
            context: "relativize_viable.terminals",
            expected: values.len(),
            actual: terminals.len(),
        });
    }

    out.clear();
    out.resize(values.len(), 0.0);
    let is_viable = |i: usize| !is_failure_terminal(terminals[i]);

    let mut count = 0usize;
    let mut first_viable = 0usize;
    let mut sum = 0.0;
    for (index, &value) in values.iter().enumerate() {
        if !is_viable(index) {
            continue;
        }
        if count == 0 {
            first_viable = index;
        }
        if !value.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "relativize.values",
                index,
                value,
            });
        }
        sum += value;
        count += 1;
    }
    if count == 0 {
        return Ok(());
    }

    let n = count as f64;
    let mean = sum / n;
    let mut sq_diff_sum = 0.0;
    for (index, &value) in values.iter().enumerate() {
        if is_viable(index) {
            let diff = value - mean;
            sq_diff_sum += diff * diff;
        }
    }
    let std = (sq_diff_sum / n).sqrt();
    if !std.is_finite() {
        return Err(FmcScoringError::NonFiniteValue {
            context: "relativize.std",
            index: first_viable,
            value: std,
        });
    }

    if std <= config.epsilon() {
        for (index, slot) in out.iter_mut().enumerate() {
            if is_viable(index) {
                *slot = 1.0;
            }
        }
        return Ok(());
    }

    let z_clip = config.z_clip();
    for (index, &value) in values.iter().enumerate() {
        if !is_viable(index) {
            continue;
        }
        let z = ((value - mean) / std).clamp(-z_clip, z_clip);
        let rel = z.exp();
        if !rel.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "relativize.output",
                index,
                value: rel,
            });
        }
        out[index] = rel;
    }
    Ok(())
}

/// Muestrea uniformemente un índice de compañero `c != walker_index` en `0..population_len`.
///
/// Devuelve `None` si `population_len < 2`.
pub fn sample_companion_index(
    walker_index: usize,
    population_len: usize,
    rng: &mut dyn RngSource,
) -> Option<usize> {
    if population_len < 2 || walker_index >= population_len {
        return None;
    }
    let span = (population_len - 1) as u64;
    let raw = (rng.next_u64() % span) as usize;
    let companion = if raw >= walker_index { raw + 1 } else { raw };
    Some(companion)
}

/// Like [`sample_companion_index`] but without modulo bias (rejection sampling).
///
/// Consumes a different RNG sequence than the legacy variant when `span` is not a power of two.
pub fn sample_companion_index_unbiased(
    walker_index: usize,
    population_len: usize,
    rng: &mut dyn RngSource,
) -> Option<usize> {
    if population_len < 2 || walker_index >= population_len {
        return None;
    }
    let raw = rng.next_usize_unbiased(population_len - 1);
    Some(if raw >= walker_index { raw + 1 } else { raw })
}

/// Empareja cada walker con un compañero aleatorio `c != i` y calcula `d_i = ||phi(s_i) - phi(s_c)||_2`.
///
/// Equivale a [`pair_companions_viable`] con todos los walkers viables.
///
/// - Para `N == 1` (`N < 2`), devuelve el caso nulo explícito: `companion = None`, `distance = 0.0`.
/// - Valida finitud de todas las proyecciones `phi(s)` y de las distancias resultantes.
pub fn pair_companions<F>(
    states: &[State],
    phi: F,
    rng: &mut dyn RngSource,
) -> Result<Vec<CompanionPair>, FmcScoringError>
where
    F: FnMut(&State) -> Vec<f64>,
{
    let terminals = vec![Terminal::Ongoing; states.len()];
    pair_companions_viable(states, &terminals, phi, rng)
}

/// Empareja solo walkers viables (`!is_failure_terminal`) entre sí.
///
/// - Cada `i` viable muestrea `c` uniformemente en `V \ {i}` y mide `||phi(s_i) - phi(s_c)||_2`.
///   Si no hay fallos terminales, el consumo del RNG y el resultado son idénticos a [`pair_companions`].
/// - Los walkers en fallo terminal no se emparejan (`companion = None`, `distance = 0.0`), no consumen
///   RNG y su `phi` no se evalúa.
/// - Si `|V| < 2`, todos los walkers reciben el caso nulo (`companion = None`, `distance = 0.0`).
/// - Devuelve siempre un elemento por walker, en orden de índice.
pub fn pair_companions_viable<F>(
    states: &[State],
    terminals: &[Terminal],
    mut phi: F,
    rng: &mut dyn RngSource,
) -> Result<Vec<CompanionPair>, FmcScoringError>
where
    F: FnMut(&State) -> Vec<f64>,
{
    let mut ws = FmcScratchWorkspace::new();
    pair_companions_core(
        states,
        terminals,
        |state, out| out.extend_from_slice(&phi(state)),
        false,
        rng,
        &mut ws,
    )?;
    Ok(std::mem::take(&mut ws.companions))
}

/// Reusable flat buffers for allocation-free FMC scoring (see [`score_population_in_place`]).
///
/// Buffers grow to the population size on first use and are then reused, so steady-state
/// scoring performs no per-walker (`O(N)`) allocations. Results of the last in-place scoring
/// are exposed through the accessors below and are overwritten by the next call.
#[derive(Debug, Clone, Default)]
pub struct FmcScratchWorkspace {
    projected_features: Vec<f64>,
    viable: Vec<usize>,
    companions: Vec<CompanionPair>,
    distances: Vec<f64>,
    d_rel: Vec<f64>,
    r_rel: Vec<f64>,
    virtual_rewards: Vec<f64>,
}

impl FmcScratchWorkspace {
    /// Creates an empty workspace; buffers are sized lazily on first use.
    pub fn new() -> Self {
        Self::default()
    }

    /// Companion pairs of the last in-place scoring.
    pub fn companions(&self) -> &[CompanionPair] {
        &self.companions
    }

    /// Raw L2 distances per walker (`0.0` for non-paired walkers).
    pub fn distances(&self) -> &[f64] {
        &self.distances
    }

    /// Relativized distances `D_i` of the last in-place scoring.
    pub fn relativized_distances(&self) -> &[f64] {
        &self.d_rel
    }

    /// Relativized rewards `R_i` of the last in-place scoring.
    pub fn relativized_rewards(&self) -> &[f64] {
        &self.r_rel
    }

    /// Virtual rewards `VR_i` of the last in-place scoring.
    pub fn virtual_rewards(&self) -> &[f64] {
        &self.virtual_rewards
    }

    /// Moves the last results out as an owned [`PopulationScore`] (buffers are re-grown on next use).
    pub fn take_score(&mut self) -> PopulationScore {
        PopulationScore {
            companions: std::mem::take(&mut self.companions),
            relativized_distances: std::mem::take(&mut self.d_rel),
            relativized_rewards: std::mem::take(&mut self.r_rel),
            virtual_rewards: std::mem::take(&mut self.virtual_rewards),
        }
    }
}

/// Shared pairing core. `fill` appends the features of one state to the flat buffer.
/// Leaves companions and distances in `ws`.
fn pair_companions_core<F>(
    states: &[State],
    terminals: &[Terminal],
    mut fill: F,
    unbiased: bool,
    rng: &mut dyn RngSource,
    ws: &mut FmcScratchWorkspace,
) -> Result<(), FmcScoringError>
where
    F: FnMut(&State, &mut Vec<f64>),
{
    if states.is_empty() {
        return Err(FmcScoringError::EmptyPopulation);
    }
    if terminals.len() != states.len() {
        return Err(FmcScoringError::LengthMismatch {
            context: "pair_companions_viable.terminals",
            expected: states.len(),
            actual: terminals.len(),
        });
    }

    ws.viable.clear();
    ws.viable
        .extend((0..states.len()).filter(|&i| !is_failure_terminal(terminals[i])));
    ws.projected_features.clear();
    let mut expected_dim: Option<usize> = None;
    for &index in &ws.viable {
        let start = ws.projected_features.len();
        fill(&states[index], &mut ws.projected_features);
        let len = ws.projected_features.len() - start;
        if let Some(dim) = expected_dim {
            if len != dim {
                return Err(FmcScoringError::FeatureDimensionMismatch {
                    walker: WalkerId::new(index),
                    expected: dim,
                    actual: len,
                });
            }
        } else {
            expected_dim = Some(len);
        }

        for &val in &ws.projected_features[start..] {
            if !val.is_finite() {
                return Err(FmcScoringError::NonFiniteValue {
                    context: "pair_companions.features",
                    index,
                    value: val,
                });
            }
        }
    }

    ws.companions.clear();
    ws.companions
        .extend((0..states.len()).map(|i| CompanionPair {
            walker: WalkerId::new(i),
            companion: None,
            distance: 0.0,
        }));
    ws.distances.clear();
    ws.distances.resize(states.len(), 0.0);

    if ws.viable.len() < 2 {
        return Ok(());
    }

    let dim = expected_dim.unwrap_or(0);
    let viable_len = ws.viable.len();
    for pos in 0..viable_len {
        let i = ws.viable[pos];
        let c_pos = if unbiased {
            sample_companion_index_unbiased(pos, viable_len, rng)
        } else {
            sample_companion_index(pos, viable_len, rng)
        }
        .expect("|V| >= 2");
        let c = ws.viable[c_pos];
        let dist = l2_distance(
            &ws.projected_features[pos * dim..(pos + 1) * dim],
            &ws.projected_features[c_pos * dim..(c_pos + 1) * dim],
        );
        if !dist.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "pair_companions.distance",
                index: i,
                value: dist,
            });
        }
        ws.companions[i].companion = Some(WalkerId::new(c));
        ws.companions[i].distance = dist;
        ws.distances[i] = dist;
    }

    Ok(())
}

fn l2_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut sum_sq = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Calcula los valores virtuales `VR_i = D_i^alpha * R_i^beta` con centinela `0.0` en fallo terminal.
pub fn compute_virtual_rewards(
    distances: &[f64],
    rewards: &[f64],
    terminals: &[Terminal],
    config: &FmcScoringConfig,
) -> Result<Vec<f64>, FmcScoringError> {
    if distances.is_empty() {
        return Err(FmcScoringError::EmptyPopulation);
    }
    if rewards.len() != distances.len() {
        return Err(FmcScoringError::LengthMismatch {
            context: "compute_virtual_rewards.rewards",
            expected: distances.len(),
            actual: rewards.len(),
        });
    }
    if terminals.len() != distances.len() {
        return Err(FmcScoringError::LengthMismatch {
            context: "compute_virtual_rewards.terminals",
            expected: distances.len(),
            actual: terminals.len(),
        });
    }

    let d_rel = relativize(distances, config)?;
    let r_rel = relativize(rewards, config)?;
    compute_vr_from_relativized(&d_rel, &r_rel, terminals, config)
}

fn compute_vr_from_relativized(
    d_rel: &[f64],
    r_rel: &[f64],
    terminals: &[Terminal],
    config: &FmcScoringConfig,
) -> Result<Vec<f64>, FmcScoringError> {
    let mut virtual_rewards = Vec::with_capacity(d_rel.len());
    compute_vr_into(d_rel, r_rel, terminals, config, &mut virtual_rewards)?;
    Ok(virtual_rewards)
}

fn compute_vr_into(
    d_rel: &[f64],
    r_rel: &[f64],
    terminals: &[Terminal],
    config: &FmcScoringConfig,
    virtual_rewards: &mut Vec<f64>,
) -> Result<(), FmcScoringError> {
    virtual_rewards.clear();
    for i in 0..d_rel.len() {
        if is_failure_terminal(terminals[i]) {
            virtual_rewards.push(0.0);
            continue;
        }
        let vr = d_rel[i].powf(config.alpha()) * r_rel[i].powf(config.beta());
        if !vr.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "compute_virtual_rewards.vr",
                index: i,
                value: vr,
            });
        }
        virtual_rewards.push(vr);
    }
    Ok(())
}

/// Ejecuta el pipeline completo de scoring (`pair_companions_viable` + `relativize_viable` + `VR`) sobre una [`WalkerPopulation`].
///
/// Los walkers en fallo terminal quedan fuera de la relativización y del emparejamiento: su
/// penalización no afecta a `R_i`/`VR_i` de los viables y reciben `VR = 0.0`. Con 0 o 1 viables no
/// hay NaN: los viables reciben `D = R = 1.0`.
pub fn score_population<F>(
    population: &WalkerPopulation,
    mut phi: F,
    config: &FmcScoringConfig,
    rng: &mut dyn RngSource,
) -> Result<PopulationScore, FmcScoringError>
where
    F: FnMut(&State) -> Vec<f64>,
{
    let mut ws = FmcScratchWorkspace::new();
    score_population_core(
        population,
        |state, out| out.extend_from_slice(&phi(state)),
        config,
        rng,
        &mut ws,
    )?;
    Ok(ws.take_score())
}

/// Allocation-free variant of [`score_population`]: `phi_slice` borrows the projection from the
/// state (e.g. `|s| s.values()`), so no `Vec` is created per walker, and all intermediate and
/// output buffers live in `workspace`.
///
/// Distances, relativized values and virtual rewards are bit-identical to [`score_population`]
/// for the same projection and RNG. Results are read from `workspace` (see
/// [`FmcScratchWorkspace::virtual_rewards`] etc.); once the workspace is warm, a call performs
/// no heap allocation.
pub fn score_population_in_place<P>(
    population: &WalkerPopulation,
    mut phi_slice: P,
    config: &FmcScoringConfig,
    rng: &mut dyn RngSource,
    workspace: &mut FmcScratchWorkspace,
) -> Result<(), FmcScoringError>
where
    P: for<'a> FnMut(&'a State) -> &'a [f64],
{
    score_population_core(
        population,
        |state, out| out.extend_from_slice(phi_slice(state)),
        config,
        rng,
        workspace,
    )
}

fn score_population_core<F>(
    population: &WalkerPopulation,
    fill: F,
    config: &FmcScoringConfig,
    rng: &mut dyn RngSource,
    ws: &mut FmcScratchWorkspace,
) -> Result<(), FmcScoringError>
where
    F: FnMut(&State, &mut Vec<f64>),
{
    let terminals = population.terminals();
    pair_companions_core(
        population.states(),
        terminals,
        fill,
        config.unbiased_sampling(),
        rng,
        ws,
    )?;
    relativize_viable_into(&ws.distances, terminals, config, &mut ws.d_rel)?;
    relativize_viable_into(
        population.cumulative_rewards(),
        terminals,
        config,
        &mut ws.r_rel,
    )?;
    compute_vr_into(
        &ws.d_rel,
        &ws.r_rel,
        terminals,
        config,
        &mut ws.virtual_rewards,
    )
}

/// Calcula la probabilidad de clonación de un walker con `vr_i` hacia un compañero con `vr_k`.
///
/// - Si `vr_i == 0.0` y `vr_k > 0.0` -> `1.0`.
/// - Si `vr_i >= vr_k` o ambos `0.0` -> `0.0`.
/// - Si `0.0 < vr_i < vr_k` -> `((vr_k - vr_i) / vr_k).clamp(0.0, 1.0)`.
pub fn cloning_probability(vr_i: f64, vr_k: f64) -> Result<f64, FmcScoringError> {
    if !vr_i.is_finite() {
        return Err(FmcScoringError::NonFiniteValue {
            context: "cloning_probability.vr_i",
            index: 0,
            value: vr_i,
        });
    }
    if !vr_k.is_finite() {
        return Err(FmcScoringError::NonFiniteValue {
            context: "cloning_probability.vr_k",
            index: 1,
            value: vr_k,
        });
    }
    if vr_i < 0.0 {
        return Err(FmcScoringError::NegativeVirtualReward {
            index: 0,
            value: vr_i,
        });
    }
    if vr_k < 0.0 {
        return Err(FmcScoringError::NegativeVirtualReward {
            index: 1,
            value: vr_k,
        });
    }

    if vr_i == 0.0 && vr_k > 0.0 {
        Ok(1.0)
    } else if vr_i >= vr_k || (vr_i == 0.0 && vr_k == 0.0) {
        Ok(0.0)
    } else {
        Ok(((vr_k - vr_i) / vr_k).clamp(0.0, 1.0))
    }
}

/// Aplica el resampling estocástico de clonación sobre `population` usando `WalkerPopulation::clone_walkers_synchronously`
/// (snapshot de lectura previo: cadenas `A->B->C` y ciclos `A<->B` no se sobreescriben entre sí).
///
/// Nunca clona desde un compañero en fallo terminal (`Failure` o `InvariantViolated`).
pub fn resample_population(
    population: &mut WalkerPopulation,
    virtual_rewards: &[f64],
    rng: &mut dyn RngSource,
    seed_salt: u64,
) -> Result<ResampleReport, FmcScoringError> {
    let n = population.len();
    if virtual_rewards.len() != n {
        return Err(FmcScoringError::LengthMismatch {
            context: "resample_population.virtual_rewards",
            expected: n,
            actual: virtual_rewards.len(),
        });
    }
    for (index, &vr) in virtual_rewards.iter().enumerate() {
        if !vr.is_finite() {
            return Err(FmcScoringError::NonFiniteValue {
                context: "resample_population.virtual_rewards",
                index,
                value: vr,
            });
        }
        if vr < 0.0 {
            return Err(FmcScoringError::NegativeVirtualReward { index, value: vr });
        }
    }

    if n < 2 {
        return Ok(ResampleReport {
            cloned_count: 0,
            skipped_failed_companion: 0,
            decisions: vec![ResampleDecision {
                walker: WalkerId::new(0),
                companion: None,
                probability: 0.0,
                cloned: false,
            }],
        });
    }

    let initial_terminals = population.terminals().to_vec();
    let mut decisions = Vec::with_capacity(n);
    let mut staged_clones = Vec::new();
    let mut skipped_failed_companion = 0usize;

    for (i, &vr_i) in virtual_rewards.iter().enumerate() {
        let k = sample_companion_index(i, n, rng).expect("population_len >= 2");
        if is_failure_terminal(initial_terminals[k]) {
            skipped_failed_companion += 1;
            decisions.push(ResampleDecision {
                walker: WalkerId::new(i),
                companion: Some(WalkerId::new(k)),
                probability: 0.0,
                cloned: false,
            });
            continue;
        }

        let prob = cloning_probability(vr_i, virtual_rewards[k])?;
        let draw = rng.next_f64();
        let should_clone = draw < prob;

        if should_clone {
            staged_clones.push((WalkerId::new(k), WalkerId::new(i)));
        }

        decisions.push(ResampleDecision {
            walker: WalkerId::new(i),
            companion: Some(WalkerId::new(k)),
            probability: prob,
            cloned: should_clone,
        });
    }

    let cloned_count = staged_clones.len();
    // Synchronous clone: every dst receives the pre-resampling state of its src.
    population.clone_walkers_synchronously(&staged_clones, seed_salt)?;

    Ok(ResampleReport {
        cloned_count,
        skipped_failed_companion,
        decisions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{Action, ActionSpace};
    use crate::determinism::MasterSeed;
    use crate::domain::Transition;
    use crate::model::{DeterministicRng, GenerativeModel};
    use crate::planner::{BudgetTracker, PlanBudget};
    use crate::units::{SimDuration, StepDt, Tick};

    struct TwoActionStepModel {
        space: ActionSpace,
    }

    impl TwoActionStepModel {
        fn new() -> Self {
            Self {
                space: ActionSpace::discrete(2).unwrap(),
            }
        }
    }

    impl GenerativeModel for TwoActionStepModel {
        fn action_space(&self) -> &ActionSpace {
            &self.space
        }

        fn step(
            &self,
            state: &State,
            action: &Action,
            dt: StepDt,
            _rng: &mut dyn RngSource,
        ) -> Result<Transition, crate::error::ModelError> {
            self.space
                .validate(action)
                .map_err(crate::error::ModelError::InvalidAction)?;
            let idx = action.as_discrete().map(|d| d.index()).unwrap_or(0);
            let next_vals = if idx == 1 {
                vec![state.values()[0] + 1.0, state.values()[1] + 2.0]
            } else {
                vec![state.values()[0] - 1.0, state.values()[1]]
            };
            let next_state = state
                .advance(dt, next_vals.clone())
                .map_err(crate::error::ModelError::TransitionFailed)?;
            let obs = crate::domain::Observation::new(
                next_state.tick(),
                next_state.elapsed(),
                next_vals.clone(),
            )
            .map_err(crate::error::ModelError::TransitionFailed)?;
            let terminal = if idx == 0 && next_vals[0] <= -1.0 {
                Terminal::Terminated(TerminalReason::Failure)
            } else {
                Terminal::Ongoing
            };
            Transition::new(state.tick(), dt, next_state, obs, terminal)
                .map_err(crate::error::ModelError::TransitionFailed)
        }
    }

    fn failure_scoring_population(xs: &[f64], failing: &[bool], penalty: f64) -> WalkerPopulation {
        let model = TwoActionStepModel::new();
        let left = Action::discrete_in_space(0, model.action_space()).unwrap();
        let right = Action::discrete_in_space(1, model.action_space()).unwrap();
        let states: Vec<State> = xs
            .iter()
            .zip(failing)
            .map(|(&x, &f)| {
                let v = if f { 0.0 } else { x };
                State::new(Tick::ZERO, SimDuration::ZERO, vec![v, 0.0]).unwrap()
            })
            .collect();
        let actions: Vec<Action> = failing
            .iter()
            .map(|&f| if f { left.clone() } else { right.clone() })
            .collect();
        let mut pop =
            WalkerPopulation::from_states(states, MasterSeed::new(1), 0, 0, false).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut budget = BudgetTracker::new(PlanBudget::new(64, None));
        pop.step_population(&model, &actions, dt, &mut budget, |s, act, _t| {
            match act.as_discrete().map(|a| a.index()) {
                Some(1) => s.values()[0],
                _ => penalty,
            }
        })
        .unwrap();
        pop
    }

    #[test]
    fn gog119_ac1_extreme_failure_penalty_does_not_alter_viable_scores() {
        let config = FmcScoringConfig::default();
        let phi = |s: &State| s.values().to_vec();
        let baseline_pop = failure_scoring_population(&[0.1, 1.0], &[false, false], 0.0);
        let mut rng = DeterministicRng::new(5);
        let baseline = score_population(&baseline_pop, phi, &config, &mut rng).unwrap();

        for penalty in [-1.0, -100.0, -1000.0] {
            let pop = failure_scoring_population(&[0.1, 1.0, 0.0], &[false, false, true], penalty);
            assert_eq!(
                pop.terminal(WalkerId::new(2)).unwrap(),
                Terminal::Terminated(TerminalReason::Failure)
            );
            let mut rng = DeterministicRng::new(5);
            let score = score_population(&pop, phi, &config, &mut rng).unwrap();
            for i in 0..2 {
                assert_eq!(
                    score.relativized_rewards()[i],
                    baseline.relativized_rewards()[i]
                );
                assert_eq!(score.virtual_rewards()[i], baseline.virtual_rewards()[i]);
            }
            assert_eq!(score.virtual_rewards()[2], 0.0);
        }
    }

    #[test]
    fn gog119_relativize_viable_ignores_failed_values() {
        let config = FmcScoringConfig::default();
        let terminals = [
            Terminal::Ongoing,
            Terminal::Terminated(TerminalReason::InvariantViolated),
            Terminal::Ongoing,
        ];
        let masked = relativize_viable(&[0.1, -1000.0, 1.0], &terminals, &config).unwrap();
        let plain = relativize(&[0.1, 1.0], &config).unwrap();
        assert_eq!(masked, vec![plain[0], 0.0, plain[1]]);
    }

    #[test]
    fn gog119_ac2_viable_walkers_never_paired_with_failed() {
        let states: Vec<State> = (0..6)
            .map(|i| State::new(Tick::ZERO, SimDuration::ZERO, vec![i as f64]).unwrap())
            .collect();
        let terminals = [
            Terminal::Ongoing,
            Terminal::Terminated(TerminalReason::Failure),
            Terminal::Ongoing,
            Terminal::Terminated(TerminalReason::InvariantViolated),
            Terminal::Ongoing,
            Terminal::Ongoing,
        ];
        for seed in 0..50 {
            let mut rng = DeterministicRng::new(seed);
            let pairs =
                pair_companions_viable(&states, &terminals, |s| s.values().to_vec(), &mut rng)
                    .unwrap();
            assert_eq!(pairs.len(), 6);
            for (i, pair) in pairs.iter().enumerate() {
                if is_failure_terminal(terminals[i]) {
                    assert_eq!(pair.companion(), None);
                    assert_eq!(pair.distance(), 0.0);
                } else {
                    let c = pair.companion().expect("viable has companion").index();
                    assert_ne!(c, i);
                    assert!(!is_failure_terminal(terminals[c]));
                }
            }
        }
    }

    #[test]
    fn gog119_viable_pairing_matches_legacy_without_failures() {
        let states: Vec<State> = (0..5)
            .map(|i| State::new(Tick::ZERO, SimDuration::ZERO, vec![i as f64 * 1.5]).unwrap())
            .collect();
        let terminals = vec![Terminal::Ongoing; 5];
        let mut rng_a = DeterministicRng::new(9);
        let mut rng_b = DeterministicRng::new(9);
        let a = pair_companions(&states, |s| s.values().to_vec(), &mut rng_a).unwrap();
        let b = pair_companions_viable(&states, &terminals, |s| s.values().to_vec(), &mut rng_b)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(rng_a.next_u64(), rng_b.next_u64());
    }

    #[test]
    fn gog119_ac3_zero_or_one_viable_has_no_nan() {
        let config = FmcScoringConfig::default();
        let phi = |s: &State| s.values().to_vec();

        let none = failure_scoring_population(&[0.0, 0.0], &[true, true], -1000.0);
        let mut rng = DeterministicRng::new(3);
        let score = score_population(&none, phi, &config, &mut rng).unwrap();
        assert!(score.companions().iter().all(|c| c.companion().is_none()));
        for v in score
            .relativized_distances()
            .iter()
            .chain(score.relativized_rewards())
            .chain(score.virtual_rewards())
        {
            assert!(v.is_finite());
        }
        assert!(score.virtual_rewards().iter().all(|&v| v == 0.0));

        let one = failure_scoring_population(&[0.5, 0.0, 0.0], &[false, true, true], -1000.0);
        let mut rng = DeterministicRng::new(3);
        let score = score_population(&one, phi, &config, &mut rng).unwrap();
        assert!(score.companions().iter().all(|c| c.companion().is_none()));
        assert_eq!(score.relativized_distances()[0], 1.0);
        assert_eq!(score.relativized_rewards()[0], 1.0);
        assert_eq!(score.virtual_rewards()[0], 1.0);
        assert_eq!(&score.virtual_rewards()[1..], &[0.0, 0.0]);
        for v in score.virtual_rewards() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn ac1_manual_vector_matches_tolerance_and_selects_right() {
        // Vector manual de especificación:
        // d = [1.0, 3.0, 5.0, 3.0] -> mean = 3.0, std = sqrt((4 + 0 + 4 + 0)/4) = sqrt(2)
        // r = [1.0, 4.0, 5.0, -2.0] -> mean = 2.0, std = sqrt((1 + 4 + 9 + 16)/4) = sqrt(7.5)
        // W3 está en Failure terminal -> VR_3 = 0.0 (centinela).
        let config = FmcScoringConfig::default();
        let d = [1.0, 3.0, 5.0, 3.0];
        let r = [1.0, 4.0, 5.0, -2.0];
        let terminals = [
            Terminal::Ongoing,
            Terminal::Ongoing,
            Terminal::Ongoing,
            Terminal::Terminated(TerminalReason::Failure),
        ];

        let d_rel = relativize(&d, &config).unwrap();
        let r_rel = relativize(&r, &config).unwrap();
        let vr = compute_virtual_rewards(&d, &r, &terminals, &config).unwrap();

        let std_d = 2.0_f64.sqrt();
        let expected_d_rel = [
            (-2.0 / std_d).exp(),
            0.0_f64.exp(),
            (2.0 / std_d).exp(),
            0.0_f64.exp(),
        ];
        let std_r = 7.5_f64.sqrt();
        let expected_r_rel = [
            (-1.0 / std_r).exp(),
            (2.0 / std_r).exp(),
            (3.0 / std_r).exp(),
            (-4.0 / std_r).exp(),
        ];

        let tol = 1e-12;
        for i in 0..4 {
            assert!((d_rel[i] - expected_d_rel[i]).abs() < tol);
            assert!((r_rel[i] - expected_r_rel[i]).abs() < tol);
        }
        assert!((vr[0] - expected_d_rel[0] * expected_r_rel[0]).abs() < tol);
        assert!((vr[1] - expected_d_rel[1] * expected_r_rel[1]).abs() < tol);
        assert!((vr[2] - expected_d_rel[2] * expected_r_rel[2]).abs() < tol);
        assert_eq!(vr[3], 0.0);

        // Población de 4 walkers W0..W3 con seed = 42 donde W3 falla tras acción Left (0)
        // y W1, W2 avanzan con Right (1). Tras scoring + resampling, la acción seleccionada es Right (1).
        let model = TwoActionStepModel::new();
        let left = Action::discrete_in_space(0, model.action_space()).unwrap();
        let right = Action::discrete_in_space(1, model.action_space()).unwrap();

        let initial_states = vec![
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 0.0]).unwrap(), // W0: Left -> [0,0] Ongoing
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap(), // W1: Right -> [1,2] Ongoing
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 1.0]).unwrap(), // W2: Right -> [2,3] Ongoing
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap(), // W3: Left -> [-1,0] Failure
        ];
        let mut pop =
            WalkerPopulation::from_states(initial_states, MasterSeed::new(42), 0, 0, true).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut budget = BudgetTracker::new(PlanBudget::new(16, None));

        pop.step_population(
            &model,
            &[left.clone(), right.clone(), right.clone(), left.clone()],
            dt,
            &mut budget,
            |_s, act, _t| match act.as_discrete().map(|a| a.index()) {
                Some(1) => 5.0,
                _ => -1.0,
            },
        )
        .unwrap();

        assert_eq!(
            pop.terminal(WalkerId::new(3)).unwrap(),
            Terminal::Terminated(TerminalReason::Failure)
        );

        let mut rng = DeterministicRng::new(42);
        let score = score_population(&pop, |s| s.values().to_vec(), &config, &mut rng).unwrap();
        assert_eq!(score.virtual_rewards()[3], 0.0);
        // Con emparejamiento enmascarado (solo viables), W1 y W2 se miden entre sí o contra W0; la
        // diversidad de W1 depende del compañero muestreado, así que se exige dominancia agregada.
        assert!(score.virtual_rewards()[2] > score.virtual_rewards()[0]);
        assert!(
            score.virtual_rewards()[1] + score.virtual_rewards()[2]
                > 2.0 * score.virtual_rewards()[0]
        );

        let report = resample_population(&mut pop, score.virtual_rewards(), &mut rng, 42).unwrap();
        assert!(report.cloned_count() >= 1);

        let right_votes = pop
            .first_actions()
            .iter()
            .filter(|a| {
                a.as_ref()
                    .and_then(|act| act.as_discrete())
                    .map(|d| d.index())
                    == Some(1)
            })
            .count();
        let left_votes = pop.len() - right_votes;
        assert!(
            right_votes > left_votes,
            "expected Right majority after resampling, got right={right_votes}, left={left_votes}"
        );
    }

    #[test]
    fn ac2_handles_constant_zero_distance_negative_rewards_and_all_terminal() {
        let config = FmcScoringConfig::default();

        // Valores constantes (std == 0) -> todos 1.0
        let const_rel = relativize(&[7.5, 7.5, 7.5, 7.5], &config).unwrap();
        assert_eq!(const_rel, vec![1.0, 1.0, 1.0, 1.0]);

        // Población de tamaño 1 (N < 2): caso nulo explícito
        let single_state =
            vec![State::new(Tick::ZERO, SimDuration::ZERO, vec![2.0, -3.0]).unwrap()];
        let mut rng = DeterministicRng::new(7);
        let single_pairs =
            pair_companions(&single_state, |s| s.values().to_vec(), &mut rng).unwrap();
        assert_eq!(single_pairs.len(), 1);
        assert_eq!(single_pairs[0].companion(), None);
        assert_eq!(single_pairs[0].distance(), 0.0);

        // Distancia cero (estados idénticos) y recompensas negativas
        let identical_states = vec![
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 1.0]).unwrap(),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 1.0]).unwrap(),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 1.0]).unwrap(),
        ];
        let zero_dist_pairs =
            pair_companions(&identical_states, |s| s.values().to_vec(), &mut rng).unwrap();
        assert!(zero_dist_pairs.iter().all(|p| p.distance() == 0.0));
        assert!(
            zero_dist_pairs
                .iter()
                .all(|p| p.companion().is_some() && p.companion().unwrap() != p.walker())
        );

        let distances: Vec<f64> = zero_dist_pairs.iter().map(|p| p.distance()).collect();
        let neg_rewards = [-10.0, -5.0, -1.0];
        let terminals = [Terminal::Ongoing; 3];
        let vr = compute_virtual_rewards(&distances, &neg_rewards, &terminals, &config).unwrap();
        assert!(vr.iter().all(|&v| v.is_finite() && v > 0.0));
        assert!(vr[2] > vr[1] && vr[1] > vr[0]);

        // Todos en fallo terminal -> VR = 0.0 y 0 clonaciones
        let mut pop = WalkerPopulation::from_states(
            vec![
                State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap(),
                State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 1.0]).unwrap(),
            ],
            MasterSeed::new(9),
            0,
            0,
            false,
        )
        .unwrap();
        let all_fail = [
            Terminal::Terminated(TerminalReason::Failure),
            Terminal::Terminated(TerminalReason::InvariantViolated),
        ];
        let vr_fail =
            compute_virtual_rewards(&[1.0, 1.0], &[1.0, 2.0], &all_fail, &config).unwrap();
        assert_eq!(vr_fail, vec![0.0, 0.0]);
        assert_eq!(cloning_probability(0.0, 0.0).unwrap(), 0.0);
        assert_eq!(cloning_probability(0.0, 2.5).unwrap(), 1.0);
        assert_eq!(cloning_probability(3.0, 2.0).unwrap(), 0.0);

        // Si forzamos un estado failure-terminal en WalkerPopulation via step_population:
        let model = TwoActionStepModel::new();
        let left = Action::discrete_in_space(0, model.action_space()).unwrap();
        let dt = StepDt::from_secs_f64(0.1).unwrap();
        let mut budget = BudgetTracker::new(PlanBudget::new(10, None));
        pop.step_population(&model, &[left.clone(), left], dt, &mut budget, |_, _, _| {
            -1.0
        })
        .unwrap();
        assert!(pop.all_terminal());
        let all_fail_report = resample_population(&mut pop, &[0.0, 0.0], &mut rng, 1).unwrap();
        assert_eq!(all_fail_report.cloned_count(), 0);
        assert_eq!(all_fail_report.skipped_failed_companion(), 2);

        let mut pop_single = WalkerPopulation::new(
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap(),
            1,
            MasterSeed::new(1),
            0,
            0,
            false,
        )
        .unwrap();
        let single_report = resample_population(&mut pop_single, &[1.0], &mut rng, 1).unwrap();
        assert_eq!(single_report.cloned_count(), 0);
    }

    #[test]
    fn ac2_rejects_non_finite_and_invalid_inputs() {
        let config = FmcScoringConfig::default();

        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                FmcScoringConfig::new(bad, 1.0, 1e-8, 5.0),
                Err(FmcScoringError::InvalidConfig { field: "alpha", .. })
            ));
            assert!(matches!(
                relativize(&[1.0, bad], &config),
                Err(FmcScoringError::NonFiniteValue {
                    context: "relativize.values",
                    index: 1,
                    ..
                })
            ));
            assert!(matches!(
                cloning_probability(bad, 1.0),
                Err(FmcScoringError::NonFiniteValue {
                    context: "cloning_probability.vr_i",
                    ..
                })
            ));
            assert!(matches!(
                cloning_probability(1.0, bad),
                Err(FmcScoringError::NonFiniteValue {
                    context: "cloning_probability.vr_k",
                    ..
                })
            ));

            let states = vec![
                State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0]).unwrap(),
                State::new(Tick::ZERO, SimDuration::ZERO, vec![2.0]).unwrap(),
            ];
            let mut rng = DeterministicRng::new(11);
            assert!(matches!(
                pair_companions(&states, |s| vec![s.values()[0], bad], &mut rng),
                Err(FmcScoringError::NonFiniteValue {
                    context: "pair_companions.features",
                    ..
                })
            ));
        }

        assert!(matches!(
            relativize(&[], &config),
            Err(FmcScoringError::EmptyPopulation)
        ));
        assert!(matches!(
            cloning_probability(-0.1, 1.0),
            Err(FmcScoringError::NegativeVirtualReward { index: 0, .. })
        ));
    }

    #[test]
    fn ac3_cloning_distribution_is_reproducible_with_fixed_seeds() {
        let config = FmcScoringConfig::default();
        let states = vec![
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap(),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![1.0, 0.0]).unwrap(),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 2.0]).unwrap(),
            State::new(Tick::ZERO, SimDuration::ZERO, vec![3.0, 4.0]).unwrap(),
        ];

        let mut pop_a =
            WalkerPopulation::from_states(states.clone(), MasterSeed::new(77), 0, 0, true).unwrap();
        let mut pop_b =
            WalkerPopulation::from_states(states, MasterSeed::new(77), 0, 0, true).unwrap();

        let mut rng_a = DeterministicRng::new(2026);
        let mut rng_b = DeterministicRng::new(2026);

        let score_a =
            score_population(&pop_a, |s| s.values().to_vec(), &config, &mut rng_a).unwrap();
        let score_b =
            score_population(&pop_b, |s| s.values().to_vec(), &config, &mut rng_b).unwrap();
        assert_eq!(score_a, score_b);

        let report_a =
            resample_population(&mut pop_a, score_a.virtual_rewards(), &mut rng_a, 99).unwrap();
        let report_b =
            resample_population(&mut pop_b, score_b.virtual_rewards(), &mut rng_b, 99).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(pop_a, pop_b);

        // Verificación empírica de la probabilidad de clonación p = (2.0 - 0.5) / 2.0 = 0.75
        let p = cloning_probability(0.5, 2.0).unwrap();
        assert!((p - 0.75).abs() < 1e-12);
        let mut emp_rng = DeterministicRng::new(123456);
        let trials = 4000usize;
        let clones = (0..trials).filter(|_| emp_rng.next_f64() < p).count();
        let empirical_rate = (clones as f64) / (trials as f64);
        assert!(
            (empirical_rate - p).abs() < 0.03,
            "empirical rate {empirical_rate} diverged from expected {p}"
        );
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn gog125_ac2_in_place_scoring_is_bit_identical_to_legacy_and_reuses_workspace() {
        let config = FmcScoringConfig::default();
        let mut ws = FmcScratchWorkspace::new();
        let xs = [0.3, 1.7, -0.4, 2.2, 0.9, 5.5, 3.1];
        let failing = [false, false, true, false, false, true, false];
        for penalty in [-1.0, -50.0] {
            let pop = failure_scoring_population(&xs, &failing, penalty);
            let mut rng_a = DeterministicRng::new(42);
            let mut rng_b = DeterministicRng::new(42);
            let legacy =
                score_population(&pop, |s| s.values().to_vec(), &config, &mut rng_a).unwrap();
            score_population_in_place(&pop, |s| s.values(), &config, &mut rng_b, &mut ws).unwrap();

            assert_eq!(legacy.companions(), ws.companions());
            let legacy_d: Vec<f64> = legacy.companions().iter().map(|c| c.distance()).collect();
            assert_eq!(bits(&legacy_d), bits(ws.distances()));
            assert_eq!(
                bits(legacy.relativized_distances()),
                bits(ws.relativized_distances())
            );
            assert_eq!(
                bits(legacy.relativized_rewards()),
                bits(ws.relativized_rewards())
            );
            assert_eq!(bits(legacy.virtual_rewards()), bits(ws.virtual_rewards()));
            // Identical RNG consumption.
            assert_eq!(rng_a.next_u64(), rng_b.next_u64());
        }
    }

    #[test]
    fn gog125_ac2_relativize_viable_into_matches_relativize_viable_on_edge_cases() {
        let config = FmcScoringConfig::default();
        let ongoing = Terminal::Ongoing;
        let failed = Terminal::Terminated(TerminalReason::Failure);
        let cases: Vec<(Vec<f64>, Vec<Terminal>)> = vec![
            (vec![1.0, 2.0, 4.0], vec![ongoing; 3]),
            (
                vec![1.0, 9e9, 4.0, 2.0],
                vec![ongoing, failed, ongoing, ongoing],
            ),
            (vec![3.0, 3.0, 3.0], vec![ongoing; 3]),
            (vec![1.0, 2.0], vec![failed; 2]),
            (vec![7.0, f64::NAN], vec![ongoing, failed]),
        ];
        let mut out = Vec::new();
        for (values, terminals) in cases {
            let expected = relativize_viable(&values, &terminals, &config).unwrap();
            relativize_viable_into(&values, &terminals, &config, &mut out).unwrap();
            assert_eq!(bits(&expected), bits(&out));
        }
        let err = relativize_viable_into(&[1.0, f64::NAN], &[ongoing, ongoing], &config, &mut out);
        assert!(matches!(
            err,
            Err(FmcScoringError::NonFiniteValue { index: 1, .. })
        ));
    }

    #[test]
    fn gog125_unbiased_sampling_is_opt_in_and_valid() {
        let legacy = FmcScoringConfig::default();
        assert!(!legacy.unbiased_sampling());
        let unbiased = legacy.with_unbiased_sampling(true);
        assert!(unbiased.unbiased_sampling());

        let pop = failure_scoring_population(&[0.3, 1.7, 0.4, 2.2, 0.9], &[false; 5], -1.0);
        let mut rng = DeterministicRng::new(7);
        let score = score_population(&pop, |s| s.values().to_vec(), &unbiased, &mut rng).unwrap();
        for pair in score.companions() {
            assert_ne!(pair.companion(), Some(pair.walker()));
            assert!(pair.companion().is_some());
        }

        let mut rng = DeterministicRng::new(1);
        for w in 0..6 {
            let c = sample_companion_index_unbiased(w, 6, &mut rng).unwrap();
            assert!(c < 6 && c != w);
        }
        assert_eq!(sample_companion_index_unbiased(0, 1, &mut rng), None);
    }
}
