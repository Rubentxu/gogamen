//! Cálculo de empowerment de referencia mediante Blahut-Arimoto en mundos discretos (`std`-only).
//!
//! # Definición matemática y alcance
//! - Dado un canal discreto finito `P(O | A_seq)` sobre secuencias de acciones de horizonte `H`
//!   (`|A|^H` entradas posibles) y un espacio de observación discreto de cardinalidad `|O|`, el
//!   **empowerment** de referencia es la **capacidad de canal** en **bits**:
//!   `C = max_{p(a_seq)} I(A_seq; O)`.
//! - El algoritmo de **Blahut-Arimoto** optimiza iterativamente la política de entrada `p*(a_seq)`
//!   hasta satisfacer la cota de dualidad `I_U - I_L <= tolerance_bits` o alcanzar
//!   `max_iterations`.
//!
//! # Límites de escalabilidad
//! - El espacio de secuencias crece exponencialmente como `|A|^H`.
//! - Cada iteración de Blahut-Arimoto cuesta `O(|A|^H * |O|)`, por lo que el coste total está
//!   acotado por `O(iter * |A|^H * |O|)`.
//! - Todo constructor de espacio de secuencias o canal exige una cota explícita
//!   `max_sequence_count` para rechazar explosiones combinatorias antes de asignar memoria.
//!
//! # Diferencia frente al proxy de diversidad
//! - El proxy de diversidad ([`crate::entropy::estimate_trajectory_entropy_proxy`] o la entropía
//!   marginal `H(O)` bajo política uniforme) mide dispersión marginal de estados u observaciones,
//!   **no** control causal. Bajo ruido independiente `P(o | a) = q(o)`, la diversidad marginal
//!   puede ser alta (`H(O) > 0`), mientras que el empowerment es exactamente `0.0` bits.

use std::error::Error;
use std::fmt;

use crate::entropy::{
    DEFAULT_PROBABILITY_SUM_TOLERANCE, EntropyError, TrajectoryEntropyEstimate,
    shannon_entropy_bits_with_tolerance,
};

/// Cota por defecto para el número máximo de secuencias de acción `|A|^H` permitidas.
pub const DEFAULT_MAX_SEQUENCE_COUNT: usize = 4_096;

/// Número máximo de iteraciones por defecto para Blahut-Arimoto.
pub const DEFAULT_BLAHUT_ARIMOTO_MAX_ITERATIONS: usize = 1_000;

/// Tolerancia de convergencia por defecto (en bits) sobre el gap de dualidad `I_U - I_L`.
pub const DEFAULT_BLAHUT_ARIMOTO_TOLERANCE_BITS: f64 = 1e-9;

/// Nota de complejidad documentada en cada [`EmpowermentScalability`].
pub const EMPOWERMENT_COMPLEXITY_NOTE: &str =
    "O(iter * |A|^H * |O|); bounded by max_sequence_count to prevent combinatorial explosion";

/// Nota epistemológica explícita que distingue empowerment de cualquier proxy de diversidad.
pub const EMPOWERMENT_VS_DIVERSITY_NOTE: &str = "El proxy de diversidad mide entropía marginal de estados/observaciones H(O) o de trayectorias bajo una política fija (p. ej. uniforme); el empowerment mide la capacidad de canal C = max_{p(a)} I(A; O) (control causal). No son equivalentes: bajo ruido independiente, la diversidad marginal puede ser alta (H(O) > 0) mientras el empowerment es exactamente 0.0 bits.";

/// Errores explícitos al construir espacios, validar canales o ejecutar Blahut-Arimoto.
#[derive(Debug, Clone, PartialEq)]
pub enum EmpowermentError {
    /// El horizonte `H` debe ser al menos `1`.
    InvalidHorizon,
    /// La cardinalidad del espacio de acciones `|A|` debe ser al menos `1`.
    EmptyActionSpace,
    /// La cardinalidad del espacio de observación `|O|` debe ser al menos `1`.
    EmptyObservationSpace,
    /// La cota `max_sequence_count` debe ser al menos `1`.
    InvalidMaxSequenceCount,
    /// El cálculo de `|A|^H` desbordó `usize`.
    SequenceCountOverflow {
        /// Cardinalidad del espacio de acciones `|A|`.
        action_cardinality: usize,
        /// Horizonte `H`.
        horizon: usize,
    },
    /// El número de secuencias `|A|^H` excede `max_sequence_count`.
    SequenceLimitExceeded {
        /// Cardinalidad del espacio de acciones `|A|`.
        action_cardinality: usize,
        /// Horizonte `H`.
        horizon: usize,
        /// Número total de secuencias `|A|^H`.
        sequence_count: usize,
        /// Cota máxima permitida.
        max_sequence_count: usize,
    },
    /// Configuración inválida de Blahut-Arimoto o del espacio de observación.
    InvalidConfig {
        /// Motivo concreto de invalidez.
        reason: &'static str,
    },
    /// El número de filas de la matriz de canal no coincide con `|A|^H`.
    ChannelRowCountMismatch {
        /// Número esperado de filas (`|A|^H`).
        expected: usize,
        /// Número observado de filas.
        actual: usize,
    },
    /// Una fila de la matriz de canal no coincide en longitud con `|O|`.
    ChannelColumnCountMismatch {
        /// Índice de la fila con longitud incorrecta.
        row: usize,
        /// Número esperado de columnas (`|O|`).
        expected: usize,
        /// Número observado de columnas.
        actual: usize,
    },
    /// El número de etiquetas de observación no coincide con la cardinalidad `|O|`.
    ObservationLabelCountMismatch {
        /// Cardinalidad declarada.
        expected: usize,
        /// Número de etiquetas recibido.
        actual: usize,
    },
    /// Una distribución de probabilidad del canal o entrada es inválida.
    InvalidProbabilityDistribution {
        /// Contexto donde falló la validación.
        context: &'static str,
        /// Índice de fila si aplica.
        row: Option<usize>,
        /// Error subyacente de validación de entropía/probabilidad.
        source: EntropyError,
    },
    /// Error propagado desde [`crate::entropy`].
    Entropy(EntropyError),
}

impl fmt::Display for EmpowermentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHorizon => write!(f, "empowerment horizon H must be >= 1"),
            Self::EmptyActionSpace => write!(f, "action space cardinality |A| must be >= 1"),
            Self::EmptyObservationSpace => {
                write!(f, "observation space cardinality |O| must be >= 1")
            }
            Self::InvalidMaxSequenceCount => write!(f, "max_sequence_count must be >= 1"),
            Self::SequenceCountOverflow {
                action_cardinality,
                horizon,
            } => write!(
                f,
                "action sequence count |A|^H overflowed usize for |A|={action_cardinality}, H={horizon}"
            ),
            Self::SequenceLimitExceeded {
                action_cardinality,
                horizon,
                sequence_count,
                max_sequence_count,
            } => write!(
                f,
                "action sequence count |A|^H = {sequence_count} (|A|={action_cardinality}, H={horizon}) exceeds max_sequence_count={max_sequence_count}"
            ),
            Self::InvalidConfig { reason } => {
                write!(f, "invalid empowerment configuration: {reason}")
            }
            Self::ChannelRowCountMismatch { expected, actual } => write!(
                f,
                "channel row count mismatch: expected {expected} action sequences, got {actual}"
            ),
            Self::ChannelColumnCountMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "channel column count mismatch at row {row}: expected {expected} observations, got {actual}"
            ),
            Self::ObservationLabelCountMismatch { expected, actual } => write!(
                f,
                "observation label count mismatch: expected {expected}, got {actual}"
            ),
            Self::InvalidProbabilityDistribution {
                context,
                row,
                source,
            } => match row {
                Some(idx) => write!(
                    f,
                    "invalid probability distribution in {context} (row {idx}): {source}"
                ),
                None => write!(f, "invalid probability distribution in {context}: {source}"),
            },
            Self::Entropy(err) => write!(f, "entropy error: {err}"),
        }
    }
}

impl Error for EmpowermentError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidProbabilityDistribution { source, .. } => Some(source),
            Self::Entropy(err) => Some(err),
            _ => None,
        }
    }
}

impl From<EntropyError> for EmpowermentError {
    fn from(err: EntropyError) -> Self {
        Self::Entropy(err)
    }
}

/// Especificación explícita del espacio discreto de observación `O`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationSpaceSpec {
    cardinality: usize,
    labels: Vec<String>,
}

impl ObservationSpaceSpec {
    /// Crea un espacio de observación con `cardinality` símbolos (`obs_0`, `obs_1`, ...).
    pub fn with_cardinality(cardinality: usize) -> Result<Self, EmpowermentError> {
        if cardinality == 0 {
            return Err(EmpowermentError::EmptyObservationSpace);
        }
        let labels = (0..cardinality).map(|i| format!("obs_{i}")).collect();
        Ok(Self {
            cardinality,
            labels,
        })
    }

    /// Crea un espacio de observación con etiquetas explícitas no vacías.
    pub fn with_labels(labels: Vec<String>) -> Result<Self, EmpowermentError> {
        if labels.is_empty() {
            return Err(EmpowermentError::EmptyObservationSpace);
        }
        if labels.iter().any(|label| label.trim().is_empty()) {
            return Err(EmpowermentError::InvalidConfig {
                reason: "observation labels cannot be empty or whitespace-only",
            });
        }
        Ok(Self {
            cardinality: labels.len(),
            labels,
        })
    }

    /// Cardinalidad `|O|` del espacio de observación.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    /// Etiquetas ordenadas de las observaciones `0..|O|`.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
}

/// Espacio acotado de secuencias de acciones de horizonte `H` sobre un alfabeto de tamaño `|A|`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionSequenceSpace {
    action_cardinality: usize,
    horizon: usize,
    sequence_count: usize,
    max_sequence_count: usize,
}

impl ActionSequenceSpace {
    /// Valida y construye un espacio de secuencias `|A|^H <= max_sequence_count`.
    pub fn new(
        action_cardinality: usize,
        horizon: usize,
        max_sequence_count: usize,
    ) -> Result<Self, EmpowermentError> {
        if action_cardinality == 0 {
            return Err(EmpowermentError::EmptyActionSpace);
        }
        if horizon == 0 {
            return Err(EmpowermentError::InvalidHorizon);
        }
        if max_sequence_count == 0 {
            return Err(EmpowermentError::InvalidMaxSequenceCount);
        }

        let horizon_u32 =
            u32::try_from(horizon).map_err(|_| EmpowermentError::SequenceCountOverflow {
                action_cardinality,
                horizon,
            })?;

        let sequence_count = action_cardinality.checked_pow(horizon_u32).ok_or(
            EmpowermentError::SequenceCountOverflow {
                action_cardinality,
                horizon,
            },
        )?;

        if sequence_count > max_sequence_count {
            return Err(EmpowermentError::SequenceLimitExceeded {
                action_cardinality,
                horizon,
                sequence_count,
                max_sequence_count,
            });
        }

        Ok(Self {
            action_cardinality,
            horizon,
            sequence_count,
            max_sequence_count,
        })
    }

    /// Cardinalidad del conjunto de acciones por paso (`|A|`).
    pub fn action_cardinality(&self) -> usize {
        self.action_cardinality
    }

    /// Horizonte de planificación (`H`).
    pub fn horizon(&self) -> usize {
        self.horizon
    }

    /// Número total de secuencias (`|A|^H`).
    pub fn sequence_count(&self) -> usize {
        self.sequence_count
    }

    /// Cota máxima configurada para `|A|^H`.
    pub fn max_sequence_count(&self) -> usize {
        self.max_sequence_count
    }

    /// Enumera en orden lexicográfico todas las secuencias de longitud `H` sobre `0..|A|`.
    pub fn enumerate_sequences(&self) -> Vec<Vec<usize>> {
        let mut sequences = Vec::with_capacity(self.sequence_count);
        for idx in 0..self.sequence_count {
            let mut seq = vec![0_usize; self.horizon];
            let mut rem = idx;
            for pos in (0..self.horizon).rev() {
                seq[pos] = rem % self.action_cardinality;
                rem /= self.action_cardinality;
            }
            sequences.push(seq);
        }
        sequences
    }
}

/// Configuración explícita del optimizador Blahut-Arimoto.
#[derive(Debug, Clone, PartialEq)]
pub struct BlahutArimotoConfig {
    /// Iteraciones máximas permitidas (`>= 1`).
    pub max_iterations: usize,
    /// Tolerancia de parada sobre el gap de dualidad `I_U - I_L` en **bits** (`>= 0.0`).
    pub tolerance_bits: f64,
    /// Tolerancia de suma unitaria al validar distribuciones del canal.
    pub probability_sum_tolerance: f64,
    /// Paso de sobre-relajación `eta >= 1.0` (finito). `1.0` (por defecto) es Blahut-Arimoto
    /// estándar; `> 1.0` es opt-in y aplica `log2 p' = log2 p + eta * D_KL` con retroceso
    /// automático a `eta = 1.0` si la cota inferior `I_L` decrece.
    pub step_size_eta: f64,
}

impl Default for BlahutArimotoConfig {
    fn default() -> Self {
        Self {
            max_iterations: DEFAULT_BLAHUT_ARIMOTO_MAX_ITERATIONS,
            tolerance_bits: DEFAULT_BLAHUT_ARIMOTO_TOLERANCE_BITS,
            probability_sum_tolerance: DEFAULT_PROBABILITY_SUM_TOLERANCE,
            step_size_eta: 1.0,
        }
    }
}

impl BlahutArimotoConfig {
    /// Valida que todos los parámetros numéricos de configuración sean finitos y positivos/válidos.
    pub fn validate(&self) -> Result<(), EmpowermentError> {
        if self.max_iterations == 0 {
            return Err(EmpowermentError::InvalidConfig {
                reason: "max_iterations must be >= 1",
            });
        }
        if !self.tolerance_bits.is_finite() || self.tolerance_bits < 0.0 {
            return Err(EmpowermentError::InvalidConfig {
                reason: "tolerance_bits must be finite and >= 0.0",
            });
        }
        if !self.probability_sum_tolerance.is_finite() || self.probability_sum_tolerance < 0.0 {
            return Err(EmpowermentError::InvalidConfig {
                reason: "probability_sum_tolerance must be finite and >= 0.0",
            });
        }
        if !self.step_size_eta.is_finite() || self.step_size_eta < 1.0 {
            return Err(EmpowermentError::InvalidConfig {
                reason: "step_size_eta must be finite and >= 1.0",
            });
        }
        Ok(())
    }
}

/// Canal discreto finito `P(O | A_seq)` con filas indexadas por secuencias de acción de horizonte `H`.
#[derive(Debug, Clone, PartialEq)]
pub struct DiscreteChannel {
    sequence_space: ActionSequenceSpace,
    observation_space: ObservationSpaceSpec,
    transition_matrix: Vec<Vec<f64>>,
}

impl DiscreteChannel {
    /// Construye y valida un canal discreto desde una matriz explícita `|A|^H x |O|`.
    pub fn from_matrix(
        action_cardinality: usize,
        horizon: usize,
        observation_space: ObservationSpaceSpec,
        transition_matrix: Vec<Vec<f64>>,
        max_sequence_count: usize,
        probability_sum_tolerance: f64,
    ) -> Result<Self, EmpowermentError> {
        let sequence_space =
            ActionSequenceSpace::new(action_cardinality, horizon, max_sequence_count)?;
        if transition_matrix.len() != sequence_space.sequence_count() {
            return Err(EmpowermentError::ChannelRowCountMismatch {
                expected: sequence_space.sequence_count(),
                actual: transition_matrix.len(),
            });
        }

        let obs_count = observation_space.cardinality();
        for (row_idx, row) in transition_matrix.iter().enumerate() {
            if row.len() != obs_count {
                return Err(EmpowermentError::ChannelColumnCountMismatch {
                    row: row_idx,
                    expected: obs_count,
                    actual: row.len(),
                });
            }
            validate_probability_row(
                row,
                probability_sum_tolerance,
                "channel transition row",
                Some(row_idx),
            )?;
        }

        Ok(Self {
            sequence_space,
            observation_space,
            transition_matrix,
        })
    }

    /// Construye un canal discreto enumerando todas las secuencias `|A|^H` mediante `model_fn`.
    pub fn from_sequence_model<F>(
        action_cardinality: usize,
        horizon: usize,
        observation_space: ObservationSpaceSpec,
        max_sequence_count: usize,
        probability_sum_tolerance: f64,
        mut model_fn: F,
    ) -> Result<Self, EmpowermentError>
    where
        F: FnMut(&[usize]) -> Result<Vec<f64>, EmpowermentError>,
    {
        let sequence_space =
            ActionSequenceSpace::new(action_cardinality, horizon, max_sequence_count)?;
        let sequences = sequence_space.enumerate_sequences();
        let mut matrix = Vec::with_capacity(sequences.len());
        for seq in &sequences {
            matrix.push(model_fn(seq)?);
        }
        Self::from_matrix(
            action_cardinality,
            horizon,
            observation_space,
            matrix,
            max_sequence_count,
            probability_sum_tolerance,
        )
    }

    /// Construye un canal identidad determinista de `N` estados (`P(o | a) = 1` si `o == a`, `0` en otro caso).
    pub fn identity(
        num_states: usize,
        max_sequence_count: usize,
    ) -> Result<Self, EmpowermentError> {
        let observation_space = ObservationSpaceSpec::with_cardinality(num_states)?;
        let mut matrix = vec![vec![0.0_f64; num_states]; num_states];
        for (idx, row) in matrix.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        Self::from_matrix(
            num_states,
            1,
            observation_space,
            matrix,
            max_sequence_count,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
    }

    /// Construye un canal de ruido independiente (`P(o | a) = q(o)` idéntica para toda secuencia `a`).
    pub fn independent_noise(
        action_cardinality: usize,
        horizon: usize,
        observation_space: ObservationSpaceSpec,
        noise_distribution: Vec<f64>,
        max_sequence_count: usize,
        probability_sum_tolerance: f64,
    ) -> Result<Self, EmpowermentError> {
        let sequence_space =
            ActionSequenceSpace::new(action_cardinality, horizon, max_sequence_count)?;
        if noise_distribution.len() != observation_space.cardinality() {
            return Err(EmpowermentError::ChannelColumnCountMismatch {
                row: 0,
                expected: observation_space.cardinality(),
                actual: noise_distribution.len(),
            });
        }
        validate_probability_row(
            &noise_distribution,
            probability_sum_tolerance,
            "independent noise distribution",
            None,
        )?;
        let matrix = vec![noise_distribution; sequence_space.sequence_count()];
        Self::from_matrix(
            action_cardinality,
            horizon,
            observation_space,
            matrix,
            max_sequence_count,
            probability_sum_tolerance,
        )
    }

    /// Espacio de secuencias de acción asociado al canal.
    pub fn sequence_space(&self) -> &ActionSequenceSpace {
        &self.sequence_space
    }

    /// Espacio de observación asociado al canal.
    pub fn observation_space(&self) -> &ObservationSpaceSpec {
        &self.observation_space
    }

    /// Matriz `P(O | A_seq)` de dimensiones `|A|^H x |O|`.
    pub fn transition_matrix(&self) -> &[Vec<f64>] {
        &self.transition_matrix
    }

    /// Calcula la distribución marginal de observaciones `q(o)` bajo política de entrada uniforme.
    pub fn uniform_marginal_observation_distribution(&self) -> Vec<f64> {
        let seq_count = self.sequence_space.sequence_count();
        let obs_count = self.observation_space.cardinality();
        let weight = 1.0 / (seq_count as f64);
        let mut marginal = vec![0.0_f64; obs_count];
        for row in &self.transition_matrix {
            for (o, &p_oa) in row.iter().enumerate() {
                marginal[o] += weight * p_oa;
            }
        }
        marginal
    }
}

/// Métricas y cotas explícitas de escalabilidad del cálculo de empowerment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmpowermentScalability {
    /// Cardinalidad de acciones por paso `|A|`.
    pub action_cardinality: usize,
    /// Horizonte `H`.
    pub horizon: usize,
    /// Número total de secuencias evaluadas `|A|^H`.
    pub sequence_count: usize,
    /// Cardinalidad del espacio de observación `|O|`.
    pub observation_cardinality: usize,
    /// Cota máxima configurada para `|A|^H`.
    pub max_sequence_count: usize,
    /// Número de iteraciones de Blahut-Arimoto ejecutadas.
    pub iterations_executed: usize,
    /// Estimación de operaciones elementales `iter * |A|^H * |O|`.
    pub estimated_operations: u128,
    /// Nota explícita de complejidad asintótica y acotación.
    pub complexity_note: &'static str,
}

/// Reporte completo del cálculo de empowerment de referencia en bits.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpowermentReport {
    /// Capacidad de canal `C = max_{p(a)} I(A; O)` en **bits**.
    pub capacity_bits: f64,
    /// Información mutua `I(A; O)` bajo política uniforme en **bits**.
    pub uniform_mutual_information_bits: f64,
    /// Política de entrada óptima `p*(a_seq)` sobre las `|A|^H` secuencias.
    pub optimal_input_distribution: Vec<f64>,
    /// Distribución marginal inducida de observaciones `q*(o)` bajo `p*(a_seq)`.
    pub induced_output_distribution: Vec<f64>,
    /// Horizonte `H` de la secuencia de acciones.
    pub horizon: usize,
    /// Espacio de observación documentado.
    pub observation_space: ObservationSpaceSpec,
    /// Indica si el algoritmo alcanzó el criterio `duality_gap_bits <= tolerance_bits`.
    pub converged: bool,
    /// Gap de dualidad `I_U - I_L` al terminar, en **bits**.
    pub duality_gap_bits: f64,
    /// Traza de la cota inferior `I_L(p_k)` en bits, una entrada por iteración evaluada.
    pub lower_bound_trace_bits: Vec<f64>,
    /// Límites y métricas de escalabilidad del cálculo.
    pub scalability: EmpowermentScalability,
}

/// Comparación explícita entre empowerment (`C = max I(A; O)`) y proxy de diversidad (`H(O)` /
/// `estimate_trajectory_entropy_proxy`) sin equipararlos.
#[derive(Debug, Clone, PartialEq)]
pub struct EmpowermentVsDiversityComparison {
    /// Capacidad de canal `C = max_{p(a)} I(A; O)` en **bits**.
    pub empowerment_bits: f64,
    /// Entropía marginal de observaciones `H(O)` bajo política uniforme en **bits**.
    pub uniform_marginal_observation_entropy_bits: f64,
    /// Proxy empírico de entropía de trayectorias si fue proporcionado, en **bits**.
    pub trajectory_entropy_proxy_bits: Option<f64>,
    /// Diferencia absoluta entre la métrica de diversidad de referencia y el empowerment en **bits**.
    pub divergence_bits: f64,
    /// Verdadero cuando existe diversidad positiva apreciable con empowerment nulo o claramente inferior.
    pub diverges_under_noise_or_uncontrolled_dynamics: bool,
    /// Nota epistemológica que prohíbe equiparar ambos conceptos.
    pub epistemological_note: &'static str,
}

/// Suelo numérico `log2` de los pesos de entrada (evita ceros irreversibles por picos transitorios).
pub const MIN_LOG2_PROB: f64 = -1022.0;

/// Calcula la capacidad de canal `C = max_{p(a)} I(A; O)` en **bits** mediante Blahut-Arimoto.
///
/// Los pesos de entrada se mantienen en dominio `log2` con normalización logsumexp y suelo
/// [`MIN_LOG2_PROB`]; ninguna secuencia con soporte inicial puede quedar a 0.0 de forma irreversible.
pub fn compute_reference_empowerment(
    channel: &DiscreteChannel,
    config: &BlahutArimotoConfig,
) -> Result<EmpowermentReport, EmpowermentError> {
    run_blahut_arimoto(channel, config, None)
}

/// Núcleo de Blahut-Arimoto; `initial_p` permite sembrar la distribución inicial (tests/diagnóstico).
fn run_blahut_arimoto(
    channel: &DiscreteChannel,
    config: &BlahutArimotoConfig,
    initial_p: Option<&[f64]>,
) -> Result<EmpowermentReport, EmpowermentError> {
    config.validate()?;

    let seq_count = channel.sequence_space().sequence_count();
    let obs_count = channel.observation_space().cardinality();
    let matrix = channel.transition_matrix();

    let uniform_p = vec![1.0_f64 / (seq_count as f64); seq_count];
    let uniform_q = compute_output_distribution(&uniform_p, matrix, obs_count);
    let uniform_mutual_information_bits =
        compute_mutual_information_bits(&uniform_p, matrix, &uniform_q);

    if seq_count == 1 || obs_count == 1 {
        return Ok(EmpowermentReport {
            capacity_bits: 0.0,
            uniform_mutual_information_bits: 0.0,
            optimal_input_distribution: uniform_p,
            induced_output_distribution: uniform_q,
            horizon: channel.sequence_space().horizon(),
            observation_space: channel.observation_space().clone(),
            converged: true,
            duality_gap_bits: 0.0,
            lower_bound_trace_bits: Vec::new(),
            scalability: build_scalability(channel, 1),
        });
    }

    let mut log2_p: Vec<f64> = match initial_p {
        Some(init) => {
            let raw: Vec<f64> = init.iter().map(|&v| v.log2()).collect();
            normalize_log2(&raw)
        }
        None => vec![-(seq_count as f64).log2(); seq_count],
    };
    let mut p = project_log2(&log2_p);
    let mut converged = false;
    let mut duality_gap_bits = f64::INFINITY;
    let mut iterations_executed = 0_usize;
    let mut trace = Vec::new();

    for iter in 1..=config.max_iterations {
        iterations_executed = iter;
        let cur = evaluate_input(&p, matrix, obs_count);
        trace.push(cur.i_l);

        duality_gap_bits = (cur.i_u - cur.i_l).max(0.0);
        if duality_gap_bits <= config.tolerance_bits {
            converged = true;
            break;
        }

        let step = |eta: f64| -> Vec<f64> {
            let raw: Vec<f64> = log2_p
                .iter()
                .zip(&cur.d_kl)
                .map(|(&l, &d)| {
                    if d.is_finite() {
                        l + eta * (d - cur.i_u)
                    } else {
                        l
                    }
                })
                .collect();
            normalize_log2(&raw)
        };

        let mut next_log2 = step(config.step_size_eta);
        let mut next_p = project_log2(&next_log2);
        if config.step_size_eta > 1.0 {
            let cand = evaluate_input(&next_p, matrix, obs_count);
            if !cand.i_l.is_finite() || cand.i_l < cur.i_l {
                next_log2 = step(1.0);
                next_p = project_log2(&next_log2);
            }
        }
        if next_log2.iter().any(|v| !v.is_finite()) {
            break;
        }
        log2_p = next_log2;
        p = next_p;
    }

    let induced_output_distribution = compute_output_distribution(&p, matrix, obs_count);
    let capacity_bits = compute_mutual_information_bits(&p, matrix, &induced_output_distribution);

    Ok(EmpowermentReport {
        capacity_bits,
        uniform_mutual_information_bits,
        optimal_input_distribution: p,
        induced_output_distribution,
        horizon: channel.sequence_space().horizon(),
        observation_space: channel.observation_space().clone(),
        converged,
        duality_gap_bits,
        lower_bound_trace_bits: trace,
        scalability: build_scalability(channel, iterations_executed),
    })
}

/// Compara explícitamente el empowerment calculado frente al proxy de diversidad marginal
/// (y opcionalmente un [`TrajectoryEntropyEstimate`] obtenido con
/// [`crate::entropy::estimate_trajectory_entropy_proxy`]) sin equipararlos.
pub fn compare_empowerment_with_diversity_proxy(
    channel: &DiscreteChannel,
    empowerment: &EmpowermentReport,
    trajectory_proxy: Option<&TrajectoryEntropyEstimate>,
    probability_sum_tolerance: f64,
) -> Result<EmpowermentVsDiversityComparison, EmpowermentError> {
    let uniform_marginal = channel.uniform_marginal_observation_distribution();
    let uniform_marginal_observation_entropy_bits =
        shannon_entropy_bits_with_tolerance(&uniform_marginal, probability_sum_tolerance)?;
    let trajectory_entropy_proxy_bits = trajectory_proxy.map(|est| est.entropy_bits());
    let reference_diversity_bits =
        trajectory_entropy_proxy_bits.unwrap_or(uniform_marginal_observation_entropy_bits);
    let divergence_bits = (reference_diversity_bits - empowerment.capacity_bits).abs();

    let tol = 1e-6_f64;
    let diverges_under_noise_or_uncontrolled_dynamics =
        reference_diversity_bits > tol && divergence_bits > tol;

    Ok(EmpowermentVsDiversityComparison {
        empowerment_bits: empowerment.capacity_bits,
        uniform_marginal_observation_entropy_bits,
        trajectory_entropy_proxy_bits,
        divergence_bits,
        diverges_under_noise_or_uncontrolled_dynamics,
        epistemological_note: EMPOWERMENT_VS_DIVERSITY_NOTE,
    })
}

fn validate_probability_row(
    probs: &[f64],
    tolerance: f64,
    context: &'static str,
    row: Option<usize>,
) -> Result<(), EmpowermentError> {
    shannon_entropy_bits_with_tolerance(probs, tolerance)
        .map(|_| ())
        .map_err(|source| EmpowermentError::InvalidProbabilityDistribution {
            context,
            row,
            source,
        })
}

struct InputEvaluation {
    d_kl: Vec<f64>,
    i_l: f64,
    i_u: f64,
}

fn evaluate_input(p: &[f64], matrix: &[Vec<f64>], obs_count: usize) -> InputEvaluation {
    let q = compute_output_distribution(p, matrix, obs_count);
    let mut d_kl = vec![0.0_f64; matrix.len()];
    let mut i_l = 0.0_f64;
    let mut i_u = f64::NEG_INFINITY;
    for (a, row) in matrix.iter().enumerate() {
        let mut kl = 0.0_f64;
        for (o, &p_oa) in row.iter().enumerate() {
            if p_oa > 0.0 {
                if q[o] > 0.0 {
                    kl += p_oa * (p_oa / q[o]).log2();
                } else {
                    kl = f64::INFINITY;
                    break;
                }
            }
        }
        d_kl[a] = kl;
        if p[a] > 0.0 {
            i_l += p[a] * kl;
        }
        if kl > i_u {
            i_u = kl;
        }
    }
    InputEvaluation { d_kl, i_l, i_u }
}

/// Normaliza pesos en dominio `log2` (logsumexp base 2) y aplica el suelo [`MIN_LOG2_PROB`].
fn normalize_log2(raw: &[f64]) -> Vec<f64> {
    let max = raw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![f64::NAN; raw.len()];
    }
    let sum: f64 = raw.iter().map(|&x| (x - max).exp2()).sum();
    let lse = max + sum.log2();
    raw.iter().map(|&x| (x - lse).max(MIN_LOG2_PROB)).collect()
}

fn project_log2(log2_p: &[f64]) -> Vec<f64> {
    log2_p.iter().map(|&l| l.exp2()).collect()
}

fn compute_output_distribution(p: &[f64], matrix: &[Vec<f64>], obs_count: usize) -> Vec<f64> {
    let mut q = vec![0.0_f64; obs_count];
    for (a, row) in matrix.iter().enumerate() {
        let pa = p[a];
        if pa > 0.0 {
            for (o, &p_oa) in row.iter().enumerate() {
                q[o] += pa * p_oa;
            }
        }
    }
    q
}

fn compute_mutual_information_bits(p: &[f64], matrix: &[Vec<f64>], q: &[f64]) -> f64 {
    let mut mi = 0.0_f64;
    for (a, row) in matrix.iter().enumerate() {
        let pa = p[a];
        if pa <= 0.0 {
            continue;
        }
        for (o, &p_oa) in row.iter().enumerate() {
            if p_oa > 0.0 && q[o] > 0.0 {
                mi += pa * p_oa * (p_oa / q[o]).log2();
            }
        }
    }
    if mi <= 0.0 { 0.0 } else { mi }
}

fn build_scalability(
    channel: &DiscreteChannel,
    iterations_executed: usize,
) -> EmpowermentScalability {
    let seq_space = channel.sequence_space();
    let seq_count = seq_space.sequence_count();
    let obs_count = channel.observation_space().cardinality();
    let estimated_operations =
        (iterations_executed as u128) * (seq_count as u128) * (obs_count as u128);
    EmpowermentScalability {
        action_cardinality: seq_space.action_cardinality(),
        horizon: seq_space.horizon(),
        sequence_count: seq_count,
        observation_cardinality: obs_count,
        max_sequence_count: seq_space.max_sequence_count(),
        iterations_executed,
        estimated_operations,
        complexity_note: EMPOWERMENT_COMPLEXITY_NOTE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entropy::{
        BinSpec, BinningVersion, DEFAULT_PROBABILITY_SUM_TOLERANCE, EntropyProxyKind,
        TrajectoryStep, TrajectorySummary, estimate_trajectory_entropy_proxy,
    };
    use crate::units::Tick;

    #[test]
    fn ac1_identity_channel_of_n_states_produces_log2_n_bits() {
        let config = BlahutArimotoConfig::default();

        for (n, expected_bits) in [(4_usize, 2.0_f64), (8_usize, 3.0_f64)] {
            let channel =
                DiscreteChannel::identity(n, DEFAULT_MAX_SEQUENCE_COUNT).expect("identity channel");
            let report =
                compute_reference_empowerment(&channel, &config).expect("blahut-arimoto succeeds");

            assert!(report.converged, "expected convergence for N={n}");
            assert!(
                (report.capacity_bits - expected_bits).abs() < 1e-9,
                "expected {expected_bits} bits for N={n}, got {}",
                report.capacity_bits
            );
            assert_eq!(report.optimal_input_distribution.len(), n);
            let expected_prob = 1.0 / (n as f64);
            for &p in &report.optimal_input_distribution {
                assert!((p - expected_prob).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn ac1_independent_noise_channel_produces_zero_bits() {
        let config = BlahutArimotoConfig::default();
        let obs_space = ObservationSpaceSpec::with_cardinality(4).expect("obs space");
        let noise = vec![0.4, 0.3, 0.2, 0.1];
        let channel = DiscreteChannel::independent_noise(
            3,
            2,
            obs_space,
            noise,
            DEFAULT_MAX_SEQUENCE_COUNT,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
        .expect("independent noise channel");

        let report =
            compute_reference_empowerment(&channel, &config).expect("blahut-arimoto succeeds");
        assert!(report.converged);
        assert!(
            report.capacity_bits.abs() < 1e-12,
            "expected 0.0 bits for independent noise, got {}",
            report.capacity_bits
        );
    }

    #[test]
    fn ac1_asymmetric_z_channel_finds_non_uniform_optimal_input() {
        let config = BlahutArimotoConfig::default();
        let obs_space = ObservationSpaceSpec::with_cardinality(2).expect("obs space");
        // Canal Z con cruce 0.5: P(0|0)=1, P(1|0)=0; P(0|1)=0.5, P(1|1)=0.5.
        // Capacidad analítica: log2(5/4) ≈ 0.32192809488736235 bits con p*(0)=0.6, p*(1)=0.4.
        let channel = DiscreteChannel::from_matrix(
            2,
            1,
            obs_space,
            vec![vec![1.0, 0.0], vec![0.5, 0.5]],
            DEFAULT_MAX_SEQUENCE_COUNT,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
        .expect("z-channel");

        let report =
            compute_reference_empowerment(&channel, &config).expect("blahut-arimoto succeeds");
        let expected_capacity = (5.0_f64 / 4.0_f64).log2();
        assert!(report.converged);
        assert!(
            (report.capacity_bits - expected_capacity).abs() < 1e-6,
            "expected {expected_capacity}, got {}",
            report.capacity_bits
        );
        assert!(report.capacity_bits > report.uniform_mutual_information_bits);
        assert!((report.optimal_input_distribution[0] - 0.6).abs() < 1e-4);
        assert!((report.optimal_input_distribution[1] - 0.4).abs() < 1e-4);
    }

    #[test]
    fn ac2_diversity_proxy_and_empowerment_diverge_under_independent_noise() {
        let config = BlahutArimotoConfig::default();
        let obs_space = ObservationSpaceSpec::with_cardinality(4).expect("obs space");
        let uniform_noise = vec![0.25, 0.25, 0.25, 0.25];
        let channel = DiscreteChannel::independent_noise(
            4,
            1,
            obs_space,
            uniform_noise,
            DEFAULT_MAX_SEQUENCE_COUNT,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
        .expect("uniform noise channel");

        let empowerment = compute_reference_empowerment(&channel, &config).expect("empowerment");

        // Construimos trayectorias sintéticas generadas por ruido uniforme en 4 bins.
        let bin_spec = BinSpec::uniform("noise_bins", BinningVersion::V1_0, vec![(0.0, 4.0, 4)])
            .expect("bin spec");
        let trajectories: Vec<TrajectorySummary> = (0..4)
            .map(|obs_idx| {
                TrajectorySummary::new(vec![
                    TrajectoryStep::new(Tick::new(1), vec![(obs_idx as f64) + 0.5]).expect("step"),
                ])
                .expect("trajectory")
            })
            .collect();
        let trajectory_proxy = estimate_trajectory_entropy_proxy(
            &trajectories,
            &bin_spec,
            EntropyProxyKind::FinalStateProxy,
        )
        .expect("proxy");

        let comparison = compare_empowerment_with_diversity_proxy(
            &channel,
            &empowerment,
            Some(&trajectory_proxy),
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
        .expect("comparison");

        assert!(comparison.empowerment_bits.abs() < 1e-12);
        assert!((comparison.uniform_marginal_observation_entropy_bits - 2.0).abs() < 1e-9);
        assert_eq!(comparison.trajectory_entropy_proxy_bits, Some(2.0));
        assert!((comparison.divergence_bits - 2.0).abs() < 1e-9);
        assert!(comparison.diverges_under_noise_or_uncontrolled_dynamics);
        assert_eq!(
            comparison.epistemological_note,
            EMPOWERMENT_VS_DIVERSITY_NOTE
        );
    }

    #[test]
    fn ac3_documents_horizon_observation_policy_and_rejects_sequence_explosion() {
        // Rechazo explícito cuando |A|^H excede max_sequence_count.
        let err = ActionSequenceSpace::new(4, 3, 32).expect_err("4^3 = 64 > 32 must fail");
        assert_eq!(
            err,
            EmpowermentError::SequenceLimitExceeded {
                action_cardinality: 4,
                horizon: 3,
                sequence_count: 64,
                max_sequence_count: 32,
            }
        );

        // Canal multi-horizonte (|A|=2, H=2 => 4 secuencias) que mapea cada secuencia a una de 4 observaciones.
        let obs_space = ObservationSpaceSpec::with_labels(vec![
            "north_west".to_string(),
            "north_east".to_string(),
            "south_west".to_string(),
            "south_east".to_string(),
        ])
        .expect("obs labels");

        let channel = DiscreteChannel::from_sequence_model(
            2,
            2,
            obs_space.clone(),
            16,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
            |seq| {
                let idx = seq[0] * 2 + seq[1];
                let mut row = vec![0.0_f64; 4];
                row[idx] = 1.0;
                Ok(row)
            },
        )
        .expect("horizon 2 channel");

        let report = compute_reference_empowerment(&channel, &BlahutArimotoConfig::default())
            .expect("empowerment");

        assert_eq!(report.horizon, 2);
        assert_eq!(report.observation_space, obs_space);
        assert_eq!(report.optimal_input_distribution.len(), 4);
        assert_eq!(report.scalability.action_cardinality, 2);
        assert_eq!(report.scalability.horizon, 2);
        assert_eq!(report.scalability.sequence_count, 4);
        assert_eq!(report.scalability.observation_cardinality, 4);
        assert_eq!(report.scalability.max_sequence_count, 16);
        assert_eq!(
            report.scalability.estimated_operations,
            (report.scalability.iterations_executed as u128) * 4 * 4
        );
        assert_eq!(
            report.scalability.complexity_note,
            EMPOWERMENT_COMPLEXITY_NOTE
        );
        assert!((report.capacity_bits - 2.0).abs() < 1e-9);
    }

    fn quasi_redundant_channel() -> DiscreteChannel {
        let obs_space = ObservationSpaceSpec::with_cardinality(3).expect("obs space");
        DiscreteChannel::from_matrix(
            4,
            1,
            obs_space,
            vec![
                vec![0.98, 0.01, 0.01],
                vec![0.97, 0.02, 0.01],
                vec![0.01, 0.98, 0.01],
                vec![0.01, 0.97, 0.02],
            ],
            DEFAULT_MAX_SEQUENCE_COUNT,
            DEFAULT_PROBABILITY_SUM_TOLERANCE,
        )
        .expect("quasi-redundant channel")
    }

    #[test]
    fn gog123_ac1_near_underflow_sequence_recovers_exact_capacity() {
        let channel = DiscreteChannel::identity(3, DEFAULT_MAX_SEQUENCE_COUNT).expect("identity");
        let init = [1.0 - 2e-300, 1e-300, 1e-300];
        let expected = 3.0_f64.log2();

        for eta in [1.0, 8.0] {
            let config = BlahutArimotoConfig {
                step_size_eta: eta,
                max_iterations: 10_000,
                ..BlahutArimotoConfig::default()
            };
            let report = run_blahut_arimoto(&channel, &config, Some(&init)).expect("ba");
            assert!(report.converged, "eta={eta}");
            assert!(
                (report.capacity_bits - expected).abs() < 1e-6,
                "eta={eta}: {}",
                report.capacity_bits
            );
            for &w in &report.optimal_input_distribution {
                assert!((w - 1.0 / 3.0).abs() < 1e-6, "eta={eta}: weight {w}");
            }
        }
    }

    #[test]
    fn gog123_ac1_large_step_transient_peak_does_not_zero_sequence() {
        // Con eta grande el factor 2^(eta*(d-max)) cae por debajo de 2^-1075 en una iteración;
        // el suelo log2 mantiene el soporte y la secuencia se recupera.
        let channel = DiscreteChannel::identity(2, DEFAULT_MAX_SEQUENCE_COUNT).expect("identity");
        let config = BlahutArimotoConfig {
            step_size_eta: 1_000.0,
            max_iterations: 10_000,
            ..BlahutArimotoConfig::default()
        };
        let report =
            run_blahut_arimoto(&channel, &config, Some(&[1.0 - 1e-300, 1e-300])).expect("ba");
        assert!((report.capacity_bits - 1.0).abs() < 1e-6);
        assert!(report.optimal_input_distribution.iter().all(|&w| w > 0.4));
    }

    #[test]
    fn gog123_ac2_adaptive_step_is_monotone_and_not_slower_than_standard() {
        let channel = quasi_redundant_channel();
        let standard = compute_reference_empowerment(
            &channel,
            &BlahutArimotoConfig {
                tolerance_bits: 1e-9,
                max_iterations: 100_000,
                ..BlahutArimotoConfig::default()
            },
        )
        .expect("standard");
        let adaptive = compute_reference_empowerment(
            &channel,
            &BlahutArimotoConfig {
                tolerance_bits: 1e-9,
                max_iterations: 100_000,
                step_size_eta: 1.5,
                ..BlahutArimotoConfig::default()
            },
        )
        .expect("adaptive");

        assert!(standard.converged && adaptive.converged);
        for w in adaptive.lower_bound_trace_bits.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "I_L decreased: {} -> {}", w[0], w[1]);
        }
        assert_eq!(
            adaptive.lower_bound_trace_bits.len(),
            adaptive.scalability.iterations_executed
        );
        assert!(
            adaptive.scalability.iterations_executed < standard.scalability.iterations_executed,
            "adaptive {} > standard {}",
            adaptive.scalability.iterations_executed,
            standard.scalability.iterations_executed
        );
        assert!((adaptive.capacity_bits - standard.capacity_bits).abs() < 1e-9);
    }

    #[test]
    fn gog123_invalid_step_size_is_rejected() {
        for eta in [0.5, f64::NAN, f64::INFINITY] {
            let config = BlahutArimotoConfig {
                step_size_eta: eta,
                ..BlahutArimotoConfig::default()
            };
            assert!(config.validate().is_err());
        }
    }
}
