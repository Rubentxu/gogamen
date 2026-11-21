//! Controles ANTIARTEFACTO para entropía, control causal y proxies (GOG-155, DEM-002).
//!
//! # Qué hace
//! Compara el oráculo exacto sobre [`BoolWorld`] (GOG-154) y proxies muestreados frente a
//! **perturbaciones representacionales inocuas** para falsar "falsas mejoras de inteligencia":
//! - [`permute_labels`]: renombrar/permutar microestados (renombración inyectiva).
//! - [`duplicate_predicate`]: copiar un predicado determinista a un bit nuevo.
//! - [`add_exogenous_noise`]: añadir un bit de ruido exógeno justo, independiente de las acciones.
//!
//! # Magnitudes (todas en bits, ver [`Unit`])
//! - `marginal_entropy`: H(S_T) de la mezcla uniforme sobre secuencias de acciones en bucle abierto.
//! - `conditional_entropy`: H(S_T | A) (media sobre secuencias).
//! - `control_bits`: I(A; S_T) = H(S_T) - H(S_T | A), control causal en bucle abierto con
//!   prior uniforme (cota inferior del empowerment, que maximizaría sobre el prior). El ruido
//!   exógeno sube H(S_T) pero también H(S_T | A) en la misma cantidad, así que I no cambia.
//! - `oracle_final_entropy`: entropía del estado final del oráculo de GOG-154 con política
//!   uniforme por paso (referencia cruzada exacta; renormaliza globalmente, no por secuencia).
//! - `proxy_marginal_entropy`: estimador muestreado (plug-in) de H sobre bins
//!   `microestado / bin_width`, con error estándar por método delta. Depende de las etiquetas
//!   si `bin_width > 1`: eso es justo lo que los controles detectan.
//!
//! El error de muestreo es 0 para magnitudes exactas. No se aplica corrección de sesgo
//! (Miller-Madow); el error reportado es solo la desviación estándar del estimador.
//!
//! # Resultado negativo explícito
//! [`compare_controls`] emite un [`ControlVerdict`]: `RealImprovement` solo si sube el control
//! causal; `ArtifactDetected` si H marginal (exacta o proxy significativo a 3 sigma) sube sin
//! ganancia de control; `NoRealImprovement` en otro caso. Fail-closed: entradas no finitas,
//! límites superados o secuencias sin masa viable devuelven [`ControlError`].
//!
//! # Límites
//! Mundos diminutos; no es empowerment del mundo real. El proxy muestrea la mezcla exacta,
//! así que mide solo error de estimación y sensibilidad a bins, no de simulación.
//! La separación H/control en el mundo con grid de GOG-20 (`freedom_proxy`) se conserva allí;
//! este módulo no lo duplica y opera solo en `BoolWorld`.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_core::{EntropyError, shannon_entropy_bits_with_tolerance};

use crate::bool_world::{BoolSamplingPolicy, BoolWorld, BoolWorldError, MicroState, OracleLimits};

/// Tolerancia por defecto (bits) para comparar magnitudes.
pub const CONTROL_DEFAULT_TOLERANCE_BITS: f64 = 1e-9;

/// Unidad de las magnitudes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unit {
    /// Bits (logaritmo base 2).
    Bits,
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bits => write!(f, "bits"),
        }
    }
}

/// Errores tipados (fail-closed).
#[derive(Debug, Clone, PartialEq)]
pub enum ControlError {
    /// Error del mundo/oráculo.
    World(BoolWorldError),
    /// Error de entropía.
    Entropy(EntropyError),
    /// Configuración inválida.
    InvalidConfig(&'static str),
    /// Demasiadas secuencias de acciones.
    TooManySequences {
        /// Límite configurado.
        limit: usize,
    },
    /// Una secuencia de acciones no deja masa viable.
    NoViableMass,
    /// Valor no finito o inconsistente (p. ej. información mutua negativa fuera de tolerancia).
    Inconsistent(&'static str),
    /// Transformación de representación inválida.
    InvalidTransform(&'static str),
}

impl fmt::Display for ControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::World(e) => write!(f, "mundo: {e}"),
            Self::Entropy(e) => write!(f, "entropía: {e}"),
            Self::InvalidConfig(r) => write!(f, "configuración inválida: {r}"),
            Self::TooManySequences { limit } => {
                write!(f, "demasiadas secuencias de acciones (límite {limit})")
            }
            Self::NoViableMass => write!(f, "secuencia de acciones sin masa viable"),
            Self::Inconsistent(r) => write!(f, "resultado inconsistente: {r}"),
            Self::InvalidTransform(r) => write!(f, "transformación inválida: {r}"),
        }
    }
}

impl Error for ControlError {}

impl From<BoolWorldError> for ControlError {
    fn from(e: BoolWorldError) -> Self {
        Self::World(e)
    }
}

impl From<EntropyError> for ControlError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Configuración de medida.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlConfig {
    /// Microestado inicial.
    pub initial: MicroState,
    /// Horizonte (>= 1).
    pub horizon: usize,
    /// Máximo de secuencias de acciones `k^horizon`.
    pub max_sequences: usize,
    /// Muestras del proxy (>= 1).
    pub samples: usize,
    /// Ancho de bin del proxy sobre la etiqueta numérica (>= 1; 1 = sin agrupar).
    pub bin_width: u8,
    /// Seed del muestreo (determinista).
    pub seed: u64,
}

impl ControlConfig {
    /// Configuración por defecto con `initial` y `horizon`.
    pub fn new(initial: MicroState, horizon: usize) -> Self {
        Self {
            initial,
            horizon,
            max_sequences: 65_536,
            samples: 20_000,
            bin_width: 1,
            seed: 155,
        }
    }
}

/// Una magnitud con unidad, error de muestreo y resolución de bins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Magnitude {
    /// Valor.
    pub value: f64,
    /// Unidad.
    pub unit: Unit,
    /// Error estándar de muestreo (0 para exactas).
    pub sampling_error: f64,
    /// `true` si es exacta (no muestreada).
    pub exact: bool,
    /// Resolución de bins (ancho sobre la etiqueta; 1 = una etiqueta por bin).
    pub bin_width: u32,
    /// Muestras usadas (0 para exactas).
    pub samples: usize,
}

impl Magnitude {
    fn exact(value: f64) -> Self {
        Self {
            value,
            unit: Unit::Bits,
            sampling_error: 0.0,
            exact: true,
            bin_width: 1,
            samples: 0,
        }
    }
}

/// Magnitudes medidas sobre un mundo.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlMeasures {
    /// H(S_T) exacta de la mezcla uniforme sobre secuencias de acciones.
    pub marginal_entropy: Magnitude,
    /// H(S_T | A) exacta.
    pub conditional_entropy: Magnitude,
    /// I(A; S_T) exacta (control causal, prior uniforme en bucle abierto).
    pub control_bits: Magnitude,
    /// Entropía del estado final del oráculo GOG-154 (política uniforme por paso).
    pub oracle_final_entropy: Magnitude,
    /// Proxy muestreado de H sobre bins.
    pub proxy_marginal_entropy: Magnitude,
    /// Número de secuencias de acciones enumeradas.
    pub sequences: usize,
}

/// Veredicto explícito, incluido el resultado negativo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ControlVerdict {
    /// Sube el control causal: mejora real (no artefacto).
    RealImprovement {
        /// Ganancia de control en bits.
        control_gain_bits: f64,
    },
    /// Sube H marginal (o el proxy) sin ganancia de control: artefacto detectado.
    ArtifactDetected {
        /// Subida de H marginal (la mayor entre exacta y proxy) en bits.
        marginal_gain_bits: f64,
    },
    /// No hay mejora real ni artefacto aparente.
    NoRealImprovement,
}

impl ControlVerdict {
    /// `true` si el resultado es negativo ("no hay mejora real" o artefacto).
    pub fn is_negative(&self) -> bool {
        !matches!(self, Self::RealImprovement { .. })
    }
}

/// Informe comparativo base vs variante.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlReport {
    /// Medidas del mundo base.
    pub baseline: ControlMeasures,
    /// Medidas del mundo variante.
    pub variant: ControlMeasures,
    /// Variación exacta de H marginal (bits).
    pub marginal_delta: f64,
    /// Variación exacta de control causal (bits).
    pub control_delta: f64,
    /// Variación del proxy muestreado (bits).
    pub proxy_delta: f64,
    /// Error estándar combinado de la variación del proxy (bits).
    pub proxy_delta_sigma: f64,
    /// Tolerancia usada (bits).
    pub tolerance: f64,
    /// Unidad de todos los deltas.
    pub unit: Unit,
    /// Veredicto.
    pub verdict: ControlVerdict,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

type Spec = BTreeMap<MicroState, (bool, Vec<Vec<(MicroState, f64)>>)>;

fn specs(world: &BoolWorld) -> Spec {
    world
        .state_specs()
        .map(|v| (v.state, (v.viable, v.actions.to_vec())))
        .collect()
}

/// Distribución final condicionada a una secuencia de acciones en bucle abierto.
fn propagate(
    spec: &Spec,
    initial: MicroState,
    seq: &[usize],
) -> Result<BTreeMap<MicroState, f64>, ControlError> {
    let mut dist: BTreeMap<MicroState, f64> = BTreeMap::new();
    dist.insert(initial, 1.0);
    for &a in seq {
        let mut next: BTreeMap<MicroState, f64> = BTreeMap::new();
        for (s, p) in &dist {
            let (_, acts) = spec.get(s).ok_or(BoolWorldError::InvalidWorld {
                reason: "transición a microestado no declarado",
            })?;
            if acts.is_empty() {
                *next.entry(*s).or_insert(0.0) += p;
                continue;
            }
            for &(d, q) in &acts[a % acts.len()] {
                let viable = spec
                    .get(&d)
                    .ok_or(BoolWorldError::InvalidWorld {
                        reason: "transición a microestado no declarado",
                    })?
                    .0;
                if viable && q > 0.0 {
                    *next.entry(d).or_insert(0.0) += p * q;
                }
            }
        }
        dist = next;
    }
    let mass: f64 = dist.values().sum();
    if !mass.is_finite() || mass <= 0.0 {
        return Err(ControlError::NoViableMass);
    }
    for v in dist.values_mut() {
        *v /= mass;
    }
    Ok(dist)
}

fn entropy(dist: &BTreeMap<MicroState, f64>) -> Result<f64, ControlError> {
    let p: Vec<f64> = dist.values().copied().collect();
    Ok(shannon_entropy_bits_with_tolerance(
        &p,
        CONTROL_DEFAULT_TOLERANCE_BITS,
    )?)
}

/// Mide el mundo: magnitudes exactas, oráculo y proxy muestreado (determinista por seed).
pub fn measure_controls(
    world: &BoolWorld,
    cfg: &ControlConfig,
) -> Result<ControlMeasures, ControlError> {
    if cfg.horizon == 0 {
        return Err(ControlError::InvalidConfig("horizon debe ser >= 1"));
    }
    if cfg.samples == 0 || cfg.bin_width == 0 || cfg.max_sequences == 0 {
        return Err(ControlError::InvalidConfig(
            "samples, bin_width y max_sequences deben ser >= 1",
        ));
    }
    let spec = specs(world);
    match spec.get(&cfg.initial) {
        Some((true, _)) => {}
        _ => return Err(BoolWorldError::InvalidInitialState(cfg.initial).into()),
    }
    let k = spec
        .values()
        .map(|(_, a)| a.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let n_seq = u32::try_from(cfg.horizon)
        .ok()
        .and_then(|h| k.checked_pow(h))
        .filter(|n| *n <= cfg.max_sequences)
        .ok_or(ControlError::TooManySequences {
            limit: cfg.max_sequences,
        })?;

    let w = 1.0 / n_seq as f64;
    let mut mixture: BTreeMap<MicroState, f64> = BTreeMap::new();
    let mut cond_mean = 0.0;
    for idx in 0..n_seq {
        let mut seq = vec![0usize; cfg.horizon];
        let mut r = idx;
        for slot in seq.iter_mut() {
            *slot = r % k;
            r /= k;
        }
        let d = propagate(&spec, cfg.initial, &seq)?;
        cond_mean += w * entropy(&d)?;
        for (s, p) in d {
            *mixture.entry(s).or_insert(0.0) += w * p;
        }
    }
    let h = entropy(&mixture)?;
    let tol = CONTROL_DEFAULT_TOLERANCE_BITS;
    let mut mi = h - cond_mean;
    if !mi.is_finite() || !h.is_finite() || mi < -tol {
        return Err(ControlError::Inconsistent("I(A;S_T) negativa o no finita"));
    }
    mi = mi.max(0.0);

    let oracle = world.oracle(
        cfg.initial,
        &BoolSamplingPolicy::UniformActions,
        cfg.horizon,
        OracleLimits::default(),
    )?;

    // Proxy muestreado sobre bins de etiqueta.
    let cum: Vec<(MicroState, f64)> = mixture
        .iter()
        .scan(0.0, |acc, (s, p)| {
            *acc += p;
            Some((*s, *acc))
        })
        .collect();
    let mut rng = cfg.seed;
    let mut counts: BTreeMap<u8, usize> = BTreeMap::new();
    for _ in 0..cfg.samples {
        let u = (splitmix(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
        let s = cum
            .iter()
            .find(|(_, c)| u < *c)
            .map(|(s, _)| *s)
            .unwrap_or(cum[cum.len() - 1].0);
        *counts.entry(s.0 / cfg.bin_width).or_insert(0) += 1;
    }
    let n = cfg.samples as f64;
    let probs: Vec<f64> = counts.values().map(|c| *c as f64 / n).collect();
    let ph = shannon_entropy_bits_with_tolerance(&probs, 1e-9)?;
    let second: f64 = probs.iter().map(|p| p * p.log2() * p.log2()).sum();
    let sigma = ((second - ph * ph).max(0.0) / n).sqrt();

    Ok(ControlMeasures {
        marginal_entropy: Magnitude::exact(h),
        conditional_entropy: Magnitude::exact(cond_mean),
        control_bits: Magnitude::exact(mi),
        oracle_final_entropy: Magnitude::exact(oracle.final_state_entropy_bits),
        proxy_marginal_entropy: Magnitude {
            value: ph,
            unit: Unit::Bits,
            sampling_error: sigma,
            exact: false,
            bin_width: u32::from(cfg.bin_width),
            samples: cfg.samples,
        },
        sequences: n_seq,
    })
}

/// Compara base y variante y emite el veredicto (incluido el resultado negativo).
pub fn compare_controls(
    baseline: ControlMeasures,
    variant: ControlMeasures,
    tolerance: f64,
) -> Result<ControlReport, ControlError> {
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(ControlError::InvalidConfig("tolerancia no válida"));
    }
    let marginal_delta = variant.marginal_entropy.value - baseline.marginal_entropy.value;
    let control_delta = variant.control_bits.value - baseline.control_bits.value;
    let proxy_delta = variant.proxy_marginal_entropy.value - baseline.proxy_marginal_entropy.value;
    let proxy_delta_sigma = variant
        .proxy_marginal_entropy
        .sampling_error
        .hypot(baseline.proxy_marginal_entropy.sampling_error);
    if ![
        marginal_delta,
        control_delta,
        proxy_delta,
        proxy_delta_sigma,
    ]
    .iter()
    .all(|v| v.is_finite())
    {
        return Err(ControlError::Inconsistent("delta no finito"));
    }
    let proxy_significant = proxy_delta > 3.0 * proxy_delta_sigma + tolerance;
    let marginal_rise = (marginal_delta > tolerance).then_some(marginal_delta);
    let proxy_rise = proxy_significant.then_some(proxy_delta);
    let verdict = if control_delta > tolerance {
        ControlVerdict::RealImprovement {
            control_gain_bits: control_delta,
        }
    } else if marginal_rise.is_some() || proxy_rise.is_some() {
        ControlVerdict::ArtifactDetected {
            marginal_gain_bits: marginal_rise.unwrap_or(0.0).max(proxy_rise.unwrap_or(0.0)),
        }
    } else {
        ControlVerdict::NoRealImprovement
    };
    Ok(ControlReport {
        baseline,
        variant,
        marginal_delta,
        control_delta,
        proxy_delta,
        proxy_delta_sigma,
        tolerance,
        unit: Unit::Bits,
        verdict,
    })
}

/// Reconstruye el mundo aplicando una función inyectiva sobre etiquetas.
fn remap(
    world: &BoolWorld,
    f: impl Fn(MicroState) -> MicroState,
) -> Result<BoolWorld, ControlError> {
    let mut out = BoolWorld::new(world.seed());
    for v in world.state_specs() {
        let acts = v
            .actions
            .iter()
            .map(|a| a.iter().map(|&(d, p)| (f(d), p)).collect())
            .collect();
        out = out
            .with_state(f(v.state), v.viable, acts)
            .map_err(|_| ControlError::InvalidTransform("mapa no inyectivo"))?;
    }
    Ok(out)
}

/// Renombra/permuta etiquetas de microestado. `perm` debe ser total e inyectiva sobre los estados
/// declarados (las etiquetas destino pueden ser nuevas).
pub fn permute_labels(
    world: &BoolWorld,
    perm: &BTreeMap<MicroState, MicroState>,
) -> Result<BoolWorld, ControlError> {
    let declared: Vec<MicroState> = world.state_specs().map(|v| v.state).collect();
    let image: std::collections::BTreeSet<MicroState> = perm.values().copied().collect();
    let total = declared.iter().all(|s| perm.contains_key(s));
    let injective = image.len() == perm.len();
    if !total || !injective {
        return Err(ControlError::InvalidTransform(
            "perm no es una renombración inyectiva total sobre los estados declarados",
        ));
    }
    remap(world, |s| perm[&s])
}

fn bit_unused(world: &BoolWorld, bit: u8) -> bool {
    bit < 8 && world.state_specs().all(|v| v.state.0 & (1 << bit) == 0)
}

/// Copia el predicado `src` a un bit nuevo `dst` (predicado determinista duplicado).
pub fn duplicate_predicate(world: &BoolWorld, src: u8, dst: u8) -> Result<BoolWorld, ControlError> {
    if src >= 8 || src == dst || !bit_unused(world, dst) {
        return Err(ControlError::InvalidTransform(
            "bit origen/destino inválido",
        ));
    }
    remap(world, |s| MicroState(s.0 | (((s.0 >> src) & 1) << dst)))
}

/// Añade un bit exógeno justo `bit`, remuestreado en cada paso e independiente de acciones y estado.
pub fn add_exogenous_noise(world: &BoolWorld, bit: u8) -> Result<BoolWorld, ControlError> {
    if !bit_unused(world, bit) {
        return Err(ControlError::InvalidTransform(
            "bit de ruido en uso o fuera de rango",
        ));
    }
    let m = 1u8 << bit;
    let mut out = BoolWorld::new(world.seed());
    for v in world.state_specs() {
        for n in [0u8, m] {
            let acts = v
                .actions
                .iter()
                .map(|a| {
                    a.iter()
                        .flat_map(|&(d, p)| {
                            [(MicroState(d.0), p / 2.0), (MicroState(d.0 | m), p / 2.0)]
                        })
                        .collect()
                })
                .collect();
            out = out
                .with_state(MicroState(v.state.0 | n), v.viable, acts)
                .map_err(|_| ControlError::InvalidTransform("ruido: estado duplicado"))?;
        }
    }
    Ok(out)
}

/// Ordena políticas por entropía del estado final del oráculo (descendente); empates (dentro de
/// `tolerance`) por índice. Devuelve los índices ordenados.
pub fn rank_policies(
    world: &BoolWorld,
    initial: MicroState,
    horizon: usize,
    policies: &[BoolSamplingPolicy],
    tolerance: f64,
) -> Result<Vec<usize>, ControlError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(ControlError::InvalidConfig("tolerancia no válida"));
    }
    let mut keyed = Vec::with_capacity(policies.len());
    for (i, p) in policies.iter().enumerate() {
        let r = world.oracle(initial, p, horizon, OracleLimits::default())?;
        keyed.push((-(r.final_state_entropy_bits / tolerance).round() as i64, i));
    }
    keyed.sort();
    Ok(keyed.into_iter().map(|(_, i)| i).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const S0: MicroState = MicroState(0);
    const S1: MicroState = MicroState(1);
    const S2: MicroState = MicroState(2);
    const S3: MicroState = MicroState(3);
    const TOL: f64 = 1e-9;

    fn absorbing(w: BoolWorld, s: MicroState) -> BoolWorld {
        w.with_state(s, true, vec![vec![(s, 1.0)]]).unwrap()
    }

    /// Control: s0 tiene 2 acciones deterministas (->1, ->2).
    fn control_world() -> BoolWorld {
        let w = BoolWorld::new(1)
            .with_state(S0, true, vec![vec![(S1, 1.0)], vec![(S2, 1.0)]])
            .unwrap();
        absorbing(absorbing(w, S1), S2)
    }

    /// Ruido puro: una sola acción, sucesores 1/2 y 1/2.
    fn noise_world() -> BoolWorld {
        let w = BoolWorld::new(2)
            .with_state(S0, true, vec![vec![(S1, 0.5), (S2, 0.5)]])
            .unwrap();
        absorbing(absorbing(w, S1), S2)
    }

    /// Fixture P: a0 -> 1; a1 -> {2: 1/2, 3: 1/2}.
    fn policy_world() -> BoolWorld {
        let w = BoolWorld::new(3)
            .with_state(S0, true, vec![vec![(S1, 1.0)], vec![(S2, 0.5), (S3, 0.5)]])
            .unwrap();
        absorbing(absorbing(absorbing(w, S1), S2), S3)
    }

    fn cfg() -> ControlConfig {
        ControlConfig::new(S0, 1)
    }

    fn perm(pairs: &[(u8, u8)]) -> BTreeMap<MicroState, MicroState> {
        pairs
            .iter()
            .map(|&(a, b)| (MicroState(a), MicroState(b)))
            .collect()
    }

    #[test]
    fn gog_155_permutation_keeps_ranking_and_measures() {
        // Fixture P, horizonte 1. Entropía final por política (a mano):
        //  Fixed(a0): {1:1} -> 0 bits; Fixed(a1): {2:.5,3:.5} -> 1 bit;
        //  Uniform: {1:.5, 2:.25, 3:.25} -> 1.5 bits. Ranking: [2, 1, 0 y 3 (empate)].
        // Control (secuencias [0],[1]): H=1.5, H|A=(0+1)/2=.5, I=1.0 bits.
        let w = policy_world();
        let pols = vec![
            BoolSamplingPolicy::Fixed(BTreeMap::from([(S0, 0)])),
            BoolSamplingPolicy::Fixed(BTreeMap::from([(S0, 1)])),
            BoolSamplingPolicy::UniformActions,
            BoolSamplingPolicy::Fixed(BTreeMap::from([(S0, 0)])),
        ];
        let base_rank = rank_policies(&w, S0, 1, &pols, TOL).unwrap();
        assert_eq!(base_rank, vec![2, 1, 0, 3]);
        let p = perm(&[(0, 5), (1, 3), (2, 7), (3, 1)]);
        let wp = permute_labels(&w, &p).unwrap();
        // Las políticas Fixed referencian el estado s0 -> 5.
        let pols_p: Vec<_> = pols
            .iter()
            .map(|pol| match pol {
                BoolSamplingPolicy::Fixed(m) => {
                    BoolSamplingPolicy::Fixed(m.iter().map(|(s, a)| (p[s], *a)).collect())
                }
                other => other.clone(),
            })
            .collect();
        let rank_p = rank_policies(&wp, p[&S0], 1, &pols_p, TOL).unwrap();
        assert_eq!(base_rank, rank_p);

        let a = measure_controls(&w, &cfg()).unwrap();
        let b = measure_controls(&wp, &ControlConfig::new(p[&S0], 1)).unwrap();
        assert!((a.control_bits.value - 1.0).abs() < TOL);
        assert!((a.marginal_entropy.value - 1.5).abs() < TOL);
        assert!((a.control_bits.value - b.control_bits.value).abs() < TOL);
        assert!((a.marginal_entropy.value - b.marginal_entropy.value).abs() < TOL);
        assert!((a.oracle_final_entropy.value - b.oracle_final_entropy.value).abs() < TOL);
        let rep = compare_controls(a, b, TOL).unwrap();
        assert_eq!(rep.verdict, ControlVerdict::NoRealImprovement);
        assert!(!is_bad_perm(&w));
    }

    fn is_bad_perm(w: &BoolWorld) -> bool {
        // No biyección (falta s3, imagen repetida) debe fallar cerrado.
        permute_labels(w, &perm(&[(0, 1), (1, 1), (2, 2)])).is_ok()
    }

    #[test]
    fn gog_155_binned_proxy_is_label_dependent_exact_is_not() {
        // Con bin_width 2 el proxy agrupa {0,1} y {2,3}. Mezcla: 1:.5, 2:.25, 3:.25.
        //  original: bins {1}=.5, {2,3}=.5 -> 1 bit.
        // Tras permutar 1->2, 2->1, 3->3: bins {1}=.25, {2,3}=.75 -> 0.8113 bits.
        // El exacto sigue en 1.5 bits en ambos.
        let w = policy_world();
        let wp = permute_labels(&w, &perm(&[(0, 0), (1, 2), (2, 1), (3, 3)])).unwrap();
        let mut c = cfg();
        c.bin_width = 2;
        let a = measure_controls(&w, &c).unwrap();
        let b = measure_controls(&wp, &c).unwrap();
        assert!(
            (a.proxy_marginal_entropy.value - 1.0).abs()
                < 5.0 * a.proxy_marginal_entropy.sampling_error + 1e-3
        );
        assert!(
            (b.proxy_marginal_entropy.value - 0.811278).abs()
                < 5.0 * b.proxy_marginal_entropy.sampling_error + 1e-3
        );
        assert!(a.proxy_marginal_entropy.value - b.proxy_marginal_entropy.value > 0.1);
        assert_eq!(a.marginal_entropy.value, b.marginal_entropy.value);
        assert_eq!(a.proxy_marginal_entropy.bin_width, 2);
        assert!(a.proxy_marginal_entropy.sampling_error > 0.0);
        assert_eq!(a.marginal_entropy.sampling_error, 0.0);
        // Determinismo por seed.
        assert_eq!(a, measure_controls(&w, &c).unwrap());
    }

    #[test]
    fn gog_155_exogenous_noise_raises_marginal_h_not_control() {
        // Mundo de control, horizonte 1, secuencias [0] y [1].
        //  Base: mezcla {1:.5, 2:.5}: H=1; H|A=0; I=1 bit.
        // Con bit de ruido justo (bit 2): cada destino se parte en dos con p/2.
        //  Mezcla: 4 estados a 1/4 -> H=2 bits; H|A: cada secuencia da 2 estados 1/2 -> 1 bit.
        //  I = 2 - 1 = 1 bit. => H marginal sube +1 bit; control causal +0.
        let w = control_world();
        let wn = add_exogenous_noise(&w, 2).unwrap();
        let a = measure_controls(&w, &cfg()).unwrap();
        let b = measure_controls(&wn, &cfg()).unwrap();
        assert!((a.marginal_entropy.value - 1.0).abs() < TOL);
        assert!((b.marginal_entropy.value - 2.0).abs() < TOL);
        assert!((a.control_bits.value - 1.0).abs() < TOL);
        assert!((b.control_bits.value - 1.0).abs() < TOL);
        let rep = compare_controls(a, b, TOL).unwrap();
        assert!((rep.marginal_delta - 1.0).abs() < TOL);
        assert!(rep.control_delta.abs() < TOL);
        assert_eq!(
            rep.verdict,
            ControlVerdict::ArtifactDetected {
                marginal_gain_bits: rep.marginal_delta.max(rep.proxy_delta)
            }
        );
        assert!(rep.verdict.is_negative());
        assert_eq!(rep.unit, Unit::Bits);
    }

    #[test]
    fn gog_155_pure_noise_has_entropy_without_control() {
        // Ruido puro: 1 secuencia, H=1, H|A=1, I=0. Frente a un mundo determinista (H=0, I=0),
        // H sube +1 pero el control no: artefacto.
        let det = absorbing(BoolWorld::new(4), S0);
        let a = measure_controls(&det, &cfg()).unwrap();
        let b = measure_controls(&noise_world(), &cfg()).unwrap();
        assert!(a.marginal_entropy.value.abs() < TOL);
        assert!((b.marginal_entropy.value - 1.0).abs() < TOL);
        assert!(b.control_bits.value.abs() < TOL);
        let rep = compare_controls(a, b, TOL).unwrap();
        assert!(matches!(
            rep.verdict,
            ControlVerdict::ArtifactDetected { .. }
        ));
    }

    #[test]
    fn gog_155_duplicated_predicate_changes_nothing() {
        // Copiar el bit 0 al bit 3 es una inyección de etiquetas: H, H|A e I idénticas.
        let w = control_world();
        let wd = duplicate_predicate(&w, 0, 3).unwrap();
        let a = measure_controls(&w, &cfg()).unwrap();
        let b = measure_controls(&wd, &cfg()).unwrap();
        assert!((a.control_bits.value - b.control_bits.value).abs() < TOL);
        assert!((a.marginal_entropy.value - b.marginal_entropy.value).abs() < TOL);
        let rep = compare_controls(a, b, TOL).unwrap();
        assert_eq!(rep.verdict, ControlVerdict::NoRealImprovement);
        assert!(duplicate_predicate(&w, 0, 1).is_err());
    }

    #[test]
    fn gog_155_real_improvement_is_reported_positive() {
        // Pasar de ruido puro (I=0) a control real (I=1): mejora real.
        let a = measure_controls(&noise_world(), &cfg()).unwrap();
        let b = measure_controls(&control_world(), &cfg()).unwrap();
        let rep = compare_controls(a, b, TOL).unwrap();
        assert!(matches!(
            rep.verdict,
            ControlVerdict::RealImprovement { control_gain_bits } if (control_gain_bits - 1.0).abs() < TOL
        ));
        assert!(!rep.verdict.is_negative());
    }

    #[test]
    fn gog_155_fail_closed_on_bad_input() {
        let w = control_world();
        let mut c = cfg();
        c.horizon = 0;
        assert!(matches!(
            measure_controls(&w, &c),
            Err(ControlError::InvalidConfig(_))
        ));
        let mut c = ControlConfig::new(S0, 20);
        c.max_sequences = 10;
        assert!(matches!(
            measure_controls(&w, &c),
            Err(ControlError::TooManySequences { .. })
        ));
        assert!(measure_controls(&w, &ControlConfig::new(S3, 1)).is_err());
        let a = measure_controls(&w, &cfg()).unwrap();
        assert!(compare_controls(a.clone(), a, f64::NAN).is_err());
    }
}
