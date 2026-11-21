//! `BoolWorld`: grafo pequeño estocástico de microestados booleanos y oráculo exhaustivo
//! de trayectorias (GOG-154, DEM-001).
//!
//! # Qué hace
//! - [`MicroState`] (un vector de hasta 8 booleanos como máscara `u8`) y [`Trajectory`]
//!   (secuencia ordenada de microestados) son tipos **distintos**: el oráculo distribuye
//!   probabilidad sobre trayectorias y deriva de ahí la distribución del microestado final.
//! - [`BoolWorld::oracle`] enumera exhaustivamente todas las trayectorias con probabilidad
//!   positiva bajo una [`BoolSamplingPolicy`] y un `horizon` explícitos, con límites configurables
//!   ([`OracleLimits`]); al superarlos devuelve [`BoolWorldError::CombinatorialExplosion`]
//!   (sin panic ni OOM: la frontera nunca supera `max_trajectories`).
//! - Las transiciones hacia microestados **inviables** se excluyen del soporte. Esa exclusión
//!   NO es un reward ni una autorización (`ActionGuard` no interviene): solo recorta el
//!   espacio de futuros viables. Las trayectorias viables se renormalizan (suma 1 con
//!   [`OracleLimits::probability_tolerance`]) y la masa excluida se reporta aparte
//!   ([`OracleReport::excluded_mass`]).
//! - Alcanzabilidad ([`Reachability`]) y entropía ([`OracleReport::trajectory_entropy_bits`],
//!   [`OracleReport::final_state_entropy_bits`], vía
//!   [`gogamen_core::shannon_entropy_bits_with_tolerance`]) son campos separados.
//! - [`OracleReport::stable_hash`] es un FNV-1a de 64 bits (mismas constantes que los
//!   checksums de snapshot de `gogamen-core::memory`) sobre esquema, seed, política, horizonte
//!   y resultados, estable entre ejecuciones.
//!
//! # Límites
//! Es un oráculo de laboratorio para mundos diminutos; no es empowerment exacto del mundo
//! real. Las probabilidades se validan en construcción (finitas, `>= 0`, suma 1 por acción).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use gogamen_core::{EntropyError, shannon_entropy_bits_with_tolerance};

/// Versión del esquema incluida en el hash estable.
pub const BOOL_WORLD_SCHEMA_VERSION: u32 = 1;
/// Tolerancia por defecto de normalización.
pub const BOOL_WORLD_DEFAULT_TOLERANCE: f64 = 1e-9;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Microestado: máscara de hasta 8 booleanos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MicroState(pub u8);

/// Trayectoria: secuencia de microestados (longitud `horizon + 1`, incluye el inicial).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Trajectory(pub Vec<MicroState>);

impl Trajectory {
    /// Microestado final.
    pub fn final_state(&self) -> MicroState {
        *self.0.last().expect("trayectoria nunca vacía")
    }
}

/// Política de muestreo de acciones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoolSamplingPolicy {
    /// Acción uniforme entre las disponibles en cada microestado.
    UniformActions,
    /// Índice de acción fijo por microestado (los ausentes usan la acción 0).
    Fixed(BTreeMap<MicroState, usize>),
}

/// Límites configurables del oráculo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OracleLimits {
    /// Máximo de trayectorias vivas simultáneas en la frontera.
    pub max_trajectories: usize,
    /// Máximo de microestados del mundo.
    pub max_states: usize,
    /// Tolerancia de normalización.
    pub probability_tolerance: f64,
}

impl Default for OracleLimits {
    fn default() -> Self {
        Self {
            max_trajectories: 100_000,
            max_states: 256,
            probability_tolerance: BOOL_WORLD_DEFAULT_TOLERANCE,
        }
    }
}

/// Errores tipados del mundo y del oráculo.
#[derive(Debug, Clone, PartialEq)]
pub enum BoolWorldError {
    /// Definición inválida del mundo.
    InvalidWorld {
        /// Motivo.
        reason: &'static str,
    },
    /// Probabilidades de una acción no válidas (negativas, no finitas o suma != 1).
    InvalidProbabilities {
        /// Microestado de origen.
        state: MicroState,
    },
    /// Microestado inicial inexistente o inviable.
    InvalidInitialState(MicroState),
    /// Horizonte cero.
    InvalidHorizon,
    /// Se superó un límite de enumeración.
    CombinatorialExplosion {
        /// Límite superado.
        limit: usize,
        /// Qué se limitaba (`"trajectories"` o `"states"`).
        what: &'static str,
        /// Profundidad alcanzada.
        depth: usize,
    },
    /// Toda la masa de probabilidad cae en microestados inviables.
    NoViableTrajectory,
    /// Error de entropía subyacente.
    Entropy(EntropyError),
}

impl fmt::Display for BoolWorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWorld { reason } => write!(f, "mundo inválido: {reason}"),
            Self::InvalidProbabilities { state } => {
                write!(f, "probabilidades inválidas en el microestado {}", state.0)
            }
            Self::InvalidInitialState(s) => write!(f, "estado inicial inválido: {}", s.0),
            Self::InvalidHorizon => write!(f, "horizonte debe ser >= 1"),
            Self::CombinatorialExplosion { limit, what, depth } => write!(
                f,
                "explosión combinatoria: límite de {what} ({limit}) superado en profundidad {depth}"
            ),
            Self::NoViableTrajectory => write!(f, "no existe ninguna trayectoria viable"),
            Self::Entropy(e) => write!(f, "error de entropía: {e}"),
        }
    }
}

impl Error for BoolWorldError {}

impl From<EntropyError> for BoolWorldError {
    fn from(e: EntropyError) -> Self {
        Self::Entropy(e)
    }
}

/// Alcanzabilidad (separada de la entropía).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reachability {
    /// Microestados viables alcanzables con probabilidad positiva (incluye el inicial).
    pub reachable: BTreeSet<MicroState>,
    /// Número de microestados viables del mundo.
    pub viable_total: usize,
}

impl Reachability {
    /// Fracción alcanzable sobre los viables.
    pub fn fraction(&self) -> f64 {
        self.reachable.len() as f64 / self.viable_total.max(1) as f64
    }
}

/// Resultado del oráculo.
#[derive(Debug, Clone, PartialEq)]
pub struct OracleReport {
    /// Trayectorias viables con probabilidad condicionada a viabilidad (suma 1), ordenadas.
    pub trajectories: Vec<(Trajectory, f64)>,
    /// Distribución del microestado final (suma 1), derivada de las trayectorias.
    pub final_states: Vec<(MicroState, f64)>,
    /// Alcanzabilidad viable.
    pub reachability: Reachability,
    /// Entropía de Shannon (bits) de la distribución de trayectorias.
    pub trajectory_entropy_bits: f64,
    /// Entropía de Shannon (bits) del microestado final.
    pub final_state_entropy_bits: f64,
    /// Masa de probabilidad cruda de trayectorias viables (antes de renormalizar).
    pub viable_mass: f64,
    /// Masa cruda excluida por transiciones a microestados inviables (`1 - viable_mass`).
    pub excluded_mass: f64,
    /// Seed del mundo.
    pub seed: u64,
    /// Horizonte usado.
    pub horizon: usize,
    /// Hash estable FNV-1a de esquema + seed + política + horizonte + resultados.
    pub stable_hash: u64,
}

#[derive(Debug, Clone)]
struct StateSpec {
    viable: bool,
    /// Acciones; cada una es una lista `(destino, probabilidad)`.
    actions: Vec<Vec<(MicroState, f64)>>,
}

/// Vista de solo lectura de un microestado declarado.
#[derive(Debug, Clone, Copy)]
pub struct StateSpecView<'a> {
    /// Microestado.
    pub state: MicroState,
    /// Si es viable.
    pub viable: bool,
    /// Acciones; cada una es una lista `(destino, probabilidad)`.
    pub actions: &'a [Vec<(MicroState, f64)>],
}

/// Grafo estocástico pequeño de microestados booleanos.
#[derive(Debug, Clone)]
pub struct BoolWorld {
    seed: u64,
    states: BTreeMap<MicroState, StateSpec>,
}

impl BoolWorld {
    /// Construye un mundo vacío con `seed` de identificación.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            states: BTreeMap::new(),
        }
    }

    /// Declara un microestado con sus acciones (`(destino, probabilidad)` por acción).
    pub fn with_state(
        mut self,
        state: MicroState,
        viable: bool,
        actions: Vec<Vec<(MicroState, f64)>>,
    ) -> Result<Self, BoolWorldError> {
        if self.states.contains_key(&state) {
            return Err(BoolWorldError::InvalidWorld {
                reason: "microestado duplicado",
            });
        }
        for a in &actions {
            let mut sum = 0.0;
            for &(_, p) in a {
                if !p.is_finite() || p < 0.0 {
                    return Err(BoolWorldError::InvalidProbabilities { state });
                }
                sum += p;
            }
            if (sum - 1.0).abs() > BOOL_WORLD_DEFAULT_TOLERANCE {
                return Err(BoolWorldError::InvalidProbabilities { state });
            }
        }
        self.states.insert(state, StateSpec { viable, actions });
        Ok(self)
    }

    /// Genera un mundo determinista a partir de `seed`: `n` estados viables (`2..=8`), una
    /// acción por estado con dos sucesores y probabilidades diádicas `k/8`.
    pub fn from_seed(seed: u64, n: u8) -> Result<Self, BoolWorldError> {
        if !(2..=8).contains(&n) {
            return Err(BoolWorldError::InvalidWorld {
                reason: "n fuera de 2..=8",
            });
        }
        let mut rng = seed;
        let mut next = move || {
            // splitmix64
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let mut w = Self::new(seed);
        for s in 0..n {
            let a = MicroState((next() % n as u64) as u8);
            let b = MicroState((next() % n as u64) as u8);
            let k = (next() % 7 + 1) as f64;
            let acts = if a == b {
                vec![vec![(a, 1.0)]]
            } else {
                vec![vec![(a, k / 8.0), (b, 1.0 - k / 8.0)]]
            };
            w = w.with_state(MicroState(s), true, acts)?;
        }
        Ok(w)
    }

    /// Seed de identificación del mundo.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Vista de solo lectura de las especificaciones `(microestado, viable, acciones)`,
    /// ordenada por microestado (usada por los controles antiartefacto, GOG-155).
    pub fn state_specs(&self) -> impl Iterator<Item = StateSpecView<'_>> {
        self.states.iter().map(|(s, spec)| StateSpecView {
            state: *s,
            viable: spec.viable,
            actions: &spec.actions,
        })
    }

    /// Distribución efectiva de sucesores en `s` bajo `policy`.
    fn successors(
        &self,
        s: MicroState,
        policy: &BoolSamplingPolicy,
    ) -> Result<Vec<(MicroState, f64)>, BoolWorldError> {
        let spec = self.states.get(&s).ok_or(BoolWorldError::InvalidWorld {
            reason: "transición a microestado no declarado",
        })?;
        if spec.actions.is_empty() {
            return Ok(vec![(s, 1.0)]); // absorbente
        }
        let weighted: Vec<(usize, f64)> = match policy {
            BoolSamplingPolicy::UniformActions => {
                let w = 1.0 / spec.actions.len() as f64;
                (0..spec.actions.len()).map(|i| (i, w)).collect()
            }
            BoolSamplingPolicy::Fixed(m) => {
                let i = m.get(&s).copied().unwrap_or(0);
                if i >= spec.actions.len() {
                    return Err(BoolWorldError::InvalidWorld {
                        reason: "índice de acción fuera de rango",
                    });
                }
                vec![(i, 1.0)]
            }
        };
        let mut out: BTreeMap<MicroState, f64> = BTreeMap::new();
        for (i, wa) in weighted {
            for &(d, p) in &spec.actions[i] {
                if p > 0.0 {
                    *out.entry(d).or_insert(0.0) += wa * p;
                }
            }
        }
        Ok(out.into_iter().collect())
    }

    /// Oráculo exhaustivo de trayectorias desde `initial` con `horizon` pasos.
    pub fn oracle(
        &self,
        initial: MicroState,
        policy: &BoolSamplingPolicy,
        horizon: usize,
        limits: OracleLimits,
    ) -> Result<OracleReport, BoolWorldError> {
        if horizon == 0 {
            return Err(BoolWorldError::InvalidHorizon);
        }
        if self.states.len() > limits.max_states {
            return Err(BoolWorldError::CombinatorialExplosion {
                limit: limits.max_states,
                what: "states",
                depth: 0,
            });
        }
        match self.states.get(&initial) {
            Some(s) if s.viable => {}
            _ => return Err(BoolWorldError::InvalidInitialState(initial)),
        }
        let mut frontier: Vec<(Vec<MicroState>, f64)> = vec![(vec![initial], 1.0)];
        for depth in 1..=horizon {
            let mut next: Vec<(Vec<MicroState>, f64)> = Vec::new();
            for (path, p) in &frontier {
                let last = *path.last().expect("ruta no vacía");
                for (d, q) in self.successors(last, policy)? {
                    // Exclusión de inviables: se descarta del soporte (no es reward).
                    if !self.states.get(&d).is_some_and(|s| s.viable) {
                        continue;
                    }
                    if next.len() >= limits.max_trajectories {
                        return Err(BoolWorldError::CombinatorialExplosion {
                            limit: limits.max_trajectories,
                            what: "trajectories",
                            depth,
                        });
                    }
                    let mut np = path.clone();
                    np.push(d);
                    next.push((np, p * q));
                }
            }
            frontier = next;
            if frontier.is_empty() {
                return Err(BoolWorldError::NoViableTrajectory);
            }
        }
        let viable_mass: f64 = frontier.iter().map(|(_, p)| p).sum();
        if viable_mass <= 0.0 {
            return Err(BoolWorldError::NoViableTrajectory);
        }
        let mut trajectories: Vec<(Trajectory, f64)> = frontier
            .into_iter()
            .map(|(t, p)| (Trajectory(t), p / viable_mass))
            .collect();
        trajectories.sort_by(|a, b| a.0.cmp(&b.0));

        let mut finals: BTreeMap<MicroState, f64> = BTreeMap::new();
        let mut reachable = BTreeSet::new();
        for (t, p) in &trajectories {
            *finals.entry(t.final_state()).or_insert(0.0) += p;
            reachable.extend(t.0.iter().copied());
        }
        // Alcanzables en pasos intermedios con trayectorias que no llegan al horizonte
        // no existen aquí: toda trayectoria viable llega al horizonte.
        let final_states: Vec<(MicroState, f64)> = finals.into_iter().collect();
        let tol = limits.probability_tolerance;
        let tp: Vec<f64> = trajectories.iter().map(|(_, p)| *p).collect();
        let fp: Vec<f64> = final_states.iter().map(|(_, p)| *p).collect();
        let trajectory_entropy_bits = shannon_entropy_bits_with_tolerance(&tp, tol)?;
        let final_state_entropy_bits = shannon_entropy_bits_with_tolerance(&fp, tol)?;
        let viable_total = self.states.values().filter(|s| s.viable).count();

        let mut report = OracleReport {
            trajectories,
            final_states,
            reachability: Reachability {
                reachable,
                viable_total,
            },
            trajectory_entropy_bits,
            final_state_entropy_bits,
            viable_mass,
            excluded_mass: (1.0 - viable_mass).max(0.0),
            seed: self.seed,
            horizon,
            stable_hash: 0,
        };
        report.stable_hash = hash_report(&report, policy);
        Ok(report)
    }
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_report(r: &OracleReport, policy: &BoolSamplingPolicy) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    fnv(&mut h, &BOOL_WORLD_SCHEMA_VERSION.to_le_bytes());
    fnv(&mut h, &r.seed.to_le_bytes());
    fnv(&mut h, &(r.horizon as u64).to_le_bytes());
    match policy {
        BoolSamplingPolicy::UniformActions => fnv(&mut h, &[0]),
        BoolSamplingPolicy::Fixed(m) => {
            fnv(&mut h, &[1]);
            for (s, a) in m {
                fnv(&mut h, &[s.0]);
                fnv(&mut h, &(*a as u64).to_le_bytes());
            }
        }
    }
    for (t, p) in &r.trajectories {
        fnv(&mut h, &(t.0.len() as u64).to_le_bytes());
        for s in &t.0 {
            fnv(&mut h, &[s.0]);
        }
        fnv(&mut h, &p.to_bits().to_le_bytes());
    }
    fnv(&mut h, &r.trajectory_entropy_bits.to_bits().to_le_bytes());
    fnv(&mut h, &r.final_state_entropy_bits.to_bits().to_le_bytes());
    fnv(&mut h, &r.viable_mass.to_bits().to_le_bytes());
    fnv(
        &mut h,
        &(r.reachability.reachable.len() as u64).to_le_bytes(),
    );
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    const S0: MicroState = MicroState(0);
    const S1: MicroState = MicroState(1);
    const S2: MicroState = MicroState(2);
    const S3: MicroState = MicroState(3);
    const EPS: f64 = 1e-12;

    /// Fixture A (moneda justa): 0 -> 1 (1/2), 0 -> 2 (1/2); 1 y 2 absorbentes.
    fn coin() -> BoolWorld {
        BoolWorld::new(7)
            .with_state(S0, true, vec![vec![(S1, 0.5), (S2, 0.5)]])
            .unwrap()
            .with_state(S1, true, vec![vec![(S1, 1.0)]])
            .unwrap()
            .with_state(S2, true, vec![vec![(S2, 1.0)]])
            .unwrap()
    }

    /// Fixture B: 0 -> 1 (1/2), 0 -> 2 (1/4), 0 -> 3 inviable (1/4).
    fn hazard() -> BoolWorld {
        BoolWorld::new(9)
            .with_state(S0, true, vec![vec![(S1, 0.5), (S2, 0.25), (S3, 0.25)]])
            .unwrap()
            .with_state(S1, true, vec![vec![(S1, 1.0)]])
            .unwrap()
            .with_state(S2, true, vec![vec![(S2, 1.0)]])
            .unwrap()
            .with_state(S3, false, vec![])
            .unwrap()
    }

    #[test]
    fn gog_154_coin_fixture_hand_computed() {
        // Cálculo a mano, horizonte 2: trayectorias [0,1,1] y [0,2,2], p = 1/2 cada una.
        // H_tray = -2 * (1/2) * log2(1/2) = 1 bit. Final: {1: 1/2, 2: 1/2} -> 1 bit.
        // Alcanzables: {0,1,2} = 3 de 3 viables -> fracción 1. Masa excluida 0.
        let r = coin()
            .oracle(
                S0,
                &BoolSamplingPolicy::UniformActions,
                2,
                OracleLimits::default(),
            )
            .unwrap();
        assert_eq!(r.trajectories.len(), 2);
        let sum: f64 = r.trajectories.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() <= OracleLimits::default().probability_tolerance);
        assert_eq!(r.trajectories[0].0, Trajectory(vec![S0, S1, S1]));
        assert!((r.trajectories[0].1 - 0.5).abs() < EPS);
        assert!((r.trajectory_entropy_bits - 1.0).abs() < EPS);
        assert!((r.final_state_entropy_bits - 1.0).abs() < EPS);
        assert_eq!(r.reachability.reachable.len(), 3);
        assert!((r.reachability.fraction() - 1.0).abs() < EPS);
        assert!(r.excluded_mass.abs() < EPS);
    }

    #[test]
    fn gog_154_hazard_fixture_excludes_without_reward() {
        // Cálculo a mano, horizonte 1: masa viable = 1/2 + 1/4 = 3/4; excluida = 1/4.
        // Condicionadas: [0,1] = (1/2)/(3/4) = 2/3, [0,2] = (1/4)/(3/4) = 1/3.
        // H = -(2/3 log2 2/3 + 1/3 log2 1/3) = log2(3) - 2/3 ≈ 0.9182958340544896 bits.
        // Alcanzables viables {0,1,2} de 3 viables (el 3 inviable no cuenta).
        let r = hazard()
            .oracle(
                S0,
                &BoolSamplingPolicy::UniformActions,
                1,
                OracleLimits::default(),
            )
            .unwrap();
        assert!((r.viable_mass - 0.75).abs() < EPS);
        assert!((r.excluded_mass - 0.25).abs() < EPS);
        assert!((r.trajectories[0].1 - 2.0 / 3.0).abs() < EPS);
        assert!((r.trajectories[1].1 - 1.0 / 3.0).abs() < EPS);
        let expected = 3f64.log2() - 2.0 / 3.0;
        assert!((r.trajectory_entropy_bits - expected).abs() < 1e-9);
        assert!(!r.reachability.reachable.contains(&S3));
        assert_eq!(r.reachability.viable_total, 3);
        // Alcanzabilidad y entropía son campos distintos y no se equiparan.
        assert_ne!(r.reachability.fraction(), r.trajectory_entropy_bits);
    }

    #[test]
    fn gog_154_policy_changes_distribution() {
        // Dos acciones deterministas (->1, ->2). Uniforme: 1/2,1/2 -> 1 bit; Fija(acción 0): 0 bits.
        let w = BoolWorld::new(1)
            .with_state(S0, true, vec![vec![(S1, 1.0)], vec![(S2, 1.0)]])
            .unwrap()
            .with_state(S1, true, vec![])
            .unwrap()
            .with_state(S2, true, vec![])
            .unwrap();
        let u = w
            .oracle(
                S0,
                &BoolSamplingPolicy::UniformActions,
                1,
                OracleLimits::default(),
            )
            .unwrap();
        assert!((u.trajectory_entropy_bits - 1.0).abs() < EPS);
        let f = w
            .oracle(
                S0,
                &BoolSamplingPolicy::Fixed(BTreeMap::from([(S0, 0)])),
                1,
                OracleLimits::default(),
            )
            .unwrap();
        assert!(f.trajectory_entropy_bits.abs() < EPS);
        assert_eq!(f.reachability.reachable.len(), 2);
    }

    #[test]
    fn gog_154_stable_hash_same_seed_same_result() {
        let run = |seed| {
            BoolWorld::from_seed(seed, 5)
                .unwrap()
                .oracle(
                    S0,
                    &BoolSamplingPolicy::UniformActions,
                    4,
                    OracleLimits::default(),
                )
                .unwrap()
        };
        let (a, b, c) = (run(42), run(42), run(43));
        assert_eq!(a.stable_hash, b.stable_hash);
        assert_eq!(a, b);
        assert_ne!(a.stable_hash, c.stable_hash);
        // Cambiar el horizonte cambia el hash.
        let d = coin()
            .oracle(
                S0,
                &BoolSamplingPolicy::UniformActions,
                1,
                OracleLimits::default(),
            )
            .unwrap();
        let e = coin()
            .oracle(
                S0,
                &BoolSamplingPolicy::UniformActions,
                2,
                OracleLimits::default(),
            )
            .unwrap();
        assert_ne!(d.stable_hash, e.stable_hash);
    }

    #[test]
    fn gog_154_explosion_is_typed_error() {
        // Mundo con 2 sucesores por paso: 2^horizon trayectorias; límite 8 con horizonte 10.
        let w = BoolWorld::new(0)
            .with_state(S0, true, vec![vec![(S0, 0.5), (S1, 0.5)]])
            .unwrap()
            .with_state(S1, true, vec![vec![(S0, 0.5), (S1, 0.5)]])
            .unwrap();
        let limits = OracleLimits {
            max_trajectories: 8,
            ..OracleLimits::default()
        };
        let err = w
            .oracle(S0, &BoolSamplingPolicy::UniformActions, 10, limits)
            .unwrap_err();
        assert_eq!(
            err,
            BoolWorldError::CombinatorialExplosion {
                limit: 8,
                what: "trajectories",
                depth: 4
            }
        );
        let limits = OracleLimits {
            max_states: 1,
            ..OracleLimits::default()
        };
        assert!(matches!(
            w.oracle(S0, &BoolSamplingPolicy::UniformActions, 1, limits),
            Err(BoolWorldError::CombinatorialExplosion { what: "states", .. })
        ));
    }

    #[test]
    fn gog_154_fail_closed_invalid_inputs() {
        let w = hazard();
        let l = OracleLimits::default();
        let p = BoolSamplingPolicy::UniformActions;
        assert_eq!(w.oracle(S0, &p, 0, l), Err(BoolWorldError::InvalidHorizon));
        assert_eq!(
            w.oracle(S3, &p, 1, l),
            Err(BoolWorldError::InvalidInitialState(S3))
        );
        assert!(
            BoolWorld::new(0)
                .with_state(S0, true, vec![vec![(S0, 0.4)]])
                .is_err()
        );
        // Toda la masa en inviables.
        let dead = BoolWorld::new(0)
            .with_state(S0, true, vec![vec![(S3, 1.0)]])
            .unwrap()
            .with_state(S3, false, vec![])
            .unwrap();
        assert_eq!(
            dead.oracle(S0, &p, 1, l),
            Err(BoolWorldError::NoViableTrajectory)
        );
    }
}
