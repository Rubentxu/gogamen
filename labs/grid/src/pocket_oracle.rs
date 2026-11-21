//! Oráculo de bolsillo exacto por programación dinámica de coste lineal (GOG-196).
//!
//! Complementa (sin sustituir) al oráculo de enumeración de [`crate::bool_world`], que sigue
//! siendo la referencia en fixtures pequeños y en mundos estocásticos.
//!
//! # Definición (mundos deterministas)
//! Cada `(s, a)` tiene un único sucesor `s'_a`. Con `N_0(s) = 1` si `s` es viable (0 si no) y
//! `N_h(s) = sum_a N_{h-1}(s'_a)` (0 si `s` inviable), `N_h(s)` cuenta las **secuencias de
//! acciones** viables de longitud `h`. La política de máxima entropía de trayectoria es
//! `pi_h(a|s) = N_{h-1}(s'_a) / N_h(s)` (`h` = pasos restantes); induce la distribución
//! uniforme sobre secuencias viables, y su entropía por regla de la cadena es `log2 N_H`.
//! La supervivencia a horizonte `H` se calcula por inducción hacia atrás.
//!
//! Los campos del informe son **separados** y tipados con unidades: supervivencia
//! (probabilidad), `log2` del recuento de secuencias viables (bits) y entropía del estado final
//! (bits). Excluir transiciones inviables NO es un reward ni una autorización (`ActionGuard`
//! no interviene): solo recorta el espacio de futuros viables.
//!
//! # Aritmética
//! Se trabaja en el dominio `log2` con log-sum-exp (sin desbordamiento a `H = 64` ni mayores);
//! error relativo del orden de `H * 2^-52`. `-inf` representa recuento 0.
//!
//! # Mundos estocásticos
//! Se **rechazan** con [`PocketOracleError::StochasticTransition`]: la enumeración de soporte
//! de `BoolWorld::oracle` sigue siendo la referencia.
//!
//! # Coste
//! `ops_backward = H * sum_s max(1, |A_s|)` (un `op` por actualización `(h, s, a)`), lineal en
//! `H * S * A`; la pasada hacia delante (entropía) cuesta lo mismo (`ops_forward`).
//! Un estado sin acciones es absorbente (una acción implícita hacia sí mismo), como en
//! `BoolWorld::oracle`. Sin afirmaciones de emergencia: es un oráculo de laboratorio.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crate::bool_world::{BoolWorld, MicroState};

/// Versión de esquema del informe DP (entra en el hash estable).
pub const POCKET_ORACLE_SCHEMA_VERSION: u32 = 1;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Errores tipados del oráculo DP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PocketOracleError {
    /// Horizonte cero.
    InvalidHorizon,
    /// Estado inicial inexistente o inviable.
    InvalidInitialState(MicroState),
    /// Una acción tiene varios sucesores con probabilidad positiva (mundo estocástico).
    StochasticTransition {
        /// Estado de origen.
        state: MicroState,
    },
    /// Transición hacia un microestado no declarado.
    UndeclaredSuccessor {
        /// Estado de origen.
        state: MicroState,
    },
    /// No existe ninguna secuencia viable de longitud `H`.
    NoViableSequence,
}

impl fmt::Display for PocketOracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidHorizon => write!(f, "horizonte debe ser >= 1"),
            Self::InvalidInitialState(s) => write!(f, "estado inicial inválido: {}", s.0),
            Self::StochasticTransition { state } => write!(
                f,
                "transición estocástica en {}: usar la enumeración de BoolWorld::oracle",
                state.0
            ),
            Self::UndeclaredSuccessor { state } => {
                write!(f, "sucesor no declarado desde {}", state.0)
            }
            Self::NoViableSequence => write!(f, "no existe ninguna secuencia viable"),
        }
    }
}

impl Error for PocketOracleError {}

/// Probabilidad de supervivencia (en `[0, 1]`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurvivalProb(pub f64);

/// Cantidad en bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bits(pub f64);

/// Informe DP; los campos son magnitudes separadas.
#[derive(Debug, Clone)]
pub struct PocketReport {
    /// Seed del mundo.
    pub seed: u64,
    /// Estado inicial.
    pub initial: MicroState,
    /// Horizonte `H`.
    pub horizon: usize,
    /// Supervivencia a `H` con la mejor política (inducción hacia atrás, `max_a`).
    pub survival_best_policy: SurvivalProb,
    /// Supervivencia a `H` con acciones uniformes (coincide con `OracleReport::viable_mass`).
    pub survival_uniform_policy: SurvivalProb,
    /// `log2 N_H(initial)`: log2 del recuento de secuencias viables (bits).
    pub log2_viable_sequences: Bits,
    /// Entropía de trayectoria de la política de máxima entropía por regla de la cadena (bits).
    pub trajectory_entropy_bits: Bits,
    /// Entropía de la distribución del estado final bajo esa política (bits).
    pub final_state_entropy_bits: Bits,
    /// Número de estados.
    pub states: usize,
    /// `sum_s max(1, |A_s|)`.
    pub state_action_pairs: usize,
    /// Operaciones de la pasada hacia atrás (== `horizon * state_action_pairs`).
    pub ops_backward: u64,
    /// Operaciones de la pasada hacia delante.
    pub ops_forward: u64,
    /// Hash FNV-1a estable (esquema, seed, horizonte, resultados).
    pub stable_hash: u64,
    index: Vec<MicroState>,
    succ: Vec<Vec<usize>>,
    /// `log2_n[h][i]` = `log2 N_h(state i)`, `h = 0..=H`.
    log2_n: Vec<Vec<f64>>,
}

impl PocketReport {
    /// `log2 N_h(state)` para `h` pasos restantes (`-inf` si 0 o estado desconocido).
    pub fn log2_count(&self, remaining: usize, state: MicroState) -> f64 {
        match (self.index.binary_search(&state), self.log2_n.get(remaining)) {
            (Ok(i), Some(row)) => row[i],
            _ => f64::NEG_INFINITY,
        }
    }

    /// Política de máxima entropía `pi_h(.|s)`: `(índice de acción, sucesor, probabilidad)`
    /// para las acciones con sucesor viable. Vacía si `N_h(s) = 0`, `h = 0` o desconocido.
    pub fn policy(&self, remaining: usize, state: MicroState) -> Vec<(usize, MicroState, f64)> {
        let Ok(i) = self.index.binary_search(&state) else {
            return Vec::new();
        };
        if remaining == 0 || remaining >= self.log2_n.len() || self.log2_n[remaining][i] == NEG {
            return Vec::new();
        }
        let ln = self.log2_n[remaining][i];
        self.succ[i]
            .iter()
            .enumerate()
            .filter_map(|(a, &j)| {
                let l = self.log2_n[remaining - 1][j];
                (l != NEG).then(|| (a, self.index[j], (l - ln).exp2()))
            })
            .collect()
    }
}

const NEG: f64 = f64::NEG_INFINITY;

/// `log2(sum 2^l_i)` estable; `-inf` si todos son `-inf`.
fn log2_sum_exp2(ls: &[f64]) -> f64 {
    let m = ls.iter().copied().fold(NEG, f64::max);
    if m == NEG {
        return NEG;
    }
    m + ls.iter().map(|l| (l - m).exp2()).sum::<f64>().log2()
}

fn entropy_bits(ps: impl Iterator<Item = f64>) -> f64 {
    ps.filter(|p| *p > 0.0).map(|p| -p * p.log2()).sum()
}

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *hash ^= u64::from(*b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

/// Resuelve el oráculo DP desde `initial` con `horizon` pasos.
pub fn solve_pocket(
    world: &BoolWorld,
    initial: MicroState,
    horizon: usize,
) -> Result<PocketReport, PocketOracleError> {
    if horizon == 0 {
        return Err(PocketOracleError::InvalidHorizon);
    }
    let specs: Vec<_> = world.state_specs().collect();
    let index: Vec<MicroState> = specs.iter().map(|s| s.state).collect();
    let viable: Vec<bool> = specs.iter().map(|s| s.viable).collect();
    let mut succ: Vec<Vec<usize>> = Vec::with_capacity(specs.len());
    for (i, spec) in specs.iter().enumerate() {
        let mut row = Vec::new();
        if spec.actions.is_empty() {
            row.push(i); // absorbente
        }
        for act in spec.actions {
            let mut it = act.iter().filter(|(_, p)| *p > 0.0);
            let (Some(&(d, _)), None) = (it.next(), it.next()) else {
                return Err(PocketOracleError::StochasticTransition { state: spec.state });
            };
            let j = index
                .binary_search(&d)
                .map_err(|_| PocketOracleError::UndeclaredSuccessor { state: spec.state })?;
            row.push(j);
        }
        succ.push(row);
    }
    let i0 = match index.binary_search(&initial) {
        Ok(i) if viable[i] => i,
        _ => return Err(PocketOracleError::InvalidInitialState(initial)),
    };
    let n = index.len();
    let pairs: usize = succ.iter().map(Vec::len).sum();

    // Pasada hacia atrás: N_h (log2), supervivencia óptima y uniforme.
    let mut log2_n: Vec<Vec<f64>> =
        vec![viable.iter().map(|&v| if v { 0.0 } else { NEG }).collect()];
    let mut best: Vec<f64> = viable.iter().map(|&v| f64::from(u8::from(v))).collect();
    let mut unif = best.clone();
    let mut ops_backward = 0u64;
    for h in 1..=horizon {
        let prev = &log2_n[h - 1];
        let mut row = vec![NEG; n];
        let mut nb = vec![0.0; n];
        let mut nu = vec![0.0; n];
        for i in 0..n {
            let mut ls = Vec::with_capacity(succ[i].len());
            let (mut mx, mut sm) = (0.0f64, 0.0f64);
            for &j in &succ[i] {
                ops_backward += 1;
                ls.push(prev[j]);
                mx = mx.max(best[j]);
                sm += unif[j];
            }
            if viable[i] {
                row[i] = log2_sum_exp2(&ls);
                nb[i] = mx;
                nu[i] = sm / succ[i].len() as f64;
            }
        }
        log2_n.push(row);
        best = nb;
        unif = nu;
    }
    let l_total = log2_n[horizon][i0];
    if l_total == NEG {
        return Err(PocketOracleError::NoViableSequence);
    }

    // Pasada hacia delante: distribución bajo pi y entropía por regla de la cadena.
    let mut dist = vec![0.0f64; n];
    dist[i0] = 1.0;
    let mut chain = 0.0;
    let mut ops_forward = 0u64;
    for t in 0..horizon {
        let rem = horizon - t;
        let mut nd = vec![0.0f64; n];
        for i in 0..n {
            if dist[i] == 0.0 || log2_n[rem][i] == NEG {
                ops_forward += succ[i].len() as u64;
                continue;
            }
            let probs: Vec<f64> = succ[i]
                .iter()
                .map(|&j| {
                    ops_forward += 1;
                    let l = log2_n[rem - 1][j];
                    if l == NEG {
                        0.0
                    } else {
                        (l - log2_n[rem][i]).exp2()
                    }
                })
                .collect();
            chain += dist[i] * entropy_bits(probs.iter().copied());
            for (&j, p) in succ[i].iter().zip(&probs) {
                nd[j] += dist[i] * p;
            }
        }
        dist = nd;
    }
    let final_h = entropy_bits(dist.iter().copied());

    let mut r = PocketReport {
        seed: world.seed(),
        initial,
        horizon,
        survival_best_policy: SurvivalProb(best[i0]),
        survival_uniform_policy: SurvivalProb(unif[i0]),
        log2_viable_sequences: Bits(l_total),
        trajectory_entropy_bits: Bits(chain),
        final_state_entropy_bits: Bits(final_h),
        states: n,
        state_action_pairs: pairs,
        ops_backward,
        ops_forward,
        stable_hash: 0,
        index,
        succ,
        log2_n,
    };
    r.stable_hash = hash_report(&r);
    Ok(r)
}

fn hash_report(r: &PocketReport) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    fnv(&mut h, &POCKET_ORACLE_SCHEMA_VERSION.to_le_bytes());
    fnv(&mut h, &r.seed.to_le_bytes());
    fnv(&mut h, &[r.initial.0]);
    fnv(&mut h, &(r.horizon as u64).to_le_bytes());
    for v in [
        r.survival_best_policy.0,
        r.survival_uniform_policy.0,
        r.log2_viable_sequences.0,
        r.trajectory_entropy_bits.0,
        r.final_state_entropy_bits.0,
    ] {
        fnv(&mut h, &v.to_bits().to_le_bytes());
    }
    fnv(&mut h, &r.ops_backward.to_le_bytes());
    fnv(&mut h, &r.ops_forward.to_le_bytes());
    let by_state: BTreeMap<u8, &Vec<usize>> =
        r.index.iter().map(|s| s.0).zip(r.succ.iter()).collect();
    for (s, row) in by_state {
        fnv(&mut h, &[s]);
        for &j in row {
            fnv(&mut h, &[r.index[j].0]);
        }
    }
    h
}
