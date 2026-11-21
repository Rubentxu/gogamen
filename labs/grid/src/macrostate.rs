//! Macroestados por equivalencia predictiva **con tolerancia** sobre [`BoolWorld`]
//! (GOG-161/GOG-201, DEM-008 v2).
//!
//! # Qué hace
//! - Dado un mundo, una función de observación `observe: MicroState -> u8` y un horizonte `H`,
//!   calcula para cada microestado su **firma predictiva**: para cada secuencia de acciones
//!   en bucle abierto de longitud `H` (índices de acción), la distribución exacta de la
//!   secuencia de observaciones `(o_0, ..., o_H)`. Una secuencia de acciones no disponible
//!   en un estado (índice fuera de rango) aporta masa 0.
//! - [`predictive_macrostates`] agrupa microestados cuyas firmas difieren en como mucho
//!   `tolerance` (máxima desviación absoluta) y devuelve `phi: micro -> macro`.
//!   El agrupamiento es voraz y determinista: se compara con el primer miembro (representante)
//!   de cada grupo, por lo que dos miembros pueden diferir hasta `2 * tolerance`; eso se
//!   audita igualmente (ver contraejemplos). Es la ruta de **producción**.
//! - [`exact_predictive_macrostates`] fuerza `tolerance = 0`: es el **oráculo** de igualdad
//!   exacta, útil solo para validar fixtures; no es la ruta de producción.
//! - [`analyze_partition`] evalúa una partición arbitraria (útil para exhibir
//!   proyecciones no markovianas).
//! - El **cociente** usa pesos uniformes sobre los miembros de cada macro; su predicción se
//!   compara con la de cada microestado ([`MacroReport::quotient_max_abs_deviation`]).
//! - [`MacroReport`] expone pérdida de información ([`MacroReport::information_loss_bits`] =
//!   `H(micro | macro)` con microestados uniformes), tamaño de la partición
//!   ([`MacroReport::partition_size`]) y, si la proyección no es markoviana, un
//!   [`NonMarkovCounterexample`] con el par microscópico y las distribuciones de sucesores.
//! - [`MacroReport::phi_scale`] es un [`gogamen_core::CanonicalScale`] (contrato de escala del
//!   núcleo): `phi` deja de ser un `BTreeMap` opaco y pasa a tener
//!   [`gogamen_core::ScaleIdentity`] `(id, versión, hash)` compartible con aprendices y
//!   planificadores. El `id` deriva del mundo (su seed, única identidad que expone
//!   [`BoolWorld`]), del horizonte y de la tolerancia; la versión es
//!   [`MACRO_PHI_SCALE_VERSION`]. El hash FNV-1a cubre la asignación `micro -> macro`
//!   ordenada (no las probabilidades crudas), así que una perturbación de probabilidades
//!   menor que la tolerancia conserva partición e identidad.
//!
//! # Límites y no-afirmaciones
//! - Todo es enumeración exacta y acotada ([`MacroLimits`]); al superar un límite se
//!   devuelve [`MacroError::LimitExceeded`] (fail-closed). Sin aleatoriedad ni reloj.
//! - La equivalencia es respecto a la observación, el horizonte y las secuencias de
//!   acciones en bucle abierto elegidos; no afirma emergencia ni causalidad. Un mismo
//!   mundo puede dar particiones distintas con otro `observe` u `H`.
//! - "Markoviana" aquí significa solo: dentro de cada macro, firmas iguales (± tolerancia)
//!   y misma distribución de macros sucesores por acción (± tolerancia).
//! - Las transiciones a estados inviables no se excluyen ni renormalizan; la viabilidad solo
//!   interviene si `observe` la codifica.
//! - La `Scale` de `phi` es identidad de discretización, no permiso ni verdad de terreno.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use gogamen_core::{BinningVersion, CanonicalScale, ScaleError, ScaleIdentity};

use crate::bool_world::{BoolWorld, BoolWorldError, MicroState};

/// Identificador de macroestado (índice denso, ordenado por primer microestado miembro).
pub type MacroId = usize;

/// Versión del esquema de la escala macro `phi` (GOG-201).
pub const MACRO_PHI_SCALE_VERSION: BinningVersion = BinningVersion::new(1, 0);

/// Prefijo del identificador estable de la escala macro `phi`.
pub const MACRO_PHI_SCALE_ID_PREFIX: &str = "gogamen.macro.phi";

/// Límites fail-closed del análisis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MacroLimits {
    /// Máximo de microestados del mundo.
    pub max_states: usize,
    /// Máximo de secuencias de acciones (`A^H`).
    pub max_action_sequences: usize,
    /// Máximo de entradas vivas en la frontera de una predicción.
    pub max_frontier: usize,
    /// Máximo de entradas de firma acumuladas (todos los microestados).
    pub max_signature_entries: usize,
    /// Tolerancia de igualdad de predicciones (finita, `>= 0`).
    pub tolerance: f64,
}

impl Default for MacroLimits {
    fn default() -> Self {
        Self {
            max_states: 256,
            max_action_sequences: 4_096,
            max_frontier: 10_000,
            max_signature_entries: 1_000_000,
            tolerance: 1e-9,
        }
    }
}

/// Errores del análisis de macroestados.
#[derive(Debug, Clone, PartialEq)]
pub enum MacroError {
    /// Tolerancia no finita o negativa.
    InvalidTolerance,
    /// Horizonte cero.
    InvalidHorizon,
    /// Mundo vacío.
    EmptyWorld,
    /// Partición inválida (falta un microestado, desconocido o ids no densos).
    InvalidPartition {
        /// Motivo.
        reason: &'static str,
    },
    /// Se superó un límite de enumeración.
    LimitExceeded {
        /// Qué se limitaba.
        what: &'static str,
        /// Valor del límite.
        limit: usize,
    },
    /// No se pudo construir la escala canónica de `phi`.
    Scale(ScaleError),
    /// Error del mundo subyacente.
    World(BoolWorldError),
}

impl fmt::Display for MacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTolerance => write!(f, "tolerancia no finita o negativa"),
            Self::InvalidHorizon => write!(f, "horizonte debe ser >= 1"),
            Self::EmptyWorld => write!(f, "mundo sin microestados"),
            Self::InvalidPartition { reason } => write!(f, "partición inválida: {reason}"),
            Self::LimitExceeded { what, limit } => {
                write!(f, "límite superado: {what} (máximo {limit})")
            }
            Self::Scale(e) => write!(f, "escala phi inválida: {e}"),
            Self::World(e) => write!(f, "error de mundo: {e}"),
        }
    }
}

impl Error for MacroError {}

impl From<BoolWorldError> for MacroError {
    fn from(e: BoolWorldError) -> Self {
        Self::World(e)
    }
}

impl From<ScaleError> for MacroError {
    fn from(e: ScaleError) -> Self {
        Self::Scale(e)
    }
}

/// Contraejemplo de que la proyección no es markoviana (el primero encontrado, determinista).
#[derive(Debug, Clone, PartialEq)]
pub enum NonMarkovCounterexample {
    /// Dos microestados del mismo macro con predicciones distintas.
    PredictionMismatch {
        /// Macro afectado.
        macro_id: MacroId,
        /// Primer microestado.
        a: MicroState,
        /// Segundo microestado.
        b: MicroState,
        /// Índice (en orden lexicográfico) de la secuencia de acciones donde difieren.
        action_sequence: Vec<usize>,
        /// Distribución de secuencias de observación del primer microestado para esa secuencia.
        a_outcomes: Vec<(Vec<u8>, f64)>,
        /// Distribución de secuencias de observación del segundo microestado para esa secuencia.
        b_outcomes: Vec<(Vec<u8>, f64)>,
        /// Desviación absoluta máxima entre ambas firmas.
        deviation: f64,
    },
    /// La distribución del macro sucesor depende del representante microscópico.
    SuccessorDependsOnRepresentative {
        /// Macro afectado.
        macro_id: MacroId,
        /// Acción.
        action: usize,
        /// Primer microestado.
        a: MicroState,
        /// Segundo microestado.
        b: MicroState,
        /// Macro sucesor donde difiere la probabilidad.
        successor: MacroId,
        /// Distribución de macros sucesores del primer microestado bajo `action`.
        a_distribution: Vec<(MacroId, f64)>,
        /// Distribución de macros sucesores del segundo microestado bajo `action`.
        b_distribution: Vec<(MacroId, f64)>,
        /// Diferencia absoluta de probabilidad.
        deviation: f64,
    },
}

/// Resultado del análisis de una partición.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroReport {
    /// Proyección `phi: micro -> macro`.
    pub phi: BTreeMap<MicroState, MacroId>,
    /// Escala canónica del núcleo que identifica esta proyección `phi`.
    ///
    /// Su [`ScaleIdentity`] deriva del mundo, del horizonte y de la tolerancia, y su hash del
    /// contenido de la partición (no de las probabilidades crudas): compartirla con un
    /// aprendiz o planificador no duplica la discretización.
    pub phi_scale: CanonicalScale,
    /// Miembros de cada macro (ordenados).
    pub macros: Vec<Vec<MicroState>>,
    /// Número de macroestados.
    pub partition_size: usize,
    /// Número de microestados.
    pub micro_count: usize,
    /// Horizonte usado.
    pub horizon: usize,
    /// `H(micro | macro)` en bits con microestados uniformes: `sum |M|/n * log2 |M|`.
    pub information_loss_bits: f64,
    /// Máxima desviación absoluta de firma entre miembros del mismo macro (todos los pares).
    pub max_within_macro_deviation: f64,
    /// Máxima desviación absoluta entre la predicción del microestado y la del cociente.
    pub quotient_max_abs_deviation: f64,
    /// `true` si no hay contraejemplo.
    pub markovian: bool,
    /// Contraejemplo si `!markovian`.
    pub counterexample: Option<NonMarkovCounterexample>,
}

impl MacroReport {
    /// Identidad de escala `(id, versión, hash)` de `phi`.
    pub fn phi_identity(&self) -> ScaleIdentity {
        self.phi_scale.identity()
    }

    /// Hash canónico de la partición (estable bajo perturbaciones menores que la tolerancia).
    pub const fn phi_hash(&self) -> u64 {
        self.phi_scale.hash()
    }
}

/// Sucesores `(nodo, probabilidad)`.
type Succ = Vec<(usize, f64)>;
type TransTable = Vec<Vec<Option<Succ>>>;
type Signature = BTreeMap<(usize, Vec<u8>), f64>;

/// Agrupa microestados por equivalencia predictiva con tolerancia (ruta de producción).
///
/// Dos microestados caen en el mismo macro si su máxima desviación absoluta de firma es
/// `<= limits.tolerance`. El agrupamiento es voraz contra el representante de cada grupo.
pub fn predictive_macrostates(
    world: &BoolWorld,
    observe: &dyn Fn(MicroState) -> u8,
    horizon: usize,
    limits: MacroLimits,
) -> Result<MacroReport, MacroError> {
    let model = Model::build(world, observe, horizon, limits)?;
    let sigs = model.signatures(limits)?;
    let mut reps: Vec<usize> = Vec::new();
    let mut assign = vec![0usize; model.n];
    for i in 0..model.n {
        match reps
            .iter()
            .position(|&r| sig_deviation(&sigs[r], &sigs[i]).0 <= limits.tolerance)
        {
            Some(m) => assign[i] = m,
            None => {
                assign[i] = reps.len();
                reps.push(i);
            }
        }
    }
    model.report(&assign, &sigs, limits)
}

/// Oráculo de equivalencia predictiva **exacta** (`tolerance = 0`).
///
/// Solo para validar fixtures y comparar contra la ruta tolerante; no es la partición de
/// producción. Con `tolerance = 0` dos firmas se agrupan solo si coinciden bit a bit.
pub fn exact_predictive_macrostates(
    world: &BoolWorld,
    observe: &dyn Fn(MicroState) -> u8,
    horizon: usize,
    limits: MacroLimits,
) -> Result<MacroReport, MacroError> {
    predictive_macrostates(
        world,
        observe,
        horizon,
        MacroLimits {
            tolerance: 0.0,
            ..limits
        },
    )
}

/// Evalúa una partición dada (`micro -> macro`, ids densos `0..k`) contra el mundo.
pub fn analyze_partition(
    world: &BoolWorld,
    observe: &dyn Fn(MicroState) -> u8,
    partition: &BTreeMap<MicroState, MacroId>,
    horizon: usize,
    limits: MacroLimits,
) -> Result<MacroReport, MacroError> {
    let model = Model::build(world, observe, horizon, limits)?;
    if partition.len() != model.n {
        return Err(MacroError::InvalidPartition {
            reason: "la partición debe cubrir exactamente los microestados",
        });
    }
    let mut assign = vec![0usize; model.n];
    for (i, s) in model.states.iter().enumerate() {
        assign[i] = *partition.get(s).ok_or(MacroError::InvalidPartition {
            reason: "microestado sin macro",
        })?;
    }
    let k = assign.iter().max().map_or(0, |m| m + 1);
    for m in 0..k {
        if !assign.contains(&m) {
            return Err(MacroError::InvalidPartition {
                reason: "ids de macro no densos",
            });
        }
    }
    let sigs = model.signatures(limits)?;
    model.report(&assign, &sigs, limits)
}

/// Máxima desviación absoluta entre dos firmas y la secuencia (índice) donde ocurre.
fn sig_deviation(a: &Signature, b: &Signature) -> (f64, usize) {
    let mut best = (0.0f64, 0usize);
    for (k, &p) in a {
        let d = (p - b.get(k).copied().unwrap_or(0.0)).abs();
        if d > best.0 {
            best = (d, k.0);
        }
    }
    for (k, &q) in b {
        if !a.contains_key(k) && q > best.0 {
            best = (q, k.0);
        }
    }
    best
}

/// Modelo denso: nodos 0..n, `trans[nodo][acción]` (None = no disponible).
struct Model {
    seed: u64,
    states: Vec<MicroState>,
    obs: Vec<u8>,
    trans: TransTable,
    n: usize,
    actions: usize,
    horizon: usize,
}

impl Model {
    fn build(
        world: &BoolWorld,
        observe: &dyn Fn(MicroState) -> u8,
        horizon: usize,
        limits: MacroLimits,
    ) -> Result<Self, MacroError> {
        if !limits.tolerance.is_finite() || limits.tolerance < 0.0 {
            return Err(MacroError::InvalidTolerance);
        }
        if horizon == 0 {
            return Err(MacroError::InvalidHorizon);
        }
        let specs: Vec<_> = world.state_specs().collect();
        if specs.is_empty() {
            return Err(MacroError::EmptyWorld);
        }
        if specs.len() > limits.max_states {
            return Err(MacroError::LimitExceeded {
                what: "states",
                limit: limits.max_states,
            });
        }
        let states: Vec<MicroState> = specs.iter().map(|s| s.state).collect();
        let index: BTreeMap<MicroState, usize> =
            states.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let actions = specs
            .iter()
            .map(|s| s.actions.len())
            .max()
            .unwrap_or(0)
            .max(1);
        let mut count: usize = 1;
        for _ in 0..horizon {
            count = count
                .checked_mul(actions)
                .filter(|c| *c <= limits.max_action_sequences)
                .ok_or(MacroError::LimitExceeded {
                    what: "action_sequences",
                    limit: limits.max_action_sequences,
                })?;
        }
        let mut trans = Vec::with_capacity(specs.len());
        for (i, spec) in specs.iter().enumerate() {
            let mut row: Vec<Option<Succ>> = vec![None; actions];
            if spec.actions.is_empty() {
                row[0] = Some(vec![(i, 1.0)]); // absorbente
            }
            for (a, dests) in spec.actions.iter().enumerate() {
                let mut v = Vec::new();
                for &(d, p) in dests {
                    let j = *index.get(&d).ok_or(BoolWorldError::InvalidWorld {
                        reason: "transición a microestado no declarado",
                    })?;
                    if p > 0.0 {
                        v.push((j, p));
                    }
                }
                row[a] = Some(v);
            }
            trans.push(row);
        }
        Ok(Self {
            seed: world.seed(),
            obs: states.iter().map(|s| observe(*s)).collect(),
            n: states.len(),
            states,
            trans,
            actions,
            horizon,
        })
    }

    fn sequences(&self) -> Vec<Vec<usize>> {
        let mut seqs: Vec<Vec<usize>> = vec![vec![]];
        for _ in 0..self.horizon {
            seqs = seqs
                .into_iter()
                .flat_map(|s| {
                    (0..self.actions).map(move |a| {
                        let mut t = s.clone();
                        t.push(a);
                        t
                    })
                })
                .collect();
        }
        seqs
    }

    fn signatures(&self, limits: MacroLimits) -> Result<Vec<Signature>, MacroError> {
        let seqs = self.sequences();
        let mut total = 0usize;
        let mut out = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let mut sig = Signature::new();
            for (si, seq) in seqs.iter().enumerate() {
                for (o, p) in predict(&self.trans, &self.obs, i, seq, limits.max_frontier)? {
                    sig.insert((si, o), p);
                }
            }
            total += sig.len();
            if total > limits.max_signature_entries {
                return Err(MacroError::LimitExceeded {
                    what: "signature_entries",
                    limit: limits.max_signature_entries,
                });
            }
            out.push(sig);
        }
        Ok(out)
    }

    fn report(
        &self,
        assign: &[usize],
        sigs: &[Signature],
        limits: MacroLimits,
    ) -> Result<MacroReport, MacroError> {
        let tol = limits.tolerance;
        let k = assign.iter().max().map_or(0, |m| m + 1);
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
        for (i, &m) in assign.iter().enumerate() {
            members[m].push(i);
        }
        let seqs = self.sequences();
        let mut counter: Option<NonMarkovCounterexample> = None;

        // Pares dentro de cada macro: firmas.
        let mut max_within = 0.0f64;
        for (m, ms) in members.iter().enumerate() {
            for (x, &a) in ms.iter().enumerate() {
                for &b in &ms[x + 1..] {
                    let (d, si) = sig_deviation(&sigs[a], &sigs[b]);
                    max_within = max_within.max(d);
                    if d > tol && counter.is_none() {
                        counter = Some(NonMarkovCounterexample::PredictionMismatch {
                            macro_id: m,
                            a: self.states[a],
                            b: self.states[b],
                            action_sequence: seqs[si].clone(),
                            a_outcomes: sequence_outcomes(&sigs[a], si),
                            b_outcomes: sequence_outcomes(&sigs[b], si),
                            deviation: d,
                        });
                    }
                }
            }
        }

        // Distribución de macros sucesores por micro y acción.
        let macro_succ = |i: usize, a: usize| -> Option<BTreeMap<MacroId, f64>> {
            self.trans[i][a].as_ref().map(|v| {
                let mut d = BTreeMap::new();
                for &(j, p) in v {
                    *d.entry(assign[j]).or_insert(0.0) += p;
                }
                d
            })
        };
        for (m, ms) in members.iter().enumerate() {
            for a in 0..self.actions {
                let dists: Vec<Option<BTreeMap<MacroId, f64>>> =
                    ms.iter().map(|&i| macro_succ(i, a)).collect();
                for x in 0..ms.len() {
                    for y in x + 1..ms.len() {
                        let empty = BTreeMap::new();
                        let dx = dists[x].as_ref().unwrap_or(&empty);
                        let dy = dists[y].as_ref().unwrap_or(&empty);
                        let keys: std::collections::BTreeSet<_> =
                            dx.keys().chain(dy.keys()).copied().collect();
                        for s in keys {
                            let d = (dx.get(&s).copied().unwrap_or(0.0)
                                - dy.get(&s).copied().unwrap_or(0.0))
                            .abs();
                            if d > tol && counter.is_none() {
                                counter = Some(
                                    NonMarkovCounterexample::SuccessorDependsOnRepresentative {
                                        macro_id: m,
                                        action: a,
                                        a: self.states[ms[x]],
                                        b: self.states[ms[y]],
                                        successor: s,
                                        a_distribution: dists[x]
                                            .clone()
                                            .unwrap_or_default()
                                            .into_iter()
                                            .collect(),
                                        b_distribution: dists[y]
                                            .clone()
                                            .unwrap_or_default()
                                            .into_iter()
                                            .collect(),
                                        deviation: d,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }

        // Cociente: transición macro = media uniforme de los miembros con la acción disponible.
        let mut mtrans: TransTable = Vec::with_capacity(k);
        for ms in &members {
            let mut row = Vec::with_capacity(self.actions);
            for a in 0..self.actions {
                let avail: Vec<_> = ms.iter().filter_map(|&i| macro_succ(i, a)).collect();
                if avail.is_empty() {
                    row.push(None);
                    continue;
                }
                let mut acc: BTreeMap<MacroId, f64> = BTreeMap::new();
                for d in &avail {
                    for (&s, &p) in d {
                        *acc.entry(s).or_insert(0.0) += p / avail.len() as f64;
                    }
                }
                row.push(Some(acc.into_iter().collect()));
            }
            mtrans.push(row);
        }
        let mobs: Vec<u8> = members.iter().map(|ms| self.obs[ms[0]]).collect();
        let mut quotient_dev = 0.0f64;
        for (i, sig) in sigs.iter().enumerate() {
            let mut qsig = Signature::new();
            for (si, seq) in seqs.iter().enumerate() {
                for (o, p) in predict(&mtrans, &mobs, assign[i], seq, limits.max_frontier)? {
                    qsig.insert((si, o), p);
                }
            }
            quotient_dev = quotient_dev.max(sig_deviation(sig, &qsig).0);
        }

        let phi_scale = self.phi_scale(assign, tol)?;
        let n = self.n as f64;
        let information_loss_bits = members
            .iter()
            .map(|ms| ms.len() as f64 / n * (ms.len() as f64).log2())
            .sum();
        Ok(MacroReport {
            phi: self
                .states
                .iter()
                .zip(assign)
                .map(|(s, &m)| (*s, m))
                .collect(),
            phi_scale,
            macros: members
                .iter()
                .map(|ms| ms.iter().map(|&i| self.states[i]).collect())
                .collect(),
            partition_size: k,
            micro_count: self.n,
            horizon: self.horizon,
            information_loss_bits,
            max_within_macro_deviation: max_within,
            quotient_max_abs_deviation: quotient_dev,
            markovian: counter.is_none(),
            counterexample: counter,
        })
    }

    /// Escala canónica del núcleo que identifica `phi` (GOG-201).
    ///
    /// El id deriva del mundo (seed), del horizonte y de la tolerancia; el hash cubre solo la
    /// asignación `micro -> macro`, de modo que una perturbación de probabilidades menor que
    /// la tolerancia conserva la identidad.
    fn phi_scale(&self, assign: &[usize], tolerance: f64) -> Result<CanonicalScale, MacroError> {
        let id = format!(
            "{MACRO_PHI_SCALE_ID_PREFIX}/seed={}/horizon={}/tolerance={:e}",
            self.seed, self.horizon, tolerance
        );
        let declaration = canonical_phi_declaration(&self.states, assign);
        Ok(CanonicalScale::new(
            id,
            MACRO_PHI_SCALE_VERSION,
            declaration,
        )?)
    }
}

/// Codificación canónica de `phi`: pares `(micro, macro)` ordenados, sin probabilidades.
///
/// Incluir solo la asignación es deliberado: dos mundos cuyas transiciones difieren por menos
/// que la tolerancia producen la misma partición y deben compartir hash de escala.
fn canonical_phi_declaration(states: &[MicroState], assign: &[usize]) -> Vec<u8> {
    let mut pairs: Vec<(u8, u64)> = states
        .iter()
        .zip(assign)
        .map(|(state, &macro_id)| (state.0, macro_id as u64))
        .collect();
    pairs.sort_unstable();
    let mut out = Vec::with_capacity(26 + pairs.len() * 9);
    out.extend_from_slice(b"gogamen.macro.phi.decl.v1");
    out.extend_from_slice(&(pairs.len() as u64).to_le_bytes());
    for (micro, macro_id) in pairs {
        out.push(micro);
        out.extend_from_slice(&macro_id.to_le_bytes());
    }
    out
}

/// Distribución de secuencias de observación para la secuencia de acciones `seq_index`.
fn sequence_outcomes(sig: &Signature, seq_index: usize) -> Vec<(Vec<u8>, f64)> {
    sig.iter()
        .filter(|((si, _), _)| *si == seq_index)
        .map(|((_, outcome), &p)| (outcome.clone(), p))
        .collect()
}

/// Distribución exacta de secuencias de observación desde `start` aplicando `seq`.
fn predict(
    trans: &[Vec<Option<Succ>>],
    obs: &[u8],
    start: usize,
    seq: &[usize],
    max_frontier: usize,
) -> Result<BTreeMap<Vec<u8>, f64>, MacroError> {
    let mut frontier: BTreeMap<(Vec<u8>, usize), f64> = BTreeMap::new();
    frontier.insert((vec![obs[start]], start), 1.0);
    for &a in seq {
        let mut next: BTreeMap<(Vec<u8>, usize), f64> = BTreeMap::new();
        for ((o, node), p) in &frontier {
            if let Some(Some(dests)) = trans[*node].get(a) {
                for &(d, q) in dests {
                    let mut o2 = o.clone();
                    o2.push(obs[d]);
                    *next.entry((o2, d)).or_insert(0.0) += p * q;
                }
            }
        }
        if next.len() > max_frontier {
            return Err(MacroError::LimitExceeded {
                what: "frontier",
                limit: max_frontier,
            });
        }
        frontier = next;
    }
    let mut out: BTreeMap<Vec<u8>, f64> = BTreeMap::new();
    for ((o, _), p) in frontier {
        *out.entry(o).or_insert(0.0) += p;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(i: u8) -> MicroState {
        MicroState(i)
    }

    /// 0 y 1 equivalentes; 4 empieza con la misma observación que 0 pero predice distinto.
    fn fixture() -> BoolWorld {
        let mut w = BoolWorld::new(1);
        let spec = |a: Vec<Vec<(MicroState, f64)>>| a;
        w = w
            .with_state(
                s(0),
                true,
                spec(vec![vec![(s(2), 0.25), (s(3), 0.75)], vec![(s(0), 1.0)]]),
            )
            .unwrap()
            .with_state(
                s(1),
                true,
                spec(vec![vec![(s(2), 0.25), (s(3), 0.75)], vec![(s(1), 1.0)]]),
            )
            .unwrap()
            .with_state(s(2), true, vec![vec![(s(2), 1.0)], vec![(s(2), 1.0)]])
            .unwrap()
            .with_state(s(3), true, vec![vec![(s(3), 1.0)], vec![(s(3), 1.0)]])
            .unwrap()
            .with_state(s(4), true, vec![vec![(s(2), 1.0)], vec![(s(4), 1.0)]])
            .unwrap()
            .with_state(s(5), true, vec![vec![(s(5), 1.0)], vec![(s(5), 1.0)]])
            .unwrap();
        w
    }

    fn obs(m: MicroState) -> u8 {
        match m.0 {
            0 | 1 | 4 => 0,
            2 => 1,
            3 => 2,
            _ => 3,
        }
    }

    #[test]
    fn gog_161_groups_equivalent_and_separates_distinct() {
        let r = predictive_macrostates(&fixture(), &obs, 2, MacroLimits::default()).unwrap();
        assert_eq!(r.phi[&s(0)], r.phi[&s(1)]);
        assert_ne!(
            r.phi[&s(0)],
            r.phi[&s(4)],
            "misma obs inicial, distinta predicción"
        );
        assert_ne!(r.phi[&s(2)], r.phi[&s(3)]);
        assert_ne!(r.phi[&s(5)], r.phi[&s(2)]);
        assert_eq!(r.partition_size, 5);
        assert_eq!(r.micro_count, 6);
    }

    #[test]
    fn gog_161_quotient_matches_and_reports_loss() {
        let tol = 1e-9;
        let r = predictive_macrostates(&fixture(), &obs, 3, MacroLimits::default()).unwrap();
        assert!(
            r.quotient_max_abs_deviation <= tol,
            "{}",
            r.quotient_max_abs_deviation
        );
        assert!(r.markovian && r.counterexample.is_none());
        // H(micro|macro) = (2/6) * log2(2)
        assert!((r.information_loss_bits - 1.0 / 3.0).abs() < 1e-12);
        assert!(r.max_within_macro_deviation <= tol);
    }

    #[test]
    fn gog_161_non_markov_prediction_mismatch_counterexample() {
        let mut part: BTreeMap<MicroState, MacroId> = BTreeMap::new();
        // Fusiona 0 y 4 (predicciones distintas).
        for (m, id) in [(0, 0), (4, 0), (1, 1), (2, 2), (3, 3), (5, 4)] {
            part.insert(s(m), id);
        }
        let r = analyze_partition(&fixture(), &obs, &part, 2, MacroLimits::default()).unwrap();
        assert!(!r.markovian);
        match r.counterexample.unwrap() {
            NonMarkovCounterexample::PredictionMismatch {
                a, b, deviation, ..
            } => {
                assert_eq!((a, b), (s(0), s(4)));
                assert!(deviation > 1e-9);
            }
            other => panic!("inesperado: {other:?}"),
        }
        assert!(r.quotient_max_abs_deviation > 1e-9);
    }

    /// 0 -> 2 y 1 -> 3 (misma obs en 2 y 3), pero 2 y 3 tienen futuros distintos.
    fn successor_world() -> BoolWorld {
        BoolWorld::new(2)
            .with_state(s(0), true, vec![vec![(s(2), 1.0)]])
            .unwrap()
            .with_state(s(1), true, vec![vec![(s(3), 1.0)]])
            .unwrap()
            .with_state(s(2), true, vec![vec![(s(4), 1.0)]])
            .unwrap()
            .with_state(s(3), true, vec![vec![(s(5), 1.0)]])
            .unwrap()
            .with_state(s(4), true, vec![vec![(s(4), 1.0)]])
            .unwrap()
            .with_state(s(5), true, vec![vec![(s(5), 1.0)]])
            .unwrap()
    }

    fn obs2(m: MicroState) -> u8 {
        match m.0 {
            0 | 1 => 0,
            2 | 3 => 1,
            4 => 2,
            _ => 3,
        }
    }

    #[test]
    fn gog_161_successor_depends_on_representative_counterexample() {
        let w = successor_world();
        let mut part: BTreeMap<MicroState, MacroId> = BTreeMap::new();
        for (m, id) in [(0, 0), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4)] {
            part.insert(s(m), id);
        }
        // Horizonte 1: las firmas de 0 y 1 coinciden, pero los macros sucesores difieren.
        let r = analyze_partition(&w, &obs2, &part, 1, MacroLimits::default()).unwrap();
        assert!(!r.markovian);
        match r.counterexample.clone().unwrap() {
            NonMarkovCounterexample::SuccessorDependsOnRepresentative {
                macro_id,
                a,
                b,
                deviation,
                ..
            } => {
                assert_eq!((macro_id, a, b), (0, s(0), s(1)));
                assert!((deviation - 1.0).abs() < 1e-12);
            }
            other => panic!("inesperado: {other:?}"),
        }
        // La partición predictiva a horizonte 2 separa 0 y 1 (refina) y es markoviana.
        let p2 = predictive_macrostates(&w, &obs2, 2, MacroLimits::default()).unwrap();
        assert_ne!(p2.phi[&s(0)], p2.phi[&s(1)]);
        assert!(p2.markovian);
    }

    #[test]
    fn gog_161_fail_closed_limits_and_tolerance() {
        let w = fixture();
        for bad in [f64::NAN, f64::INFINITY, -1e-3] {
            let l = MacroLimits {
                tolerance: bad,
                ..MacroLimits::default()
            };
            assert_eq!(
                predictive_macrostates(&w, &obs, 1, l).unwrap_err(),
                MacroError::InvalidTolerance
            );
        }
        assert_eq!(
            predictive_macrostates(&w, &obs, 0, MacroLimits::default()).unwrap_err(),
            MacroError::InvalidHorizon
        );
        let l = MacroLimits {
            max_action_sequences: 7,
            ..MacroLimits::default()
        };
        assert!(matches!(
            predictive_macrostates(&w, &obs, 3, l),
            Err(MacroError::LimitExceeded {
                what: "action_sequences",
                ..
            })
        ));
        let l = MacroLimits {
            max_states: 2,
            ..MacroLimits::default()
        };
        assert!(matches!(
            predictive_macrostates(&w, &obs, 1, l),
            Err(MacroError::LimitExceeded { what: "states", .. })
        ));
        let l = MacroLimits {
            max_frontier: 1,
            ..MacroLimits::default()
        };
        assert!(matches!(
            predictive_macrostates(&w, &obs, 1, l),
            Err(MacroError::LimitExceeded {
                what: "frontier",
                ..
            })
        ));
        let l = MacroLimits {
            max_signature_entries: 1,
            ..MacroLimits::default()
        };
        assert!(matches!(
            predictive_macrostates(&w, &obs, 1, l),
            Err(MacroError::LimitExceeded {
                what: "signature_entries",
                ..
            })
        ));
    }

    #[test]
    fn gog_161_invalid_partition_rejected() {
        let mut part: BTreeMap<MicroState, MacroId> = BTreeMap::new();
        part.insert(s(0), 0);
        assert!(matches!(
            analyze_partition(&fixture(), &obs, &part, 1, MacroLimits::default()),
            Err(MacroError::InvalidPartition { .. })
        ));
    }

    #[test]
    fn gog_161_seeded_worlds_deterministic_and_markov_implies_exact_quotient() {
        let parity = |m: MicroState| m.0 % 2;
        for seed in 0..20u64 {
            let w = BoolWorld::from_seed(seed, 6).unwrap();
            let a = predictive_macrostates(&w, &parity, 3, MacroLimits::default()).unwrap();
            let b = predictive_macrostates(&w, &parity, 3, MacroLimits::default()).unwrap();
            assert_eq!(a, b);
            assert!(a.partition_size >= 1 && a.partition_size <= 6);
            if a.markovian {
                assert!(a.quotient_max_abs_deviation <= 1e-9, "seed {seed}");
            }
        }
    }
}
