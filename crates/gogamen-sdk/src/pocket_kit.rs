//! Kit de invariantes de métricas a nivel de MUNDO de bolsillo (GOG-197, DEM-002 v2 / GOG-155).
//!
//! Extiende [`crate::metric_kit`] (distribuciones etiquetadas) a mundos pequeños con estados y
//! acciones etiquetados, transiciones (posiblemente estocásticas) y una medida
//! `measure(mundo, horizonte, bin) -> {H marginal, control, puntuaciones por acción}` aportada por
//! el tercero mediante [`PocketMeasure`]. Es independiente de `labs/*` y de `gogamen-core`
//! (solo `std`); el adaptador al laboratorio vive en `labs/grid/tests/metric_invariants_grid.rs`.
//!
//! # Invariantes (ids estables, parte del hash)
//! - `label_permutation_measure`: permutar etiquetas de estados Y acciones no cambia H ni control
//!   (tolerancia + sigma * errores de muestreo).
//! - `label_permutation_ranking`: ...ni el ranking de acciones por estado (exacto, tras deshacer
//!   la permutación; empates = diferencia <= tolerancia).
//! - `duplicate_predicate_no_gain`: copiar un predicado determinista a una etiqueta nueva no sube
//!   el control.
//! - `exogenous_noise_no_gain`: añadir ruido exógeno independiente no sube el control (sí puede
//!   subir H marginal: el informe las separa).
//! - `random_memory_no_gain`: una memoria con escrituras aleatorias independientes no sube el
//!   control.
//! - `mesh_refinement_ranking`: refinar la malla x2 (`bin_width` mitad) no invierte la acción
//!   preferida en más estados que `max_flip_fraction` de los estados con decisión.
//!
//! # Unidades y errores
//! Toda magnitud lleva unidad (`"bits"`), error estándar de muestreo (0 si exacta) y resolución de
//! bin (`bin_width`, etiquetas por bin). Un fallo devuelve [`PocketFailure`]: id y seed del mundo,
//! seed del kit y contraejemplo MÍNIMO LOCAL (se eliminan estados, acciones y aristas mientras el
//! fallo persista).
//!
//! # Límites
//! Etiquetas `< 32` (los bits 5, 6 y 7 se reservan a las transformaciones); mundos diminutos; las
//! transformaciones son un conjunto finito y fijo, no una prueba de corrección; el contraejemplo es
//! mínimo local, no global; la "acción preferida" depende de la medida aportada. No afirma
//! emergencia ni agencia.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Versión del kit.
pub const POCKET_KIT_VERSION: &str = "0.1.0";
/// Unidad de todas las magnitudes.
pub const POCKET_UNIT: &str = "bits";
/// Las etiquetas de los mundos de entrada deben ser menores que este límite.
pub const POCKET_LABEL_LIMIT: u32 = 32;
const PREDICATE_BIT: u32 = 5;
const MEMORY_BIT: u32 = 6;
const NOISE_BIT: u32 = 7;
/// Ids estables de invariantes, en orden de ejecución.
pub const POCKET_INVARIANT_IDS: [&str; 6] = [
    "label_permutation_measure",
    "label_permutation_ranking",
    "duplicate_predicate_no_gain",
    "exogenous_noise_no_gain",
    "random_memory_no_gain",
    "mesh_refinement_ranking",
];
/// Límites honestos, incluidos en el informe.
pub const POCKET_KIT_LIMITATIONS: &str = "worlds <= 8 states, labels < 32; fixed finite set of \
transformations; counterexample is locally minimal; preferred action is relative to the supplied \
measure; no emergence or agency claim";

/// Hash FNV-1a (64 bits) del contenido del kit (versión, ids, límites de etiqueta).
pub fn pocket_kit_hash() -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    };
    feed(POCKET_KIT_VERSION.as_bytes());
    feed(POCKET_UNIT.as_bytes());
    feed(&u64::from(POCKET_LABEL_LIMIT).to_le_bytes());
    feed(&[PREDICATE_BIT as u8, MEMORY_BIT as u8, NOISE_BIT as u8]);
    for id in POCKET_INVARIANT_IDS {
        feed(id.as_bytes());
    }
    h
}

/// Una acción etiquetada con sus resultados `(estado destino, probabilidad)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketAction {
    /// Etiqueta de acción (única dentro del estado).
    pub label: u32,
    /// Resultados; las probabilidades suman 1.
    pub outcomes: Vec<(u32, f64)>,
}

/// Un estado etiquetado. Sin acciones = absorbente.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketState {
    /// Etiqueta de estado (única).
    pub label: u32,
    /// Acciones disponibles.
    pub actions: Vec<PocketAction>,
}

/// Mundo de bolsillo como dato puro.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketWorld {
    /// Identificador estable.
    pub id: String,
    /// Seed con la que se generó (documentación; 0 si es escrito a mano).
    pub seed: u64,
    /// Estado inicial.
    pub initial: u32,
    /// Estados.
    pub states: Vec<PocketState>,
}

impl PocketWorld {
    /// Estado por etiqueta.
    pub fn state(&self, label: u32) -> Option<&PocketState> {
        self.states.iter().find(|s| s.label == label)
    }

    /// Valida coherencia (etiquetas únicas, destinos declarados, probabilidades que suman 1).
    pub fn validate(&self) -> Result<(), String> {
        if self.state(self.initial).is_none() {
            return Err("estado inicial no declarado".into());
        }
        for (i, s) in self.states.iter().enumerate() {
            if self.states[..i].iter().any(|t| t.label == s.label) {
                return Err(format!("estado duplicado {}", s.label));
            }
            for (j, a) in s.actions.iter().enumerate() {
                if s.actions[..j].iter().any(|b| b.label == a.label) {
                    return Err(format!("acción duplicada {} en {}", a.label, s.label));
                }
                let sum: f64 = a.outcomes.iter().map(|o| o.1).sum();
                if (sum - 1.0).abs() > 1e-9 || a.outcomes.iter().any(|o| o.1.is_nan() || o.1 < 0.0)
                {
                    return Err(format!(
                        "probabilidades inválidas en {}/{}",
                        s.label, a.label
                    ));
                }
                if a.outcomes.iter().any(|o| self.state(o.0).is_none()) {
                    return Err(format!("destino no declarado en {}/{}", s.label, a.label));
                }
            }
        }
        Ok(())
    }

    /// Texto compacto determinista: `world <id> seed=<n> init=<s>: s{a->d:p|d:p; ...}`.
    pub fn render(&self) -> String {
        let mut s = format!(
            "world {} seed={} init={}:",
            self.id, self.seed, self.initial
        );
        for st in &self.states {
            let _ = write!(s, " s{}{{", st.label);
            for (i, a) in st.actions.iter().enumerate() {
                if i > 0 {
                    s.push_str("; ");
                }
                let _ = write!(s, "a{}->", a.label);
                for (j, (d, p)) in a.outcomes.iter().enumerate() {
                    if j > 0 {
                        s.push('|');
                    }
                    let _ = write!(s, "s{d}:{p}");
                }
            }
            s.push('}');
        }
        s
    }
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn world(id: &str, seed: u64, initial: u32, states: Vec<PocketState>) -> PocketWorld {
    PocketWorld {
        id: id.into(),
        seed,
        initial,
        states,
    }
}

fn st(label: u32, actions: Vec<PocketAction>) -> PocketState {
    PocketState { label, actions }
}

fn act(label: u32, outcomes: &[(u32, f64)]) -> PocketAction {
    PocketAction {
        label,
        outcomes: outcomes.to_vec(),
    }
}

fn sink(label: u32) -> PocketState {
    st(label, vec![act(0, &[(label, 1.0)])])
}

/// Los CINCO mundos de bolsillo fijos del kit (documentados en cada `id`):
/// 1. `fork_sinks`: s0 elige entre dos sumideros deterministas (control puro, 1 bit).
/// 2. `pure_noise`: una sola acción con sucesores 1/2-1/2 (H sin control).
/// 3. `lottery`: tres acciones con loterías distintas (determinista, 50/50, sesgada).
/// 4. `two_step_chain`: decisiones encadenadas en dos niveles (opción valiosa en s1).
/// 5. `seeded_random`: 6 estados, 2 acciones, probabilidades diádicas `k/8` generadas con seed 197.
pub fn pocket_worlds() -> Vec<PocketWorld> {
    let fork = world(
        "fork_sinks",
        0,
        0,
        vec![
            st(0, vec![act(0, &[(1, 1.0)]), act(1, &[(2, 1.0)])]),
            sink(1),
            sink(2),
        ],
    );
    let noise = world(
        "pure_noise",
        0,
        0,
        vec![st(0, vec![act(0, &[(1, 0.5), (2, 0.5)])]), sink(1), sink(2)],
    );
    let lottery = world(
        "lottery",
        0,
        0,
        vec![
            st(
                0,
                vec![
                    act(0, &[(1, 1.0)]),
                    act(1, &[(2, 0.5), (3, 0.5)]),
                    act(2, &[(1, 0.75), (3, 0.25)]),
                ],
            ),
            sink(1),
            sink(2),
            sink(3),
        ],
    );
    let chain = world(
        "two_step_chain",
        0,
        0,
        vec![
            st(0, vec![act(0, &[(1, 1.0)]), act(1, &[(2, 1.0)])]),
            st(1, vec![act(0, &[(3, 1.0)]), act(1, &[(4, 1.0)])]),
            st(2, vec![act(0, &[(5, 1.0)])]),
            sink(3),
            sink(4),
            sink(5),
        ],
    );
    let mut rng = 197u64;
    let mut states = Vec::new();
    for s in 0..6u32 {
        let mut actions = Vec::new();
        for a in 0..2u32 {
            let d1 = (splitmix(&mut rng) % 6) as u32;
            let d2 = (splitmix(&mut rng) % 6) as u32;
            let k = (splitmix(&mut rng) % 7 + 1) as f64;
            actions.push(if d1 == d2 {
                act(a, &[(d1, 1.0)])
            } else {
                act(a, &[(d1, k / 8.0), (d2, 1.0 - k / 8.0)])
            });
        }
        states.push(st(s, actions));
    }
    let random = world("seeded_random", 197, 0, states);
    vec![fork, noise, lottery, chain, random]
}

/// Permutaciones aplicadas por [`permute_labels`] (original -> nueva).
#[derive(Debug, Clone, PartialEq)]
pub struct LabelPermutation {
    /// Estados.
    pub states: BTreeMap<u32, u32>,
    /// Acciones.
    pub actions: BTreeMap<u32, u32>,
}

fn shuffled(labels: &[u32], rng: &mut u64) -> BTreeMap<u32, u32> {
    let mut img = labels.to_vec();
    for i in (1..img.len()).rev() {
        let j = (splitmix(rng) % (i as u64 + 1)) as usize;
        img.swap(i, j);
    }
    if img == labels && img.len() > 1 {
        img.rotate_left(1); // evita la permutación identidad
    }
    labels.iter().copied().zip(img).collect()
}

/// Permuta de forma determinista (por `seed`) las etiquetas de estados y de acciones.
pub fn permute_labels(w: &PocketWorld, seed: u64) -> (PocketWorld, LabelPermutation) {
    let mut rng = seed ^ 0x5eed;
    let mut sl: Vec<u32> = w.states.iter().map(|s| s.label).collect();
    sl.sort_unstable();
    let mut al: Vec<u32> = w
        .states
        .iter()
        .flat_map(|s| s.actions.iter().map(|a| a.label))
        .collect();
    al.sort_unstable();
    al.dedup();
    let perm = LabelPermutation {
        states: shuffled(&sl, &mut rng),
        actions: shuffled(&al, &mut rng),
    };
    let mut out = PocketWorld {
        id: w.id.clone(),
        seed: w.seed,
        initial: perm.states[&w.initial],
        states: w
            .states
            .iter()
            .map(|s| PocketState {
                label: perm.states[&s.label],
                actions: s
                    .actions
                    .iter()
                    .map(|a| PocketAction {
                        label: perm.actions[&a.label],
                        outcomes: a
                            .outcomes
                            .iter()
                            .map(|o| (perm.states[&o.0], o.1))
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
    };
    out.states.sort_by_key(|s| s.label);
    for s in &mut out.states {
        s.actions.sort_by_key(|a| a.label);
    }
    (out, perm)
}

fn check_small(w: &PocketWorld) -> Result<(), String> {
    if w.states.iter().any(|s| s.label >= POCKET_LABEL_LIMIT) {
        return Err("etiqueta >= POCKET_LABEL_LIMIT".into());
    }
    Ok(())
}

/// Copia un predicado determinista (paridad de la etiqueta) a un bit nuevo de la etiqueta.
pub fn duplicate_predicate(w: &PocketWorld) -> Result<PocketWorld, String> {
    check_small(w)?;
    let f = |l: u32| l | ((l & 1) << PREDICATE_BIT);
    Ok(PocketWorld {
        id: w.id.clone(),
        seed: w.seed,
        initial: f(w.initial),
        states: w
            .states
            .iter()
            .map(|s| PocketState {
                label: f(s.label),
                actions: s
                    .actions
                    .iter()
                    .map(|a| PocketAction {
                        label: a.label,
                        outcomes: a.outcomes.iter().map(|o| (f(o.0), o.1)).collect(),
                    })
                    .collect(),
            })
            .collect(),
    })
}

/// Añade un bit exógeno justo (remuestreado cada paso, independiente de estado y acción).
pub fn add_exogenous_noise(w: &PocketWorld) -> Result<PocketWorld, String> {
    check_small(w)?;
    let m = 1u32 << NOISE_BIT;
    let mut states = Vec::new();
    for s in &w.states {
        for n in [0, m] {
            states.push(PocketState {
                label: s.label | n,
                actions: s
                    .actions
                    .iter()
                    .map(|a| PocketAction {
                        label: a.label,
                        outcomes: a
                            .outcomes
                            .iter()
                            .flat_map(|o| [(o.0, o.1 / 2.0), (o.0 | m, o.1 / 2.0)])
                            .collect(),
                    })
                    .collect(),
            });
        }
    }
    Ok(PocketWorld {
        id: w.id.clone(),
        seed: w.seed,
        initial: w.initial,
        states,
    })
}

/// Añade un bit de memoria con escrituras aleatorias: en cada paso, con prob. 1/2 se reescribe
/// con un bit justo independiente de estado y acción; si no, se conserva.
pub fn write_random_memory(w: &PocketWorld) -> Result<PocketWorld, String> {
    check_small(w)?;
    let m = 1u32 << MEMORY_BIT;
    let mut states = Vec::new();
    for s in &w.states {
        for mem in [0, m] {
            states.push(PocketState {
                label: s.label | mem,
                actions: s
                    .actions
                    .iter()
                    .map(|a| PocketAction {
                        label: a.label,
                        outcomes: a
                            .outcomes
                            .iter()
                            .flat_map(|o| [(o.0 | mem, o.1 * 0.75), (o.0 | (mem ^ m), o.1 * 0.25)])
                            .collect(),
                    })
                    .collect(),
            });
        }
    }
    Ok(PocketWorld {
        id: w.id.clone(),
        seed: w.seed,
        initial: w.initial,
        states,
    })
}

/// Magnitud con unidad, error de muestreo y resolución de bin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantity {
    /// Valor.
    pub value: f64,
    /// Error estándar de muestreo (0 si exacta).
    pub sampling_error: f64,
    /// Etiquetas por bin (1 = sin agrupar).
    pub bin_width: u32,
    /// Muestras usadas (0 si exacta).
    pub samples: usize,
}

impl Quantity {
    /// Magnitud exacta.
    pub fn exact(value: f64, bin_width: u32) -> Self {
        Self {
            value,
            sampling_error: 0.0,
            bin_width,
            samples: 0,
        }
    }

    fn render(&self) -> String {
        format!(
            "{:.6} {POCKET_UNIT} (se={:.6}, bin_width={}, samples={})",
            self.value, self.sampling_error, self.bin_width, self.samples
        )
    }
}

/// Puntuación de una acción en un estado.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionScore {
    /// Estado.
    pub state: u32,
    /// Acción.
    pub action: u32,
    /// Puntuación (mayor = preferida), en bits.
    pub score: f64,
}

/// Resultado de una medida.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketMeasurement {
    /// H marginal del estado final desde el estado inicial.
    pub marginal_h: Quantity,
    /// Control del estado final desde el estado inicial (p. ej. I(A;S_T)).
    pub control: Quantity,
    /// Puntuaciones por estado y acción.
    pub action_scores: Vec<ActionScore>,
    /// Error estándar de las puntuaciones (0 si exactas).
    pub score_sampling_error: f64,
}

/// Medida de un tercero.
pub trait PocketMeasure {
    /// Mide `world` con `horizon` pasos y malla `bin_width` (etiquetas por bin; menor = más fina).
    fn measure(
        &mut self,
        world: &PocketWorld,
        horizon: usize,
        bin_width: u32,
        seed: u64,
    ) -> Result<PocketMeasurement, String>;
}

impl<F: FnMut(&PocketWorld, usize, u32, u64) -> Result<PocketMeasurement, String>> PocketMeasure
    for F
{
    fn measure(
        &mut self,
        world: &PocketWorld,
        horizon: usize,
        bin_width: u32,
        seed: u64,
    ) -> Result<PocketMeasurement, String> {
        self(world, horizon, bin_width, seed)
    }
}

/// Distribución sobre etiquetas.
pub type Dist = BTreeMap<u32, f64>;

/// Para cada acción de `from`: distribución del estado tras `horizon` pasos (primera acción
/// fija, después acciones uniformes por paso; estados sin acciones son absorbentes).
pub fn first_action_distributions(
    w: &PocketWorld,
    from: u32,
    horizon: usize,
) -> Result<Vec<(u32, Dist)>, String> {
    let s0 = w.state(from).ok_or("estado no declarado")?;
    let mut out = Vec::new();
    for a in &s0.actions {
        let mut dist = Dist::new();
        for (d, p) in &a.outcomes {
            *dist.entry(*d).or_insert(0.0) += p;
        }
        for _ in 1..horizon {
            let mut next = Dist::new();
            for (s, p) in &dist {
                let sp = w.state(*s).ok_or("destino no declarado")?;
                if sp.actions.is_empty() {
                    *next.entry(*s).or_insert(0.0) += p;
                    continue;
                }
                let wa = p / sp.actions.len() as f64;
                for b in &sp.actions {
                    for (d, q) in &b.outcomes {
                        *next.entry(*d).or_insert(0.0) += wa * q;
                    }
                }
            }
            dist = next;
        }
        out.push((a.label, dist));
    }
    Ok(out)
}

fn bin(d: &Dist, bin_width: u32) -> Dist {
    let mut o = Dist::new();
    for (l, p) in d {
        *o.entry(l / bin_width.max(1)).or_insert(0.0) += p;
    }
    o
}

fn entropy_bits(d: &Dist) -> f64 {
    d.values()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.log2())
        .sum()
}

fn kl_bits(p: &Dist, q: &Dist) -> f64 {
    p.iter()
        .filter(|(_, v)| **v > 0.0)
        .map(|(k, v)| v * (v / q[k]).log2())
        .sum()
}

/// Medida de referencia EXACTA (sin muestreo): control = I(A1; S_T) con A1 uniforme sobre la
/// primera acción y acciones uniformes después; H marginal = H(S_T) de la mezcla; puntuación de
/// `(s, a)` = KL(P(S_T|a) || P(S_T)) desde `s` (cuánto "dice" elegir `a`). Binning
/// `etiqueta / bin_width`. Es invariante bajo renombrado por construcción.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferencePocketMeasure;

/// `(H marginal, control, puntuaciones por acción)` en un estado.
type AtState = (f64, f64, Vec<(u32, f64)>);

impl ReferencePocketMeasure {
    fn at(w: &PocketWorld, s: u32, h: usize, bw: u32) -> Result<AtState, String> {
        let per = first_action_distributions(w, s, h)?;
        if per.is_empty() {
            return Ok((
                entropy_bits(&bin(&BTreeMap::from([(s, 1.0)]), bw)),
                0.0,
                vec![],
            ));
        }
        let n = per.len() as f64;
        let mut mix = Dist::new();
        let binned: Vec<(u32, Dist)> = per.iter().map(|(a, d)| (*a, bin(d, bw))).collect();
        for (_, d) in &binned {
            for (k, v) in d {
                *mix.entry(*k).or_insert(0.0) += v / n;
            }
        }
        let scores: Vec<(u32, f64)> = binned.iter().map(|(a, d)| (*a, kl_bits(d, &mix))).collect();
        let control = scores.iter().map(|x| x.1).sum::<f64>() / n;
        Ok((entropy_bits(&mix), control.max(0.0), scores))
    }
}

impl PocketMeasure for ReferencePocketMeasure {
    fn measure(
        &mut self,
        world: &PocketWorld,
        horizon: usize,
        bin_width: u32,
        _seed: u64,
    ) -> Result<PocketMeasurement, String> {
        if horizon == 0 || bin_width == 0 {
            return Err("horizon y bin_width deben ser >= 1".into());
        }
        world.validate()?;
        let (h, c, _) = Self::at(world, world.initial, horizon, bin_width)?;
        let mut action_scores = Vec::new();
        for s in &world.states {
            let (_, _, sc) = Self::at(world, s.label, horizon, bin_width)?;
            action_scores.extend(sc.into_iter().map(|(action, score)| ActionScore {
                state: s.label,
                action,
                score,
            }));
        }
        Ok(PocketMeasurement {
            marginal_h: Quantity::exact(h, bin_width),
            control: Quantity::exact(c, bin_width),
            action_scores,
            score_sampling_error: 0.0,
        })
    }
}

/// Configuración del kit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PocketKitConfig {
    /// Seed del kit (permutaciones y seed pasada a la medida).
    pub seed: u64,
    /// Horizonte (>= 1).
    pub horizon: usize,
    /// Malla gruesa (etiquetas por bin); la fina es la mitad (mínimo 1).
    pub coarse_bin_width: u32,
    /// Tolerancia numérica en bits.
    pub tolerance: f64,
    /// Número de errores estándar añadidos a la tolerancia.
    pub sigma: f64,
    /// Fracción máxima declarada de estados con decisión cuya acción preferida puede cambiar.
    pub max_flip_fraction: f64,
}

impl Default for PocketKitConfig {
    fn default() -> Self {
        Self {
            seed: 197,
            horizon: 2,
            coarse_bin_width: 2,
            tolerance: 1e-9,
            sigma: 3.0,
            max_flip_fraction: 0.5,
        }
    }
}

/// Fallo con mundo, seed y contraejemplo mínimo.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketFailure {
    /// Id del mundo original.
    pub world_id: String,
    /// Seed del mundo original.
    pub world_seed: u64,
    /// Seed del kit.
    pub kit_seed: u64,
    /// Estados del mundo original.
    pub original_states: usize,
    /// Contraejemplo reducido.
    pub counterexample: PocketWorld,
    /// Detalle de los valores observados sobre el contraejemplo.
    pub detail: String,
    /// Pasos de shrinking aceptados.
    pub shrink_steps: usize,
}

/// Resultado por invariante.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketInvariantResult {
    /// Id.
    pub id: &'static str,
    /// Mundos evaluados hasta el primer fallo.
    pub worlds_run: usize,
    /// Primer fallo, si lo hay.
    pub failure: Option<PocketFailure>,
}

/// Fila de control por mundo: separa H marginal y control, con unidades y resolución.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketControlRow {
    /// Mundo.
    pub world_id: String,
    /// Seed del mundo.
    pub seed: u64,
    /// Base.
    pub base: Option<(Quantity, Quantity)>,
    /// Con ruido exógeno.
    pub noise: Option<(Quantity, Quantity)>,
    /// Con memoria aleatoria.
    pub memory: Option<(Quantity, Quantity)>,
}

/// Informe del kit.
#[derive(Debug, Clone, PartialEq)]
pub struct PocketKitReport {
    /// Nombre de la medida.
    pub measure: String,
    /// Versión.
    pub kit_version: &'static str,
    /// Hash FNV-1a.
    pub kit_hash: u64,
    /// Configuración.
    pub config: PocketKitConfig,
    /// Resultados por invariante.
    pub results: Vec<PocketInvariantResult>,
    /// Filas H marginal / control por mundo.
    pub rows: Vec<PocketControlRow>,
}

impl PocketKitReport {
    /// `true` si todo pasa.
    pub fn passed(&self) -> bool {
        self.results.iter().all(|r| r.failure.is_none())
    }

    /// Resultado por id.
    pub fn result(&self, id: &str) -> Option<&PocketInvariantResult> {
        self.results.iter().find(|r| r.id == id)
    }

    /// Texto determinista.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "gogamen pocket-kit v{} hash={:016x} unit={POCKET_UNIT}",
            self.kit_version, self.kit_hash
        );
        let _ = writeln!(
            s,
            "measure={} seed={} horizon={} coarse_bin_width={} verdict={}",
            self.measure,
            self.config.seed,
            self.config.horizon,
            self.config.coarse_bin_width,
            if self.passed() { "PASS" } else { "FAIL" }
        );
        for r in &self.rows {
            let _ = writeln!(s, "  world {} seed={}", r.world_id, r.seed);
            for (name, v) in [("base", r.base), ("noise", r.noise), ("memory", r.memory)] {
                if let Some((h, c)) = v {
                    let _ = writeln!(s, "    {name}: H={} | control={}", h.render(), c.render());
                }
            }
        }
        for r in &self.results {
            match &r.failure {
                None => {
                    let _ = writeln!(s, "  [PASS] {} ({} worlds)", r.id, r.worlds_run);
                }
                Some(f) => {
                    let _ = writeln!(
                        s,
                        "  [FAIL] {} world={} world_seed={} kit_seed={} states {} -> {} (shrink steps {})\n         {}\n         {}",
                        r.id,
                        f.world_id,
                        f.world_seed,
                        f.kit_seed,
                        f.original_states,
                        f.counterexample.states.len(),
                        f.shrink_steps,
                        f.counterexample.render(),
                        f.detail
                    );
                }
            }
        }
        let _ = writeln!(s, "  limitations: {POCKET_KIT_LIMITATIONS}");
        s
    }
}

enum Outcome {
    Pass,
    Fail(String),
}

type Canon = BTreeMap<u32, Vec<Vec<u32>>>;

fn canon(
    m: &PocketMeasurement,
    smap: &BTreeMap<u32, u32>,
    amap: &BTreeMap<u32, u32>,
    tol: f64,
) -> Canon {
    let mut per: BTreeMap<u32, Vec<(f64, u32)>> = BTreeMap::new();
    for a in &m.action_scores {
        per.entry(smap.get(&a.state).copied().unwrap_or(a.state))
            .or_default()
            .push((a.score, amap.get(&a.action).copied().unwrap_or(a.action)));
    }
    per.into_iter()
        .map(|(s, mut v)| {
            v.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
            let mut groups: Vec<Vec<u32>> = Vec::new();
            let mut last = f64::NAN;
            for (sc, a) in v {
                if groups.is_empty() || (last - sc) > tol {
                    groups.push(vec![]);
                }
                last = sc;
                groups.last_mut().expect("grupo").push(a);
            }
            for g in &mut groups {
                g.sort_unstable();
            }
            (s, groups)
        })
        .collect()
}

fn identity(w: &PocketWorld) -> (BTreeMap<u32, u32>, BTreeMap<u32, u32>) {
    let s = w.states.iter().map(|s| (s.label, s.label)).collect();
    let a = w
        .states
        .iter()
        .flat_map(|s| s.actions.iter().map(|a| (a.label, a.label)))
        .collect();
    (s, a)
}

fn invert(m: &BTreeMap<u32, u32>) -> BTreeMap<u32, u32> {
    m.iter().map(|(k, v)| (*v, *k)).collect()
}

fn check(id: &str, m: &mut dyn PocketMeasure, w: &PocketWorld, cfg: &PocketKitConfig) -> Outcome {
    let horizon = cfg.horizon;
    let tol = |a: f64, b: f64| cfg.tolerance + cfg.sigma * (a + b);
    let mut meas = |x: &PocketWorld, bw: u32| {
        m.measure(x, horizon, bw, cfg.seed ^ w.seed)
            .map_err(|e| format!("measure error: {e}"))
    };
    let r: Result<Option<String>, String> = (|| match id {
        "label_permutation_measure" | "label_permutation_ranking" => {
            let (pw, perm) = permute_labels(w, cfg.seed ^ w.seed);
            let a = meas(w, 1)?;
            let b = meas(&pw, 1)?;
            if id == "label_permutation_measure" {
                for (n, x, y) in [
                    ("H", a.marginal_h, b.marginal_h),
                    ("control", a.control, b.control),
                ] {
                    let d = (x.value - y.value).abs();
                    if d > tol(x.sampling_error, y.sampling_error) {
                        return Ok(Some(format!(
                            "{n}: {:.9} vs {:.9} bits after permutation {:?}/{:?}",
                            x.value, y.value, perm.states, perm.actions
                        )));
                    }
                }
                return Ok(None);
            }
            let t = tol(a.score_sampling_error, b.score_sampling_error);
            let (is, ia) = identity(w);
            let ca = canon(&a, &is, &ia, t);
            let cb = canon(&b, &invert(&perm.states), &invert(&perm.actions), t);
            Ok((ca != cb).then(|| format!("ranking {ca:?} vs permuted {cb:?}")))
        }
        "duplicate_predicate_no_gain" | "exogenous_noise_no_gain" | "random_memory_no_gain" => {
            let tw = match id {
                "duplicate_predicate_no_gain" => duplicate_predicate(w)?,
                "exogenous_noise_no_gain" => add_exogenous_noise(w)?,
                _ => write_random_memory(w)?,
            };
            let a = meas(w, 1)?;
            let b = meas(&tw, 1)?;
            let gain = b.control.value - a.control.value;
            Ok(
                (gain > tol(a.control.sampling_error, b.control.sampling_error)).then(|| {
                    format!(
                        "control rose {:.9} -> {:.9} bits (+{gain:.9}); H {:.9} -> {:.9} bits",
                        a.control.value, b.control.value, a.marginal_h.value, b.marginal_h.value
                    )
                }),
            )
        }
        "mesh_refinement_ranking" => {
            let coarse = cfg.coarse_bin_width.max(1);
            let fine = (coarse / 2).max(1);
            let a = meas(w, coarse)?;
            let b = meas(w, fine)?;
            let t = tol(a.score_sampling_error, b.score_sampling_error);
            let (is, ia) = identity(w);
            let (ca, cb) = (canon(&a, &is, &ia, t), canon(&b, &is, &ia, t));
            let decision = ca
                .values()
                .filter(|g| g.iter().map(Vec::len).sum::<usize>() > 1)
                .count();
            let flipped: Vec<u32> = ca
                .iter()
                .filter(|(s, g)| g.first() != cb.get(*s).and_then(|x| x.first()))
                .map(|(s, _)| *s)
                .collect();
            let allowed = (cfg.max_flip_fraction * decision as f64).floor() as usize;
            Ok((flipped.len() > allowed).then(|| {
                format!(
                    "preferred action flipped in states {flipped:?} ({} > allowed {allowed} of {decision} decision states) refining bin_width {coarse} -> {fine}",
                    flipped.len()
                )
            }))
        }
        other => Err(format!("invariante desconocido {other}")),
    })();
    match r {
        Ok(None) => Outcome::Pass,
        Ok(Some(d)) => Outcome::Fail(d),
        Err(e) => Outcome::Fail(e),
    }
}

fn renormalize(a: &mut PocketAction) {
    let sum: f64 = a.outcomes.iter().map(|o| o.1).sum();
    for o in &mut a.outcomes {
        o.1 /= sum;
    }
}

fn shrink_candidates(w: &PocketWorld) -> Vec<PocketWorld> {
    let mut out = Vec::new();
    for s in &w.states {
        if s.label != w.initial {
            let mut c = w.clone();
            c.states.retain(|x| x.label != s.label);
            for x in &mut c.states {
                for a in &mut x.actions {
                    a.outcomes.retain(|o| o.0 != s.label);
                }
                x.actions.retain(|a| !a.outcomes.is_empty());
                for a in &mut x.actions {
                    renormalize(a);
                }
            }
            out.push(c);
        }
    }
    for (i, s) in w.states.iter().enumerate() {
        for j in 0..s.actions.len() {
            let mut c = w.clone();
            c.states[i].actions.remove(j);
            out.push(c);
        }
    }
    for (i, s) in w.states.iter().enumerate() {
        for (j, a) in s.actions.iter().enumerate() {
            if a.outcomes.len() > 1 {
                for k in 0..a.outcomes.len() {
                    let mut c = w.clone();
                    c.states[i].actions[j].outcomes.remove(k);
                    renormalize(&mut c.states[i].actions[j]);
                    out.push(c);
                }
            }
        }
    }
    out.retain(|c| c.validate().is_ok());
    out
}

fn shrink(
    id: &str,
    m: &mut dyn PocketMeasure,
    w: &PocketWorld,
    cfg: &PocketKitConfig,
    first_detail: String,
) -> (PocketWorld, String, usize) {
    let (mut cur, mut detail, mut steps) = (w.clone(), first_detail, 0);
    'outer: while steps < 1000 {
        for c in shrink_candidates(&cur) {
            if let Outcome::Fail(d) = check(id, m, &c, cfg) {
                if d.starts_with("measure error") {
                    continue; // un error de medida no es el mismo fallo
                }
                cur = c;
                detail = d;
                steps += 1;
                continue 'outer;
            }
        }
        break;
    }
    (cur, detail, steps)
}

fn row(
    m: &mut dyn PocketMeasure,
    w: &PocketWorld,
    tw: Result<PocketWorld, String>,
    cfg: &PocketKitConfig,
) -> Option<(Quantity, Quantity)> {
    let x = tw.ok()?;
    let r = m.measure(&x, cfg.horizon, 1, cfg.seed ^ w.seed).ok()?;
    Some((r.marginal_h, r.control))
}

/// Ejecuta el kit sobre `worlds` (usar [`pocket_worlds`] para los cinco fijos).
pub fn run_pocket_kit(
    name: &str,
    m: &mut dyn PocketMeasure,
    worlds: &[PocketWorld],
    cfg: &PocketKitConfig,
) -> PocketKitReport {
    let mut results = Vec::new();
    for id in POCKET_INVARIANT_IDS {
        let mut run = 0;
        let mut failure = None;
        for w in worlds {
            run += 1;
            let outcome = match w.validate() {
                Ok(()) => check(id, m, w, cfg),
                Err(e) => Outcome::Fail(format!("mundo inválido: {e}")),
            };
            if let Outcome::Fail(d) = outcome {
                let (ce, detail, steps) = shrink(id, m, w, cfg, d);
                failure = Some(PocketFailure {
                    world_id: w.id.clone(),
                    world_seed: w.seed,
                    kit_seed: cfg.seed,
                    original_states: w.states.len(),
                    counterexample: ce,
                    detail,
                    shrink_steps: steps,
                });
                break;
            }
        }
        results.push(PocketInvariantResult {
            id,
            worlds_run: run,
            failure,
        });
    }
    let rows = worlds
        .iter()
        .map(|w| PocketControlRow {
            world_id: w.id.clone(),
            seed: w.seed,
            base: row(m, w, Ok(w.clone()), cfg),
            noise: row(m, w, add_exogenous_noise(w), cfg),
            memory: row(m, w, write_random_memory(w), cfg),
        })
        .collect();
    PocketKitReport {
        measure: name.into(),
        kit_version: POCKET_KIT_VERSION,
        kit_hash: pocket_kit_hash(),
        config: *cfg,
        results,
        rows,
    }
}
