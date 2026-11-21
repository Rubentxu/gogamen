//! Escala adaptativa como árbol de preguntas binarias (DEM-035 / GOG-189).
//!
//! Cada nodo interno es un predicado booleano `sentido[i] <= umbral` (un `ExprAst` booleano
//! mínimo, local a este módulo porque `gogamen-core` no depende de `gogamen-spec`). Cada hoja es
//! un macroestado y su identidad es la cadena de respuestas desde la raíz.
//!
//! - **Partir** una hoja: se evalúan cortes `sentido <= umbral` y se elige el de mayor ganancia
//!   en bits sobre tres impurezas ponderadas por muestras: mezcla viable/no viable, error de
//!   predicción (`H(resultado | acción)`) y mezcla de acciones preferidas.
//! - **Parada por longitud mínima de descripción (MDL):** un corte se acepta solo si la ganancia
//!   en bits supera su coste de descripción (`log2(#sentidos) + log2(#umbrales) +
//!   0.5·log2(n)·parámetros`).
//! - **Fundir** hermanas hojas predictivamente equivalentes dentro de una tolerancia.
//! - **Versiones:** cada partir/fundir/reconteo produce un nuevo `Arc<TreeScale>` inmutable con
//!   versión mayor; quien conserve el `Arc` anterior (decisión en curso) mantiene el suyo.
//! - **Límites deterministas:** profundidad, hojas, evaluaciones de candidatos y plazo opcional.
//!
//! No es un inductor de programas ni un árbol de decisión sobre recompensas: ninguna muestra
//! contiene recompensa. Los sentidos duplicados (columnas idénticas) se descartan antes de
//! cortar, de modo que duplicar un sentido no cambia el árbol.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::memory::{EpisodeId, EpisodicRecord};

/// Observación reproducible (p. ej. extraída de memoria episódica) para ajustar el árbol.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// Valores de los sentidos (finitos).
    pub senses: Vec<f64>,
    /// Acción ejecutada.
    pub action: u32,
    /// Clase de resultado observado tras la acción (lo que se predice).
    pub outcome: u32,
    /// Si el estado sigue siendo viable (futuro viable existe).
    pub viable: bool,
    /// Acción preferida en ese estado, si se conoce.
    pub preferred: Option<u32>,
}

/// Etiquetas de ensayo que acompañan a un episodio episódico.
///
/// Los registros episódicos no transportan acción, resultado ni viabilidad; este sidecar las
/// aporta para adaptarlos a [`Sample`].
#[derive(Debug, Clone, PartialEq)]
pub struct EpisodeTrial {
    /// Episodio al que pertenecen las etiquetas.
    pub episode_id: EpisodeId,
    /// Acción ejecutada.
    pub action: u32,
    /// Clase de resultado observado tras la acción.
    pub outcome: u32,
    /// Si el estado sigue siendo viable.
    pub viable: bool,
    /// Acción preferida en ese estado, si se conoce.
    pub preferred: Option<u32>,
}

/// Adapta registros episódicos a muestras usando `trials` como sidecar de etiquetas.
///
/// Los sentidos son `record.numeric_features`. Se conserva el orden de `records` y se omiten los
/// registros sin ensayo asociado; un `episode_id` repetido en `trials` es un error.
pub fn samples_from_episodes(
    records: &[EpisodicRecord],
    trials: &[EpisodeTrial],
) -> Result<Vec<Sample>, QuestionTreeError> {
    let mut by_episode: BTreeMap<EpisodeId, &EpisodeTrial> = BTreeMap::new();
    for trial in trials {
        if by_episode.insert(trial.episode_id, trial).is_some() {
            return Err(QuestionTreeError::DuplicateEpisodeTrial(trial.episode_id));
        }
    }
    Ok(records
        .iter()
        .filter_map(|record| {
            by_episode.get(&record.episode_id).map(|trial| Sample {
                senses: record.numeric_features.clone(),
                action: trial.action,
                outcome: trial.outcome,
                viable: trial.viable,
                preferred: trial.preferred,
            })
        })
        .collect())
}

/// Errores del árbol de preguntas.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionTreeError {
    /// No hay muestras.
    Empty,
    /// Número de sentidos inconsistente entre muestras o con la consulta.
    SenseCountMismatch,
    /// Valor no finito en un sentido.
    NonFiniteSense,
    /// Configuración inválida.
    InvalidConfig(&'static str),
    /// `episode_id` repetido en el sidecar de ensayos.
    DuplicateEpisodeTrial(EpisodeId),
}

impl fmt::Display for QuestionTreeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("no samples"),
            Self::SenseCountMismatch => f.write_str("inconsistent sense count"),
            Self::NonFiniteSense => f.write_str("non-finite sense value"),
            Self::InvalidConfig(m) => write!(f, "invalid config: {m}"),
            Self::DuplicateEpisodeTrial(id) => write!(f, "duplicate episode trial for {id}"),
        }
    }
}

impl Error for QuestionTreeError {}

/// Predicado binario: `senses[sense] <= threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Predicate {
    /// Índice del sentido.
    pub sense: usize,
    /// Umbral.
    pub threshold: f64,
}

impl Predicate {
    /// Evalúa el predicado.
    pub fn holds(&self, senses: &[f64]) -> bool {
        senses[self.sense] <= self.threshold
    }
}

/// Predicado con su respuesta (literal de la conjunción de una hoja).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Literal {
    /// Pregunta.
    pub predicate: Predicate,
    /// Respuesta (`true` = se cumple `<=`).
    pub answer: bool,
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let op = if self.answer { "<=" } else { ">" };
        write!(
            f,
            "s{} {} {}",
            self.predicate.sense, op, self.predicate.threshold
        )
    }
}

/// Identificador de hoja (índice estable dentro de una versión).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LeafId(pub usize);

/// Macroestado: hoja alcanzada y cadena de respuestas desde la raíz.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroAnswer {
    /// Versión de escala usada.
    pub version: (u32, u32),
    /// Hoja.
    pub leaf: LeafId,
    /// Respuestas en orden raíz-hoja.
    pub answers: Vec<bool>,
}

/// Estadísticas de una hoja (conteos reconstruibles por reproducción).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LeafStats {
    /// Muestras.
    pub n: u64,
    /// Muestras viables.
    pub viable: u64,
    /// Conteos `(acción, resultado)`.
    pub outcomes: BTreeMap<(u32, u32), u64>,
    /// Conteos de acción preferida.
    pub preferred: BTreeMap<u32, u64>,
}

impl LeafStats {
    fn add(&mut self, s: &Sample) {
        self.n += 1;
        self.viable += u64::from(s.viable);
        *self.outcomes.entry((s.action, s.outcome)).or_insert(0) += 1;
        if let Some(p) = s.preferred {
            *self.preferred.entry(p).or_insert(0) += 1;
        }
    }

    fn absorb(&mut self, o: &Self) {
        self.n += o.n;
        self.viable += o.viable;
        for (k, v) in &o.outcomes {
            *self.outcomes.entry(*k).or_insert(0) += v;
        }
        for (k, v) in &o.preferred {
            *self.preferred.entry(*k).or_insert(0) += v;
        }
    }

    fn viable_rate(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.viable as f64 / self.n as f64
        }
    }

    /// Distribución de resultados para una acción, o `None` si no se observó.
    fn outcome_dist(&self, action: u32) -> Option<BTreeMap<u32, f64>> {
        let tot: u64 = self
            .outcomes
            .iter()
            .filter(|((a, _), _)| *a == action)
            .map(|(_, c)| *c)
            .sum();
        if tot == 0 {
            return None;
        }
        Some(
            self.outcomes
                .iter()
                .filter(|((a, _), _)| *a == action)
                .map(|((_, o), c)| (*o, *c as f64 / tot as f64))
                .collect(),
        )
    }

    fn preferred_mode(&self) -> Option<u32> {
        self.preferred
            .iter()
            .max_by_key(|(a, c)| (**c, std::cmp::Reverse(**a)))
            .map(|(a, _)| *a)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    Leaf(LeafStats),
    Split {
        pred: Predicate,
        yes: usize,
        no: usize,
    },
}

/// Versión inmutable de la escala (el árbol en un instante).
#[derive(Debug, Clone, PartialEq)]
pub struct TreeScale {
    version: (u32, u32),
    nodes: Vec<Node>,
}

impl TreeScale {
    /// Versión `(mayor, menor)`: el mayor sube con cada partir/fundir/reconteo.
    pub fn version(&self) -> (u32, u32) {
        self.version
    }

    /// Número de hojas.
    pub fn leaf_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| matches!(n, Node::Leaf(_)))
            .count()
    }

    /// Profundidad máxima (raíz = 0).
    pub fn depth(&self) -> usize {
        fn go(nodes: &[Node], i: usize) -> usize {
            match &nodes[i] {
                Node::Leaf(_) => 0,
                Node::Split { yes, no, .. } => 1 + go(nodes, *yes).max(go(nodes, *no)),
            }
        }
        go(&self.nodes, self.root())
    }

    fn root(&self) -> usize {
        0
    }

    /// Clasifica unos sentidos y devuelve el macroestado (cadena de respuestas).
    pub fn classify(&self, senses: &[f64]) -> Result<MacroAnswer, QuestionTreeError> {
        if senses.iter().any(|v| !v.is_finite()) {
            return Err(QuestionTreeError::NonFiniteSense);
        }
        let mut i = self.root();
        let mut answers = Vec::new();
        loop {
            match &self.nodes[i] {
                Node::Leaf(_) => {
                    return Ok(MacroAnswer {
                        version: self.version,
                        leaf: LeafId(i),
                        answers,
                    });
                }
                Node::Split { pred, yes, no } => {
                    if pred.sense >= senses.len() {
                        return Err(QuestionTreeError::SenseCountMismatch);
                    }
                    let a = pred.holds(senses);
                    answers.push(a);
                    i = if a { *yes } else { *no };
                }
            }
        }
    }

    /// Hojas con su conjunción legible de predicados.
    pub fn leaves(&self) -> Vec<(LeafId, Vec<Literal>, &LeafStats)> {
        let mut out = Vec::new();
        fn go<'a>(
            t: &'a TreeScale,
            i: usize,
            path: &mut Vec<Literal>,
            out: &mut Vec<(LeafId, Vec<Literal>, &'a LeafStats)>,
        ) {
            match &t.nodes[i] {
                Node::Leaf(s) => out.push((LeafId(i), path.clone(), s)),
                Node::Split { pred, yes, no } => {
                    path.push(Literal {
                        predicate: *pred,
                        answer: true,
                    });
                    go(t, *yes, path, out);
                    path.pop();
                    path.push(Literal {
                        predicate: *pred,
                        answer: false,
                    });
                    go(t, *no, path, out);
                    path.pop();
                }
            }
        }
        go(self, self.root(), &mut Vec::new(), &mut out);
        out
    }

    /// Conjunción legible de una hoja, p. ej. `s0 <= 3.5 ∧ s1 > 2.5` (`true` si es la raíz).
    pub fn describe_leaf(&self, leaf: LeafId) -> Option<String> {
        self.leaves()
            .into_iter()
            .find(|(l, _, _)| *l == leaf)
            .map(|(_, lits, _)| {
                if lits.is_empty() {
                    "true".to_owned()
                } else {
                    lits.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" ∧ ")
                }
            })
    }

    /// Representación canónica de toda la estructura (para comparar árboles).
    pub fn canonical(&self) -> String {
        fn go(t: &TreeScale, i: usize) -> String {
            match &t.nodes[i] {
                Node::Leaf(s) => format!("L(n={})", s.n),
                Node::Split { pred, yes, no } => format!(
                    "(s{}<={} ? {} : {})",
                    pred.sense,
                    pred.threshold,
                    go(t, *yes),
                    go(t, *no)
                ),
            }
        }
        go(self, self.root())
    }
}

/// Corte registrado.
#[derive(Debug, Clone, PartialEq)]
pub struct CutRecord {
    /// Predicado.
    pub predicate: Predicate,
    /// Conjunción de la hoja partida.
    pub parent: String,
    /// Ganancia en bits.
    pub gain_bits: f64,
    /// Coste MDL en bits.
    pub cost_bits: f64,
    /// Versión resultante.
    pub version: (u32, u32),
}

/// Por qué terminó el crecimiento.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Ningún corte supera su coste MDL.
    Mdl,
    /// Se alcanzó el máximo de hojas.
    MaxLeaves,
    /// Se agotó el presupuesto de evaluaciones o el plazo.
    Budget,
}

/// Límites y tolerancias.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionTreeConfig {
    /// Profundidad máxima.
    pub max_depth: usize,
    /// Hojas máximas.
    pub max_leaves: usize,
    /// Evaluaciones máximas de cortes candidatos (límite determinista de trabajo).
    pub max_evaluations: u64,
    /// Plazo de pared opcional.
    pub time_limit: Option<Duration>,
    /// Muestras mínimas por hoja hija.
    pub min_leaf_samples: u64,
    /// Parámetros libres estimados por hoja en el coste MDL.
    pub params_per_leaf: f64,
    /// Tolerancia de equivalencia predictiva para fundir (variación total, tasa de viabilidad).
    pub merge_tolerance: f64,
}

impl Default for QuestionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: 12,
            max_leaves: 256,
            max_evaluations: 5_000_000,
            time_limit: None,
            min_leaf_samples: 1,
            params_per_leaf: 2.0,
            merge_tolerance: 0.0,
        }
    }
}

/// Entropía de Shannon en bits de conteos.
fn h_counts<'a>(counts: impl Iterator<Item = &'a u64>) -> (f64, u64) {
    let c: Vec<u64> = counts.copied().collect();
    let n: u64 = c.iter().sum();
    if n == 0 {
        return (0.0, 0);
    }
    let h = c
        .iter()
        .filter(|&&k| k > 0)
        .map(|&k| {
            let p = k as f64 / n as f64;
            -p * p.log2()
        })
        .sum();
    (h, n)
}

/// Coste de datos (bits totales de impureza) de un conjunto de muestras.
fn impurity_bits(samples: &[&Sample]) -> f64 {
    let mut via = [0u64; 2];
    let mut out: BTreeMap<(u32, u32), u64> = BTreeMap::new();
    let mut act: BTreeMap<u32, u64> = BTreeMap::new();
    let mut pref: BTreeMap<u32, u64> = BTreeMap::new();
    for s in samples {
        via[usize::from(s.viable)] += 1;
        *out.entry((s.action, s.outcome)).or_insert(0) += 1;
        *act.entry(s.action).or_insert(0) += 1;
        if let Some(p) = s.preferred {
            *pref.entry(p).or_insert(0) += 1;
        }
    }
    let (hv, nv) = h_counts(via.iter());
    let (ho, no) = h_counts(out.values());
    let (ha, _) = h_counts(act.values());
    let (hp, np) = h_counts(pref.values());
    // H(outcome | action) = H(action, outcome) - H(action)
    nv as f64 * hv + no as f64 * (ho - ha).max(0.0) + np as f64 * hp
}

/// Error predictivo ponderado (bits por muestra) de cualquier partición `assign`:
/// `[H(viable|celda) + H(resultado|celda,acción)]`. Útil para comparar contra mallas uniformes.
pub fn predictive_error_bits<F: Fn(&Sample) -> usize>(samples: &[Sample], assign: F) -> f64 {
    let mut groups: BTreeMap<usize, Vec<&Sample>> = BTreeMap::new();
    for s in samples {
        groups.entry(assign(s)).or_default().push(s);
    }
    let mut total = 0.0;
    for g in groups.values() {
        let mut via = [0u64; 2];
        let mut out: BTreeMap<(u32, u32), u64> = BTreeMap::new();
        let mut act: BTreeMap<u32, u64> = BTreeMap::new();
        for s in g {
            via[usize::from(s.viable)] += 1;
            *out.entry((s.action, s.outcome)).or_insert(0) += 1;
            *act.entry(s.action).or_insert(0) += 1;
        }
        let (hv, nv) = h_counts(via.iter());
        let (ho, no) = h_counts(out.values());
        let (ha, _) = h_counts(act.values());
        total += nv as f64 * hv + no as f64 * (ho - ha).max(0.0);
    }
    if samples.is_empty() {
        0.0
    } else {
        total / samples.len() as f64
    }
}

fn validate(samples: &[Sample]) -> Result<usize, QuestionTreeError> {
    let first = samples.first().ok_or(QuestionTreeError::Empty)?;
    let k = first.senses.len();
    for s in samples {
        if s.senses.len() != k {
            return Err(QuestionTreeError::SenseCountMismatch);
        }
        if s.senses.iter().any(|v| !v.is_finite()) {
            return Err(QuestionTreeError::NonFiniteSense);
        }
    }
    Ok(k)
}

/// Mejor corte candidato de una hoja.
struct Candidate {
    pred: Predicate,
    gain: f64,
    cost: f64,
}

/// Árbol adaptativo con historial de cortes y fusiones.
#[derive(Debug, Clone)]
pub struct QuestionTree {
    config: QuestionTreeConfig,
    current: Arc<TreeScale>,
    history: Vec<CutRecord>,
    merges: Vec<(u32, u32)>,
    evaluations: u64,
    stop: Option<StopReason>,
}

impl QuestionTree {
    /// Árbol de una sola hoja vacía (versión `(1,0)`).
    pub fn new(config: QuestionTreeConfig) -> Result<Self, QuestionTreeError> {
        if config.max_leaves == 0 || config.max_evaluations == 0 {
            return Err(QuestionTreeError::InvalidConfig("limits must be positive"));
        }
        if config.merge_tolerance.is_nan()
            || config.merge_tolerance < 0.0
            || config.params_per_leaf.is_nan()
            || config.params_per_leaf < 0.0
        {
            return Err(QuestionTreeError::InvalidConfig("tolerances must be >= 0"));
        }
        Ok(Self {
            config,
            current: Arc::new(TreeScale {
                version: (1, 0),
                nodes: vec![Node::Leaf(LeafStats::default())],
            }),
            history: Vec::new(),
            merges: Vec::new(),
            evaluations: 0,
            stop: None,
        })
    }

    /// Versión vigente; clonar el `Arc` fija la versión para una decisión en curso.
    pub fn scale(&self) -> Arc<TreeScale> {
        Arc::clone(&self.current)
    }

    /// Cortes aceptados en orden; el primero es el «primer corte».
    pub fn history(&self) -> &[CutRecord] {
        &self.history
    }

    /// Versiones producidas por fusiones.
    pub fn merge_versions(&self) -> &[(u32, u32)] {
        &self.merges
    }

    /// Motivo de la última parada del crecimiento.
    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop
    }

    /// Evaluaciones de candidatos consumidas.
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn bump(&mut self, nodes: Vec<Node>) {
        let v = self.current.version;
        self.current = Arc::new(TreeScale {
            version: (v.0 + 1, 0),
            nodes,
        });
    }

    /// Reproduce `replay` y recompone los conteos con la estructura actual (nueva versión).
    pub fn rebuild_counts(
        &mut self,
        replay: &[Sample],
    ) -> Result<Arc<TreeScale>, QuestionTreeError> {
        validate(replay)?;
        let t = recount(&self.current, replay)?;
        self.bump(t.nodes);
        Ok(self.scale())
    }

    /// Crece desde la escala actual con las muestras (partir hasta MDL/límites). Reajusta conteos.
    pub fn grow(&mut self, samples: &[Sample]) -> Result<Arc<TreeScale>, QuestionTreeError> {
        let k = validate(samples)?;
        let started = Instant::now();
        let columns = distinct_columns(samples, k);
        self.rebuild_counts(samples)?;
        self.stop = Some(StopReason::Mdl);
        loop {
            // hojas con sus muestras
            let mut by_leaf: BTreeMap<usize, Vec<&Sample>> = BTreeMap::new();
            for s in samples {
                let leaf = self.current.classify(&s.senses)?.leaf.0;
                by_leaf.entry(leaf).or_default().push(s);
            }
            if self.current.leaf_count() >= self.config.max_leaves {
                self.stop = Some(StopReason::MaxLeaves);
                break;
            }
            let mut best: Option<(usize, Candidate)> = None;
            let mut out_of_budget = false;
            for (leaf, group) in &by_leaf {
                if self.depth_of(*leaf) >= self.config.max_depth {
                    continue;
                }
                match self.best_cut(group, &columns, started) {
                    Ok(Some(c)) => {
                        if best
                            .as_ref()
                            .is_none_or(|(_, b)| c.gain - c.cost > b.gain - b.cost)
                        {
                            best = Some((*leaf, c));
                        }
                    }
                    Ok(None) => {}
                    Err(()) => {
                        out_of_budget = true;
                        break;
                    }
                }
            }
            if out_of_budget {
                self.stop = Some(StopReason::Budget);
                break;
            }
            let Some((leaf, cand)) = best else { break };
            let parent = self.current.describe_leaf(LeafId(leaf)).unwrap_or_default();
            let mut nodes = self.current.nodes.clone();
            let yes = nodes.len();
            nodes.push(Node::Leaf(LeafStats::default()));
            nodes.push(Node::Leaf(LeafStats::default()));
            nodes[leaf] = Node::Split {
                pred: cand.pred,
                yes,
                no: yes + 1,
            };
            let v = self.current.version;
            let version = (v.0 + 1, 0);
            self.current = Arc::new(TreeScale { version, nodes });
            self.history.push(CutRecord {
                predicate: cand.pred,
                parent,
                gain_bits: cand.gain,
                cost_bits: cand.cost,
                version,
            });
            // reponer conteos sin subir la versión otra vez
            let recounted = recount(&self.current, samples)?;
            self.current = Arc::new(recounted);
        }
        Ok(self.scale())
    }

    fn depth_of(&self, target: usize) -> usize {
        fn go(nodes: &[Node], i: usize, target: usize, d: usize) -> Option<usize> {
            if i == target {
                return Some(d);
            }
            match &nodes[i] {
                Node::Leaf(_) => None,
                Node::Split { yes, no, .. } => {
                    go(nodes, *yes, target, d + 1).or_else(|| go(nodes, *no, target, d + 1))
                }
            }
        }
        go(&self.current.nodes, 0, target, 0).unwrap_or(0)
    }

    /// `Err(())` si se agota el presupuesto.
    fn best_cut(
        &mut self,
        group: &[&Sample],
        columns: &[usize],
        started: Instant,
    ) -> Result<Option<Candidate>, ()> {
        let n = group.len() as u64;
        if n < 2 * self.config.min_leaf_samples.max(1) {
            return Ok(None);
        }
        let base = impurity_bits(group);
        if base <= 1e-12 {
            return Ok(None);
        }
        let mut best: Option<Candidate> = None;
        for &sense in columns {
            let mut vals: Vec<f64> = group.iter().map(|s| s.senses[sense]).collect();
            vals.sort_by(f64::total_cmp);
            vals.dedup();
            let thresholds: Vec<f64> = vals.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();
            for &t in &thresholds {
                self.evaluations += 1;
                if self.evaluations > self.config.max_evaluations
                    || self
                        .config
                        .time_limit
                        .is_some_and(|l| started.elapsed() > l)
                {
                    return Err(());
                }
                let (l, r): (Vec<&Sample>, Vec<&Sample>) =
                    group.iter().partition(|s| s.senses[sense] <= t);
                if (l.len() as u64) < self.config.min_leaf_samples
                    || (r.len() as u64) < self.config.min_leaf_samples
                {
                    continue;
                }
                let gain = base - impurity_bits(&l) - impurity_bits(&r);
                let cost = (columns.len() as f64).log2()
                    + (thresholds.len() as f64).log2()
                    + 0.5 * ((n + 1) as f64).log2() * self.config.params_per_leaf;
                if gain > cost + 1e-12 {
                    // empate: gana el sentido menor (se itera en orden) y el umbral menor
                    if best
                        .as_ref()
                        .is_none_or(|b| gain - cost > b.gain - b.cost + 1e-12)
                    {
                        best = Some(Candidate {
                            pred: Predicate {
                                sense,
                                threshold: t,
                            },
                            gain,
                            cost,
                        });
                    }
                }
            }
        }
        Ok(best)
    }

    /// Funde hermanas hojas predictivamente equivalentes (tolerancia de configuración).
    /// Cada fusión produce una versión nueva. Devuelve el número de fusiones.
    pub fn merge_equivalent(&mut self) -> usize {
        let tol = self.config.merge_tolerance;
        let mut merged = 0;
        loop {
            let nodes = &self.current.nodes;
            let target = nodes.iter().enumerate().find_map(|(i, n)| match n {
                Node::Split { yes, no, .. } => match (&nodes[*yes], &nodes[*no]) {
                    (Node::Leaf(a), Node::Leaf(b)) if equivalent(a, b, tol) => {
                        Some((i, a.clone(), b))
                    }
                    _ => None,
                },
                Node::Leaf(_) => None,
            });
            let Some((i, mut stats, b)) = target else {
                break;
            };
            stats.absorb(b);
            let mut nodes = self.current.nodes.clone();
            nodes[i] = Node::Leaf(stats);
            let nodes = compact(nodes);
            self.bump(nodes);
            self.merges.push(self.current.version);
            merged += 1;
        }
        merged
    }
}

fn recount(scale: &TreeScale, samples: &[Sample]) -> Result<TreeScale, QuestionTreeError> {
    let mut t = scale.clone();
    for n in &mut t.nodes {
        if let Node::Leaf(s) = n {
            *s = LeafStats::default();
        }
    }
    for s in samples {
        let leaf = scale.classify(&s.senses)?.leaf.0;
        if let Node::Leaf(st) = &mut t.nodes[leaf] {
            st.add(s);
        }
    }
    Ok(t)
}

fn equivalent(a: &LeafStats, b: &LeafStats, tol: f64) -> bool {
    if a.n == 0 || b.n == 0 {
        return true;
    }
    if (a.viable_rate() - b.viable_rate()).abs() > tol {
        return false;
    }
    let actions: std::collections::BTreeSet<u32> = a
        .outcomes
        .keys()
        .chain(b.outcomes.keys())
        .map(|(act, _)| *act)
        .collect();
    for act in actions {
        if let (Some(da), Some(db)) = (a.outcome_dist(act), b.outcome_dist(act)) {
            let keys: std::collections::BTreeSet<u32> =
                da.keys().chain(db.keys()).copied().collect();
            let tv: f64 = keys
                .iter()
                .map(|k| {
                    (da.get(k).copied().unwrap_or(0.0) - db.get(k).copied().unwrap_or(0.0)).abs()
                })
                .sum::<f64>()
                / 2.0;
            if tv > tol {
                return false;
            }
        }
    }
    a.preferred_mode() == b.preferred_mode()
}

/// Elimina nodos huérfanos y reindexa (raíz en 0).
fn compact(nodes: Vec<Node>) -> Vec<Node> {
    fn copy(src: &[Node], i: usize, dst: &mut Vec<Node>) -> usize {
        let at = dst.len();
        match &src[i] {
            Node::Leaf(s) => dst.push(Node::Leaf(s.clone())),
            Node::Split { pred, yes, no } => {
                dst.push(Node::Leaf(LeafStats::default()));
                let y = copy(src, *yes, dst);
                let n = copy(src, *no, dst);
                dst[at] = Node::Split {
                    pred: *pred,
                    yes: y,
                    no: n,
                };
            }
        }
        at
    }
    let mut dst = Vec::new();
    copy(&nodes, 0, &mut dst);
    dst
}

/// Índices de columnas distintas y no constantes (la primera ocurrencia gana).
fn distinct_columns(samples: &[Sample], k: usize) -> Vec<usize> {
    let cols: Vec<Vec<u64>> = (0..k)
        .map(|j| samples.iter().map(|s| s.senses[j].to_bits()).collect())
        .collect();
    let mut keep: Vec<usize> = Vec::new();
    for j in 0..k {
        let constant = cols[j].windows(2).all(|w| w[0] == w[1]);
        if constant || keep.iter().any(|&i| cols[i] == cols[j]) {
            continue;
        }
        keep.push(j);
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(x: f64, y: f64, action: u32, outcome: u32, viable: bool) -> Sample {
        Sample {
            senses: vec![x, y],
            action,
            outcome,
            viable,
            preferred: None,
        }
    }

    /// Viabilidad depende solo de x > 4.5; y es irrelevante.
    fn data() -> Vec<Sample> {
        let mut v = Vec::new();
        for x in 0..10 {
            for y in 0..10 {
                let bad = x > 4;
                v.push(s(f64::from(x), f64::from(y), 0, u32::from(bad), !bad));
            }
        }
        v
    }

    fn cfg() -> QuestionTreeConfig {
        QuestionTreeConfig::default()
    }

    #[test]
    fn question_tree_first_cut_is_relevant_sense_and_leaves_are_conjunctions() {
        let mut t = QuestionTree::new(cfg()).unwrap();
        let sc = t.grow(&data()).unwrap();
        let first = &t.history()[0];
        assert_eq!(first.predicate.sense, 0);
        assert_eq!(first.predicate.threshold, 4.5);
        assert_eq!(first.parent, "true");
        assert_eq!(sc.leaf_count(), 2, "MDL stops after the only useful cut");
        assert_eq!(t.stop_reason(), Some(StopReason::Mdl));
        let m = sc.classify(&[7.0, 3.0]).unwrap();
        assert_eq!(m.answers, vec![false]);
        assert_eq!(sc.describe_leaf(m.leaf).unwrap(), "s0 > 4.5");
        let m = sc.classify(&[1.0, 3.0]).unwrap();
        assert_eq!(sc.describe_leaf(m.leaf).unwrap(), "s0 <= 4.5");
    }

    #[test]
    fn question_tree_duplicated_irrelevant_sense_does_not_change_tree() {
        let base = data();
        let dup: Vec<Sample> = base
            .iter()
            .map(|b| {
                let mut c = b.clone();
                c.senses.push(b.senses[1]); // duplica el sentido irrelevante
                c.senses.push(b.senses[1]);
                c.senses.push(7.0); // constante
                c
            })
            .collect();
        let mut a = QuestionTree::new(cfg()).unwrap();
        let mut b = QuestionTree::new(cfg()).unwrap();
        let ta = a.grow(&base).unwrap();
        let tb = b.grow(&dup).unwrap();
        assert_eq!(ta.canonical(), tb.canonical());
        assert_eq!(a.history()[0].predicate, b.history()[0].predicate);
    }

    #[test]
    fn question_tree_duplicating_relevant_sense_keeps_lowest_index() {
        let dup: Vec<Sample> = data()
            .iter()
            .map(|b| {
                let mut c = b.clone();
                c.senses.insert(0, b.senses[0]);
                c
            })
            .collect();
        let mut t = QuestionTree::new(cfg()).unwrap();
        t.grow(&dup).unwrap();
        assert_eq!(t.history()[0].predicate.sense, 0);
        assert_eq!(t.scale().leaf_count(), 2);
    }

    #[test]
    fn question_tree_splits_on_mixed_preferred_actions() {
        // viable y resultado constantes; solo difiere la acción preferida según y.
        let v: Vec<Sample> = (0..8)
            .flat_map(|y| {
                (0..4).map(move |x| Sample {
                    senses: vec![f64::from(x), f64::from(y)],
                    action: 0,
                    outcome: 0,
                    viable: true,
                    preferred: Some(u32::from(y >= 4)),
                })
            })
            .collect();
        let mut t = QuestionTree::new(cfg()).unwrap();
        let sc = t.grow(&v).unwrap();
        assert_eq!(t.history()[0].predicate.sense, 1);
        assert_eq!(sc.leaf_count(), 2);
    }

    #[test]
    fn question_tree_versions_are_immutable_for_in_flight_decisions() {
        let mut t = QuestionTree::new(cfg()).unwrap();
        let v1 = t.scale();
        let in_flight = v1.classify(&[7.0, 1.0]);
        // sin datos todavía, la hoja única responde con la raíz
        assert!(in_flight.is_ok());
        let v2 = t.grow(&data()).unwrap();
        assert!(v2.version().0 > v1.version().0);
        assert_eq!(v1.leaf_count(), 1, "old Arc keeps old structure");
        assert_eq!(v1.classify(&[7.0, 1.0]).unwrap().version, (1, 0));
        assert_eq!(v2.classify(&[7.0, 1.0]).unwrap().version, v2.version());
    }

    #[test]
    fn question_tree_merge_creates_new_version_and_respects_tolerance() {
        let mut t = QuestionTree::new(cfg()).unwrap();
        let grown = t.grow(&data()).unwrap();
        assert_eq!(grown.leaf_count(), 2);
        // Nuevo régimen: ya no hay diferencia entre lados -> hermanas equivalentes.
        let calm: Vec<Sample> = data()
            .iter()
            .map(|d| s(d.senses[0], d.senses[1], 0, 0, true))
            .collect();
        let rebuilt = t.rebuild_counts(&calm).unwrap();
        assert!(rebuilt.version().0 > grown.version().0);
        assert_eq!(
            rebuilt.leaf_count(),
            2,
            "rebuild alone does not change structure"
        );
        assert_eq!(t.merge_equivalent(), 1);
        let merged = t.scale();
        assert_eq!(merged.leaf_count(), 1);
        assert!(merged.version().0 > rebuilt.version().0);
        assert_eq!(rebuilt.leaf_count(), 2, "in-flight version unaffected");
        assert_eq!(t.merge_versions().len(), 1);
        // sin tolerancia no se funden hojas distintas
        let mut t2 = QuestionTree::new(cfg()).unwrap();
        t2.grow(&data()).unwrap();
        assert_eq!(t2.merge_equivalent(), 0);
    }

    #[test]
    fn question_tree_limits_prevent_explosion() {
        // cada celda distinta: sin límites crecería mucho (resultado = x*10+y).
        let noisy: Vec<Sample> = (0..16)
            .flat_map(|x| {
                (0..16).map(move |y| s(f64::from(x), f64::from(y), 0, (x * 16 + y) as u32, true))
            })
            .collect();
        let mut c = cfg();
        c.max_leaves = 5;
        let mut t = QuestionTree::new(c).unwrap();
        assert_eq!(t.grow(&noisy).unwrap().leaf_count(), 5);
        assert_eq!(t.stop_reason(), Some(StopReason::MaxLeaves));

        let mut c = cfg();
        c.max_depth = 2;
        let mut t = QuestionTree::new(c).unwrap();
        assert!(t.grow(&noisy).unwrap().depth() <= 2);

        let mut c = cfg();
        c.max_evaluations = 10;
        let mut t = QuestionTree::new(c).unwrap();
        t.grow(&noisy).unwrap();
        assert_eq!(t.stop_reason(), Some(StopReason::Budget));
        assert!(t.evaluations() <= 11);

        let mut c = cfg();
        c.time_limit = Some(Duration::ZERO);
        let mut t = QuestionTree::new(c).unwrap();
        t.grow(&noisy).unwrap();
        assert_eq!(t.stop_reason(), Some(StopReason::Budget));
    }

    #[test]
    fn question_tree_rejects_bad_input() {
        let mut t = QuestionTree::new(cfg()).unwrap();
        assert_eq!(t.grow(&[]).unwrap_err(), QuestionTreeError::Empty);
        let bad = vec![s(f64::NAN, 0.0, 0, 0, true)];
        assert_eq!(t.grow(&bad).unwrap_err(), QuestionTreeError::NonFiniteSense);
        let mut c = cfg();
        c.max_leaves = 0;
        assert!(QuestionTree::new(c).is_err());
    }

    /// 32 episodios de una misma entidad con rasgos `[x, y]`, recuperados por el puerto real.
    /// Las etiquetas viven en el sidecar: viabilidad según `x`, acción según su paridad,
    /// resultado según el régimen de `x` y acción preferida según `y`.
    fn episodic_fixture() -> (Vec<EpisodicRecord>, Vec<EpisodeTrial>) {
        use crate::clock::MonotonicInstant;
        use crate::memory::{
            BoundedEpisodicMemory, EpisodeInput, EpisodeProvenance, MemoryPort, OwnerId,
            RetentionPolicy, SummaryCode,
        };
        use crate::observability::EntityId;
        use crate::provenance::EvidenceRef;
        use crate::units::Tick;

        let policy = RetentionPolicy::new(64, 64, None).unwrap();
        let mut memory = BoundedEpisodicMemory::new(policy);
        let owner = OwnerId::new(7);
        let entity = EntityId::new(1);
        let mut trials = Vec::new();
        for x in 0..8u32 {
            for y in 0..4u32 {
                let tick = Tick::new(u64::from(x) * 4 + u64::from(y));
                let episode_id = memory
                    .record_episode(EpisodeInput::new(
                        owner,
                        entity,
                        tick,
                        MonotonicInstant::from_nanos(u128::from(tick.as_u64()) * 10),
                        tick.as_u64(),
                        EpisodeProvenance::from_evidence(EvidenceRef::new(tick.as_u64())),
                        SummaryCode::new(tick.as_u64()),
                        vec![f64::from(x), f64::from(y)],
                    ))
                    .unwrap();
                let action = x % 2;
                trials.push(EpisodeTrial {
                    episode_id,
                    action,
                    outcome: u32::from(x > 3),
                    viable: x <= 3,
                    preferred: (y >= 2).then_some(action),
                });
            }
        }
        let records = memory
            .query_by_time_and_entity(
                owner,
                entity,
                MonotonicInstant::ZERO,
                MonotonicInstant::from_nanos(10_000),
            )
            .unwrap();
        (records, trials)
    }

    #[test]
    fn question_tree_samples_from_episodes_reproduce_tree_counts() {
        let (records, trials) = episodic_fixture();
        let adapted = samples_from_episodes(&records, &trials).unwrap();
        assert_eq!(adapted.len(), records.len());
        for (i, sample) in adapted.iter().enumerate() {
            assert_eq!(sample.senses, records[i].numeric_features);
            assert_eq!(sample.action, trials[i].action);
            assert_eq!(sample.outcome, trials[i].outcome);
            assert_eq!(sample.viable, trials[i].viable);
            assert_eq!(sample.preferred, trials[i].preferred);
        }

        let hand: Vec<Sample> = records
            .iter()
            .zip(&trials)
            .map(|(r, t)| Sample {
                senses: r.numeric_features.clone(),
                action: t.action,
                outcome: t.outcome,
                viable: t.viable,
                preferred: t.preferred,
            })
            .collect();

        let mut from_memory = QuestionTree::new(cfg()).unwrap();
        let mut from_samples = QuestionTree::new(cfg()).unwrap();
        let memory_scale = from_memory.grow(&adapted).unwrap();
        let sample_scale = from_samples.grow(&hand).unwrap();
        assert!(memory_scale.leaf_count() >= 2, "the fixture must split");
        assert_eq!(memory_scale.canonical(), sample_scale.canonical());
        assert_eq!(memory_scale.leaves(), sample_scale.leaves());
        assert_eq!(from_memory.history(), from_samples.history());
    }

    #[test]
    fn question_tree_samples_from_episodes_skip_unlabelled_records() {
        let (records, trials) = episodic_fixture();
        let labelled: Vec<EpisodeTrial> = trials.iter().step_by(3).cloned().collect();
        let adapted = samples_from_episodes(&records, &labelled).unwrap();
        assert_eq!(adapted.len(), labelled.len());
        let expected: Vec<Vec<f64>> = records
            .iter()
            .enumerate()
            .filter(|(i, _)| i % 3 == 0)
            .map(|(_, r)| r.numeric_features.clone())
            .collect();
        let got: Vec<Vec<f64>> = adapted.into_iter().map(|s| s.senses).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn question_tree_samples_from_episodes_reject_duplicate_trials() {
        let (records, trials) = episodic_fixture();
        let mut duplicated = trials.clone();
        duplicated.push(trials[0].clone());
        assert_eq!(
            samples_from_episodes(&records, &duplicated).unwrap_err(),
            QuestionTreeError::DuplicateEpisodeTrial(trials[0].episode_id)
        );

        let orphan = EpisodeTrial {
            episode_id: EpisodeId::new(9_999),
            ..trials[0].clone()
        };
        assert_eq!(
            samples_from_episodes(&records, &[orphan.clone(), orphan]).unwrap_err(),
            QuestionTreeError::DuplicateEpisodeTrial(EpisodeId::new(9_999))
        );
    }
}
