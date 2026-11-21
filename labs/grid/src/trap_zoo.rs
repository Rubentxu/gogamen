//! Zoológico de trampas entrópicas: reductor 1-minimal y galería (GOG-192).
//!
//! # Qué hace
//! - [`TrapWorld`] es un mundo de bolsillo editable (estados viables/inviables, acciones con
//!   aristas ponderadas por enteros) convertible a [`BoolWorld`] (GOG-154) para usar su oráculo
//!   exacto. Las probabilidades son `peso / suma`, de modo que quitar una arista renormaliza.
//! - Un [`Principle`] es un puntuador de acciones en el estado inicial; elige la acción de
//!   mayor puntuación (empates dentro de `1e-9`: la de menor índice, determinista).
//!   - [`Principle::GreedyFinalEntropy`]: entropía del microestado final a 1 paso.
//!   - [`Principle::TrajectoryEntropy`]: entropía de trayectorias a horizonte `H`.
//!   - [`Principle::SurvivalMass`]: control; maximiza la masa viable (no puede fallar).
//! - **Predicado de fallo (exacto, basado en el oráculo)**: sea `m(a)` la `viable_mass` cruda
//!   del oráculo con horizonte `H` en el mundo restringido a la acción `a` en el inicial y
//!   política uniforme después. El principio *falla* si hay `b` con `m(b) > m(elegida) + 1e-9`:
//!   eligió una acción con menos masa de futuros viables que otra. Las entropías del oráculo se
//!   calculan sobre masa renormalizada y ahí está la ceguera a la trampa.
//! - [`reduce`] elimina estados y aristas de uno en uno (primer ajuste, reinicio tras aceptar)
//!   hasta que ninguna eliminación individual conserve el fallo (1-minimal, estilo
//!   delta-debugging). Cada eliminación aceptada reduce el tamaño, luego termina en
//!   `<= estados + aristas` iteraciones.
//!
//! # Límites
//! Evidencia de laboratorio sobre mundos diminutos; no hace afirmaciones de emergencia ni
//! agencia. La minimalidad es local (1-minimal), no mínima global.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::bool_world::{BoolSamplingPolicy, BoolWorld, BoolWorldError, MicroState, OracleLimits};

/// Cabecera del formato de fixture.
pub const TRAP_ZOO_FORMAT: &str = "gogamen-trap-zoo v1";
const EPS: f64 = 1e-9;

/// Principio de selección de acción en el estado inicial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Principle {
    /// Maximiza la entropía del microestado final a un paso.
    GreedyFinalEntropy,
    /// Maximiza la entropía de trayectorias a horizonte `H`.
    TrajectoryEntropy,
    /// Control: maximiza la masa viable a horizonte `H`.
    SurvivalMass,
}

impl Principle {
    /// Todos los principios.
    pub const ALL: [Principle; 3] = [
        Principle::GreedyFinalEntropy,
        Principle::TrajectoryEntropy,
        Principle::SurvivalMass,
    ];

    /// Nombre estable.
    pub fn name(self) -> &'static str {
        match self {
            Self::GreedyFinalEntropy => "greedy_final_entropy",
            Self::TrajectoryEntropy => "trajectory_entropy",
            Self::SurvivalMass => "survival_mass",
        }
    }

    /// Parsea un nombre estable.
    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|p| p.name() == s)
    }
}

/// Estado de un [`TrapWorld`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrapState {
    /// Si es viable (los inviables se excluyen del soporte del oráculo).
    pub viable: bool,
    /// Acciones; cada una es una lista de aristas `(destino, peso)`.
    pub actions: Vec<Vec<(u8, u32)>>,
}

/// Mundo de bolsillo editable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrapWorld {
    /// Seed de identificación.
    pub seed: u64,
    /// Estado inicial (nunca se elimina).
    pub initial: u8,
    /// Estados por id.
    pub states: BTreeMap<u8, TrapState>,
}

impl TrapWorld {
    /// Número de estados.
    pub fn num_states(&self) -> usize {
        self.states.len()
    }

    /// Número de aristas.
    pub fn num_edges(&self) -> usize {
        self.states
            .values()
            .map(|s| s.actions.iter().map(Vec::len).sum::<usize>())
            .sum()
    }

    /// Genera un mundo determinista: `n` estados (`2..=8`), el 0 viable con 2-3 acciones, el
    /// resto viable con prob. 2/3 y 1-3 acciones de 1-3 aristas de peso 1-4.
    pub fn generate(seed: u64, n: u8) -> Self {
        let mut rng = seed ^ (u64::from(n) << 56);
        let mut next = move || {
            rng = rng.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        };
        let n = n.clamp(2, 8);
        let mut states = BTreeMap::new();
        for id in 0..n {
            let viable = id == 0 || next() % 3 != 0;
            let mut actions = Vec::new();
            if viable {
                let na = if id == 0 {
                    2 + next() % 2
                } else {
                    1 + next() % 3
                };
                for _ in 0..na {
                    let ne = 1 + next() % 3;
                    let mut edges: Vec<(u8, u32)> = Vec::new();
                    for _ in 0..ne {
                        let d = (next() % u64::from(n)) as u8;
                        let w = 1 + (next() % 4) as u32;
                        if !edges.iter().any(|(x, _)| *x == d) {
                            edges.push((d, w));
                        }
                    }
                    actions.push(edges);
                }
            }
            states.insert(id, TrapState { viable, actions });
        }
        Self {
            seed,
            initial: 0,
            states,
        }
    }

    /// Convierte a [`BoolWorld`]; `restrict` deja solo esa acción en el estado inicial.
    pub fn to_bool(&self, restrict: Option<usize>) -> Result<BoolWorld, BoolWorldError> {
        let mut w = BoolWorld::new(self.seed);
        for (id, st) in &self.states {
            let acts: Vec<&Vec<(u8, u32)>> = match restrict {
                Some(a) if *id == self.initial => st.actions.get(a).into_iter().collect(),
                _ => st.actions.iter().collect(),
            };
            let acts = acts
                .into_iter()
                .map(|edges| {
                    let sum: u32 = edges.iter().map(|(_, w)| *w).sum();
                    edges
                        .iter()
                        .map(|(d, w)| (MicroState(*d), f64::from(*w) / f64::from(sum)))
                        .collect()
                })
                .collect();
            w = w.with_state(MicroState(*id), st.viable, acts)?;
        }
        Ok(w)
    }
}

/// Evaluación de un principio sobre un mundo.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// Puntuación del principio por acción.
    pub scores: Vec<f64>,
    /// Masa viable del oráculo por acción (horizonte `H`).
    pub masses: Vec<f64>,
    /// Acción elegida por el principio.
    pub chosen: usize,
    /// Acción con mayor masa viable (menor índice entre empates).
    pub best: usize,
}

impl Evaluation {
    /// Predicado de fallo exacto: la elegida tiene estrictamente menos masa viable.
    pub fn fails(&self) -> bool {
        self.masses[self.best] > self.masses[self.chosen] + EPS
    }
}

fn first_argmax(v: &[f64]) -> usize {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    v.iter().position(|x| *x >= max - EPS).unwrap_or(0)
}

/// Evalúa `principle` en `world` con horizonte `horizon`. `None` si no hay elección (menos de
/// 2 acciones), el inicial es inválido o el oráculo falla por otra causa que la extinción.
pub fn evaluate(world: &TrapWorld, principle: Principle, horizon: usize) -> Option<Evaluation> {
    let init = world.states.get(&world.initial)?;
    if init.actions.len() < 2 || !init.viable {
        return None;
    }
    let pol = BoolSamplingPolicy::UniformActions;
    let lim = OracleLimits::default();
    let (mut scores, mut masses) = (Vec::new(), Vec::new());
    for a in 0..init.actions.len() {
        let bw = world.to_bool(Some(a)).ok()?;
        let at = |h: usize| match bw.oracle(MicroState(world.initial), &pol, h, lim) {
            Ok(r) => Some(Some(r)),
            Err(BoolWorldError::NoViableTrajectory) => Some(None),
            Err(_) => None,
        };
        let full = at(horizon)?;
        let mass = full.as_ref().map_or(0.0, |r| r.viable_mass);
        let score = match principle {
            Principle::GreedyFinalEntropy => {
                at(1)?.map_or(f64::NEG_INFINITY, |r| r.final_state_entropy_bits)
            }
            Principle::TrajectoryEntropy => full
                .as_ref()
                .map_or(f64::NEG_INFINITY, |r| r.trajectory_entropy_bits),
            Principle::SurvivalMass => mass,
        };
        scores.push(score);
        masses.push(mass);
    }
    let chosen = first_argmax(&scores);
    let best = first_argmax(&masses);
    Some(Evaluation {
        scores,
        masses,
        chosen,
        best,
    })
}

/// `true` si `principle` falla en `world` (ver predicado en la documentación del módulo).
pub fn fails(world: &TrapWorld, principle: Principle, horizon: usize) -> bool {
    evaluate(world, principle, horizon).is_some_and(|e| e.fails())
}

/// Eliminación atómica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Removal {
    /// Quita un estado (y las aristas que llegan a él).
    State(u8),
    /// Quita la arista `index` de la acción `action` del estado `state`.
    Edge {
        /// Estado origen.
        state: u8,
        /// Índice de acción.
        action: usize,
        /// Índice de arista.
        index: usize,
    },
}

/// Todas las eliminaciones individuales posibles, en orden determinista (estados, aristas).
pub fn single_removals(w: &TrapWorld) -> Vec<Removal> {
    let mut v: Vec<Removal> = w
        .states
        .keys()
        .filter(|s| **s != w.initial)
        .map(|s| Removal::State(*s))
        .collect();
    for (id, st) in &w.states {
        for (a, edges) in st.actions.iter().enumerate() {
            for index in 0..edges.len() {
                v.push(Removal::Edge {
                    state: *id,
                    action: a,
                    index,
                });
            }
        }
    }
    v
}

/// Aplica una eliminación (acciones vacías se descartan).
pub fn apply_removal(w: &TrapWorld, r: &Removal) -> TrapWorld {
    let mut out = w.clone();
    match r {
        Removal::State(s) => {
            out.states.remove(s);
            for st in out.states.values_mut() {
                for a in &mut st.actions {
                    a.retain(|(d, _)| d != s);
                }
            }
        }
        Removal::Edge {
            state,
            action,
            index,
        } => {
            if let Some(a) = out
                .states
                .get_mut(state)
                .and_then(|s| s.actions.get_mut(*action))
                && *index < a.len()
            {
                a.remove(*index);
            }
        }
    }
    for st in out.states.values_mut() {
        st.actions.retain(|a| !a.is_empty());
    }
    out
}

/// Resultado de la reducción.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reduction {
    /// Mundo 1-minimal.
    pub world: TrapWorld,
    /// Eliminaciones aceptadas.
    pub accepted: usize,
}

/// Reduce `world` conservando el fallo de `principle` a horizonte `horizon`. `None` si el mundo
/// original no falla. Determinista y acotado por `estados + aristas` aceptaciones.
pub fn reduce(world: &TrapWorld, principle: Principle, horizon: usize) -> Option<Reduction> {
    if !fails(world, principle, horizon) {
        return None;
    }
    let mut cur = world.clone();
    let mut accepted = 0;
    let bound = world.num_states() + world.num_edges();
    while accepted <= bound {
        let next = single_removals(&cur)
            .iter()
            .map(|r| apply_removal(&cur, r))
            .find(|c| fails(c, principle, horizon));
        match next {
            Some(c) => {
                cur = c;
                accepted += 1;
            }
            None => break,
        }
    }
    Some(Reduction {
        world: cur,
        accepted,
    })
}

/// `true` si ninguna eliminación individual conserva el fallo.
pub fn is_one_minimal(w: &TrapWorld, principle: Principle, horizon: usize) -> bool {
    single_removals(w)
        .iter()
        .all(|r| !fails(&apply_removal(w, r), principle, horizon))
}

/// Trampa archivada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrapRecord {
    /// Principio que falla.
    pub principle: Principle,
    /// Escala: estados del mundo original.
    pub scale: usize,
    /// Horizonte.
    pub horizon: usize,
    /// Seed del mundo original.
    pub seed: u64,
    /// Estados antes de reducir.
    pub orig_states: usize,
    /// Aristas antes de reducir.
    pub orig_edges: usize,
    /// Acción elegida por el principio.
    pub chosen: usize,
    /// Acción con más masa viable.
    pub alternative: usize,
    /// `stable_hash` del oráculo (acción elegida, horizonte `H`).
    pub failing_hash: u64,
    /// Mundo mínimo.
    pub world: TrapWorld,
}

impl TrapRecord {
    /// Nombre estable del archivo/tarjeta.
    pub fn name(&self) -> String {
        format!("{}_n{}_s{}", self.principle.name(), self.scale, self.seed)
    }

    /// Construye el registro desde un mundo que falla (reduce y archiva).
    pub fn from_failing(
        principle: Principle,
        horizon: usize,
        original: &TrapWorld,
    ) -> Option<Self> {
        let red = reduce(original, principle, horizon)?;
        let ev = evaluate(&red.world, principle, horizon)?;
        let hash = chosen_hash(&red.world, ev.chosen, horizon)?;
        Some(Self {
            principle,
            scale: original.num_states(),
            horizon,
            seed: original.seed,
            orig_states: original.num_states(),
            orig_edges: original.num_edges(),
            chosen: ev.chosen,
            alternative: ev.best,
            failing_hash: hash,
            world: red.world,
        })
    }

    /// Reverifica fallo, hash y 1-minimalidad.
    pub fn verify(&self) -> Result<(), String> {
        let ev = evaluate(&self.world, self.principle, self.horizon).ok_or("sin evaluación")?;
        if !ev.fails() {
            return Err("el principio ya no falla".into());
        }
        if ev.chosen != self.chosen || ev.best != self.alternative {
            return Err("acciones elegida/alternativa difieren".into());
        }
        if chosen_hash(&self.world, ev.chosen, self.horizon) != Some(self.failing_hash) {
            return Err("failing_hash difiere".into());
        }
        if !is_one_minimal(&self.world, self.principle, self.horizon) {
            return Err("no es 1-minimal".into());
        }
        Ok(())
    }

    /// Serialización textual estable.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "{TRAP_ZOO_FORMAT}");
        let _ = writeln!(s, "principle={}", self.principle.name());
        let _ = writeln!(s, "scale={}", self.scale);
        let _ = writeln!(s, "horizon={}", self.horizon);
        let _ = writeln!(s, "seed={}", self.seed);
        let _ = writeln!(s, "initial={}", self.world.initial);
        let _ = writeln!(s, "orig_states={}", self.orig_states);
        let _ = writeln!(s, "orig_edges={}", self.orig_edges);
        let _ = writeln!(s, "chosen={}", self.chosen);
        let _ = writeln!(s, "alternative={}", self.alternative);
        let _ = writeln!(s, "failing_hash={:#018x}", self.failing_hash);
        for (id, st) in &self.world.states {
            let _ = writeln!(
                s,
                "state {id} {}",
                if st.viable { "viable" } else { "dead" }
            );
            for (a, edges) in st.actions.iter().enumerate() {
                for (d, w) in edges {
                    let _ = writeln!(s, "edge {id} {a} {d} {w}");
                }
            }
        }
        s
    }

    /// Parsea el formato de [`TrapRecord::to_text`].
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut lines = text.lines();
        if lines.next() != Some(TRAP_ZOO_FORMAT) {
            return Err("cabecera desconocida".into());
        }
        let mut kv: BTreeMap<&str, &str> = BTreeMap::new();
        let mut states: BTreeMap<u8, TrapState> = BTreeMap::new();
        for line in lines {
            let f: Vec<&str> = line.split_whitespace().collect();
            match f.as_slice() {
                [] => {}
                ["state", id, kind] => {
                    let id: u8 = id.parse().map_err(|_| "id de estado")?;
                    states.insert(
                        id,
                        TrapState {
                            viable: *kind == "viable",
                            actions: Vec::new(),
                        },
                    );
                }
                ["edge", id, a, d, w] => {
                    let p = |x: &str| x.parse::<usize>().map_err(|_| "número de arista");
                    let (id, a, d, w) = (p(id)? as u8, p(a)?, p(d)? as u8, p(w)? as u32);
                    let st = states.get_mut(&id).ok_or("arista de estado no declarado")?;
                    while st.actions.len() <= a {
                        st.actions.push(Vec::new());
                    }
                    st.actions[a].push((d, w));
                }
                [one] => {
                    let (k, v) = one.split_once('=').ok_or("línea inválida")?;
                    kv.insert(k, v);
                }
                _ => return Err(format!("línea inválida: {line}")),
            }
        }
        let get = |k: &str| kv.get(k).copied().ok_or(format!("falta {k}"));
        let num = |k: &str| get(k)?.parse::<u64>().map_err(|_| format!("{k} inválido"));
        let hash = u64::from_str_radix(get("failing_hash")?.trim_start_matches("0x"), 16)
            .map_err(|_| "failing_hash inválido")?;
        let seed = num("seed")?;
        Ok(Self {
            principle: Principle::from_name(get("principle")?).ok_or("principio desconocido")?,
            scale: num("scale")? as usize,
            horizon: num("horizon")? as usize,
            seed,
            orig_states: num("orig_states")? as usize,
            orig_edges: num("orig_edges")? as usize,
            chosen: num("chosen")? as usize,
            alternative: num("alternative")? as usize,
            failing_hash: hash,
            world: TrapWorld {
                seed,
                initial: num("initial")? as u8,
                states,
            },
        })
    }
}

fn chosen_hash(w: &TrapWorld, chosen: usize, horizon: usize) -> Option<u64> {
    w.to_bool(Some(chosen))
        .ok()?
        .oracle(
            MicroState(w.initial),
            &BoolSamplingPolicy::UniformActions,
            horizon,
            OracleLimits::default(),
        )
        .ok()
        .map(|r| r.stable_hash)
}

/// Construye el zoo: por principio falible y escala, los primeros `per_scale` seeds que fallan.
pub fn build_zoo(scales: &[u8], per_scale: usize, max_seed: u64) -> Vec<TrapRecord> {
    let mut zoo: Vec<TrapRecord> = Vec::new();
    for (principle, horizon) in [
        (Principle::GreedyFinalEntropy, 2),
        (Principle::TrajectoryEntropy, 3),
    ] {
        for &n in scales {
            let mut found = 0;
            for seed in 0..max_seed {
                if found >= per_scale {
                    break;
                }
                let w = TrapWorld::generate(seed, n);
                let Some(rec) = TrapRecord::from_failing(principle, horizon, &w) else {
                    continue;
                };
                if zoo
                    .iter()
                    .any(|z| z.principle == principle && z.world.states == rec.world.states)
                {
                    continue;
                }
                zoo.push(rec);
                found += 1;
            }
        }
    }
    zoo
}

/// Escribe el zoo (`<name>.trap`) en `dir`.
pub fn write_zoo(dir: &Path, zoo: &[TrapRecord]) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    for r in zoo {
        std::fs::write(dir.join(format!("{}.trap", r.name())), r.to_text())?;
    }
    Ok(())
}

/// Carga todos los `*.trap` de `dir`, ordenados por nombre.
pub fn load_zoo(dir: &Path) -> Result<Vec<TrapRecord>, String> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "trap"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            TrapRecord::from_text(&t).map_err(|e| format!("{}: {e}", p.display()))
        })
        .collect()
}

/// Fila de puntuación.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreRow {
    /// Nombre de la trampa.
    pub trap: String,
    /// `true` si el principio evaluado falla en esa trampa.
    pub fails: bool,
    /// Acción elegida (si hay evaluación).
    pub chosen: Option<usize>,
    /// Masa viable de la elegida.
    pub mass_chosen: f64,
    /// Mejor masa viable.
    pub mass_best: f64,
}

/// Puntúa `principle` contra el zoo.
pub fn score_principle(principle: Principle, zoo: &[TrapRecord]) -> Vec<ScoreRow> {
    zoo.iter()
        .map(|r| match evaluate(&r.world, principle, r.horizon) {
            Some(e) => ScoreRow {
                trap: r.name(),
                fails: e.fails(),
                chosen: Some(e.chosen),
                mass_chosen: e.masses[e.chosen],
                mass_best: e.masses[e.best],
            },
            None => ScoreRow {
                trap: r.name(),
                fails: false,
                chosen: None,
                mass_chosen: 0.0,
                mass_best: 0.0,
            },
        })
        .collect()
}

/// Veredicto sobre el zoo: no es una afirmación general del principio.
pub fn verdict(rows: &[ScoreRow]) -> &'static str {
    if rows.iter().any(|r| r.fails) {
        "FAILS_ON_ZOO"
    } else {
        "NULL_ON_ZOO"
    }
}

/// Tabla de texto.
pub fn render_table(principle: Principle, rows: &[ScoreRow]) -> String {
    let mut s = format!("principle={}\n", principle.name());
    let _ = writeln!(
        s,
        "{:<36} {:<5} {:>6} {:>9} {:>9}",
        "trap", "fails", "chosen", "m_chosen", "m_best"
    );
    for r in rows {
        let c = r.chosen.map_or("-".to_string(), |c| c.to_string());
        let _ = writeln!(
            s,
            "{:<36} {:<5} {:>6} {:>9.4} {:>9.4}",
            r.trap, r.fails, c, r.mass_chosen, r.mass_best
        );
    }
    let k = rows.iter().filter(|r| r.fails).count();
    let _ = writeln!(s, "failed={k}/{} verdict={}", rows.len(), verdict(rows));
    s
}

/// Tarjeta SVG escrita a mano.
pub fn render_svg(r: &TrapRecord) -> String {
    let ids: Vec<u8> = r.world.states.keys().copied().collect();
    let (cx, cy, rad) = (180.0_f64, 170.0_f64, 110.0_f64);
    let pos = |id: u8| {
        let i = ids.iter().position(|x| *x == id).unwrap_or(0) as f64;
        let a = std::f64::consts::TAU * i / ids.len().max(1) as f64 - std::f64::consts::FRAC_PI_2;
        (cx + rad * a.cos(), cy + rad * a.sin())
    };
    let mut s = String::from(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"360\" height=\"340\" font-family=\"monospace\" font-size=\"11\">\n<rect width=\"360\" height=\"340\" fill=\"#fff\" stroke=\"#333\"/>\n<defs><marker id=\"ar\" markerWidth=\"6\" markerHeight=\"6\" refX=\"5\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L6,3 L0,6 z\" fill=\"#555\"/></marker></defs>\n",
    );
    let _ = writeln!(
        s,
        "<text x=\"8\" y=\"16\">{} (H={})</text>",
        r.name(),
        r.horizon
    );
    let _ = writeln!(
        s,
        "<text x=\"8\" y=\"30\">principle={} scale={} seed={}</text>",
        r.principle.name(),
        r.scale,
        r.seed
    );
    let _ = writeln!(
        s,
        "<text x=\"8\" y=\"324\">states {}-&gt;{} edges {}-&gt;{} chosen=a{} (red) better=a{} (green)</text>",
        r.orig_states,
        r.world.num_states(),
        r.orig_edges,
        r.world.num_edges(),
        r.chosen,
        r.alternative
    );
    for (id, st) in &r.world.states {
        let (x1, y1) = pos(*id);
        for (a, edges) in st.actions.iter().enumerate() {
            let colour = if *id != r.world.initial {
                "#999"
            } else if a == r.chosen {
                "#d22"
            } else if a == r.alternative {
                "#2a2"
            } else {
                "#999"
            };
            let sum: u32 = edges.iter().map(|(_, w)| *w).sum();
            for (d, w) in edges {
                if d == id {
                    let _ = writeln!(
                        s,
                        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"9\" fill=\"none\" stroke=\"{colour}\"/>",
                        x1,
                        y1 - 24.0
                    );
                    continue;
                }
                let (x2, y2) = pos(*d);
                let (dx, dy) = (x2 - x1, y2 - y1);
                let len = dx.hypot(dy).max(1.0);
                let off = (a as f64 - 1.0) * 5.0;
                let (nx, ny) = (-dy / len * off, dx / len * off);
                let (sx, sy) = (x1 + dx / len * 16.0 + nx, y1 + dy / len * 16.0 + ny);
                let (ex, ey) = (x2 - dx / len * 18.0 + nx, y2 - dy / len * 18.0 + ny);
                let _ = writeln!(
                    s,
                    "<line x1=\"{sx:.1}\" y1=\"{sy:.1}\" x2=\"{ex:.1}\" y2=\"{ey:.1}\" stroke=\"{colour}\" marker-end=\"url(#ar)\"/>"
                );
                let _ = writeln!(
                    s,
                    "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{colour}\">{w}/{sum}</text>",
                    (sx + ex) / 2.0,
                    (sy + ey) / 2.0
                );
            }
        }
    }
    for (id, st) in &r.world.states {
        let (x, y) = pos(*id);
        let fill = if st.viable { "#def" } else { "#bbb" };
        let _ = writeln!(
            s,
            "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"15\" fill=\"{fill}\" stroke=\"#333\"/>"
        );
        let mark = if st.viable { "" } else { "x" };
        let _ = writeln!(
            s,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\">{id}{mark}</text>",
            x,
            y + 4.0
        );
    }
    s.push_str("</svg>\n");
    s
}

/// Directorio de fixtures por defecto.
pub fn default_zoo_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/trap_zoo")
}

/// Puntúa y escribe `score.txt` + tarjetas SVG en `out_dir`; devuelve la tabla.
pub fn score_and_gallery(
    principle: Principle,
    zoo: &[TrapRecord],
    out_dir: &Path,
) -> std::io::Result<String> {
    std::fs::create_dir_all(out_dir)?;
    let table = render_table(principle, &score_principle(principle, zoo));
    std::fs::write(out_dir.join("score.txt"), &table)?;
    for r in zoo {
        std::fs::write(out_dir.join(format!("{}.svg", r.name())), render_svg(r))?;
    }
    Ok(table)
}

/// CLI sin UI: `score --principle <n> --out-dir <d> [--zoo-dir <d>]` o `build [--zoo-dir <d>]`.
pub fn run_cli(args: &[String]) -> Result<String, String> {
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .map(PathBuf::from)
    };
    let zoo_dir = flag("--zoo-dir").unwrap_or_else(default_zoo_dir);
    match args.first().map(String::as_str) {
        Some("build") => {
            let zoo = build_zoo(&[5, 6, 7], 2, 5000);
            write_zoo(&zoo_dir, &zoo).map_err(|e| e.to_string())?;
            Ok(format!(
                "wrote {} traps to {}\n",
                zoo.len(),
                zoo_dir.display()
            ))
        }
        Some("score") => {
            let name = flag("--principle").ok_or("falta --principle")?;
            let principle = Principle::from_name(&name.to_string_lossy())
                .ok_or_else(|| format!("principio desconocido: {}", name.display()))?;
            let out = flag("--out-dir").ok_or("falta --out-dir")?;
            let zoo = load_zoo(&zoo_dir)?;
            score_and_gallery(principle, &zoo, &out).map_err(|e| e.to_string())
        }
        _ => Err(
            "uso: trap_zoo build | score --principle <name> --out-dir <dir> [--zoo-dir <dir>]"
                .into(),
        ),
    }
}
