//! Kit de invariantes para métricas de entropía futura de TERCEROS (GOG-191, DEM-002 / GOG-155).
//!
//! Publica, de forma independiente de `labs/*`, los controles anti-artefacto de DEM-002: una
//! métrica externa `f(distribución sobre microestados etiquetados, seed) -> bits` se somete a
//! invariantes y, si alguno falla, el informe incluye el contraejemplo MÍNIMO (reducido por
//! shrinking determinista). No toca `gogamen-core`; solo `std`.
//!
//! # Invariantes (ids estables, parte del hash del kit)
//! - `relabel_invariance`: renombrar inyectivamente los microestados no cambia el valor.
//! - `bounds`: `0 <= f <= log2(n)` y finito.
//! - `zero_on_deterministic`: un mundo determinista (un único estado) vale 0.
//! - `no_gain_from_duplication`: copiar un predicado determinista a un bit nuevo no sube el valor.
//! - `mixing_monotonicity`: mezclar con el uniforme sobre el mismo soporte no baja el valor.
//! - `seed_determinism`: misma entrada y seed dan exactamente el mismo valor.
//!
//! # Protocolo de línea (métricas fuera de proceso, p. ej. Python)
//! Ver [`ProcessMetric`]. El kit escribe una línea por evaluación en el `stdin` del proceso:
//! `EVAL <seed> <label>:<prob>,<label>:<prob>,...` y espera `OK <valor>` o `ERR <mensaje>`
//! en una línea de `stdout`. Las probabilidades se escriben con la representación round-trip
//! de `f64`.
//!
//! # Límites
//! Solo distribuciones discretas pequeñas (≤ 6 estados); no demuestra corrección de la métrica,
//! solo falsa con casos finitos. El contraejemplo es MÍNIMO LOCAL (ningún paso de shrinking
//! individual -quitar un estado, igualar probabilidades, bajar una etiqueta- lo reduce más); no
//! es el mínimo global garantizado (p. ej. `mixing_monotonicity` con una métrica dependiente de
//! etiquetas puede quedarse en 3 estados). No afirma emergencia ni agencia.

use std::fmt::Write as _;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

/// Versión del kit.
pub const METRIC_KIT_VERSION: &str = "0.1.0";
/// Versión del protocolo de línea.
pub const METRIC_LINE_PROTOCOL: &str = "GOGAMEN-METRIC/1";
/// Tolerancia numérica (bits).
pub const METRIC_KIT_TOLERANCE: f64 = 1e-9;
/// Casos aleatorios por invariante por defecto.
pub const DEFAULT_CASES: usize = 200;
/// Máximo de estados generados.
pub const MAX_STATES: usize = 6;

/// Ids estables de invariantes, en orden de ejecución.
pub const INVARIANT_IDS: [&str; 6] = [
    "relabel_invariance",
    "bounds",
    "zero_on_deterministic",
    "no_gain_from_duplication",
    "mixing_monotonicity",
    "seed_determinism",
];

/// Hash FNV-1a (64 bits) determinista del contenido del kit: versión, protocolo, tolerancia,
/// tope de estados e ids de invariantes.
pub fn metric_kit_hash() -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut feed = |bytes: &[u8]| {
        for b in bytes {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h ^= 0xff;
        h = h.wrapping_mul(0x100000001b3);
    };
    feed(METRIC_KIT_VERSION.as_bytes());
    feed(METRIC_LINE_PROTOCOL.as_bytes());
    feed(&METRIC_KIT_TOLERANCE.to_bits().to_le_bytes());
    feed(&(MAX_STATES as u64).to_le_bytes());
    for id in INVARIANT_IDS {
        feed(id.as_bytes());
    }
    h
}

/// Distribución sobre microestados etiquetados (etiquetas distintas, probabilidades suman 1).
#[derive(Debug, Clone, PartialEq)]
pub struct MetricInput {
    /// Pares `(etiqueta, probabilidad)`.
    pub states: Vec<(u64, f64)>,
}

impl MetricInput {
    fn render(&self) -> String {
        let mut s = String::new();
        for (i, (l, p)) in self.states.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            let _ = write!(s, "{l}:{p:?}");
        }
        s
    }
}

/// Métrica externa: devuelve bits o un error textual.
pub trait ExternalMetric {
    /// Evalúa la métrica.
    fn eval(&mut self, input: &MetricInput, seed: u64) -> Result<f64, String>;
}

impl<F: FnMut(&MetricInput, u64) -> Result<f64, String>> ExternalMetric for F {
    fn eval(&mut self, input: &MetricInput, seed: u64) -> Result<f64, String> {
        self(input, seed)
    }
}

/// Métrica en un proceso externo (cualquier lenguaje) mediante el protocolo de línea.
pub struct ProcessMetric {
    child: Child,
    stdin: ChildStdin,
    lines: Receiver<std::io::Result<String>>,
    timeout: Duration,
}

impl ProcessMetric {
    /// Lanza `program args...` con `stdin`/`stdout` entubados y un timeout por evaluación.
    pub fn spawn(program: &str, args: &[String], timeout: Duration) -> std::io::Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let (tx, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            lines,
            timeout,
        })
    }
}

impl ExternalMetric for ProcessMetric {
    fn eval(&mut self, input: &MetricInput, seed: u64) -> Result<f64, String> {
        writeln!(self.stdin, "EVAL {seed} {}", input.render())
            .and_then(|()| self.stdin.flush())
            .map_err(|e| format!("escritura al proceso: {e}"))?;
        let line = self
            .lines
            .recv_timeout(self.timeout)
            .map_err(|e| format!("sin respuesta del proceso ({e})"))?
            .map_err(|e| format!("lectura del proceso: {e}"))?;
        if let Some(v) = line.strip_prefix("OK ") {
            v.trim()
                .parse::<f64>()
                .map_err(|e| format!("valor no numérico {v:?}: {e}"))
        } else if let Some(m) = line.strip_prefix("ERR ") {
            Err(format!("el proceso informó error: {m}"))
        } else {
            Err(format!("respuesta fuera de protocolo: {line:?}"))
        }
    }
}

impl Drop for ProcessMetric {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Configuración del kit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KitConfig {
    /// Casos aleatorios por invariante (además de los casos esquina fijos).
    pub cases: usize,
    /// Seed del generador de casos (determinista).
    pub seed: u64,
}

impl Default for KitConfig {
    fn default() -> Self {
        Self {
            cases: DEFAULT_CASES,
            seed: 191,
        }
    }
}

/// Contraejemplo mínimo.
#[derive(Debug, Clone, PartialEq)]
pub struct Counterexample {
    /// Entrada reducida.
    pub input: MetricInput,
    /// Explicación (valores observados).
    pub detail: String,
    /// Pasos de shrinking aceptados.
    pub shrink_steps: usize,
}

/// Resultado por invariante.
#[derive(Debug, Clone, PartialEq)]
pub struct InvariantResult {
    /// Id del invariante.
    pub id: &'static str,
    /// Casos evaluados hasta el primer fallo (o todos).
    pub cases_run: usize,
    /// Contraejemplo mínimo si falla.
    pub counterexample: Option<Counterexample>,
}

impl InvariantResult {
    /// `true` si pasó.
    pub fn passed(&self) -> bool {
        self.counterexample.is_none()
    }
}

/// Informe del kit.
#[derive(Debug, Clone, PartialEq)]
pub struct KitReport {
    /// Nombre de la métrica enviada.
    pub metric: String,
    /// Versión del kit.
    pub kit_version: &'static str,
    /// Hash FNV-1a del contenido del kit.
    pub kit_hash: u64,
    /// Configuración usada.
    pub config: KitConfig,
    /// Resultados por invariante.
    pub results: Vec<InvariantResult>,
}

impl KitReport {
    /// `true` si todos los invariantes pasan.
    pub fn passed(&self) -> bool {
        self.results.iter().all(InvariantResult::passed)
    }

    /// Resultado de un invariante por id.
    pub fn result(&self, id: &str) -> Option<&InvariantResult> {
        self.results.iter().find(|r| r.id == id)
    }

    /// Texto legible determinista.
    pub fn render(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(
            s,
            "gogamen metric-kit v{} hash={:016x} protocol={}",
            self.kit_version, self.kit_hash, METRIC_LINE_PROTOCOL
        );
        let _ = writeln!(
            s,
            "metric={} cases={} seed={} verdict={}",
            self.metric,
            self.config.cases,
            self.config.seed,
            if self.passed() { "PASS" } else { "FAIL" }
        );
        for r in &self.results {
            match &r.counterexample {
                None => {
                    let _ = writeln!(s, "  [PASS] {} ({} cases)", r.id, r.cases_run);
                }
                Some(c) => {
                    let _ = writeln!(
                        s,
                        "  [FAIL] {} after {} cases; minimal counterexample (shrink steps {}):\n         input: {}\n         {}",
                        r.id,
                        r.cases_run,
                        c.shrink_steps,
                        c.input.render(),
                        c.detail
                    );
                }
            }
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

fn from_weights(labels: &[u64], weights: &[u64]) -> MetricInput {
    let total: u64 = weights.iter().sum();
    MetricInput {
        states: labels
            .iter()
            .zip(weights)
            .map(|(l, w)| (*l, *w as f64 / total as f64))
            .collect(),
    }
}

fn uniform(labels: &[u64]) -> MetricInput {
    from_weights(labels, &vec![1; labels.len()])
}

fn is_uniform(i: &MetricInput) -> bool {
    i.states.windows(2).all(|w| w[0].1 == w[1].1)
}

fn size_key(i: &MetricInput) -> (usize, u64, bool) {
    (
        i.states.len(),
        i.states.iter().map(|s| s.0).sum(),
        !is_uniform(i),
    )
}

fn corner_cases() -> Vec<MetricInput> {
    vec![
        uniform(&[0]),
        uniform(&[0, 1]),
        uniform(&[0, 1, 2]),
        uniform(&[0, 1, 2, 3]),
        from_weights(&[0, 1], &[1, 3]),
    ]
}

fn random_case(rng: &mut u64) -> MetricInput {
    let n = 1 + (splitmix(rng) % MAX_STATES as u64) as usize;
    let mut labels: Vec<u64> = Vec::new();
    while labels.len() < n {
        let l = splitmix(rng) % 64;
        if !labels.contains(&l) {
            labels.push(l);
        }
    }
    let weights: Vec<u64> = (0..n).map(|_| 1 + splitmix(rng) % 4).collect();
    from_weights(&labels, &weights)
}

fn shrink_candidates(i: &MetricInput) -> Vec<MetricInput> {
    let n = i.states.len();
    let mut out = Vec::new();
    if n > 1 {
        for k in 0..n {
            let mut s = i.states.clone();
            s.remove(k);
            let total: f64 = s.iter().map(|x| x.1).sum();
            for x in &mut s {
                x.1 /= total;
            }
            out.push(MetricInput { states: s });
        }
    }
    let labels: Vec<u64> = i.states.iter().map(|s| s.0).collect();
    if !is_uniform(i) {
        out.push(uniform(&labels));
    }
    let canon: Vec<u64> = (0..n as u64).collect();
    if labels != canon {
        let mut c = i.clone();
        for (k, s) in c.states.iter_mut().enumerate() {
            s.0 = k as u64;
        }
        out.push(c);
    }
    // bajar la etiqueta de un estado a cualquier valor menor libre
    for k in 0..n {
        for new in 0..i.states[k].0 {
            if !labels.contains(&new) {
                let mut c = i.clone();
                c.states[k].0 = new;
                out.push(c);
            }
        }
    }
    out
}

type Check<'a> = dyn FnMut(&mut dyn ExternalMetric, &MetricInput, u64) -> Result<(), String> + 'a;

fn eval(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<f64, String> {
    let v = m.eval(i, seed)?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!("valor no finito {v:?}"))
    }
}

fn permuted(i: &MetricInput, f: impl Fn(u64) -> u64) -> MetricInput {
    MetricInput {
        states: i.states.iter().map(|(l, p)| (f(*l), *p)).collect(),
    }
}

fn check_relabel(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let base = eval(m, i, seed)?;
    let max = i.states.iter().map(|s| s.0).max().unwrap_or(0);
    type Relabel = (&'static str, Box<dyn Fn(u64) -> u64>);
    let maps: Vec<Relabel> = vec![
        ("l->3l+1", Box::new(|l| 3 * l + 1)),
        ("l->max-l", Box::new(move |l| max - l)),
        ("l->l+100", Box::new(|l| l + 100)),
    ];
    for (name, f) in maps {
        let t = permuted(i, f);
        let v = eval(m, &t, seed)?;
        if (v - base).abs() > METRIC_KIT_TOLERANCE {
            return Err(format!(
                "relabel {name} -> input {}: f={v:?} != f(original)={base:?}",
                t.render()
            ));
        }
    }
    Ok(())
}

fn check_bounds(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let v = eval(m, i, seed)?;
    let hi = (i.states.len() as f64).log2();
    if v < -METRIC_KIT_TOLERANCE || v > hi + METRIC_KIT_TOLERANCE {
        return Err(format!("f={v:?} fuera de [0, log2(n)={hi:?}]"));
    }
    Ok(())
}

fn check_zero(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let v = eval(m, i, seed)?;
    if v.abs() > METRIC_KIT_TOLERANCE {
        return Err(format!("mundo determinista: f={v:?} != 0"));
    }
    Ok(())
}

fn check_dup(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let base = eval(m, i, seed)?;
    // bit nuevo = copia determinista del predicado (l & 1); etiqueta' = (l << 1) | bit.
    let t = permuted(i, |l| (l << 1) | (l & 1));
    let v = eval(m, &t, seed)?;
    if v > base + METRIC_KIT_TOLERANCE {
        return Err(format!(
            "duplicación -> input {}: f={v:?} > f(original)={base:?}",
            t.render()
        ));
    }
    Ok(())
}

fn check_mixing(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let base = eval(m, i, seed)?;
    let n = i.states.len() as f64;
    let t = MetricInput {
        states: i
            .states
            .iter()
            .map(|(l, p)| (*l, 0.5 * p + 0.5 / n))
            .collect(),
    };
    let v = eval(m, &t, seed)?;
    if v < base - METRIC_KIT_TOLERANCE {
        return Err(format!(
            "mezcla con uniforme -> input {}: f={v:?} < f(original)={base:?}",
            t.render()
        ));
    }
    Ok(())
}

fn check_seed(m: &mut dyn ExternalMetric, i: &MetricInput, seed: u64) -> Result<(), String> {
    let a = eval(m, i, seed)?;
    let b = eval(m, i, seed)?;
    if a.to_bits() != b.to_bits() {
        return Err(format!("misma seed {seed}: {a:?} != {b:?}"));
    }
    Ok(())
}

fn run_invariant(
    id: &'static str,
    metric: &mut dyn ExternalMetric,
    cases: Vec<MetricInput>,
    seed: u64,
    check: &mut Check<'_>,
) -> InvariantResult {
    for (n, case) in cases.iter().enumerate() {
        if let Err(first) = check(metric, case, seed) {
            let mut cur = case.clone();
            let mut detail = first;
            let mut steps = 0;
            'outer: loop {
                for cand in shrink_candidates(&cur) {
                    if size_key(&cand) < size_key(&cur)
                        && let Err(d) = check(metric, &cand, seed)
                    {
                        cur = cand;
                        detail = d;
                        steps += 1;
                        continue 'outer;
                    }
                }
                break;
            }
            return InvariantResult {
                id,
                cases_run: n + 1,
                counterexample: Some(Counterexample {
                    input: cur,
                    detail,
                    shrink_steps: steps,
                }),
            };
        }
    }
    InvariantResult {
        id,
        cases_run: cases.len(),
        counterexample: None,
    }
}

/// Somete una métrica externa a todos los invariantes y devuelve el informe.
pub fn run_metric_kit(name: &str, metric: &mut dyn ExternalMetric, cfg: &KitConfig) -> KitReport {
    let mut rng = cfg.seed;
    let mut dists = corner_cases();
    for _ in 0..cfg.cases {
        dists.push(random_case(&mut rng));
    }
    let mut dets = vec![uniform(&[0])];
    for _ in 0..cfg.cases.min(20) {
        dets.push(uniform(&[splitmix(&mut rng) % 64]));
    }
    let s = cfg.seed;
    let results = vec![
        run_invariant(
            INVARIANT_IDS[0],
            metric,
            dists.clone(),
            s,
            &mut check_relabel,
        ),
        run_invariant(
            INVARIANT_IDS[1],
            metric,
            dists.clone(),
            s,
            &mut check_bounds,
        ),
        run_invariant(INVARIANT_IDS[2], metric, dets, s, &mut check_zero),
        run_invariant(INVARIANT_IDS[3], metric, dists.clone(), s, &mut check_dup),
        run_invariant(
            INVARIANT_IDS[4],
            metric,
            dists.clone(),
            s,
            &mut check_mixing,
        ),
        run_invariant(INVARIANT_IDS[5], metric, dists, s, &mut check_seed),
    ];
    KitReport {
        metric: name.to_string(),
        kit_version: METRIC_KIT_VERSION,
        kit_hash: metric_kit_hash(),
        config: *cfg,
        results,
    }
}
