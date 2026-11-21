//! Vista de microestado tipada (DEM-007): resolución explícita y proyección reproducible
//! sobre [`State`] sin imponer bits al mundo.
//!
//! # Qué es (y qué no es)
//! - Un [`MicrostateView`] declara variables con **nombre, unidad y resolución**
//!   (continua mediante [`BinAxisSpec`], discreta con cardinalidad, o booleana) leídas de
//!   índices de un [`State`]. Registra [`BinningVersion`] y un hash FNV-1a estable
//!   ([`MicrostateView::spec_hash`]) sobre una codificación canónica (sin reloj ni azar).
//! - Cada celda proyectada es un [`Cell`] **tri-estado**: `Known(bin)`, `Unknown`
//!   (propiedad no observada) o `Impossible` (bin declarado físicamente prohibido). `Unknown`
//!   nunca se confunde con `False`: en variables booleanas `False` es `Known(0)`.
//! - La vista **no almacena estados** ni crea una base de datos paralela; es una función
//!   pura sobre un `State` + máscara de observabilidad.
//!
//! # Frontera de ground truth
//! Las proyecciones deben construirse solo con el `State`/[`DiscreteBeliefState`] que el
//! agente puede observar. Las variables no observadas (máscara `false`) se proyectan como
//! `Unknown` y **no** se lee su valor; el planificador nunca recibe verdad de terreno por esta vía.
//!
//! # Alcance
//! Es una discretización de bookkeeping. No afirma equivalencia con entropía física ni
//! ningún fenómeno multiescala: cualquier interpretación requiere evidencia adicional.
//!
//! # Contrato `Scale` (GOG-200)
//! [`crate::Scale`] reutiliza esta vista como única implementación de proyección y le añade
//! identificador, versión y hash estables compartidos con el aprendiz tabular y el planificador.
//! La proyección parcial (`Unknown` vs `False` vs `Impossible`) se mantiene intacta.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use crate::belief::DiscreteBeliefState;
use crate::determinism::TraceHasher;
use crate::domain::State;
use crate::entropy::{BinAxisSpec, BinSpec, BinningVersion, EntropyError};

/// Máximo de variables por vista (límite fail-closed).
pub const MAX_MICROSTATE_VARIABLES: usize = 1024;
/// Máxima longitud en bytes de nombres y unidades.
pub const MAX_MICROSTATE_NAME_BYTES: usize = 128;
/// Máxima cardinalidad de una variable discreta.
pub const MAX_DISCRETE_CARDINALITY: usize = 1 << 20;
/// Tolerancia para considerar colapsada una creencia sobre una categoría.
pub const BELIEF_COLLAPSE_TOLERANCE: f64 = 1e-9;

/// Errores fail-closed de la vista de microestado.
#[derive(Debug, Clone, PartialEq)]
pub enum MicrostateError {
    /// La vista no declara variables.
    EmptyView,
    /// Demasiadas variables.
    TooManyVariables {
        /// Cantidad recibida.
        count: usize,
    },
    /// Nombre o unidad vacíos o demasiado largos.
    InvalidName {
        /// Campo afectado.
        field: &'static str,
    },
    /// Nombre de variable duplicado.
    DuplicateVariable {
        /// Nombre repetido.
        name: String,
    },
    /// Cardinalidad discreta fuera de `[2, MAX_DISCRETE_CARDINALITY]`.
    InvalidCardinality {
        /// Valor recibido.
        cardinality: usize,
    },
    /// Bin prohibido fuera del rango de bins de la variable.
    ForbiddenBinOutOfRange {
        /// Bin indicado.
        bin: usize,
    },
    /// El estado no tiene la dimensión que exige la vista.
    DimensionMismatch {
        /// Dimensión mínima esperada.
        expected: usize,
        /// Dimensión recibida (estado o máscara).
        actual: usize,
    },
    /// Valor observado no finito.
    NonFiniteValue {
        /// Variable afectada.
        variable: String,
    },
    /// Valor observado fuera del dominio declarado (se rechaza, no se satura).
    OutOfDomain {
        /// Variable afectada.
        variable: String,
        /// Valor recibido.
        value: f64,
    },
    /// Valor discreto no entero o fuera de cardinalidad.
    InvalidDiscreteValue {
        /// Variable afectada.
        variable: String,
        /// Valor recibido.
        value: f64,
    },
    /// Índice de variable inexistente.
    UnknownVariableIndex {
        /// Índice recibido.
        index: usize,
    },
    /// La creencia no tiene tantas categorías como la variable.
    BeliefCardinalityMismatch {
        /// Esperadas.
        expected: usize,
        /// Recibidas.
        actual: usize,
    },
    /// Error del módulo de bins reutilizado.
    Entropy(EntropyError),
}

impl fmt::Display for MicrostateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyView => write!(f, "microstate view has no variables"),
            Self::TooManyVariables { count } => {
                write!(
                    f,
                    "too many variables: {count} > {MAX_MICROSTATE_VARIABLES}"
                )
            }
            Self::InvalidName { field } => write!(f, "invalid or empty/oversized {field}"),
            Self::DuplicateVariable { name } => write!(f, "duplicate variable '{name}'"),
            Self::InvalidCardinality { cardinality } => {
                write!(f, "invalid discrete cardinality {cardinality}")
            }
            Self::ForbiddenBinOutOfRange { bin } => {
                write!(f, "forbidden bin {bin} outside variable bins")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteValue { variable } => {
                write!(f, "non-finite value for variable '{variable}'")
            }
            Self::OutOfDomain { variable, value } => {
                write!(f, "value {value} outside domain of variable '{variable}'")
            }
            Self::InvalidDiscreteValue { variable, value } => {
                write!(
                    f,
                    "invalid discrete value {value} for variable '{variable}'"
                )
            }
            Self::UnknownVariableIndex { index } => write!(f, "unknown variable index {index}"),
            Self::BeliefCardinalityMismatch { expected, actual } => {
                write!(f, "belief has {actual} categories, expected {expected}")
            }
            Self::Entropy(e) => write!(f, "bin spec error: {e}"),
        }
    }
}

impl Error for MicrostateError {}

impl From<EntropyError> for MicrostateError {
    fn from(value: EntropyError) -> Self {
        Self::Entropy(value)
    }
}

/// Tipo y resolución de una variable.
#[derive(Debug, Clone, PartialEq)]
pub enum VariableKind {
    /// Variable continua discretizada por un eje de bins (dominio = rango del eje).
    Continuous(BinAxisSpec),
    /// Variable discreta con `cardinality` categorías (valor entero en `[0, cardinality)`).
    Discrete {
        /// Número de categorías (`>= 2`).
        cardinality: usize,
    },
    /// Variable booleana: `Known(0)` = falso, `Known(1)` = verdadero.
    Boolean,
}

impl VariableKind {
    fn num_bins(&self) -> usize {
        match self {
            Self::Continuous(axis) => axis.num_bins(),
            Self::Discrete { cardinality } => *cardinality,
            Self::Boolean => 2,
        }
    }
}

/// Declaración de una variable de la vista.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrostateVariable {
    name: String,
    unit: String,
    source_index: usize,
    kind: VariableKind,
    forbidden_bins: BTreeSet<usize>,
}

fn check_text(text: &str, field: &'static str) -> Result<(), MicrostateError> {
    if text.trim().is_empty() || text.len() > MAX_MICROSTATE_NAME_BYTES {
        return Err(MicrostateError::InvalidName { field });
    }
    Ok(())
}

impl MicrostateVariable {
    /// Declara una variable leída de `State::values()[source_index]`.
    ///
    /// `unit` es obligatoria (use `"1"` para adimensional).
    pub fn new(
        name: impl Into<String>,
        unit: impl Into<String>,
        source_index: usize,
        kind: VariableKind,
    ) -> Result<Self, MicrostateError> {
        let name = name.into();
        let unit = unit.into();
        check_text(&name, "name")?;
        check_text(&unit, "unit")?;
        if let VariableKind::Discrete { cardinality } = &kind
            && !(2..=MAX_DISCRETE_CARDINALITY).contains(cardinality)
        {
            return Err(MicrostateError::InvalidCardinality {
                cardinality: *cardinality,
            });
        }
        Ok(Self {
            name,
            unit,
            source_index,
            kind,
            forbidden_bins: BTreeSet::new(),
        })
    }

    /// Declara bins físicamente imposibles (se proyectan como [`Cell::Impossible`]).
    pub fn with_forbidden_bins(mut self, bins: &[usize]) -> Result<Self, MicrostateError> {
        let n = self.kind.num_bins();
        for &bin in bins {
            if bin >= n {
                return Err(MicrostateError::ForbiddenBinOutOfRange { bin });
            }
            self.forbidden_bins.insert(bin);
        }
        Ok(self)
    }

    /// Nombre de la variable.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Unidad declarada.
    pub fn unit(&self) -> &str {
        &self.unit
    }

    /// Índice fuente en `State::values()`.
    pub const fn source_index(&self) -> usize {
        self.source_index
    }

    /// Tipo y resolución.
    pub const fn kind(&self) -> &VariableKind {
        &self.kind
    }

    fn domain_check(&self, value: f64) -> Result<(), MicrostateError> {
        if let VariableKind::Continuous(axis) = &self.kind {
            let (lo, hi) = match axis {
                BinAxisSpec::Uniform { min, max, .. } => (*min, *max),
                BinAxisSpec::ExplicitEdges { edges } => (edges[0], edges[edges.len() - 1]),
            };
            if value < lo || value > hi {
                return Err(MicrostateError::OutOfDomain {
                    variable: self.name.clone(),
                    value,
                });
            }
        }
        Ok(())
    }

    pub(crate) fn cell_of(&self, value: f64) -> Result<Cell, MicrostateError> {
        if !value.is_finite() {
            return Err(MicrostateError::NonFiniteValue {
                variable: self.name.clone(),
            });
        }
        let bin = match &self.kind {
            VariableKind::Continuous(axis) => {
                self.domain_check(value)?;
                axis.bin_of(value)?
            }
            VariableKind::Discrete { .. } | VariableKind::Boolean => {
                let n = self.kind.num_bins();
                if value.fract() != 0.0 || value < 0.0 || value >= n as f64 {
                    return Err(MicrostateError::InvalidDiscreteValue {
                        variable: self.name.clone(),
                        value,
                    });
                }
                value as usize
            }
        };
        Ok(if self.forbidden_bins.contains(&bin) {
            Cell::Impossible
        } else {
            Cell::Known(bin)
        })
    }
}

/// Celda tri-estado de una variable proyectada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cell {
    /// Valor observado, resuelto al bin/categoría indicado.
    Known(usize),
    /// Propiedad no observada: ausencia de información, **no** un valor falso.
    Unknown,
    /// El valor cae en un bin declarado físicamente imposible.
    Impossible,
}

/// Lectura lógica de una variable booleana.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Truth {
    /// Observada verdadera.
    True,
    /// Observada falsa (distinto de `Unknown`).
    False,
    /// No observada.
    Unknown,
    /// Estado declarado imposible.
    Impossible,
}

/// Microestado proyectado: celdas más identidad (hash/versión) de la vista que lo produjo.
#[derive(Debug, Clone, PartialEq)]
pub struct Microstate {
    view_hash: u64,
    version: BinningVersion,
    cells: Vec<Cell>,
}

impl Microstate {
    /// Celdas en el orden de declaración de variables.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Hash de la vista que produjo esta proyección.
    pub const fn view_hash(&self) -> u64 {
        self.view_hash
    }

    /// Versión de binning de la vista.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Lectura booleana de la celda `index` (devuelve `None` si el índice no existe).
    ///
    /// Solo es significativa para variables [`VariableKind::Boolean`].
    pub fn truth(&self, index: usize) -> Option<Truth> {
        Some(match self.cells.get(index)? {
            Cell::Known(0) => Truth::False,
            Cell::Known(_) => Truth::True,
            Cell::Unknown => Truth::Unknown,
            Cell::Impossible => Truth::Impossible,
        })
    }

    /// Número de celdas `Unknown`.
    pub fn unknown_count(&self) -> usize {
        self.cells.iter().filter(|c| **c == Cell::Unknown).count()
    }

    /// Hash FNV-1a estable de la proyección (vista + celdas).
    pub fn hash(&self) -> u64 {
        let mut h = TraceHasher::new();
        h.write_u64(self.view_hash);
        h.write_u64(self.cells.len() as u64);
        for cell in &self.cells {
            match cell {
                Cell::Known(b) => {
                    h.write_u8(1);
                    h.write_u64(*b as u64);
                }
                Cell::Unknown => h.write_u8(2),
                Cell::Impossible => h.write_u8(3),
            }
        }
        h.finish()
    }
}

/// Vista de microestado: variables con unidades y resolución explícita sobre [`State`].
#[derive(Debug, Clone, PartialEq)]
pub struct MicrostateView {
    label: String,
    version: BinningVersion,
    variables: Vec<MicrostateVariable>,
    spec_hash: u64,
}

impl MicrostateView {
    /// Construye la vista validando límites y unicidad, y calcula su hash canónico.
    pub fn new(
        label: impl Into<String>,
        version: BinningVersion,
        variables: Vec<MicrostateVariable>,
    ) -> Result<Self, MicrostateError> {
        let label = label.into();
        check_text(&label, "label")?;
        if variables.is_empty() {
            return Err(MicrostateError::EmptyView);
        }
        if variables.len() > MAX_MICROSTATE_VARIABLES {
            return Err(MicrostateError::TooManyVariables {
                count: variables.len(),
            });
        }
        let mut seen = BTreeSet::new();
        for v in &variables {
            if !seen.insert(v.name.as_str()) {
                return Err(MicrostateError::DuplicateVariable {
                    name: v.name.clone(),
                });
            }
        }
        let spec_hash = canonical_hash(&label, version, &variables);
        Ok(Self {
            label,
            version,
            variables,
            spec_hash,
        })
    }

    /// Etiqueta de la vista.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Versión de binning registrada.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Variables declaradas.
    pub fn variables(&self) -> &[MicrostateVariable] {
        &self.variables
    }

    /// Hash FNV-1a estable de la especificación (etiqueta, versión, variables, unidades, bins).
    pub const fn spec_hash(&self) -> u64 {
        self.spec_hash
    }

    /// Embedding numérico de **todas** las variables como ejes de [`BinSpec`] (orden declarado).
    ///
    /// Las variables continuas conservan su eje; las discretas se convierten en un eje uniforme
    /// `[0, cardinality)` con `cardinality` bins y las booleanas en `[0, 2)` con 2 bins, de modo
    /// que `bin_coords(values)` coincide con el bin observado de la proyección tri-estado para
    /// cualquier valor entero válido. Es el puente que usa [`crate::Scale`] para compartir la
    /// misma discretización con aprendices tabulares y estimadores de entropía.
    ///
    /// Nota: es una incrustación numérica, no una proyección semántica; `Impossible`/`Unknown`
    /// solo existen en [`Self::project`].
    pub fn numeric_bin_spec(&self) -> BinSpec {
        let axes = self
            .variables
            .iter()
            .map(|v| match &v.kind {
                VariableKind::Continuous(axis) => axis.clone(),
                VariableKind::Discrete { cardinality } => {
                    BinAxisSpec::uniform(0.0, *cardinality as f64, *cardinality)
                        .expect("discrete cardinality >= 2 is validated at construction")
                }
                VariableKind::Boolean => BinAxisSpec::uniform(0.0, 2.0, 2)
                    .expect("the boolean axis [0, 2) with 2 bins is always valid"),
            })
            .collect();
        BinSpec::from_axes(self.label.clone(), self.version, axes)
            .expect("a view has a non-empty label and at least one variable")
    }

    /// Reutiliza [`BinSpec`] con los ejes continuos de la vista (para estimadores de entropía).
    ///
    /// Devuelve error si la vista no tiene variables continuas.
    pub fn continuous_bin_spec(&self) -> Result<BinSpec, MicrostateError> {
        let axes = self
            .variables
            .iter()
            .filter_map(|v| match &v.kind {
                VariableKind::Continuous(a) => Some(a.clone()),
                _ => None,
            })
            .collect();
        Ok(BinSpec::from_axes(self.label.clone(), self.version, axes)?)
    }

    /// Proyecta un [`State`] observable. `observed[i]` indica si el índice `i` de
    /// `State::values()` es observable; los no observables se proyectan `Unknown` sin leerse.
    pub fn project(&self, state: &State, observed: &[bool]) -> Result<Microstate, MicrostateError> {
        let needed = self
            .variables
            .iter()
            .map(|v| v.source_index + 1)
            .max()
            .unwrap_or(0);
        if state.dimension() < needed {
            return Err(MicrostateError::DimensionMismatch {
                expected: needed,
                actual: state.dimension(),
            });
        }
        if observed.len() != state.dimension() {
            return Err(MicrostateError::DimensionMismatch {
                expected: state.dimension(),
                actual: observed.len(),
            });
        }
        let mut cells = Vec::with_capacity(self.variables.len());
        for v in &self.variables {
            if observed[v.source_index] {
                cells.push(v.cell_of(state.values()[v.source_index])?);
            } else {
                cells.push(Cell::Unknown);
            }
        }
        Ok(Microstate {
            view_hash: self.spec_hash,
            version: self.version,
            cells,
        })
    }

    /// Resuelve una celda `Unknown` de la variable `index` con una creencia del agente.
    ///
    /// Si la creencia colapsa (`p >= 1 - BELIEF_COLLAPSE_TOLERANCE`) en una categoría, la celda
    /// pasa a `Known` (o `Impossible` si esa categoría está prohibida); si no, permanece
    /// `Unknown`. Las celdas ya conocidas no se sobrescriben.
    pub fn resolve_with_belief(
        &self,
        microstate: &Microstate,
        index: usize,
        belief: &DiscreteBeliefState,
    ) -> Result<Microstate, MicrostateError> {
        let var = self
            .variables
            .get(index)
            .ok_or(MicrostateError::UnknownVariableIndex { index })?;
        if microstate.view_hash != self.spec_hash || microstate.cells.len() != self.variables.len()
        {
            return Err(MicrostateError::DimensionMismatch {
                expected: self.variables.len(),
                actual: microstate.cells.len(),
            });
        }
        let n = var.kind.num_bins();
        if belief.num_states() != n {
            return Err(MicrostateError::BeliefCardinalityMismatch {
                expected: n,
                actual: belief.num_states(),
            });
        }
        let mut out = microstate.clone();
        if out.cells[index] == Cell::Unknown {
            let bin = belief.map_state();
            if belief.probabilities()[bin] >= 1.0 - BELIEF_COLLAPSE_TOLERANCE {
                out.cells[index] = if var.forbidden_bins.contains(&bin) {
                    Cell::Impossible
                } else {
                    Cell::Known(bin)
                };
            }
        }
        Ok(out)
    }
}

fn canonical_hash(label: &str, version: BinningVersion, vars: &[MicrostateVariable]) -> u64 {
    let mut h = TraceHasher::new();
    h.write_str("gogamen.microstate.view.v1");
    h.write_str(label);
    h.write_u64(u64::from(version.major()));
    h.write_u64(u64::from(version.minor()));
    h.write_u64(vars.len() as u64);
    for v in vars {
        h.write_str(&v.name);
        h.write_str(&v.unit);
        h.write_u64(v.source_index as u64);
        match &v.kind {
            VariableKind::Continuous(BinAxisSpec::Uniform { min, max, num_bins }) => {
                h.write_u8(1);
                h.write_f64(*min);
                h.write_f64(*max);
                h.write_u64(*num_bins as u64);
            }
            VariableKind::Continuous(BinAxisSpec::ExplicitEdges { edges }) => {
                h.write_u8(2);
                h.write_u64(edges.len() as u64);
                for e in edges {
                    h.write_f64(*e);
                }
            }
            VariableKind::Discrete { cardinality } => {
                h.write_u8(3);
                h.write_u64(*cardinality as u64);
            }
            VariableKind::Boolean => h.write_u8(4),
        }
        h.write_u64(v.forbidden_bins.len() as u64);
        for b in &v.forbidden_bins {
            h.write_u64(*b as u64);
        }
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{SimDuration, Tick};

    fn st(values: Vec<f64>) -> State {
        State::new(
            Tick::new(0),
            SimDuration::from_secs_f64(0.0).unwrap(),
            values,
        )
        .unwrap()
    }

    fn view(bins: usize, minor: u32) -> MicrostateView {
        MicrostateView::new(
            "v",
            BinningVersion::new(1, minor),
            vec![
                MicrostateVariable::new(
                    "x",
                    "m",
                    0,
                    VariableKind::Continuous(BinAxisSpec::uniform(0.0, 10.0, bins).unwrap()),
                )
                .unwrap(),
                MicrostateVariable::new("door_open", "1", 1, VariableKind::Boolean).unwrap(),
                MicrostateVariable::new("mode", "1", 2, VariableKind::Discrete { cardinality: 3 })
                    .unwrap()
                    .with_forbidden_bins(&[2])
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn gog_160_projection_and_hash_are_reproducible() {
        let s = st(vec![2.5, 1.0, 0.0]);
        let all = [true; 3];
        let a = view(5, 0).project(&s, &all).unwrap();
        let b = view(5, 0).project(&s, &all).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.hash(), b.hash());
        assert_eq!(view(5, 0).spec_hash(), view(5, 0).spec_hash());
        assert_eq!(a.cells()[0], Cell::Known(1));
        assert_eq!(a.truth(1), Some(Truth::True));
    }

    #[test]
    fn gog_160_different_resolution_or_version_changes_hash() {
        let s = st(vec![2.5, 1.0, 0.0]);
        let all = [true; 3];
        assert_ne!(view(5, 0).spec_hash(), view(10, 0).spec_hash());
        assert_ne!(view(5, 0).spec_hash(), view(5, 1).spec_hash());
        let p5 = view(5, 0).project(&s, &all).unwrap();
        let p10 = view(10, 0).project(&s, &all).unwrap();
        assert_ne!(p5.hash(), p10.hash());
        assert_eq!(view(5, 1).version(), BinningVersion::new(1, 1));
        assert_eq!(p5.version(), BinningVersion::new(1, 0));
    }

    #[test]
    fn gog_160_unknown_is_not_false_and_partial_observation() {
        let v = view(5, 0);
        // El valor del índice 1 es 0.0 (falso) pero no observado: debe ser Unknown.
        let s = st(vec![2.5, 0.0, 1.0]);
        let partial = v.project(&s, &[true, false, true]).unwrap();
        assert_eq!(partial.truth(1), Some(Truth::Unknown));
        assert_eq!(partial.unknown_count(), 1);
        let full = v.project(&s, &[true; 3]).unwrap();
        assert_eq!(full.truth(1), Some(Truth::False));
        assert_ne!(partial.hash(), full.hash());
        assert_ne!(Truth::Unknown, Truth::False);
    }

    #[test]
    fn gog_160_unobserved_value_is_not_read() {
        // Un valor no observado inválido no debe filtrarse ni provocar error: no se lee.
        let v = view(5, 0);
        let s = st(vec![2.5, 99.0, 1.0]);
        let p = v.project(&s, &[true, false, true]).unwrap();
        assert_eq!(p.cells()[1], Cell::Unknown);
        assert!(v.project(&s, &[true; 3]).is_err());
    }

    #[test]
    fn gog_160_impossible_differs_from_false_and_unknown() {
        let v = view(5, 0);
        let s = st(vec![2.5, 1.0, 2.0]);
        let p = v.project(&s, &[true; 3]).unwrap();
        assert_eq!(p.cells()[2], Cell::Impossible);
        assert_ne!(p.cells()[2], Cell::Unknown);
        assert_ne!(p.cells()[2], Cell::Known(0));
    }

    #[test]
    fn gog_160_rejects_out_of_range_and_bad_input() {
        let v = view(5, 0);
        let all = [true; 3];
        assert!(matches!(
            v.project(&st(vec![11.0, 1.0, 0.0]), &all),
            Err(MicrostateError::OutOfDomain { .. })
        ));
        assert!(matches!(
            v.project(&st(vec![-0.1, 1.0, 0.0]), &all),
            Err(MicrostateError::OutOfDomain { .. })
        ));
        assert!(matches!(
            v.project(&st(vec![1.0, 0.5, 0.0]), &all),
            Err(MicrostateError::InvalidDiscreteValue { .. })
        ));
        assert!(matches!(
            v.project(&st(vec![1.0, 1.0, 3.0]), &all),
            Err(MicrostateError::InvalidDiscreteValue { .. })
        ));
        assert!(
            State::new(
                Tick::new(0),
                SimDuration::from_secs_f64(0.0).unwrap(),
                vec![f64::NAN]
            )
            .is_err()
        );
        assert!(matches!(
            v.project(&st(vec![1.0, 1.0]), &[true, true]),
            Err(MicrostateError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            v.project(&st(vec![1.0, 1.0, 0.0]), &[true]),
            Err(MicrostateError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn gog_160_construction_is_bounded_and_validated() {
        let mk = |n: &str, u: &str| MicrostateVariable::new(n, u, 0, VariableKind::Boolean);
        assert!(mk("", "1").is_err());
        assert!(mk("a", " ").is_err());
        assert!(
            MicrostateVariable::new("a", "1", 0, VariableKind::Discrete { cardinality: 1 })
                .is_err()
        );
        assert!(mk("a", "1").unwrap().with_forbidden_bins(&[2]).is_err());
        let v = BinningVersion::V1_0;
        assert_eq!(
            MicrostateView::new("l", v, vec![]),
            Err(MicrostateError::EmptyView)
        );
        assert!(matches!(
            MicrostateView::new("l", v, vec![mk("a", "1").unwrap(), mk("a", "1").unwrap()]),
            Err(MicrostateError::DuplicateVariable { .. })
        ));
        let many = (0..=MAX_MICROSTATE_VARIABLES)
            .map(|i| mk(&format!("v{i}"), "1").unwrap())
            .collect();
        assert!(matches!(
            MicrostateView::new("l", v, many),
            Err(MicrostateError::TooManyVariables { .. })
        ));
    }

    #[test]
    fn gog_160_belief_resolves_unknown_only_when_collapsed() {
        let v = view(5, 0);
        let s = st(vec![2.5, 0.0, 0.0]);
        let p = v.project(&s, &[true, false, false]).unwrap();
        let diffuse = DiscreteBeliefState::new(vec![0.5, 0.5]).unwrap();
        let q = v.resolve_with_belief(&p, 1, &diffuse).unwrap();
        assert_eq!(q.truth(1), Some(Truth::Unknown));
        let sure = DiscreteBeliefState::new(vec![0.0, 1.0]).unwrap();
        let q = v.resolve_with_belief(&p, 1, &sure).unwrap();
        assert_eq!(q.truth(1), Some(Truth::True));
        let forbidden = DiscreteBeliefState::new(vec![0.0, 0.0, 1.0]).unwrap();
        let q = v.resolve_with_belief(&p, 2, &forbidden).unwrap();
        assert_eq!(q.cells()[2], Cell::Impossible);
        assert!(matches!(
            v.resolve_with_belief(&p, 2, &sure),
            Err(MicrostateError::BeliefCardinalityMismatch { .. })
        ));
        assert!(matches!(
            v.resolve_with_belief(&p, 9, &sure),
            Err(MicrostateError::UnknownVariableIndex { .. })
        ));
    }

    #[test]
    fn gog_160_reuses_bin_spec() {
        let spec = view(5, 0).continuous_bin_spec().unwrap();
        assert_eq!(spec.dimension(), 1);
        assert_eq!(spec.version(), BinningVersion::new(1, 0));
        assert_eq!(spec.bin_coords(&[2.5]).unwrap(), vec![1]);
    }
}
