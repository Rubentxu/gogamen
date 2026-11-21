//! Perfilado opt-in de la representación de microestados (GOG-179).
//!
//! Compara, sobre fixtures deterministas, la proyección baseline
//! ([`MicrostateView::project`], `Vec<Cell>`) con una codificación empaquetada `u32`
//! ([`PackedMicrostate`], opt-in vía [`project_packed`] / [`project_with`]). El camino
//! por defecto de producción sigue siendo `MicrostateView::project`; activar la
//! variante empaquetada solo tiene sentido cuando el crossover medido lo justifica.
//!
//! # Codificación empaquetada
//! Cada celda ocupa un `u32`: [`PACKED_UNKNOWN`] (`u32::MAX`) = `Unknown`,
//! [`PACKED_IMPOSSIBLE`] (`u32::MAX - 1`) = `Impossible`, y `bin as u32` = `Known(bin)`.
//! El diseño asume bins por debajo de `2^20` ([`crate::MAX_DISCRETE_CARDINALITY`]) y
//! reserva los dos valores altos para los estados lógicos; un bin que no quepa se
//! rechaza con error en vez de truncarse. [`PackedMicrostate::hash`] reproduce byte a
//! byte la secuencia FNV-1a de [`Microstate::hash`] (mismo `view_hash`, número de
//! celdas y tags `1/2/3` con el bin como `u64`), por lo que ambas codificaciones
//! comparan iguales; hashes idénticos implican la misma proyección lógica y, por
//! tanto, la misma calidad predictiva aguas abajo bajo fixtures fijos.
//!
//! # Coste de memoria (fórmula afirmada)
//! Solo layout de tipos, no una medición del asignador:
//! `size_of::<Microstate>() + n * size_of::<Cell>()` frente a
//! `size_of::<PackedMicrostate>() + n * size_of::<u32>()`.
//!
//! # Métricas
//! `transitions_per_sec` cuenta proyecciones `State -> Microstate` por segundo, es
//! decir, una por tick simulado; el tiempo medido cubre solo la proyección (los
//! [`State`] de la fixture se construyen antes de medir y se comparten entre ambas
//! codificaciones).

use std::error::Error;
use std::fmt::{self, Write as _};
use std::mem::size_of;

use crate::clock::MonotonicClock;
use crate::determinism::TraceHasher;
use crate::domain::State;
use crate::entropy::{BinAxisSpec, BinningVersion, EntropyError};
use crate::microstate::{
    Cell, MAX_MICROSTATE_VARIABLES, Microstate, MicrostateError, MicrostateVariable,
    MicrostateView, VariableKind,
};
use crate::units::{SimDuration, Tick};

/// Palabra empaquetada de una celda [`Cell::Unknown`].
pub const PACKED_UNKNOWN: u32 = u32::MAX;
/// Palabra empaquetada de una celda [`Cell::Impossible`].
pub const PACKED_IMPOSSIBLE: u32 = u32::MAX - 1;
/// Mayor bin representable en la codificación empaquetada.
pub const MAX_PACKED_BIN: u32 = PACKED_IMPOSSIBLE - 1;

/// Tamaños de vista usados por [`MicrostateProfileConfig::default`].
pub const DEFAULT_PROFILE_VARIABLE_COUNTS: [usize; 4] = [4, 16, 64, 256];
/// Ticks medidos por tamaño en [`MicrostateProfileConfig::default`].
pub const DEFAULT_PROFILE_TICKS: usize = 2_000;

/// Aviso: la variante empaquetada es opt-in y no altera DSL ni modelo lógico.
const PACKED_OPT_IN_NOTE: &str = "la codificación empaquetada es opt-in (MicrostateEncoding::Packed) y no es el camino por defecto; no cambia el DSL ni el modelo lógico y conserva los límites de validación; el profiler existente de gogamen-arena (BottleneckStage) no expone una etapa de microestado, así que no hay un cuello de botella medido que justifique activarla";

/// Codificación de celdas proyectadas. [`MicrostateEncoding::Cells`] es el baseline y
/// el único camino por defecto; [`MicrostateEncoding::Packed`] es opt-in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MicrostateEncoding {
    /// Baseline: `Vec<Cell>` producido por [`MicrostateView::project`].
    Cells,
    /// Opt-in: celdas empaquetadas en `u32`.
    Packed,
}

impl MicrostateEncoding {
    /// Etiqueta estable usada en los informes.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cells => "cells",
            Self::Packed => "packed",
        }
    }
}

/// Microestado empaquetado: identidad de la vista y una palabra `u32` por celda.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedMicrostate {
    view_hash: u64,
    version: BinningVersion,
    packed: Vec<u32>,
}

impl PackedMicrostate {
    /// Hash de la vista que produjo esta proyección.
    pub const fn view_hash(&self) -> u64 {
        self.view_hash
    }

    /// Versión de binning de la vista.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Decodifica las celdas en el orden de declaración de variables.
    pub fn decode(&self) -> Vec<Cell> {
        self.packed.iter().map(|&word| decode_cell(word)).collect()
    }

    /// Número de celdas `Unknown`.
    pub fn unknown_count(&self) -> usize {
        self.packed
            .iter()
            .filter(|&&word| word == PACKED_UNKNOWN)
            .count()
    }

    /// Hash FNV-1a idéntico a [`Microstate::hash`] para las mismas celdas.
    pub fn hash(&self) -> u64 {
        let mut h = TraceHasher::new();
        h.write_u64(self.view_hash);
        h.write_u64(self.packed.len() as u64);
        for &word in &self.packed {
            write_packed_cell(&mut h, word);
        }
        h.finish()
    }
}

/// Proyección devuelta por [`project_with`]: baseline ([`Microstate`]) o empaquetada.
#[derive(Debug, Clone, PartialEq)]
pub enum MicrostateProjection {
    /// Proyección baseline `Vec<Cell>`.
    Cells(Microstate),
    /// Proyección empaquetada opt-in.
    Packed(PackedMicrostate),
}

impl MicrostateProjection {
    /// Codificación de la que procede esta proyección.
    pub const fn encoding(&self) -> MicrostateEncoding {
        match self {
            Self::Cells(_) => MicrostateEncoding::Cells,
            Self::Packed(_) => MicrostateEncoding::Packed,
        }
    }

    /// Hash de la proyección (equivalente entre codificaciones).
    pub fn hash(&self) -> u64 {
        match self {
            Self::Cells(microstate) => microstate.hash(),
            Self::Packed(packed) => packed.hash(),
        }
    }

    /// Número de celdas `Unknown`.
    pub fn unknown_count(&self) -> usize {
        match self {
            Self::Cells(microstate) => microstate.unknown_count(),
            Self::Packed(packed) => packed.unknown_count(),
        }
    }
}

const fn encode_cell(cell: Cell) -> Result<u32, MicrostateError> {
    match cell {
        Cell::Unknown => Ok(PACKED_UNKNOWN),
        Cell::Impossible => Ok(PACKED_IMPOSSIBLE),
        Cell::Known(bin) => {
            if bin as u128 > MAX_PACKED_BIN as u128 {
                return Err(MicrostateError::Entropy(EntropyError::InvalidBinSpec {
                    reason: "packed microstate requires bins below 2^20",
                }));
            }
            Ok(bin as u32)
        }
    }
}

const fn decode_cell(word: u32) -> Cell {
    match word {
        PACKED_UNKNOWN => Cell::Unknown,
        PACKED_IMPOSSIBLE => Cell::Impossible,
        bin => Cell::Known(bin as usize),
    }
}

fn write_packed_cell(hasher: &mut TraceHasher, word: u32) {
    match word {
        PACKED_UNKNOWN => hasher.write_u8(2),
        PACKED_IMPOSSIBLE => hasher.write_u8(3),
        bin => {
            hasher.write_u8(1);
            hasher.write_u64(u64::from(bin));
        }
    }
}

/// Valida dimensión y máscara de observabilidad con el mismo orden y errores que
/// [`MicrostateView::project`].
fn validate_projection(
    view: &MicrostateView,
    state: &State,
    observed: &[bool],
) -> Result<(), MicrostateError> {
    let needed = view
        .variables()
        .iter()
        .map(|v| v.source_index() + 1)
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
    Ok(())
}

/// Proyecta un [`State`] observable a la representación empaquetada.
///
/// Aplica la misma validación que [`MicrostateView::project`] (dimensión, longitud de
/// la máscara, valores no finitos/fuera de dominio y discretos inválidos) y empaqueta
/// directamente en `Vec<u32>` sin construir `Vec<Cell>`.
pub fn project_packed(
    view: &MicrostateView,
    state: &State,
    observed: &[bool],
) -> Result<PackedMicrostate, MicrostateError> {
    validate_projection(view, state, observed)?;
    let mut packed = Vec::with_capacity(view.variables().len());
    for variable in view.variables() {
        let index = variable.source_index();
        let word = if observed[index] {
            encode_cell(variable.cell_of(state.values()[index])?)?
        } else {
            PACKED_UNKNOWN
        };
        packed.push(word);
    }
    Ok(PackedMicrostate {
        view_hash: view.spec_hash(),
        version: view.version(),
        packed,
    })
}

/// Proyecta con la codificación indicada; [`MicrostateEncoding::Cells`] delega en
/// [`MicrostateView::project`], que sigue siendo el camino por defecto.
pub fn project_with(
    encoding: MicrostateEncoding,
    view: &MicrostateView,
    state: &State,
    observed: &[bool],
) -> Result<MicrostateProjection, MicrostateError> {
    Ok(match encoding {
        MicrostateEncoding::Cells => MicrostateProjection::Cells(view.project(state, observed)?),
        MicrostateEncoding::Packed => {
            MicrostateProjection::Packed(project_packed(view, state, observed)?)
        }
    })
}

/// Error de configuración del perfilado de microestados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MicrostateProfileError {
    /// No se declaró ningún tamaño de vista.
    EmptyVariableCounts,
    /// Tamaño de vista fuera de `1..=MAX_MICROSTATE_VARIABLES`.
    VariableCountOutOfRange {
        /// Tamaño recibido.
        count: usize,
    },
    /// `ticks_per_size` debe ser al menos `1`.
    ZeroTicksPerSize,
}

impl fmt::Display for MicrostateProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVariableCounts => write!(f, "profile requires at least one variable count"),
            Self::VariableCountOutOfRange { count } => {
                write!(
                    f,
                    "variable count {count} outside 1..={MAX_MICROSTATE_VARIABLES}"
                )
            }
            Self::ZeroTicksPerSize => write!(f, "ticks_per_size must be at least 1"),
        }
    }
}

impl Error for MicrostateProfileError {}

/// Configuración del perfilado: tamaños de vista y ticks medidos por tamaño.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicrostateProfileConfig {
    /// Número de variables de cada fixture.
    pub variable_counts: Vec<usize>,
    /// Proyecciones medidas por tamaño y codificación.
    pub ticks_per_size: usize,
}

impl MicrostateProfileConfig {
    /// Valida tamaños no vacíos, dentro de `1..=MAX_MICROSTATE_VARIABLES`, y ticks `>= 1`.
    pub fn new(
        variable_counts: Vec<usize>,
        ticks_per_size: usize,
    ) -> Result<Self, MicrostateProfileError> {
        if variable_counts.is_empty() {
            return Err(MicrostateProfileError::EmptyVariableCounts);
        }
        for &count in &variable_counts {
            if !(1..=MAX_MICROSTATE_VARIABLES).contains(&count) {
                return Err(MicrostateProfileError::VariableCountOutOfRange { count });
            }
        }
        if ticks_per_size == 0 {
            return Err(MicrostateProfileError::ZeroTicksPerSize);
        }
        Ok(Self {
            variable_counts,
            ticks_per_size,
        })
    }
}

impl Default for MicrostateProfileConfig {
    fn default() -> Self {
        Self {
            variable_counts: DEFAULT_PROFILE_VARIABLE_COUNTS.to_vec(),
            ticks_per_size: DEFAULT_PROFILE_TICKS,
        }
    }
}

/// Medición de una codificación sobre una fixture.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrostateProfileReport {
    /// Codificación medida.
    pub encoding: MicrostateEncoding,
    /// Variables de la vista.
    pub variables: usize,
    /// Proyecciones medidas (una por tick simulado).
    pub ticks: usize,
    /// Bytes por microestado según la fórmula de layout documentada.
    pub memory_bytes_per_microstate: usize,
    /// Proyecciones `State -> Microstate` por segundo.
    pub transitions_per_sec: f64,
    /// Tiempo medio por proyección (tick) en nanosegundos.
    pub nanos_per_tick: f64,
    /// Digest determinista del lote (FNV-1a sobre los hashes de cada proyección).
    pub oracle_hash: u64,
    /// Celdas `Unknown` acumuladas en el lote.
    pub unknown_count: usize,
}

/// Comparación baseline/empaquetado con el crossover medido.
#[derive(Debug, Clone, PartialEq)]
pub struct MicrostateCrossoverReport {
    /// Medición baseline (`Vec<Cell>`).
    pub baseline: Vec<MicrostateProfileReport>,
    /// Medición empaquetada (`u32`).
    pub packed: Vec<MicrostateProfileReport>,
    /// Hashes de oráculo y conteos de `Unknown` coinciden en todos los tamaños.
    pub equivalent: bool,
    /// Menor tamaño donde el empaquetado gana en ns/tick y en memoria (si existe).
    pub crossover_variables: Option<usize>,
    /// Siempre `false`: la variante empaquetada no es el camino por defecto.
    pub packed_enabled_by_default: bool,
    /// Conclusión honesta sobre el crossover y el estado opt-in.
    pub note: String,
}

impl MicrostateCrossoverReport {
    /// Resumen textual con una línea por tamaño/codificación y la conclusión.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        for (baseline, packed) in self.baseline.iter().zip(&self.packed) {
            for report in [baseline, packed] {
                let _ = writeln!(
                    out,
                    "microstate-profile encoding={} variables={} ticks={} memory_bytes={} transitions_per_sec={:.0} ns_per_tick={:.3} unknowns={} oracle_hash=0x{:016x}",
                    report.encoding.as_str(),
                    report.variables,
                    report.ticks,
                    report.memory_bytes_per_microstate,
                    report.transitions_per_sec,
                    report.nanos_per_tick,
                    report.unknown_count,
                    report.oracle_hash,
                );
            }
        }
        let crossover = match self.crossover_variables {
            Some(variables) => format!("crossover_variables={variables}"),
            None => "crossover_variables=none".to_string(),
        };
        let _ = writeln!(
            out,
            "microstate-profile equivalent={} {} packed_enabled_by_default={}",
            self.equivalent, crossover, self.packed_enabled_by_default
        );
        let _ = writeln!(
            out,
            "microstate-profile memory_formula=baseline:size_of::<Microstate>()+n*size_of::<Cell>() packed:size_of::<PackedMicrostate>()+n*size_of::<u32>()"
        );
        let _ = writeln!(out, "microstate-profile note={}", self.note);
        out
    }
}

const FIXTURE_FORBIDDEN_VARIABLE: usize = 0;
const FIXTURE_FORBIDDEN_BIN: usize = 3;
const FIXTURE_CONTINUOUS_BINS: usize = 16;
const FIXTURE_DISCRETE_CARDINALITY: usize = 8;
const FIXTURE_CONTINUOUS_STEPS: usize = 1_600;

struct ProfileFixture {
    view: MicrostateView,
    states: Vec<State>,
    observed: Vec<bool>,
}

fn fixture_value(kind: usize, variable: usize, tick: usize) -> f64 {
    match kind {
        0 => {
            ((tick * 7 + variable * 13) % FIXTURE_CONTINUOUS_STEPS) as f64
                / FIXTURE_CONTINUOUS_STEPS as f64
        }
        1 => ((tick * 5 + variable * 7) % FIXTURE_DISCRETE_CARDINALITY) as f64,
        _ => ((tick + variable) % 2) as f64,
    }
}

fn build_fixture(variables: usize, ticks: usize) -> Result<ProfileFixture, MicrostateError> {
    let mut declared = Vec::with_capacity(variables);
    for index in 0..variables {
        let kind = match index % 3 {
            0 => VariableKind::Continuous(BinAxisSpec::uniform(0.0, 1.0, FIXTURE_CONTINUOUS_BINS)?),
            1 => VariableKind::Discrete {
                cardinality: FIXTURE_DISCRETE_CARDINALITY,
            },
            _ => VariableKind::Boolean,
        };
        let mut variable = MicrostateVariable::new(format!("v{index}"), "1", index, kind)?;
        if index == FIXTURE_FORBIDDEN_VARIABLE {
            variable = variable.with_forbidden_bins(&[FIXTURE_FORBIDDEN_BIN])?;
        }
        declared.push(variable);
    }
    let view = MicrostateView::new(
        format!("gog-179-profile-{variables}"),
        BinningVersion::V1_0,
        declared,
    )?;
    let observed = (0..variables).map(|index| index % 4 != 3).collect();
    let mut states = Vec::with_capacity(ticks);
    for tick in 0..ticks {
        let values = (0..variables)
            .map(|index| fixture_value(index % 3, index, tick))
            .collect();
        states.push(
            State::new(Tick::new(tick as u64), SimDuration::ZERO, values)
                .expect("fixture values are finite"),
        );
    }
    Ok(ProfileFixture {
        view,
        states,
        observed,
    })
}

fn memory_bytes(encoding: MicrostateEncoding, variables: usize) -> usize {
    match encoding {
        MicrostateEncoding::Cells => size_of::<Microstate>() + variables * size_of::<Cell>(),
        MicrostateEncoding::Packed => size_of::<PackedMicrostate>() + variables * size_of::<u32>(),
    }
}

struct Measurement {
    nanos: u128,
    oracle_hash: u64,
    unknown_count: usize,
}

fn measure<C: MonotonicClock>(
    encoding: MicrostateEncoding,
    fixture: &ProfileFixture,
    clock: &C,
) -> Measurement {
    let mut digest = TraceHasher::new();
    let mut unknown_count = 0usize;
    let start = clock.now();
    for state in &fixture.states {
        let projection = project_with(encoding, &fixture.view, state, &fixture.observed)
            .expect("validated fixtures project without error");
        digest.write_u64(projection.hash());
        unknown_count += projection.unknown_count();
    }
    let end = clock.now();
    Measurement {
        nanos: end.as_nanos().saturating_sub(start.as_nanos()),
        oracle_hash: digest.finish(),
        unknown_count,
    }
}

fn build_report(
    encoding: MicrostateEncoding,
    variables: usize,
    ticks: usize,
    measurement: &Measurement,
) -> MicrostateProfileReport {
    let nanos_per_tick = measurement.nanos.max(1) as f64 / ticks.max(1) as f64;
    MicrostateProfileReport {
        encoding,
        variables,
        ticks,
        memory_bytes_per_microstate: memory_bytes(encoding, variables),
        transitions_per_sec: 1_000_000_000.0 / nanos_per_tick,
        nanos_per_tick,
        oracle_hash: measurement.oracle_hash,
        unknown_count: measurement.unknown_count,
    }
}

/// Ejecuta el perfilado determinista completo y devuelve la comparación con crossover.
///
/// Ambas codificaciones proyectan exactamente los mismos [`State`] de fixture y se
/// contrastan por hash de oráculo y conteo de `Unknown`. El camino por defecto no se
/// altera: el informe siempre declara `packed_enabled_by_default = false`.
pub fn run_microstate_profile<C: MonotonicClock>(
    config: &MicrostateProfileConfig,
    clock: &C,
) -> MicrostateCrossoverReport {
    let mut baseline = Vec::with_capacity(config.variable_counts.len());
    let mut packed = Vec::with_capacity(config.variable_counts.len());
    let mut equivalent = true;
    for &variables in &config.variable_counts {
        let fixture = build_fixture(variables, config.ticks_per_size)
            .expect("validated variable counts build fixtures");
        let baseline_measurement = measure(MicrostateEncoding::Cells, &fixture, clock);
        let packed_measurement = measure(MicrostateEncoding::Packed, &fixture, clock);
        equivalent &= baseline_measurement.oracle_hash == packed_measurement.oracle_hash
            && baseline_measurement.unknown_count == packed_measurement.unknown_count;
        baseline.push(build_report(
            MicrostateEncoding::Cells,
            variables,
            config.ticks_per_size,
            &baseline_measurement,
        ));
        packed.push(build_report(
            MicrostateEncoding::Packed,
            variables,
            config.ticks_per_size,
            &packed_measurement,
        ));
    }
    let crossover_variables = baseline
        .iter()
        .zip(&packed)
        .find(|(base, pack)| {
            pack.nanos_per_tick < base.nanos_per_tick
                && pack.memory_bytes_per_microstate < base.memory_bytes_per_microstate
        })
        .map(|(_, pack)| pack.variables);
    let conclusion = match crossover_variables {
        Some(variables) => format!(
            "crossover medido en {variables} variables: el empaquetado es estrictamente más rápido por tick y usa menos memoria"
        ),
        None => "sin crossover medido: el empaquetado no superó al baseline en ns/tick en ninguna fixture, así que no se activa por defecto".to_string(),
    };
    MicrostateCrossoverReport {
        baseline,
        packed,
        equivalent,
        crossover_variables,
        packed_enabled_by_default: false,
        note: format!("{PACKED_OPT_IN_NOTE}; {conclusion}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SystemMonotonicClock;

    fn st(values: Vec<f64>) -> State {
        State::new(Tick::new(0), SimDuration::ZERO, values).expect("finite state")
    }

    #[test]
    fn gog_179_packed_matches_baseline_cells_hashes_and_unknowns() {
        for variables in [4, 16, 64] {
            let fixture = build_fixture(variables, 64).expect("fixture builds");
            let (mut unknowns, mut impossibles) = (0, 0);
            for state in &fixture.states {
                let baseline = fixture
                    .view
                    .project(state, &fixture.observed)
                    .expect("baseline projection");
                let packed =
                    project_packed(&fixture.view, state, &fixture.observed).expect("packed");
                assert_eq!(packed.decode(), baseline.cells().to_vec());
                assert_eq!(packed.hash(), baseline.hash());
                assert_eq!(packed.unknown_count(), baseline.unknown_count());
                assert_eq!(packed.view_hash(), baseline.view_hash());
                assert_eq!(packed.version(), baseline.version());
                unknowns += baseline.unknown_count();
                impossibles += baseline
                    .cells()
                    .iter()
                    .filter(|cell| **cell == Cell::Impossible)
                    .count();
            }
            assert!(unknowns > 0, "fixture must produce Unknown cells");
            assert!(impossibles > 0, "fixture must produce Impossible cells");
        }
    }

    #[test]
    fn gog_179_packed_preserves_validation_limits_and_errors() {
        let fixture = build_fixture(4, 1).expect("fixture builds");
        let view = &fixture.view;
        let observed = &fixture.observed;

        let short = st(vec![0.1, 1.0, 1.0]);
        let cells_err = view.project(&short, &[true; 3]).unwrap_err();
        let packed_err = project_packed(view, &short, &[true; 3]).unwrap_err();
        assert_eq!(cells_err, packed_err);
        assert!(matches!(
            cells_err,
            MicrostateError::DimensionMismatch {
                expected: 4,
                actual: 3
            }
        ));

        let full = st(vec![0.1, 1.0, 1.0, 0.0]);
        let cells_err = view.project(&full, &[true; 3]).unwrap_err();
        let packed_err = project_packed(view, &full, &[true; 3]).unwrap_err();
        assert_eq!(cells_err, packed_err);
        assert!(matches!(
            cells_err,
            MicrostateError::DimensionMismatch {
                expected: 4,
                actual: 3
            }
        ));

        let out_of_domain = st(vec![1.5, 1.0, 1.0, 0.0]);
        let cells_err = view.project(&out_of_domain, observed).unwrap_err();
        let packed_err = project_packed(view, &out_of_domain, observed).unwrap_err();
        assert_eq!(cells_err, packed_err);
        assert!(matches!(cells_err, MicrostateError::OutOfDomain { .. }));

        let bad_discrete = st(vec![0.1, 8.0, 1.0, 0.0]);
        let cells_err = view.project(&bad_discrete, observed).unwrap_err();
        let packed_err = project_packed(view, &bad_discrete, observed).unwrap_err();
        assert_eq!(cells_err, packed_err);
        assert!(matches!(
            cells_err,
            MicrostateError::InvalidDiscreteValue { .. }
        ));

        let fractional = st(vec![0.1, 1.5, 1.0, 0.0]);
        assert_eq!(
            view.project(&fractional, observed).unwrap_err(),
            project_packed(view, &fractional, observed).unwrap_err()
        );

        // Un no finito no puede entrar en un `State`: el límite se aplica antes y por igual.
        assert!(
            State::new(Tick::new(0), SimDuration::ZERO, vec![f64::NAN]).is_err(),
            "non-finite state must be rejected upstream"
        );

        // Un bin que no cabe en la codificación se rechaza; nunca se trunca.
        let wide = MicrostateView::new(
            "wide",
            BinningVersion::V1_0,
            vec![
                MicrostateVariable::new(
                    "x",
                    "1",
                    0,
                    VariableKind::Continuous(
                        BinAxisSpec::uniform(0.0, 1.0, u32::MAX as usize).expect("wide axis"),
                    ),
                )
                .expect("wide variable"),
            ],
        )
        .expect("wide view");
        let edge = st(vec![1.0]);
        assert!(wide.project(&edge, &[true]).is_ok());
        assert!(matches!(
            project_packed(&wide, &edge, &[true]),
            Err(MicrostateError::Entropy(
                EntropyError::InvalidBinSpec { .. }
            ))
        ));
    }

    #[test]
    fn gog_179_default_project_path_unchanged_and_usable() {
        let fixture = build_fixture(4, 8).expect("fixture builds");
        let state = &fixture.states[1];
        let default_path = fixture
            .view
            .project(state, &fixture.observed)
            .expect("default projection");
        assert_eq!(default_path.cells().len(), 4);
        assert_eq!(default_path.view_hash(), fixture.view.spec_hash());
        match project_with(
            MicrostateEncoding::Cells,
            &fixture.view,
            state,
            &fixture.observed,
        )
        .expect("cells projection")
        {
            MicrostateProjection::Cells(microstate) => assert_eq!(microstate, default_path),
            MicrostateProjection::Packed(_) => panic!("cells encoding must not pack"),
        }
        let packed = project_with(
            MicrostateEncoding::Packed,
            &fixture.view,
            state,
            &fixture.observed,
        )
        .expect("packed projection");
        assert_eq!(packed.encoding(), MicrostateEncoding::Packed);
        assert_eq!(packed.hash(), default_path.hash());
    }

    #[test]
    fn gog_179_profile_reports_equivalent_hashes_and_memory() {
        let config = MicrostateProfileConfig::new(vec![4, 16], 200).expect("valid config");
        let clock = SystemMonotonicClock::new();
        let report = run_microstate_profile(&config, &clock);

        assert_eq!(report.baseline.len(), 2);
        assert_eq!(report.packed.len(), 2);
        assert!(report.equivalent);
        assert!(!report.packed_enabled_by_default);
        for (base, pack) in report.baseline.iter().zip(&report.packed) {
            assert_eq!(base.encoding, MicrostateEncoding::Cells);
            assert_eq!(pack.encoding, MicrostateEncoding::Packed);
            assert_eq!(base.variables, pack.variables);
            assert_eq!(base.ticks, 200);
            assert_eq!(base.oracle_hash, pack.oracle_hash);
            assert_eq!(base.unknown_count, pack.unknown_count);
            assert!(pack.memory_bytes_per_microstate < base.memory_bytes_per_microstate);
            for value in [
                base.transitions_per_sec,
                base.nanos_per_tick,
                pack.transitions_per_sec,
                pack.nanos_per_tick,
            ] {
                assert!(value.is_finite() && value >= 0.0);
            }
            assert!(base.transitions_per_sec > 0.0);
            assert!(pack.transitions_per_sec > 0.0);
            assert!(base.memory_bytes_per_microstate > 0);
        }

        let summary = report.format_summary();
        for token in [
            "memory_bytes=",
            "transitions_per_sec=",
            "ns_per_tick=",
            "crossover",
            "packed_enabled_by_default=false",
            "equivalent=true",
            "opt-in",
            "variables=4",
            "variables=16",
        ] {
            assert!(summary.contains(token), "summary must mention {token}");
        }
        println!("{summary}");
    }

    #[test]
    fn gog_179_config_validates_bounds() {
        assert_eq!(
            MicrostateProfileConfig::new(vec![], 10),
            Err(MicrostateProfileError::EmptyVariableCounts)
        );
        assert_eq!(
            MicrostateProfileConfig::new(vec![0], 10),
            Err(MicrostateProfileError::VariableCountOutOfRange { count: 0 })
        );
        assert_eq!(
            MicrostateProfileConfig::new(vec![MAX_MICROSTATE_VARIABLES + 1], 10),
            Err(MicrostateProfileError::VariableCountOutOfRange {
                count: MAX_MICROSTATE_VARIABLES + 1
            })
        );
        assert_eq!(
            MicrostateProfileConfig::new(vec![4], 0),
            Err(MicrostateProfileError::ZeroTicksPerSize)
        );
        let default = MicrostateProfileConfig::default();
        assert_eq!(
            default.variable_counts,
            DEFAULT_PROFILE_VARIABLE_COUNTS.to_vec()
        );
        assert_eq!(default.ticks_per_size, DEFAULT_PROFILE_TICKS);
    }
}
