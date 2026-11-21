//! Contrato tipado de escala (`Scale`, DEM-007 v2, GOG-200): una única identidad de
//! discretización compartida por aprendiz y planificador.
//!
//! # Qué es
//! Un [`Scale`] eleva `phi`/`BinSpec` (GOG-160) a un contrato con **identificador estable**,
//! **versión** de binning y **hash** FNV-1a canónico. Envuelve:
//! - una [`MicrostateView`] que declara variables, unidades, resolución y bins prohibidos, y
//!   que es la única implementación de proyección (`Known`/`Unknown`/`Impossible`);
//! - un [`BinSpec`] derivado ([`MicrostateView::numeric_bin_spec`]) para aprendices tabulares,
//!   estimadores de entropía y control bits.
//!
//! No hay lógica de proyección duplicada: [`Scale::project`] y [`Scale::resolve_with_belief`]
//! delegan en la vista; [`Scale::bins`] expone la incrustación numérica que ya usan
//! [`crate::learned_model::TabularScale`] y compañía.
//!
//! # Declaraciones que no son una rejilla
//! [`CanonicalScale`] ofrece el mismo contrato de identidad ([`ScaleIdentity`]) para
//! proyecciones que no se expresan como ejes de bins, como la partición macro `phi` de
//! equivalencia predictiva (GOG-201). Envuelve una descripción canónica determinista y
//! documenta el mapeo; no fabrica un [`BinSpec`] falso ni duplica la lógica de la vista.
//!
//! # Identidad y reproducibilidad
//! [`Scale::identity`] devuelve un [`ScaleIdentity`] con `(id, version, hash)`. El hash es
//! `MicrostateView::spec_hash` (FNV-1a sobre etiqueta, versión, variables, unidades, ejes y bins
//! prohibidos), por lo que **las fixtures existentes conservan sus hashes** y dos construcciones
//! de la misma declaración producen la misma proyección y el mismo hash. [`ScaleIdentity`] no
//! tiene constructor público: la única forma de obtenerlo es a partir de un [`Scale`], así que un
//! planificador no puede fabricar una identidad desde parámetros sueltos.
//!
//! ```no_run
//! use gogamen_core::*;
//! let scale =
//!     Scale::uniform_grid("demo.grid", BinningVersion::V1_0, vec![(0.0, 4.0, 4)]).unwrap();
//! let tabular = TabularScale::from_scale(scale.clone()).unwrap();
//! let learner =
//!     TabularTransitionLearner::new(tabular.num_states(), 2, LearningPolicy::laplace()).unwrap();
//! let version = ModelVersion::freeze(&learner, tabular).unwrap();
//! let model = TabularLearnedModel::new(version, LearnedModelConfig::default()).unwrap();
//! // One contract: the model owns it and every planner decision records this identity.
//! assert_eq!(model.version().scale().as_scale().identity(), scale.identity());
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! fn consumes(_: &Scale) {}
//! fn bad(spec: BinSpec) {
//!     // A raw `BinSpec` is not a `Scale`: learner and planner must share the typed handle.
//!     consumes(&spec);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! fn bad(spec: BinSpec) {
//!     let _ = TabularScale::from_scale(&spec);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! fn bad(
//!     planner: &FmcPlanner,
//!     model: &TabularLearnedModel,
//!     learner_side: &TabularScale,
//!     request: &PlanRequest,
//!     rng: &mut DeterministicRng,
//! ) {
//!     // The learner-side bucket layout is not the shared contract either: both sides must take
//!     // the one `&Scale`, so a planner cannot receive a separately built discretization.
//!     let _ = plan_fmc_with_scale(planner, model, learner_side, request, rng, |_, _, _| 0.0);
//! }
//! ```
//!
//! ```compile_fail
//! use gogamen_core::*;
//! fn bad(trace: FmcTraceSummary, raw: (String, BinningVersion, u64)) -> FmcTraceSummary {
//!     // Identity cannot be forged from loose parts; it must come from a `Scale`.
//!     trace.with_scale_identity(raw)
//! }
//! ```
//!
//! # Frontera de ground truth
//! Un [`Scale`] es **declaración**, nunca estado: no almacena muestras ni verdad de terreno.
//! [`Scale::project`] exige una máscara `observed` y no lee los índices no observados (los
//! proyecta como `Cell::Unknown`). El planificador solo recibe la identidad de la escala y los
//! `State` que produce el modelo; la escala no abre ninguna vía hacia el estado real del mundo.
//!
//! # Alcance
//! Es bookkeeping de discretización reproducible. No afirma entropía física ni equivalencia
//! multiescala; cualquier interpretación requiere evidencia adicional.

use std::error::Error;
use std::fmt;

use crate::belief::DiscreteBeliefState;
use crate::determinism::TraceHasher;
use crate::domain::State;
use crate::entropy::{BinSpec, BinningVersion, EntropyError};
use crate::microstate::{
    Microstate, MicrostateError, MicrostateVariable, MicrostateView, VariableKind,
};

/// Etiqueta de dominio del hash de una [`CanonicalScale`].
const CANONICAL_SCALE_HASH_DOMAIN: &str = "gogamen.scale.canonical.v1";

/// Errores fail-closed al construir una [`Scale`].
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleError {
    /// La especificación de bins no es válida.
    BinSpec(EntropyError),
    /// La declaración de la vista no es válida (nombres, límites, bins prohibidos).
    View(MicrostateError),
    /// Una declaración canónica (no rejilla) no es válida.
    InvalidDeclaration {
        /// Motivo.
        reason: &'static str,
    },
}

impl fmt::Display for ScaleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BinSpec(err) => write!(f, "invalid bin spec for scale: {err}"),
            Self::View(err) => write!(f, "invalid microstate view for scale: {err}"),
            Self::InvalidDeclaration { reason } => {
                write!(f, "invalid scale declaration: {reason}")
            }
        }
    }
}

impl Error for ScaleError {}

impl From<EntropyError> for ScaleError {
    fn from(value: EntropyError) -> Self {
        Self::BinSpec(value)
    }
}

impl From<MicrostateError> for ScaleError {
    fn from(value: MicrostateError) -> Self {
        Self::View(value)
    }
}

/// Identidad estable de una [`Scale`]: identificador, versión de binning y hash canónico.
///
/// Sin constructor público: solo [`Scale::identity`] puede producir una instancia, de modo que
/// una decisión no puede registrar una identidad inventada a partir de parámetros sueltos.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScaleIdentity {
    id: String,
    version: BinningVersion,
    hash: u64,
}

impl ScaleIdentity {
    /// Identificador estable de la escala.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Versión del esquema de binning.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Hash FNV-1a canónico de la especificación (`MicrostateView::spec_hash`).
    pub const fn hash(&self) -> u64 {
        self.hash
    }
}

impl fmt::Display for ScaleIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {} #{:016x}", self.id, self.version, self.hash)
    }
}

/// Contrato tipado de discretización: identidad `(id, versión, hash)` más la declaración que
/// proyecta `State` en celdas tri-estado y en buckets tabulares.
///
/// Construcción:
/// - [`Scale::uniform_grid`] / [`Scale::from_bin_spec`]: rejilla numérica (el caso uniforme de
///   [`BinSpec`] es una `Scale` más).
/// - [`Scale::from_view`]: declaración completa con unidades, variables discretas/booleanas y
///   bins prohibidos.
///
/// El identificador coincide con la etiqueta de la vista/BinSpec; el hash es el `spec_hash` de la
/// vista, así que las fixtures de GOG-160 mantienen exactamente sus hashes.
#[derive(Debug, Clone, PartialEq)]
pub struct Scale {
    id: String,
    version: BinningVersion,
    hash: u64,
    bins: BinSpec,
    view: MicrostateView,
}

impl Scale {
    /// Construye una escala numérica a partir de un [`BinSpec`] ya validado.
    ///
    /// Genera una vista canónica con una variable continua `axis{i}` por eje (unidad `"1"`,
    /// índice fuente `i`), de modo que la proyección tri-estado reutiliza la vista existente.
    pub fn from_bin_spec(bins: BinSpec) -> Result<Self, ScaleError> {
        let label = bins.label().to_owned();
        let version = bins.version();
        let mut variables = Vec::with_capacity(bins.dimension());
        for (index, axis) in bins.axes().iter().enumerate() {
            variables.push(MicrostateVariable::new(
                format!("axis{index}"),
                "1",
                index,
                VariableKind::Continuous(axis.clone()),
            )?);
        }
        let view = MicrostateView::new(label, version, variables)?;
        Ok(Self::from_parts(view, bins))
    }

    /// Atajo para una rejilla uniforme: `axes` son tuplas `(min, max, num_bins)`.
    ///
    /// El caso uniforme de [`BinSpec`] es una [`Scale`] de primera clase, no un contrato aparte.
    pub fn uniform_grid(
        label: impl Into<String>,
        version: BinningVersion,
        axes: Vec<(f64, f64, usize)>,
    ) -> Result<Self, ScaleError> {
        let bins = BinSpec::uniform(label, version, axes)?;
        Self::from_bin_spec(bins)
    }

    /// Construye una escala desde una [`MicrostateView`] completa.
    ///
    /// El [`BinSpec`] asociado es [`MicrostateView::numeric_bin_spec`] (todas las variables en
    /// orden de declaración). El hash es exactamente `view.spec_hash()`, por lo que una
    /// proyección de la vista y la de la escala comparten hash de especificación.
    pub fn from_view(view: MicrostateView) -> Self {
        let bins = view.numeric_bin_spec();
        Self::from_parts(view, bins)
    }

    fn from_parts(view: MicrostateView, bins: BinSpec) -> Self {
        debug_assert_eq!(bins.version(), view.version());
        debug_assert_eq!(bins.label(), view.label());
        Self {
            id: view.label().to_owned(),
            version: view.version(),
            hash: view.spec_hash(),
            bins,
            view,
        }
    }

    /// Identificador estable de la escala.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Versión del esquema de binning.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Hash FNV-1a canónico de la especificación.
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Identidad `(id, versión, hash)` que debe registrarse en cada decisión.
    pub fn identity(&self) -> ScaleIdentity {
        ScaleIdentity {
            id: self.id.clone(),
            version: self.version,
            hash: self.hash,
        }
    }

    /// Incrustación numérica de la escala (aprendiz tabular, entropía, control bits).
    pub const fn bins(&self) -> &BinSpec {
        &self.bins
    }

    /// Declaración completa (unidades, tipos y bins prohibidos).
    pub const fn view(&self) -> &MicrostateView {
        &self.view
    }

    /// Número de ejes del [`BinSpec`] (una coordenada por variable declarada).
    pub fn dimension(&self) -> usize {
        self.bins.dimension()
    }

    /// Proyecta un [`State`] observable; `observed[i]` habilita el índice `i` de
    /// `State::values()`. Los índices no observados se proyectan `Unknown` **sin leerse**.
    pub fn project(&self, state: &State, observed: &[bool]) -> Result<Microstate, MicrostateError> {
        self.view.project(state, observed)
    }

    /// Resuelve una celda `Unknown` con una creencia colapsada del agente (ver
    /// [`MicrostateView::resolve_with_belief`]).
    pub fn resolve_with_belief(
        &self,
        microstate: &Microstate,
        index: usize,
        belief: &DiscreteBeliefState,
    ) -> Result<Microstate, MicrostateError> {
        self.view.resolve_with_belief(microstate, index, belief)
    }
}

/// Escala canónica para declaraciones que **no** son una rejilla de bins.
///
/// El caso rejilla lo cubre [`Scale`] (vista de microestado + [`BinSpec`]). Hay
/// discretizaciones legítimas que no se expresan como ejes de bins, por ejemplo una partición
/// macro `phi: micro -> macro` calculada por equivalencia predictiva (GOG-201). Para esas,
/// [`CanonicalScale`] conserva el mismo contrato de identidad que [`Scale`]
/// ([`ScaleIdentity`] con `(id, versión, hash)`) sin inventar un `BinSpec` falso:
///
/// - `id` es la etiqueta estable de la declaración (mundo/partición, horizonte y tolerancia);
/// - `version` es la versión del esquema de la proyección;
/// - `hash` es FNV-1a sobre etiqueta, versión y los **bytes canónicos** de la declaración.
///
/// Los bytes canónicos son responsabilidad del llamante y deben describir solo el contenido
/// de la proyección (p. ej. `micro -> macro` ordenado), nunca probabilidades crudas que no
/// formen parte de la partición. Así dos declaraciones con la misma partición comparten hash
/// aunque sus transiciones difieran por menos que la tolerancia, y una partición distinta
/// produce un hash distinto. No otorga permisos ni afirma nada sobre el mundo.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalScale {
    id: String,
    version: BinningVersion,
    hash: u64,
    declaration: Vec<u8>,
}

impl CanonicalScale {
    /// Construye una escala a partir de una etiqueta, una versión y los bytes canónicos de la
    /// declaración. Falla de forma cerrada si la etiqueta o la declaración están vacías.
    pub fn new(
        id: impl Into<String>,
        version: BinningVersion,
        declaration: Vec<u8>,
    ) -> Result<Self, ScaleError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(ScaleError::InvalidDeclaration {
                reason: "scale id must not be empty",
            });
        }
        if declaration.is_empty() {
            return Err(ScaleError::InvalidDeclaration {
                reason: "canonical declaration must not be empty",
            });
        }
        let hash = canonical_scale_hash(&id, version, &declaration);
        Ok(Self {
            id,
            version,
            hash,
            declaration,
        })
    }

    /// Identificador estable de la escala.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Versión del esquema de la proyección.
    pub const fn version(&self) -> BinningVersion {
        self.version
    }

    /// Hash FNV-1a canónico de `(id, versión, declaración)`.
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Bytes canónicos que definen la proyección.
    pub fn declaration(&self) -> &[u8] {
        &self.declaration
    }

    /// Identidad `(id, versión, hash)` que debe registrarse en cada decisión.
    pub fn identity(&self) -> ScaleIdentity {
        ScaleIdentity {
            id: self.id.clone(),
            version: self.version,
            hash: self.hash,
        }
    }
}

fn canonical_scale_hash(id: &str, version: BinningVersion, declaration: &[u8]) -> u64 {
    let mut hasher = TraceHasher::new();
    hasher.write_str(CANONICAL_SCALE_HASH_DOMAIN);
    hasher.write_str(id);
    hasher.write_u64(u64::from(version.major()));
    hasher.write_u64(u64::from(version.minor()));
    hasher.write_bytes(declaration);
    hasher.finish()
}

#[cfg(test)]
mod scale_contract_tests {
    use super::*;
    use crate::action::{Action, ActionSpace};
    use crate::belief::DiscreteBeliefState;
    use crate::control_bits::{
        ControlBitsConfig, control_bits_from_walkers, control_bits_from_walkers_on_scale,
    };
    use crate::determinism::MasterSeed;
    use crate::domain::{State, Terminal};
    use crate::entropy::BinAxisSpec;
    use crate::fmc_planner::{FmcPlanner, FmcPlannerConfig};
    use crate::fmc_scoring::FmcScoringConfig;
    use crate::learned_model::{
        LearnedModelConfig, ModelVersion, TabularLearnedModel, TabularScale, plan_fmc_on_version,
        plan_fmc_with_scale,
    };
    use crate::learning::{
        LearningPolicy, ObservationId, TabularTransitionLearner, TransitionObservation,
    };
    use crate::microstate::{Cell, Truth};
    use crate::model::DeterministicRng;
    use crate::planner::{CancelToken, PlanBudget, PlanRequest, PlanTarget};
    use crate::provenance::EvidenceRef;
    use crate::units::{SimDuration, StepDt, Tick};

    fn state(values: Vec<f64>) -> State {
        State::new(Tick::ZERO, SimDuration::ZERO, values).unwrap()
    }

    fn fixture_view(bins: usize, minor: u32) -> MicrostateView {
        MicrostateView::new(
            "fixture.view",
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
    fn scale_contract_uniform_grid_is_a_scale_and_reproducible() {
        let a = Scale::uniform_grid(
            "grid.uniform",
            BinningVersion::V1_0,
            vec![(0.0, 10.0, 10), (0.0, 10.0, 10)],
        )
        .unwrap();
        let b = Scale::uniform_grid(
            "grid.uniform",
            BinningVersion::V1_0,
            vec![(0.0, 10.0, 10), (0.0, 10.0, 10)],
        )
        .unwrap();
        // Two scales of the same world: same identity and reproducible projections.
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a.id(), "grid.uniform");
        assert_eq!(a.version(), BinningVersion::V1_0);
        assert_eq!(a.hash(), b.hash());
        assert_eq!(a.bins(), b.bins());
        assert_eq!(a.dimension(), 2);

        let s = state(vec![2.5, 7.5]);
        let all = [true, true];
        let pa = a.project(&s, &all).unwrap();
        let pb = b.project(&s, &all).unwrap();
        assert_eq!(pa, pb);
        assert_eq!(pa.hash(), pb.hash());
        assert_eq!(pa.cells(), &[Cell::Known(2), Cell::Known(7)]);

        // Resolution and version are part of the identity.
        let finer = Scale::uniform_grid(
            "grid.uniform",
            BinningVersion::V1_0,
            vec![(0.0, 10.0, 20), (0.0, 10.0, 10)],
        )
        .unwrap();
        let other_version = Scale::uniform_grid(
            "grid.uniform",
            BinningVersion::new(1, 1),
            vec![(0.0, 10.0, 10), (0.0, 10.0, 10)],
        )
        .unwrap();
        assert_ne!(a.hash(), finer.hash());
        assert_ne!(a.hash(), other_version.hash());
        assert_ne!(a.identity(), other_version.identity());
        assert_eq!(other_version.version(), BinningVersion::new(1, 1));

        // `Scale::new`-style construction from an existing BinSpec yields the same contract.
        let spec = BinSpec::uniform(
            "grid.uniform",
            BinningVersion::V1_0,
            vec![(0.0, 10.0, 10), (0.0, 10.0, 10)],
        )
        .unwrap();
        assert_eq!(Scale::from_bin_spec(spec).unwrap().identity(), a.identity());
    }

    #[test]
    fn scale_contract_view_fixture_hash_is_preserved() {
        let view = fixture_view(5, 0);
        let spec_hash = view.spec_hash();
        // Golden hash of the fixture declaration: `Scale` must not alter existing hashes.
        assert_eq!(spec_hash, 0x7546_7a88_e436_9d20);
        let scale = Scale::from_view(view.clone());
        // Existing fixtures keep their hashes: the Scale hash *is* the view spec hash.
        assert_eq!(scale.hash(), spec_hash);
        assert_eq!(scale.identity().hash(), spec_hash);
        assert_eq!(scale.id(), view.label());
        assert_eq!(scale.version(), view.version());
        assert_eq!(scale.view(), &view);

        // The numeric BinSpec embeds every declared variable in declaration order.
        assert_eq!(scale.dimension(), 3);
        assert_eq!(scale.bins().axes().len(), 3);
        let s = state(vec![2.5, 1.0, 1.0]);
        let all = [true; 3];
        assert_eq!(scale.bins().bin_coords(s.values()).unwrap(), vec![1, 1, 1]);
        assert_eq!(
            scale.project(&s, &all).unwrap(),
            view.project(&s, &all).unwrap()
        );

        // Rebuilding the same declaration reproduces identity, hash and projection.
        let again = Scale::from_view(fixture_view(5, 0));
        assert_eq!(again.identity(), scale.identity());
        assert_eq!(
            again.project(&s, &all).unwrap().hash(),
            scale.project(&s, &all).unwrap().hash()
        );
    }

    #[test]
    fn scale_contract_unknown_is_not_false_and_impossible_is_distinct() {
        let scale = Scale::from_view(fixture_view(5, 0));
        // Index 1 (boolean) is false but unobserved: it must project Unknown, not False.
        let s = state(vec![2.5, 0.0, 1.0]);
        let partial = scale.project(&s, &[true, false, true]).unwrap();
        assert_eq!(partial.truth(1), Some(Truth::Unknown));
        assert_eq!(partial.unknown_count(), 1);
        let full = scale.project(&s, &[true, true, true]).unwrap();
        assert_eq!(full.truth(1), Some(Truth::False));
        assert_ne!(partial.hash(), full.hash());
        assert_ne!(Truth::Unknown, Truth::False);

        // A declared-forbidden bin is Impossible, not Unknown and not False.
        let forbidden = scale
            .project(&state(vec![2.5, 1.0, 2.0]), &[true, true, true])
            .unwrap();
        assert_eq!(forbidden.cells()[2], crate::microstate::Cell::Impossible);
        assert_ne!(forbidden.cells()[2], crate::microstate::Cell::Unknown);

        // Belief only fills Unknown cells when collapsed; it never invents observations.
        let unsure = DiscreteBeliefState::new(vec![0.5, 0.5]).unwrap();
        let still = scale.resolve_with_belief(&partial, 1, &unsure).unwrap();
        assert_eq!(still.truth(1), Some(Truth::Unknown));
        let sure = DiscreteBeliefState::new(vec![0.0, 1.0]).unwrap();
        assert_eq!(
            scale
                .resolve_with_belief(&partial, 1, &sure)
                .unwrap()
                .truth(1),
            Some(Truth::True)
        );

        // Unobserved values are not read: an out-of-domain value cannot leak or fail.
        let leak = scale
            .project(&state(vec![2.5, 99.0, 1.0]), &[true, false, true])
            .unwrap();
        assert_eq!(leak.cells()[1], crate::microstate::Cell::Unknown);
        assert!(
            scale
                .project(&state(vec![2.5, 99.0, 1.0]), &[true; 3])
                .is_err()
        );
    }

    #[test]
    fn scale_contract_control_bits_use_the_same_bins() {
        let space = ActionSpace::discrete(2).unwrap();
        let scale =
            Scale::uniform_grid("grid.control", BinningVersion::V1_0, vec![(0.0, 4.0, 4)]).unwrap();
        let actions: Vec<Option<Action>> = (0..8)
            .map(|i| Some(Action::discrete_in_space(i % 2, &space).unwrap()))
            .collect();
        let states: Vec<State> = (0..8).map(|i| state(vec![(i % 4) as f64 + 0.5])).collect();
        let cfg = ControlBitsConfig::default();
        // The typed entry point is exactly the `BinSpec` entry point on `scale.bins()`.
        let via_bins = control_bits_from_walkers(&actions, &states, scale.bins(), &cfg).unwrap();
        let via_scale =
            control_bits_from_walkers_on_scale(&actions, &states, &scale, &cfg).unwrap();
        assert_eq!(via_bins, via_scale);
        assert_eq!(via_scale.scale_label, scale.id());
        assert_eq!(via_scale.scale_version, scale.version());
    }

    #[test]
    fn scale_contract_learner_and_planner_share_one_identity() {
        let scale =
            Scale::uniform_grid("grid.learned", BinningVersion::V1_0, vec![(0.0, 4.0, 4)]).unwrap();
        // The learner-side tabular scale is built *from* the same Scale handle.
        let tabular = TabularScale::from_scale(scale.clone()).unwrap();
        assert_eq!(tabular.as_scale().identity(), scale.identity());
        assert_eq!(tabular.bins(), scale.bins());

        let mut learner =
            TabularTransitionLearner::new(tabular.num_states(), 2, LearningPolicy::laplace())
                .unwrap();
        for bucket in 0..4u64 {
            let x = bucket as f64 + 0.5;
            let sample = tabular
                .sample_from_transition(&[x], 0, &[x], Terminal::Ongoing)
                .unwrap();
            learner
                .observe(&TransitionObservation::new(
                    ObservationId::new(bucket + 1),
                    EvidenceRef::new(100 + bucket),
                    sample.state_bucket(),
                    sample.action_idx(),
                    sample.next_state_bucket(),
                ))
                .unwrap();
        }
        let version = ModelVersion::freeze(&learner, tabular).unwrap();
        let model = TabularLearnedModel::new(version, LearnedModelConfig::default()).unwrap();
        assert_eq!(
            model.version().scale().as_scale().identity(),
            scale.identity()
        );

        let cfg = FmcPlannerConfig::new(
            8,
            2,
            StepDt::from_millis_f64(100.0).unwrap(),
            FmcScoringConfig::default(),
            MasterSeed::new(3),
        )
        .unwrap();
        let planner = FmcPlanner::new(cfg);
        let request = PlanRequest::new(
            PlanTarget::State(state(vec![1.5])),
            ActionSpace::discrete(2).unwrap(),
            PlanBudget::with_transitions(1_000),
            CancelToken::new(),
            None,
        )
        .unwrap();

        // The planner receives the very same Scale; every decision records its identity.
        let mut rng = DeterministicRng::new(5);
        let (_, trace) =
            plan_fmc_with_scale(&planner, &model, &scale, &request, &mut rng, |_, _, _| 0.0)
                .unwrap();
        let recorded = trace.scale_identity().expect("decision records the scale");
        assert_eq!(recorded, &scale.identity());
        assert_eq!(recorded.id(), "grid.learned");
        assert_eq!(recorded.version(), BinningVersion::V1_0);
        assert_eq!(recorded.hash(), scale.hash());

        // The default learned-model path derives the identity from the model itself.
        let mut rng = DeterministicRng::new(5);
        let (_, trace) =
            plan_fmc_on_version(&planner, &model, &request, &mut rng, |_, _, _| 0.0).unwrap();
        assert_eq!(trace.scale_identity(), Some(&scale.identity()));

        // A divergent Scale is rejected instead of silently changing the projection.
        let divergent =
            Scale::uniform_grid("grid.learned", BinningVersion::V1_0, vec![(0.0, 4.0, 8)]).unwrap();
        let mut rng = DeterministicRng::new(5);
        let err = plan_fmc_with_scale(
            &planner,
            &model,
            &divergent,
            &request,
            &mut rng,
            |_, _, _| 0.0,
        )
        .unwrap_err();
        assert!(matches!(err, crate::error::ModelError::InvalidState { .. }));
    }

    #[test]
    fn canonical_scale_identity_is_reproducible_and_content_addressed() {
        let version = BinningVersion::new(2, 1);
        let declaration = vec![0u8, 1, 0, 0, 0, 0, 0, 0, 0, 2];
        let a = CanonicalScale::new("macro.phi/demo", version, declaration.clone()).unwrap();
        let b = CanonicalScale::new("macro.phi/demo", version, declaration.clone()).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a.hash(), b.hash());
        assert_eq!(a.declaration(), declaration.as_slice());
        assert_eq!(a.id(), "macro.phi/demo");
        assert_eq!(a.version(), version);
        // The identity is a first-class `ScaleIdentity`, shareable with planners.
        let identity: ScaleIdentity = a.identity();
        assert_eq!(identity.id(), a.id());
        assert_eq!(identity.version(), a.version());
        assert_eq!(identity.hash(), a.hash());

        // Any change of label, version or canonical content changes the identity hash.
        let other_label =
            CanonicalScale::new("macro.phi/other", version, declaration.clone()).unwrap();
        let other_version = CanonicalScale::new(
            "macro.phi/demo",
            BinningVersion::new(2, 2),
            declaration.clone(),
        )
        .unwrap();
        let mut other_content = declaration;
        other_content[9] = 3;
        let other_scale = CanonicalScale::new("macro.phi/demo", version, other_content).unwrap();
        assert_ne!(a.hash(), other_label.hash());
        assert_ne!(a.hash(), other_version.hash());
        assert_ne!(a.hash(), other_scale.hash());
        assert_ne!(a.identity(), other_scale.identity());

        // A bin `Scale` and a canonical declaration never collide silently.
        let bin = Scale::uniform_grid("macro.phi/demo", version, vec![(0.0, 2.0, 2)]).unwrap();
        assert_ne!(bin.hash(), a.hash());
    }

    #[test]
    fn canonical_scale_rejects_empty_declarations() {
        let version = BinningVersion::V1_0;
        assert_eq!(
            CanonicalScale::new("  ", version, vec![1]).unwrap_err(),
            ScaleError::InvalidDeclaration {
                reason: "scale id must not be empty"
            }
        );
        assert_eq!(
            CanonicalScale::new("x", version, Vec::new()).unwrap_err(),
            ScaleError::InvalidDeclaration {
                reason: "canonical declaration must not be empty"
            }
        );
    }
}
