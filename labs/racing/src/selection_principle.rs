//! Principio de selección y sonda de libertad para el Kart FMC (GOG-198, EXPERIMENTAL).
//!
//! [`SelectionPrinciple`] (definido en `gogamen-core`) decide **cómo se construye la recompensa**
//! que recibe [`crate::FmcPlanner`] desde [`KartFmcController`]:
//!
//! - [`SelectionPrinciple::Directed`]: recompensa histórica del kart
//!   ([`KartObjectiveWeights`]: checkpoints, vueltas, progreso, velocidad, colisión, muros,
//!   energía). Es el valor por defecto y su aritmética no cambia.
//! - [`SelectionPrinciple::Pure`]: recompensa constante [`PURE_CONSTANT_REWARD`]. La puntuación no
//!   lee estado, pista, pesos ni transición: no entran checkpoints, vueltas, progreso ni
//!   velocidad. La conducta resultante es la del mismo controlador FMC con retorno neutro.
//! - [`SelectionPrinciple::Situated`]: declarado y **no implementado**; su construcción falla de
//!   forma explícita (`SelectionPrinciple::validate`). No se simula.
//!
//! # Sonda de libertad (`Pure` + proxies)
//! [`run_pure_selection_probe`] ejecuta el controlador `Pure` sobre un circuito y, **por tick**,
//! mide con `gogamen-core` el proxy empírico de entropía de walkers simulados en la modalidad
//! indicada ([`EntropyProxyKind::FutureTrajectoryProxy`] o [`EntropyProxyKind::FinalStateProxy`])
//! y registra [`SelectionTickRecord`]: principio, modalidad, horizonte, escala de bins (m), walkers
//! totales/viables, colisiones y entropía. La medición es **observacional**: no alimenta la
//! recompensa `Pure` ni cambia la decisión.
//!
//! # Qué demuestra y qué NO
//! Los tests `selection_principle_*` demuestran dos cosas acotadas:
//! 1. `Directed` conserva el hash de traza del kart en las semillas DEV (conducta histórica).
//! 2. `Pure` produce puntuaciones constantes: ninguna variable de checkpoint, vuelta, progreso o
//!    velocidad entra en el score, y cambiar los pesos no altera su traza.
//!
//! La sonda documenta cuándo esta demo concreta se detiene, choca o no avanza (ver
//! [`SelectionEpisodeTrace::stopped_ticks`], [`SelectionEpisodeTrace::collision_ticks`] y
//! [`SelectionEpisodeTrace::non_advancing_ticks`]). **No** es una prueba general de inteligencia,
//! de emergencia ni de superioridad de un principio: es un controlador FMC de horizonte corto
//! sobre un modelo cinemático conocido, con poblaciones y semillas fijas.
//!
//! # Observaciones medidas (semilla 7, 27 walkers/tick, horizonte 6, escala 1 m, 60 ticks)
//! Estos valores son los que imprimen los tests `selection_principle_pure_runs_*` en esta
//! máquina; describen **estas ejecuciones concretas**, no una ley general.
//! - `Pure` + `rectangular-circuit` (ambas modalidades): desenlace `Timeout`, 0 colisiones,
//!   3/60 ticks `Stopped`, `no_progress_ticks = 60/60`, 0 checkpoints. La velocidad se vuelve
//!   negativa (hasta `-2,18 m/s`): el kart retrocede y no completa la vuelta. La entropía del
//!   proxy varía entre `0,0` y `3,78` bits según el tick.
//! - `Pure` + `chicane-circuit` (ambas modalidades): mismo desenlace, mismas velocidades, 0
//!   checkpoints y `no_progress_ticks = 60/60`. La secuencia de control es **idéntica** a la del
//!   circuito base (comprobado en el test): en esta configuración la recompensa constante no usa
//!   la geometría de la pista para elegir.
//! - `Directed` + `rectangular-circuit`: completa la vuelta en 177 (semilla 7), 175 (19) y 169
//!   (42) ticks con hashes de traza estables (test de preservación).
//!
//! Conclusión acotada: `Pure` es un controlador neutro determinista que aquí **no avanza** hacia la
//! meta; el valor de la demo es la auditabilidad de la puntuación y la trazabilidad por tick, no
//! un resultado de rendimiento. Que el kart se detenga, retroceda o choque es un resultado válido
//! y se informa tal cual.

use gogamen_core::{
    Action, Actuator, BinSpec, BinningVersion, DomainError, EntropyError, EntropyProxyKind,
    EnvironmentError, EnvironmentObserver, GenerativeModel, LiveEnvironment, MasterSeed,
    ModelError, RngSource, SelectionPrinciple, State, StreamKey, Terminal, TerminalReason,
    TrajectoryStep, TrajectorySummary, Transition, estimate_trajectory_entropy_proxy,
};

use crate::autonomous_fmc::{
    KartDecisionReport, KartFmcController, KartFmcControllerConfig, KartObjectiveWeights,
    KartRunError, evaluate_kart_transition,
};
use crate::physics::{KartControl, KartState, KinematicBicycleConfig};
use crate::track::TrackMap;
use crate::world::{KartEnv, KartModel};

/// Recompensa constante de [`SelectionPrinciple::Pure`] (no depende de estado ni pesos).
pub const PURE_CONSTANT_REWARD: f64 = 0.0;

/// Etiqueta estable del esquema de bins 2D `(x, y)` usado por la sonda.
pub const PURE_PROBE_SCALE_LABEL: &str = "kart_xy_uniform_m";
/// Versión del esquema de bins de la sonda.
pub const PURE_PROBE_SCALE_VERSION: BinningVersion = BinningVersion::V1_0;
/// Escala por defecto de la sonda: lado de celda en metros.
pub const DEFAULT_PURE_PROBE_SCALE_M: f64 = 1.0;
/// Walkers por acción raíz por defecto.
pub const DEFAULT_PURE_PROBE_WALKERS_PER_ROOT: usize = 4;
/// Horizonte de rollout por defecto de la sonda (pasos).
pub const DEFAULT_PURE_PROBE_HORIZON: usize = 8;
/// Límite duro de walkers por acción raíz.
pub const MAX_PURE_PROBE_WALKERS_PER_ROOT: usize = 64;
/// Límite duro de horizonte de la sonda.
pub const MAX_PURE_PROBE_HORIZON: usize = 64;
/// Número de acciones raíz de la retícula fija (`3 direcciones x 3 aceleraciones`).
pub const PURE_PROBE_ROOTS: usize = 9;
/// Velocidad (m/s) bajo la cual un tick se etiqueta [`SelectionBehavior::Stopped`].
const STOPPED_SPEED: f64 = 0.1;

/// Construye la recompensa de una transición según un [`SelectionPrinciple`].
///
/// Es la única vía por la que [`KartFmcController`] obtiene recompensas: `Directed` delega en la
/// aritmética histórica [`evaluate_kart_transition`] sin cambios; `Pure` devuelve la constante
/// [`PURE_CONSTANT_REWARD`] sin leer estado, pista, pesos ni transición.
#[derive(Debug, Clone, Copy)]
pub struct KartRewardFunction<'a> {
    principle: SelectionPrinciple,
    track: &'a TrackMap,
    kinematics: &'a KinematicBicycleConfig,
    weights: &'a KartObjectiveWeights,
}

impl<'a> KartRewardFunction<'a> {
    /// Construye la función validando el principio (fail-closed para `Situated`).
    pub fn new(
        principle: SelectionPrinciple,
        track: &'a TrackMap,
        kinematics: &'a KinematicBicycleConfig,
        weights: &'a KartObjectiveWeights,
    ) -> Result<Self, DomainError> {
        principle.validate()?;
        Ok(Self {
            principle,
            track,
            kinematics,
            weights,
        })
    }

    /// Principio con el que se construyó la función.
    pub const fn principle(&self) -> SelectionPrinciple {
        self.principle
    }

    /// Recompensa de una transición simulada.
    pub fn reward(&self, prev_state: &State, action: &Action, transition: &Transition) -> f64 {
        match self.principle {
            SelectionPrinciple::Directed => evaluate_kart_transition(
                self.track,
                self.kinematics,
                self.weights,
                prev_state,
                action,
                transition,
            ),
            // Recompensa constante: no se lee `prev_state`, `action` ni `transition`.
            SelectionPrinciple::Pure => PURE_CONSTANT_REWARD,
            // `KartRewardFunction::new` rechaza `Situated`; esta rama es inalcanzable.
            SelectionPrinciple::Situated => {
                unreachable!("SelectionPrinciple::Situated is rejected at construction")
            }
        }
    }
}

/// Configuración de la sonda de libertad por tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectionProbeConfig {
    /// Walkers simulados por cada acción raíz de la retícula (`1..=MAX_PURE_PROBE_WALKERS_PER_ROOT`).
    pub walkers_per_root: usize,
    /// Pasos por rollout (`1..=MAX_PURE_PROBE_HORIZON`).
    pub horizon: usize,
    /// Lado de celda en metros para discretizar posiciones `(x, y)`.
    pub scale_m: f64,
    /// Semilla determinista de los walkers.
    pub seed: MasterSeed,
}

impl SelectionProbeConfig {
    /// Construye la configuración validando límites y escala.
    pub fn new(
        walkers_per_root: usize,
        horizon: usize,
        scale_m: f64,
        seed: MasterSeed,
    ) -> Result<Self, DomainError> {
        if walkers_per_root == 0 || walkers_per_root > MAX_PURE_PROBE_WALKERS_PER_ROOT {
            return Err(DomainError::InvalidTransition {
                reason: "pure probe walkers_per_root must be in 1..=MAX_PURE_PROBE_WALKERS_PER_ROOT",
            });
        }
        if horizon == 0 || horizon > MAX_PURE_PROBE_HORIZON {
            return Err(DomainError::InvalidTransition {
                reason: "pure probe horizon must be in 1..=MAX_PURE_PROBE_HORIZON",
            });
        }
        if !scale_m.is_finite() || scale_m <= 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "pure probe scale_m must be finite and > 0.0",
            });
        }
        Ok(Self {
            walkers_per_root,
            horizon,
            scale_m,
            seed,
        })
    }

    /// Número total de walkers por tick (`PURE_PROBE_ROOTS * walkers_per_root`).
    pub const fn walkers_per_tick(&self) -> usize {
        PURE_PROBE_ROOTS * self.walkers_per_root
    }

    /// Especificación de bins 2D `(x, y)` que cubre la caja límite del circuito.
    pub fn bin_spec(&self, track: &TrackMap) -> Result<BinSpec, EntropyError> {
        let min = track.bounds_min();
        let max = track.bounds_max();
        let span_x = max.x - min.x;
        let span_y = max.y - min.y;
        if !(span_x > 0.0) || !(span_y > 0.0) {
            return Err(EntropyError::InvalidBinSpec {
                reason: "track bounds must have positive span",
            });
        }
        let bins_x = (span_x / self.scale_m).ceil().max(1.0) as usize;
        let bins_y = (span_y / self.scale_m).ceil().max(1.0) as usize;
        BinSpec::uniform(
            PURE_PROBE_SCALE_LABEL,
            PURE_PROBE_SCALE_VERSION,
            vec![(min.x, max.x, bins_x), (min.y, max.y, bins_y)],
        )
    }
}

/// Comportamiento observado del kart tras aplicar la decisión del tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionBehavior {
    /// Aceleración positiva elegida.
    Accelerate,
    /// Aceleración nula elegida.
    Coast,
    /// Aceleración negativa elegida con el kart en marcha.
    Brake,
    /// El kart queda (casi) parado tras el tick: elige no avanzar.
    Stopped,
    /// El tick termina en colisión o violación de invariante.
    Collided,
}

/// Registro por tick y modalidad de la sonda `Pure`.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionTickRecord {
    /// Índice de tick de decisión (0-based).
    pub tick: usize,
    /// Principio de selección activo (siempre `Pure` en esta sonda).
    pub principle: SelectionPrinciple,
    /// Modalidad del proxy de entropía.
    pub modality: EntropyProxyKind,
    /// Horizonte de rollout usado por la sonda.
    pub horizon: usize,
    /// Escala de discretización en metros.
    pub scale_m: f64,
    /// Etiqueta del esquema de bins.
    pub scale_label: String,
    /// Versión del esquema de bins.
    pub scale_version: (u32, u32),
    /// Walkers simulados en el tick.
    pub walkers: usize,
    /// Walkers viables (sin terminal de fallo).
    pub viable_walkers: usize,
    /// Walkers que chocaron o violaron un invariante.
    pub collisions: usize,
    /// Entropía empírica (bits) de los walkers viables en la modalidad indicada.
    pub entropy_bits: f64,
    /// Número de bins ocupados por los walkers viables.
    pub occupied_bins: usize,
    /// Control aplicado en el tick.
    pub control: KartControl,
    /// Comportamiento observado.
    pub behavior: SelectionBehavior,
    /// Posición X tras aplicar el control.
    pub x: f64,
    /// Posición Y tras aplicar el control.
    pub y: f64,
    /// Velocidad tras aplicar el control.
    pub speed: f64,
    /// Checkpoints superados tras el tick (solo observación; no entra en el score de `Pure`).
    pub checkpoints_cleared: usize,
}

/// Desenlace de una sonda `Pure`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionProbeOutcome {
    /// El kart alcanzó la meta (efecto externo del entorno; no se puntúa).
    Completed,
    /// El kart chocó o violó un invariante.
    Collided,
    /// Se agotaron los ticks sin terminar.
    Timeout,
    /// El planificador no autorizó ninguna acción válida.
    NoAuthorizedAction,
}

/// Traza de una sonda `Pure` sobre un circuito y una modalidad.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionEpisodeTrace {
    /// Nombre de la pista.
    pub track_name: String,
    /// Principio activo.
    pub principle: SelectionPrinciple,
    /// Modalidad medida en todos los registros.
    pub modality: EntropyProxyKind,
    /// Registros por tick.
    pub records: Vec<SelectionTickRecord>,
    /// Desenlace.
    pub outcome: SelectionProbeOutcome,
}

impl SelectionEpisodeTrace {
    /// Hash FNV-1a 64 determinista de la traza (excluye latencias y tiempo real).
    pub fn trace_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        feed(self.principle as u64);
        feed(modality_tag(self.modality));
        for r in &self.records {
            feed(r.tick as u64);
            feed(r.principle as u64);
            feed(modality_tag(r.modality));
            feed(r.horizon as u64);
            feed(r.scale_m.to_bits());
            for b in r.scale_label.bytes() {
                feed(u64::from(b));
            }
            feed(u64::from(r.scale_version.0));
            feed(u64::from(r.scale_version.1));
            feed(r.walkers as u64);
            feed(r.viable_walkers as u64);
            feed(r.collisions as u64);
            feed(r.entropy_bits.to_bits());
            feed(r.occupied_bins as u64);
            feed(r.control.steering_angle.to_bits());
            feed(r.control.throttle_accel.to_bits());
            feed(r.behavior as u64);
            feed(r.x.to_bits());
            feed(r.y.to_bits());
            feed(r.speed.to_bits());
            feed(r.checkpoints_cleared as u64);
        }
        feed(self.outcome as u64);
        h
    }

    /// Ticks etiquetados [`SelectionBehavior::Stopped`].
    pub fn stopped_ticks(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.behavior == SelectionBehavior::Stopped)
            .count()
    }

    /// Ticks etiquetados [`SelectionBehavior::Collided`].
    pub fn collision_ticks(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.behavior == SelectionBehavior::Collided)
            .count()
    }

    /// Ticks sin avance de posición medible (`< 1e-6 m` desde el tick anterior).
    ///
    /// El primer tick no tiene predecesor y no se cuenta. Un kart que retrocede físicamente no
    /// cuenta aquí; para el avance hacia la meta usa [`Self::no_progress_ticks`].
    pub fn non_advancing_ticks(&self) -> usize {
        self.records
            .windows(2)
            .filter(|pair| {
                let a = &pair[0];
                let b = &pair[1];
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                (dx * dx + dy * dy).sqrt() < 1e-6
            })
            .count()
    }

    /// Ticks que no incrementan `checkpoints_cleared` respecto al tick anterior (el primero cuenta
    /// como sin progreso). En `Pure` el score no lee checkpoints; este contador solo documenta la
    /// conducta observada.
    pub fn no_progress_ticks(&self) -> usize {
        let mut previous = 0usize;
        let mut count = 0usize;
        for record in &self.records {
            if record.checkpoints_cleared <= previous {
                count += 1;
            }
            previous = record.checkpoints_cleared;
        }
        count
    }

    /// Checkpoints superados al final de la traza (`0` si no hay registros).
    pub fn checkpoints_cleared(&self) -> usize {
        self.records
            .last()
            .map_or(0, |record| record.checkpoints_cleared)
    }

    /// Línea de texto por tick (modalidad, horizonte, escala, walkers viables, colisiones).
    pub fn format_line(r: &SelectionTickRecord) -> String {
        format!(
            "t={} principle={} modality={:?} h={} scale={:.2}m walkers={} viable={} collisions={} H={:.3} behavior={:?} v={:.2} cp={}",
            r.tick,
            r.principle.name(),
            r.modality,
            r.horizon,
            r.scale_m,
            r.walkers,
            r.viable_walkers,
            r.collisions,
            r.entropy_bits,
            r.behavior,
            r.speed,
            r.checkpoints_cleared
        )
    }
}

/// Errores explícitos de la sonda de selección.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionProbeError {
    /// Configuración o principio inválido.
    Domain(DomainError),
    /// Error del entorno vivo.
    Environment(EnvironmentError),
    /// Error del modelo generativo o del planificador.
    Model(ModelError),
    /// Error al discretizar o estimar el proxy de entropía.
    Entropy(EntropyError),
    /// La sonda exige [`SelectionPrinciple::Pure`].
    PrincipleNotPure(SelectionPrinciple),
    /// Error del bucle del controlador.
    Controller(KartRunError),
}

impl std::fmt::Display for SelectionProbeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Domain(err) => write!(f, "domain error: {err}"),
            Self::Environment(err) => write!(f, "environment error: {err}"),
            Self::Model(err) => write!(f, "model error: {err}"),
            Self::Entropy(err) => write!(f, "entropy error: {err}"),
            Self::PrincipleNotPure(p) => {
                write!(f, "pure selection probe requires Pure, got {}", p.name())
            }
            Self::Controller(KartRunError::Environment(err)) => {
                write!(f, "controller environment error: {err}")
            }
            Self::Controller(KartRunError::Model(err)) => {
                write!(f, "controller model error: {err}")
            }
        }
    }
}

impl std::error::Error for SelectionProbeError {}

impl From<DomainError> for SelectionProbeError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

impl From<EnvironmentError> for SelectionProbeError {
    fn from(err: EnvironmentError) -> Self {
        Self::Environment(err)
    }
}

impl From<ModelError> for SelectionProbeError {
    fn from(err: ModelError) -> Self {
        Self::Model(err)
    }
}

impl From<EntropyError> for SelectionProbeError {
    fn from(err: EntropyError) -> Self {
        Self::Entropy(err)
    }
}

impl From<KartRunError> for SelectionProbeError {
    fn from(err: KartRunError) -> Self {
        Self::Controller(err)
    }
}

fn modality_tag(modality: EntropyProxyKind) -> u64 {
    match modality {
        EntropyProxyKind::FutureTrajectoryProxy => 1,
        EntropyProxyKind::FinalStateProxy => 2,
    }
}

struct TickMeasurement {
    walkers: usize,
    viable_walkers: usize,
    collisions: usize,
    entropy_bits: f64,
    occupied_bins: usize,
}

/// Mide el proxy de entropía de walkers simulados desde `start` en la modalidad dada.
fn measure_tick(
    model: &KartModel,
    start: &State,
    bin_spec: &BinSpec,
    probe: &SelectionProbeConfig,
    tick: usize,
    modality: EntropyProxyKind,
) -> Result<TickMeasurement, SelectionProbeError> {
    let kin = model.config().clone();
    let space = model.action_space().clone();
    let mut trajectories: Vec<TrajectorySummary> = Vec::new();
    let mut walkers = 0usize;
    let mut collisions = 0usize;

    for (root_idx, (steer, accel)) in root_controls(&kin).into_iter().enumerate() {
        for walker_idx in 0..probe.walkers_per_root {
            walkers += 1;
            let mut rng = probe.seed.derive(StreamKey::walker(
                tick as u64,
                root_idx as u64,
                walker_idx as u64,
            ));
            let mut state = start.clone();
            let mut steps: Vec<TrajectoryStep> = Vec::with_capacity(probe.horizon);
            let mut collided = false;

            for step in 0..probe.horizon {
                let action = if step == 0 {
                    Action::continuous_in_space(vec![steer, accel], &space)
                        .map_err(ModelError::InvalidAction)?
                } else {
                    let st = (2.0 * rng.next_f64() - 1.0) * kin.max_steering_angle;
                    let ac = kin.min_accel + rng.next_f64() * (kin.max_accel - kin.min_accel);
                    Action::continuous_in_space(vec![st, ac], &space)
                        .map_err(ModelError::InvalidAction)?
                };
                let transition =
                    match GenerativeModel::step(model, &state, &action, kin.dt, &mut rng) {
                        Ok(tr) => tr,
                        Err(ModelError::TerminalState) => {
                            collided = true;
                            break;
                        }
                        Err(err) => return Err(err.into()),
                    };
                state = transition.next_state().clone();
                let kart = KartState::from_slice(&state.values()[0..4])
                    .map_err(ModelError::TransitionFailed)?;
                steps.push(TrajectoryStep::new(state.tick(), vec![kart.x, kart.y])?);
                match transition.terminal() {
                    Terminal::Ongoing => {}
                    // La meta externa no es un fallo del walker: es viable y termina ahí.
                    Terminal::Terminated(TerminalReason::GoalReached) => break,
                    Terminal::Terminated(_) => {
                        collided = true;
                        break;
                    }
                }
            }

            if collided || steps.is_empty() {
                collisions += 1;
            } else {
                trajectories.push(TrajectorySummary::new(steps)?);
            }
        }
    }

    let viable_walkers = trajectories.len();
    let (entropy_bits, occupied_bins) = if trajectories.is_empty() {
        (0.0, 0)
    } else {
        let estimate = estimate_trajectory_entropy_proxy(&trajectories, bin_spec, modality)?;
        (estimate.entropy_bits(), estimate.occupied_bins())
    };

    Ok(TickMeasurement {
        walkers,
        viable_walkers,
        collisions,
        entropy_bits,
        occupied_bins,
    })
}

/// Retícula fija de controles raíz `{-max, 0, +max}` dirección x `{min, 0, max}` aceleración.
fn root_controls(kin: &KinematicBicycleConfig) -> [(f64, f64); PURE_PROBE_ROOTS] {
    [
        (-kin.max_steering_angle, kin.min_accel),
        (-kin.max_steering_angle, 0.0),
        (-kin.max_steering_angle, kin.max_accel),
        (0.0, kin.min_accel),
        (0.0, 0.0),
        (0.0, kin.max_accel),
        (kin.max_steering_angle, kin.min_accel),
        (kin.max_steering_angle, 0.0),
        (kin.max_steering_angle, kin.max_accel),
    ]
}

/// Ejecuta el controlador `Pure` y registra, por tick, el proxy de entropía en `modality`.
///
/// La medición no alimenta la recompensa: `Pure` sigue siendo retorno constante. El bucle del
/// entorno es determinista (semillas fijas, sin reloj de pared en la traza).
pub fn run_pure_selection_probe(
    kinematics: KinematicBicycleConfig,
    track: TrackMap,
    controller_config: KartFmcControllerConfig,
    probe: SelectionProbeConfig,
    modality: EntropyProxyKind,
    max_ticks: usize,
) -> Result<SelectionEpisodeTrace, SelectionProbeError> {
    if controller_config.principle != SelectionPrinciple::Pure {
        return Err(SelectionProbeError::PrincipleNotPure(
            controller_config.principle,
        ));
    }
    if max_ticks == 0 {
        return Err(DomainError::InvalidTransition {
            reason: "pure selection probe max_ticks must be >= 1",
        }
        .into());
    }
    let bin_spec = probe.bin_spec(&track)?;
    let mut controller =
        KartFmcController::new(kinematics.clone(), track.clone(), controller_config)?;
    let mut env = KartEnv::new(kinematics, track)?;
    LiveEnvironment::reset(&mut env)?;

    let mut records = Vec::with_capacity(max_ticks);
    let mut outcome = SelectionProbeOutcome::Timeout;

    for tick in 0..max_ticks {
        let obs = EnvironmentObserver::observe(&env)?;
        let start = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
            .map_err(ModelError::TransitionFailed)?;
        let measurement = measure_tick(
            controller.model(),
            &start,
            &bin_spec,
            &probe,
            tick,
            modality,
        )?;

        let report: KartDecisionReport = controller.decide_from_env(&env)?;
        let (Some(action), Some(control)) = (report.action, report.control) else {
            outcome = SelectionProbeOutcome::NoAuthorizedAction;
            break;
        };

        let before_speed = env.kart_state().speed;
        let apply_out = env.apply(&action)?;
        let kart = env.kart_state();
        let terminal = apply_out.terminal();
        let behavior = if matches!(
            terminal,
            Terminal::Terminated(TerminalReason::Failure | TerminalReason::InvariantViolated)
        ) {
            SelectionBehavior::Collided
        } else if kart.speed.abs() < STOPPED_SPEED {
            SelectionBehavior::Stopped
        } else if control.throttle_accel > 0.0 {
            SelectionBehavior::Accelerate
        } else if control.throttle_accel < 0.0 && before_speed.abs() > STOPPED_SPEED {
            SelectionBehavior::Brake
        } else {
            SelectionBehavior::Coast
        };

        records.push(SelectionTickRecord {
            tick,
            principle: SelectionPrinciple::Pure,
            modality,
            horizon: probe.horizon,
            scale_m: probe.scale_m,
            scale_label: bin_spec.label().to_owned(),
            scale_version: (bin_spec.version().major(), bin_spec.version().minor()),
            walkers: measurement.walkers,
            viable_walkers: measurement.viable_walkers,
            collisions: measurement.collisions,
            entropy_bits: measurement.entropy_bits,
            occupied_bins: measurement.occupied_bins,
            control,
            behavior,
            x: kart.x,
            y: kart.y,
            speed: kart.speed,
            checkpoints_cleared: env.checkpoints_cleared(),
        });

        match terminal {
            Terminal::Ongoing => {}
            Terminal::Terminated(TerminalReason::GoalReached) => {
                outcome = SelectionProbeOutcome::Completed;
                break;
            }
            Terminal::Terminated(_) => {
                outcome = SelectionProbeOutcome::Collided;
                break;
            }
        }
    }

    Ok(SelectionEpisodeTrace {
        track_name: env.track().name().to_owned(),
        principle: SelectionPrinciple::Pure,
        modality,
        records,
        outcome,
    })
}

// `KartEpisodeOutcome` is part of the trace contract re-exported by the crate root.

#[cfg(test)]
mod tests {
    use super::*;

    use gogamen_core::{Observation, SimDuration, Tick};

    use crate::autonomous_fmc::{DEV_SEEDS, KartEpisodeOutcome};

    fn kin() -> KinematicBicycleConfig {
        KinematicBicycleConfig::default()
    }

    fn synth_transition(
        prev_values: Vec<f64>,
        next_values: Vec<f64>,
        terminal: Terminal,
        action: &Action,
    ) -> (State, Transition) {
        let tick = Tick::new(3);
        let prev =
            State::new(tick, SimDuration::from_secs_f64(0.15).unwrap(), prev_values).unwrap();
        let next = State::new(
            Tick::new(4),
            SimDuration::from_secs_f64(0.20).unwrap(),
            next_values,
        )
        .unwrap();
        let obs = Observation::new(next.tick(), next.elapsed(), vec![next.values()[0]]).unwrap();
        let tr = Transition::new(prev.tick(), kin().dt, next, obs, terminal).unwrap();
        let _ = action;
        (prev, tr)
    }

    #[test]
    fn selection_principle_pure_reward_is_constant_under_objective_perturbations() {
        let track = TrackMap::rectangular_circuit();
        let k = kin();
        let weights = KartObjectiveWeights::default();
        let pure = KartRewardFunction::new(SelectionPrinciple::Pure, &track, &k, &weights).unwrap();
        let directed =
            KartRewardFunction::new(SelectionPrinciple::Directed, &track, &k, &weights).unwrap();
        let target = track.checkpoints()[0].center;

        // Casos que mueven SOLO variables de objetivo: cursor de checkpoint, vuelta (GoalReached),
        // progreso (acercarse/alejarse), velocidad y aceleración aplicada.
        let cases: Vec<(Vec<f64>, Vec<f64>, Terminal, f64)> = vec![
            (
                vec![target.x - 3.0, target.y, 0.0, 0.0, 0.0],
                vec![target.x - 2.0, target.y, 0.0, 0.0, 0.0],
                Terminal::Ongoing,
                0.0,
            ),
            (
                vec![target.x - 3.0, target.y, 0.0, 0.0, 0.0],
                vec![target.x - 2.0, target.y, 0.0, 0.0, 1.0],
                Terminal::Ongoing,
                4.0,
            ),
            (
                vec![target.x - 3.0, target.y, 0.0, 0.0, 0.0],
                vec![target.x, target.y, 0.0, 6.0, 1.0],
                Terminal::Terminated(TerminalReason::GoalReached),
                8.0,
            ),
            (
                vec![target.x - 1.0, target.y, 0.0, 5.0, 1.0],
                vec![target.x - 4.0, target.y, 0.0, -2.0, 1.0],
                Terminal::Terminated(TerminalReason::Failure),
                -8.0,
            ),
        ];

        let mut directed_values = Vec::new();
        for (prev_vals, next_vals, terminal, throttle) in cases {
            let action = Action::continuous(vec![0.3, throttle]).unwrap();
            let (prev, tr) = synth_transition(prev_vals, next_vals, terminal, &action);
            assert_eq!(
                pure.reward(&prev, &action, &tr),
                PURE_CONSTANT_REWARD,
                "Pure must be constant regardless of objective state"
            );
            directed_values.push(directed.reward(&prev, &action, &tr));
        }
        assert_eq!(
            pure.principle(),
            SelectionPrinciple::Pure,
            "pure function keeps its principle"
        );
        // No vacuidad: la misma variación sí mueve la recompensa dirigida.
        assert!(
            directed_values
                .windows(2)
                .any(|w| w[0].to_bits() != w[1].to_bits()),
            "Directed rewards should differ across objective perturbations: {directed_values:?}"
        );
    }

    fn pure_controller_config(seed: u64) -> KartFmcControllerConfig {
        let k = kin();
        KartFmcControllerConfig::with_dt_and_seed(k.dt, MasterSeed::new(seed))
            .with_principle(SelectionPrinciple::Pure)
            .unwrap()
    }

    #[test]
    fn selection_principle_pure_fmc_score_is_exactly_zero_and_state_independent() {
        let k = kin();
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(k.clone(), track.clone()).unwrap();
        let mut ctl =
            KartFmcController::new(k.clone(), track, pure_controller_config(DEV_SEEDS[0])).unwrap();
        LiveEnvironment::reset(&mut env).unwrap();

        for _ in 0..5 {
            let report = ctl.decide_from_env(&env).unwrap();
            assert!(!report.trace.root_action_stats().is_empty());
            for stat in report.trace.root_action_stats() {
                // Retorno medio y score constantes: ni checkpoint, ni vuelta, ni progreso, ni
                // velocidad entran en la puntuación (los pesos no se consultan).
                assert_eq!(stat.mean_return, 0.0, "pure mean_return must be constant");
                assert_eq!(stat.score, 0.0, "pure score must be constant");
            }
            assert_eq!(report.trace.selected_action_score(), Some(0.0));
            let action = report.action.clone().unwrap();
            let obs = EnvironmentObserver::observe(&env).unwrap();
            let start = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec()).unwrap();
            for fut in ctl.candidate_futures(&start, &report).unwrap() {
                assert_eq!(fut.hold_return, 0.0);
            }
            env.apply(&action).unwrap();
        }
    }

    #[test]
    fn selection_principle_pure_trace_ignores_weights_but_directed_does_not() {
        let k = kin();
        let track = TrackMap::rectangular_circuit();
        let base = pure_controller_config(DEV_SEEDS[1]);
        let mut zeroed = base.clone();
        zeroed.weights = KartObjectiveWeights {
            checkpoint_bonus: 0.0,
            lap_goal_bonus: 0.0,
            progress_weight: 0.0,
            forward_speed_weight: 0.0,
            collision_penalty: 0.0,
            wall_proximity_weight: 0.0,
            wall_safety_margin: 0.5,
            energy_weight: 0.0,
        };

        let mut env_a = KartEnv::new(k.clone(), track.clone()).unwrap();
        let mut ctl_a = KartFmcController::new(k.clone(), track.clone(), base.clone()).unwrap();
        let trace_a = ctl_a.run_trace(&mut env_a, 300).unwrap();

        let mut env_b = KartEnv::new(k.clone(), track.clone()).unwrap();
        let mut ctl_b = KartFmcController::new(k.clone(), track.clone(), zeroed).unwrap();
        let trace_b = ctl_b.run_trace(&mut env_b, 300).unwrap();
        assert_eq!(trace_a, trace_b);
        assert_eq!(trace_a.trace_hash(), trace_b.trace_hash());

        // Directed con los mismos pesos sí cambia: confirma que la invariancia de Pure no es vacua.
        let directed_cfg =
            KartFmcControllerConfig::with_dt_and_seed(k.dt, MasterSeed::new(DEV_SEEDS[1]));
        let mut env_c = KartEnv::new(k.clone(), track.clone()).unwrap();
        let mut ctl_c = KartFmcController::new(k.clone(), track, directed_cfg).unwrap();
        let trace_c = ctl_c.run_trace(&mut env_c, 300).unwrap();
        assert_ne!(trace_a.trace_hash(), trace_c.trace_hash());
        assert_eq!(trace_c.principle, SelectionPrinciple::Directed);
    }

    #[test]
    fn selection_principle_directed_preserves_dev_seed_trace_hashes() {
        // Hashes medidos ANTES de introducir el principio (semillas DEV, circuito base, 900 ticks).
        const GOLDEN: [(u64, u64); 3] = [
            (DEV_SEEDS[0], 0xee4c_86ad_4494_9b46),
            (DEV_SEEDS[1], 0x909c_02b2_cff7_f73e),
            (DEV_SEEDS[2], 0xba6b_ef32_42c5_f725),
        ];
        let k = kin();
        for (seed, expected) in GOLDEN {
            let track = TrackMap::rectangular_circuit();
            let mut env = KartEnv::new(k.clone(), track.clone()).unwrap();
            let cfg = KartFmcControllerConfig::with_dt_and_seed(k.dt, MasterSeed::new(seed));
            assert_eq!(cfg.principle, SelectionPrinciple::Directed);
            let mut ctl = KartFmcController::new(k.clone(), track, cfg).unwrap();
            let trace = ctl.run_trace(&mut env, 900).unwrap();
            assert_eq!(trace.principle, SelectionPrinciple::Directed);
            assert_eq!(
                trace.trace_hash(),
                expected,
                "Directed trace hash changed for seed {seed}"
            );
            assert_eq!(trace.outcome, KartEpisodeOutcome::Completed);
            println!(
                "GOG198 directed seed={seed} ticks={} hash={:#018x}",
                trace.records.len(),
                trace.trace_hash()
            );
        }
    }

    #[test]
    fn selection_principle_situated_is_declared_and_fails_closed() {
        let k = kin();
        let track = TrackMap::rectangular_circuit();
        let weights = KartObjectiveWeights::default();
        assert!(
            KartRewardFunction::new(SelectionPrinciple::Situated, &track, &k, &weights).is_err()
        );
        let cfg = KartFmcControllerConfig::with_dt_and_seed(k.dt, MasterSeed::new(7));
        assert!(matches!(
            cfg.with_principle(SelectionPrinciple::Situated),
            Err(DomainError::InvalidTransition { .. })
        ));
        assert!(!SelectionPrinciple::Situated.is_implemented());
    }

    #[test]
    fn selection_principle_pure_runs_two_circuits_with_both_proxies() {
        let k = kin();
        let probe = SelectionProbeConfig::new(3, 6, 1.0, MasterSeed::new(DEV_SEEDS[0])).unwrap();
        let modalities = [
            EntropyProxyKind::FutureTrajectoryProxy,
            EntropyProxyKind::FinalStateProxy,
        ];
        let mut hashes = Vec::new();
        let mut control_sequences: Vec<Vec<(u64, u64)>> = Vec::new();

        for track in [TrackMap::rectangular_circuit(), TrackMap::chicane_circuit()] {
            let name = track.name().to_owned();
            for modality in modalities {
                let cfg = pure_controller_config(DEV_SEEDS[0]);
                let trace = run_pure_selection_probe(
                    k.clone(),
                    track.clone(),
                    cfg.clone(),
                    probe,
                    modality,
                    MAX_TICKS,
                )
                .unwrap();
                assert_eq!(trace.track_name, name);
                assert_eq!(trace.principle, SelectionPrinciple::Pure);
                assert_eq!(trace.modality, modality);
                assert!(!trace.records.is_empty());
                for r in &trace.records {
                    assert_eq!(r.principle, SelectionPrinciple::Pure);
                    assert_eq!(r.modality, modality);
                    assert_eq!(r.horizon, 6);
                    assert_eq!(r.scale_m, 1.0);
                    assert_eq!(r.scale_label, PURE_PROBE_SCALE_LABEL);
                    assert_eq!(r.scale_version, (1, 0));
                    assert_eq!(r.walkers, PURE_PROBE_ROOTS * 3);
                    assert!(r.viable_walkers + r.collisions <= r.walkers);
                    assert!(r.entropy_bits.is_finite() && r.entropy_bits >= 0.0);
                    assert!(r.occupied_bins <= r.viable_walkers.max(1));
                }
                // Reproducibilidad: misma semilla, misma traza y hash.
                let again = run_pure_selection_probe(
                    k.clone(),
                    track.clone(),
                    cfg,
                    probe,
                    modality,
                    MAX_TICKS,
                )
                .unwrap();
                assert_eq!(trace, again);
                hashes.push(trace.trace_hash());
                control_sequences.push(
                    trace
                        .records
                        .iter()
                        .map(|r| {
                            (
                                r.control.steering_angle.to_bits(),
                                r.control.throttle_accel.to_bits(),
                            )
                        })
                        .collect(),
                );
                println!(
                    "GOG198 probe track={} modality={:?} outcome={:?} ticks={} stopped={} collisions={} non_advancing={} no_progress={} checkpoints={} hash={:#018x}",
                    name,
                    modality,
                    trace.outcome,
                    trace.records.len(),
                    trace.stopped_ticks(),
                    trace.collision_ticks(),
                    trace.non_advancing_ticks(),
                    trace.no_progress_ticks(),
                    trace.checkpoints_cleared(),
                    trace.trace_hash()
                );
                for r in trace.records.iter().step_by(10) {
                    println!("  {}", SelectionEpisodeTrace::format_line(r));
                }
                if let Some(last) = trace.records.last() {
                    println!("  last: {}", SelectionEpisodeTrace::format_line(last));
                }
            }
        }
        assert_eq!(hashes.len(), 4);
        // Modalidades distintas producen mediciones distintas (salvo degeneración total).
        assert_ne!(hashes[0], hashes[1]);
        // Observación medida: con recompensa constante la secuencia de control no depende de la
        // geometría del circuito en esta configuración (mismas acciones en ambos circuitos).
        assert_eq!(control_sequences[0], control_sequences[2]);
        assert_eq!(control_sequences[1], control_sequences[3]);
    }

    const MAX_TICKS: usize = 60;

    #[test]
    fn selection_principle_pure_probe_modality_is_persisted_in_scientific_trace() {
        use gogamen_core::PlanStatus;
        use gogamen_runtime::{
            DecisionAction, RecordedGuardDecision, ScientificRecord, SelectionMeasurementRecord,
            StreamSeeds, TickInputs,
        };

        let k = kin();
        let probe = SelectionProbeConfig::new(2, 4, 1.0, MasterSeed::new(DEV_SEEDS[0])).unwrap();
        let trace = run_pure_selection_probe(
            k,
            TrackMap::rectangular_circuit(),
            pure_controller_config(DEV_SEEDS[0]),
            probe,
            EntropyProxyKind::FutureTrajectoryProxy,
            5,
        )
        .unwrap();
        let r = trace.records.first().unwrap();

        // Mapeo 1:1 del registro del lab al registro científico durable.
        let measurement = SelectionMeasurementRecord {
            principle: r.principle,
            modality: Some(r.modality),
            horizon: r.horizon as u32,
            scale_m: r.scale_m,
            scale_label: r.scale_label.clone(),
            scale_version: r.scale_version,
            walkers: r.walkers as u64,
            viable_walkers: r.viable_walkers as u64,
            collisions: r.collisions as u64,
            entropy_bits: r.entropy_bits,
        };
        let tick = Tick::new(r.tick as u64);
        let obs = Observation::new(
            tick,
            SimDuration::from_secs_f64(0.05).unwrap(),
            vec![r.x, r.y, r.speed],
        )
        .unwrap();
        let action =
            Action::continuous(vec![r.control.steering_angle, r.control.throttle_accel]).unwrap();
        let record = ScientificRecord {
            revision: gogamen_core::StreamRevision::new(1),
            inputs: TickInputs::new(tick, obs),
            decision: DecisionAction {
                plan_status: PlanStatus::Completed,
                proposed_action: Some(action.clone()),
                guard_decision: RecordedGuardDecision::Allow(action.clone()),
                applied_action: Some(action),
                used_fallback: false,
            },
            stream_seeds: StreamSeeds::new(MasterSeed::new(DEV_SEEDS[0])),
            checkpoint: None,
            control_bits: None,
            selection_measurement: None,
        }
        .with_selection_measurement(Some(measurement));

        let decoded = ScientificRecord::from_bytes(&record.to_bytes().unwrap()).unwrap();
        let persisted = decoded
            .selection_measurement
            .as_ref()
            .expect("measurement persisted");
        assert_eq!(
            persisted.modality,
            Some(EntropyProxyKind::FutureTrajectoryProxy),
            "modality written into the scientific trace"
        );
        assert_eq!(persisted.scale_m, 1.0);
        assert_eq!(persisted.horizon, r.horizon as u32);
        assert_eq!(persisted.viable_walkers, r.viable_walkers as u64);
        assert_eq!(persisted.collisions, r.collisions as u64);
        assert_eq!(decoded, record);
    }

    #[test]
    fn selection_principle_pure_probe_rejects_non_pure_principle() {
        let k = kin();
        let track = TrackMap::rectangular_circuit();
        let cfg = KartFmcControllerConfig::with_dt_and_seed(k.dt, MasterSeed::new(7));
        let probe = SelectionProbeConfig::new(1, 2, 1.0, MasterSeed::new(7)).unwrap();
        assert!(matches!(
            run_pure_selection_probe(k, track, cfg, probe, EntropyProxyKind::FinalStateProxy, 10),
            Err(SelectionProbeError::PrincipleNotPure(
                SelectionPrinciple::Directed
            ))
        ));
    }

    #[test]
    fn selection_principle_probe_config_is_fail_closed() {
        let seed = MasterSeed::new(1);
        assert!(SelectionProbeConfig::new(0, 2, 1.0, seed).is_err());
        assert!(
            SelectionProbeConfig::new(MAX_PURE_PROBE_WALKERS_PER_ROOT + 1, 2, 1.0, seed).is_err()
        );
        assert!(SelectionProbeConfig::new(1, 0, 1.0, seed).is_err());
        assert!(SelectionProbeConfig::new(1, MAX_PURE_PROBE_HORIZON + 1, 1.0, seed).is_err());
        assert!(SelectionProbeConfig::new(1, 1, 0.0, seed).is_err());
        assert!(SelectionProbeConfig::new(1, 1, f64::NAN, seed).is_err());
        let cfg = SelectionProbeConfig::new(2, 3, 1.0, seed).unwrap();
        assert_eq!(cfg.walkers_per_tick(), PURE_PROBE_ROOTS * 2);
        let spec = cfg.bin_spec(&TrackMap::rectangular_circuit()).unwrap();
        assert_eq!(spec.dimension(), 2);
        assert_eq!(spec.label(), PURE_PROBE_SCALE_LABEL);
    }
}
