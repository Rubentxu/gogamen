//! Kart "entrópico puro" (GOG-156, EXPERIMENTAL, `std`-only + `gogamen-core`).
//!
//! Controlador de laboratorio que decide valorando únicamente la **viabilidad y la entropía de
//! futuros** simulados con la física existente ([`KartModel`]) y la entropía de Shannon de
//! `gogamen-core` ([`shannon_entropy_bits`]). La decisión usa solo:
//! - posición `(x, y)` del estado de partida y la dinámica del modelo;
//! - si el rollout termina en fallo (choque/salida de límites) o sobrevive todo el horizonte;
//! - la entropía (bits) de la distribución de celdas de 1 m alcanzadas por los futuros viables.
//!
//! No lee ponderaciones de objetivo, ni contadores de hitos de pista, ni progreso, ni velocidad
//! como recompensa. Este fichero está aislado a propósito: un test estático (`include_str!`)
//! verifica que no contiene los identificadores prohibidos, así que la puntuación es auditable.
//!
//! # Algoritmo por tick
//! 1. Candidatos raíz: retícula fija 3x3 `{-max, 0, +max}` dirección x `{min, 0, max}` aceleración.
//! 2. Por candidato, `population` rollouts: la acción raíz y luego acciones uniformes acotadas
//!    (RNG determinista derivado de la semilla) durante `horizon` pasos.
//! 3. `score = H_bits(celdas finales viables) + log2(1 + futuros_viables)`.
//! 4. Se elige el máximo (empate: primer candidato). Si ningún candidato tiene futuros viables,
//!    fail-closed: freno máximo recto y comportamiento `FailClosedBrake`.
//!
//! # Qué hace y qué NO demuestra
//! Es una demo acotada (pocos ticks, poblaciones pequeñas, dos circuitos, semillas fijas). Puede
//! detenerse, chocar o elegir no avanzar según la geometría y el horizonte; los comportamientos
//! observados se registran por tick en [`EntropicTickRecord`] y se resumen en el README del lab.
//! NO es una prueba general de inteligencia ni de emergencia: es un controlador de maximización de
//! opciones sobre un modelo cinemático conocido, con horizonte corto.
//!
//! # Observaciones medidas (población 12, horizonte 10, 60 ticks, 20 Hz; test `gog_156_*`)
//! - `rectangular-circuit`, semilla 7: 0 choques, desenlace `Timeout`, 23/60 ticks `Stopped`.
//! - `chicane-circuit`, semilla 7: 0 choques, desenlace `Timeout`, 35/60 ticks `Stopped`.
//! - El kart oscila con velocidad |v| < 0.4 m/s cerca del inicio: no recorre el circuito. Con
//!   rollouts de acciones uniformes (aceleración media negativa) el horizonte de 0.5 s no
//!   "ve" muros, luego no hay presión de evitación y la entropía de celdas apenas premia avanzar.
//!   Es decir, en esta configuración el controlador elige en la práctica no avanzar. No se
//!   afirma ninguna conducta emergente; solo que estas ejecuciones concretas se comportan así.

use std::collections::BTreeMap;

use gogamen_core::{
    Action, Actuator, DeterministicRng, DomainError, EnvironmentError, EnvironmentObserver,
    GenerativeModel, LiveEnvironment, MasterSeed, ModelError, RngSource, State, StepDt, StreamKey,
    Terminal, TerminalReason, shannon_entropy_bits,
};

use crate::physics::{KartControl, KartState, KinematicBicycleConfig};
use crate::track::TrackMap;
use crate::world::{KartEnv, KartModel};

/// Nombre de política registrado en cada tick.
pub const ENTROPIC_POLICY_NAME: &str = "entropic_pure_v1";
/// Límite duro de población por candidato (recursos acotados).
pub const MAX_ENTROPIC_POPULATION: usize = 256;
/// Límite duro de horizonte.
pub const MAX_ENTROPIC_HORIZON: usize = 64;
/// Lado de celda (m) para discretizar posiciones finales.
const CELL_SIZE_M: f64 = 1.0;
/// Umbral de velocidad (m/s) bajo el cual se etiqueta `Stopped`.
const STOPPED_SPEED: f64 = 0.1;

/// Configuración del controlador entrópico puro (sin pesos de objetivo).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntropicPureConfig {
    /// Rollouts por acción raíz candidata (`1..=MAX_ENTROPIC_POPULATION`).
    pub population: usize,
    /// Pasos por rollout (`1..=MAX_ENTROPIC_HORIZON`).
    pub horizon: usize,
    /// Semilla maestra.
    pub seed: MasterSeed,
}

impl EntropicPureConfig {
    /// Crea la configuración validando límites (fail-closed).
    pub fn new(population: usize, horizon: usize, seed: MasterSeed) -> Result<Self, DomainError> {
        if population == 0 || population > MAX_ENTROPIC_POPULATION {
            return Err(DomainError::InvalidTransition {
                reason: "entropic population must be in 1..=MAX_ENTROPIC_POPULATION",
            });
        }
        if horizon == 0 || horizon > MAX_ENTROPIC_HORIZON {
            return Err(DomainError::InvalidTransition {
                reason: "entropic horizon must be in 1..=MAX_ENTROPIC_HORIZON",
            });
        }
        Ok(Self {
            population,
            horizon,
            seed,
        })
    }
}

/// Comportamiento observado tras aplicar la decisión.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropicBehavior {
    /// Aceleración positiva elegida.
    Accelerate,
    /// Aceleración nula elegida (mantener).
    Coast,
    /// Aceleración negativa elegida con el kart en marcha.
    Brake,
    /// El kart está (casi) parado tras el tick, es decir, elige no avanzar.
    Stopped,
    /// Ningún candidato tenía futuros viables: freno máximo.
    FailClosedBrake,
}

/// Registro por tick del controlador.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicTickRecord {
    /// Índice de tick (0-based) de la decisión.
    pub tick: usize,
    /// Nombre de política.
    pub policy: &'static str,
    /// Población por candidato.
    pub population: usize,
    /// Horizonte.
    pub horizon: usize,
    /// Futuros viables del candidato elegido.
    pub viable_futures: usize,
    /// Futuros viables sumando todos los candidatos.
    pub viable_total: usize,
    /// Futuros evaluados en total (candidatos x población).
    pub futures_total: usize,
    /// Futuros que colisionan (todos los candidatos).
    pub colliding_futures: usize,
    /// Entropía (bits) del candidato elegido.
    pub entropy_bits: f64,
    /// Score del candidato elegido.
    pub score: f64,
    /// Control aplicado.
    pub control: KartControl,
    /// Comportamiento.
    pub behavior: EntropicBehavior,
    /// Posición y velocidad tras aplicar el control.
    pub x: f64,
    /// Coordenada y tras aplicar.
    pub y: f64,
    /// Velocidad tras aplicar.
    pub speed: f64,
}

/// Resultado de una decisión.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicDecision {
    /// Acción elegida.
    pub action: Action,
    /// Control acotado equivalente.
    pub control: KartControl,
    /// Futuros viables del candidato elegido.
    pub viable_futures: usize,
    /// Viables totales.
    pub viable_total: usize,
    /// Futuros totales evaluados.
    pub futures_total: usize,
    /// Colisiones totales.
    pub colliding_futures: usize,
    /// Entropía del elegido.
    pub entropy_bits: f64,
    /// Score del elegido.
    pub score: f64,
    /// `true` si se aplicó el fallback fail-closed.
    pub fail_closed: bool,
}

/// Desenlace de un episodio puro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropicOutcome {
    /// El kart chocó o salió de límites.
    Collided,
    /// El entorno terminó por meta (efecto externo; no se puntúa).
    EnvGoal,
    /// Se agotaron los ticks sin terminar.
    Timeout,
}

/// Traza de un episodio.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicEpisodeTrace {
    /// Nombre de la pista.
    pub track_name: String,
    /// Registros por tick.
    pub records: Vec<EntropicTickRecord>,
    /// Desenlace.
    pub outcome: EntropicOutcome,
}

impl EntropicEpisodeTrace {
    /// Hash FNV-1a 64 determinista de la traza (excluye cualquier medida de tiempo real).
    pub fn trace_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for r in &self.records {
            feed(r.tick as u64);
            feed(r.population as u64);
            feed(r.horizon as u64);
            feed(r.viable_futures as u64);
            feed(r.viable_total as u64);
            feed(r.colliding_futures as u64);
            feed(r.entropy_bits.to_bits());
            feed(r.score.to_bits());
            feed(r.control.steering_angle.to_bits());
            feed(r.control.throttle_accel.to_bits());
            feed(r.behavior as u64);
            feed(r.x.to_bits());
            feed(r.y.to_bits());
            feed(r.speed.to_bits());
        }
        feed(self.outcome as u64);
        h
    }

    /// Número de ticks etiquetados `Stopped`.
    pub fn stopped_ticks(&self) -> usize {
        self.records
            .iter()
            .filter(|r| r.behavior == EntropicBehavior::Stopped)
            .count()
    }

    /// Línea de texto por tick (política, población, horizonte, viables, colisiones, comportamiento).
    pub fn format_line(r: &EntropicTickRecord) -> String {
        format!(
            "t={} policy={} pop={} h={} viable={}/{} colliding={} H={:.3} behavior={:?} v={:.2}",
            r.tick,
            r.policy,
            r.population,
            r.horizon,
            r.viable_futures,
            r.futures_total,
            r.colliding_futures,
            r.entropy_bits,
            r.behavior,
            r.speed
        )
    }
}

/// Controlador entrópico puro.
#[derive(Debug, Clone)]
pub struct EntropicPureController {
    config: EntropicPureConfig,
    model: KartModel,
    dt: StepDt,
    rng: DeterministicRng,
}

impl EntropicPureController {
    /// Crea el controlador sobre la física y pista dadas.
    pub fn new(
        kinematics: KinematicBicycleConfig,
        track: TrackMap,
        config: EntropicPureConfig,
    ) -> Result<Self, DomainError> {
        let dt = kinematics.dt;
        let model = KartModel::new(kinematics, track, config.seed)?;
        let rng = config.seed.derive(StreamKey::Agent {
            episode: 0,
            agent: 0,
        });
        Ok(Self {
            config,
            model,
            dt,
            rng,
        })
    }

    /// Configuración.
    pub const fn config(&self) -> &EntropicPureConfig {
        &self.config
    }

    /// Decide una acción desde un estado explícito.
    pub fn decide_from_state(&mut self, state: &State) -> Result<EntropicDecision, ModelError> {
        let kin = self.model.config().clone();
        let space = self.model.action_space().clone();
        let mut best: Option<(f64, Action, KartControl, usize, f64)> = None;
        let (mut viable_total, mut futures_total, mut colliding) = (0usize, 0usize, 0usize);

        for steer in [-kin.max_steering_angle, 0.0, kin.max_steering_angle] {
            for accel in [kin.min_accel, 0.0, kin.max_accel] {
                let root = Action::continuous_in_space(vec![steer, accel], &space)
                    .map_err(ModelError::InvalidAction)?;
                let mut cells: BTreeMap<(i64, i64), usize> = BTreeMap::new();
                let mut viable = 0usize;
                for _ in 0..self.config.population {
                    futures_total += 1;
                    let mut s = state.clone();
                    let mut alive = true;
                    for step in 0..self.config.horizon {
                        let act = if step == 0 {
                            root.clone()
                        } else {
                            let st = (2.0 * self.rng.next_f64() - 1.0) * kin.max_steering_angle;
                            let ac = kin.min_accel
                                + self.rng.next_f64() * (kin.max_accel - kin.min_accel);
                            Action::continuous_in_space(vec![st, ac], &space)
                                .map_err(ModelError::InvalidAction)?
                        };
                        let tr = match GenerativeModel::step(
                            &self.model,
                            &s,
                            &act,
                            self.dt,
                            &mut self.rng,
                        ) {
                            Ok(tr) => tr,
                            Err(ModelError::TerminalState) => break,
                            Err(e) => return Err(e),
                        };
                        s = tr.next_state().clone();
                        match tr.terminal() {
                            Terminal::Terminated(TerminalReason::GoalReached)
                            | Terminal::Ongoing => {}
                            Terminal::Terminated(_) => {
                                alive = false;
                                break;
                            }
                        }
                        if tr.terminal().is_terminal() {
                            break;
                        }
                    }
                    if alive {
                        viable += 1;
                        let k = KartState::from_slice(&s.values()[0..4])
                            .map_err(ModelError::TransitionFailed)?;
                        let cell = (
                            (k.x / CELL_SIZE_M).floor() as i64,
                            (k.y / CELL_SIZE_M).floor() as i64,
                        );
                        *cells.entry(cell).or_insert(0) += 1;
                    } else {
                        colliding += 1;
                    }
                }
                viable_total += viable;
                if viable == 0 {
                    continue;
                }
                let probs: Vec<f64> = cells.values().map(|c| *c as f64 / viable as f64).collect();
                let h = shannon_entropy_bits(&probs).map_err(|_| ModelError::InvalidState {
                    reason: "entropy of viable futures failed",
                })?;
                let score = h + (1.0 + viable as f64).log2();
                let control = KartControl {
                    steering_angle: steer,
                    throttle_accel: accel,
                };
                if best.as_ref().is_none_or(|b| score > b.0) {
                    best = Some((score, root, control, viable, h));
                }
            }
        }

        match best {
            Some((score, action, control, viable, h)) => Ok(EntropicDecision {
                action,
                control,
                viable_futures: viable,
                viable_total,
                futures_total,
                colliding_futures: colliding,
                entropy_bits: h,
                score,
                fail_closed: false,
            }),
            None => {
                let action = Action::continuous_in_space(vec![0.0, kin.min_accel], &space)
                    .map_err(ModelError::InvalidAction)?;
                Ok(EntropicDecision {
                    action,
                    control: KartControl {
                        steering_angle: 0.0,
                        throttle_accel: kin.min_accel,
                    },
                    viable_futures: 0,
                    viable_total,
                    futures_total,
                    colliding_futures: colliding,
                    entropy_bits: 0.0,
                    score: 0.0,
                    fail_closed: true,
                })
            }
        }
    }

    /// Ejecuta un episodio acotado (`max_ticks`) registrando traza por tick.
    pub fn run_episode(
        &mut self,
        env: &mut KartEnv,
        max_ticks: usize,
    ) -> Result<EntropicEpisodeTrace, EntropicRunError> {
        LiveEnvironment::reset(env).map_err(EntropicRunError::Environment)?;
        let mut records = Vec::with_capacity(max_ticks);
        let mut outcome = EntropicOutcome::Timeout;
        for tick in 0..max_ticks {
            let obs = EnvironmentObserver::observe(env).map_err(EntropicRunError::Environment)?;
            let state = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
                .map_err(|e| EntropicRunError::Model(ModelError::TransitionFailed(e)))?;
            let d = self
                .decide_from_state(&state)
                .map_err(EntropicRunError::Model)?;
            let before_speed = env.kart_state().speed;
            let out = env
                .apply(&d.action)
                .map_err(EntropicRunError::Environment)?;
            let k = env.kart_state();
            let behavior = if d.fail_closed {
                EntropicBehavior::FailClosedBrake
            } else if k.speed.abs() < STOPPED_SPEED {
                EntropicBehavior::Stopped
            } else if d.control.throttle_accel > 0.0 {
                EntropicBehavior::Accelerate
            } else if d.control.throttle_accel < 0.0 && before_speed.abs() > STOPPED_SPEED {
                EntropicBehavior::Brake
            } else {
                EntropicBehavior::Coast
            };
            records.push(EntropicTickRecord {
                tick,
                policy: ENTROPIC_POLICY_NAME,
                population: self.config.population,
                horizon: self.config.horizon,
                viable_futures: d.viable_futures,
                viable_total: d.viable_total,
                futures_total: d.futures_total,
                colliding_futures: d.colliding_futures,
                entropy_bits: d.entropy_bits,
                score: d.score,
                control: d.control,
                behavior,
                x: k.x,
                y: k.y,
                speed: k.speed,
            });
            match out.terminal() {
                Terminal::Ongoing => {}
                Terminal::Terminated(TerminalReason::GoalReached) => {
                    outcome = EntropicOutcome::EnvGoal;
                    break;
                }
                Terminal::Terminated(_) => {
                    outcome = EntropicOutcome::Collided;
                    break;
                }
            }
        }
        Ok(EntropicEpisodeTrace {
            track_name: env.track().name().to_string(),
            records,
            outcome,
        })
    }
}

/// Errores de ejecución del episodio puro.
#[derive(Debug, Clone, PartialEq)]
pub enum EntropicRunError {
    /// Error del entorno.
    Environment(EnvironmentError),
    /// Error del modelo.
    Model(ModelError),
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = include_str!("entropic_pure.rs");

    fn run(track: TrackMap, seed: u64, ticks: usize) -> EntropicEpisodeTrace {
        let kin = KinematicBicycleConfig::default();
        let cfg = EntropicPureConfig::new(12, 10, MasterSeed::new(seed)).unwrap();
        let mut c = EntropicPureController::new(kin.clone(), track.clone(), cfg).unwrap();
        let mut env = KartEnv::new(kin, track).unwrap();
        c.run_episode(&mut env, ticks).unwrap()
    }

    #[test]
    fn gog_156_static_pure_module_has_no_forbidden_identifiers() {
        let prod = SRC.split("#[cfg(test)]").next().unwrap().to_lowercase();
        // `lap` se comprueba como token completo (evita falsos positivos como `elapsed`).
        assert!(
            !prod
                .split(|c: char| !c.is_alphanumeric())
                .any(|t| t == "lap" || t == "laps"),
            "forbidden identifier: lap"
        );
        for banned in [
            "kartobjectiveweights",
            "evaluate_kart_transition",
            "checkpoint",
            "progress",
            "forward_speed",
            "bonus",
            "weights",
        ] {
            assert!(!prod.contains(banned), "forbidden identifier: {banned}");
        }
    }

    #[test]
    fn gog_156_decision_independent_of_checkpoint_counter() {
        let kin = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let cfg = EntropicPureConfig::new(8, 8, MasterSeed::new(7)).unwrap();
        let env = KartEnv::new(kin.clone(), track.clone()).unwrap();
        let obs = EnvironmentObserver::observe(&env).unwrap();
        let mut vals = obs.features().to_vec();
        let s0 = State::new(obs.tick(), obs.elapsed(), vals.clone()).unwrap();
        vals[4] = 1.0;
        let s1 = State::new(obs.tick(), obs.elapsed(), vals).unwrap();
        let mut a = EntropicPureController::new(kin.clone(), track.clone(), cfg).unwrap();
        let mut b = EntropicPureController::new(kin, track, cfg).unwrap();
        let da = a.decide_from_state(&s0).unwrap();
        let db = b.decide_from_state(&s1).unwrap();
        assert_eq!(da.action, db.action);
        assert_eq!(da.score.to_bits(), db.score.to_bits());
    }

    #[test]
    fn gog_156_config_fail_closed() {
        assert!(EntropicPureConfig::new(0, 4, MasterSeed::new(1)).is_err());
        assert!(EntropicPureConfig::new(4, 0, MasterSeed::new(1)).is_err());
        assert!(
            EntropicPureConfig::new(MAX_ENTROPIC_POPULATION + 1, 4, MasterSeed::new(1)).is_err()
        );
        assert!(EntropicPureConfig::new(4, MAX_ENTROPIC_HORIZON + 1, MasterSeed::new(1)).is_err());
    }

    #[test]
    fn gog_156_two_circuits_per_tick_trace_reproducible() {
        for track in [TrackMap::rectangular_circuit(), TrackMap::chicane_circuit()] {
            let name = track.name().to_string();
            let a = run(track.clone(), 7, 60);
            let b = run(track.clone(), 7, 60);
            let c = run(track, 19, 60);
            assert_eq!(a, b, "same seed same trace on {name}");
            assert_eq!(a.trace_hash(), b.trace_hash());
            assert!(!a.records.is_empty());
            for r in &a.records {
                assert_eq!(r.policy, ENTROPIC_POLICY_NAME);
                assert_eq!(r.population, 12);
                assert_eq!(r.horizon, 10);
                assert_eq!(r.futures_total, 9 * 12);
                assert!(r.viable_total + r.colliding_futures <= r.futures_total);
            }
            println!(
                "GOG156 track={name} seed=7 ticks={} outcome={:?} stopped_ticks={} hash={:#x} | seed=19 outcome={:?} hash={:#x}",
                a.records.len(),
                a.outcome,
                a.stopped_ticks(),
                a.trace_hash(),
                c.outcome,
                c.trace_hash()
            );
            for r in a.records.iter().step_by(10) {
                println!("  {}", EntropicEpisodeTrace::format_line(r));
            }
            if let Some(last) = a.records.last() {
                println!("  last: {}", EntropicEpisodeTrace::format_line(last));
            }
        }
    }
}
