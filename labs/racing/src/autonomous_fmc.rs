//! Controlador autónomo de Kart con FMC sobre muestreo continuo (`std`-only + `gogamen-core`).
//!
//! Implementa:
//! - [`KartObjectiveWeights`]: ponderación de progreso de checkpoints/vueltas, integridad
//!   (penalización fuerte por cercanía y colisión con muros) y eficiencia energética (`throttle^2`).
//! - [`KartFmcController`]: adapta [`FmcPlanner`] al espacio de control continuo
//!   `[steering_angle, throttle_accel]` sobre [`KartModel`], tomando controles acotados por
//!   tick sin mutar [`KartEnv`] durante la planificación.
//! - [`SimpleTrackBaseline`]: controlador reactivo/geométrico hacia el siguiente checkpoint.
//! - [`KartAutonomousMetrics`]: métricas exportadas por episodio (`laps_completed`,
//!   `checkpoints_cleared`, `collisions`, `episode_ticks`, `total_control_energy`,
//!   `mean_transitions_per_decision`, `mean_decision_latency_ns` y estado de terminación limpia).
//!
//! # Circuitos y semillas de evaluación
//! - Circuito base de desarrollo: [`TrackMap::rectangular_circuit`].
//! - Circuito no visto de generalización: [`TrackMap::chicane_circuit`].
//! - Semillas de desarrollo: [`DEV_SEEDS`] (`[7, 19, 42]`).
//! - Semillas no vistas: [`UNSEEN_SEEDS`] (`[101, 202, 303]`).
//!
//! # Principio de selección (GOG-198)
//! [`KartFmcControllerConfig::principle`] decide cómo se construye la recompensa
//! ([`crate::KartRewardFunction`]). El valor por defecto es
//! [`SelectionPrinciple::Directed`], que reproduce la aritmética histórica de
//! [`evaluate_kart_transition`] sin cambios; [`SelectionPrinciple::Pure`] usa una recompensa
//! constante y [`SelectionPrinciple::Situated`] está declarado pero no implementado (fail-closed).
//! [`KartFmcController::run_trace`] exporta una traza por tick con hash determinista apta para
//! comparar conducta entre principios.

use std::time::Instant;

use gogamen_core::{
    Action, Actuator, DeterministicRng, DomainError, EnvironmentError, EnvironmentObserver,
    FmcPlanner, FmcPlannerConfig, FmcScoringConfig, FmcTraceSummary, GenerativeModel,
    LiveEnvironment, MasterSeed, ModelError, PlanBudget, PlanRequest, PlanStatus, PlanningContext,
    SelectionPrinciple, State, StepDt, StreamKey, Terminal, TerminalReason, Transition,
};

use crate::physics::{KartControl, KartState, KinematicBicycleConfig, wrap_angle};
use crate::selection_principle::KartRewardFunction;
use crate::track::{Point2, TrackMap};
use crate::world::{KartEnv, KartModel};

/// Semillas deterministas de desarrollo para ajuste y verificación base.
pub const DEV_SEEDS: [u64; 3] = [7, 19, 42];

/// Semillas deterministas no vistas para validación de generalización.
pub const UNSEEN_SEEDS: [u64; 3] = [101, 202, 303];

/// Pesos de la función objetivo multi-criterio del kart autónomo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartObjectiveWeights {
    /// Bonificación por cada checkpoint superado en orden.
    pub checkpoint_bonus: f64,
    /// Bonificación adicional al completar la vuelta (`GoalReached`).
    pub lap_goal_bonus: f64,
    /// Peso por metro de acercamiento al siguiente checkpoint.
    pub progress_weight: f64,
    /// Recompensa auxiliar por velocidad hacia adelante (penaliza marcha atrás).
    pub forward_speed_weight: f64,
    /// Penalización fuerte por colisión con muro o salida de límites.
    pub collision_penalty: f64,
    /// Peso de penalización por cercanía a muros dentro de `wall_safety_margin`.
    pub wall_proximity_weight: f64,
    /// Margen de seguridad libre respecto a muros (m) a partir del borde del kart.
    pub wall_safety_margin: f64,
    /// Peso de penalización energética sobre `throttle_accel^2`.
    pub energy_weight: f64,
}

impl Default for KartObjectiveWeights {
    fn default() -> Self {
        Self {
            checkpoint_bonus: 120.0,
            lap_goal_bonus: 300.0,
            progress_weight: 18.0,
            forward_speed_weight: 0.15,
            collision_penalty: 600.0,
            wall_proximity_weight: 40.0,
            wall_safety_margin: 1.6,
            energy_weight: 0.02,
        }
    }
}

impl KartObjectiveWeights {
    /// Valida que todos los pesos sean finitos y no negativos (y `wall_safety_margin > 0`).
    pub fn validate(&self) -> Result<(), DomainError> {
        let non_negative = [
            self.checkpoint_bonus,
            self.lap_goal_bonus,
            self.progress_weight,
            self.forward_speed_weight,
            self.collision_penalty,
            self.wall_proximity_weight,
            self.energy_weight,
        ];
        if non_negative.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(DomainError::InvalidTransition {
                reason: "kart objective weights must be finite and >= 0.0",
            });
        }
        if !self.wall_safety_margin.is_finite() || self.wall_safety_margin <= 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "wall_safety_margin must be finite and > 0.0",
            });
        }
        Ok(())
    }
}

/// Resultado terminal limpio de un episodio autónomo de kart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KartEpisodeOutcome {
    /// El kart completó el circuito objetivo sin colisiones.
    Completed,
    /// El kart falló limpiamente al colisionar contra un muro o límite.
    FailedCollision,
    /// El kart falló limpiamente al agotar el máximo de ticks permitidos.
    FailedTimeout,
    /// El planificador no autorizó ninguna acción válida y detuvo el episodio limpiamente.
    FailedNoAuthorizedAction,
}

impl KartEpisodeOutcome {
    /// Indica si el episodio completó la vuelta objetivo.
    pub const fn is_completed(self) -> bool {
        matches!(self, Self::Completed)
    }

    /// Indica si el episodio terminó limpiamente por fallo controlado.
    pub const fn is_clean_failure(self) -> bool {
        !self.is_completed()
    }
}

/// Métricas agregadas de un episodio autónomo de kart.
#[derive(Debug, Clone, PartialEq)]
pub struct KartAutonomousMetrics {
    /// Número de vueltas completadas (`1` si alcanzó `GoalReached`, `0` en otro caso).
    pub laps_completed: usize,
    /// Número de checkpoints superados en orden.
    pub checkpoints_cleared: usize,
    /// Número de colisiones registradas en el episodio (`0` o `1`).
    pub collisions: usize,
    /// Número total de ticks ejecutados en el entorno vivo.
    pub episode_ticks: usize,
    /// Energía de control acumulada (`sum(throttle_accel^2 * dt)`).
    pub total_control_energy: f64,
    /// Media de transiciones simuladas por decisión (`0.0` en baseline reactivo).
    pub mean_transitions_per_decision: f64,
    /// Latencia media de decisión en nanosegundos.
    pub mean_decision_latency_ns: f64,
    /// Total de transiciones simuladas acumuladas en todas las decisiones.
    pub total_transitions_used: usize,
    /// Duración física simulada del episodio en segundos.
    pub sim_elapsed_secs: f64,
    /// Resultado terminal limpio del episodio.
    pub outcome: KartEpisodeOutcome,
}

impl KartAutonomousMetrics {
    /// Indica si el episodio completó la vuelta objetivo sin colisiones.
    pub const fn completed(&self) -> bool {
        self.outcome.is_completed()
    }

    /// Indica si el episodio falló limpiamente (colisión, timeout o sin acción autorizada).
    pub const fn failed_cleanly(&self) -> bool {
        self.outcome.is_clean_failure()
    }
}

/// Registro por tick de una traza de episodio de kart.
///
/// No contiene latencias ni tiempo real: es determinista para semilla, configuración y principio
/// fijos y por eso alimenta [`KartEpisodeTrace::trace_hash`].
#[derive(Debug, Clone, PartialEq)]
pub struct KartTickRecord {
    /// Tick del entorno tras aplicar el control (1-based).
    pub tick: usize,
    /// Control continuo aplicado.
    pub control: KartControl,
    /// Posición X tras el tick.
    pub x: f64,
    /// Posición Y tras el tick.
    pub y: f64,
    /// Orientación (rad) tras el tick.
    pub yaw: f64,
    /// Velocidad (m/s) tras el tick.
    pub speed: f64,
    /// Checkpoints superados tras el tick.
    pub checkpoints_cleared: usize,
    /// Terminal del entorno tras el tick.
    pub terminal: Terminal,
}

/// Traza por tick determinista de un episodio autónomo de kart.
#[derive(Debug, Clone, PartialEq)]
pub struct KartEpisodeTrace {
    /// Principio de selección activo.
    pub principle: SelectionPrinciple,
    /// Nombre de la pista.
    pub track_name: String,
    /// Registros por tick aplicado.
    pub records: Vec<KartTickRecord>,
    /// Desenlace del episodio.
    pub outcome: KartEpisodeOutcome,
}

impl KartEpisodeTrace {
    /// Hash FNV-1a 64 determinista de la traza (excluye latencias y tiempo real).
    ///
    /// El terminal de cada registro codifica el desenlace, de modo que el hash cubre también el
    /// final del episodio sin duplicarlo.
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
            feed(r.control.steering_angle.to_bits());
            feed(r.control.throttle_accel.to_bits());
            feed(r.x.to_bits());
            feed(r.y.to_bits());
            feed(r.yaw.to_bits());
            feed(r.speed.to_bits());
            feed(r.checkpoints_cleared as u64);
            feed(match r.terminal {
                Terminal::Ongoing => 0,
                Terminal::Terminated(TerminalReason::GoalReached) => 1,
                Terminal::Terminated(_) => 2,
            });
        }
        h
    }
}

/// Configuración del controlador autónomo [`KartFmcController`].
#[derive(Debug, Clone, PartialEq)]
pub struct KartFmcControllerConfig {
    /// Configuración del planificador [`FmcPlanner`].
    pub planner: FmcPlannerConfig,
    /// Presupuesto máximo de transiciones simuladas por decisión.
    pub max_transitions_per_decision: usize,
    /// Pesos del objetivo multi-criterio (solo los consulta `Directed`).
    pub weights: KartObjectiveWeights,
    /// Principio que construye la recompensa FMC (`Directed` por defecto).
    pub principle: SelectionPrinciple,
}

impl KartFmcControllerConfig {
    /// Crea una configuración validando pesos y presupuesto.
    ///
    /// El principio por defecto es [`SelectionPrinciple::Directed`], que conserva la conducta
    /// histórica. Usa [`Self::with_principle`] para seleccionar otro principio.
    pub fn new(
        planner: FmcPlannerConfig,
        max_transitions_per_decision: usize,
        weights: KartObjectiveWeights,
    ) -> Result<Self, DomainError> {
        weights.validate()?;
        if max_transitions_per_decision == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "max_transitions_per_decision must be >= 1",
            });
        }
        Ok(Self {
            planner,
            max_transitions_per_decision,
            weights,
            principle: SelectionPrinciple::Directed,
        })
    }

    /// Asigna el principio de selección validándolo (fail-closed para `Situated`).
    pub fn with_principle(mut self, principle: SelectionPrinciple) -> Result<Self, DomainError> {
        principle.validate()?;
        self.principle = principle;
        Ok(self)
    }

    /// Construye la configuración por defecto parametrizada por `dt` y `seed`.
    pub fn with_dt_and_seed(dt: StepDt, seed: MasterSeed) -> Self {
        let population_size = 32;
        let horizon = 16;
        let scoring = FmcScoringConfig::default();
        let planner = FmcPlannerConfig::new(population_size, horizon, dt, scoring, seed)
            .expect("default kart FMC planner config is valid");
        Self {
            planner,
            max_transitions_per_decision: population_size * horizon,
            weights: KartObjectiveWeights::default(),
            principle: SelectionPrinciple::Directed,
        }
    }
}

impl Default for KartFmcControllerConfig {
    fn default() -> Self {
        let dt = KinematicBicycleConfig::default().dt;
        Self::with_dt_and_seed(dt, MasterSeed::new(DEV_SEEDS[0]))
    }
}

/// Informe de una decisión individual de [`KartFmcController`].
#[derive(Debug, Clone, PartialEq)]
pub struct KartDecisionReport {
    /// Control continuo acotado autorizado, si existe.
    pub control: Option<KartControl>,
    /// Acción continua validada en el espacio del modelo, si existe.
    pub action: Option<Action>,
    /// Estado devuelto por [`FmcPlanner`].
    pub plan_status: PlanStatus,
    /// Transiciones simuladas consumidas en esta decisión.
    pub transitions_used: usize,
    /// Latencia medida de la decisión en nanosegundos.
    pub decision_latency_ns: u64,
    /// Traza resumida del rollout FMC.
    pub trace: FmcTraceSummary,
}

/// Futuro candidato de una decisión: acción raíz mantenida durante el horizonte (ver
/// [`KartFmcController::candidate_futures`]).
#[derive(Debug, Clone, PartialEq)]
pub struct KartCandidateFuture {
    /// Acción raíz evaluada.
    pub action: Action,
    /// Control acotado equivalente, si es válido.
    pub control: Option<KartControl>,
    /// `true` si es la acción autorizada por el planificador.
    pub chosen: bool,
    /// Soporte `N(a_0)` según el planificador (0 si es el fallback).
    pub support_count: usize,
    /// Retorno acumulado medio de los walkers de la acción.
    pub mean_return: f64,
    /// Score de selección del planificador (0 si no es una acción raíz expuesta).
    pub score: f64,
    /// Retorno acumulado de mantener la acción durante el horizonte con el objetivo del lab.
    pub hold_return: f64,
    /// Posiciones `(x, y)` desde `start` (incluido) hasta el final del horizonte o la terminación.
    pub points: Vec<Point2>,
    /// Terminal alcanzado en el futuro (`Failure` = colisión).
    pub terminal: Terminal,
}

/// Controlador autónomo de Kart basado en [`FmcPlanner`] sobre muestreo continuo.
#[derive(Debug, Clone)]
pub struct KartFmcController {
    config: KartFmcControllerConfig,
    model: KartModel,
    planner: FmcPlanner,
    rng: DeterministicRng,
    fallback_action: Action,
}

impl KartFmcController {
    /// Crea un controlador FMC autónomo para la configuración física y pista dadas.
    pub fn new(
        kinematics: KinematicBicycleConfig,
        track: TrackMap,
        config: KartFmcControllerConfig,
    ) -> Result<Self, DomainError> {
        config.weights.validate()?;
        if config.max_transitions_per_decision == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "max_transitions_per_decision must be >= 1",
            });
        }
        let seed = config.planner.seed();
        let model = KartModel::new(kinematics.clone(), track, seed)?;
        let fallback_accel = (-2.0_f64).clamp(kinematics.min_accel, kinematics.max_accel);
        let fallback_action =
            Action::continuous_in_space(vec![0.0, fallback_accel], model.action_space())?;
        let planner = FmcPlanner::new(config.planner.clone());
        let rng = seed.derive(StreamKey::Agent {
            episode: 0,
            agent: 0,
        });

        Ok(Self {
            config,
            model,
            planner,
            rng,
            fallback_action,
        })
    }

    /// Devuelve la configuración del controlador.
    pub const fn config(&self) -> &KartFmcControllerConfig {
        &self.config
    }

    /// Devuelve el modelo generativo interno usado para planificar.
    pub const fn model(&self) -> &KartModel {
        &self.model
    }

    /// Cambia el presupuesto de transiciones simuladas por decisión (`>= 1`).
    ///
    /// Es el único parámetro de planificación mutable en caliente; se aplica desde la
    /// siguiente decisión.
    pub fn set_max_transitions_per_decision(&mut self, budget: usize) -> Result<(), DomainError> {
        if budget == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "max_transitions_per_decision must be >= 1",
            });
        }
        self.config.max_transitions_per_decision = budget;
        Ok(())
    }

    /// Futuros candidatos de una decisión ya tomada, para observabilidad (sin replanificar).
    ///
    /// Candidatos: las acciones raíz que el planificador expone en `report.trace` (tras el
    /// resampling FMC suele sobrevivir una sola), la acción autorizada (p. ej. el `fallback`) y
    /// una retícula fija de alternativas contrafactuales (`{-max, 0, +max}` de dirección x
    /// `{min, 0, max}` de aceleración). Cada candidato se mantiene durante el horizonte del
    /// planificador sobre [`KartModel`] (`GenerativeModel::step`, sin RNG ni mutación del
    /// controlador) desde `start` y se puntúa (`hold_return`) con el mismo objetivo que el
    /// planificador. No es una trayectoria de walker ni replanifica: las alternativas son
    /// *rechazadas* en el sentido de que no fueron la acción autorizada.
    pub fn candidate_futures(
        &self,
        start: &State,
        report: &KartDecisionReport,
    ) -> Result<Vec<KartCandidateFuture>, ModelError> {
        let horizon = self.config.planner.horizon();
        let dt = self.config.planner.dt();
        let kin = self.model.config();
        let reward_fn = KartRewardFunction::new(
            self.config.principle,
            self.model.track(),
            kin,
            &self.config.weights,
        )
        .map_err(|_| ModelError::InvalidState {
            reason: "unsupported selection principle for candidate futures",
        })?;
        let mut rng = MasterSeed::new(0).derive(StreamKey::Episode(0));
        let mut candidates: Vec<(Action, usize, f64, f64)> = report
            .trace
            .root_action_stats()
            .iter()
            .map(|s| (s.action.clone(), s.support_count, s.mean_return, s.score))
            .collect();
        if let Some(chosen) = &report.action
            && !candidates.iter().any(|c| &c.0 == chosen)
        {
            candidates.push((chosen.clone(), 0, 0.0, 0.0));
        }
        for steer in [-kin.max_steering_angle, 0.0, kin.max_steering_angle] {
            for accel in [kin.min_accel, 0.0, kin.max_accel] {
                let action =
                    Action::continuous_in_space(vec![steer, accel], self.model.action_space())
                        .map_err(ModelError::InvalidAction)?;
                if !candidates.iter().any(|c| c.0 == action) {
                    candidates.push((action, 0, 0.0, 0.0));
                }
            }
        }
        let mut out = Vec::with_capacity(candidates.len());
        for (action, support_count, mean_return, score) in candidates {
            let mut state = start.clone();
            let mut points = Vec::with_capacity(horizon + 1);
            let first = KartState::from_slice(&state.values()[0..4])
                .map_err(ModelError::TransitionFailed)?;
            points.push(Point2 {
                x: first.x,
                y: first.y,
            });
            let mut terminal = Terminal::Ongoing;
            let mut hold_return = 0.0;
            for _ in 0..horizon {
                let tr = match GenerativeModel::step(&self.model, &state, &action, dt, &mut rng) {
                    Ok(tr) => tr,
                    Err(ModelError::TerminalState) => break,
                    Err(e) => return Err(e),
                };
                hold_return += reward_fn.reward(&state, &action, &tr);
                state = tr.next_state().clone();
                let k = KartState::from_slice(&state.values()[0..4])
                    .map_err(ModelError::TransitionFailed)?;
                points.push(Point2 { x: k.x, y: k.y });
                terminal = tr.terminal();
                if terminal.is_terminal() {
                    break;
                }
            }
            out.push(KartCandidateFuture {
                chosen: report.action.as_ref() == Some(&action),
                control: extract_bounded_control(&action, kin),
                action,
                support_count,
                mean_return,
                score,
                hold_return,
                points,
                terminal,
            });
        }
        Ok(out)
    }

    /// Planifica una acción acotada observando `env` de forma estrictamente inmutable.
    ///
    /// Nunca muta [`KartEnv`] durante la planificación.
    pub fn decide_from_env(&mut self, env: &KartEnv) -> Result<KartDecisionReport, ModelError> {
        let obs = EnvironmentObserver::observe(env).map_err(|_| ModelError::InvalidState {
            reason: "failed to observe KartEnv for planning",
        })?;
        let state = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
            .map_err(ModelError::TransitionFailed)?;
        self.decide_from_state(&state)
    }

    /// Planifica una acción acotada a partir de un [`State`] explícito usando [`KartModel`].
    pub fn decide_from_state(&mut self, state: &State) -> Result<KartDecisionReport, ModelError> {
        let start = Instant::now();
        let request = PlanRequest::from_state(
            state.clone(),
            self.model.action_space().clone(),
            PlanBudget::with_transitions(self.config.max_transitions_per_decision),
        )
        .with_fallback(self.fallback_action.clone())
        .map_err(ModelError::InvalidAction)?;

        let track = self.model.track().clone();
        let kinematics = self.model.config().clone();
        let weights = self.config.weights;
        let reward_fn =
            KartRewardFunction::new(self.config.principle, &track, &kinematics, &weights).map_err(
                |_| ModelError::InvalidState {
                    reason: "unsupported selection principle for planning",
                },
            )?;

        let mut ctx = PlanningContext::new(&self.model, &mut self.rng);
        let (result, trace) = self.planner.plan_with_trace(
            &mut ctx,
            &request,
            move |prev_state, action, transition| reward_fn.reward(prev_state, action, transition),
            kart_phi_projection,
        )?;

        let elapsed_ns = start.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let authorized = result.authorized_action().cloned();
        let control = authorized
            .as_ref()
            .and_then(|act| extract_bounded_control(act, self.model.config()));

        Ok(KartDecisionReport {
            control,
            action: authorized,
            plan_status: result.status(),
            transitions_used: result.transitions_used(),
            decision_latency_ns: elapsed_ns,
            trace,
        })
    }

    /// Ejecuta un episodio completo en `env` hasta completar, colisionar o alcanzar `max_ticks`.
    pub fn run_episode(
        &mut self,
        env: &mut KartEnv,
        max_ticks: usize,
    ) -> Result<KartAutonomousMetrics, KartRunError> {
        LiveEnvironment::reset(env).map_err(KartRunError::Environment)?;
        let dt_secs = env.config().dt.as_secs_f64();

        let mut total_energy = 0.0_f64;
        let mut total_transitions = 0usize;
        let mut total_latency_ns = 0u128;
        let mut decisions_made = 0usize;
        let mut outcome = KartEpisodeOutcome::FailedTimeout;

        for _ in 0..max_ticks {
            let report = self.decide_from_env(env).map_err(KartRunError::Model)?;
            total_transitions += report.transitions_used;
            total_latency_ns += report.decision_latency_ns as u128;
            decisions_made += 1;

            let (Some(action), Some(control)) = (report.action, report.control) else {
                outcome = KartEpisodeOutcome::FailedNoAuthorizedAction;
                break;
            };

            let apply_out = env.apply(&action).map_err(KartRunError::Environment)?;
            total_energy += control.throttle_accel * control.throttle_accel * dt_secs;

            match apply_out.terminal() {
                Terminal::Ongoing => {}
                Terminal::Terminated(TerminalReason::GoalReached) => {
                    outcome = KartEpisodeOutcome::Completed;
                    break;
                }
                Terminal::Terminated(_) => {
                    outcome = KartEpisodeOutcome::FailedCollision;
                    break;
                }
            }
        }

        let laps_completed = usize::from(outcome.is_completed());
        let collisions = usize::from(env.collided());
        let mean_transitions = if decisions_made == 0 {
            0.0
        } else {
            total_transitions as f64 / decisions_made as f64
        };
        let mean_latency_ns = if decisions_made == 0 {
            0.0
        } else {
            total_latency_ns as f64 / decisions_made as f64
        };

        Ok(KartAutonomousMetrics {
            laps_completed,
            checkpoints_cleared: env.checkpoints_cleared(),
            collisions,
            episode_ticks: env.tick().as_u64() as usize,
            total_control_energy: total_energy,
            mean_transitions_per_decision: mean_transitions,
            mean_decision_latency_ns: mean_latency_ns,
            total_transitions_used: total_transitions,
            sim_elapsed_secs: env.elapsed().as_secs_f64(),
            outcome,
        })
    }

    /// Ejecuta un episodio en `env` registrando una traza determinista por tick.
    ///
    /// Comparte la semántica de terminación de [`Self::run_episode`] y no reutiliza latencias ni
    /// tiempo real, de modo que [`KartEpisodeTrace::trace_hash`] es comparable entre principios.
    pub fn run_trace(
        &mut self,
        env: &mut KartEnv,
        max_ticks: usize,
    ) -> Result<KartEpisodeTrace, KartRunError> {
        LiveEnvironment::reset(env).map_err(KartRunError::Environment)?;
        let mut records = Vec::with_capacity(max_ticks);
        let mut outcome = KartEpisodeOutcome::FailedTimeout;

        for _ in 0..max_ticks {
            let report = self.decide_from_env(env).map_err(KartRunError::Model)?;
            let (Some(action), Some(control)) = (report.action, report.control) else {
                outcome = KartEpisodeOutcome::FailedNoAuthorizedAction;
                break;
            };

            let apply_out = env.apply(&action).map_err(KartRunError::Environment)?;
            let kart = env.kart_state();
            let terminal = apply_out.terminal();
            records.push(KartTickRecord {
                tick: env.tick().as_u64() as usize,
                control,
                x: kart.x,
                y: kart.y,
                yaw: kart.yaw,
                speed: kart.speed,
                checkpoints_cleared: env.checkpoints_cleared(),
                terminal,
            });

            match terminal {
                Terminal::Ongoing => {}
                Terminal::Terminated(TerminalReason::GoalReached) => {
                    outcome = KartEpisodeOutcome::Completed;
                    break;
                }
                Terminal::Terminated(_) => {
                    outcome = KartEpisodeOutcome::FailedCollision;
                    break;
                }
            }
        }

        Ok(KartEpisodeTrace {
            principle: self.config.principle,
            track_name: env.track().name().to_owned(),
            records,
            outcome,
        })
    }
}

/// Errores explícitos de ejecución de un episodio autónomo de kart.
#[derive(Debug, Clone, PartialEq)]
pub enum KartRunError {
    /// Error del entorno vivo [`KartEnv`].
    Environment(EnvironmentError),
    /// Error del modelo generativo o del planificador.
    Model(ModelError),
}

/// Parámetros del controlador reactivo/geométrico [`SimpleTrackBaseline`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SimpleTrackBaselineConfig {
    /// Ganancia proporcional sobre el error de orientación hacia el checkpoint.
    pub steering_gain: f64,
    /// Ganancia proporcional sobre el error de velocidad.
    pub speed_gain: f64,
    /// Velocidad objetivo en recta (m/s).
    pub target_speed: f64,
    /// Reducción de velocidad proporcional al error angular en curvas.
    pub corner_slowdown_gain: f64,
}

impl Default for SimpleTrackBaselineConfig {
    fn default() -> Self {
        Self {
            steering_gain: 2.2,
            speed_gain: 2.0,
            target_speed: 6.5,
            corner_slowdown_gain: 1.5,
        }
    }
}

/// Baseline simple de seguimiento de pista (controlador reactivo/geométrico al siguiente checkpoint).
#[derive(Debug, Clone)]
pub struct SimpleTrackBaseline {
    kinematics: KinematicBicycleConfig,
    track: TrackMap,
    config: SimpleTrackBaselineConfig,
}

impl SimpleTrackBaseline {
    /// Construye un baseline reactivo validando configuración física y parámetros.
    pub fn new(
        kinematics: KinematicBicycleConfig,
        track: TrackMap,
        config: SimpleTrackBaselineConfig,
    ) -> Result<Self, DomainError> {
        kinematics.validate()?;
        if !config.steering_gain.is_finite()
            || !config.speed_gain.is_finite()
            || !config.target_speed.is_finite()
            || !config.corner_slowdown_gain.is_finite()
            || config.target_speed <= 0.0
        {
            return Err(DomainError::InvalidTransition {
                reason: "simple baseline parameters must be finite with target_speed > 0",
            });
        }
        Ok(Self {
            kinematics,
            track,
            config,
        })
    }

    /// Calcula el control continuo acotado hacia el siguiente checkpoint activo.
    pub fn decide(&self, kart: KartState, checkpoints_cleared: usize) -> KartControl {
        let total_cps = self.track.checkpoints().len();
        let cp_idx = checkpoints_cleared.min(total_cps.saturating_sub(1));
        let target = self.track.checkpoints()[cp_idx].center;

        let dx = target.x - kart.x;
        let dy = target.y - kart.y;
        let desired_yaw = dy.atan2(dx);
        let heading_err = wrap_angle(desired_yaw - kart.yaw);

        let steering_angle = (self.config.steering_gain * heading_err).clamp(
            -self.kinematics.max_steering_angle,
            self.kinematics.max_steering_angle,
        );
        let desired_speed =
            self.config.target_speed / (1.0 + self.config.corner_slowdown_gain * heading_err.abs());
        let throttle_accel = (self.config.speed_gain * (desired_speed - kart.speed))
            .clamp(self.kinematics.min_accel, self.kinematics.max_accel);

        KartControl {
            steering_angle,
            throttle_accel,
        }
    }

    /// Ejecuta un episodio completo del baseline reactivo sobre `env`.
    pub fn run_episode(
        &self,
        env: &mut KartEnv,
        max_ticks: usize,
    ) -> Result<KartAutonomousMetrics, EnvironmentError> {
        LiveEnvironment::reset(env)?;
        let dt_secs = env.config().dt.as_secs_f64();
        let mut total_energy = 0.0_f64;
        let mut total_latency_ns = 0u128;
        let mut decisions_made = 0usize;
        let mut outcome = KartEpisodeOutcome::FailedTimeout;

        for _ in 0..max_ticks {
            let start = Instant::now();
            let control = self.decide(env.kart_state(), env.checkpoints_cleared());
            let action = Action::continuous_in_space(
                vec![control.steering_angle, control.throttle_accel],
                Actuator::action_space(env),
            )
            .map_err(EnvironmentError::InvalidAction)?;
            total_latency_ns += start.elapsed().as_nanos();
            decisions_made += 1;

            let apply_out = env.apply(&action)?;
            total_energy += control.throttle_accel * control.throttle_accel * dt_secs;

            match apply_out.terminal() {
                Terminal::Ongoing => {}
                Terminal::Terminated(TerminalReason::GoalReached) => {
                    outcome = KartEpisodeOutcome::Completed;
                    break;
                }
                Terminal::Terminated(_) => {
                    outcome = KartEpisodeOutcome::FailedCollision;
                    break;
                }
            }
        }

        let mean_latency_ns = if decisions_made == 0 {
            0.0
        } else {
            total_latency_ns as f64 / decisions_made as f64
        };

        Ok(KartAutonomousMetrics {
            laps_completed: usize::from(outcome.is_completed()),
            checkpoints_cleared: env.checkpoints_cleared(),
            collisions: usize::from(env.collided()),
            episode_ticks: env.tick().as_u64() as usize,
            total_control_energy: total_energy,
            mean_transitions_per_decision: 0.0,
            mean_decision_latency_ns: mean_latency_ns,
            total_transitions_used: 0,
            sim_elapsed_secs: env.elapsed().as_secs_f64(),
            outcome,
        })
    }
}

fn extract_bounded_control(
    action: &Action,
    config: &KinematicBicycleConfig,
) -> Option<KartControl> {
    let vals = action.as_continuous()?;
    if vals.len() != 2 {
        return None;
    }
    let steering = vals[0];
    let throttle = vals[1];
    if !steering.is_finite() || !throttle.is_finite() {
        return None;
    }
    if steering < -config.max_steering_angle - 1e-12
        || steering > config.max_steering_angle + 1e-12
        || throttle < config.min_accel - 1e-12
        || throttle > config.max_accel + 1e-12
    {
        return None;
    }
    Some(KartControl {
        steering_angle: steering.clamp(-config.max_steering_angle, config.max_steering_angle),
        throttle_accel: throttle.clamp(config.min_accel, config.max_accel),
    })
}

fn min_wall_clearance(track: &TrackMap, pt: Point2, kart_radius: f64) -> f64 {
    let dx_min = (pt.x - track.bounds_min().x).max(0.0);
    let dx_max = (track.bounds_max().x - pt.x).max(0.0);
    let dy_min = (pt.y - track.bounds_min().y).max(0.0);
    let dy_max = (track.bounds_max().y - pt.y).max(0.0);
    let bounds_dist = dx_min.min(dx_max).min(dy_min).min(dy_max);

    let wall_dist = track
        .walls()
        .iter()
        .map(|w| w.distance_to_point(pt))
        .fold(f64::INFINITY, f64::min);

    bounds_dist.min(wall_dist) - kart_radius
}

fn kart_phi_projection(state: &State) -> Vec<f64> {
    let v = state.values();
    if v.len() < 5 {
        return v.to_vec();
    }
    let yaw = v[2];
    vec![
        v[0],
        v[1],
        2.0 * yaw.cos(),
        2.0 * yaw.sin(),
        v[3],
        8.0 * v[4],
    ]
}

/// Aritmética histórica de la recompensa `Directed` (sin cambios desde antes de GOG-198).
pub(crate) fn evaluate_kart_transition(
    track: &TrackMap,
    config: &KinematicBicycleConfig,
    weights: &KartObjectiveWeights,
    prev_state: &State,
    action: &Action,
    transition: &Transition,
) -> f64 {
    let prev = prev_state.values();
    let next = transition.next_state().values();
    if prev.len() < 5 || next.len() < 5 {
        return 0.0;
    }

    let prev_pt = Point2 {
        x: prev[0],
        y: prev[1],
    };
    let next_pt = Point2 {
        x: next[0],
        y: next[1],
    };
    let prev_cp = prev[4].round().max(0.0) as usize;
    let next_cp = next[4].round().max(0.0) as usize;

    let total_cps = track.checkpoints().len();
    let target_cp = prev_cp.min(total_cps.saturating_sub(1));
    let target = track.checkpoints()[target_cp].center;

    let prev_dist = prev_pt.distance_to(target);
    let next_dist = next_pt.distance_to(target);
    let progress = prev_dist - next_dist;

    let checkpoint_reward = if next_cp > prev_cp {
        weights.checkpoint_bonus
    } else {
        0.0
    };
    let goal_reward = match transition.terminal() {
        Terminal::Terminated(TerminalReason::GoalReached) => weights.lap_goal_bonus,
        _ => 0.0,
    };
    let collision_penalty = match transition.terminal() {
        Terminal::Terminated(TerminalReason::Failure | TerminalReason::InvariantViolated) => {
            weights.collision_penalty
        }
        _ => 0.0,
    };

    let clearance = min_wall_clearance(track, next_pt, config.kart_radius);
    let wall_penalty = if clearance < weights.wall_safety_margin {
        let ratio = ((weights.wall_safety_margin - clearance.max(0.0))
            / weights.wall_safety_margin)
            .clamp(0.0, 1.0);
        weights.wall_proximity_weight * ratio * ratio
    } else {
        0.0
    };

    let next_speed = next[3];
    let speed_term = if next_speed >= 0.0 {
        weights.forward_speed_weight * next_speed.min(8.0)
    } else {
        -weights.forward_speed_weight * 2.5 * (-next_speed)
    };

    let throttle = action
        .as_continuous()
        .and_then(|vals| vals.get(1).copied())
        .unwrap_or(0.0);
    let energy_penalty = weights.energy_weight * throttle * throttle;

    weights.progress_weight * progress + checkpoint_reward + goal_reward + speed_term
        - collision_penalty
        - wall_penalty
        - energy_penalty
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_control_within_bounds(control: KartControl, cfg: &KinematicBicycleConfig) {
        assert!(control.steering_angle.is_finite());
        assert!(control.throttle_accel.is_finite());
        assert!(control.steering_angle >= -cfg.max_steering_angle - 1e-12);
        assert!(control.steering_angle <= cfg.max_steering_angle + 1e-12);
        assert!(control.throttle_accel >= cfg.min_accel - 1e-12);
        assert!(control.throttle_accel <= cfg.max_accel + 1e-12);
    }

    #[test]
    fn ac1_fmc_takes_bounded_controls_without_mutating_env_and_terminates_cleanly() {
        let kinematics = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(kinematics.clone(), track.clone()).unwrap();

        let ctrl_cfg =
            KartFmcControllerConfig::with_dt_and_seed(kinematics.dt, MasterSeed::new(DEV_SEEDS[0]));
        let mut controller =
            KartFmcController::new(kinematics.clone(), track.clone(), ctrl_cfg).unwrap();

        // Verifica que planificar desde KartEnv no muta tick, estado ni checkpoints de KartEnv.
        let tick_before = env.tick();
        let state_before = env.kart_state();
        let cp_before = env.checkpoints_cleared();
        let report = controller.decide_from_env(&env).unwrap();
        assert_eq!(env.tick(), tick_before);
        assert_eq!(env.kart_state(), state_before);
        assert_eq!(env.checkpoints_cleared(), cp_before);

        let control = report
            .control
            .expect("FMC should authorize a bounded control");
        assert_control_within_bounds(control, &kinematics);
        assert!(report.transitions_used > 0);
        assert!(report.transitions_used <= controller.config().max_transitions_per_decision);

        // Ejecuta episodio completo en circuito base: debe completar la vuelta o fallar limpiamente.
        let metrics = controller.run_episode(&mut env, 900).unwrap();
        assert!(metrics.completed() || metrics.failed_cleanly());
        assert!(
            metrics.completed(),
            "expected FMC to complete base circuit, got {metrics:?}"
        );
        assert_eq!(metrics.laps_completed, 1);
        assert_eq!(metrics.checkpoints_cleared, track.checkpoints().len());
        assert_eq!(metrics.collisions, 0);
        assert!(metrics.episode_ticks > 0);
        assert!(metrics.total_control_energy.is_finite() && metrics.total_control_energy > 0.0);
        assert!(metrics.mean_transitions_per_decision > 0.0);
        assert!(metrics.mean_decision_latency_ns >= 0.0);
    }

    #[test]
    fn ac2_simple_baseline_and_unseen_circuits_and_seeds_evaluated() {
        let kinematics = KinematicBicycleConfig::default();

        // 1. Baseline simple en circuito base y circuito no visto (chicane).
        for track in [TrackMap::rectangular_circuit(), TrackMap::chicane_circuit()] {
            let mut env = KartEnv::new(kinematics.clone(), track.clone()).unwrap();
            let baseline = SimpleTrackBaseline::new(
                kinematics.clone(),
                track.clone(),
                SimpleTrackBaselineConfig::default(),
            )
            .unwrap();
            let metrics = baseline.run_episode(&mut env, 1200).unwrap();
            assert!(
                metrics.completed(),
                "baseline failed on {}: {metrics:?}",
                track.name()
            );
            assert_eq!(metrics.laps_completed, 1);
            assert_eq!(metrics.collisions, 0);
            assert_eq!(metrics.mean_transitions_per_decision, 0.0);
        }

        // 2. FMC con seeds de desarrollo y seeds no vistas en circuito base y circuito no visto.
        let mut completed_unseen = 0usize;
        for &seed in DEV_SEEDS.iter().chain(UNSEEN_SEEDS.iter()) {
            let track = TrackMap::chicane_circuit();
            let mut env = KartEnv::new(kinematics.clone(), track.clone()).unwrap();
            let cfg =
                KartFmcControllerConfig::with_dt_and_seed(kinematics.dt, MasterSeed::new(seed));
            let mut controller = KartFmcController::new(kinematics.clone(), track, cfg).unwrap();
            let metrics = controller.run_episode(&mut env, 1200).unwrap();

            // Toda ejecución debe terminar limpiamente (completar o fallar de forma controlada)
            // y exportar métricas válidas.
            assert!(metrics.completed() || metrics.failed_cleanly());
            assert!(metrics.episode_ticks > 0);
            assert!(metrics.mean_transitions_per_decision > 0.0);
            if metrics.completed() {
                completed_unseen += 1;
            }
        }
        assert!(
            completed_unseen >= 1,
            "expected at least one seed to complete unseen chicane circuit"
        );
    }

    #[test]
    fn ac3_metrics_export_laps_collisions_time_and_decision_cost() {
        let kinematics = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(kinematics.clone(), track.clone()).unwrap();

        // Caso 1: Timeout limpio (max_ticks pequeño).
        let cfg = KartFmcControllerConfig::with_dt_and_seed(
            kinematics.dt,
            MasterSeed::new(UNSEEN_SEEDS[0]),
        );
        let mut controller = KartFmcController::new(kinematics.clone(), track, cfg).unwrap();
        let short_metrics = controller.run_episode(&mut env, 15).unwrap();

        assert_eq!(short_metrics.outcome, KartEpisodeOutcome::FailedTimeout);
        assert!(short_metrics.failed_cleanly());
        assert_eq!(short_metrics.laps_completed, 0);
        assert_eq!(short_metrics.episode_ticks, 15);
        assert_eq!(short_metrics.collisions, 0);
        assert!(short_metrics.sim_elapsed_secs > 0.0);
        assert!(short_metrics.total_transitions_used > 0);
        assert!(short_metrics.mean_transitions_per_decision > 0.0);
        assert!(short_metrics.mean_decision_latency_ns >= 0.0);
        assert!(short_metrics.total_control_energy >= 0.0);
    }

    #[test]
    fn candidate_futures_expose_chosen_and_rejected_without_mutating_controller() {
        let kin = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let env = KartEnv::new(kin.clone(), track.clone()).unwrap();
        let cfg = KartFmcControllerConfig::with_dt_and_seed(kin.dt, MasterSeed::new(7));
        let mut ctl = KartFmcController::new(kin, track, cfg).unwrap();
        let report = ctl.decide_from_env(&env).unwrap();
        let obs = EnvironmentObserver::observe(&env).unwrap();
        let start = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec()).unwrap();
        let before = ctl.clone();
        let fut = ctl.candidate_futures(&start, &report).unwrap();
        assert_eq!(ctl.config(), before.config());
        assert!(fut.len() >= 2 && fut.iter().any(|f| !f.chosen));
        assert_eq!(fut.iter().filter(|f| f.chosen).count(), 1);
        let chosen = fut.iter().find(|f| f.chosen).unwrap();
        assert_eq!(Some(&chosen.action), report.action.as_ref());
        assert!(fut.iter().all(|f| f.points.len() >= 2));
        let p0 = fut[0].points[0];
        assert!(fut.iter().all(|f| f.points[0] == p0));
        assert_eq!(fut, ctl.candidate_futures(&start, &report).unwrap());
    }

    #[test]
    fn budget_setter_rejects_zero_and_applies() {
        let kin = KinematicBicycleConfig::default();
        let cfg = KartFmcControllerConfig::with_dt_and_seed(kin.dt, MasterSeed::new(7));
        let mut ctl = KartFmcController::new(kin, TrackMap::rectangular_circuit(), cfg).unwrap();
        assert!(ctl.set_max_transitions_per_decision(0).is_err());
        ctl.set_max_transitions_per_decision(64).unwrap();
        assert_eq!(ctl.config().max_transitions_per_decision, 64);
    }
}
