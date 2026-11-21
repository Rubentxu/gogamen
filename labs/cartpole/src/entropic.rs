//! Experimento entrópico de péndulo invertido (CartPole) sin reward angular de upright.
//!
//! # Alcance epistemológico
//! - [`EntropicProxyController`] selecciona acciones maximizando un proxy empírico de
//!   entropía de trayectorias futuras viables ([`estimate_future_trajectory_entropy_proxy`])
//!   mediante bifurcación determinista de [`CartPoleModel`] (`BranchableSnapshot`).
//! - **No** utiliza ninguna función de recompensa angular (`theta`, `cos(theta)`, etc.):
//!   únicamente excluye trayectorias que alcanzan un fallo terminal físico (`|x| > x_threshold`
//!   o `|theta| > theta_threshold_radians`) y mide la diversidad discretizada por bins de los
//!   futuros viables.
//! - [`EntropicComparisonReport`] incluye una nota epistemológica explícita declarando que
//!   este experimento **no afirma replicación literal de experimentos PRL de fuerzas entrópicas
//!   sin concordancia cuantitativa demostrada**, y distingue la adaptación por muestreo de
//!   futuros de una reproducción física literal.

use std::fmt;

use gogamen_core::{
    Action, ActionSpace, Actuator, BinSpec, BinningVersion, BranchableSnapshot,
    DeterminismMetadata, DeterministicRng, DomainError, EnvironmentError, EnvironmentObserver,
    MasterSeed, Observation, RngSource, SimulationSnapshot, State, StreamKey, TraceHasher,
    TrajectoryStep, TrajectorySummary, estimate_future_trajectory_entropy_proxy,
};

use crate::controller::{CartPolePolicy, LinearFeedbackController};
use crate::physics::{
    CartPoleConfig, CartPoleKinematics, DEFAULT_THETA_THRESHOLD_RADIANS, DEFAULT_X_THRESHOLD,
};
use crate::world::{CartPoleEnv, CartPoleModel};

/// Nota epistemológica canónica incluida en todo [`EntropicComparisonReport`].
pub const EPISTEMOLOGICAL_NOTE: &str = "Nota epistemológica: este experimento no afirma replicación literal de experimentos PRL de fuerzas entrópicas sin concordancia cuantitativa demostrada. El controlador entrópico implementa una adaptación heurística por muestreo finito de trayectorias futuras viables y estimación empírica discretizada por bins (FutureTrajectoryProxy), lo cual es distinto de una reproducción física literal de fuerzas entrópicas continuas.";

/// Configuración explícita del experimento entrópico de péndulo invertido.
///
/// Documenta estado inicial, horizonte de planificación, esquema de bins, acciones candidatas,
/// temperatura opcional y la ausencia explícita de reward angular de upright.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicExperimentConfig {
    /// Estado cinemático inicial `[x, x_dot, theta, theta_dot]` del experimento.
    pub initial_state: CartPoleKinematics,
    /// Parámetros físicos y límites terminales de CartPole.
    pub cartpole: CartPoleConfig,
    /// Horizonte `H` en pasos discretos para cada rollout futuro (`>= 1`).
    pub horizon: usize,
    /// Número de rollouts futuros muestreados por acción candidata en cada paso (`>= 1`).
    pub rollouts_per_step: usize,
    /// Acciones candidatas (fuerzas en Newtons) evaluadas en cada decisión.
    pub candidate_actions: Vec<f64>,
    /// Especificación de bins sobre `(x, theta)` (dimensión 2) o `(x, x_dot, theta, theta_dot)` (dimensión 4).
    pub bin_spec: BinSpec,
    /// Temperatura opcional para muestreo tipo Boltzmann (`None` = selección greedy argmax).
    pub temperature: Option<f64>,
    /// Debe ser `false`: prohíbe explícitamente cualquier reward angular de upright.
    pub has_angular_upright_reward: bool,
    /// Número máximo de ticks por episodio en [`run_entropic_comparison`] (`>= 1`).
    pub episode_ticks: u64,
}

impl Default for EntropicExperimentConfig {
    fn default() -> Self {
        let initial_state = CartPoleKinematics::new(0.0, 0.0, 0.05, 0.0)
            .expect("default entropic initial state must be valid");
        let cartpole = CartPoleConfig::default().with_initial_state(initial_state);
        let bin_spec = BinSpec::uniform(
            "cartpole-x-theta-v1",
            BinningVersion::V1_0,
            vec![
                (-DEFAULT_X_THRESHOLD, DEFAULT_X_THRESHOLD, 12),
                (
                    -DEFAULT_THETA_THRESHOLD_RADIANS,
                    DEFAULT_THETA_THRESHOLD_RADIANS,
                    12,
                ),
            ],
        )
        .expect("default entropic bin spec must be valid");

        Self {
            initial_state,
            cartpole,
            horizon: 12,
            rollouts_per_step: 16,
            candidate_actions: vec![-10.0, -5.0, 0.0, 5.0, 10.0],
            bin_spec,
            temperature: None,
            has_angular_upright_reward: false,
            episode_ticks: 150,
        }
    }
}

impl EntropicExperimentConfig {
    /// Devuelve una copia de [`CartPoleConfig`] sincronizada con `self.initial_state`.
    pub fn effective_cartpole_config(&self) -> CartPoleConfig {
        let mut cfg = self.cartpole.clone();
        cfg.initial_state = self.initial_state;
        cfg
    }

    /// Valida todos los invariantes de la configuración experimental.
    pub fn validate(&self) -> Result<(), DomainError> {
        self.cartpole.validate()?;
        self.initial_state.validate()?;

        if self.has_angular_upright_reward {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment forbids angular upright reward (has_angular_upright_reward must be false)",
            });
        }
        if self.horizon == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment horizon must be >= 1",
            });
        }
        if self.rollouts_per_step == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment rollouts_per_step must be >= 1",
            });
        }
        if self.episode_ticks == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment episode_ticks must be >= 1",
            });
        }
        if self.candidate_actions.is_empty() {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment candidate_actions cannot be empty",
            });
        }
        for &force in &self.candidate_actions {
            if !force.is_finite() {
                return Err(DomainError::NonFiniteValue {
                    field: "entropic.candidate_action",
                    value: force,
                });
            }
            if force.abs() > self.cartpole.max_force {
                return Err(DomainError::ContinuousActionOutOfRange {
                    dimension: 0,
                    value: force,
                    min: -self.cartpole.max_force,
                    max: self.cartpole.max_force,
                });
            }
        }

        let dim = self.bin_spec.dimension();
        if dim != 2 && dim != 4 {
            return Err(DomainError::InvalidTransition {
                reason: "entropic experiment bin_spec must have dimension 2 (x, theta) or 4 (x, x_dot, theta, theta_dot)",
            });
        }

        if let Some(temp) = self.temperature {
            if !temp.is_finite() {
                return Err(DomainError::NonFiniteValue {
                    field: "entropic.temperature",
                    value: temp,
                });
            }
            if temp <= 0.0 {
                return Err(DomainError::InvalidTransition {
                    reason: "entropic experiment temperature must be strictly positive when specified",
                });
            }
        }

        Ok(())
    }
}

fn extract_bin_features(
    kin: CartPoleKinematics,
    bin_spec: &BinSpec,
) -> Result<Vec<f64>, DomainError> {
    match bin_spec.dimension() {
        2 => Ok(vec![kin.x, kin.theta]),
        4 => Ok(vec![kin.x, kin.x_dot, kin.theta, kin.theta_dot]),
        _ => Err(DomainError::InvalidTransition {
            reason: "entropic bin_spec must have dimension 2 or 4",
        }),
    }
}

/// Política sin control (`force = 0.0`) para línea base experimental.
#[derive(Debug, Clone, Copy, Default)]
struct NoControlPolicy;

impl CartPolePolicy for NoControlPolicy {
    fn select_action(
        &mut self,
        _observation: &Observation,
        action_space: &ActionSpace,
    ) -> Result<Action, DomainError> {
        Action::continuous_in_space(vec![0.0], action_space)
    }
}

/// Controlador por proxy empírico de entropía de trayectorias futuras viables.
///
/// En cada paso:
/// 1. Sincroniza el [`CartPoleModel`] interno con la observación actual.
/// 2. Para cada acción candidata `a`, bifurca (`fork`) el modelo `rollouts_per_step` veces,
///    ejecuta `a` en el primer paso y acciones candidatas muestreadas con [`DeterministicRng`]
///    hasta completar el horizonte `H`.
/// 3. Conserva únicamente las trayectorias completas no-terminales (sin usar reward angular).
/// 4. Estima el proxy empírico de entropía con [`estimate_future_trajectory_entropy_proxy`] y
///    selecciona la acción que maximiza la entropía futura (o muestreo Boltzmann si `temperature`
///    está configurada), con fallback seguro si todas las trayectorias fallan.
#[derive(Debug, Clone)]
pub struct EntropicProxyController {
    config: EntropicExperimentConfig,
    model: CartPoleModel,
    rng: DeterministicRng,
}

#[derive(Debug, Clone, Copy)]
struct CandidateEvaluation {
    force: f64,
    viable_count: usize,
    entropy_bits: Option<f64>,
    max_survival_steps: usize,
}

impl EntropicProxyController {
    /// Crea un nuevo [`EntropicProxyController`] validando `config` e inicializando su modelo y RNG.
    pub fn new(config: EntropicExperimentConfig, seed: MasterSeed) -> Result<Self, DomainError> {
        config.validate()?;
        let model = CartPoleModel::new(config.effective_cartpole_config(), seed)?;
        let rng = seed.derive(StreamKey::Agent {
            episode: 0,
            agent: 1,
        });
        Ok(Self { config, model, rng })
    }

    /// Devuelve la configuración experimental asociada al controlador.
    pub const fn config(&self) -> &EntropicExperimentConfig {
        &self.config
    }

    fn sync_model_to_observation(&mut self, observation: &Observation) -> Result<(), DomainError> {
        let kin = CartPoleKinematics::from_slice(observation.features())?;
        let state = State::new(observation.tick(), observation.elapsed(), kin.to_vec())?;
        let terminal = kin.terminal_status(self.model.config());
        let snap = SimulationSnapshot::new(state, &self.rng, terminal);
        self.model
            .restore(&snap)
            .map_err(|_| DomainError::InvalidTransition {
                reason: "failed to synchronize entropic controller model with observation",
            })
    }

    fn evaluate_candidate(
        &mut self,
        first_force: f64,
        action_space: &ActionSpace,
    ) -> Result<CandidateEvaluation, DomainError> {
        let first_action = Action::continuous_in_space(vec![first_force], action_space)?;
        let mut viable_trajectories = Vec::with_capacity(self.config.rollouts_per_step);
        let mut max_survival_steps = 0_usize;
        let num_actions = self.config.candidate_actions.len();

        for _ in 0..self.config.rollouts_per_step {
            let mut branch = self
                .model
                .fork()
                .map_err(|_| DomainError::InvalidTransition {
                    reason: "failed to fork CartPoleModel for entropic rollout",
                })?;

            let mut steps = Vec::with_capacity(self.config.horizon);
            let mut survived_all = true;

            for h in 0..self.config.horizon {
                let action = if h == 0 {
                    first_action.clone()
                } else {
                    let idx = (self.rng.next_u64() as usize) % num_actions;
                    let f = self.config.candidate_actions[idx];
                    Action::continuous_in_space(vec![f], action_space)?
                };

                let transition = match branch.step_in_place(&action) {
                    Ok(tr) => tr,
                    Err(_) => {
                        survived_all = false;
                        break;
                    }
                };

                if transition.terminal().is_terminal() {
                    survived_all = false;
                    break;
                }

                let kin = branch.kinematics();
                let features = extract_bin_features(kin, &self.config.bin_spec)?;
                let step = TrajectoryStep::new(transition.next_state().tick(), features).map_err(
                    |_| DomainError::InvalidTransition {
                        reason: "failed to build trajectory step for entropic proxy",
                    },
                )?;
                steps.push(step);
                max_survival_steps = max_survival_steps.max(steps.len());
            }

            if survived_all && !steps.is_empty() {
                let summary =
                    TrajectorySummary::new(steps).map_err(|_| DomainError::InvalidTransition {
                        reason: "failed to summarize viable trajectory",
                    })?;
                viable_trajectories.push(summary);
            }
        }

        let viable_count = viable_trajectories.len();
        let entropy_bits = if viable_trajectories.is_empty() {
            None
        } else {
            let estimate = estimate_future_trajectory_entropy_proxy(
                &viable_trajectories,
                &self.config.bin_spec,
            )
            .map_err(|_| DomainError::InvalidTransition {
                reason: "failed to estimate future trajectory entropy proxy",
            })?;
            Some(estimate.entropy_bits())
        };

        Ok(CandidateEvaluation {
            force: first_force,
            viable_count,
            entropy_bits,
            max_survival_steps,
        })
    }

    fn choose_force(&mut self, evals: &[CandidateEvaluation]) -> f64 {
        let viable: Vec<&CandidateEvaluation> =
            evals.iter().filter(|e| e.viable_count > 0).collect();

        if viable.is_empty() {
            // Fallback seguro si todas las trayectorias terminan antes del horizonte:
            // elige la acción con mayor supervivencia observada (o 0.0 si todas fallan en paso 1).
            return evals
                .iter()
                .max_by(|a, b| a.max_survival_steps.cmp(&b.max_survival_steps))
                .filter(|e| e.max_survival_steps > 0)
                .map_or(0.0, |e| e.force);
        }

        if let Some(temp) = self.config.temperature {
            let max_entropy = viable
                .iter()
                .filter_map(|e| e.entropy_bits)
                .fold(f64::NEG_INFINITY, f64::max);
            let mut weights = Vec::with_capacity(viable.len());
            let mut sum_w = 0.0_f64;
            for e in &viable {
                let h = e.entropy_bits.unwrap_or(0.0);
                let w = ((h - max_entropy) / temp).exp();
                weights.push(w);
                sum_w += w;
            }
            if sum_w.is_finite() && sum_w > 0.0 {
                let target = self.rng.next_f64() * sum_w;
                let mut acc = 0.0_f64;
                for (e, w) in viable.iter().zip(weights.iter()) {
                    acc += *w;
                    if target <= acc {
                        return e.force;
                    }
                }
                return viable[viable.len() - 1].force;
            }
        }

        viable
            .iter()
            .max_by(|a, b| {
                let ha = a.entropy_bits.unwrap_or(0.0);
                let hb = b.entropy_bits.unwrap_or(0.0);
                ha.total_cmp(&hb)
                    .then_with(|| a.viable_count.cmp(&b.viable_count))
            })
            .map_or(0.0, |e| e.force)
    }
}

impl CartPolePolicy for EntropicProxyController {
    fn select_action(
        &mut self,
        observation: &Observation,
        action_space: &ActionSpace,
    ) -> Result<Action, DomainError> {
        self.sync_model_to_observation(observation)?;
        let candidate_actions = self.config.candidate_actions.clone();
        let mut evals = Vec::with_capacity(candidate_actions.len());
        for force in candidate_actions {
            evals.push(self.evaluate_candidate(force, action_space)?);
        }
        let chosen_force = self.choose_force(&evals);
        Action::continuous_in_space(vec![chosen_force], action_space)
    }
}

/// Condición experimental comparada en [`run_entropic_comparison`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EntropicConditionKind {
    /// Sin control aplicado (`force = 0.0`).
    NoControl,
    /// Controlador clásico de realimentación lineal (`LinearFeedbackController`).
    ClassicalController,
    /// Controlador por proxy empírico de entropía futura sin reward angular.
    EntropicProxy,
}

impl fmt::Display for EntropicConditionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoControl => write!(f, "NoControl"),
            Self::ClassicalController => write!(f, "ClassicalController"),
            Self::EntropicProxy => write!(f, "EntropicProxy"),
        }
    }
}

/// Métricas medidas para una condición y una semilla fija.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionSeedMetrics {
    /// Semilla maestra de la ejecución.
    pub seed: MasterSeed,
    /// Condición experimental evaluada.
    pub condition: EntropicConditionKind,
    /// Número de ticks completados en estado erguido (no terminal).
    pub upright_ticks: u64,
    /// Energía total de control `sum(force^2 * dt)`.
    pub total_control_energy: f64,
    /// Indica si el episodio terminó en fallo (`TerminalReason::Failure`).
    pub failed: bool,
    /// Tick exacto en el que ocurrió el fallo, si falló.
    pub failure_tick: Option<u64>,
    /// Estado cinemático final observado.
    pub final_kinematics: CartPoleKinematics,
    /// Hash determinista de la traza completa del episodio.
    pub trace_hash: u64,
}

/// Métricas agregadas sobre todas las semillas para una condición experimental.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionAggregateMetrics {
    /// Condición experimental evaluada.
    pub condition: EntropicConditionKind,
    /// Resultados individuales por semilla.
    pub runs: Vec<ConditionSeedMetrics>,
    /// Número total de semillas en las que ocurrió fallo.
    pub failure_count: usize,
    /// Número total de semillas que completaron el episodio sin fallar.
    pub success_count: usize,
    /// Promedio de `upright_ticks` sobre las semillas.
    pub mean_upright_ticks: f64,
    /// Promedio de `total_control_energy` sobre las semillas.
    pub mean_control_energy: f64,
    /// Indica si al menos una semilla falló.
    pub any_failed: bool,
    /// Resumen explícito y honesto de fallos si `any_failed == true`.
    pub failure_summary: Option<String>,
}

impl ConditionAggregateMetrics {
    fn from_runs(condition: EntropicConditionKind, runs: Vec<ConditionSeedMetrics>) -> Self {
        let total = runs.len();
        let failure_count = runs.iter().filter(|r| r.failed).count();
        let success_count = total.saturating_sub(failure_count);
        let mean_upright_ticks = if total == 0 {
            0.0
        } else {
            runs.iter().map(|r| r.upright_ticks as f64).sum::<f64>() / (total as f64)
        };
        let mean_control_energy = if total == 0 {
            0.0
        } else {
            runs.iter().map(|r| r.total_control_energy).sum::<f64>() / (total as f64)
        };
        let any_failed = failure_count > 0;
        let failure_summary = if any_failed {
            let first_fail = runs
                .iter()
                .find_map(|r| r.failure_tick.map(|t| (r.seed.as_u64(), t)));
            match first_fail {
                Some((seed_val, tick)) => Some(format!(
                    "{condition} failed in {failure_count}/{total} seeds (first failure at tick {tick}, seed {seed_val})"
                )),
                None => Some(format!(
                    "{condition} failed in {failure_count}/{total} seeds"
                )),
            }
        } else {
            None
        };

        Self {
            condition,
            runs,
            failure_count,
            success_count,
            mean_upright_ticks,
            mean_control_energy,
            any_failed,
            failure_summary,
        }
    }
}

/// Reporte comparativo completo entre `NoControl`, `ClassicalController` y `EntropicProxy`.
#[derive(Debug, Clone, PartialEq)]
pub struct EntropicComparisonReport {
    /// Configuración experimental validada utilizada.
    pub config: EntropicExperimentConfig,
    /// Semillas fijas compartidas entre las 3 condiciones.
    pub seeds: Vec<MasterSeed>,
    /// Resultados agregados para `NoControl`.
    pub no_control: ConditionAggregateMetrics,
    /// Resultados agregados para `ClassicalController`.
    pub classical: ConditionAggregateMetrics,
    /// Resultados agregados para `EntropicProxy`.
    pub entropic_proxy: ConditionAggregateMetrics,
    /// Nota epistemológica explícita sobre el alcance del experimento.
    pub epistemological_note: String,
}

impl EntropicComparisonReport {
    /// Formatea un resumen legible con métricas, fallos publicados honestamente y nota epistemológica.
    pub fn format_summary(&self) -> String {
        let mut out = String::new();
        for agg in [&self.no_control, &self.classical, &self.entropic_proxy] {
            out.push_str(&format!(
                "{}: mean_upright_ticks={:.2}, mean_energy={:.4}, failures={}/{}",
                agg.condition,
                agg.mean_upright_ticks,
                agg.mean_control_energy,
                agg.failure_count,
                agg.runs.len()
            ));
            if let Some(ref summary) = agg.failure_summary {
                out.push_str(&format!(" [{summary}]"));
            }
            out.push('\n');
        }
        out.push_str(&self.epistemological_note);
        out
    }
}

fn run_single_condition_episode(
    config: &CartPoleConfig,
    ticks: u64,
    seed: MasterSeed,
    condition: EntropicConditionKind,
    policy: &mut dyn CartPolePolicy,
) -> Result<ConditionSeedMetrics, DomainError> {
    let mut env = CartPoleEnv::new(config.clone())?;
    let mut hasher = TraceHasher::with_metadata(&DeterminismMetadata::current(seed));
    let dt_secs = config.dt.as_secs_f64();

    let mut upright_ticks = 0_u64;
    let mut total_control_energy = 0.0_f64;
    let mut failed = env.terminal().is_terminal();
    let mut failure_tick = if failed { Some(0) } else { None };

    if !failed {
        for _ in 0..ticks {
            let obs =
                EnvironmentObserver::observe(&env).map_err(|_| DomainError::InvalidTransition {
                    reason: "failed to observe cartpole environment during entropic comparison",
                })?;
            let action = policy.select_action(&obs, Actuator::action_space(&env))?;
            let force = action
                .as_continuous()
                .and_then(|vals| vals.first().copied())
                .ok_or(DomainError::IncompatibleActionKind {
                    expected: "continuous",
                    actual: "discrete",
                })?;
            total_control_energy += force * force * dt_secs;

            let outcome = env.apply(&action).map_err(|err| match err {
                EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
                EnvironmentError::InvalidAction(d) => d,
                _ => DomainError::InvalidTransition {
                    reason: "failed to apply cartpole action during entropic comparison",
                },
            })?;

            hasher.write_tick(env.tick());
            hasher.write_observation(&obs);
            hasher.write_f64(force);
            hasher.write_observation(outcome.observation());
            hasher.write_terminal(outcome.terminal());

            if outcome.terminal().is_terminal() {
                failed = true;
                failure_tick = Some(env.tick().as_u64());
                break;
            }
            upright_ticks += 1;
        }
    }

    Ok(ConditionSeedMetrics {
        seed,
        condition,
        upright_ticks,
        total_control_energy,
        failed,
        failure_tick,
        final_kinematics: env.kinematics(),
        trace_hash: hasher.finish(),
    })
}

/// Ejecuta la comparación determinista en las mismas `seeds` fijas entre:
/// 1. `NoControl` (fuerza = 0.0)
/// 2. `ClassicalController` (`LinearFeedbackController`)
/// 3. `EntropicProxy` (`EntropicProxyController`)
///
/// Mide tiempo erguido (`upright_ticks`), energía de control (`sum(force^2 * dt)`) y fallos,
/// publicando honestamente cualquier fracaso ocurrido e incluyendo la nota epistemológica.
pub fn run_entropic_comparison(
    seeds: &[MasterSeed],
    config: &EntropicExperimentConfig,
) -> Result<EntropicComparisonReport, DomainError> {
    config.validate()?;
    if seeds.is_empty() {
        return Err(DomainError::InvalidTransition {
            reason: "entropic comparison requires at least one fixed MasterSeed",
        });
    }

    let cartpole_cfg = config.effective_cartpole_config();
    let mut no_control_runs = Vec::with_capacity(seeds.len());
    let mut classical_runs = Vec::with_capacity(seeds.len());
    let mut entropic_runs = Vec::with_capacity(seeds.len());

    for &seed in seeds {
        let mut no_ctrl = NoControlPolicy;
        no_control_runs.push(run_single_condition_episode(
            &cartpole_cfg,
            config.episode_ticks,
            seed,
            EntropicConditionKind::NoControl,
            &mut no_ctrl,
        )?);

        let mut classical = LinearFeedbackController::default_stabilizing();
        classical_runs.push(run_single_condition_episode(
            &cartpole_cfg,
            config.episode_ticks,
            seed,
            EntropicConditionKind::ClassicalController,
            &mut classical,
        )?);

        let mut entropic = EntropicProxyController::new(config.clone(), seed)?;
        entropic_runs.push(run_single_condition_episode(
            &cartpole_cfg,
            config.episode_ticks,
            seed,
            EntropicConditionKind::EntropicProxy,
            &mut entropic,
        )?);
    }

    Ok(EntropicComparisonReport {
        config: config.clone(),
        seeds: seeds.to_vec(),
        no_control: ConditionAggregateMetrics::from_runs(
            EntropicConditionKind::NoControl,
            no_control_runs,
        ),
        classical: ConditionAggregateMetrics::from_runs(
            EntropicConditionKind::ClassicalController,
            classical_runs,
        ),
        entropic_proxy: ConditionAggregateMetrics::from_runs(
            EntropicConditionKind::EntropicProxy,
            entropic_runs,
        ),
        epistemological_note: EPISTEMOLOGICAL_NOTE.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropic_ac1_config_documents_parameters_and_rejects_angular_upright_reward() {
        let cfg = EntropicExperimentConfig::default();
        cfg.validate().unwrap();
        assert!(!cfg.has_angular_upright_reward);
        assert!(cfg.horizon >= 1);
        assert!(cfg.rollouts_per_step >= 1);
        assert!(!cfg.candidate_actions.is_empty());
        assert_eq!(cfg.bin_spec.dimension(), 2);
        assert_eq!(cfg.temperature, None);

        // Rechaza explícitamente inyección de reward angular.
        let mut with_reward = cfg.clone();
        with_reward.has_angular_upright_reward = true;
        assert!(with_reward.validate().is_err());

        // Rechaza horizonte 0, rollouts 0, acciones vacías y temperatura no positiva.
        let mut bad_horizon = cfg.clone();
        bad_horizon.horizon = 0;
        assert!(bad_horizon.validate().is_err());

        let mut bad_rollouts = cfg.clone();
        bad_rollouts.rollouts_per_step = 0;
        assert!(bad_rollouts.validate().is_err());

        let mut empty_actions = cfg.clone();
        empty_actions.candidate_actions.clear();
        assert!(empty_actions.validate().is_err());

        let mut bad_temp = cfg.clone();
        bad_temp.temperature = Some(0.0);
        assert!(bad_temp.validate().is_err());

        // Rechaza BinSpec con dimensionalidad distinta de 2 o 4.
        let mut bad_bins = cfg.clone();
        bad_bins.bin_spec =
            BinSpec::uniform("1d-invalid", BinningVersion::V1_0, vec![(-1.0, 1.0, 4)]).unwrap();
        assert!(bad_bins.validate().is_err());

        // Acepta BinSpec 4D sobre (x, x_dot, theta, theta_dot) y temperatura positiva.
        let mut valid_4d = cfg;
        valid_4d.bin_spec = BinSpec::uniform(
            "4d-cartpole",
            BinningVersion::V1_0,
            vec![
                (-DEFAULT_X_THRESHOLD, DEFAULT_X_THRESHOLD, 6),
                (-3.0, 3.0, 6),
                (
                    -DEFAULT_THETA_THRESHOLD_RADIANS,
                    DEFAULT_THETA_THRESHOLD_RADIANS,
                    6,
                ),
                (-3.0, 3.0, 6),
            ],
        )
        .unwrap();
        valid_4d.temperature = Some(0.5);
        valid_4d.validate().unwrap();
    }

    #[test]
    fn entropic_ac2_ac3_comparison_is_deterministic_measures_energy_and_publishes_failures() {
        let seeds = [MasterSeed::new(101), MasterSeed::new(202)];
        let cfg = EntropicExperimentConfig::default();

        let report_a = run_entropic_comparison(&seeds, &cfg).unwrap();
        let report_b = run_entropic_comparison(&seeds, &cfg).unwrap();
        assert_eq!(report_a, report_b);

        // NoControl debe fallar y publicar su fallo honestamente.
        assert!(report_a.no_control.any_failed);
        assert_eq!(report_a.no_control.failure_count, 2);
        assert_eq!(report_a.no_control.success_count, 0);
        assert!(report_a.no_control.failure_summary.is_some());
        assert_eq!(report_a.no_control.mean_control_energy, 0.0);
        for run in &report_a.no_control.runs {
            assert!(run.failed);
            assert!(run.failure_tick.is_some());
            assert_eq!(run.total_control_energy, 0.0);
            assert!(run.upright_ticks < cfg.episode_ticks);
        }

        // ClassicalController estabiliza el escenario por defecto con energía positiva.
        assert!(!report_a.classical.any_failed);
        assert_eq!(report_a.classical.failure_count, 0);
        assert_eq!(
            report_a.classical.mean_upright_ticks,
            cfg.episode_ticks as f64
        );
        assert!(report_a.classical.mean_control_energy > 0.0);

        // EntropicProxy ejecuta sin reward angular, mantiene tiempo erguido mayor que NoControl
        // y mide energía de control positiva.
        assert!(
            report_a.entropic_proxy.mean_upright_ticks > report_a.no_control.mean_upright_ticks,
            "entropic proxy ({}) should outperform NoControl ({}) in upright_ticks",
            report_a.entropic_proxy.mean_upright_ticks,
            report_a.no_control.mean_upright_ticks
        );
        assert!(report_a.entropic_proxy.mean_control_energy > 0.0);

        // Nota epistemológica explícita sobre alcance PRL y muestreo de futuros.
        let note = report_a.epistemological_note.to_lowercase();
        assert!(note.contains("no afirma replicación literal"));
        assert!(note.contains("prl"));
        assert!(note.contains("muestreo"));
        assert!(report_a.format_summary().contains("NoControl"));
    }

    #[test]
    fn entropic_publishes_honest_failure_on_extreme_initial_state_and_supports_boltzmann() {
        // Estado inicial extremo cerca del límite angular: incluso controladores pueden fallar.
        let extreme_init =
            CartPoleKinematics::new(0.0, 0.0, DEFAULT_THETA_THRESHOLD_RADIANS - 0.005, 2.5)
                .unwrap();
        let mut cfg = EntropicExperimentConfig::default();
        cfg.initial_state = extreme_init;
        cfg.episode_ticks = 40;
        cfg.temperature = Some(0.25);
        cfg.cartpole = cfg.cartpole.clone().with_initial_state(extreme_init);

        let report = run_entropic_comparison(&[MasterSeed::new(7)], &cfg).unwrap();
        assert!(report.entropic_proxy.any_failed);
        assert_eq!(report.entropic_proxy.failure_count, 1);
        assert!(report.entropic_proxy.failure_summary.is_some());
        assert!(report.entropic_proxy.runs[0].failed);
        assert!(report.entropic_proxy.runs[0].failure_tick.is_some());
    }
}
