//! Comparación pareada Kart en arena: cuatro roles justos y un techo privilegiado separado.
//!
//! Todos los roles corren sobre la misma semilla, pista, física cinemática, contrato de
//! observación (`KartEnv::observe`) y límite de ticks. Los roles que simulan futuros consumen el
//! mismo presupuesto de transiciones por decisión ([`KartArenaBudget::transitions_per_decision`]);
//! [`KartArenaConfig::validate`] rechaza cualquier desigualdad ([`ArenaError::UnequalBudget`]).
//!
//! # Roles
//! - [`KartRole::PureEntropic`]: controlador puro sin recompensa (`EntropicPureController`).
//! - [`KartRole::GoalOrientedFmc`]: FMC con pesos de objetivo (`KartFmcController`).
//! - [`KartRole::Random`]: control uniforme acotado, sin simulación.
//! - [`KartRole::GeometricBaseline`]: seguimiento geométrico de checkpoints (`SimpleTrackBaseline`).
//! - [`KartRole::PrivilegedFmcCeiling`]: FMC con presupuesto de transiciones mayor. Está marcado
//!   `is_privileged = true`, vive en campos separados del informe y no entra en la comparación
//!   justa ni en `fair_*`.
//!
//! # Alcance
//! Es un arnés de medición con semillas y pistas fijas; los números de un informe describen esas
//! ejecuciones concretas y no demuestran propiedades generales de ningún rol. Los episodios
//! fallidos (choques, timeouts, errores de controlador) se conservan siempre en el informe.

use gogamen_core::{
    Action, Actuator, DomainError, EnvironmentObserver, FmcPlannerConfig, FmcScoringConfig,
    LiveEnvironment, MasterSeed, PlanStatus, RngSource, RunId, State, StreamKey, Terminal,
    TerminalReason, Tick,
};
use gogamen_lab_grid::SeedSplit;
use gogamen_lab_racing::{
    EntropicPureConfig, EntropicPureController, KartControl, KartEnv, KartFmcController,
    KartFmcControllerConfig, KartObjectiveWeights, KinematicBicycleConfig, SimpleTrackBaseline,
    SimpleTrackBaselineConfig, TrackMap,
};
use gogamen_runtime::scientific_trace::{ScientificTraceArchive, StreamSeeds, TraceManifest};

use crate::runner::{ArenaError, ArenaSeedSet, TraceRecorder};
use crate::stats::{DispersionStats, RunOutcomeStatus};

/// Candidatos raíz del controlador puro (retícula 3x3 fija de su algoritmo).
const PURE_ROOT_CANDIDATES: usize = 9;
/// Factor del techo privilegiado sobre el presupuesto justo.
const CEILING_FACTOR: usize = 4;

/// Rol de control en la arena Kart.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KartRole {
    /// Controlador puro sin recompensa.
    PureEntropic,
    /// FMC orientado a objetivo (pesos de objetivo).
    GoalOrientedFmc,
    /// Control aleatorio uniforme acotado.
    Random,
    /// Baseline geométrico a checkpoints.
    GeometricBaseline,
    /// Techo FMC con presupuesto privilegiado (fuera de la comparación justa).
    PrivilegedFmcCeiling,
}

impl KartRole {
    /// Etiqueta estable del rol.
    pub const fn label(self) -> &'static str {
        match self {
            Self::PureEntropic => "pure_reward_free_entropic",
            Self::GoalOrientedFmc => "goal_oriented_fmc",
            Self::Random => "random",
            Self::GeometricBaseline => "geometric_baseline",
            Self::PrivilegedFmcCeiling => "privileged_fmc_ceiling",
        }
    }

    /// `true` solo para el techo privilegiado.
    pub const fn is_privileged(self) -> bool {
        matches!(self, Self::PrivilegedFmcCeiling)
    }

    /// `true` si el rol simula futuros (consume transiciones de planificación).
    pub const fn plans(self) -> bool {
        matches!(
            self,
            Self::PureEntropic | Self::GoalOrientedFmc | Self::PrivilegedFmcCeiling
        )
    }
}

/// Presupuesto compartido por todos los roles justos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KartArenaBudget {
    /// Ticks máximos por episodio.
    pub max_ticks: usize,
    /// Horizonte de simulación de los roles que planifican.
    pub horizon: usize,
    /// Transiciones simuladas por decisión (igual para los roles que planifican).
    pub transitions_per_decision: usize,
}

impl KartArenaBudget {
    /// Construye el presupuesto; `transitions_per_decision` debe ser múltiplo de `9 * horizon`.
    pub fn new(
        max_ticks: usize,
        horizon: usize,
        transitions_per_decision: usize,
    ) -> Result<Self, ArenaError> {
        if max_ticks == 0 || horizon == 0 || transitions_per_decision == 0 {
            return Err(ArenaError::InvalidBudget {
                reason: "kart budget fields must be > 0",
            });
        }
        if horizon > 64 || !transitions_per_decision.is_multiple_of(PURE_ROOT_CANDIDATES * horizon)
        {
            return Err(ArenaError::InvalidBudget {
                reason: "kart transitions_per_decision must be a multiple of 9 * horizon (horizon <= 64)",
            });
        }
        Ok(Self {
            max_ticks,
            horizon,
            transitions_per_decision,
        })
    }
}

impl Default for KartArenaBudget {
    fn default() -> Self {
        Self {
            max_ticks: 60,
            horizon: 8,
            transitions_per_decision: PURE_ROOT_CANDIDATES * 8 * 4,
        }
    }
}

/// Rol más su dimensionamiento de planificación (`population`, `horizon`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KartRoleSpec {
    /// Rol.
    pub role: KartRole,
    /// Población (0 para roles sin planificación).
    pub population: usize,
    /// Horizonte (0 para roles sin planificación).
    pub horizon: usize,
}

impl KartRoleSpec {
    /// Transiciones simuladas por decisión que consume este rol.
    pub const fn transitions_per_decision(&self) -> usize {
        match self.role {
            KartRole::PureEntropic => PURE_ROOT_CANDIDATES * self.population * self.horizon,
            KartRole::GoalOrientedFmc | KartRole::PrivilegedFmcCeiling => {
                self.population * self.horizon
            }
            KartRole::Random | KartRole::GeometricBaseline => 0,
        }
    }
}

/// Configuración de una comparación Kart pareada.
#[derive(Debug, Clone)]
pub struct KartArenaConfig {
    /// Identificador de especificación para la traza científica.
    pub spec_id: String,
    /// Pista compartida.
    pub track: TrackMap,
    /// Física compartida.
    pub kinematics: KinematicBicycleConfig,
    /// Presupuesto compartido.
    pub budget: KartArenaBudget,
    /// Semillas `dev` y `holdout` disjuntas.
    pub seeds: ArenaSeedSet,
    /// Pesos de objetivo del FMC orientado (también usados por el techo).
    pub weights: KartObjectiveWeights,
    /// Roles ejecutados (cada uno como máximo una vez).
    pub roles: Vec<KartRoleSpec>,
}

impl KartArenaConfig {
    /// Construye la configuración estándar: cuatro roles justos con presupuesto igual más el techo
    /// privilegiado (`4x` transiciones por decisión) etiquetado aparte.
    pub fn new(
        spec_id: impl Into<String>,
        track: TrackMap,
        budget: KartArenaBudget,
        seeds: ArenaSeedSet,
    ) -> Self {
        let h = budget.horizon;
        let t = budget.transitions_per_decision;
        let roles = vec![
            KartRoleSpec {
                role: KartRole::PureEntropic,
                population: t / (PURE_ROOT_CANDIDATES * h),
                horizon: h,
            },
            KartRoleSpec {
                role: KartRole::GoalOrientedFmc,
                population: t / h,
                horizon: h,
            },
            KartRoleSpec {
                role: KartRole::Random,
                population: 0,
                horizon: 0,
            },
            KartRoleSpec {
                role: KartRole::GeometricBaseline,
                population: 0,
                horizon: 0,
            },
            KartRoleSpec {
                role: KartRole::PrivilegedFmcCeiling,
                population: CEILING_FACTOR * t / h,
                horizon: h,
            },
        ];
        Self {
            spec_id: spec_id.into(),
            track,
            kinematics: KinematicBicycleConfig::default(),
            budget,
            seeds,
            weights: KartObjectiveWeights::default(),
            roles,
        }
    }

    /// Valida semillas, roles y la igualdad de presupuesto de los roles justos.
    pub fn validate(&self) -> Result<(), ArenaError> {
        self.seeds.validate()?;
        self.kinematics.validate()?;
        self.weights.validate()?;
        if self.roles.is_empty() {
            return Err(ArenaError::InvalidRoles {
                reason: "no roles configured",
            });
        }
        let mut seen = std::collections::BTreeSet::new();
        for spec in &self.roles {
            if !seen.insert(spec.role) {
                return Err(ArenaError::InvalidRoles {
                    reason: "duplicate role",
                });
            }
            let actual = spec.transitions_per_decision();
            let expected = self.budget.transitions_per_decision;
            match spec.role {
                KartRole::PureEntropic | KartRole::GoalOrientedFmc => {
                    if spec.horizon != self.budget.horizon || actual != expected {
                        return Err(ArenaError::UnequalBudget {
                            role: spec.role.label(),
                            expected,
                            actual,
                        });
                    }
                }
                KartRole::Random | KartRole::GeometricBaseline => {
                    if spec.population != 0 || spec.horizon != 0 {
                        return Err(ArenaError::InvalidRoles {
                            reason: "random/baseline roles must not plan",
                        });
                    }
                }
                KartRole::PrivilegedFmcCeiling => {
                    if spec.population == 0 || spec.horizon == 0 || actual <= expected {
                        return Err(ArenaError::InvalidRoles {
                            reason: "privileged ceiling must have a strictly larger budget",
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Resultado de un rol en una semilla (incluye episodios fallidos).
#[derive(Debug, Clone, PartialEq)]
pub struct KartRunResult {
    /// Rol.
    pub role: KartRole,
    /// Marca explícita de rol privilegiado.
    pub is_privileged: bool,
    /// Split de la semilla.
    pub split: SeedSplit,
    /// Semilla compartida.
    pub seed: MasterSeed,
    /// Estado de finalización (`Succeeded` = meta alcanzada; el resto se conserva).
    pub status: RunOutcomeStatus,
    /// Ticks ejecutados.
    pub steps_executed: u64,
    /// Checkpoints superados.
    pub checkpoints_cleared: usize,
    /// Eficacia: checkpoints superados / checkpoints de la pista.
    pub efficacy: f64,
    /// Choques (0 o 1 por episodio).
    pub collisions: usize,
    /// `true` si la trayectoria no terminó en choque ni error (viable dentro del presupuesto).
    pub viable_trajectory: bool,
    /// Fracción de futuros simulados viables (solo el controlador puro la mide).
    pub viable_future_fraction: Option<f64>,
    /// Transiciones simuladas totales.
    pub transitions_total: u64,
    /// Coste medio: transiciones simuladas por decisión.
    pub cost_per_decision: f64,
    /// Latencia de pared (no entra en el hash determinista).
    pub latency_ns: u128,
    /// Hash FNV-1a determinista de controles y resultado.
    pub control_hash: u64,
    /// Manifiesto de la traza científica.
    pub trace_manifest: TraceManifest,
    /// Archivo de traza científica.
    pub trace_archive: ScientificTraceArchive,
}

/// Resultados de una misma semilla: roles justos y techo, en campos separados.
#[derive(Debug, Clone, PartialEq)]
pub struct KartSeedOutcome {
    /// Split.
    pub split: SeedSplit,
    /// Semilla.
    pub seed: MasterSeed,
    /// Roles justos (mismo presupuesto).
    pub fair_runs: Vec<KartRunResult>,
    /// Techo privilegiado (nunca mezclado con `fair_runs`).
    pub ceiling_runs: Vec<KartRunResult>,
}

/// Agregado por rol con denominador total intacto.
#[derive(Debug, Clone, PartialEq)]
pub struct KartRoleSummary {
    /// Rol.
    pub role: KartRole,
    /// Marca explícita de rol privilegiado.
    pub is_privileged: bool,
    /// Episodios totales (denominador intacto).
    pub total_runs: usize,
    /// Episodios con meta alcanzada.
    pub succeeded_count: usize,
    /// Episodios fallidos (choque o error de controlador).
    pub failed_count: usize,
    /// Episodios sin meta por límite de ticks.
    pub timed_out_count: usize,
    /// Choques totales.
    pub collisions_total: usize,
    /// Trayectorias viables (sin choque ni error).
    pub viable_trajectories: usize,
    /// Eficacia media.
    pub mean_efficacy: f64,
    /// Dispersión de checkpoints superados.
    pub checkpoints_dispersion: DispersionStats,
    /// Dispersión de la eficacia por semilla.
    pub efficacy_dispersion: DispersionStats,
    /// Coste medio por decisión (transiciones simuladas).
    pub mean_cost_per_decision: f64,
    /// Semillas evaluadas, en orden.
    pub seeds: Vec<u64>,
    /// Notas de todos los episodios no exitosos.
    pub failure_notes: Vec<String>,
}

/// Informe completo de la comparación Kart.
#[derive(Debug, Clone, PartialEq)]
pub struct KartArenaReport {
    /// Especificación.
    pub spec_id: String,
    /// Nombre de la pista.
    pub track_label: String,
    /// Contrato de observación compartido.
    pub observation_contract: String,
    /// Presupuesto compartido.
    pub budget: KartArenaBudget,
    /// Semillas.
    pub seeds: ArenaSeedSet,
    /// Resultados por semilla.
    pub outcomes: Vec<KartSeedOutcome>,
    /// Resúmenes de roles justos.
    pub fair_summaries: Vec<KartRoleSummary>,
    /// Resúmenes del techo privilegiado (separados).
    pub ceiling_summaries: Vec<KartRoleSummary>,
}

impl KartArenaReport {
    /// Hash determinista del informe (excluye latencias de pared).
    pub fn report_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut feed = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for o in &self.outcomes {
            feed(o.seed.as_u64());
            for r in o.fair_runs.iter().chain(&o.ceiling_runs) {
                feed(r.role as u64);
                feed(u64::from(r.is_privileged));
                feed(r.status.label().len() as u64);
                feed(r.steps_executed);
                feed(r.checkpoints_cleared as u64);
                feed(r.collisions as u64);
                feed(r.transitions_total);
                feed(r.control_hash);
                feed(r.trace_archive.records.len() as u64);
            }
        }
        h
    }

    /// Resumen legible; el techo privilegiado aparece en sección propia.
    pub fn format_summary(&self) -> String {
        let mut out = format!(
            "=== Kart Arena [{}] track={} obs={} ===\nBudget: max_ticks={}, horizon={}, transitions/decision={}\n-- fair comparison --\n",
            self.spec_id,
            self.track_label,
            self.observation_contract,
            self.budget.max_ticks,
            self.budget.horizon,
            self.budget.transitions_per_decision
        );
        let line = |s: &KartRoleSummary| {
            format!(
                "{} (privileged={}): runs={} ok={} failed={} timeout={} collisions={} viable={} efficacy={:.3} (std={:.3}) cost/decision={:.1}\n",
                s.role.label(),
                s.is_privileged,
                s.total_runs,
                s.succeeded_count,
                s.failed_count,
                s.timed_out_count,
                s.collisions_total,
                s.viable_trajectories,
                s.mean_efficacy,
                s.efficacy_dispersion.std_dev,
                s.mean_cost_per_decision
            )
        };
        for s in &self.fair_summaries {
            out.push_str(&line(s));
        }
        out.push_str("-- privileged ceiling (not part of the fair comparison) --\n");
        for s in &self.ceiling_summaries {
            out.push_str(&line(s));
        }
        out
    }
}

fn domain(reason: &'static str) -> ArenaError {
    ArenaError::Domain(DomainError::InvalidTransition { reason })
}

enum Driver {
    Pure(Box<EntropicPureController>),
    Fmc(Box<KartFmcController>),
    Random(gogamen_core::DeterministicRng),
    Baseline(Box<SimpleTrackBaseline>),
}

struct Decision {
    action: Action,
    plan_status: PlanStatus,
    fallback: bool,
    transitions: u64,
    viable: Option<(usize, usize)>,
}

fn build_driver(
    cfg: &KartArenaConfig,
    spec: &KartRoleSpec,
    seed: MasterSeed,
) -> Result<Driver, ArenaError> {
    Ok(match spec.role {
        KartRole::PureEntropic => Driver::Pure(Box::new(EntropicPureController::new(
            cfg.kinematics.clone(),
            cfg.track.clone(),
            EntropicPureConfig::new(spec.population, spec.horizon, seed)?,
        )?)),
        KartRole::GoalOrientedFmc | KartRole::PrivilegedFmcCeiling => {
            let planner = FmcPlannerConfig::new(
                spec.population,
                spec.horizon,
                cfg.kinematics.dt,
                FmcScoringConfig::default(),
                seed,
            )
            .map_err(|_| domain("invalid FMC planner config for kart arena"))?;
            let fmc_cfg = KartFmcControllerConfig::new(
                planner,
                spec.transitions_per_decision(),
                cfg.weights,
            )?;
            Driver::Fmc(Box::new(KartFmcController::new(
                cfg.kinematics.clone(),
                cfg.track.clone(),
                fmc_cfg,
            )?))
        }
        KartRole::Random => Driver::Random(seed.derive(StreamKey::Agent {
            episode: 0,
            agent: 1,
        })),
        KartRole::GeometricBaseline => Driver::Baseline(Box::new(SimpleTrackBaseline::new(
            cfg.kinematics.clone(),
            cfg.track.clone(),
            SimpleTrackBaselineConfig::default(),
        )?)),
    })
}

fn decide(driver: &mut Driver, env: &KartEnv, spec: &KartRoleSpec) -> Result<Decision, String> {
    let obs = EnvironmentObserver::observe(env).map_err(|e| format!("observe: {e:?}"))?;
    let state = State::new(obs.tick(), obs.elapsed(), obs.features().to_vec())
        .map_err(|e| format!("state: {e}"))?;
    let space = Actuator::action_space(env).clone();
    match driver {
        Driver::Pure(c) => {
            let d = c
                .decide_from_state(&state)
                .map_err(|e| format!("pure: {e:?}"))?;
            Ok(Decision {
                action: d.action,
                plan_status: PlanStatus::Completed,
                fallback: d.fail_closed,
                transitions: (d.futures_total * spec.horizon) as u64,
                viable: Some((d.viable_total, d.futures_total)),
            })
        }
        Driver::Fmc(c) => {
            let r = c
                .decide_from_state(&state)
                .map_err(|e| format!("fmc: {e:?}"))?;
            let transitions = r.transitions_used as u64;
            let status = r.plan_status;
            match r.action {
                Some(action) => Ok(Decision {
                    action,
                    plan_status: status,
                    fallback: false,
                    transitions,
                    viable: None,
                }),
                None => Err("fmc: no authorized action".to_owned()),
            }
        }
        Driver::Random(rng) => {
            let k = env.config();
            let steer = (2.0 * rng.next_f64() - 1.0) * k.max_steering_angle;
            let accel = k.min_accel + rng.next_f64() * (k.max_accel - k.min_accel);
            let action = Action::continuous_in_space(vec![steer, accel], &space)
                .map_err(|e| format!("random: {e}"))?;
            Ok(Decision {
                action,
                plan_status: PlanStatus::Completed,
                fallback: false,
                transitions: 0,
                viable: None,
            })
        }
        Driver::Baseline(b) => {
            let KartControl {
                steering_angle,
                throttle_accel,
            } = b.decide(env.kart_state(), env.checkpoints_cleared());
            let action = Action::continuous_in_space(vec![steering_angle, throttle_accel], &space)
                .map_err(|e| format!("baseline: {e}"))?;
            Ok(Decision {
                action,
                plan_status: PlanStatus::Completed,
                fallback: false,
                transitions: 0,
                viable: None,
            })
        }
    }
}

fn fold(h: &mut u64, v: u64) {
    for b in v.to_le_bytes() {
        *h ^= u64::from(b);
        *h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn run_role(
    cfg: &KartArenaConfig,
    spec: &KartRoleSpec,
    split: SeedSplit,
    seed: MasterSeed,
    run_id: RunId,
    hypotheses_hash: Option<u64>,
) -> Result<KartRunResult, ArenaError> {
    let model_id = format!("{}/{}", cfg.track.name(), spec.role.label());
    let mut recorder =
        TraceRecorder::new_preregistered(run_id, &cfg.spec_id, &model_id, seed, hypotheses_hash)?;
    let started = std::time::Instant::now();
    let stream_seeds = StreamSeeds::new(seed);

    let mut env = KartEnv::new(cfg.kinematics.clone(), cfg.track.clone())?;
    LiveEnvironment::reset(&mut env).map_err(|_| domain("kart env reset failed"))?;
    let mut driver = build_driver(cfg, spec, seed)?;

    let mut steps = 0_u64;
    let mut transitions_total = 0_u64;
    let mut viable_sum = (0_u64, 0_u64);
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut status = RunOutcomeStatus::TimedOut {
        reason: "max_ticks_reached_without_goal".to_owned(),
    };

    for tick in 0..cfg.budget.max_ticks {
        let obs = EnvironmentObserver::observe(&env).map_err(|_| domain("kart observe failed"))?;
        let d = match decide(&mut driver, &env, spec) {
            Ok(d) => d,
            Err(reason) => {
                status = RunOutcomeStatus::Failed {
                    reason: format!("controller_error: {reason}"),
                };
                break;
            }
        };
        transitions_total += d.transitions;
        if let Some((v, t)) = d.viable {
            viable_sum.0 += v as u64;
            viable_sum.1 += t as u64;
        }
        if let Some(vals) = d.action.as_continuous() {
            for v in vals {
                fold(&mut hash, v.to_bits());
            }
        }
        let out = env
            .apply(&d.action)
            .map_err(|_| domain("kart apply failed"))?;
        recorder.record_step(
            Tick::new(tick as u64),
            obs,
            d.plan_status,
            d.action,
            d.fallback,
            stream_seeds.clone(),
        )?;
        steps += 1;
        match out.terminal() {
            Terminal::Ongoing => {}
            Terminal::Terminated(TerminalReason::GoalReached) => {
                status = RunOutcomeStatus::Succeeded;
                break;
            }
            Terminal::Terminated(_) => {
                status = RunOutcomeStatus::Failed {
                    reason: "kart_collision".to_owned(),
                };
                break;
            }
        }
    }

    let latency_ns = started.elapsed().as_nanos();
    let (trace_manifest, trace_archive) = recorder.finalize()?;
    let total_cps = cfg.track.checkpoints().len().max(1);
    let checkpoints_cleared = env.checkpoints_cleared();
    let collisions = usize::from(env.collided());
    fold(&mut hash, checkpoints_cleared as u64);
    fold(&mut hash, steps);
    let viable_trajectory = !status.is_failed();

    Ok(KartRunResult {
        role: spec.role,
        is_privileged: spec.role.is_privileged(),
        split,
        seed,
        status,
        steps_executed: steps,
        checkpoints_cleared,
        efficacy: checkpoints_cleared as f64 / total_cps as f64,
        collisions,
        viable_trajectory,
        viable_future_fraction: (viable_sum.1 > 0)
            .then(|| viable_sum.0 as f64 / viable_sum.1 as f64),
        transitions_total,
        cost_per_decision: if steps == 0 {
            0.0
        } else {
            transitions_total as f64 / steps as f64
        },
        latency_ns,
        control_hash: hash,
        trace_manifest,
        trace_archive,
    })
}

fn summarize(role: KartRole, runs: &[&KartRunResult]) -> KartRoleSummary {
    let notes = runs
        .iter()
        .filter_map(|r| {
            r.status.failure_reason().map(|reason| {
                format!(
                    "{} {:?} seed={} [{}]: {} (steps={}, checkpoints={})",
                    role.label(),
                    r.split,
                    r.seed.as_u64(),
                    r.status.label(),
                    reason,
                    r.steps_executed,
                    r.checkpoints_cleared
                )
            })
        })
        .collect();
    let n = runs.len();
    let cps: Vec<f64> = runs.iter().map(|r| r.checkpoints_cleared as f64).collect();
    let eff: Vec<f64> = runs.iter().map(|r| r.efficacy).collect();
    let mean = |v: &[f64]| {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };
    let costs: Vec<f64> = runs.iter().map(|r| r.cost_per_decision).collect();
    KartRoleSummary {
        role,
        is_privileged: role.is_privileged(),
        total_runs: n,
        succeeded_count: runs.iter().filter(|r| r.status.is_succeeded()).count(),
        failed_count: runs.iter().filter(|r| r.status.is_failed()).count(),
        timed_out_count: runs.iter().filter(|r| r.status.is_timed_out()).count(),
        collisions_total: runs.iter().map(|r| r.collisions).sum(),
        viable_trajectories: runs.iter().filter(|r| r.viable_trajectory).count(),
        mean_efficacy: mean(&eff),
        checkpoints_dispersion: DispersionStats::compute(&cps),
        efficacy_dispersion: DispersionStats::compute(&eff),
        mean_cost_per_decision: mean(&costs),
        seeds: runs.iter().map(|r| r.seed.as_u64()).collect(),
        failure_notes: notes,
    }
}

/// Ejecuta la comparación Kart pareada headless. Ningún episodio se descarta del informe.
pub fn run_kart_arena(config: &KartArenaConfig) -> Result<KartArenaReport, ArenaError> {
    run_kart_arena_preregistered(config, None)
}

/// Same as [`run_kart_arena`], stamping the pre-registered hypotheses hash
/// (`gogamen_spec::hypotheses_hash`) into every run's trace manifest before execution.
pub fn run_kart_arena_preregistered(
    config: &KartArenaConfig,
    hypotheses_hash: Option<u64>,
) -> Result<KartArenaReport, ArenaError> {
    config.validate()?;
    let mut outcomes = Vec::new();
    let mut run_counter = 0_u64;
    for (split, seed) in config.seeds.iter_tagged() {
        let mut fair_runs = Vec::new();
        let mut ceiling_runs = Vec::new();
        for spec in &config.roles {
            run_counter += 1;
            let r = run_role(
                config,
                spec,
                split,
                seed,
                RunId::new(run_counter),
                hypotheses_hash,
            )?;
            if spec.role.is_privileged() {
                ceiling_runs.push(r);
            } else {
                fair_runs.push(r);
            }
        }
        outcomes.push(KartSeedOutcome {
            split,
            seed,
            fair_runs,
            ceiling_runs,
        });
    }

    let collect = |privileged: bool| -> Vec<KartRoleSummary> {
        config
            .roles
            .iter()
            .filter(|s| s.role.is_privileged() == privileged)
            .map(|s| {
                let runs: Vec<&KartRunResult> = outcomes
                    .iter()
                    .flat_map(|o| o.fair_runs.iter().chain(&o.ceiling_runs))
                    .filter(|r| r.role == s.role)
                    .collect();
                summarize(s.role, &runs)
            })
            .collect()
    };
    let fair_summaries = collect(false);
    let ceiling_summaries = collect(true);

    Ok(KartArenaReport {
        spec_id: config.spec_id.clone(),
        track_label: config.track.name().to_owned(),
        observation_contract: "KartObservation::FullKinematics+checkpoints_cleared".to_owned(),
        budget: config.budget,
        seeds: config.seeds.clone(),
        outcomes,
        fair_summaries,
        ceiling_summaries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(seeds: (Vec<u64>, Vec<u64>)) -> KartArenaConfig {
        let seeds = ArenaSeedSet::new(
            seeds.0.into_iter().map(MasterSeed::new).collect(),
            seeds.1.into_iter().map(MasterSeed::new).collect(),
        )
        .unwrap();
        KartArenaConfig::new(
            "gog-157-test",
            TrackMap::rectangular_circuit(),
            KartArenaBudget::new(40, 6, 9 * 6 * 2).unwrap(),
            seeds,
        )
    }

    #[test]
    fn gog_157_same_seed_same_report_hash() {
        let c = cfg((vec![7], vec![101]));
        let a = run_kart_arena(&c).unwrap();
        let b = run_kart_arena(&c).unwrap();
        assert_eq!(a.report_hash(), b.report_hash());
        assert_eq!(a.fair_summaries, b.fair_summaries);
    }

    #[test]
    fn gog_157_four_roles_present_and_ceiling_separate() {
        let c = cfg((vec![7], vec![101]));
        let r = run_kart_arena(&c).unwrap();
        let fair: Vec<KartRole> = r.fair_summaries.iter().map(|s| s.role).collect();
        assert_eq!(
            fair,
            vec![
                KartRole::PureEntropic,
                KartRole::GoalOrientedFmc,
                KartRole::Random,
                KartRole::GeometricBaseline
            ]
        );
        assert!(r.fair_summaries.iter().all(|s| !s.is_privileged));
        assert_eq!(r.ceiling_summaries.len(), 1);
        assert!(r.ceiling_summaries[0].is_privileged);
        for o in &r.outcomes {
            assert_eq!(o.fair_runs.len(), 4);
            assert!(o.fair_runs.iter().all(|x| !x.is_privileged));
            assert_eq!(o.ceiling_runs.len(), 1);
            assert!(o.ceiling_runs[0].is_privileged);
            assert!(!o.fair_runs[0].trace_archive.records.is_empty());
        }
        let text = r.format_summary();
        assert!(text.contains("privileged ceiling (not part of the fair comparison)"));
    }

    #[test]
    fn gog_157_equal_budget_enforced() {
        let mut c = cfg((vec![7], vec![101]));
        c.roles[1].population += 1;
        assert!(matches!(
            run_kart_arena(&c),
            Err(ArenaError::UnequalBudget { .. })
        ));
        let mut c = cfg((vec![7], vec![101]));
        c.roles[4].population = 1;
        assert!(matches!(c.validate(), Err(ArenaError::InvalidRoles { .. })));
        // Budget with the fair roles: equal by construction.
        let c = cfg((vec![7], vec![101]));
        for s in c
            .roles
            .iter()
            .filter(|s| s.role.plans() && !s.role.is_privileged())
        {
            assert_eq!(
                s.transitions_per_decision(),
                c.budget.transitions_per_decision
            );
        }
    }

    #[test]
    fn gog_157_failed_episodes_retained_in_report() {
        // Random en pista rectangular: se esperan fallos; el denominador debe quedar intacto.
        let c = cfg((vec![7, 19, 42], vec![101]));
        let r = run_kart_arena(&c).unwrap();
        for s in r.fair_summaries.iter().chain(&r.ceiling_summaries) {
            assert_eq!(s.total_runs, 4);
            assert_eq!(s.seeds.len(), 4);
            assert_eq!(
                s.succeeded_count + s.failed_count + s.timed_out_count,
                s.total_runs
            );
            assert_eq!(s.failure_notes.len(), s.failed_count + s.timed_out_count);
        }
        let non_success: usize = r
            .fair_summaries
            .iter()
            .map(|s| s.failed_count + s.timed_out_count)
            .sum();
        assert!(non_success > 0, "40 ticks cannot complete a lap");
        let count: usize = r
            .outcomes
            .iter()
            .flat_map(|o| &o.fair_runs)
            .filter(|x| !x.status.is_succeeded())
            .count();
        assert_eq!(count, non_success);
    }

    #[test]
    fn gog_157_fair_roles_share_observation_and_transition_cost() {
        let c = cfg((vec![7], vec![]));
        let r = run_kart_arena(&c).unwrap();
        let o = &r.outcomes[0];
        let cap = c.budget.transitions_per_decision as f64;
        for x in &o.fair_runs {
            assert!(x.cost_per_decision <= cap + 1e-9, "{:?}", x.role);
            assert!(x.steps_executed as usize <= c.budget.max_ticks);
        }
        assert!(o.ceiling_runs[0].cost_per_decision <= cap * 4.0 + 1e-9);
    }

    #[test]
    fn gog_157_sample_scenarios_report_all_roles() {
        for track in [TrackMap::rectangular_circuit()] {
            let seeds = ArenaSeedSet::new(
                [7].map(MasterSeed::new).to_vec(),
                [101].map(MasterSeed::new).to_vec(),
            )
            .unwrap();
            let c = KartArenaConfig::new(
                "gog-157-sample",
                track,
                KartArenaBudget::new(400, 8, 288).unwrap(),
                seeds,
            );
            let r = run_kart_arena(&c).unwrap();
            println!("{}", r.format_summary());
            for s in r.fair_summaries.iter().chain(&r.ceiling_summaries) {
                println!(
                    "  {} cps={:?} notes={}",
                    s.role.label(),
                    s.checkpoints_dispersion.mean,
                    s.failure_notes.len()
                );
            }
            assert_eq!(r.fair_summaries.len(), 4);
        }
    }
}
