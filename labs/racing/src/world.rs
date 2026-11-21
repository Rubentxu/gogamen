//! Entorno vivo (`KartEnv`) y modelo generativo ramificable (`KartModel`) para el laboratorio Kart.

use gogamen_core::{
    Action, ActionBounds, ActionSpace, Actuator, ApplyOutcome, BranchableSnapshot, DomainError,
    EnvironmentError, EnvironmentObserver, GenerativeModel, LiveEnvironment, MasterSeed,
    ModelError, Observation, RngSource, SimDuration, SimulationSnapshot, SnapshotError, State,
    StepDt, StreamKey, Terminal, TerminalReason, Tick, Transition, World,
};

use crate::physics::{
    KartControl, KartState, KinematicBicycleConfig, MODEL_APPROXIMATIONS_NOTE,
    integrate_bicycle_step,
};
use crate::track::{Point2, TrackMap};

fn build_action_space(config: &KinematicBicycleConfig) -> Result<ActionSpace, DomainError> {
    let steering_bounds = ActionBounds::new(-config.max_steering_angle, config.max_steering_angle)?;
    let accel_bounds = ActionBounds::new(config.min_accel, config.max_accel)?;
    ActionSpace::continuous(vec![steering_bounds, accel_bounds])
}

fn extract_control(
    action_space: &ActionSpace,
    action: &Action,
) -> Result<KartControl, DomainError> {
    action_space.validate(action)?;
    let values = action
        .as_continuous()
        .ok_or(DomainError::IncompatibleActionKind {
            expected: "continuous",
            actual: "discrete",
        })?;
    if values.len() != 2 {
        return Err(DomainError::ActionDimensionMismatch {
            expected: 2,
            actual: values.len(),
        });
    }
    Ok(KartControl {
        steering_angle: values[0],
        throttle_accel: values[1],
    })
}

fn encode_state_values(kinematics: KartState, checkpoints_cleared: usize) -> Vec<f64> {
    vec![
        kinematics.x,
        kinematics.y,
        kinematics.yaw,
        kinematics.speed,
        checkpoints_cleared as f64,
    ]
}

fn decode_state_values(
    values: &[f64],
    total_checkpoints: usize,
) -> Result<(KartState, usize), DomainError> {
    if values.len() != 5 {
        return Err(DomainError::InvalidTransition {
            reason: "kart state vector must have 5 elements [x, y, yaw, speed, checkpoints_cleared]",
        });
    }
    let kinematics = KartState::from_slice(&values[0..4])?;
    let cp_raw = values[4];
    if !cp_raw.is_finite() || cp_raw < 0.0 {
        return Err(DomainError::InvalidTransition {
            reason: "checkpoints_cleared must be finite and >= 0",
        });
    }
    let cp_rounded = cp_raw.round();
    if (cp_raw - cp_rounded).abs() > 1e-9 {
        return Err(DomainError::InvalidTransition {
            reason: "checkpoints_cleared must be an integer value",
        });
    }
    let cp_count = cp_rounded as usize;
    if cp_count > total_checkpoints {
        return Err(DomainError::InvalidTransition {
            reason: "checkpoints_cleared exceeds total track checkpoints",
        });
    }
    Ok((kinematics, cp_count))
}

struct StepResult {
    next_kinematics: KartState,
    next_checkpoints: usize,
    terminal: Terminal,
    collided: bool,
}

/// Número máximo de sub-pasos internos por tick (acota el coste de los rollouts).
pub const MAX_KART_SUBSTEPS: usize = 8;

/// Número determinista de sub-pasos para un tick: `ceil(|v|*dt / (0.5*kart_radius))` en `[1, max]`.
///
/// Garantiza que cada sub-paso desplace como mucho medio radio del kart (salvo saturación en `max`),
/// de modo que el barrido de colisión por sub-paso no pueda saltar muros ni cortar esquinas.
pub fn kart_substep_count(
    config: &KinematicBicycleConfig,
    state: KartState,
    dt: StepDt,
    max_substeps: usize,
) -> usize {
    let travel = state.speed.abs() * dt.as_secs_f64();
    let raw = (travel / (0.5 * config.kart_radius)).ceil();
    if !raw.is_finite() {
        return max_substeps.max(1);
    }
    (raw as usize).clamp(1, max_substeps.max(1))
}

fn advance_kart_step(
    config: &KinematicBicycleConfig,
    track: &TrackMap,
    prev_kinematics: KartState,
    prev_checkpoints: usize,
    control: KartControl,
    dt: StepDt,
) -> Result<StepResult, DomainError> {
    advance_kart_step_with_max_substeps(
        config,
        track,
        prev_kinematics,
        prev_checkpoints,
        control,
        dt,
        MAX_KART_SUBSTEPS,
    )
}

/// Igual que el paso estándar pero con tope configurable de sub-pasos
/// (`max_substeps = 1` reproduce el camino legado de un único RK4 con colisión por cuerda).
fn advance_kart_step_with_max_substeps(
    config: &KinematicBicycleConfig,
    track: &TrackMap,
    prev_kinematics: KartState,
    prev_checkpoints: usize,
    control: KartControl,
    dt: StepDt,
    max_substeps: usize,
) -> Result<StepResult, DomainError> {
    let n = kart_substep_count(config, prev_kinematics, dt, max_substeps);
    let sub_dt = if n == 1 {
        dt
    } else {
        StepDt::from_secs_f64(dt.as_secs_f64() / n as f64)?
    };

    let mut current = prev_kinematics;
    let mut next_checkpoints = prev_checkpoints;
    for _ in 0..n {
        let next = integrate_bicycle_step(config, current, control, sub_dt)?;
        let prev_pt = Point2 {
            x: current.x,
            y: current.y,
        };
        let next_pt = Point2 {
            x: next.x,
            y: next.y,
        };

        if track.check_collision(prev_pt, next_pt, config.kart_radius) {
            return Ok(StepResult {
                next_kinematics: next,
                next_checkpoints: prev_checkpoints,
                terminal: Terminal::Terminated(TerminalReason::Failure),
                collided: true,
            });
        }

        if next_checkpoints < track.checkpoints().len() {
            let target = track.checkpoints()[next_checkpoints];
            if target.cleared_by_sweep(prev_pt, next_pt) {
                next_checkpoints += 1;
            }
        }
        current = next;
    }

    let terminal = if next_checkpoints >= track.checkpoints().len() {
        Terminal::Terminated(TerminalReason::GoalReached)
    } else {
        Terminal::Ongoing
    };

    Ok(StepResult {
        next_kinematics: current,
        next_checkpoints,
        terminal,
        collided: false,
    })
}

/// Entorno vivo del laboratorio Kart (`World` + `LiveEnvironment`), sin renderer ni UI.
#[derive(Debug, Clone)]
pub struct KartEnv {
    config: KinematicBicycleConfig,
    track: TrackMap,
    action_space: ActionSpace,
    kinematics: KartState,
    checkpoints_cleared: usize,
    collided: bool,
    tick: Tick,
    elapsed: SimDuration,
    terminal: Terminal,
}

impl KartEnv {
    /// Crea un nuevo entorno [`KartEnv`] inicializado en `track.start_pose()`.
    pub fn new(config: KinematicBicycleConfig, track: TrackMap) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = build_action_space(&config)?;
        let kinematics = track.start_pose();
        let start_pt = Point2 {
            x: kinematics.x,
            y: kinematics.y,
        };
        if track.check_collision(start_pt, start_pt, config.kart_radius) {
            return Err(DomainError::InvalidTransition {
                reason: "track start_pose collides with track walls or bounds",
            });
        }
        Ok(Self {
            config,
            track,
            action_space,
            kinematics,
            checkpoints_cleared: 0,
            collided: false,
            tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
            terminal: Terminal::Ongoing,
        })
    }

    /// Devuelve la declaración explícita de aproximaciones del modelo físico.
    pub const fn approximations_note(&self) -> &'static str {
        MODEL_APPROXIMATIONS_NOTE
    }

    /// Devuelve la configuración física del kart.
    pub const fn config(&self) -> &KinematicBicycleConfig {
        &self.config
    }

    /// Devuelve el mapa de pista 2D asociado.
    pub const fn track(&self) -> &TrackMap {
        &self.track
    }

    /// Devuelve el estado cinemático actual `(x, y, yaw, speed)`.
    pub const fn kart_state(&self) -> KartState {
        self.kinematics
    }

    /// Devuelve el número de checkpoints superados en orden.
    pub const fn checkpoints_cleared(&self) -> usize {
        self.checkpoints_cleared
    }

    /// Indica si el kart ha colisionado contra muros o límites de pista.
    pub const fn collided(&self) -> bool {
        self.collided
    }

    /// Devuelve el tick lógico actual del episodio.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve la duración simulada acumulada en el episodio.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Devuelve el estado de terminación actual del episodio.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    fn current_state(&self) -> Result<State, DomainError> {
        State::new(
            self.tick,
            self.elapsed,
            encode_state_values(self.kinematics, self.checkpoints_cleared),
        )
    }

    fn reset_internal(&mut self) {
        self.kinematics = self.track.start_pose();
        self.checkpoints_cleared = 0;
        self.collided = false;
        self.tick = Tick::ZERO;
        self.elapsed = SimDuration::ZERO;
        self.terminal = Terminal::Ongoing;
    }
}

impl World for KartEnv {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn reset(&mut self) -> Result<State, DomainError> {
        self.reset_internal();
        self.current_state()
    }

    fn observe(&self, state: &State) -> Result<Observation, DomainError> {
        let (kinematics, cp) = decode_state_values(state.values(), self.track.checkpoints().len())?;
        Observation::new(
            state.tick(),
            state.elapsed(),
            encode_state_values(kinematics, cp),
        )
    }

    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        if self.terminal.is_terminal() {
            return Err(DomainError::AlreadyTerminal);
        }
        let control = extract_control(&self.action_space, action)?;
        let prev_tick = self.tick;
        let step_res = advance_kart_step(
            &self.config,
            &self.track,
            self.kinematics,
            self.checkpoints_cleared,
            control,
            dt,
        )?;
        let next_tick = self.tick.next()?;
        let next_elapsed = self.elapsed.checked_add_dt(dt)?;
        let next_state = State::new(
            next_tick,
            next_elapsed,
            encode_state_values(step_res.next_kinematics, step_res.next_checkpoints),
        )?;
        let obs = World::observe(self, &next_state)?;
        let transition = Transition::new(prev_tick, dt, next_state, obs, step_res.terminal)?;

        self.kinematics = step_res.next_kinematics;
        self.checkpoints_cleared = step_res.next_checkpoints;
        self.collided = step_res.collided;
        self.tick = next_tick;
        self.elapsed = next_elapsed;
        self.terminal = step_res.terminal;

        Ok(transition)
    }
}

impl EnvironmentObserver for KartEnv {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        Observation::new(
            self.tick,
            self.elapsed,
            encode_state_values(self.kinematics, self.checkpoints_cleared),
        )
        .map_err(|_| EnvironmentError::ObservationFailed {
            reason: "kart observation contains non-finite values",
        })
    }
}

impl Actuator for KartEnv {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        let transition = World::step(self, action, self.config.dt).map_err(|err| match err {
            DomainError::AlreadyTerminal => EnvironmentError::AlreadyTerminal,
            DomainError::DiscreteActionOutOfRange { .. }
            | DomainError::ContinuousActionOutOfRange { .. }
            | DomainError::ActionDimensionMismatch { .. }
            | DomainError::IncompatibleActionKind { .. } => EnvironmentError::InvalidAction(err),
            _ => EnvironmentError::ActuationFailed {
                reason: "failed to integrate kart step",
            },
        })?;
        Ok(ApplyOutcome::new(
            transition.observation().clone(),
            transition.terminal(),
        ))
    }
}

impl LiveEnvironment for KartEnv {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        self.reset_internal();
        EnvironmentObserver::observe(self)
    }
}

/// Modelo generativo puro y ramificable de Kart (`GenerativeModel` + `BranchableSnapshot`).
#[derive(Debug, Clone)]
pub struct KartModel {
    config: KinematicBicycleConfig,
    track: TrackMap,
    action_space: ActionSpace,
    state: State,
    rng: gogamen_core::DeterministicRng,
    terminal: Terminal,
}

impl KartModel {
    /// Crea un nuevo [`KartModel`] inicializado en `track.start_pose()`.
    pub fn new(
        config: KinematicBicycleConfig,
        track: TrackMap,
        seed: MasterSeed,
    ) -> Result<Self, DomainError> {
        config.validate()?;
        let action_space = build_action_space(&config)?;
        let start = track.start_pose();
        let start_pt = Point2 {
            x: start.x,
            y: start.y,
        };
        if track.check_collision(start_pt, start_pt, config.kart_radius) {
            return Err(DomainError::InvalidTransition {
                reason: "track start_pose collides with track walls or bounds",
            });
        }
        let state = State::new(Tick::ZERO, SimDuration::ZERO, encode_state_values(start, 0))?;
        Ok(Self {
            config,
            track,
            action_space,
            state,
            rng: seed.derive(StreamKey::Episode(0)),
            terminal: Terminal::Ongoing,
        })
    }

    /// Devuelve la declaración explícita de aproximaciones del modelo físico.
    pub const fn approximations_note(&self) -> &'static str {
        MODEL_APPROXIMATIONS_NOTE
    }

    /// Devuelve la configuración física del modelo.
    pub const fn config(&self) -> &KinematicBicycleConfig {
        &self.config
    }

    /// Devuelve el mapa de pista 2D asociado.
    pub const fn track(&self) -> &TrackMap {
        &self.track
    }

    /// Devuelve el [`State`] interno actual del modelo.
    pub const fn state(&self) -> &State {
        &self.state
    }

    /// Devuelve el estado cinemático actual decodificado del estado interno.
    pub fn kart_state(&self) -> KartState {
        decode_state_values(self.state.values(), self.track.checkpoints().len())
            .expect("internal kart model state is valid")
            .0
    }

    /// Devuelve el número de checkpoints superados en el estado interno actual.
    pub fn checkpoints_cleared(&self) -> usize {
        decode_state_values(self.state.values(), self.track.checkpoints().len())
            .expect("internal kart model state is valid")
            .1
    }

    /// Devuelve el estado terminal interno del modelo.
    pub const fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Reinicia el estado interno del modelo a `track.start_pose()`.
    pub fn reset(&mut self) -> Result<&State, DomainError> {
        self.state = State::new(
            Tick::ZERO,
            SimDuration::ZERO,
            encode_state_values(self.track.start_pose(), 0),
        )?;
        self.terminal = Terminal::Ongoing;
        Ok(&self.state)
    }

    /// Avanza el estado interno del modelo un paso usando `self.config.dt`.
    pub fn step_in_place(&mut self, action: &Action) -> Result<Transition, ModelError> {
        if self.terminal.is_terminal() {
            return Err(ModelError::TerminalState);
        }
        let mut rng = self.rng.clone();
        let transition =
            GenerativeModel::step(self, &self.state, action, self.config.dt, &mut rng)?;
        self.rng = rng;
        self.state = transition.next_state().clone();
        self.terminal = transition.terminal();
        Ok(transition)
    }
}

impl GenerativeModel for KartModel {
    fn action_space(&self) -> &ActionSpace {
        &self.action_space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        _rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        let control =
            extract_control(&self.action_space, action).map_err(ModelError::InvalidAction)?;
        let (prev_kinematics, prev_checkpoints) =
            decode_state_values(state.values(), self.track.checkpoints().len()).map_err(|_| {
                ModelError::InvalidState {
                    reason: "kart state must have 5 finite elements [x, y, yaw, speed, checkpoints_cleared]",
                }
            })?;
        let prev_pt = Point2 {
            x: prev_kinematics.x,
            y: prev_kinematics.y,
        };
        if self
            .track
            .check_collision(prev_pt, prev_pt, self.config.kart_radius)
            || prev_checkpoints >= self.track.checkpoints().len()
        {
            return Err(ModelError::TerminalState);
        }

        let step_res = advance_kart_step(
            &self.config,
            &self.track,
            prev_kinematics,
            prev_checkpoints,
            control,
            dt,
        )
        .map_err(ModelError::TransitionFailed)?;

        let next_values = encode_state_values(step_res.next_kinematics, step_res.next_checkpoints);
        let next_state = state
            .advance(dt, next_values.clone())
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(next_state.tick(), next_state.elapsed(), next_values)
            .map_err(ModelError::TransitionFailed)?;

        Transition::new(state.tick(), dt, next_state, obs, step_res.terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

impl BranchableSnapshot for KartModel {
    fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
        Ok(SimulationSnapshot::new(
            self.state.clone(),
            &self.rng,
            self.terminal,
        ))
    }

    fn fork(&self) -> Result<Self, SnapshotError> {
        Ok(self.clone())
    }

    fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
        snapshot.verify()?;
        decode_state_values(snapshot.state().values(), self.track.checkpoints().len()).map_err(
            |_| {
                SnapshotError::Domain(DomainError::InvalidTransition {
                    reason: "snapshot state is incompatible with kart model",
                })
            },
        )?;
        self.state = snapshot.state().clone();
        self.rng = snapshot.rng();
        self.terminal = snapshot.terminal();
        Ok(())
    }
}

#[cfg(test)]
mod substep_tests {
    use super::*;
    use crate::track::{TrackCheckpoint, WallSegment};

    const TURN: KartControl = KartControl {
        steering_angle: 0.6,
        throttle_accel: 0.0,
    };

    fn fast_start() -> KartState {
        KartState {
            x: 20.0,
            y: 20.0,
            yaw: 0.0,
            speed: 18.0,
        }
    }

    /// Pista con un unico "poste" (muro diminuto) a 0.40 m del punto medio del arco RK4
    /// por el lado exterior de la curva: dentro del radio del kart (0.45) pero a > 0.45 m de la cuerda.
    fn corner_track(config: &KinematicBicycleConfig) -> TrackMap {
        let start = fast_start();
        let half = StepDt::from_secs_f64(config.dt.as_secs_f64() / 2.0).unwrap();
        let mid = integrate_bicycle_step(config, start, TURN, half).unwrap();
        // Giro a la izquierda: exterior = lado derecho del heading.
        let (nx, ny) = (mid.yaw.sin(), -mid.yaw.cos());
        let post = Point2 {
            x: mid.x + 0.40 * nx,
            y: mid.y + 0.40 * ny,
        };
        let post_end = Point2 {
            x: post.x + 0.001 * mid.yaw.cos(),
            y: post.y + 0.001 * mid.yaw.sin(),
        };
        TrackMap::new(
            "corner-post",
            start,
            Point2 { x: 0.0, y: 0.0 },
            Point2 { x: 60.0, y: 60.0 },
            vec![WallSegment::new(post, post_end).unwrap()],
            vec![TrackCheckpoint::new(0, Point2 { x: 50.0, y: 50.0 }, 1.0).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn ac1_single_step_clips_corner_but_substepping_detects_it() {
        let config = KinematicBicycleConfig::default();
        assert!(18.0 * config.dt.as_secs_f64() > config.kart_radius);
        let track = corner_track(&config);

        let legacy = advance_kart_step_with_max_substeps(
            &config,
            &track,
            fast_start(),
            0,
            TURN,
            config.dt,
            1,
        )
        .unwrap();
        assert!(
            !legacy.collided,
            "legacy N=1 chord check must miss the post"
        );

        let n = kart_substep_count(&config, fast_start(), config.dt, MAX_KART_SUBSTEPS);
        assert!(n >= 2);
        let adaptive =
            advance_kart_step(&config, &track, fast_start(), 0, TURN, config.dt).unwrap();
        assert!(adaptive.collided, "sub-stepped path must detect the post");
        assert_eq!(
            adaptive.terminal,
            Terminal::Terminated(TerminalReason::Failure)
        );
    }

    #[test]
    fn ac1_substepped_path_keeps_kart_clear_of_walls_every_tick() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::chicane_circuit();
        let mut s = track.start_pose();
        s.speed = 18.0;
        let c = KartControl {
            steering_angle: 0.6,
            throttle_accel: config.max_accel,
        };
        let mut cp = 0;
        for _ in 0..200 {
            let r = advance_kart_step(&config, &track, s, cp, c, config.dt).unwrap();
            if r.collided || r.terminal.is_terminal() {
                break;
            }
            let p = Point2 {
                x: r.next_kinematics.x,
                y: r.next_kinematics.y,
            };
            for w in track.walls() {
                assert!(w.distance_to_point(p) > config.kart_radius);
            }
            assert!(!track.is_outside_bounds(p, config.kart_radius));
            s = r.next_kinematics;
            cp = r.next_checkpoints;
        }
    }

    #[test]
    fn slow_kart_uses_single_substep_bit_identical_to_legacy() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut s = track.start_pose();
        s.speed = 4.0; // 4 * 0.05 = 0.2 <= 0.5 * 0.45
        assert_eq!(
            kart_substep_count(&config, s, config.dt, MAX_KART_SUBSTEPS),
            1
        );
        let c = KartControl {
            steering_angle: 0.2,
            throttle_accel: 1.0,
        };
        let a = advance_kart_step(&config, &track, s, 0, c, config.dt).unwrap();
        let b =
            advance_kart_step_with_max_substeps(&config, &track, s, 0, c, config.dt, 1).unwrap();
        let direct = integrate_bicycle_step(&config, s, c, config.dt).unwrap();
        assert_eq!(a.next_kinematics, b.next_kinematics);
        assert_eq!(a.next_kinematics, direct);
    }

    #[test]
    fn substep_count_is_clamped() {
        let config = KinematicBicycleConfig::default();
        let mut s = fast_start();
        assert_eq!(
            kart_substep_count(&config, s, config.dt, MAX_KART_SUBSTEPS),
            4
        );
        s.speed = 1.0e6;
        assert_eq!(
            kart_substep_count(&config, s, config.dt, MAX_KART_SUBSTEPS),
            MAX_KART_SUBSTEPS
        );
        s.speed = 0.0;
        assert_eq!(
            kart_substep_count(&config, s, config.dt, MAX_KART_SUBSTEPS),
            1
        );
    }

    #[test]
    fn ac3_same_seed_yields_identical_trajectories() {
        let run = || {
            let config = KinematicBicycleConfig::default();
            let mut m = KartModel::new(
                config.clone(),
                TrackMap::chicane_circuit(),
                MasterSeed::new(42),
            )
            .unwrap();
            let a =
                Action::continuous_in_space(vec![0.3, config.max_accel], m.action_space()).unwrap();
            let mut out = Vec::new();
            for _ in 0..60 {
                if m.step_in_place(&a).is_err() {
                    break;
                }
                out.push(m.kart_state().to_vec());
            }
            out
        };
        let (r1, r2) = (run(), run());
        assert!(!r1.is_empty());
        assert_eq!(r1, r2);
    }
}
