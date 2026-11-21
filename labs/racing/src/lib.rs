//! Laboratorio Kart con física cinemática 2D (`gogamen-lab-racing`, `std`-only).
//!
//! Implementa sin renderer ni UI:
//! - Modelo cinemático de bicicleta 2D ([`KinematicBicycleConfig`], [`KartState`], `dt` fijo con `StepDt`,
//!   control continuo acotado `[steering_angle, throttle_accel]` en `ActionSpace::Continuous`).
//! - Pista 2D con muros, límites de contorno y checkpoints ordenados ([`TrackMap`], [`WallSegment`], [`TrackCheckpoint`]).
//! - [`KartEnv`] (`World` + `LiveEnvironment`) y [`KartModel`] (`GenerativeModel` + `BranchableSnapshot`).
//! - Declaración explícita de aproximaciones en [`MODEL_APPROXIMATIONS_NOTE`].

pub mod autonomous_fmc;
pub mod physics;
pub mod track;
pub mod world;

pub use autonomous_fmc::{
    DEV_SEEDS, KartAutonomousMetrics, KartCandidateFuture, KartDecisionReport, KartEpisodeOutcome,
    KartFmcController, KartFmcControllerConfig, KartObjectiveWeights, KartRunError,
    SimpleTrackBaseline, SimpleTrackBaselineConfig, UNSEEN_SEEDS,
};
pub use physics::{
    DEFAULT_KART_DT_SECS, DEFAULT_KART_RADIUS, DEFAULT_LINEAR_DRAG, DEFAULT_MAX_ACCEL,
    DEFAULT_MAX_SPEED, DEFAULT_MAX_STEERING_ANGLE, DEFAULT_MIN_ACCEL, DEFAULT_MIN_SPEED,
    DEFAULT_WHEELBASE, KartControl, KartDerivatives, KartState, KinematicBicycleConfig,
    MODEL_APPROXIMATIONS_NOTE, compute_bicycle_derivatives, integrate_bicycle_step, wrap_angle,
};
pub use track::{
    Point2, TrackCheckpoint, TrackMap, WallSegment, point_to_segment_distance, segments_intersect,
};
pub use world::{KartEnv, KartModel, MAX_KART_SUBSTEPS, kart_substep_count};

#[cfg(test)]
mod tests {
    use gogamen_core::{
        Action, Actuator, BranchableSnapshot, DomainError, EnvironmentError, GenerativeModel,
        LiveEnvironment, MasterSeed, Terminal, TerminalReason, World,
    };

    use super::*;

    fn manual_control_toward(
        kart: KartState,
        target: Point2,
        config: &KinematicBicycleConfig,
        target_speed: f64,
    ) -> [f64; 2] {
        let dx = target.x - kart.x;
        let dy = target.y - kart.y;
        let desired_yaw = dy.atan2(dx);
        let heading_err = wrap_angle(desired_yaw - kart.yaw);
        let steering =
            (2.0 * heading_err).clamp(-config.max_steering_angle, config.max_steering_angle);
        let accel = (2.0 * (target_speed - kart.speed)).clamp(config.min_accel, config.max_accel);
        [steering, accel]
    }

    #[test]
    fn ac1_manual_kart_completes_circuit_with_checkpoints() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(config.clone(), track.clone()).unwrap();

        assert_eq!(env.terminal(), Terminal::Ongoing);
        assert_eq!(env.checkpoints_cleared(), 0);
        assert!(!env.collided());

        let total_cps = track.checkpoints().len();
        let mut max_steps = 0;

        for _ in 0..800 {
            let cp_idx = env.checkpoints_cleared().min(total_cps - 1);
            let target = track.checkpoints()[cp_idx].center;
            let [steering, accel] = manual_control_toward(env.kart_state(), target, &config, 6.0);
            let action =
                Action::continuous_in_space(vec![steering, accel], Actuator::action_space(&env))
                    .unwrap();
            let outcome = env.apply(&action).unwrap();
            max_steps += 1;

            if outcome.terminal().is_terminal() {
                break;
            }
        }

        assert_eq!(
            env.terminal(),
            Terminal::Terminated(TerminalReason::GoalReached)
        );
        assert_eq!(env.checkpoints_cleared(), total_cps);
        assert!(!env.collided());
        assert!(max_steps < 800);

        // No se permiten pasos adicionales tras alcanzar el objetivo.
        let stop_action =
            Action::continuous_in_space(vec![0.0, 0.0], World::action_space(&env)).unwrap();
        assert_eq!(
            World::step(&mut env, &stop_action, config.dt).unwrap_err(),
            DomainError::AlreadyTerminal
        );

        // Reinicio restaura el episodio activo en la pose inicial.
        let obs = LiveEnvironment::reset(&mut env).unwrap();
        assert_eq!(obs.tick().as_u64(), 0);
        assert_eq!(env.terminal(), Terminal::Ongoing);
        assert_eq!(env.checkpoints_cleared(), 0);
        assert_eq!(env.kart_state(), track.start_pose());
    }

    #[test]
    fn ac1_wall_collision_bounds_and_action_limits_checked() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut env = KartEnv::new(config.clone(), track).unwrap();

        // Verifica rechazo de acciones continuas fuera de los límites declarados.
        let bad_steering = Action::continuous(vec![config.max_steering_angle + 0.5, 0.0]).unwrap();
        assert!(matches!(
            env.apply(&bad_steering),
            Err(EnvironmentError::InvalidAction(
                DomainError::ContinuousActionOutOfRange { .. }
            ))
        ));

        // Acelera recto hacia el muro derecho (x = 40) hasta colisionar.
        let crash_action =
            Action::continuous_in_space(vec![0.0, config.max_accel], Actuator::action_space(&env))
                .unwrap();

        for _ in 0..200 {
            let outcome = env.apply(&crash_action).unwrap();
            if outcome.terminal().is_terminal() {
                break;
            }
        }

        assert_eq!(
            env.terminal(),
            Terminal::Terminated(TerminalReason::Failure)
        );
        assert!(env.collided());
        assert_eq!(
            env.apply(&crash_action).unwrap_err(),
            EnvironmentError::AlreadyTerminal
        );

        // Verifica también detección directa de salida de límites exteriores en TrackMap.
        let out_of_bounds = Point2::new(45.0, 12.0).unwrap();
        assert!(env.track().is_outside_bounds(out_of_bounds, 0.0));
        assert!(env.track().check_collision(
            Point2::new(39.8, 12.0).unwrap(),
            Point2::new(40.1, 12.0).unwrap(),
            config.kart_radius
        ));
    }

    #[test]
    fn ac2_headless_snapshots_fork_restore_and_trajectory_continuity() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let mut model = KartModel::new(config.clone(), track.clone(), MasterSeed::new(7)).unwrap();
        let mut env = KartEnv::new(config.clone(), track).unwrap();

        let accel_straight =
            Action::continuous_in_space(vec![0.0, 4.0], model.action_space()).unwrap();

        // Avanza 15 pasos en ambos (KartModel y KartEnv) y comprueba continuidad y equivalencia.
        let mut prev_pos = Point2 {
            x: model.kart_state().x,
            y: model.kart_state().y,
        };
        let max_step_dist = config.max_speed * config.dt.as_secs_f64() + 1e-9;

        for _ in 0..15 {
            let m_tr = model.step_in_place(&accel_straight).unwrap();
            let e_out = env.apply(&accel_straight).unwrap();
            assert_eq!(
                m_tr.observation().features(),
                e_out.observation().features()
            );

            let cur_pos = Point2 {
                x: model.kart_state().x,
                y: model.kart_state().y,
            };
            assert!(prev_pos.distance_to(cur_pos) <= max_step_dist);
            prev_pos = cur_pos;
        }

        // Captura snapshot en el tick 15 y ramifica en dos trayectorias divergentes.
        let snap = model.snapshot().unwrap();
        let mut left_branch = model.fork().unwrap();
        let mut right_branch = model.fork().unwrap();

        let steer_left = Action::continuous_in_space(vec![0.4, 1.0], model.action_space()).unwrap();
        let steer_right =
            Action::continuous_in_space(vec![-0.4, 1.0], model.action_space()).unwrap();

        let mut expected_left_states = Vec::new();
        for _ in 0..10 {
            let tr = left_branch.step_in_place(&steer_left).unwrap();
            expected_left_states.push(tr.next_state().clone());
            right_branch.step_in_place(&steer_right).unwrap();
        }

        assert!(left_branch.kart_state().y > right_branch.kart_state().y);
        assert!(left_branch.kart_state().yaw > right_branch.kart_state().yaw);
        // El modelo original permanece intacto en el tick 15.
        assert_eq!(model.state().tick().as_u64(), 15);

        // Avanza el modelo original por la rama derecha, luego restaura el snapshot y reproduce la rama izquierda.
        for _ in 0..5 {
            model.step_in_place(&steer_right).unwrap();
        }
        assert_eq!(model.state().tick().as_u64(), 20);

        model.restore(&snap).unwrap();
        assert_eq!(model.state().tick().as_u64(), 15);
        assert_eq!(model.kart_state(), snap.state().values()[0..4].pipe_kart());

        for expected in &expected_left_states {
            let replayed = model.step_in_place(&steer_left).unwrap();
            assert_eq!(replayed.next_state(), expected);
        }
    }

    trait PipeKart {
        fn pipe_kart(&self) -> KartState;
    }

    impl PipeKart for [f64] {
        fn pipe_kart(&self) -> KartState {
            KartState::from_slice(self).unwrap()
        }
    }

    #[test]
    fn ac3_model_declares_approximations_without_complex_tire_friction() {
        let config = KinematicBicycleConfig::default();
        let track = TrackMap::rectangular_circuit();
        let env = KartEnv::new(config.clone(), track.clone()).unwrap();
        let model = KartModel::new(config.clone(), track, MasterSeed::new(1)).unwrap();

        let note = MODEL_APPROXIMATIONS_NOTE;
        assert!(!note.is_empty());
        assert_eq!(config.approximations_note(), note);
        assert_eq!(env.approximations_note(), note);
        assert_eq!(model.approximations_note(), note);

        let lower = note.to_lowercase();
        assert!(lower.contains("cinemático"));
        assert!(lower.contains("sin deslizamiento lateral"));
        assert!(lower.contains("sin transferencia de pesos"));
        assert!(lower.contains("sin fricción compleja"));
    }
}
