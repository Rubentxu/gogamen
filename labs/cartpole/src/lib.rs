//! Laboratorio de péndulo invertido sobre carro (`gogamen-lab-cartpole`, `std`-only).
//!
//! Implementa:
//! - Dinámica física clásica Barto/Sutton CartPole con integradores `Euler`, `SemiImplicitEuler` y `Rk4`.
//! - Límites terminales explícitos (`|x| > x_threshold` o `|theta| > theta_threshold_radians`).
//! - [`CartPoleEnv`] (`World` + `LiveEnvironment`) y [`CartPoleModel`] (`GenerativeModel` + `BranchableSnapshot`).
//! - [`LinearFeedbackController`] como controlador clásico de referencia.
//! - [`run_cartpole_headless`] para ejecución headless reproducible sin renderer ni UI.
//! - [`EntropicProxyController`] y [`run_entropic_comparison`] para el experimento entrópico sin reward angular.

pub mod controller;
pub mod entropic;
pub mod physics;
pub mod world;

pub use controller::{CartPolePolicy, LinearFeedbackController};
pub use entropic::{
    ConditionAggregateMetrics, ConditionSeedMetrics, EPISTEMOLOGICAL_NOTE,
    EntropicComparisonReport, EntropicConditionKind, EntropicExperimentConfig,
    EntropicProxyController, run_entropic_comparison,
};
pub use physics::{
    CartPoleConfig, CartPoleDerivatives, CartPoleKinematics, DEFAULT_DT_SECS, DEFAULT_GRAVITY,
    DEFAULT_HALF_POLE_LENGTH, DEFAULT_MASS_CART, DEFAULT_MASS_POLE, DEFAULT_MAX_FORCE,
    DEFAULT_THETA_THRESHOLD_RADIANS, DEFAULT_X_THRESHOLD, IntegratorKind, compute_derivatives,
    integrate_step,
};
pub use world::{CartPoleEnv, CartPoleEpisodeSummary, CartPoleModel, run_cartpole_headless};

#[cfg(test)]
mod tests {
    use gogamen_core::{
        Action, Actuator, BranchableSnapshot, DomainError, EnvironmentError, GenerativeModel,
        LiveEnvironment, MasterSeed, ModelError, StepDt, Terminal, TerminalReason, World,
    };

    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ac1_step_integration_matches_numerical_fixtures() {
        // Fixture 1: equilibrio inestable exacto en reposo con fuerza nula.
        let cfg_euler = CartPoleConfig::default().with_integrator(IntegratorKind::Euler);
        let dt = StepDt::from_secs_f64(0.02).unwrap();
        let eq = CartPoleKinematics::new(0.0, 0.0, 0.0, 0.0).unwrap();
        let eq_next = integrate_step(&cfg_euler, eq, 0.0, dt).unwrap();
        assert_eq!(eq_next, eq);

        // Fixture 2: paso único Barto/Sutton desde theta=0.1 rad, force=10.0 N, dt=0.02 s.
        // Valores analíticos de referencia:
        // sin(0.1) = 0.09983341664682815, cos(0.1) = 0.9950041652780258
        // temp = 10.0 / 1.1 = 9.090909090909092
        // denom = 0.5 * (4/3 - 0.1 * cos(0.1)^2 / 1.1) = 0.6216651535196895
        // theta_acc = (9.8 * sin(0.1) - cos(0.1) * temp) / denom = -12.976640049102327
        // x_acc = temp - 0.05 * theta_acc * cos(0.1) / 1.1 = 9.677809586371383
        let perturbed = CartPoleKinematics::new(0.0, 0.0, 0.1, 0.0).unwrap();
        let expected_theta_acc = -12.976_640_049_102_327;
        let expected_x_acc = 9.677_809_586_371_383;

        let derivs = compute_derivatives(&cfg_euler, perturbed, 10.0);
        assert!(approx_eq(derivs.theta_acc, expected_theta_acc, 1e-12));
        assert!(approx_eq(derivs.x_acc, expected_x_acc, 1e-12));

        let next_euler = integrate_step(&cfg_euler, perturbed, 10.0, dt).unwrap();
        assert!(approx_eq(next_euler.x, 0.0, 1e-12));
        assert!(approx_eq(next_euler.x_dot, 0.02 * expected_x_acc, 1e-12));
        assert!(approx_eq(next_euler.theta, 0.1, 1e-12));
        assert!(approx_eq(
            next_euler.theta_dot,
            0.02 * expected_theta_acc,
            1e-12
        ));

        let cfg_semi = CartPoleConfig::default().with_integrator(IntegratorKind::SemiImplicitEuler);
        let next_semi = integrate_step(&cfg_semi, perturbed, 10.0, dt).unwrap();
        let expected_x_dot_semi = 0.02 * expected_x_acc;
        let expected_theta_dot_semi = 0.02 * expected_theta_acc;
        assert!(approx_eq(next_semi.x_dot, expected_x_dot_semi, 1e-12));
        assert!(approx_eq(next_semi.x, 0.02 * expected_x_dot_semi, 1e-12));
        assert!(approx_eq(
            next_semi.theta_dot,
            expected_theta_dot_semi,
            1e-12
        ));
        assert!(approx_eq(
            next_semi.theta,
            0.1 + 0.02 * expected_theta_dot_semi,
            1e-12
        ));

        // Fixture 3: integrador RK4 mantiene consistencia y orden superior en caída libre pequeña.
        let cfg_rk4 = CartPoleConfig::default().with_integrator(IntegratorKind::Rk4);
        let next_rk4 = integrate_step(&cfg_rk4, perturbed, 10.0, dt).unwrap();
        assert!(next_rk4.theta < 0.1);
        assert!(next_rk4.x > 0.0);
    }

    #[test]
    fn ac1_terminal_conditions_trigger_failure_and_reject_further_steps() {
        let dt = StepDt::from_secs_f64(0.02).unwrap();

        // Caso 1: cruce de límite angular |theta| > theta_threshold_radians.
        let near_angle_limit =
            CartPoleKinematics::new(0.0, 0.0, DEFAULT_THETA_THRESHOLD_RADIANS - 0.001, 1.0)
                .unwrap();
        let mut env =
            CartPoleEnv::new(CartPoleConfig::default().with_initial_state(near_angle_limit))
                .unwrap();
        let zero_action =
            Action::continuous_in_space(vec![0.0], World::action_space(&env)).unwrap();
        let tr = World::step(&mut env, &zero_action, dt).unwrap();
        assert_eq!(tr.terminal(), Terminal::Terminated(TerminalReason::Failure));
        assert!(env.kinematics().theta.abs() > DEFAULT_THETA_THRESHOLD_RADIANS);

        let err = World::step(&mut env, &zero_action, dt).unwrap_err();
        assert_eq!(err, DomainError::AlreadyTerminal);

        // Caso 2: cruce de límite de carril |x| > x_threshold.
        let near_x_limit =
            CartPoleKinematics::new(DEFAULT_X_THRESHOLD - 0.005, 1.0, 0.0, 0.0).unwrap();
        let mut env_x =
            CartPoleEnv::new(CartPoleConfig::default().with_initial_state(near_x_limit)).unwrap();
        let outcome = env_x.apply(&zero_action).unwrap();
        assert_eq!(
            outcome.terminal(),
            Terminal::Terminated(TerminalReason::Failure)
        );
        assert_eq!(
            env_x.apply(&zero_action).unwrap_err(),
            EnvironmentError::AlreadyTerminal
        );

        // Reinicio restaura episodio activo.
        let reset_obs = LiveEnvironment::reset(&mut env_x).unwrap();
        assert_eq!(reset_obs.tick().as_u64(), 0);
        assert_eq!(env_x.terminal(), Terminal::Ongoing);
    }

    #[test]
    fn ac2_classical_controller_stabilizes_perturbed_scenarios() {
        let scenarios = [
            CartPoleKinematics::new(0.25, -0.1, 0.12, 0.15).unwrap(),
            CartPoleKinematics::new(-0.3, 0.15, -0.14, -0.1).unwrap(),
        ];

        for init in scenarios {
            let cfg = CartPoleConfig::default().with_initial_state(init);
            let mut controller = LinearFeedbackController::default_stabilizing();
            let summary =
                run_cartpole_headless(cfg.clone(), 300, MasterSeed::new(42), &mut controller)
                    .unwrap();

            assert_eq!(summary.ticks_executed, 300);
            assert_eq!(summary.terminal, Terminal::Ongoing);
            assert!(summary.max_abs_x <= cfg.x_threshold);
            assert!(summary.max_abs_theta <= cfg.theta_threshold_radians);
            assert!(
                summary.final_kinematics.theta.abs() < 0.02,
                "pole should converge near upright, got theta={}",
                summary.final_kinematics.theta
            );
            assert!(
                summary.final_kinematics.x.abs() < 0.25,
                "cart should remain near center, got x={}",
                summary.final_kinematics.x
            );
        }

        // Sin control (fuerza 0) desde perturbación pequeña, el péndulo cae antes de 200 ticks.
        struct ZeroPolicy;
        impl CartPolePolicy for ZeroPolicy {
            fn select_action(
                &mut self,
                _observation: &gogamen_core::Observation,
                action_space: &gogamen_core::ActionSpace,
            ) -> Result<Action, DomainError> {
                Action::continuous_in_space(vec![0.0], action_space)
            }
        }

        let cfg = CartPoleConfig::default()
            .with_initial_state(CartPoleKinematics::new(0.0, 0.0, 0.05, 0.0).unwrap());
        let mut zero = ZeroPolicy;
        let uncontrolled = run_cartpole_headless(cfg, 200, MasterSeed::new(42), &mut zero).unwrap();
        assert_eq!(
            uncontrolled.terminal,
            Terminal::Terminated(TerminalReason::Failure)
        );
        assert!(uncontrolled.ticks_executed < 200);
    }

    #[test]
    fn ac3_snapshot_fork_restore_and_headless_are_reproducible_without_renderer() {
        let cfg = CartPoleConfig::default()
            .with_initial_state(CartPoleKinematics::new(0.1, 0.0, 0.08, 0.0).unwrap());
        let mut model = CartPoleModel::new(cfg.clone(), MasterSeed::new(7)).unwrap();
        let push_right =
            Action::continuous_in_space(vec![6.0], GenerativeModel::action_space(&model)).unwrap();
        let push_left =
            Action::continuous_in_space(vec![-6.0], GenerativeModel::action_space(&model)).unwrap();

        // Avanzamos 3 pasos y capturamos snapshot.
        for _ in 0..3 {
            model.step_in_place(&push_right).unwrap();
        }
        let snap = model.snapshot().unwrap();
        let expected_kin = model.kinematics();

        // Bifurcación aislada: avanzar el fork con otra acción no contamina `model`.
        let mut branch = model.fork().unwrap();
        branch.step_in_place(&push_left).unwrap();
        assert_eq!(model.kinematics(), expected_kin);
        assert_ne!(branch.kinematics(), expected_kin);

        // Avanzar `model` y restaurar `snap` reproduce exactamente la transición siguiente.
        let expected_next = model.step_in_place(&push_right).unwrap();
        model.step_in_place(&push_left).unwrap();
        model.restore(&snap).unwrap();
        assert_eq!(model.kinematics(), expected_kin);
        let replayed_next = model.step_in_place(&push_right).unwrap();
        assert_eq!(replayed_next, expected_next);

        // El modelo rechaza simular desde un estado terminal.
        let terminal_state = gogamen_core::State::new(
            gogamen_core::Tick::ZERO,
            gogamen_core::SimDuration::ZERO,
            vec![0.0, 0.0, DEFAULT_THETA_THRESHOLD_RADIANS + 0.1, 0.0],
        )
        .unwrap();
        let mut rng = gogamen_core::DeterministicRng::new(1);
        let dt = StepDt::from_secs_f64(0.02).unwrap();
        assert_eq!(
            GenerativeModel::step(&model, &terminal_state, &push_right, dt, &mut rng).unwrap_err(),
            ModelError::TerminalState
        );

        // Ejecución headless determinista reproducible.
        let mut ctrl_a = LinearFeedbackController::default_stabilizing();
        let mut ctrl_b = LinearFeedbackController::default_stabilizing();
        let run_a =
            run_cartpole_headless(cfg.clone(), 200, MasterSeed::new(99), &mut ctrl_a).unwrap();
        let run_b = run_cartpole_headless(cfg, 200, MasterSeed::new(99), &mut ctrl_b).unwrap();
        assert_eq!(run_a, run_b);
        assert_eq!(run_a.trace_hash_hex(), run_b.trace_hash_hex());
    }
}
