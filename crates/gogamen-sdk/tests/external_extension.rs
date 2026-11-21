//! Test de integración de extensión de terceros usando únicamente `gogamen_sdk::*`.
//!
//! Demuestra que un consumidor externo puede:
//! 1. Registrar un nuevo `Planner` y un nuevo sensor (`EnvironmentObserver`) estáticamente
//!    al construir su binario, sin modificar ni forkear el código fuente core.
//! 2. Validar compatibilidad de capacidades, esquema y versiones (`CapabilityVersion`) al
//!    registrar, rechazando extensiones incompatibles con errores tipados (`ExtensionError`).
//! 3. Garantizar que no se descargan ni cargan plugins remotos/dinámicos y que declarar
//!    capacidades en el manifiesto no otorga permisos implícitos (`PermissionSet` + `ActionGuard`).

use gogamen_sdk::*;

struct ExternalRangeSensor {
    reading: f64,
}

impl EnvironmentObserver for ExternalRangeSensor {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        Observation::new(Tick::ZERO, SimDuration::ZERO, vec![self.reading]).map_err(|_| {
            EnvironmentError::ObservationFailed {
                reason: "invalid sensor reading",
            }
        })
    }
}

struct ExternalStepModel {
    space: ActionSpace,
}

impl GenerativeModel for ExternalStepModel {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        _rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.space
            .validate(action)
            .map_err(ModelError::TransitionFailed)?;
        let idx = action
            .as_discrete()
            .ok_or(ModelError::TransitionFailed(
                DomainError::IncompatibleActionKind {
                    expected: "discrete",
                    actual: "continuous",
                },
            ))?
            .index();
        let delta = match idx {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
        let next_pos = state.values()[0] + delta;
        let next_state = state
            .advance(dt, vec![next_pos])
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(next_state.tick(), next_state.elapsed(), vec![next_pos])
            .map_err(ModelError::TransitionFailed)?;
        Transition::new(state.tick(), dt, next_state, obs, Terminal::Ongoing)
            .map_err(ModelError::TransitionFailed)
    }
}

struct ExternalGreedyPlanner;

impl Planner for ExternalGreedyPlanner {
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        let state = request.state().ok_or(ModelError::InvalidState {
            reason: "missing state in PlanRequest",
        })?;
        let dt = StepDt::from_secs_f64(0.1).map_err(ModelError::TransitionFailed)?;
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for idx in 0..3 {
            let candidate = Action::discrete_in_space(idx, ctx.model().action_space())
                .map_err(ModelError::TransitionFailed)?;
            let tr = ctx.simulate_step(state, &candidate, dt)?;
            let val = tr.next_state().values()[0];
            if val > best_val {
                best_val = val;
                best_idx = idx;
            }
        }
        let chosen = Action::discrete_in_space(best_idx, ctx.model().action_space())
            .map_err(ModelError::TransitionFailed)?;
        Ok(PlanResult::completed(chosen, 3))
    }
}

#[test]
fn external_binary_registers_and_runs_third_party_planner_and_sensor() {
    let mut registry = StaticExtensionRegistry::new()
        .with_supported_capability("sensor.lidar1d", CapabilityVersion::new(0, 1, 0))
        .with_supported_capability("planner.greedy1d", CapabilityVersion::new(0, 1, 0));

    let sensor_manifest = ExtensionManifest::new(
        "thirdparty.lidar1d",
        ExtensionKind::Sensor,
        ModuleOrigin::ThirdParty,
        "external_lidar_crate",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_capability("sensor.lidar1d", CapabilityVersion::new(0, 1, 0))
    .with_required_permission(Permission::state_read());

    let planner_manifest = ExtensionManifest::new(
        "thirdparty.greedy_planner",
        ExtensionKind::Planner,
        ModuleOrigin::ThirdParty,
        "external_planner_crate",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_capability("planner.greedy1d", CapabilityVersion::new(0, 1, 0))
    .with_required_permission(Permission::state_read());

    registry
        .register_sensor(sensor_manifest, |_| {
            Ok(Box::new(ExternalRangeSensor { reading: 4.25 }))
        })
        .expect("static sensor registration succeeds");

    registry
        .register_planner(planner_manifest, |_| Ok(Box::new(ExternalGreedyPlanner)))
        .expect("static planner registration succeeds");

    assert_eq!(
        registry.registered_ids(),
        vec![
            ExtensionId::new("thirdparty.greedy_planner"),
            ExtensionId::new("thirdparty.lidar1d"),
        ]
    );

    // Declarar capacidades en el manifest NO otorga permisos implícitos.
    let unprivileged_ctx = ExtensionActivationContext::strict_empty();
    assert_eq!(
        registry
            .instantiate_sensor(&ExtensionId::new("thirdparty.lidar1d"), &unprivileged_ctx)
            .err(),
        Some(ExtensionError::InsufficientPermissions {
            id: ExtensionId::new("thirdparty.lidar1d"),
            missing: PermissionSet::from_permissions([Permission::state_read()]),
        })
    );

    // Con permisos explícitos y ActionGuard separado, se instancian y ejecutan.
    let granted_ctx =
        ExtensionActivationContext::with_permissions(PermissionSet::from_permissions([
            Permission::state_read(),
        ]))
        .with_guard_policy(SafeFallbackPolicy::RejectOnly);

    let sensor = registry
        .instantiate_sensor(&ExtensionId::new("thirdparty.lidar1d"), &granted_ctx)
        .expect("sensor instantiates with explicit permissions");
    let obs = sensor.observe().expect("sensor observe succeeds");
    assert_eq!(obs.features(), &[4.25]);

    let mut planner = registry
        .instantiate_planner(&ExtensionId::new("thirdparty.greedy_planner"), &granted_ctx)
        .expect("planner instantiates with explicit permissions");

    let space = ActionSpace::discrete(3).expect("valid space");
    let model = ExternalStepModel {
        space: space.clone(),
    };
    let mut rng = DeterministicRng::new(7);
    let mut planning_ctx = PlanningContext::new(&model as &dyn GenerativeModel, &mut rng);

    let root_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).expect("valid state");
    let request = PlanRequest::from_state(root_state, space.clone(), PlanBudget::new(10, None));

    let result = planner
        .plan_dyn(&mut planning_ctx, &request)
        .expect("planner executes");
    assert_eq!(result.status(), PlanStatus::Completed);
    let action = result
        .selected_action()
        .expect("planner selected an action")
        .clone();
    assert_eq!(action.as_discrete().expect("discrete").index(), 2);

    // ActionGuard independiente valida la propuesta autorizada por separado.
    let guard = granted_ctx.action_guard();
    let proposal = ActionProposal::authorized(action.clone(), Tick::ZERO);
    let guard_ctx = GuardContext::new(
        &space,
        Tick::ZERO,
        std::time::Instant::now(),
        Terminal::Ongoing,
    );
    assert_eq!(
        guard.evaluate(&proposal, &guard_ctx),
        GuardDecision::Allow(action)
    );
}

#[test]
fn external_registry_rejects_incompatible_and_remote_extensions() {
    let mut registry = StaticExtensionRegistry::new()
        .with_supported_capability("sensor.lidar1d", CapabilityVersion::new(0, 1, 0));

    let incompatible = ExtensionManifest::new(
        "thirdparty.future_lidar",
        ExtensionKind::Sensor,
        ModuleOrigin::ThirdParty,
        "future_crate",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_capability("sensor.lidar1d", CapabilityVersion::new(0, 2, 0));

    assert!(matches!(
        registry.register_sensor(incompatible, |_| Ok(Box::new(ExternalRangeSensor {
            reading: 1.0
        }))),
        Err(ExtensionError::IncompatibleVersion { .. })
    ));

    let remote = ExtensionManifest::new(
        "thirdparty.remote_sensor",
        ExtensionKind::Sensor,
        ModuleOrigin::ThirdParty,
        "unused",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_source(ExtensionSource::from_locator(
        "https://example.com/plugin.so",
    ));

    assert!(matches!(
        registry.register_sensor(remote, |_| Ok(Box::new(ExternalRangeSensor {
            reading: 1.0
        }))),
        Err(ExtensionError::RemoteSourceRejected { .. })
    ));
}
