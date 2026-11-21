//! Test de integración de consumidor externo usando únicamente `gogamen_sdk::*`.
//!
//! Demuestra que un proyecto externo puede:
//! 1. Implementar su propio `World`, `LiveEnvironment`, `GenerativeModel`, `BranchableSnapshot`
//!    y `Planner` usando solo `gogamen_sdk::*` (sin `labs/*` ni Jev).
//! 2. Ejecutar su mundo y planificador con `HeadlessRuntime`.
//! 3. Verificar determinismo local, aislamiento `fork`, presupuesto `PlanBudget` y rechazo
//!    de acción con `ActionGuard` mediante `gogamen_sdk::conformance`.
//! 4. Autorar, serializar, compilar y ejecutar un sistema declarativo con `SystemSpec`,
//!    `SpecCompiler` y `PlanRuntimeEngine`.

use std::sync::Arc;

use gogamen_sdk::*;

#[derive(Clone)]
struct BeaconCartWorld {
    space: ActionSpace,
    state: State,
    rng: DeterministicRng,
    dt: StepDt,
    goal_x: f64,
    terminal: Terminal,
}

impl BeaconCartWorld {
    fn new(seed: MasterSeed, dt: StepDt) -> Self {
        let space = ActionSpace::discrete(3).expect("valid discrete action space");
        let state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).expect("finite state");
        Self {
            space,
            state,
            rng: DeterministicRng::new(seed.as_u64()),
            dt,
            goal_x: 3.0,
            terminal: Terminal::Ongoing,
        }
    }

    fn compute_reward(pos: f64, goal_x: f64) -> Reward {
        -(goal_x - pos).abs()
    }

    fn step_state(
        space: &ActionSpace,
        goal_x: f64,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, DomainError> {
        space.validate(action)?;
        let idx = action
            .as_discrete()
            .ok_or(DomainError::IncompatibleActionKind {
                expected: "discrete",
                actual: "continuous",
            })?
            .index();
        let dir = match idx {
            0 => -0.5,
            1 => 0.0,
            2 => 0.5,
            _ => unreachable!("validated against cardinality 3"),
        };
        let jitter = (rng.next_f64() - 0.5) * 0.02;
        let next_pos = state.values()[0] + dir + jitter;
        let next_state = state.advance(dt, vec![next_pos])?;
        let reward = Self::compute_reward(next_pos, goal_x);
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            vec![next_pos, reward],
        )?;
        let terminal = if (next_pos - goal_x).abs() < 0.25 {
            Terminal::Terminated(TerminalReason::GoalReached)
        } else {
            Terminal::Ongoing
        };
        Transition::new(state.tick(), dt, next_state, obs, terminal)
    }
}

impl World for BeaconCartWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self) -> Result<State, DomainError> {
        self.state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0])?;
        self.terminal = Terminal::Ongoing;
        Ok(self.state.clone())
    }

    fn observe(&self, state: &State) -> Result<Observation, DomainError> {
        let pos = state.values()[0];
        let reward = Self::compute_reward(pos, self.goal_x);
        Observation::new(state.tick(), state.elapsed(), vec![pos, reward])
    }

    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        if self.terminal.is_terminal() {
            return Err(DomainError::AlreadyTerminal);
        }
        let tr = Self::step_state(
            &self.space,
            self.goal_x,
            &self.state,
            action,
            dt,
            &mut self.rng,
        )?;
        self.state = tr.next_state().clone();
        self.terminal = tr.terminal();
        Ok(tr)
    }
}

impl EnvironmentObserver for BeaconCartWorld {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        World::observe(self, &self.state).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to observe BeaconCartWorld",
        })
    }
}

impl Actuator for BeaconCartWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        let tr = World::step(self, action, self.dt).map_err(|err| match err {
            DomainError::AlreadyTerminal => EnvironmentError::AlreadyTerminal,
            other => EnvironmentError::InvalidAction(other),
        })?;
        Ok(ApplyOutcome::new(tr.observation().clone(), tr.terminal()))
    }
}

impl LiveEnvironment for BeaconCartWorld {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        let state = World::reset(self).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to reset BeaconCartWorld",
        })?;
        World::observe(self, &state).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to observe BeaconCartWorld after reset",
        })
    }
}

#[derive(Clone)]
struct BeaconCartModel {
    world: BeaconCartWorld,
}

impl GenerativeModel for BeaconCartModel {
    fn action_space(&self) -> &ActionSpace {
        &self.world.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        BeaconCartWorld::step_state(&self.world.space, self.world.goal_x, state, action, dt, rng)
            .map_err(ModelError::TransitionFailed)
    }
}

impl BranchableSnapshot for BeaconCartWorld {
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
        self.state = snapshot.state().clone();
        self.rng = snapshot.rng();
        self.terminal = snapshot.terminal();
        Ok(())
    }
}

struct GreedyBeaconPlanner {
    dt: StepDt,
    goal_x: f64,
}

impl Planner for GreedyBeaconPlanner {
    fn plan<M: GenerativeModel + ?Sized>(
        &mut self,
        ctx: &mut PlanningContext<'_, M>,
        request: &PlanRequest,
    ) -> Result<PlanResult, ModelError> {
        let mut tracker = request.budget().tracker();
        let Some(origin_state) = request.state().cloned().or_else(|| {
            request
                .observation()
                .and_then(|obs| State::new(obs.tick(), obs.elapsed(), vec![obs.features()[0]]).ok())
        }) else {
            return Ok(PlanResult::no_valid_action(0));
        };

        let mut best_action: Option<Action> = None;
        let mut best_dist = f64::INFINITY;

        for idx in 0..3 {
            if !tracker.try_consume_transition() {
                return Ok(PlanResult::budget_exhausted(
                    request.fallback().cloned(),
                    tracker.transitions_used(),
                ));
            }
            let candidate = Action::discrete_in_space(idx, request.action_space())
                .map_err(ModelError::TransitionFailed)?;
            let tr = ctx.simulate_step(&origin_state, &candidate, self.dt)?;
            let next_pos = tr.next_state().values()[0];
            let dist = (self.goal_x - next_pos).abs();
            if dist < best_dist {
                best_dist = dist;
                best_action = Some(candidate);
            }
        }

        match best_action {
            Some(action) => Ok(PlanResult::completed(action, tracker.transitions_used())),
            None => Ok(PlanResult::no_valid_action(tracker.transitions_used())),
        }
    }
}

#[test]
fn external_consumer_runs_custom_world_planner_and_conformance_kit() {
    assert!(SDK_SEMVER.starts_with("0.1.0-alpha"));
    assert!(API_STABILITY_POLICY.contains("pre-1.0"));

    let seed = MasterSeed::new(2026);
    let dt = StepDt::from_secs_f64(0.1).unwrap();
    let world = BeaconCartWorld::new(seed, dt);
    let model = BeaconCartModel {
        world: world.clone(),
    };
    let safe_fallback = Action::discrete_in_space(1, World::action_space(&world)).unwrap();
    let move_right = Action::discrete_in_space(2, World::action_space(&world)).unwrap();
    let actions = vec![
        move_right.clone(),
        move_right.clone(),
        safe_fallback.clone(),
    ];
    let initial_state = world.state.clone();
    let mut planner = GreedyBeaconPlanner { dt, goal_x: 3.0 };

    // AC #2: Conformance Kit completo sobre implementaciones propias del consumidor.
    let report = run_conformance_suite(
        ConformanceConfig::new(seed, dt),
        move |s| Ok(BeaconCartWorld::new(s, dt)),
        &actions,
        &world,
        |forked| {
            forked.step(&move_right, dt)?;
            Ok(())
        },
        &model,
        &initial_state,
        &mut planner,
        &safe_fallback,
    )
    .expect("custom world and planner must pass conformance kit");
    assert!(report.all_passed());

    // También verifica LiveEnvironment (determinismo + aislamiento fork respecto al entorno vivo).
    conformance::verify_live_env_determinism(
        move |s| Ok(BeaconCartWorld::new(s, dt)),
        seed,
        &actions,
    )
    .expect("live environment must be locally deterministic");

    conformance::verify_live_env_fork_isolation(&world, &world, |forked| {
        forked.step(&actions[0], dt)?;
        Ok(())
    })
    .expect("forking simulation must not mutate live environment");

    // AC #1: Ejecución completa en HeadlessRuntime usando solo gogamen_sdk::*.
    let clock = Arc::new(VirtualMonotonicClock::new());
    let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback));
    let mut runtime = HeadlessRuntime::new(
        RuntimeConfig::new(seed, dt),
        world,
        model,
        GreedyBeaconPlanner { dt, goal_x: 3.0 },
        guard,
        clock,
    )
    .expect("headless runtime must construct");

    runtime.run_ticks(6).expect("runtime must execute ticks");
    runtime.with_env(|env| {
        assert!(env.state.values()[0] > 2.5);
    });
}

#[test]
fn external_consumer_authors_and_executes_system_spec() {
    // AC #1: Autoría declarativa vía SystemSpec + compilación + ejecución sin labs/* ni Jev.
    let observer = SystemNodeSpec::new("sensor", "observer")
        .with_output("observation", PortType::F64)
        .with_config("version", SpecConfigValue::String("1.0.0".to_owned()));
    let policy = SystemNodeSpec::new("controller", "policy")
        .with_input("observation", PortType::F64)
        .with_output("action", PortType::I64)
        .with_config("version", SpecConfigValue::String("1.0.0".to_owned()));

    let spec = SystemSpec::new("sdk_consumer_loop_v0")
        .with_config("dt_secs", SpecConfigValue::f64(0.05).unwrap())
        .with_node(observer)
        .with_node(policy)
        .with_connection(SystemConnectionSpec::instant(
            "sensor",
            "observation",
            "controller",
            "observation",
        ));

    let json = spec.to_canonical_json().expect("spec must serialize");
    let parsed = SystemSpec::from_json(&json).expect("spec must parse back");
    assert_eq!(parsed, spec);

    let catalog = builtin_plan_catalog();
    let compiled = SpecCompiler::compile(&parsed, &catalog, TemporalProfile::BoundedTick)
        .expect("spec must compile against builtin catalog");

    let engine = PlanRuntimeEngine::new();
    let summary = engine
        .run_plan(&compiled, 4)
        .expect("plan must run 4 ticks");
    assert_eq!(summary.ticks_executed, 4);
}
