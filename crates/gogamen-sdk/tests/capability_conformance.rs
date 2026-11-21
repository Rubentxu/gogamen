//! Test de integración de conformance de capacidades core desde extensión/consumidor externo.
//!
//! Usa únicamente `gogamen_sdk::*` (sin `gogamen_core`, sin `labs/*`, sin Jev ni UI) para:
//! 1. Implementar un mundo mínimo (`ThermalBeaconWorld`) y combinar al menos cuatro capacidades
//!    core reutilizándolas sin copiar algoritmos:
//!    - Sensores/procedencia y creencia (`ObservationEnvelope`, `ProvenanceTracker`, `BinaryBelief`)
//!    - Memoria episódica (`BoundedEpisodicMemory`, `AgentMemoryScope`)
//!    - Aprendizaje tabular (`TabularTransitionLearner`, `TransitionObservation`)
//!    - Motivación/homeostasis (`HomeostaticState`, `DriveSpec`, `UtilityEvaluator`, `ForbiddenActionRule`)
//!    - Planificación y guardia (`FmcPlanner`, `ActionGuard`, `SafeFallbackPolicy`)
//! 2. Ejecutar el Conformance Kit (`gogamen_sdk::conformance`) sobre catálogo de capacidades,
//!    temporalidad, aislamiento (`fork` y separación de autoridad/permisos) y replay de efectos.
//! 3. Demostrar un ejemplo sin IA (controlador termostático determinista por regla fija) usando
//!    únicamente catálogo, reloj virtual/runtime y guardia, sin activar planners ni modelos.

use std::time::Duration as StdDuration;

use gogamen_sdk::*;

#[derive(Clone)]
struct ThermalBeaconWorld {
    space: ActionSpace,
    state: State,
    rng: DeterministicRng,
    dt: StepDt,
    terminal: Terminal,
}

impl ThermalBeaconWorld {
    fn new(seed: MasterSeed, dt: StepDt) -> Self {
        let space = ActionSpace::discrete(3).expect("valid 3-action discrete space");
        let state = State::new(Tick::ZERO, SimDuration::ZERO, vec![18.0]).expect("valid state");
        Self {
            space,
            state,
            rng: DeterministicRng::new(seed.as_u64()),
            dt,
            terminal: Terminal::Ongoing,
        }
    }

    fn step_physics(
        space: &ActionSpace,
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
        let delta = match idx {
            0 => -1.5, // enfriar
            1 => 0.0,  // mantener
            2 => 1.5,  // calentar
            _ => unreachable!("validated against cardinality 3"),
        };
        let ambient_drift = (rng.next_f64() - 0.5) * 0.1;
        let next_temp = (state.values()[0] + delta + ambient_drift).clamp(0.0, 50.0);
        let next_state = state.advance(dt, vec![next_temp])?;
        let obs = Observation::new(next_state.tick(), next_state.elapsed(), vec![next_temp])?;
        let terminal = if next_state.tick().as_u64() >= 12 {
            Terminal::Terminated(TerminalReason::HorizonReached)
        } else {
            Terminal::Ongoing
        };
        Transition::new(state.tick(), dt, next_state, obs, terminal)
    }

    fn bucket_from_temp(temp: f64) -> usize {
        if temp < 20.0 {
            0
        } else if temp <= 24.0 {
            1
        } else {
            2
        }
    }
}

impl World for ThermalBeaconWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn reset(&mut self) -> Result<State, DomainError> {
        self.state = State::new(Tick::ZERO, SimDuration::ZERO, vec![18.0])?;
        self.terminal = Terminal::Ongoing;
        Ok(self.state.clone())
    }

    fn observe(&self, state: &State) -> Result<Observation, DomainError> {
        Observation::new(state.tick(), state.elapsed(), state.values().to_vec())
    }

    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError> {
        if self.terminal.is_terminal() {
            return Err(DomainError::AlreadyTerminal);
        }
        let tr = Self::step_physics(&self.space, &self.state, action, dt, &mut self.rng)?;
        self.state = tr.next_state().clone();
        self.terminal = tr.terminal();
        Ok(tr)
    }
}

impl GenerativeModel for ThermalBeaconWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        Self::step_physics(&self.space, state, action, dt, rng)
            .map_err(ModelError::TransitionFailed)
    }
}

impl EnvironmentObserver for ThermalBeaconWorld {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        World::observe(self, &self.state).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to observe thermal state",
        })
    }
}

impl Actuator for ThermalBeaconWorld {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        let tr = World::step(self, action, self.dt).map_err(|err| {
            EnvironmentError::ActuationFailed {
                reason: match err {
                    DomainError::AlreadyTerminal => "already terminal",
                    _ => "invalid action",
                },
            }
        })?;
        Ok(ApplyOutcome::new(tr.observation().clone(), tr.terminal()))
    }
}

impl LiveEnvironment for ThermalBeaconWorld {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        let state = World::reset(self).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to reset thermal world",
        })?;
        World::observe(self, &state).map_err(|_| EnvironmentError::ObservationFailed {
            reason: "failed to observe initial thermal state",
        })
    }
}

impl BranchableSnapshot for ThermalBeaconWorld {
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

struct ThermalSensorExtension {
    temp: f64,
}

impl EnvironmentObserver for ThermalSensorExtension {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        Observation::new(Tick::ZERO, SimDuration::ZERO, vec![self.temp]).map_err(|_| {
            EnvironmentError::ObservationFailed {
                reason: "invalid external thermal reading",
            }
        })
    }
}

fn build_capability_catalog() -> CapabilityCatalog {
    let mut catalog = CapabilityCatalog::new();
    let manifest = ModuleManifest::new(
        "thirdparty.thermal_controller",
        ModuleOrigin::ThirdParty,
        CapabilityVersion::new(0, 1, 0),
    )
    .with_granted_permissions(PermissionSet::from_permissions([
        Permission::state_read(),
        Permission::state_write(),
    ]))
    .with_operation(
        OperationSpec::new("sense_and_decide", CapabilityVersion::new(0, 1, 0))
            .with_required_permission(Permission::state_read())
            .with_effect(EffectKind::Pure)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("apply_thermal_step", CapabilityVersion::new(0, 1, 0))
            .with_required_permissions(PermissionSet::from_permissions([
                Permission::state_read(),
                Permission::state_write(),
            ]))
            .with_effect(EffectKind::StateMutation)
            .with_temporal_profile(TemporalProfile::BoundedTick),
    )
    .with_operation(
        OperationSpec::new("restricted_net_sync", CapabilityVersion::new(0, 1, 0))
            .with_required_permission(Permission::network())
            .with_effect(EffectKind::Network)
            .with_temporal_profile(TemporalProfile::BestEffort),
    );
    catalog
        .register_module(manifest)
        .expect("catalog registration succeeds");
    catalog
}

fn run_cognitive_episode(
    seed: MasterSeed,
    dt: StepDt,
    steps: usize,
) -> Result<(Vec<ReplayStepRecord>, TemporalTrace), ConformanceError> {
    let mut world = ThermalBeaconWorld::new(seed, dt);
    let init_obs = LiveEnvironment::reset(&mut world)?;
    let space = World::action_space(&world).clone();
    let safe_fallback = Action::discrete_in_space(1, &space)?;
    let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback.clone()));

    // 1. Sensores / Procedencia + Creencia binaria
    let mut provenance = ProvenanceTracker::new();
    let mut safety_belief = BinaryBelief::new(0.85).expect("valid prior");
    let likelihood_model =
        LikelihoodModel::from_sensitivity_specificity(0.9, 0.85).expect("valid likelihood");
    let mut deduplicator = EvidenceDeduplicator::new();
    let mut belief_watermark = None;

    // 2. Memoria episódica acotada
    let retention = RetentionPolicy::new(16, 32, None).expect("valid retention policy");
    let mut memory = BoundedEpisodicMemory::new(retention);
    let mem_scope = AgentMemoryScope::new(OwnerId::new(1));

    // 3. Aprendizaje tabular incremental
    let policy = LearningPolicy::new(1.0).expect("valid prior alpha");
    let mut learner = TabularTransitionLearner::new(3, 3, policy).expect("valid learner");

    // 4. Motivación / Homeostasis + restricción dura
    let temp_drive = DriveId::new("thermal_comfort").expect("valid drive id");
    let drive_spec =
        DriveSpec::new(temp_drive.clone(), 22.0, 10.0, 35.0, 1.0, 4.0).expect("valid drive spec");
    let mut homeostasis = HomeostaticState::new(vec![drive_spec.clone()]).expect("valid state");
    let utility = DynamicUtility::new(vec![
        ObjectiveTerm::from_drive("comfort", &drive_spec).expect("valid term"),
    ])
    .expect("valid utility");
    let evaluator = UtilityEvaluator::new(utility).with_constraint(
        ForbiddenActionRule::discrete_index("no_cool_when_cold", 0, "forbid cooling below 19C"),
    );

    // 5. Planificador FMC
    let mut planner = FmcPlanner::new(FmcPlannerConfig::default());

    let clock = VirtualMonotonicClock::new();
    let mut trace = TemporalTrace::new();
    trace.record_with_clock(Tick::ZERO, WorkUnits::ZERO, &clock);

    let mut records = Vec::with_capacity(steps);
    let mut current_obs = init_obs;
    let mut prev_bucket = ThermalBeaconWorld::bucket_from_temp(current_obs.features()[0]);

    for step_idx in 0..steps {
        let tick = Tick::new(step_idx as u64);
        let temp = current_obs.features()[0];
        let now = clock.now();

        let metadata = ProvenanceMetadata {
            message_id: MessageId::new(step_idx as u64 + 1),
            observed_at: now,
            received_at: now,
            sensor_id: SensorId::new(10),
            frame_id: FrameId::world(),
            units: UnitTag::new("celsius").expect("valid unit"),
            confidence: Confidence::new(0.95).expect("valid confidence"),
            revision: step_idx as u64 + 1,
            evidence_ref: EvidenceRef::new(step_idx as u64 + 100),
            claim_kind: ObservationKind::Measurement,
        };
        let envelope = ObservationEnvelope::new(metadata.clone(), current_obs.clone());
        assert_eq!(
            provenance.record_metadata(envelope.metadata()),
            IngressStatus::Accepted
        );

        let bool_envelope =
            ObservationEnvelope::new(metadata.clone(), (20.0..=24.0).contains(&temp));
        let _ = safety_belief
            .update_with_observation(
                &bool_envelope,
                &likelihood_model,
                &mut deduplicator,
                &mut belief_watermark,
            )
            .expect("belief update succeeds");

        homeostasis
            .set_reading(&temp_drive, temp)
            .expect("valid reading");

        // Evaluación motivacional sobre candidatos discretos.
        let candidates = (0..3)
            .map(|idx| {
                let act = Action::discrete_in_space(idx, &space).expect("valid action");
                let predicted_temp = temp
                    + match idx {
                        0 => -1.5,
                        1 => 0.0,
                        _ => 1.5,
                    };
                let score = -(22.0 - predicted_temp).abs();
                ActionCandidate::new(act).with_score("comfort", score)
            })
            .collect::<Result<Vec<_>, _>>()
            .expect("valid action candidates");
        let eval_outcome = evaluator
            .evaluate_candidates(&candidates, &homeostasis)
            .expect("evaluation succeeds");
        assert!(eval_outcome.selected().is_some());

        // Planificación acotada con FmcPlanner usando el mundo como GenerativeModel.
        let mut plan_rng = DeterministicRng::new(seed.as_u64() ^ (step_idx as u64 + 1));
        let mut plan_ctx = PlanningContext::new(&world, &mut plan_rng);
        let sim_state = State::new(tick, current_obs.elapsed(), vec![temp])?;
        let request =
            PlanRequest::from_state(sim_state, space.clone(), PlanBudget::with_transitions(8))
                .with_fallback(safe_fallback.clone())?;
        let plan_result = planner.plan(&mut plan_ctx, &request)?;
        let chosen_action = plan_result
            .authorized_action()
            .cloned()
            .unwrap_or_else(|| safe_fallback.clone());

        // Guardia independiente.
        let proposal = ActionProposal::authorized(chosen_action.clone(), tick);
        let guard_ctx =
            GuardContext::new(&space, tick, std::time::Instant::now(), Terminal::Ongoing);
        let decision = guard.evaluate(&proposal, &guard_ctx);
        let final_action = match &decision {
            GuardDecision::Allow(a) | GuardDecision::Fallback(a) => a.clone(),
            GuardDecision::Reject(_) | GuardDecision::NoOp => safe_fallback.clone(),
        };

        let outcome = world.apply(&final_action)?;
        let next_bucket = ThermalBeaconWorld::bucket_from_temp(outcome.observation().features()[0]);

        // Registro en memoria episódica y aprendizaje tabular.
        let ep_input = EpisodeInput::new(
            OwnerId::new(1),
            EntityId::new(1),
            tick,
            now,
            step_idx as u64 + 1,
            EpisodeProvenance {
                evidence_ref: envelope.evidence_ref(),
                message_id: envelope.message_id(),
                sensor_id: envelope.sensor_id(),
                claim_kind: envelope.claim_kind(),
                confidence: envelope.confidence(),
            },
            SummaryCode::new(next_bucket as u64),
            outcome.observation().features().to_vec(),
        );
        mem_scope
            .record_episode(&mut memory, ep_input)
            .expect("episode recorded");

        let transition_envelope = envelope.clone().map(|_| {
            TransitionSample::new(
                prev_bucket,
                final_action.as_discrete().expect("discrete").index(),
                next_bucket,
            )
        });
        let transition_obs = TransitionObservation::from_envelope(
            &transition_envelope,
            ObservationId::new(step_idx as u64 + 1),
        );
        let _ = learner.observe(&transition_obs).expect("learner update");

        clock.advance(StdDuration::from_secs_f64(dt.as_secs_f64()));
        let work_units = WorkUnits::new(step_idx as u64 + 1);
        trace.record_with_clock(Tick::new(step_idx as u64 + 1), work_units, &clock);

        records.push(ReplayStepRecord {
            tick,
            action: final_action,
            guard_decision: decision,
            observation: outcome.observation().clone(),
            terminal: outcome.terminal(),
            work_units,
        });

        current_obs = outcome.observation().clone();
        prev_bucket = next_bucket;
        if outcome.terminal().is_terminal() {
            break;
        }
    }

    assert!(!mem_scope.is_empty(&memory));
    assert!(learner.seen_observation_count() > 0);
    assert!(safety_belief.p_true() > 0.0);

    Ok((records, trace))
}

#[test]
fn external_world_combines_core_capabilities_and_passes_capability_conformance() {
    let seed = MasterSeed::new(2026);
    let dt = StepDt::from_secs_f64(0.1).expect("valid dt");
    let config = ConformanceConfig::new(seed, dt).with_horizon_steps(6);

    // Registro estático de extensión de sensor externo.
    let mut registry = StaticExtensionRegistry::new()
        .with_supported_capability("sensor.thermal1d", CapabilityVersion::new(0, 1, 0));
    let sensor_manifest = ExtensionManifest::new(
        "thirdparty.thermal_sensor",
        ExtensionKind::Sensor,
        ModuleOrigin::ThirdParty,
        "external_thermal_crate",
        CapabilityVersion::new(0, 1, 0),
    )
    .with_capability("sensor.thermal1d", CapabilityVersion::new(0, 1, 0))
    .with_required_permission(Permission::state_read());
    registry
        .register_sensor(sensor_manifest, |_| {
            Ok(Box::new(ThermalSensorExtension { temp: 18.0 }))
        })
        .expect("sensor registered");

    verify_extension_registry_contract(
        &registry,
        &[ExtensionId::new("thirdparty.thermal_sensor")],
        &PermissionSet::from_permissions([Permission::state_read()]),
    )
    .expect("extension registry contract passes");

    let catalog = build_capability_catalog();
    let valid_bindings = vec![
        OperationBinding::new(
            "thirdparty.thermal_controller",
            "sense_and_decide",
            CapabilityVersion::new(0, 1, 0),
        ),
        OperationBinding::new(
            "thirdparty.thermal_controller",
            "apply_thermal_step",
            CapabilityVersion::new(0, 1, 0),
        ),
    ];
    let restricted_binding = OperationBinding::new(
        "thirdparty.thermal_controller",
        "restricted_net_sync",
        CapabilityVersion::new(0, 1, 0),
    );

    let mut live_env = ThermalBeaconWorld::new(seed, dt);
    let _ = LiveEnvironment::reset(&mut live_env).expect("live env reset");
    let branchable = live_env.clone();
    let space = World::action_space(&live_env).clone();
    let safe_fallback = Action::discrete_in_space(1, &space).expect("valid fallback");

    let report = run_capability_conformance_suite(
        config,
        &catalog,
        TemporalProfile::BoundedTick,
        &valid_bindings,
        &restricted_binding,
        &live_env,
        &branchable,
        |forked| {
            let heat = Action::discrete_in_space(2, World::action_space(forked))?;
            World::step(forked, &heat, dt)?;
            Ok(())
        },
        &space,
        &safe_fallback,
        |s| run_cognitive_episode(s, dt, 6),
    )
    .expect("full capability conformance suite passes");

    assert!(report.all_passed());
    assert_eq!(report.validated_operations, 2);
    assert_eq!(report.steps_verified, 6);
}

#[test]
fn deterministic_thermostat_controller_without_ai_passes_catalog_guard_and_replay() {
    let seed = MasterSeed::new(9090);
    let dt = StepDt::from_secs_f64(0.1).expect("valid dt");
    let catalog = build_capability_catalog();
    let binding = OperationBinding::new(
        "thirdparty.thermal_controller",
        "apply_thermal_step",
        CapabilityVersion::new(0, 1, 0),
    );
    let validated =
        verify_capability_catalog_contract(&catalog, TemporalProfile::BoundedTick, &[binding])
            .expect("non-AI controller operation validates against catalog");
    assert_eq!(validated.len(), 1);

    // Controlador convencional determinista sin Planner ni GenerativeModel.
    let replay = verify_effect_replay(
        |s| {
            let mut env = ThermalBeaconWorld::new(s, dt);
            let mut obs = LiveEnvironment::reset(&mut env)?;
            let space = World::action_space(&env).clone();
            let safe_fallback = Action::discrete_in_space(1, &space)?;
            let guard =
                ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(safe_fallback.clone()));
            let clock = VirtualMonotonicClock::new();
            let mut trace = TemporalTrace::new();
            trace.record_with_clock(Tick::ZERO, WorkUnits::ZERO, &clock);

            let mut records = Vec::with_capacity(5);
            for step in 0..5usize {
                let tick = Tick::new(step as u64);
                let temp = obs.features()[0];
                let rule_idx = if temp < 20.0 {
                    2 // calentar
                } else if temp > 24.0 {
                    0 // enfriar
                } else {
                    1 // mantener
                };
                let proposed = Action::discrete_in_space(rule_idx, &space)?;
                let proposal = ActionProposal::authorized(proposed.clone(), tick);
                let guard_ctx =
                    GuardContext::new(&space, tick, std::time::Instant::now(), Terminal::Ongoing);
                let decision = guard.evaluate(&proposal, &guard_ctx);
                let applied = match &decision {
                    GuardDecision::Allow(a) | GuardDecision::Fallback(a) => a.clone(),
                    GuardDecision::Reject(_) | GuardDecision::NoOp => safe_fallback.clone(),
                };
                let outcome = env.apply(&applied)?;

                clock.advance(StdDuration::from_secs_f64(dt.as_secs_f64()));
                let work_units = WorkUnits::new(step as u64 + 1);
                trace.record_with_clock(Tick::new(step as u64 + 1), work_units, &clock);

                records.push(ReplayStepRecord {
                    tick,
                    action: applied,
                    guard_decision: decision,
                    observation: outcome.observation().clone(),
                    terminal: outcome.terminal(),
                    work_units,
                });

                obs = outcome.observation().clone();
                if outcome.terminal().is_terminal() {
                    break;
                }
            }

            Ok((records, trace))
        },
        seed,
    )
    .expect("non-AI deterministic controller passes replay conformance");

    assert_eq!(replay.steps_verified, 5);
    assert_eq!(
        replay.temporal_verdict,
        TemporalReplayVerdict::DeterministicMatch
    );
}
