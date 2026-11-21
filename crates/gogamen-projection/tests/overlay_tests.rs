use gogamen_core::model::{GenerativeModel, PlanningContext, RngSource};
use gogamen_core::planner::{PlanBudget, PlanRequest};
use gogamen_core::{
    Action, ActionSpace, DeterministicRng, DiscreteAction, EntityId, FmcPlanner, FmcPlannerConfig,
    FmcScoringConfig, MasterSeed, ModelError, Observation, SimDuration, State, StepDt, Terminal,
    Tick, Transition,
};
use gogamen_projection::{
    CoordinateSpace, DashPattern, DecisionOverlayFrame, ExecutedTrajectory, FutureKind,
    FutureTrajectorySample, LayerId, LayerVisibility, LegendCategory, OverlayBudget, OverlayConfig,
    OverlayPalette, Point2, ProjectionError, ScaleUnits, SceneId, SensorRaySample, SensorSample,
    TerminalMarker, TerminalOutcome, project_decision_overlay,
};

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).unwrap()
}

fn build_rich_fixture_frame() -> DecisionOverlayFrame {
    let executed = ExecutedTrajectory::new(
        1,
        vec![pt(0.0, 0.0), pt(1.0, 0.2), pt(2.0, 0.5)],
        Some(EntityId::new(7)),
    )
    .unwrap();

    let futures = vec![
        FutureTrajectorySample::new(
            10,
            FutureKind::ObservedReality,
            vec![pt(0.0, 0.0), pt(2.0, 0.5)],
            0.0,
        )
        .unwrap(),
        FutureTrajectorySample::new(
            20,
            FutureKind::SelectedFuture,
            vec![pt(2.0, 0.5), pt(3.2, 1.0), pt(4.5, 1.4)],
            2.4,
        )
        .unwrap()
        .with_walker_index(0)
        .with_terminal(
            TerminalMarker::new(pt(4.5, 1.4), TerminalOutcome::Success, 2, 2.4).unwrap(),
        ),
        FutureTrajectorySample::new(
            30,
            FutureKind::Hypothesis,
            vec![pt(2.0, 0.5), pt(3.0, 0.2), pt(4.0, -0.3)],
            1.1,
        )
        .unwrap()
        .with_walker_index(1),
        FutureTrajectorySample::new(
            40,
            FutureKind::RejectedFuture,
            vec![pt(2.0, 0.5), pt(2.5, -1.0), pt(2.8, -2.0)],
            -1.5,
        )
        .unwrap()
        .with_walker_index(2)
        .with_terminal(
            TerminalMarker::new(pt(2.8, -2.0), TerminalOutcome::Failure, 2, -1.5).unwrap(),
        ),
    ];

    let sensors: Vec<SensorSample> = vec![
        SensorRaySample::new(1, pt(2.0, 0.5), pt(3.5, 0.8), true, 0.92).unwrap(),
        SensorRaySample::new(2, pt(2.0, 0.5), pt(4.0, 0.5), false, 0.50).unwrap(),
    ];

    let extra_terminals =
        vec![TerminalMarker::new(pt(4.0, -0.3), TerminalOutcome::HorizonReached, 2, 1.1).unwrap()];

    DecisionOverlayFrame::new(SceneId::new(11), Tick::new(5), 42)
        .with_executed_trajectory(executed)
        .with_futures(futures)
        .with_sensors(sensors)
        .with_terminals(extra_terminals)
}

#[test]
fn ac1_futures_sensors_and_executed_trajectory_project_to_ordered_layers_with_legend_and_styles() {
    let frame = build_rich_fixture_frame();
    let config = OverlayConfig::default();

    let projected_a = project_decision_overlay(&frame, &config).unwrap();
    let projected_b = project_decision_overlay(&frame, &config).unwrap();
    assert_eq!(
        projected_a.to_canonical_json(),
        projected_b.to_canonical_json()
    );

    let layer_order: Vec<(LayerId, i32)> = projected_a
        .scene
        .layers()
        .iter()
        .map(|l| (l.id(), l.z_order()))
        .collect();
    let palette = OverlayPalette::default();
    assert_eq!(
        layer_order,
        vec![
            (
                palette.observed_reality.layer_id,
                palette.observed_reality.z_order
            ),
            (
                palette.executed_trajectory.layer_id,
                palette.executed_trajectory.z_order
            ),
            (palette.sensor_ray.layer_id, palette.sensor_ray.z_order),
            (
                palette.rejected_future.layer_id,
                palette.rejected_future.z_order
            ),
            (palette.hypothesis.layer_id, palette.hypothesis.z_order),
            (
                palette.selected_future.layer_id,
                palette.selected_future.z_order
            ),
            (
                palette.terminal_marker.layer_id,
                palette.terminal_marker.z_order
            ),
            (LayerId::Labels, LayerId::Labels.default_z_order()),
        ]
    );

    // Trayectoria ejecutada separada de futuros
    assert_ne!(
        palette.executed_trajectory.layer_id,
        palette.selected_future.layer_id
    );
    assert_ne!(
        palette.executed_trajectory.layer_id,
        palette.observed_reality.layer_id
    );

    // Estilos diferenciados entre las 4 categorías de futuros + trayectoria ejecutada + sensores + terminales
    let entries = projected_a.legend.entries();
    assert_eq!(entries.len(), 7);
    let obs = projected_a
        .legend
        .find(LegendCategory::ObservedReality)
        .unwrap();
    let exe = projected_a
        .legend
        .find(LegendCategory::ExecutedTrajectory)
        .unwrap();
    let hyp = projected_a.legend.find(LegendCategory::Hypothesis).unwrap();
    let sel = projected_a
        .legend
        .find(LegendCategory::SelectedFuture)
        .unwrap();
    let rej = projected_a
        .legend
        .find(LegendCategory::RejectedFuture)
        .unwrap();

    assert_ne!(obs.style.color, sel.style.color);
    assert_ne!(sel.style.color, hyp.style.color);
    assert_ne!(hyp.style.color, rej.style.color);
    assert_ne!(exe.style.color, sel.style.color);
    assert_eq!(obs.style.dash, DashPattern::Solid);
    assert_eq!(hyp.style.dash, DashPattern::Dashed);
    assert_eq!(sel.style.dash, DashPattern::DashDot);
    assert_eq!(rej.style.dash, DashPattern::Dotted);
    assert!(sel.style.opacity > hyp.style.opacity);
    assert!(hyp.style.opacity > rej.style.opacity);

    assert_eq!(obs.sample_count, 1);
    assert_eq!(exe.sample_count, 1);
    assert_eq!(hyp.sample_count, 1);
    assert_eq!(sel.sample_count, 1);
    assert_eq!(rej.sample_count, 1);

    // Leyenda también proyectada dentro de Scene2D en LayerId::Labels
    let scene_json = projected_a.scene.to_canonical_json();
    assert!(scene_json.contains("Observed Reality"));
    assert!(scene_json.contains("Executed Trajectory"));
    assert!(scene_json.contains("Selected Future"));
    assert!(scene_json.contains("Hypothesis"));
    assert!(scene_json.contains("Rejected Future"));
    assert!(scene_json.contains("Sensor Rays"));
    assert!(scene_json.contains("Terminal Markers"));
}

#[derive(Debug, Clone)]
struct ToyPointMassModel {
    goal_x: f64,
    actions: ActionSpace,
}

impl ToyPointMassModel {
    fn new(goal_x: f64) -> Self {
        Self {
            goal_x,
            actions: ActionSpace::Discrete { cardinality: 3 },
        }
    }
}

impl GenerativeModel for ToyPointMassModel {
    fn action_space(&self) -> &ActionSpace {
        &self.actions
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.actions
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        let idx = action.as_discrete().map(DiscreteAction::index).unwrap_or(1);
        let accel = match idx {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
        let noise = (rng.next_f64() - 0.5) * 0.02;
        let x = state.values()[0];
        let v = state.values()[1];
        let next_v = (v + accel * dt.as_secs_f64() + noise).clamp(-2.0, 2.0);
        let next_x = x + next_v * dt.as_secs_f64();
        let next_state = state
            .advance(dt, vec![next_x, next_v])
            .map_err(ModelError::TransitionFailed)?;
        let obs = Observation::new(
            next_state.tick(),
            next_state.elapsed(),
            vec![next_x, next_v],
        )
        .map_err(ModelError::TransitionFailed)?;
        Transition::new(state.tick(), dt, next_state, obs, Terminal::Ongoing)
            .map_err(ModelError::TransitionFailed)
    }
}

#[test]
fn ac2_visibility_and_sampling_rate_do_not_alter_planner_rng_reward_or_choice() {
    let model = ToyPointMassModel::new(3.0);
    let initial_state = State::new(Tick::ZERO, SimDuration::ZERO, vec![0.0, 0.0]).unwrap();
    let planner_cfg = FmcPlannerConfig::new(
        16,
        6,
        StepDt::from_millis_f64(100.0).unwrap(),
        FmcScoringConfig::default(),
        MasterSeed::new(777),
    )
    .unwrap();
    let planner = FmcPlanner::new(planner_cfg);

    let configs = [
        OverlayConfig::default(),
        OverlayConfig {
            visibility: LayerVisibility {
                sensors: false,
                hypotheses: false,
                rejected_futures: false,
                terminals: false,
                ..LayerVisibility::all_visible()
            },
            sample_stride: 3,
            sampling_rate: 0.5,
            ..OverlayConfig::default()
        },
        OverlayConfig {
            visibility: LayerVisibility::minimal(),
            sample_stride: 2,
            sampling_rate: 0.25,
            ..OverlayConfig::default()
        },
    ];

    let frame = build_rich_fixture_frame();
    let mut results = Vec::new();
    let mut scenes = Vec::new();

    for cfg in &configs {
        let mut ctx_rng = DeterministicRng::new(12345);
        let mut ctx = PlanningContext::new(&model, &mut ctx_rng);
        let request = PlanRequest::from_state(
            initial_state.clone(),
            model.action_space().clone(),
            PlanBudget::with_transitions(256),
        )
        .with_fallback(Action::Discrete(DiscreteAction::new(1, 3).unwrap()))
        .unwrap();
        let (plan_result, trace) = planner
            .plan_with_trace(
                &mut ctx,
                &request,
                |_s, _a, tr| -(model.goal_x - tr.next_state().values()[0]).abs(),
                |s| s.values().to_vec(),
            )
            .unwrap();
        let sim_rng_val = ctx_rng.next_u64();
        results.push((plan_result, trace, sim_rng_val));
        scenes.push(project_decision_overlay(&frame, cfg).unwrap());
    }

    for i in 1..results.len() {
        assert_eq!(
            results[0].0.authorized_action(),
            results[i].0.authorized_action()
        );
        assert_eq!(
            results[0].1.mean_virtual_reward(),
            results[i].1.mean_virtual_reward()
        );
        assert_eq!(
            results[0].1.rollout_transitions_used(),
            results[i].1.rollout_transitions_used()
        );
        assert_eq!(results[0].1.cloning_events(), results[i].1.cloning_events());
        assert_eq!(results[0].2, results[i].2);
    }

    // La visualización sí refleja el cambio de visibilidad y muestreo
    assert_ne!(
        scenes[0].scene.to_canonical_json(),
        scenes[1].scene.to_canonical_json()
    );
    assert_ne!(
        scenes[1].scene.to_canonical_json(),
        scenes[2].scene.to_canonical_json()
    );
}

#[test]
fn ac3_budget_caps_samples_and_bytes_per_frame_without_exporting_all_walkers_by_default() {
    let default_budget = OverlayBudget::default();
    assert!(!default_budget.export_all_walkers);

    let mut futures = vec![
        FutureTrajectorySample::new(
            0,
            FutureKind::SelectedFuture,
            vec![pt(0.0, 0.0), pt(2.0, 2.0)],
            5.0,
        )
        .unwrap(),
    ];
    for walker_id in 1..=64_u64 {
        let kind = if walker_id % 2 == 0 {
            FutureKind::Hypothesis
        } else {
            FutureKind::RejectedFuture
        };
        futures.push(
            FutureTrajectorySample::new(
                walker_id,
                kind,
                vec![
                    pt(0.0, 0.0),
                    pt(1.0, walker_id as f64 * 0.1),
                    pt(2.0, walker_id as f64 * 0.2),
                ],
                -(walker_id as f64) * 0.1,
            )
            .unwrap()
            .with_walker_index(walker_id as usize),
        );
    }

    let mut sensors = Vec::new();
    for sid in 0..48_u64 {
        sensors.push(
            SensorRaySample::new(
                sid,
                pt(0.0, 0.0),
                pt(1.0 + sid as f64 * 0.05, 0.5),
                true,
                0.8,
            )
            .unwrap(),
        );
    }

    let frame = DecisionOverlayFrame::new(SceneId::new(2), Tick::new(10), 99)
        .with_futures(futures)
        .with_sensors(sensors);

    // Presupuesto por defecto: no exporta los 64 walkers
    let default_cfg = OverlayConfig::default();
    let projected = project_decision_overlay(&frame, &default_cfg).unwrap();
    assert!(projected.stats.truncated_by_sample_cap);
    assert!(projected.stats.exported_futures <= default_cfg.budget.max_future_samples);
    assert!(projected.stats.exported_futures < 64);
    assert!(projected.stats.exported_sensors <= default_cfg.budget.max_sensor_samples);
    assert!(projected.stats.serialized_bytes <= default_cfg.budget.max_bytes_per_frame);
    assert_eq!(
        projected.stats.serialized_bytes,
        projected.scene.to_canonical_json().len()
    );

    // Presupuesto estricto en bytes: recorta progresivamente hasta cumplir el cap exacto
    let tight_cfg = OverlayConfig {
        coordinate_space: CoordinateSpace::from_units(ScaleUnits::Meters),
        budget: OverlayBudget {
            max_future_samples: 32,
            max_sensor_samples: 32,
            max_terminal_markers: 16,
            max_bytes_per_frame: 1800,
            export_all_walkers: false,
        },
        ..OverlayConfig::default()
    };
    let tight_projected_a = project_decision_overlay(&frame, &tight_cfg).unwrap();
    let tight_projected_b = project_decision_overlay(&frame, &tight_cfg).unwrap();
    assert!(tight_projected_a.stats.truncated_by_byte_cap);
    assert!(tight_projected_a.scene.to_canonical_json().len() <= 1800);
    assert_eq!(
        tight_projected_a.to_canonical_json(),
        tight_projected_b.to_canonical_json()
    );

    // Presupuesto imposiblemente pequeño devuelve error explícito
    let impossible_cfg = OverlayConfig {
        budget: OverlayBudget {
            max_bytes_per_frame: 32,
            ..OverlayBudget::default()
        },
        ..OverlayConfig::default()
    };
    assert!(matches!(
        project_decision_overlay(&frame, &impossible_cfg),
        Err(ProjectionError::OverlayBudgetTooSmall { .. })
    ));
}
