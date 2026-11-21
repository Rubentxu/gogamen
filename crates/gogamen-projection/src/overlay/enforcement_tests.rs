//! Pruebas de equivalencia del recorte exacto por bytes frente al bucle lineal original.

use std::cell::Cell;

use super::*;

thread_local! {
    /// Nº de serializaciones canónicas de escena realizadas en la ruta de enforcement.
    pub(super) static CANONICAL_SERIALIZATIONS: Cell<usize> = const { Cell::new(0) };
}

fn serializations() -> usize {
    CANONICAL_SERIALIZATIONS.with(Cell::get)
}

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).unwrap()
}

/// Implementación de referencia: el bucle original (una reconstrucción y serialización por paso).
fn reference_project(
    frame: &DecisionOverlayFrame,
    config: &OverlayConfig,
) -> Result<ProjectedOverlay, ProjectionError> {
    config.validate()?;
    let transform = ScaleTransform::from_meters(config.coordinate_space);
    let mut futures_sorted: Vec<&FutureTrajectorySample> = frame.futures().iter().collect();
    futures_sorted.sort_by_key(|f| (f.kind().priority_rank(), f.trajectory_id()));
    let mut sensors_sorted: Vec<&SensorRaySample> = frame.sensors().iter().collect();
    sensors_sorted.sort_by_key(|s| s.sensor_id());
    let mut all_terminals: Vec<TerminalMarker> = frame.terminals().to_vec();
    for f in &futures_sorted {
        if let Some(term) = f.terminal() {
            all_terminals.push(term);
        }
    }
    all_terminals.sort_by_key(|t| (t.step_index(), t.outcome()));
    let input_futures = futures_sorted.len();
    let input_sensors = sensors_sorted.len();
    let input_terminals = all_terminals.len();
    let (mut sf, mut trunc) = select_futures(&futures_sorted, config);
    let (mut ss, c1) = select_evenly(&sensors_sorted, config, config.budget.max_sensor_samples);
    let (mut st, c2) = select_evenly(&all_terminals, config, config.budget.max_terminal_markers);
    trunc |= c1 || c2;
    let mut inc_exec =
        frame.executed_trajectory().is_some() && config.visibility.executed_trajectory;
    let mut inc_legend = config.visibility.legend;
    let mut byte_cap = false;
    loop {
        let (scene, legend) = build_scene_and_legend(
            frame, config, &transform, inc_exec, &sf, &ss, &st, inc_legend,
        )?;
        let bytes = scene.to_canonical_json().len();
        if bytes <= config.budget.max_bytes_per_frame {
            let stats = OverlayStats {
                input_futures,
                exported_futures: sf.len(),
                input_sensors,
                exported_sensors: ss.len(),
                input_terminals,
                exported_terminals: st.len(),
                truncated_by_sample_cap: trunc,
                truncated_by_byte_cap: byte_cap,
                serialized_bytes: bytes,
            };
            return Ok(ProjectedOverlay {
                scene,
                legend,
                stats,
            });
        }
        byte_cap = true;
        if !shed_lowest_priority(&mut sf, &mut ss, &mut st, &mut inc_exec, &mut inc_legend) {
            return Err(ProjectionError::OverlayBudgetTooSmall {
                required_bytes: bytes,
                max_bytes: config.budget.max_bytes_per_frame,
            });
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn seeded_frame(seed: u64, n_futures: usize) -> DecisionOverlayFrame {
    let mut rng = Lcg(seed);
    let mut futures = Vec::new();
    for id in 0..n_futures as u64 {
        let kind = match rng.below(4) {
            0 => FutureKind::ObservedReality,
            1 => FutureKind::SelectedFuture,
            2 => FutureKind::Hypothesis,
            _ => FutureKind::RejectedFuture,
        };
        let n_pts = 2 + rng.below(6) as usize;
        let pts: Vec<Point2> = (0..n_pts)
            .map(|i| {
                pt(
                    i as f64 * (0.1 + rng.below(1000) as f64 / 37.0),
                    -(rng.below(100_000) as f64) / 13.0,
                )
            })
            .collect();
        let mut f = FutureTrajectorySample::new(id, kind, pts, rng.below(100) as f64).unwrap();
        if rng.below(5) == 0 {
            f = f.with_terminal(
                TerminalMarker::new(
                    pt(1.0, 2.0),
                    TerminalOutcome::Success,
                    rng.below(30) as u32,
                    1.0,
                )
                .unwrap(),
            );
        }
        futures.push(f);
    }
    let sensors: Vec<SensorRaySample> = (0..rng.below(40))
        .map(|sid| {
            SensorRaySample::new(
                sid,
                pt(0.0, 0.0),
                pt(rng.below(5000) as f64 / 7.0, 0.5),
                rng.below(2) == 0,
                0.8,
            )
            .unwrap()
        })
        .collect();
    let terminals: Vec<TerminalMarker> = (0..rng.below(20))
        .map(|i| {
            TerminalMarker::new(
                pt(i as f64 / 3.0, 1.0),
                TerminalOutcome::Success,
                i as u32,
                1.0,
            )
            .unwrap()
        })
        .collect();
    let executed = ExecutedTrajectory::new(
        rng.below(50),
        vec![pt(0.0, 0.0), pt(1.0, 0.2), pt(2.0, 0.5)],
        None,
    )
    .unwrap();
    DecisionOverlayFrame::new(SceneId::new(seed), Tick::new(seed), seed)
        .with_futures(futures)
        .with_sensors(sensors)
        .with_terminals(terminals)
        .with_executed_trajectory(executed)
}

fn config_with(max_bytes: usize, variant: u64) -> OverlayConfig {
    let mut cfg = OverlayConfig {
        budget: OverlayBudget {
            max_future_samples: 600,
            max_sensor_samples: 64,
            max_terminal_markers: 64,
            max_bytes_per_frame: max_bytes,
            export_all_walkers: false,
        },
        ..OverlayConfig::default()
    };
    if variant & 1 == 1 {
        cfg.visibility.sensors = false;
    }
    if variant & 2 == 2 {
        cfg.visibility.terminals = false;
    }
    if variant & 4 == 4 {
        cfg.visibility.legend = false;
    }
    if variant & 8 == 8 {
        cfg.visibility.executed_trajectory = false;
    }
    cfg
}

#[test]
fn exact_shed_matches_linear_reference_across_seeds_sizes_and_budgets() {
    let mut compared = 0;
    for seed in 1..=8_u64 {
        for &n in &[0_usize, 1, 3, 17, 60] {
            let frame = seeded_frame(seed * 7919, n);
            let full = reference_project(&frame, &config_with(usize::MAX, 0))
                .unwrap()
                .stats
                .serialized_bytes;
            let variant = seed % 16;
            let mut budgets = vec![0, 1, 50, 200, full, full.saturating_sub(1), full + 1];
            budgets.extend((1..12).map(|i| full * i / 12));
            // Fronteras exactas: tamaños de cada estado de la secuencia de recorte.
            for b in budgets.clone().into_iter().skip(4).step_by(2) {
                if let Ok(p) = reference_project(&frame, &config_with(b, variant)) {
                    let s = p.stats.serialized_bytes;
                    budgets.extend([s, s.saturating_sub(1), s + 1]);
                }
            }
            for b in budgets {
                let cfg = config_with(b, variant);
                let expected = reference_project(&frame, &cfg);
                let before = serializations();
                let actual = project_decision_overlay(&frame, &cfg);
                assert!(serializations() - before <= 2, "fallback lineal inesperado");
                assert_eq!(actual, expected, "seed={seed} n={n} budget={b} v={variant}");
                if let Ok(p) = &actual {
                    assert!(p.stats.serialized_bytes <= b);
                    assert_eq!(p.stats.serialized_bytes, p.scene.to_canonical_json().len());
                }
                compared += 1;
            }
        }
    }
    assert!(compared > 500);
}

#[test]
fn impossible_and_below_base_overhead_budgets_report_same_error_as_reference() {
    let frame = seeded_frame(42, 30);
    for b in [0usize, 1, 10] {
        let cfg = config_with(b, 0);
        let expected = reference_project(&frame, &cfg);
        assert!(matches!(
            expected,
            Err(ProjectionError::OverlayBudgetTooSmall { .. })
        ));
        assert_eq!(project_decision_overlay(&frame, &cfg), expected);
    }
}

#[test]
fn heavy_shedding_of_500_futures_uses_at_most_three_canonical_serializations() {
    let frame = seeded_frame(2026, 500);
    let full = reference_project(&frame, &config_with(usize::MAX, 0)).unwrap();
    assert!(full.stats.exported_futures >= 400);

    for budget in [full.stats.serialized_bytes / 10, 3_000, 0] {
        let cfg = config_with(budget, 0);
        let before = serializations();
        let actual = project_decision_overlay(&frame, &cfg);
        let used = serializations() - before;
        assert!(used <= 3, "budget={budget}: {used} serializaciones");
        assert_eq!(actual, reference_project(&frame, &cfg));
    }

    let cfg = config_with(full.stats.serialized_bytes / 10, 0);
    let p = project_decision_overlay(&frame, &cfg).unwrap();
    assert!(p.stats.truncated_by_byte_cap);
    assert!(p.stats.exported_futures < full.stats.exported_futures / 2);
}

#[test]
fn shedding_order_drops_rejected_then_hypotheses_before_selected() {
    let frame = seeded_frame(5, 60);
    let full = reference_project(&frame, &config_with(usize::MAX, 8)).unwrap();
    let cfg = config_with(full.stats.serialized_bytes * 2 / 3, 8);
    let p = project_decision_overlay(&frame, &cfg).unwrap();
    let count = |s: &ProjectedOverlay, id: LayerId| {
        s.scene
            .layers()
            .iter()
            .find(|l| l.id() == id)
            .map_or(0, |l| l.nodes().len())
    };
    let pal = &cfg.palette;
    let rej = pal.rejected_future.layer_id;
    let sel = pal.selected_future.layer_id;
    assert!(count(&p, rej) < count(&full, rej));
    assert_eq!(count(&p, sel), count(&full, sel));
}
