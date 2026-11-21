//! GOG-115: aislamiento temporal del visor. Aserciones duras sólo estructurales; los tiempos se
//! reportan (no son WCET) con, a lo sumo, cotas muy generosas.

use std::time::Duration;

use gogamen_studio::viewer_isolation::{
    Budget, CountingAllocator, IsolationConfig, IsolationReport, ModeResult, ViewerMode,
    common_prefix_len, counting_allocator_active, hash_sequences_identical, run_mode,
};

#[global_allocator]
static ALLOC: CountingAllocator = CountingAllocator;

fn cfg() -> IsolationConfig {
    IsolationConfig {
        ticks: 1500,
        ..IsolationConfig::default()
    }
}

fn structural(r: &ModeResult) {
    // Cola acotada y control nunca bloqueado/rechazado.
    assert!(r.queue_max_depth <= r.queue_capacity, "{:?}", r.mode);
    assert_eq!(r.control_sent, r.ticks_done);
    assert_eq!(
        r.control_applied, r.control_sent,
        "{:?} control acks",
        r.mode
    );
    // Huecos contados == producidos - entregados, y desglose exacto.
    assert_eq!(r.gaps_total, r.frames_produced - r.frames_delivered);
    assert_eq!(
        r.gaps_total,
        r.dropped_ring + r.undelivered + r.frames_not_published,
        "{:?} gap breakdown",
        r.mode
    );
    assert_eq!(
        r.frames_produced,
        r.frames_published + r.frames_not_published
    );
    assert!(r.consumer_observed_gaps <= r.dropped_ring);
    // Cota generosa de cordura (no es una afirmación de tiempo real): 5 s por tick.
    assert!(r.tick_total.max < 5_000_000_000);
}

#[test]
fn four_modes_structural_properties_and_identical_decisions() {
    let c = cfg();
    let rs: Vec<_> = ViewerMode::ALL
        .iter()
        .map(|m| run_mode(&c, *m, Budget::Logical))
        .collect();
    for r in &rs {
        assert_eq!(r.ticks_done, c.ticks);
        structural(r);
    }
    // Determinismo bajo presupuesto lógico: misma secuencia de hashes en los cuatro modos.
    assert!(hash_sequences_identical(&rs));
    let by = |m: ViewerMode| rs.iter().find(|r| r.mode == m).unwrap();
    // NoViewer: nada publicado, nada entregado.
    let nv = by(ViewerMode::NoViewer);
    assert_eq!(nv.frames_published, 0);
    assert_eq!(nv.frames_delivered, 0);
    // Slow: la cola se satura, hay descartes visibles y el control sigue (ya asertado).
    let slow = by(ViewerMode::Slow);
    assert!(
        slow.dropped_ring > 0,
        "slow consumer must produce visible drops"
    );
    assert!(slow.frames_delivered < slow.frames_produced);
    // Disconnected: el visor se fue a mitad; el resto no se publica y el motor completó.
    let d = by(ViewerMode::Disconnected);
    assert!(d.viewer_disconnected);
    assert!(d.frames_not_published > 0);
    assert_eq!(d.ticks_done, c.ticks);
    // Normal: el visor renderizó.
    assert!(by(ViewerMode::Normal).viewer_renders > 0);
    // Alloc: con el allocator instalado se mide.
    assert!(counting_allocator_active());
    assert!(nv.allocs_sensor.is_some());
}

#[test]
fn realtime_deadline_difference_is_recorded_not_asserted_equal() {
    let c = IsolationConfig {
        ticks: 5_000_000, // tope alto: manda el plazo de reloj real
        ..cfg()
    };
    let d = Duration::from_millis(30);
    let rs: Vec<_> = ViewerMode::ALL
        .iter()
        .map(|m| run_mode(&c, *m, Budget::WallClock(d)))
        .collect();
    let base = &rs[0];
    for r in &rs {
        assert!(r.ticks_done > 0);
        // Lo único garantizado: las decisiones coinciden en el prefijo común (mismo origen).
        let n = base.ticks_done.min(r.ticks_done) as usize;
        assert_eq!(common_prefix_len(base, r), n, "{:?} prefix", r.mode);
        // Se registra (no se asevera igualdad) el nº de ticks completados.
        eprintln!(
            "realtime {:?}: ticks_done={} (NoViewer {})",
            r.mode, r.ticks_done, base.ticks_done
        );
    }
}

#[test]
fn report_contains_platform_seed_disclaimer_and_percentiles() {
    let rep = IsolationReport::run(
        IsolationConfig {
            ticks: 300,
            ..cfg()
        },
        Some(Duration::from_millis(5)),
    );
    let md = rep.to_markdown();
    for needle in [
        "not WCET",
        "os=",
        "arch=",
        "cpus=",
        "seed (master): 42",
        "p50/p99/max",
        "NoViewer",
        "Disconnected",
        "identical across modes: true",
        "Real-time deadline",
    ] {
        assert!(md.contains(needle), "missing {needle}\n{md}");
    }
}
