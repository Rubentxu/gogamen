use gogamen_lab_grid::trap_zoo::*;

fn failing_worlds(p: Principle, h: usize, n: u8, k: usize) -> Vec<TrapWorld> {
    (0..5000)
        .map(|s| TrapWorld::generate(s, n))
        .filter(|w| fails(w, p, h))
        .take(k)
        .collect()
}

#[test]
fn trap_zoo_reducer_is_deterministic_and_one_minimal() {
    for (p, h) in [
        (Principle::GreedyFinalEntropy, 2),
        (Principle::TrajectoryEntropy, 3),
    ] {
        let ws = failing_worlds(p, h, 7, 3);
        assert_eq!(ws.len(), 3, "{p:?}: se esperaban 3 mundos que fallan");
        for w in ws {
            let a = reduce(&w, p, h).unwrap();
            assert_eq!(a, reduce(&w, p, h).unwrap(), "determinista");
            assert!(fails(&a.world, p, h));
            // Exhaustivo: toda eliminación individual hace desaparecer el fallo.
            for r in single_removals(&a.world) {
                assert!(!fails(&apply_removal(&a.world, &r), p, h), "{r:?}");
            }
            assert!(a.world.num_states() <= w.num_states());
            assert!(a.world.num_edges() <= w.num_edges());
            assert!(a.accepted <= w.num_states() + w.num_edges());
        }
    }
}

#[test]
fn trap_zoo_control_principle_never_fails() {
    for n in [5u8, 6, 7] {
        for s in 0..300 {
            assert!(!fails(
                &TrapWorld::generate(s, n),
                Principle::SurvivalMass,
                3
            ));
        }
    }
}

#[test]
fn trap_zoo_fixtures_reload_and_verify() {
    let zoo = load_zoo(&default_zoo_dir()).unwrap();
    assert!(zoo.len() >= 6, "zoo size {}", zoo.len());
    let seeds: std::collections::BTreeSet<_> =
        zoo.iter().map(|r| (r.principle.name(), r.seed)).collect();
    assert!(seeds.len() >= 3);
    for r in &zoo {
        r.verify().unwrap_or_else(|e| panic!("{}: {e}", r.name()));
        assert_eq!(TrapRecord::from_text(&r.to_text()).unwrap(), *r);
        // El archivo en disco es la serialización estable.
        let disk =
            std::fs::read_to_string(default_zoo_dir().join(format!("{}.trap", r.name()))).unwrap();
        assert_eq!(disk, r.to_text());
    }
}

#[test]
fn trap_zoo_fixtures_match_regeneration() {
    let zoo = load_zoo(&default_zoo_dir()).unwrap();
    let mut regen = build_zoo(&[5, 6, 7], 2, 5000);
    regen.sort_by_key(|r| r.name());
    let mut names: Vec<_> = zoo.iter().map(|r| r.name()).collect();
    names.sort();
    assert_eq!(names, regen.iter().map(|r| r.name()).collect::<Vec<_>>());
}

fn tmp(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("trap_zoo_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

#[test]
fn trap_zoo_score_command_generates_gallery() {
    let out = tmp("lib");
    let args: Vec<String> = ["score", "--principle", "greedy_final_entropy", "--out-dir"]
        .iter()
        .map(|s| s.to_string())
        .chain([out.to_string_lossy().into_owned()])
        .collect();
    let table = run_cli(&args).unwrap();
    assert!(table.contains("verdict=FAILS_ON_ZOO"));
    let zoo = load_zoo(&default_zoo_dir()).unwrap();
    for r in &zoo {
        let svg = std::fs::read_to_string(out.join(format!("{}.svg", r.name()))).unwrap();
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
    }
    assert!(out.join("score.txt").exists());
    let ctrl = run_cli(&[
        "score".into(),
        "--principle".into(),
        "survival_mass".into(),
        "--out-dir".into(),
        out.to_string_lossy().into_owned(),
    ])
    .unwrap();
    assert!(ctrl.contains("failed=0/") && ctrl.contains("NULL_ON_ZOO"));
    assert!(
        run_cli(&[
            "score".into(),
            "--principle".into(),
            "nope".into(),
            "--out-dir".into(),
            "x".into()
        ])
        .is_err()
    );
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn trap_zoo_binary_runs_once() {
    let out = tmp("bin");
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_trap_zoo"))
        .args(["score", "--principle", "trajectory_entropy", "--out-dir"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(o.status.success());
    assert!(String::from_utf8_lossy(&o.stdout).contains("principle=trajectory_entropy"));
    let _ = std::fs::remove_dir_all(&out);
}
