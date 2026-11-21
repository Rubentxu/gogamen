//! GOG-197: kit de invariantes de métricas a nivel de mundo de bolsillo.

use gogamen_sdk::pocket_kit::{
    ActionScore, POCKET_INVARIANT_IDS, POCKET_KIT_VERSION, PocketKitConfig, PocketMeasure,
    PocketMeasurement, PocketWorld, Quantity, ReferencePocketMeasure, add_exogenous_noise,
    duplicate_predicate, first_action_distributions, permute_labels, pocket_kit_hash,
    pocket_worlds, run_pocket_kit, write_random_memory,
};

fn cfg() -> PocketKitConfig {
    PocketKitConfig::default()
}

/// Envuelve la referencia y aplica un `tweak` defectuoso.
struct Broken<F: FnMut(&PocketWorld, u32, &mut PocketMeasurement)>(F);
impl<F: FnMut(&PocketWorld, u32, &mut PocketMeasurement)> PocketMeasure for Broken<F> {
    fn measure(
        &mut self,
        w: &PocketWorld,
        h: usize,
        bw: u32,
        seed: u64,
    ) -> Result<PocketMeasurement, String> {
        let mut m = ReferencePocketMeasure.measure(w, h, bw, seed)?;
        (self.0)(w, bw, &mut m);
        Ok(m)
    }
}

fn q(m: &mut Quantity, v: f64) {
    m.value = v;
}

#[test]
fn metric_invariants_five_worlds_and_reference_passes() {
    let worlds = pocket_worlds();
    assert_eq!(worlds.len(), 5);
    for w in &worlds {
        w.validate().unwrap();
    }
    let r = run_pocket_kit("reference", &mut ReferencePocketMeasure, &worlds, &cfg());
    assert!(r.passed(), "{}", r.render());
    assert_eq!(r.results.len(), POCKET_INVARIANT_IDS.len());
    assert_eq!(r.kit_version, POCKET_KIT_VERSION);
    assert_eq!(r.kit_hash, pocket_kit_hash());
    // Determinista.
    let r2 = run_pocket_kit("reference", &mut ReferencePocketMeasure, &worlds, &cfg());
    assert_eq!(r.render(), r2.render());
    println!("{}", r.render());
}

#[test]
fn metric_invariants_report_separates_h_and_control() {
    let r = run_pocket_kit(
        "reference",
        &mut ReferencePocketMeasure,
        &pocket_worlds(),
        &cfg(),
    );
    let noise_row = r.rows.iter().find(|x| x.world_id == "fork_sinks").unwrap();
    let (bh, bc) = noise_row.base.unwrap();
    let (nh, nc) = noise_row.noise.unwrap();
    assert!((bc.value - 1.0).abs() < 1e-9, "fork: 1 bit de control");
    assert!(nh.value > bh.value + 0.5, "ruido sube H");
    assert!((nc.value - bc.value).abs() < 1e-9, "ruido no sube control");
    assert_eq!((bc.sampling_error, bc.bin_width, bc.samples), (0.0, 1, 0));
    let pure = r.rows.iter().find(|x| x.world_id == "pure_noise").unwrap();
    let (ph, pc) = pure.base.unwrap();
    assert!((ph.value - 1.0).abs() < 1e-9 && pc.value.abs() < 1e-9);
    let text = r.render();
    assert!(text.contains("bits") && text.contains("bin_width=") && text.contains("se="));
}

#[test]
fn metric_invariants_transforms_are_valid_and_deterministic() {
    for w in pocket_worlds() {
        let (p1, perm1) = permute_labels(&w, 7);
        let (p2, _) = permute_labels(&w, 7);
        assert_eq!(p1, p2);
        assert_ne!(
            perm1.states,
            perm1.states.keys().map(|k| (*k, *k)).collect()
        );
        p1.validate().unwrap();
        duplicate_predicate(&w).unwrap().validate().unwrap();
        add_exogenous_noise(&w).unwrap().validate().unwrap();
        write_random_memory(&w).unwrap().validate().unwrap();
    }
}

#[test]
fn metric_invariants_broken_label_sensitive_fails_permutation() {
    let mut m = Broken(|w: &PocketWorld, _bw, m: &mut PocketMeasurement| {
        m.control.value += 0.05 * f64::from(w.initial);
        for s in &mut m.action_scores {
            s.score += 0.01 * f64::from(s.action);
        }
    });
    let r = run_pocket_kit("label_sensitive", &mut m, &pocket_worlds(), &cfg());
    for id in ["label_permutation_measure", "label_permutation_ranking"] {
        let f = r
            .result(id)
            .unwrap()
            .failure
            .as_ref()
            .unwrap_or_else(|| panic!("{id}\n{}", r.render()));
        assert!(f.counterexample.states.len() <= f.original_states);
        assert!(!f.world_id.is_empty());
        println!("{id}: {}\n  {}", f.counterexample.render(), f.detail);
    }
    assert!(
        r.result("duplicate_predicate_no_gain")
            .unwrap()
            .failure
            .is_none()
    );
}

#[test]
fn metric_invariants_broken_bitwise_sum_fails_duplication() {
    // Suma de I(A1; bit_j(S_T)): una copia del predicado cuenta dos veces.
    let mut m = Broken(|w: &PocketWorld, _bw, m: &mut PocketMeasurement| {
        let per = first_action_distributions(w, w.initial, 2).unwrap();
        let n = per.len() as f64;
        let mut total = 0.0;
        for bit in 0..8u32 {
            let p1 = |d: &std::collections::BTreeMap<u32, f64>| {
                d.iter()
                    .filter(|(l, _)| (*l >> bit) & 1 == 1)
                    .map(|x| x.1)
                    .sum::<f64>()
            };
            let h = |p: f64| {
                if p <= 0.0 || p >= 1.0 {
                    0.0
                } else {
                    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
                }
            };
            let mean = per.iter().map(|(_, d)| p1(d)).sum::<f64>() / n;
            total += h(mean) - per.iter().map(|(_, d)| h(p1(d))).sum::<f64>() / n;
        }
        q(&mut m.control, total);
    });
    let r = run_pocket_kit("bitwise_sum", &mut m, &pocket_worlds(), &cfg());
    let f = r
        .result("duplicate_predicate_no_gain")
        .unwrap()
        .failure
        .as_ref()
        .unwrap_or_else(|| panic!("{}", r.render()));
    assert!(f.detail.contains("control rose"));
    assert!(f.counterexample.states.len() < f.original_states || f.original_states <= 3);
    println!("duplication: {}\n  {}", f.counterexample.render(), f.detail);
}

#[test]
fn metric_invariants_broken_marginal_as_control_fails_noise_and_memory() {
    let mut m = Broken(|_w: &PocketWorld, _bw, m: &mut PocketMeasurement| {
        let h = m.marginal_h.value;
        q(&mut m.control, h);
    });
    let r = run_pocket_kit("marginal_as_control", &mut m, &pocket_worlds(), &cfg());
    for id in ["exogenous_noise_no_gain", "random_memory_no_gain"] {
        let f = r
            .result(id)
            .unwrap()
            .failure
            .as_ref()
            .unwrap_or_else(|| panic!("{id}"));
        assert_eq!(f.kit_seed, 197);
        println!(
            "{id}: world={} seed={} {}\n  {}",
            f.world_id,
            f.world_seed,
            f.counterexample.render(),
            f.detail
        );
    }
}

#[test]
fn metric_invariants_broken_mesh_flipper_fails_refinement() {
    let mut m = Broken(|_w: &PocketWorld, bw, m: &mut PocketMeasurement| {
        if bw == 1 {
            for ActionScore { score, .. } in &mut m.action_scores {
                *score = -*score;
            }
        }
    });
    let r = run_pocket_kit("mesh_flipper", &mut m, &pocket_worlds(), &cfg());
    let f = r
        .result("mesh_refinement_ranking")
        .unwrap()
        .failure
        .as_ref()
        .unwrap_or_else(|| panic!("{}", r.render()));
    assert!(f.detail.contains("flipped"));
    println!("mesh: {}\n  {}", f.counterexample.render(), f.detail);
    // Con un umbral declarado del 100% la misma medida pasa (el umbral es explícito).
    let lax = PocketKitConfig {
        max_flip_fraction: 1.0,
        ..cfg()
    };
    let r2 = run_pocket_kit("mesh_flipper", &mut m, &pocket_worlds(), &lax);
    assert!(
        r2.result("mesh_refinement_ranking")
            .unwrap()
            .failure
            .is_none()
    );
}

#[test]
fn metric_invariants_failure_is_reported_in_render() {
    let mut m = Broken(|_w: &PocketWorld, _bw, m: &mut PocketMeasurement| {
        let h = m.marginal_h.value;
        q(&mut m.control, h);
    });
    let text = run_pocket_kit("x", &mut m, &pocket_worlds(), &cfg()).render();
    assert!(text.contains("[FAIL] exogenous_noise_no_gain world="));
    assert!(
        text.contains("world_seed=") && text.contains("kit_seed=") && text.contains("limitations")
    );
}
