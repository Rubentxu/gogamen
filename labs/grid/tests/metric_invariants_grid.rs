//! GOG-197: adaptador labs/grid -> kit de invariantes de mundo de `gogamen-sdk::pocket_kit`.
//!
//! Conecta `entropy_controls::measure_controls` (GOG-155) a `PocketMeasure`:
//! - `marginal_h` = `proxy_marginal_entropy` (muestreado, con error estándar y `bin_width`).
//! - `control` = `control_bits` exacto I(A;S_T) (secuencias de acciones en bucle abierto; la acción
//!   `i` de un estado es la `i`-ésima por etiqueta, índice `% n_acciones`).
//! - puntuación `(s, a)` = control exacto tras fijar `a` como única acción de `s`
//!   ("control restante tras elegir a"). Es exacta, por lo que NO depende de la malla: el invariante
//!   de refinamiento pasa por construcción (límite documentado).
//!
//! Los mundos solo usan etiquetas < 256 (`MicroState(u8)`).

use gogamen_lab_grid::bool_world::{BoolWorld, MicroState};
use gogamen_lab_grid::entropy_controls::{ControlConfig, measure_controls};
use gogamen_sdk::pocket_kit::{
    ActionScore, PocketKitConfig, PocketMeasure, PocketMeasurement, PocketWorld, Quantity,
    pocket_worlds, run_pocket_kit,
};

struct GridMeasure;

fn ms(l: u32) -> Result<MicroState, String> {
    u8::try_from(l)
        .map(MicroState)
        .map_err(|_| format!("etiqueta {l} fuera de u8"))
}

fn to_bool(w: &PocketWorld, only: Option<(u32, u32)>) -> Result<BoolWorld, String> {
    let mut out = BoolWorld::new(w.seed);
    for s in &w.states {
        let mut acts: Vec<_> = s.actions.iter().collect();
        acts.sort_by_key(|a| a.label);
        let vec = acts
            .iter()
            .filter(|a| only.is_none_or(|(st, al)| st != s.label || al == a.label))
            .map(|a| {
                a.outcomes
                    .iter()
                    .map(|o| ms(o.0).map(|d| (d, o.1)))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        out = out
            .with_state(ms(s.label)?, true, vec)
            .map_err(|e| format!("{e:?}"))?;
    }
    Ok(out)
}

impl PocketMeasure for GridMeasure {
    fn measure(
        &mut self,
        w: &PocketWorld,
        horizon: usize,
        bin_width: u32,
        seed: u64,
    ) -> Result<PocketMeasurement, String> {
        let mut cfg = ControlConfig::new(ms(w.initial)?, horizon);
        cfg.bin_width = u8::try_from(bin_width).map_err(|e| e.to_string())?;
        cfg.seed = seed;
        let m = measure_controls(&to_bool(w, None)?, &cfg).map_err(|e| format!("{e:?}"))?;
        let p = m.proxy_marginal_entropy;
        let mut scores = Vec::new();
        for s in &w.states {
            for a in &s.actions {
                let mut c = ControlConfig::new(ms(s.label)?, horizon);
                c.samples = 1;
                let r = measure_controls(&to_bool(w, Some((s.label, a.label)))?, &c)
                    .map_err(|e| format!("{e:?}"))?;
                scores.push(ActionScore {
                    state: s.label,
                    action: a.label,
                    score: r.control_bits.value,
                });
            }
        }
        Ok(PocketMeasurement {
            marginal_h: Quantity {
                value: p.value,
                sampling_error: p.sampling_error,
                bin_width: p.bin_width,
                samples: p.samples,
            },
            control: Quantity::exact(m.control_bits.value, bin_width),
            action_scores: scores,
            score_sampling_error: 0.0,
        })
    }
}

#[test]
fn metric_invariants_grid_adapter_reports_real_controls() {
    let cfg = PocketKitConfig::default();
    let r = run_pocket_kit(
        "labs/grid entropy_controls",
        &mut GridMeasure,
        &pocket_worlds(),
        &cfg,
    );
    println!("{}", r.render());
    // Separación H / control medida con el código real: el ruido sube H, no el control.
    let fork = r.rows.iter().find(|x| x.world_id == "fork_sinks").unwrap();
    let (bh, bc) = fork.base.unwrap();
    let (nh, nc) = fork.noise.unwrap();
    assert!((bc.value - 1.0).abs() < 1e-9);
    assert!(nh.value > bh.value + 0.5, "H {} -> {}", bh.value, nh.value);
    assert!((nc.value - bc.value).abs() < 1e-9);
    assert!(bh.sampling_error > 0.0 && bh.samples > 0, "H es muestreada");
    assert_eq!(bc.sampling_error, 0.0);
    // Invariantes sobre el código real.
    for id in [
        "label_permutation_measure",
        "label_permutation_ranking",
        "duplicate_predicate_no_gain",
        "exogenous_noise_no_gain",
        "random_memory_no_gain",
        "mesh_refinement_ranking",
    ] {
        let res = r.result(id).unwrap();
        assert!(res.failure.is_none(), "{}", r.render());
    }
}
