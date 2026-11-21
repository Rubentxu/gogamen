//! Tests del oráculo de bolsillo DP (GOG-196) frente a la enumeración de `BoolWorld`.

use std::collections::BTreeMap;

use gogamen_lab_grid::bool_world::{
    BoolSamplingPolicy, BoolWorld, BoolWorldError, MicroState, OracleLimits,
};
use gogamen_lab_grid::pocket_oracle::{PocketOracleError, PocketReport, solve_pocket};

const EPS: f64 = 1e-12;

fn st(i: u8) -> MicroState {
    MicroState(i)
}

fn det(targets: &[u8]) -> Vec<Vec<(MicroState, f64)>> {
    targets.iter().map(|&t| vec![(st(t), 1.0)]).collect()
}

/// Enumera todas las secuencias de acciones viables de longitud `h`; devuelve
/// `(probabilidad bajo pi_DP, estado final)` por secuencia.
fn brute(
    world: &BoolWorld,
    rep: &PocketReport,
    s: MicroState,
    h: usize,
    p: f64,
    out: &mut Vec<(f64, MicroState)>,
) {
    if h == 0 {
        out.push((p, s));
        return;
    }
    let pol = rep.policy(h, s);
    let spec = world.state_specs().find(|x| x.state == s).unwrap();
    let succs: Vec<MicroState> = if spec.actions.is_empty() {
        vec![s]
    } else {
        spec.actions.iter().map(|a| a[0].0).collect()
    };
    for (a, d) in succs.into_iter().enumerate() {
        let viable = world.state_specs().find(|x| x.state == d).unwrap().viable;
        if !viable {
            continue;
        }
        let q = pol.iter().find(|(i, _, _)| *i == a).map_or(0.0, |x| x.2);
        brute(world, rep, d, h - 1, p * q, out);
    }
}

fn check_vs_brute(world: &BoolWorld, rep: &PocketReport) {
    let mut seqs = Vec::new();
    brute(world, rep, rep.initial, rep.horizon, 1.0, &mut seqs);
    let n = seqs.len() as f64;
    assert!((rep.log2_viable_sequences.0 - n.log2()).abs() < 1e-9);
    assert!((rep.trajectory_entropy_bits.0 - n.log2()).abs() < 1e-9);
    let total: f64 = seqs.iter().map(|x| x.0).sum();
    assert!((total - 1.0).abs() < 1e-9);
    let mut fin: BTreeMap<MicroState, f64> = BTreeMap::new();
    for (p, s) in &seqs {
        assert!((p - 1.0 / n).abs() < 1e-9, "uniforme sobre secuencias");
        *fin.entry(*s).or_default() += p;
    }
    let fh: f64 = fin
        .values()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.log2())
        .sum();
    assert!((rep.final_state_entropy_bits.0 - fh).abs() < 1e-9);
}

/// Fixture A: 0:{->1,->2}, 1:{->0,->3(inviable)}, 2 absorbente, 3 inviable.
fn fixture_a() -> BoolWorld {
    BoolWorld::new(11)
        .with_state(st(0), true, det(&[1, 2]))
        .unwrap()
        .with_state(st(1), true, det(&[0, 3]))
        .unwrap()
        .with_state(st(2), true, vec![])
        .unwrap()
        .with_state(st(3), false, vec![])
        .unwrap()
}

/// Fixture B: 0:{->0,->1}, 1:{->2}, 2 absorbente; todos viables.
fn fixture_b() -> BoolWorld {
    BoolWorld::new(12)
        .with_state(st(0), true, det(&[0, 1]))
        .unwrap()
        .with_state(st(1), true, det(&[2]))
        .unwrap()
        .with_state(st(2), true, vec![])
        .unwrap()
}

#[test]
fn pocket_oracle_fixture_a_hand_numbers() {
    // Calculo a mano: N_3(s0) = 3 -> log2 3; final: s1 1/3, s2 2/3 -> H(1/3);
    // supervivencia optima 1; uniforme 3/4.
    let w = fixture_a();
    let r = solve_pocket(&w, st(0), 3).unwrap();
    assert!((r.log2_viable_sequences.0 - 3f64.log2()).abs() < EPS);
    assert!((r.trajectory_entropy_bits.0 - 3f64.log2()).abs() < EPS);
    let h13 = -(1.0 / 3.0) * (1.0f64 / 3.0).log2() - (2.0 / 3.0) * (2.0f64 / 3.0).log2();
    assert!((r.final_state_entropy_bits.0 - h13).abs() < EPS);
    assert!((h13 - 0.9182958340544896).abs() < 1e-12);
    assert!((r.survival_best_policy.0 - 1.0).abs() < EPS);
    assert!((r.survival_uniform_policy.0 - 0.75).abs() < EPS);
    let pol = r.policy(3, st(0));
    assert_eq!(pol.len(), 2);
    assert!((pol[0].2 - 2.0 / 3.0).abs() < EPS && (pol[1].2 - 1.0 / 3.0).abs() < EPS);
    check_vs_brute(&w, &r);
    // Enumeracion: 3 trayectorias de estados, viable_mass uniforme 3/4.
    let e = w
        .oracle(
            st(0),
            &BoolSamplingPolicy::UniformActions,
            3,
            OracleLimits::default(),
        )
        .unwrap();
    assert_eq!(e.trajectories.len(), 3);
    assert!((e.viable_mass - r.survival_uniform_policy.0).abs() < EPS);
}

#[test]
fn pocket_oracle_fixture_b_hand_numbers() {
    // N_3(s0) = 4 -> 2 bits; final: s0 1/4, s1 1/4, s2 1/2 -> 1.5 bits; survival 1 y 1.
    let w = fixture_b();
    let r = solve_pocket(&w, st(0), 3).unwrap();
    assert!((r.log2_viable_sequences.0 - 2.0).abs() < EPS);
    assert!((r.trajectory_entropy_bits.0 - 2.0).abs() < EPS);
    assert!((r.final_state_entropy_bits.0 - 1.5).abs() < EPS);
    assert!((r.survival_best_policy.0 - 1.0).abs() < EPS);
    assert!((r.survival_uniform_policy.0 - 1.0).abs() < EPS);
    check_vs_brute(&w, &r);
    let e = w
        .oracle(
            st(0),
            &BoolSamplingPolicy::UniformActions,
            3,
            OracleLimits::default(),
        )
        .unwrap();
    assert_eq!(e.trajectories.len(), 4);
}

#[test]
fn pocket_oracle_rejects_stochastic_and_bad_input() {
    let w = BoolWorld::new(1)
        .with_state(st(0), true, vec![vec![(st(0), 0.5), (st(1), 0.5)]])
        .unwrap()
        .with_state(st(1), true, vec![])
        .unwrap();
    assert_eq!(
        solve_pocket(&w, st(0), 2).unwrap_err(),
        PocketOracleError::StochasticTransition { state: st(0) }
    );
    let a = fixture_a();
    assert_eq!(
        solve_pocket(&a, st(0), 0).unwrap_err(),
        PocketOracleError::InvalidHorizon
    );
    assert_eq!(
        solve_pocket(&a, st(3), 2).unwrap_err(),
        PocketOracleError::InvalidInitialState(st(3))
    );
    // Sin secuencia viable: 0 -> 1 inviable.
    let dead = BoolWorld::new(2)
        .with_state(st(0), true, det(&[1]))
        .unwrap()
        .with_state(st(1), false, vec![])
        .unwrap();
    assert_eq!(
        solve_pocket(&dead, st(0), 1).unwrap_err(),
        PocketOracleError::NoViableSequence
    );
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// Mundo determinista con sucesores distintos por estado (incluye inviables y absorbentes).
fn random_world(seed: u64) -> BoolWorld {
    let mut rng = seed;
    let n = 2 + (splitmix(&mut rng) % 7) as u8; // 2..=8
    let mut w = BoolWorld::new(seed);
    for s in 0..n {
        let viable = !splitmix(&mut rng).is_multiple_of(5);
        let k = (splitmix(&mut rng) % 4) as usize; // 0..=3 acciones
        let mut targets: Vec<u8> = Vec::new();
        while targets.len() < k.min(n as usize) {
            let t = (splitmix(&mut rng) % u64::from(n)) as u8;
            if !targets.contains(&t) {
                targets.push(t);
            }
        }
        w = w.with_state(st(s), viable, det(&targets)).unwrap();
    }
    w
}

#[test]
fn pocket_oracle_property_1000_worlds() {
    let mut compared = 0;
    for seed in 0..1000u64 {
        let w = random_world(seed.wrapping_mul(0x2545F4914F6CDD1D) ^ 0xC0FFEE);
        let Some(init) = w.state_specs().find(|s| s.viable).map(|s| s.state) else {
            continue;
        };
        for h in 1..=4usize {
            let e = w.oracle(
                init,
                &BoolSamplingPolicy::UniformActions,
                h,
                OracleLimits::default(),
            );
            match (solve_pocket(&w, init, h), e) {
                (Ok(r), Ok(e)) => {
                    compared += 1;
                    assert_eq!(
                        e.trajectories.len() as f64,
                        r.log2_viable_sequences.0.exp2().round()
                    );
                    assert!((e.viable_mass - r.survival_uniform_policy.0).abs() < 1e-9);
                    assert_eq!(r.survival_best_policy.0, 1.0);
                    check_vs_brute(&w, &r);
                    // pi_h(a|s) = N_{h-1}(s'_a)/N_h(s) y suma 1 en todo estado con N_h>0.
                    for spec in w.state_specs() {
                        let pol = r.policy(h, spec.state);
                        if pol.is_empty() {
                            continue;
                        }
                        let sum: f64 = pol.iter().map(|x| x.2).sum();
                        assert!((sum - 1.0).abs() < 1e-9);
                        for (_, d, p) in pol {
                            let want =
                                (r.log2_count(h - 1, d) - r.log2_count(h, spec.state)).exp2();
                            assert!((p - want).abs() < 1e-12);
                        }
                    }
                    // Determinismo: misma seed + esquema => mismo hash.
                    assert_eq!(
                        r.stable_hash,
                        solve_pocket(&w, init, h).unwrap().stable_hash
                    );
                }
                (
                    Err(PocketOracleError::NoViableSequence),
                    Err(BoolWorldError::NoViableTrajectory),
                ) => {}
                (a, b) => panic!(
                    "discrepancia seed {seed} h {h}: {:?} vs {:?}",
                    a.map(|r| r.horizon),
                    b.map(|e| e.horizon)
                ),
            }
        }
    }
    assert!(compared > 1000, "comparaciones efectivas: {compared}");
}

#[test]
fn pocket_oracle_linear_cost_64_states_h64() {
    let mut w = BoolWorld::new(64);
    for i in 0..64u8 {
        w = w
            .with_state(
                st(i),
                true,
                det(&[
                    (i + 1) % 64,
                    ((u16::from(i) * 5 + 2) % 64) as u8,
                    ((u16::from(i) * 3 + 7) % 64) as u8,
                ]),
            )
            .unwrap();
    }
    let r = solve_pocket(&w, st(0), 64).unwrap();
    assert_eq!(r.states, 64);
    assert_eq!(r.state_action_pairs, 64 * 3);
    assert_eq!(r.ops_backward, 64 * 64 * 3); // H*S*A exacto
    assert_eq!(r.ops_forward, r.ops_backward);
    // Todos viables con 3 acciones: N_64 = 3^64 (> u64), sin desbordar en log2.
    assert!((r.log2_viable_sequences.0 - 64.0 * 3f64.log2()).abs() < 1e-9);
    assert!((r.trajectory_entropy_bits.0 - r.log2_viable_sequences.0).abs() < 1e-9);
    assert!(r.final_state_entropy_bits.0 <= 6.0 + 1e-9);
    let again = solve_pocket(&w, st(0), 64).unwrap();
    assert_eq!(r.stable_hash, again.stable_hash);
    let other = solve_pocket(&w, st(0), 63).unwrap();
    assert_ne!(r.stable_hash, other.stable_hash);
}
