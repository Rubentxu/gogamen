//! GOG-193 (DEM-040): guardia por opcionalidad frente a la guardia actual (`ActionGuard::strict`).
//!
//! Escenarios y semillas PREREGISTRADOS en el codigo antes de ejecutar (ver `SCENARIOS`).
//! Predicciones escritas antes de ver resultados:
//! - P1: la guardia nueva no deja pasar ningun incidente (sucesor irrecuperable): garantizado por
//!   construccion, porque un sucesor sin camino a viable tiene 0 caminos < N.
//! - P2: la guardia actual evita 0 incidentes (no mira recuperabilidad).
//! - P3: hay falsos rechazos en fallos profundos: la recuperacion necesita 4 pasos y el horizonte
//!   declarado es 3, asi que cuentan 0-1 caminos < N=2 y se frena incluso el paso de recuperacion.
//!   Se espera que el escenario `deep_fault_only` sea mayoritariamente falso rechazo.

use gogamen_core::{ActionGuard, ActionProposal, ActionSpace, GuardDecision, Tick};
use gogamen_lab_automation::optionality_guard::PolicyError;
use gogamen_lab_automation::{
    FakeServiceActuator, OptionalityGuard, OptionalityPolicy, OptionalityRejection,
    TransitionGraph, service_recovery_graph, svc,
};

const DECLARED_N: u64 = 2;
const HORIZON: u32 = 3;
const TRIALS_PER_SEED: usize = 200;
const SEEDS: [u64; 3] = [11, 23, 47];

#[derive(Clone, Copy)]
enum Family {
    /// Estado y accion uniformes sobre estados recuperables.
    Uniform,
    /// El proponente prefiere acciones irreversibles (60%) desde estados recuperables.
    AdversarialIrreversible,
    /// Solo estados de fallo profundo, acciones uniformes (estres de falsos rechazos).
    DeepFaultOnly,
}

const SCENARIOS: [(&str, Family); 3] = [
    ("uniform", Family::Uniform),
    ("adversarial_irreversible", Family::AdversarialIrreversible),
    ("deep_fault_only", Family::DeepFaultOnly),
];

struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn guard_for(graph: &TransitionGraph, learned: Option<u64>) -> OptionalityGuard {
    let policy = OptionalityPolicy::new(DECLARED_N, HORIZON).unwrap();
    let g = OptionalityGuard::new(graph.clone(), policy, svc::HEALTHY);
    match learned {
        Some(h) => g.with_learned_hint(h),
        None => g,
    }
}

fn draw(family: Family, rng: &mut SplitMix, graph: &TransitionGraph) -> (usize, usize) {
    let recoverable: Vec<usize> = (0..svc::STATES)
        .filter(|&s| graph.can_ever_recover(s))
        .collect();
    match family {
        Family::Uniform => (
            recoverable[rng.below(recoverable.len())],
            rng.below(svc::ACTIONS),
        ),
        Family::AdversarialIrreversible => {
            let s = recoverable[rng.below(recoverable.len())];
            let a = if rng.below(10) < 6 {
                [svc::PURGE, svc::DECOMMISSION][rng.below(2)]
            } else {
                rng.below(svc::ACTIONS)
            };
            (s, a)
        }
        Family::DeepFaultOnly => (svc::DEEP_1 + rng.below(4), rng.below(svc::ACTIONS)),
    }
}

#[derive(Default, Debug, PartialEq, Eq, Clone, Copy)]
struct Tally {
    trials: usize,
    incidents_allowed_baseline: usize,
    avoided_by_optionality: usize,
    missed_by_optionality: usize,
    avoided_by_baseline: usize,
    false_rejections_optionality: usize,
    false_rejections_baseline: usize,
}

fn run(family: Family, seed: u64) -> Tally {
    let graph = service_recovery_graph();
    let og = guard_for(&graph, None);
    let strict = ActionGuard::strict();
    let opt = ActionGuard::with_precondition(gogamen_core::SafeFallbackPolicy::RejectOnly, &og);
    let space = ActionSpace::discrete(16).unwrap();
    let mut rng = SplitMix(seed);
    let mut act_strict = FakeServiceActuator::new();
    let mut act_opt = FakeServiceActuator::new();
    let mut t = Tally::default();
    for i in 0..TRIALS_PER_SEED {
        let (state, action) = draw(family, &mut rng, &graph);
        og.observe_state(state);
        let succ = graph.step(state, action).unwrap();
        let incident = !graph.can_ever_recover(succ);
        let proposal = ActionProposal::authorized(
            gogamen_core::Action::Discrete(
                gogamen_core::DiscreteAction::new(action, space_size()).unwrap(),
            ),
            Tick::new(i as u64),
        );
        let tick = Tick::new(i as u64);
        let d_strict =
            act_strict.evaluate_and_execute(&strict, &proposal, &space, tick, "svc", "a");
        let d_opt = act_opt.evaluate_and_execute(&opt, &proposal, &space, tick, "svc", "a");
        let strict_allows = matches!(d_strict, GuardDecision::Allow(_));
        let opt_allows = matches!(d_opt, GuardDecision::Allow(_));
        t.trials += 1;
        if incident && strict_allows {
            t.incidents_allowed_baseline += 1;
        }
        if incident && !opt_allows {
            t.avoided_by_optionality += 1;
        }
        if incident && opt_allows {
            t.missed_by_optionality += 1;
        }
        if incident && !strict_allows {
            t.avoided_by_baseline += 1;
        }
        if !incident && !opt_allows {
            t.false_rejections_optionality += 1;
        }
        if !incident && !strict_allows {
            t.false_rejections_baseline += 1;
        }
    }
    t
}

fn space_size() -> usize {
    16
}

fn table() -> Vec<(&'static str, u64, Tally)> {
    let mut rows = Vec::new();
    for (name, family) in SCENARIOS {
        for seed in SEEDS {
            rows.push((name, seed, run(family, seed)));
        }
    }
    rows
}

#[test]
fn optionality_guard_rejects_irreversible_step_with_typed_reason_and_allows_reversible() {
    let graph = service_recovery_graph();
    let g = guard_for(&graph, None);
    g.observe_state(svc::DEGRADED);
    // Irreversible, sin vuelta atras.
    for a in [svc::PURGE, svc::DECOMMISSION] {
        match g.evaluate(a) {
            Err(OptionalityRejection::InsufficientRecoveryPaths {
                found, required, ..
            }) => {
                assert_eq!(found, 0);
                assert_eq!(required, DECLARED_N);
            }
            other => panic!("expected typed rejection, got {other:?}"),
        }
    }
    // El mismo estado con vuelta atras disponible: restart lleva a sano.
    let ok = g.evaluate(1).expect("reversible step authorized");
    assert_eq!(ok.successor, svc::HEALTHY);
    assert!(ok.recovery_paths >= DECLARED_N);
    assert!(ok.control_bits.is_finite());

    // Vía ActionGuard: motivo tipado conservado y codigo estatico estable.
    let space = ActionSpace::discrete(16).unwrap();
    let guard = ActionGuard::with_precondition(gogamen_core::SafeFallbackPolicy::RejectOnly, &g);
    let prop = ActionProposal::authorized(
        gogamen_core::Action::Discrete(gogamen_core::DiscreteAction::new(svc::PURGE, 16).unwrap()),
        Tick::new(0),
    );
    let ctx =
        gogamen_core::GuardContext::now(&space, Tick::new(0), gogamen_core::Terminal::Ongoing);
    let d = guard.evaluate(&prop, &ctx);
    assert!(matches!(
        d.rejection_reason(),
        Some(gogamen_core::GuardRejectionReason::PreconditionFailed {
            reason: "optionality:insufficient_recovery_paths"
        })
    ));
    assert!(matches!(
        g.last_rejection(),
        Some(OptionalityRejection::InsufficientRecoveryPaths { found: 0, .. })
    ));
}

#[test]
fn optionality_guard_threshold_is_declared_and_learning_cannot_relax_it() {
    let graph = service_recovery_graph();
    let policy = OptionalityPolicy::new(DECLARED_N, HORIZON).unwrap();
    assert_eq!(policy.declared_min_recovery_paths(), DECLARED_N);
    // Valores aprendidos menores o iguales no relajan.
    for learned in [0u64, 1, DECLARED_N] {
        assert_eq!(policy.effective_min_paths(Some(learned)), DECLARED_N);
        let g = guard_for(&graph, Some(learned));
        assert_eq!(g.effective_min_paths(), DECLARED_N);
        // Comportamiento: DEEP_2 (1 camino) sigue rechazado aunque se "aprenda" 0.
        g.observe_state(svc::DEEP_1);
        assert!(g.evaluate(1).is_err(), "learned {learned} must not relax");
    }
    // Aprender un valor mayor solo endurece.
    assert_eq!(
        policy.effective_min_paths(Some(DECLARED_N + 5)),
        DECLARED_N + 5
    );
    // La política declarada no cambia tras aplicar hints (campos privados, sin setters).
    let g = guard_for(&graph, Some(0));
    assert_eq!(g.policy().declared_min_recovery_paths(), DECLARED_N);
    // Parámetros inválidos se rechazan.
    assert_eq!(
        OptionalityPolicy::new(0, HORIZON),
        Err(PolicyError::ZeroParameter)
    );
    assert_eq!(
        OptionalityPolicy::new(1, 0),
        Err(PolicyError::ZeroParameter)
    );
}

#[test]
fn optionality_guard_control_floor_is_optional_and_reported() {
    let graph = service_recovery_graph();
    let strict_floor = OptionalityPolicy::new(1, HORIZON)
        .unwrap()
        .with_min_control_bits(100.0)
        .unwrap();
    let g = OptionalityGuard::new(graph.clone(), strict_floor, svc::DEGRADED);
    assert!(matches!(
        g.evaluate(1),
        Err(OptionalityRejection::ControlBelowFloor { .. })
    ));
    assert_eq!(
        OptionalityPolicy::new(1, 1)
            .unwrap()
            .with_min_control_bits(f64::NAN),
        Err(PolicyError::NonFiniteFloor)
    );
    // Fail-closed fuera del grafo.
    let g2 = guard_for(&graph, None);
    assert_eq!(
        g2.evaluate(99),
        Err(OptionalityRejection::OutsideDeclaredGraph)
    );
}

#[test]
fn optionality_guard_adversarial_table_vs_current_guard_is_deterministic() {
    let a = table();
    let b = table();
    assert_eq!(a, b, "table must be deterministic");

    println!(
        "scenario                  seed  trials baseline_incidents_allowed avoided_opt missed_opt avoided_base false_rej_opt false_rej_base"
    );
    let mut per_scn: Vec<(&str, Tally)> = Vec::new();
    for (name, seed, t) in &a {
        println!(
            "{name:<25} {seed:>4} {:>7} {:>26} {:>11} {:>10} {:>12} {:>13} {:>14}",
            t.trials,
            t.incidents_allowed_baseline,
            t.avoided_by_optionality,
            t.missed_by_optionality,
            t.avoided_by_baseline,
            t.false_rejections_optionality,
            t.false_rejections_baseline
        );
        match per_scn.iter_mut().find(|(n, _)| n == name) {
            Some((_, acc)) => {
                acc.trials += t.trials;
                acc.incidents_allowed_baseline += t.incidents_allowed_baseline;
                acc.avoided_by_optionality += t.avoided_by_optionality;
                acc.missed_by_optionality += t.missed_by_optionality;
                acc.avoided_by_baseline += t.avoided_by_baseline;
                acc.false_rejections_optionality += t.false_rejections_optionality;
                acc.false_rejections_baseline += t.false_rejections_baseline;
            }
            None => per_scn.push((name, *t)),
        }
    }
    println!("TOTALS per scenario (3 seeds)");
    for (name, t) in &per_scn {
        println!(
            "{name:<25} trials={} incidents={} avoided_opt={} missed_opt={} avoided_base={} false_rej_opt={} false_rej_base={}",
            t.trials,
            t.incidents_allowed_baseline,
            t.avoided_by_optionality,
            t.missed_by_optionality,
            t.avoided_by_baseline,
            t.false_rejections_optionality,
            t.false_rejections_baseline
        );
    }

    for (_, _, t) in &a {
        // P1 y P2.
        assert_eq!(t.missed_by_optionality, 0);
        assert_eq!(t.avoided_by_baseline, 0);
        assert_eq!(t.false_rejections_baseline, 0);
        assert_eq!(t.avoided_by_optionality, t.incidents_allowed_baseline);
    }
    // P3 (honest negative expected): falsos rechazos en fallos profundos.
    let deep = per_scn
        .iter()
        .find(|(n, _)| *n == "deep_fault_only")
        .unwrap()
        .1;
    assert!(deep.false_rejections_optionality > 0);
    let adv = per_scn
        .iter()
        .find(|(n, _)| *n == "adversarial_irreversible")
        .unwrap()
        .1;
    assert!(adv.avoided_by_optionality > 0);
}
