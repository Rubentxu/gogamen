//! Contratos de modelo generativo simulado (`std`-only), aislados del entorno vivo.
//!
//! Un [`GenerativeModel`] representa una capacidad **pura** de simulación: dado un
//! [`State`] de origen, una [`Action`], un paso temporal [`StepDt`] y una fuente de
//! aleatoriedad [`RngSource`], produce una [`Transition`] sin mutar el estado de origen
//! ni ejecutar efectos sobre un [`crate::Actuator`] o [`crate::LiveEnvironment`].

use crate::action::{Action, ActionSpace};
use crate::domain::{State, Transition};
use crate::error::ModelError;
use crate::units::StepDt;

/// Fuente ligera de números pseudoaleatorios para simulación (`std`-only).
///
/// Permite inyectar aleatoriedad determinista o controlada en [`GenerativeModel::step`]
/// sin depender de crates externos.
pub trait RngSource {
    /// Genera el siguiente entero pseudoaleatorio de 64 bits.
    fn next_u64(&mut self) -> u64;

    /// Genera un escalar pseudoaleatorio uniforme en el intervalo semiabierto `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    /// Generates a uniform integer in `0..span` using rejection sampling (no modulo bias).
    ///
    /// Draws are rejected when they fall in the first `2^64 mod span` values, so the accepted
    /// range is an exact multiple of `span`. `span <= 1` returns `0` without consuming the RNG
    /// (`span == 0` is a caller bug and trips a `debug_assert!`).
    ///
    /// This consumes a different RNG sequence than `next_u64() % span`; callers that must keep
    /// legacy seeded oracles stay on the modulo path (see `FmcScoringConfig::unbiased_sampling`).
    fn next_usize_unbiased(&mut self, span: usize) -> usize {
        debug_assert!(span > 0, "next_usize_unbiased requires span > 0");
        if span <= 1 {
            return 0;
        }
        let span = span as u64;
        let threshold = span.wrapping_neg() % span;
        loop {
            let draw = self.next_u64();
            if draw >= threshold {
                return (draw % span) as usize;
            }
        }
    }
}

/// Generador pseudoaleatorio determinista basado en SplitMix64 (`std`-only).
///
/// Dada una misma semilla inicial, produce exactamente la misma secuencia de valores,
/// facilitando simulaciones y rollouts reproducibles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Crea un nuevo generador determinista a partir de una semilla de 64 bits.
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Devuelve el estado interno actual del generador.
    pub const fn state(&self) -> u64 {
        self.state
    }
}

impl RngSource for DeterministicRng {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Capacidad pura de simulación de transiciones para planificación y rollouts.
///
/// # Separación respecto al entorno vivo
/// - [`GenerativeModel::step`] toma `&self` y `&State` inmutable: no muta el modelo
///   ni el estado de origen, devolviendo una nueva [`Transition`].
/// - No expone ningún [`crate::Actuator`] ni [`crate::LiveEnvironment`]; los algoritmos
///   de planificación que operan sobre `GenerativeModel` no pueden actuar sobre el entorno real.
pub trait GenerativeModel {
    /// Devuelve el espacio de acciones válido para este modelo generativo.
    fn action_space(&self) -> &ActionSpace;

    /// Simula un paso desde `state` aplicando `action` durante `dt` segundos.
    ///
    /// No debe mutar `state` ni `self`. Toda aleatoriedad debe provenir de `rng`.
    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError>;
}

/// Contexto de planificación aislado que solo concede acceso al modelo simulado.
///
/// Estructuralmente impide que un planificador reciba o extraiga un [`crate::Actuator`]
/// o [`crate::LiveEnvironment`] para actuar sobre el mundo real.
///
/// # Ejemplo de aislamiento en tiempo de compilación
///
/// ```compile_fail
/// use gogamen_core::{
///     Action, ActionSpace, ApplyOutcome, DeterministicRng, EnvironmentError,
///     EnvironmentObserver, Actuator, LiveEnvironment, Observation, PlanningContext,
///     SimDuration, Terminal, Tick,
/// };
///
/// struct DummyLiveEnv {
///     space: ActionSpace,
/// }
///
/// impl EnvironmentObserver for DummyLiveEnv {
///     fn observe(&self) -> Result<Observation, EnvironmentError> {
///         Ok(Observation::new(Tick::ZERO, SimDuration::ZERO, vec![0.0]).unwrap())
///     }
/// }
///
/// impl Actuator for DummyLiveEnv {
///     fn action_space(&self) -> &ActionSpace {
///         &self.space
///     }
///
///     fn apply(&mut self, _action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
///         let obs = self.observe()?;
///         Ok(ApplyOutcome::new(obs, Terminal::Ongoing))
///     }
/// }
///
/// impl LiveEnvironment for DummyLiveEnv {
///     fn reset(&mut self) -> Result<Observation, EnvironmentError> {
///         self.observe()
///     }
/// }
///
/// let env = DummyLiveEnv {
///     space: ActionSpace::discrete(2).unwrap(),
/// };
/// let mut rng = DeterministicRng::new(42);
/// // Error de compilación: `DummyLiveEnv` implementa `LiveEnvironment`/`Actuator`,
/// // pero NO implementa `GenerativeModel`.
/// let _ctx = PlanningContext::new(&env, &mut rng);
/// ```
pub struct PlanningContext<'a, M: GenerativeModel + ?Sized> {
    model: &'a M,
    rng: &'a mut dyn RngSource,
}

impl<'a, M: GenerativeModel + ?Sized> PlanningContext<'a, M> {
    /// Crea un contexto de planificación con acceso exclusivo al modelo generativo y su RNG.
    pub fn new(model: &'a M, rng: &'a mut dyn RngSource) -> Self {
        Self { model, rng }
    }

    /// Devuelve una referencia de solo lectura al modelo generativo simulado.
    pub const fn model(&self) -> &M {
        self.model
    }

    /// Simula una transición en el modelo sin mutar `state` ni actuar en entorno vivo.
    pub fn simulate_step(
        &mut self,
        state: &State,
        action: &Action,
        dt: StepDt,
    ) -> Result<Transition, ModelError> {
        self.model.step(state, action, dt, self.rng)
    }
}

#[cfg(test)]
mod unbiased_tests {
    use super::*;

    struct Scripted(Vec<u64>, usize);

    impl RngSource for Scripted {
        fn next_u64(&mut self) -> u64 {
            let v = self.0[self.1];
            self.1 += 1;
            v
        }
    }

    #[test]
    fn next_usize_unbiased_rejects_biased_zone_and_skips_trivial_spans() {
        // span 3: 2^64 mod 3 == 1, so draw 0 is rejected and draw 4 maps to 1.
        let mut rng = Scripted(vec![0, 4], 0);
        assert_eq!(rng.next_usize_unbiased(3), 1);
        assert_eq!(rng.1, 2);

        let mut rng = Scripted(vec![], 0);
        assert_eq!(rng.next_usize_unbiased(1), 0);
        assert_eq!(rng.1, 0, "span 1 must not consume the RNG");
    }

    #[test]
    fn next_usize_unbiased_stays_in_range_and_is_deterministic() {
        let mut a = DeterministicRng::new(9);
        let mut b = DeterministicRng::new(9);
        for span in [2usize, 3, 7, 10, 1000] {
            for _ in 0..200 {
                let x = a.next_usize_unbiased(span);
                assert!(x < span);
                assert_eq!(x, b.next_usize_unbiased(span));
            }
        }
    }
}
