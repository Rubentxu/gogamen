//! Contrato puro de sistema de transición tipada (`std`-only).
//!
//! Este módulo define las abstracciones base para sistemas de transición deterministas
//! y componibles directamente **sin agentes, cognición, IA ni simulación**:
//! - [`TypedSystem`]: contrato puro de transición `(&self, &State, Input, &SystemContext) -> Result<SystemStep<...>, Error>`.
//! - [`SystemContext`]: contexto temporal inmutable del paso ([`Tick`], [`SimDuration`]).
//! - [`SystemStep`]: resultado por valor con el nuevo estado, la salida tipada y los efectos propuestos.
//! - [`ProposedEffect`]: representación pura de efectos propuestos como datos, sin ejecutar IO.
//! - [`SystemError`]: error tipado de transición o validación de entrada/estado.
//! - [`ComposedSystem`]: composición directa en tubería (`A -> B`) donde la salida de `A`
//!   alimenta la entrada de `B`, acumulando efectos propuestos y propagando errores sin
//!   mutación parcial de estado.
//!
//! # Semántica de ownership y pureza
//! - **Ownership del estado:** [`TypedSystem::step`] toma el estado previo por referencia
//!   inmutable (`&Self::State`) y devuelve el estado sucesor **por valor** dentro de
//!   [`SystemStep::next_state`]. El llamador conserva en todo momento el ownership del estado
//!   previo; si la transición devuelve `Err(_)`, el estado previo permanece intacto por
//!   construcción del sistema de tipos.
//! - **Pureza de efectos:** los efectos devueltos en [`SystemStep::effects`] (como
//!   [`ProposedEffect`]) son valores puros que describen intenciones de efecto (eventos,
//!   registros, etc.). Nunca ejecutan IO ni contienen handles mutables del entorno.
//! - **Aislamiento:** este módulo no depende de [`crate::World`], [`crate::GenerativeModel`],
//!   [`crate::Planner`], agentes ni motores externos.

use std::error::Error;
use std::fmt;

use crate::error::DomainError;
use crate::units::{SimDuration, Tick};

/// Contexto temporal inmutable proporcionado a cada paso de un [`TypedSystem`].
///
/// No contiene handles de IO, generadores aleatorios ni estado de agentes.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct SystemContext {
    tick: Tick,
    elapsed: SimDuration,
}

impl SystemContext {
    /// Crea un contexto de sistema para un `tick` y duración acumulada `elapsed`.
    pub const fn new(tick: Tick, elapsed: SimDuration) -> Self {
        Self { tick, elapsed }
    }

    /// Crea un contexto de sistema en `tick` con duración acumulada cero.
    pub const fn at_tick(tick: Tick) -> Self {
        Self {
            tick,
            elapsed: SimDuration::ZERO,
        }
    }

    /// Tick lógico actual de la transición.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Tiempo acumulado de ejecución asociado al paso.
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }
}

/// Severidad pura para propuestas de registro en [`ProposedEffect::Log`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EffectSeverity {
    /// Traza informativa.
    Info,
    /// Advertencia recuperable.
    Warn,
    /// Condición de error reportada como dato.
    Error,
}

/// Descripción pura de un efecto propuesto por un [`TypedSystem`].
///
/// `ProposedEffect` es un valor de datos inmutable que describe qué efecto debería
/// interpretarse externamente tras la transición. Construir o devolver un `ProposedEffect`
/// **no ejecuta IO** ni produce efectos secundarios observables.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProposedEffect {
    /// Propuesta de emisión de evento con tópico y carga útil textual.
    EmitEvent {
        /// Canal o nombre lógico del evento.
        topic: String,
        /// Carga útil serializable del evento.
        payload: String,
    },
    /// Propuesta de registro estructurado sin escribir a stdout/stderr/archivos.
    Log {
        /// Severidad propuesta.
        level: EffectSeverity,
        /// Mensaje propuesto.
        message: String,
    },
    /// Propuesta de efecto genérico tipado por nombre de operación y detalle.
    Custom {
        /// Identificador estático del tipo de efecto.
        kind: &'static str,
        /// Detalle textual del efecto propuesto.
        detail: String,
    },
}

impl ProposedEffect {
    /// Construye una propuesta pura de emisión de evento.
    pub fn emit_event(topic: impl Into<String>, payload: impl Into<String>) -> Self {
        Self::EmitEvent {
            topic: topic.into(),
            payload: payload.into(),
        }
    }

    /// Construye una propuesta pura de registro.
    pub fn log(level: EffectSeverity, message: impl Into<String>) -> Self {
        Self::Log {
            level,
            message: message.into(),
        }
    }

    /// Construye una propuesta pura personalizada.
    pub fn custom(kind: &'static str, detail: impl Into<String>) -> Self {
        Self::Custom {
            kind,
            detail: detail.into(),
        }
    }
}

/// Error tipado estándar para sistemas de transición pura.
#[derive(Debug, Clone, PartialEq)]
pub enum SystemError {
    /// La entrada recibida por el sistema es inválida para la transición.
    InvalidInput {
        /// Motivo legible del rechazo.
        reason: &'static str,
    },
    /// El estado recibido viola una invariante requerida por el sistema.
    InvalidState {
        /// Motivo legible de la inconsistencia.
        reason: &'static str,
    },
    /// La transición fue rechazada por una regla de negocio o desbordamiento.
    TransitionRejected {
        /// Motivo legible del rechazo.
        reason: &'static str,
    },
    /// Error de dominio subyacente propagado durante la transición.
    Domain(DomainError),
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { reason } => write!(f, "system invalid input: {reason}"),
            Self::InvalidState { reason } => write!(f, "system invalid state: {reason}"),
            Self::TransitionRejected { reason } => {
                write!(f, "system transition rejected: {reason}")
            }
            Self::Domain(err) => write!(f, "system domain error: {err}"),
        }
    }
}

impl Error for SystemError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Domain(err) => Some(err),
            Self::InvalidInput { .. }
            | Self::InvalidState { .. }
            | Self::TransitionRejected { .. } => None,
        }
    }
}

impl From<DomainError> for SystemError {
    fn from(err: DomainError) -> Self {
        Self::Domain(err)
    }
}

/// Resultado exitoso de un paso de transición en un [`TypedSystem`].
///
/// Posee el nuevo estado (`next_state`) por valor, la salida tipada (`output`)
/// y la secuencia ordenada de efectos propuestos (`effects`) pendientes de
/// interpretación externa.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemStep<S, O, E = ProposedEffect> {
    /// Nuevo estado producido por la transición (ownership transferido al llamador).
    pub next_state: S,
    /// Salida tipada producida para consumidores o sistemas posteriores.
    pub output: O,
    /// Efectos propuestos como datos puros durante el paso.
    pub effects: Vec<E>,
}

impl<S, O, E> SystemStep<S, O, E> {
    /// Construye un [`SystemStep`] sin efectos propuestos.
    pub fn new(next_state: S, output: O) -> Self {
        Self {
            next_state,
            output,
            effects: Vec::new(),
        }
    }

    /// Construye un [`SystemStep`] con un único efecto propuesto.
    pub fn with_effect(next_state: S, output: O, effect: E) -> Self {
        Self {
            next_state,
            output,
            effects: vec![effect],
        }
    }

    /// Construye un [`SystemStep`] con una lista explícita de efectos propuestos.
    pub fn with_effects(next_state: S, output: O, effects: Vec<E>) -> Self {
        Self {
            next_state,
            output,
            effects,
        }
    }

    /// Referencia inmutable al nuevo estado producido.
    pub fn next_state(&self) -> &S {
        &self.next_state
    }

    /// Referencia inmutable a la salida producida.
    pub fn output(&self) -> &O {
        &self.output
    }

    /// Slice de efectos propuestos durante este paso.
    pub fn effects(&self) -> &[E] {
        &self.effects
    }

    /// Descompone el resultado en `(next_state, output, effects)` por valor.
    pub fn into_parts(self) -> (S, O, Vec<E>) {
        (self.next_state, self.output, self.effects)
    }
}

/// Resultado tipado de ejecutar un paso sobre un [`TypedSystem`].
pub type SystemResult<S> = Result<
    SystemStep<<S as TypedSystem>::State, <S as TypedSystem>::Output, <S as TypedSystem>::Effect>,
    <S as TypedSystem>::Error,
>;

/// Contrato puro de sistema de transición tipada sin agentes ni IA.
///
/// # Ownership y garantía de no-mutación
/// - `self` se toma por referencia compartida (`&self`), impidiendo mutación interna oculta.
/// - `state` se toma por referencia compartida (`&Self::State`), por lo que el llamador
///   conserva el estado previo intacto en todo momento.
/// - En caso de éxito (`Ok(step)`), `step.next_state` entrega un nuevo estado con ownership
///   transferido al llamador.
/// - En caso de error (`Err(err)`), no se produce ningún estado nuevo y el estado previo
///   del llamador permanece completamente inalterado.
pub trait TypedSystem {
    /// Tipo de estado propio del sistema.
    type State;
    /// Tipo de entrada aceptada en cada paso.
    type Input;
    /// Tipo de salida emitida en cada paso.
    type Output;
    /// Tipo de efecto propuesto (por ejemplo [`ProposedEffect`]).
    type Effect;
    /// Tipo de error tipado devuelto cuando la transición falla.
    type Error;

    /// Ejecuta un paso puro de transición a partir de `state`, `input` y `ctx`.
    fn step(
        &self,
        state: &Self::State,
        input: Self::Input,
        ctx: &SystemContext,
    ) -> SystemResult<Self>;

    /// Compone secuencialmente `self` con `next`, conectando `Self::Output` a `next::Input`.
    fn pipe<B>(self, next: B) -> ComposedSystem<Self, B>
    where
        Self: Sized,
        B: TypedSystem<Input = Self::Output>,
    {
        ComposedSystem::new(self, next)
    }
}

/// Estado compuesto de dos sistemas conectados mediante [`ComposedSystem`].
///
/// Mantiene el ownership explícito e independiente de ambos subestados (`first` y `second`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedState<SA, SB> {
    /// Estado del primer sistema en la tubería.
    pub first: SA,
    /// Estado del segundo sistema en la tubería.
    pub second: SB,
}

impl<SA, SB> ComposedState<SA, SB> {
    /// Crea un estado compuesto a partir de los estados iniciales de ambos sistemas.
    pub const fn new(first: SA, second: SB) -> Self {
        Self { first, second }
    }
}

/// Efecto propuesto por un [`ComposedSystem`], preservando el origen de cada etapa.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ComposedEffect<EA, EB> {
    /// Efecto propuesto por el primer sistema.
    First(EA),
    /// Efecto propuesto por el segundo sistema.
    Second(EB),
}

impl<E> ComposedEffect<E, E> {
    /// Extrae el efecto uniforme cuando ambas etapas comparten el mismo tipo de efecto.
    pub fn into_uniform(self) -> E {
        match self {
            Self::First(effect) | Self::Second(effect) => effect,
        }
    }
}

/// Error tipado propagado por un [`ComposedSystem`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposedError<EA, EB> {
    /// Fallo ocurrido en el primer sistema (el segundo no llega a ejecutarse).
    First(EA),
    /// Fallo ocurrido en el segundo sistema (el estado compuesto previo permanece intacto).
    Second(EB),
}

impl<EA: fmt::Display, EB: fmt::Display> fmt::Display for ComposedError<EA, EB> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::First(err) => write!(f, "first system error: {err}"),
            Self::Second(err) => write!(f, "second system error: {err}"),
        }
    }
}

impl<EA, EB> Error for ComposedError<EA, EB>
where
    EA: Error + 'static,
    EB: Error + 'static,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::First(err) => Some(err),
            Self::Second(err) => Some(err),
        }
    }
}

/// Composición directa en tubería de dos sistemas [`TypedSystem`] (`A -> B`).
///
/// En cada paso:
/// 1. Ejecuta `first` sobre `&state.first` y `input`. Si falla, retorna `Err(ComposedError::First(_))`
///    sin invocar `second` y sin mutar `state`.
/// 2. Alimenta la salida de `first` como entrada de `second` sobre `&state.second`.
///    Si falla, descarta el estado intermedio de `first` y retorna `Err(ComposedError::Second(_))`,
///    manteniendo intacto todo el [`ComposedState`] previo.
/// 3. Si ambos tienen éxito, acumula los efectos propuestos de `first` seguidos de los de
///    `second` sin ejecutar IO.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComposedSystem<A, B> {
    first: A,
    second: B,
}

impl<A, B> ComposedSystem<A, B> {
    /// Construye la composición secuencial `first -> second`.
    pub const fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Referencia al primer sistema.
    pub const fn first(&self) -> &A {
        &self.first
    }

    /// Referencia al segundo sistema.
    pub const fn second(&self) -> &B {
        &self.second
    }
}

impl<A, B> TypedSystem for ComposedSystem<A, B>
where
    A: TypedSystem,
    B: TypedSystem<Input = A::Output>,
{
    type State = ComposedState<A::State, B::State>;
    type Input = A::Input;
    type Output = B::Output;
    type Effect = ComposedEffect<A::Effect, B::Effect>;
    type Error = ComposedError<A::Error, B::Error>;

    fn step(
        &self,
        state: &Self::State,
        input: Self::Input,
        ctx: &SystemContext,
    ) -> SystemResult<Self> {
        let step_a = self
            .first
            .step(&state.first, input, ctx)
            .map_err(ComposedError::First)?;
        let step_b = self
            .second
            .step(&state.second, step_a.output, ctx)
            .map_err(ComposedError::Second)?;

        let mut effects = Vec::with_capacity(step_a.effects.len() + step_b.effects.len());
        effects.extend(step_a.effects.into_iter().map(ComposedEffect::First));
        effects.extend(step_b.effects.into_iter().map(ComposedEffect::Second));

        Ok(SystemStep::with_effects(
            ComposedState::new(step_a.next_state, step_b.next_state),
            step_b.output,
            effects,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Entrada del sistema contador puro.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CounterInput {
        Increment(u64),
        Reset,
    }

    /// Salida del sistema contador puro.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct CounterEvent {
        tick: Tick,
        previous: u64,
        current: u64,
    }

    /// Sistema puro de conteo con límite máximo configurable.
    struct CounterSystem {
        max_value: u64,
    }

    impl TypedSystem for CounterSystem {
        type State = u64;
        type Input = CounterInput;
        type Output = CounterEvent;
        type Effect = ProposedEffect;
        type Error = SystemError;

        fn step(
            &self,
            state: &Self::State,
            input: Self::Input,
            ctx: &SystemContext,
        ) -> Result<SystemStep<Self::State, Self::Output, Self::Effect>, Self::Error> {
            let next_value = match input {
                CounterInput::Increment(0) => {
                    return Err(SystemError::InvalidInput {
                        reason: "increment delta must be > 0",
                    });
                }
                CounterInput::Increment(delta) => state
                    .checked_add(delta)
                    .filter(|&sum| sum <= self.max_value)
                    .ok_or(SystemError::TransitionRejected {
                        reason: "counter exceeds max_value",
                    })?,
                CounterInput::Reset => 0,
            };

            let event = CounterEvent {
                tick: ctx.tick(),
                previous: *state,
                current: next_value,
            };
            let effect = ProposedEffect::emit_event(
                "counter.updated",
                format!("tick={} {}->{}", ctx.tick().as_u64(), state, next_value),
            );

            Ok(SystemStep::with_effect(next_value, event, effect))
        }
    }

    /// Estado del sistema transformador de eventos.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TransformState {
        processed_events: u64,
        last_alert_tick: Option<Tick>,
    }

    /// Evento transformado emitido por el segundo sistema.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct TransformedEvent {
        tick: Tick,
        delta: i64,
        threshold_crossed: bool,
    }

    /// Sistema puro que transforma eventos de conteo y detecta cruces de umbral.
    struct EventTransformSystem {
        alert_threshold: u64,
        reject_reset_at_zero: bool,
    }

    impl TypedSystem for EventTransformSystem {
        type State = TransformState;
        type Input = CounterEvent;
        type Output = TransformedEvent;
        type Effect = ProposedEffect;
        type Error = SystemError;

        fn step(
            &self,
            state: &Self::State,
            input: Self::Input,
            ctx: &SystemContext,
        ) -> Result<SystemStep<Self::State, Self::Output, Self::Effect>, Self::Error> {
            if self.reject_reset_at_zero && input.previous == 0 && input.current == 0 {
                return Err(SystemError::TransitionRejected {
                    reason: "redundant reset from zero",
                });
            }

            let delta = (input.current as i64) - (input.previous as i64);
            let threshold_crossed =
                input.previous < self.alert_threshold && input.current >= self.alert_threshold;

            let next_state = TransformState {
                processed_events: state.processed_events + 1,
                last_alert_tick: if threshold_crossed {
                    Some(ctx.tick())
                } else {
                    state.last_alert_tick
                },
            };

            let mut effects = vec![ProposedEffect::log(
                EffectSeverity::Info,
                format!("transformed delta={delta} at tick={}", input.tick.as_u64()),
            )];
            if threshold_crossed {
                effects.push(ProposedEffect::emit_event(
                    "counter.threshold_crossed",
                    format!("value={} threshold={}", input.current, self.alert_threshold),
                ));
            }

            Ok(SystemStep::with_effects(
                next_state,
                TransformedEvent {
                    tick: input.tick,
                    delta,
                    threshold_crossed,
                },
                effects,
            ))
        }
    }

    #[test]
    fn counter_and_event_transform_produce_expected_results_without_cognition_or_simulation() {
        let counter = CounterSystem { max_value: 10 };
        let transformer = EventTransformSystem {
            alert_threshold: 5,
            reject_reset_at_zero: false,
        };

        let ctx0 = SystemContext::at_tick(Tick::new(1));
        let counter_state = 2_u64;
        let step_c = counter
            .step(&counter_state, CounterInput::Increment(3), &ctx0)
            .expect("valid counter increment must succeed");

        // El estado previo no fue mutado y el sucesor se entrega por valor.
        assert_eq!(counter_state, 2);
        assert_eq!(step_c.next_state, 5);
        assert_eq!(
            step_c.output,
            CounterEvent {
                tick: Tick::new(1),
                previous: 2,
                current: 5,
            }
        );
        assert_eq!(
            step_c.effects,
            vec![ProposedEffect::emit_event("counter.updated", "tick=1 2->5")]
        );

        let transform_state = TransformState {
            processed_events: 0,
            last_alert_tick: None,
        };
        let step_t = transformer
            .step(&transform_state, step_c.output, &ctx0)
            .expect("valid event transform must succeed");

        assert_eq!(transform_state.processed_events, 0);
        assert_eq!(transform_state.last_alert_tick, None);
        assert_eq!(
            step_t.next_state,
            TransformState {
                processed_events: 1,
                last_alert_tick: Some(Tick::new(1)),
            }
        );
        assert_eq!(
            step_t.output,
            TransformedEvent {
                tick: Tick::new(1),
                delta: 3,
                threshold_crossed: true,
            }
        );
        assert_eq!(step_t.effects.len(), 2);
    }

    #[test]
    fn proposed_effects_are_pure_data_and_errors_never_mutate_previous_state() {
        let counter = CounterSystem { max_value: 5 };
        let state = 4_u64;
        let ctx = SystemContext::new(Tick::new(3), SimDuration::from_secs_f64(0.3).unwrap());

        let err_zero = counter
            .step(&state, CounterInput::Increment(0), &ctx)
            .expect_err("zero increment must fail with typed error");
        assert_eq!(
            err_zero,
            SystemError::InvalidInput {
                reason: "increment delta must be > 0",
            }
        );
        assert_eq!(state, 4);

        let err_overflow = counter
            .step(&state, CounterInput::Increment(2), &ctx)
            .expect_err("exceeding max_value must fail");
        assert_eq!(
            err_overflow,
            SystemError::TransitionRejected {
                reason: "counter exceeds max_value",
            }
        );
        assert_eq!(state, 4);
    }

    #[test]
    fn composed_system_fixture_pipes_output_accumulates_effects_and_preserves_state_on_error() {
        let pipeline = CounterSystem { max_value: 10 }.pipe(EventTransformSystem {
            alert_threshold: 5,
            reject_reset_at_zero: true,
        });

        let initial_state = ComposedState::new(
            2_u64,
            TransformState {
                processed_events: 0,
                last_alert_tick: None,
            },
        );
        let ctx1 = SystemContext::at_tick(Tick::new(1));

        let step1 = pipeline
            .step(&initial_state, CounterInput::Increment(4), &ctx1)
            .expect("composed step must succeed");

        assert_eq!(step1.next_state.first, 6);
        assert_eq!(
            step1.next_state.second,
            TransformState {
                processed_events: 1,
                last_alert_tick: Some(Tick::new(1)),
            }
        );
        assert_eq!(
            step1.output,
            TransformedEvent {
                tick: Tick::new(1),
                delta: 4,
                threshold_crossed: true,
            }
        );

        let uniform_effects: Vec<ProposedEffect> = step1
            .effects
            .into_iter()
            .map(ComposedEffect::into_uniform)
            .collect();
        assert_eq!(
            uniform_effects,
            vec![
                ProposedEffect::emit_event("counter.updated", "tick=1 2->6"),
                ProposedEffect::log(EffectSeverity::Info, "transformed delta=4 at tick=1"),
                ProposedEffect::emit_event("counter.threshold_crossed", "value=6 threshold=5"),
            ]
        );

        // Si la primera etapa falla, se propaga ComposedError::First sin mutar el estado previo.
        let ctx2 = SystemContext::at_tick(Tick::new(2));
        let err_first = pipeline
            .step(&initial_state, CounterInput::Increment(99), &ctx2)
            .expect_err("overflow in first system must fail");
        assert_eq!(
            err_first,
            ComposedError::First(SystemError::TransitionRejected {
                reason: "counter exceeds max_value",
            })
        );
        assert_eq!(initial_state.first, 2);
        assert_eq!(initial_state.second.processed_events, 0);

        // Si la segunda etapa falla tras tener éxito la primera, se propaga ComposedError::Second
        // y todo el estado compuesto previo permanece intacto (sin mutación parcial).
        let zero_state = ComposedState::new(
            0_u64,
            TransformState {
                processed_events: 7,
                last_alert_tick: None,
            },
        );
        let err_second = pipeline
            .step(&zero_state, CounterInput::Reset, &ctx2)
            .expect_err("redundant reset in second system must fail");
        assert_eq!(
            err_second,
            ComposedError::Second(SystemError::TransitionRejected {
                reason: "redundant reset from zero",
            })
        );
        assert_eq!(zero_state.first, 0);
        assert_eq!(zero_state.second.processed_events, 7);
    }
}
