//! Contrato de capacidad core `World` sin presuponer agentes ni sistemas externos.
//!
//! La capa de `System` / bucle de agentes se construirá encima de estos contratos.

use crate::action::{Action, ActionSpace};
use crate::domain::{Observation, State, Transition};
use crate::error::DomainError;
use crate::units::StepDt;

/// Capacidad central de un mundo simulado o entorno determinista/estocástico.
///
/// # Mutabilidad
/// - [`World::reset`] y [`World::step`] toman `&mut self` porque mutan el estado interno
///   del mundo al reiniciar o avanzar un episodio.
/// - [`World::action_space`] y [`World::observe`] toman `&self` y no deben mutar el estado
///   del mundo.
/// - Todos los valores devueltos ([`State`], [`Observation`], [`Transition`]) son snapshots
///   inmutables independientes de mutaciones posteriores sobre el mundo.
///
/// # Unidades
/// - El paso temporal `dt` en [`World::step`] es un [`StepDt`] expresado en **segundos**
///   finitos y estrictamente positivos (`> 0.0`).
///
/// # Semántica de terminal
/// - Cuando un paso devuelve [`crate::Terminal::Terminated`], el episodio actual finaliza.
/// - Las implementaciones deben rechazar llamadas posteriores a [`World::step`] devolviendo
///   [`DomainError::AlreadyTerminal`] hasta que se invoque [`World::reset`].
pub trait World {
    /// Devuelve el espacio de acciones válido para este mundo.
    fn action_space(&self) -> &ActionSpace;

    /// Reinicia el mundo a un nuevo episodio y devuelve el estado inicial.
    fn reset(&mut self) -> Result<State, DomainError>;

    /// Proyecta un [`State`] inmutable en una [`Observation`] sin mutar el mundo.
    fn observe(&self, state: &State) -> Result<Observation, DomainError>;

    /// Aplica `action` durante un intervalo `dt` (en segundos) y devuelve la [`Transition`].
    ///
    /// Debe validar que la acción pertenezca a [`World::action_space`] y que el mundo no
    /// haya alcanzado previamente un estado terminal ([`DomainError::AlreadyTerminal`]).
    fn step(&mut self, action: &Action, dt: StepDt) -> Result<Transition, DomainError>;
}
