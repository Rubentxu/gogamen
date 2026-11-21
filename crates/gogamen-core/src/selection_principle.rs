//! Principio de selección que construye la recompensa FMC (`std`-only, GOG-198).
//!
//! Un [`SelectionPrinciple`] declara **qué información puede entrar en la puntuación** de un
//! planificador FMC. El tipo vive en `gogamen-core` para que el planificador, los laboratorios y la
//! traza científica compartan una única definición auditable.
//!
//! - [`SelectionPrinciple::Directed`]: recompensa dirigida a un objetivo de dominio (checkpoints,
//!   vueltas, progreso, velocidad, colisión, energía). Es el comportamiento histórico del kart
//!   autónomo y su aritmética no cambia.
//! - [`SelectionPrinciple::Pure`]: recompensa constante y neutral. La puntuación no lee estado,
//!   pista, pesos ni transición; cualquier diferencia de conducta procede de la viabilidad de los
//!   walkers del planificador, nunca de una variable de objetivo.
//! - [`SelectionPrinciple::Situated`]: reservado para un intermedio dependiente del contexto
//!   situado. **No está implementado** en el controlador de kart de GOG-198: declararlo hace
//!   fallar la construcción de forma explícita ([`SelectionPrinciple::validate`]) en lugar de
//!   simular una conducta.
//!
//! Este módulo solo define el tipo y su validación; la construcción concreta de la recompensa
//! (por ejemplo `KartRewardFunction` en `gogamen-lab-racing`) es responsabilidad de cada dominio.

use crate::error::DomainError;

/// Principio que determina cómo se construye la recompensa de una decisión FMC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SelectionPrinciple {
    /// Recompensa constante y neutral: la puntuación no depende del estado ni de los pesos.
    Pure,
    /// Intermedio situado (dependiente del contexto). Declarado pero no implementado todavía.
    Situated,
    /// Recompensa dirigida a un objetivo de dominio explícito (conducta histórica del kart).
    #[default]
    Directed,
}

impl SelectionPrinciple {
    /// Etiqueta estable para trazas, recibos y documentación.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Pure => "pure",
            Self::Situated => "situated",
            Self::Directed => "directed",
        }
    }

    /// Indica si el principio tiene una construcción de recompensa implementada.
    pub const fn is_implemented(self) -> bool {
        !matches!(self, Self::Situated)
    }

    /// Valida que el principio pueda usarse para construir una recompensa.
    ///
    /// Falla de forma explícita (fail-closed) para [`SelectionPrinciple::Situated`], que está
    /// declarado pero no implementado; no se simula su conducta.
    pub const fn validate(self) -> Result<(), DomainError> {
        match self {
            Self::Pure | Self::Directed => Ok(()),
            Self::Situated => Err(DomainError::InvalidTransition {
                reason: "SelectionPrinciple::Situated is declared but not implemented",
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_is_default_and_situated_fails_closed() {
        assert_eq!(SelectionPrinciple::default(), SelectionPrinciple::Directed);
        assert!(SelectionPrinciple::Directed.validate().is_ok());
        assert!(SelectionPrinciple::Pure.validate().is_ok());
        assert!(SelectionPrinciple::Situated.validate().is_err());
        assert!(!SelectionPrinciple::Situated.is_implemented());
        assert_eq!(SelectionPrinciple::Pure.name(), "pure");
    }
}
