//! Eventos sintéticos de servicio y reglas deterministas de umbral.

use std::fmt;

use crate::spec::AutomationError;

/// Evento sintético emitido por un servicio local supervisado (sin red ni SO real).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SyntheticServiceEvent {
    /// Degradación de salud por fallos consecutivos de comprobación.
    HealthDegraded {
        /// Número de comprobaciones de salud fallidas consecutivamente.
        failed_checks: u32,
    },
    /// Pico de latencia p99 observado en milisegundos.
    LatencySpike {
        /// Latencia p99 medida en milisegundos.
        p99_ms: u64,
    },
    /// Acumulación de elementos en la cola interna del servicio.
    QueueBacklog {
        /// Profundidad actual de la cola.
        depth: u64,
    },
    /// Sonda de verificación superada tras una remediación.
    ProbePassed {
        /// Número de sondas exitosas consecutivas.
        consecutive_ok: u32,
    },
    /// Confirmación sostenida de recuperación del servicio.
    RecoveryConfirmed {
        /// Ticks saludables consecutivos observados.
        healthy_ticks: u32,
    },
}

impl fmt::Display for SyntheticServiceEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HealthDegraded { failed_checks } => {
                write!(f, "HealthDegraded(failed_checks={failed_checks})")
            }
            Self::LatencySpike { p99_ms } => write!(f, "LatencySpike(p99_ms={p99_ms})"),
            Self::QueueBacklog { depth } => write!(f, "QueueBacklog(depth={depth})"),
            Self::ProbePassed { consecutive_ok } => {
                write!(f, "ProbePassed(consecutive_ok={consecutive_ok})")
            }
            Self::RecoveryConfirmed { healthy_ticks } => {
                write!(f, "RecoveryConfirmed(healthy_ticks={healthy_ticks})")
            }
        }
    }
}

/// Resultado determinista de evaluar un [`SyntheticServiceEvent`] frente a [`ServiceThresholds`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventClassification {
    /// El evento no alcanza ningún umbral de alerta, ambigüedad ni resolución.
    BelowThreshold,
    /// El evento supera un umbral crítico configurado y dispara alerta determinista.
    Breach {
        /// Código corto del tipo de incumplimiento.
        kind: &'static str,
        /// Detalle textual determinista del valor observado.
        detail: String,
    },
    /// El evento cae en una banda gris configurable y requiere clasificación cognitiva opcional.
    Ambiguous {
        /// Código corto del síntoma ambiguo.
        kind: &'static str,
        /// Detalle textual determinista del valor observado.
        detail: String,
    },
    /// La sonda tras remediación alcanza el umbral requerido para resolver.
    ProbeReady {
        /// Detalle textual determinista.
        detail: String,
    },
    /// La confirmación de recuperación alcanza el umbral requerido para resolver.
    RecoveryReady {
        /// Detalle textual determinista.
        detail: String,
    },
}

/// Reglas exactas de umbral configurables para el supervisor de servicio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServiceThresholds {
    /// Fallos consecutivos de salud mínimos para disparar alerta (`>=`).
    pub max_failed_checks: u32,
    /// Latencia p99 en milisegundos a partir de la cual se dispara alerta (`>=`).
    pub latency_threshold_ms: u64,
    /// Banda inferior opcional de ambigüedad (`amb <= p99_ms < latency_threshold_ms`).
    pub latency_ambiguity_ms: Option<u64>,
    /// Profundidad mínima de cola para disparar alerta (`>=`).
    pub queue_backlog_threshold: u64,
    /// Sondas consecutivas exitosas requeridas para confirmar recuperación (`>=`).
    pub required_probe_passes: u32,
    /// Ticks saludables consecutivos requeridos para confirmar resolución (`>=`).
    pub required_recovery_ticks: u32,
}

impl Default for ServiceThresholds {
    fn default() -> Self {
        Self {
            max_failed_checks: 3,
            latency_threshold_ms: 250,
            latency_ambiguity_ms: Some(150),
            queue_backlog_threshold: 100,
            required_probe_passes: 2,
            required_recovery_ticks: 3,
        }
    }
}

impl ServiceThresholds {
    /// Valida que todos los umbrales sean positivos y coherentes.
    pub fn validate(&self) -> Result<(), AutomationError> {
        if self.max_failed_checks == 0 {
            return Err(AutomationError::InvalidConfig(
                "max_failed_checks must be >= 1".to_owned(),
            ));
        }
        if self.latency_threshold_ms == 0 {
            return Err(AutomationError::InvalidConfig(
                "latency_threshold_ms must be >= 1".to_owned(),
            ));
        }
        if let Some(amb) = self.latency_ambiguity_ms
            && (amb == 0 || amb >= self.latency_threshold_ms)
        {
            return Err(AutomationError::InvalidConfig(
                "latency_ambiguity_ms must satisfy 1 <= amb < latency_threshold_ms".to_owned(),
            ));
        }
        if self.queue_backlog_threshold == 0 {
            return Err(AutomationError::InvalidConfig(
                "queue_backlog_threshold must be >= 1".to_owned(),
            ));
        }
        if self.required_probe_passes == 0 {
            return Err(AutomationError::InvalidConfig(
                "required_probe_passes must be >= 1".to_owned(),
            ));
        }
        if self.required_recovery_ticks == 0 {
            return Err(AutomationError::InvalidConfig(
                "required_recovery_ticks must be >= 1".to_owned(),
            ));
        }
        Ok(())
    }

    /// Clasifica de forma pura y determinista un [`SyntheticServiceEvent`].
    pub fn classify(&self, event: &SyntheticServiceEvent) -> EventClassification {
        match event {
            SyntheticServiceEvent::HealthDegraded { failed_checks } => {
                if *failed_checks >= self.max_failed_checks {
                    EventClassification::Breach {
                        kind: "health_degraded",
                        detail: format!(
                            "failed_checks={failed_checks}>={}",
                            self.max_failed_checks
                        ),
                    }
                } else {
                    EventClassification::BelowThreshold
                }
            }
            SyntheticServiceEvent::LatencySpike { p99_ms } => {
                if *p99_ms >= self.latency_threshold_ms {
                    EventClassification::Breach {
                        kind: "latency_spike",
                        detail: format!("p99_ms={p99_ms}>={}", self.latency_threshold_ms),
                    }
                } else if let Some(amb) = self.latency_ambiguity_ms
                    && *p99_ms >= amb
                {
                    EventClassification::Ambiguous {
                        kind: "latency_ambiguous",
                        detail: format!(
                            "p99_ms={p99_ms} in [{amb}, {})",
                            self.latency_threshold_ms
                        ),
                    }
                } else {
                    EventClassification::BelowThreshold
                }
            }
            SyntheticServiceEvent::QueueBacklog { depth } => {
                if *depth >= self.queue_backlog_threshold {
                    EventClassification::Breach {
                        kind: "queue_backlog",
                        detail: format!("depth={depth}>={}", self.queue_backlog_threshold),
                    }
                } else {
                    EventClassification::BelowThreshold
                }
            }
            SyntheticServiceEvent::ProbePassed { consecutive_ok } => {
                if *consecutive_ok >= self.required_probe_passes {
                    EventClassification::ProbeReady {
                        detail: format!(
                            "consecutive_ok={consecutive_ok}>={}",
                            self.required_probe_passes
                        ),
                    }
                } else {
                    EventClassification::BelowThreshold
                }
            }
            SyntheticServiceEvent::RecoveryConfirmed { healthy_ticks } => {
                if *healthy_ticks >= self.required_recovery_ticks {
                    EventClassification::RecoveryReady {
                        detail: format!(
                            "healthy_ticks={healthy_ticks}>={}",
                            self.required_recovery_ticks
                        ),
                    }
                } else {
                    EventClassification::BelowThreshold
                }
            }
        }
    }
}
