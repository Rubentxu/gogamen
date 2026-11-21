//! Simple circuit breaker driven by the injected monotonic clock (deterministic in tests).

use std::time::Duration;

use gogamen_core::clock::MonotonicInstant;

/// Breaker tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BreakerConfig {
    /// Consecutive failures that open the circuit.
    pub failure_threshold: u32,
    /// Default open duration.
    pub cooldown: Duration,
    /// Upper bound for a server-requested `Retry-After`.
    pub max_cooldown: Duration,
}

impl Default for BreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            cooldown: Duration::from_secs(30),
            max_cooldown: Duration::from_secs(300),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Closed { failures: u32 },
    Open { until: MonotonicInstant },
    HalfOpen,
}

/// Closed -> Open -> HalfOpen (one trial) -> Closed/Open.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    config: BreakerConfig,
    state: State,
}

impl CircuitBreaker {
    /// Creates a closed breaker.
    pub const fn new(config: BreakerConfig) -> Self {
        Self {
            config,
            state: State::Closed { failures: 0 },
        }
    }

    /// `Ok` if a call may proceed (possibly as the half-open trial), else the remaining cool-down.
    pub fn allow(&mut self, now: MonotonicInstant) -> Result<(), Duration> {
        if let State::Open { until } = self.state {
            if now < until {
                return Err(until.saturating_duration_since(now));
            }
            self.state = State::HalfOpen;
        }
        Ok(())
    }

    /// Records a successful call.
    pub fn on_success(&mut self) {
        self.state = State::Closed { failures: 0 };
    }

    /// Records a failed call. `open_for` forces an immediate open (e.g. `Retry-After`).
    pub fn on_failure(&mut self, now: MonotonicInstant, open_for: Option<Duration>) {
        let failures = match self.state {
            State::Closed { failures } => failures.saturating_add(1),
            State::HalfOpen | State::Open { .. } => self.config.failure_threshold,
        };
        if open_for.is_some() || failures >= self.config.failure_threshold {
            let span = open_for
                .unwrap_or(self.config.cooldown)
                .min(self.config.max_cooldown);
            self.state = State::Open {
                until: now.saturating_add(span),
            };
        } else {
            self.state = State::Closed { failures };
        }
    }

    /// True while open (cool-down not yet elapsed is decided by [`Self::allow`]).
    pub const fn is_open(&self) -> bool {
        matches!(self.state, State::Open { .. })
    }
}
