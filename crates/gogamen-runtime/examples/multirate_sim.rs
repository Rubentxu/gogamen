//! Simulated multirate control demo (SIMULATION_ONLY, no hardware, virtual clock).
//!
//! A fast control loop (10 ms period) runs every tick without waiting for a slow best-effort
//! planner whose proposals arrive late. Times are virtual: they are NOT hard real-time guarantees.
//! Run: `cargo run -p gogamen-runtime --example multirate_sim --offline`

use std::sync::Arc;
use std::time::Duration;

use gogamen_core::{
    Action, ActionBounds, ActionGuard, ActionSpace, MonotonicClock, MonotonicDeadline, Observation,
    SafeFallbackPolicy, SimDuration, Terminal, Tick, VirtualMonotonicClock,
};
use gogamen_runtime::{
    ControlPeriod, MultirateConfig, MultirateController, MultirateFallbackReason, SensorReading,
    WorkerProposal,
};

const TICKS: u64 = 40;
/// The simulated planner needs 3 control periods per plan, so its proposals are stale on arrival
/// for some ticks and the safe fallback must take over without blocking.
const PLANNER_LATENCY_TICKS: u64 = 3;

fn main() {
    println!("SIMULATION_ONLY multirate control demo (virtual clock, not hard real-time)");
    let clock = Arc::new(VirtualMonotonicClock::new());
    let space = ActionSpace::continuous(vec![ActionBounds::new(-1.0, 1.0).expect("bounds")])
        .expect("action space");
    let fallback = Action::continuous(vec![0.0]).expect("fallback");
    let guard = ActionGuard::new(SafeFallbackPolicy::ExplicitFallback(fallback));
    let config = MultirateConfig::new(ControlPeriod::TEN_MS)
        .with_proposal_max_age(Duration::from_millis(25))
        .with_observation_max_age(Duration::from_millis(40))
        .with_watchdog_timeout(Duration::from_millis(100));
    let mut controller = MultirateController::new(config, space, guard, clock.clone());

    let mut pending: Option<(u64, WorkerProposal)> = None;
    let mut revision = 0_u64;
    for k in 0..TICKS {
        let now = clock.now();
        // Sensor thread: fresh reading every tick.
        revision += 1;
        let obs = Observation::new(Tick::new(k), SimDuration::ZERO, vec![k as f64 * 0.01])
            .expect("observation");
        controller
            .submit_observation(SensorReading::new(obs, now, revision, 1))
            .expect("submit observation");
        // Slow planner: starts a plan every 5 ticks, delivers it PLANNER_LATENCY_TICKS later.
        if k % 5 == 0 {
            let action = Action::continuous(vec![0.5]).expect("action");
            pending = Some((
                k + PLANNER_LATENCY_TICKS,
                WorkerProposal {
                    action,
                    proposal_tick: Tick::new(k),
                    revision: k / 5 + 1,
                    epoch: 1,
                    observation_revision: revision,
                    observation_timestamp: now,
                    produced_at: now,
                    deadline: MonotonicDeadline::at(now.saturating_add(Duration::from_millis(30))),
                },
            ));
        }
        if pending.as_ref().is_some_and(|(due, _)| *due <= k) {
            let (_, mut p) = pending.take().expect("pending");
            // The plan is only complete now: stamp it with the delivery instant.
            p.produced_at = now;
            p.deadline = MonotonicDeadline::at(now.saturating_add(Duration::from_millis(30)));
            controller.submit_proposal(p).expect("submit proposal");
        }
        controller.step_control_tick(Terminal::Ongoing);
        clock.advance(Duration::from_millis(10));
    }

    let trace = controller.trace();
    println!(
        "period_ms: {}",
        trace.control_period().as_duration().as_millis()
    );
    println!("ticks: {}", trace.len());
    println!("fallbacks: {}", trace.fallback_count());
    for reason in [
        MultirateFallbackReason::NoProposalYet,
        MultirateFallbackReason::StalePlan,
        MultirateFallbackReason::StaleObservation,
        MultirateFallbackReason::WatchdogExpired,
    ] {
        println!(
            "fallback_{reason:?}: {}",
            trace.count_by_fallback_reason(reason)
        );
    }
    println!("max_abs_jitter_ns: {}", trace.max_abs_jitter_nanos());
    println!(
        "max_control_latency_ns: {}",
        trace.max_control_latency().as_nanos()
    );
    println!("note: {}", trace.timing_guarantee_note());
}
