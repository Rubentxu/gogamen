//! Planner side: the only place where advice meets the pad.

use super::advice::{
    AdviceCatalog, AdviceKind, AdviceQueue, AdviceRecord, CoachAdvice, CoachObservation,
    CoachProvider, Outcome,
};
use crate::policy::{Policy, Rng};
use crate::port::{EmulatorPort, PadState};

const FORWARD: [PadState; 4] = [
    PadState::RIGHT,
    PadState::RIGHT.with(PadState::JUMP),
    PadState::RIGHT.with(PadState::RUN),
    PadState::RIGHT.with(PadState::RUN).with(PadState::JUMP),
];

/// Independent check on the planner's output: only forward pads are allowed.
/// Whatever the hypothesis says, a pad outside this set is replaced by RIGHT.
#[derive(Debug, Clone, Copy, Default)]
pub struct PadGuard;

impl PadGuard {
    pub fn check(&self, pad: PadState) -> Result<PadState, PadState> {
        if FORWARD.contains(&pad) {
            Ok(pad)
        } else {
            Err(pad)
        }
    }
}

/// Planner beliefs/preferences derived from the currently active advice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Hypothesis {
    /// Prior that a hazard (gap/spikes) is imminent: rollouts jump more often.
    pub hazard: bool,
    /// Authorised objective: reward gaining height.
    pub climb: bool,
    /// Authorised preference: rollouts run less often.
    pub cautious: bool,
}

impl Hypothesis {
    fn from_active(active: &[(AdviceKind, u64)], tick: u64) -> Self {
        let mut h = Self::default();
        for &(k, until) in active {
            if tick > until {
                continue;
            }
            match k {
                AdviceKind::GapAhead | AdviceKind::AvoidSpikes => h.hazard = true,
                AdviceKind::ClimbPlatform => h.climb = true,
                AdviceKind::SlowDown => h.cautious = true,
            }
        }
        h
    }
}

/// Everything the coach link did, for metrics.
#[derive(Debug, Clone, Default)]
pub struct CoachReport {
    pub records: Vec<AdviceRecord>,
    /// Decisions in which the hypothesis differed from the default.
    pub decisions_with_advice: u64,
}

impl CoachReport {
    pub fn count(&self, f: impl Fn(&Outcome) -> bool) -> usize {
        self.records.iter().filter(|r| f(&r.outcome)).count()
    }
    pub fn applied(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Applied { .. }))
    }
    pub fn expired(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Expired { .. }))
    }
    pub fn rejected(&self) -> usize {
        self.count(|o| matches!(o, Outcome::Rejected(_)))
    }
    /// (mean, max) emission-to-effect latency over applied advice, in ticks.
    pub fn latency(&self) -> Option<(f64, u64)> {
        let l: Vec<u64> = self
            .records
            .iter()
            .filter_map(AdviceRecord::latency)
            .collect();
        if l.is_empty() {
            return None;
        }
        Some((
            l.iter().sum::<u64>() as f64 / l.len() as f64,
            *l.iter().max().unwrap(),
        ))
    }
}

/// Monte Carlo planner with an optional coach. The tick of the virtual clock is
/// the decision index; `act` always returns after bounded work and never waits
/// for the coach.
pub struct CoachedPlanner {
    rng: Rng,
    rollouts: u32,
    horizon: u32,
    steps: u64,
    tick: u64,
    coach: Option<Box<dyn CoachProvider>>,
    catalog: AdviceCatalog,
    queue: AdviceQueue,
    active: Vec<(AdviceKind, u64)>,
    guard: PadGuard,
    pub report: CoachReport,
    last_hypothesis: Hypothesis,
}

const DEATH_PENALTY: i64 = 1_000;
const CLIMB_WEIGHT: i64 = 6;

impl CoachedPlanner {
    pub fn new(seed: u64, rollouts: u32, horizon: u32) -> Self {
        Self {
            rng: Rng::new(seed),
            rollouts,
            horizon,
            steps: 0,
            tick: 0,
            coach: None,
            catalog: AdviceCatalog::standard(),
            queue: AdviceQueue::default(),
            active: Vec::new(),
            guard: PadGuard,
            report: CoachReport::default(),
            last_hypothesis: Hypothesis::default(),
        }
    }
    pub fn with_coach(mut self, c: Box<dyn CoachProvider>) -> Self {
        self.coach = Some(c);
        self
    }
    pub fn with_catalog(mut self, c: AdviceCatalog) -> Self {
        self.catalog = c;
        self
    }
    pub fn hypothesis(&self) -> Hypothesis {
        self.last_hypothesis
    }

    /// Non-blocking coach step: collect emissions, admit them, deliver what is
    /// due on the virtual clock and drop what is stale.
    fn pump(&mut self, obs: CoachObservation) {
        if let Some(c) = self.coach.as_mut() {
            for em in c.emit(&obs) {
                let idx = self.report.records.len();
                let mut rec = AdviceRecord {
                    text: em.raw.text.clone(),
                    kind: None,
                    emitted_tick: obs.tick,
                    outcome: Outcome::Pending,
                };
                match self.catalog.admit(&em.raw, obs.tick, &em.timing) {
                    Ok(a) => {
                        rec.kind = Some(a.kind());
                        self.report.records.push(rec);
                        self.queue.schedule(a, idx);
                    }
                    Err(e) => {
                        rec.outcome = Outcome::Rejected(e);
                        self.report.records.push(rec);
                    }
                }
            }
        }
        for (a, idx) in self.queue.drain_due(obs.tick) {
            self.report.records[idx].outcome = self.deliver(&a, obs.tick);
        }
        self.active.retain(|&(_, until)| until >= obs.tick);
    }

    fn deliver(&mut self, a: &CoachAdvice, tick: u64) -> Outcome {
        if tick > a.deadline_tick() {
            return Outcome::Expired { at_tick: tick };
        }
        self.active.push((a.kind(), tick + a.ttl_ticks()));
        Outcome::Applied { at_tick: tick }
    }

    fn sample(&mut self, h: Hypothesis) -> PadState {
        let jump_pm = if h.hazard { 850 } else { 500 };
        let run_pm = if h.cautious { 100 } else { 500 };
        let jump = self.rng.below(1000) < jump_pm;
        let run = self.rng.below(1000) < run_pm;
        let mut p = PadState::RIGHT;
        if jump {
            p = p.with(PadState::JUMP);
        }
        if run {
            p = p.with(PadState::RUN);
        }
        p
    }
}

impl Policy for CoachedPlanner {
    fn name(&self) -> &'static str {
        if self.coach.is_some() {
            "coached_monte_carlo"
        } else {
            "monte_carlo_no_coach"
        }
    }
    fn sim_steps(&self) -> u64 {
        self.steps
    }
    fn act(&mut self, port: &mut dyn EmulatorPort, frames: u32) -> PadState {
        let tick = self.tick;
        self.tick += 1;
        let base = port.features();
        self.pump(CoachObservation {
            tick,
            x: base.x,
            y: base.y,
            on_ground: base.on_ground,
            deaths: base.deaths,
        });
        let h = Hypothesis::from_active(&self.active, tick);
        self.last_hypothesis = h;
        if h != Hypothesis::default() {
            self.report.decisions_with_advice += 1;
        }
        let Ok(root) = port.snapshot() else {
            return PadState::RIGHT;
        };
        // Advice is a soft prior: it biases rollout sampling and the score, but
        // never prunes candidates, so rollouts can still contradict a wrong hint.
        let cands: Vec<PadState> = FORWARD.to_vec();
        let mut best = (i64::MIN, PadState::RIGHT);
        for &first in &cands {
            let mut total = 0i64;
            for _ in 0..self.rollouts {
                let mut pad = first;
                for _ in 0..self.horizon {
                    let r = port.step(pad, frames);
                    self.steps += 1;
                    if r.features.is_terminal() || r.features.deaths > base.deaths {
                        break;
                    }
                    pad = self.sample(h);
                }
                let f = port.features();
                let mut score = i64::from(f.x - base.x);
                score -= DEATH_PENALTY * i64::from(f.deaths - base.deaths);
                if f.level_complete {
                    score += DEATH_PENALTY;
                }
                if h.climb {
                    score += CLIMB_WEIGHT * i64::from(f.y - base.y);
                }
                total += score;
                if port.restore(&root).is_err() {
                    return PadState::RIGHT;
                }
            }
            if total > best.0 {
                best = (total, first);
            }
        }
        self.guard.check(best.1).unwrap_or(PadState::RIGHT)
    }
}
