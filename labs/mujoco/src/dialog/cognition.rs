//! Cognition state machine: perceive utterances through the oracle, update beliefs, decide
//! whether to `AskHuman`, and finally emit at most one goal [`Command`].
//!
//! It is tick driven and non-blocking by construction: [`Cognition::poll`] is a pure function of
//! the current tick and the (already scheduled) oracle/human outcomes. It has no handle to the
//! simulator, the action guard or the safety stop, so it cannot stall or override them; the only
//! thing it can say is "this door is my goal now".

use gogamen_core::{NoulVerdict, SystemOneProvider};

use super::fixtures::DialogFixture;
use super::oracle::{OracleMode, OracleOutcome, build_oracle, consult};
use super::perception::{Perception, Source, UpdateRecord};
use super::policy::{DialogPolicy, Door, PreAsk};

/// Output of cognition towards the control side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Command {
    /// Adopt this door as the navigation goal (still subject to the guard and the safety stop).
    Go(Door),
}

/// Something that happened in cognition, for traces and tests.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EventKind {
    /// Oracle verdict applied as evidence.
    OracleEvidence(UpdateRecord),
    /// Oracle said "unknown".
    OracleUnknown,
    /// Oracle missed its deadline.
    OracleTimedOut,
    /// Oracle unavailable (error).
    OracleUnavailable,
    /// An `AskHuman` question was sent.
    AskSent,
    /// The human replied and the reply was applied.
    HumanEvidence(UpdateRecord),
    /// The human did not reply before the deadline (belief preserved).
    AskTimedOut,
    /// Final decision: drive to a door.
    DecidedGo(Door),
    /// Final decision: keep waiting.
    DecidedWait,
}

/// Timestamped event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Event {
    /// Control tick.
    pub tick: u64,
    /// What happened.
    pub kind: EventKind,
}

#[derive(Debug, Clone, Copy)]
enum Phase {
    Perceiving { next: usize, decision_tick: u64 },
    Asking { reply_tick: u64, replies: bool },
    Done,
}

/// The cognition side of an episode.
#[derive(Debug)]
pub struct Cognition {
    fixture: DialogFixture,
    policy: DialogPolicy,
    perception: Perception,
    outcomes: Vec<(usize, OracleOutcome)>,
    phase: Phase,
    events: Vec<Event>,
    asks: u32,
    decided_at: Option<u64>,
}

impl Cognition {
    /// Start cognition at tick 0: all utterances are submitted to the oracle (virtual time, no
    /// blocking) and their outcome ticks are scheduled.
    pub fn new(fixture: &DialogFixture, policy: DialogPolicy, mode: OracleMode) -> Self {
        let provider = build_oracle(mode, fixture.id, &fixture.claims);
        Self::with_provider(fixture, policy, provider.as_ref())
    }

    /// Like [`Cognition::new`] with a caller-provided oracle port.
    pub fn with_provider(
        fixture: &DialogFixture,
        policy: DialogPolicy,
        provider: &dyn SystemOneProvider,
    ) -> Self {
        let mut outcomes: Vec<(usize, OracleOutcome)> = (0..fixture.claims.len())
            .map(|i| {
                (
                    i,
                    consult(provider, fixture.id, i, 0, policy.oracle_timeout_ticks),
                )
            })
            .collect();
        outcomes.sort_by_key(|(i, o)| (o.at_tick(), *i));
        let decision_tick = outcomes.iter().map(|(_, o)| o.at_tick()).max().unwrap_or(0);
        Self {
            fixture: fixture.clone(),
            policy,
            perception: Perception::new(
                policy.prior_p_left,
                policy.speaker_reliability,
                policy.human_reliability,
            )
            .expect("policy reliabilities in [0,1]"),
            outcomes,
            phase: Phase::Perceiving {
                next: 0,
                decision_tick,
            },
            events: Vec::new(),
            asks: 0,
            decided_at: None,
        }
    }

    /// Perception state.
    pub fn perception(&self) -> &Perception {
        &self.perception
    }

    /// Event log.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Number of `AskHuman` questions sent.
    pub fn asks(&self) -> u32 {
        self.asks
    }

    /// Tick of the final decision (go or wait), once made.
    pub fn decided_at(&self) -> Option<u64> {
        self.decided_at
    }

    /// Whether cognition is finished.
    pub fn is_done(&self) -> bool {
        matches!(self.phase, Phase::Done)
    }

    fn log(&mut self, tick: u64, kind: EventKind) {
        self.events.push(Event { tick, kind });
    }

    fn finish(&mut self, tick: u64, door: Option<Door>) -> Option<Command> {
        self.phase = Phase::Done;
        self.decided_at = Some(tick);
        match door {
            Some(d) => {
                self.log(tick, EventKind::DecidedGo(d));
                Some(Command::Go(d))
            }
            None => {
                self.log(tick, EventKind::DecidedWait);
                None
            }
        }
    }

    /// Advance cognition to `tick`; returns the goal command at most once.
    pub fn poll(&mut self, tick: u64) -> Option<Command> {
        loop {
            match self.phase {
                Phase::Done => return None,
                Phase::Perceiving {
                    next,
                    decision_tick,
                } => {
                    if next < self.outcomes.len() && self.outcomes[next].1.at_tick() <= tick {
                        let (_, outcome) = self.outcomes[next];
                        let at = outcome.at_tick();
                        match outcome {
                            OracleOutcome::Answered { verdict, .. } => {
                                let rec = self
                                    .perception
                                    .ingest(Source::Speaker, verdict == NoulVerdict::True, at)
                                    .expect("valid envelope");
                                self.log(at, EventKind::OracleEvidence(rec));
                            }
                            OracleOutcome::Unknown { .. } => self.log(at, EventKind::OracleUnknown),
                            OracleOutcome::TimedOut { .. } => {
                                self.log(at, EventKind::OracleTimedOut)
                            }
                            OracleOutcome::Unavailable { .. } => {
                                self.log(at, EventKind::OracleUnavailable)
                            }
                        }
                        self.phase = Phase::Perceiving {
                            next: next + 1,
                            decision_tick,
                        };
                        continue;
                    }
                    if next >= self.outcomes.len() && tick >= decision_tick {
                        let at = decision_tick;
                        match self.policy.pre_ask(self.perception.belief()) {
                            PreAsk::Go(d) => return self.finish(at, Some(d)),
                            PreAsk::Wait => return self.finish(at, None),
                            PreAsk::Ask => {
                                self.asks += 1;
                                self.log(at, EventKind::AskSent);
                                let h = self.fixture.human;
                                let replies = h.answer_left.is_some()
                                    && h.latency_ticks < self.policy.ask_timeout_ticks;
                                let reply_tick = at
                                    + if replies {
                                        h.latency_ticks
                                    } else {
                                        self.policy.ask_timeout_ticks
                                    };
                                self.phase = Phase::Asking {
                                    reply_tick,
                                    replies,
                                };
                                continue;
                            }
                        }
                    }
                    return None;
                }
                Phase::Asking {
                    reply_tick,
                    replies,
                } => {
                    if tick < reply_tick {
                        return None;
                    }
                    if replies {
                        let left = self
                            .fixture
                            .human
                            .answer_left
                            .expect("replies implies Some");
                        let rec = self
                            .perception
                            .ingest(Source::Human, left, reply_tick)
                            .expect("valid envelope");
                        self.log(reply_tick, EventKind::HumanEvidence(rec));
                    } else {
                        self.log(reply_tick, EventKind::AskTimedOut);
                    }
                    let door = self.policy.final_choice(self.perception.belief());
                    return self.finish(reply_tick, door);
                }
            }
        }
    }
}
