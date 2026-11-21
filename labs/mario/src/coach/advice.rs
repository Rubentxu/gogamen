//! Coach side of GOG-63: untrusted semantic advice, the authorised catalog and
//! the virtual-clock delivery queue.
//!
//! Layering rule (enforced by a textual test): this module never names the pad
//! or the emulator port. The coach produces [`CoachAdvice`], never an action;
//! only the planner (`coach::planner`) turns a hypothesis into a pad, and its
//! output still goes through the pad guard.

use std::fmt;

/// Authorised advice vocabulary (the "catalog of allowed preferences").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdviceKind {
    /// Hypothesis: a gap is ahead (prior on danger raised; jumps favoured).
    GapAhead,
    /// Hypothesis: spikes are ahead.
    AvoidSpikes,
    /// Objective: reach higher ground (authorised preference: height bonus).
    ClimbPlatform,
    /// Preference: be cautious (fewer run actions considered).
    SlowDown,
}

impl AdviceKind {
    pub const ALL: [AdviceKind; 4] = [
        Self::GapAhead,
        Self::AvoidSpikes,
        Self::ClimbPlatform,
        Self::SlowDown,
    ];
    /// Canonical catalog key.
    pub const fn key(self) -> &'static str {
        match self {
            Self::GapAhead => "gap_ahead",
            Self::AvoidSpikes => "avoid_spikes",
            Self::ClimbPlatform => "climb_platform",
            Self::SlowDown => "slow_down",
        }
    }
}

/// Untrusted text exactly as the coach produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawAdvice {
    pub text: String,
}

impl RawAdvice {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Typed rejection of an advice that is not admissible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdviceError {
    Empty,
    TooLong {
        len: usize,
    },
    /// Text is not a catalog key (e.g. an attempt to dictate controls).
    NotInCatalog {
        text: String,
    },
    /// Known key, but not enabled in this catalog instance.
    Unauthorized(AdviceKind),
}

impl fmt::Display for AdviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "empty advice"),
            Self::TooLong { len } => write!(f, "advice too long ({len} bytes)"),
            Self::NotInCatalog { text } => write!(f, "advice not in catalog: {text:?}"),
            Self::Unauthorized(k) => write!(f, "advice kind {} not authorised", k.key()),
        }
    }
}

impl std::error::Error for AdviceError {}

/// Admitted advice. Fields are private and the only constructor is
/// [`AdviceCatalog::admit`], so an instance always came through the catalog.
/// There is deliberately no conversion to any action type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoachAdvice {
    id: u64,
    kind: AdviceKind,
    emitted_tick: u64,
    deliver_tick: u64,
    deadline_tick: u64,
    ttl_ticks: u64,
}

impl CoachAdvice {
    pub fn id(&self) -> u64 {
        self.id
    }
    pub fn kind(&self) -> AdviceKind {
        self.kind
    }
    pub fn emitted_tick(&self) -> u64 {
        self.emitted_tick
    }
    pub fn deliver_tick(&self) -> u64 {
        self.deliver_tick
    }
    /// Last tick at which delivery is still accepted.
    pub fn deadline_tick(&self) -> u64 {
        self.deadline_tick
    }
    /// Ticks the hypothesis/preference stays active once applied.
    pub fn ttl_ticks(&self) -> u64 {
        self.ttl_ticks
    }
}

/// Catalog of authorised advice.
#[derive(Debug, Clone)]
pub struct AdviceCatalog {
    allowed: Vec<AdviceKind>,
    next_id: u64,
}

pub const MAX_ADVICE_LEN: usize = 48;
/// Upper bound for ttl/max-age a coach may request (ticks).
pub const MAX_TTL_TICKS: u64 = 32;

impl AdviceCatalog {
    pub fn standard() -> Self {
        Self::with_allowed(&AdviceKind::ALL)
    }
    pub fn with_allowed(allowed: &[AdviceKind]) -> Self {
        Self {
            allowed: allowed.to_vec(),
            next_id: 0,
        }
    }
    pub fn allowed(&self) -> &[AdviceKind] {
        &self.allowed
    }

    /// Parse and authorise one emission. Pure in `raw`; assigns the next id.
    pub fn admit(
        &mut self,
        raw: &RawAdvice,
        emitted_tick: u64,
        e: &Timing,
    ) -> Result<CoachAdvice, AdviceError> {
        let t = raw.text.trim();
        if t.is_empty() {
            return Err(AdviceError::Empty);
        }
        if t.len() > MAX_ADVICE_LEN {
            return Err(AdviceError::TooLong { len: t.len() });
        }
        let kind = AdviceKind::ALL
            .into_iter()
            .find(|k| k.key() == t)
            .ok_or_else(|| AdviceError::NotInCatalog {
                text: t.to_string(),
            })?;
        if !self.allowed.contains(&kind) {
            return Err(AdviceError::Unauthorized(kind));
        }
        let id = self.next_id;
        self.next_id += 1;
        let deliver = emitted_tick + e.latency_ticks;
        Ok(CoachAdvice {
            id,
            kind,
            emitted_tick,
            deliver_tick: deliver,
            deadline_tick: emitted_tick + e.max_age_ticks.min(MAX_TTL_TICKS * 4),
            ttl_ticks: e.ttl_ticks.clamp(1, MAX_TTL_TICKS),
        })
    }
}

/// Timing requested by the coach for one emission (virtual clock, ticks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timing {
    /// Network/model latency from emission to delivery.
    pub latency_ticks: u64,
    /// Effect duration once applied.
    pub ttl_ticks: u64,
    /// Delivery after `emitted + max_age_ticks` is discarded as stale.
    pub max_age_ticks: u64,
}

/// One coach output before admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Emission {
    pub raw: RawAdvice,
    pub timing: Timing,
}

/// What the coach is told about the game (a copy; no handle to the port).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CoachObservation {
    pub tick: u64,
    pub x: i32,
    pub y: i32,
    pub on_ground: bool,
    pub deaths: u32,
}

/// Coach back-end (human-like, LLM, mock). `emit` must return immediately;
/// latency is modelled by `Timing`, never by blocking the decision loop.
pub trait CoachProvider {
    fn name(&self) -> &'static str;
    fn emit(&mut self, obs: &CoachObservation) -> Vec<Emission>;
}

/// Terminal state of one advice in the log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Pending,
    Rejected(AdviceError),
    /// Delivered after its deadline; dropped without effect.
    Expired {
        at_tick: u64,
    },
    /// Took effect on the decision of this tick.
    Applied {
        at_tick: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdviceRecord {
    pub text: String,
    pub kind: Option<AdviceKind>,
    pub emitted_tick: u64,
    pub outcome: Outcome,
}

impl AdviceRecord {
    /// Emission-to-effect latency in ticks when applied.
    pub fn latency(&self) -> Option<u64> {
        match self.outcome {
            Outcome::Applied { at_tick } => Some(at_tick - self.emitted_tick),
            _ => None,
        }
    }
}

/// Virtual-clock queue; ordered by `(deliver_tick, seq)`. Never blocks.
#[derive(Debug, Default)]
pub struct AdviceQueue {
    pending: Vec<(u64, CoachAdvice, usize)>, // (seq, advice, record index)
    seq: u64,
}

impl AdviceQueue {
    pub fn schedule(&mut self, a: CoachAdvice, record: usize) {
        let key = (a.deliver_tick, self.seq);
        self.seq += 1;
        let pos = self
            .pending
            .partition_point(|(s, p, _)| (p.deliver_tick, *s) <= key);
        self.pending.insert(pos, (key.1, a, record));
    }
    /// Remove and return everything due at `tick` (never waits for the rest).
    pub fn drain_due(&mut self, tick: u64) -> Vec<(CoachAdvice, usize)> {
        let n = self
            .pending
            .partition_point(|(_, p, _)| p.deliver_tick <= tick);
        self.pending.drain(..n).map(|(_, a, r)| (a, r)).collect()
    }
    pub fn len(&self) -> usize {
        self.pending.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}
