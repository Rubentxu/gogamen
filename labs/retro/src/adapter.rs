//! GOG-61 digital adapter: an in-process, resident, headless emulator bridge.
//!
//! The adapter owns a [`Session`] over any [`LibretroCorePort`] (no per-frame
//! IPC, no window, no audio) and exposes it through the existing
//! `gogamen-core` traits ([`LiveEnvironment`], [`BranchableSnapshot`]) plus a
//! pure [`GenerativeModel`] ([`SnapshotForkModel`]) for planners such as FMC.
//!
//! # Declared access mode
//! `gogamen-core` has no explicit "access mode" type: capability is expressed
//! structurally (a planner only receives a [`GenerativeModel`] through a
//! `PlanningContext`; a live environment is `Actuator` + `EnvironmentObserver`).
//! [`AccessMode`] is therefore a small adapter-local declaration, part of the
//! spec hash. This adapter declares [`AccessMode::SnapshotForkable`]: planners
//! reach the emulator ONLY through [`SnapshotForkModel`], which forks the
//! adapter and restores an encoded state per simulated step; it never mutates
//! the live adapter nor exposes the core.
//!
//! # Versioning
//! [`AdapterSpec`] bundles the adapter version, access mode, [`AdapterConfig`]
//! (`frame_skip`, `reward_fn_version`, `terminal_fn_version`, goal/horizon),
//! core geometry and action table version. Its [`AdapterSpec::hash`] is embedded
//! in every adapter snapshot and in encoded planning states; restoring across
//! different specs is rejected with [`AdapterError::SpecMismatch`].
//!
//! # Functions v1
//! * reward v1: wrapping score delta over the macro-step; score is the `u16`
//!   LE at `ram[4..6]` (FreePlay layout).
//! * terminal v1: `GoalReached` when score >= `goal_score`; `HorizonReached`
//!   when `steps >= max_steps`.
//! * action table v1: 9 discrete actions (noop + 8 directions).

use std::error::Error;
use std::fmt;

use gogamen_core::{
    Action, ActionSpace, Actuator, ApplyOutcome, BranchableSnapshot, DomainError, EnvironmentError,
    EnvironmentObserver, GenerativeModel, LiveEnvironment, ModelError, Observation, RngSource,
    SimDuration, SimulationSnapshot, SnapshotError, State, StepDt, Terminal, TerminalReason, Tick,
    Transition,
};

use crate::{LibretroCorePort, Session, StateError, buttons, fnv1a};

/// Adapter implementation version (part of the spec hash).
pub const ADAPTER_VERSION: u32 = 1;
/// Version of the discrete action to joypad mask table.
pub const ACTION_TABLE_VERSION: u32 = 1;

const SNAP_MAGIC: &[u8; 4] = b"GAD1";
const FPS: f64 = 60.0;
const SCORE_OFFSET: usize = 4;

/// Joypad masks per discrete action index (table v1).
const ACTION_MASKS: [u16; 9] = [
    0,
    buttons::UP,
    buttons::DOWN,
    buttons::LEFT,
    buttons::RIGHT,
    buttons::UP | buttons::LEFT,
    buttons::UP | buttons::RIGHT,
    buttons::DOWN | buttons::LEFT,
    buttons::DOWN | buttons::RIGHT,
];

/// Access the adapter grants to controllers/planners.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessMode {
    /// Planners may only fork/restore snapshots (via [`SnapshotForkModel`]);
    /// the live adapter is acted on only by the real agent loop.
    SnapshotForkable,
}

impl AccessMode {
    fn tag(self) -> u8 {
        match self {
            Self::SnapshotForkable => 1,
        }
    }
}

/// Versioned adapter configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterConfig {
    /// Frames the core runs per decision (>= 1); the action is held.
    pub frame_skip: u32,
    /// Reward function version (only 1 exists).
    pub reward_fn_version: u32,
    /// Terminal function version (only 1 exists).
    pub terminal_fn_version: u32,
    /// Score at which terminal v1 reports `GoalReached`.
    pub goal_score: u16,
    /// Decisions after which terminal v1 reports `HorizonReached`.
    pub max_steps: u64,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            frame_skip: 4,
            reward_fn_version: 1,
            terminal_fn_version: 1,
            goal_score: 5,
            max_steps: 1_000,
        }
    }
}

/// Full identity of an adapter; hashed into snapshots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdapterSpec {
    pub adapter_version: u32,
    pub action_table_version: u32,
    pub access: AccessMode,
    pub config: AdapterConfig,
    pub frame_width: u32,
    pub frame_height: u32,
    pub ram_len: u32,
}

impl AdapterSpec {
    /// Stable FNV-1a identity hash over every field.
    pub fn hash(&self) -> u64 {
        let c = &self.config;
        let mut b = Vec::with_capacity(64);
        for v in [
            self.adapter_version,
            self.action_table_version,
            u32::from(self.access.tag()),
            c.frame_skip,
            c.reward_fn_version,
            c.terminal_fn_version,
            u32::from(c.goal_score),
            self.frame_width,
            self.frame_height,
            self.ram_len,
        ] {
            b.extend_from_slice(&v.to_le_bytes());
        }
        b.extend_from_slice(&c.max_steps.to_le_bytes());
        fnv1a(&b)
    }
}

/// Typed adapter errors (never panics on core/state failures).
#[derive(Debug, Clone, PartialEq)]
pub enum AdapterError {
    /// Invalid configuration or unsupported function version.
    Config(&'static str),
    /// Action rejected by the action space.
    InvalidAction(DomainError),
    /// Stepping a terminated episode.
    AlreadyTerminal,
    /// Snapshot written under a different spec.
    SpecMismatch { expected: u64, actual: u64 },
    /// Snapshot envelope is malformed (magic, checksum, fields).
    BadSnapshot(&'static str),
    /// The core rejected the state; the adapter was rolled back.
    Core(StateError),
    /// The core rejected the state and the rollback also failed.
    RollbackFailed(StateError),
    /// Core data inconsistent with the declared layout.
    Layout(&'static str),
    /// Domain value construction failed.
    Domain(DomainError),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "adapter error: {self:?}")
    }
}

impl Error for AdapterError {}

impl From<DomainError> for AdapterError {
    fn from(e: DomainError) -> Self {
        Self::Domain(e)
    }
}

/// Result of one decision step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOutcome {
    pub reward: f64,
    pub terminal: Terminal,
    /// Core frames executed (== `frame_skip`).
    pub frames: u32,
}

fn terminal_tag(t: Terminal) -> u8 {
    match t {
        Terminal::Ongoing => 0,
        Terminal::Terminated(TerminalReason::GoalReached) => 1,
        Terminal::Terminated(_) => 2,
    }
}

fn terminal_from_tag(t: u8) -> Option<Terminal> {
    match t {
        0 => Some(Terminal::Ongoing),
        1 => Some(Terminal::Terminated(TerminalReason::GoalReached)),
        2 => Some(Terminal::Terminated(TerminalReason::HorizonReached)),
        _ => None,
    }
}

fn score_of(ram: &[u8]) -> Result<u16, AdapterError> {
    ram.get(SCORE_OFFSET..SCORE_OFFSET + 2)
        .map(|s| u16::from_le_bytes([s[0], s[1]]))
        .ok_or(AdapterError::Layout("ram too short for score"))
}

/// Resident headless emulator environment. See the module docs.
pub struct Adapter<C: LibretroCorePort> {
    session: Session<C>,
    spec: AdapterSpec,
    space: ActionSpace,
    steps: u64,
    terminal: Terminal,
    last_reward: f64,
}

impl<C: LibretroCorePort> Adapter<C> {
    /// Build an adapter (resets the core). Rejects unknown function versions.
    pub fn new(core: C, config: AdapterConfig) -> Result<Self, AdapterError> {
        if config.frame_skip == 0 {
            return Err(AdapterError::Config("frame_skip must be >= 1"));
        }
        if config.reward_fn_version != 1 {
            return Err(AdapterError::Config("unsupported reward_fn_version"));
        }
        if config.terminal_fn_version != 1 {
            return Err(AdapterError::Config("unsupported terminal_fn_version"));
        }
        let (w, h) = core.frame_size();
        let spec = AdapterSpec {
            adapter_version: ADAPTER_VERSION,
            action_table_version: ACTION_TABLE_VERSION,
            access: AccessMode::SnapshotForkable,
            config,
            frame_width: w as u32,
            frame_height: h as u32,
            ram_len: core.ram().len() as u32,
        };
        let mut session = Session::new(core);
        session.reset();
        score_of(session.core().ram())?;
        Ok(Self {
            session,
            spec,
            space: ActionSpace::discrete(ACTION_MASKS.len())?,
            steps: 0,
            terminal: Terminal::Ongoing,
            last_reward: 0.0,
        })
    }

    pub fn spec(&self) -> &AdapterSpec {
        &self.spec
    }

    /// Declared access mode.
    pub fn access_mode(&self) -> AccessMode {
        self.spec.access
    }

    /// Decisions taken since reset/restore point.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    pub fn terminal(&self) -> Terminal {
        self.terminal
    }

    /// Reward of the last step (0 after reset).
    pub fn last_reward(&self) -> f64 {
        self.last_reward
    }

    /// Read-only core frame (palette indices), for headless inspection.
    pub fn framebuffer(&self) -> &[u8] {
        self.session.core().framebuffer()
    }

    pub fn frame_hash(&self) -> u64 {
        self.session.frame_hash()
    }

    /// Simulated time per decision (`frame_skip / 60` s).
    pub fn step_dt(&self) -> Result<StepDt, DomainError> {
        StepDt::from_secs_f64(f64::from(self.spec.config.frame_skip) / FPS)
    }

    /// Reset to power-on; returns nothing (see [`LiveEnvironment::reset`]).
    pub fn reset_episode(&mut self) {
        self.session.reset();
        self.steps = 0;
        self.terminal = Terminal::Ongoing;
        self.last_reward = 0.0;
    }

    /// Execute one decision (`frame_skip` frames holding the action).
    pub fn step(&mut self, action_index: usize) -> Result<StepOutcome, AdapterError> {
        let action = Action::discrete_in_space(action_index, &self.space)
            .map_err(AdapterError::InvalidAction)?;
        self.step_action(&action)
    }

    fn step_action(&mut self, action: &Action) -> Result<StepOutcome, AdapterError> {
        self.space
            .validate(action)
            .map_err(AdapterError::InvalidAction)?;
        if self.terminal.is_terminal() {
            return Err(AdapterError::AlreadyTerminal);
        }
        let idx = action
            .as_discrete()
            .ok_or(AdapterError::Layout("non-discrete action"))?
            .index();
        let mask = ACTION_MASKS[idx];
        let before = score_of(self.session.core().ram())?;
        for _ in 0..self.spec.config.frame_skip {
            self.session.step(mask);
        }
        let after = score_of(self.session.core().ram())?;
        self.steps += 1;
        let reward = f64::from(after.wrapping_sub(before));
        self.terminal = if after >= self.spec.config.goal_score {
            Terminal::Terminated(TerminalReason::GoalReached)
        } else if self.steps >= self.spec.config.max_steps {
            Terminal::Terminated(TerminalReason::HorizonReached)
        } else {
            Terminal::Ongoing
        };
        self.last_reward = reward;
        Ok(StepOutcome {
            reward,
            terminal: self.terminal,
            frames: self.spec.config.frame_skip,
        })
    }

    fn elapsed(&self) -> Result<SimDuration, DomainError> {
        SimDuration::from_secs_f64(self.steps as f64 * f64::from(self.spec.config.frame_skip) / FPS)
    }

    fn observation(&self) -> Result<Observation, DomainError> {
        let f = self.session.core().ram().iter().map(|&b| f64::from(b));
        Observation::new(Tick::new(self.steps), self.elapsed()?, f.collect())
    }

    /// Versioned, checksummed snapshot:
    /// `GAD1 | spec_hash | steps | terminal | len | session snapshot | fnv`.
    pub fn snapshot_bytes(&self) -> Vec<u8> {
        let inner = self.session.snapshot();
        let mut v = Vec::with_capacity(inner.len() + 33);
        v.extend_from_slice(SNAP_MAGIC);
        v.extend_from_slice(&self.spec.hash().to_le_bytes());
        v.extend_from_slice(&self.steps.to_le_bytes());
        v.push(terminal_tag(self.terminal));
        v.extend_from_slice(&(inner.len() as u32).to_le_bytes());
        v.extend_from_slice(&inner);
        let sum = fnv1a(&v);
        v.extend_from_slice(&sum.to_le_bytes());
        v
    }

    /// Restore a snapshot. On ANY error the adapter is left unchanged: the
    /// envelope and spec are validated before touching the core, and if the
    /// core fails (even after partially mutating itself) the previous state is
    /// re-applied.
    pub fn restore_bytes(&mut self, data: &[u8]) -> Result<(), AdapterError> {
        const HEAD: usize = 4 + 8 + 8 + 1 + 4;
        if data.len() < HEAD + 8 {
            return Err(AdapterError::BadSnapshot("truncated"));
        }
        if &data[..4] != SNAP_MAGIC {
            return Err(AdapterError::BadSnapshot("bad magic"));
        }
        let (body, sum) = data.split_at(data.len() - 8);
        if fnv1a(body) != u64::from_le_bytes(sum.try_into().expect("8 bytes")) {
            return Err(AdapterError::BadSnapshot("bad checksum"));
        }
        let actual = u64::from_le_bytes(body[4..12].try_into().expect("8 bytes"));
        let expected = self.spec.hash();
        if actual != expected {
            return Err(AdapterError::SpecMismatch { expected, actual });
        }
        let steps = u64::from_le_bytes(body[12..20].try_into().expect("8 bytes"));
        let terminal =
            terminal_from_tag(body[20]).ok_or(AdapterError::BadSnapshot("bad terminal tag"))?;
        let len = u32::from_le_bytes(body[21..25].try_into().expect("4 bytes")) as usize;
        if body.len() - HEAD != len {
            return Err(AdapterError::BadSnapshot("bad length"));
        }
        let backup = self.session.snapshot();
        if let Err(e) = self.session.restore(&body[HEAD..]) {
            return match self.session.restore(&backup) {
                Ok(()) => Err(AdapterError::Core(e)),
                Err(r) => Err(AdapterError::RollbackFailed(r)),
            };
        }
        self.steps = steps;
        self.terminal = terminal;
        self.last_reward = 0.0;
        Ok(())
    }

    /// Encode the adapter as a planning [`State`]:
    /// `[ram_len, ram bytes.., snapshot bytes..]` (all exact `f64` bytes).
    fn encode_state(&self) -> Result<State, AdapterError> {
        let ram = self.session.core().ram();
        let snap = self.snapshot_bytes();
        let mut v = Vec::with_capacity(1 + ram.len() + snap.len());
        v.push(ram.len() as f64);
        v.extend(ram.iter().map(|&b| f64::from(b)));
        v.extend(snap.iter().map(|&b| f64::from(b)));
        Ok(State::new(Tick::new(self.steps), self.elapsed()?, v)?)
    }

    fn decode_state(&mut self, state: &State) -> Result<(), AdapterError> {
        let vals = state.values();
        let ram_len = self.spec.ram_len as usize;
        if vals.first().copied() != Some(ram_len as f64) || vals.len() <= 1 + ram_len {
            return Err(AdapterError::BadSnapshot("bad planning state header"));
        }
        let bytes: Option<Vec<u8>> = vals[1 + ram_len..]
            .iter()
            .map(|&x| (x.fract() == 0.0 && (0.0..=255.0).contains(&x)).then_some(x as u8))
            .collect();
        let bytes = bytes.ok_or(AdapterError::BadSnapshot("non-byte value in state"))?;
        self.restore_bytes(&bytes)
    }
}

impl<C: LibretroCorePort + Clone> Adapter<C> {
    /// Independent copy (core cloned, session position carried over).
    pub fn fork_adapter(&self) -> Result<Self, AdapterError> {
        let mut session = Session::new(self.session.core().clone());
        session
            .restore(&self.session.snapshot())
            .map_err(AdapterError::Core)?;
        Ok(Self {
            session,
            spec: self.spec,
            space: self.space.clone(),
            steps: self.steps,
            terminal: self.terminal,
            last_reward: self.last_reward,
        })
    }

    /// The only planning entry point: a pure model that forks this adapter.
    pub fn planning_model(&self) -> Result<SnapshotForkModel<C>, AdapterError> {
        Ok(SnapshotForkModel {
            template: self.fork_adapter()?,
        })
    }

    /// Current state encoded for planners (see [`SnapshotForkModel`]).
    pub fn planning_state(&self) -> Result<State, AdapterError> {
        self.encode_state()
    }
}

fn env_err(e: AdapterError) -> EnvironmentError {
    match e {
        AdapterError::InvalidAction(d) => EnvironmentError::InvalidAction(d),
        AdapterError::AlreadyTerminal => EnvironmentError::AlreadyTerminal,
        AdapterError::Domain(_) => EnvironmentError::ObservationFailed {
            reason: "invalid observation values",
        },
        _ => EnvironmentError::ActuationFailed {
            reason: "adapter core failure",
        },
    }
}

impl<C: LibretroCorePort> EnvironmentObserver for Adapter<C> {
    fn observe(&self) -> Result<Observation, EnvironmentError> {
        self.observation()
            .map_err(|_| EnvironmentError::ObservationFailed {
                reason: "invalid observation values",
            })
    }
}

impl<C: LibretroCorePort> Actuator for Adapter<C> {
    fn action_space(&self) -> &ActionSpace {
        &self.space
    }

    fn apply(&mut self, action: &Action) -> Result<ApplyOutcome, EnvironmentError> {
        let out = self.step_action(action).map_err(env_err)?;
        Ok(ApplyOutcome::new(self.observe()?, out.terminal))
    }
}

impl<C: LibretroCorePort> LiveEnvironment for Adapter<C> {
    fn reset(&mut self) -> Result<Observation, EnvironmentError> {
        self.reset_episode();
        self.observe()
    }
}

fn snap_err(e: AdapterError) -> SnapshotError {
    match e {
        AdapterError::Domain(d) => SnapshotError::Domain(d),
        AdapterError::SpecMismatch { .. } => {
            SnapshotError::Domain(DomainError::InvalidTransition {
                reason: "adapter spec mismatch",
            })
        }
        _ => SnapshotError::Domain(DomainError::InvalidTransition {
            reason: "adapter snapshot rejected",
        }),
    }
}

/// Snapshot-forkable access through the core contract. The encoded
/// [`State`] carries the full versioned adapter snapshot, so
/// `restore` rejects snapshots from another spec and keeps the adapter
/// unchanged on failure. `fork` needs `C: Clone`.
impl<C: LibretroCorePort + Clone> BranchableSnapshot for Adapter<C> {
    fn snapshot(&self) -> Result<SimulationSnapshot, SnapshotError> {
        let state = self.encode_state().map_err(snap_err)?;
        SimulationSnapshot::from_parts(
            state,
            0,
            self.terminal,
            Tick::new(self.steps),
            self.elapsed()?,
        )
    }

    fn fork(&self) -> Result<Self, SnapshotError> {
        self.fork_adapter().map_err(snap_err)
    }

    fn restore(&mut self, snapshot: &SimulationSnapshot) -> Result<(), SnapshotError> {
        snapshot.verify()?;
        self.decode_state(snapshot.state()).map_err(snap_err)
    }
}

/// Pure [`GenerativeModel`] over a snapshot-forkable adapter. Each simulated
/// step forks the private template, restores the given [`State`], steps the
/// fork and encodes the result; the template and any live adapter are never
/// mutated. Planning states are `[ram_len, ram.., snapshot..]`; use
/// [`planning_features`] as `phi` and [`planning_reward`] as reward function.
pub struct SnapshotForkModel<C: LibretroCorePort + Clone> {
    template: Adapter<C>,
}

impl<C: LibretroCorePort + Clone> SnapshotForkModel<C> {
    pub fn spec(&self) -> &AdapterSpec {
        self.template.spec()
    }
}

impl<C: LibretroCorePort + Clone> GenerativeModel for SnapshotForkModel<C> {
    fn action_space(&self) -> &ActionSpace {
        &self.template.space
    }

    fn step(
        &self,
        state: &State,
        action: &Action,
        dt: StepDt,
        _rng: &mut dyn RngSource,
    ) -> Result<Transition, ModelError> {
        self.template
            .space
            .validate(action)
            .map_err(ModelError::InvalidAction)?;
        let bad = |reason| ModelError::InvalidState { reason };
        let mut env = self
            .template
            .fork_adapter()
            .map_err(|_| bad("fork failed"))?;
        env.decode_state(state)
            .map_err(|_| bad("state rejected by adapter"))?;
        let out = env.step_action(action).map_err(|e| match e {
            AdapterError::AlreadyTerminal => ModelError::TerminalState,
            _ => bad("simulated step failed"),
        })?;
        let enc = env.encode_state().map_err(|_| bad("encode failed"))?;
        let next = state
            .advance(dt, enc.values().to_vec())
            .map_err(ModelError::TransitionFailed)?;
        let features = env
            .session
            .core()
            .ram()
            .iter()
            .map(|&b| f64::from(b))
            .collect();
        let obs = Observation::new(next.tick(), next.elapsed(), features)
            .map_err(ModelError::TransitionFailed)?;
        Transition::new(state.tick(), dt, next, obs, out.terminal)
            .map_err(ModelError::TransitionFailed)
    }
}

fn ram_prefix(state: &State) -> &[f64] {
    let v = state.values();
    let n = v.first().map_or(0, |&n| n as usize);
    v.get(1..1 + n).unwrap_or(&[])
}

/// Planner projection `phi`: the RAM bytes of a planning state.
pub fn planning_features(state: &State) -> Vec<f64> {
    ram_prefix(state).to_vec()
}

/// Reward v1 for planners: score delta between two planning states
/// (identical to what [`Adapter::step`] reports).
pub fn planning_reward(prev: &State, _action: &Action, tr: &Transition) -> f64 {
    let s = |st: &State| {
        let r = ram_prefix(st);
        match (r.get(SCORE_OFFSET), r.get(SCORE_OFFSET + 1)) {
            (Some(lo), Some(hi)) => (*lo as u16) | ((*hi as u16) << 8),
            _ => 0,
        }
    };
    f64::from(s(tr.next_state()).wrapping_sub(s(prev)))
}
