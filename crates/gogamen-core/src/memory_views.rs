//! Memory views: private augmented state and ambient (external) surface (GOG-170 / DEM-017).
//!
//! Two views, one contract:
//! - **Private memory** is the agent's *augmented state*: it extends what the agent predicts
//!   from, but is never part of the live physical world state. It is backed by the existing
//!   [`MemoryPort`] (no second memory implementation); [`PrivateMemoryView`] only adds the
//!   owner guard, a write quota and the derivation of an [`AugmentedState`].
//! - **Ambient memory** is an external surface written into the world state, observable by
//!   other agents, and only where an explicit [`SurfaceGrant`] exists ([`AmbientSurface`]).
//!
//! No type here grants permissions on its own: grants are plain data supplied by the caller
//! (e.g. compiled from a declarative spec) and are checked on every access.
//! Everything is deterministic (`std`-only, FNV-1a hashes, `BTreeMap` ordering).

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crate::clock::MonotonicInstant;
use crate::domain::State;
use crate::error::DomainError;
use crate::memory::{EpisodeId, EpisodeInput, EpisodicRecord, MemoryError, MemoryPort, OwnerId};
use crate::observability::EntityId;
use crate::provenance::EvidenceRef;
use crate::units::Tick;

const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

fn fnv(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn canon_bits(v: f64) -> u64 {
    if v == 0.0 {
        0.0f64.to_bits()
    } else {
        v.to_bits()
    }
}

/// Number of memory summary dimensions appended to the physical state by [`AugmentedState`].
pub const MEMORY_SUMMARY_DIM: usize = 2;

/// Identifier of an ambient surface (an authorized region of the world state).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SurfaceId(u64);

impl SurfaceId {
    /// Creates a [`SurfaceId`].
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    /// Returns the numeric value.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SurfaceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "surface#{}", self.0)
    }
}

/// Kind of access to an ambient surface (mirrors read/write of the declarative permissions).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SurfaceAccess {
    /// Observe the surface.
    Read,
    /// Write the surface (implies observing it).
    Write,
}

/// Explicit authorization of `agent` to access `surface`. Plain data: it grants nothing
/// until placed in an [`AmbientSurface`] by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SurfaceGrant {
    /// Agent that is authorized.
    pub agent: OwnerId,
    /// Surface the grant applies to.
    pub surface: SurfaceId,
    /// Access level.
    pub access: SurfaceAccess,
}

/// Typed errors of the memory views.
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryViewError {
    /// `requester` tried to read or write the private memory of `owner`.
    PrivateAccessDenied {
        /// Owner of the private memory.
        owner: OwnerId,
        /// Agent that attempted the access.
        requester: OwnerId,
    },
    /// The agent has no write grant for the surface.
    SurfaceWriteDenied {
        /// Agent that attempted the write.
        agent: OwnerId,
        /// Target surface.
        surface: SurfaceId,
    },
    /// The agent has no read grant for the surface.
    SurfaceReadDenied {
        /// Agent that attempted the read.
        agent: OwnerId,
        /// Target surface.
        surface: SurfaceId,
    },
    /// A capacity limit would be exceeded; nothing was mutated.
    CapacityExceeded {
        /// Which limit.
        what: &'static str,
        /// The configured limit.
        limit: usize,
    },
    /// Invalid configuration (zero capacity, ...).
    InvalidConfig {
        /// Reason.
        reason: &'static str,
    },
    /// Non-finite value written to a surface.
    NonFiniteValue,
    /// Error forwarded from the underlying [`MemoryPort`].
    Memory(MemoryError),
    /// Error building the extended state.
    Domain(DomainError),
}

impl fmt::Display for MemoryViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PrivateAccessDenied { owner, requester } => write!(
                f,
                "private memory of {owner} is not accessible to {requester}"
            ),
            Self::SurfaceWriteDenied { agent, surface } => {
                write!(f, "agent {agent} has no write grant for {surface}")
            }
            Self::SurfaceReadDenied { agent, surface } => {
                write!(f, "agent {agent} has no read grant for {surface}")
            }
            Self::CapacityExceeded { what, limit } => {
                write!(f, "capacity exceeded: {what} (limit {limit})")
            }
            Self::InvalidConfig { reason } => write!(f, "invalid memory view config: {reason}"),
            Self::NonFiniteValue => write!(f, "surface value must be finite"),
            Self::Memory(e) => write!(f, "memory port error: {e}"),
            Self::Domain(e) => write!(f, "domain error: {e}"),
        }
    }
}

impl Error for MemoryViewError {}

impl From<MemoryError> for MemoryViewError {
    fn from(e: MemoryError) -> Self {
        Self::Memory(e)
    }
}

/// Stable hash of a physical [`State`] (tick, elapsed and value bits).
pub fn physical_state_hash(state: &State) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    fnv(&mut h, &state.tick().as_u64().to_le_bytes());
    fnv(&mut h, &(state.dimension() as u64).to_le_bytes());
    for v in state.values() {
        fnv(&mut h, &canon_bits(*v).to_le_bytes());
    }
    h
}

/// Hash of a sequence of episodic records including provenance (who/when/evidence).
pub fn records_digest(records: &[EpisodicRecord]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    fnv(&mut h, &(records.len() as u64).to_le_bytes());
    for r in records {
        fnv(&mut h, &r.episode_id.as_u64().to_le_bytes());
        fnv(&mut h, &r.owner.as_u64().to_le_bytes());
        fnv(&mut h, &r.entity.as_u64().to_le_bytes());
        fnv(&mut h, &r.tick.as_u64().to_le_bytes());
        fnv(&mut h, &r.observed_at.as_nanos().to_le_bytes());
        fnv(&mut h, &r.revision.to_le_bytes());
        fnv(&mut h, &r.provenance.evidence_ref.as_u64().to_le_bytes());
        fnv(&mut h, &r.provenance.message_id.as_u64().to_le_bytes());
        fnv(&mut h, &r.provenance.sensor_id.as_u64().to_le_bytes());
        fnv(&mut h, &r.summary_code.as_u64().to_le_bytes());
        fnv(&mut h, &(r.numeric_features.len() as u64).to_le_bytes());
        for v in &r.numeric_features {
            fnv(&mut h, &canon_bits(*v).to_le_bytes());
        }
    }
    h
}

/// Augmented state: the physical state the agent sees plus a summary of its private memory.
///
/// It is a *derived value*; building it never mutates the physical state or the memory.
/// Memory summary (fixed, [`MEMORY_SUMMARY_DIM`] dims): `[episode_count, sum of first numeric
/// feature of each episode]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AugmentedState {
    physical: State,
    physical_hash: u64,
    memory_summary: [f64; MEMORY_SUMMARY_DIM],
    memory_digest: u64,
}

impl AugmentedState {
    /// The untouched physical state.
    pub fn physical(&self) -> &State {
        &self.physical
    }
    /// Hash of the physical state at derivation time.
    pub fn physical_hash(&self) -> u64 {
        self.physical_hash
    }
    /// Memory summary dims.
    pub fn memory_summary(&self) -> [f64; MEMORY_SUMMARY_DIM] {
        self.memory_summary
    }
    /// Digest of the contributing private records (including provenance).
    pub fn memory_digest(&self) -> u64 {
        self.memory_digest
    }
    /// Combined hash of physical state and private memory.
    pub fn augmented_hash(&self) -> u64 {
        let mut h = FNV_OFFSET_BASIS;
        fnv(&mut h, &self.physical_hash.to_le_bytes());
        fnv(&mut h, &self.memory_digest.to_le_bytes());
        h
    }
    /// Extended [`State`] `physical values ++ memory summary`, for a model whose state space
    /// is augmented. The physical state itself is not modified.
    pub fn to_extended_state(&self) -> Result<State, DomainError> {
        let mut values = self.physical.values().to_vec();
        values.extend_from_slice(&self.memory_summary);
        State::new(self.physical.tick(), self.physical.elapsed(), values)
    }
}

/// Guarded view of one agent's private memory over a [`MemoryPort`].
///
/// The view holds no memory itself; the port is passed per call. Every call carries the
/// `requester`, which must be the owner, otherwise [`MemoryViewError::PrivateAccessDenied`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrivateMemoryView {
    owner: OwnerId,
    max_episodes: usize,
}

impl PrivateMemoryView {
    /// Creates a view for `owner` allowing at most `max_episodes` retained episodes.
    pub fn new(owner: OwnerId, max_episodes: usize) -> Result<Self, MemoryViewError> {
        if max_episodes == 0 {
            return Err(MemoryViewError::InvalidConfig {
                reason: "max_episodes must be >= 1",
            });
        }
        Ok(Self {
            owner,
            max_episodes,
        })
    }

    /// Owner of this private memory.
    pub const fn owner(&self) -> OwnerId {
        self.owner
    }

    fn guard(&self, requester: OwnerId) -> Result<(), MemoryViewError> {
        if requester != self.owner {
            return Err(MemoryViewError::PrivateAccessDenied {
                owner: self.owner,
                requester,
            });
        }
        Ok(())
    }

    /// Writes an episode (provenance travels in the input). Rejects foreign requesters,
    /// inputs for another owner, and writes beyond the quota (no mutation on error).
    pub fn write<M: MemoryPort + ?Sized>(
        &self,
        memory: &mut M,
        requester: OwnerId,
        input: EpisodeInput,
    ) -> Result<EpisodeId, MemoryViewError> {
        self.guard(requester)?;
        self.guard(input.owner)?;
        if memory.len_for_owner(self.owner) >= self.max_episodes {
            return Err(MemoryViewError::CapacityExceeded {
                what: "private episodes",
                limit: self.max_episodes,
            });
        }
        Ok(memory.record_episode(input)?)
    }

    /// Reads own episodes for `entity` in `[start, end]`.
    pub fn read<M: MemoryPort + ?Sized>(
        &self,
        memory: &M,
        requester: OwnerId,
        entity: EntityId,
        start: MonotonicInstant,
        end: MonotonicInstant,
    ) -> Result<Vec<EpisodicRecord>, MemoryViewError> {
        self.guard(requester)?;
        Ok(memory.query_by_time_and_entity(self.owner, entity, start, end)?)
    }

    /// Derives the [`AugmentedState`] from `physical` and the owner's episodes for
    /// `entities` in `[start, end]`. Read-only on both inputs.
    pub fn augment<M: MemoryPort + ?Sized>(
        &self,
        memory: &M,
        requester: OwnerId,
        physical: &State,
        entities: &[EntityId],
        start: MonotonicInstant,
        end: MonotonicInstant,
    ) -> Result<AugmentedState, MemoryViewError> {
        self.guard(requester)?;
        let mut all = Vec::new();
        for e in entities {
            all.extend(memory.query_by_time_and_entity(self.owner, *e, start, end)?);
        }
        all.sort_by_key(|r| r.episode_id);
        let sum: f64 = all
            .iter()
            .map(|r| r.numeric_features.first().copied().unwrap_or(0.0))
            .sum();
        Ok(AugmentedState {
            physical: physical.clone(),
            physical_hash: physical_state_hash(physical),
            memory_summary: [all.len() as f64, sum],
            memory_digest: records_digest(&all),
        })
    }
}

/// Provenance of an ambient write: who, when (tick) and on what evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceProvenance {
    /// Writer.
    pub writer: OwnerId,
    /// Simulation tick of the write.
    pub tick: Tick,
    /// Supporting evidence.
    pub evidence: EvidenceRef,
}

/// A value currently on an ambient surface, with provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceCell {
    /// Finite value.
    pub value: f64,
    /// Who/when/why.
    pub provenance: SurfaceProvenance,
    /// Number of writes to this cell (starts at 1).
    pub revision: u64,
}

/// One recorded ambient write (the replay unit).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceWrite {
    /// Target surface.
    pub surface: SurfaceId,
    /// Written value.
    pub value: f64,
    /// Provenance.
    pub provenance: SurfaceProvenance,
}

/// Bounded limits of an [`AmbientSurface`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SurfaceLimits {
    /// Max distinct surfaces holding a value.
    pub max_cells: usize,
    /// Max writes retained in the log (exceeding is an error, never silent eviction).
    pub max_writes: usize,
    /// Cells not rewritten for `>= ttl_ticks` are removed by [`AmbientSurface::expire`].
    pub ttl_ticks: Option<u64>,
}

/// Ambient memory: authorized external surface in the world state, observable by others.
///
/// Holds only data written through [`AmbientSurface::write`]; it never touches the physical
/// [`State`]. Grants are fixed at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbientSurface {
    grants: BTreeSet<SurfaceGrant>,
    limits: SurfaceLimits,
    cells: BTreeMap<SurfaceId, SurfaceCell>,
    log: Vec<SurfaceWrite>,
}

impl AmbientSurface {
    /// Creates an empty surface with explicit grants and limits.
    pub fn new(
        grants: impl IntoIterator<Item = SurfaceGrant>,
        limits: SurfaceLimits,
    ) -> Result<Self, MemoryViewError> {
        if limits.max_cells == 0 || limits.max_writes == 0 || limits.ttl_ticks == Some(0) {
            return Err(MemoryViewError::InvalidConfig {
                reason: "limits must be >= 1",
            });
        }
        Ok(Self {
            grants: grants.into_iter().collect(),
            limits,
            cells: BTreeMap::new(),
            log: Vec::new(),
        })
    }

    fn allows(&self, agent: OwnerId, surface: SurfaceId, access: SurfaceAccess) -> bool {
        match access {
            SurfaceAccess::Write => self.grants.contains(&SurfaceGrant {
                agent,
                surface,
                access: SurfaceAccess::Write,
            }),
            SurfaceAccess::Read => {
                [SurfaceAccess::Read, SurfaceAccess::Write]
                    .into_iter()
                    .any(|a| {
                        self.grants.contains(&SurfaceGrant {
                            agent,
                            surface,
                            access: a,
                        })
                    })
            }
        }
    }

    /// Writes `value` to `surface` on behalf of `writer`. Unauthorized, non-finite or
    /// over-capacity writes are rejected without mutation.
    pub fn write(
        &mut self,
        writer: OwnerId,
        surface: SurfaceId,
        value: f64,
        tick: Tick,
        evidence: EvidenceRef,
    ) -> Result<(), MemoryViewError> {
        if !self.allows(writer, surface, SurfaceAccess::Write) {
            return Err(MemoryViewError::SurfaceWriteDenied {
                agent: writer,
                surface,
            });
        }
        if !value.is_finite() {
            return Err(MemoryViewError::NonFiniteValue);
        }
        if self.log.len() >= self.limits.max_writes {
            return Err(MemoryViewError::CapacityExceeded {
                what: "surface writes",
                limit: self.limits.max_writes,
            });
        }
        if !self.cells.contains_key(&surface) && self.cells.len() >= self.limits.max_cells {
            return Err(MemoryViewError::CapacityExceeded {
                what: "surface cells",
                limit: self.limits.max_cells,
            });
        }
        let provenance = SurfaceProvenance {
            writer,
            tick,
            evidence,
        };
        let revision = self.cells.get(&surface).map_or(1, |c| c.revision + 1);
        self.cells.insert(
            surface,
            SurfaceCell {
                value,
                provenance,
                revision,
            },
        );
        self.log.push(SurfaceWrite {
            surface,
            value,
            provenance,
        });
        Ok(())
    }

    /// Observes `surface` as `reader` (needs a read or write grant).
    pub fn read(
        &self,
        reader: OwnerId,
        surface: SurfaceId,
    ) -> Result<Option<SurfaceCell>, MemoryViewError> {
        if !self.allows(reader, surface, SurfaceAccess::Read) {
            return Err(MemoryViewError::SurfaceReadDenied {
                agent: reader,
                surface,
            });
        }
        Ok(self.cells.get(&surface).copied())
    }

    /// Removes cells whose last write is at least `ttl_ticks` older than `now`; returns the
    /// number removed. The write log is kept (it is the replay record).
    pub fn expire(&mut self, now: Tick) -> usize {
        let Some(ttl) = self.limits.ttl_ticks else {
            return 0;
        };
        let before = self.cells.len();
        self.cells
            .retain(|_, c| now.as_u64().saturating_sub(c.provenance.tick.as_u64()) < ttl);
        before - self.cells.len()
    }

    /// The write log in order.
    pub fn log(&self) -> &[SurfaceWrite] {
        &self.log
    }

    /// Deterministic hash of the current cells (value, provenance, revision).
    pub fn state_hash(&self) -> u64 {
        let mut h = FNV_OFFSET_BASIS;
        fnv(&mut h, &(self.cells.len() as u64).to_le_bytes());
        for (id, c) in &self.cells {
            fnv(&mut h, &id.as_u64().to_le_bytes());
            fnv(&mut h, &canon_bits(c.value).to_le_bytes());
            fnv(&mut h, &c.provenance.writer.as_u64().to_le_bytes());
            fnv(&mut h, &c.provenance.tick.as_u64().to_le_bytes());
            fnv(&mut h, &c.provenance.evidence.as_u64().to_le_bytes());
            fnv(&mut h, &c.revision.to_le_bytes());
        }
        h
    }

    /// Rebuilds a surface by re-applying `writes` (grants and limits are re-checked), so the
    /// same sequence always yields the same [`AmbientSurface::state_hash`].
    pub fn replay(
        grants: impl IntoIterator<Item = SurfaceGrant>,
        limits: SurfaceLimits,
        writes: &[SurfaceWrite],
    ) -> Result<Self, MemoryViewError> {
        let mut s = Self::new(grants, limits)?;
        for w in writes {
            s.write(
                w.provenance.writer,
                w.surface,
                w.value,
                w.provenance.tick,
                w.provenance.evidence,
            )?;
        }
        Ok(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{BoundedEpisodicMemory, EpisodeProvenance, RetentionPolicy, SummaryCode};

    const A: OwnerId = OwnerId::new(1);
    const B: OwnerId = OwnerId::new(2);
    const E: EntityId = EntityId::new(7);

    fn ep(owner: OwnerId, tick: u64, feat: f64) -> EpisodeInput {
        EpisodeInput::new(
            owner,
            E,
            Tick::new(tick),
            MonotonicInstant::from_nanos(u128::from(tick) * 10),
            tick,
            EpisodeProvenance::from_evidence(EvidenceRef::new(100 + tick)),
            SummaryCode::new(1),
            vec![feat],
        )
    }

    fn mem() -> BoundedEpisodicMemory {
        BoundedEpisodicMemory::new(RetentionPolicy::new(16, 32, None).unwrap())
    }

    fn world_state() -> State {
        State::new(
            Tick::new(3),
            crate::units::SimDuration::ZERO,
            vec![1.0, 2.0],
        )
        .unwrap()
    }

    fn range() -> (MonotonicInstant, MonotonicInstant) {
        (MonotonicInstant::ZERO, MonotonicInstant::from_nanos(10_000))
    }

    #[test]
    fn ac1_private_write_changes_augmented_prediction_not_physical_state() {
        let view = PrivateMemoryView::new(A, 8).unwrap();
        let mut m = mem();
        let phys = world_state();
        let (s, e) = range();
        let before = view.augment(&m, A, &phys, &[E], s, e).unwrap();
        let hash_before = physical_state_hash(&phys);

        view.write(&mut m, A, ep(A, 1, 4.0)).unwrap();
        let after = view.augment(&m, A, &phys, &[E], s, e).unwrap();

        assert_eq!(physical_state_hash(&phys), hash_before);
        assert_eq!(before.physical_hash(), after.physical_hash());
        assert_eq!(before.physical(), after.physical());
        assert_ne!(before.augmented_hash(), after.augmented_hash());
        assert_eq!(after.memory_summary(), [1.0, 4.0]);
        // The extended state (what a memory-conditioned model predicts from) differs.
        let (xb, xa) = (
            before.to_extended_state().unwrap(),
            after.to_extended_state().unwrap(),
        );
        assert_eq!(xa.dimension(), phys.dimension() + MEMORY_SUMMARY_DIM);
        assert_ne!(xb.values(), xa.values());
        // Prediction function: next[0] = physical[0] + memory sum.
        let predict = |x: &State| x.values()[0] + x.values()[x.dimension() - 1];
        assert_ne!(predict(&xb), predict(&xa));
    }

    #[test]
    fn ac2_ambient_write_authorized_only_and_observable_by_other_agent() {
        let surf = SurfaceId::new(9);
        let limits = SurfaceLimits {
            max_cells: 2,
            max_writes: 4,
            ttl_ticks: None,
        };
        let mut s = AmbientSurface::new(
            [
                SurfaceGrant {
                    agent: A,
                    surface: surf,
                    access: SurfaceAccess::Write,
                },
                SurfaceGrant {
                    agent: B,
                    surface: surf,
                    access: SurfaceAccess::Read,
                },
            ],
            limits,
        )
        .unwrap();
        let phys = world_state();
        let ph = physical_state_hash(&phys);
        let h0 = s.state_hash();

        // Unauthorized write (reader-only B, and unknown surface) rejected, no mutation.
        assert_eq!(
            s.write(B, surf, 1.0, Tick::new(1), EvidenceRef::new(1)),
            Err(MemoryViewError::SurfaceWriteDenied {
                agent: B,
                surface: surf
            })
        );
        let other = SurfaceId::new(10);
        assert!(matches!(
            s.write(A, other, 1.0, Tick::new(1), EvidenceRef::new(1)),
            Err(MemoryViewError::SurfaceWriteDenied { .. })
        ));
        assert_eq!(s.state_hash(), h0);

        s.write(A, surf, 5.0, Tick::new(2), EvidenceRef::new(77))
            .unwrap();
        assert_ne!(s.state_hash(), h0);
        assert_eq!(physical_state_hash(&phys), ph);
        let seen = s.read(B, surf).unwrap().unwrap();
        assert_eq!(seen.value, 5.0);
        assert_eq!(seen.provenance.writer, A);
        assert_eq!(seen.provenance.tick, Tick::new(2));
        // Agent without any grant cannot observe.
        assert!(matches!(
            s.read(OwnerId::new(3), surf),
            Err(MemoryViewError::SurfaceReadDenied { .. })
        ));
    }

    #[test]
    fn ac3_isolation_capacity_provenance_and_replay() {
        // Isolation: B cannot read/write/augment A's private memory.
        let view = PrivateMemoryView::new(A, 2).unwrap();
        let mut m = mem();
        view.write(&mut m, A, ep(A, 1, 1.0)).unwrap();
        let (s, e) = range();
        let denied = MemoryViewError::PrivateAccessDenied {
            owner: A,
            requester: B,
        };
        assert_eq!(view.read(&m, B, E, s, e), Err(denied.clone()));
        assert_eq!(view.write(&mut m, B, ep(A, 2, 1.0)), Err(denied.clone()));
        assert_eq!(view.augment(&m, B, &world_state(), &[E], s, e), Err(denied));
        // Input addressed to another owner is rejected even from the owner.
        assert!(matches!(
            view.write(&mut m, A, ep(B, 2, 1.0)),
            Err(MemoryViewError::PrivateAccessDenied { .. })
        ));
        assert_eq!(m.len_for_owner(B), 0);

        // Capacity: third write exceeds quota 2, no mutation.
        view.write(&mut m, A, ep(A, 2, 2.0)).unwrap();
        assert_eq!(
            view.write(&mut m, A, ep(A, 3, 3.0)),
            Err(MemoryViewError::CapacityExceeded {
                what: "private episodes",
                limit: 2
            })
        );
        assert_eq!(m.len_for_owner(A), 2);

        // Provenance survives the round trip.
        let recs = view.read(&m, A, E, s, e).unwrap();
        assert_eq!(recs[0].provenance.evidence_ref, EvidenceRef::new(101));

        // Private replay determinism.
        let run = || {
            let mut m2 = mem();
            for t in 1..=2 {
                view.write(&mut m2, A, ep(A, t, t as f64)).unwrap();
            }
            view.augment(&m2, A, &world_state(), &[E], s, e)
                .unwrap()
                .augmented_hash()
        };
        assert_eq!(run(), run());

        // Ambient: capacity errors, TTL retention, replay.
        let surf = SurfaceId::new(1);
        let grants = [
            SurfaceGrant {
                agent: A,
                surface: surf,
                access: SurfaceAccess::Write,
            },
            SurfaceGrant {
                agent: A,
                surface: SurfaceId::new(2),
                access: SurfaceAccess::Write,
            },
            SurfaceGrant {
                agent: A,
                surface: SurfaceId::new(3),
                access: SurfaceAccess::Write,
            },
        ];
        let limits = SurfaceLimits {
            max_cells: 2,
            max_writes: 3,
            ttl_ticks: Some(5),
        };
        let mut a = AmbientSurface::new(grants, limits).unwrap();
        a.write(A, surf, 1.0, Tick::new(1), EvidenceRef::new(1))
            .unwrap();
        a.write(A, SurfaceId::new(2), 2.0, Tick::new(2), EvidenceRef::new(2))
            .unwrap();
        assert_eq!(
            a.write(A, SurfaceId::new(3), 3.0, Tick::new(3), EvidenceRef::new(3)),
            Err(MemoryViewError::CapacityExceeded {
                what: "surface cells",
                limit: 2
            })
        );
        a.write(A, surf, 4.0, Tick::new(4), EvidenceRef::new(4))
            .unwrap();
        assert_eq!(a.read(A, surf).unwrap().unwrap().revision, 2);
        assert_eq!(
            a.write(A, surf, 5.0, Tick::new(5), EvidenceRef::new(5)),
            Err(MemoryViewError::CapacityExceeded {
                what: "surface writes",
                limit: 3
            })
        );
        assert_eq!(
            a.write(A, surf, f64::NAN, Tick::new(5), EvidenceRef::new(5)),
            Err(MemoryViewError::NonFiniteValue)
        );
        let replayed = AmbientSurface::replay(grants, limits, a.log()).unwrap();
        assert_eq!(replayed.state_hash(), a.state_hash());
        assert_eq!(replayed, a);
        // Replay re-checks grants: no grants -> denied.
        assert!(AmbientSurface::replay([], limits, a.log()).is_err());
        // Retention: cell at tick 2 expires at now=7 (ttl 5), cell at tick 4 stays.
        assert_eq!(a.expire(Tick::new(7)), 1);
        assert!(a.read(A, SurfaceId::new(2)).unwrap().is_none());
        assert!(a.read(A, surf).unwrap().is_some());
    }
}
