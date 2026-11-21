use std::cell::Cell;
use std::rc::Rc;

use gogamen_core::{
    Action, Actuator, BranchableSnapshot, DeterministicRng, EnvironmentObserver, FmcPlanner,
    FmcPlannerConfig, FmcScoringConfig, GenerativeModel, LiveEnvironment, MasterSeed, PlanBudget,
    PlanRequest, PlanStatus, PlanningContext, Terminal, TerminalReason,
};
use gogamen_lab_retro::*;

fn cfg() -> AdapterConfig {
    AdapterConfig {
        frame_skip: 2,
        goal_score: 50,
        max_steps: 40,
        ..AdapterConfig::default()
    }
}

fn adapter() -> Adapter<FreePlayCore> {
    Adapter::new(FreePlayCore::new(), cfg()).unwrap()
}

fn script(i: u64) -> usize {
    ((i * 7 + 3) % 9) as usize
}

#[test]
fn step_matches_manual_frames_with_frame_skip_and_is_deterministic() {
    let mut a = adapter();
    let mut manual = Session::new(FreePlayCore::new());
    manual.reset();
    for i in 0..30 {
        let out = a.step(script(i)).unwrap();
        assert_eq!(out.frames, 2);
        let mask = [
            0,
            buttons::UP,
            buttons::DOWN,
            buttons::LEFT,
            buttons::RIGHT,
            buttons::UP | buttons::LEFT,
            buttons::UP | buttons::RIGHT,
            buttons::DOWN | buttons::LEFT,
            buttons::DOWN | buttons::RIGHT,
        ][script(i)];
        manual.step(mask);
        manual.step(mask);
        assert_eq!(a.frame_hash(), manual.frame_hash());
    }
    assert_eq!(a.steps(), 30);
    let mut b = adapter();
    for i in 0..30 {
        b.step(script(i)).unwrap();
    }
    assert_eq!(a.snapshot_bytes(), b.snapshot_bytes());
}

#[test]
fn reset_contract_and_terminal() {
    let mut a = adapter();
    let fresh = a.snapshot_bytes();
    for i in 0..40 {
        a.step(script(i)).unwrap();
    }
    assert_eq!(
        a.terminal(),
        Terminal::Terminated(TerminalReason::HorizonReached)
    );
    assert!(matches!(a.step(0), Err(AdapterError::AlreadyTerminal)));
    let obs = LiveEnvironment::reset(&mut a).unwrap();
    assert_eq!(obs.tick().as_u64(), 0);
    assert_eq!(a.snapshot_bytes(), fresh);
    assert!(a.terminal().is_ongoing());
    assert!(matches!(a.step(99), Err(AdapterError::InvalidAction(_))));
}

#[test]
fn goal_terminal_and_reward_agree_with_score() {
    let mut a = Adapter::new(
        FreePlayCore::new(),
        AdapterConfig {
            goal_score: 1,
            max_steps: 100_000,
            ..AdapterConfig::default()
        },
    )
    .unwrap();
    let mut total = 0.0;
    for i in 0..20_000u64 {
        // sweep the board so a coin is collected
        let out = a.step(if i % 33 == 32 { 2 } else { 4 }).unwrap();
        total += out.reward;
        if out.terminal.is_terminal() {
            assert_eq!(
                out.terminal,
                Terminal::Terminated(TerminalReason::GoalReached)
            );
            assert_eq!(total, 1.0);
            return;
        }
    }
    panic!("goal never reached");
}

#[test]
fn live_environment_traits_work() {
    let mut a = adapter();
    let space = a.action_space().clone();
    let act = Action::discrete_in_space(4, &space).unwrap();
    let out = a.apply(&act).unwrap();
    assert_eq!(out.observation(), &a.observe().unwrap());
    assert_eq!(out.observation().dimension(), RAM_SIZE);
    assert_eq!(a.framebuffer().len(), FB_WIDTH * FB_HEIGHT);
}

#[test]
fn fork_is_independent_and_snapshot_restore_roundtrips() {
    let mut a = adapter();
    for i in 0..10 {
        a.step(script(i)).unwrap();
    }
    let snap = BranchableSnapshot::snapshot(&a).unwrap();
    let mut f = a.fork().unwrap();
    assert_eq!(f.snapshot_bytes(), a.snapshot_bytes());
    let before = a.snapshot_bytes();
    for i in 0..10 {
        f.step(script(i + 100)).unwrap();
    }
    assert_eq!(
        a.snapshot_bytes(),
        before,
        "stepping the fork must not touch the parent"
    );
    assert_ne!(f.snapshot_bytes(), before);
    // restore the fork from the parent's snapshot, then both continue identically
    f.restore(&snap).unwrap();
    a.step(3).unwrap();
    f.step(3).unwrap();
    assert_eq!(a.snapshot_bytes(), f.snapshot_bytes());
}

/// Core that can be told to fail unserialize after mutating itself.
#[derive(Clone)]
struct FaultyCore {
    inner: FreePlayCore,
    fail: Rc<Cell<bool>>,
}

impl LibretroCorePort for FaultyCore {
    fn frame_size(&self) -> (usize, usize) {
        self.inner.frame_size()
    }
    fn framebuffer(&self) -> &[u8] {
        self.inner.framebuffer()
    }
    fn ram(&self) -> &[u8] {
        self.inner.ram()
    }
    fn run_frame(&mut self, m: u16) {
        self.inner.run_frame(m)
    }
    fn reset(&mut self) {
        self.inner.reset()
    }
    fn serialize_state(&self) -> Vec<u8> {
        self.inner.serialize_state()
    }
    fn unserialize_state(&mut self, data: &[u8]) -> Result<(), StateError> {
        if self.fail.get() {
            // misbehaving core: corrupts itself, then reports an error
            self.inner.run_frame(buttons::RIGHT);
            return Err(StateError::BadChecksum);
        }
        self.inner.unserialize_state(data)
    }
}

#[test]
fn unrecoverable_core_failure_is_typed_and_malformed_input_changes_nothing() {
    let fail = Rc::new(Cell::new(false));
    let core = FaultyCore {
        inner: FreePlayCore::new(),
        fail: fail.clone(),
    };
    let mut a = Adapter::new(core, cfg()).unwrap();
    for i in 0..5 {
        a.step(script(i)).unwrap();
    }
    let snap = a.snapshot_bytes();
    // Malformed payloads are rejected before the core is touched.
    let before = a.snapshot_bytes();
    let mut flipped = snap.clone();
    let n = flipped.len() / 2;
    flipped[n] ^= 0xff;
    for bad in [&b"x"[..], &[0u8; 80][..], &flipped[..]] {
        assert!(matches!(
            a.restore_bytes(bad),
            Err(AdapterError::BadSnapshot(_))
        ));
        assert_eq!(a.snapshot_bytes(), before);
    }
    // A core that fails every unserialize (including the rollback) yields a typed
    // error, not a panic.
    fail.set(true);
    assert_eq!(
        a.restore_bytes(&snap),
        Err(AdapterError::RollbackFailed(StateError::BadChecksum))
    );
}

#[test]
fn core_failure_with_working_rollback_restores_state() {
    // Core fails only the first unserialize (the requested one) after mutating.
    #[derive(Clone)]
    struct OnceFaulty {
        inner: FreePlayCore,
        armed: Rc<Cell<bool>>,
    }
    impl LibretroCorePort for OnceFaulty {
        fn frame_size(&self) -> (usize, usize) {
            self.inner.frame_size()
        }
        fn framebuffer(&self) -> &[u8] {
            self.inner.framebuffer()
        }
        fn ram(&self) -> &[u8] {
            self.inner.ram()
        }
        fn run_frame(&mut self, m: u16) {
            self.inner.run_frame(m)
        }
        fn reset(&mut self) {
            self.inner.reset()
        }
        fn serialize_state(&self) -> Vec<u8> {
            self.inner.serialize_state()
        }
        fn unserialize_state(&mut self, d: &[u8]) -> Result<(), StateError> {
            if self.armed.replace(false) {
                self.inner.run_frame(buttons::DOWN);
                return Err(StateError::BadLength);
            }
            self.inner.unserialize_state(d)
        }
    }
    let armed = Rc::new(Cell::new(false));
    let mut a = Adapter::new(
        OnceFaulty {
            inner: FreePlayCore::new(),
            armed: armed.clone(),
        },
        cfg(),
    )
    .unwrap();
    for i in 0..4 {
        a.step(script(i)).unwrap();
    }
    let snap = a.snapshot_bytes();
    for i in 4..8 {
        a.step(script(i)).unwrap();
    }
    let before = a.snapshot_bytes();
    armed.set(true);
    assert_eq!(
        a.restore_bytes(&snap),
        Err(AdapterError::Core(StateError::BadLength))
    );
    assert_eq!(
        a.snapshot_bytes(),
        before,
        "adapter must be unchanged and usable"
    );
    a.step(1).unwrap();
}

#[test]
fn spec_version_mismatch_is_rejected_in_snapshots() {
    let mut a = adapter();
    a.step(1).unwrap();
    let snap = a.snapshot_bytes();
    let variants = [
        AdapterConfig {
            frame_skip: 3,
            ..cfg()
        },
        AdapterConfig {
            goal_score: 51,
            ..cfg()
        },
        AdapterConfig {
            max_steps: 41,
            ..cfg()
        },
    ];
    let mut hashes = vec![a.spec().hash()];
    for v in variants {
        let mut other = Adapter::new(FreePlayCore::new(), v).unwrap();
        hashes.push(other.spec().hash());
        let before = other.snapshot_bytes();
        assert!(matches!(
            other.restore_bytes(&snap),
            Err(AdapterError::SpecMismatch { .. })
        ));
        assert_eq!(other.snapshot_bytes(), before);
    }
    hashes.sort();
    hashes.dedup();
    assert_eq!(hashes.len(), 4, "every config field changes the spec hash");
    // unsupported function versions are refused at construction
    for bad in [
        AdapterConfig {
            reward_fn_version: 2,
            ..cfg()
        },
        AdapterConfig {
            terminal_fn_version: 0,
            ..cfg()
        },
        AdapterConfig {
            frame_skip: 0,
            ..cfg()
        },
    ] {
        assert!(matches!(
            Adapter::new(FreePlayCore::new(), bad),
            Err(AdapterError::Config(_))
        ));
    }
    // same spec restores fine; core-trait snapshots carry the check too
    let mut same = adapter();
    same.restore_bytes(&snap).unwrap();
    let csnap = BranchableSnapshot::snapshot(&a).unwrap();
    let mut other = Adapter::new(
        FreePlayCore::new(),
        AdapterConfig {
            frame_skip: 3,
            ..cfg()
        },
    )
    .unwrap();
    assert!(other.restore(&csnap).is_err());
}

/// Counts core activity. `own_*` is per instance (a fork starts at zero);
/// `branch_frames` is shared by all clones.
struct CountingCore {
    inner: FreePlayCore,
    own_frames: Rc<Cell<u32>>,
    own_resets: Rc<Cell<u32>>,
    branch_frames: Rc<Cell<u32>>,
    clones: Rc<Cell<u32>>,
}

impl Clone for CountingCore {
    fn clone(&self) -> Self {
        self.clones.set(self.clones.get() + 1);
        Self {
            inner: self.inner.clone(),
            own_frames: Rc::new(Cell::new(0)),
            own_resets: Rc::new(Cell::new(0)),
            branch_frames: self.branch_frames.clone(),
            clones: self.clones.clone(),
        }
    }
}

impl LibretroCorePort for CountingCore {
    fn frame_size(&self) -> (usize, usize) {
        self.inner.frame_size()
    }
    fn framebuffer(&self) -> &[u8] {
        self.inner.framebuffer()
    }
    fn ram(&self) -> &[u8] {
        self.inner.ram()
    }
    fn run_frame(&mut self, m: u16) {
        self.own_frames.set(self.own_frames.get() + 1);
        self.branch_frames.set(self.branch_frames.get() + 1);
        self.inner.run_frame(m)
    }
    fn reset(&mut self) {
        self.own_resets.set(self.own_resets.get() + 1);
        self.inner.reset()
    }
    fn serialize_state(&self) -> Vec<u8> {
        self.inner.serialize_state()
    }
    fn unserialize_state(&mut self, d: &[u8]) -> Result<(), StateError> {
        self.inner.unserialize_state(d)
    }
}

#[test]
fn fmc_controls_via_declared_snapshot_fork_access_only() {
    let own_frames = Rc::new(Cell::new(0));
    let own_resets = Rc::new(Cell::new(0));
    let branch_frames = Rc::new(Cell::new(0));
    let clones = Rc::new(Cell::new(0));
    let core = CountingCore {
        inner: FreePlayCore::new(),
        own_frames: own_frames.clone(),
        own_resets: own_resets.clone(),
        branch_frames: branch_frames.clone(),
        clones: clones.clone(),
    };
    let mut live = Adapter::new(core, cfg()).unwrap();
    assert_eq!(live.access_mode(), AccessMode::SnapshotForkable);
    for i in 0..3 {
        live.step(script(i)).unwrap();
    }
    let (frames0, resets0) = (own_frames.get(), own_resets.get());
    let live_before = live.snapshot_bytes();

    let model = live.planning_model().unwrap();
    let start = live.planning_state().unwrap();
    let dt = live.step_dt().unwrap();
    let planner_cfg =
        FmcPlannerConfig::new(6, 3, dt, FmcScoringConfig::default(), MasterSeed::new(11)).unwrap();
    let planner = FmcPlanner::new(planner_cfg);
    let space = model.action_space().clone();
    let request = PlanRequest::from_state(start.clone(), space, PlanBudget::with_transitions(200));
    let mut rng = DeterministicRng::new(5);
    let mut ctx = PlanningContext::new(&model, &mut rng);
    let (result, trace) = planner
        .plan_with_trace(&mut ctx, &request, planning_reward, planning_features)
        .unwrap();
    assert_eq!(result.status(), PlanStatus::Completed);
    let chosen = result.authorized_action().expect("an action").clone();
    assert!(trace.rollout_transitions_used() > 0);

    // Planning went through forks only: branches ran frames, the live core did not.
    assert!(branch_frames.get() > 0 && clones.get() > 0);
    assert_eq!(
        own_frames.get(),
        frames0,
        "live core must not run frames while planning"
    );
    assert_eq!(
        own_resets.get(),
        resets0,
        "live core must not be reset while planning"
    );
    assert_eq!(live.snapshot_bytes(), live_before, "live adapter untouched");

    // Deterministic: same seeds, same decision.
    let model2 = live.planning_model().unwrap();
    let mut rng2 = DeterministicRng::new(5);
    let mut ctx2 = PlanningContext::new(&model2, &mut rng2);
    let (result2, _) = FmcPlanner::new(planner.config().clone())
        .plan_with_trace(&mut ctx2, &request, planning_reward, planning_features)
        .unwrap();
    assert_eq!(result2.authorized_action(), Some(&chosen));

    // The real agent loop then acts on the live adapter with the chosen action.
    let out = live.apply(&chosen).unwrap();
    assert_eq!(live.steps(), 4);
    assert_eq!(out.observation().tick().as_u64(), 4);
    // The model's simulated step agrees with the live step (same state, reward).
    let mut r = DeterministicRng::new(0);
    let tr = model.step(&start, &chosen, dt, &mut r).unwrap();
    assert_eq!(
        tr.next_state().values(),
        live.planning_state().unwrap().values()
    );
    assert_eq!(planning_reward(&start, &chosen, &tr), live.last_reward());
}

#[test]
fn model_rejects_foreign_or_corrupt_planning_states() {
    let a = adapter();
    let model = a.planning_model().unwrap();
    let start = a.planning_state().unwrap();
    let dt = a.step_dt().unwrap();
    let mut rng = DeterministicRng::new(1);
    let act = Action::discrete_in_space(0, model.action_space()).unwrap();
    // state from a different spec (frame_skip 3)
    let other = Adapter::new(
        FreePlayCore::new(),
        AdapterConfig {
            frame_skip: 3,
            ..cfg()
        },
    )
    .unwrap();
    assert!(
        model
            .step(&other.planning_state().unwrap(), &act, dt, &mut rng)
            .is_err()
    );
    // corrupted byte inside the snapshot payload
    let mut vals = start.values().to_vec();
    let n = vals.len();
    vals[n - 20] = (vals[n - 20] + 1.0) % 256.0;
    let bad = gogamen_core::State::new(start.tick(), start.elapsed(), vals).unwrap();
    assert!(model.step(&bad, &act, dt, &mut rng).is_err());
}
