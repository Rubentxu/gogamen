use gogamen_lab_retro::*;

/// Board sweep (so coins get collected) with occasional varied buttons.
fn input(i: u64) -> u16 {
    let base = if (i % 33).eq(&32) {
        buttons::DOWN
    } else {
        buttons::RIGHT
    };
    let noise = (i.wrapping_mul(2654435761) >> 9) as u16;
    base | if i.is_multiple_of(97) { buttons::UP } else { 0 } | (noise & buttons::A)
}

fn run(s: &mut Session<FreePlayCore>, from: u64, to: u64) {
    for i in from..to {
        s.step(input(i));
    }
}

fn observe(s: &Session<FreePlayCore>) -> (u64, Vec<u8>, u64) {
    (s.frame_hash(), s.core().ram().to_vec(), s.frame())
}

#[test]
fn restore_and_replay_reproduces_frame_and_ram_from_several_points() {
    let mut reference = Session::new(FreePlayCore::new());
    let points = [0u64, 1, 777, 5_000, 20_000, 49_999];
    let mut snaps = vec![];
    for i in 0..50_000u64 {
        if points.contains(&i) {
            snaps.push((i, reference.snapshot()));
        }
        reference.step(input(i));
    }
    let want = observe(&reference);
    assert!(
        reference.core().score() > 0,
        "sequence should exercise the game"
    );
    for (at, snap) in snaps {
        let mut s = Session::new(FreePlayCore::new());
        run(&mut s, 0, 123); // unrelated progress first
        s.restore(&snap).unwrap();
        run(&mut s, at, 50_000);
        assert_eq!(observe(&s), want, "snapshot at {at}");
    }
}

#[test]
fn snapshot_carries_emulator_and_wrapper_state() {
    let mut s = Session::new(FreePlayCore::new());
    run(&mut s, 0, 300);
    let snap = s.snapshot();
    let saved = observe(&s);
    let hist = *s.history();
    assert_eq!(hist[HISTORY_LEN - 1], input(299));
    run(&mut s, 300, 900); // diverge
    assert_ne!(observe(&s), saved);
    s.restore(&snap).unwrap();
    assert_eq!(observe(&s), saved);
    assert_eq!(s.frame(), 300);
    assert_eq!(*s.history(), hist);
    // Core-only state would not restore the wrapper frame counter.
    let mut core_only = Session::new(FreePlayCore::new());
    core_only.step(0);
    assert_ne!(core_only.frame(), 300);
}

#[test]
fn reset_returns_to_power_on() {
    let fresh = observe(&Session::new(FreePlayCore::new()));
    let mut s = Session::new(FreePlayCore::new());
    run(&mut s, 0, 500);
    s.reset();
    assert_eq!(observe(&s), fresh);
}

#[test]
fn unserialize_rejects_bad_data_without_panic_or_mutation() {
    let mut s = Session::new(FreePlayCore::new());
    run(&mut s, 0, 100);
    let snap = s.snapshot();
    let before = observe(&s);
    // Every truncation fails.
    for n in 0..snap.len() {
        assert!(s.restore(&snap[..n]).is_err(), "len {n}");
    }
    // Every single-byte corruption fails (checksum / magic / length).
    for i in 0..snap.len() {
        let mut bad = snap.clone();
        bad[i] ^= 0xff;
        assert!(s.restore(&bad).is_err(), "byte {i}");
    }
    let mut extended = snap.clone();
    extended.push(0);
    assert!(s.restore(&extended).is_err());
    assert_eq!(observe(&s), before, "failed restores must not mutate");

    // Raw core port: truncated, oversized, bad magic, out-of-range coords.
    let mut c = FreePlayCore::new();
    let good = c.serialize_state();
    assert_eq!(
        c.unserialize_state(&good[..good.len() - 1]),
        Err(StateError::Truncated)
    );
    assert_eq!(c.unserialize_state(&[]), Err(StateError::Truncated));
    let mut long = good.clone();
    long.push(0);
    assert!(c.unserialize_state(&long).is_err());
    let mut magic = good.clone();
    magic[0] = 0;
    assert_eq!(c.unserialize_state(&magic), Err(StateError::BadMagic));
    let mut coords = good.clone();
    coords[4] = 200; // player x out of range
    assert!(c.unserialize_state(&coords).is_err());
    assert!(c.unserialize_state(&good).is_ok());
}
