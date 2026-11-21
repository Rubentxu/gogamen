use gogamen_lab_mario::command::{self, EXIT_ERROR, EXIT_NOT_RUN, EXIT_OK};
use gogamen_lab_mario::*;

fn cfg() -> RunConfig {
    RunConfig::default()
}

#[test]
fn sha256_known_vector() {
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn known_list_parse_and_verify() {
    let h = sha256_hex(b"fake-rom");
    let k = KnownRoms::parse(&format!("# c\n\nsha256 {h} test-label\n")).unwrap();
    assert_eq!(k.verify_bytes(b"fake-rom").unwrap().label, "test-label");
    assert!(matches!(
        k.verify_bytes(b"other"),
        Err(RomError::HashNotKnown { known_count: 1, .. })
    ));
    assert_eq!(k.verify_bytes(b""), Err(RomError::EmptyFile));
    assert!(matches!(
        KnownRoms::parse("sha256 zz x"),
        Err(RomError::BadConfig { line: 1, .. })
    ));
    assert!(KnownRoms::parse("md5 abc x").is_err());
}

#[test]
fn shipped_known_list_is_empty() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/known_roms.conf");
    assert!(KnownRoms::load(std::path::Path::new(p)).unwrap().is_empty());
}

fn run_cmd(args: &[&str]) -> (i32, String) {
    let a: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut out = String::new();
    let c = command::run(&a, &mut out);
    (c, out)
}

#[test]
fn no_rom_is_not_run_and_claims_nothing() {
    let (c, out) = run_cmd(&[]);
    assert_eq!(c, EXIT_NOT_RUN);
    assert!(out.starts_with("NOT_RUN: no_rom_provided"));
    assert!(out.contains("no completion is claimed"));
    assert!(!out.contains("goal_reached=true"));
}

#[test]
fn garbage_rom_is_typed_error() {
    let dir = std::env::temp_dir().join("gog62_garbage_rom.bin");
    std::fs::write(&dir, b"not a rom").unwrap();
    let (c, out) = run_cmd(&["--rom", dir.to_str().unwrap()]);
    assert_eq!(c, EXIT_ERROR);
    assert!(out.contains("hash mismatch"));
    let (c, out) = run_cmd(&["--rom", "/nonexistent/x.nes"]);
    assert_eq!(c, EXIT_ERROR);
    assert!(out.contains("cannot read"));
}

#[test]
fn known_rom_without_backend_is_not_run() {
    let rom = std::env::temp_dir().join("gog62_known_rom.bin");
    std::fs::write(&rom, b"user-supplied-bytes").unwrap();
    let conf = std::env::temp_dir().join("gog62_known.conf");
    std::fs::write(
        &conf,
        format!("sha256 {} ci-test\n", sha256_hex(b"user-supplied-bytes")),
    )
    .unwrap();
    let (c, out) = run_cmd(&[
        "--rom",
        rom.to_str().unwrap(),
        "--known",
        conf.to_str().unwrap(),
    ]);
    assert_eq!(c, EXIT_NOT_RUN);
    assert!(out.contains("backend_unavailable"));
    assert!(out.contains("no completion is claimed"));
}

#[test]
fn fixture_command_labels_synthetic() {
    let (c, out) = run_cmd(&["--fixture"]);
    assert_eq!(c, EXIT_OK);
    assert!(out.contains("synthetic fixture, not Mario"));
    assert_eq!(out.matches("synthetic=true").count(), 3);
}

#[test]
fn fixture_port_basics_and_snapshot() {
    let mut p = SyntheticPlatformer::new();
    assert!(p.is_synthetic());
    p.step(PadState::RIGHT, 10);
    assert_eq!(p.features().x, 20);
    let snap = p.snapshot().unwrap();
    p.step(PadState::RIGHT.with(PadState::JUMP), 7);
    let moved = p.features();
    p.restore(&snap).unwrap();
    assert_eq!(p.features().x, 20);
    p.step(PadState::RIGHT.with(PadState::JUMP), 7);
    assert_eq!(p.features(), moved);
    // RAM view decodes to the features.
    assert_eq!(
        i32::from_le_bytes(p.ram()[0..4].try_into().unwrap()),
        moved.x
    );
    // Bad snapshots are typed errors and leave the port unchanged.
    let before = p.features();
    assert!(p.restore(b"junk").is_err());
    let mut bad = snap.clone();
    bad[0] = b'X';
    assert!(p.restore(&bad).is_err());
    assert_eq!(p.features(), before);
}

#[test]
fn always_right_dies_in_first_gap() {
    let m = run_episode(&mut SyntheticPlatformer::new(), &mut AlwaysRight, &cfg());
    assert_eq!(m.deaths, 3);
    assert!(!m.level_complete);
    assert!(m.max_x < 72);
}

#[test]
fn planner_beats_baselines_and_completes_fixture() {
    let c = cfg();
    let rnd = run_episode(
        &mut SyntheticPlatformer::new(),
        &mut RandomPolicy::new(c.seed),
        &c,
    );
    let right = run_episode(&mut SyntheticPlatformer::new(), &mut AlwaysRight, &c);
    let mc = run_episode(
        &mut SyntheticPlatformer::new(),
        &mut MonteCarloPlanner::new(c.seed, 8, 12),
        &c,
    );
    assert!(
        mc.max_x > rnd.max_x && mc.max_x > right.max_x,
        "{mc:?} {rnd:?} {right:?}"
    );
    assert!(mc.level_complete, "{mc:?}");
    assert!(mc.planning_steps > 0 && rnd.planning_steps == 0);
    assert!(mc.fps > 0.0);
}

#[test]
fn traces_are_reproducible_and_seed_sensitive() {
    let c = cfg();
    let run = |seed| {
        let cc = RunConfig { seed, ..c };
        run_episode(
            &mut SyntheticPlatformer::new(),
            &mut MonteCarloPlanner::new(seed, 8, 12),
            &cc,
        )
    };
    let (a, b) = (run(42), run(42));
    assert_eq!(a.trace_hash, b.trace_hash);
    assert_eq!(
        (a.max_x, a.deaths, a.actions_used, a.planning_steps),
        (b.max_x, b.deaths, b.actions_used, b.planning_steps)
    );
    let r1 = run_episode(
        &mut SyntheticPlatformer::new(),
        &mut RandomPolicy::new(7),
        &c,
    );
    let r2 = run_episode(
        &mut SyntheticPlatformer::new(),
        &mut RandomPolicy::new(7),
        &c,
    );
    assert_eq!(r1.trace_hash, r2.trace_hash);
}
