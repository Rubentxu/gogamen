//! Logic of the `mario_run` example (kept in the lib so it is testable).

use crate::fixture::{SYNTHETIC_LABEL, SyntheticPlatformer};
use crate::metrics::{EpisodeMetrics, RunConfig, run_episode};
use crate::policy::{AlwaysRight, MonteCarloPlanner, RandomPolicy};
use crate::rom::{KnownRoms, NotRunReason, RomError};
use std::fmt::Write;
use std::path::PathBuf;

pub const EXIT_OK: i32 = 0;
pub const EXIT_ERROR: i32 = 2;
/// NOT_RUN is deliberately non-zero so scripts cannot mistake it for a pass.
pub const EXIT_NOT_RUN: i32 = 3;

/// Run the command; appends human output to `out` and returns the exit code.
pub fn run(args: &[String], out: &mut String) -> i32 {
    let mut rom: Option<PathBuf> = None;
    let mut known_path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/known_roms.conf"));
    let mut fixture = false;
    let mut seed = 42u64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--fixture" => fixture = true,
            "--rom" => rom = it.next().map(PathBuf::from),
            "--known" => {
                if let Some(p) = it.next() {
                    known_path = PathBuf::from(p)
                }
            }
            "--seed" => match it.next().and_then(|s| s.parse().ok()) {
                Some(s) => seed = s,
                None => {
                    let _ = writeln!(out, "ERROR: --seed needs an integer");
                    return EXIT_ERROR;
                }
            },
            other => {
                let _ = writeln!(
                    out,
                    "ERROR: unknown argument '{other}'\nusage: mario_run [--rom PATH] [--known FILE] [--fixture] [--seed N]"
                );
                return EXIT_ERROR;
            }
        }
    }
    if fixture {
        return run_fixture(seed, out);
    }
    let Some(rom) = rom else {
        let _ = writeln!(
            out,
            "NOT_RUN: {}\nno level was run; no completion is claimed. Pass --rom <your ROM> or --fixture ({SYNTHETIC_LABEL}).",
            NotRunReason::NoRomProvided
        );
        return EXIT_NOT_RUN;
    };
    let known = match KnownRoms::load(&known_path) {
        Ok(k) => k,
        Err(e) => return error(out, &e),
    };
    match known.verify_file(&rom) {
        Err(e) => error(out, &e),
        Ok(v) => {
            let reason = NotRunReason::BackendUnavailable {
                label: v.label,
                sha256: v.sha256,
            };
            let _ = writeln!(
                out,
                "NOT_RUN: {reason}\nno level was run; no completion is claimed."
            );
            EXIT_NOT_RUN
        }
    }
}

fn error(out: &mut String, e: &RomError) -> i32 {
    let _ = writeln!(
        out,
        "ERROR: {e}\nnothing was run; no completion is claimed."
    );
    EXIT_ERROR
}

fn run_fixture(seed: u64, out: &mut String) -> i32 {
    let cfg = RunConfig {
        seed,
        ..RunConfig::default()
    };
    let _ = writeln!(
        out,
        "FIXTURE RUN ({SYNTHETIC_LABEL}); seed={seed}; this is NOT a Mario result"
    );
    let rows = [
        run_episode(
            &mut SyntheticPlatformer::new(),
            &mut RandomPolicy::new(seed),
            &cfg,
        ),
        run_episode(&mut SyntheticPlatformer::new(), &mut AlwaysRight, &cfg),
        run_episode(
            &mut SyntheticPlatformer::new(),
            &mut MonteCarloPlanner::new(seed, 8, 12),
            &cfg,
        ),
    ];
    for r in &rows {
        let _ = writeln!(out, "{}", format_row(r));
    }
    EXIT_OK
}

pub fn format_row(r: &EpisodeMetrics) -> String {
    format!(
        "policy={} backend={} synthetic={} max_x={} deaths={} actions={} planning_steps={} live_frames={} fixture_goal_reached={} fps={:.0} trace_hash={}",
        r.policy,
        r.backend,
        r.synthetic,
        r.max_x,
        r.deaths,
        r.actions_used,
        r.planning_steps,
        r.live_frames,
        r.level_complete,
        r.fps,
        r.trace_hash
    )
}
