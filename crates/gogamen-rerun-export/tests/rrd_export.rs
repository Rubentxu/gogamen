//! Integración con el SDK de Rerun (feature `rerun`): escribe `.rrd` reales a un directorio
//! temporal. AC1 (archivo válido con timelines), AC2 (no fuga en el archivo) y AC3.

use std::path::{Path, PathBuf};

use gogamen_rerun_export::fixtures::{
    FIXTURE_MESSAGE, FIXTURE_REJECT, FIXTURE_SECRET, grid_recording_with_gap, scientific_archive,
};
use gogamen_rerun_export::rerun_sink::{export_archive, export_recording};
use gogamen_rerun_export::{ExportOptions, ExportPermissions};
use gogamen_runtime::scientific_trace::{SensitiveFieldMode, SensitiveFieldPolicy};

fn tmpdir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gogamen-rerun-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Locates the Rerun CLI matching the SDK format (`RERUN_BIN`, `~/.cargo/bin/rerun`, PATH).
fn rerun_cli() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("RERUN_BIN") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let candidate = home.join(".cargo/bin/rerun");
    if candidate.exists() {
        return Some(candidate);
    }
    Some(PathBuf::from("rerun")).filter(|p| {
        std::process::Command::new(p)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

/// Decodes the whole .rrd (all chunks, data included) with the official CLI. The .rrd is
/// LZ4-compressed, so raw byte search is NOT a valid oracle for absence of secrets.
/// Panics unless `GOGAMEN_ALLOW_NO_RERUN_CLI=1` (then returns `None` and logs NOT_RUN).
fn decode(path: &Path) -> Option<String> {
    let Some(cli) = rerun_cli() else {
        assert_eq!(
            std::env::var("GOGAMEN_ALLOW_NO_RERUN_CLI").as_deref(),
            Ok("1"),
            "rerun CLI not found; set RERUN_BIN or GOGAMEN_ALLOW_NO_RERUN_CLI=1 (decode checks NOT_RUN)"
        );
        eprintln!("NOT_RUN: rerun CLI decode checks");
        return None;
    };
    let verify = std::process::Command::new(&cli)
        .args(["rrd", "verify"])
        .arg(path)
        .output()
        .unwrap();
    assert!(
        verify.status.success(),
        "rerun rrd verify failed: {}",
        String::from_utf8_lossy(&verify.stderr)
    );
    let out = std::process::Command::new(&cli)
        .args(["rrd", "print", "-vvv"])
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success());
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn read(p: &Path) -> Vec<u8> {
    std::fs::read(p).unwrap()
}

#[test]
fn recording_exports_a_nonempty_rrd_with_timelines_and_entities() {
    let dir = tmpdir("grid");
    let out = dir.join("grid.rrd");
    let rec = grid_recording_with_gap(40, 10, 14);
    export_recording(&rec, &ExportOptions::default(), &out).unwrap();
    assert!(read(&out).len() > 256);
    assert!(!out.with_extension("rrd.partial").exists());
    if let Some(text) = decode(&out) {
        for needle in [
            "index_name: sim_time",
            "index_name: tick",
            "/gogamen/scene/agent",
            "/gogamen/scene/map/wall",
            "/gogamen/series/total_reward",
            "total_reward [reward]",
            "collisions [count]",
            "PT1S", // sim_time is a duration in seconds (tick 10 -> 1 s)
            "missing frame at tick 10",
            "missing frame at tick 14",
        ] {
            assert!(text.contains(needle), "missing `{needle}` in decoded rrd");
        }
        assert!(
            text.contains("NaN"),
            "gap scalars must be explicit NaN, not zero"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn archive_rrd_never_contains_denied_secrets_or_messages() {
    let dir = tmpdir("sci");
    let out = dir.join("sci.rrd");
    export_archive(&scientific_archive(), &ExportOptions::default(), &out).unwrap();
    if let Some(text) = decode(&out) {
        for leaked in [
            FIXTURE_SECRET,
            FIXTURE_MESSAGE,
            FIXTURE_REJECT,
            "/secret/path",
            "EIO",
        ] {
            assert!(!text.contains(leaked), "leaked `{leaked}`");
        }
        assert!(text.contains("[REDACTED]"));
        assert!(text.contains("[WITHHELD"));
        assert!(text.contains("index_name: revision"));
        assert!(text.contains("revisions 3..=4"));
    }

    // Con permiso de mensajes aparecen; el secreto sigue sin aparecer.
    let out2 = dir.join("sci2.rrd");
    let opts = ExportOptions {
        permissions: ExportPermissions {
            messages: true,
            secrets: false,
        },
        ..ExportOptions::default()
    };
    export_archive(&scientific_archive(), &opts, &out2).unwrap();
    if let Some(text) = decode(&out2) {
        assert!(text.contains(FIXTURE_MESSAGE));
        assert!(text.contains(FIXTURE_REJECT));
        assert!(!text.contains(FIXTURE_SECRET));
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn failed_export_is_discarded_and_replay_is_byte_identical() {
    let dir = tmpdir("fail");
    let rec = grid_recording_with_gap(25, 3, 4);
    let archive = scientific_archive();
    let policy = SensitiveFieldPolicy::new(SensitiveFieldMode::Redact);
    let (rb, ab) = (rec.to_bytes(), archive.to_bytes(&policy).unwrap());

    // Destino imposible: el directorio no existe.
    let bad = dir.join("no/such/dir/out.rrd");
    assert!(export_recording(&rec, &ExportOptions::default(), &bad).is_err());
    assert!(export_archive(&archive, &ExportOptions::default(), &bad).is_err());
    assert!(!bad.exists());

    // Export omitido/descartado tras éxito: borrar el .rrd no afecta al replay.
    let ok = dir.join("ok.rrd");
    export_recording(&rec, &ExportOptions::default(), &ok).unwrap();
    std::fs::remove_file(&ok).unwrap();

    assert_eq!(rec.to_bytes(), rb);
    assert_eq!(archive.to_bytes(&policy).unwrap(), ab);
    // El replay sigue abriendo la grabación original.
    let reopened = gogamen_studio::replay::Recording::open(&rb).unwrap();
    assert_eq!(reopened, rec);
    std::fs::remove_dir_all(dir).unwrap();
}
