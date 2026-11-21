mod protocol;

use std::env;
use std::fs;
use std::process;

use gogamen_compiler::CompiledPlanArtifact;
use gogamen_core::MasterSeed;
use gogamen_lab_grid::{run_grid_episode, run_grid_scenario, run_language_world_demo};
use gogamen_runtime::{PlanRuntimeEngine, run_1d_episode};
use gogamen_spec::{
    ExperimentSpec, SystemSpec, canonical_two_node_fixture, canonical_two_node_fixture_json,
};

const DEFAULT_TICKS: u64 = 100;
const DEFAULT_SEED: u64 = 42;
/// Tope por defecto de bytes al leer un fichero JSON de entrada (alineado con `JsonLimits`).
const DEFAULT_MAX_INPUT_BYTES: usize = gogamen_spec::DEFAULT_JSON_MAX_BYTES;

/// Lee un fichero UTF-8 rechazando entradas que excedan `max_bytes` sin asignarlas enteras:
/// lee como máximo `max_bytes + 1` bytes y falla si se alcanza ese último.
fn read_bounded(path: &str, max_bytes: usize) -> Result<String, std::io::Error> {
    use std::io::{Error, ErrorKind, Read};
    let file = fs::File::open(path)?;
    let limit = (max_bytes as u64).saturating_add(1);
    let mut buf = Vec::new();
    file.take(limit).read_to_end(&mut buf)?;
    if buf.len() > max_bytes {
        return Err(Error::new(
            ErrorKind::InvalidData,
            format!("la entrada excede el máximo de {max_bytes} bytes"),
        ));
    }
    String::from_utf8(buf).map_err(|e| Error::new(ErrorKind::InvalidData, e))
}

fn help_text() -> String {
    format!(
        "{name} {version}\nUso: {name} [--version | --help | run-1d [--ticks N] [--seed S] | run-grid [--ticks N] [--seed S] [--partial-obs] | run-grid-spec --json-file <path> [--ticks N] [--seed S] | run-language-world [--seed S] | run-spec [--fixture | --json-file <path>] [--ticks N] [--seed S] | compile-artifact [--fixture | --json-file <path>] --out <path> | inspect-artifact --artifact-file <path> | run-artifact --artifact-file <path> [--ticks N] [--seed S] | serve-protocol]\n",
        name = gogamen_core::project_name(),
        version = gogamen_core::core_version(),
    )
}

fn version_text() -> String {
    format!(
        "{name} {version}\n",
        name = gogamen_core::project_name(),
        version = gogamen_core::core_version(),
    )
}

fn parse_run_1d_args(args: &[String]) -> Result<(u64, u64), String> {
    let mut ticks = DEFAULT_TICKS;
    let mut seed = DEFAULT_SEED;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--ticks" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --ticks".to_string())?;
                ticks = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --ticks: {val}"))?;
                idx += 2;
            }
            "--seed" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --seed: {val}"))?;
                idx += 2;
            }
            other => {
                return Err(format!("opción desconocida para run-1d: {other}"));
            }
        }
    }

    Ok((ticks, seed))
}

fn run_1d_command(args: &[String]) -> Result<String, String> {
    let (ticks, seed) = parse_run_1d_args(args)?;
    let summary = run_1d_episode(ticks, MasterSeed::new(seed))
        .map_err(|err| format!("error ejecutando run-1d: {err}"))?;

    Ok(format!(
        "gogamen run-1d\n  ticks: {}\n  seed: {}\n  actions_applied: {}\n  fallbacks: {}\n  rejected: {}\n  final_position: {:.3}\n  trace_hash: {}\n",
        summary.ticks_executed,
        summary.seed.as_u64(),
        summary.actions_applied,
        summary.fallbacks_used,
        summary.rejected_count,
        summary.final_position,
        summary.trace.trace_hash_hex(),
    ))
}

fn parse_run_grid_args(args: &[String]) -> Result<(u64, u64, bool), String> {
    let mut ticks = DEFAULT_TICKS;
    let mut seed = DEFAULT_SEED;
    let mut partial_obs = false;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--ticks" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --ticks".to_string())?;
                ticks = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --ticks: {val}"))?;
                idx += 2;
            }
            "--seed" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --seed: {val}"))?;
                idx += 2;
            }
            "--partial-obs" => {
                partial_obs = true;
                idx += 1;
            }
            other => {
                return Err(format!("opción desconocida para run-grid: {other}"));
            }
        }
    }

    Ok((ticks, seed, partial_obs))
}

fn run_grid_command(args: &[String]) -> Result<String, String> {
    let (ticks, seed, partial_obs) = parse_run_grid_args(args)?;
    let summary = run_grid_episode(ticks, MasterSeed::new(seed), partial_obs)
        .map_err(|err| format!("error ejecutando run-grid: {err}"))?;

    Ok(format!(
        "gogamen run-grid\n  ticks: {}\n  seed: {}\n  partial_obs: {}\n  final_pos: ({}, {})\n  total_reward: {:.3}\n  collisions: {}\n  terminal: {:?}\n  trace_hash: {}\n",
        summary.ticks_executed,
        summary.seed.as_u64(),
        summary.partial_obs,
        summary.final_pos.x,
        summary.final_pos.y,
        summary.total_reward,
        summary.collisions,
        summary.terminal,
        summary.trace_hash_hex(),
    ))
}

fn run_grid_spec_command(args: &[String]) -> Result<String, String> {
    let mut json_file: Option<String> = None;
    let mut ticks = DEFAULT_TICKS;
    let mut seed: Option<u64> = None;
    let mut idx = 0;
    while idx < args.len() {
        let val = args.get(idx + 1);
        match args[idx].as_str() {
            "--json-file" => {
                json_file = Some(
                    val.ok_or_else(|| "falta valor para --json-file".to_string())?
                        .clone(),
                );
            }
            "--ticks" => {
                let v = val.ok_or_else(|| "falta valor para --ticks".to_string())?;
                ticks = v
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --ticks: {v}"))?;
            }
            "--seed" => {
                let v = val.ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = Some(
                    v.parse::<u64>()
                        .map_err(|_| format!("valor inválido para --seed: {v}"))?,
                );
            }
            other => return Err(format!("opción desconocida para run-grid-spec: {other}")),
        }
        idx += 2;
    }
    let path =
        json_file.ok_or_else(|| "falta --json-file <path> para run-grid-spec".to_string())?;
    let raw = read_bounded(&path, DEFAULT_MAX_INPUT_BYTES)
        .map_err(|err| format!("error leyendo --json-file '{path}': {err}"))?;
    let spec = ExperimentSpec::from_json(&raw)
        .map_err(|err| format!("error parseando ExperimentSpec '{path}': {err}"))?;
    let summary = run_grid_scenario(&spec, ticks, seed)
        .map_err(|err| format!("error ejecutando run-grid-spec: {err}"))?;
    Ok(format!(
        "gogamen run-grid-spec\n  source: json-file ({path})\n  experiment_id: {}\n  ticks: {}\n  seed: {}\n  final_pos: ({}, {})\n  total_reward: {:.3}\n  collisions: {}\n  terminal: {:?}\n  trace_hash: {}\n",
        spec.experiment_id,
        summary.ticks_executed,
        summary.seed.as_u64(),
        summary.final_pos.x,
        summary.final_pos.y,
        summary.total_reward,
        summary.collisions,
        summary.terminal,
        summary.trace_hash_hex(),
    ))
}

fn parse_run_spec_args(args: &[String]) -> Result<(Option<String>, u64, u64), String> {
    let mut use_fixture = false;
    let mut json_file: Option<String> = None;
    let mut ticks = DEFAULT_TICKS;
    let mut seed = DEFAULT_SEED;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--fixture" => {
                use_fixture = true;
                idx += 1;
            }
            "--json-file" => {
                let path = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --json-file".to_string())?;
                json_file = Some(path.clone());
                idx += 2;
            }
            "--ticks" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --ticks".to_string())?;
                ticks = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --ticks: {val}"))?;
                idx += 2;
            }
            "--seed" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --seed: {val}"))?;
                idx += 2;
            }
            other => {
                return Err(format!("opción desconocida para run-spec: {other}"));
            }
        }
    }

    if use_fixture && json_file.is_some() {
        return Err("no se puede combinar --fixture y --json-file".to_string());
    }

    Ok((json_file, ticks, seed))
}

fn run_spec_command(args: &[String]) -> Result<String, String> {
    let (json_file, ticks, seed) = parse_run_spec_args(args)?;
    let engine = PlanRuntimeEngine::new();
    let master_seed = MasterSeed::new(seed);

    if let Some(path) = json_file {
        let json = read_bounded(&path, DEFAULT_MAX_INPUT_BYTES)
            .map_err(|err| format!("error leyendo --json-file '{path}': {err}"))?;
        let summary = engine
            .run_json_with_seed(&json, ticks, master_seed)
            .map_err(|err| format!("error ejecutando run-spec desde JSON: {err}"))?;

        return Ok(format!(
            "gogamen run-spec\n  source: json-file ({path})\n  spec_id: {}\n  ticks: {}\n  seed: {seed}\n  semantic_fingerprint: {:016x}\n  trace_hash: {}\n",
            summary.spec_id,
            summary.ticks_executed,
            summary.semantic_fingerprint,
            summary.trace_hash_hex,
        ));
    }

    let rust_summary = engine
        .run_spec_with_seed(&canonical_two_node_fixture(), ticks, master_seed)
        .map_err(|err| format!("error ejecutando run-spec por API Rust: {err}"))?;
    let json_summary = engine
        .run_json_with_seed(&canonical_two_node_fixture_json(), ticks, master_seed)
        .map_err(|err| format!("error ejecutando run-spec por JSON: {err}"))?;

    if rust_summary.semantic_fingerprint != json_summary.semantic_fingerprint {
        return Err(format!(
            "huella semántica divergente entre API Rust ({:016x}) y JSON ({:016x})",
            rust_summary.semantic_fingerprint, json_summary.semantic_fingerprint
        ));
    }
    if rust_summary.trace_hash != json_summary.trace_hash {
        return Err(format!(
            "trace_hash divergente entre API Rust ({}) y JSON ({})",
            rust_summary.trace_hash_hex, json_summary.trace_hash_hex
        ));
    }

    Ok(format!(
        "gogamen run-spec\n  source: canonical-fixture\n  spec_id: {}\n  ticks: {}\n  semantic_fingerprint: {:016x}\n  rust_trace_hash: {}\n  json_trace_hash: {}\n  equivalent: true\n",
        rust_summary.spec_id,
        rust_summary.ticks_executed,
        rust_summary.semantic_fingerprint,
        rust_summary.trace_hash_hex,
        json_summary.trace_hash_hex,
    ))
}

fn parse_run_language_world_args(args: &[String]) -> Result<u64, String> {
    let mut seed = DEFAULT_SEED;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--seed" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --seed: {val}"))?;
                idx += 2;
            }
            other => {
                return Err(format!(
                    "opción desconocida para run-language-world: {other}"
                ));
            }
        }
    }

    Ok(seed)
}

fn run_language_world_command(args: &[String]) -> Result<String, String> {
    let seed = parse_run_language_world_args(args)?;
    let report = run_language_world_demo(MasterSeed::new(seed))
        .map_err(|err| format!("error ejecutando run-language-world: {err}"))?;
    Ok(report.format_trace())
}

fn compile_artifact_command(args: &[String]) -> Result<String, String> {
    let mut use_fixture = false;
    let mut json_file: Option<String> = None;
    let mut out_file: Option<String> = None;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--fixture" => {
                use_fixture = true;
                idx += 1;
            }
            "--json-file" => {
                let path = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --json-file".to_string())?;
                json_file = Some(path.clone());
                idx += 2;
            }
            "--out" => {
                let path = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --out".to_string())?;
                out_file = Some(path.clone());
                idx += 2;
            }
            other => {
                return Err(format!("opción desconocida para compile-artifact: {other}"));
            }
        }
    }

    if use_fixture && json_file.is_some() {
        return Err("no se puede combinar --fixture y --json-file".to_string());
    }
    let out_path =
        out_file.ok_or_else(|| "falta --out <path> para compile-artifact".to_string())?;

    let spec = if let Some(path) = json_file {
        let raw = read_bounded(&path, DEFAULT_MAX_INPUT_BYTES)
            .map_err(|err| format!("error leyendo --json-file '{path}': {err}"))?;
        SystemSpec::from_json(&raw)
            .map_err(|err| format!("error parseando SystemSpec '{path}': {err}"))?
    } else {
        canonical_two_node_fixture()
    };

    let engine = PlanRuntimeEngine::new();
    let artifact = engine
        .compile_artifact(&spec)
        .map_err(|err| format!("error compilando artefacto: {err}"))?;
    let json = artifact
        .to_canonical_json()
        .map_err(|err| format!("error serializando artefacto: {err}"))?;
    fs::write(&out_path, &json)
        .map_err(|err| format!("error escribiendo artefacto '{out_path}': {err}"))?;

    Ok(format!(
        "gogamen compile-artifact\n  out: {out_path}\n  spec_id: {}\n  artifact_version: {}\n  runtime_version: {}\n  artifact_hash: {}\n  spec_hash: {}\n",
        spec.spec_id,
        artifact.artifact_version(),
        artifact.runtime_version(),
        artifact.artifact_hash(),
        artifact.spec_hash(),
    ))
}

fn inspect_artifact_command(args: &[String]) -> Result<String, String> {
    let mut artifact_file: Option<String> = None;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--artifact-file" => {
                let path = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --artifact-file".to_string())?;
                artifact_file = Some(path.clone());
                idx += 2;
            }
            other => {
                return Err(format!("opción desconocida para inspect-artifact: {other}"));
            }
        }
    }

    let path = artifact_file
        .ok_or_else(|| "falta --artifact-file <path> para inspect-artifact".to_string())?;
    let raw = read_bounded(&path, DEFAULT_MAX_INPUT_BYTES)
        .map_err(|err| format!("error leyendo --artifact-file '{path}': {err}"))?;
    let info = CompiledPlanArtifact::inspect_json(&raw)
        .map_err(|err| format!("error inspeccionando artefacto '{path}': {err}"))?;

    Ok(format!(
        "gogamen inspect-artifact\n  file: {path}\n  spec_id: {}\n  artifact_version: {}\n  runtime_version: {}\n  compiler_version: {}\n  catalog_fingerprint: {}\n  artifact_hash: {}\n  spec_hash: {}\n  nodes: {}\n  factories: {}\n  operations: {}\n",
        info.spec_id,
        info.artifact_version,
        info.runtime_version,
        info.compiler_version,
        info.catalog_fingerprint,
        info.artifact_hash,
        info.spec_hash,
        info.node_count,
        info.factory_count,
        info.operation_count,
    ))
}

fn run_artifact_command(args: &[String]) -> Result<String, String> {
    let mut artifact_file: Option<String> = None;
    let mut ticks = DEFAULT_TICKS;
    let mut seed = DEFAULT_SEED;
    let mut idx = 0;

    while idx < args.len() {
        match args[idx].as_str() {
            "--artifact-file" => {
                let path = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --artifact-file".to_string())?;
                artifact_file = Some(path.clone());
                idx += 2;
            }
            "--ticks" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --ticks".to_string())?;
                ticks = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --ticks: {val}"))?;
                idx += 2;
            }
            "--seed" => {
                let val = args
                    .get(idx + 1)
                    .ok_or_else(|| "falta valor para --seed".to_string())?;
                seed = val
                    .parse::<u64>()
                    .map_err(|_| format!("valor inválido para --seed: {val}"))?;
                idx += 2;
            }
            other => {
                return Err(format!("opción desconocida para run-artifact: {other}"));
            }
        }
    }

    let path = artifact_file
        .ok_or_else(|| "falta --artifact-file <path> para run-artifact".to_string())?;
    let raw = read_bounded(&path, DEFAULT_MAX_INPUT_BYTES)
        .map_err(|err| format!("error leyendo --artifact-file '{path}': {err}"))?;
    let engine = PlanRuntimeEngine::new();
    let summary = engine
        .run_artifact_json_with_seed(&raw, ticks, MasterSeed::new(seed))
        .map_err(|err| format!("error ejecutando run-artifact '{path}': {err}"))?;

    Ok(format!(
        "gogamen run-artifact\n  file: {path}\n  spec_id: {}\n  ticks: {}\n  seed: {seed}\n  semantic_fingerprint: {:016x}\n  trace_hash: {}\n",
        summary.spec_id,
        summary.ticks_executed,
        summary.semantic_fingerprint,
        summary.trace_hash_hex,
    ))
}

fn serve_protocol_command(args: &[String]) -> Result<String, String> {
    if let Some(other) = args.first() {
        return Err(format!("opción desconocida para serve-protocol: {other}"));
    }
    protocol::serve_stdio()?;
    Ok(String::new())
}

fn dispatch(args: &[String]) -> Result<String, String> {
    match args.get(1).map(String::as_str) {
        Some("--version" | "-V") => Ok(version_text()),
        Some("run-1d") => run_1d_command(&args[2..]),
        Some("run-grid") => run_grid_command(&args[2..]),
        Some("run-language-world") => run_language_world_command(&args[2..]),
        Some("run-grid-spec") => run_grid_spec_command(&args[2..]),
        Some("run-spec") => run_spec_command(&args[2..]),
        Some("compile-artifact") => compile_artifact_command(&args[2..]),
        Some("inspect-artifact") => inspect_artifact_command(&args[2..]),
        Some("run-artifact") => run_artifact_command(&args[2..]),
        Some("serve-protocol") => serve_protocol_command(&args[2..]),
        _ => Ok(help_text()),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match dispatch(&args) {
        Ok(output) => print!("{output}"),
        Err(err) => {
            eprintln!("{err}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gog_138_read_bounded_rejects_oversized_file_and_accepts_exact() {
        let dir = env::temp_dir().join(format!("gog138_{}", process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("in.json");
        fs::write(&path, "0123456789").unwrap();
        let p = path.to_str().unwrap();
        assert_eq!(read_bounded(p, 10).unwrap(), "0123456789");
        let err = read_bounded(p, 9).unwrap_err();
        assert!(err.to_string().contains("excede"), "{err}");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cli_prints_version_and_usage() {
        let help = dispatch(&["gogamen".to_string()]).unwrap();
        assert!(help.contains("gogamen"));
        assert!(help.contains("Uso:"));

        let version = dispatch(&["gogamen".to_string(), "--version".to_string()]).unwrap();
        assert_eq!(
            version,
            format!("gogamen {}\n", gogamen_core::core_version())
        );
    }

    #[test]
    fn cli_run_1d_executes_100_ticks_with_default_or_explicit_flags() {
        let out_default = dispatch(&["gogamen".to_string(), "run-1d".to_string()]).unwrap();
        assert!(out_default.contains("ticks: 100"));
        assert!(out_default.contains("seed: 42"));
        assert!(out_default.contains("actions_applied: 100"));
        assert!(out_default.contains("fallbacks: 0"));
        assert!(out_default.contains("trace_hash: "));

        let out_explicit = dispatch(&[
            "gogamen".to_string(),
            "run-1d".to_string(),
            "--ticks".to_string(),
            "100".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ])
        .unwrap();
        assert_eq!(out_default, out_explicit);
    }

    #[test]
    fn cli_run_1d_rejects_invalid_arguments() {
        let err = dispatch(&[
            "gogamen".to_string(),
            "run-1d".to_string(),
            "--ticks".to_string(),
            "abc".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("--ticks"));
    }

    #[test]
    fn cli_run_grid_reproduces_episode_by_seed_and_supports_partial_obs() {
        let args_a = vec![
            "gogamen".to_string(),
            "run-grid".to_string(),
            "--ticks".to_string(),
            "40".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ];
        let out_a = dispatch(&args_a).unwrap();
        let out_b = dispatch(&args_a).unwrap();
        assert_eq!(out_a, out_b);
        assert!(out_a.contains("gogamen run-grid"));
        assert!(out_a.contains("ticks: 40"));
        assert!(out_a.contains("seed: 42"));
        assert!(out_a.contains("final_pos: "));
        assert!(out_a.contains("total_reward: "));
        assert!(out_a.contains("terminal: "));
        assert!(out_a.contains("trace_hash: "));

        let mut args_partial = args_a;
        args_partial.push("--partial-obs".to_string());
        let out_partial = dispatch(&args_partial).unwrap();
        assert!(out_partial.contains("partial_obs: true"));
        assert_ne!(out_a, out_partial);
    }

    #[test]
    fn cli_run_grid_rejects_unknown_flags() {
        let err = dispatch(&[
            "gogamen".to_string(),
            "run-grid".to_string(),
            "--unknown".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("opción desconocida"));
    }

    #[test]
    fn cli_run_spec_executes_canonical_fixture_with_equivalent_rust_and_json_traces() {
        let out_default = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--ticks".to_string(),
            "10".to_string(),
        ])
        .unwrap();
        let out_fixture = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--fixture".to_string(),
            "--ticks".to_string(),
            "10".to_string(),
        ])
        .unwrap();
        assert_eq!(out_default, out_fixture);
        assert!(out_default.contains("gogamen run-spec"));
        assert!(out_default.contains("spec_id: two_node_canonical_v0"));
        assert!(out_default.contains("ticks: 10"));
        assert!(out_default.contains("equivalent: true"));

        let temp_path =
            env::temp_dir().join(format!("gogamen_run_spec_test_{}.json", process::id()));
        fs::write(&temp_path, canonical_two_node_fixture_json()).unwrap();
        let out_file = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--json-file".to_string(),
            temp_path.to_string_lossy().into_owned(),
            "--ticks".to_string(),
            "10".to_string(),
        ])
        .unwrap();
        let _ = fs::remove_file(&temp_path);

        assert!(out_file.contains("spec_id: two_node_canonical_v0"));
        assert!(out_file.contains("ticks: 10"));

        // Verify the json-file trace_hash matches rust_trace_hash from fixture output.
        let file_hash = out_file
            .lines()
            .find_map(|line| line.trim().strip_prefix("trace_hash: "))
            .unwrap();
        let rust_hash = out_default
            .lines()
            .find_map(|line| line.trim().strip_prefix("rust_trace_hash: "))
            .unwrap();
        assert_eq!(file_hash, rust_hash);
    }

    #[test]
    fn cli_run_spec_rejects_conflicting_and_unknown_flags() {
        let err = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--fixture".to_string(),
            "--json-file".to_string(),
            "foo.json".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("combinar"));

        let err_unknown = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--bad".to_string(),
        ])
        .unwrap_err();
        assert!(err_unknown.contains("opción desconocida"));
    }

    #[test]
    fn cli_run_language_world_prints_offline_trace_and_validates_flags() {
        let out_a = dispatch(&[
            "gogamen".to_string(),
            "run-language-world".to_string(),
            "--seed".to_string(),
            "42".to_string(),
        ])
        .unwrap();
        let out_b = dispatch(&["gogamen".to_string(), "run-language-world".to_string()]).unwrap();
        assert_eq!(out_a, out_b);
        assert!(out_a.contains("gogamen run-language-world"));
        assert!(out_a.contains("speech_act: notice"));
        assert!(out_a.contains("speech_act: order"));
        assert!(out_a.contains("grounded_outcome:"));
        assert!(out_a.contains("belief_obstacle:"));
        assert!(out_a.contains("planner_fmc_action:"));
        assert!(out_a.contains("guard_decision:"));

        let err = dispatch(&[
            "gogamen".to_string(),
            "run-language-world".to_string(),
            "--unknown".to_string(),
        ])
        .unwrap_err();
        assert!(err.contains("opción desconocida"));
    }

    #[test]
    fn cli_compile_inspect_and_run_artifact_matches_run_spec() {
        let mut artifact_path = env::temp_dir();
        artifact_path.push(format!(
            "gogamen_cli_artifact_{}_{}.json",
            process::id(),
            DEFAULT_TICKS
        ));
        let path_str = artifact_path.to_string_lossy().into_owned();

        let compile_out = dispatch(&[
            "gogamen".to_string(),
            "compile-artifact".to_string(),
            "--fixture".to_string(),
            "--out".to_string(),
            path_str.clone(),
        ])
        .unwrap();
        assert!(compile_out.contains("gogamen compile-artifact"));
        assert!(compile_out.contains("artifact_hash:"));

        let inspect_out = dispatch(&[
            "gogamen".to_string(),
            "inspect-artifact".to_string(),
            "--artifact-file".to_string(),
            path_str.clone(),
        ])
        .unwrap();
        assert!(inspect_out.contains("gogamen inspect-artifact"));
        assert!(inspect_out.contains("nodes: 2"));

        let run_art_out = dispatch(&[
            "gogamen".to_string(),
            "run-artifact".to_string(),
            "--artifact-file".to_string(),
            path_str.clone(),
            "--ticks".to_string(),
            "10".to_string(),
        ])
        .unwrap();
        let run_spec_out = dispatch(&[
            "gogamen".to_string(),
            "run-spec".to_string(),
            "--fixture".to_string(),
            "--ticks".to_string(),
            "10".to_string(),
        ])
        .unwrap();

        let _ = fs::remove_file(&artifact_path);

        let art_hash = run_art_out
            .lines()
            .find_map(|l| l.trim().strip_prefix("trace_hash: "))
            .unwrap();
        let spec_hash = run_spec_out
            .lines()
            .find_map(|l| l.trim().strip_prefix("rust_trace_hash: "))
            .unwrap();
        assert_eq!(art_hash, spec_hash);
    }

    #[test]
    fn cli_run_grid_spec_runs_experiment_json_and_rejects_invalid() {
        let rows =
            gogamen_lab_grid::map_to_ascii_rows(&gogamen_lab_grid::GridMap::cul_de_sac_map());
        let spec = gogamen_lab_grid::grid_experiment_spec("cli_grid", 42, &rows);
        let path = env::temp_dir().join(format!("gogamen_run_grid_spec_{}.json", process::id()));
        fs::write(&path, spec.to_canonical_json().unwrap()).unwrap();
        let p = path.to_string_lossy().into_owned();
        let args = |extra: &[&str]| {
            let mut v: Vec<String> = [
                "gogamen",
                "run-grid-spec",
                "--json-file",
                &p,
                "--ticks",
                "40",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect();
            v.extend(extra.iter().map(|s| s.to_string()));
            v
        };
        let out = dispatch(&args(&[])).unwrap();
        assert_eq!(out, dispatch(&args(&["--seed", "42"])).unwrap());
        let grid = dispatch(&[
            "gogamen".into(),
            "run-grid".into(),
            "--ticks".into(),
            "40".into(),
        ])
        .unwrap();
        let hash = |t: &str| {
            t.lines()
                .find_map(|l| l.trim().strip_prefix("trace_hash: "))
                .unwrap()
                .to_owned()
        };
        assert_eq!(hash(&out), hash(&grid));
        assert_ne!(
            hash(&out),
            hash(&dispatch(&args(&["--seed", "7"])).unwrap())
        );

        fs::write(&path, "{\"not\":\"a spec\"}").unwrap();
        assert!(dispatch(&args(&[])).unwrap_err().contains("ExperimentSpec"));
        let _ = fs::remove_file(&path);
    }
}
