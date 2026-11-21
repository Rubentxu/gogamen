//! GOG-150: límites de la fase de carga/compilación del sandbox WASM.
#![cfg(feature = "wasm")]

use std::time::{Duration, Instant};

use gogamen_sdk::wasm::{WasmEffect, WasmError, WasmSandboxConfig};

const SMALL: &str =
    r#"(module (func (export "run") (param i64) (result i64) local.get 0 i64.const 1 i64.add))"#;

fn heavy_module() -> String {
    // Muchas funciones con cuerpos grandes: compilación cara pero módulo moderado.
    let mut s = String::from("(module\n");
    for i in 0..200 {
        s.push_str(&format!(
            "(func $f{i} (param i64) (result i64) (local i64)\n"
        ));
        s.push_str("local.get 0\n");
        for j in 0..200 {
            s.push_str(&format!(
                "i64.const {j}\ni64.add\nlocal.tee 1\nlocal.get 1\ni64.mul\n"
            ));
        }
        s.push_str(")\n");
    }
    s.push_str("(func (export \"run\") (param i64) (result i64) local.get 0))");
    s
}

#[test]
fn gog_150_oversized_module_rejected_before_compile() {
    let cfg = WasmSandboxConfig::strict(1000).with_max_module_bytes(16);
    let err = WasmEffect::load(SMALL.as_bytes(), cfg).unwrap_err();
    assert!(
        matches!(err, WasmError::ModuleTooLarge { limit: 16, size } if size == SMALL.len()),
        "{err:?}"
    );
    // Basura no-Wasm sobredimensionada: debe ser ModuleTooLarge, no InvalidModule.
    let junk = vec![0xFFu8; 64];
    let cfg = WasmSandboxConfig::strict(1000).with_max_module_bytes(8);
    assert!(matches!(
        WasmEffect::load(&junk, cfg),
        Err(WasmError::ModuleTooLarge { .. })
    ));
}

#[test]
fn gog_150_small_valid_module_still_works_with_defaults() {
    let e = WasmEffect::load(SMALL.as_bytes(), WasmSandboxConfig::strict(10_000)).unwrap();
    assert_eq!(e.invoke(41).unwrap(), 42);
}

#[test]
fn gog_150_expensive_compilation_times_out_bounded() {
    let wat = heavy_module();
    let cfg = WasmSandboxConfig::strict(1000)
        .with_max_module_bytes(8 * 1024 * 1024)
        .with_compile_timeout(Duration::from_millis(1));
    let t = Instant::now();
    let err = WasmEffect::load(wat.as_bytes(), cfg).unwrap_err();
    assert!(
        matches!(err, WasmError::CompileTimeout { timeout_ms: 1 }),
        "{err:?}"
    );
    assert!(t.elapsed() < Duration::from_secs(2), "caller blocked");
}
