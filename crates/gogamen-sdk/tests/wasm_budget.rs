//! GOG-86: fixtures WAT para presupuesto, imports y memoria del sandbox WASM.
#![cfg(feature = "wasm")]

use gogamen_sdk::wasm::{
    WasmEffect, WasmEffectExtension, WasmError, WasmHostImport, WasmSandboxConfig,
};
use gogamen_sdk::{CapabilityVersion, ExtensionKind, ExtensionManifest, ModuleOrigin};

/// Efecto puro: acumula 100 iteraciones de `x*3 + i`.
pub const PURE_WAT: &str = r#"(module
  (func (export "run") (param i64) (result i64)
    (local $i i64) (local $acc i64)
    (loop $l
      (local.set $acc (i64.add (local.get $acc) (i64.add (i64.mul (local.get 0) (i64.const 3)) (local.get $i))))
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br_if $l (i64.lt_u (local.get $i) (i64.const 100))))
    (local.get $acc)))"#;

const LOOP_WAT: &str = r#"(module
  (func (export "run") (param i64) (result i64)
    (loop $l (br $l))
    (local.get 0)))"#;

const BAD_IMPORT_WAT: &str = r#"(module
  (import "env" "launch_missiles" (func))
  (func (export "run") (param i64) (result i64) (call 0) (local.get 0)))"#;

const WASI_IMPORT_WAT: &str = r#"(module
  (import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))
  (func (export "run") (param i64) (result i64) (local.get 0)))"#;

const BIG_MEMORY_WAT: &str = r#"(module
  (memory 1000)
  (func (export "run") (param i64) (result i64) (local.get 0)))"#;

const GROW_WAT: &str = r#"(module
  (memory 1)
  (func (export "run") (param i64) (result i64)
    (drop (memory.grow (i32.const 100)))
    (local.get 0)))"#;

const HOST_IMPORT_WAT: &str = r#"(module
  (import "gogamen" "max_i64" (func $max (param i64 i64) (result i64)))
  (func (export "run") (param i64) (result i64) (call $max (local.get 0) (i64.const 7))))"#;

fn native_pure(x: i64) -> i64 {
    (0..100i64).fold(0i64, |acc, i| {
        acc.wrapping_add(x.wrapping_mul(3).wrapping_add(i))
    })
}

#[test]
fn pure_effect_succeeds_and_matches_native() {
    let fx = WasmEffect::load(PURE_WAT.as_bytes(), WasmSandboxConfig::strict(100_000)).unwrap();
    let (v, used) = fx.invoke_with_fuel(5).unwrap();
    assert_eq!(v, native_pure(5));
    assert!(used > 0 && used < 100_000);
    // Determinista: mismo fuel consumido en cada llamada.
    assert_eq!(fx.invoke_with_fuel(5).unwrap().1, used);
}

#[test]
fn ac1_infinite_loop_fails_with_controlled_out_of_fuel() {
    let fx = WasmEffect::load(LOOP_WAT.as_bytes(), WasmSandboxConfig::strict(10_000)).unwrap();
    assert_eq!(fx.invoke(1), Err(WasmError::OutOfFuel { budget: 10_000 }));
    // El sandbox sigue utilizable tras el fallo.
    assert_eq!(fx.invoke(1), Err(WasmError::OutOfFuel { budget: 10_000 }));
}

#[test]
fn ac1_budget_too_small_for_pure_effect_fails_and_larger_succeeds() {
    let small = WasmEffect::load(PURE_WAT.as_bytes(), WasmSandboxConfig::strict(50)).unwrap();
    assert!(matches!(small.invoke(1), Err(WasmError::OutOfFuel { .. })));
    let ok = WasmEffect::load(PURE_WAT.as_bytes(), WasmSandboxConfig::strict(100_000)).unwrap();
    assert!(ok.invoke(1).is_ok());
}

#[test]
fn ac2_unauthorized_imports_rejected_before_instantiation() {
    for (wat, module) in [
        (BAD_IMPORT_WAT, "env"),
        (WASI_IMPORT_WAT, "wasi_snapshot_preview1"),
    ] {
        match WasmEffect::load(wat.as_bytes(), WasmSandboxConfig::strict(1000)) {
            Err(WasmError::UnauthorizedImport { module: m, .. }) => assert_eq!(m, module),
            other => panic!("expected UnauthorizedImport, got {other:?}"),
        }
    }
    // Un import del anfitrión no listado también se rechaza.
    assert!(matches!(
        WasmEffect::load(HOST_IMPORT_WAT.as_bytes(), WasmSandboxConfig::strict(1000)),
        Err(WasmError::UnauthorizedImport { .. })
    ));
}

#[test]
fn allowlisted_pure_host_import_works() {
    let cfg = WasmSandboxConfig::strict(1000).allow_import(WasmHostImport::MaxI64);
    let fx = WasmEffect::load(HOST_IMPORT_WAT.as_bytes(), cfg).unwrap();
    assert_eq!(fx.invoke(3), Ok(7));
    assert_eq!(fx.invoke(42), Ok(42));
}

#[test]
fn ac2_excessive_declared_memory_rejected() {
    let fx = WasmEffect::load(BIG_MEMORY_WAT.as_bytes(), WasmSandboxConfig::strict(1000)).unwrap();
    assert_eq!(
        fx.invoke(1),
        Err(WasmError::MemoryLimitExceeded {
            limit_bytes: 64 * 1024
        })
    );
}

#[test]
fn ac2_memory_growth_beyond_limit_rejected_and_within_limit_allowed() {
    let fx = WasmEffect::load(GROW_WAT.as_bytes(), WasmSandboxConfig::strict(1000)).unwrap();
    assert_eq!(
        fx.invoke(1),
        Err(WasmError::MemoryLimitExceeded {
            limit_bytes: 64 * 1024
        })
    );
    let roomy = WasmSandboxConfig::strict(1000).with_max_memory_bytes(101 * 64 * 1024);
    assert_eq!(
        WasmEffect::load(GROW_WAT.as_bytes(), roomy)
            .unwrap()
            .invoke(1),
        Ok(1)
    );
}

#[test]
fn invalid_module_and_missing_entry_are_typed() {
    assert!(matches!(
        WasmEffect::load(b"(module (func", WasmSandboxConfig::strict(10)),
        Err(WasmError::InvalidModule(_))
    ));
    assert!(matches!(
        WasmEffect::load(b"(module)", WasmSandboxConfig::strict(10)),
        Err(WasmError::MissingEntry)
    ));
}

fn manifest() -> ExtensionManifest {
    ExtensionManifest::new(
        "wasm.pure",
        ExtensionKind::Sensor,
        ModuleOrigin::ThirdParty,
        "wasm-fixture",
        CapabilityVersion::new(0, 1, 0),
    )
}

#[test]
fn extension_manifest_must_be_pure_and_static() {
    let ok = WasmEffectExtension::new(
        manifest(),
        PURE_WAT.as_bytes(),
        WasmSandboxConfig::strict(100_000),
    )
    .unwrap();
    assert_eq!(ok.manifest().id.as_str(), "wasm.pure");
    assert_eq!(ok.effect().invoke(1).unwrap(), native_pure(1));

    let with_perm = manifest().with_required_permission("actuate");
    assert!(matches!(
        WasmEffectExtension::new(with_perm, PURE_WAT.as_bytes(), WasmSandboxConfig::strict(1)),
        Err(WasmError::Manifest(_))
    ));
    let remote = manifest().with_source(gogamen_sdk::ExtensionSource::from_locator(
        "https://x/y.wasm",
    ));
    assert!(matches!(
        WasmEffectExtension::new(remote, PURE_WAT.as_bytes(), WasmSandboxConfig::strict(1)),
        Err(WasmError::Manifest(_))
    ));
}
