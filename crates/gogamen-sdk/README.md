# gogamen-sdk

Public pre-1.0 Rust SDK for Gogamen. See the crate docs (`cargo doc -p gogamen-sdk`).

## Optional: WASM effect extensions with budget (`wasm` feature, GOG-86)

Off by default: default builds of `gogamen-core`, `gogamen-cli` and the offline core do
not depend on `wasmtime`. Enable with `--features wasm` (Wasmtime 51, cranelift + runtime
only; no WASI, no async, no component model).

Scope: **pure, non-critical effects** (`run: (i64) -> i64`). A WASM effect is not a new
`ExtensionKind`; it never receives actuators, permissions or capabilities.

- **Fuel**: `WasmSandboxConfig::fuel` is a deterministic instruction budget per call.
  Exhaustion yields `WasmError::OutOfFuel` and the sandbox stays usable.
- **Memory**: `max_memory_bytes` is enforced through a `ResourceLimiter`. Excessive
  declared memory or `memory.grow` past the cap yields `WasmError::MemoryLimitExceeded`
  (a denied grow is a violation even if the module ignores the `-1` result).
- **Imports**: validated against a closed allowlist (`WasmHostImport`, module `gogamen`,
  pure functions only) *before* instantiation -> `WasmError::UnauthorizedImport`.
  Default allowlist is empty, so WASI/env/anything else is rejected.
- **Manifest**: `WasmEffectExtension` requires a static-linked manifest with no
  required permissions.

Tests: `cargo test -p gogamen-sdk --features wasm --test wasm_budget` (WAT fixtures:
infinite loop, unauthorized import, huge declared memory, growth, pure success).

### Cost vs native (no hard real-time guarantee)

`cargo run --release -p gogamen-sdk --features wasm --example wasm_bench` compares a
100-iteration pure effect. Measured on the dev machine: one-off compile ~2.9 ms;
~1.6 us per invocation (fresh Store + instantiate + call, ~1600 fuel) vs ~1.2 ns for the
native Rust function (the compiler likely optimizes the native loop heavily, so the
ratio of ~1300x is an upper bound dominated by per-call instantiation, not by Wasm
instruction speed). **No hard real-time guarantee is made**: fuel bounds executed Wasm
instructions deterministically, not wall-clock time; compile/instantiate latency varies.
Do not use for deadline-critical paths.
