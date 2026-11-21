# Gogamen MVP release (pre-1.0, local distribution)

Status: **pre-1.0, `0.1.0`, local builds only.** There is **no automatic publication**: no `cargo publish`,
no registry/GitHub release/upload workflow and no remote operation is part of this repository. A release
is a reproducible local build plus a passing (or honestly reported) `scripts/release_check.sh`.

## Profiles

| Profile | What | Build/run | Notes |
|---|---|---|---|
| **core / cli** (default) | `gogamen-core`, `gogamen-spec`, `gogamen-compiler`, `gogamen-runtime`, `gogamen-cli` (+ Grid lab). No graphics. | `cargo build --release -p gogamen-cli` -> `target/release/gogamen` | Dependency boundary: `cargo tree -p gogamen-core -p gogamen-cli --offline` must not show studio/bevy/egui/eframe/winit/wasmtime/rerun/mujoco (checked by the release script). |
| **studio** (optional) | `gogamen-studio`: common shell, SVG/headless views for the tour labs. Default build is std-only and headless. | `cargo run -p gogamen-studio --offline --example grid_svg -- 5 42 > grid.svg` | Real window needs the opt-in feature `window` (eframe/egui, glow; X11/Wayland): `cargo run -p gogamen-studio --features window --example studio_tour`. Without a display it prints `NOT_RUN`. |
| **labs MuJoCo** (optional) | `gogamen-lab-mujoco`: pure-Rust unicycle backend always; MuJoCo backend through a Python sidecar (`mujoco-sidecar` feature, off by default). | `cargo test -p gogamen-lab-mujoco --features mujoco-sidecar` | Needs `python3` + pip module `mujoco`; if missing the sidecar tests print `SKIPPED` and pass (the release script reports `NOT_RUN` when the module is missing). |
| **SDKs** (optional) | `python/` (stdlib-only JSON-Lines client of `gogamen serve-protocol`), `kotlin/` (JVM DSL that emits canonical `SystemSpec` JSON). | see `python/README.md`, `kotlin/README.md` | Neither is needed to run the engine. |

## Tour labs (what a new user can run offline)

The Studio tour registers six labs: **Grid, CartPole, Kart (racing), Life, Language World, Automation**
(plus a Replay session and a Robot3D view). Offline and headless:

```sh
export CARGO_TARGET_DIR=/path/to/target     # optional
cargo build --release -p gogamen-cli
./target/release/gogamen run-grid --ticks 20 --seed 42          # Grid
./target/release/gogamen run-language-world --seed 42           # Language World
cargo run -p gogamen-studio --offline --example cartpole_svg -- 20 42 > cartpole.svg
cargo run -p gogamen-studio --offline --example kart_svg -- 20 42 > kart.svg
cargo run -p gogamen-studio --offline --example life_svg -- 20 42 > life.svg
cargo run -p gogamen-studio --offline --example automation_svg -- incident > automation.svg
cargo run -p gogamen-runtime --offline --example multirate_sim    # simulated multirate control
```

Only Grid and Language World have `gogamen` CLI subcommands; the other labs are exposed through the Studio
crate (SVG examples headless, window with the `window` feature). The CLI also runs generic specs:
`run-spec`, `compile-artifact`, `run-artifact`.

### Kotlin without a JVM

The Kotlin DSL is an **authoring** frontend: it is not needed at run time. The Language World spec authored in
Kotlin is dumped to `crates/gogamen-spec/fixtures/language_world_canonical_v0.json` (the Kotlin test
`EpistemicDslTest` checks byte equality; the Rust test `language_world_kotlin_golden` checks the file equals the
Rust fixture and replays deterministically). The JVM-free path is therefore:

```sh
./target/release/gogamen run-spec --json-file crates/gogamen-spec/fixtures/language_world_canonical_v0.json --ticks 10 --seed 42
./target/release/gogamen compile-artifact --json-file <spec.json> --out spec.artifact && ./target/release/gogamen run-artifact --artifact-file spec.artifact
```

**Limitation (declared, not faked):** the Kotlin DSL itself still compiles and runs on the **JVM** only
(`kotlin("jvm")` with Gradle). Kotlin/Native is not configured and its toolchain (`~/.konan`, `konanc`) is not
available in the validated environment, and Gradle runs offline there, so a native Kotlin binary was not
attempted. What runs without a JVM is the Kotlin-*generated* JSON, not the Kotlin compiler/DSL.

## Release check (local CI)

```sh
scripts/release_check.sh                    # everything below
scripts/release_check.sh --quick            # skip the per-crate tests
scripts/release_check.sh --skip-visual --skip-kotlin --with-mujoco
```

It prints `PASS` / `FAIL` / `NOT_RUN` per step with log paths and exits 1 on any `FAIL`; `NOT_RUN` is never a
pass. Steps: `cargo fmt --check`; clippy `-D warnings` and tests over an explicit crate list (never
`--workspace`); core/cli dependency boundary; `cargo deny` (`deny.toml`: licenses, bans, sources, advisories
from the local RustSec cache); release CLI built and run from a clean temp dir (Grid, Language World, spec,
Kotlin golden JSON, compile/run artifact); the six labs headless + Studio tour contracts; simulated multirate
control; external conformance (SDK, third-party extension, Python JSON-Lines client); temporal metrics and
golden fixtures; Kotlin DSL gradle tests; optional MuJoCo sidecar; real-window Studio tour smoke
(`tests/tour_real.rs`: `NOT_RUN` without `DISPLAY`/`WAYLAND_DISPLAY`); `cargo package --list`.

Scientific/test failures are reported as-is; do not edit expectations to turn a step green.
`deny.toml` ignores only `RUSTSEC-2024-0436` (`paste` unmaintained, transitive of the optional rerun exporter,
not a vulnerability); the advisory DB is the local cache and may be stale.

## Safety and time limits

- **SIMULATION_ONLY.** No crate talks to hardware; `labs/mujoco` fixes `failsafe::SIMULATION_ONLY = true`.
  This is not a safety certification (no SIL/ISO 13849/IEC 61508 claim).
- **Simulated time is not hard real time.** Multirate control periods (1/10/50 ms) are illustrative; the demo
  uses a virtual clock and the runtime trace carries the notice that they are not hard real-time guarantees.
  Linux RT profile code only measures latency; it does not certify anything.
- Planner/cognition work is best effort; the control path never waits for it and falls back to the safe action.

## Compatibility (validated on the release machine)

| Component | Version |
|---|---|
| Rust | `rust-toolchain.toml`: `1.99.0` (rustfmt, clippy), edition 2024 |
| Studio window | `eframe =0.37.0` (glow, X11/Wayland), optional |
| MuJoCo (python sidecar) | `mujoco` 4.0.1, Python 3.10+ (SDK is stdlib-only) |
| Kotlin DSL | Kotlin 2.4.10, Gradle 8.14.5, JDK Temurin 21 (`kotlin/.tool-versions`; JDK 25 was also present as default `java`) |
| Tools | `cargo-deny` 0.20.x |

## Local packaging (no publication)

```sh
cargo build --release --offline -p gogamen-cli      # binary: target/release/gogamen
cargo package --list --offline --allow-dirty -p gogamen-cli   # contents only; nothing is uploaded
```

`cargo publish` is intentionally not used and not scripted. Manifests have no `description`/`repository`
metadata and path dependencies carry no versions, so a registry upload would not be accepted as-is.
