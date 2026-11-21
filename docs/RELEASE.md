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

## CI (GitHub Actions, escalonado por riesgo)

No workflow publica nada (permisos `contents: read`). Todos usan acciones fijadas a version mayor, toolchain de
`rust-toolchain.toml` y caches con clave `toolchain + hash(Cargo.lock)` (`.github/actions/setup-rust`); cada job
escribe duracion en `$GITHUB_STEP_SUMMARY` y sube logs como artefacto si falla.

| Escalon | Disparador | Workflow | Contenido |
|---|---|---|---|
| PR | `pull_request` -> main (cancela ejecuciones obsoletas) | `.github/workflows/ci-pr.yml` | `fmt` + `clippy -D warnings` + `cargo test` solo de los crates afectados (`scripts/ci_affected.py`: `git diff base...head` -> crates + dependientes inversos; fallback a todo el workspace si cambian `Cargo.lock`, `Cargo.toml` raiz, `rust-toolchain.toml`, `deny.toml`, `scripts/`, `.github/`). Sin crates afectados se informa `NOT_RUN` (no PASS). Job final `gate`. |
| Merge | `push` a main | `.github/workflows/ci-main.yml` | `cargo test --workspace`, integracion core/cli (CLI release, labs, golden Kotlin sin JVM, conformance SDK, cliente Python), Kotlin `gradle test`, matriz de features, `gate`. |
| Release | tag `v*` o `workflow_dispatch` | `.github/workflows/release.yml` | selftest + `scripts/release_check.sh` integro (cualquier exit != 0 falla: 1 FAIL, 3 REQUIRED NOT_RUN), matriz de features, `gate`. |
| Features | reutilizable (main y release) | `.github/workflows/features.yml` | REQUIRED: `gogamen-jev` `http` y `https`, `gogamen-sdk` `wasm`, `gogamen-studio` `window` (clippy + tests lib; `tour_real` necesita display y no se cuenta como PASS). OPTIONAL (`continue-on-error`, fuera del gate): `gogamen-lab-mujoco` `mujoco-sidecar`. |

Un job omitido, cancelado o fallido nunca es PASS: cada `gate` (`if: always()`, `needs`) falla salvo que todos los
requeridos terminen en `success`.

Decisiones: `labs/ros2` (`gogamen-lab-ros2`) se excluye de fmt/clippy/tests de CI: tiene diferencias de rustfmt
preexistentes (no reformateadas aqui) y requiere ROS2 (`scripts/ros2_contract_smoke.sh`). Cuando se reformatee,
quitarlo de `EXCLUDED` en `scripts/ci_affected.py`. El gate Kotlin de release precalienta la cache de Gradle porque
`release_check.sh` usa `gradle --offline`. Estado: workflows validados con `actionlint` y ejecucion local de los
comandos clave; **no ejecutados en GitHub** (NOT_RUN).

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

## Medicion de gates (GOG-145: medir antes de fragmentar crates)

Hallazgo verificado: `release_check.sh` (modo completo) invoca `cargo clippy`, `cargo test` y varios `cargo test -p`
sobre las mismas listas de crates, sobre un `CARGO_TARGET_DIR` compartido; los perfiles `check`/`clippy`/`test`/`release`
generan artefactos distintos, de modo que hay recompilacion por perfil. Los filtros de tests de los pasos 7-8 son
distintos por invocacion (lo comprueba el selftest). El coste real solo se conoce midiendo.

Mecanismo (sin dependencias nuevas):

- `scripts/release_check.sh` escribe `$LOGDIR/timings.tsv` (`gate, status, class, seconds`; la ruta se imprime en el
  resumen). `seconds` es el tiempo de pared desde que acabo el gate anterior; las filas `NOT_RUN` llevan `NA`, nunca
  una duracion, y no entran en los percentiles.
- `scripts/measure_gates.sh [-n N<=10] [-w] [-o out.tsv] ["nombre::comando" ...]` mide gates baratos N veces
  (`-w` = un calentamiento sin medir). Nunca ejecuta el workspace completo.
- `scripts/gate_stats.py FILE.tsv ...` calcula n, p50, p95 (nearest-rank), min, max por gate; los no-PASS se cuentan aparte.
- En CI los jobs ya escriben duracion en `$GITHUB_STEP_SUMMARY`; para p50/p95 de CI, concatenar los TSV de
  `release_check.sh` de varias ejecuciones y pasarlos a `gate_stats.py` (sin meta de speedup arbitraria).

Baseline WARM medida (3 muestras, `CARGO_TARGET_DIR=/var/mnt/DiscoChino2-fast/cargo-targets`, `--offline`, maquina de
desarrollo, tiempos de pared en segundos):

| Gate (target ya compilado) | n | p50 | p95 | min | max |
|---|---|---|---|---|---|
| `cargo check -p gogamen-core` | 3 | 0.319 | 0.337 | 0.310 | 0.337 |
| `cargo test -p gogamen-spec --lib` | 3 | 0.321 | 0.324 | 0.313 | 0.324 |
| `cargo check -p gogamen-core -p gogamen-spec` | 3 | 0.311 | 0.319 | 0.305 | 0.319 |
| `cargo tree -p gogamen-core -p gogamen-cli` (frontera) | 3 | 0.247 | 0.251 | 0.241 | 0.251 |
| tras `touch` de `gogamen-spec/src/lib.rs`: `cargo check -p gogamen-spec` | 3 | 0.426 | 0.427 | 0.409 | 0.427 |
| tras `touch` de `gogamen-spec/src/lib.rs`: `cargo check -p gogamen-cli` (spec + dependientes) | 3 | 0.853 | 2.534 | 0.851 | 2.534 |

Son baselines no-op/incrementales pequenas: fijan el coste de arranque/fingerprint de cargo, no el de un release
completo. **NOT_RUN (explicito, no PASS):** baseline *clean* (target vacio; no viable/acotada aqui), baseline
Kotlin/`gradle test` (no medida en este encargo) y la suite completa `release_check.sh`/`cargo test --workspace`
(prohibida por coste). Se miden en CI/maquina de release con el TSV.

Smoke < completo sin perder gates (evidencia): (1) en el selftest con stubs `--fast` ejecuta 5 gates (PASS) frente a
32 del modo completo, y todo gate no ejecutado aparece como fila `NOT_RUN` (verificado, junto con que el TSV no da
duracion a los `NOT_RUN` y que `gate_stats.py` los excluye de los percentiles); `--fast` nunca emite veredicto de
release. (2) Seleccion por diff en PR con `scripts/ci_affected.py affected BASE HEAD` sobre commits reales:
`b5a7577` (solo `README.md`) -> 0 crates (el job informa `NOT_RUN`); `f59c648` -> 3 de 22 crates (afectados +
dependientes inversos); `8b47c6d` (cambia `scripts/`/`rust-toolchain.toml`) -> `ALL`. Cambios de infraestructura
vuelven siempre al conjunto completo, y `main`/release siguen corriendo el conjunto completo.

Comparacion y decision (solo se adopta mejora medida):

| Opcion | Evidencia medida | Decision |
|---|---|---|
| Incremental cargo por crate (`-p` del crate tocado) vs crate + dependientes | 0.426 s vs 0.853 s (p50) tras tocar `gogamen-spec`: ya proporcional al radio de impacto | Mantener (ya implementado con `ci_affected.py`) |
| Cache (`target`/registry en CI, clave toolchain+`Cargo.lock`) | Warm no-op ~0.3 s local; beneficio en CI no medido (no ejecutado en GitHub) | Sin cambio; medir con el TSV antes de tocar |
| Particion por responsabilidad (mas crates) | Ninguna medicion muestra que la compilacion incremental por crate sea cuello de botella (arranque ~0.3 s, check incremental < 1 s) | **NO cambiar la estructura de crates**; reabrir solo con p50/p95 de CI que lo justifiquen |

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
