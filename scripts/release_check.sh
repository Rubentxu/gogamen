#!/usr/bin/env bash
# Gogamen pre-1.0 local release check. LOCAL ONLY: it never publishes, pushes or uploads anything.
#
# Usage: scripts/release_check.sh [--quick] [--skip-visual] [--skip-kotlin] [--with-mujoco] [--only STEP]
#   --quick        skip the per-crate test step (everything else still runs)
#   --skip-visual  do not run the real-window Studio tour (reported NOT_RUN)
#   --skip-kotlin  do not run the Kotlin DSL gradle tests (reported NOT_RUN)
#   --with-mujoco  also run the MuJoCo python sidecar tests (reported NOT_RUN if module missing)
#
# Each step prints PASS / FAIL / NOT_RUN. NOT_RUN is never a pass. Scientific/test failures are
# reported as FAIL with the log path; nothing is retried or hidden. Exit code 1 if any FAIL.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${CARGO_TARGET_DIR:=$ROOT/target}"
export CARGO_TARGET_DIR
CARGO="cargo"
OFFLINE="--offline"
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/gogamen-release-check.XXXXXX")"

QUICK=0; SKIP_VISUAL=0; SKIP_KOTLIN=0; WITH_MUJOCO=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --skip-visual) SKIP_VISUAL=1 ;;
    --skip-kotlin) SKIP_KOTLIN=1 ;;
    --with-mujoco) WITH_MUJOCO=1 ;;
    -h|--help) sed -n 2,12p "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

# Crates covered by the per-crate test/clippy gates (explicit list, never `--workspace`).
CORE_CRATES=(gogamen-core gogamen-spec gogamen-compiler gogamen-runtime gogamen-projection gogamen-sdk gogamen-arena gogamen-embedded gogamen-cli)
LAB_CRATES=(gogamen-lab-grid gogamen-lab-cartpole gogamen-lab-life gogamen-lab-racing gogamen-lab-automation gogamen-lab-retro gogamen-example-third-party)
STUDIO_CRATES=(gogamen-studio)
pflags() { local f=""; for c in "$@"; do f="$f -p $c"; done; echo "$f"; }

RESULTS=()
FAILS=0
record() { # name status detail
  RESULTS+=("$(printf '%-10s %-34s %s' "$2" "$1" "$3")")
  [ "$2" = FAIL ] && FAILS=$((FAILS + 1))
  echo ">>> [$2] $1 $3"
}
run_step() { # name cmd... ; log in LOGDIR
  local name="$1"; shift
  local log="$LOGDIR/${name//[^A-Za-z0-9]/_}.log"
  echo "--- $name: $*"
  if "$@" >"$log" 2>&1; then record "$name" PASS "(log: $log)"; else record "$name" FAIL "(log: $log; tail: $(tail -n 3 "$log" | tr '\n' ' '))"; fi
}
have() { command -v "$1" >/dev/null 2>&1; }

echo "Gogamen release check - root=$ROOT target=$CARGO_TARGET_DIR logs=$LOGDIR"
echo "rust-toolchain: $(grep channel rust-toolchain.toml | tr -d ' ') / rustc: $(rustc --version)"

# 1. Format and lints over the release crates.
run_step "fmt" $CARGO fmt --all -- --check
# shellcheck disable=SC2046
run_step "clippy core+labs+studio" $CARGO clippy $OFFLINE $(pflags "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}") --all-targets -- -D warnings

# 2. Dependency boundary: core and cli must stay free of UI/viewer/runtime-heavy deps.
BOUNDARY_LOG="$LOGDIR/boundary.log"
$CARGO tree -p gogamen-core -p gogamen-cli $OFFLINE >"$BOUNDARY_LOG.tree" 2>&1
if grep -Eiq 'studio|bevy|egui|eframe|winit|wasmtime|rerun|mujoco' "$BOUNDARY_LOG.tree"; then
  record "core/cli dependency boundary" FAIL "(forbidden crate found, see $BOUNDARY_LOG.tree)"
elif [ ! -s "$BOUNDARY_LOG.tree" ] || grep -q '^error' "$BOUNDARY_LOG.tree"; then
  record "core/cli dependency boundary" FAIL "(cargo tree failed, see $BOUNDARY_LOG.tree)"
else
  record "core/cli dependency boundary" PASS "(no studio/bevy/egui/eframe/winit/wasmtime/rerun/mujoco)"
fi

# 3. Licenses, bans, sources, advisories (deny.toml). Advisories use the local RustSec DB cache.
if have cargo-deny; then
  run_step "cargo-deny licenses/bans/sources" cargo deny --offline check licenses bans sources
  if cargo deny --offline check advisories >"$LOGDIR/deny_advisories.log" 2>&1; then
    record "cargo-deny advisories" PASS "(local advisory DB cache, may be stale; log: $LOGDIR/deny_advisories.log)"
  elif grep -qiE 'failed to (fetch|open|load)|network|could not resolve|no such file' "$LOGDIR/deny_advisories.log"; then
    record "cargo-deny advisories" NOT_RUN "(advisory DB unavailable offline; log: $LOGDIR/deny_advisories.log)"
  else
    record "cargo-deny advisories" FAIL "(log: $LOGDIR/deny_advisories.log)"
  fi
else
  record "cargo-deny" NOT_RUN "(cargo-deny not installed)"
fi

# 4. Per-crate tests (explicit crate list; no workspace-wide run).
if [ "$QUICK" = 1 ]; then
  record "tests core+labs+studio" NOT_RUN "(--quick)"
else
  # shellcheck disable=SC2046
  run_step "tests core+labs+studio" $CARGO test $OFFLINE $(pflags "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}")
fi

# 5. Release CLI binary + offline labs from a clean temp dir.
run_step "build release gogamen-cli" $CARGO build --release $OFFLINE -p gogamen-cli
BIN="$CARGO_TARGET_DIR/release/gogamen"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/gogamen-clean.XXXXXX")"
GOLDEN="$ROOT/crates/gogamen-spec/fixtures/language_world_canonical_v0.json"
if [ -x "$BIN" ]; then
  cp "$BIN" "$WORK/gogamen"
  GOLDEN_COPY="$WORK/language_world_canonical_v0.json"; cp "$GOLDEN" "$GOLDEN_COPY"
  cd "$WORK" || exit 2
  run_step "cli: run-grid (Grid lab)" ./gogamen run-grid --ticks 20 --seed 42
  run_step "cli: run-language-world" ./gogamen run-language-world --seed 42
  run_step "cli: run-spec fixture" ./gogamen run-spec --fixture --ticks 10 --seed 42
  # Kotlin-authored SystemSpec (golden JSON dumped from the Kotlin DSL) runs WITHOUT a JVM.
  run_step "cli: run-spec Kotlin golden (no JVM)" ./gogamen run-spec --json-file "$GOLDEN_COPY" --ticks 10 --seed 42
  run_step "cli: compile-artifact" ./gogamen compile-artifact --json-file "$GOLDEN_COPY" --out "$WORK/lw.artifact"
  run_step "cli: run-artifact" ./gogamen run-artifact --artifact-file "$WORK/lw.artifact" --ticks 10 --seed 42
  cd "$ROOT" || exit 2
else
  record "cli offline labs" FAIL "(release binary missing: $BIN)"
fi

# 6. The six tour labs headless (std-only SVG renders, no display needed) + multirate control.
SVG_DIR="$WORK/svg"; mkdir -p "$SVG_DIR"
svg_lab() { # lab example [args...]
  local lab="$1" ex="$2"; shift 2
  local out="$SVG_DIR/$lab.svg"
  if $CARGO run -q $OFFLINE -p gogamen-studio --example "$ex" -- "$@" >"$out" 2>"$LOGDIR/svg_$lab.err" && head -c 400 "$out" | grep -q '<svg'; then
    record "lab headless: $lab" PASS "($(wc -c <"$out") bytes SVG)"
  else
    record "lab headless: $lab" FAIL "(see $LOGDIR/svg_$lab.err)"
  fi
}
svg_lab grid grid_svg 5 42
svg_lab cartpole cartpole_svg 20 42
svg_lab kart kart_svg 20 42
svg_lab life life_svg 20 42
svg_lab language_world language_world_svg agent true
svg_lab automation automation_svg incident
run_step "studio tour headless contracts" $CARGO test $OFFLINE -p gogamen-studio --lib tour
MULTI_LOG="$LOGDIR/multirate_sim.log"
if $CARGO run -q $OFFLINE -p gogamen-runtime --example multirate_sim >"$MULTI_LOG" 2>&1 && grep -q 'SIMULATION_ONLY' "$MULTI_LOG"; then
  record "multirate control (simulated)" PASS "($(grep -E '^fallbacks:' "$MULTI_LOG"); virtual clock, not hard real-time)"
else
  record "multirate control (simulated)" FAIL "(log: $MULTI_LOG)"
fi

# 7. External conformance: SDK/extension/third-party consumers + python JSON-Lines client.
run_step "conformance: sdk+third-party" $CARGO test $OFFLINE -p gogamen-sdk --test capability_conformance --test external_consumer --test external_extension -p gogamen-example-third-party --test third_party
if have python3; then
  run_step "conformance: python SDK" env GOGAMEN_BIN="$BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/python" python3 -m unittest discover -s "$ROOT/python/tests"
else
  record "conformance: python SDK" NOT_RUN "(python3 missing)"
fi

# 8. Temporal metrics + fixtures (virtual-clock telemetry, golden JSON, frontend equivalence).
run_step "temporal metrics (multirate/telemetry)" $CARGO test $OFFLINE -p gogamen-runtime --lib -- multirate temporal_telemetry
run_step "fixtures: golden + equivalence" $CARGO test $OFFLINE -p gogamen-runtime --test language_world_kotlin_golden --test frontend_equivalence
run_step "fixtures: spec/projection json" $CARGO test $OFFLINE -p gogamen-spec --test json_public -p gogamen-projection --test viz_spec_tests

# 9. Kotlin DSL (JVM). The runtime path does not need it (step 5 runs the dumped golden JSON).
if [ "$SKIP_KOTLIN" = 1 ]; then
  record "kotlin DSL gradle tests" NOT_RUN "(--skip-kotlin)"
elif have gradle && have java; then
  run_step "kotlin DSL gradle tests" bash -c "cd '$ROOT/kotlin' && gradle test --offline"
else
  record "kotlin DSL gradle tests" NOT_RUN "(gradle/java missing)"
fi

# 10. Optional MuJoCo sidecar.
if [ "$WITH_MUJOCO" = 1 ]; then
  if have python3 && python3 -c 'import mujoco' 2>/dev/null; then
    run_step "mujoco sidecar tests" $CARGO test $OFFLINE -p gogamen-lab-mujoco --features mujoco-sidecar
  else
    record "mujoco sidecar tests" NOT_RUN "(python3 module 'mujoco' missing)"
  fi
else
  record "mujoco sidecar tests" NOT_RUN "(optional; use --with-mujoco)"
fi

# 11. Visual smoke of the Studio tour in a real window (feature `window`). PASS only if it ran.
if [ "$SKIP_VISUAL" = 1 ]; then
  record "studio tour_real (visual)" NOT_RUN "(--skip-visual)"
elif [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
  record "studio tour_real (visual)" NOT_RUN "(no DISPLAY/WAYLAND_DISPLAY)"
else
  VLOG="$LOGDIR/tour_real.log"
  if GOGAMEN_STUDIO_CAPTURE_DIR="$WORK/tour" $CARGO test $OFFLINE -p gogamen-studio --features window --test tour_real -- --nocapture >"$VLOG" 2>&1; then
    if grep -q '^PASS: real window tour' "$VLOG"; then record "studio tour_real (visual)" PASS "($(grep '^PASS:' "$VLOG"))"
    elif grep -q 'NOT_RUN' "$VLOG"; then record "studio tour_real (visual)" NOT_RUN "($(grep NOT_RUN "$VLOG" | head -1))"
    else record "studio tour_real (visual)" FAIL "(exit 0 but no PASS line; log: $VLOG)"; fi
  else
    record "studio tour_real (visual)" FAIL "(log: $VLOG)"
  fi
fi

# 12. Local package contents (no publish): `cargo package --list` only.
run_step "package list gogamen-cli" $CARGO package --list $OFFLINE --allow-dirty -p gogamen-cli

echo
echo "================ SUMMARY ================"
printf '%s\n' "${RESULTS[@]}"
echo "logs: $LOGDIR   clean workdir: $WORK"
if [ "$FAILS" -gt 0 ]; then echo "RESULT: FAIL ($FAILS failing step(s)); NOT_RUN steps are not passes"; exit 1; fi
echo "RESULT: no failing step (NOT_RUN steps above were NOT verified)"
