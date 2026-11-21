#!/usr/bin/env bash
# Gogamen pre-1.0 local release check. LOCAL ONLY: it never publishes, pushes or uploads anything.
#
# Usage: scripts/release_check.sh [--fast] [--quick] [--skip-visual] [--skip-kotlin] [--with-mujoco] [--only STEP]
#   --fast         dev smoke (fmt, dependency boundary, cargo check of CRITICAL crates, studio tour contracts).
#                  Every other gate is reported NOT_RUN. It is NOT a release certification: prints the banner
#                  "FAST MODE - NOT A RELEASE GATE" and exits 0 (all run gates ok) or 1 (a gate FAILED); it never
#                  exits 0 with a "release" verdict.
#   --quick        skip the per-crate test step (required gate => NOT_RUN => exit 3)
#   --skip-visual  do not run the real-window Studio tour (OPTIONAL gate, reported NOT_RUN)
#   --skip-kotlin  do not run the Kotlin DSL gradle tests (REQUIRED gate => NOT_RUN => exit 3)
#   --with-mujoco  also run the MuJoCo python sidecar tests (OPTIONAL gate)
#
# Gates are REQUIRED unless labelled OPTIONAL in the summary. NOT_RUN is never a pass.
# Exit codes: 0 = release check passed (every REQUIRED gate PASS; OPTIONAL NOT_RUN listed in the summary)
#             1 = at least one gate FAIL (also in --fast)
#             3 = no FAIL but at least one REQUIRED gate NOT_RUN (not a release certification)
#             2 = usage error;  0 with the FAST banner = dev smoke only.
# Crate coverage: explicit lists below, reconciled against `cargo metadata --offline --no-deps`; a workspace
# crate that is not classified as CRITICAL / OPTIONAL / EXCLUDED (with justification) is a FAIL.
# Smoke for devs:  scripts/release_check.sh --fast      Self-test of this script: scripts/tests/release_check_selftest.sh
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
: "${CARGO_TARGET_DIR:=$ROOT/target}"
export CARGO_TARGET_DIR
CARGO="cargo"
OFFLINE="--offline"
LOGDIR="$(mktemp -d "${TMPDIR:-/tmp}/gogamen-release-check.XXXXXX")"

FAST=0; QUICK=0; SKIP_VISUAL=0; SKIP_KOTLIN=0; WITH_MUJOCO=0
for a in "$@"; do
  case "$a" in
    --fast) FAST=1 ;;
    --quick) QUICK=1 ;;
    --skip-visual) SKIP_VISUAL=1 ;;
    --skip-kotlin) SKIP_KOTLIN=1 ;;
    --with-mujoco) WITH_MUJOCO=1 ;;
    -h|--help) sed -n 2,19p "$0"; exit 0 ;;
    *) echo "unknown argument: $a" >&2; exit 2 ;;
  esac
done

# Crates covered by the per-crate test/clippy gates (explicit list, never `--workspace`).
CORE_CRATES=(gogamen-core gogamen-spec gogamen-compiler gogamen-runtime gogamen-projection gogamen-sdk gogamen-arena gogamen-embedded gogamen-cli)
LAB_CRATES=(gogamen-lab-grid gogamen-lab-cartpole gogamen-lab-life gogamen-lab-racing gogamen-lab-automation gogamen-lab-retro gogamen-example-third-party)
STUDIO_CRATES=(gogamen-studio)
# Not release-gated; NOT_RUN of their gates never fails the release. mujoco needs a python sidecar (--with-mujoco).
OPTIONAL_CRATES=(gogamen-lab-mujoco)
# Deliberately outside the release gates (justification required; keep in sync with docs/RELEASE.md).
EXCLUDED_CRATES=(
  "gogamen-experimental:experimental/unstable surface, not part of the pre-1.0 release contract"
  "gogamen-jev:not in the release MVP scope"
  "gogamen-rerun-export:heavy viewer dependency, outside the core/cli boundary"
  "gogamen-lab-mario:not one of the six tour labs"
  "gogamen-lab-ros2:needs a ROS2 toolchain; covered by scripts/ros2_contract_smoke.sh"
)
pflags() { local f=""; for c in "$@"; do f="$f -p $c"; done; echo "$f"; }

RESULTS=()
FAILS=0
REQ_NOT_RUN=0
OPT_NOT_RUN=0
GATE_CLASS=REQUIRED   # set to OPTIONAL around optional gates (always reset afterwards)
now() { date +%s.%N; }
LAST_TS="$(now)"
TIMINGS="$LOGDIR/timings.tsv"   # GOG-145: one row per gate: name<TAB>status<TAB>class<TAB>seconds (wall time since previous gate ended)
printf 'gate\tstatus\tclass\tseconds\n' >"$TIMINGS"
record() { # name status detail
  local cls="$GATE_CLASS" t; t="$(now)"
  # NOT_RUN gates did no work: duration NA so they never look like measured passes.
  if [ "$2" = NOT_RUN ]; then printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$cls" "NA" >>"$TIMINGS"
  else printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$cls" "$(awk -v a="$LAST_TS" -v b="$t" 'BEGIN{printf "%.3f", b-a}')" >>"$TIMINGS"; fi
  LAST_TS="$t"
  RESULTS+=("$(printf '%-9s %-8s %-34s %s' "$2" "$cls" "$1" "$3")")
  [ "$2" = FAIL ] && FAILS=$((FAILS + 1))
  if [ "$2" = NOT_RUN ]; then
    if [ "$cls" = REQUIRED ]; then REQ_NOT_RUN=$((REQ_NOT_RUN + 1)); else OPT_NOT_RUN=$((OPT_NOT_RUN + 1)); fi
  fi
  echo ">>> [$2] ($cls) $1 $3"
}
skip_fast() { record "$1" NOT_RUN "(--fast)"; }
run_step() { # name cmd... ; log in LOGDIR
  local name="$1"; shift
  local log="$LOGDIR/${name//[^A-Za-z0-9]/_}.log"
  echo "--- $name: $*"
  if "$@" >"$log" 2>&1; then record "$name" PASS "(log: $log)"; else record "$name" FAIL "(log: $log; tail: $(tail -n 3 "$log" | tr '\n' ' '))"; fi
}
workspace_crates() {
  if have jq; then "$CARGO" metadata $OFFLINE --no-deps --format-version 1 2>/dev/null | jq -r '.packages[].name'
  else "$CARGO" metadata $OFFLINE --no-deps --format-version 1 2>/dev/null | python3 -c 'import json,sys;[print(p["name"]) for p in json.load(sys.stdin)["packages"]]'; fi
}
have() { command -v "$1" >/dev/null 2>&1; }

echo "Gogamen release check - root=$ROOT target=$CARGO_TARGET_DIR logs=$LOGDIR"
echo "rust-toolchain: $(grep channel rust-toolchain.toml | tr -d ' ') / rustc: $(rustc --version)"

# 0. Crate manifest: every workspace crate must be classified (no silent omissions).
MANIFEST_LOG="$LOGDIR/crate_manifest.log"
WS_LIST="$(workspace_crates)"
if [ -z "$WS_LIST" ]; then
  record "crate manifest vs cargo metadata" FAIL "(cargo metadata failed/empty)"
else
  declare -A KNOWN=()
  for c in "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}" "${OPTIONAL_CRATES[@]}"; do KNOWN[$c]=1; done
  for e in "${EXCLUDED_CRATES[@]}"; do KNOWN[${e%%:*}]=1; done
  : >"$MANIFEST_LOG"
  while IFS= read -r c; do [ -n "${KNOWN[$c]:-}" ] || echo "unclassified workspace crate: $c" >>"$MANIFEST_LOG"; done <<<"$WS_LIST"
  for c in "${!KNOWN[@]}"; do grep -qx "$c" <<<"$WS_LIST" || echo "listed but not in workspace: $c" >>"$MANIFEST_LOG"; done
  if [ -s "$MANIFEST_LOG" ]; then record "crate manifest vs cargo metadata" FAIL "($(tr '\n' ';' <"$MANIFEST_LOG"))"
  else record "crate manifest vs cargo metadata" PASS "(all $(wc -l <<<"$WS_LIST") workspace crates classified)"; fi
fi

# 1. Format and lints over the release crates.
run_step "fmt" $CARGO fmt --all -- --check
if [ "$FAST" = 1 ]; then
  # shellcheck disable=SC2046
  run_step "check core+labs+studio (fast)" $CARGO check $OFFLINE $(pflags "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}") --all-targets
  skip_fast "clippy core+labs+studio"
else
# shellcheck disable=SC2046
run_step "clippy core+labs+studio" $CARGO clippy $OFFLINE $(pflags "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}") --all-targets -- -D warnings
fi

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

summary() {
  echo
  echo "================ SUMMARY ================"
  [ "$FAST" = 1 ] && echo "*** FAST MODE - NOT A RELEASE GATE (dev smoke only; skipped gates are NOT_RUN) ***"
  printf '%-9s %-8s %-34s %s\n' STATUS CLASS GATE DETAIL
  printf '%s\n' "${RESULTS[@]}"
  echo "timings: $TIMINGS (see scripts/gate_stats.py)"
  echo "logs: $LOGDIR   clean workdir: ${WORK:-n/a}"
  echo "REQUIRED NOT_RUN: $REQ_NOT_RUN   OPTIONAL NOT_RUN: $OPT_NOT_RUN (optional ones never fail the check)"
  if [ "$FAILS" -gt 0 ]; then echo "RESULT: FAIL ($FAILS failing gate(s)); NOT_RUN gates are not passes"; exit 1; fi
  if [ "$FAST" = 1 ]; then echo "RESULT: FAST MODE - NOT A RELEASE GATE (no gate failed among those run; NOT a release certification)"; exit 0; fi
  if [ "$REQ_NOT_RUN" -gt 0 ]; then echo "RESULT: INCOMPLETE ($REQ_NOT_RUN REQUIRED gate(s) NOT_RUN): NOT a release certification"; exit 3; fi
  echo "RESULT: RELEASE CHECK PASSED (all REQUIRED gates PASS; $OPT_NOT_RUN OPTIONAL gate(s) NOT_RUN, see above)"
  exit 0
}

if [ "$FAST" = 1 ]; then
  run_step "studio tour headless contracts" $CARGO test $OFFLINE -p gogamen-studio --lib tour
  skip_fast "deny/tests/build/cli/svg/conformance/fixtures/kotlin/package"
  summary
fi

# 3. Licenses, bans, sources, advisories (deny.toml). Advisories use the local RustSec DB cache.
if have cargo-deny; then
  run_step "cargo-deny licenses/bans/sources" cargo deny --offline check licenses bans sources
  GATE_CLASS=OPTIONAL
  if cargo deny --offline check advisories >"$LOGDIR/deny_advisories.log" 2>&1; then
    record "cargo-deny advisories" PASS "(local advisory DB cache, may be stale; log: $LOGDIR/deny_advisories.log)"
  elif grep -qiE 'failed to (fetch|open|load)|network|could not resolve|no such file' "$LOGDIR/deny_advisories.log"; then
    record "cargo-deny advisories" NOT_RUN "(advisory DB unavailable offline; log: $LOGDIR/deny_advisories.log)"
  else
    record "cargo-deny advisories" FAIL "(log: $LOGDIR/deny_advisories.log)"
  fi
  GATE_CLASS=REQUIRED
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
run_step "conformance: sdk" $CARGO test $OFFLINE -p gogamen-sdk --test capability_conformance --test external_consumer --test external_extension
run_step "conformance: third-party" $CARGO test $OFFLINE -p gogamen-example-third-party --test third_party
if have python3; then
  run_step "conformance: python SDK" env GOGAMEN_BIN="$BIN" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/python" python3 -m unittest discover -s "$ROOT/python/tests"
else
  record "conformance: python SDK" NOT_RUN "(python3 missing)"
fi

# 8. Temporal metrics + fixtures (virtual-clock telemetry, golden JSON, frontend equivalence).
# One cargo invocation per name filter: the libtest filters after `--` are independent substrings.
run_step "temporal metrics: multirate" $CARGO test $OFFLINE -p gogamen-runtime --lib -- multirate
run_step "temporal metrics: telemetry" $CARGO test $OFFLINE -p gogamen-runtime --lib -- temporal_telemetry
run_step "fixtures: golden + equivalence" $CARGO test $OFFLINE -p gogamen-runtime --test language_world_kotlin_golden --test frontend_equivalence
run_step "fixtures: spec json" $CARGO test $OFFLINE -p gogamen-spec --test json_public
run_step "fixtures: projection viz" $CARGO test $OFFLINE -p gogamen-projection --test viz_spec_tests

# 9. Kotlin DSL (JVM). The runtime path does not need it (step 5 runs the dumped golden JSON).
if [ "$SKIP_KOTLIN" = 1 ]; then
  record "kotlin DSL gradle tests" NOT_RUN "(--skip-kotlin)"
elif have gradle && have java; then
  run_step "kotlin DSL gradle tests" bash -c "cd '$ROOT/kotlin' && gradle test --offline"
else
  record "kotlin DSL gradle tests" NOT_RUN "(gradle/java missing)"
fi

# 10. Optional MuJoCo sidecar.
GATE_CLASS=OPTIONAL
if [ "$WITH_MUJOCO" = 1 ]; then
  if have python3 && python3 -c 'import mujoco' 2>/dev/null; then
    run_step "mujoco sidecar tests" $CARGO test $OFFLINE -p gogamen-lab-mujoco --features mujoco-sidecar
  else
    record "mujoco sidecar tests" NOT_RUN "(python3 module 'mujoco' missing)"
  fi
else
  record "mujoco sidecar tests" NOT_RUN "(optional; use --with-mujoco)"
fi

GATE_CLASS=REQUIRED
# 11. Visual smoke of the Studio tour in a real window (feature `window`). PASS only if it ran. OPTIONAL: needs a display.
GATE_CLASS=OPTIONAL
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

GATE_CLASS=REQUIRED
# 12. Local package contents (no publish): `cargo package --list` only.
run_step "package list gogamen-cli" $CARGO package --list $OFFLINE --allow-dirty -p gogamen-cli

summary
