#!/usr/bin/env bash
# Self-test for scripts/release_check.sh using stub tools (cargo, deny, gradle, java, python3) on PATH.
# No real build/test is executed. Usage: scripts/tests/release_check_selftest.sh
# Env knobs understood by the stub cargo: STUB_FAIL_MATCH (regex on args => exit 1),
# STUB_EXTRA_CRATE (adds an unclassified crate to `cargo metadata`).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SUT="$ROOT/scripts/release_check.sh"
command -v jq >/dev/null || { echo "self-test needs jq"; exit 2; }
T="$(mktemp -d "${TMPDIR:-/tmp}/rc-selftest.XXXXXX")"; STUBS="$T/bin"; mkdir -p "$STUBS"
trap 'rm -rf "$T"' EXIT

cat >"$STUBS/cargo" <<'STUB'
#!/usr/bin/env bash
echo "cargo $*" >>"$STUBLOG"
if [ -n "${STUB_FAIL_MATCH:-}" ] && [[ "$*" =~ $STUB_FAIL_MATCH ]]; then echo "stub failure"; exit 1; fi
case "$1" in
  metadata)
    names="$(sed -n 's/^\(.*\):.*/\1/p;t;s/.*/&/p' "$STUB_CRATES")"
    { echo -n '{"packages":['; first=1; for n in $names $STUB_EXTRA_CRATE; do [ $first = 1 ] || echo -n ,; first=0; echo -n "{\"name\":\"$n\"}"; done; echo ']}'; } ;;
  tree) echo gogamen-core ;;
  build) mkdir -p "$CARGO_TARGET_DIR/release"; printf '#!/bin/sh\nexit 0\n' >"$CARGO_TARGET_DIR/release/gogamen"; chmod +x "$CARGO_TARGET_DIR/release/gogamen" ;;
  run) echo "<svg SIMULATION_ONLY"; echo "fallbacks: 0" ;;
esac
exit 0
STUB
printf '#!/usr/bin/env bash\nexit 0\n' >"$STUBS/ok"; chmod +x "$STUBS/cargo" "$STUBS/ok"
for t in deny gradle java python3 cargo-deny; do cp "$STUBS/ok" "$STUBS/$t"; done
# cargo-deny is invoked as `cargo deny`; the stub cargo handles it (exit 0) and `have cargo-deny` finds the stub.
# Crates the stub reports = every crate the script classifies (read from the script's own arrays).
bash -c 'source <(sed -n "/^CORE_CRATES=/,/^pflags/p" "$1" | grep -v "^pflags"); printf "%s\n" "${CORE_CRATES[@]}" "${LAB_CRATES[@]}" "${STUDIO_CRATES[@]}" "${OPTIONAL_CRATES[@]}"; for e in "${EXCLUDED_CRATES[@]}"; do echo "${e%%:*}"; done' _ "$SUT" >"$T/crates.txt"
export STUB_CRATES="$T/crates.txt" STUBLOG="$T/cargo.log"

PASS=0; FAIL=0
run() { # label expected_exit grep_pattern args...   (env passes through)
  local label="$1" want="$2" pat="$3"; shift 3
  : >"$STUBLOG"
  local out; out="$(CARGO_TARGET_DIR="$T/target" PATH="$STUBS:$PATH" "$SUT" "$@" 2>&1)"; local rc=$?
  if [ "$rc" = "$want" ] && grep -Eq "$pat" <<<"$out"; then PASS=$((PASS+1)); echo "ok   - $label (exit=$rc)"
  else FAIL=$((FAIL+1)); echo "FAIL - $label (exit=$rc, want $want, pattern /$pat/)"; sed -n '/SUMMARY/,$p' <<<"$out" | tail -n 12; fi
  LAST_OUT="$out"
}

run "all gates stubbed green, optional NOT_RUN -> release exit 0 labelled OPTIONAL" 0 'RELEASE CHECK PASSED.*1 OPTIONAL|RELEASE CHECK PASSED' --skip-visual
if grep -Eq '^NOT_RUN +OPTIONAL +mujoco' <<<"$LAST_OUT"; then PASS=$((PASS+1)); echo "ok   - mujoco NOT_RUN is labelled OPTIONAL in summary"; else FAIL=$((FAIL+1)); echo "FAIL - mujoco OPTIONAL label"; fi
run "REQUIRED gate NOT_RUN (--skip-kotlin) -> exit 3" 3 'INCOMPLETE .*REQUIRED.*NOT_RUN' --skip-visual --skip-kotlin
run "REQUIRED gate NOT_RUN (--quick tests) -> exit 3" 3 'NOT_RUN +REQUIRED +tests core' --skip-visual --quick
STUB_FAIL_MATCH='^test .*gogamen-runtime --lib -- multirate' run "failing gate -> exit 1" 1 'RESULT: FAIL' --skip-visual
STUB_EXTRA_CRATE=gogamen-newcomer run "unclassified workspace crate -> exit 1" 1 'FAIL +REQUIRED +crate manifest.*gogamen-newcomer' --skip-visual
run "--fast prints not-a-release banner, exit 0" 0 'FAST MODE - NOT A RELEASE GATE' --fast
if grep -q 'RELEASE CHECK PASSED' <<<"$LAST_OUT"; then FAIL=$((FAIL+1)); echo "FAIL - --fast must never claim release"; else PASS=$((PASS+1)); echo "ok   - --fast never claims release"; fi
run "no cargo test carries two positional filters" 0 'RELEASE CHECK PASSED' --skip-visual
if grep -E '^cargo test' "$STUBLOG" | grep -Eq -- '-- [A-Za-z_]+ [A-Za-z_]+'; then FAIL=$((FAIL+1)); echo "FAIL - multi-filter cargo test invocation found"; else PASS=$((PASS+1)); echo "ok   - one filter per cargo test invocation"; fi

# GOG-145: timings TSV per gate; NOT_RUN rows carry NA (never a duration); --fast runs fewer gates than full and
# every gate it does not run is an explicit NOT_RUN row (nothing omitted counts as PASS).
run "full run for timing evidence" 0 'RELEASE CHECK PASSED' --skip-visual
FULL_OUT="$LAST_OUT"; TS_FULL="$(sed -n 's/^timings: \(.*\.tsv\) .*/\1/p' <<<"$FULL_OUT")"
if [ -s "$TS_FULL" ] && head -1 "$TS_FULL" | grep -q '^gate.status.class.seconds$' && awk -F'\t' 'NR>1 && $2=="PASS" && $4!~/^[0-9]+\.[0-9]+$/{bad=1} NR>1 && $2=="NOT_RUN" && $4!="NA"{bad=1} END{exit bad}' "$TS_FULL"; then PASS=$((PASS+1)); echo "ok   - timings.tsv: PASS rows numeric, NOT_RUN rows NA"; else FAIL=$((FAIL+1)); echo "FAIL - timings.tsv malformed ($TS_FULL)"; fi
run "fast run for timing evidence" 0 'FAST MODE' --fast
TS_FAST="$(sed -n 's/^timings: \(.*\.tsv\) .*/\1/p' <<<"$LAST_OUT")"
nf="$(awk -F'\t' 'NR>1 && $2=="PASS"' "$TS_FAST" | wc -l)"; nfull="$(awk -F'\t' 'NR>1 && $2=="PASS"' "$TS_FULL" | wc -l)"
if [ "$nf" -lt "$nfull" ] && awk -F'\t' 'NR>1 && $2=="NOT_RUN"{n++} END{exit !(n>=2)}' "$TS_FAST" && grep -q 'NOT_RUN .*deny/tests/build' <<<"$LAST_OUT"; then PASS=$((PASS+1)); echo "ok   - --fast runs fewer gates ($nf < $nfull PASS) and reports the rest NOT_RUN"; else FAIL=$((FAIL+1)); echo "FAIL - fast vs full gate selection ($nf vs $nfull)"; fi
printf 'gate\tstatus\tclass\tseconds\nx\tPASS\tR\t1.000\nx\tPASS\tR\t3.000\nx\tNOT_RUN\tR\tNA\ny\tNOT_RUN\tR\tNA\n' >"$T/s.tsv"
st="$(python3 -I "$ROOT/scripts/gate_stats.py" "$T/s.tsv" 2>&1)"
if grep -Eq '^x +2 +1\.000 +3\.000' <<<"$st" && grep -Eq '^y +0 +NA.*NOT_RUN=1' <<<"$st" && grep -q 'NOT_RUN=1' <<<"$st"; then PASS=$((PASS+1)); echo "ok   - gate_stats: NOT_RUN excluded from percentiles"; else FAIL=$((FAIL+1)); echo "FAIL - gate_stats"; echo "$st"; fi

echo "self-test: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
