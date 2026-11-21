#!/usr/bin/env bash
# GOG-145: lightweight, BOUNDED gate timing. LOCAL ONLY. Never runs the workspace suite.
# Usage: scripts/measure_gates.sh [-n SAMPLES] [-o OUT.tsv] [--] GATE...
#   GATE = "name::command"  e.g. "check core::cargo check --offline -p gogamen-core"
# With no GATE the default cheap set is used (check gogamen-core, test gogamen-spec --lib).
# Each sample is run sequentially on the current target dir state (WARM if already built; run one
# untimed warm-up first with -w). Clean baselines are NOT produced here (they need an empty CARGO_TARGET_DIR:
# set it yourself and pass -n 1). Output: TSV rows gate/status/class/seconds, then p50/p95 via gate_stats.py.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT" || exit 2
N=3; OUT=""; WARM=0
while [ $# -gt 0 ]; do
  case "$1" in -n) N="$2"; shift 2 ;; -o) OUT="$2"; shift 2 ;; -w) WARM=1; shift ;; --) shift; break ;; -h|--help) sed -n 2,9p "$0"; exit 0 ;; *) break ;; esac
done
[[ "$N" =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] && [ "$N" -le 10 ] || { echo "SAMPLES must be 1..10 (bounded by design)" >&2; exit 2; }
GATES=("$@")
[ ${#GATES[@]} -gt 0 ] || GATES=("cargo check gogamen-core::cargo check --offline -p gogamen-core" "cargo test gogamen-spec --lib::cargo test --offline -p gogamen-spec --lib")
: "${OUT:=$(mktemp "${TMPDIR:-/tmp}/gogamen-gate-timings.XXXXXX.tsv")}"
printf 'gate\tstatus\tclass\tseconds\n' >"$OUT"
for g in "${GATES[@]}"; do
  name="${g%%::*}"; cmd="${g#*::}"
  [ "$WARM" = 1 ] && bash -c "$cmd" >/dev/null 2>&1
  for ((i = 1; i <= N; i++)); do
    t0="$(date +%s.%N)"; bash -c "$cmd" >/dev/null 2>&1; rc=$?; t1="$(date +%s.%N)"
    st=PASS; [ "$rc" = 0 ] || st=FAIL
    printf '%s\t%s\tMEASURE\t%s\n' "$name" "$st" "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", b-a}')" >>"$OUT"
  done
done
echo "timings: $OUT"; python3 -I "$ROOT/scripts/gate_stats.py" "$OUT"
