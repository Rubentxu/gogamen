#!/usr/bin/env python3
"""GOG-145: p50/p95/min/max per gate over N timing samples. Stdlib only.

  gate_stats.py FILE.tsv [FILE.tsv ...]   (TSV: gate<TAB>status<TAB>class<TAB>seconds, as written by
                                           release_check.sh / measure_gates.sh)
Only PASS rows with a numeric duration are samples. NOT_RUN / FAIL rows are counted separately and never
mixed into the percentiles (a skipped gate is not a fast pass). Percentile = nearest-rank (no interpolation).
"""
import math, sys, collections

def pct(xs, p):
    xs = sorted(xs); return xs[max(0, math.ceil(p / 100 * len(xs)) - 1)]

def main(files):
    if not files:
        print(__doc__, file=sys.stderr); return 2
    S, other = collections.OrderedDict(), collections.defaultdict(lambda: collections.Counter())
    for f in files:
        for i, line in enumerate(open(f)):
            c = line.rstrip("\n").split("\t")
            if i == 0 and c[:1] == ["gate"] or len(c) < 4:
                continue
            name, st, _, sec = c[:4]
            if st == "PASS":
                try: S.setdefault(name, []).append(float(sec)); continue
                except ValueError: pass
            other[name][st] += 1
    print(f"{'gate':<44} {'n':>3} {'p50_s':>8} {'p95_s':>8} {'min_s':>8} {'max_s':>8}  non-PASS")
    for n in list(S) + [k for k in other if k not in S]:
        xs = S.get(n)
        np = ",".join(f"{k}={v}" for k, v in other[n].items()) or "-"
        if xs: print(f"{n:<44} {len(xs):>3} {pct(xs,50):>8.3f} {pct(xs,95):>8.3f} {min(xs):>8.3f} {max(xs):>8.3f}  {np}")
        else: print(f"{n:<44} {0:>3} {'NA':>8} {'NA':>8} {'NA':>8} {'NA':>8}  {np}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
