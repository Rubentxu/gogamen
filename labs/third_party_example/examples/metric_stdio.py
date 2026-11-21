#!/usr/bin/env python3
"""Metrica de tercero fuera de proceso para `gogamen metric-kit` (GOG-191).

Protocolo GOGAMEN-METRIC/1: por stdin `EVAL <seed> <label>:<prob> ...`; por stdout
`OK <valor>` o `ERR <mensaje>`. Uso: `gogamen metric-kit -- python3 -I metric_stdio.py [--broken]`.
"""
import math
import sys

BROKEN = "--broken" in sys.argv[1:]

for line in sys.stdin:
    try:
        parts = line.split()
        assert parts[0] == "EVAL"
        probs = {}
        for tok in parts[2:]:
            label, p = tok.split(":")
            probs[int(label)] = float(p)
        if BROKEN:  # artefacto: bins por etiqueta (depende del nombre de los estados)
            binned = {}
            for l, p in probs.items():
                binned[l // 2] = binned.get(l // 2, 0.0) + p
            probs = binned
        h = sum(-p * math.log2(p) for p in probs.values() if p > 0)
        print(f"OK {h!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"ERR {e}", flush=True)
