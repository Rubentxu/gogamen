"""Lanzador de barridos (GOG-194 AC2) sobre la frontera existente, sin dominio en Python.

El CLI `gogamen` no tiene subcomando de barrido. Lo más cercano que existe es el ejemplo
Rust de GOG-185 / DEM-028 `gogamen-arena/examples/sweep_cartpole`
(`cargo build -p gogamen-arena --example sweep_cartpole`), que ejecuta `run_sweep` y escribe
`sweep.csv` en el directorio dado. Sus ejes son fijos (compilados en el ejemplo): este
lanzador no los parametriza ni los reimplementa; solo lo ejecuta con timeout y devuelve el CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io
import os
from pathlib import Path
import re
import subprocess
import tempfile

from .errors import GogamenSweepError, GogamenTimeoutError
from .experiment_spec import fnv1a64

DEFAULT_SWEEP_TIMEOUT_S = 120.0
SWEEP_EXAMPLE = "sweep_cartpole"
_CSV_HASH_RE = re.compile(r"\bcsv_hash=([0-9a-f]{16})\b")


@dataclass(frozen=True)
class SweepResult:
    """CSV nativo tal cual (`csv_text`), filas como dicts de str y el hash que Rust declaró."""

    csv_text: str
    rows: tuple[dict[str, str], ...]
    csv_hash: str
    report: str


def resolve_sweep_binary(explicit_path: str | os.PathLike[str] | None = None) -> str:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    elif os.environ.get("GOGAMEN_SWEEP_BIN"):
        candidates.append(Path(os.environ["GOGAMEN_SWEEP_BIN"]))
    else:
        if os.environ.get("CARGO_TARGET_DIR"):
            base = Path(os.environ["CARGO_TARGET_DIR"])
            candidates += [base / p / "examples" / SWEEP_EXAMPLE for p in ("debug", "release")]
        root = Path(__file__).resolve().parents[2] / "target"
        candidates += [root / p / "examples" / SWEEP_EXAMPLE for p in ("debug", "release")]
    for c in candidates:
        if c.is_file():
            return str(c)
    raise GogamenSweepError(
        f"binario del barrido no encontrado ({candidates[0] if candidates else SWEEP_EXAMPLE}); "
        "compila con 'cargo build -p gogamen-arena --example sweep_cartpole' o define GOGAMEN_SWEEP_BIN",
        code="BinaryNotFound",
    )


def run_sweep(
    *,
    binary: str | os.PathLike[str] | None = None,
    timeout: float = DEFAULT_SWEEP_TIMEOUT_S,
) -> SweepResult:
    """Ejecuta el barrido nativo y devuelve su CSV; verifica el `csv_hash` declarado por Rust."""
    exe = resolve_sweep_binary(binary)
    with tempfile.TemporaryDirectory(prefix="gogamen-sweep-") as out_dir:
        try:
            proc = subprocess.run(
                [exe, out_dir],
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise GogamenTimeoutError(f"el barrido excedió el timeout de {timeout}s") from exc
        if proc.returncode != 0:
            raise GogamenSweepError(
                f"el barrido terminó con código {proc.returncode}: {proc.stderr.strip()[-2000:]}",
                code="SweepFailed",
            )
        csv_path = Path(out_dir) / "sweep.csv"
        try:
            data = csv_path.read_bytes()
        except OSError as exc:
            raise GogamenSweepError(f"no se pudo leer {csv_path}: {exc}", code="InvalidOutput") from exc
    m = _CSV_HASH_RE.search(proc.stdout)
    if not m:
        raise GogamenSweepError("la salida del barrido no declara csv_hash", code="InvalidOutput")
    actual = f"{fnv1a64(data):016x}"
    if actual != m.group(1):
        raise GogamenSweepError(
            f"csv_hash divergente: declarado={m.group(1)} calculado={actual}", code="CsvHashMismatch"
        )
    text = data.decode("utf-8")
    rows = tuple(csv.DictReader(io.StringIO(text)))
    return SweepResult(csv_text=text, rows=rows, csv_hash=actual, report=proc.stdout)
