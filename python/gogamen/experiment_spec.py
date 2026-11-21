"""Builders Python (stdlib-only) de experimentos `CognitiveSpec` (GOG-194 / DEM-041).

Contrato: el mismo JSON canónico y el mismo hash que Rust y Kotlin; sin IR nuevo.

- Forma canónica (réplica de `gogamen-spec::json`): objetos compactos con claves
  ordenadas por bytes UTF-8, escapes de `write_json_string`, enteros sin punto y
  `f64` con la notación `Display` de Rust (sin exponente) más `.0` si es entero.
- Hash: FNV-1a de 64 bits sobre los bytes UTF-8 del JSON canónico, en hex de 16
  dígitos (el mismo `spec_hash` que imprime `gogamen run-experiment --demo`).
- Python no valida el dominio: la validación semántica la hace el parser Rust
  mediante `validate_native` (subproceso con timeout explícito).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

from .client import resolve_gogamen_binary
from .errors import GogamenSpecError, GogamenTimeoutError

DEFAULT_NATIVE_TIMEOUT_S = 60.0
EXPERIMENT_SCHEMA = "gogamen.experiment_spec.v0.1"
AGENT_SCHEMA = "gogamen.adaptive_agent_spec.v0.1"
SYSTEM_SCHEMA = "gogamen.system_spec.v0"
COGNITIVE_SCHEMA = "gogamen.cognitive_spec.v0"

_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_MASK64 = 0xFFFFFFFFFFFFFFFF


# --------------------------------------------------------------------------- canonical JSON


def _escape(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\b":
            out.append("\\b")
        elif ch == "\f":
            out.append("\\f")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


class Expr(dict):  # type: ignore[type-arg]
    """Nodo de expresión: el writer nativo emite `op` primero y el resto ordenado.

    (`write_expr_json` en `gogamen-spec/src/expr.rs`; el resto de objetos va ordenado por clave.)
    """


def mark_exprs(document: Any) -> Any:
    """Marca como `Expr` las expresiones de un documento ya parseado (goals/constraints)."""

    def expr(node: Any) -> Any:
        if not isinstance(node, Mapping):
            return node
        return Expr({k: expr(v) if k in ("lhs", "rhs", "expr") else v for k, v in node.items()})

    def walk(node: Any) -> Any:
        if isinstance(node, Mapping):
            out = {k: walk(v) for k, v in node.items()}
            if "goal_id" in out and "expr" in out:
                out["expr"] = expr(node["expr"])
            if "constraint_id" in out and "condition" in out:
                out["condition"] = expr(node["condition"])
            return out
        if isinstance(node, (list, tuple)):
            return [walk(v) for v in node]
        return node

    return walk(document)


def _pointer_token(key: object) -> str:
    return str(key).replace("~", "~0").replace("/", "~1")


def format_f64(value: float, pointer: str = "") -> str:
    """`write_canonical_f64` de Rust: `Display` sin exponente y `.0` si es entero."""
    if not math.isfinite(value):
        raise GogamenSpecError(
            "non-finite float value", code="NonFiniteFloat", pointer=pointer or "/"
        )
    if value == 0.0:
        value = 0.0  # normaliza -0.0
    text = format(Decimal(repr(value)), "f")
    if "." not in text:
        text += ".0"
    return text


def _write(value: Any, out: list[str], pointer: str) -> None:
    if value is None:
        out.append("null")
    elif isinstance(value, bool):
        out.append("true" if value else "false")
    elif isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, float):
        out.append(format_f64(value, pointer))
    elif isinstance(value, str):
        out.append(_escape(value))
    elif isinstance(value, Mapping):
        for k in value:
            if not isinstance(k, str):
                raise GogamenSpecError(
                    f"object key must be a string, got {type(k).__name__}",
                    code="InvalidKey",
                    pointer=pointer or "/",
                )
        out.append("{")
        keys = sorted(value, key=lambda s: s.encode("utf-8"))
        if isinstance(value, Expr) and "op" in value:
            keys.remove("op")
            keys.insert(0, "op")
        for i, k in enumerate(keys):
            if i:
                out.append(",")
            out.append(_escape(k))
            out.append(":")
            _write(value[k], out, f"{pointer}/{_pointer_token(k)}")
        out.append("}")
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _write(item, out, f"{pointer}/{i}")
        out.append("]")
    else:
        raise GogamenSpecError(
            f"unsupported value of type {type(value).__name__}",
            code="UnsupportedType",
            pointer=pointer or "/",
        )


def canonical_json(value: Any) -> str:
    """JSON canónico (mismas reglas que `AdaptiveExperimentSpec::to_canonical_json`)."""
    out: list[str] = []
    _write(value, out, "")
    return "".join(out)


def fnv1a64(data: bytes) -> int:
    h = _FNV_OFFSET
    for b in data:
        h = ((h ^ b) * _FNV_PRIME) & _MASK64
    return h


def spec_hash(value: Any) -> str:
    """Hash FNV-1a 64 del JSON canónico, en hex de 16 dígitos."""
    return f"{fnv1a64(canonical_json(value).encode('utf-8')):016x}"


# --------------------------------------------------------------------------- builders
# Cada builder devuelve un dict JSON-compatible con los nombres de campo del contrato
# nativo. Los `f64` deben pasarse como `float` (0.25, 1.0): el entero 1 se emite como `1`.


def custom(name: str) -> dict[str, str]:
    return {"custom": name}


def port(port_id: str, direction: str, port_type: Any) -> dict[str, Any]:
    return {"id": port_id, "direction": direction, "port_type": port_type}


def node(
    node_id: str, kind: str, ports: Sequence[Mapping[str, Any]] = (), config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {"id": node_id, "kind": kind, "ports": list(ports), "config": dict(config or {})}


def connection(
    from_node: str, from_port: str, to_node: str, to_port: str, delay_ticks: int = 0
) -> dict[str, Any]:
    return {
        "from_node": from_node,
        "from_port": from_port,
        "to_node": to_node,
        "to_port": to_port,
        "delay_ticks": delay_ticks,
    }


def system(
    spec_id: str,
    nodes: Sequence[Mapping[str, Any]],
    connections: Sequence[Mapping[str, Any]] = (),
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SYSTEM_SCHEMA,
        "spec_id": spec_id,
        "nodes": list(nodes),
        "connections": list(connections),
        "config": dict(config or {}),
    }


def var(name: str) -> dict[str, Any]:
    return Expr({"op": "var", "name": name})


def const(value: float, unit: str = "dimensionless") -> dict[str, Any]:
    return Expr({"op": "const_f64", "value": float(value), "unit": unit})


def add(lhs: Any, rhs: Any) -> dict[str, Any]:
    return Expr({"op": "add", "lhs": lhs, "rhs": rhs})


def mul(lhs: Any, rhs: Any) -> dict[str, Any]:
    return Expr({"op": "mul", "lhs": lhs, "rhs": rhs})


def gt(lhs: Any, rhs: Any) -> dict[str, Any]:
    return Expr({"op": "gt", "lhs": lhs, "rhs": rhs})


def sensor(
    sensor_id: str,
    kind: str,
    source_node: str,
    source_port: str,
    port_type: Any,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "sensor_id": sensor_id,
        "kind": kind,
        "source_node": source_node,
        "source_port": source_port,
        "port_type": port_type,
        "config": dict(config or {}),
    }


def model(model_id: str, kind: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"model_id": model_id, "kind": kind, "config": dict(config or {})}


def goal(goal_id: str, maximize: bool, expr: Any, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"goal_id": goal_id, "maximize": maximize, "expr": expr, "config": dict(config or {})}


def constraint(
    constraint_id: str, hard: bool, condition: Any, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "constraint_id": constraint_id,
        "hard": hard,
        "condition": condition,
        "config": dict(config or {}),
    }


def cognitive(
    *,
    binning_version: str,
    levels: Sequence[tuple[str, str, int]],
    learning_model: str,
    learning_rule: str,
    learning_rate: float = 0.1,
    update_every_ticks: int = 4,
    entropic_objective: str = "fmc",
    horizon: int = 16,
    temperature: float = 0.5,
    entropy_weight: float = 0.25,
    memory_kind: str = "episodic",
    memory_capacity: int = 256,
    memory_permissions: Sequence[str] = ("read", "append"),
    memory_node: str = "memory_store",
    memory_port: str = "memory",
) -> dict[str, Any]:
    """CognitiveSpec; `levels` son tuplas `(level_id, sensor, bins)`.

    Los valores por defecto son los de la fixture canónica; no son recomendaciones.
    """
    return {
        "schema_version": COGNITIVE_SCHEMA,
        "resolution": {
            "binning_version": binning_version,
            "levels": [{"level_id": i, "sensor": s, "bins": b} for i, s, b in levels],
        },
        "learning": {
            "model": learning_model,
            "rule": learning_rule,
            "learning_rate": float(learning_rate),
            "update_every_ticks": update_every_ticks,
        },
        "entropic": {
            "objective": entropic_objective,
            "horizon": horizon,
            "temperature": float(temperature),
            "entropy_weight": float(entropy_weight),
        },
        "memory": {
            "kind": memory_kind,
            "capacity": memory_capacity,
            "permissions": list(memory_permissions),
            "port": {"node": memory_node, "port": memory_port},
        },
    }


def agent(
    agent_id: str,
    *,
    system_spec: Mapping[str, Any],
    model_spec: Mapping[str, Any],
    sensors: Sequence[Mapping[str, Any]],
    actions: Sequence[str],
    goals: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]] = (),
    cognitive_spec: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": AGENT_SCHEMA,
        "agent_id": agent_id,
        "system": dict(system_spec),
        "model": dict(model_spec),
        "sensors": list(sensors),
        "actions": list(actions),
        "goals": list(goals),
        "constraints": list(constraints),
        "config": dict(config or {}),
    }
    if cognitive_spec is not None:
        out["cognitive"] = dict(cognitive_spec)
    return out


def world(world_id: str, kind: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {"world_id": world_id, "kind": kind, "config": dict(config or {})}


@dataclass(frozen=True)
class NativeReceipt:
    """Resultado de la validación nativa: lo que el parser Rust reportó."""

    spec_hash: str
    stdout: str


class ExperimentSpec:
    """Experimento CognitiveSpec: documento JSON + forma canónica + hash."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        self.document: dict[str, Any] = dict(document)

    @classmethod
    def build(
        cls,
        experiment_id: str,
        *,
        system_spec: Mapping[str, Any],
        world_spec: Mapping[str, Any],
        agents: Sequence[Mapping[str, Any]],
        cognitive_spec: Mapping[str, Any],
        budget: tuple[int, int, int],
        dev_seeds: Sequence[int],
        holdout_seeds: Sequence[int],
        goals: Sequence[Mapping[str, Any]] = (),
        constraints: Sequence[Mapping[str, Any]] = (),
        config: Mapping[str, Any] | None = None,
    ) -> ExperimentSpec:
        """`budget` es `(max_ticks, max_transitions, deadline_ms)`."""
        max_ticks, max_transitions, deadline_ms = budget
        return cls(
            {
                "schema_version": EXPERIMENT_SCHEMA,
                "experiment_id": experiment_id,
                "system": dict(system_spec),
                "world": dict(world_spec),
                "agents": list(agents),
                "cognitive": dict(cognitive_spec),
                "goals": list(goals),
                "constraints": list(constraints),
                "budget": {
                    "max_ticks": max_ticks,
                    "max_transitions": max_transitions,
                    "deadline_ms": deadline_ms,
                },
                "seeds": {"dev_seeds": list(dev_seeds), "holdout_seeds": list(holdout_seeds)},
                "config": dict(config or {}),
            }
        )

    @classmethod
    def from_json(cls, text: str) -> ExperimentSpec:
        """Carga un JSON existente (p. ej. un fixture) rechazando claves duplicadas."""

        def no_dups(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            d = dict(pairs)
            if len(d) != len(pairs):
                raise GogamenSpecError("duplicate JSON key", code="DuplicateKey")
            return d

        import json

        return cls(mark_exprs(json.loads(text, object_pairs_hook=no_dups)))

    def to_canonical_json(self) -> str:
        return canonical_json(self.document)

    def hash(self) -> str:
        return spec_hash(self.document)

    def write(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.write_text(self.to_canonical_json(), encoding="utf-8")
        return p

    def validate_native(
        self,
        *,
        binary: str | os.PathLike[str] | None = None,
        timeout: float = DEFAULT_NATIVE_TIMEOUT_S,
    ) -> NativeReceipt:
        """Envía el JSON al parser Rust (CLI) y comprueba que su hash nativo coincide."""
        with tempfile.TemporaryDirectory(prefix="gogamen-spec-") as tmp:
            path = self.write(Path(tmp) / "experiment.json")
            receipt = validate_file_native(path, binary=binary, timeout=timeout)
        if receipt.spec_hash != self.hash():
            raise GogamenSpecError(
                f"hash divergente: python={self.hash()} native={receipt.spec_hash}",
                code="HashMismatch",
            )
        return receipt


# --------------------------------------------------------------------------- native boundary

_FIELD_RE = re.compile(r"(?:missing required field|unknown field|invalid field) '([^']+)' in (\w+)")


def _spec_error_from_cli(message: str, path: str) -> GogamenSpecError:
    """Extrae el diagnóstico del parser Rust sin reinterpretarlo; añade pointer si es de primer nivel."""
    m = _FIELD_RE.search(message)
    field, context = (m.group(1), m.group(2)) if m else (None, None)
    pointer = f"/{_pointer_token(field)}" if context == "AdaptiveExperimentSpec" and field else None
    return GogamenSpecError(
        message.strip(), code="NativeSpecRejected", pointer=pointer, path=path, context=context, field=field
    )


def run_cli(
    args: Sequence[str],
    *,
    binary: str | os.PathLike[str] | None = None,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Ejecuta el CLI `gogamen` con timeout explícito; al vencer mata y recoge el proceso."""
    exe = resolve_gogamen_binary(binary)
    try:
        return subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tail = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        raise GogamenTimeoutError(
            f"gogamen {' '.join(args[:2])} excedió el timeout de {timeout}s", stderr_tail=tail[-4096:]
        ) from exc


_HASH_RE = re.compile(r"^\s*spec_hash:\s*([0-9a-f]{16})\s*$", re.MULTILINE)


def validate_file_native(
    path: str | os.PathLike[str],
    *,
    binary: str | os.PathLike[str] | None = None,
    timeout: float = DEFAULT_NATIVE_TIMEOUT_S,
) -> NativeReceipt:
    """Frontera existente: `gogamen run-experiment --demo grid-transfer --dsl-file <path>`.

    El CLI parsea con `AdaptiveExperimentSpec::from_json`, re-emite la forma canónica y
    reporta `spec_hash` (FNV-1a 64). No existe un subcomando solo-validar: el CLI además
    ejecuta su demo fija (no derivada de la spec; ver `binding_note` del recibo).
    """
    spec_path = str(path)
    with tempfile.TemporaryDirectory(prefix="gogamen-out-") as out_dir:
        proc = run_cli(
            ["run-experiment", "--demo", "grid-transfer", "--dsl-file", spec_path, "--out-dir", out_dir],
            binary=binary,
            timeout=timeout,
        )
    if proc.returncode != 0:
        raise _spec_error_from_cli(proc.stderr or proc.stdout, spec_path)
    m = _HASH_RE.search(proc.stdout)
    if not m:
        raise GogamenSpecError(
            "el CLI no reportó spec_hash", code="InvalidResponse", path=spec_path
        )
    return NativeReceipt(spec_hash=m.group(1), stdout=proc.stdout)
