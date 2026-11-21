"""Cliente Python stdlib-only para el protocolo residente JSON-Lines de Gogamen."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .errors import GogamenProtocolError, raise_for_protocol_error

ACTION_STAY = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4


@dataclass(frozen=True)
class ProtocolEvent:
    """Evento ligero emitido por una sesión residente en Rust."""

    seq: int
    tick: int
    kind: str
    detail: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtocolEvent:
        return cls(
            seq=int(data["seq"]),
            tick=int(data["tick"]),
            kind=str(data["kind"]),
            detail=str(data["detail"]),
        )


@dataclass(frozen=True)
class TerminalState:
    """Estado de terminación de un paso o episodio."""

    is_terminal: bool
    reason: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TerminalState:
        reason = data.get("reason")
        return cls(
            is_terminal=bool(data["is_terminal"]),
            reason=str(reason) if reason is not None else None,
        )


@dataclass(frozen=True)
class StepDelta:
    """Delta ligero de transición retornado por `step` (sin snapshot completo)."""

    dx: int
    dy: int
    collided: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepDelta:
        return cls(
            dx=int(data["dx"]),
            dy=int(data["dy"]),
            collided=bool(data["collided"]),
        )


@dataclass(frozen=True)
class SessionInfo:
    """Respuesta inicial al crear o reiniciar una sesión Grid."""

    session_id: str
    world: str
    action_cardinality: int
    tick: int
    elapsed_s: float
    observation: tuple[float, ...]
    terminal: TerminalState
    events: tuple[ProtocolEvent, ...]


@dataclass(frozen=True)
class StepResult:
    """Resultado ligero de un paso ejecutado en la sesión residente Rust."""

    session_id: str
    tick: int
    elapsed_s: float
    reward: float
    total_reward: float
    delta: StepDelta
    observation: tuple[float, ...]
    terminal: TerminalState
    events: tuple[ProtocolEvent, ...]
    raw_keys: frozenset[str]


def resolve_gogamen_binary(explicit_path: str | os.PathLike[str] | None = None) -> str:
    """Localiza el binario `gogamen` sin dependencias externas."""
    if explicit_path is not None:
        candidate = Path(explicit_path)
        if candidate.is_file():
            return str(candidate)
        raise GogamenProtocolError(
            f"binario gogamen no encontrado en ruta explícita: {candidate}",
            code="BinaryNotFound",
        )

    env_bin = os.environ.get("GOGAMEN_BIN")
    if env_bin:
        candidate = Path(env_bin)
        if candidate.is_file():
            return str(candidate)
        raise GogamenProtocolError(
            f"GOGAMEN_BIN apunta a un archivo inexistente: {candidate}",
            code="BinaryNotFound",
        )

    candidates: list[Path] = []
    cargo_target = os.environ.get("CARGO_TARGET_DIR")
    if cargo_target:
        candidates.append(Path(cargo_target) / "debug" / "gogamen")
        candidates.append(Path(cargo_target) / "release" / "gogamen")

    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "target" / "debug" / "gogamen")
    candidates.append(repo_root / "target" / "release" / "gogamen")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    # Fallback: query cargo metadata for configured target_directory (e.g. .cargo/config.toml).
    try:
        meta = subprocess.run(
            ["cargo", "metadata", "--no-deps", "--format-version", "1", "--offline"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if meta.returncode == 0:
            target_dir = json.loads(meta.stdout).get("target_directory")
            if target_dir:
                for profile in ("debug", "release"):
                    candidate = Path(target_dir) / profile / "gogamen"
                    if candidate.is_file():
                        return str(candidate)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass

    which_bin = shutil.which("gogamen")
    if which_bin:
        return which_bin

    raise GogamenProtocolError(
        "no se encontró el binario 'gogamen'; compila con 'cargo build -p gogamen-cli --offline' o define GOGAMEN_BIN",
        code="BinaryNotFound",
    )


class GridSession:
    """Manejador ergonómico de una sesión `GridWorldEnv` residente en Rust."""

    def __init__(self, client: GogamenClient, info: SessionInfo) -> None:
        self._client = client
        self._info = info
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._info.session_id

    @property
    def initial_info(self) -> SessionInfo:
        return self._info

    def step(self, action: int) -> StepResult:
        """Aplica una acción discreta en la sesión residente Rust."""
        return self._client.step(self.session_id, action)

    def events(self, *, since_seq: int = 0, drain: bool = False) -> list[ProtocolEvent]:
        """Consulta eventos emitidos por la sesión residente."""
        return self._client.events(self.session_id, since_seq=since_seq, drain=drain)

    def reset(self) -> SessionInfo:
        """Reinicia el episodio en la sesión residente Rust."""
        info = self._client.reset(self.session_id)
        self._info = info
        return info

    def close(self) -> bool:
        """Cierra la sesión en el servidor Rust."""
        if self._closed:
            return False
        self._closed = True
        return self._client.close_session(self.session_id)

    def __enter__(self) -> GridSession:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        try:
            self.close()
        except Exception:
            pass


class GogamenClient:
    """Cliente residente por `stdin`/`stdout` JSON-Lines sobre `gogamen serve-protocol`."""

    def __init__(self, binary_path: str | os.PathLike[str] | None = None) -> None:
        resolved = resolve_gogamen_binary(binary_path)
        self._proc = subprocess.Popen(
            [resolved, "serve-protocol"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self._next_id = 1

    def __enter__(self) -> GogamenClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Cierra el proceso residente `gogamen serve-protocol`."""
        if self._proc.poll() is None:
            if self._proc.stdin is not None and not self._proc.stdin.closed:
                self._proc.stdin.close()
            try:
                self._proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=1.0)
        for stream in (self._proc.stdout, self._proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()

    def send_raw_line(self, raw_line: str) -> dict[str, Any]:
        """Envía una línea cruda al protocolo y decodifica la respuesta JSON."""
        if self._proc.stdin is None or self._proc.stdout is None:
            raise GogamenProtocolError("streams del subproceso no disponibles", code="StreamClosed")
        if self._proc.poll() is not None:
            stderr_out = self._proc.stderr.read() if self._proc.stderr is not None else ""
            raise GogamenProtocolError(
                f"proceso gogamen terminado prematuramente (rc={self._proc.returncode}): {stderr_out.strip()}",
                code="ProcessExited",
            )

        self._proc.stdin.write(raw_line.rstrip("\n") + "\n")
        self._proc.stdin.flush()

        response_line = self._proc.stdout.readline()
        if not response_line:
            stderr_out = self._proc.stderr.read() if self._proc.stderr is not None else ""
            raise GogamenProtocolError(
                f"EOF inesperado del servidor de protocolo: {stderr_out.strip()}",
                code="UnexpectedEOF",
            )

        try:
            payload = json.loads(response_line)
        except json.JSONDecodeError as exc:
            raise GogamenProtocolError(
                f"respuesta JSON inválida del servidor: {exc}",
                code="InvalidJSON",
            ) from exc

        if not isinstance(payload, dict):
            raise GogamenProtocolError(
                "la respuesta del protocolo debe ser un objeto JSON",
                code="InvalidResponse",
            )
        return payload

    def _request(self, op: str, **params: Any) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        req = {"id": req_id, "op": op, **params}
        payload = self.send_raw_line(json.dumps(req, separators=(",", ":")))
        if not payload.get("ok", False):
            raise_for_protocol_error(payload.get("error"))
        return payload

    def create_grid_session(
        self,
        *,
        partial_obs: bool = False,
        partial_radius: int = 2,
    ) -> GridSession:
        """Crea una sesión residente `GridWorldEnv` en Rust."""
        data = self._request(
            "create_session",
            world="grid",
            partial_obs=partial_obs,
            partial_radius=partial_radius,
        )
        info = SessionInfo(
            session_id=str(data["session_id"]),
            world=str(data["world"]),
            action_cardinality=int(data["action_cardinality"]),
            tick=int(data["tick"]),
            elapsed_s=float(data["elapsed_s"]),
            observation=tuple(float(x) for x in data["observation"]),
            terminal=TerminalState.from_dict(data["terminal"]),
            events=tuple(ProtocolEvent.from_dict(ev) for ev in data.get("events", [])),
        )
        return GridSession(self, info)

    def step(self, session_id: str, action: int) -> StepResult:
        """Ejecuta un paso en la sesión indicada."""
        data = self._request("step", session_id=session_id, action=action)
        return StepResult(
            session_id=str(data["session_id"]),
            tick=int(data["tick"]),
            elapsed_s=float(data["elapsed_s"]),
            reward=float(data["reward"]),
            total_reward=float(data["total_reward"]),
            delta=StepDelta.from_dict(data["delta"]),
            observation=tuple(float(x) for x in data["observation"]),
            terminal=TerminalState.from_dict(data["terminal"]),
            events=tuple(ProtocolEvent.from_dict(ev) for ev in data.get("events", [])),
            raw_keys=frozenset(data.keys()),
        )

    def events(
        self,
        session_id: str,
        *,
        since_seq: int = 0,
        drain: bool = False,
    ) -> list[ProtocolEvent]:
        """Recupera eventos emitidos por la sesión."""
        data = self._request(
            "events",
            session_id=session_id,
            since_seq=since_seq,
            drain=drain,
        )
        return [ProtocolEvent.from_dict(ev) for ev in data.get("events", [])]

    def reset(self, session_id: str) -> SessionInfo:
        """Reinicia la sesión residente."""
        data = self._request("reset", session_id=session_id)
        return SessionInfo(
            session_id=str(data["session_id"]),
            world="grid",
            action_cardinality=5,
            tick=int(data["tick"]),
            elapsed_s=float(data["elapsed_s"]),
            observation=tuple(float(x) for x in data["observation"]),
            terminal=TerminalState.from_dict(data["terminal"]),
            events=tuple(ProtocolEvent.from_dict(ev) for ev in data.get("events", [])),
        )

    def close_session(self, session_id: str) -> bool:
        """Cierra y libera la sesión residente en Rust."""
        data = self._request("close_session", session_id=session_id)
        return bool(data.get("closed", False))
