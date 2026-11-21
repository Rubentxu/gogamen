"""Cliente Python stdlib-only para el protocolo residente JSON-Lines de Gogamen."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import queue
import subprocess
import threading
from typing import Any, Sequence

from .errors import GogamenProtocolError, GogamenTimeoutError, raise_for_protocol_error

DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_STDERR_TAIL_BYTES = 64 * 1024
_TERMINATE_GRACE_S = 2.0
_KILL_WAIT_S = 5.0

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

    def __init__(
        self,
        binary_path: str | os.PathLike[str] | None = None,
        *,
        request_timeout: float | None = DEFAULT_REQUEST_TIMEOUT_S,
        stderr_tail_bytes: int = DEFAULT_STDERR_TAIL_BYTES,
        command: Sequence[str] | None = None,
    ) -> None:
        """Lanza el proceso residente.

        `request_timeout` (segundos, `None` = sin límite) acota cada request. Tras un
        timeout el cliente se cierra y NO es reutilizable (crear uno nuevo).
        `command` permite sustituir el argv completo (tests con servidores falsos).
        """
        argv = list(command) if command is not None else [resolve_gogamen_binary(binary_path), "serve-protocol"]
        self._request_timeout = request_timeout
        self._stderr_tail: deque[bytes] = deque()
        self._stderr_tail_len = 0
        self._stderr_tail_max = max(1, stderr_tail_bytes)
        self._stderr_lock = threading.Lock()
        self._stdout_queue: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        self._next_id = 1
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout, name="gogamen-stdout", daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name="gogamen-stderr", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _pump_stdout(self) -> None:
        stream = self._proc.stdout
        try:
            with os.fdopen(os.dup(stream.fileno()), "rb") as buffered:
                for line in iter(buffered.readline, b""):
                    self._stdout_queue.put(line)
        except (OSError, ValueError):
            pass
        finally:
            self._stdout_queue.put(None)

    def _drain_stderr(self) -> None:
        """Drena stderr continuamente a un buffer acotado (últimos N bytes)."""
        stream = self._proc.stderr
        try:
            while True:
                chunk = os.read(stream.fileno(), 65536)
                if not chunk:
                    return
                with self._stderr_lock:
                    self._stderr_tail.append(chunk)
                    self._stderr_tail_len += len(chunk)
                    while self._stderr_tail_len - len(self._stderr_tail[0]) >= self._stderr_tail_max:
                        self._stderr_tail_len -= len(self._stderr_tail.popleft())
        except (OSError, ValueError):
            return

    def stderr_tail(self) -> str:
        """Últimos bytes de stderr del hijo (acotado), decodificados con reemplazo."""
        with self._stderr_lock:
            data = b"".join(self._stderr_tail)[-self._stderr_tail_max :]
        return data.decode("utf-8", errors="replace")

    def __enter__(self) -> GogamenClient:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()

    def close(self) -> None:
        """Cierra y recoge el proceso: cierra stdin, terminate, kill tras gracia, wait. Idempotente."""
        self._closed = True
        proc = self._proc
        try:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=_TERMINATE_GRACE_S / 2)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=_TERMINATE_GRACE_S)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=_KILL_WAIT_S)
        self._stdout_thread.join(timeout=1.0)
        self._stderr_thread.join(timeout=1.0)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                try:
                    stream.close()
                except OSError:
                    pass

    def _timeout(self, what: str) -> GogamenTimeoutError:
        tail = self.stderr_tail()
        self.close()
        return GogamenTimeoutError(
            f"{what} (timeout={self._request_timeout}s); cliente cerrado, crear uno nuevo. stderr: {tail.strip()[-2000:]}",
            stderr_tail=tail,
        )

    def send_raw_line(self, raw_line: str) -> dict[str, Any]:
        """Envía una línea cruda al protocolo y decodifica la respuesta JSON.

        Lanza `GogamenTimeoutError` si no hay respuesta en `request_timeout`.
        """
        if self._closed:
            raise GogamenProtocolError("cliente cerrado", code="ClientClosed")
        if self._proc.stdin is None or self._proc.stdout is None:
            raise GogamenProtocolError("streams del subproceso no disponibles", code="StreamClosed")
        if self._proc.poll() is not None:
            self._stderr_thread.join(timeout=1.0)
            raise GogamenProtocolError(
                f"proceso gogamen terminado prematuramente (rc={self._proc.returncode}): {self.stderr_tail().strip()}",
                code="ProcessExited",
            )

        try:
            self._proc.stdin.write((raw_line.rstrip("\n") + "\n").encode("utf-8"))
        except (BrokenPipeError, OSError) as exc:
            raise GogamenProtocolError(
                f"no se pudo escribir al servidor: {exc}; stderr: {self.stderr_tail().strip()}",
                code="ProcessExited",
            ) from exc

        try:
            raw = self._stdout_queue.get(timeout=self._request_timeout)
        except queue.Empty:
            raise self._timeout("sin respuesta del servidor de protocolo") from None
        if raw is None:
            self._stdout_queue.put(None)
            self._stderr_thread.join(timeout=1.0)
            raise GogamenProtocolError(
                f"EOF inesperado del servidor de protocolo: {self.stderr_tail().strip()}",
                code="UnexpectedEOF",
            )
        response_line = raw.decode("utf-8", errors="replace")

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
