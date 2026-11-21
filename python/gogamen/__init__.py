"""SDK Python stdlib-only de Gogamen por protocolo residente JSON-Lines."""

from .client import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    GogamenClient,
    GridSession,
    ProtocolEvent,
    SessionInfo,
    StepDelta,
    StepResult,
    TerminalState,
    resolve_gogamen_binary,
)
from .errors import (
    GogamenDomainError,
    GogamenError,
    GogamenProtocolError,
    GogamenTimeoutError,
    GogamenSessionError,
)

__all__ = [
    "ACTION_DOWN",
    "ACTION_LEFT",
    "ACTION_RIGHT",
    "ACTION_STAY",
    "ACTION_UP",
    "GogamenClient",
    "GogamenDomainError",
    "GogamenError",
    "GogamenProtocolError",
    "GogamenTimeoutError",
    "GogamenSessionError",
    "GridSession",
    "ProtocolEvent",
    "SessionInfo",
    "StepDelta",
    "StepResult",
    "TerminalState",
    "resolve_gogamen_binary",
]
