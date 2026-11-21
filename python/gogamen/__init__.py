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
    GogamenSpecError,
    GogamenSweepError,
    GogamenDomainError,
    GogamenError,
    GogamenProtocolError,
    GogamenTimeoutError,
    GogamenSessionError,
)
from .experiment_spec import ExperimentSpec, canonical_json, spec_hash

__all__ = [
    "ExperimentSpec",
    "GogamenSpecError",
    "GogamenSweepError",
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
    "canonical_json",
    "spec_hash",
]
