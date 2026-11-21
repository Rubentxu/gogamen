"""Excepciones tipadas del SDK Python de Gogamen."""

from __future__ import annotations


class GogamenError(Exception):
    """Excepción base para errores del SDK Python de Gogamen."""

    def __init__(self, message: str, *, code: str = "GogamenError") -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


class GogamenProtocolError(GogamenError):
    """Error de transporte, serialización JSON o petición inválida del protocolo."""


class GogamenTimeoutError(GogamenProtocolError):
    """El servidor residente no respondió dentro del timeout configurado.

    Tras este error el cliente queda cerrado (proceso terminado y recogido) y es
    inutilizable: hay que crear un `GogamenClient` nuevo. `stderr_tail` contiene
    los últimos bytes de stderr del hijo (decodificados) para diagnóstico.
    """

    def __init__(self, message: str, *, code: str = "Timeout", stderr_tail: str = "") -> None:
        super().__init__(message, code=code)
        self.stderr_tail = stderr_tail


class GogamenSessionError(GogamenError):
    """Error de ciclo de vida de sesión (desconocida, cerrada o ya terminal)."""


class GogamenDomainError(GogamenError):
    """Error de dominio reportado por el entorno residente en Rust."""


def raise_for_protocol_error(error_payload: object) -> None:
    """Convierte un objeto `error` JSON del protocolo en la excepción tipada correspondiente."""
    if not isinstance(error_payload, dict):
        raise GogamenProtocolError(
            f"respuesta de error malformada: {error_payload!r}",
            code="InvalidResponse",
        )

    code = str(error_payload.get("code", "InternalError"))
    message = str(error_payload.get("message", "error desconocido del protocolo"))

    if code in ("UnknownSession", "Terminated"):
        raise GogamenSessionError(message, code=code)
    if code == "DomainError":
        raise GogamenDomainError(message, code=code)
    raise GogamenProtocolError(message, code=code)
