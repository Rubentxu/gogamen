"""Ejemplo ejecutable del SDK Python de Gogamen sobre GridWorldEnv."""

from __future__ import annotations

from gogamen import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    GogamenClient,
    GogamenDomainError,
    GogamenSessionError,
)

SOLVE_CUL_DE_SAC_PATH = [
    ACTION_UP,
    ACTION_UP,
    ACTION_LEFT,
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_DOWN,
    ACTION_DOWN,
    ACTION_DOWN,
    ACTION_RIGHT,
    ACTION_RIGHT,
    ACTION_RIGHT,
    ACTION_RIGHT,
    ACTION_RIGHT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTION_UP,
    ACTION_UP,
    ACTION_UP,
]


def main() -> None:
    with GogamenClient() as client:
        with client.create_grid_session(partial_obs=False) as session:
            print(f"Sesión creada: {session.session_id}, tick={session.initial_info.tick}")
            for action in SOLVE_CUL_DE_SAC_PATH:
                res = session.step(action)
                ev_kinds = [ev.kind for ev in res.events]
                print(
                    f"tick={res.tick:2d} reward={res.reward:6.2f} total={res.total_reward:6.2f} "
                    f"delta=({res.delta.dx:+d},{res.delta.dy:+d}) terminal={res.terminal.is_terminal} "
                    f"events={ev_kinds}"
                )

            try:
                session.step(ACTION_STAY)
            except GogamenSessionError as exc:
                print(f"Excepción de sesión esperada: {exc}")

            session.reset()
            try:
                session.step(99)
            except GogamenDomainError as exc:
                print(f"Excepción de dominio esperada: {exc}")


if __name__ == "__main__":
    main()
