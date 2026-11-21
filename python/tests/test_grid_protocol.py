"""Tests de equivalencia Rust/Python y protocolo residente para GridWorldEnv."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import unittest

from gogamen import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_STAY,
    ACTION_UP,
    GogamenClient,
    GogamenDomainError,
    GogamenProtocolError,
    GogamenSessionError,
    resolve_gogamen_binary,
)
from gogamen.examples.run_grid import SOLVE_CUL_DE_SAC_PATH


def _ensure_gogamen_binary() -> str:
    try:
        return resolve_gogamen_binary()
    except GogamenProtocolError:
        repo_root = Path(__file__).resolve().parents[2]
        subprocess.run(
            ["cargo", "build", "-p", "gogamen-cli", "--offline"],
            cwd=repo_root,
            check=True,
        )
        return resolve_gogamen_binary()


class GridProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_path = _ensure_gogamen_binary()

    def test_grid_episode_equivalence_and_no_snapshot_per_step(self) -> None:
        """Verifica equivalencia de episodio Grid y ausencia de snapshot completo por paso."""
        with GogamenClient(self.bin_path) as client:
            with client.create_grid_session(partial_obs=False) as session:
                init = session.initial_info
                self.assertEqual(init.world, "grid")
                self.assertEqual(init.action_cardinality, 5)
                self.assertEqual(init.tick, 0)
                self.assertFalse(init.terminal.is_terminal)
                # Full obs: [agent_x, agent_y, goal_x, goal_y, ...9x7 cells] = 4 + 63 = 67
                self.assertEqual(len(init.observation), 67)
                self.assertEqual(init.observation[0], 3.0)
                self.assertEqual(init.observation[1], 3.0)
                self.assertEqual(init.observation[2], 7.0)
                self.assertEqual(init.observation[3], 1.0)

                # First collide with wall below start (3, 3) -> (3, 4)
                wall_step = session.step(ACTION_DOWN)
                self.assertEqual(wall_step.tick, 1)
                self.assertAlmostEqual(wall_step.reward, -0.5)
                self.assertAlmostEqual(wall_step.total_reward, -0.5)
                self.assertTrue(wall_step.delta.collided)
                self.assertEqual((wall_step.delta.dx, wall_step.delta.dy), (0, 0))
                self.assertEqual(wall_step.events[0].kind, "wall_collision")
                for forbidden in ("snapshot", "map", "config", "cells"):
                    self.assertNotIn(forbidden, wall_step.raw_keys)

                # Reset before running canonical path to Goal (7, 1)
                reset_info = session.reset()
                self.assertEqual(reset_info.tick, 0)
                self.assertEqual(reset_info.observation[0], 3.0)
                self.assertEqual(reset_info.observation[1], 3.0)

                expected_total = 0.0
                final_res = None
                for idx, action in enumerate(SOLVE_CUL_DE_SAC_PATH, start=1):
                    res = session.step(action)
                    final_res = res
                    self.assertEqual(res.tick, idx)
                    self.assertFalse(res.delta.collided)
                    for forbidden in ("snapshot", "map", "config", "cells"):
                        self.assertNotIn(forbidden, res.raw_keys)
                    if idx < len(SOLVE_CUL_DE_SAC_PATH):
                        self.assertAlmostEqual(res.reward, -0.05)
                        self.assertFalse(res.terminal.is_terminal)
                        expected_total += -0.05
                    else:
                        self.assertAlmostEqual(res.reward, 10.0)
                        self.assertTrue(res.terminal.is_terminal)
                        self.assertEqual(res.terminal.reason, "GoalReached")
                        expected_total += 10.0
                    self.assertAlmostEqual(res.total_reward, expected_total)

                assert final_res is not None
                self.assertEqual(final_res.observation[0], 7.0)
                self.assertEqual(final_res.observation[1], 1.0)
                kinds = [ev.kind for ev in final_res.events]
                self.assertEqual(kinds, ["step_applied", "terminal_reached"])

                # Query events log with since_seq
                all_events = session.events(since_seq=0)
                self.assertGreaterEqual(len(all_events), len(SOLVE_CUL_DE_SAC_PATH) + 2)

    def test_partial_observation_mode_and_lightweight_window(self) -> None:
        with GogamenClient(self.bin_path) as client:
            with client.create_grid_session(partial_obs=True, partial_radius=2) as session:
                # Partial obs with radius=2: [x, y, radius, ...(2*2+1)^2] = 3 + 25 = 28
                self.assertEqual(len(session.initial_info.observation), 28)
                self.assertEqual(session.initial_info.observation[2], 2.0)
                step = session.step(ACTION_UP)
                self.assertEqual(len(step.observation), 28)
                self.assertEqual((step.delta.dx, step.delta.dy), (0, -1))

    def test_typed_exceptions_protocol_session_and_domain(self) -> None:
        with GogamenClient(self.bin_path) as client:
            # 1. ProtocolError on malformed JSON
            bad_payload = client.send_raw_line("{invalid json")
            self.assertFalse(bad_payload["ok"])
            self.assertEqual(bad_payload["error"]["code"], "InvalidRequest")

            # ProtocolError via _request on unknown op
            with self.assertRaises(GogamenProtocolError) as proto_ctx:
                client._request("nonexistent_op")
            self.assertEqual(proto_ctx.exception.code, "InvalidRequest")

            # 2. SessionError on unknown session
            with self.assertRaises(GogamenSessionError) as unk_ctx:
                client.step("sess-missing", ACTION_STAY)
            self.assertEqual(unk_ctx.exception.code, "UnknownSession")

            # 3. DomainError on out-of-range action
            session = client.create_grid_session()
            with self.assertRaises(GogamenDomainError) as dom_ctx:
                session.step(99)
            self.assertEqual(dom_ctx.exception.code, "DomainError")

            # 4. SessionError on stepping after terminal
            for action in SOLVE_CUL_DE_SAC_PATH:
                session.step(action)
            with self.assertRaises(GogamenSessionError) as term_ctx:
                session.step(ACTION_STAY)
            self.assertEqual(term_ctx.exception.code, "Terminated")

            # 5. Closing session invalidates further calls
            self.assertTrue(session.close())
            with self.assertRaises(GogamenSessionError) as closed_ctx:
                session.step(ACTION_STAY)
            self.assertEqual(closed_ctx.exception.code, "UnknownSession")

    def test_rust_cli_run_grid_determinism_smoke(self) -> None:
        """Comprueba que el binario gogamen usado por el SDK también ejecuta run-grid."""
        proc = subprocess.run(
            [self.bin_path, "run-grid", "--ticks", "20", "--seed", "42"],
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("gogamen run-grid", proc.stdout)
        self.assertIn("trace_hash:", proc.stdout)


if __name__ == "__main__":
    unittest.main()
