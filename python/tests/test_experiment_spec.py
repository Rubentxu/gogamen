"""GOG-194 / DEM-041: experimentos CognitiveSpec desde Python y barrido por la frontera CLI.

AC1: test_ac1_*  (JSON canónico y hash idénticos al fixture Kotlin/Rust y al hash nativo del CLI)
AC2: test_ac2_*  (barrido lanzado desde Python; CSV nativo sin dominio en Python)
AC3: test_ac3_*  (timeouts explícitos y errores con ruta)
"""

from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest

from gogamen import (
    ExperimentSpec,
    GogamenSpecError,
    GogamenSweepError,
    GogamenTimeoutError,
    canonical_json,
    experiment_spec as es,
    resolve_gogamen_binary,
)
from gogamen import sweep as sw
from gogamen.errors import GogamenProtocolError

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "spec" / "golden" / "cognitive_dsl"
# Hash nativo de h0 obtenido con `gogamen run-experiment --demo grid-transfer --dsl-file h0.json`.
H0_NATIVE_HASH = "46d838f79d6de6df"
# csv_hash de GOG-185 (determinista; el test Rust del barrido exige reproducibilidad).
SWEEP_CSV_HASH = "48d430a9b2822880"


def _native_binary() -> str | None:
    try:
        return resolve_gogamen_binary()
    except GogamenProtocolError:
        return None


def _fake_binary(tmp: str, body: str) -> str:
    path = Path(tmp) / "fake"
    path.write_text(f"#!{sys.executable} -I\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def build_h0() -> ExperimentSpec:
    """Réplica en Python de `h0()` de CognitiveCorpus.kt (misma fixture de sistema)."""
    lfov, belief, mem_t = es.custom("LocalObservationGrid"), es.custom("BeliefDistribution"), es.custom("MemoryPort")
    sysspec = es.system(
        "cognitive_grid_pomdp_system",
        [
            es.node(
                "pomdp_grid_env", "grid_pomdp",
                [es.port("local_fov", "output", lfov), es.port("action", "input", "string")],
                {"fov_radius": 2, "partial_observability": True},
            ),
            es.node(
                "belief_updater", "belief_state",
                [es.port("local_fov", "input", lfov), es.port("belief_state", "output", belief)],
            ),
            es.node(
                "cognitive_planner", "active_inference",
                [es.port("belief_state", "input", belief), es.port("action", "output", "string")],
            ),
            es.node("memory_store", "memory", [es.port("memory", "output", mem_t)]),
        ],
        [
            es.connection("pomdp_grid_env", "local_fov", "belief_updater", "local_fov"),
            es.connection("belief_updater", "belief_state", "cognitive_planner", "belief_state"),
            es.connection("cognitive_planner", "action", "pomdp_grid_env", "action", delay_ticks=1),
        ],
    )
    cog = es.cognitive(
        binning_version="v0",
        levels=[("coarse", "local_fov_sensor", 3), ("mid", "local_fov_sensor", 9), ("fine", "local_fov_sensor", 25)],
        learning_model="pomdp_belief_v0",
        learning_rule="bayesian",
    )
    goals = [
        es.goal(
            "epistemic_utility", True,
            es.add(es.var("reward"), es.mul(es.const(0.25), es.var("info_gain"))),
        )
    ]
    constraints = [es.constraint("energy_guard", True, es.gt(es.var("energy"), es.const(0.0)))]
    ag = es.agent(
        "cognitive_explorer",
        system_spec=sysspec,
        model_spec=es.model("pomdp_belief_v0", "belief_state", {"belief_horizon": 8, "entropy_weight": 0.25}),
        sensors=[es.sensor("local_fov_sensor", "partial_fov", "pomdp_grid_env", "local_fov", lfov, {"radius": 2})],
        actions=["move_north", "move_south", "move_east", "move_west", "observe", "sense_local"],
        goals=goals,
        constraints=constraints,
        cognitive_spec=cog,
    )
    return ExperimentSpec.build(
        "grid_cognitive_spec_v0",
        system_spec=sysspec,
        world_spec=es.world("grid_pomdp_7x7", "grid_pomdp", {"width": 7, "height": 7, "partial_observability": True}),
        agents=[ag],
        cognitive_spec=cog,
        goals=goals,
        constraints=constraints,
        budget=(1000, 2500, 10_000),
        dev_seeds=[11, 12, 13],
        holdout_seeds=[1001, 1002],
        config={"headless": True},
    )


class CanonicalFormTests(unittest.TestCase):
    def test_float_and_string_rules_match_rust_writer(self) -> None:
        self.assertEqual(canonical_json([1.0, 0.25, 1e21, 1e-7, -0.0, 1, True, None]),
                         "[1.0,0.25,1000000000000000000000.0,0.0000001,0.0,1,true,null]")
        self.assertEqual(canonical_json({"b": 1, "a": "x\"\n\u0001é"}), '{"a":"x\\"\\n\\u0001é","b":1}')

    def test_ac3_non_finite_float_reports_json_pointer(self) -> None:
        with self.assertRaises(GogamenSpecError) as ctx:
            canonical_json({"a": [0.0, float("nan")]})
        self.assertEqual(ctx.exception.pointer, "/a/1")
        self.assertIn("/a/1", str(ctx.exception))
        with self.assertRaises(GogamenSpecError) as ctx:
            canonical_json({"k": object()})
        self.assertEqual(ctx.exception.pointer, "/k")


class Ac1HashTests(unittest.TestCase):
    def test_ac1_python_builder_is_byte_identical_to_kotlin_fixture(self) -> None:
        fixture = (CORPUS / "positive" / "h0.json").read_text(encoding="utf-8")
        spec = build_h0()
        self.assertEqual(spec.to_canonical_json(), fixture)
        self.assertEqual(spec.hash(), H0_NATIVE_HASH)

    def test_ac1_all_checked_in_positive_fixtures_recanonicalize_identically(self) -> None:
        for i in range(8):
            text = (CORPUS / "positive" / f"h{i}.json").read_text(encoding="utf-8")
            self.assertEqual(ExperimentSpec.from_json(text).to_canonical_json(), text, f"h{i}")

    def test_ac1_hash_equals_native_cli_hash(self) -> None:
        exe = _native_binary()
        if exe is None:
            self.skipTest("NOT_RUN: binario gogamen no compilado (cargo build -p gogamen-cli)")
        receipt = build_h0().validate_native(binary=exe, timeout=60.0)
        self.assertEqual(receipt.spec_hash, build_h0().hash())
        self.assertEqual(receipt.spec_hash, H0_NATIVE_HASH)


class Ac3ErrorTests(unittest.TestCase):
    def test_ac3_native_rejection_carries_path_and_field(self) -> None:
        exe = _native_binary()
        if exe is None:
            self.skipTest("NOT_RUN: binario gogamen no compilado")
        path = CORPUS / "negative" / "no_seeds.json"
        with self.assertRaises(GogamenSpecError) as ctx:
            es.validate_file_native(path, binary=exe, timeout=60.0)
        err = ctx.exception
        self.assertEqual(err.pointer, "/seeds")
        self.assertEqual(err.path, str(path))
        self.assertEqual((err.context, err.field), ("AdaptiveExperimentSpec", "seeds"))
        self.assertIn(str(path), str(err) + err.path)

    def test_ac3_native_rejection_without_field_still_carries_file_path(self) -> None:
        exe = _native_binary()
        if exe is None:
            self.skipTest("NOT_RUN: binario gogamen no compilado")
        path = CORPUS / "negative" / "seeds_overlap.json"
        with self.assertRaises(GogamenSpecError) as ctx:
            es.validate_file_native(path, binary=exe, timeout=60.0)
        self.assertEqual(ctx.exception.path, str(path))
        self.assertIn("12", ctx.exception.message)

    def test_ac3_spec_cli_timeout_is_bounded_and_typed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _fake_binary(tmp, "import time\ntime.sleep(60)\n")
            t0 = time.monotonic()
            with self.assertRaises(GogamenTimeoutError):
                es.validate_file_native(CORPUS / "positive" / "h0.json", binary=exe, timeout=0.5)
            self.assertLess(time.monotonic() - t0, 10.0)

    def test_ac3_sweep_timeout_is_bounded_and_typed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            exe = _fake_binary(tmp, "import time\ntime.sleep(60)\n")
            t0 = time.monotonic()
            with self.assertRaises(GogamenTimeoutError):
                sw.run_sweep(binary=exe, timeout=0.5)
            self.assertLess(time.monotonic() - t0, 10.0)

    def test_ac3_missing_sweep_binary_is_typed(self) -> None:
        with self.assertRaises(GogamenSweepError) as ctx:
            sw.run_sweep(binary="/nonexistent/sweep", timeout=1.0)
        self.assertEqual(ctx.exception.code, "BinaryNotFound")


class Ac2SweepTests(unittest.TestCase):
    def test_ac2_sweep_returns_native_dem028_csv(self) -> None:
        try:
            exe = sw.resolve_sweep_binary()
        except GogamenSweepError:
            self.skipTest("NOT_RUN: cargo build -p gogamen-arena --example sweep_cartpole")
        result = sw.run_sweep(binary=exe, timeout=120.0)
        self.assertTrue(result.csv_text.startswith("split,horizon,scale,budget,endowment,total,run_errors,"))
        self.assertEqual(len(result.rows), 30)  # 15 celdas x {dev, holdout}
        self.assertEqual({r["split"] for r in result.rows}, {"dev", "holdout"})
        self.assertEqual(result.csv_hash, SWEEP_CSV_HASH)

    def test_ac2_csv_hash_mismatch_and_failure_are_detected(self) -> None:
        body_bad = (
            "import sys\nopen(sys.argv[1]+'/sweep.csv','w').write('a,b\\n1,2\\n')\n"
            "print('sweep x cells=1 declared_cost=1 csv_hash=0000000000000000')\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GogamenSweepError) as ctx:
                sw.run_sweep(binary=_fake_binary(tmp, body_bad), timeout=10.0)
            self.assertEqual(ctx.exception.code, "CsvHashMismatch")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(GogamenSweepError) as ctx:
                sw.run_sweep(binary=_fake_binary(tmp, "import sys\nsys.exit(3)\n"), timeout=10.0)
            self.assertEqual(ctx.exception.code, "SweepFailed")


if __name__ == "__main__":
    unittest.main()
