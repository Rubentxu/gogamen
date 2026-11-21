package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.absolutePathString
import kotlin.io.path.exists
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class SystemSpecDslTest {
    private fun repoRoot(): Path {
        val cwd = Path.of("").toAbsolutePath().normalize()
        return if (cwd.fileName?.toString() == "kotlin") cwd.parent else cwd
    }

    private fun runRustValidator(spec: SystemSpec, ticks: Int = 1): Pair<Int, String> {
        val jsonFile = Files.createTempFile("gogamen-dsl-test-", ".json")
        try {
            Files.writeString(jsonFile, spec.toCanonicalJson())
            val root = repoRoot()
            val process = ProcessBuilder(
                "cargo",
                "run",
                "-q",
                "-p",
                "gogamen-cli",
                "--offline",
                "--",
                "run-spec",
                "--json-file",
                jsonFile.absolutePathString(),
                "--ticks",
                ticks.toString(),
            )
                .directory(root.toFile())
                .redirectErrorStream(true)
                .start()

            val output = process.inputStream.bufferedReader().readText()
            val exitCode = process.waitFor()
            return exitCode to output
        } finally {
            Files.deleteIfExists(jsonFile)
        }
    }

    @Test
    fun `dsl classes carry GogamenDsl marker annotation`() {
        val dslClasses = listOf(
            SystemSpecBuilder::class.java,
            NodeBuilder::class.java,
            ConnectionsBuilder::class.java,
            ConfigBuilder::class.java,
        )
        for (cls in dslClasses) {
            assertTrue(
                cls.isAnnotationPresent(GogamenDsl::class.java),
                "${cls.simpleName} must be annotated with @GogamenDsl",
            )
        }
    }

    @Test
    fun `canonicalTwoNode matches gogamen-spec canonical JSON and executes in Rust CLI`() {
        val expectedJson =
            """{"config":{"deterministic":true,"dt_secs":0.02},"connections":[{"delay_ticks":0,"from_node":"sensor","from_port":"observation","to_node":"controller","to_port":"observation"}],"nodes":[{"config":{"normalized":true,"rate_hz":50},"id":"sensor","kind":"observer","ports":[{"direction":"output","id":"observation","port_type":"f64"}]},{"config":{"gain":0.5,"mode":"greedy"},"id":"controller","kind":"policy","ports":[{"direction":"input","id":"observation","port_type":"f64"},{"direction":"output","id":"action","port_type":"i64"}]}],"schema_version":"gogamen.system_spec.v0","spec_id":"two_node_canonical_v0"}"""

        val spec = Fixtures.canonicalTwoNode()
        assertEquals(expectedJson, spec.toCanonicalJson())

        val (exitCode, output) = runRustValidator(spec, ticks = 5)
        assertEquals(0, exitCode, "Rust validator failed:\n$output")
        assertTrue(output.contains("spec_id: two_node_canonical_v0"), output)
        assertTrue(output.contains("semantic_fingerprint:"), output)
    }

    @Test
    fun `canonicalGridSystem generates canonical SystemSpec JSON for Grid 5x5`() {
        val expectedJson =
            """{"config":{},"connections":[{"delay_ticks":0,"from_node":"grid_env","from_port":"grid_obs","to_node":"agent_policy","to_port":"grid_obs"},{"delay_ticks":1,"from_node":"agent_policy","from_port":"action","to_node":"grid_env","to_port":"action"}],"nodes":[{"config":{"height":5,"width":5},"id":"grid_env","kind":"grid_world","ports":[{"direction":"output","id":"grid_obs","port_type":{"custom":"GridObservation"}},{"direction":"input","id":"action","port_type":"string"}]},{"config":{},"id":"agent_policy","kind":"policy","ports":[{"direction":"input","id":"grid_obs","port_type":{"custom":"GridObservation"}},{"direction":"output","id":"action","port_type":"string"}]}],"schema_version":"gogamen.system_spec.v0","spec_id":"grid_5x5_system"}"""

        val spec = Fixtures.canonicalGridSystem()
        assertEquals(expectedJson, spec.toCanonicalJson())
    }

    @Test
    fun `supports all PortType variants and normalizes negative zero`() {
        val spec = systemSpec("all_types_test") {
            node("n1", "observer") {
                input("p_bool", PortType.Bool)
                input("p_i64", PortType.I64)
                input("p_u64", PortType.U64)
                output("p_f64", PortType.F64)
                output("p_str", PortType.StringType)
                output("p_unit", PortType.UnitType)
                output("p_custom", PortType.Custom("Tensor2D"))
                config {
                    "neg_zero" to -0.0
                }
            }
        }
        val json = spec.toCanonicalJson()
        assertTrue(json.contains("\"neg_zero\":0.0"), json)
        assertTrue(json.contains("{\"custom\":\"Tensor2D\"}"), json)

        assertFailsWith<IllegalArgumentException> {
            SpecConfigValue.F64(Double.NaN)
        }
        assertFailsWith<IllegalArgumentException> {
            SpecConfigValue.F64(Double.POSITIVE_INFINITY)
        }
    }

    @Test
    fun `oracle rejects unknown node name with origin context from Rust validator`() {
        val invalid = systemSpec("invalid_node_ref") {
            node("sensor", "observer") {
                output("observation", PortType.F64)
            }
            node("controller", "policy") {
                input("observation", PortType.F64)
                output("action", PortType.I64)
            }
            connections {
                instant("missing_sensor", "observation", "controller", "observation")
            }
        }

        val (exitCode, output) = runRustValidator(invalid)
        assertNotEquals(0, exitCode, "Expected Rust validator to reject unknown node")
        assertTrue(output.contains("missing_sensor"), "Expected missing node name in output:\n$output")
        assertTrue(
            output.contains("unknown source node") && output.contains("connection graph validation error"),
            "Expected origin context in output:\n$output",
        )
    }

    @Test
    fun `oracle rejects unknown port name with origin node and port from Rust validator`() {
        val invalid = systemSpec("invalid_port_ref") {
            node("sensor", "observer") {
                output("observation", PortType.F64)
            }
            node("controller", "policy") {
                input("observation", PortType.F64)
                output("action", PortType.I64)
            }
            connections {
                instant("sensor", "observation", "controller", "nonexistent_port")
            }
        }

        val (exitCode, output) = runRustValidator(invalid)
        assertNotEquals(0, exitCode, "Expected Rust validator to reject unknown port")
        assertTrue(output.contains("nonexistent_port"), "Expected missing port name in output:\n$output")
        assertTrue(output.contains("controller"), "Expected target node name in output:\n$output")
        assertTrue(
            output.contains("unknown target port") && output.contains("connection graph validation error"),
            "Expected origin context in output:\n$output",
        )
    }

    @Test
    fun `oracle rejects unregistered capability or operation with node and capability path`() {
        val invalidModule = systemSpec("invalid_capability_module") {
            node("bad_node", "nonexistent_cap::missing_op") {
                output("out", PortType.F64)
            }
        }

        val (exitCodeMod, outputMod) = runRustValidator(invalidModule)
        assertNotEquals(0, exitCodeMod, "Expected Rust validator to reject unregistered capability module")
        assertTrue(
            outputMod.contains("capability contract error"),
            "Expected capability contract error in output:\n$outputMod",
        )
        assertTrue(
            outputMod.contains("nonexistent_cap"),
            "Expected capability module name in output:\n$outputMod",
        )

        val invalidOperation = systemSpec("invalid_capability_operation") {
            node("bad_op_node", "observer::missing_op") {
                output("observation", PortType.F64)
            }
        }

        val (exitCodeOp, outputOp) = runRustValidator(invalidOperation)
        assertNotEquals(0, exitCodeOp, "Expected Rust validator to reject unregistered operation")
        assertTrue(
            outputOp.contains("capability contract error"),
            "Expected capability contract error in output:\n$outputOp",
        )
        assertTrue(
            outputOp.contains("observer") && outputOp.contains("missing_op"),
            "Expected module and operation path in output:\n$outputOp",
        )
        assertTrue(
            repoRoot().resolve("Cargo.toml").exists(),
            "Expected repoRoot() to point to Rust workspace",
        )
    }
}
