package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.absolutePathString
import kotlin.system.measureNanoTime
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class FrontendEquivalenceTest {
    private fun repoRoot(): Path {
        val cwd = Path.of("").toAbsolutePath().normalize()
        return if (cwd.fileName?.toString() == "kotlin") cwd.parent else cwd
    }

    private fun runCli(vararg args: String): Pair<Int, String> {
        val root = repoRoot()
        val cmd = listOf("cargo", "run", "-q", "-p", "gogamen-cli", "--offline", "--") + args
        val process = ProcessBuilder(cmd)
            .directory(root.toFile())
            .redirectErrorStream(true)
            .start()
        val output = process.inputStream.bufferedReader().readText()
        val exitCode = process.waitFor()
        return exitCode to output
    }

    private fun extractField(output: String, field: String): String {
        val prefix = "$field: "
        return output.lines()
            .firstNotNullOfOrNull { line ->
                val trimmed = line.trim()
                if (trimmed.startsWith(prefix)) trimmed.removePrefix(prefix) else null
            }
            ?: error("Missing field '$field' in CLI output:\n$output")
    }

    @Test
    fun `kotlin dsl canonicalTwoNode and canonicalCognitiveAgent compile and execute equivalently across seeds`() {
        val expectedCognitiveJson =
            """{"config":{"deterministic":true,"dt_secs":0.02},"connections":[{"delay_ticks":0,"from_node":"sensor","from_port":"observation","to_node":"belief","to_port":"observation"},{"delay_ticks":0,"from_node":"sensor","from_port":"energy","to_node":"belief","to_port":"energy"},{"delay_ticks":0,"from_node":"belief","from_port":"belief_score","to_node":"planner","to_port":"belief_score"},{"delay_ticks":0,"from_node":"belief","from_port":"should_explore","to_node":"planner","to_port":"should_explore"}],"nodes":[{"config":{"base_energy":1.0,"noise_scale":0.1},"id":"sensor","kind":"agent::sensor","ports":[{"direction":"output","id":"observation","port_type":"f64"},{"direction":"output","id":"energy","port_type":"f64"}]},{"config":{"curiosity":0.6,"expr_json":"{\"op\":\"gt\",\"lhs\":{\"op\":\"mul\",\"lhs\":{\"op\":\"var\",\"name\":\"energy\"},\"rhs\":{\"op\":\"var\",\"name\":\"curiosity\"}},\"rhs\":{\"op\":\"var\",\"name\":\"threshold\"}}","threshold":0.4},"id":"belief","kind":"agent::belief","ports":[{"direction":"input","id":"observation","port_type":"f64"},{"direction":"input","id":"energy","port_type":"f64"},{"direction":"output","id":"belief_score","port_type":"f64"},{"direction":"output","id":"should_explore","port_type":"bool"}]},{"config":{"exploit_gain":1.0,"explore_gain":2.0},"id":"planner","kind":"agent::planner","ports":[{"direction":"input","id":"belief_score","port_type":"f64"},{"direction":"input","id":"should_explore","port_type":"bool"},{"direction":"output","id":"action","port_type":"i64"}]}],"schema_version":"gogamen.system_spec.v0","spec_id":"cognitive_agent_canonical_v0"}"""

        val tempDir = Files.createTempDirectory("gogamen-kt-equiv-")
        try {
            val seeds = listOf(42L, 7L, 2026L)

            // 1. canonicalTwoNode: Kotlin DSL vs Rust --fixture
            lateinit var twoNodeJson: String
            val twoNodeAuthNs = measureNanoTime {
                twoNodeJson = Fixtures.canonicalTwoNode().toCanonicalJson()
            }
            val twoNodeSpecFile = tempDir.resolve("two_node.json")
            val twoNodeArtKt = tempDir.resolve("two_node_kt.artifact.json")
            val twoNodeArtRust = tempDir.resolve("two_node_rust.artifact.json")
            Files.writeString(twoNodeSpecFile, twoNodeJson)

            lateinit var compileKtOut: String
            val twoNodeCompileNs = measureNanoTime {
                val (code, out) = runCli(
                    "compile-artifact",
                    "--json-file",
                    twoNodeSpecFile.absolutePathString(),
                    "--out",
                    twoNodeArtKt.absolutePathString(),
                )
                assertEquals(0, code, "compile-artifact failed:\n$out")
                compileKtOut = out
            }
            val (codeRustComp, compileRustOut) = runCli(
                "compile-artifact",
                "--fixture",
                "--out",
                twoNodeArtRust.absolutePathString(),
            )
            assertEquals(0, codeRustComp, "compile-artifact --fixture failed:\n$compileRustOut")

            assertEquals(
                extractField(compileRustOut, "spec_hash"),
                extractField(compileKtOut, "spec_hash"),
            )
            assertEquals(
                extractField(compileRustOut, "artifact_hash"),
                extractField(compileKtOut, "artifact_hash"),
            )
            assertEquals(Files.readString(twoNodeArtRust), Files.readString(twoNodeArtKt))

            var twoNodeRunNs = 0L
            for (seed in seeds) {
                lateinit var runKtOut: String
                twoNodeRunNs += measureNanoTime {
                    val (code, out) = runCli(
                        "run-artifact",
                        "--artifact-file",
                        twoNodeArtKt.absolutePathString(),
                        "--ticks",
                        "25",
                        "--seed",
                        seed.toString(),
                    )
                    assertEquals(0, code, "run-artifact failed:\n$out")
                    runKtOut = out
                }
                val (codeRustRun, runRustOut) = runCli(
                    "run-artifact",
                    "--artifact-file",
                    twoNodeArtRust.absolutePathString(),
                    "--ticks",
                    "25",
                    "--seed",
                    seed.toString(),
                )
                assertEquals(0, codeRustRun, "run-artifact rust failed:\n$runRustOut")
                assertEquals(
                    extractField(runRustOut, "semantic_fingerprint"),
                    extractField(runKtOut, "semantic_fingerprint"),
                )
                assertEquals(
                    extractField(runRustOut, "trace_hash"),
                    extractField(runKtOut, "trace_hash"),
                )
            }

            // 2. canonicalCognitiveAgent: Kotlin DSL (with Expr) -> compile-artifact -> run-artifact across seeds
            lateinit var cogJson: String
            val cogAuthNs = measureNanoTime {
                cogJson = Fixtures.canonicalCognitiveAgent().toCanonicalJson()
            }
            assertEquals(expectedCognitiveJson, cogJson)

            val cogSpecFile = tempDir.resolve("cognitive_agent.json")
            val cogArtFile = tempDir.resolve("cognitive_agent.artifact.json")
            Files.writeString(cogSpecFile, cogJson)

            lateinit var cogCompileOut: String
            val cogCompileNs = measureNanoTime {
                val (code, out) = runCli(
                    "compile-artifact",
                    "--json-file",
                    cogSpecFile.absolutePathString(),
                    "--out",
                    cogArtFile.absolutePathString(),
                )
                assertEquals(0, code, "cognitive compile-artifact failed:\n$out")
                cogCompileOut = out
            }
            assertTrue(cogCompileOut.contains("spec_id: cognitive_agent_canonical_v0"))

            val (inspectCode, inspectOut) = runCli(
                "inspect-artifact",
                "--artifact-file",
                cogArtFile.absolutePathString(),
            )
            assertEquals(0, inspectCode, "inspect-artifact failed:\n$inspectOut")
            assertTrue(inspectOut.contains("nodes: 3"), inspectOut)

            val traceHashes = mutableListOf<String>()
            var cogRunNs = 0L
            for (seed in seeds) {
                lateinit var runArtOut: String
                cogRunNs += measureNanoTime {
                    val (code, out) = runCli(
                        "run-artifact",
                        "--artifact-file",
                        cogArtFile.absolutePathString(),
                        "--ticks",
                        "25",
                        "--seed",
                        seed.toString(),
                    )
                    assertEquals(0, code, "cognitive run-artifact failed:\n$out")
                    runArtOut = out
                }
                val (specCode, runSpecOut) = runCli(
                    "run-spec",
                    "--json-file",
                    cogSpecFile.absolutePathString(),
                    "--ticks",
                    "25",
                    "--seed",
                    seed.toString(),
                )
                assertEquals(0, specCode, "cognitive run-spec failed:\n$runSpecOut")
                assertEquals(
                    extractField(runSpecOut, "semantic_fingerprint"),
                    extractField(runArtOut, "semantic_fingerprint"),
                )
                val traceHash = extractField(runArtOut, "trace_hash")
                assertEquals(extractField(runSpecOut, "trace_hash"), traceHash)
                traceHashes += traceHash
            }
            assertNotEquals(traceHashes[0], traceHashes[1], "Different seeds must yield distinct traces")

            val notice = "Aviso: no se promete coste cero de abstracción; se miden por separado autoría Kotlin, compilación CLI y ejecución CLI."
            println(
                "$notice\n" +
                    "  two_node_canonical_v0: authoring_ns=$twoNodeAuthNs compile_cli_ns=$twoNodeCompileNs run_cli_ns=$twoNodeRunNs\n" +
                    "  cognitive_agent_canonical_v0: authoring_ns=$cogAuthNs compile_cli_ns=$cogCompileNs run_cli_ns=$cogRunNs",
            )
            assertTrue(notice.contains("no se promete coste cero de abstracción"))
        } finally {
            tempDir.toFile().deleteRecursively()
        }
    }
}
