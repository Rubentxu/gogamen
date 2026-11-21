package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.absolutePathString
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class EpistemicDslTest {
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

    private fun runRustSpec(spec: SystemSpec, ticks: Int = 5, seed: Long = 42L): Pair<Int, String> {
        val jsonFile = Files.createTempFile("gogamen-epistemic-", ".json")
        try {
            Files.writeString(jsonFile, spec.toCanonicalJson())
            return runCli(
                "run-spec",
                "--json-file",
                jsonFile.absolutePathString(),
                "--ticks",
                ticks.toString(),
                "--seed",
                seed.toString(),
            )
        } finally {
            Files.deleteIfExists(jsonFile)
        }
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
    fun `epistemic builders carry GogamenDsl and encode Choice Noul Score Abstention Ask Inspect`() {
        val dslClasses = listOf(
            ChoiceQuestionBuilder::class.java,
            NoulQuestionBuilder::class.java,
            ScoreQuestionBuilder::class.java,
            AbstentionPolicyBuilder::class.java,
            AskActionBuilder::class.java,
            InspectActionBuilder::class.java,
            ProvenanceBuilder::class.java,
            GuardPolicyBuilder::class.java,
        )
        for (cls in dslClasses) {
            assertTrue(
                cls.isAnnotationPresent(GogamenDsl::class.java),
                "${cls.simpleName} must be annotated with @GogamenDsl",
            )
        }

        val choice = EpistemicQuestion.Choice(
            queryId = "q-choice",
            prompt = "Select cell",
            options = listOf("cell_a", "cell_b"),
            modelId = "system_one_choice_v1",
            deadlineTicks = 4L,
        )
        assertTrue(choice.toCanonicalJson().contains("\"kind\":\"choice\""))

        val noul = EpistemicQuestion.Noul(
            queryId = "q-noul",
            prompt = "Is path clear?",
            hypothesisId = "path_clear",
            modelId = "system_one_noul_v1",
        )
        assertTrue(noul.toCanonicalJson().contains("\"kind\":\"noul\""))

        val score = EpistemicQuestion.Score(
            queryId = "q-score",
            prompt = "Score candidates",
            options = listOf("opt1", "opt2"),
            minScore = 0.0,
            maxScore = 10.0,
            modelId = "system_one_score_v1",
        )
        assertTrue(score.toCanonicalJson().contains("\"kind\":\"score\""))
        assertTrue(score.toCanonicalJson().contains("\"max_score\":10.0"))
    }

    @Test
    fun `AC1 canonicalLanguageWorld JSON equals Rust golden fixture byte-for-byte`() {
        val golden = Files.readString(
            repoRoot().resolve("crates/gogamen-spec/fixtures/language_world_canonical_v0.json"),
        )
        assertEquals(golden, Fixtures.canonicalLanguageWorld().toCanonicalJson())
    }

    @Test
    fun `AC1 canonicalLanguageWorld compiles and executes in Rust CLI with deterministic trace across seeds`() {
        val spec = Fixtures.canonicalLanguageWorld()
        val tempDir = Files.createTempDirectory("gogamen-lw-test-")
        try {
            val specFile = tempDir.resolve("language_world.json")
            val artFile = tempDir.resolve("language_world.artifact.json")
            Files.writeString(specFile, spec.toCanonicalJson())

            val (compCode, compOut) = runCli(
                "compile-artifact",
                "--json-file",
                specFile.absolutePathString(),
                "--out",
                artFile.absolutePathString(),
            )
            assertEquals(0, compCode, "compile-artifact failed:\n$compOut")
            assertTrue(compOut.contains("spec_id: language_world_canonical_v0"), compOut)

            val (inspCode, inspOut) = runCli(
                "inspect-artifact",
                "--artifact-file",
                artFile.absolutePathString(),
            )
            assertEquals(0, inspCode, "inspect-artifact failed:\n$inspOut")
            assertTrue(inspOut.contains("nodes: 5"), inspOut)

            val (runA1, outA1) = runCli(
                "run-artifact",
                "--artifact-file",
                artFile.absolutePathString(),
                "--ticks",
                "10",
                "--seed",
                "42",
            )
            val (runA2, outA2) = runCli(
                "run-artifact",
                "--artifact-file",
                artFile.absolutePathString(),
                "--ticks",
                "10",
                "--seed",
                "42",
            )
            val (runB, outB) = runCli(
                "run-artifact",
                "--artifact-file",
                artFile.absolutePathString(),
                "--ticks",
                "10",
                "--seed",
                "99",
            )
            assertEquals(0, runA1, outA1)
            assertEquals(0, runA2, outA2)
            assertEquals(0, runB, outB)

            assertEquals(extractField(outA1, "semantic_fingerprint"), extractField(outA2, "semantic_fingerprint"))
            assertEquals(extractField(outA1, "trace_hash"), extractField(outA2, "trace_hash"))
            assertNotEquals(extractField(outA1, "trace_hash"), extractField(outB, "trace_hash"))
        } finally {
            tempDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `AC2 human messages cannot mutate permissions and provenance source_id sensor_id evidence_ref are recorded`() {
        val spec = Fixtures.canonicalLanguageWorld()
        val json = spec.toCanonicalJson()

        // Explicit provenance fields recorded in Spec
        val humanNode = spec.nodes.first { it.id == "human_channel" }
        assertEquals(SpecConfigValue.Str("human_channel"), humanNode.config["source_id"])
        assertEquals(SpecConfigValue.I64(100L), humanNode.config["sensor_id"])
        assertEquals(SpecConfigValue.I64(101L), humanNode.config["evidence_ref"])
        assertTrue(json.contains("\"source_id\":\"human_channel\""), json)
        assertTrue(json.contains("\"evidence_ref\":101"), json)

        val directNode = spec.nodes.first { it.id == "direct_sensor" }
        assertEquals(SpecConfigValue.Str("direct_sensor"), directNode.config["source_id"])
        assertEquals(SpecConfigValue.I64(200L), directNode.config["sensor_id"])
        assertEquals(SpecConfigValue.I64(500L), directNode.config["evidence_ref"])

        // Immutable guard / permission policy
        val guardNode = spec.nodes.first { it.id == "guarded_planner" }
        assertEquals(SpecConfigValue.Str("reject_only"), guardNode.config["guard_policy"])
        assertEquals(SpecConfigValue.Bool(true), guardNode.config["immutable_permissions"])
        assertEquals(SpecConfigValue.Bool(false), guardNode.config["human_can_mutate_permissions"])

        // Kotlin DSL rejects mutable permissions
        assertFailsWith<IllegalArgumentException> {
            GuardPolicySpec(humanMessagesCannotMutatePermissions = false)
        }
        assertFailsWith<IllegalArgumentException> {
            GuardPolicySpec(immutablePermissions = false)
        }

        // Rust validator also rejects a raw Spec attempting to allow human permission mutation
        val tamperedSpec = systemSpec("tampered_guard") {
            node("guarded_planner", "language::guarded_planner") {
                input("belief_prob", PortType.F64)
                input("should_ask", PortType.Bool)
                input("should_inspect", PortType.Bool)
                input("is_order", PortType.Bool)
                output("action", PortType.I64)
                output("guard_rejected", PortType.Bool)
                config {
                    "guard_policy" to "reject_only"
                    "immutable_permissions" to true
                    "human_can_mutate_permissions" to true
                }
            }
        }
        val (exitCode, out) = runRustSpec(tamperedSpec)
        assertNotEquals(0, exitCode)
        assertTrue(out.contains("human messages cannot mutate ActionGuard or PermissionSet"), out)
    }

    @Test
    fun `AC3 Kotlin builders and Rust validator reject incompatible references and unavailable models`() {
        // 1. Unavailable model rejected by Kotlin epistemicSystemSpec
        val ktModelErr = assertFailsWith<IllegalArgumentException> {
            epistemicSystemSpec("bad_model_kt") {
                node("human_channel", "language::human_channel") {
                    output("observation", PortType.F64)
                    output("message_confidence", PortType.F64)
                    output("is_order", PortType.Bool)
                    provenance {
                        sourceId = "human_channel"
                        sensorId = 100L
                        evidenceRef = 101L
                    }
                }
                node("grounder", "language::grounder") {
                    input("observation", PortType.F64)
                    input("message_confidence", PortType.F64)
                    output("grounded_prob", PortType.F64)
                    output("abstained", PortType.Bool)
                    choiceQuestion("q1") {
                        prompt = "Pick"
                        options("a", "b")
                        modelId = "nonexistent_llm_v99"
                    }
                    config {
                        "source_id" to "human_channel"
                    }
                }
            }
        }
        assertTrue(ktModelErr.message!!.contains("Unavailable model 'nonexistent_llm_v99'"), ktModelErr.message)

        // 2. Unavailable model rejected by Rust validator
        val rawBadModel = systemSpec("bad_model_rust") {
            node("human_channel", "language::human_channel") {
                output("observation", PortType.F64)
                output("message_confidence", PortType.F64)
                output("is_order", PortType.Bool)
                provenance {
                    sourceId = "human_channel"
                    sensorId = 100L
                    evidenceRef = 101L
                }
            }
            node("grounder", "language::grounder") {
                input("observation", PortType.F64)
                input("message_confidence", PortType.F64)
                output("grounded_prob", PortType.F64)
                output("abstained", PortType.Bool)
                choiceQuestion("q1") {
                    prompt = "Pick"
                    options("a", "b")
                    modelId = "nonexistent_llm_v99"
                }
                config {
                    "source_id" to "human_channel"
                }
            }
        }
        val (rustModelCode, rustModelOut) = runRustSpec(rawBadModel)
        assertNotEquals(0, rustModelCode)
        assertTrue(rustModelOut.contains("unavailable model 'nonexistent_llm_v99'"), rustModelOut)

        // 3. Incompatible source reference rejected by both Kotlin and Rust
        val ktSourceErr = assertFailsWith<IllegalArgumentException> {
            epistemicSystemSpec("bad_source_kt") {
                node("grounder", "language::grounder") {
                    input("observation", PortType.F64)
                    input("message_confidence", PortType.F64)
                    output("grounded_prob", PortType.F64)
                    output("abstained", PortType.Bool)
                    choiceQuestion("q1") {
                        prompt = "Pick"
                        options("a", "b")
                        modelId = "offline_replay_v1"
                    }
                    config {
                        "source_id" to "missing_sensor_source"
                    }
                }
            }
        }
        assertTrue(ktSourceErr.message!!.contains("Incompatible source reference 'missing_sensor_source'"), ktSourceErr.message)

        val rawBadSource = systemSpec("bad_source_rust") {
            node("grounder", "language::grounder") {
                input("observation", PortType.F64)
                input("message_confidence", PortType.F64)
                output("grounded_prob", PortType.F64)
                output("abstained", PortType.Bool)
                choiceQuestion("q1") {
                    prompt = "Pick"
                    options("a", "b")
                    modelId = "offline_replay_v1"
                }
                config {
                    "source_id" to "missing_sensor_source"
                }
            }
        }
        val (rustSourceCode, rustSourceOut) = runRustSpec(rawBadSource)
        assertNotEquals(0, rustSourceCode)
        assertTrue(rustSourceOut.contains("incompatible source reference 'missing_sensor_source'"), rustSourceOut)

        // 4. Incompatible node/port reference rejected by Kotlin epistemicSystemSpec
        val ktConnErr = assertFailsWith<IllegalArgumentException> {
            epistemicSystemSpec("bad_conn_kt") {
                node("human_channel", "language::human_channel") {
                    output("observation", PortType.F64)
                    output("message_confidence", PortType.F64)
                    output("is_order", PortType.Bool)
                    provenance {
                        sourceId = "human_channel"
                        sensorId = 100L
                        evidenceRef = 101L
                    }
                }
                connections {
                    instant("human_channel", "missing_port", "missing_node", "in")
                }
            }
        }
        assertTrue(ktConnErr.message!!.contains("Incompatible connection"), ktConnErr.message)
    }
}
