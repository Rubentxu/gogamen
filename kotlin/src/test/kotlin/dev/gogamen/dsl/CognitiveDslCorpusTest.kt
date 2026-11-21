package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * GOG-159: golden corpus of cognitive experiments. Positive cases must compile to the canonical JSON
 * checked in under `spec/golden/cognitive_dsl/positive`; the Rust test `cognitive_dsl_corpus` parses
 * the same files. Negative cases must be rejected with the code and path of `cases.tsv`.
 * Set `GOGAMEN_REGEN_CORPUS=1` to rewrite the positive files.
 */
class CognitiveDslCorpusTest {
    private fun repoRoot(): Path {
        val cwd = Path.of("").toAbsolutePath().normalize()
        return if (cwd.fileName?.toString() == "kotlin") cwd.parent else cwd
    }

    private fun corpusDir(): Path = repoRoot().resolve("spec/golden/cognitive_dsl")

    private class Case(val id: String, val code: String, val path: String, val message: String)

    private fun cases(): List<Case> = Files.readAllLines(corpusDir().resolve("cases.tsv"))
        .filter { it.isNotBlank() && !it.startsWith("#") }
        .map { line -> line.split('\t').let { Case(it[0], it[1], it[2], it[3]) } }

    @Test
    fun positiveCorpusMatchesCheckedInCanonicalJson() {
        val regen = System.getenv("GOGAMEN_REGEN_CORPUS") == "1"
        assertEquals(8, POSITIVE_CORPUS.size)
        for ((id, build) in POSITIVE_CORPUS) {
            val json = build().toCanonicalJson()
            val file = corpusDir().resolve("positive/$id.json")
            if (regen) Files.writeString(file, json)
            assertEquals(Files.readString(file), json, "positive/$id.json is stale (run with GOGAMEN_REGEN_CORPUS=1)")
        }
    }

    @Test
    fun everyPositiveExperimentFitsInFifteenLines() {
        val src = Files.readAllLines(
            repoRoot().resolve("kotlin/src/test/kotlin/dev/gogamen/dsl/CognitiveCorpus.kt"),
        )
        for (n in 0..7) {
            val begin = src.indexOf("// BEGIN H$n")
            val end = src.indexOf("// END H$n")
            assertTrue(begin >= 0 && end > begin, "markers for H$n")
            val lines = end - begin - 1
            assertTrue(lines <= 15, "H$n has $lines lines (max 15)")
        }
    }

    @Test
    fun negativeCorpusIsRejectedWithCodeAndPath() {
        val table = cases()
        assertTrue(table.size >= 7)
        assertEquals(NEGATIVE_CORPUS.keys, table.map { it.id }.toSet())
        for (c in table) {
            val e = assertFailsWith<DslRejection>(c.id) { NEGATIVE_CORPUS.getValue(c.id)() }
            assertEquals(c.code, e.diagnostic.code, "${c.id} code")
            assertEquals(c.path, e.diagnostic.path, "${c.id} path")
            assertTrue(c.message in e.diagnostic.message, "${c.id} message '${e.diagnostic.message}'")
        }
    }

    @Test
    fun grammarHasNoConstructToProgramBehavior() {
        val attempts = mapOf<String, AgentBuilder.() -> Unit>(
            "rule" to { rule("x") },
            "policy" to { policy("x") },
            "act" to { act("x") },
            "script" to { script("x") },
            "behavior" to { behavior {} },
            "onTick" to { onTick {} },
            "onObservation" to { onObservation {} },
        )
        for ((name, attempt) in attempts) {
            val e = assertFailsWith<DslRejection>(name) {
                experiment("b") { gridBase(); agent("a", attempt) }
            }
            assertEquals(DslCodes.BEHAVIOR_RULE, e.diagnostic.code)
            assertEquals("agents[0].$name", e.diagnostic.path)
        }
    }

    @Test
    fun varyExpandsToTheExactDeclaredMatrix() {
        val expected = mapOf("h0" to 5, "h1" to 15, "h2" to 10, "h3" to 30, "h4" to 20, "h5" to 40, "h6" to 5, "h7" to 5)
        for ((id, n) in expected) assertEquals(n, POSITIVE_CORPUS.getValue(id)().matrix.size, id)

        val h3 = POSITIVE_CORPUS.getValue("h3")().matrix
        assertEquals(setOf(0.01, 0.1, 0.5), h3.map { it.assignments.getValue("learning.learning_rate") }.toSet())
        // Seeds are the innermost axis: dev then holdout for every cell, indices contiguous.
        assertEquals(listOf(11L, 12L, 13L, 1001L, 1002L), h3.take(5).map { it.seed })
        assertEquals(listOf("dev", "dev", "dev", "holdout", "holdout"), h3.take(5).map { it.split })
        assertEquals(h3.indices.toList(), h3.map { it.index })
        assertEquals(h3.size, h3.map { Triple(it.assignments, it.seed, it.split) }.toSet().size)
    }

    @Test
    fun matrixLargerThanTheLimitIsRejected() {
        val e = assertFailsWith<DslRejection> {
            experiment("big") {
                gridBase()
                gridExplorer()
                cognitive {
                    resolution("v0") { level("only", "local_fov_sensor", 9) }
                    learning { algorithm = LearningAlgorithm.NONE }
                }
                vary("entropic.horizon", *DoubleArray(2500) { it + 1.0 })
            }
        }
        assertEquals(DslCodes.BAD_VARY, e.diagnostic.code)
    }

    @Test
    fun phenomenaAreEvaluatedOverTracesOutsideTheAgent() {
        val ph = POSITIVE_CORPUS.getValue("h6")().phenomena.associateBy { it.id }
        fun tick(e: Double, r: Double, g: Double): TraceSample = mapOf("energy" to e, "reward" to r, "info_gain" to g)

        val ok = listOf(tick(1.0, 0.0, 0.5), tick(0.8, 1.0, 0.0), tick(0.6, 0.0, 0.4), tick(0.4, 0.0, 0.0))
        assertTrue(ph.getValue("stays_alive").let { TraceChecker.holds(it, ok) })
        assertTrue(TraceChecker.holds(ph.getValue("finds_reward"), ok))
        assertTrue(TraceChecker.holds(ph.getValue("alive_after_reward"), ok))
        assertTrue(TraceChecker.holds(ph.getValue("mostly_informative"), ok))

        val dying = listOf(tick(1.0, 0.0, 0.0), tick(0.5, 1.0, 0.0), tick(0.0, 0.0, 0.0))
        assertTrue(!TraceChecker.holds(ph.getValue("stays_alive"), dying))
        assertTrue(!TraceChecker.holds(ph.getValue("alive_after_reward"), dying))
        assertTrue(!TraceChecker.holds(ph.getValue("mostly_informative"), dying))
        // `eventually` with a deadline: the reward arrives at tick index 1, but a window of 1 tick misses it.
        val early = PhenomenonSpec.Eventually("e", ExprAst.Gt(ExprAst.Var("reward"), ExprAst.ConstF64(0.0)), 1)
        assertTrue(!TraceChecker.holds(early, dying))
        // `after` is vacuously true when the trigger never fires.
        assertTrue(TraceChecker.holds(ph.getValue("alive_after_reward"), listOf(tick(0.0, 0.0, 0.0))))
    }

    @Test
    fun plainExperimentOmitsOptionalQuestionKeys() {
        val h0 = h0().toCanonicalJson()
        for (key in listOf("phenomena_json", "vary_json", "readings_json", "\"mode\"", "privileged", "observes", "\"hypotheses\"")) {
            assertTrue(key !in h0, "H0 must not contain $key")
        }
        assertTrue("\"hypotheses\":" in h7().toCanonicalJson())
    }
}
