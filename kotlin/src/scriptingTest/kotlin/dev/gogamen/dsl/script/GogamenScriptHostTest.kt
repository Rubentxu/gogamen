package dev.gogamen.dsl.script

import dev.gogamen.dsl.Fixtures
import java.nio.file.Files
import java.nio.file.Path
import kotlin.system.measureNanoTime
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

class GogamenScriptHostTest {
    private val scriptFile = Path.of("src/scriptingTest/resources/language_world.gogamen.kts").toFile()
    private val golden = Files.readString(
        Path.of("..").resolve("crates/gogamen-spec/fixtures/language_world_canonical_v0.json"),
    )

    @Test
    fun `AC1 script output equals library output and Rust golden byte-for-byte`() {
        val outcome = GogamenScriptHost().evalFile(scriptFile)
        assertIs<ScriptOutcome.Success>(outcome, outcome.toString())
        val json = outcome.spec.toCanonicalJson()
        assertEquals(Fixtures.canonicalLanguageWorld().toCanonicalJson(), json)
        assertEquals(golden, json)
        assertEquals(golden.toByteArray(Charsets.UTF_8).toList(), json.toByteArray(Charsets.UTF_8).toList())
    }

    @Test
    fun `defaultImports expose kotlin math`() {
        val outcome = GogamenScriptHost().evalText(
            """
            systemSpec("m") {
                node("a", "k") { output("o", PortType.F64); config { "v" to sqrt(16.0) } }
            }
            """.trimIndent(),
        )
        assertIs<ScriptOutcome.Success>(outcome, outcome.toString())
        assertTrue(outcome.spec.toCanonicalJson().contains("\"v\":4.0"))
    }

    @Test
    fun `AC1 compile error reports line and column`() {
        val outcome = GogamenScriptHost().evalText("val a = 1\nval b: Int = \"x\"\nsystemSpec(\"s\") {}\n")
        assertIs<ScriptOutcome.Failure>(outcome)
        val d = outcome.diagnostics.first { it.severity == "ERROR" }
        assertEquals(2, d.line, outcome.toString())
        assertEquals(12, d.column, outcome.toString())
    }

    @Test
    fun `unresolved reference reports position`() {
        val outcome = GogamenScriptHost().evalText("systemSpec(\"s\") {\n  nodee(\"a\", \"k\")\n}\n")
        assertIs<ScriptOutcome.Failure>(outcome)
        val d = outcome.diagnostics.first { it.severity == "ERROR" }
        assertEquals(2, d.line, outcome.toString())
        assertEquals(3, d.column, outcome.toString())
    }

    @Test
    fun `wrong result type is a failure`() {
        val outcome = GogamenScriptHost().evalText("42")
        assertIs<ScriptOutcome.Failure>(outcome)
        assertTrue(outcome.diagnostics.single().message.contains("SystemSpec"))
    }

    @Test
    fun `runtime exception in script is a failure`() {
        val outcome = GogamenScriptHost().evalText("error(\"boom\")")
        assertIs<ScriptOutcome.Failure>(outcome)
        assertTrue(outcome.diagnostics.any { it.message.contains("boom") }, outcome.toString())
    }

    /** Not an assertion-heavy test: prints measurements used by the ADR (AC3). */
    @Test
    fun `measure cold and warm evaluation`() {
        val host = GogamenScriptHost()
        val lib = List(5) { measureNanoTime { Fixtures.canonicalLanguageWorld().toCanonicalJson() } }
        val cold = measureNanoTime { host.evalFile(scriptFile) }
        val warm = List(5) { measureNanoTime { host.evalFile(scriptFile) } }
        fun ms(n: Long) = "%.2f".format(n / 1e6)
        println("MEASURE library first=${ms(lib.first())}ms warm-min=${ms(lib.drop(1).min())}ms")
        println("MEASURE script cold=${ms(cold)}ms warm(5 runs, new compile each, host cached)=${warm.joinToString { ms(it) }}ms")
    }
}
