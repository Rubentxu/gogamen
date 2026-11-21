package dev.gogamen.dsl.script

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs
import kotlin.test.assertTrue

/** The lint is hygiene only. These tests also pin what it does NOT cover. */
class ImportLintTest {
    @Test
    fun `flags banned imports with line and column`() {
        val f = ImportLint.lint("import dev.gogamen.dsl.*\nimport java.io.File\n")
        assertEquals(1, f.size)
        assertEquals(2, f[0].line)
        assertEquals(8, f[0].column)
    }

    @Test
    fun `flags fully qualified use and process APIs`() {
        val src = "val r = java.lang.Runtime.getRuntime()\nval p = ProcessBuilder(\"ls\")\n"
        val f = ImportLint.lint(src)
        assertTrue(f.any { it.line == 1 } && f.any { it.line == 2 }, f.toString())
    }

    @Test
    fun `host rejects linted script before evaluation`() {
        val outcome = GogamenScriptHost().evalText("import java.net.URL\nsystemSpec(\"s\") {}\n")
        assertIs<ScriptOutcome.Failure>(outcome)
        assertEquals(2 - 1, outcome.diagnostics.single().line)
    }

    @Test
    fun `KNOWN BYPASS not covered - kotlin io path reads files with no banned token`() {
        assertTrue(ImportLint.lint("""kotlin.io.path.Path("/etc/hostname").toFile().readText()""").isEmpty())
    }

    @Test
    fun `KNOWN BYPASS not covered - string concatenation plus Class forName`() {
        val src = """Class.forName("java." + "lang." + "Runtime")"""
        assertTrue(ImportLint.lint(src).isEmpty())
    }

    @Test
    fun `KNOWN BYPASS demonstrated end to end - lint passes and script reads a file`() {
        val host = GogamenScriptHost()
        val outcome = host.evalText(
            """
            val txt = kotlin.io.path.Path("/etc/hostname").toFile().readText()
            systemSpec("bypass_" + txt.isNotEmpty()) {}
            """.trimIndent(),
        )
        assertIs<ScriptOutcome.Success>(outcome, outcome.toString())
        assertEquals("bypass_true", outcome.spec.specId)
    }
}
