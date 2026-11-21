package dev.gogamen.dsl

import org.jetbrains.kotlin.cli.common.ExitCode
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSeverity
import org.jetbrains.kotlin.cli.common.messages.CompilerMessageSourceLocation
import org.jetbrains.kotlin.cli.common.messages.MessageCollector
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import org.jetbrains.kotlin.config.Services
import java.io.File
import java.nio.file.Files
import kotlin.io.path.absolutePathString
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class ExprDslTest {
    private data class CompileResult(
        val exitCode: ExitCode,
        val errors: List<String>,
    )

    private fun compileSnippet(snippetBody: String): CompileResult {
        val tempDir = Files.createTempDirectory("gogamen-expr-compile-test-")
        try {
            val srcFile = tempDir.resolve("Snippet.kt")
            val outDir = tempDir.resolve("out")
            Files.createDirectories(outDir)

            val source = """
                package dev.gogamen.dsl.testcompile

                import dev.gogamen.dsl.*

                fun buildExpr(): Any {
                    $snippetBody
                }
            """.trimIndent()
            Files.writeString(srcFile, source)

            val classpath = System.getProperty("java.class.path")
                .split(File.pathSeparator)
                .filter { it.isNotBlank() }
                .joinToString(File.pathSeparator)

            val errors = mutableListOf<String>()
            val collector = object : MessageCollector {
                override fun clear() {}
                override fun hasErrors(): Boolean = errors.isNotEmpty()
                override fun report(
                    severity: CompilerMessageSeverity,
                    message: String,
                    location: CompilerMessageSourceLocation?,
                ) {
                    if (severity.isError) {
                        errors += message
                    }
                }
            }

            val compiler = K2JVMCompiler()
            val args = compiler.createArguments().apply {
                freeArgs = listOf(srcFile.absolutePathString())
                destination = outDir.absolutePathString()
                this.classpath = classpath
                noStdlib = true
                noReflect = true
            }
            val exitCode = compiler.exec(collector, Services.EMPTY, args)
            return CompileResult(exitCode = exitCode, errors = errors)
        } finally {
            tempDir.toFile().deleteRecursively()
        }
    }

    @Test
    fun `energy times curiosity gt threshold matches Rust ExprAst canonical JSON byte-for-byte`() {
        val energy = scalarVar<Energy>("energy")
        val curiosity = scalarVar<Ratio>("curiosity")
        val threshold = scalarVar<Energy>("threshold")

        val expr: BoolExpr = (energy * curiosity) gt threshold

        val expectedRustCanonicalJson =
            """{"op":"gt","lhs":{"op":"mul","lhs":{"op":"var","name":"energy"},"rhs":{"op":"var","name":"curiosity"}},"rhs":{"op":"var","name":"threshold"}}"""

        assertEquals(expectedRustCanonicalJson, expr.toCanonicalJson())
    }

    @Test
    fun `all ExprUnit variants and literals match Rust ExprAst canonical JSON`() {
        val cases = listOf(
            const(1.5) to """{"op":"const_f64","unit":"dimensionless","value":1.5}""",
            2.0.energy to """{"op":"const_f64","unit":"energy","value":2.0}""",
            const(3.0, Units.Seconds) to """{"op":"const_f64","unit":"seconds","value":3.0}""",
            4.0.meters to """{"op":"const_f64","unit":"meters","value":4.0}""",
            0.25.ratio to """{"op":"const_f64","unit":"ratio","value":0.25}""",
            12.5.metersPerSecond to """{"op":"const_f64","unit":"meters_per_second","value":12.5}""",
            100.0.power to """{"op":"const_f64","unit":"power","value":100.0}""",
            const(-0.0, Units.Energy) to """{"op":"const_f64","unit":"energy","value":0.0}""",
            const(true) to """{"op":"const_bool","value":true}""",
            const(false) to """{"op":"const_bool","value":false}""",
        )

        for ((expr, expectedJson) in cases) {
            assertEquals(expectedJson, expr.toCanonicalJson())
        }
    }

    @Test
    fun `all arithmetic comparison and logical operators emit canonical Rust JSON`() {
        val e1 = scalarVar<Energy>("e1")
        val e2 = scalarVar<Energy>("e2")
        val d = scalarVar<Meters>("d")
        val t = scalarVar<Seconds>("t")
        val b1 = boolVar("b1")
        val b2 = boolVar("b2")

        assertEquals(
            """{"op":"add","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 + e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"sub","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 - e2).toCanonicalJson(),
        )
        val speed: ScalarExpr<MetersPerSecond> = d / t
        assertEquals(
            """{"op":"div","lhs":{"op":"var","name":"d"},"rhs":{"op":"var","name":"t"}}""",
            speed.toCanonicalJson(),
        )
        val distanceBack: ScalarExpr<Meters> = speed * t
        assertEquals(
            """{"op":"mul","lhs":{"op":"div","lhs":{"op":"var","name":"d"},"rhs":{"op":"var","name":"t"}},"rhs":{"op":"var","name":"t"}}""",
            distanceBack.toCanonicalJson(),
        )
        val power: ScalarExpr<Power> = e1 / t
        val energyBack: ScalarExpr<Energy> = power * t
        assertEquals(
            """{"op":"mul","lhs":{"op":"div","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"t"}},"rhs":{"op":"var","name":"t"}}""",
            energyBack.toCanonicalJson(),
        )
        val ratioSelf: ScalarExpr<Dimensionless> = e1 / e2
        assertEquals(
            """{"op":"div","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            ratioSelf.toCanonicalJson(),
        )

        assertEquals(
            """{"op":"lt","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 lt e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"le","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 le e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"ge","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 ge e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"eq","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 eq e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"ne","lhs":{"op":"var","name":"e1"},"rhs":{"op":"var","name":"e2"}}""",
            (e1 ne e2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"and","lhs":{"op":"var","name":"b1"},"rhs":{"op":"not","expr":{"op":"var","name":"b2"}}}""",
            (b1 and !b2).toCanonicalJson(),
        )
        assertEquals(
            """{"op":"or","lhs":{"op":"var","name":"b1"},"rhs":{"op":"var","name":"b2"}}""",
            (b1 or b2).toCanonicalJson(),
        )
    }

    @Test
    fun `validates non-finite floats blank var names and max depth`() {
        assertFailsWith<IllegalArgumentException> { const(Double.NaN) }
        assertFailsWith<IllegalArgumentException> { const(Double.POSITIVE_INFINITY, Units.Energy) }
        assertFailsWith<IllegalArgumentException> { scalarVar<Energy>("   ") }
        assertFailsWith<IllegalArgumentException> { boolVar("") }

        var deep: BoolExpr = const(true)
        repeat(EXPR_AST_MAX_DEPTH) {
            deep = !deep
        }
        assertFailsWith<IllegalArgumentException> { deep.toCanonicalJson() }
    }

    @Test
    fun `kotlin compiler accepts valid typed expression and rejects incompatible units and types`() {
        val valid = compileSnippet(
            """
            val e = scalarVar<Energy>("energy")
            val c = scalarVar<Ratio>("curiosity")
            val t = scalarVar<Energy>("threshold")
            return (e * c) gt t
            """.trimIndent(),
        )
        assertEquals(
            ExitCode.OK,
            valid.exitCode,
            "Expected valid typed expression to compile, got errors: ${valid.errors}",
        )

        val invalidSnippets = listOf(
            "Energy + Seconds" to """
                val e = scalarVar<Energy>("e")
                val s = scalarVar<Seconds>("s")
                return e + s
            """.trimIndent(),
            "Energy gt Seconds" to """
                val e = scalarVar<Energy>("e")
                val s = scalarVar<Seconds>("s")
                return e gt s
            """.trimIndent(),
            "Energy * Seconds" to """
                val e = scalarVar<Energy>("e")
                val s = scalarVar<Seconds>("s")
                return e * s
            """.trimIndent(),
            "Scalar used where BoolExpr expected (and)" to """
                val e = scalarVar<Energy>("e")
                val b = boolVar("b")
                return e and b
            """.trimIndent(),
            "Scalar used with logical not" to """
                val d = scalarVar<Dimensionless>("d")
                return !d
            """.trimIndent(),
            "Scalar eq BoolExpr" to """
                val e = scalarVar<Energy>("e")
                val b = boolVar("b")
                return e eq b
            """.trimIndent(),
        )

        for ((label, snippet) in invalidSnippets) {
            val result = compileSnippet(snippet)
            assertNotEquals(
                ExitCode.OK,
                result.exitCode,
                "Expected compile-time rejection for '$label', but compilation succeeded",
            )
            assertTrue(
                result.errors.isNotEmpty(),
                "Expected compiler error diagnostics for '$label'",
            )
        }
    }
}
