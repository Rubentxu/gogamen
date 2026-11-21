package dev.gogamen.dsl.script

import dev.gogamen.dsl.SystemSpec
import java.io.File
import kotlin.script.experimental.api.EvaluationResult
import kotlin.script.experimental.api.ResultValue
import kotlin.script.experimental.api.ResultWithDiagnostics
import kotlin.script.experimental.api.ScriptDiagnostic
import kotlin.script.experimental.api.SourceCode
import kotlin.script.experimental.api.valueOrNull
import kotlin.script.experimental.host.toScriptSource
import kotlin.script.experimental.jvm.util.isError
import kotlin.script.experimental.jvmhost.BasicJvmScriptingHost
import kotlin.script.experimental.jvmhost.createJvmEvaluationConfigurationFromTemplate

/** A diagnostic with a 1-based source position (null when the engine gives none). */
data class ScriptDiag(
    val severity: String,
    val message: String,
    val line: Int?,
    val column: Int?,
) {
    override fun toString(): String = "$severity ${line ?: "?"}:${column ?: "?"}: $message"
}

sealed interface ScriptOutcome {
    data class Success(val spec: SystemSpec, val warnings: List<ScriptDiag>) : ScriptOutcome
    data class Failure(val diagnostics: List<ScriptDiag>) : ScriptOutcome
}

/**
 * Minimal host that evaluates TRUSTED `*.gogamen.kts` authoring scripts into a [SystemSpec].
 *
 * Never feed untrusted input (received specs, user uploads, network data) to this class:
 * evaluation executes arbitrary JVM code with the process permissions. Untrusted JSON must
 * go through the existing spec parser/validator only.
 */
class GogamenScriptHost(private val lintEnabled: Boolean = true) {
    private val host = BasicJvmScriptingHost()
    private val evalConfig = createJvmEvaluationConfigurationFromTemplate<GogamenScript>()

    fun evalFile(file: File): ScriptOutcome = eval(file.toScriptSource())

    fun evalText(text: String, name: String = "inline.gogamen.kts"): ScriptOutcome =
        eval(TextSource(text, name))

    fun eval(source: SourceCode): ScriptOutcome {
        if (lintEnabled) {
            val text = source.text
            val findings = ImportLint.lint(text)
            if (findings.isNotEmpty()) {
                return ScriptOutcome.Failure(findings.map { ScriptDiag("ERROR", it.message, it.line, it.column) })
            }
        }
        val result: ResultWithDiagnostics<EvaluationResult> =
            host.eval(source, GogamenScriptCompilationConfiguration, evalConfig)
        val diags = result.reports
            .filter { it.severity > ScriptDiagnostic.Severity.DEBUG }
            .map { it.toDiag() }
        if (result.isError()) return ScriptOutcome.Failure(diags)
        return when (val rv = result.valueOrNull()?.returnValue) {
            is ResultValue.Value -> (rv.value as? SystemSpec)
                ?.let { ScriptOutcome.Success(it, diags) }
                ?: ScriptOutcome.Failure(
                    diags + ScriptDiag(
                        "ERROR",
                        "script must end with a SystemSpec expression, got ${rv.value?.let { it::class.qualifiedName }}",
                        null, null,
                    ),
                )
            is ResultValue.Error -> ScriptOutcome.Failure(
                diags + ScriptDiag("ERROR", "script threw ${rv.error}", null, null),
            )
            else -> ScriptOutcome.Failure(diags + ScriptDiag("ERROR", "script produced no value", null, null))
        }
    }

    private fun ScriptDiagnostic.toDiag() = ScriptDiag(
        severity.name,
        message,
        location?.start?.line,
        location?.start?.col,
    )

    private class TextSource(override val text: String, override val name: String) : SourceCode {
        override val locationId: String? = name
    }
}
