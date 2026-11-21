package dev.gogamen.dsl.script

import kotlin.script.experimental.annotations.KotlinScript
import kotlin.script.experimental.api.ScriptCompilationConfiguration
import kotlin.script.experimental.api.defaultImports
import kotlin.script.experimental.api.implicitReceivers
import kotlin.script.experimental.jvm.dependenciesFromClassContext
import kotlin.script.experimental.jvm.jvm

/**
 * Script template for trusted authoring scripts named `*.gogamen.kts`.
 *
 * The last expression of the script must be a `dev.gogamen.dsl.SystemSpec`
 * (e.g. the result of `systemSpec(...) { ... }` or `epistemicSystemSpec(...) { ... }`).
 *
 * SECURITY: scripts are arbitrary JVM code. See `kotlin/docs/scripting-threat-model.md`.
 */
@KotlinScript(
    fileExtension = "gogamen.kts",
    compilationConfiguration = GogamenScriptCompilationConfiguration::class,
)
abstract class GogamenScript

object GogamenScriptCompilationConfiguration : ScriptCompilationConfiguration({
    defaultImports("dev.gogamen.dsl.*", "kotlin.math.*")
    jvm {
        // Expose the DSL (main output) and the Kotlin stdlib to the script compiler.
        dependenciesFromClassContext(GogamenScriptCompilationConfiguration::class, wholeClasspath = true)
    }
})
