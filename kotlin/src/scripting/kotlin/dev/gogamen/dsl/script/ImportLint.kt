package dev.gogamen.dsl.script

/**
 * Hygiene lint for authoring scripts. This is NOT a security boundary and NOT a sandbox.
 *
 * It textually flags obviously unintended use of IO / network / process / reflection APIs so
 * that authors notice mistakes early. A trivially obfuscated script bypasses it; known
 * uncovered cases are pinned in `ImportLintTest` and listed in the threat model.
 * Trusted scripts run with the full permissions of the host JVM regardless of this lint.
 */
object ImportLint {
    data class Finding(val line: Int, val column: Int, val match: String, val message: String)

    private val banned = Regex(
        """\b(java\.io|java\.nio|java\.net|javax\.net|java\.lang\.reflect|java\.lang\.invoke|""" +
            """kotlin\.reflect|java\.lang\.Runtime|java\.lang\.ProcessBuilder|java\.lang\.Process|""" +
            """Runtime\.getRuntime|ProcessBuilder)\b""",
    )

    fun lint(source: String): List<Finding> =
        source.lineSequence().withIndex().flatMap { (idx, text) ->
            banned.findAll(text).map { m ->
                Finding(
                    line = idx + 1,
                    column = m.range.first + 1,
                    match = m.value,
                    message = "hygiene lint: '${m.value}' is not expected in authoring scripts " +
                        "(lint only, not a security boundary)",
                )
            }
        }.toList()
}
