# Threat model: Kotlin authoring scripts (`*.gogamen.kts`)

Scope: the optional `scripting` source set (`dev.gogamen.dsl.script`). GOG-83 spike.

## Two kinds of input

| Input | Trust | Path | Allowed through script evaluator? |
|---|---|---|---|
| Authoring code: `*.gogamen.kts` written/reviewed by the project author | Trusted | `GogamenScriptHost` -> `SystemSpec` -> canonical JSON | Yes |
| Data: received `SystemSpec` JSON, uploaded/exchanged specs, anything from a network, user or agent | Untrusted | existing Rust parser/validator (`gogamen-spec`, `run-spec`/`compile-artifact`) | **Never** |

## Authoring code is arbitrary JVM code

A `.gogamen.kts` file is compiled and executed in the host JVM. It can read/write files,
open sockets, spawn processes, use reflection, loop forever or exhaust memory, with the
permissions of the process that evaluates it. There is no sandbox, no `SecurityManager`
(deprecated for removal), no classloader isolation, no time or memory limit. Treat a script
like source code you build and run: review it, keep it under version control, and evaluate
it only in a process whose privileges you accept.

Rules:
1. Never pass received JSON/specs/text to `GogamenScriptHost`. Data crosses the boundary as
   JSON only, and is validated by the existing parser.
2. Do not accept `.kts` files from untrusted sources (uploads, network, other users, LLM
   output that has not been reviewed).
3. The scripting layer is optional; the main artifact has no dependency on it.

## `ImportLint` is hygiene, not security

`ImportLint` (enabled by default in `GogamenScriptHost`) textually flags `java.io`, `java.nio`,
`java.net`, `javax.net`, `java.lang.reflect/invoke`, `kotlin.reflect`, `Runtime`,
`ProcessBuilder`/`Process`. Purpose: catch accidental use and give a line/column diagnostic.
It is **not** a security boundary. Known cases it does NOT cover (pinned in `ImportLintTest`):

- Kotlin stdlib wrappers with no banned token, e.g. `kotlin.io.path.Path("/etc/hostname").toFile().readText()`
  (demonstrated end to end by the test `KNOWN BYPASS demonstrated end to end`).
- String concatenation plus `Class.forName("java." + "lang." + "Runtime")`.
- Aliases, helper libraries on the classpath, `@file:DependsOn`-style dependencies, `eval`-like
  indirection, and any other obfuscation. Also false positives are possible (text inside strings/comments).

Do not rely on the lint to run hostile scripts. If hostile scripts ever become a requirement,
the answer is OS/container-level isolation (separate process, no network, resource limits),
not a stronger lint.
