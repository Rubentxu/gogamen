# ADR 0001: keep the Kotlin library as the primary authoring path; scripting is an optional add-on

Status: accepted (spike GOG-83). Date: 2026-11-21.

## Context
The Kotlin DSL (`dev.gogamen.dsl`) emits canonical `gogamen.system_spec.v0` JSON. Question: add
`*.gogamen.kts` scripts (kotlin-scripting) as an authoring front end?

## What was built
Optional source set `scripting` (+ `scriptingTest`, task `scriptingTest`, jar classifier `scripting`):
`GogamenScript` template (`defaultImports dev.gogamen.dsl.*, kotlin.math.*`), `GogamenScriptHost`
(diagnostics with line/column), `ImportLint` (hygiene only). `language_world.gogamen.kts` evaluates to
JSON byte-identical to `Fixtures.canonicalLanguageWorld()` and to
`crates/gogamen-spec/fixtures/language_world_canonical_v0.json`.

## Measurements (Temurin 21, Kotlin 2.4.10, this machine, single run of `gradle test`)
| Metric | Library | Script |
|---|---|---|
| Time, first evaluation in JVM (cold) | ~12 ms | ~840-905 ms |
| Time, subsequent evaluations (warm JIT, host reused, no compile cache) | ~0.5-0.6 ms | ~415-680 ms (script is recompiled every call) |
| Main artifact `gogamen-dsl-0.1.0.jar` | 183,579 B | unchanged (no scripting dependency) |
| `gogamen-dsl-0.1.0-scripting.jar` (our code) | n/a | 19,963 B |
| Extra runtime dependencies (`printScriptingClasspathSizes`) | 0 | 11 jars, 67,015,045 B (~64 MiB) |
| of which `kotlin-compiler-embeddable` | | 60,352,373 B |
| Source added | | 150 lines main + tests/docs |

Extra jars: scripting-common/jvm/jvm-host/compiler-embeddable/compiler-impl-embeddable, script-runtime,
compiler-embeddable, build-tools-api, kotlin-reflect 1.6.10, daemon-embeddable, coroutines 1.8.0.

## Decision
1. Keep the Kotlin library (type-checked in the author's normal Kotlin project) as the supported
   authoring path. It is ~3 orders of magnitude cheaper at startup and adds no dependencies.
2. Keep scripting only as the opt-in `scripting` source set. It works and is exact, but costs ~64 MiB of
   dependencies (a full Kotlin compiler at runtime), ~1 s cold start, and brings an arbitrary-code
   execution surface (see `scripting-threat-model.md`) that the library path does not have.
3. Do not extend the spike into a language/VM. Revisit if there is a concrete need for authoring
   without a Gradle project (e.g. a CLI that takes a `.kts` file); then consider script compilation
   caching and process isolation first.

## Consequences
- No change to the Rust core/CLI; JSON remains the interchange and the only accepted form for untrusted data.
- Scripts are trusted code. The import lint must not be described as a sandbox.
