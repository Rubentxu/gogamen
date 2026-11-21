package dev.gogamen.dsl

/**
 * Scope marker for the declarative Gogamen `SystemSpec` Kotlin DSL.
 *
 * Prevents accidental receiver leakage across nested DSL blocks (`systemSpec`, `node`,
 * `connections`, `config`). Kotlin executes purely as a build-time/offline spec
 * generator producing `gogamen.system_spec.v0` JSON without runtime callbacks.
 */
@DslMarker
@Target(AnnotationTarget.CLASS, AnnotationTarget.TYPE)
annotation class GogamenDsl
