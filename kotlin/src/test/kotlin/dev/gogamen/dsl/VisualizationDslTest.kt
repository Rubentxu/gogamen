package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import kotlin.test.Test
import kotlin.test.assertEquals

/** GOG-102 AC #1: Kotlin builder emits the exact golden shared with the Rust tests (no JVM needed on the Rust side). */
class VisualizationDslTest {
    private fun fixture(name: String): String {
        val cwd = Path.of("").toAbsolutePath().normalize()
        val root = if (cwd.fileName?.toString() == "kotlin") cwd.parent else cwd
        return Files.readString(root.resolve("crates/gogamen-projection/fixtures/$name"))
    }

    @Test
    fun `visualization builder matches shared golden`() {
        assertEquals(fixture("visualization_canonical_v0.json"), VisualizationFixtures.canonical().toCanonicalJson())
    }

    @Test
    fun `system spec with optional visualization section matches golden`() {
        assertEquals(
            fixture("system_with_visualization_v0.json"),
            Fixtures.canonicalTwoNode().toCanonicalJsonWith(VisualizationFixtures.canonical()),
        )
        assertEquals(Fixtures.canonicalTwoNode().toCanonicalJson(), Fixtures.canonicalTwoNode().toCanonicalJsonWith(null))
    }
}
