package dev.gogamen.dsl

import java.nio.file.Files
import java.nio.file.Path
import java.util.TreeMap
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * GOG-152: reads the SAME golden corpus as the Rust test `gog_152_canonical_json_corpus`
 * (`spec/golden/canonical_json/cases.json`) and checks that the Kotlin serializer emits the exact
 * canonical bytes. Norm: the Rust codec (`codec.rs`/`json.rs`) is the reference.
 * No Rust runtime is involved; the corpus is only a data file.
 */
class Gog152CanonicalJsonCorpusTest {
    private fun repoRoot(): Path {
        val cwd = Path.of("").toAbsolutePath().normalize()
        return if (cwd.fileName?.toString() == "kotlin") cwd.parent else cwd
    }

    /** Minimal JSON reader (objects, arrays, strings only: that is all the corpus uses). */
    private class MiniJson(private val s: String) {
        private var i = 0

        fun parse(): Any {
            val v = value()
            ws()
            check(i == s.length) { "trailing data at $i" }
            return v
        }

        private fun ws() {
            while (i < s.length && s[i].isWhitespace()) i++
        }

        private fun value(): Any {
            ws()
            return when (s[i]) {
                '{' -> {
                    i++
                    val m = LinkedHashMap<String, Any>()
                    ws()
                    if (s[i] == '}') { i++; return m }
                    while (true) {
                        ws()
                        val k = string()
                        ws(); check(s[i++] == ':')
                        m[k] = value()
                        ws()
                        when (s[i++]) { ',' -> continue; '}' -> return m; else -> error("bad object at $i") }
                    }
                    @Suppress("UNREACHABLE_CODE") m
                }
                '[' -> {
                    i++
                    val l = ArrayList<Any>()
                    ws()
                    if (s[i] == ']') { i++; return l }
                    while (true) {
                        l.add(value())
                        ws()
                        when (s[i++]) { ',' -> continue; ']' -> return l; else -> error("bad array at $i") }
                    }
                    @Suppress("UNREACHABLE_CODE") l
                }
                '"' -> string()
                else -> error("unsupported JSON value at $i")
            }
        }

        private fun string(): String {
            check(s[i++] == '"')
            val sb = StringBuilder()
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> when (val e = s[i++]) {
                        '"' -> sb.append('"'); '\\' -> sb.append('\\'); '/' -> sb.append('/')
                        'b' -> sb.append('\b'); 'f' -> sb.append('\u000C'); 'n' -> sb.append('\n')
                        'r' -> sb.append('\r'); 't' -> sb.append('\t')
                        'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                        else -> error("bad escape \\$e")
                    }
                    else -> sb.append(c)
                }
            }
        }
    }

    private fun configValue(type: String, lit: String): SpecConfigValue = when (type) {
        "bool" -> SpecConfigValue.of(lit == "true")
        "i64" -> SpecConfigValue.of(lit.toLong())
        "f64" -> SpecConfigValue.of(lit.toDouble())
        "str" -> SpecConfigValue.of(lit)
        else -> error("unknown config type $type")
    }

    @Suppress("UNCHECKED_CAST")
    private fun configMap(entries: Any?): java.util.SortedMap<String, SpecConfigValue> {
        val m = TreeMap<String, SpecConfigValue>()
        for (e in entries as List<List<String>>) m[e[0]] = configValue(e[1], e[2])
        return m
    }

    private fun portType(t: String): PortType = when {
        t.startsWith("custom:") -> PortType.Custom(t.removePrefix("custom:"))
        t == "bool" -> PortType.Bool
        t == "f64" -> PortType.F64
        else -> error("unknown port type $t")
    }

    @Suppress("UNCHECKED_CAST")
    private fun emit(c: Map<String, Any>): String = when (val kind = c["kind"]) {
        "system_spec" -> {
            val nodes = ((c["nodes"] as List<Map<String, Any>>?) ?: emptyList()).map { n ->
                SystemNodeSpec(
                    id = n["id"] as String,
                    kind = n["kind"] as String,
                    ports = (n["ports"] as List<List<String>>).map { p ->
                        PortSpec(
                            p[0],
                            if (p[1] == "input") PortDirection.INPUT else PortDirection.OUTPUT,
                            portType(p[2]),
                        )
                    },
                    config = configMap(n["config"]),
                )
            }
            CanonicalJson.encodeSystemSpec(
                SystemSpec(specId = c["spec_id"] as String, nodes = nodes, config = configMap(c["config"])),
            )
        }
        "expr_const_f64" -> CanonicalJson.encodeExpr(ExprAst.ConstF64((c["value"] as String).toDouble()))
        "expr_nested_add" -> CanonicalJson.encodeExpr(
            ExprAst.Add(ExprAst.ConstF64((c["value"] as String).toDouble()), ExprAst.Var("xé")),
        )
        else -> error("unknown case kind $kind")
    }

    @Test
    fun `gog152 kotlin emits exact golden bytes for every corpus case`() {
        val path = repoRoot().resolve("spec/golden/canonical_json/cases.json")
        @Suppress("UNCHECKED_CAST")
        val root = MiniJson(Files.readString(path)).parse() as Map<String, Any>
        assertEquals("gogamen.golden.canonical_json.v1", root["schema"])
        @Suppress("UNCHECKED_CAST")
        val cases = root["cases"] as List<Map<String, Any>>
        assertTrue(cases.size >= 40, "corpus unexpectedly small")
        val failures = cases.mapNotNull { c ->
            val actual = emit(c)
            if (actual == c["expected"]) null else "case ${c["name"]}:\n  expected ${c["expected"]}\n  actual   $actual"
        }
        assertTrue(failures.isEmpty(), failures.joinToString("\n"))
    }
}
