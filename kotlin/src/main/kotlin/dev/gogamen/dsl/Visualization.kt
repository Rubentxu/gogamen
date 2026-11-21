package dev.gogamen.dsl

import java.util.SortedMap
import java.util.TreeMap

/**
 * Declarative `VisualizationSpec` (`gogamen.visualization_spec.v0`, GOG-102): optional, portable section
 * next to a `SystemSpec`. Mirrors `gogamen-projection::viz_spec`; catalog validation (projections,
 * layers, parameter descriptors, units) lives in Rust. No JNI, no runtime callbacks.
 */
const val VISUALIZATION_SPEC_SCHEMA_VERSION: String = "gogamen.visualization_spec.v0"

enum class ScaleUnits(val wireName: String) {
    METERS("meters"),
    CENTIMETERS("centimeters"),
    MILLIMETERS("millimeters"),
    KILOMETERS("kilometers"),
    GRID_CELLS("grid_cells"),
    CUSTOM("custom"),
}

enum class InspectorLod(val wireName: String) {
    MINIMAL("minimal"),
    STANDARD("standard"),
    FULL("full"),
}

sealed interface ParamValue {
    data class Number(val value: Double, val unit: ExprUnit? = null) : ParamValue
    data class Integer(val value: Long) : ParamValue
    data class Bool(val value: Boolean) : ParamValue
    data class Text(val value: String) : ParamValue
}

data class InspectorField(
    val name: String,
    val label: String,
    val minLod: InspectorLod,
    val staleAfter: Long? = null,
)

data class LayerBinding(
    val layer: String,
    val visible: Boolean,
    val sourceNode: String?,
    val params: SortedMap<String, ParamValue>,
)

data class ViewSpec(
    val id: String,
    val projection: String,
    val units: ScaleUnits,
    val layers: List<LayerBinding>,
    val inspector: List<InspectorField>?,
)

data class VisualizationSpec(
    val schemaVersion: String,
    val systemSpecId: String,
    val views: List<ViewSpec>,
) {
    fun toCanonicalJson(): String = VisualizationJson.encode(this)
}

@GogamenDsl
class LayerBuilder(private val layer: String) {
    var visible: Boolean = true
    var sourceNode: String? = null
    private val params: SortedMap<String, ParamValue> = TreeMap()

    fun number(name: String, value: Double, unit: ExprUnit? = null) {
        params[name] = ParamValue.Number(value, unit)
    }

    fun integer(name: String, value: Long) {
        params[name] = ParamValue.Integer(value)
    }

    fun bool(name: String, value: Boolean) {
        params[name] = ParamValue.Bool(value)
    }

    fun text(name: String, value: String) {
        params[name] = ParamValue.Text(value)
    }

    internal fun build() = LayerBinding(layer, visible, sourceNode, TreeMap(params))
}

@GogamenDsl
class ViewBuilder(
    private val id: String,
    private val projection: String,
    private val units: ScaleUnits,
) {
    private val layers = mutableListOf<LayerBinding>()
    private var inspector: MutableList<InspectorField>? = null

    fun layer(layer: String, block: LayerBuilder.() -> Unit = {}) {
        layers += LayerBuilder(layer).apply(block).build()
    }

    fun inspector(block: InspectorBuilder.() -> Unit) {
        inspector = InspectorBuilder().apply(block).build().toMutableList()
    }

    internal fun build() = ViewSpec(id, projection, units, layers.toList(), inspector?.toList())
}

@GogamenDsl
class InspectorBuilder {
    private val fields = mutableListOf<InspectorField>()

    fun field(name: String, label: String, minLod: InspectorLod, staleAfter: Long? = null) {
        fields += InspectorField(name, label, minLod, staleAfter)
    }

    internal fun build(): List<InspectorField> = fields.toList()
}

@GogamenDsl
class VisualizationBuilder(private val systemSpecId: String) {
    private val views = mutableListOf<ViewSpec>()

    fun view(id: String, projection: String, units: ScaleUnits, block: ViewBuilder.() -> Unit) {
        views += ViewBuilder(id, projection, units).apply(block).build()
    }

    internal fun build() = VisualizationSpec(VISUALIZATION_SPEC_SCHEMA_VERSION, systemSpecId, views.toList())
}

fun visualizationSpec(systemSpecId: String, block: VisualizationBuilder.() -> Unit): VisualizationSpec =
    VisualizationBuilder(systemSpecId).apply(block).build()

/** Canonical JSON identical to `VisualizationSpec::to_canonical_json` (sorted keys). */
internal object VisualizationJson {
    fun encode(spec: VisualizationSpec): String {
        val sb = StringBuilder(512)
        sb.append("{\"schema_version\":")
        str(sb, spec.schemaVersion)
        sb.append(",\"system_spec_id\":")
        str(sb, spec.systemSpecId)
        sb.append(",\"views\":[")
        spec.views.forEachIndexed { i, v ->
            if (i > 0) sb.append(',')
            view(sb, v)
        }
        sb.append("]}")
        return sb.toString()
    }

    private fun view(sb: StringBuilder, v: ViewSpec) {
        sb.append("{\"id\":")
        str(sb, v.id)
        if (v.inspector != null) {
            sb.append(",\"inspector\":{\"fields\":[")
            v.inspector.forEachIndexed { i, f ->
                if (i > 0) sb.append(',')
                sb.append("{\"label\":")
                str(sb, f.label)
                sb.append(",\"min_lod\":")
                str(sb, f.minLod.wireName)
                sb.append(",\"name\":")
                str(sb, f.name)
                if (f.staleAfter != null) sb.append(",\"stale_after\":").append(f.staleAfter)
                sb.append('}')
            }
            sb.append("]}")
        }
        sb.append(",\"layers\":[")
        v.layers.forEachIndexed { i, l ->
            if (i > 0) sb.append(',')
            sb.append("{\"layer\":")
            str(sb, l.layer)
            sb.append(",\"params\":{")
            var first = true
            for ((name, value) in l.params) {
                if (!first) sb.append(',')
                first = false
                str(sb, name)
                sb.append(':')
                param(sb, value)
            }
            sb.append('}')
            if (l.sourceNode != null) {
                sb.append(",\"source_node\":")
                str(sb, l.sourceNode)
            }
            sb.append(",\"visible\":").append(if (l.visible) "true" else "false")
            sb.append('}')
        }
        sb.append("],\"projection\":")
        str(sb, v.projection)
        sb.append(",\"units\":")
        str(sb, v.units.wireName)
        sb.append('}')
    }

    private fun param(sb: StringBuilder, p: ParamValue) {
        when (p) {
            is ParamValue.Number -> {
                require(p.value.isFinite()) { "non-finite number" }
                val text = (if (p.value == 0.0) 0.0 else p.value).toString()
                if (p.unit == null) {
                    sb.append(text)
                } else {
                    sb.append("{\"unit\":")
                    str(sb, p.unit.wireName)
                    sb.append(",\"value\":").append(text).append('}')
                }
            }
            is ParamValue.Integer -> sb.append(p.value)
            is ParamValue.Bool -> sb.append(if (p.value) "true" else "false")
            is ParamValue.Text -> str(sb, p.value)
        }
    }

    private fun str(sb: StringBuilder, value: String) {
        sb.append('"')
        for (ch in value) {
            when (ch) {
                '"' -> sb.append("\\\"")
                '\\' -> sb.append("\\\\")
                '\b' -> sb.append("\\b")
                '\u000C' -> sb.append("\\f")
                '\n' -> sb.append("\\n")
                '\r' -> sb.append("\\r")
                '\t' -> sb.append("\\t")
                else -> if (ch.code < 0x20) {
                    sb.append("\\u").append(ch.code.toString(16).padStart(4, '0'))
                } else {
                    sb.append(ch)
                }
            }
        }
        sb.append('"')
    }
}

/** Embeds the optional section into a `SystemSpec` canonical JSON (`visualization` sorts last). */
fun SystemSpec.toCanonicalJsonWith(visualization: VisualizationSpec?): String {
    val base = toCanonicalJson()
    if (visualization == null) return base
    return base.removeSuffix("}") + ",\"visualization\":" + visualization.toCanonicalJson() + "}"
}

object VisualizationFixtures {
    fun canonical(): VisualizationSpec = visualizationSpec("two_node_canonical_v0") {
        view("main", "scene2d", ScaleUnits.METERS) {
            layer("environment") { number("opacity", 1.0) }
            layer("sensors") {
                sourceNode = "sensor"
                number("ray_length", 2.5, ExprUnit.METERS)
            }
            inspector {
                field("position", "Position", InspectorLod.MINIMAL)
                field("energy", "Energy", InspectorLod.FULL, staleAfter = 10)
            }
        }
        view("decisions", "decision_overlay", ScaleUnits.GRID_CELLS) {
            layer("imagination") {
                visible = false
                integer("horizon_ticks", 8)
            }
            layer("labels") { bool("show_legend", true) }
        }
    }
}
