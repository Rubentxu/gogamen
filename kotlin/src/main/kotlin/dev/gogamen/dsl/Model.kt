package dev.gogamen.dsl

import java.util.SortedMap
import java.util.TreeMap

const val SYSTEM_SPEC_SCHEMA_VERSION: String = "gogamen.system_spec.v0"

enum class PortDirection(val wireName: String) {
    INPUT("input"),
    OUTPUT("output"),
}

sealed interface PortType {
    data object Bool : PortType
    data object I64 : PortType
    data object U64 : PortType
    data object F64 : PortType
    data object StringType : PortType
    data object UnitType : PortType
    data class Custom(val name: String) : PortType {
        init {
            require(name.isNotBlank()) { "PortType.Custom name must not be blank" }
        }
    }
}

sealed interface SpecConfigValue {
    data class Bool(val value: Boolean) : SpecConfigValue
    data class I64(val value: Long) : SpecConfigValue
    data class F64(val value: Double) : SpecConfigValue {
        init {
            require(value.isFinite()) { "SpecConfigValue.F64 must be finite, got $value" }
        }

        val normalized: Double
            get() = if (value == 0.0) 0.0 else value
    }
    data class Str(val value: String) : SpecConfigValue

    companion object {
        fun of(value: Boolean): SpecConfigValue = Bool(value)
        fun of(value: Int): SpecConfigValue = I64(value.toLong())
        fun of(value: Long): SpecConfigValue = I64(value)
        fun of(value: Double): SpecConfigValue = F64(if (value == 0.0) 0.0 else value)
        fun of(value: String): SpecConfigValue = Str(value)
    }
}

data class PortSpec(
    val id: String,
    val direction: PortDirection,
    val portType: PortType,
) {
    init {
        require(id.isNotBlank()) { "PortSpec.id must not be blank" }
    }
}

data class SystemNodeSpec(
    val id: String,
    val kind: String,
    val ports: List<PortSpec> = emptyList(),
    val config: SortedMap<String, SpecConfigValue> = TreeMap(),
) {
    init {
        require(id.isNotBlank()) { "SystemNodeSpec.id must not be blank" }
        require(kind.isNotBlank()) { "SystemNodeSpec.kind must not be blank" }
    }
}

data class SystemConnectionSpec(
    val fromNode: String,
    val fromPort: String,
    val toNode: String,
    val toPort: String,
    val delayTicks: Long = 0L,
) {
    init {
        require(fromNode.isNotBlank()) { "SystemConnectionSpec.fromNode must not be blank" }
        require(fromPort.isNotBlank()) { "SystemConnectionSpec.fromPort must not be blank" }
        require(toNode.isNotBlank()) { "SystemConnectionSpec.toNode must not be blank" }
        require(toPort.isNotBlank()) { "SystemConnectionSpec.toPort must not be blank" }
        require(delayTicks >= 0L) { "SystemConnectionSpec.delayTicks must be non-negative, got $delayTicks" }
    }
}

data class SystemSpec(
    val specId: String,
    val nodes: List<SystemNodeSpec> = emptyList(),
    val connections: List<SystemConnectionSpec> = emptyList(),
    val config: SortedMap<String, SpecConfigValue> = TreeMap(),
    val schemaVersion: String = SYSTEM_SPEC_SCHEMA_VERSION,
) {
    init {
        require(schemaVersion == SYSTEM_SPEC_SCHEMA_VERSION) {
            "Unsupported schemaVersion '$schemaVersion'; expected '$SYSTEM_SPEC_SCHEMA_VERSION'"
        }
        require(specId.isNotBlank()) { "SystemSpec.specId must not be blank" }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeSystemSpec(this)
}
