package dev.gogamen.dsl

import java.util.SortedMap
import java.util.TreeMap

@GogamenDsl
class ConfigBuilder {
    private val entries: SortedMap<String, SpecConfigValue> = TreeMap()

    infix fun String.to(value: Boolean) {
        entries[this] = SpecConfigValue.of(value)
    }

    infix fun String.to(value: Int) {
        entries[this] = SpecConfigValue.of(value)
    }

    infix fun String.to(value: Long) {
        entries[this] = SpecConfigValue.of(value)
    }

    infix fun String.to(value: Double) {
        entries[this] = SpecConfigValue.of(value)
    }

    infix fun String.to(value: String) {
        entries[this] = SpecConfigValue.of(value)
    }

    fun put(key: String, value: SpecConfigValue) {
        entries[key] = value
    }

    internal fun build(): SortedMap<String, SpecConfigValue> = TreeMap(entries)
}

@GogamenDsl
class NodeBuilder(
    private val id: String,
    private val kind: String,
) {
    private val ports = mutableListOf<PortSpec>()
    private val config: SortedMap<String, SpecConfigValue> = TreeMap()

    fun input(id: String, portType: PortType) {
        ports += PortSpec(id = id, direction = PortDirection.INPUT, portType = portType)
    }

    fun output(id: String, portType: PortType) {
        ports += PortSpec(id = id, direction = PortDirection.OUTPUT, portType = portType)
    }

    fun port(id: String, direction: PortDirection, portType: PortType) {
        ports += PortSpec(id = id, direction = direction, portType = portType)
    }

    fun config(block: ConfigBuilder.() -> Unit) {
        config.putAll(ConfigBuilder().apply(block).build())
    }

    fun choiceQuestion(queryId: String, block: ChoiceQuestionBuilder.() -> Unit) {
        val q = ChoiceQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun noulQuestion(queryId: String, block: NoulQuestionBuilder.() -> Unit) {
        val q = NoulQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun scoreQuestion(queryId: String, block: ScoreQuestionBuilder.() -> Unit) {
        val q = ScoreQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun abstentionPolicy(block: AbstentionPolicyBuilder.() -> Unit) {
        val policy = AbstentionPolicyBuilder().apply(block).build()
        config["abstention_json"] = SpecConfigValue.of(policy.toCanonicalJson())
    }

    fun askAction(block: AskActionBuilder.() -> Unit) {
        val ask = AskActionBuilder().apply(block).build()
        config["ask_json"] = SpecConfigValue.of(ask.toCanonicalJson())
    }

    fun inspectAction(block: InspectActionBuilder.() -> Unit) {
        val inspect = InspectActionBuilder().apply(block).build()
        config["inspect_json"] = SpecConfigValue.of(inspect.toCanonicalJson())
    }

    fun provenance(block: ProvenanceBuilder.() -> Unit) {
        val prov = ProvenanceBuilder().apply(block).build()
        config["source_id"] = SpecConfigValue.of(prov.sourceId)
        config["sensor_id"] = SpecConfigValue.of(prov.sensorId)
        config["evidence_ref"] = SpecConfigValue.of(prov.evidenceRef)
        config["provenance_json"] = SpecConfigValue.of(prov.toCanonicalJson())
    }

    fun guardPolicy(block: GuardPolicyBuilder.() -> Unit) {
        val guard = GuardPolicyBuilder().apply(block).build()
        config["guard_policy"] = SpecConfigValue.of(guard.guardPolicy)
        config["immutable_permissions"] = SpecConfigValue.of(guard.immutablePermissions)
        config["human_can_mutate_permissions"] = SpecConfigValue.of(!guard.humanMessagesCannotMutatePermissions)
        config["guard_json"] = SpecConfigValue.of(guard.toCanonicalJson())
    }

    internal fun build(): SystemNodeSpec = SystemNodeSpec(
        id = id,
        kind = kind,
        ports = ports.toList(),
        config = TreeMap(config),
    )
}

@GogamenDsl
class ConnectionsBuilder {
    private val connections = mutableListOf<SystemConnectionSpec>()

    fun instant(fromNode: String, fromPort: String, toNode: String, toPort: String) {
        connections += SystemConnectionSpec(
            fromNode = fromNode,
            fromPort = fromPort,
            toNode = toNode,
            toPort = toPort,
            delayTicks = 0L,
        )
    }

    fun delayed(
        fromNode: String,
        fromPort: String,
        toNode: String,
        toPort: String,
        delayTicks: Long,
    ) {
        connections += SystemConnectionSpec(
            fromNode = fromNode,
            fromPort = fromPort,
            toNode = toNode,
            toPort = toPort,
            delayTicks = delayTicks,
        )
    }

    internal fun build(): List<SystemConnectionSpec> = connections.toList()
}

@GogamenDsl
class SystemSpecBuilder(private val specId: String) {
    private val nodes = mutableListOf<SystemNodeSpec>()
    private val connections = mutableListOf<SystemConnectionSpec>()
    private val config: SortedMap<String, SpecConfigValue> = TreeMap()

    fun node(id: String, kind: String, block: NodeBuilder.() -> Unit = {}) {
        nodes += NodeBuilder(id, kind).apply(block).build()
    }

    fun connections(block: ConnectionsBuilder.() -> Unit) {
        connections += ConnectionsBuilder().apply(block).build()
    }

    fun connect(
        fromNode: String,
        fromPort: String,
        toNode: String,
        toPort: String,
        delayTicks: Long = 0L,
    ) {
        connections += SystemConnectionSpec(
            fromNode = fromNode,
            fromPort = fromPort,
            toNode = toNode,
            toPort = toPort,
            delayTicks = delayTicks,
        )
    }

    fun config(block: ConfigBuilder.() -> Unit) {
        config.putAll(ConfigBuilder().apply(block).build())
    }

    fun choiceQuestion(queryId: String, block: ChoiceQuestionBuilder.() -> Unit) {
        val q = ChoiceQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun noulQuestion(queryId: String, block: NoulQuestionBuilder.() -> Unit) {
        val q = NoulQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun scoreQuestion(queryId: String, block: ScoreQuestionBuilder.() -> Unit) {
        val q = ScoreQuestionBuilder(queryId).apply(block).build()
        config["model_id"] = SpecConfigValue.of(q.modelId)
        config["question_json"] = SpecConfigValue.of(q.toCanonicalJson())
    }

    fun abstentionPolicy(block: AbstentionPolicyBuilder.() -> Unit) {
        val policy = AbstentionPolicyBuilder().apply(block).build()
        config["abstention_json"] = SpecConfigValue.of(policy.toCanonicalJson())
    }

    fun askAction(block: AskActionBuilder.() -> Unit) {
        val ask = AskActionBuilder().apply(block).build()
        config["ask_json"] = SpecConfigValue.of(ask.toCanonicalJson())
    }

    fun inspectAction(block: InspectActionBuilder.() -> Unit) {
        val inspect = InspectActionBuilder().apply(block).build()
        config["inspect_json"] = SpecConfigValue.of(inspect.toCanonicalJson())
    }

    fun provenance(block: ProvenanceBuilder.() -> Unit) {
        val prov = ProvenanceBuilder().apply(block).build()
        config["source_id"] = SpecConfigValue.of(prov.sourceId)
        config["sensor_id"] = SpecConfigValue.of(prov.sensorId)
        config["evidence_ref"] = SpecConfigValue.of(prov.evidenceRef)
        config["provenance_json"] = SpecConfigValue.of(prov.toCanonicalJson())
    }

    fun guardPolicy(block: GuardPolicyBuilder.() -> Unit) {
        val guard = GuardPolicyBuilder().apply(block).build()
        config["guard_policy"] = SpecConfigValue.of(guard.guardPolicy)
        config["immutable_permissions"] = SpecConfigValue.of(guard.immutablePermissions)
        config["human_can_mutate_permissions"] = SpecConfigValue.of(!guard.humanMessagesCannotMutatePermissions)
        config["guard_json"] = SpecConfigValue.of(guard.toCanonicalJson())
    }

    internal fun build(): SystemSpec = SystemSpec(
        specId = specId,
        nodes = nodes.toList(),
        connections = connections.toList(),
        config = TreeMap(config),
    )
}

fun systemSpec(specId: String, block: SystemSpecBuilder.() -> Unit): SystemSpec =
    SystemSpecBuilder(specId).apply(block).build()

fun epistemicSystemSpec(
    specId: String,
    availableModels: Set<String> = EpistemicValidator.DEFAULT_AVAILABLE_MODELS,
    block: SystemSpecBuilder.() -> Unit,
): SystemSpec = SystemSpecBuilder(specId).apply(block).build().also {
    EpistemicValidator.validate(it, availableModels)
}
