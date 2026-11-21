package dev.gogamen.dsl

/**
 * Typed epistemic question specification (`Choice`, `Noul`, `Score`) matching `gogamen-spec`.
 */
sealed interface EpistemicQuestion {
    val queryId: String
    val prompt: String
    val modelId: String
    val deadlineTicks: Long?

    fun toCanonicalJson(): String = CanonicalJson.encodeEpistemicQuestion(this)

    data class Choice(
        override val queryId: String,
        override val prompt: String,
        val options: List<String>,
        override val modelId: String,
        override val deadlineTicks: Long? = null,
    ) : EpistemicQuestion {
        init {
            require(queryId.isNotBlank()) { "ChoiceQuestion.queryId must not be blank" }
            require(prompt.isNotBlank()) { "ChoiceQuestion.prompt must not be blank" }
            require(modelId.isNotBlank()) { "ChoiceQuestion.modelId must not be blank" }
            require(options.isNotEmpty()) { "ChoiceQuestion.options must not be empty" }
            require(options.all { it.isNotBlank() }) { "ChoiceQuestion.options must not contain blank entries" }
            require(options.distinct().size == options.size) { "ChoiceQuestion.options must not contain duplicates" }
            require(deadlineTicks == null || deadlineTicks > 0L) {
                "ChoiceQuestion.deadlineTicks must be > 0, got $deadlineTicks"
            }
        }
    }

    data class Noul(
        override val queryId: String,
        override val prompt: String,
        val hypothesisId: String,
        override val modelId: String,
        override val deadlineTicks: Long? = null,
    ) : EpistemicQuestion {
        init {
            require(queryId.isNotBlank()) { "NoulQuestion.queryId must not be blank" }
            require(prompt.isNotBlank()) { "NoulQuestion.prompt must not be blank" }
            require(hypothesisId.isNotBlank()) { "NoulQuestion.hypothesisId must not be blank" }
            require(modelId.isNotBlank()) { "NoulQuestion.modelId must not be blank" }
            require(deadlineTicks == null || deadlineTicks > 0L) {
                "NoulQuestion.deadlineTicks must be > 0, got $deadlineTicks"
            }
        }
    }

    data class Score(
        override val queryId: String,
        override val prompt: String,
        val options: List<String>,
        val minScore: Double = 0.0,
        val maxScore: Double = 1.0,
        override val modelId: String,
        override val deadlineTicks: Long? = null,
    ) : EpistemicQuestion {
        init {
            require(queryId.isNotBlank()) { "ScoreQuestion.queryId must not be blank" }
            require(prompt.isNotBlank()) { "ScoreQuestion.prompt must not be blank" }
            require(modelId.isNotBlank()) { "ScoreQuestion.modelId must not be blank" }
            require(options.isNotEmpty()) { "ScoreQuestion.options must not be empty" }
            require(options.all { it.isNotBlank() }) { "ScoreQuestion.options must not contain blank entries" }
            require(options.distinct().size == options.size) { "ScoreQuestion.options must not contain duplicates" }
            require(minScore.isFinite() && maxScore.isFinite() && minScore < maxScore) {
                "ScoreQuestion minScore ($minScore) and maxScore ($maxScore) must be finite with minScore < maxScore"
            }
            require(deadlineTicks == null || deadlineTicks > 0L) {
                "ScoreQuestion.deadlineTicks must be > 0, got $deadlineTicks"
            }
        }
    }
}

/**
 * Explicit abstention policy when confidence is below threshold, ambiguous, or timed out.
 */
data class AbstentionPolicy(
    val minConfidence: Double,
    val fallbackAction: String,
    val abstainOnTimeout: Boolean = true,
    val abstainOnAmbiguity: Boolean = true,
    val ambiguityMargin: Double = 0.05,
) {
    init {
        require(minConfidence.isFinite() && minConfidence in 0.0..1.0) {
            "AbstentionPolicy.minConfidence ($minConfidence) must be finite and within [0.0, 1.0]"
        }
        require(fallbackAction.isNotBlank()) { "AbstentionPolicy.fallbackAction must not be blank" }
        require(ambiguityMargin.isFinite() && ambiguityMargin in 0.0..1.0) {
            "AbstentionPolicy.ambiguityMargin ($ambiguityMargin) must be finite and within [0.0, 1.0]"
        }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeAbstentionPolicy(this)
}

/**
 * Declarative specification for epistemic action `Ask` (no Kotlin interpretation at runtime).
 */
data class AskActionSpec(
    val recipient: String,
    val scope: String,
    val question: EpistemicQuestion,
    val commCost: Double,
    val stepCost: Double = 0.0,
    val responseProbability: Double,
    val timeoutTicks: Long,
) {
    init {
        require(recipient.isNotBlank()) { "AskActionSpec.recipient must not be blank" }
        require(scope.isNotBlank()) { "AskActionSpec.scope must not be blank" }
        require(commCost.isFinite() && commCost >= 0.0) {
            "AskActionSpec.commCost ($commCost) must be finite and >= 0.0"
        }
        require(stepCost.isFinite() && stepCost >= 0.0) {
            "AskActionSpec.stepCost ($stepCost) must be finite and >= 0.0"
        }
        require(responseProbability.isFinite() && responseProbability in 0.0..1.0) {
            "AskActionSpec.responseProbability ($responseProbability) must be finite and within [0.0, 1.0]"
        }
        require(timeoutTicks > 0L) { "AskActionSpec.timeoutTicks ($timeoutTicks) must be > 0" }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeAskAction(this)
}

/**
 * Declarative specification for epistemic action `Inspect` (direct sensor observation).
 */
data class InspectActionSpec(
    val sensorId: String,
    val inspectCost: Double,
    val targetEntity: String? = null,
) {
    init {
        require(sensorId.isNotBlank()) { "InspectActionSpec.sensorId must not be blank" }
        require(inspectCost.isFinite() && inspectCost >= 0.0) {
            "InspectActionSpec.inspectCost ($inspectCost) must be finite and >= 0.0"
        }
        require(targetEntity == null || targetEntity.isNotBlank()) {
            "InspectActionSpec.targetEntity must not be blank when provided"
        }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeInspectAction(this)
}

/**
 * Explicit provenance metadata (`source_id`, `sensor_id`, `evidence_ref`) for observations.
 */
data class ProvenanceSpec(
    val sourceId: String,
    val sensorId: Long,
    val evidenceRef: Long,
    val epistemicStatus: String = "observed",
) {
    init {
        require(sourceId.isNotBlank()) { "ProvenanceSpec.sourceId must not be blank" }
        require(sensorId > 0L) { "ProvenanceSpec.sensorId ($sensorId) must be > 0" }
        require(evidenceRef > 0L) { "ProvenanceSpec.evidenceRef ($evidenceRef) must be > 0" }
        require(epistemicStatus in VALID_STATUSES) {
            "ProvenanceSpec.epistemicStatus '$epistemicStatus' must be one of $VALID_STATUSES"
        }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeProvenance(this)

    companion object {
        private val VALID_STATUSES = setOf("observed", "reported", "inferred", "hypothetical")
    }
}

/**
 * Immutable `ActionGuard` and `PermissionSet` policy ensuring human messages never mutate permissions.
 */
data class GuardPolicySpec(
    val guardPolicy: String = "reject_only",
    val allowedActions: List<String> = emptyList(),
    val immutablePermissions: Boolean = true,
    val humanMessagesCannotMutatePermissions: Boolean = true,
) {
    init {
        require(guardPolicy in setOf("reject_only", "explicit_fallback")) {
            "GuardPolicySpec.guardPolicy '$guardPolicy' must be 'reject_only' or 'explicit_fallback'"
        }
        require(immutablePermissions) {
            "PermissionSet must remain immutable (immutablePermissions=true)"
        }
        require(humanMessagesCannotMutatePermissions) {
            "Human proposals/messages cannot mutate ActionGuard or PermissionSet"
        }
        require(allowedActions.all { it.isNotBlank() }) {
            "GuardPolicySpec.allowedActions must not contain blank entries"
        }
        require(allowedActions.distinct().size == allowedActions.size) {
            "GuardPolicySpec.allowedActions must not contain duplicates"
        }
    }

    fun toCanonicalJson(): String = CanonicalJson.encodeGuardPolicy(this)
}

@GogamenDsl
class ChoiceQuestionBuilder(private val queryId: String) {
    var prompt: String = ""
    var modelId: String = "offline_replay_v1"
    var deadlineTicks: Long? = null
    private val options = mutableListOf<String>()

    fun option(id: String) {
        options += id
    }

    fun options(vararg ids: String) {
        options.addAll(ids)
    }

    internal fun build(): EpistemicQuestion.Choice = EpistemicQuestion.Choice(
        queryId = queryId,
        prompt = prompt,
        options = options.toList(),
        modelId = modelId,
        deadlineTicks = deadlineTicks,
    )
}

@GogamenDsl
class NoulQuestionBuilder(private val queryId: String) {
    var prompt: String = ""
    var hypothesisId: String = ""
    var modelId: String = "system_one_noul_v1"
    var deadlineTicks: Long? = null

    internal fun build(): EpistemicQuestion.Noul = EpistemicQuestion.Noul(
        queryId = queryId,
        prompt = prompt,
        hypothesisId = hypothesisId,
        modelId = modelId,
        deadlineTicks = deadlineTicks,
    )
}

@GogamenDsl
class ScoreQuestionBuilder(private val queryId: String) {
    var prompt: String = ""
    var modelId: String = "system_one_score_v1"
    var minScore: Double = 0.0
    var maxScore: Double = 1.0
    var deadlineTicks: Long? = null
    private val options = mutableListOf<String>()

    fun option(id: String) {
        options += id
    }

    fun options(vararg ids: String) {
        options.addAll(ids)
    }

    internal fun build(): EpistemicQuestion.Score = EpistemicQuestion.Score(
        queryId = queryId,
        prompt = prompt,
        options = options.toList(),
        minScore = minScore,
        maxScore = maxScore,
        modelId = modelId,
        deadlineTicks = deadlineTicks,
    )
}

@GogamenDsl
class AbstentionPolicyBuilder {
    var minConfidence: Double = 0.65
    var fallbackAction: String = "abstain"
    var abstainOnTimeout: Boolean = true
    var abstainOnAmbiguity: Boolean = true
    var ambiguityMargin: Double = 0.05

    internal fun build(): AbstentionPolicy = AbstentionPolicy(
        minConfidence = minConfidence,
        fallbackAction = fallbackAction,
        abstainOnTimeout = abstainOnTimeout,
        abstainOnAmbiguity = abstainOnAmbiguity,
        ambiguityMargin = ambiguityMargin,
    )
}

@GogamenDsl
class AskActionBuilder {
    var recipient: String = "human_operator"
    var scope: String = "language_world"
    var commCost: Double = 0.5
    var stepCost: Double = 0.0
    var responseProbability: Double = 0.85
    var timeoutTicks: Long = 3L
    private var question: EpistemicQuestion? = null

    fun choiceQuestion(queryId: String, block: ChoiceQuestionBuilder.() -> Unit) {
        question = ChoiceQuestionBuilder(queryId).apply(block).build()
    }

    fun noulQuestion(queryId: String, block: NoulQuestionBuilder.() -> Unit) {
        question = NoulQuestionBuilder(queryId).apply(block).build()
    }

    fun scoreQuestion(queryId: String, block: ScoreQuestionBuilder.() -> Unit) {
        question = ScoreQuestionBuilder(queryId).apply(block).build()
    }

    internal fun build(): AskActionSpec {
        val q = requireNotNull(question) {
            "AskAction requires a question (choiceQuestion, noulQuestion, or scoreQuestion)"
        }
        return AskActionSpec(
            recipient = recipient,
            scope = scope,
            question = q,
            commCost = commCost,
            stepCost = stepCost,
            responseProbability = responseProbability,
            timeoutTicks = timeoutTicks,
        )
    }
}

@GogamenDsl
class InspectActionBuilder {
    var sensorId: String = ""
    var inspectCost: Double = 0.25
    var targetEntity: String? = null

    internal fun build(): InspectActionSpec = InspectActionSpec(
        sensorId = sensorId,
        inspectCost = inspectCost,
        targetEntity = targetEntity,
    )
}

@GogamenDsl
class ProvenanceBuilder {
    var sourceId: String = ""
    var sensorId: Long = 1L
    var evidenceRef: Long = 1L
    var epistemicStatus: String = "observed"

    internal fun build(): ProvenanceSpec = ProvenanceSpec(
        sourceId = sourceId,
        sensorId = sensorId,
        evidenceRef = evidenceRef,
        epistemicStatus = epistemicStatus,
    )
}

@GogamenDsl
class GuardPolicyBuilder {
    var guardPolicy: String = "reject_only"
    var immutablePermissions: Boolean = true
    var humanMessagesCannotMutatePermissions: Boolean = true
    private val allowedActions = mutableListOf<String>()

    fun allowAction(action: String) {
        allowedActions += action
    }

    fun allowActions(vararg actions: String) {
        allowedActions.addAll(actions)
    }

    internal fun build(): GuardPolicySpec = GuardPolicySpec(
        guardPolicy = guardPolicy,
        allowedActions = allowedActions.toList(),
        immutablePermissions = immutablePermissions,
        humanMessagesCannotMutatePermissions = humanMessagesCannotMutatePermissions,
    )
}

/**
 * Validates epistemic references (nodes, ports, registered sources, available models, and guard immutability).
 */
object EpistemicValidator {
    val DEFAULT_AVAILABLE_MODELS: Set<String> = setOf(
        "offline_replay_v1",
        "system_one_choice_v1",
        "system_one_noul_v1",
        "system_one_score_v1",
    )

    fun validate(
        spec: SystemSpec,
        availableModels: Set<String> = DEFAULT_AVAILABLE_MODELS,
    ) {
        val nodeById = mutableMapOf<String, SystemNodeSpec>()
        for (node in spec.nodes) {
            val prev = nodeById.put(node.id, node)
            require(prev == null) { "Duplicate node id '${node.id}' in SystemSpec '${spec.specId}'" }
        }

        for (conn in spec.connections) {
            val fromNode = nodeById[conn.fromNode]
                ?: throw IllegalArgumentException(
                    "Incompatible connection in '${spec.specId}': unknown source node '${conn.fromNode}'",
                )
            val toNode = nodeById[conn.toNode]
                ?: throw IllegalArgumentException(
                    "Incompatible connection in '${spec.specId}': unknown target node '${conn.toNode}'",
                )
            val fromPort = fromNode.ports.firstOrNull { it.id == conn.fromPort }
                ?: throw IllegalArgumentException(
                    "Incompatible connection in '${spec.specId}': unknown source port '${conn.fromNode}.${conn.fromPort}'",
                )
            val toPort = toNode.ports.firstOrNull { it.id == conn.toPort }
                ?: throw IllegalArgumentException(
                    "Incompatible connection in '${spec.specId}': unknown target port '${conn.toNode}.${conn.toPort}'",
                )
            require(fromPort.direction == PortDirection.OUTPUT) {
                "Incompatible port direction: source '${conn.fromNode}.${conn.fromPort}' must be OUTPUT"
            }
            require(toPort.direction == PortDirection.INPUT) {
                "Incompatible port direction: target '${conn.toNode}.${conn.toPort}' must be INPUT"
            }
            require(fromPort.portType == toPort.portType) {
                "Incompatible port types: '${conn.fromNode}.${conn.fromPort}' (${fromPort.portType}) != '${conn.toNode}.${conn.toPort}' (${toPort.portType})"
            }
        }

        val registeredSources = spec.nodes
            .filter { it.kind == "language::human_channel" || it.kind == "language::direct_sensor" }
            .mapNotNull { (it.config["source_id"] as? SpecConfigValue.Str)?.value }
            .toSet()

        for (node in spec.nodes) {
            val modelId = (node.config["model_id"] as? SpecConfigValue.Str)?.value
            if (modelId != null) {
                require(modelId in availableModels) {
                    "Unavailable model '$modelId' on node '${node.id}': available models are ${availableModels.sorted()}"
                }
            }

            if (node.kind == "language::grounder") {
                val sourceId = (node.config["source_id"] as? SpecConfigValue.Str)?.value
                    ?: throw IllegalArgumentException(
                        "Node '${node.id}' (${node.kind}) is missing required 'source_id'",
                    )
                require(sourceId in registeredSources) {
                    "Incompatible source reference '$sourceId' on node '${node.id}': registered sources are ${registeredSources.sorted()}"
                }
            }

            val inspectJson = (node.config["inspect_json"] as? SpecConfigValue.Str)?.value
            if (inspectJson != null) {
                val sensorId = extractJsonStringField(inspectJson, "sensor_id")
                if (sensorId != null) {
                    require(sensorId in registeredSources) {
                        "Incompatible sensor reference '$sensorId' in InspectAction on node '${node.id}': registered sources are ${registeredSources.sorted()}"
                    }
                }
            }

            val askJson = (node.config["ask_json"] as? SpecConfigValue.Str)?.value
            if (askJson != null) {
                val askModelId = extractJsonStringField(askJson, "model_id")
                if (askModelId != null) {
                    require(askModelId in availableModels) {
                        "Unavailable model '$askModelId' in AskAction on node '${node.id}': available models are ${availableModels.sorted()}"
                    }
                }
            }

            if (node.kind == "language::guarded_planner") {
                val canMutate = (node.config["human_can_mutate_permissions"] as? SpecConfigValue.Bool)?.value
                require(canMutate != true) {
                    "Security invariant violated on node '${node.id}': human messages cannot mutate permissions"
                }
                val immutable = (node.config["immutable_permissions"] as? SpecConfigValue.Bool)?.value
                require(immutable != false) {
                    "Security invariant violated on node '${node.id}': PermissionSet must remain immutable"
                }
            }
        }
    }

    private fun extractJsonStringField(json: String, field: String): String? {
        val marker = "\"$field\":\""
        val start = json.indexOf(marker)
        if (start < 0) return null
        val valueStart = start + marker.length
        val end = json.indexOf('"', valueStart)
        return if (end > valueStart) json.substring(valueStart, end) else null
    }
}
