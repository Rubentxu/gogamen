package dev.gogamen.dsl

import java.util.SortedMap

/**
 * Deterministic canonical JSON encoder matching `gogamen-spec/src/codec.rs` and `expr.rs`.
 */
internal object CanonicalJson {
    fun encodeSystemSpec(spec: SystemSpec): String {
        val sb = StringBuilder(256)
        writeSystemSpec(sb, spec)
        return sb.toString()
    }

    fun encodeExpr(expr: ExprAst): String {
        val sb = StringBuilder(128)
        writeExpr(sb, expr, depth = 1)
        return sb.toString()
    }

    fun encodeEpistemicQuestion(question: EpistemicQuestion): String {
        val sb = StringBuilder(128)
        writeEpistemicQuestion(sb, question)
        return sb.toString()
    }

    fun encodeAbstentionPolicy(policy: AbstentionPolicy): String {
        val sb = StringBuilder(128)
        sb.append("{\"abstain_on_ambiguity\":")
        sb.append(if (policy.abstainOnAmbiguity) "true" else "false")
        sb.append(",\"abstain_on_timeout\":")
        sb.append(if (policy.abstainOnTimeout) "true" else "false")
        sb.append(",\"ambiguity_margin\":")
        sb.append(normalizeDouble(policy.ambiguityMargin))
        sb.append(",\"fallback_action\":")
        writeJsonString(sb, policy.fallbackAction)
        sb.append(",\"min_confidence\":")
        sb.append(normalizeDouble(policy.minConfidence))
        sb.append('}')
        return sb.toString()
    }

    fun encodeAskAction(ask: AskActionSpec): String {
        val sb = StringBuilder(192)
        sb.append("{\"comm_cost\":")
        sb.append(normalizeDouble(ask.commCost))
        sb.append(",\"question\":")
        writeEpistemicQuestion(sb, ask.question)
        sb.append(",\"recipient\":")
        writeJsonString(sb, ask.recipient)
        sb.append(",\"response_probability\":")
        sb.append(normalizeDouble(ask.responseProbability))
        sb.append(",\"scope\":")
        writeJsonString(sb, ask.scope)
        sb.append(",\"step_cost\":")
        sb.append(normalizeDouble(ask.stepCost))
        sb.append(",\"timeout_ticks\":").append(ask.timeoutTicks)
        sb.append('}')
        return sb.toString()
    }

    fun encodeInspectAction(inspect: InspectActionSpec): String {
        val sb = StringBuilder(96)
        sb.append("{\"inspect_cost\":")
        sb.append(normalizeDouble(inspect.inspectCost))
        sb.append(",\"sensor_id\":")
        writeJsonString(sb, inspect.sensorId)
        if (inspect.targetEntity != null) {
            sb.append(",\"target_entity\":")
            writeJsonString(sb, inspect.targetEntity)
        }
        sb.append('}')
        return sb.toString()
    }

    fun encodeProvenance(prov: ProvenanceSpec): String {
        val sb = StringBuilder(96)
        sb.append("{\"epistemic_status\":")
        writeJsonString(sb, prov.epistemicStatus)
        sb.append(",\"evidence_ref\":").append(prov.evidenceRef)
        sb.append(",\"sensor_id\":").append(prov.sensorId)
        sb.append(",\"source_id\":")
        writeJsonString(sb, prov.sourceId)
        sb.append('}')
        return sb.toString()
    }

    fun encodeGuardPolicy(guard: GuardPolicySpec): String {
        val sb = StringBuilder(128)
        sb.append("{\"allowed_actions\":[")
        guard.allowedActions.forEachIndexed { idx, action ->
            if (idx > 0) sb.append(',')
            writeJsonString(sb, action)
        }
        sb.append("],\"guard_policy\":")
        writeJsonString(sb, guard.guardPolicy)
        sb.append(",\"human_messages_cannot_mutate_permissions\":")
        sb.append(if (guard.humanMessagesCannotMutatePermissions) "true" else "false")
        sb.append(",\"immutable_permissions\":")
        sb.append(if (guard.immutablePermissions) "true" else "false")
        sb.append('}')
        return sb.toString()
    }

    private fun normalizeDouble(value: Double): String = canonicalF64(value)

    /**
     * Canonical f64 text, identical to Rust `write_canonical_f64` (`format!("{}")` + `.0` if integral):
     * shortest round-trip digits, never exponent notation, `-0.0` normalized to `0.0`.
     * (GOG-152: `Double.toString` uses `1.0E7` / `1.0E-4`, which diverged from Rust.)
     */
    internal fun canonicalF64(value: Double): String {
        require(value.isFinite()) { "canonical JSON f64 must be finite, got $value" }
        if (value == 0.0) return "0.0"
        val plain = java.math.BigDecimal(value.toString()).stripTrailingZeros().toPlainString()
        return if ('.' in plain) plain else "$plain.0"
    }

    /** Rust `BTreeMap<String,_>` orders by UTF-8 bytes == Unicode code point order; Java's natural order is UTF-16. */
    private val codePointOrder = Comparator<String> { a, b ->
        var i = 0
        var j = 0
        while (i < a.length && j < b.length) {
            val ca = a.codePointAt(i)
            val cb = b.codePointAt(j)
            if (ca != cb) return@Comparator ca.compareTo(cb)
            i += Character.charCount(ca)
            j += Character.charCount(cb)
        }
        (a.length - i).compareTo(b.length - j)
    }

    private fun writeEpistemicQuestion(out: StringBuilder, q: EpistemicQuestion) {
        out.append('{')
        when (q) {
            is EpistemicQuestion.Choice -> {
                if (q.deadlineTicks != null) {
                    out.append("\"deadline_ticks\":").append(q.deadlineTicks).append(',')
                }
                out.append("\"kind\":\"choice\",\"model_id\":")
                writeJsonString(out, q.modelId)
                out.append(",\"options\":[")
                q.options.forEachIndexed { idx, opt ->
                    if (idx > 0) out.append(',')
                    writeJsonString(out, opt)
                }
                out.append("],\"prompt\":")
                writeJsonString(out, q.prompt)
                out.append(",\"query_id\":")
                writeJsonString(out, q.queryId)
            }
            is EpistemicQuestion.Noul -> {
                if (q.deadlineTicks != null) {
                    out.append("\"deadline_ticks\":").append(q.deadlineTicks).append(',')
                }
                out.append("\"hypothesis_id\":")
                writeJsonString(out, q.hypothesisId)
                out.append(",\"kind\":\"noul\",\"model_id\":")
                writeJsonString(out, q.modelId)
                out.append(",\"prompt\":")
                writeJsonString(out, q.prompt)
                out.append(",\"query_id\":")
                writeJsonString(out, q.queryId)
            }
            is EpistemicQuestion.Score -> {
                if (q.deadlineTicks != null) {
                    out.append("\"deadline_ticks\":").append(q.deadlineTicks).append(',')
                }
                out.append("\"kind\":\"score\",\"max_score\":")
                out.append(normalizeDouble(q.maxScore))
                out.append(",\"min_score\":")
                out.append(normalizeDouble(q.minScore))
                out.append(",\"model_id\":")
                writeJsonString(out, q.modelId)
                out.append(",\"options\":[")
                q.options.forEachIndexed { idx, opt ->
                    if (idx > 0) out.append(',')
                    writeJsonString(out, opt)
                }
                out.append("],\"prompt\":")
                writeJsonString(out, q.prompt)
                out.append(",\"query_id\":")
                writeJsonString(out, q.queryId)
            }
        }
        out.append('}')
    }

    internal fun writeExpr(out: StringBuilder, node: ExprAst, depth: Int) {
        require(depth <= EXPR_AST_MAX_DEPTH) {
            "ExprAst depth $depth exceeds maximum $EXPR_AST_MAX_DEPTH"
        }
        when (node) {
            is ExprAst.ConstF64 -> {
                out.append("{\"op\":\"const_f64\",\"unit\":")
                writeJsonString(out, node.unit.wireName)
                out.append(",\"value\":")
                out.append(canonicalF64(node.value))
                out.append('}')
            }
            is ExprAst.ConstBool -> {
                out.append("{\"op\":\"const_bool\",\"value\":")
                out.append(if (node.value) "true" else "false")
                out.append('}')
            }
            is ExprAst.Var -> {
                out.append("{\"op\":\"var\",\"name\":")
                writeJsonString(out, node.name)
                out.append('}')
            }
            is ExprAst.Not -> {
                out.append("{\"op\":\"not\",\"expr\":")
                writeExpr(out, node.expr, depth + 1)
                out.append('}')
            }
            is ExprAst.Add -> writeBinaryExpr(out, "add", node.lhs, node.rhs, depth)
            is ExprAst.Sub -> writeBinaryExpr(out, "sub", node.lhs, node.rhs, depth)
            is ExprAst.Mul -> writeBinaryExpr(out, "mul", node.lhs, node.rhs, depth)
            is ExprAst.Div -> writeBinaryExpr(out, "div", node.lhs, node.rhs, depth)
            is ExprAst.Lt -> writeBinaryExpr(out, "lt", node.lhs, node.rhs, depth)
            is ExprAst.Le -> writeBinaryExpr(out, "le", node.lhs, node.rhs, depth)
            is ExprAst.Gt -> writeBinaryExpr(out, "gt", node.lhs, node.rhs, depth)
            is ExprAst.Ge -> writeBinaryExpr(out, "ge", node.lhs, node.rhs, depth)
            is ExprAst.Eq -> writeBinaryExpr(out, "eq", node.lhs, node.rhs, depth)
            is ExprAst.Ne -> writeBinaryExpr(out, "ne", node.lhs, node.rhs, depth)
            is ExprAst.And -> writeBinaryExpr(out, "and", node.lhs, node.rhs, depth)
            is ExprAst.Or -> writeBinaryExpr(out, "or", node.lhs, node.rhs, depth)
        }
    }

    private fun writeBinaryExpr(
        out: StringBuilder,
        op: String,
        lhs: ExprAst,
        rhs: ExprAst,
        depth: Int,
    ) {
        out.append("{\"op\":")
        writeJsonString(out, op)
        out.append(",\"lhs\":")
        writeExpr(out, lhs, depth + 1)
        out.append(",\"rhs\":")
        writeExpr(out, rhs, depth + 1)
        out.append('}')
    }

    internal fun writeSystemSpec(out: StringBuilder, spec: SystemSpec) {
        out.append('{')
        out.append("\"config\":")
        writeConfigMap(out, spec.config)
        out.append(",\"connections\":[")
        spec.connections.forEachIndexed { idx, conn ->
            if (idx > 0) out.append(',')
            writeConnectionSpec(out, conn)
        }
        out.append("],\"nodes\":[")
        spec.nodes.forEachIndexed { idx, node ->
            if (idx > 0) out.append(',')
            writeNodeSpec(out, node)
        }
        out.append("],\"schema_version\":")
        writeJsonString(out, spec.schemaVersion)
        out.append(",\"spec_id\":")
        writeJsonString(out, spec.specId)
        out.append('}')
    }

    private fun writeNodeSpec(out: StringBuilder, node: SystemNodeSpec) {
        out.append('{')
        out.append("\"config\":")
        writeConfigMap(out, node.config)
        out.append(",\"id\":")
        writeJsonString(out, node.id)
        out.append(",\"kind\":")
        writeJsonString(out, node.kind)
        out.append(",\"ports\":[")
        node.ports.forEachIndexed { idx, port ->
            if (idx > 0) out.append(',')
            writePortSpec(out, port)
        }
        out.append("]}")
    }

    private fun writePortSpec(out: StringBuilder, port: PortSpec) {
        out.append('{')
        out.append("\"direction\":")
        writeJsonString(out, port.direction.wireName)
        out.append(",\"id\":")
        writeJsonString(out, port.id)
        out.append(",\"port_type\":")
        writePortType(out, port.portType)
        out.append('}')
    }

    internal fun writePortType(out: StringBuilder, portType: PortType) {
        when (portType) {
            PortType.Bool -> writeJsonString(out, "bool")
            PortType.I64 -> writeJsonString(out, "i64")
            PortType.U64 -> writeJsonString(out, "u64")
            PortType.F64 -> writeJsonString(out, "f64")
            PortType.StringType -> writeJsonString(out, "string")
            PortType.UnitType -> writeJsonString(out, "unit")
            is PortType.Custom -> {
                out.append("{\"custom\":")
                writeJsonString(out, portType.name)
                out.append('}')
            }
        }
    }

    private fun writeConnectionSpec(out: StringBuilder, conn: SystemConnectionSpec) {
        out.append('{')
        out.append("\"delay_ticks\":").append(conn.delayTicks)
        out.append(",\"from_node\":")
        writeJsonString(out, conn.fromNode)
        out.append(",\"from_port\":")
        writeJsonString(out, conn.fromPort)
        out.append(",\"to_node\":")
        writeJsonString(out, conn.toNode)
        out.append(",\"to_port\":")
        writeJsonString(out, conn.toPort)
        out.append('}')
    }

    internal fun writeConfigMap(out: StringBuilder, config: SortedMap<String, SpecConfigValue>) {
        out.append('{')
        var first = true
        for ((key, value) in config.entries.sortedWith { x, y -> codePointOrder.compare(x.key, y.key) }) {
            if (!first) out.append(',')
            first = false
            writeJsonString(out, key)
            out.append(':')
            when (value) {
                is SpecConfigValue.Bool -> out.append(if (value.value) "true" else "false")
                is SpecConfigValue.I64 -> out.append(value.value)
                is SpecConfigValue.F64 -> out.append(canonicalF64(value.value))
                is SpecConfigValue.Str -> writeJsonString(out, value.value)
            }
        }
        out.append('}')
    }

    internal fun writeJsonString(out: StringBuilder, value: String) {
        out.append('"')
        for (ch in value) {
            when (ch) {
                '"' -> out.append("\\\"")
                '\\' -> out.append("\\\\")
                '\b' -> out.append("\\b")
                '\u000C' -> out.append("\\f")
                '\n' -> out.append("\\n")
                '\r' -> out.append("\\r")
                '\t' -> out.append("\\t")
                else -> {
                    if (ch.code < 0x20) {
                        out.append("\\u")
                        out.append(ch.code.toString(16).padStart(4, '0'))
                    } else {
                        out.append(ch)
                    }
                }
            }
        }
        out.append('"')
    }
}
