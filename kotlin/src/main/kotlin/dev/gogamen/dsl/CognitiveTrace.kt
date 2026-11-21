package dev.gogamen.dsl

/** Comparator shared by hypotheses and `rate` phenomena (wire names match `HypothesisComparator`). */
enum class Cmp(val wire: String) {
    GT("gt"), GE("ge"), LT("lt"), LE("le"), EQ("eq");

    fun holds(observed: Double, threshold: Double): Boolean = when (this) {
        GT -> observed > threshold
        GE -> observed >= threshold
        LT -> observed < threshold
        LE -> observed <= threshold
        EQ -> observed == threshold
    }
}

/**
 * A phenomenon is a predicate over the recorded trace, expressed with the restricted [ExprAst].
 * It is evaluated outside the agent: the agent never sees it.
 */
sealed interface PhenomenonSpec {
    val id: String

    /** `expr` holds on every tick. */
    data class Always(override val id: String, val expr: ExprAst) : PhenomenonSpec

    /** `expr` holds on at least one tick (before `withinTicks`, when given). */
    data class Eventually(override val id: String, val expr: ExprAst, val withinTicks: Long?) : PhenomenonSpec

    /** From the first tick where `trigger` holds, `then` holds on every strictly later tick. */
    data class After(override val id: String, val trigger: ExprAst, val then: ExprAst) : PhenomenonSpec

    /** The fraction of ticks where `expr` holds satisfies `cmp threshold`. */
    data class Rate(override val id: String, val expr: ExprAst, val cmp: Cmp, val threshold: Double) : PhenomenonSpec
}

/** One recorded tick: variable name to `Double` or `Boolean`. */
typealias TraceSample = Map<String, Any>

/**
 * Reference evaluator of phenomena over a trace. It is a specification helper for tests and
 * tooling; the runtime evaluator is the Rust side. It never touches an agent.
 */
object TraceChecker {
    fun holds(p: PhenomenonSpec, trace: List<TraceSample>): Boolean = when (p) {
        is PhenomenonSpec.Always -> trace.all { bool(p.expr, it) }
        is PhenomenonSpec.Eventually -> {
            val limit = p.withinTicks?.let { minOf(it, trace.size.toLong()).toInt() } ?: trace.size
            trace.take(limit).any { bool(p.expr, it) }
        }
        is PhenomenonSpec.After -> {
            val start = trace.indexOfFirst { bool(p.trigger, it) }
            start < 0 || trace.drop(start + 1).all { bool(p.then, it) }
        }
        is PhenomenonSpec.Rate -> p.cmp.holds(holdFraction(p.expr, trace), p.threshold)
    }

    /** Fraction of ticks where `expr` holds (0.0 for an empty trace). */
    fun holdFraction(expr: ExprAst, trace: List<TraceSample>): Double =
        if (trace.isEmpty()) 0.0 else trace.count { bool(expr, it) }.toDouble() / trace.size

    private fun bool(n: ExprAst, s: TraceSample): Boolean = eval(n, s) as Boolean

    private fun num(n: ExprAst, s: TraceSample): Double = eval(n, s) as Double

    private fun eval(n: ExprAst, s: TraceSample): Any = when (n) {
        is ExprAst.ConstF64 -> n.value
        is ExprAst.ConstBool -> n.value
        is ExprAst.Var -> s[n.name] ?: throw IllegalArgumentException("trace sample has no variable '${n.name}'")
        is ExprAst.Not -> !bool(n.expr, s)
        is ExprAst.Add -> num(n.lhs, s) + num(n.rhs, s)
        is ExprAst.Sub -> num(n.lhs, s) - num(n.rhs, s)
        is ExprAst.Mul -> num(n.lhs, s) * num(n.rhs, s)
        is ExprAst.Div -> num(n.lhs, s) / num(n.rhs, s)
        is ExprAst.Lt -> num(n.lhs, s) < num(n.rhs, s)
        is ExprAst.Le -> num(n.lhs, s) <= num(n.rhs, s)
        is ExprAst.Gt -> num(n.lhs, s) > num(n.rhs, s)
        is ExprAst.Ge -> num(n.lhs, s) >= num(n.rhs, s)
        is ExprAst.Eq -> eval(n.lhs, s) == eval(n.rhs, s)
        is ExprAst.Ne -> eval(n.lhs, s) != eval(n.rhs, s)
        is ExprAst.And -> bool(n.lhs, s) && bool(n.rhs, s)
        is ExprAst.Or -> bool(n.lhs, s) || bool(n.rhs, s)
    }
}
