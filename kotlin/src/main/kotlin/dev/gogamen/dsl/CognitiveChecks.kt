package dev.gogamen.dsl

/**
 * Stable diagnostic codes of the cognitive-experiment DSL (GOG-159).
 *
 * Every rejection carries a code, a JSON-like path into the experiment and a message.
 */
object DslCodes {
    const val BEHAVIOR_RULE = "E_BEHAVIOR_RULE"
    const val OBSERVABLE_LEAK = "E_OBSERVABLE_LEAK"
    const val PRIVILEGED_HIDDEN = "E_PRIVILEGED_HIDDEN"
    const val LEARNING_NO_ALGORITHM = "E_LEARNING_NO_ALGORITHM"
    const val MISSING_BUDGET = "E_MISSING_BUDGET"
    const val MISSING_SEEDS = "E_MISSING_SEEDS"
    const val SEEDS_OVERLAP = "E_SEEDS_OVERLAP"
    const val UNIT_MISMATCH = "E_UNIT_MISMATCH"
    const val TYPE_MISMATCH = "E_TYPE_MISMATCH"
    const val UNKNOWN_VARIABLE = "E_UNKNOWN_VARIABLE"
    const val UNKNOWN_REFERENCE = "E_UNKNOWN_REFERENCE"
    const val HYPOTHESIS_NO_REFUTATION = "E_HYPOTHESIS_NO_REFUTATION"
    const val UNKNOWN_METRIC = "E_UNKNOWN_METRIC"
    const val MATRIX_SIZE_MISMATCH = "E_MATRIX_SIZE_MISMATCH"
    const val BAD_VARY = "E_BAD_VARY"
    const val BAD_PHENOMENON = "E_BAD_PHENOMENON"
    const val MISSING_WORLD = "E_MISSING_WORLD"
    const val MISSING_AGENT = "E_MISSING_AGENT"
    const val MISSING_SYSTEM = "E_MISSING_SYSTEM"
    const val DUPLICATE_ID = "E_DUPLICATE_ID"
    const val MISSING_FIELD = "E_MISSING_FIELD"
}

data class DslDiagnostic(val code: String, val path: String, val message: String)

/** Thrown when the DSL refuses to build an experiment; never produced at runtime by the Rust side. */
class DslRejection(val diagnostic: DslDiagnostic) :
    IllegalArgumentException("${diagnostic.code} at ${diagnostic.path}: ${diagnostic.message}")

internal fun reject(code: String, path: String, message: String): Nothing =
    throw DslRejection(DslDiagnostic(code, path, message))

/** Static type of a declared variable or sub-expression. */
internal sealed interface VType {
    data class Num(val unit: ExprUnit) : VType
    data object Flag : VType
}

internal fun VType.describe(): String = when (this) {
    is VType.Num -> "scalar(${unit.wireName})"
    VType.Flag -> "bool"
}

internal fun combineMul(a: ExprUnit, b: ExprUnit): ExprUnit? = when {
    a == ExprUnit.DIMENSIONLESS -> b
    b == ExprUnit.DIMENSIONLESS -> a
    a == ExprUnit.RATIO -> b
    b == ExprUnit.RATIO -> a
    (a == ExprUnit.METERS_PER_SECOND && b == ExprUnit.SECONDS) ||
        (a == ExprUnit.SECONDS && b == ExprUnit.METERS_PER_SECOND) -> ExprUnit.METERS
    (a == ExprUnit.POWER && b == ExprUnit.SECONDS) ||
        (a == ExprUnit.SECONDS && b == ExprUnit.POWER) -> ExprUnit.ENERGY
    else -> null
}

internal fun combineDiv(a: ExprUnit, b: ExprUnit): ExprUnit? = when {
    b == ExprUnit.DIMENSIONLESS || b == ExprUnit.RATIO -> a
    a == ExprUnit.METERS && b == ExprUnit.SECONDS -> ExprUnit.METERS_PER_SECOND
    a == ExprUnit.METERS && b == ExprUnit.METERS_PER_SECOND -> ExprUnit.SECONDS
    a == ExprUnit.ENERGY && b == ExprUnit.SECONDS -> ExprUnit.POWER
    a == ExprUnit.ENERGY && b == ExprUnit.POWER -> ExprUnit.SECONDS
    a == b -> ExprUnit.DIMENSIONLESS
    else -> null
}

/**
 * Type and unit checker for [ExprAst] over a declared variable environment.
 * Mirrors `ExprUnit::combine_mul/combine_div`; paths append `.lhs`, `.rhs`, `.expr`.
 */
internal object ExprChecker {
    fun typeOf(n: ExprAst, env: Map<String, VType>, path: String): VType = when (n) {
        is ExprAst.ConstF64 -> VType.Num(n.unit)
        is ExprAst.ConstBool -> VType.Flag
        is ExprAst.Var -> env[n.name]
            ?: reject(DslCodes.UNKNOWN_VARIABLE, path, "variable '${n.name}' is not declared")
        is ExprAst.Not -> {
            flag(typeOf(n.expr, env, "$path.expr"), "$path.expr")
            VType.Flag
        }
        is ExprAst.Add -> additive(n.lhs, n.rhs, env, path, "add")
        is ExprAst.Sub -> additive(n.lhs, n.rhs, env, path, "subtract")
        is ExprAst.Mul -> product(n.lhs, n.rhs, env, path, "multiply", ::combineMul)
        is ExprAst.Div -> product(n.lhs, n.rhs, env, path, "divide", ::combineDiv)
        is ExprAst.Lt -> compare(n.lhs, n.rhs, env, path)
        is ExprAst.Le -> compare(n.lhs, n.rhs, env, path)
        is ExprAst.Gt -> compare(n.lhs, n.rhs, env, path)
        is ExprAst.Ge -> compare(n.lhs, n.rhs, env, path)
        is ExprAst.Eq -> equality(n.lhs, n.rhs, env, path)
        is ExprAst.Ne -> equality(n.lhs, n.rhs, env, path)
        is ExprAst.And -> logical(n.lhs, n.rhs, env, path)
        is ExprAst.Or -> logical(n.lhs, n.rhs, env, path)
    }

    fun expectBool(n: ExprAst, env: Map<String, VType>, path: String) {
        flag(typeOf(n, env, path), path)
    }

    fun expectScalar(n: ExprAst, env: Map<String, VType>, path: String): ExprUnit =
        num(typeOf(n, env, path), path).unit

    /** Every `Var` occurrence with its path, in traversal order. */
    fun variables(n: ExprAst, path: String, out: MutableList<Pair<String, String>>) {
        when (n) {
            is ExprAst.Var -> out += n.name to path
            is ExprAst.ConstF64, is ExprAst.ConstBool -> Unit
            is ExprAst.Not -> variables(n.expr, "$path.expr", out)
            is ExprAst.Add -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Sub -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Mul -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Div -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Lt -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Le -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Gt -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Ge -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Eq -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Ne -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.And -> binary(n.lhs, n.rhs, path, out)
            is ExprAst.Or -> binary(n.lhs, n.rhs, path, out)
        }
    }

    private fun binary(l: ExprAst, r: ExprAst, path: String, out: MutableList<Pair<String, String>>) {
        variables(l, "$path.lhs", out)
        variables(r, "$path.rhs", out)
    }

    private fun num(t: VType, path: String): VType.Num = t as? VType.Num
        ?: reject(DslCodes.TYPE_MISMATCH, path, "expected a scalar but found ${t.describe()}")

    private fun flag(t: VType, path: String) {
        if (t !is VType.Flag) {
            reject(DslCodes.TYPE_MISMATCH, path, "expected bool but found ${t.describe()}")
        }
    }

    private fun sameUnit(l: ExprAst, r: ExprAst, env: Map<String, VType>, path: String, verb: String): ExprUnit {
        val a = num(typeOf(l, env, "$path.lhs"), "$path.lhs").unit
        val b = num(typeOf(r, env, "$path.rhs"), "$path.rhs").unit
        if (a != b) {
            reject(DslCodes.UNIT_MISMATCH, path, "cannot $verb ${a.wireName} and ${b.wireName}")
        }
        return a
    }

    private fun additive(l: ExprAst, r: ExprAst, env: Map<String, VType>, path: String, verb: String): VType =
        VType.Num(sameUnit(l, r, env, path, verb))

    private fun compare(l: ExprAst, r: ExprAst, env: Map<String, VType>, path: String): VType {
        sameUnit(l, r, env, path, "compare")
        return VType.Flag
    }

    private fun product(
        l: ExprAst,
        r: ExprAst,
        env: Map<String, VType>,
        path: String,
        verb: String,
        combine: (ExprUnit, ExprUnit) -> ExprUnit?,
    ): VType {
        val a = num(typeOf(l, env, "$path.lhs"), "$path.lhs").unit
        val b = num(typeOf(r, env, "$path.rhs"), "$path.rhs").unit
        val unit = combine(a, b)
            ?: reject(DslCodes.UNIT_MISMATCH, path, "cannot $verb ${a.wireName} and ${b.wireName}")
        return VType.Num(unit)
    }

    private fun equality(l: ExprAst, r: ExprAst, env: Map<String, VType>, path: String): VType {
        val a = typeOf(l, env, "$path.lhs")
        val b = typeOf(r, env, "$path.rhs")
        if (a is VType.Num && b is VType.Num && a.unit != b.unit) {
            reject(DslCodes.UNIT_MISMATCH, path, "cannot compare ${a.unit.wireName} and ${b.unit.wireName}")
        }
        if (a::class != b::class) {
            reject(DslCodes.TYPE_MISMATCH, path, "cannot compare ${a.describe()} with ${b.describe()}")
        }
        return VType.Flag
    }

    private fun logical(l: ExprAst, r: ExprAst, env: Map<String, VType>, path: String): VType {
        flag(typeOf(l, env, "$path.lhs"), "$path.lhs")
        flag(typeOf(r, env, "$path.rhs"), "$path.rhs")
        return VType.Flag
    }
}
