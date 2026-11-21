package dev.gogamen.dsl

const val EXPR_AST_MAX_DEPTH: Int = 64

/**
 * Physical/dimensional units matching `gogamen_spec::expr::ExprUnit`.
 */
enum class ExprUnit(val wireName: String) {
    DIMENSIONLESS("dimensionless"),
    ENERGY("energy"),
    SECONDS("seconds"),
    METERS("meters"),
    RATIO("ratio"),
    METERS_PER_SECOND("meters_per_second"),
    POWER("power"),
}

/**
 * Phantom type hierarchy for compile-time expression type and unit safety.
 */
sealed interface ExprKind

sealed interface UnitTag

sealed interface Dimensionless : UnitTag
sealed interface Energy : UnitTag
sealed interface Seconds : UnitTag
sealed interface Meters : UnitTag
sealed interface Ratio : UnitTag
sealed interface MetersPerSecond : UnitTag
sealed interface Power : UnitTag

sealed interface Scalar<U : UnitTag> : ExprKind
sealed interface BoolKind : ExprKind

/**
 * Compile-time unit witness mapping a [UnitTag] type parameter to its wire [ExprUnit].
 */
sealed class UnitOf<U : UnitTag>(val wireUnit: ExprUnit)

object Units {
    data object Dimensionless : UnitOf<dev.gogamen.dsl.Dimensionless>(ExprUnit.DIMENSIONLESS)
    data object Energy : UnitOf<dev.gogamen.dsl.Energy>(ExprUnit.ENERGY)
    data object Seconds : UnitOf<dev.gogamen.dsl.Seconds>(ExprUnit.SECONDS)
    data object Meters : UnitOf<dev.gogamen.dsl.Meters>(ExprUnit.METERS)
    data object Ratio : UnitOf<dev.gogamen.dsl.Ratio>(ExprUnit.RATIO)
    data object MetersPerSecond : UnitOf<dev.gogamen.dsl.MetersPerSecond>(ExprUnit.METERS_PER_SECOND)
    data object Power : UnitOf<dev.gogamen.dsl.Power>(ExprUnit.POWER)
}

/**
 * Declarative symbolic AST matching `gogamen_spec::expr::ExprAst`.
 */
sealed interface ExprAst {
    data class ConstF64(
        val value: Double,
        val unit: ExprUnit = ExprUnit.DIMENSIONLESS,
    ) : ExprAst {
        init {
            require(value.isFinite()) { "ExprAst.ConstF64 must be finite, got $value" }
        }

        val normalizedValue: Double
            get() = if (value == 0.0) 0.0 else value
    }

    data class ConstBool(val value: Boolean) : ExprAst

    data class Var(val name: String) : ExprAst {
        init {
            require(name.isNotBlank()) { "ExprAst.Var name must not be blank" }
        }
    }

    data class Not(val expr: ExprAst) : ExprAst
    data class Add(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Sub(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Mul(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Div(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Lt(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Le(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Gt(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Ge(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Eq(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Ne(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class And(val lhs: ExprAst, val rhs: ExprAst) : ExprAst
    data class Or(val lhs: ExprAst, val rhs: ExprAst) : ExprAst

    fun toCanonicalJson(): String = CanonicalJson.encodeExpr(this)
}

/**
 * Strongly-typed declarative expression wrapper.
 *
 * The type parameter [T] is intentionally invariant so incompatible units or
 * Scalar/Bool expressions cannot unify to a common supertype.
 */
data class Expr<T : ExprKind>(val ast: ExprAst) {
    fun toCanonicalJson(): String = ast.toCanonicalJson()
}

typealias ScalarExpr<U> = Expr<Scalar<U>>
typealias BoolExpr = Expr<BoolKind>

fun <U : UnitTag> scalarVar(name: String): Expr<Scalar<U>> = Expr(ExprAst.Var(name))

fun boolVar(name: String): Expr<BoolKind> = Expr(ExprAst.Var(name))

fun <T : ExprKind> varRef(name: String): Expr<T> = Expr(ExprAst.Var(name))

fun const(value: Double): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.ConstF64(value, ExprUnit.DIMENSIONLESS))

fun <U : UnitTag> const(value: Double, unit: UnitOf<U>): Expr<Scalar<U>> =
    Expr(ExprAst.ConstF64(value, unit.wireUnit))

fun const(value: Boolean): Expr<BoolKind> =
    Expr(ExprAst.ConstBool(value))

val Double.dimensionless: Expr<Scalar<Dimensionless>>
    get() = const(this, Units.Dimensionless)

val Double.energy: Expr<Scalar<Energy>>
    get() = const(this, Units.Energy)

val Double.seconds: Expr<Scalar<Seconds>>
    get() = const(this, Units.Seconds)

val Double.meters: Expr<Scalar<Meters>>
    get() = const(this, Units.Meters)

val Double.ratio: Expr<Scalar<Ratio>>
    get() = const(this, Units.Ratio)

val Double.metersPerSecond: Expr<Scalar<MetersPerSecond>>
    get() = const(this, Units.MetersPerSecond)

val Double.power: Expr<Scalar<Power>>
    get() = const(this, Units.Power)

// --- Additive operators (same unit required) ---

operator fun <U : UnitTag> Expr<Scalar<U>>.plus(rhs: Expr<Scalar<U>>): Expr<Scalar<U>> =
    Expr(ExprAst.Add(ast, rhs.ast))

operator fun <U : UnitTag> Expr<Scalar<U>>.minus(rhs: Expr<Scalar<U>>): Expr<Scalar<U>> =
    Expr(ExprAst.Sub(ast, rhs.ast))

// --- Comparison operators (same type/unit required) ---

infix fun <U : UnitTag> Expr<Scalar<U>>.lt(rhs: Expr<Scalar<U>>): Expr<BoolKind> =
    Expr(ExprAst.Lt(ast, rhs.ast))

infix fun <U : UnitTag> Expr<Scalar<U>>.le(rhs: Expr<Scalar<U>>): Expr<BoolKind> =
    Expr(ExprAst.Le(ast, rhs.ast))

infix fun <U : UnitTag> Expr<Scalar<U>>.gt(rhs: Expr<Scalar<U>>): Expr<BoolKind> =
    Expr(ExprAst.Gt(ast, rhs.ast))

infix fun <U : UnitTag> Expr<Scalar<U>>.ge(rhs: Expr<Scalar<U>>): Expr<BoolKind> =
    Expr(ExprAst.Ge(ast, rhs.ast))

infix fun <T : ExprKind> Expr<T>.eq(rhs: Expr<T>): Expr<BoolKind> =
    Expr(ExprAst.Eq(ast, rhs.ast))

infix fun <T : ExprKind> Expr<T>.ne(rhs: Expr<T>): Expr<BoolKind> =
    Expr(ExprAst.Ne(ast, rhs.ast))

// --- Boolean operators ---

infix fun Expr<BoolKind>.and(rhs: Expr<BoolKind>): Expr<BoolKind> =
    Expr(ExprAst.And(ast, rhs.ast))

infix fun Expr<BoolKind>.or(rhs: Expr<BoolKind>): Expr<BoolKind> =
    Expr(ExprAst.Or(ast, rhs.ast))

operator fun Expr<BoolKind>.not(): Expr<BoolKind> =
    Expr(ExprAst.Not(ast))

// --- Multiplication (matching ExprUnit::combine_mul) ---

@JvmName("mulDimDim")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimEnergy")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Energy>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimSeconds")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimMeters")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Meters>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimRatio")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Ratio>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimMps")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<MetersPerSecond>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulDimPower")
operator fun Expr<Scalar<Dimensionless>>.times(rhs: Expr<Scalar<Power>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulEnergyDim")
operator fun Expr<Scalar<Energy>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulSecondsDim")
operator fun Expr<Scalar<Seconds>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulMetersDim")
operator fun Expr<Scalar<Meters>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioDim")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Ratio>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulMpsDim")
operator fun Expr<Scalar<MetersPerSecond>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulPowerDim")
operator fun Expr<Scalar<Power>>.times(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioEnergy")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Energy>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioSeconds")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioMeters")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Meters>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioRatio")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Ratio>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioMps")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<MetersPerSecond>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulRatioPower")
operator fun Expr<Scalar<Ratio>>.times(rhs: Expr<Scalar<Power>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulEnergyRatio")
operator fun Expr<Scalar<Energy>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulSecondsRatio")
operator fun Expr<Scalar<Seconds>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulMetersRatio")
operator fun Expr<Scalar<Meters>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulMpsRatio")
operator fun Expr<Scalar<MetersPerSecond>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulPowerRatio")
operator fun Expr<Scalar<Power>>.times(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulMpsSeconds")
operator fun Expr<Scalar<MetersPerSecond>>.times(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulSecondsMps")
operator fun Expr<Scalar<Seconds>>.times(rhs: Expr<Scalar<MetersPerSecond>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulPowerSeconds")
operator fun Expr<Scalar<Power>>.times(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

@JvmName("mulSecondsPower")
operator fun Expr<Scalar<Seconds>>.times(rhs: Expr<Scalar<Power>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Mul(ast, rhs.ast))

// --- Division (matching ExprUnit::combine_div) ---

@JvmName("divDimDim")
operator fun Expr<Scalar<Dimensionless>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divEnergyDim")
operator fun Expr<Scalar<Energy>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divSecondsDim")
operator fun Expr<Scalar<Seconds>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMetersDim")
operator fun Expr<Scalar<Meters>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divRatioDim")
operator fun Expr<Scalar<Ratio>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Ratio>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMpsDim")
operator fun Expr<Scalar<MetersPerSecond>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divPowerDim")
operator fun Expr<Scalar<Power>>.div(rhs: Expr<Scalar<Dimensionless>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divDimRatio")
operator fun Expr<Scalar<Dimensionless>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divEnergyRatio")
operator fun Expr<Scalar<Energy>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Energy>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divSecondsRatio")
operator fun Expr<Scalar<Seconds>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMetersRatio")
operator fun Expr<Scalar<Meters>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Meters>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divRatioRatio")
operator fun Expr<Scalar<Ratio>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Ratio>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMpsRatio")
operator fun Expr<Scalar<MetersPerSecond>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divPowerRatio")
operator fun Expr<Scalar<Power>>.div(rhs: Expr<Scalar<Ratio>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMetersSeconds")
operator fun Expr<Scalar<Meters>>.div(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<MetersPerSecond>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMetersMps")
operator fun Expr<Scalar<Meters>>.div(rhs: Expr<Scalar<MetersPerSecond>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divEnergySeconds")
operator fun Expr<Scalar<Energy>>.div(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Power>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divEnergyPower")
operator fun Expr<Scalar<Energy>>.div(rhs: Expr<Scalar<Power>>): Expr<Scalar<Seconds>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divEnergyEnergy")
operator fun Expr<Scalar<Energy>>.div(rhs: Expr<Scalar<Energy>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divSecondsSeconds")
operator fun Expr<Scalar<Seconds>>.div(rhs: Expr<Scalar<Seconds>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMetersMeters")
operator fun Expr<Scalar<Meters>>.div(rhs: Expr<Scalar<Meters>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divMpsMps")
operator fun Expr<Scalar<MetersPerSecond>>.div(rhs: Expr<Scalar<MetersPerSecond>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))

@JvmName("divPowerPower")
operator fun Expr<Scalar<Power>>.div(rhs: Expr<Scalar<Power>>): Expr<Scalar<Dimensionless>> =
    Expr(ExprAst.Div(ast, rhs.ast))
