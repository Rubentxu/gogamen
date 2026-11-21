package dev.gogamen.dsl

import java.util.TreeMap

/*
 * GOG-159 cognitive-experiment DSL: a frontend for `AdaptiveExperimentSpec` + `CognitiveSpec`.
 *
 * Four layers, and only these exist in the grammar:
 *   world      -> `world { given(...) }`      given by the environment
 *   agent      -> `agent { model/sensor/... }` given by design (no behavior construct)
 *   questions  -> `phenomenon`, `hypothesis`, `vary`
 *   readings   -> `reading`
 *
 * There is no parser and no VM here: builders emit canonical JSON that the Rust parser accepts.
 * Constructs the Rust schema has no field for (pure mode, privileged models, phenomena, vary,
 * readings) travel as plain string/bool entries of the existing `config` maps and are only emitted
 * when used, so a plain experiment stays byte-identical to the native fixture.
 */

enum class Observability { FULL, PARTIAL, HIDDEN }

enum class LearningAlgorithm(val wire: String) { NONE("none"), BAYESIAN("bayesian"), TD("td"), HEBBIAN("hebbian") }

enum class Objective(val wire: String) {
    FMC("fmc"), ENTROPY("entropy"), EMPOWERMENT("empowerment"), INFO_GAIN("info_gain")
}

enum class MemoryKind(val wire: String) { WORKING("working"), EPISODIC("episodic") }

enum class MemoryPermission(val wire: String) { READ("read"), WRITE("write"), APPEND("append") }

enum class Aggregate(val wire: String) { MEAN("mean"), MAX("max"), MIN("min"), LAST("last"), HOLD_FRACTION("hold_fraction") }

/** One cell of the expanded run matrix. */
data class Run(val index: Int, val assignments: Map<String, Double>, val seed: Long, val split: String)

private const val EXPERIMENT_SCHEMA = "gogamen.experiment_spec.v0.1"
private const val AGENT_SCHEMA = "gogamen.adaptive_agent_spec.v0.1"
private const val COGNITIVE_SCHEMA = "gogamen.cognitive_spec.v0"

/** Upper bound of the expanded matrix; larger sweeps are rejected instead of silently truncated. */
const val MAX_MATRIX_RUNS: Int = 10_000

// ---------------------------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------------------------

internal data class WorldDecl(
    val id: String,
    val kind: String,
    val given: Map<String, SpecConfigValue>,
    val observability: Observability,
)

internal data class ModelDecl(
    val id: String,
    val kind: String,
    val config: Map<String, SpecConfigValue>,
    val privileged: Boolean,
)

internal data class SensorDecl(
    val id: String,
    val kind: String,
    val node: String,
    val port: String,
    val type: PortType,
    val observes: List<String>,
    val config: Map<String, SpecConfigValue>,
)

internal data class GoalDecl(val id: String, val maximize: Boolean, val expr: ExprAst)

internal data class ConstraintDecl(val id: String, val hard: Boolean, val expr: ExprAst)

internal data class AgentDecl(
    val id: String,
    val model: ModelDecl?,
    val sensors: List<SensorDecl>,
    val actions: List<String>,
    val goals: List<GoalDecl>,
    val constraints: List<ConstraintDecl>,
    val pure: Boolean,
)

internal data class LevelDecl(val id: String, val sensor: String, val bins: Long)

internal data class CognitiveDecl(
    val objective: Objective,
    val temperature: Double,
    val entropyWeight: Double,
    val horizon: Long,
    val binning: String?,
    val levels: List<LevelDecl>,
    val learningDeclared: Boolean,
    val algorithm: LearningAlgorithm?,
    val learningRate: Double,
    val updateEvery: Long,
    val memoryDeclared: Boolean,
    val memoryKind: MemoryKind,
    val capacity: Long,
    val memoryNode: String,
    val memoryPort: String,
    val permissions: List<MemoryPermission>,
)

internal data class HypothesisDecl(
    val id: String,
    val claim: String,
    val metric: String,
    val support: Pair<Cmp, Double>?,
    val refute: Pair<Cmp, Double>?,
)

internal data class VaryDecl(val param: String, val values: List<Double>)

internal data class ReadingDecl(val name: String, val unit: ExprUnit, val aggregate: Aggregate, val of: String)

// ---------------------------------------------------------------------------------------------
// Builders
// ---------------------------------------------------------------------------------------------

@GogamenDsl
class WorldBuilder {
    private val given = TreeMap<String, SpecConfigValue>()
    private var observability = Observability.FULL

    /** Facts the environment fixes; the agent cannot choose them. */
    fun given(block: ConfigBuilder.() -> Unit) {
        given.putAll(ConfigBuilder().apply(block).build())
    }

    fun observability(value: Observability) {
        observability = value
    }

    internal fun build(id: String, kind: String): WorldDecl = WorldDecl(id, kind, given.toMap(), observability)
}

@GogamenDsl
class ModelBuilder {
    /** A privileged model reads world state directly; it is incompatible with hidden worlds. */
    var privileged: Boolean = false
    private val config = TreeMap<String, SpecConfigValue>()

    fun config(block: ConfigBuilder.() -> Unit) {
        config.putAll(ConfigBuilder().apply(block).build())
    }

    internal fun build(id: String, kind: String): ModelDecl = ModelDecl(id, kind, config.toMap(), privileged)
}

@GogamenDsl
class SensorBuilder {
    private val observes = mutableListOf<String>()
    private val config = TreeMap<String, SpecConfigValue>()

    /** Declared variables this sensor exposes to the agent. */
    fun observes(vararg names: String) {
        observes.addAll(names)
    }

    fun config(block: ConfigBuilder.() -> Unit) {
        config.putAll(ConfigBuilder().apply(block).build())
    }

    internal fun build(id: String, kind: String, node: String, port: String, type: PortType): SensorDecl =
        SensorDecl(id, kind, node, port, type, observes.toList(), config.toMap())
}

@GogamenDsl
class AgentBuilder internal constructor(private val index: Int, private val id: String) {
    private var model: ModelDecl? = null
    private val sensors = mutableListOf<SensorDecl>()
    private var actions: List<String> = emptyList()
    private val goals = mutableListOf<GoalDecl>()
    private val constraints = mutableListOf<ConstraintDecl>()
    private var pure = false

    fun model(id: String, kind: String, block: ModelBuilder.() -> Unit = {}) {
        model = ModelBuilder().apply(block).build(id, kind)
    }

    fun sensor(
        id: String,
        kind: String,
        node: String,
        port: String,
        type: PortType,
        block: SensorBuilder.() -> Unit = {},
    ) {
        sensors += SensorBuilder().apply(block).build(id, kind, node, port, type)
    }

    fun actions(vararg ids: String) {
        actions = ids.toList()
    }

    fun goal(id: String, maximize: Boolean = true, expr: Expr<*>) {
        goals += GoalDecl(id, maximize, expr.ast)
    }

    fun constraint(id: String, hard: Boolean = true, condition: Expr<*>) {
        constraints += ConstraintDecl(id, hard, condition.ast)
    }

    /** Pure agent: goals and constraints may only mention variables its sensors observe. */
    fun pure() {
        pure = true
    }

    // The grammar offers no way to program behavior. These names exist only so that an attempt
    // is rejected with a precise diagnostic instead of a generic "unresolved reference".
    fun rule(vararg any: Any?): Nothing = noBehavior("rule")
    fun policy(vararg any: Any?): Nothing = noBehavior("policy")
    fun act(vararg any: Any?): Nothing = noBehavior("act")
    fun script(vararg any: Any?): Nothing = noBehavior("script")
    fun behavior(block: () -> Unit): Nothing = noBehavior("behavior")
    fun onTick(block: () -> Unit): Nothing = noBehavior("onTick")
    fun onObservation(block: () -> Unit): Nothing = noBehavior("onObservation")

    private fun noBehavior(name: String): Nothing = reject(
        DslCodes.BEHAVIOR_RULE,
        "agents[$index].$name",
        "agent '$id' is given by design (model, sensors, actions, goals); behavior rules are not part of the grammar",
    )

    internal fun build(): AgentDecl = AgentDecl(id, model, sensors.toList(), actions, goals.toList(), constraints.toList(), pure)
}

@GogamenDsl
class EntropicBuilder {
    var objective: Objective = Objective.FMC
    var temperature: Double = 0.5
    var entropyWeight: Double = 0.25
    var horizon: Long = 16
}

@GogamenDsl
class ResolutionBuilder {
    internal val levels = mutableListOf<LevelDecl>()

    fun level(id: String, sensor: String, bins: Long) {
        levels += LevelDecl(id, sensor, bins)
    }

    /** Declares several levels over one sensor, coarse to fine. */
    fun levels(sensor: String, vararg bins: Pair<String, Int>) {
        bins.forEach { (id, n) -> level(id, sensor, n.toLong()) }
    }
}

@GogamenDsl
class LearningBuilder {
    /** Required: pick [LearningAlgorithm.NONE] explicitly to disable learning. */
    var algorithm: LearningAlgorithm? = null
    var rate: Double = 0.1
    var every: Long = 4
}

@GogamenDsl
class MemoryBuilder {
    var kind: MemoryKind = MemoryKind.EPISODIC
    var capacity: Long = 256
    internal var node: String = "memory_store"
    internal var port: String = "memory"
    internal var permissions: List<MemoryPermission> = listOf(MemoryPermission.READ, MemoryPermission.APPEND)

    fun port(node: String, port: String) {
        this.node = node
        this.port = port
    }

    fun permissions(vararg values: MemoryPermission) {
        permissions = values.toList()
    }
}

@GogamenDsl
class CognitiveBuilder {
    private val entropic = EntropicBuilder()
    private var binning: String? = null
    private val levels = mutableListOf<LevelDecl>()
    private var learning: LearningBuilder? = null
    private var memory: MemoryBuilder? = null

    fun entropic(block: EntropicBuilder.() -> Unit) {
        entropic.apply(block)
    }

    fun resolution(binningVersion: String, block: ResolutionBuilder.() -> Unit) {
        binning = binningVersion
        levels += ResolutionBuilder().apply(block).levels
    }

    fun learning(block: LearningBuilder.() -> Unit) {
        learning = LearningBuilder().apply(block)
    }

    fun memory(block: MemoryBuilder.() -> Unit) {
        memory = MemoryBuilder().apply(block)
    }

    internal fun build(): CognitiveDecl {
        val l = learning
        val m = memory ?: MemoryBuilder()
        return CognitiveDecl(
            objective = entropic.objective,
            temperature = entropic.temperature,
            entropyWeight = entropic.entropyWeight,
            horizon = entropic.horizon,
            binning = binning,
            levels = levels.toList(),
            learningDeclared = l != null,
            algorithm = l?.algorithm,
            learningRate = l?.rate ?: 0.1,
            updateEvery = l?.every ?: 4,
            memoryDeclared = memory != null,
            memoryKind = m.kind,
            capacity = m.capacity,
            memoryNode = m.node,
            memoryPort = m.port,
            permissions = m.permissions,
        )
    }
}

@GogamenDsl
class HypothesisBuilder {
    var claim: String = ""
    var metric: String = ""
    private var support: Pair<Cmp, Double>? = null
    private var refute: Pair<Cmp, Double>? = null

    fun supportedIf(cmp: Cmp, threshold: Double) {
        support = cmp to threshold
    }

    fun refutedIf(cmp: Cmp, threshold: Double) {
        refute = cmp to threshold
    }

    internal fun build(id: String): HypothesisDecl = HypothesisDecl(id, claim, metric, support, refute)
}

@GogamenDsl
class PhenomenonBuilder internal constructor(private val id: String, private val index: Int) {
    internal var spec: PhenomenonSpec? = null

    private fun set(value: PhenomenonSpec) {
        if (spec != null) {
            reject(DslCodes.BAD_PHENOMENON, "phenomena[$index]", "phenomenon '$id' declares more than one predicate")
        }
        spec = value
    }

    fun always(expr: Expr<BoolKind>) = set(PhenomenonSpec.Always(id, expr.ast))

    fun eventually(expr: Expr<BoolKind>, within: Long? = null) =
        set(PhenomenonSpec.Eventually(id, expr.ast, within))

    fun after(trigger: Expr<BoolKind>, then: Expr<BoolKind>) =
        set(PhenomenonSpec.After(id, trigger.ast, then.ast))

    fun rate(expr: Expr<BoolKind>, cmp: Cmp, threshold: Double) =
        set(PhenomenonSpec.Rate(id, expr.ast, cmp, threshold))
}

// ---------------------------------------------------------------------------------------------
// Experiment
// ---------------------------------------------------------------------------------------------

/** A validated experiment: canonical JSON for the Rust parser plus the declared questions. */
class CognitiveExperiment internal constructor(
    val experimentId: String,
    private val json: String,
    val phenomena: List<PhenomenonSpec>,
    val matrix: List<Run>,
) {
    /** Canonical `gogamen.experiment_spec.v0.1` JSON. */
    fun toCanonicalJson(): String = json
}

@GogamenDsl
class ExperimentBuilder internal constructor(private val id: String) {
    private var system: SystemSpec? = null
    private var world: WorldDecl? = null
    private val agents = mutableListOf<AgentDecl>()
    private val vars = LinkedHashMap<String, VType>()
    private var cognitive: CognitiveDecl? = null
    private var budget: Triple<Long, Long, Long>? = null
    private var seeds: Pair<List<Long>, List<Long>>? = null
    private val config = TreeMap<String, SpecConfigValue>()
    private val hypotheses = mutableListOf<HypothesisDecl>()
    private val phenomena = mutableListOf<PhenomenonSpec>()
    private val vary = mutableListOf<VaryDecl>()
    private val readings = mutableListOf<ReadingDecl>()
    private var expectedRuns: Int? = null

    fun system(spec: SystemSpec) {
        system = spec
    }

    fun world(id: String, kind: String, block: WorldBuilder.() -> Unit = {}) {
        world = WorldBuilder().apply(block).build(id, kind)
    }

    /** Declares a scalar variable with a unit; the returned expression is typed by that unit. */
    fun <U : UnitTag> scalar(name: String, unit: UnitOf<U>): Expr<Scalar<U>> {
        declare(name, VType.Num(unit.wireUnit))
        return scalarVar(name)
    }

    fun flag(name: String): Expr<BoolKind> {
        declare(name, VType.Flag)
        return boolVar(name)
    }

    private fun declare(name: String, type: VType) {
        if (vars.put(name, type) != null) {
            reject(DslCodes.DUPLICATE_ID, "variables.$name", "variable '$name' is declared twice")
        }
    }

    fun agent(id: String, block: AgentBuilder.() -> Unit) {
        agents += AgentBuilder(agents.size, id).apply(block).build()
    }

    fun cognitive(block: CognitiveBuilder.() -> Unit) {
        cognitive = CognitiveBuilder().apply(block).build()
    }

    fun budget(maxTicks: Long, maxTransitions: Long, deadlineMs: Long) {
        budget = Triple(maxTicks, maxTransitions, deadlineMs)
    }

    fun seeds(dev: List<Number>, holdout: List<Number>) {
        seeds = dev.map { it.toLong() } to holdout.map { it.toLong() }
    }

    fun config(block: ConfigBuilder.() -> Unit) {
        config.putAll(ConfigBuilder().apply(block).build())
    }

    fun hypothesis(id: String, block: HypothesisBuilder.() -> Unit) {
        hypotheses += HypothesisBuilder().apply(block).build(id)
    }

    fun phenomenon(id: String, block: PhenomenonBuilder.() -> Unit) {
        val b = PhenomenonBuilder(id, phenomena.size).apply(block)
        phenomena += b.spec
            ?: reject(DslCodes.BAD_PHENOMENON, "phenomena[${phenomena.size}]", "phenomenon '$id' declares no predicate")
    }

    /** Sweeps one declared parameter; the run matrix is the Cartesian product of all axes and seeds. */
    fun vary(param: String, vararg values: Double) {
        vary += VaryDecl(param, values.toList())
    }

    fun expectRuns(count: Int) {
        expectedRuns = count
    }

    fun reading(name: String, unit: ExprUnit, aggregate: Aggregate, of: String) {
        readings += ReadingDecl(name, unit, aggregate, of)
    }

    // -----------------------------------------------------------------------------------------

    internal fun build(): CognitiveExperiment {
        val sys = system ?: reject(DslCodes.MISSING_SYSTEM, "system", "experiment '$id' declares no system")
        val w = world ?: reject(DslCodes.MISSING_WORLD, "world", "experiment '$id' declares no world")
        if (agents.isEmpty()) reject(DslCodes.MISSING_AGENT, "agents", "experiment '$id' declares no agent")
        val b = budget ?: reject(DslCodes.MISSING_BUDGET, "budget", "experiment '$id' declares no budget")
        val s = seeds
        if (s == null || s.first.isEmpty()) {
            reject(DslCodes.MISSING_SEEDS, "seeds.dev_seeds", "experiment '$id' declares no development seeds")
        }
        s.second.forEachIndexed { i, seed ->
            if (seed in s.first) {
                reject(DslCodes.SEEDS_OVERLAP, "seeds.holdout_seeds[$i]", "seed $seed is both dev and holdout")
            }
        }
        checkUnique(agents.map { it.id }, "agents")
        agents.forEachIndexed { i, a -> checkAgent(i, a, w) }
        cognitive?.let { checkCognitive(it) }
        checkPhenomena()
        checkReadings()
        checkHypotheses()
        val matrix = expandMatrix(w, s)
        expectedRuns?.let {
            if (it != matrix.size) {
                reject(DslCodes.MATRIX_SIZE_MISMATCH, "vary", "declared $it runs but the matrix expands to ${matrix.size}")
            }
        }
        return CognitiveExperiment(id, emit(sys, w, b, s), phenomena.toList(), matrix)
    }

    private fun checkUnique(ids: List<String>, path: String) {
        val seen = HashSet<String>()
        ids.forEachIndexed { i, x ->
            if (!seen.add(x)) reject(DslCodes.DUPLICATE_ID, "$path[$i]", "duplicate id '$x'")
        }
    }

    private fun checkAgent(i: Int, a: AgentDecl, w: WorldDecl) {
        val base = "agents[$i]"
        val model = a.model ?: reject(DslCodes.MISSING_FIELD, "$base.model", "agent '${a.id}' declares no model")
        if (model.privileged && w.observability == Observability.HIDDEN) {
            reject(
                DslCodes.PRIVILEGED_HIDDEN,
                "$base.model.privileged",
                "privileged model '${model.id}' reads world state, which a hidden world forbids",
            )
        }
        a.sensors.forEachIndexed { j, sensor ->
            sensor.observes.forEachIndexed { k, name ->
                if (name !in vars) {
                    reject(DslCodes.UNKNOWN_VARIABLE, "$base.sensors[$j].observes[$k]", "variable '$name' is not declared")
                }
            }
        }
        val observed = a.sensors.flatMap { it.observes }.toSet()
        a.goals.forEachIndexed { j, g ->
            val path = "$base.goals[$j].expr"
            ExprChecker.expectScalar(g.expr, vars, path)
            checkLeak(a, observed, g.expr, path)
        }
        a.constraints.forEachIndexed { j, c ->
            val path = "$base.constraints[$j].condition"
            ExprChecker.expectBool(c.expr, vars, path)
            checkLeak(a, observed, c.expr, path)
        }
    }

    private fun checkLeak(a: AgentDecl, observed: Set<String>, expr: ExprAst, path: String) {
        if (!a.pure) return
        val used = mutableListOf<Pair<String, String>>()
        ExprChecker.variables(expr, path, used)
        for ((name, at) in used) {
            if (name !in observed) {
                reject(
                    DslCodes.OBSERVABLE_LEAK,
                    at,
                    "pure agent '${a.id}' uses '$name', which none of its sensors observe",
                )
            }
        }
    }

    private fun checkCognitive(c: CognitiveDecl) {
        if (!c.learningDeclared || c.algorithm == null) {
            reject(DslCodes.LEARNING_NO_ALGORITHM, "cognitive.learning.rule", "learning requires an explicit algorithm")
        }
        if (c.binning == null || c.levels.isEmpty()) {
            reject(DslCodes.MISSING_FIELD, "cognitive.resolution", "cognitive resolution needs a binning version and a level")
        }
        val sensorIds = agents.flatMap { a -> a.sensors.map { it.id } }.toSet()
        c.levels.forEachIndexed { i, level ->
            if (level.sensor !in sensorIds) {
                reject(
                    DslCodes.UNKNOWN_REFERENCE,
                    "cognitive.resolution.levels[$i].sensor",
                    "sensor '${level.sensor}' is not declared by any agent",
                )
            }
        }
    }

    private fun checkPhenomena() {
        checkUnique(phenomena.map { it.id }, "phenomena")
        phenomena.forEachIndexed { i, p ->
            val base = "phenomena[$i]"
            when (p) {
                is PhenomenonSpec.Always -> ExprChecker.expectBool(p.expr, vars, "$base.expr")
                is PhenomenonSpec.Eventually -> ExprChecker.expectBool(p.expr, vars, "$base.expr")
                is PhenomenonSpec.After -> {
                    ExprChecker.expectBool(p.trigger, vars, "$base.trigger")
                    ExprChecker.expectBool(p.then, vars, "$base.then")
                }
                is PhenomenonSpec.Rate -> {
                    ExprChecker.expectBool(p.expr, vars, "$base.expr")
                    if (!p.threshold.isFinite() || p.threshold < 0.0 || p.threshold > 1.0) {
                        reject(DslCodes.BAD_PHENOMENON, "$base.threshold", "rate threshold must be within [0, 1]")
                    }
                }
            }
        }
    }

    private fun checkReadings() {
        checkUnique(readings.map { it.name }, "readings")
        readings.forEachIndexed { i, r ->
            val base = "readings[$i]"
            if (r.aggregate == Aggregate.HOLD_FRACTION) {
                if (phenomena.none { it.id == r.of }) {
                    reject(DslCodes.UNKNOWN_METRIC, "$base.of", "phenomenon '${r.of}' is not declared")
                }
                if (r.unit != ExprUnit.RATIO) {
                    reject(DslCodes.UNIT_MISMATCH, "$base.unit", "hold_fraction is a ratio, not ${r.unit.wireName}")
                }
            } else {
                val t = vars[r.of] ?: reject(DslCodes.UNKNOWN_VARIABLE, "$base.of", "variable '${r.of}' is not declared")
                if (t != VType.Num(r.unit)) {
                    reject(DslCodes.UNIT_MISMATCH, "$base.unit", "'${r.of}' is ${t.describe()}, reading declares ${r.unit.wireName}")
                }
            }
        }
    }

    private fun checkHypotheses() {
        checkUnique(hypotheses.map { it.id }, "hypotheses")
        hypotheses.forEachIndexed { i, h ->
            val base = "hypotheses[$i]"
            if (h.support == null) {
                reject(DslCodes.MISSING_FIELD, "$base.comparator", "hypothesis '${h.id}' declares no support condition")
            }
            if (h.refute == null) {
                reject(
                    DslCodes.HYPOTHESIS_NO_REFUTATION,
                    "$base.refutation_comparator",
                    "hypothesis '${h.id}' must declare what would refute it",
                )
            }
            if (readings.none { it.name == h.metric }) {
                reject(DslCodes.UNKNOWN_METRIC, "$base.metric", "metric '${h.metric}' is not a declared reading")
            }
        }
    }

    // -----------------------------------------------------------------------------------------
    // Run matrix
    // -----------------------------------------------------------------------------------------

    private fun checkAxis(i: Int, v: VaryDecl, w: WorldDecl) {
        val path = "vary[$i]"
        if (v.values.isEmpty()) reject(DslCodes.BAD_VARY, "$path.values", "vary '${v.param}' has no values")
        val integral = v.param in setOf("entropic.horizon", "learning.update_every_ticks", "memory.capacity")
        val known = v.param in setOf("entropic.temperature", "entropic.entropy_weight", "learning.learning_rate")
        val isWorld = v.param.startsWith("world.")
        if (!integral && !known && !isWorld) {
            reject(DslCodes.BAD_VARY, "$path.param", "'${v.param}' is not a varyable parameter")
        }
        if (isWorld && v.param.removePrefix("world.") !in w.given) {
            reject(DslCodes.BAD_VARY, "$path.param", "world has no given '${v.param.removePrefix("world.")}'")
        }
        if (!isWorld && cognitive == null) {
            reject(DslCodes.BAD_VARY, "$path.param", "'${v.param}' needs a cognitive block")
        }
        v.values.forEachIndexed { j, x ->
            val bad = !x.isFinite() ||
                (integral && (x != Math.rint(x) || x < 1.0)) ||
                (v.param == "entropic.temperature" && x <= 0.0) ||
                (v.param == "entropic.entropy_weight" && x < 0.0) ||
                (v.param == "learning.learning_rate" && x <= 0.0)
            if (bad) reject(DslCodes.BAD_VARY, "$path.values[$j]", "value $x is out of range for '${v.param}'")
        }
    }

    private fun expandMatrix(w: WorldDecl, s: Pair<List<Long>, List<Long>>): List<Run> {
        vary.forEachIndexed { i, v -> checkAxis(i, v, w) }
        val dup = vary.groupBy { it.param }.entries.firstOrNull { it.value.size > 1 }
        if (dup != null) reject(DslCodes.BAD_VARY, "vary", "parameter '${dup.key}' is varied twice")
        var combos = 1L
        for (v in vary) combos *= v.values.size
        val total = combos * (s.first.size + s.second.size)
        if (total > MAX_MATRIX_RUNS) {
            reject(DslCodes.BAD_VARY, "vary", "matrix expands to $total runs (maximum $MAX_MATRIX_RUNS)")
        }
        var cells: List<Map<String, Double>> = listOf(emptyMap())
        for (v in vary) cells = cells.flatMap { c -> v.values.map { x -> c + (v.param to x) } }
        val seedList = s.first.map { it to "dev" } + s.second.map { it to "holdout" }
        val out = ArrayList<Run>(total.toInt())
        for (cell in cells) for ((seed, split) in seedList) out += Run(out.size, cell, seed, split)
        return out
    }

    // -----------------------------------------------------------------------------------------
    // Canonical JSON emission (key order follows the Rust writer)
    // -----------------------------------------------------------------------------------------

    private fun str(o: StringBuilder, v: String) = CanonicalJson.writeJsonString(o, v)

    private fun cfg(o: StringBuilder, map: Map<String, SpecConfigValue>) =
        CanonicalJson.writeConfigMap(o, TreeMap(map))

    private fun <T> list(o: StringBuilder, items: List<T>, each: (T) -> Unit) {
        o.append('[')
        items.forEachIndexed { i, x ->
            if (i > 0) o.append(',')
            each(x)
        }
        o.append(']')
    }

    private fun expr(o: StringBuilder, e: ExprAst) = CanonicalJson.writeExpr(o, e, 1)

    private fun f64(o: StringBuilder, v: Double) {
        o.append(CanonicalJson.canonicalF64(v))
    }

    private fun emit(sys: SystemSpec, w: WorldDecl, b: Triple<Long, Long, Long>, s: Pair<List<Long>, List<Long>>): String {
        val o = StringBuilder(4096)
        val modelId = agents.first().model!!.id
        o.append("{\"agents\":")
        list(o, agents) { emitAgent(o, it, sys, modelId) }
        o.append(",\"budget\":{\"deadline_ms\":").append(b.third)
        o.append(",\"max_ticks\":").append(b.first)
        o.append(",\"max_transitions\":").append(b.second).append('}')
        cognitive?.let {
            o.append(",\"cognitive\":")
            emitCognitive(o, it, modelId)
        }
        o.append(",\"config\":")
        cfg(o, experimentConfig())
        o.append(",\"constraints\":")
        list(o, agents.flatMap { it.constraints }.distinctBy { it.id }) { emitConstraint(o, it) }
        o.append(",\"experiment_id\":")
        str(o, id)
        o.append(",\"goals\":")
        list(o, agents.flatMap { it.goals }.distinctBy { it.id }) { emitGoal(o, it) }
        if (hypotheses.isNotEmpty()) {
            o.append(",\"hypotheses\":")
            list(o, hypotheses.sortedBy { it.id }) { emitHypothesis(o, it) }
        }
        o.append(",\"schema_version\":\"").append(EXPERIMENT_SCHEMA).append("\",\"seeds\":{\"dev_seeds\":")
        list(o, s.first) { o.append(it) }
        o.append(",\"holdout_seeds\":")
        list(o, s.second) { o.append(it) }
        o.append("},\"system\":")
        CanonicalJson.writeSystemSpec(o, sys)
        o.append(",\"world\":{\"config\":")
        cfg(o, worldConfig(w))
        o.append(",\"kind\":")
        str(o, w.kind)
        o.append(",\"world_id\":")
        str(o, w.id)
        o.append("}}")
        return o.toString()
    }

    private fun worldConfig(w: WorldDecl): Map<String, SpecConfigValue> {
        val m = TreeMap(w.given)
        when (w.observability) {
            Observability.FULL -> Unit
            Observability.PARTIAL -> m["partial_observability"] = SpecConfigValue.of(true)
            Observability.HIDDEN -> m["observability"] = SpecConfigValue.of("hidden")
        }
        return m
    }

    private fun experimentConfig(): Map<String, SpecConfigValue> {
        val m = TreeMap(config)
        if (phenomena.isNotEmpty()) m["phenomena_json"] = SpecConfigValue.of(phenomenaJson())
        if (vary.isNotEmpty()) m["vary_json"] = SpecConfigValue.of(varyJson())
        if (readings.isNotEmpty()) m["readings_json"] = SpecConfigValue.of(readingsJson())
        return m
    }

    private fun phenomenaJson(): String {
        val o = StringBuilder()
        list(o, phenomena.sortedBy { it.id }) { p ->
            when (p) {
                is PhenomenonSpec.Always -> {
                    o.append("{\"expr\":"); expr(o, p.expr)
                    o.append(",\"id\":"); str(o, p.id); o.append(",\"kind\":\"always\"}")
                }
                is PhenomenonSpec.Eventually -> {
                    o.append("{\"expr\":"); expr(o, p.expr)
                    o.append(",\"id\":"); str(o, p.id); o.append(",\"kind\":\"eventually\"")
                    p.withinTicks?.let { o.append(",\"within_ticks\":").append(it) }
                    o.append('}')
                }
                is PhenomenonSpec.After -> {
                    o.append("{\"id\":"); str(o, p.id); o.append(",\"kind\":\"after\",\"then\":")
                    expr(o, p.then); o.append(",\"trigger\":"); expr(o, p.trigger); o.append('}')
                }
                is PhenomenonSpec.Rate -> {
                    o.append("{\"comparator\":\"").append(p.cmp.wire).append("\",\"expr\":"); expr(o, p.expr)
                    o.append(",\"id\":"); str(o, p.id); o.append(",\"kind\":\"rate\",\"threshold\":")
                    f64(o, p.threshold); o.append('}')
                }
            }
        }
        return o.toString()
    }

    private fun varyJson(): String {
        val o = StringBuilder()
        list(o, vary) { v ->
            o.append("{\"param\":"); str(o, v.param); o.append(",\"values\":")
            list(o, v.values) { f64(o, it) }; o.append('}')
        }
        return o.toString()
    }

    private fun readingsJson(): String {
        val o = StringBuilder()
        list(o, readings.sortedBy { it.name }) { r ->
            o.append("{\"aggregate\":\"").append(r.aggregate.wire).append("\",\"name\":"); str(o, r.name)
            o.append(",\"of\":"); str(o, r.of); o.append(",\"unit\":"); str(o, r.unit.wireName); o.append('}')
        }
        return o.toString()
    }

    private fun emitHypothesis(o: StringBuilder, h: HypothesisDecl) {
        val sup = h.support!!
        val ref = h.refute!!
        o.append("{\"claim\":"); str(o, h.claim)
        o.append(",\"comparator\":\"").append(sup.first.wire).append("\",\"hypothesis_id\":"); str(o, h.id)
        o.append(",\"metric\":"); str(o, h.metric)
        o.append(",\"refutation_comparator\":\"").append(ref.first.wire).append("\",\"refutation_threshold\":")
        f64(o, ref.second); o.append(",\"threshold\":"); f64(o, sup.second); o.append('}')
    }

    private fun emitGoal(o: StringBuilder, g: GoalDecl) {
        o.append("{\"config\":{},\"expr\":"); expr(o, g.expr)
        o.append(",\"goal_id\":"); str(o, g.id)
        o.append(",\"maximize\":").append(g.maximize).append('}')
    }

    private fun emitConstraint(o: StringBuilder, c: ConstraintDecl) {
        o.append("{\"condition\":"); expr(o, c.expr)
        o.append(",\"config\":{},\"constraint_id\":"); str(o, c.id)
        o.append(",\"hard\":").append(c.hard).append('}')
    }

    private fun emitCognitive(o: StringBuilder, c: CognitiveDecl, modelId: String) {
        o.append("{\"entropic\":{\"entropy_weight\":"); f64(o, c.entropyWeight)
        o.append(",\"horizon\":").append(c.horizon).append(",\"objective\":\"").append(c.objective.wire)
        o.append("\",\"temperature\":"); f64(o, c.temperature)
        o.append("},\"learning\":{\"learning_rate\":"); f64(o, c.learningRate)
        o.append(",\"model\":"); str(o, modelId)
        o.append(",\"rule\":\"").append(c.algorithm!!.wire)
        o.append("\",\"update_every_ticks\":").append(c.updateEvery)
        o.append("},\"memory\":{\"capacity\":").append(c.capacity)
        o.append(",\"kind\":\"").append(c.memoryKind.wire).append("\",\"permissions\":")
        list(o, c.permissions) { str(o, it.wire) }
        o.append(",\"port\":{\"node\":"); str(o, c.memoryNode)
        o.append(",\"port\":"); str(o, c.memoryPort)
        o.append("}},\"resolution\":{\"binning_version\":"); str(o, c.binning!!)
        o.append(",\"levels\":")
        list(o, c.levels) { l ->
            o.append("{\"bins\":").append(l.bins).append(",\"level_id\":"); str(o, l.id)
            o.append(",\"sensor\":"); str(o, l.sensor); o.append('}')
        }
        o.append("},\"schema_version\":\"").append(COGNITIVE_SCHEMA).append("\"}")
    }

    private fun emitAgent(o: StringBuilder, a: AgentDecl, sys: SystemSpec, modelId: String) {
        val model = a.model!!
        o.append("{\"actions\":")
        list(o, a.actions) { str(o, it) }
        o.append(",\"agent_id\":"); str(o, a.id)
        cognitive?.let {
            o.append(",\"cognitive\":")
            emitCognitive(o, it, modelId)
        }
        o.append(",\"config\":")
        cfg(o, if (a.pure) mapOf("mode" to SpecConfigValue.of("pure")) else emptyMap())
        o.append(",\"constraints\":")
        list(o, a.constraints) { emitConstraint(o, it) }
        o.append(",\"goals\":")
        list(o, a.goals) { emitGoal(o, it) }
        o.append(",\"model\":{\"config\":")
        val mc = TreeMap(model.config)
        if (model.privileged) mc["privileged"] = SpecConfigValue.of(true)
        cfg(o, mc)
        o.append(",\"kind\":"); str(o, model.kind)
        o.append(",\"model_id\":"); str(o, model.id)
        o.append("},\"schema_version\":\"").append(AGENT_SCHEMA).append("\",\"sensors\":")
        list(o, a.sensors) { s ->
            val sc = TreeMap(s.config)
            if (s.observes.isNotEmpty()) sc["observes"] = SpecConfigValue.of(s.observes.joinToString(","))
            o.append("{\"config\":"); cfg(o, sc)
            o.append(",\"kind\":"); str(o, s.kind)
            o.append(",\"port_type\":"); CanonicalJson.writePortType(o, s.type)
            o.append(",\"sensor_id\":"); str(o, s.id)
            o.append(",\"source_node\":"); str(o, s.node)
            o.append(",\"source_port\":"); str(o, s.port)
            o.append('}')
        }
        o.append(",\"system\":")
        CanonicalJson.writeSystemSpec(o, sys)
        o.append('}')
    }
}

/** Entry point of the cognitive-experiment DSL. Throws [DslRejection] with a code and a path. */
fun experiment(id: String, block: ExperimentBuilder.() -> Unit): CognitiveExperiment =
    ExperimentBuilder(id).apply(block).build()
