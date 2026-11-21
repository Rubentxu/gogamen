package dev.gogamen.dsl

/**
 * GOG-159 golden corpus of cognitive experiments. Each `hN` between its `// BEGIN` and `// END`
 * markers must stay within 15 lines (checked by `CognitiveDslCorpusTest`). The helpers below the
 * markers only declare the shared grid scenario (world given by the environment, agent given by design).
 */
class GridVars(
    val energy: Expr<Scalar<Dimensionless>>,
    val reward: Expr<Scalar<Dimensionless>>,
    val infoGain: Expr<Scalar<Dimensionless>>,
)

private val UNIT = Units.Dimensionless

fun ExperimentBuilder.gridBase(
    observability: Observability = Observability.PARTIAL,
    withBudget: Boolean = true,
    withSeeds: Boolean = true,
) {
    system(Fixtures.gridCognitiveSystem())
    world("grid_pomdp_7x7", "grid_pomdp") {
        given { "width" to 7; "height" to 7 }
        observability(observability)
    }
    if (withBudget) budget(1000, 2500, 10_000)
    if (withSeeds) seeds(listOf(11, 12, 13), listOf(1001, 1002))
    config { "headless" to true }
}

fun ExperimentBuilder.gridExplorer(
    isPure: Boolean = false,
    seen: List<String> = emptyList(),
    isPrivileged: Boolean = false,
): GridVars {
    val energy = scalar("energy", UNIT)
    val reward = scalar("reward", UNIT)
    val gain = scalar("info_gain", UNIT)
    agent("cognitive_explorer") {
        model("pomdp_belief_v0", "belief_state") {
            privileged = isPrivileged
            config {
                "belief_horizon" to 8
                "entropy_weight" to 0.25
            }
        }
        sensor("local_fov_sensor", "partial_fov", "pomdp_grid_env", "local_fov", PortType.Custom("LocalObservationGrid")) {
            observes(*seen.toTypedArray())
            config { "radius" to 2 }
        }
        actions("move_north", "move_south", "move_east", "move_west", "observe", "sense_local")
        goal("epistemic_utility", true, reward + const(0.25) * gain)
        constraint("energy_guard", true, energy gt const(0.0))
        if (isPure) pure()
    }
    return GridVars(energy, reward, gain)
}

fun ExperimentBuilder.singleLevelCognitive() = cognitive {
    resolution("v0") { level("only", "local_fov_sensor", 9) }
    learning { algorithm = LearningAlgorithm.BAYESIAN }
}

private val ALL_VARS = listOf("energy", "reward", "info_gain")

// BEGIN H0
fun h0() = experiment("grid_cognitive_spec_v0") {
    system(Fixtures.gridCognitiveSystem())
    world("grid_pomdp_7x7", "grid_pomdp") { given { "width" to 7; "height" to 7 }; observability(Observability.PARTIAL) }
    gridExplorer()
    cognitive {
        resolution("v0") { levels("local_fov_sensor", "coarse" to 3, "mid" to 9, "fine" to 25) }
        learning { algorithm = LearningAlgorithm.BAYESIAN }
    }
    budget(1000, 2500, 10_000)
    seeds(listOf(11, 12, 13), listOf(1001, 1002))
    config { "headless" to true }
}
// END H0

// BEGIN H1
fun h1() = experiment("h1_temperature_sweep") {
    gridBase()
    gridExplorer()
    cognitive {
        resolution("v0") { levels("local_fov_sensor", "coarse" to 3, "mid" to 9) }
        learning { algorithm = LearningAlgorithm.BAYESIAN }
    }
    vary("entropic.temperature", 0.1, 0.5, 1.0)
    expectRuns(15)
}
// END H1

// BEGIN H2
fun h2() = experiment("h2_multi_scale_resolution") {
    gridBase()
    gridExplorer()
    cognitive {
        resolution("v1") { levels("local_fov_sensor", "c" to 2, "m" to 6, "f" to 18, "xf" to 54) }
        learning { algorithm = LearningAlgorithm.NONE }
    }
    vary("world.width", 5.0, 7.0)
    expectRuns(10)
}
// END H2

// BEGIN H3
fun h3() = experiment("h3_learning_rate") {
    gridBase()
    gridExplorer()
    cognitive {
        resolution("v0") { level("only", "local_fov_sensor", 9) }
        learning { algorithm = LearningAlgorithm.TD }
    }
    vary("learning.learning_rate", 0.01, 0.1, 0.5)
    vary("learning.update_every_ticks", 1.0, 8.0)
    expectRuns(30)
}
// END H3

// BEGIN H4
fun h4() = experiment("h4_memory_capacity") {
    gridBase()
    gridExplorer()
    cognitive {
        resolution("v0") { level("only", "local_fov_sensor", 9) }
        learning { algorithm = LearningAlgorithm.HEBBIAN }
        memory { kind = MemoryKind.WORKING; capacity = 64 }
    }
    vary("memory.capacity", 16.0, 64.0, 256.0, 1024.0)
    expectRuns(20)
}
// END H4

// BEGIN H5
fun h5() = experiment("h5_planning_horizon") {
    gridBase()
    gridExplorer()
    cognitive {
        entropic { objective = Objective.INFO_GAIN; horizon = 8 }
        resolution("v0") { level("only", "local_fov_sensor", 9) }
        learning { algorithm = LearningAlgorithm.BAYESIAN }
    }
    vary("entropic.horizon", 4.0, 8.0, 16.0, 32.0)
    vary("entropic.entropy_weight", 0.0, 0.5)
    expectRuns(40)
}
// END H5

// BEGIN H6
fun h6() = experiment("h6_trace_phenomena") {
    gridBase()
    val g = gridExplorer()
    cognitive {
        resolution("v0") { level("only", "local_fov_sensor", 9) }
        learning { algorithm = LearningAlgorithm.BAYESIAN }
    }
    phenomenon("stays_alive") { always(g.energy gt const(0.0)) }
    phenomenon("finds_reward") { eventually(g.reward gt const(0.0), within = 200) }
    phenomenon("alive_after_reward") { after(g.reward gt const(0.0), g.energy gt const(0.0)) }
    phenomenon("mostly_informative") { rate(g.infoGain gt const(0.0), Cmp.GE, 0.3) }
}
// END H6

// BEGIN H7
fun h7() = experiment("h7_pure_agent_readings") {
    gridBase()
    val g = gridExplorer(isPure = true, seen = ALL_VARS)
    singleLevelCognitive()
    phenomenon("stays_alive") { always(g.energy gt const(0.0)) }
    reading("alive_fraction", ExprUnit.RATIO, Aggregate.HOLD_FRACTION, "stays_alive")
    hypothesis("curious_agents_survive") {
        claim = "a pure curious agent keeps energy positive on 90% of ticks"
        metric = "alive_fraction"
        supportedIf(Cmp.GE, 0.9)
        refutedIf(Cmp.LT, 0.5)
    }
}
// END H7

val POSITIVE_CORPUS: Map<String, () -> CognitiveExperiment> = linkedMapOf(
    "h0" to ::h0, "h1" to ::h1, "h2" to ::h2, "h3" to ::h3,
    "h4" to ::h4, "h5" to ::h5, "h6" to ::h6, "h7" to ::h7,
)

/** Experiments that must be rejected; each id has a row in `spec/golden/cognitive_dsl/cases.tsv`. */
val NEGATIVE_CORPUS: Map<String, () -> CognitiveExperiment> = linkedMapOf(
    "behavior_rule" to {
        experiment("n_behavior_rule") {
            gridBase()
            agent("scripted") { rule("if wall then go east") }
        }
    },
    "pure_leak" to {
        experiment("n_pure_leak") {
            gridBase()
            gridExplorer(isPure = true, seen = listOf("energy"))
        }
    },
    "privileged_hidden" to {
        experiment("n_privileged_hidden") {
            gridBase(observability = Observability.HIDDEN)
            gridExplorer(isPrivileged = true)
        }
    },
    "learning_no_algorithm" to {
        experiment("n_learning_no_algorithm") {
            gridBase()
            gridExplorer()
            cognitive {
                resolution("v0") { level("only", "local_fov_sensor", 9) }
                learning { rate = 0.2 }
            }
        }
    },
    "no_budget" to {
        experiment("n_no_budget") {
            gridBase(withBudget = false)
            gridExplorer()
        }
    },
    "no_seeds" to {
        experiment("n_no_seeds") {
            gridBase(withSeeds = false)
            gridExplorer()
        }
    },
    "seeds_overlap" to {
        experiment("n_seeds_overlap") {
            gridBase(withSeeds = false)
            gridExplorer()
            seeds(listOf(11, 12), listOf(12, 1001))
        }
    },
    "unit_mismatch" to {
        experiment("n_unit_mismatch") {
            gridBase()
            val e = scalar("stored", Units.Energy)
            val r = scalar("curiosity", Units.Ratio)
            agent("cognitive_explorer") {
                model("pomdp_belief_v0", "belief_state")
                goal("bad_sum", true, Expr<Scalar<Energy>>(ExprAst.Add(e.ast, r.ast)))
            }
        }
    },
    "hypothesis_no_refutation" to {
        experiment("n_hypothesis_no_refutation") {
            gridBase()
            gridExplorer()
            hypothesis("unfalsifiable") {
                claim = "the agent always wins"
                metric = "win_rate"
                supportedIf(Cmp.GE, 0.9)
            }
        }
    },
    "matrix_size_mismatch" to {
        experiment("n_matrix_size_mismatch") {
            gridBase()
            gridExplorer()
            cognitive {
                resolution("v0") { level("only", "local_fov_sensor", 9) }
                learning { algorithm = LearningAlgorithm.BAYESIAN }
            }
            vary("entropic.temperature", 0.1, 0.5)
            expectRuns(7)
        }
    },
)
