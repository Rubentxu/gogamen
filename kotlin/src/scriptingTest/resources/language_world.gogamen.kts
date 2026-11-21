// Trusted authoring script: equivalent of Fixtures.canonicalLanguageWorld().
epistemicSystemSpec("language_world_canonical_v0") {
    node("human_channel", "language::human_channel") {
        output("observation", PortType.F64)
        output("message_confidence", PortType.F64)
        output("is_order", PortType.Bool)
        provenance {
            sourceId = "human_channel"
            sensorId = 100L
            evidenceRef = 101L
            epistemicStatus = "reported"
        }
        config {
            "speech_act" to "notice"
        }
    }
    node("direct_sensor", "language::direct_sensor") {
        output("direct_obs", PortType.F64)
        provenance {
            sourceId = "direct_sensor"
            sensorId = 200L
            evidenceRef = 500L
            epistemicStatus = "observed"
        }
    }
    node("grounder", "language::grounder") {
        input("observation", PortType.F64)
        input("message_confidence", PortType.F64)
        output("grounded_prob", PortType.F64)
        output("abstained", PortType.Bool)
        choiceQuestion("lw-entity-notice-1") {
            prompt = "Which registered cell does the human notice refer to?"
            options("cell_2_1", "cell_1_0")
            modelId = "offline_replay_v1"
            deadlineTicks = 5L
        }
        abstentionPolicy {
            minConfidence = 0.65
            fallbackAction = "abstain"
            abstainOnTimeout = true
            abstainOnAmbiguity = true
            ambiguityMargin = 0.05
        }
        config {
            "source_id" to "human_channel"
        }
    }
    node("belief", "language::belief") {
        input("grounded_prob", PortType.F64)
        input("abstained", PortType.Bool)
        input("direct_obs", PortType.F64)
        output("belief_prob", PortType.F64)
        output("should_ask", PortType.Bool)
        output("should_inspect", PortType.Bool)
        askAction {
            recipient = "human_operator"
            scope = "language_world"
            commCost = 0.5
            stepCost = 0.1
            responseProbability = 0.85
            timeoutTicks = 3L
            noulQuestion("lw-ask-obstacle-1") {
                prompt = "Is cell (2,1) blocked by an obstacle?"
                hypothesisId = "obstacle_at_2_1"
                modelId = "system_one_noul_v1"
                deadlineTicks = 3L
            }
        }
        inspectAction {
            sensorId = "direct_sensor"
            inspectCost = 0.25
            targetEntity = "cell_2_1"
        }
        config {
            "prior_obstacle_prob" to 0.5
        }
    }
    node("guarded_planner", "language::guarded_planner") {
        input("belief_prob", PortType.F64)
        input("should_ask", PortType.Bool)
        input("should_inspect", PortType.Bool)
        input("is_order", PortType.Bool)
        output("action", PortType.I64)
        output("guard_rejected", PortType.Bool)
        guardPolicy {
            guardPolicy = "reject_only"
            allowActions("stay", "right", "down", "left")
            immutablePermissions = true
            humanMessagesCannotMutatePermissions = true
        }
    }
    connections {
        instant("human_channel", "observation", "grounder", "observation")
        instant("human_channel", "message_confidence", "grounder", "message_confidence")
        instant("grounder", "grounded_prob", "belief", "grounded_prob")
        instant("grounder", "abstained", "belief", "abstained")
        instant("direct_sensor", "direct_obs", "belief", "direct_obs")
        instant("belief", "belief_prob", "guarded_planner", "belief_prob")
        instant("belief", "should_ask", "guarded_planner", "should_ask")
        instant("belief", "should_inspect", "guarded_planner", "should_inspect")
        instant("human_channel", "is_order", "guarded_planner", "is_order")
    }
    config {
        "deterministic" to true
        "dt_secs" to 0.02
    }
}
