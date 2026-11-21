//! Explicit dialog policy: speaker reliability, act threshold, ask cost and payoffs.
//!
//! The ask/advance trade-off is delegated to `gogamen_core::EpistemicActionSelector`
//! (EVOI minus explicit cost) over a `DecisionPayoffModel`; this module only supplies the numbers
//! and the final act-or-wait rule.

use std::time::Duration;

use gogamen_core::{
    Action, AdvanceSpec, AskSpec, BinaryBelief, CommCost, CommCostWeights, ConversationScope,
    DecisionOption, DecisionPayoffModel, EpistemicActionKind, EpistemicActionSelector,
    EpistemicSelectionReport, InterlocutorId, LikelihoodModel, OptionId,
};

use super::oracle::TICK_NANOS;

/// One of the two doors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Door {
    /// Left door (+y side), hypothesis `true`.
    Left,
    /// Right door (-y side), hypothesis `false`.
    Right,
}

impl Door {
    /// Goal position in the corridor frame.
    pub const fn position(self) -> [f64; 2] {
        match self {
            Self::Left => [5.0, 0.8],
            Self::Right => [5.0, -0.8],
        }
    }

    /// MAP door for a belief (`P(left) >= 0.5` -> left; the tie goes left, deterministically).
    pub fn map(belief: &BinaryBelief) -> Self {
        if belief.p_true() >= belief.p_false() {
            Self::Left
        } else {
            Self::Right
        }
    }

    /// `true` for the left door.
    pub const fn is_left(self) -> bool {
        matches!(self, Self::Left)
    }
}

/// Policy family compared in the evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    /// Never ask; always drive to the MAP door, however weak the belief.
    NoAskGuess,
    /// Never ask; drive only if confidence reaches the threshold, otherwise wait.
    NoAskAbstain,
    /// Ask only when the core selector says EVOI exceeds the cost, then act under the threshold.
    EpistemicAsk,
    /// Always ask once, then act under the threshold.
    AlwaysAsk,
}

impl PolicyKind {
    /// Label for tables.
    pub const fn label(self) -> &'static str {
        match self {
            Self::NoAskGuess => "no-ask (always guess)",
            Self::NoAskAbstain => "no-ask (threshold)",
            Self::EpistemicAsk => "ask-when-worth-it (EVOI)",
            Self::AlwaysAsk => "always-ask",
        }
    }
}

/// What to do right after the utterances were perceived.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreAsk {
    /// Send an `AskHuman` question.
    Ask,
    /// Drive to this door.
    Go(Door),
    /// Hold position.
    Wait,
}

/// Explicit policy parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DialogPolicy {
    /// Policy family.
    pub kind: PolicyKind,
    /// Prior `P(left)`.
    pub prior_p_left: f64,
    /// Speaker reliability (sensitivity = specificity of the utterance likelihood).
    pub speaker_reliability: f64,
    /// Human reply reliability (likelihood of an `AskHuman` answer).
    pub human_reliability: f64,
    /// Minimum confidence in the MAP door to drive.
    pub act_threshold: f64,
    /// Utility of reaching the right door.
    pub reward: f64,
    /// Utility of reaching the wrong door (negative).
    pub penalty: f64,
    /// Communication cost units of one question (weight 1.0 per unit).
    pub ask_cost_units: u64,
    /// Estimated probability that the human replies at all.
    pub response_probability: f64,
    /// Oracle deadline in control ticks.
    pub oracle_timeout_ticks: u64,
    /// `AskHuman` deadline in control ticks.
    pub ask_timeout_ticks: u64,
}

impl DialogPolicy {
    /// Default numbers (documented in the README).
    pub fn new(kind: PolicyKind) -> Self {
        Self {
            kind,
            prior_p_left: 0.5,
            speaker_reliability: 0.9,
            human_reliability: 0.95,
            act_threshold: 0.85,
            reward: 10.0,
            penalty: -20.0,
            ask_cost_units: 2,
            response_probability: 0.9,
            oracle_timeout_ticks: 50,
            ask_timeout_ticks: 120,
        }
    }

    /// Payoff model over `go_left` / `go_right` / `wait`.
    pub fn payoff_model(&self) -> DecisionPayoffModel {
        let opt = |id: &str, f: f64, t: f64, door: Option<Door>| {
            let o = DecisionOption::new(OptionId::new(id).expect("static id"), f, t)
                .expect("finite utilities");
            match door {
                Some(d) => {
                    o.with_action(Action::continuous(d.position().to_vec()).expect("finite"))
                }
                None => o,
            }
        };
        DecisionPayoffModel::new(vec![
            opt("go_left", self.penalty, self.reward, Some(Door::Left)),
            opt("go_right", self.reward, self.penalty, Some(Door::Right)),
            opt("wait", 0.0, 0.0, None),
        ])
        .expect("distinct options")
    }

    /// The `AskHuman` specification (human reply likelihood, cost, timeout).
    pub fn ask_spec(&self) -> AskSpec {
        let opt = |s: &str| OptionId::new(s).expect("static id");
        AskSpec::new(
            InterlocutorId::new("human").expect("static id"),
            ConversationScope::session("door-disambiguation").expect("static scope"),
            "Which door: left or right?",
            vec![opt("left"), opt("right")],
            CommCost::from_units(self.ask_cost_units),
            0.0,
            LikelihoodModel::from_sensitivity_specificity(
                self.human_reliability,
                self.human_reliability,
            )
            .expect("reliability in [0,1]"),
            self.response_probability,
            Duration::from_nanos((self.ask_timeout_ticks as u128 * TICK_NANOS) as u64),
        )
        .expect("valid ask spec")
    }

    /// Core selector report (EVOI vs. cost) for `belief`.
    pub fn selector_report(&self, belief: &BinaryBelief) -> EpistemicSelectionReport {
        EpistemicActionSelector::evaluate(
            belief,
            &self.payoff_model(),
            CommCostWeights::from_unit_weight(1.0).expect("valid weight"),
            &AdvanceSpec::default(),
            None,
            Some(&self.ask_spec()),
        )
        .expect("valid epistemic inputs")
    }

    /// Act-or-wait under the threshold (used after the last piece of evidence).
    pub fn final_choice(&self, belief: &BinaryBelief) -> Option<Door> {
        let conf = belief.p_true().max(belief.p_false());
        (conf >= self.act_threshold).then(|| Door::map(belief))
    }

    /// Decision after the utterances were perceived.
    pub fn pre_ask(&self, belief: &BinaryBelief) -> PreAsk {
        match self.kind {
            PolicyKind::NoAskGuess => PreAsk::Go(Door::map(belief)),
            PolicyKind::NoAskAbstain => self.final_choice(belief).map_or(PreAsk::Wait, PreAsk::Go),
            PolicyKind::AlwaysAsk => PreAsk::Ask,
            PolicyKind::EpistemicAsk => {
                if self.selector_report(belief).selected == EpistemicActionKind::Ask {
                    PreAsk::Ask
                } else {
                    self.final_choice(belief).map_or(PreAsk::Wait, PreAsk::Go)
                }
            }
        }
    }
}
