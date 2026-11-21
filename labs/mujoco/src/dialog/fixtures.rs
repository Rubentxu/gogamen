//! Annotated human-utterance fixtures (synthetic, hand-written, deterministic).
//!
//! The hypothesis is always "the target is the LEFT door" (`true`) vs "the RIGHT door" (`false`).
//! Each fixture carries the ground-truth annotation, the recorded Jev interpretation of every
//! utterance (a [`NoulVerdict`] plus latency) and what the human would answer if asked.

use gogamen_core::NoulVerdict;

/// Coarse category of a fixture (what the human did, not whether they were right).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixtureKind {
    /// A clear single claim (it may still be factually wrong: the speaker can be mistaken).
    Correct,
    /// The utterance carries no usable claim.
    Ambiguous,
    /// Two utterances that claim opposite doors.
    Contradictory,
}

/// One utterance and its recorded interpretation.
#[derive(Debug, Clone, PartialEq)]
pub struct Claim {
    /// Surface text.
    pub text: &'static str,
    /// Recorded Jev verdict on "target is the left door".
    pub verdict: NoulVerdict,
    /// Recorded Jev latency in control ticks (10 ms each).
    pub latency_ticks: u64,
}

/// What the human answers when asked "left or right?".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanReply {
    /// `Some(true)` = "left", `Some(false)` = "right", `None` = no reply at all.
    pub answer_left: Option<bool>,
    /// Reply latency in control ticks.
    pub latency_ticks: u64,
}

/// One annotated dialog scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct DialogFixture {
    /// Stable id.
    pub id: &'static str,
    /// Category.
    pub kind: FixtureKind,
    /// Utterances, in order.
    pub claims: Vec<Claim>,
    /// Reply to an `AskHuman` question.
    pub human: HumanReply,
    /// Ground truth: the target door is the left one.
    pub truth_left: bool,
}

const L: NoulVerdict = NoulVerdict::True;
const R: NoulVerdict = NoulVerdict::False;
const U: NoulVerdict = NoulVerdict::Unknown;

fn c(text: &'static str, verdict: NoulVerdict) -> Claim {
    Claim {
        text,
        verdict,
        latency_ticks: 5,
    }
}

fn h(answer_left: Option<bool>) -> HumanReply {
    HumanReply {
        answer_left,
        latency_ticks: 60,
    }
}

fn fx(
    id: &'static str,
    kind: FixtureKind,
    claims: Vec<Claim>,
    human: HumanReply,
    truth_left: bool,
) -> DialogFixture {
    DialogFixture {
        id,
        kind,
        claims,
        human,
        truth_left,
    }
}

/// The 16 annotated fixtures used by the evaluation (8 clear claims, 4 ambiguous, 4 contradictory).
/// Three of the clear claims are factually wrong (a mistaken speaker) and two of the human
/// answers are wrong or missing in each of the ambiguous/contradictory groups, so calibration is
/// not trivially perfect.
pub fn annotated_fixtures() -> Vec<DialogFixture> {
    use FixtureKind::{Ambiguous, Contradictory, Correct};
    vec![
        fx(
            "c01-left",
            Correct,
            vec![c("go to the left door", L)],
            h(Some(true)),
            true,
        ),
        fx(
            "c02-right",
            Correct,
            vec![c("take the right door", R)],
            h(Some(false)),
            false,
        ),
        fx(
            "c03-left",
            Correct,
            vec![c("the left one, please", L)],
            h(Some(true)),
            true,
        ),
        fx(
            "c04-right",
            Correct,
            vec![c("right door", R)],
            h(Some(false)),
            false,
        ),
        fx(
            "c05-not-left",
            Correct,
            vec![c("not the left door", R)],
            h(Some(false)),
            false,
        ),
        // The speaker is mistaken in the next three (annotation: truth differs from the claim).
        fx(
            "c06-left-mistaken",
            Correct,
            vec![c("go to the left door", L)],
            h(Some(false)),
            false,
        ),
        fx(
            "c07-right-mistaken",
            Correct,
            vec![c("the right door", R)],
            h(Some(true)),
            true,
        ),
        fx(
            "c08-left-mistaken-mute",
            Correct,
            vec![c("left door", L)],
            h(None),
            false,
        ),
        fx(
            "a01-that-door",
            Ambiguous,
            vec![c("that door over there", U)],
            h(Some(true)),
            true,
        ),
        fx(
            "a02-whichever",
            Ambiguous,
            vec![c("whichever door", U)],
            h(Some(false)),
            false,
        ),
        fx(
            "a03-mumble",
            Ambiguous,
            vec![c("uh, the... door", U)],
            h(Some(false)),
            true,
        ),
        fx(
            "a04-pointing",
            Ambiguous,
            vec![c("that one", U)],
            h(None),
            false,
        ),
        fx(
            "x01-left-then-right",
            Contradictory,
            vec![c("left door", L), c("no wait, the right door", R)],
            h(Some(false)),
            false,
        ),
        fx(
            "x02-right-then-left",
            Contradictory,
            vec![c("right door", R), c("sorry, I meant left", L)],
            h(Some(true)),
            true,
        ),
        fx(
            "x03-flip-flop",
            Contradictory,
            vec![c("the left one", L), c("actually the right one", R)],
            h(Some(false)),
            true,
        ),
        fx(
            "x04-silent",
            Contradictory,
            vec![c("right door", R), c("no, left door", L)],
            h(None),
            false,
        ),
    ]
}

/// First fixture of a kind (convenience for the canonical AC1 cases).
pub fn first_of(kind: FixtureKind) -> DialogFixture {
    annotated_fixtures()
        .into_iter()
        .find(|f| f.kind == kind)
        .expect("every kind has fixtures")
}
