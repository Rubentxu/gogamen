//! Drift-aware, observation-only Kart agent (experimental, `std`-only).
//!
//! # What this module adds over `kart_learned`
//! [`KartAdaptiveAgent`] wraps the GOG-173 learner and watches its *own* prediction surprise: for
//! every executed transition it scores the current frozen model (Brier score of the predicted
//! bucket distribution against the bucket actually reached, collisions included). The detector is
//! purely statistical over that signal:
//! 1. `calibrating`: the first `baseline_len` transitions give a mean/std of the surprise;
//! 2. `monitoring`: a rolling mean over `window` transitions is compared with
//!    `max(mean + z * std / sqrt(window), highest baseline rolling mean) + min_margin`;
//! 3. on a crossing ([`KartAdaptMode::ForgetOnDrift`] only) the learner is *hard-forgotten*
//!    (replaced by a fresh one, counts discarded) and re-fed with the most recent `window`
//!    transitions, publishing a new model version; afterwards it keeps learning online.
//!
//! The other modes are references: [`KartAdaptMode::Frozen`] never changes the model (the fixed
//! controller) and [`KartAdaptMode::AccumulateNoForget`] learns every transition without forgetting.
//!
//! # Architectural boundary
//! Like `kart_learned`, this file only sees observables and step outcomes handed in by the caller.
//! It cannot name the privileged simulator pieces; the test
//! `kart_adaptive_agent_source_has_no_privileged_tokens` scans this very file. That scan is a
//! source-level tripwire, not a capability system: numbers passed in are plain `f64`s.
//! The agent is never told that, when, or how the dynamics changed.

use std::collections::VecDeque;

use gogamen_core::{StepDt, TabularLearnedModel};

use crate::kart_learned::{
    KartExecutedTransition, KartLearnedAgent, KartLearnedError, KartMacroAction, KartObservable,
    KartPredictionError,
};

/// Learning policy of the agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KartAdaptMode {
    /// The model never changes after construction (fixed controller reference).
    Frozen,
    /// Detect drift from own surprise, then forget and relearn (the subject of the experiment).
    ForgetOnDrift,
    /// Start learning every transition after calibration, never forgetting (ablation).
    AccumulateNoForget,
}

/// Scalar surprise derived from the model's own prediction of one transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KartSurpriseSignal {
    /// Brier score over the full bucket distribution (collisions included).
    Brier,
    /// Circular yaw error of the predictive mean (rad) plus speed error / 10 (m/s), failures
    /// contribute 0 (their mean errors are not defined).
    MeanError,
}

/// Scalar summary of a prediction error under a signal.
pub fn surprise_of(signal: KartSurpriseSignal, e: &KartPredictionError) -> f64 {
    match signal {
        KartSurpriseSignal::Brier => e.mean_brier,
        KartSurpriseSignal::MeanError => e.mean_yaw_error + e.mean_speed_error / 10.0,
    }
}

/// Detector and adaptation parameters (declared before any result was inspected).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartAdaptConfig {
    /// Surprise signal used by the detector.
    pub signal: KartSurpriseSignal,
    /// Transitions used to estimate the baseline surprise (model frozen meanwhile).
    pub baseline_len: usize,
    /// Rolling window length of the detector.
    pub window: usize,
    /// Number of standard errors above the baseline mean required to alarm.
    pub z: f64,
    /// Minimum absolute margin above the baseline mean required to alarm.
    pub min_margin: f64,
    /// A new model version is published every this many online updates once adapting.
    pub republish_every: usize,
    /// Capacity of the recent-transition buffer.
    pub buffer_cap: usize,
}

impl Default for KartAdaptConfig {
    fn default() -> Self {
        Self {
            signal: KartSurpriseSignal::MeanError,
            baseline_len: 350,
            window: 12,
            z: 3.0,
            min_margin: 0.05,
            republish_every: 10,
            buffer_cap: 64,
        }
    }
}

/// Why a model version was published by the adaptive layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KartVersionReason {
    /// The version the agent was constructed with.
    Initial,
    /// Published right after a drift alarm (counts forgotten, recent window replayed).
    DriftRelearn,
    /// Periodic republish while learning online.
    Online,
}

/// One published version of the adaptive agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KartAdaptiveVersion {
    /// Learner generation (increments on every hard forget).
    pub generation: u32,
    /// Learner revision inside the generation.
    pub revision: u64,
    /// Learner checksum.
    pub checksum: u64,
    /// Number of observed transitions when published.
    pub at_step: usize,
    /// Reason of the publication.
    pub reason: KartVersionReason,
}

/// A drift alarm raised by the detector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KartDriftAlarm {
    /// Observed-transition index (0-based) at which the alarm fired.
    pub at_step: usize,
    /// Rolling surprise at the alarm.
    pub window_mean: f64,
    /// Threshold that was crossed.
    pub threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Calibrating,
    Monitoring,
    Adapting,
}

/// Observation-only agent with a surprise-based drift detector.
#[derive(Debug, Clone)]
pub struct KartAdaptiveAgent {
    mode: KartAdaptMode,
    cfg: KartAdaptConfig,
    inner: KartLearnedAgent,
    model: TabularLearnedModel,
    phase: Phase,
    steps: usize,
    generation: u32,
    baseline: Vec<f64>,
    baseline_mean: f64,
    baseline_std: f64,
    baseline_peak: f64,
    window: VecDeque<f64>,
    recent: VecDeque<KartExecutedTransition>,
    since_publish: usize,
    alarms: Vec<KartDriftAlarm>,
    versions: Vec<KartAdaptiveVersion>,
}

impl KartAdaptiveAgent {
    /// Wraps an already trained learner; the current learner state becomes the initial version.
    pub fn new(
        mut trained: KartLearnedAgent,
        mode: KartAdaptMode,
        cfg: KartAdaptConfig,
    ) -> Result<Self, KartLearnedError> {
        if cfg.window == 0 || cfg.baseline_len < 2 || cfg.republish_every == 0 {
            return Err(KartLearnedError::new("invalid adaptation parameters"));
        }
        let model = trained.publish()?;
        let rec = *trained
            .versions()
            .last()
            .ok_or_else(|| KartLearnedError::new("missing initial version"))?;
        Ok(Self {
            mode,
            cfg,
            inner: trained,
            model,
            phase: Phase::Calibrating,
            steps: 0,
            generation: 0,
            baseline: Vec::new(),
            baseline_mean: 0.0,
            baseline_std: 0.0,
            baseline_peak: 0.0,
            window: VecDeque::new(),
            recent: VecDeque::new(),
            since_publish: 0,
            alarms: Vec::new(),
            versions: vec![KartAdaptiveVersion {
                generation: 0,
                revision: rec.revision,
                checksum: rec.checksum,
                at_step: 0,
                reason: KartVersionReason::Initial,
            }],
        })
    }

    /// Current frozen model used for planning.
    pub fn model(&self) -> &TabularLearnedModel {
        &self.model
    }

    /// Alarms raised so far.
    pub fn alarms(&self) -> &[KartDriftAlarm] {
        &self.alarms
    }

    /// Versions published so far (including the initial one).
    pub fn versions(&self) -> &[KartAdaptiveVersion] {
        &self.versions
    }

    /// Baseline surprise `(mean, std)`; `(0, 0)` until calibration finished.
    pub fn baseline(&self) -> (f64, f64) {
        (self.baseline_mean, self.baseline_std)
    }

    /// Number of transitions observed.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Prediction error of the *current* model on executed transitions.
    pub fn prediction_error(
        &self,
        held: &[KartExecutedTransition],
    ) -> Result<KartPredictionError, KartLearnedError> {
        self.inner.prediction_error(&self.model, held)
    }

    /// Picks a macro action by planning on the current model.
    pub fn decide(
        &self,
        observed: KartObservable,
        macro_dt: StepDt,
        seed: u64,
    ) -> Result<KartMacroAction, KartLearnedError> {
        KartLearnedAgent::decide(&self.model, observed, macro_dt, seed)
    }

    fn surprise(&self, t: &KartExecutedTransition) -> Result<f64, KartLearnedError> {
        let e = self
            .inner
            .prediction_error(&self.model, std::slice::from_ref(t))?;
        Ok(surprise_of(self.cfg.signal, &e))
    }

    fn threshold(&self) -> f64 {
        let w = self.cfg.window as f64;
        let se_rule = self.baseline_mean + self.cfg.z * self.baseline_std / w.sqrt();
        se_rule.max(self.baseline_peak) + self.cfg.min_margin
    }

    fn record_version(&mut self, reason: KartVersionReason) {
        if let Some(rec) = self.inner.versions().last() {
            self.versions.push(KartAdaptiveVersion {
                generation: self.generation,
                revision: rec.revision,
                checksum: rec.checksum,
                at_step: self.steps,
                reason,
            });
        }
    }

    fn learn_online(&mut self, t: &KartExecutedTransition) -> Result<(), KartLearnedError> {
        self.inner.learn(t)?;
        self.since_publish += 1;
        if self.since_publish >= self.cfg.republish_every {
            self.model = self.inner.publish()?;
            self.since_publish = 0;
            self.record_version(KartVersionReason::Online);
        }
        Ok(())
    }

    /// Feeds one executed transition (the evaluator calls this after every macro step).
    ///
    /// Returns `true` when a drift alarm fired on this very transition.
    pub fn observe(&mut self, t: &KartExecutedTransition) -> Result<bool, KartLearnedError> {
        let s = self.surprise(t)?;
        self.steps += 1;
        self.recent.push_back(*t);
        while self.recent.len() > self.cfg.buffer_cap {
            self.recent.pop_front();
        }
        let mut alarm = false;
        match self.phase {
            Phase::Calibrating => {
                self.baseline.push(s);
                if self.baseline.len() >= self.cfg.baseline_len {
                    let n = self.baseline.len() as f64;
                    self.baseline_mean = self.baseline.iter().sum::<f64>() / n;
                    let var = self
                        .baseline
                        .iter()
                        .map(|x| (x - self.baseline_mean).powi(2))
                        .sum::<f64>()
                        / (n - 1.0);
                    self.baseline_std = var.sqrt();
                    // Highest rolling mean seen while calibrating: the stream is not i.i.d.
                    // (episodes differ), so a pure standard-error rule alarms on normal data.
                    let w = self.cfg.window.min(self.baseline.len());
                    self.baseline_peak = self
                        .baseline
                        .windows(w)
                        .map(|x| x.iter().sum::<f64>() / w as f64)
                        .fold(self.baseline_mean, f64::max);
                    self.phase = match self.mode {
                        KartAdaptMode::AccumulateNoForget => Phase::Adapting,
                        _ => Phase::Monitoring,
                    };
                }
            }
            Phase::Monitoring => {
                self.window.push_back(s);
                while self.window.len() > self.cfg.window {
                    self.window.pop_front();
                }
                if self.mode == KartAdaptMode::ForgetOnDrift && self.window.len() == self.cfg.window
                {
                    let mean = self.window.iter().sum::<f64>() / self.window.len() as f64;
                    let threshold = self.threshold();
                    if mean > threshold {
                        self.alarms.push(KartDriftAlarm {
                            at_step: self.steps - 1,
                            window_mean: mean,
                            threshold,
                        });
                        self.forget_and_relearn()?;
                        alarm = true;
                    }
                }
            }
            Phase::Adapting => self.learn_online(t)?,
        }
        Ok(alarm)
    }

    /// Hard forget: fresh learner, replay the most recent `window` transitions, publish.
    fn forget_and_relearn(&mut self) -> Result<(), KartLearnedError> {
        self.inner = KartLearnedAgent::new()?;
        self.generation += 1;
        let skip = self.recent.len().saturating_sub(self.cfg.window);
        let replay: Vec<KartExecutedTransition> = self.recent.iter().skip(skip).copied().collect();
        for t in &replay {
            self.inner.learn(t)?;
        }
        self.model = self.inner.publish()?;
        self.since_publish = 0;
        self.phase = Phase::Adapting;
        self.record_version(KartVersionReason::DriftRelearn);
        Ok(())
    }
}
