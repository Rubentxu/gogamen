//! Learning-progress drive reader (GOG-187).
//!
//! A [`crate::DriveSpec`] whose *reading* is the recent **drop** of the learner's prequential
//! `log_loss` per region (learning progress), next to a baseline reader that uses the **raw**
//! recent `log_loss` (prediction error). Both readers are pure, typed and deterministic: no
//! clocks, no RNG, only the sequence of `(region, log_loss)` samples fed to
//! [`LearningProgressReader::record`].
//!
//! The reading is meant to be fed to [`crate::HomeostaticState::set_reading`] so that
//! [`crate::DynamicUtility`] weights an exploration objective by the drive's deficit; no
//! dedicated exploration mechanism exists here.
//!
//! # Windows
//! Per region only the last `2 * window` losses are kept. With `old` = mean of the older
//! `window` samples and `recent` = mean of the newer `window` samples:
//! - [`DriveSignal::RawError`]: `recent / loss_scale`, needs `window` samples.
//! - [`DriveSignal::LearningProgress`]: `max(old - recent, 0) / loss_scale`, needs
//!   `2 * window` samples. Below that the reading is [`RegionReading::Insufficient`]
//!   (the caller decides how to treat the unknown, e.g. optimistically).
//!
//! Both readings are clamped to `[0, 1]`.

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

use crate::motivation::{DriveId, DriveSpec, MotivationError};

/// Errors of the learning-progress reader.
#[derive(Debug, Clone, PartialEq)]
pub enum LearningProgressError {
    /// `window` must be at least 1.
    InvalidWindow,
    /// `regions` must be at least 1.
    InvalidRegionCount,
    /// `loss_scale` must be finite and strictly positive.
    InvalidLossScale(f64),
    /// Region index outside `0..regions`.
    RegionOutOfRange {
        /// Offending region.
        region: usize,
        /// Number of regions.
        regions: usize,
    },
    /// A loss must be finite and non-negative.
    InvalidLoss(f64),
    /// Drive construction failed.
    Drive(MotivationError),
}

impl fmt::Display for LearningProgressError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindow => write!(f, "window must be >= 1"),
            Self::InvalidRegionCount => write!(f, "regions must be >= 1"),
            Self::InvalidLossScale(v) => write!(f, "loss_scale must be finite and > 0, got {v}"),
            Self::RegionOutOfRange { region, regions } => {
                write!(f, "region {region} out of range (regions = {regions})")
            }
            Self::InvalidLoss(v) => write!(f, "loss must be finite and >= 0, got {v}"),
            Self::Drive(e) => write!(f, "drive error: {e}"),
        }
    }
}

impl Error for LearningProgressError {}

impl From<MotivationError> for LearningProgressError {
    fn from(e: MotivationError) -> Self {
        Self::Drive(e)
    }
}

/// Which quantity a drive reads from the loss stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriveSignal {
    /// Recent mean `log_loss` (raw prediction error).
    RawError,
    /// Recent drop of `log_loss` (learning progress).
    LearningProgress,
}

impl DriveSignal {
    /// Stable label for reports.
    pub const fn label(self) -> &'static str {
        match self {
            Self::RawError => "raw_error",
            Self::LearningProgress => "learning_progress",
        }
    }
}

/// Modality of a drive: where its reading comes from.
///
/// No modality type pre-existed in the repo, so this minimal marker lives here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriveModality {
    /// Reading depends only on a quantity that is independent of the agent's own history.
    Pure,
    /// Reading depends on the agent's situated interaction history (what it visited and
    /// what it has learned from it).
    Situated,
}

impl DriveModality {
    /// Modality of the reading of a given signal. Both signals are computed from the
    /// learner's losses on the agent's own transitions, hence situated.
    pub const fn of_signal(_signal: DriveSignal) -> Self {
        Self::Situated
    }
}

/// Reading of one region.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegionReading {
    /// Not enough samples yet.
    Insufficient {
        /// Samples held.
        samples: usize,
        /// Samples needed.
        needed: usize,
    },
    /// Normalised reading in `[0, 1]`.
    Value(f64),
}

impl RegionReading {
    /// The value, or `fallback` when insufficient.
    pub const fn value_or(self, fallback: f64) -> f64 {
        match self {
            Self::Value(v) => v,
            Self::Insufficient { .. } => fallback,
        }
    }
}

/// Reader configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearningProgressConfig {
    window: usize,
    regions: usize,
    loss_scale: f64,
}

impl LearningProgressConfig {
    /// Validates and builds a configuration.
    pub fn new(
        window: usize,
        regions: usize,
        loss_scale: f64,
    ) -> Result<Self, LearningProgressError> {
        if window == 0 {
            return Err(LearningProgressError::InvalidWindow);
        }
        if regions == 0 {
            return Err(LearningProgressError::InvalidRegionCount);
        }
        if !loss_scale.is_finite() || loss_scale <= 0.0 {
            return Err(LearningProgressError::InvalidLossScale(loss_scale));
        }
        Ok(Self {
            window,
            regions,
            loss_scale,
        })
    }

    /// Window length (samples per half-window).
    pub const fn window(&self) -> usize {
        self.window
    }

    /// Number of regions.
    pub const fn regions(&self) -> usize {
        self.regions
    }

    /// Loss normalisation scale (e.g. `ln(num_states)`).
    pub const fn loss_scale(&self) -> f64 {
        self.loss_scale
    }
}

/// Windowed per-region `log_loss` reader.
#[derive(Debug, Clone, PartialEq)]
pub struct LearningProgressReader {
    config: LearningProgressConfig,
    losses: Vec<VecDeque<f64>>,
}

impl LearningProgressReader {
    /// Creates an empty reader.
    pub fn new(config: LearningProgressConfig) -> Self {
        Self {
            losses: vec![VecDeque::with_capacity(2 * config.window + 1); config.regions],
            config,
        }
    }

    /// Configuration.
    pub const fn config(&self) -> &LearningProgressConfig {
        &self.config
    }

    /// Appends one loss for `region` (oldest samples beyond `2 * window` are dropped).
    pub fn record(&mut self, region: usize, log_loss: f64) -> Result<(), LearningProgressError> {
        if !log_loss.is_finite() || log_loss < 0.0 {
            return Err(LearningProgressError::InvalidLoss(log_loss));
        }
        let regions = self.config.regions;
        let q = self
            .losses
            .get_mut(region)
            .ok_or(LearningProgressError::RegionOutOfRange { region, regions })?;
        q.push_back(log_loss);
        if q.len() > 2 * self.config.window {
            q.pop_front();
        }
        Ok(())
    }

    /// Samples currently held for `region` (at most `2 * window`).
    pub fn samples(&self, region: usize) -> usize {
        self.losses.get(region).map_or(0, VecDeque::len)
    }

    fn mean(xs: impl Iterator<Item = f64>, n: usize) -> f64 {
        xs.sum::<f64>() / n as f64
    }

    /// Reading of `region` for `signal`.
    pub fn read(
        &self,
        region: usize,
        signal: DriveSignal,
    ) -> Result<RegionReading, LearningProgressError> {
        let regions = self.config.regions;
        let q = self
            .losses
            .get(region)
            .ok_or(LearningProgressError::RegionOutOfRange { region, regions })?;
        let (w, scale) = (self.config.window, self.config.loss_scale);
        let n = q.len();
        Ok(match signal {
            DriveSignal::RawError => {
                if n < w {
                    return Ok(RegionReading::Insufficient {
                        samples: n,
                        needed: w,
                    });
                }
                let recent = Self::mean(q.iter().skip(n - w).copied(), w);
                RegionReading::Value((recent / scale).clamp(0.0, 1.0))
            }
            DriveSignal::LearningProgress => {
                if n < 2 * w {
                    return Ok(RegionReading::Insufficient {
                        samples: n,
                        needed: 2 * w,
                    });
                }
                let old = Self::mean(q.iter().take(w).copied(), w);
                let recent = Self::mean(q.iter().skip(n - w).copied(), w);
                RegionReading::Value(((old - recent) / scale).clamp(0.0, 1.0))
            }
        })
    }

    /// Mean reading over the regions that have a value, with the count of such regions.
    pub fn mean_reading(&self, signal: DriveSignal) -> Option<(f64, usize)> {
        let (mut sum, mut k) = (0.0, 0usize);
        for r in 0..self.config.regions {
            if let Ok(RegionReading::Value(v)) = self.read(r, signal) {
                sum += v;
                k += 1;
            }
        }
        (k > 0).then(|| (sum / k as f64, k))
    }

    /// Largest reading over the regions that have a value.
    pub fn max_reading(&self, signal: DriveSignal) -> Option<f64> {
        (0..self.config.regions)
            .filter_map(|r| match self.read(r, signal) {
                Ok(RegionReading::Value(v)) => Some(v),
                _ => None,
            })
            .reduce(f64::max)
    }
}

/// Builds the [`DriveSpec`] of a drive whose reading is a normalised signal in `[0, 1]`.
///
/// `target = 0` (nothing left to learn / no error), bounds `[0, 1]`, so the deficit equals
/// the reading. The weights are given by the designer.
pub fn signal_drive_spec(
    id: DriveId,
    base_weight: f64,
    deficit_gain: f64,
) -> Result<DriveSpec, LearningProgressError> {
    Ok(DriveSpec::new(
        id,
        0.0,
        0.0,
        1.0,
        base_weight,
        deficit_gain,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motivation::HomeostaticState;

    fn reader(window: usize) -> LearningProgressReader {
        LearningProgressReader::new(LearningProgressConfig::new(window, 2, 2.0).unwrap())
    }

    #[test]
    fn config_validation() {
        assert_eq!(
            LearningProgressConfig::new(0, 1, 1.0),
            Err(LearningProgressError::InvalidWindow)
        );
        assert_eq!(
            LearningProgressConfig::new(1, 0, 1.0),
            Err(LearningProgressError::InvalidRegionCount)
        );
        assert!(LearningProgressConfig::new(1, 1, 0.0).is_err());
        assert!(LearningProgressConfig::new(1, 1, f64::NAN).is_err());
        let mut r = reader(1);
        assert!(r.record(5, 1.0).is_err());
        assert!(r.record(0, -1.0).is_err());
        assert!(r.record(0, f64::INFINITY).is_err());
        assert!(r.read(9, DriveSignal::RawError).is_err());
    }

    #[test]
    fn insufficient_until_window_is_full() {
        let mut r = reader(2);
        r.record(0, 1.0).unwrap();
        r.record(0, 1.0).unwrap();
        r.record(0, 1.0).unwrap();
        assert!(matches!(
            r.read(0, DriveSignal::LearningProgress).unwrap(),
            RegionReading::Insufficient {
                samples: 3,
                needed: 4
            }
        ));
        assert!(matches!(
            r.read(0, DriveSignal::RawError).unwrap(),
            RegionReading::Value(_)
        ));
        assert!(matches!(
            r.read(1, DriveSignal::RawError).unwrap(),
            RegionReading::Insufficient { samples: 0, .. }
        ));
    }

    #[test]
    fn progress_is_drop_and_error_is_level() {
        let mut r = reader(2);
        for l in [2.0, 2.0, 0.0, 0.0] {
            r.record(0, l).unwrap();
        }
        // old = 2, recent = 0, scale = 2 -> progress 1.0, error 0.0
        assert_eq!(
            r.read(0, DriveSignal::LearningProgress).unwrap(),
            RegionReading::Value(1.0)
        );
        assert_eq!(
            r.read(0, DriveSignal::RawError).unwrap(),
            RegionReading::Value(0.0)
        );
    }

    #[test]
    fn constant_noise_has_error_but_no_progress() {
        let mut r = reader(3);
        for _ in 0..50 {
            r.record(0, 1.5).unwrap();
        }
        assert_eq!(
            r.read(0, DriveSignal::LearningProgress).unwrap(),
            RegionReading::Value(0.0)
        );
        assert_eq!(
            r.read(0, DriveSignal::RawError).unwrap(),
            RegionReading::Value(0.75)
        );
    }

    #[test]
    fn rising_loss_gives_zero_progress_not_negative() {
        let mut r = reader(1);
        r.record(0, 0.0).unwrap();
        r.record(0, 2.0).unwrap();
        assert_eq!(
            r.read(0, DriveSignal::LearningProgress).unwrap(),
            RegionReading::Value(0.0)
        );
    }

    #[test]
    fn window_slides_and_is_deterministic() {
        let mut a = reader(1);
        let mut b = reader(1);
        for l in [2.0, 1.0, 0.0, 0.0, 0.0] {
            a.record(0, l).unwrap();
            b.record(0, l).unwrap();
        }
        assert_eq!(a.samples(0), 2);
        assert_eq!(a, b);
        assert_eq!(
            a.read(0, DriveSignal::LearningProgress).unwrap(),
            RegionReading::Value(0.0)
        );
        assert_eq!(a.mean_reading(DriveSignal::RawError), Some((0.0, 1)));
        assert_eq!(a.max_reading(DriveSignal::RawError), Some(0.0));
    }

    #[test]
    fn reading_feeds_homeostatic_deficit() {
        let id = DriveId::new("learning_progress").unwrap();
        let spec = signal_drive_spec(id.clone(), 0.0, 1.0).unwrap();
        let mut st = HomeostaticState::new(vec![spec]).unwrap();
        let mut r = reader(1);
        r.record(0, 2.0).unwrap();
        r.record(0, 1.0).unwrap();
        let v = r
            .read(0, DriveSignal::LearningProgress)
            .unwrap()
            .value_or(0.0);
        st.set_reading(&id, v).unwrap();
        assert!((st.deficit(&id).unwrap() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn situated_modality() {
        assert_eq!(
            DriveModality::of_signal(DriveSignal::LearningProgress),
            DriveModality::Situated
        );
        assert_eq!(DriveSignal::RawError.label(), "raw_error");
    }
}
