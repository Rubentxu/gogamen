//! Minimal odometry + range fusion (complementary filter).
//!
//! * Pose: `odometry pose + offset`; the offset is corrected by *position fixes*: every scan ray that
//!   hits the known static map within a gate nudges the pose along the ray by the (expected -
//!   measured) range, scaled by a gain. The scan is applied against the pose history at its sample
//!   tick, so latency is respected. Heading is not corrected (caveat).
//! * Ranges: per-ray fused range is predicted forward with odometry speed each tick and blended
//!   with every delivered scan after compensating its age by ego motion (static-scene assumption).

use std::collections::VecDeque;

use super::geometry::World;
use super::sensors::Scan;

/// Fusion parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionConfig {
    /// Blend factor of a new measurement into the fused range (0..1).
    pub range_alpha: f64,
    /// Gain of the map-based position fix (0 disables it).
    pub fix_gain: f64,
    /// Innovation gate (m): fixes with |expected - measured| above this are ignored (moving obstacles).
    pub fix_gate: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            range_alpha: 0.6,
            fix_gain: 0.3,
            fix_gate: 0.3,
        }
    }
}

/// Fusion state.
#[derive(Debug, Clone)]
pub struct Fusion {
    cfg: FusionConfig,
    angles: Vec<f64>,
    max_range: f64,
    offset: [f64; 2],
    odom_pose: [f64; 3],
    hist: VecDeque<[f64; 3]>,
    hist_tick0: u64,
    ranges: Vec<f64>,
    dynamic: Vec<bool>,
    last_scan_tick: Option<u64>,
    fixes_applied: u64,
}

const HIST_CAP: usize = 2048;

impl Fusion {
    /// New filter for a sensor with the given ray bearings and maximum range.
    pub fn new(cfg: FusionConfig, angles: &[f64], max_range: f64, pose: [f64; 3]) -> Self {
        Self {
            cfg,
            angles: angles.to_vec(),
            max_range,
            offset: [0.0; 2],
            odom_pose: pose,
            hist: VecDeque::new(),
            hist_tick0: 0,
            ranges: vec![max_range; angles.len()],
            dynamic: vec![false; angles.len()],
            last_scan_tick: None,
            fixes_applied: 0,
        }
    }

    /// Fused pose estimate.
    pub fn pose(&self) -> [f64; 3] {
        [
            self.odom_pose[0] + self.offset[0],
            self.odom_pose[1] + self.offset[1],
            self.odom_pose[2],
        ]
    }

    /// Fused per-ray ranges (valid only once a scan has arrived).
    pub fn ranges(&self) -> &[f64] {
        &self.ranges
    }

    /// Ray bearings.
    pub fn angles(&self) -> &[f64] {
        &self.angles
    }

    /// Maximum sensor range.
    pub fn max_range(&self) -> f64 {
        self.max_range
    }

    /// Number of applied map position fixes.
    pub fn fixes_applied(&self) -> u64 {
        self.fixes_applied
    }

    /// Age in ticks of the newest delivered scan (`None` before the first one).
    pub fn scan_age(&self, tick: u64) -> Option<u64> {
        self.last_scan_tick.map(|s| tick.saturating_sub(s))
    }

    /// Per-tick prediction from the odometry pose and measured forward speed. `tick` is the current tick.
    pub fn predict(&mut self, tick: u64, odom_pose: [f64; 3], v: f64, dt: f64) {
        self.odom_pose = odom_pose;
        for (r, a) in self.ranges.iter_mut().zip(&self.angles) {
            if *r < self.max_range {
                *r = (*r - v * a.cos() * dt).max(0.0);
            }
        }
        if self.hist.is_empty() {
            self.hist_tick0 = tick;
        }
        self.hist.push_back(self.pose());
        if self.hist.len() > HIST_CAP {
            self.hist.pop_front();
            self.hist_tick0 += 1;
        }
    }

    fn pose_at(&self, tick: u64) -> Option<[f64; 3]> {
        let i = tick.checked_sub(self.hist_tick0)? as usize;
        self.hist.get(i).copied()
    }

    /// Fuse a delivered scan at current tick `tick` with ego speed `v` and tick length `dt`.
    pub fn on_scan(&mut self, tick: u64, scan: &Scan, map: &World, v: f64, dt: f64) {
        let age_s = tick.saturating_sub(scan.sample_tick) as f64 * dt;
        // Position fix against the known static map, using the pose at the sample tick.
        if self.cfg.fix_gain > 0.0
            && let Some(p) = self.pose_at(scan.sample_tick)
        {
            let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
            for (a, m) in self.angles.iter().zip(&scan.ranges) {
                if *m >= self.max_range - 1e-9 {
                    continue;
                }
                let th = p[2] + a;
                let d = [th.cos(), th.sin()];
                if let Some(e) = map.cast_static([p[0], p[1]], d, self.max_range) {
                    let innov = e - m;
                    if innov.abs() <= self.cfg.fix_gate {
                        sx += innov * d[0];
                        sy += innov * d[1];
                        n += 1;
                    }
                }
            }
            if n > 0 {
                self.offset[0] += self.cfg.fix_gain * sx / n as f64;
                self.offset[1] += self.cfg.fix_gain * sy / n as f64;
                self.fixes_applied += 1;
            }
        }
        // Classify each ray: a hit that the known static map does not explain (nearer than the
        // expected static range, or no static surface expected) is an unknown, possibly moving obstacle.
        let pose = self.pose_at(scan.sample_tick);
        for ((d, a), m) in self.dynamic.iter_mut().zip(&self.angles).zip(&scan.ranges) {
            *d = if *m >= self.max_range - 1e-9 {
                false
            } else if let Some(p) = pose {
                let th = p[2] + a;
                let e = map.cast_static([p[0], p[1]], [th.cos(), th.sin()], self.max_range);
                e.is_none_or(|e| *m < e - self.cfg.fix_gate)
            } else {
                true
            };
        }
        // Range blending with ego-motion compensation of the scan age.
        for ((r, a), m) in self.ranges.iter_mut().zip(&self.angles).zip(&scan.ranges) {
            if *m >= self.max_range - 1e-9 {
                *r = self.max_range;
                continue;
            }
            let proj = (*m - v * a.cos() * age_s).max(0.0);
            *r = if *r >= self.max_range {
                proj
            } else {
                *r + self.cfg.range_alpha * (proj - *r)
            };
        }
        self.last_scan_tick = Some(match self.last_scan_tick {
            Some(t) => t.max(scan.sample_tick),
            None => scan.sample_tick,
        });
    }

    /// Smallest forward clearance among rays whose hit point could enter the robot corridor.
    /// Only rays classified as dynamic (not explained by the known static map) get the lateral allowance.
    /// A hit at forward distance `fwd` is a threat when `|lateral| <= half_width + lat_rate * t`
    /// with `t = delay + min((fwd + pass_len) / v, t_cap)` (time the robot needs to get past it, capped; `lat_rate` is the
    /// assumed worst-case lateral speed of an unknown moving obstacle). `max_range` if none.
    /// Shortest fused range among rays classified as unknown/moving obstacles (`max_range` if none).
    pub fn nearest_dynamic(&self) -> f64 {
        self.ranges
            .iter()
            .zip(&self.dynamic)
            .filter(|(_, d)| **d)
            .map(|(r, _)| *r)
            .fold(self.max_range, f64::min)
    }

    pub fn forward_clearance(
        &self,
        half_width: f64,
        lat_rate: f64,
        delay: f64,
        v: f64,
        t_cap: f64,
        pass_len: f64,
    ) -> f64 {
        let mut best = self.max_range;
        for (i, (r, a)) in self.ranges.iter().zip(&self.angles).enumerate() {
            let (fwd, lat) = (r * a.cos(), r * a.sin());
            let rate = if self.dynamic[i] { lat_rate } else { 0.0 };
            let reach = half_width + rate * (delay + ((fwd + pass_len) / v.max(1e-3)).min(t_cap));
            if *r < self.max_range && fwd >= 0.0 && lat.abs() <= reach && fwd < best {
                best = fwd;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::geometry::Shape;
    use crate::nav::sensors::{RangeConfig, RangeSensor};

    #[test]
    fn map_fix_pulls_a_wrong_estimate_towards_truth() {
        let world = World {
            statics: vec![Shape::Aabb {
                min: [2.0, -5.0],
                max: [3.0, 5.0],
            }],
            movers: vec![],
        };
        let cfg = RangeConfig {
            sigma: 0.0,
            ..Default::default()
        };
        let mut sensor = RangeSensor::new(cfg, 1);
        // truth at x=1.0; estimate believes x=0.9 (within the gate).
        let mut f = Fusion::new(
            FusionConfig::default(),
            sensor.angles(),
            cfg.max_range,
            [0.9, 0.0, 0.0],
        );
        let mut err = 0.1_f64;
        for t in 0..400u64 {
            f.predict(t, [0.9, 0.0, 0.0], 0.0, 0.01);
            if let Some(sc) = sensor.tick(t, 0.0, [1.0, 0.0, 0.0], &world) {
                f.on_scan(t, &sc, &world, 0.0, 0.01);
                err = (f.pose()[0] - 1.0).abs();
            }
        }
        assert!(err < 0.01, "err {err}");
        assert!(f.fixes_applied() > 10);
    }

    #[test]
    fn stale_scan_is_projected_by_ego_motion() {
        let world = World::default();
        let mut f = Fusion::new(
            FusionConfig {
                fix_gain: 0.0,
                ..Default::default()
            },
            &[0.0],
            3.0,
            [0.0; 3],
        );
        f.predict(0, [0.0; 3], 1.0, 0.01);
        f.on_scan(
            20,
            &Scan {
                sample_tick: 0,
                ranges: vec![2.0],
            },
            &world,
            1.0,
            0.01,
        );
        assert!((f.ranges()[0] - 1.8).abs() < 1e-9);
        assert_eq!(f.scan_age(25), Some(25));
    }
}
