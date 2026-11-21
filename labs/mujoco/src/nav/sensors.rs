//! Seeded odometry (noise, scale error, bias drift) and a 2D range sensor with latency.

use std::collections::VecDeque;

use super::geometry::World;
use crate::control::Rng;

/// Odometry error model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OdomConfig {
    /// White noise on measured forward speed (m/s, 1 sigma).
    pub v_sigma: f64,
    /// White noise on measured yaw rate (rad/s, 1 sigma).
    pub w_sigma: f64,
    /// Multiplicative scale error on forward speed (1.0 = perfect wheels).
    pub v_scale: f64,
    /// Initial yaw-rate bias (rad/s).
    pub w_bias0: f64,
    /// Random-walk of the yaw-rate bias (rad/s per sqrt(s)).
    pub w_bias_walk: f64,
}

impl Default for OdomConfig {
    fn default() -> Self {
        Self {
            v_sigma: 0.02,
            w_sigma: 0.01,
            v_scale: 1.0,
            w_bias0: 0.0,
            w_bias_walk: 0.002,
        }
    }
}

/// Dead-reckoning odometry. Pose integrates the *measured* twist, so errors accumulate.
#[derive(Debug, Clone)]
pub struct Odometry {
    cfg: OdomConfig,
    rng: Rng,
    w_bias: f64,
    pose: [f64; 3],
    twist: (f64, f64),
}

impl Odometry {
    /// New odometry starting at `pose` (the robot is assumed to know its start).
    pub fn new(cfg: OdomConfig, seed: u64, pose: [f64; 3]) -> Self {
        Self {
            cfg,
            rng: Rng::new(seed),
            w_bias: cfg.w_bias0,
            pose,
            twist: (0.0, 0.0),
        }
    }

    /// Consume the true body twist for one step of `dt` and update the dead-reckoned pose.
    pub fn update(&mut self, true_v: f64, true_w: f64, dt: f64) {
        self.w_bias += self.cfg.w_bias_walk * dt.sqrt() * self.rng.normal();
        let v = true_v * self.cfg.v_scale + self.cfg.v_sigma * self.rng.normal();
        let w = true_w + self.w_bias + self.cfg.w_sigma * self.rng.normal();
        self.twist = (v, w);
        let (s, c) = self.pose[2].sin_cos();
        self.pose[0] += v * c * dt;
        self.pose[1] += v * s * dt;
        self.pose[2] += w * dt;
    }

    /// Dead-reckoned pose `[x, y, yaw]`.
    pub fn pose(&self) -> [f64; 3] {
        self.pose
    }

    /// Last measured `(v, w)`.
    pub fn twist(&self) -> (f64, f64) {
        self.twist
    }
}

/// Range sensor parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeConfig {
    /// Number of rays (>= 2), evenly spread over `fov`, centred on the heading.
    pub n_rays: usize,
    /// Field of view (rad).
    pub fov: f64,
    /// Maximum range (m); a ray with no hit reports exactly this value.
    pub max_range: f64,
    /// Range noise (m, 1 sigma).
    pub sigma: f64,
    /// Ticks between scans.
    pub period_ticks: u64,
    /// Delivery latency in ticks (scan sampled at tick `s` is visible at tick `s + latency`).
    pub latency_ticks: u64,
    /// If set, no scan is delivered at or after this tick (sensor dropout).
    pub dropout_from_tick: Option<u64>,
}

impl Default for RangeConfig {
    fn default() -> Self {
        Self {
            n_rays: 21,
            fov: 210.0_f64.to_radians(),
            max_range: 3.0,
            sigma: 0.02,
            period_ticks: 5,
            latency_ticks: 0,
            dropout_from_tick: None,
        }
    }
}

impl RangeConfig {
    /// Ray bearings relative to the heading (rad).
    pub fn angles(&self) -> Vec<f64> {
        (0..self.n_rays)
            .map(|i| -self.fov / 2.0 + self.fov * i as f64 / (self.n_rays - 1) as f64)
            .collect()
    }
}

/// One scan: per-ray ranges stamped with the tick at which it was *sampled*.
#[derive(Debug, Clone, PartialEq)]
pub struct Scan {
    /// Tick at which the scan was physically sampled.
    pub sample_tick: u64,
    /// Ranges (m), `max_range` for no hit.
    pub ranges: Vec<f64>,
}

/// Range sensor with a FIFO latency line.
#[derive(Debug, Clone)]
pub struct RangeSensor {
    cfg: RangeConfig,
    angles: Vec<f64>,
    rng: Rng,
    line: VecDeque<(u64, Scan)>,
}

impl RangeSensor {
    /// New sensor.
    pub fn new(cfg: RangeConfig, seed: u64) -> Self {
        Self {
            angles: cfg.angles(),
            cfg,
            rng: Rng::new(seed),
            line: VecDeque::new(),
        }
    }

    /// Configuration.
    pub fn config(&self) -> &RangeConfig {
        &self.cfg
    }

    /// Ray bearings.
    pub fn angles(&self) -> &[f64] {
        &self.angles
    }

    /// Advance to `tick`: sample (if due) against the world at `time` from the *true* `pose`,
    /// and return the scan whose latency elapsed this tick, if any.
    pub fn tick(&mut self, tick: u64, time: f64, pose: [f64; 3], world: &World) -> Option<Scan> {
        let dropped = self.cfg.dropout_from_tick.is_some_and(|d| tick >= d);
        if !dropped && tick.is_multiple_of(self.cfg.period_ticks) {
            let ranges =
                self.angles
                    .iter()
                    .map(|a| {
                        let th = pose[2] + a;
                        let hit = world.cast(
                            [pose[0], pose[1]],
                            [th.cos(), th.sin()],
                            self.cfg.max_range,
                            time,
                        );
                        match hit {
                            Some(h) => (h + self.cfg.sigma * self.rng.normal())
                                .clamp(0.0, self.cfg.max_range),
                            None => self.cfg.max_range,
                        }
                    })
                    .collect();
            self.line.push_back((
                tick + self.cfg.latency_ticks,
                Scan {
                    sample_tick: tick,
                    ranges,
                },
            ));
        }
        if dropped {
            self.line.clear();
            return None;
        }
        match self.line.front() {
            Some((due, _)) if *due <= tick => self.line.pop_front().map(|(_, s)| s),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::geometry::Shape;

    fn world() -> World {
        World {
            statics: vec![Shape::Aabb {
                min: [2.0, -5.0],
                max: [3.0, 5.0],
            }],
            movers: vec![],
        }
    }

    #[test]
    fn latency_delays_delivery_and_dropout_stops_it() {
        let cfg = RangeConfig {
            latency_ticks: 7,
            sigma: 0.0,
            dropout_from_tick: Some(30),
            ..Default::default()
        };
        let mut s = RangeSensor::new(cfg, 1);
        let mut got = vec![];
        for t in 0..40 {
            if let Some(sc) = s.tick(t, t as f64 * 0.01, [0.0, 0.0, 0.0], &world()) {
                got.push((t, sc.sample_tick));
            }
        }
        assert_eq!(got[0], (7, 0));
        assert!(got.iter().all(|(t, s)| t - s == 7));
        assert!(got.iter().all(|(_, s)| *s < 30));
        let mid = RangeSensor::new(cfg, 1);
        let centre = mid.angles().len() / 2;
        let mut s = RangeSensor::new(
            RangeConfig {
                latency_ticks: 0,
                sigma: 0.0,
                ..cfg
            },
            1,
        );
        let sc = s.tick(0, 0.0, [0.0, 0.0, 0.0], &world()).unwrap();
        assert!((sc.ranges[centre] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn odometry_is_seeded_and_drifts() {
        let run = |seed| {
            let mut o = Odometry::new(
                OdomConfig {
                    v_scale: 1.05,
                    w_bias0: 0.02,
                    ..Default::default()
                },
                seed,
                [0.0; 3],
            );
            for _ in 0..500 {
                o.update(1.0, 0.0, 0.01);
            }
            o.pose()
        };
        assert_eq!(run(3).map(f64::to_bits), run(3).map(f64::to_bits));
        assert_ne!(run(3), run(4));
        let p = run(3);
        assert!(p[2] > 0.05 && p[0] > 5.0, "{p:?}");
    }
}
