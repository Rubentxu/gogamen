//! 2D obstacle geometry: circles, AABBs, deterministic moving obstacles and ray casting.

/// A static 2D shape.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape {
    /// Disc.
    Circle {
        /// Centre `[x, y]`.
        c: [f64; 2],
        /// Radius (m).
        r: f64,
    },
    /// Axis-aligned box.
    Aabb {
        /// Minimum corner.
        min: [f64; 2],
        /// Maximum corner.
        max: [f64; 2],
    },
}

impl Shape {
    /// Distance along unit direction `d` from `o` to the first surface hit (`0` if `o` is inside).
    pub fn ray(&self, o: [f64; 2], d: [f64; 2]) -> Option<f64> {
        match *self {
            Shape::Circle { c, r } => {
                let oc = [o[0] - c[0], o[1] - c[1]];
                let b = oc[0] * d[0] + oc[1] * d[1];
                let cc = oc[0] * oc[0] + oc[1] * oc[1] - r * r;
                if cc <= 0.0 {
                    return Some(0.0);
                }
                let disc = b * b - cc;
                if disc < 0.0 {
                    return None;
                }
                let t = -b - disc.sqrt();
                (t >= 0.0).then_some(t)
            }
            Shape::Aabb { min, max } => {
                let (mut t0, mut t1) = (0.0_f64, f64::INFINITY);
                for i in 0..2 {
                    if d[i].abs() < 1e-12 {
                        if o[i] < min[i] || o[i] > max[i] {
                            return None;
                        }
                    } else {
                        let (a, b) = ((min[i] - o[i]) / d[i], (max[i] - o[i]) / d[i]);
                        let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                        t0 = t0.max(lo);
                        t1 = t1.min(hi);
                        if t0 > t1 {
                            return None;
                        }
                    }
                }
                Some(t0)
            }
        }
    }

    /// Signed distance from point `p` to the surface (negative inside).
    pub fn distance(&self, p: [f64; 2]) -> f64 {
        match *self {
            Shape::Circle { c, r } => (p[0] - c[0]).hypot(p[1] - c[1]) - r,
            Shape::Aabb { min, max } => {
                let cx = (min[0] + max[0]) / 2.0;
                let cy = (min[1] + max[1]) / 2.0;
                let (hx, hy) = ((max[0] - min[0]) / 2.0, (max[1] - min[1]) / 2.0);
                let q = [(p[0] - cx).abs() - hx, (p[1] - cy).abs() - hy];
                q[0].max(0.0).hypot(q[1].max(0.0)) + q[0].max(q[1]).min(0.0)
            }
        }
    }
}

/// A circular obstacle moving back and forth on a segment at constant speed.
/// Its position is a pure function of time, so it is deterministic and needs no snapshot state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingObstacle {
    /// Segment start.
    pub a: [f64; 2],
    /// Segment end.
    pub b: [f64; 2],
    /// Speed (m/s).
    pub speed: f64,
    /// Initial offset along the path (m).
    pub phase: f64,
    /// Radius (m).
    pub radius: f64,
}

impl MovingObstacle {
    /// Centre at time `t` (s).
    pub fn center_at(&self, t: f64) -> [f64; 2] {
        let l = (self.b[0] - self.a[0]).hypot(self.b[1] - self.a[1]);
        if l < 1e-12 {
            return self.a;
        }
        let s = (self.speed * t + self.phase).rem_euclid(2.0 * l);
        let d = if s < l { s } else { 2.0 * l - s } / l;
        [
            self.a[0] + (self.b[0] - self.a[0]) * d,
            self.a[1] + (self.b[1] - self.a[1]) * d,
        ]
    }

    /// Shape at time `t`.
    pub fn shape_at(&self, t: f64) -> Shape {
        Shape::Circle {
            c: self.center_at(t),
            r: self.radius,
        }
    }
}

/// Static obstacles plus moving obstacles.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct World {
    /// Static obstacles (also the "known map" used by the fusion position fix).
    pub statics: Vec<Shape>,
    /// Moving obstacles (unknown to the fusion map).
    pub movers: Vec<MovingObstacle>,
}

impl World {
    /// First hit over static and moving obstacles at time `t`, capped at `max_range`.
    pub fn cast(&self, o: [f64; 2], d: [f64; 2], max_range: f64, t: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        let all = self
            .statics
            .iter()
            .copied()
            .chain(self.movers.iter().map(|m| m.shape_at(t)));
        for s in all {
            if let Some(h) = s.ray(o, d)
                && h <= max_range
                && best.is_none_or(|b| h < b)
            {
                best = Some(h);
            }
        }
        best
    }

    /// Cast against static obstacles only (the known map).
    pub fn cast_static(&self, o: [f64; 2], d: [f64; 2], max_range: f64) -> Option<f64> {
        self.statics
            .iter()
            .filter_map(|s| s.ray(o, d))
            .filter(|h| *h <= max_range)
            .fold(None, |b: Option<f64>, h| Some(b.map_or(h, |x| x.min(h))))
    }

    /// Minimum signed distance from point `p` to any obstacle surface at time `t`.
    pub fn min_distance(&self, p: [f64; 2], t: f64) -> f64 {
        let st = self.statics.iter().map(|s| s.distance(p));
        let mv = self.movers.iter().map(|m| m.shape_at(t).distance(p));
        st.chain(mv).fold(f64::INFINITY, f64::min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rays_hit_circle_and_aabb() {
        let c = Shape::Circle {
            c: [3.0, 0.0],
            r: 1.0,
        };
        assert!((c.ray([0.0, 0.0], [1.0, 0.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!(c.ray([0.0, 0.0], [0.0, 1.0]).is_none());
        assert!(c.ray([0.0, 0.0], [-1.0, 0.0]).is_none());
        let b = Shape::Aabb {
            min: [2.0, -1.0],
            max: [4.0, 1.0],
        };
        assert!((b.ray([0.0, 0.0], [1.0, 0.0]).unwrap() - 2.0).abs() < 1e-12);
        assert!(b.ray([0.0, 2.0], [1.0, 0.0]).is_none());
        assert_eq!(b.ray([3.0, 0.0], [1.0, 0.0]), Some(0.0));
        assert!((b.distance([0.0, 0.0]) - 2.0).abs() < 1e-12);
        assert!(b.distance([3.0, 0.0]) < 0.0);
    }

    #[test]
    fn mover_ping_pongs_deterministically() {
        let m = MovingObstacle {
            a: [0.0, 0.0],
            b: [0.0, 2.0],
            speed: 1.0,
            phase: 0.0,
            radius: 0.1,
        };
        assert_eq!(m.center_at(1.0), [0.0, 1.0]);
        assert_eq!(m.center_at(2.0), [0.0, 2.0]);
        assert_eq!(m.center_at(3.0), [0.0, 1.0]);
        assert_eq!(m.center_at(4.0), [0.0, 0.0]);
        assert_eq!(
            m.center_at(1.234).map(f64::to_bits),
            m.center_at(1.234).map(f64::to_bits)
        );
    }
}
