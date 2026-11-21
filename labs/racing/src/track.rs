//! Pista 2D con muros, límites de contorno y checkpoints ordenados (sin renderer ni UI).

use gogamen_core::DomainError;

use crate::physics::KartState;

/// Punto 2D en coordenadas del mundo (metros).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    /// Coordenada X en metros.
    pub x: f64,
    /// Coordenada Y en metros.
    pub y: f64,
}

impl Point2 {
    /// Construye un punto 2D validando que sus coordenadas sean finitas.
    pub fn new(x: f64, y: f64) -> Result<Self, DomainError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(DomainError::InvalidTransition {
                reason: "point coordinates must be finite",
            });
        }
        Ok(Self { x, y })
    }

    /// Distancia euclídea entre dos puntos.
    pub fn distance_to(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Distancia mínima de un punto `p` al segmento `[a, b]`.
pub fn point_to_segment_distance(p: Point2, a: Point2, b: Point2) -> f64 {
    let abx = b.x - a.x;
    let aby = b.y - a.y;
    let len_sq = abx * abx + aby * aby;
    if len_sq <= f64::EPSILON {
        return p.distance_to(a);
    }
    let apx = p.x - a.x;
    let apy = p.y - a.y;
    let t = ((apx * abx + apy * aby) / len_sq).clamp(0.0, 1.0);
    let proj = Point2 {
        x: a.x + t * abx,
        y: a.y + t * aby,
    };
    p.distance_to(proj)
}

fn orient(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn on_segment(a: Point2, b: Point2, p: Point2) -> bool {
    p.x >= a.x.min(b.x) - 1e-12
        && p.x <= a.x.max(b.x) + 1e-12
        && p.y >= a.y.min(b.y) - 1e-12
        && p.y <= a.y.max(b.y) + 1e-12
}

/// Devuelve `true` si los segmentos `[a, b]` y `[c, d]` se intersectan.
pub fn segments_intersect(a: Point2, b: Point2, c: Point2, d: Point2) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);

    if (o1 > 0.0 && o2 < 0.0 || o1 < 0.0 && o2 > 0.0)
        && (o3 > 0.0 && o4 < 0.0 || o3 < 0.0 && o4 > 0.0)
    {
        return true;
    }

    (o1.abs() <= 1e-12 && on_segment(a, b, c))
        || (o2.abs() <= 1e-12 && on_segment(a, b, d))
        || (o3.abs() <= 1e-12 && on_segment(c, d, a))
        || (o4.abs() <= 1e-12 && on_segment(c, d, b))
}

/// Segmento de muro sólido 2D en la pista.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WallSegment {
    /// Extremo inicial del muro.
    pub start: Point2,
    /// Extremo final del muro.
    pub end: Point2,
}

impl WallSegment {
    /// Construye un segmento de muro validando sus extremos.
    pub fn new(start: Point2, end: Point2) -> Result<Self, DomainError> {
        if !start.x.is_finite() || !start.y.is_finite() || !end.x.is_finite() || !end.y.is_finite()
        {
            return Err(DomainError::InvalidTransition {
                reason: "wall segment endpoints must be finite",
            });
        }
        Ok(Self { start, end })
    }

    /// Distancia mínima del muro al punto `p`.
    pub fn distance_to_point(self, p: Point2) -> f64 {
        point_to_segment_distance(p, self.start, self.end)
    }

    /// Comprueba si el barrido del kart `[prev, next]` con radio `radius` colisiona con este muro.
    pub fn collides_with_sweep(self, prev: Point2, next: Point2, radius: f64) -> bool {
        if segments_intersect(prev, next, self.start, self.end) {
            return true;
        }
        let d_prev = self.distance_to_point(prev);
        let d_next = self.distance_to_point(next);
        let d_start = point_to_segment_distance(self.start, prev, next);
        let d_end = point_to_segment_distance(self.end, prev, next);
        d_prev.min(d_next).min(d_start).min(d_end) <= radius
    }
}

/// Checkpoint circular ordenado dentro de la pista.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackCheckpoint {
    /// Índice secuencial del checkpoint (`0..N`).
    pub index: usize,
    /// Centro del checkpoint en coordenadas del mundo.
    pub center: Point2,
    /// Radio de captura en metros (`> 0.0`).
    pub radius: f64,
}

impl TrackCheckpoint {
    /// Construye un checkpoint validando su centro y radio.
    pub fn new(index: usize, center: Point2, radius: f64) -> Result<Self, DomainError> {
        if !center.x.is_finite() || !center.y.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return Err(DomainError::InvalidTransition {
                reason: "checkpoint center must be finite and radius > 0",
            });
        }
        Ok(Self {
            index,
            center,
            radius,
        })
    }

    /// Devuelve `true` si el desplazamiento `[prev, next]` atraviesa o entra en el checkpoint.
    pub fn cleared_by_sweep(self, prev: Point2, next: Point2) -> bool {
        point_to_segment_distance(self.center, prev, next) <= self.radius
    }
}

/// Mapa de pista 2D con muros, caja límite exterior y secuencia de checkpoints.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackMap {
    name: String,
    start_pose: KartState,
    bounds_min: Point2,
    bounds_max: Point2,
    walls: Vec<WallSegment>,
    checkpoints: Vec<TrackCheckpoint>,
}

impl TrackMap {
    /// Construye un [`TrackMap`] validando límites, muros y checkpoints.
    pub fn new(
        name: impl Into<String>,
        start_pose: KartState,
        bounds_min: Point2,
        bounds_max: Point2,
        walls: Vec<WallSegment>,
        checkpoints: Vec<TrackCheckpoint>,
    ) -> Result<Self, DomainError> {
        if !bounds_min.x.is_finite()
            || !bounds_min.y.is_finite()
            || !bounds_max.x.is_finite()
            || !bounds_max.y.is_finite()
            || bounds_min.x >= bounds_max.x
            || bounds_min.y >= bounds_max.y
        {
            return Err(DomainError::InvalidTransition {
                reason: "track bounds must be finite with bounds_min < bounds_max",
            });
        }
        if checkpoints.is_empty() {
            return Err(DomainError::InvalidTransition {
                reason: "track must define at least one checkpoint",
            });
        }
        for (expected_idx, cp) in checkpoints.iter().enumerate() {
            if cp.index != expected_idx {
                return Err(DomainError::InvalidTransition {
                    reason: "track checkpoints must be indexed sequentially 0..N",
                });
            }
        }
        let map = Self {
            name: name.into(),
            start_pose,
            bounds_min,
            bounds_max,
            walls,
            checkpoints,
        };
        let start_pt = Point2 {
            x: start_pose.x,
            y: start_pose.y,
        };
        if map.is_outside_bounds(start_pt, 0.0) {
            return Err(DomainError::InvalidTransition {
                reason: "start_pose must lie inside track bounds",
            });
        }
        Ok(map)
    }

    /// Circuito rectangular cerrado por defecto (`[0, 40] x [0, 24]`) con isla central (`[10, 30] x [8, 16]`).
    ///
    /// Incluye 5 checkpoints en sentido antihorario para completar una vuelta completa.
    pub fn rectangular_circuit() -> Self {
        let p = |x: f64, y: f64| Point2 { x, y };
        let walls = vec![
            // Perímetro exterior
            WallSegment {
                start: p(0.0, 0.0),
                end: p(40.0, 0.0),
            },
            WallSegment {
                start: p(40.0, 0.0),
                end: p(40.0, 24.0),
            },
            WallSegment {
                start: p(40.0, 24.0),
                end: p(0.0, 24.0),
            },
            WallSegment {
                start: p(0.0, 24.0),
                end: p(0.0, 0.0),
            },
            // Isla interior
            WallSegment {
                start: p(10.0, 8.0),
                end: p(30.0, 8.0),
            },
            WallSegment {
                start: p(30.0, 8.0),
                end: p(30.0, 16.0),
            },
            WallSegment {
                start: p(30.0, 16.0),
                end: p(10.0, 16.0),
            },
            WallSegment {
                start: p(10.0, 16.0),
                end: p(10.0, 8.0),
            },
        ];

        let checkpoints = vec![
            TrackCheckpoint {
                index: 0,
                center: p(35.0, 4.0),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 1,
                center: p(35.0, 20.0),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 2,
                center: p(5.0, 20.0),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 3,
                center: p(5.0, 4.0),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 4,
                center: p(14.0, 4.0),
                radius: 3.0,
            },
        ];

        Self {
            name: "rectangular-circuit".to_string(),
            start_pose: KartState {
                x: 12.0,
                y: 4.0,
                yaw: 0.0,
                speed: 0.0,
            },
            bounds_min: p(0.0, 0.0),
            bounds_max: p(40.0, 24.0),
            walls,
            checkpoints,
        }
    }

    /// Circuito no visto con chicane (`[0, 48] x [0, 30]`), isla central y dos deflectores en S.
    ///
    /// Incluye 7 checkpoints en sentido antihorario para evaluar generalización en curvas cerradas.
    pub fn chicane_circuit() -> Self {
        let p = |x: f64, y: f64| Point2 { x, y };
        let walls = vec![
            // Perímetro exterior
            WallSegment {
                start: p(0.0, 0.0),
                end: p(48.0, 0.0),
            },
            WallSegment {
                start: p(48.0, 0.0),
                end: p(48.0, 30.0),
            },
            WallSegment {
                start: p(48.0, 30.0),
                end: p(0.0, 30.0),
            },
            WallSegment {
                start: p(0.0, 30.0),
                end: p(0.0, 0.0),
            },
            // Isla interior
            WallSegment {
                start: p(10.0, 9.0),
                end: p(38.0, 9.0),
            },
            WallSegment {
                start: p(38.0, 9.0),
                end: p(38.0, 19.0),
            },
            WallSegment {
                start: p(38.0, 19.0),
                end: p(10.0, 19.0),
            },
            WallSegment {
                start: p(10.0, 19.0),
                end: p(10.0, 9.0),
            },
            // Deflectores de chicane en la recta superior
            WallSegment {
                start: p(28.0, 30.0),
                end: p(28.0, 24.0),
            },
            WallSegment {
                start: p(20.0, 19.0),
                end: p(20.0, 25.0),
            },
        ];

        let checkpoints = vec![
            TrackCheckpoint {
                index: 0,
                center: p(43.0, 4.5),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 1,
                center: p(43.0, 24.5),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 2,
                center: p(28.0, 21.5),
                radius: 2.5,
            },
            TrackCheckpoint {
                index: 3,
                center: p(20.0, 27.5),
                radius: 2.5,
            },
            TrackCheckpoint {
                index: 4,
                center: p(5.0, 24.5),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 5,
                center: p(5.0, 4.5),
                radius: 3.0,
            },
            TrackCheckpoint {
                index: 6,
                center: p(14.0, 4.5),
                radius: 3.0,
            },
        ];

        Self {
            name: "chicane-circuit".to_string(),
            start_pose: KartState {
                x: 12.0,
                y: 4.5,
                yaw: 0.0,
                speed: 0.0,
            },
            bounds_min: p(0.0, 0.0),
            bounds_max: p(48.0, 30.0),
            walls,
            checkpoints,
        }
    }

    /// Nombre identificador de la pista.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Pose inicial del kart en la pista.
    pub const fn start_pose(&self) -> KartState {
        self.start_pose
    }

    /// Límite inferior `(min_x, min_y)` de la pista.
    pub const fn bounds_min(&self) -> Point2 {
        self.bounds_min
    }

    /// Límite superior `(max_x, max_y)` de la pista.
    pub const fn bounds_max(&self) -> Point2 {
        self.bounds_max
    }

    /// Lista de segmentos de muro de la pista.
    pub fn walls(&self) -> &[WallSegment] {
        &self.walls
    }

    /// Secuencia ordenada de checkpoints de la pista.
    pub fn checkpoints(&self) -> &[TrackCheckpoint] {
        &self.checkpoints
    }

    /// Comprueba si un punto está fuera de la caja límite exterior considerando `margin`.
    pub fn is_outside_bounds(&self, pt: Point2, margin: f64) -> bool {
        pt.x - margin < self.bounds_min.x
            || pt.x + margin > self.bounds_max.x
            || pt.y - margin < self.bounds_min.y
            || pt.y + margin > self.bounds_max.y
    }

    /// Comprueba si el barrido `[prev, next]` con radio `kart_radius` colisiona con muros o límites.
    pub fn check_collision(&self, prev: Point2, next: Point2, kart_radius: f64) -> bool {
        if self.is_outside_bounds(prev, kart_radius) || self.is_outside_bounds(next, kart_radius) {
            return true;
        }
        self.walls
            .iter()
            .any(|wall| wall.collides_with_sweep(prev, next, kart_radius))
    }

    /// Lanza un rayo desde `origin` con ángulo absoluto `angle` (rad) contra los muros.
    ///
    /// Devuelve `Some(distancia)` al primer muro dentro de `max_range` o `None` si no hay
    /// impacto (lectura de sensor ideal, sin ruido).
    pub fn cast_ray(&self, origin: Point2, angle: f64, max_range: f64) -> Option<f64> {
        let (dx, dy) = (angle.cos(), angle.sin());
        let mut best: Option<f64> = None;
        for wall in &self.walls {
            let (sx, sy) = (wall.end.x - wall.start.x, wall.end.y - wall.start.y);
            let denom = dx * sy - dy * sx;
            if denom.abs() < 1e-12 {
                continue;
            }
            let (ox, oy) = (wall.start.x - origin.x, wall.start.y - origin.y);
            let t = (ox * sy - oy * sx) / denom;
            let u = (ox * dy - oy * dx) / denom;
            if t >= 0.0 && t <= max_range && (0.0..=1.0).contains(&u) && best.is_none_or(|b| t < b)
            {
                best = Some(t);
            }
        }
        best
    }
}

#[cfg(test)]
mod ray_tests {
    use super::*;

    #[test]
    fn cast_ray_hits_nearest_wall_and_respects_range() {
        let track = TrackMap::rectangular_circuit();
        let o = Point2 { x: 5.0, y: 12.0 };
        // hacia -x: muro exterior en x = 0
        let d = track.cast_ray(o, std::f64::consts::PI, 50.0).unwrap();
        assert!((d - 5.0).abs() < 1e-9, "{d}");
        // hacia +x: isla central en x = 10
        let d = track.cast_ray(o, 0.0, 50.0).unwrap();
        assert!((d - 5.0).abs() < 1e-9, "{d}");
        assert!(track.cast_ray(o, 0.0, 2.0).is_none());
    }
}
