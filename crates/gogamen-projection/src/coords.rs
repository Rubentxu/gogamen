//! Coordenadas 2D, unidades explícitas y transformaciones de escala (`std`-only).

use std::fmt;

use crate::error::{ProjectionError, ensure_finite, ensure_non_negative, ensure_positive};

/// Punto bidimensional con coordenadas finitas garantizadas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    x: f64,
    y: f64,
}

impl Point2 {
    /// Origen `(0.0, 0.0)`.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// Crea un [`Point2`] validando que ambas coordenadas sean finitas.
    pub fn new(x: f64, y: f64) -> Result<Self, ProjectionError> {
        let x = ensure_finite("point2.x", x)?;
        let y = ensure_finite("point2.y", y)?;
        Ok(Self {
            x: normalize_zero(x),
            y: normalize_zero(y),
        })
    }

    /// Coordenada `x`.
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Coordenada `y`.
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Distancia euclídea hasta `other`.
    pub fn distance_to(self, other: Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx.hypot(dy)
    }
}

impl fmt::Display for Point2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

/// Unidades físicas o lógicas asociadas a un [`CoordinateSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleUnits {
    /// Metros (`1.0` m/unidad).
    Meters,
    /// Centímetros (`0.01` m/unidad).
    Centimeters,
    /// Milímetros (`0.001` m/unidad).
    Millimeters,
    /// Kilómetros (`1000.0` m/unidad).
    Kilometers,
    /// Celdas discretas de grilla (por defecto `1.0` m/celda salvo escala personalizada).
    GridCells,
    /// Unidad personalizada definida explícitamente por `meters_per_unit`.
    Custom,
}

impl ScaleUnits {
    /// Factor de conversión por defecto en metros por unidad.
    pub const fn default_meters_per_unit(self) -> f64 {
        match self {
            Self::Meters | Self::GridCells | Self::Custom => 1.0,
            Self::Centimeters => 0.01,
            Self::Millimeters => 0.001,
            Self::Kilometers => 1000.0,
        }
    }

    /// Nombre canónico estable para serialización determinista.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Meters => "meters",
            Self::Centimeters => "centimeters",
            Self::Millimeters => "millimeters",
            Self::Kilometers => "kilometers",
            Self::GridCells => "grid_cells",
            Self::Custom => "custom",
        }
    }
}

impl fmt::Display for ScaleUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Definición explícita de un espacio de coordenadas 2D y su escala respecto a metros del mundo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordinateSpace {
    units: ScaleUnits,
    meters_per_unit: f64,
    origin_meters: Point2,
    y_axis_up: bool,
}

impl CoordinateSpace {
    /// Espacio canónico en metros con origen `(0, 0)` y eje `Y` hacia arriba.
    pub const fn meters() -> Self {
        Self {
            units: ScaleUnits::Meters,
            meters_per_unit: 1.0,
            origin_meters: Point2::ZERO,
            y_axis_up: true,
        }
    }

    /// Construye un espacio estándar usando el factor por defecto de `units`.
    pub const fn from_units(units: ScaleUnits) -> Self {
        Self {
            units,
            meters_per_unit: units.default_meters_per_unit(),
            origin_meters: Point2::ZERO,
            y_axis_up: true,
        }
    }

    /// Construye un [`CoordinateSpace`] validando que `meters_per_unit` sea finito y `> 0.0`.
    pub fn new(
        units: ScaleUnits,
        meters_per_unit: f64,
        origin_meters: Point2,
        y_axis_up: bool,
    ) -> Result<Self, ProjectionError> {
        let meters_per_unit = ensure_positive("coordinate_space.meters_per_unit", meters_per_unit)?;
        Ok(Self {
            units,
            meters_per_unit,
            origin_meters,
            y_axis_up,
        })
    }

    /// Unidades declaradas del espacio.
    pub const fn units(&self) -> ScaleUnits {
        self.units
    }

    /// Metros físicos correspondientes a `1.0` unidad de este espacio.
    pub const fn meters_per_unit(&self) -> f64 {
        self.meters_per_unit
    }

    /// Origen de este espacio expresado en metros del mundo.
    pub const fn origin_meters(&self) -> Point2 {
        self.origin_meters
    }

    /// Indica si el eje `Y` positivo apunta hacia arriba (`true`) o hacia abajo (`false`).
    pub const fn y_axis_up(&self) -> bool {
        self.y_axis_up
    }

    /// Convierte un punto en metros del mundo a coordenadas locales de este espacio.
    pub fn world_to_local(&self, world_meters: Point2) -> Result<Point2, ProjectionError> {
        let x = (world_meters.x() - self.origin_meters.x()) / self.meters_per_unit;
        let raw_y = (world_meters.y() - self.origin_meters.y()) / self.meters_per_unit;
        let y = if self.y_axis_up { raw_y } else { -raw_y };
        Point2::new(x, y)
    }

    /// Convierte un punto en coordenadas locales de este espacio a metros del mundo.
    pub fn local_to_world(&self, local: Point2) -> Result<Point2, ProjectionError> {
        let x = self.origin_meters.x() + local.x() * self.meters_per_unit;
        let signed_y = if self.y_axis_up {
            local.y()
        } else {
            -local.y()
        };
        let y = self.origin_meters.y() + signed_y * self.meters_per_unit;
        Point2::new(x, y)
    }

    /// Convierte una longitud no negativa en metros a unidades de este espacio.
    pub fn meters_to_local_length(&self, meters: f64) -> Result<f64, ProjectionError> {
        let meters = ensure_non_negative("length_meters", meters)?;
        let scaled = meters / self.meters_per_unit;
        ensure_non_negative("scaled_length", scaled)
    }

    /// Convierte una longitud no negativa en unidades de este espacio a metros.
    pub fn local_to_meters_length(&self, local_len: f64) -> Result<f64, ProjectionError> {
        let local_len = ensure_non_negative("local_length", local_len)?;
        let meters = local_len * self.meters_per_unit;
        ensure_non_negative("length_meters", meters)
    }
}

impl Default for CoordinateSpace {
    fn default() -> Self {
        Self::meters()
    }
}

/// Transformación afín de escala/origen/orientación entre dos [`CoordinateSpace`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaleTransform {
    source: CoordinateSpace,
    target: CoordinateSpace,
}

impl ScaleTransform {
    /// Crea una transformación explícita desde `source` hacia `target`.
    pub const fn new(source: CoordinateSpace, target: CoordinateSpace) -> Self {
        Self { source, target }
    }

    /// Crea una transformación desde metros canónicos hacia `target`.
    pub const fn from_meters(target: CoordinateSpace) -> Self {
        Self {
            source: CoordinateSpace::meters(),
            target,
        }
    }

    /// Espacio de coordenadas de origen.
    pub const fn source(&self) -> CoordinateSpace {
        self.source
    }

    /// Espacio de coordenadas de destino.
    pub const fn target(&self) -> CoordinateSpace {
        self.target
    }

    /// Transforma un punto desde `source` hacia `target`.
    pub fn transform_point(&self, point: Point2) -> Result<Point2, ProjectionError> {
        let world = self.source.local_to_world(point)?;
        self.target.world_to_local(world)
    }

    /// Transforma un punto desde `target` de vuelta hacia `source`.
    pub fn inverse_transform_point(&self, point: Point2) -> Result<Point2, ProjectionError> {
        let world = self.target.local_to_world(point)?;
        self.source.world_to_local(world)
    }

    /// Escala una longitud no negativa desde `source` hacia `target`.
    pub fn transform_length(&self, length: f64) -> Result<f64, ProjectionError> {
        let meters = self.source.local_to_meters_length(length)?;
        self.target.meters_to_local_length(meters)
    }

    /// Escala una longitud no negativa desde `target` de vuelta hacia `source`.
    pub fn inverse_transform_length(&self, length: f64) -> Result<f64, ProjectionError> {
        let meters = self.target.local_to_meters_length(length)?;
        self.source.meters_to_local_length(meters)
    }
}

pub(crate) fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
