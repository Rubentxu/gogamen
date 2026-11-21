//! Mapa discreto 2D (`GridMap`), tipos de celda y mapa predefinido con callejón sin salida.

use gogamen_core::DomainError;

/// Dimensiones básicas de una rejilla 2D.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridSize {
    /// Ancho de la rejilla en celdas.
    pub width: usize,
    /// Alto de la rejilla en celdas.
    pub height: usize,
}

impl GridSize {
    /// Crea un tamaño de rejilla y devuelve el número total de celdas.
    pub const fn cell_count(self) -> usize {
        self.width * self.height
    }
}

/// Coordenada discreta `(x, y)` dentro de un [`GridMap`] (`x` columna, `y` fila).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GridPos {
    /// Índice de columna (`0..width`).
    pub x: usize,
    /// Índice de fila (`0..height`).
    pub y: usize,
}

impl GridPos {
    /// Construye una coordenada discreta `(x, y)`.
    pub const fn new(x: usize, y: usize) -> Self {
        Self { x, y }
    }

    /// Distancia Manhattan hasta `other`.
    pub const fn manhattan_distance(self, other: Self) -> usize {
        self.x.abs_diff(other.x) + self.y.abs_diff(other.y)
    }
}

/// Tipo de celda en un [`GridMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CellKind {
    /// Celda transitable libre.
    Empty,
    /// Obstáculo sólido o muro infranqueable.
    Wall,
    /// Celda objetivo que finaliza el episodio con éxito.
    Goal,
    /// Celda de peligro/trampa que penaliza y puede finalizar el episodio.
    Hazard,
}

impl CellKind {
    /// Codificación escalar determinista para vectores de observación.
    pub const fn as_code(self) -> f64 {
        match self {
            Self::Empty => 0.0,
            Self::Wall => 1.0,
            Self::Goal => 2.0,
            Self::Hazard => 3.0,
        }
    }

    /// Código escalar reservado para celdas fuera de los límites del mapa en ventanas parciales.
    pub const OUT_OF_BOUNDS_CODE: f64 = -1.0;

    /// Indica si la celda bloquea el paso del agente.
    pub const fn is_blocking(self) -> bool {
        matches!(self, Self::Wall)
    }
}

/// Mapa 2D discreto inmutable con obstáculos, punto de inicio y objetivo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GridMap {
    width: usize,
    height: usize,
    cells: Vec<CellKind>,
    start: GridPos,
    goal: GridPos,
}

impl GridMap {
    /// Construye un [`GridMap`] validando dimensiones, consistencia de `cells`, `start` y `goal`.
    pub fn new(
        width: usize,
        height: usize,
        cells: Vec<CellKind>,
        start: GridPos,
        goal: GridPos,
    ) -> Result<Self, DomainError> {
        if width == 0 || height == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "grid map dimensions must be non-zero",
            });
        }
        if cells.len() != width * height {
            return Err(DomainError::InvalidTransition {
                reason: "grid map cells length must equal width * height",
            });
        }
        if start.x >= width || start.y >= height {
            return Err(DomainError::InvalidTransition {
                reason: "grid map start position is out of bounds",
            });
        }
        if goal.x >= width || goal.y >= height {
            return Err(DomainError::InvalidTransition {
                reason: "grid map goal position is out of bounds",
            });
        }
        if cells[start.y * width + start.x].is_blocking() {
            return Err(DomainError::InvalidTransition {
                reason: "grid map start position cannot be a wall",
            });
        }
        if cells[goal.y * width + goal.x] != CellKind::Goal {
            return Err(DomainError::InvalidTransition {
                reason: "grid map goal cell must have CellKind::Goal",
            });
        }
        Ok(Self {
            width,
            height,
            cells,
            start,
            goal,
        })
    }

    /// Construye un [`GridMap`] a partir de filas ASCII:
    /// `#` = `Wall`, `.` = `Empty`, `S` = inicio (`Empty`), `G` = `Goal`, `H` = `Hazard`.
    pub fn from_ascii(rows: &[&str]) -> Result<Self, DomainError> {
        let height = rows.len();
        if height == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "ascii grid map cannot be empty",
            });
        }
        let width = rows[0].chars().count();
        if width == 0 {
            return Err(DomainError::InvalidTransition {
                reason: "ascii grid map rows cannot be empty",
            });
        }

        let mut cells = Vec::with_capacity(width * height);
        let mut start = None;
        let mut goal = None;

        for (y, row) in rows.iter().enumerate() {
            if row.chars().count() != width {
                return Err(DomainError::InvalidTransition {
                    reason: "ascii grid map rows must have uniform width",
                });
            }
            for (x, ch) in row.chars().enumerate() {
                let cell = match ch {
                    '#' => CellKind::Wall,
                    '.' => CellKind::Empty,
                    'S' => {
                        if start.is_some() {
                            return Err(DomainError::InvalidTransition {
                                reason: "ascii grid map must have exactly one start 'S'",
                            });
                        }
                        start = Some(GridPos::new(x, y));
                        CellKind::Empty
                    }
                    'G' => {
                        if goal.is_some() {
                            return Err(DomainError::InvalidTransition {
                                reason: "ascii grid map must have exactly one goal 'G'",
                            });
                        }
                        goal = Some(GridPos::new(x, y));
                        CellKind::Goal
                    }
                    'H' => CellKind::Hazard,
                    _ => {
                        return Err(DomainError::InvalidTransition {
                            reason: "unknown character in ascii grid map",
                        });
                    }
                };
                cells.push(cell);
            }
        }

        let start = start.ok_or(DomainError::InvalidTransition {
            reason: "ascii grid map is missing start 'S'",
        })?;
        let goal = goal.ok_or(DomainError::InvalidTransition {
            reason: "ascii grid map is missing goal 'G'",
        })?;

        Self::new(width, height, cells, start, goal)
    }

    /// Devuelve el mapa predefinido con callejón sin salida (`9x7`).
    ///
    /// El agente comienza en `(3, 3)` frente a un callejón hacia la derecha (`(4, 3) -> (5, 3)`)
    /// que apunta hacia la meta `G` en `(7, 1)`. Para alcanzar `G` es obligatorio **retroceder**
    /// (aumentando inicialmente la distancia Manhattan a `G` desde 6 hasta 10) rodeando el muro
    /// por el corredor izquierdo e inferior.
    pub fn cul_de_sac_map() -> Self {
        Self::from_ascii(&[
            "#########",
            "#...#..G#",
            "#.#.###.#",
            "#.#S..#.#",
            "#.#####.#",
            "#.......#",
            "#########",
        ])
        .expect("predefined cul-de-sac map must be valid")
    }

    /// Ancho del mapa en celdas.
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Alto del mapa en celdas.
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Tamaño de la rejilla como [`GridSize`].
    pub const fn size(&self) -> GridSize {
        GridSize {
            width: self.width,
            height: self.height,
        }
    }

    /// Posición inicial del agente al reiniciar un episodio.
    pub const fn start(&self) -> GridPos {
        self.start
    }

    /// Posición de la celda objetivo `Goal`.
    pub const fn goal(&self) -> GridPos {
        self.goal
    }

    /// Slice inmutable de todas las celdas en orden fila-mayor (`y * width + x`).
    pub fn cells(&self) -> &[CellKind] {
        &self.cells
    }

    /// Indica si `pos` está dentro de los límites `[0, width) x [0, height)`.
    pub const fn in_bounds(&self, pos: GridPos) -> bool {
        pos.x < self.width && pos.y < self.height
    }

    /// Devuelve el [`CellKind`] de `pos`, o `None` si está fuera de límites.
    pub fn cell(&self, pos: GridPos) -> Option<CellKind> {
        if self.in_bounds(pos) {
            Some(self.cells[pos.y * self.width + pos.x])
        } else {
            None
        }
    }

    /// Devuelve `true` si la celda es transitable (dentro de límites y no es [`CellKind::Wall`]).
    pub fn is_passable(&self, pos: GridPos) -> bool {
        self.cell(pos).is_some_and(|c| !c.is_blocking())
    }
}

/// Alias de conveniencia para [`GridMap::cul_de_sac_map`].
pub fn cul_de_sac_map() -> GridMap {
    GridMap::cul_de_sac_map()
}
