//! Modelo de escena 2D tipada, capas explícitas, picking estable y serialización JSON canónica (`std`-only).

use std::fmt::{self, Write};

use gogamen_core::{EntityId, EpistemicKind, Tick};

use crate::coords::{CoordinateSpace, Point2, normalize_zero};
use crate::error::{ProjectionError, ensure_non_negative};
use crate::image::Texture;

/// Identificador único de una escena 2D proyectada.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SceneId(u64);

impl SceneId {
    /// Crea un [`SceneId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SceneId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "scene({})", self.0)
    }
}

/// Identificador estable de un nodo dentro de una [`Scene2D`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SceneNodeId(u64);

impl SceneNodeId {
    /// Crea un [`SceneNodeId`] a partir de un entero sin signo.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Devuelve el valor numérico subyacente.
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SceneNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node({})", self.0)
    }
}

/// Capas explícitas de la escena 2D con ordenamiento determinista.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LayerId {
    /// Entorno físico o estado observado del mundo.
    Environment,
    /// Lecturas de sensores o estimaciones inferidas.
    Sensors,
    /// Trayectorias o estados hipotéticos / imaginación / rollouts.
    Imagination,
    /// Indicadores de decisión, guardia o estado terminal.
    Decisions,
    /// Etiquetas textuales de depuración o inspección.
    Labels,
    /// Capa personalizada identificada por un código numérico explícito.
    Custom(u32),
}

impl LayerId {
    /// Orden Z por defecto para apilamiento determinista.
    pub const fn default_z_order(self) -> i32 {
        match self {
            Self::Environment => 0,
            Self::Sensors => 10,
            Self::Imagination => 20,
            Self::Decisions => 30,
            Self::Labels => 40,
            Self::Custom(code) => 100_i32.saturating_add(code as i32),
        }
    }

    /// Nombre canónico estable para serialización determinista.
    pub fn canonical_name(self) -> String {
        match self {
            Self::Environment => "environment".to_string(),
            Self::Sensors => "sensors".to_string(),
            Self::Imagination => "imagination".to_string(),
            Self::Decisions => "decisions".to_string(),
            Self::Labels => "labels".to_string(),
            Self::Custom(code) => format!("custom:{code}"),
        }
    }
}

impl fmt::Display for LayerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical_name())
    }
}

/// Primitivas geométricas 2D independientes del renderer.
#[derive(Debug, Clone, PartialEq)]
pub enum ScenePrimitive {
    /// Círculo definido por centro y radio no negativo.
    Circle {
        /// Centro del círculo en coordenadas de la escena.
        center: Point2,
        /// Radio en unidades de la escena (`>= 0.0`).
        radius: f64,
    },
    /// Polilínea abierta o cerrada de al menos dos vértices.
    Polyline {
        /// Secuencia ordenada de vértices en coordenadas de la escena.
        points: Vec<Point2>,
        /// Indica si el último vértice se conecta con el primero.
        closed: bool,
    },
    /// Rectángulo alineado a los ejes definido por esquina mínima, ancho y alto.
    Rect {
        /// Esquina inferior-izquierda (mínima en X e Y).
        min: Point2,
        /// Ancho no negativo (`>= 0.0`).
        width: f64,
        /// Alto no negativo (`>= 0.0`).
        height: f64,
    },
    /// Etiqueta de texto posicionada en 2D.
    Label {
        /// Posición de anclaje en coordenadas de la escena.
        position: Point2,
        /// Contenido textual de la etiqueta.
        text: String,
    },
    /// Imagen texturizada (p. ej. framebuffer) ocupando un rectángulo de la escena.
    Image {
        /// Esquina mínima (arriba-izquierda en escenas con eje Y hacia abajo).
        min: Point2,
        /// Ancho no negativo en unidades de la escena.
        width: f64,
        /// Alto no negativo en unidades de la escena.
        height: f64,
        /// Píxeles de la imagen, sin modificar.
        texture: Texture,
    },
}

impl ScenePrimitive {
    /// Construye un círculo validando que `radius >= 0.0` y sea finito.
    pub fn circle(center: Point2, radius: f64) -> Result<Self, ProjectionError> {
        let radius = ensure_non_negative("circle.radius", radius)?;
        Ok(Self::Circle {
            center,
            radius: normalize_zero(radius),
        })
    }

    /// Construye una polilínea validando que contenga al menos dos puntos.
    pub fn polyline(points: Vec<Point2>, closed: bool) -> Result<Self, ProjectionError> {
        if points.len() < 2 {
            return Err(ProjectionError::InsufficientPolylinePoints {
                count: points.len(),
            });
        }
        Ok(Self::Polyline { points, closed })
    }

    /// Construye un rectángulo validando que `width >= 0.0` y `height >= 0.0`.
    pub fn rect(min: Point2, width: f64, height: f64) -> Result<Self, ProjectionError> {
        let width = ensure_non_negative("rect.width", width)?;
        let height = ensure_non_negative("rect.height", height)?;
        Ok(Self::Rect {
            min,
            width: normalize_zero(width),
            height: normalize_zero(height),
        })
    }

    /// Construye una imagen validando `width >= 0.0` y `height >= 0.0`.
    pub fn image(
        min: Point2,
        width: f64,
        height: f64,
        texture: Texture,
    ) -> Result<Self, ProjectionError> {
        let width = ensure_non_negative("image.width", width)?;
        let height = ensure_non_negative("image.height", height)?;
        Ok(Self::Image {
            min,
            width: normalize_zero(width),
            height: normalize_zero(height),
            texture,
        })
    }

    /// Construye una etiqueta posicionada en `position`.
    pub fn label(position: Point2, text: impl Into<String>) -> Self {
        Self::Label {
            position,
            text: text.into(),
        }
    }

    /// Punto de anclaje representativo de la primitiva en coordenadas de la escena.
    pub fn anchor_point(&self) -> Point2 {
        match self {
            Self::Circle { center, .. } => *center,
            Self::Polyline { points, .. } => points.first().copied().unwrap_or(Point2::ZERO),
            Self::Rect { min, width, height }
            | Self::Image {
                min, width, height, ..
            } => Point2::new(min.x() + width * 0.5, min.y() + height * 0.5).unwrap_or(*min),
            Self::Label { position, .. } => *position,
        }
    }

    /// Distancia mínima desde `point` hasta la primitiva (`0.0` si está contenida).
    pub fn hit_distance(&self, point: Point2) -> f64 {
        match self {
            Self::Circle { center, radius } => (center.distance_to(point) - *radius).max(0.0),
            Self::Rect { min, width, height }
            | Self::Image {
                min, width, height, ..
            } => {
                let max_x = min.x() + *width;
                let max_y = min.y() + *height;
                let dx = if point.x() < min.x() {
                    min.x() - point.x()
                } else if point.x() > max_x {
                    point.x() - max_x
                } else {
                    0.0
                };
                let dy = if point.y() < min.y() {
                    min.y() - point.y()
                } else if point.y() > max_y {
                    point.y() - max_y
                } else {
                    0.0
                };
                dx.hypot(dy)
            }
            Self::Polyline { points, closed } => {
                if points.len() < 2 {
                    return points
                        .first()
                        .map_or(f64::INFINITY, |p| p.distance_to(point));
                }
                let mut best = f64::INFINITY;
                for pair in points.windows(2) {
                    let dist = distance_point_to_segment(point, pair[0], pair[1]);
                    if dist < best {
                        best = dist;
                    }
                }
                if *closed && let (Some(first), Some(last)) = (points.first(), points.last()) {
                    let dist = distance_point_to_segment(point, *last, *first);
                    if dist < best {
                        best = dist;
                    }
                }
                best
            }
            Self::Label { position, .. } => position.distance_to(point),
        }
    }
}

/// Nodo tipado dentro de una [`SceneLayer`] con identidad estable para selección/picking.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneNode {
    node_id: SceneNodeId,
    entity_id: Option<EntityId>,
    epistemic_kind: EpistemicKind,
    primitive: ScenePrimitive,
    selectable: bool,
    state_code: Option<u32>,
    color: Option<[u8; 3]>,
}

impl SceneNode {
    /// Longitud en bytes del fragmento canónico JSON de este nodo.
    pub(crate) fn canonical_json_len(&self) -> usize {
        let mut out = String::with_capacity(256);
        write_node(&mut out, self);
        out.len()
    }

    /// Crea un nuevo [`SceneNode`].
    pub const fn new(
        node_id: SceneNodeId,
        entity_id: Option<EntityId>,
        epistemic_kind: EpistemicKind,
        primitive: ScenePrimitive,
        selectable: bool,
    ) -> Self {
        Self {
            node_id,
            entity_id,
            epistemic_kind,
            primitive,
            selectable,
            state_code: None,
            color: None,
        }
    }

    /// Overrides the stroke colour (`[r, g, b]`) that renderers would otherwise derive from the
    /// epistemic kind. Used e.g. by 3D views to colour coordinate axes. The epistemic kind is
    /// unchanged. Serialized (`"color":"#rrggbb"`) only when set, so scenes without it keep their
    /// canonical JSON byte-for-byte.
    pub const fn with_color(mut self, rgb: [u8; 3]) -> Self {
        self.color = Some(rgb);
        self
    }

    /// Explicit colour override, if any.
    pub const fn color(&self) -> Option<[u8; 3]> {
        self.color
    }

    /// Asocia un código de estado numérico opcional (por ejemplo de `VisualEntitySample`).
    pub const fn with_state_code(mut self, state_code: u32) -> Self {
        self.state_code = Some(state_code);
        self
    }

    /// Identificador estable del nodo.
    pub const fn node_id(&self) -> SceneNodeId {
        self.node_id
    }

    /// Identificador de entidad de dominio asociado, si aplica.
    pub const fn entity_id(&self) -> Option<EntityId> {
        self.entity_id
    }

    /// Clasificación epistémica del dato proyectado.
    pub const fn epistemic_kind(&self) -> EpistemicKind {
        self.epistemic_kind
    }

    /// Primitiva geométrica 2D.
    pub const fn primitive(&self) -> &ScenePrimitive {
        &self.primitive
    }

    /// Indica si el nodo participa en selección/picking.
    pub const fn is_selectable(&self) -> bool {
        self.selectable
    }

    /// Código numérico de estado asociado, si existe.
    pub const fn state_code(&self) -> Option<u32> {
        self.state_code
    }
}

/// Capa ordenada de nodos dentro de una [`Scene2D`].
#[derive(Debug, Clone, PartialEq)]
pub struct SceneLayer {
    id: LayerId,
    z_order: i32,
    visible: bool,
    nodes: Vec<SceneNode>,
}

impl SceneLayer {
    /// Crea una capa vacía con su `z_order` por defecto y visible.
    pub const fn new(id: LayerId) -> Self {
        Self {
            id,
            z_order: id.default_z_order(),
            visible: true,
            nodes: Vec::new(),
        }
    }

    /// Crea una capa vacía con `z_order` y visibilidad explícitos.
    pub const fn with_visibility(id: LayerId, z_order: i32, visible: bool) -> Self {
        Self {
            id,
            z_order,
            visible,
            nodes: Vec::new(),
        }
    }

    /// Identificador de la capa.
    pub const fn id(&self) -> LayerId {
        self.id
    }

    /// Orden Z de apilamiento.
    pub const fn z_order(&self) -> i32 {
        self.z_order
    }

    /// Indica si la capa es visible.
    pub const fn is_visible(&self) -> bool {
        self.visible
    }

    /// Bytes que aportan los nodos de la capa al JSON canónico (nodos más separadores),
    /// sin cabecera de capa. Permite razonar sobre tamaños sin reserializar la escena.
    pub(crate) fn canonical_nodes_len(&self) -> usize {
        let sum: usize = self.nodes.iter().map(SceneNode::canonical_json_len).sum();
        sum + self.nodes.len().saturating_sub(1)
    }

    /// Nodos contenidos en la capa.
    pub fn nodes(&self) -> &[SceneNode] {
        &self.nodes
    }

    /// Añade un nodo a la capa verificando que no exista un `SceneNodeId` duplicado.
    pub fn add_node(&mut self, node: SceneNode) -> Result<(), ProjectionError> {
        if self.nodes.iter().any(|n| n.node_id() == node.node_id()) {
            return Err(ProjectionError::DuplicateNodeId {
                node_id: node.node_id().as_u64(),
            });
        }
        self.nodes.push(node);
        Ok(())
    }
}

/// Resultado de una consulta de selección/picking en una [`Scene2D`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SelectionHit {
    /// Capa donde se encontró el nodo seleccionado.
    pub layer_id: LayerId,
    /// Identificador estable del nodo seleccionado.
    pub node_id: SceneNodeId,
    /// Identificador de entidad asociado, si el nodo representa una entidad.
    pub entity_id: Option<EntityId>,
    /// Distancia en unidades de la escena desde el punto consultado (`0.0` si es impacto directo).
    pub distance: f64,
}

/// Escena 2D tipada, determinista e independiente del renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct Scene2D {
    scene_id: SceneId,
    tick: Tick,
    epistemic_kind: EpistemicKind,
    coordinate_space: CoordinateSpace,
    layers: Vec<SceneLayer>,
}

impl Scene2D {
    /// Crea una [`Scene2D`] vacía en el [`CoordinateSpace`] indicado.
    pub const fn new(
        scene_id: SceneId,
        tick: Tick,
        epistemic_kind: EpistemicKind,
        coordinate_space: CoordinateSpace,
    ) -> Self {
        Self {
            scene_id,
            tick,
            epistemic_kind,
            coordinate_space,
            layers: Vec::new(),
        }
    }

    /// Identificador de la escena.
    pub const fn scene_id(&self) -> SceneId {
        self.scene_id
    }

    /// Tick lógico asociado a la escena.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Clasificación epistémica dominante de la escena.
    pub const fn epistemic_kind(&self) -> EpistemicKind {
        self.epistemic_kind
    }

    /// Espacio de coordenadas y escala en el que están expresadas las primitivas.
    pub const fn coordinate_space(&self) -> CoordinateSpace {
        self.coordinate_space
    }

    /// Capas ordenadas de forma determinista por `(z_order, layer_id)`.
    pub fn layers(&self) -> &[SceneLayer] {
        &self.layers
    }

    /// Añade una capa completa verificando unicidad de `LayerId` y de `SceneNodeId` globales.
    pub fn add_layer(&mut self, layer: SceneLayer) -> Result<(), ProjectionError> {
        if self.layers.iter().any(|existing| existing.id == layer.id) {
            return Err(ProjectionError::DuplicateLayerId { layer_id: layer.id });
        }
        for node in &layer.nodes {
            if self.find_by_node_id(node.node_id()).is_some() {
                return Err(ProjectionError::DuplicateNodeId {
                    node_id: node.node_id().as_u64(),
                });
            }
        }
        self.layers.push(layer);
        self.layers
            .sort_by(|a, b| a.z_order.cmp(&b.z_order).then_with(|| a.id.cmp(&b.id)));
        Ok(())
    }

    /// Añade un nodo a `layer_id` (creando la capa si no existe), manteniendo unicidad global de `SceneNodeId`.
    pub fn add_node(&mut self, layer_id: LayerId, node: SceneNode) -> Result<(), ProjectionError> {
        if self.find_by_node_id(node.node_id()).is_some() {
            return Err(ProjectionError::DuplicateNodeId {
                node_id: node.node_id().as_u64(),
            });
        }
        if let Some(layer) = self.layers.iter_mut().find(|l| l.id == layer_id) {
            layer.add_node(node)
        } else {
            let mut layer = SceneLayer::new(layer_id);
            layer.add_node(node)?;
            self.add_layer(layer)
        }
    }

    /// Busca un nodo por su [`SceneNodeId`].
    pub fn find_by_node_id(&self, node_id: SceneNodeId) -> Option<(LayerId, &SceneNode)> {
        for layer in &self.layers {
            if let Some(node) = layer.nodes.iter().find(|n| n.node_id() == node_id) {
                return Some((layer.id, node));
            }
        }
        None
    }

    /// Busca todos los nodos asociados a un [`EntityId`] en orden determinista de capas.
    pub fn find_by_entity_id(&self, entity_id: EntityId) -> Vec<(LayerId, &SceneNode)> {
        let mut matches = Vec::new();
        for layer in &self.layers {
            for node in &layer.nodes {
                if node.entity_id() == Some(entity_id) {
                    matches.push((layer.id, node));
                }
            }
        }
        matches
    }

    /// Selecciona el nodo visible y seleccionable más cercano a `point` dentro de `tolerance` (en unidades de la escena).
    pub fn pick_at(&self, point: Point2, tolerance: f64) -> Option<SelectionHit> {
        if !tolerance.is_finite() || tolerance < 0.0 {
            return None;
        }
        let mut best: Option<(i32, SelectionHit)> = None;
        for layer in self.layers.iter().rev() {
            if !layer.visible {
                continue;
            }
            for node in &layer.nodes {
                if !node.is_selectable() {
                    continue;
                }
                let distance = node.primitive().hit_distance(point);
                if distance > tolerance {
                    continue;
                }
                let candidate = SelectionHit {
                    layer_id: layer.id,
                    node_id: node.node_id(),
                    entity_id: node.entity_id(),
                    distance,
                };
                let replace = match best {
                    None => true,
                    Some((best_z, best_hit)) => {
                        distance < best_hit.distance
                            || (distance == best_hit.distance
                                && (layer.z_order > best_z
                                    || (layer.z_order == best_z
                                        && candidate.node_id < best_hit.node_id)))
                    }
                };
                if replace {
                    best = Some((layer.z_order, candidate));
                }
            }
        }
        best.map(|(_, hit)| hit)
    }

    /// Selecciona un nodo usando coordenadas del mundo en metros, invariante respecto a la escala de la escena.
    pub fn pick_at_world(
        &self,
        world_meters: Point2,
        tolerance_meters: f64,
    ) -> Result<Option<SelectionHit>, ProjectionError> {
        let local_point = self.coordinate_space.world_to_local(world_meters)?;
        let local_tolerance = self
            .coordinate_space
            .meters_to_local_length(tolerance_meters)?;
        Ok(self.pick_at(local_point, local_tolerance))
    }

    /// Serializa la escena a un formato JSON canónico y determinista sin dependencias externas.
    pub fn to_canonical_json(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push('{');
        let _ = write!(
            out,
            "\"scene_id\":{},\"tick\":{},\"epistemic_kind\":\"{}\",\"coordinate_space\":",
            self.scene_id.as_u64(),
            self.tick.as_u64(),
            epistemic_str(self.epistemic_kind)
        );
        write_coordinate_space(&mut out, &self.coordinate_space);
        out.push_str(",\"layers\":[");
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if layer_idx > 0 {
                out.push(',');
            }
            write_layer(&mut out, layer);
        }
        out.push_str("]}");
        out
    }
}

fn distance_point_to_segment(p: Point2, a: Point2, b: Point2) -> f64 {
    let abx = b.x() - a.x();
    let aby = b.y() - a.y();
    let len_sq = abx * abx + aby * aby;
    if len_sq == 0.0 {
        return p.distance_to(a);
    }
    let t = (((p.x() - a.x()) * abx + (p.y() - a.y()) * aby) / len_sq).clamp(0.0, 1.0);
    let proj_x = a.x() + t * abx;
    let proj_y = a.y() + t * aby;
    (p.x() - proj_x).hypot(p.y() - proj_y)
}

fn epistemic_str(kind: EpistemicKind) -> &'static str {
    match kind {
        EpistemicKind::Observed => "observed",
        EpistemicKind::Inferred => "inferred",
        EpistemicKind::Hypothetical => "hypothetical",
    }
}

fn write_f64(out: &mut String, value: f64) {
    let normalized = normalize_zero(value);
    let _ = write!(out, "{normalized}");
}

fn write_json_str(out: &mut String, text: &str) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_point(out: &mut String, p: Point2) {
    out.push_str("{\"x\":");
    write_f64(out, p.x());
    out.push_str(",\"y\":");
    write_f64(out, p.y());
    out.push('}');
}

fn write_coordinate_space(out: &mut String, space: &CoordinateSpace) {
    out.push_str("{\"units\":\"");
    out.push_str(space.units().as_str());
    out.push_str("\",\"meters_per_unit\":");
    write_f64(out, space.meters_per_unit());
    out.push_str(",\"origin_meters\":");
    write_point(out, space.origin_meters());
    let _ = write!(out, ",\"y_axis_up\":{}}}", space.y_axis_up());
}

fn write_primitive(out: &mut String, primitive: &ScenePrimitive) {
    match primitive {
        ScenePrimitive::Circle { center, radius } => {
            out.push_str("{\"kind\":\"circle\",\"center\":");
            write_point(out, *center);
            out.push_str(",\"radius\":");
            write_f64(out, *radius);
            out.push('}');
        }
        ScenePrimitive::Polyline { points, closed } => {
            out.push_str("{\"kind\":\"polyline\",\"points\":[");
            for (idx, point) in points.iter().enumerate() {
                if idx > 0 {
                    out.push(',');
                }
                write_point(out, *point);
            }
            let _ = write!(out, "],\"closed\":{closed}}}");
        }
        ScenePrimitive::Rect { min, width, height } => {
            out.push_str("{\"kind\":\"rect\",\"min\":");
            write_point(out, *min);
            out.push_str(",\"width\":");
            write_f64(out, *width);
            out.push_str(",\"height\":");
            write_f64(out, *height);
            out.push('}');
        }
        ScenePrimitive::Image {
            min,
            width,
            height,
            texture,
        } => {
            out.push_str("{\"kind\":\"image\",\"min\":");
            write_point(out, *min);
            out.push_str(",\"width\":");
            write_f64(out, *width);
            out.push_str(",\"height\":");
            write_f64(out, *height);
            out.push_str(",\"texture\":");
            texture.write_canonical_json(out);
            out.push('}');
        }
        ScenePrimitive::Label { position, text } => {
            out.push_str("{\"kind\":\"label\",\"position\":");
            write_point(out, *position);
            out.push_str(",\"text\":");
            write_json_str(out, text);
            out.push('}');
        }
    }
}

fn write_node(out: &mut String, node: &SceneNode) {
    let _ = write!(
        out,
        "{{\"node_id\":{},\"entity_id\":",
        node.node_id().as_u64()
    );
    match node.entity_id() {
        Some(id) => {
            let _ = write!(out, "{}", id.as_u64());
        }
        None => out.push_str("null"),
    }
    let _ = write!(
        out,
        ",\"epistemic_kind\":\"{}\",\"selectable\":{},\"state_code\":",
        epistemic_str(node.epistemic_kind()),
        node.is_selectable()
    );
    match node.state_code() {
        Some(code) => {
            let _ = write!(out, "{code}");
        }
        None => out.push_str("null"),
    }
    out.push_str(",\"primitive\":");
    write_primitive(out, node.primitive());
    if let Some([r, g, b]) = node.color() {
        let _ = write!(out, ",\"color\":\"#{r:02x}{g:02x}{b:02x}\"");
    }
    out.push('}');
}

fn write_layer(out: &mut String, layer: &SceneLayer) {
    out.push_str("{\"id\":");
    write_json_str(out, &layer.id().canonical_name());
    let _ = write!(
        out,
        ",\"z_order\":{},\"visible\":{},\"nodes\":[",
        layer.z_order(),
        layer.is_visible()
    );
    for (idx, node) in layer.nodes().iter().enumerate() {
        if idx > 0 {
            out.push(',');
        }
        write_node(out, node);
    }
    out.push_str("]}");
}
