//! Proyector puro 2D desde [`SimulationSnapshot`] y [`VisualFrame`] hacia [`Scene2D`] (`std`-only).
//!
//! Este módulo no tiene acceso alguno a [`gogamen_core::LiveEnvironment`] ni
//! [`gogamen_core::Actuator`]; opera exclusivamente como transformación pura e
//! inmutable (`&self`) sobre datos capturados o frames visuales.

use gogamen_core::{
    EntityId, EpistemicKind, EventSource, SimulationSnapshot, Terminal, VisualFrame,
};

use crate::coords::{CoordinateSpace, Point2, ScaleTransform};
use crate::error::{ProjectionError, ensure_positive};
use crate::scene::{LayerId, Scene2D, SceneId, SceneNode, SceneNodeId, ScenePrimitive};

const LABEL_NODE_OFFSET: u64 = 1_000_000_000_000;
const SNAPSHOT_TRAJECTORY_NODE_ID: u64 = 100_000;
const SNAPSHOT_DECISION_NODE_ID: u64 = 200_000;
const SNAPSHOT_STATUS_LABEL_NODE_ID: u64 = LABEL_NODE_OFFSET;

/// Configuración determinista para proyectar estados o frames a [`Scene2D`].
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionConfig {
    scene_id: SceneId,
    coordinate_space: CoordinateSpace,
    default_entity_radius_meters: f64,
    min_entity_radius_meters: f64,
    include_labels: bool,
}

impl ProjectionConfig {
    /// Crea una configuración base para `scene_id` y `coordinate_space`.
    pub const fn new(scene_id: SceneId, coordinate_space: CoordinateSpace) -> Self {
        Self {
            scene_id,
            coordinate_space,
            default_entity_radius_meters: 0.25,
            min_entity_radius_meters: 0.05,
            include_labels: true,
        }
    }

    /// Configura los radios por defecto y mínimo en metros (ambos estrictamente positivos).
    pub fn with_radii_meters(
        mut self,
        default_meters: f64,
        min_meters: f64,
    ) -> Result<Self, ProjectionError> {
        self.default_entity_radius_meters =
            ensure_positive("config.default_entity_radius_meters", default_meters)?;
        self.min_entity_radius_meters =
            ensure_positive("config.min_entity_radius_meters", min_meters)?;
        Ok(self)
    }

    /// Activa o desactiva la emisión de nodos en la capa [`LayerId::Labels`].
    pub const fn with_labels(mut self, include_labels: bool) -> Self {
        self.include_labels = include_labels;
        self
    }

    /// Identificador de escena configurado.
    pub const fn scene_id(&self) -> SceneId {
        self.scene_id
    }

    /// Espacio de coordenadas destino.
    pub const fn coordinate_space(&self) -> CoordinateSpace {
        self.coordinate_space
    }

    /// Radio por defecto en metros cuando la muestra no aporta escalar positivo.
    pub const fn default_entity_radius_meters(&self) -> f64 {
        self.default_entity_radius_meters
    }

    /// Radio mínimo en metros para entidades proyectadas.
    pub const fn min_entity_radius_meters(&self) -> f64 {
        self.min_entity_radius_meters
    }

    /// Indica si se emiten etiquetas textuales en [`LayerId::Labels`].
    pub const fn include_labels(&self) -> bool {
        self.include_labels
    }
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self::new(SceneId::new(1), CoordinateSpace::meters())
    }
}

/// Proyector puro 2D independiente del renderer y sin acceso al entorno vivo.
#[derive(Debug, Clone, PartialEq)]
pub struct Projector2D {
    config: ProjectionConfig,
    transform: ScaleTransform,
}

impl Projector2D {
    /// Crea un [`Projector2D`] a partir de `config`.
    pub const fn new(config: ProjectionConfig) -> Self {
        let transform = ScaleTransform::from_meters(config.coordinate_space);
        Self { config, transform }
    }

    /// Configuración activa del proyector.
    pub const fn config(&self) -> &ProjectionConfig {
        &self.config
    }

    /// Transformación de escala desde metros del mundo hacia el espacio destino.
    pub const fn transform(&self) -> ScaleTransform {
        self.transform
    }

    /// Proyecta un [`VisualFrame`] validado a una [`Scene2D`] determinista.
    pub fn project_frame(&self, frame: &VisualFrame) -> Result<Scene2D, ProjectionError> {
        frame.validate()?;

        let mut scene = Scene2D::new(
            self.config.scene_id,
            frame.tick(),
            frame.epistemic_kind(),
            self.config.coordinate_space,
        );

        let target_layer = layer_for_frame(frame.epistemic_kind(), frame.source());
        let mut sorted_indices: Vec<usize> = (0..frame.entities().len()).collect();
        sorted_indices.sort_by_key(|&idx| (frame.entities()[idx].entity_id().as_u64(), idx));

        for idx in sorted_indices {
            let sample = &frame.entities()[idx];
            let pos = sample.position();
            let world_pt = Point2::new(pos[0], pos[1])?;
            let local_pt = self.transform.transform_point(world_pt)?;

            let radius_meters = if sample.scalar() > 0.0 {
                sample.scalar().max(self.config.min_entity_radius_meters)
            } else {
                self.config.default_entity_radius_meters
            };
            let local_radius = self.transform.transform_length(radius_meters)?;

            let entity_id = sample.entity_id();
            let node = SceneNode::new(
                SceneNodeId::new(entity_id.as_u64()),
                Some(entity_id),
                frame.epistemic_kind(),
                ScenePrimitive::circle(local_pt, local_radius)?,
                true,
            )
            .with_state_code(sample.state_code());
            scene.add_node(target_layer, node)?;

            if self.config.include_labels {
                let label_id = SceneNodeId::new(LABEL_NODE_OFFSET.wrapping_add(entity_id.as_u64()));
                let label_node = SceneNode::new(
                    label_id,
                    Some(entity_id),
                    frame.epistemic_kind(),
                    ScenePrimitive::label(
                        local_pt,
                        format!("E{}:{}", entity_id.as_u64(), sample.state_code()),
                    ),
                    false,
                );
                scene.add_node(LayerId::Labels, label_node)?;
            }
        }

        Ok(scene)
    }

    /// Proyecta un [`SimulationSnapshot`] verificado a una [`Scene2D`] determinista.
    pub fn project_snapshot(
        &self,
        snapshot: &SimulationSnapshot,
    ) -> Result<Scene2D, ProjectionError> {
        snapshot.verify()?;

        let mut scene = Scene2D::new(
            self.config.scene_id,
            snapshot.tick(),
            EpistemicKind::Observed,
            self.config.coordinate_space,
        );

        let world_points = extract_snapshot_points(snapshot.state().values())?;
        let local_radius = self
            .transform
            .transform_length(self.config.default_entity_radius_meters)?;

        let mut local_points = Vec::with_capacity(world_points.len());
        for (idx, world_pt) in world_points.into_iter().enumerate() {
            let local_pt = self.transform.transform_point(world_pt)?;
            local_points.push(local_pt);

            let entity_id = EntityId::new(idx as u64);
            let node = SceneNode::new(
                SceneNodeId::new(idx as u64),
                Some(entity_id),
                EpistemicKind::Observed,
                ScenePrimitive::circle(local_pt, local_radius)?,
                true,
            );
            scene.add_node(LayerId::Environment, node)?;
        }

        if local_points.len() >= 2 {
            let path_node = SceneNode::new(
                SceneNodeId::new(SNAPSHOT_TRAJECTORY_NODE_ID),
                None,
                EpistemicKind::Observed,
                ScenePrimitive::polyline(local_points.clone(), false)?,
                false,
            );
            scene.add_node(LayerId::Environment, path_node)?;
        }

        let anchor_local = local_points.first().copied().unwrap_or(Point2::ZERO);
        let indicator_size = self
            .transform
            .transform_length(self.config.default_entity_radius_meters * 2.0)?;
        let indicator_min = Point2::new(
            anchor_local.x() - indicator_size * 0.5,
            anchor_local.y() - indicator_size * 0.5,
        )?;
        let decision_code = match snapshot.terminal() {
            Terminal::Ongoing => 0,
            Terminal::Terminated(_) => 1,
        };
        let decision_node = SceneNode::new(
            SceneNodeId::new(SNAPSHOT_DECISION_NODE_ID),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::rect(indicator_min, indicator_size, indicator_size)?,
            false,
        )
        .with_state_code(decision_code);
        scene.add_node(LayerId::Decisions, decision_node)?;

        if self.config.include_labels {
            let status = match snapshot.terminal() {
                Terminal::Ongoing => "ongoing",
                Terminal::Terminated(_) => "terminated",
            };
            let label_node = SceneNode::new(
                SceneNodeId::new(SNAPSHOT_STATUS_LABEL_NODE_ID),
                None,
                EpistemicKind::Observed,
                ScenePrimitive::label(
                    anchor_local,
                    format!("tick={} status={status}", snapshot.tick().as_u64()),
                ),
                false,
            );
            scene.add_node(LayerId::Labels, label_node)?;
        }

        Ok(scene)
    }
}

impl Default for Projector2D {
    fn default() -> Self {
        Self::new(ProjectionConfig::default())
    }
}

/// Función pura de conveniencia para proyectar un [`VisualFrame`] a [`Scene2D`].
pub fn project_frame_to_scene(
    frame: &VisualFrame,
    config: &ProjectionConfig,
) -> Result<Scene2D, ProjectionError> {
    Projector2D::new(config.clone()).project_frame(frame)
}

/// Función pura de conveniencia para proyectar un [`SimulationSnapshot`] a [`Scene2D`].
pub fn project_snapshot_to_scene(
    snapshot: &SimulationSnapshot,
    config: &ProjectionConfig,
) -> Result<Scene2D, ProjectionError> {
    Projector2D::new(config.clone()).project_snapshot(snapshot)
}

const fn layer_for_frame(epistemic_kind: EpistemicKind, source: EventSource) -> LayerId {
    match epistemic_kind {
        EpistemicKind::Hypothetical => LayerId::Imagination,
        EpistemicKind::Inferred => LayerId::Sensors,
        EpistemicKind::Observed => match source {
            EventSource::LiveEnvironment => LayerId::Sensors,
            EventSource::GenerativeModel | EventSource::Planner => LayerId::Imagination,
            EventSource::Guard | EventSource::Scheduler => LayerId::Decisions,
            EventSource::SimulationWorld | EventSource::Module(_) => LayerId::Environment,
        },
    }
}

fn extract_snapshot_points(values: &[f64]) -> Result<Vec<Point2>, ProjectionError> {
    if values.is_empty() {
        return Ok(vec![Point2::ZERO]);
    }
    if values.len() == 1 {
        return Ok(vec![Point2::new(values[0], 0.0)?]);
    }
    let mut points = Vec::with_capacity(values.len().div_ceil(2));
    for chunk in values.chunks(2) {
        let x = chunk[0];
        let y = if chunk.len() > 1 { chunk[1] } else { 0.0 };
        points.push(Point2::new(x, y)?);
    }
    Ok(points)
}
