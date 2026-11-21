//! Contrato gráfico de proyecciones para terceros (API **v0**, GOG-103).
//!
//! Este módulo es el punto de entrada estable (en el sentido `v0`: *inestable*) para que un
//! proyecto externo registre una proyección 2D y campos de inspector **sin tocar Studio**.
//! Separa dos contratos:
//! - **Dominio** (`gogamen_sdk::lab::LabDescriptor`): id, nombre, versión, parámetros. Sin tipos
//!   gráficos.
//! - **Gráfico** (este módulo): [`ProjectionRegistration`] (proyección, unidades, límites por
//!   frame, campos de inspector, permisos solicitados) + [`LabProjector`] (produce [`Scene2D`]).
//!
//! # Estabilidad y versionado
//! [`API_VERSION`] es `"v0"`: pueden producirse cambios incompatibles entre versiones `0.x`.
//!
//! # Ejecutar sin dependencias de UI
//! Este crate es `std`-only y no depende de Studio, Bevy ni egui. [`run_headless`] avanza un
//! [`LabProjector`] tick a tick y valida cada escena contra los límites declarados.
//!
//! # Conformidad
//! [`check_registration`] valida ids (únicos, bien formados), unidades, límites de frame y la
//! **ausencia de permisos de actuación/efectos**; una proyección es sólo lectura. Los
//! diagnósticos ([`RegistrationDiagnostic`]) son tipados y localizables (`code()` + argumentos).
//!
//! # Vista especializada
//! La vía de escape para vistas especializadas no existe aquí: sólo en `gogamen-studio`.

use std::fmt;

use gogamen_core::{EffectKind, Permission, PermissionSet, Tick};

use crate::coords::ScaleUnits;
use crate::error::ProjectionError;
use crate::scene::Scene2D;
use crate::viz_spec::InspectorLod;

/// Versión del contrato de registro. `v0` = inestable.
pub const API_VERSION: &str = "v0";
/// Longitud máxima de un id.
pub const MAX_ID_LEN: usize = 64;
/// Techo duro de nodos por frame que una registración puede declarar.
pub const MAX_NODES_CEILING: usize = 100_000;
/// Techo duro de bytes por frame que una registración puede declarar.
pub const MAX_BYTES_CEILING: usize = 16 * 1024 * 1024;
/// Símbolos de unidad aceptados en campos de inspector.
pub const KNOWN_UNIT_SYMBOLS: [&str; 7] = ["1", "s", "m", "rad", "deg", "Hz", "N"];

/// Límites por frame de una proyección.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameLimits {
    /// Máximo de nodos por escena.
    pub max_nodes: usize,
    /// Máximo de bytes del JSON canónico de la escena.
    pub max_bytes: usize,
}

/// Campo de inspector declarado por la proyección.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectorFieldDecl {
    /// Nombre estable (`[a-z][a-z0-9._-]*`).
    pub name: String,
    /// Etiqueta visible.
    pub label: String,
    /// Símbolo de unidad (véase [`KNOWN_UNIT_SYMBOLS`]).
    pub unit: String,
    /// Nivel mínimo de detalle.
    pub min_lod: InspectorLod,
}

/// Contrato gráfico de un lab: libre de dominio, enlazado por `lab_id`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionRegistration {
    /// Id del lab (de dominio) al que se asocia.
    pub lab_id: String,
    /// Id de la proyección.
    pub projection_id: String,
    /// Unidades de las escenas emitidas.
    pub units: ScaleUnits,
    /// Límites por frame.
    pub limits: FrameLimits,
    /// Campos de inspector.
    pub inspector_fields: Vec<InspectorFieldDecl>,
    /// Permisos solicitados (sólo `state:read` es admisible).
    pub requested_permissions: PermissionSet,
    /// Efectos solicitados (sólo [`EffectKind::Pure`] es admisible).
    pub requested_effects: Vec<EffectKind>,
}

impl ProjectionRegistration {
    /// Registración sólo lectura: `state:read`, efecto `Pure`, sin campos.
    pub fn new(
        lab_id: impl Into<String>,
        projection_id: impl Into<String>,
        units: ScaleUnits,
        limits: FrameLimits,
    ) -> Self {
        Self {
            lab_id: lab_id.into(),
            projection_id: projection_id.into(),
            units,
            limits,
            inspector_fields: Vec::new(),
            requested_permissions: PermissionSet::empty().with_permission(Permission::state_read()),
            requested_effects: vec![EffectKind::Pure],
        }
    }

    /// Añade un campo de inspector (builder).
    pub fn with_field(mut self, field: InspectorFieldDecl) -> Self {
        self.inspector_fields.push(field);
        self
    }

    /// Solicita un permiso adicional (será rechazado salvo `state:read`).
    pub fn with_permission(mut self, p: Permission) -> Self {
        self.requested_permissions = self.requested_permissions.with_permission(p);
        self
    }

    /// Solicita un efecto adicional (será rechazado salvo `Pure`).
    pub fn with_effect(mut self, e: EffectKind) -> Self {
        self.requested_effects.push(e);
        self
    }
}

/// Causa tipada y localizable de rechazo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistrationIssue {
    /// Id vacío o mal formado.
    MalformedId {
        /// Id recibido.
        id: String,
    },
    /// Id repetido (campos de inspector o registros).
    DuplicateId {
        /// Id repetido.
        id: String,
    },
    /// Símbolo de unidad desconocido en un campo de inspector.
    UnknownUnit {
        /// Campo.
        field: String,
        /// Símbolo recibido.
        unit: String,
    },
    /// Límites de frame cero o por encima del techo.
    InvalidFrameLimits {
        /// Límite de nodos declarado.
        max_nodes: usize,
        /// Límite de bytes declarado.
        max_bytes: usize,
    },
    /// Se solicita un permiso distinto de `state:read` (actuación/E-S/escritura).
    ActuationPermission {
        /// Permiso solicitado.
        permission: String,
    },
    /// Se solicita un efecto distinto de `Pure`.
    ActuationEffect {
        /// Efecto solicitado.
        effect: String,
    },
    /// La escena emitida usa unidades distintas de las declaradas.
    UnitsMismatch {
        /// Declaradas.
        expected: &'static str,
        /// Encontradas.
        found: &'static str,
    },
    /// La escena supera el máximo de nodos.
    TooManyNodes {
        /// Nodos de la escena.
        nodes: usize,
        /// Límite.
        max: usize,
    },
    /// La escena supera el máximo de bytes.
    TooManyBytes {
        /// Bytes de la escena.
        bytes: usize,
        /// Límite.
        max: usize,
    },
    /// El proyector falló al construir la escena.
    Projection {
        /// Mensaje del error de proyección.
        message: String,
    },
}

impl RegistrationIssue {
    /// Código estable para localización.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MalformedId { .. } => "registration.malformed_id",
            Self::DuplicateId { .. } => "registration.duplicate_id",
            Self::UnknownUnit { .. } => "registration.unknown_unit",
            Self::InvalidFrameLimits { .. } => "registration.invalid_frame_limits",
            Self::ActuationPermission { .. } => "registration.actuation_permission",
            Self::ActuationEffect { .. } => "registration.actuation_effect",
            Self::UnitsMismatch { .. } => "frame.units_mismatch",
            Self::TooManyNodes { .. } => "frame.too_many_nodes",
            Self::TooManyBytes { .. } => "frame.too_many_bytes",
            Self::Projection { .. } => "frame.projection_error",
        }
    }
}

/// Diagnóstico: ruta + causa tipada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationDiagnostic {
    /// Ubicación (`inspector_fields[1].unit`).
    pub path: String,
    /// Causa.
    pub issue: RegistrationIssue,
}

impl RegistrationDiagnostic {
    fn new(path: impl Into<String>, issue: RegistrationIssue) -> Self {
        Self {
            path: path.into(),
            issue,
        }
    }
}

impl fmt::Display for RegistrationDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: [{}] ", self.path, self.issue.code())?;
        match &self.issue {
            RegistrationIssue::MalformedId { id } => write!(f, "malformed id '{id}'"),
            RegistrationIssue::DuplicateId { id } => write!(f, "duplicate id '{id}'"),
            RegistrationIssue::UnknownUnit { field, unit } => {
                write!(f, "field '{field}' uses unknown unit '{unit}'")
            }
            RegistrationIssue::InvalidFrameLimits {
                max_nodes,
                max_bytes,
            } => write!(
                f,
                "invalid frame limits (max_nodes={max_nodes}, max_bytes={max_bytes})"
            ),
            RegistrationIssue::ActuationPermission { permission } => {
                write!(f, "permission '{permission}' is not allowed (read-only)")
            }
            RegistrationIssue::ActuationEffect { effect } => {
                write!(f, "effect '{effect}' is not allowed (pure only)")
            }
            RegistrationIssue::UnitsMismatch { expected, found } => {
                write!(f, "scene units '{found}' differ from declared '{expected}'")
            }
            RegistrationIssue::TooManyNodes { nodes, max } => {
                write!(f, "scene has {nodes} nodes (max {max})")
            }
            RegistrationIssue::TooManyBytes { bytes, max } => {
                write!(f, "scene has {bytes} bytes (max {max})")
            }
            RegistrationIssue::Projection { message } => write!(f, "projection error: {message}"),
        }
    }
}

impl std::error::Error for RegistrationDiagnostic {}

/// Id bien formado: `[a-z][a-z0-9._-]*`, longitud `<= MAX_ID_LEN`.
pub fn is_well_formed_id(id: &str) -> bool {
    let mut chars = id.chars();
    id.len() <= MAX_ID_LEN
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c))
}

/// Conformidad estática de una registración. Devuelve **todas** las incidencias.
pub fn check_registration(reg: &ProjectionRegistration) -> Result<(), Vec<RegistrationDiagnostic>> {
    let mut out = Vec::new();
    for (path, id) in [
        ("lab_id", &reg.lab_id),
        ("projection_id", &reg.projection_id),
    ] {
        if !is_well_formed_id(id) {
            out.push(RegistrationDiagnostic::new(
                path,
                RegistrationIssue::MalformedId { id: id.clone() },
            ));
        }
    }
    let l = reg.limits;
    if l.max_nodes == 0
        || l.max_bytes == 0
        || l.max_nodes > MAX_NODES_CEILING
        || l.max_bytes > MAX_BYTES_CEILING
    {
        out.push(RegistrationDiagnostic::new(
            "limits",
            RegistrationIssue::InvalidFrameLimits {
                max_nodes: l.max_nodes,
                max_bytes: l.max_bytes,
            },
        ));
    }
    for (i, f) in reg.inspector_fields.iter().enumerate() {
        if !is_well_formed_id(&f.name) {
            out.push(RegistrationDiagnostic::new(
                format!("inspector_fields[{i}].name"),
                RegistrationIssue::MalformedId { id: f.name.clone() },
            ));
        }
        if reg.inspector_fields[..i].iter().any(|o| o.name == f.name) {
            out.push(RegistrationDiagnostic::new(
                format!("inspector_fields[{i}].name"),
                RegistrationIssue::DuplicateId { id: f.name.clone() },
            ));
        }
        if !KNOWN_UNIT_SYMBOLS.contains(&f.unit.as_str()) {
            out.push(RegistrationDiagnostic::new(
                format!("inspector_fields[{i}].unit"),
                RegistrationIssue::UnknownUnit {
                    field: f.name.clone(),
                    unit: f.unit.clone(),
                },
            ));
        }
    }
    let read = Permission::state_read();
    for p in reg.requested_permissions.iter() {
        if *p != read {
            out.push(RegistrationDiagnostic::new(
                "requested_permissions",
                RegistrationIssue::ActuationPermission {
                    permission: p.as_str().to_owned(),
                },
            ));
        }
    }
    for (i, e) in reg.requested_effects.iter().enumerate() {
        if *e != EffectKind::Pure {
            out.push(RegistrationDiagnostic::new(
                format!("requested_effects[{i}]"),
                RegistrationIssue::ActuationEffect {
                    effect: format!("{e:?}"),
                },
            ));
        }
    }
    if out.is_empty() { Ok(()) } else { Err(out) }
}

/// Conformidad de una escena emitida frente a la registración (unidades y límites por frame).
pub fn check_scene(
    reg: &ProjectionRegistration,
    scene: &Scene2D,
) -> Result<(), Vec<RegistrationDiagnostic>> {
    let mut out = Vec::new();
    let found = scene.coordinate_space().units();
    if found != reg.units {
        out.push(RegistrationDiagnostic::new(
            "scene.coordinate_space.units",
            RegistrationIssue::UnitsMismatch {
                expected: reg.units.as_str(),
                found: found.as_str(),
            },
        ));
    }
    let nodes: usize = scene.layers().iter().map(|l| l.nodes().len()).sum();
    if nodes > reg.limits.max_nodes {
        out.push(RegistrationDiagnostic::new(
            "scene.nodes",
            RegistrationIssue::TooManyNodes {
                nodes,
                max: reg.limits.max_nodes,
            },
        ));
    }
    let bytes = scene.to_canonical_json().len();
    if bytes > reg.limits.max_bytes {
        out.push(RegistrationDiagnostic::new(
            "scene.bytes",
            RegistrationIssue::TooManyBytes {
                bytes,
                max: reg.limits.max_bytes,
            },
        ));
    }
    if out.is_empty() { Ok(()) } else { Err(out) }
}

/// Registro de registraciones: rechaza ids de lab/proyección duplicados y no conformes.
#[derive(Debug, Default)]
pub struct ProjectionRegistry {
    items: Vec<ProjectionRegistration>,
}

impl ProjectionRegistry {
    /// Registro vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra tras comprobar conformidad y unicidad.
    pub fn register(
        &mut self,
        reg: ProjectionRegistration,
    ) -> Result<(), Vec<RegistrationDiagnostic>> {
        let mut diags = check_registration(&reg).err().unwrap_or_default();
        for o in &self.items {
            if o.lab_id == reg.lab_id {
                diags.push(RegistrationDiagnostic::new(
                    "lab_id",
                    RegistrationIssue::DuplicateId {
                        id: reg.lab_id.clone(),
                    },
                ));
            }
            if o.projection_id == reg.projection_id {
                diags.push(RegistrationDiagnostic::new(
                    "projection_id",
                    RegistrationIssue::DuplicateId {
                        id: reg.projection_id.clone(),
                    },
                ));
            }
        }
        if diags.is_empty() {
            self.items.push(reg);
            Ok(())
        } else {
            Err(diags)
        }
    }

    /// Registraciones en orden de alta.
    pub fn items(&self) -> &[ProjectionRegistration] {
        &self.items
    }
}

/// Proyector de un lab: avanza hasta un tick y emite la escena 2D. Sin dependencias de UI.
pub trait LabProjector {
    /// Alinea el estado del lab con `tick` (idempotente para el mismo tick).
    fn advance_to(&mut self, tick: Tick);
    /// Escena del estado actual.
    fn project(&self) -> Result<Scene2D, ProjectionError>;
}

/// Ejecuta el lab sin UI: escena inicial y una por tick `1..=ticks`, validadas contra `reg`.
pub fn run_headless(
    projector: &mut dyn LabProjector,
    reg: &ProjectionRegistration,
    ticks: u64,
) -> Result<Vec<Scene2D>, Vec<RegistrationDiagnostic>> {
    check_registration(reg)?;
    let mut scenes = Vec::new();
    for t in 0..=ticks {
        projector.advance_to(Tick::new(t));
        let scene = projector.project().map_err(|e| {
            vec![RegistrationDiagnostic::new(
                "projector",
                RegistrationIssue::Projection {
                    message: e.to_string(),
                },
            )]
        })?;
        check_scene(reg, &scene)?;
        scenes.push(scene);
    }
    Ok(scenes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::{CoordinateSpace, Point2};
    use crate::scene::{LayerId, SceneId, SceneLayer, SceneNode, SceneNodeId, ScenePrimitive};
    use gogamen_core::EpistemicKind;

    fn limits() -> FrameLimits {
        FrameLimits {
            max_nodes: 10,
            max_bytes: 10_000,
        }
    }

    fn good() -> ProjectionRegistration {
        ProjectionRegistration::new("acme.lab", "acme.proj", ScaleUnits::Meters, limits())
            .with_field(InspectorFieldDecl {
                name: "x".into(),
                label: "X".into(),
                unit: "m".into(),
                min_lod: InspectorLod::Minimal,
            })
    }

    fn codes(r: Result<(), Vec<RegistrationDiagnostic>>) -> Vec<&'static str> {
        r.unwrap_err().iter().map(|d| d.issue.code()).collect()
    }

    fn scene(n: usize, space: CoordinateSpace) -> Scene2D {
        let mut s = Scene2D::new(
            SceneId::new(1),
            Tick::new(0),
            EpistemicKind::Observed,
            space,
        );
        s.add_layer(SceneLayer::new(LayerId::Environment)).unwrap();
        for i in 0..n {
            let c = ScenePrimitive::circle(Point2::new(i as f64, 0.0).unwrap(), 1.0).unwrap();
            s.add_node(
                LayerId::Environment,
                SceneNode::new(
                    SceneNodeId::new(i as u64 + 1),
                    None,
                    EpistemicKind::Observed,
                    c,
                    true,
                ),
            )
            .unwrap();
        }
        s
    }

    #[test]
    fn api_is_explicitly_v0() {
        assert_eq!(API_VERSION, "v0");
    }

    #[test]
    fn passing_registration() {
        assert_eq!(check_registration(&good()), Ok(()));
        let mut reg = ProjectionRegistry::new();
        reg.register(good()).unwrap();
        assert_eq!(reg.items().len(), 1);
    }

    #[test]
    fn rejects_malformed_ids() {
        let mut r = good();
        r.lab_id = "Bad Id".into();
        r.projection_id = String::new();
        assert_eq!(
            codes(check_registration(&r)),
            ["registration.malformed_id"; 2]
        );
    }

    #[test]
    fn rejects_duplicate_ids() {
        let mut r = good();
        let dup = r.inspector_fields[0].clone();
        r.inspector_fields.push(dup);
        assert_eq!(codes(check_registration(&r)), ["registration.duplicate_id"]);
        let mut reg = ProjectionRegistry::new();
        reg.register(good()).unwrap();
        let d = reg.register(good()).unwrap_err();
        assert_eq!(d.len(), 2);
        assert!(
            d.iter()
                .all(|d| d.issue.code() == "registration.duplicate_id")
        );
        assert_eq!(reg.items().len(), 1);
    }

    #[test]
    fn rejects_unknown_units() {
        let mut r = good();
        r.inspector_fields[0].unit = "furlong".into();
        let d = check_registration(&r).unwrap_err();
        assert_eq!(d[0].path, "inspector_fields[0].unit");
        assert_eq!(d[0].issue.code(), "registration.unknown_unit");
        assert!(d[0].to_string().contains("furlong"));
    }

    #[test]
    fn rejects_scene_with_wrong_units() {
        let r = good();
        let s = scene(1, CoordinateSpace::from_units(ScaleUnits::Kilometers));
        assert_eq!(codes(check_scene(&r, &s)), ["frame.units_mismatch"]);
    }

    #[test]
    fn rejects_invalid_frame_limits_and_oversized_frames() {
        let mut r = good();
        r.limits.max_nodes = 0;
        assert_eq!(
            codes(check_registration(&r)),
            ["registration.invalid_frame_limits"]
        );
        r.limits = FrameLimits {
            max_nodes: MAX_NODES_CEILING + 1,
            max_bytes: 1,
        };
        assert_eq!(
            codes(check_registration(&r)),
            ["registration.invalid_frame_limits"]
        );
        let r = good();
        assert_eq!(
            check_scene(&r, &scene(10, CoordinateSpace::meters())),
            Ok(())
        );
        assert_eq!(
            codes(check_scene(&r, &scene(11, CoordinateSpace::meters()))),
            ["frame.too_many_nodes"]
        );
        let mut tiny = good();
        tiny.limits.max_bytes = 20;
        assert_eq!(
            codes(check_scene(&tiny, &scene(2, CoordinateSpace::meters()))),
            ["frame.too_many_bytes"]
        );
    }

    #[test]
    fn rejects_actuation_permissions_and_effects() {
        for p in [
            Permission::state_write(),
            Permission::network(),
            Permission::new("actuator:drive"),
        ] {
            let d = check_registration(&good().with_permission(p.clone())).unwrap_err();
            assert_eq!(d.len(), 1);
            assert_eq!(d[0].issue.code(), "registration.actuation_permission");
            assert_eq!(
                d[0].issue,
                RegistrationIssue::ActuationPermission {
                    permission: p.as_str().to_owned()
                }
            );
        }
        for e in [
            EffectKind::StateMutation,
            EffectKind::Network,
            EffectKind::Filesystem,
        ] {
            let d = check_registration(&good().with_effect(e)).unwrap_err();
            assert_eq!(d[0].issue.code(), "registration.actuation_effect");
        }
    }

    struct Dots(u64);
    impl LabProjector for Dots {
        fn advance_to(&mut self, tick: Tick) {
            self.0 = tick.as_u64();
        }
        fn project(&self) -> Result<Scene2D, ProjectionError> {
            Ok(scene(self.0 as usize + 1, CoordinateSpace::meters()))
        }
    }

    #[test]
    fn headless_runs_and_enforces_limits() {
        let scenes = run_headless(&mut Dots(0), &good(), 3).unwrap();
        assert_eq!(scenes.len(), 4);
        let err = run_headless(&mut Dots(0), &good(), 10).unwrap_err();
        assert_eq!(err[0].issue.code(), "frame.too_many_nodes");
    }
}
