//! Registro de labs de terceros en Studio (API **v0**, GOG-103).
//!
//! Un proyecto externo une dos contratos independientes:
//! - dominio: [`gogamen_sdk::lab::LabDescriptor`] (sin tipos gráficos);
//! - gráfico: [`gogamen_projection::ProjectionRegistration`] + [`LabProjector`].
//!
//! [`LabRegistration::register`] valida ambos (conformidad GOG-103), comprueba que enlazan por
//! id y produce un [`RegisteredLab`] con el [`LabEntry`] listo para [`crate::StudioShell::new`]
//! y el [`FieldSchema`] del inspector derivado de los campos declarados. No hay que modificar
//! ningún código de Studio.
//!
//! # Estabilidad
//! [`API_VERSION`] = `"v0"`: inestable entre versiones `0.x`.
//!
//! # Vista especializada (vía de escape, sólo Studio)
//! [`SpecializedView`] permite a un host Studio sustituir el render SVG por defecto de un lab.
//! Es un tipo **exclusivo de este crate**: `gogamen-core`, `gogamen-cli`, `gogamen-sdk` y
//! `gogamen-projection` no exponen ningún tipo de vista (hay un test guardián).
//! Un lab que sólo implemente [`LabProjector`] funciona sin ella y sin dependencias de UI.

use std::fmt;
use std::sync::Arc;

use gogamen_projection::{
    LabProjector, ProjectionRegistration, RegistrationDiagnostic, RegistrationIssue, Scene2D,
    check_registration, check_scene, viz_spec::InspectorLod,
};
use gogamen_sdk::lab::{LabDescriptor, LabDescriptorIssue, LabUnit};

use crate::control::Observation;
use crate::inspector::{FieldDescriptor, FieldSchema, Lod};
use crate::params::{
    Mutability, ParamDescriptor, ParamId, ParamKind, ParamSchema, ParamUnit, ParamValue,
};
use crate::shell::{LabEntry, LabSession};

/// Versión del contrato de registro de terceros.
pub const API_VERSION: &str = "v0";

/// Vía de escape: vista especializada de un lab. **Sólo existe en Studio.**
pub trait SpecializedView: Send + Sync {
    /// Id estable de la vista.
    fn view_id(&self) -> &str;
    /// Render alternativo de la escena del lab.
    fn render(&self, scene: &Scene2D) -> String;
}

/// Fábrica de proyectores (una instancia por sesión).
pub type ProjectorFactory = Arc<dyn Fn() -> Box<dyn LabProjector> + Send + Sync>;

/// Rechazo del registro: tipado y localizable.
#[derive(Debug, Clone, PartialEq)]
pub enum RegisterError {
    /// Descriptor de dominio inválido.
    Descriptor(Vec<LabDescriptorIssue>),
    /// Contrato gráfico no conforme.
    Projection(Vec<RegistrationDiagnostic>),
    /// El `lab_id` gráfico no coincide con el id del descriptor.
    LabIdMismatch {
        /// Id del descriptor.
        descriptor: String,
        /// Id de la proyección.
        projection: String,
    },
    /// Esquema de parámetros no construible.
    Params,
}

impl RegisterError {
    /// Códigos estables de todas las incidencias (localización).
    pub fn codes(&self) -> Vec<&'static str> {
        match self {
            Self::Descriptor(v) => v.iter().map(LabDescriptorIssue::code).collect(),
            Self::Projection(v) => v.iter().map(|d| d.issue.code()).collect(),
            Self::LabIdMismatch { .. } => vec!["registration.lab_id_mismatch"],
            Self::Params => vec!["registration.invalid_params"],
        }
    }
}

impl fmt::Display for RegisterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Descriptor(v) => {
                let t: Vec<_> = v.iter().map(ToString::to_string).collect();
                write!(f, "invalid lab descriptor: {}", t.join("; "))
            }
            Self::Projection(v) => {
                let t: Vec<_> = v.iter().map(ToString::to_string).collect();
                write!(f, "non-conformant projection: {}", t.join("; "))
            }
            Self::LabIdMismatch {
                descriptor,
                projection,
            } => write!(
                f,
                "[registration.lab_id_mismatch] projection targets '{projection}' but descriptor is '{descriptor}'"
            ),
            Self::Params => write!(f, "[registration.invalid_params] invalid param schema"),
        }
    }
}

impl std::error::Error for RegisterError {}

/// Paquete de registro de un lab de terceros.
pub struct LabRegistration {
    /// Contrato de dominio.
    pub descriptor: LabDescriptor,
    /// Contrato gráfico.
    pub projection: ProjectionRegistration,
    /// Fábrica de proyectores.
    pub factory: ProjectorFactory,
    /// Vista especializada opcional (escape hatch).
    pub specialized_view: Option<Arc<dyn SpecializedView>>,
}

/// Resultado del registro: entrada para el shell + esquema de inspector.
pub struct RegisteredLab {
    entry: LabEntry,
    inspector_schema: FieldSchema,
    specialized_view: Option<Arc<dyn SpecializedView>>,
    descriptor: LabDescriptor,
    projection: ProjectionRegistration,
}

impl RegisteredLab {
    /// Entrada lista para `StudioShell::new`.
    pub fn into_entry(self) -> LabEntry {
        self.entry
    }
    /// Esquema de inspector derivado de los campos declarados.
    pub fn inspector_schema(&self) -> &FieldSchema {
        &self.inspector_schema
    }
    /// Vista especializada, si se declaró.
    pub fn specialized_view(&self) -> Option<&Arc<dyn SpecializedView>> {
        self.specialized_view.as_ref()
    }
    /// Descriptor de dominio.
    pub fn descriptor(&self) -> &LabDescriptor {
        &self.descriptor
    }
    /// Contrato gráfico.
    pub fn projection(&self) -> &ProjectionRegistration {
        &self.projection
    }
}

fn lod(l: InspectorLod) -> Lod {
    match l {
        InspectorLod::Minimal => Lod::Minimal,
        InspectorLod::Standard => Lod::Standard,
        InspectorLod::Full => Lod::Full,
    }
}

fn unit(u: LabUnit) -> ParamUnit {
    match u {
        LabUnit::None => ParamUnit::None,
        LabUnit::Seconds => ParamUnit::Seconds,
        LabUnit::Meters => ParamUnit::Meters,
        LabUnit::Radians => ParamUnit::Radians,
        LabUnit::Degrees => ParamUnit::Degrees,
        LabUnit::Hertz => ParamUnit::Hertz,
        LabUnit::Newtons => ParamUnit::Newtons,
    }
}

impl LabRegistration {
    /// Valida ambos contratos y construye el [`RegisteredLab`].
    pub fn register(self) -> Result<RegisteredLab, RegisterError> {
        self.descriptor.check().map_err(RegisterError::Descriptor)?;
        check_registration(&self.projection).map_err(RegisterError::Projection)?;
        if self.projection.lab_id != self.descriptor.id.as_str() {
            return Err(RegisterError::LabIdMismatch {
                descriptor: self.descriptor.id.as_str().to_owned(),
                projection: self.projection.lab_id.clone(),
            });
        }
        let params = ParamSchema::new(
            self.descriptor
                .params
                .iter()
                .map(|p| ParamDescriptor {
                    id: ParamId::new(p.id.clone()),
                    label: p.label.clone(),
                    kind: ParamKind::Float {
                        min: p.min,
                        max: p.max,
                        step: ((p.max - p.min) / 100.0).max(f64::MIN_POSITIVE),
                    },
                    unit: unit(p.unit),
                    mutability: Mutability::Mutable,
                    default: ParamValue::Float(p.default),
                })
                .collect(),
        )
        .map_err(|_| RegisterError::Params)?;
        let mut inspector_schema = FieldSchema::new();
        for f in &self.projection.inspector_fields {
            inspector_schema.register(FieldDescriptor::new(
                f.name.clone(),
                format!("{} [{}]", f.label, f.unit),
                lod(f.min_lod),
            ));
        }
        let reg = self.projection.clone();
        let factory = self.factory.clone();
        let entry = LabEntry::new(
            self.descriptor.id.clone(),
            self.descriptor.name.clone(),
            Arc::new(move || {
                let mut projector = factory();
                projector.advance_to(gogamen_core::Tick::ZERO);
                let mut s = ProjectedSession {
                    projector,
                    reg: reg.clone(),
                    scene: Scene2D::new(
                        gogamen_projection::SceneId::new(0),
                        gogamen_core::Tick::ZERO,
                        gogamen_core::EpistemicKind::Observed,
                        gogamen_projection::CoordinateSpace::from_units(reg.units),
                    ),
                };
                s.refresh();
                Box::new(s) as Box<dyn LabSession>
            }),
        )
        .with_param_schema(params);
        Ok(RegisteredLab {
            entry,
            inspector_schema,
            specialized_view: self.specialized_view,
            descriptor: self.descriptor,
            projection: self.projection,
        })
    }
}

/// Registra varios labs; falla si dos comparten id (la unicidad también la exige el shell).
pub fn register_all(
    regs: impl IntoIterator<Item = LabRegistration>,
) -> Result<Vec<RegisteredLab>, RegisterError> {
    let mut out: Vec<RegisteredLab> = Vec::new();
    for r in regs {
        let lab = r.register()?;
        if out.iter().any(|o| o.descriptor.id == lab.descriptor.id) {
            return Err(RegisterError::Projection(vec![RegistrationDiagnostic {
                path: "lab_id".into(),
                issue: RegistrationIssue::DuplicateId {
                    id: lab.descriptor.id.as_str().to_owned(),
                },
            }]));
        }
        out.push(lab);
    }
    Ok(out)
}

struct ProjectedSession {
    projector: Box<dyn LabProjector>,
    reg: ProjectionRegistration,
    scene: Scene2D,
}

impl ProjectedSession {
    /// Proyecta y adopta la escena sólo si cumple unidades y límites; si no, conserva la
    /// anterior (el shell nunca recibe un frame fuera de contrato).
    fn refresh(&mut self) {
        if let Ok(scene) = self.projector.project()
            && check_scene(&self.reg, &scene).is_ok()
        {
            self.scene = scene;
        }
    }
}

impl LabSession for ProjectedSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }
    fn sync_to(&mut self, observation: &Observation) {
        self.projector.advance_to(observation.tick);
        self.refresh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StudioShell;
    use crate::fixtures::scene_alpha;
    use gogamen_core::{EffectKind, Permission, Tick};
    use gogamen_projection::{FrameLimits, ProjectionError, ScaleUnits};

    struct Alpha;
    impl LabProjector for Alpha {
        fn advance_to(&mut self, _: Tick) {}
        fn project(&self) -> Result<Scene2D, ProjectionError> {
            Ok(scene_alpha())
        }
    }

    fn reg_with(limits: FrameLimits) -> LabRegistration {
        LabRegistration {
            descriptor: LabDescriptor::new("t.lab", "T", "0.1.0"),
            projection: ProjectionRegistration::new("t.lab", "t.proj", ScaleUnits::Meters, limits),
            factory: Arc::new(|| Box::new(Alpha)),
            specialized_view: None,
        }
    }

    fn ok_limits() -> FrameLimits {
        FrameLimits {
            max_nodes: 10,
            max_bytes: 10_000,
        }
    }

    #[test]
    fn api_is_explicitly_v0() {
        assert_eq!(API_VERSION, "v0");
    }

    #[test]
    fn passing_registration_loads_in_shell() {
        let lab = reg_with(ok_limits()).register().unwrap();
        let mut shell = StudioShell::new(vec![lab.into_entry()], 100.0, 100.0).unwrap();
        shell.select_lab("t.lab").unwrap();
        assert!(shell.render_svg().unwrap().contains("<svg"));
    }

    #[test]
    fn actuation_permission_and_effect_are_rejected() {
        let mut r = reg_with(ok_limits());
        r.projection = r.projection.with_permission(Permission::state_write());
        let e = r.register().err().unwrap();
        assert_eq!(e.codes(), ["registration.actuation_permission"]);
        let mut r = reg_with(ok_limits());
        r.projection = r.projection.with_effect(EffectKind::StateMutation);
        assert_eq!(
            r.register().err().unwrap().codes(),
            ["registration.actuation_effect"]
        );
    }

    #[test]
    fn id_mismatch_bad_descriptor_and_duplicates_are_rejected() {
        let mut r = reg_with(ok_limits());
        r.projection.lab_id = "other.lab".into();
        assert_eq!(
            r.register().err().unwrap().codes(),
            ["registration.lab_id_mismatch"]
        );
        let mut r = reg_with(ok_limits());
        r.descriptor.version = "x".into();
        assert_eq!(
            r.register().err().unwrap().codes(),
            ["lab.malformed_version"]
        );
        let e = register_all([reg_with(ok_limits()), reg_with(ok_limits())])
            .err()
            .unwrap();
        assert_eq!(e.codes(), ["registration.duplicate_id"]);
    }

    #[test]
    fn frame_over_limits_is_never_adopted_by_the_session() {
        // scene_alpha tiene 2 nodos: con max_nodes=1 no se adopta (escena vacía de respaldo).
        let lab = reg_with(FrameLimits {
            max_nodes: 1,
            max_bytes: 10_000,
        })
        .register()
        .unwrap();
        let mut shell = StudioShell::new(vec![lab.into_entry()], 100.0, 100.0).unwrap();
        shell.select_lab("t.lab").unwrap();
        assert!(!shell.render_svg().unwrap().contains("<circle"));
    }

    /// AC3: la vía de escape `SpecializedView` sólo existe en Studio.
    #[test]
    fn specialized_view_escape_hatch_is_studio_only() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        for krate in [
            "gogamen-core",
            "gogamen-cli",
            "gogamen-sdk",
            "gogamen-projection",
        ] {
            let mut stack = vec![root.join(krate).join("src")];
            while let Some(dir) = stack.pop() {
                for e in std::fs::read_dir(&dir).unwrap() {
                    let p = e.unwrap().path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if p.extension().is_some_and(|x| x == "rs") {
                        let t = std::fs::read_to_string(&p).unwrap();
                        // los docs pueden mencionarla; el código no debe declararla ni usarla
                        let code = t
                            .lines()
                            .filter(|l| !l.trim_start().starts_with("//"))
                            .any(|l| l.contains("SpecializedView") || l.contains("gogamen_studio"));
                        assert!(!code, "{} exposes studio view types", p.display());
                    }
                }
            }
        }
    }
}
