//! `VisualizationSpec` declarativa y portable (`GOG-102`): vistas, capas e inspector por IDs de catálogo.
//!
//! # Modelo
//! - [`VisualizationSpec`] (`gogamen.visualization_spec.v0`): lista de [`ViewSpec`]; cada vista elige una
//!   proyección registrada (`projection`), unidades de coordenadas ([`ScaleUnits`]), capas
//!   ([`LayerBinding`]) con parámetros tipados/unidades ([`ParamValue`], [`ExprUnit`]) e inspector
//!   opcional ([`InspectorSpec`], espejo declarativo de `FieldDescriptor` de `gogamen-studio`, sin depender de él).
//! - [`VisualizationCatalog`] registra proyecciones y capas con sus descriptores de parámetros
//!   ([`ParamDescriptor`]); [`VisualizationCatalog::builtin`] cubre las capas de [`LayerId`].
//! - [`VisualizationSpec::validate`] devuelve diagnósticos ([`VizDiagnostic`]) con ubicación tipo
//!   JSON-pointer (`views[0].layers[1].projection`), ordenados de forma determinista (ruta, código).
//!
//! # Sección opcional headless
//! [`SystemWithVisualization`] transporta un [`SystemSpec`] más la clave opcional `visualization`
//! del mismo documento JSON. El plan headless sólo consume `system`: `to_canonical_json` del
//! `SystemSpec` (y por tanto hashes/fingerprints del artefacto) no cambia con o sin sección.
//! Sin JVM ni dependencias gráficas; el builder Kotlin (`kotlin/`) emite el mismo JSON canónico.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use gogamen_spec::json::{
    JsonNumber, JsonValue, parse_json, write_canonical_f64, write_json_string,
};
use gogamen_spec::{ExprUnit, SpecError, SystemSpec, UnknownFieldPolicy};

use crate::coords::ScaleUnits;
use crate::scene::LayerId;

/// Versión de esquema de [`VisualizationSpec`].
pub const VISUALIZATION_SPEC_SCHEMA_VERSION: &str = "gogamen.visualization_spec.v0";

// ---------------------------------------------------------------------------------------------
// Errores
// ---------------------------------------------------------------------------------------------

/// Error de parseo/serialización de una [`VisualizationSpec`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizSpecError {
    /// Error JSON o de primitivas de `gogamen-spec`.
    Spec(SpecError),
    /// Campo estructuralmente inválido en `path` (ruta tipo JSON-pointer).
    Field {
        /// Ubicación localizable (`views[0].layers[1].layer`).
        path: String,
        /// Motivo.
        reason: String,
    },
}

impl fmt::Display for VizSpecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "{e}"),
            Self::Field { path, reason } => {
                write!(f, "invalid visualization spec at {path}: {reason}")
            }
        }
    }
}

impl std::error::Error for VizSpecError {}

impl From<SpecError> for VizSpecError {
    fn from(e: SpecError) -> Self {
        Self::Spec(e)
    }
}

fn field_err(path: &str, reason: impl Into<String>) -> VizSpecError {
    VizSpecError::Field {
        path: path.to_owned(),
        reason: reason.into(),
    }
}

/// Tipo de diagnóstico de validación semántica.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VizDiagnosticKind {
    /// Identificador vacío.
    EmptyId,
    /// Identificador duplicado dentro de su colección.
    DuplicateId {
        /// Identificador repetido.
        id: String,
    },
    /// Proyección no registrada en el catálogo.
    UnregisteredProjection {
        /// ID de proyección.
        projection: String,
    },
    /// Capa no registrada en el catálogo.
    UnknownLayer {
        /// ID de capa.
        layer: String,
    },
    /// Capa registrada pero incompatible con la proyección elegida.
    LayerNotInProjection {
        /// ID de proyección.
        projection: String,
        /// ID de capa.
        layer: String,
    },
    /// Unidades de coordenadas no soportadas por la proyección.
    UnsupportedUnits {
        /// ID de proyección.
        projection: String,
        /// Unidades solicitadas.
        units: &'static str,
    },
    /// Parámetro no declarado por la capa.
    UnknownParameter {
        /// ID de capa.
        layer: String,
        /// Nombre del parámetro.
        param: String,
    },
    /// Parámetro obligatorio ausente.
    MissingParameter {
        /// Nombre del parámetro.
        param: String,
    },
    /// Tipo de valor incompatible con el descriptor.
    ParameterType {
        /// Nombre del parámetro.
        param: String,
        /// Tipo esperado.
        expected: &'static str,
    },
    /// Valor fuera del rango del descriptor o no finito.
    ParameterOutOfRange {
        /// Nombre del parámetro.
        param: String,
    },
    /// Unidad distinta de la declarada por el descriptor.
    UnitMismatch {
        /// Nombre del parámetro.
        param: String,
        /// Unidad esperada.
        expected: &'static str,
        /// Unidad recibida.
        found: &'static str,
    },
    /// El descriptor exige unidad y el valor no la trae.
    MissingUnit {
        /// Nombre del parámetro.
        param: String,
        /// Unidad esperada.
        expected: &'static str,
    },
    /// Campo de inspector no expuesto por la proyección.
    UnknownInspectorField {
        /// ID de proyección.
        projection: String,
        /// Nombre del campo.
        field: String,
    },
    /// Referencia a nodo inexistente en el `SystemSpec`.
    UnknownNodeRef {
        /// Identificador de nodo.
        node: String,
    },
    /// `system_spec_id` no coincide con el `spec_id` del sistema.
    SystemRefMismatch {
        /// `spec_id` del sistema.
        expected: String,
        /// Valor declarado.
        found: String,
    },
}

impl VizDiagnosticKind {
    /// Código estable y ordenable del diagnóstico.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::EmptyId => "empty_id",
            Self::DuplicateId { .. } => "duplicate_id",
            Self::UnregisteredProjection { .. } => "unregistered_projection",
            Self::UnknownLayer { .. } => "unknown_layer",
            Self::LayerNotInProjection { .. } => "layer_not_in_projection",
            Self::UnsupportedUnits { .. } => "unsupported_units",
            Self::UnknownParameter { .. } => "unknown_parameter",
            Self::MissingParameter { .. } => "missing_parameter",
            Self::ParameterType { .. } => "parameter_type",
            Self::ParameterOutOfRange { .. } => "parameter_out_of_range",
            Self::UnitMismatch { .. } => "unit_mismatch",
            Self::MissingUnit { .. } => "missing_unit",
            Self::UnknownInspectorField { .. } => "unknown_inspector_field",
            Self::UnknownNodeRef { .. } => "unknown_node_ref",
            Self::SystemRefMismatch { .. } => "system_ref_mismatch",
        }
    }
}

/// Diagnóstico localizable: ruta tipo JSON-pointer + causa tipada.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VizDiagnostic {
    /// Ubicación (`views[0].layers[1].projection`).
    pub path: String,
    /// Causa tipada.
    pub kind: VizDiagnosticKind,
}

impl fmt::Display for VizDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: [{}] ", self.path, self.kind.code())?;
        match &self.kind {
            VizDiagnosticKind::EmptyId => write!(f, "empty identifier"),
            VizDiagnosticKind::DuplicateId { id } => write!(f, "duplicate id '{id}'"),
            VizDiagnosticKind::UnregisteredProjection { projection } => {
                write!(f, "projection '{projection}' is not registered")
            }
            VizDiagnosticKind::UnknownLayer { layer } => write!(f, "unknown layer '{layer}'"),
            VizDiagnosticKind::LayerNotInProjection { projection, layer } => {
                write!(
                    f,
                    "layer '{layer}' is not supported by projection '{projection}'"
                )
            }
            VizDiagnosticKind::UnsupportedUnits { projection, units } => {
                write!(
                    f,
                    "projection '{projection}' does not support units '{units}'"
                )
            }
            VizDiagnosticKind::UnknownParameter { layer, param } => {
                write!(f, "layer '{layer}' has no parameter '{param}'")
            }
            VizDiagnosticKind::MissingParameter { param } => {
                write!(f, "missing required parameter '{param}'")
            }
            VizDiagnosticKind::ParameterType { param, expected } => {
                write!(f, "parameter '{param}' expects {expected}")
            }
            VizDiagnosticKind::ParameterOutOfRange { param } => {
                write!(f, "parameter '{param}' is out of range")
            }
            VizDiagnosticKind::UnitMismatch {
                param,
                expected,
                found,
            } => {
                write!(
                    f,
                    "parameter '{param}' expects unit '{expected}', found '{found}'"
                )
            }
            VizDiagnosticKind::MissingUnit { param, expected } => {
                write!(f, "parameter '{param}' requires unit '{expected}'")
            }
            VizDiagnosticKind::UnknownInspectorField { projection, field } => {
                write!(
                    f,
                    "projection '{projection}' does not expose inspector field '{field}'"
                )
            }
            VizDiagnosticKind::UnknownNodeRef { node } => write!(f, "unknown node '{node}'"),
            VizDiagnosticKind::SystemRefMismatch { expected, found } => {
                write!(f, "expected system spec id '{expected}', found '{found}'")
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Catálogo
// ---------------------------------------------------------------------------------------------

/// Tipo de un parámetro de capa.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamKind {
    /// Número finito con rango opcional.
    Number {
        /// Mínimo inclusivo.
        min: Option<f64>,
        /// Máximo inclusivo.
        max: Option<f64>,
    },
    /// Entero con rango opcional.
    Integer {
        /// Mínimo inclusivo.
        min: Option<i64>,
        /// Máximo inclusivo.
        max: Option<i64>,
    },
    /// Booleano.
    Bool,
    /// Texto.
    Text,
}

impl ParamKind {
    const fn name(&self) -> &'static str {
        match self {
            Self::Number { .. } => "number",
            Self::Integer { .. } => "integer",
            Self::Bool => "bool",
            Self::Text => "text",
        }
    }
}

/// Descriptor de un parámetro de capa (análogo a `FieldDescriptor` del inspector).
#[derive(Debug, Clone, PartialEq)]
pub struct ParamDescriptor {
    /// Nombre del parámetro.
    pub name: String,
    /// Tipo y rango.
    pub kind: ParamKind,
    /// Unidad exigida para valores numéricos (`None`: sin unidad).
    pub unit: Option<ExprUnit>,
    /// Obligatorio.
    pub required: bool,
}

/// Capa registrada con sus parámetros.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerDescriptor {
    /// ID de capa (`LayerId::canonical_name`).
    pub id: String,
    /// Parámetros admitidos.
    pub params: Vec<ParamDescriptor>,
}

/// Proyección registrada.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionDescriptor {
    /// ID de proyección.
    pub id: String,
    /// Unidades de coordenadas admitidas.
    pub units: Vec<ScaleUnits>,
    /// Capas admitidas.
    pub layers: Vec<String>,
    /// Campos de inspector expuestos.
    pub inspector_fields: Vec<String>,
}

/// Registro de proyecciones y capas contra el que se valida una [`VisualizationSpec`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VisualizationCatalog {
    projections: BTreeMap<String, ProjectionDescriptor>,
    layers: BTreeMap<String, LayerDescriptor>,
}

impl VisualizationCatalog {
    /// Catálogo vacío.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registra (o reemplaza) una proyección.
    pub fn register_projection(&mut self, d: ProjectionDescriptor) -> &mut Self {
        self.projections.insert(d.id.clone(), d);
        self
    }

    /// Registra (o reemplaza) una capa.
    pub fn register_layer(&mut self, d: LayerDescriptor) -> &mut Self {
        self.layers.insert(d.id.clone(), d);
        self
    }

    /// Proyección por ID.
    pub fn projection(&self, id: &str) -> Option<&ProjectionDescriptor> {
        self.projections.get(id)
    }

    /// Capa por ID.
    pub fn layer(&self, id: &str) -> Option<&LayerDescriptor> {
        self.layers.get(id)
    }

    /// Catálogo incorporado: `scene2d` ([`crate::project_snapshot_to_scene`]) y `decision_overlay`
    /// ([`crate::project_decision_overlay`]) sobre las capas de [`LayerId`].
    pub fn builtin() -> Self {
        fn num(name: &str, unit: Option<ExprUnit>, min: f64, max: Option<f64>) -> ParamDescriptor {
            ParamDescriptor {
                name: name.to_owned(),
                kind: ParamKind::Number {
                    min: Some(min),
                    max,
                },
                unit,
                required: false,
            }
        }
        let opacity = || num("opacity", Some(ExprUnit::Ratio), 0.0, Some(1.0));
        let mut c = Self::new();
        let layer = |id: LayerId, params: Vec<ParamDescriptor>| LayerDescriptor {
            id: id.canonical_name(),
            params,
        };
        c.register_layer(layer(LayerId::Environment, vec![opacity()]));
        c.register_layer(layer(
            LayerId::Sensors,
            vec![
                opacity(),
                num("ray_length", Some(ExprUnit::Meters), 0.0, None),
            ],
        ));
        c.register_layer(layer(
            LayerId::Imagination,
            vec![
                opacity(),
                ParamDescriptor {
                    name: "horizon_ticks".to_owned(),
                    kind: ParamKind::Integer {
                        min: Some(1),
                        max: None,
                    },
                    unit: None,
                    required: false,
                },
            ],
        ));
        c.register_layer(layer(
            LayerId::Decisions,
            vec![
                opacity(),
                num("marker_size", Some(ExprUnit::Meters), 0.0, None),
            ],
        ));
        c.register_layer(layer(
            LayerId::Labels,
            vec![
                opacity(),
                ParamDescriptor {
                    name: "show_legend".to_owned(),
                    kind: ParamKind::Bool,
                    unit: None,
                    required: false,
                },
            ],
        ));
        let all_units = vec![
            ScaleUnits::Meters,
            ScaleUnits::Centimeters,
            ScaleUnits::Millimeters,
            ScaleUnits::Kilometers,
            ScaleUnits::GridCells,
            ScaleUnits::Custom,
        ];
        let names = |ls: &[LayerId]| ls.iter().map(|l| l.canonical_name()).collect::<Vec<_>>();
        let fields = |f: &[&str]| f.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        c.register_projection(ProjectionDescriptor {
            id: "scene2d".to_owned(),
            units: all_units,
            layers: names(&[
                LayerId::Environment,
                LayerId::Sensors,
                LayerId::Imagination,
                LayerId::Decisions,
                LayerId::Labels,
            ]),
            inspector_fields: fields(&["energy", "label", "position", "velocity"]),
        });
        c.register_projection(ProjectionDescriptor {
            id: "decision_overlay".to_owned(),
            units: vec![
                ScaleUnits::Meters,
                ScaleUnits::GridCells,
                ScaleUnits::Custom,
            ],
            layers: names(&[
                LayerId::Sensors,
                LayerId::Imagination,
                LayerId::Decisions,
                LayerId::Labels,
            ]),
            inspector_fields: fields(&["decision", "label", "position"]),
        });
        c
    }
}

// ---------------------------------------------------------------------------------------------
// Modelo
// ---------------------------------------------------------------------------------------------

/// Valor de parámetro de capa.
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    /// Número con unidad opcional (JSON: `2.5` o `{"unit":"meters","value":2.5}`).
    Number {
        /// Valor.
        value: f64,
        /// Unidad declarada.
        unit: Option<ExprUnit>,
    },
    /// Entero exacto.
    Integer(i64),
    /// Booleano.
    Bool(bool),
    /// Texto.
    Text(String),
}

/// Nivel de detalle mínimo de un campo de inspector (`Minimal < Standard < Full`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InspectorLod {
    /// Mínimo.
    Minimal,
    /// Estándar.
    Standard,
    /// Completo.
    Full,
}

impl InspectorLod {
    /// Nombre canónico.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Standard => "standard",
            Self::Full => "full",
        }
    }

    fn parse(raw: &str) -> Option<Self> {
        match raw {
            "minimal" => Some(Self::Minimal),
            "standard" => Some(Self::Standard),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

/// Descriptor declarativo de un campo de inspector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InspectorFieldSpec {
    /// Nombre del campo (debe ser expuesto por la proyección).
    pub name: String,
    /// Etiqueta visible.
    pub label: String,
    /// Nivel mínimo de visibilidad.
    pub min_lod: InspectorLod,
    /// Umbral de obsolescencia en ticks.
    pub stale_after: Option<u64>,
}

/// Configuración declarativa del inspector de una vista.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct InspectorSpec {
    /// Campos, en orden de presentación.
    pub fields: Vec<InspectorFieldSpec>,
}

/// Uso de una capa dentro de una vista.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBinding {
    /// ID de capa del catálogo.
    pub layer: String,
    /// Visibilidad inicial.
    pub visible: bool,
    /// Nodo del `SystemSpec` que alimenta la capa (opcional).
    pub source_node: Option<String>,
    /// Parámetros por nombre.
    pub params: BTreeMap<String, ParamValue>,
}

impl LayerBinding {
    /// Capa visible sin parámetros.
    pub fn new(layer: impl Into<String>) -> Self {
        Self {
            layer: layer.into(),
            visible: true,
            source_node: None,
            params: BTreeMap::new(),
        }
    }

    /// Añade un parámetro.
    pub fn with_param(mut self, name: impl Into<String>, value: ParamValue) -> Self {
        self.params.insert(name.into(), value);
        self
    }

    /// Fija el nodo de origen.
    pub fn with_source_node(mut self, node: impl Into<String>) -> Self {
        self.source_node = Some(node.into());
        self
    }
}

/// Vista: una proyección con sus capas e inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct ViewSpec {
    /// ID único de la vista.
    pub id: String,
    /// ID de proyección del catálogo.
    pub projection: String,
    /// Unidades de coordenadas.
    pub units: ScaleUnits,
    /// Capas en orden de declaración.
    pub layers: Vec<LayerBinding>,
    /// Inspector opcional.
    pub inspector: Option<InspectorSpec>,
}

impl ViewSpec {
    /// Vista vacía.
    pub fn new(id: impl Into<String>, projection: impl Into<String>, units: ScaleUnits) -> Self {
        Self {
            id: id.into(),
            projection: projection.into(),
            units,
            layers: Vec::new(),
            inspector: None,
        }
    }
}

/// Especificación declarativa portable de visualización.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizationSpec {
    /// Versión de esquema.
    pub schema_version: String,
    /// `spec_id` del `SystemSpec` al que se asocia.
    pub system_spec_id: String,
    /// Vistas.
    pub views: Vec<ViewSpec>,
}

impl VisualizationSpec {
    /// Especificación vacía con la versión de esquema vigente.
    pub fn new(system_spec_id: impl Into<String>) -> Self {
        Self {
            schema_version: VISUALIZATION_SPEC_SCHEMA_VERSION.to_owned(),
            system_spec_id: system_spec_id.into(),
            views: Vec::new(),
        }
    }

    /// Valida contra el catálogo. Vacío = válida. Orden determinista por (ruta, código).
    pub fn validate(&self, catalog: &VisualizationCatalog) -> Vec<VizDiagnostic> {
        self.validate_inner(catalog, None)
    }

    /// Como [`Self::validate`] y además comprueba `system_spec_id` y `source_node` contra `system`.
    pub fn validate_against_system(
        &self,
        catalog: &VisualizationCatalog,
        system: &SystemSpec,
    ) -> Vec<VizDiagnostic> {
        self.validate_inner(catalog, Some(system))
    }

    fn validate_inner(
        &self,
        catalog: &VisualizationCatalog,
        system: Option<&SystemSpec>,
    ) -> Vec<VizDiagnostic> {
        let mut out = Vec::new();
        let mut push =
            |path: String, kind: VizDiagnosticKind| out.push(VizDiagnostic { path, kind });
        if let Some(sys) = system
            && sys.spec_id != self.system_spec_id
        {
            push(
                "system_spec_id".to_owned(),
                VizDiagnosticKind::SystemRefMismatch {
                    expected: sys.spec_id.clone(),
                    found: self.system_spec_id.clone(),
                },
            );
        }
        let mut view_ids = BTreeSet::new();
        for (vi, view) in self.views.iter().enumerate() {
            let vp = format!("views[{vi}]");
            if view.id.trim().is_empty() {
                push(format!("{vp}.id"), VizDiagnosticKind::EmptyId);
            } else if !view_ids.insert(view.id.as_str()) {
                push(
                    format!("{vp}.id"),
                    VizDiagnosticKind::DuplicateId {
                        id: view.id.clone(),
                    },
                );
            }
            let projection = catalog.projection(&view.projection);
            match projection {
                None => push(
                    format!("{vp}.projection"),
                    VizDiagnosticKind::UnregisteredProjection {
                        projection: view.projection.clone(),
                    },
                ),
                Some(p) if !p.units.contains(&view.units) => push(
                    format!("{vp}.units"),
                    VizDiagnosticKind::UnsupportedUnits {
                        projection: p.id.clone(),
                        units: view.units.as_str(),
                    },
                ),
                Some(_) => {}
            }
            for (li, binding) in view.layers.iter().enumerate() {
                let lp = format!("{vp}.layers[{li}]");
                let Some(layer) = catalog.layer(&binding.layer) else {
                    push(
                        format!("{lp}.layer"),
                        VizDiagnosticKind::UnknownLayer {
                            layer: binding.layer.clone(),
                        },
                    );
                    continue;
                };
                if let Some(p) = projection
                    && !p.layers.contains(&binding.layer)
                {
                    push(
                        format!("{lp}.layer"),
                        VizDiagnosticKind::LayerNotInProjection {
                            projection: p.id.clone(),
                            layer: binding.layer.clone(),
                        },
                    );
                }
                if let (Some(sys), Some(node)) = (system, &binding.source_node)
                    && !sys.nodes.iter().any(|n| n.id.as_str() == node)
                {
                    push(
                        format!("{lp}.source_node"),
                        VizDiagnosticKind::UnknownNodeRef { node: node.clone() },
                    );
                }
                for (name, value) in &binding.params {
                    let pp = format!("{lp}.params.{name}");
                    match layer.params.iter().find(|d| &d.name == name) {
                        None => push(
                            pp,
                            VizDiagnosticKind::UnknownParameter {
                                layer: layer.id.clone(),
                                param: name.clone(),
                            },
                        ),
                        Some(d) => check_param(&pp, d, value, &mut push),
                    }
                }
                for d in layer.params.iter().filter(|d| d.required) {
                    if !binding.params.contains_key(&d.name) {
                        push(
                            format!("{lp}.params"),
                            VizDiagnosticKind::MissingParameter {
                                param: d.name.clone(),
                            },
                        );
                    }
                }
            }
            if let Some(inspector) = &view.inspector {
                let mut names = BTreeSet::new();
                for (fi, f) in inspector.fields.iter().enumerate() {
                    let fp = format!("{vp}.inspector.fields[{fi}]");
                    if f.name.trim().is_empty() {
                        push(format!("{fp}.name"), VizDiagnosticKind::EmptyId);
                    } else if !names.insert(f.name.as_str()) {
                        push(
                            format!("{fp}.name"),
                            VizDiagnosticKind::DuplicateId { id: f.name.clone() },
                        );
                    } else if let Some(p) = projection
                        && !p.inspector_fields.contains(&f.name)
                    {
                        push(
                            format!("{fp}.name"),
                            VizDiagnosticKind::UnknownInspectorField {
                                projection: p.id.clone(),
                                field: f.name.clone(),
                            },
                        );
                    }
                }
            }
        }
        out.sort_by(|a, b| (&a.path, a.kind.code()).cmp(&(&b.path, b.kind.code())));
        out
    }

    /// Valida y devuelve `Err(diagnósticos)` si hay alguno.
    pub fn check(&self, catalog: &VisualizationCatalog) -> Result<(), Vec<VizDiagnostic>> {
        let d = self.validate(catalog);
        if d.is_empty() { Ok(()) } else { Err(d) }
    }

    /// Serializa a JSON canónico (claves ordenadas, `f64` canónico).
    pub fn to_canonical_json(&self) -> Result<String, VizSpecError> {
        let mut out = String::new();
        write_viz(&mut out, self)?;
        Ok(out)
    }

    /// Hash FNV-1a de 64 bits (hex) de los bytes canónicos; estable entre Rust y goldens.
    pub fn canonical_hash(&self) -> Result<String, VizSpecError> {
        Ok(fnv1a_hex(self.to_canonical_json()?.as_bytes()))
    }

    /// Parsea JSON estricto (campos desconocidos rechazados).
    pub fn from_json(json: &str) -> Result<Self, VizSpecError> {
        Self::from_value(&parse_json(json)?)
    }

    /// Parsea desde un [`JsonValue`] ya leído.
    pub fn from_value(value: &JsonValue) -> Result<Self, VizSpecError> {
        parse_viz(value)
    }
}

fn fnv1a_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{h:016x}")
}

fn check_param(
    path: &str,
    d: &ParamDescriptor,
    value: &ParamValue,
    push: &mut impl FnMut(String, VizDiagnosticKind),
) {
    let type_err = |push: &mut dyn FnMut(String, VizDiagnosticKind)| {
        push(
            path.to_owned(),
            VizDiagnosticKind::ParameterType {
                param: d.name.clone(),
                expected: d.kind.name(),
            },
        );
    };
    let range_err = |push: &mut dyn FnMut(String, VizDiagnosticKind)| {
        push(
            path.to_owned(),
            VizDiagnosticKind::ParameterOutOfRange {
                param: d.name.clone(),
            },
        );
    };
    match (&d.kind, value) {
        (ParamKind::Number { min, max }, ParamValue::Number { value: v, unit }) => {
            check_unit(path, d, *unit, push);
            if !v.is_finite() || min.is_some_and(|m| *v < m) || max.is_some_and(|m| *v > m) {
                range_err(push);
            }
        }
        (ParamKind::Number { min, max }, ParamValue::Integer(i)) => {
            check_unit(path, d, None, push);
            let v = *i as f64;
            if min.is_some_and(|m| v < m) || max.is_some_and(|m| v > m) {
                range_err(push);
            }
        }
        (ParamKind::Integer { min, max }, ParamValue::Integer(i)) => {
            if min.is_some_and(|m| *i < m) || max.is_some_and(|m| *i > m) {
                range_err(push);
            }
        }
        (ParamKind::Bool, ParamValue::Bool(_)) | (ParamKind::Text, ParamValue::Text(_)) => {}
        _ => type_err(push),
    }
}

fn check_unit(
    path: &str,
    d: &ParamDescriptor,
    found: Option<ExprUnit>,
    push: &mut impl FnMut(String, VizDiagnosticKind),
) {
    let param = d.name.clone();
    match (d.unit, found) {
        (Some(e), Some(f)) if e != f => push(
            path.to_owned(),
            VizDiagnosticKind::UnitMismatch {
                param,
                expected: e.as_str(),
                found: f.as_str(),
            },
        ),
        (Some(e), None) if e != ExprUnit::Dimensionless && e != ExprUnit::Ratio => push(
            path.to_owned(),
            VizDiagnosticKind::MissingUnit {
                param,
                expected: e.as_str(),
            },
        ),
        (None, Some(f)) if f != ExprUnit::Dimensionless => push(
            path.to_owned(),
            VizDiagnosticKind::UnitMismatch {
                param,
                expected: ExprUnit::Dimensionless.as_str(),
                found: f.as_str(),
            },
        ),
        _ => {}
    }
}

// ---------------------------------------------------------------------------------------------
// Sección opcional sobre SystemSpec
// ---------------------------------------------------------------------------------------------

/// Clave JSON de la sección opcional dentro de un documento `SystemSpec`.
pub const VISUALIZATION_SECTION_KEY: &str = "visualization";

/// `SystemSpec` + sección `visualization` opcional del mismo documento JSON.
///
/// El plan headless usa únicamente [`Self::system`]; la sección nunca entra en el JSON canónico
/// del `SystemSpec`, por lo que hashes/fingerprints/artefactos no cambian.
#[derive(Debug, Clone, PartialEq)]
pub struct SystemWithVisualization {
    /// Sistema headless.
    pub system: SystemSpec,
    /// Sección gráfica opcional.
    pub visualization: Option<VisualizationSpec>,
}

impl SystemWithVisualization {
    /// Parsea un documento `SystemSpec` que puede incluir la clave `visualization`.
    pub fn from_json(json: &str) -> Result<Self, VizSpecError> {
        let root = parse_json(json)?;
        let visualization = match &root {
            JsonValue::Object(map) => map
                .get(VISUALIZATION_SECTION_KEY)
                .map(|v| prefix_path(VISUALIZATION_SECTION_KEY, VisualizationSpec::from_value(v)))
                .transpose()?,
            _ => None,
        };
        let system = SystemSpec::from_json_with_policy(json, UnknownFieldPolicy::AllowIgnored)?;
        Ok(Self {
            system,
            visualization,
        })
    }

    /// JSON canónico: el del `SystemSpec` con `,"visualization":{..}` al final (orden de claves).
    pub fn to_canonical_json(&self) -> Result<String, VizSpecError> {
        let mut out = self.system.to_canonical_json()?;
        if let Some(v) = &self.visualization {
            debug_assert!(out.ends_with('}'));
            out.pop();
            out.push_str(",\"visualization\":");
            write_viz(&mut out, v)?;
            out.push('}');
        }
        Ok(out)
    }
}

fn prefix_path<T>(prefix: &str, r: Result<T, VizSpecError>) -> Result<T, VizSpecError> {
    r.map_err(|e| match e {
        VizSpecError::Field { path, reason } => VizSpecError::Field {
            path: if path.is_empty() {
                prefix.to_owned()
            } else {
                format!("{prefix}.{path}")
            },
            reason,
        },
        other => other,
    })
}

// ---------------------------------------------------------------------------------------------
// Serialización canónica
// ---------------------------------------------------------------------------------------------

fn write_viz(out: &mut String, v: &VisualizationSpec) -> Result<(), VizSpecError> {
    out.push_str("{\"schema_version\":");
    write_json_string(out, &v.schema_version);
    out.push_str(",\"system_spec_id\":");
    write_json_string(out, &v.system_spec_id);
    out.push_str(",\"views\":[");
    for (i, view) in v.views.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        write_view(out, view, &format!("views[{i}]"))?;
    }
    out.push_str("]}");
    Ok(())
}

fn write_view(out: &mut String, v: &ViewSpec, path: &str) -> Result<(), VizSpecError> {
    out.push_str("{\"id\":");
    write_json_string(out, &v.id);
    if let Some(ins) = &v.inspector {
        out.push_str(",\"inspector\":{\"fields\":[");
        for (i, f) in ins.fields.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"label\":");
            write_json_string(out, &f.label);
            out.push_str(",\"min_lod\":");
            write_json_string(out, f.min_lod.as_str());
            out.push_str(",\"name\":");
            write_json_string(out, &f.name);
            if let Some(s) = f.stale_after {
                out.push_str(&format!(",\"stale_after\":{s}"));
            }
            out.push('}');
        }
        out.push_str("]}");
    }
    out.push_str(",\"layers\":[");
    for (i, l) in v.layers.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"layer\":");
        write_json_string(out, &l.layer);
        out.push_str(",\"params\":{");
        for (j, (name, value)) in l.params.iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            write_json_string(out, name);
            out.push(':');
            let pp = format!("{path}.layers[{i}].params.{name}");
            write_param(out, value, &pp)?;
        }
        out.push('}');
        if let Some(n) = &l.source_node {
            out.push_str(",\"source_node\":");
            write_json_string(out, n);
        }
        out.push_str(",\"visible\":");
        out.push_str(if l.visible { "true" } else { "false" });
        out.push('}');
    }
    out.push_str("],\"projection\":");
    write_json_string(out, &v.projection);
    out.push_str(",\"units\":");
    write_json_string(out, v.units.as_str());
    out.push('}');
    Ok(())
}

fn write_param(out: &mut String, v: &ParamValue, path: &str) -> Result<(), VizSpecError> {
    let num = |out: &mut String, x: f64| {
        write_canonical_f64(out, x).map_err(|_| field_err(path, "non-finite number"))
    };
    match v {
        ParamValue::Number { value, unit: None } => num(out, *value)?,
        ParamValue::Number {
            value,
            unit: Some(u),
        } => {
            out.push_str("{\"unit\":");
            write_json_string(out, u.as_str());
            out.push_str(",\"value\":");
            num(out, *value)?;
            out.push('}');
        }
        ParamValue::Integer(i) => out.push_str(&i.to_string()),
        ParamValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        ParamValue::Text(s) => write_json_string(out, s),
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Parseo estricto
// ---------------------------------------------------------------------------------------------

type Obj = BTreeMap<String, JsonValue>;

fn join(base: &str, key: &str) -> String {
    if base.is_empty() {
        key.to_owned()
    } else {
        format!("{base}.{key}")
    }
}

fn as_obj<'a>(v: &'a JsonValue, path: &str) -> Result<&'a Obj, VizSpecError> {
    match v {
        JsonValue::Object(m) => Ok(m),
        _ => Err(field_err(path, "expected object")),
    }
}

fn as_arr<'a>(v: &'a JsonValue, path: &str) -> Result<&'a Vec<JsonValue>, VizSpecError> {
    match v {
        JsonValue::Array(a) => Ok(a),
        _ => Err(field_err(path, "expected array")),
    }
}

fn as_string(v: &JsonValue, path: &str) -> Result<String, VizSpecError> {
    match v {
        JsonValue::String(s) => Ok(s.clone()),
        _ => Err(field_err(path, "expected string")),
    }
}

fn reject_unknown(m: &Obj, allowed: &[&str], path: &str) -> Result<(), VizSpecError> {
    match m.keys().find(|k| !allowed.contains(&k.as_str())) {
        Some(k) => Err(field_err(&join(path, k), "unknown field")),
        None => Ok(()),
    }
}

fn req<'a>(m: &'a Obj, key: &str, path: &str) -> Result<&'a JsonValue, VizSpecError> {
    m.get(key)
        .ok_or_else(|| field_err(&join(path, key), "missing required field"))
}

fn req_string(m: &Obj, key: &str, path: &str) -> Result<String, VizSpecError> {
    as_string(req(m, key, path)?, &join(path, key))
}

fn parse_viz(v: &JsonValue) -> Result<VisualizationSpec, VizSpecError> {
    let m = as_obj(v, "")?;
    reject_unknown(m, &["schema_version", "system_spec_id", "views"], "")?;
    let schema_version = req_string(m, "schema_version", "")?;
    if schema_version != VISUALIZATION_SPEC_SCHEMA_VERSION {
        return Err(field_err(
            "schema_version",
            format!("expected '{VISUALIZATION_SPEC_SCHEMA_VERSION}', found '{schema_version}'"),
        ));
    }
    let system_spec_id = req_string(m, "system_spec_id", "")?;
    let mut views = Vec::new();
    for (i, item) in as_arr(req(m, "views", "")?, "views")?.iter().enumerate() {
        views.push(parse_view(item, &format!("views[{i}]"))?);
    }
    Ok(VisualizationSpec {
        schema_version,
        system_spec_id,
        views,
    })
}

fn parse_view(v: &JsonValue, path: &str) -> Result<ViewSpec, VizSpecError> {
    let m = as_obj(v, path)?;
    reject_unknown(
        m,
        &["id", "inspector", "layers", "projection", "units"],
        path,
    )?;
    let units_raw = req_string(m, "units", path)?;
    let units = [
        ScaleUnits::Meters,
        ScaleUnits::Centimeters,
        ScaleUnits::Millimeters,
        ScaleUnits::Kilometers,
        ScaleUnits::GridCells,
        ScaleUnits::Custom,
    ]
    .into_iter()
    .find(|u| u.as_str() == units_raw)
    .ok_or_else(|| field_err(&join(path, "units"), format!("unknown units '{units_raw}'")))?;
    let mut layers = Vec::new();
    let lpath = join(path, "layers");
    for (i, item) in as_arr(req(m, "layers", path)?, &lpath)?.iter().enumerate() {
        layers.push(parse_layer(item, &format!("{lpath}[{i}]"))?);
    }
    let inspector = m
        .get("inspector")
        .map(|v| parse_inspector(v, &join(path, "inspector")))
        .transpose()?;
    Ok(ViewSpec {
        id: req_string(m, "id", path)?,
        projection: req_string(m, "projection", path)?,
        units,
        layers,
        inspector,
    })
}

fn parse_layer(v: &JsonValue, path: &str) -> Result<LayerBinding, VizSpecError> {
    let m = as_obj(v, path)?;
    reject_unknown(m, &["layer", "params", "source_node", "visible"], path)?;
    let visible = match m.get("visible") {
        None => true,
        Some(JsonValue::Bool(b)) => *b,
        Some(_) => return Err(field_err(&join(path, "visible"), "expected bool")),
    };
    let source_node = m
        .get("source_node")
        .map(|v| as_string(v, &join(path, "source_node")))
        .transpose()?;
    let mut params = BTreeMap::new();
    if let Some(p) = m.get("params") {
        let ppath = join(path, "params");
        for (k, pv) in as_obj(p, &ppath)? {
            params.insert(k.clone(), parse_param(pv, &join(&ppath, k))?);
        }
    }
    Ok(LayerBinding {
        layer: req_string(m, "layer", path)?,
        visible,
        source_node,
        params,
    })
}

fn parse_param(v: &JsonValue, path: &str) -> Result<ParamValue, VizSpecError> {
    Ok(match v {
        JsonValue::Number(JsonNumber::F64(f)) => ParamValue::Number {
            value: *f,
            unit: None,
        },
        JsonValue::Number(JsonNumber::I64(i)) => ParamValue::Integer(*i),
        JsonValue::Number(JsonNumber::U64(u)) => ParamValue::Integer(
            i64::try_from(*u).map_err(|_| field_err(path, "integer out of range"))?,
        ),
        JsonValue::Bool(b) => ParamValue::Bool(*b),
        JsonValue::String(s) => ParamValue::Text(s.clone()),
        JsonValue::Object(m) => {
            reject_unknown(m, &["unit", "value"], path)?;
            let upath = join(path, "unit");
            let unit = ExprUnit::parse(&req_string(m, "unit", path)?)
                .map_err(|e| field_err(&upath, e.to_string()))?;
            let vpath = join(path, "value");
            let value = match req(m, "value", path)? {
                JsonValue::Number(JsonNumber::F64(f)) => *f,
                JsonValue::Number(JsonNumber::I64(i)) => *i as f64,
                JsonValue::Number(JsonNumber::U64(u)) => *u as f64,
                _ => return Err(field_err(&vpath, "expected number")),
            };
            ParamValue::Number {
                value,
                unit: Some(unit),
            }
        }
        _ => return Err(field_err(path, "unsupported parameter value")),
    })
}

fn parse_inspector(v: &JsonValue, path: &str) -> Result<InspectorSpec, VizSpecError> {
    let m = as_obj(v, path)?;
    reject_unknown(m, &["fields"], path)?;
    let fpath = join(path, "fields");
    let mut fields = Vec::new();
    for (i, item) in as_arr(req(m, "fields", path)?, &fpath)?.iter().enumerate() {
        let p = format!("{fpath}[{i}]");
        let f = as_obj(item, &p)?;
        reject_unknown(f, &["label", "min_lod", "name", "stale_after"], &p)?;
        let lod_raw = req_string(f, "min_lod", &p)?;
        let min_lod = InspectorLod::parse(&lod_raw)
            .ok_or_else(|| field_err(&join(&p, "min_lod"), format!("unknown lod '{lod_raw}'")))?;
        let stale_after = match f.get("stale_after") {
            None => None,
            Some(JsonValue::Number(JsonNumber::U64(u))) => Some(*u),
            Some(JsonValue::Number(JsonNumber::I64(i))) if *i >= 0 => Some(*i as u64),
            Some(_) => {
                return Err(field_err(
                    &join(&p, "stale_after"),
                    "expected unsigned integer",
                ));
            }
        };
        fields.push(InspectorFieldSpec {
            name: req_string(f, "name", &p)?,
            label: req_string(f, "label", &p)?,
            min_lod,
            stale_after,
        });
    }
    Ok(InspectorSpec { fields })
}

// ---------------------------------------------------------------------------------------------
// Fixture canónico (espejo del builder Kotlin y del golden)
// ---------------------------------------------------------------------------------------------

/// Fixture canónico: vista `scene2d` + vista `decision_overlay` sobre `two_node_canonical_v0`.
pub fn canonical_visualization_fixture() -> VisualizationSpec {
    let mut main = ViewSpec::new("main", "scene2d", ScaleUnits::Meters);
    main.layers
        .push(LayerBinding::new("environment").with_param(
            "opacity",
            ParamValue::Number {
                value: 1.0,
                unit: None,
            },
        ));
    main.layers.push(
        LayerBinding::new("sensors")
            .with_source_node("sensor")
            .with_param(
                "ray_length",
                ParamValue::Number {
                    value: 2.5,
                    unit: Some(ExprUnit::Meters),
                },
            ),
    );
    main.inspector = Some(InspectorSpec {
        fields: vec![
            InspectorFieldSpec {
                name: "position".to_owned(),
                label: "Position".to_owned(),
                min_lod: InspectorLod::Minimal,
                stale_after: None,
            },
            InspectorFieldSpec {
                name: "energy".to_owned(),
                label: "Energy".to_owned(),
                min_lod: InspectorLod::Full,
                stale_after: Some(10),
            },
        ],
    });
    let mut dec = ViewSpec::new("decisions", "decision_overlay", ScaleUnits::GridCells);
    let mut hidden =
        LayerBinding::new("imagination").with_param("horizon_ticks", ParamValue::Integer(8));
    hidden.visible = false;
    dec.layers.push(hidden);
    dec.layers
        .push(LayerBinding::new("labels").with_param("show_legend", ParamValue::Bool(true)));
    let mut spec = VisualizationSpec::new("two_node_canonical_v0");
    spec.views.push(main);
    spec.views.push(dec);
    spec
}

/// JSON canónico del fixture ([`canonical_visualization_fixture`]).
pub fn canonical_visualization_fixture_json() -> String {
    canonical_visualization_fixture()
        .to_canonical_json()
        .expect("canonical visualization fixture must serialize")
}
