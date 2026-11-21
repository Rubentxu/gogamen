//! Descriptor de dominio de un lab de terceros (API **v0**, GOG-103).
//!
//! [`LabDescriptor`] describe *qué* es un lab (identidad, nombre, versión, parámetros) y **no
//! contiene ningún tipo gráfico**: ni escenas, ni proyecciones, ni esquemas de inspector. El
//! contrato gráfico vive aparte, en `gogamen_projection::registration`, y se une al descriptor
//! sólo en el host (Studio) mediante el id del lab.
//!
//! # Estabilidad y versionado
//! La API es explícitamente **v0** ([`API_VERSION`]): puede cambiar entre versiones `0.x` sin
//! aviso de compatibilidad (véase [`crate::API_STABILITY_POLICY`]). Cualquier ruptura incrementa
//! la versión menor del crate y se reflejará en [`API_VERSION`] cuando el contrato cambie de forma.
//!
//! # Ejecutar un lab sin dependencias de UI
//! Un lab se implementa sólo contra `gogamen-sdk` (dominio) y, si quiere proyectarse,
//! `gogamen-projection` (escena 2D, `std`-only). Ninguno de los dos depende de Studio, Bevy ni
//! egui: basta `LabProjector` + `run_headless` de `gogamen_projection::registration` para
//! ejecutarlo en tests o CLI. Studio es sólo un *host* opcional.
//!
//! # Vista especializada
//! La vía de escape para vistas especializadas (`SpecializedView`) existe **únicamente** en
//! `gogamen-studio`; ni el SDK ni `gogamen-core`/`gogamen-cli` exponen tipos de vista.

use std::fmt;

use crate::extensions::ExtensionId;

/// Versión del contrato de API de labs de terceros. `v0` = inestable.
pub const API_VERSION: &str = "v0";

/// Longitud máxima de un id.
pub const MAX_ID_LEN: usize = 64;

/// Unidad explícita de un parámetro de dominio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabUnit {
    /// Adimensional.
    None,
    /// Segundos.
    Seconds,
    /// Metros.
    Meters,
    /// Radianes.
    Radians,
    /// Grados.
    Degrees,
    /// Hercios.
    Hertz,
    /// Newtons.
    Newtons,
}

impl LabUnit {
    /// Símbolo estable.
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::None => "1",
            Self::Seconds => "s",
            Self::Meters => "m",
            Self::Radians => "rad",
            Self::Degrees => "deg",
            Self::Hertz => "Hz",
            Self::Newtons => "N",
        }
    }
}

/// Parámetro escalar de dominio con rango y unidad.
#[derive(Debug, Clone, PartialEq)]
pub struct LabParam {
    /// Id estable (`[a-z][a-z0-9._-]*`).
    pub id: String,
    /// Etiqueta legible.
    pub label: String,
    /// Unidad.
    pub unit: LabUnit,
    /// Mínimo (finito).
    pub min: f64,
    /// Máximo (finito, `>= min`).
    pub max: f64,
    /// Valor por defecto (dentro de `[min, max]`).
    pub default: f64,
}

/// Descriptor de dominio de un lab: sin tipos gráficos.
#[derive(Debug, Clone, PartialEq)]
pub struct LabDescriptor {
    /// Identidad estable.
    pub id: ExtensionId,
    /// Nombre legible.
    pub name: String,
    /// Versión del lab (SemVer textual `X.Y.Z`).
    pub version: String,
    /// Parámetros de dominio.
    pub params: Vec<LabParam>,
}

/// Causa tipada y localizable (`code()` + argumentos) de un descriptor inválido.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LabDescriptorIssue {
    /// Id vacío o mal formado.
    MalformedId {
        /// Id recibido.
        id: String,
    },
    /// Nombre vacío.
    EmptyName,
    /// Versión no `X.Y.Z`.
    MalformedVersion {
        /// Versión recibida.
        version: String,
    },
    /// Id de parámetro duplicado.
    DuplicateParam {
        /// Id repetido.
        id: String,
    },
    /// Rango/default incoherentes o no finitos.
    InvalidParamRange {
        /// Id del parámetro.
        id: String,
    },
}

impl LabDescriptorIssue {
    /// Código estable para localización.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MalformedId { .. } => "lab.malformed_id",
            Self::EmptyName => "lab.empty_name",
            Self::MalformedVersion { .. } => "lab.malformed_version",
            Self::DuplicateParam { .. } => "lab.duplicate_param",
            Self::InvalidParamRange { .. } => "lab.invalid_param_range",
        }
    }
}

impl fmt::Display for LabDescriptorIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.code())?;
        match self {
            Self::MalformedId { id } => write!(f, "malformed id '{id}'"),
            Self::EmptyName => write!(f, "empty name"),
            Self::MalformedVersion { version } => write!(f, "malformed version '{version}'"),
            Self::DuplicateParam { id } => write!(f, "duplicate param '{id}'"),
            Self::InvalidParamRange { id } => write!(f, "invalid range/default for param '{id}'"),
        }
    }
}

fn well_formed_id(id: &str) -> bool {
    let mut chars = id.chars();
    id.len() <= MAX_ID_LEN
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || "._-".contains(c))
}

impl LabDescriptor {
    /// Crea un descriptor sin parámetros.
    pub fn new(id: impl Into<ExtensionId>, name: impl Into<String>, version: &str) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.to_owned(),
            params: Vec::new(),
        }
    }

    /// Añade un parámetro (builder).
    pub fn with_param(mut self, param: LabParam) -> Self {
        self.params.push(param);
        self
    }

    /// Valida el descriptor; devuelve todas las incidencias.
    pub fn check(&self) -> Result<(), Vec<LabDescriptorIssue>> {
        let mut out = Vec::new();
        if !well_formed_id(self.id.as_str()) {
            out.push(LabDescriptorIssue::MalformedId {
                id: self.id.as_str().to_owned(),
            });
        }
        if self.name.trim().is_empty() {
            out.push(LabDescriptorIssue::EmptyName);
        }
        let parts: Vec<&str> = self.version.split('.').collect();
        if parts.len() != 3
            || parts
                .iter()
                .any(|p| p.is_empty() || p.parse::<u32>().is_err())
        {
            out.push(LabDescriptorIssue::MalformedVersion {
                version: self.version.clone(),
            });
        }
        for (i, p) in self.params.iter().enumerate() {
            if self.params[..i].iter().any(|o| o.id == p.id) {
                out.push(LabDescriptorIssue::DuplicateParam { id: p.id.clone() });
            }
            let ok = well_formed_id(&p.id)
                && p.min.is_finite()
                && p.max.is_finite()
                && p.default.is_finite()
                && p.min <= p.max
                && (p.min..=p.max).contains(&p.default);
            if !ok {
                out.push(LabDescriptorIssue::InvalidParamRange { id: p.id.clone() });
            }
        }
        if out.is_empty() { Ok(()) } else { Err(out) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn param(id: &str) -> LabParam {
        LabParam {
            id: id.into(),
            label: id.into(),
            unit: LabUnit::Meters,
            min: 0.0,
            max: 2.0,
            default: 1.0,
        }
    }

    #[test]
    fn api_is_explicitly_v0() {
        assert_eq!(API_VERSION, "v0");
    }

    #[test]
    fn valid_descriptor_passes() {
        let d = LabDescriptor::new("acme.orbit", "Orbit", "0.1.0").with_param(param("radius"));
        assert_eq!(d.check(), Ok(()));
    }

    #[test]
    fn each_issue_is_reported_with_code() {
        let mut bad = param("Bad");
        bad.default = 9.0;
        let d = LabDescriptor::new("Bad Id", " ", "1.0")
            .with_param(param("p"))
            .with_param(param("p"))
            .with_param(bad);
        let codes: Vec<_> = d.check().unwrap_err().iter().map(|i| i.code()).collect();
        for c in [
            "lab.malformed_id",
            "lab.empty_name",
            "lab.malformed_version",
            "lab.duplicate_param",
            "lab.invalid_param_range",
        ] {
            assert!(codes.contains(&c), "{c} missing in {codes:?}");
        }
    }
}
