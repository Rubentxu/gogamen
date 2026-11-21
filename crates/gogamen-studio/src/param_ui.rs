//! Generador puro de descriptores de control dirigido por esquema (GOG-95).
//!
//! La UI sólo recibe: esquema + permisos + valores observados. No hay referencia al motor
//! (`SessionHost`, puertos de control): los cambios se envían como [`crate::ParamCommand`] por
//! el puerto de control. No se ejecuta Kotlin ni se accede a actuadores.

use crate::params::{Mutability, ParamId, ParamKind, ParamSchema, ParamUnit, ParamValue};

/// Permisos visibles para la UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UiPermissions {
    /// `false` en sesiones de sólo lectura.
    pub session_controllable: bool,
}

/// Widget sugerido.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlWidget {
    Slider { min: f64, max: f64, step: f64 },
    Number { min: i64, max: i64, step: i64 },
    Toggle,
    Choice { options: Vec<String> },
}

/// Motivo por el que un control está deshabilitado.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisabledReason {
    ReadOnlySession,
    ImmutableParam,
}

/// Descriptor de control para la UI.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlDescriptor {
    pub id: ParamId,
    pub label: String,
    pub widget: ControlWidget,
    pub unit: ParamUnit,
    pub enabled: bool,
    pub disabled_reason: Option<DisabledReason>,
    /// Último valor observado (o el por defecto si no aparece en la observación).
    pub value: ParamValue,
}

/// Función pura: mismo esquema/permisos/valores => mismos descriptores.
pub fn control_descriptors(
    schema: &ParamSchema,
    permissions: &UiPermissions,
    observed: &[(ParamId, ParamValue)],
) -> Vec<ControlDescriptor> {
    schema
        .descriptors()
        .iter()
        .map(|d| {
            let widget = match &d.kind {
                ParamKind::Float { min, max, step } => ControlWidget::Slider {
                    min: *min,
                    max: *max,
                    step: *step,
                },
                ParamKind::Int { min, max, step } => ControlWidget::Number {
                    min: *min,
                    max: *max,
                    step: *step,
                },
                ParamKind::Bool => ControlWidget::Toggle,
                ParamKind::Enum { variants } => ControlWidget::Choice {
                    options: variants.clone(),
                },
            };
            let disabled_reason = if !permissions.session_controllable {
                Some(DisabledReason::ReadOnlySession)
            } else if d.mutability == Mutability::Immutable {
                Some(DisabledReason::ImmutableParam)
            } else {
                None
            };
            let value = observed
                .iter()
                .find(|(id, _)| id == &d.id)
                .map(|(_, v)| v.clone())
                .unwrap_or_else(|| d.default.clone());
            ControlDescriptor {
                id: d.id.clone(),
                label: d.label.clone(),
                widget,
                unit: d.unit,
                enabled: disabled_reason.is_none(),
                disabled_reason,
                value,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{ObservationPort, SessionHost, SessionKind, SessionOptions};
    use crate::params::demo_schema;

    #[test]
    fn generates_widgets_with_range_unit_step_and_enabled_flags() {
        let schema = demo_schema();
        let host = SessionHost::with_params(1, SessionOptions::simulated(), schema.clone());
        let obs = host.observation_port().observe();
        let perms = UiPermissions {
            session_controllable: obs.controllable,
        };
        let d = control_descriptors(&schema, &perms, &obs.params);
        assert_eq!(d.len(), 5);
        assert_eq!(
            d[0].widget,
            ControlWidget::Slider {
                min: 0.0,
                max: 10.0,
                step: 0.5
            }
        );
        assert_eq!(d[0].unit, ParamUnit::Newtons);
        assert!(d[0].enabled && d[0].disabled_reason.is_none());
        assert_eq!(
            d[1].widget,
            ControlWidget::Number {
                min: 1,
                max: 8,
                step: 1
            }
        );
        assert_eq!(d[2].widget, ControlWidget::Toggle);
        assert_eq!(
            d[3].widget,
            ControlWidget::Choice {
                options: vec!["eco".into(), "sport".into()]
            }
        );
        // immutable param disabled even in a controllable session
        assert!(!d[4].enabled);
        assert_eq!(d[4].disabled_reason, Some(DisabledReason::ImmutableParam));
        assert_eq!(d[4].unit, ParamUnit::Custom("kg"));
    }

    #[test]
    fn read_only_session_disables_everything() {
        let schema = demo_schema();
        let host = SessionHost::with_params(
            1,
            SessionOptions::of_kind(SessionKind::Physical),
            schema.clone(),
        );
        let obs = host.observation_port().observe();
        let perms = UiPermissions {
            session_controllable: obs.controllable,
        };
        let d = control_descriptors(&schema, &perms, &obs.params);
        assert!(d.iter().all(|c| !c.enabled));
        assert!(
            d.iter()
                .all(|c| c.disabled_reason == Some(DisabledReason::ReadOnlySession))
        );
    }

    #[test]
    fn generator_is_pure_and_takes_only_schema_permissions_and_values() {
        // The signature admits no engine/port reference: schema + permissions + observed values.
        type Gen =
            fn(&ParamSchema, &UiPermissions, &[(ParamId, ParamValue)]) -> Vec<ControlDescriptor>;
        let _f: Gen = control_descriptors;
        let schema = demo_schema();
        let perms = UiPermissions {
            session_controllable: true,
        };
        let vals = vec![(ParamId::new("gain"), ParamValue::Float(7.0))];
        let a = control_descriptors(&schema, &perms, &vals);
        let b = control_descriptors(&schema, &perms, &vals);
        assert_eq!(a, b);
        assert_eq!(a[0].value, ParamValue::Float(7.0));
        assert_eq!(a[1].value, ParamValue::Int(2), "falls back to default");
        // The module source must not reference engine-side types.
        let src = include_str!("param_ui.rs");
        let code = src.split("#[cfg(test)]").next().unwrap();
        let code_no_docs: String = code
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "SessionHost",
            "ControlHandle",
            "ControlPort",
            "ParamHandle",
            "Mutex",
        ] {
            assert!(
                !code_no_docs.contains(banned),
                "param_ui references {banned}"
            );
        }
    }
}
