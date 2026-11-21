//! Estado de pruebas de render: distingue PASS gráfico real de "no ejecutado".

use crate::viewport::{BackendKind, ViewportBackend};

/// Resultado de una prueba de render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderTestStatus {
    Pass,
    Fail,
    /// No ejecutada: el host no tiene entorno gráfico (sin DISPLAY/WAYLAND_DISPLAY).
    NotRunHeadlessHost,
    /// No ejecutada: hay entorno gráfico pero no existe backend de ventana compilado.
    NotRunNoWindowBackend,
}

/// Informe: el estado va siempre atado al backend que lo produjo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderTestReport {
    pub backend: Option<BackendKind>,
    pub status: RenderTestStatus,
}

impl RenderTestReport {
    /// Solo un PASS de un backend `Window` es un PASS gráfico real.
    pub fn is_graphical_window_pass(&self) -> bool {
        self.backend == Some(BackendKind::Window) && self.status == RenderTestStatus::Pass
    }
}

/// `true` si `get` expone `DISPLAY` o `WAYLAND_DISPLAY` no vacíos.
pub fn has_graphical_display(get: impl Fn(&str) -> Option<String>) -> bool {
    ["DISPLAY", "WAYLAND_DISPLAY"]
        .iter()
        .any(|k| get(k).is_some_and(|v| !v.trim().is_empty()))
}

/// Harness de render en ventana. Sin entorno gráfico -> `NotRunHeadlessHost`; con entorno pero
/// sin backend `Window` -> `NotRunNoWindowBackend`. Nunca devuelve `Pass` con backend software.
pub fn run_window_render_test(
    graphical_env: bool,
    window_backend: Option<&dyn ViewportBackend>,
    check: impl Fn(&dyn ViewportBackend) -> bool,
) -> RenderTestReport {
    if !graphical_env {
        return RenderTestReport {
            backend: None,
            status: RenderTestStatus::NotRunHeadlessHost,
        };
    }
    match window_backend {
        Some(b) if b.kind() == BackendKind::Window => RenderTestReport {
            backend: Some(BackendKind::Window),
            status: if check(b) {
                RenderTestStatus::Pass
            } else {
                RenderTestStatus::Fail
            },
        },
        _ => RenderTestReport {
            backend: None,
            status: RenderTestStatus::NotRunNoWindowBackend,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera2D;
    use crate::fixtures::scene_alpha;
    use crate::viewport::SoftwareSvgBackend;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |k| {
            pairs
                .iter()
                .find(|(n, _)| *n == k)
                .map(|(_, v)| (*v).to_owned())
        }
    }

    #[test]
    fn detects_graphical_env() {
        assert!(!has_graphical_display(env(&[])));
        assert!(!has_graphical_display(env(&[("DISPLAY", "  ")])));
        assert!(has_graphical_display(env(&[("DISPLAY", ":0")])));
        assert!(has_graphical_display(env(&[(
            "WAYLAND_DISPLAY",
            "wayland-0"
        )])));
    }

    #[test]
    fn headless_reports_not_run_never_pass() {
        let sw = SoftwareSvgBackend;
        let r = run_window_render_test(false, Some(&sw), |_| true);
        assert_eq!(r.status, RenderTestStatus::NotRunHeadlessHost);
        assert_ne!(r.status, RenderTestStatus::Pass);
        assert!(!r.is_graphical_window_pass());
    }

    #[test]
    fn software_backend_is_never_a_window_pass() {
        let sw = SoftwareSvgBackend;
        // incluso con entorno gráfico, un backend software no produce PASS de ventana
        let r = run_window_render_test(true, Some(&sw), |_| true);
        assert_eq!(r.status, RenderTestStatus::NotRunNoWindowBackend);
        assert!(!r.is_graphical_window_pass());
        let r = run_window_render_test(true, None, |_| true);
        assert_ne!(r.status, RenderTestStatus::Pass);
    }

    #[test]
    fn software_self_check_passes_but_is_not_graphical() {
        let sw = SoftwareSvgBackend;
        let cam = Camera2D::new(100.0, 100.0, true);
        let ok = sw.render(&scene_alpha(), &cam) == sw.render(&scene_alpha(), &cam);
        let report = RenderTestReport {
            backend: Some(sw.kind()),
            status: if ok {
                RenderTestStatus::Pass
            } else {
                RenderTestStatus::Fail
            },
        };
        assert_eq!(report.status, RenderTestStatus::Pass);
        assert!(!report.is_graphical_window_pass());
    }

    struct FakeWindow;
    impl ViewportBackend for FakeWindow {
        fn kind(&self) -> BackendKind {
            BackendKind::Window
        }
        fn name(&self) -> &'static str {
            "fake-window"
        }
        fn render(&self, _: &gogamen_projection::Scene2D, _: &Camera2D) -> Vec<u8> {
            Vec::new()
        }
    }

    #[test]
    fn window_backend_pass_and_fail_paths() {
        let w = FakeWindow;
        assert!(run_window_render_test(true, Some(&w), |_| true).is_graphical_window_pass());
        let f = run_window_render_test(true, Some(&w), |_| false);
        assert_eq!(f.status, RenderTestStatus::Fail);
        assert_ne!(RenderTestStatus::NotRunHeadlessHost, RenderTestStatus::Pass);
    }
}
