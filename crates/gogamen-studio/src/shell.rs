//! Shell de Studio: selector de labs sobre un registro estático con una única sesión activa.

use std::fmt;
use std::sync::Arc;

use gogamen_projection::Scene2D;
use gogamen_sdk::extensions::ExtensionId;

use crate::camera::{Camera2D, scene_bounds};
use crate::control::{
    ControlHandle, ControlUnavailable, Observation, ObservationHandle, ObservationPort,
    SessionHost, SessionOptions,
};
use crate::params::ParamSchema;
use crate::viewport::SoftwareSvgBackend;

/// Graphic input forwarded to a screen-space session (e.g. the 3D lab). View-only: it can never
/// reach the physics of a lab.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ViewInput {
    /// Orbit by a pointer drag in pixels.
    Orbit { dx_px: f64, dy_px: f64 },
    /// Pan by a pointer drag in pixels.
    Pan { dx_px: f64, dy_px: f64 },
    /// Zoom by a magnification factor (`> 1` zooms in).
    Zoom { factor: f64 },
    /// Click at a viewport pixel (selection).
    Click { x: f64, y: f64 },
    /// Back to the default view.
    Reset,
    /// Switch perspective / orthographic.
    ToggleProjection,
}

/// Sesión de un lab abierta en el shell. Al soltarla se liberan todos sus recursos.
pub trait LabSession {
    fn scene(&self) -> &Scene2D;

    /// `true` if the scene is already in pixel space (the session owns its own camera, e.g. a 3D
    /// view). The shell then uses an identity [`Camera2D`] and forwards sizes and
    /// [`ViewInput`]s to the session. Default: `false`.
    fn screen_space(&self) -> bool {
        false
    }

    /// Viewport size in pixels (only called for screen-space sessions).
    fn resize(&mut self, _w: f64, _h: f64) {}

    /// Graphic input (only called for screen-space sessions).
    fn view_input(&mut self, _input: &ViewInput) {}

    /// Lado motor: alinea la simulación del lab con la observación de control del host
    /// (`tick`/`generation`). Por defecto no hace nada (labs estáticos).
    fn sync_to(&mut self, _observation: &Observation) {}
}

/// Fábrica estática (enlazada, sin carga dinámica) de sesiones.
pub type LabFactory = Arc<dyn Fn() -> Box<dyn LabSession> + Send + Sync>;

/// Entrada del registro de labs.
pub struct LabEntry {
    id: ExtensionId,
    title: String,
    factory: LabFactory,
    param_schema: ParamSchema,
}

impl LabEntry {
    pub fn new(id: ExtensionId, title: impl Into<String>, factory: LabFactory) -> Self {
        Self {
            id,
            title: title.into(),
            factory,
            param_schema: ParamSchema::default(),
        }
    }

    /// Declara el esquema de parámetros dinámicos (GOG-95) de las sesiones de este lab.
    /// Por defecto el esquema es vacío.
    pub fn with_param_schema(mut self, schema: ParamSchema) -> Self {
        self.param_schema = schema;
        self
    }

    pub fn param_schema(&self) -> &ParamSchema {
        &self.param_schema
    }
    pub fn id(&self) -> &ExtensionId {
        &self.id
    }
    pub fn title(&self) -> &str {
        &self.title
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellError {
    UnknownLab(String),
    DuplicateLab(String),
}

impl fmt::Display for ShellError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownLab(id) => write!(f, "unknown lab `{id}`"),
            Self::DuplicateLab(id) => write!(f, "duplicate lab `{id}`"),
        }
    }
}

impl std::error::Error for ShellError {}

/// Resultado tipado de [`StudioShell::select_lab_outcome`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectOutcome {
    /// Se creó una sesión nueva con las opciones pedidas.
    Selected,
    /// El lab ya estaba activo con las mismas opciones; no se hizo nada.
    Unchanged,
    /// El lab ya estaba activo con otras opciones; se conservó la sesión y no se aplicó nada.
    OptionsConflict {
        active: SessionOptions,
        requested: SessionOptions,
    },
}

struct Active {
    index: usize,
    session: Box<dyn LabSession>,
    host: SessionHost,
}

/// Shell único (no `Clone`): como máximo una sesión de lab activa.
pub struct StudioShell {
    labs: Vec<LabEntry>,
    active: Option<Active>,
    camera: Camera2D,
    padding_px: f64,
    next_epoch: u64,
    session_options: SessionOptions,
}

impl StudioShell {
    pub fn new(labs: Vec<LabEntry>, window_w: f64, window_h: f64) -> Result<Self, ShellError> {
        for (i, l) in labs.iter().enumerate() {
            if labs[..i].iter().any(|o| o.id == l.id) {
                return Err(ShellError::DuplicateLab(l.id.as_str().to_owned()));
            }
        }
        Ok(Self {
            labs,
            active: None,
            camera: Camera2D::new(window_w, window_h, true),
            padding_px: 16.0,
            next_epoch: 1,
            session_options: SessionOptions::simulated(),
        })
    }

    /// Ids disponibles en orden de registro.
    pub fn lab_ids(&self) -> Vec<&str> {
        self.labs.iter().map(|l| l.id.as_str()).collect()
    }

    pub fn active_lab_id(&self) -> Option<&str> {
        self.active.as_ref().map(|a| self.labs[a.index].id.as_str())
    }

    pub fn camera(&self) -> &Camera2D {
        &self.camera
    }

    pub fn camera_mut(&mut self) -> &mut Camera2D {
        &mut self.camera
    }

    /// Activa un lab. La sesión previa se suelta **antes** de crear la nueva; reseleccionar el
    /// lab activo es un no-op (no duplica la sesión).
    pub fn select_lab(&mut self, id: &str) -> Result<(), ShellError> {
        self.select_lab_with(id, SessionOptions::simulated())
    }

    /// Como [`Self::select_lab`] con opciones de sesión (tipo, capacidad, `stop_on_close`).
    ///
    /// Compatibilidad: delega en [`Self::select_lab_outcome`] y descarta el resultado tipado, de
    /// modo que reseleccionar el lab activo con opciones distintas sigue siendo un no-op que
    /// conserva la sesión y sus opciones. Quien necesite detectarlo debe usar
    /// [`Self::select_lab_outcome`].
    pub fn select_lab_with(&mut self, id: &str, options: SessionOptions) -> Result<(), ShellError> {
        self.select_lab_outcome(id, options).map(|_| ())
    }

    /// Opciones con las que se creó la sesión activa (`None` sin sesión).
    pub fn active_session_options(&self) -> Option<SessionOptions> {
        self.active.as_ref().map(|_| self.session_options)
    }

    /// Selección transaccional con resultado tipado ([`SelectOutcome`]).
    ///
    /// Primero se valida y se decide; sólo se muta si se crea una sesión nueva:
    /// - lab desconocido: `Err(UnknownLab)` sin tocar opciones ni sesión;
    /// - lab ya activo con las mismas opciones: `Unchanged`;
    /// - lab ya activo con opciones distintas: `OptionsConflict`, sin cambios (la sesión viva
    ///   conserva sus opciones; para aplicarlas hay que `close_lab()` y volver a seleccionar);
    /// - en otro caso se cierra la sesión previa, se crea la nueva y devuelve `Selected`.
    pub fn select_lab_outcome(
        &mut self,
        id: &str,
        options: SessionOptions,
    ) -> Result<SelectOutcome, ShellError> {
        let index = self
            .labs
            .iter()
            .position(|l| l.id.as_str() == id)
            .ok_or_else(|| ShellError::UnknownLab(id.to_owned()))?;
        if self.active.as_ref().is_some_and(|a| a.index == index) {
            return Ok(if self.session_options == options {
                SelectOutcome::Unchanged
            } else {
                SelectOutcome::OptionsConflict {
                    active: self.session_options,
                    requested: options,
                }
            });
        }
        self.session_options = options;
        self.close_lab();
        let session = (self.labs[index].factory)();
        let mut session = session;
        let (w, h) = self.camera.window();
        let cam = if session.screen_space() {
            session.resize(w, h);
            Camera2D::screen_space(w, h)
        } else {
            let mut cam = Camera2D::new(w, h, true);
            if let Some(b) = scene_bounds(session.scene()) {
                cam.fit_to_window(w, h, &b, self.padding_px);
            }
            cam
        };
        self.camera = cam;
        let host = SessionHost::with_params(
            self.next_epoch,
            self.session_options,
            self.labs[index].param_schema.clone(),
        );
        self.next_epoch += 1;
        self.active = Some(Active {
            index,
            session,
            host,
        });
        Ok(SelectOutcome::Selected)
    }

    /// Cierra la sesión activa liberando sus recursos.
    /// El motor (`SessionHost`) sólo se detiene si la sesión se creó con `stop_on_close`.
    pub fn close_lab(&mut self) {
        if let Some(a) = self.active.take() {
            a.host.close_viewer();
        }
    }

    /// Handle del lado motor de la sesión activa (no pasar a la UI).
    pub fn engine_handle(&self) -> Option<SessionHost> {
        self.active.as_ref().map(|a| a.host.clone())
    }

    /// Puerto de observación de la sesión activa (siempre disponible con sesión).
    pub fn observation_port(&self) -> Option<ObservationHandle> {
        self.active.as_ref().map(|a| a.host.observation_port())
    }

    /// Puerto de parámetros de la sesión activa (rechaza en sesiones de sólo lectura).
    pub fn param_port(&self) -> Option<crate::control::ParamHandle> {
        self.active.as_ref().map(|a| a.host.param_port())
    }

    /// Puerto de control de la sesión activa; `None` sin sesión, `Some(Err)` si es sólo lectura.
    pub fn control_port(&self) -> Option<Result<ControlHandle, ControlUnavailable>> {
        self.active.as_ref().map(|a| a.host.control_port())
    }

    /// Paso del lado motor: alinea la sesión activa con el estado del host (p. ej. tras `Step`,
    /// `Reset` o `pump`). Debe llamarse antes de renderizar para ver el estado actual.
    pub fn sync_active(&mut self) {
        if let Some(a) = self.active.as_mut() {
            let obs = a.host.observation_port().observe();
            a.session.sync_to(&obs);
        }
    }

    /// `true` if the active session owns its camera (screen-space scene, e.g. the 3D lab).
    pub fn active_is_screen_space(&self) -> bool {
        self.active
            .as_ref()
            .is_some_and(|a| a.session.screen_space())
    }

    /// Forwards a graphic input to the active screen-space session. Returns `false` (and does
    /// nothing) for ordinary 2D sessions or without a session.
    pub fn view_input(&mut self, input: ViewInput) -> bool {
        match self.active.as_mut() {
            Some(a) if a.session.screen_space() => {
                a.session.view_input(&input);
                true
            }
            _ => false,
        }
    }

    /// Escena de la sesión activa, si la hay.
    pub fn scene(&self) -> Option<&Scene2D> {
        self.active.as_ref().map(|a| a.session.scene())
    }

    /// Redimensiona la ventana y reajusta la escena activa.
    pub fn resize(&mut self, w: f64, h: f64) {
        if let Some(a) = self.active.as_mut()
            && a.session.screen_space()
        {
            a.session.resize(w, h);
            self.camera = Camera2D::screen_space(w, h);
            return;
        }
        match self
            .active
            .as_ref()
            .and_then(|a| scene_bounds(a.session.scene()))
        {
            Some(b) => self.camera.fit_to_window(w, h, &b, self.padding_px),
            None => self.camera.resize(w, h),
        }
    }

    /// Renderiza la escena activa con el backend de software (sin ventana).
    pub fn render_svg(&self) -> Option<String> {
        self.active
            .as_ref()
            .map(|a| SoftwareSvgBackend.render_svg(a.session.scene(), &self.camera))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::{fixture_entries, fixture_entry, scene_alpha, scene_beta};

    #[test]
    fn two_fixture_labs_alternate_with_different_output() {
        let mut shell = StudioShell::new(fixture_entries(), 300.0, 200.0).unwrap();
        assert_eq!(shell.lab_ids(), ["fixture.alpha", "fixture.beta"]);
        assert!(shell.render_svg().is_none());
        shell.select_lab("fixture.alpha").unwrap();
        let a = shell.render_svg().unwrap();
        shell.select_lab("fixture.beta").unwrap();
        assert_eq!(shell.active_lab_id(), Some("fixture.beta"));
        let b = shell.render_svg().unwrap();
        assert_ne!(a, b);
        shell.select_lab("fixture.alpha").unwrap();
        assert_eq!(shell.render_svg().unwrap(), a);
    }

    #[test]
    fn switching_releases_previous_session_completely() {
        let ta = Arc::new(());
        let tb = Arc::new(());
        let wa = Arc::downgrade(&ta);
        let labs = vec![
            fixture_entry("a", "A", scene_alpha, Some(ta.clone())),
            fixture_entry("b", "B", scene_beta, Some(tb.clone())),
        ];
        // las entradas retienen 1 clon cada una; la sesión activa añade otro
        let mut shell = StudioShell::new(labs, 100.0, 100.0).unwrap();
        drop(ta);
        assert_eq!(Arc::strong_count(&wa.upgrade().unwrap()), 2); // entry + upgrade temp
        let base_a = 1;
        let base_b = Arc::strong_count(&tb);
        shell.select_lab("a").unwrap();
        assert_eq!(wa.strong_count(), base_a + 1);
        shell.select_lab("b").unwrap();
        assert_eq!(wa.strong_count(), base_a, "sesión A liberada");
        assert_eq!(Arc::strong_count(&tb), base_b + 1);
        shell.select_lab("a").unwrap();
        assert_eq!(Arc::strong_count(&tb), base_b, "sesión B liberada");
        assert_eq!(wa.strong_count(), base_a + 1);
        shell.close_lab();
        assert_eq!(wa.strong_count(), base_a);
        assert!(shell.active_lab_id().is_none());
        drop(shell);
        assert_eq!(wa.strong_count(), 0, "todo liberado al soltar el shell");
    }

    #[test]
    fn reselecting_active_lab_does_not_duplicate_session() {
        let t = Arc::new(());
        let mut shell = StudioShell::new(
            vec![fixture_entry("a", "A", scene_alpha, Some(t.clone()))],
            100.0,
            100.0,
        )
        .unwrap();
        let base = Arc::strong_count(&t);
        for _ in 0..5 {
            shell.select_lab("a").unwrap();
            assert_eq!(Arc::strong_count(&t), base + 1);
        }
    }

    #[test]
    fn shell_exposes_ports_and_lab_switch_makes_old_commands_stale() {
        use crate::control::*;
        let mut shell = StudioShell::new(fixture_entries(), 100.0, 100.0).unwrap();
        assert!(shell.control_port().is_none() && shell.observation_port().is_none());
        shell.select_lab("fixture.alpha").unwrap();
        let c = shell.control_port().unwrap().unwrap();
        let o = shell.observation_port().unwrap();
        let old = o.observe().session;
        let step = |id, s| ControlCommand {
            id: CommandId(id),
            session: s,
            action: ControlAction::Step,
        };
        assert_eq!(c.submit(step(1, old)).outcome, AckOutcome::Applied);
        shell.select_lab("fixture.beta").unwrap();
        let o2 = shell.observation_port().unwrap();
        assert_ne!(o2.observe().session, old);
        // el puerto antiguo apunta a la sesión descartada: no afecta a la nueva
        c.submit(step(2, old));
        assert_eq!(o2.observe().tick, gogamen_core::Tick::ZERO);
        let c2 = shell.control_port().unwrap().unwrap();
        assert!(matches!(
            c2.submit(step(3, old)).outcome,
            AckOutcome::Rejected {
                reason: RejectReason::StaleSession { .. }
            }
        ));
        assert_eq!(o2.observe().rejection_count, 1);
    }

    #[test]
    fn shell_physical_session_read_only_and_close_keeps_engine() {
        use crate::control::*;
        let mut shell = StudioShell::new(fixture_entries(), 100.0, 100.0).unwrap();
        shell
            .select_lab_with(
                "fixture.alpha",
                SessionOptions::of_kind(SessionKind::Physical),
            )
            .unwrap();
        assert!(matches!(shell.control_port(), Some(Err(_))));
        assert!(shell.observation_port().is_some());
        let engine = shell.engine_handle().unwrap();
        shell.close_lab();
        let obs = engine.observation_port().observe();
        assert!(!obs.viewer_attached);
        assert_ne!(obs.run_state, RunState::Stopped);
    }

    #[test]
    fn errors_for_unknown_and_duplicate() {
        let mut shell = StudioShell::new(fixture_entries(), 100.0, 100.0).unwrap();
        assert_eq!(
            shell.select_lab("nope"),
            Err(ShellError::UnknownLab("nope".into()))
        );
        assert!(shell.active_lab_id().is_none());
        let dup = vec![
            fixture_entry("x", "X", scene_alpha, None),
            fixture_entry("x", "X2", scene_beta, None),
        ];
        assert!(matches!(
            StudioShell::new(dup, 1.0, 1.0),
            Err(ShellError::DuplicateLab(_))
        ));
    }

    #[test]
    fn gog_149_unknown_lab_is_transactional() {
        use crate::control::*;
        let mut shell = StudioShell::new(fixture_entries(), 100.0, 100.0).unwrap();
        let phys = SessionOptions::of_kind(SessionKind::Physical);
        for active_first in [false, true] {
            if active_first {
                shell.select_lab("fixture.alpha").unwrap();
            }
            let before = (
                shell.active_lab_id().map(str::to_owned),
                shell.active_session_options(),
                shell.session_options,
                shell.next_epoch,
                shell.camera().window(),
            );
            assert_eq!(
                shell.select_lab_with("nope", phys),
                Err(ShellError::UnknownLab("nope".into()))
            );
            let after = (
                shell.active_lab_id().map(str::to_owned),
                shell.active_session_options(),
                shell.session_options,
                shell.next_epoch,
                shell.camera().window(),
            );
            assert_eq!(before, after);
        }
        assert_eq!(shell.session_options, SessionOptions::simulated());
    }

    #[test]
    fn gog_149_reselect_returns_typed_outcome() {
        use crate::control::*;
        let mut shell = StudioShell::new(fixture_entries(), 100.0, 100.0).unwrap();
        let sim = SessionOptions::simulated();
        let phys = SessionOptions::of_kind(SessionKind::Physical);
        assert_eq!(
            shell.select_lab_outcome("fixture.alpha", sim),
            Ok(SelectOutcome::Selected)
        );
        assert_eq!(
            shell.select_lab_outcome("fixture.alpha", sim),
            Ok(SelectOutcome::Unchanged)
        );
        assert_eq!(
            shell.select_lab_outcome("fixture.alpha", phys),
            Ok(SelectOutcome::OptionsConflict {
                active: sim,
                requested: phys
            })
        );
        // Conflict leaves the live session and its options untouched.
        assert_eq!(shell.active_session_options(), Some(sim));
        assert!(matches!(shell.control_port(), Some(Ok(_))));
        // Legacy wrapper keeps no-op semantics.
        assert_eq!(shell.select_lab_with("fixture.alpha", phys), Ok(()));
        assert_eq!(shell.active_session_options(), Some(sim));
        // Explicit close + reselect applies the new options.
        shell.close_lab();
        assert_eq!(
            shell.select_lab_outcome("fixture.alpha", phys),
            Ok(SelectOutcome::Selected)
        );
        assert_eq!(shell.active_session_options(), Some(phys));
    }

    #[test]
    fn resize_refits_active_scene() {
        let mut shell = StudioShell::new(fixture_entries(), 300.0, 200.0).unwrap();
        shell.select_lab("fixture.beta").unwrap();
        shell.resize(600.0, 100.0);
        assert_eq!(shell.camera().window(), (600.0, 100.0));
        let (x, y) = shell.camera().world_to_screen(-5.0, -2.0);
        assert!((0.0..=600.0).contains(&x) && (0.0..=100.0).contains(&y));
    }
}
