//! Spike GOG-109: Grid Scene2D (JSON canónico de gogamen-projection) -> frontera wasm -> JS.
//! Sin wasm-bindgen: ABI C cruda (ptr/len) para contar copias exactamente.

use gogamen_lab_grid::world::{GridWorldConfig, GridWorldEnv};
use gogamen_projection::Scene2D;
use gogamen_studio::grid_lab::{GridWorldView, project_grid_scene};
use gogamen_studio::{Camera2D, SoftwareSvgBackend, scene_bounds};

pub fn grid_scene() -> Scene2D {
    let env = GridWorldEnv::new(GridWorldConfig::default()).expect("default grid config");
    project_grid_scene(&GridWorldView::of(&env))
}

pub fn grid_json() -> String {
    grid_scene().to_canonical_json()
}

pub fn grid_svg(w: f64, h: f64) -> String {
    let scene = grid_scene();
    let mut cam = Camera2D::new(w, h, false);
    if let Some(b) = scene_bounds(&scene) {
        cam.fit_to_window(w, h, &b, 8.0);
    }
    SoftwareSvgBackend.render_svg(&scene, &cam)
}

// ---- ABI cruda ----
static mut OUT: Vec<u8> = Vec::new();

#[allow(static_mut_refs)]
fn publish(s: String) -> usize {
    unsafe {
        OUT = s.into_bytes();
        OUT.len()
    }
}

/// Genera el JSON del Grid en un buffer interno; devuelve su longitud.
#[unsafe(no_mangle)]
pub extern "C" fn grid_json_len() -> usize {
    publish(grid_json())
}

/// Genera el SVG nativo (SoftwareSvgBackend) en el buffer interno; devuelve su longitud.
#[unsafe(no_mangle)]
pub extern "C" fn grid_svg_len(w: f64, h: f64) -> usize {
    publish(grid_svg(w, h))
}

/// Puntero al buffer interno (válido hasta la siguiente llamada).
#[unsafe(no_mangle)]
#[allow(static_mut_refs)]
pub extern "C" fn out_ptr() -> *const u8 {
    unsafe { OUT.as_ptr() }
}
