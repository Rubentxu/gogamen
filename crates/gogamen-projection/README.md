# gogamen-projection

SDK de proyección 2D independiente del renderer (`std` + `gogamen-core` + `gogamen-spec`; sin Bevy/egui/wgpu).

## VisualizationSpec declarativa (GOG-102)

`viz_spec` define vistas, capas e inspector por IDs de catálogo (`gogamen.visualization_spec.v0`):

- `VisualizationSpec { system_spec_id, views[] }`; cada `ViewSpec` = proyección + unidades (`ScaleUnits`) + capas
  (`LayerBinding` con parámetros tipados y unidades `ExprUnit`) + inspector opcional (`InspectorSpec`).
- `VisualizationCatalog::builtin()` registra `scene2d` y `decision_overlay` con sus capas y descriptores de parámetros.
- `validate` / `validate_against_system` devuelven `VizDiagnostic { path, kind }` con ruta tipo JSON-pointer
  (`views[0].layers[1].layer`), ordenados por (ruta, código). Parseo estricto con `VizSpecError::Field { path }`.
- JSON canónico y hash (`canonical_hash`, FNV-1a 64) compartidos con el builder Kotlin
  (`kotlin/.../Visualization.kt`) mediante goldens en `fixtures/`; Rust los verifica sin JVM.
- Sección opcional headless: `SystemWithVisualization` lee la clave `visualization` de un documento `SystemSpec`;
  el plan/artefacto compilado sólo usa `system`, por lo que hashes y fingerprints no cambian con o sin sección.
