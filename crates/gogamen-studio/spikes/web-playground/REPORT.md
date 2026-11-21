# GOG-109 spike: playground web portable (informe)

Decision: **KEEP_AS_IS** para el MVP (ver rationale). El spike demuestra viabilidad técnica de
wasm + contrato Scene2D JSON + render WebGL2/Canvas2D en Chromium headless (medido, ver sección
"Navegador"). WebGPU: no utilizable en este entorno. Otros navegadores: no probados.

## Qué es
Crate aislado (`[workspace]` vacío, no miembro del workspace; no añade dependencias a core/cli) que
compila a `wasm32-unknown-unknown`, proyecta el fixture Grid (`GridWorldConfig::default()` ->
`project_grid_scene` -> `Scene2D`) y expone por ABI C cruda (sin wasm-bindgen, 0 imports):
`grid_json_len`, `grid_svg_len`, `out_ptr`. `run_node.mjs` lo carga en node y mide.
Contrato usado: `Scene2D::to_canonical_json()` existente. **Nota:** el contrato es solo
serialización; gogamen-projection no tiene parser JSON (el navegador usa `JSON.parse`).

## Reproducir
```
export CARGO_TARGET_DIR=/var/mnt/DiscoChino2-fast/cargo-targets
cd crates/gogamen-studio/spikes/web-playground
cargo build --offline --release --target wasm32-unknown-unknown
cargo run --offline --release --bin native_baseline     # escribe fixture/*.json|svg
node run_node.mjs $CARGO_TARGET_DIR/wasm32-unknown-unknown/release/gogamen_web_spike.wasm
```
Salida registrada en `fixture/node_run_output.txt`.

## Medido (node 25.9.0, release opt-level=s+lto, una sola máquina, sin repetir entre máquinas)
- `gogamen-projection` y `gogamen-studio` (con todos los labs) compilan a wasm32 offline sin cambios.
- Wasm: 119 924 B (sin strip/wasm-opt; incluye studio + lab grid + SVG). Memoria lineal 1 114 112 B
  inicial, 1 245 184 B tras los benchmarks. compile 0.9 ms, instantiate 6 ms.
- Escena Grid: 105 nodos (63 rect, 1 círculo, 41 etiquetas), JSON 17 619 B, SVG nativo 8 257 B.
- Corrección: JSON generado en wasm == JSON nativo (true); SVG wasm == SVG nativo (true) (byte a byte).
- Copias por frame (JSON): (1) wasm->JS `TextDecoder.decode` 17 619 B (la vista sobre la memoria
  wasm es cero-copia; el `String`->`Vec` en wasm es move); (2) `JSON.parse` materializa objetos JS;
  (3) draw data a `Float32Array` 1 272 B (lo que se subiría a GPU; la subida GPU NO medida).
  El contador del harness (1102 copias / 19.4 MB) incluye warmups y bucles de benchmark.
- Latencia (mediana, n=500, µs): escena+JSON en wasm 85.6; + decode 86.5; JSON.parse 141.9;
  parse+draw-data 190.4; extremo a extremo wasm->decode->parse->draw-data 272.3;
  wasm escena+SVG 155.4; SVG generado en JS desde el JSON parseado 68.2.
  Nativo (host, media de 2000): escena+JSON 75.5 µs, escena+SVG 107.1 µs.
  Máximos con picos de ms (GC/JIT) en node.
- node RSS 61.5 MB, heapUsed 6.8 MB (proceso entero, no atribuible solo al spike).
- Complejidad vs renderer nativo (`viewport.rs` 238 líneas + `camera.rs` 332): el front JS del spike
  (parse -> instancias -> SVG) son ~35 líneas, pero cubre solo rect/círculo/label, sin cámara
  interactiva, picking, polilíneas ni imágenes/texturas. Un renderer WebGL2/WebGPU real añadiría
  shaders, texto (atlas), pipelines por primitiva, gestión de contexto/fallback y toolchain JS/wasm
  (empaquetado, wasm-bindgen o ABI manual): estimación cualitativa, NO medida.

## Navegador (Playwright, Chromium headless) — medido
Reproducir: `node run_browser.mjs $CARGO_TARGET_DIR/wasm32-unknown-unknown/release/gogamen_web_spike.wasm`
(usa playwright-core de `PLAYWRIGHT_CORE` o la ruta asdf por defecto; servidor estático local efímero).
Página: `web/index.html` + `web/main.js` (175 líneas): carga el wasm, obtiene el JSON canónico de
`Scene2D` desde wasm, `JSON.parse`, dibuja con WebGL2 (rects+círculo; etiquetas en overlay Canvas2D)
y con Canvas2D completo para comparar. Salidas: `fixture/grid_chromium.png` (captura),
`fixture/browser_run_output.json` (medidas), `fixture/browser_run_stdout.txt`.

Verificación automática (16 checks, todos OK): sin errores de página; JSON wasm == fixture nativo;
105 nodos; 105 primitivas dibujadas (64 rect+círculo en GL + 41 etiquetas overlay; 105 en Canvas2D);
píxeles muestreados (readPixels en GL, getImageData en Canvas2D) = esperados: meta (34,170,34),
muro (68,68,68), celda libre (255,255,255), agente círculo (0,102,204), fondo (238,238,238);
captura PNG no en blanco (10 KB; inspeccionada visualmente: laberinto, meta "G", agente azul).

Medidas en navegador (Chromium 148.0.7778.56 headless, una máquina, renderer GL = ANGLE/Vulkan
SwiftShader, es decir GPU por software; NO hay GPU real):
- wasm instantiate (streaming) 6.6 ms; memoria lineal 1 114 112 B -> 1 245 184 B.
- Wasm 119 924 B, gzip 44 102 B; JSON 17 619 B, gzip 1 007 B (medidos con gzip local).
- Copias por frame: 1 (decode wasm->JS string, 17 619 B) + JSON.parse (objetos JS) + subida a GPU
  `bufferData` 9 480 B/frame (Float32Array de vértices, 64 prims; es CPU->GPU, no GPU real aquí).
- Latencia por frame (n=300, `performance.now` con resolución cuantizada a 0.1 ms en Chromium, por
  tanto orientativa): mediana end-to-end wasm->decode->parse->drawdata->GL+finish 0.3 ms (p95 0.5, max 0.5);
  wasm+decode+parse 0.2 ms; drawdata+GL upload+draw+finish 0.1 ms; Canvas2D draw 0.1 ms (max 43 ms, outlier).
  Media por lotes (n=1000, sin cuantización): 0.283 ms/frame end-to-end con GL.
  rAF headless: mediana 16.7 ms (vsync), no es medida de coste.
- JS heap 14.3 MB usado / 24.5 MB total (`performance.memory`, valor redondeado por Chromium, página entera).

### Matriz de soporte (solo lo observado; 17 checks por modo)
Reproducir: `BROWSER_MODE=<modo> node run_browser.mjs <wasm>` (modos: headless, headed-gpu, headed-vulkan, chrome-flatpak-cdp).
| Navegador / modo | WebGL2 (renderer) | WebGPU | Estado |
|---|---|---|---|
| Chromium 148 headless (SwiftShader) | SÍ, software | `requestAdapter()` null | PASS 17/17 |
| Chromium 148 headed, ANGLE/OpenGL, NVIDIA RTX 4060 Ti | SÍ, GPU real | adapter ok (nvidia/lovelace) | PASS 17/17; 0.289 ms/frame e2e (n=1000) |
| Chromium 148 headed, ANGLE/Vulkan, NVIDIA RTX 4060 Ti | `readPixels` correcto, pero el canvas GL NO se presenta (toDataURL y captura en blanco; hasta un `clear` rojo) y el 2D acelerado devuelve ceros | adapter ok | FAIL gl_canvas_presented + c2d_*: anomalía del backend Vulkan en este host, no del spike; el modo OpenGL funciona |
| Chrome 158 Flatpak (CDP vía Playwright `connectOverCDP`) | SÍ, NVIDIA (ANGLE OpenGL 4.5) | adapter ok pero `swiftshader` (software) | PASS 17/17 |
| Firefox 160.0.1 Flatpak | NOT_RUN | NOT_RUN | Playwright no maneja Firefox estándar; arnés alternativo (`run_post_browser.mjs`, la página hace POST del resultado) no recibió resultado en 45 s (causa no diagnosticada: sandbox/red/perfil) |
| Safari | no disponible | | NOT_RUN |

Nota: en Chromium con GPU el primer `getImageData` justo tras dibujar en Canvas2D acelerado devolvía ceros; el spike espera un rAF antes de muestrear. Tolerancia de color 12: la GPU da 33 en vez de 34 en el verde meta.
WebGPU: solo se verificó la obtención de adaptador; el camino de dibujo WebGPU NO se implementó ni ejecutó.

## NOT_RUN (declarado)
- Camino de dibujo WebGPU (el adaptador sí se obtiene con GPU real); Firefox y Safari; backend Vulkan de Chromium (falla la presentación en este host); navegadores móviles.
- Escenas mayores (Image/texturas, trazas largas): el JSON crece lineal (≈168 B/nodo aquí).
- Texto en GL (atlas de glifos): las etiquetas se dibujan con Canvas2D superpuesto.

## Comparación de complejidad vs renderer nativo
- Nativo: `viewport.rs` 238 + `camera.rs` 332 líneas (cámara interactiva, picking, SVG/software).
- Web (spike): `main.js` 175 + `index.html` 8 + `run_browser.mjs` 48 (harness) + `src/lib.rs` 55 de
  ABI wasm. Cubre rect/círculo/label estático, ajuste a ventana, GL y Canvas2D.
- Falta en la vía web: cámara interactiva/zoom/pan, picking, polilíneas, imágenes/texturas, texto GL,
  manejo de pérdida de contexto, DPI, empaquetado/toolchain (wasm-bindgen o ABI manual), y un
  parser/esquema versionado (hoy `JSON.parse` sobre esquema implícito). Un renderer completo
  duplicaría la lógica de `camera.rs`/`viewport.rs` en JS o exigiría exponerla por wasm.

## Limitaciones conocidas
- Sin parser JSON en Rust: lectura en navegador depende de `JSON.parse` y del esquema implícito.
- `static mut` buffer en el ABI crudo: solo spike, un hilo.
- JSON con floats en texto: coste de parseo > coste de generarlo (142 vs ~86 µs).

## Recomendación
**KEEP_AS_IS** (AC3). Viabilidad demostrada: el contrato `Scene2D` JSON existente basta como frontera,
se dibuja correctamente en WebGL2 y Canvas2D en Chromium 148 (~0.28 ms/frame end-to-end, wasm 120 KB /
44 KB gzip). Pero el valor extra es bajo: el renderer nativo ya cubre Studio, la vía web duplicaría
cámara/picking/texto, WebGPU no es verificable aquí y no hay GPU real ni otros navegadores probados.
Ningún requisito web bloquea el MVP. Si se retoma (FUT): Canvas2D primero (más simple, 0.1 ms), parser/
esquema versionado del contrato, probar Firefox/Chrome con GPU real y WebGPU.
