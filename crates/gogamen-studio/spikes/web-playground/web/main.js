// GOG-109 spike: wasm (Scene2D canonical JSON from gogamen-projection) -> JSON.parse -> WebGL2 / Canvas2D / WebGPU(detect only)
const W = 480, H = 360;
const COLORS = [[1, 1, 1], [0.267, 0.267, 0.267], [0.133, 0.667, 0.133], [0.867, 0.133, 0.133]]; // state_code -> rgb
const CIRCLE = [0, 0.4, 0.8];

async function loadWasm() {
  const t0 = performance.now();
  const { instance } = await WebAssembly.instantiateStreaming(fetch('/gogamen_web_spike.wasm'), {});
  return { x: instance.exports, ms: performance.now() - t0 };
}

const dec = new TextDecoder();
let copiedBytes = 0, copies = 0;
function readOut(x, len) {
  copies++; copiedBytes += len;
  return dec.decode(new Uint8Array(x.memory.buffer, x.out_ptr(), len));
}

// Scene2D -> draw lists (prims in layer z order; y-up scene coords)
function toDrawData(scene) {
  const rects = [], circles = [], labels = [];
  const layers = scene.layers.filter(l => l.visible).sort((a, b) => a.z_order - b.z_order);
  for (const l of layers) for (const n of l.nodes) {
    const p = n.primitive;
    if (p.kind === 'rect') rects.push([p.min.x, p.min.y, p.width, p.height, n.state_code ?? 0]);
    else if (p.kind === 'circle') circles.push([p.center.x, p.center.y, p.radius]);
    else if (p.kind === 'label') labels.push([p.position.x, p.position.y, p.text]);
  }
  return { rects, circles, labels };
}
function fit(d) {
  let a = 1e9, b = 1e9, c = -1e9, e = -1e9;
  for (const r of d.rects) { a = Math.min(a, r[0]); b = Math.min(b, r[1]); c = Math.max(c, r[0] + r[2]); e = Math.max(e, r[1] + r[3]); }
  const s = Math.min((W - 16) / (c - a), (H - 16) / (e - b));
  // scene -> pixel (y down)
  return { s, X: v => W / 2 + (v - (a + c) / 2) * s, Y: v => H / 2 - (v - (b + e) / 2) * s };
}

// ---- Canvas2D ----
function drawCanvas2D(ctx, d, f) {
  ctx.fillStyle = '#eee'; ctx.fillRect(0, 0, W, H);
  for (const r of d.rects) {
    const c = COLORS[r[4]] ?? [.6, .6, .6];
    ctx.fillStyle = `rgb(${c.map(v => v * 255 | 0)})`;
    ctx.fillRect(f.X(r[0]), f.Y(r[1] + r[3]), r[2] * f.s, r[3] * f.s);
  }
  ctx.fillStyle = '#0066cc';
  for (const c of d.circles) { ctx.beginPath(); ctx.arc(f.X(c[0]), f.Y(c[1]), c[2] * f.s, 0, 6.2832); ctx.fill(); }
  ctx.fillStyle = '#000'; ctx.font = '9px monospace';
  for (const l of d.labels) ctx.fillText(l[2], f.X(l[0]), f.Y(l[1]));
  return d.rects.length + d.circles.length + d.labels.length;
}

// ---- WebGL2 ----
function makeGL(canvas) {
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false });
  if (!gl) return null;
  const vs = `#version 300 es
in vec2 p; in vec3 c; out vec3 v; uniform vec2 res;
void main(){ v=c; gl_Position=vec4(p.x/res.x*2.0-1.0, 1.0-p.y/res.y*2.0, 0.0, 1.0); }`;
  const fs = `#version 300 es
precision mediump float; in vec3 v; out vec4 o; void main(){ o=vec4(v,1.0); }`;
  const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(o)); return o; };
  const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr);
  if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(pr));
  gl.useProgram(pr);
  const vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  const pl = gl.getAttribLocation(pr, 'p'), cl = gl.getAttribLocation(pr, 'c');
  gl.enableVertexAttribArray(pl); gl.vertexAttribPointer(pl, 2, gl.FLOAT, false, 20, 0);
  gl.enableVertexAttribArray(cl); gl.vertexAttribPointer(cl, 3, gl.FLOAT, false, 20, 8);
  gl.uniform2f(gl.getUniformLocation(pr, 'res'), W, H);
  gl.viewport(0, 0, W, H);
  return gl;
}
const SEG = 32;
// returns {bytesUploaded, primitives(rect+circle)}
function drawGL(gl, d, f) {
  const verts = [];
  const quad = (x0, y0, x1, y1, c) => { for (const [x, y] of [[x0, y0], [x1, y0], [x0, y1], [x0, y1], [x1, y0], [x1, y1]]) verts.push(x, y, ...c); };
  for (const r of d.rects) quad(f.X(r[0]), f.Y(r[1] + r[3]), f.X(r[0] + r[2]), f.Y(r[1]), COLORS[r[4]] ?? [.6, .6, .6]);
  for (const c of d.circles) {
    const cx = f.X(c[0]), cy = f.Y(c[1]), r = c[2] * f.s;
    for (let i = 0; i < SEG; i++) {
      const a0 = i / SEG * 6.2832, a1 = (i + 1) / SEG * 6.2832;
      verts.push(cx, cy, ...CIRCLE, cx + r * Math.cos(a0), cy + r * Math.sin(a0), ...CIRCLE, cx + r * Math.cos(a1), cy + r * Math.sin(a1), ...CIRCLE);
    }
  }
  const arr = new Float32Array(verts);
  gl.clearColor(0.933, 0.933, 0.933, 1); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bufferData(gl.ARRAY_BUFFER, arr, gl.DYNAMIC_DRAW); // CPU->GPU upload
  gl.drawArrays(gl.TRIANGLES, 0, arr.length / 5);
  return { bytes: arr.byteLength, prims: d.rects.length + d.circles.length };
}
function drawLabels(ctx, d, f) { // text is not GL: 2D overlay (a real web renderer would need a glyph atlas)
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#000'; ctx.font = '9px monospace';
  for (const l of d.labels) ctx.fillText(l[2], f.X(l[0]), f.Y(l[1]));
  return d.labels.length;
}

const median = a => a.slice().sort((p, q) => p - q)[a.length >> 1];
const q = (a, p) => a.slice().sort((x, y) => x - y)[Math.min(a.length - 1, Math.floor(a.length * p))];
function stats(a) { return { median_ms: +median(a).toFixed(4), p95_ms: +q(a, .95).toFixed(4), max_ms: +Math.max(...a).toFixed(4), n: a.length }; }

window.runSpike = async function (N = 300) {
  const res = { userAgent: navigator.userAgent };
  const { x, ms } = await loadWasm();
  res.wasm_instantiate_ms = +ms.toFixed(2);
  res.wasm_memory_initial_bytes = x.memory.buffer.byteLength;
  const len = x.grid_json_len();
  const json = readOut(x, len);
  const fx = await (await fetch('/fixture/grid_scene.json')).text();
  res.json_bytes = len; res.wasm_json_equals_native_fixture = json === fx;
  const scene = JSON.parse(json);
  res.scene_nodes = scene.layers.reduce((a, l) => a + l.nodes.length, 0);

  // feature detection
  const probe = document.createElement('canvas');
  res.webgl2 = !!probe.getContext('webgl2');
  res.webgpu_navigator_gpu = typeof navigator.gpu !== 'undefined';
  if (res.webgpu_navigator_gpu) {
    try { const a = await navigator.gpu.requestAdapter(); res.webgpu_adapter = a ? 'ok' : 'null'; } catch (e) { res.webgpu_adapter = 'error ' + e.message; }
  } else res.webgpu_adapter = 'unsupported (navigator.gpu undefined); WebGPU path NOT_RUN';
  const glc = document.getElementById('gl');
  const gl = makeGL(glc);
  if (gl) { const e = gl.getExtension('WEBGL_debug_renderer_info'); res.gl_renderer = e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); res.gl_version = gl.getParameter(gl.VERSION); }
  const lctx = document.getElementById('lbl').getContext('2d');
  const cctx = document.getElementById('c2d').getContext('2d');

  // one real render
  let d = toDrawData(scene), f = fit(d);
  const g = gl ? drawGL(gl, d, f) : null;
  const nl = drawLabels(lctx, d, f);
  const n2 = drawCanvas2D(cctx, d, f);
  res.drawn = { gl_rects_circles: g?.prims, gl_labels_overlay: nl, canvas2d_total: n2 };
  res.gl_upload_bytes_per_frame = g?.bytes;
  res.layout = { scale: f.s, rectX: r => 0 }; delete res.layout.rectX;
  res.rects = d.rects; res.circles = d.circles; res.labels_n = d.labels.length;
  // pixel samples (page-side readback of GL framebuffer; y from bottom)
  const px = (X, Y) => { const o = new Uint8Array(4); gl.readPixels(Math.round(X), H - 1 - Math.round(Y), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, o); return [...o]; };
  const c2 = (X, Y) => [...cctx.getImageData(Math.round(X), Math.round(Y), 1, 1).data];
  const centre = r => [f.X(r[0] + r[2] / 2), f.Y(r[1] + r[3] / 2)];
  const goal = d.rects.find(r => r[4] === 2), wall = d.rects.find(r => r[4] === 1), open = d.rects.find(r => r[4] === 0), circ = d.circles[0];
  await new Promise(r => requestAnimationFrame(() => r())); // let accelerated Canvas2D flush before readback (GPU Chromium returned zeros otherwise)
  res.samples = {};
  for (const [name, p] of [['goal', centre(goal)], ['wall', centre(wall)], ['open', centre(open)], ['agent_circle', [f.X(circ[0]), f.Y(circ[1])]], ['background', [2, 2]]]) {
    res.samples[name] = { at: p.map(Math.round), gl: gl ? px(...p) : null, canvas2d: c2(...p) };
  }

  // latency: end-to-end wasm -> decode -> parse -> drawdata -> GL draw (sync CPU side) ; gl.finish to include GPU completion
  const e2e = [], parse = [], glonly = [], c2only = [];
  for (let i = 0; i < 30; i++) { drawGL(gl, toDrawData(JSON.parse(readOut(x, x.grid_json_len()))), f); }
  const copies0 = copies, bytes0 = copiedBytes;
  for (let i = 0; i < N; i++) {
    const a = performance.now();
    const s = JSON.parse(readOut(x, x.grid_json_len()));
    const b = performance.now();
    const dd = toDrawData(s); const gg = drawGL(gl, dd, f); gl.finish();
    const c = performance.now();
    e2e.push(c - a); parse.push(b - a); glonly.push(c - b);
  }
  res.frame_copies = (copies - copies0) / N; res.frame_copied_bytes = (copiedBytes - bytes0) / N;
  for (let i = 0; i < N; i++) { const a = performance.now(); drawCanvas2D(cctx, d, f); c2only.push(performance.now() - a); }
  res.latency = { e2e_wasm_decode_parse_drawdata_gl: stats(e2e), wasm_decode_parse: stats(parse), drawdata_plus_gl_upload_draw_finish: stats(glonly), canvas2d_draw_only: stats(c2only) };
  { const a = performance.now(); for (let i = 0; i < 1000; i++) { drawGL(gl, toDrawData(JSON.parse(readOut(x, x.grid_json_len()))), f); } gl.finish(); res.batch_e2e_mean_ms_per_frame_n1000 = +((performance.now() - a) / 1000).toFixed(4); }
  // rAF-paced frame interval (headless; vsync-bound, informative only)
  const ft = []; let last = performance.now();
  await new Promise(r => { let k = 0; const step = t => { ft.push(t - last); last = t; if (++k < 60) requestAnimationFrame(step); else r(); }; requestAnimationFrame(step); });
  res.raf_interval = stats(ft.slice(1));
  res.wasm_memory_final_bytes = x.memory.buffer.byteLength;
  res.js_heap = performance.memory ? { used: performance.memory.usedJSHeapSize, total: performance.memory.totalJSHeapSize } : null;
  document.getElementById('out').textContent = JSON.stringify({ webgl2: res.webgl2, webgpu: res.webgpu_adapter, nodes: res.scene_nodes, drawn: res.drawn }, null, 1);
  document.body.dataset.ready = '1';
  return res;
};
runSpike.auto = true;
// Harness for browsers Playwright cannot drive (stock Firefox): ?post=1 runs the spike and POSTs the result back.
if (new URLSearchParams(location.search).has('post')) {
  (async () => {
    let body;
    try {
      const r = await window.runSpike(300);
      const gl = document.getElementById('gl');
      const im = new Image(); im.src = gl.toDataURL('image/png'); await im.decode();
      const c = document.createElement('canvas'); c.width = 480; c.height = 360; const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(im, 0, 0);
      const d = x.getImageData(0, 0, 480, 360).data; const set = new Set(); for (let i = 0; i < d.length; i += 4) set.add(d[i] + ',' + d[i + 1] + ',' + d[i + 2]);
      let rend = null; const g2 = document.createElement('canvas').getContext('webgl2'); if (g2) { const e = g2.getExtension('WEBGL_debug_renderer_info'); rend = e ? g2.getParameter(e.UNMASKED_RENDERER_WEBGL) : g2.getParameter(g2.RENDERER); }
      delete r.rects; delete r.circles; r.presented = { distinct_colours: set.size }; r.renderer = rend; body = JSON.stringify(r);
    } catch (e) { body = JSON.stringify({ error: String(e) }); }
    await fetch('/result', { method: 'POST', body });
  })();
}
