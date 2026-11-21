// GOG-109: drive web/index.html in headless Chromium via Playwright; assert + measure.
// Usage: node run_browser.mjs <wasm> [chromium|firefox ...]
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path';
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const PW = process.env.PLAYWRIGHT_CORE || '/var/home/rubentxu/.asdf/installs/nodejs/25.9.0/lib/node_modules/@playwright/cli/node_modules/playwright-core';
const { chromium } = require(PW);
const MODE = process.env.BROWSER_MODE || 'headless'; // headless | headed-gpu | chrome-flatpak-cdp
const here = path.dirname(new URL(import.meta.url).pathname);
const wasm = process.argv[2];
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
const server = http.createServer((q, r) => {
  const u = q.url.split('?')[0]; if (u === '/favicon.ico') { r.writeHead(204); return r.end(); }
  const f = u === '/' ? path.join(here, 'web/index.html') : u === '/gogamen_web_spike.wasm' ? wasm
    : u.startsWith('/fixture/') ? path.join(here, u) : path.join(here, 'web', u);
  if (!f.startsWith(here) && f !== wasm || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f));
}).listen(0);
await new Promise(r => server.on('listening', r));
const url = `http://127.0.0.1:${server.address().port}/`;
let browser, page, ctxClose = async () => {}, extra = {};
const vp = { width: 1100, height: 420 };
if (MODE === 'headless') { browser = await chromium.launch({ headless: true }); page = await browser.newPage({ viewport: vp }); }
else if (MODE === 'headed-gpu' || MODE === 'headed-vulkan') {
  // headed-gpu: real NVIDIA GPU through ANGLE/OpenGL. headed-vulkan: ANGLE/Vulkan (+WebGPU flags); see REPORT for its presentation anomaly.
  const args = MODE === 'headed-gpu' ? ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-features=Vulkan', '--use-gl=angle', '--use-angle=gl'] : ['--enable-unsafe-webgpu', '--enable-features=Vulkan,WebGPU', '--ignore-gpu-blocklist', '--use-angle=vulkan', '--enable-gpu'];
  browser = await chromium.launch({ headless: false, args });
  page = await browser.newPage({ viewport: vp });
} else if (MODE === 'chrome-flatpak-cdp') {
  const { spawn } = await import('node:child_process');
  const proc = spawn('flatpak', ['run', '--socket=wayland', 'com.google.Chrome', '--remote-debugging-port=9333', '--user-data-dir=/tmp/g109-chrome', '--no-first-run', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--ozone-platform=x11', 'about:blank'], { stdio: 'ignore' });
  let ok = false; for (let i = 0; i < 60 && !ok; i++) { await new Promise(r => setTimeout(r, 500)); try { ok = (await fetch('http://127.0.0.1:9333/json/version')).ok; } catch {} }
  if (!ok) throw new Error('chrome flatpak CDP not reachable');
  browser = await chromium.connectOverCDP('http://127.0.0.1:9333');
  page = await browser.contexts()[0].newPage(); await page.setViewportSize(vp);
  ctxClose = async () => { try { await browser.close(); } catch {} proc.kill(); };
}
const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => m.type() === 'error' && errors.push(m.text()));
await page.goto(url);
const res = await page.evaluate(() => window.runSpike(300));
await page.screenshot({ path: path.join(here, `fixture/grid_${MODE}.png`) });
const presented = await page.evaluate(async () => { await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  const im = new Image(); im.src = document.getElementById('gl').toDataURL('image/png'); await im.decode();
  const c = document.createElement('canvas'); c.width = 480; c.height = 360; const x = c.getContext('2d', { willReadFrequently: true }); x.drawImage(im, 0, 0);
  const d = x.getImageData(0, 0, 480, 360).data; const set = new Set(); for (let i = 0; i < d.length; i += 4) set.add(d[i] + ',' + d[i + 1] + ',' + d[i + 2]); return { distinct_colours: set.size }; });
const after = await page.evaluate(() => ({ heap: performance.memory && performance.memory.usedJSHeapSize }));
extra = await page.evaluate(async () => { const gl = document.createElement('canvas').getContext('webgl2'); const d = gl && gl.getExtension('WEBGL_debug_renderer_info'); let wg = 'no navigator.gpu'; if (navigator.gpu) { try { const a = await navigator.gpu.requestAdapter(); wg = a ? 'adapter ok: ' + JSON.stringify(a.info ? { vendor: a.info.vendor, architecture: a.info.architecture, description: a.info.description } : {}) : 'requestAdapter null'; } catch (e) { wg = 'error ' + e; } } return { ua: navigator.userAgent, renderer: d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : null, webgl2: !!gl, webgpu: wg }; });
res.browser = { mode: MODE, version: browser.version(), ...extra };
await ctxClose(); if (MODE !== 'chrome-flatpak-cdp') await browser.close(); server.close();
const near = (a, b, t = 12) => a.slice(0, 3).every((v, i) => Math.abs(v - b[i]) <= t);
const exp = { goal: [34, 170, 34], wall: [68, 68, 68], open: [255, 255, 255], agent_circle: [0, 102, 204], background: [238, 238, 238] };
const checks = {};
checks.no_page_errors = errors.length === 0;
checks.json_equals_native = res.wasm_json_equals_native_fixture;
checks.nodes_105 = res.scene_nodes === 105;
checks.gl_drawn_105 = res.drawn.gl_rects_circles + res.drawn.gl_labels_overlay === 105;
checks.canvas2d_drawn_105 = res.drawn.canvas2d_total === 105;
for (const k of Object.keys(exp)) { checks[`gl_${k}`] = near(res.samples[k].gl, exp[k]); checks[`c2d_${k}`] = near(res.samples[k].canvas2d, exp[k]); }
// screenshot not blank: count distinct colours
const png = fs.statSync(path.join(here, `fixture/grid_${MODE}.png`)).size;
checks.gl_canvas_presented = presented.distinct_colours >= 4; res.gl_presented = presented;
checks.screenshot_nonblank_size = png > 3000;
delete res.rects; delete res.circles;
res.errors = errors; res.checks = checks; res.screenshot_bytes = png; res.js_heap_after = after.heap;
fs.writeFileSync(path.join(here, `fixture/browser_run_output_${MODE}.json`), JSON.stringify(res, null, 2));
console.log(JSON.stringify(res, null, 2));
const failed = Object.entries(checks).filter(([, v]) => !v).map(([k]) => k);
assert.equal(failed.length, 0, 'failed checks: ' + failed.join(','));
console.log('ALL CHECKS PASSED', Object.keys(checks).length);
