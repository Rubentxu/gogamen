// GOG-109: run the spike in a browser Playwright cannot drive (Firefox Flatpak). Page POSTs its result.
// Usage: node run_post_browser.mjs <wasm> <name> <cmd> [args...]   (URL is appended as last arg)
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { spawn } from 'node:child_process'; import assert from 'node:assert/strict';
const here = path.dirname(new URL(import.meta.url).pathname); const [wasm, name, cmd, ...args] = process.argv.slice(2);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm' };
let resolveRes; const got = new Promise(r => resolveRes = r);
const server = http.createServer((q, r) => {
  const u = q.url.split('?')[0];
  if (q.method === 'POST' && u === '/result') { let b = ''; q.on('data', c => b += c); q.on('end', () => { r.writeHead(200); r.end('ok'); resolveRes(JSON.parse(b)); }); return; }
  if (u === '/favicon.ico') { r.writeHead(204); return r.end(); }
  const f = u === '/' ? path.join(here, 'web/index.html') : u === '/gogamen_web_spike.wasm' ? wasm : u.startsWith('/fixture/') ? path.join(here, u) : path.join(here, 'web', u);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': types[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f));
}).listen(0);
await new Promise(r => server.on('listening', r));
const url = `http://127.0.0.1:${server.address().port}/?post=1`;
const proc = spawn(cmd, [...args, url], { stdio: 'ignore' });
const res = await Promise.race([got, new Promise((_, j) => setTimeout(() => j(new Error('timeout waiting for result')), 45000))]).finally(() => { proc.kill(); server.close(); });
assert.ok(!res.error, 'page error: ' + res.error);
const near = (a, b, t = 12) => a.slice(0, 3).every((v, i) => Math.abs(v - b[i]) <= t);
const exp = { goal: [34, 170, 34], wall: [68, 68, 68], open: [255, 255, 255], agent_circle: [0, 102, 204], background: [238, 238, 238] };
const checks = { json_equals_native: res.wasm_json_equals_native_fixture, nodes_105: res.scene_nodes === 105, gl_drawn_105: res.drawn.gl_rects_circles + res.drawn.gl_labels_overlay === 105, canvas2d_drawn_105: res.drawn.canvas2d_total === 105, gl_canvas_presented: res.presented.distinct_colours >= 4 };
for (const k of Object.keys(exp)) { checks[`gl_${k}`] = near(res.samples[k].gl, exp[k]); checks[`c2d_${k}`] = near(res.samples[k].canvas2d, exp[k]); }
res.checks = checks; fs.writeFileSync(path.join(here, `fixture/browser_run_output_${name}.json`), JSON.stringify(res, null, 2));
console.log(JSON.stringify({ ua: res.userAgent, renderer: res.renderer, webgl2: res.webgl2, webgpu: res.webgpu_adapter, e2e_ms: res.batch_e2e_mean_ms_per_frame_n1000, presented: res.presented, checks }, null, 1));
const failed = Object.entries(checks).filter(([, v]) => !v).map(([k]) => k); assert.equal(failed.length, 0, 'failed: ' + failed.join(','));
console.log('ALL CHECKS PASSED', Object.keys(checks).length);
