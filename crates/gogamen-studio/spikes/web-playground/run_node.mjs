// GOG-109 spike harness (node only; NO browser). Usage: node run_node.mjs <wasm> 
import fs from 'node:fs';
import { performance } from 'node:perf_hooks';
const wasmPath = process.argv[2];
const bytes = fs.readFileSync(wasmPath);
const t0 = performance.now();
const mod = await WebAssembly.compile(bytes);
const t1 = performance.now();
console.log('wasm_bytes', bytes.length, 'imports', JSON.stringify(WebAssembly.Module.imports(mod)));
const inst = await WebAssembly.instantiate(mod, {});
const t2 = performance.now();
console.log(`compile_ms=${(t1-t0).toFixed(2)} instantiate_ms=${(t2-t1).toFixed(2)}`);
const x = inst.exports;
const wasmMem = () => x.memory.buffer.byteLength;
console.log('wasm_memory_initial_bytes', wasmMem());
const dec = new TextDecoder();
let copies = 0, copiedBytes = 0;
function readOut(len) { // copy #1: wasm linear memory -> JS string (TextDecoder copies/decodes)
  const v = new Uint8Array(x.memory.buffer, x.out_ptr(), len); // view: zero-copy
  copies++; copiedBytes += len;
  return dec.decode(v);
}
// JS "web renderer" front-end: parse contract -> instance buffers (what a WebGL2/WebGPU pass would upload)
function toDrawData(scene) {
  const layers = [...scene.layers].filter(l => l.visible).sort((a,b)=>a.z_order-b.z_order);
  const rects = [], circles = [], labels = [];
  for (const l of layers) for (const n of l.nodes) {
    const p = n.primitive;
    if (p.kind==='rect') rects.push(p.min.x,p.min.y,p.width,p.height,n.state_code??0);
    else if (p.kind==='circle') circles.push(p.center.x,p.center.y,p.radius);
    else if (p.kind==='label') labels.push({x:p.position.x,y:p.position.y,t:p.text});
  }
  return { rects: new Float32Array(rects), circles: new Float32Array(circles), labels };
}
function jsSvg(scene, w=480, h=360) {
  const d = toDrawData(scene);
  // fit y-up scene to window (bounds from rects)
  let minx=1e9,miny=1e9,maxx=-1e9,maxy=-1e9;
  for (let i=0;i<d.rects.length;i+=5){minx=Math.min(minx,d.rects[i]);miny=Math.min(miny,d.rects[i+1]);maxx=Math.max(maxx,d.rects[i]+d.rects[i+2]);maxy=Math.max(maxy,d.rects[i+1]+d.rects[i+3]);}
  const s=Math.min((w-16)/(maxx-minx),(h-16)/(maxy-miny));
  const X=v=>(w/2+(v-(minx+maxx)/2)*s).toFixed(2), Y=v=>(h/2-(v-(miny+maxy)/2)*s).toFixed(2);
  const cols=['#fff','#444','#2a2','#d22'];
  let o=`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}">`;
  for (let i=0;i<d.rects.length;i+=5) o+=`<rect x="${X(d.rects[i])}" y="${Y(d.rects[i+1]+d.rects[i+3])}" width="${(d.rects[i+2]*s).toFixed(2)}" height="${(d.rects[i+3]*s).toFixed(2)}" fill="${cols[d.rects[i+4]]??'#999'}" stroke="#888"/>`;
  for (let i=0;i<d.circles.length;i+=3) o+=`<circle cx="${X(d.circles[i])}" cy="${Y(d.circles[i+1])}" r="${(d.circles[i+2]*s).toFixed(2)}" fill="#06c"/>`;
  for (const l of d.labels) o+=`<text x="${X(l.x)}" y="${Y(l.y)}">${l.t}</text>`;
  return o+'</svg>';
}
const median = a => a.slice().sort((p,q)=>p-q)[a.length>>1];
async function bench(name, fn, n=500) {
  for (let i=0;i<50;i++) fn();
  const ts=[]; for (let i=0;i<n;i++){const a=performance.now(); fn(); ts.push((performance.now()-a)*1000);}
  console.log(`${name}: median_us=${median(ts).toFixed(1)} min_us=${Math.min(...ts).toFixed(1)} max_us=${Math.max(...ts).toFixed(1)} n=${n}`);
}
// functional check: wasm JSON == native fixture
const len = x.grid_json_len();
const json = readOut(len);
const fixture = fs.readFileSync(new URL('./fixture/grid_scene.json', import.meta.url), 'utf8');
console.log('json_bytes', len, 'wasm_json_equals_native_fixture', json === fixture);
const scene = JSON.parse(json);
const nodes = scene.layers.reduce((a,l)=>a+l.nodes.length,0);
console.log('layers', scene.layers.map(l=>`${l.id}:${l.nodes.length}`).join(','), 'nodes', nodes);
const d = toDrawData(scene);
console.log('draw_data_bytes', d.rects.byteLength + d.circles.byteLength, '(rects', d.rects.length/5, 'circles', d.circles.length/3, 'labels', d.labels.length + ')');
const wasmSvgLen = x.grid_svg_len(480,360); const wasmSvg = readOut(wasmSvgLen);
const nativeSvg = fs.readFileSync(new URL('./fixture/grid_native.svg', import.meta.url), 'utf8');
console.log('wasm_svg_bytes', wasmSvgLen, 'wasm_svg_equals_native_svg', wasmSvg === nativeSvg);
fs.writeFileSync(new URL('./fixture/grid_js.svg', import.meta.url), jsSvg(scene));
console.log('js_svg_bytes', jsSvg(scene).length, '(written fixture/grid_js.svg)');
// latency
await bench('wasm scene+json (no copy out)', () => x.grid_json_len());
await bench('wasm scene+json + copy-out decode', () => readOut(x.grid_json_len()));
await bench('JSON.parse only', () => JSON.parse(json));
await bench('JSON.parse + toDrawData', () => toDrawData(JSON.parse(json)));
await bench('end-to-end wasm->decode->parse->drawdata', () => toDrawData(JSON.parse(readOut(x.grid_json_len()))));
await bench('wasm scene+native-SVG (no copy out)', () => x.grid_svg_len(480,360));
await bench('JS jsSvg from parsed scene', () => jsSvg(scene));
console.log(`copies=${copies} copied_bytes=${copiedBytes} (copies counted in this run: reads of wasm out buffer via TextDecoder)`);
console.log('wasm_memory_final_bytes', wasmMem());
const m = process.memoryUsage();
console.log('node_rss_MB', (m.rss/1048576).toFixed(1), 'heapUsed_MB', (m.heapUsed/1048576).toFixed(1));
