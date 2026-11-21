//! Benchmark GOG-86: latencia por invocación WASM (sandbox con fuel) vs función Rust nativa.
//! Ejecutar: `cargo run --release -p gogamen-sdk --features wasm --example wasm_bench`.
//! No es una garantía de tiempo real duro: el fuel es un presupuesto determinista, no de reloj.
use gogamen_sdk::wasm::{WasmEffect, WasmSandboxConfig};
use std::hint::black_box;
use std::time::Instant;

const WAT: &str = r#"(module
  (func (export "run") (param i64) (result i64)
    (local $i i64) (local $acc i64)
    (loop $l
      (local.set $acc (i64.add (local.get $acc) (i64.add (i64.mul (local.get 0) (i64.const 3)) (local.get $i))))
      (local.set $i (i64.add (local.get $i) (i64.const 1)))
      (br_if $l (i64.lt_u (local.get $i) (i64.const 100))))
    (local.get $acc)))"#;

fn native(x: i64) -> i64 {
    let mut acc = 0i64;
    let mut i = 0i64;
    while i < 100 {
        acc = acc.wrapping_add(x.wrapping_mul(3).wrapping_add(i));
        i += 1;
    }
    acc
}

fn main() {
    let t = Instant::now();
    let fx = WasmEffect::load(WAT.as_bytes(), WasmSandboxConfig::strict(1_000_000)).unwrap();
    println!("load+compile (once): {:?}", t.elapsed());
    let n = 20_000u32;
    let (_, fuel) = fx.invoke_with_fuel(1).unwrap();
    for i in 0..1000 {
        black_box(fx.invoke(black_box(i)).unwrap());
    }
    let t = Instant::now();
    let mut s = 0i64;
    for i in 0..n {
        s = s.wrapping_add(fx.invoke(black_box(i as i64)).unwrap());
    }
    let w = t.elapsed();
    let t = Instant::now();
    let mut s2 = 0i64;
    for i in 0..n {
        s2 = s2.wrapping_add(black_box(native(black_box(i as i64))));
    }
    let nat = t.elapsed();
    assert_eq!(s, s2);
    let (wn, nn) = (
        w.as_nanos() as f64 / n as f64,
        nat.as_nanos() as f64 / n as f64,
    );
    println!("fuel per call: {fuel}");
    println!("wasm (fresh store+instantiate+call): {wn:.0} ns/call");
    println!("native Rust: {nn:.1} ns/call");
    println!("ratio wasm/native: {:.0}x", wn / nn.max(0.1));
    println!("(no hard real-time guarantee; fuel is deterministic, not wall-clock)");
}
