//! Verifica que `BoundedExprEvaluator::eval` no realiza asignaciones de heap.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use gogamen_compiler::{BoundedExprEvaluator, CompiledExpr, ExprSchema, ExprValue};
use gogamen_spec::{EXPR_AST_MAX_DEPTH, ExprAst, ExprType, ExprUnit};

struct CountingAlloc;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

// SAFETY: delega íntegramente en `System`; solo cuenta por hilo.
unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        unsafe { System.realloc(p, l, n) }
    }
}

#[global_allocator]
static A: CountingAlloc = CountingAlloc;

#[test]
fn eval_performs_zero_heap_allocations_including_max_depth() {
    let mut schema = ExprSchema::new();
    schema
        .register("x", ExprType::Scalar(ExprUnit::Ratio))
        .unwrap();
    let mut ast = ExprAst::var("x").unwrap();
    for _ in 0..(EXPR_AST_MAX_DEPTH - 1) {
        ast = ExprAst::add(ExprAst::var("x").unwrap(), ast);
    }
    let compiled = CompiledExpr::compile(&ast, &schema).unwrap();
    let values = [ExprValue::Scalar(1.0)];

    let before = ALLOCS.with(Cell::get);
    let mut last = None;
    for _ in 0..100 {
        last = Some(BoundedExprEvaluator::eval(&compiled, &values, 10_000));
    }
    let after = ALLOCS.with(Cell::get);
    assert_eq!(after - before, 0, "eval must not allocate");
    assert_eq!(
        last.unwrap(),
        Ok(ExprValue::Scalar(EXPR_AST_MAX_DEPTH as f64))
    );
}
