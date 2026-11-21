//! Test de integración aislado con `#[global_allocator]` personalizado para demostrar
//! cero asignaciones en heap durante `step` en ruta crítica tras warmup (`GOG-58`).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use gogamen_core::{
    CapabilityCatalog, CapabilityVersion, EffectKind, ModuleManifest, ModuleOrigin, OperationSpec,
    TemporalProfile,
};
use gogamen_runtime::{
    CriticalBufferConfig, CriticalOperationDescriptor, CriticalPathPlan, DeclaredCapabilityClaim,
    MemoryBoundClaim, ObservedAllocationProof, RuntimeTemporalProfile,
    STRICT_REALTIME_NON_CERTIFICATION_NOTICE, TemporalValidationError, prepare_critical_path,
    validate_before_start,
};

struct CountingAllocator;

thread_local! {
    static THREAD_MEASURING: Cell<bool> = const { Cell::new(false) };
}

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static REALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);
static ALLOC_BYTES: AtomicUsize = AtomicUsize::new(0);
static TEST_SERIALIZE: Mutex<()> = Mutex::new(());

fn is_measuring_this_thread() -> bool {
    THREAD_MEASURING
        .try_with(|flag| flag.get())
        .unwrap_or(false)
}

// SAFETY: Delega todas las operaciones en `std::alloc::System` y únicamente actualiza
// contadores atómicos lock-free cuando `THREAD_MEASURING` está activo en el hilo actual.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if is_measuring_this_thread() {
            ALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if is_measuring_this_thread() {
            REALLOC_COUNT.fetch_add(1, Ordering::Relaxed);
            ALLOC_BYTES.fetch_add(new_size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ControlSample {
    tick: u64,
    setpoint: f64,
    output: f64,
}

#[test]
fn critical_step_performs_zero_heap_allocations_after_warmup_and_records_limits() {
    let _guard = TEST_SERIALIZE.lock().unwrap();

    let mut catalog = CapabilityCatalog::new();
    catalog
        .register_module(
            ModuleManifest::new(
                "motion_ctrl",
                ModuleOrigin::Builtin,
                CapabilityVersion::new(1, 0, 0),
            )
            .with_operation(
                OperationSpec::new("pid_step", CapabilityVersion::new(1, 0, 0))
                    .with_effect(EffectKind::StateMutation)
                    .with_temporal_profile(TemporalProfile::BoundedRt),
            ),
        )
        .unwrap();

    let buffer_cfg = CriticalBufferConfig::try_new(16, 128).unwrap();
    let claim = DeclaredCapabilityClaim::critical_non_blocking(
        RuntimeTemporalProfile::DeadlineAware,
        16 * std::mem::size_of::<Option<ControlSample>>() + 128,
    );
    let plan = CriticalPathPlan::new(
        RuntimeTemporalProfile::DeadlineAware,
        vec![CriticalOperationDescriptor::new(
            "motion_ctrl",
            "pid_step",
            claim,
        )],
        buffer_cfg,
    );

    // Fase de preparación: preasigna todos los buffers fijos antes de `start`.
    let mut critical_path = prepare_critical_path::<ControlSample>(&plan, Some(&catalog)).unwrap();
    critical_path.start().unwrap();

    // Fase de warmup (llena y envuelve el ring buffer antes de medir).
    let warmup_steps: u64 = 32;
    for tick in 0..warmup_steps {
        let sample = ControlSample {
            tick,
            setpoint: 1.0,
            output: 0.5,
        };
        critical_path
            .step(sample, |bufs| {
                let scratch = bufs.scratch_mut();
                scratch[0] = (tick & 0xff) as u8;
                scratch[127] = 0xAA;
            })
            .unwrap();
    }

    // Inicializa TLS antes de medir, reinicia contadores y activa medición estricta en este hilo.
    THREAD_MEASURING.with(|flag| flag.set(false));
    ALLOC_COUNT.store(0, Ordering::Release);
    REALLOC_COUNT.store(0, Ordering::Release);
    ALLOC_BYTES.store(0, Ordering::Release);
    THREAD_MEASURING.with(|flag| flag.set(true));

    let measured_steps: u64 = 512;
    let mut step_ok = true;
    for idx in 0..measured_steps {
        let tick = warmup_steps + idx;
        let sample = ControlSample {
            tick,
            setpoint: 1.0,
            output: (idx as f64) * 0.01,
        };
        if critical_path
            .step(sample, |bufs| {
                let scratch = bufs.scratch_mut();
                let bytes = tick.to_le_bytes();
                scratch[0..8].copy_from_slice(&bytes);
                // Ejercita también lectura/extracción periódica sin heap.
                if tick.is_multiple_of(17) {
                    let _ = bufs.pop_sample();
                }
            })
            .is_err()
        {
            step_ok = false;
        }
    }

    THREAD_MEASURING.with(|flag| flag.set(false));
    let allocs = ALLOC_COUNT.load(Ordering::Acquire);
    let reallocs = REALLOC_COUNT.load(Ordering::Acquire);
    let bytes = ALLOC_BYTES.load(Ordering::Acquire);

    assert!(step_ok);
    assert_eq!(
        allocs, 0,
        "expected 0 heap allocations during critical step"
    );
    assert_eq!(
        reallocs, 0,
        "expected 0 heap reallocations during critical step"
    );
    assert_eq!(bytes, 0, "expected 0 heap bytes allocated during step");

    let limits = critical_path.buffer_limits();
    assert_eq!(limits.ring_capacity, 16);
    assert_eq!(limits.scratch_capacity_bytes, 128);
    assert!(limits.total_preallocated_bytes >= 128);

    let proof = ObservedAllocationProof::new(
        warmup_steps,
        measured_steps,
        allocs,
        reallocs,
        bytes,
        limits,
    );
    assert!(proof.zero_alloc_verified());
    assert_eq!(
        proof.non_certification_notice,
        STRICT_REALTIME_NON_CERTIFICATION_NOTICE
    );

    critical_path.record_observed_proof(proof);
    critical_path.stop().unwrap();

    let report = critical_path.report();
    assert_eq!(report.profile, RuntimeTemporalProfile::DeadlineAware);
    assert_eq!(report.core_profile, TemporalProfile::BoundedRt);
    assert_eq!(report.buffer_limits, limits);
    assert!(report.observed_proof.unwrap().zero_alloc_verified());
    assert_eq!(
        report.non_certification_notice,
        STRICT_REALTIME_NON_CERTIFICATION_NOTICE
    );
}

#[test]
fn rejects_blocking_and_unbounded_operations_before_start() {
    let _guard = TEST_SERIALIZE.lock().unwrap();

    let cfg = CriticalBufferConfig::try_new(8, 32).unwrap();
    let blocking_plan = CriticalPathPlan::new(
        RuntimeTemporalProfile::Interactive,
        vec![CriticalOperationDescriptor::new(
            "io",
            "net_sync",
            DeclaredCapabilityClaim::critical_non_blocking(RuntimeTemporalProfile::Interactive, 32)
                .with_blocking_io(true),
        )],
        cfg,
    );
    assert!(matches!(
        validate_before_start(&blocking_plan, None),
        Err(TemporalValidationError::BlockingIoRejected { .. })
    ));

    let unbounded_plan = CriticalPathPlan::new(
        RuntimeTemporalProfile::DeadlineAware,
        vec![CriticalOperationDescriptor::new(
            "planner",
            "dynamic_tree",
            DeclaredCapabilityClaim::critical_non_blocking(
                RuntimeTemporalProfile::DeadlineAware,
                32,
            )
            .with_memory_bound(MemoryBoundClaim::Unbounded),
        )],
        cfg,
    );
    assert!(matches!(
        validate_before_start(&unbounded_plan, None),
        Err(TemporalValidationError::UnboundedMemoryRejected { .. })
    ));
}
