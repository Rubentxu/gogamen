//! Subset `#![no_std]` puro sin `alloc` de Gogamen para control de referencia
//! en tiempo acotado sobre memoria estática o de stack.
//!
//! Este crate está diseñado para ejecutarse sin JVM, sin red, sin sistema de
//! archivos y sin asignador dinámico (`0` crates en `[dependencies]`):
//! - [`FixedRingBuffer`]: buffer circular de capacidad fija `N` en stack/estático.
//! - [`ActionClampGuard`]: guardia determinista de límites de acción y fallback seguro.
//! - [`BoundedControlCycle`]: lazo de control PD de referencia en tiempo acotado O(1).
//! - [`EmbeddedAnalysisScope`] y [`EMBEDDED_WCET_AND_RTOS_SCOPE`]: alcance explícito de
//!   memoria estática (`core::mem::size_of`), plazos, dependencias y requisitos RTOS/ISR.
//! - [`NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE`]: política explícita de
//!   no certificación hard real-time sin análisis WCET en la plataforma concreta.

#![no_std]
#![deny(unsafe_code)]
#![warn(missing_docs)]

#[cfg(test)]
extern crate std;

use core::fmt;
use core::mem::size_of;

/// Política explícita: este crate no promueve ni certifica garantías hard real-time
/// sin un análisis WCET específico del binario y de la plataforma hardware/RTOS concreta.
pub const NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE: &str = "Gogamen embedded no promueve ni certifica garantías hard real-time sin análisis WCET específico del binario, compilador, interrupciones y plataforma hardware/RTOS concreta (p. ej. FreeRTOS, Zephyr o Embassy).";

/// Capacidad de historial por defecto para [`EMBEDDED_WCET_AND_RTOS_SCOPE`].
pub const DEFAULT_EMBEDDED_HISTORY_CAPACITY: usize = 32;

/// Plazo por defecto de referencia (en microsegundos) para [`EMBEDDED_WCET_AND_RTOS_SCOPE`].
pub const DEFAULT_EMBEDDED_DEADLINE_MICROS: u64 = 1_000;

/// Requisitos mínimos de RTOS, interrupciones y plataforma objetivo para un análisis WCET real.
pub const DEFAULT_RTOS_AND_INTERRUPT_REQUIREMENTS: &[&str] = &[
    "RTOS determinista o ejecutor bare-metal/async embebido (p. ej. FreeRTOS, Zephyr o Embassy) con prioridades fijas y sin asignación dinámica en la ruta crítica.",
    "Interrupción periódica de timer hardware dedicada al disparo del lazo de control con jitter de despacho acotado y medido.",
    "Análisis WCET estático/híbrido por binario final y arquitectura objetivo (p. ej. Cortex-M, RISC-V) considerando caché, flash wait-states, FPU y secciones críticas.",
    "Presupuesto acotado de latencia de ISR y ausencia de bloqueos no acotados compartidos con tareas de menor prioridad.",
];

/// Descripción explícita del alcance de análisis, presupuesto de memoria estática,
/// plazos, dependencias y requisitos de RTOS/interrupciones para un análisis WCET real.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddedAnalysisScope {
    /// Nombre del crate analizado.
    pub crate_name: &'static str,
    /// Indica que el código de producción compila con `#![no_std]`.
    pub no_std: bool,
    /// Indica que el código de producción no utiliza `alloc` ni heap.
    pub no_alloc: bool,
    /// Número de dependencias externas en `[dependencies]`.
    pub external_dependencies_count: usize,
    /// Capacidad `N` del historial circular considerado en el presupuesto estático.
    pub history_capacity: usize,
    /// Tamaño estático en bytes de [`BoundedControlCycle`] para `history_capacity`.
    pub static_cycle_bytes: usize,
    /// Tamaño estático en bytes de cada [`ControlSample`].
    pub static_sample_bytes: usize,
    /// Tamaño estático en bytes de [`ActionClampGuard`].
    pub static_guard_bytes: usize,
    /// Plazo objetivo de referencia por ciclo en microsegundos.
    pub target_deadline_micros: u64,
    /// Alcance cubierto por este subset respecto al análisis temporal.
    pub wcet_analysis_scope: &'static str,
    /// Requisitos de RTOS, interrupciones y plataforma necesarios para WCET real.
    pub rtos_and_interrupt_requirements: &'static [&'static str],
    /// Política de no certificación hard real-time sin WCET y plataforma concreta.
    pub non_certification_notice: &'static str,
}

impl EmbeddedAnalysisScope {
    /// Indica si este alcance certifica por sí solo tiempo real estricto (siempre `false`).
    pub const fn certifies_hard_realtime(&self) -> bool {
        false
    }
}

/// Calcula en tiempo de compilación el presupuesto de memoria estática en bytes
/// de [`BoundedControlCycle<N>`] mediante [`core::mem::size_of`].
pub const fn static_memory_budget_bytes<const N: usize>() -> usize {
    size_of::<BoundedControlCycle<N>>()
}

/// Construye en tiempo de compilación un [`EmbeddedAnalysisScope`] parametrizado
/// por la capacidad `N` del buffer circular y el plazo objetivo en microsegundos.
pub const fn analysis_scope_for_capacity<const N: usize>(
    target_deadline_micros: u64,
) -> EmbeddedAnalysisScope {
    EmbeddedAnalysisScope {
        crate_name: "gogamen-embedded",
        no_std: true,
        no_alloc: true,
        external_dependencies_count: 0,
        history_capacity: N,
        static_cycle_bytes: size_of::<BoundedControlCycle<N>>(),
        static_sample_bytes: size_of::<ControlSample>(),
        static_guard_bytes: size_of::<ActionClampGuard>(),
        target_deadline_micros,
        wcet_analysis_scope: "Subset puro no_std/no_alloc de control PD de referencia con complejidad O(1) por paso, memoria estática acotada y sin E/S, red, JVM ni heap; no incluye análisis WCET de código máquina ni jitter de plataforma.",
        rtos_and_interrupt_requirements: DEFAULT_RTOS_AND_INTERRUPT_REQUIREMENTS,
        non_certification_notice: NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE,
    }
}

/// Alcance estático de referencia para el subset embebido (`N = 32`, plazo `1000 us`).
pub const EMBEDDED_WCET_AND_RTOS_SCOPE: EmbeddedAnalysisScope = analysis_scope_for_capacity::<
    DEFAULT_EMBEDDED_HISTORY_CAPACITY,
>(DEFAULT_EMBEDDED_DEADLINE_MICROS);

/// Errores de configuración en tiempo de inicialización del lazo embebido.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddedControlError {
    /// El rango `[min_action, max_action]` es inválido o no finito.
    InvalidActionBounds,
    /// El valor de fallback seguro es no finito o queda fuera de `[min_action, max_action]`.
    InvalidSafeFallback,
    /// Alguna ganancia PD o el paso temporal `dt_secs` es inválido (`<= 0` o no finito).
    InvalidControllerConfig,
}

impl fmt::Display for EmbeddedControlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidActionBounds => {
                f.write_str("invalid action bounds: min/max must be finite with min <= max")
            }
            Self::InvalidSafeFallback => {
                f.write_str("invalid safe fallback: must be finite and within [min, max]")
            }
            Self::InvalidControllerConfig => {
                f.write_str("invalid PD controller config: gains must be finite and dt_secs > 0")
            }
        }
    }
}

/// Buffer circular de capacidad fija `N` alojado íntegramente en stack o memoria estática.
///
/// Todas las operaciones (`push`, `pop`, `get`, `latest`) se ejecutan en tiempo
/// constante `O(1)` sin asignaciones dinámicas ni código `unsafe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FixedRingBuffer<T, const N: usize> {
    slots: [Option<T>; N],
    head: usize,
    len: usize,
}

impl<T, const N: usize> Default for FixedRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> FixedRingBuffer<T, N> {
    /// Crea un buffer circular vacío en tiempo de compilación.
    pub const fn new() -> Self {
        Self {
            slots: [const { None }; N],
            head: 0,
            len: 0,
        }
    }

    /// Capacidad máxima fija `N`.
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Número actual de elementos almacenados (`<= N`).
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Indica si el buffer está vacío.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Indica si el buffer alcanzó su capacidad máxima `N`.
    pub const fn is_full(&self) -> bool {
        self.len == N
    }

    /// Inserta un elemento al final del buffer en `O(1)`.
    ///
    /// Si el buffer está lleno, sobrescribe el elemento más antiguo y lo devuelve.
    /// Si `N == 0`, devuelve `Some(value)` sin almacenar nada.
    pub fn push(&mut self, value: T) -> Option<T> {
        if N == 0 {
            return Some(value);
        }

        if self.len < N {
            let tail = (self.head + self.len) % N;
            self.slots[tail] = Some(value);
            self.len += 1;
            None
        } else {
            let evicted = self.slots[self.head].take();
            self.slots[self.head] = Some(value);
            self.head = (self.head + 1) % N;
            evicted
        }
    }

    /// Extrae el elemento más antiguo en `O(1)`.
    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let value = self.slots[self.head].take();
        self.head = (self.head + 1) % N;
        self.len -= 1;
        value
    }

    /// Obtiene una referencia al elemento en `index` (`0` = más antiguo) en `O(1)`.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        self.slots[(self.head + index) % N].as_ref()
    }

    /// Obtiene una referencia al último elemento insertado en `O(1)`.
    pub fn latest(&self) -> Option<&T> {
        if self.len == 0 {
            return None;
        }
        self.get(self.len - 1)
    }

    /// Vacía el buffer en `O(N)` sin liberar ni reasignar memoria.
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
        self.head = 0;
        self.len = 0;
    }
}

/// Resultado de aplicar [`ActionClampGuard`] sobre una acción calculada.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClampStatus {
    /// La acción estaba dentro de `[min_action, max_action]`.
    WithinBounds,
    /// La acción fue saturada al límite inferior `min_action`.
    ClampedToMin,
    /// La acción fue saturada al límite superior `max_action`.
    ClampedToMax,
    /// Se activó el fallback seguro por valor no finito (`NaN` / `±inf`) o por plazo excedido.
    SafeFallback,
}

/// Resultado de [`ActionClampGuard::apply`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GuardedAction {
    /// Valor final acotado y finito listo para actuador.
    pub action: f32,
    /// Estado del guardia aplicado.
    pub status: ClampStatus,
}

/// Guardia determinista de límites para acciones de control y fallback seguro.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionClampGuard {
    min_action: f32,
    max_action: f32,
    safe_fallback: f32,
}

impl ActionClampGuard {
    /// Crea un guardia validando límites finitos y que `safe_fallback` esté dentro del rango.
    pub fn new(
        min_action: f32,
        max_action: f32,
        safe_fallback: f32,
    ) -> Result<Self, EmbeddedControlError> {
        if !min_action.is_finite() || !max_action.is_finite() || min_action > max_action {
            return Err(EmbeddedControlError::InvalidActionBounds);
        }
        if !safe_fallback.is_finite() || safe_fallback < min_action || safe_fallback > max_action {
            return Err(EmbeddedControlError::InvalidSafeFallback);
        }
        Ok(Self {
            min_action,
            max_action,
            safe_fallback,
        })
    }

    /// Límite inferior permitido.
    pub const fn min_action(&self) -> f32 {
        self.min_action
    }

    /// Límite superior permitido.
    pub const fn max_action(&self) -> f32 {
        self.max_action
    }

    /// Acción de fallback seguro garantizada dentro de `[min_action, max_action]`.
    pub const fn safe_fallback(&self) -> f32 {
        self.safe_fallback
    }

    /// Acota `raw_action` dentro de `[min_action, max_action]`.
    ///
    /// Si `raw_action` no es finito (`NaN` o infinito), devuelve `safe_fallback`.
    pub fn apply(&self, raw_action: f32) -> GuardedAction {
        if !raw_action.is_finite() {
            return GuardedAction {
                action: self.safe_fallback,
                status: ClampStatus::SafeFallback,
            };
        }
        if raw_action < self.min_action {
            GuardedAction {
                action: self.min_action,
                status: ClampStatus::ClampedToMin,
            }
        } else if raw_action > self.max_action {
            GuardedAction {
                action: self.max_action,
                status: ClampStatus::ClampedToMax,
            }
        } else {
            GuardedAction {
                action: raw_action,
                status: ClampStatus::WithinBounds,
            }
        }
    }

    /// Fuerza la salida segura configurada.
    pub const fn fallback(&self) -> GuardedAction {
        GuardedAction {
            action: self.safe_fallback,
            status: ClampStatus::SafeFallback,
        }
    }
}

/// Configuración del controlador PD discreto de referencia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdControllerConfig {
    /// Ganancia proporcional `Kp`.
    pub kp: f32,
    /// Ganancia derivativa `Kd`.
    pub kd: f32,
    /// Paso temporal nominal `dt` en segundos (`> 0.0`).
    pub dt_secs: f32,
}

impl PdControllerConfig {
    /// Valida y construye una configuración PD.
    pub fn new(kp: f32, kd: f32, dt_secs: f32) -> Result<Self, EmbeddedControlError> {
        if !kp.is_finite() || !kd.is_finite() || !dt_secs.is_finite() || dt_secs <= 0.0 {
            return Err(EmbeddedControlError::InvalidControllerConfig);
        }
        Ok(Self { kp, kd, dt_secs })
    }
}

/// Muestra compacta almacenada en [`FixedRingBuffer`] por cada ciclo de control.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlSample {
    /// Número secuencial de tick.
    pub tick: u64,
    /// Error de seguimiento (`target - measurement`).
    pub error: f32,
    /// Acción final emitida tras [`ActionClampGuard`].
    pub action: f32,
    /// Duración observada del ciclo en microsegundos.
    pub elapsed_micros: u64,
    /// Estado del guardia en este ciclo.
    pub clamp_status: ClampStatus,
    /// Indica si el ciclo excedió `deadline_micros`.
    pub deadline_missed: bool,
}

/// Salida determinista de un paso de [`BoundedControlCycle::step`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControlStepOutput {
    /// Número secuencial de tick ejecutado.
    pub tick: u64,
    /// Acción acotada final lista para el actuador.
    pub action: f32,
    /// Error calculado (`0.0` si las entradas no fueron finitas).
    pub error: f32,
    /// Estado del guardia aplicado.
    pub clamp_status: ClampStatus,
    /// Indica si `elapsed_micros > deadline_micros` (cuando `deadline_micros > 0`).
    pub deadline_missed: bool,
    /// Indica si se activó el fallback seguro en este paso.
    pub used_fallback: bool,
}

/// Lazo de control PD de referencia en tiempo acotado `O(1)` y memoria estática fija `N`.
#[derive(Debug, Clone)]
pub struct BoundedControlCycle<const N: usize> {
    config: PdControllerConfig,
    guard: ActionClampGuard,
    history: FixedRingBuffer<ControlSample, N>,
    prev_error: f32,
    has_prev_error: bool,
    tick_count: u64,
    deadline_misses: u64,
    fallback_count: u64,
    clamp_count: u64,
    max_observed_micros: u64,
}

impl<const N: usize> BoundedControlCycle<N> {
    /// Crea un lazo de control acotado con historial de capacidad fija `N`.
    pub const fn new(config: PdControllerConfig, guard: ActionClampGuard) -> Self {
        Self {
            config,
            guard,
            history: FixedRingBuffer::new(),
            prev_error: 0.0,
            has_prev_error: false,
            tick_count: 0,
            deadline_misses: 0,
            fallback_count: 0,
            clamp_count: 0,
            max_observed_micros: 0,
        }
    }

    /// Ejecuta un paso determinista del lazo PD con guardia de límites y fallback seguro.
    ///
    /// Si `measurement` o `target` no son finitos, si la salida PD no es finita, o si
    /// `deadline_micros > 0 && elapsed_micros > deadline_micros`, se emite la acción
    /// `safe_fallback` del [`ActionClampGuard`] y se contabiliza el evento sin asignar memoria.
    pub fn step(
        &mut self,
        measurement: f32,
        target: f32,
        elapsed_micros: u64,
        deadline_micros: u64,
    ) -> ControlStepOutput {
        self.tick_count = self.tick_count.saturating_add(1);
        if elapsed_micros > self.max_observed_micros {
            self.max_observed_micros = elapsed_micros;
        }

        let deadline_missed = deadline_micros > 0 && elapsed_micros > deadline_micros;
        if deadline_missed {
            self.deadline_misses = self.deadline_misses.saturating_add(1);
        }

        let inputs_valid = measurement.is_finite() && target.is_finite();
        let (error, guarded) = if !inputs_valid || deadline_missed {
            self.has_prev_error = false;
            self.prev_error = 0.0;
            (0.0, self.guard.fallback())
        } else {
            let error = target - measurement;
            let derivative = if self.has_prev_error {
                (error - self.prev_error) / self.config.dt_secs
            } else {
                0.0
            };
            let raw_action = self.config.kp * error + self.config.kd * derivative;
            let guarded = self.guard.apply(raw_action);

            if guarded.status == ClampStatus::SafeFallback {
                self.has_prev_error = false;
                self.prev_error = 0.0;
            } else {
                self.prev_error = error;
                self.has_prev_error = true;
            }
            (error, guarded)
        };

        let used_fallback = guarded.status == ClampStatus::SafeFallback;
        if used_fallback {
            self.fallback_count = self.fallback_count.saturating_add(1);
        }
        if matches!(
            guarded.status,
            ClampStatus::ClampedToMin | ClampStatus::ClampedToMax
        ) {
            self.clamp_count = self.clamp_count.saturating_add(1);
        }

        let sample = ControlSample {
            tick: self.tick_count,
            error,
            action: guarded.action,
            elapsed_micros,
            clamp_status: guarded.status,
            deadline_missed,
        };
        let _ = self.history.push(sample);

        ControlStepOutput {
            tick: self.tick_count,
            action: guarded.action,
            error,
            clamp_status: guarded.status,
            deadline_missed,
            used_fallback,
        }
    }

    /// Reinicia contadores e historial manteniendo intacta la memoria estática asignada.
    pub fn reset(&mut self) {
        self.history.clear();
        self.prev_error = 0.0;
        self.has_prev_error = false;
        self.tick_count = 0;
        self.deadline_misses = 0;
        self.fallback_count = 0;
        self.clamp_count = 0;
        self.max_observed_micros = 0;
    }

    /// Referencia al historial circular de muestras recientes.
    pub const fn history(&self) -> &FixedRingBuffer<ControlSample, N> {
        &self.history
    }

    /// Total de ticks ejecutados.
    pub const fn tick_count(&self) -> u64 {
        self.tick_count
    }

    /// Total de plazos excedidos.
    pub const fn deadline_misses(&self) -> u64 {
        self.deadline_misses
    }

    /// Total de activaciones de fallback seguro.
    pub const fn fallback_count(&self) -> u64 {
        self.fallback_count
    }

    /// Total de saturaciones aplicadas por [`ActionClampGuard`].
    pub const fn clamp_count(&self) -> u64 {
        self.clamp_count
    }

    /// Duración máxima observada entre pasos (en microsegundos).
    pub const fn max_observed_micros(&self) -> u64 {
        self.max_observed_micros
    }

    /// Política explícita de no certificación hard real-time sin WCET/plataforma.
    pub const fn non_certification_notice(&self) -> &'static str {
        NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::thread_local;

    struct CountingAllocator;

    // Contador por hilo: los tests del harness corren en paralelo y el contador global
    // mezclaba asignaciones ajenas a la región medida.
    thread_local! {
        static ALLOC_CALLS: Cell<usize> = const { Cell::new(0) };
    }

    fn alloc_calls() -> usize {
        ALLOC_CALLS.with(Cell::get)
    }

    #[global_allocator]
    static GLOBAL: CountingAllocator = CountingAllocator;

    #[allow(unsafe_code)]
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            ALLOC_CALLS.with(|calls| calls.set(calls.get() + 1));
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    #[test]
    fn ac1_fixed_ring_buffer_overwrites_in_o1_without_heap() {
        let before = alloc_calls();

        let mut ring: FixedRingBuffer<u32, 3> = FixedRingBuffer::new();
        assert!(ring.is_empty());
        assert_eq!(ring.capacity(), 3);

        assert_eq!(ring.push(10), None);
        assert_eq!(ring.push(20), None);
        assert_eq!(ring.push(30), None);
        assert!(ring.is_full());
        assert_eq!(ring.latest(), Some(&30));

        assert_eq!(ring.push(40), Some(10));
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.get(0), Some(&20));
        assert_eq!(ring.get(1), Some(&30));
        assert_eq!(ring.get(2), Some(&40));

        assert_eq!(ring.pop(), Some(20));
        assert_eq!(ring.pop(), Some(30));
        assert_eq!(ring.pop(), Some(40));
        assert_eq!(ring.pop(), None);
        assert!(ring.is_empty());

        let after = alloc_calls();
        assert_eq!(
            before, after,
            "FixedRingBuffer must perform 0 heap allocations"
        );
    }

    #[test]
    fn ac1_bounded_control_cycle_executes_deterministically_with_zero_allocations() {
        let config = PdControllerConfig::new(2.0, 0.5, 0.01).expect("valid PD config");
        let guard = ActionClampGuard::new(-5.0, 5.0, 0.0).expect("valid guard");

        let mut cycle_a: BoundedControlCycle<8> = BoundedControlCycle::new(config, guard);
        let mut cycle_b: BoundedControlCycle<8> = BoundedControlCycle::new(config, guard);

        let before = alloc_calls();

        let out_a1 = cycle_a.step(0.0, 1.0, 120, 1_000);
        let out_b1 = cycle_b.step(0.0, 1.0, 120, 1_000);
        assert_eq!(out_a1, out_b1);
        assert_eq!(out_a1.action, 2.0);
        assert_eq!(out_a1.clamp_status, ClampStatus::WithinBounds);
        assert!(!out_a1.used_fallback);

        // Large target change saturates at max_action = 5.0
        let out_a2 = cycle_a.step(0.0, 10.0, 150, 1_000);
        let out_b2 = cycle_b.step(0.0, 10.0, 150, 1_000);
        assert_eq!(out_a2, out_b2);
        assert_eq!(out_a2.action, 5.0);
        assert_eq!(out_a2.clamp_status, ClampStatus::ClampedToMax);
        assert_eq!(cycle_a.clamp_count(), 1);

        // Non-finite input triggers safe fallback
        let out_nan = cycle_a.step(f32::NAN, 1.0, 100, 1_000);
        assert_eq!(out_nan.action, 0.0);
        assert_eq!(out_nan.clamp_status, ClampStatus::SafeFallback);
        assert!(out_nan.used_fallback);

        // Deadline miss triggers safe fallback and increments deadline_misses
        let out_miss = cycle_a.step(0.5, 1.0, 1_250, 1_000);
        assert_eq!(out_miss.action, 0.0);
        assert_eq!(out_miss.clamp_status, ClampStatus::SafeFallback);
        assert!(out_miss.deadline_missed);
        assert!(out_miss.used_fallback);
        assert_eq!(cycle_a.deadline_misses(), 1);
        assert_eq!(cycle_a.fallback_count(), 2);
        assert_eq!(cycle_a.max_observed_micros(), 1_250);
        assert_eq!(cycle_a.history().len(), 4);

        let after = alloc_calls();
        assert_eq!(
            before, after,
            "BoundedControlCycle::step must perform 0 heap allocations"
        );
    }

    #[test]
    fn ac2_scope_documents_static_memory_deadlines_dependencies_and_rtos_requirements() {
        let scope = EMBEDDED_WCET_AND_RTOS_SCOPE;
        assert_eq!(scope.crate_name, "gogamen-embedded");
        assert!(scope.no_std);
        assert!(scope.no_alloc);
        assert_eq!(scope.external_dependencies_count, 0);
        assert_eq!(
            scope.static_cycle_bytes,
            static_memory_budget_bytes::<DEFAULT_EMBEDDED_HISTORY_CAPACITY>()
        );
        assert!(scope.static_cycle_bytes > 0);
        assert!(scope.static_sample_bytes > 0);
        assert!(scope.static_guard_bytes > 0);
        assert_eq!(
            scope.target_deadline_micros,
            DEFAULT_EMBEDDED_DEADLINE_MICROS
        );
        assert!(!scope.wcet_analysis_scope.is_empty());
        assert!(scope.rtos_and_interrupt_requirements.len() >= 3);

        let joined = scope.rtos_and_interrupt_requirements.join(" ");
        assert!(
            joined.contains("FreeRTOS") || joined.contains("Zephyr") || joined.contains("Embassy")
        );
        assert!(joined.contains("WCET"));
        assert!(!scope.certifies_hard_realtime());
    }

    #[test]
    fn ac3_no_hard_realtime_guarantee_without_wcet_and_platform_notice() {
        assert!(!NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE.is_empty());
        assert!(NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE.contains("hard real-time"));
        assert!(NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE.contains("WCET"));
        assert_eq!(
            EMBEDDED_WCET_AND_RTOS_SCOPE.non_certification_notice,
            NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE
        );

        let config = PdControllerConfig::new(1.0, 0.1, 0.01).expect("valid config");
        let guard = ActionClampGuard::new(-1.0, 1.0, 0.0).expect("valid guard");
        let cycle: BoundedControlCycle<4> = BoundedControlCycle::new(config, guard);
        assert_eq!(
            cycle.non_certification_notice(),
            NO_HARD_REALTIME_WITHOUT_WCET_AND_PLATFORM_NOTICE
        );
    }
}
