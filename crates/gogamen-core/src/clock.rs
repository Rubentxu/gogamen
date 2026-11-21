//! Relojes monotónico y virtual, plazos monotónicos y trazas por unidades de trabajo (`std`-only).
//!
//! Este módulo separa explícitamente el tiempo monotónico (para plazos, presupuestos y
//! medición de intervalos) del tiempo civil/wall-clock ([`WallClockSample`]), que nunca
//! debe usarse para expirar [`MonotonicDeadline`]:
//! - [`MonotonicInstant`]: instante en nanosegundos monotónicos (`u128`), nunca `SystemTime`.
//! - [`MonotonicClock`]: contrato común de lectura temporal monotónica (`now(&self)`).
//! - [`VirtualMonotonicClock`]: reloj de fixture determinista que avanza sin dormir.
//! - [`SystemMonotonicClock`]: reloj monotónico de sistema anclado a un origen fijo.
//! - [`MonotonicDeadline`]: plazo relativo o absoluto inmune a saltos del reloj civil.
//! - [`WorkUnits`] y [`TemporalTrace`]: registro `(tick, work_units, monotonic_ns)` para
//!   distinguir durante un replay por unidades de trabajo la variabilidad temporal real
//!   no determinista frente a una ejecución puramente determinista.

use std::fmt;
use std::sync::Mutex;
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};

use crate::units::Tick;

const NANOS_PER_SEC_U128: u128 = 1_000_000_000;

/// Instante monotónico expresado en nanosegundos desde un origen fijo de reloj (`u128`).
///
/// Nunca representa tiempo civil (`SystemTime`) ni fecha calendario.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MonotonicInstant(u128);

impl MonotonicInstant {
    /// Instante origen (`0` nanosegundos).
    pub const ZERO: Self = Self(0);

    /// Crea un [`MonotonicInstant`] a partir de nanosegundos monotónicos.
    pub const fn from_nanos(nanos: u128) -> Self {
        Self(nanos)
    }

    /// Devuelve el valor en nanosegundos desde el origen del reloj.
    pub const fn as_nanos(self) -> u128 {
        self.0
    }

    /// Suma `duration` de forma verificada, devolviendo `None` si hay desbordamiento.
    pub const fn checked_add(self, duration: StdDuration) -> Option<Self> {
        match self.0.checked_add(duration.as_nanos()) {
            Some(total) => Some(Self(total)),
            None => None,
        }
    }

    /// Suma `duration` saturando en `u128::MAX` si hubiera desbordamiento.
    pub const fn saturating_add(self, duration: StdDuration) -> Self {
        Self(self.0.saturating_add(duration.as_nanos()))
    }

    /// Devuelve la duración transcurrida desde `earlier`, o `None` si `self < earlier`.
    pub fn checked_duration_since(self, earlier: Self) -> Option<StdDuration> {
        let diff = self.0.checked_sub(earlier.0)?;
        let secs = u64::try_from(diff / NANOS_PER_SEC_U128).ok()?;
        let subsec_nanos = (diff % NANOS_PER_SEC_U128) as u32;
        Some(StdDuration::new(secs, subsec_nanos))
    }

    /// Devuelve la duración transcurrida desde `earlier`, saturando a `Duration::ZERO` si `self < earlier`
    /// o a `Duration::MAX` si excede el rango representable por `std::time::Duration`.
    pub fn saturating_duration_since(self, earlier: Self) -> StdDuration {
        let diff = self.0.saturating_sub(earlier.0);
        let secs = diff / NANOS_PER_SEC_U128;
        let subsec_nanos = (diff % NANOS_PER_SEC_U128) as u32;
        match u64::try_from(secs) {
            Ok(s) => StdDuration::new(s, subsec_nanos),
            Err(_) => StdDuration::MAX,
        }
    }
}

impl fmt::Display for MonotonicInstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ns", self.0)
    }
}

/// Contrato de lectura de tiempo monotónico para producción y fixtures deterministas.
pub trait MonotonicClock {
    /// Devuelve el instante monotónico actual del reloj.
    fn now(&self) -> MonotonicInstant;
}

/// Reloj monotónico virtual para tests y fixtures deterministas.
///
/// Permite avanzar el tiempo explícitamente mediante [`VirtualMonotonicClock::advance`]
/// sin dormir hilos ni depender del reloj del sistema operativo.
#[derive(Debug, Default)]
pub struct VirtualMonotonicClock {
    now_nanos: Mutex<u128>,
}

impl VirtualMonotonicClock {
    /// Crea un reloj virtual iniciado en [`MonotonicInstant::ZERO`].
    pub fn new() -> Self {
        Self {
            now_nanos: Mutex::new(0),
        }
    }

    /// Crea un reloj virtual iniciado en un instante monotónico específico.
    pub fn at(initial: MonotonicInstant) -> Self {
        Self {
            now_nanos: Mutex::new(initial.as_nanos()),
        }
    }

    /// Avanza el reloj virtual en `duration` sin dormir y devuelve el nuevo instante.
    pub fn advance(&self, duration: StdDuration) -> MonotonicInstant {
        self.advance_nanos(duration.as_nanos())
    }

    /// Avanza el reloj virtual en `nanos` nanosegundos y devuelve el nuevo instante.
    pub fn advance_nanos(&self, nanos: u128) -> MonotonicInstant {
        let mut guard = self
            .now_nanos
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        *guard = guard.saturating_add(nanos);
        MonotonicInstant::from_nanos(*guard)
    }
}

impl MonotonicClock for VirtualMonotonicClock {
    fn now(&self) -> MonotonicInstant {
        let guard = self
            .now_nanos
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        MonotonicInstant::from_nanos(*guard)
    }
}

/// Reloj monotónico real respaldado por `std::time::Instant`.
///
/// Anclado a un origen fijo creado en [`SystemMonotonicClock::new`], de modo que
/// retrocesos o ajustes NTP del reloj civil (`SystemTime`) no afectan sus lecturas.
#[derive(Debug, Clone)]
pub struct SystemMonotonicClock {
    origin: Instant,
}

impl SystemMonotonicClock {
    /// Crea un reloj de sistema cuyo origen (`0 ns`) corresponde al instante de creación.
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Default for SystemMonotonicClock {
    fn default() -> Self {
        Self::new()
    }
}

impl MonotonicClock for SystemMonotonicClock {
    fn now(&self) -> MonotonicInstant {
        MonotonicInstant::from_nanos(self.origin.elapsed().as_nanos())
    }
}

/// Muestra informativa de reloj civil (`SystemTime`), explícitamente separada de plazos.
///
/// Útil únicamente para telemetría o diagnóstico humano, y en tests para simular saltos o
/// retrocesos del reloj civil sin afectar [`MonotonicDeadline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WallClockSample {
    unix_nanos: i128,
}

impl WallClockSample {
    /// Captura el instante actual del reloj civil del sistema (`SystemTime::now()`).
    pub fn now() -> Self {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => Self {
                unix_nanos: i128::try_from(d.as_nanos()).unwrap_or(i128::MAX),
            },
            Err(err) => {
                let d = err.duration();
                let abs = i128::try_from(d.as_nanos()).unwrap_or(i128::MAX);
                Self { unix_nanos: -abs }
            }
        }
    }

    /// Construye una muestra civil a partir de nanosegundos respecto a `UNIX_EPOCH`.
    pub const fn from_unix_nanos(unix_nanos: i128) -> Self {
        Self { unix_nanos }
    }

    /// Devuelve los nanosegundos respecto a `UNIX_EPOCH` (pueden ser negativos o retroceder).
    pub const fn unix_nanos(self) -> i128 {
        self.unix_nanos
    }
}

/// Plazo monotónico inmutable basado exclusivamente en [`MonotonicInstant`] y [`MonotonicClock`].
///
/// Al depender únicamente de un reloj monotónico, los retrocesos o saltos hacia adelante
/// del reloj civil (`SystemTime` / [`WallClockSample`]) nunca alteran ni expiran el plazo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicDeadline {
    target: MonotonicInstant,
}

impl MonotonicDeadline {
    /// Construye un plazo absoluto en el instante monotónico `target`.
    pub const fn at(target: MonotonicInstant) -> Self {
        Self { target }
    }

    /// Construye un plazo relativo a `clock.now() + duration`.
    pub fn from_now<C: MonotonicClock + ?Sized>(clock: &C, duration: StdDuration) -> Self {
        Self {
            target: clock.now().saturating_add(duration),
        }
    }

    /// Devuelve el instante monotónico objetivo del plazo.
    pub const fn target(self) -> MonotonicInstant {
        self.target
    }

    /// Indica si el plazo ha expirado respecto al instante `now`.
    pub const fn is_expired_at(self, now: MonotonicInstant) -> bool {
        now.0 >= self.target.0
    }

    /// Indica si el plazo ha expirado leyendo el reloj monotónico `clock`.
    pub fn is_expired<C: MonotonicClock + ?Sized>(self, clock: &C) -> bool {
        self.is_expired_at(clock.now())
    }

    /// Devuelve el tiempo restante hasta el plazo respecto a `now` (`Duration::ZERO` si expiró).
    pub fn remaining_at(self, now: MonotonicInstant) -> StdDuration {
        self.target.saturating_duration_since(now)
    }

    /// Devuelve el tiempo restante hasta el plazo leyendo `clock` (`Duration::ZERO` si expiró).
    pub fn remaining<C: MonotonicClock + ?Sized>(self, clock: &C) -> StdDuration {
        self.remaining_at(clock.now())
    }
}

/// Conteo discreto de unidades de trabajo ejecutadas en simulación o planificación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct WorkUnits(u64);

impl WorkUnits {
    /// Cero unidades de trabajo.
    pub const ZERO: Self = Self(0);

    /// Crea un contador de unidades de trabajo.
    pub const fn new(units: u64) -> Self {
        Self(units)
    }

    /// Devuelve el valor numérico de unidades de trabajo.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Suma `delta` unidades de trabajo de forma verificada.
    pub const fn checked_add(self, delta: u64) -> Option<Self> {
        match self.0.checked_add(delta) {
            Some(total) => Some(Self(total)),
            None => None,
        }
    }

    /// Suma `delta` unidades de trabajo saturando en `u64::MAX`.
    pub const fn saturating_add(self, delta: u64) -> Self {
        Self(self.0.saturating_add(delta))
    }
}

impl fmt::Display for WorkUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} work_units", self.0)
    }
}

/// Entrada individual de una traza temporal: `(tick, work_units, monotonic_ns)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TemporalTraceEntry {
    tick: Tick,
    work_units: WorkUnits,
    monotonic_ns: u128,
}

impl TemporalTraceEntry {
    /// Construye una entrada de traza temporal con valores explícitos.
    pub const fn new(tick: Tick, work_units: WorkUnits, monotonic_ns: u128) -> Self {
        Self {
            tick,
            work_units,
            monotonic_ns,
        }
    }

    /// Construye una entrada de traza a partir de un [`MonotonicInstant`].
    pub const fn from_instant(
        tick: Tick,
        work_units: WorkUnits,
        instant: MonotonicInstant,
    ) -> Self {
        Self {
            tick,
            work_units,
            monotonic_ns: instant.as_nanos(),
        }
    }

    /// Tick lógico asociado a la muestra.
    pub const fn tick(self) -> Tick {
        self.tick
    }

    /// Unidades de trabajo acumuladas o ejecutadas en la muestra.
    pub const fn work_units(self) -> WorkUnits {
        self.work_units
    }

    /// Marca temporal monotónica en nanosegundos.
    pub const fn monotonic_ns(self) -> u128 {
        self.monotonic_ns
    }
}

/// Resultado de comparar dos [`TemporalTrace`] durante un replay por unidades de trabajo.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalReplayVerdict {
    /// Tanto `(tick, work_units)` como los deltas monotónicos coinciden exactamente.
    DeterministicMatch,
    /// Los `(tick, work_units)` coinciden exactamente, pero los tiempos monotónicos difieren
    /// (indica variabilidad temporal real no determinista).
    WorkDeterministicWithTemporalVariance {
        /// Diferencia máxima en nanosegundos observada entre marcas monotónicas equivalentes.
        max_ns_diff: u128,
        /// Índices de entrada donde la marca o delta temporal difiere.
        divergent_indices: Vec<usize>,
    },
    /// El replay difiere en longitud o en `(tick, work_units)`.
    WorkMismatch {
        /// Primer índice donde se detectó la divergencia de trabajo o longitud.
        index: usize,
    },
}

impl TemporalReplayVerdict {
    /// Indica si la secuencia `(tick, work_units)` es idéntica (con o sin jitter temporal).
    pub const fn is_work_deterministic(&self) -> bool {
        matches!(
            self,
            Self::DeterministicMatch | Self::WorkDeterministicWithTemporalVariance { .. }
        )
    }

    /// Indica si se detectó variabilidad temporal no determinista entre ambas trazas.
    pub const fn has_temporal_variability(&self) -> bool {
        matches!(self, Self::WorkDeterministicWithTemporalVariance { .. })
    }
}

/// Registro ordenado de muestras `(tick, work_units, monotonic_ns)` para análisis de replay.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TemporalTrace {
    entries: Vec<TemporalTraceEntry>,
}

impl TemporalTrace {
    /// Crea una traza temporal vacía.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Registra una entrada leyendo el instante actual de `clock`.
    pub fn record_with_clock<C: MonotonicClock + ?Sized>(
        &mut self,
        tick: Tick,
        work_units: WorkUnits,
        clock: &C,
    ) {
        self.record(tick, work_units, clock.now());
    }

    /// Registra una entrada con un instante monotónico explícito.
    pub fn record(&mut self, tick: Tick, work_units: WorkUnits, instant: MonotonicInstant) {
        self.entries
            .push(TemporalTraceEntry::from_instant(tick, work_units, instant));
    }

    /// Devuelve las entradas registradas en orden cronológico.
    pub fn entries(&self) -> &[TemporalTraceEntry] {
        &self.entries
    }

    /// Número de entradas registradas.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Indica si la traza está vacía.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Indica si alguna marca `monotonic_ns` retrocedió respecto a la entrada anterior.
    pub fn has_monotonic_regression(&self) -> bool {
        self.entries
            .windows(2)
            .any(|pair| pair[1].monotonic_ns < pair[0].monotonic_ns)
    }

    /// Indica si el tiempo monotónico avanza de forma estrictamente proporcional a `work_units`
    /// entre pasos consecutivos con incremento de trabajo positivo.
    pub fn is_strictly_proportional_to_work(&self) -> bool {
        let mut expected_ns_per_unit: Option<u128> = None;
        for pair in self.entries.windows(2) {
            let dw = pair[1]
                .work_units
                .as_u64()
                .saturating_sub(pair[0].work_units.as_u64());
            let Some(dt) = pair[1].monotonic_ns.checked_sub(pair[0].monotonic_ns) else {
                return false;
            };
            if dw == 0 {
                if dt != 0 {
                    return false;
                }
                continue;
            }
            let dw_u128 = u128::from(dw);
            if dt % dw_u128 != 0 {
                return false;
            }
            let ratio = dt / dw_u128;
            match expected_ns_per_unit {
                None => expected_ns_per_unit = Some(ratio),
                Some(expected) if expected != ratio => return false,
                Some(_) => {}
            }
        }
        true
    }

    /// Compara esta traza frente a otra durante un replay por unidades de trabajo.
    ///
    /// Distingue entre:
    /// - ejecución puramente determinista en trabajo y tiempo ([`TemporalReplayVerdict::DeterministicMatch`]),
    /// - ejecución determinista en trabajo pero con variabilidad temporal real
    ///   ([`TemporalReplayVerdict::WorkDeterministicWithTemporalVariance`]),
    /// - divergencia en la secuencia `(tick, work_units)` ([`TemporalReplayVerdict::WorkMismatch`]).
    pub fn compare_work_replay(&self, other: &Self) -> TemporalReplayVerdict {
        let min_len = self.entries.len().min(other.entries.len());
        let mut max_ns_diff: u128 = 0;
        let mut divergent_indices = Vec::new();

        for idx in 0..min_len {
            let a = self.entries[idx];
            let b = other.entries[idx];
            if a.tick != b.tick || a.work_units != b.work_units {
                return TemporalReplayVerdict::WorkMismatch { index: idx };
            }
            let diff = a.monotonic_ns.abs_diff(b.monotonic_ns);
            if diff > 0 {
                max_ns_diff = max_ns_diff.max(diff);
                divergent_indices.push(idx);
            }
        }

        if self.entries.len() != other.entries.len() {
            return TemporalReplayVerdict::WorkMismatch { index: min_len };
        }

        if divergent_indices.is_empty() {
            TemporalReplayVerdict::DeterministicMatch
        } else {
            TemporalReplayVerdict::WorkDeterministicWithTemporalVariance {
                max_ns_diff,
                divergent_indices,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn virtual_clock_advances_and_expires_deadlines_without_sleeping() {
        let clock = VirtualMonotonicClock::new();
        assert_eq!(clock.now(), MonotonicInstant::ZERO);

        let deadline = MonotonicDeadline::from_now(&clock, StdDuration::from_millis(250));
        assert!(!deadline.is_expired(&clock));
        assert_eq!(deadline.remaining(&clock), StdDuration::from_millis(250));

        // Avanza parcialmente sin dormir.
        let t1 = clock.advance(StdDuration::from_millis(100));
        assert_eq!(t1.as_nanos(), 100_000_000);
        assert!(!deadline.is_expired(&clock));
        assert_eq!(deadline.remaining(&clock), StdDuration::from_millis(150));

        // Alcanza exactamente el deadline.
        clock.advance(StdDuration::from_millis(150));
        assert!(deadline.is_expired(&clock));
        assert_eq!(deadline.remaining(&clock), StdDuration::ZERO);

        // Supera el deadline.
        clock.advance(StdDuration::from_millis(50));
        assert!(deadline.is_expired(&clock));
        assert_eq!(deadline.remaining(&clock), StdDuration::ZERO);
    }

    #[test]
    fn civil_clock_jumps_and_rollbacks_do_not_alter_monotonic_deadline() {
        let virtual_clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000_000_000));
        let deadline = MonotonicDeadline::from_now(&virtual_clock, StdDuration::from_secs(5));

        // Simula muestras de reloj civil retrocediendo 1 hora y saltando +10 años.
        let wall_before = WallClockSample::from_unix_nanos(1_700_000_000_000_000_000);
        let wall_rollback =
            WallClockSample::from_unix_nanos(wall_before.unix_nanos() - 3_600 * 1_000_000_000_i128);
        let wall_forward_jump = WallClockSample::from_unix_nanos(
            wall_before.unix_nanos() + 10 * 365 * 86_400 * 1_000_000_000_i128,
        );
        assert!(wall_rollback < wall_before);
        assert!(wall_forward_jump > wall_before);

        // El deadline monotónico no cambia ni expira por los saltos del reloj civil.
        assert_eq!(
            deadline.target(),
            MonotonicInstant::from_nanos(6_000_000_000)
        );
        assert!(!deadline.is_expired(&virtual_clock));
        assert_eq!(
            deadline.remaining(&virtual_clock),
            StdDuration::from_secs(5)
        );

        // Solo el avance del reloj monotónico altera el tiempo restante y la expiración.
        virtual_clock.advance(StdDuration::from_secs(5));
        assert!(deadline.is_expired(&virtual_clock));
        assert_eq!(deadline.remaining(&virtual_clock), StdDuration::ZERO);

        // También verificamos que SystemMonotonicClock avanza monotónicamente.
        let sys_clock = SystemMonotonicClock::new();
        let sys_deadline = MonotonicDeadline::from_now(&sys_clock, StdDuration::from_secs(60));
        assert!(!sys_deadline.is_expired(&sys_clock));
        assert!(sys_deadline.remaining(&sys_clock) <= StdDuration::from_secs(60));
    }

    #[test]
    fn work_unit_replay_distinguishes_deterministic_vs_nondeterministic_temporal_traces() {
        let mut trace_det_a = TemporalTrace::new();
        let mut trace_det_b = TemporalTrace::new();
        let clock_a = VirtualMonotonicClock::new();
        let clock_b = VirtualMonotonicClock::new();

        // Dos ejecuciones deterministas donde el reloj virtual avanza 1_000 ns por unidad de trabajo.
        for step in 0..4_u64 {
            let tick = Tick::new(step);
            let work = WorkUnits::new(step * 10);
            trace_det_a.record_with_clock(tick, work, &clock_a);
            trace_det_b.record_with_clock(tick, work, &clock_b);
            clock_a.advance_nanos(10 * 1_000);
            clock_b.advance_nanos(10 * 1_000);
        }

        assert!(!trace_det_a.has_monotonic_regression());
        assert!(trace_det_a.is_strictly_proportional_to_work());
        assert_eq!(
            trace_det_a.compare_work_replay(&trace_det_b),
            TemporalReplayVerdict::DeterministicMatch
        );

        // Una traza con los mismos (tick, work_units) pero con jitter temporal real no determinista.
        let mut trace_real_jitter = TemporalTrace::new();
        let jitter_ns = [0_u128, 12_500, 19_800, 41_200];
        for (step, &ns) in jitter_ns.iter().enumerate() {
            trace_real_jitter.record(
                Tick::new(step as u64),
                WorkUnits::new((step as u64) * 10),
                MonotonicInstant::from_nanos(ns),
            );
        }

        assert!(!trace_real_jitter.is_strictly_proportional_to_work());
        let verdict = trace_det_a.compare_work_replay(&trace_real_jitter);
        assert!(verdict.is_work_deterministic());
        assert!(verdict.has_temporal_variability());
        assert_eq!(
            verdict,
            TemporalReplayVerdict::WorkDeterministicWithTemporalVariance {
                max_ns_diff: 11_200,
                divergent_indices: vec![1, 2, 3],
            }
        );

        // Una traza que diverge en unidades de trabajo se clasifica como WorkMismatch.
        let mut trace_work_mismatch = TemporalTrace::new();
        trace_work_mismatch.record(Tick::new(0), WorkUnits::new(0), MonotonicInstant::ZERO);
        trace_work_mismatch.record(
            Tick::new(1),
            WorkUnits::new(15),
            MonotonicInstant::from_nanos(10_000),
        );
        assert_eq!(
            trace_det_a.compare_work_replay(&trace_work_mismatch),
            TemporalReplayVerdict::WorkMismatch { index: 1 }
        );
    }
}
