//! Evaluación experimental de perfil Linux de baja latencia (`std`-only, sin `unsafe`).
//!
//! Este módulo inspecciona el entorno Linux actual mediante interfaces estándar de solo
//! lectura (`/proc` y `/sys`), mide cumplimiento temporal (`deadline_misses`, `fallback_counts`,
//! percentiles `p50`/`p95`/`p99`/`max` y jitter) sobre [`gogamen_core::SystemMonotonicClock`]
//! y [`TemporalTelemetryRecorder`] bajo múltiples perfiles de carga, y reporta de forma
//! explícita cuando no existe plataforma `PREEMPT_RT` disponible (`NOT_RUN`).
//!
//! # Política de no certificación en tiempo real
//!
//! Todas las mediciones mantienen [`RuntimeTemporalProfile::DeadlineAware`] y **no**
//! promueven ni certifican tiempo real estricto (hard real-time) ni WCET. Consulte
//! [`STRICT_REALTIME_NON_CERTIFICATION_NOTICE`] y [`NO_WCET_CLAIM_NOTICE`].

use std::error::Error;
use std::fmt;
use std::fs;
use std::hint::black_box;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use gogamen_core::{MonotonicClock, SystemMonotonicClock, Tick};

use crate::multirate::MultirateFallbackReason;
use crate::temporal_profile::{RuntimeTemporalProfile, STRICT_REALTIME_NON_CERTIFICATION_NOTICE};
use crate::temporal_telemetry::{
    NO_WCET_CLAIM_NOTICE, TemporalComplianceReport, TemporalTelemetryConfig,
    TemporalTelemetryError, TemporalTelemetryRecorder,
};

/// Aviso explícito de que el sondeo experimental en Linux conserva `DeadlineAware` y no
/// promueve certificación de tiempo real estricto.
pub const LINUX_RT_NON_CERTIFICATION_NOTE: &str = "La evaluación experimental en Linux conserva RuntimeTemporalProfile::DeadlineAware y no promueve ni certifica tiempo real estricto.";

const DEFAULT_TICKS_PER_PROFILE: usize = 64;
const DEFAULT_TARGET_PERIOD: Duration = Duration::from_millis(1);
const DEFAULT_DEADLINE: Duration = Duration::from_millis(1);
const DEFAULT_NOMINAL_WORK_ITERS: u32 = 1_000;
const DEFAULT_CONTENTION_THREADS: usize = 2;
const DEFAULT_CONTENTION_WORK_ITERS: u32 = 50_000;

/// Errores de configuración o ejecución del sondeo de latencia en Linux.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinuxLatencyProbeError {
    /// Parámetro de configuración inválido.
    InvalidConfig(&'static str),
    /// Error propagado desde [`TemporalTelemetryError`].
    Telemetry(TemporalTelemetryError),
}

impl fmt::Display for LinuxLatencyProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid linux latency probe config: {msg}"),
            Self::Telemetry(err) => write!(f, "telemetry error: {err}"),
        }
    }
}

impl Error for LinuxLatencyProbeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::InvalidConfig(_) => None,
            Self::Telemetry(err) => Some(err),
        }
    }
}

impl From<TemporalTelemetryError> for LinuxLatencyProbeError {
    fn from(value: TemporalTelemetryError) -> Self {
        Self::Telemetry(value)
    }
}

/// Modo de preempción detectado en el kernel Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPreemptMode {
    /// Kernel con parche/configuración `PREEMPT_RT`.
    PreemptRt,
    /// Kernel con `PREEMPT_DYNAMIC`.
    PreemptDynamic,
    /// Kernel con `PREEMPT` voluntario o estándar sin `PREEMPT_RT`.
    PreemptStandard,
    /// Modo de preempción no determinado.
    Unknown,
}

impl fmt::Display for KernelPreemptMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreemptRt => f.write_str("PREEMPT_RT"),
            Self::PreemptDynamic => f.write_str("PREEMPT_DYNAMIC"),
            Self::PreemptStandard => f.write_str("PREEMPT_STANDARD"),
            Self::Unknown => f.write_str("UNKNOWN"),
        }
    }
}

/// Información del kernel Linux inspeccionada desde `/proc` y `/sys`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxKernelInfo {
    /// Cadena de versión del kernel (`/proc/sys/kernel/osrelease`).
    pub kernel_release: String,
    /// Cadena completa de compilación (`/proc/version`).
    pub kernel_version: String,
    /// Indica si se detectó un entorno `PREEMPT_RT`.
    pub preempt_rt_detected: bool,
    /// Modo de preempción inferido.
    pub preempt_mode: KernelPreemptMode,
    /// Valor de `/sys/kernel/realtime` si existe y es legible.
    pub sys_kernel_realtime: Option<bool>,
}

/// Información de CPU inspeccionada en el host Linux.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxCpuInfo {
    /// Arquitectura de compilación (`std::env::consts::ARCH`).
    pub arch: &'static str,
    /// Número de CPUs lógicas/en línea detectadas.
    pub cpu_count: usize,
    /// Contenido crudo de `/sys/devices/system/cpu/online`, si está disponible.
    pub online_cpus_raw: Option<String>,
    /// Lista de CPUs permitidas al proceso (`Cpus_allowed_list` en `/proc/self/status`).
    pub cpus_allowed_list: Option<String>,
    /// Modelo de procesador reportado por `/proc/cpuinfo`.
    pub model_name: String,
}

/// Límite de recursos POSIX leído de `/proc/self/limits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtLimit {
    /// Límite ilimitado (`unlimited`).
    Unlimited,
    /// Límite numérico explícito.
    Value(u64),
}

/// Estado observado de permisos para planificación en tiempo real en Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtSchedulingStatus {
    /// El proceso dispone de `RLIMIT_RTPRIO > 0` o capacidad efectiva `CAP_SYS_NICE`.
    Permitted,
    /// Proceso sin privilegios RT (`RLIMIT_RTPRIO == 0` y sin `CAP_SYS_NICE`).
    UnprivilegedLimited,
    /// No fue posible determinar los permisos RT.
    Unknown,
}

impl fmt::Display for RtSchedulingStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Permitted => f.write_str("PERMITTED"),
            Self::UnprivilegedLimited => f.write_str("UNPRIVILEGED_LIMITED"),
            Self::Unknown => f.write_str("UNKNOWN"),
        }
    }
}

/// Permisos y límites relevantes para baja latencia en el proceso actual.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxRtPermissions {
    /// Estado agregado de permisos de planificación RT.
    pub rt_scheduling_status: RtSchedulingStatus,
    /// Límite blando `Max realtime priority`.
    pub rtprio_soft_limit: Option<RtLimit>,
    /// Límite duro `Max realtime priority`.
    pub rtprio_hard_limit: Option<RtLimit>,
    /// Límite blando `Max locked memory` en bytes.
    pub memlock_soft_bytes: Option<RtLimit>,
    /// Límite duro `Max locked memory` en bytes.
    pub memlock_hard_bytes: Option<RtLimit>,
    /// Máscara hexadecimal `CapEff` de `/proc/self/status`.
    pub cap_eff_hex: Option<String>,
    /// Indica si `CAP_SYS_NICE` (bit 23) está activo en `CapEff`.
    pub cap_sys_nice: Option<bool>,
    /// Indica si `CAP_IPC_LOCK` (bit 14) está activo en `CapEff`.
    pub cap_ipc_lock: Option<bool>,
}

/// Instantánea cruda de ficheros `/proc` y `/sys` para inspección reproducible.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LinuxHostSnapshot {
    /// Contenido de `/proc/sys/kernel/osrelease`.
    pub osrelease: Option<String>,
    /// Contenido de `/proc/version`.
    pub version: Option<String>,
    /// Contenido de `/sys/kernel/realtime`.
    pub sys_kernel_realtime: Option<String>,
    /// Contenido de `/proc/cpuinfo`.
    pub cpuinfo: Option<String>,
    /// Contenido de `/sys/devices/system/cpu/online`.
    pub online_cpus: Option<String>,
    /// Contenido de `/proc/self/status`.
    pub self_status: Option<String>,
    /// Contenido de `/proc/self/limits`.
    pub self_limits: Option<String>,
}

impl LinuxHostSnapshot {
    /// Lee la instantánea actual del host desde `/proc` y `/sys` sin `unsafe`.
    pub fn from_live_host() -> Self {
        Self {
            osrelease: fs::read_to_string("/proc/sys/kernel/osrelease").ok(),
            version: fs::read_to_string("/proc/version").ok(),
            sys_kernel_realtime: fs::read_to_string("/sys/kernel/realtime").ok(),
            cpuinfo: fs::read_to_string("/proc/cpuinfo").ok(),
            online_cpus: fs::read_to_string("/sys/devices/system/cpu/online").ok(),
            self_status: fs::read_to_string("/proc/self/status").ok(),
            self_limits: fs::read_to_string("/proc/self/limits").ok(),
        }
    }
}

/// Perfil completo del host Linux inspeccionado (AC #1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxHostProfile {
    /// Sistema operativo objetivo (`std::env::consts::OS`).
    pub os: &'static str,
    /// Información del kernel y detección de `PREEMPT_RT`.
    pub kernel: LinuxKernelInfo,
    /// Información de CPU y paralelismo disponible.
    pub cpu: LinuxCpuInfo,
    /// Permisos y límites RT observados.
    pub permissions: LinuxRtPermissions,
}

impl LinuxHostProfile {
    /// Inspecciona el host Linux actual de forma reproducible.
    pub fn inspect_live() -> Self {
        let snapshot = LinuxHostSnapshot::from_live_host();
        Self::from_snapshot(&snapshot)
    }

    /// Construye el perfil del host a partir de una [`LinuxHostSnapshot`].
    pub fn from_snapshot(snapshot: &LinuxHostSnapshot) -> Self {
        let kernel = parse_kernel_info(snapshot);
        let cpu = parse_cpu_info(snapshot);
        let permissions = parse_rt_permissions(snapshot);
        Self {
            os: std::env::consts::OS,
            kernel,
            cpu,
            permissions,
        }
    }
}

/// Perfil de carga sintético ejecutado durante el sondeo experimental (AC #2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadProfile {
    /// Carga mínima en reposo relativo.
    Idle,
    /// Carga de control nominal acotada y determinista.
    NominalControl,
    /// Carga con contención concurrente en hilos de fondo.
    HeavyContention,
}

impl LoadProfile {
    /// Conjunto estándar de perfiles de carga evaluados en el sondeo.
    pub const ALL: [Self; 3] = [Self::Idle, Self::NominalControl, Self::HeavyContention];

    /// Etiqueta estable del perfil de carga.
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::NominalControl => "nominal_control",
            Self::HeavyContention => "heavy_contention",
        }
    }
}

impl fmt::Display for LoadProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_label())
    }
}

/// Estado de disponibilidad de plataforma `PREEMPT_RT` para comparación (AC #3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RtPlatformStatus {
    /// No hay plataforma `PREEMPT_RT` disponible; la comparación RT queda marcada como `NOT_RUN`.
    NotRunNoPreemptRtPlatform,
    /// Se detectó un entorno `PREEMPT_RT` para sondeo experimental (sin certificar tiempo real duro).
    PreemptRtAvailable,
}

impl RtPlatformStatus {
    /// Código estable del estado de comparación RT.
    pub const fn as_code(self) -> &'static str {
        match self {
            Self::NotRunNoPreemptRtPlatform => "NOT_RUN",
            Self::PreemptRtAvailable => "PREEMPT_RT_AVAILABLE",
        }
    }

    /// Indica si la comparación RT no se ejecutó por ausencia de plataforma `PREEMPT_RT`.
    pub const fn is_not_run(self) -> bool {
        matches!(self, Self::NotRunNoPreemptRtPlatform)
    }
}

impl fmt::Display for RtPlatformStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_code())
    }
}

/// Configuración del sondeo experimental de baja latencia en Linux.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxLatencyProbeConfig {
    /// Perfiles de carga a evaluar.
    pub profiles: Vec<LoadProfile>,
    /// Número de ticks medidos por cada perfil de carga.
    pub ticks_per_profile: usize,
    /// Periodo objetivo por tick.
    pub target_period: Duration,
    /// Plazo máximo por tick antes de contabilizar incumplimiento (`deadline_misses`) y fallback.
    pub deadline: Duration,
    /// Iteraciones de trabajo determinista en [`LoadProfile::NominalControl`].
    pub nominal_work_iterations: u32,
    /// Hilos de contención concurrente en [`LoadProfile::HeavyContention`].
    pub contention_threads: usize,
    /// Iteraciones de trabajo dentro del paso crítico en [`LoadProfile::HeavyContention`].
    pub contention_work_iterations: u32,
}

impl Default for LinuxLatencyProbeConfig {
    fn default() -> Self {
        Self {
            profiles: LoadProfile::ALL.to_vec(),
            ticks_per_profile: DEFAULT_TICKS_PER_PROFILE,
            target_period: DEFAULT_TARGET_PERIOD,
            deadline: DEFAULT_DEADLINE,
            nominal_work_iterations: DEFAULT_NOMINAL_WORK_ITERS,
            contention_threads: DEFAULT_CONTENTION_THREADS,
            contention_work_iterations: DEFAULT_CONTENTION_WORK_ITERS,
        }
    }
}

impl LinuxLatencyProbeConfig {
    /// Valida los parámetros de configuración del sondeo.
    pub fn validate(&self) -> Result<(), LinuxLatencyProbeError> {
        if self.profiles.is_empty() {
            return Err(LinuxLatencyProbeError::InvalidConfig(
                "profiles must not be empty",
            ));
        }
        if self.ticks_per_profile == 0 {
            return Err(LinuxLatencyProbeError::InvalidConfig(
                "ticks_per_profile must be greater than zero",
            ));
        }
        if self.target_period.is_zero() {
            return Err(LinuxLatencyProbeError::InvalidConfig(
                "target_period must be greater than zero",
            ));
        }
        if self.deadline.is_zero() {
            return Err(LinuxLatencyProbeError::InvalidConfig(
                "deadline must be greater than zero",
            ));
        }
        Ok(())
    }
}

/// Resultado de telemetría para un perfil de carga individual (AC #2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadProfileProbeResult {
    /// Perfil de carga medido.
    pub load_profile: LoadProfile,
    /// Perfil temporal efectivo aplicado (`RuntimeTemporalProfile::DeadlineAware`).
    pub temporal_profile: RuntimeTemporalProfile,
    /// Reporte completo generado por [`TemporalTelemetryRecorder`].
    pub compliance: TemporalComplianceReport,
}

/// Reporte reproducible de evaluación experimental de baja latencia en Linux (AC #1, #2, #3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinuxLatencyProbeReport {
    /// Metadatos del host Linux (kernel, CPU, permisos).
    pub host: LinuxHostProfile,
    /// Perfil temporal efectivo mantenido durante toda la evaluación (`DeadlineAware`).
    pub effective_temporal_profile: RuntimeTemporalProfile,
    /// Estado de la comparación contra plataforma `PREEMPT_RT` (`NOT_RUN` si no hay `PREEMPT_RT`).
    pub rt_platform_status: RtPlatformStatus,
    /// Nota descriptiva sobre el estado de comparación RT y ausencia de promoción de certificación.
    pub rt_comparison_note: String,
    /// Indica explícitamente que no se promueve certificación en tiempo real (`false`).
    pub certification_promoted: bool,
    /// Aviso explícito de no certificación en tiempo real estricto.
    pub non_realtime_notice: &'static str,
    /// Aviso explícito de no afirmación de WCET.
    pub non_wcet_notice: &'static str,
    /// Resultados por cada perfil de carga evaluado.
    pub load_results: Vec<LoadProfileProbeResult>,
}

/// Ejecuta el sondeo experimental en el host Linux actual usando [`SystemMonotonicClock`]
/// y [`TemporalTelemetryRecorder`].
pub fn run_linux_latency_probe(
    config: &LinuxLatencyProbeConfig,
) -> Result<LinuxLatencyProbeReport, LinuxLatencyProbeError> {
    let snapshot = LinuxHostSnapshot::from_live_host();
    run_linux_latency_probe_with_snapshot(&snapshot, config)
}

/// Ejecuta el sondeo experimental usando una [`LinuxHostSnapshot`] explícita para inspección
/// reproducible del host.
pub fn run_linux_latency_probe_with_snapshot(
    snapshot: &LinuxHostSnapshot,
    config: &LinuxLatencyProbeConfig,
) -> Result<LinuxLatencyProbeReport, LinuxLatencyProbeError> {
    config.validate()?;

    let host = LinuxHostProfile::from_snapshot(snapshot);
    let effective_temporal_profile = RuntimeTemporalProfile::DeadlineAware;
    let rt_platform_status = if host.kernel.preempt_rt_detected {
        RtPlatformStatus::PreemptRtAvailable
    } else {
        RtPlatformStatus::NotRunNoPreemptRtPlatform
    };
    let rt_comparison_note = match rt_platform_status {
        RtPlatformStatus::NotRunNoPreemptRtPlatform => format!(
            "{}: kernel '{}' sin soporte PREEMPT_RT; se conserva perfil {:?} y no se promueve certificación RT. {}",
            rt_platform_status.as_code(),
            host.kernel.kernel_release,
            effective_temporal_profile,
            LINUX_RT_NON_CERTIFICATION_NOTE,
        ),
        RtPlatformStatus::PreemptRtAvailable => format!(
            "{}: kernel '{}' con indicadores PREEMPT_RT; medición experimental conserva perfil {:?} y no promueve certificación RT. {}",
            rt_platform_status.as_code(),
            host.kernel.kernel_release,
            effective_temporal_profile,
            LINUX_RT_NON_CERTIFICATION_NOTE,
        ),
    };

    let mut load_results = Vec::with_capacity(config.profiles.len());
    for &profile in &config.profiles {
        let compliance = measure_load_profile(profile, config)?;
        load_results.push(LoadProfileProbeResult {
            load_profile: profile,
            temporal_profile: effective_temporal_profile,
            compliance,
        });
    }

    Ok(LinuxLatencyProbeReport {
        host,
        effective_temporal_profile,
        rt_platform_status,
        rt_comparison_note,
        certification_promoted: false,
        non_realtime_notice: STRICT_REALTIME_NON_CERTIFICATION_NOTICE,
        non_wcet_notice: NO_WCET_CLAIM_NOTICE,
        load_results,
    })
}

fn measure_load_profile(
    profile: LoadProfile,
    config: &LinuxLatencyProbeConfig,
) -> Result<TemporalComplianceReport, LinuxLatencyProbeError> {
    let window_cap = NonZeroUsize::new(config.ticks_per_profile.max(1))
        .ok_or(LinuxLatencyProbeError::InvalidConfig("invalid window"))?;
    let telemetry_cfg =
        TemporalTelemetryConfig::new(profile.as_label(), config.target_period, config.deadline)?
            .with_window_capacity(window_cap);
    let mut recorder = TemporalTelemetryRecorder::new(SystemMonotonicClock::new(), telemetry_cfg);

    let stop_flag = Arc::new(AtomicBool::new(false));
    let mut workers = Vec::new();
    if profile == LoadProfile::HeavyContention {
        for worker_idx in 0..config.contention_threads {
            let stop = Arc::clone(&stop_flag);
            let handle = thread::spawn(move || {
                let mut acc: u64 = (worker_idx as u64).wrapping_add(1);
                while !stop.load(Ordering::Relaxed) {
                    acc = deterministic_spin_mix(acc, 256);
                    black_box(acc);
                    thread::yield_now();
                }
            });
            workers.push(handle);
        }
    }

    let deadline_nanos = u64::try_from(config.deadline.as_nanos()).unwrap_or(u64::MAX);
    let target_period_nanos = u64::try_from(config.target_period.as_nanos()).unwrap_or(u64::MAX);

    for idx in 0..config.ticks_per_profile {
        let tick = Tick::new((idx + 1) as u64);
        let started_at = recorder.start_tick();

        match profile {
            LoadProfile::Idle => {
                black_box(deterministic_spin_mix(tick.as_u64(), 8));
            }
            LoadProfile::NominalControl => {
                black_box(deterministic_spin_mix(
                    tick.as_u64(),
                    config.nominal_work_iterations,
                ));
            }
            LoadProfile::HeavyContention => {
                black_box(deterministic_spin_mix(
                    tick.as_u64(),
                    config.contention_work_iterations,
                ));
            }
        }

        let finished_at = recorder.clock().now();
        let latency = finished_at.saturating_duration_since(started_at);
        let latency_nanos = u64::try_from(latency.as_nanos()).unwrap_or(u64::MAX);
        let fallback_reason = if latency_nanos > deadline_nanos {
            Some(MultirateFallbackReason::DeadlineExpired)
        } else {
            None
        };
        let jitter_nanos = (latency_nanos as i128) - (target_period_nanos as i128);
        let _ = recorder.record_sample(tick, finished_at, latency, jitter_nanos, fallback_reason);
    }

    stop_flag.store(true, Ordering::Relaxed);
    for worker in workers {
        let _ = worker.join();
    }

    Ok(recorder.report())
}

fn deterministic_spin_mix(seed: u64, iterations: u32) -> u64 {
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    for step in 0..iterations {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(u64::from(step) | 1);
    }
    state
}

fn parse_kernel_info(snapshot: &LinuxHostSnapshot) -> LinuxKernelInfo {
    let kernel_release = snapshot
        .osrelease
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string();
    let kernel_version = snapshot
        .version
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("unknown")
        .to_string();
    let sys_kernel_realtime = snapshot.sys_kernel_realtime.as_deref().and_then(|raw| {
        let trimmed = raw.trim();
        match trimmed {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        }
    });

    let has_preempt_rt_marker = sys_kernel_realtime == Some(true)
        || kernel_version.contains("PREEMPT_RT")
        || kernel_release.contains("PREEMPT_RT")
        || release_has_rt_suffix(&kernel_release);

    let preempt_mode = if has_preempt_rt_marker {
        KernelPreemptMode::PreemptRt
    } else if kernel_version.contains("PREEMPT_DYNAMIC") {
        KernelPreemptMode::PreemptDynamic
    } else if kernel_version.contains("PREEMPT") {
        KernelPreemptMode::PreemptStandard
    } else {
        KernelPreemptMode::Unknown
    };

    LinuxKernelInfo {
        kernel_release,
        kernel_version,
        preempt_rt_detected: has_preempt_rt_marker,
        preempt_mode,
        sys_kernel_realtime,
    }
}

fn release_has_rt_suffix(release: &str) -> bool {
    release.split(['.', '-', '_']).any(|part| {
        part == "rt"
            || part
                .strip_prefix("rt")
                .is_some_and(|rest| !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()))
    })
}

fn parse_cpu_info(snapshot: &LinuxHostSnapshot) -> LinuxCpuInfo {
    let online_cpus_raw = snapshot
        .online_cpus
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string);

    let cpus_allowed_list = snapshot
        .self_status
        .as_deref()
        .and_then(|status| extract_proc_status_field(status, "Cpus_allowed_list"));

    let mut cpuinfo_processors = 0usize;
    let mut model_name = None;
    if let Some(cpuinfo) = snapshot.cpuinfo.as_deref() {
        for line in cpuinfo.lines() {
            if let Some((key, val)) = line.split_once(':') {
                let key = key.trim();
                let val = val.trim();
                if key == "processor" {
                    cpuinfo_processors = cpuinfo_processors.saturating_add(1);
                } else if model_name.is_none()
                    && matches!(key, "model name" | "Model" | "Hardware" | "cpu model")
                    && !val.is_empty()
                {
                    model_name = Some(val.to_string());
                }
            }
        }
    }

    let cpu_count = online_cpus_raw
        .as_deref()
        .and_then(parse_cpu_list_count)
        .filter(|&count| count > 0)
        .or(if cpuinfo_processors > 0 {
            Some(cpuinfo_processors)
        } else {
            None
        })
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(NonZeroUsize::get)
                .unwrap_or(1)
        });

    LinuxCpuInfo {
        arch: std::env::consts::ARCH,
        cpu_count,
        online_cpus_raw,
        cpus_allowed_list,
        model_name: model_name.unwrap_or_else(|| "unknown".to_string()),
    }
}

fn parse_cpu_list_count(list: &str) -> Option<usize> {
    let trimmed = list.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut total = 0usize;
    for part in trimmed.split(',') {
        let part = part.trim();
        if part.is_empty() {
            return None;
        }
        if let Some((start_str, end_str)) = part.split_once('-') {
            let start = start_str.trim().parse::<usize>().ok()?;
            let end = end_str.trim().parse::<usize>().ok()?;
            if end < start {
                return None;
            }
            total = total.checked_add(end - start + 1)?;
        } else {
            let _ = part.parse::<usize>().ok()?;
            total = total.checked_add(1)?;
        }
    }
    Some(total)
}

fn parse_rt_permissions(snapshot: &LinuxHostSnapshot) -> LinuxRtPermissions {
    let cap_eff_hex = snapshot
        .self_status
        .as_deref()
        .and_then(|status| extract_proc_status_field(status, "CapEff"));

    let cap_eff_mask = cap_eff_hex
        .as_deref()
        .and_then(|hex| u64::from_str_radix(hex.trim(), 16).ok());

    const CAP_IPC_LOCK_BIT: u32 = 14;
    const CAP_SYS_NICE_BIT: u32 = 23;

    let cap_ipc_lock = cap_eff_mask.map(|mask| (mask & (1u64 << CAP_IPC_LOCK_BIT)) != 0);
    let cap_sys_nice = cap_eff_mask.map(|mask| (mask & (1u64 << CAP_SYS_NICE_BIT)) != 0);

    let (rtprio_soft_limit, rtprio_hard_limit) = snapshot
        .self_limits
        .as_deref()
        .and_then(|limits| parse_proc_limit_row(limits, "Max realtime priority"))
        .map_or((None, None), |(soft, hard)| (Some(soft), Some(hard)));

    let (memlock_soft_bytes, memlock_hard_bytes) = snapshot
        .self_limits
        .as_deref()
        .and_then(|limits| parse_proc_limit_row(limits, "Max locked memory"))
        .map_or((None, None), |(soft, hard)| (Some(soft), Some(hard)));

    let rt_scheduling_status = match (cap_sys_nice, rtprio_soft_limit) {
        (Some(true), _) | (_, Some(RtLimit::Unlimited)) | (_, Some(RtLimit::Value(1..))) => {
            RtSchedulingStatus::Permitted
        }
        (Some(false), Some(RtLimit::Value(0)))
        | (None, Some(RtLimit::Value(0)))
        | (Some(false), None) => RtSchedulingStatus::UnprivilegedLimited,
        (None, None) => RtSchedulingStatus::Unknown,
    };

    LinuxRtPermissions {
        rt_scheduling_status,
        rtprio_soft_limit,
        rtprio_hard_limit,
        memlock_soft_bytes,
        memlock_hard_bytes,
        cap_eff_hex,
        cap_sys_nice,
        cap_ipc_lock,
    }
}

fn extract_proc_status_field(status: &str, field_name: &str) -> Option<String> {
    for line in status.lines() {
        if let Some((key, val)) = line.split_once(':')
            && key.trim() == field_name
        {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn parse_proc_limit_row(limits: &str, label: &str) -> Option<(RtLimit, RtLimit)> {
    for line in limits.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix(label) {
            let mut tokens = rest.split_whitespace();
            let soft = parse_limit_token(tokens.next()?)?;
            let hard = parse_limit_token(tokens.next()?)?;
            return Some((soft, hard));
        }
    }
    None
}

fn parse_limit_token(token: &str) -> Option<RtLimit> {
    if token.eq_ignore_ascii_case("unlimited") {
        Some(RtLimit::Unlimited)
    } else {
        token.parse::<u64>().ok().map(RtLimit::Value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_non_rt_snapshot() -> LinuxHostSnapshot {
        LinuxHostSnapshot {
            osrelease: Some("7.3.4-ogc8.1.fc45.x86_64\n".to_string()),
            version: Some(
                "Linux version 7.3.4-ogc8.1.fc45.x86_64 (gcc) #1 SMP PREEMPT_DYNAMIC\n"
                    .to_string(),
            ),
            sys_kernel_realtime: None,
            cpuinfo: Some(
                "processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GHz\n\nprocessor\t: 1\nmodel name\t: Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GHz\n"
                    .to_string(),
            ),
            online_cpus: Some("0-63\n".to_string()),
            self_status: Some(
                "Name:\tgogamen\nCapEff:\t0000000000000000\nCpus_allowed_list:\t0-63\n"
                    .to_string(),
            ),
            self_limits: Some(
                "Limit                     Soft Limit           Hard Limit           Units\nMax locked memory         2199023616           2199023616           bytes\nMax realtime priority     0                    0                    us\n"
                    .to_string(),
            ),
        }
    }

    fn sample_preempt_rt_snapshot() -> LinuxHostSnapshot {
        LinuxHostSnapshot {
            osrelease: Some("6.12.5-rt14-x86_64\n".to_string()),
            version: Some(
                "Linux version 6.12.5-rt14-x86_64 (gcc) #1 SMP PREEMPT_RT\n".to_string(),
            ),
            sys_kernel_realtime: Some("1\n".to_string()),
            cpuinfo: Some("processor\t: 0\nmodel name\t: Test RT CPU\n".to_string()),
            online_cpus: Some("0-3\n".to_string()),
            self_status: Some("CapEff:\t0000000000804000\nCpus_allowed_list:\t0-3\n".to_string()),
            self_limits: Some(
                "Max locked memory         unlimited            unlimited            bytes\nMax realtime priority     99                   99                   us\n"
                    .to_string(),
            ),
        }
    }

    #[test]
    fn ac1_host_inspection_reproducibly_identifies_kernel_cpu_permissions_and_load() {
        let snapshot = sample_non_rt_snapshot();
        let profile_a = LinuxHostProfile::from_snapshot(&snapshot);
        let profile_b = LinuxHostProfile::from_snapshot(&snapshot);

        assert_eq!(profile_a, profile_b);
        assert_eq!(profile_a.kernel.kernel_release, "7.3.4-ogc8.1.fc45.x86_64");
        assert!(!profile_a.kernel.preempt_rt_detected);
        assert_eq!(
            profile_a.kernel.preempt_mode,
            KernelPreemptMode::PreemptDynamic
        );

        assert_eq!(profile_a.cpu.arch, std::env::consts::ARCH);
        assert_eq!(profile_a.cpu.cpu_count, 64);
        assert_eq!(profile_a.cpu.online_cpus_raw.as_deref(), Some("0-63"));
        assert_eq!(profile_a.cpu.cpus_allowed_list.as_deref(), Some("0-63"));
        assert_eq!(
            profile_a.cpu.model_name,
            "Intel(R) Xeon(R) CPU E5-2682 v4 @ 2.50GHz"
        );

        assert_eq!(
            profile_a.permissions.rt_scheduling_status,
            RtSchedulingStatus::UnprivilegedLimited
        );
        assert_eq!(
            profile_a.permissions.rtprio_soft_limit,
            Some(RtLimit::Value(0))
        );
        assert_eq!(
            profile_a.permissions.memlock_soft_bytes,
            Some(RtLimit::Value(2_199_023_616))
        );
        assert_eq!(profile_a.permissions.cap_sys_nice, Some(false));
        assert_eq!(profile_a.permissions.cap_ipc_lock, Some(false));

        let live = LinuxHostProfile::inspect_live();
        assert!(!live.kernel.kernel_release.is_empty());
        assert!(live.cpu.cpu_count >= 1);
    }

    #[test]
    fn ac2_probe_measures_misses_fallbacks_and_percentiles_across_multiple_load_profiles() {
        let config = LinuxLatencyProbeConfig {
            profiles: vec![
                LoadProfile::Idle,
                LoadProfile::NominalControl,
                LoadProfile::HeavyContention,
            ],
            ticks_per_profile: 20,
            target_period: Duration::from_micros(50),
            deadline: Duration::from_nanos(1),
            nominal_work_iterations: 200,
            contention_threads: 2,
            contention_work_iterations: 5_000,
        };

        let report = run_linux_latency_probe(&config).expect("probe should succeed");

        assert_eq!(report.load_results.len(), 3);
        assert_eq!(
            report.effective_temporal_profile,
            RuntimeTemporalProfile::DeadlineAware
        );
        assert!(!report.certification_promoted);
        assert_eq!(
            report.non_realtime_notice,
            STRICT_REALTIME_NON_CERTIFICATION_NOTICE
        );
        assert_eq!(report.non_wcet_notice, NO_WCET_CLAIM_NOTICE);

        for (expected_profile, result) in LoadProfile::ALL.iter().zip(report.load_results.iter()) {
            assert_eq!(result.load_profile, *expected_profile);
            assert_eq!(
                result.temporal_profile,
                RuntimeTemporalProfile::DeadlineAware
            );
            assert_eq!(result.compliance.total_samples, 20);
            assert_eq!(result.compliance.load_label, expected_profile.as_label());
            assert!(result.compliance.p50_nanos <= result.compliance.p95_nanos);
            assert!(result.compliance.p95_nanos <= result.compliance.p99_nanos);
            assert!(result.compliance.p99_nanos <= result.compliance.max_observed_nanos);
            assert_eq!(
                result.compliance.deadline_misses,
                result
                    .compliance
                    .fallback_counts
                    .get(MultirateFallbackReason::DeadlineExpired)
            );
            assert!(result.compliance.deadline_misses > 0);
            assert!(
                result.compliance.jitter.max_abs_nanos >= result.compliance.jitter.p95_abs_nanos
            );
        }
    }

    #[test]
    fn ac3_non_preempt_rt_host_marks_not_run_and_preserves_deadline_aware_profile() {
        let config = LinuxLatencyProbeConfig {
            profiles: vec![LoadProfile::Idle, LoadProfile::NominalControl],
            ticks_per_profile: 10,
            target_period: Duration::from_millis(1),
            deadline: Duration::from_millis(1),
            nominal_work_iterations: 100,
            contention_threads: 1,
            contention_work_iterations: 500,
        };

        let non_rt_report =
            run_linux_latency_probe_with_snapshot(&sample_non_rt_snapshot(), &config)
                .expect("non-RT probe should succeed");
        assert!(!non_rt_report.host.kernel.preempt_rt_detected);
        assert_eq!(
            non_rt_report.rt_platform_status,
            RtPlatformStatus::NotRunNoPreemptRtPlatform
        );
        assert_eq!(non_rt_report.rt_platform_status.as_code(), "NOT_RUN");
        assert!(non_rt_report.rt_platform_status.is_not_run());
        assert_eq!(
            non_rt_report.effective_temporal_profile,
            RuntimeTemporalProfile::DeadlineAware
        );
        assert!(!non_rt_report.certification_promoted);
        assert!(non_rt_report.rt_comparison_note.contains("NOT_RUN"));
        assert!(non_rt_report.rt_comparison_note.contains("DeadlineAware"));

        let rt_report =
            run_linux_latency_probe_with_snapshot(&sample_preempt_rt_snapshot(), &config)
                .expect("synthetic RT probe should succeed");
        assert!(rt_report.host.kernel.preempt_rt_detected);
        assert_eq!(
            rt_report.rt_platform_status,
            RtPlatformStatus::PreemptRtAvailable
        );
        assert_eq!(
            rt_report.host.permissions.rt_scheduling_status,
            RtSchedulingStatus::Permitted
        );
        assert_eq!(
            rt_report.effective_temporal_profile,
            RuntimeTemporalProfile::DeadlineAware
        );
        assert!(!rt_report.certification_promoted);
    }
}
