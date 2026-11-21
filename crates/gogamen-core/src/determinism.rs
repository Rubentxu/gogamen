//! Semillas deterministas, streams RNG aislados, reloj lógico y hashing de trazas (`std`-only).
//!
//! Este módulo provee los primitivos de reproducibilidad de `gogamen-core`:
//! - [`MasterSeed`] (alias [`Seed`]), [`StreamKey`] y [`RngStreamRegistry`] para derivar
//!   streams pseudoaleatorios independientes por jerarquía `(episode, agent, walker)`,
//!   de modo que el orden de ejecución o instanciación nunca altere un stream individual.
//! - [`LogicalClock`] (alias [`DeterministicClock`]) con paso fijo [`StepDt`] y tiempo
//!   acumulado calculado como `dt * tick` para evitar deriva acumulada de punto flotante.
//! - [`DeterminismMetadata`] y [`TraceHasher`] para registrar `(seed, build_version, platform)`
//!   y calcular hashes deterministas de trazas en el mismo build y plataforma, sin prometer
//!   determinismo transversal entre plataformas distintas.

use std::collections::BTreeMap;
use std::fmt;

use crate::domain::{Observation, State, Terminal, TerminalReason, Transition};
use crate::error::DomainError;
use crate::model::DeterministicRng;
use crate::units::{SimDuration, StepDt, Tick};

const FNV1A_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

const TAG_EPISODE: u64 = 0x4550_4953_4f44_4501;
const TAG_AGENT: u64 = 0x4147_454e_5400_0002;
const TAG_WALKER: u64 = 0x5741_4c4b_4552_0003;

const fn splitmix64_mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

const fn fold_u64(acc: u64, value: u64) -> u64 {
    splitmix64_mix(acc ^ splitmix64_mix(value))
}

const fn derive_raw_seed(base: u64, key: StreamKey) -> u64 {
    let h = splitmix64_mix(base);
    match key {
        StreamKey::Episode(episode) => {
            let h = fold_u64(h, TAG_EPISODE);
            fold_u64(h, episode)
        }
        StreamKey::Agent { episode, agent } => {
            let h = fold_u64(h, TAG_AGENT);
            let h = fold_u64(h, episode);
            fold_u64(h, agent)
        }
        StreamKey::Walker {
            episode,
            agent,
            walker,
        } => {
            let h = fold_u64(h, TAG_WALKER);
            let h = fold_u64(h, episode);
            let h = fold_u64(h, agent);
            fold_u64(h, walker)
        }
    }
}

/// Semilla maestra inmutable para derivar streams pseudoaleatorios deterministas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct MasterSeed(u64);

/// Alias conveniente para [`MasterSeed`].
pub type Seed = MasterSeed;

impl MasterSeed {
    /// Crea una nueva [`MasterSeed`] a partir de un entero de 64 bits.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// Devuelve el valor numérico subyacente de la semilla.
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Deriva una semilla de 64 bits determinista e independiente para `key`.
    pub const fn derive_seed(self, key: StreamKey) -> u64 {
        derive_raw_seed(self.0, key)
    }

    /// Deriva un [`DeterministicRng`] aislado para `key` sin afectar a ningún otro stream.
    pub const fn derive(self, key: StreamKey) -> DeterministicRng {
        DeterministicRng::new(self.derive_seed(key))
    }

    /// Crea un [`RngStreamRegistry`] respaldado por esta semilla maestra.
    pub const fn registry(self) -> RngStreamRegistry {
        RngStreamRegistry::new(self)
    }
}

impl From<u64> for MasterSeed {
    fn from(value: u64) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for MasterSeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seed(0x{:016x})", self.0)
    }
}

/// Clave jerárquica que identifica de forma única un stream RNG independiente.
///
/// Cada variante incorpora separación de dominio explícita para evitar colisiones
/// entre niveles de la jerarquía `(episode, agent, walker)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StreamKey {
    /// Stream asociado a un episodio completo.
    Episode(u64),
    /// Stream asociado a un agente concreto dentro de un episodio.
    Agent {
        /// Identificador del episodio.
        episode: u64,
        /// Identificador del agente.
        agent: u64,
    },
    /// Stream asociado a un walker o rollout concreto de un agente en un episodio.
    Walker {
        /// Identificador del episodio.
        episode: u64,
        /// Identificador del agente.
        agent: u64,
        /// Identificador del walker.
        walker: u64,
    },
}

impl StreamKey {
    /// Construye una clave de nivel episodio.
    pub const fn episode(episode: u64) -> Self {
        Self::Episode(episode)
    }

    /// Construye una clave de nivel agente dentro de un episodio.
    pub const fn agent(episode: u64, agent: u64) -> Self {
        Self::Agent { episode, agent }
    }

    /// Construye una clave de nivel walker dentro de un agente y episodio.
    pub const fn walker(episode: u64, agent: u64, walker: u64) -> Self {
        Self::Walker {
            episode,
            agent,
            walker,
        }
    }
}

impl fmt::Display for StreamKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Episode(episode) => write!(f, "episode({episode})"),
            Self::Agent { episode, agent } => write!(f, "agent({episode}, {agent})"),
            Self::Walker {
                episode,
                agent,
                walker,
            } => write!(f, "walker({episode}, {agent}, {walker})"),
        }
    }
}

impl DeterministicRng {
    /// Deriva un nuevo [`DeterministicRng`] hijo para `key` sin mutar `self`.
    pub const fn derive(&self, key: StreamKey) -> Self {
        Self::new(derive_raw_seed(self.state(), key))
    }
}

/// Registro determinista de streams RNG indexados por [`StreamKey`].
///
/// Al derivar cada stream directamente desde [`MasterSeed`] y la clave jerárquica,
/// el orden en que se crean, consultan o consumen distintos streams nunca altera la
/// secuencia de ningún stream individual.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RngStreamRegistry {
    master_seed: MasterSeed,
    streams: BTreeMap<StreamKey, DeterministicRng>,
}

impl RngStreamRegistry {
    /// Crea un registro vacío asociado a `master_seed`.
    pub const fn new(master_seed: MasterSeed) -> Self {
        Self {
            master_seed,
            streams: BTreeMap::new(),
        }
    }

    /// Devuelve la [`MasterSeed`] asociada al registro.
    pub const fn master_seed(&self) -> MasterSeed {
        self.master_seed
    }

    /// Devuelve una referencia mutable al stream persistente de `key`,
    /// derivándolo deterministamente si aún no existía.
    pub fn stream(&mut self, key: StreamKey) -> &mut DeterministicRng {
        let seed = self.master_seed;
        self.streams.entry(key).or_insert_with(|| seed.derive(key))
    }

    /// Deriva un [`DeterministicRng`] efímero para `key` sin almacenarlo ni alterar
    /// el estado de los streams ya registrados.
    pub const fn derive_ephemeral(&self, key: StreamKey) -> DeterministicRng {
        self.master_seed.derive(key)
    }

    /// Reinicia el stream asociado a `key` a su estado inicial derivado (si existía).
    pub fn reset_stream(&mut self, key: StreamKey) {
        self.streams.remove(&key);
    }

    /// Devuelve el número de streams instanciados en el registro.
    pub fn len(&self) -> usize {
        self.streams.len()
    }

    /// Devuelve `true` si no hay streams instanciados todavía.
    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// Reloj lógico determinista con paso fijo [`StepDt`].
///
/// Calcula el tiempo acumulado directamente como `dt * tick` en lugar de sumar
/// repetidamente `elapsed + dt`, evitando así la deriva acumulada de punto flotante.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalClock {
    dt: StepDt,
    tick: Tick,
    elapsed: SimDuration,
}

/// Alias conveniente para [`LogicalClock`].
pub type DeterministicClock = LogicalClock;

impl LogicalClock {
    /// Crea un reloj lógico en `Tick::ZERO` y `SimDuration::ZERO` con paso fijo `dt`.
    pub const fn new(dt: StepDt) -> Self {
        Self {
            dt,
            tick: Tick::ZERO,
            elapsed: SimDuration::ZERO,
        }
    }

    /// Construye un reloj lógico posicionado en un `tick` específico.
    pub fn with_tick(dt: StepDt, tick: Tick) -> Result<Self, DomainError> {
        let elapsed = compute_elapsed(dt, tick, "LogicalClock::with_tick")?;
        Ok(Self { dt, tick, elapsed })
    }

    /// Devuelve el paso temporal fijo [`StepDt`] del reloj.
    pub const fn dt(&self) -> StepDt {
        self.dt
    }

    /// Devuelve el [`Tick`] actual del reloj.
    pub const fn tick(&self) -> Tick {
        self.tick
    }

    /// Devuelve la [`SimDuration`] acumulada exacta (`dt * tick`).
    pub const fn elapsed(&self) -> SimDuration {
        self.elapsed
    }

    /// Avanza el reloj exactamente un paso [`StepDt`], devolviendo `(tick, elapsed)`.
    pub fn advance(&mut self) -> Result<(Tick, SimDuration), DomainError> {
        let next_tick = self.tick.next()?;
        let next_elapsed = compute_elapsed(self.dt, next_tick, "LogicalClock::advance")?;
        self.tick = next_tick;
        self.elapsed = next_elapsed;
        Ok((next_tick, next_elapsed))
    }

    /// Avanza el reloj `steps` pasos de forma directa y verificada.
    pub fn advance_by(&mut self, steps: u64) -> Result<(Tick, SimDuration), DomainError> {
        let next_tick = self.tick.checked_add(steps)?;
        let next_elapsed = compute_elapsed(self.dt, next_tick, "LogicalClock::advance_by")?;
        self.tick = next_tick;
        self.elapsed = next_elapsed;
        Ok((next_tick, next_elapsed))
    }

    /// Reinicia el reloj a `Tick::ZERO` y `SimDuration::ZERO` conservando `dt`.
    pub fn reset(&mut self) {
        self.tick = Tick::ZERO;
        self.elapsed = SimDuration::ZERO;
    }
}

fn compute_elapsed(
    dt: StepDt,
    tick: Tick,
    context: &'static str,
) -> Result<SimDuration, DomainError> {
    let secs = dt.as_secs_f64() * (tick.as_u64() as f64);
    if !secs.is_finite() {
        return Err(DomainError::TimeOverflow { context });
    }
    SimDuration::from_secs_f64(secs).map_err(|_| DomainError::TimeOverflow { context })
}

/// Metadatos explícitos de reproducibilidad asociados a una traza determinista.
///
/// Registra `seed`, `build_version` y `platform` (`OS`/`ARCH`). El determinismo
/// garantizado se restringe al **mismo build y misma plataforma**; no promete
/// determinismo transversal bit-a-bit entre plataformas o arquitecturas distintas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeterminismMetadata {
    seed: MasterSeed,
    build_version: &'static str,
    platform_os: &'static str,
    platform_arch: &'static str,
}

impl DeterminismMetadata {
    /// Captura los metadatos de determinismo del build y plataforma actuales.
    pub const fn current(seed: MasterSeed) -> Self {
        Self {
            seed,
            build_version: crate::core_version(),
            platform_os: std::env::consts::OS,
            platform_arch: std::env::consts::ARCH,
        }
    }

    /// Construye metadatos explícitos con versión de build y plataforma dadas.
    pub const fn new(
        seed: MasterSeed,
        build_version: &'static str,
        platform_os: &'static str,
        platform_arch: &'static str,
    ) -> Self {
        Self {
            seed,
            build_version,
            platform_os,
            platform_arch,
        }
    }

    /// Devuelve la [`MasterSeed`] registrada.
    pub const fn seed(&self) -> MasterSeed {
        self.seed
    }

    /// Devuelve la versión de build registrada.
    pub const fn build_version(&self) -> &'static str {
        self.build_version
    }

    /// Devuelve el sistema operativo registrado (`std::env::consts::OS`).
    pub const fn platform_os(&self) -> &'static str {
        self.platform_os
    }

    /// Devuelve la arquitectura registrada (`std::env::consts::ARCH`).
    pub const fn platform_arch(&self) -> &'static str {
        self.platform_arch
    }

    /// Devuelve el identificador compuesto de plataforma (`"<os>-<arch>"`).
    pub fn platform(&self) -> String {
        format!("{}-{}", self.platform_os, self.platform_arch)
    }

    /// Indica explícitamente que no se promete determinismo transversal entre plataformas.
    pub const fn cross_platform_guaranteed(&self) -> bool {
        false
    }
}

impl fmt::Display for DeterminismMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "seed={}, build={}, platform={}-{}, cross_platform={}",
            self.seed,
            self.build_version,
            self.platform_os,
            self.platform_arch,
            self.cross_platform_guaranteed()
        )
    }
}

/// Hasher incremental determinista (FNV-1a de 64 bits, `std`-only) para trazas de simulación.
///
/// Produce un hash estable dentro del mismo build y plataforma para secuencias idénticas
/// de metadatos, estados, observaciones y transiciones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceHasher {
    hash: u64,
}

impl TraceHasher {
    /// Crea un nuevo [`TraceHasher`] con estado inicial FNV-1a.
    pub const fn new() -> Self {
        Self {
            hash: FNV1A_OFFSET_BASIS,
        }
    }

    /// Crea un [`TraceHasher`] inicializado con [`DeterminismMetadata`].
    pub fn with_metadata(metadata: &DeterminismMetadata) -> Self {
        let mut hasher = Self::new();
        hasher.write_metadata(metadata);
        hasher
    }

    /// Incorpora un byte al hash.
    pub fn write_u8(&mut self, byte: u8) {
        self.hash ^= u64::from(byte);
        self.hash = self.hash.wrapping_mul(FNV1A_PRIME);
    }

    /// Incorpora un entero `u64` en formato little-endian canónico.
    pub fn write_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.write_u8(byte);
        }
    }

    /// Incorpora un slice de bytes precedido por su longitud.
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        self.write_u64(bytes.len() as u64);
        for &byte in bytes {
            self.write_u8(byte);
        }
    }

    /// Incorpora una cadena UTF-8 precedida por su longitud.
    pub fn write_str(&mut self, text: &str) {
        self.write_bytes(text.as_bytes());
    }

    /// Incorpora un escalar `f64` normalizando `-0.0` a `+0.0`.
    pub fn write_f64(&mut self, value: f64) {
        let normalized = if value == 0.0 { 0.0 } else { value };
        self.write_u64(normalized.to_bits());
    }

    /// Incorpora [`DeterminismMetadata`] (semilla, build y plataforma).
    pub fn write_metadata(&mut self, metadata: &DeterminismMetadata) {
        self.write_u64(metadata.seed().as_u64());
        self.write_str(metadata.build_version());
        self.write_str(metadata.platform_os());
        self.write_str(metadata.platform_arch());
    }

    /// Incorpora un [`Tick`].
    pub fn write_tick(&mut self, tick: Tick) {
        self.write_u64(tick.as_u64());
    }

    /// Incorpora una [`SimDuration`].
    pub fn write_duration(&mut self, duration: SimDuration) {
        self.write_f64(duration.as_secs_f64());
    }

    /// Incorpora un [`StepDt`].
    pub fn write_step_dt(&mut self, dt: StepDt) {
        self.write_f64(dt.as_secs_f64());
    }

    /// Incorpora un [`State`] completo (`tick`, `elapsed` y `values`).
    pub fn write_state(&mut self, state: &State) {
        self.write_tick(state.tick());
        self.write_duration(state.elapsed());
        self.write_u64(state.values().len() as u64);
        for &value in state.values() {
            self.write_f64(value);
        }
    }

    /// Incorpora una [`Observation`] completa (`tick`, `elapsed` y `features`).
    pub fn write_observation(&mut self, observation: &Observation) {
        self.write_tick(observation.tick());
        self.write_duration(observation.elapsed());
        self.write_u64(observation.features().len() as u64);
        for &feature in observation.features() {
            self.write_f64(feature);
        }
    }

    /// Incorpora un estado de terminación [`Terminal`].
    pub fn write_terminal(&mut self, terminal: Terminal) {
        match terminal {
            Terminal::Ongoing => self.write_u8(0),
            Terminal::Terminated(reason) => {
                self.write_u8(1);
                let code = match reason {
                    TerminalReason::GoalReached => 1,
                    TerminalReason::Failure => 2,
                    TerminalReason::HorizonReached => 3,
                    TerminalReason::Cancelled => 4,
                    TerminalReason::InvariantViolated => 5,
                };
                self.write_u8(code);
            }
        }
    }

    /// Incorpora una [`Transition`] completa.
    pub fn write_transition(&mut self, transition: &Transition) {
        self.write_tick(transition.prev_tick());
        self.write_step_dt(transition.dt());
        self.write_state(transition.next_state());
        self.write_observation(transition.observation());
        self.write_terminal(transition.terminal());
    }

    /// Devuelve el hash de 64 bits calculado hasta el momento.
    pub const fn finish(&self) -> u64 {
        self.hash
    }

    /// Devuelve el hash de 64 bits formateado en hexadecimal de 16 dígitos.
    pub fn finish_hex(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

impl Default for TraceHasher {
    fn default() -> Self {
        Self::new()
    }
}
