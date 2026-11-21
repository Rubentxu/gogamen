//! Timeline y replay desconectado (GOG-94).
//!
//! Abre una grabación Grid ya escrita (sin motor, sin sensores, sin actuadores, sin red) y la
//! proyecta con el **mismo** [`project_grid_scene`] que el modo en vivo.
//!
//! - [`Recording`]: formato de texto versionado (`GOGAMEN-GRID-REC v1`), cabecera con semilla,
//!   mapa y último tick declarado, más un frame por línea. Se escribe desde el motor real en
//!   [`crate::recorder`] (lado motor); este módulo sólo lee y proyecta.
//! - Huecos: los ticks ausentes dentro de `0..=last_tick` se exponen como [`Gap`]; nunca se
//!   interpolan (en un hueco [`ReplayPlayer::scene`] devuelve `None`).
//! - [`ReplayPlayer`] toma `&Recording` (sólo lectura) y mantiene un **reloj de reproducción**
//!   propio (segundos de pared, [`ReplayPlayer::advance`]) separado del tick virtual grabado.
//!   Cambiar velocidad, pausa o cámara nunca toca la grabación.

use std::fmt;

use gogamen_core::{Terminal, TerminalReason, Tick};
use gogamen_lab_grid::{CellKind, GridMap, GridPos};
use gogamen_projection::Scene2D;

use crate::camera::Camera2D;
use crate::grid_lab::{GridWorldView, project_grid_scene};
use crate::viewport::SoftwareSvgBackend;

/// Versión del formato que este lector entiende.
pub const SUPPORTED_VERSION: u32 = 1;
const MAGIC: &str = "GOGAMEN-GRID-REC";
/// Ticks virtuales por segundo de reproducción a velocidad 1x (`dt` del Grid = 100 ms).
pub const TICKS_PER_SECOND_1X: f64 = 10.0;
/// Rango de velocidad aceptado.
pub const MIN_SPEED: f64 = 0.01;
pub const MAX_SPEED: f64 = 64.0;

/// Frame grabado de un tick.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub tick: u64,
    pub revision: u64,
    pub agent: GridPos,
    pub last_reward: f64,
    pub total_reward: f64,
    pub collisions: u64,
    pub terminal: Terminal,
}

/// Rango inclusivo de ticks sin frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gap {
    pub from: u64,
    pub to: u64,
}

/// Error al abrir una grabación.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpenError {
    IncompatibleVersion {
        found: u32,
        supported: u32,
    },
    /// No es una grabación Grid (cabecera ausente).
    NotARecording,
    Corrupt(String),
}

impl fmt::Display for OpenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleVersion { found, supported } => write!(
                f,
                "incompatible recording version v{found} (supported: v{supported})"
            ),
            Self::NotARecording => write!(f, "not a gogamen grid recording"),
            Self::Corrupt(m) => write!(f, "corrupt recording: {m}"),
        }
    }
}

impl std::error::Error for OpenError {}

fn corrupt<T>(msg: impl Into<String>) -> Result<T, OpenError> {
    Err(OpenError::Corrupt(msg.into()))
}

/// Grabación inmutable una vez construida (sin métodos `&mut self`).
#[derive(Debug, Clone, PartialEq)]
pub struct Recording {
    version: u32,
    seed: u64,
    map_rows: Vec<String>,
    map: GridMap,
    last_tick: u64,
    frames: Vec<Frame>,
    gaps: Vec<Gap>,
}

/// Filas ASCII de un mapa (`#`, `.`, `S`, `G`, `H`), formato de `GridMap::from_ascii`.
pub fn map_to_rows(map: &GridMap) -> Vec<String> {
    (0..map.height())
        .map(|y| {
            (0..map.width())
                .map(|x| {
                    let p = GridPos::new(x, y);
                    match map.cell(p).unwrap_or(CellKind::Empty) {
                        CellKind::Wall => '#',
                        CellKind::Goal => 'G',
                        CellKind::Hazard => 'H',
                        CellKind::Empty if p == map.start() => 'S',
                        CellKind::Empty => '.',
                    }
                })
                .collect()
        })
        .collect()
}

fn terminal_str(t: Terminal) -> &'static str {
    match t {
        Terminal::Ongoing => "ongoing",
        Terminal::Terminated(TerminalReason::GoalReached) => "goal",
        Terminal::Terminated(TerminalReason::Failure) => "failure",
        Terminal::Terminated(TerminalReason::HorizonReached) => "horizon",
        Terminal::Terminated(TerminalReason::Cancelled) => "cancelled",
        Terminal::Terminated(TerminalReason::InvariantViolated) => "invariant",
    }
}

fn parse_terminal(s: &str) -> Option<Terminal> {
    Some(match s {
        "ongoing" => Terminal::Ongoing,
        "goal" => Terminal::Terminated(TerminalReason::GoalReached),
        "failure" => Terminal::Terminated(TerminalReason::Failure),
        "horizon" => Terminal::Terminated(TerminalReason::HorizonReached),
        "cancelled" => Terminal::Terminated(TerminalReason::Cancelled),
        "invariant" => Terminal::Terminated(TerminalReason::InvariantViolated),
        _ => return None,
    })
}

fn compute_gaps(frames: &[Frame], last_tick: u64) -> Vec<Gap> {
    let mut gaps = Vec::new();
    let mut expected = 0u64;
    for f in frames {
        if f.tick > expected {
            gaps.push(Gap {
                from: expected,
                to: f.tick - 1,
            });
        }
        expected = f.tick + 1;
    }
    if expected <= last_tick {
        gaps.push(Gap {
            from: expected,
            to: last_tick,
        });
    }
    gaps
}

impl Recording {
    /// Construye una grabación (versión actual) validando orden estricto de ticks.
    pub fn new(
        seed: u64,
        map_rows: Vec<String>,
        last_tick: u64,
        frames: Vec<Frame>,
    ) -> Result<Self, OpenError> {
        let refs: Vec<&str> = map_rows.iter().map(String::as_str).collect();
        let map =
            GridMap::from_ascii(&refs).map_err(|e| OpenError::Corrupt(format!("map: {e}")))?;
        if frames.windows(2).any(|w| w[0].tick >= w[1].tick) {
            return corrupt("frame ticks must be strictly increasing");
        }
        let last_tick = last_tick.max(frames.last().map_or(0, |f| f.tick));
        let gaps = compute_gaps(&frames, last_tick);
        Ok(Self {
            version: SUPPORTED_VERSION,
            seed,
            map_rows,
            map,
            last_tick,
            frames,
            gaps,
        })
    }

    /// Lee sólo la versión de la cabecera (para mostrar incompatibilidades sin abrir).
    pub fn probe_version(bytes: &[u8]) -> Result<u32, OpenError> {
        let text = std::str::from_utf8(bytes).map_err(|_| OpenError::NotARecording)?;
        let first = text
            .lines()
            .find(|l| !l.trim().is_empty())
            .ok_or(OpenError::NotARecording)?;
        let rest = first
            .strip_prefix(MAGIC)
            .ok_or(OpenError::NotARecording)?
            .trim();
        rest.strip_prefix('v')
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| OpenError::Corrupt(format!("bad version field `{rest}`")))
    }

    /// Abre una grabación serializada.
    pub fn open(bytes: &[u8]) -> Result<Self, OpenError> {
        let found = Self::probe_version(bytes)?;
        if found != SUPPORTED_VERSION {
            return Err(OpenError::IncompatibleVersion {
                found,
                supported: SUPPORTED_VERSION,
            });
        }
        let text = std::str::from_utf8(bytes).map_err(|_| OpenError::NotARecording)?;
        let (mut seed, mut last, mut rows, mut frames) = (None, None, Vec::new(), Vec::new());
        for line in text.lines().filter(|l| !l.trim().is_empty()).skip(1) {
            let (key, rest) = line.split_once(' ').unwrap_or((line, ""));
            match key {
                "seed" => seed = Some(num::<u64>(rest, "seed")?),
                "last_tick" => last = Some(num::<u64>(rest, "last_tick")?),
                "map" => rows.push(rest.to_string()),
                "frame" => frames.push(parse_frame(rest)?),
                other => return corrupt(format!("unknown record `{other}`")),
            }
        }
        let seed = seed.ok_or_else(|| OpenError::Corrupt("missing seed".into()))?;
        let last = last.ok_or_else(|| OpenError::Corrupt("missing last_tick".into()))?;
        Self::new(seed, rows, last, frames)
    }

    /// Serialización determinista.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut s = format!(
            "{MAGIC} v{}\nseed {}\nlast_tick {}\n",
            self.version, self.seed, self.last_tick
        );
        for r in &self.map_rows {
            s.push_str(&format!("map {r}\n"));
        }
        for f in &self.frames {
            s.push_str(&format!(
                "frame {} {} {} {} {:?} {:?} {} {}\n",
                f.tick,
                f.revision,
                f.agent.x,
                f.agent.y,
                f.last_reward,
                f.total_reward,
                f.collisions,
                terminal_str(f.terminal)
            ));
        }
        s.into_bytes()
    }

    pub fn version(&self) -> u32 {
        self.version
    }
    pub fn seed(&self) -> u64 {
        self.seed
    }
    pub fn map(&self) -> &GridMap {
        &self.map
    }
    /// Último tick del rango grabado (declarado en cabecera o último frame).
    pub fn last_tick(&self) -> u64 {
        self.last_tick
    }
    pub fn frames(&self) -> &[Frame] {
        &self.frames
    }
    pub fn gaps(&self) -> &[Gap] {
        &self.gaps
    }

    /// Frame exacto de `tick` (`None` en un hueco o fuera de rango).
    pub fn frame_at(&self, tick: u64) -> Option<&Frame> {
        self.frames
            .binary_search_by_key(&tick, |f| f.tick)
            .ok()
            .map(|i| &self.frames[i])
    }

    /// Vista inmutable del mundo en un frame (entrada del proyector en vivo).
    pub fn world_view(&self, frame: &Frame) -> GridWorldView {
        GridWorldView {
            map: self.map.clone(),
            agent: frame.agent,
            tick: Tick::new(frame.tick),
            total_reward: frame.total_reward,
            last_reward: frame.last_reward,
            collisions: frame.collisions,
            terminal: frame.terminal,
        }
    }
}

fn num<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, OpenError> {
    s.trim()
        .parse()
        .map_err(|_| OpenError::Corrupt(format!("bad {what} `{s}`")))
}

fn parse_frame(rest: &str) -> Result<Frame, OpenError> {
    let p: Vec<&str> = rest.split_whitespace().collect();
    if p.len() != 8 {
        return corrupt(format!("frame needs 8 fields, got {}", p.len()));
    }
    let fl = |s: &str, w: &str| -> Result<f64, OpenError> {
        let v: f64 = num(s, w)?;
        if v.is_finite() {
            Ok(v)
        } else {
            corrupt(format!("non-finite {w}"))
        }
    };
    Ok(Frame {
        tick: num(p[0], "tick")?,
        revision: num(p[1], "revision")?,
        agent: GridPos::new(num(p[2], "x")?, num(p[3], "y")?),
        last_reward: fl(p[4], "reward")?,
        total_reward: fl(p[5], "total_reward")?,
        collisions: num(p[6], "collisions")?,
        terminal: parse_terminal(p[7])
            .ok_or_else(|| OpenError::Corrupt(format!("bad terminal `{}`", p[7])))?,
    })
}

/// Estado de versión mostrado en la vista.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VersionStatus {
    pub found: u32,
    pub supported: u32,
}

impl VersionStatus {
    pub fn compatible(&self) -> bool {
        self.found == self.supported
    }
}

/// Qué hay en el tick actual.
#[derive(Debug, Clone, PartialEq)]
pub enum Slot<'a> {
    Frame(&'a Frame),
    /// Tick sin datos grabados: se muestra explícitamente, no se interpola.
    Missing(Gap),
}

/// Estado de reproducción (reloj propio).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackState {
    pub playing: bool,
    pub speed: f64,
    /// Segundos de reproducción acumulados (independiente del tick virtual).
    pub clock_secs: f64,
}

/// Vista completa para la UI.
#[derive(Debug, Clone, PartialEq)]
pub struct ReplayView<'a> {
    pub tick: u64,
    pub last_tick: u64,
    pub slot: Slot<'a>,
    pub gaps: &'a [Gap],
    pub version: VersionStatus,
    pub playback: PlaybackState,
}

/// Reproductor desconectado: sólo lee la grabación.
pub struct ReplayPlayer<'a> {
    rec: &'a Recording,
    /// Posición en ticks virtuales (fraccionaria mientras se reproduce).
    cursor: f64,
    playing: bool,
    speed: f64,
    clock_secs: f64,
    camera: Camera2D,
}

impl<'a> ReplayPlayer<'a> {
    pub fn new(rec: &'a Recording, window_w: f64, window_h: f64) -> Self {
        Self {
            rec,
            cursor: 0.0,
            playing: false,
            speed: 1.0,
            clock_secs: 0.0,
            camera: Camera2D::new(window_w, window_h, true),
        }
    }

    pub fn tick(&self) -> u64 {
        self.cursor.floor() as u64
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn speed(&self) -> f64 {
        self.speed
    }

    pub fn clock_secs(&self) -> f64 {
        self.clock_secs
    }

    pub fn camera_mut(&mut self) -> &mut Camera2D {
        &mut self.camera
    }

    /// Salta a `tick` (acotado a `0..=last_tick`). No toca el reloj de reproducción.
    pub fn seek(&mut self, tick: u64) {
        self.cursor = tick.min(self.rec.last_tick()) as f64;
    }

    /// Avanza un tick y pausa (paso a paso).
    pub fn step_forward(&mut self) {
        self.playing = false;
        self.seek(self.tick().saturating_add(1));
    }

    pub fn step_back(&mut self) {
        self.playing = false;
        self.seek(self.tick().saturating_sub(1));
    }

    pub fn pause(&mut self) {
        self.playing = false;
    }

    pub fn play(&mut self) {
        if self.tick() < self.rec.last_tick() {
            self.playing = true;
        }
    }

    /// Fija la velocidad; rechaza valores no finitos o fuera de rango.
    pub fn set_speed(&mut self, speed: f64) -> Result<(), f64> {
        if speed.is_finite() && (MIN_SPEED..=MAX_SPEED).contains(&speed) {
            self.speed = speed;
            Ok(())
        } else {
            Err(speed)
        }
    }

    /// Avanza el reloj de reproducción `wall_dt` segundos. Sólo si está reproduciendo se mueve el
    /// cursor (`wall_dt * speed * TICKS_PER_SECOND_1X` ticks); al llegar al final se pausa.
    pub fn advance(&mut self, wall_dt: f64) {
        if !wall_dt.is_finite() || wall_dt <= 0.0 {
            return;
        }
        self.clock_secs += wall_dt;
        if self.playing {
            let end = self.rec.last_tick() as f64;
            self.cursor = (self.cursor + wall_dt * self.speed * TICKS_PER_SECOND_1X).min(end);
            if self.cursor >= end {
                self.playing = false;
            }
        }
    }

    pub fn view(&self) -> ReplayView<'a> {
        let tick = self.tick();
        let slot = match self.rec.frame_at(tick) {
            Some(f) => Slot::Frame(f),
            None => Slot::Missing(
                self.rec
                    .gaps()
                    .iter()
                    .copied()
                    .find(|g| (g.from..=g.to).contains(&tick))
                    .unwrap_or(Gap {
                        from: tick,
                        to: tick,
                    }),
            ),
        };
        ReplayView {
            tick,
            last_tick: self.rec.last_tick(),
            slot,
            gaps: self.rec.gaps(),
            version: VersionStatus {
                found: self.rec.version(),
                supported: SUPPORTED_VERSION,
            },
            playback: PlaybackState {
                playing: self.playing,
                speed: self.speed,
                clock_secs: self.clock_secs,
            },
        }
    }

    /// Escena del tick actual con el proyector en vivo; `None` en un hueco.
    pub fn scene(&self) -> Option<Scene2D> {
        self.rec
            .frame_at(self.tick())
            .map(|f| project_grid_scene(&self.rec.world_view(f)))
    }

    /// SVG del tick actual con la cámara del reproductor (`None` en un hueco).
    pub fn render_svg(&self) -> Option<String> {
        self.scene()
            .map(|s| SoftwareSvgBackend.render_svg(&s, &self.camera))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recorder::record_grid;
    use gogamen_core::MasterSeed;
    use gogamen_lab_grid::world::GridWorldConfig;
    use gogamen_projection::{LayerId, ScenePrimitive};

    use crate::grid_lab::GridRollout;

    const SEED: u64 = 42;
    const TICKS: u64 = 12;

    fn bytes() -> Vec<u8> {
        record_grid(SEED, GridMap::cul_de_sac_map(), TICKS)
            .unwrap()
            .to_bytes()
    }

    fn hash(b: &[u8]) -> u64 {
        b.iter().fold(0xcbf29ce484222325u64, |h, &x| {
            (h ^ x as u64).wrapping_mul(0x100000001b3)
        })
    }

    fn agent_center(scene: &Scene2D) -> (f64, f64) {
        let layer = scene
            .layers()
            .iter()
            .find(|l| l.id() == LayerId::Decisions)
            .unwrap();
        match layer.nodes()[0].primitive() {
            ScenePrimitive::Circle { center, .. } => (center.x(), center.y()),
            p => panic!("agent is not a circle: {p:?}"),
        }
    }

    /// Escena en vivo al tick `t` (motor real, sólo en tests).
    fn live_scene(t: u64) -> Scene2D {
        let mut r = GridRollout::new(GridWorldConfig::default(), MasterSeed::new(SEED)).unwrap();
        for _ in 0..t {
            r.step().unwrap();
        }
        project_grid_scene(&r.view())
    }

    #[test]
    fn ac1_seek_matches_fixtures_and_live_scene() {
        let b = bytes();
        let rec = Recording::open(&b).unwrap();
        assert_eq!(
            (rec.seed(), rec.last_tick(), rec.frames().len()),
            (42, 12, 13)
        );
        assert!(rec.gaps().is_empty());
        assert_eq!(rec.to_bytes(), b, "serialization round-trips");
        assert_eq!(rec.map(), &GridMap::cul_de_sac_map());
        // (tick, agente x, y, colisiones, recompensa del paso) conocidos para seed 42.
        let fixtures = [
            (0u64, 3usize, 3usize, 0u64, 0.0f64),
            (2, 3, 3, 1, -0.5),
            (4, 3, 2, 1, -0.05),
            (7, 2, 1, 1, -0.05),
            (10, 1, 2, 2, -0.05),
            (12, 1, 2, 4, -0.5),
        ];
        let mut p = ReplayPlayer::new(&rec, 480.0, 360.0);
        for (t, x, y, col, rw) in fixtures {
            p.seek(t);
            let view = p.view();
            let Slot::Frame(f) = view.slot else {
                panic!("tick {t} must have a frame")
            };
            assert_eq!(
                (f.tick, f.agent, f.collisions, f.last_reward),
                (t, GridPos::new(x, y), col, rw)
            );
            let scene = p.scene().unwrap();
            assert_eq!(scene.tick().as_u64(), t);
            assert_eq!(agent_center(&scene), (x as f64 + 0.5, -(y as f64) - 0.5));
            assert_eq!(scene, live_scene(t), "replay == live at tick {t}");
        }
        // hacia atrás y paso a paso
        p.seek(5);
        p.step_back();
        assert_eq!(p.tick(), 4);
        p.step_forward();
        p.step_forward();
        assert_eq!(p.scene().unwrap(), live_scene(6));
        // seek acotado
        p.seek(999);
        assert_eq!(p.tick(), 12);
    }

    #[test]
    fn ac2_gaps_are_shown_never_interpolated() {
        let text = String::from_utf8(bytes()).unwrap();
        let holed: String = text
            .lines()
            .filter(|l| {
                !["frame 3 ", "frame 4 ", "frame 5 ", "frame 12 "]
                    .iter()
                    .any(|p| l.starts_with(p))
            })
            .map(|l| format!("{l}\n"))
            .collect();
        let rec = Recording::open(holed.as_bytes()).unwrap();
        assert_eq!(
            rec.gaps(),
            &[Gap { from: 3, to: 5 }, Gap { from: 12, to: 12 }]
        );
        let mut p = ReplayPlayer::new(&rec, 480.0, 360.0);
        p.seek(4);
        let v = p.view();
        assert_eq!(v.slot, Slot::Missing(Gap { from: 3, to: 5 }));
        assert_eq!(v.gaps.len(), 2);
        assert!(p.scene().is_none() && p.render_svg().is_none());
        p.seek(6);
        assert_eq!(p.scene().unwrap(), live_scene(6));
        // recorte al final: la cola ausente es un hueco
        p.seek(12);
        assert!(matches!(p.view().slot, Slot::Missing(_)));
    }

    #[test]
    fn ac2_incompatible_version_and_corrupt_inputs() {
        let text = String::from_utf8(bytes()).unwrap();
        let v9 = text.replacen("REC v1", "REC v9", 1);
        assert_eq!(
            Recording::open(v9.as_bytes()),
            Err(OpenError::IncompatibleVersion {
                found: 9,
                supported: 1
            })
        );
        assert_eq!(Recording::probe_version(v9.as_bytes()), Ok(9));
        let msg = Recording::open(v9.as_bytes()).unwrap_err().to_string();
        assert!(msg.contains("v9") && msg.contains("v1"));
        assert_eq!(Recording::open(b"hello"), Err(OpenError::NotARecording));
        assert!(matches!(
            Recording::open(text.replace("frame 2 ", "frame 7 ").as_bytes()),
            Err(OpenError::Corrupt(_))
        ));
        assert!(matches!(
            Recording::open(text.replace("ongoing", "wat").as_bytes()),
            Err(OpenError::Corrupt(_))
        ));
        let rec = Recording::open(text.as_bytes()).unwrap();
        let p = ReplayPlayer::new(&rec, 10.0, 10.0);
        assert!(p.view().version.compatible());
    }

    #[test]
    fn ac2_replay_module_has_no_engine_sensor_actuator_or_network() {
        let src = include_str!("replay.rs");
        let prod: String = src[..src.find("#[cfg(test)]").unwrap()]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "GridRollout",
            "GridWorldEnv",
            "gogamen_lab_grid::world",
            "recorder",
            "Actuator",
            "Sensor",
            "ControlPort",
            "SessionHost",
            ".apply(",
            ".step(",
            "std::net",
            "std::fs",
            "TcpStream",
        ] {
            assert!(!prod.contains(banned), "replay.rs contains `{banned}`");
        }
        // el reproductor sólo recibe la grabación por referencia compartida
        assert!(prod.contains("rec: &'a Recording"));
        assert!(!prod.contains("&mut Recording") && !prod.contains("&'a mut Recording"));
    }

    #[test]
    fn ac3_view_speed_pause_never_alter_recording() {
        let b = bytes();
        let rec = Recording::open(&b).unwrap();
        let before = (hash(&rec.to_bytes()), rec.clone());
        let mut p = ReplayPlayer::new(&rec, 480.0, 360.0);
        let svg0 = {
            p.seek(3);
            p.render_svg().unwrap()
        };
        p.camera_mut().pan(30.0, -10.0);
        p.camera_mut().zoom_at(2.0, 10.0, 10.0);
        assert_ne!(p.render_svg().unwrap(), svg0, "camera changes view only");
        p.set_speed(4.0).unwrap();
        p.play();
        p.advance(0.3);
        p.pause();
        p.seek(9);
        p.step_back();
        p.set_speed(0.25).unwrap();
        assert!(p.set_speed(f64::NAN).is_err() && p.set_speed(0.0).is_err());
        assert_eq!(before.0, hash(&rec.to_bytes()));
        assert_eq!(before.1, rec);
        assert_eq!(hash(&b), before.0);
    }

    #[test]
    fn ac3_playback_clock_is_independent_of_virtual_tick() {
        let rec = Recording::open(&bytes()).unwrap();
        let run = |speed: f64| {
            let mut p = ReplayPlayer::new(&rec, 480.0, 360.0);
            p.set_speed(speed).unwrap();
            p.play();
            p.advance(0.5);
            (p.tick(), p.clock_secs())
        };
        // 0.5 s a 1x = 5 ticks; 2x = 10; 0.5x = 2 (2.5 floor). Mismo reloj.
        assert_eq!(run(1.0), (5, 0.5));
        assert_eq!(run(2.0), (10, 0.5));
        assert_eq!(run(0.5), (2, 0.5));

        let mut p = ReplayPlayer::new(&rec, 480.0, 360.0);
        // en pausa el reloj corre pero el tick no
        p.advance(1.0);
        assert_eq!((p.tick(), p.clock_secs()), (0, 1.0));
        // seek no altera el reloj; el reloj no altera el tick pausado
        p.seek(8);
        assert_eq!((p.tick(), p.clock_secs()), (8, 1.0));
        p.advance(2.0);
        assert_eq!((p.tick(), p.clock_secs()), (8, 3.0));
        // step pausa y mueve un tick sin tocar el reloj
        p.play();
        p.step_forward();
        assert!(!p.is_playing());
        assert_eq!((p.tick(), p.clock_secs()), (9, 3.0));
        // al llegar al final se pausa solo
        p.play();
        p.advance(10.0);
        assert_eq!(
            (p.tick(), p.is_playing(), p.clock_secs()),
            (12, false, 13.0)
        );
        assert_eq!(p.view().playback.clock_secs, 13.0);
    }
}
