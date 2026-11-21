//! Captura y anfitrión headless del emulador libre (GOG-111).
//!
//! Aquí vive todo lo que toca al emulador (adaptador GOG-61 sobre `FreePlayCore`, un núcleo
//! "homebrew" generado por código, sin ROM comercial); [`crate::retro_view`] sólo recibe datos.
//!
//! # Frames vs comandos (semántica GOG-115)
//! - **Comandos de emulador** ([`EmuCommand`]: pause/resume/step/input) tienen su propio registro
//!   ([`CommandRecord`]) y [`CommandCounters`]. Se procesan siempre, vean o no el visor.
//! - **Frames** pasan por un anillo acotado hacia el visor ([`FrameCounters`]): si el visor es
//!   lento se descarta el frame más viejo (`dropped_slow`); si está desconectado el frame no se
//!   publica (`dropped_disconnected`). Sin visor (`headless`) no hay anillo ni descartes.
//!   Descartar frames nunca afecta al emulador ni a la cola de comandos.
//!
//! Invariantes: `produced = published + dropped_disconnected` y
//! `published = consumed + dropped_slow + en_cola`.
//!
//! Sin dependencias de UI: el modo headless sólo usa `gogamen-lab-retro`.

use std::collections::VecDeque;

use gogamen_core::EnvironmentObserver;
use gogamen_lab_retro::{Adapter, AdapterConfig, AdapterError, FB_HEIGHT, FB_WIDTH, FreePlayCore};
use gogamen_projection::{ProjectionError, Texture};

use crate::retro_view::{
    CommandCounters, FrameCounters, OverlayItem, PerceptionSource, RetroFrameView, RetroViewError,
};

/// Licencia/atribución de la fuente homebrew (se conserva en la vista, el SVG y el README).
pub const HOMEBREW_LICENSE: &str = "FreePlay homebrew-style core (gogamen-lab-retro): original code, MIT license, no ROM, no third-party assets; frames generated procedurally";

/// Paleta RGBA de `FreePlayCore` (índices 0..=3: fondo, jugador, moneda, enemigo).
pub const FREEPLAY_PALETTE: [[u8; 4]; 4] = [
    [16, 16, 32, 255],
    [64, 224, 96, 255],
    [255, 208, 32, 255],
    [224, 48, 64, 255],
];

/// Número de acciones discretas del adaptador (tabla v1).
pub const ACTION_COUNT: usize = 9;

/// Comandos de control del emulador.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmuCommand {
    Pause,
    Resume,
    /// Una decisión (`frame_skip` frames); sólo en pausa.
    Step,
    /// Fija la acción discreta mantenida (0..9).
    Input(usize),
}

/// Motivo de rechazo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandReject {
    NotPaused,
    InvalidAction,
    EpisodeEnded,
}

/// Resultado de un comando.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandOutcome {
    Applied,
    Rejected(CommandReject),
}

/// Registro de un comando recibido.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandRecord {
    pub seq: u64,
    pub command: EmuCommand,
    pub outcome: CommandOutcome,
}

/// Frame publicado hacia el visor (bytes sin modificar + items privilegiados del instante).
#[derive(Debug, Clone, PartialEq)]
pub struct FramePacket {
    pub index: u64,
    pub texture: Texture,
    pub privileged: Vec<OverlayItem>,
}

#[derive(Debug)]
struct FrameRing {
    capacity: usize,
    connected: bool,
    queue: VecDeque<FramePacket>,
}

/// Errores del anfitrión.
#[derive(Debug, Clone, PartialEq)]
pub enum HostError {
    Adapter(AdapterError),
    Projection(ProjectionError),
}

impl From<AdapterError> for HostError {
    fn from(e: AdapterError) -> Self {
        Self::Adapter(e)
    }
}

impl From<ProjectionError> for HostError {
    fn from(e: ProjectionError) -> Self {
        Self::Projection(e)
    }
}

/// Anfitrión del emulador: comandos, decisiones y publicación de frames hacia un visor opcional.
pub struct RetroHost {
    adapter: Adapter<FreePlayCore>,
    paused: bool,
    held: usize,
    log: Vec<CommandRecord>,
    commands: CommandCounters,
    frames: FrameCounters,
    ring: Option<FrameRing>,
    decisions: u64,
}

impl RetroHost {
    /// Sin visor ni UI (ejecución headless).
    pub fn headless(config: AdapterConfig) -> Result<Self, HostError> {
        Self::build(config, None)
    }

    /// Con un visor de cola acotada `capacity` (mínimo 1).
    pub fn with_viewer(config: AdapterConfig, capacity: usize) -> Result<Self, HostError> {
        Self::build(
            config,
            Some(FrameRing {
                capacity: capacity.max(1),
                connected: true,
                queue: VecDeque::new(),
            }),
        )
    }

    fn build(config: AdapterConfig, ring: Option<FrameRing>) -> Result<Self, HostError> {
        Ok(Self {
            adapter: Adapter::new(FreePlayCore::new(), config)?,
            paused: false,
            held: 0,
            log: Vec::new(),
            commands: CommandCounters::default(),
            frames: FrameCounters::default(),
            ring,
            decisions: 0,
        })
    }

    pub fn is_paused(&self) -> bool {
        self.paused
    }

    pub fn adapter(&self) -> &Adapter<FreePlayCore> {
        &self.adapter
    }

    pub fn command_log(&self) -> &[CommandRecord] {
        &self.log
    }

    pub fn command_counters(&self) -> CommandCounters {
        self.commands
    }

    pub fn frame_counters(&self) -> FrameCounters {
        self.frames
    }

    /// Conecta/desconecta el visor (sin efecto en headless).
    pub fn set_viewer_connected(&mut self, connected: bool) {
        if let Some(r) = &mut self.ring {
            r.connected = connected;
        }
    }

    /// Frames en cola del visor.
    pub fn queued_frames(&self) -> usize {
        self.ring.as_ref().map_or(0, |r| r.queue.len())
    }

    /// El visor consume el frame más antiguo.
    pub fn take_frame(&mut self) -> Option<FramePacket> {
        let p = self.ring.as_mut()?.queue.pop_front()?;
        self.frames.consumed += 1;
        Some(p)
    }

    /// Procesa un comando de emulador (independiente del estado del visor).
    pub fn submit(&mut self, command: EmuCommand) -> Result<CommandOutcome, HostError> {
        let outcome = match command {
            EmuCommand::Pause => {
                self.paused = true;
                CommandOutcome::Applied
            }
            EmuCommand::Resume => {
                self.paused = false;
                CommandOutcome::Applied
            }
            EmuCommand::Input(a) if a >= ACTION_COUNT => {
                CommandOutcome::Rejected(CommandReject::InvalidAction)
            }
            EmuCommand::Input(a) => {
                self.held = a;
                CommandOutcome::Applied
            }
            EmuCommand::Step if !self.paused => CommandOutcome::Rejected(CommandReject::NotPaused),
            EmuCommand::Step => {
                if self.adapter.terminal().is_terminal() {
                    CommandOutcome::Rejected(CommandReject::EpisodeEnded)
                } else {
                    self.decide()?;
                    CommandOutcome::Applied
                }
            }
        };
        self.commands.received += 1;
        match outcome {
            CommandOutcome::Applied => self.commands.applied += 1,
            CommandOutcome::Rejected(_) => self.commands.rejected += 1,
        }
        self.log.push(CommandRecord {
            seq: self.commands.received,
            command,
            outcome,
        });
        Ok(outcome)
    }

    /// Avance del bucle: ejecuta una decisión si no está en pausa. Devuelve si hubo frame.
    pub fn tick(&mut self) -> Result<bool, HostError> {
        if self.paused || self.adapter.terminal().is_terminal() {
            return Ok(false);
        }
        self.decide()?;
        Ok(true)
    }

    fn decide(&mut self) -> Result<(), HostError> {
        self.adapter.step(self.held)?;
        self.decisions += 1;
        self.frames.produced += 1;
        let packet = FramePacket {
            index: self.decisions,
            texture: self.texture()?,
            privileged: self.privileged_items(),
        };
        if let Some(r) = &mut self.ring {
            if !r.connected {
                self.frames.dropped_disconnected += 1;
            } else {
                if r.queue.len() >= r.capacity {
                    r.queue.pop_front();
                    self.frames.dropped_slow += 1;
                }
                r.queue.push_back(packet);
                self.frames.published += 1;
            }
        }
        Ok(())
    }

    /// Framebuffer actual como textura indexada (bytes idénticos a los del núcleo).
    pub fn texture(&self) -> Result<Texture, ProjectionError> {
        Texture::indexed8(
            FB_WIDTH as u32,
            FB_HEIGHT as u32,
            FREEPLAY_PALETTE.to_vec(),
            self.adapter.framebuffer().to_vec(),
        )
    }

    /// Items privilegiados leídos de la RAM (observación del adaptador).
    pub fn privileged_items(&self) -> Vec<OverlayItem> {
        let Ok(obs) = self.adapter.observe() else {
            return Vec::new();
        };
        privileged_from_ram(obs.features())
    }

    /// Vista del frame actual. `run_detector` activa el detector de sólo-píxeles; con `false` la
    /// lista de detecciones queda vacía.
    pub fn current_view(&self, run_detector: bool) -> Result<RetroFrameView, RetroViewError> {
        self.view_of(
            self.decisions,
            self.texture()?,
            self.privileged_items(),
            run_detector,
        )
    }

    /// Vista de un paquete publicado.
    pub fn packet_view(
        &self,
        packet: &FramePacket,
        run_detector: bool,
    ) -> Result<RetroFrameView, RetroViewError> {
        self.view_of(
            packet.index,
            packet.texture.clone(),
            packet.privileged.clone(),
            run_detector,
        )
    }

    fn view_of(
        &self,
        index: u64,
        texture: Texture,
        privileged: Vec<OverlayItem>,
        run_detector: bool,
    ) -> Result<RetroFrameView, RetroViewError> {
        let detections = if run_detector {
            detect_index_boxes(&texture, &[(1, "player"), (2, "coin"), (3, "enemy")])
        } else {
            Vec::new()
        };
        RetroFrameView::new(
            index,
            texture,
            privileged,
            detections,
            HOMEBREW_LICENSE,
            self.frames,
            self.commands,
        )
    }
}

/// Posiciones del layout de RAM de `FreePlayCore` (jugador `[0,1]`, moneda `[2,3]`, enemigo `[10,11]`).
pub fn privileged_from_ram(ram: &[f64]) -> Vec<OverlayItem> {
    let at = |i: usize| ram.get(i).map(|v| *v as u32);
    [
        ("player", 0usize, 1usize),
        ("coin", 2, 3),
        ("enemy", 10, 11),
    ]
    .iter()
    .filter_map(|&(label, ix, iy)| {
        Some(OverlayItem {
            source: PerceptionSource::Privileged,
            label: label.to_owned(),
            x: at(ix)?,
            y: at(iy)?,
            w: 1,
            h: 1,
        })
    })
    .collect()
}

/// Detector de sólo píxeles: caja envolvente de todos los píxeles con cada índice de paleta
/// pedido. No inventa: si el índice no aparece en el frame, no devuelve nada para él.
pub fn detect_index_boxes(texture: &Texture, targets: &[(u8, &str)]) -> Vec<OverlayItem> {
    let (w, h) = (texture.width(), texture.height());
    let mut out = Vec::new();
    for &(index, label) in targets {
        let mut bb: Option<(u32, u32, u32, u32)> = None;
        for y in 0..h {
            for x in 0..w {
                if texture.pixels()[y as usize * w as usize + x as usize] == index {
                    bb = Some(match bb {
                        None => (x, y, x, y),
                        Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                    });
                }
            }
        }
        if let Some((x0, y0, x1, y1)) = bb {
            out.push(OverlayItem {
                source: PerceptionSource::PixelOnly,
                label: label.to_owned(),
                x: x0,
                y: y0,
                w: x1 - x0 + 1,
                h: y1 - y0 + 1,
            });
        }
    }
    out
}
