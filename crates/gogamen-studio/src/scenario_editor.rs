//! Edición mínima de escenario Grid en modo diseño (GOG-107). No es un editor generalista.
//!
//! - [`ScenarioDraft`] es un modelo puro y headless: guarda la definición *committed* como un
//!   `ExperimentSpec` (el mismo contrato JSON que consume la CLI; no hay formato propio de Studio).
//! - Cada [`ScenarioEdit`] genera un spec candidato que se valida con
//!   [`grid_scenario_from_experiment`] (spec + mapa + instanciación del motor) ANTES de sustituir
//!   la definición committed. Una edición inválida devuelve [`EditError`] y no cambia nada
//!   (ni la definición, ni las pilas undo/redo).
//! - Undo/redo opera sólo sobre definiciones de diseño ya validadas. No tiene referencia alguna a
//!   `SessionHost`, `ControlPort`, recorder ni replay: no puede hacer rewind externo.
//! - Guardar/cargar usa `ExperimentSpec::{to_canonical_json, from_json}`; el archivo resultante se
//!   ejecuta tal cual con `gogamen run-grid-spec --json-file <path>`.

use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

use gogamen_core::MasterSeed;
use gogamen_lab_grid::{
    GRID_MAP_CONFIG_KEY, GRID_WORLD_NODE_KIND, GridEpisodeSummary, GridMap, GridPos, GridScenario,
    GridScenarioError, grid_experiment_spec, grid_scenario_from_experiment, map_to_ascii_rows,
    run_grid_episode_with_config,
};
use gogamen_spec::{ExperimentSpec, SpecConfigValue, SpecError};

/// Recursos colocables en el mapa (el resto de celdas no libres son obstáculos).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Start,
    Goal,
    Hazard,
}

impl ResourceKind {
    const fn glyph(self) -> char {
        match self {
            Self::Start => 'S',
            Self::Goal => 'G',
            Self::Hazard => 'H',
        }
    }

    const fn from_glyph(c: char) -> Option<Self> {
        match c {
            'S' => Some(Self::Start),
            'G' => Some(Self::Goal),
            'H' => Some(Self::Hazard),
            _ => None,
        }
    }
}

/// Comandos de edición mínimos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenarioEdit {
    AddResource { kind: ResourceKind, pos: GridPos },
    RemoveResource { pos: GridPos },
    MoveResource { from: GridPos, to: GridPos },
    AddObstacle { pos: GridPos },
    RemoveObstacle { pos: GridPos },
    SetSeed(u64),
}

/// Error tipado de edición/carga. La definición committed nunca cambia cuando se devuelve.
#[derive(Debug, Clone, PartialEq)]
pub enum EditError {
    OutOfBounds(GridPos),
    CellNotEmpty(GridPos),
    NoResourceAt(GridPos),
    NoObstacleAt(GridPos),
    /// El spec candidato no pasó la validación estructural (p. ej. sin meta o dos metas).
    Invalid(GridScenarioError),
    /// El JSON no cumple el contrato `ExperimentSpec`.
    Spec(SpecError),
    Io(String),
}

impl fmt::Display for EditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfBounds(p) => write!(f, "celda ({}, {}) fuera del mapa", p.x, p.y),
            Self::CellNotEmpty(p) => write!(f, "celda ({}, {}) no está libre", p.x, p.y),
            Self::NoResourceAt(p) => write!(f, "no hay recurso en ({}, {})", p.x, p.y),
            Self::NoObstacleAt(p) => write!(f, "no hay obstáculo en ({}, {})", p.x, p.y),
            Self::Invalid(e) => write!(f, "edición inválida: {e}"),
            Self::Spec(e) => write!(f, "spec inválido: {e}"),
            Self::Io(e) => write!(f, "e/s: {e}"),
        }
    }
}

impl std::error::Error for EditError {}

impl From<io::Error> for EditError {
    fn from(value: io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

/// Borrador de escenario: definición committed + historial de diseño.
#[derive(Debug, Clone)]
pub struct ScenarioDraft {
    committed: ExperimentSpec,
    undo: Vec<ExperimentSpec>,
    redo: Vec<ExperimentSpec>,
}

impl ScenarioDraft {
    /// Crea un borrador desde un mapa Grid existente.
    pub fn new(experiment_id: &str, seed: u64, map: &GridMap) -> Result<Self, EditError> {
        let spec = grid_experiment_spec(experiment_id, seed, &map_to_ascii_rows(map));
        Self::from_spec(spec)
    }

    /// Crea un borrador desde un spec; debe ser un escenario Grid válido.
    pub fn from_spec(spec: ExperimentSpec) -> Result<Self, EditError> {
        grid_scenario_from_experiment(&spec).map_err(EditError::Invalid)?;
        Ok(Self {
            committed: spec,
            undo: Vec::new(),
            redo: Vec::new(),
        })
    }

    /// Carga desde el JSON del contrato `ExperimentSpec` (el mismo que lee la CLI).
    pub fn from_json(json: &str) -> Result<Self, EditError> {
        let spec = ExperimentSpec::from_json(json).map_err(EditError::Spec)?;
        Self::from_spec(spec)
    }

    pub fn load(path: &Path) -> Result<Self, EditError> {
        Self::from_json(&fs::read_to_string(path)?)
    }

    /// JSON canónico de la definición committed.
    pub fn to_json(&self) -> Result<String, EditError> {
        self.committed.to_canonical_json().map_err(EditError::Spec)
    }

    pub fn save(&self, path: &Path) -> Result<(), EditError> {
        fs::write(path, self.to_json()?)?;
        Ok(())
    }

    /// Definición committed (siempre válida).
    pub fn spec(&self) -> &ExperimentSpec {
        &self.committed
    }

    /// Plan ejecutable derivado de la definición committed.
    pub fn plan(&self) -> GridScenario {
        grid_scenario_from_experiment(&self.committed)
            .expect("committed definition is validated on every commit")
    }

    /// Ejecuta el plan `ticks` pasos con la semilla del spec.
    pub fn run(&self, ticks: u64) -> Result<GridEpisodeSummary, EditError> {
        let plan = self.plan();
        run_grid_episode_with_config(plan.config, ticks, MasterSeed::new(plan.seed))
            .map_err(|e| EditError::Invalid(GridScenarioError::Engine(e)))
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Aplica una edición: valida el candidato y sólo entonces lo confirma.
    pub fn apply(&mut self, edit: ScenarioEdit) -> Result<(), EditError> {
        let candidate = self.build_candidate(edit)?;
        grid_scenario_from_experiment(&candidate).map_err(EditError::Invalid)?;
        let previous = std::mem::replace(&mut self.committed, candidate);
        self.undo.push(previous);
        self.redo.clear();
        Ok(())
    }

    /// Deshace la última edición confirmada (sólo diseño). `false` si no hay historial.
    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(prev) => {
                self.redo.push(std::mem::replace(&mut self.committed, prev));
                true
            }
            None => false,
        }
    }

    pub fn redo(&mut self) -> bool {
        match self.redo.pop() {
            Some(next) => {
                self.undo.push(std::mem::replace(&mut self.committed, next));
                true
            }
            None => false,
        }
    }

    fn rows(&self) -> Vec<Vec<char>> {
        let ascii = self
            .committed
            .system
            .nodes
            .iter()
            .find(|n| n.kind == GRID_WORLD_NODE_KIND)
            .and_then(|n| n.config.get(GRID_MAP_CONFIG_KEY));
        match ascii {
            Some(SpecConfigValue::String(s)) => {
                s.split('\n').map(|r| r.chars().collect()).collect()
            }
            _ => Vec::new(), // imposible: la definición committed está validada
        }
    }

    fn build_candidate(&self, edit: ScenarioEdit) -> Result<ExperimentSpec, EditError> {
        let mut seed = self.committed.seed;
        let mut grid = self.rows();
        let cell = |grid: &Vec<Vec<char>>, p: GridPos| -> Result<char, EditError> {
            grid.get(p.y)
                .and_then(|r| r.get(p.x))
                .copied()
                .ok_or(EditError::OutOfBounds(p))
        };
        match edit {
            ScenarioEdit::SetSeed(s) => seed = s,
            ScenarioEdit::AddResource { kind, pos } => {
                if cell(&grid, pos)? != '.' {
                    return Err(EditError::CellNotEmpty(pos));
                }
                grid[pos.y][pos.x] = kind.glyph();
            }
            ScenarioEdit::RemoveResource { pos } => {
                if ResourceKind::from_glyph(cell(&grid, pos)?).is_none() {
                    return Err(EditError::NoResourceAt(pos));
                }
                grid[pos.y][pos.x] = '.';
            }
            ScenarioEdit::MoveResource { from, to } => {
                let c = cell(&grid, from)?;
                if ResourceKind::from_glyph(c).is_none() {
                    return Err(EditError::NoResourceAt(from));
                }
                if cell(&grid, to)? != '.' {
                    return Err(EditError::CellNotEmpty(to));
                }
                grid[from.y][from.x] = '.';
                grid[to.y][to.x] = c;
            }
            ScenarioEdit::AddObstacle { pos } => {
                if cell(&grid, pos)? != '.' {
                    return Err(EditError::CellNotEmpty(pos));
                }
                grid[pos.y][pos.x] = '#';
            }
            ScenarioEdit::RemoveObstacle { pos } => {
                if cell(&grid, pos)? != '#' {
                    return Err(EditError::NoObstacleAt(pos));
                }
                grid[pos.y][pos.x] = '.';
            }
        }
        let rows: Vec<String> = grid.iter().map(|r| r.iter().collect()).collect();
        let mut spec = self.committed.clone();
        spec.seed = seed;
        if let Some(node) = spec
            .system
            .nodes
            .iter_mut()
            .find(|n| n.kind == GRID_WORLD_NODE_KIND)
        {
            node.config.insert(
                GRID_MAP_CONFIG_KEY.to_owned(),
                SpecConfigValue::String(rows.join("\n")),
            );
        }
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    use crate::grid_lab::{GRID_LAB_ID, grid_lab_entry_with_probe};
    use crate::recorder::record_grid;
    use crate::shell::StudioShell;
    use gogamen_lab_grid::GridWorldConfig;
    use std::path::PathBuf;
    use std::process::Command;

    fn p(x: usize, y: usize) -> GridPos {
        GridPos::new(x, y)
    }

    fn draft() -> ScenarioDraft {
        let map = GridMap::from_ascii(&["S..#.", ".#...", "...#G", "....."]).unwrap();
        ScenarioDraft::new("playground", 42, &map).unwrap()
    }

    fn edited() -> ScenarioDraft {
        let mut d = draft();
        d.apply(ScenarioEdit::AddObstacle { pos: p(1, 3) }).unwrap();
        d.apply(ScenarioEdit::RemoveObstacle { pos: p(3, 0) })
            .unwrap();
        d.apply(ScenarioEdit::AddResource {
            kind: ResourceKind::Hazard,
            pos: p(2, 1),
        })
        .unwrap();
        d.apply(ScenarioEdit::MoveResource {
            from: p(4, 2),
            to: p(4, 3),
        })
        .unwrap();
        d.apply(ScenarioEdit::SetSeed(7)).unwrap();
        d
    }

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("gogamen_scn_{}_{name}", std::process::id()))
    }

    #[test]
    fn ac1_save_load_compile_run_twice_same_trace_hash() {
        let d = edited();
        let path = tmp("ac1.json");
        d.save(&path).unwrap();
        let loaded = ScenarioDraft::load(&path).unwrap();
        let _ = fs::remove_file(&path);
        assert_eq!(loaded.spec(), d.spec());
        assert_eq!(loaded.to_json().unwrap(), d.to_json().unwrap());
        let a = loaded.run(60).unwrap();
        let b = loaded.run(60).unwrap();
        let c = d.run(60).unwrap();
        assert_eq!(a.trace_hash, b.trace_hash);
        assert_eq!(a.trace_hash, c.trace_hash);
        assert_eq!(loaded.plan().seed, 7);
        assert!(loaded.plan().config.map.cell(p(3, 0)).is_some());
        // la semilla importa
        let mut other = loaded.clone();
        other.apply(ScenarioEdit::SetSeed(8)).unwrap();
        assert_ne!(other.run(60).unwrap().trace_hash, a.trace_hash);
    }

    #[test]
    fn ac2_invalid_edits_keep_definition_byte_identical() {
        let mut d = edited();
        let before = d.to_json().unwrap();
        let undo_len = d.undo.len();
        let bad = [
            ScenarioEdit::RemoveResource { pos: p(4, 3) }, // quita la meta
            ScenarioEdit::AddResource {
                kind: ResourceKind::Goal,
                pos: p(0, 3),
            }, // 2 metas
            ScenarioEdit::AddResource {
                kind: ResourceKind::Start,
                pos: p(0, 2),
            }, // 2 inicios
            ScenarioEdit::AddObstacle { pos: p(0, 0) },    // celda ocupada (S)
            ScenarioEdit::AddObstacle { pos: p(99, 0) },   // fuera de mapa
            ScenarioEdit::RemoveObstacle { pos: p(0, 3) }, // no hay obstáculo
            ScenarioEdit::MoveResource {
                from: p(0, 3),
                to: p(1, 3),
            }, // no es recurso
        ];
        for edit in bad {
            assert!(d.apply(edit).is_err(), "{edit:?}");
            assert_eq!(d.to_json().unwrap(), before, "{edit:?}");
            assert_eq!(d.undo.len(), undo_len);
        }
        assert!(matches!(
            d.apply(ScenarioEdit::RemoveResource { pos: p(4, 3) }),
            Err(EditError::Invalid(GridScenarioError::Map(_)))
        ));
        assert!(matches!(
            ScenarioDraft::from_json("{\"x\":1}"),
            Err(EditError::Spec(_))
        ));
    }

    #[test]
    fn ac2_undo_redo_only_touch_drafts_and_not_session_or_recorder() {
        let (entry, probe) =
            grid_lab_entry_with_probe(GridWorldConfig::default(), MasterSeed::new(42)).unwrap();
        let mut shell = StudioShell::new(vec![entry], 400.0, 300.0).unwrap();
        shell.select_lab(GRID_LAB_ID).unwrap();
        let port = shell.control_port().unwrap().unwrap();
        let session = shell.observation_port().unwrap().observe().session;
        for i in 1..=3 {
            let ack = port.submit(ControlCommand {
                id: CommandId(i),
                session,
                action: ControlAction::Step,
            });
            assert!(matches!(ack.outcome, AckOutcome::Applied));
            shell.sync_active();
        }
        let obs_before = shell.observation_port().unwrap().observe();
        let snap_before = probe.snapshot();
        let svg_before = shell.render_svg().unwrap();
        let map = GridMap::cul_de_sac_map();
        let rec_before = record_grid(42, map.clone(), 10).unwrap().to_bytes();

        let mut d = ScenarioDraft::new("sandbox", 42, &map).unwrap();
        let original = d.to_json().unwrap();
        d.apply(ScenarioEdit::AddObstacle { pos: p(1, 1) }).unwrap();
        d.apply(ScenarioEdit::SetSeed(9)).unwrap();
        assert!(d.undo() && d.undo());
        assert!(!d.undo());
        assert_eq!(d.to_json().unwrap(), original);
        assert!(d.redo() && d.can_undo());
        assert!(d.redo() && !d.redo());
        // una edición nueva invalida redo
        assert!(d.undo());
        d.apply(ScenarioEdit::SetSeed(10)).unwrap();
        assert!(!d.can_redo());

        shell.sync_active();
        assert_eq!(shell.observation_port().unwrap().observe(), obs_before);
        assert_eq!(probe.snapshot(), snap_before);
        assert_eq!(shell.render_svg().unwrap(), svg_before);
        assert_eq!(record_grid(42, map, 10).unwrap().to_bytes(), rec_before);
    }

    fn cli_binary() -> PathBuf {
        let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
        let status = Command::new(cargo)
            .args(["build", "-p", "gogamen-cli", "--offline", "--quiet"])
            .status()
            .expect("spawn cargo build");
        assert!(status.success(), "cargo build -p gogamen-cli failed");
        // <target>/debug/deps/<test-exe> -> <target>/debug/gogamen
        let exe = std::env::current_exe().unwrap();
        exe.parent().unwrap().parent().unwrap().join("gogamen")
    }

    #[test]
    fn ac3_edited_spec_runs_through_cli_binary_without_studio() {
        let d = edited();
        let path = tmp("ac3.json");
        d.save(&path).unwrap();
        let bin = cli_binary();
        let run = |extra: &[&str]| {
            let out = Command::new(&bin)
                .args(["run-grid-spec", "--json-file"])
                .arg(&path)
                .args(["--ticks", "60"])
                .args(extra)
                .output()
                .unwrap();
            (out.status.success(), String::from_utf8(out.stdout).unwrap())
        };
        let (ok1, out1) = run(&[]);
        let (ok2, out2) = run(&[]);
        let _ = fs::remove_file(&path);
        assert!(ok1 && ok2, "{out1}");
        assert_eq!(out1, out2);
        let hash = out1
            .lines()
            .find_map(|l| l.trim().strip_prefix("trace_hash: "))
            .unwrap();
        assert_eq!(hash, d.run(60).unwrap().trace_hash_hex());
        assert!(out1.contains("seed: 7"));
    }
}
