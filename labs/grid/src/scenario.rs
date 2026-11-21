//! Contrato de escenario Grid sobre `ExperimentSpec` (GOG-107).
//!
//! Un escenario Grid es un [`ExperimentSpec`] normal (mismo JSON que consume la CLI) cuyo
//! `system` contiene exactamente un nodo de tipo [`GRID_WORLD_NODE_KIND`] con la clave de
//! configuración [`GRID_MAP_CONFIG_KEY`]: el mapa en filas ASCII (`#` muro, `.` libre, `S` inicio,
//! `G` meta, `H` peligro) separadas por `\n`. La semilla es `ExperimentSpec::seed`.
//! No existe formato propio de Studio: Studio y CLI usan estas mismas funciones.

use std::fmt;

use gogamen_core::{DomainError, MasterSeed};
use gogamen_spec::{ExperimentSpec, SpecConfigValue, SpecError, SystemNodeSpec, SystemSpec};

use crate::map::{CellKind, GridMap, GridPos};
use crate::world::{
    GridEpisodeSummary, GridWorldConfig, GridWorldEnv, run_grid_episode_with_config,
};

/// Tipo de nodo que identifica el mundo Grid dentro de un `SystemSpec`.
pub const GRID_WORLD_NODE_KIND: &str = "grid_world";
/// Clave de configuración del nodo con el mapa ASCII.
pub const GRID_MAP_CONFIG_KEY: &str = "map_ascii";
const GRID_NODE_ID: &str = "grid_env";

/// Error al derivar o ejecutar un escenario Grid desde un `ExperimentSpec`.
#[derive(Debug, Clone, PartialEq)]
pub enum GridScenarioError {
    /// El spec no es válido según el contrato `gogamen-spec`.
    Spec(SpecError),
    /// No hay ningún nodo `grid_world`.
    MissingGridNode,
    /// Hay más de un nodo `grid_world`.
    DuplicateGridNode,
    /// Falta (o no es string) la clave `map_ascii`.
    MissingMap,
    /// El mapa no es válido (filas, S/G únicos, límites...).
    Map(DomainError),
    /// El motor rechazó la configuración o falló al ejecutar.
    Engine(DomainError),
}

impl fmt::Display for GridScenarioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spec(e) => write!(f, "spec inválido: {e}"),
            Self::MissingGridNode => write!(f, "falta un nodo '{GRID_WORLD_NODE_KIND}'"),
            Self::DuplicateGridNode => write!(f, "más de un nodo '{GRID_WORLD_NODE_KIND}'"),
            Self::MissingMap => write!(f, "falta la configuración string '{GRID_MAP_CONFIG_KEY}'"),
            Self::Map(e) => write!(f, "mapa inválido: {e:?}"),
            Self::Engine(e) => write!(f, "motor Grid: {e:?}"),
        }
    }
}

impl std::error::Error for GridScenarioError {}

impl From<SpecError> for GridScenarioError {
    fn from(value: SpecError) -> Self {
        Self::Spec(value)
    }
}

/// Plan ejecutable de un escenario: configuración del motor + semilla maestra.
#[derive(Debug, Clone)]
pub struct GridScenario {
    pub seed: u64,
    pub config: GridWorldConfig,
}

/// Filas ASCII canónicas de un mapa (`S` en la celda de inicio).
pub fn map_to_ascii_rows(map: &GridMap) -> Vec<String> {
    (0..map.height())
        .map(|y| {
            (0..map.width())
                .map(|x| {
                    let pos = GridPos::new(x, y);
                    match map.cell(pos).unwrap_or(CellKind::Empty) {
                        CellKind::Wall => '#',
                        CellKind::Goal => 'G',
                        CellKind::Hazard => 'H',
                        CellKind::Empty if pos == map.start() => 'S',
                        CellKind::Empty => '.',
                    }
                })
                .collect()
        })
        .collect()
}

/// Construye el `ExperimentSpec` de un escenario a partir de filas ASCII (sin validar el mapa).
pub fn grid_experiment_spec(experiment_id: &str, seed: u64, rows: &[String]) -> ExperimentSpec {
    let node = SystemNodeSpec::new(GRID_NODE_ID, GRID_WORLD_NODE_KIND).with_config(
        GRID_MAP_CONFIG_KEY,
        SpecConfigValue::String(rows.join("\n")),
    );
    let system = SystemSpec::new(format!("{experiment_id}_system")).with_node(node);
    ExperimentSpec::new(experiment_id, seed, system)
}

/// Valida un `ExperimentSpec` y deriva el plan de ejecución. Es la ÚNICA puerta de validación:
/// la usan el editor de Studio (antes de aceptar una edición) y la CLI (antes de ejecutar).
pub fn grid_scenario_from_experiment(
    spec: &ExperimentSpec,
) -> Result<GridScenario, GridScenarioError> {
    spec.validate()?;
    let mut nodes = spec
        .system
        .nodes
        .iter()
        .filter(|n| n.kind == GRID_WORLD_NODE_KIND);
    let node = nodes.next().ok_or(GridScenarioError::MissingGridNode)?;
    if nodes.next().is_some() {
        return Err(GridScenarioError::DuplicateGridNode);
    }
    let Some(SpecConfigValue::String(ascii)) = node.config.get(GRID_MAP_CONFIG_KEY) else {
        return Err(GridScenarioError::MissingMap);
    };
    let rows: Vec<&str> = ascii.split('\n').collect();
    let map = GridMap::from_ascii(&rows).map_err(GridScenarioError::Map)?;
    let config = GridWorldConfig {
        map,
        ..GridWorldConfig::default()
    };
    // Instanciar el motor una vez valida la configuración completa antes de aceptarla.
    GridWorldEnv::new(config.clone()).map_err(GridScenarioError::Engine)?;
    Ok(GridScenario {
        seed: spec.seed,
        config,
    })
}

/// Ejecuta el escenario `ticks` pasos con `seed` (por defecto, la del spec).
pub fn run_grid_scenario(
    spec: &ExperimentSpec,
    ticks: u64,
    seed_override: Option<u64>,
) -> Result<GridEpisodeSummary, GridScenarioError> {
    let scenario = grid_scenario_from_experiment(spec)?;
    let seed = MasterSeed::new(seed_override.unwrap_or(scenario.seed));
    run_grid_episode_with_config(scenario.config, ticks, seed).map_err(GridScenarioError::Engine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<String> {
        map_to_ascii_rows(&GridMap::cul_de_sac_map())
    }

    #[test]
    fn spec_json_round_trip_and_matches_default_episode() {
        let spec = grid_experiment_spec("exp", 42, &rows());
        let json = spec.to_canonical_json().unwrap();
        let back = ExperimentSpec::from_json(&json).unwrap();
        assert_eq!(back, spec);
        let a = run_grid_scenario(&back, 40, None).unwrap();
        let b = crate::world::run_grid_episode(40, MasterSeed::new(42), false).unwrap();
        assert_eq!(a.trace_hash, b.trace_hash);
    }

    #[test]
    fn rejects_missing_node_map_and_bad_map() {
        let empty = ExperimentSpec::new("e", 1, SystemSpec::new("s"));
        assert_eq!(
            grid_scenario_from_experiment(&empty).unwrap_err(),
            GridScenarioError::MissingGridNode
        );
        let bad = grid_experiment_spec("e", 1, &["S.#".to_owned()]);
        assert!(matches!(
            grid_scenario_from_experiment(&bad),
            Err(GridScenarioError::Map(_))
        ));
        let mut no_map = grid_experiment_spec("e", 1, &rows());
        no_map.system.nodes[0].config.clear();
        assert_eq!(
            grid_scenario_from_experiment(&no_map).unwrap_err(),
            GridScenarioError::MissingMap
        );
    }
}
