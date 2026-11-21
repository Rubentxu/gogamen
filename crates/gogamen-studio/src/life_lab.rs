//! Life en Studio (GOG-112): proyección compartida de organismos, recursos, especies, energía
//! y genealogía del mundo Life real.
//!
//! - El único mundo, agente y métricas son los del lab `gogamen-lab-life` ([`LifeWorld`],
//!   [`HomeostaticFmcAgent`], [`PopulationMetrics`]); aquí no hay dinámica, planificación ni
//!   cálculo de métricas de población propios (las métricas NOE-104 vienen del lab).
//! - [`LifeRollout`] es el lado motor (headless): por tick `step_tick_with` con el agente FMC
//!   del lab (reproduce al alcanzar el umbral de energía), con hash de traza que incluye los
//!   cambios autorizados de la tasa de mutación.
//! - [`LifeWorldView`] / [`project_life_layers`] / [`project_life_scene`]: proyector puro.
//!   Recursos (Environment), organismos (Decisions; tamaño = energía, `state_code` = especie),
//!   genealogía padre->hijo (capa `Custom(21)`), y *sólo para el organismo seleccionado* los
//!   futuros FMC por la capa común `gogamen-projection::overlay`.
//! - Sólo se proyectan organismos vivos: un organismo muerto desaparece de la escena y del
//!   snapshot del inspector (no hay entidades fantasma). Los ids del lab son monótonos; el epoch
//!   de [`EntityRef`] es el de la sesión ([`life_epoch`]) y cambia en cada `Reset`.
//! - Selección: [`LifeProbe::select`] (con `EntityRef`); se aplica en el siguiente `sync_to`
//!   sólo si su epoch coincide con el vigente.

use std::sync::{Arc, Mutex, MutexGuard};

use gogamen_core::{EntityId, EpistemicKind, MasterSeed, Tick, TraceHasher};
use gogamen_lab_life::{
    Genome, HomeostaticFmcAgent, LifeAction, LifeAgentError, LifeError, LifeWorld, LifeWorldConfig,
    PopulationMetrics,
};
use gogamen_projection::{
    DecisionOverlayFrame, FutureKind, FutureTrajectorySample, LayerId, LayerVisibility,
    OverlayConfig, OverlayLegend, OverlayStats, Point2, ProjectedOverlay, Scene2D, SceneId,
    SceneNode, SceneNodeId, ScenePrimitive, TerminalMarker, TerminalOutcome,
    project_decision_overlay,
};
use gogamen_sdk::extensions::ExtensionId;

use crate::control::Observation;
use crate::inspector::{
    EntityRef, EntityView, Field, FieldDescriptor, FieldSchema, FieldValue, Lod, Provenance,
    SnapshotView,
};
use crate::params::{
    Mutability, ParamDescriptor, ParamId, ParamKind, ParamSchema, ParamUnit, ParamValue,
};
use crate::shell::{LabEntry, LabSession};

/// Id del lab en el registro del shell.
pub const LIFE_LAB_ID: &str = "lab.life";
/// Parámetro mutable: tasa de mutación por gen al reproducirse.
pub const PARAM_MUTATION_RATE: &str = "life.mutation_rate";
/// Parámetro inmutable en caliente: límite de población.
pub const PARAM_MAX_POPULATION: &str = "life.max_population";
/// Id de la entidad "población" (métricas del lab) en el inspector.
pub const LIFE_POPULATION_ID: u64 = 900_000_000;
/// Base de ids de entidades recurso (`base + y * width + x`).
pub const LIFE_RESOURCE_BASE: u64 = 1_000_000_000;
/// Capa de genealogía (padre -> hijo).
pub const GENEALOGY_LAYER: LayerId = LayerId::Custom(21);
/// `state_code` de organismo: `ORGANISM_CODE_BASE + species` (no colisiona con 101..=107 del overlay).
pub const ORGANISM_CODE_BASE: u32 = 200;

/// Población, horizonte y presupuesto del agente FMC del rollout.
const AGENT_POPULATION: usize = 8;
const AGENT_HORIZON: usize = 3;
const AGENT_TRANSITIONS: usize = 48;
const AGENT_SEED_SALT: u64 = 0x11fe_0112;

const SCENE_ID: u64 = 112;
const RESOURCE_NODE_BASE: u64 = 1_000_000;
const EDGE_NODE_BASE: u64 = 2_000_000;
const ORGANISM_NODE_BASE: u64 = 3_000_000;
const BORDER_NODE: u64 = 4_000_000;
const SELECTION_NODE: u64 = 4_000_001;
const SELECTED_LABEL_NODE: u64 = 4_000_002;
const STATUS_LABEL_NODE: u64 = 4_000_003;
const BORDER_CODE: u32 = 5;
const SELECTION_CODE: u32 = 950;

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite life coordinate")
}

fn center(x: usize, y: usize) -> Point2 {
    pt(x as f64 + 0.5, y as f64 + 0.5)
}

/// Configuración demo del lab: mundo pequeño y escaso, con nacimientos y muertes en pocos ticks.
pub fn life_demo_config() -> LifeWorldConfig {
    LifeWorldConfig {
        width: 10,
        height: 10,
        initial_population: 10,
        max_population: 24,
        initial_energy: 8.0,
        max_energy: 14.0,
        basal_cost: 0.8,
        move_cost: 0.3,
        max_eat_per_tick: 3.0,
        max_cell_food: 6.0,
        initial_cell_food: 2.0,
        food_regen_per_cell: 0.6,
        food_regen_prob: 0.08,
        vision_radius: 1,
        reproduction_threshold: 10.0,
        reproduction_cost: 5.0,
        reproduction_overhead: 0.5,
        mutation_rate: 0.3,
        mutation_scale: 0.2,
    }
}

/// Organismo vivo en la vista.
#[derive(Debug, Clone, PartialEq)]
pub struct OrganismView {
    pub id: u64,
    pub x: usize,
    pub y: usize,
    pub energy: f64,
    pub age: u64,
    pub genome: Genome,
    pub species: u32,
    pub parent: Option<u64>,
    pub generation: u64,
}

/// Celda de recurso con alimento.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceView {
    pub x: usize,
    pub y: usize,
    pub food: f64,
}

/// Futuro FMC candidato del organismo seleccionado (posiciones absolutas, centros de celda).
#[derive(Debug, Clone, PartialEq)]
pub struct FutureView {
    pub action: LifeAction,
    pub chosen: bool,
    pub points: Vec<(f64, f64)>,
    pub final_energy: f64,
    pub dies: bool,
}

/// Vista inmutable del mundo Life (entrada del proyector y del inspector).
#[derive(Debug, Clone, PartialEq)]
pub struct LifeWorldView {
    pub tick: Tick,
    pub width: usize,
    pub height: usize,
    pub max_energy: f64,
    pub max_cell_food: f64,
    /// Tasa de mutación vigente.
    pub mutation_rate: f64,
    pub max_population: usize,
    /// Sólo organismos vivos.
    pub organisms: Vec<OrganismView>,
    pub resources: Vec<ResourceView>,
    /// Organismo seleccionado (id del lab), si está vivo.
    pub selected: Option<u64>,
    /// Futuros FMC: no vacío sólo si hay un organismo seleccionado vivo.
    pub futures: Vec<FutureView>,
    /// Métricas de población del lab (NOE-104), sin recalcular aquí.
    pub metrics: PopulationMetrics,
}

fn food_code(food: f64, max: f64) -> u32 {
    let q = if max > 0.0 { food / max } else { 0.0 };
    match q {
        q if q <= 0.25 => 1,
        q if q <= 0.5 => 2,
        q if q <= 0.75 => 3,
        _ => 4,
    }
}

fn overlay_frame(view: &LifeWorldView) -> DecisionOverlayFrame {
    let tick = view.tick;
    let frame = DecisionOverlayFrame::new(SceneId::new(SCENE_ID), tick, tick.as_u64());
    let futures = view
        .futures
        .iter()
        .enumerate()
        .filter(|(_, f)| f.points.len() >= 2)
        .map(|(i, f)| {
            let kind = if f.chosen {
                FutureKind::SelectedFuture
            } else {
                FutureKind::RejectedFuture
            };
            let pts: Vec<Point2> = f.points.iter().map(|&(x, y)| pt(x, y)).collect();
            let end = *pts.last().expect("non-empty");
            let outcome = if f.dies {
                TerminalOutcome::Failure
            } else {
                TerminalOutcome::HorizonReached
            };
            let marker = TerminalMarker::new(end, outcome, (pts.len() - 1) as u32, f.final_energy)
                .expect("finite energy");
            FutureTrajectorySample::new(i as u64, kind, pts, f.final_energy)
                .expect("finite energy")
                .with_terminal(marker)
        })
        .collect();
    frame.with_futures(futures)
}

/// Configuración del overlay común para Life.
pub fn life_overlay_config(visibility: LayerVisibility) -> OverlayConfig {
    let mut cfg = OverlayConfig {
        visibility,
        ..OverlayConfig::default()
    };
    cfg.budget.max_future_samples = 16;
    cfg.budget.max_terminal_markers = 16;
    cfg.budget.max_bytes_per_frame = 1 << 22;
    cfg.budget.export_all_walkers = true;
    cfg
}

/// Proyección pura con capas comunes: escena + leyenda + estadísticas.
///
/// Capas: `Environment` (borde, recursos: `state_code` 1..=4 por nivel de alimento),
/// `Decisions` (organismos: círculo seleccionable `EntityId(id)`, radio por energía,
/// `state_code = 200 + especie`; anillo de selección), `Custom(21)` genealogía (una polilínea
/// padre->hijo por cada par de vivos, nodo `EDGE_NODE_BASE + id_hijo`), futuros FMC (capas
/// comunes del overlay; sólo con organismo seleccionado) y `Labels` (estado + métricas del lab).
pub fn project_life_layers_with(
    view: &LifeWorldView,
    visibility: LayerVisibility,
) -> ProjectedOverlay {
    let mut overlay =
        project_decision_overlay(&overlay_frame(view), &life_overlay_config(visibility))
            .expect("life overlay fits its budget");
    let scene = &mut overlay.scene;
    let mut add = |layer: LayerId, node: SceneNode| {
        scene.add_node(layer, node).expect("unique life node id");
    };
    let border = ScenePrimitive::polyline(
        vec![
            pt(0.0, 0.0),
            pt(view.width as f64, 0.0),
            pt(view.width as f64, view.height as f64),
            pt(0.0, view.height as f64),
        ],
        true,
    )
    .expect("border");
    add(
        LayerId::Environment,
        SceneNode::new(
            SceneNodeId::new(BORDER_NODE),
            None,
            EpistemicKind::Observed,
            border,
            false,
        )
        .with_state_code(BORDER_CODE),
    );
    for r in &view.resources {
        let idx = (r.y * view.width + r.x) as u64;
        add(
            LayerId::Environment,
            SceneNode::new(
                SceneNodeId::new(RESOURCE_NODE_BASE + idx),
                Some(EntityId::new(LIFE_RESOURCE_BASE + idx)),
                EpistemicKind::Observed,
                ScenePrimitive::rect(pt(r.x as f64 + 0.1, r.y as f64 + 0.1), 0.8, 0.8)
                    .expect("resource rect"),
                true,
            )
            .with_state_code(food_code(r.food, view.max_cell_food)),
        );
    }
    for o in &view.organisms {
        if let Some(pid) = o.parent
            && let Some(p) = view.organisms.iter().find(|p| p.id == pid)
        {
            add(
                GENEALOGY_LAYER,
                SceneNode::new(
                    SceneNodeId::new(EDGE_NODE_BASE + o.id),
                    None,
                    EpistemicKind::Observed,
                    ScenePrimitive::polyline(vec![center(p.x, p.y), center(o.x, o.y)], false)
                        .expect("edge"),
                    false,
                ),
            );
        }
    }
    for o in &view.organisms {
        let frac = if view.max_energy > 0.0 {
            (o.energy / view.max_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };
        add(
            LayerId::Decisions,
            SceneNode::new(
                SceneNodeId::new(ORGANISM_NODE_BASE + o.id),
                Some(EntityId::new(o.id)),
                EpistemicKind::Observed,
                ScenePrimitive::circle(center(o.x, o.y), 0.1 + 0.3 * frac).expect("organism"),
                true,
            )
            .with_state_code(ORGANISM_CODE_BASE + o.species),
        );
    }
    if let Some(sel) = view
        .selected
        .and_then(|id| view.organisms.iter().find(|o| o.id == id))
    {
        add(
            LayerId::Decisions,
            SceneNode::new(
                SceneNodeId::new(SELECTION_NODE),
                None,
                EpistemicKind::Observed,
                ScenePrimitive::circle(center(sel.x, sel.y), 0.48).expect("ring"),
                false,
            )
            .with_state_code(SELECTION_CODE),
        );
        add(
            LayerId::Labels,
            SceneNode::new(
                SceneNodeId::new(SELECTED_LABEL_NODE),
                None,
                EpistemicKind::Observed,
                ScenePrimitive::label(
                    pt(sel.x as f64 + 0.6, sel.y as f64 + 0.9),
                    format!(
                        "#{} e={:.2} sp={} gen={}",
                        sel.id, sel.energy, sel.species, sel.generation
                    ),
                ),
                false,
            ),
        );
    }
    let m = &view.metrics;
    let status = format!(
        "tick {} alive {} species {} H {:.3} E {:.2} mut {:.2}",
        view.tick.as_u64(),
        m.alive,
        m.species_richness,
        m.shannon_diversity + 0.0, // normaliza -0.0
        m.mean_energy,
        view.mutation_rate
    );
    add(
        LayerId::Labels,
        SceneNode::new(
            SceneNodeId::new(STATUS_LABEL_NODE),
            None,
            EpistemicKind::Observed,
            ScenePrimitive::label(pt(0.0, view.height as f64 + 0.6), status),
            false,
        ),
    );
    overlay
}

/// [`project_life_layers_with`] con todas las capas visibles.
pub fn project_life_layers(view: &LifeWorldView) -> ProjectedOverlay {
    project_life_layers_with(view, LayerVisibility::all_visible())
}

/// Proyección pura de la vista a `Scene2D` (y arriba, celdas).
pub fn project_life_scene(view: &LifeWorldView) -> Scene2D {
    project_life_layers(view).scene
}

/// Cambio autorizado de la tasa de mutación registrado en la traza del rollout.
#[derive(Debug, Clone, PartialEq)]
pub struct MutationChange {
    /// Pasos lógicos ya ejecutados cuando se aplicó (afecta a las reproducciones siguientes).
    pub at_step: u64,
    pub from: f64,
    pub to: f64,
}

/// Registro lógico de un paso.
#[derive(Debug, Clone, PartialEq)]
pub struct LifeStepRecord {
    /// Tick lógico del mundo tras el paso (1-based).
    pub tick: u64,
    pub alive: usize,
    pub births: usize,
    pub deaths: usize,
    /// Tasa de mutación con la que se ejecutó el paso.
    pub mutation_rate: f64,
    /// Hash de traza (mundo + cambios autorizados) tras el paso.
    pub trace_hash: u64,
}

/// Motor headless: mundo Life real + agente FMC homeostático real.
pub struct LifeRollout {
    config: LifeWorldConfig,
    seed: MasterSeed,
    world: LifeWorld,
    agent: HomeostaticFmcAgent,
    change_digest: u64,
    records: Vec<LifeStepRecord>,
    mutation_changes: Vec<MutationChange>,
}

fn build_agent(
    config: &LifeWorldConfig,
    seed: MasterSeed,
) -> Result<HomeostaticFmcAgent, LifeAgentError> {
    HomeostaticFmcAgent::new(
        config,
        AGENT_POPULATION,
        AGENT_HORIZON,
        AGENT_TRANSITIONS,
        MasterSeed::new(seed.as_u64() ^ AGENT_SEED_SALT),
    )
}

impl LifeRollout {
    pub fn new(config: LifeWorldConfig, seed: MasterSeed) -> Result<Self, LifeAgentError> {
        let world = LifeWorld::new(config.clone(), seed)?;
        let agent = build_agent(&config, seed)?;
        Ok(Self {
            config,
            seed,
            world,
            agent,
            change_digest: seed.as_u64(),
            records: Vec::new(),
            mutation_changes: Vec::new(),
        })
    }

    /// Restaura el mundo inicial sembrado y la tasa de mutación por defecto.
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone(), self.seed).expect("config already validated");
    }

    /// El episodio termina con la extinción.
    pub fn is_terminal(&self) -> bool {
        self.world.alive_count() == 0
    }

    pub fn steps(&self) -> u64 {
        self.records.len() as u64
    }

    pub fn records(&self) -> &[LifeStepRecord] {
        &self.records
    }

    pub fn world(&self) -> &LifeWorld {
        &self.world
    }

    pub fn mutation_rate(&self) -> f64 {
        self.world.config().mutation_rate
    }

    pub fn mutation_changes(&self) -> &[MutationChange] {
        &self.mutation_changes
    }

    /// Hash combinado: estado del mundo + cambios autorizados.
    pub fn trace_hash(&self) -> u64 {
        let mut h = TraceHasher::new();
        h.write_u64(self.world.trace_hash());
        h.write_u64(self.change_digest);
        h.finish()
    }

    /// Aplica una nueva tasa de mutación desde el siguiente paso; queda en el log y en el hash.
    /// Sin efecto (ni traza) si el valor no cambia.
    pub fn set_mutation_rate(&mut self, rate: f64) -> Result<(), LifeError> {
        let from = self.mutation_rate();
        if rate == from {
            return Ok(());
        }
        self.world.set_mutation_rate(rate)?;
        self.mutation_changes.push(MutationChange {
            at_step: self.steps(),
            from,
            to: rate,
        });
        let mut h = TraceHasher::new();
        h.write_u64(self.change_digest);
        h.write_str("mutation_rate");
        h.write_u64(self.steps());
        h.write_f64(rate);
        self.change_digest = h.finish();
        Ok(())
    }

    /// Vista del mundo; los futuros FMC sólo se calculan para `selected` si está vivo.
    pub fn view(&self, selected: Option<u64>) -> LifeWorldView {
        let global = self.world.privileged_global_view();
        let orgs = global.organisms();
        let mut organisms = Vec::new();
        for i in 0..orgs.len() {
            if !orgs.alive()[i] {
                continue;
            }
            let id = orgs.ids()[i];
            let rec = self.world.lineage_of(id);
            organisms.push(OrganismView {
                id,
                x: orgs.pos_x()[i],
                y: orgs.pos_y()[i],
                energy: orgs.energy()[i],
                age: orgs.age()[i],
                genome: orgs.genomes()[i],
                species: orgs.genomes()[i].species_id(),
                parent: rec.and_then(|r| r.parent_id),
                generation: rec.map_or(0, |r| r.generation),
            });
        }
        let (w, h) = (global.width(), global.height());
        let mut resources = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let food = global.food_grid().get(x, y).unwrap_or(0.0);
                if food > 0.0 {
                    resources.push(ResourceView { x, y, food });
                }
            }
        }
        let selected = selected.filter(|id| organisms.iter().any(|o| o.id == *id));
        let futures = selected
            .and_then(|id| {
                let idx = (0..orgs.len()).find(|&i| orgs.alive()[i] && orgs.ids()[i] == id)?;
                let obs = self.world.observe_organism(idx)?;
                let o = organisms.iter().find(|o| o.id == id)?;
                let cands = self.agent.candidate_futures(&obs, AGENT_HORIZON).ok()?;
                Some(
                    cands
                        .into_iter()
                        .map(|c| FutureView {
                            action: c.action,
                            chosen: c.chosen,
                            points: c
                                .offsets
                                .iter()
                                .map(|&(dx, dy)| (o.x as f64 + 0.5 + dx, o.y as f64 + 0.5 + dy))
                                .collect(),
                            final_energy: c.final_energy,
                            dies: c.dies,
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .unwrap_or_default();
        let metrics = PopulationMetrics::from_view(&global).expect("finite population stats");
        LifeWorldView {
            tick: global.tick(),
            width: w,
            height: h,
            max_energy: self.config.max_energy,
            max_cell_food: self.config.max_cell_food,
            mutation_rate: self.mutation_rate(),
            max_population: self.config.max_population,
            organisms,
            resources,
            selected,
            futures,
            metrics,
        }
    }

    /// Un paso lógico (`step_tick_with` + agente FMC del lab). `None` si hay extinción.
    pub fn step(&mut self) -> Option<LifeStepRecord> {
        if self.is_terminal() {
            return None;
        }
        let threshold = self.config.reproduction_threshold;
        let rate = self.mutation_rate();
        let agent = &mut self.agent;
        let report = self
            .world
            .step_tick_with(|obs| agent.decide_with_reproduction(obs, threshold));
        let rec = LifeStepRecord {
            tick: report.tick_completed.as_u64(),
            alive: report.alive_after,
            births: report.births,
            deaths: report.deaths,
            mutation_rate: rate,
            trace_hash: self.trace_hash(),
        };
        self.records.push(rec.clone());
        Some(rec)
    }
}

/// Esquema de parámetros: tasa de mutación mutable + límite de población inmutable en caliente.
pub fn life_param_schema(rollout: &LifeRollout) -> ParamSchema {
    ParamSchema::new(vec![
        ParamDescriptor {
            id: ParamId::new(PARAM_MUTATION_RATE),
            label: "Mutation rate".into(),
            kind: ParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            unit: ParamUnit::None,
            mutability: Mutability::Mutable,
            default: ParamValue::Float(rollout.mutation_rate()),
        },
        ParamDescriptor {
            id: ParamId::new(PARAM_MAX_POPULATION),
            label: "Max population".into(),
            kind: ParamKind::Int {
                min: 1,
                max: 4096,
                step: 1,
            },
            unit: ParamUnit::Custom("organisms"),
            mutability: Mutability::Immutable,
            default: ParamValue::Int(rollout.world().config().max_population as i64),
        },
    ])
    .expect("life param schema is valid")
}

/// Estado compartido sólo-lectura de una sesión Life (observación y tests).
#[derive(Debug, Clone, PartialEq)]
pub struct LifeSnapshot {
    pub scene: Scene2D,
    pub legend: OverlayLegend,
    pub overlay_stats: OverlayStats,
    pub view: LifeWorldView,
    pub records: Vec<LifeStepRecord>,
    pub mutation_changes: Vec<MutationChange>,
    pub trace_hash: u64,
    /// Epoch de sesión vigente (para construir [`EntityRef`]).
    pub epoch: u64,
}

struct Shared {
    snapshot: Option<LifeSnapshot>,
    selected: Option<EntityRef>,
}

/// Sonda de la sesión Life (clonable): snapshot de sólo lectura y selección.
#[derive(Clone)]
pub struct LifeProbe {
    shared: Arc<Mutex<Shared>>,
}

impl LifeProbe {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.shared.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Último snapshot publicado (`None` si no hay sesión creada).
    pub fn snapshot(&self) -> Option<LifeSnapshot> {
        self.lock().snapshot.clone()
    }

    /// Selecciona (o limpia) un organismo; efectivo en el siguiente `sync_to`.
    pub fn select(&self, entity: Option<EntityRef>) {
        self.lock().selected = entity;
    }

    /// Observación combinada: control (GOG-90) + escena proyectada del mundo.
    pub fn observe_with(&self, control: Observation) -> LifeObservation {
        LifeObservation {
            control,
            life: self.snapshot(),
        }
    }
}

/// Observación del lab Life: la del puerto de control más la escena proyectada.
#[derive(Debug, Clone, PartialEq)]
pub struct LifeObservation {
    pub control: Observation,
    pub life: Option<LifeSnapshot>,
}

/// Sesión Life del shell. Sólo avanza el rollout en [`LabSession::sync_to`].
pub struct LifeLabSession {
    rollout: LifeRollout,
    generation: u64,
    epoch: u64,
    scene: Scene2D,
    probe: LifeProbe,
}

impl LifeLabSession {
    fn publish(&mut self) {
        let selected = {
            let mut sh = self.probe.lock();
            if sh.selected.is_some_and(|e| e.epoch != self.epoch) {
                sh.selected = None;
            }
            sh.selected.map(|e| e.id)
        };
        let view = self.rollout.view(selected);
        let overlay = project_life_layers(&view);
        self.scene = overlay.scene.clone();
        self.probe.lock().snapshot = Some(LifeSnapshot {
            scene: overlay.scene,
            legend: overlay.legend,
            overlay_stats: overlay.stats,
            view,
            records: self.rollout.records().to_vec(),
            mutation_changes: self.rollout.mutation_changes().to_vec(),
            trace_hash: self.rollout.trace_hash(),
            epoch: self.epoch,
        });
    }
}

impl LabSession for LifeLabSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, observation: &Observation) {
        if observation.session.generation != self.generation {
            self.generation = observation.session.generation;
            self.rollout.reset();
        }
        self.epoch = life_epoch(observation);
        // Frontera de tick: los parámetros vigentes ya fueron autorizados por el host.
        let rate = observation
            .params
            .iter()
            .find(|(id, _)| id.0 == PARAM_MUTATION_RATE)
            .and_then(|(_, v)| match v {
                ParamValue::Float(r) => Some(*r),
                _ => None,
            });
        if let Some(r) = rate {
            let _ = self.rollout.set_mutation_rate(r);
        }
        while self.rollout.steps() < observation.tick.as_u64() && !self.rollout.is_terminal() {
            if self.rollout.step().is_none() {
                break;
            }
        }
        self.publish();
    }
}

/// Entrada de registro del lab Life con sonda y esquema de parámetros.
pub fn life_lab_entry_with_probe(
    config: LifeWorldConfig,
    seed: MasterSeed,
) -> Result<(LabEntry, LifeProbe), LifeAgentError> {
    let schema = life_param_schema(&LifeRollout::new(config.clone(), seed)?);
    let probe = LifeProbe {
        shared: Arc::new(Mutex::new(Shared {
            snapshot: None,
            selected: None,
        })),
    };
    let p = probe.clone();
    let entry = LabEntry::new(
        ExtensionId::new(LIFE_LAB_ID),
        "Life",
        Arc::new(move || {
            let rollout = LifeRollout::new(config.clone(), seed).expect("validated config");
            let scene = project_life_scene(&rollout.view(None));
            let mut s = LifeLabSession {
                rollout,
                generation: 0,
                epoch: 0,
                scene,
                probe: p.clone(),
            };
            s.publish();
            Box::new(s)
        }),
    )
    .with_param_schema(schema);
    Ok((entry, probe))
}

/// Entrada de registro del lab Life (sin sonda).
pub fn life_lab_entry(
    config: LifeWorldConfig,
    seed: MasterSeed,
) -> Result<LabEntry, LifeAgentError> {
    life_lab_entry_with_probe(config, seed).map(|(e, _)| e)
}

// ---------------------------------------------------------------------------------------------
// Cableado al inspector (GOG-92)
// ---------------------------------------------------------------------------------------------

/// Id de la entidad recurso de la celda `(x, y)`.
pub fn life_resource_id(view: &LifeWorldView, x: usize, y: usize) -> u64 {
    LIFE_RESOURCE_BASE + (y * view.width + x) as u64
}

/// Epoch de entidades Life: sesión (`epoch`, `generation`); cambia con cada `Reset`.
pub fn life_epoch(obs: &Observation) -> u64 {
    (obs.session.epoch << 32) | (obs.session.generation & 0xffff_ffff)
}

/// Descriptores del lab Life.
pub fn life_schema() -> FieldSchema {
    let mut s = FieldSchema::new();
    for (n, l, lod) in [
        ("position", "Position", Lod::Minimal),
        ("energy", "Energy", Lod::Minimal),
        ("species", "Species", Lod::Minimal),
        ("genome", "Genome", Lod::Standard),
        ("speed_gene", "Speed gene", Lod::Full),
        ("metabolism_efficiency", "Metabolism efficiency", Lod::Full),
        ("sensing_bias", "Sensing bias", Lod::Full),
        ("age", "Age", Lod::Standard),
        ("generation", "Generation", Lod::Standard),
        ("parent", "Parent", Lod::Standard),
        ("food", "Food", Lod::Minimal),
        ("alive", "Alive", Lod::Minimal),
        ("species_richness", "Species richness", Lod::Standard),
        ("shannon_diversity", "Shannon diversity", Lod::Standard),
        ("mean_energy", "Mean energy", Lod::Standard),
        ("total_born", "Total born", Lod::Full),
        ("total_died", "Total died", Lod::Full),
        ("max_generation", "Max generation", Lod::Full),
        ("mutation_rate", "Mutation rate", Lod::Full),
    ] {
        s.register(FieldDescriptor::new(n, l, lod));
    }
    s
}

/// Adapta la vista real del mundo Life + la observación de sesión a un [`SnapshotView`]:
/// una entidad por organismo vivo (id del lab), una por recurso y la entidad población con las
/// métricas del lab. Los organismos muertos no aparecen (no hay entidades fantasma).
pub fn life_snapshot_view(world: &LifeWorldView, obs: &Observation) -> SnapshotView {
    let tick = world.tick.as_u64();
    let epoch = life_epoch(obs);
    let mut entities = Vec::new();
    for o in &world.organisms {
        let mut fields = vec![
            Field::new(
                "position",
                FieldValue::Vector(vec![o.x as f64, o.y as f64]),
                tick,
            )
            .unit("cell"),
            Field::new("energy", FieldValue::Number(o.energy), tick).unit("energy"),
            Field::new("species", FieldValue::Integer(o.species as i64), tick)
                .provenance(Provenance::Derived),
            Field::new(
                "genome",
                FieldValue::Vector(vec![
                    o.genome.speed_gene,
                    o.genome.metabolism_efficiency,
                    o.genome.sensing_bias,
                ]),
                tick,
            ),
            Field::new("speed_gene", FieldValue::Number(o.genome.speed_gene), tick),
            Field::new(
                "metabolism_efficiency",
                FieldValue::Number(o.genome.metabolism_efficiency),
                tick,
            ),
            Field::new(
                "sensing_bias",
                FieldValue::Number(o.genome.sensing_bias),
                tick,
            ),
            Field::new("age", FieldValue::Integer(o.age as i64), tick).unit("tick"),
            Field::new("generation", FieldValue::Integer(o.generation as i64), tick),
        ];
        if let Some(p) = o.parent {
            fields.push(Field::new("parent", FieldValue::Integer(p as i64), tick));
        }
        entities.push(EntityView {
            entity: EntityRef::new(o.id, epoch),
            kind: "organism".into(),
            fields,
        });
    }
    for r in &world.resources {
        entities.push(EntityView {
            entity: EntityRef::new(life_resource_id(world, r.x, r.y), epoch),
            kind: "resource".into(),
            fields: vec![
                Field::new(
                    "position",
                    FieldValue::Vector(vec![r.x as f64, r.y as f64]),
                    tick,
                )
                .unit("cell"),
                Field::new("food", FieldValue::Number(r.food), tick).unit("energy"),
            ],
        });
    }
    let m = &world.metrics;
    entities.push(EntityView {
        entity: EntityRef::new(LIFE_POPULATION_ID, epoch),
        kind: "population".into(),
        fields: vec![
            Field::new("alive", FieldValue::Integer(m.alive as i64), tick),
            Field::new(
                "species_richness",
                FieldValue::Integer(m.species_richness as i64),
                tick,
            ),
            Field::new(
                "shannon_diversity",
                FieldValue::Number(m.shannon_diversity),
                tick,
            ),
            Field::new("mean_energy", FieldValue::Number(m.mean_energy), tick),
            Field::new("total_born", FieldValue::Integer(m.total_born as i64), tick),
            Field::new("total_died", FieldValue::Integer(m.total_died as i64), tick),
            Field::new(
                "max_generation",
                FieldValue::Integer(m.max_generation as i64),
                tick,
            ),
            Field::new(
                "mutation_rate",
                FieldValue::Number(world.mutation_rate),
                tick,
            ),
        ]
        .into_iter()
        .map(|f| f.provenance(Provenance::Derived))
        .collect(),
    });
    SnapshotView {
        revision: obs.revision,
        tick,
        entities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    use crate::inspector::{Inspector, InspectorStatus, VanishReason};
    use crate::params::{
        ParamChange, ParamCommand, ParamControlPort, ParamOutcome, ParamRejectReason, ParamRequest,
    };
    use crate::shell::StudioShell;

    const SEED: u64 = 7;
    const N: u64 = 12;

    fn headless(seed: u64, n: u64) -> LifeRollout {
        let mut r = LifeRollout::new(life_demo_config(), MasterSeed::new(seed)).unwrap();
        for _ in 0..n {
            if r.step().is_none() {
                break;
            }
        }
        r
    }

    struct Ui {
        shell: StudioShell,
        probe: LifeProbe,
        next: u64,
    }

    impl Ui {
        fn new(seed: u64) -> Self {
            let (entry, probe) =
                life_lab_entry_with_probe(life_demo_config(), MasterSeed::new(seed)).unwrap();
            let mut shell = StudioShell::new(vec![entry], 640.0, 480.0).unwrap();
            shell.select_lab(LIFE_LAB_ID).unwrap();
            Self {
                shell,
                probe,
                next: 1,
            }
        }
        fn obs(&self) -> Observation {
            self.shell.observation_port().unwrap().observe()
        }
        fn send(&mut self, action: ControlAction) -> crate::control::Ack {
            let port = self.shell.control_port().unwrap().unwrap();
            let session = self.obs().session;
            let id = CommandId(self.next);
            self.next += 1;
            let ack = port.submit(ControlCommand {
                id,
                session,
                action,
            });
            self.shell.sync_active();
            ack
        }
        fn set_param(&mut self, id: &str, value: ParamValue) -> ParamOutcome {
            let cid = CommandId(self.next);
            self.next += 1;
            let session = self.obs().session;
            self.shell
                .param_port()
                .unwrap()
                .submit_params(ParamCommand {
                    id: cid,
                    session,
                    request: ParamRequest::Set(vec![ParamChange {
                        id: ParamId::new(id),
                        value,
                    }]),
                })
                .outcome
        }
        fn snapshot_view(&self) -> SnapshotView {
            life_snapshot_view(&self.probe.snapshot().unwrap().view, &self.obs())
        }
    }

    fn nodes(scene: &Scene2D) -> Vec<&SceneNode> {
        scene.layers().iter().flat_map(|l| l.nodes()).collect()
    }

    fn organism_node_ids(scene: &Scene2D) -> Vec<u64> {
        let mut v: Vec<u64> = nodes(scene)
            .iter()
            .filter(|n| {
                n.is_selectable()
                    && n.entity_id()
                        .is_some_and(|e| e.as_u64() < LIFE_RESOURCE_BASE)
            })
            .filter_map(|n| n.entity_id().map(|e| e.as_u64()))
            .collect();
        v.sort_unstable();
        v
    }

    fn by_name(v: &crate::inspector::InspectorView, n: &str) -> FieldValue {
        v.rows.iter().find(|r| r.name == n).unwrap().value.clone()
    }

    /// Nodos de futuro FMC de la capa común (rechazado 104, elegido 106, terminal 107).
    fn has_future_nodes(scene: &Scene2D) -> bool {
        scene
            .layers()
            .iter()
            .filter(|l| l.id() != LayerId::Labels) // la leyenda vive en Labels
            .flat_map(|l| l.nodes())
            .any(|n| matches!(n.state_code(), Some(104 | 106 | 107)))
    }

    #[test]
    fn demo_run_has_births_and_deaths() {
        let r = headless(SEED, N);
        let births: usize = r.records().iter().map(|x| x.births).sum();
        let deaths: usize = r.records().iter().map(|x| x.deaths).sum();
        assert!(births > 0, "births {births}");
        assert!(deaths > 0, "deaths {deaths}");
    }

    #[test]
    fn ac1_scene_projects_organisms_resources_and_selection_shows_genome_energy() {
        let mut ui = Ui::new(SEED);
        for _ in 0..N {
            assert_eq!(ui.send(ControlAction::Step).outcome, AckOutcome::Applied);
        }
        let snap = ui.probe.snapshot().unwrap();
        let v = &snap.view;
        assert_eq!(snap.scene, project_life_scene(v));
        assert!(!v.organisms.is_empty() && !v.resources.is_empty());
        // un nodo seleccionable por organismo vivo y por celda con alimento
        let mut alive: Vec<u64> = v.organisms.iter().map(|o| o.id).collect();
        alive.sort_unstable();
        assert_eq!(organism_node_ids(&snap.scene), alive);
        let res_nodes = nodes(&snap.scene)
            .iter()
            .filter(|n| {
                n.entity_id()
                    .is_some_and(|e| e.as_u64() >= LIFE_RESOURCE_BASE)
            })
            .count();
        assert_eq!(res_nodes, v.resources.len());
        // picking -> inspector con genoma y energía
        let o = v.organisms[0].clone();
        let hit = snap
            .scene
            .pick_at_world(pt(o.x as f64 + 0.5, o.y as f64 + 0.5), 0.05)
            .unwrap()
            .unwrap();
        let sel = hit.entity_id.expect("organism hit");
        let obs = ui.obs();
        let mut i = Inspector::new(life_schema(), Lod::Full, 0);
        // el pick puede caer sobre otro organismo en la misma celda: inspeccionar el golpeado
        let target = v.organisms.iter().find(|x| x.id == sel.as_u64()).unwrap();
        i.select(EntityRef::new(sel.as_u64(), life_epoch(&obs)));
        let iv = i.inspect(&ui.snapshot_view());
        assert_eq!(iv.status, InspectorStatus::Live);
        assert_eq!(iv.kind.as_deref(), Some("organism"));
        assert_eq!(by_name(&iv, "energy"), FieldValue::Number(target.energy));
        assert_eq!(
            by_name(&iv, "genome"),
            FieldValue::Vector(vec![
                target.genome.speed_gene,
                target.genome.metabolism_efficiency,
                target.genome.sensing_bias
            ])
        );
        assert_eq!(
            by_name(&iv, "species"),
            FieldValue::Integer(target.species as i64)
        );
        // recurso accesible por el mismo inspector
        let r = v.resources[0];
        i.select(EntityRef::new(
            life_resource_id(v, r.x, r.y),
            life_epoch(&obs),
        ));
        let rv = i.inspect(&ui.snapshot_view());
        assert_eq!(rv.kind.as_deref(), Some("resource"));
        assert_eq!(by_name(&rv, "food"), FieldValue::Number(r.food));
        // LOD mínimo oculta el genoma
        i.select(EntityRef::new(target.id, life_epoch(&obs)));
        i.set_lod(Lod::Minimal);
        let mv = i.inspect(&ui.snapshot_view());
        assert!(mv.rows.iter().all(|r| r.name != "genome"));
        assert!(mv.rows.iter().any(|r| r.name == "energy"));
    }

    #[test]
    fn ac1_genealogy_edges_link_alive_parent_and_child() {
        let ui_steps = headless(SEED, N);
        let view = ui_steps.view(None);
        let scene = project_life_scene(&view);
        let edges = scene
            .layers()
            .iter()
            .find(|l| l.id() == GENEALOGY_LAYER)
            .map(|l| l.nodes().len())
            .unwrap_or(0);
        let expected = view
            .organisms
            .iter()
            .filter(|o| {
                o.parent
                    .is_some_and(|p| view.organisms.iter().any(|x| x.id == p))
            })
            .count();
        assert_eq!(edges, expected);
        assert!(view.organisms.iter().any(|o| o.generation > 0));
    }

    #[test]
    fn ac2_births_and_deaths_leave_no_ghosts() {
        let mut ui = Ui::new(SEED);
        let mut ever: std::collections::BTreeSet<u64> = Default::default();
        let mut seen_birth = false;
        let mut seen_death = false;
        let mut prev_alive: Vec<u64> = Vec::new();
        for _ in 0..N {
            ui.send(ControlAction::Step);
            let snap = ui.probe.snapshot().unwrap();
            let mut alive: Vec<u64> = snap.view.organisms.iter().map(|o| o.id).collect();
            alive.sort_unstable();
            // escena y snapshot del inspector == vivos del mundo real
            assert_eq!(organism_node_ids(&snap.scene), alive);
            let sv = ui.snapshot_view();
            let mut inspected: Vec<u64> = sv
                .entities
                .iter()
                .filter(|e| e.kind == "organism")
                .map(|e| e.entity.id)
                .collect();
            inspected.sort_unstable();
            assert_eq!(inspected, alive);
            assert_eq!(alive.len(), snap.view.metrics.alive);
            if alive.iter().any(|i| !prev_alive.contains(i)) {
                seen_birth = true;
            }
            for id in &prev_alive {
                if !alive.contains(id) {
                    seen_death = true;
                    // muerto: ni en escena ni inspeccionable
                    assert!(!organism_node_ids(&snap.scene).contains(id));
                    let mut i = Inspector::new(life_schema(), Lod::Full, 0);
                    i.select(EntityRef::new(*id, life_epoch(&ui.obs())));
                    assert!(matches!(
                        i.inspect(&sv).status,
                        InspectorStatus::Vanished(_)
                    ));
                }
            }
            // ids monótonos: un id muerto nunca reaparece
            for id in &alive {
                if !prev_alive.contains(id) {
                    assert!(!ever.contains(id), "id {id} reused");
                }
            }
            ever.extend(alive.iter().copied());
            prev_alive = alive;
        }
        assert!(seen_birth && seen_death);
        // Reset: la selección previa queda reciclada y la sonda la descarta
        let old = EntityRef::new(prev_alive[0], life_epoch(&ui.obs()));
        ui.probe.select(Some(old));
        ui.send(ControlAction::Step);
        assert_eq!(ui.probe.snapshot().unwrap().view.selected, Some(old.id));
        ui.send(ControlAction::Reset);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.view.selected, None);
        assert!(!has_future_nodes(&snap.scene));
        let mut i = Inspector::new(life_schema(), Lod::Full, 0);
        i.select(old);
        assert!(matches!(
            i.inspect(&ui.snapshot_view()).status,
            InspectorStatus::Vanished(VanishReason::Recycled { .. })
        ));
    }

    #[test]
    fn ac2_authorized_mutation_rate_change_is_traced_and_rejections_are_not() {
        let mut ui = Ui::new(SEED);
        assert_eq!(ui.probe.snapshot().unwrap().view.mutation_rate, 0.3);
        ui.send(ControlAction::Step);
        let before = ui.probe.snapshot().unwrap().trace_hash;
        let out = ui.set_param(PARAM_MUTATION_RATE, ParamValue::Float(0.9));
        assert!(matches!(out, ParamOutcome::Queued { .. }), "{out:?}");
        ui.send(ControlAction::Step);
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.view.mutation_rate, 0.9);
        assert_eq!(
            snap.mutation_changes,
            vec![MutationChange {
                at_step: 1,
                from: 0.3,
                to: 0.9
            }]
        );
        assert_eq!(snap.records[0].mutation_rate, 0.3);
        assert_eq!(snap.records[1].mutation_rate, 0.9);
        assert_ne!(snap.trace_hash, before);
        // la traza difiere de un rollout sin el cambio (el cambio forma parte del hash)
        assert_ne!(snap.trace_hash, headless(SEED, 3).trace_hash());
        // rechazos: no entran en la traza
        let n = snap.mutation_changes.len();
        for (id, val, why) in [
            (
                PARAM_MAX_POPULATION,
                ParamValue::Int(10),
                ParamRejectReason::Immutable(ParamId::new(PARAM_MAX_POPULATION)),
            ),
            (
                PARAM_MUTATION_RATE,
                ParamValue::Float(1.5),
                ParamRejectReason::OutOfRange(ParamId::new(PARAM_MUTATION_RATE)),
            ),
            (
                PARAM_MUTATION_RATE,
                ParamValue::Int(1),
                ParamRejectReason::WrongType(ParamId::new(PARAM_MUTATION_RATE)),
            ),
            (
                "life.nope",
                ParamValue::Float(0.1),
                ParamRejectReason::UnknownParam(ParamId::new("life.nope")),
            ),
        ] {
            assert_eq!(
                ui.set_param(id, val),
                ParamOutcome::Rejected { reason: why },
                "{id}"
            );
        }
        ui.send(ControlAction::Step);
        let after = ui.probe.snapshot().unwrap();
        assert_eq!(after.mutation_changes.len(), n);
        assert_eq!(after.view.mutation_rate, 0.9);
        assert_eq!(ui.obs().param_rejection_count, 4);
    }

    #[test]
    fn ac3_fmc_layers_only_for_selected_sample() {
        let mut ui = Ui::new(SEED);
        for _ in 0..3 {
            ui.send(ControlAction::Step);
        }
        let snap = ui.probe.snapshot().unwrap();
        assert!(snap.view.futures.is_empty());
        assert!(!has_future_nodes(&snap.scene));
        let id = snap.view.organisms[0].id;
        ui.probe
            .select(Some(EntityRef::new(id, life_epoch(&ui.obs()))));
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(snap.view.selected, Some(id), "selected organism alive");
        assert!(!snap.view.futures.is_empty());
        assert!(has_future_nodes(&snap.scene));
        assert_eq!(snap.view.futures.iter().filter(|f| f.chosen).count(), 1);
        ui.probe.select(None);
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert!(snap.view.futures.is_empty());
        assert!(!has_future_nodes(&snap.scene));
    }

    #[test]
    fn ac3_population_metrics_come_from_lab_source() {
        let mut ui = Ui::new(SEED);
        for _ in 0..N {
            ui.send(ControlAction::Step);
        }
        let snap = ui.probe.snapshot().unwrap();
        let expected = headless(SEED, N);
        let global = expected.world().privileged_global_view();
        assert_eq!(
            snap.view.metrics,
            PopulationMetrics::from_view(&global).unwrap()
        );
        let sv = ui.snapshot_view();
        let pop = sv
            .entities
            .iter()
            .find(|e| e.entity.id == LIFE_POPULATION_ID)
            .unwrap();
        let f = |n: &str| {
            pop.fields
                .iter()
                .find(|f| f.name == n)
                .unwrap()
                .value
                .clone()
        };
        assert_eq!(
            f("alive"),
            FieldValue::Integer(snap.view.metrics.alive as i64)
        );
        assert_eq!(
            f("shannon_diversity"),
            FieldValue::Number(snap.view.metrics.shannon_diversity)
        );
        assert_eq!(
            f("total_born"),
            FieldValue::Integer(snap.view.metrics.total_born as i64)
        );
    }

    #[test]
    fn ac3_studio_equals_headless_per_step() {
        let expected = headless(SEED, N);
        let mut ui = Ui::new(SEED);
        for i in 0..N as usize {
            ui.send(ControlAction::Step);
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.records, expected.records()[..=i]);
            assert_eq!(snap.trace_hash, expected.records()[i].trace_hash);
        }
        assert_eq!(headless(SEED, N).records(), expected.records());
        assert_eq!(headless(SEED, N).trace_hash(), expected.trace_hash());
        assert_ne!(headless(SEED + 1, N).trace_hash(), expected.trace_hash());
    }

    fn strip_comments_and_tests(src: &str) -> String {
        let cut = src.find("#[cfg(test)]").unwrap_or(src.len());
        src[..cut]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn ac3_no_life_dynamics_planner_or_metrics_in_renderer_modules_and_single_engine() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let tokens = [
            "LifeWorld",
            "HomeostaticFmcAgent",
            "PopulationMetrics",
            "gogamen_lab_life",
            "LifeRollout",
            "step_tick",
            ".step(",
        ];
        for f in ["viewport.rs", "camera.rs", "shell.rs"] {
            let text = strip_comments_and_tests(&std::fs::read_to_string(src.join(f)).unwrap());
            for t in tokens {
                assert!(!text.contains(t), "{f} contains `{t}`");
            }
        }
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let text = strip_comments_and_tests(&std::fs::read_to_string(&path).unwrap());
            assert!(!text.contains("impl World for"), "{path:?}");
            assert!(!text.contains("impl Actuator for"), "{path:?}");
            assert!(!text.contains("fn try_reproduce"), "{path:?}");
            assert!(!text.contains("fn shannon"), "{path:?}");
            assert!(!text.contains("ln()"), "{path:?} recomputes diversity");
        }
        let manifest = std::fs::read_to_string(src.join("..").join("Cargo.toml")).unwrap();
        assert!(manifest.contains("gogamen-lab-life = { path = \"../../labs/life\" }"));
        let me = std::fs::read_to_string(src.join("life_lab.rs")).unwrap();
        assert!(me.contains("pub fn project_life_scene(view: &LifeWorldView)"));
        assert!(me.contains("project_decision_overlay("));
        assert!(me.contains("PopulationMetrics::from_view("));
        let lab = std::fs::read_to_string(src.join("../../../labs/life/src/lib.rs")).unwrap();
        assert!(lab.contains("PopulationMetrics") && lab.contains("HomeostaticFmcAgent"));
    }
}
