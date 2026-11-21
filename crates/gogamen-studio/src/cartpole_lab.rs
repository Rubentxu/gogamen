//! CartPole en Studio (GOG-114): proyección 2D del péndulo invertido real, sin duplicar física.
//!
//! - La única física, el entorno y el controlador son los del lab `gogamen-lab-cartpole`
//!   ([`CartPoleEnv`], [`LinearFeedbackController`], el mismo bucle y hash que
//!   `run_cartpole_headless`); aquí no hay dinámica ni controlador propios.
//! - [`CartPoleRollout`] es el lado motor (headless, sin tipos de Studio). Parámetro permitido
//!   (GOG-95): `cartpole.force_limit` (N), tope de la magnitud de fuerza aplicada; sólo cuando
//!   cambia se escribe en la traza, de modo que sin cambios el hash es idéntico al del lab.
//! - [`CartPoleView`] / [`project_cartpole_scene`]: proyector puro. Capas:
//!   * `Environment` = ESTADO (carril, límites de `x`, carro, péndulo);
//!   * `Decisions` = ACCIÓN (flecha de la fuerza aplicada, `F` en N) y estado terminal;
//!   * [`GOAL_LAYER`] = META visual (zona vertical `|theta| <= umbral` y ángulo objetivo 0);
//!   * `Labels` = leyenda y lecturas.
//! - [`CartPoleLabSession`] sólo avanza el rollout en [`LabSession::sync_to`]; el
//!   viewport/render nunca avanza la simulación.
//! - Inspector (GOG-92): [`cartpole_schema`] / [`cartpole_snapshot_view`] con unidades
//!   (m, m/s, rad, rad/s, N).

use std::sync::{Arc, Mutex, MutexGuard};

use gogamen_core::{
    Action, Actuator, DeterminismMetadata, DomainError, EntityId, EnvironmentError,
    EnvironmentObserver, EpistemicKind, MasterSeed, Terminal, Tick, TraceHasher,
};
use gogamen_lab_cartpole::{
    CartPoleConfig, CartPoleEnv, CartPoleKinematics, CartPolePolicy, LinearFeedbackController,
};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
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
pub const CARTPOLE_LAB_ID: &str = "lab.cartpole";
/// Parámetro mutable: tope de la magnitud de fuerza aplicada (N).
pub const PARAM_FORCE_LIMIT: &str = "cartpole.force_limit";
/// Capa de la meta visual.
pub const GOAL_LAYER: LayerId = LayerId::Custom(31);
/// Entidades del inspector.
pub const CART_ID: u64 = 1;
pub const POLE_ID: u64 = 2;
pub const ACTION_ID: u64 = 3;
pub const LIMITS_ID: u64 = 4;
/// `state_code` de nodos (estado/acción/meta) para distinguirlos sin depender del color.
pub const CODE_TRACK: u32 = 301;
pub const CODE_LIMIT: u32 = 302;
pub const CODE_CART: u32 = 303;
pub const CODE_POLE: u32 = 304;
pub const CODE_FORCE: u32 = 310;
pub const CODE_GOAL_ZONE: u32 = 320;
pub const CODE_GOAL_TARGET: u32 = 321;

/// Altura del eje del péndulo sobre el carril (m, geometría de dibujo).
const CART_HEIGHT: f64 = 0.3;
const CART_HALF_WIDTH: f64 = 0.25;
/// Longitud de la flecha de fuerza a `max_force` (m, sólo dibujo).
const ARROW_SCALE: f64 = 0.8;
const ARROW_Y: f64 = -0.12;

/// Configuración demo: perturbación pequeña que el controlador de referencia estabiliza.
pub fn cartpole_demo_config() -> CartPoleConfig {
    CartPoleConfig::default()
        .with_initial_state(CartPoleKinematics::new(0.25, -0.1, 0.12, 0.15).expect("finite"))
}

/// Vista inmutable del mundo CartPole (entrada del proyector y del inspector).
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleView {
    pub kinematics: CartPoleKinematics,
    /// Última fuerza aplicada (N); 0 antes del primer paso.
    pub force: f64,
    /// Tope vigente de |F| (N).
    pub force_limit: f64,
    /// Fuerza máxima del espacio de acciones (N).
    pub max_force: f64,
    /// Límite del carril `|x|` (m).
    pub x_threshold: f64,
    /// Límite angular `|theta|` (rad).
    pub theta_threshold: f64,
    /// Longitud total del péndulo (m) = 2 * half_pole_length.
    pub pole_length: f64,
    pub tick: Tick,
    pub terminal: Terminal,
}

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite cartpole coordinate")
}

/// Proyección pura de la vista a `Scene2D` (y hacia arriba, metros).
pub fn project_cartpole_scene(view: &CartPoleView) -> Scene2D {
    let mut scene = Scene2D::new(
        SceneId::new(114),
        view.tick,
        EpistemicKind::Observed,
        CoordinateSpace::meters(),
    );
    for layer in [
        LayerId::Environment,
        GOAL_LAYER,
        LayerId::Decisions,
        LayerId::Labels,
    ] {
        scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }
    let mut next_id = 1u64;
    let mut id = || {
        let n = next_id;
        next_id += 1;
        SceneNodeId::new(n)
    };
    let mut add = |scene: &mut Scene2D,
                   layer: LayerId,
                   entity: Option<u64>,
                   kind: EpistemicKind,
                   prim: ScenePrimitive,
                   code: Option<u32>| {
        let mut n = SceneNode::new(
            id(),
            entity.map(EntityId::new),
            kind,
            prim,
            entity.is_some(),
        );
        if let Some(c) = code {
            n = n.with_state_code(c);
        }
        scene.add_node(layer, n).expect("node");
    };
    let k = view.kinematics;
    let xt = view.x_threshold;
    let obs = EpistemicKind::Observed;
    let env = LayerId::Environment;

    // ESTADO: carril, límites, carro, péndulo.
    add(
        &mut scene,
        env,
        None,
        obs,
        ScenePrimitive::polyline(vec![pt(-xt, 0.0), pt(xt, 0.0)], false).expect("track"),
        Some(CODE_TRACK),
    );
    for s in [-1.0, 1.0] {
        add(
            &mut scene,
            env,
            None,
            obs,
            ScenePrimitive::polyline(vec![pt(s * xt, 0.0), pt(s * xt, 0.6)], false).expect("limit"),
            Some(CODE_LIMIT),
        );
    }
    add(
        &mut scene,
        env,
        Some(CART_ID),
        obs,
        ScenePrimitive::rect(
            pt(k.x - CART_HALF_WIDTH, 0.0),
            2.0 * CART_HALF_WIDTH,
            CART_HEIGHT,
        )
        .expect("cart"),
        Some(CODE_CART),
    );
    let tip = pt(
        k.x + view.pole_length * k.theta.sin(),
        CART_HEIGHT + view.pole_length * k.theta.cos(),
    );
    add(
        &mut scene,
        env,
        Some(POLE_ID),
        obs,
        ScenePrimitive::polyline(vec![pt(k.x, CART_HEIGHT), tip], false).expect("pole"),
        Some(CODE_POLE),
    );

    // META visual: zona vertical admisible y ángulo objetivo (theta = 0).
    let reach = view.pole_length * 1.15;
    let tt = view.theta_threshold;
    let inf = EpistemicKind::Inferred;
    add(
        &mut scene,
        GOAL_LAYER,
        None,
        inf,
        ScenePrimitive::polyline(
            vec![
                pt(k.x, CART_HEIGHT),
                pt(k.x - reach * tt.sin(), CART_HEIGHT + reach * tt.cos()),
                pt(k.x + reach * tt.sin(), CART_HEIGHT + reach * tt.cos()),
            ],
            true,
        )
        .expect("goal zone"),
        Some(CODE_GOAL_ZONE),
    );
    add(
        &mut scene,
        GOAL_LAYER,
        None,
        inf,
        ScenePrimitive::polyline(
            vec![pt(k.x, CART_HEIGHT), pt(k.x, CART_HEIGHT + reach)],
            false,
        )
        .expect("goal target"),
        Some(CODE_GOAL_TARGET),
    );

    // ACCIÓN: flecha de la fuerza aplicada (longitud proporcional a F / max_force).
    let dec = LayerId::Decisions;
    let f = view.force;
    if f == 0.0 {
        add(
            &mut scene,
            dec,
            Some(ACTION_ID),
            obs,
            ScenePrimitive::circle(pt(k.x, ARROW_Y), 0.03).expect("zero force"),
            Some(CODE_FORCE),
        );
    } else {
        let len = ARROW_SCALE * f / view.max_force;
        let end = k.x + len;
        let dir = len.signum();
        add(
            &mut scene,
            dec,
            Some(ACTION_ID),
            obs,
            ScenePrimitive::polyline(
                vec![
                    pt(k.x, ARROW_Y),
                    pt(end, ARROW_Y),
                    pt(end - dir * 0.08, ARROW_Y + 0.05),
                    pt(end, ARROW_Y),
                    pt(end - dir * 0.08, ARROW_Y - 0.05),
                ],
                false,
            )
            .expect("force arrow"),
            Some(CODE_FORCE),
        );
    }
    if view.terminal.is_terminal() {
        add(
            &mut scene,
            dec,
            None,
            obs,
            ScenePrimitive::label(
                pt(-xt, 1.4),
                format!("TERMINATED {}", terminal_text(view.terminal)),
            ),
            None,
        );
    }

    // Leyenda y lecturas (estado vs acción vs meta).
    let lab = LayerId::Labels;
    let lines = [
        format!(
            "STATE: x={:.3} m  theta={:.3} rad  tick={}",
            k.x,
            k.theta,
            view.tick.as_u64()
        ),
        format!(
            "ACTION: F={:+.2} N (limit {:.2} N)",
            view.force, view.force_limit
        ),
        format!("GOAL: upright |theta| <= {:.3} rad", tt),
        format!("LIMITS: |x| <= {:.2} m", xt),
        "LEGEND: Environment=state, Decisions=action, goal layer=target".to_string(),
    ];
    for (i, text) in lines.into_iter().enumerate() {
        add(
            &mut scene,
            lab,
            None,
            obs,
            ScenePrimitive::label(pt(-xt, -0.35 - 0.2 * i as f64), text),
            None,
        );
    }
    scene
}

fn terminal_text(t: Terminal) -> String {
    format!("{t:?}")
}

/// Registro lógico de un paso.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRecord {
    /// Tick lógico del mundo tras el paso (1-based).
    pub tick: u64,
    /// Fuerza efectivamente aplicada (N).
    pub force: f64,
    pub kinematics: CartPoleKinematics,
    pub terminal: Terminal,
    /// Hash de traza acumulado (comparable con `run_cartpole_headless` si no hubo cambios).
    pub trace_hash: u64,
}

/// Cambio autorizado del tope de fuerza.
#[derive(Debug, Clone, PartialEq)]
pub struct ForceLimitChange {
    pub at_step: u64,
    pub from: f64,
    pub to: f64,
}

/// Motor headless: entorno real + controlador lineal del lab, mismo bucle y hash que
/// `run_cartpole_headless`.
pub struct CartPoleRollout {
    config: CartPoleConfig,
    seed: MasterSeed,
    gains: LinearFeedbackController,
    env: CartPoleEnv,
    hasher: TraceHasher,
    force_limit: f64,
    last_force: f64,
    changes: Vec<ForceLimitChange>,
    records: Vec<StepRecord>,
}

impl CartPoleRollout {
    pub fn new(config: CartPoleConfig, seed: MasterSeed) -> Result<Self, DomainError> {
        Self::with_gains(config, seed, LinearFeedbackController::default())
    }

    pub fn with_gains(
        config: CartPoleConfig,
        seed: MasterSeed,
        gains: LinearFeedbackController,
    ) -> Result<Self, DomainError> {
        let env = CartPoleEnv::new(config.clone())?;
        Ok(Self {
            force_limit: config.max_force,
            config,
            seed,
            gains,
            env,
            hasher: TraceHasher::with_metadata(&DeterminismMetadata::current(seed)),
            last_force: 0.0,
            changes: Vec::new(),
            records: Vec::new(),
        })
    }

    /// Restaura el episodio inicial sembrado.
    pub fn reset(&mut self) {
        *self = Self::with_gains(self.config.clone(), self.seed, self.gains)
            .expect("config already validated");
    }

    pub fn is_terminal(&self) -> bool {
        self.env.terminal().is_terminal()
    }

    pub fn steps(&self) -> u64 {
        self.records.len() as u64
    }

    pub fn records(&self) -> &[StepRecord] {
        &self.records
    }

    pub fn trace_hash(&self) -> u64 {
        self.hasher.finish()
    }

    pub fn force_limit(&self) -> f64 {
        self.force_limit
    }

    pub fn force_limit_changes(&self) -> &[ForceLimitChange] {
        &self.changes
    }

    /// Aplica un nuevo tope de |F| desde el siguiente paso; queda en el log y en el hash.
    /// Sin efecto (ni traza) si no cambia.
    pub fn set_force_limit(&mut self, limit: f64) -> Result<(), DomainError> {
        if !limit.is_finite() || limit <= 0.0 || limit > self.config.max_force {
            return Err(DomainError::NonFiniteValue {
                field: "cartpole.force_limit",
                value: limit,
            });
        }
        let from = self.force_limit;
        if limit == from {
            return Ok(());
        }
        self.force_limit = limit;
        self.changes.push(ForceLimitChange {
            at_step: self.steps(),
            from,
            to: limit,
        });
        self.hasher.write_str("force_limit");
        self.hasher.write_u64(self.steps());
        self.hasher.write_f64(limit);
        Ok(())
    }

    pub fn view(&self) -> CartPoleView {
        let c = self.env.config();
        CartPoleView {
            kinematics: self.env.kinematics(),
            force: self.last_force,
            force_limit: self.force_limit,
            max_force: c.max_force,
            x_threshold: c.x_threshold,
            theta_threshold: c.theta_threshold_radians,
            pole_length: 2.0 * c.half_pole_length,
            tick: self.env.tick(),
            terminal: self.env.terminal(),
        }
    }

    /// Un paso lógico. `Ok(None)` si el episodio ya terminó.
    pub fn step(&mut self) -> Result<Option<StepRecord>, DomainError> {
        if self.is_terminal() {
            return Ok(None);
        }
        let obs = EnvironmentObserver::observe(&self.env).map_err(|_| {
            DomainError::InvalidTransition {
                reason: "failed to observe cartpole environment",
            }
        })?;
        let space = Actuator::action_space(&self.env).clone();
        let proposed = self.gains.select_action(&obs, &space)?;
        let force = proposed
            .as_continuous()
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0)
            .clamp(-self.force_limit, self.force_limit);
        let action = Action::continuous_in_space(vec![force], &space)?;
        let outcome = self.env.apply(&action).map_err(|e| match e {
            EnvironmentError::AlreadyTerminal => DomainError::AlreadyTerminal,
            EnvironmentError::InvalidAction(d) => d,
            _ => DomainError::InvalidTransition {
                reason: "failed to apply cartpole action",
            },
        })?;
        let h = &mut self.hasher;
        h.write_tick(self.env.tick());
        h.write_observation(&obs);
        h.write_f64(force);
        h.write_observation(outcome.observation());
        h.write_terminal(outcome.terminal());
        self.last_force = force;
        let rec = StepRecord {
            tick: self.env.tick().as_u64(),
            force,
            kinematics: self.env.kinematics(),
            terminal: outcome.terminal(),
            trace_hash: h.finish(),
        };
        self.records.push(rec.clone());
        Ok(Some(rec))
    }
}

/// Esquema de parámetros: tope de fuerza (N), mutable.
pub fn cartpole_param_schema(rollout: &CartPoleRollout) -> ParamSchema {
    ParamSchema::new(vec![ParamDescriptor {
        id: ParamId::new(PARAM_FORCE_LIMIT),
        label: "Force limit".into(),
        kind: ParamKind::Float {
            min: 0.5,
            max: rollout.config.max_force,
            step: 0.5,
        },
        unit: ParamUnit::Newtons,
        mutability: Mutability::Mutable,
        default: ParamValue::Float(rollout.force_limit()),
    }])
    .expect("cartpole param schema is valid")
}

/// Estado compartido sólo-lectura de una sesión CartPole.
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleSnapshot {
    pub scene: Scene2D,
    pub view: CartPoleView,
    pub records: Vec<StepRecord>,
    pub force_limit_changes: Vec<ForceLimitChange>,
    pub trace_hash: u64,
    pub epoch: u64,
}

/// Sonda de observación (clonable, sólo lectura).
#[derive(Clone)]
pub struct CartPoleProbe {
    shared: Arc<Mutex<Option<CartPoleSnapshot>>>,
}

impl CartPoleProbe {
    fn lock(&self) -> MutexGuard<'_, Option<CartPoleSnapshot>> {
        self.shared.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn snapshot(&self) -> Option<CartPoleSnapshot> {
        self.lock().clone()
    }

    /// Observación combinada: control (GOG-90) + escena proyectada.
    pub fn observe_with(&self, control: Observation) -> CartPoleObservation {
        CartPoleObservation {
            control,
            cartpole: self.snapshot(),
        }
    }
}

/// Observación del lab: la del puerto de control más la escena proyectada.
#[derive(Debug, Clone, PartialEq)]
pub struct CartPoleObservation {
    pub control: Observation,
    pub cartpole: Option<CartPoleSnapshot>,
}

/// Epoch de entidades: sesión (`epoch`, `generation`); cambia con cada `Reset`.
pub fn cartpole_epoch(obs: &Observation) -> u64 {
    (obs.session.epoch << 32) | (obs.session.generation & 0xffff_ffff)
}

/// Sesión CartPole del shell. Sólo avanza el rollout en [`LabSession::sync_to`].
pub struct CartPoleLabSession {
    rollout: CartPoleRollout,
    generation: u64,
    epoch: u64,
    scene: Scene2D,
    probe: CartPoleProbe,
}

impl CartPoleLabSession {
    fn publish(&mut self) {
        let view = self.rollout.view();
        self.scene = project_cartpole_scene(&view);
        *self.probe.lock() = Some(CartPoleSnapshot {
            scene: self.scene.clone(),
            view,
            records: self.rollout.records().to_vec(),
            force_limit_changes: self.rollout.force_limit_changes().to_vec(),
            trace_hash: self.rollout.trace_hash(),
            epoch: self.epoch,
        });
    }
}

impl LabSession for CartPoleLabSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, observation: &Observation) {
        if observation.session.generation != self.generation {
            self.generation = observation.session.generation;
            self.rollout.reset();
        }
        self.epoch = cartpole_epoch(observation);
        let limit = observation
            .params
            .iter()
            .find(|(id, _)| id.0 == PARAM_FORCE_LIMIT)
            .and_then(|(_, v)| match v {
                ParamValue::Float(x) => Some(*x),
                _ => None,
            });
        if let Some(l) = limit {
            let _ = self.rollout.set_force_limit(l);
        }
        while self.rollout.steps() < observation.tick.as_u64() && !self.rollout.is_terminal() {
            if !matches!(self.rollout.step(), Ok(Some(_))) {
                break;
            }
        }
        self.publish();
    }
}

/// Entrada de registro con sonda y esquema de parámetros (controlador de referencia del lab).
pub fn cartpole_lab_entry_with_probe(
    config: CartPoleConfig,
    seed: MasterSeed,
) -> Result<(LabEntry, CartPoleProbe), DomainError> {
    cartpole_lab_entry_with_gains(config, seed, LinearFeedbackController::default())
}

/// Como [`cartpole_lab_entry_with_probe`] con ganancias explícitas del controlador del lab.
pub fn cartpole_lab_entry_with_gains(
    config: CartPoleConfig,
    seed: MasterSeed,
    gains: LinearFeedbackController,
) -> Result<(LabEntry, CartPoleProbe), DomainError> {
    let schema = cartpole_param_schema(&CartPoleRollout::with_gains(config.clone(), seed, gains)?);
    let probe = CartPoleProbe {
        shared: Arc::new(Mutex::new(None)),
    };
    let p = probe.clone();
    let entry = LabEntry::new(
        ExtensionId::new(CARTPOLE_LAB_ID),
        "CartPole",
        Arc::new(move || {
            let rollout =
                CartPoleRollout::with_gains(config.clone(), seed, gains).expect("validated config");
            let scene = project_cartpole_scene(&rollout.view());
            let mut s = CartPoleLabSession {
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

/// Entrada de registro (sin sonda).
pub fn cartpole_lab_entry(
    config: CartPoleConfig,
    seed: MasterSeed,
) -> Result<LabEntry, DomainError> {
    cartpole_lab_entry_with_probe(config, seed).map(|(e, _)| e)
}

// ---------------------------------------------------------------------------------------------
// Cableado al inspector (GOG-92)
// ---------------------------------------------------------------------------------------------

/// Descriptores del lab CartPole.
pub fn cartpole_schema() -> FieldSchema {
    let mut s = FieldSchema::new();
    for (n, l, lod) in [
        ("position", "Position", Lod::Minimal),
        ("velocity", "Velocity", Lod::Standard),
        ("angle", "Angle", Lod::Minimal),
        ("angular_velocity", "Angular velocity", Lod::Standard),
        ("force", "Applied force", Lod::Minimal),
        ("force_limit", "Force limit", Lod::Standard),
        ("x_limit", "Track limit", Lod::Standard),
        ("angle_limit", "Angle limit", Lod::Standard),
        ("terminal", "Terminal", Lod::Minimal),
    ] {
        s.register(FieldDescriptor::new(n, l, lod));
    }
    s
}

/// Adapta la vista real + la observación de sesión a un [`SnapshotView`]: `cart`, `pole`,
/// `action` (fuerza aplicada) y `limits`.
pub fn cartpole_snapshot_view(view: &CartPoleView, obs: &Observation) -> SnapshotView {
    let tick = view.tick.as_u64();
    let epoch = cartpole_epoch(obs);
    let k = view.kinematics;
    let ent = |id: u64, kind: &str, fields: Vec<Field>| EntityView {
        entity: EntityRef::new(id, epoch),
        kind: kind.into(),
        fields,
    };
    SnapshotView {
        revision: obs.revision,
        tick,
        entities: vec![
            ent(
                CART_ID,
                "cart",
                vec![
                    Field::new("position", FieldValue::Number(k.x), tick).unit("m"),
                    Field::new("velocity", FieldValue::Number(k.x_dot), tick).unit("m/s"),
                ],
            ),
            ent(
                POLE_ID,
                "pole",
                vec![
                    Field::new("angle", FieldValue::Number(k.theta), tick).unit("rad"),
                    Field::new("angular_velocity", FieldValue::Number(k.theta_dot), tick)
                        .unit("rad/s"),
                ],
            ),
            ent(
                ACTION_ID,
                "action",
                vec![
                    Field::new("force", FieldValue::Number(view.force), tick).unit("N"),
                    Field::new("force_limit", FieldValue::Number(view.force_limit), tick)
                        .unit("N")
                        .provenance(Provenance::Assumed),
                ],
            ),
            ent(
                LIMITS_ID,
                "limits",
                vec![
                    Field::new("x_limit", FieldValue::Number(view.x_threshold), tick)
                        .unit("m")
                        .provenance(Provenance::Assumed),
                    Field::new(
                        "angle_limit",
                        FieldValue::Number(view.theta_threshold),
                        tick,
                    )
                    .unit("rad")
                    .provenance(Provenance::Assumed),
                    Field::new(
                        "terminal",
                        FieldValue::Text(terminal_text(view.terminal)),
                        tick,
                    )
                    .provenance(Provenance::Derived),
                ],
            ),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{
        AckOutcome, CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort,
    };
    use crate::inspector::{Inspector, InspectorStatus};
    use crate::params::{
        ParamChange, ParamCommand, ParamControlPort, ParamOutcome, ParamRejectReason, ParamRequest,
    };
    use crate::shell::StudioShell;
    use gogamen_lab_cartpole::run_cartpole_headless;

    const SEED: u64 = 42;
    const N: u64 = 40;

    fn zero_gains() -> LinearFeedbackController {
        LinearFeedbackController::new(0.0, 0.0, 0.0, 0.0).unwrap()
    }

    fn falling_cfg() -> CartPoleConfig {
        CartPoleConfig::default()
            .with_initial_state(CartPoleKinematics::new(0.0, 0.0, 0.05, 0.0).unwrap())
    }

    fn headless(
        config: CartPoleConfig,
        gains: LinearFeedbackController,
        n: u64,
    ) -> Vec<StepRecord> {
        let mut r = CartPoleRollout::with_gains(config, MasterSeed::new(SEED), gains).unwrap();
        for _ in 0..n {
            if r.step().unwrap().is_none() {
                break;
            }
        }
        r.records().to_vec()
    }

    struct Ui {
        shell: StudioShell,
        probe: CartPoleProbe,
        next: u64,
    }

    impl Ui {
        fn new(config: CartPoleConfig, gains: LinearFeedbackController) -> Self {
            let (entry, probe) =
                cartpole_lab_entry_with_gains(config, MasterSeed::new(SEED), gains).unwrap();
            let mut shell = StudioShell::new(vec![entry], 640.0, 480.0).unwrap();
            shell.select_lab(CARTPOLE_LAB_ID).unwrap();
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
        fn svg(&self) -> String {
            self.shell.render_svg().unwrap()
        }
    }

    fn layer(scene: &Scene2D, id: LayerId) -> &SceneLayer {
        scene.layers().iter().find(|l| l.id() == id).unwrap()
    }

    fn pts(n: &SceneNode) -> Vec<(f64, f64)> {
        match n.primitive() {
            ScenePrimitive::Polyline { points, .. } => {
                points.iter().map(|p| (p.x(), p.y())).collect()
            }
            other => panic!("not a polyline: {other:?}"),
        }
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    /// Snapshot fijo (valores escritos a mano, no salidos del simulador).
    fn fixture_view() -> CartPoleView {
        CartPoleView {
            kinematics: CartPoleKinematics::new(0.5, 0.1, 0.1, -0.2).unwrap(),
            force: 5.0,
            force_limit: 10.0,
            max_force: 10.0,
            x_threshold: 2.4,
            theta_threshold: 0.2,
            pole_length: 1.0,
            tick: Tick::new(7),
            terminal: Terminal::Ongoing,
        }
    }

    #[test]
    fn ac1_scene_values_match_fixture_snapshot() {
        let scene = project_cartpole_scene(&fixture_view());
        let env = layer(&scene, LayerId::Environment).nodes();
        // carril y límites de x
        assert_eq!(pts(&env[0]), vec![(-2.4, 0.0), (2.4, 0.0)]);
        assert_eq!(pts(&env[1]), vec![(-2.4, 0.0), (-2.4, 0.6)]);
        assert_eq!(pts(&env[2]), vec![(2.4, 0.0), (2.4, 0.6)]);
        // carro: posición x = 0.5 m
        match env[3].primitive() {
            ScenePrimitive::Rect { min, width, height } => {
                close(min.x(), 0.25);
                close(min.y(), 0.0);
                close(*width, 0.5);
                close(*height, 0.3);
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(env[3].entity_id(), Some(EntityId::new(CART_ID)));
        // péndulo: base en el carro, punta a theta = 0.1 rad
        let pole = pts(&env[4]);
        close(pole[0].0, 0.5);
        close(pole[0].1, 0.3);
        close(pole[1].0, 0.5 + 0.1f64.sin());
        close(pole[1].1, 0.3 + 0.1f64.cos());
        assert!((pole[1].0 - 0.599_833_416_646_828).abs() < 1e-12);
        assert!((pole[1].1 - 1.295_004_165_278_026).abs() < 1e-12);
        // fuerza 5 N de 10 N: flecha de 0.4 m hacia +x
        let arrow = pts(&layer(&scene, LayerId::Decisions).nodes()[0]);
        close(arrow[0].0, 0.5);
        close(arrow[1].0, 0.9);
        close(arrow[1].1, ARROW_Y);
        // meta: zona |theta| <= 0.2 con alcance 1.15 m
        let goal = layer(&scene, GOAL_LAYER).nodes();
        let zone = pts(&goal[0]);
        close(zone[1].0, 0.5 - 1.15 * 0.2f64.sin());
        close(zone[2].0, 0.5 + 1.15 * 0.2f64.sin());
        let target = pts(&goal[1]);
        close(target[1].0, 0.5);
        close(target[1].1, 0.3 + 1.15);
        // lecturas textuales
        let texts: Vec<String> = layer(&scene, LayerId::Labels)
            .nodes()
            .iter()
            .filter_map(|n| match n.primitive() {
                ScenePrimitive::Label { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect();
        assert!(texts[0].starts_with("STATE: x=0.500 m  theta=0.100 rad  tick=7"));
        assert!(texts[1].starts_with("ACTION: F=+5.00 N (limit 10.00 N)"));
        assert!(texts[2].starts_with("GOAL: upright |theta| <= 0.200 rad"));
        assert!(texts[3].contains("|x| <= 2.40 m"));
        // inspector con unidades
        let obs = Ui::new(falling_cfg(), zero_gains()).obs();
        let snap = cartpole_snapshot_view(&fixture_view(), &obs);
        let mut i = Inspector::new(cartpole_schema(), Lod::Full, 0);
        i.select(snap.entities[0].entity);
        let v = i.inspect(&snap);
        assert_eq!(v.status, InspectorStatus::Live);
        let row = v.rows.iter().find(|r| r.name == "position").unwrap();
        assert_eq!(
            (row.value.clone(), row.unit.as_deref()),
            (FieldValue::Number(0.5), Some("m"))
        );
        i.select(snap.entities[1].entity);
        let v = i.inspect(&snap);
        let row = v.rows.iter().find(|r| r.name == "angle").unwrap();
        assert_eq!(
            (row.value.clone(), row.unit.as_deref()),
            (FieldValue::Number(0.1), Some("rad"))
        );
        i.select(snap.entities[2].entity);
        let v = i.inspect(&snap);
        let row = v.rows.iter().find(|r| r.name == "force").unwrap();
        assert_eq!(
            (row.value.clone(), row.unit.as_deref()),
            (FieldValue::Number(5.0), Some("N"))
        );
    }

    #[test]
    fn ac1_headless_matches_lab_run_cartpole_headless_hash_per_step() {
        let recs = headless(
            cartpole_demo_config(),
            LinearFeedbackController::default(),
            N,
        );
        assert_eq!(recs.len() as u64, N);
        for (i, r) in recs.iter().enumerate() {
            let mut c = LinearFeedbackController::default();
            let ep = run_cartpole_headless(
                cartpole_demo_config(),
                i as u64 + 1,
                MasterSeed::new(SEED),
                &mut c,
            )
            .unwrap();
            assert_eq!(ep.trace_hash, r.trace_hash, "step {}", i + 1);
            assert_eq!(ep.final_kinematics, r.kinematics);
            assert_eq!(ep.terminal, r.terminal);
        }
    }

    #[test]
    fn ac1_studio_step_by_step_equals_headless_incl_terminal() {
        for (config, gains) in [
            (cartpole_demo_config(), LinearFeedbackController::default()),
            (falling_cfg(), zero_gains()),
        ] {
            let expected = headless(config.clone(), gains, 200);
            let mut ui = Ui::new(config, gains);
            for i in 0..expected.len() {
                let ack = ui.send(ControlAction::Step);
                assert_eq!(ack.outcome, AckOutcome::Applied);
                let snap = ui.probe.snapshot().unwrap();
                assert_eq!(snap.records, expected[..=i], "step {}", i + 1);
                assert_eq!(snap.trace_hash, expected[i].trace_hash);
                assert_eq!(snap.scene, project_cartpole_scene(&snap.view));
            }
        }
        // el episodio sin control termina y se marca en la escena
        let expected = headless(falling_cfg(), zero_gains(), 200);
        assert!(expected.last().unwrap().terminal.is_terminal());
        let mut ui = Ui::new(falling_cfg(), zero_gains());
        for _ in 0..expected.len() {
            ui.send(ControlAction::Step);
        }
        assert!(ui.svg().contains("TERMINATED"));
    }

    #[test]
    fn ac2_pause_step_reset_with_real_acks_goal_and_state_vs_action_layers() {
        let mut ui = Ui::new(cartpole_demo_config(), LinearFeedbackController::default());
        let svg0 = ui.svg();
        assert_eq!(svg0, ui.svg(), "render is pure and byte-stable");
        let a = ui.send(ControlAction::Pause);
        assert_eq!(
            (a.outcome, a.revision, a.tick.as_u64()),
            (AckOutcome::Applied, 1, 0)
        );
        assert_eq!(ui.svg(), svg0, "pause does not move the pole");

        let mut prev = svg0.clone();
        for i in 0..5u64 {
            let a = ui.send(ControlAction::Step);
            assert_eq!(a.outcome, AckOutcome::Applied);
            assert_eq!((a.revision, a.tick.as_u64()), (i + 2, i + 1));
            let snap = ui.probe.snapshot().unwrap();
            assert_eq!(snap.view.tick.as_u64(), i + 1, "exactly one physics step");
            let svg = ui.svg();
            assert_ne!(svg, prev, "SVG tracks state");
            prev = svg;
        }
        // estado vs acción vs meta: capas y etiquetas distintas
        let snap = ui.probe.snapshot().unwrap();
        for id in [
            LayerId::Environment,
            LayerId::Decisions,
            GOAL_LAYER,
            LayerId::Labels,
        ] {
            assert!(!layer(&snap.scene, id).nodes().is_empty(), "{id:?}");
        }
        let svg = ui.svg();
        for needle in ["STATE:", "ACTION:", "GOAL:", "LEGEND:"] {
            assert!(svg.contains(needle), "{needle}");
        }
        let goal_codes: Vec<_> = layer(&snap.scene, GOAL_LAYER)
            .nodes()
            .iter()
            .filter_map(|n| n.state_code())
            .collect();
        assert_eq!(goal_codes, vec![CODE_GOAL_ZONE, CODE_GOAL_TARGET]);
        assert_eq!(
            layer(&snap.scene, LayerId::Decisions).nodes()[0].state_code(),
            Some(CODE_FORCE)
        );

        let generation = ui.obs().session.generation;
        let a = ui.send(ControlAction::Reset);
        assert_eq!((a.outcome, a.tick), (AckOutcome::Applied, Tick::ZERO));
        let obs = ui.probe.observe_with(ui.obs());
        assert_eq!(obs.control.session.generation, generation + 1);
        let snap = obs.cartpole.unwrap();
        assert!(snap.records.is_empty());
        assert_eq!(
            snap.trace_hash,
            CartPoleRollout::new(cartpole_demo_config(), MasterSeed::new(SEED))
                .unwrap()
                .trace_hash()
        );
        assert_eq!(ui.svg(), svg0, "reset restores the seeded initial world");
    }

    #[test]
    fn ac2_step_while_running_rejected_and_pump_advances_via_sync() {
        let mut ui = Ui::new(cartpole_demo_config(), LinearFeedbackController::default());
        ui.send(ControlAction::Resume);
        let before = ui.probe.snapshot().unwrap();
        let a = ui.send(ControlAction::Step);
        assert!(matches!(a.outcome, AckOutcome::Rejected { .. }));
        assert_eq!(ui.probe.snapshot().unwrap(), before);
        assert_eq!(ui.shell.engine_handle().unwrap().pump(3), 3);
        ui.shell.sync_active();
        assert_eq!(
            ui.probe.snapshot().unwrap().records,
            headless(
                cartpole_demo_config(),
                LinearFeedbackController::default(),
                3
            )
        );
    }

    #[test]
    fn ac2_authorized_force_limit_is_traced_and_rejections_are_not() {
        let mut ui = Ui::new(cartpole_demo_config(), LinearFeedbackController::default());
        ui.send(ControlAction::Step);
        let out = ui.set_param(PARAM_FORCE_LIMIT, ParamValue::Float(0.5));
        assert!(matches!(out, ParamOutcome::Queued { .. }), "{out:?}");
        ui.send(ControlAction::Step);
        ui.send(ControlAction::Step);
        let snap = ui.probe.snapshot().unwrap();
        assert_eq!(
            snap.force_limit_changes,
            vec![ForceLimitChange {
                at_step: 1,
                from: 10.0,
                to: 0.5
            }]
        );
        assert!(snap.records[1].force.abs() <= 0.5 && snap.records[2].force.abs() <= 0.5);
        assert!(
            snap.records[0].force.abs() > 0.5,
            "first step was unclamped"
        );
        assert_ne!(
            snap.trace_hash,
            headless(
                cartpole_demo_config(),
                LinearFeedbackController::default(),
                3
            )[2]
            .trace_hash
        );
        let n = snap.force_limit_changes.len();
        for (id, val, why) in [
            (
                PARAM_FORCE_LIMIT,
                ParamValue::Float(50.0),
                ParamRejectReason::OutOfRange(ParamId::new(PARAM_FORCE_LIMIT)),
            ),
            (
                PARAM_FORCE_LIMIT,
                ParamValue::Int(1),
                ParamRejectReason::WrongType(ParamId::new(PARAM_FORCE_LIMIT)),
            ),
            (
                "cartpole.nope",
                ParamValue::Float(1.0),
                ParamRejectReason::UnknownParam(ParamId::new("cartpole.nope")),
            ),
        ] {
            assert_eq!(
                ui.set_param(id, val),
                ParamOutcome::Rejected { reason: why },
                "{id}"
            );
        }
        ui.send(ControlAction::Step);
        assert_eq!(ui.probe.snapshot().unwrap().force_limit_changes.len(), n);
        assert_eq!(ui.obs().param_rejection_count, 3);
        assert_eq!(ui.probe.snapshot().unwrap().view.force_limit, 0.5);
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
    fn ac2_ac3_no_cartpole_dynamics_or_controller_in_viewport_and_single_engine() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let tokens = [
            "CartPoleEnv",
            "CartPoleModel",
            "gogamen_lab_cartpole",
            "CartPoleRollout",
            "LinearFeedbackController",
            "CartPolePolicy",
            "integrate_step",
            "compute_derivatives",
            "select_action",
            ".apply(",
            ".step(",
        ];
        for f in ["viewport.rs", "camera.rs", "shell.rs"] {
            let text = strip_comments_and_tests(&std::fs::read_to_string(src.join(f)).unwrap());
            for t in tokens {
                assert!(!text.contains(t), "{f} contains `{t}`");
            }
        }
        // ningún módulo de studio define dinámica ni controlador CartPole propios.
        for entry in std::fs::read_dir(&src).unwrap() {
            let path = entry.unwrap().path();
            let text = strip_comments_and_tests(&std::fs::read_to_string(&path).unwrap());
            for t in [
                "fn integrate_step",
                "fn compute_derivatives",
                "impl World for",
                "impl Actuator for",
                "impl CartPolePolicy for",
            ] {
                assert!(!text.contains(t), "{path:?} contains `{t}`");
            }
        }
        // el proyector sólo recibe una vista inmutable; el motor vive en el lab (path dep).
        let me = std::fs::read_to_string(src.join("cartpole_lab.rs")).unwrap();
        assert!(me.contains("pub fn project_cartpole_scene(view: &CartPoleView)"));
        let manifest = std::fs::read_to_string(src.join("..").join("Cargo.toml")).unwrap();
        assert!(manifest.contains("gogamen-lab-cartpole = { path = \"../../labs/cartpole\" }"));
        let lab =
            std::fs::read_to_string(src.join("../../../labs/cartpole/src/physics.rs")).unwrap();
        assert!(lab.contains("pub fn integrate_step"));
    }

    #[test]
    fn ac3_lab_headless_episode_runs_without_studio_types() {
        // Mismo punto de entrada público del lab, sin shell ni escena.
        let mut c = LinearFeedbackController::default();
        let a = run_cartpole_headless(cartpole_demo_config(), 300, MasterSeed::new(SEED), &mut c)
            .unwrap();
        let mut c = LinearFeedbackController::default();
        let b = run_cartpole_headless(cartpole_demo_config(), 300, MasterSeed::new(SEED), &mut c)
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a.ticks_executed, 300);
        assert_eq!(a.terminal, Terminal::Ongoing);
        // el manifiesto del lab no depende de studio ni de crates gráficos
        let m = std::fs::read_to_string(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../labs/cartpole/Cargo.toml"),
        )
        .unwrap();
        for banned in ["gogamen-studio", "gogamen-projection", "bevy", "egui"] {
            assert!(!m.contains(banned), "cartpole lab references {banned}");
        }
    }
}
