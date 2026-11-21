//! Inspector genérico por entidad (GOG-92): modelo de datos puro + render texto/SVG.
//!
//! - Entrada: [`SnapshotView`] (revisión, tick, entidades con campos tipados).
//! - Selección por identidad estable [`EntityRef`] = `(id, epoch)`: si el `id` se recicla con otro
//!   `epoch` el inspector devuelve [`InspectorStatus::Vanished`] y **nunca** los datos de la
//!   entidad nueva.
//! - Nivel de detalle ([`Lod`]) por suscripción, filtrado declarativo vía [`FieldSchema`].
//! - Dirigido por esquema: un campo nuevo (sin descriptor) se renderiza como fila con el
//!   descriptor por defecto; no hay paneles dedicados por entidad.
//! - Sin Bevy/egui; `gogamen-core` y `gogamen-cli` no dependen de este módulo.

use std::collections::BTreeMap;
use std::fmt::Write;

/// Identidad estable de una entidad: `id` reutilizable + `epoch` (generación) que cambia al reciclar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityRef {
    pub id: u64,
    pub epoch: u64,
}

impl EntityRef {
    pub const fn new(id: u64, epoch: u64) -> Self {
        Self { id, epoch }
    }
}

/// Valor tipado de un campo.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Number(f64),
    Integer(i64),
    Bool(bool),
    Text(String),
    Vector(Vec<f64>),
}

impl FieldValue {
    /// Texto determinista (números a 3 decimales).
    pub fn display(&self) -> String {
        match self {
            Self::Number(v) => num(*v),
            Self::Integer(v) => v.to_string(),
            Self::Bool(v) => v.to_string(),
            Self::Text(v) => v.clone(),
            Self::Vector(v) => {
                let parts: Vec<String> = v.iter().map(|x| num(*x)).collect();
                format!("({})", parts.join(", "))
            }
        }
    }
}

/// Procedencia del dato.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    Observed,
    Derived,
    Assumed,
    Unknown,
}

/// Calidad del dato.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Good,
    Degraded,
    Bad,
    Unknown,
}

/// Campo tipado de una entidad en un snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub value: FieldValue,
    pub unit: Option<String>,
    pub provenance: Provenance,
    /// Tick en el que se observó el valor (edad = `snapshot.tick - observed_tick`).
    pub observed_tick: u64,
    pub quality: Quality,
}

impl Field {
    /// Campo observado, calidad buena, observado en `tick`.
    pub fn new(name: impl Into<String>, value: FieldValue, tick: u64) -> Self {
        Self {
            name: name.into(),
            value,
            unit: None,
            provenance: Provenance::Observed,
            observed_tick: tick,
            quality: Quality::Good,
        }
    }
    pub fn unit(mut self, unit: impl Into<String>) -> Self {
        self.unit = Some(unit.into());
        self
    }
    pub fn provenance(mut self, p: Provenance) -> Self {
        self.provenance = p;
        self
    }
    pub fn quality(mut self, q: Quality) -> Self {
        self.quality = q;
        self
    }
}

/// Entidad dentro de un snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityView {
    pub entity: EntityRef,
    pub kind: String,
    pub fields: Vec<Field>,
}

/// Vista inmutable de un snapshot (entrada común del inspector).
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotView {
    pub revision: u64,
    pub tick: u64,
    pub entities: Vec<EntityView>,
}

/// Nivel de detalle de una suscripción (ordenado: `Minimal < Standard < Full`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Lod {
    Minimal,
    Standard,
    Full,
}

/// Descriptor declarativo de un campo.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDescriptor {
    pub name: String,
    pub label: String,
    /// Nivel mínimo en el que el campo es visible.
    pub min_lod: Lod,
    /// Umbral de obsolescencia propio (ticks); `None` = el del inspector.
    pub stale_after: Option<u64>,
}

impl FieldDescriptor {
    pub fn new(name: impl Into<String>, label: impl Into<String>, min_lod: Lod) -> Self {
        Self {
            name: name.into(),
            label: label.into(),
            min_lod,
            stale_after: None,
        }
    }
    pub fn stale_after(mut self, ticks: u64) -> Self {
        self.stale_after = Some(ticks);
        self
    }
}

/// Registro de descriptores. Los campos sin descriptor usan uno por defecto (`Full`, etiqueta = nombre).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldSchema {
    descriptors: BTreeMap<String, FieldDescriptor>,
}

impl FieldSchema {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn register(&mut self, d: FieldDescriptor) -> &mut Self {
        self.descriptors.insert(d.name.clone(), d);
        self
    }
    pub fn get(&self, name: &str) -> Option<&FieldDescriptor> {
        self.descriptors.get(name)
    }
    fn resolve(&self, name: &str) -> FieldDescriptor {
        self.get(name)
            .cloned()
            .unwrap_or_else(|| FieldDescriptor::new(name, name, Lod::Full))
    }
}

/// Motivo de obsolescencia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StaleReason {
    /// La edad de algún campo supera el umbral.
    AgeExceeded {
        field: String,
        age: u64,
        threshold: u64,
    },
    /// El snapshot recibido es más antiguo que otro ya visto.
    RevisionRegressed { seen: u64, got: u64 },
}

/// Motivo de desaparición.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VanishReason {
    /// No existe ninguna entidad con ese id.
    NotPresent,
    /// El id existe pero con otro `epoch` (reciclado): no se muestran sus datos.
    Recycled { current_epoch: u64 },
}

/// Estado del inspector respecto a la selección.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InspectorStatus {
    NoSelection,
    Live,
    Stale(Vec<StaleReason>),
    Vanished(VanishReason),
}

/// Fila renderizable (una por campo visible).
#[derive(Debug, Clone, PartialEq)]
pub struct InspectorRow {
    pub name: String,
    pub label: String,
    pub value: FieldValue,
    pub unit: Option<String>,
    pub provenance: Provenance,
    pub quality: Quality,
    pub age: u64,
    pub stale: bool,
}

impl InspectorRow {
    pub fn value_text(&self) -> String {
        match &self.unit {
            Some(u) => format!("{} {u}", self.value.display()),
            None => self.value.display(),
        }
    }
}

/// Salida del inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct InspectorView {
    pub revision: u64,
    pub tick: u64,
    pub selection: Option<EntityRef>,
    pub kind: Option<String>,
    pub lod: Lod,
    pub status: InspectorStatus,
    pub rows: Vec<InspectorRow>,
}

impl InspectorView {
    /// Panel de texto determinista.
    pub fn render_text(&self) -> String {
        let mut o = String::new();
        let _ = writeln!(
            o,
            "Inspector rev={} tick={} lod={:?}",
            self.revision, self.tick, self.lod
        );
        match (&self.selection, &self.status) {
            (None, _) | (_, InspectorStatus::NoSelection) => {
                let _ = writeln!(o, "(no selection)");
            }
            (Some(e), status) => {
                let kind = self.kind.as_deref().unwrap_or("?");
                let _ = writeln!(
                    o,
                    "entity {}@{} [{kind}] {}",
                    e.id,
                    e.epoch,
                    status_text(status)
                );
            }
        }
        for r in &self.rows {
            let _ = writeln!(
                o,
                "{}: {} | prov={:?} age={} q={:?}{}",
                r.label,
                r.value_text(),
                r.provenance,
                r.age,
                r.quality,
                if r.stale { " STALE" } else { "" }
            );
        }
        o
    }

    /// Panel SVG determinista (una línea por fila).
    pub fn render_svg(&self) -> String {
        let text = self.render_text();
        let lines: Vec<&str> = text.lines().collect();
        let h = 20 + lines.len() * 16;
        let mut o = String::new();
        let _ = writeln!(
            o,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"420\" height=\"{h}\" viewBox=\"0 0 420 {h}\">"
        );
        let _ = writeln!(o, "<rect width=\"100%\" height=\"100%\" fill=\"#f5f5f5\"/>");
        for (i, l) in lines.iter().enumerate() {
            let _ = writeln!(
                o,
                "<text x=\"8\" y=\"{}\" font-family=\"monospace\" font-size=\"12\">{}</text>",
                18 + i * 16,
                escape(l)
            );
        }
        o.push_str("</svg>\n");
        o
    }
}

fn status_text(s: &InspectorStatus) -> String {
    match s {
        InspectorStatus::NoSelection => "NO_SELECTION".into(),
        InspectorStatus::Live => "LIVE".into(),
        InspectorStatus::Stale(rs) => {
            let parts: Vec<String> = rs
                .iter()
                .map(|r| match r {
                    StaleReason::AgeExceeded {
                        field,
                        age,
                        threshold,
                    } => format!("{field} age {age}>{threshold}"),
                    StaleReason::RevisionRegressed { seen, got } => {
                        format!("revision regressed {seen}->{got}")
                    }
                })
                .collect();
            format!("STALE ({})", parts.join("; "))
        }
        InspectorStatus::Vanished(VanishReason::NotPresent) => "VANISHED".into(),
        InspectorStatus::Vanished(VanishReason::Recycled { current_epoch }) => {
            format!("VANISHED (id recycled, now epoch {current_epoch})")
        }
    }
}

/// Inspector: selección estable + LOD + esquema + umbral de edad.
#[derive(Debug, Clone)]
pub struct Inspector {
    selection: Option<EntityRef>,
    lod: Lod,
    schema: FieldSchema,
    stale_after: u64,
    last_revision: Option<u64>,
}

impl Inspector {
    pub fn new(schema: FieldSchema, lod: Lod, stale_after: u64) -> Self {
        Self {
            selection: None,
            lod,
            schema,
            stale_after,
            last_revision: None,
        }
    }
    pub fn select(&mut self, entity: EntityRef) {
        self.selection = Some(entity);
    }
    pub fn clear_selection(&mut self) {
        self.selection = None;
    }
    pub fn selection(&self) -> Option<EntityRef> {
        self.selection
    }
    pub fn set_lod(&mut self, lod: Lod) {
        self.lod = lod;
    }
    pub fn lod(&self) -> Lod {
        self.lod
    }
    pub fn schema_mut(&mut self) -> &mut FieldSchema {
        &mut self.schema
    }

    /// Proyecta el snapshot sobre la selección actual.
    pub fn inspect(&mut self, snap: &SnapshotView) -> InspectorView {
        let regression = match self.last_revision {
            Some(seen) if snap.revision < seen => Some(StaleReason::RevisionRegressed {
                seen,
                got: snap.revision,
            }),
            _ => {
                self.last_revision = Some(snap.revision);
                None
            }
        };
        let mut view = InspectorView {
            revision: snap.revision,
            tick: snap.tick,
            selection: self.selection,
            kind: None,
            lod: self.lod,
            status: InspectorStatus::NoSelection,
            rows: Vec::new(),
        };
        let Some(sel) = self.selection else {
            return view;
        };
        let Some(ent) = snap.entities.iter().find(|e| e.entity.id == sel.id) else {
            view.status = InspectorStatus::Vanished(VanishReason::NotPresent);
            return view;
        };
        if ent.entity.epoch != sel.epoch {
            view.status = InspectorStatus::Vanished(VanishReason::Recycled {
                current_epoch: ent.entity.epoch,
            });
            return view;
        }
        view.kind = Some(ent.kind.clone());
        let mut reasons: Vec<StaleReason> = regression.into_iter().collect();
        for f in &ent.fields {
            let d = self.schema.resolve(&f.name);
            if d.min_lod > self.lod {
                continue;
            }
            let age = snap.tick.saturating_sub(f.observed_tick);
            let threshold = d.stale_after.unwrap_or(self.stale_after);
            let stale = age > threshold;
            if stale {
                reasons.push(StaleReason::AgeExceeded {
                    field: f.name.clone(),
                    age,
                    threshold,
                });
            }
            view.rows.push(InspectorRow {
                name: f.name.clone(),
                label: d.label,
                value: f.value.clone(),
                unit: f.unit.clone(),
                provenance: f.provenance,
                quality: f.quality,
                age,
                stale,
            });
        }
        view.status = if reasons.is_empty() {
            InspectorStatus::Live
        } else {
            InspectorStatus::Stale(reasons)
        };
        view
    }
}

fn num(v: f64) -> String {
    let s = format!("{v:.3}");
    if s == "-0.000" { "0.000".to_owned() } else { s }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------------------------
// Cableado al lab Grid (GOG-110)
// ---------------------------------------------------------------------------------------------

use crate::control::Observation;
use crate::grid_lab::GridWorldView;

/// Id estable de la entidad agente.
pub const GRID_AGENT_ID: u64 = 1;
const GRID_CELL_BASE: u64 = 1_000;

/// Id de la celda `(x, y)` (estable mientras no cambie el mapa).
pub fn grid_cell_id(view: &GridWorldView, x: usize, y: usize) -> u64 {
    GRID_CELL_BASE + (y * view.map.width() + x) as u64
}

/// Epoch de entidades Grid: sesión (`epoch`, `generation`); cambia con cada `Reset`.
pub fn grid_epoch(obs: &Observation) -> u64 {
    (obs.session.epoch << 32) | (obs.session.generation & 0xffff_ffff)
}

/// Descriptores del lab Grid.
pub fn grid_schema() -> FieldSchema {
    let mut s = FieldSchema::new();
    s.register(FieldDescriptor::new("position", "Position", Lod::Minimal))
        .register(FieldDescriptor::new("tick", "Tick", Lod::Minimal))
        .register(FieldDescriptor::new(
            "last_reward",
            "Last reward",
            Lod::Standard,
        ))
        .register(FieldDescriptor::new(
            "total_reward",
            "Total reward",
            Lod::Standard,
        ))
        .register(FieldDescriptor::new("terminal", "Terminal", Lod::Standard))
        .register(FieldDescriptor::new("collisions", "Collisions", Lod::Full))
        .register(FieldDescriptor::new("cell_kind", "Cell kind", Lod::Minimal));
    s
}

/// Adapta la vista real del mundo Grid + la observación de sesión a un [`SnapshotView`]:
/// una entidad agente y una por celda. `revision` = revisión de la sesión.
pub fn grid_snapshot_view(world: &GridWorldView, obs: &Observation) -> SnapshotView {
    use gogamen_lab_grid::{CellKind, GridPos};
    let tick = world.tick.as_u64();
    let epoch = grid_epoch(obs);
    let pos = |x: usize, y: usize| FieldValue::Vector(vec![x as f64, y as f64]);
    let mut entities = vec![EntityView {
        entity: EntityRef::new(GRID_AGENT_ID, epoch),
        kind: "agent".into(),
        fields: vec![
            Field::new("position", pos(world.agent.x, world.agent.y), tick).unit("cell"),
            Field::new("tick", FieldValue::Integer(tick as i64), tick).unit("tick"),
            Field::new("last_reward", FieldValue::Number(world.last_reward), tick).unit("reward"),
            Field::new("total_reward", FieldValue::Number(world.total_reward), tick)
                .unit("reward")
                .provenance(Provenance::Derived),
            Field::new(
                "collisions",
                FieldValue::Integer(world.collisions as i64),
                tick,
            ),
            Field::new(
                "terminal",
                FieldValue::Text(format!("{:?}", world.terminal)),
                tick,
            ),
        ],
    }];
    for y in 0..world.map.height() {
        for x in 0..world.map.width() {
            let kind = world
                .map
                .cell(GridPos::new(x, y))
                .unwrap_or(CellKind::Empty);
            entities.push(EntityView {
                entity: EntityRef::new(grid_cell_id(world, x, y), epoch),
                kind: "cell".into(),
                fields: vec![
                    Field::new("position", pos(x, y), tick).unit("cell"),
                    Field::new("cell_kind", FieldValue::Text(format!("{kind:?}")), tick),
                ],
            });
        }
    }
    SnapshotView {
        revision: obs.revision,
        tick,
        entities,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{CommandId, ControlAction, ControlCommand, ControlPort, ObservationPort};
    use crate::grid_lab::{GRID_LAB_ID, grid_lab_entry_with_probe};
    use crate::shell::StudioShell;
    use gogamen_core::MasterSeed;
    use gogamen_lab_grid::world::GridWorldConfig;

    fn ent(id: u64, epoch: u64, fields: Vec<Field>) -> EntityView {
        EntityView {
            entity: EntityRef::new(id, epoch),
            kind: "robot".into(),
            fields,
        }
    }
    fn snap(rev: u64, tick: u64, entities: Vec<EntityView>) -> SnapshotView {
        SnapshotView {
            revision: rev,
            tick,
            entities,
        }
    }
    fn insp(lod: Lod) -> Inspector {
        let mut s = FieldSchema::new();
        s.register(FieldDescriptor::new("speed", "Speed", Lod::Minimal))
            .register(FieldDescriptor::new("temp", "Temperature", Lod::Standard).stale_after(2))
            .register(FieldDescriptor::new("debug", "Debug", Lod::Full));
        Inspector::new(s, lod, 5)
    }
    fn fields(t: u64) -> Vec<Field> {
        vec![
            Field::new("speed", FieldValue::Number(1.5), t).unit("m/s"),
            Field::new("temp", FieldValue::Number(36.6), t)
                .unit("degC")
                .quality(Quality::Degraded),
            Field::new("debug", FieldValue::Bool(true), t),
        ]
    }

    #[test]
    fn ac1_rows_show_values_units_provenance_age_quality_revision() {
        let mut i = insp(Lod::Full);
        i.select(EntityRef::new(7, 1));
        let v = i.inspect(&snap(42, 10, vec![ent(7, 1, fields(9))]));
        assert_eq!(v.revision, 42);
        assert_eq!(v.status, InspectorStatus::Live);
        assert_eq!(v.rows.len(), 3);
        assert_eq!(v.rows[0].unit.as_deref(), Some("m/s"));
        assert_eq!(v.rows[0].age, 1);
        assert_eq!(v.rows[1].quality, Quality::Degraded);
        let t = v.render_text();
        assert!(t.contains("rev=42"));
        assert!(t.contains("Speed: 1.500 m/s | prov=Observed age=1 q=Good"));
        assert!(v.render_svg().contains("Speed: 1.500 m/s"));
        assert_eq!(
            t,
            i.inspect(&snap(42, 10, vec![ent(7, 1, fields(9))]))
                .render_text()
        );
    }

    #[test]
    fn ac2_recycled_id_is_vanished_never_new_entity_data() {
        let mut i = insp(Lod::Full);
        i.select(EntityRef::new(7, 1));
        let mut new_fields = fields(10);
        new_fields[0].value = FieldValue::Number(99.0);
        let v = i.inspect(&snap(43, 10, vec![ent(7, 2, new_fields)]));
        assert_eq!(
            v.status,
            InspectorStatus::Vanished(VanishReason::Recycled { current_epoch: 2 })
        );
        assert!(v.rows.is_empty());
        assert!(!v.render_text().contains("99.000"));
        assert!(v.render_text().contains("VANISHED"));
    }

    #[test]
    fn ac2_vanished_entity_and_stale_age_and_revision_regression() {
        let mut i = insp(Lod::Full);
        i.select(EntityRef::new(7, 1));
        let v = i.inspect(&snap(1, 10, vec![ent(8, 1, fields(10))]));
        assert_eq!(
            v.status,
            InspectorStatus::Vanished(VanishReason::NotPresent)
        );
        // age 3 > per-field threshold 2 for temp, <= global 5 for others
        let v = i.inspect(&snap(2, 10, vec![ent(7, 1, fields(7))]));
        match &v.status {
            InspectorStatus::Stale(rs) => {
                assert_eq!(
                    rs,
                    &vec![StaleReason::AgeExceeded {
                        field: "temp".into(),
                        age: 3,
                        threshold: 2
                    }]
                );
            }
            s => panic!("{s:?}"),
        }
        assert!(v.rows[1].stale && !v.rows[0].stale);
        assert!(v.render_text().contains("STALE"));
        // out-of-order older snapshot
        let v = i.inspect(&snap(1, 10, vec![ent(7, 1, fields(10))]));
        assert_eq!(
            v.status,
            InspectorStatus::Stale(vec![StaleReason::RevisionRegressed { seen: 2, got: 1 }])
        );
    }

    #[test]
    fn no_selection_is_reported() {
        let mut i = insp(Lod::Full);
        let v = i.inspect(&snap(1, 1, vec![ent(7, 1, fields(1))]));
        assert_eq!(v.status, InspectorStatus::NoSelection);
        assert!(v.render_text().contains("(no selection)"));
    }

    #[test]
    fn lod_filters_fields_declaratively() {
        let s = snap(1, 1, vec![ent(7, 1, fields(1))]);
        let names = |lod| {
            let mut i = insp(lod);
            i.select(EntityRef::new(7, 1));
            i.inspect(&s)
                .rows
                .into_iter()
                .map(|r| r.name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(Lod::Minimal), ["speed"]);
        assert_eq!(names(Lod::Standard), ["speed", "temp"]);
        assert_eq!(names(Lod::Full), ["speed", "temp", "debug"]);
        let mut i = insp(Lod::Minimal);
        i.select(EntityRef::new(7, 1));
        i.set_lod(Lod::Full);
        assert_eq!(i.inspect(&s).rows.len(), 3);
    }

    #[test]
    fn ac3_new_field_renders_without_dedicated_panel_or_descriptor() {
        let mut i = insp(Lod::Full);
        i.select(EntityRef::new(7, 1));
        let mut f = fields(1);
        f.push(
            Field::new("battery", FieldValue::Vector(vec![0.5, -0.0]), 1)
                .unit("V")
                .provenance(Provenance::Assumed),
        );
        f.push(Field::new("note", FieldValue::Text("a<b".into()), 1));
        let v = i.inspect(&snap(1, 1, vec![ent(7, 1, f)]));
        assert_eq!(v.rows.len(), 5);
        let r = &v.rows[3];
        assert_eq!(
            (r.label.as_str(), r.value_text()),
            ("battery", "(0.500, 0.000) V".into())
        );
        assert_eq!(r.provenance, Provenance::Assumed);
        assert!(v.render_svg().contains("a&lt;b"));
        // unknown fields default to Full, so they hide under lower LODs
        i.set_lod(Lod::Standard);
        let mut f2 = fields(1);
        f2.push(Field::new("battery", FieldValue::Integer(3), 1));
        assert_eq!(i.inspect(&snap(2, 1, vec![ent(7, 1, f2)])).rows.len(), 2);
        // registering a descriptor later changes label/LOD with no code
        i.schema_mut()
            .register(FieldDescriptor::new("battery", "Battery", Lod::Minimal));
        let mut f3 = fields(1);
        f3.push(Field::new("battery", FieldValue::Integer(3), 1));
        let v = i.inspect(&snap(3, 1, vec![ent(7, 1, f3)]));
        assert_eq!(v.rows.last().unwrap().label, "Battery");
    }

    struct Ui {
        shell: StudioShell,
        probe: crate::grid_lab::GridProbe,
        next: u64,
    }
    impl Ui {
        fn new() -> Self {
            let (entry, probe) =
                grid_lab_entry_with_probe(GridWorldConfig::default(), MasterSeed::new(42)).unwrap();
            let mut shell = StudioShell::new(vec![entry], 400.0, 300.0).unwrap();
            shell.select_lab(GRID_LAB_ID).unwrap();
            Self {
                shell,
                probe,
                next: 1,
            }
        }
        fn send(&mut self, action: ControlAction) {
            let port = self.shell.control_port().unwrap().unwrap();
            let session = self.obs().session;
            let id = CommandId(self.next);
            self.next += 1;
            port.submit(ControlCommand {
                id,
                session,
                action,
            });
            self.shell.sync_active();
        }
        fn obs(&self) -> Observation {
            self.shell.observation_port().unwrap().observe()
        }
        fn snapshot(&self) -> SnapshotView {
            let w = self.probe.snapshot().unwrap().view;
            grid_snapshot_view(&w, &self.obs())
        }
    }

    #[test]
    fn grid_agent_and_cell_show_real_fields_and_reset_vanishes_selection() {
        let mut ui = Ui::new();
        for _ in 0..3 {
            ui.send(ControlAction::Step);
        }
        let world = ui.probe.snapshot().unwrap().view;
        let obs = ui.obs();
        let mut i = Inspector::new(grid_schema(), Lod::Full, 0);
        i.select(EntityRef::new(GRID_AGENT_ID, grid_epoch(&obs)));
        let v = i.inspect(&ui.snapshot());
        assert_eq!(v.status, InspectorStatus::Live);
        assert_eq!(v.revision, obs.revision);
        let by = |n: &str| v.rows.iter().find(|r| r.name == n).unwrap().clone();
        assert_eq!(
            by("position").value,
            FieldValue::Vector(vec![world.agent.x as f64, world.agent.y as f64])
        );
        assert_eq!(by("position").unit.as_deref(), Some("cell"));
        assert_eq!(
            by("tick").value,
            FieldValue::Integer(world.tick.as_u64() as i64)
        );
        assert_eq!(
            by("total_reward").value,
            FieldValue::Number(world.total_reward)
        );
        assert_eq!(by("total_reward").unit.as_deref(), Some("reward"));
        assert!(v.render_text().contains(&format!("rev={}", obs.revision)));
        assert!(v.render_svg().starts_with("<svg"));

        // minimal LOD hides reward rows
        i.set_lod(Lod::Minimal);
        assert!(
            i.inspect(&ui.snapshot())
                .rows
                .iter()
                .all(|r| r.name != "total_reward")
        );

        // cell selection
        i.set_lod(Lod::Full);
        let (x, y) = (world.agent.x, world.agent.y);
        i.select(EntityRef::new(grid_cell_id(&world, x, y), grid_epoch(&obs)));
        let v = i.inspect(&ui.snapshot());
        assert_eq!(v.kind.as_deref(), Some("cell"));
        assert!(v.rows.iter().any(|r| r.name == "cell_kind"));

        // Reset -> new generation: old selection is vanished (recycled), not the new world's data
        i.select(EntityRef::new(GRID_AGENT_ID, grid_epoch(&obs)));
        ui.send(ControlAction::Reset);
        let v = i.inspect(&ui.snapshot());
        assert!(matches!(
            v.status,
            InspectorStatus::Vanished(VanishReason::Recycled { .. })
        ));
        assert!(v.rows.is_empty());
    }
}
