//! Inspector cognitivo y de comunicación (GOG-96): proyector puro `CognitiveView -> Scene2D`.
//!
//! - **Datos, no motores:** el proyector sólo recibe un [`CognitiveView`] inmutable (creencias,
//!   evidencia, episodios, intenciones, mensajes, enlaces causales *registrados*). No hay red,
//!   LLM, planificador ni dinámica en esta ruta de render (test textual de guarda).
//! - **Tipos de core reutilizados:** [`BinaryBelief`] (creencia), [`EvidenceRef`], [`Confidence`],
//!   [`ObservationKind`], [`EpisodicRecord`] (memoria) y [`SituatedMessage`] (comunicación, GOG-44).
//! - **Confidencial por defecto:** [`Authorization::default`] es `Confidential`; mensajes y memoria
//!   muestran marcadores explícitos `[redacted]` (nunca se omiten). Con `Authorized` se ve el contenido.
//!   Creencias, evidencia e intenciones son estructuradas (sin texto libre bruto) y se muestran siempre.
//! - **Incertidumbre explícita:** campos no registrados son [`Known::Unknown`] y se pintan `unknown`;
//!   la edad se deriva de `view.tick - observed_tick`; la confianza (de la creencia o declarada) se
//!   muestra separada de la fiabilidad de la fuente.
//! - **Sin causalidad inventada:** sólo se dibujan los [`CausalLink`] presentes en la vista, y sólo si
//!   ambos extremos existen. Los demás se cuentan como no resueltos.
//!
//! `gogamen-core` y `gogamen-cli` no dependen de este módulo.

use gogamen_core::{
    BinaryBelief, CommPayload, Confidence, EpisodeId, EpisodicRecord, EpistemicKind, EvidenceRef,
    InterlocutorKind, MessageId, ObservationKind, SituatedMessage, Tick,
};
use gogamen_projection::{
    CoordinateSpace, LayerId, Point2, Scene2D, SceneId, SceneLayer, SceneNode, SceneNodeId,
    ScenePrimitive,
};

use crate::camera::Camera2D;
use crate::viewport::SoftwareSvgBackend;

/// Nivel de autorización del observador. El valor por defecto es el más restrictivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Authorization {
    /// Contenido de mensajes y memoria oculto (`[redacted]`).
    #[default]
    Confidential,
    /// Contenido visible.
    Authorized,
}

/// Valor registrado o explícitamente desconocido (nunca adivinado).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Known<T> {
    Known(T),
    Unknown,
}

impl<T> From<Option<T>> for Known<T> {
    fn from(v: Option<T>) -> Self {
        v.map_or(Self::Unknown, Self::Known)
    }
}

/// Origen de una evidencia.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceSource {
    Human,
    DirectSensor,
    Agent,
    Unknown,
}

impl EvidenceSource {
    fn label(self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::DirectSensor => "direct-sensor",
            Self::Agent => "agent",
            Self::Unknown => "unknown",
        }
    }
}

/// Evidencia asociada a una hipótesis.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceView {
    pub evidence: EvidenceRef,
    pub source: EvidenceSource,
    pub claim_kind: ObservationKind,
    /// Lo que la evidencia afirma sobre la hipótesis (`true` la apoya).
    pub says: Known<bool>,
    pub observed_tick: Known<u64>,
    /// Confianza declarada en esta observación concreta.
    pub reported_confidence: Known<Confidence>,
    /// Fiabilidad histórica de la fuente (distinta de la confianza anterior).
    pub source_reliability: Known<Confidence>,
}

/// Hipótesis binaria sobre un hecho (`subject`). Dos hipótesis con el mismo `subject` y distinto
/// `statement` se muestran como contradictorias.
#[derive(Debug, Clone, PartialEq)]
pub struct HypothesisView {
    pub id: u64,
    pub subject: String,
    pub statement: String,
    pub belief: Known<BinaryBelief>,
    pub last_update_tick: Known<u64>,
    pub evidence: Vec<EvidenceView>,
}

/// Intención registrada (estructurada).
#[derive(Debug, Clone, PartialEq)]
pub struct IntentionView {
    pub id: u64,
    pub label: String,
    pub since_tick: Known<u64>,
}

/// Mensaje situado (core) más metadatos que el mensaje no transporta.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageView {
    pub message: SituatedMessage,
    pub sender_kind: Known<InterlocutorKind>,
    pub sent_tick: Known<u64>,
}

/// Referencia a un elemento de la vista.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CogRef {
    Hypothesis(u64),
    Evidence(EvidenceRef),
    Episode(EpisodeId),
    Intention(u64),
    Message(MessageId),
}

/// Enlace causal **registrado** (`from` motivó/produjo `to`); el proyector nunca los infiere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CausalLink {
    pub from: CogRef,
    pub to: CogRef,
}

/// Entrada común del inspector cognitivo.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CognitiveView {
    pub title: String,
    pub tick: u64,
    pub authorization: Authorization,
    pub hypotheses: Vec<HypothesisView>,
    pub episodes: Vec<EpisodicRecord>,
    pub intentions: Vec<IntentionView>,
    pub messages: Vec<MessageView>,
    pub links: Vec<CausalLink>,
}

const REDACTED: &str = "[redacted]";
const ROW_H: f64 = 20.0;
const WIDTH: f64 = 900.0;

fn conf(c: Known<Confidence>) -> String {
    match c {
        Known::Known(c) => format!("{:.3}", c.as_f64()),
        Known::Unknown => "unknown".to_owned(),
    }
}

fn age(now: u64, then: Known<u64>) -> String {
    match then {
        Known::Known(t) if t <= now => format!("{}t", now - t),
        Known::Known(_) => "unknown(future-dated)".to_owned(),
        Known::Unknown => "unknown".to_owned(),
    }
}

fn payload_text(p: &CommPayload) -> String {
    match p {
        CommPayload::Notice { summary, .. } => format!("notice: {summary}"),
        CommPayload::Question { prompt, .. } => format!("question: {prompt}"),
        CommPayload::Answer { text, .. } => format!("answer: {text}"),
        CommPayload::NoResponse { reason } => format!("no-response: {reason}"),
        CommPayload::StructuredFact {
            fact_key,
            fact_value,
            ..
        } => format!("fact: {fact_key}={fact_value}"),
    }
}

struct Row {
    text: String,
    indent: u8,
    kind: EpistemicKind,
    anchor: Option<CogRef>,
    header: bool,
}

fn row(text: String, indent: u8, kind: EpistemicKind, anchor: Option<CogRef>) -> Row {
    Row {
        text,
        indent,
        kind,
        anchor,
        header: false,
    }
}

fn header(text: String) -> Row {
    Row {
        text,
        indent: 0,
        kind: EpistemicKind::Observed,
        anchor: None,
        header: true,
    }
}

fn hypothesis_rows(view: &CognitiveView, out: &mut Vec<Row>) {
    out.push(header(format!("== Beliefs ({}) ==", view.hypotheses.len())));
    if view.hypotheses.is_empty() {
        out.push(row(
            "(none recorded)".into(),
            1,
            EpistemicKind::Observed,
            None,
        ));
    }
    for h in &view.hypotheses {
        let p = match &h.belief {
            Known::Known(b) => format!("P(true)={:.3}", b.p_true()),
            Known::Unknown => "P(true)=unknown".to_owned(),
        };
        let rivals: Vec<String> = view
            .hypotheses
            .iter()
            .filter(|o| o.id != h.id && o.subject == h.subject && o.statement != h.statement)
            .map(|o| format!("H{}", o.id))
            .collect();
        let conflict = if rivals.is_empty() {
            String::new()
        } else {
            format!(" CONFLICT with {}", rivals.join(","))
        };
        out.push(row(
            format!(
                "H{} [{}] {} {p} updated_age={}{conflict}",
                h.id,
                h.subject,
                h.statement,
                age(view.tick, h.last_update_tick)
            ),
            0,
            EpistemicKind::Inferred,
            Some(CogRef::Hypothesis(h.id)),
        ));
        if h.evidence.is_empty() {
            out.push(row(
                "no evidence recorded".into(),
                1,
                EpistemicKind::Observed,
                None,
            ));
        }
        for e in &h.evidence {
            let says = match e.says {
                Known::Known(true) => "supports",
                Known::Known(false) => "refutes",
                Known::Unknown => "says=unknown",
            };
            out.push(row(
                format!(
                    "E{} src={} kind={} {says} age={} reported_conf={} source_reliability={}",
                    e.evidence.as_u64(),
                    e.source.label(),
                    e.claim_kind,
                    age(view.tick, e.observed_tick),
                    conf(e.reported_confidence),
                    conf(e.source_reliability)
                ),
                1,
                EpistemicKind::Observed,
                Some(CogRef::Evidence(e.evidence)),
            ));
        }
    }
}

fn episode_rows(view: &CognitiveView, out: &mut Vec<Row>) {
    out.push(header(format!("== Episodes ({}) ==", view.episodes.len())));
    if view.episodes.is_empty() {
        out.push(row(
            "(none recorded)".into(),
            1,
            EpistemicKind::Observed,
            None,
        ));
    }
    let open = view.authorization == Authorization::Authorized;
    for r in &view.episodes {
        let a = age(view.tick, Known::Known(r.tick.as_u64()));
        let text = if open {
            let f: Vec<String> = r
                .numeric_features
                .iter()
                .map(|v| format!("{v:.3}"))
                .collect();
            let raw = if r.raw_text_redacted {
                " raw_text=redacted-at-source"
            } else {
                ""
            };
            format!(
                "EP{} owner={} entity={} summary_code={} features=({}) claim={} conf={} age={a}{raw}",
                r.episode_id.as_u64(),
                r.owner.as_u64(),
                r.entity.as_u64(),
                r.summary_code.as_u64(),
                f.join(", "),
                r.provenance.claim_kind,
                r.provenance.confidence,
            )
        } else {
            format!(
                "EP{} owner={REDACTED} summary={REDACTED} features={REDACTED} age={a}",
                r.episode_id.as_u64()
            )
        };
        out.push(row(
            text,
            0,
            EpistemicKind::Observed,
            Some(CogRef::Episode(r.episode_id)),
        ));
    }
}

fn intention_rows(view: &CognitiveView, out: &mut Vec<Row>) {
    out.push(header(format!(
        "== Intentions ({}) ==",
        view.intentions.len()
    )));
    if view.intentions.is_empty() {
        out.push(row(
            "(none recorded)".into(),
            1,
            EpistemicKind::Observed,
            None,
        ));
    }
    for i in &view.intentions {
        out.push(row(
            format!(
                "I{} {} since_age={}",
                i.id,
                i.label,
                age(view.tick, i.since_tick)
            ),
            0,
            EpistemicKind::Hypothetical,
            Some(CogRef::Intention(i.id)),
        ));
    }
}

fn message_rows(view: &CognitiveView, out: &mut Vec<Row>) {
    out.push(header(format!("== Messages ({}) ==", view.messages.len())));
    if view.messages.is_empty() {
        out.push(row(
            "(none recorded)".into(),
            1,
            EpistemicKind::Observed,
            None,
        ));
    }
    let open = view.authorization == Authorization::Authorized;
    for m in &view.messages {
        let id = m.message.message_id();
        let a = age(view.tick, m.sent_tick);
        let text = if open {
            let kind = match m.sender_kind {
                Known::Known(k) => k.to_string(),
                Known::Unknown => "unknown".to_owned(),
            };
            format!(
                "M{} {}({kind}) -> {} {} sent_age={a}",
                id.as_u64(),
                m.message.sender(),
                m.message.recipient(),
                payload_text(m.message.payload())
            )
        } else {
            format!(
                "M{} from={REDACTED} to={REDACTED} content={REDACTED} sent_age={a}",
                id.as_u64()
            )
        };
        out.push(row(
            text,
            0,
            EpistemicKind::Observed,
            Some(CogRef::Message(id)),
        ));
    }
}

struct Builder {
    scene: Scene2D,
    next: u64,
}

impl Builder {
    fn add(&mut self, layer: LayerId, prim: ScenePrimitive, kind: EpistemicKind) {
        let node = SceneNode::new(SceneNodeId::new(self.next), None, kind, prim, true);
        self.next += 1;
        // las capas se crean de antemano; el id de nodo es único por construcción
        let _ = self.scene.add_node(layer, node);
    }
}

fn pt(x: f64, y: f64) -> Point2 {
    Point2::new(x, y).expect("finite layout point")
}

/// Proyecta la vista cognitiva a una [`Scene2D`] (coordenadas en píxeles, `y` hacia abajo).
///
/// Función pura y determinista: sólo lee `view`.
pub fn project_cognitive_scene(view: &CognitiveView) -> Scene2D {
    let mut rows = vec![header(format!(
        "Cognitive inspector: {} tick={} authorization={}",
        if view.title.is_empty() {
            "untitled"
        } else {
            &view.title
        },
        view.tick,
        match view.authorization {
            Authorization::Confidential => "confidential",
            Authorization::Authorized => "authorized",
        }
    ))];
    hypothesis_rows(view, &mut rows);
    episode_rows(view, &mut rows);
    intention_rows(view, &mut rows);
    message_rows(view, &mut rows);

    let mut b = Builder {
        scene: Scene2D::new(
            SceneId::new(96),
            Tick::new(view.tick),
            EpistemicKind::Observed,
            CoordinateSpace::meters(),
        ),
        next: 1,
    };
    for layer in [LayerId::Environment, LayerId::Decisions, LayerId::Labels] {
        b.scene
            .add_layer(SceneLayer::new(layer))
            .expect("unique layer");
    }

    let y_of = |anchor: CogRef, rows: &[Row]| {
        rows.iter()
            .position(|r| r.anchor == Some(anchor))
            .map(|i| 10.0 + i as f64 * ROW_H)
    };

    for (i, r) in rows.iter().enumerate() {
        let y = 10.0 + i as f64 * ROW_H;
        let x = 10.0 + f64::from(r.indent) * 24.0;
        if !r.header {
            b.add(
                LayerId::Environment,
                ScenePrimitive::rect(pt(x - 4.0, y - 14.0), WIDTH - x, ROW_H - 4.0)
                    .expect("finite rect"),
                r.kind,
            );
        }
        b.add(
            LayerId::Labels,
            ScenePrimitive::label(pt(x, y), r.text.clone()),
            r.kind,
        );
    }

    // enlaces: sólo los registrados y con ambos extremos presentes
    let mut unresolved = 0usize;
    for (n, l) in view.links.iter().enumerate() {
        match (y_of(l.from, &rows), y_of(l.to, &rows)) {
            (Some(y0), Some(y1)) => {
                let x = WIDTH + 6.0 + 8.0 * n as f64;
                b.add(
                    LayerId::Decisions,
                    ScenePrimitive::polyline(
                        vec![pt(WIDTH - 4.0, y0 - 6.0), pt(x, y0 - 6.0), pt(x, y1 - 6.0)],
                        false,
                    )
                    .expect("finite polyline"),
                    EpistemicKind::Observed,
                );
                b.add(
                    LayerId::Labels,
                    ScenePrimitive::label(pt(x + 4.0, y0 - 6.0), format!("L{} recorded", n + 1)),
                    EpistemicKind::Observed,
                );
            }
            _ => unresolved += 1,
        }
    }
    let base = 10.0 + rows.len() as f64 * ROW_H;
    b.add(
        LayerId::Labels,
        ScenePrimitive::label(
            pt(10.0, base),
            format!(
                "== Links: {} recorded, {} unresolved (no causality inferred) ==",
                view.links.len(),
                unresolved
            ),
        ),
        EpistemicKind::Observed,
    );
    b.scene
}

/// Renderiza la vista a SVG determinista mediante el backend de software.
pub fn render_cognitive_svg(view: &CognitiveView, width: f64, height: f64) -> String {
    let scene = project_cognitive_scene(view);
    let mut cam = Camera2D::new(width, height, false);
    if let Some(bounds) = crate::camera::scene_bounds(&scene) {
        cam.fit_to_window(width, height, &bounds, 8.0);
    }
    SoftwareSvgBackend.render_svg(&scene, &cam)
}

/// Textos (etiquetas) de la escena, en orden; útil para pruebas y volcado textual.
pub fn scene_texts(scene: &Scene2D) -> Vec<String> {
    scene
        .layers()
        .iter()
        .flat_map(|l| l.nodes())
        .filter_map(|n| match n.primitive() {
            ScenePrimitive::Label { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_fixtures::{grid_language_fixture, life_signal_fixture};

    fn texts(v: &CognitiveView) -> Vec<String> {
        scene_texts(&project_cognitive_scene(v))
    }
    fn find<'a>(t: &'a [String], prefix: &str) -> &'a String {
        t.iter()
            .find(|s| s.starts_with(prefix))
            .unwrap_or_else(|| panic!("no line {prefix}: {t:#?}"))
    }

    #[test]
    fn ac1_contradictory_hypotheses_before_and_after_update() {
        let before = texts(&grid_language_fixture(false, Authorization::Confidential));
        let after = texts(&grid_language_fixture(true, Authorization::Confidential));
        // dos hipótesis sobre el mismo hecho, marcadas como contradictorias
        assert!(find(&before, "H1").contains("CONFLICT with H2"));
        assert!(find(&before, "H2").contains("CONFLICT with H1"));
        // antes: la afirmación humana inclina hacia "libre"
        assert!(find(&before, "H1").contains("P(true)=0.300"), "{before:#?}");
        assert!(find(&before, "H2").contains("P(true)=0.700"));
        // después: el sensor directo invierte el balance (actualización de core)
        assert!(find(&after, "H1").contains("P(true)=0.794"), "{after:#?}");
        assert!(find(&after, "H2").contains("P(true)=0.206"));
        assert_ne!(before, after);
        // la edad se deriva de tick (12) y observed_tick (4 humano, 9 sensor)
        assert!(find(&before, "H1").contains("updated_age=8t"));
        assert!(find(&after, "H1").contains("updated_age=3t"));
        let human = after.iter().find(|s| s.contains("src=human")).unwrap();
        let sensor = after
            .iter()
            .find(|s| s.contains("src=direct-sensor"))
            .unwrap();
        assert!(human.contains("age=8t") && sensor.contains("age=3t"));
        // confianza declarada y fiabilidad de la fuente por separado
        assert!(
            human.contains("reported_conf=0.900") && human.contains("source_reliability=0.700")
        );
        assert!(
            sensor.contains("reported_conf=0.600") && sensor.contains("source_reliability=0.900")
        );
        assert!(human.contains("kind=assertion") && sensor.contains("kind=measurement"));
    }

    #[test]
    fn ac2_default_is_confidential_and_redacts_explicitly() {
        assert_eq!(Authorization::default(), Authorization::Confidential);
        assert_eq!(
            CognitiveView::default().authorization,
            Authorization::Confidential
        );
        let t = texts(&grid_language_fixture(true, Authorization::default()));
        let all = t.join("\n");
        let m = find(&t, "M10");
        assert!(m.contains("from=[redacted]") && m.contains("content=[redacted]"));
        let e = find(&t, "EP1");
        assert!(e.contains("summary=[redacted]") && e.contains("features=[redacted]"));
        // el contenido no aparece en ninguna parte (ni en el SVG)
        for secret in ["operator", "agent-1", "fact:", "summary_code", "3.000"] {
            assert!(!all.contains(secret), "leak {secret}");
        }
        let svg = render_cognitive_svg(
            &grid_language_fixture(true, Authorization::default()),
            1000.0,
            500.0,
        );
        assert!(!svg.contains("operator") && svg.contains("[redacted]"));
    }

    #[test]
    fn ac2_authorized_shows_content() {
        let t = texts(&grid_language_fixture(true, Authorization::Authorized));
        let m = find(&t, "M10");
        assert!(m.contains("operator(Human) -> agent-1") && m.contains("fact: cell(3,2)=free"));
        let e = find(&t, "EP1");
        assert!(e.contains("summary_code=42") && e.contains("features=(3.000, 2.000)"));
        assert!(e.contains("raw_text=redacted-at-source"));
        assert!(!t.join("\n").contains("[redacted]"));
    }

    #[test]
    fn ac2_unknown_is_explicit_never_guessed() {
        let t = texts(&life_signal_fixture(Authorization::Authorized));
        assert!(
            find(&t, "M20").contains("(unknown)") && find(&t, "M20").contains("sent_age=unknown")
        );
        assert!(find(&t, "E200").contains("source_reliability=unknown"));
        // belief desconocida y fecha futura
        let mut v = life_signal_fixture(Authorization::Authorized);
        v.hypotheses[0].belief = Known::Unknown;
        v.hypotheses[0].last_update_tick = Known::Known(999);
        v.hypotheses[0].evidence[0].says = Known::Unknown;
        let t = texts(&v);
        assert!(find(&t, "H1").contains("P(true)=unknown"));
        assert!(find(&t, "H1").contains("updated_age=unknown(future-dated)"));
        assert!(find(&t, "E200").contains("says=unknown"));
        // vista vacía: marcadores explícitos, no omisión
        let t = texts(&CognitiveView::default());
        assert_eq!(
            t.iter().filter(|s| s.as_str() == "(none recorded)").count(),
            4
        );
    }

    #[test]
    fn ac2_links_only_when_recorded_and_resolvable() {
        let polylines = |s: &Scene2D| {
            s.layers()
                .iter()
                .flat_map(|l| l.nodes())
                .filter(|n| matches!(n.primitive(), ScenePrimitive::Polyline { .. }))
                .count()
        };
        let mut v = grid_language_fixture(true, Authorization::Confidential);
        assert_eq!(polylines(&project_cognitive_scene(&v)), 2);
        v.links.clear();
        let s = project_cognitive_scene(&v);
        assert_eq!(polylines(&s), 0, "no causal link without a record");
        assert!(
            scene_texts(&s)
                .iter()
                .any(|t| t.contains("0 recorded, 0 unresolved"))
        );
        v.links.push(CausalLink {
            from: CogRef::Hypothesis(1),
            to: CogRef::Intention(99),
        });
        let s = project_cognitive_scene(&v);
        assert_eq!(polylines(&s), 0);
        assert!(
            scene_texts(&s)
                .iter()
                .any(|t| t.contains("1 recorded, 1 unresolved"))
        );
    }

    #[test]
    fn ac3_same_projector_for_two_fixtures_deterministic() {
        let project: fn(&CognitiveView) -> Scene2D = project_cognitive_scene;
        let a = grid_language_fixture(true, Authorization::Authorized);
        let b = life_signal_fixture(Authorization::Authorized);
        let (sa, sb) = (project(&a), project(&b));
        assert_ne!(sa.to_canonical_json(), sb.to_canonical_json());
        assert_eq!(sa.to_canonical_json(), project(&a).to_canonical_json());
        assert!(scene_texts(&sa)[0].contains("grid language-world"));
        assert!(scene_texts(&sb)[0].contains("life signal channel"));
        assert_eq!(
            render_cognitive_svg(&b, 800.0, 400.0),
            render_cognitive_svg(&b, 800.0, 400.0)
        );
        for s in [&sa, &sb] {
            for sec in [
                "== Beliefs",
                "== Episodes",
                "== Intentions",
                "== Messages",
                "== Links",
            ] {
                assert!(scene_texts(s).iter().any(|t| t.starts_with(sec)), "{sec}");
            }
        }
    }

    #[test]
    fn ac3_projector_is_data_only_no_network_llm_or_dynamics() {
        let src = include_str!("cognitive.rs");
        let prod: String = src[..src.find("#[cfg(test)]").unwrap()]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in [
            "std::net",
            "std::fs",
            "TcpStream",
            "http",
            "Llm",
            "LLM",
            "SystemOne",
            "Planner",
            "GenerativeModel",
            "CommunicationPort",
            "LocalCommunicationBus",
            "Actuator",
            "ControlPort",
            "SessionHost",
            "gogamen_lab_",
            "update_with_observation",
            ".step(",
            ".apply(",
            "step_grid_dynamics",
            "async ",
        ] {
            assert!(!prod.contains(banned), "cognitive.rs contains `{banned}`");
        }
        assert!(prod.contains("pub fn project_cognitive_scene(view: &CognitiveView) -> Scene2D"));
    }
}
