//! Canal de comunicación arbitraria (`A-D`) en `gogamen-lab-life` (`std`-only + `gogamen-core`).
//!
//! Implementa un protocolo experimental de señalización discreta entre organismos Life
//! **sin Jev dentro de los organismos**, reutilizando directamente:
//! - `gogamen_core::communication`: [`CommunicationPort`], [`LocalCommunicationBus`],
//!   [`SituatedMessage`], [`CommPayload`], [`CommCost`], [`InterlocutorId`], [`ConversationScope`].
//! - `gogamen_core::learning`: [`TabularTransitionLearner`], [`LearningPolicy`],
//!   [`TransitionObservation`], [`ObservationId`], [`gogamen_core::IncrementalLearner`].
//! - `LifeWorld` y [`LifeAction`] para ejecutar el forrajeo físico y el balance energético.
//!
//! # Alcance epistémico
//! Este módulo evalúa únicamente **señalización discreta útil** (asociación aprendida sobre
//! símbolos atómicos `A | B | C | D` con utilidad causal positiva neta de coste), y
//! **no** afirma ni implementa **lenguaje composicional**.

use std::fmt;
use std::time::Duration as StdDuration;

use gogamen_core::{
    CommCost, CommPayload, CommunicationError, CommunicationPort, ConversationScope,
    DeterministicRng, EvidenceRef, InterlocutorId, LearningError, LearningPolicy,
    LocalCommunicationBus, MasterSeed, MessageId, MonotonicInstant, ObservationId, RngSource,
    SituatedMessage, StreamKey, TabularTransitionLearner, TransitionObservation,
};

use crate::config::{LifeError, LifeWorldConfig};
use crate::organism::LifeAction;
use crate::world::LifeWorld;

/// Alfabeto discreto de comunicación arbitraria en `gogamen-lab-life`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Symbol {
    /// Símbolo discreto `A`.
    A,
    /// Símbolo discreto `B`.
    B,
    /// Símbolo discreto `C`.
    C,
    /// Símbolo discreto `D`.
    D,
}

impl Symbol {
    /// Todos los símbolos del alfabeto en orden canónico.
    pub const ALL: [Self; 4] = [Self::A, Self::B, Self::C, Self::D];

    /// Índice discreto `0..4` del símbolo.
    pub const fn index(self) -> usize {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::C => 2,
            Self::D => 3,
        }
    }

    /// Construye un [`Symbol`] a partir de su índice `0..4`.
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::A),
            1 => Some(Self::B),
            2 => Some(Self::C),
            3 => Some(Self::D),
            _ => None,
        }
    }

    /// Representación textual canónica (`"A"`, `"B"`, `"C"` o `"D"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
        }
    }

    /// Parsea un código textual canónico (`"A".."D"`).
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "A" => Some(Self::A),
            "B" => Some(Self::B),
            "C" => Some(Self::C),
            "D" => Some(Self::D),
            _ => None,
        }
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Dirección cardinal oculta de alimento que el emisor observa y el receptor desconoce sin señal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ForageCue {
    /// Alimento ubicado al norte del receptor.
    North,
    /// Alimento ubicado al sur del receptor.
    South,
    /// Alimento ubicado al este del receptor.
    East,
    /// Alimento ubicado al oeste del receptor.
    West,
}

impl ForageCue {
    /// Todas las direcciones cardinales en orden canónico.
    pub const ALL: [Self; 4] = [Self::North, Self::South, Self::East, Self::West];

    /// Índice `0..4` de la dirección.
    pub const fn index(self) -> usize {
        match self {
            Self::North => 0,
            Self::South => 1,
            Self::East => 2,
            Self::West => 3,
        }
    }

    /// Construye una dirección a partir de su índice `0..4`.
    pub const fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(Self::North),
            1 => Some(Self::South),
            2 => Some(Self::East),
            3 => Some(Self::West),
            _ => None,
        }
    }

    /// Acción física equivalente en [`LifeWorld`].
    pub const fn to_life_action(self) -> LifeAction {
        match self {
            Self::North => LifeAction::MoveUp,
            Self::South => LifeAction::MoveDown,
            Self::East => LifeAction::MoveRight,
            Self::West => LifeAction::MoveLeft,
        }
    }

    /// Celda objetivo `(x, y)` en una grilla `3x3` con el receptor centrado en `(1, 1)`.
    pub const fn target_cell_3x3(self) -> (usize, usize) {
        match self {
            Self::North => (1, 0),
            Self::South => (1, 2),
            Self::East => (2, 1),
            Self::West => (0, 1),
        }
    }
}

/// Código arbitrario (biyección `ForageCue -> Symbol`) del emisor.
///
/// El receptor nunca recibe este diccionario; debe aprender la asociación
/// exclusivamente mediante observación incremental con [`TabularTransitionLearner`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignalCodebook {
    cue_to_symbol: [Symbol; 4],
}

impl SignalCodebook {
    /// Código canónico por defecto (`North->A, South->B, East->C, West->D`).
    pub const fn identity() -> Self {
        Self {
            cue_to_symbol: Symbol::ALL,
        }
    }

    /// Genera una permutación arbitraria determinista de `A..D` a partir de una semilla.
    pub fn from_seed(seed: MasterSeed) -> Self {
        let mut rng = seed.derive(StreamKey::episode(1));
        let mut syms = Symbol::ALL;
        for i in (1..syms.len()).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            syms.swap(i, j);
        }
        Self {
            cue_to_symbol: syms,
        }
    }

    /// Símbolo asignado a una dirección de alimento.
    pub const fn symbol_for(&self, cue: ForageCue) -> Symbol {
        self.cue_to_symbol[cue.index()]
    }
}

/// Variante experimental del canal de señalización.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalMode {
    /// Baseline sin señal: no se envían mensajes ni se paga coste comunicativo.
    NoSignal,
    /// Señal aleatoria: se envían símbolos aleatorios `A..D` pagando coste comunicativo.
    RandomSignal,
    /// Canal aprendido asociativamente sobre [`LocalCommunicationBus`] y [`TabularTransitionLearner`].
    LearnedChannel,
}

impl fmt::Display for SignalMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSignal => f.write_str("NoSignal"),
            Self::RandomSignal => f.write_str("RandomSignal"),
            Self::LearnedChannel => f.write_str("LearnedChannel"),
        }
    }
}

/// Tipo de intervención causal aplicada sobre los símbolos recibidos en evaluación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalInterventionKind {
    /// Permuta/baraja los símbolos recibidos conservando exactamente su distribución marginal.
    PermuteSymbols,
    /// Interviene fijando el símbolo recibido a una constante.
    ConstantSymbol(Symbol),
}

impl fmt::Display for CausalInterventionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PermuteSymbols => f.write_str("PermuteSymbols"),
            Self::ConstantSymbol(sym) => write!(f, "ConstantSymbol({sym})"),
        }
    }
}

/// Errores explícitos del experimento de comunicación en `gogamen-lab-life`.
#[derive(Debug, Clone, PartialEq)]
pub enum LifeCommError {
    /// Parámetro de configuración inválido.
    InvalidConfig(&'static str),
    /// Error del bus o validación de comunicación situada en `gogamen-core`.
    Communication(CommunicationError),
    /// Error del aprendiz tabular en `gogamen-core`.
    Learning(LearningError),
    /// Error del entorno físico [`LifeWorld`].
    Life(LifeError),
}

impl fmt::Display for LifeCommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "invalid life communication config: {msg}"),
            Self::Communication(err) => write!(f, "communication error: {err}"),
            Self::Learning(err) => write!(f, "learning error: {err}"),
            Self::Life(err) => write!(f, "life world error: {err}"),
        }
    }
}

impl std::error::Error for LifeCommError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidConfig(_) => None,
            Self::Communication(err) => Some(err),
            Self::Learning(err) => Some(err),
            Self::Life(err) => Some(err),
        }
    }
}

impl From<CommunicationError> for LifeCommError {
    fn from(value: CommunicationError) -> Self {
        Self::Communication(value)
    }
}

impl From<LearningError> for LifeCommError {
    fn from(value: LearningError) -> Self {
        Self::Learning(value)
    }
}

impl From<LifeError> for LifeCommError {
    fn from(value: LifeError) -> Self {
        Self::Life(value)
    }
}

/// Configuración determinista del experimento de comunicación arbitraria en Life.
#[derive(Debug, Clone, PartialEq)]
pub struct LifeCommConfig {
    /// Alcance máximo del canal en distancia Manhattan.
    pub range: usize,
    /// Distancia Manhattan efectiva entre emisor y receptor.
    pub sender_distance: usize,
    /// Coste explícito [`CommCost`] por cada mensaje enviado.
    pub cost_per_message: CommCost,
    /// Factor de conversión de `CommCost::units()` a coste energético escalar.
    pub energy_cost_per_unit: f64,
    /// Factor de conversión de `CommCost::tokens()` a coste energético escalar.
    pub energy_cost_per_token: f64,
    /// Alimento ganado cuando el receptor elige la dirección correcta.
    pub food_reward: f64,
    /// Número de rondas de aprendizaje asociativo previo a la evaluación.
    pub training_rounds: usize,
    /// Número de rondas de evaluación sobre las que se comparan variantes e intervenciones.
    pub eval_rounds: usize,
    /// Presupuesto máximo de mensajes emitibles durante la fase de evaluación.
    pub max_eval_messages: usize,
    /// Tasa de exploración `epsilon` en `[0.0, 1.0]` durante el entrenamiento.
    pub exploration_rate: f64,
    /// Parámetro `prior_alpha` de Dirichlet/Laplace para [`TabularTransitionLearner`].
    pub prior_alpha: f64,
}

impl Default for LifeCommConfig {
    fn default() -> Self {
        Self {
            range: 2,
            sender_distance: 1,
            cost_per_message: CommCost::new(1, 1, StdDuration::from_millis(1)),
            energy_cost_per_unit: 0.2,
            energy_cost_per_token: 0.05,
            food_reward: 2.0,
            training_rounds: 120,
            eval_rounds: 80,
            max_eval_messages: 80,
            exploration_rate: 0.2,
            prior_alpha: 1.0,
        }
    }
}

impl LifeCommConfig {
    /// Valida los invariantes de configuración.
    pub fn validate(&self) -> Result<(), LifeCommError> {
        if !self.energy_cost_per_unit.is_finite() || self.energy_cost_per_unit < 0.0 {
            return Err(LifeCommError::InvalidConfig(
                "energy_cost_per_unit must be finite and >= 0.0",
            ));
        }
        if !self.energy_cost_per_token.is_finite() || self.energy_cost_per_token < 0.0 {
            return Err(LifeCommError::InvalidConfig(
                "energy_cost_per_token must be finite and >= 0.0",
            ));
        }
        if !self.food_reward.is_finite() || self.food_reward <= 0.0 {
            return Err(LifeCommError::InvalidConfig(
                "food_reward must be finite and > 0.0",
            ));
        }
        if self.eval_rounds == 0 {
            return Err(LifeCommError::InvalidConfig("eval_rounds must be >= 1"));
        }
        if !self.exploration_rate.is_finite()
            || self.exploration_rate < 0.0
            || self.exploration_rate > 1.0
        {
            return Err(LifeCommError::InvalidConfig(
                "exploration_rate must be finite in [0.0, 1.0]",
            ));
        }
        if !self.prior_alpha.is_finite() || self.prior_alpha <= 0.0 {
            return Err(LifeCommError::InvalidConfig(
                "prior_alpha must be finite and > 0.0",
            ));
        }
        Ok(())
    }

    /// Coste energético escalar de un [`CommCost`].
    pub fn scalar_energy_cost(&self, cost: CommCost) -> f64 {
        (cost.units() as f64) * self.energy_cost_per_unit
            + (cost.tokens() as f64) * self.energy_cost_per_token
    }
}

/// Métricas agregadas de una variante en la fase de evaluación.
#[derive(Debug, Clone, PartialEq)]
pub struct VariantMetrics {
    /// Variante evaluada.
    pub mode: SignalMode,
    /// Rondas de evaluación ejecutadas.
    pub eval_rounds: usize,
    /// Aciertos de forrajeo en evaluación.
    pub hits: usize,
    /// Tasa de acierto (`hits / eval_rounds`).
    pub hit_rate: f64,
    /// Mensajes enviados durante evaluación.
    pub messages_sent: usize,
    /// Mensajes entregados dentro de `range`.
    pub messages_delivered: usize,
    /// Mensajes descartados por superar `range`.
    pub out_of_range_drops: usize,
    /// Rondas donde no se envió por agotarse `max_eval_messages`.
    pub budget_exhausted_skips: usize,
    /// Coste acumulado de comunicación en [`CommCost`].
    pub total_comm_cost: CommCost,
    /// Coste energético escalar descontado por comunicación.
    pub comm_energy_cost: f64,
    /// Utilidad bruta (energía de alimento recolectada en [`LifeWorld`]).
    pub gross_utility: f64,
    /// Utilidad neta (`gross_utility - comm_energy_cost`).
    pub net_utility: f64,
    /// Versión final del modelo asociativo [`TabularTransitionLearner`].
    pub learner_model_version: u64,
    /// Checksum determinista final del modelo asociativo.
    pub learner_checksum: u64,
}

/// Resultado de medición causal del canal mediante intervención sobre los símbolos.
#[derive(Debug, Clone, PartialEq)]
pub struct CausalUtilityReport {
    /// Intervención causal aplicada.
    pub intervention_kind: CausalInterventionKind,
    /// Utilidad bruta bajo el canal observado (sin intervención).
    pub observed_gross_utility: f64,
    /// Utilidad bruta bajo la intervención causal.
    pub intervened_gross_utility: f64,
    /// Utilidad causal bruta (`observed_gross_utility - intervened_gross_utility`).
    pub gross_causal_utility: f64,
    /// Coste energético de comunicación descontado.
    pub comm_energy_cost: f64,
    /// Utilidad causal neta (`gross_causal_utility - comm_energy_cost`).
    pub net_causal_utility: f64,
    /// Tasa de acierto observada.
    pub observed_hit_rate: f64,
    /// Tasa de acierto bajo intervención.
    pub intervened_hit_rate: f64,
    /// Distribución marginal de símbolos `[A, B, C, D]` observada e intervenida.
    pub marginal_symbol_counts: [usize; 4],
}

/// Evaluación epistémica explícita que separa señal útil de lenguaje composicional.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpistemicChannelAssessment {
    /// `true` si la utilidad causal neta de coste es estrictamente positiva.
    pub useful_signal_detected: bool,
    /// Siempre `false`: este experimento no implementa ni afirma lenguaje composicional.
    pub compositional_language_claimed: bool,
    /// Explicación explícita de lo demostrado y sus límites epistémicos.
    pub summary: String,
}

/// Reporte completo del experimento comparativo y causal de comunicación Life.
#[derive(Debug, Clone, PartialEq)]
pub struct CommunicationExperimentReport {
    /// Semilla maestra usada para todas las variantes e intervenciones.
    pub seed: MasterSeed,
    /// Configuración del experimento.
    pub config: LifeCommConfig,
    /// Diccionario arbitrario del emisor (oculto al receptor).
    pub codebook: SignalCodebook,
    /// Resultados de la variante `NoSignal`.
    pub no_signal: VariantMetrics,
    /// Resultados de la variante `RandomSignal`.
    pub random_signal: VariantMetrics,
    /// Resultados de la variante `LearnedChannel`.
    pub learned_channel: VariantMetrics,
    /// Utilidad causal medida por permutación de símbolos (misma distribución marginal).
    pub causal_permutation: CausalUtilityReport,
    /// Utilidad causal medida por intervención a símbolo constante.
    pub causal_constant_intervention: CausalUtilityReport,
    /// Evaluación epistémica explícita (señal útil vs lenguaje composicional).
    pub epistemic_assessment: EpistemicChannelAssessment,
}

impl CommunicationExperimentReport {
    /// Renderiza un informe legible con métricas comparativas, utilidad causal y distinción epistémica.
    pub fn render_report(&self) -> String {
        format!(
            "=== Life Arbitrary Communication Report (seed={}) ===\n\
             NoSignal:       hit_rate={:.3}, gross={:.3}, comm_cost={:.3}, net={:.3}\n\
             RandomSignal:   hit_rate={:.3}, gross={:.3}, comm_cost={:.3}, net={:.3}\n\
             LearnedChannel: hit_rate={:.3}, gross={:.3}, comm_cost={:.3}, net={:.3}, delivered={}/{}\n\
             Causal ({}): gross_causal={:.3}, comm_cost={:.3}, net_causal={:.3} (hit_rate {:.3} -> {:.3})\n\
             Causal ({}): gross_causal={:.3}, comm_cost={:.3}, net_causal={:.3} (hit_rate {:.3} -> {:.3})\n\
             Epistemic: useful_signal_detected={}, compositional_language_claimed={}\n\
             Note: {}",
            self.seed.as_u64(),
            self.no_signal.hit_rate,
            self.no_signal.gross_utility,
            self.no_signal.comm_energy_cost,
            self.no_signal.net_utility,
            self.random_signal.hit_rate,
            self.random_signal.gross_utility,
            self.random_signal.comm_energy_cost,
            self.random_signal.net_utility,
            self.learned_channel.hit_rate,
            self.learned_channel.gross_utility,
            self.learned_channel.comm_energy_cost,
            self.learned_channel.net_utility,
            self.learned_channel.messages_delivered,
            self.learned_channel.messages_sent,
            self.causal_permutation.intervention_kind,
            self.causal_permutation.gross_causal_utility,
            self.causal_permutation.comm_energy_cost,
            self.causal_permutation.net_causal_utility,
            self.causal_permutation.observed_hit_rate,
            self.causal_permutation.intervened_hit_rate,
            self.causal_constant_intervention.intervention_kind,
            self.causal_constant_intervention.gross_causal_utility,
            self.causal_constant_intervention.comm_energy_cost,
            self.causal_constant_intervention.net_causal_utility,
            self.causal_constant_intervention.observed_hit_rate,
            self.causal_constant_intervention.intervened_hit_rate,
            self.epistemic_assessment.useful_signal_detected,
            self.epistemic_assessment.compositional_language_claimed,
            self.epistemic_assessment.summary,
        )
    }
}

/// Ejecuta el experimento completo comparando `NoSignal`, `RandomSignal` y `LearnedChannel`
/// sobre la misma semilla y presupuesto, midiendo utilidad causal neta de coste y
/// generando la evaluación epistémica explícita.
pub fn run_communication_experiment(
    seed: MasterSeed,
    config: &LifeCommConfig,
) -> Result<CommunicationExperimentReport, LifeCommError> {
    config.validate()?;

    let codebook = SignalCodebook::from_seed(seed);
    let mut cue_rng = seed.derive(StreamKey::episode(2));
    let total_rounds = config.training_rounds + config.eval_rounds;
    let mut cues = Vec::with_capacity(total_rounds);
    for _ in 0..total_rounds {
        let idx = (cue_rng.next_u64() as usize) % ForageCue::ALL.len();
        cues.push(ForageCue::ALL[idx]);
    }
    let train_cues = &cues[..config.training_rounds];
    let eval_cues = &cues[config.training_rounds..];

    let (no_signal, _) = run_variant(
        seed,
        config,
        codebook,
        SignalMode::NoSignal,
        train_cues,
        eval_cues,
    )?;
    let (random_signal, _) = run_variant(
        seed,
        config,
        codebook,
        SignalMode::RandomSignal,
        train_cues,
        eval_cues,
    )?;
    let (learned_channel, learned_artifacts) = run_variant(
        seed,
        config,
        codebook,
        SignalMode::LearnedChannel,
        train_cues,
        eval_cues,
    )?;

    let causal_permutation = evaluate_causal_intervention(
        seed,
        config,
        eval_cues,
        &learned_channel,
        &learned_artifacts.learner,
        &learned_artifacts.eval_symbols,
        CausalInterventionKind::PermuteSymbols,
    )?;

    let causal_constant_intervention = evaluate_causal_intervention(
        seed,
        config,
        eval_cues,
        &learned_channel,
        &learned_artifacts.learner,
        &learned_artifacts.eval_symbols,
        CausalInterventionKind::ConstantSymbol(Symbol::A),
    )?;

    let useful_signal_detected = causal_permutation.gross_causal_utility > 0.0
        && causal_permutation.net_causal_utility > 0.0;

    let summary = if useful_signal_detected {
        format!(
            "Se detecta señal útil: asociación aprendida sobre símbolos discretos A-D \
             (net_causal_utility={:.3} tras descontar coste {:.3}). \
             No se afirma lenguaje composicional: no existe sintaxis, gramática combinatoria \
             ni composición jerárquica de significados.",
            causal_permutation.net_causal_utility, causal_permutation.comm_energy_cost
        )
    } else {
        format!(
            "Resultado neutro/negativo de señal útil: gross_causal_utility={:.3}, \
             comm_energy_cost={:.3}, net_causal_utility={:.3} (p. ej. coste comunicativo \
             superior a la ganancia o mensajes fuera de alcance). \
             Tampoco se afirma lenguaje composicional.",
            causal_permutation.gross_causal_utility,
            causal_permutation.comm_energy_cost,
            causal_permutation.net_causal_utility
        )
    };

    Ok(CommunicationExperimentReport {
        seed,
        config: config.clone(),
        codebook,
        no_signal,
        random_signal,
        learned_channel,
        causal_permutation,
        causal_constant_intervention,
        epistemic_assessment: EpistemicChannelAssessment {
            useful_signal_detected,
            compositional_language_claimed: false,
            summary,
        },
    })
}

struct VariantArtifacts {
    learner: TabularTransitionLearner,
    eval_symbols: Vec<Option<Symbol>>,
}

fn run_variant(
    seed: MasterSeed,
    config: &LifeCommConfig,
    codebook: SignalCodebook,
    mode: SignalMode,
    train_cues: &[ForageCue],
    eval_cues: &[ForageCue],
) -> Result<(VariantMetrics, VariantArtifacts), LifeCommError> {
    let policy = LearningPolicy::new(config.prior_alpha)?;
    let mut learner =
        TabularTransitionLearner::new(Symbol::ALL.len(), ForageCue::ALL.len(), policy)?;
    let mut bus = LocalCommunicationBus::new(16, 16)?;
    let sender_id = InterlocutorId::new("life-sender")?;
    let receiver_id = InterlocutorId::new("life-receiver")?;
    let scope = ConversationScope::channel("life-symbols-abcd")?;

    let mut action_rng = seed.derive(StreamKey::agent(1, 1));
    let mut random_sig_rng = seed.derive(StreamKey::agent(1, 2));
    let mut next_id: u64 = 1;

    // Fase de entrenamiento online (sin límite de presupuesto de evaluación).
    for (round_idx, &cue) in train_cues.iter().enumerate() {
        let emitted = match mode {
            SignalMode::NoSignal => None,
            SignalMode::RandomSignal => {
                let idx = (random_sig_rng.next_u64() as usize) % Symbol::ALL.len();
                Some(Symbol::ALL[idx])
            }
            SignalMode::LearnedChannel => Some(codebook.symbol_for(cue)),
        };

        let delivered = if let Some(sym) = emitted {
            if config.sender_distance <= config.range {
                send_symbol_message(
                    &mut bus,
                    &mut next_id,
                    round_idx as u64,
                    &sender_id,
                    &receiver_id,
                    &scope,
                    sym,
                    config.cost_per_message,
                )?;
                poll_symbol_message(&mut bus, &receiver_id)
            } else {
                None
            }
        } else {
            None
        };

        let state_bucket = delivered.map_or(0, Symbol::index);
        let action_idx = choose_action(
            &learner,
            state_bucket,
            delivered.is_some(),
            config.exploration_rate,
            &mut action_rng,
        )?;
        let chosen_cue = ForageCue::ALL[action_idx];
        let reward =
            execute_forage_tick(seed, round_idx as u64, cue, chosen_cue, config.food_reward)?;
        let hit = reward > 0.0;

        if delivered.is_some() || mode == SignalMode::NoSignal {
            let obs_id = ObservationId::new(next_id);
            let ev_ref = EvidenceRef::new(next_id);
            next_id += 1;
            let obs = TransitionObservation::new(
                obs_id,
                ev_ref,
                state_bucket,
                action_idx,
                usize::from(hit),
            );
            learner.observe(&obs)?;
        }
    }

    // Fase de evaluación bajo presupuesto acotado `max_eval_messages`.
    let mut hits = 0usize;
    let mut messages_sent = 0usize;
    let mut messages_delivered = 0usize;
    let mut out_of_range_drops = 0usize;
    let mut budget_exhausted_skips = 0usize;
    let mut total_comm_cost = CommCost::ZERO;
    let mut gross_utility = 0.0;
    let mut eval_symbols = Vec::with_capacity(eval_cues.len());

    for (eval_idx, &cue) in eval_cues.iter().enumerate() {
        let round_num = (train_cues.len() + eval_idx) as u64;
        let planned_symbol = match mode {
            SignalMode::NoSignal => None,
            SignalMode::RandomSignal => {
                let idx = (random_sig_rng.next_u64() as usize) % Symbol::ALL.len();
                Some(Symbol::ALL[idx])
            }
            SignalMode::LearnedChannel => Some(codebook.symbol_for(cue)),
        };

        let delivered = if let Some(sym) = planned_symbol {
            if messages_sent >= config.max_eval_messages {
                budget_exhausted_skips += 1;
                None
            } else {
                messages_sent += 1;
                total_comm_cost = total_comm_cost.saturating_add(config.cost_per_message);
                if config.sender_distance <= config.range {
                    send_symbol_message(
                        &mut bus,
                        &mut next_id,
                        round_num,
                        &sender_id,
                        &receiver_id,
                        &scope,
                        sym,
                        config.cost_per_message,
                    )?;
                    let rx = poll_symbol_message(&mut bus, &receiver_id);
                    if rx.is_some() {
                        messages_delivered += 1;
                    }
                    rx
                } else {
                    out_of_range_drops += 1;
                    None
                }
            }
        } else {
            None
        };

        eval_symbols.push(delivered);
        let state_bucket = delivered.map_or(0, Symbol::index);
        let action_idx = choose_action(
            &learner,
            state_bucket,
            delivered.is_some(),
            0.0,
            &mut action_rng,
        )?;
        let chosen_cue = ForageCue::ALL[action_idx];
        let reward = execute_forage_tick(seed, round_num, cue, chosen_cue, config.food_reward)?;
        if reward > 0.0 {
            hits += 1;
            gross_utility += reward;
        }
    }

    let comm_energy_cost = config.scalar_energy_cost(total_comm_cost);
    let net_utility = gross_utility - comm_energy_cost;
    let hit_rate = (hits as f64) / (eval_cues.len() as f64);

    let metrics = VariantMetrics {
        mode,
        eval_rounds: eval_cues.len(),
        hits,
        hit_rate,
        messages_sent,
        messages_delivered,
        out_of_range_drops,
        budget_exhausted_skips,
        total_comm_cost,
        comm_energy_cost,
        gross_utility,
        net_utility,
        learner_model_version: learner.model_version(),
        learner_checksum: learner.checksum(),
    };

    Ok((
        metrics,
        VariantArtifacts {
            learner,
            eval_symbols,
        },
    ))
}

fn evaluate_causal_intervention(
    seed: MasterSeed,
    config: &LifeCommConfig,
    eval_cues: &[ForageCue],
    observed_metrics: &VariantMetrics,
    learner: &TabularTransitionLearner,
    observed_symbols: &[Option<Symbol>],
    intervention: CausalInterventionKind,
) -> Result<CausalUtilityReport, LifeCommError> {
    let mut marginal_symbol_counts = [0usize; 4];
    for sym in observed_symbols.iter().flatten() {
        marginal_symbol_counts[sym.index()] += 1;
    }

    let intervened_symbols: Vec<Option<Symbol>> = match intervention {
        CausalInterventionKind::PermuteSymbols => {
            let mut shuffled = observed_symbols.to_vec();
            let mut rng = seed.derive(StreamKey::episode(3));
            for i in (1..shuffled.len()).rev() {
                let j = (rng.next_u64() as usize) % (i + 1);
                shuffled.swap(i, j);
            }
            shuffled
        }
        CausalInterventionKind::ConstantSymbol(sym) => observed_symbols
            .iter()
            .map(|entry| entry.map(|_| sym))
            .collect(),
    };

    let mut action_rng = seed.derive(StreamKey::agent(2, 1));
    let mut intervened_hits = 0usize;
    let mut intervened_gross_utility = 0.0;

    for (eval_idx, (&cue, &sym_opt)) in eval_cues.iter().zip(intervened_symbols.iter()).enumerate()
    {
        let round_num = (config.training_rounds + eval_idx) as u64;
        let state_bucket = sym_opt.map_or(0, Symbol::index);
        let action_idx = choose_action(
            learner,
            state_bucket,
            sym_opt.is_some(),
            0.0,
            &mut action_rng,
        )?;
        let reward = execute_forage_tick(
            seed,
            round_num,
            cue,
            ForageCue::ALL[action_idx],
            config.food_reward,
        )?;
        if reward > 0.0 {
            intervened_hits += 1;
            intervened_gross_utility += reward;
        }
    }

    let intervened_hit_rate = (intervened_hits as f64) / (eval_cues.len() as f64);
    let gross_causal_utility = observed_metrics.gross_utility - intervened_gross_utility;
    let net_causal_utility = gross_causal_utility - observed_metrics.comm_energy_cost;

    Ok(CausalUtilityReport {
        intervention_kind: intervention,
        observed_gross_utility: observed_metrics.gross_utility,
        intervened_gross_utility,
        gross_causal_utility,
        comm_energy_cost: observed_metrics.comm_energy_cost,
        net_causal_utility,
        observed_hit_rate: observed_metrics.hit_rate,
        intervened_hit_rate,
        marginal_symbol_counts,
    })
}

#[allow(clippy::too_many_arguments)]
fn send_symbol_message(
    bus: &mut LocalCommunicationBus,
    next_id: &mut u64,
    round_num: u64,
    sender: &InterlocutorId,
    recipient: &InterlocutorId,
    scope: &ConversationScope,
    symbol: Symbol,
    cost: CommCost,
) -> Result<(), LifeCommError> {
    let msg_id = MessageId::new(*next_id);
    let ev_ref = EvidenceRef::new(*next_id);
    *next_id += 1;

    let payload = CommPayload::StructuredFact {
        fact_key: "life.symbol".to_string(),
        fact_value: symbol.as_str().to_string(),
        attributes: std::collections::BTreeMap::new(),
        confidence: None,
        evidence: Some(ev_ref),
    };

    let msg = SituatedMessage::new(
        msg_id,
        sender.clone(),
        recipient.clone(),
        scope.clone(),
        MonotonicInstant::from_nanos(u128::from((round_num + 1) * 1_000_000)),
        cost,
        payload,
    )?;
    bus.send(msg)?;
    Ok(())
}

fn poll_symbol_message(
    bus: &mut LocalCommunicationBus,
    recipient: &InterlocutorId,
) -> Option<Symbol> {
    let msgs = bus.poll_inbox(recipient, 1);
    let msg = msgs.into_iter().next()?;
    if let CommPayload::StructuredFact { fact_value, .. } = msg.payload() {
        Symbol::from_code(fact_value)
    } else {
        None
    }
}

fn choose_action(
    learner: &TabularTransitionLearner,
    state_bucket: usize,
    has_signal: bool,
    exploration_rate: f64,
    rng: &mut DeterministicRng,
) -> Result<usize, LifeCommError> {
    if !has_signal {
        // Sin señal recibida no hay diferenciación simbólica: elección uniforme determinista.
        return Ok((rng.next_u64() as usize) % ForageCue::ALL.len());
    }

    let u01 = ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64);
    if u01 < exploration_rate {
        return Ok((rng.next_u64() as usize) % ForageCue::ALL.len());
    }

    let mut best_action = 0usize;
    let mut best_prob = -1.0f64;
    for action_idx in 0..ForageCue::ALL.len() {
        let probs = learner.predict(state_bucket, action_idx)?;
        let hit_prob = probs[1];
        if hit_prob > best_prob + 1e-12 {
            best_prob = hit_prob;
            best_action = action_idx;
        }
    }
    Ok(best_action)
}

fn execute_forage_tick(
    seed: MasterSeed,
    round_num: u64,
    food_cue: ForageCue,
    chosen_cue: ForageCue,
    food_reward: f64,
) -> Result<f64, LifeCommError> {
    let world_cfg = LifeWorldConfig {
        width: 3,
        height: 3,
        initial_population: 0,
        max_population: 2,
        initial_energy: 5.0,
        max_energy: 20.0,
        basal_cost: 0.0,
        move_cost: 0.0,
        max_eat_per_tick: food_reward,
        max_cell_food: food_reward,
        initial_cell_food: 0.0,
        food_regen_per_cell: 0.0,
        food_regen_prob: 0.0,
        vision_radius: 0,
        reproduction_threshold: 15.0,
        reproduction_cost: 5.0,
        reproduction_overhead: 0.0,
        mutation_rate: 0.0,
        mutation_scale: 0.0,
    };
    let round_seed = MasterSeed::new(seed.as_u64().wrapping_add(round_num + 1));
    let mut world = LifeWorld::empty(world_cfg, round_seed)?;
    world.spawn_organism(1, 1, 5.0)?;
    let (fx, fy) = food_cue.target_cell_3x3();
    world.set_cell_food(fx, fy, food_reward)?;

    let report = world.step_tick(&[chosen_cue.to_life_action()])?;
    Ok(report.food_consumed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ac1_arbitrary_symbol_channel_outperforms_no_signal_and_random_signal() {
        let seed = MasterSeed::new(2_026_056);
        let config = LifeCommConfig::default();

        let report_a = run_communication_experiment(seed, &config).unwrap();
        let report_b = run_communication_experiment(seed, &config).unwrap();
        assert_eq!(report_a, report_b);

        assert!(
            report_a.learned_channel.hit_rate > 0.85,
            "expected learned channel hit_rate > 0.85, got {}",
            report_a.learned_channel.hit_rate
        );
        assert!(
            report_a.learned_channel.net_utility > report_a.no_signal.net_utility,
            "learned net_utility ({}) should beat no_signal ({})",
            report_a.learned_channel.net_utility,
            report_a.no_signal.net_utility
        );
        assert!(
            report_a.no_signal.net_utility > report_a.random_signal.net_utility,
            "no_signal ({}) should beat random_signal ({}) due to wasted CommCost",
            report_a.no_signal.net_utility,
            report_a.random_signal.net_utility
        );
        assert_eq!(report_a.no_signal.comm_energy_cost, 0.0);
        assert!(report_a.random_signal.comm_energy_cost > 0.0);
        assert!(report_a.learned_channel.learner_model_version > 0);
    }

    #[test]
    fn ac1_range_limit_drops_out_of_range_messages() {
        let seed = MasterSeed::new(99);
        let config = LifeCommConfig {
            range: 1,
            sender_distance: 3,
            ..LifeCommConfig::default()
        };
        let report = run_communication_experiment(seed, &config).unwrap();
        assert_eq!(report.learned_channel.messages_delivered, 0);
        assert_eq!(
            report.learned_channel.out_of_range_drops,
            report.learned_channel.messages_sent
        );
        assert!(!report.epistemic_assessment.useful_signal_detected);
        assert!(!report.epistemic_assessment.compositional_language_claimed);
    }

    #[test]
    fn ac2_causal_utility_by_permutation_and_constant_intervention_accounts_for_cost() {
        let seed = MasterSeed::new(4242);
        let config = LifeCommConfig::default();
        let report = run_communication_experiment(seed, &config).unwrap();

        // La permutación conserva exactamente la distribución marginal de símbolos A-D.
        let total_marginal: usize = report
            .causal_permutation
            .marginal_symbol_counts
            .iter()
            .sum();
        assert_eq!(total_marginal, config.eval_rounds);
        assert!(
            report.causal_permutation.gross_causal_utility > 0.0,
            "gross_causal_utility should be > 0.0, got {}",
            report.causal_permutation.gross_causal_utility
        );
        assert!(
            report.causal_permutation.net_causal_utility > 0.0,
            "net_causal_utility should be > 0.0, got {}",
            report.causal_permutation.net_causal_utility
        );
        assert!(
            (report.causal_permutation.net_causal_utility
                - (report.causal_permutation.gross_causal_utility
                    - report.causal_permutation.comm_energy_cost))
                .abs()
                < 1e-12
        );

        assert!(report.causal_constant_intervention.gross_causal_utility > 0.0);
        assert!(report.causal_constant_intervention.net_causal_utility > 0.0);

        // Si el coste comunicativo supera la ganancia causal, net_causal_utility es negativo/neutro.
        let expensive_cfg = LifeCommConfig {
            energy_cost_per_unit: 3.0,
            energy_cost_per_token: 1.0,
            ..config
        };
        let expensive_report = run_communication_experiment(seed, &expensive_cfg).unwrap();
        assert!(expensive_report.causal_permutation.gross_causal_utility > 0.0);
        assert!(expensive_report.causal_permutation.net_causal_utility <= 0.0);
        assert!(!expensive_report.epistemic_assessment.useful_signal_detected);
    }

    #[test]
    fn ac3_report_distinguishes_useful_discrete_signal_from_compositional_language() {
        let seed = MasterSeed::new(777);
        let report = run_communication_experiment(seed, &LifeCommConfig::default()).unwrap();

        assert!(report.epistemic_assessment.useful_signal_detected);
        assert!(!report.epistemic_assessment.compositional_language_claimed);

        let rendered = report.render_report();
        assert!(rendered.contains("compositional_language_claimed=false"));
        assert!(rendered.contains("lenguaje composicional"));
        assert!(rendered.contains("señal útil"));
    }
}
