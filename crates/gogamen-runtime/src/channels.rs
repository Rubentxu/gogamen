//! Canales acotados sin bloqueo y control de caducidad para el runtime (`std`-only).
//!
//! Este módulo implementa el transporte acotado entre productores críticos y consumidores
//! potencialmente lentos, garantizando que el productor nunca se bloquee y que los datos
//! caducados o de revisiones obsoletas nunca sustituyan una observación vigente:
//! - [`ChannelConfig`]: configuración de capacidad ([`NonZeroUsize`]), política de
//!   saturación ([`ChannelOverflowPolicy`]) y caducidad opcional (`max_age_nanos`),
//!   resuelta de forma inmutable al preparar el canal mediante [`ChannelConfig::prepare`]
//!   o [`BoundedChannel::prepare`].
//! - [`ChannelMessage<T>`]: envuelve una carga útil `T` con marca temporal monotónica
//!   (`timestamp`), número de revisión (`revision`) y caducidad explícita opcional
//!   (`expires_at`).
//! - [`BoundedChannel<T>`]: cola acotada preasignada en `prepare` cuyas operaciones de
//!   envío (`try_send`) ejecutan en O(1) sin bloquear ni esperar al consumidor,
//!   contabilizando cada pérdida o rechazo en [`ChannelStats`].

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use gogamen_core::{MonotonicClock, MonotonicInstant, ObservationEnvelope};

/// Política de saturación aplicada en O(1) cuando el canal alcanza su capacidad máxima.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelOverflowPolicy {
    /// Descarta el elemento más antiguo de la cola y conserva el entrante más reciente.
    Latest,
    /// Descarta el elemento entrante preservando los ya encolados.
    DropNewest,
    /// Rechaza explícitamente el elemento entrante devolviendo [`ChannelError::Full`].
    Reject,
}

/// Configuración de un canal acotado, resuelta al preparar ([`ChannelConfig::prepare`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelConfig {
    /// Capacidad máxima estrictamente positiva de mensajes en cola.
    pub capacity: NonZeroUsize,
    /// Política aplicada cuando el canal está lleno durante `try_send`.
    pub overflow_policy: ChannelOverflowPolicy,
    /// Edad máxima permitida en nanosegundos (`now - timestamp <= max_age_nanos`), si aplica.
    pub max_age_nanos: Option<u128>,
}

impl ChannelConfig {
    /// Crea una configuración de canal acotado con capacidad no nula explícita.
    pub const fn new(
        capacity: NonZeroUsize,
        overflow_policy: ChannelOverflowPolicy,
        max_age_nanos: Option<u128>,
    ) -> Self {
        Self {
            capacity,
            overflow_policy,
            max_age_nanos,
        }
    }

    /// Crea una configuración validando que `capacity > 0`.
    pub fn try_new(
        capacity: usize,
        overflow_policy: ChannelOverflowPolicy,
        max_age_nanos: Option<u128>,
    ) -> Result<Self, ChannelError> {
        let capacity = NonZeroUsize::new(capacity).ok_or(ChannelError::ZeroCapacity)?;
        Ok(Self {
            capacity,
            overflow_policy,
            max_age_nanos,
        })
    }

    /// Establece la edad máxima permitida en nanosegundos monotónicos.
    pub const fn with_max_age_nanos(mut self, max_age_nanos: u128) -> Self {
        self.max_age_nanos = Some(max_age_nanos);
        self
    }

    /// Establece la edad máxima permitida a partir de una [`Duration`].
    pub const fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age_nanos = Some(max_age.as_nanos());
        self
    }

    /// Resuelve la capacidad y la política preparando una instancia de [`BoundedChannel`].
    pub fn prepare<T>(self) -> BoundedChannel<T> {
        BoundedChannel::prepare(self)
    }
}

/// Errores explícitos de configuración o rechazo en [`BoundedChannel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelError {
    /// Se intentó construir una configuración con capacidad `0`.
    ZeroCapacity,
    /// El canal está lleno y la política configurada es [`ChannelOverflowPolicy::Reject`].
    Full,
    /// El mensaje está caducado en el instante de envío (`now > expires_at` o `age > max_age_nanos`).
    Stale {
        /// Instante monotónico de origen del mensaje.
        timestamp: MonotonicInstant,
        /// Instante monotónico actual al intentar enviar.
        now: MonotonicInstant,
        /// Instante explícito de expiración del mensaje, si tenía uno.
        expires_at: Option<MonotonicInstant>,
        /// Edad máxima configurada en el canal en nanosegundos, si existe.
        max_age_nanos: Option<u128>,
    },
    /// La revisión del mensaje es estrictamente anterior a la última revisión vigente aceptada.
    OldRevision {
        /// Revisión del mensaje entrante.
        revision: u64,
        /// Última revisión vigente aceptada en el canal.
        latest_revision: u64,
    },
}

impl fmt::Display for ChannelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCapacity => write!(f, "channel capacity must be greater than zero"),
            Self::Full => write!(f, "bounded channel is full (overflow policy: Reject)"),
            Self::Stale {
                timestamp,
                now,
                expires_at,
                max_age_nanos,
            } => write!(
                f,
                "stale channel message rejected at {now} (timestamp={timestamp}, expires_at={expires_at:?}, max_age_nanos={max_age_nanos:?})"
            ),
            Self::OldRevision {
                revision,
                latest_revision,
            } => write!(
                f,
                "obsolete message revision {revision} rejected (latest_revision={latest_revision})"
            ),
        }
    }
}

impl Error for ChannelError {}

/// Resultado exitoso de [`BoundedChannel::try_send`] sin bloqueo del productor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SendOutcome {
    /// El mensaje se encoló directamente porque había capacidad disponible.
    Enqueued,
    /// El canal estaba lleno bajo [`ChannelOverflowPolicy::Latest`]: se descartó el
    /// elemento más antiguo en O(1) y se encoló el nuevo mensaje.
    EnqueuedAfterDropOldest,
    /// El canal estaba lleno bajo [`ChannelOverflowPolicy::DropNewest`]: el mensaje
    /// entrante fue descartado sin bloquear al productor ni alterar los ya encolados.
    DroppedNewest,
}

/// Mensaje tipado con metadatos temporales y de revisión para canales acotados.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelMessage<T> {
    /// Carga útil transportada por el mensaje.
    pub payload: T,
    /// Instante monotónico de creación u observación del dato.
    pub timestamp: MonotonicInstant,
    /// Revisión monotónica del dato u observación.
    pub revision: u64,
    /// Instante monotónico explícito a partir del cual el dato caduca (`now > expires_at`).
    pub expires_at: Option<MonotonicInstant>,
}

impl<T> ChannelMessage<T> {
    /// Construye un mensaje vigente sin expiración explícita por mensaje.
    pub const fn new(payload: T, timestamp: MonotonicInstant, revision: u64) -> Self {
        Self {
            payload,
            timestamp,
            revision,
            expires_at: None,
        }
    }

    /// Asocia un instante absoluto de caducidad monotónica (`expires_at`).
    pub const fn with_expires_at(mut self, expires_at: MonotonicInstant) -> Self {
        self.expires_at = Some(expires_at);
        self
    }

    /// Asocia un tiempo de vida relativo a `timestamp` en nanosegundos.
    pub const fn with_ttl_nanos(mut self, ttl_nanos: u128) -> Self {
        self.expires_at = Some(MonotonicInstant::from_nanos(
            self.timestamp.as_nanos().saturating_add(ttl_nanos),
        ));
        self
    }

    /// Asocia un tiempo de vida relativo a `timestamp` usando [`Duration`].
    pub const fn with_ttl(self, ttl: Duration) -> Self {
        self.with_ttl_nanos(ttl.as_nanos())
    }

    /// Evalúa si el mensaje está caducado en el instante `now` dada una edad máxima opcional.
    ///
    /// Un mensaje se considera caducado si:
    /// - `now > expires_at` (cuando `expires_at` es `Some`), o
    /// - `now - timestamp > max_age_nanos` (cuando `max_age_nanos` es `Some`).
    pub const fn is_expired_at(&self, now: MonotonicInstant, max_age_nanos: Option<u128>) -> bool {
        if let Some(expires_at) = self.expires_at
            && now.as_nanos() > expires_at.as_nanos()
        {
            return true;
        }
        if let Some(max_age) = max_age_nanos
            && now.as_nanos().saturating_sub(self.timestamp.as_nanos()) > max_age
        {
            return true;
        }
        false
    }
}

impl<T> ChannelMessage<ObservationEnvelope<T>> {
    /// Construye un [`ChannelMessage`] a partir de un [`ObservationEnvelope`], tomando
    /// `observed_at()` como `timestamp` y `revision()` como `revision`.
    pub const fn from_envelope(
        envelope: ObservationEnvelope<T>,
        expires_at: Option<MonotonicInstant>,
    ) -> Self {
        let timestamp = envelope.observed_at();
        let revision = envelope.revision();
        Self {
            payload: envelope,
            timestamp,
            revision,
            expires_at,
        }
    }
}

/// Contadores observables de tráfico, saturación, caducidad y rechazos del canal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ChannelStats {
    /// Mensajes aceptados y encolados correctamente.
    pub sent_ok: u64,
    /// Mensajes antiguos descartados de la cola por saturación bajo [`ChannelOverflowPolicy::Latest`].
    pub dropped_oldest: u64,
    /// Mensajes entrantes descartados por saturación bajo [`ChannelOverflowPolicy::DropNewest`].
    pub dropped_newest: u64,
    /// Mensajes rechazados por canal lleno bajo [`ChannelOverflowPolicy::Reject`].
    pub rejected_full: u64,
    /// Mensajes rechazados al enviar por estar ya caducados.
    pub rejected_stale: u64,
    /// Mensajes rechazados al enviar por tener `revision < latest_revision`.
    pub rejected_old_revision: u64,
    /// Mensajes descartados automáticamente al consumir por haber caducado en cola.
    pub expired_in_queue: u64,
    /// Mensajes vigentes entregados correctamente al consumidor.
    pub received_ok: u64,
}

impl ChannelStats {
    /// Total de mensajes perdidos por descarte (saturación o caducidad en cola).
    pub const fn total_dropped(&self) -> u64 {
        self.dropped_oldest + self.dropped_newest + self.expired_in_queue
    }

    /// Total de mensajes rechazados en el intento de envío.
    pub const fn total_rejected(&self) -> u64 {
        self.rejected_full + self.rejected_stale + self.rejected_old_revision
    }

    /// Total agregado de pérdidas y rechazos observables en el canal.
    pub const fn total_losses(&self) -> u64 {
        self.total_dropped() + self.total_rejected()
    }
}

/// Canal acotado sin bloqueo del productor con control de caducidad y revisión monotónica.
///
/// La capacidad y la política de saturación se resuelven de forma inmutable al invocar
/// [`BoundedChannel::prepare`] (o [`ChannelConfig::prepare`]), preasignando el búfer
/// interno para evitar realojamientos durante el bucle crítico.
#[derive(Debug)]
pub struct BoundedChannel<T> {
    config: ChannelConfig,
    queue: VecDeque<ChannelMessage<T>>,
    latest_revision: Option<u64>,
    stats: ChannelStats,
}

impl<T> BoundedChannel<T> {
    /// Prepara un canal acotado resolviendo capacidad, política de saturación y caducidad.
    pub fn prepare(config: ChannelConfig) -> Self {
        Self {
            queue: VecDeque::with_capacity(config.capacity.get()),
            config,
            latest_revision: None,
            stats: ChannelStats::default(),
        }
    }

    /// Devuelve la configuración inmutable resuelta al preparar el canal.
    pub const fn config(&self) -> ChannelConfig {
        self.config
    }

    /// Devuelve la capacidad máxima resuelta del canal.
    pub const fn capacity(&self) -> NonZeroUsize {
        self.config.capacity
    }

    /// Devuelve la política de saturación resuelta del canal.
    pub const fn overflow_policy(&self) -> ChannelOverflowPolicy {
        self.config.overflow_policy
    }

    /// Devuelve la edad máxima configurada en nanosegundos, si existe.
    pub const fn max_age_nanos(&self) -> Option<u128> {
        self.config.max_age_nanos
    }

    /// Devuelve el número actual de mensajes en cola.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Indica si la cola está vacía.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Indica si la cola ha alcanzado su capacidad máxima.
    pub fn is_full(&self) -> bool {
        self.queue.len() == self.config.capacity.get()
    }

    /// Devuelve la última revisión vigente aceptada por el canal, si existe.
    pub const fn latest_revision(&self) -> Option<u64> {
        self.latest_revision
    }

    /// Devuelve una instantánea de los contadores observables del canal.
    pub const fn stats(&self) -> ChannelStats {
        self.stats
    }

    /// Intenta enviar `msg` en el instante monotónico `now` sin bloquear al productor.
    ///
    /// Reglas aplicadas en O(1):
    /// 1. Si `msg` ya está caducado (`now > expires_at` o `now - timestamp > max_age_nanos`),
    ///    incrementa `rejected_stale` y devuelve [`ChannelError::Stale`] sin modificar la cola
    ///    ni `latest_revision`.
    /// 2. Si `msg.revision < latest_revision`, incrementa `rejected_old_revision` y devuelve
    ///    [`ChannelError::OldRevision`] sin sustituir ninguna observación vigente.
    /// 3. Si la cola está llena, aplica [`ChannelOverflowPolicy`] en O(1) sin esperar al
    ///    consumidor.
    pub fn try_send(
        &mut self,
        msg: ChannelMessage<T>,
        now: MonotonicInstant,
    ) -> Result<SendOutcome, ChannelError> {
        if msg.is_expired_at(now, self.config.max_age_nanos) {
            self.stats.rejected_stale += 1;
            return Err(ChannelError::Stale {
                timestamp: msg.timestamp,
                now,
                expires_at: msg.expires_at,
                max_age_nanos: self.config.max_age_nanos,
            });
        }

        if let Some(latest) = self.latest_revision
            && msg.revision < latest
        {
            self.stats.rejected_old_revision += 1;
            return Err(ChannelError::OldRevision {
                revision: msg.revision,
                latest_revision: latest,
            });
        }

        let revision = msg.revision;

        if self.queue.len() == self.config.capacity.get() {
            match self.config.overflow_policy {
                ChannelOverflowPolicy::Latest => {
                    let _ = self.queue.pop_front();
                    self.stats.dropped_oldest += 1;
                    self.queue.push_back(msg);
                    self.latest_revision = Some(revision);
                    self.stats.sent_ok += 1;
                    Ok(SendOutcome::EnqueuedAfterDropOldest)
                }
                ChannelOverflowPolicy::DropNewest => {
                    self.stats.dropped_newest += 1;
                    Ok(SendOutcome::DroppedNewest)
                }
                ChannelOverflowPolicy::Reject => {
                    self.stats.rejected_full += 1;
                    Err(ChannelError::Full)
                }
            }
        } else {
            self.queue.push_back(msg);
            self.latest_revision = Some(revision);
            self.stats.sent_ok += 1;
            Ok(SendOutcome::Enqueued)
        }
    }

    /// Envía `msg` leyendo el instante actual de `clock` sin bloquear al productor.
    pub fn try_send_with_clock<C: MonotonicClock + ?Sized>(
        &mut self,
        msg: ChannelMessage<T>,
        clock: &C,
    ) -> Result<SendOutcome, ChannelError> {
        self.try_send(msg, clock.now())
    }

    /// Intenta recibir el siguiente mensaje vigente en el instante `now`.
    ///
    /// Descarta automáticamente cualquier entrada en cabeza de cola que haya caducado
    /// mientras esperaba, incrementando `expired_in_queue` por cada descarte.
    pub fn try_recv(&mut self, now: MonotonicInstant) -> Option<ChannelMessage<T>> {
        while let Some(msg) = self.queue.pop_front() {
            if msg.is_expired_at(now, self.config.max_age_nanos) {
                self.stats.expired_in_queue += 1;
                continue;
            }
            self.stats.received_ok += 1;
            return Some(msg);
        }
        None
    }

    /// Intenta recibir el siguiente mensaje vigente leyendo el instante actual de `clock`.
    pub fn try_recv_with_clock<C: MonotonicClock + ?Sized>(
        &mut self,
        clock: &C,
    ) -> Option<ChannelMessage<T>> {
        self.try_recv(clock.now())
    }

    /// Purga explícitamente todas las entradas en cola que ya hayan caducado en `now`,
    /// contabilizándolas en `expired_in_queue` y devolviendo cuántas se eliminaron.
    pub fn purge_expired(&mut self, now: MonotonicInstant) -> usize {
        let before = self.queue.len();
        let max_age = self.config.max_age_nanos;
        self.queue.retain(|msg| !msg.is_expired_at(now, max_age));
        let removed = before - self.queue.len();
        self.stats.expired_in_queue += removed as u64;
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use gogamen_core::{
        Confidence, EvidenceRef, FrameId, MessageId, ObservationKind, ProvenanceMetadata, SensorId,
        UnitTag, VirtualMonotonicClock,
    };

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("non-zero capacity in test")
    }

    #[test]
    fn config_resolves_capacity_and_policy_at_prepare_and_rejects_zero_capacity() {
        assert_eq!(
            ChannelConfig::try_new(0, ChannelOverflowPolicy::Latest, None),
            Err(ChannelError::ZeroCapacity)
        );

        let cfg = ChannelConfig::try_new(4, ChannelOverflowPolicy::Latest, None)
            .unwrap()
            .with_max_age_nanos(500);
        let channel: BoundedChannel<u64> = cfg.prepare();

        assert_eq!(channel.config(), cfg);
        assert_eq!(channel.capacity(), nz(4));
        assert_eq!(channel.overflow_policy(), ChannelOverflowPolicy::Latest);
        assert_eq!(channel.max_age_nanos(), Some(500));
        assert!(channel.is_empty());
        assert!(!channel.is_full());
        assert_eq!(channel.latest_revision(), None);
        assert_eq!(channel.stats(), ChannelStats::default());
    }

    #[test]
    fn saturation_latest_evicts_oldest_and_keeps_most_recent_without_blocking() {
        let clock = VirtualMonotonicClock::new();
        let mut ch = ChannelConfig::new(nz(3), ChannelOverflowPolicy::Latest, None).prepare();

        for i in 1..=10_u64 {
            let now = clock.advance_nanos(10);
            let outcome = ch.try_send(ChannelMessage::new(i, now, i), now).unwrap();
            if i <= 3 {
                assert_eq!(outcome, SendOutcome::Enqueued);
            } else {
                assert_eq!(outcome, SendOutcome::EnqueuedAfterDropOldest);
            }
            assert!(ch.len() <= 3);
        }

        let stats = ch.stats();
        assert_eq!(stats.sent_ok, 10);
        assert_eq!(stats.dropped_oldest, 7);
        assert_eq!(stats.dropped_newest, 0);
        assert_eq!(stats.rejected_full, 0);
        assert_eq!(stats.total_losses(), 7);
        assert_eq!(ch.len(), 3);
        assert!(ch.is_full());

        let now = clock.now();
        let received: Vec<u64> = (0..4)
            .filter_map(|_| ch.try_recv(now).map(|m| m.payload))
            .collect();
        assert_eq!(received, vec![8, 9, 10]);
        assert_eq!(ch.stats().received_ok, 3);
    }

    #[test]
    fn saturation_drop_newest_preserves_queued_items_and_counts_drops() {
        let clock = VirtualMonotonicClock::new();
        let mut ch = ChannelConfig::new(nz(2), ChannelOverflowPolicy::DropNewest, None).prepare();

        let t1 = clock.advance_nanos(10);
        assert_eq!(
            ch.try_send(ChannelMessage::new("first", t1, 1), t1),
            Ok(SendOutcome::Enqueued)
        );
        let t2 = clock.advance_nanos(10);
        assert_eq!(
            ch.try_send(ChannelMessage::new("second", t2, 2), t2),
            Ok(SendOutcome::Enqueued)
        );

        // Saturación: los entrantes se descartan sin bloquear ni alterar la cola.
        for rev in 3..=6_u64 {
            let t = clock.advance_nanos(10);
            assert_eq!(
                ch.try_send(ChannelMessage::new("dropped", t, rev), t),
                Ok(SendOutcome::DroppedNewest)
            );
        }

        let stats = ch.stats();
        assert_eq!(stats.sent_ok, 2);
        assert_eq!(stats.dropped_newest, 4);
        assert_eq!(stats.dropped_oldest, 0);
        assert_eq!(stats.rejected_full, 0);
        assert_eq!(ch.latest_revision(), Some(2));

        let now = clock.now();
        assert_eq!(ch.try_recv(now).unwrap().payload, "first");
        assert_eq!(ch.try_recv(now).unwrap().payload, "second");
        assert_eq!(ch.try_recv(now), None);
    }

    #[test]
    fn saturation_reject_returns_full_error_and_counts_rejections() {
        let clock = VirtualMonotonicClock::new();
        let mut ch = ChannelConfig::new(nz(2), ChannelOverflowPolicy::Reject, None).prepare();

        let t1 = clock.advance_nanos(10);
        assert_eq!(
            ch.try_send(ChannelMessage::new(100_u32, t1, 1), t1),
            Ok(SendOutcome::Enqueued)
        );
        let t2 = clock.advance_nanos(10);
        assert_eq!(
            ch.try_send(ChannelMessage::new(200_u32, t2, 2), t2),
            Ok(SendOutcome::Enqueued)
        );

        // Saturación: devuelve ChannelError::Full y contabiliza rejected_full.
        for rev in 3..=5_u64 {
            let t = clock.advance_nanos(10);
            assert_eq!(
                ch.try_send(ChannelMessage::new(300_u32, t, rev), t),
                Err(ChannelError::Full)
            );
        }

        let stats = ch.stats();
        assert_eq!(stats.sent_ok, 2);
        assert_eq!(stats.rejected_full, 3);
        assert_eq!(stats.dropped_oldest, 0);
        assert_eq!(stats.dropped_newest, 0);
        assert_eq!(ch.len(), 2);
        assert_eq!(ch.latest_revision(), Some(2));

        // Al liberar un hueco, vuelve a aceptar envíos normalmente.
        let now = clock.now();
        assert_eq!(ch.try_recv(now).unwrap().payload, 100);
        assert_eq!(
            ch.try_send(ChannelMessage::new(400_u32, now, 3), now),
            Ok(SendOutcome::Enqueued)
        );
        assert_eq!(ch.try_recv(now).unwrap().payload, 200);
        assert_eq!(ch.try_recv(now).unwrap().payload, 400);
    }

    #[test]
    fn slow_consumer_never_blocks_critical_producer_and_losses_are_observable() {
        // Test concurrente real: un consumidor lento duerme entre lecturas mientras
        // el productor emite una ráfaga rápida sin quedar bloqueado esperando al consumidor.
        let shared = Arc::new(Mutex::new(
            ChannelConfig::new(nz(4), ChannelOverflowPolicy::Latest, None).prepare(),
        ));
        let producer_ch = Arc::clone(&shared);
        let consumer_ch = Arc::clone(&shared);

        let producer = thread::spawn(move || {
            for seq in 1..=200_u64 {
                let now = MonotonicInstant::from_nanos(u128::from(seq) * 1_000);
                let mut guard = producer_ch.lock().unwrap();
                guard
                    .try_send(ChannelMessage::new(seq, now, seq), now)
                    .expect("Latest policy never fails on full");
            }
        });

        let consumer = thread::spawn(move || {
            let mut consumed = Vec::new();
            for step in 0..5_u64 {
                thread::sleep(Duration::from_millis(2));
                let now = MonotonicInstant::from_nanos(u128::from(step + 1) * 50_000);
                let mut guard = consumer_ch.lock().unwrap();
                if let Some(msg) = guard.try_recv(now) {
                    consumed.push(msg.payload);
                }
            }
            consumed
        });

        producer
            .join()
            .expect("producer must finish without blocking");
        let consumed = consumer.join().expect("consumer must finish");

        let guard = shared.lock().unwrap();
        let stats = guard.stats();
        assert_eq!(stats.sent_ok, 200);
        assert_eq!(stats.received_ok, consumed.len() as u64);
        // Conservación exacta: enviados = recibidos + en cola + descartados por saturación.
        assert_eq!(
            stats.sent_ok,
            stats.received_ok + guard.len() as u64 + stats.dropped_oldest
        );
        assert!(
            stats.dropped_oldest > 0,
            "slow consumer must trigger observable drops"
        );
    }

    #[test]
    fn stale_and_obsolete_revision_messages_never_replace_valid_observation() {
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000));
        let mut ch = ChannelConfig::new(nz(1), ChannelOverflowPolicy::Latest, Some(100)).prepare();

        // 1. Insertamos una observación vigente (rev=10, timestamp=1000, expires_at=1080).
        let valid_msg = ChannelMessage::new("valid_rev_10", clock.now(), 10)
            .with_expires_at(MonotonicInstant::from_nanos(1_080));
        assert_eq!(
            ch.try_send_with_clock(valid_msg, &clock),
            Ok(SendOutcome::Enqueued)
        );

        // Avanzamos 50 ns -> t=1050 (la observación vigente sigue viva).
        clock.advance_nanos(50);

        // 2. Mensaje caducado por max_age_nanos (age = 1050 - 900 = 150 > 100) con revisión mayor (rev=11).
        let stale_by_age =
            ChannelMessage::new("stale_age_rev_11", MonotonicInstant::from_nanos(900), 11);
        assert!(matches!(
            ch.try_send_with_clock(stale_by_age, &clock),
            Err(ChannelError::Stale { .. })
        ));

        // 3. Mensaje caducado por expires_at explícito (expires_at=1049 < now=1050) con revisión mayor (rev=12).
        let stale_by_exp = ChannelMessage::new("stale_exp_rev_12", clock.now(), 12)
            .with_expires_at(MonotonicInstant::from_nanos(1_049));
        assert!(matches!(
            ch.try_send_with_clock(stale_by_exp, &clock),
            Err(ChannelError::Stale { .. })
        ));

        // 4. Mensaje vigente en tiempo pero con revisión obsoleta (rev=9 < latest_revision=10).
        let old_rev = ChannelMessage::new("old_rev_9", clock.now(), 9);
        assert_eq!(
            ch.try_send_with_clock(old_rev, &clock),
            Err(ChannelError::OldRevision {
                revision: 9,
                latest_revision: 10,
            })
        );

        // Verificamos que ninguna de las 3 entradas inválidas sustituyó la observación vigente
        // ni alteró latest_revision ni incrementó dropped_oldest.
        let stats = ch.stats();
        assert_eq!(stats.sent_ok, 1);
        assert_eq!(stats.dropped_oldest, 0);
        assert_eq!(stats.rejected_stale, 2);
        assert_eq!(stats.rejected_old_revision, 1);
        assert_eq!(ch.latest_revision(), Some(10));

        let delivered = ch
            .try_recv_with_clock(&clock)
            .expect("valid obs must remain");
        assert_eq!(delivered.payload, "valid_rev_10");
        assert_eq!(delivered.revision, 10);

        // Incluso tras vaciar la cola, una revisión anterior sigue siendo rechazada.
        let late_old_rev = ChannelMessage::new("late_rev_8", clock.now(), 8);
        assert_eq!(
            ch.try_send_with_clock(late_old_rev, &clock),
            Err(ChannelError::OldRevision {
                revision: 8,
                latest_revision: 10,
            })
        );
        assert_eq!(ch.stats().rejected_old_revision, 2);
        assert!(ch.is_empty());
    }

    #[test]
    fn try_recv_discards_entries_expired_in_queue_and_counts_them() {
        let clock = VirtualMonotonicClock::at(MonotonicInstant::from_nanos(1_000));
        let mut ch = ChannelConfig::new(nz(4), ChannelOverflowPolicy::Reject, Some(200)).prepare();

        // m1: caduca por expires_at en 1050.
        let m1 = ChannelMessage::new("m1_short_ttl", clock.now(), 1).with_ttl_nanos(50);
        ch.try_send_with_clock(m1, &clock).unwrap();

        // m2: sin expires_at, pero timestamp=1000 -> caduca por max_age=200 en t > 1200.
        let m2 = ChannelMessage::new("m2_max_age", clock.now(), 2);
        ch.try_send_with_clock(m2, &clock).unwrap();

        // Avanzamos a t=1150 y enviamos m3 (vigente hasta 1350).
        clock.advance_nanos(150);
        let m3 = ChannelMessage::new("m3_fresh", clock.now(), 3);
        ch.try_send_with_clock(m3, &clock).unwrap();

        // Avanzamos a t=1250: m1 (exp 1050) y m2 (age 250 > 200) caducaron en cola; m3 (age 100 <= 200) sigue vigente.
        clock.advance_nanos(100);
        let received = ch
            .try_recv_with_clock(&clock)
            .expect("m3 must be delivered");
        assert_eq!(received.payload, "m3_fresh");
        assert_eq!(received.revision, 3);
        assert_eq!(ch.try_recv_with_clock(&clock), None);

        let stats = ch.stats();
        assert_eq!(stats.sent_ok, 3);
        assert_eq!(stats.expired_in_queue, 2);
        assert_eq!(stats.received_ok, 1);
    }

    #[test]
    fn exact_boundary_of_expiry_is_still_valid_and_one_nanosecond_later_is_expired() {
        let mut ch = ChannelConfig::new(nz(2), ChannelOverflowPolicy::Reject, Some(100)).prepare();
        let t0 = MonotonicInstant::from_nanos(500);

        // expires_at exacto en 550.
        let msg = ChannelMessage::new("boundary", t0, 1)
            .with_expires_at(MonotonicInstant::from_nanos(550));
        assert_eq!(
            ch.try_send(msg.clone(), MonotonicInstant::from_nanos(550)),
            Ok(SendOutcome::Enqueued)
        );

        // En t=550 sigue vigente para recibir.
        assert_eq!(
            ch.try_recv(MonotonicInstant::from_nanos(550))
                .unwrap()
                .payload,
            "boundary"
        );

        // En t=551 (1 ns después de expires_at) se rechaza como caducado.
        assert!(matches!(
            ch.try_send(msg, MonotonicInstant::from_nanos(551)),
            Err(ChannelError::Stale { .. })
        ));

        // Borde exacto de max_age_nanos: timestamp=500, max_age=100 -> t=600 válido, t=601 caducado.
        let msg_age = ChannelMessage::new("age_boundary", t0, 2);
        assert_eq!(
            ch.try_send(msg_age.clone(), MonotonicInstant::from_nanos(600)),
            Ok(SendOutcome::Enqueued)
        );
        // En t=601 caduca en cola.
        assert_eq!(ch.try_recv(MonotonicInstant::from_nanos(601)), None);
        assert_eq!(ch.stats().expired_in_queue, 1);
    }

    #[test]
    fn observation_envelope_bridge_preserves_timestamp_and_revision() {
        let mut ch: BoundedChannel<ObservationEnvelope<f64>> =
            ChannelConfig::new(nz(2), ChannelOverflowPolicy::Latest, Some(500)).prepare();

        let meta = ProvenanceMetadata {
            message_id: MessageId::new(1),
            observed_at: MonotonicInstant::from_nanos(1_000),
            received_at: MonotonicInstant::from_nanos(1_020),
            sensor_id: SensorId::new(7),
            frame_id: FrameId::world(),
            units: UnitTag::dimensionless(),
            confidence: Confidence::FULL,
            revision: 42,
            evidence_ref: EvidenceRef::new(9),
            claim_kind: ObservationKind::Measurement,
        };
        let env = ObservationEnvelope::new(meta, 3.25_f64);
        let msg = ChannelMessage::from_envelope(env, Some(MonotonicInstant::from_nanos(1_200)));

        assert_eq!(msg.timestamp, MonotonicInstant::from_nanos(1_000));
        assert_eq!(msg.revision, 42);

        ch.try_send(msg, MonotonicInstant::from_nanos(1_100))
            .unwrap();
        let recv = ch.try_recv(MonotonicInstant::from_nanos(1_150)).unwrap();
        assert_eq!(recv.revision, 42);
        assert_eq!(*recv.payload.payload(), 3.25_f64);
    }

    #[test]
    fn purge_expired_removes_only_expired_entries_and_updates_stats() {
        let mut ch = ChannelConfig::new(nz(4), ChannelOverflowPolicy::Reject, Some(100)).prepare();
        let t0 = MonotonicInstant::from_nanos(100);
        let t1 = MonotonicInstant::from_nanos(180);

        ch.try_send(ChannelMessage::new("old", t0, 1), t0).unwrap();
        ch.try_send(ChannelMessage::new("fresh", t1, 2), t1)
            .unwrap();

        let removed = ch.purge_expired(MonotonicInstant::from_nanos(250));
        assert_eq!(removed, 1);
        assert_eq!(ch.len(), 1);
        assert_eq!(ch.stats().expired_in_queue, 1);

        let item = ch.try_recv(MonotonicInstant::from_nanos(250)).unwrap();
        assert_eq!(item.payload, "fresh");
    }
}
