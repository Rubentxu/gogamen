//! Robot-side low-level controller with watchdog. Independent of the remote planner: it only sees
//! messages that actually arrive, enforces limits locally and can only make the robot less mobile.

use crate::contract::{Mode, Odom, Twist};

/// Axis-aligned geofence in the odometry frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Geofence {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
}

impl Geofence {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.x_min && x <= self.x_max && y >= self.y_min && y <= self.y_max
    }
}

/// Conservative defaults (see README); every value is enforced locally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalConfig {
    /// No accepted command for this long -> safe stop (not latched).
    pub cmd_timeout_ms: u64,
    /// No heartbeat for this long -> e-stop (latched).
    pub heartbeat_timeout_ms: u64,
    /// Commands older than this when received (`now - stamp`) are dropped.
    pub cmd_deadline_ms: u64,
    /// No odometry for this long -> safe stop (geofence cannot be evaluated).
    pub odom_timeout_ms: u64,
    pub max_linear: f64,
    pub max_angular: f64,
    pub geofence: Geofence,
    /// Motion is predicted this far ahead against the geofence; blocked if it would leave.
    pub lookahead_ms: u64,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            cmd_timeout_ms: 300,
            heartbeat_timeout_ms: 500,
            cmd_deadline_ms: 150,
            odom_timeout_ms: 200,
            max_linear: 0.3,
            max_angular: 0.5,
            geofence: Geofence { x_min: -2.0, x_max: 2.0, y_min: -2.0, y_max: 2.0 },
            lookahead_ms: 500,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopCause {
    AwaitingHeartbeat,
    NoCommand,
    CommandTimeout,
    OdomStale,
    /// Forward motion would leave the geofence within the lookahead.
    GeofenceLookahead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EStopCause {
    HeartbeatLost,
    GeofenceBreach,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Run,
    Stop(StopCause),
    EStop(EStopCause),
}

impl Status {
    pub fn mode(&self) -> Mode {
        match self {
            Status::Run => Mode::Run,
            Status::Stop(_) => Mode::Stop,
            Status::EStop(_) => Mode::EStop,
        }
    }
}

/// What the controller applies to the drive this tick.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Applied {
    pub twist: Twist,
    pub status: Status,
    /// State of the independent e-stop line (would cut drive power in a real robot).
    pub estop_line: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CmdVerdict {
    Accepted,
    DroppedLate,
    DroppedNonFinite,
    DroppedLatched,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RearmRefused {
    NotLatched,
    HeartbeatNotFresh,
    OdomNotFresh,
    OutsideGeofence,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub accepted: u64,
    pub dropped_late: u64,
    pub dropped_non_finite: u64,
    pub clamped: u64,
}

#[derive(Debug, Clone)]
pub struct LocalController {
    cfg: LocalConfig,
    latch: Option<EStopCause>,
    last_cmd: Option<(Twist, u64)>,
    last_hb: Option<u64>,
    odom: Option<(Odom, u64)>,
    pub stats: Stats,
}

impl LocalController {
    pub fn new(cfg: LocalConfig) -> Self {
        Self { cfg, latch: None, last_cmd: None, last_hb: None, odom: None, stats: Stats::default() }
    }

    pub fn config(&self) -> &LocalConfig {
        &self.cfg
    }

    pub fn latched(&self) -> Option<EStopCause> {
        self.latch
    }

    /// Heartbeat received at local time `now`.
    pub fn on_heartbeat(&mut self, now: u64) {
        self.last_hb = Some(now);
    }

    /// Local odometry sample received at `now`.
    pub fn on_odom(&mut self, now: u64, odom: Odom) {
        self.odom = Some((odom, now));
    }

    /// Command (stamped by the planner at `stamp_ms`) received at `now`.
    pub fn on_cmd(&mut self, now: u64, stamp_ms: u64, cmd: Twist) -> CmdVerdict {
        if self.latch.is_some() {
            return CmdVerdict::DroppedLatched;
        }
        if !cmd.is_finite() {
            self.stats.dropped_non_finite += 1;
            return CmdVerdict::DroppedNonFinite;
        }
        if now.saturating_sub(stamp_ms) > self.cfg.cmd_deadline_ms {
            self.stats.dropped_late += 1;
            return CmdVerdict::DroppedLate;
        }
        let limited = self.limit(cmd);
        if limited != Twist::planar(cmd.linear.x, cmd.angular.z) {
            self.stats.clamped += 1;
        }
        self.stats.accepted += 1;
        self.last_cmd = Some((limited, now));
        CmdVerdict::Accepted
    }

    /// Local limits: planar only, speed caps. Never trusts the planner.
    fn limit(&self, c: Twist) -> Twist {
        Twist::planar(
            c.linear.x.clamp(-self.cfg.max_linear, self.cfg.max_linear),
            c.angular.z.clamp(-self.cfg.max_angular, self.cfg.max_angular),
        )
    }

    /// Run the watchdog at time `now` and return what to apply to the drive.
    pub fn tick(&mut self, now: u64) -> Applied {
        // Geofence breach (current pose) latches.
        if let Some((o, _)) = self.odom
            && self.latch.is_none()
            && !self.cfg.geofence.contains(o.x, o.y)
        {
            self.latch = Some(EStopCause::GeofenceBreach);
        }
        // Heartbeat loss latches (only once a heartbeat has ever been seen; before that we stay stopped).
        if let Some(hb) = self.last_hb
            && self.latch.is_none()
            && now.saturating_sub(hb) > self.cfg.heartbeat_timeout_ms
        {
            self.latch = Some(EStopCause::HeartbeatLost);
        }
        if let Some(c) = self.latch {
            return Applied { twist: Twist::ZERO, status: Status::EStop(c), estop_line: true };
        }
        let stop = |cause| Applied { twist: Twist::ZERO, status: Status::Stop(cause), estop_line: false };
        if self.last_hb.is_none() {
            return stop(StopCause::AwaitingHeartbeat);
        }
        let Some((cmd, rx)) = self.last_cmd else { return stop(StopCause::NoCommand) };
        if now.saturating_sub(rx) > self.cfg.cmd_timeout_ms {
            return stop(StopCause::CommandTimeout);
        }
        let Some((o, orx)) = self.odom else { return stop(StopCause::OdomStale) };
        if now.saturating_sub(orx) > self.cfg.odom_timeout_ms {
            return stop(StopCause::OdomStale);
        }
        let mut out = cmd;
        let t = self.cfg.lookahead_ms as f64 / 1000.0;
        let (px, py) = (o.x + out.linear.x * t * o.yaw.cos(), o.y + out.linear.x * t * o.yaw.sin());
        if !self.cfg.geofence.contains(px, py) {
            // Block translation, keep rotation so the robot can turn back inside.
            out = Twist::planar(0.0, out.angular.z);
            return Applied { twist: out, status: Status::Stop(StopCause::GeofenceLookahead), estop_line: false };
        }
        Applied { twist: out, status: Status::Run, estop_line: false }
    }

    /// Explicit operator re-arm. Never called by the controller itself.
    pub fn rearm(&mut self, now: u64) -> Result<(), RearmRefused> {
        if self.latch.is_none() {
            return Err(RearmRefused::NotLatched);
        }
        match self.last_hb {
            Some(hb) if now.saturating_sub(hb) <= self.cfg.heartbeat_timeout_ms => {}
            _ => return Err(RearmRefused::HeartbeatNotFresh),
        }
        match self.odom {
            Some((o, orx)) if now.saturating_sub(orx) <= self.cfg.odom_timeout_ms => {
                if !self.cfg.geofence.contains(o.x, o.y) {
                    return Err(RearmRefused::OutsideGeofence);
                }
            }
            _ => return Err(RearmRefused::OdomNotFresh),
        }
        self.latch = None;
        self.last_cmd = None; // a re-armed robot needs a fresh command
        Ok(())
    }
}
