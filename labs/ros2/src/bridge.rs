//! Deterministic emulator of the ROS 2 link + unicycle robot, driven by a virtual clock.
//! The planner side only sends messages; the robot side runs [`LocalController`] on what arrives.

use crate::contract::{Heartbeat, Odom, PlannerState, Twist};
use crate::controller::{Applied, CmdVerdict, LocalConfig, LocalController};

/// Fault injection on the planner->robot link (deterministic, seeded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LinkFaults {
    /// Extra one-way delay for `/cmd_vel` (ms).
    pub cmd_delay_ms: u64,
    /// Loss probability for `/cmd_vel`, per mille.
    pub cmd_loss_permille: u32,
    /// Loss probability for heartbeats, per mille.
    pub hb_loss_permille: u32,
    /// Heartbeat channel cut completely.
    pub hb_cut: bool,
}

enum Msg {
    Cmd { stamp: u64, twist: Twist },
    Hb(Heartbeat),
}

pub const TICK_MS: u64 = 10;

pub struct Emulator {
    now: u64,
    pub faults: LinkFaults,
    pub ctrl: LocalController,
    inflight: Vec<(u64, Msg)>,
    rng: u64,
    seq: u64,
    pub x: f64,
    pub y: f64,
    pub yaw: f64,
    pub last: Applied,
    /// Sequence number of the last heartbeat delivered to the controller.
    pub last_hb_seq: u64,
    pub verdicts: Vec<CmdVerdict>,
}

impl Emulator {
    pub fn new(cfg: LocalConfig, seed: u64) -> Self {
        let mut ctrl = LocalController::new(cfg);
        let last = ctrl.tick(0);
        Self {
            now: 0,
            faults: LinkFaults::default(),
            ctrl,
            inflight: Vec::new(),
            rng: seed | 1,
            seq: 0,
            x: 0.0,
            y: 0.0,
            yaw: 0.0,
            last,
            last_hb_seq: 0,
            verdicts: Vec::new(),
        }
    }

    pub fn now(&self) -> u64 {
        self.now
    }

    fn lost(&mut self, permille: u32) -> bool {
        // xorshift64
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng % 1000) < permille as u64
    }

    /// Planner publishes `/cmd_vel` now.
    pub fn planner_cmd(&mut self, twist: Twist) {
        if self.lost(self.faults.cmd_loss_permille) {
            return;
        }
        self.inflight.push((self.now + self.faults.cmd_delay_ms, Msg::Cmd { stamp: self.now, twist }));
    }

    /// Planner publishes a heartbeat now.
    pub fn planner_heartbeat(&mut self) {
        if self.faults.hb_cut || self.lost(self.faults.hb_loss_permille) {
            return;
        }
        self.seq += 1;
        let hb = Heartbeat { stamp_ms: self.now, seq: self.seq, state: PlannerState::Planning };
        self.inflight.push((self.now, Msg::Hb(hb)));
    }

    /// Advance the virtual clock by `ms` (multiple of [`TICK_MS`]).
    pub fn run_ms(&mut self, ms: u64) {
        for _ in 0..ms / TICK_MS {
            self.step();
        }
    }

    fn step(&mut self) {
        self.now += TICK_MS;
        let now = self.now;
        let mut due = Vec::new();
        let mut i = 0;
        while i < self.inflight.len() {
            if self.inflight[i].0 <= now {
                due.push(self.inflight.remove(i));
            } else {
                i += 1;
            }
        }
        for (_, m) in due {
            match m {
                Msg::Cmd { stamp, twist } => {
                    let v = self.ctrl.on_cmd(now, stamp, twist);
                    self.verdicts.push(v);
                }
                Msg::Hb(hb) => {
                    self.last_hb_seq = hb.seq;
                    self.ctrl.on_heartbeat(now);
                }
            }
        }
        // Local odometry (not subject to link faults).
        self.ctrl.on_odom(now, Odom { stamp_ms: now, x: self.x, y: self.y, yaw: self.yaw, v: 0.0, w: 0.0 });
        self.last = self.ctrl.tick(now);
        let dt = TICK_MS as f64 / 1000.0;
        self.x += self.last.twist.linear.x * self.yaw.cos() * dt;
        self.y += self.last.twist.linear.x * self.yaw.sin() * dt;
        self.yaw += self.last.twist.angular.z * dt;
    }
}
