//! With/without coach comparison on the synthetic fixture.

use super::advice::AdviceCatalog;
use super::mock::MockCoach;
use super::planner::CoachedPlanner;
use crate::fixture::SyntheticPlatformer;
use crate::metrics::{RunConfig, run_episode};
use crate::rom::sha256_hex;
use std::fmt::Write;

/// Planner budget (rollouts per candidate, horizon in decisions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    pub name: &'static str,
    pub rollouts: u32,
    pub horizon: u32,
}

/// Deliberately myopic planner: the regime where a hint can matter.
pub const MYOPIC: Budget = Budget {
    name: "myopic(1x2)",
    rollouts: 1,
    horizon: 2,
};
/// Comfortable planner: the fixture is easy for it, so the coach has little to add.
pub const FULL: Budget = Budget {
    name: "full(3x5)",
    rollouts: 3,
    horizon: 5,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scenario {
    NoCoach,
    Truthful,
    Late,
    Lying,
    /// Truthful coach that goes silent for a long network pause.
    Mute,
}

impl Scenario {
    pub const ALL: [Scenario; 5] = [
        Self::NoCoach,
        Self::Truthful,
        Self::Late,
        Self::Lying,
        Self::Mute,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::NoCoach => "no_coach",
            Self::Truthful => "truthful_lat2",
            Self::Late => "truthful_late30",
            Self::Lying => "lying_lat1",
            Self::Mute => "mute_after20",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub budget: &'static str,
    pub scenario: Scenario,
    pub seed: u64,
    pub max_x: i32,
    pub deaths: u32,
    pub complete: bool,
    pub decisions: u32,
    pub planning_steps: u64,
    pub emitted: usize,
    pub applied: usize,
    pub expired: usize,
    pub rejected: usize,
    pub decisions_with_advice: u64,
    pub latency_mean: Option<f64>,
    pub latency_max: Option<u64>,
    pub trace_hash: String,
}

pub fn run_scenario(s: Scenario, seed: u64, cfg: &RunConfig, b: Budget) -> Row {
    let mut p = CoachedPlanner::new(seed, b.rollouts, b.horizon);
    p = match s {
        Scenario::NoCoach => p,
        Scenario::Truthful => p.with_coach(Box::new(MockCoach::truthful(2))),
        Scenario::Late => p.with_coach(Box::new(MockCoach::truthful(30))),
        Scenario::Lying => p.with_coach(Box::new(MockCoach::lying(1))),
        Scenario::Mute => p.with_coach(Box::new(MockCoach::truthful(2).muted(20, 100_000))),
    }
    .with_catalog(AdviceCatalog::standard());
    let mut port = SyntheticPlatformer::new();
    let m = run_episode(&mut port, &mut p, &RunConfig { seed, ..*cfg });
    let lat = p.report.latency();
    Row {
        budget: b.name,
        scenario: s,
        seed,
        max_x: m.max_x,
        deaths: m.deaths,
        complete: m.level_complete,
        decisions: m.actions_used,
        planning_steps: m.planning_steps,
        emitted: p.report.records.len(),
        applied: p.report.applied(),
        expired: p.report.expired(),
        rejected: p.report.rejected(),
        decisions_with_advice: p.report.decisions_with_advice,
        latency_mean: lat.map(|l| l.0),
        latency_max: lat.map(|l| l.1),
        trace_hash: m.trace_hash,
    }
}

pub fn evaluate(seeds: &[u64], cfg: &RunConfig, budgets: &[Budget]) -> Vec<Row> {
    let mut v = Vec::new();
    for &b in budgets {
        for &s in &Scenario::ALL {
            for &seed in seeds {
                v.push(run_scenario(s, seed, cfg, b));
            }
        }
    }
    v
}

/// SHA-256 over every deterministic field of every row.
pub fn suite_hash(rows: &[Row]) -> String {
    let mut b = String::new();
    for r in rows {
        let _ = write!(
            b,
            "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{:?}|{:?}|{};",
            r.budget,
            r.scenario.name(),
            r.seed,
            r.max_x,
            r.deaths,
            r.complete,
            r.decisions,
            r.planning_steps,
            r.emitted,
            r.applied,
            r.expired,
            r.rejected,
            r.decisions_with_advice,
            r.latency_mean,
            r.latency_max,
            r.trace_hash
        );
    }
    sha256_hex(b.as_bytes())
}

/// Per-budget, per-scenario aggregates (mean over seeds).
pub fn summary(rows: &[Row]) -> String {
    let mut o = String::new();
    let mut budgets: Vec<&str> = Vec::new();
    for r in rows {
        if !budgets.contains(&r.budget) {
            budgets.push(r.budget);
        }
    }
    for b in budgets {
        let _ = writeln!(o, "planner budget: {b}");
        o.push_str("scenario          seeds  mean_max_x  mean_deaths  done  mean_plan_steps  emitted applied expired rejected  lat_mean lat_max\n");
        for &s in &Scenario::ALL {
            let r: Vec<&Row> = rows
                .iter()
                .filter(|r| r.scenario == s && r.budget == b)
                .collect();
            if r.is_empty() {
                continue;
            }
            let n = r.len() as f64;
            let lat: Vec<f64> = r.iter().filter_map(|r| r.latency_mean).collect();
            let lmax = r.iter().filter_map(|r| r.latency_max).max();
            let _ = writeln!(
                o,
                "{:<17} {:>5} {:>11.1} {:>12.2} {:>2}/{:<2} {:>16.0} {:>8} {:>7} {:>7} {:>8} {:>9} {:>7}",
                s.name(),
                r.len(),
                r.iter().map(|r| f64::from(r.max_x)).sum::<f64>() / n,
                r.iter().map(|r| f64::from(r.deaths)).sum::<f64>() / n,
                r.iter().filter(|r| r.complete).count(),
                r.len(),
                r.iter().map(|r| r.planning_steps as f64).sum::<f64>() / n,
                r.iter().map(|r| r.emitted).sum::<usize>(),
                r.iter().map(|r| r.applied).sum::<usize>(),
                r.iter().map(|r| r.expired).sum::<usize>(),
                r.iter().map(|r| r.rejected).sum::<usize>(),
                if lat.is_empty() {
                    "-".to_string()
                } else {
                    format!("{:.1}", lat.iter().sum::<f64>() / lat.len() as f64)
                },
                lmax.map_or("-".to_string(), |m| m.to_string()),
            );
        }
    }
    o
}
