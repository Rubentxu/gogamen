//! Integrated Studio tour (GOG-116): the six real labs, a replay session and the 3D robot view (GOG-105), registered in the
//! common [`StudioShell`] and driven through a reproducible, deterministic walk.
//!
//! Everything here is `std`-only and headless: the same [`drive_step`] is used by the CI contract
//! tests and by the real-window tour (`window::run_tour`, feature `window`).
//!
//! Labs: `lab.grid`, `lab.cartpole`, `lab.kart` (racing), `lab.life`, `lab.language_world`,
//! `lab.automation`, plus `lab.replay` (a disconnected [`ReplayPlayer`] over a recorded Grid run;
//! the live/replay pair).

use std::sync::{Arc, Mutex};
use std::time::Duration;

use gogamen_core::{MasterSeed, Permission, PermissionSet};
use gogamen_lab_racing::{KinematicBicycleConfig, TrackMap};
use gogamen_projection::{Scene2D, ScenePrimitive};
use gogamen_sdk::extensions::ExtensionId;

use crate::automation::{
    AutomationScenario, incident_scenario, project_automation_scene, run_automation,
};
use crate::cartpole_lab::{cartpole_demo_config, cartpole_lab_entry};
use crate::control::ObservationPort;
use crate::control::{CommandId, ControlAction, ControlCommand, ControlPort, Observation};
use crate::grid_lab::grid_lab_entry;
use crate::inspector::EntityRef;
use crate::kart_lab::kart_lab_entry;
use crate::language_world_lab::{
    ASK_PERMISSION, CannedBehavior, LanguageWorldLabConfig, LanguageWorldRollout,
    OfflineAskProvider, ViewMode, WARNING_TEXT, scenario_map,
};
use crate::life_lab::{LifeProbe, life_demo_config, life_epoch, life_lab_entry_with_probe};
use crate::recorder::record_grid;
use crate::replay::{Recording, ReplayPlayer};
use crate::shell::{LabEntry, LabSession, StudioShell};
use gogamen_lab_grid::GridMap;
use gogamen_lab_grid::world::GridWorldConfig;

pub const GRID_ID: &str = "lab.grid";
pub const CARTPOLE_ID: &str = "lab.cartpole";
pub const KART_ID: &str = "lab.kart";
pub const LIFE_ID: &str = "lab.life";
pub const LANGUAGE_WORLD_ID: &str = "lab.language_world";
pub const AUTOMATION_ID: &str = "lab.automation";
pub const REPLAY_ID: &str = "lab.replay";

/// Ticks recorded for the replay lab (also the replay horizon).
pub const REPLAY_TICKS: u64 = 12;

/// Seed shared by every lab of the tour.
pub const TOUR_SEED: u64 = 42;

// --------------------------------------------------------------------------------------------
// Sessions for the labs that were not shell labs yet
// --------------------------------------------------------------------------------------------

fn new_language_world() -> LanguageWorldRollout {
    let provider = OfflineAskProvider::new(CannedBehavior::Answer {
        obstacle: true,
        confidence: 0.9,
        latency: Duration::from_millis(200),
    });
    let perms = PermissionSet::empty().with_permission(Permission::new(ASK_PERMISSION));
    let mut r = LanguageWorldRollout::new(
        LanguageWorldLabConfig::new(scenario_map(true)),
        provider,
        perms,
    )
    .expect("canonical language world config");
    r.submit_human_notice(WARNING_TEXT)
        .expect("canonical notice");
    r
}

struct LanguageWorldSession {
    rollout: LanguageWorldRollout,
    generation: u64,
    scene: Scene2D,
}

impl LabSession for LanguageWorldSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, o: &Observation) {
        if o.session.generation != self.generation {
            self.generation = o.session.generation;
            self.rollout = new_language_world();
        }
        while self.rollout.tick_count() < o.tick.as_u64() {
            if self.rollout.tick().is_err() {
                break;
            }
        }
        self.scene = self.rollout.scene(ViewMode::Agent);
    }
}

struct AutomationSession {
    scenario: AutomationScenario,
    scene: Scene2D,
}

fn project_automation_at(scenario: &AutomationScenario, tick: u64) -> Scene2D {
    let mut partial = scenario.clone();
    partial.script.retain(|s| s.tick <= tick);
    let view = run_automation("incident", &partial).expect("headless automation rollout");
    project_automation_scene(&view)
}

impl LabSession for AutomationSession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, o: &Observation) {
        self.scene = project_automation_at(&self.scenario, o.tick.as_u64());
    }
}

/// Replay side of the live/replay pair: owns a decoded recording and seeks it to the control tick.
struct ReplaySession {
    recording: Recording,
    scene: Scene2D,
}

impl ReplaySession {
    fn scene_at(rec: &Recording, tick: u64) -> Option<Scene2D> {
        let mut player = ReplayPlayer::new(rec, 480.0, 360.0);
        player.seek(tick);
        player.scene()
    }
}

impl LabSession for ReplaySession {
    fn scene(&self) -> &Scene2D {
        &self.scene
    }

    fn sync_to(&mut self, o: &Observation) {
        // A gap keeps the previous scene (the player reports it explicitly; never interpolated).
        if let Some(s) = Self::scene_at(&self.recording, o.tick.as_u64()) {
            self.scene = s;
        }
    }
}

// --------------------------------------------------------------------------------------------
// Registry
// --------------------------------------------------------------------------------------------

/// The eight registered entries plus the Life probe used to exercise selection.
pub struct TourKit {
    pub entries: Vec<LabEntry>,
    pub life: LifeProbe,
}

/// Builds the static registry of the tour (no dynamic loading).
pub fn tour_kit() -> TourKit {
    let seed = MasterSeed::new(TOUR_SEED);
    let grid = grid_lab_entry(GridWorldConfig::default(), seed).expect("grid config");
    let cartpole = cartpole_lab_entry(cartpole_demo_config(), seed).expect("cartpole config");
    let kart = kart_lab_entry(
        KinematicBicycleConfig::default(),
        TrackMap::rectangular_circuit(),
        seed,
    )
    .expect("kart config");
    let (life, probe) =
        life_lab_entry_with_probe(life_demo_config(), MasterSeed::new(7)).expect("life config");
    let language = LabEntry::new(
        ExtensionId::new(LANGUAGE_WORLD_ID),
        "Language World",
        Arc::new(|| {
            let rollout = new_language_world();
            let scene = rollout.scene(ViewMode::Agent);
            Box::new(LanguageWorldSession {
                rollout,
                generation: 0,
                scene,
            })
        }),
    );
    let automation = LabEntry::new(
        ExtensionId::new(AUTOMATION_ID),
        "Automation",
        Arc::new(|| {
            let scenario = incident_scenario();
            let scene = project_automation_at(&scenario, 0);
            let s = AutomationSession { scenario, scene };
            Box::new(s)
        }),
    );
    let recording = Arc::new(Mutex::new(
        record_grid(TOUR_SEED, GridMap::cul_de_sac_map(), REPLAY_TICKS).expect("record grid"),
    ));
    let replay = LabEntry::new(
        ExtensionId::new(REPLAY_ID),
        "Replay (Grid)",
        Arc::new(move || {
            let rec = recording.lock().expect("recording lock").clone();
            let scene = ReplaySession::scene_at(&rec, 0).expect("frame at tick 0");
            Box::new(ReplaySession {
                recording: rec,
                scene,
            })
        }),
    );
    TourKit {
        entries: vec![
            grid,
            cartpole,
            kart,
            life,
            language,
            automation,
            replay,
            crate::robot3d::robot3d_lab_entry(),
        ],
        life: probe,
    }
}

/// A shell holding the tour registry (no lab selected yet).
pub fn tour_shell(w: f64, h: f64) -> (StudioShell, LifeProbe) {
    let kit = tour_kit();
    (
        StudioShell::new(kit.entries, w, h).expect("unique lab ids"),
        kit.life,
    )
}

// --------------------------------------------------------------------------------------------
// The walk
// --------------------------------------------------------------------------------------------

/// One stop of the tour: select `lab`, advance `ticks` with `Step` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TourStep {
    pub lab: &'static str,
    pub ticks: u64,
}

/// The reproducible walk: every lab once, in registry order.
pub const TOUR_PLAN: [TourStep; 8] = [
    TourStep {
        lab: GRID_ID,
        ticks: 6,
    },
    TourStep {
        lab: CARTPOLE_ID,
        ticks: 10,
    },
    TourStep {
        lab: KART_ID,
        ticks: 8,
    },
    TourStep {
        lab: LIFE_ID,
        ticks: 8,
    },
    TourStep {
        lab: LANGUAGE_WORLD_ID,
        ticks: 3,
    },
    TourStep {
        lab: AUTOMATION_ID,
        ticks: 3,
    },
    TourStep {
        lab: REPLAY_ID,
        ticks: 9,
    },
    // GOG-105: 30 control ticks = 300 sim ticks, past the scanner dropout, so the scene is STALE.
    TourStep {
        lab: crate::robot3d::ROBOT3D_ID,
        ticks: 30,
    },
];

/// FNV-1a 64 over bytes (same constants as the window screenshots).
pub fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Facts about the scene shown after a tour step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepFacts {
    pub lab: String,
    pub tick: u64,
    pub primitives: usize,
    pub labels: Vec<String>,
    /// Hash of the software SVG of the scene through the shell camera.
    pub scene_hash: u64,
    /// Selected entity id (Life only).
    pub selected: Option<u64>,
}

/// Number of primitives in a scene.
pub fn primitive_count(scene: &Scene2D) -> usize {
    scene.layers().iter().map(|l| l.nodes().len()).sum()
}

/// Label texts of a scene, in order.
pub fn labels(scene: &Scene2D) -> Vec<String> {
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

/// Submits `Step` `ticks` times through the control port (syncing the lab after each), selects the
/// first living organism in Life, and returns facts about the resulting scene.
/// `next_command` is a monotonically increasing command-id counter owned by the caller.
pub fn drive_step(
    shell: &mut StudioShell,
    life: &LifeProbe,
    step: &TourStep,
    next_command: &mut u64,
) -> Result<StepFacts, String> {
    shell.select_lab(step.lab).map_err(|e| e.to_string())?;
    let control = shell
        .control_port()
        .ok_or("no active session")?
        .map_err(|e| format!("{e:?}"))?;
    let observe = shell.observation_port().ok_or("no active session")?;
    for _ in 0..step.ticks {
        *next_command += 1;
        let ack = control.submit(ControlCommand {
            id: CommandId(*next_command),
            session: observe.observe().session,
            action: ControlAction::Step,
        });
        if !matches!(ack.outcome, crate::control::AckOutcome::Applied) {
            return Err(format!("step rejected: {ack:?}"));
        }
        shell.sync_active();
        if step.lab == crate::life_lab::LIFE_LAB_ID
            && let Some(o) = life
                .snapshot()
                .and_then(|s| s.view.organisms.first().cloned())
        {
            life.select(Some(EntityRef::new(o.id, life_epoch(&observe.observe()))));
        }
    }
    shell.sync_active();
    let obs = observe.observe();
    let scene = shell.scene().ok_or("no scene")?;
    let svg = shell.render_svg().ok_or("no svg")?;
    Ok(StepFacts {
        lab: step.lab.to_owned(),
        tick: obs.tick.as_u64(),
        primitives: primitive_count(scene),
        labels: labels(scene),
        scene_hash: fnv64(svg.as_bytes()),
        selected: if step.lab == crate::life_lab::LIFE_LAB_ID {
            life.snapshot().and_then(|s| s.view.selected)
        } else {
            None
        },
    })
}

/// Runs the whole [`TOUR_PLAN`] headlessly on a fresh shell (800x600).
pub fn run_headless_tour() -> Result<Vec<StepFacts>, String> {
    let (mut shell, life) = tour_shell(800.0, 600.0);
    let mut cmd = 0;
    TOUR_PLAN
        .iter()
        .map(|s| drive_step(&mut shell, &life, s, &mut cmd))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::{AckOutcome, ObservationPort, RunState};

    fn facts_of<'a>(all: &'a [StepFacts], lab: &str) -> &'a StepFacts {
        all.iter().find(|f| f.lab == lab).expect("lab in tour")
    }

    #[test]
    fn registry_has_the_real_labs_plus_replay_and_robot3d() {
        let (shell, _) = tour_shell(800.0, 600.0);
        assert_eq!(
            shell.lab_ids(),
            [
                GRID_ID,
                CARTPOLE_ID,
                KART_ID,
                LIFE_ID,
                LANGUAGE_WORLD_ID,
                AUTOMATION_ID,
                REPLAY_ID,
                crate::robot3d::ROBOT3D_ID
            ]
        );
        // The tour plan covers every registered lab exactly once, in registry order.
        let plan: Vec<&str> = TOUR_PLAN.iter().map(|s| s.lab).collect();
        assert_eq!(plan, shell.lab_ids());
    }

    #[test]
    fn every_lab_projects_a_non_empty_scene_at_the_planned_tick() {
        let all = run_headless_tour().unwrap();
        for (f, step) in all.iter().zip(TOUR_PLAN) {
            assert!(f.primitives > 0, "{} has an empty scene", f.lab);
            assert_eq!(f.tick, step.ticks, "{} tick", f.lab);
        }
    }

    #[test]
    fn tour_is_deterministic_and_each_lab_looks_different() {
        let a = run_headless_tour().unwrap();
        let b = run_headless_tour().unwrap();
        assert_eq!(
            a, b,
            "two independent tours must match (scene hash included)"
        );
        for i in 0..a.len() {
            for j in i + 1..a.len() {
                assert_ne!(
                    a[i].scene_hash, a[j].scene_hash,
                    "{} vs {}",
                    a[i].lab, a[j].lab
                );
            }
        }
    }

    #[test]
    fn expected_ids_and_labels_per_lab() {
        let all = run_headless_tour().unwrap();
        let has = |lab: &str, needle: &str| {
            facts_of(&all, lab)
                .labels
                .iter()
                .any(|l| l.contains(needle))
        };
        assert!(has(GRID_ID, "G"), "grid goal glyph");
        assert!(has(CARTPOLE_ID, "STATE: x="));
        assert!(has(CARTPOLE_ID, "GOAL: upright"));
        assert!(has(KART_ID, "Executed Trajectory"));
        assert!(has(KART_ID, "ckpt "));
        assert!(has(LIFE_ID, "Selected Future"));
        assert!(has(LIFE_ID, "alive "));
        assert!(has(LANGUAGE_WORLD_ID, "agent view: beliefs only"));
        assert!(has(LANGUAGE_WORLD_ID, "== Belief updates"));
        assert!(has(AUTOMATION_ID, "Automation graph: incident"));
        assert!(has(AUTOMATION_ID, "state Resolved [terminal] [ACTIVE]"));
        assert!(has(AUTOMATION_ID, "PROPOSED (not executed)"));
        let r3 = crate::robot3d::ROBOT3D_ID;
        assert!(
            has(r3, "STALE: scan age"),
            "robot3d stale marker at tick 30"
        );
        assert!(has(r3, "uncertainty (heuristic drift)"));
        assert!(has(r3, "x (forward)"));
    }

    #[test]
    fn life_selection_is_reflected_in_the_view() {
        let all = run_headless_tour().unwrap();
        assert_eq!(facts_of(&all, LIFE_ID).selected, Some(1));
        assert!(
            all.iter()
                .filter(|f| f.lab != LIFE_ID)
                .all(|f| f.selected.is_none())
        );
    }

    #[test]
    fn live_grid_and_replay_agree_at_the_same_tick() {
        let (mut shell, life) = tour_shell(800.0, 600.0);
        let mut cmd = 0;
        let step = |lab, ticks| TourStep { lab, ticks };
        let live = drive_step(&mut shell, &life, &step(GRID_ID, 6), &mut cmd).unwrap();
        let replay = drive_step(&mut shell, &life, &step(REPLAY_ID, 6), &mut cmd).unwrap();
        assert_eq!(live.scene_hash, replay.scene_hash);
        // A different replay tick differs.
        let (mut shell, life) = tour_shell(800.0, 600.0);
        let mut cmd = 0;
        let later = drive_step(&mut shell, &life, &step(REPLAY_ID, 9), &mut cmd).unwrap();
        assert_ne!(live.scene_hash, later.scene_hash);
    }

    #[test]
    fn replay_seeks_beyond_the_recording_without_panicking() {
        let (mut shell, life) = tour_shell(800.0, 600.0);
        let mut cmd = 0;
        let f = drive_step(
            &mut shell,
            &life,
            &TourStep {
                lab: REPLAY_ID,
                ticks: REPLAY_TICKS + 5,
            },
            &mut cmd,
        )
        .unwrap();
        assert!(f.primitives > 0);
    }

    #[test]
    fn pause_step_reset_controls_work_on_every_lab() {
        for step in TOUR_PLAN {
            let (mut shell, life) = tour_shell(800.0, 600.0);
            let mut cmd = 0;
            let first = drive_step(&mut shell, &life, &step, &mut cmd).unwrap();
            let control = shell.control_port().unwrap().unwrap();
            let observe = shell.observation_port().unwrap();
            assert_eq!(
                observe.observe().run_state,
                RunState::Paused,
                "{}",
                step.lab
            );
            let ack = control.submit(ControlCommand {
                id: CommandId(9_000),
                session: observe.observe().session,
                action: ControlAction::Reset,
            });
            assert_eq!(ack.outcome, AckOutcome::Applied, "{}", step.lab);
            shell.sync_active();
            assert_eq!(observe.observe().tick.as_u64(), 0, "{}", step.lab);
            // Replaying after the reset reproduces the exact scene.
            for i in 0..step.ticks {
                control.submit(ControlCommand {
                    id: CommandId(9_001 + i),
                    session: observe.observe().session,
                    action: ControlAction::Step,
                });
                shell.sync_active();
            }
            // Life keeps its (probe-side) selection across a Reset, so its replayed scene
            // legitimately differs from the first walk through the FMC futures overlay.
            if step.lab != LIFE_ID {
                let svg = shell.render_svg().unwrap();
                assert_eq!(
                    fnv64(svg.as_bytes()),
                    first.scene_hash,
                    "{} after reset",
                    step.lab
                );
            }
        }
    }

    #[test]
    fn unknown_lab_is_a_typed_error() {
        let (mut shell, life) = tour_shell(800.0, 600.0);
        let mut cmd = 0;
        let err = drive_step(
            &mut shell,
            &life,
            &TourStep {
                lab: "lab.nope",
                ticks: 1,
            },
            &mut cmd,
        )
        .unwrap_err();
        assert!(err.contains("unknown lab"));
    }
}
