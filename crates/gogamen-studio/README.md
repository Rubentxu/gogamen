# gogamen-studio

Studio shell for Gogamen: lab selector, 2D camera and a deterministic software viewport.
Depends only on `std`, `gogamen-core`, `gogamen-projection` and `gogamen-sdk`.

## Window backend and dependency spike (GOG-89)

Default build: `std`-only (no graphical dependency). The real window lives behind the optional
`window` feature (off by default), implemented with **eframe/egui 0.37.0** (glow/OpenGL, winit
0.30). `gogamen-core` and `gogamen-cli` never depend on it (`cargo tree` check below).

Dependency spike (Backlog asked for Bevy + `bevy_egui` with a fallback):

| Attempt | Result |
|---|---|
| `bevy =0.20.1` (default features) + `bevy_egui =0.43.1` (needs bevy ^0.20, egui ^0.36), `cargo check` | FAILED after 1m38s: `alsa-sys` build script, `alsa.pc` not found (system dev package; `sudo` unavailable). |
| same, `bevy` with `default-features=false, features=[bevy_winit, bevy_render, bevy_core_pipeline, bevy_ui, x11, wayland, std]`, `cargo check` | Resolves and checks OK in 1m04s (fresh target dir). A Bevy-based `ViewportBackend` was NOT implemented (NOT_RUN): not needed once eframe worked and it would add a ~10x larger dependency graph. |
| `eframe =0.37.0` (`glow`, `default_fonts`, `wayland`, `x11`, no wgpu) | Chosen. `cargo check --features window` 43s on a warm cache, window opens and renders on the real session. |

`window::WindowBackend` implements `ViewportBackend` with `BackendKind::Window`. Its `render`
returns the deterministic draw list handed to the painter; the pixels only exist in the running
window, so a graphical PASS comes only from `window::run_script` / `window::run_interactive`
evidence (`ScriptReport::to_render_test_report`), never from `render` alone. The window reuses
`StudioShell` (lab selector, single active session) and `Camera2D`; nothing is duplicated.

Interactive controls: drag = pan, wheel = zoom about the cursor, double-click or `Fit` = re-fit,
window resize = re-fit (`StudioShell::resize` / `Camera2D::fit_to_window`), top bar = lab selector.

```text
cargo run  -p gogamen-studio --features window --example studio_window                    # interactive
cargo run  -p gogamen-studio --features window --example studio_window -- --script /tmp/g89
cargo test -p gogamen-studio --features window --test window_real                         # real-display test
```

## Graphical environment and evidence (AC1)

Status values: `Pass` (executed with a `Window` backend and pixel evidence), `Fail`,
`NotRunHeadlessHost` (no DISPLAY/WAYLAND_DISPLAY), `NotRunNoWindowBackend` (display but no window
backend compiled in). Software-backend tests are NOT a graphical PASS
(`RenderTestReport::is_graphical_window_pass()` is true only for a `Window` backend).

`tests/window_real.rs` (`harness = false`, winit must own the main thread; only built with
`--features window`) prints `NOT_RUN` and exits 0 when there is no display, otherwise it really opens
the window and asserts on captured pixels.

Verified run (2026-11-21): KDE Plasma 6.8.3, Wayland session (`WAYLAND_DISPLAY=wayland-1`, winit
picked Wayland/XWayland `DISPLAY=:0`), eframe 0.37.0 glow, GL renderer `NVIDIA GeForce RTX 4060
Ti/PCIe/SSE2`, GL `3.3.0 NVIDIA 620.70.06` (real GPU, not software GL), rustc 1.99.0. The script
(programmatic input: `zoom_at(1.6)`, `pan(40,-25)`, `InnerSize(720x380)`, lab switch) captured the
framebuffer with egui's internal screenshot (PNG under `/tmp/g89/`, not in the repo):

| step | size | FNV-1a hash | scene pixels |
|---|---|---|---|
| 1_initial_fit (alpha) | 900x620 | `7255d7e0a770fb9b` | 2177 |
| 2_zoom_pan | 900x620 | `a961a605cf1938f2` | 1155 |
| 3_resized_refit | 720x380 | `c868614122d0df56` | 1295 |
| 4_lab_beta | 720x380 | `66ce639902a78e6f` | 4417 |

All four hashes differ and the scene (palette-coloured strokes) is visible in each; two runs gave
identical hashes. Input provenance: steps 2, 3 (as requested by the app) and 4 are PROGRAMMATIC.
Additionally, an external X11 resize done with `xdotool windowsize` (window manager level, real)
changed 900x620 to 700x400 and re-fit the scene, captured with ImageMagick `import -window`
(manual check, not part of the automated test). Pointer drag/wheel injected by `xdotool` (XTEST)
and `ydotool` did NOT reach the window in this XWayland/KDE session, so real mouse drag and wheel
input is NOT_RUN (the handlers are covered only by the camera unit tests and by the scripted
camera calls). `grim` is unsupported by KDE ("compositor doesn't support the screen capture
protocol").

## Session controls (GOG-90)

`control` module. `ControlPort::submit(ControlCommand) -> Ack` is the only mutating entry;
`ObservationPort::observe() -> Observation` returns immutable snapshots. The UI only receives the
ports (`StudioShell::control_port` / `observation_port`); the engine side holds `SessionHost`
(`engine_handle`, `pump`, `close_viewer`). Time is `gogamen_core::Tick`; no wall clock.

- Commands: `Pause`, `Resume`, `Step`, `Reset`, `SetSpeed(f64)`; each has a `CommandId` + `SessionId`
  (`epoch` per lab selection, `generation` bumped by `Reset`). Sessions start `Paused`.
- ACK always carries `revision` and `tick` (unchanged on rejection). `revision` bumps on every applied
  command. `Step` advances exactly one tick, only when paused; **while `Running` it is rejected**
  (`StepWhileRunning`). `Reset` -> tick 0, paused, new generation, id history cleared.
- Rejections (`Rejected { reason }`: `InvalidSpeed`, `DuplicateCommandId`, `StaleSession`,
  `StepWhileRunning`, `EngineStopped`, `TickOverflow`) never change state and are visible in
  `Observation::{rejection_count, recent_rejections}` (last 64). Check order: stale session,
  duplicate id (stale commands do not consume ids), stopped, validity. Speed must be finite, `> 0`, `<= MAX_SPEED` (1000).
- `SessionKind::{Simulated, Physical, Remote}`: only `Simulated` is controllable by default;
  otherwise `control_port()` returns `ControlUnavailable::ReadOnlySession` while observation works.
- Closing the viewer does not stop the engine unless `SessionOptions::stop_on_close` (default
  `false`) is set; then the engine becomes `Stopped` and rejects commands.

## Dynamic parameters and typed commands (GOG-95)

`params` + `param_ui` modules (new minimal types: `gogamen-core`/`gogamen-spec` only define module
capabilities, no parameter/unit types).

- `ParamSchema` publishes `ParamDescriptor { id, label, kind (Float/Int range+step, Bool, Enum), unit,
  mutability, default }`; validated at construction (unique ids, valid defaults). Created with
  `SessionHost::with_params`; read via `ObservationHandle::param_schema()`.
- Only mutating entry: `ParamControlPort::submit_params(ParamCommand)` (`SessionHost::param_port()`).
  `ParamRequest::Set(Vec<ParamChange>)` is atomic (all entries validated before anything is queued).
- Accepted -> `ParamOutcome::Queued { revision, applies_at }` (`applies_at` = current tick + 1). Nothing
  changes mid-tick: the change is applied at the next tick boundary (`Step`/`pump`), FIFO, and each
  applied entry appends a `ParamEvent { param, old, new, revision, tick, command }` to the deterministic
  log (`ObservationHandle::param_events()`). Same initial state + same command sequence => identical
  values, ACKs and event log. `Reset` restores defaults, drops the queue and clears events/revision.
- Rejections (`ParamRejectReason`: `OutOfRange`, `WrongType`, `UnknownParam`, `Immutable`,
  `ReadOnlySession`, `StructuralChange`, `EmptyCommand`, `DuplicateParamInCommand`,
  `Session(RejectReason)` for stale/duplicate/stopped/overflow) never mutate state and are visible in
  `Observation::{param_rejection_count, recent_param_rejections}`. `Structural` requests always require
  explicit compilation/replacement and are rejected here. Read-only sessions reject via `param_port()`.
- `Observation` also exposes `params`, `param_revision`, `pending_param_commands`.
- UI: `control_descriptors(&ParamSchema, &UiPermissions, &[(ParamId, ParamValue)]) -> Vec<ControlDescriptor>`
  is pure (Slider/Number/Toggle/Choice with range, step, unit, enabled + `DisabledReason`); everything is
  disabled for read-only sessions or immutable params. It takes no engine reference; the UI sends typed
  commands through the port. No Kotlin execution, no actuator access.

## GridWorld in Studio (GOG-110)

`grid_lab` module: first vertical slice with the **real** Grid engine (`labs/grid`, package
`gogamen-lab-grid`; studio only adds a path dependency, no second engine).

- `project_grid_scene(&GridWorldView) -> Scene2D`: pure projector (cells, walls, goal, hazards, agent).
- `GridRollout`: headless engine side (real `GridWorldEnv` + seeded policy `seed.derive(Episode(0))`,
  action = `next_u64 % 5`); its per-step trace hash equals `run_grid_episode` of the lab.
- `grid_lab_entry[_with_probe](config, seed)`: registers lab `lab.grid`. The session advances the world
  only in `StudioShell::sync_active()` (engine side): `Step` -> exactly one grid step, `Reset` -> seeded
  initial world (new generation). Render (`render_svg`) never steps. `GridProbe` exposes the projected
  scene + step records/hash. After a terminal state the host tick may keep growing; the Grid stays
  terminal until `Reset`.
- Run: `cargo test -p gogamen-studio --offline grid_lab` and
  `cargo run -p gogamen-studio --offline --example grid_svg -- 5 42 > /tmp/grid.svg`.

## Kart in Studio (GOG-111)

`kart_lab` projects the real Kart world (`gogamen-lab-racing`: `KartEnv` + `KartFmcController`) through the same shell, inspector and parameter port as GridWorld.

- `KartRollout` (headless) runs the lab's own loop (`decide_from_env` + `Actuator::apply`) and keeps a trace hash that also records authorized budget changes (`budget_changes()`). `project_kart_scene(&KartWorldView)` is the pure projector; sensors, chosen/rejected futures, executed trail, collision marker and the legend come from the common `gogamen-projection` decision overlay (`project_kart_layers`). Rendering never advances the sim; `sync_to` does.
- Layers: `Environment` walls/checkpoints, `Decisions` car (selectable `EntityId(1)`) + heading, `Sensors` rays (`TrackMap::cast_ray`), futures (chosen vs rejected), terminal markers, `Labels` applied action + status.
- Inspector: `kart_snapshot_view` / `kart_schema` expose `car`, `checkpoint` and `future` entities (`EntityRef{id, epoch}`, epoch changes on Reset).
- Params (GOG-95): `planner.max_transitions` (mutable, 1..=4096, applied at the tick boundary and logged) and `planner.horizon` (immutable). Rejections stay in the param log; accepted changes land in `KartSnapshot::budget_changes` and the param events.
- Caveats: futures belong to the decision that produced the last applied action (drawn from the pre-step pose); rejected futures are counterfactual root actions held through the real `KartModel`, not FMC walkers. Budget application is exact when syncing per Step (a `pump` coalesces ticks). Additive lab changes: `TrackMap::cast_ray`, `KartFmcController::{candidate_futures, set_max_transitions_per_decision}`.
- Run: `cargo run -p gogamen-studio --offline --example kart_svg -- [steps] [seed] > kart.svg`.

## Life in Studio (GOG-112)

`life_lab` projects the real Life world (`gogamen-lab-life`: `LifeWorld` + `HomeostaticFmcAgent`) through the same shell, inspector and parameter port as the other labs.

- `LifeRollout` (headless) runs `step_tick_with` with the lab's FMC agent (`decide_with_reproduction`) and keeps a trace hash that also records authorized mutation-rate changes (`mutation_changes()`). `project_life_scene(&LifeWorldView)` is the pure projector: `Environment` food cells (selectable, by id) and border, `Decisions` alive organisms (selectable `EntityId(organism_id)`, radius = energy, `state_code` 200 + species), `Custom(21)` genealogy edges parent -> child, `Labels` status + metrics.
- Only alive organisms are projected (scene and inspector snapshot), so births/deaths leave no ghosts. Organism ids are monotonic; the `EntityRef` epoch changes on Reset, so a stale selection becomes `Recycled`.
- FMC futures (chosen/rejected, terminal markers) use the common `gogamen-projection` decision overlay and are computed only for the organism selected via `LifeProbe::select`; with no selection there are no future layers.
- Population metrics (alive, species richness, Shannon, mean energy, births/deaths, max generation) come from `gogamen_lab_life::PopulationMetrics::from_view` (the lab source) and are never recomputed in studio; they appear in the `population` inspector entity.
- Inspector: `life_snapshot_view` / `life_schema` expose `organism` (position, energy, species, genome, age, generation, parent), `resource` and `population` entities.
- Params (GOG-95): `life.mutation_rate` (mutable, 0..=1, applied at the tick boundary and recorded in the trace) and `life.max_population` (immutable). Rejections never enter the trace.
- Additive lab changes: `Genome::species_id`, `LifeWorld::set_mutation_rate`, `PopulationMetrics`, `HomeostaticFmcAgent::{decide_with_reproduction, candidate_futures}`.
- Caveats: futures are counterfactual root actions of the FMC local model from the current state; mutation changes apply to reproductions from the next step.
- Run: `cargo run -p gogamen-studio --offline --example life_svg -- [steps] [seed] > life.svg`.

## CartPole in Studio (GOG-114)

`cartpole_lab` projects the real CartPole simulator (`gogamen-lab-cartpole`: `CartPoleEnv` + `LinearFeedbackController`, same loop and trace hash as `run_cartpole_headless`) through the common shell, camera, inspector and parameter port. No physics or controller lives in studio.

- `CartPoleRollout` (headless) steps the lab env with the lab's controller. `project_cartpole_scene(&CartPoleView)` is the pure projector: `Environment` = STATE (track, `x` limit bars, cart, pole at angle), `Decisions` = ACTION (applied-force arrow, length proportional to F / max force; terminal marker), `Custom(31)` = visual GOAL (upright wedge `|theta| <= limit` and the target angle 0), `Labels` = legend + readouts (x in m, theta in rad, F in N).
- Studio step-by-step equals headless: per-step state and trace hash match `run_cartpole_headless` for the same seed (tests also cover the uncontrolled, terminating episode).
- Inspector: `cartpole_snapshot_view` / `cartpole_schema` expose `cart` (position m, velocity m/s), `pole` (angle rad, angular velocity rad/s), `action` (force N, force limit N) and `limits`.
- Params (GOG-95): `cartpole.force_limit` (N, mutable, `0.5..=max_force`) caps |F| after the lab controller; applied at the tick boundary and recorded in the trace only when it changes, so without changes the hash equals the lab's. Rejections never enter the trace.
- Caveats: a `pump` coalescing several ticks applies a param change at the sync boundary; no lab change was needed (its public API sufficed).
- Run: `cargo run -p gogamen-studio --offline --example cartpole_svg -- [steps] [seed] > cartpole.svg`.

## Generic entity inspector (GOG-92)

`inspector` module: pure data model + text/SVG panel (no Bevy/egui).

- `Inspector::select(EntityRef{id, epoch})`, `inspect(&SnapshotView) -> InspectorView` (rows with typed
  value, unit, provenance, age, quality, plus snapshot revision/tick).
- Stable selection: a recycled `id` (different `epoch`) gives `Vanished(Recycled)` with no rows; a missing
  id gives `Vanished(NotPresent)`; field age over the threshold (global or per-descriptor) or a regressed
  snapshot revision gives `Stale(reasons)` and per-row `STALE` flags.
- `Lod` (`Minimal < Standard < Full`) per subscription filters fields via `FieldSchema` descriptors
  (`min_lod`, label, `stale_after`). Fields without a descriptor render with a default (`Full`) row, so
  new fields need no panel code.
- Grid lab: `grid_snapshot_view(&GridWorldView, &Observation)` exposes the agent and every cell
  (epoch derived from the session generation, so `Reset` invalidates selections); `grid_schema()`.
- Run: `cargo test -p gogamen-studio --offline inspector`.

## Viewer temporal isolation (GOG-115)

`viewer_isolation` module + `tests/viewer_isolation.rs` + `examples/viewer_isolation.rs`. Rehearses four
viewer modes under the same seed/workload (headless `GridRollout`): `NoViewer`, `Normal`, `Slow`
(consumer sleeps per frame), `Disconnected` (consumer drops its subscription and calls `close_viewer`
mid-run). The **virtual sensor** is `GridRollout::step`; the **viewer** is a read-only consumer thread
(bounded `ObservationHub` of GOG-52 + `SoftwareSvgBackend` render) that cannot feed back into decisions.
Control commands (`SetSpeed`, via `ControlPort`) are sent and ACKed every tick.

- Run: `cargo run --release -p gogamen-studio --offline --example viewer_isolation -- [ticks] [seed] [wall_ms] > report.md`
  and `cargo test -p gogamen-studio --offline --test viewer_isolation`.
- Read the report: per mode, p50/p99/max (ns) of sensor step, publication, whole tick and control ACK;
  `misses` = ticks over the *configured* tick budget; `produced/published/not_published/delivered`,
  `dropped(ring)`, `undelivered` and `gaps = produced - delivered = dropped + undelivered +
  not_published`; `q_max/cap` is the max queue depth sampled by the consumer (always <= capacity);
  allocations of the producer thread (sensor vs publish) via `CountingAllocator`, or `NOT_MEASURED` if
  the binary does not install it.
- Determinism: under the **logical budget** (N ticks) the per-tick decision/trace hash sequence is
  identical in all four modes (hard-asserted). Under a **real-time deadline** the number of completed
  ticks may differ per mode; it is recorded (common prefix still identical) and never asserted equal.
- Limits: NOT a WCET claim; percentiles are over N samples on this host (debug builds are much slower
  than `--release`); no zero-impact promise (the viewer shares CPU, locks and allocator with the engine).
  Tests assert only structural properties (bounded depth, controls acknowledged, gap accounting) plus
  very generous sanity bounds; timings are only reported. `Disconnected` makes the producer wait
  (bounded, outside the timed region) for the viewer to leave at mid-run. Queue depth is sampled by the
  consumer, so it can under-sample peaks.

## Timeline and offline replay (GOG-94)

`replay` opens a recorded Grid run with no engine, sensors, actuators or network.

- `Recording`: versioned text format (`GOGAMEN-GRID-REC v1`: seed, map rows, `last_tick`, one `frame` line
  per tick with tick, revision, agent pos, rewards, collisions, terminal). Written from the real
  `GridRollout` by `recorder::record_grid(seed, map, ticks)` (engine side; `replay.rs` never imports it).
  `Recording::open(bytes)` -> `OpenError::{IncompatibleVersion{found,supported}, NotARecording, Corrupt}`;
  `probe_version` reads the header only. Missing ticks become `Gap{from,to}` (also a truncated tail);
  they are shown in `ReplayView`, `scene()` is `None` inside a gap, nothing is interpolated.
- `ReplayPlayer<'a>` takes `&'a Recording` (read-only): `seek`, `step_forward/back`, `play/pause`,
  `set_speed`, `advance(wall_dt)`, `view()`, `scene()` (same `project_grid_scene` as live), `render_svg()`.
  The playback clock (`clock_secs`, wall time) is separate from the recorded virtual tick: ticks move
  by `wall_dt * speed * 10` only while playing; seek/step never touch the clock.
- Run: `cargo test -p gogamen-studio --offline replay` and
  `cargo run -p gogamen-studio --offline --example replay_svg -- 12 5 42 > /tmp/replay.svg`.

## Minimal scenario editing (GOG-107)

`scenario_editor` module: headless design-mode model for Grid scenarios. NOT a general-purpose editor.

- Single configuration contract: a scenario is a plain `ExperimentSpec` JSON (`gogamen-spec`), with one
  `grid_world` node whose `map_ascii` config holds the map (`#` wall, `.` free, `S` start, `G` goal,
  `H` hazard; rows joined by `\n`); the seed is `ExperimentSpec::seed`. The mapping and validation live in
  `gogamen-lab-grid` (`scenario` module: `grid_experiment_spec`, `grid_scenario_from_experiment`,
  `run_grid_scenario`) and are shared by Studio and the CLI. There is no Studio-only file format.
- `ScenarioDraft`: `apply(ScenarioEdit)` with `AddResource/RemoveResource/MoveResource` (start, goal,
  hazard), `AddObstacle/RemoveObstacle`, `SetSeed`. Each edit builds a candidate spec that is validated
  (spec contract + map + engine instantiation) BEFORE it replaces the committed definition. An invalid
  edit returns a typed `EditError` and leaves the definition (and undo/redo stacks) untouched.
- `undo()/redo()` only move between already-validated draft definitions. The draft holds no reference to
  `SessionHost`, controls, recorder or replay, so design-mode undo never rewinds a running session.
- `save(path)/load(path)` use `ExperimentSpec` canonical JSON; the saved file runs without Studio:
  `gogamen run-grid-spec --json-file scenario.json [--ticks N] [--seed S]` (seed defaults to the spec's).
- Caveat: validation is Grid-specific (`grid_world` is not in the generic plan-runtime catalog, so
  `run-spec` does not execute it; `run-grid-spec` does).
- Run: `cargo test -p gogamen-studio --offline scenario_editor`.

## Verify

```
cargo test -p gogamen-studio --offline
cargo tree -p gogamen-cli --offline | grep -Ei 'studio|bevy|egui'   # no output
cargo tree -p gogamen-core --offline | grep -Ei 'studio|bevy|egui'  # no output
```

## Automation graph and events view (GOG-106)

`automation` module: pure projector `project_automation_scene(&AutomationView) -> Scene2D` (plus
`render_automation_svg`). `run_automation(title, &AutomationScenario)` drives the REAL
`gogamen-lab-automation` `ServiceSupervisor` headless (virtual monotonic clock, synthetic events,
no FMC, no real Jev, no network) and compiles the equivalent `SystemSpec` to show the compiled plan
graph (nodes, dependencies). The trace shown is the supervisor's own (`trace_hash` equals
`ServiceSupervisor::replay`). Views: compiled plan nodes/edges, workflow states/transitions with the
active state and active transition highlighted, and an events/timings timeline with `RETRY n/m`,
`ERROR-BRANCH`, `TIMEOUT-BRANCH`, virtual timestamps and deltas.

- Effects are `PROPOSED (not executed)` (`Hypothetical`) unless the fake actuator recorded them at that
  tick (`EXECUTED`, `Observed`); the distinction exists in data (`EffectStatus`) and in the scene.
- Layout is layered and deterministic (topological depth -> x, index -> y); not a graph editor.
  Global `OnError`/`OnTimeout` branches are annotations on the target state.
- Common selection: nodes/states/transitions/steps carry stable `EntityId`s; `AutomationView::snapshot_view()`
  feeds the generic `Inspector`.
- Textual guard test: the projector contains no dynamics, planner or supervisor code.
- Limits: deterministic mode only (no cognitive provider path); the plan is used for structure, not executed.

```
cargo run -p gogamen-studio --example automation_svg --offline -- [incident|retry|timeout] > automation.svg
```

## Cognitive and communication inspector (GOG-96)

`cognitive` module: pure projector `project_cognitive_scene(&CognitiveView) -> Scene2D` (plus
`render_cognitive_svg`). Input is data only (no network/LLM/planner/dynamics; guarded by a textual test).
Reuses core types: `BinaryBelief`, `EvidenceRef`, `Confidence`, `ObservationKind`, `EpisodicRecord`,
`SituatedMessage`.

- Hypotheses sharing a `subject` with a different `statement` are flagged `CONFLICT`; each evidence row
  shows source (human / direct-sensor / agent / unknown), claim kind, age (`view.tick - observed_tick`),
  reported confidence and source reliability separately.
- `Authorization` defaults to `Confidential`: message sender/recipient/content and episode
  owner/summary/features render as explicit `[redacted]`; `Authorized` shows them. Beliefs, evidence and
  intentions are structured and always shown.
- Missing data is `Known::Unknown` and renders `unknown` (never guessed; future-dated ticks too).
- Only `CausalLink`s recorded in the view are drawn, and only when both endpoints exist; others are counted
  as `unresolved`.
- Fixtures (`cognitive_fixtures`): `grid_language_fixture(updated, auth)` (two contradictory hypotheses about
  one cell, before/after a sensor update computed with core `BinaryBelief`) and `life_signal_fixture(auth)`.
  Example: `cargo run -p gogamen-studio --example cognitive_svg -- grid after authorized`.

## Language World: partial map vs ground truth and human input (GOG-97)

`language_world_lab` module: two pure projectors, `project_agent_scene(&AgentView)` (beliefs, update
traces and epistemic actions only; no map) and `project_evaluator_scene(&EvaluatorView)` (adds the real
map and belief/truth mismatches), selectable with `ViewMode::{Agent,Evaluator}`. The headless
`LanguageWorldRollout` drives the real lab pieces: `GroundTruthGrid` occluded sensor, `GridBeliefMap`
(core Bayesian updates), `BeliefGridPlanner` fed only a `PlannerInput` (beliefs + own position), the
offline semantic grounding path `language_world::ground_human_notice`, and core's epistemic `Ask`
(`EpistemicController`, `LocalCommunicationBus`).

- Human warnings show before/after `P(obstacle)` per cell, source=human, reliability and evidence id; a
  false warning is corrected by later direct evidence or stays flagged `uncorroborated(source=human)`.
- `Ask` uses a deterministic `OfflineAskProvider` with simulated latency (virtual clock, no sleeps).
  Denied permission, timeout and unavailable provider are recorded as non-blocking outcomes and the
  observe-plan-act loop continues the next tick.
- Textual guards: render code has no dynamics/planner/belief update; `PlannerInput` has no truth field.

Example: `cargo run -p gogamen-studio --example language_world_svg -- [agent|evaluator] [true|false]`.

## Counterfactual debugger from a checkpoint (GOG-99)

`counterfactual` branches a saved `Recording` (GOG-94) at a tick and compares the original action with an
alternative. Never a physical rewind and never an external effect.

- `Checkpoint::capture(&Recording, tick)` rebuilds the real core `GridWorldModel` (`BranchableSnapshot`, GOG-5)
  at `tick`, verifying every recorded frame; each `run(alt, horizon)` `restore`s the snapshot into a fresh model, so
  runs are repeatable and isolated and the parent (`&Recording`) is read-only.
- Every action (original and alternative) goes through the independent `ActionGuard` (GOG-7); a rejected one is
  stored as `Rejection` and not executed. Branch header: parent `RecordingId`, tick, seed, `ModelRef`.
- `CheckpointError::{TickOutOfRange, MissingFrame, IncompleteHistory, ModelUnavailable, Snapshot, ReconstructionMismatch}`
  reject insufficient checkpoints.
- `CounterfactualView` keeps observation (recording) apart from predictions (model) with per-tick `Fidelity`;
  `project_counterfactual_scene` is a pure projector (textual guard test: no dynamics/model/guard in render code).
- Run: `cargo run -p gogamen-studio --offline --example counterfactual_svg -- 3 4 42 > /tmp/cf.svg`.

## Arena synchronized views (GOG-101)

`arena_view` compares two `Recording`s (GOG-94) side by side through one shared viewport and one logical tick
cursor (`ArenaSession::seek`). Data-only: no planner, dynamics or engine runs in the UI.

- Alignment is by logical tick only: each panel shows its exact recorded frame, an explicit `GAP` marker inside a
  gap, or `ENDED ... no data` past its end. Nothing is repeated, extrapolated or interpolated.
- Each panel shows a distinctive run id (`A-`/`B-` + recording hash) and its seed; panels reuse `project_grid_scene`.
- Metrics and the winner come from `gogamen-arena` (`PairedSeedOutcome`, GOG-100), never recomputed here. Without a
  matching paired outcome (missing report, seed or map mismatch) the verdict is `not comparable` and no winner is shown.
- Textual guard: the render section has no dynamics/planner/engine. Run: `cargo run -p gogamen-studio --offline --example arena_svg -- 6 42 7 > /tmp/arena.svg`.

## Integrated tour and optional distribution (GOG-116)

`src/tour.rs` registers the six real labs plus a replay session in the common `StudioShell`
(static registry, no dynamic loading) and defines one reproducible walk, `TOUR_PLAN`:

| Lab id | Lab | Ticks advanced | Notes |
|---|---|---|---|
| `lab.grid` | GridWorld | 6 | live Grid |
| `lab.cartpole` | CartPole | 10 | |
| `lab.kart` | Racing (Kart) | 8 | |
| `lab.life` | Life | 8 | first living organism selected through `LifeProbe::select` |
| `lab.language_world` | Language World | 3 | agent view, offline Ask provider, canonical human notice |
| `lab.automation` | Automation | 3 | scenario steps with `tick <= observed tick` are replayed on the real supervisor |
| `lab.replay` | Replay (Grid) | 9 | `ReplayPlayer` over a recorded Grid run; at the same tick it renders exactly the live Grid scene |

For every stop `tour::drive_step` selects the lab, submits `Step` commands through the control
port (the session starts paused), syncs the lab and returns `StepFacts` (tick, primitive count,
labels, scene hash). The same function drives the CI tests and the real window. The window top bar
adds `Step` and `Reset` buttons that go through the same control port. Not covered (no API in the
shell): clicking entities to select them (Life selection is programmatic via the probe),
Play/Resume at wall-clock speed in the tour (it uses deterministic `Step` only), and real pointer
input (zoom/pan are exercised programmatically in `window_real`; real mouse input is NOT_RUN).

Headless CI contracts (no display, no feature): `cargo test -p gogamen-studio --offline --lib tour`
checks the registry, non-empty scenes, determinism by hash (two independent tours), pairwise
distinct scenes, expected labels per lab, Life selection, live/replay agreement, replay past the
recording end, and Pause/Step/Reset on every lab.

Real graphical tour (needs a display; `NOT_RUN` otherwise, never a pass):

```text
cargo run  -p gogamen-studio --features window --example studio_tour -- --script /tmp/g116
cargo run  -p gogamen-studio --features window --example studio_tour                   # interactive
cargo test -p gogamen-studio --features window --test tour_real                        # 2 runs, hash-stable
```

`tests/tour_real.rs` re-executes itself twice (a process can open only one winit event loop) and
asserts: one capture per lab, scene pixels visible, captures pairwise distinct, identical window
hashes in both runs. Without `DISPLAY`/`WAYLAND_DISPLAY` it prints `NOT_RUN: NotRunHeadlessHost`.
Known limitation of the window painter: label font size is fixed in points, so dense scenes
(Automation, Language World log text) overlap when the fit zoom is small.

### Optional distribution, permissions and licenses

- `gogamen-core` and `gogamen-cli` do not depend on Studio or any graphical crate (the unit test
  `dependency_isolation` plus `cargo tree -p gogamen-core -p gogamen-cli --offline | grep -Ei
  'studio|bevy|egui|wasmtime|rerun|mujoco|eframe|winit'` must print nothing). The CLI therefore
  runs without GPU or UI.
- Studio is a separate crate. Its default build is `std`-only; the graphical stack is only
  compiled with `--features window` (adds 123 crates over the default Studio graph, `cargo tree -e
  normal`).
- Runtime permissions of the `window` build: access to the Wayland or X11 socket, an OpenGL 3.3+
  driver (loaded at runtime with `libloading`/`dlopen`: libEGL/libGL, libxkbcommon, wayland and X11
  client libraries). No network crates (no tokio/reqwest/openssl), no clipboard (`arboard`), no
  file dialogs. The only file writes are the PNG captures of `--script DIR` / `tour_real`.
- Licenses of the 123 window-only crates, from `cargo metadata` (`cargo deny list` confirms the
  same set; no `deny.toml` is committed):

| License expression | Crates |
|---|---|
| MIT OR Apache-2.0 (also `Apache-2.0 OR MIT`, `MIT/Apache-2.0`) | 69 (incl. eframe, egui, egui_glow, egui-winit, epaint, emath, ecolor, accesskit, image, png, x11rb) |
| MIT | 23 (incl. wayland-client, smithay-client-toolkit) |
| Apache-2.0 | 6 (incl. winit 0.30.13, glutin 0.32.3) |
| MIT OR Apache-2.0 OR Zlib (any order) | 8 (incl. glow 0.17, raw-window-handle, bytemuck, miniz_oxide, cursor-icon, xkeysym) |
| Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT | 4 (rustix x2, linux-raw-sys x2) |
| Unlicense OR MIT | 2 (memchr, byteorder-lite) |
| BSD-3-Clause OR Apache-2.0 | 2 (moxcms, pxfm) |
| (MIT OR Apache-2.0) AND Unicode-3.0 | 2 (unicode-ident, unicode-properties) |
| Zlib | 1 (foldhash) |
| ISC | 1 (libloading) |
| 0BSD OR MIT OR Apache-2.0 / BSD-2-Clause OR Apache-2.0 OR MIT / Apache-2.0 AND MIT | 3 (adler2, zerocopy, dpi (`Apache-2.0 AND MIT`)) |
| `Apache-2.0 OR GPL-2.0-only` | 1 (`self_cell`; choose Apache-2.0) |
| `(MIT OR Apache-2.0) AND OFL-1.1 AND Ubuntu-font-1.0` | 1 (`epaint_default_fonts`: the embedded fonts carry OFL-1.1 and the Ubuntu Font Licence; redistributing a binary with default fonts must keep those notices) |

All are permissive; no copyleft is mandatory (the only GPL appears as an alternative of a dual
license). The Studio crate itself inherits the workspace license.

## Verify (GOG-116)

```
cargo test -p gogamen-studio --offline
cargo test -p gogamen-studio --features window --offline
cargo clippy -p gogamen-studio --offline --all-targets -- -D warnings
cargo clippy -p gogamen-studio --features window --offline --all-targets -- -D warnings
cargo tree -p gogamen-core -p gogamen-cli --offline | grep -Ei 'studio|bevy|egui|wasmtime|rerun|mujoco|eframe|winit'   # no output
```

## 3D robot view: poses, joints and a simple map (GOG-105)

`lab.robot3d` (registered in the tour, last stop) shows the navigation `Sim` of `labs/mujoco`
(corridor, scanner dropout from sim tick 100) as a wireframe 3D scene. There is **no Bevy**: the 3D
view is a deterministic perspective/orthographic projection (`gogamen_projection::Scene3D` +
`OrbitCamera`) that produces the same `Scene2D` primitives used by the SVG and egui backends, so
it needs no GPU, no mesh/asset loading and no physics of its own.

### Coordinates and units
- World frame: right-handed, **x forward, y left, z up**, metres; angles in radians; **yaw is
  counter-clockwise about +z** (yaw 0 -> +x, yaw pi/2 -> +y).
- Axes are drawn at the origin: x red "x (forward)", y green "y (left)", z blue "z (up)"; a HUD line
  states the units. Grey = grid and walls; blue = true (Observed); orange = estimate and rays
  (Inferred); purple/red = hypothetical drift / stale.
- 1 control tick = 10 sim ticks (dt = 0.01 s).

### Camera
Orbit camera (target, azimuth, elevation clamped below 89 degrees, distance, 60 degree vertical FOV).
Input mapping in the window: left-drag orbit, right-drag or shift-drag pan, wheel zoom, click select
(picking by distance to the projected segments), double-click or "Reset view" reset,
"Ortho/Persp" toggle. The same inputs exist headlessly as `StudioShell::view_input(ViewInput)`.

### Graphic vs physical state
`ViewState` (camera + selection) is separate from `Robot3dPhysics` / `Robot3dSnapshot` (the `Sim`
and its snapshot). Camera/selection inputs never reach the `Sim`; tests compare the trace hash
with and without camera moves and guard the struct bodies by source text. A snapshot has no camera.

### Stale data and uncertainty
Scan age above the safety threshold (`stale_ticks`, 0.40 s) switches to a red "STALE: scan age ..."
label, red rays/estimate and a red cross; "no scan yet" is also treated as stale. The ellipse on the
ground is a **heuristic drift region** (radius 0.05 m + 0.20 m/s * scan age, b = a/2), not a filter
covariance, and is labelled so.

### Run
```text
cargo run -p gogamen-studio --example robot3d_svg -- /tmp/g105        # deterministic, no window
cargo run -p gogamen-studio --features window --example robot3d_window -- --script /tmp/g105
cargo run -p gogamen-studio --features window --example robot3d_window   # interactive
```

### Caveats
Wireframe only (no solids, hidden-line removal or textures); labels may overlap near the robot;
the scripted captures use programmatic view input - real pointer input was not exercised in
automated runs. Depends on `gogamen-lab-mujoco` (studio only; `gogamen-core` and `gogamen-cli` stay
free of it).
