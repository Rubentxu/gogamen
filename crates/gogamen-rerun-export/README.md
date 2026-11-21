# gogamen-rerun-export (GOG-108)

Optional, **one-way** exporter from an existing Gogamen trace to a [Rerun](https://rerun.io)
recording (`.rrd`). It reads `gogamen_studio::replay::Recording` (GOG-94) or
`gogamen_runtime::ScientificTraceArchive` (GOG-93) by shared reference. It defines no new source
format and does not duplicate the scientific record: the `.rrd` is a derived, disposable artefact.

## Optional by construction

- Default build (`cargo build`): only the pure projection (`plan`) is compiled. No `rerun`,
  `wasmtime`, `bevy` or `egui` is in the graph of `gogamen-core` / `gogamen-cli`
  (`cargo tree -p gogamen-core -p gogamen-cli | grep -Ei 'rerun|wasmtime|studio|bevy|egui'` is empty).
- `--features rerun` enables the emitter (`rerun_sink`), using `rerun = 0.43` with
  `default-features = false, features = ["sdk"]` (no viewer linked). The SDK version must match the
  `rerun` CLI that opens the file (`rerun --version`).

## Usage

```text
cargo run -p gogamen-rerun-export --features rerun --example export_fixture -- out.rrd grid
cargo run -p gogamen-rerun-export --features rerun --example export_fixture -- out.rrd scientific [--allow-messages] [--allow-secrets]
rerun out.rrd            # or: rerun --serve-web out.rrd
```

API: `plan_from_recording` / `plan_from_archive` (pure, no Rerun) and
`rerun_sink::{export_recording, export_archive}`. Writing goes to `<out>.partial` and is renamed
only on success; on failure the partial file is removed, so a failed or skipped export leaves the
source and replay untouched (they are only borrowed immutably).

## Mapping

| Source | Rerun |
|---|---|
| tick | timeline `tick` (sequence, dimensionless) |
| simulated time | timeline `sim_time` (duration, **seconds**; Grid: `tick / 10`) |
| trace revision | timeline `revision` (sequence) |
| series (reward, collisions, obs/action) | `Scalars` under `gogamen/series|obs|action`, unit in the series name (`total_reward [reward]`) |
| Grid map / agent | static `Boxes2D` / per-tick `Points2D` under `gogamen/scene` |
| decisions, status, gaps | `TextLog` under `gogamen/decision|status|checkpoint` |
| views | blueprint: Scene (2D), Series (time series), Status/decisions/gaps (text log) |

## Missing data, redaction, permissions

- Missing ticks/records are explicit: scalar series get `NaN` (a visible break, never `0`), scene
  entities get `Clear`, and `gogamen/status/missing` logs the gap with its reason.
- Values already redacted in the source (`[REDACTED]`) or classified sensitive by the trace
  `SensitiveFieldPolicy` stay as `[REDACTED]` markers.
- Free text (checkpoint annotations, guard rejection reasons, failure details) is **messages**:
  withheld (`[WITHHELD ...]` marker) unless `ExportPermissions { messages: true }`. Secrets still
  in clear in a source are exported only with `messages && secrets`. Everything defaults to deny.

## Limits

- Grid `Recording` frames carry no simulated clock; `sim_time = tick / 10` s (Grid `dt` = 100 ms).
- Scientific-trace gaps are indexed by revision, so they appear on the `revision` timeline only.
- `.rrd` files are LZ4-compressed: leak checks must decode them (`rerun rrd print -vvv`), not grep bytes.
