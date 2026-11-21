# gogamen-lab-mario (GOG-62, GOG-63)

Mario lab scaffold: an `EmulatorPort`, a hash-gated user-ROM command and a free
synthetic fixture so the whole pipeline runs in CI without any ROM.

## What is verified here, and what is not
- **Verified (CI, no ROM)**: port contract, ROM hashing/known-hash gate, typed
  errors, metrics, baselines vs a snapshot Monte Carlo planner, reproducible
  trace hashes. All on `SyntheticPlatformer`, labelled
  **"synthetic fixture, not Mario"** (original code, no third-party assets).
- **NOT_RUN**: no emulator backend is compiled in, so a real Mario level has
  never been run by this crate. A ROM whose hash is in the known list still ends
  in `NOT_RUN: backend_unavailable`. The command never claims a level was completed
  unless it is the synthetic fixture (`fixture_goal_reached`).

## Bring your own ROM
No ROM or BIOS is distributed and none may be added to the repo. Supply your own
legally obtained dump by path:

```
cargo run -p gogamen-lab-mario --example mario_run -- --rom /path/to/your.rom
cargo run -p gogamen-lab-mario --example mario_run -- --fixture [--seed N]
```

The file's SHA-256 (over the whole file, headers included) is compared to
`known_roms.conf` (`--known FILE` to override). The shipped list is **empty on
purpose**: no hash is asserted as an official dump. Add the hash of the dump you
verified yourself (`sha256 <64 hex> <label>`).

Exit codes: `0` ok (fixture), `2` typed error (unreadable file, empty file, hash
not in list, bad config), `3` NOT_RUN (no ROM, or known ROM without backend).

## Metrics (fixture)
Per policy: max x, deaths, actions used, planning cost (simulated decision-steps
on snapshots), live frames, frames/s (wall clock, incl. planning; not hashed),
and `trace_hash` (SHA-256 of pad+x+y+deaths per decision; identical across runs
with the same seed). Policies: `random`, `always_right`, `monte_carlo`.

## Coach (GOG-63)
`coach` module: asynchronous, human-like advice that can only change the
planner's *hypothesis* (hazard prior) or an *authorised preference/objective*
(`cautious`, height bonus); it has no path to the pad.
- Coach output is untrusted text -> `AdviceCatalog::admit` -> `CoachAdvice`
  (`gap_ahead`, `avoid_spikes`, `climb_platform`, `slow_down`). Anything else
  (e.g. `emit_pad:RIGHT+JUMP`) is a typed `AdviceError`. `coach/advice.rs` and
  `mock.rs` never name `PadState`/`EmulatorPort` (textual test); the planner's
  pad still passes a `PadGuard`.
- Delivery uses a virtual clock (tick = decision index): `AdviceQueue` ordered by
  `(deliver_tick, arrival)`, per-advice `max_age` deadline (stale -> `Expired`),
  `ttl`. The decision loop never waits: late, wrong or mute coaches just have
  no/less effect (a coach whose advice never arrives gives a trace identical to
  no coach).
- Advice is a soft prior (biases rollout sampling/score, never prunes candidates)
  so rollouts can contradict a wrong hint.

```
cargo run -p gogamen-lab-mario --example coach_eval [-- --seeds N]
```
Compares no coach / truthful (latency 2) / truthful but late (30) / lying
(+ out-of-catalog text) / mute-after-20, over fixed seeds, for a myopic (1x2)
and a comfortable (3x5) planner; reports max_x, deaths, planning steps,
emission-to-effect latency (ticks) and an overall `suite_hash` (identical across
runs). **Honest scope**: everything is on `SyntheticPlatformer` with
deterministic mock coaches (not a real LLM/human, not Mario). The truthful
coach helps only the myopic planner; the comfortable planner already solves the
fixture. With a real ROM this is **NOT_RUN**. The `gogamen-experimental` hub
(`CandidateValidator`) is not used; the catalog here is a local equivalent.

## Limits
- The planner is a local Monte Carlo rollout over `EmulatorPort::snapshot/restore`;
  it is **not** wired to `gogamen-core` FMC, `ActionGuard` or `ObservationEnvelope`
  (that needs a `GenerativeModel`/`State` encoding like `gogamen-lab-retro`'s
  adapter). Future work once a real backend exists.
- Fixture speed says nothing about real emulator speed.
- A real backend (libretro via `libloading`, or Stable-Retro) would implement
  `EmulatorPort` and must decode features from RAM; none exists yet.
