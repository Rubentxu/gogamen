# gogamen-experimental

Opt-in experimental ports (GOG-82): visual/Box perception and generative "System Two".
Mock/fixture providers only; no network, no real models.

* **Off by default.** `ExperimentalConfig::default()` enables nothing; a disabled, unregistered,
  unavailable or failing capability returns `Outcome::Abstain(AbstainReason)` (typed). `ExperimentalHub::discover()` reports
  availability ("use if available").
* **Generated output is untrusted.** `CandidateValidator` checks: labelled generative origin and
  `ObservationKind::Proposal`; grounding (every claim cites known, non-generative evidence);
  capability catalog (module/operation/version/permissions); known goals; and the core
  `ActionGuard` (only `Allow` counts). Actions are only obtainable as `GuardedAction` (no public
  constructor). Results are advisory, outside the critical control loop.
* **No coupling.** `gogamen-core`, `gogamen-cli` and the labs do not depend on this crate.
  The `lab-baseline` feature (default off) pulls `gogamen-lab-mujoco` for the cost/benefit harness.
  `gogamen-jev` is not used (no verified contract); a Jev-backed provider could implement
  `GenerativeProposalPort` later behind its own feature.

```
cargo test -p gogamen-experimental --offline
cargo run -p gogamen-experimental --features lab-baseline --example cost_benefit --offline
```

## Cost/benefit caveat

The harness compares a local keyword detector with a *scripted mock* generative provider on 20
synthetic utterances from the lab dialog fixtures. The mock's errors and declared latency/units
are inventions of the harness. The results validate the pipeline (rejection counting, determinism,
abstention) and are **not** evidence about real models, real latency or real cost. The baseline is
expected to win on these keyword-style fixtures.
