# SP-5a -- Calibration Enablement Design (#140)

**Goal:** Land the two HW-free prerequisites that de-risk the eventual
`calibrated` flip: a broadcast-provenance fix that makes the timer-reset
single-source guard config-independent, and a cross-language contract test
proving a real emulator-produced `origin_d.json` round-trips through the
Python inference engine.

**Architecture:** Two independent deliverables, both pure software, both
fully testable without Phoenix silicon. Deliverable 1 tags every queued
broadcast with its provenance (genuine origination vs L1-relay re-injection)
and records flood sources only for genuine originations. Deliverable 2 adds a
committed real-export fixture (guarded against staleness by a Rust liveness
assertion) plus a Python test that consumes it through `run_engine`.

**Tech stack:** Rust (emulator core, `src/device/`, `crates/xdna-archspec`),
Python 3.13 (inference engine under `tools/inference/`), pytest.

## Position in the arc

SP-5a is the first of three SP-5 sub-projects (the silicon-characterization
capstone of the timer-sync faithful-broadcast arc):

- **SP-5a -- calibration enablement (this spec, no HW):** clear the two §9a
  gating prerequisites to the extent they are testable without measured
  constants.
- **SP-5b -- measurement apparatus (mostly no HW):** build and emu-validate
  the measurement kernel(s) + analysis (tile-distance sweep and/or the
  route-3b direct on-silicon timer-read).
- **SP-5c -- Phoenix characterization + go-live (HW):** run the apparatus on
  Phoenix, resolve the model-structure questions (per-hop uniformity, d_h/d_v
  collapse, intra-tile sign), extract the four constants, set them, flip
  `calibrated`, update the regression guards, and validate the now-live causal
  decomposition against hardware. The P1 in-domain round-trip gate folds in
  here.

The two SP-5 gating prerequisites are recorded in
`docs/superpowers/specs/2026-06-30-sp4b-skew-export-design.md` §9a. SP-5a
addresses prerequisite #1 in full and the testable-now half of prerequisite #2.

## §1 Scope and non-goals

**In scope (SP-5a):**

1. Broadcast-provenance fix -- closes §9a prerequisite #1 (fixpoint channel-15
   multi-source coverage) robustly and config-independently.
2. Sidecar contract-verification test -- closes the testable-now half of §9a
   prerequisite #2 (proves a real Rust-produced sidecar is consumable by the
   Python engine).

**Explicit non-goals (deferred to SP-5b/5c):**

- Flipping `calibrated` to `true` (SP-5c). The three regression guards that
  assert the uncalibrated/zero-constant state stay green and unchanged:
  `crates/xdna-archspec/src/runtime.rs:799` (`broadcast_timing_defaults_uncalibrated`),
  `src/interpreter/engine/coordinator.rs:4077` (sidecar `calibrated == false`),
  `src/device/state/effects.rs:1248` (`broadcast_timing_consts_default_to_zero`).
- Setting the four timing constants (`d_h`, `d_v`, intra-tile core/mem
  offsets) -- those are SP-5c silicon measurements.
- The full in-sweep consumption integration: making `trace-sweep.py` an engine
  caller (adapting decoded events into the `run_dir/batch_*/hw/trace.events.json`
  layout `load_fired` expects, constructing a structural ledger, computing
  candidate pairs, folding `rep["causal"]` into the per-tile summary). That is
  SP-5c go-live work -- pre-calibration it produces no causal output to
  validate, so building it now would be integration without a validation
  target.
- Any hardware run. SP-5a touches nothing that requires silicon; the EMU is
  exercised only in-process / functionally.

## §2 Deliverable 1 -- broadcast provenance

### §2a Problem (confirmed defect)

The flood-source recorder in `propagate_broadcasts_with_timing`
(`src/device/state/effects.rs:603-605`) is:

```rust
if channel == 15 {
    self.channel15_flood_sources.insert((col, source_row));
}
```

It inserts whatever tile is currently draining a channel-15 entry, with no way
to distinguish a genuine timer-reset **origin** from a **relay**. The fixpoint
re-floods channel 15 generically: every reached tile with an L1 controller is
re-injected via `tap_l1_interrupt(EventModuleType::Pl.broadcast_event_base() + channel)`
= event `110 + 15 = 125` (`effects.rs:660-663`). If a reached shim's L1 switch
has an IRQ_EVENT slot mapped to event 125, that slot enabled, and its
`IRQ_NO == 15`, then `tap_l1_interrupt` pushes channel 15 back into *that
shim's* `pending_broadcasts` (`src/device/tile/mod.rs:719-721`). On the next
`propagate_broadcasts_fixpoint` iteration (`effects.rs:702-703`) the shim
drains channel 15 and the recorder inserts the **shim** as a second source.

`export_origin_d_sidecar` (`src/interpreter/engine/coordinator.rs:378-408`)
then sees `flood_sources().len() == 2` and emits `flood_source: null` with empty
`modules` -- silently degrading a legitimate single-origin timer-reset flood to
the multi-source failure path. The set is never cleared (initialized empty at
`src/device/state/mod.rs:144`, no `clear()`), so this is durable for the run.

Channel 15 is special-cased in exactly one place -- the recorder guard above.
Everywhere else (the L1 latch, the fixpoint scan, drain) it is handled
identically to channels 0-14. No existing test drives the fixpoint with channel
15: every fixpoint test uses channel 2 or 3
(`effects.rs:877`, `:899`, `:1068`); every channel-15 test uses the single-drain
`propagate_broadcasts` or hand-fabricated multi-source state
(`coordinator.rs:4063`, `:4096`).

**Root cause (semantic):** a relay tile re-emitting channel 15 as L1-interrupt
*transport* is not a timer-reset *origin*. The recorder conflates the two uses
of the channel. The fix tags the distinction at the source.

### §2b Data model

`pending_broadcasts` currently stores bare channel numbers
(`src/device/tile/mod.rs:310`: `pub pending_broadcasts: Vec<u8>`). Grow the
element to carry provenance:

```rust
/// How a queued broadcast entered the network. A flood "source" is a tile
/// that ORIGINATES a channel (Event_Generate / hardware-error seeding); a
/// tile that re-emits a channel because its L1 controller latched and drove
/// its IRQ_NO line is a RELAY, not a source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BroadcastProvenance {
    Originated,
    Relayed,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PendingBroadcast {
    pub channel: u8,
    pub provenance: BroadcastProvenance,
}
```

`pending_broadcasts: Vec<u8>` becomes `Vec<PendingBroadcast>`;
`drain_pending_broadcasts` returns `Vec<PendingBroadcast>`. Binary provenance is
sufficient -- a relay-of-relay is still `Relayed` (only genuine origination is
distinguished). YAGNI: no per-producer granularity beyond origin/relay.

### §2c Producer tagging

There are exactly two non-test producer functions (verified: the only
`push`/`extend` mutators of `pending_broadcasts` in `src/`):

- **`seed_broadcasts_for_event`** (`src/device/tile/mod.rs:743-758`) -- the
  genuine-origination path, shared by the Event_Generate register path and the
  `raise_instr_error` hardware-error path (`effects.rs` Event_Generate handler
  near `:403-413`; `src/interpreter/core/interpreter.rs:32`). All entries it
  pushes are tagged `Originated`.
- **`tap_l1_interrupt`** (`src/device/tile/mod.rs:708-722`) -- the L1 relay
  path ("L1 is thus a second independent producer into the broadcast network",
  per its own doc). All entries it pushes are tagged `Relayed`.

### §2d Recording-site change

`effects.rs:603` becomes:

```rust
if pb.channel == 15 && pb.provenance == BroadcastProvenance::Originated {
    self.channel15_flood_sources.insert((col, source_row));
}
```

(with the `for &channel in &channels` loop iterating `PendingBroadcast` entries
and `channel` reads becoming `pb.channel`).

### §2e Propagation is unchanged (load-bearing)

The provenance tag gates **only** the flood-source recording. A `Relayed`
channel-15 entry still propagates fully: the fixpoint still re-floods it, L2
interrupts still latch, trace events still fire. This matches the hardware --
the L1->L2 interrupt chain is a real signal that must flood; it simply is not a
timer-reset origin. Concretely: `propagate_broadcasts_fixpoint`,
`broadcast_origin_d`, the `is_empty()` scan, and the L2/trace notifications are
untouched. Only the `insert` into `channel15_flood_sources` is conditioned on
provenance.

### §2f Tests (deliverable 1)

- **New (RED -> GREEN), fixpoint + channel-15 relay.** Configure a genuine
  channel-15 origin tile plus a reached shim whose L1 switch latches event 125
  (slot mapped + enabled) with `IRQ_NO == 15`, then call
  `propagate_broadcasts_fixpoint`. Drive the relay through the real
  `tap_l1_interrupt` path (do not hand-push a `Relayed` entry -- exercise the
  actual re-queue). Pre-fix expectation: `flood_sources().len() == 2`. Post-fix
  assertion: `flood_sources()` is exactly the single origin tile.
- **Regression preserved, genuine multi-origin.** The existing
  `export_origin_d_sidecar_omits_flood_source_with_multiple_sources`
  (`coordinator.rs:4096`) pushes channel 15 to two tiles; under the new type
  these are `Originated`, so the set is still size 2 and the export still emits
  `flood_source: null` + empty `modules`. The fix must not mask real
  multi-origin floods. (Add an assertion or comment making the "two genuine
  origins" intent explicit.)
- **Termination still holds.** The self-feeding fixpoint termination test
  (`effects.rs:899`) must still pass -- provenance does not change termination.
- **Mechanical churn.** The `broadcast_wavefront_tests` and the
  export-contract tests direct-push channels
  (`effects.rs:1165,1196,1215,1226,1238,1240,1276,1290,1314`;
  `coordinator.rs:4072,4104`) and assert with `.contains(&n)`
  (`effects.rs:760,983`; `interpreter.rs:1157`). These adopt the new type:
  pushes become `PendingBroadcast { channel: n, provenance: Originated }` (a
  constructor helper, e.g. `PendingBroadcast::originated(n)`, keeps the edit
  uniform); `.contains(&n)` becomes `.iter().any(|pb| pb.channel == n)`. This
  is a compiler-guided mechanical edit.

## §3 Deliverable 2 -- sidecar contract verification

### §3a Goal and gap

SP-4b proved two halves separately: a Rust in-process export-contract test
(`coordinator.rs:4063`, asserts the in-memory `serde_json::Value` shape) and
Python tests that consume **hand-written** synthetic sidecars
(`tools/test_inference_sp4b_e2e.py`). The untested gap is the real
cross-language round-trip: does an actual Rust-**produced** `origin_d.json`
load through Python `load_model` + `run_engine` without schema/key/type drift
(module-kind strings vs the `MODULE_PKT_TYPE` translation in
`tools/inference/model_io.py`, the `flood_source`/`modules`/`calibrated` keys,
value types)?

### §3b Mechanism (fixture + liveness guard)

- **Real-export fixture.** Extend the Rust export-contract test
  (`coordinator.rs:4063`) to also serialize its real export `Value` to a
  committed fixture file (e.g. `tools/tests/fixtures/origin_d_real_export.json`),
  and assert the live export equals that fixture's content. Any Rust-side schema
  change then fails this Rust test and forces the fixture to be regenerated --
  the fixture cannot silently go stale.
- **Python consumer test.** A new Python test reads that committed fixture and
  runs it through `load_model` + `run_engine` (`tools/inference/engine.py:28`,
  `model_path=<fixture>`), asserting:
  - **(a) Pre-calibration no-op.** The real fixture carries
    `calibrated: false` (SP-5a does not flip it), so `run_engine(...,
    model_path=fixture)` returns `causal == []`. This pins that a real
    uncalibrated sidecar is a clean no-op end-to-end.
  - **(b) Calibrated overlay emits.** Construct a synthetic calibrated variant
    of the fixture (flip `calibrated: true`, populate `modules` for both
    domains of a cross-domain candidate pair, set a `flood_source`) and confirm
    `run_engine` emits a provenance-clean `causal` triple. This reuses the
    SP-4b synthetic-causal pattern (`test_inference_sp4b_e2e.py`) but seeded
    from the *real* fixture's schema, so it exercises the real key/translation
    path rather than a hand-built dict.

Rationale for fixture+guard over a live in-test EMU run: it pins the same
cross-language contract without orchestrating a flood kernel + FFI `.so`
subprocess inside the Python test. The liveness assertion neutralizes the only
real downside (staleness). The higher-fidelity live-EMU-run alternative was
considered and rejected for SP-5a (more test infra, same contract coverage).

### §3c Test placement

Follow the established flat convention (`tools/test_inference_*.py`); do not use
`tools/inference/tests/`. The Python consumer test gates on the fixture's
presence the same way sibling real-fixture tests do, but since the fixture is
committed (not a gitignored capture), it is always present -- no skipif needed.

## §4 Testing and validation

- `cargo test --lib` green: the provenance type change compiles across all
  producers/consumers/tests; the new fixpoint+channel-15 test passes; the
  genuine-multi-origin and termination regressions still pass; the Rust
  export/liveness fixture assertion passes.
- Python inference suite green: `tools/test_inference_*.py` (including the new
  contract test) pass.
- The three "still uncalibrated / still zero" regression guards (§1) remain
  asserting the unchanged pre-SP-5c state.
- No bridge/HW run required.

## §5 Risks and open mechanics

- **Type-change ripple.** Growing `pending_broadcasts` to `Vec<PendingBroadcast>`
  touches every producer, `drain`, the fixpoint scan, and direct-push/`contains`
  tests. Mitigated by a constructor helper and the compiler enumerating every
  site. Implementer confirms nothing serializes `pending_broadcasts` in a way
  that depends on its `Vec<u8>` shape (state snapshot/restore).
- **Origination-vs-relay completeness.** The fix assumes the only relay path
  into `pending_broadcasts` is `tap_l1_interrupt` and every other producer is
  genuine origination. Verified against the current producer surface (two
  functions); the implementer re-confirms no new producer was added between
  spec and implementation.
- **Fixture staleness.** Neutralized by the Rust liveness assertion (live
  export must equal the committed fixture, else the Rust test fails).
- **Genuine multi-origin semantics.** The fix preserves detection of truly
  multiple timer-reset origins (two `Originated` channel-15 floods still trip
  the guard). This is intended behavior, not a regression -- a genuinely
  multi-origin flood has an ambiguous reference clock and must surface as
  `flood_source: null`.

## §6 Reference map (verified file:line)

- Recorder: `src/device/state/effects.rs:603-605`.
- `propagate_broadcasts_with_timing`: `effects.rs:568-666`; L1 re-injection
  `:660-663`.
- `propagate_broadcasts_fixpoint`: `effects.rs:684-712`; seed `:686`; re-prop
  loop `:702-703`.
- `pending_broadcasts` decl: `src/device/tile/mod.rs:310`; `drain` `:691-693`.
- Producers: `seed_broadcasts_for_event` `tile/mod.rs:743-758`;
  `tap_l1_interrupt` `tile/mod.rs:708-722`.
- L1 latch internals: `src/device/interrupts/l1.rs:245-260` (`signal_event`),
  `:159` (`read_irq_no & 0x0F`).
- Flood-source field: `src/device/state/mod.rs:121` (decl), `:144` (init),
  `:267-268` (`flood_sources()`), `:116-120` (doc).
- Export: `src/interpreter/engine/coordinator.rs:378-408`
  (`export_origin_d_sidecar`), contract test `:4063-4083`, multi-source test
  `:4096-4110`.
- Engine: `tools/inference/engine.py:28-31` (signature), `:38-39` (model_path
  install), `:86` (`causal`).
- Python model loader: `tools/inference/loader_model.py:15-41`,
  `tools/inference/model_io.py` (`MODULE_PKT_TYPE`, `to_domain_key`).
- Existing tests: fixpoint `effects.rs:877,899,1068`; channel-15
  `coordinator.rs:4063,4096`; wavefront `effects.rs:1165-1315`; SP-4b e2e
  `tools/test_inference_sp4b_e2e.py`.
- Regression guards (unchanged): `crates/xdna-archspec/src/runtime.rs:799`,
  `coordinator.rs:4077`, `effects.rs:1248`.
- Gating prerequisites of record:
  `docs/superpowers/specs/2026-06-30-sp4b-skew-export-design.md` §9a.
