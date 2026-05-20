# Interrupt Tier C — TDR / Context-Restart (Wedge-Recovery, Plumbed for Multi-Context)

**Status:** Spec, awaiting plan.
**Predecessor:** [Tier B Spec 1 (firmware async-error plumbing + INSTR_ERROR producer)](2026-05-19-interrupt-tier-b-firmware-mailbox-design.md) (shipped 2026-05-19, commits `d33e9d2..429df84` on `dev`).
**Tracking context:** [Tier B findings note](../findings/2026-05-19-interrupt-tier-b-firmware-delivery.md) — Tier C is the third bullet in its "what's separate" list.

## 1. Purpose & scope

Tier A landed the AIE interrupt path to the firmware-notify boundary. Tier B
added the async-error mailbox surface the driver reads. Tier C is the
context-lifecycle surface the driver acts on when a workload wedges —
specifically, the firmware/device signals a driver's TDR (Timeout Detecting &
Recovering) reads, plus enough plugin-side handling to give XRT consumers an
EIO-shaped result today instead of a hang.

The reframing that shaped this spec: **TDR is a driver-side function.** Real
`amdxdna.ko` runs `aie2_tdr.c`'s 2-second kernel timer, checks `aie2_rq_is_all_context_stuck`,
and on stuck calls `aie2_rq_stop_all`/`aie2_rq_restart_all`. The emulator's job
is not to *implement* that algorithm — it is to *expose the signals a driver
TDRs on* (per-context completion progress, context state) so that when a real
driver eventually drives the emulator, its TDR fires correctly against us.

In the current `XDNA_EMU=1` configuration the XRT plugin replaces the kernel
driver entirely, so today nobody is running TDR against the emulator and
wedged workloads hang `run.wait()`. The plugin therefore carries a *temporary*
minimal "observe-then-EIO" path until a real driver-against-emu setup
exists. Once the real driver is in the loop, that plugin code is replaced by
the driver's own TDR consuming the device-side signals.

### 1.1 In scope (this spec)

- **Per-context state model on the device** (`Context`, `ContextId`,
  `ContextState`) — single implicit context (id=0) today, per-context-shaped
  accessors throughout so multi-context expansion is a storage-only change.
- **Device-side `TdrDetector`** that classifies engine run state per cycle
  into `Progressing` / `NaturalCompletion` / `MaskPollUnsatisfied` / `Wedged`.
  Consolidates the inline classification logic currently scattered through
  `xdna_emu_run`.
- **Lift `QuiescenceDetector` and `StallDetector`** from `src/testing/quiescence.rs`
  into `src/device/tdr/` (they are device-state classifiers misfiled under testing).
  The in-process xclbin runner switches to the new path; no behavior change.
- **FFI surface additions:** new `XdnaEmuHaltReason::WedgeRecovered`, new
  per-context-state accessor (`xdna_emu_get_context_state`), `xdna_emu_reset_context`
  becomes context-aware (transitions Failed → Connected, clears Tier B sink).
- **Plugin wire-up:** `transport_inprocess` resolve-requires the new accessor,
  maps `WedgeRecovered` halt_reason to an EIO-shaped XRT command state in
  `run.wait()`.
- **Tests:** unit per layer + one integration test for end-to-end
  wedge → EIO → reset → next-submission-clean cycle, plus a Tier B/Tier C
  orthogonality test.

### 1.2 Out of scope (follow-ups, separate specs)

- **Multi-context model proper.** Storage is `Vec<Context>` with `len==1`
  today; all APIs already take `ContextId`. The actual multi-context engine
  scheduling, context lifecycle ioctls (`DRM_AMDXDNA_CREATE_HWCTX`-equivalent),
  and inter-context isolation are a separate spec when a real consumer exists.
- **`Disconnected` context state** (the driver's "needs firmware reload"
  variant). Reserved in the enum for forward-compat but not entered by Tier C
  flows.
- **Driver TDR algorithm replication.** Two-tick stuck check, `tdr_dump_ctx`
  knob, periodic-timer cadence — these are driver-side concerns. We expose the
  signals; we do not run the algorithm.
- **Real-clock TDR timeout.** Wedge detection is cycle-driven (matches the
  rest of the emulator's deterministic posture). Wall-clock cadence is left
  for the future real-driver-against-emu setup.
- **Bridge test addition.** Same rationale as Tier B Spec 1: a "deadlock kernel"
  is not in our corpus and real-NPU TDR is a driver-side concern we cannot
  isolate cleanly. The first multi-context follow-up can add one.
- **Auto-reset on wedge.** Plugin caller is responsible for calling
  `xdna_emu_reset_context` between submissions. Auto-reset is a behavioral
  knob that can be added later without changing the architecture.

## 2. Architectural decisions (locked during brainstorming)

| Choice | Decision | Rationale |
|---|---|---|
| **TDR scope** | Wedge-recovery now, plumbed for multi-context. | Maya: "contexts are gonna be important EVENTUALLY... could we do wedge-recovery only FOR NOW while plumbing for the multi-context model when it becomes the next target?" |
| **Boundary split** | Device exposes the signals a driver TDRs on; plugin does temporary minimal TDR until a real driver consumer exists. | Maya: "TDR is a DRIVER-SIDE function... the emulator explicitly ISN'T to replicate the driver's own functions, but to wire INTO the driver so the emulated NPU is driven like a real one." Mirrors Tier B's device-records / plugin-reads pattern. |
| **Context model depth** | Minimal `Context` type with single implicit `ContextId(0)`, per-context-shaped accessors throughout. `Vec<Context>` storage. | Plumbed for multi-context without overbuilding behavior today. Storage-only expansion later. |
| **Architecture (where things live)** | Approach B: device-localized. `TdrDetector` on device, classifier consolidates inline `xdna_emu_run` logic, lifted `Quiescence`/`Stall` move into `src/device/tdr/`. | Maya: "moving stuff around is fine and even good if it genuinely leads to better code. Faithfulness matters!" Trigger lives where the abstraction belongs, even at the cost of a wider diff. |
| **Reset policy** | Plugin-explicit (call `xdna_emu_reset_context` after observing wedge). No auto-reset. | Behavioral knob; Maya: "minor things like auto-resetting or not, those genuinely aren't major concerns... we care most about architectural cleanliness." |
| **Tier B coupling** | Independent paths. Wedge does not auto-emit a Tier B record; Tier B errors and Tier C wedges can co-occur and clear together on reset. | Driver-side TDR doesn't synthesize an async-error mailbox message either. |

## 3. Architecture overview

Three architectural moves:

1. **Lift `QuiescenceDetector` and `StallDetector`** from `src/testing/quiescence.rs`
   into `src/device/tdr/detector.rs`. Public API preserved. `QuiescenceDiagnosis`
   renames to `TdrDiagnosis`.
2. **Introduce `Context` / `ContextId` in `src/device/context/`.** Device owns
   `Vec<Context>` (len==1 today). Holds `state`, `completed_counter`,
   `pending_cmd_count`. All read/mutate paths take `ContextId`.
3. **Introduce `TdrDetector` in `src/device/tdr/`.** Per-context. Composes
   lifted `QuiescenceDetector` + `StallDetector` + a `poll_stall_cycles`
   budget. Exposes one method: `classify(engine, executor) -> TdrVerdict`.

`xdna_emu_run`'s body thins: each cycle it calls `classify`, transitions
context state on terminal verdicts, breaks the loop, returns the appropriate
`XdnaEmuHaltReason`. The existing inline `POLL_STALL_LIMIT` / `is_blocked_on_poll`
/ `EngineStatus::Stalled` discrimination consolidates into the detector.

The plugin observes wedges via two paths:
- **Synchronous:** new `XdnaEmuHaltReason::WedgeRecovered` halt code returned
  from `xdna_emu_run`. Plugin's `run.wait()` translates to an EIO-shaped XRT
  command state.
- **Asynchronous:** `xdna_emu_get_context_state` accessor. Cosmetic for the
  plugin today (the halt code is sufficient); load-bearing when the real
  driver is the consumer.

Recovery is plugin-driven: on observing `WedgeRecovered`, the plugin calls
`xdna_emu_reset_context`, which transitions the context Failed → Connected,
clears that context's Tier B async-error sink, and calls `engine.reset_for_new_context()`.

## 4. Components

### 4.1 Context model

```rust
// src/device/context/mod.rs

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ContextId(pub u32);

pub const DEFAULT_CONTEXT: ContextId = ContextId(0);

#[derive(Clone, Debug)]
pub enum ContextState {
    Connected,                // ready to accept submissions
    Stopped,                  // reserved (future explicit pause/resume; not entered by Tier C)
    Failed { reason: WedgeReason, diagnosis: TdrDiagnosis },
    // Disconnected reserved for multi-context (firmware reload required)
}

pub struct Context {
    pub id: ContextId,
    pub state: ContextState,
    pub completed_counter: u64,
    pub pending_cmd_count: u32,
}

impl Context {
    pub fn mark_failed(&mut self, reason: WedgeReason, diagnosis: TdrDiagnosis);
    pub fn mark_connected(&mut self);
    pub fn note_submission_complete(&mut self);  // bumps completed_counter
}
```

### 4.2 TdrDetector and verdicts

```rust
// src/device/tdr/mod.rs

pub use detector::{QuiescenceDetector, StallDetector, TdrDiagnosis};

#[derive(Debug)]
pub enum TdrVerdict {
    Progressing,
    NaturalCompletion,                       // engine Halted + syncs satisfied (today's natural-halt path)
    MaskPollUnsatisfied,                     // existing semantic, now classified here
    Wedged { reason: WedgeReason, diagnosis: TdrDiagnosis },
}

#[derive(Copy, Clone, Debug)]
pub enum WedgeReason {
    Quiescent,        // every subsystem terminal, no possible progress
    Stalled,          // pending syncs, no DMA-bytes/lock-release progress
    PollExhausted,    // BlockedOnPoll past budget; not a clean poll-unsatisfiable case
}

pub struct TdrDetector {
    context_id: ContextId,
    quiescence: QuiescenceDetector,
    stall: StallDetector,
    poll_stall_cycles: u64,
    poll_stall_limit: u64,        // 20_000, matches current POLL_STALL_LIMIT
}

impl TdrDetector {
    pub fn new(context_id: ContextId) -> Self;
    pub fn classify(
        &mut self,
        engine: &InterpreterEngine,
        executor: Option<&NpuExecutor>,
    ) -> TdrVerdict;
}
```

**Precedence inside `classify`:** `NaturalCompletion` > `MaskPollUnsatisfied` > `Wedged` > `Progressing`. Covered by a precedence-collision unit test.

**`WedgeReason` precedence (when more than one applies):** `Quiescent` > `Stalled` > `PollExhausted`. A truly-quiescent system is also trivially "stalled" — pick the strongest description. `Stalled` (pending syncs + no DMA/lock progress with cores still running) is reported only when not Quiescent. `PollExhausted` is reported only when the BlockedOnPoll path is the proximate cause AND the cleaner `MaskPollUnsatisfied` test did not already catch it (engine wasn't quiescent yet).

### 4.3 Device root

```rust
// src/device/mod.rs (or wherever DeviceState lives)
pub struct DeviceState {
    // ...existing fields...
    pub contexts: Vec<Context>,           // len==1 today, ContextId(0)
    pub tdr_detectors: Vec<TdrDetector>,  // parallel index by ContextId
}
```

### 4.4 FFI surface additions

```rust
// crates/xdna-emu-ffi/src/context.rs (new module)

#[repr(u32)]
pub enum XdnaEmuContextState {
    Connected = 0,
    Stopped = 1,
    Failed = 2,
}

pub unsafe extern "C" fn xdna_emu_get_context_state(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
    out_state: *mut XdnaEmuContextState,
    out_completed_counter: *mut u64,
) -> i32;  // 0=ok, -1=null, -2=invalid context

// XdnaEmuHaltReason gains one variant:
#[repr(u32)]
pub enum XdnaEmuHaltReason {
    Completed = 0,
    Budget = 1,
    Error = 2,
    MaskPollUnsatisfied = 3,
    WedgeRecovered = 4,    // NEW
}

// xdna_emu_reset_context gains a context_id parameter:
pub unsafe extern "C" fn xdna_emu_reset_context(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
) -> XdnaEmuResult;
```

### 4.5 Plugin surface (`xrt-plugin/src/transport_inprocess.{h,cpp}`)

- Resolve-required for `xdna_emu_get_context_state` (fail-loud on stale `.so`,
  same convention as Tier B's 5 FFI symbols).
- On `xdna_emu_run` returning `halt_reason = WedgeRecovered`: map to an
  EIO-shaped XRT command state (e.g., `ERT_CMD_STATE_ABORT`) so `run.wait()`
  consumers observe failure instead of hanging.
- `xdna_emu_reset_context` calls update to pass `context_id = 0` explicitly.
  This is a **breaking FFI signature change** (one parameter added); all
  existing C++ callers in `xrt-plugin/` update in the same commit. No external
  consumers outside the workspace.

### 4.6 What moves (refactor footprint)

- `src/testing/quiescence.rs` → `src/device/tdr/detector.rs`. Public API of
  `QuiescenceDetector` and `StallDetector` preserved. `QuiescenceDiagnosis`
  renames to `TdrDiagnosis` (in-tree find-replace; the in-process xclbin
  runner is the only other consumer).
- The inline `POLL_STALL_LIMIT`, `poll_stall_cycles`, and `EngineStatus::Stalled`
  discrimination in `xdna_emu_run` collapses into `TdrDetector::classify`.
- The Tier B `record_error` path in `effects.rs` does not change. Tier B and
  Tier C remain independent paths.

## 5. Data flow

### 5.1 Per-cycle run flow (inside `xdna_emu_run`, after warm-up)

```
loop:
  npu_executor.try_advance()        -> Error?  return XdnaEmuHaltReason::Error
  engine.step()
  fire_async_callbacks_for(handle)  // Tier B drain, unchanged
  detector.classify(engine, executor) -> verdict
  match verdict:
    Progressing                  -> continue
    NaturalCompletion            -> context.note_submission_complete()
                                    break with Completed
    MaskPollUnsatisfied          -> break with MaskPollUnsatisfied
    Wedged { reason, diagnosis } -> context.mark_failed(reason, diagnosis)
                                    break with WedgeRecovered
  if cycles >= max && !unbounded -> break with Budget
```

### 5.2 Submission lifecycle (sequential submissions, single context)

```
plugin -> xdna_emu_execute_npu_instructions(handle, ...)    [load]
plugin -> xdna_emu_run(handle)
   on natural exit:     context(0).note_submission_complete()
                        return halt_reason = Completed
   on wedge:            context(0) is already Failed
                        return halt_reason = WedgeRecovered
plugin sees halt_reason -> translate to XRT command state
   Completed         -> ERT_CMD_STATE_COMPLETED
   WedgeRecovered    -> ERT_CMD_STATE_ABORT (EIO-shaped)
plugin (before next submission, if previous was wedge):
   xdna_emu_reset_context(handle, 0)
        -> context(0).mark_connected()
        -> async_errors.clear() for that context
        -> engine.reset_for_new_context()
```

### 5.3 Context state transitions

```
        submission complete
       (counter++, stay)
              ┌──┐
              ▼  │
        ┌─ Connected ──── wedge ──→ Failed { reason, diagnosis }
        │                                      │
        │   reset_context ◄──────── reset ─────┘
        ▼
     Stopped       (reserved: future explicit pause/resume; not entered by
                    Tier C flows. Present in the enum for multi-context
                    forward-compat.)
```

### 5.4 Engine signal flow into the detector

`TdrDetector::classify` is read-only over `engine` and `executor`. Inspects:

- `engine.status()` — `Halted` / `Stalled` / `Running`
- `engine.device().array.any_dma_active()` and `any_data_in_flight()`
- `engine.device().array.total_dma_bytes_transferred()` and `total_lock_releases()`
- `npu_executor.is_done()`, `syncs_satisfied()`, `is_blocked_on_poll()`, `pending_syncs()`

No new engine outputs needed.

## 6. Error handling and edge cases

- **Classify on a Failed context:** the run loop checks `context(0).state == Connected`
  at entry; non-Connected returns a distinct `InvalidState` result. The plugin
  should have reset between submissions. A `debug_assert!` at the loop top
  catches violations in dev builds.
- **Reset on a Connected (non-Failed) context:** idempotent. Cheap no-op on
  already-clean state.
- **Warm-up wedges:** warm-up loop (the 100k-cycle pre-NPU phase) does not run
  the detector — cores running to first-blocking-point cannot deadlock by
  construction.
- **Executor `Error`:** still short-circuits before classify (existing path at
  `execution.rs:177`). `XdnaEmuHaltReason::Error` stays the surface. Context
  state untouched (parse/structural failure, not runtime wedge). Distinct from
  `WedgeRecovered`.
- **Budget exhaustion:** `XdnaEmuHaltReason::Budget` unchanged. Context state
  stays Connected. `completed_counter` does not advance. Caller decides
  whether to re-run or reset.
- **Stalled but classifier returns NaturalCompletion:** possible when
  `syncs_satisfied` flips true the same cycle the engine stalls.
  `NaturalCompletion` precedence wins. Documented and unit-tested.
- **Reset mid-run:** plugin contract is "don't." `debug_assert!` in
  `xdna_emu_reset_context`; release behavior proceeds (engine reset wipes
  in-flight state cleanly, but the caller almost certainly has a bug).
- **Tier B / Tier C overlapping:** independent. A workload can record Tier B
  errors and then wedge; both surfaces remain valid until the next reset
  clears them. Plugin reads both surfaces independently.

## 7. Testing

### 7.1 Unit tests (lifted modules)

`QuiescenceDetector` and `StallDetector` tests move with the modules. No
behavior change → no test changes beyond `use` paths. Existing 5 tests
(threshold counting, reset-on-progress, diagnosis display variations) stay.

### 7.2 Unit tests (new)

- **`Context`:** state transitions (Connected → Failed → Connected),
  `note_submission_complete` advances counter, reset clears state, reset
  idempotent on Connected.
- **`TdrDetector::classify`:** one test per `TdrVerdict` variant
  (Progressing, NaturalCompletion, MaskPollUnsatisfied, Wedged with each
  `WedgeReason`), plus the precedence-collision test.
- **`WedgeReason` derivation:** quiescent-deadlock fires `Quiescent`,
  livelock with pending syncs fires `Stalled`, poll-stall past budget fires
  `PollExhausted`.

### 7.3 Integration tests

- **End-to-end wedge → EIO:** build a device with a wedging workload, run
  `xdna_emu_run`, assert `halt_reason == WedgeRecovered` AND
  `xdna_emu_get_context_state` returns `Failed`.
- **Recovery:** after the above, call `xdna_emu_reset_context`, assert state
  is `Connected` and the Tier B async-error sink for that context is empty.
- **Tier B/Tier C orthogonality:** workload that records a Tier B error and
  then wedges. Assert both surfaces populated; both independently cleared by
  reset.

### 7.4 FFI completeness test

Extend the existing FFI completeness test to assert:
- `xdna_emu_get_context_state` symbol exported.
- `XdnaEmuHaltReason::WedgeRecovered` discriminant present (value 4).
- `XdnaEmuContextState` discriminants present.
- `xdna_emu_reset_context` signature includes `context_id`.

### 7.5 Bridge test

Deferred (same rationale as Tier B Spec 1). Real-NPU TDR is a driver-side
concern; we cannot isolate the wedge → EIO → reset cycle cleanly without
inventing a deadlock kernel that the real driver's TDR would also fire on.
First multi-context follow-up can add one.

## 8. Forward-compatibility notes (multi-context)

The design ships single-context behavior but uses multi-context shapes
throughout. The work to make the engine *actually* multi-context-aware is
storage-and-scheduling: `Vec<Context>` grows past length 1; the engine learns
to schedule across them; the FFI/plugin grows context-create/destroy ioctls.
None of that changes the abstractions landed here.

Specifically, post-Tier-C:

- `ContextId` already threads through every read/mutate path.
- `Vec<Context>` and `Vec<TdrDetector>` already index by ContextId.
- `ContextState` already includes `Stopped` (for pause/resume) and reserves
  space for `Disconnected` (for firmware-reload-required).
- `xdna_emu_reset_context` already takes a `context_id` parameter.
- The plugin's "context_id = 0" is the only hardcoded site that needs
  generalization.

When the real driver-against-emu setup lands, the plugin's temporary
"observe-then-EIO" code is removed entirely; the driver's own TDR consumes
`xdna_emu_get_context_state` (or its evolved successor) directly.

## 9. Open follow-ups (tracked, not in this spec)

- Multi-context engine scheduling + lifecycle ioctls.
- `Disconnected` context state and firmware-reload semantics.
- Real-clock TDR timeout cadence (vs current cycle-driven).
- Bridge test for wedge → EIO behavior (needs a deadlock-kernel fixture).
- Auto-reset-on-wedge behavioral knob (if a consumer asks for it).
- Driver TDR algorithm replication (two-tick stuck check, `tdr_dump_ctx`
  knob) — explicitly out of scope per the boundary decision; only relevant
  if we ever want emulator-internal TDR independent of a driver consumer.
