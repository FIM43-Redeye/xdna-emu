# Interrupt Tier C — TDR / Context-Restart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the device-side context model + `TdrDetector` that classifies engine run state, plus the FFI/plugin surface that lets `xdna_emu_run` return an EIO-shaped result on wedge today and exposes the per-context signals a real driver's TDR will eventually consume.

**Architecture:** Approach B (device-localized) per [spec §2](../specs/2026-05-19-interrupt-tier-c-tdr-design.md). Lift `QuiescenceDetector`/`StallDetector` out of `src/testing/quiescence.rs` into `src/device/tdr/`; add `TdrDetector` that composes them into a single per-cycle classifier; introduce a minimal `Context` type with `Vec<Context>` storage (len==1 today, multi-context-shaped APIs throughout); refactor `xdna_emu_run` to consume the classifier; add a new `WedgeRecovered` halt reason and a per-context state FFI accessor; upgrade the plugin's `xdna_emu_reset_context` resolution from optional to required while also adding the `context_id` parameter.

**Tech Stack:** Rust (workspace crates `xdna-emu-core`, `xdna-emu-ffi`), C++ XRT plugin (`xrt-plugin/`), driver cross-references against `~/npu-work/xdna-driver/src/driver/amdxdna/` (`aie2_tdr.c`, `aie2_ctx_runqueue.c`).

**Driver cross-check discipline (per Maya):** Tasks that touch a driver-mirrored surface (Context state vocabulary, completed_counter advance semantics, reset chain shape, wedge-detection signals) start with a **Step 0** that reads the relevant driver source and pins the design decision against it. Tasks 1, 9, and 12 have this baked in. Other tasks can call out drift if they notice it during implementation.

---

## File-touch overview

**Create:**
- `src/device/tdr/mod.rs` — module root + `TdrVerdict`, `WedgeReason`, `EngineSignals`, `ExecutorSignals` snapshot structs, `TdrDetector` struct
- `src/device/tdr/detector.rs` — lifted `QuiescenceDetector`, `StallDetector`, `TdrDiagnosis`
- `src/device/context/mod.rs` — `ContextId`, `ContextState`, `Context`
- `crates/xdna-emu-ffi/src/context.rs` — `XdnaEmuContextState`, `xdna_emu_get_context_state`

**Modify:**
- `src/device/mod.rs` — add `pub mod tdr; pub mod context;`
- `src/device/state/mod.rs` — `contexts: Vec<Context>` + `tdr_detectors: Vec<TdrDetector>` fields on `DeviceState`, constructor population, reset method
- `src/testing/xclbin_suite.rs` — update import path from `super::quiescence` to `crate::device::tdr`
- `crates/xdna-emu-ffi/src/lib.rs` — add `WedgeRecovered` variant to `XdnaEmuHaltReason`, change `xdna_emu_reset_context` signature to take `context_id: u32`, `pub mod context;`
- `crates/xdna-emu-ffi/src/execution.rs` — replace inline stall/poll logic with `TdrDetector::classify` calls, transition context state on terminal verdicts, enforce entry-guard (context must be `Connected`)
- `xrt-plugin/src/transport.h` — add `get_context_state` virtual method on `emu_transport`; change `reset_context()` to `reset_context(uint32_t context_id)`
- `xrt-plugin/src/transport_inprocess.h` — `fn_get_context_state` typedef, override declarations, symbol member
- `xrt-plugin/src/transport_inprocess.cpp` — upgrade `sym_reset_context_` resolution to `resolve_required`, resolve `xdna_emu_get_context_state` as required, change reset_context impl to pass `context_id`, add halt-reason mapping for `WedgeRecovered` at the `xdna_emu_run` consumer site
- `crates/xdna-archspec/src/coverage/units.rs` — refresh `interrupt` narrative for Tier C shipping
- `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md` — note Tier C status (this finding currently mentions Tier C as a follow-up)

**Delete:**
- `src/testing/quiescence.rs` (contents lifted to `src/device/tdr/detector.rs` in Task 2)

---

## Task 1: Driver cross-check + create `src/device/tdr/` module skeleton

**Files:**
- Create: `src/device/tdr/mod.rs`
- Modify: `src/device/mod.rs`

- [ ] **Step 0: Driver cross-check** — read `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c` end-to-end (~103 lines) and `aie2_ctx_runqueue.c:735-790` (the `aie2_rq_is_all_context_stuck` block). Note for the implementer:
  - Driver's "stuck" signal: pending cmds exist AND completed counter unchanged across two ticks
  - Recovery chain: `aie2_rq_dump_all` → `aie2_rq_stop_all` → `aie2_rq_restart_all`
  - These map (loosely) to our: `TdrDiagnosis` (dump-equivalent), `xdna_emu_reset_context` (combined stop+restart for single context)
  - We are **not** implementing the two-tick check — spec §1.2 makes this out of scope

- [ ] **Step 1: Add `pub mod tdr;` to `src/device/mod.rs`**

Find the existing `pub mod async_errors;` line in `src/device/mod.rs` and add immediately below:

```rust
pub mod tdr;
```

- [ ] **Step 2: Create empty module root `src/device/tdr/mod.rs`**

```rust
//! Tier C TDR (Timeout Detection & Recovery) / context-restart support.
//!
//! Exposes the per-cycle classifier that decides whether the in-flight
//! submission is progressing, completing naturally, exhausting a satisfiable
//! poll, or wedged. The actual TDR algorithm (periodic timer, two-tick stuck
//! check, recovery chain) is a driver-side concern; this module exposes the
//! signals a driver TDRs on. See:
//! - `docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md`
//! - `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c` (the driver-side
//!   algorithm this lets a driver consumer drive)

pub mod detector;
```

- [ ] **Step 3: Create empty `src/device/tdr/detector.rs`**

```rust
//! Lifted from `src/testing/quiescence.rs` — detectors are device-state
//! classifiers, not test infrastructure. Populated in Task 2.
```

- [ ] **Step 4: Build to verify module wiring**

Run: `cargo build -p xdna-emu-core`
Expected: clean build (empty modules compile).

- [ ] **Step 5: Commit**

```bash
git add src/device/mod.rs src/device/tdr/mod.rs src/device/tdr/detector.rs
git commit -m "$(cat <<'EOF'
interrupt: scaffold Tier C tdr module

Empty module skeleton at src/device/tdr/. Task 2 lifts the detectors
from src/testing/quiescence.rs into detector.rs; Task 4 adds the
TdrDetector classifier in mod.rs. Driver cross-check landed (see plan
Task 1 Step 0): no two-tick check, no recovery-chain replication --
both are driver-side concerns out of scope per spec section 1.2.

Generated using Claude Code.
EOF
)"
```

---

## Task 2: Lift `QuiescenceDetector` + `StallDetector` + `QuiescenceDiagnosis` from testing to device/tdr

This is a pure move/rename — no behavior change, no API change beyond the rename. Tests come with the move.

**Files:**
- Modify: `src/device/tdr/detector.rs` (populate from existing `src/testing/quiescence.rs`)
- Modify: `src/testing/mod.rs` (remove `pub mod quiescence;`)
- Modify: `src/testing/xclbin_suite.rs:1170,1287` (update import path)
- Delete: `src/testing/quiescence.rs`

- [ ] **Step 1: Copy `src/testing/quiescence.rs` content into `src/device/tdr/detector.rs`**

Replace the placeholder content of `src/device/tdr/detector.rs` with the full contents of `src/testing/quiescence.rs`. Rename `QuiescenceDiagnosis` → `TdrDiagnosis` everywhere in the new file (struct definition, Display impl, all references inside this file's tests, all return types). The detector struct names (`QuiescenceDetector`, `StallDetector`) and their method signatures stay the same.

After the rename, the file's public surface is:
- `pub enum QuiescenceStatus { Running, Quiescent(TdrDiagnosis) }`
- `pub struct TdrDiagnosis { ... }` (was `QuiescenceDiagnosis`)
- `pub struct QuiescenceDetector { ... }` (unchanged)
- `pub enum StallStatus { Progressing, Stalled(TdrDiagnosis) }` (was `Stalled(QuiescenceDiagnosis)`)
- `pub struct StallDetector { ... }` (unchanged)

Update the module-level doc comment from "Quiescence-based timeout detection" to:
```rust
//! Device-state classifiers used by Tier C TDR.
//!
//! `QuiescenceDetector` catches deadlocks (every subsystem terminal,
//! no possible forward progress). `StallDetector` catches livelocks
//! (cores running but no DMA-bytes/lock-release progress with pending
//! syncs). Composed by [`super::TdrDetector`] which classifies engine
//! run state into a single per-cycle verdict.
//!
//! Lifted from `src/testing/quiescence.rs` (where they were misfiled
//! as test infrastructure) on 2026-05-19 as part of Tier C.
```

- [ ] **Step 2: Re-export from `src/device/tdr/mod.rs`**

Add to `src/device/tdr/mod.rs`:

```rust
pub use detector::{
    QuiescenceDetector, QuiescenceStatus, StallDetector, StallStatus, TdrDiagnosis,
};
```

- [ ] **Step 3: Update the single in-tree consumer**

In `src/testing/xclbin_suite.rs` find line ~1170:
```rust
use super::quiescence::{QuiescenceDetector, QuiescenceStatus, StallDetector, StallStatus};
```
Change to:
```rust
use crate::device::tdr::{QuiescenceDetector, QuiescenceStatus, StallDetector, StallStatus};
```

Also at line ~1287:
```rust
let diagnosis = QuiescenceDetector::diagnose(engine, npu_executor.as_deref());
```
This call stays as-is; only the import path changed. The return type changes from `QuiescenceDiagnosis` to `TdrDiagnosis` but variable typing is inferred. If the variable is annotated elsewhere in the file (grep first), update it.

- [ ] **Step 4: Drop the now-empty testing module**

In `src/testing/mod.rs`, remove the line:
```rust
mod quiescence;   // or `pub mod quiescence;` -- delete whichever form exists
```

Then delete the file:
```bash
git rm src/testing/quiescence.rs
```

- [ ] **Step 5: Run the lifted unit tests at their new path**

Run: `cargo test --lib -p xdna-emu-core device::tdr::detector`
Expected: 5 tests pass (test_quiescence_threshold_counting, test_quiescence_reset_on_progress, test_diagnosis_display_all_halted, test_diagnosis_display_waiting_core, plus any other tests that were in the file).

If the test count differs from what the old `src/testing/quiescence.rs` had, you've lost a test — re-check the move.

- [ ] **Step 6: Run full library tests to catch any missed import**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu-core`
Expected: all pass. If a build error or test failure appears, it's because some consumer of `QuiescenceDiagnosis` (renamed) or `crate::testing::quiescence` (deleted) wasn't updated. Fix and re-run.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
interrupt: lift Quiescence/Stall detectors into device::tdr

Pure move + rename. QuiescenceDiagnosis -> TdrDiagnosis to match
the module's purpose; detector struct names unchanged. One consumer
in src/testing/xclbin_suite.rs updates its import path.

The detectors were misfiled under src/testing/ -- they are device-state
classifiers, used by both the in-process xclbin runner and (now) the
Tier C TdrDetector classifier in the FFI run loop. Lifting them here
makes the abstraction layer match the actual dependency direction.

Generated using Claude Code.
EOF
)"
```

---

## Task 3: Add `TdrVerdict`, `WedgeReason`, `EngineSignals`/`ExecutorSignals` snapshots, `TdrDetector` skeleton

The classifier takes primitive snapshot structs rather than `&InterpreterEngine` directly, so unit tests can construct inputs without standing up an engine. The run loop in Task 12 builds the snapshot from the engine once per cycle and passes it in.

**Files:**
- Modify: `src/device/tdr/mod.rs`
- Test: `src/device/tdr/mod.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write failing test for `WedgeReason` and `TdrVerdict` enums + derives**

Append to `src/device/tdr/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wedge_reason_derives_copy_clone_debug() {
        let r = WedgeReason::Quiescent;
        let r2 = r;            // Copy
        let _ = r;             // still usable after copy
        let _ = format!("{:?}", r2); // Debug
    }

    #[test]
    fn tdr_verdict_progressing_is_default_construction() {
        let v = TdrVerdict::Progressing;
        let _ = format!("{:?}", v);
        assert!(matches!(v, TdrVerdict::Progressing));
    }

    #[test]
    fn tdr_verdict_wedged_carries_reason_and_diagnosis() {
        let diag = TdrDiagnosis {
            core_states: vec![],
            dma_states: vec![],
            data_in_flight: false,
            pending_syncs: vec![],
        };
        let v = TdrVerdict::Wedged { reason: WedgeReason::Quiescent, diagnosis: diag };
        match v {
            TdrVerdict::Wedged { reason, .. } => assert!(matches!(reason, WedgeReason::Quiescent)),
            _ => panic!("expected Wedged"),
        }
    }
}
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests`
Expected: compile errors — `WedgeReason`, `TdrVerdict` don't exist yet.

- [ ] **Step 3: Define the enums + snapshot structs + skeleton in `src/device/tdr/mod.rs`**

Add (above the `#[cfg(test)]` block):

```rust
use crate::device::context::ContextId;
use crate::interpreter::core::CoreStatus;

/// Reason a context's submission is classified as wedged.
///
/// Precedence (when more than one would apply): `Quiescent` > `Stalled` >
/// `PollExhausted`. A truly-quiescent system is also trivially "stalled" --
/// pick the strongest description. See spec section 4.2.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum WedgeReason {
    /// Every subsystem terminal, no possible forward progress.
    Quiescent,
    /// Pending syncs, cores still running, but no DMA-bytes/lock-release progress.
    Stalled,
    /// Executor parked in `BlockedOnPoll` past the cycle budget,
    /// and the cleaner `MaskPollUnsatisfied` test did not catch it
    /// (engine wasn't quiescent yet).
    PollExhausted,
}

/// Per-cycle verdict from [`TdrDetector::classify`].
///
/// Precedence inside the classifier (when more than one would apply):
/// `NaturalCompletion` > `MaskPollUnsatisfied` > `Wedged` > `Progressing`.
#[derive(Debug)]
pub enum TdrVerdict {
    /// Forward progress this cycle. Run loop continues.
    Progressing,
    /// Engine halted with all syncs satisfied. The normal happy path.
    NaturalCompletion,
    /// Existing semantic, now classified here. Run loop breaks with
    /// `XdnaEmuHaltReason::MaskPollUnsatisfied`.
    MaskPollUnsatisfied,
    /// Submission is wedged. Caller transitions context state to
    /// `Failed { reason, diagnosis }` and breaks the run loop with
    /// `XdnaEmuHaltReason::WedgeRecovered`.
    Wedged { reason: WedgeReason, diagnosis: TdrDiagnosis },
}

/// Per-cycle snapshot of engine signals the classifier reads.
///
/// Built by the run loop from `&InterpreterEngine` once per cycle.
/// Tests construct directly without standing up an engine.
#[derive(Debug, Clone)]
pub struct EngineSignals {
    pub engine_status: EngineStatusSnapshot,
    pub any_dma_active: bool,
    pub any_data_in_flight: bool,
    pub total_dma_bytes_transferred: u64,
    pub total_lock_releases: u64,
    /// (col, row, status) for every enabled compute core.
    pub core_statuses: Vec<(u8, u8, CoreStatus)>,
    /// (col, row, channel, fsm_description) for every non-idle DMA channel.
    pub dma_states: Vec<(u8, u8, u8, String)>,
}

/// Mirror of the live `EngineStatus` enum, snapshotted for the classifier.
///
/// Kept narrow on purpose -- the classifier only needs the variants that
/// affect its decision tree, so this is decoupled from the engine's full
/// `EngineStatus` enum (which has additional intermediate variants).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EngineStatusSnapshot {
    Running,
    Halted,
    Stalled,
    Other,
}

/// Per-cycle snapshot of NPU executor signals the classifier reads.
#[derive(Debug, Clone)]
pub struct ExecutorSignals {
    pub is_done: bool,
    pub syncs_satisfied: bool,
    pub is_blocked_on_poll: bool,
    /// (col, row, channel, direction) for each pending sync.
    pub pending_syncs: Vec<(u8, u8, u8, u8)>,
}

/// Per-context classifier composing the lifted Quiescence/Stall detectors
/// with a poll-stall budget.
pub struct TdrDetector {
    context_id: ContextId,
    quiescence: QuiescenceDetector,
    stall: StallDetector,
    poll_stall_cycles: u64,
    poll_stall_limit: u64,
}

/// Cycle budget after which a sustained `BlockedOnPoll` is reported as
/// `PollExhausted`. Mirrors the current inline `POLL_STALL_LIMIT` in
/// `xdna_emu_run` (20_000 cycles). Generous enough that any legitimately
/// satisfiable poll resolves first, well below the bridge wall-clock.
pub const DEFAULT_POLL_STALL_LIMIT: u64 = 20_000;

/// Cycle threshold for [`QuiescenceDetector`] when used inside [`TdrDetector`].
/// Same value the in-process xclbin runner uses today.
pub const DEFAULT_QUIESCENCE_CYCLES: u64 = 5;

/// Cycle threshold for [`StallDetector`] when used inside [`TdrDetector`].
pub const DEFAULT_STALL_CYCLES: u64 = 100_000;

impl TdrDetector {
    /// Construct a detector for the given context using the default thresholds.
    pub fn new(context_id: ContextId) -> Self {
        Self {
            context_id,
            quiescence: QuiescenceDetector::new(DEFAULT_QUIESCENCE_CYCLES),
            stall: StallDetector::new(DEFAULT_STALL_CYCLES),
            poll_stall_cycles: 0,
            poll_stall_limit: DEFAULT_POLL_STALL_LIMIT,
        }
    }

    /// The context this detector classifies.
    pub fn context_id(&self) -> ContextId {
        self.context_id
    }

    // classify(...) added in Tasks 5-8 (TDD per verdict variant).
}
```

(`ContextId` is referenced via `use` even though Task 9 is the one that creates it. Reorder the task or stub the type ahead — the implementer can decide; see Step 4.)

- [ ] **Step 4: Stub `ContextId` to keep this task self-contained**

Until Task 9 lands the real definition, add at the top of `src/device/tdr/mod.rs`, BEFORE the `use crate::device::context::ContextId;` line, this temporary inline definition (delete in Task 9):

Actually no — simpler: defer this `use` line and the `context_id` field until Task 9 reorders. **In this task, use `u32` for `context_id` instead of `ContextId`.** Task 9 then changes the field type when the real `ContextId` lands. Apply this substitution in Step 3:

- Field: `context_id: u32`
- Constructor: `pub fn new(context_id: u32) -> Self`
- Accessor: `pub fn context_id(&self) -> u32`
- Remove `use crate::device::context::ContextId;` line

- [ ] **Step 5: Run tests to verify they now pass**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests`
Expected: 3 tests pass (the two enum-derive tests + the Wedged construction test).

- [ ] **Step 6: Commit**

```bash
git add src/device/tdr/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: add Tier C verdict/reason enums + TdrDetector skeleton

Defines TdrVerdict (Progressing/NaturalCompletion/MaskPollUnsatisfied/
Wedged) and WedgeReason (Quiescent/Stalled/PollExhausted) with their
precedence rules. EngineSignals/ExecutorSignals snapshot structs let
the classifier (added incrementally in Tasks 5-8) be unit-tested
without standing up a full engine. TdrDetector composes the lifted
Quiescence/Stall detectors with a poll-stall budget.

context_id is u32 in this task; tightens to ContextId in Task 9 when
the Context module lands.

Generated using Claude Code.
EOF
)"
```

---

## Task 4: TDD `classify()` — `Progressing` + `NaturalCompletion` variants

**Files:**
- Modify: `src/device/tdr/mod.rs` (add `classify` method + tests)

- [ ] **Step 1: Write two failing tests for the trivial verdicts**

Append to the `#[cfg(test)] mod tests` in `src/device/tdr/mod.rs`:

```rust
    fn empty_engine_signals(status: EngineStatusSnapshot) -> EngineSignals {
        EngineSignals {
            engine_status: status,
            any_dma_active: false,
            any_data_in_flight: false,
            total_dma_bytes_transferred: 0,
            total_lock_releases: 0,
            core_statuses: vec![],
            dma_states: vec![],
        }
    }

    fn natural_completion_executor() -> ExecutorSignals {
        ExecutorSignals {
            is_done: true,
            syncs_satisfied: true,
            is_blocked_on_poll: false,
            pending_syncs: vec![],
        }
    }

    fn no_executor() -> Option<ExecutorSignals> { None }

    #[test]
    fn classify_returns_progressing_when_engine_running_and_no_executor() {
        let mut detector = TdrDetector::new(0);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let verdict = detector.classify(&signals, no_executor().as_ref());
        assert!(matches!(verdict, TdrVerdict::Progressing), "got {verdict:?}");
    }

    #[test]
    fn classify_returns_natural_completion_when_engine_halted_and_syncs_satisfied() {
        let mut detector = TdrDetector::new(0);
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = natural_completion_executor();
        let verdict = detector.classify(&signals, Some(&exec));
        assert!(matches!(verdict, TdrVerdict::NaturalCompletion), "got {verdict:?}");
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: compile error — `classify` not defined.

- [ ] **Step 3: Implement `classify` (minimum to pass these two)**

Add to `impl TdrDetector` in `src/device/tdr/mod.rs`:

```rust
    /// Classify the engine's run state this cycle.
    ///
    /// Precedence (when more than one would apply): `NaturalCompletion` >
    /// `MaskPollUnsatisfied` > `Wedged` > `Progressing`. The classifier
    /// returns the strongest applicable verdict.
    ///
    /// Read-only over signals; mutates internal counters
    /// (`poll_stall_cycles`, quiescence/stall thresholds) per cycle.
    pub fn classify(
        &mut self,
        signals: &EngineSignals,
        executor: Option<&ExecutorSignals>,
    ) -> TdrVerdict {
        // Highest precedence: natural completion.
        if signals.engine_status == EngineStatusSnapshot::Halted {
            if let Some(exec) = executor {
                if exec.is_done && exec.syncs_satisfied {
                    return TdrVerdict::NaturalCompletion;
                }
            } else if !signals.any_dma_active && !signals.any_data_in_flight {
                return TdrVerdict::NaturalCompletion;
            }
        }

        // Default: still making progress (or the lower-precedence checks
        // added in Tasks 5-7 will refine).
        TdrVerdict::Progressing
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: both tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/device/tdr/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: classify() handles Progressing + NaturalCompletion

First two verdicts. Halted engine + executor reports done/syncs-satisfied
yields NaturalCompletion; otherwise Progressing. Lower-precedence
checks (MaskPollUnsatisfied, Wedged variants) added in Tasks 5-7.

Generated using Claude Code.
EOF
)"
```

---

## Task 5: TDD `classify()` — `MaskPollUnsatisfied`

**Files:**
- Modify: `src/device/tdr/mod.rs`

- [ ] **Step 1: Write failing tests**

Append to `mod tests`:

```rust
    fn blocked_on_poll_executor() -> ExecutorSignals {
        ExecutorSignals {
            is_done: false,
            syncs_satisfied: false,
            is_blocked_on_poll: true,
            pending_syncs: vec![],
        }
    }

    #[test]
    fn classify_returns_mask_poll_unsatisfied_when_engine_halted_and_executor_blocked_on_poll() {
        let mut detector = TdrDetector::new(0);
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        signals.any_dma_active = false;     // matches existing run-loop fast-path condition
        let exec = blocked_on_poll_executor();
        let verdict = detector.classify(&signals, Some(&exec));
        assert!(matches!(verdict, TdrVerdict::MaskPollUnsatisfied), "got {verdict:?}");
    }

    #[test]
    fn classify_returns_mask_poll_unsatisfied_after_poll_stall_budget() {
        let mut detector = TdrDetector::new(0);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let exec = blocked_on_poll_executor();
        // Burn through the poll-stall budget. Detector accumulates internally.
        let mut last = TdrVerdict::Progressing;
        for _ in 0..DEFAULT_POLL_STALL_LIMIT {
            last = detector.classify(&signals, Some(&exec));
        }
        // On the budget-th cycle (or one after), should report MaskPollUnsatisfied.
        // Run one more cycle to be safe.
        last = detector.classify(&signals, Some(&exec));
        assert!(matches!(last, TdrVerdict::MaskPollUnsatisfied), "got {last:?}");
    }

    #[test]
    fn classify_resets_poll_stall_when_executor_unblocks() {
        let mut detector = TdrDetector::new(0);
        let signals = empty_engine_signals(EngineStatusSnapshot::Running);
        let exec_blocked = blocked_on_poll_executor();
        let mut exec_unblocked = blocked_on_poll_executor();
        exec_unblocked.is_blocked_on_poll = false;

        // Accumulate near the limit while blocked.
        for _ in 0..(DEFAULT_POLL_STALL_LIMIT - 10) {
            detector.classify(&signals, Some(&exec_blocked));
        }
        // Unblock for a cycle.
        detector.classify(&signals, Some(&exec_unblocked));
        // Now block again; should NOT fire immediately (counter reset).
        for _ in 0..10 {
            let v = detector.classify(&signals, Some(&exec_blocked));
            assert!(!matches!(v, TdrVerdict::MaskPollUnsatisfied), "fired too early after reset");
        }
    }
```

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: the 3 new tests fail (classify returns Progressing instead).

- [ ] **Step 3: Extend `classify` with the MaskPollUnsatisfied path**

In `src/device/tdr/mod.rs`, update the `classify` method to insert the MaskPollUnsatisfied check after the NaturalCompletion check and before the final `Progressing`:

```rust
        // Highest precedence: natural completion.
        // ... (existing block)

        // Second precedence: MaskPollUnsatisfied.
        // Mirrors the two paths the inline xdna_emu_run logic uses today:
        //  (a) engine Halted + executor BlockedOnPoll + no DMA in flight
        //      (the "fast-path" -- nothing left running can satisfy the poll)
        //  (b) executor BlockedOnPoll for poll_stall_limit consecutive cycles
        //      (the "budget" -- caps unsatisfiable polls in the running case)
        if let Some(exec) = executor {
            if exec.is_blocked_on_poll {
                // Fast-path (a):
                if signals.engine_status == EngineStatusSnapshot::Halted
                    && !signals.any_dma_active
                {
                    return TdrVerdict::MaskPollUnsatisfied;
                }
                // Budget (b):
                self.poll_stall_cycles += 1;
                if self.poll_stall_cycles >= self.poll_stall_limit {
                    return TdrVerdict::MaskPollUnsatisfied;
                }
            } else {
                self.poll_stall_cycles = 0;
            }
        } else {
            self.poll_stall_cycles = 0;
        }

        TdrVerdict::Progressing
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: all classify tests (Task 4's 2 + these 3) pass.

- [ ] **Step 5: Commit**

```bash
git add src/device/tdr/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: classify() handles MaskPollUnsatisfied

Two paths mirror the inline xdna_emu_run logic today:
  (a) engine Halted + executor BlockedOnPoll + no DMA  -- fast-path
  (b) executor BlockedOnPoll for poll_stall_limit cycles -- budget

Counter resets when the executor unblocks (covered by test).

Generated using Claude Code.
EOF
)"
```

---

## Task 6: TDD `classify()` — Wedged variants (`Quiescent`, `Stalled`, `PollExhausted`)

**Files:**
- Modify: `src/device/tdr/mod.rs`

- [ ] **Step 1: Write failing tests for each WedgeReason**

Append to `mod tests`:

```rust
    #[test]
    fn classify_returns_wedged_quiescent_when_all_subsystems_terminal() {
        let mut detector = TdrDetector::new(0);
        // All terminal: engine NOT Halted (otherwise we'd hit NaturalCompletion path),
        // but no DMA, no data in flight, all cores in terminal state.
        // Use the "no executor" path so NaturalCompletion's executor check
        // doesn't satisfy. Engine status Stalled here.
        let signals = empty_engine_signals(EngineStatusSnapshot::Stalled);
        // Drive the quiescence detector long enough for it to fire.
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, no_executor().as_ref());
            // Final cycle should be Wedged{Quiescent}; intermediate cycles
            // are Progressing.
            if let TdrVerdict::Wedged { reason, .. } = v {
                assert_eq!(reason, WedgeReason::Quiescent);
                return;
            }
        }
        panic!("Quiescent verdict never fired across {} cycles", DEFAULT_QUIESCENCE_CYCLES * 2);
    }

    #[test]
    fn classify_returns_wedged_stalled_with_pending_syncs_and_no_byte_progress() {
        let mut detector = TdrDetector::new(0);
        // Cores still "running" (engine not Stalled), but no DMA byte progress
        // and pending syncs. Stall detector fires after its threshold.
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Running);
        signals.any_dma_active = true;  // suppress Quiescent path
        let exec = ExecutorSignals {
            is_done: false,
            syncs_satisfied: false,
            is_blocked_on_poll: false,
            pending_syncs: vec![(0, 0, 0, 0)],
        };
        // StallDetector fires after DEFAULT_STALL_CYCLES of no byte/lock progress.
        let mut fired = false;
        for _ in 0..(DEFAULT_STALL_CYCLES + 100) {
            let v = detector.classify(&signals, Some(&exec));
            if let TdrVerdict::Wedged { reason, .. } = v {
                assert_eq!(reason, WedgeReason::Stalled);
                fired = true;
                break;
            }
        }
        assert!(fired, "Stalled verdict never fired");
    }

    #[test]
    fn classify_returns_wedged_poll_exhausted_when_budget_burned_without_clean_fastpath() {
        // PollExhausted differs from MaskPollUnsatisfied: it fires only when
        // the budget is burned AND the cleaner fast-path conditions are not
        // met (e.g. DMA still active, masking the "engine quiescent" tell).
        let mut detector = TdrDetector::new(0);
        let mut signals = empty_engine_signals(EngineStatusSnapshot::Running);
        signals.any_dma_active = true;  // disqualifies the MaskPollUnsatisfied fast-path
        let exec = blocked_on_poll_executor();
        let mut last = TdrVerdict::Progressing;
        for _ in 0..(DEFAULT_POLL_STALL_LIMIT + 1) {
            last = detector.classify(&signals, Some(&exec));
        }
        // PRECEDENCE NOTE: when the poll-stall budget is burned, the
        // classifier returns MaskPollUnsatisfied (b-path from Task 5),
        // NOT PollExhausted. PollExhausted is only the reason inside a
        // Wedged verdict when neither MaskPollUnsatisfied path applies;
        // since path (b) always applies once the budget burns, this test
        // expects MaskPollUnsatisfied. Kept here to lock in the precedence.
        assert!(matches!(last, TdrVerdict::MaskPollUnsatisfied), "got {last:?}");
    }
```

(The third test inverts what its name suggests — see the inline comment. The Wedged{PollExhausted} variant exists in the enum for completeness and for the case when an alternative classifier wiring would treat poll-stall as a wedge rather than a distinct halt reason. With the current precedence it does not fire; the variant + test stay to document the design.)

- [ ] **Step 2: Run tests to verify failure**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: the new Quiescent + Stalled tests fail (classify returns Progressing); the PollExhausted test may pass already (it expects MaskPollUnsatisfied, which is already implemented).

- [ ] **Step 3: Extend `classify` with Wedged{Quiescent} and Wedged{Stalled} paths**

Update `classify` in `src/device/tdr/mod.rs` to insert the wedge checks AFTER MaskPollUnsatisfied and BEFORE the final `Progressing`. The detectors are consumed via small wrapper adapters (since QuiescenceDetector::check expects `&InterpreterEngine`, but the classifier has snapshots).

Add two private helpers below `classify`:

```rust
    /// Run the quiescence rule against snapshot inputs. Returns the count's
    /// state -- true when the threshold is met (i.e. quiescent for
    /// `DEFAULT_QUIESCENCE_CYCLES` consecutive cycles).
    fn check_quiescence(&mut self, signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> bool {
        // Same predicate the existing QuiescenceDetector enforces, snapshot-driven:
        //   - executor (if present) is done
        //   - engine not Halted
        //   - no runnable core
        //   - no DMA active
        //   - no data in flight
        let executor_done = executor.map_or(true, |e| e.is_done);
        let engine_terminal = signals.engine_status != EngineStatusSnapshot::Halted;
        let no_runnable_core = signals.core_statuses.iter().all(|(_, _, s)| {
            !matches!(s, CoreStatus::Running | CoreStatus::Ready)
        });
        let cond = executor_done && engine_terminal && no_runnable_core
            && !signals.any_dma_active && !signals.any_data_in_flight;

        if cond {
            self.quiescence.bump_quiescent_cycle();
            self.quiescence.threshold_met()
        } else {
            self.quiescence.reset_quiescent_cycles();
            false
        }
    }

    fn check_stall(&mut self, signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> bool {
        let has_pending_syncs = executor.map_or(false, |e| !e.pending_syncs.is_empty());
        if !has_pending_syncs {
            self.stall.reset();
            return false;
        }
        self.stall.note_progress(signals.total_dma_bytes_transferred, signals.total_lock_releases)
    }

    fn build_diagnosis(signals: &EngineSignals, executor: Option<&ExecutorSignals>) -> TdrDiagnosis {
        TdrDiagnosis {
            core_states: signals.core_statuses.iter()
                .map(|(c, r, s)| (*c, *r, format!("{:?}", s))).collect(),
            dma_states: signals.dma_states.clone(),
            data_in_flight: signals.any_data_in_flight,
            pending_syncs: executor.map(|e| {
                e.pending_syncs.iter()
                    .map(|(c, r, ch, dir)| {
                        let dir_s = if *dir == 0 { "S2MM" } else { "MM2S" };
                        format!("col={c} row={r} ch={ch} {dir_s}")
                    }).collect()
            }).unwrap_or_default(),
        }
    }
```

The `QuiescenceDetector` and `StallDetector` lifted in Task 2 need three small accessor methods added (their current `check` method takes `&InterpreterEngine`; the snapshot adapter pattern needs the internal-state operations split out). Add to `src/device/tdr/detector.rs`:

```rust
impl QuiescenceDetector {
    pub(crate) fn bump_quiescent_cycle(&mut self) {
        self.quiescent_cycles += 1;
    }
    pub(crate) fn reset_quiescent_cycles(&mut self) {
        self.quiescent_cycles = 0;
    }
    pub(crate) fn threshold_met(&self) -> bool {
        self.quiescent_cycles >= self.threshold
    }
}

impl StallDetector {
    pub(crate) fn reset(&mut self) {
        self.cycles_since_progress = 0;
    }
    /// Returns true when threshold met (stalled).
    pub(crate) fn note_progress(&mut self, dma_bytes: u64, lock_releases: u64) -> bool {
        if dma_bytes != self.last_dma_bytes || lock_releases != self.last_lock_releases {
            self.last_dma_bytes = dma_bytes;
            self.last_lock_releases = lock_releases;
            self.cycles_since_progress = 0;
            false
        } else {
            self.cycles_since_progress += 1;
            self.cycles_since_progress >= self.threshold
        }
    }
}
```

Now extend `classify` (in `mod.rs`) to call these and emit Wedged verdicts:

```rust
        // Third precedence: Wedged{Quiescent}.
        if self.check_quiescence(signals, executor) {
            return TdrVerdict::Wedged {
                reason: WedgeReason::Quiescent,
                diagnosis: Self::build_diagnosis(signals, executor),
            };
        }
        // Fourth precedence: Wedged{Stalled}.
        if self.check_stall(signals, executor) {
            return TdrVerdict::Wedged {
                reason: WedgeReason::Stalled,
                diagnosis: Self::build_diagnosis(signals, executor),
            };
        }

        TdrVerdict::Progressing
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: all classify tests pass (Task 4's 2 + Task 5's 3 + Task 6's 3 = 8 total).

- [ ] **Step 5: Commit**

```bash
git add src/device/tdr/mod.rs src/device/tdr/detector.rs
git commit -m "$(cat <<'EOF'
interrupt: classify() handles Wedged variants

Wedged{Quiescent} fires when every subsystem is terminal for
DEFAULT_QUIESCENCE_CYCLES consecutive cycles. Wedged{Stalled} fires
when pending syncs exist but no DMA-byte / lock-release progress for
DEFAULT_STALL_CYCLES consecutive cycles. PollExhausted variant exists
in the enum for completeness (documented in the test inversion); with
current precedence, poll-stall budget burns always report through
MaskPollUnsatisfied.

QuiescenceDetector and StallDetector gain pub(crate) accessor methods
that let the classifier drive them from snapshot inputs rather than
&InterpreterEngine -- preserves their existing public API for the
xclbin_suite consumer.

Generated using Claude Code.
EOF
)"
```

---

## Task 7: TDD `classify()` — precedence-collision test

**Files:**
- Modify: `src/device/tdr/mod.rs`

- [ ] **Step 1: Write failing precedence tests**

Append to `mod tests`:

```rust
    #[test]
    fn classify_precedence_natural_completion_wins_over_wedge_signals() {
        let mut detector = TdrDetector::new(0);
        // Build inputs that would qualify as both NaturalCompletion AND
        // (after enough cycles) Wedged{Quiescent}. NaturalCompletion is
        // the higher-precedence verdict.
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = natural_completion_executor();
        // First cycle and every subsequent cycle should be NaturalCompletion --
        // we never accumulate into Wedged territory.
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, Some(&exec));
            assert!(matches!(v, TdrVerdict::NaturalCompletion), "got {v:?}");
        }
    }

    #[test]
    fn classify_precedence_mask_poll_wins_over_wedge_signals() {
        let mut detector = TdrDetector::new(0);
        // Engine Halted + executor BlockedOnPoll + no DMA -- fast-path
        // MaskPollUnsatisfied. Also satisfies Wedged{Quiescent} structurally.
        let signals = empty_engine_signals(EngineStatusSnapshot::Halted);
        let exec = blocked_on_poll_executor();
        let v = detector.classify(&signals, Some(&exec));
        assert!(matches!(v, TdrVerdict::MaskPollUnsatisfied), "got {v:?}");
    }

    #[test]
    fn classify_precedence_quiescent_wins_over_stalled() {
        // When BOTH would apply (cores terminal AND pending syncs with no
        // byte progress), Quiescent reports first because it's the stronger
        // description.
        let mut detector = TdrDetector::new(0);
        let signals = empty_engine_signals(EngineStatusSnapshot::Stalled);
        let exec = ExecutorSignals {
            is_done: false,
            syncs_satisfied: false,
            is_blocked_on_poll: false,
            pending_syncs: vec![(0, 0, 0, 0)],
        };
        // Burn enough cycles for quiescence threshold to fire.
        let mut fired_as = None;
        for _ in 0..(DEFAULT_QUIESCENCE_CYCLES * 2) {
            let v = detector.classify(&signals, Some(&exec));
            if let TdrVerdict::Wedged { reason, .. } = v {
                fired_as = Some(reason);
                break;
            }
        }
        assert_eq!(fired_as, Some(WedgeReason::Quiescent));
    }
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test --lib -p xdna-emu-core device::tdr::tests::classify`
Expected: all 3 new tests pass on the existing classify implementation (precedence is built into the if-chain order from Tasks 4-6).

If any fail, the precedence is wrong — fix the order in `classify` and re-run.

- [ ] **Step 3: Commit**

```bash
git add src/device/tdr/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: lock classify() precedence with collision tests

NaturalCompletion > MaskPollUnsatisfied > Wedged > Progressing, and
within Wedged: Quiescent > Stalled > PollExhausted (the latter
unreachable today, see Task 6). Three tests cover the cross-tier
collisions that matter: NaturalCompletion over Wedged signals,
MaskPollUnsatisfied over Wedged signals, Quiescent over Stalled.

Generated using Claude Code.
EOF
)"
```

---

## Task 8: Driver cross-check + create `src/device/context/` module

**Files:**
- Create: `src/device/context/mod.rs`
- Modify: `src/device/mod.rs`

- [ ] **Step 0: Driver cross-check** — read `~/npu-work/xdna-driver/src/driver/amdxdna/amdxdna_ctx.h` and `aie2_ctx_runqueue.c:1-100` for the driver's hwctx state vocabulary. Driver-side states (`enum amdxdna_ctx_state` or equivalent) typically include: Connected, Disconnected, Dead, Bad. Confirm the spec's enum is a faithful subset:
  - `Connected` ↔ driver's connected/running state ✓
  - `Stopped` ↔ reserved (driver has explicit stop; we don't enter it via Tier C)
  - `Failed { reason, diagnosis }` ↔ driver's Dead/Bad (we add the diagnosis payload, driver dumps via `aie2_rq_dump_all`)
  - `Disconnected` ↔ deferred to multi-context spec (firmware reload semantics)

If the driver vocabulary has materially changed since the spec was written, flag it and pause — Maya wants to know.

- [ ] **Step 1: Write failing tests for the Context type**

Create `src/device/context/mod.rs`:

```rust
//! Per-context state model for Tier C.
//!
//! Driver cross-reference: this mirrors `amdxdna_ctx`'s observable state
//! (state enum + completion counter) at the level the emulator needs to
//! expose to a future driver consumer doing TDR. Implementation is
//! emulator-original; behavior is constrained by the spec (see
//! `docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md`).

use crate::device::tdr::{TdrDiagnosis, WedgeReason};

/// Identifies a context. `Vec<Context>` is indexed by this value; today
/// there is always exactly one (`DEFAULT_CONTEXT`).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ContextId(pub u32);

pub const DEFAULT_CONTEXT: ContextId = ContextId(0);

/// Per-context state. Driver's hwctx vocabulary subset.
#[derive(Clone, Debug)]
pub enum ContextState {
    /// Ready to accept submissions.
    Connected,
    /// Idle but re-Connectable without firmware reload. Reserved for the
    /// multi-context spec; not entered by any Tier C path.
    #[allow(dead_code)]
    Stopped,
    /// Submission wedged; carries reason + diagnostic snapshot.
    Failed { reason: WedgeReason, diagnosis: TdrDiagnosis },
    // Disconnected (firmware-reload required) reserved for multi-context.
}

pub struct Context {
    pub id: ContextId,
    pub state: ContextState,
    pub completed_counter: u64,
    pub pending_cmd_count: u32,
}

impl Context {
    pub fn new(id: ContextId) -> Self {
        Self {
            id,
            state: ContextState::Connected,
            completed_counter: 0,
            pending_cmd_count: 0,
        }
    }

    pub fn mark_failed(&mut self, reason: WedgeReason, diagnosis: TdrDiagnosis) {
        self.state = ContextState::Failed { reason, diagnosis };
    }

    pub fn mark_connected(&mut self) {
        self.state = ContextState::Connected;
    }

    pub fn note_submission_complete(&mut self) {
        self.completed_counter = self.completed_counter.saturating_add(1);
    }

    pub fn is_connected(&self) -> bool {
        matches!(self.state, ContextState::Connected)
    }

    pub fn is_failed(&self) -> bool {
        matches!(self.state, ContextState::Failed { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_diagnosis() -> TdrDiagnosis {
        TdrDiagnosis {
            core_states: vec![],
            dma_states: vec![],
            data_in_flight: false,
            pending_syncs: vec![],
        }
    }

    #[test]
    fn new_context_starts_connected_with_zero_counter() {
        let ctx = Context::new(DEFAULT_CONTEXT);
        assert!(ctx.is_connected());
        assert!(!ctx.is_failed());
        assert_eq!(ctx.completed_counter, 0);
        assert_eq!(ctx.pending_cmd_count, 0);
        assert_eq!(ctx.id, DEFAULT_CONTEXT);
    }

    #[test]
    fn mark_failed_transitions_state_and_preserves_counter() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.completed_counter = 7;
        ctx.mark_failed(WedgeReason::Quiescent, fake_diagnosis());
        assert!(ctx.is_failed());
        assert!(!ctx.is_connected());
        assert_eq!(ctx.completed_counter, 7, "counter should not reset on failure");
    }

    #[test]
    fn mark_connected_clears_failed_state() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.mark_failed(WedgeReason::Stalled, fake_diagnosis());
        ctx.mark_connected();
        assert!(ctx.is_connected());
    }

    #[test]
    fn note_submission_complete_advances_counter() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.note_submission_complete();
        ctx.note_submission_complete();
        ctx.note_submission_complete();
        assert_eq!(ctx.completed_counter, 3);
    }

    #[test]
    fn mark_connected_is_idempotent_on_connected_context() {
        let mut ctx = Context::new(DEFAULT_CONTEXT);
        ctx.mark_connected();
        ctx.mark_connected();
        assert!(ctx.is_connected());
    }

    #[test]
    fn default_context_id_is_zero() {
        assert_eq!(DEFAULT_CONTEXT, ContextId(0));
    }
}
```

- [ ] **Step 2: Wire the module into the device tree**

In `src/device/mod.rs`, add:
```rust
pub mod context;
```
(next to the `pub mod tdr;` line from Task 1).

- [ ] **Step 3: Update Task 3's `u32` field to `ContextId`**

In `src/device/tdr/mod.rs`:
- Add at the top: `use crate::device::context::ContextId;`
- Change `context_id: u32` → `context_id: ContextId` on the struct field
- Change `pub fn new(context_id: u32) -> Self` → `pub fn new(context_id: ContextId) -> Self`
- Change `pub fn context_id(&self) -> u32` → `pub fn context_id(&self) -> ContextId`
- Update all classify tests that called `TdrDetector::new(0)` to `TdrDetector::new(DEFAULT_CONTEXT)` (need to also import: `use crate::device::context::DEFAULT_CONTEXT;` in the tests module).

- [ ] **Step 4: Run tests**

Run: `cargo test --lib -p xdna-emu-core device::context::tests`
Expected: 6 tests pass.

Run: `cargo test --lib -p xdna-emu-core device::tdr`
Expected: all tdr tests still pass after the ContextId tightening.

- [ ] **Step 5: Commit**

```bash
git add src/device/mod.rs src/device/context/mod.rs src/device/tdr/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: add Context module + tighten TdrDetector ContextId type

ContextId(pub u32) newtype, DEFAULT_CONTEXT = ContextId(0) constant.
Context struct holds state (Connected/Stopped/Failed), completed_counter,
pending_cmd_count. Stopped is reserved (multi-context spec); Failed
carries WedgeReason + TdrDiagnosis. mark_failed preserves the counter
(failures don't reset progress tracking) -- covered by unit test.

TdrDetector's previously-u32 context_id field tightens to ContextId
now that the type exists. Tests updated to DEFAULT_CONTEXT.

Driver cross-check: amdxdna_ctx state vocabulary noted in module doc.

Generated using Claude Code.
EOF
)"
```

---

## Task 9: Wire `Vec<Context>` + `Vec<TdrDetector>` into `DeviceState`

**Files:**
- Modify: `src/device/state/mod.rs`

- [ ] **Step 1: Write failing test for new constructor populating one context**

Append to `src/device/state/tests.rs` (this file exists):

```rust
#[test]
fn device_state_new_populates_one_default_context() {
    use crate::device::context::{ContextState, DEFAULT_CONTEXT};

    let state = DeviceState::new_npu1();
    assert_eq!(state.contexts.len(), 1, "should ship with one default context");
    assert_eq!(state.contexts[0].id, DEFAULT_CONTEXT);
    assert!(matches!(state.contexts[0].state, ContextState::Connected));
    assert_eq!(state.tdr_detectors.len(), 1, "one detector per context");
    assert_eq!(state.tdr_detectors[0].context_id(), DEFAULT_CONTEXT);
}

#[test]
fn device_state_reset_context_transitions_failed_to_connected() {
    use crate::device::context::{ContextState, DEFAULT_CONTEXT};
    use crate::device::tdr::{TdrDiagnosis, WedgeReason};

    let mut state = DeviceState::new_npu1();
    let diag = TdrDiagnosis {
        core_states: vec![],
        dma_states: vec![],
        data_in_flight: false,
        pending_syncs: vec![],
    };
    state.contexts[0].mark_failed(WedgeReason::Quiescent, diag);
    assert!(state.contexts[0].is_failed());

    state.reset_context(DEFAULT_CONTEXT).expect("reset failed");
    assert!(matches!(state.contexts[0].state, ContextState::Connected));
}
```


- [ ] **Step 2: Run to verify failure**

Run: `cargo test --lib -p xdna-emu-core device::state::tests::device_state`
Expected: compile errors — `state.contexts`, `state.tdr_detectors`, `state.reset_context` don't exist.

- [ ] **Step 3: Add fields, constructor population, and reset method**

In `src/device/state/mod.rs`, update the imports:

```rust
use super::context::{Context, ContextId, DEFAULT_CONTEXT, ContextState};
use super::tdr::TdrDetector;
```

Add fields to `DeviceState`:

```rust
pub struct DeviceState {
    // ...existing fields, after async_errors:
    pub async_errors: AsyncErrorSink,
    /// Per-context state. Single context (DEFAULT_CONTEXT) today; multi-context
    /// expansion is storage-only -- all APIs already key by ContextId.
    pub contexts: Vec<Context>,
    /// Per-context TDR classifier. Parallel index to `contexts`.
    pub tdr_detectors: Vec<TdrDetector>,
}
```

Update the constructor:

```rust
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        let array = TileArray::new(arch);
        let num_cols = array.cols() as usize;
        let contexts = vec![Context::new(DEFAULT_CONTEXT)];
        let tdr_detectors = vec![TdrDetector::new(DEFAULT_CONTEXT)];
        Self {
            array,
            stats: CdoStats::default(),
            pending_core_enables: Vec::new(),
            start_col: 0,
            async_errors: AsyncErrorSink::new(num_cols),
            contexts,
            tdr_detectors,
        }
    }
```

Add a `reset_context` method:

```rust
    /// Reset the given context to Connected and clear its Tier B sink slot.
    ///
    /// Idempotent on an already-Connected context. Returns an error if the
    /// context_id is out of range.
    pub fn reset_context(&mut self, context_id: ContextId) -> Result<(), ResetContextError> {
        let idx = context_id.0 as usize;
        let ctx = self.contexts.get_mut(idx)
            .ok_or(ResetContextError::InvalidContextId)?;
        ctx.mark_connected();
        // Tier B: clear async errors for this context. Today AsyncErrorSink
        // is global; multi-context spec will give it per-context slots.
        self.async_errors.clear();
        // The engine.reset_for_new_context() call happens at the FFI layer
        // (xdna_emu_reset_context); this method only touches device-side state.
        Ok(())
    }
```

Add the error type near the top of the file:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetContextError {
    InvalidContextId,
}
```

- [ ] **Step 4: Run to verify passing**

Run: `cargo test --lib -p xdna-emu-core device::state`
Expected: the 2 new tests pass; existing state tests unaffected.

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu-core`
Expected: full library passes.

- [ ] **Step 5: Commit**

```bash
git add src/device/state/mod.rs
git commit -m "$(cat <<'EOF'
interrupt: wire Context/TdrDetector into DeviceState

DeviceState gains contexts: Vec<Context> and tdr_detectors:
Vec<TdrDetector>, both populated with one DEFAULT_CONTEXT element by
the constructor. reset_context(ContextId) transitions a context's
state to Connected and clears the Tier B async-error sink for it.
ResetContextError::InvalidContextId for out-of-range ids.

The engine-side reset_for_new_context() is still called by the FFI
layer (xdna_emu_reset_context); this device-side method only touches
device-state.

Generated using Claude Code.
EOF
)"
```

---

## Task 10: Add `XdnaEmuHaltReason::WedgeRecovered` variant

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs`

- [ ] **Step 1: Write failing test for the variant**

The canonical Tier C completeness file is created in Task 16. For this task, add the test inline in the existing `#[cfg(test)] mod tests` block at `crates/xdna-emu-ffi/src/lib.rs:291`:

```rust
    #[test]
    fn halt_reason_wedge_recovered_has_discriminant_four() {
        let r = XdnaEmuHaltReason::WedgeRecovered;
        assert_eq!(r as u32, 4);
    }
```

(Task 16 promotes the canonical assertion to `tests/tier_c_completeness.rs`; this inline test stays as a unit-level guard.)

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p xdna-emu-ffi`
Expected: compile error — `WedgeRecovered` variant doesn't exist.

- [ ] **Step 3: Add the variant**

In `crates/xdna-emu-ffi/src/lib.rs` (line ~152, after `MaskPollUnsatisfied = 3`):

```rust
    MaskPollUnsatisfied = 3,
    /// Tier C: the in-flight submission wedged; the per-context state is
    /// Failed. Caller should observe this, call `xdna_emu_reset_context`
    /// before the next submission, and translate to an EIO-shaped XRT
    /// command state in `run.wait()`.
    ///
    /// "Recovered" describes the contract: the device is ready to accept
    /// the next submission once reset is called; this status code is the
    /// signal that triggered recovery, not that recovery has already run.
    WedgeRecovered = 4,
}
```

- [ ] **Step 4: Run to verify passing**

Run: `cargo test -p xdna-emu-ffi halt_reason_wedge_recovered`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs
git commit -m "$(cat <<'EOF'
interrupt: add XdnaEmuHaltReason::WedgeRecovered = 4

The halt status xdna_emu_run returns when TdrDetector classifies the
in-flight submission as Wedged. Plugin translates to EIO-shaped XRT
command state; caller calls xdna_emu_reset_context before the next
submission.

"Recovered" describes the contract, not that recovery has run -- the
status is the trigger signal.

Generated using Claude Code.
EOF
)"
```

---

## Task 11: Driver cross-check + integrate TdrDetector into `xdna_emu_run`

This is the biggest task: refactor `execution.rs`'s run loop to consume the classifier, transition context state on terminal verdicts, and enforce the entry guard.

**Files:**
- Modify: `crates/xdna-emu-ffi/src/execution.rs`

- [ ] **Step 0: Driver cross-check** — re-read `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c:38-48` (the `aie2_tdr_force_recover` chain). Note the contract: when TDR fires, the driver dumps, then stops, then restarts. Our equivalent: when classify returns Wedged, we mark the context Failed (the dump-equivalent is in the diagnosis payload). Restart is plugin-driven via xdna_emu_reset_context, not auto.

- [ ] **Step 1: Write integration test asserting WedgeRecovered halt and Failed context state**

Append to `crates/xdna-emu-ffi/tests/` (create a new test file `tier_c_wedge.rs` if appropriate):

```rust
//! Tier C: verify the refactored run loop preserves the happy-path
//! Completed return. The full wedge -> WedgeRecovered cycle is exercised
//! at the device-state layer in Task 15 (the FFI-end-to-end wedge fixture
//! is deferred -- no wedging xclbin in our corpus today).

use xdna_emu_ffi::*;

#[test]
fn run_preserves_happy_path_completed_after_classifier_refactor() {
    // Minimal handle, no NPU instructions, no cores enabled -- the engine
    // halts immediately at warm-up, then the classifier hits NaturalCompletion
    // (syncs trivially satisfied: none pending). Refactor regression guard.
    let handle = unsafe { xdna_emu_create() };
    assert!(!handle.is_null());

    unsafe { xdna_emu_set_max_cycles(handle, 1000) };
    let status = unsafe { xdna_emu_run(handle) };

    assert!(matches!(status.halt_reason, XdnaEmuHaltReason::Completed),
            "got {:?}", status.halt_reason);

    unsafe { xdna_emu_destroy(handle) };
}
```

- [ ] **Step 2: Run to confirm baseline**

Run: `cargo test -p xdna-emu-ffi run_returns_wedge_recovered_on_synthetic_deadlock`
Expected: passes against current (pre-refactor) code, since the run path leads to NaturalCompletion.

- [ ] **Step 3: Refactor `xdna_emu_run` to use `TdrDetector::classify`**

This is a substantial edit to `crates/xdna-emu-ffi/src/execution.rs`. The end state, replacing the body of `xdna_emu_run` starting at the `'run: while unbounded || cycles < max` loop:

```rust
    // Entry guard: context(0) must be Connected. Otherwise the caller
    // forgot to call xdna_emu_reset_context after a prior wedge.
    {
        use xdna_emu_core::device::context::DEFAULT_CONTEXT;
        let device = handle.engine.device();
        let ctx = &device.contexts[DEFAULT_CONTEXT.0 as usize];
        if !ctx.is_connected() {
            log::error!("xdna_emu_run: context {:?} not Connected (state: {:?}); \
                         caller must reset_context before re-submitting",
                        DEFAULT_CONTEXT, std::mem::discriminant(&ctx.state));
            return XdnaEmuExecStatus {
                result: XdnaEmuResult::ExecutionError,
                cycles_executed: 0,
                halted: false,
                halt_reason: XdnaEmuHaltReason::Error,
            };
        }
    }

    // Per-cycle: build snapshots, classify, dispatch on verdict.
    use xdna_emu_core::device::tdr::{
        EngineSignals, EngineStatusSnapshot, ExecutorSignals, TdrVerdict, WedgeReason,
    };
    use xdna_emu_core::device::context::DEFAULT_CONTEXT;

    let mut natural_halt = false;
    let mut maskpoll_unsatisfied = false;
    let mut wedged: Option<(WedgeReason, _)> = None;  // captures diagnosis for context update

    'run: while unbounded || cycles < max {
        handle.engine.device_mut().array.set_dma_cycle(cycles);

        let npu_progressed;
        {
            let (device, host_mem) = handle.engine.device_and_host_memory();
            let result = handle.npu_executor.try_advance(device, host_mem);
            if let xdna_emu_core::npu::AdvanceResult::Error(msg) = result {
                log::error!("NPU executor fatal: {}", msg);
                handle.engine.flush_trace_to_host();
                return XdnaEmuExecStatus {
                    result: XdnaEmuResult::ExecutionError,
                    cycles_executed: cycles,
                    halted: false,
                    halt_reason: XdnaEmuHaltReason::Error,
                };
            }
            npu_progressed = matches!(result, xdna_emu_core::npu::AdvanceResult::Progressed);
        }

        if npu_progressed {
            handle.engine.flush_ctrl_packets();
        }

        handle.engine.step();
        crate::async_errors::fire_async_callbacks_for(handle);
        cycles += 1;

        // Build per-cycle snapshots and classify.
        let engine_signals = build_engine_signals(&handle.engine);
        let executor_signals = build_executor_signals(&handle.npu_executor, &handle.engine);

        let device = handle.engine.device_mut();
        let detector = &mut device.tdr_detectors[DEFAULT_CONTEXT.0 as usize];
        let verdict = detector.classify(&engine_signals, executor_signals.as_ref());

        match verdict {
            TdrVerdict::Progressing => continue,
            TdrVerdict::NaturalCompletion => {
                log::info!("Natural completion after {} cycles", cycles);
                natural_halt = true;
                break 'run;
            }
            TdrVerdict::MaskPollUnsatisfied => {
                log::info!("MASKPOLL unsatisfiable after {} cycles", cycles);
                maskpoll_unsatisfied = true;
                natural_halt = true;
                break 'run;
            }
            TdrVerdict::Wedged { reason, diagnosis } => {
                log::warn!("Tier C wedge after {} cycles: {:?} -- {}", cycles, reason, diagnosis);
                wedged = Some((reason, diagnosis));
                break 'run;
            }
        }
    }

    handle.engine.flush_trace_to_host();

    // On wedge, transition context state.
    if let Some((reason, diagnosis)) = wedged {
        let device = handle.engine.device_mut();
        device.contexts[DEFAULT_CONTEXT.0 as usize].mark_failed(reason, diagnosis);
        return XdnaEmuExecStatus {
            result: XdnaEmuResult::Success,
            cycles_executed: cycles,
            // halted=true: the run terminated (not still running) -- the
            // plugin's last_run_complete_ flag should fire so run.wait() exits
            // its wait loop. The distinction between "completed cleanly" and
            // "aborted via TDR" is carried by halt_reason, not halted.
            halted: true,
            halt_reason: XdnaEmuHaltReason::WedgeRecovered,
        };
    }

    // On natural completion, advance the context's completed_counter.
    if natural_halt && !maskpoll_unsatisfied {
        let device = handle.engine.device_mut();
        device.contexts[DEFAULT_CONTEXT.0 as usize].note_submission_complete();
    }

    let halted = maskpoll_unsatisfied || natural_halt;
    let halt_reason = if maskpoll_unsatisfied {
        XdnaEmuHaltReason::MaskPollUnsatisfied
    } else if natural_halt {
        XdnaEmuHaltReason::Completed
    } else {
        XdnaEmuHaltReason::Budget
    };

    XdnaEmuExecStatus { result: XdnaEmuResult::Success, cycles_executed: cycles, halted, halt_reason }
}
```

Add the snapshot builder helpers at the bottom of `execution.rs`:

```rust
fn build_engine_signals(engine: &xdna_emu_core::interpreter::engine::InterpreterEngine)
    -> xdna_emu_core::device::tdr::EngineSignals
{
    use xdna_emu_core::device::tdr::{EngineSignals, EngineStatusSnapshot};
    use xdna_emu_core::interpreter::engine::EngineStatus;

    let status = match engine.status() {
        EngineStatus::Running => EngineStatusSnapshot::Running,
        EngineStatus::Halted => EngineStatusSnapshot::Halted,
        EngineStatus::Stalled => EngineStatusSnapshot::Stalled,
        _ => EngineStatusSnapshot::Other,
    };
    let device = engine.device();

    let mut core_statuses = Vec::new();
    for col in 0..device.cols() {
        for row in 2..device.rows() {
            if engine.is_core_enabled(col, row) {
                if let Some(s) = engine.core_status(col, row) {
                    core_statuses.push((col as u8, row as u8, s));
                }
            }
        }
    }

    let mut dma_states = Vec::new();
    use xdna_emu_core::device::dma::engine::ChannelState;
    for col in 0..device.cols() {
        for row in 0..device.rows() {
            if let Some(dma) = device.array.dma_engine(col as u8, row as u8) {
                for ch in 0..dma.num_channels() {
                    let state = dma.channel_state(ch as u8);
                    if !matches!(state, ChannelState::Idle) {
                        let desc = dma.channel_fsm_description(ch as u8);
                        dma_states.push((col as u8, row as u8, ch as u8, desc));
                    }
                }
            }
        }
    }

    EngineSignals {
        engine_status: status,
        any_dma_active: device.array.any_dma_active(),
        any_data_in_flight: device.array.any_data_in_flight(),
        total_dma_bytes_transferred: device.array.total_dma_bytes_transferred(),
        total_lock_releases: device.array.total_lock_releases(),
        core_statuses,
        dma_states,
    }
}

fn build_executor_signals(
    executor: &xdna_emu_core::npu::NpuExecutor,
    engine: &xdna_emu_core::interpreter::engine::InterpreterEngine,
) -> Option<xdna_emu_core::device::tdr::ExecutorSignals> {
    use xdna_emu_core::device::tdr::ExecutorSignals;
    Some(ExecutorSignals {
        is_done: executor.is_done(),
        syncs_satisfied: executor.syncs_satisfied(engine.device()),
        is_blocked_on_poll: executor.is_blocked_on_poll(),
        pending_syncs: executor.pending_syncs().iter()
            .map(|s| (s.column, s.row, s.channel, s.direction))
            .collect(),
    })
}
```

(If `NpuExecutor` isn't always present, refactor `build_executor_signals` to take `Option<&NpuExecutor>` and return `None` when input is None.)

- [ ] **Step 4: Run all FFI tests**

Run: `cargo test -p xdna-emu-ffi`
Expected: all pass, including the new `run_returns_wedge_recovered_on_synthetic_deadlock` test and any existing run-related tests.

- [ ] **Step 5: Run full library tests**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: full pass.

- [ ] **Step 6: Build the cdylib so the plugin can pick it up later**

Run: `cargo build -p xdna-emu-ffi`
Expected: clean build.

- [ ] **Step 7: Commit**

```bash
git add crates/xdna-emu-ffi/src/execution.rs crates/xdna-emu-ffi/tests/tier_c_wedge.rs
git commit -m "$(cat <<'EOF'
interrupt: route xdna_emu_run through TdrDetector::classify

The per-cycle stall/poll discrimination that lived inline in the run
loop (POLL_STALL_LIMIT, EngineStatus::Stalled handling) consolidates
into TdrDetector::classify. The run loop now: builds engine + executor
snapshots, classifies, dispatches on the verdict.

On Wedged the context is marked Failed (with reason + diagnosis) and
the run returns halt_reason = WedgeRecovered. On NaturalCompletion the
context's completed_counter advances. Entry guard rejects runs against
a non-Connected context with ExecutionError + log.

Behavior preservation: MaskPollUnsatisfied path stays semantically
identical (now reached via the classifier, two-path internally:
fast-path engine-halted + budget-exhausted blocked-on-poll).

Driver cross-check: aie2_tdr_force_recover chain noted; mark_failed is
the dump-equivalent (diagnosis carries the snapshot); reset stays
plugin-driven, not auto-fired on wedge.

Generated using Claude Code.
EOF
)"
```

---

## Task 12: Add `xdna_emu_get_context_state` FFI accessor

**Files:**
- Create: `crates/xdna-emu-ffi/src/context.rs`
- Modify: `crates/xdna-emu-ffi/src/lib.rs` (declare module, re-export)

- [ ] **Step 1: Write failing test**

Create `crates/xdna-emu-ffi/tests/context_accessor.rs`:

```rust
use xdna_emu_ffi::*;

#[test]
fn get_context_state_returns_connected_for_default_context_after_create() {
    let handle = unsafe { xdna_emu_create() };
    assert!(!handle.is_null());

    let mut state: u32 = 99;
    let mut counter: u64 = 99;
    let rc = unsafe {
        xdna_emu_get_context_state(
            handle,
            0,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, 0, "expected Success");
    assert_eq!(state, XdnaEmuContextState::Connected as u32);
    assert_eq!(counter, 0);

    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn get_context_state_returns_invalid_for_unknown_context_id() {
    let handle = unsafe { xdna_emu_create() };
    let mut state: u32 = 0;
    let mut counter: u64 = 0;
    let rc = unsafe {
        xdna_emu_get_context_state(
            handle,
            999,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, -2, "expected invalid-context-id");
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn get_context_state_rejects_null_handle() {
    let mut state: u32 = 0;
    let mut counter: u64 = 0;
    let rc = unsafe {
        xdna_emu_get_context_state(
            std::ptr::null_mut(),
            0,
            &mut state as *mut u32 as *mut XdnaEmuContextState,
            &mut counter as *mut u64,
        )
    };
    assert_eq!(rc, -1, "expected null-handle");
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p xdna-emu-ffi --test context_accessor`
Expected: compile error — `xdna_emu_get_context_state`, `XdnaEmuContextState` don't exist.

- [ ] **Step 3: Implement the FFI module**

Create `crates/xdna-emu-ffi/src/context.rs`:

```rust
//! Tier C FFI: per-context state accessor.

use super::XdnaEmuHandle;

/// Mirror of [`xdna_emu_core::device::context::ContextState`] discriminants
/// for the C ABI.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum XdnaEmuContextState {
    Connected = 0,
    Stopped = 1,
    Failed = 2,
}

/// Read the current state and completion counter of a context.
///
/// # Returns
/// - `0` on success
/// - `-1` if `handle` is null or any out-pointer is null
/// - `-2` if `context_id` is out of range
///
/// # Safety
/// - `handle` must be valid
/// - `out_state` must point to writable `XdnaEmuContextState`
/// - `out_completed_counter` must point to writable `u64`
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_get_context_state(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
    out_state: *mut XdnaEmuContextState,
    out_completed_counter: *mut u64,
) -> i32 {
    if handle.is_null() || out_state.is_null() || out_completed_counter.is_null() {
        return -1;
    }
    let handle = &*handle;
    let device = handle.engine.device();
    let ctx = match device.contexts.get(context_id as usize) {
        Some(c) => c,
        None => return -2,
    };

    use xdna_emu_core::device::context::ContextState;
    let state = match ctx.state {
        ContextState::Connected => XdnaEmuContextState::Connected,
        ContextState::Stopped => XdnaEmuContextState::Stopped,
        ContextState::Failed { .. } => XdnaEmuContextState::Failed,
    };
    *out_state = state;
    *out_completed_counter = ctx.completed_counter;
    0
}
```

Add to `crates/xdna-emu-ffi/src/lib.rs` (near other `pub mod` declarations):

```rust
pub mod context;
pub use context::{xdna_emu_get_context_state, XdnaEmuContextState};
```

- [ ] **Step 4: Run to verify passing**

Run: `cargo test -p xdna-emu-ffi --test context_accessor`
Expected: 3 tests pass.

- [ ] **Step 5: Build cdylib + check symbol export**

Run: `cargo build -p xdna-emu-ffi`
Then verify the symbol is exported:
```bash
nm -D target/debug/libxdna_emu.so | grep xdna_emu_get_context_state
```
Expected: one matching line with `T` (text, exported).

- [ ] **Step 6: Commit**

```bash
git add crates/xdna-emu-ffi/src/context.rs crates/xdna-emu-ffi/src/lib.rs crates/xdna-emu-ffi/tests/context_accessor.rs
git commit -m "$(cat <<'EOF'
interrupt: add xdna_emu_get_context_state FFI accessor

XdnaEmuContextState enum (Connected/Stopped/Failed) mirrors the
device-side ContextState discriminants. xdna_emu_get_context_state
reads state + completed_counter for a context_id (-1 null, -2 invalid
id, 0 success).

Plugin wires up in Task 13 with resolve_required so a stale
libxdna_emu.so fails dlopen-time loudly.

Generated using Claude Code.
EOF
)"
```

---

## Task 13: Update `xdna_emu_reset_context` to take `context_id` (breaking) + upgrade plugin to `resolve_required`

**Files:**
- Modify: `crates/xdna-emu-ffi/src/lib.rs`
- Modify: `xrt-plugin/src/transport.h`
- Modify: `xrt-plugin/src/transport_inprocess.h`
- Modify: `xrt-plugin/src/transport_inprocess.cpp`
- Modify: `xrt-plugin/src/platform_emu.cpp` (other callers of reset_context)

- [ ] **Step 1: Write failing test for the new signature on the Rust side**

Append to `crates/xdna-emu-ffi/tests/context_accessor.rs`:

```rust
#[test]
fn reset_context_with_default_id_succeeds_on_fresh_handle() {
    let handle = unsafe { xdna_emu_create() };
    let rc = unsafe { xdna_emu_reset_context(handle, 0) };
    assert_eq!(rc, XdnaEmuResult::Success);
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn reset_context_returns_error_for_invalid_id() {
    let handle = unsafe { xdna_emu_create() };
    let rc = unsafe { xdna_emu_reset_context(handle, 999) };
    assert_eq!(rc, XdnaEmuResult::ExecutionError);
    unsafe { xdna_emu_destroy(handle) };
}

#[test]
fn reset_context_transitions_failed_context_to_connected() {
    // After a wedge marks context Failed, reset_context restores Connected.
    // This test fakes the Failed state via the device API (full wedge path
    // is exercised in the integration test in Task 15).
    let handle = unsafe { xdna_emu_create() };

    // Reach into the device through the handle to mark it Failed.
    // (Test-only access; if the FFI doesn't expose this, the integration
    // test in Task 15 is the place to test the full cycle.)
    {
        let handle_mut = unsafe { &mut *handle };
        let device = handle_mut.engine.device_mut();
        use xdna_emu_core::device::tdr::{TdrDiagnosis, WedgeReason};
        device.contexts[0].mark_failed(
            WedgeReason::Quiescent,
            TdrDiagnosis {
                core_states: vec![],
                dma_states: vec![],
                data_in_flight: false,
                pending_syncs: vec![],
            },
        );
        assert!(device.contexts[0].is_failed());
    }

    let rc = unsafe { xdna_emu_reset_context(handle, 0) };
    assert_eq!(rc, XdnaEmuResult::Success);

    {
        let handle_ref = unsafe { &*handle };
        assert!(handle_ref.engine.device().contexts[0].is_connected());
    }

    unsafe { xdna_emu_destroy(handle) };
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p xdna-emu-ffi --test context_accessor`
Expected: compile error — `xdna_emu_reset_context` takes only one arg.

- [ ] **Step 3: Update the Rust FFI signature**

In `crates/xdna-emu-ffi/src/lib.rs` (line ~245):

```rust
/// Reset a context for a fresh submission.
///
/// Transitions the named context from Failed -> Connected (no-op on already-
/// Connected), clears its Tier B async-error sink, and calls the engine's
/// per-context reset to wipe stale tile state.
///
/// Call this between submissions, and especially after observing a
/// `WedgeRecovered` halt -- the next `xdna_emu_run` entry rejects a non-
/// Connected context.
///
/// Returns:
/// - `Success` on a clean reset
/// - `InvalidHandle` for a null handle
/// - `ExecutionError` for an invalid context_id
///
/// # Safety
/// - `handle` must be valid
#[no_mangle]
pub unsafe extern "C" fn xdna_emu_reset_context(
    handle: *mut XdnaEmuHandle,
    context_id: u32,
) -> XdnaEmuResult {
    if handle.is_null() {
        return XdnaEmuResult::InvalidHandle;
    }

    let handle = &mut *handle;
    use xdna_emu_core::device::context::ContextId;
    let cid = ContextId(context_id);
    if let Err(_) = handle.engine.device_mut().reset_context(cid) {
        log::error!("xdna_emu_reset_context: invalid context_id {}", context_id);
        return XdnaEmuResult::ExecutionError;
    }
    handle.engine.reset_for_new_context();
    log::debug!("xdna_emu_reset_context: cleared per-context tile state for ctx {}", context_id);
    XdnaEmuResult::Success
}
```

- [ ] **Step 4: Run to verify Rust-side tests pass**

Run: `cargo test -p xdna-emu-ffi --test context_accessor`
Expected: 3 reset tests pass.

- [ ] **Step 5: Update plugin C++ declarations**

In `xrt-plugin/src/transport.h`:

```cpp
// Change:
//   virtual void reset_context() {}
// To:
virtual void reset_context(uint32_t context_id) {}
```

In `xrt-plugin/src/transport_inprocess.h`:

```cpp
// Change function pointer typedef:
//   using fn_reset_context = Result (*)(XdnaEmuHandle*);
// To:
using fn_reset_context = Result (*)(XdnaEmuHandle*, uint32_t);

// Change override declaration:
//   void reset_context() override;
// To:
void reset_context(uint32_t context_id) override;
```

In `xrt-plugin/src/transport_inprocess.cpp`:

Find `sym_reset_context_ = resolve_optional<...>(...)` (line ~110) and change to:
```cpp
sym_reset_context_ = resolve_required<fn_reset_context>("xdna_emu_reset_context");
```

Find the `void emu_transport_inprocess::reset_context()` impl (line ~240) and change to:
```cpp
void emu_transport_inprocess::reset_context(uint32_t context_id)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    Result rc = sym_reset_context_(emu_, context_id);
    check(rc, "reset_context");
}
```

(Note the fallback-on-null-symbol code is removed; we now fail-loud at dlopen time.)

- [ ] **Step 6: Update plugin callers of `reset_context()`**

Grep for callers:
```bash
grep -rn "reset_context()" xrt-plugin/src/
```

Update each to pass `0` (DEFAULT_CONTEXT id). Likely just `platform_emu.cpp`; verify by grep.

- [ ] **Step 7: Build the plugin to verify the signature change compiles**

Run: `./scripts/rebuild-plugin.sh`
Expected: clean build and install.

- [ ] **Step 8: Commit**

```bash
git add crates/xdna-emu-ffi/src/lib.rs crates/xdna-emu-ffi/tests/context_accessor.rs \
        xrt-plugin/src/transport.h xrt-plugin/src/transport_inprocess.h \
        xrt-plugin/src/transport_inprocess.cpp xrt-plugin/src/platform_emu.cpp
git commit -m "$(cat <<'EOF'
interrupt: xdna_emu_reset_context takes context_id; plugin resolves required

Breaking FFI signature: xdna_emu_reset_context(handle, context_id: u32).
All plugin callers updated to pass 0 (DEFAULT_CONTEXT) -- they are the
only consumers outside the workspace.

Plugin upgrades the symbol resolution from resolve_optional to
resolve_required, matching the Tier B convention: a stale libxdna_emu.so
fails dlopen-time loudly instead of silently no-oping. The prior optional
fallback was a holdover from when the FFI was less stable.

reset_context returns ExecutionError on invalid context_id (covered by
test); InvalidHandle on null; Success on clean reset including the
Failed -> Connected transition.

Generated using Claude Code.
EOF
)"
```

---

## Task 14: Plugin halt-reason mapping + `get_context_state` virtual method

**Files:**
- Modify: `xrt-plugin/src/transport.h`
- Modify: `xrt-plugin/src/transport_inprocess.h`
- Modify: `xrt-plugin/src/transport_inprocess.cpp`
- Modify: wherever the run-status consumer translates `XdnaEmuHaltReason` (grep for `XdnaEmuHaltReason::Completed` in C++ to find it; likely `platform_emu.cpp`)

- [ ] **Step 1: Add the virtual method on emu_transport base**

In `xrt-plugin/src/transport.h`, add alongside `reset_context`:

```cpp
struct ContextStateRecord {
    uint32_t state;            // 0=Connected, 1=Stopped, 2=Failed
    uint64_t completed_counter;
};

// Returns false if the context_id is invalid or unsupported by the transport.
virtual bool get_context_state(uint32_t context_id, ContextStateRecord& out) {
    (void)context_id; (void)out;
    return false;
}
```

- [ ] **Step 2: Wire the in-process override**

In `xrt-plugin/src/transport_inprocess.h`:

```cpp
// Add typedef alongside other fn_ typedefs:
using fn_get_context_state = int (*)(XdnaEmuHandle*, uint32_t, uint32_t*, uint64_t*);

// Add override declaration:
bool get_context_state(uint32_t context_id, ContextStateRecord& out) override;

// Add symbol member:
fn_get_context_state sym_get_context_state_ = nullptr;
```

In `xrt-plugin/src/transport_inprocess.cpp`:

```cpp
// In the constructor's symbol-resolution block (near sym_reset_context_):
sym_get_context_state_ = resolve_required<fn_get_context_state>("xdna_emu_get_context_state");

// Add the override impl:
bool emu_transport_inprocess::get_context_state(uint32_t context_id, ContextStateRecord& out)
{
    std::lock_guard<std::recursive_mutex> lock(ffi_lock_);
    uint32_t state = 0;
    uint64_t counter = 0;
    int rc = sym_get_context_state_(emu_, context_id, &state, &counter);
    if (rc != 0) {
        return false;
    }
    out.state = state;
    out.completed_counter = counter;
    return true;
}
```

- [ ] **Step 3: Mirror the new variant in the plugin's HaltReason enum**

In `xrt-plugin/src/transport_inprocess.h` (line ~106), extend the `HaltReason` enum:

```cpp
enum HaltReason : int {      // Mirrors Rust `XdnaEmuHaltReason`.
    HALT_COMPLETED            = 0,
    HALT_BUDGET               = 1,
    HALT_ERROR                = 2,
    HALT_MASKPOLL_UNSATISFIED = 3,
    HALT_WEDGE_RECOVERED      = 4,   // NEW: Tier C wedge; context now Failed
};
```

- [ ] **Step 4: Extend the diagnostic switch in transport_inprocess.cpp**

At `xrt-plugin/src/transport_inprocess.cpp:316` (the switch that builds `reason_str` for the `XDNA_EMU_STATUS:` log line), add the new case:

```cpp
switch (status.halt_reason) {
    case HALT_COMPLETED:            reason_str = "completed";            break;
    case HALT_BUDGET:               reason_str = "budget";               break;
    case HALT_ERROR:                reason_str = "error";                break;
    case HALT_MASKPOLL_UNSATISFIED: reason_str = "maskpoll_unsatisfied"; break;
    case HALT_WEDGE_RECOVERED:      reason_str = "wedge_recovered";      break;  // NEW
}
```

- [ ] **Step 5: Investigate + wire the EIO-shaped translation at the run-wait consumer**

The plugin's submission-completion path needs to surface `WedgeRecovered` as an error to XRT consumers (so `run.wait()` returns an abort-equivalent state instead of treating it like a clean completion).

Investigation steps (the implementer runs these and writes a small note in the commit message documenting what they found):

```bash
# Find where run.wait() / submission completion is decided in the plugin:
grep -rn "last_run_complete_\|run_wait\|ert_cmd_state\|ERT_CMD_STATE" xrt-plugin/src/

# Find where the bridge harness asserts success/failure:
grep -rn "halt_reason\|reason_str\|XDNA_EMU_STATUS" xrt-plugin/ bridge-runner/ scripts/
```

Expected findings:
- `transport_inprocess.cpp:310` sets `last_run_complete_ = (status.halted != 0)`. With our `halted: true` on wedge (Task 11), this fires correctly -- the wait exits.
- The plugin likely doesn't currently set an explicit `ert_cmd_state`; the bridge harness reads the `XDNA_EMU_STATUS:` log line to classify (verify this via grep).

What to add:
- If there's an explicit `ert_cmd_state` setter, map `HALT_WEDGE_RECOVERED` to `ERT_CMD_STATE_ABORT` (or whichever variant XRT declares for "driver killed this submission" -- check `<core/include/ert.h>` or wherever XRT defines the enum).
- If submission classification is only via the log line (no programmatic state), add a follow-up `last_run_wedged_` flag on `emu_transport_inprocess` and surface it via a virtual method on `emu_transport`. The bridge harness can read it through a new accessor in a future spec.

Document what was found and what was wired in the commit message. If the investigation reveals a structural blocker (e.g. no clean place for the abort-state), pause and surface to Maya -- this is a judgement call worth checking.

- [ ] **Step 6: Build the plugin**

Run: `./scripts/rebuild-plugin.sh`
Expected: clean build and install.

- [ ] **Step 7: Smoke-test (no HW)**

Run: `XDNA_EMU=1 cargo test -p xdna-emu-ffi`
Expected: all pass. Quick proxy that the FFI surface still matches the plugin's expectations.

- [ ] **Step 8: Commit**

```bash
git add xrt-plugin/src/transport.h xrt-plugin/src/transport_inprocess.h \
        xrt-plugin/src/transport_inprocess.cpp xrt-plugin/src/platform_emu.cpp
git commit -m "$(cat <<'EOF'
interrupt: plugin wires Tier C get_context_state + WedgeRecovered mapping

emu_transport gains a virtual get_context_state(context_id) returning a
ContextStateRecord (state + completed_counter). emu_transport_inprocess
overrides it via the FFI, resolved required at dlopen time -- stale .so
fails loudly. ContextStateRecord is the plugin-side mirror of the
device-side Context state.

xdna_emu_run's WedgeRecovered halt_reason maps to ERT_CMD_STATE_ABORT
at the platform_emu run-consumer site. xrt::run.wait() consumers
observe abort instead of hanging. The caller (test harness, application)
is responsible for calling reset_context before the next submission --
the FFI entry guard rejects non-Connected contexts.

Generated using Claude Code.
EOF
)"
```

---

## Task 15: Integration test — end-to-end wedge → recovery + Tier B orthogonality

**Files:**
- Create: `src/device/state/tests/tier_c_integration.rs` (or extend existing tests file)

- [ ] **Step 1: Write a wedge-triggering integration test using the device API directly**

Add to `src/device/state/tests.rs` (or create the file):

```rust
//! Tier C integration: synthesize a wedge via classifier, observe state
//! transition, recover, verify Tier B orthogonality.

use crate::device::context::{ContextState, DEFAULT_CONTEXT};
use crate::device::tdr::{
    EngineSignals, EngineStatusSnapshot, ExecutorSignals, TdrVerdict, WedgeReason,
};
use crate::device::state::DeviceState;

fn quiescent_signals() -> EngineSignals {
    EngineSignals {
        engine_status: EngineStatusSnapshot::Stalled,
        any_dma_active: false,
        any_data_in_flight: false,
        total_dma_bytes_transferred: 0,
        total_lock_releases: 0,
        core_statuses: vec![],
        dma_states: vec![],
    }
}

#[test]
fn classify_into_wedged_then_mark_failed_then_reset_recovers() {
    let mut state = DeviceState::new_npu1();
    assert!(matches!(state.contexts[0].state, ContextState::Connected));

    // Drive the detector through enough cycles to fire Wedged{Quiescent}.
    let signals = quiescent_signals();
    let detector = &mut state.tdr_detectors[0];
    let mut fired = None;
    for _ in 0..50 {
        let v = detector.classify(&signals, None);
        if let TdrVerdict::Wedged { reason, diagnosis } = v {
            fired = Some((reason, diagnosis));
            break;
        }
    }
    let (reason, diagnosis) = fired.expect("Wedged verdict never fired");
    assert_eq!(reason, WedgeReason::Quiescent);

    // Apply the verdict to the context.
    state.contexts[0].mark_failed(reason, diagnosis);
    assert!(state.contexts[0].is_failed());
    assert!(!state.contexts[0].is_connected());

    // Reset and verify recovery.
    state.reset_context(DEFAULT_CONTEXT).expect("reset");
    assert!(state.contexts[0].is_connected());
}

#[test]
fn tier_b_and_tier_c_are_independent_paths() {
    // A workload that records a Tier B error and then wedges should leave
    // both surfaces populated, and reset clears both.
    let mut state = DeviceState::new_npu1();

    // Inject a Tier B error via the existing AsyncErrorSink API.
    // (Use whatever record_error helper the test infra has; if not,
    // construct a synthetic record directly.)
    use crate::device::async_errors::{AmdxdnaAsyncError};
    // Either: state.async_errors.record_error(...) -- pseudo-call
    // Or: directly populate the cache for the test.
    // Adapt to whatever AsyncErrorSink exposes for testing.

    // Synthesize Tier C wedge.
    use crate::device::tdr::TdrDiagnosis;
    let diag = TdrDiagnosis {
        core_states: vec![],
        dma_states: vec![],
        data_in_flight: false,
        pending_syncs: vec![],
    };
    state.contexts[0].mark_failed(WedgeReason::Quiescent, diag);
    assert!(state.contexts[0].is_failed());

    // Reset clears both.
    state.reset_context(DEFAULT_CONTEXT).expect("reset");
    assert!(state.contexts[0].is_connected());
    // Verify Tier B sink is empty after reset:
    assert!(state.async_errors.last_cache().is_none(),
            "Tier B cache should be cleared by reset");
}
```

(The Tier B injection in the second test depends on what `AsyncErrorSink` exposes. If there's no test-friendly `record_error_test_helper`, simplify the test to only verify that `reset_context` calls `async_errors.clear()` indirectly: precondition the sink in some way, observe it cleared post-reset.)

- [ ] **Step 2: Run**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu-core device::state::tests::tier_c`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/device/state/tests.rs
git commit -m "$(cat <<'EOF'
interrupt: Tier C integration tests (wedge -> failed -> reset, Tier B orthogonality)

Two integration tests at the device-state layer:

1. End-to-end wedge cycle: drive TdrDetector to Wedged{Quiescent},
   apply to context (mark_failed), verify state is Failed, reset
   to Connected.

2. Tier B/Tier C orthogonality: a context can carry a Tier B async
   error AND be Failed; reset_context clears both surfaces.

FFI-level wedge tests (constructing a wedging workload through the
run loop) are deferred to a richer test harness; the device-layer
classify path is the load-bearing thing for Tier C correctness.

Generated using Claude Code.
EOF
)"
```

---

## Task 16: Extend FFI completeness test + doc polish

**Files:**
- Create: `crates/xdna-emu-ffi/tests/tier_c_completeness.rs`
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (interrupt narrative)
- Modify: `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md` (Tier C status update)
- Create: `docs/superpowers/findings/2026-05-19-interrupt-tier-c-tdr.md`

(The existing FFI tests live as a `#[cfg(test)] mod tests` block inside `crates/xdna-emu-ffi/src/lib.rs:291` plus integration tests in `crates/xdna-emu-ffi/tests/max_cycles.rs`. Tier B Spec 1's completeness assertions went into the integration tests, so Tier C follows: new file `tests/tier_c_completeness.rs`.)

- [ ] **Step 1: Create the completeness test file**

Create `crates/xdna-emu-ffi/tests/tier_c_completeness.rs`:

```rust
//! Tier C completeness: lock the FFI surface against drift.

use xdna_emu_ffi::*;

#[test]
fn halt_reason_wedge_recovered_has_discriminant_four() {
    assert_eq!(XdnaEmuHaltReason::WedgeRecovered as u32, 4);
}

#[test]
fn context_state_discriminants_match_spec() {
    assert_eq!(XdnaEmuContextState::Connected as u32, 0);
    assert_eq!(XdnaEmuContextState::Stopped as u32, 1);
    assert_eq!(XdnaEmuContextState::Failed as u32, 2);
}

#[test]
fn reset_context_signature_takes_context_id() {
    // Type-level check: the fn pointer must accept (handle, context_id).
    type FnReset = unsafe extern "C" fn(*mut XdnaEmuHandle, u32) -> XdnaEmuResult;
    let _: FnReset = xdna_emu_reset_context;
}

#[test]
fn get_context_state_signature_matches_spec() {
    type FnGet = unsafe extern "C" fn(
        *mut XdnaEmuHandle,
        u32,
        *mut XdnaEmuContextState,
        *mut u64,
    ) -> i32;
    let _: FnGet = xdna_emu_get_context_state;
}
```

The Task 10 inline discriminant test in `completeness.rs` (if it ended up there) can be deleted now -- the canonical assertion lives here.

- [ ] **Step 2: Run**

Run: `cargo test -p xdna-emu-ffi`
Expected: all pass.

- [ ] **Step 3: Refresh `interrupt` narrative in `units.rs`**

In `crates/xdna-archspec/src/coverage/units.rs` (line ~170, the `interrupt` block), append to the existing narrative (after the Tier B mention):

```
Tier C (wedge-recovery / context-restart) is shipped: per-context state
model (Connected/Stopped/Failed) with completed_counter; device-side
TdrDetector classifying engine run state into Progressing/NaturalCompletion/
MaskPollUnsatisfied/Wedged per cycle; xdna_emu_run consumes the classifier
and returns the new XdnaEmuHaltReason::WedgeRecovered halt code on wedge,
which the plugin translates to an EIO-shaped XRT command state. Plugin
xdna_emu_reset_context resolution upgraded to fail-loud (required).
Plumbed for multi-context throughout (single ContextId(0) today, all APIs
take ContextId). Multi-context engine scheduling and lifecycle ioctls are a
separate spec. See docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md
and docs/superpowers/findings/2026-05-19-interrupt-tier-c-tdr.md.
```

- [ ] **Step 4: Create the Tier C findings note**

Create `docs/superpowers/findings/2026-05-19-interrupt-tier-c-tdr.md`:

```markdown
# Interrupt Tier C — TDR / Context-Restart (shipped)

**Status:** Shipped on `dev`.
Spec: [`../specs/2026-05-19-interrupt-tier-c-tdr-design.md`](../specs/2026-05-19-interrupt-tier-c-tdr-design.md).
Plan: [`../plans/2026-05-19-interrupt-tier-c-tdr.md`](../plans/2026-05-19-interrupt-tier-c-tdr.md).

## What landed

Per-context state model + device-side TdrDetector that classifies engine
run state per cycle. xdna_emu_run consumes the classifier; on Wedged
verdict the context transitions to Failed and the run returns
`XdnaEmuHaltReason::WedgeRecovered`. Plugin translates to an EIO-shaped
XRT command state; caller calls `xdna_emu_reset_context` (now required-
resolved, takes a context_id) before the next submission.

QuiescenceDetector and StallDetector lifted out of `src/testing/` into
`src/device/tdr/` where they belong. Single in-tree consumer
(`xclbin_suite.rs`) updated; behavior unchanged.

## Plumbed for multi-context

All APIs take `ContextId` even though `Vec<Context>` has length 1 today.
The expansion path is storage + engine scheduling, not API reshape.

## Out of scope (tracked)

- Multi-context engine scheduling + lifecycle ioctls
- `Disconnected` context state and firmware-reload semantics
- Real-clock TDR timeout cadence
- Bridge test for wedge → EIO behavior (needs deadlock-kernel fixture)
- Auto-reset-on-wedge behavioral knob (currently plugin-explicit)
```

- [ ] **Step 5: Update the Tier B findings note's Tier C reference**

In `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md`, find the line that says "Tier C (TDR / context-restart) is a further follow-up." Update to:

```
Tier C (TDR / context-restart) shipped 2026-05-19 -- see
[../findings/2026-05-19-interrupt-tier-c-tdr.md](../findings/2026-05-19-interrupt-tier-c-tdr.md).
```

- [ ] **Step 6: Run library tests one more time**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: full pass.

- [ ] **Step 7: Commit doc polish**

```bash
git add crates/xdna-archspec/src/coverage/units.rs \
        docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md \
        docs/superpowers/findings/2026-05-19-interrupt-tier-c-tdr.md \
        crates/xdna-emu-ffi/tests/tier_c_completeness.rs
git commit -m "$(cat <<'EOF'
interrupt: doc polish + completeness assertions for Tier C

Coverage narrative for the interrupt unit captures Tier C shipping.
Tier C findings note created; Tier B findings updated to point at it.
FFI completeness test extended to lock the new halt-reason discriminant
(WedgeRecovered = 4), XdnaEmuContextState discriminants, and the
xdna_emu_reset_context signature change.

Generated using Claude Code.
EOF
)"
```

---

## Plan self-review

After all 16 tasks land, dispatch a final whole-implementation reviewer (subagent) to read the full diff against the spec, confirm READY status, and surface any follow-ups beyond what's already tracked in the spec section 9.

