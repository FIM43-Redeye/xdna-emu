# Trace Level-Event Emission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the emulator's trace unit emit level-class events (LOCK_STALL, the stall family, PORT_*, DMA starvation/backpressure) as held signals -- one B..E span per assertion of the real duration -- instead of per-cycle pulses, matching real NPU1 hardware.

**Architecture:** The trace unit gains a persistent **held mask** alongside the existing per-cycle pulse mask. Each cycle the active set is `held | pulse`; a frame is committed whenever that set *changes* from the last emitted set (capturing rising AND falling edges, including a falling edge when a held bit clears with nothing else firing -- which today's `mask == 0` early-return suppresses). The trace unit owns an hw_id-keyed level/pulse classifier (the authority; `compare.rs` consumes it). Emission sites drive level assert/deassert at each condition's enter/exit edges; the per-cycle LOCK_STALL re-emission is deleted.

**Tech Stack:** Rust. Files under `src/device/trace_unit/`, `src/interpreter/`, `src/trace/`, `crates/xdna-archspec/`. Tests: `cargo test --lib` (sandbox: `TMPDIR=/tmp/claude-1000`). Fidelity calibration via `build/experiments/bcast-bridge/run_distlat_emu.sh` decoded against `trace_hw.json`.

**Calibration ground truth (in hand):**
- `vec_mul_trace_distribute_lateral` HW LOCK_STALL = **19** = 1 startup (6354ns) + 2 waits (1930ns, ...) + 16 per-transaction 1ns arb spans (9 acquire + 9 release).
- `_diag_phase_b_add_one_instrumented` HW LOCK_STALL = **46** (peano) / **40** (chess).
- HW captures: `build/experiments/bcast-bridge/trace_hw.json`; `build/bridge-test-results/20260606/_diag_phase_b_add_one_instrumented.peano.hw/events.json`.

**Reference:** spec `docs/superpowers/specs/2026-06-07-lock-stall-level-emission-design.md`; finding `docs/superpowers/findings/2026-06-07-lock-stall-overemission-interp-vs-hw.md`.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/xdna-archspec/src/aie2/trace_events.rs` | trace event metadata (the hw_id authority) | Add `is_level_event(hw_id, module)` -> bool, derived from event semantics. New. |
| `src/device/trace_unit/mod.rs` | HW-accurate trace encoder | Add `held_mask`, `last_emitted_mask`; `set_event_level(hw_id, cycle, active)`; commit-on-change. |
| `src/device/trace_unit/tests.rs` | trace unit unit tests | Held-level span tests; pulse byte-identity guard. |
| `src/trace/compare.rs` | post-decode comparison | `is_level_event(name)` consumes the archspec authority via name->hw_id. |
| `src/interpreter/execute/control.rs` | lock acquire/release exec | Drive LOCK_STALL level assert on arbitration; reconcile release emit. |
| `src/interpreter/execute/cycle_accurate.rs` | stall entry emits | LOCK_STALL/MEMORY_STALL/STREAM_STALL assert at stall entry. |
| `src/interpreter/core/interpreter.rs` | held-stall loop | Delete `LOCK_STALL_TRACE_PERIOD`; deassert at lock-acquired. |
| `src/interpreter/engine/coordinator.rs` | per-cycle drain / port + memory events | Forward level transitions; PORT_* and DMA starvation via held path. |
| `src/device/dma/engine/stepping.rs` | DMA starvation edge | Add falling-edge deassert to the `prev_starving` pattern. |

---

## Milestone 1 -- Trace-unit held-level mechanism + LOCK_STALL (keystone)

This milestone produces working, testable software on its own: LOCK_STALL emission matches HW (19 / 46). Everything after reuses this mechanism.

### Task 1: hw_id-keyed level classifier owned near the event definitions

**Files:**
- Modify/Create: `crates/xdna-archspec/src/aie2/trace_events.rs`
- Test: `crates/xdna-archspec/src/aie2/trace_events.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Locate the existing event-id definitions.** Read `crates/xdna-archspec/src/aie2/trace_events.rs` to find how core vs mem event IDs are represented (enum / constants). Confirm the hw_ids for LOCK_STALL (core 26), MEMORY_STALL (core 23), STREAM_STALL (core 24), CASCADE_STALL (core 25), PORT_RUNNING/IDLE/STALLED (core 75/74/76 + slot*4), and the mem DMA stall/starvation/backpressure ids (35/36 starvation etc.). Cross-check against `src/trace/compare.rs:350-410` (the name list) and the classification table in the finding doc.

- [ ] **Step 2: Write the failing test.**

```rust
#[cfg(test)]
mod level_classifier_tests {
    use super::*;

    #[test]
    fn level_events_classified_level() {
        // core module
        assert!(is_level_event(26, TraceModule::Core));  // LOCK_STALL
        assert!(is_level_event(23, TraceModule::Core));  // MEMORY_STALL
        assert!(is_level_event(24, TraceModule::Core));  // STREAM_STALL
        assert!(is_level_event(25, TraceModule::Core));  // CASCADE_STALL
        assert!(is_level_event(75, TraceModule::Core));  // PORT_RUNNING_0
        // mem module
        assert!(is_level_event(35, TraceModule::Mem));   // DMA_S2MM_0_STREAM_STARVATION
    }

    #[test]
    fn pulse_events_classified_pulse() {
        assert!(!is_level_event(44, TraceModule::Core));  // INSTR_LOCK_ACQUIRE_REQ
        assert!(!is_level_event(45, TraceModule::Core));  // INSTR_LOCK_RELEASE_REQ
        assert!(!is_level_event(33, TraceModule::Core));  // INSTR_EVENT_0
        assert!(!is_level_event(19, TraceModule::Mem));   // DMA_S2MM_0_START_TASK
    }
}
```

(Adapt `TraceModule` to whatever module discriminator the file already uses; if none exists, add a 2-variant enum `Core`/`Mem` -- memtile shares mem semantics for this purpose.)

- [ ] **Step 3: Run test to verify it fails.** Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec level_classifier -- --nocapture`. Expected: FAIL (`is_level_event` not found).

- [ ] **Step 4: Implement `is_level_event(hw_id, module)`.** Mirror the name list in `src/trace/compare.rs:350-410` but keyed by (hw_id, module), deriving each id from the existing constants in this file (no magic numbers -- reference the named constants). Document each LEVEL id with a one-line "why" (held signal) per the finding's table.

- [ ] **Step 5: Run test to verify it passes.** Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec level_classifier`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add crates/xdna-archspec/src/aie2/trace_events.rs
git commit -m "archspec: hw_id-keyed level-event classifier (trace-unit authority)

Generated using Claude Code."
```

### Task 2: `compare.rs` consumes the archspec classifier

**Files:**
- Modify: `src/trace/compare.rs:350-410`
- Test: existing `src/trace/compare.rs` tests + `cargo test --lib`

- [ ] **Step 1: Write the failing test** (in `compare.rs` test module): assert `is_level_event("LOCK_STALL")` and `is_level_event("DMA_S2MM_0_STREAM_STARVATION")` are true and `is_level_event("INSTR_LOCK_ACQUIRE_REQ")` is false -- after the function is rerouted. (If such a test already exists implicitly, add an explicit one.)

- [ ] **Step 2: Run to verify current behavior.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib compare::`. Expected: PASS currently (name list). This guards against regressions when rerouting.

- [ ] **Step 3: Reroute.** Replace the hardcoded `matches!` body of `is_level_event(name: &str)` with: map `name` -> (hw_id, module) using the existing name<->id mapping (see `src/trace/mod.rs` `core_event_to_hw_id` and the port/mem helpers), then call the archspec `is_level_event(hw_id, module)`. Keep the `&str` signature (callers unchanged). For names with no hw_id (e.g. `"TRUE"`), preserve current behavior with an explicit fallback.

- [ ] **Step 4: Run to verify it still passes.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected: PASS (no comparison regressions).

- [ ] **Step 5: Commit.**

```bash
git add src/trace/compare.rs
git commit -m "trace: compare.rs consumes the archspec level classifier (single source)

Generated using Claude Code."
```

### Task 3: Trace-unit held mask + commit-on-change

**Files:**
- Modify: `src/device/trace_unit/mod.rs` (struct ~139-230; `new`/`reset`; `notify_event` ~512; `commit_pending_frame` ~845; `commit_cycle` ~707)
- Test: `src/device/trace_unit/tests.rs`

- [ ] **Step 1: Write the failing test** (held level -> one span). The trace unit's byte output for a held level must be: a frame at the rising-edge cycle and a frame at the falling-edge cycle, nothing per-cycle in between. Decode via the in-tree decoder to assert ONE B..E span of the held duration.

```rust
#[test]
fn held_level_emits_single_span() {
    let mut tu = configured_core_unit_with_lock_stall_slot(); // helper: slot N = hw_id 26
    // start tracing
    tu.notify_event(START_EVENT, 0, None);
    tu.notify_event(SOME_PULSE_EVENT, 1, None); // advance to Running
    let bytes_before = tu.byte_buffer.len();
    // assert LOCK_STALL level at cycle 10, hold, deassert at cycle 1010
    tu.set_event_level(26, 10, true);
    for c in 11..1010 { tu.commit_cycle(c); }   // 999 held cycles, NO new frames
    tu.set_event_level(26, 1010, false);
    tu.flush();
    // Exactly TWO frames added for the level (rising + falling), not ~1000.
    // Decode and assert a single LOCK_STALL span [10,1010].
    let spans = decode_lock_stall_spans(&tu);   // helper in tests
    assert_eq!(spans.len(), 1, "held level must be one span, got {:?}", spans);
    assert_eq!(spans[0], (10, 1010));
    assert!(tu.byte_buffer.len() - bytes_before <= 8, "no per-cycle frames");
}
```

(Write `configured_core_unit_with_lock_stall_slot`, `decode_lock_stall_spans` as small test helpers; reuse the register-config pattern from `test_register_configuration` and the in-tree decoder `tools/trace_decoder` invoked over `tu.byte_buffer` packed into a packet, mirroring however existing tests decode -- if no Rust decode helper exists, assert on the frame bytes directly: two Single frames with the LOCK_STALL slot and deltas 10 and 1000.)

- [ ] **Step 2: Run to verify it fails.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib trace_unit::tests::held_level_emits_single_span`. Expected: FAIL (`set_event_level` not found / per-cycle frames).

- [ ] **Step 3: Add held-mask state.** In `TraceUnit` struct: add `held_mask: u8` and `last_emitted_mask: u8`, init `0` in `new()` and `reset()`. Doc-comment: held_mask = level-event slot bits currently asserted; last_emitted_mask = the active set as of the last committed frame.

- [ ] **Step 4: Add `set_event_level`.**

```rust
/// Assert (active=true) or deassert (active=false) a LEVEL event's slot.
/// Unlike notify_event (a one-cycle pulse), the bit persists in held_mask
/// until deasserted. A frame is committed when the active set changes.
pub fn set_event_level(&mut self, hw_event_id: u8, cycle: u64, active: bool) {
    if !self.configured || self.state != TraceState::Running { return; }
    let slot = match self.event_slots.iter().position(|&s| s == hw_event_id) {
        Some(i) => i as u8, None => return,
    };
    let bit = 1u8 << slot;
    let new_held = if active { self.held_mask | bit } else { self.held_mask & !bit };
    if new_held == self.held_mask { return; }     // no edge
    // Flush any pending pulse for the previous cycle first.
    if cycle != self.pending_cycle && self.pending_slot_mask != 0 {
        self.commit_pending_frame();
    }
    self.held_mask = new_held;
    self.pending_cycle = cycle;
    self.commit_active_set_change(cycle);          // emit rising/falling frame
}
```

- [ ] **Step 5: Generalize commit to active-set change.** Refactor so commit emits whenever `held | pulse` differs from `last_emitted_mask`. Add:

```rust
fn commit_active_set_change(&mut self, cycle: u64) {
    let active = self.held_mask | self.pending_slot_mask;
    if active == self.last_emitted_mask { return; }
    let delta = cycle.saturating_sub(self.last_event_cycle);
    self.last_event_cycle = cycle;
    self.pending_slot_mask = 0;
    self.last_emitted_mask = active;
    // EventTime mode: encode the new active set (held bits stay set across frames).
    if active.count_ones() == 1 { self.encode_single(active.trailing_zeros() as u8, delta); }
    else if active != 0 { self.encode_multiple(active, delta); }
    else { /* falling-to-empty: emit the slot(s) that cleared as a 0-set marker */ }
    self.try_emit_packet();
}
```

RESOLVED BY SPIKE (2026-06-07) -- do not re-derive:
- The in-tree decoder (`tools/trace_decoder/decode.py` `rebuild_perfetto_mode0` / `_emit_be`, lines 287, 327-354) is a **snapshot-mask-diff** engine: every EventCmd carries the FULL currently-asserted slot mask; the decoder diffs consecutive masks (bit 0->1 = B, 1->0 = E). A held level closes only when a later frame DROPS the bit, or at end-of-segment (`_emit_be(0, timer)`, line 396).
- Therefore: emit a frame carrying the full `held | pulse` mask **whenever it changes** from `last_emitted_mask`. Do NOT emit per-cycle. While a level is held with nothing else changing, emit NOTHING (the level stays asserted between frames).
- **Falling edge is carried by the next event's frame**, exactly as HW does it. HW emits NO synthetic empty frame: the startup LOCK_STALL (HW span 1->6355) closes precisely at the coincident acquire event whose frame's mask excludes LOCK_STALL. For LOCK_STALL this always lines up -- the wait ends when the lock is acquired, and the acquire is a traced event at that cycle. So on deassert: clear the held bit; the frame committed at the deassert cycle (carrying the now-reduced mask) closes the span via the diff.
- Do NOT synthesize a `mask == 0` frame on deassert-to-empty. HW doesn't, and `encode_multiple` asserts >= 2 bits (mask=0 isn't naturally encodable). A transition to a truly-empty active set is left to the next real frame / end-of-segment -- this does not occur for LOCK_STALL (the resuming acquire always coincides).
- **Empirical anchor:** the in-tree "ours" decoder already renders add_one HW as 46 held LOCK_STALL spans (`build/bridge-test-results/20260606/_diag_phase_b_add_one_instrumented.peano.hw/events.json`), confirming the decoder is level-diff and HW uses frame-on-transition.
- **Test consequence:** the held_level test must include a COINCIDENT closing event at the deassert cycle (mirroring stall->acquire), then assert one span `[assert_cycle, deassert_cycle]`. Do NOT test a lone held level with no closing event -- that correctly stays open to end-of-segment. Use the in-tree decoder as the oracle, or assert byte-level: one frame at the assert cycle, NO per-cycle frames during the hold, and a mask-reducing frame at the deassert cycle.
Keep `commit_pending_frame` for pulse-only paths; route held-mask changes through `commit_active_set_change`.

- [ ] **Step 6: Run to verify it passes.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib trace_unit::tests::held_level_emits_single_span`. Expected: PASS.

- [ ] **Step 7: Pulse byte-identity guard.** Add a test asserting that a sequence of pulse events (INSTR_EVENT_0 at cycles 5, 9, 14) produces byte-for-byte identical `byte_buffer` to the pre-change behavior (capture current bytes first, hardcode them). Run full `TMPDIR=/tmp/claude-1000 cargo test --lib trace_unit`. Expected: PASS (pulses unchanged).

- [ ] **Step 8: Commit.**

```bash
git add src/device/trace_unit/mod.rs src/device/trace_unit/tests.rs
git commit -m "trace_unit: held-level mask with commit-on-active-set-change

Generated using Claude Code."
```

### Task 4: Route LOCK_STALL through the held-level path; delete periodic

**Files:**
- Modify: `src/interpreter/core/interpreter.rs` (101 const; 721-743 try_resume_stall), `src/interpreter/execute/cycle_accurate.rs:818-822`, `src/interpreter/execute/control.rs:224-306, 308-368`, `src/interpreter/engine/coordinator.rs:860-872`
- Test: fidelity calibration (Task 5) plus a focused interpreter test

- [ ] **Step 1: Decide the plumbing.** The coordinator drains the EventLog and calls the trace unit (`coordinator.rs:860-872`). LOCK_STALL must become assert/deassert edges, not point events. Two recorded markers suffice: a `LockStallAssert { pc }` and `LockStallDeassert` (or reuse `EventType::LockStall` with an `active: bool`). Add the variant(s) to `EventType` (`src/interpreter/state/event_trace.rs:63`) and map them in the coordinator drain to `trace_unit.set_event_level(26, cycle, active)` instead of `notify_event`.

- [ ] **Step 2: Write the failing interpreter test.** A minimal kernel/sequence: one immediate lock acquire (free lock) then one acquire that waits K cycles then succeeds. Assert the recorded level edges are: assert+deassert (1-cycle arb) for the first, assert at wait-entry + deassert at acquire (K-cycle span) for the second -- and NO per-cycle LockStall records. (Use the EventLog inspection pattern from `cycle_accurate.rs:1037`.)

- [ ] **Step 3: Delete periodic re-emission.** Remove `LOCK_STALL_TRACE_PERIOD` (`interpreter.rs:101`) and the `lock_stall_periodic` block (`interpreter.rs:729-743`). In `try_resume_stall` success (`interpreter.rs:721-726`), record `LockStallDeassert` at the resume cycle.

- [ ] **Step 4: Assert at the edges.**
  - `cycle_accurate.rs:818-822` (WaitLock entry): record `LockStallAssert { pc }` (rising edge of the wait).
  - `control.rs:308-368` (release): replace the point `LockStall` with assert-then-deassert across the arbitration cycle (1-cycle span), keeping `InstrLockReleaseReq` as the pulse.
  - `control.rs:224-306` (acquire success, the immediate case): add assert-then-deassert across the arbitration cycle so a free acquire yields a 1ns arb span (matches HW's 16 short-arb spans). Ensure a *contended* acquire (becomes WaitLock) records ONE assert (at entry) and ONE deassert (at acquire), not entry-pulse + success-pulse.

- [ ] **Step 5: Map edges in coordinator drain** (`coordinator.rs:860-872`): for `LockStallAssert`/`LockStallDeassert` call `set_event_level(26, total_cycle, true/false)`; keep all other events on `notify_event`.

- [ ] **Step 6: Run interpreter test.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected: PASS; no other `--lib` regressions.

- [ ] **Step 7: Commit.**

```bash
git add src/interpreter/ 
git commit -m "interpreter: LOCK_STALL as held level (assert/deassert), drop per-cycle period

Generated using Claude Code."
```

### Task 5: LOCK_STALL HW calibration (the fidelity gate)

**Files:**
- Use: `build/experiments/bcast-bridge/run_distlat_emu.sh`, `trace_hw.json`
- Build: `cargo build -p xdna-emu-ffi` (refresh the .so the bridge loads)

- [ ] **Step 1: Rebuild the FFI lib.** Run: `nice -n 19 cargo build -p xdna-emu-ffi`. Expected: Finished.

- [ ] **Step 2: Run EMU-side trace.** Run: `bash build/experiments/bcast-bridge/run_distlat_emu.sh > /tmp/claude-1000/run_emu_m1.log 2>&1`. Read the log. Expected: PASS, trace_emu.json regenerated.

- [ ] **Step 3: Diff LOCK_STALL count + decomposition vs HW.** Run the count+span script (the one in the session: counts B-events per name and span durations) over `trace_emu.json` vs `trace_hw.json`. Expected: LOCK_STALL = **19**, with the 2 long spans at ~6354 / ~1930 ns (not 1ns) and 16 ~1ns arb spans. INSTR_*/EVENT_*/ACQ/REL unchanged.

- [ ] **Step 4: Iterate** Step 4 of Task 4 / Step 5 of Task 3 until count AND decomposition match. If the immediate-acquire arb span count is off, adjust whether a free acquire emits an arb span (the 16-vs-18 question -- HW shows one per traced transaction).

- [ ] **Step 5: add_one cross-check (optional, HW capture already exists).** Decode `build/bridge-test-results/20260606/_diag_phase_b_add_one_instrumented.peano.hw/events.json` (HW=46) and compare to an EMU add_one run if cheaply available; else note the target for the bridge sweep.

- [ ] **Step 6: Full regression.** Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected: PASS. Verify against baseline trace.log per the team rule (a "CLEAN" verdict can miss level-event duration regressions).

- [ ] **Step 7: Commit** (docs/calibration notes only if any).

```bash
git add -A
git commit -m "calibrate: LOCK_STALL EMU trace matches HW (19 distribute_lateral)

Generated using Claude Code."
```

---

## Milestone 2 -- Core stall family (MEMORY_STALL, STREAM_STALL, CASCADE_STALL)

Same shape as LOCK_STALL: these are level events currently emitted as per-cycle/point pulses. Reuse the Task 3 mechanism and the Task 4 edge pattern.

### Task 6: MEMORY_STALL / STREAM_STALL / CASCADE_STALL as held levels

**Files:**
- Modify: `src/interpreter/execute/cycle_accurate.rs` (241 MemoryStall; 677/690/826 StreamStall), `src/interpreter/engine/coordinator.rs:1162-1172` (MemoryStall), `src/interpreter/state/event_trace.rs` (variants), coordinator drain map.

- [ ] **Step 1: Audit the emission sites.** Read `cycle_accurate.rs:235-250` (MemoryStall) and `:660-700, :818-827` (StreamStall) to see the current point emission and the stall enter/exit conditions. Find where each stall *resolves* (the deassert edge). CASCADE_STALL: locate its emission (grep `Cascade`); if not yet emitted, note as out-of-scope-but-classified.
- [ ] **Step 2: Failing test** -- a sequence that holds a memory stall K cycles asserts one K-span (EventLog inspection).
- [ ] **Step 3: Implement** assert at stall entry, deassert at resolution, for each; route through `set_event_level(hw_id, ...)` in the coordinator drain (hw_id 23/24/25).
- [ ] **Step 4: Run** `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected PASS.
- [ ] **Step 5: Fidelity** -- pick a kernel that exercises memory/stream stalls (audit the bridge corpus; e.g. a memtile or shared-buffer kernel), run EMU-side, diff vs an HW capture; target proper span counts/durations.
- [ ] **Step 6: Commit** `interpreter: MEMORY/STREAM/CASCADE_STALL as held levels`.

---

## Milestone 3 -- Stream-port events (PORT_RUNNING / PORT_IDLE / PORT_STALLED)

These already emit via the coordinator's per-cycle port evaluation (`coordinator.rs:1031-1069`). Re-express through the held path so the mechanism is uniform. NOTE: the residual count gap (PORT_RUNNING 1 vs 6) is DMA bursty-delivery timing, NOT this change -- target emission *shape* (proper spans), document the count residual as expected.

### Task 7: PORT_* through the held path

**Files:** `src/interpreter/engine/coordinator.rs:1031-1069`, trace unit (no change -- reuse held mask).

- [ ] **Step 1: Audit** `coordinator.rs:1025-1075` -- how PortRunning/Idle/Stalled/Tlast are currently pushed (`port_events`) and forwarded to the trace unit. Determine if they currently pulse per cycle or transition.
- [ ] **Step 2: Failing test** -- a port held running K cycles -> one span (coordinator-level or trace-unit-level test).
- [ ] **Step 3: Implement** -- forward PORT_RUNNING/IDLE/STALLED as `set_event_level(hw_id, cycle, active)` on their transitions; keep PORT_TLAST as a pulse (`notify_event`). Use `core_port_running_hw_id` / `memtile_port_running_hw_id` / `shim_port_running_hw_id` (already in `coordinator.rs:1031-1034`).
- [ ] **Step 4: Run** `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected PASS.
- [ ] **Step 5: Fidelity** -- distribute_lateral: PORT_RUNNING span *shape* correct; document the count residual (1 vs 6) as DMA-timing-bound (link the spec's Scope honesty section).
- [ ] **Step 6: Commit** `coordinator: PORT_* through held-level path (shape; count residual is DMA-timing)`.

---

## Milestone 4 -- DMA stall / starvation / backpressure

The DMA engine already edge-fires the rising edge (`stepping.rs:438-444`, `prev_starving`). Add the falling edge so the trace unit closes the span; generalize the `prev_*` pattern to the stalled-lock / backpressure / memory-starvation conditions.

### Task 8: DMA level events emit both edges

**Files:** `src/device/dma/engine/stepping.rs:410-449`, `src/device/dma/channel.rs:250-270`, coordinator forward.

- [ ] **Step 1: Audit** `stepping.rs:410-449` (the `prev_starving` edge) and `channel.rs:250-270` (the starvation-event comment) -- find where starvation/stall *ends* (data arrives / lock acquired / backpressure clears).
- [ ] **Step 2: Failing test** -- a channel starved K cycles -> one starvation span (DMA engine test).
- [ ] **Step 3: Implement** -- where `prev_starving` falls true->false, fire the deassert; route DMA level events (hw_id 35/36 starvation, 31-34 stalled_lock, 37-42 backpressure/mem-starvation) through `set_event_level`. Generalize `prev_starving` into per-condition `prev_*` flags.
- [ ] **Step 4: Run** `TMPDIR=/tmp/claude-1000 cargo test --lib`. Expected PASS.
- [ ] **Step 5: Fidelity** -- distribute_lateral STREAM_STARVATION span shape; document count residual (1 vs 2) as DMA-timing-bound.
- [ ] **Step 6: Commit** `dma: stream starvation/stall/backpressure as held levels (both edges)`.

---

## Milestone 5 -- Close-out

### Task 9: Full validation + docs

- [ ] **Step 1:** Rebuild `cargo build` + `cargo build -p xdna-emu-ffi`. Run `TMPDIR=/tmp/claude-1000 cargo test --lib` -- all pass.
- [ ] **Step 2:** Re-run distribute_lateral + add_one EMU-side; confirm LOCK_STALL 19 / 46 and the stall-family spans correct; PORT_*/starvation shape correct with residuals documented.
- [ ] **Step 3:** Run the broader trace test suite / a bridge `--no-hw` pass over trace kernels; verify against baseline trace.log.
- [ ] **Step 4:** Update `docs/known-fidelity-gaps.md`: LOCK_STALL row -> closed (EMU now edge-emits, matches HW); note PORT_RUNNING/starvation residual as DMA-timing (cross-ref DDR-sim axis). Update the finding doc status to implemented.
- [ ] **Step 5:** Commit docs; the feature is "done" only when every level class is migrated and passing (finish-what-you-start).

---

## Self-review notes
- **Spec coverage:** mechanism (M1 T3), classification ownership + compare consumer (T1/T2), LOCK_STALL (T4/T5), stall family (M2), PORT_* (M3), DMA (M4), scope-honesty residuals documented (M3/M4/M5), testing two-tier (mechanism + fidelity) -- all present.
- **Known soft spots** (resolve at execution, against real code, NOT placeholders to skip): (a) the exact mode-0 falling-edge byte encoding -- the held_level test is the oracle (T3 S5); (b) whether a free acquire emits an arb span -- calibration decides (T5 S4); (c) CASCADE_STALL may have no current emission site (T6 S1). Each is a concrete decision a worker makes by reading code + running the calibration, not a hand-wave.
- **Off-by-one** (12298 vs 12297) is out of scope; do not chase.
