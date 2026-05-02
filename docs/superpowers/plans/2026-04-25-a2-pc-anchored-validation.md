# A.2: PC-Anchored Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace today's cycle-anchored mode-0 trace sweep with a PC-anchored mode-1 sweep, using performance-counter overflow events as a deterministic in-batch cycle clock, so HW vs EMU comparison becomes "does the same event fire at the same set of PCs?" — a structural correctness signal that survives HW jitter.

**Architecture:** Three EMU pieces (perfcnt→trace wiring, mode-1 byte encoder, PC threading from coordinator) land first; then three Python tool extensions (injector, multi-tile patcher, sweep orchestrator); then two Rust compare-side pieces (`compare_pc_anchored` + sweep aggregation); finally an integration gate on `add_one`.

**Tech Stack:**
- Rust (`src/device/trace_unit/`, `src/device/tile/`, `src/interpreter/engine/coordinator.rs`, `src/trace/compare.rs`) — encoder, PC threading, comparator
- `src/device/perf_counters/` — already implemented; needs wiring only (no new module)
- Python (`tools/mlir-trace-inject.py`, `tools/trace-patch-events.py`, `tools/trace-sweep.py`) — IR injection, multi-tile patch, sweep orchestration
- mlir-aie declarative trace ops (`aie.trace`, `aie.trace.config`, `aie.trace.reg`, `aie.trace.mode`) — no upstream dialect changes needed
- Fixtures: `tools/trace_decoder/fixtures/mode1_*` — byte-level oracles

**Source spec:** [`docs/superpowers/specs/2026-04-25-a2-pc-anchored-validation-design.md`](../specs/2026-04-25-a2-pc-anchored-validation-design.md). Read the spec for design rationale; the plan is self-contained for execution.

---

> **Sweep-as-of 2026-05-01:** Plan completed. Tasks 1-9 landed via the gate-A.2 commit family: 51fdb02 (mode-1 lockstep sweep + mode-2 baseline + manifest), d9069a6 (PC-anchored set/multiset diff + perfcnt cycle bands), 61a8cdd (aggregate PC-anchored reports), 6b7fbb2 (Task 9 integration gate + --trace=pc-anchored), 684ba21 (code review followups), 8129808 (housekeeping). Closing marker: 97a9edf `docs(next-steps): D.8 master merge complete, A.2 landed, D.3 still open`. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Branching Strategy

All work lands on `xdna-emu` `dev`. No worktree — A.2 builds on the spec already committed there (89b8110), and the prior threads (A.1, A.5, C-thread, D.x) are also on `dev`. The injection-side mlir-aie work uses the existing `aie.trace.reg` primitive; no mlir-aie patch needed for this plan. (The sugar PR for `aie.trace.perf_counter` is an optional follow-up tracked in the spec.)

---

## Context for Engineers

**The problem.** Mode-0 traces record `(slot, cycle_delta)` per event. Real silicon does not run two consecutive batches with bit-identical cycle traces, so cross-batch cycle anchors are noisy. Worse, when HW and EMU disagree on which events fire at all (cascade_flows finding, A.1), `max(ts) - min(ts)` measures different sets and the resulting drift ratio is meaningless. A.2 reframes joining around PCs (kernel-state, jitter-invariant) and uses perfcnt overflow as a deterministic in-batch cycle clock.

**Mode 1 semantics.** Per `aie_registers_aie2.json`, only the core module's `Trace_Control0` has a `Mode` bitfield; memmod, memtile, shim variants have only Trace_Start_Event/Trace_Stop_Event. Mode 1 is therefore *core-only*. Memmod/memtile/shim continue in mode 0 in this sweep; the EMU enforces this at config-time.

**Mode-1 frame format.** Per `tools/trace_decoder/modes/mode1.py:109-125`, the EventPC opcode is 4 bytes:

```
byte0 = 1100 01ee  (top 6 bits opcode 0x31; low 2 bits = mask high 2)
byte1 = eeeeeerr   (mask low 6 bits + 2 reserved bits)
byte2 = rrpppppp   (2 reserved bits + PC high 6 bits)
byte3 = pppppppp   (PC low 8 bits)
```

8-bit event-slot mask + 14-bit PC. The decoder's reverse mask/PC extraction is the byte-equivalence oracle.

**Perfcnt is already implemented.** `src/device/perf_counters/mod.rs` (the spec mentions `src/device/tile/perfcnt.rs` — that's a path from an earlier draft; the actual location is `perf_counters/`). All four module types instantiate correctly per regdb counts (compute=4 core/2 mem; memtile=0 core/4 mem; shim=2 core/0 mem). `PerfCounterBank::tick_active_cycles()` and `tick_idle_cycles()` return `Vec<usize>` of indices that crossed threshold this cycle. The load-bearing gap: `coordinator.rs:1027-1038` discards those returns, so PERF_CNT_N events never propagate to trace units. Task 1 closes that gap; we do *not* re-implement perfcnt.

**Event ID truth.** Per `aie-rt/driver/src/events/xaie_events_aieml.h`, `PERF_CNT_N` is event id `5+N` consistently across core (5..8), memmod (5..6), shim/PL (5..6), memtile (5..8). Generated tables at `crates/xdna-archspec/src/aie2/trace_events.rs` re-export these as `core_events::PERF_CNT_0` etc.

**PC source.** `EventType` variants for instruction-class events (`InstrVector { pc, .. }`, `InstrLoad { pc, .. }`, `InstrLockAcquireReq { pc, .. }`, `InstrEvent { pc, id }`, etc.) already carry the PC where the core retired the instruction. Stall variants (`MemoryStall { cycles }`, `LockStall { cycles }`, `StreamStall { cycles }`) and core-state variants (`CoreActive`, `CoreDisabled`) do not. Memory-side events (DMA, lock release) come through `drain_mem_trace_events()` and are PC-less by construction. Plan: thread `Some(pc)` for instruction-class variants; pass `None` for stalls and memory-side events. The encoder writes `pc=0` as a sentinel for `None` and emits a rate-limited warning.

**Tile event-fire integration points.** Per `src/device/tile/mod.rs:562-583`: `notify_core_trace_event(hw_id, cycle)` and `notify_mem_trace_event(hw_id, cycle)` forward to the trace unit and the edge detectors. These are the only entry points for events into trace units. Task 3 extends both with a `pc: Option<u32>` parameter; mode-0 ignores it.

---

## File Structure

### Modified files (Rust)

| File | Responsibility | Roughly what changes |
|------|----------------|----------------------|
| `src/device/trace_unit/mod.rs` | Trace unit state, byte encoding | Add `EventPc` frame encoder, `mode_supports_pc()`, mode-set guard, extend `notify_event` with `pc: Option<u32>` |
| `src/device/trace_unit/tests.rs` | Encoder unit tests | Add mode-1 byte-equivalence + sentinel + non-core-panic cases |
| `src/device/tile/mod.rs` | Tile struct, event dispatch | Extend `notify_core_trace_event` and `notify_mem_trace_event` with `pc: Option<u32>` |
| `src/device/tile/tests.rs` | Tile dispatch tests | Update existing call sites to pass `None`; add mode-1 PC-thread test |
| `src/device/state/effects.rs` | Event effect application | Update both notify call sites (line 377-378) |
| `src/interpreter/engine/coordinator.rs` | Main step loop | Capture perfcnt firings + emit PERF_CNT_N events; thread PC from `EventType` to trace via new helper `event_pc(evt) -> Option<u32>` |
| `src/trace/mod.rs` | Hw-id mapping | Add `event_pc()` helper that pulls PC from `EventType` variants |
| `src/trace/compare.rs` | HW/EMU comparator | Add `compare_pc_anchored`, `PCAnchoredReport`, cycle-band interpolation; extend `BatchResult`; aggregate in `compare_sweep_dir_with_opts`; new report sections |
| `src/bin/trace_compare.rs` | Comparator CLI | Add `--pc-anchored` flag → routes to new code path |

### Modified files (Python)

| File | Responsibility | Roughly what changes |
|------|----------------|----------------------|
| `tools/mlir-trace-inject.py` | MLIR injection | Add `--trace-mode`, per-module-type `--{core,memmod,memtile,shim}-grounding/-sweep-events`, `--perfcnt-period`; emit `aie.trace.config @perf_<type>_<col>_<row>` blocks via `aie.trace.reg`; warn on `--trace-mode event_pc` with non-core sweeps |
| `tools/trace-patch-events.py` | Per-batch patcher | Add `--multi-tile <json>` reading a list of patch specs and applying them in one process invocation |
| `tools/trace-sweep.py` | Sweep orchestrator | Add `--mode`, per-tile cursor lockstep, mode-2 finishing batch, grounding-PC consistency check; emit `sweep-manifest.json` |
| `tools/test_trace_inject.py` | Injector tests | Add mode-1 + perfcnt config round-trip cases |
| `tools/test_trace_sweep.py` | Sweep tests | Add multi-cursor lockstep, exhaustion, mode-2 finishing, drift-detection cases |

### New files

None. All work extends existing files. (The spec mentioned a `src/device/tile/perfcnt.rs` new module; that's stale — `perf_counters/` already exists.)

---

## Tasks

Numbering matches the spec's "Implementation order" sketch.

---

### Task 1: Wire perfcnt threshold firings into trace unit and event system

**Goal:** Close the load-bearing gap where `coordinator.rs` discards the `Vec<usize>` returned by `PerfCounterBank::tick_*_cycles()`. After this task, every cycle a counter overflows, `PERF_CNT_N` (id=5+N) lands as a `notify_*_trace_event(...)` call on the owning module's trace unit and is also fed back to `handle_event()` so self-reset configurations work.

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` (the `tick` block at lines ~1010-1038)
- Modify: `src/device/perf_counters/tests.rs` (add wiring-level test if one doesn't already exist)

#### Step 1.1 — Write the failing wiring test

- [x] Open `src/device/perf_counters/tests.rs` and add a unit test that:
  - Creates a `PerfCounterBank::new(4)`, configures counter 0 with `event_value=10` and `start_event=1` (TRUE — always firing).
  - Calls `handle_event(1)` to start.
  - Calls `tick_active_cycles()` 10 times.
  - Asserts the 10th call returns `vec![0]`.
  - Asserts subsequent ticks (without re-arming) return empty vec — counter stays at threshold and does not re-fire.

```rust
#[test]
fn threshold_fires_once_at_event_value() {
    let mut bank = PerfCounterBank::new(4);
    // Configure cnt0: start on TRUE (event 1), threshold 10, no reset.
    bank.write_event_value(0, 10);
    let ctrl0 = (1u32) | (0u32 << 8); // start=1, stop=0
    bank.write_control_start_stop(ctrl0, 0, 1, 7);
    bank.handle_event(1); // start

    for cycle in 1..10 {
        let fired = bank.tick_active_cycles();
        assert!(fired.is_empty(), "cycle {} fired prematurely: {:?}", cycle, fired);
    }
    let fired = bank.tick_active_cycles();
    assert_eq!(fired, vec![0], "cycle 10 should fire counter 0");
    let fired = bank.tick_active_cycles();
    assert!(fired.is_empty(), "cycle 11 should not re-fire (no reset)");
}
```

- [x] Run: `cargo test --lib -p xdna-emu perf_counters::tests::threshold_fires_once_at_event_value -- --exact`
- [x] Expected: PASS (this exercises pre-existing logic; if it fails, fix `perf_counters/mod.rs` first before continuing).

#### Step 1.2 — Write a coordinator-level test for perfcnt → trace wiring

- [x] Create a new test file or extend an existing coordinator integration test (e.g. add to `src/interpreter/engine/coordinator/tests.rs` if one exists, otherwise add to `tests/` as a small integration fixture). The test:
  1. Builds a minimal device with one compute tile.
  2. Configures `tile.core_perf_counters` counter 0 with start_event=TRUE, threshold=5, reset_event=PERF_CNT_0 (5) — self-reset for free-run.
  3. Configures `tile.core_trace` to record event id 5 (PERF_CNT_0) in slot 0, mode=EventTime, start_event=TRUE.
  4. Steps the coordinator for 20 cycles.
  5. Asserts the trace unit recorded ≥3 frames (one every 5 cycles).
  6. Asserts the trace bytes contain Single0/1/2 frames matching cycles 5, 10, 15.

```rust
#[test]
fn perfcnt_overflow_emits_trace_event() {
    let mut harness = test_helpers::single_compute_tile_harness();
    let tile = harness.tile_mut(0, 2);

    // perfcnt cnt0: start=TRUE(1), threshold=5, reset=PERF_CNT_0(5) for free-run
    tile.core_perf_counters.write_event_value(0, 5);
    let ctrl0 = 1u32; // start=TRUE on counter 0
    tile.core_perf_counters.write_control_start_stop(ctrl0, 0, 1, 7);
    tile.core_perf_counters.write_control_reset(5, 7); // reset=PERF_CNT_0
    tile.core_perf_counters.handle_event(1); // arm

    // trace: slot 0 = PERF_CNT_0, start_event=TRUE, mode=EventTime
    tile.core_trace.set_event_slots([5,0,0,0,0,0,0,0]);
    tile.core_trace.set_mode(0); // EventTime
    tile.core_trace.set_start_event(1);

    for _ in 0..20 { harness.step(); }

    let bytes = tile.core_trace.byte_buffer_for_test();
    let frames = decode_mode0(bytes);
    let perf_frames: Vec<_> = frames.iter()
        .filter(|f| f.slot_mask & 0b1 != 0)
        .collect();
    assert!(perf_frames.len() >= 3, "expected >=3 PERF_CNT_0 frames, got {}: {:?}",
        perf_frames.len(), frames);
}
```

- [x] Note: `single_compute_tile_harness`, `byte_buffer_for_test`, `decode_mode0` may need small helper additions in test scaffolding. Add what's missing as part of this step.
- [x] Run: `cargo test --lib coordinator perfcnt_overflow_emits_trace_event -- --exact`
- [x] Expected: FAIL (perf firings are still discarded in coordinator).

#### Step 1.3 — Implement the wiring

- [x] In `src/interpreter/engine/coordinator.rs`, replace the existing tick block (~1027-1038) with a version that captures the returned `Vec<usize>` and emits PERF_CNT_N events:

```rust
// Phase 3e: Tick tile timers and performance counters; route firings.
//
// Per aie-rt xaie_events_aieml.h, PERF_CNT_N hw event id is 5+N in every
// module type (core 5..8, memmod 5..6, memtile 5..8, shim/PL 5..6).
// We emit each firing both to the trace unit (so it can be sampled) and
// back through handle_event (so a self-resetting config recycles).
const PERF_CNT_BASE: u8 = 5;
let cycle = self.total_cycles;
for (i, tile) in self.device.array.tiles.iter_mut().enumerate() {
    tile.core_timer.tick();
    tile.mem_timer.tick();

    let core_active = self.cores.get(i).map_or(false, |c| c.active_this_cycle);
    let core_fired = if core_active {
        tile.core_perf_counters.tick_active_cycles()
    } else {
        tile.core_perf_counters.tick_idle_cycles()
    };
    for cnt_idx in core_fired {
        let hw_id = PERF_CNT_BASE + cnt_idx as u8;
        // Feed back so self-reset configs work, then trace-notify with no PC.
        tile.core_perf_counters.handle_event(hw_id);
        tile.notify_core_trace_event(hw_id, cycle, None);
    }

    let mem_fired = tile.mem_perf_counters.tick_active_cycles();
    for cnt_idx in mem_fired {
        let hw_id = PERF_CNT_BASE + cnt_idx as u8;
        tile.mem_perf_counters.handle_event(hw_id);
        tile.notify_mem_trace_event(hw_id, cycle, None);
    }
}
```

(Note: `notify_core_trace_event` / `notify_mem_trace_event` get their `Option<u32>` PC parameter in Task 3. To keep this task self-contained without touching every call site, keep the existing two-arg signature for now and add the PC parameter in Task 3 across all call sites. Replace the `, None` in this snippet with nothing for Task 1.)

- [x] Adjusted Task-1 snippet (no PC param yet):

```rust
        tile.notify_core_trace_event(hw_id, cycle);
        // ...
        tile.notify_mem_trace_event(hw_id, cycle);
```

- [x] Run: `cargo test --lib coordinator perfcnt_overflow_emits_trace_event -- --exact`
- [x] Expected: PASS.

#### Step 1.4 — Full lib test sanity

- [x] Run: `cargo test --lib`
- [x] Expected: PASS, no regressions vs prior 2755+/5-ignored baseline. Investigate any new failures before continuing.

#### Step 1.5 — Commit

```bash
git add src/interpreter/engine/coordinator.rs src/device/perf_counters/tests.rs \
        src/interpreter/engine/coordinator/  # if test scaffolding added
git commit -m "feat(perfcnt): route threshold firings to trace unit and self-reset

PerfCounterBank::tick_*_cycles() returned a Vec<usize> of indices that
hit threshold this cycle, but coordinator discarded it. Capture the
return, look up PERF_CNT_N hw event ids (base 5 + counter index per
xaie_events_aieml.h), feed back through handle_event so self-resetting
configs recycle, and notify the corresponding trace unit. Closes the
load-bearing gap for A.2 PC-anchored validation."
```

---

### Task 2: Mode-1 (EventPC) trace encoder for core trace units

**Goal:** Extend `TraceUnit` so that when `mode == EventPc`, `notify_event(hw_id, cycle, pc)` accumulates `(slot_mask, pc)` and `commit_pending_frame` emits a 4-byte EventPC frame matching `tools/trace_decoder/modes/mode1.py`. Mode-1 is enforced core-only at config-time. Cycle deltas are NOT emitted in mode 1; the previous mode-0 `last_event_cycle` machinery is bypassed when in mode 1.

**Files:**
- Modify: `src/device/trace_unit/mod.rs`
- Modify: `src/device/trace_unit/tests.rs`

#### Step 2.1 — Write the encoder fixture-equivalence test

- [x] In `src/device/trace_unit/tests.rs`, add a test that drives a synthetic event sequence through a mode-1 core trace_unit and asserts the resulting bytes round-trip cleanly through `tools/trace_decoder/modes/mode1.py`. Use the existing `mode1_mixed_r0_core_expected.json` as the oracle — pick a small subset (e.g., the first ~10 EventPC entries) and reconstruct an equivalent test driver.

```rust
#[test]
fn mode1_encoder_byte_equivalent_to_decoder_fixture() {
    let mut tu = TraceUnit::new(0, 2);
    // Configure: mode=EventPc, slot 3 = LOCK_STALL (hw_id 26), start_event=TRUE.
    tu.set_mode(TraceMode::EventPc);
    tu.set_packet_type(0); // core
    tu.set_event_slot(3, 26);
    tu.set_start_event(1);

    // Synthetic drive: feed events at known PCs.
    tu.notify_event(1, 0, None);          // start
    tu.notify_event(26, 0, Some(816));    // EventPC: mask=0b1000 (slot 3), pc=816
    // Different cycle, same PC, same slot:
    tu.notify_event(26, 1, Some(816));    // committed in next cycle's frame
    tu.commit_cycle(1);

    let bytes = tu.byte_buffer_for_test();

    // Verify by feeding back through the in-tree decoder fixture format:
    // expected pattern is 8 bytes for Start + 4 bytes per EventPC frame.
    // Decode and compare structurally.
    let decoded = run_mode1_decoder(bytes);
    assert_eq!(decoded.len(), 3); // Start + 2x EventPC
    assert!(matches!(decoded[0], TraceCommand::Start { .. }));
    assert!(matches!(
        decoded[1],
        TraceCommand::EventPc { mask: 0b1000, pc: 816 }
    ));
    assert!(matches!(
        decoded[2],
        TraceCommand::EventPc { mask: 0b1000, pc: 816 }
    ));
}
```

- [x] `run_mode1_decoder` invokes the in-tree Python decoder via `Command::new("python3")` against the bytes — or better, port the small mode-1 byte parsing logic into a Rust test helper so the test runs without Python (4 byte read, mask split, PC split). Implement as a private helper in the test module.
- [x] Run: `cargo test --lib trace_unit::tests::mode1_encoder_byte_equivalent_to_decoder_fixture -- --exact`
- [x] Expected: FAIL (mode-1 encoder not implemented; falls through to mode-0 path or panics).

#### Step 2.2 — Add `mode_supports_pc` and config-time guard

- [x] In `src/device/trace_unit/mod.rs`, add a `pkt_type` field if not already present (it's needed to distinguish core from memmod/memtile/shim trace units; the trace unit has `packet_type` in Trace_Control1 — verify by grep). Add:

```rust
impl TraceUnit {
    /// True only when this trace unit is configured as a core-module
    /// packet type and may legitimately enter EventPc mode.
    pub fn mode_supports_pc(&self) -> bool {
        // packet_type 0 = core, 1 = memmod, 2 = shim/PL, 3 = memtile.
        // Per regdb, only core's Trace_Control0 has a Mode bitfield;
        // setting EventPc on others is a HW-impossible state.
        self.packet_type == 0
    }
}
```

- [x] Modify `set_mode` (or wherever Trace_Control0 mode bits are written; grep for the mode-write site). On a non-zero mode write to a non-core trace unit, log an error and clamp to `EventTime`:

```rust
pub fn set_mode_from_register(&mut self, raw_mode: u32) {
    let new_mode = TraceMode::from_u32(raw_mode);
    if matches!(new_mode, TraceMode::EventPc | TraceMode::Execution)
        && !self.mode_supports_pc()
    {
        log::error!(
            "TraceUnit ({},{}) pkt_type={}: mode={:?} requires core module; \
             clamping to EventTime (regdb: only core has Trace_Control0.Mode)",
            self.col, self.row, self.packet_type, new_mode
        );
        self.mode = TraceMode::EventTime;
        return;
    }
    self.mode = new_mode;
}
```

- [x] Add a unit test:

```rust
#[test]
fn mode1_on_non_core_trace_unit_clamps_to_event_time() {
    let mut tu = TraceUnit::new(0, 1);
    tu.set_packet_type(3); // memtile
    tu.set_mode_from_register(1); // try EventPc
    assert_eq!(tu.mode(), TraceMode::EventTime,
        "memtile trace unit must reject EventPc mode");
}
```

- [x] Run: `cargo test --lib trace_unit -- mode1_on_non_core` ; expected PASS once `set_mode_from_register` is in place.

#### Step 2.3 — Implement the EventPC frame encoder

- [x] Add to `TraceUnit`:

```rust
/// Encode a 4-byte EventPC frame: 8b mask + 14b PC.
///
/// Layout (MSB-first), per tools/trace_decoder/modes/mode1.py:
///   byte0 = 0b110001_ee  (top 6 bits = 0x31; low 2 bits = mask high 2)
///   byte1 = eeeeee_rr    (mask low 6 bits + 2 reserved bits = 0)
///   byte2 = rr_pppppp    (2 reserved bits = 0 + PC high 6 bits)
///   byte3 = pppppppp     (PC low 8 bits)
fn encode_event_pc(&mut self, mask: u8, pc: u16) {
    debug_assert!(pc < (1 << 14), "PC {} exceeds 14-bit range", pc);
    let byte0 = 0b1100_0100 | ((mask >> 6) & 0b11);
    let byte1 = ((mask & 0b0011_1111) << 2);
    let byte2 = ((pc >> 8) as u8) & 0b0011_1111;
    let byte3 = (pc & 0xFF) as u8;
    self.byte_buffer.push(byte0);
    self.byte_buffer.push(byte1);
    self.byte_buffer.push(byte2);
    self.byte_buffer.push(byte3);
}
```

#### Step 2.4 — Branch `commit_pending_frame` on mode

- [x] Modify `commit_pending_frame` to dispatch:

```rust
fn commit_pending_frame(&mut self) {
    let mask = self.pending_slot_mask;
    if mask == 0 {
        return;
    }
    self.pending_slot_mask = 0;

    match self.mode {
        TraceMode::EventTime => {
            let delta = self.pending_cycle.saturating_sub(self.last_event_cycle);
            self.last_event_cycle = self.pending_cycle;
            if mask.count_ones() == 1 {
                let slot = mask.trailing_zeros() as u8;
                self.encode_single(slot, delta);
            } else {
                self.encode_multiple(mask, delta);
            }
        }
        TraceMode::EventPc => {
            // Truncate PC to 14 bits per HW; warn (rate-limited) if upper bits set.
            let pc14 = (self.pending_pc & 0x3FFF) as u16;
            if (self.pending_pc as u32) > 0x3FFF && self.pc_truncate_warnings < 4 {
                log::warn!(
                    "TraceUnit ({},{}): PC 0x{:X} truncated to 14 bits (0x{:X})",
                    self.col, self.row, self.pending_pc, pc14
                );
                self.pc_truncate_warnings += 1;
            }
            self.encode_event_pc(mask, pc14);
        }
        TraceMode::Execution | TraceMode::Reserved => {
            // Mode 2 not implemented in EMU per spec; mode 3 is reserved.
            // Skip emission rather than corrupt the stream.
        }
    }
    self.try_emit_packet();
}
```

- [x] Add `pending_pc: u32` and `pc_truncate_warnings: u32` fields to `TraceUnit` struct, initialized to 0 in `new`.

#### Step 2.5 — Extend `notify_event` with `pc: Option<u32>` (encoder-side only)

- [x] Change the signature of `TraceUnit::notify_event`:

```rust
pub fn notify_event(&mut self, hw_event_id: u8, cycle: u64, pc: Option<u32>) {
    // ... unchanged start/stop handling ...

    // Mask accumulation (unchanged):
    self.pending_cycle = cycle;
    self.pending_slot_mask |= 1 << slot;

    // PC tracking for mode 1:
    if matches!(self.mode, TraceMode::EventPc) {
        match pc {
            Some(p) => self.pending_pc = p,
            None => {
                if self.no_pc_warnings < 4 {
                    log::warn!(
                        "TraceUnit ({},{}): EventPc mode received event hw_id={} \
                         with no PC; encoding sentinel pc=0",
                        self.col, self.row, hw_event_id
                    );
                    self.no_pc_warnings += 1;
                }
                self.pending_pc = 0;
            }
        }
    }
}
```

- [x] Add `no_pc_warnings: u32` to `TraceUnit`.

- [x] **Note:** every existing caller of `notify_event` in `trace_unit/mod.rs`, `tile/mod.rs`, `interpreter/engine/coordinator.rs`, `device/state/effects.rs`, and tests must add a third `None` argument. This is a coordinated rename — make sure every call site compiles before moving to Task 2.6. Any call site that has access to the core's PC (Task 3 will identify these) gets `Some(pc)` — for now during Task 2 keep them all `None`.

#### Step 2.6 — Run the encoder fixture test

- [x] Run: `cargo test --lib trace_unit::tests::mode1_encoder_byte_equivalent_to_decoder_fixture -- --exact`
- [x] Expected: PASS.

#### Step 2.7 — Add no-PC sentinel test

- [x] Add test:

```rust
#[test]
fn mode1_no_pc_emits_sentinel_zero() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_packet_type(0);
    tu.set_mode(TraceMode::EventPc);
    tu.set_event_slot(0, 23); // MEMORY_STALL
    tu.set_start_event(1);

    tu.notify_event(1, 0, None);            // start
    tu.notify_event(23, 0, None);           // memory stall fires -- no PC
    tu.commit_cycle(0);

    let frames = decode_mode1(tu.byte_buffer_for_test());
    let event_frames: Vec<_> = frames.iter()
        .filter(|f| matches!(f, TraceCommand::EventPc { .. }))
        .collect();
    assert_eq!(event_frames.len(), 1);
    if let TraceCommand::EventPc { mask, pc } = event_frames[0] {
        assert_eq!(*mask, 0b1, "slot 0 set");
        assert_eq!(*pc, 0, "no-PC sentinel must be 0");
    }
}
```

- [x] Run: `cargo test --lib trace_unit::tests::mode1_no_pc_emits_sentinel_zero -- --exact`
- [x] Expected: PASS.

#### Step 2.8 — Full lib test pass

- [x] Run: `cargo test --lib`
- [x] Expected: PASS. Investigate any new failures (most likely missing `, None` arg additions in test files).

#### Step 2.9 — Commit

```bash
git add src/device/trace_unit/ src/device/tile/ src/device/state/effects.rs \
        src/interpreter/engine/coordinator.rs
git commit -m "feat(trace): mode-1 (EventPC) frame encoder for core trace units

Adds the 4-byte EventPC encoding (8b mask + 14b PC) per AM025 mode-1
documentation and tools/trace_decoder/modes/mode1.py. Extends
TraceUnit::notify_event with pc: Option<u32>; mode-0 ignores it,
mode-1 records pc=0 as the no-PC sentinel and rate-limit-warns.
Config-time guard rejects EventPc on non-core packet types per
regdb (only core's Trace_Control0 has a Mode bitfield)."
```

---

### Task 3: Thread PC from coordinator to trace unit

**Goal:** Where `notify_core_trace_event` and `notify_mem_trace_event` are called from the coordinator's main loop, supply the actual PC for instruction-class events. Stalls and memory-side events pass `None`.

**Files:**
- Modify: `src/trace/mod.rs` (add `event_pc()` helper)
- Modify: `src/device/tile/mod.rs` (extend notify signatures)
- Modify: `src/device/state/effects.rs` (pass PC at call site)
- Modify: `src/interpreter/engine/coordinator.rs` (pass PC at every call site, audit)
- Modify: `src/device/tile/tests.rs` and `src/device/trace_unit/tests.rs` (update call sites)

#### Step 3.1 — Add the PC-extraction helper

- [x] In `src/trace/mod.rs`, add after `core_event_to_hw_id`:

```rust
/// Extract the PC field from an EventType variant, if it carries one.
///
/// Returns `Some(pc)` for instruction-class events (InstrVector,
/// InstrLoad, InstrStore, InstrCall, InstrReturn, InstrLockAcquireReq,
/// InstrLockReleaseReq, InstrStreamGet, InstrStreamPut, InstrEvent).
/// Returns `None` for core-state events (CoreActive, CoreDisabled),
/// stall events (MemoryStall, LockStall, StreamStall), memory-module
/// events (DMA, lock, port), branch events, etc. -- their PC is either
/// not meaningful or not directly available at notify time.
pub fn event_pc(event: &EventType) -> Option<u32> {
    match event {
        EventType::InstrVector { pc }            |
        EventType::InstrLoad { pc }              |
        EventType::InstrStore { pc }             |
        EventType::InstrCall { pc }              |
        EventType::InstrReturn { pc }            |
        EventType::InstrLockAcquireReq { pc }    |
        EventType::InstrLockReleaseReq { pc }    |
        EventType::InstrStreamGet { pc }         |
        EventType::InstrStreamPut { pc }         => Some(*pc),
        EventType::InstrEvent { pc, .. }         => Some(*pc),
        _ => None,
    }
}
```

- [x] Run: `cargo build --lib` to confirm it compiles.

#### Step 3.2 — Extend `notify_*_trace_event` signatures

- [x] Modify `src/device/tile/mod.rs:562-583`:

```rust
/// Notify a core module event for both tracing and edge detection.
#[inline]
pub fn notify_core_trace_event(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>) {
    self.core_trace.notify_event(hw_id, cycle, pc);
    for det in &mut self.core_edge_detectors {
        if det.input_event == hw_id {
            det.curr_active = true;
        }
    }
}

#[inline]
pub fn notify_mem_trace_event(&mut self, hw_id: u8, cycle: u64, pc: Option<u32>) {
    self.mem_trace.notify_event(hw_id, cycle, pc);
    for det in &mut self.mem_edge_detectors {
        if det.input_event == hw_id {
            det.curr_active = true;
        }
    }
}
```

- [x] Update the two callers in `src/device/state/effects.rs:377-378`:

```rust
// Before:
//   tile.notify_core_trace_event(*hw_id, current_cycle);
//   tile.notify_mem_trace_event(*hw_id, current_cycle);
// After:
tile.notify_core_trace_event(*hw_id, current_cycle, None);
tile.notify_mem_trace_event(*hw_id, current_cycle, None);
```

(Effects-driven events are pre-recorded and don't have a live PC handle — `None` is correct.)

- [x] Update internal `core_trace.notify_event` calls inside `evaluate_edge_detectors` (`src/device/tile/mod.rs:603, 617`): pass `None` (edge events fire post-hoc on level transitions; no instantaneous PC).

#### Step 3.3 — Audit and update coordinator.rs call sites

The full set of `notify_*_trace_event` call sites in `src/interpreter/engine/coordinator.rs` (per the grep):

| Line (approx) | Context | PC source |
|---------------|---------|-----------|
| ~681 | Core event drain loop (instruction-class) | `Some(pc)` from `crate::trace::event_pc(&evt.event)` |
| ~786, 790, 794 | `drain_mem_trace_events` (DMA, locks) | `None` (memory-side, no PC) |
| ~876, 878 | Port stream events | `None` (level events, post-hoc) |
| ~936 | DM bank conflict (memmod side) | `None` (memmod) |
| ~949 | MEMORY_STALL synthetic (core side) | `Some(core_pc)` from `self.cores[core_idx].context.pc` |

- [x] Replace the core-event-drain loop (around line 670-686):

```rust
let events = core.context.timing_context().events.events();
let new_start = core.trace_events_consumed;
if new_start < events.len() {
    let cycle = self.total_cycles;
    for evt in &events[new_start..] {
        if let Some(hw_id) = crate::trace::core_event_to_hw_id(&evt.event) {
            let pc = crate::trace::event_pc(&evt.event);
            tile.notify_core_trace_event(hw_id, cycle, pc);
        }
    }
    core.trace_events_consumed = events.len();
}
```

- [x] Lines ~786, 790, 794: drain_mem_trace_events. These are memory-module events drained from the array. PC is not meaningful for DMA / lock events that originate from memory modules. Pass `None`:

```rust
tile.notify_core_trace_event(id, cycle, None);  // shim PL
tile.notify_mem_trace_event(id, cycle, None);   // memtile
tile.notify_mem_trace_event(id, cycle, None);   // compute mem module
```

- [x] Lines ~876, 878: port events. `None`:

```rust
tile.notify_mem_trace_event(hw_id, cycle, None);
tile.notify_core_trace_event(hw_id, cycle, None);
```

- [x] Line ~936: bank-conflict DM event on memmod. `None`:

```rust
tile.notify_mem_trace_event(hw_id, cycle, None);
```

- [x] Line ~949: MEMORY_STALL synthetic on core side. The core's PC IS available — use it:

```rust
let pc = self.cores[core_idx].context.pc;  // if `pc` is the field name; verify via grep
if let Some(id) = hw_id {
    tile.notify_core_trace_event(id, cycle, Some(pc));
}
```

If the field is named differently (e.g., `program_counter`), adjust accordingly. Verify via `grep -n "pub.*pc\|fn pc\b" src/interpreter/state/`.

- [x] Line ~973: TRUE event broadcast (Phase 3c). HW fires TRUE every cycle, no specific PC. `None` is correct:

```rust
if tile.core_trace.is_configured() {
    tile.core_trace.notify_event(TRUE_EVENT, cycle, None);
}
if tile.mem_trace.is_configured() {
    tile.mem_trace.notify_event(TRUE_EVENT, cycle, None);
}
```

- [x] Update Task 1's wiring snippet to pass `None` for perfcnt firings (the firings happen at the cycle boundary, not at a specific instruction):

```rust
tile.notify_core_trace_event(hw_id, cycle, None);
// ...
tile.notify_mem_trace_event(hw_id, cycle, None);
```

#### Step 3.4 — Update test call sites

- [x] In `src/device/tile/tests.rs:325, 334, 347, 408, 452`: add a `None` third argument to every `notify_*_trace_event` call.
- [x] In `src/device/trace_unit/tests.rs`: any direct `tu.notify_event(hw, cycle)` calls already get the `, None` from Task 2.5 — verify nothing was missed.

#### Step 3.5 — Add a coordinator-level PC threading test

- [x] Add a coordinator integration test that:
  1. Builds a single compute tile, loads a tiny kernel (or synthetic core program) that emits an `InstrVector` at a known PC.
  2. Configures `tile.core_trace` mode=EventPc, slot 0 = INSTR_VECTOR (37).
  3. Steps the coordinator until the kernel completes.
  4. Decodes the trace bytes and asserts an EventPC frame at the expected PC.

```rust
#[test]
fn coordinator_threads_pc_to_mode1_trace() {
    let mut harness = test_helpers::single_compute_tile_harness();
    let tile = harness.tile_mut(0, 2);
    tile.core_trace.set_packet_type(0);
    tile.core_trace.set_mode(TraceMode::EventPc);
    tile.core_trace.set_event_slot(0, 37); // INSTR_VECTOR
    tile.core_trace.set_start_event(1);

    // Synthetically push an InstrVector event at pc=0x100 onto the core's
    // event log -- minimal way to exercise the coordinator drain path.
    harness.push_core_event(0, 2, EventType::InstrVector { pc: 0x100 });
    harness.step();

    let frames = decode_mode1(tile.core_trace.byte_buffer_for_test());
    let pc_frames: Vec<_> = frames.iter()
        .filter_map(|f| match f {
            TraceCommand::EventPc { mask, pc } if *mask & 0b1 != 0 => Some(*pc),
            _ => None,
        }).collect();
    assert_eq!(pc_frames, vec![0x100]);
}
```

- [x] Run: `cargo test --lib coordinator_threads_pc_to_mode1_trace -- --exact`
- [x] Expected: PASS.

#### Step 3.6 — Full lib test sweep

- [x] Run: `cargo test --lib`
- [x] Expected: PASS, baseline at 2755+ / 5 ignored or higher (Task 1, 2, 3 all add tests).

#### Step 3.7 — Commit

```bash
git add src/trace/mod.rs src/device/tile/ src/device/trace_unit/ \
        src/device/state/effects.rs src/interpreter/engine/coordinator.rs
git commit -m "feat(trace): thread current PC into core trace events

Extends notify_core_trace_event / notify_mem_trace_event with
pc: Option<u32>. Instruction-class events (InstrVector, InstrLoad,
InstrLockAcquireReq, ...) carry their pc field via the new
trace::event_pc(EventType) helper; stalls, memory-side events,
and edge events pass None. The trace unit ignores PC in mode 0
and uses it in mode 1 (encoded as 14-bit field, with no-PC events
recorded with sentinel pc=0)."
```

---

### Task 4: Extend `tools/mlir-trace-inject.py`

**Goal:** Add CLI surface to control trace mode, per-module-type grounding/sweep events, and perfcnt period. Emit `aie.trace.config @perf_<type>_<col>_<row>(%tile)` blocks containing `aie.trace.reg` ops that program Performance_Control0/1 + Performance_Counter0_Event_Value, alongside the existing `aie.trace` blocks for the 8 event slots.

**Files:**
- Modify: `tools/mlir-trace-inject.py`
- Modify: `tools/test_trace_inject.py`

#### Step 4.1 — Test: `--trace-mode event_pc` round-trips

- [x] In `tools/test_trace_inject.py`, add:

```python
def test_inject_mode_event_pc_round_trips(tmp_path, simple_design_mlir):
    inp = tmp_path / "in.mlir"
    inp.write_text(simple_design_mlir)
    out = tmp_path / "out.mlir"
    rc = run_inject([
        "--input", str(inp), "--out", str(out),
        "--trace-mode", "event_pc",
    ])
    assert rc == 0
    text = out.read_text()
    # Mode line:
    assert "aie.trace.mode EventPC" in text
    # Re-parse round-trip (idempotency check):
    rc2 = run_inject([
        "--input", str(out), "--out", str(tmp_path / "out2.mlir"),
        "--trace-mode", "event_pc",
    ])
    assert rc2 == 2  # second injection refused
```

- [x] Run: `pytest tools/test_trace_inject.py::test_inject_mode_event_pc_round_trips -v`
- [x] Expected: FAIL.

#### Step 4.2 — Add `--trace-mode` and per-module event flags

- [x] In `tools/mlir-trace-inject.py`'s `parse_args`, append:

```python
p.add_argument("--trace-mode", choices=("event_time", "event_pc"),
               default="event_time",
               help="trace mode for compute-core trace units. "
                    "event_time (mode 0, default) records cycle deltas; "
                    "event_pc (mode 1) records PCs. Mode 1 is core-only -- "
                    "memmod/memtile/shim trace units always remain in mode 0 "
                    "(their Trace_Control0 has no Mode bitfield per regdb).")

# Grounding (fixed slots, never overwritten by sweep):
p.add_argument("--core-grounding",
               default="PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
               help="comma-separated event names reserved in fixed slots "
                    "of every compute-core trace unit. Default reserves "
                    "perfcnt cycle clock + two software pin events.")
p.add_argument("--memmod-grounding", default="PERF_CNT_0",
               help="grounding events for compute-tile memmod trace unit.")
p.add_argument("--memtile-grounding", default="PERF_CNT_0",
               help="grounding events for memtile trace unit.")
p.add_argument("--shim-grounding", default="PERF_CNT_0",
               help="grounding events for shim PL trace unit.")

# Sweep events (rotated by the orchestrator; injection only writes the
# default initial pattern, which the patcher will overwrite per batch):
p.add_argument("--core-sweep-events", default=None,
               help="comma-separated event names to sweep on compute cores; "
                    "'all' enumerates from the event header. Default uses "
                    "the existing 5 hard-coded core defaults.")
p.add_argument("--memmod-sweep-events", default=None,
               help="comma-separated event names to sweep on compute memmod. "
                    "Default: don't inject memmod trace.")
p.add_argument("--memtile-sweep-events", default=None,
               help="comma-separated event names to sweep on memtile. "
                    "Default: don't inject memtile trace.")
p.add_argument("--shim-sweep-events", default=None,
               help="comma-separated event names to sweep on shim. "
                    "Default: don't inject shim trace.")

p.add_argument("--perfcnt-period", type=int, default=1024,
               help="cycles between PERF_CNT_0_EVENT fires when grounding "
                    "includes PERF_CNT_0 (default: 1024).")
```

- [x] Run: `python3 tools/mlir-trace-inject.py --help` → confirm new flags appear.

#### Step 4.3 — Plumb mode selection into the existing injector body

- [x] Modify the per-tile injection loop (~line 254-274) so the trace_mode op reflects the CLI flag for compute cores:

```python
mode_attr = (aied.TraceMode.EventPC
             if args.trace_mode == "event_pc"
             else aied.TraceMode.EventTime)

@aied.trace(tile_val, _trace_sym(col, row))
def _trace_body():
    aied.trace_mode(mode_attr)
    aied.trace_packet(_TRACE_PACKET_ID_START, aied.TracePacketType.Core)
    # ... grounding events first, then sweep events ...
```

- [x] Run the Step 4.1 test again — should pass for the `aie.trace.mode EventPC` line check.

#### Step 4.4 — Inject grounding + sweep events per tile type

- [x] Refactor the event-list construction to derive from the per-module grounding/sweep CLI flags. For each module type the injector touches:

```python
def _resolve_events(grounding: str, sweep: str | None,
                    defaults: tuple[str, ...] | None,
                    all_events_for_type: list[str]) -> list[str]:
    """Combine grounding (fixed) + sweep (rotated) into the 8 slots.
    Grounding fills first; sweep fills the rest. 'all' means enumerate
    every event; missing sweep falls back to defaults; if defaults is
    also None this module gets no trace injection."""
    g = [s.strip() for s in grounding.split(",") if s.strip()]
    if sweep is None:
        if defaults is None:
            return []
        s = list(defaults)
    elif sweep == "all":
        s = [e for e in all_events_for_type if e not in g]
    else:
        s = [s.strip() for s in sweep.split(",") if s.strip()]
    seen = set(g)
    final = list(g)
    for e in s:
        if e not in seen and len(final) < 8:
            final.append(e)
            seen.add(e)
    return final
```

- [x] In the trace body, emit `trace_event` for each event in the resolved list, in slot order. Keep grounding events in their fixed positions.

- [x] Add module-type detection: rows == 0 → shim, row == 1 → memtile, row >= 2 → compute. For compute tiles, also inject memmod traces if `--memmod-sweep-events` is set.

#### Step 4.5 — Emit perfcnt config blocks

- [x] After every `aie.trace` block, if `PERF_CNT_0` (or any perfcnt event) is in that tile's grounding set, emit:

```python
def _emit_perfcnt_config(aied, tile_val, sym_prefix, period: int,
                        module_type: str, col: int, row: int):
    """Emit aie.trace.config @perf_<type>_<col>_<row> with three trace.reg ops.

    Lowering: AIEXInlineTraceConfig in mlir-aie translates each trace.reg op
    into npu.write32 ops in the runtime sequence. No new dialect ops required.

    Per regdb (aie_registers_aie2.json):
      Performance_Control0
        bits [6:0]   = Cnt0_Start_Event   <- ACTIVE (event 28 on core)
        bits [14:8]  = Cnt0_Stop_Event    <- 0 (no stop)
        ... (other counters left zero)
      Performance_Control1
        bits [6:0]   = Cnt0_Reset_Event   <- PERF_CNT_0 (event 5) for free-run
      Performance_Counter0_Event_Value
        full 32-bit  = period

    The chosen start_event=ACTIVE means the counter only ticks during cycles
    the core is in Execute state -- matching the spec's semantics for
    "useful work cycles only." Sweep cycle counts will be slightly less than
    wall-clock cycles but more meaningful for divergence comparison.
    """
    sym = f"perf_{module_type}_{col}_{row}"
    @aied.trace_config(tile_val, sym)
    def _perf_body():
        # ACTIVE = event 28 (core), stop=0
        aied.trace_reg(register="Performance_Control0",
                       field="Cnt0_Start_Event", value=28)
        # Self-reset on overflow
        aied.trace_reg(register="Performance_Control1",
                       field="Cnt0_Reset_Event", value=5)
        # Threshold
        aied.trace_reg(register="Performance_Counter0_Event_Value",
                       value=period)
    # Reference it from the runtime sequence:
    return sym
```

- [x] In the runtime-sequence prologue, append `aied.trace_start_config(perf_sym)` for each perfcnt config emitted, alongside the existing `aied.trace_start_config(_trace_sym(...))` for the trace blocks.

- [x] **Important:** the exact `aied.trace_reg(...)` builder signature comes from mlir-aie's Python bindings; verify the kwarg names with `grep -rn "TraceRegOp\|trace_reg" /home/triple/npu-work/mlir-aie/python/`. If the wrapper expects positional args or different kwargs, adjust accordingly. The spec says these compile via `AIEXInlineTraceConfig`, so the test gate is whether `aiecc.py` can lower the resulting MLIR.

#### Step 4.6 — Test: perfcnt config blocks emitted

- [x] Add to `tools/test_trace_inject.py`:

```python
def test_inject_perfcnt_config_emitted(tmp_path, simple_design_mlir):
    inp = tmp_path / "in.mlir"
    inp.write_text(simple_design_mlir)
    out = tmp_path / "out.mlir"
    rc = run_inject([
        "--input", str(inp), "--out", str(out),
        "--trace-mode", "event_pc",
        "--core-grounding", "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1",
        "--perfcnt-period", "1024",
    ])
    assert rc == 0
    text = out.read_text()
    # Should contain at least one perf_core_<col>_<row> trace.config block
    assert "@perf_core_" in text, text[:1000]
    # Should reference Performance_Counter0_Event_Value with value=1024
    assert "Performance_Counter0_Event_Value" in text
    assert "1024" in text
    # Should appear in start_config list in runtime_sequence
    assert "aie.trace.start_config @perf_core_" in text
```

- [x] Run: `pytest tools/test_trace_inject.py -v`
- [x] Expected: PASS for the new test plus existing tests.

#### Step 4.7 — Test: warn-and-continue on `--trace-mode event_pc` with non-core sweeps

- [x] Add:

```python
def test_inject_event_pc_with_non_core_sweep_warns(tmp_path, simple_design_mlir, capsys):
    inp = tmp_path / "in.mlir"
    inp.write_text(simple_design_mlir)
    out = tmp_path / "out.mlir"
    rc = run_inject([
        "--input", str(inp), "--out", str(out),
        "--trace-mode", "event_pc",
        "--memmod-sweep-events", "DMA_S2MM_0_FINISHED_BD,LOCK_0_ACQUIRED",
    ])
    assert rc == 0  # NOT exit 2 -- warn and continue
    captured = capsys.readouterr()
    assert "warning" in captured.err.lower()
    assert "memmod" in captured.err.lower()
    text = out.read_text()
    # core trace block still in EventPC mode:
    assert "aie.trace.mode EventPC" in text
    # memmod block (if emitted) should NOT have EventPC -- regdb forces mode 0
    # (the actual mlir-aie lowering won't even emit Mode bits for memmod
    # variant of Trace_Control0, so absence is the behavioral test).
```

- [x] Implement the warning in the injector — when `--trace-mode event_pc` and any of `--memmod-sweep-events`, `--memtile-sweep-events`, `--shim-sweep-events` are non-empty, print to stderr:

```python
if args.trace_mode == "event_pc" and any(s for s in [
    args.memmod_sweep_events, args.memtile_sweep_events,
    args.shim_sweep_events
] if s):
    print(
        "warning: --trace-mode event_pc applies to compute-core trace units "
        "only; memmod/memtile/shim trace units stay in event_time per regdb "
        "(Mode bitfield exists only in core's Trace_Control0). Non-core "
        "sweep events will be recorded in mode 0.",
        file=sys.stderr,
    )
```

- [x] Run: `pytest tools/test_trace_inject.py -v` → all PASS.

#### Step 4.8 — Commit

```bash
git add tools/mlir-trace-inject.py tools/test_trace_inject.py
git commit -m "feat(trace-inject): mode-1 trace + per-module grounding + perfcnt config

Extends mlir-trace-inject.py with --trace-mode (event_time | event_pc),
per-tile-type grounding/sweep event flags, and --perfcnt-period.
Emits one aie.trace.config @perf_<type>_<col>_<row> block per traced
tile that contains PERF_CNT_0; the block uses aie.trace.reg ops that
mlir-aie's AIEXInlineTraceConfig pass lowers to npu.write32 in the
runtime sequence -- no new dialect ops needed.

event_pc mode applies to core trace units only; memmod/memtile/shim
remain in mode 0 (regdb-enforced -- their Trace_Control0 has no Mode
bitfield). Combining --trace-mode event_pc with non-core sweep events
warns and continues."
```

---

### Task 5: Extend `tools/trace-patch-events.py` with `--multi-tile`

**Goal:** Replace the per-tile subprocess-per-batch overhead with a single invocation that takes a JSON list of patch specs and applies them all in one pass over `insts.bin`.

**Files:**
- Modify: `tools/trace-patch-events.py`
- Modify: `tools/test_trace_prepare.py` (or extend `tools/test_cpp_trace_patch.py` if patch-events tests live there)

#### Step 5.1 — Test: multi-tile patch matches per-tile chain

- [x] Add a test that compares N sequential `--col/--row` invocations vs. one `--multi-tile` invocation with the same N specs, and asserts byte-identical output:

```python
def test_multi_tile_matches_chained_single_tile(tmp_path, sample_insts_bin):
    spec = [
        {"col": 0, "row": 2, "tile_type": "core", "events": [37, 23, 26, 28]},
        {"col": 1, "row": 2, "tile_type": "core", "events": [37, 23, 26, 28]},
        {"col": 0, "row": 3, "tile_type": "core", "events": [37, 23, 26, 28]},
    ]
    spec_json = tmp_path / "spec.json"
    spec_json.write_text(json.dumps(spec))

    # Multi-tile path:
    multi_out = tmp_path / "multi.bin"
    rc = run_patch([
        str(sample_insts_bin),
        "--multi-tile", str(spec_json),
        "--output", str(multi_out),
    ])
    assert rc == 0

    # Chained per-tile path:
    chain_in = sample_insts_bin
    for i, s in enumerate(spec):
        chain_out = tmp_path / f"chain{i}.bin"
        rc = run_patch([
            str(chain_in),
            "--col", str(s["col"]),
            "--row", str(s["row"]),
            "--tile-type", s["tile_type"],
            "--events", ",".join(str(e) for e in s["events"]),
            "--output", str(chain_out),
        ])
        assert rc == 0
        chain_in = chain_out

    assert multi_out.read_bytes() == chain_in.read_bytes()
```

- [x] Run: `pytest tools/test_cpp_trace_patch.py::test_multi_tile_matches_chained_single_tile -v` (or wherever the test is added).
- [x] Expected: FAIL.

#### Step 5.2 — Implement `--multi-tile`

- [x] In `tools/trace-patch-events.py`'s `main`, add:

```python
ap.add_argument("--multi-tile", type=Path, default=None,
                help="JSON file containing a list of patch specs: "
                     "[{col, row, tile_type, events?, start_event?, "
                     "stop_event?, mode?}, ...]. Applies all patches in "
                     "one process invocation (avoids per-tile subprocess "
                     "overhead in trace-sweep). Mutually exclusive with "
                     "--col/--row/--tile-type.")
```

- [x] After arg parsing, if `--multi-tile` is set, branch:

```python
if args.multi_tile is not None:
    if any([args.col is not None, args.row is not None, args.tile_type is not None]):
        ap.error("--multi-tile is mutually exclusive with --col/--row/--tile-type")
    spec_list = json.loads(args.multi_tile.read_text())
    data = Path(args.input).read_bytes()
    summaries = []
    for s in spec_list:
        col = int(s["col"]); row = int(s["row"])
        tile_type = s["tile_type"]
        if "events" in s:
            events = _parse_events_arg(",".join(str(e) for e in s["events"]))
            data, n = patch_events(data, col, row, tile_type, events)
            summaries.append(f"({col},{row},{tile_type}) events={events} ({n})")
        if any(k in s for k in ("start_event", "stop_event", "mode")):
            data, n = patch_trace_control(
                data, col, row, tile_type,
                start_event=s.get("start_event"),
                stop_event=s.get("stop_event"),
                mode=s.get("mode"),
            )
            summaries.append(f"({col},{row},{tile_type}) control ({n})")
    Path(args.output).write_bytes(data)
    print("trace-patch-events multi-tile: " + "; ".join(summaries))
    return 0
```

- [x] Run: `pytest tools/test_cpp_trace_patch.py::test_multi_tile_matches_chained_single_tile -v`
- [x] Expected: PASS.

#### Step 5.3 — Test: `--multi-tile` rejects conflicting args

- [x] Add a test asserting `--multi-tile` with `--col` exits non-zero with a clear error message.

#### Step 5.4 — Commit

```bash
git add tools/trace-patch-events.py tools/test_cpp_trace_patch.py
git commit -m "feat(trace-patch-events): --multi-tile JSON spec for batch sweep

Replaces per-tile subprocess overhead with one invocation that applies
N patch specs in a single pass over insts.bin. Byte-identical to the
chained-single-tile path. Used by trace-sweep's per-batch lockstep
patching across all tile types simultaneously."
```

---

### Task 6: Extend `tools/trace-sweep.py` with mode-1 sweep + mode-2 finishing batch

**Goal:** Add per-tile cursors that lockstep across batches, drive the orchestrator in mode 1 (cores) + mode 0 (others), patch via the new `--multi-tile`, run a final HW-only mode-2 baseline batch, and produce a `sweep-manifest.json` with the cross-batch grounding-PC consistency self-check result.

**Files:**
- Modify: `tools/trace-sweep.py`
- Modify: `tools/test_trace_sweep.py`

#### Step 6.1 — Test: per-tile-type cursor lockstep

- [x] In `tools/test_trace_sweep.py`, add a unit test for a `_build_lockstep_batches` helper:

```python
def test_lockstep_batches_one_per_tile_per_batch():
    cursors = {
        "core_0_2":   {"sweep": ["INSTR_VECTOR", "MEMORY_STALL", "STREAM_STALL"], "remaining_slots": 5},
        "core_0_3":   {"sweep": ["INSTR_VECTOR", "MEMORY_STALL"],                  "remaining_slots": 5},
        "memmod_0_2": {"sweep": ["DMA_S2MM_0_FINISHED_BD"],                        "remaining_slots": 7},
    }
    batches = _build_lockstep_batches(cursors)
    # Three sweep entries -> max(ceil(3/5), ceil(2/5), ceil(1/7)) = max(1,1,1) = 1 batch
    assert len(batches) == 1
    b0 = batches[0]
    assert set(b0["core_0_2"]) == {"INSTR_VECTOR", "MEMORY_STALL", "STREAM_STALL"}
    assert set(b0["core_0_3"]) == {"INSTR_VECTOR", "MEMORY_STALL"}
    assert set(b0["memmod_0_2"]) == {"DMA_S2MM_0_FINISHED_BD"}

    # Now exhaust:
    cursors2 = {
        "core_0_2": {"sweep": list(range(12)), "remaining_slots": 5},
    }
    batches2 = _build_lockstep_batches(cursors2)
    assert len(batches2) == 3  # ceil(12/5) = 3
    assert sum(len(b["core_0_2"]) for b in batches2) == 12

    # Cursor exhaustion behavior: if one cursor exhausts before others,
    # later batches emit grounding-only for that cursor.
    cursors3 = {
        "core_0_2":   {"sweep": ["A", "B"],         "remaining_slots": 1},
        "core_0_3":   {"sweep": ["A", "B", "C", "D"], "remaining_slots": 1},
    }
    batches3 = _build_lockstep_batches(cursors3)
    assert len(batches3) == 4
    assert batches3[2]["core_0_2"] == []  # exhausted
    assert batches3[3]["core_0_3"] == ["D"]
```

- [x] Run: `pytest tools/test_trace_sweep.py::test_lockstep_batches_one_per_tile_per_batch -v`
- [x] Expected: FAIL.

#### Step 6.2 — Implement `_build_lockstep_batches`

- [x] Add helper function in `tools/trace-sweep.py`:

```python
def _build_lockstep_batches(cursors: dict) -> list[dict]:
    """Generate per-batch event assignments across all tile cursors.

    Each cursor (one per (tile, module-type)) holds a sweep list and a
    remaining_slots count (= 8 - len(grounding)). Per batch, each cursor
    consumes its next remaining_slots events. Total batch count =
    max(ceil(per-cursor sweep length / per-cursor remaining_slots)).
    Cursors that exhaust early emit empty assignments (grounding-only).
    """
    if not cursors:
        return []
    n_batches = max(
        1 if not c["sweep"] else
        (len(c["sweep"]) + c["remaining_slots"] - 1) // c["remaining_slots"]
        for c in cursors.values()
    )
    batches = []
    for batch_idx in range(n_batches):
        batch = {}
        for key, c in cursors.items():
            start = batch_idx * c["remaining_slots"]
            end = start + c["remaining_slots"]
            batch[key] = list(c["sweep"][start:end])
        batches.append(batch)
    return batches
```

- [x] Run: `pytest tools/test_trace_sweep.py::test_lockstep_batches_one_per_tile_per_batch -v` → PASS.

#### Step 6.3 — Wire lockstep into the sweep main loop

- [x] Modify the sweep `main()` to add `--mode`, per-module-type grounding/sweep flags mirroring the injector, and `--with-mode2-baseline`:

```python
ap.add_argument("--mode", choices=("event_time", "event_pc"),
                default="event_time")
ap.add_argument("--core-grounding", default="PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1")
ap.add_argument("--memmod-grounding", default="PERF_CNT_0")
ap.add_argument("--memtile-grounding", default="PERF_CNT_0")
ap.add_argument("--shim-grounding", default="PERF_CNT_0")
ap.add_argument("--core-sweep", default="all")
ap.add_argument("--memmod-sweep", default="all")
ap.add_argument("--memtile-sweep", default="all")
ap.add_argument("--shim-sweep", default="all")
ap.add_argument("--perfcnt-period", type=int, default=1024)
ap.add_argument("--with-mode2-baseline", action="store_true", default=True)
ap.add_argument("--no-mode2-baseline", action="store_false",
                dest="with_mode2_baseline")
```

- [x] Replace the existing single-tile-per-batch dispatch with one that:
  1. Builds cursors for every tile type in scope (compute cores + their memmods, memtiles, shims).
  2. Calls `_build_lockstep_batches(cursors)`.
  3. For each batch:
     a. Constructs a multi-tile patch spec (list of `{col, row, tile_type, events: [hw_ids...]}` dicts) preserving the grounding slots.
     b. Calls `trace-patch-events.py --multi-tile <spec>` once.
     c. Runs HW + EMU via `bridge-trace-runner --batch-stdin`.
     d. Decodes via `parse-trace.py --decoder=ours --trace-mode <mode>`.
     e. Records per-batch HW grounding-PC sets in a manifest.

- [x] Implement after the lockstep loop:

```python
# Mode-2 finishing batch (HW only):
if args.with_mode2_baseline:
    mode2_spec = [
        {"col": col, "row": row, "tile_type": "core", "mode": 2}
        for (col, row, _) in compute_tile_specs
    ]
    spec_path = out_dir / "mode2_patch.json"
    spec_path.write_text(json.dumps(mode2_spec))
    mode2_insts = out_dir / "mode2_insts.bin"
    subprocess.run([sys.executable, str(PATCHER), str(base_insts),
                    "--multi-tile", str(spec_path),
                    "--output", str(mode2_insts)], check=True)
    # Run HW only:
    hw_runner.run_one(insts=mode2_insts, ...)
    # Decode:
    subprocess.run([sys.executable, str(PARSE_TOOL),
                    "--input", str(mode2_trace_bin),
                    "--out-events", str(out_dir / "mode2-baseline" / f"{test}.events.json"),
                    "--trace-mode", "inst_exec",
                    "--decoder", "ours"], check=True)
```

#### Step 6.4 — Cross-batch grounding-PC self-check

- [x] After the sweep completes, before writing `sweep-manifest.json`, compute the per-batch HW grounding-event PC sets:

```python
def _check_grounding_pc_invariance(batches_dir: Path,
                                   grounding_events: list[str]) -> dict:
    """Returns {'unsafe_for_pc_join': bool, 'reason': str | None,
                'per_batch_grounding_pcs': {batch_idx: {event_name: set[pc]}}}.

    Per-spec, INSTR_EVENT_0 / INSTR_EVENT_1 are kernel-state pins fired by
    the kernel's instructions; their PCs MUST be batch-invariant for
    PC-anchored cross-batch joining to be sound.
    """
    per_batch = {}
    for batch_dir in sorted(batches_dir.glob("batch_*")):
        events_json = batch_dir / "hw" / "trace.events.json"
        if not events_json.exists():
            continue
        bidx = int(batch_dir.name.split("_")[1])
        per_batch[bidx] = {}
        records = json.loads(events_json.read_text())
        events_list = records.get("events", records)
        for rec in events_list:
            name = rec.get("name", "")
            if name in grounding_events:
                per_batch[bidx].setdefault(name, set()).add(rec["ts"])
    # Compare across batches:
    for ev in grounding_events:
        seen_pcs = set()
        for bidx, by_ev in per_batch.items():
            if ev in by_ev:
                if seen_pcs and by_ev[ev] != seen_pcs:
                    return {
                        "unsafe_for_pc_join": True,
                        "reason": f"grounding event {ev} PC drifted: "
                                  f"first_batch={sorted(seen_pcs)} "
                                  f"batch_{bidx}={sorted(by_ev[ev])}",
                        "per_batch_grounding_pcs": {
                            k: {n: sorted(s) for n, s in v.items()}
                            for k, v in per_batch.items()
                        },
                    }
                if not seen_pcs:
                    seen_pcs = by_ev[ev]
    return {
        "unsafe_for_pc_join": False, "reason": None,
        "per_batch_grounding_pcs": {
            k: {n: sorted(s) for n, s in v.items()}
            for k, v in per_batch.items()
        },
    }
```

- [x] Write the manifest:

```python
manifest = {
    "test_name": args.test,
    "compiler": args.compiler,
    "mode": args.mode,
    "perfcnt_period": args.perfcnt_period,
    "tiles": [...],
    "n_batches": len(batches),
    "grounding": {
        "core": args.core_grounding.split(","),
        "memmod": args.memmod_grounding.split(","),
        "memtile": args.memtile_grounding.split(","),
        "shim": args.shim_grounding.split(","),
    },
    "mode2_baseline": (args.with_mode2_baseline and mode2_succeeded),
    **_check_grounding_pc_invariance(out_dir, args.core_grounding.split(",")),
}
(out_dir / "sweep-manifest.json").write_text(json.dumps(manifest, indent=2))
```

#### Step 6.5 — Test: `unsafe_for_pc_join` flagged on drift

- [x] Add a test that builds a fake `batches_dir` with two batches whose `INSTR_EVENT_0` PCs differ, calls `_check_grounding_pc_invariance`, asserts `unsafe_for_pc_join=True`.

#### Step 6.6 — Test: mode-2 finishing batch runs HW-only

- [x] Add a test (with mocked `bridge-trace-runner` / `parse-trace.py`) that asserts the mode-2 batch is invoked with `--no-emu` (or equivalent) and produces `mode2-baseline/<test>/<tile>.events.json`.

#### Step 6.7 — Run all sweep tests

- [x] Run: `pytest tools/test_trace_sweep.py -v`
- [x] Expected: PASS.

#### Step 6.8 — Commit

```bash
git add tools/trace-sweep.py tools/test_trace_sweep.py
git commit -m "feat(trace-sweep): mode-1 lockstep sweep + mode-2 baseline + manifest

Replaces single-tile-per-batch sweep with per-tile-type lockstep
cursor batching. All compute cores plus their memmods plus memtiles
plus shims advance together; one --multi-tile patch + one HW + one
EMU run per batch. Mode-1 (EventPC) on cores; mode-0 on others
(regdb-forced). After the mode-1 sweep, an optional HW-only mode-2
baseline batch captures inst_exec traces for future A.2b consumption.

Writes sweep-manifest.json with per-batch grounding PC sets and a
cross-batch invariance check; flags unsafe_for_pc_join=true if
INSTR_EVENT_0/1 PCs drift across batches."
```

---

### Task 7: `compare_pc_anchored` in `src/trace/compare.rs`

**Goal:** Add an entry point that, given per-batch HW + EMU events JSON pairs, partitions tiles by `pkt_type` (core vs others), runs PC-set + multiset diff on cores, and computes perfcnt-anchored cycle bands for non-core tiles.

**Files:**
- Modify: `src/trace/compare.rs`
- Modify: `src/bin/trace_compare.rs`
- Modify: `src/trace/compare/tests.rs` (or wherever existing compare tests live; grep)

#### Step 7.1 — Add `PCAnchoredReport` and extend `BatchResult`

- [x] In `src/trace/compare.rs`, after the existing `BatchResult`:

```rust
/// PC-anchored comparison output for one batch, one tile.
#[derive(Debug, Default, Clone)]
pub struct PCAnchoredReport {
    pub pkt_type: u32,
    /// Per event name: (PCs only in HW, PCs only in EMU).
    pub set_diff: std::collections::HashMap<String, (std::collections::HashSet<u64>, std::collections::HashSet<u64>)>,
    /// Per event name: per-PC (hw_count, emu_count, delta=hw-emu).
    pub multiset_diff: std::collections::HashMap<String, std::collections::HashMap<u64, (u32, u32, i32)>>,
    /// Per event name: per-PC perfcnt-anchored cycle band.
    pub cycle_bands: std::collections::HashMap<String, std::collections::HashMap<u64, CycleBand>>,
    pub unanchored_count_hw: usize,
    pub unanchored_count_emu: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct CycleBand {
    pub hw_cycle_est: u64,
    pub emu_cycle_est: u64,
    pub delta_cycles: i64,
    pub exceeds_tolerance: bool,
}

// Add to BatchResult:
pub struct BatchResult {
    pub batch_idx: usize,
    pub config: EventsConfig,
    pub tiles: Vec<(TileKey, TileResult)>,
    pub stall_attributions: Vec<StallAttribution>,
    pub cross_tile: Option<CrossTileResult>,
    /// PC-anchored per-tile results (populated when --pc-anchored flag is set).
    /// Keyed by TileKey for quick lookup.
    pub pc_anchored: std::collections::HashMap<TileKey, PCAnchoredReport>,
}
```

- [x] Update every `BatchResult { ... }` literal (Step 7.4 finds them) to initialize `pc_anchored: HashMap::new()`.

#### Step 7.2 — Test: PC-set diff with synthetic events

- [x] Add to compare's tests module:

```rust
#[test]
fn pc_anchored_set_diff_finds_hw_only_and_emu_only() {
    // HW: INSTR_VECTOR fires at PCs {100, 200, 300}
    // EMU: INSTR_VECTOR fires at PCs {100, 250, 300}
    // Expected hw_only={200}, emu_only={250}.

    let hw_events = vec![
        TileEvent { slot: 0, abs_cycle: 100 },
        TileEvent { slot: 0, abs_cycle: 200 },
        TileEvent { slot: 0, abs_cycle: 300 },
    ];
    let emu_events = vec![
        TileEvent { slot: 0, abs_cycle: 100 },
        TileEvent { slot: 0, abs_cycle: 250 },
        TileEvent { slot: 0, abs_cycle: 300 },
    ];
    let names = vec!["INSTR_VECTOR".to_string()];
    let report = compare_pc_anchored_for_tile(
        TileKey { col: 0, row: 2, pkt_type: 0 },
        &hw_events, &emu_events, &names,
    );

    let (hw_only, emu_only) = &report.set_diff["INSTR_VECTOR"];
    assert_eq!(hw_only, &HashSet::from([200u64]));
    assert_eq!(emu_only, &HashSet::from([250u64]));
}
```

- [x] Run: `cargo test --lib compare::pc_anchored_set_diff_finds_hw_only_and_emu_only -- --exact`
- [x] Expected: FAIL.

#### Step 7.3 — Implement `compare_pc_anchored_for_tile`

- [x] Add to `src/trace/compare.rs`:

```rust
pub fn compare_pc_anchored_for_tile(
    key: TileKey,
    hw_events: &[TileEvent],
    emu_events: &[TileEvent],
    slot_names: &[String],
) -> PCAnchoredReport {
    let mut report = PCAnchoredReport {
        pkt_type: key.pkt_type as u32,
        ..Default::default()
    };

    // Drop ts==0 events into unanchored bucket.
    let hw_anchored: Vec<&TileEvent> = hw_events.iter()
        .filter(|e| {
            if e.abs_cycle == 0 {
                report.unanchored_count_hw += 1;
                false
            } else { true }
        }).collect();
    let emu_anchored: Vec<&TileEvent> = emu_events.iter()
        .filter(|e| {
            if e.abs_cycle == 0 {
                report.unanchored_count_emu += 1;
                false
            } else { true }
        }).collect();

    // Group by event name (slot -> name lookup).
    let mut hw_by_name: HashMap<String, Vec<u64>> = HashMap::new();
    let mut emu_by_name: HashMap<String, Vec<u64>> = HashMap::new();
    for ev in hw_anchored {
        let name = slot_name(ev.slot, slot_names);
        hw_by_name.entry(name).or_default().push(ev.abs_cycle);
    }
    for ev in emu_anchored {
        let name = slot_name(ev.slot, slot_names);
        emu_by_name.entry(name).or_default().push(ev.abs_cycle);
    }

    // Set + multiset diff per event name.
    let all_names: BTreeSet<String> = hw_by_name.keys().chain(emu_by_name.keys()).cloned().collect();
    for name in all_names {
        let hw_pcs: &[u64] = hw_by_name.get(&name).map(|v| v.as_slice()).unwrap_or(&[]);
        let emu_pcs: &[u64] = emu_by_name.get(&name).map(|v| v.as_slice()).unwrap_or(&[]);

        let hw_set: HashSet<u64> = hw_pcs.iter().copied().collect();
        let emu_set: HashSet<u64> = emu_pcs.iter().copied().collect();
        let hw_only: HashSet<u64> = hw_set.difference(&emu_set).copied().collect();
        let emu_only: HashSet<u64> = emu_set.difference(&hw_set).copied().collect();
        report.set_diff.insert(name.clone(), (hw_only, emu_only));

        let mut multiset = HashMap::new();
        let union: HashSet<u64> = hw_set.union(&emu_set).copied().collect();
        for pc in union {
            let hw_count = hw_pcs.iter().filter(|&&p| p == pc).count() as u32;
            let emu_count = emu_pcs.iter().filter(|&&p| p == pc).count() as u32;
            let delta = hw_count as i32 - emu_count as i32;
            multiset.insert(pc, (hw_count, emu_count, delta));
        }
        report.multiset_diff.insert(name, multiset);
    }

    report
}
```

- [x] Run: `cargo test --lib compare::pc_anchored_set_diff_finds_hw_only_and_emu_only -- --exact`
- [x] Expected: PASS.

#### Step 7.4 — Test + implement perfcnt-anchored cycle band

- [x] Test: synthesize a perfcnt overflow stream at PCs `[10, 60, 110]` (3 firings of period=50), then a regular event at PC 35 (between perfcnt[0]=10 and perfcnt[1]=60). The interpolated cycle for PC 35 should be `period * (0 + (35-10)/(60-10)) = 50 * 0.5 = 25`.

```rust
#[test]
fn cycle_band_linear_interpolation() {
    let perfcnt_pcs = vec![10u64, 60, 110];
    let period = 50u64;
    let est = interpolate_cycle_from_perfcnt(35, &perfcnt_pcs, period);
    assert_eq!(est, 25);
    let est2 = interpolate_cycle_from_perfcnt(85, &perfcnt_pcs, period);
    assert_eq!(est2, 75); // 50 + (85-60)/(110-60)*50 = 75
}
```

- [x] Implement:

```rust
fn interpolate_cycle_from_perfcnt(pc: u64, perfcnt_pcs: &[u64], period: u64) -> u64 {
    if perfcnt_pcs.is_empty() {
        return 0;
    }
    let idx = perfcnt_pcs.partition_point(|&p| p < pc);
    if idx == 0 {
        return 0;
    }
    if idx >= perfcnt_pcs.len() {
        return (perfcnt_pcs.len() as u64 - 1) * period;
    }
    let pc_below = perfcnt_pcs[idx - 1];
    let pc_above = perfcnt_pcs[idx];
    let span = pc_above.saturating_sub(pc_below).max(1);
    let frac = (pc - pc_below) as f64 / span as f64;
    ((idx as f64 - 1.0 + frac) * period as f64) as u64
}
```

- [x] Wire into `compare_pc_anchored_for_tile`: after multiset diff, find the slot whose name is `PERF_CNT_0_EVENT` or `PERF_CNT_0` (depending on what mlir-aie names emit; both candidates), pull that slot's PC sequence as `perfcnt_pcs`, and compute `cycle_bands` for every other event at every PC.

- [x] Run: `cargo test --lib compare:: -- --exact` → all PC-anchored tests PASS.

#### Step 7.5 — Add `compare_pc_anchored_batch` driver

- [x] Add a higher-level entry point `compare_pc_anchored(hw_path, emu_path, config, batch_idx)` that loads JSON and dispatches per-tile, populating `BatchResult::pc_anchored` for every core-pkt_type tile.

#### Step 7.6 — Add `--pc-anchored` to `trace_compare` binary

- [x] In `src/bin/trace_compare.rs`, add:

```rust
"--pc-anchored" => { opts.pc_anchored = true; }
```

- [x] Add `pub pc_anchored: bool` to `AnalysisOptions`. When set, `compare_batch_with_opts` calls `compare_pc_anchored(...)` and merges results into `BatchResult::pc_anchored`.

#### Step 7.7 — Commit

```bash
git add src/trace/compare.rs src/bin/trace_compare.rs
git commit -m "feat(compare): PC-anchored set/multiset diff + perfcnt cycle bands

Adds compare_pc_anchored_for_tile and PCAnchoredReport. Per event name,
computes PC-set diff (HW-only / EMU-only PCs) and per-PC multiset diff
with hw/emu counts and delta. Perfcnt overflow PCs (PERF_CNT_0 stream)
provide a deterministic cycle clock; non-perfcnt events get linearly-
interpolated cycle estimates with a period/2 jitter tolerance.
Unanchored events (ts=0 sentinel from no-PC mode-1 frames) accumulate
into a separate count, excluded from diff. Driven by --pc-anchored
flag on trace-compare."
```

---

### Task 8: Sweep aggregation + `format_report` extensions

**Goal:** When `compare_sweep_dir_with_opts` is called against a sweep produced by Task 6, reduce the per-batch PC-anchored reports into three new report sections: coverage matrix, per-event divergence summary, perfcnt-anchored cycle deltas. Surface the `unsafe_for_pc_join` self-check from `sweep-manifest.json` as a warning at the top.

**Files:**
- Modify: `src/trace/compare.rs`

#### Step 8.1 — Test: coverage matrix format

- [x] Add a unit test that builds a synthetic two-batch report (batch 0 covers `INSTR_VECTOR`, batch 1 covers `MEMORY_STALL`; both batches have `PERF_CNT_0` grounding) and asserts the formatted report contains:

```
PC-anchored coverage:
  INSTR_VECTOR    : batch 0=swept       batch 1=absent
  MEMORY_STALL    : batch 0=absent      batch 1=swept
  PERF_CNT_0      : batch 0=grounding   batch 1=grounding
```

- [x] Run: `cargo test --lib compare::pc_anchored_coverage_matrix_formats -- --exact`
- [x] Expected: FAIL.

#### Step 8.2 — Implement aggregation

- [x] In `compare_sweep_dir_with_opts`, after collecting `batch_results: Vec<BatchResult>`:

```rust
// Build per-event aggregate from PC-anchored reports.
let mut event_per_batch: BTreeMap<String, BTreeMap<usize, &str>> = BTreeMap::new();
let mut event_total_diff: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // (set_size, multiset_magnitude)
let mut event_cycle_deltas: BTreeMap<String, Vec<i64>> = BTreeMap::new();

let grounding: BTreeSet<String> = read_grounding_from_manifest(sweep_dir);
for batch in &batch_results {
    for (key, report) in &batch.pc_anchored {
        for (name, (hw_only, emu_only)) in &report.set_diff {
            let status = if grounding.contains(name) {
                "grounding"
            } else { "swept" };
            event_per_batch.entry(name.clone()).or_default()
                .insert(batch.batch_idx, status);
            let entry = event_total_diff.entry(name.clone()).or_default();
            entry.0 += hw_only.len() + emu_only.len();
            if let Some(ms) = report.multiset_diff.get(name) {
                entry.1 += ms.values().map(|(_, _, d)| d.unsigned_abs() as usize).sum::<usize>();
            }
        }
        for (name, by_pc) in &report.cycle_bands {
            for band in by_pc.values() {
                event_cycle_deltas.entry(name.clone()).or_default().push(band.delta_cycles);
            }
        }
    }
}
```

- [x] Implement `read_grounding_from_manifest(sweep_dir: &Path) -> BTreeSet<String>` to load `sweep-manifest.json` and pull `grounding.core` ∪ `grounding.memmod` ∪ … into a set.

#### Step 8.3 — Extend `format_report` with three new sections

- [x] Add to `format_report`:

```rust
// --- PC-anchored sections (only if any batch has pc_anchored data) ---
if batch_results.iter().any(|b| !b.pc_anchored.is_empty()) {
    writeln!(out, "{}", "=".repeat(76)).ok();
    writeln!(out, "PC-anchored coverage").ok();
    writeln!(out, "{}", "=".repeat(76)).ok();
    for (name, by_batch) in &event_per_batch {
        write!(out, "  {:24}: ", name).ok();
        for (idx, status) in by_batch {
            write!(out, "batch {}={:10} ", idx, status).ok();
        }
        writeln!(out).ok();
    }
    writeln!(out).ok();

    writeln!(out, "{}", "=".repeat(76)).ok();
    writeln!(out, "PC-anchored divergences (sorted by total)").ok();
    writeln!(out, "{}", "=".repeat(76)).ok();
    let mut sorted: Vec<_> = event_total_diff.iter().collect();
    sorted.sort_by_key(|(_, (s, m))| std::cmp::Reverse(s + m));
    for (name, (set_size, multiset_mag)) in sorted {
        writeln!(out, "  {:24}: set_diff={} multiset_mag={}",
                 name, set_size, multiset_mag).ok();
    }
    writeln!(out).ok();

    writeln!(out, "{}", "=".repeat(76)).ok();
    writeln!(out, "Perfcnt-anchored cycle deltas (avg |delta_cycles| per event)").ok();
    writeln!(out, "{}", "=".repeat(76)).ok();
    for (name, deltas) in &event_cycle_deltas {
        if deltas.is_empty() { continue; }
        let avg_abs = deltas.iter().map(|d| d.unsigned_abs() as f64).sum::<f64>()
                    / deltas.len() as f64;
        let max_abs = deltas.iter().map(|d| d.unsigned_abs()).max().unwrap_or(0);
        writeln!(out, "  {:24}: avg={:.1} max={} n={}",
                 name, avg_abs, max_abs, deltas.len()).ok();
    }
    writeln!(out).ok();
}

// --- Self-check warning ---
if let Ok(manifest) = read_sweep_manifest(sweep_dir) {
    if manifest["unsafe_for_pc_join"].as_bool() == Some(true) {
        writeln!(out, "WARNING: sweep-manifest flagged unsafe_for_pc_join=true").ok();
        writeln!(out, "  reason: {}", manifest["reason"].as_str().unwrap_or("?")).ok();
        writeln!(out, "  PC-anchored cross-batch joining was skipped; results are per-batch only.").ok();
        writeln!(out).ok();
    }
}

// --- Mode-2 baseline note ---
if let Ok(baselines) = list_mode2_baselines(sweep_dir) {
    writeln!(out, "Mode-2 baselines captured (deferred to A.2b for comparison):").ok();
    for path in baselines {
        writeln!(out, "  {}", path.display()).ok();
    }
}
```

#### Step 8.4 — Run aggregation tests

- [x] Run: `cargo test --lib compare:: -- --exact`
- [x] Expected: PASS.

#### Step 8.5 — Commit

```bash
git add src/trace/compare.rs
git commit -m "feat(compare): aggregate PC-anchored reports across sweep batches

format_report gains three sections: coverage matrix (per-event,
per-batch swept | absent | grounding status), per-event divergence
summary (sorted by set+multiset magnitude), and perfcnt-anchored
cycle delta summary (avg/max abs delta per event). Surfaces the
unsafe_for_pc_join self-check from sweep-manifest.json as a top-level
warning when set, and lists captured mode-2 baselines for downstream
A.2b consumption."
```

---

### Task 9: Integration test gate on `add_one`

**Goal:** End-to-end smoke gate. Run a full mode-1 sweep on `add_one` (or whichever small bridge test currently passes both compilers), verify `unsafe_for_pc_join=false`, capture a mode-2 baseline, and produce a non-trivial PC-anchored report.

**Files:**
- Modify: `scripts/emu-bridge-test.sh` (optional — add a `--trace=pc-anchored` flag mirroring existing `--trace=sweep`) OR create `scripts/a2-gate.sh` for now.

#### Step 9.1 — Compile a traced binary in mode 1

- [x] Manually drive the pipeline once on `add_one` to confirm everything works end-to-end. From `mlir-aie/build/test/npu-xrt/add_one/`:

```bash
# Inject mode-1 + perfcnt config
python3 /home/triple/npu-work/xdna-emu/tools/mlir-trace-inject.py \
  --input ../add_one.mlir \
  --out ../add_one.traced.mlir \
  --trace-mode event_pc \
  --core-grounding "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1" \
  --perfcnt-period 1024

# Recompile
aiecc.py --xchesscc --xbridge ../add_one.traced.mlir
```

- [x] Verify the resulting `.xclbin` contains the perfcnt write32s — `xclbinutil --info --input add_one.xclbin | grep -A4 PDI` then check the CDO blob size grew compared to mode-0.

#### Step 9.2 — Run the sweep

```bash
python3 /home/triple/npu-work/xdna-emu/tools/trace-sweep.py \
  --test add_one --compiler chess \
  --tiles "0:2:core,0:2:memmod" \
  --out-dir /tmp/claude-1000/a2-gate-add_one \
  --mode event_pc \
  --core-grounding "PERF_CNT_0,INSTR_EVENT_0,INSTR_EVENT_1" \
  --memmod-grounding "PERF_CNT_0" \
  --core-sweep all --memmod-sweep all \
  --with-mode2-baseline \
  --reuse-ctx
```

- [x] Check `/tmp/claude-1000/a2-gate-add_one/sweep-manifest.json`:

```bash
cat /tmp/claude-1000/a2-gate-add_one/sweep-manifest.json | jq '.unsafe_for_pc_join'
# Expected: false
```

- [x] Check mode-2 baseline was captured:

```bash
ls /tmp/claude-1000/a2-gate-add_one/mode2-baseline/add_one/
# Expected: at least one .events.json file
```

#### Step 9.3 — Run the comparator with `--pc-anchored`

```bash
cargo build --release --bin trace-compare
./target/release/trace-compare --sweep /tmp/claude-1000/a2-gate-add_one --pc-anchored \
  -o /tmp/claude-1000/a2-gate-add_one/report.txt
```

- [x] Inspect `/tmp/claude-1000/a2-gate-add_one/report.txt`:

```bash
grep -A20 "PC-anchored" /tmp/claude-1000/a2-gate-add_one/report.txt
```

- [x] Expected: at least one event name with `set_diff > 0` listing the divergence between HW and EMU; PERF_CNT_0 in the coverage matrix as `grounding` for every batch.

#### Step 9.4 — Verify the bridge suite hasn't regressed

- [x] Run the existing bridge gate to confirm A.2 work didn't break anything mode-0 still uses:

```bash
./scripts/emu-bridge-test.sh add_one passthrough_kernel
```

- [x] Expected: PASS for both tests with both compilers.

#### Step 9.5 — Add `--trace=pc-anchored` to `emu-bridge-test.sh` (optional but useful)

- [x] Mirror the existing `--trace=sweep` machinery to add a `--trace=pc-anchored` invocation that calls `trace-sweep.py --mode event_pc` and `trace-compare --pc-anchored`. Make this opt-in; don't change the default trace mode.

#### Step 9.6 — Commit

```bash
git add scripts/emu-bridge-test.sh  # if Step 9.5 lands
git commit -m "feat(bridge): --trace=pc-anchored gate for A.2 mode-1 sweep

End-to-end mode-1 sweep + perfcnt-anchored cycle bands + mode-2
baseline, opt-in via --trace=pc-anchored. Default trace path remains
mode-0 for backwards compatibility. Manual gate run on add_one
verifies unsafe_for_pc_join=false and produces a non-trivial
PC-anchored report."
```

---

## Validation gate

A.2 is "done" when:

- [x] All unit + tool tests in this plan pass (`cargo test --lib`, `pytest tools/`).
- [x] Integration sweep on `add_one` (Step 9.2) produces `sweep-manifest.json` with `unsafe_for_pc_join=false`.
- [x] `cargo test --lib` baseline holds at 2755+ / 5 ignored throughout (the perfcnt+trace tests should add to this number).
- [x] At least one mode-2 baseline captured under `mode2-baseline/` confirming the hook works end-to-end.
- [x] Full bridge test suite (`./scripts/emu-bridge-test.sh`) has not regressed from the pre-A.2 baseline.

---

## Self-Review Notes

A self-review pass against the spec finds the following coverage:

| Spec section | Tasks |
|--------------|-------|
| §1 Inject (mlir-trace-inject.py) | Task 4 |
| §2a Mode-1 encoder | Task 2 |
| §2b Perfcnt | Task 1 (wiring; perfcnt itself already implemented) |
| §2c PC threading | Task 3 |
| §3 Sweep orchestration | Task 6 (incl. mode-2 baseline + grounding-PC self-check) |
| §3 Multi-tile patcher | Task 5 |
| §4 Compare logic | Task 7 |
| §5 Sweep aggregation | Task 8 |
| Validation gate | Task 9 |

No type-name drift detected. `PCAnchoredReport`, `CycleBand`, `compare_pc_anchored_for_tile`, `_build_lockstep_batches`, `_check_grounding_pc_invariance`, `event_pc()` are referenced consistently across tasks. The signature change for `notify_event` / `notify_*_trace_event` (adding `pc: Option<u32>`) lands in Task 2 and is propagated through Task 3 — coordinated rename, with a checklist in Step 2.5 + Step 3.4 so no call site is missed.

Mode-1 encoder in Task 2 references `tools/trace_decoder/modes/mode1.py` for the byte format and the `mode1_mixed_r0_core_expected.json` fixture as the oracle. The fixture has been read and matches the encoded layout described.

The discarded-`Vec<usize>` gap from `coordinator.rs:1027-1038` is closed in Task 1; subsequent tasks build on the assumption that PERF_CNT_N events now propagate.

Risk note: Task 3's PC-source enumeration in coordinator.rs is the most fragile piece. If a new event-fire site is added between this plan being written and execution, the auditor must add it to the table in Step 3.3. A grep `grep -rn "notify_core_trace_event\|notify_mem_trace_event" src/` at execution time catches any new sites.
