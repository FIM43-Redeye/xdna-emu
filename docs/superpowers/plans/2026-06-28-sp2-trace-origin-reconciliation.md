# SP-2: Trace-Origin Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make each traced module's trace Start-frame absolute timestamp carry its broadcast-propagation skew, so cross-domain trace timestamps reproduce the modeled BROADCAST_15 timer offset, without changing any within-domain delta.

**Architecture:** The trace byte stream has exactly one absolute timestamp (the Start frame, `encode_start`); every other frame is a delta. Add a static `origin_offset: u64` to `TraceUnit`, applied **only** when encoding that one absolute (`timer.wrapping_add(self.origin_offset)`), leaving all cycle bookkeeping in the raw frame. The broadcast flood (`propagate_broadcasts_with_timing`, SP-1) sets the offset to the tile timer's own reset value (`max_delay - module_delay`, the `core_target`/`mem_target` locals it already computes), so the trace and timer use an identical per-tile term.

**Tech Stack:** Rust; `cargo test --lib`. Files: `src/device/trace_unit/mod.rs`, `src/device/trace_unit/tests.rs`, `src/device/state/effects.rs`.

**Design of record:** `docs/superpowers/specs/2026-06-28-sp2-trace-origin-reconciliation-design.md`. Parent arc: `docs/superpowers/specs/2026-06-28-timer-sync-faithful-broadcast-arc.md` (SP-2).

## Global Constraints

- **Within-domain deltas stay byte-identical.** Only the Start frame's 7 absolute bytes may change. (Spec Sec.1, Sec.4.4.)
- **Exact byte-neutrality at zero constants.** With all `BroadcastTiming` consts 0, every `origin_offset == 0` and output is byte-for-byte identical to today. (Spec Sec.4.4.)
- **The offset is `max_delay - module_delay`** = the `core_target`/`mem_target` SP-1 already computes at `effects.rs:585-586`. Reuse those locals; introduce no new arithmetic. (Spec Sec.4.2.)
- **Apply the offset only in `encode_start`.** Do not touch `notify_event`, `set_event_level`, `commit_cycle`, or any other cycle-bearing method. (Spec Sec.3, Sec.4.3.)
- **Set the offset before the broadcast notify** in the flood loop, so a tile whose `start_event` equals the broadcast id arms with the offset already applied. (Spec Sec.4.5.)
- **No emoji anywhere.** Comments explain *why*, not *what*. (Project CLAUDE.md.)
- `cargo test --lib` must pass after each task.

---

### Task 1: TraceUnit carries and applies an origin offset

Add the field, a setter (production) and getter (test-only), and apply the offset at the single absolute-encode point. `reset()` is `*self = Self::new(...)`, so initializing the field to 0 in `new()` makes a full reset clear it; the Trace_Control0-rewrite path (`mod.rs:470-484`) clears fields individually and does not list `origin_offset`, so it persists across per-batch reconfig. Both behaviors are locked by a test.

**Files:**
- Modify: `src/device/trace_unit/mod.rs` (struct field ~`:159`, `new()` ~`:285`, new setter/getter after `reset()` ~`:322`, `encode_start` `:1344-1354`)
- Test: `src/device/trace_unit/tests.rs`

**Interfaces:**
- Produces: `TraceUnit::set_origin_offset(&mut self, offset: u64)` (pub(crate)); `TraceUnit::origin_offset(&self) -> u64` (`#[cfg(test)]` pub(crate)); `origin_offset: u64` field, default 0, added to the encoded Start absolute.
- Consumes: nothing new (existing `TraceUnit::new`, `write_register`, `reset`).

- [ ] **Step 1: Write the failing tests**

Add to the end of `src/device/trace_unit/tests.rs` (the module already has `use super::*;` and the `force_start` / `notify_commit` helpers):

```rust
#[test]
fn origin_offset_shifts_only_the_start_absolute() {
    // Build the same stream twice, differing only in origin_offset. The Start
    // frame's 7 absolute bytes must shift by exactly the offset; every other
    // byte (deltas, event frames) must be identical -- the within-domain
    // byte-identity invariant.
    fn stream(offset: u64) -> Vec<u8> {
        let mut tu = TraceUnit::new(0, 2);
        // mode=EventTime, start=28, stop=29 (same idiom as test_register_configuration).
        tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
        tu.write_register(0x10, 37); // slot 0 = event 37
        tu.set_origin_offset(offset);
        force_start(&mut tu, 28); // arm at cycle 0 -> Start absolute = 0 + offset
        notify_commit(&mut tu, 37, 5); // one event at cycle 5 (small: whole stream stays in byte_buffer)
        tu.encoded_bytes().to_vec()
    }
    let base = stream(0);
    let shifted = stream(64);
    assert_eq!(base.len(), shifted.len(), "offset must not change stream length");
    assert_eq!(base[0] & 0xF0, 0xF0, "Start marker prefix present");
    for i in 0..base.len() {
        if (1..=7).contains(&i) {
            continue; // the 7 Start-absolute bytes are allowed to differ
        }
        assert_eq!(base[i], shifted[i], "byte {i} changed outside the Start absolute");
    }
    let decode_start = |b: &[u8]| (0..7).fold(0u64, |v, i| (v << 8) | b[1 + i] as u64);
    assert_eq!(decode_start(&base), 0, "zero-offset Start absolute == raw arm cycle (0)");
    assert_eq!(
        decode_start(&shifted),
        decode_start(&base) + 64,
        "Start absolute must shift by exactly the offset"
    );
}

#[test]
fn origin_offset_persists_across_control0_rewrite_and_clears_on_reset() {
    let mut tu = TraceUnit::new(0, 2);
    tu.set_origin_offset(42);
    assert_eq!(tu.origin_offset(), 42);
    // Rewriting Trace_Control0 (per-batch reconfig) clears trace bookkeeping
    // but NOT the broadcast-network origin offset.
    tu.write_register(0x00, 0 | (28 << 16) | (29 << 24));
    assert_eq!(tu.origin_offset(), 42, "Control0 rewrite must preserve origin offset");
    // A full reset returns to the power-on default.
    tu.reset();
    assert_eq!(tu.origin_offset(), 0, "reset must clear origin offset");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib origin_offset`
Expected: FAIL to compile -- `no method named set_origin_offset` / `origin_offset` found for `TraceUnit`.

- [ ] **Step 3: Add the struct field**

In `src/device/trace_unit/mod.rs`, immediately after the `timer: u64,` field (line 159):

```rust
    /// Broadcast-propagation origin offset (SP-2). Added to the Start frame's
    /// absolute timestamp by `encode_start` so cross-domain trace timestamps
    /// carry the modeled BROADCAST_15 timer skew. Equals the tile timer's
    /// reset baseline (`max_delay - module_delay`) set by the flood. Default 0
    /// = behavior-neutral. Applied ONLY in `encode_start`; every delta
    /// bookkeeping value stays in the raw cycle frame.
    origin_offset: u64,
```

- [ ] **Step 4: Initialize it in `new()`**

In `TraceUnit::new` (after `timer: 0,`, line 285):

```rust
            origin_offset: 0,
```

- [ ] **Step 5: Add the setter and test-only getter**

In `src/device/trace_unit/mod.rs`, immediately after `reset()` (after line 322):

```rust
    /// Set the broadcast-propagation origin offset (SP-2), added to the Start
    /// frame's absolute timestamp by `encode_start`. Called by the broadcast
    /// flood with the tile timer's reset value (`max_delay - module_delay`).
    /// Persists across Trace_Control0 rewrites (it is broadcast-network config,
    /// not trace config); cleared by `reset`.
    pub(crate) fn set_origin_offset(&mut self, offset: u64) {
        self.origin_offset = offset;
    }

    /// Read the current origin offset. Test-only accessor.
    #[cfg(test)]
    pub(crate) fn origin_offset(&self) -> u64 {
        self.origin_offset
    }
```

- [ ] **Step 6: Apply the offset in `encode_start`**

In `src/device/trace_unit/mod.rs`, `encode_start` (`:1344`), insert the offset application after `self.byte_buffer.push(prefix);` and before the 7-byte loop:

```rust
    fn encode_start(&mut self, timer: u64) {
        let prefix = match self.mode {
            TraceMode::EventPc => 0xF1u8,
            _ => 0xF0u8,
        };
        self.byte_buffer.push(prefix);
        // The Start frame is the trace's single absolute timestamp. Add the
        // broadcast-propagation origin offset (SP-2) so cross-domain timestamps
        // carry the modeled timer skew; every other frame is a delta and stays
        // in the raw cycle frame. The offset is the tile timer's own reset
        // baseline (max_delay - module_delay), so trace and timer agree.
        let timer = timer.wrapping_add(self.origin_offset);
        // 7 bytes of timer, big-endian (56 bits)
        for i in (0..7).rev() {
            self.byte_buffer.push(((timer >> (i * 8)) & 0xFF) as u8);
        }
    }
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test --lib origin_offset`
Expected: PASS (`origin_offset_shifts_only_the_start_absolute`, `origin_offset_persists_across_control0_rewrite_and_clears_on_reset`).

- [ ] **Step 8: Run the full library suite (no regression)**

Run: `cargo test --lib`
Expected: PASS, same count as before plus the two new tests.

- [ ] **Step 9: Commit**

```bash
git add src/device/trace_unit/mod.rs src/device/trace_unit/tests.rs
git commit -m "feat(#140): SP-2 TraceUnit origin_offset applied at encode_start

Add a static origin_offset to TraceUnit, applied only to the Start
frame's absolute timestamp (the trace's single absolute), leaving all
delta bookkeeping in the raw cycle frame. Default 0 is byte-neutral;
persists across Trace_Control0 rewrites, cleared by reset.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

---

### Task 2: The broadcast flood sets the trace origin offset

Wire SP-1's flood to push each reached module's reset value into its trace unit, before the broadcast notify so a tile armed by the broadcast event already has the offset.

**Files:**
- Modify: `src/device/state/effects.rs` (`propagate_broadcasts_with_timing`, insert before the notify at `:602`)
- Test: `src/device/state/effects.rs` (new `#[cfg(test)] mod` at end of file, mirroring `broadcast_flood_timing_tests`)

**Interfaces:**
- Consumes: `TraceUnit::set_origin_offset` and `TraceUnit::origin_offset` (Task 1); the existing `core_target`/`mem_target` u64 locals at `effects.rs:585-586`; `EventModuleType::Core.broadcast_event_base()`.
- Produces: nothing for later tasks (terminal task).

- [ ] **Step 1: Write the failing tests**

Append to `src/device/state/effects.rs` a new test module:

```rust
#[cfg(test)]
mod broadcast_origin_offset_tests {
    use super::*;
    use crate::device::events::EventModuleType;

    #[test]
    fn flood_sets_trace_origin_offset_cross_tile() {
        // Two tiles one vertical hop apart take origin offsets differing by the
        // hop skew d_v, independent of max_delay. (offset = max_delay - origin_d;
        // source origin_d=0, hop origin_d=d_v.)
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let src = (0u8, 2u8);
        let hop = (0u8, 3u8); // one vertical hop north
        dev.array.get_mut(src.0, src.1).unwrap().pending_broadcasts.push(channel);
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 4, 0, 0); // d_v=4
        let off_src = dev.array.get(src.0, src.1).unwrap().core_trace.origin_offset();
        let off_hop = dev.array.get(hop.0, hop.1).unwrap().core_trace.origin_offset();
        assert_eq!(off_src - off_hop, 4, "cross-tile origin offset diff == d_v");
    }

    #[test]
    fn flood_sets_trace_origin_offset_intra_tile_asymmetry() {
        // On one tile, core and mem trace units differ by the intra-tile
        // pipeline asymmetry: core_target - mem_target = mem_off - core_off.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let src = (0u8, 2u8);
        dev.array.get_mut(src.0, src.1).unwrap().pending_broadcasts.push(channel);
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 2, 4); // core_off=2, mem_off=4
        let tile = dev.array.get(src.0, src.1).unwrap();
        let core = tile.core_trace.origin_offset();
        let mem = tile.mem_trace.origin_offset();
        assert_eq!(core - mem, 2, "core vs mem origin offset diff == mem_off - core_off");
    }

    #[test]
    fn flood_sets_origin_offset_before_arming_reached_trace() {
        // Configure the source core trace to START on the broadcast event id, so
        // the flood's own notify (effects.rs:602) arms it. If set_origin_offset
        // ran AFTER that notify, the Start would encode offset 0. A fresh device
        // has array.current_cycle == 0, so the arm cycle is 0 and the Start
        // absolute equals the offset.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel;
        let src = (0u8, 2u8);
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .core_trace
            .write_register(0x00, (bcast_id as u32) << 16); // start_event = bcast_id
        dev.array.get_mut(src.0, src.1).unwrap().pending_broadcasts.push(channel);
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 2, 4); // core_off=2 -> nonzero offset
        let tile = dev.array.get(src.0, src.1).unwrap();
        let off = tile.core_trace.origin_offset();
        assert!(off > 0, "source core trace must receive a nonzero offset");
        let bytes = tile.core_trace.encoded_bytes();
        assert_eq!(bytes[0] & 0xF0, 0xF0, "Start marker emitted by the flood's arming notify");
        let start_abs = (0..7).fold(0u64, |v, i| (v << 8) | bytes[1 + i] as u64);
        assert_eq!(start_abs, off, "Start absolute carries the offset (set before arm)");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib broadcast_origin_offset`
Expected: FAIL -- the reached tiles' `origin_offset()` returns 0 (the flood does not set it yet): `assertion failed: off > 0` and the diff asserts return `0`.

- [ ] **Step 3: Set the offset in the flood loop**

In `src/device/state/effects.rs`, `propagate_broadcasts_with_timing`, insert the two setter calls immediately before the existing notify calls (between the `match tile.tile_kind { ... }` that produces `core_hw_id`/`mem_hw_id` and the `tile.notify_core_trace_event_with_target(...)` at `:602`):

```rust
                // SP-2: give the trace units the same skew baseline the timer
                // holds (core_target/mem_target = max_delay - module_delay). Set
                // BEFORE the notify below so a tile whose start_event is this
                // broadcast id arms with the offset already applied (design Sec.4.5).
                tile.core_trace.set_origin_offset(core_target);
                tile.mem_trace.set_origin_offset(mem_target);
                tile.notify_core_trace_event_with_target(core_hw_id, current_cycle, core_pc, core_target);
                tile.notify_mem_trace_event_with_target(mem_hw_id, current_cycle, None, mem_target);
```

(The original two `notify_*` lines are unchanged; the two `set_origin_offset` lines are added directly above them.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib broadcast_origin_offset`
Expected: PASS (all three tests).

- [ ] **Step 5: Run the full library suite (no regression, neutrality intact)**

Run: `cargo test --lib`
Expected: PASS. In particular the existing `flood_is_behavior_neutral_at_zero_delays` and all trace-unit tests stay green -- at zero consts every `core_target`/`mem_target` is 0, so `set_origin_offset(0)` is a no-op.

- [ ] **Step 6: Commit**

```bash
git add src/device/state/effects.rs
git commit -m "feat(#140): SP-2 broadcast flood sets trace origin offset

propagate_broadcasts_with_timing pushes each reached module's reset
value (core_target/mem_target = max_delay - module_delay) into its trace
unit, before the broadcast notify so a tile armed by the broadcast event
already carries the offset. Trace Start-frame absolutes now reflect the
modeled BROADCAST_15 skew; zero consts stay byte-neutral.

Generated using Claude Code.
Claude-Session: https://claude.ai/code/session_012P8xnhCsbxDDE462FAvGRh"
```

- [ ] **Step 7: Integration confirmation (origin-invariant trace sweep)**

The unit tests above are the positive gate (the sweep re-anchors per tile and is blind to the cross-tile shift by construction -- spec Sec.5/Sec.6). As a no-regression confirmation that within-domain deltas are unchanged, run the HW-free trace sweep once after both tasks land:

Run: `./scripts/emu-bridge-test.sh --no-hw --sweep -v add_one`
Expected: `TRACE_VERDICT` unchanged from the pre-SP-2 baseline (no new divergences/mismatches introduced). This is a heavier gate; run it once at the end, not per task. (Note: emu-only, so no NPU contention; do not run concurrently with a HW suite.)

---

## Notes for the implementer

- **Do not** add the offset anywhere but `encode_start`. Subtracting at `notify_event`/`set_event_level`/`commit_cycle` was the rejected first design (it saturates at config-time start and entangles with `commit_cycle`); see design Sec.3.
- **Do not** introduce new arithmetic for the offset -- it is exactly the `core_target`/`mem_target` already computed at `effects.rs:585-586`.
- `encode_mode2_start` (PC-anchored, `:1407`) takes no cycle and must NOT be touched; PC-anchored traces have no cycle origin to skew.
- The config-time source-tile `Event_Generate` path (`effects.rs:396-397`) runs before the flood and so cannot carry the offset; this is benign because the timer-sync trigger event is never a real trace `start_event` (design Sec.4.5, bullet 2). It is out of scope for these tests.
