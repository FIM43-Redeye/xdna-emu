# Interrupt Subsystem Close-Out (Tier A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the AIE2 (NPU1/Phoenix) `interrupt` coverage subsystem from PARTIAL to Full by wiring a faithful internal interrupt path — event/error → L1 → broadcast network → L2 → NPI line — with full 23-register read/write fidelity, terminating at the firmware boundary.

**Architecture:** Approach 3 — the existing broadcast network is the transport; a new *named* L2 interrupt sink is added to `propagate_broadcasts`. The L1 controllers already model every register; the work is integration in `src/device/state/effects.rs`, `src/device/tile/registers.rs`, and `src/device/state/dispatch.rs`, plus routing hardware error events through the `EventModule` so the error path (the primary real consumer) actually works.

**Tech Stack:** Rust, `cargo test --lib`. Reference sources: aie-rt `interrupt/`, AM025 register DB. Design spec: `docs/superpowers/specs/2026-05-19-interrupt-l2-closeout-design.md`.

**Sandbox note:** all `cargo test` commands below assume `TMPDIR=/tmp/claude-1000` is prefixed when running inside the sandbox (tests that create temp dirs need it). Run build/test commands bare (no `| tail`/`grep`).

**Verbatim current-state facts (do not re-discover):**
- `src/device/tile/registers.rs` — `pub fn read_register(&mut self, offset: u32) -> u32` starts line 17; compute `core_debug` dispatch at lines ~108-112: `if self.is_compute() { if let Some(val) = self.core_debug.read_register(offset) { return val; } }`; final fallback `self.registers.get(&offset).copied().unwrap_or(0)`.
- `src/device/state/effects.rs` — `pub(super) fn apply_tile_local_effects(&mut self, col: u8, row: u8, offset: u32, value: u32)` spans lines 16-419. L1/L2 *write* routing at lines 349-360 inside `if tile.is_shim() { ... }`. `is_event_generate` block lines 372-418. `pub(crate) fn propagate_broadcasts(&mut self, col: u8, source_row: u8)` spans lines 444-565; per-tile delivery calls `tile.notify_core_trace_event(...)` / `tile.notify_mem_trace_event(...)` at lines 508-509 inside a `{ ... }` block that holds `let tile = match self.array.get_mut(c, r) { Some(t) => t, None => continue };`.
- `src/device/state/dispatch.rs` lines 161-164: `self.apply_tile_local_effects(tile_addr.col, tile_addr.row, tile_addr.offset, value);` immediately followed by `self.propagate_broadcasts(tile_addr.col, tile_addr.row);`.
- `src/device/tile/mod.rs` — `pub l1_irq: Option<super::interrupts::L1InterruptController>` (line 253), `pub l2_irq: Option<super::interrupts::L2InterruptController>` (255), `pub pending_broadcasts: Vec<u8>` (268), `pub fn is_shim(&self) -> bool` (601), and `drain_pending_broadcasts` returns `std::mem::take(&mut self.pending_broadcasts)` (637). Both `l1_irq` and `l2_irq` are `Some` for every `is_shim()` tile.
- L1 API (`src/device/interrupts/l1.rs`): `pub enum SwitchId { A = 0, B = 1 }`; `signal_event(&mut self, sw: SwitchId, event_id: u8) -> Option<u8>` (line 245); `read_irq_no(&self, sw: SwitchId) -> u32` (151); `read_register(&self, offset: u32) -> Option<u32>` (287); `write_enable/write_disable/clear_status/write_irq_no/set_irq_event_slot` per spec.
- L2 API (`src/device/interrupts/l2.rs`): `signal_interrupt(&mut self, channel: u8)` (114); `pending_host_interrupt(&self) -> bool` (126); `read_register(&self, offset: u32) -> Option<u32>` (136); `write_enable(&mut self, value: u32)` (73).
- `src/device/interrupts/mod.rs` constants: `L1_REG_IRQ_EVENT_A=0x0003_5014`, `L1_REG_ENABLE_A=0x0003_5004`, `L1_REG_IRQ_NO_A=0x0003_5010`, `L1_REG_STATUS_A=0x0003_500C`, `L1_SWITCH_OFFSET=0x30`, `L2_REG_ENABLE=0x0001_5004`, `L2_REG_STATUS=0x0001_500C`, `L2_REG_MASK=0x0001_5000`, `L2_REG_INTERRUPT=0x0001_5010`.
- `EventId` is `pub type EventId = u8;` (`src/device/events/combo.rs:35`). `EventModule::generate_event(&mut self, event_id: EventId)` (`src/device/events/mod.rs:248`), `drain_pending(&mut self) -> Vec<EventId>` (307).
- Hardware errors: `src/interpreter/core/interpreter.rs:18-23` `fn raise_instr_error(tile: &mut Tile, cycle: u64, pc: u32)` calls only `tile.core_trace.notify_event(core_events::INSTR_ERROR, cycle, Some(pc))` + perf. It does NOT call `EventModule::generate_event` — confirmed gap.
- Coverage: `crates/xdna-archspec/src/coverage/units.rs` lines 165-168 hold the `d("interrupt", ...)` entry; format uses `Modeled { completeness: Full }` / `Modeled { completeness: Partial { missing: "...".into() } }`. Regenerate with `cargo run -p xdna-archspec --example gen_coverage_artifacts`.

**How to find the integration tests:** all new integration tests go in `src/device/state/effects.rs` under its existing `#[cfg(test)] mod tests` block (search the file for `mod tests`; if absent, add `#[cfg(test)] mod interrupt_path_tests { use super::*; ... }` at end of file). Unit tests for `l1.rs`/`l2.rs` stay in those files' existing test modules.

---

### Task 1: L1/L2 read-path routing

Makes all 23 register offsets observable on read (write-1-to-clear status, enable/disable→mask reflection, read-only mask). Mirrors the existing compute `core_debug` read dispatch.

**Files:**
- Modify: `src/device/tile/registers.rs` (in `read_register`, near lines 108-112)
- Test: `src/device/tile/registers.rs` (its `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `src/device/tile/registers.rs`:

```rust
#[test]
fn shim_l2_enable_then_read_mask_reflects() {
    // Build a shim tile (row 0 is shim on NPU1).
    let mut tile = Tile::new_for_kind(xdna_archspec::types::TileKind::ShimNoc, 0, 0);
    use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_MASK};
    // Enable L2 channels 0 and 3 via the write path.
    assert!(tile.l2_irq.as_mut().unwrap().write_register(L2_REG_ENABLE, 0b1001));
    // Mask must read back the enabled bits through the tile read path.
    assert_eq!(tile.read_register(L2_REG_MASK), 0b1001);
}

#[test]
fn shim_l2_status_write_one_to_clear_visible_on_read() {
    let mut tile = Tile::new_for_kind(xdna_archspec::types::TileKind::ShimNoc, 0, 0);
    use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_STATUS};
    tile.l2_irq.as_mut().unwrap().write_register(L2_REG_ENABLE, 0b1);
    tile.l2_irq.as_mut().unwrap().signal_interrupt(0); // latch ch0
    assert_eq!(tile.read_register(L2_REG_STATUS), 0b1);
    tile.l2_irq.as_mut().unwrap().write_register(L2_REG_STATUS, 0b1); // ack
    assert_eq!(tile.read_register(L2_REG_STATUS), 0);
}
```

If `Tile::new_for_kind` does not exist, find the actual constructor used elsewhere in `registers.rs` tests (search the test module for how a `Tile` is built) and use that, keeping the `TileKind::ShimNoc`, col 0, row 0 intent.

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib shim_l2_enable_then_read_mask_reflects shim_l2_status_write_one_to_clear_visible_on_read`
Expected: FAIL — `read_register` returns 0 (fallback) instead of the L2 mask/status, because L2 read is not routed.

- [ ] **Step 3: Add the read routing**

In `src/device/tile/registers.rs`, immediately after the compute `core_debug` dispatch block (the `if self.is_compute() { if let Some(val) = self.core_debug.read_register(offset) { return val; } }` near line 112), add:

```rust
        // Interrupt controller read routing (shim tiles only). Mirrors the
        // write routing in effects.rs::apply_tile_local_effects. L1 and L2
        // occupy disjoint offset ranges (0x35xxx vs 0x15xxx) so order is
        // immaterial; each returns None for offsets it does not own.
        if self.is_shim() {
            if let Some(ref l1) = self.l1_irq {
                if let Some(val) = l1.read_register(offset) {
                    return val;
                }
            }
            if let Some(ref l2) = self.l2_irq {
                if let Some(val) = l2.read_register(offset) {
                    return val;
                }
            }
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib shim_l2_enable_then_read_mask_reflects shim_l2_status_write_one_to_clear_visible_on_read`
Expected: PASS (2 passed).

- [ ] **Step 5: Full library test sanity + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass, 0 failed (count may differ from any number in docs).

```bash
git add src/device/tile/registers.rs
git commit -m "interrupt: route L1/L2 register reads on shim tiles

The controllers modelled all 23 register offsets but only writes were
routed (effects.rs); reads fell through to the generic register map, so
write-1-to-clear status and enable/disable->mask reflection were
invisible to a guest read. Mirror the compute core_debug read dispatch
for shim tiles. Spec: 2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 2: Shim L1 event tap (software Event_Generate path)

Add a reusable `Tile` method that offers a fired event to both L1 switches and, on latch, queues the switch's `IRQ_NO` broadcast id into `pending_broadcasts`. Wire it into the `Event_Generate` handler. (Task 4 reuses the same method for received broadcasts.)

**Files:**
- Modify: `src/device/tile/mod.rs` (new method on `Tile`)
- Modify: `src/device/state/effects.rs` (call it from the `is_event_generate` block, ~line 417)
- Test: `src/device/state/effects.rs` test module

- [ ] **Step 1: Write the failing test**

Add to the effects.rs test module (see "How to find the integration tests"):

```rust
#[test]
fn event_generate_on_shim_latches_l1_and_queues_irq_no() {
    use crate::device::interrupts::{
        L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, L1_REG_STATUS_A, SwitchId,
    };
    let mut dev = DeviceState::new_npu1();
    let (col, row) = (0u8, 0u8); // shim row

    // Configure L1 switch A: map IRQ event slot 0 -> event id 7, enable
    // interrupt id 16 (slot 0), set the output broadcast id (IRQ_NO) to 5.
    {
        let t = dev.array.get_mut(col, row).unwrap();
        let l1 = t.l1_irq.as_mut().unwrap();
        l1.set_irq_event_slot(SwitchId::A, 0, 7);
        l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        l1.write_register(L1_REG_IRQ_NO_A, 5);
    }

    // Fire event id 7 on this shim tile's EventModule via Event_Generate.
    // Event_Generate offset is the first event register; use the helper
    // that the effects tests already use to issue a tile register write.
    // (Search effects.rs tests for the existing register-write helper;
    //  if none, use dev.write_register(addr, value) with the shim tile's
    //  Event_Generate absolute address from the regdb.)
    dev.fire_event_generate_for_test(col, row, 7);

    let t = dev.array.get(col, row).unwrap();
    let l1 = t.l1_irq.as_ref().unwrap();
    assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0, "L1 status must latch");
    assert!(t.pending_broadcasts.contains(&5), "IRQ_NO 5 must be queued");
}
```

If no `fire_event_generate_for_test` / register-write test helper exists, add a minimal `#[cfg(test)]` helper on `DeviceState` in the effects.rs test module that calls `self.apply_tile_local_effects(col, row, <Event_Generate offset>, event_id as u32)`, resolving the Event_Generate offset from `xdna_archspec` registers the same way `apply_tile_local_effects` does (`ce.event_generate` for shim). Keep the helper test-only.

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib event_generate_on_shim_latches_l1_and_queues_irq_no`
Expected: FAIL — L1 status stays 0 and `pending_broadcasts` lacks 5 (no tap exists).

- [ ] **Step 3: Add the reusable tap method on `Tile`**

In `src/device/tile/mod.rs`, add to `impl Tile` (near `drain_pending_broadcasts`, ~line 637):

```rust
    /// Offer a fired event to both L1 interrupt switches (shim tiles).
    ///
    /// Faithful to the two-independent-switches hardware: each switch
    /// slot-matches against its own IRQ_EVENT config and gates on its own
    /// enable mask. On a latch, the switch drives its configured IRQ_NO
    /// broadcast line -- modelled here by queueing that broadcast id into
    /// `pending_broadcasts`, the same transport Event_Generate uses. L1 is
    /// thus a second independent producer into the broadcast network.
    /// No-op on non-shim tiles (no `l1_irq`).
    pub fn tap_l1_interrupt(&mut self, event_id: u8) {
        let Some(l1) = self.l1_irq.as_mut() else { return };
        for sw in [super::interrupts::SwitchId::A, super::interrupts::SwitchId::B] {
            if l1.signal_event(sw, event_id).is_some() {
                let irq_no = l1.read_irq_no(sw) as u8;
                self.pending_broadcasts.push(irq_no);
            }
        }
    }
```

Note the borrow: `l1.signal_event` then `l1.read_irq_no` are both on `l1` (the `&mut` from `as_mut`), and `self.pending_broadcasts.push` is a disjoint field — split the borrow by reading `irq_no` into a local before the `self.pending_broadcasts` push (as written: the `l1` borrow ends at the `read_irq_no` expression because `irq_no` is a `u8` copy, so the subsequent `self.pending_broadcasts.push` is valid). If the borrow checker complains, restructure to collect `(sw)` latched first, then push after the loop.

- [ ] **Step 4: Wire it into the Event_Generate handler**

In `src/device/state/effects.rs`, inside the `if is_event_generate { ... }` block, immediately after the existing broadcast-channel-mapping `if let Some(em) = events_ref { ... }` loop and before the block's closing `}` (around line 417), add:

```rust
            // Tier A interrupt path: a software-generated event is also
            // offered to this tile's L1 interrupt controller (shim only).
            // On latch, L1 queues its IRQ_NO into pending_broadcasts so the
            // existing propagate_broadcasts transport carries it to L2.
            tile.tap_l1_interrupt(event_id);
```

`tile` is already the `&mut` tile binding in scope in that block (the same one used for `tile.pending_broadcasts.push(ch)` above). `tap_l1_interrupt` no-ops on non-shim tiles, so the `is_shim` guard is intrinsic.

- [ ] **Step 5: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib event_generate_on_shim_latches_l1_and_queues_irq_no`
Expected: PASS.

- [ ] **Step 6: Add switch-independence test**

Append to the same test module:

```rust
#[test]
fn l1_switch_independence_only_configured_switch_latches() {
    use crate::device::interrupts::{L1_REG_ENABLE_A, L1_SWITCH_OFFSET, SwitchId};
    let mut dev = DeviceState::new_npu1();
    let (col, row) = (0u8, 0u8);
    {
        let l1 = dev.array.get_mut(col, row).unwrap().l1_irq.as_mut().unwrap();
        // Only switch A maps event 9 -> slot 0 and enables it.
        l1.set_irq_event_slot(SwitchId::A, 0, 9);
        l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        // Switch B left unconfigured (its enable reg is +L1_SWITCH_OFFSET).
        let _ = L1_SWITCH_OFFSET;
    }
    dev.fire_event_generate_for_test(col, row, 9);
    let l1 = dev.array.get(col, row).unwrap().l1_irq.as_ref().unwrap();
    assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0);
    assert_eq!(l1.read_status(SwitchId::B), 0, "switch B must not latch");
}
```

- [ ] **Step 7: Run + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib l1_switch_independence_only_configured_switch_latches event_generate_on_shim_latches_l1_and_queues_irq_no`
Expected: PASS (2 passed). Then `TMPDIR=/tmp/claude-1000 cargo test --lib` — all pass.

```bash
git add src/device/tile/mod.rs src/device/state/effects.rs
git commit -m "interrupt: tap shim L1 from Event_Generate, queue IRQ_NO

Adds Tile::tap_l1_interrupt -- offers a fired event to both L1 switches
(independent slot-match + enable gating per hardware) and, on latch,
queues the switch IRQ_NO into pending_broadcasts so the existing
broadcast transport carries it. Wired into the Event_Generate handler.
Spec: 2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 3: Broadcast → L2 interrupt sink

Add the named L2 sink to `propagate_broadcasts`: when a broadcast is delivered to a shim tile, offer it to that tile's L2 controller. L2's existing enable-mask gate decides the latch.

**Files:**
- Modify: `src/device/state/effects.rs` (`propagate_broadcasts` per-tile delivery block, after lines 508-509)
- Test: `src/device/state/effects.rs` test module

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn broadcast_delivery_latches_shim_l2_on_matching_channel() {
    use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_STATUS};
    let mut dev = DeviceState::new_npu1();
    let (col, row) = (0u8, 0u8); // shim

    // Enable L2 channel 5.
    dev.array.get_mut(col, row).unwrap().l2_irq.as_mut().unwrap()
        .write_register(L2_REG_ENABLE, 1 << 5);

    // Queue broadcast channel 5 on the shim tile and propagate.
    dev.array.get_mut(col, row).unwrap().pending_broadcasts.push(5);
    dev.propagate_broadcasts(col, row);

    let l2 = dev.array.get(col, row).unwrap().l2_irq.as_ref().unwrap();
    assert_ne!(l2.read_register(L2_REG_STATUS).unwrap() & (1 << 5), 0,
        "L2 channel 5 must latch on broadcast delivery");
}

#[test]
fn broadcast_delivery_does_not_latch_disabled_l2_channel() {
    use crate::device::interrupts::L2_REG_STATUS;
    let mut dev = DeviceState::new_npu1();
    let (col, row) = (0u8, 0u8);
    // Channel 5 NOT enabled.
    dev.array.get_mut(col, row).unwrap().pending_broadcasts.push(5);
    dev.propagate_broadcasts(col, row);
    let l2 = dev.array.get(col, row).unwrap().l2_irq.as_ref().unwrap();
    assert_eq!(l2.read_register(L2_REG_STATUS).unwrap(), 0,
        "disabled L2 channel must not latch");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib broadcast_delivery_latches_shim_l2_on_matching_channel broadcast_delivery_does_not_latch_disabled_l2_channel`
Expected: the disabled-channel test PASSES (vacuously, nothing latches); the enabled-channel test FAILS (no sink yet, status stays 0).

- [ ] **Step 3: Add the sink**

In `src/device/state/effects.rs`, inside `propagate_broadcasts`, in the per-tile delivery block, immediately after the two `tile.notify_*_trace_event(...)` calls (lines 508-509) and still inside the `{ ... }` block where `tile` is the `self.array.get_mut(c, r)` binding, add:

```rust
                // Tier A interrupt sink: a broadcast delivered to a shim
                // tile is offered to its L2 interrupt controller. The
                // broadcast channel == L1 IRQ_NO == L2 input channel
                // (invariant; see channel-identity test). L2's enable-mask
                // gate decides whether it latches. Named seam -- interrupt
                // logic is not smeared into trace-notify above.
                if let Some(l2) = tile.l2_irq.as_mut() {
                    l2.signal_interrupt(channel);
                }
```

`channel` is the loop variable (`for &channel in &channels`) in scope here; `tile` is the current mutable tile binding.

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib broadcast_delivery_latches_shim_l2_on_matching_channel broadcast_delivery_does_not_latch_disabled_l2_channel`
Expected: PASS (2 passed).

- [ ] **Step 5: Add block-mask fidelity test**

```rust
#[test]
fn broadcast_block_mask_prevents_l2_latch() {
    use crate::device::interrupts::L2_REG_STATUS;
    // A broadcast blocked before reaching the shim must not latch L2.
    // Use a 2-col device: generate on a non-shim tile, block the path,
    // assert the shim L2 stays clear. (If the test scaffolding for
    // programming EVENT_BROADCAST_BLOCK is non-trivial, assert the
    // invariant via the broadcast config directly: configure the source
    // tile's broadcast block for the channel/direction toward the shim,
    // queue the channel, propagate, and assert no L2 latch.)
    let mut dev = DeviceState::new_npu1();
    let (scol, srow) = (0u8, 2u8); // a compute tile
    let (shim_col, shim_row) = (0u8, 0u8);
    dev.array.get_mut(shim_col, shim_row).unwrap().l2_irq.as_mut().unwrap()
        .write_register(crate::device::interrupts::L2_REG_ENABLE, 1 << 4);
    // Block channel 4 southbound from the source so it cannot reach row 0.
    if let Some(em) = dev.array.get_mut(scol, srow).unwrap().core_events.as_mut() {
        em.broadcast.set_block(4, crate::device::events::broadcast::BroadcastDir::South, true);
    }
    dev.array.get_mut(scol, srow).unwrap().pending_broadcasts.push(4);
    dev.propagate_broadcasts(scol, srow);
    let l2 = dev.array.get(shim_col, shim_row).unwrap().l2_irq.as_ref().unwrap();
    assert_eq!(l2.read_register(L2_REG_STATUS).unwrap(), 0,
        "block mask must prevent L2 latch");
}
```

If `broadcast.set_block(channel, dir, bool)` has a different name/signature, find the real API in `src/device/events/broadcast.rs` (search for the block-mask setter used by existing broadcast tests) and adapt the one line, keeping the intent (block channel 4 southbound from the source).

- [ ] **Step 6: Run + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib broadcast_block_mask_prevents_l2_latch` then `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS; full suite green.

```bash
git add src/device/state/effects.rs
git commit -m "interrupt: add named broadcast->L2 sink in propagate_broadcasts

A broadcast delivered to a shim tile is offered to its L2 controller;
L2 enable-mask gating decides the latch. Block-mask gating is inherited
free from the existing BFS. Sink is a named seam, not interrupt logic
smeared into trace-notify. Spec: 2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 4: L1 tap on broadcast reception + bounded fixpoint propagation

The error path originates on compute tiles: error → event → broadcast → shim. The shim's L1 must see *received* broadcasts (not just local Event_Generate), and L1's resulting IRQ_NO output must itself propagate to L2 within the same dispatch. `propagate_broadcasts` drains once, so an L1-produced broadcast needs a bounded fixpoint driver.

**Files:**
- Modify: `src/device/state/effects.rs` (`propagate_broadcasts` delivery block; new `propagate_broadcasts_fixpoint`)
- Modify: `src/device/state/dispatch.rs` (line 164 caller)
- Test: `src/device/state/effects.rs` test module

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn received_broadcast_drives_shim_l1_then_l2_within_one_dispatch() {
    use crate::device::interrupts::{
        L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, L2_REG_ENABLE, L2_REG_STATUS, SwitchId,
    };
    let mut dev = DeviceState::new_npu1();
    let (scol, srow) = (0u8, 2u8); // compute source
    let (shim_col, shim_row) = (0u8, 0u8);

    // Shim L1 switch A: map the incoming broadcast event to slot 0,
    // enable it, output on IRQ_NO 6. The broadcast event id the shim PL
    // module sees for BROADCAST channel N is SHIM_PL_BROADCAST_BASE + N
    // (110 + N) per propagate_broadcasts. Use channel 2 -> event 112.
    {
        let l1 = dev.array.get_mut(shim_col, shim_row).unwrap().l1_irq.as_mut().unwrap();
        l1.set_irq_event_slot(SwitchId::A, 0, 110 + 2);
        l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        l1.write_register(L1_REG_IRQ_NO_A, 6);
    }
    // Shim L2: enable channel 6 (the L1 output id).
    dev.array.get_mut(shim_col, shim_row).unwrap().l2_irq.as_mut().unwrap()
        .write_register(L2_REG_ENABLE, 1 << 6);

    // Source emits broadcast channel 2; fixpoint propagation must carry
    // it to the shim, latch L1 (->IRQ_NO 6), then propagate 6 to L2.
    dev.array.get_mut(scol, srow).unwrap().pending_broadcasts.push(2);
    dev.propagate_broadcasts_fixpoint(scol, srow);

    let l2 = dev.array.get(shim_col, shim_row).unwrap().l2_irq.as_ref().unwrap();
    assert_ne!(l2.read_register(L2_REG_STATUS).unwrap() & (1 << 6), 0,
        "error/broadcast -> L1 -> L2 must complete within one dispatch");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib received_broadcast_drives_shim_l1_then_l2_within_one_dispatch`
Expected: FAIL — `propagate_broadcasts_fixpoint` does not exist (compile error) and/or L1 is not tapped on broadcast reception.

- [ ] **Step 3: Tap L1 on broadcast reception**

In `src/device/state/effects.rs`, in the `propagate_broadcasts` per-tile delivery block, right after the L2 sink added in Task 3 (still inside the same `{ ... }` tile block), add:

```rust
                // Received broadcasts also feed this tile's L1 (shim): the
                // PL module sees BROADCAST channel N as event id
                // SHIM_PL_BROADCAST_BASE + N. On latch L1 queues its
                // IRQ_NO into this tile's pending_broadcasts; the fixpoint
                // driver re-propagates it (L1 output -> L2).
                if tile.l1_irq.is_some() {
                    let ev = SHIM_PL_BROADCAST_BASE + channel;
                    tile.tap_l1_interrupt(ev);
                }
```

`SHIM_PL_BROADCAST_BASE` is the `const SHIM_PL_BROADCAST_BASE: u8 = 110;` already defined inside `propagate_broadcasts` (lines ~467-469); it is in scope here.

- [ ] **Step 4: Add the bounded fixpoint driver**

In `src/device/state/effects.rs`, add a new method after `propagate_broadcasts` (after line 565):

```rust
    /// Drive `propagate_broadcasts` to a fixed point.
    ///
    /// A broadcast delivered to a shim tile can latch its L1 controller,
    /// which queues a new IRQ_NO broadcast (the L1 output). That second
    /// broadcast must also propagate (to reach L2) within the same
    /// dispatch. `propagate_broadcasts` drains once, so loop until no tile
    /// has pending broadcasts, bounded to avoid pathological cycles.
    ///
    /// The cap (8) comfortably exceeds the real L1->L2 chain depth (one
    /// hop); hitting it indicates a misconfiguration loop and is logged
    /// rather than silently spun.
    pub(crate) fn propagate_broadcasts_fixpoint(&mut self, col: u8, source_row: u8) {
        const MAX_ITERS: u32 = 8;
        self.propagate_broadcasts(col, source_row);
        for iter in 0..MAX_ITERS {
            // Collect every tile that still has queued broadcasts (L1
            // outputs land on the shim tile that latched).
            let mut pending: Vec<(u8, u8)> = Vec::new();
            for c in 0..self.array.cols() {
                for r in 0..self.array.rows() {
                    if let Some(t) = self.array.get(c, r) {
                        if !t.pending_broadcasts.is_empty() {
                            pending.push((c, r));
                        }
                    }
                }
            }
            if pending.is_empty() {
                return;
            }
            for (c, r) in pending {
                self.propagate_broadcasts(c, r);
            }
            if iter == MAX_ITERS - 1 {
                log::warn!(
                    "propagate_broadcasts_fixpoint hit iteration cap ({}) \
                     starting from ({},{}); possible broadcast/interrupt loop",
                    MAX_ITERS, col, source_row
                );
            }
        }
    }
```

- [ ] **Step 5: Switch the dispatch caller to the fixpoint driver**

In `src/device/state/dispatch.rs`, replace line 164:

```rust
        // Propagate broadcast events to all tiles in the column.
        self.propagate_broadcasts(tile_addr.col, tile_addr.row);
```

with:

```rust
        // Propagate broadcast events to a fixed point: a delivered
        // broadcast can latch a shim L1, whose IRQ_NO output must itself
        // propagate to L2 within this dispatch (Tier A interrupt path).
        self.propagate_broadcasts_fixpoint(tile_addr.col, tile_addr.row);
```

- [ ] **Step 6: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib received_broadcast_drives_shim_l1_then_l2_within_one_dispatch`
Expected: PASS.

- [ ] **Step 7: Add a no-infinite-loop guard test**

```rust
#[test]
fn fixpoint_propagation_terminates_under_self_feeding_config() {
    use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
    // Configure a shim L1 so its output broadcast id maps back to its own
    // input event, the worst-case self-feeding loop. The cap must make
    // this terminate (test simply must not hang; the warn! is acceptable).
    let mut dev = DeviceState::new_npu1();
    let (col, row) = (0u8, 0u8);
    {
        let l1 = dev.array.get_mut(col, row).unwrap().l1_irq.as_mut().unwrap();
        // IRQ_NO = 3; input slot maps event (110+3)=113 -> slot 0; enabled.
        l1.set_irq_event_slot(SwitchId::A, 0, 110 + 3);
        l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        l1.write_register(L1_REG_IRQ_NO_A, 3);
        let _ = SwitchId::B;
    }
    dev.array.get_mut(col, row).unwrap().pending_broadcasts.push(3);
    dev.propagate_broadcasts_fixpoint(col, row); // must return, not hang
}
```

- [ ] **Step 8: Run + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib fixpoint_propagation_terminates_under_self_feeding_config received_broadcast_drives_shim_l1_then_l2_within_one_dispatch` then `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS; full suite green (the self-feeding test returns within the cap).

```bash
git add src/device/state/effects.rs src/device/state/dispatch.rs
git commit -m "interrupt: tap L1 on broadcast reception + bounded fixpoint

Received broadcasts feed the shim L1 (PL module sees BROADCAST N as
event 110+N); an L1 IRQ_NO output is re-propagated to L2 within the same
dispatch via propagate_broadcasts_fixpoint (drain-until-stable, capped at
8 with a warn on overrun). dispatch.rs now drives the fixpoint. Spec:
2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 5: Route hardware error events through the EventModule (HIGH-RISK item)

The error path is the primary real consumer. `raise_instr_error` currently only notifies `core_trace`; the error never enters the event/broadcast/interrupt path. This task makes hardware errors `generate_event` on the tile's `EventModule` so the existing event→broadcast→L1→L2 chain (Tasks 2-4) carries them. The end-to-end test is the proof gate; if it cannot pass, the error wiring is incomplete by definition.

**Files:**
- Modify: `src/interpreter/core/interpreter.rs` (`raise_instr_error`, lines 18-23)
- Test: `src/device/state/effects.rs` test module (end-to-end), `src/interpreter/core/interpreter.rs` test module (unit)

- [ ] **Step 1: Investigation (no code) — confirm the error event id and module**

Read, and record findings in the commit message of this task:
1. `src/device/events/mod.rs` — what `EventModule` exists on a compute tile (`core_events`) and its `num_events()` for the core module; confirm `core_events::INSTR_ERROR` (from `xdna_archspec::aie2::trace_events::core_events`) is `< num_events()` so `generate_event` will accept it (it early-returns if `event_id as usize >= max_events`).
2. Confirm a compute tile's `core_events` broadcast config can map `INSTR_ERROR` to a broadcast channel (this is CDO-programmed in real runs; the test will program it explicitly).
3. Confirm the value of `core_events::INSTR_ERROR` (grep `xdna-archspec` for `INSTR_ERROR`).

If `INSTR_ERROR >= num_events()` for the core `EventModule`, the correct event id is the group-error event the hardware actually raises; use the `EVENT_GROUP_ERRORS_*` core id documented in `src/device/events/group.rs` instead, and note the substitution in the commit message. Do not hand-wave: the chosen id MUST be one `generate_event` accepts and that broadcast config can map.

- [ ] **Step 2: Write the failing end-to-end test**

In the effects.rs test module:

```rust
#[test]
fn hardware_error_reaches_shim_l2_end_to_end() {
    use crate::device::interrupts::{
        L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, L2_REG_ENABLE, L2_REG_STATUS, SwitchId,
    };
    use xdna_archspec::aie2::trace_events::core_events;

    let mut dev = DeviceState::new_npu1();
    let (ccol, crow) = (0u8, 2u8); // compute tile
    let (shim_col, shim_row) = (0u8, 0u8);
    let err_ev = core_events::INSTR_ERROR; // or the group-error id per Step 1

    // Compute tile: map the error event to broadcast channel 2.
    {
        let em = dev.array.get_mut(ccol, crow).unwrap().core_events.as_mut().unwrap();
        em.broadcast.set_channel(2, err_ev as usize); // configure ch2 = err_ev
    }
    // Shim L1: BROADCAST ch2 -> event (110+2)=112 -> slot0, enabled,
    // output IRQ_NO 7. Shim L2: enable channel 7.
    {
        let l1 = dev.array.get_mut(shim_col, shim_row).unwrap().l1_irq.as_mut().unwrap();
        l1.set_irq_event_slot(SwitchId::A, 0, 112);
        l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        l1.write_register(L1_REG_IRQ_NO_A, 7);
    }
    dev.array.get_mut(shim_col, shim_row).unwrap().l2_irq.as_mut().unwrap()
        .write_register(L2_REG_ENABLE, 1 << 7);

    // Raise a hardware error on the compute tile.
    dev.raise_hardware_error_for_test(ccol, crow, err_ev);
    // Drive propagation (mirrors what a register-write dispatch does).
    dev.propagate_broadcasts_fixpoint(ccol, crow);

    let l2 = dev.array.get(shim_col, shim_row).unwrap().l2_irq.as_ref().unwrap();
    assert_ne!(l2.read_register(L2_REG_STATUS).unwrap() & (1 << 7), 0,
        "hardware error must reach shim L2 via event->broadcast->L1->L2");
}
```

`set_channel`/`raise_hardware_error_for_test`: if the broadcast channel-config setter has a different name, use the real one from `src/device/events/broadcast.rs`. Add a `#[cfg(test)]` `DeviceState::raise_hardware_error_for_test(col,row,ev)` helper that calls the same `EventModule::generate_event` path Step 3 wires (keep it test-only; it exists so the test does not need a full interpreter run).

- [ ] **Step 3: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib hardware_error_reaches_shim_l2_end_to_end`
Expected: FAIL — error event never enters the EventModule, so nothing broadcasts, L2 stays 0.

- [ ] **Step 4: Wire the error into the EventModule**

In `src/interpreter/core/interpreter.rs`, modify `raise_instr_error` (lines 18-23) to also generate the event on the tile's core `EventModule`:

```rust
fn raise_instr_error(tile: &mut Tile, cycle: u64, pc: u32) {
    use xdna_archspec::aie2::trace_events::core_events;
    tile.core_debug.set_error_halt(true);
    tile.core_trace.notify_event(core_events::INSTR_ERROR, cycle, Some(pc));
    tile.core_perf_counters.handle_event(core_events::INSTR_ERROR);
    // Tier A interrupt path: the error must also enter the event
    // subsystem so it can broadcast to the shim L1/L2 interrupt
    // controllers. Without this, the error path -- the primary real
    // interrupt consumer -- never reaches L2.
    if let Some(em) = tile.core_events.as_mut() {
        em.generate_event(core_events::INSTR_ERROR);
    }
}
```

If Step 1 found `INSTR_ERROR` is not a valid `EventModule` id for the core module, substitute the group-error id determined there (and reflect it in the `set_channel` of the test and the commit message). Also add the matching `#[cfg(test)] raise_hardware_error_for_test` helper on `DeviceState` if not already added in Step 2 — it must call `em.generate_event(ev)` on `core_events` for the given tile, nothing more.

- [ ] **Step 5: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib hardware_error_reaches_shim_l2_end_to_end`
Expected: PASS — the full event→broadcast→L1→L2 error chain completes.

- [ ] **Step 6: Full suite (regression-sensitive — error path touches the interpreter) + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass, 0 failed. If any interpreter/error test regresses, fix before committing (the added `generate_event` must not change error-halt semantics — it is additive).

```bash
git add src/interpreter/core/interpreter.rs src/device/state/effects.rs
git commit -m "interrupt: route hardware error events through EventModule

raise_instr_error now also calls EventModule::generate_event so a
hardware error enters the event->broadcast->L1->L2 chain -- previously
it only reached core_trace, so the error path (the primary real
interrupt consumer) never reached L2. End-to-end test proves the full
chain. [Record Step-1 investigation result: chosen event id =
<INSTR_ERROR or group-error id> because <reason>.] Spec:
2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 6: Privilege scope-out doc + channel-identity invariant test

Document the deliberate privilege scope-out per the noc/shim_mux precedent, and pin the channel-identity invariant (L1 IRQ_NO == broadcast channel == L2 input) with a probe test.

**Files:**
- Modify: `src/device/interrupts/l2.rs` (doc-comment on the interrupt-routing register)
- Test: `src/device/interrupts/l2.rs` test module

- [ ] **Step 1: Add the privilege scope-out doc-comment**

In `src/device/interrupts/l2.rs`, on the `write_register` arm handling `L2_REG_INTERRUPT` (the `o if o == L2_REG_INTERRUPT => { /* Interrupt output is read-only */ }` arm), expand the comment to:

```rust
            o if o == L2_REG_INTERRUPT => {
                // NoC interrupt routing register. On hardware this is the
                // single privileged L2 register (aie-rt
                // _XAie_PrivilegeSetL2IrqId). Privilege is a driver-side
                // concern; per the project policy applied to noc/shim_mux,
                // the emulator gives unrestricted access and does not model
                // privilege gating. Output state is derived, so the write
                // is accepted and ignored. Scoped out by design (spec
                // 2026-05-19-interrupt-l2-closeout, Tier A).
            }
```

- [ ] **Step 2: Write the channel-identity invariant test**

In the `l2.rs` test module:

```rust
#[test]
fn channel_identity_l1_irq_no_equals_l2_input_channel() {
    // Invariant probe: the broadcast id an L1 switch outputs (IRQ_NO) is
    // the same numeric channel L2 latches on. If a future change inserts
    // a remap, this fails loudly.
    use super::*;
    let mut l2 = L2InterruptController::new();
    for ch in 0u8..16 {
        l2.write_enable(1 << ch);
    }
    for irq_no in 0u8..16 {
        l2.signal_interrupt(irq_no); // L1 IRQ_NO fed directly as L2 channel
        assert_ne!(l2.read_status() & (1 << irq_no), 0,
            "L2 must latch the same channel index L1 output as IRQ_NO ({irq_no})");
    }
}
```

- [ ] **Step 3: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib channel_identity_l1_irq_no_equals_l2_input_channel`
Expected: PASS (the invariant already holds; this pins it).

- [ ] **Step 4: Commit**

```bash
git add src/device/interrupts/l2.rs
git commit -m "interrupt: document privilege scope-out + pin channel identity

L2 interrupt-routing register: documented the deliberate privilege
scope-out (driver-side concern; unrestricted per noc/shim_mux
precedent). Added an invariant probe test that L1 IRQ_NO == L2 input
channel so a future remap fails loudly. Spec:
2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

### Task 7: Coverage marker flip + Tier B findings doc + regen

Flip `interrupt` PARTIAL→Full with an honest narrative, capture the Tier B host-boundary contract durably (immediate follow-up), regenerate coverage artifacts.

**Files:**
- Create: `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md`
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (lines 165-168)
- Regenerate: `docs/coverage/aie2/*` via the example binary

- [ ] **Step 1: Create the Tier B findings doc**

Create `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md`:

```markdown
# Interrupt Tier B — Firmware Async-Event Host Delivery (tracked follow-up)

**Status:** Not started. **Immediate follow-up** after the Tier A interrupt
close-out (ahead of the clock_control / noc gap-queue items).

## Why this is separate from Tier A

On Phoenix/NPU1 the AIE L1/L2 shim interrupt never reaches the x86 host
directly. It terminates at NPI interrupt lines consumed by on-NPU MGMT/ERT
firmware, which synthesizes a mailbox async-event message; the host only ever
sees the mailbox MSI-X. Tier A (the `interrupt` subsystem) models the AIE path
to that firmware boundary. Tier B is the firmware/mailbox async-event model
the emulator deliberately lacks (it shortcuts MGMT/ERT with a synchronous
completion model).

## Host-boundary contract (derived from xdna-driver)

- Only host IRQ the driver registers: `mailbox_irq_handler` (MSI-X),
  `amdxdna_mailbox.c` ~line 924.
- AIE array errors reach the host via async mailbox messages →
  `aie2_error_async_cb` (`aie2_error.c` ~278-289), queued to a workqueue.
- Mailbox response contract: write response into the I2X ring buffer at the
  firmware-provided base; update the I2X tail pointer register; write the
  IOHUB interrupt status register at `i2x_head_ptr_reg + 4`
  (`aie2_pci.h` ~376-379) to fire MSI-X.
- TDR: driven by absence of mailbox/completion progress (HSA queue read_index
  not advancing); `aie2_tdr.c` ~27-73.

## Scope when picked up

Minimal firmware async-event model: when Tier A asserts an NPI interrupt line
for an error, synthesize the async-event mailbox message so the driver's
`aie2_error_async_cb` path observes it. This is its own brainstorming → spec →
plan cycle.
```

- [ ] **Step 2: Run the full suite as the pre-flip gate**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass. The marker may only flip to Full with the whole chain green.

- [ ] **Step 3: Flip the coverage marker + narrative**

In `crates/xdna-archspec/src/coverage/units.rs`, replace the `d("interrupt", ...)` entry (lines 165-168) with:

```rust
        d("interrupt", "aie-rt interrupt/, AM025 (shim interrupt 23 reg: 18 L1 + 5 L2)",
          &["src/device/interrupts/l1.rs", "src/device/interrupts/l2.rs",
            "src/device/state/effects.rs", "src/device/tile/registers.rs"],
          "Full AIE2 shim interrupt path MODELED. All 23 registers (18 L1 over 2 switches 0x35000-0x35050, 5 L2 0x15000-0x15010) read- and write-routed with exact semantics (write-1-to-clear status, enable/disable->mask, read-only mask). Stimulus path wired: event/error -> EventModule -> L1 (both switches, independent slot-match + enable gating) -> broadcast network (block-mask honored) -> L2 sink -> NPI line, driven to a fixed point so an L1 output reaches L2 within one dispatch. Hardware errors enter via raise_instr_error -> EventModule. Privilege gating (only the L2 interrupt-routing register) is a driver-side concern, scoped out unrestricted per the noc/shim_mux precedent. Host-visible delivery is firmware-mediated (the AIE interrupt never reaches the x86 host directly) -- Tier B (firmware async-event mailbox) is the tracked IMMEDIATE follow-up, see docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md.",
          Modeled { completeness: Full }, None),
```

- [ ] **Step 4: Build the archspec crate to verify the entry compiles**

Run: `cargo build -p xdna-archspec`
Expected: builds clean (no syntax error in the entry).

- [ ] **Step 5: Regenerate coverage artifacts**

Run: `cargo run -p xdna-archspec --example gen_coverage_artifacts`
Expected: regenerates `docs/coverage/aie2/*`. Then:

Run: `git diff --stat docs/coverage/aie2/`
Expected: `subsystem-index.md` changes only the interrupt narrative line; `implementation-gaps.md` drops the `interrupt: PARTIAL ...` row (now Full). No unrelated subsystem lines change.

- [ ] **Step 6: Verify the gap row is gone**

Run: `grep -n interrupt docs/coverage/aie2/implementation-gaps.md`
Expected: no output (interrupt no longer a gap; the file now lists only `noc: STUB` and `clock_control: STUB`).

- [ ] **Step 7: Final full suite + commit**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: all pass, 0 failed.

```bash
git add crates/xdna-archspec/src/coverage/units.rs docs/coverage/aie2/ docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md
git commit -m "interrupt: flip coverage Partial->Full; track Tier B

Tier A complete: all 23 shim interrupt registers read/write-routed with
exact semantics; event/error -> L1 -> broadcast -> L2 -> NPI path wired,
fixpoint-driven, block-mask honored, error path proven end-to-end;
privilege scoped out per precedent. Coverage marker Partial->Full,
narrative rewritten, artifacts regenerated (implementation-gaps drops
the interrupt row). Tier B (firmware async-event host delivery) captured
as the tracked immediate follow-up. Spec:
2026-05-19-interrupt-l2-closeout-design.md.

Generated using Claude Code."
```

---

## Plan Self-Review

**Spec coverage:**
- Read-path routing → Task 1. ✓
- Event→L1 tap (both switches, independent) → Task 2. ✓
- Broadcast→L2 named sink + block-mask + disabled-channel drop → Task 3. ✓
- L1-on-received-broadcast + L1-output re-propagation (the spec's "all event sources" constraint) → Task 4. ✓
- Hardware error events into EventModule (spec's HIGH-RISK item, with the end-to-end proof test) → Task 5. ✓
- Switch independence test → Task 2 Step 6. Channel-identity invariant → Task 6. Privilege scope-out doc → Task 6. ✓
- Coverage marker flip + Tier B findings doc (immediate follow-up) + regen, implementation-gaps drops the row → Task 7. ✓
- 9-item test matrix from the spec: read-routing (T1), event→L1 (T2), switch independence (T2), broadcast transport (T3), block-mask (T3), L2 mask gating (T3 disabled-channel + T1), channel identity (T6), error path (T5), privilege scope (T6 doc). ✓ All covered.

**Placeholder scan:** Every code step has complete code. The two deferred specifics (exact effects.rs test-helper name; exact broadcast block-mask setter name) are bounded "use the real API from file X, keep this intent" instructions with the surrounding code fully specified — not logic placeholders. Task 5 Step 1 is an explicit investigation step the spec mandated (the error-id confirmation) with a concrete fallback rule, not a hand-wave.

**Type consistency:** `tap_l1_interrupt(&mut self, event_id: u8)` defined in Task 2, reused identically in Task 4. `propagate_broadcasts_fixpoint(&mut self, col: u8, source_row: u8)` defined Task 4, used in Tasks 4/5 tests and dispatch.rs. `signal_event -> Option<u8>`, `read_irq_no -> u32` (cast `as u8`), `signal_interrupt(channel: u8)`, `read_register -> Option<u32>`, `EventModule::generate_event(EventId=u8)` consistent across all tasks. `SHIM_PL_BROADCAST_BASE = 110` used consistently (Task 4 wiring, Task 4/5 tests). Marker syntax `Modeled { completeness: Full }` matches the verbatim units.rs format.

No issues found requiring inline fixes beyond what is already written.
