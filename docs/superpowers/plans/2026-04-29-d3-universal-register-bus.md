# D.3 Universal Register Bus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `write_register` / `mask_write_register` the single register bus for the device, removing the CDO-specific typed-promotion handlers (`apply_core_enable`, `start_dma_channel`) that bypass the bus.

**Architecture:** Real silicon has one register bus. The current CDO path bypasses it via parallel typed handlers, causing two latent bugs (CORE_CONTROL readback divergence; MaskWrite-to-CORE_CONTROL bit corruption). Reroute `apply_device_op::CoreEnable` and `DmaStart` through `write_register`, stop promoting MaskWrite-to-CORE_CONTROL, and delete the now-redundant typed handlers. Three commits: failing regression tests, refactor, dead-code deletion. Bridge-test gate between commits 2 and 3.

**Tech Stack:** Rust, `xdna-emu` crate. Existing tests live in `src/device/state/tests.rs`. Lowering helpers in `src/parser/cdo/semantics.rs`. Apply layer in `src/device/state/cdo.rs`. Subsystem branches in `src/device/state/compute.rs`.

**Background reading (do this first):**
- `docs/superpowers/specs/2026-04-29-d3-universal-register-bus-design.md` — the full design rationale and the two bugs being fixed.
- `src/device/state/cdo.rs` — current `apply_device_op` implementation.
- `src/device/state/dispatch.rs` — `write_register` / `mask_write_register` (the universal bus).
- `src/device/state/compute.rs:171-289` (`write_dma_channel`) and `:394-408` (`apply_core_enable`) and `:306-385` (`start_dma_channel`) and `:620-661` (`write_core_register`) — the typed handlers and their offset-dispatch counterparts.
- `src/parser/cdo/semantics.rs:107-163` (`match_dma_start_queue`, `match_strided_start_queue`) — the lowering side, especially the constants used.

---

> **Sweep-as-of 2026-05-01:** D.3 completed -- commits 11d2b66 (regression tests), 61d3a94 (refactor), 9e5fecf (delete dead typed-handler code), d297b98 (post-cleanup). No tag, but the three-commit landing structure called out in the plan was followed. Steps below were executed organically rather than ticked one-by-one; this sweep flips the checkboxes to match the verified completion state.


## Setup

- [x] **Step 0.1: Establish baseline**

Run: `cargo test --lib 2>&1 | tail -20`
Expected: all tests pass. Note the test count for comparison.

- [x] **Step 0.2: Confirm spec is on disk**

Run: `ls -la docs/superpowers/specs/2026-04-29-d3-universal-register-bus-design.md`
Expected: file exists, ~278 lines.

---

## Task 1: Commit 1 — Three regression tests

**Files:**
- Modify: `src/device/state/tests.rs` (append three tests at the end)

**Goal:** Add three regression tests. Two are RED (proving the bugs exist); one is GREEN (proving the side-effect-ordering invariant). Commit them in the RED state so bisect tells the story.

- [x] **Step 1.1: Read the existing tests file to find the end and the import patterns**

Run: `wc -l src/device/state/tests.rs`
Expected: a line count (no specific value; we're appending).

Read `src/device/state/tests.rs` lines 1-15 to confirm the imports already at the top of the file (`use super::*;` etc.). The new tests don't need additional imports beyond what `super::*` provides plus a few specific items used in the tests — the imports go inline within each test for clarity.

- [x] **Step 1.2: Append Test A — CORE_CONTROL readback divergence**

Append the following test to `src/device/state/tests.rs` (at the end of the file):

```rust
/// Regression test for D.3 Bug 1: CORE_CONTROL readback divergence.
///
/// A CDO write to CORE_CONTROL must be visible via the register-bus
/// read path (`tile.read_register_pure(offset)`). The CDO promotion
/// path currently routes through `apply_core_enable` which only
/// updates the typed `tile.core.control` mirror and skips
/// `tile.registers`. After D.3 commit 2 the promotion path goes
/// through `write_register` which stores raw, fixing this.
#[test]
fn core_control_cdo_write_is_readable_via_register_bus() {
    use crate::parser::cdo::semantics::lower;
    use crate::parser::cdo::CdoRaw;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let address = TileAddress::encode(col, row, cc_offset);

    let cmd = CdoRaw::Write { address, value: 0x1 };
    for op in lower(&cmd) {
        state.apply_device_op(&op).unwrap();
    }

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.read_register_pure(cc_offset),
        0x1,
        "CDO write to CORE_CONTROL must be visible via register-bus reads"
    );
}
```

- [x] **Step 1.3: Append Test B — MaskWrite preservation**

Append the following test to `src/device/state/tests.rs`:

```rust
/// Regression test for D.3 Bug 2: MaskWrite-to-CORE_CONTROL bit
/// corruption.
///
/// A CDO MaskWrite to CORE_CONTROL with a partial mask must only
/// modify the bits covered by the mask, leaving other bits unchanged.
/// The current CDO promotion path drops the mask in
/// `lower_mask_write` and `apply_core_enable` overwrites the full
/// word, corrupting bits 31..1. After D.3 commit 2 MaskWrite stops
/// promoting and rides through `mask_write_register` which mask-
/// blends correctly.
#[test]
fn core_control_cdo_mask_write_preserves_unmasked_bits() {
    use crate::parser::cdo::semantics::lower;
    use crate::parser::cdo::CdoRaw;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let address = TileAddress::encode(col, row, cc_offset);

    // Pre-set tile.core.control = 0xABCD_0001 directly (simulates a
    // prior write that stored a non-trivial value).
    {
        let tile = state.array.get_mut(col, row).unwrap();
        tile.core.control = 0xABCD_0001;
        tile.core.enabled = true;
    }

    // Clear bit 0 (disable) but leave bits 31..1 alone.
    let cmd = CdoRaw::MaskWrite { address, mask: 0x1, value: 0x0 };
    for op in lower(&cmd) {
        state.apply_device_op(&op).unwrap();
    }

    let tile = state.array.tile(col, row);
    assert_eq!(
        tile.core.control, 0xABCD_0000,
        "MaskWrite with mask=0x1 must only clear bit 0, preserving bits 31..1"
    );
    assert!(!tile.core.enabled, "Bit 0 cleared -> core disabled");
}
```

- [x] **Step 1.4: Append Test C — Side-effect-ordering safety net**

Append the following test to `src/device/state/tests.rs`:

```rust
/// Invariant proof for D.3 commit 2: writing CORE_CONTROL via the
/// register bus does not match any branch of
/// `apply_tile_local_effects` (cascade, shim mux, lock overflow/
/// underflow clear, perf counters, trace registers).
///
/// The non-CDO path already exercises this code path; this test
/// pins the invariant so that when D.3 commit 2 routes the CDO
/// promotion path through `write_register` (and thus through
/// `apply_tile_local_effects`), no surprise side effect fires.
///
/// GREEN today, GREEN after commit 2.
#[test]
fn register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects() {
    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2; // compute row
    let cc_offset = xdna_archspec::aie2::registers::CORE_CONTROL;
    let cc_addr = TileAddress::encode(col, row, cc_offset);

    // Snapshot fields apply_tile_local_effects can touch.
    let tile = state.array.tile(col, row);
    let cascade_in_pre = tile.cascade_input_dir;
    let cascade_out_pre = tile.cascade_output_dir;
    let lock_over_pre: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_pre: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    let mut reg_pre: Vec<(u32, u32)> = tile
        .registers_ref()
        .iter()
        .filter(|(&offset, _)| offset != cc_offset)
        .map(|(&offset, &value)| (offset, value))
        .collect();
    reg_pre.sort_by_key(|(o, _)| *o);

    state.write_register(cc_addr, 0x1).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(tile.cascade_input_dir, cascade_in_pre, "cascade_input_dir must not change");
    assert_eq!(tile.cascade_output_dir, cascade_out_pre, "cascade_output_dir must not change");

    let lock_over_post: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_post: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    assert_eq!(lock_over_pre, lock_over_post, "lock overflow bits must not change");
    assert_eq!(lock_under_pre, lock_under_post, "lock underflow bits must not change");

    let mut reg_post: Vec<(u32, u32)> = tile
        .registers_ref()
        .iter()
        .filter(|(&offset, _)| offset != cc_offset)
        .map(|(&offset, &value)| (offset, value))
        .collect();
    reg_post.sort_by_key(|(o, _)| *o);
    assert_eq!(
        reg_pre, reg_post,
        "Only CORE_CONTROL should change in the register map; other entries must be untouched"
    );

    // Sanity: the targeted effects DID happen.
    assert_eq!(tile.core.control, 0x1);
    assert_eq!(*tile.registers_ref().get(&cc_offset).unwrap(), 0x1);
}

/// Same invariant as the CORE_CONTROL test, but for a compute
/// Start_Queue offset. Verifies that writing Start_Queue via the
/// register bus does not trigger any apply_tile_local_effects branch.
#[test]
fn register_bus_write_to_compute_start_queue_does_not_trigger_unrelated_tile_effects() {
    use xdna_archspec::aie2::registers::dma::COMPUTE_DMA_S2MM_0_START_QUEUE;

    let mut state = DeviceState::new_npu1();
    let col: u8 = 1;
    let row: u8 = 2;
    let sq_offset = COMPUTE_DMA_S2MM_0_START_QUEUE;
    let sq_addr = TileAddress::encode(col, row, sq_offset);

    let tile = state.array.tile(col, row);
    let cascade_in_pre = tile.cascade_input_dir;
    let cascade_out_pre = tile.cascade_output_dir;
    let lock_over_pre: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_pre: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();

    // Write 0 (BD 0, no repeat) — innocuous payload, just need the
    // write to land on the offset.
    state.write_register(sq_addr, 0x0).unwrap();

    let tile = state.array.tile(col, row);
    assert_eq!(tile.cascade_input_dir, cascade_in_pre);
    assert_eq!(tile.cascade_output_dir, cascade_out_pre);

    let lock_over_post: Vec<bool> = tile.locks.iter().map(|l| l.overflow).collect();
    let lock_under_post: Vec<bool> = tile.locks.iter().map(|l| l.underflow).collect();
    assert_eq!(lock_over_pre, lock_over_post);
    assert_eq!(lock_under_pre, lock_under_post);
}
```

- [x] **Step 1.5: Run the new tests, confirm A and B FAIL, C passes**

Run: `cargo test --lib core_control_cdo_write_is_readable_via_register_bus core_control_cdo_mask_write_preserves_unmasked_bits register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects register_bus_write_to_compute_start_queue_does_not_trigger_unrelated_tile_effects 2>&1 | tail -40`

Expected:
- `core_control_cdo_write_is_readable_via_register_bus`: **FAIL** with `assertion `left == right` failed`, `left: 0`, `right: 1`.
- `core_control_cdo_mask_write_preserves_unmasked_bits`: **FAIL** with `left: 0`, `right: 0xABCD0000` (or similar).
- `register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects`: **PASS**.
- `register_bus_write_to_compute_start_queue_does_not_trigger_unrelated_tile_effects`: **PASS**.

If A or B unexpectedly pass, the bug has already been fixed elsewhere — STOP and investigate before continuing.
If C fails, the invariant assumed by the design is wrong — STOP and re-read the spec.

- [x] **Step 1.6: Run the full library test suite, confirm only the two new RED tests fail**

Run: `cargo test --lib 2>&1 | tail -15`
Expected: total failures = 2; both are the two new RED tests.

- [x] **Step 1.7: Commit**

Run:
```
git add src/device/state/tests.rs
git commit -m "$(cat <<'EOF'
test(d3): regression tests for universal register bus refactor

Two RED tests prove the bugs the D.3 refactor will fix:
- CORE_CONTROL readback divergence: a CDO write goes through
  apply_core_enable which never stores into tile.registers, so
  read_register_pure(CORE_CONTROL) returns 0 instead of the written
  value.
- MaskWrite-to-CORE_CONTROL bit corruption: lower_mask_write drops
  the mask and apply_core_enable overwrites tile.core.control with
  the raw value, clobbering bits not covered by the mask.

Two GREEN safety-net tests prove the invariant relied on by the
refactor: writing CORE_CONTROL or Start_Queue via the register bus
does not trigger any apply_tile_local_effects branch (cascade,
shim mux, lock over/underflow clears, perf counters, trace
registers). After commit 2 the CDO path also exercises this code
path; the invariant must hold for both.

The two RED tests are committed in the RED state intentionally so
git bisect tells the story of the fix.

Generated using Claude Code.
EOF
)"
```

Verify with `git log --oneline -3` that the commit landed.

---

## Task 2: Commit 2 — Route CoreEnable / DmaStart through the register bus

**Files:**
- Modify: `src/device/state/cdo.rs` (rewrite two arms in `apply_device_op`, add `start_queue_offset` helper)
- Modify: `src/parser/cdo/semantics.rs` (remove CORE_CONTROL branch from `lower_mask_write`)

**Goal:** Make `apply_device_op::CoreEnable` and `apply_device_op::DmaStart` route through `write_register` instead of the typed handlers. Stop promoting MaskWrite-to-CORE_CONTROL. After this commit, all four tests from Task 1 pass.

- [x] **Step 2.1: Add the `start_queue_offset` helper to `src/device/state/cdo.rs`**

Append this helper function at the bottom of `src/device/state/cdo.rs` (after the existing `encode_addr` helper at line 184; the file currently ends at line 184). The helper inverts `match_dma_start_queue` from `parser/cdo/semantics.rs`, reconstructing the Start_Queue offset from `(row, channel, dir)` so `apply_device_op::DmaStart` can call `write_register` with the right address.

```rust
/// Inverse of `match_dma_start_queue` in `parser::cdo::semantics`:
/// reconstruct the Start_Queue / Task_Queue offset for a tile
/// from `(row, channel, dir)`. Used by `apply_device_op::DmaStart`
/// to build the address it routes through `write_register`.
///
/// Returns an error for invalid (channel, dir) combinations on the
/// detected tile kind. The lowering side never produces such
/// combinations (it derives them by matching valid offsets), so
/// this branch is defensive: an error here means an upstream bug.
fn start_queue_offset(
    row: u8,
    channel: u8,
    dir: xdna_archspec::types::DmaDirection,
) -> Result<u32> {
    use crate::device::registers::{tile_kind_from_row, TileKind};
    use xdna_archspec::aie2::registers::dma::{
        COMPUTE_DMA_MM2S_0_START_QUEUE, COMPUTE_DMA_MM2S_1_START_QUEUE,
        COMPUTE_DMA_S2MM_0_START_QUEUE, COMPUTE_DMA_S2MM_1_START_QUEUE,
        MEMTILE_CHANNELS_PER_DIR, MEMTILE_CHANNEL_STRIDE,
        MEMTILE_DMA_MM2S_0_START_QUEUE, MEMTILE_DMA_S2MM_0_START_QUEUE,
        SHIM_CHANNELS_PER_DIR, SHIM_CHANNEL_STRIDE,
        SHIM_DMA_MM2S_0_TASK_QUEUE, SHIM_DMA_S2MM_0_TASK_QUEUE,
    };
    use xdna_archspec::types::DmaDirection;

    match tile_kind_from_row(row) {
        TileKind::Compute => match (dir, channel) {
            (DmaDirection::S2mm, 0) => Ok(COMPUTE_DMA_S2MM_0_START_QUEUE),
            (DmaDirection::S2mm, 1) => Ok(COMPUTE_DMA_S2MM_1_START_QUEUE),
            (DmaDirection::Mm2s, 0) => Ok(COMPUTE_DMA_MM2S_0_START_QUEUE),
            (DmaDirection::Mm2s, 1) => Ok(COMPUTE_DMA_MM2S_1_START_QUEUE),
            _ => anyhow::bail!(
                "start_queue_offset: invalid compute DMA channel dir={:?} ch={}",
                dir, channel
            ),
        },
        TileKind::Mem => {
            if channel >= MEMTILE_CHANNELS_PER_DIR {
                anyhow::bail!(
                    "start_queue_offset: memtile channel {} >= {}",
                    channel, MEMTILE_CHANNELS_PER_DIR
                );
            }
            let base = match dir {
                DmaDirection::S2mm => MEMTILE_DMA_S2MM_0_START_QUEUE,
                DmaDirection::Mm2s => MEMTILE_DMA_MM2S_0_START_QUEUE,
            };
            Ok(base + (channel as u32) * MEMTILE_CHANNEL_STRIDE)
        }
        TileKind::ShimNoc | TileKind::ShimPl => {
            if channel >= SHIM_CHANNELS_PER_DIR {
                anyhow::bail!(
                    "start_queue_offset: shim channel {} >= {}",
                    channel, SHIM_CHANNELS_PER_DIR
                );
            }
            let base = match dir {
                DmaDirection::S2mm => SHIM_DMA_S2MM_0_TASK_QUEUE,
                DmaDirection::Mm2s => SHIM_DMA_MM2S_0_TASK_QUEUE,
            };
            Ok(base + (channel as u32) * SHIM_CHANNEL_STRIDE)
        }
    }
}
```

- [x] **Step 2.2: Verify the helper compiles standalone**

Run: `cargo build --lib 2>&1 | tail -10`
Expected: clean build. If any of the imported constants don't exist by that name in `xdna_archspec::aie2::registers::dma`, the compiler will say so — open `xdna-archspec/src/aie2/registers/dma.rs` (or similar path), find the actual constant name, and update the import. The constant names listed above are taken directly from the imports already used in `src/parser/cdo/semantics.rs` lines 42-47 and 129-132, so they should match.

- [x] **Step 2.3: Rewrite the `CoreEnable` arm in `apply_device_op`**

In `src/device/state/cdo.rs`, locate the `apply_device_op` match — currently at lines 132-174. Replace the `DeviceOp::CoreEnable` arm. The current arm (around lines 153-155) reads:

```rust
            DeviceOp::CoreEnable { tile, enabled, value } => {
                self.apply_core_enable(tile.col, tile.row, *enabled, *value);
            }
```

Replace with:

```rust
            DeviceOp::CoreEnable { tile, enabled: _, value } => {
                // Route through the universal register bus. The
                // typed `enabled` flag is intentionally ignored on
                // apply -- `write_core_register`'s CORE_CONTROL
                // branch derives it from `value & 1`. The variant
                // keeps `enabled` as a parser-side semantic marker
                // (room for option (a) future work; see D.3 spec).
                let addr = encode_addr(*tile, xdna_archspec::aie2::registers::CORE_CONTROL);
                self.write_register(addr, *value)?;
            }
```

- [x] **Step 2.4: Rewrite the `DmaStart` arm in `apply_device_op`**

In `src/device/state/cdo.rs`, locate the `DeviceOp::DmaStart` arm (around lines 156-158). The current arm reads:

```rust
            DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
                self.start_dma_channel(tile.col, tile.row, *channel, *dir, *bd_id);
            }
```

Replace with:

```rust
            DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
                // Route through the universal register bus.
                // `start_queue_offset` reconstructs the offset from
                // (row, channel, dir); `write_register` then dispatches
                // to `write_dma_channel` (compute) or
                // `write_memtile_dma_channel` / `write_shim_dma_channel`
                // (memtile / shim), each of which has the existing
                // Start_Queue branch that does the typed effect.
                let offset = start_queue_offset(tile.row, *channel, *dir)?;
                let addr = encode_addr(*tile, offset);
                self.write_register(addr, *bd_id)?;
            }
```

- [x] **Step 2.5: Update the `apply_device_op` doc comment**

In `src/device/state/cdo.rs`, the doc comment for `apply_device_op` (lines 118-131) currently describes the typed-handler architecture. Replace lines 127-131 — the section that reads:

```
    /// Structured ops (`CoreEnable`, `DmaStart`) call dedicated helpers
    /// (`apply_core_enable`, `start_dma_channel`) that
    /// replicate the CDO-specific side effects that used to live
    /// inline in `write_core_register`'s CORE_CONTROL branch and
    /// `write_dma_channel`'s Start_Queue branch.
```

with:

```
    /// Structured ops (`CoreEnable`, `DmaStart`) also route through
    /// `write_register`: the existing offset-dispatch branches in
    /// `write_core_register` (CORE_CONTROL) and `write_dma_channel`
    /// (Start_Queue) are the single source of truth for the typed
    /// effects, so the CDO and non-CDO paths produce identical
    /// observable state. The typed variants survive on `DeviceOp`
    /// as parser-side semantic markers, not parallel device
    /// handlers (see D.3 spec for the rationale).
```

- [x] **Step 2.6: Remove the CORE_CONTROL branch from `lower_mask_write`**

In `src/parser/cdo/semantics.rs`, locate `lower_mask_write` (lines 269-297). The current implementation has a CORE_CONTROL promotion branch at lines 282-294. Replace the body of `lower_mask_write` so it stops promoting:

```rust
fn lower_mask_write(address: u32, mask: u32, value: u32) -> DeviceOp {
    let (tile, offset) = decode_aie2_address(address);

    // MaskWrites are NOT promoted to typed ops. Mask-blending lives
    // on the apply side (`mask_write_register` -> module-specific
    // mask handler), so the typed handler would either duplicate
    // it or drop the mask. Letting MaskWrite ride as `RegMask`
    // keeps mask-blend correctness in one place. If we later want
    // mask-aware promotion, add `mask: Option<u32>` to
    // `DeviceOp::CoreEnable` (see D.3 spec, "room preserved for
    // option (a)").
    DeviceOp::RegMask { tile, offset, mask, value }
}
```

The `use xdna_archspec::aie2::registers::CORE_CONTROL` import on line 49 is still used by `lower_write` (line 245) — leave it in place.

- [x] **Step 2.7: Update / remove tests in `semantics.rs` that asserted MaskWrite promotion**

In `src/parser/cdo/semantics.rs`, the test module includes:
- `core_control_mask_write_touching_bit0_promotes_to_core_enable` (around line 624) — this test now FAILS because the promotion was removed. Either update it to assert the new behavior (RegMask, not CoreEnable) or delete it.
- `core_control_mask_write_not_touching_bit0_stays_reg_mask` (around line 646) — this test still passes (MaskWrite to CORE_CONTROL with mask not touching bit 0 was already RegMask).

Replace the `core_control_mask_write_touching_bit0_promotes_to_core_enable` test body with the inverted assertion:

```rust
    #[test]
    fn core_control_mask_write_touching_bit0_stays_reg_mask() {
        // Post-D.3: MaskWrites are never promoted to CoreEnable, even
        // when the mask touches bit 0. Mask-blend correctness lives
        // on the apply side (mask_write_register -> mask_write_core_register).
        let (col, row) = (1u8, 2u8);
        let addr = aie_addr(col, row, CORE_CONTROL);
        let op = lower_mask_write(addr, 0x1, 0x0);
        match op {
            DeviceOp::RegMask { tile, offset, mask, value } => {
                assert_eq!(tile.col, col);
                assert_eq!(tile.row, row);
                assert_eq!(offset, CORE_CONTROL);
                assert_eq!(mask, 0x1);
                assert_eq!(value, 0x0);
            }
            other => panic!(
                "MaskWrite to CORE_CONTROL must lower to RegMask post-D.3, got {:?}",
                other
            ),
        }
    }
```

- [x] **Step 2.8: Build, then run targeted tests**

Run: `cargo build --lib 2>&1 | tail -10`
Expected: clean build.

Run: `cargo test --lib core_control_cdo_write_is_readable_via_register_bus core_control_cdo_mask_write_preserves_unmasked_bits register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects register_bus_write_to_compute_start_queue_does_not_trigger_unrelated_tile_effects core_control_mask_write 2>&1 | tail -20`

Expected: all tests PASS, including the four D.3 regression tests and the renamed `core_control_mask_write_touching_bit0_stays_reg_mask`.

- [x] **Step 2.9: Run full library suite to catch regressions**

Run: `cargo test --lib 2>&1 | tail -15`
Expected: **0 failures**, total test count = baseline + 4 (the four new D.3 tests; one semantics test was renamed but count is unchanged for it).

If any pre-existing tests now fail, STOP. The most likely candidate: a test in `src/device/state/tests.rs` or `src/parser/cdo/semantics.rs` that explicitly asserted the old promotion behavior. Read the test, decide if its assertion is now stale (update) or if the refactor actually broke something (back out the change and rethink).

- [x] **Step 2.10: Commit**

Run:
```
git add src/device/state/cdo.rs src/parser/cdo/semantics.rs
git commit -m "$(cat <<'EOF'
refactor(d3): route CoreEnable/DmaStart through the universal register bus

Make apply_device_op::CoreEnable and DmaStart route through
write_register instead of the typed handlers (apply_core_enable,
start_dma_channel). The existing offset-dispatch branches in
write_core_register / write_dma_channel are the single source of
truth for the typed effects, so the CDO and non-CDO paths now
produce identical observable state.

Two latent bugs are fixed as a consequence:
- CORE_CONTROL readback divergence: tile.registers now records
  CDO writes to CORE_CONTROL, so read_register_pure returns the
  written value.
- MaskWrite-to-CORE_CONTROL bit corruption: MaskWrite no longer
  promotes to CoreEnable; it rides as RegMask through
  mask_write_register -> mask_write_core_register, which mask-
  blends correctly.

DeviceOp variants stay shape-stable (CoreEnable still carries
{tile, enabled, value}) so a future option-(a) refactor can add
mask: Option<u32> without disturbing existing call sites.

apply_core_enable and start_dma_channel become unused after this
commit; their deletion lands in the next commit so the bridge-
test gate has them on disk for diagnosis if anything regresses.

Generated using Claude Code.
EOF
)"
```

Verify with `git log --oneline -3`.

---

## Task 3: Bridge-test gate (intermediate, NOT a commit)

**Goal:** Run the bridge-test suite to confirm the refactor didn't regress any hardware-equivalent flow before deleting the dead code.

- [x] **Step 3.1: Rebuild the FFI crate so bridge tests pick up the new code**

Run: `cargo build -p xdna-emu-ffi 2>&1 | tail -5`
Expected: clean build.

- [x] **Step 3.2: Run the bridge test suite**

Run: `./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/d3-bridge-after-commit2.log`

This takes 15-30 minutes. While it runs, do not start any other hardware-touching test. Once complete:

- [x] **Step 3.3: Inspect the bridge results**

Run: `tail -60 /tmp/claude-1000/d3-bridge-after-commit2.log`
Expected: PASS count = 116/118 (the two pre-existing failures are HW-side, unrelated to D.3). If the PASS count is lower, STOP. The most likely culprit is a side-effect-ordering issue we missed in the spec; read the failed test names, locate them in `mlir-aie/build/test/npu-xrt/<name>/`, and diagnose. Do NOT proceed to commit 3 — `apply_core_enable` / `start_dma_channel` are still on disk and provide a comparison reference.

If PASS count matches baseline, proceed.

---

## Task 4: Commit 3 — Delete the now-dead typed handlers

**Files:**
- Modify: `src/device/state/compute.rs` (delete `apply_core_enable` and `start_dma_channel`, update doc comments)
- Modify: `src/device/state/cdo.rs` (no doc-comment change here; that already happened in commit 2)

**Goal:** Remove `apply_core_enable` and `start_dma_channel`, plus update the inline doc comments in `write_core_register`, `write_dma_channel`, and their masked twins to drop the "non-CDO path" qualifier (the CDO and non-CDO paths are now indistinguishable).

- [x] **Step 4.1: Verify no callers of `apply_core_enable` and `start_dma_channel` remain**

Run: `grep -rn "apply_core_enable\|start_dma_channel" src/ --include="*.rs"`
Expected output: only the function definitions themselves (in `src/device/state/compute.rs`) and possibly doc-comment references that mention them. No actual call sites.

If a call site is found that we missed, stop and route it through `write_register` first.

- [x] **Step 4.2: Delete `apply_core_enable`**

In `src/device/state/compute.rs`, locate `apply_core_enable` (currently at lines 387-408 — a doc comment plus the function body). Delete the entire block (doc comment + function). The lines immediately preceding it (`mask_write_dma_channel`'s closing brace) and following it (`mask_write_dma_channel` documentation continues on the next item) should remain intact.

After deletion, the section between `mask_write_dma_channel` (which closes around line 385) and the next function should flow smoothly. Verify with `grep -n "fn " src/device/state/compute.rs | head -10` that the function list looks intact (no stray fragments).

- [x] **Step 4.3: Delete `start_dma_channel`**

In `src/device/state/compute.rs`, locate `start_dma_channel` (currently at lines 292-385 — doc comment plus function body). Delete the entire block. Verify the surrounding functions (`write_dma_channel` before it, `apply_core_enable`'s old position which was already deleted in Step 4.2, then `mask_write_dma_channel`) are still well-formed.

- [x] **Step 4.4: Update the doc comment on `write_dma_channel`**

In `src/device/state/compute.rs`, the doc comment for `write_dma_channel` (currently at lines 162-170) reads:

```
    /// Write to a compute-tile DMA channel register.
    ///
    /// Handles both Ctrl and Start_Queue writes. The CDO path now
    /// promotes Start_Queue writes to `DeviceOp::DmaStart` and applies
    /// them via `start_dma_channel` from `apply_device_op`; the
    /// Start_Queue branch below remains live for non-CDO paths (NPU
    /// instructions via `write_tile_register`, control packets, FFI)
    /// that still feed raw register writes through this function. See
    /// the commit message on Stage 8b Half 2 for the rationale.
```

Replace with:

```
    /// Write to a compute-tile DMA channel register.
    ///
    /// Handles both Ctrl and Start_Queue writes. This is the single
    /// source of truth for compute DMA channel effects; both the CDO
    /// path (via `apply_device_op::DmaStart` -> `write_register`)
    /// and non-CDO paths (NPU instructions, control packets, FFI
    /// via `write_tile_register`) reach this function.
```

- [x] **Step 4.5: Update the doc comment on `mask_write_dma_channel`**

In `src/device/state/compute.rs`, locate `mask_write_dma_channel`'s doc comment (around the function's current location after deletions, formerly at lines 410-416). It reads:

```
    /// Masked write to a compute-tile DMA channel register.
    ///
    /// MaskWrite to Start_Queue is not promoted to `DeviceOp::DmaStart`
    /// (promotion gates on Write, not MaskWrite), and non-CDO paths
    /// (NPU instructions, control packets, FFI) also reach this
    /// function directly. The legacy Start_Queue branch below remains
    /// live for both cases.
```

Replace with:

```
    /// Masked write to a compute-tile DMA channel register.
    ///
    /// Reached by both CDO MaskWrites (via `apply_device_op::RegMask`
    /// -> `mask_write_register`) and non-CDO paths (NPU instructions,
    /// control packets, FFI). MaskWrite is never promoted to a typed
    /// op; mask-blend correctness lives in `mask_write_core_register`
    /// and per-channel handlers, applied to the merged value here.
```

- [x] **Step 4.6: Update the doc comment on `write_core_register`**

In `src/device/state/compute.rs`, the doc comment for `write_core_register` (currently lines 612-619) reads:

```
    /// Write to a core register.
    ///
    /// The CORE_CONTROL branch is also exercised by non-CDO paths
    /// (NPU instructions, control packets, FFI via
    /// `write_tile_register`); the CDO path now promotes
    /// CORE_CONTROL writes to `DeviceOp::CoreEnable` and handles them
    /// via `apply_core_enable`. The branch below stays live for the
    /// non-CDO callers.
```

Replace with:

```
    /// Write to a core register.
    ///
    /// The CORE_CONTROL branch is the single source of truth for
    /// core-enable effects: both the CDO path (via
    /// `apply_device_op::CoreEnable` -> `write_register`) and
    /// non-CDO paths (NPU instructions, control packets, FFI via
    /// `write_tile_register`) reach this function.
```

- [x] **Step 4.7: Build and run the full library suite**

Run: `cargo build --lib 2>&1 | tail -5`
Expected: clean build.

Run: `cargo test --lib 2>&1 | tail -15`
Expected: 0 failures, total test count unchanged from end of Task 2.

- [x] **Step 4.8: Commit**

Run:
```
git add src/device/state/compute.rs
git commit -m "$(cat <<'EOF'
chore(d3): delete dead typed-handler code

apply_core_enable (~22 lines) and start_dma_channel (~94 lines)
become unreachable once apply_device_op routes CoreEnable/DmaStart
through write_register (commit 2). Delete them along with their
doc comments.

Update inline comments on write_core_register, write_dma_channel,
and mask_write_dma_channel to drop the "non-CDO path" qualifier --
the CDO and non-CDO paths are now indistinguishable from these
functions' point of view.

cargo test --lib green; bridge-test gate passed between commits 2
and 3 with the same baseline as pre-refactor.

Generated using Claude Code.
EOF
)"
```

Verify with `git log --oneline -4` that the three D.3 commits land in sequence.

---

## Task 5: Final bridge gate

**Goal:** Run the bridge suite once more to confirm the deletions didn't regress anything.

- [x] **Step 5.1: Rebuild FFI crate**

Run: `cargo build -p xdna-emu-ffi 2>&1 | tail -5`
Expected: clean build.

- [x] **Step 5.2: Run bridge tests**

Run: `./scripts/emu-bridge-test.sh 2>&1 | tee /tmp/claude-1000/d3-bridge-after-commit3.log`
Wait for completion (15-30 minutes).

- [x] **Step 5.3: Verify baseline**

Run: `tail -60 /tmp/claude-1000/d3-bridge-after-commit3.log`
Expected: PASS count matches the post-commit-2 run (116/118).

If anything regressed between commits 2 and 3, the most likely culprit is a stale doc comment or import that was deleted along with the dead code. Check `git diff HEAD~1 HEAD` for `src/device/state/compute.rs`.

---

## Self-Review

**Spec coverage:**
- ✅ "Three regression tests (two RED, one GREEN safety-net)" — Task 1, Steps 1.2-1.4. (One GREEN test became two for clarity: CORE_CONTROL safety + Start_Queue safety.)
- ✅ "Rewrite apply_device_op CoreEnable/DmaStart arms" — Task 2, Steps 2.3-2.4.
- ✅ "Stop promoting MaskWrite-to-CORE_CONTROL" — Task 2, Step 2.6.
- ✅ "Bridge-test gate between commits 2 and 3" — Task 3.
- ✅ "Delete now-dead apply_core_enable and start_dma_channel" — Task 4, Steps 4.2-4.3.
- ✅ "Variant shapes preserved" — Step 2.3 ignores `enabled` but doesn't change the variant; Step 2.4 doesn't touch `DmaStart`'s shape.
- ✅ "Update inline doc comments" — Steps 2.5, 4.4, 4.5, 4.6.
- ✅ "Acceptance: 116/118 bridge baseline" — Task 5.

**Placeholder scan:** None. Every code step has the actual code; every command has expected output.

**Type consistency:** `start_queue_offset(row, channel, dir)` signature defined in Step 2.1 matches how it's called in Step 2.4. `encode_addr(*tile, offset)` matches the existing helper at `src/device/state/cdo.rs:181`. `DeviceOp::CoreEnable { tile, enabled: _, value }` (Step 2.3) and `DeviceOp::DmaStart { tile, channel, dir, bd_id }` (Step 2.4) match the variant definitions in `src/device/ops.rs:68,80`.

---

## Done criteria

- Three commits land cleanly on `dev`:
  1. `test(d3): regression tests for universal register bus refactor`
  2. `refactor(d3): route CoreEnable/DmaStart through the universal register bus`
  3. `chore(d3): delete dead typed-handler code`
- `cargo test --lib` green at the end (and at every commit boundary except commit 1, where the two RED tests are intentionally failing).
- Bridge test suite at the same baseline (116/118 PASS) after commit 2 and again after commit 3.
- `apply_core_enable` and `start_dma_channel` no longer exist (`grep -rn 'apply_core_enable\|start_dma_channel' src/` returns nothing).
- `apply_device_op` doc comment in `src/device/state/cdo.rs` reflects the universal-bus architecture.
