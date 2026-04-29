# D.3 Design: Universal register bus

**Status:** approved design, ready for implementation plan.
**Supersedes:** `2026-04-25-d3-deviceop-universal-design.md`. The
earlier sketch is kept on disk as a historical record (its framing
turned out to be partly out of date and its recommendation has been
reversed -- see "Why the 2026-04-25 sketch was reversed" at the
bottom of this document).

## Why this exists

Two bugs and a structural asymmetry, all rooted in the same place:
the CDO path's typed-promotion handlers (`apply_core_enable`,
`start_dma_channel`) bypass the central register bus. Every non-CDO
caller -- NPU instructions, control packets, FFI -- already routes
through `write_register` / `mask_write_register` (the central
dispatcher). The CDO path is the outlier.

### Bug 1: CORE_CONTROL readback divergence

A CDO write to `CORE_CONTROL` lands in `apply_core_enable`, which
sets `tile.core.control = value` but never stores into
`tile.registers`. A subsequent control-packet read of `CORE_CONTROL`
goes through `Tile::read_register_pure`, which has typed shortcuts
for DMA channel control / start_queue / BD words / lock values but
**not** for `CORE_CONTROL` -- so it falls back to
`self.registers.get(&offset)` and returns 0.

A non-CDO write to `CORE_CONTROL` goes through `write_register`,
which stores raw at `dispatch.rs:36` *and* runs the offset-branch in
`write_core_register`. Reads return the right value.

So the same write (CDO vs non-CDO) produces different observable
state.

### Bug 2: MaskWrite-to-CORE_CONTROL bit corruption

`lower_mask_write` promotes a CDO MaskWrite to `CORE_CONTROL` with
`mask & 1 != 0` into `DeviceOp::CoreEnable`. The lowering drops the
mask. The doc comment claims `device::state` will mask-blend on
apply, but `apply_core_enable` does
`tile.core.control = value` -- **no mask-blend**, so bits not covered
by the mask get clobbered to whatever was in `value`.

The non-CDO path's `mask_write_core_register` does
`tile.core.control = (tile.core.control & !mask) | (value & mask)`
correctly.

So a CDO MaskWrite to `CORE_CONTROL` with a partial mask currently
corrupts unmasked bits.

### The structural asymmetry

Both bugs are surface symptoms of one design choice: state currently
sees `apply_core_enable` and `start_dma_channel` as parallel handlers
to the register bus. They skip raw register storage,
`apply_tile_local_effects`, and `propagate_broadcasts`. The
register-bus path does all three, plus runs the existing
offset-dispatch branches in `write_core_register` /
`write_dma_channel` / their masked twins -- which already do the
"typed effect" (set `tile.core.control`, push
`pending_core_enables`, decode Start_Queue and enqueue DMA tasks)
correctly.

So the typed handlers are not just bypassing the bus; they are also
*duplicating* logic that lives in the offset-dispatch branches.

## What this design changes

**Principle:** real silicon has one register bus. Every register
write -- CDO, NPU instruction, control packet, FFI -- lands on the
same bus and produces the same side effects. The emulator's
`write_register` / `mask_write_register` is that bus.

**Role of `DeviceOp`:** parser-side semantic vocabulary, not a
parallel device-level vocabulary. `CoreEnable` and `DmaStart` are
*labels* documenting what the parser knows about a write, not
handlers that bypass the bus. State sees register-bus writes.

**Room preserved for typed-handler return:** `DeviceOp::CoreEnable`
keeps its current shape (`{ tile, enabled, value }`) -- we just stop
*using* `enabled` on the apply side. If we later want mask-aware
promotion or non-CDO sources of typed promotion, adding
`mask: Option<u32>` to `CoreEnable` is a non-breaking addition. The
typed handlers (`apply_core_enable`, `start_dma_channel`) get
deleted in commit 3, but reintroducing them later is a localized
edit.

## Commit structure

Three commits. Bisect-friendly: each commit is a coherent unit, and
the bridge-test gate between commits 2 and 3 means the dead code is
still on disk if commit 2 breaks something.

### Commit 1: failing regression tests (RED)

Three tests in `src/device/state/tests.rs`.

#### Test A: `core_control_cdo_write_is_readable_via_register_bus`

**Currently RED.** Build a CDO with a Write to `CORE_CONTROL` on a
compute tile, apply it, then call `tile.read_register_pure(CORE_CONTROL)`
and assert it returns the written value. Today returns 0 because
`apply_core_enable` skips `tile.registers`.

#### Test B: `core_control_cdo_mask_write_preserves_unmasked_bits`

**Currently RED.** Pre-set `tile.core.control = 0xABCD_0001` directly,
build a CDO MaskWrite with `mask=0x1, value=0x0`, apply, assert
`tile.core.control == 0xABCD_0000`. Today returns `0x0000_0000`
because `apply_core_enable` overwrites the full word.

#### Test C: `register_bus_write_to_core_control_does_not_trigger_unrelated_tile_effects`

**Currently GREEN, stays GREEN.** This is the safety-net test for the
side-effect-ordering question: prove that running
`apply_tile_local_effects` and `propagate_broadcasts` for
`CORE_CONTROL` and Start_Queue offsets is safe (i.e., none of the
side-effect handlers match those offsets).

Implementation: snapshot every field that
`apply_tile_local_effects` and `propagate_broadcasts` can touch:
- `tile.cascade_input_dir`, `tile.cascade_output_dir`
- `tile.shim_mux` config (if shim)
- lock overflow / underflow bits
- per-tile `core_perf_counters` and `mem_perf_counters` state
- core/memory/memtile trace register state
- broadcast-event state on neighboring tiles in the column

Call `write_tile_register(col, row, CORE_CONTROL, value)` on a
compute tile; assert every snapshotted field is unchanged. Repeat for
`COMPUTE_DMA_S2MM_0_START_QUEUE` and one memtile Start_Queue offset.

Test C is the invariant-prover. Today the non-CDO path already runs
`apply_tile_local_effects` on these offsets, so the assertion holds;
after commit 2 the CDO path also runs it, and the test continues to
pass -- demonstrating that the new exposure is safe.

### Commit 2: route CoreEnable/DmaStart through the register bus

Two files touched.

#### `src/device/state/cdo.rs`

`apply_device_op::CoreEnable` arm rewritten:

```rust
DeviceOp::CoreEnable { tile, enabled: _, value } => {
    let addr = encode_addr(*tile, xdna_archspec::aie2::registers::CORE_CONTROL);
    self.write_register(addr, *value)?;
}
```

The `enabled` field is intentionally ignored on apply
(`write_core_register`'s CORE_CONTROL branch derives enabled from
`value & 1`); the field stays in the variant as a parser-side
semantic marker / room for option (a).

`apply_device_op::DmaStart` arm rewritten:

```rust
DeviceOp::DmaStart { tile, channel, dir, bd_id } => {
    let offset = start_queue_offset(*tile, *channel, *dir, &self.array)?;
    let addr = encode_addr(*tile, offset);
    self.write_register(addr, *bd_id)?;
}
```

`start_queue_offset` is a small free function added to `cdo.rs` that
reconstructs the Start_Queue offset from `(tile, channel, dir,
tile_kind)` using `regdb::device_reg_layout()`. Tile kind comes from
`self.array.arch().tile_kind(tile.col, tile.row)`. Returns the
correct offset for compute / memtile / shim tiles.

#### `src/parser/cdo/semantics.rs`

`lower_mask_write`'s CORE_CONTROL branch removed: a MaskWrite to
`CORE_CONTROL` always lowers to `RegMask`. Update the comment to
explain the choice (we keep mask-blending in
`mask_write_core_register` rather than duplicating it in
`apply_core_enable`; future work may reintroduce mask-aware
promotion, in which case add `mask: Option<u32>` to `CoreEnable`).

After commit 2: tests A and B pass; test C continues to pass; the
existing CDO test suite (`cdo.rs`, `semantics.rs`) and full
`cargo test --lib` are green.

#### Bridge gate (between commits 2 and 3)

Run `./scripts/emu-bridge-test.sh` after commit 2 lands. Baseline
must match pre-refactor (116/118 PASS, single pre-existing HW
failure). If anything breaks, the dead code in `compute.rs`
(`apply_core_enable`, `start_dma_channel`) is still present and
provides a comparison reference for diagnosis.

Only proceed to commit 3 once the bridge gate is clean.

### Commit 3: delete dead code

`src/device/state/compute.rs`:
- Delete `apply_core_enable` (~15 lines) -- redundant with
  `write_core_register`'s CORE_CONTROL branch.
- Delete `start_dma_channel` (~80 lines) -- redundant with
  `write_dma_channel`'s Start_Queue branch.

Verify with `grep -rn 'apply_core_enable\|start_dma_channel'
src/` that no callers remain (they only existed in
`apply_device_op`).

`src/device/state/cdo.rs`: update the `apply_device_op` doc comment
to reflect the new architecture -- state sees register-bus writes;
typed promotions are parser semantic markers; the offset-dispatch
branches in `write_core_register` / `write_dma_channel` are the
single source of truth for CORE_CONTROL / Start_Queue effects.

Update inline doc comments in `compute.rs::write_core_register` and
`compute.rs::write_dma_channel` (and their masked twins) to drop the
"non-CDO path" qualifier -- they're the only path now.

`cargo test --lib` green; full bridge run green.

## Acceptance

- Tests A and B pass (were RED before commit 2).
- Test C passes throughout (invariant proven, never regressed).
- `cargo test --lib` green at all three commit boundaries (RED tests
  in commit 1 are RED only at that commit; commit 2 turns them
  GREEN).
- Bridge test suite at the same baseline as pre-refactor (116/118
  PASS) after commit 2 and again after commit 3.
- `apply_core_enable` and `start_dma_channel` no longer exist after
  commit 3.
- `apply_device_op` doc comment reflects the universal-bus
  architecture.

## What this is NOT

To prevent scope drift -- explicit non-goals:

- **Not** changing the `DeviceOp` enum shape. `CoreEnable` keeps
  `enabled` and `value`; `DmaStart` keeps `channel`, `dir`, `bd_id`.
- **Not** moving the lowering helpers (`lower_write`,
  `lower_mask_write`) to `device::ops::lower` or making them public.
  Non-CDO callers don't need them; they go through
  `write_tile_register` directly.
- **Not** touching call sites in `coordinator.rs`, `executor.rs`, or
  control-packet dispatch. They already route through
  `write_tile_register` -> `write_register`.
- **Not** removing the offset-dispatch branches from
  `write_core_register` / `write_dma_channel`. They are the
  load-bearing logic for the universal-bus path; deleting them is
  the opposite direction (and was the 2026-04-25 sketch's
  recommendation, which this design reverses).
- **Not** introducing new `DeviceOp` variants for other subsystems
  (the original sketch's "Option B"). Out of scope.

## Why the 2026-04-25 sketch was reversed

The sketch's framing -- "non-CDO call sites bypass DeviceOp" -- was
already false at the time of writing: every non-CDO call site goes
through `write_tile_register` -> `write_register`. The sketch
recommended pushing harder in the direction of "make DeviceOp the
universal device-level vocabulary," which would have required
exposing lowering helpers, auditing 9+ call sites, and verifying
side-effect ordering across all of them.

Reading the code with fresh eyes on 2026-04-29, the simpler picture
emerged: real silicon's register bus IS the universal vocabulary;
DeviceOp is a parser-side abstraction. The typed promotions
duplicate logic that lives in the offset branches. Routing the
typed apply paths through the register bus instead removes the
duplication, fixes two latent bugs, and shrinks the refactor surface
from 9+ call sites to two functions in one file.

The earlier sketch is preserved on disk for historical reference. If
the universal-bus direction proves wrong somehow, returning to the
sketch's direction is still possible -- the variant shapes are
intentionally untouched in this refactor.
