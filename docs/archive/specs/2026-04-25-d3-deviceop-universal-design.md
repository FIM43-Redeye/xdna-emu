# D.3 Design Sketch: extending DeviceOp to non-CDO write paths

**Status:** design only -- not implemented in the 2026-04-25 hygiene
pass. Subsystem 8 gate flagged this as a separate audit-first refactor;
this document refines that note into a concrete plan for a future
brainstorm cycle.

## Problem

`device::state` exposes four write entry points used by every register
write that lands on a tile:

- `write_core_register(col, row, offset, value)`
- `mask_write_core_register(col, row, offset, mask, value)`
- `write_dma_channel(col, row, offset, value)`
- `mask_write_dma_channel(col, row, offset, mask, value)`

Each carries an offset-dispatch branch for the well-known control
register on its subsystem:

- `write_core_register`: `cm::CORE_CONTROL` -> stores `value` and
  applies enable/disable side effect.
- `write_dma_channel`: `Start_Queue` (compute) -> enqueues a DMA task,
  reparses dirty BDs, sets per-channel task config.

These branches are the load-bearing logic that `apply_device_op` now
covers via `DeviceOp::CoreEnable` / `DeviceOp::DmaStart`. The CDO
path is fully typed; non-CDO callers still hit the branches:

- `src/interpreter/engine/coordinator.rs` -- 3 sites for NPU
  instruction register writes.
- `src/npu/executor.rs` -- 6 sites: `Write32`, `BlockWrite`,
  `MaskWrite`, `DdrPatch` NPU instructions.
- `src/device/state/dispatch.rs` -- the central `write_register` /
  `mask_write_register` dispatchers, reached by control packets and
  the FFI tile-register entry points.

So the offset-dispatch branches still exist (because non-CDO callers
need them), even though the CDO path doesn't. The branches are
duplicated logic with `apply_core_enable` / `start_dma_channel`.

## Goal

Move the boundary so `DeviceOp` is the universal device-write API:
- Non-CDO callers lower (address, value) to `DeviceOp` and apply via
  `apply_device_op`.
- The four `write_*_register` functions either disappear entirely or
  shrink to "deliver a `RegWrite`/`RegMask` op" thin wrappers without
  any offset-dispatch branches.

End state: the only place CORE_CONTROL or Start_Queue gets special-
cased is the lowering helper. State sees only typed ops.

## Why this is non-trivial

Three complications make this a real refactor, not a sed-replace.

### 1. Tile-local side effects

`dispatch.rs::write_register` post-processes every write with
`apply_tile_local_effects(col, row, offset, value)` and
`propagate_broadcasts(col, row)`. Trace config, edge detection, event
port selection, cascade config, shim mux, lock overflow/underflow
clear, event broadcast, and Event_Generate all live there.

If non-CDO paths route through `apply_device_op`, the typed
RegWrite/RegMask handlers must replicate those calls -- *or* we move
the side-effect dispatch up so it runs after every `apply_device_op`
returns. The latter is cleaner; the former subtler. Either way,
ordering matters: hardware fires side effects on the register write,
not before it, so lock writes need to happen before broadcast
propagation reads the lock state.

### 2. Subsystem dispatch

`write_register` decodes the address into `(tile, offset)` and routes
by `subsystem_from_offset` to one of: `write_dma_channel`,
`write_dma_bd`, `write_shim_dma_channel`, `write_core_register`,
`write_stream_switch`, `write_memtile_stream_switch`, program memory,
data memory.

Most subsystem branches don't have a typed DeviceOp equivalent.
`RegWrite { tile, offset, value }` is a generic carrier; without a
matching subsystem-aware handler, we'd need either:

- **Option A:** keep `write_register` as the dispatcher and have
  `apply_device_op` call back into it for `RegWrite` / `RegMask`.
  `apply_device_op` becomes a wrapper that adds typed-op promotion;
  the actual register-write logic stays in dispatch.rs.
- **Option B:** grow DeviceOp variants for each subsystem (e.g.,
  `DmaBdWrite`, `StreamSwitchWrite`). Larger refactor; cleaner end
  state.

Option A is the path of least churn and matches what `apply_cdo`
effectively does today (it routes RegWrite back through
`write_register`). Recommend Option A.

### 3. Lowering helper visibility and placement

`crate::parser::cdo::semantics::lower_write` and `lower_mask_write`
are private inside the parser module. The non-CDO paths would need
them either:

- **Option a:** Made `pub` in their current location. Implies the
  parser module is the home of (address, value) -> DeviceOp lowering,
  even when the input isn't CDO.
- **Option b:** Moved to `device::ops` (alongside DeviceOp itself) or
  a new `device::lower` module. Cleaner conceptual home: lowering is
  about device semantics, not parser concerns.

Option (b) is the right end state. Option (a) is a stepping stone.

## Migration plan (sketch)

1. **Move/expose the lowering helpers.** Either pub them in
   `parser::cdo::semantics` or relocate to `device::ops::lower`.
   Single commit, no behavior change.

2. **Convert `dispatch.rs::write_register` to lower-then-apply.** For
   the Processor + DmaTile branches, where the typed promotion
   applies, lower into DeviceOp first and route through
   `apply_device_op`. Other branches stay direct. Preserve the
   tile-local effects + broadcast propagation post-processing.

3. **Convert `dispatch.rs::mask_write_register` similarly.**

4. **Audit `coordinator.rs` 3 sites.** They likely already go through
   `dispatch.rs::write_register`; if so, no change needed. If they
   call `write_core_register` directly, route through `write_register`
   first (or through the lowering helper).

5. **Audit `executor.rs` 6 sites.** Same pattern: ensure they go
   through `dispatch.rs::write_register` rather than directly to the
   subsystem-specific writers.

6. **Audit control-packet dispatch.** Same pattern.

7. **Remove offset-dispatch branches from
   `write_{core_register,dma_channel}`.** With all non-CDO callers
   now going through `apply_device_op`, the CORE_CONTROL and
   Start_Queue branches are unreachable. Delete them. The
   `_register` functions shrink to "store value in register map +
   trigger any per-write logging."

8. **Run the bridge test suite.** Side-effect ordering changes are
   the highest-risk slice of this refactor; bridge tests are the
   gate.

Each step is its own commit. Step 7 is the visible cleanup; steps
1-6 are the enabling work.

## Acceptance

- Zero offset-dispatch branches in `write_core_register` /
  `mask_write_core_register` / `write_dma_channel` /
  `mask_write_dma_channel`.
- `cargo test --lib` green.
- Bridge test green at the same baseline as pre-refactor.
- Side-effect ordering verified by spot-testing a known
  CORE_CONTROL-write-then-broadcast sequence.

## Why deferred from 2026-04-25 hygiene

Per the original Phase 1 spec, "each [hygiene item] warrants its own
brainstorm + plan cycle." The 2026-04-25 hygiene pass closed D.1
(dead-code), D.2 (stale examples), D.4 (memtile/shim DMA promotion in
the *CDO* lower path), D.5 (arch-generic MemoryRegion::from_address),
D.6 (TABLEGEN_210_PREFIX), D.7 (misc Phase 1a follow-ups). D.3 is
genuinely a Subsystem-9-shaped piece: 9+ call sites, side-effect
ordering risk, decision required on Option A vs B for variant scope.
Folding it into a hygiene pass would either cut corners (Option A
without analysis) or balloon scope (Option B with new variants).

Picking it up next: read this doc + the Subsystem 8 audit's
"Load-bearing finding: non-CDO write paths still bypass DeviceOp"
section, then run the brainstorming skill on the Option A vs B
question before choosing the migration plan above.
