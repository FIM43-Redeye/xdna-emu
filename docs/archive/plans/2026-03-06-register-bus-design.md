# Hardware-Modeled Register Bus

**Date:** 2026-03-06
**Status:** Approved
**Scope:** Write path unification, executor simplification, test consistency

## Problem

The emulator has five write paths to tile registers, with overlapping side
effects that cause double DMA enqueue and other subtle bugs:

1. **CDO commands** -> `state.write_register(address, value)` (full dispatch)
2. **NPU executor** -> `device.write_tile_register()` -> state dispatch
   THEN `check_dma_trigger()` (redundant enqueue)
   THEN `sync_bd_from_registers()` (redundant BD parse)
3. **Control packets** -> `device.write_tile_register()` -> state dispatch
4. **FFI** -> `device.write_tile_register()` -> state dispatch
5. **Tests** -> `tile.write_register()` (bypasses state dispatch entirely)

Additionally, `state.write_register()` calls `tile.write_register()` as a
fallthrough, creating duplicate handling for DMA BDs, locks, and DMA channels.

The immediate regression: commit 103d861 unified NPU executor writes through
the state dispatch, but left `check_dma_trigger()` in place. Every DMA start
queue write now enqueues the task twice (once via `write_dma_channel`, once
via `check_dma_trigger`), breaking 7 previously-passing bridge tests.

## Design Principle

**Mirror the hardware architecture.** The real NPU has one register bus per
tile. A write to any offset produces the same side effects regardless of
source. The bus is dumb about callers; callers are dumb about hardware.

## Architecture

### Public API

Three methods on `DeviceState`, used by ALL callers:

```
DeviceState::write_register(col, row, offset, value)
DeviceState::mask_write_register(col, row, offset, mask, value)
DeviceState::read_register(col, row, offset) -> u32
```

CDO commands that arrive as encoded 32-bit addresses decode internally
before calling these.

### Bus Dispatch

```
write_register(col, row, offset, value)
  |
  +-- tile.registers.insert(offset, value)     [raw storage, always first]
  |
  +-- match module_from_offset(row, offset):
  |     DmaBd              -> handle_dma_bd_write(...)
  |     DmaChannel         -> handle_dma_channel_write(...)
  |     ShimDmaBd          -> handle_shim_dma_bd_write(...)
  |     ShimDmaChannel     -> handle_shim_dma_channel_write(...)
  |     MemTileDmaBd       -> handle_memtile_dma_bd_write(...)
  |     MemTileDmaChannel  -> handle_memtile_dma_channel_write(...)
  |     Locks              -> handle_lock_write(...)
  |     MemTileLocks       -> handle_memtile_lock_write(...)
  |     StreamSwitch       -> handle_stream_switch_write(...)
  |     MemTileStreamSwitch-> handle_memtile_stream_switch_write(...)
  |     CoreModule         -> handle_core_write(...)        [cascade, core ctrl]
  |     ProgramMemory      -> handle_program_memory_write(...)
  |     TraceModule        -> handle_trace_write(...)       [NEW]
  |     ShimMux            -> handle_shim_mux_write(...)    [NEW]
  |     EventModule        -> handle_event_write(...)       [NEW]
  |     _                  -> ()                            [raw storage only]
  |
  +-- propagate_broadcasts(col, row)           [always last]
```

Raw storage happens once at the top. Module handlers write structured state
(DMA engine, lock arbiter, stream switch tables). No handler calls
`tile.write_register()`.

### mask_write_register

Reads current value from `tile.registers`, applies mask, dispatches the
merged value through the same module handlers. No separate mask_write
variants needed in handlers.

### RegisterModule Routing

Pass `row` (and optionally tile_type) into `module_from_offset()` to
disambiguate shim-specific register ranges. Eliminates the current
fallthrough hacks in the `_ =>` match arm. Best-effort; if complex,
keep the fallthrough and fix in the project-wide cleanup.

## Component Changes

### tile.write_register() -- DELETED

Currently handles 10 concerns. Each moves to a bus module handler:

| Concern | New location |
|---------|-------------|
| registers.insert() | Bus top-level (before dispatch) |
| DMA BD raw fields | handle_dma_bd_write (existing) |
| Lock values | handle_lock_write (existing) |
| DMA channel ctrl/start_queue | handle_dma_channel_write (existing) |
| Cascade config | handle_core_write (existing) |
| Shim mux/demux | handle_shim_mux_write (new module) |
| Lock overflow/underflow | handle_lock_write (extend) |
| Trace registers | handle_trace_write (new module) |
| Event port selection | handle_event_write (new module) |
| Event broadcast / Event_Generate | handle_event_write (new module) |

The tile struct becomes a passive data container: register HashMap, data
memory, program memory, and structural fields. `tile.registers` becomes
`pub(crate)` for direct bus access.

### NPU Executor -- SIMPLIFIED

**Deleted:**
- `check_dma_trigger()` (~130 lines) -- redundant with bus's DMA channel handler
- `sync_bd_from_registers()` (~40 lines) -- redundant with bus's BD handler + dirty/reparse
- All `propagate_broadcasts()` calls -- handled by bus

**Added:**
- Pre-flight queue depth check (~15 lines) before writing DMA start queue
  registers. If queue full, set BlockedOnQueue state and don't write.
  Matches real hardware where firmware polls Task_Queue_Size before
  writing Start_Queue (aie-rt `_XAieMl_DmaWaitForBdTaskQueue`).

**Result:**
```rust
fn execute_write32(...) {
    let (col, row, offset) = decode_npu_address(reg_off);

    if is_data_memory_offset(...) {
        tile.write_data_u32(offset, value);
        return Ok(());
    }

    // Pre-flight: block if writing to a full DMA task queue
    if self.would_block_on_queue(col, row, offset, value, device) {
        return Ok(());  // state already set to BlockedOnQueue
    }

    // Single write through the bus. Done.
    device.write_register(col, row, offset, value);
    Ok(())
}
```

The executor becomes a thin loop: decode address, check preconditions,
call bus. No knowledge of DMA internals, BD parsing, or any hardware
subsystem.

### Test Infrastructure

New convenience constructor:
```rust
impl DeviceState {
    pub fn new_for_test() -> Self { ... }
}
```

All tests that call `tile.write_register()` migrate to
`device.write_register(col, row, offset, value)`. Roughly 20 mechanical
updates across array.rs, tile.rs, and interpreter/execute/memory.rs.

## Migration Order

1. Add `registers.insert()` at top of state.write_register, before dispatch
2. Move tile.write_register() concerns into bus module handlers
3. Delete tile.write_register()
4. Remove check_dma_trigger and sync_bd_from_registers from executor
5. Add pre-flight queue check to executor
6. Rename write_tile_register -> write_register (public API)
7. Update CDO path to use (col, row, offset) instead of encoded address
8. Migrate tests to DeviceState::new_for_test() + bus writes
9. Clean up RegisterModule routing (shim disambiguation)

## Expected Outcome

- ~200 lines deleted (check_dma_trigger, sync_bd_from_registers, tile.write_register)
- ~30 lines added (pre-flight check, test helper, module routing)
- Double-enqueue regression eliminated
- All future write paths forced through the bus
- 14/40 bridge test pass rate restored (pre-regression baseline)
- ctrl_packet_reconfig also passing (from the core enable fix)
