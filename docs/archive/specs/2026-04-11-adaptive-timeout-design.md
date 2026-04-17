# Adaptive Timeout (Stall Detection)

**Date:** 2026-04-11
**Status:** Approved

## Problem

The emulator uses a fixed cycle limit (`max_cycles`, default 500,000) to detect
hangs. This causes false timeouts on tests that do legitimate heavy scalar work:

- `dma_task_large_linear` (Peano): 65,536 element-wise add operations. Completed
  only 2/32 tiles in 500K cycles. Core was actively processing every cycle.
- `objectfifo_repeat/init_values_repeat` (Peano): 4,096 element-wise copy, 4
  repeats. Core couldn't finish even one tile within the cycle budget.

Both tests pass on real hardware. The emulator is functionally correct -- it
simply runs out of cycle budget before finishing.

Raising the fixed limit penalizes true deadlock detection: a test stuck in a
real deadlock would have to burn through millions of cycles before timing out.

## Design

Replace the fixed cycle limit with **monotonic progress detection**. Track
indicators that can only advance in a correctly-functioning program. If none
of them advance for a configurable number of cycles, declare the system
stalled.

### Monotonic Progress Indicators

Two indicators, tracked per cycle:

1. **Total DMA bytes transferred** -- sum of `bytes_transferred` across all
   DMA channels on all tiles (shim, compute, memtile). This catches:
   - Active data movement at any level (shim edges, internal ping-pong)
   - Mid-BD stalls (stream starvation, backpressure) -- `bytes_transferred`
     flatlines when no data moves
   - Shim output completion progress (the ultimate test-done signal)

2. **Core lock release count** -- cumulative number of lock release
   instructions executed across all cores. This catches:
   - Infinite compute loops (core executes instructions but never releases a
     lock)
   - Lock-gated forward progress (each release enables the next DMA phase)

**Progress rule:** if EITHER indicator advances on a given cycle, reset the
stall counter to zero. If NEITHER advances, increment the stall counter. When
the stall counter reaches `stall_threshold`, declare timeout.

### Why These Two Are Sufficient

- **Deadlock** (all actors blocked on each other): zero bytes transferred,
  zero lock releases. Detected in `stall_threshold` cycles.
- **Livelock** (core in infinite loop, locks cycling without output): lock
  releases stop (core never reaches a release instruction), DMA bytes stop
  flowing once buffers drain. Detected.
- **Slow but working** (heavy scalar compute): core is advancing toward a
  lock release, DMA bytes flow between tiles. Stall counter keeps resetting.
  Test runs to completion.
- **Mid-BD stall** (DMA waiting for stream data): `bytes_transferred` stops
  advancing, but the core continues processing and releasing locks. Stall
  counter resets on each lock release.
- **Large DMA transfer** (long BD with no lock releases): `bytes_transferred`
  advances every cycle as data moves. Stall counter stays at zero.

### What About Lock Cycling / BD Looping?

Compute tile DMAs can legitimately cycle through the same BDs forever
(ping-pong buffering). This causes `bytes_transferred` to keep climbing.
This is fine: the shim output DMA has a finite transfer size and its progress
is included in the same byte sum. If the interior is cycling without driving
shim output forward, eventually the shim stops advancing, the core stalls
waiting on a lock, lock releases stop, and the stall counter fires.

## Configuration

### New Fields

| Field | Default | Env Var | Description |
|-------|---------|---------|-------------|
| `stall_threshold` | 100,000 | `XDNA_EMU_STALL_THRESHOLD` | Cycles of zero progress before timeout |

### Changed Defaults

| Field | Old Default | New Default | Rationale |
|-------|-------------|-------------|-----------|
| `max_cycles` | 500,000 | 10,000,000 | Now a safety cap, not the primary timeout |

### Behavior Matrix

| `stall_threshold` | `max_cycles` | Behavior |
|-------------------|--------------|----------|
| 100,000 (default) | 10,000,000 (default) | Stall detection primary, hard cap as safety net |
| 0 | 10,000,000 | Stall detection disabled, pure cycle limit |
| 100,000 | 0 | Stall detection only, no hard cap |
| 0 | 0 | Run forever (not recommended) |

## Implementation

### Coordinator (`src/interpreter/engine/coordinator.rs`)

The coordinator already tracks `no_progress_cycles` and `last_dma_bytes`.
Replace this with the new stall detection:

```
struct StallDetector {
    last_dma_bytes: u64,
    last_lock_releases: u64,
    stall_cycles: u64,
    threshold: u64,
}
```

Each cycle, after `step()`:

```
let dma_bytes = device.array.total_dma_bytes_transferred();
let lock_releases = device.array.total_lock_releases();

if dma_bytes > self.last_dma_bytes || lock_releases > self.last_lock_releases {
    self.stall_cycles = 0;
} else {
    self.stall_cycles += 1;
}

self.last_dma_bytes = dma_bytes;
self.last_lock_releases = lock_releases;
```

Add `EngineStatus::Stalled` variant to distinguish stall-timeout from normal
halt. The coordinator returns this when `stall_cycles >= threshold`.

### Lock Release Counter

Add a `lock_releases: u64` counter to the tile state (or array-level
aggregate). Increment it in the lock release path:
- Core `rel` instruction execution (`src/interpreter/execute/control.rs`)
- DMA lock release after BD completion (`src/device/dma/engine/stepping.rs`)

Both core and DMA lock releases count as progress -- a DMA releasing a lock
after BD completion is just as meaningful as a core doing it.

### FFI Execution (`crates/xdna-emu-ffi/src/execution.rs`)

The main loop already checks `cycles >= max_cycles`. Add a check for
`engine.status() == EngineStatus::Stalled`. Log message distinguishes the
two cases:

- `"Stall detected: no progress for {threshold} cycles (at cycle {n})"`
- `"Absolute cycle limit reached: {max_cycles} cycles"`

### Test Runner (`src/testing/xclbin_suite.rs`)

Same pattern. The existing stall detection at line ~1170 (50,000 cycle
threshold with pending syncs) can be replaced by the coordinator's stall
detector, which is more principled.

### Config (`src/config.rs`)

Add `stall_threshold: Option<u64>` field. Parse from config file and
`XDNA_EMU_STALL_THRESHOLD` env var. Default: 100,000. Change `max_cycles`
default from 500,000 to 10,000,000.

## Stall Threshold Justification

The threshold must exceed the longest legitimate gap between progress events.

From our investigation:
- A 2048-element scalar loop at ~15 cycles/element = ~30,000 cycles between
  lock releases. During this time, DMA bytes may also be flowing (double
  buffering), but in the worst case (single buffering, ObjectFifo depth 1)
  only lock releases signal progress.
- 100,000 cycles provides ~3x margin over the worst observed case.
- At ~1 cycle/us on the emulator (debug build), 100K cycles = ~0.1 seconds
  wall time. True deadlocks are detected quickly.

## Testing

- Verify `dma_task_large_linear` (Peano) completes successfully with new defaults
- Verify `objectfifo_repeat/init_values_repeat` (Peano) completes successfully
- Verify existing deadlock tests still trigger timeout (stall detector fires)
- Unit test: `StallDetector` with synthetic progress sequences
- Regression: full bridge test suite, full ISA test suite

## Migration

This is backward-compatible. Existing `max_cycles` config and
`XDNA_EMU_MAX_CYCLES` env var continue to work as a hard cap. The new stall
detection layer sits underneath and fires first in most cases. Users who have
customized `max_cycles` to a low value for fast deadlock detection can set
`stall_threshold` to the same value and raise `max_cycles`.
