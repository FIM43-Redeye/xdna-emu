# DMA Unified FSM Design

**Date**: 2026-02-28
**Status**: Approved
**Problem**: DMA engine has three parallel state machines (ChannelState, TransferState,
TransferPhase) that can desync, causing lock releases to be silently dropped.

## Root Cause

The timing FSM counts words with a free-running countdown timer. The Transfer
struct counts actual bytes moved. The engine glue code must keep both in sync
across every code path (stall, success, FoT, error, padding). When they
disagree, `complete_transfer()` finds the Transfer in `Active` state instead of
`ReleasingLock` and skips the lock release silently. This is the root cause of
the fuzzer "all zeros" bug.

## Solution: Single Unified FSM Per Channel

Replace three state machines with one. The FSM variant IS the state -- no
coordination needed, no desync possible.

### ChannelFsm Enum

```rust
enum ChannelFsm {
    Idle,
    BdSetup { cycles_remaining: u16, transfer: Box<Transfer> },
    AcquiringLock { lock_id: u8, cycles_remaining: u16, transfer: Box<Transfer> },
    MemoryLatency { cycles_remaining: u16, transfer: Box<Transfer> },
    Transferring { transfer: Box<Transfer> },
    ReleasingLock { lock_id: u8, cycles_remaining: u16, completion: CompletionInfo },
    BdChaining { cycles_remaining: u16, next_bd: u8 },
    Paused { saved: Box<ChannelFsm> },
    Error,
}
```

Key design decisions:
- **Transfer is boxed** (~200 bytes). Moving between variants is a pointer swap.
- **Transferring has no countdown timer.** It checks `transfer.remaining_bytes()`
  to know when data movement is complete. This eliminates the desync by
  construction.
- **ReleasingLock carries CompletionInfo** (next_bd, bd_index, stats snapshot)
  instead of the full Transfer, because data movement is finished.
- **BdChaining is its own state** with a cycle count, not deferred to the next
  step() call via a "Complete" state.

### ChannelContext Struct

All per-channel state in one place. Replaces 11 parallel Vec<T> arrays on
DmaEngine.

```rust
struct ChannelContext {
    fsm: ChannelFsm,
    current_bd: Option<u8>,
    chain_start_bd: Option<u8>,
    queued_bd: Option<u8>,
    task_queue: VecDeque<TaskQueueEntry>,
    task_config: ChannelTaskConfig,
    task_queue_overflow: bool,
    error_bd_unavailable: bool,
    repeat_count: u32,
    stats: ChannelStats,
}
```

DmaEngine shrinks to:

```rust
struct DmaEngine {
    channels: Vec<ChannelContext>,
    bds: Vec<BufferDescriptor>,
    stream_out: VecDeque<StreamData>,
    stream_in: VecDeque<StreamData>,
    task_tokens: VecDeque<TaskCompleteToken>,
    timing_config: DmaTimingConfig,
    trace_events: Vec<(u64, EventType)>,
    current_cycle: u64,
    col: u8,
    row: u8,
    tile_type: TileType,
    num_locks: u8,
}
```

### Transfer Simplification

Transfer loses its state field and all state transition methods. It becomes a
pure data carrier:

- **Removed**: `state: TransferState`, `lock_acquired()`, `lock_released()`,
  `is_complete()`, `is_active()`, `needs_processing()`
- **Renamed**: `data_transferred()` -> `advance(bytes)` (updates counters only,
  no state transition logic)
- **Preserved**: `remaining_bytes()`, `current_address()`, `progress()`,
  `next_output_action()`, `generate_packet_header()`, `advance()`, `tick()`

The `advance()` method updates `bytes_transferred` and drives the address
generator and padding state. It does NOT decide whether the transfer is
"done" -- the FSM checks `remaining_bytes() == 0` after calling `advance()`.

### FSM Transition Logic

Each cycle, one match arm executes per channel:

- **BdSetup**: decrement cycles. When 0, go to AcquiringLock (if lock) or
  MemoryLatency (if no lock).
- **AcquiringLock**: try to acquire lock via snapshot. If available, decrement
  latency cycles. When 0, go to MemoryLatency.
- **MemoryLatency**: decrement cycles. When 0, go to Transferring.
- **Transferring**: move one word of data. If S2MM stall, stay put. If done
  (remaining == 0 or FoT TLAST), go to ReleasingLock (if lock) or begin
  completion (chaining/repeat/idle).
- **ReleasingLock**: decrement cycles. When 0, execute lock release, then begin
  completion.
- **BdChaining**: decrement cycles. When 0, load next BD, create Transfer, go
  to BdSetup.

Completion logic checks: next_bd chain? repeat_count? task_queue? Then
transitions to BdChaining or Idle accordingly.

### Public API Surface

External API preserved. `ChannelState` derived from FSM:

```rust
pub enum ChannelState {
    Idle,
    Active,   // all processing states
    Paused,
    Error,
}
```

`WaitingForLock(u8)` and `Complete` dropped -- external code treats both as
variants of "not idle / not terminal" which maps to `Active`.

`get_transfer()` reaches into FSM variants that carry a Transfer.

### Observability

- Every FSM transition logs at info level with before/after state.
- `ChannelFsm` implements Display with progress detail (bytes, addresses, lock IDs).
- Trace events fire from FSM transitions, not scattered through engine code.
- `channel_debug()` method for one-call diagnostic dumps.

### Test Strategy

1. **Existing 122 tests**: migrate mechanically. State assertions change from
   three-way checks to single FSM variant matches.
2. **New structural tests**: assert lock release guarantee (Transferring with
   release_lock must pass through ReleasingLock).
3. **Fuzzer validation**: re-run fuzzer after refactor. If the all-zeros bug
   was caused by timing/transfer desync, it should be fixed by construction.

### Files Changed

- `src/device/dma/engine.rs` -- major rewrite (DmaEngine, step logic)
- `src/device/dma/transfer.rs` -- simplify (remove TransferState, state methods)
- `src/device/dma/timing.rs` -- delete or reduce to just DmaTimingConfig
- `src/device/dma/mod.rs` -- update exports
- `src/device/mod.rs` -- update re-exports
- `src/device/array.rs` -- adapt to new API (minimal)
- `src/interpreter/engine/coordinator.rs` -- adapt ChannelState matches
- `src/testing/quiescence.rs` -- adapt ChannelState matches
