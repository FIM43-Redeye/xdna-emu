# DMA Unified FSM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the triple-state-machine DMA engine (ChannelState + TransferState + TransferPhase) with a single unified FSM per channel, eliminating desync bugs.

**Architecture:** One `ChannelFsm` enum per channel carries all state. One `ChannelContext` struct replaces 11 parallel Vec<T> arrays. Transfer becomes a pure data carrier. DmaTimingConfig preserved; ChannelTimingState/TransferPhase deleted.

**Tech Stack:** Rust, existing DMA subsystem in `src/device/dma/`

**Design doc:** `docs/plans/2026-02-28-dma-unified-fsm-design.md`

---

### Task 1: Define New Types

**Files:**
- Create: `src/device/dma/channel.rs` (new file for ChannelFsm + ChannelContext)
- Modify: `src/device/dma/mod.rs` (add module declaration)

Define the core new types that will replace the old state machines. These
compile alongside the old code -- nothing breaks yet.

**Step 1: Create channel.rs with the ChannelFsm enum**

```rust
use std::collections::VecDeque;
use std::fmt;

use super::engine::{ChannelStats, ChannelTaskConfig, TaskQueueEntry, MAX_TASK_QUEUE_DEPTH};
use super::transfer::Transfer;

/// Information carried from a completed transfer into the lock release phase.
/// Extracted from Transfer so the Transfer can be dropped once data movement
/// is done.
#[derive(Debug, Clone)]
pub struct CompletionInfo {
    pub bd_index: u8,
    pub next_bd: Option<u8>,
    pub cycles_elapsed: u64,
    pub channel: u8,
}

/// Unified per-channel DMA state machine.
///
/// Each variant represents one phase of the DMA channel lifecycle. The
/// Transfer (when present) is boxed because it's ~200 bytes -- moving
/// between variants is a pointer swap.
///
/// The key design property: `Transferring` has NO countdown timer. It
/// checks `transfer.remaining_bytes() == 0` to know when data movement is
/// complete. This eliminates the desync that existed between the old
/// ChannelTimingState word counter and the Transfer byte counter.
#[derive(Debug)]
pub enum ChannelFsm {
    /// No active transfer.
    Idle,

    /// Loading BD configuration.
    /// Latency: DmaTimingConfig::bd_setup_cycles (default 4).
    BdSetup {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// Waiting to acquire lock before data movement.
    /// Stalls if lock unavailable. Counts down lock_acquire_cycles once
    /// available.
    AcquiringLock {
        lock_id: u8,
        cycles_remaining: u16,
        acquired: bool,
        transfer: Box<Transfer>,
    },

    /// Memory pipeline warmup.
    /// Latency: DmaTimingConfig::memory_latency_cycles (default 5).
    MemoryLatency {
        cycles_remaining: u16,
        transfer: Box<Transfer>,
    },

    /// Actively moving data word by word.
    /// Exits when transfer.remaining_bytes() == 0 or FoT TLAST received.
    /// S2MM stalls transparently (stays in this state, no advancement).
    Transferring {
        transfer: Box<Transfer>,
    },

    /// Releasing lock after all data moved.
    /// Latency: DmaTimingConfig::lock_release_cycles (default 1).
    ReleasingLock {
        lock_id: u8,
        release_value: i8,
        cycles_remaining: u16,
        completion: CompletionInfo,
    },

    /// Transitioning between chained BDs.
    /// Latency: DmaTimingConfig::bd_chain_cycles (default 2).
    BdChaining {
        cycles_remaining: u16,
        next_bd: u8,
    },

    /// Channel paused by host. Resumes to saved state.
    Paused {
        saved: Box<ChannelFsm>,
    },

    /// Unrecoverable error.
    Error,
}

impl Default for ChannelFsm {
    fn default() -> Self {
        ChannelFsm::Idle
    }
}

impl ChannelFsm {
    /// Short human-readable phase name for logging.
    pub fn phase_name(&self) -> &'static str {
        match self {
            ChannelFsm::Idle => "Idle",
            ChannelFsm::BdSetup { .. } => "BdSetup",
            ChannelFsm::AcquiringLock { .. } => "AcquiringLock",
            ChannelFsm::MemoryLatency { .. } => "MemoryLatency",
            ChannelFsm::Transferring { .. } => "Transferring",
            ChannelFsm::ReleasingLock { .. } => "ReleasingLock",
            ChannelFsm::BdChaining { .. } => "BdChaining",
            ChannelFsm::Paused { .. } => "Paused",
            ChannelFsm::Error => "Error",
        }
    }

    /// Whether this channel is doing work (not idle, not terminal).
    pub fn is_active(&self) -> bool {
        !matches!(self, ChannelFsm::Idle | ChannelFsm::Error | ChannelFsm::Paused { .. })
    }

    /// Access the in-flight Transfer, if the FSM is in a phase that has one.
    pub fn transfer(&self) -> Option<&Transfer> {
        match self {
            ChannelFsm::BdSetup { transfer, .. }
            | ChannelFsm::AcquiringLock { transfer, .. }
            | ChannelFsm::MemoryLatency { transfer, .. }
            | ChannelFsm::Transferring { transfer } => Some(transfer),
            _ => None,
        }
    }

    /// Mutable access to the in-flight Transfer.
    pub fn transfer_mut(&mut self) -> Option<&mut Transfer> {
        match self {
            ChannelFsm::BdSetup { transfer, .. }
            | ChannelFsm::AcquiringLock { transfer, .. }
            | ChannelFsm::MemoryLatency { transfer, .. }
            | ChannelFsm::Transferring { transfer } => Some(transfer),
            _ => None,
        }
    }
}

impl fmt::Display for ChannelFsm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChannelFsm::Idle => write!(f, "Idle"),
            ChannelFsm::BdSetup { cycles_remaining, .. } =>
                write!(f, "BdSetup(cycles={})", cycles_remaining),
            ChannelFsm::AcquiringLock { lock_id, acquired, .. } =>
                write!(f, "AcquiringLock(lock={}, acquired={})", lock_id, acquired),
            ChannelFsm::MemoryLatency { cycles_remaining, .. } =>
                write!(f, "MemoryLatency(cycles={})", cycles_remaining),
            ChannelFsm::Transferring { transfer } =>
                write!(f, "Transferring({}/{} bytes, addr=0x{:x})",
                    transfer.bytes_transferred, transfer.total_bytes,
                    transfer.current_address()),
            ChannelFsm::ReleasingLock { lock_id, cycles_remaining, .. } =>
                write!(f, "ReleasingLock(lock={}, cycles={})", lock_id, cycles_remaining),
            ChannelFsm::BdChaining { cycles_remaining, next_bd } =>
                write!(f, "BdChaining(next_bd={}, cycles={})", next_bd, cycles_remaining),
            ChannelFsm::Paused { saved } =>
                write!(f, "Paused(was={})", saved.phase_name()),
            ChannelFsm::Error => write!(f, "Error"),
        }
    }
}

/// All per-channel state in one struct.
///
/// Replaces the 11 parallel Vec<T> arrays that DmaEngine previously used
/// for per-channel state.
pub struct ChannelContext {
    /// The unified state machine.
    pub fsm: ChannelFsm,

    /// Which channel index this is (0-based).
    pub index: u8,

    /// Currently active BD index.
    pub current_bd: Option<u8>,

    /// First BD in a chain (for repeat restart).
    pub chain_start_bd: Option<u8>,

    /// Next BD to load after current transfer completes (set by chaining).
    pub queued_bd: Option<u8>,

    /// Repeat count for current task.
    pub repeat_count: u32,

    /// Task queue (8-deep FIFO per AM025).
    pub task_queue: VecDeque<TaskQueueEntry>,

    /// Per-task configuration (token issue, FoT mode, compression).
    pub task_config: ChannelTaskConfig,

    /// Task queue overflow flag.
    pub task_queue_overflow: bool,

    /// BD unavailable error flag (out-of-order mode).
    pub error_bd_unavailable: bool,

    /// Performance counters.
    pub stats: ChannelStats,
}

impl ChannelContext {
    pub fn new(index: u8) -> Self {
        Self {
            fsm: ChannelFsm::Idle,
            index,
            current_bd: None,
            chain_start_bd: None,
            queued_bd: None,
            repeat_count: 0,
            task_queue: VecDeque::with_capacity(MAX_TASK_QUEUE_DEPTH),
            task_config: ChannelTaskConfig::default(),
            task_queue_overflow: false,
            error_bd_unavailable: false,
            stats: ChannelStats::default(),
        }
    }

    /// Derive the simplified public ChannelState from the FSM.
    pub fn state(&self) -> super::ChannelState {
        match &self.fsm {
            ChannelFsm::Idle => super::ChannelState::Idle,
            ChannelFsm::Paused { .. } => super::ChannelState::Paused,
            ChannelFsm::Error => super::ChannelState::Error,
            _ => super::ChannelState::Active,
        }
    }

    /// Whether this channel has active work.
    pub fn is_active(&self) -> bool {
        self.fsm.is_active()
    }

    /// Whether this channel has any pending work (active or queued).
    pub fn has_pending_work(&self) -> bool {
        self.fsm.is_active() || self.queued_bd.is_some() || !self.task_queue.is_empty()
    }

    /// Access the in-flight Transfer, if any.
    pub fn transfer(&self) -> Option<&Transfer> {
        self.fsm.transfer()
    }

    /// One-call debug dump.
    pub fn debug_string(&self, col: u8, row: u8) -> String {
        format!("DMA({},{}) ch{}: fsm={}, bd={:?}, repeat={}, queue={}",
            col, row, self.index, self.fsm, self.current_bd,
            self.repeat_count, self.task_queue.len())
    }

    /// Reset to initial state.
    pub fn reset(&mut self) {
        self.fsm = ChannelFsm::Idle;
        self.current_bd = None;
        self.chain_start_bd = None;
        self.queued_bd = None;
        self.repeat_count = 0;
        self.task_queue.clear();
        self.task_config = ChannelTaskConfig::default();
        self.task_queue_overflow = false;
        self.error_bd_unavailable = false;
        self.stats = ChannelStats::default();
    }
}
```

**Step 2: Add module declaration to mod.rs**

In `src/device/dma/mod.rs`, add `pub mod channel;` and add the new types
to the public exports.

**Step 3: Verify it compiles**

Run: `cargo check --lib 2>&1 | head -5`
Expected: compiles (new types exist alongside old, nothing references them yet)

**Step 4: Commit**

```
git add src/device/dma/channel.rs src/device/dma/mod.rs
git commit -m "feat(dma): define ChannelFsm and ChannelContext types"
```

---

### Task 2: Simplify Transfer

**Files:**
- Modify: `src/device/dma/transfer.rs`

Strip `TransferState` from Transfer. The FSM variant IS the state now.

**Step 1: Remove TransferState from Transfer struct**

- Delete the `state: TransferState` field from `Transfer` (line ~454)
- Remove `TransferState` assignments from all constructors (`new()`,
  `new_mem_copy()`, `new_host_to_tile()`, `new_tile_to_host()`)
- Keep the `TransferState` enum definition for now (old engine still
  references it)

**Step 2: Rename data_transferred() to advance()**

- Rename `pub fn data_transferred(&mut self, bytes: u64)` to
  `pub fn advance(&mut self, bytes: u64)`
- Remove the state transition logic at the end (the part that checks
  `bytes_transferred >= total_bytes` and sets `state = ReleasingLock` or
  `Complete`). The advance method should ONLY update counters:
  - Increment `bytes_transferred`
  - Advance address gen / padding state
  - Do NOT transition state

**Step 3: Remove state-dependent methods**

Remove these methods from Transfer (the FSM replaces them):
- `is_complete()`, `is_active()`, `is_waiting_for_lock()`, `has_error()`
- `needs_processing()`
- `lock_acquired()`, `lock_released()`

Keep all data query methods:
- `remaining_bytes()`, `current_address()`, `progress()`
- `next_output_action()`, `generate_packet_header()`, `needs_packet_header()`
- `mark_packet_header_sent()`, `tick()`, `set_error()`

**Step 4: Make bytes_transferred and total_bytes pub**

Change these from private to `pub` so the FSM Display impl can read them
without getter methods. The Transfer is owned by the FSM, so encapsulation
is at the FSM level, not the Transfer level.

**Step 5: Verify it compiles**

Run: `cargo check --lib 2>&1 | head -20`
Expected: compile errors in engine.rs (references to removed methods).
These are expected -- engine.rs is rewritten in Task 3.

**Step 6: Commit (allow compile errors)**

```
git add src/device/dma/transfer.rs
git commit -m "refactor(dma): strip TransferState from Transfer, add advance()"
```

---

### Task 3: Rewrite DmaEngine Core

This is the largest task. Rewrite the DmaEngine struct and its core stepping
logic to use ChannelContext and ChannelFsm.

**Files:**
- Modify: `src/device/dma/engine.rs` (major rewrite)

**Step 1: Rewrite DmaEngine struct definition**

Replace the 11 parallel Vec<T> fields with `channels: Vec<ChannelContext>`.
Keep shared state (stream buffers, BDs, trace, timing config, tile identity).

Old fields to REMOVE from DmaEngine:
- `transfers: Vec<Option<Transfer>>`
- `channel_states: Vec<ChannelState>`
- `channel_stats: Vec<ChannelStats>`
- `queued_bds: Vec<Option<u8>>`
- `repeat_counts: Vec<u8>`
- `current_bds: Vec<Option<u8>>`
- `chain_start_bds: Vec<Option<u8>>`
- `timing_states: Vec<ChannelTimingState>`
- `channel_task_configs: Vec<ChannelTaskConfig>`
- `task_queues: Vec<VecDeque<TaskQueueEntry>>`
- `task_queue_overflow: Vec<bool>`
- `error_bd_unavailable: Vec<bool>`

New field to ADD:
- `channels: Vec<ChannelContext>`

**Step 2: Rewrite constructors**

`new()` creates `channels` by iterating `0..num_channels` and calling
`ChannelContext::new(i)`. All parallel vec initialization goes away.

`new_compute_tile()`, `new_mem_tile()`, `new_shim_tile()` stay as thin
wrappers calling `new()` with the right parameters.

**Step 3: Rewrite step() -- the outer loop**

```rust
pub fn step(&mut self, tile: &mut Tile, neighbors: &mut NeighborLocks<'_>,
            host_memory: &mut HostMemory) -> DmaResult {
    let mut any_active = false;
    let mut any_waiting = false;

    for ch_idx in 0..self.channels.len() {
        let phase_before = self.channels[ch_idx].fsm.phase_name();

        match &self.channels[ch_idx].fsm {
            ChannelFsm::Idle => {
                // Check for queued BD from chaining or task queue
                if let Some(next_bd) = self.channels[ch_idx].queued_bd.take() {
                    self.start_bd(ch_idx, next_bd);
                    any_active = true;
                }
            }
            ChannelFsm::Paused { .. } | ChannelFsm::Error => {}
            _ => {
                // Active channel -- run one FSM cycle
                self.step_channel(ch_idx, tile, neighbors, host_memory);
                any_active = true;
            }
        }

        // Log transitions
        let phase_after = self.channels[ch_idx].fsm.phase_name();
        if phase_before != phase_after {
            log::info!("DMA({},{}) ch{}: {} -> {}",
                self.col, self.row, ch_idx, phase_before, phase_after);
        }
    }

    if any_active { DmaResult::InProgress }
    else if any_waiting { DmaResult::WaitingForLock(0) }
    else { DmaResult::Complete }
}
```

**Step 4: Write step_channel() -- the unified FSM**

This is the single match statement that replaces `step_channel()` +
`step_channel_timed()` + `complete_transfer()` + `finish_complete_transfer()`.

See the design doc Section 2 for the full pseudocode. Key points:
- Each match arm reads its state, does ONE cycle of work, optionally
  transitions to a new state
- `Transferring` arm calls `self.do_transfer()` (existing method, mostly
  unchanged), then `transfer.advance()`, then checks
  `transfer.remaining_bytes() == 0`
- Lock acquire uses `self.try_acquire_lock()` (existing method, adapted)
- Lock release uses `self.release_lock()` (extracted from old
  `complete_transfer()`)
- Completion logic (chaining, repeat, task queue) is in a helper
  `self.after_transfer_done()`

The `do_transfer()`, `transfer_mm2s()`, `transfer_s2mm()`, and all the
compression/decompression methods stay largely unchanged -- they operate on
a `&mut Transfer` and tile/host memory, returning `TransferResult`. The only
change is they no longer need to know about `TransferState`.

**Step 5: Rewrite start_channel() and start_channel_with_repeat()**

These create a `Transfer` from a `BdConfig`, box it, and set the channel's
FSM to `BdSetup { cycles_remaining, transfer }`.

```rust
pub fn start_channel(&mut self, channel: u8, bd_index: u8) -> Result<(), DmaError> {
    self.start_channel_with_repeat(channel, bd_index, 0)
}

pub fn start_channel_with_repeat(&mut self, channel: u8, bd_index: u8,
                                  repeat_count: u32) -> Result<(), DmaError> {
    let ch_idx = channel as usize;
    if ch_idx >= self.channels.len() {
        return Err(DmaError::InvalidChannel(channel));
    }
    if self.channels[ch_idx].is_active() {
        return Err(DmaError::ChannelBusy(channel));
    }

    let bd = self.bd_configs.get(bd_index as usize)
        .ok_or(DmaError::InvalidBd(bd_index))?
        .clone();

    let transfer = Transfer::new(
        &bd, bd_index, channel, /* direction, tile coords, tile_type */
    );

    self.channels[ch_idx].current_bd = Some(bd_index);
    self.channels[ch_idx].chain_start_bd = Some(bd_index);
    self.channels[ch_idx].repeat_count = repeat_count;

    self.channels[ch_idx].fsm = ChannelFsm::BdSetup {
        cycles_remaining: self.timing_config.bd_setup_cycles as u16,
        transfer: Box::new(transfer),
    };

    self.trace(EventType::DmaStartTask { channel });
    Ok(())
}
```

**Step 6: Rewrite stop/pause/resume**

```rust
pub fn stop_channel(&mut self, channel: u8) {
    let ch = &mut self.channels[channel as usize];
    ch.fsm = ChannelFsm::Idle;
    ch.queued_bd = None;
}

pub fn pause_channel(&mut self, channel: u8) {
    let ch = &mut self.channels[channel as usize];
    if ch.fsm.is_active() {
        let saved = std::mem::take(&mut ch.fsm);  // takes Idle, puts in saved
        ch.fsm = ChannelFsm::Paused { saved: Box::new(saved) };
    }
}

pub fn resume_channel(&mut self, channel: u8) {
    let ch = &mut self.channels[channel as usize];
    if let ChannelFsm::Paused { saved } = std::mem::take(&mut ch.fsm) {
        ch.fsm = *saved;
    }
}
```

**Step 7: Rewrite public query methods**

These become thin wrappers around ChannelContext:

```rust
pub fn channel_state(&self, channel: u8) -> ChannelState {
    self.channels[channel as usize].state()
}
pub fn channel_active(&self, channel: u8) -> bool {
    self.channels[channel as usize].is_active()
}
pub fn any_channel_active(&self) -> bool {
    self.channels.iter().any(|ch| ch.is_active())
}
pub fn channel_stats(&self, channel: u8) -> Option<&ChannelStats> {
    self.channels.get(channel as usize).map(|ch| &ch.stats)
}
pub fn get_transfer(&self, channel: u8) -> Option<&Transfer> {
    self.channels.get(channel as usize).and_then(|ch| ch.transfer())
}
```

**Step 8: Rewrite task queue methods**

These move from `self.task_queues[ch_idx]` to
`self.channels[ch_idx].task_queue`. Same logic, different location.

**Step 9: Rewrite reset()**

```rust
pub fn reset(&mut self) {
    for ch in &mut self.channels {
        ch.reset();
    }
    self.stream_out.clear();
    self.stream_in.clear();
    self.task_tokens.clear();
    self.trace_events.clear();
}
```

**Step 10: Verify compilation**

Run: `cargo check --lib 2>&1 | head -30`
Fix any remaining references to old field names.

**Step 11: Commit**

```
git commit -m "refactor(dma): rewrite DmaEngine with unified ChannelFsm"
```

---

### Task 4: Simplify timing.rs

**Files:**
- Modify: `src/device/dma/timing.rs`

**Step 1: Remove ChannelTimingState and TransferPhase**

These are fully replaced by `ChannelFsm`. Delete:
- `TransferPhase` enum (lines 125-144)
- `ChannelTimingState` struct (lines 103-122)
- `impl ChannelTimingState` (lines 146-293)

Keep:
- `DmaTimingConfig` struct and its impl (lines 32-100) -- the timing
  CONSTANTS are still needed. The FSM reads `timing_config.bd_setup_cycles`,
  etc.
- `ChannelArbiter` struct if used (lines 296-386) -- check if anything
  still references it after the engine rewrite. If not, remove.

**Step 2: Delete timing tests that tested the old FSM**

Tests `test_channel_timing_phases`, `test_channel_timing_with_lock_stall`,
`test_bd_chain_cycles` tested TransferPhase transitions. These are replaced
by ChannelFsm tests.

Keep `test_timing_config_default`, `test_transfer_cycles_simple`,
`test_transfer_cycles_with_locks` -- these test DmaTimingConfig which still
exists.

**Step 3: Verify and commit**

```
git commit -m "refactor(dma): remove ChannelTimingState and TransferPhase"
```

---

### Task 5: Update Exports and External Callers

**Files:**
- Modify: `src/device/dma/mod.rs` (update pub use)
- Modify: `src/device/mod.rs:70-76` (update re-exports)
- Modify: `src/device/array.rs:415,432` (ChannelState match arms)
- Modify: `src/interpreter/engine/coordinator.rs:13,825-826` (ChannelState)
- Modify: `src/testing/quiescence.rs:203,211-218` (ChannelState match)

**Step 1: Update mod.rs exports**

Remove from exports: `TransferState`, `ChannelTimingState`, `TransferPhase`
Add to exports: `ChannelFsm`, `ChannelContext`, `CompletionInfo`

**Step 2: Update device/mod.rs re-exports**

Same removals and additions at the crate level.

**Step 3: Update ChannelState in external callers**

`ChannelState` loses `WaitingForLock(u8)` and `Complete` variants.
External code needs updating:

- `coordinator.rs:825-826`: Change
  `matches!(state, ChannelState::Complete | ChannelState::Idle)` to
  `matches!(state, ChannelState::Idle)`. The coordinator only needs to know
  if the channel is idle to clear `tile.dma_channels[ch].running`.

- `array.rs:432`: Change `matches!(..., ChannelState::WaitingForLock(_))`
  to a check on the FSM directly if needed, or to
  `matches!(..., ChannelState::Active)` since lock wait is now a subset of
  Active.

- `quiescence.rs:211-218`: Simplify the match to Idle/Active/Paused/Error.
  Remove the `WaitingForLock` and `Complete` arms.

**Step 4: Verify and commit**

Run: `cargo check --lib`

```
git commit -m "refactor(dma): update exports and external ChannelState callers"
```

---

### Task 6: Migrate Tests

**Files:**
- Modify: `src/device/dma/engine.rs` (test module at line 2741+)
- Modify: `src/device/dma/transfer.rs` (test module)
- Modify: `src/device/dma/channel.rs` (add new tests)

**Step 1: Fix engine.rs tests**

The 41 engine tests mostly use the public API (`start_channel`, `step`,
`channel_state`, `channel_active`). Many will compile without changes.

Tests that assert `channel_states[ch]` or `timing_states[ch]` or
`transfers[ch].unwrap().state` need updating to use:
- `channels[ch].fsm` matches
- `channels[ch].state()`
- `channels[ch].transfer()`

Go through each test, fix assertions. Key patterns:

Old: `assert!(matches!(engine.channel_states[0], ChannelState::Active));`
New: `assert!(engine.channels[0].is_active());`

Old: `assert_eq!(engine.timing_states[0].phase, TransferPhase::DataTransfer);`
New: `assert!(matches!(engine.channels[0].fsm, ChannelFsm::Transferring { .. }));`

Old: `assert_eq!(engine.transfers[0].as_ref().unwrap().state, TransferState::Active);`
New: `assert!(matches!(engine.channels[0].fsm, ChannelFsm::Transferring { .. }));`

**Step 2: Fix transfer.rs tests**

Tests that construct a Transfer and check `transfer.state` need updating.
Since Transfer no longer has a state field:
- Tests that check `TransferState::Active` after construction: remove the
  assertion (the Transfer is always "active" as a data carrier)
- Tests that call `transfer.data_transferred()`: change to
  `transfer.advance()`
- Tests that check `transfer.is_complete()`: check
  `transfer.remaining_bytes() == 0` instead

**Step 3: Add new ChannelFsm tests in channel.rs**

Write tests for the FSM itself:
- Test that `phase_name()` returns correct strings
- Test Display formatting
- Test `transfer()` accessor returns Some/None for correct variants
- Test `is_active()` for each variant

**Step 4: Run full test suite**

Run: `cargo test --lib 2>&1 | tail -20`
Expected: all tests pass

**Step 5: Commit**

```
git commit -m "test(dma): migrate tests to unified ChannelFsm"
```

---

### Task 7: Cleanup and Delete Dead Code

**Files:**
- Modify: `src/device/dma/transfer.rs` (remove TransferState enum)
- Modify: `src/device/dma/engine.rs` (remove ChannelState::WaitingForLock, Complete)

**Step 1: Delete TransferState enum**

Now that nothing references it, remove the `TransferState` enum entirely
from transfer.rs.

**Step 2: Simplify ChannelState**

Final form:
```rust
pub enum ChannelState {
    #[default]
    Idle,
    Active,
    Paused,
    Error,
}
```

**Step 3: Remove any remaining dead code**

Search for unused imports, functions, or types related to the old state
machines. The compiler will warn about these.

Run: `cargo check --lib 2>&1 | grep warning`

**Step 4: Final test run**

Run: `cargo test --lib`
Expected: all pass, no warnings

**Step 5: Commit**

```
git commit -m "cleanup(dma): remove TransferState and dead timing code"
```

---

### Task 8: Integration Validation

**Step 1: Run cargo test (all tests including doc tests)**

Run: `./scripts/run-tests.sh 2>&1 | tail -30`

Fix any integration test failures.

**Step 2: Run the fuzzer to test the all-zeros bug**

Run a single fuzz iteration to see if lock release now works:

```bash
cargo run --release -- --fuzz --seed 1 --iterations 1 2>&1 | tee /tmp/fuzz-post-refactor.log
```

Check the output for:
- Non-zero emulator output (the bug was all zeros)
- Lock release log messages firing
- FSM transitions visible in the log

**Step 3: Run npu-test suite**

From a terminal (not sandbox):
```bash
cargo run --bin npu-test 2>&1 | tee /tmp/npu-test-post-refactor.log
```

Compare pass/fail counts with pre-refactor baseline (from memory:
Peano 23/65, Chess 22/65).

**Step 4: Commit any fixes and final state**

```
git commit -m "refactor(dma): unified FSM complete, all tests passing"
```

---

## Execution Notes

### Ordering constraints

Tasks 1-2 can be done independently (new types, Transfer simplification).
Task 3 depends on both. Tasks 4-5 depend on Task 3. Task 6 depends on
Tasks 3-5. Task 7 depends on Task 6. Task 8 depends on everything.

### Risk areas

- **do_transfer() and friends**: These methods operate on `&mut Transfer`
  and tile/host memory. They should mostly work unchanged, but the borrow
  checker may complain about accessing `self.channels[ch_idx].fsm.transfer_mut()`
  while also borrowing `self.stream_out`. Extract transfer from the FSM
  temporarily if needed.

- **Borrow splitting**: The old code used `self.transfers[ch_idx]` which was
  independent of `self.stream_out`. With the unified FSM, the Transfer is
  inside `self.channels[ch_idx].fsm`. If `do_transfer` needs both `&mut transfer`
  and `&mut self.stream_out`, extract the transfer from the Box temporarily:
  ```rust
  let mut transfer = match std::mem::take(&mut ch.fsm) {
      ChannelFsm::Transferring { transfer } => transfer,
      other => { ch.fsm = other; return; }
  };
  // ... use transfer and self.stream_out ...
  ch.fsm = ChannelFsm::Transferring { transfer };
  ```

- **Packet header insertion**: Currently handled by `maybe_insert_packet_header()`
  called at the top of `step_channel_timed()`. In the new design, this should
  happen on entry to `Transferring` state (first cycle only), or be tracked
  by the Transfer's `packet_header_sent` flag.

### What NOT to change

- `addressing.rs` -- untouched, AddressGenerator is independent
- `bd.rs` -- untouched, BD parsing is independent
- `compression.rs` -- untouched, compression logic is independent
- `stream_io.rs` -- untouched, stream word format is independent
- `DmaTimingConfig` -- preserved, timing constants unchanged
