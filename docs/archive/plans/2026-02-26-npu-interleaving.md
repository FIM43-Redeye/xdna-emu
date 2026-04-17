# NPU Instruction Interleaving Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace batch NPU instruction execution with a state machine that interleaves instruction-by-instruction with full system stepping, so DMA queue backpressure works correctly without dropping tasks.

**Architecture:** `NpuExecutor` becomes a state machine with states (Ready, Executing, BlockedOnQueue, Done). Each engine cycle, the caller invokes `try_advance()` which executes at most one instruction or checks if a blocked queue has drained. The existing `execute()` batch method is preserved for backward compatibility (unit tests, FFI) by calling `try_advance()` in a loop with DMA-only stepping as fallback. The 100K-cycle drain loop and "task dropped" path are eliminated.

**Tech Stack:** Pure Rust, no new dependencies. Changes only to `src/npu/executor.rs`, `src/npu/mod.rs`, `src/testing/xclbin_suite.rs`, `src/ffi/mod.rs`.

---

## Background

### The Problem

Real NPU hardware runs the host firmware (NPU instructions) concurrently with the AIE array (cores, DMA, stream routing). When the host writes a Start_Queue register and the queue is full, the host blocks until the queue drains -- with the full array running.

Our emulator batches ALL NPU instructions before starting the engine loop. The `check_dma_trigger()` method has a fallback drain loop that steps DMA only (not cores, not stream routing) for up to 100K cycles, then drops the task if the queue is still full. This causes:

1. **Shim DMA can never drain** during instruction execution because shim transfers require stream routing and potentially core participation.
2. **Tasks get dropped** when the drain loop fails, producing warnings and incorrect behavior.
3. **15 warnings** in sync_task_complete_token tests from dropped shim DMA tasks.

### The Fix

Interleave NPU instruction execution with `engine.step()` in the `run_engine()` loop. Each cycle:
1. `executor.try_advance()` -- execute one NPU instruction (or check if blocked queue drained)
2. `engine.step()` -- step full system (cores + DMA + stream routing + locks)
3. Check sync completion and TDR as before

### Key Design Decisions

- **State machine, not coroutines**: Explicit state enum, easy to test and extend
- **Preserve `execute()` batch method**: FFI and unit tests keep working unchanged
- **Never drop tasks**: `BlockedOnQueue` waits indefinitely; TDR catches real hangs
- **Future-ready**: States for host timing (`WaitHostClock`) and MaskPoll can be added later
- **No changes to InterpreterEngine**: The engine is oblivious to this change

### Files Reference

| File | Role |
|------|------|
| `src/npu/executor.rs` | NPU instruction executor (main changes) |
| `src/npu/mod.rs` | Module exports |
| `src/testing/xclbin_suite.rs` | Test runner that calls executor (line 873: batch execute, line 1244: engine loop) |
| `src/ffi/mod.rs` | FFI entry point (line 420: batch execute, line 522: engine loop) |
| `src/device/dma/engine.rs` | DMA engine (line 2265: `enqueue_task`, line 2372: `task_queue_size`) |

---

## Task 1: Add State Machine Types to NpuExecutor

**Files:**
- Modify: `src/npu/executor.rs`
- Modify: `src/npu/mod.rs`

This task adds the new types without changing any behavior. Pure additive.

**Step 1: Write tests for the new types**

Add at bottom of `src/npu/executor.rs` tests module:

```rust
#[test]
fn test_advance_result_variants() {
    // Verify all variants exist and can be matched
    let results = [
        AdvanceResult::Progressed,
        AdvanceResult::Blocked,
        AdvanceResult::Done,
        AdvanceResult::Idle,
    ];
    for r in &results {
        match r {
            AdvanceResult::Progressed => {}
            AdvanceResult::Blocked => {}
            AdvanceResult::Done => {}
            AdvanceResult::Idle => {}
        }
    }
}

#[test]
fn test_executor_initial_state_is_idle() {
    let executor = NpuExecutor::new();
    // New executors should report Idle (no instructions loaded)
    assert!(matches!(executor.state(), ExecutorState::Idle));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_advance_result_variants test_executor_initial_state_is_idle`
Expected: FAIL (types don't exist yet)

**Step 3: Add the types**

In `src/npu/executor.rs`, add after the `use` block (before `HostBuffer`):

```rust
/// Result of a single `try_advance()` step.
///
/// The caller (run_engine or FFI loop) uses this to know whether
/// execution is still in progress. The caller does not need to
/// inspect internal state -- just check the variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdvanceResult {
    /// Executed one instruction successfully. Call again next cycle.
    Progressed,
    /// Blocked waiting for a DMA queue to drain. Engine should keep
    /// stepping; the executor will retry on the next try_advance() call.
    Blocked,
    /// All instructions have been executed. Executor is finished.
    Done,
    /// No instructions loaded or executor not started.
    Idle,
}

/// Internal state of the NPU instruction executor.
///
/// Drives the state machine in `try_advance()`. Not exposed to callers
/// except through `AdvanceResult`.
#[derive(Debug, Clone)]
pub(crate) enum ExecutorState {
    /// No instruction stream loaded.
    Idle,
    /// Processing instructions. `next_index` is the index of the next
    /// instruction to execute in the loaded stream.
    Executing { next_index: usize },
    /// Blocked on a full DMA task queue. Holds the pending enqueue
    /// parameters so we can retry without re-executing the instruction.
    BlockedOnQueue {
        /// Index of the instruction that triggered the block (for logging).
        instr_index: usize,
        /// Index of the NEXT instruction after the blocked one completes.
        next_index: usize,
        /// Tile column with the full queue.
        col: u8,
        /// Tile row with the full queue.
        row: u8,
        /// Absolute channel index.
        channel: u8,
        /// BD index to enqueue.
        bd_id: u8,
        /// Repeat count for the task.
        repeat: u8,
        /// Enable_Token_Issue bit (Start_Queue bit 31). Stubbed for future use.
        enable_token: bool,
    },
    /// All instructions executed.
    Done,
}
```

Add a `state()` accessor to `NpuExecutor` impl block:

```rust
/// Get the current executor state (for testing/debugging).
pub(crate) fn state(&self) -> &ExecutorState {
    &self.state
}
```

Add the `state` field to `NpuExecutor` struct (after `warnings`):

```rust
/// Internal state machine state.
state: ExecutorState,
```

Initialize it in `NpuExecutor::new()`:

```rust
state: ExecutorState::Idle,
```

**Step 4: Export types from mod.rs**

In `src/npu/mod.rs`, change the executor re-export line:

```rust
pub use executor::{NpuExecutor, HostBuffer, AdvanceResult};
```

**Step 5: Run tests to verify they pass**

Run: `cargo test --lib test_advance_result_variants test_executor_initial_state_is_idle`
Expected: PASS

**Step 6: Run full test suite for regression**

Run: `cargo test --lib`
Expected: All existing tests still pass (no behavioral change)

**Step 7: Commit**

```
feat(npu): add state machine types for interleaved execution

Add AdvanceResult and ExecutorState enums to NpuExecutor in preparation
for interleaving NPU instruction execution with engine stepping.
No behavioral change -- types are additive only.
```

---

## Task 2: Implement `load()` and `try_advance()` Methods

**Files:**
- Modify: `src/npu/executor.rs`

This task adds the new methods alongside the existing `execute()`. Both
paths coexist -- `execute()` is not modified yet.

**Step 1: Write tests for load and try_advance**

```rust
#[test]
fn test_load_transitions_to_executing() {
    let mut executor = NpuExecutor::new();
    let stream = NpuInstructionStream::parse(&[]).unwrap_or_else(|_| {
        // Empty stream
        NpuInstructionStream::empty()
    });
    executor.load(&stream);
    // After loading an empty stream, try_advance should immediately return Done
    // (0 instructions to execute)
}

#[test]
fn test_try_advance_idle_returns_idle() {
    let mut executor = NpuExecutor::new();
    let mut device = DeviceState::new_npu1();
    let mut host_mem = HostMemory::new();
    assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Idle);
}

#[test]
fn test_try_advance_done_returns_done() {
    let mut executor = NpuExecutor::new();
    let mut device = DeviceState::new_npu1();
    let mut host_mem = HostMemory::new();

    // Load empty stream -> immediately Done
    executor.load_instructions(Vec::new());
    assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
    // Subsequent calls also return Done
    assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_load_transitions test_try_advance_idle test_try_advance_done`
Expected: FAIL (methods don't exist)

**Step 3: Add an `empty()` constructor to NpuInstructionStream (if needed)**

Check if `NpuInstructionStream` has an `empty()` or equivalent. If not, add
`load_instructions()` on the executor that takes a `Vec<NpuInstruction>` directly,
avoiding the need to construct a stream from bytes.

**Step 4: Implement `load_instructions()` and `try_advance()`**

Add to `NpuExecutor` impl:

```rust
/// Load a parsed instruction stream for interleaved execution.
///
/// Copies the instructions into the executor and transitions to
/// the Executing state. Call `try_advance()` each engine cycle
/// to process instructions one at a time.
///
/// This is the interleaved counterpart to `execute()`. Use this
/// when NPU instruction execution should be interleaved with
/// engine stepping (the test runner and future FFI path).
pub fn load(&mut self, stream: &NpuInstructionStream) {
    self.load_instructions(stream.instructions().to_vec());
}

/// Load instructions directly (for testing or internal use).
pub fn load_instructions(&mut self, instructions: Vec<NpuInstruction>) {
    self.instructions = instructions;
    if self.instructions.is_empty() {
        self.state = ExecutorState::Done;
    } else {
        self.state = ExecutorState::Executing { next_index: 0 };
    }
}

/// Try to advance execution by one step.
///
/// Called once per engine cycle. Returns the result of the step:
/// - `Progressed`: executed one instruction, call again next cycle
/// - `Blocked`: waiting for DMA queue to drain, engine should keep stepping
/// - `Done`: all instructions processed
/// - `Idle`: no instructions loaded
///
/// When blocked on a full queue, the executor holds the pending enqueue
/// parameters and retries on the next call. The caller's engine.step()
/// naturally drains the queue by stepping the full system (DMA + cores +
/// stream routing), so the queue will eventually have space.
pub fn try_advance(
    &mut self,
    device: &mut DeviceState,
    host_memory: &mut HostMemory,
) -> AdvanceResult {
    match self.state.clone() {
        ExecutorState::Idle => AdvanceResult::Idle,
        ExecutorState::Done => AdvanceResult::Done,

        ExecutorState::Executing { next_index } => {
            if next_index >= self.instructions.len() {
                self.state = ExecutorState::Done;
                return AdvanceResult::Done;
            }

            let instr = self.instructions[next_index].clone();
            if let Err(e) = self.execute_instruction(&instr, device, host_memory) {
                log::error!("NPU instruction {} execution error: {}", next_index, e);
                self.warnings.push(format!("Instruction {} error: {}", next_index, e));
            }
            self.executed_count += 1;

            // check_dma_trigger may have transitioned us to BlockedOnQueue.
            // If so, return Blocked. Otherwise advance to next instruction.
            if matches!(self.state, ExecutorState::BlockedOnQueue { .. }) {
                AdvanceResult::Blocked
            } else {
                let new_index = next_index + 1;
                if new_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    AdvanceResult::Done
                } else {
                    self.state = ExecutorState::Executing { next_index: new_index };
                    AdvanceResult::Progressed
                }
            }
        }

        ExecutorState::BlockedOnQueue {
            next_index, col, row, channel, bd_id, repeat, enable_token, ..
        } => {
            // Check if the queue has drained enough for our enqueue
            let has_space = device.array.dma_engine(col, row)
                .map_or(true, |dma| dma.task_queue_size(channel) < MAX_TASK_QUEUE_DEPTH);

            if has_space {
                // Enqueue the pending task
                if let Some(dma) = device.array.dma_engine_mut(col, row) {
                    if dma.enqueue_task(channel, bd_id, repeat, enable_token) {
                        log::info!("  DMA ch{} enqueued BD {} (queue drained)", channel, bd_id);
                    }
                }
                // Advance to next instruction
                if next_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    AdvanceResult::Done
                } else {
                    self.state = ExecutorState::Executing { next_index };
                    AdvanceResult::Progressed
                }
            } else {
                // Still blocked
                AdvanceResult::Blocked
            }
        }
    }
}
```

Add the `instructions` field to `NpuExecutor`:

```rust
/// Loaded instructions for interleaved execution.
instructions: Vec<NpuInstruction>,
```

Initialize in `new()`:

```rust
instructions: Vec::new(),
```

**Step 5: Run tests to verify they pass**

Run: `cargo test --lib test_load_transitions test_try_advance_idle test_try_advance_done`
Expected: PASS

**Step 6: Commit**

```
feat(npu): implement load() and try_advance() for interleaved execution

NpuExecutor can now process instructions one-at-a-time via try_advance(),
enabling interleaving with engine stepping. The existing execute() batch
method is unchanged.
```

---

## Task 3: Refactor `check_dma_trigger` to Signal BlockedOnQueue

**Files:**
- Modify: `src/npu/executor.rs`

This is the core behavioral change. `check_dma_trigger` stops trying to
drain queues itself and instead transitions the state machine to
`BlockedOnQueue` when a queue is full.

**Step 1: Write a test for the blocked-on-queue behavior**

```rust
#[test]
fn test_try_advance_blocks_on_full_queue() {
    use crate::device::dma::{BdConfig, MAX_TASK_QUEUE_DEPTH};

    let mut device = DeviceState::new_npu1();
    let mut host_mem = HostMemory::new();
    let mut executor = NpuExecutor::new();

    // We need to construct an NPU instruction that writes to a shim DMA
    // Task_Queue register. Use the register layout to find the correct offset.
    let reg_layout = crate::device::regdb::device_reg_layout();
    let col: u8 = 0;
    let row: u8 = 0; // shim

    // Configure BD 0 with a simple transfer so enqueue doesn't fail
    // on missing BD config.
    if let Some(dma) = device.array.dma_engine_mut(col, row) {
        let bd = BdConfig::simple_1d(0x0, 256);
        dma.configure_bd(0, bd).unwrap();
    }

    // Fill the task queue for S2MM channel 0 by enqueuing MAX_TASK_QUEUE_DEPTH tasks
    if let Some(dma) = device.array.dma_engine_mut(col, row) {
        for _ in 0..MAX_TASK_QUEUE_DEPTH {
            assert!(dma.enqueue_task(0, 0, 0, false));
        }
        assert_eq!(dma.task_queue_size(0), MAX_TASK_QUEUE_DEPTH);
    }

    // Create an NPU instruction that writes to the S2MM ch0 Task_Queue register.
    // Task_Queue offset = channel_base + 0*stride + 4
    let queue_offset = reg_layout.shim_channel_base + 4;
    let npu_addr = ((col as u32) << 25) | ((row as u32) << 20) | queue_offset;
    let task_value: u32 = 0; // BD 0, repeat 0

    executor.load_instructions(vec![
        NpuInstruction::Write32 { reg_off: npu_addr, value: task_value },
    ]);

    // First try_advance should detect queue full and return Blocked
    let result = executor.try_advance(&mut device, &mut host_mem);
    assert_eq!(result, AdvanceResult::Blocked,
        "Should block when task queue is full");

    // Drain one task from the queue by stepping DMA
    device.array.step_all_dma(&mut host_mem);

    // Now try_advance should succeed (queue has space)
    let result = executor.try_advance(&mut device, &mut host_mem);
    assert!(
        matches!(result, AdvanceResult::Progressed | AdvanceResult::Done),
        "Should progress after queue drains, got {:?}", result
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_try_advance_blocks_on_full_queue`
Expected: FAIL (check_dma_trigger still uses old drain loop, doesn't set BlockedOnQueue)

**Step 3: Refactor check_dma_trigger**

The key change: `check_dma_trigger` needs to know whether it's being called
from `try_advance` (interleaved mode) or from `execute` (batch mode). Rather
than adding a flag parameter, restructure as follows:

1. Extract the "is queue full?" check and "enqueue" logic into a helper method
   `try_enqueue_task()` that returns a `QueueResult` enum.
2. `check_dma_trigger` calls `try_enqueue_task`. If it returns `QueueFull`,
   `check_dma_trigger` transitions state to `BlockedOnQueue` (no drain loop).
3. The old `execute()` batch method calls `try_advance` in a loop. When
   `try_advance` returns `Blocked`, `execute()` falls back to stepping DMA
   (preserving backward compatibility for FFI/tests that don't have an engine).

Replace the backpressure section of `check_dma_trigger` (lines 526-575 approximately):

**Before:**
```rust
// Try to enqueue. If queue is full, apply backpressure...
use crate::device::dma::MAX_TASK_QUEUE_DEPTH;
let needs_drain = device.array.dma_engine(col, row)
    .map_or(false, |dma| dma.task_queue_size(abs_channel) >= MAX_TASK_QUEUE_DEPTH);
if needs_drain {
    // ... 100K drain loop, drop on failure ...
}
if let Some(dma) = device.array.dma_engine_mut(col, row) {
    if dma.enqueue_task(abs_channel, bd_index, repeat_count, false) { ... }
}
```

**After:**
```rust
use crate::device::dma::MAX_TASK_QUEUE_DEPTH;

let queue_full = device.array.dma_engine(col, row)
    .map_or(false, |dma| dma.task_queue_size(abs_channel) >= MAX_TASK_QUEUE_DEPTH);

if queue_full {
    // In interleaved mode: transition to BlockedOnQueue and let the
    // engine's full system stepping drain the queue naturally.
    // In batch mode (execute()): the caller handles the retry loop.
    log::debug!(
        "DMA tile({},{}) ch{} queue full, deferring BD {} enqueue",
        col, row, abs_channel, bd_index
    );
    self.state = ExecutorState::BlockedOnQueue {
        instr_index: self.executed_count,
        next_index: self.executed_count + 1, // will be updated by try_advance
        col, row,
        channel: abs_channel,
        bd_id: bd_index,
        repeat: repeat_count,
        enable_token: false, // TODO: extract from value bit 31
    };
    return;
}

if let Some(dma) = device.array.dma_engine_mut(col, row) {
    if dma.enqueue_task(abs_channel, bd_index, repeat_count, false) {
        log::info!("  DMA channel {} enqueued BD {} repeat={}", abs_channel, bd_index, repeat_count);
    } else {
        log::warn!("  DMA channel {} enqueue failed unexpectedly for BD {}", abs_channel, bd_index);
    }
}
```

**Important:** The `next_index` in BlockedOnQueue gets corrected by `try_advance()`
which knows the actual instruction index. `check_dma_trigger` sets it to
`executed_count + 1` as a rough estimate; `try_advance()` overwrites the state
with the correct `next_index` before returning `Blocked`.

Actually, a cleaner approach: `try_advance` sets the state to `Executing { next_index }`
BEFORE calling `execute_instruction`. If `check_dma_trigger` transitions to
`BlockedOnQueue`, the `next_index` field should be the instruction AFTER the
current one. Adjust `try_advance` accordingly:

In `try_advance`, the `Executing` arm becomes:
```rust
ExecutorState::Executing { next_index } => {
    // ...execute instruction...
    // After execution, check if check_dma_trigger blocked us
    if let ExecutorState::BlockedOnQueue { ref mut next_index: ni, .. } = self.state {
        // Fix up next_index to point past the current instruction
        *ni = next_index + 1;
        return AdvanceResult::Blocked;
    }
    // Normal progression...
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test --lib test_try_advance_blocks_on_full_queue`
Expected: PASS

**Step 5: Run full test suite for regression**

Run: `cargo test --lib`
Expected: All tests pass. The batch `execute()` path still uses the old
drain loop, so existing tests are unaffected.

**Step 6: Commit**

```
refactor(npu): check_dma_trigger signals BlockedOnQueue instead of drain loop

When a DMA task queue is full, check_dma_trigger now transitions the
executor to BlockedOnQueue state instead of running a 100K-cycle drain
loop. try_advance() handles the retry when the queue drains. The batch
execute() path is not yet modified.
```

---

## Task 4: Refactor `execute()` to Use `try_advance()` Internally

**Files:**
- Modify: `src/npu/executor.rs`

Make the batch `execute()` method delegate to `try_advance()` in a loop,
with a DMA-only stepping fallback when blocked. This ensures both paths
share the same instruction execution logic.

**Step 1: Write a test that exercises the batch path with a full queue**

The existing `test_sync_requires_channel_started` test exercises `execute()`
indirectly. Add a dedicated test:

```rust
#[test]
fn test_execute_batch_handles_full_queue() {
    use crate::device::dma::{BdConfig, MAX_TASK_QUEUE_DEPTH};

    let mut device = DeviceState::new_npu1();
    let mut host_mem = HostMemory::new();
    let mut executor = NpuExecutor::new();

    let reg_layout = crate::device::regdb::device_reg_layout();
    let col: u8 = 0;
    let row: u8 = 2; // compute tile (has DMA channels)

    // Configure BD 0
    if let Some(dma) = device.array.dma_engine_mut(col, row) {
        let bd = BdConfig::simple_1d(0x100, 64);
        dma.configure_bd(0, bd).unwrap();
    }

    // Fill the queue
    if let Some(dma) = device.array.dma_engine_mut(col, row) {
        for _ in 0..MAX_TASK_QUEUE_DEPTH {
            assert!(dma.enqueue_task(0, 0, 0, false));
        }
    }

    // Build NPU instruction to enqueue another task
    let queue_offset = reg_layout.memory_channel_base + 4; // S2MM ch0 Task_Queue
    let npu_addr = ((col as u32) << 25) | ((row as u32) << 20) | queue_offset;

    let stream = NpuInstructionStream::from_instructions(vec![
        NpuInstruction::Write32 { reg_off: npu_addr, value: 0 },
    ]);

    // execute() should handle the full queue without dropping the task.
    // It may need many DMA-step retries but should eventually succeed
    // (or at worst, warn -- but never panic).
    let result = executor.execute(&stream, &mut device, &mut host_mem);
    assert!(result.is_ok());
}
```

Note: This test may need `NpuInstructionStream::from_instructions()` --
add that helper to the parser if it doesn't exist. Alternatively, use
`load_instructions()` on the executor and test via `try_advance()` loop.

**Step 2: Rewrite `execute()` to delegate to `try_advance()`**

```rust
/// Execute all instructions in a stream against the device.
///
/// This is the batch execution path, preserved for backward compatibility
/// with FFI callers and unit tests that don't have an engine loop. It
/// calls `try_advance()` internally, falling back to DMA-only stepping
/// when blocked on a full queue.
///
/// For interleaved execution (where full system stepping handles queue
/// draining), use `load()` + `try_advance()` from the engine loop instead.
pub fn execute(
    &mut self,
    stream: &NpuInstructionStream,
    device: &mut DeviceState,
    host_memory: &mut HostMemory,
) -> Result<(), String> {
    self.load(stream);

    const MAX_BLOCKED_CYCLES: u32 = 100_000;

    loop {
        match self.try_advance(device, host_memory) {
            AdvanceResult::Progressed => continue,
            AdvanceResult::Done => return Ok(()),
            AdvanceResult::Idle => return Ok(()),
            AdvanceResult::Blocked => {
                // Fall back to DMA-only stepping since we don't have
                // a full engine loop here.
                let mut drained = false;
                for _ in 0..MAX_BLOCKED_CYCLES {
                    device.array.step_all_dma(host_memory);
                    // try_advance checks queue space internally
                    match self.try_advance(device, host_memory) {
                        AdvanceResult::Blocked => continue,
                        AdvanceResult::Progressed => { drained = true; break; }
                        AdvanceResult::Done => return Ok(()),
                        AdvanceResult::Idle => return Ok(()),
                    }
                }
                if !drained {
                    // Same behavior as before: warn and move on.
                    // Force-transition out of BlockedOnQueue to avoid
                    // infinite loop.
                    if let ExecutorState::BlockedOnQueue {
                        col, row, channel, bd_id, next_index, ..
                    } = self.state.clone() {
                        let msg = format!(
                            "DMA tile({},{}) ch{} task queue full, BD {} dropped \
                             (batch mode: queue could not drain)",
                            col, row, channel, bd_id
                        );
                        log::warn!("{}", msg);
                        self.warnings.push(msg);
                        if next_index >= self.instructions.len() {
                            self.state = ExecutorState::Done;
                        } else {
                            self.state = ExecutorState::Executing { next_index };
                        }
                    }
                }
            }
        }
    }
}
```

**Step 3: Run tests**

Run: `cargo test --lib`
Expected: All tests pass (execute() behavior is equivalent to before)

**Step 4: Commit**

```
refactor(npu): execute() now delegates to try_advance() internally

Batch execution uses the same state machine as interleaved mode,
with DMA-only stepping as fallback when blocked. Behavior is equivalent
to the previous drain loop but shares the instruction dispatch code path.
```

---

## Task 5: Integrate Interleaved Execution into run_engine

**Files:**
- Modify: `src/testing/xclbin_suite.rs` (lines ~860-880 and ~1240-1300)

This is where the architectural change pays off. The test runner switches
from batch execution to interleaved execution.

**Step 1: Change run_single_inner to use load() instead of execute()**

In `run_single_inner` (around line 860-880), replace:

```rust
let (device, host_mem) = engine.device_and_host_memory();
if let Err(e) = executor.execute(&stream, device, host_mem) {
    let msg = format!("NPU instruction execution error: {}", e);
    log::warn!("[{}] {}", test.name, msg);
    test_warnings.push(msg);
}
```

With:

```rust
// Load instructions for interleaved execution.
// Instructions will be executed one-per-cycle in run_engine()
// alongside full system stepping, so DMA queue backpressure
// works correctly (the full array runs while waiting for
// queues to drain).
executor.load(&stream);
```

**Step 2: Change run_engine to call try_advance each cycle**

In `run_engine` (around line 1244), change the while loop:

```rust
while cycles < cycle_limit {
    // Advance NPU instruction execution (one instruction per cycle).
    // In BlockedOnQueue state, this checks if the queue drained and
    // retries the enqueue. Engine stepping below drains queues
    // naturally via full system simulation.
    if let Some(executor) = npu_executor.as_mut() {
        executor.try_advance(engine.device_mut(), engine.host_memory_mut());
    }

    engine.step();
    cycles = engine.total_cycles();

    // ... rest of loop unchanged (status check, syncs_satisfied, TDR) ...
}
```

**Important borrow issue:** `try_advance` needs `&mut DeviceState` and
`&mut HostMemory`, but `engine.step()` also needs `&mut self`. These
can't overlap. The solution: use `engine.device_and_host_memory()` to
get mutable references, call `try_advance`, then call `engine.step()`.

Actually, looking at the code more carefully: `engine.step()` takes `&mut self`
which includes device and host_memory. So we need to call `try_advance` BEFORE
`engine.step()`, using the split borrow:

```rust
while cycles < cycle_limit {
    // Advance NPU instruction execution before engine step.
    // try_advance needs &mut device + &mut host_memory, which
    // engine.device_and_host_memory() provides via split borrow.
    if let Some(executor) = npu_executor.as_mut() {
        let (device, host_mem) = engine.device_and_host_memory();
        executor.try_advance(device, host_mem);
    }

    engine.step();
    cycles = engine.total_cycles();

    match engine.status() {
        // ... existing status handling unchanged ...
    }

    // Check if DMA syncs are satisfied
    if let Some(executor) = npu_executor.as_mut() {
        if executor.syncs_satisfied(engine.device()) {
            // ... existing completion handling ...
        }
    }

    // TDR check unchanged
}
```

**Step 3: Run full test suite**

Run: `cargo test --lib`
Expected: All unit tests pass

**Step 4: Run npu-test to verify behavioral improvement**

Run: `cargo run --bin npu-test -- --no-build 2>&1 | tee /tmp/npu-test-interleaved.log`

Expected improvements:
- The 15 warnings from sync_task_complete_token tests should be GONE (shim DMA
  queues now drain via full system stepping, not DMA-only)
- Some timeouts may convert to passes or validation failures (queue-blocked
  tests now make progress)
- No regressions in existing passes

**Step 5: Commit**

```
feat(npu): interleave NPU instruction execution with engine stepping

Replace batch NPU instruction execution with per-cycle interleaving in
the test runner. Each engine cycle, try_advance() executes one NPU
instruction; when a DMA queue is full, the executor blocks and the
engine's full system stepping (cores + DMA + stream routing) drains
the queue naturally. This matches real hardware behavior where the host
firmware and AIE array run concurrently.

Eliminates task dropping and the 15 shim DMA queue warnings in
sync_task_complete_token tests.
```

---

## Task 6: Update FFI Path for Interleaved Execution

**Files:**
- Modify: `src/ffi/mod.rs` (lines ~418-426 and ~522-536)

The FFI path has the same batch-then-loop structure. Update it to match.

**Step 1: Change xdna_emu_execute_instructions to use load()**

In `xdna_emu_execute_instructions` (line ~420), replace:

```rust
if let Err(e) = handle.npu_executor.execute(&stream, device, host_mem) {
```

With:

```rust
handle.npu_executor.load(&stream);
// Instructions will be executed interleaved with engine stepping in xdna_emu_run()
```

**Step 2: Change xdna_emu_run to call try_advance each cycle**

In `xdna_emu_run` (line ~522), change the while loop:

```rust
while cycles < max {
    // Advance NPU instruction execution (interleaved with engine step)
    {
        let (device, host_mem) = handle.engine.device_and_host_memory();
        handle.npu_executor.try_advance(device, host_mem);
    }

    handle.engine.step();
    cycles += 1;

    if handle.engine.status() == EngineStatus::Halted {
        log::info!("Cores halted after {} cycles", cycles);
        break;
    }

    if handle.npu_executor.syncs_satisfied(handle.engine.device()) {
        log::info!("All DMA syncs satisfied after {} cycles", cycles);
        break;
    }
}
```

**Step 3: Run tests and verify**

Run: `cargo test --lib`
Expected: All tests pass

**Step 4: Commit**

```
feat(ffi): interleave NPU instruction execution in FFI engine loop

Update the FFI path to use load() + try_advance() per cycle, matching
the test runner's interleaved execution model. FFI callers now get
correct DMA backpressure behavior.
```

---

## Task 7: Clean Up and Final Verification

**Files:**
- Modify: `src/npu/executor.rs` (remove dead code)
- Modify: `docs/next-steps-2026-02-26.md` (update status)

**Step 1: Remove the old drain loop dead code**

If any remnants of the old `MAX_DRAIN_CYCLES` / `step_all_dma` drain loop
remain in `check_dma_trigger`, remove them. The `execute()` batch method
has its own fallback loop now.

**Step 2: Run full test suite**

Run: `cargo test --lib`
Expected: All tests pass

**Step 3: Run npu-test suite and save results**

Run: `cargo run --bin npu-test -- --no-build 2>&1 | tee /tmp/npu-test-interleaved-final.log`

Compare against baseline:
- Baseline: 19 pass, 17 valfail, 32 timeout, 1 skip, Warnings: 15 (3 tests)
- Expected: 19+ pass, 17- valfail, 32- timeout, Warnings: 0 (or near 0)

**Step 4: Update next-steps doc**

Update `docs/next-steps-2026-02-26.md` to record:
- Interleaving implemented
- Updated warning count
- Any tests that changed category (timeout -> pass, etc.)

**Step 5: Commit**

```
chore(npu): clean up after interleaving implementation

Remove dead drain loop code, update next-steps documentation with
post-interleaving test results.
```

---

## Summary

| Task | What | Risk |
|------|------|------|
| 1 | Add types (AdvanceResult, ExecutorState) | None (additive) |
| 2 | Implement load() + try_advance() | Low (new code alongside old) |
| 3 | check_dma_trigger signals BlockedOnQueue | Medium (behavioral change for interleaved path) |
| 4 | execute() delegates to try_advance() | Medium (batch path must be equivalent) |
| 5 | Integrate into run_engine | High (this is THE change) |
| 6 | Update FFI path | Low (mirrors Task 5) |
| 7 | Cleanup and verification | None |

Tasks 1-4 are safe incremental refactors. Task 5 is the critical integration.
Task 6 is a mirror of Task 5 for the FFI path. Task 7 is verification.

Each task has a commit point. If something goes wrong, `git revert` the last
commit and re-approach.
