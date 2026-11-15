//! NPU instruction executor.
//!
//! Executes NPU instructions against the device state, performing
//! register writes, address patches, and DMA triggers.
//!
//! The executor tracks Sync (TCT) instructions that specify completion
//! conditions. A "run" is complete when all Sync conditions are satisfied.

use super::cycle_cost::OpTarget;
use super::{NpuInstruction, NpuInstructionStream};
use crate::device::DeviceState;
use crate::device::host_memory::HostMemory;
use xdna_archspec::types::TileKind;
use xdna_archspec::aie2::SHIM_ROW;

/// Result of a single `try_advance()` step.
///
/// The caller (run_engine or FFI loop) uses this to know whether
/// execution is still in progress. The caller does not need to
/// inspect internal state -- just check the variant.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Fatal error -- execution must stop immediately.
    Error(String),
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
    /// Blocked waiting for a Sync (dma_await_task) to be satisfied.
    ///
    /// On real hardware, the NPU firmware blocks at each dma_await_task
    /// until the DMA channel signals completion (via task token or
    /// channel idle). This prevents subsequent BD reconfiguration from
    /// racing against live transfers.
    BlockedOnSync {
        /// Index of the NEXT instruction after the sync completes.
        next_index: usize,
        /// Index into pending_syncs identifying which sync we're waiting on.
        sync_index: usize,
        /// Executor cycle at which the sync was issued.
        issued_at: u64,
    },
    /// Blocked waiting for a MaskPoll condition to be satisfied.
    ///
    /// MaskPoll checks `(register & mask) == value` and blocks until true.
    /// Used by firmware to wait for DMA completion or other status bits.
    BlockedOnPoll {
        /// Index of the NEXT instruction after the poll completes.
        next_index: usize,
        /// Full NPU register address (encodes col/row/offset).
        reg_off: u32,
        /// Expected value after masking.
        value: u32,
        /// Mask to apply before comparison.
        mask: u32,
    },
    /// The rest of a resolved sync's wait: the controller resumes when it
    /// drains (`SyncWaitModel`, decided in `NpuExecutor::sync_charge`). The
    /// array keeps running, so stream data in flight when the sync resolved
    /// reaches its tiles before the next operation.
    FlushingStreams {
        /// Index of the NEXT instruction to execute after flush.
        next_index: usize,
        /// Remaining cycles before resuming.
        remaining: u32,
    },
    /// The controller is busy with an operation it has issued (its class
    /// cost, `CycleCostModel::cost_of`) or with servicing task-complete
    /// tokens; the next operation issues when this drains.
    RetiringInstruction {
        /// Index of the NEXT instruction to issue once retire completes.
        next_index: usize,
        /// Cycles still owed before the previously-issued instruction
        /// can be considered fully retired. Strictly > 0; reaching 0
        /// transitions back to Executing.
        cycles_remaining: u64,
    },
    /// All instructions executed.
    Done,
}

/// Host buffer information for address patching.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostBuffer {
    /// Base address in host memory.
    pub address: u64,
    /// Size in bytes.
    pub size: usize,
}

/// A pending sync condition.
///
/// Represents a DMA channel that must complete before the run is done.
/// Tracks whether the channel has ever been observed running, to avoid
/// declaring a sync satisfied before the transfer starts (initial idle
/// matches the completed-idle state on real hardware, but the host only
/// polls *after* submitting the task).
#[derive(Debug, Clone)]
pub struct PendingSync {
    /// Tile column.
    pub column: u8,
    /// Tile row.
    pub row: u8,
    /// DMA channel (relative within direction: 0 or 1).
    pub channel: u8,
    /// Direction: 0 = S2MM (receive), 1 = MM2S (send).
    pub direction: u8,
    /// Whether the channel has been observed in a running state.
    /// A sync is only satisfied after the channel has been running AND
    /// returned to idle -- never on the initial idle.
    started: bool,
    /// Consume-once latch. Set the moment this sync first observes its
    /// completion signal (a task-completion token, or the Channel_Running
    /// fallback). Once latched, the sync stays satisfied without re-examining
    /// -- and, critically, without popping a second token. This gives correct
    /// multi-`WAIT_TCTS`-on-one-channel semantics: each sync consumes exactly
    /// its own token.
    satisfied: bool,
}

/// What resolves a pending sync (result of the non-consuming `sync_signal`
/// peek). See `NpuExecutor::sync_signal` for the semantics of each variant.
enum SyncSignal {
    /// A task-completion token for the channel is pending (faithful HW path).
    Token,
    /// No token, but Channel_Running went running->idle (legacy fallback).
    Fallback,
    /// Still pending.
    None,
}

/// NPU instruction executor.
///
/// Executes the host-to-NPU command stream that triggers DMA transfers
/// and configures the shim tiles.
pub struct NpuExecutor {
    /// Host buffers for address patching (indexed by arg_idx).
    host_buffers: Vec<HostBuffer>,
    /// Number of instructions executed.
    executed_count: usize,
    /// Pending sync conditions from Sync instructions.
    pending_syncs: Vec<PendingSync>,
    /// Warnings collected during execution (surfaced in test output).
    warnings: Vec<String>,
    /// Internal state machine state.
    state: ExecutorState,
    /// Loaded instructions for interleaved execution via try_advance().
    instructions: Vec<NpuInstruction>,
    /// Operation-class costs of the controller, in controller cycles.
    cycle_model: super::CycleCostModel,
    /// Converts controller-cycle costs to array cycles, carrying the fraction
    /// across operations so a run of operations does not drift.
    to_array: crate::timebase::FirmwareToArray,
    /// Task-complete tokens issued array-wide as of the last time the
    /// controller looked (`None` until the sequence's first operation). The
    /// tokens issued since are the ones it has not yet serviced.
    tokens_seen: Option<u64>,
    /// Whether the last operation issued was a masked write to a tile
    /// register (the next one is then not the first of a run).
    after_tile_mask_write: bool,
    /// What a sync's wait costs, from the lateness of its token.
    sync_wait: super::SyncWaitModel,

    /// Executor-local cycle counter (array cycles), incremented once per
    /// `try_advance` call. Reset by `load_instructions`.
    npu_cycle: u64,
}

impl NpuExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            host_buffers: Vec::new(),
            executed_count: 0,
            pending_syncs: Vec::new(),
            warnings: Vec::new(),
            state: ExecutorState::Idle,
            instructions: Vec::new(),
            cycle_model: super::CycleCostModel::default(),
            to_array: crate::timebase::FirmwareToArray::new(super::CycleCostModel::default().clocks),
            tokens_seen: None,
            after_tile_mask_write: false,
            sync_wait: super::SyncWaitModel::default(),
            npu_cycle: 0,
        }
    }

    /// Replace the operation-class cost model (and its clocks).
    pub fn set_cycle_model(&mut self, model: super::CycleCostModel) {
        self.to_array = crate::timebase::FirmwareToArray::new(model.clocks);
        self.sync_wait = super::SyncWaitModel::npu1(model.clocks);
        self.cycle_model = model;
    }

    /// Get warnings collected during execution.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Get the current executor state (for testing/debugging).
    #[cfg(test)]
    pub(crate) fn state(&self) -> &ExecutorState {
        &self.state
    }

    /// Whether the executor has finished processing all instructions.
    ///
    /// Returns true when all NPU instructions have been executed (state is
    /// Done) or when no instructions were loaded (Idle). This is needed to
    /// gate `syncs_satisfied()` checks -- sync conditions are only populated
    /// as Sync instructions are executed, so checking before all instructions
    /// are processed would see an empty pending_syncs and falsely report
    /// completion.
    pub fn is_done(&self) -> bool {
        matches!(self.state, ExecutorState::Done | ExecutorState::Idle)
    }

    /// Whether the executor is currently blocked on an unsatisfied MaskPoll.
    ///
    /// Used by the run loop to detect the graceful-poll-termination condition:
    /// when the engine becomes quiescent (stalled or halted with no pending
    /// work) while the executor is stuck in `BlockedOnPoll`, the poll can never
    /// be satisfied and the run should end deterministically with a distinct
    /// `MaskPollUnsatisfied` terminal reason rather than hanging forever.
    ///
    /// This is the emulator-contract check for the MASKPOLL halt-synchronization
    /// injected into `insts.bin` by the debug_halt_probe (spec §4.2). The poll
    /// satisfies when `Core_Status[16]` (DEBUG_HALT) sets after a breakpoint
    /// fires -- on the real NPU and, since Phase B Units 1/1b, on the emulator
    /// too (the breakpoint arms, the pre-execute seam halts before-commit, and
    /// both read paths observe DEBUG_HALT). This method instead guards a
    /// *genuinely* unsatisfiable poll (e.g. polling a register nothing ever
    /// drives), letting the run loop terminate deterministically rather than
    /// spinning forever -- retained as independent hardening, no longer the
    /// debug_halt_probe's path.
    pub fn is_blocked_on_poll(&self) -> bool {
        matches!(self.state, ExecutorState::BlockedOnPoll { .. })
    }

    /// Get pending sync conditions.
    pub fn pending_syncs(&self) -> &[PendingSync] {
        &self.pending_syncs
    }

    /// What the instruction stream is stuck on, when it is blocked on a Start_Queue write.
    ///
    /// A Start_Queue push into a full task queue blocks the writer (silicon, shim-queue probes).
    /// When the write that would drain the queue sits behind the blocked push, nothing can make
    /// progress: the emulator halts with this diagnosis (a deadlock) instead of spinning.
    pub fn blocked_write_description(&self, device: &DeviceState) -> Option<String> {
        let ExecutorState::BlockedOnQueue { col, row, channel, bd_id, .. } = self.state else {
            return None;
        };
        let dma = device.array.dma_engine(col, row)?;
        let s2mm = dma.s2mm_channel_count() as u8;
        let (dir, ch) = if channel < s2mm {
            ("S2MM", channel)
        } else {
            ("MM2S", channel - s2mm)
        };
        Some(format!(
            "NPU instruction stream blocked on a Start_Queue write (BD {bd_id}) to tile ({col},{row}) {dir} ch{ch}: \
             task queue full ({} waiting, {} of them staged, queue {}/{}); nothing else can drain it (deadlock: \
             the push blocks the writer, and the writes behind it never run)",
            dma.pending_tasks(channel),
            dma.staged_task_slots(channel),
            dma.task_queue_size(channel),
            crate::device::dma::MAX_TASK_QUEUE_DEPTH,
        ))
    }

    /// Map a sync's (direction, channel) to absolute DMA channel index.
    fn sync_abs_channel(sync: &PendingSync, device: &DeviceState) -> u8 {
        // S2MM channels come first, then MM2S. The number of S2MM channels
        // varies by tile type: 2 for Compute/Shim, 6 for MemTile.
        let s2mm_count = device
            .array
            .dma_engine(sync.column, sync.row)
            .map_or(2, |dma| dma.s2mm_channel_count() as u8);
        if sync.direction == 1 {
            s2mm_count + sync.channel
        } else {
            sync.channel
        }
    }

    /// Non-consuming completion signal for a single sync (peek).
    ///
    /// This computes *what* satisfies the sync without side effects on device
    /// state (it may still advance the sync's own `started` flag, which is the
    /// running->idle latch the fallback needs). Both the consuming transition
    /// (`is_sync_satisfied`) and the read-only termination check
    /// (`syncs_satisfied`) share it.
    ///
    /// - `Token`: a task-completion token for this channel is pending. This is
    ///   the faithful hardware signal -- `dma_await_task`/`WAIT_TCTS` waits for
    ///   exactly this. It fires even if the channel is still running a *later*
    ///   task, which is correct: the token means the awaited task finished.
    /// - `Fallback`: no token, but Channel_Running has gone running->idle (or a
    ///   fast completion already retired a transfer). Tasks that never set
    ///   Enable_Token_Issue emit no token, so the legacy poll still resolves
    ///   them. The consuming path logs when it takes this route so a sweep can
    ///   surface which kernels depend on the non-token path.
    /// - `None`: still pending.
    fn sync_signal(sync: &mut PendingSync, device: &DeviceState) -> SyncSignal {
        let abs_channel = Self::sync_abs_channel(sync, device);
        let dma = match device.array.dma_engine(sync.column, sync.row) {
            Some(d) => d,
            None => return SyncSignal::None,
        };

        // Faithful path: the task-completion token is the hardware signal.
        if dma.has_task_token_for_channel(abs_channel) {
            return SyncSignal::Token;
        }

        // Fallback: Channel_Running running->idle, or fast completion.
        let reg_layout = crate::device::regdb::device_reg_layout();
        let status = dma.get_channel_status(abs_channel);
        let status_layout = if dma.tile_kind.is_mem() {
            &reg_layout.memtile_status
        } else {
            &reg_layout.memory_status
        };
        let channel_running = status_layout.channel_running.extract_bool(status);

        if channel_running {
            // Channel is active -- mark as started and keep waiting.
            sync.started = true;
            return SyncSignal::None;
        }

        // Channel is idle. Two fallback satisfaction conditions:
        // 1. We previously observed it running (normal polling path).
        if sync.started {
            return SyncSignal::Fallback;
        }
        // 2. Fast completion: channel finished before we first polled. No
        //    pending work AND at least one transfer retired (not initial idle).
        if let Some(stats) = dma.channel_stats(abs_channel) {
            if stats.transfers_completed > 0 && dma.pending_tasks(abs_channel) == 0 {
                return SyncSignal::Fallback;
            }
        }

        SyncSignal::None
    }

    /// Resolve a single sync condition, consuming its token if present.
    ///
    /// This is the authoritative blocked->satisfied *transition* (driven from
    /// `try_advance`'s `BlockedOnSync` arm, which holds `&mut DeviceState`).
    /// On the token path it pops exactly one matching token; the `satisfied`
    /// latch then keeps the sync resolved without re-popping. See `sync_signal`
    /// for the token-vs-fallback semantics.
    fn is_sync_satisfied(sync: &mut PendingSync, device: &mut DeviceState) -> bool {
        if sync.satisfied {
            return true;
        }
        match Self::sync_signal(sync, device) {
            SyncSignal::Token => {
                // Consume exactly one token for this channel (the faithful TCT
                // completion). abs_channel is recomputed for the mut borrow.
                let abs_channel = Self::sync_abs_channel(sync, device);
                if let Some(dma) = device.array.dma_engine_mut(sync.column, sync.row) {
                    dma.pop_task_token_for_channel(abs_channel);
                }
                sync.satisfied = true;
                true
            }
            SyncSignal::Fallback => {
                log::trace!(
                    "NPU sync ({},{}) ch{} dir{} satisfied via Channel_Running fallback (no token)",
                    sync.column,
                    sync.row,
                    sync.channel,
                    sync.direction,
                );
                sync.satisfied = true;
                true
            }
            SyncSignal::None => false,
        }
    }

    /// Check if all sync conditions are satisfied (read-only termination check).
    ///
    /// Non-consuming: this is the run loop's "are we done yet?" poll, called
    /// only once the executor `is_done()` -- by which point every sync has
    /// already transitioned (and consumed its token) through the `BlockedOnSync`
    /// arm, so this just reads the `satisfied` latch. The `sync_signal` peek
    /// remains for direct callers (unit tests) that never route through
    /// `try_advance`; it never pops a token.
    pub fn syncs_satisfied(&mut self, device: &DeviceState) -> bool {
        if self.pending_syncs.is_empty() {
            return true;
        }

        for sync in &mut self.pending_syncs {
            if sync.satisfied {
                continue;
            }
            if matches!(Self::sync_signal(sync, device), SyncSignal::None) {
                return false;
            }
        }
        true
    }

    /// Set host buffers for address patching.
    ///
    /// The runtime_sequence in mlir-aie takes arguments like:
    /// `%arg0: memref<64xi32>, %arg1: memref<32xi32>, %arg2: memref<64xi32>`
    ///
    /// These correspond to host_buffers[0], host_buffers[1], host_buffers[2].
    pub fn set_host_buffers(&mut self, buffers: Vec<HostBuffer>) {
        self.host_buffers = buffers;
    }

    /// Add a host buffer.
    pub fn add_host_buffer(&mut self, address: u64, size: usize) {
        self.host_buffers.push(HostBuffer { address, size });
    }

    /// Get all host buffers (for reading back trace data, etc.).
    pub fn host_buffers(&self) -> &[HostBuffer] {
        &self.host_buffers
    }

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
        // Reset the executor's cycle accounting -- each runtime sequence
        // starts fresh.
        self.npu_cycle = 0;
        self.to_array = crate::timebase::FirmwareToArray::new(self.cycle_model.clocks);
        self.tokens_seen = None;
        self.after_tile_mask_write = false;
        // A runtime sequence waits only on its own Syncs; a runlist's
        // previous command's must not gate (or "stall") this one.
        self.pending_syncs.clear();
        // Log instruction summary for debugging complex multi-tile flows
        if instructions.len() > 10 {
            let mut syncs = 0;
            let mut enqueues = 0;
            let mut writes = 0;
            let mut patches = 0;
            for instr in &instructions {
                match instr {
                    NpuInstruction::Sync { .. } => syncs += 1,
                    NpuInstruction::BlockWrite { .. } => {
                        enqueues += 1;
                        writes += 1;
                    }
                    NpuInstruction::Write32 { .. } => writes += 1,
                    NpuInstruction::DdrPatch { .. } => patches += 1,
                    _ => {}
                }
            }
            log::debug!(
                "NPU instruction buffer: {} total ({} Write, {} BlockWrite, {} DdrPatch, {} Sync)",
                instructions.len(),
                writes,
                enqueues,
                patches,
                syncs
            );
        }
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
    pub fn try_advance(&mut self, device: &mut DeviceState, host_memory: &mut HostMemory) -> AdvanceResult {
        // One array cycle per advance call. Idle/Done ticks advance it too.
        self.npu_cycle = self.npu_cycle.saturating_add(1);

        match self.state.clone() {
            ExecutorState::Idle => AdvanceResult::Idle,
            ExecutorState::Done => AdvanceResult::Done,

            ExecutorState::Executing { next_index } => {
                if next_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    return AdvanceResult::Done;
                }

                // Operation boundary: tokens that arrived while the controller
                // executed the previous operations are serviced before the next
                // one issues.
                let due = self.unserviced_tokens(device);
                if due > 0 {
                    let cycles = self.to_array.array_cycles(due * self.cycle_model.token_service);
                    log::debug!("NPU controller services {due} task-complete token(s): {cycles} cycles");
                    if cycles > 1 {
                        self.state =
                            ExecutorState::RetiringInstruction { next_index, cycles_remaining: cycles - 1 };
                        return AdvanceResult::Blocked;
                    }
                }

                let instr = self.instructions[next_index].clone();
                let target = match &instr {
                    NpuInstruction::Write32 { reg_off, .. } | NpuInstruction::MaskWrite { reg_off, .. } => {
                        Self::op_target(*reg_off, device)
                    }
                    _ => OpTarget::Shim { push: false },
                };
                let cost = self.cycle_model.cost_of(&instr, target, self.after_tile_mask_write);

                if let Err(e) = self.execute_instruction(&instr, device, host_memory) {
                    let msg = format!("NPU instruction {} execution error: {}", next_index, e);
                    log::error!("{}", msg);
                    return AdvanceResult::Error(msg);
                }
                self.executed_count += 1;
                self.after_tile_mask_write = matches!(instr, NpuInstruction::MaskWrite { .. })
                    && matches!(target, OpTarget::MemTile | OpTarget::Compute);

                // would_block_on_queue may have transitioned us to BlockedOnQueue,
                // or execute_instruction may have transitioned to BlockedOnSync.
                // Fix up next_index in either case and return Blocked.
                match &mut self.state {
                    ExecutorState::BlockedOnQueue { next_index: ref mut ni, .. } => {
                        *ni = next_index + 1;
                        return AdvanceResult::Blocked;
                    }
                    ExecutorState::BlockedOnSync { next_index: ref mut ni, .. } => {
                        *ni = next_index + 1;
                        return AdvanceResult::Blocked;
                    }
                    ExecutorState::BlockedOnPoll { next_index: ref mut ni, .. } => {
                        *ni = next_index + 1;
                        return AdvanceResult::Blocked;
                    }
                    _ => {}
                }

                // The operation holds the controller for its class cost; the
                // next one issues that many array cycles after this one.
                let new_index = next_index + 1;
                if new_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    AdvanceResult::Done
                } else {
                    self.retire(new_index, cost);
                    AdvanceResult::Progressed
                }
            }

            ExecutorState::BlockedOnQueue {
                next_index,
                col,
                row,
                channel,
                bd_id,
                repeat,
                enable_token,
                ..
            } => {
                // Check if the queue has drained enough for our enqueue
                let has_space = device
                    .array
                    .dma_engine(col, row)
                    .map_or(true, |dma| dma.task_queue_has_space(channel));

                if has_space {
                    // Re-parse BD before snapshot (control packets may have deferred parsing)
                    device.reparse_dirty_bd(col, row, bd_id as usize);
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
                    AdvanceResult::Blocked
                }
            }

            ExecutorState::BlockedOnSync { next_index, sync_index, issued_at } => {
                // Poll the sync condition each cycle. On real hardware,
                // dma_await_task blocks until the DMA channel's task token
                // returns (or Channel_Running goes idle).
                let satisfied = if sync_index < self.pending_syncs.len() {
                    Self::is_sync_satisfied(&mut self.pending_syncs[sync_index], device)
                } else {
                    true // Invalid index -- treat as satisfied
                };

                if satisfied {
                    log::info!("NPU Sync #{} satisfied, resuming instruction {}", sync_index, next_index,);
                    // Tokens that arrived during the wait are covered by the
                    // sync's own cost, not serviced one by one.
                    self.absorb_tokens(device);
                    let charge = self.sync_charge(issued_at, self.npu_cycle);
                    self.state = ExecutorState::FlushingStreams { next_index, remaining: charge };
                    AdvanceResult::Progressed
                } else {
                    AdvanceResult::Blocked
                }
            }

            ExecutorState::BlockedOnPoll { next_index, reg_off, value, mask } => {
                let (col, row, offset) = decode_npu_address(reg_off, device.start_col);
                let current = device
                    .tile_mut(col as usize, row as usize)
                    .map_or(0, |tile| tile.read_register(offset));
                if (current & mask) == value {
                    log::debug!("NPU MaskPoll: reg=0x{:08X} satisfied (current=0x{:08X})", reg_off, current);
                    if next_index >= self.instructions.len() {
                        self.state = ExecutorState::Done;
                        AdvanceResult::Done
                    } else {
                        // The satisfying read retires at the mask-poll cost.
                        self.retire(next_index, self.cycle_model.mask_poll);
                        AdvanceResult::Progressed
                    }
                } else {
                    AdvanceResult::Blocked
                }
            }

            ExecutorState::FlushingStreams { next_index, remaining } => {
                // The rest of the sync's wait. Each cycle the caller runs the
                // array, so in-transit stream data keeps moving meanwhile.
                if remaining <= 1 {
                    self.absorb_tokens(device);
                    self.state = ExecutorState::Executing { next_index };
                } else {
                    self.state = ExecutorState::FlushingStreams { next_index, remaining: remaining - 1 };
                }
                AdvanceResult::Progressed
            }

            ExecutorState::RetiringInstruction { next_index, cycles_remaining } => {
                // Decrement the cycles-owed counter; resume issuing
                // instructions when it drains. Mirrors the FlushingStreams
                // pattern above.
                if cycles_remaining <= 1 {
                    self.state = ExecutorState::Executing { next_index };
                } else {
                    self.state = ExecutorState::RetiringInstruction {
                        next_index,
                        cycles_remaining: cycles_remaining - 1,
                    };
                }
                AdvanceResult::Blocked
            }
        }
    }

    /// Park the controller for an operation of `controller_cycles` issued this
    /// cycle: the next operation (`next_index`) issues that many array cycles later.
    fn retire(&mut self, next_index: usize, controller_cycles: u64) {
        let cycles = self.to_array.array_cycles(controller_cycles);
        self.state = if cycles > 1 {
            ExecutorState::RetiringInstruction { next_index, cycles_remaining: cycles - 1 }
        } else {
            ExecutorState::Executing { next_index }
        };
    }

    /// Tokens issued since the controller last looked, which it now services.
    /// The first look of a sequence only records the count.
    fn unserviced_tokens(&mut self, device: &DeviceState) -> u64 {
        let total = device.array.task_tokens_issued();
        let due = self.tokens_seen.map_or(0, |seen| total.saturating_sub(seen));
        self.tokens_seen = Some(total);
        due
    }

    /// Mark every token issued so far as covered (by a sync's wait).
    fn absorb_tokens(&mut self, device: &DeviceState) {
        self.tokens_seen = Some(device.array.task_tokens_issued());
    }

    /// Array cycles a resolved sync holds the controller after its awaited
    /// token was observed: the one place a sync's cost is decided
    /// (`SyncWaitModel`, from the token's lateness after the issue).
    ///
    /// `issued_at` is the executor cycle the sync was issued at and
    /// `token_arrival` the cycle its token was observed. The first poll runs
    /// the cycle after the issue, so a token observed there was already
    /// waiting (the fast path; its service was charged at the operation
    /// boundary before the sync). The charge holds the controller in
    /// `FlushingStreams`, whose last cycle hands over to the next operation:
    /// the next operation issues `resume_after_issue` cycles after the sync.
    fn sync_charge(&self, issued_at: u64, token_arrival: u64) -> u32 {
        use super::TokenWait;
        let waited = token_arrival.saturating_sub(issued_at);
        let wait = if waited <= 1 {
            TokenWait::AlreadyArrived
        } else {
            TokenWait::ArrivedAfter(waited)
        };
        let resume = self.sync_wait.resume_after_issue(wait);
        log::debug!(
            "NPU sync: token after {waited} cycles, {:?}, resume {resume} after issue",
            self.sync_wait.region(wait)
        );
        u32::try_from(resume.saturating_sub(waited + 1).max(1)).unwrap_or(u32::MAX)
    }

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
                AdvanceResult::Done | AdvanceResult::Idle => return Ok(()),
                AdvanceResult::Error(msg) => return Err(msg),
                AdvanceResult::Blocked => {
                    // Fall back to DMA-only stepping since we don't have
                    // a full engine loop here.
                    let mut drained = false;
                    for _ in 0..MAX_BLOCKED_CYCLES {
                        device.array.step_all_dma(host_memory);
                        match self.try_advance(device, host_memory) {
                            AdvanceResult::Blocked => continue,
                            AdvanceResult::Progressed => {
                                drained = true;
                                break;
                            }
                            AdvanceResult::Done | AdvanceResult::Idle => return Ok(()),
                            AdvanceResult::Error(msg) => return Err(msg),
                        }
                    }
                    if !drained {
                        let msg = match self.state.clone() {
                            ExecutorState::BlockedOnQueue { col, row, channel, bd_id, .. } => format!(
                                "DMA tile({},{}) ch{} task queue full, BD {} dropped \
                                 (batch mode: queue could not drain) -- task lost",
                                col, row, channel, bd_id,
                            ),
                            ExecutorState::BlockedOnSync { sync_index, .. } => {
                                let sync = &self.pending_syncs[sync_index];
                                let dir = if sync.direction == 0 { "S2MM" } else { "MM2S" };
                                format!(
                                    "Sync #{} on ({},{}) {} ch{} not satisfied after {} DMA-only cycles \
                                     (batch mode: needs full engine loop)",
                                    sync_index, sync.column, sync.row, dir, sync.channel, MAX_BLOCKED_CYCLES,
                                )
                            }
                            _ => "Blocked state could not resolve (batch mode)".to_string(),
                        };
                        log::error!("{}", msg);
                        return Err(msg);
                    }
                }
            }
        }
    }

    /// Execute a single instruction.
    fn execute_instruction(
        &mut self,
        instr: &NpuInstruction,
        device: &mut DeviceState,
        host_memory: &mut HostMemory,
    ) -> Result<(), String> {
        match instr {
            NpuInstruction::Write32 { reg_off, value } => {
                self.execute_write32(*reg_off, *value, device, host_memory)
            }

            NpuInstruction::BlockWrite { reg_off, values } => {
                self.execute_blockwrite(*reg_off, values, device, host_memory)
            }

            NpuInstruction::MaskWrite { reg_off, value, mask } => {
                self.execute_maskwrite(*reg_off, *value, *mask, device, host_memory)
            }

            NpuInstruction::MaskPoll { reg_off, value, mask } => {
                // MaskPoll blocks until (register & mask) == value.
                // Check the condition now; if not met, transition to
                // BlockedOnPoll and let try_advance() poll each cycle.
                let (col, row, offset) = decode_npu_address(*reg_off, device.start_col);
                let current = device
                    .tile_mut(col as usize, row as usize)
                    .map_or(0, |tile| tile.read_register(offset));
                if (current & mask) == *value {
                    log::debug!(
                        "NPU MaskPoll: reg=0x{:08X} satisfied immediately (current=0x{:08X} & 0x{:08X} == 0x{:08X})",
                        reg_off, current, mask, value
                    );
                    Ok(())
                } else {
                    log::debug!(
                        "NPU MaskPoll: reg=0x{:08X} blocking (current=0x{:08X} & 0x{:08X} = 0x{:08X}, want 0x{:08X})",
                        reg_off, current, mask, current & mask, value
                    );
                    self.state = ExecutorState::BlockedOnPoll {
                        next_index: 0, // fixed up by try_advance
                        reg_off: *reg_off,
                        value: *value,
                        mask: *mask,
                    };
                    Ok(())
                }
            }

            NpuInstruction::DdrPatch { reg_addr, arg_idx, arg_plus } => {
                self.execute_ddr_patch(*reg_addr, *arg_idx, *arg_plus, device)
            }

            NpuInstruction::Sync { channel, column, row, direction, .. } => {
                let dir_str = if *direction == 0 { "S2MM" } else { "MM2S" };
                // Sync's column field is a logical (partition-relative)
                // tile column; shift to physical so the wait targets the
                // correct DMA engine after partition relocation.
                let physical_col = column.saturating_add(device.start_col);
                log::info!(
                    "NPU Sync: blocking on ({},{}) {} ch{} (sync #{})",
                    physical_col,
                    row,
                    dir_str,
                    channel,
                    self.pending_syncs.len(),
                );
                // Record this sync condition and block until it's satisfied.
                // On real hardware, the firmware blocks at dma_await_task until
                // the DMA channel signals completion. This prevents subsequent
                // BD reconfiguration from racing against live transfers.
                self.pending_syncs.push(PendingSync {
                    column: physical_col,
                    row: *row,
                    channel: *channel,
                    direction: *direction,
                    started: false,
                    satisfied: false,
                });
                let sync_index = self.pending_syncs.len() - 1;
                // Transition to BlockedOnSync -- try_advance() will poll this
                // sync each cycle and only advance when it's satisfied.
                // (The caller sets next_index after execute_instruction returns.)
                self.state = ExecutorState::BlockedOnSync {
                    next_index: 0, // placeholder, fixed up by try_advance
                    sync_index,
                    issued_at: self.npu_cycle,
                };
                Ok(())
            }

            NpuInstruction::Unknown { opcode, data } => Err(format!(
                "NPU Unknown instruction: opcode=0x{:02X} data_len={} -- \
                     unrecognized opcode, cannot continue",
                opcode,
                data.len()
            )),
        }
    }

    /// Execute a Write32 instruction.
    fn execute_write32(
        &mut self,
        reg_off: u32,
        value: u32,
        device: &mut DeviceState,
        _host_memory: &mut HostMemory,
    ) -> Result<(), String> {
        log::debug!("NPU Write32: reg=0x{:08X} value=0x{:08X}", reg_off, value);

        // Decode tile address from register offset (logical -> physical via start_col).
        let (col, row, offset) = decode_npu_address(reg_off, device.start_col);

        // Data memory writes go directly to tile memory (not the register bus).
        let is_data_mem = device
            .tile(col as usize, row as usize)
            .map_or(false, |tile| is_data_memory_offset(tile, offset));

        if is_data_mem {
            if let Some(tile) = device.tile_mut(col as usize, row as usize) {
                log::debug!(
                    "NPU Write32 -> data memory: tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    col,
                    row,
                    offset,
                    value
                );
                tile.write_data_u32(offset as usize, value);
            }
        } else if device.tile(col as usize, row as usize).is_some() {
            // Pre-flight: block if writing to a full DMA task queue.
            if self.would_block_on_queue(col, row, offset, value, device) {
                return Ok(());
            }
            // Route through the unified register bus (same path as CDO and
            // control packets). This handles stream switch config, DMA engine
            // integration, locks, trace, shim mux, and all other register
            // types through a single dispatch.
            device.write_tile_register(col, row, offset, value);
        } else {
            return Err(format!(
                "NPU Write32 to non-existent tile ({}, {}): offset=0x{:05X} -- config targets missing tile",
                col, row, offset,
            ));
        }

        Ok(())
    }

    /// Execute a BlockWrite instruction.
    fn execute_blockwrite(
        &mut self,
        reg_off: u32,
        values: &[u32],
        device: &mut DeviceState,
        _host_memory: &mut HostMemory,
    ) -> Result<(), String> {
        log::debug!("NPU BlockWrite: reg=0x{:08X} count={}", reg_off, values.len());

        let (col, row, base_offset) = decode_npu_address(reg_off, device.start_col);

        // Data memory writes go directly to tile memory (contiguous block).
        let is_data_mem = device
            .tile(col as usize, row as usize)
            .map_or(false, |tile| is_data_memory_offset(tile, base_offset));

        if is_data_mem {
            if let Some(tile) = device.tile_mut(col as usize, row as usize) {
                log::debug!(
                    "NPU BlockWrite -> data memory: tile({},{}) offset=0x{:05X} count={}",
                    col,
                    row,
                    base_offset,
                    values.len()
                );
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                tile.write_data(base_offset as usize, &bytes);
            }
        } else if device.tile(col as usize, row as usize).is_some() {
            // Pre-flight: check if any word targets a full DMA task queue.
            for (i, &value) in values.iter().enumerate() {
                let offset = base_offset + (i as u32) * 4;
                if self.would_block_on_queue(col, row, offset, value, device) {
                    return Ok(());
                }
            }
            // Route each word through the unified register bus.
            for (i, &value) in values.iter().enumerate() {
                let offset = base_offset + (i as u32) * 4;
                device.write_tile_register(col, row, offset, value);
            }
        }

        Ok(())
    }

    /// Execute a MaskWrite instruction.
    fn execute_maskwrite(
        &mut self,
        reg_off: u32,
        value: u32,
        mask: u32,
        device: &mut DeviceState,
        _host_memory: &mut HostMemory,
    ) -> Result<(), String> {
        let (col, row, offset) = decode_npu_address(reg_off, device.start_col);
        log::debug!(
            "NPU MaskWrite: reg=0x{:08X} -> tile({},{}) offset=0x{:05X} value=0x{:08X} mask=0x{:08X}",
            reg_off,
            col,
            row,
            offset,
            value,
            mask
        );

        let is_data_mem = device
            .tile(col as usize, row as usize)
            .map_or(false, |tile| is_data_memory_offset(tile, offset));

        if is_data_mem {
            if let Some(tile) = device.tile_mut(col as usize, row as usize) {
                let current = tile.read_data_u32(offset as usize).unwrap_or(0);
                let new_value = (current & !mask) | (value & mask);
                log::info!(
                    "NPU MaskWrite -> data memory: tile({},{}) offset=0x{:05X} \
                    current=0x{:08X} -> 0x{:08X}",
                    col,
                    row,
                    offset,
                    current,
                    new_value
                );
                tile.write_data_u32(offset as usize, new_value);
            }
        } else if device.tile(col as usize, row as usize).is_some() {
            // Read current value, apply mask, write through unified register bus.
            let current = device
                .tile_mut(col as usize, row as usize)
                .map(|t| t.read_register(offset))
                .unwrap_or(0);
            let new_value = (current & !mask) | (value & mask);
            if self.would_block_on_queue(col, row, offset, new_value, device) {
                return Ok(());
            }
            device.write_tile_register(col, row, offset, new_value);
        }

        Ok(())
    }

    /// Execute a DDR patch instruction, as the firmware does.
    ///
    /// Two modes of operation:
    ///
    /// 1. **Host buffer mode** (non-ELF path): host_buffers holds the host
    ///    address of each bound argument, from the ERT regmap. The BD gets
    ///    `slot + arg_plus`, plus `DDR_AIE_ADDR_OFFSET` for the first
    ///    `FIRMWARE_TRANSLATED_ARGS` arguments (firmware translates those into
    ///    the AIE view; a later argument's arg_plus carries the offset itself).
    ///    An argument the host did not bind has slot value 0, so its BD points
    ///    at host address 0 upward, which faults like silicon does.
    ///
    /// 2. **Pre-patched mode** (ELF module path): host_buffers is empty.  XRT
    ///    already patched the block-write payloads with the AIE-view address
    ///    (BO address plus the offset), so DdrPatch is already satisfied.
    fn execute_ddr_patch(
        &mut self,
        reg_addr: u32,
        arg_idx: u8,
        arg_plus: u32,
        device: &mut DeviceState,
    ) -> Result<(), String> {
        use crate::device::dma::{DDR_AIE_ADDR_OFFSET, FIRMWARE_TRANSLATED_ARGS};
        if self.host_buffers.is_empty() {
            log::debug!(
                "NPU DdrPatch (pre-patched): reg=0x{reg_addr:08X} arg_idx={arg_idx} \
                 arg_plus={arg_plus}; preserving XRT's AIE-visible address"
            );
            return Ok(());
        }

        let (col, row, offset) = decode_npu_address(reg_addr, device.start_col);

        let slot = match self.host_buffers.get(arg_idx as usize) {
            Some(buffer) => buffer.address,
            None => {
                log::warn!(
                    "NPU DdrPatch: arg_idx {arg_idx} is not bound ({} host buffers); its slot value is 0",
                    self.host_buffers.len()
                );
                0
            }
        };
        let translation = if arg_idx < FIRMWARE_TRANSLATED_ARGS {
            DDR_AIE_ADDR_OFFSET
        } else {
            0
        };
        let patched_addr = slot.wrapping_add(arg_plus as u64).wrapping_add(translation);

        log::debug!(
            "NPU DdrPatch: reg=0x{:08X} arg_idx={} arg_plus={} -> AIE addr=0x{:016X}",
            reg_addr,
            arg_idx,
            arg_plus,
            patched_addr
        );

        self.write_bd_address(col, row, offset, patched_addr, device);
        Ok(())
    }

    /// Write a 48-bit address to a BD register pair, preserving non-address
    /// bits in the high word for shim BDs (Enable_Packet, Packet_ID, etc.).
    fn write_bd_address(&self, col: u8, row: u8, offset: u32, addr: u64, device: &mut DeviceState) {
        // Write low 32 bits through the unified register bus.
        device.write_tile_register(col, row, offset, addr as u32);

        // Write high 16 bits to the next word.  For shim DMA BDs, the next
        // word shares its 32 bits between Base_Address_High[15:0] and other
        // fields (Enable_Packet, Packet_ID, Packet_Type, OoO_BD_ID).  Use
        // read-modify-write to preserve the non-address bits.
        let high_bits = (addr >> 32) as u32;
        let is_shim_bd = row == SHIM_ROW && bd_index_for_blockwrite(row, offset).is_some();
        if is_shim_bd {
            let current = device
                .tile_mut(col as usize, row as usize)
                .map(|t| t.read_register(offset + 4))
                .unwrap_or(0);
            let merged = (current & 0xFFFF_0000) | (high_bits & 0x0000_FFFF);
            device.write_tile_register(col, row, offset + 4, merged);
        } else {
            device.write_tile_register(col, row, offset + 4, high_bits);
        }

        // The register bus marks BD words dirty when written via
        // write_tile_register(). They will be reparsed at enqueue time
        // (reparse_dirty_bd is called from write_dma_channel).
    }

    /// Pre-flight check: would writing this value block on a full DMA task queue?
    ///
    /// If the offset is a DMA Start_Queue register and the channel's task queue
    /// is already full, transitions to `BlockedOnQueue` and returns `true`.
    /// The caller must skip the write and return early when this returns `true`.
    ///
    /// This replaces the old post-write `check_dma_trigger()` approach: we now
    /// block BEFORE the write (matching real hardware where firmware polls
    /// Task_Queue_Size before writing Start_Queue).
    fn would_block_on_queue(
        &mut self,
        col: u8,
        row: u8,
        offset: u32,
        value: u32,
        device: &DeviceState,
    ) -> bool {
        let Some((tile_kind, abs_channel)) = Self::queue_channel(col, row, offset, device) else {
            return false;
        };

        // Check queue depth
        let queue_full = device
            .array
            .dma_engine(col, row)
            .map_or(false, |dma| !dma.task_queue_has_space(abs_channel));

        if queue_full {
            let bd_mask = match tile_kind {
                TileKind::Mem => 0x3F,
                _ => 0xF,
            };
            let bd_id = (value & bd_mask) as u8;
            let repeat = ((value >> 16) & 0xFF) as u8;
            let enable_token = (value >> 31) & 1 != 0;

            log::debug!(
                "DMA tile({},{}) ch{} queue full, deferring BD {} enqueue",
                col,
                row,
                abs_channel,
                bd_id
            );
            self.state = ExecutorState::BlockedOnQueue {
                next_index: self.executed_count + 1,
                col,
                row,
                channel: abs_channel,
                bd_id,
                repeat,
                enable_token,
            };
            return true;
        }

        false
    }

    /// What a write at `reg_off` targets, for its operation cost: a shim
    /// register (and whether it is a task-queue push), a memtile or a compute
    /// tile.
    fn op_target(reg_off: u32, device: &DeviceState) -> OpTarget {
        let (col, row, offset) = decode_npu_address(reg_off, device.start_col);
        match device.tile(col as usize, row as usize).map(|t| t.tile_kind) {
            Some(TileKind::Mem) => OpTarget::MemTile,
            Some(TileKind::Compute) => OpTarget::Compute,
            Some(TileKind::ShimNoc | TileKind::ShimPl) | None => {
                OpTarget::Shim { push: Self::queue_channel(col, row, offset, device).is_some() }
            }
        }
    }

    /// The tile kind and absolute DMA channel whose task queue `offset` is,
    /// or `None` when the write is not a task-queue push.
    fn queue_channel(col: u8, row: u8, offset: u32, device: &DeviceState) -> Option<(TileKind, u8)> {
        let tile_kind = device.tile(col as usize, row as usize).map(|t| t.tile_kind)?;
        let dma = device.array.dma_engine(col, row)?;
        let s2mm_channels = dma.s2mm_channel_count() as u8;
        let mm2s_channels = dma.mm2s_channel_count() as u8;

        let reg_layout = crate::device::regdb::device_reg_layout();

        // Identify if this offset is a start queue write
        let (abs_channel, _is_mm2s) = match tile_kind {
            TileKind::Compute => {
                let base = reg_layout.memory_channel_base;
                let stride = reg_layout.memory_channel_stride;
                match Self::channel_from_queue_write(
                    offset,
                    base,
                    stride,
                    s2mm_channels,
                    s2mm_channels + mm2s_channels,
                ) {
                    Some(r) => r,
                    None => return None,
                }
            }
            TileKind::Mem => {
                let stride = reg_layout.memtile_channel_stride;
                let s2mm_base = reg_layout.memtile_channel_s2mm_base;
                let mm2s_base = reg_layout.memtile_channel_mm2s_base;
                if let Some((ch, _)) =
                    Self::channel_from_queue_write(offset, s2mm_base, stride, s2mm_channels, s2mm_channels)
                {
                    (ch, false)
                } else if let Some((ch, _)) =
                    Self::channel_from_queue_write(offset, mm2s_base, stride, mm2s_channels, mm2s_channels)
                {
                    (s2mm_channels + ch, true)
                } else {
                    return None;
                }
            }
            TileKind::ShimNoc | TileKind::ShimPl => {
                let base = reg_layout.shim_channel_base;
                let stride = reg_layout.shim_channel_stride;
                match Self::channel_from_queue_write(
                    offset,
                    base,
                    stride,
                    s2mm_channels,
                    s2mm_channels + mm2s_channels,
                ) {
                    Some(r) => r,
                    None => return None,
                }
            }
        };

        Some((tile_kind, abs_channel))
    }

    /// Check if a register offset is a DMA Task_Queue write within a channel block.
    ///
    /// Returns `(absolute_channel_index, is_mm2s)` if the offset matches a queue
    /// register. Queue registers are at base + ch*stride + 4 within each block.
    fn channel_from_queue_write(
        offset: u32,
        base: u32,
        stride: u32,
        s2mm_count: u8,
        total: u8,
    ) -> Option<(u8, bool)> {
        if offset < base || stride == 0 {
            return None;
        }
        let rel = offset - base;
        if rel % stride != 4 {
            return None; // Not a Task_Queue register (Ctrl is at +0, Queue at +4)
        }
        let ch = (rel / stride) as u8;
        if ch >= total {
            return None;
        }
        let is_mm2s = ch >= s2mm_count;
        Some((ch, is_mm2s))
    }

    /// Get the count of executed instructions.
    pub fn executed_count(&self) -> usize {
        self.executed_count
    }
}

impl Default for NpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a register offset falls in the tile's data memory range.
///
/// The AIE tile address space has data memory at the lowest offsets:
/// - Compute tiles: 0x00000-0x0FFFF (64KB)
/// - Memory tiles: 0x00000-0x7FFFF (512KB)
/// - Shim tiles: no data memory
///
/// Writes to these offsets must go through tile.write_data() rather
/// than the register bus to actually reach data memory.
fn is_data_memory_offset(tile: &crate::device::tile::Tile, offset: u32) -> bool {
    let dm_size = tile.data_memory().len() as u32;
    dm_size > 0 && offset < dm_size
}

/// Check if an offset falls in the shim DMA BD register region.
///
/// Returns the BD index if the offset is within the BD region, None otherwise.
/// BD region bounds are derived from the AM025 register database.
/// Check if a BlockWrite targets a DMA BD region in any tile type.
///
/// Returns `(bd_index, tile_kind)` if the offset falls within a BD register
/// range. Row determines tile type: 0=shim, 1=memtile, >=2=compute.
fn bd_index_for_blockwrite(row: u8, offset: u32) -> Option<(u8, TileKind)> {
    let layout = crate::device::regdb::device_reg_layout();

    match row {
        0 => {
            // Shim BD region
            if offset >= layout.shim_bd_base && offset < layout.shim_channel_base {
                let idx = (offset - layout.shim_bd_base) / layout.shim_bd_stride;
                Some((idx as u8, TileKind::ShimNoc))
            } else {
                None
            }
        }
        1 => {
            // MemTile BD region
            if offset >= layout.memtile_bd_base {
                let rel = offset - layout.memtile_bd_base;
                if rel < layout.memtile_bd_stride * 48 {
                    let idx = rel / layout.memtile_bd_stride;
                    Some((idx as u8, TileKind::Mem))
                } else {
                    None
                }
            } else {
                None
            }
        }
        _ => {
            // Compute tile BD region (memory module)
            if offset >= layout.memory_bd_base {
                let rel = offset - layout.memory_bd_base;
                if rel < layout.memory_bd_stride * 16 {
                    let idx = rel / layout.memory_bd_stride;
                    Some((idx as u8, TileKind::Compute))
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
}

/// Decode an NPU address into (col, row, offset).
///
/// Decode a runtime_sequence register address into a *physical* tile
/// address by extracting the logical column from the encoded bits and
/// shifting it by the partition's `start_col`.
///
/// NPU1 address format:
/// - bits[31:25]: Column (7 bits)
/// - bits[24:20]: Row (5 bits)
/// - bits[19:0]: Register offset within tile (20 bits)
///
/// Runtime_sequence ops emit logical column indices (0 = leftmost partition
/// column).  Real HW resolves these against the partition's start_col chosen
/// by the driver allocator.  Mirroring that here keeps EMU in lockstep with
/// HW: the same logical address lands on the same physical tile.
fn decode_npu_address(addr: u32, start_col: u8) -> (u8, u8, u32) {
    use xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_ROW_SHIFT, TILE_OFFSET_MASK};
    let logical_col = ((addr >> TILE_COL_SHIFT) & 0x7F) as u8;
    let row = ((addr >> TILE_ROW_SHIFT) & 0x1F) as u8;
    let offset = addr & TILE_OFFSET_MASK;
    let physical_col = logical_col.saturating_add(start_col);
    (physical_col, row, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_npu_address() {
        // Test logical shim tile address (col 0, row 0) with no shift.
        let (col, row, offset) = decode_npu_address(0x0001D000, 0);
        assert_eq!(col, 0);
        assert_eq!(row, 0);
        assert_eq!(offset, 0x1D000);
    }

    #[test]
    fn test_decode_npu_address_with_start_col() {
        // Same logical address, partition relocated to physical col 1.
        let (col, row, offset) = decode_npu_address(0x0001D000, 1);
        assert_eq!(col, 1, "logical 0 + start_col 1 -> physical 1");
        assert_eq!(row, 0);
        assert_eq!(offset, 0x1D000);

        // Logical col 2 in the address bits + start_col 1 -> physical 3.
        use xdna_archspec::aie2::TILE_COL_SHIFT;
        let addr_logical_col2 = (2u32 << TILE_COL_SHIFT) | 0x0001D000;
        let (col, _, _) = decode_npu_address(addr_logical_col2, 1);
        assert_eq!(col, 3, "logical 2 + start_col 1 -> physical 3");
    }

    #[test]
    fn test_executor_new() {
        let executor = NpuExecutor::new();
        assert_eq!(executor.executed_count(), 0);
    }

    #[test]
    fn test_advance_result_variants() {
        let results: Vec<AdvanceResult> = vec![
            AdvanceResult::Progressed,
            AdvanceResult::Blocked,
            AdvanceResult::Done,
            AdvanceResult::Idle,
            AdvanceResult::Error("test".to_string()),
        ];
        for r in &results {
            match r {
                AdvanceResult::Progressed => {}
                AdvanceResult::Blocked => {}
                AdvanceResult::Done => {}
                AdvanceResult::Idle => {}
                AdvanceResult::Error(_) => {}
            }
        }
    }

    #[test]
    fn test_executor_initial_state_is_idle() {
        let executor = NpuExecutor::new();
        assert!(matches!(executor.state(), ExecutorState::Idle));
    }

    #[test]
    fn test_try_advance_idle_returns_idle() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();
        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Idle);
    }

    #[test]
    fn test_try_advance_empty_stream_returns_done() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        executor.load_instructions(Vec::new());
        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
        // Subsequent calls also return Done
        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
    }

    /// The NPU MaskWrite reads the register before it writes. A register no one has
    /// written must read its AM025 reset value, so clearing one field leaves the rest.
    #[test]
    fn maskwrite_of_one_clock_control_bit_starts_from_the_reset_value() {
        for (row, offset, reset) in [(2u32, 0x60000u32, 0x37u32), (1, 0xFFF00, 0x33), (0, 0xFFF00, 0x3B)] {
            let mut executor = NpuExecutor::new();
            let mut device = DeviceState::new_npu1();
            let mut host_mem = HostMemory::new();
            device.set_start_col(1);
            let reg_off = (row << 20) | offset;
            executor.load_instructions(vec![NpuInstruction::MaskWrite { reg_off, value: 0, mask: 1 << 4 }]);
            assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
            assert_eq!(device.read_tile_register(1, row as u8, offset), reset & !(1 << 4), "row {row}");
            assert!(device.array.clock().is_adaptive_dma_enabled(1, row as u8), "row {row}");
        }
    }

    /// Base address (48-bit byte address) of shim BD `bd` in physical column `col`.
    fn shim_bd_base(device: &mut DeviceState, col: u8, bd: u32) -> u64 {
        let lo = device.read_tile_register(col, 0, 0x1D004 + 0x20 * bd) as u64;
        let hi = (device.read_tile_register(col, 0, 0x1D008 + 0x20 * bd) & 0xFFFF) as u64;
        (hi << 32) | lo
    }

    fn run_patches(host_buffers: Vec<HostBuffer>, patches: &[(u32, u8, u32)]) -> (NpuExecutor, DeviceState) {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();
        device.set_start_col(1);
        executor.set_host_buffers(host_buffers);
        executor.load_instructions(
            patches
                .iter()
                .map(|&(reg_addr, arg_idx, arg_plus)| NpuInstruction::DdrPatch {
                    reg_addr,
                    arg_idx,
                    arg_plus,
                })
                .collect(),
        );
        // Each call is one executor cycle; an op's firmware cost holds the next one back.
        for _ in 0..100_000 {
            if executor.try_advance(&mut device, &mut host_mem) == AdvanceResult::Done {
                return (executor, device);
            }
        }
        panic!("the patches did not complete");
    }

    /// Firmware adds DDR_AIE_ADDR_OFFSET to the first five host arguments of a DDR patch
    /// (mlir-aie TxnEncoding.h, kNumFirmwareTranslatedArgs); a later argument is patched raw and
    /// carries the offset in its arg_plus, as the compiler folds it there.
    #[test]
    fn a_ddr_patch_adds_the_ddr_offset_for_the_first_five_arguments_only() {
        use crate::device::dma::DDR_AIE_ADDR_OFFSET;
        let bufs: Vec<HostBuffer> = (0..6)
            .map(|i| HostBuffer { address: 0x8000_4000_0000 + 0x10_0000 * i, size: 4096 })
            .collect();
        let (_, mut device) = run_patches(bufs, &[(0x1D004, 4, 0x40), (0x1D024, 5, 0x40)]);
        assert_eq!(shim_bd_base(&mut device, 1, 0), 0x8000_4040_0040 + DDR_AIE_ADDR_OFFSET);
        assert_eq!(shim_bd_base(&mut device, 1, 1), 0x8000_4050_0040);
    }

    /// An argument the host never bound has slot value 0: the patched base is the offset plus
    /// arg_plus, host address 0 upward (silicon faults there), and no buffer is invented.
    #[test]
    fn a_ddr_patch_of_an_unbound_argument_points_at_host_address_zero() {
        use crate::device::dma::DDR_AIE_ADDR_OFFSET;
        let bufs = vec![HostBuffer { address: 0x8000_4000_0000, size: 4096 }; 2];
        let (executor, mut device) = run_patches(bufs, &[(0x1D1E4, 3, 0x10)]);
        assert_eq!(shim_bd_base(&mut device, 1, 15), DDR_AIE_ADDR_OFFSET + 0x10);
        assert_eq!(executor.host_buffers().len(), 2, "no buffer invented for the unbound argument");
    }

    #[test]
    fn test_try_advance_processes_write32() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        // Write to logical compute-tile column 0, row 2, offset 0.
        let addr = (0u32 << 25) | (2u32 << 20) | 0x0;
        executor.load_instructions(vec![NpuInstruction::Write32 { reg_off: addr, value: 0x42 }]);

        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
        // After executing the only instruction, state should be Done
        assert!(matches!(executor.state(), ExecutorState::Done));
    }

    /// With start_col=1, a Write32 to logical col 0 must land on physical col 1.
    /// This is the core invariant for partition-aware addressing: runtime_sequence
    /// ops emit logical column 0 (= leftmost partition column), and the executor
    /// must shift them into the physical column the driver allocated.
    #[test]
    fn test_write32_respects_start_col() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        device.set_start_col(1);

        // Write to logical (col 0, row 2) -- should land on physical (1, 2).
        let logical_addr = (0u32 << 25) | (2u32 << 20) | 0x0;
        let value = 0xdeadbeef;
        executor.load_instructions(vec![NpuInstruction::Write32 { reg_off: logical_addr, value }]);

        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);

        // Compute-tile offset 0 lands in data memory (write_data_u32).
        // Physical (1, 2) should hold the value; the reserved DPU tile at
        // physical (0, 2) must remain untouched.
        let phys_target = device.tile_mut(1, 2).expect("physical (1,2) tile exists").read_data_u32(0);
        assert_eq!(phys_target, Some(value), "logical col 0 + start_col 1 must reach physical col 1",);
        let dpu_tile = device.tile_mut(0, 2).expect("reserved DPU tile exists").read_data_u32(0);
        assert_eq!(dpu_tile, Some(0), "partition-relative write leaked into the reserved DPU tile");
    }

    /// Sync's column field is also logical and must be shifted by start_col,
    /// otherwise the executor waits on the wrong DMA engine and stalls forever.
    #[test]
    fn test_sync_respects_start_col() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        device.set_start_col(1);

        // Sync on logical col 0 -- should record physical col 1 in pending_syncs.
        executor.load_instructions(vec![NpuInstruction::Sync {
            channel: 0,
            column: 0,
            direction: 1, // MM2S
            column_num: 1,
            row: 2,
            row_num: 1,
        }]);

        // First try_advance issues the sync; subsequent advances poll until satisfied.
        // We don't care about completion here -- just that the recorded column is shifted.
        let _ = executor.try_advance(&mut device, &mut host_mem);

        assert_eq!(executor.pending_syncs.len(), 1);
        assert_eq!(
            executor.pending_syncs[0].column, 1,
            "Sync's logical col 0 + start_col 1 must record physical col 1",
        );
    }

    /// Each loaded runtime sequence waits only on its own syncs: a runlist's
    /// next command must not inherit the previous command's.
    #[test]
    fn load_starts_without_the_previous_sequences_syncs() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();
        executor.load_instructions(vec![NpuInstruction::Sync {
            channel: 0,
            column: 0,
            direction: 0,
            column_num: 1,
            row: 0,
            row_num: 1,
        }]);
        let _ = executor.try_advance(&mut device, &mut host_mem);
        assert_eq!(executor.pending_syncs().len(), 1);

        executor.load_instructions(vec![]);
        assert!(executor.pending_syncs().is_empty());
    }

    /// Verify that sync completion requires the channel to have been running
    /// before it returns to idle, preventing false-positive on initial idle.
    #[test]
    fn test_sync_requires_channel_started() {
        use crate::device::dma::BdConfig;
        use crate::device::host_memory::HostMemory;
        use crate::device::DeviceState;
        let reg_layout = crate::device::regdb::device_reg_layout();

        let mut device = DeviceState::new_npu1();
        device.array.clock_mut().ungate_all();
        let mut host_mem = HostMemory::new();
        let mut executor = NpuExecutor::new();

        // Use a compute tile (row 2) which has proper DMA channels (4 ch, 16 BDs).
        // Shim tiles (row 0) currently have 0 channels in the ArchConfig.
        let test_col: u8 = 1;
        let test_row: u8 = 2;

        // Add a sync on physical compute tile col 1, row 2, MM2S channel 0.
        // direction=1 (MM2S) -> absolute channel = 2.
        executor.pending_syncs.push(PendingSync {
            column: test_col,
            row: test_row,
            channel: 0,
            direction: 1, // MM2S
            started: false,
            satisfied: false,
        });

        // Before any DMA activity, the channel is idle but the sync should
        // NOT be satisfied (channel was never running).
        assert!(!executor.syncs_satisfied(&device), "sync must not be satisfied on initial idle");

        // Verify Channel_Running bit is 0 initially.
        let abs_ch = 2u8; // MM2S ch0 = abs channel 2
        let dma = device.array.dma_engine(test_col, test_row).unwrap();
        let status = dma.get_channel_status(abs_ch);
        assert!(
            !reg_layout.memory_status.channel_running.extract_bool(status),
            "Channel_Running should be 0 initially"
        );

        // Write source data into tile data memory so the MM2S transfer has
        // something to read (address 0x100, 64 bytes).
        let tile = device.tile_mut(test_col as usize, test_row as usize).unwrap();
        for i in 0..64usize {
            tile.data_memory_mut()[0x100 + i] = i as u8;
        }

        // Configure BD 0 for a 64-byte MM2S transfer from local address 0x100.
        let bd = BdConfig::simple_1d(0x100, 64);
        let dma = device.array.dma_engine_mut(test_col, test_row).unwrap();
        dma.configure_bd(0, bd).unwrap();
        dma.start_channel(abs_ch, 0).unwrap();

        // Channel should now be running.
        let status = device.array.dma_engine(test_col, test_row).unwrap().get_channel_status(abs_ch);
        assert!(
            reg_layout.memory_status.channel_running.extract_bool(status),
            "Channel_Running should be 1 after start"
        );

        // Sync still not satisfied (channel is running).
        assert!(!executor.syncs_satisfied(&device), "sync must not be satisfied while channel is running");

        // Step DMA until the channel completes. Drain stream_out each cycle
        // to simulate a downstream consumer; without it the MM2S backpressure
        // cap stalls the channel because step_dma alone doesn't run array
        // routing.
        for _ in 0..10_000 {
            device.array.step_dma(test_col, test_row, &mut host_mem);
            let dma = device.array.dma_engine_mut(test_col, test_row).unwrap();
            while dma.pop_stream_out().is_some() {}
            let status = device.array.dma_engine(test_col, test_row).unwrap().get_channel_status(abs_ch);
            if !reg_layout.memory_status.channel_running.extract_bool(status) {
                break;
            }
        }

        // Channel should be done.
        let status = device.array.dma_engine(test_col, test_row).unwrap().get_channel_status(abs_ch);
        assert!(
            !reg_layout.memory_status.channel_running.extract_bool(status),
            "Channel_Running should be 0 after completion"
        );

        // NOW the sync should be satisfied (was running, now idle).
        assert!(
            executor.syncs_satisfied(&device),
            "sync should be satisfied after channel ran and completed"
        );
    }

    /// Phase 2 (TCT): a sync resolves on a task-completion TOKEN, and each sync
    /// consumes exactly its own token -- two WAIT_TCTS on one channel need two
    /// tokens. This is the faithful hardware path (vs. the Channel_Running
    /// fallback exercised by `test_sync_requires_channel_started`).
    #[test]
    fn test_sync_consumes_matching_token_once() {
        use crate::device::host_memory::HostMemory;
        use crate::device::DeviceState;

        let mut device = DeviceState::new_npu1();
        device.array.clock_mut().ungate_all();
        let mut host_mem = HostMemory::new();
        let mut executor = NpuExecutor::new();
        executor.instructions = Vec::new();

        // Compute tile col 1, row 2; MM2S ch0 -> absolute channel 2. The channel
        // is never started, so the fallback (running->idle) can never fire --
        // only a token can satisfy these syncs.
        let (col, row, abs) = (1u8, 2u8, 2u8);
        for _ in 0..2 {
            executor.pending_syncs.push(PendingSync {
                column: col,
                row,
                channel: 0,
                direction: 1, // MM2S
                started: false,
                satisfied: false,
            });
        }

        // No token yet, channel never ran -> unsatisfied.
        assert!(!executor.syncs_satisfied(&device), "no token + never ran -> unsatisfied");

        // One token for abs channel 2. Only the first sync can resolve.
        device.array.dma_engine_mut(col, row).unwrap().issue_task_token(abs, 0);

        // Drive sync 0 through the consuming BlockedOnSync transition.
        executor.state = ExecutorState::BlockedOnSync { next_index: 0, sync_index: 0, issued_at: 0 };
        let _ = executor.try_advance(&mut device, &mut host_mem);
        assert!(executor.pending_syncs[0].satisfied, "sync 0 satisfied by the token");
        assert!(
            !device.array.dma_engine(col, row).unwrap().has_task_token_for_channel(abs),
            "sync 0 consumed the only token"
        );

        // Sync 1 has no token of its own -> still blocked (consume-once).
        executor.state = ExecutorState::BlockedOnSync { next_index: 0, sync_index: 1, issued_at: 0 };
        let _ = executor.try_advance(&mut device, &mut host_mem);
        assert!(
            !executor.pending_syncs[1].satisfied,
            "sync 1 needs its own token; the first sync's token must not double-count"
        );

        // Its own token -> sync 1 resolves.
        device.array.dma_engine_mut(col, row).unwrap().issue_task_token(abs, 0);
        let _ = executor.try_advance(&mut device, &mut host_mem);
        assert!(executor.pending_syncs[1].satisfied, "sync 1 satisfied by its own token");
    }

    // -----------------------------------------------------------------------
    // MaskPoll graceful-termination tests (spec §4.2, debug_halt_probe)
    //
    // These cover the three cases for the MASKPOLL halt-synchronization
    // instruction injected into insts.bin by inject-maskpoll.py:
    //   (a) poll satisfied immediately -> proceeds to next instruction
    //   (b) poll satisfied after N cycles -> proceeds
    //   (c) poll never satisfied + engine quiescent -> is_blocked_on_poll()
    //       detects the condition so the run loop can exit MaskPollUnsatisfied
    // -----------------------------------------------------------------------

    /// (a) MaskPoll satisfied immediately: the condition is already met when
    /// the instruction first executes.  The executor advances past it without
    /// ever entering BlockedOnPoll.
    #[test]
    fn test_maskpoll_satisfied_immediately() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        // Use the tile register map directly to preset a register value so
        // the poll is satisfied on the first try.  Lock0_value (0x1F000) on
        // compute tile (1, 2) is a safe non-side-effecting register to poke.
        // write_tile_register routes through the unified register bus.
        let lock0_reg = 0x1F000u32;
        device.write_tile_register(1, 2, lock0_reg, 0x0000_0001);

        // Encode the MaskPoll NPU address: col=0, row=2, offset=0x1F000.
        let reg_off = (0u32 << 25) | (2u32 << 20) | lock0_reg;

        // Poll: (reg & 0x01) == 0x01 -- satisfied immediately.
        executor.load_instructions(vec![NpuInstruction::MaskPoll { reg_off, value: 0x01, mask: 0x01 }]);

        let result = executor.try_advance(&mut device, &mut host_mem);
        // With a one-instruction stream, after the poll succeeds we expect Done.
        assert!(
            matches!(result, AdvanceResult::Done | AdvanceResult::Progressed),
            "immediately-satisfied MaskPoll should not block: got {:?}",
            result
        );
        assert!(
            !executor.is_blocked_on_poll(),
            "executor must not be blocked on poll after immediate satisfaction"
        );
    }

    /// (b) MaskPoll satisfied after N cycles: the condition is not met
    /// initially, but becomes met after the polled register is updated.
    /// The executor blocks (returns Blocked) until satisfied, then proceeds.
    #[test]
    fn test_maskpoll_satisfied_after_n_cycles() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        // Use Lock0_value on physical tile (1, 2) as the polled register.
        let lock0_reg = 0x1F000u32;
        let reg_off = (0u32 << 25) | (2u32 << 20) | lock0_reg;

        // Initially the register is 0; poll waits for bit 0 to be set.
        executor.load_instructions(vec![NpuInstruction::MaskPoll { reg_off, value: 0x01, mask: 0x01 }]);

        // First advance: condition not met -> Blocked, enters BlockedOnPoll.
        let r1 = executor.try_advance(&mut device, &mut host_mem);
        assert_eq!(r1, AdvanceResult::Blocked, "unmet poll must return Blocked");
        assert!(executor.is_blocked_on_poll(), "must be in BlockedOnPoll state");

        // Simulate N cycles of Blocked returns while register is still 0.
        for _ in 0..10 {
            let r = executor.try_advance(&mut device, &mut host_mem);
            assert_eq!(r, AdvanceResult::Blocked);
            assert!(executor.is_blocked_on_poll());
        }

        // Now satisfy the condition by writing the register via the device bus.
        device.write_tile_register(1, 2, lock0_reg, 0x01);

        // Next advance: condition now met -> Done (single-instruction stream).
        let r_done = executor.try_advance(&mut device, &mut host_mem);
        assert!(
            matches!(r_done, AdvanceResult::Done | AdvanceResult::Progressed),
            "poll must resolve after register is set: got {:?}",
            r_done
        );
        assert!(!executor.is_blocked_on_poll(), "must not be blocked on poll after condition is met");
    }

    /// (c) MaskPoll never satisfied + engine quiescent: the polled register
    /// stays 0 forever (as when the emulator cannot set DEBUG_HALT because
    /// control-packet writes to core/debug registers are dropped).
    /// The executor remains in BlockedOnPoll indefinitely; is_blocked_on_poll()
    /// stays true so the run loop can detect quiescence and exit cleanly.
    /// Also verifies:
    ///  - the polled register is never modified (no fakery)
    ///  - no subsequent instruction is issued (no OP_READ skip)
    #[test]
    fn test_maskpoll_unsatisfied_blocks_indefinitely() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        // Poll for bit 16 of Core_Status (0x32004) on logical tile (0, 2):
        // This is exactly the DEBUG_HALT condition used by the probe injector.
        // Here the device is fresh: no breakpoint is armed and no halt-seam has
        // fired, so the core is not debug-halted and Core_Status[16] stays 0 --
        // a genuinely unsatisfiable poll (this is the retained-hardening path,
        // NOT the debug_halt_probe's path, which now satisfies post-Unit-1b).
        let core_status_reg = 0x32004u32;
        let reg_off = (0u32 << 25) | (2u32 << 20) | core_status_reg;

        // A two-instruction stream: MaskPoll then a Write32 (the "OP_READ push"
        // stand-in).  The Write32 must never be issued if the poll blocks.
        let sentinel_reg = 0x1F000u32; // Lock0_value -- starts at 0, never written if poll blocks
        let sentinel_addr = (0u32 << 25) | (2u32 << 20) | sentinel_reg;
        executor.load_instructions(vec![
            NpuInstruction::MaskPoll { reg_off, value: 0x0001_0000, mask: 0x0001_0000 },
            // Sentinel instruction -- must NOT execute while poll is unsatisfied.
            NpuInstruction::Write32 { reg_off: sentinel_addr, value: 0xDEAD_BEEF },
        ]);

        // Spin a large number of cycles -- poll must never satisfy.
        for i in 0..1000 {
            let r = executor.try_advance(&mut device, &mut host_mem);
            assert_eq!(
                r,
                AdvanceResult::Blocked,
                "unsatisfied poll must return Blocked forever (cycle {})",
                i
            );
            assert!(executor.is_blocked_on_poll(), "must remain in BlockedOnPoll (cycle {})", i);
        }

        // DEBUG_HALT must not be faked to force poll satisfaction (no fakery).
        // Note: Core_Status may carry other live-computed bits (e.g. RESET on a
        // fresh core); the contract is specifically that bit 16 stays 0.
        let core_status_val = device.tile_mut(1, 2).map(|t| t.read_register(core_status_reg)).unwrap_or(0);
        assert_eq!(
            core_status_val & 0x0001_0000,
            0,
            "Core_Status DEBUG_HALT bit must not be faked by the emulator"
        );

        // Sentinel register must be untouched (sentinel Write32 must not have
        // executed while the poll is unsatisfied -- no OP_READ skip).
        let sentinel_val = device.tile_mut(1, 2).map(|t| t.read_register(sentinel_reg)).unwrap_or(0);
        assert_eq!(sentinel_val, 0, "sentinel Write32 must not execute while poll is unsatisfied");
    }

    // -----------------------------------------------------------------
    // Start_Queue pushes into a full task queue (silicon: shim-queue probes)
    // -----------------------------------------------------------------

    /// A shim S2MM 0 on (1,0) whose first task parks on lock 5 (value 0), and a runtime sequence
    /// that pushes `pushes` tasks and then opens the gate.
    fn queue_push_sequence(pushes: u32) -> (NpuExecutor, DeviceState, HostMemory) {
        use crate::device::BdConfig;
        let mut device = DeviceState::new_npu1();
        device.array.clock_mut().ungate_all();
        let mut host_mem = HostMemory::new();
        host_mem.allocate_region("bds".to_string(), 0x100, 0x1000).unwrap(); // a shim reads its BDs from here
        let dma = device.array.dma_engine_mut(1, 0).unwrap();
        for bd in 0..pushes as u8 {
            dma.configure_bd(bd, BdConfig::simple_1d(0x100 + 0x40 * bd as u64, 8).with_acquire(5, -1))
                .unwrap();
        }
        let reg = |offset: u32| offset; // column 0 of the NPU address is physical column 1
        let mut stream = vec![];
        for bd in 0..pushes {
            stream.push(NpuInstruction::Write32 { reg_off: reg(shim_task_queue_offset(0)), value: bd });
        }
        let lock5 = 0x14000 + 5 * 0x10; // shim Lock<n>_value
        stream.push(NpuInstruction::Write32 { reg_off: reg(lock5), value: 20 });
        let mut executor = NpuExecutor::new();
        executor.load_instructions(stream);
        (executor, device, host_mem)
    }

    fn run_until_settled(
        executor: &mut NpuExecutor,
        device: &mut DeviceState,
        host_mem: &mut HostMemory,
    ) -> AdvanceResult {
        let mut last = AdvanceResult::Progressed;
        // the controller paces Start_Queue dispatches, so give it plenty of cycles
        for _ in 0..200_000 {
            device.array.step_data_movement_with_denied(host_mem, &[]);
            last = executor.try_advance(device, host_mem);
            if matches!(last, AdvanceResult::Done | AdvanceResult::Idle)
                || executor.blocked_write_description(device).is_some()
            {
                break;
            }
        }
        last
    }

    #[test]
    fn seven_pushes_into_a_lock_parked_shim_channel_all_go_through_and_the_gate_opens() {
        // dma_rel_c6 queues six; the staging and the 4-deep queue hold seven.
        for pushes in [6, 7] {
            let (mut executor, mut device, mut host_mem) = queue_push_sequence(pushes);
            let last = run_until_settled(&mut executor, &mut device, &mut host_mem);
            assert!(matches!(last, AdvanceResult::Done | AdvanceResult::Idle), "{pushes} pushes: {last:?}");
            assert!(executor.blocked_write_description(&device).is_none());
            // the gate write ran behind the pushes: the head takes its lock and waits for data
            for _ in 0..5000 {
                device.array.step_data_movement_with_denied(&mut host_mem, &[]);
            }
            assert_eq!(
                device.tile(1, 0).unwrap().locks[5].value,
                19,
                "{pushes} pushes: the head acquired the gate"
            );
        }
    }

    #[test]
    fn an_eighth_push_blocks_the_instruction_stream_and_the_diagnosis_says_what_silicon_does() {
        let (mut executor, mut device, mut host_mem) = queue_push_sequence(8);
        let last = run_until_settled(&mut executor, &mut device, &mut host_mem);
        assert_eq!(last, AdvanceResult::Blocked, "the push into a full queue stalls the writer");
        let text = executor.blocked_write_description(&device).expect("blocked on a queue write");
        for needle in ["Start_Queue", "(1,0)", "S2MM ch0", "deadlock", "blocks the writer"] {
            assert!(text.contains(needle), "{needle:?} missing from: {text}");
        }
        // the gate write sits behind the blocked push, so nothing can drain the queue
        for _ in 0..5000 {
            device.array.step_data_movement_with_denied(&mut host_mem, &[]);
        }
        assert_eq!(device.array.dma_engine(1, 0).unwrap().pending_tasks(0), 6);
    }

    #[test]
    fn a_blocked_push_resumes_once_the_channel_drains() {
        let (mut executor, mut device, mut host_mem) = queue_push_sequence(8);
        let last = run_until_settled(&mut executor, &mut device, &mut host_mem);
        assert_eq!(last, AdvanceResult::Blocked);
        // something else (a core, a control packet) opens the gate: the head takes it, the
        // queue moves up, and the blocked write goes through
        device.tile_mut(1, 0).unwrap().locks[5].set(20);
        let last = run_until_settled(&mut executor, &mut device, &mut host_mem);
        assert!(matches!(last, AdvanceResult::Done | AdvanceResult::Idle), "{last:?}");
        assert!(executor.blocked_write_description(&device).is_none());
    }

    /// Build a shim DMA Task_Queue register offset for `channel` (0-based
    /// in the DMA engine's combined channel index).
    ///
    /// Shim channel registers are laid out as `Ctrl @ +0`, `Queue @ +4`
    /// within each per-channel block of size `shim_channel_stride`.
    fn shim_task_queue_offset(channel: u8) -> u32 {
        let layout = crate::device::regdb::device_reg_layout();
        layout.shim_channel_base + (channel as u32) * layout.shim_channel_stride + 4
    }

    #[test]
    fn load_instructions_resets_the_controller_clock() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();
        executor.try_advance(&mut device, &mut host_mem);
        assert_eq!(executor.npu_cycle, 1, "one array cycle per advance call");
        executor.tokens_seen = Some(7);
        executor.load_instructions(Vec::new());
        assert_eq!(executor.npu_cycle, 0);
        assert_eq!(executor.tokens_seen, None, "a new sequence starts with nothing to service");
    }

    // -----------------------------------------------------------------
    // Runtime-sequence pacing: every operation retires at its class cost on
    // the controller's clock, serially, with no gate on task pushes.
    // Silicon numbers are array cycles (shim timer) at the default clocks.
    // -----------------------------------------------------------------

    /// A shim register with no side effect on the transfer path (lock 5's value).
    const PACING_SCRATCH: u32 = 0x14050;
    /// Shim BD 0 (8 words) and its address word, the target of a DDR patch.
    const PACING_SHIM_BD: u32 = 0x1D000;
    /// Memtile BD 0 (8 words), logical column 0.
    const PACING_MEMTILE_BD: u32 = (1 << 20) | 0xA0000;

    fn pacing_device() -> (DeviceState, HostMemory) {
        let mut device = DeviceState::new_npu1();
        device.set_start_col(1);
        device.array.clock_mut().ungate_all();
        (device, HostMemory::new())
    }

    /// Run `stream` to completion (stepping data movement each cycle) and return
    /// the executor cycle at which each instruction issued.
    fn issue_cycles(
        executor: &mut NpuExecutor,
        device: &mut DeviceState,
        host_mem: &mut HostMemory,
    ) -> Vec<u64> {
        let mut issued = Vec::new();
        let mut last_count = executor.executed_count();
        for _ in 0..2_000_000 {
            device.array.step_data_movement_with_denied(host_mem, &[]);
            let r = executor.try_advance(device, host_mem);
            if executor.executed_count() > last_count {
                last_count = executor.executed_count();
                issued.push(executor.npu_cycle);
            }
            if matches!(r, AdvanceResult::Done | AdvanceResult::Idle) {
                return issued;
            }
            assert!(!matches!(r, AdvanceResult::Error(_)), "{r:?}");
        }
        panic!("stream did not finish");
    }

    fn mean_gap(issued: &[u64], from: usize, to: usize) -> f64 {
        (issued[to] - issued[from]) as f64 / (to - from) as f64
    }

    fn scratch_write() -> NpuInstruction {
        NpuInstruction::Write32 { reg_off: PACING_SCRATCH, value: 0 }
    }

    /// Mean cost, in array cycles, of `n` copies of `op` issued back to back.
    fn mean_cost_of(op: NpuInstruction, n: usize) -> f64 {
        let (mut device, mut host_mem) = pacing_device();
        let mut executor = NpuExecutor::new();
        executor.set_host_buffers(vec![HostBuffer { address: 0x8000_4000_0000, size: 4096 }]);
        let mut stream = vec![op; n];
        stream.push(scratch_write());
        executor.load_instructions(stream);
        let issued = issue_cycles(&mut executor, &mut device, &mut host_mem);
        mean_gap(&issued, 0, n)
    }

    /// Compute tile (logical column 0, row 2): a data-memory word and BD 0.
    const PACING_COMPUTE_DM: u32 = (2 << 20) | 0x400;
    const PACING_COMPUTE_BD: u32 = (2 << 20) | 0x1D000;

    /// Cost in array cycles of each op of `stream` (issue to next issue); the
    /// stream gets a trailing write so its last op retires too.
    fn op_costs(mut stream: Vec<NpuInstruction>) -> Vec<f64> {
        let (mut device, mut host_mem) = pacing_device();
        let mut executor = NpuExecutor::new();
        executor.set_host_buffers(vec![HostBuffer { address: 0x8000_4000_0000, size: 4096 }]);
        let n = stream.len();
        stream.push(scratch_write());
        executor.load_instructions(stream);
        let issued = issue_cycles(&mut executor, &mut device, &mut host_mem);
        (0..n).map(|i| (issued[i + 1] - issued[i]) as f64).collect()
    }

    /// A register write to a shim register: 102.7 array cycles each on silicon
    /// (runs of single-word writes to a shim register, 36 runs, slope over the
    /// write count).
    #[test]
    fn a_shim_register_write_retires_at_the_measured_shim_write_cost() {
        let c = mean_cost_of(scratch_write(), 200);
        assert!((c - 102.7).abs() < 0.5, "shim write32 mean cost {c}");
    }

    /// A register write to tile memory: 100.47 array cycles each on silicon
    /// (runs of 4..64 single-word writes, slope +-0.04).
    #[test]
    fn a_tile_write_retires_at_the_measured_tile_write_cost() {
        let c = mean_cost_of(NpuInstruction::Write32 { reg_off: PACING_COMPUTE_DM, value: 0 }, 200);
        assert!((c - 100.47).abs() < 0.7, "tile write32 mean cost {c}");
    }

    /// A task-queue push: 104.85 array cycles (free push gaps of the bridge
    /// corpus and the runtime-sequence probes; not yet compared with the
    /// other write classes by a dedicated probe).
    #[test]
    fn a_task_queue_push_retires_at_the_push_cost() {
        let push = NpuInstruction::Write32 { reg_off: shim_task_queue_offset(0), value: 0 };
        let c = op_costs(vec![push])[0];
        assert!((c - 104.85).abs() <= 1.0, "push cost {c}");
    }

    /// The masked write a shim push carries (the channel's controller id):
    /// 231 array cycles on silicon (push gaps with and without it, two probes).
    #[test]
    fn a_shim_masked_write_retires_at_the_measured_cost() {
        let c = mean_cost_of(NpuInstruction::MaskWrite { reg_off: PACING_SCRATCH, value: 0, mask: 1 }, 200);
        assert!((c - 231.0).abs() < 0.7, "shim maskwrite32 mean cost {c}");
    }

    /// Masked writes to tile registers (dedicated probe, one and four in a
    /// row): each costs 216.9 array cycles on a memtile and 225.9 on a compute
    /// tile, and the first of a run about 32 more (251.3 / 256.0).
    #[test]
    fn tile_masked_writes_cost_the_measured_tile_level_and_the_first_of_a_run_more() {
        for (name, reg_off, marginal, first) in
            [("memtile", PACING_MEMTILE_BD, 216.9, 251.3), ("compute tile", PACING_COMPUTE_BD, 225.9, 256.0)]
        {
            let mw = NpuInstruction::MaskWrite { reg_off, value: 0, mask: 1 };
            let costs = op_costs(vec![scratch_write(), mw.clone(), mw.clone(), mw.clone(), mw]);
            assert!((costs[1] - first).abs() <= 3.0, "{name}: first of a run {}", costs[1]);
            for (i, c) in costs.iter().enumerate().skip(2) {
                // One op measured alone: whole cycles, so up to one more than the mean.
                assert!((c - marginal).abs() <= 1.5, "{name}: masked write {i} {c}");
            }
        }
    }

    /// A tile block write grows with its word count (dedicated probes, warm
    /// command, any tile and register or memory target alike): exactly linear
    /// from 8 words on, 13.98 array cycles per word, 290 at 8 words (the line
    /// through N = 8..64 gives 179.4 + 13.98 N).
    #[test]
    fn a_tile_block_write_costs_its_fixed_part_plus_the_measured_cost_per_word() {
        let bw = |reg_off: u32, n: usize| NpuInstruction::BlockWrite { reg_off, values: vec![0; n] };
        let c8 = mean_cost_of(bw(PACING_MEMTILE_BD, 8), 50);
        assert!((c8 - 291.2).abs() <= 1.5, "8 words {c8}");
        let c64 = mean_cost_of(bw(PACING_COMPUTE_DM, 64), 50);
        let per_word = (c64 - c8) / 56.0;
        assert!((per_word - 13.98).abs() <= 0.05, "per word {per_word}");
        let ct6 = mean_cost_of(bw(PACING_COMPUTE_BD, 6), 50);
        let mt6 = mean_cost_of(bw(PACING_MEMTILE_BD, 6), 50);
        assert!((ct6 - mt6).abs() <= 1.0, "tile-independent: compute {ct6}, memtile {mt6}");
    }

    /// A shim BD block write and its address patch: 584.15 array cycles together on
    /// silicon (block write + patch + push lead 689, less the push's own write).
    #[test]
    fn a_shim_bd_write_with_its_patch_retires_at_the_measured_pair_cost() {
        let (mut device, mut host_mem) = pacing_device();
        let mut executor = NpuExecutor::new();
        executor.set_host_buffers(vec![HostBuffer { address: 0x8000_4000_0000, size: 4096 }]);
        let n = 50;
        let mut stream = Vec::new();
        for _ in 0..n {
            stream.push(NpuInstruction::BlockWrite { reg_off: PACING_SHIM_BD, values: vec![0; 8] });
            stream.push(NpuInstruction::DdrPatch { reg_addr: PACING_SHIM_BD + 4, arg_idx: 0, arg_plus: 0 });
        }
        stream.push(scratch_write());
        executor.load_instructions(stream);
        let issued = issue_cycles(&mut executor, &mut device, &mut host_mem);
        let pair = (issued[2 * n] - issued[0]) as f64 / n as f64;
        assert!((pair - 584.15).abs() < 0.5, "shim BD write + patch mean cost {pair}");
    }

    /// A mask-poll satisfied at once: about 215 array cycles (bridge corpus fit).
    #[test]
    fn a_satisfied_mask_poll_retires_at_the_fitted_poll_cost() {
        let c = mean_cost_of(NpuInstruction::MaskPoll { reg_off: PACING_SCRATCH, value: 0, mask: 0 }, 100);
        assert!((c - 215.0).abs() < 1.0, "satisfied mask-poll mean cost {c}");
    }

    /// The gap between two pushes is the cost of the operations between them,
    /// linear in their number (silicon: k = 0 / 4 / 16 writes, no plateau).
    #[test]
    fn the_gap_between_two_pushes_is_linear_in_the_writes_between_them() {
        let tq0 = shim_task_queue_offset(0);
        let tq2 = shim_task_queue_offset(2);
        let mut gaps = Vec::new();
        for k in [0usize, 4, 16, 64] {
            let (mut device, mut host_mem) = pacing_device();
            let mut executor = NpuExecutor::new();
            let mut stream = vec![NpuInstruction::Write32 { reg_off: tq0, value: 0 }];
            stream.extend(std::iter::repeat_n(scratch_write(), k));
            stream.push(NpuInstruction::Write32 { reg_off: tq2, value: 1 });
            stream.push(scratch_write());
            executor.load_instructions(stream);
            let issued = issue_cycles(&mut executor, &mut device, &mut host_mem);
            gaps.push((k, (issued[k + 1] - issued[0]) as f64));
        }
        // Slope per write inside the silicon probes' band, no plateau: every gap
        // sits on the line through the first and the last.
        let (k0, g0) = gaps[0];
        let (kn, gn) = gaps[gaps.len() - 1];
        let slope = (gn - g0) / (kn - k0) as f64;
        assert!((102.5..=106.0).contains(&slope), "cost per write {slope}");
        for &(k, gap) in &gaps {
            let line = g0 + (k - k0) as f64 * slope;
            assert!((gap - line).abs() <= 1.0, "k={k}: push gap {gap}, line {line}");
        }
    }

    /// A second task pushed on the same channel does not wait for the first: each
    /// push lands one write cost after the previous one, while the head task is
    /// still parked on its lock (the lock is released only by the last write).
    #[test]
    fn a_second_push_on_a_busy_channel_does_not_wait_for_the_first_task() {
        let (mut executor, mut device, mut host_mem) = queue_push_sequence(3);
        let issued = issue_cycles(&mut executor, &mut device, &mut host_mem);
        for i in 1..3 {
            let gap = (issued[i] - issued[i - 1]) as f64;
            assert!((gap - 104.85).abs() <= 1.0, "push {i} landed {gap} cycles after push {}", i - 1);
        }
    }

    /// Issue a token on shim (1,0) S2MM 0 at executor cycle `at` while the stream
    /// runs, and return the issue cycles.
    fn issue_cycles_with_token_at(stream: Vec<NpuInstruction>, at: u64) -> Vec<u64> {
        issue_cycles_with_tokens_at(stream, &[at])
    }

    /// As `issue_cycles_with_token_at`, one token at each cycle of `at`.
    fn issue_cycles_with_tokens_at(stream: Vec<NpuInstruction>, at: &[u64]) -> Vec<u64> {
        let (mut device, mut host_mem) = pacing_device();
        let mut executor = NpuExecutor::new();
        executor.load_instructions(stream);
        let mut issued = Vec::new();
        let mut last_count = 0;
        for _ in 0..2_000_000 {
            for _ in at.iter().filter(|&&t| t == executor.npu_cycle) {
                device.array.dma_engine_mut(1, 0).unwrap().issue_task_token(0, 0);
            }
            let r = executor.try_advance(&mut device, &mut host_mem);
            if executor.executed_count() > last_count {
                last_count = executor.executed_count();
                issued.push(executor.npu_cycle);
            }
            if matches!(r, AdvanceResult::Done) {
                return issued;
            }
        }
        panic!("stream did not finish");
    }

    /// A task-complete token that arrives while the controller executes operations
    /// costs about 2300 array cycles once, at the next operation boundary (silicon:
    /// FINISHED to the next push).
    #[test]
    fn a_token_arriving_during_execution_is_serviced_once_at_the_next_boundary() {
        let stream: Vec<NpuInstruction> = (0..12).map(|_| scratch_write()).collect();
        let plain = issue_cycles_with_token_at(stream.clone(), u64::MAX);
        let with = issue_cycles_with_token_at(stream, plain[3] + 10);
        // Ops issued before the token are unaffected.
        assert_eq!(&with[..4], &plain[..4]);
        // The next boundary carries the service, and only it.
        let delay = (with[4] - plain[4]) as f64;
        assert!((delay - 2300.0).abs() <= 1.0, "token service {delay}");
        for i in 5..12 {
            // The carried fraction may move one cycle between ops; never more.
            let (w, p) = (with[i] - with[i - 1], plain[i] - plain[i - 1]);
            assert!(w.abs_diff(p) <= 1, "op {i}: serviced once ({w} against {p})");
        }
    }

    /// A token that arrives while the controller waits in a sync is not serviced
    /// separately: the sync's own cost covers it.
    #[test]
    fn a_token_arriving_during_a_sync_wait_costs_nothing_extra() {
        // Wait on (1,0) S2MM 0; its token arrives at cycle 500. A second token, for
        // S2MM 1, arrives during the same wait.
        let sync =
            NpuInstruction::Sync { channel: 0, column: 0, direction: 0, column_num: 1, row: 0, row_num: 1 };
        let stream = vec![scratch_write(), sync, scratch_write(), scratch_write(), scratch_write()];
        let run = |extra: bool| {
            let (mut device, mut host_mem) = pacing_device();
            let mut executor = NpuExecutor::new();
            executor.load_instructions(stream.clone());
            let mut issued = Vec::new();
            let mut last_count = 0;
            for _ in 0..200_000 {
                if executor.npu_cycle == 400 && extra {
                    device.array.dma_engine_mut(1, 0).unwrap().issue_task_token(1, 0);
                }
                if executor.npu_cycle == 500 {
                    device.array.dma_engine_mut(1, 0).unwrap().issue_task_token(0, 0);
                }
                let r = executor.try_advance(&mut device, &mut host_mem);
                if executor.executed_count() > last_count {
                    last_count = executor.executed_count();
                    issued.push(executor.npu_cycle);
                }
                if matches!(r, AdvanceResult::Done) {
                    return issued;
                }
            }
            panic!("stream did not finish");
        };
        assert_eq!(run(true), run(false), "a token absorbed by the wait adds no service time");
    }

    /// Run `stream`, issuing a token on shim (1,0) S2MM 0 `x` cycles after the
    /// k-th sync issues (k = 0, 1, ...), and return the issue cycles.
    fn issue_cycles_with_late_tokens(stream: Vec<NpuInstruction>, x: u64) -> Vec<u64> {
        let (mut device, mut host_mem) = pacing_device();
        let mut executor = NpuExecutor::new();
        executor.load_instructions(stream.clone());
        let mut issued: Vec<u64> = Vec::new();
        let mut due: Vec<u64> = Vec::new();
        let mut last_count = 0;
        for _ in 0..2_000_000 {
            if due.first() == Some(&executor.npu_cycle) {
                due.remove(0);
                device.array.dma_engine_mut(1, 0).unwrap().issue_task_token(0, 0);
            }
            let r = executor.try_advance(&mut device, &mut host_mem);
            if executor.executed_count() > last_count {
                last_count = executor.executed_count();
                issued.push(executor.npu_cycle);
                if matches!(stream[last_count - 1], NpuInstruction::Sync { .. }) {
                    due.push(executor.npu_cycle + x);
                }
            }
            if matches!(r, AdvanceResult::Done) {
                return issued;
            }
        }
        panic!("stream did not finish");
    }

    fn shim_s2mm0_sync() -> NpuInstruction {
        NpuInstruction::Sync { channel: 0, column: 0, direction: 0, column_num: 1, row: 0, row_num: 1 }
    }

    /// A sync whose token arrives 3000 cycles after its issue resumes the
    /// controller on the cold plateau, about 9540 cycles after the issue
    /// (silicon: bridge corpus single-sync tails, dedicated probes).
    #[test]
    fn a_sync_after_a_late_token_costs_the_measured_plateau() {
        let stream = vec![scratch_write(), shim_s2mm0_sync(), scratch_write(), scratch_write()];
        let issued = issue_cycles_with_late_tokens(stream, 3000);
        let resume = (issued[2] - issued[1]) as f64;
        assert!((resume - 9540.0).abs() <= 3.0, "sync issue to next op {resume}");
    }

    /// Every wait gets the first-wait structure: no one-time charge on the
    /// first sync, nothing cheaper on later ones.
    #[test]
    fn every_sync_costs_the_same_level_for_the_same_lateness() {
        let stream =
            vec![scratch_write(), shim_s2mm0_sync(), scratch_write(), shim_s2mm0_sync(), scratch_write()];
        let issued = issue_cycles_with_late_tokens(stream, 3000);
        assert_eq!(issued[2] - issued[1], issued[4] - issued[3], "first and second sync");
    }

    /// A token that arrived before the sync's issue is serviced once at the
    /// operation boundary and the sync then takes its fast path: about 3000
    /// cycles in total, the silicon level for a sync on a finished task.
    #[test]
    fn a_sync_on_an_arrived_token_charges_the_token_once() {
        let stream = vec![scratch_write(), shim_s2mm0_sync(), scratch_write(), scratch_write()];
        let plain: Vec<NpuInstruction> = vec![scratch_write(); 4];
        let base = issue_cycles_with_token_at(plain, u64::MAX);
        let issued = issue_cycles_with_token_at(stream, base[0] + 10);
        // Write, token service at the boundary, then the sync's fast path.
        let total = (issued[2] - issued[0]) as f64;
        assert!((total - (104.5 + 3000.0)).abs() <= 3.0, "write + token + sync {total}");
        let sync_alone = (issued[2] - issued[1]) as f64;
        assert!((sync_alone - 700.0).abs() <= 3.0, "the sync's own fast path {sync_alone}");
    }
}
