//! NPU instruction executor.
//!
//! Executes NPU instructions against the device state, performing
//! register writes, address patches, and DMA triggers.
//!
//! The executor tracks Sync (TCT) instructions that specify completion
//! conditions. A "run" is complete when all Sync conditions are satisfied.

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
    /// Flushing stream switch after sync satisfaction.
    ///
    /// Control packet data may still be in-transit through the stream
    /// switch when the sync completes (sync only confirms the shim DMA
    /// finished sending). On real hardware, the firmware's instruction
    /// pipeline latency gives the stream switch time to deliver data
    /// before follow-up writes take effect. We model this by waiting
    /// a few routing cycles before resuming instruction execution.
    FlushingStreams {
        /// Index of the NEXT instruction to execute after flush.
        next_index: usize,
        /// Remaining routing cycles before resuming.
        remaining: u8,
    },
    /// Charging cycles for a previously-issued instruction to retire.
    ///
    /// AIE2's IPU command processor takes multiple cycles to drain a
    /// control packet's writes through the AXI fabric to the target
    /// tile. While those writes are in flight, the next instruction
    /// can't be issued. EMU models this by parking in this state for
    /// `cycles_remaining` extra cycles after each non-trivial instruction.
    /// See `NpuInstruction::cycle_cost` for the per-variant cost
    /// (Phase 1: all variants return 1, so this state is never entered;
    /// Phase 2 will calibrate against HW timing).
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
#[derive(Debug, Clone)]
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
    /// Cycle-cost model used to compute per-instruction retirement cycles.
    /// Default is `legacy_one_per_packet` which preserves prior behaviour;
    /// callers can opt into `with_known_constants` (or future calibrated
    /// profiles) via `set_cycle_model`.
    cycle_model: super::CycleCostModel,
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
        }
    }

    /// Replace the cycle-cost model used to compute retirement cycles.
    ///
    /// `CycleCostModel::with_known_constants()` engages stream-switch
    /// fabric / PLIO bridge / register-write costs derived from open
    /// sources. Calibrated per-tile-type CMP costs land later via
    /// #322 and a JSON config file mirroring AMD's
    /// `AIE_CONTROL_PATH_LATENCY` schema.
    pub fn set_cycle_model(&mut self, model: super::CycleCostModel) {
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

    /// Get pending sync conditions.
    pub fn pending_syncs(&self) -> &[PendingSync] {
        &self.pending_syncs
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

    /// Check if a single sync condition is satisfied.
    ///
    /// A sync is satisfied when the DMA channel has no remaining work:
    /// it was started, ran its BD chain, and is now idle with nothing
    /// queued. Two detection paths handle timing variations:
    ///
    /// 1. **Normal**: we observe channel_running=true, then later
    ///    channel_running=false. The `started` flag tracks this.
    ///
    /// 2. **Fast completion**: the DMA finished between the task start
    ///    instruction and the sync registration. The channel is already
    ///    idle when we first poll. We detect this by checking that the
    ///    channel has no pending work (no active FSM, no queued tasks)
    ///    AND has completed at least one transfer total.
    fn is_sync_satisfied(sync: &mut PendingSync, device: &DeviceState) -> bool {
        let abs_channel = Self::sync_abs_channel(sync, device);

        if let Some(dma) = device.array.dma_engine(sync.column, sync.row) {
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
                return false;
            }

            // Channel is idle. Check the two satisfaction conditions:

            // 1. We previously observed it running (normal polling path)
            if sync.started {
                return true;
            }

            // 2. Fast completion: channel finished before we first polled.
            //    The channel has no pending work AND has done at least one
            //    transfer (not just initial idle).
            if let Some(stats) = dma.channel_stats(abs_channel) {
                if stats.transfers_completed > 0 && dma.task_queue_size(abs_channel) == 0 {
                    return true;
                }
            }

            false
        } else {
            false
        }
    }

    /// Check if all sync conditions are satisfied.
    pub fn syncs_satisfied(&mut self, device: &DeviceState) -> bool {
        if self.pending_syncs.is_empty() {
            return true;
        }

        for sync in &mut self.pending_syncs {
            if !Self::is_sync_satisfied(sync, device) {
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
                    let msg = format!("NPU instruction {} execution error: {}", next_index, e);
                    log::error!("{}", msg);
                    return AdvanceResult::Error(msg);
                }
                self.executed_count += 1;

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

                // Normal progression. If the instruction's cycle_cost is
                // greater than 1, park in RetiringInstruction so the next
                // issue is delayed by (cost - 1) cycles. With Phase 1's
                // all-1 defaults this branch transitions straight back to
                // Executing; the retire state only engages once Phase 2
                // populates realistic costs.
                let new_index = next_index + 1;
                if new_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    AdvanceResult::Done
                } else {
                    let retire_cycles = self.cycle_model.cost_of(&instr).saturating_sub(1);
                    if retire_cycles > 0 {
                        self.state = ExecutorState::RetiringInstruction {
                            next_index: new_index,
                            cycles_remaining: retire_cycles,
                        };
                    } else {
                        self.state = ExecutorState::Executing { next_index: new_index };
                    }
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
                use crate::device::dma::MAX_TASK_QUEUE_DEPTH;

                // Check if the queue has drained enough for our enqueue
                let has_space = device
                    .array
                    .dma_engine(col, row)
                    .map_or(true, |dma| dma.task_queue_size(channel) < MAX_TASK_QUEUE_DEPTH);

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

            ExecutorState::BlockedOnSync { next_index, sync_index } => {
                // Poll the sync condition each cycle. On real hardware,
                // dma_await_task blocks until the DMA channel's task token
                // returns (or Channel_Running goes idle). We poll the same
                // Channel_Running status.
                let satisfied = if sync_index < self.pending_syncs.len() {
                    Self::is_sync_satisfied(&mut self.pending_syncs[sync_index], device)
                } else {
                    true // Invalid index -- treat as satisfied
                };

                if satisfied {
                    log::info!("NPU Sync #{} satisfied, resuming instruction {}", sync_index, next_index,);
                    if next_index >= self.instructions.len() {
                        self.state = ExecutorState::Done;
                        AdvanceResult::Done
                    } else {
                        // Flush stream switch before resuming. Control packet
                        // data from the just-completed shim DMA may still be
                        // in-transit through the stream switch pipeline. On real
                        // hardware, firmware instruction latency (~4-8 cycles)
                        // gives the switch time to deliver this data. We model
                        // the same delay by waiting before executing the next
                        // instruction.
                        const STREAM_FLUSH_CYCLES: u8 = 4;
                        self.state =
                            ExecutorState::FlushingStreams { next_index, remaining: STREAM_FLUSH_CYCLES };
                        AdvanceResult::Progressed
                    }
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
                        self.state = ExecutorState::Executing { next_index };
                        AdvanceResult::Progressed
                    }
                } else {
                    AdvanceResult::Blocked
                }
            }

            ExecutorState::FlushingStreams { next_index, remaining } => {
                // Wait for stream switch to propagate in-transit data.
                // Each cycle the caller runs the routing, delivering
                // control packet data one hop closer to target tiles.
                if remaining <= 1 {
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
                });
                let sync_index = self.pending_syncs.len() - 1;
                // Transition to BlockedOnSync -- try_advance() will poll this
                // sync each cycle and only advance when it's satisfied.
                // (The caller sets next_index after execute_instruction returns.)
                self.state = ExecutorState::BlockedOnSync {
                    next_index: 0, // placeholder, fixed up by try_advance
                    sync_index,
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

    /// Execute a DDR patch instruction.
    ///
    /// Two modes of operation:
    ///
    /// 1. **Host buffer mode** (non-ELF path): host_buffers is populated by the
    ///    plugin from the ERT regmap.  DdrPatch looks up `host_buffers[arg_idx]`
    ///    and writes `buffer.address + arg_plus` to the BD register.
    ///
    /// 2. **Pre-patched mode** (ELF module path): host_buffers is empty.  XRT
    ///    already patched the block-write payloads with correct BO addresses
    ///    plus the DDR AIE address offset (0x80000000).  The preceding
    ///    block-write has set the register to this pre-patched value.  We read
    ///    the current register, subtract the DDR offset to translate from the
    ///    AIE address space to our emulator address space, and write back.
    ///    This models the hardware's DDR address translation.
    fn execute_ddr_patch(
        &mut self,
        reg_addr: u32,
        arg_idx: u8,
        arg_plus: u32,
        device: &mut DeviceState,
    ) -> Result<(), String> {
        let (col, row, offset) = decode_npu_address(reg_addr, device.start_col);

        if self.host_buffers.is_empty() {
            // ELF module path: addresses are pre-patched by XRT with DDR
            // offset.  Read the current register value (set by the preceding
            // block-write), subtract the DDR AIE address offset, write back.
            return self.execute_ddr_patch_prepatched(col, row, offset, reg_addr, arg_idx, arg_plus, device);
        }

        // Host buffer mode: look up the buffer address by arg_idx.
        //
        // Extend host_buffers with trace-sized entries if the xclbin
        // references arg indices beyond what the test harness allocated.
        // This happens with trace-injected xclbins: trace injection adds an
        // extra DDR buffer for HW packet trace collection.
        while self.host_buffers.len() <= arg_idx as usize {
            let trace_addr = self.host_buffers.last().map(|b| b.address + b.size as u64).unwrap_or(0x10_0000);
            let trace_size = 1_048_576;
            log::info!(
                "DDR patch references arg_idx {} beyond {} known buffers -- \
                 allocating trace buffer at 0x{:X} ({}KB) for binary \
                 trace packet collection.",
                arg_idx,
                self.host_buffers.len(),
                trace_addr,
                trace_size / 1024,
            );
            self.host_buffers.push(HostBuffer { address: trace_addr, size: trace_size });
        }

        let buffer = &self.host_buffers[arg_idx as usize];
        let patched_addr = buffer.address + arg_plus as u64;

        log::debug!(
            "NPU DdrPatch: reg=0x{:08X} arg_idx={} arg_plus={} -> addr=0x{:016X}",
            reg_addr,
            arg_idx,
            arg_plus,
            patched_addr
        );

        self.write_bd_address(col, row, offset, patched_addr, device);
        Ok(())
    }

    /// DDR patch for the ELF module (pre-patched) path.
    ///
    /// XRT's ELF relocation processing patches block-write payloads with
    /// `BO_addr + addend + DDR_AIE_ADDR_OFFSET` (where DDR_AIE_ADDR_OFFSET
    /// = 0x80000000).  The preceding block-write wrote this value to the BD
    /// register.  We read it, subtract the DDR offset to recover the
    /// emulator-space address, and write back.
    ///
    /// This models the AIE's DDR address translation: the NPU sees DDR at a
    /// 2 GB offset from the CPU's view.  On real hardware the memory
    /// controller handles this; in the emulator we do it explicitly.
    fn execute_ddr_patch_prepatched(
        &self,
        col: u8,
        row: u8,
        offset: u32,
        reg_addr: u32,
        arg_idx: u8,
        arg_plus: u32,
        device: &mut DeviceState,
    ) -> Result<(), String> {
        const DDR_AIE_ADDR_OFFSET: u64 = 0x8000_0000;

        // Read the current BD word pair (set by the preceding block-write).
        let word_lo = device
            .tile_mut(col as usize, row as usize)
            .map(|t| t.read_register(offset))
            .unwrap_or(0);
        let word_hi = device
            .tile_mut(col as usize, row as usize)
            .map(|t| t.read_register(offset + 4))
            .unwrap_or(0);

        // Reconstruct the 48-bit address (shim BD format: low 32 in word 1,
        // high 16 in word 2 bits[15:0]).
        let xrt_addr = ((word_hi as u64 & 0xFFFF) << 32) | word_lo as u64;

        // Subtract the DDR AIE address offset to get emulator-space address.
        let emu_addr = xrt_addr.wrapping_sub(DDR_AIE_ADDR_OFFSET);

        log::debug!(
            "NPU DdrPatch (pre-patched): reg=0x{:08X} arg_idx={} arg_plus={} \
             xrt_addr=0x{:012X} -> emu_addr=0x{:012X}",
            reg_addr,
            arg_idx,
            arg_plus,
            xrt_addr,
            emu_addr
        );

        self.write_bd_address(col, row, offset, emu_addr, device);
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
        use crate::device::dma::MAX_TASK_QUEUE_DEPTH;

        let tile_kind = match device.tile(col as usize, row as usize).map(|t| t.tile_kind) {
            Some(tt) => tt,
            None => return false,
        };

        let dma = match device.array.dma_engine(col, row) {
            Some(d) => d,
            None => return false,
        };
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
                    None => return false,
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
                    return false;
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
                    None => return false,
                }
            }
        };

        // Check queue depth
        let queue_full = device
            .array
            .dma_engine(col, row)
            .map_or(false, |dma| dma.task_queue_size(abs_channel) >= MAX_TASK_QUEUE_DEPTH);

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
        // Test shim tile address (col 0, row 0) with no shift.
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

    #[test]
    fn test_try_advance_processes_write32() {
        let mut executor = NpuExecutor::new();
        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();

        // Write to a compute tile register (col 0, row 2, offset 0)
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
        // Physical (1, 2) data mem should hold the value; physical (0, 2) untouched.
        let phys_target = device.tile_mut(1, 2).expect("physical (1,2) tile exists").read_data_u32(0);
        let phys_orig = device.tile_mut(0, 2).expect("physical (0,2) tile exists").read_data_u32(0);
        assert_eq!(phys_target, Some(value), "logical col 0 + start_col 1 must reach physical col 1",);
        assert_eq!(phys_orig, Some(0), "physical col 0 must NOT be written when start_col=1",);
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

    /// Verify that sync completion requires the channel to have been running
    /// before it returns to idle, preventing false-positive on initial idle.
    #[test]
    fn test_sync_requires_channel_started() {
        use crate::device::dma::BdConfig;
        use crate::device::host_memory::HostMemory;
        use crate::device::DeviceState;
        let reg_layout = crate::device::regdb::device_reg_layout();

        let mut device = DeviceState::new_npu1();
        let mut host_mem = HostMemory::new();
        let mut executor = NpuExecutor::new();

        // Use a compute tile (row 2) which has proper DMA channels (4 ch, 16 BDs).
        // Shim tiles (row 0) currently have 0 channels in the ArchConfig.
        let test_col: u8 = 0;
        let test_row: u8 = 2;

        // Add a sync on compute tile col 0, row 2, MM2S channel 0.
        // direction=1 (MM2S) -> absolute channel = 2.
        executor.pending_syncs.push(PendingSync {
            column: test_col,
            row: test_row,
            channel: 0,
            direction: 1, // MM2S
            started: false,
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

        // Step DMA until the channel completes.
        for _ in 0..10_000 {
            device.array.step_dma(test_col, test_row, &mut host_mem);
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
}
