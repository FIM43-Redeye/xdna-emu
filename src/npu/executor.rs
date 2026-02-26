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
use crate::device::tile::TileType;

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
        }
    }

    /// Get warnings collected during execution.
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Get the current executor state (for testing/debugging).
    pub(crate) fn state(&self) -> &ExecutorState {
        &self.state
    }

    /// Get pending sync conditions.
    pub fn pending_syncs(&self) -> &[PendingSync] {
        &self.pending_syncs
    }

    /// Check if all sync conditions are satisfied.
    ///
    /// Polls the DMA status register's `Channel_Running` bit (bit 19),
    /// matching the hardware polling loop in `_XAieMl_DmaWaitForDone()`.
    /// A sync is only satisfied when the channel has been observed running
    /// at least once AND has since returned to not-running. This avoids
    /// the false-positive where initial idle is mistaken for completion.
    pub fn syncs_satisfied(&mut self, device: &DeviceState) -> bool {
        let reg_layout = crate::device::regdb::device_reg_layout();

        if self.pending_syncs.is_empty() {
            return true;
        }

        for sync in &mut self.pending_syncs {
            // Map (channel, direction) to absolute channel index, matching
            // the encoding in check_dma_trigger: S2MM ch0=0, ch1=1;
            // MM2S ch0=2, ch1=3.
            let abs_channel = if sync.direction == 1 {
                2 + sync.channel
            } else {
                sync.channel
            };

            if let Some(dma) = device.array.dma_engine(sync.column, sync.row) {
                let status = dma.get_channel_status(abs_channel);
                let status_layout = if dma.tile_type.is_mem_tile() {
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

                // Channel is not running. Only count as "done" if it was
                // previously observed running.
                if !sync.started {
                    return false;
                }
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
                // If so, fix up the next_index and return Blocked.
                if let ExecutorState::BlockedOnQueue { next_index: ref mut ni, .. } = self.state {
                    *ni = next_index + 1;
                    return AdvanceResult::Blocked;
                }

                // Normal progression
                let new_index = next_index + 1;
                if new_index >= self.instructions.len() {
                    self.state = ExecutorState::Done;
                    AdvanceResult::Done
                } else {
                    self.state = ExecutorState::Executing { next_index: new_index };
                    AdvanceResult::Progressed
                }
            }

            ExecutorState::BlockedOnQueue {
                next_index, col, row, channel, bd_id, repeat, enable_token, ..
            } => {
                use crate::device::dma::MAX_TASK_QUEUE_DEPTH;

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
                    AdvanceResult::Blocked
                }
            }
        }
    }

    /// Execute all instructions in a stream against the device.
    pub fn execute(&mut self, stream: &NpuInstructionStream, device: &mut DeviceState, host_memory: &mut HostMemory) -> Result<(), String> {
        for instr in stream.instructions() {
            self.execute_instruction(instr, device, host_memory)?;
            self.executed_count += 1;
        }
        Ok(())
    }

    /// Execute a single instruction.
    fn execute_instruction(&mut self, instr: &NpuInstruction, device: &mut DeviceState, host_memory: &mut HostMemory) -> Result<(), String> {
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
                // MaskPoll is a synchronization point - we simulate it by
                // assuming the condition is already met (the DMA has completed).
                // In a cycle-accurate simulation, we'd need to actually poll.
                log::debug!(
                    "NPU MaskPoll: reg=0x{:08X} value=0x{:08X} mask=0x{:08X} (assuming satisfied)",
                    reg_off, value, mask
                );
                Ok(())
            }

            NpuInstruction::DdrPatch { reg_addr, arg_idx, arg_plus } => {
                self.execute_ddr_patch(*reg_addr, *arg_idx, *arg_plus, device)
            }

            NpuInstruction::Sync { channel, column, row, direction, .. } => {
                log::debug!(
                    "NPU Sync: channel={} column={} row={} direction={}",
                    channel, column, row, direction
                );
                // Record this sync condition - the run is complete when all syncs are satisfied
                self.pending_syncs.push(PendingSync {
                    column: *column,
                    row: *row,
                    channel: *channel,
                    direction: *direction,
                    started: false,
                });
                Ok(())
            }

            NpuInstruction::Unknown { opcode, data } => {
                log::warn!(
                    "NPU Unknown instruction: opcode=0x{:02X} data_len={}",
                    opcode,
                    data.len()
                );
                Ok(())
            }
        }
    }

    /// Execute a Write32 instruction.
    fn execute_write32(&mut self, reg_off: u32, value: u32, device: &mut DeviceState, host_memory: &mut HostMemory) -> Result<(), String> {
        log::debug!("NPU Write32: reg=0x{:08X} value=0x{:08X}", reg_off, value);

        // Decode tile address from register offset
        // NPU1 address format: bits[31:25]=col, bits[24:20]=row, bits[19:0]=offset
        let (col, row, offset) = decode_npu_address(reg_off);

        if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            // Route to data memory or register space based on offset
            if is_data_memory_offset(tile, offset) {
                log::debug!("NPU Write32 -> data memory: tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    col, row, offset, value);
                tile.write_data_u32(offset as usize, value);
            } else {
                tile.write_register(offset, value);
                // Check if this is a DMA channel control register (triggers DMA start)
                self.check_dma_trigger(col, row, offset, value, device, host_memory);
            }
        } else {
            log::warn!(
                "NPU Write32 to non-existent tile ({}, {}): offset=0x{:05X}",
                col, row, offset
            );
        }

        Ok(())
    }

    /// Execute a BlockWrite instruction.
    fn execute_blockwrite(&mut self, reg_off: u32, values: &[u32], device: &mut DeviceState, host_memory: &mut HostMemory) -> Result<(), String> {
        log::debug!(
            "NPU BlockWrite: reg=0x{:08X} count={}",
            reg_off,
            values.len()
        );

        let (col, row, base_offset) = decode_npu_address(reg_off);

        if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            // Route to data memory or register space based on offset
            if is_data_memory_offset(tile, base_offset) {
                log::debug!("NPU BlockWrite -> data memory: tile({},{}) offset=0x{:05X} count={}",
                    col, row, base_offset, values.len());
                // Write all values as a contiguous byte block to data memory
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                tile.write_data(base_offset as usize, &bytes);
            } else {
                for (i, &value) in values.iter().enumerate() {
                    let offset = base_offset + (i as u32) * 4;
                    tile.write_register(offset, value);
                }

                // Check for DMA triggers in the written range
                for (i, &value) in values.iter().enumerate() {
                    let offset = base_offset + (i as u32) * 4;
                    self.check_dma_trigger(col, row, offset, value, device, host_memory);
                }
            }
        }

        // If writing to a DMA BD region, sync BD data to the DMA engine.
        // Read the FULL BD back from tile registers rather than using the
        // blockwrite values directly -- the compiler may write fewer than 8
        // words when trailing words are zero, and partial writes compose
        // correctly when read back from register space.
        if let Some((bd_index, tile_type)) = bd_index_for_blockwrite(row, base_offset) {
            Self::sync_bd_from_registers(col, row, bd_index, tile_type, device);
        }

        Ok(())
    }

    /// Read a full BD from tile registers and sync it to the DMA engine.
    ///
    /// Uses `BufferDescriptor::from_registers()` to parse ALL BD fields
    /// (multi-dimensional addressing, BD chaining, locks, packet mode, iteration,
    /// AXI parameters) from the register words, then converts to the runtime
    /// BdConfig. Reads from tile register space so partial writes compose
    /// correctly.
    fn sync_bd_from_registers(
        col: u8, row: u8, bd_index: u8, tile_type: TileType, device: &mut DeviceState,
    ) {
        use crate::device::dma::bd::BufferDescriptor;

        let layout = crate::device::regdb::device_reg_layout();
        let (bd_base, bd_stride) = match tile_type {
            TileType::Shim => (layout.shim_bd_base, layout.shim_bd_stride),
            TileType::MemTile => (layout.memtile_bd_base, layout.memtile_bd_stride),
            TileType::Compute => (layout.memory_bd_base, layout.memory_bd_stride),
        };
        let bd_words = (bd_stride / 4) as usize;

        // Read full BD from tile register space
        let values = if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            let start = bd_base + bd_index as u32 * bd_stride;
            (0..bd_words).map(|i| tile.read_register(start + i as u32 * 4)).collect::<Vec<_>>()
        } else {
            return;
        };

        log::debug!("{:?} BD {} (col={},row={}) raw: {:08X?}", tile_type, bd_index, col, row, values);

        let bd = BufferDescriptor::from_registers(&values, tile_type);
        let config = bd.to_bd_config();

        if let Some(dma) = device.array.dma_engine_mut(col, row) {
            if let Err(e) = dma.configure_bd(bd_index, config.clone()) {
                log::warn!("Failed to configure {:?} BD {}: {:?}", tile_type, bd_index, e);
            } else {
                log::debug!("Configured {:?} BD {} addr=0x{:X} len={} d0=[{},{}] d1=[{},{}] next={:?} acq={:?} rel={:?}",
                    tile_type, bd_index, config.base_addr, config.length,
                    config.d0.size, config.d0.stride, config.d1.size, config.d1.stride,
                    config.next_bd,
                    config.acquire_lock.map(|id| (id, config.acquire_value)),
                    config.release_lock.map(|id| (id, config.release_value)));
            }
        }
    }

    /// Execute a MaskWrite instruction.
    fn execute_maskwrite(&mut self, reg_off: u32, value: u32, mask: u32, device: &mut DeviceState, host_memory: &mut HostMemory) -> Result<(), String> {
        log::debug!(
            "NPU MaskWrite: reg=0x{:08X} value=0x{:08X} mask=0x{:08X}",
            reg_off, value, mask
        );

        let (col, row, offset) = decode_npu_address(reg_off);

        if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            if is_data_memory_offset(tile, offset) {
                // Read-modify-write in data memory
                let current = tile.read_data_u32(offset as usize).unwrap_or(0);
                let new_value = (current & !mask) | (value & mask);
                log::debug!("NPU MaskWrite -> data memory: tile({},{}) offset=0x{:05X} \
                    current=0x{:08X} -> 0x{:08X}", col, row, offset, current, new_value);
                tile.write_data_u32(offset as usize, new_value);
            } else {
                // Read-modify-write in register space
                let current = tile.read_register(offset);
                let new_value = (current & !mask) | (value & mask);
                tile.write_register(offset, new_value);
                self.check_dma_trigger(col, row, offset, new_value, device, host_memory);
            }
        }

        Ok(())
    }

    /// Execute a DDR patch instruction.
    fn execute_ddr_patch(&mut self, reg_addr: u32, arg_idx: u8, arg_plus: u32, device: &mut DeviceState) -> Result<(), String> {
        // Extend host_buffers with trace-sized entries if the xclbin
        // references arg indices beyond what the test harness allocated.
        // This happens with trace-injected xclbins: trace injection adds an
        // extra DDR buffer for HW packet trace collection. The emulator's
        // trace units produce real binary trace packets that flow through
        // the stream switch and shim DMA into this buffer, matching the
        // format expected by mlir-aie's parse.py.
        while self.host_buffers.len() <= arg_idx as usize {
            let trace_addr = self.host_buffers.last()
                .map(|b| b.address + b.size as u64)
                .unwrap_or(0x10_0000);
            // 1 MB trace buffer, matching the default trace_size
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
            self.host_buffers.push(HostBuffer {
                address: trace_addr,
                size: trace_size,
            });
        }

        let buffer = &self.host_buffers[arg_idx as usize];

        let patched_addr = buffer.address + arg_plus as u64;

        log::debug!(
            "NPU DdrPatch: reg=0x{:08X} arg_idx={} arg_plus={} -> addr=0x{:016X}",
            reg_addr, arg_idx, arg_plus, patched_addr
        );

        // Write the patched address to the register
        // For 64-bit addresses, we write the low 32 bits
        let (col, row, offset) = decode_npu_address(reg_addr);

        if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            tile.write_register(offset, patched_addr as u32);

            // Write high 32 bits of the patched address to the next word.
            // For shim DMA BDs, the next word (word 2) shares its 32 bits between
            // Base_Address_High[15:0] and other fields (Enable_Packet, Packet_ID,
            // Packet_Type, OoO_BD_ID). A blind write would clobber those fields.
            // Use read-modify-write to preserve the non-address bits.
            let high_bits = (patched_addr >> 32) as u32;
            let is_shim_bd = row == 0 && bd_index_for_blockwrite(row, offset).is_some();
            if is_shim_bd {
                // Shim BD: Base_Address_High occupies bits 15:0 of word 2
                let current = tile.read_register(offset + 4);
                let merged = (current & 0xFFFF_0000) | (high_bits & 0x0000_FFFF);
                tile.write_register(offset + 4, merged);
            } else {
                tile.write_register(offset + 4, high_bits);
            }
        }

        // Re-sync the BD to DMA engine if patching BD address field.
        // DDR patches always target word 1 (address low) within a BD, so
        // we only re-sync when the patched offset falls on that word.
        if let Some((bd_index, tile_type)) = bd_index_for_blockwrite(row, offset) {
            let layout = crate::device::regdb::device_reg_layout();
            let (bd_base_addr, bd_stride) = match tile_type {
                TileType::Shim => (layout.shim_bd_base, layout.shim_bd_stride),
                TileType::MemTile => (layout.memtile_bd_base, layout.memtile_bd_stride),
                TileType::Compute => (layout.memory_bd_base, layout.memory_bd_stride),
            };
            let bd_rel = offset - bd_base_addr;
            let word_in_bd = (bd_rel % bd_stride) / 4;

            if word_in_bd == 1 {
                Self::sync_bd_from_registers(col, row, bd_index, tile_type, device);
            }
        }

        Ok(())
    }

    /// Check if a register write triggers DMA operation.
    fn check_dma_trigger(&mut self, col: u8, row: u8, offset: u32, value: u32, device: &mut DeviceState, host_memory: &mut HostMemory) {
        // DMA Task_Queue registers trigger DMA channel start when written.
        // Each channel has Ctrl at +0 and Task_Queue at +4 within its stride.
        // Register layouts differ by tile type but all share the same format:
        //   Bits 3:0  = BD_ID (buffer descriptor index)
        //   Bits 23:16 = Repeat_Count (additional BD executions)
        use crate::device::dma::{
            COMPUTE_S2MM_CHANNELS, COMPUTE_MM2S_CHANNELS,
            MEM_TILE_S2MM_CHANNELS, MEM_TILE_MM2S_CHANNELS,
        };

        let reg_layout = crate::device::regdb::device_reg_layout();
        let tile_type = device.tile(col as usize, row as usize)
            .map(|t| t.tile_type);
        let tile_type = match tile_type {
            Some(tt) => tt,
            None => return,
        };

        // Determine channel base(s), stride, and S2MM/MM2S channel counts
        // for the tile type. Compute and shim have a single contiguous block
        // (S2MM channels first, then MM2S). MemTile has separate S2MM and
        // MM2S base addresses because it has 6 channels of each type.
        let (abs_channel, is_mm2s) = match tile_type {
            TileType::Shim => {
                let base = reg_layout.shim_channel_base;
                let stride = reg_layout.shim_channel_stride;
                let s2mm = COMPUTE_S2MM_CHANNELS as u8;
                match Self::channel_from_queue_write(offset, base, stride, s2mm, s2mm + COMPUTE_MM2S_CHANNELS as u8) {
                    Some(r) => r,
                    None => return,
                }
            }
            TileType::Compute => {
                let base = reg_layout.memory_channel_base;
                let stride = reg_layout.memory_channel_stride;
                let s2mm = COMPUTE_S2MM_CHANNELS as u8;
                match Self::channel_from_queue_write(offset, base, stride, s2mm, s2mm + COMPUTE_MM2S_CHANNELS as u8) {
                    Some(r) => r,
                    None => return,
                }
            }
            TileType::MemTile => {
                // MemTile has separate S2MM and MM2S register blocks
                let stride = reg_layout.memtile_channel_stride;
                let s2mm_base = reg_layout.memtile_channel_s2mm_base;
                let mm2s_base = reg_layout.memtile_channel_mm2s_base;
                let s2mm_count = MEM_TILE_S2MM_CHANNELS as u8;
                let mm2s_count = MEM_TILE_MM2S_CHANNELS as u8;

                // Try S2MM block first
                if let Some((ch, _)) = Self::channel_from_queue_write(offset, s2mm_base, stride, s2mm_count, s2mm_count) {
                    (ch, false)
                } else if let Some((ch_raw, _)) = Self::channel_from_queue_write(offset, mm2s_base, stride, mm2s_count, mm2s_count) {
                    // MM2S channels are numbered after S2MM in the DMA engine
                    (s2mm_count + ch_raw, true)
                } else {
                    return;
                }
            }
        };

        let bd_index = (value & 0xF) as u8;
        let repeat_count = ((value >> 16) & 0xFF) as u8;

        log::info!(
            "NPU DMA start: col={} row={} {:?} {} ch={} bd={} repeat={}",
            col, row, tile_type,
            if is_mm2s { "MM2S" } else { "S2MM" }, abs_channel, bd_index, repeat_count
        );

        // Try to enqueue. If queue is full, transition to BlockedOnQueue
        // and let the caller handle draining (engine stepping in interleaved
        // mode, or bounded DMA-only stepping in batch execute() mode).
        // This matches real hardware where the host firmware blocks on
        // Start_Queue writes when the queue is full (aie-rt
        // _XAieMl_DmaWaitForBdTaskQueue polls Task_Queue_Size before writing).
        use crate::device::dma::MAX_TASK_QUEUE_DEPTH;

        let queue_full = device.array.dma_engine(col, row)
            .map_or(false, |dma| dma.task_queue_size(abs_channel) >= MAX_TASK_QUEUE_DEPTH);

        if queue_full {
            log::debug!(
                "DMA tile({},{}) ch{} queue full, deferring BD {} enqueue",
                col, row, abs_channel, bd_index
            );
            self.state = ExecutorState::BlockedOnQueue {
                instr_index: self.executed_count,
                next_index: self.executed_count + 1,
                col, row,
                channel: abs_channel,
                bd_id: bd_index,
                repeat: repeat_count,
                enable_token: (value >> 31) & 1 != 0,
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
    }

    /// Check if a register offset is a DMA Task_Queue write within a channel block.
    ///
    /// Returns `(absolute_channel_index, is_mm2s)` if the offset matches a queue
    /// register. Queue registers are at base + ch*stride + 4 within each block.
    fn channel_from_queue_write(
        offset: u32, base: u32, stride: u32, s2mm_count: u8, total: u8,
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
/// than tile.write_register() to actually reach data memory.
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
/// Returns `(bd_index, tile_type)` if the offset falls within a BD register
/// range. Row determines tile type: 0=shim, 1=memtile, >=2=compute.
fn bd_index_for_blockwrite(row: u8, offset: u32) -> Option<(u8, TileType)> {
    let layout = crate::device::regdb::device_reg_layout();

    match row {
        0 => {
            // Shim BD region
            if offset >= layout.shim_bd_base && offset < layout.shim_channel_base {
                let idx = (offset - layout.shim_bd_base) / layout.shim_bd_stride;
                Some((idx as u8, TileType::Shim))
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
                    Some((idx as u8, TileType::MemTile))
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
                    Some((idx as u8, TileType::Compute))
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
/// NPU1 address format:
/// - bits[31:25]: Column (7 bits, but typically 0-4 for NPU1)
/// - bits[24:20]: Row (5 bits)
/// - bits[19:0]: Register offset within tile (20 bits)
fn decode_npu_address(addr: u32) -> (u8, u8, u32) {
    // NPU1 uses a different encoding than what I described
    // Looking at actual addresses like 0x1D000 for shim DMA:
    // The high bits encode tile location, low bits are register offset
    //
    // For shim tiles (row 0), addresses are in the 0x1xxxx range
    // For compute tiles, addresses are higher
    //
    // Let me use a simpler approach: assume shim tile for known ranges

    // Shim DMA registers are typically at offset 0x1D000-0x1DFFF
    // This is for column 0, row 0

    // For now, decode based on known patterns:
    // 0x0001xxxx = col 0, row 0 (shim)
    // 0x0002xxxx = col 0, row 1 (memtile)
    // etc.

    // Actually the real format from mlir-aie for NPU1:
    // Address = (col << 25) | (row << 20) | offset

    let col = ((addr >> 25) & 0x7F) as u8;
    let row = ((addr >> 20) & 0x1F) as u8;
    let offset = addr & 0xFFFFF;

    (col, row, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_npu_address() {
        // Test shim tile address (col 0, row 0)
        let (col, row, offset) = decode_npu_address(0x0001D000);
        assert_eq!(col, 0);
        assert_eq!(row, 0);
        assert_eq!(offset, 0x1D000);
    }

    #[test]
    fn test_executor_new() {
        let executor = NpuExecutor::new();
        assert_eq!(executor.executed_count(), 0);
    }

    #[test]
    fn test_advance_result_variants() {
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
        executor.load_instructions(vec![
            NpuInstruction::Write32 { reg_off: addr, value: 0x42 },
        ]);

        assert_eq!(executor.try_advance(&mut device, &mut host_mem), AdvanceResult::Done);
        // After executing the only instruction, state should be Done
        assert!(matches!(executor.state(), ExecutorState::Done));
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
        assert!(
            !executor.syncs_satisfied(&device),
            "sync must not be satisfied on initial idle"
        );

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
        let status = device.array.dma_engine(test_col, test_row).unwrap()
            .get_channel_status(abs_ch);
        assert!(
            reg_layout.memory_status.channel_running.extract_bool(status),
            "Channel_Running should be 1 after start"
        );

        // Sync still not satisfied (channel is running).
        assert!(
            !executor.syncs_satisfied(&device),
            "sync must not be satisfied while channel is running"
        );

        // Step DMA until the channel completes.
        for _ in 0..10_000 {
            device.array.step_dma(test_col, test_row, &mut host_mem);
            let status = device.array.dma_engine(test_col, test_row).unwrap()
                .get_channel_status(abs_ch);
            if !reg_layout.memory_status.channel_running.extract_bool(status) {
                break;
            }
        }

        // Channel should be done.
        let status = device.array.dma_engine(test_col, test_row).unwrap()
            .get_channel_status(abs_ch);
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
