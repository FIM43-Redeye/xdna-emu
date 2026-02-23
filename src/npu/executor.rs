//! NPU instruction executor.
//!
//! Executes NPU instructions against the device state, performing
//! register writes, address patches, and DMA triggers.
//!
//! The executor tracks Sync (TCT) instructions that specify completion
//! conditions. A "run" is complete when all Sync conditions are satisfied.

use super::{NpuInstruction, NpuInstructionStream};
use crate::device::DeviceState;

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
}

impl NpuExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            host_buffers: Vec::new(),
            executed_count: 0,
            pending_syncs: Vec::new(),
        }
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

    /// Execute all instructions in a stream against the device.
    pub fn execute(&mut self, stream: &NpuInstructionStream, device: &mut DeviceState) -> Result<(), String> {
        for instr in stream.instructions() {
            self.execute_instruction(instr, device)?;
            self.executed_count += 1;
        }
        Ok(())
    }

    /// Execute a single instruction.
    fn execute_instruction(&mut self, instr: &NpuInstruction, device: &mut DeviceState) -> Result<(), String> {
        match instr {
            NpuInstruction::Write32 { reg_off, value } => {
                self.execute_write32(*reg_off, *value, device)
            }

            NpuInstruction::BlockWrite { reg_off, values } => {
                self.execute_blockwrite(*reg_off, values, device)
            }

            NpuInstruction::MaskWrite { reg_off, value, mask } => {
                self.execute_maskwrite(*reg_off, *value, *mask, device)
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
    fn execute_write32(&self, reg_off: u32, value: u32, device: &mut DeviceState) -> Result<(), String> {
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
                self.check_dma_trigger(col, row, offset, value, device);
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
    fn execute_blockwrite(&self, reg_off: u32, values: &[u32], device: &mut DeviceState) -> Result<(), String> {
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
                    self.check_dma_trigger(col, row, offset, value, device);
                }
            }
        }

        // If writing to shim DMA BD region (0x1D000-0x1D1FF), sync to DMA engine
        // BD addresses: 0x1D000, 0x1D020, 0x1D040, etc. (32 bytes each)
        if row == 0 && base_offset >= 0x1D000 && base_offset < 0x1D200 {
            self.sync_bd_to_dma_engine(col, base_offset, values, device);
        }

        Ok(())
    }

    /// Sync BD configuration from tile registers to DMA engine.
    fn sync_bd_to_dma_engine(&self, col: u8, base_offset: u32, values: &[u32], device: &mut DeviceState) {
        // Each BD is 32 bytes (8 words) starting at 0x1D000
        let bd_base = base_offset - 0x1D000;
        let bd_index = bd_base / 0x20;

        // Only sync if we have a complete BD (8 words)
        if values.len() < 8 {
            log::debug!("Partial BD write at 0x{:05X}, {} words", base_offset, values.len());
            return;
        }

        self.sync_bd_values_to_dma_engine(col, bd_index as u8, values, device);
    }

    /// Sync BD values to DMA engine using the data-driven register database parser.
    ///
    /// This uses `BufferDescriptor::from_registers()` to parse ALL BD fields
    /// (multi-dimensional addressing, BD chaining, locks, packet mode, iteration,
    /// AXI parameters) from the register words, then converts to the runtime
    /// BdConfig. This replaces the old manual extraction that only handled
    /// length, address, and D0.
    fn sync_bd_values_to_dma_engine(&self, col: u8, bd_index: u8, values: &[u32], device: &mut DeviceState) {
        use crate::device::dma::bd::BufferDescriptor;
        use crate::device::tile::TileType;

        log::debug!("Shim BD {} raw values: {:08X?}", bd_index, values);

        let bd = BufferDescriptor::from_registers(values, TileType::Shim);
        let config = bd.to_bd_config();

        if let Some(dma) = device.array.dma_engine_mut(col, 0) {
            if let Err(e) = dma.configure_bd(bd_index, config.clone()) {
                log::warn!("Failed to configure shim BD {}: {:?}", bd_index, e);
            } else {
                log::debug!("Configured shim BD {} addr=0x{:X} len={} d0=[{},{}] d1=[{},{}] next={:?} acq={:?} rel={:?}",
                    bd_index, config.base_addr, config.length,
                    config.d0.size, config.d0.stride, config.d1.size, config.d1.stride,
                    config.next_bd,
                    config.acquire_lock.map(|id| (id, config.acquire_value)),
                    config.release_lock.map(|id| (id, config.release_value)));
            }
        }
    }

    /// Execute a MaskWrite instruction.
    fn execute_maskwrite(&self, reg_off: u32, value: u32, mask: u32, device: &mut DeviceState) -> Result<(), String> {
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
                self.check_dma_trigger(col, row, offset, new_value, device);
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
            if row == 0 && offset >= 0x1D000 && offset < 0x1D200 {
                // Shim BD: Base_Address_High occupies bits 15:0 of word 2
                let current = tile.read_register(offset + 4);
                let merged = (current & 0xFFFF_0000) | (high_bits & 0x0000_FFFF);
                tile.write_register(offset + 4, merged);
            } else {
                tile.write_register(offset + 4, high_bits);
            }
        }

        // Re-sync the BD to DMA engine if patching BD address field
        // BD addresses start at 0x1D000, each BD is 0x20 bytes
        // Word 1 (address low) is at BD offset +4
        if row == 0 && offset >= 0x1D000 && offset < 0x1D200 {
            let bd_offset = offset - 0x1D000;
            let bd_index = bd_offset / 0x20;
            let word_in_bd = (bd_offset % 0x20) / 4;

            // If patching word 1 (address low), re-read and sync the full BD
            if word_in_bd == 1 {
                // Read BD values first (needs mutable borrow), then sync
                let bd_values = if let Some(tile) = device.tile_mut(col as usize, row as usize) {
                    let bd_base = 0x1D000 + bd_index * 0x20;
                    let mut values = [0u32; 8];
                    for i in 0..8u32 {
                        values[i as usize] = tile.read_register(bd_base + i * 4);
                    }
                    Some(values)
                } else {
                    None
                };
                if let Some(values) = bd_values {
                    self.sync_bd_values_to_dma_engine(col, bd_index as u8, &values, device);
                }
            }
        }

        Ok(())
    }

    /// Check if a register write triggers DMA operation.
    fn check_dma_trigger(&self, col: u8, row: u8, offset: u32, value: u32, device: &mut DeviceState) {
        // Shim tile DMA control registers are at specific offsets
        // S2MM Channel 0 queue: 0x1D204 (writing BD index starts transfer)
        // MM2S Channel 0 queue: 0x1D214 (writing BD index starts transfer)
        //
        // For shim DMA, writing to the queue register starts the DMA with that BD.
        // The value written contains the BD index to use.

        // Shim tile is at row 0
        if row != 0 {
            return;
        }

        // Queue registers for starting DMA
        // S2MM ch0 queue: 0x1D204, S2MM ch1 queue: 0x1D20C
        // MM2S ch0 queue: 0x1D214, MM2S ch1 queue: 0x1D21C
        const SHIM_DMA_S2MM_QUEUE_CH0: u32 = 0x1D204;
        const SHIM_DMA_S2MM_QUEUE_CH1: u32 = 0x1D20C;
        const SHIM_DMA_MM2S_QUEUE_CH0: u32 = 0x1D214;
        const SHIM_DMA_MM2S_QUEUE_CH1: u32 = 0x1D21C;

        let (channel, is_mm2s) = match offset {
            SHIM_DMA_S2MM_QUEUE_CH0 => (0u8, false),
            SHIM_DMA_S2MM_QUEUE_CH1 => (1u8, false),
            SHIM_DMA_MM2S_QUEUE_CH0 => (0u8, true),
            SHIM_DMA_MM2S_QUEUE_CH1 => (1u8, true),
            _ => return, // Not a queue register
        };

        // Start_Queue register format (same as compute/memtile):
        // - Bits 3:0: BD_ID (buffer descriptor index)
        // - Bits 23:16: Repeat_Count (run BD this many additional times)
        let bd_index = (value & 0xF) as u8;
        let repeat_count = ((value >> 16) & 0xFF) as u8;

        log::info!(
            "NPU DMA start: col={} {} ch={} bd={} repeat={}",
            col, if is_mm2s { "MM2S" } else { "S2MM" }, channel, bd_index, repeat_count
        );

        // Enqueue to the channel's task queue (matches CDO path behavior).
        if let Some(dma) = device.array.dma_engine_mut(col, 0) {
            // Convert to absolute channel index
            // For shim tiles, S2MM = 0-1, MM2S = 2-3
            let abs_channel = if is_mm2s { 2 + channel } else { channel };

            if !dma.enqueue_task(abs_channel, bd_index, repeat_count, false) {
                log::warn!("  DMA channel {} task queue overflow", abs_channel);
            } else {
                log::info!("  DMA channel {} enqueued BD {} repeat={}", abs_channel, bd_index, repeat_count);
            }
        }
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
