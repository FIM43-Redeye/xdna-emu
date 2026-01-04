//! NPU instruction executor.
//!
//! Executes NPU instructions against the device state, performing
//! register writes, address patches, and DMA triggers.

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

/// NPU instruction executor.
///
/// Executes the host-to-NPU command stream that triggers DMA transfers
/// and configures the shim tiles.
pub struct NpuExecutor {
    /// Host buffers for address patching (indexed by arg_idx).
    host_buffers: Vec<HostBuffer>,
    /// Number of instructions executed.
    executed_count: usize,
}

impl NpuExecutor {
    /// Create a new executor.
    pub fn new() -> Self {
        Self {
            host_buffers: Vec::new(),
            executed_count: 0,
        }
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

    /// Execute all instructions in a stream against the device.
    pub fn execute(&mut self, stream: &NpuInstructionStream, device: &mut DeviceState) -> Result<(), String> {
        for instr in stream.instructions() {
            self.execute_instruction(instr, device)?;
            self.executed_count += 1;
        }
        Ok(())
    }

    /// Execute a single instruction.
    fn execute_instruction(&self, instr: &NpuInstruction, device: &mut DeviceState) -> Result<(), String> {
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

            NpuInstruction::Sync { channel, column, direction, .. } => {
                log::debug!(
                    "NPU Sync: channel={} column={} direction={}",
                    channel, column, direction
                );
                // Sync is a wait point - in our immediate execution model,
                // we assume the operation completes.
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

        // Write to the device register
        if let Some(tile) = device.tile_mut(col as usize, row as usize) {
            tile.write_register(offset, value);

            // Check if this is a DMA channel control register (triggers DMA start)
            self.check_dma_trigger(col, row, offset, value, device);
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

    /// Sync BD values to DMA engine (internal helper).
    fn sync_bd_values_to_dma_engine(&self, col: u8, bd_index: u8, values: &[u32], device: &mut DeviceState) {
        use crate::device::dma::BdConfig;

        // Parse BD fields from the values
        // Dump raw values for debugging
        log::debug!("BD {} raw values: {:08X?}", bd_index, values);

        // BD format (from AM020 - note: shim DMA has different layout than tile DMA):
        // Shim DMA BD format (32 bytes):
        // Word 0: Buffer length (bytes) - bits[13:0] = length
        // Word 1: Buffer address low
        // Word 2: Buffer address high + packet info
        // Word 3: D0 config (size, stride)
        // Word 4: D1 config
        // Word 5: D2 config
        // Word 6: Iteration config
        // Word 7: Control/next BD

        let length_word = values[0];
        // Buffer length is in 32-bit words, not bytes (AM025 Shim DMA BD format)
        let length_words = length_word & 0x3FFF;  // Bottom 14 bits = word count
        let length = length_words * 4;  // Convert to bytes

        let addr_low = values[1];
        let addr_high = values[2] & 0xFFFF;  // Bits 15:0 are address high

        // Combine low and high address parts
        let base_addr = ((addr_high as u64) << 32) | (addr_low as u64);

        // D0 dimension: word 3
        // Bits 13:0 = size in 32-bit words
        let d0_word = values.get(3).copied().unwrap_or(0);
        let d0_size_words = d0_word & 0x3FFF;
        let d0_size_bytes = d0_size_words * 4;

        // Use D0 size if buffer length is 0
        let actual_length = if length > 0 { length } else { d0_size_bytes };

        // Create a simple 1D linear transfer
        // BdConfig::simple_1d properly sets d0.size=length, d0.stride=1 for sequential access
        let config = BdConfig::simple_1d(base_addr, actual_length);

        if let Some(dma) = device.array.dma_engine_mut(col, 0) {
            if let Err(e) = dma.configure_bd(bd_index as u8, config) {
                log::warn!("Failed to configure BD {}: {:?}", bd_index, e);
            } else {
                log::debug!("Configured BD {} with addr=0x{:X} length={}", bd_index, base_addr, length);
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
            // Read-modify-write
            let current = tile.read_register(offset);
            let new_value = (current & !mask) | (value & mask);
            tile.write_register(offset, new_value);

            self.check_dma_trigger(col, row, offset, new_value, device);
        }

        Ok(())
    }

    /// Execute a DDR patch instruction.
    fn execute_ddr_patch(&self, reg_addr: u32, arg_idx: u8, arg_plus: u32, device: &mut DeviceState) -> Result<(), String> {
        // Get the host buffer address
        let buffer = self.host_buffers.get(arg_idx as usize).ok_or_else(|| {
            format!(
                "DDR patch references arg_idx {} but only {} buffers available",
                arg_idx,
                self.host_buffers.len()
            )
        })?;

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
            // Some BDs need the high bits too at offset+4
            tile.write_register(offset + 4, (patched_addr >> 32) as u32);
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
                if let Some(tile) = device.tile(col as usize, row as usize) {
                    let bd_base = 0x1D000 + bd_index * 0x20;
                    let mut values = [0u32; 8];
                    for i in 0..8u32 {
                        values[i as usize] = tile.read_register(bd_base + i * 4);
                    }
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

        // BD index is in bits 0-3 of the value
        let bd_index = (value & 0xF) as u8;

        log::info!(
            "NPU DMA start: col={} {} ch={} bd={}",
            col, if is_mm2s { "MM2S" } else { "S2MM" }, channel, bd_index
        );

        // Actually start the DMA channel with this BD
        if let Some(dma) = device.array.dma_engine_mut(col, 0) {
            // Convert to absolute channel index
            // S2MM channels are 0-1, MM2S channels are 2-3 for compute tiles
            // For shim tiles, S2MM = 0-5, MM2S = 6-11 but we only use 0-1 of each
            let abs_channel = if is_mm2s { 2 + channel } else { channel };

            match dma.start_channel(abs_channel, bd_index) {
                Ok(()) => {
                    log::info!("  DMA channel {} started with BD {}", abs_channel, bd_index);
                }
                Err(e) => {
                    log::warn!("  Failed to start DMA channel: {:?}", e);
                }
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
}
