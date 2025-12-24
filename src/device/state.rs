//! Device state and CDO application.
//!
//! This module connects the parser (CDO commands) to the device model
//! (tile array). It applies configuration commands to build up the
//! device state that will be emulated.
//!
//! # CDO Application Process
//!
//! 1. Create a `DeviceState` with a `TileArray`
//! 2. Call `apply_cdo()` to process all CDO commands
//! 3. Device state is now configured and ready for emulation
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::device::DeviceState;
//! use xdna_emu::parser::Cdo;
//!
//! let mut state = DeviceState::new_npu1();
//! state.apply_cdo(&cdo)?;
//! state.print_summary();
//! ```

use anyhow::Result;

use super::array::TileArray;
use super::registers::{RegisterModule, TileAddress};
use super::tile::{Tile, NUM_DMA_BDS, NUM_DMA_CHANNELS, NUM_LOCKS};
use super::AieArch;
use crate::parser::cdo::{Cdo, CdoCommand};

/// Statistics about CDO application.
#[derive(Debug, Default)]
pub struct CdoStats {
    /// Total commands processed
    pub commands: usize,
    /// WRITE commands
    pub writes: usize,
    /// MASK_WRITE commands
    pub mask_writes: usize,
    /// DMA_WRITE commands (memory/program loads)
    pub dma_writes: usize,
    /// NOP commands (skipped)
    pub nops: usize,
    /// Unknown/unhandled commands
    pub unknown: usize,
    /// Bytes written to data memory
    pub data_bytes: usize,
    /// Bytes written to program memory
    pub program_bytes: usize,
}

/// Device state wrapping a tile array.
///
/// Provides methods to apply CDO commands and query device configuration.
pub struct DeviceState {
    /// The tile array
    pub array: TileArray,
    /// Statistics from last CDO application
    pub stats: CdoStats,
}

impl DeviceState {
    /// Create a new device state for the given architecture.
    pub fn new(arch: AieArch) -> Self {
        Self {
            array: TileArray::new(arch),
            stats: CdoStats::default(),
        }
    }

    /// Create an NPU1 device state.
    #[inline]
    pub fn new_npu1() -> Self {
        Self::new(AieArch::Aie2)
    }

    /// Create an NPU2 device state.
    #[inline]
    pub fn new_npu2() -> Self {
        Self::new(AieArch::Aie2P)
    }

    /// Apply a CDO to configure the device.
    ///
    /// Processes all commands in the CDO and updates the tile array accordingly.
    pub fn apply_cdo(&mut self, cdo: &Cdo) -> Result<()> {
        self.stats = CdoStats::default();

        for cmd in cdo.commands() {
            self.stats.commands += 1;
            self.apply_command(&cmd)?;
        }

        Ok(())
    }

    /// Apply a single CDO command.
    fn apply_command(&mut self, cmd: &CdoCommand) -> Result<()> {
        match cmd {
            CdoCommand::Nop { .. } => {
                self.stats.nops += 1;
            }

            CdoCommand::Write { address, value } => {
                self.stats.writes += 1;
                self.write_register(*address, *value)?;
            }

            CdoCommand::MaskWrite { address, mask, value } => {
                self.stats.mask_writes += 1;
                self.mask_write_register(*address, *mask, *value)?;
            }

            CdoCommand::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                self.dma_write(*address, data)?;
            }

            _ => {
                self.stats.unknown += 1;
            }
        }

        Ok(())
    }

    /// Write a value to a register.
    fn write_register(&mut self, address: u32, value: u32) -> Result<()> {
        let tile_addr = TileAddress::decode(address);

        // Get the tile
        let tile = match self.array.get_mut(tile_addr.col, tile_addr.row) {
            Some(t) => t,
            None => return Ok(()), // Ignore writes to non-existent tiles
        };

        // Dispatch based on register module
        match tile_addr.module() {
            RegisterModule::Locks => {
                let lock_idx = ((tile_addr.offset - 0x1F000) / 4) as usize;
                if lock_idx < NUM_LOCKS {
                    // CDO writes 32-bit, but lock value is 6-bit (0-63)
                    tile.locks[lock_idx].set(value as u8);
                }
            }

            RegisterModule::DmaBufferDescriptor => {
                self.write_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::DmaChannel => {
                self.write_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::CoreModule => {
                self.write_core_register(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::StreamSwitch => {
                self.write_stream_switch(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            _ => {
                // Ignore other writes for now
            }
        }

        Ok(())
    }

    /// Masked write to a register.
    fn mask_write_register(&mut self, address: u32, mask: u32, value: u32) -> Result<()> {
        let tile_addr = TileAddress::decode(address);

        // Get the tile
        let tile = match self.array.get_mut(tile_addr.col, tile_addr.row) {
            Some(t) => t,
            None => return Ok(()),
        };

        // For most registers, we can just apply the masked value directly
        // since we're initializing from zero state
        match tile_addr.module() {
            RegisterModule::Locks => {
                let lock_idx = ((tile_addr.offset - 0x1F000) / 4) as usize;
                if lock_idx < NUM_LOCKS {
                    // CDO mask_write uses 32-bit, but lock value is 6-bit
                    let current = tile.locks[lock_idx].value as u32;
                    let new_value = (current & !mask) | (value & mask);
                    tile.locks[lock_idx].set(new_value as u8);
                }
            }

            RegisterModule::DmaChannel => {
                self.mask_write_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
            }

            RegisterModule::CoreModule => {
                self.mask_write_core_register(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
            }

            _ => {
                // For unhandled registers, just do a regular write
                self.write_register(address, value)?;
            }
        }

        Ok(())
    }

    /// DMA write to memory.
    fn dma_write(&mut self, address: u32, data: &[u8]) -> Result<()> {
        let tile_addr = TileAddress::decode(address);

        let tile = match self.array.get_mut(tile_addr.col, tile_addr.row) {
            Some(t) => t,
            None => return Ok(()),
        };

        let offset = tile_addr.offset as usize;

        match tile_addr.module() {
            RegisterModule::Memory => {
                // Write to data memory
                if tile.write_data(offset, data) {
                    self.stats.data_bytes += data.len();
                }
            }

            RegisterModule::ProgramMemory => {
                // Write to program memory
                let pm_offset = offset - 0x20000;
                if tile.write_program(pm_offset, data) {
                    self.stats.program_bytes += data.len();
                }
            }

            _ => {
                // Could be register array writes - handle as data
                if offset < 0x10000 {
                    if tile.write_data(offset, data) {
                        self.stats.data_bytes += data.len();
                    }
                }
            }
        }

        Ok(())
    }

    /// Write to a DMA buffer descriptor.
    fn write_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // BD registers start at 0x1D000, each BD is 0x20 bytes (6 words + padding)
        let rel = offset - 0x1D000;
        let bd_idx = (rel / 0x20) as usize;
        let word = ((rel % 0x20) / 4) as usize;

        if bd_idx >= NUM_DMA_BDS || word > 5 {
            return;
        }

        let bd = &mut tile.dma_bds[bd_idx];
        match word {
            0 => bd.addr_low = value,
            1 => bd.addr_high = value,
            2 => bd.length = value,
            3 => bd.control = value,
            4 => bd.d0 = value,
            5 => bd.d1 = value,
            _ => {}
        }
    }

    /// Write to a DMA channel register.
    fn write_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // Channel registers:
        // 0x1DE00: S2MM_0_CTRL, 0x1DE04: S2MM_0_START_QUEUE
        // 0x1DE08: S2MM_1_CTRL, 0x1DE0C: S2MM_1_START_QUEUE
        // 0x1DE10: MM2S_0_CTRL, 0x1DE14: MM2S_0_START_QUEUE
        // 0x1DE18: MM2S_1_CTRL, 0x1DE1C: MM2S_1_START_QUEUE
        let rel = offset - 0x1DE00;
        let ch_idx = (rel / 8) as usize;
        let is_start_queue = (rel % 8) >= 4;

        if ch_idx >= NUM_DMA_CHANNELS {
            return;
        }

        let ch = &mut tile.dma_channels[ch_idx];
        if is_start_queue {
            ch.start_queue = value;
            ch.current_bd = (value & 0xF) as u8;
        } else {
            ch.control = value;
            ch.running = value & 1 != 0;
        }
    }

    /// Masked write to a DMA channel register.
    fn mask_write_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        let rel = offset - 0x1DE00;
        let ch_idx = (rel / 8) as usize;
        let is_start_queue = (rel % 8) >= 4;

        if ch_idx >= NUM_DMA_CHANNELS {
            return;
        }

        let ch = &mut tile.dma_channels[ch_idx];
        if is_start_queue {
            ch.start_queue = (ch.start_queue & !mask) | (value & mask);
        } else {
            ch.control = (ch.control & !mask) | (value & mask);
            ch.running = ch.control & 1 != 0;
        }
    }

    /// Write to a core register.
    fn write_core_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        match offset {
            0x32000 => {
                // CORE_CONTROL
                tile.core.control = value;
                tile.core.enabled = value & 1 != 0;
            }
            0x32004 => {
                // CORE_STATUS
                tile.core.status = value;
            }
            0x31100 => {
                // CORE_PC
                tile.core.pc = value;
            }
            0x31120 => {
                // CORE_SP
                tile.core.sp = value;
            }
            0x31130 => {
                // CORE_LR
                tile.core.lr = value;
            }
            _ => {}
        }
    }

    /// Masked write to a core register.
    fn mask_write_core_register(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        match offset {
            0x32000 => {
                // CORE_CONTROL
                tile.core.control = (tile.core.control & !mask) | (value & mask);
                tile.core.enabled = tile.core.control & 1 != 0;
            }
            0x32004 => {
                // CORE_STATUS
                tile.core.status = (tile.core.status & !mask) | (value & mask);
            }
            _ => {
                // For other registers, do full write
                self.write_core_register(col, row, offset, value);
            }
        }
    }

    /// Write to stream switch.
    fn write_stream_switch(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // Master ports: 0x3F000 + (port * 4)
        // Slave ports: 0x3F100 + (port * 4)
        if offset >= 0x3F000 && offset < 0x3F020 {
            let port = ((offset - 0x3F000) / 4) as usize;
            if port < 8 {
                tile.stream_switch.master[port].config = value;
            }
        } else if offset >= 0x3F100 && offset < 0x3F120 {
            let port = ((offset - 0x3F100) / 4) as usize;
            if port < 8 {
                tile.stream_switch.slave[port].config = value;
            }
        } else if offset == 0x3F500 {
            tile.stream_switch.ctrl_pkt = value;
        }
    }

    /// Print a summary of the device state.
    pub fn print_summary(&self) {
        println!("Device State Summary");
        println!("====================");
        self.array.print_summary();

        println!();
        println!("CDO Application Stats:");
        println!("  Commands: {}", self.stats.commands);
        println!("  Writes: {}", self.stats.writes);
        println!("  MaskWrites: {}", self.stats.mask_writes);
        println!("  DmaWrites: {}", self.stats.dma_writes);
        println!("  NOPs: {}", self.stats.nops);
        println!("  Unknown: {}", self.stats.unknown);
        println!("  Data bytes: {}", self.stats.data_bytes);
        println!("  Program bytes: {}", self.stats.program_bytes);

        // Show configured tiles
        println!();
        println!("Configured Tiles:");
        for tile in self.array.iter() {
            let has_code = tile.program_memory()
                .map(|pm| pm.iter().any(|&b| b != 0))
                .unwrap_or(false);
            let has_locks = tile.locks.iter().any(|l| l.value != 0);
            let has_bds = tile.dma_bds.iter().any(|bd| bd.is_valid());

            if has_code || has_locks || has_bds || tile.core.enabled {
                print!("  tile({},{}):", tile.col, tile.row);
                if tile.core.enabled {
                    print!(" core-enabled");
                }
                if has_code {
                    print!(" has-code");
                }
                if has_locks {
                    let active = tile.locks.iter().filter(|l| l.value != 0).count();
                    print!(" locks:{}", active);
                }
                if has_bds {
                    let active = tile.dma_bds.iter().filter(|bd| bd.is_valid()).count();
                    print!(" bds:{}", active);
                }
                println!();
            }
        }
    }

    /// Get count of enabled cores.
    pub fn enabled_cores(&self) -> usize {
        self.array.compute_tiles().filter(|t| t.core.enabled).count()
    }

    /// Get count of tiles with program code.
    pub fn tiles_with_code(&self) -> usize {
        self.array.compute_tiles()
            .filter(|t| {
                t.program_memory()
                    .map(|pm| pm.iter().any(|&b| b != 0))
                    .unwrap_or(false)
            })
            .count()
    }

    /// Get number of columns.
    #[inline]
    pub fn cols(&self) -> usize {
        self.array.cols() as usize
    }

    /// Get number of rows.
    #[inline]
    pub fn rows(&self) -> usize {
        self.array.rows() as usize
    }

    /// Get a tile by coordinates, or None if out of bounds.
    #[inline]
    pub fn tile(&self, col: usize, row: usize) -> Option<&Tile> {
        if col < self.cols() && row < self.rows() {
            Some(self.array.tile(col as u8, row as u8))
        } else {
            None
        }
    }

    /// Get a mutable tile by coordinates, or None if out of bounds.
    #[inline]
    pub fn tile_mut(&mut self, col: usize, row: usize) -> Option<&mut Tile> {
        if col < self.cols() && row < self.rows() {
            Some(self.array.tile_mut(col as u8, row as u8))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::xclbin::SectionKind;
    use crate::parser::{Xclbin, AiePartition};
    use crate::parser::cdo::find_cdo_offset;

    #[test]
    fn test_device_state_creation() {
        let state = DeviceState::new_npu1();
        assert_eq!(state.array.cols(), 5);
        assert_eq!(state.array.rows(), 6);
    }

    #[test]
    fn test_apply_real_cdo() {
        let test_xclbin = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin";

        if !std::path::Path::new(test_xclbin).exists() {
            eprintln!("Skipping real CDO test: file not found");
            return;
        }

        // Load xclbin
        let xclbin = Xclbin::from_file(test_xclbin).unwrap();
        let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
        let partition = AiePartition::parse(section.data()).unwrap();
        let pdi = partition.primary_pdi().unwrap();
        let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
        let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

        // Apply to device
        let mut state = DeviceState::new_npu1();
        state.apply_cdo(&cdo).unwrap();

        // Verify something was configured
        assert!(state.stats.commands > 0, "Expected commands to be processed");
        assert!(state.stats.dma_writes > 0, "Expected DMA writes (shim control packets)");
        assert!(state.stats.writes > 0 || state.stats.mask_writes > 0,
            "Expected register writes");

        // Note: For this xclbin, DMA_WRITE goes to shim tile for control packets,
        // not to compute tiles for code/data. That's expected - core code is loaded
        // via XRT at runtime, not embedded in CDO.
    }

    #[test]
    fn test_lock_write() {
        let mut state = DeviceState::new_npu1();

        // Write to lock 5 in tile(1,2)
        let addr = TileAddress::encode(1, 2, 0x1F014); // Lock 5
        state.write_register(addr, 42).unwrap();

        let tile = state.array.tile(1, 2);
        assert_eq!(tile.locks[5].value, 42);
    }

    #[test]
    fn test_dma_channel_write() {
        let mut state = DeviceState::new_npu1();

        // Write to DMA_MM2S_0_CTRL in tile(1,2)
        let addr = TileAddress::encode(1, 2, 0x1DE10);
        state.write_register(addr, 0x01).unwrap(); // Enable

        let tile = state.array.tile(1, 2);
        assert!(tile.dma_channels[2].is_enabled()); // Channel 2 is MM2S_0
    }

    #[test]
    fn test_core_control_mask_write() {
        let mut state = DeviceState::new_npu1();

        // Mask write to enable core
        let addr = TileAddress::encode(1, 2, 0x32000);
        state.mask_write_register(addr, 0x1, 0x1).unwrap();

        let tile = state.array.tile(1, 2);
        assert!(tile.core.enabled);
    }
}
