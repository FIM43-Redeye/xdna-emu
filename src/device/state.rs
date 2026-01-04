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
use std::sync::Arc;

use super::arch_config::{ArchConfig, Aie2Config, Aie2pConfig};
use super::array::TileArray;
use super::registers::{RegisterModule, TileAddress};
use super::tile::{Tile, TileType, NUM_DMA_BDS, NUM_DMA_CHANNELS, NUM_LOCKS};
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
    /// Create a new device state for the given architecture configuration.
    pub fn new(arch: Arc<dyn ArchConfig>) -> Self {
        Self {
            array: TileArray::new(arch),
            stats: CdoStats::default(),
        }
    }

    /// Create an NPU1 device state.
    #[inline]
    pub fn new_npu1() -> Self {
        Self::new(Arc::new(Aie2Config))
    }

    /// Create an NPU2 device state.
    #[inline]
    pub fn new_npu2() -> Self {
        Self::new(Arc::new(Aie2pConfig))
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
                let tile_addr = TileAddress::decode(*address);
                log::trace!("CDO Write: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
                self.write_register(*address, *value)?;
            }

            CdoCommand::MaskWrite { address, mask, value } => {
                self.stats.mask_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::trace!("CDO MaskWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset);
                self.mask_write_register(*address, *mask, *value)?;
            }

            CdoCommand::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::debug!("CDO DmaWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} module={:?} len={}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, tile_addr.module(), data.len());
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
        // Import register address constants
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};

        match tile_addr.module() {
            RegisterModule::Locks => {
                Self::write_lock_value(tile, tile_addr, mm::LOCK_BASE, mm::LOCK_STRIDE, value, false);
            }

            RegisterModule::MemTileLocks => {
                Self::write_lock_value(tile, tile_addr, mt::LOCK_BASE, mt::LOCK_STRIDE, value, true);
            }

            RegisterModule::DmaBufferDescriptor => {
                self.write_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::MemTileDmaBufferDescriptor => {
                self.write_memtile_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::DmaChannel => {
                self.write_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::MemTileDmaChannel => {
                self.write_memtile_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::CoreModule => {
                self.write_core_register(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::StreamSwitch => {
                self.write_stream_switch(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            RegisterModule::MemTileStreamSwitch => {
                self.write_memtile_stream_switch(tile_addr.col, tile_addr.row, tile_addr.offset, value);
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
        // Import register address constants
        use super::registers_spec::{memory_module as mm, mem_tile_module as mt};

        match tile_addr.module() {
            RegisterModule::Locks => {
                Self::mask_write_lock_value(tile, tile_addr.offset, mm::LOCK_BASE, mm::LOCK_STRIDE, mask, value);
            }

            RegisterModule::MemTileLocks => {
                Self::mask_write_lock_value(tile, tile_addr.offset, mt::LOCK_BASE, mt::LOCK_STRIDE, mask, value);
            }

            RegisterModule::DmaChannel => {
                self.mask_write_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
            }

            RegisterModule::MemTileDmaChannel => {
                self.mask_write_memtile_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
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
    ///
    /// Writes bulk data to program memory, data memory, or DMA BD registers
    /// depending on the address offset.
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
                use super::registers_spec::PROGRAM_MEMORY_BASE;
                // Write to program memory
                let pm_offset = offset - PROGRAM_MEMORY_BASE as usize;
                if tile.write_program(pm_offset, data) {
                    self.stats.program_bytes += data.len();
                }
            }

            RegisterModule::DmaBufferDescriptor => {
                // Write to DMA BD registers
                self.dma_write_bd_data(tile_addr.col, tile_addr.row, tile_addr.offset, data);
            }

            RegisterModule::MemTileDmaBufferDescriptor => {
                // Write to MemTile DMA BD registers (8 words per BD)
                self.dma_write_memtile_bd_data(tile_addr.col, tile_addr.row, tile_addr.offset, data);
            }

            _ => {
                use super::registers_spec::MEM_TILE_DATA_MEMORY_END;
                // Could be register array writes - handle as data
                if offset <= MEM_TILE_DATA_MEMORY_END as usize
                    && tile.write_data(offset, data) {
                        self.stats.data_bytes += data.len();
                    }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Lock helper functions (consolidate duplicate compute/memtile code)
    // =========================================================================

    /// Write a lock value (direct write, no mask).
    ///
    /// Lock registers are 16 bytes (LOCK_STRIDE) apart per AM025.
    fn write_lock_value(
        tile: &mut Tile,
        tile_addr: TileAddress,
        base: u32,
        stride: u32,
        value: u32,
        is_memtile: bool,
    ) {
        let lock_idx = ((tile_addr.offset - base) / stride) as usize;
        if lock_idx < NUM_LOCKS {
            tile.locks[lock_idx].set(value as u8);
            if value != 0 {
                let tile_type = if is_memtile { "MemTile" } else { "Compute" };
                log::info!("CDO init {} lock {} on tile ({},{}) = {}",
                    tile_type, lock_idx, tile_addr.col, tile_addr.row, value);
            }
        }
    }

    /// Masked write to a lock value.
    fn mask_write_lock_value(
        tile: &mut Tile,
        offset: u32,
        base: u32,
        stride: u32,
        mask: u32,
        value: u32,
    ) {
        let lock_idx = ((offset - base) / stride) as usize;
        if lock_idx < NUM_LOCKS {
            let current = tile.locks[lock_idx].value as u32;
            let new_value = (current & !mask) | (value & mask);
            tile.locks[lock_idx].set(new_value as u8);
        }
    }

    /// Write BD data from a byte array.
    fn dma_write_bd_data(&mut self, col: u8, row: u8, offset: u32, data: &[u8]) {
        let words: Vec<u32> = data
            .chunks(4)
            .map(|chunk| {
                let mut arr = [0u8; 4];
                arr[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(arr)
            })
            .collect();

        log::debug!("dma_write_bd_data tile({},{}) offset=0x{:05X} data_len={} words={:X?}",
            col, row, offset, data.len(), words);

        for (i, &word) in words.iter().enumerate() {
            let reg_offset = offset + (i as u32) * 4;
            self.write_dma_bd(col, row, reg_offset, word);
        }
    }

    /// Write to a DMA buffer descriptor.
    fn write_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::registers_spec::{memory_module as mm, sign_extend_7bit};

        // BD registers: base at DMA_BD_BASE, stride DMA_BD_STRIDE (AM025)
        let rel = offset - mm::DMA_BD_BASE;
        let bd_idx = (rel / mm::DMA_BD_STRIDE) as usize;
        let word = ((rel % mm::DMA_BD_STRIDE) / 4) as usize;

        if bd_idx >= NUM_DMA_BDS || word >= mm::DMA_BD_WORDS {
            return;
        }

        // Update the legacy tile BD storage
        if let Some(tile) = self.array.get_mut(col, row) {
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

        // Also update the DmaEngine's BD config
        // First, read the tile BD values (need to do this before mutable borrow)
        let bd_values = self.array.get(col, row).map(|tile| {
            let tile_bd = &tile.dma_bds[bd_idx];
            // AIE-ML memory module BD format (AM025 memory_module/dma/bd.txt)
            // Word 0: bits 27:14 = Base_Address (word address), bits 13:0 = Buffer_Length (words)
            // Word 5: bit 25 = Valid_BD, bits 24:18 = Lock_Rel_Value, bits 16:13 = Lock_Rel_ID,
            //         bit 12 = Lock_Acq_Enable, bits 11:5 = Lock_Acq_Value, bits 3:0 = Lock_Acq_ID
            let word0 = tile_bd.addr_low;
            let word5 = tile_bd.d1;

            // Parse word0 using constants from AM025
            let base_addr_words = ((word0 >> mm::bd::WORD0_BASE_ADDR_SHIFT) & mm::bd::WORD0_BASE_ADDR_MASK) as u64;
            let base_addr = base_addr_words * 4; // Convert word address to byte address

            let length_words = word0 & mm::bd::WORD0_BUFFER_LEN_MASK;
            let length = length_words * 4; // Convert word count to byte count

            // Parse word5 using constants from AM025
            let valid = (word5 >> mm::bd::WORD5_VALID_BD_BIT) & 1 != 0;

            // Lock configuration from word5
            let lock_acq_id = (word5 & mm::bd::WORD5_LOCK_ACQ_ID_MASK) as u8;
            let lock_acq_raw = (word5 >> mm::bd::WORD5_LOCK_ACQ_VALUE_SHIFT) & mm::bd::WORD5_LOCK_ACQ_VALUE_MASK;
            let lock_acq_value = sign_extend_7bit(lock_acq_raw);
            let lock_acq_enable = (word5 >> mm::bd::WORD5_LOCK_ACQ_ENABLE_BIT) & 1 != 0;
            let lock_rel_id = ((word5 >> mm::bd::WORD5_LOCK_REL_ID_SHIFT) & mm::bd::WORD5_LOCK_REL_ID_MASK) as u8;
            let lock_rel_raw = (word5 >> mm::bd::WORD5_LOCK_REL_VALUE_SHIFT) & mm::bd::WORD5_LOCK_REL_VALUE_MASK;
            let lock_rel_value = sign_extend_7bit(lock_rel_raw);

            // BD chaining: bits 30:27 = Next_BD, bit 26 = Use_Next_BD
            let next_bd_id = ((word5 >> mm::bd::WORD5_NEXT_BD_SHIFT) & mm::bd::WORD5_NEXT_BD_MASK) as u8;
            let use_next_bd = (word5 >> mm::bd::WORD5_USE_NEXT_BD_BIT) & 1 != 0;
            let next_bd = if use_next_bd { Some(next_bd_id) } else { None };

            log::trace!("  BD {} word5=0x{:08X} next_bd_id={} use_next_bd={}",
                bd_idx, word5, next_bd_id, use_next_bd);

            // For backwards compatibility, also check if any data is non-zero
            let has_any_data = word0 != 0 || tile_bd.addr_high != 0 ||
                               tile_bd.length != 0 || tile_bd.control != 0 ||
                               tile_bd.d0 != 0 || word5 != 0;
            let is_valid = valid || has_any_data;

            (base_addr, length, is_valid, lock_acq_enable, lock_acq_id, lock_acq_value,
             lock_rel_id, lock_rel_value, next_bd)
        });

        // Now configure the DmaEngine if we have values
        if let Some((base_addr, length, valid, lock_acq_enable, lock_acq_id, lock_acq_value,
                     lock_rel_id, lock_rel_value, next_bd)) = bd_values {
            log::trace!("  BD {} word {} tile({},{}) -> addr=0x{:X} len={} valid={}",
                bd_idx, word, col, row, base_addr, length, valid);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                use crate::device::dma::BdConfig;
                // Use simple_1d to properly set d0.size and d0.stride for sequential access
                let mut config = BdConfig::simple_1d(base_addr, length);
                config.valid = valid;

                // Set lock configuration if acquire is enabled
                if lock_acq_enable {
                    config.acquire_lock = Some(lock_acq_id);
                    config.acquire_value = lock_acq_value;
                }

                // Set lock release if non-zero
                if lock_rel_value != 0 {
                    config.release_lock = Some(lock_rel_id);
                    config.release_value = lock_rel_value;
                }

                // Set BD chaining if enabled
                config.next_bd = next_bd;

                if let Err(e) = dma.configure_bd(bd_idx as u8, config.clone()) {
                    log::debug!("Failed to configure BD {} on DmaEngine ({},{}): {:?}",
                        bd_idx, col, row, e);
                } else if valid {
                    log::info!("CDO configured BD {} on tile ({},{}) addr=0x{:X} len={} acq={:?} rel={:?} next={:?}",
                        bd_idx, col, row, base_addr, length,
                        config.acquire_lock.map(|id| (id, config.acquire_value)),
                        config.release_lock.map(|id| (id, config.release_value)),
                        next_bd);
                }
            }
        }
    }

    /// Write to a DMA channel register.
    fn write_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::registers_spec::memory_module as mm;

        // Channel registers: base at DMA_CHANNEL_BASE, stride DMA_CHANNEL_STRIDE (AM025)
        // Each channel has CTRL register and START_QUEUE register (4 bytes each)
        let rel = offset - mm::DMA_CHANNEL_BASE;
        let ch_idx = (rel / mm::DMA_CHANNEL_STRIDE) as usize;
        let is_start_queue = (rel % mm::DMA_CHANNEL_STRIDE) >= 4;

        if ch_idx >= NUM_DMA_CHANNELS {
            return;
        }

        // Update the tile's DMA channel state
        if let Some(tile) = self.array.get_mut(col, row) {
            let ch = &mut tile.dma_channels[ch_idx];
            if is_start_queue {
                ch.start_queue = value;
                ch.current_bd = (value & 0xF) as u8;
            } else {
                ch.control = value;
                ch.running = value & 1 != 0;
            }
        }

        // Start the DmaEngine channel when START_QUEUE is written
        // Note: value=0 means BD 0, not "don't start"
        // We start the channel whenever the start queue register is written
        // Task queue register format:
        // - Bits 3:0: BD_ID (buffer descriptor index)
        // - Bits 23:16: Repeat_Count (run BD this many additional times)
        if is_start_queue {
            let bd_idx = (value & 0xF) as u8;
            let repeat_count = ((value >> 16) & 0xFF) as u8;
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                // Start the channel with the specified BD and repeat count
                if let Err(e) = dma.start_channel_with_repeat(ch_idx as u8, bd_idx, repeat_count) {
                    log::warn!("Failed to start DMA channel {} on tile ({},{}): {:?}",
                        ch_idx, col, row, e);
                } else {
                    if repeat_count > 0 {
                        log::info!("CDO started DMA channel {} with BD {} repeat={} on tile ({},{})",
                            ch_idx, bd_idx, repeat_count, col, row);
                    } else {
                        log::info!("CDO started DMA channel {} with BD {} on tile ({},{})",
                            ch_idx, bd_idx, col, row);
                    }
                }
            }
        }
    }

    /// Masked write to a DMA channel register.
    fn mask_write_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        use super::registers_spec::memory_module as mm;
        let rel = offset - mm::DMA_CHANNEL_BASE;
        let ch_idx = (rel / 8) as usize;
        let is_start_queue = (rel % 8) >= 4;

        if ch_idx >= NUM_DMA_CHANNELS {
            return;
        }

        // Update the tile's DMA channel state
        let new_start_queue = if let Some(tile) = self.array.get_mut(col, row) {
            let ch = &mut tile.dma_channels[ch_idx];
            if is_start_queue {
                ch.start_queue = (ch.start_queue & !mask) | (value & mask);
                Some(ch.start_queue)
            } else {
                ch.control = (ch.control & !mask) | (value & mask);
                ch.running = ch.control & 1 != 0;
                None
            }
        } else {
            None
        };

        // Also start the DmaEngine channel when START_QUEUE is written
        // Start the channel whenever START_QUEUE is written (even if BD index is 0)
        // Task queue register format:
        // - Bits 3:0: BD_ID (buffer descriptor index)
        // - Bits 23:16: Repeat_Count (run BD this many additional times)
        if let Some(queue_val) = new_start_queue {
            let bd_idx = (queue_val & 0xF) as u8;
            let repeat_count = ((queue_val >> 16) & 0xFF) as u8;
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if let Err(e) = dma.start_channel_with_repeat(ch_idx as u8, bd_idx, repeat_count) {
                    log::warn!("Failed to start DMA channel {} on tile ({},{}): {:?}",
                        ch_idx, col, row, e);
                } else {
                    if repeat_count > 0 {
                        log::info!("CDO started DMA channel {} with BD {} repeat={} on tile ({},{})",
                            ch_idx, bd_idx, repeat_count, col, row);
                    } else {
                        log::info!("CDO started DMA channel {} with BD {} on tile ({},{})",
                            ch_idx, bd_idx, col, row);
                    }
                }
            }
        }
    }

    /// Write to a core register.
    fn write_core_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::registers_spec::core_module as cm;

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        match offset {
            cm::CORE_CONTROL => {
                tile.core.control = value;
                tile.core.enabled = value & 1 != 0;
            }
            cm::CORE_STATUS => {
                tile.core.status = value;
            }
            cm::CORE_PC => {
                tile.core.pc = value;
            }
            cm::CORE_SP => {
                tile.core.sp = value;
            }
            cm::CORE_LR => {
                tile.core.lr = value;
            }
            _ => {}
        }
    }

    /// Masked write to a core register.
    fn mask_write_core_register(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        use super::registers_spec::core_module as cm;

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        match offset {
            cm::CORE_CONTROL => {
                tile.core.control = (tile.core.control & !mask) | (value & mask);
                tile.core.enabled = tile.core.control & 1 != 0;
            }
            cm::CORE_STATUS => {
                tile.core.status = (tile.core.status & !mask) | (value & mask);
            }
            _ => {
                // For other registers, do full write
                self.write_core_register(col, row, offset, value);
            }
        }
    }

    /// Write to stream switch (compute/shim tiles).
    ///
    /// Master config format: bit 31 = enable, bits 4:0 = slave select
    /// Slave config format: bit 31 = enable
    fn write_stream_switch(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::aie2_spec::stream_switch::{ENABLE_BIT, SLAVE_SELECT_MASK};
        use super::registers_spec::memory_module as mm;

        // Master ports: base + (port * 4), Slave ports: base + (port * 4)
        if (mm::STREAM_SWITCH_MASTER_BASE..mm::STREAM_SWITCH_MASTER_END).contains(&offset) {
            let port = ((offset - mm::STREAM_SWITCH_MASTER_BASE) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let slave_select = (value & SLAVE_SELECT_MASK) as usize;

            // Get port type before mutable borrow
            let port_type = self.array.get(col, row)
                .and_then(|t| t.stream_switch.masters.get(port))
                .map(|p| p.port_type);

            if let Some(tile) = self.array.get_mut(col, row) {
                if port < tile.stream_switch.masters.len() {
                    tile.stream_switch.masters[port].config = value;
                    tile.stream_switch.masters[port].enabled = enable;

                    // Configure local route if enabled
                    if enable && slave_select < tile.stream_switch.slaves.len() {
                        tile.stream_switch.configure_local_route(slave_select, port);
                        log::info!("Tile ({},{}) stream switch: master[{}] enabled with slave_select={} (local route: slave[{}] -> master[{}])",
                            col, row, port, slave_select, slave_select, port);
                    } else if enable {
                        log::info!("Tile ({},{}) stream switch: master[{}] enabled with slave_select={} (no local route - slave out of range)",
                            col, row, port, slave_select);
                    }
                }
            }

            // Create global route for external ports (North/South)
            if enable {
                if let Some(pt) = port_type {
                    self.derive_global_route(col, row, port as u8, pt);
                }
            }
        // Slave ports - extend range to cover all ports
        } else if (mm::STREAM_SWITCH_SLAVE_BASE..mm::STREAM_SWITCH_SLAVE_END).contains(&offset) {
            let port = ((offset - mm::STREAM_SWITCH_SLAVE_BASE) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;

            if let Some(tile) = self.array.get_mut(col, row) {
                if port < tile.stream_switch.slaves.len() {
                    tile.stream_switch.slaves[port].config = value;
                    tile.stream_switch.slaves[port].enabled = enable;
                }
            }
        }
        // Note: ctrl_pkt (0x3F500) not stored - functional switch doesn't need it
    }

    /// Derive global route from master port type to adjacent tile.
    ///
    /// When a master port of type North/South is enabled, create a route
    /// to the adjacent tile's corresponding slave port.
    fn derive_global_route(&mut self, col: u8, row: u8, master_port: u8, port_type: crate::device::stream_switch::PortType) {
        use crate::device::stream_switch::PortType;

        match port_type {
            PortType::North => {
                // North master connects to tile above's South slave
                // Determine the destination slave port based on tile type
                let dest_row = row + 1;
                if dest_row < self.array.rows() {
                    // For compute tiles, South slave ports start at specific indices
                    // Map master port to appropriate destination slave
                    let dest_slave = self.map_north_master_to_south_slave(col, row, dest_row, master_port);
                    log::info!("Derived global route: ({},{}) master[{}] -> ({},{}) slave[{}] (North)",
                        col, row, master_port, col, dest_row, dest_slave);
                    self.array.stream_router.add_route(col, row, master_port, col, dest_row, dest_slave);
                }
            }
            PortType::South => {
                // South master connects to tile below's North slave
                if row > 0 {
                    let dest_row = row - 1;
                    let dest_slave = self.map_south_master_to_north_slave(col, row, dest_row, master_port);
                    log::info!("Derived global route: ({},{}) master[{}] -> ({},{}) slave[{}] (South)",
                        col, row, master_port, col, dest_row, dest_slave);
                    self.array.stream_router.add_route(col, row, master_port, col, dest_row, dest_slave);
                }
            }
            _ => {
                // DMA, Core, etc. - no global route needed
            }
        }
    }

    /// Map a North master port to the corresponding South slave on the tile above.
    ///
    /// When a tile sends data North (upward), the receiving tile receives it on its
    /// South slave port (since the data is coming from below).
    ///
    /// Port layouts (per AM025 and aie2_spec::stream_switch):
    /// - MemTile South slaves: 7-12 (6 ports, South0-South5)
    /// - Compute South slaves: 5-10 (6 ports, South0-South5)
    fn map_north_master_to_south_slave(&self, col: u8, src_row: u8, dest_row: u8, master_port: u8) -> u8 {
        use super::aie2_spec::stream_switch::{shim, mem_tile, compute};

        let arch = self.array.arch();
        let src_type = arch.tile_type(col, src_row);
        let dest_type = arch.tile_type(col, dest_row);

        match dest_type {
            TileType::MemTile => {
                // Destination is MemTile: South slaves receive from Shim or Compute
                match src_type {
                    TileType::Shim => {
                        // From Shim: North masters 12-17 → MemTile South slaves 7-12
                        if master_port >= shim::NORTH_MASTER_START && master_port <= shim::NORTH_MASTER_END {
                            mem_tile::SOUTH_SLAVE_START + (master_port - shim::NORTH_MASTER_START)
                        } else {
                            mem_tile::SOUTH_SLAVE_START
                        }
                    }
                    _ => {
                        // From compute tile above MemTile (shouldn't happen in Phoenix)
                        mem_tile::SOUTH_SLAVE_START + (master_port % 6)
                    }
                }
            }
            TileType::Compute => {
                // Destination is Compute tile: South slaves receive from below
                match src_type {
                    TileType::MemTile => {
                        // From MemTile: North masters 11-16 → Compute South slaves 5-10
                        if master_port >= mem_tile::NORTH_MASTER_START && master_port <= mem_tile::NORTH_MASTER_END {
                            compute::SOUTH_SLAVE_START + (master_port - mem_tile::NORTH_MASTER_START)
                        } else {
                            compute::SOUTH_SLAVE_START
                        }
                    }
                    _ => {
                        // From another compute tile: North masters 15-18 → South slaves 5-8
                        if master_port >= compute::NORTH_MASTER_START && master_port <= compute::NORTH_MASTER_END {
                            compute::SOUTH_SLAVE_START + (master_port - compute::NORTH_MASTER_START)
                        } else {
                            compute::SOUTH_SLAVE_START
                        }
                    }
                }
            }
            TileType::Shim => {
                // Shim tiles don't receive from North (they're at the bottom)
                shim::NORTH_SLAVE_START
            }
        }
    }

    /// Map a South master port to the corresponding North slave on the tile below.
    ///
    /// When a tile sends data South (downward), the receiving tile receives it on its
    /// North slave port (since the data is coming from above).
    ///
    /// Port layouts (per AM025 and aie2_spec::stream_switch):
    /// - Shim North slaves: 14-17 (4 ports, North0-North3)
    /// - MemTile North slaves: 13-16 (4 ports, North0-North3)
    /// - Compute North slaves: 15-18 (4 ports, North0-North3)
    fn map_south_master_to_north_slave(&self, col: u8, src_row: u8, dest_row: u8, master_port: u8) -> u8 {
        use super::aie2_spec::stream_switch::{shim, mem_tile, compute};

        let arch = self.array.arch();
        let src_type = arch.tile_type(col, src_row);
        let dest_type = arch.tile_type(col, dest_row);

        match dest_type {
            TileType::Shim => {
                // Destination is Shim: North slaves 14-17
                // MemTile South masters 7-10 → Shim North slaves 14-17
                if src_type.is_mem_tile() && master_port >= mem_tile::SOUTH_MASTER_START && master_port <= mem_tile::SOUTH_MASTER_END {
                    shim::NORTH_SLAVE_START + (master_port - mem_tile::SOUTH_MASTER_START)
                } else {
                    shim::NORTH_SLAVE_START
                }
            }
            TileType::MemTile => {
                // Destination is MemTile: North slaves 13-16
                // Compute South masters 5-10 → MemTile North slaves 13-16
                if src_type.is_compute() && master_port >= compute::SOUTH_MASTER_START && master_port <= compute::SOUTH_MASTER_END {
                    // Map first 4 South masters to 4 North slaves
                    let offset = (master_port - compute::SOUTH_MASTER_START).min(3);
                    mem_tile::NORTH_SLAVE_START + offset
                } else {
                    mem_tile::NORTH_SLAVE_START
                }
            }
            TileType::Compute => {
                // Destination is Compute tile: North slaves 15-18 (NOT 8-9!)
                // Source compute tile South masters 5-10 → destination North slaves 15-18
                if master_port >= compute::SOUTH_MASTER_START && master_port <= compute::SOUTH_MASTER_END {
                    // Map first 4 South masters to 4 North slaves
                    let offset = (master_port - compute::SOUTH_MASTER_START).min(3);
                    compute::NORTH_SLAVE_START + offset
                } else {
                    compute::NORTH_SLAVE_START
                }
            }
        }
    }

    // =========================================================================
    // MemTile handlers (row 1) - different register layouts from compute tiles
    // =========================================================================

    /// Write to a MemTile DMA buffer descriptor.
    ///
    /// MemTile BDs are 8 words each (32 bytes), at base 0xA0000.
    /// Format (AM025 memory_tile_module/dma/bd.txt):
    /// - Word 0: Buffer_Length[16:0], Out_Of_Order_BD_ID[22:17], Packet_ID[27:23], Packet_Type[30:28], Enable_Packet[31]
    /// - Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20], D0_Zero_Before[31:26]
    /// - Word 2: D0_Stepsize[16:0], D0_Wrap[26:17], TLAST_Suppress[31]
    /// - Word 3: D1_Stepsize[16:0], D1_Wrap[26:17], D1_Zero_Before[31:27]
    /// - Word 4: D2_Stepsize[16:0], D2_Wrap[26:17], D2_Zero_Before[30:27], Enable_Compression[31]
    /// - Word 5: D3_Stepsize[16:0], D0_Zero_After[22:17], D1_Zero_After[27:23], D2_Zero_After[31:28]
    /// - Word 6: Iteration_Stepsize[16:0], Iteration_Wrap[22:17], Iteration_Current[28:23]
    /// - Word 7: Lock_Acq_ID[7:0], Lock_Acq_Value[14:8], Lock_Acq_Enable[15], Lock_Rel_ID[23:16], Lock_Rel_Value[30:24], Valid_BD[31]
    fn write_memtile_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::registers_spec::mem_tile_module as mt;
        // MemTile BDs: base, 8 words (0x20 bytes) per BD, up to 48 BDs
        let rel = offset - mt::DMA_BD_BASE;
        let bd_idx = (rel / mt::DMA_BD_STRIDE) as usize;
        let word = ((rel % mt::DMA_BD_STRIDE) / 4) as usize;

        // MemTile has 48 BDs (24 for S2MM + 24 for MM2S), up to index 47
        if bd_idx >= mt::DMA_BD_COUNT || word > 7 {
            return;
        }

        // Store raw BD words in tile for reference (using first 16 legacy slots if available)
        if bd_idx < NUM_DMA_BDS {
            if let Some(tile) = self.array.get_mut(col, row) {
                let bd = &mut tile.dma_bds[bd_idx];
                match word {
                    0 => bd.addr_low = value,  // Actually length for MemTile
                    1 => bd.addr_high = value, // Address + next_bd
                    2 => bd.length = value,    // D0 config
                    3 => bd.control = value,   // D1 config
                    4 => bd.d0 = value,        // D2 config
                    5 => bd.d1 = value,        // D3 config
                    // Words 6,7 don't fit in legacy struct
                    _ => {}
                }
            }
        }

        // Now parse and configure the DmaEngine
        // We need all 8 words to properly parse, so read back from tile
        let bd_config = self.parse_memtile_bd(col, row, bd_idx);

        if let Some(config) = bd_config {
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if let Err(e) = dma.configure_bd(bd_idx as u8, config.clone()) {
                    log::debug!("Failed to configure MemTile BD {} on DmaEngine ({},{}): {:?}",
                        bd_idx, col, row, e);
                } else if config.valid {
                    log::info!("CDO configured MemTile BD {} on tile ({},{}) addr=0x{:X} len={} acq={:?} rel={:?} next={:?}",
                        bd_idx, col, row, config.base_addr, config.length,
                        config.acquire_lock.map(|id| (id, config.acquire_value)),
                        config.release_lock.map(|id| (id, config.release_value)),
                        config.next_bd);
                }
            }
        }
    }

    /// Parse a MemTile BD from stored word values.
    fn parse_memtile_bd(&self, col: u8, row: u8, bd_idx: usize) -> Option<crate::device::dma::BdConfig> {
        use crate::device::dma::BdConfig;
        use super::registers_spec::mem_tile_module::bd as mt_bd;

        if bd_idx >= NUM_DMA_BDS {
            return None;
        }

        let tile = self.array.get(col, row)?;
        let bd = &tile.dma_bds[bd_idx];

        // Word 0: Buffer_Length[16:0]
        let buffer_length = bd.addr_low & mt_bd::WORD0_BUFFER_LEN_MASK; // 17 bits, in 32-bit words

        // Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20]
        let base_addr_words = bd.addr_high & mt_bd::WORD1_BASE_ADDR_MASK; // 19 bits, word address
        let use_next_bd = (bd.addr_high >> mt_bd::WORD1_USE_NEXT_BD_BIT) & 1 != 0;
        let next_bd_id = ((bd.addr_high >> mt_bd::WORD1_NEXT_BD_SHIFT) & mt_bd::WORD1_NEXT_BD_MASK) as u8;

        // Word 2: D0_Stepsize[16:0], D0_Wrap[26:17]
        let d0_stepsize = bd.length & mt_bd::WORD0_BUFFER_LEN_MASK; // Same 17-bit mask
        let d0_wrap = (bd.length >> 17) & 0x3FF;

        // We don't have word 6,7 in legacy struct, check if bd is valid by looking at d1 field
        // Word 7: Lock_Acq_ID[7:0], Lock_Acq_Value[14:8], Lock_Acq_Enable[15],
        //         Lock_Rel_ID[23:16], Lock_Rel_Value[30:24], Valid_BD[31]
        // Since we only have 6 words, we'll need to read this differently
        // For now, assume valid if we have any address/length
        let valid = buffer_length > 0 || base_addr_words > 0;

        // Convert word address to byte address
        let base_addr = (base_addr_words as u64) * 4;
        let length = buffer_length * 4; // Convert word count to bytes

        let mut config = BdConfig::simple_1d(base_addr, length);
        config.valid = valid;

        // Set D0 size and stride from wrap/stepsize
        if d0_wrap > 0 {
            config.d0.size = d0_wrap as u32;
            config.d0.stride = d0_stepsize as i32;
        }

        // BD chaining
        config.next_bd = if use_next_bd { Some(next_bd_id) } else { None };

        // We can't fully parse lock config without words 6-7, but let's check d1 field
        // which might have been repurposed to store lock info during bulk writes
        // For now, mark as needing extended BD storage

        Some(config)
    }

    /// Write MemTile BD data from a byte array.
    ///
    /// MemTile DMA_WRITE commands write 32 bytes (8 words) per BD.
    fn dma_write_memtile_bd_data(&mut self, col: u8, row: u8, offset: u32, data: &[u8]) {
        // Parse into 32-bit words
        let words: Vec<u32> = data
            .chunks(4)
            .map(|chunk| {
                let mut arr = [0u8; 4];
                arr[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(arr)
            })
            .collect();

        // Log words for debugging BD parsing
        if words.len() >= 8 {
            log::info!("MemTile BD raw: tile({},{}) offset=0x{:05X} word7=0x{:08X} words=[0x{:08X}, 0x{:08X}..., 0x{:08X}]",
                col, row, offset, words.get(7).copied().unwrap_or(0),
                words.get(0).copied().unwrap_or(0), words.get(1).copied().unwrap_or(0), words.get(7).copied().unwrap_or(0));
        }

        // MemTile BDs: base, 8 words per BD
        use super::registers_spec::mem_tile_module as mt;
        let rel = offset - mt::DMA_BD_BASE;
        let bd_idx = (rel / mt::DMA_BD_STRIDE) as usize;

        // Store words in tile BD structure (limited to first 6 for legacy compat)
        if bd_idx < NUM_DMA_BDS {
            if let Some(tile) = self.array.get_mut(col, row) {
                let bd = &mut tile.dma_bds[bd_idx];
                if words.len() > 0 { bd.addr_low = words[0]; }
                if words.len() > 1 { bd.addr_high = words[1]; }
                if words.len() > 2 { bd.length = words[2]; }
                if words.len() > 3 { bd.control = words[3]; }
                if words.len() > 4 { bd.d0 = words[4]; }
                if words.len() > 5 { bd.d1 = words[5]; }
            }
        }

        // Parse using full 8 words and configure DmaEngine
        self.configure_memtile_bd_from_words(col, row, bd_idx, &words);
    }

    /// Configure MemTile BD from raw words.
    fn configure_memtile_bd_from_words(&mut self, col: u8, row: u8, bd_idx: usize, words: &[u32]) {
        use crate::device::dma::BdConfig;
        use super::registers_spec::mem_tile_module::bd as mt_bd;

        if words.is_empty() {
            return;
        }

        // Parse MemTile BD format (8 words)
        // Word 0: Buffer_Length[16:0]
        let buffer_length = words.get(0).copied().unwrap_or(0) & mt_bd::WORD0_BUFFER_LEN_MASK;

        // Word 1: Base_Address[18:0], Use_Next_BD[19], Next_BD[25:20]
        let word1 = words.get(1).copied().unwrap_or(0);
        let base_addr_words = word1 & mt_bd::WORD1_BASE_ADDR_MASK;
        let use_next_bd = (word1 >> mt_bd::WORD1_USE_NEXT_BD_BIT) & 1 != 0;
        let next_bd_id = ((word1 >> mt_bd::WORD1_NEXT_BD_SHIFT) & mt_bd::WORD1_NEXT_BD_MASK) as u8;

        // Word 2: D0_Stepsize[16:0], D0_Wrap[26:17]
        let word2 = words.get(2).copied().unwrap_or(0);
        let d0_stepsize = word2 & mt_bd::WORD0_BUFFER_LEN_MASK; // Same 17-bit mask
        let d0_wrap = (word2 >> 17) & 0x3FF;

        // Word 7: Lock config and Valid bit (if present)
        // Note: Lock IDs in MemTile BD encoding use 8 bits, but MemTile only has 64 locks.
        // Lock IDs 64+ appear to map back to 0+ (likely tile-relative), so we mask to 6 bits.
        use super::registers_spec::sign_extend_7bit;
        let word7 = words.get(7).copied().unwrap_or(0);
        let lock_acq_id = (word7 & 0x3F) as u8;  // Mask to 6 bits for 64 locks
        let lock_acq_value_raw = (word7 >> mt_bd::WORD7_LOCK_ACQ_VALUE_SHIFT) & mt_bd::WORD7_LOCK_ACQ_VALUE_MASK;
        let lock_acq_value = sign_extend_7bit(lock_acq_value_raw);
        let lock_acq_enable = (word7 >> mt_bd::WORD7_LOCK_ACQ_ENABLE_BIT) & 1 != 0;
        let lock_rel_id = ((word7 >> mt_bd::WORD7_LOCK_REL_ID_SHIFT) & 0x3F) as u8;  // Mask to 6 bits for 64 locks
        let lock_rel_value_raw = (word7 >> mt_bd::WORD7_LOCK_REL_VALUE_SHIFT) & mt_bd::WORD7_LOCK_REL_VALUE_MASK;
        let lock_rel_value = sign_extend_7bit(lock_rel_value_raw);
        let valid = (word7 >> mt_bd::WORD7_VALID_BD_BIT) & 1 != 0;

        // Convert to bytes
        let base_addr = (base_addr_words as u64) * 4;
        let length = buffer_length * 4;

        let mut config = BdConfig::simple_1d(base_addr, length);
        config.valid = valid || buffer_length > 0;

        // D0 config
        if d0_wrap > 0 {
            config.d0.size = d0_wrap;
            config.d0.stride = d0_stepsize as i32;
        }

        // BD chaining
        config.next_bd = if use_next_bd { Some(next_bd_id) } else { None };

        // Lock config
        if lock_acq_enable && words.len() >= 8 {
            config.acquire_lock = Some(lock_acq_id);
            config.acquire_value = lock_acq_value;
        }
        if lock_rel_value != 0 && words.len() >= 8 {
            config.release_lock = Some(lock_rel_id);
            config.release_value = lock_rel_value;
        }

        // Configure the DMA engine
        if let Some(dma) = self.array.dma_engine_mut(col, row) {
            // Log all MemTile BD configurations at debug level
            log::debug!("MemTile BD {} tile({},{}) valid={} addr=0x{:X} len={} word7=0x{:08X}",
                bd_idx, col, row, config.valid, base_addr, length, word7);

            if let Err(e) = dma.configure_bd(bd_idx as u8, config.clone()) {
                log::warn!("Failed to configure MemTile BD {} on DmaEngine ({},{}): {:?}",
                    bd_idx, col, row, e);
            } else if config.valid {
                log::info!("CDO configured MemTile BD {} on tile ({},{}) addr=0x{:X} len={} d0=({},{}) acq={:?} rel={:?} next={:?}",
                    bd_idx, col, row, base_addr, length, d0_wrap, d0_stepsize,
                    config.acquire_lock.map(|id| (id, config.acquire_value)),
                    config.release_lock.map(|id| (id, config.release_value)),
                    config.next_bd);
            }
        }
    }

    /// Write to a MemTile DMA channel register.
    ///
    /// MemTile has 6 S2MM + 6 MM2S channels:
    /// - S2MM 0-5 control: 0xA0600, 0xA0608, 0xA0610, 0xA0618, 0xA0620, 0xA0628
    /// - S2MM 0-5 queue:   0xA0604, 0xA060C, 0xA0614, 0xA061C, 0xA0624, 0xA062C
    /// - MM2S 0-5 control: 0xA0630, 0xA0638, 0xA0640, 0xA0648, 0xA0650, 0xA0658
    /// - MM2S 0-5 queue:   0xA0634, 0xA063C, 0xA0644, 0xA064C, 0xA0654, 0xA065C
    fn write_memtile_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        // Determine channel index and whether this is a start queue write
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= 12 {
            return;
        }

        // Update tile DMA channel state
        if let Some(tile) = self.array.get_mut(col, row) {
            if ch_idx < NUM_DMA_CHANNELS {
                let ch = &mut tile.dma_channels[ch_idx];
                if is_start_queue {
                    ch.start_queue = value;
                    ch.current_bd = (value & 0x3F) as u8; // 6-bit BD index for MemTile
                } else {
                    ch.control = value;
                    ch.running = value & 1 != 0;
                }
            }
        }

        // Start DmaEngine channel when START_QUEUE is written
        if is_start_queue {
            let bd_idx = (value & 0x3F) as u8; // 6-bit BD index
            let repeat_count = ((value >> 16) & 0xFF) as u8;
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if let Err(e) = dma.start_channel_with_repeat(ch_idx as u8, bd_idx, repeat_count) {
                    log::warn!("Failed to start MemTile DMA channel {} on tile ({},{}): {:?}",
                        ch_idx, col, row, e);
                } else {
                    let dir = if ch_idx < 6 { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < 6 { ch_idx } else { ch_idx - 6 };
                    log::info!("CDO started MemTile DMA {} ch {} with BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Masked write to MemTile DMA channel register.
    fn mask_write_memtile_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= 12 {
            return;
        }

        let new_start_queue = if let Some(tile) = self.array.get_mut(col, row) {
            if ch_idx < NUM_DMA_CHANNELS {
                let ch = &mut tile.dma_channels[ch_idx];
                if is_start_queue {
                    ch.start_queue = (ch.start_queue & !mask) | (value & mask);
                    Some(ch.start_queue)
                } else {
                    ch.control = (ch.control & !mask) | (value & mask);
                    ch.running = ch.control & 1 != 0;
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Start channel if START_QUEUE was written
        if let Some(queue_val) = new_start_queue {
            let bd_idx = (queue_val & 0x3F) as u8;
            let repeat_count = ((queue_val >> 16) & 0xFF) as u8;
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if let Err(e) = dma.start_channel_with_repeat(ch_idx as u8, bd_idx, repeat_count) {
                    log::warn!("Failed to start MemTile DMA channel {} on tile ({},{}): {:?}",
                        ch_idx, col, row, e);
                } else {
                    let dir = if ch_idx < 6 { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < 6 { ch_idx } else { ch_idx - 6 };
                    log::info!("CDO started MemTile DMA {} ch {} with BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Decode MemTile DMA channel offset into (channel_index, is_start_queue).
    fn decode_memtile_channel_offset(&self, offset: u32) -> (usize, bool) {
        use super::registers_spec::mem_tile_module as mt;
        // S2MM channels 0-5: control at base+n*8, queue at base+4+n*8
        // MM2S channels 0-5: control at base+n*8, queue at base+4+n*8

        if (mt::DMA_CHANNEL_S2MM_BASE..mt::DMA_CHANNEL_MM2S_BASE).contains(&offset) {
            // S2MM range (channels 0-5)
            let rel = offset - mt::DMA_CHANNEL_S2MM_BASE;
            let ch = (rel / 8) as usize;
            let is_queue = (rel % 8) >= 4;
            (ch, is_queue)
        } else if (mt::DMA_CHANNEL_MM2S_BASE..mt::DMA_CHANNEL_MM2S_BASE + 0x30).contains(&offset) {
            // MM2S range - channel indices 6-11
            let rel = offset - mt::DMA_CHANNEL_MM2S_BASE;
            let ch = 6 + (rel / 8) as usize;
            let is_queue = (rel % 8) >= 4;
            (ch, is_queue)
        } else {
            (usize::MAX, false)
        }
    }

    /// Write to MemTile stream switch.
    ///
    /// MemTile stream switch registers are at 0xB0000.
    /// Master config format: bit 31 = enable, bits 4:0 = slave select
    /// Slave config format: bit 31 = enable
    fn write_memtile_stream_switch(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use super::aie2_spec::stream_switch::{ENABLE_BIT, SLAVE_SELECT_MASK};
        use super::registers_spec::mem_tile_module as mt;

        // MemTile master ports: base + (port * 4), slave ports: base + (port * 4)
        if (mt::STREAM_SWITCH_MASTER_BASE..mt::STREAM_SWITCH_MASTER_END).contains(&offset) {
            let port = ((offset - mt::STREAM_SWITCH_MASTER_BASE) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let slave_select = (value & SLAVE_SELECT_MASK) as usize;

            // Get port type before mutable borrow
            let port_type = self.array.get(col, row)
                .and_then(|t| t.stream_switch.masters.get(port))
                .map(|p| p.port_type);

            let tile = match self.array.get_mut(col, row) {
                Some(t) => t,
                None => return,
            };

            // Store config in port if within range
            if port < tile.stream_switch.masters.len() {
                tile.stream_switch.masters[port].config = value;
                tile.stream_switch.masters[port].enabled = enable;
            }

            log::debug!("MemTile ({},{}) stream switch master[{}] = 0x{:08X} (en={}, slave={})",
                col, row, port, value, enable, slave_select);

            // If enabled, create a local route from the selected slave to this master
            if enable && slave_select < tile.stream_switch.slaves.len() && port < tile.stream_switch.masters.len() {
                tile.stream_switch.configure_local_route(slave_select, port);
                log::info!("MemTile ({},{}) local route: slave[{}] -> master[{}]",
                    col, row, slave_select, port);
            }

            // Derive global route for external ports (North/South)
            if enable {
                if let Some(pt) = port_type {
                    self.derive_global_route(col, row, port as u8, pt);
                }
            }
        } else if (mt::STREAM_SWITCH_SLAVE_BASE..mt::STREAM_SWITCH_SLAVE_END).contains(&offset) {
            let port = ((offset - mt::STREAM_SWITCH_SLAVE_BASE) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;

            let tile = match self.array.get_mut(col, row) {
                Some(t) => t,
                None => return,
            };

            // Store config in port if within range
            if port < tile.stream_switch.slaves.len() {
                tile.stream_switch.slaves[port].config = value;
                tile.stream_switch.slaves[port].enabled = enable;
            }

            log::debug!("MemTile ({},{}) stream switch slave[{}] = 0x{:08X} (en={})",
                col, row, port, value, enable);
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
        // AIE-ML lock registers are 16 bytes apart (0x10 spacing per lock)
        let addr = TileAddress::encode(1, 2, 0x1F050); // Lock 5 = 0x1F000 + 5*0x10
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
