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

use super::arch_config::{ArchConfig, ModelConfig};
use super::array::TileArray;
use super::registers::{RegisterModule, TileAddress};
use super::tile::{Tile, TileType};
use crate::parser::cdo::{Cdo, CdoCommand};

/// Sign-extend a lock value from a register u32 to i8.
///
/// Delegates to DeviceRegLayout::sign_extend_lock_value() which derives the
/// field width from the AM025 register database (6 bits for AIE2).
fn sign_extend_lock_value(reg_layout: &super::regdb::DeviceRegLayout, raw: u32) -> i8 {
    reg_layout.sign_extend_lock_value(raw)
}

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
        Self::new(ModelConfig::npu1())
    }

    /// Create an NPU2 device state.
    #[inline]
    pub fn new_npu2() -> Self {
        Self::new(ModelConfig::npu2())
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

        if self.stats.unknown > 0 {
            log::warn!(
                "CDO application complete: {} commands processed, {} unknown opcodes skipped",
                self.stats.commands,
                self.stats.unknown
            );
        }

        Ok(())
    }

    /// Write a register via the full module-dispatch path, reconstructing the
    /// 32-bit tile address from (col, row, offset).
    ///
    /// Control packets arrive at individual tiles with a tile-local register
    /// offset. The tile's own `write_register()` only handles a subset of
    /// register types (compute-tile DMA BDs, locks, channel control). This
    /// method encodes the full address and routes through
    /// `DeviceState::write_register()`, which dispatches to the correct
    /// module handler (MemTile DMA BDs, MemTile stream switch, core module,
    /// shim DMA channels, etc.).
    pub fn ctrl_packet_write(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let address = TileAddress::encode(col, row, offset);
        log::debug!("ctrl_packet_write: tile({},{}) offset=0x{:05X} value=0x{:08X} -> addr=0x{:08X}",
            col, row, offset, value, address);
        if let Err(e) = self.write_register(address, value) {
            log::error!("ctrl_packet_write failed: tile({},{}) offset=0x{:05X}: {:?}",
                col, row, offset, e);
        }
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
                log::trace!("CDO MaskWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} mask=0x{:08X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
                self.mask_write_register(*address, *mask, *value)?;
            }

            // Write64/MaskWrite64 use 64-bit addresses but AIE tiles are 32-bit addressed.
            // The high 32 bits are always 0 for AIE, so we use the low 32 bits.
            CdoCommand::Write64 { address, value } => {
                self.stats.writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!("CDO Write64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X} value=0x{:08X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, value);
                self.write_register(addr32, *value)?;
            }

            CdoCommand::MaskWrite64 { address, mask, value } => {
                self.stats.mask_writes += 1;
                let addr32 = *address as u32;
                let tile_addr = TileAddress::decode(addr32);
                log::trace!("CDO MaskWrite64: addr=0x{:016X} -> tile({},{}) offset=0x{:05X}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset);
                self.mask_write_register(addr32, *mask, *value)?;
            }

            CdoCommand::DmaWrite { address, data } => {
                self.stats.dma_writes += 1;
                let tile_addr = TileAddress::decode(*address);
                log::debug!("CDO DmaWrite: addr=0x{:08X} -> tile({},{}) offset=0x{:05X} module={:?} len={}",
                    address, tile_addr.col, tile_addr.row, tile_addr.offset, tile_addr.module(), data.len());
                self.dma_write(*address, data)?;
            }

            // Synchronization/timing commands - no-ops in emulation.
            // MaskPoll waits for a register to match a value on real hardware;
            // in the emulator, configuration writes take effect immediately.
            CdoCommand::MaskPoll { .. } | CdoCommand::MaskPoll64 { .. } => {
                log::trace!("CDO MaskPoll: skipped (emulator writes are synchronous)");
            }

            // Delay inserts a wait on real hardware; no-op in emulation.
            CdoCommand::Delay { .. } => {
                log::trace!("CDO Delay: skipped (emulator has no real-time clock)");
            }

            // Structural markers - no functional effect.
            CdoCommand::EndMark | CdoCommand::Marker { .. } => {
                log::trace!("CDO marker/end: skipped");
            }

            CdoCommand::Unknown { opcode, payload } => {
                self.stats.unknown += 1;
                anyhow::bail!(
                    "CDO opcode {:#06x} not implemented ({} payload words) -- unknown hardware config",
                    opcode,
                    payload.len(),
                );
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
        let reg_layout = super::regdb::device_reg_layout();

        match tile_addr.module() {
            RegisterModule::Locks => {
                Self::write_lock_value(reg_layout, tile, tile_addr,
                    reg_layout.memory_lock_base, reg_layout.memory_lock_stride, value, false);
            }

            RegisterModule::MemTileLocks => {
                Self::write_lock_value(reg_layout, tile, tile_addr,
                    reg_layout.memtile_lock_base, reg_layout.memtile_lock_stride, value, true);
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
                // Shim DMA channels live at a different offset (0x1D200) than
                // compute channels (0x1DE00). RegisterModule::from_offset()
                // is row-agnostic and cannot distinguish them, so we catch
                // shim channel writes here.
                let reg_layout = super::regdb::device_reg_layout();
                let shim_ch_base = reg_layout.shim_channel_base;
                let shim_ch_end = shim_ch_base + 4 * reg_layout.shim_channel_stride;
                if tile_addr.row == 0 && (shim_ch_base..shim_ch_end).contains(&tile_addr.offset) {
                    self.write_shim_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                }
            }
        }

        // Forward all writes to tile.write_register() for general handling:
        // register HashMap storage, trace config, edge detection, event port
        // selection, cascade config, event broadcast, and Event_Generate.
        // The specialized handlers above extract structured data (BD fields,
        // channel control, etc.); this ensures the tile-level handlers also
        // run for every write.
        if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
            tile.write_register(tile_addr.offset, value);
        }

        // Propagate broadcast events to all tiles in the column.
        self.propagate_broadcasts(tile_addr.col, tile_addr.row);

        Ok(())
    }

    /// Masked write to a register.
    fn mask_write_register(&mut self, address: u32, mask: u32, value: u32) -> Result<()> {
        let tile_addr = TileAddress::decode(address);
        let module = tile_addr.module();

        log::trace!("mask_write_register: addr=0x{:08X} tile({},{}) offset=0x{:05X} module={:?}",
            address, tile_addr.col, tile_addr.row, tile_addr.offset, module);

        // Get the tile
        let tile = match self.array.get_mut(tile_addr.col, tile_addr.row) {
            Some(t) => t,
            None => {
                log::trace!("mask_write_register: tile({},{}) not in array", tile_addr.col, tile_addr.row);
                return Ok(());
            }
        };

        // For most registers, we can just apply the masked value directly
        // since we're initializing from zero state
        let reg_layout = super::regdb::device_reg_layout();

        match module {
            RegisterModule::Locks => {
                // Lock offsets (0x1F000+) collide with Shim Mux/Demux Config
                // (0x1F000/0x1F004). For shim tiles, route through write_register()
                // so the Shim Mux parser fires. For other tiles, handle as lock write.
                if tile.is_shim() {
                    let current = *tile.registers_ref().get(&tile_addr.offset).unwrap_or(&0);
                    let new_value = (current & !mask) | (value & mask);
                    tile.write_register(tile_addr.offset, new_value);
                } else {
                    Self::mask_write_lock_value(reg_layout, tile, tile_addr.offset,
                        reg_layout.memory_lock_base, reg_layout.memory_lock_stride, mask, value);
                }
            }

            RegisterModule::MemTileLocks => {
                Self::mask_write_lock_value(reg_layout, tile, tile_addr.offset,
                    reg_layout.memtile_lock_base, reg_layout.memtile_lock_stride, mask, value);
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
                // Shim DMA channels at 0x1D200 (see write_register for details).
                let shim_ch_base = reg_layout.shim_channel_base;
                let shim_ch_end = shim_ch_base + 4 * reg_layout.shim_channel_stride;
                if tile_addr.row == 0 && (shim_ch_base..shim_ch_end).contains(&tile_addr.offset) {
                    self.mask_write_shim_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
                } else {
                    // Read-modify-write: preserve bits not covered by mask.
                    // Multiple MaskWrites to the same register (e.g., Shim Mux config at
                    // 0x1F000) must accumulate rather than clobber each other.
                    if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
                        let current = *tile.registers_ref().get(&tile_addr.offset).unwrap_or(&0);
                        let new_value = (current & !mask) | (value & mask);
                        log::debug!("CDO MaskWrite RMW: tile({},{}) offset=0x{:05X} current=0x{:08X} -> 0x{:08X} (mask=0x{:08X} val=0x{:08X})",
                            tile_addr.col, tile_addr.row, tile_addr.offset, current, new_value, mask, value);
                        tile.write_register(tile_addr.offset, new_value);
                    } else {
                        anyhow::bail!(
                            "CDO MaskWrite: tile({},{}) not found for offset=0x{:05X} -- config targets missing tile",
                            tile_addr.col, tile_addr.row, tile_addr.offset,
                        );
                    }
                }
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
    /// Lock_Value field width is derived from the register database.
    fn write_lock_value(
        reg_layout: &super::regdb::DeviceRegLayout,
        tile: &mut Tile,
        tile_addr: TileAddress,
        base: u32,
        stride: u32,
        value: u32,
        is_memtile: bool,
    ) {
        let lock_idx = ((tile_addr.offset - base) / stride) as usize;
        if lock_idx < tile.locks.len() {
            let signed = sign_extend_lock_value(reg_layout, value);
            tile.locks[lock_idx].set(signed);
            if value != 0 {
                let tile_type = if is_memtile { "MemTile" } else { "Compute" };
                log::info!("CDO init {} lock {} on tile ({},{}) = {}",
                    tile_type, lock_idx, tile_addr.col, tile_addr.row, signed);
            }
        }
    }

    /// Masked write to a lock value.
    fn mask_write_lock_value(
        reg_layout: &super::regdb::DeviceRegLayout,
        tile: &mut Tile,
        offset: u32,
        base: u32,
        stride: u32,
        mask: u32,
        value: u32,
    ) {
        let lock_idx = ((offset - base) / stride) as usize;
        if lock_idx < tile.locks.len() {
            // Read current value in unsigned representation for mask ops
            let current_raw = (tile.locks[lock_idx].value as u8 & reg_layout.lock_value_mask as u8) as u32;
            let new_raw = (current_raw & !mask) | (value & mask);
            let signed = sign_extend_lock_value(reg_layout, new_raw);
            tile.locks[lock_idx].set(signed);
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

    /// Write to a DMA buffer descriptor (compute or shim tile).
    ///
    /// Compute tile BDs have 6 words (14-bit address, 14-bit length).
    /// Shim tile BDs have 8 words (46-bit DDR address, 32-bit length).
    /// Both share the same BD base address (0x1D000) and stride (0x20),
    /// so we determine tile type from the row to select the correct parser.
    fn write_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile_type = self.array.arch().tile_type(col, row);
        let reg_layout = super::regdb::device_reg_layout();

        // Select base/stride/max words based on tile type.
        // Shim and compute share the same BD base and stride, but shim uses
        // 8 words per BD while compute uses 6.
        let (bd_base, bd_stride, max_words) = match tile_type {
            TileType::Shim => (reg_layout.shim_bd_base, reg_layout.shim_bd_stride, reg_layout.shim_bd_words),
            _ => (reg_layout.memory_bd_base, reg_layout.memory_bd_stride, reg_layout.memory_bd_words),
        };

        let rel = offset - bd_base;
        let bd_idx = (rel / bd_stride) as usize;
        let word = ((rel % bd_stride) / 4) as usize;

        let num_bds = self.array.get(col, row).map_or(0, |t| t.dma_bds.len());
        if bd_idx >= num_bds || word >= max_words {
            return;
        }

        // Update the legacy tile BD storage (CDO writes one word at a time).
        // The legacy struct has 6 fields; for shim BDs, words 6-7 are stored
        // in the tile's register map and reconstructed at parse time.
        if let Some(tile) = self.array.get_mut(col, row) {
            let bd = &mut tile.dma_bds[bd_idx];
            match word {
                0 => bd.addr_low = value,
                1 => bd.addr_high = value,
                2 => bd.length = value,
                3 => bd.control = value,
                4 => bd.d0 = value,
                5 => bd.d1 = value,
                // Words 6-7 (shim): store in the tile register map
                w if w >= 6 => {
                    let reg_off = bd_base + (bd_idx as u32) * bd_stride + (w as u32) * 4;
                    tile.write_register(reg_off, value);
                }
                _ => {}
            }
        }

        // Mark BD dirty -- it will be re-parsed from raw words when the
        // DMA channel start queue is written. This avoids corrupt intermediate
        // configs when control packets write one word at a time.
        if let Some(dma) = self.array.dma_engine_mut(col, row) {
            dma.mark_bd_dirty(bd_idx as u8);
        }

        log::trace!("  BD {} word {} tile({},{}) marked dirty (deferred parse)",
            bd_idx, word, col, row);
    }

    /// Write to a DMA channel register.
    fn write_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memory_channel;

        // Channel registers: base/stride derived from register database
        // Layout: S2MM_0, S2MM_1, MM2S_0, MM2S_1
        // Each channel has CTRL register and START_QUEUE register (4 bytes each)
        let rel = offset - reg_layout.memory_channel_base;
        let ch_idx = (rel / reg_layout.memory_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.memory_channel_stride) >= 4;
        let is_s2mm = ch_idx < 2; // First 2 channels are S2MM

        let num_channels = self.array.get(col, row).map_or(0, |t| t.dma_channels.len());
        if ch_idx >= num_channels {
            return;
        }

        // Parse enable_token_issue before updating tile state
        let enable_token_issue = if is_start_queue {
            lay.enable_token_issue.extract_bool(value)
        } else {
            false
        };

        // Parse compression/out-of-order bits before tile update
        let compression_enable = lay.decompression_enable.extract_bool(value);
        let out_of_order_enable = lay.enable_out_of_order.extract_bool(value);

        // Update the tile's DMA channel state
        if let Some(tile) = self.array.get_mut(col, row) {
            let dma_ch = &mut tile.dma_channels[ch_idx];
            if is_start_queue {
                dma_ch.start_queue = value;
                dma_ch.current_bd = lay.start_bd_id.extract(value) as u8;
                dma_ch.enable_token_issue = enable_token_issue;
            } else {
                dma_ch.control = value;
                dma_ch.running = value & 1 != 0;
                dma_ch.controller_id = lay.controller_id.extract(value) as u8;
                // Parse FoT mode (S2MM only, but harmless to store for MM2S)
                if is_s2mm {
                    dma_ch.fot_mode = lay.fot_mode.extract(value) as u8;
                    // S2MM: decompression enable (bit 4), out-of-order enable (bit 3)
                    dma_ch.decompression_enable = compression_enable;
                    dma_ch.out_of_order_enable = out_of_order_enable;
                } else {
                    // MM2S: compression enable (bit 4)
                    dma_ch.compression_enable = compression_enable;
                }
            }
        }

        // Update DMA engine compression/OOO config when control register is written
        if !is_start_queue {
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                let (decompression_en, ooo_en) = if is_s2mm {
                    (compression_enable, out_of_order_enable)
                } else {
                    (false, false)
                };
                let compress_en = if is_s2mm { false } else { compression_enable };

                dma.set_channel_compression_config(
                    ch_idx as u8,
                    compress_en,
                    decompression_en,
                    ooo_en,
                );
            }
        }

        // Start the DmaEngine channel when START_QUEUE is written
        if is_start_queue {
            let bd_idx = lay.start_bd_id.extract(value) as u8;
            let repeat_count = lay.repeat_count.extract(value) as u8;

            // Read controller_id and fot_mode from the tile's channel state
            // (which was set earlier when the control register was written)
            let (controller_id, fot_mode) = self.array.get(col, row)
                .map(|t| {
                    let dma_ch = &t.dma_channels[ch_idx];
                    (dma_ch.controller_id, dma_ch.fot_mode)
                })
                .unwrap_or((0, 0));

            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                // Set task config before enqueuing the task
                dma.set_channel_task_config(ch_idx as u8, enable_token_issue, controller_id, fot_mode);

                // Enqueue to the channel's task queue (hardware has 8-deep queue).
                // Unlike start_channel_with_repeat(), this never rejects a busy
                // channel -- it queues the task for execution when the current
                // transfer finishes.
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, enable_token_issue) {
                    dma.fatal_errors.push(format!(
                        "DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else if enable_token_issue {
                    log::debug!("CDO enqueued DMA channel {} BD {} repeat={} token_issue=true controller_id={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, controller_id, col, row);
                } else {
                    log::info!("CDO enqueued DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Masked write to a DMA channel register.
    fn mask_write_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memory_channel;
        let rel = offset - reg_layout.memory_channel_base;
        let ch_idx = (rel / reg_layout.memory_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.memory_channel_stride) >= 4;

        let num_channels = self.array.get(col, row).map_or(0, |t| t.dma_channels.len());
        if ch_idx >= num_channels {
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

        // Enqueue to the DMA channel's task queue when START_QUEUE is written.
        // Field positions derived from AM025 via ChannelFieldLayout.
        if let Some(queue_val) = new_start_queue {
            let bd_idx = lay.start_bd_id.extract(queue_val) as u8;
            let repeat_count = lay.repeat_count.extract(queue_val) as u8;
            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    dma.fatal_errors.push(format!(
                        "DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else {
                    log::info!("CDO enqueued DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Write to a shim DMA channel register.
    ///
    /// Shim tiles have the same 2 S2MM + 2 MM2S channel layout as compute
    /// tiles, but at a different base address (0x1D200 vs 0x1DE00).
    /// The register bit layout is the same (same ChannelFieldLayout).
    fn write_shim_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memory_channel; // Same bit layout as compute

        let rel = offset - reg_layout.shim_channel_base;
        let ch_idx = (rel / reg_layout.shim_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.shim_channel_stride) >= 4;
        let is_s2mm = ch_idx < 2;

        let num_channels = self.array.get(col, row).map_or(0, |t| t.dma_channels.len());
        if ch_idx >= num_channels {
            return;
        }

        let enable_token_issue = if is_start_queue {
            lay.enable_token_issue.extract_bool(value)
        } else {
            false
        };

        let compression_enable = lay.decompression_enable.extract_bool(value);
        let out_of_order_enable = lay.enable_out_of_order.extract_bool(value);

        if let Some(tile) = self.array.get_mut(col, row) {
            let dma_ch = &mut tile.dma_channels[ch_idx];
            if is_start_queue {
                dma_ch.start_queue = value;
                dma_ch.current_bd = lay.start_bd_id.extract(value) as u8;
                dma_ch.enable_token_issue = enable_token_issue;
            } else {
                dma_ch.control = value;
                dma_ch.running = value & 1 != 0;
                dma_ch.controller_id = lay.controller_id.extract(value) as u8;
                if is_s2mm {
                    dma_ch.fot_mode = lay.fot_mode.extract(value) as u8;
                    dma_ch.decompression_enable = compression_enable;
                    dma_ch.out_of_order_enable = out_of_order_enable;
                } else {
                    dma_ch.compression_enable = compression_enable;
                }
            }
        }

        if !is_start_queue {
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                let (decompression_en, ooo_en) = if is_s2mm {
                    (compression_enable, out_of_order_enable)
                } else {
                    (false, false)
                };
                let compress_en = if is_s2mm { false } else { compression_enable };
                dma.set_channel_compression_config(ch_idx as u8, compress_en, decompression_en, ooo_en);
            }
        }

        if is_start_queue {
            let bd_idx = lay.start_bd_id.extract(value) as u8;
            let repeat_count = lay.repeat_count.extract(value) as u8;

            let (controller_id, fot_mode) = self.array.get(col, row)
                .map(|t| {
                    let dma_ch = &t.dma_channels[ch_idx];
                    (dma_ch.controller_id, dma_ch.fot_mode)
                })
                .unwrap_or((0, 0));

            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                dma.set_channel_task_config(ch_idx as u8, enable_token_issue, controller_id, fot_mode);

                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, enable_token_issue) {
                    dma.fatal_errors.push(format!(
                        "Shim DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else {
                    log::info!("CDO enqueued Shim DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Masked write to a shim DMA channel register.
    fn mask_write_shim_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memory_channel; // Same bit layout as compute
        let rel = offset - reg_layout.shim_channel_base;
        let ch_idx = (rel / reg_layout.shim_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.shim_channel_stride) >= 4;

        let num_channels = self.array.get(col, row).map_or(0, |t| t.dma_channels.len());
        if ch_idx >= num_channels {
            return;
        }

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

        if let Some(queue_val) = new_start_queue {
            let bd_idx = lay.start_bd_id.extract(queue_val) as u8;
            let repeat_count = lay.repeat_count.extract(queue_val) as u8;
            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    dma.fatal_errors.push(format!(
                        "Shim DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else {
                    log::info!("CDO enqueued Shim DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, col, row);
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
        let ss = &super::regdb::device_reg_layout().memory_stream_switch;

        // Master ports: base + (port * 4), Slave ports: base + (port * 4)
        if (ss.master_base..ss.master_end).contains(&offset) {
            let port = ((offset - ss.master_base) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let packet_enable = (value >> 30) & 1 != 0;
            let slave_select = (value & SLAVE_SELECT_MASK) as usize;

            if let Some(tile) = self.array.get_mut(col, row) {
                if port < tile.stream_switch.masters.len() {
                    tile.stream_switch.masters[port].config = value;
                    tile.stream_switch.masters[port].enabled = enable;

                    // Always configure packet mode settings from the register
                    tile.stream_switch.configure_master_packet(port, value);

                    if packet_enable {
                        // Packet mode: routing is done by arbiter/msel, not circuit route
                        let (dh, arb, msel) = tile.stream_switch.master_packet_cfg(port)
                            .map_or((false, 0, 0), |c| (c.drop_header, c.arbiter, c.msel_enable));
                        log::info!("Tile ({},{}) stream switch: master[{}] packet mode = 0x{:08X} (drop_hdr={} arb={} msel_en=0b{:04b})",
                            col, row, port, value, dh, arb, msel);
                    } else if enable && slave_select < tile.stream_switch.slaves.len() {
                        // Circuit mode: configure local route
                        tile.stream_switch.configure_local_route(slave_select, port);
                        log::info!("Tile ({},{}) stream switch: master[{}] circuit mode slave_select={} (local route: slave[{}] -> master[{}])",
                            col, row, port, slave_select, slave_select, port);
                    } else if enable {
                        log::info!("Tile ({},{}) stream switch: master[{}] enabled with slave_select={} (no local route - slave out of range)",
                            col, row, port, slave_select);
                    }
                }
            }

        // Slave ports
        } else if (ss.slave_base..ss.slave_end).contains(&offset) {
            let port = ((offset - ss.slave_base) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let packet_enable = (value >> 30) & 1 != 0;

            if let Some(tile) = self.array.get_mut(col, row) {
                if port < tile.stream_switch.slaves.len() {
                    tile.stream_switch.slaves[port].config = value;
                    tile.stream_switch.slaves[port].enabled = enable;
                    tile.stream_switch.slaves[port].packet_enable = packet_enable;

                    if packet_enable {
                        log::info!("Tile ({},{}) stream switch: slave[{}] packet mode enabled",
                            col, row, port);
                    }
                }
            }
        // Slave slot registers (packet routing config per slave port)
        } else if (ss.slave_slot_base..ss.slave_slot_end).contains(&offset) {
            let slot_offset = offset - ss.slave_slot_base;
            let slave_port = (slot_offset / ss.slave_slot_port_stride) as usize;
            let slot = ((slot_offset % ss.slave_slot_port_stride) / 4) as usize;

            if let Some(tile) = self.array.get_mut(col, row) {
                tile.stream_switch.configure_slave_slot(slave_port, slot, value);
                log::info!("Tile ({},{}) stream switch: slave[{}] slot[{}] = 0x{:08X}",
                    col, row, slave_port, slot, value);
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
        let reg_layout = super::regdb::device_reg_layout();

        // MemTile BDs: base/stride derived from register database
        let rel = offset - reg_layout.memtile_bd_base;
        let bd_idx = (rel / reg_layout.memtile_bd_stride) as usize;
        let word = ((rel % reg_layout.memtile_bd_stride) / 4) as usize;

        // Bounds check against tile's actual BD count and words per BD
        let num_bds = self.array.get(col, row).map_or(0, |t| t.dma_bds.len());
        if bd_idx >= num_bds || word >= reg_layout.memtile_bd_words {
            return;
        }

        // Store raw BD words in the tile's BD structure (first 6 words only;
        // legacy struct has 6 fields). Words 6-7 are stored in the tile
        // register map (same approach as shim BDs).
        if let Some(tile) = self.array.get_mut(col, row) {
            let bd = &mut tile.dma_bds[bd_idx];
            match word {
                0 => bd.addr_low = value,
                1 => bd.addr_high = value,
                2 => bd.length = value,
                3 => bd.control = value,
                4 => bd.d0 = value,
                5 => bd.d1 = value,
                // Words 6-7: store in tile register map (lock config, valid bit)
                w if w >= 6 => {
                    let reg_off = reg_layout.memtile_bd_base
                        + (bd_idx as u32) * reg_layout.memtile_bd_stride
                        + (w as u32) * 4;
                    tile.write_register(reg_off, value);
                }
                _ => {}
            }
        }

        // Mark BD dirty -- it will be re-parsed from raw words when the
        // DMA channel start queue is written. This avoids corrupt intermediate
        // configs when control packets write one word at a time.
        if let Some(dma) = self.array.dma_engine_mut(col, row) {
            dma.mark_bd_dirty(bd_idx as u8);
        }

        log::trace!("  MemTile BD {} word {} tile({},{}) marked dirty (deferred parse)",
            bd_idx, word, col, row);
    }

    /// Parse MemTile BD words into a BdConfig using the data-driven parser.
    ///
    /// MemTile lock IDs are 8 bits in the register encoding, addressing a
    /// 192-entry cross-tile lock space (West 0-63, Own 64-127, East 128-191).
    /// The DMA engine's resolve_lock_id() maps these to local lock indices
    /// at runtime -- do NOT mask here.
    fn parse_memtile_bd_from_words(&self, words: &[u32]) -> crate::device::dma::BdConfig {
        use crate::device::dma::bd::BufferDescriptor;
        use crate::device::tile::TileType;

        let parsed = BufferDescriptor::from_registers(words, TileType::MemTile);
        let mut config = parsed.to_bd_config();

        // For backwards compatibility, mark as valid if any data is present
        if !config.valid && words.iter().any(|&w| w != 0) {
            config.valid = true;
        }

        config
    }

    /// Re-parse a dirty BD from tile raw storage into the DMA engine's BdConfig.
    ///
    /// Called just before the DMA engine snapshots a BD (enqueue_task), to ensure
    /// single-word control packet writes have been fully assembled. If the BD is
    /// not dirty, this is a no-op.
    fn reparse_dirty_bd(&mut self, col: u8, row: u8, bd_idx: usize) {
        use crate::device::dma::bd::BufferDescriptor;

        let is_dirty = self.array.dma_engine(col, row)
            .map_or(false, |dma| dma.is_bd_dirty(bd_idx as u8));
        if !is_dirty {
            return;
        }

        let tile_type = self.array.arch().tile_type(col, row);
        let reg_layout = super::regdb::device_reg_layout();

        let bd_config = self.array.get(col, row).map(|tile| {
            if bd_idx >= tile.dma_bds.len() {
                return None;
            }
            let bd = &tile.dma_bds[bd_idx];

            match tile_type {
                TileType::MemTile => {
                    let w6_off = reg_layout.memtile_bd_base
                        + (bd_idx as u32) * reg_layout.memtile_bd_stride + 24;
                    let w7_off = reg_layout.memtile_bd_base
                        + (bd_idx as u32) * reg_layout.memtile_bd_stride + 28;
                    let w6 = *tile.registers_ref().get(&w6_off).unwrap_or(&0);
                    let w7 = *tile.registers_ref().get(&w7_off).unwrap_or(&0);
                    let words = [bd.addr_low, bd.addr_high, bd.length,
                                 bd.control, bd.d0, bd.d1, w6, w7];
                    Some(self.parse_memtile_bd_from_words(&words))
                }
                TileType::Shim => {
                    let w6_off = reg_layout.shim_bd_base
                        + (bd_idx as u32) * reg_layout.shim_bd_stride + 24;
                    let w7_off = reg_layout.shim_bd_base
                        + (bd_idx as u32) * reg_layout.shim_bd_stride + 28;
                    let w6 = *tile.registers_ref().get(&w6_off).unwrap_or(&0);
                    let w7 = *tile.registers_ref().get(&w7_off).unwrap_or(&0);
                    let words = [bd.addr_low, bd.addr_high, bd.length,
                                 bd.control, bd.d0, bd.d1, w6, w7];
                    let parsed = BufferDescriptor::from_registers(&words, TileType::Shim);
                    let mut config = parsed.to_bd_config();
                    if !config.valid && words.iter().any(|&w| w != 0) {
                        config.valid = true;
                    }
                    Some(config)
                }
                _ => {
                    // Compute tile: 6 words
                    let words = [bd.addr_low, bd.addr_high, bd.length,
                                 bd.control, bd.d0, bd.d1];
                    let parsed = BufferDescriptor::from_registers(&words, TileType::Compute);
                    let mut config = parsed.to_bd_config();
                    if !config.valid && words.iter().any(|&w| w != 0) {
                        config.valid = true;
                    }
                    Some(config)
                }
            }
        }).flatten();

        if let Some(config) = bd_config {
            log::debug!("Re-parsed dirty BD {} on tile ({},{}) addr=0x{:X} len={} valid={}",
                bd_idx, col, row, config.base_addr, config.length, config.valid);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                // configure_bd automatically clears the dirty flag
                let _ = dma.configure_bd(bd_idx as u8, config);
            }
        }
    }

    /// Write MemTile BD data from a byte array.
    ///
    /// MemTile DMA_WRITE commands write 32 bytes (8 words) per BD.
    /// This path has all 8 words available, so the data-driven parser
    /// can extract all fields including iteration and locks.
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

        if words.len() >= 8 {
            log::debug!("MemTile BD raw: tile({},{}) offset=0x{:05X} words=[{:08X?}]",
                col, row, offset, &words[..8]);
        }

        // MemTile BDs: base/stride derived from register database
        let reg_layout = super::regdb::device_reg_layout();
        let rel = offset - reg_layout.memtile_bd_base;
        let bd_idx = (rel / reg_layout.memtile_bd_stride) as usize;

        // Store words in tile BD structure (first 6 only -- legacy struct limit)
        if let Some(tile) = self.array.get_mut(col, row) {
            if bd_idx < tile.dma_bds.len() {
                let bd = &mut tile.dma_bds[bd_idx];
                if !words.is_empty() { bd.addr_low = words[0]; }
                if words.len() > 1 { bd.addr_high = words[1]; }
                if words.len() > 2 { bd.length = words[2]; }
                if words.len() > 3 { bd.control = words[3]; }
                if words.len() > 4 { bd.d0 = words[4]; }
                if words.len() > 5 { bd.d1 = words[5]; }
            }
        }

        // Parse using data-driven parser with all 8 words available
        if words.len() >= 8 {
            let config = self.parse_memtile_bd_from_words(&words);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if let Err(e) = dma.configure_bd(bd_idx as u8, config.clone()) {
                    dma.fatal_errors.push(format!(
                        "Failed to configure MemTile BD {} on DmaEngine ({},{}): {:?}",
                        bd_idx, col, row, e,
                    ));
                } else if config.valid {
                    log::info!("CDO configured MemTile BD {} on tile ({},{}) addr=0x{:X} len={} d0=[{},{}] d1=[{},{}] acq={:?} rel={:?} next={:?} pkt={}(id={},type={})",
                        bd_idx, col, row, config.base_addr, config.length,
                        config.d0.size, config.d0.stride, config.d1.size, config.d1.stride,
                        config.acquire_lock.map(|id| (id, config.acquire_value)),
                        config.release_lock.map(|id| (id, config.release_value)),
                        config.next_bd,
                        config.enable_packet, config.packet_id, config.packet_type);
                }
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
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memtile_channel;

        // Determine channel index and whether this is a start queue write
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= 12 {
            return;
        }

        // Update tile DMA channel state
        if let Some(tile) = self.array.get_mut(col, row) {
            if ch_idx < tile.dma_channels.len() {
                let ch = &mut tile.dma_channels[ch_idx];
                if is_start_queue {
                    ch.start_queue = value;
                    ch.current_bd = lay.start_bd_id.extract(value) as u8;
                } else {
                    ch.control = value;
                    ch.running = value & 1 != 0;
                }
            }
        }

        // Enqueue to DMA channel's task queue when START_QUEUE is written
        if is_start_queue {
            let bd_idx = lay.start_bd_id.extract(value) as u8;
            let repeat_count = lay.repeat_count.extract(value) as u8;
            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    dma.fatal_errors.push(format!(
                        "MemTile DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else {
                    let dir = if ch_idx < 6 { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < 6 { ch_idx } else { ch_idx - 6 };
                    log::info!("CDO enqueued MemTile DMA {} ch {} BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Masked write to MemTile DMA channel register.
    fn mask_write_memtile_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let reg_layout = super::regdb::device_reg_layout();
        let lay = &reg_layout.memtile_channel;
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= 12 {
            return;
        }

        let new_start_queue = if let Some(tile) = self.array.get_mut(col, row) {
            if ch_idx < tile.dma_channels.len() {
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

        // Enqueue to task queue when START_QUEUE was written
        if let Some(queue_val) = new_start_queue {
            let bd_idx = lay.start_bd_id.extract(queue_val) as u8;
            let repeat_count = lay.repeat_count.extract(queue_val) as u8;
            // Re-parse BD if it was written word-by-word (control packets)
            self.reparse_dirty_bd(col, row, bd_idx as usize);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    dma.fatal_errors.push(format!(
                        "MemTile DMA channel {} task queue overflow on tile ({},{}) -- task lost",
                        ch_idx, col, row,
                    ));
                } else {
                    let dir = if ch_idx < 6 { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < 6 { ch_idx } else { ch_idx - 6 };
                    log::info!("CDO enqueued MemTile DMA {} ch {} BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Decode MemTile DMA channel offset into (channel_index, is_start_queue).
    fn decode_memtile_channel_offset(&self, offset: u32) -> (usize, bool) {
        let reg_layout = super::regdb::device_reg_layout();
        let s2mm_base = reg_layout.memtile_channel_s2mm_base;
        let mm2s_base = reg_layout.memtile_channel_mm2s_base;
        let stride = reg_layout.memtile_channel_stride;

        // S2MM channels 0-5: control at base+n*stride, queue at base+4+n*stride
        // MM2S channels 0-5: control at base+n*stride, queue at base+4+n*stride
        if (s2mm_base..mm2s_base).contains(&offset) {
            let rel = offset - s2mm_base;
            let ch = (rel / stride) as usize;
            let is_queue = (rel % stride) >= 4;
            (ch, is_queue)
        } else if (mm2s_base..mm2s_base + 6 * stride).contains(&offset) {
            // MM2S range - channel indices 6-11
            let rel = offset - mm2s_base;
            let ch = 6 + (rel / stride) as usize;
            let is_queue = (rel % stride) >= 4;
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
        let ss = &super::regdb::device_reg_layout().memtile_stream_switch;

        // MemTile master ports: base + (port * 4), slave ports: base + (port * 4)
        if (ss.master_base..ss.master_end).contains(&offset) {
            let port = ((offset - ss.master_base) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let packet_enable = (value >> 30) & 1 != 0;
            let slave_select = (value & SLAVE_SELECT_MASK) as usize;

            let tile = match self.array.get_mut(col, row) {
                Some(t) => t,
                None => return,
            };

            // Store config in port if within range
            if port < tile.stream_switch.masters.len() {
                tile.stream_switch.masters[port].config = value;
                tile.stream_switch.masters[port].enabled = enable;
            }

            // Always configure packet mode settings
            tile.stream_switch.configure_master_packet(port, value);

            if packet_enable {
                log::info!("MemTile ({},{}) stream switch master[{}] packet mode = 0x{:08X}",
                    col, row, port, value);
            } else {
                log::debug!("MemTile ({},{}) stream switch master[{}] = 0x{:08X} (en={}, slave={})",
                    col, row, port, value, enable, slave_select);

                // Circuit mode: create local route if enabled
                if enable && slave_select < tile.stream_switch.slaves.len() && port < tile.stream_switch.masters.len() {
                    tile.stream_switch.configure_local_route(slave_select, port);
                    log::info!("MemTile ({},{}) local route: slave[{}] -> master[{}]",
                        col, row, slave_select, port);
                }
            }

        } else if (ss.slave_base..ss.slave_end).contains(&offset) {
            let port = ((offset - ss.slave_base) / 4) as usize;
            let enable = (value >> ENABLE_BIT) & 1 != 0;
            let packet_enable = (value >> 30) & 1 != 0;

            let tile = match self.array.get_mut(col, row) {
                Some(t) => t,
                None => return,
            };

            // Store config in port if within range
            if port < tile.stream_switch.slaves.len() {
                tile.stream_switch.slaves[port].config = value;
                tile.stream_switch.slaves[port].enabled = enable;
                tile.stream_switch.slaves[port].packet_enable = packet_enable;
            }

            if packet_enable {
                log::info!("MemTile ({},{}) stream switch slave[{}] packet mode (0x{:08X})",
                    col, row, port, value);
            } else {
                log::debug!("MemTile ({},{}) stream switch slave[{}] = 0x{:08X} (en={})",
                    col, row, port, value, enable);
            }
        // Slave slot registers (packet routing config per slave port)
        } else if (ss.slave_slot_base..ss.slave_slot_end).contains(&offset) {
            let slot_offset = offset - ss.slave_slot_base;
            let slave_port = (slot_offset / ss.slave_slot_port_stride) as usize;
            let slot = ((slot_offset % ss.slave_slot_port_stride) / 4) as usize;

            if let Some(tile) = self.array.get_mut(col, row) {
                tile.stream_switch.configure_slave_slot(slave_port, slot, value);
                log::info!("MemTile ({},{}) stream switch: slave[{}] slot[{}] = 0x{:08X}",
                    col, row, slave_port, slot, value);
            }
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

    /// Get the architecture name (e.g. "AIE2", "AIE2P").
    #[inline]
    pub fn arch_name(&self) -> &str {
        self.array.arch().name()
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

    /// Propagate pending broadcast events from a tile to all tiles in its column.
    ///
    /// On real hardware, broadcast events travel through a dedicated broadcast
    /// network spanning the entire column. When a tile generates a broadcast
    /// (via Event_Generate matching a broadcast channel config), the BROADCAST_N
    /// event reaches every tile in the same column.
    ///
    /// This is the mechanism that synchronizes trace start/stop: the CDO writes
    /// Event_Generate on the shim tile, which generates BROADCAST_15 (start) or
    /// BROADCAST_14 (stop), and all tiles' trace units see their start/stop event.
    pub(crate) fn propagate_broadcasts(&mut self, col: u8, source_row: u8) {
        // Drain pending broadcasts from the source tile.
        let broadcasts = if let Some(tile) = self.array.get_mut(col, source_row) {
            tile.drain_pending_broadcasts()
        } else {
            return;
        };

        if broadcasts.is_empty() {
            return;
        }

        // Propagate each broadcast event to every tile in the column.
        let rows = self.array.rows();
        for hw_id in &broadcasts {
            log::info!(
                "Propagating BROADCAST_{} (hw_id={}) from tile ({},{}) to column {}",
                hw_id - 107, hw_id, col, source_row, col
            );
            for row in 0..rows {
                if let Some(tile) = self.array.get_mut(col as u8, row as u8) {
                    // Notify both trace units -- the trace unit checks if the
                    // event matches its configured start/stop event.
                    tile.notify_core_trace_event(*hw_id, 0);
                    tile.notify_mem_trace_event(*hw_id, 0);
                }
            }
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
        use crate::config::Config;

        let Some(test_xclbin) = Config::get().add_one_xclbin() else {
            eprintln!("Skipping real CDO test: file not found (set MLIR_AIE_PATH)");
            return;
        };

        // Load xclbin
        let xclbin = Xclbin::from_file(&test_xclbin).unwrap();
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

        // Lock_value field is 6-bit signed (bits [5:0], per AM025 regdb).
        // Value 5 fits in unsigned 6-bit range -> positive.
        state.write_register(addr, 5).unwrap();
        assert_eq!(state.array.tile(1, 2).locks[5].value, 5);

        // Value 0x3F = 63 = all 6 bits set -> sign-extends to -1.
        state.write_register(addr, 0x3F).unwrap();
        assert_eq!(state.array.tile(1, 2).locks[5].value, -1);

        // Value 0x20 = bit 5 set -> sign-extends to -32.
        state.write_register(addr, 0x20).unwrap();
        assert_eq!(state.array.tile(1, 2).locks[5].value, -32);

        // Value 31 = max positive in 6-bit signed.
        state.write_register(addr, 31).unwrap();
        assert_eq!(state.array.tile(1, 2).locks[5].value, 31);
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

    #[test]
    fn test_cdo_writes_tile_init_data() {
        // Verify that CDO DmaWrite correctly loads in2_mem_buff_0 into
        // tile(0,2) data memory at offset 0x400.
        // This is a regression test for the vec_vec_add_tile_init failure.
        let xclbin_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../mlir-aie/build/test/npu-xrt/vec_vec_add_tile_init/aie.xclbin");
        if !xclbin_path.exists() {
            eprintln!("Skipping tile init test: {:?} not found", xclbin_path);
            return;
        }

        let xclbin = Xclbin::from_file(&xclbin_path).unwrap();
        let section = xclbin.find_section(SectionKind::AiePartition).unwrap();
        let partition = AiePartition::parse(section.data()).unwrap();
        let pdi = partition.primary_pdi().unwrap();
        let cdo_offset = find_cdo_offset(pdi.pdi_image).unwrap();
        let cdo = Cdo::parse(&pdi.pdi_image[cdo_offset..]).unwrap();

        let mut state = DeviceState::new_npu1();
        state.apply_cdo(&cdo).unwrap();

        // Check tile(0,2) data memory at offset 0x400
        // Expected: in2_mem_buff_0 = [0, 1, 2, ..., 255] as i32
        let tile = state.array.tile(0, 2);
        let dm = tile.data_memory();

        // Read all 256 words from offset 0x400 and verify CDO init is correct.
        let mut values = Vec::new();
        for i in 0..256 {
            let off = 0x400 + i * 4;
            let word = u32::from_le_bytes([dm[off], dm[off+1], dm[off+2], dm[off+3]]);
            values.push(word);
        }

        for i in 0..256 {
            assert_eq!(values[i], i as u32,
                "in2_mem_buff_0[{}] should be {} but was {}", i, i, values[i]);
        }

        // Also verify data_bytes was counted
        assert!(state.stats.data_bytes >= 1024,
            "Expected at least 1024 data bytes written, got {}", state.stats.data_bytes);
    }

    /// Test that single-word BD writes (as from control packets) defer parsing
    /// until the DMA channel start queue is written.
    #[test]
    fn test_lazy_bd_parsing_single_word_writes() {
        use crate::device::registers::TileAddress;
        use crate::device::regdb::device_reg_layout;

        let mut state = DeviceState::new_npu1();
        let col: u8 = 1;
        let row: u8 = 2; // Compute tile

        let reg_layout = device_reg_layout();
        let bd_base = reg_layout.memory_bd_base;   // 0x1D000
        let bd_stride = reg_layout.memory_bd_stride; // 0x20
        let bd_idx: usize = 3;

        // Build known BD words for a compute tile (6 words).
        // Use the real parser to construct expected values, then write
        // those words one at a time.
        let base_addr_words: u32 = 0x100; // 256 words = 1024 bytes
        let length_words: u32 = 64;       // 64 words = 256 bytes

        // Word 0: Base_Address | Buffer_Length (from regdb field layout)
        let lay = &reg_layout.memory_bd;
        let w0 = lay.base_address.insert(0, base_addr_words)
                   | lay.buffer_length.insert(0, length_words);
        // Word 1-4: leave as zero (no packet, no strides, no iteration)
        let w1: u32 = 0;
        let w2: u32 = 0;
        let w3: u32 = 0;
        let w4: u32 = 0;
        // Word 5: Valid_BD = 1 (and no locks, no chaining)
        let w5 = lay.valid_bd.insert(0, 1);

        let words = [w0, w1, w2, w3, w4, w5];

        // Write each word individually (simulating control packet path)
        for (i, &word) in words.iter().enumerate() {
            let offset = bd_base + (bd_idx as u32) * bd_stride + (i as u32) * 4;
            let addr = TileAddress::encode(col, row, offset);
            state.write_register(addr, word).unwrap();
        }

        // Verify the BD is marked dirty in the DMA engine
        let dma = state.array.dma_engine(col, row).unwrap();
        assert!(dma.is_bd_dirty(bd_idx as u8),
            "BD should be dirty after single-word writes");

        // Verify the BD config has NOT been updated yet (should be default)
        let bd_before = dma.get_bd(bd_idx as u8).unwrap();
        assert_eq!(bd_before.base_addr, 0,
            "BD config should not be updated until channel start");
        assert_eq!(bd_before.length, 0,
            "BD length should be 0 before channel start");

        // Now write the channel start queue register to trigger re-parse.
        // MM2S channel 0 = channel index 2 (after S2MM_0, S2MM_1).
        // Start queue offset = channel_base + ch_idx * stride + 4
        let ch_idx: usize = 2; // MM2S_0
        let start_queue_offset = reg_layout.memory_channel_base
            + (ch_idx as u32) * reg_layout.memory_channel_stride + 4;

        // Start_BD_ID field value = bd_idx, repeat_count = 0
        let queue_val = reg_layout.memory_channel.start_bd_id.insert(0, bd_idx as u32);

        let addr = TileAddress::encode(col, row, start_queue_offset);
        state.write_register(addr, queue_val).unwrap();

        // Now the BD should have been re-parsed and configured
        let dma = state.array.dma_engine(col, row).unwrap();
        assert!(!dma.is_bd_dirty(bd_idx as u8),
            "BD should no longer be dirty after channel start");

        let bd_after = dma.get_bd(bd_idx as u8).unwrap();
        assert_eq!(bd_after.base_addr, (base_addr_words * 4) as u64,
            "BD base_addr should be {} bytes", base_addr_words * 4);
        assert_eq!(bd_after.length, length_words * 4,
            "BD length should be {} bytes", length_words * 4);
        assert!(bd_after.valid,
            "BD should be valid after re-parse");
    }
}
