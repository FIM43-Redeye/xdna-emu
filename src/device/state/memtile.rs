//! MemTile register handlers.

use super::*;

impl DeviceState {
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
    pub(super) fn write_memtile_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();

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
                    tile.registers.insert(reg_off, value);
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
    pub(super) fn parse_memtile_bd_from_words(&self, words: &[u32]) -> crate::device::dma::BdConfig {
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
    pub(crate) fn reparse_dirty_bd(&mut self, col: u8, row: u8, bd_idx: usize) {
        use crate::device::dma::bd::BufferDescriptor;

        let is_dirty = self.array.dma_engine(col, row)
            .map_or(false, |dma| dma.is_bd_dirty(bd_idx as u8));
        log::debug!("reparse_dirty_bd: tile({},{}) BD {} is_dirty={}",
            col, row, bd_idx, is_dirty);
        if !is_dirty {
            return;
        }

        let tile_type = self.array.arch().tile_type(col, row);
        let reg_layout = regdb::device_reg_layout();

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

    /// Re-parse ALL dirty BDs on a tile.
    ///
    /// Called when a DMA channel's START_QUEUE register is written, to ensure
    /// that every BD in the chain has been configured -- not just the starting
    /// BD. Without this, chained BDs (e.g., BD0->BD1 for double buffering)
    /// remain unconfigured and cause BdNotValid errors at chain time.
    pub(super) fn reparse_all_dirty_bds(&mut self, col: u8, row: u8) {
        let num_bds = self.array.dma_engine(col, row)
            .map_or(0, |dma| dma.num_bds());
        for i in 0..num_bds {
            self.reparse_dirty_bd(col, row, i);
        }
    }

    /// Write MemTile BD data from a byte array.
    ///
    /// MemTile DMA_WRITE commands write 32 bytes (8 words) per BD.
    /// This path has all 8 words available, so the data-driven parser
    /// can extract all fields including iteration and locks.
    pub(super) fn dma_write_memtile_bd_data(&mut self, col: u8, row: u8, offset: u32, data: &[u8]) {
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
        let reg_layout = regdb::device_reg_layout();
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
                    log::info!("CDO configured MemTile BD {} on tile ({},{}) addr=0x{:X} len={} d0=[{},{}] d1=[{},{}] d2=[{},{}] d3_stride={} iter=[wrap={},step={}] acq={:?} rel={:?} next={:?} pkt={}(id={},type={})",
                        bd_idx, col, row, config.base_addr, config.length,
                        config.d0.size, config.d0.stride, config.d1.size, config.d1.stride,
                        config.d2.size, config.d2.stride, config.d3.stride,
                        config.iteration.wrap, config.iteration.stepsize,
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
    pub(super) fn write_memtile_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();
        let lay = &reg_layout.memtile_channel;

        // Determine channel index and whether this is a start queue write
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= crate::arch::memtile::NUM_DMA_CHANNELS as usize * 2 {
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
            log::debug!("MemTile Start_Queue raw=0x{:08X} bd={} repeat={} (actual {}x) ch={}",
                value, bd_idx, repeat_count, repeat_count as u32 + 1, ch_idx);
            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    log::warn!(
                        "MemTile DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col, row, ch_idx, bd_idx,
                    );
                } else {
                    let mt_s2mm = crate::arch::memtile::NUM_DMA_CHANNELS as usize;
                    let dir = if ch_idx < mt_s2mm { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < mt_s2mm { ch_idx } else { ch_idx - mt_s2mm };
                    log::info!("CDO enqueued MemTile DMA {} ch {} BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Masked write to MemTile DMA channel register.
    pub(super) fn mask_write_memtile_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();
        let lay = &reg_layout.memtile_channel;
        let (ch_idx, is_start_queue) = self.decode_memtile_channel_offset(offset);

        if ch_idx >= crate::arch::memtile::NUM_DMA_CHANNELS as usize * 2 {
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
            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    log::warn!(
                        "MemTile DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col, row, ch_idx, bd_idx,
                    );
                } else {
                    let mt_s2mm = crate::arch::memtile::NUM_DMA_CHANNELS as usize;
                    let dir = if ch_idx < mt_s2mm { "S2MM" } else { "MM2S" };
                    let local_ch = if ch_idx < mt_s2mm { ch_idx } else { ch_idx - mt_s2mm };
                    log::info!("CDO enqueued MemTile DMA {} ch {} BD {} repeat={} on tile ({},{})",
                        dir, local_ch, bd_idx, repeat_count, col, row);
                }
            }
        }
    }

    /// Decode MemTile DMA channel offset into (channel_index, is_start_queue).
    pub(super) fn decode_memtile_channel_offset(&self, offset: u32) -> (usize, bool) {
        let reg_layout = regdb::device_reg_layout();
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
    pub(super) fn write_memtile_stream_switch(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use crate::arch::stream_switch::{ENABLE_BIT, SLAVE_SELECT_MASK};
        let ss = &regdb::device_reg_layout().memtile_stream_switch;

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
}
