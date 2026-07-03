//! Compute / Shim tile register handlers.

use super::*;

/// CORE_CONTROL RESET bit (bit 1). A compute core executes only when
/// ENABLE (bit 0) = 1 AND RESET (bit 1) = 0. Power-on value is 0x02 (reset
/// asserted, per aie-rt `xaiemlgbl_params.h` RESET_LSB=1); the CDO `enable`
/// phase is a masked bit-0 write that leaves RESET set, so after config the
/// core is `0x03` = enabled-but-held. Firmware deasserts reset at kernel
/// launch (`release_core_resets`). Deriving run-state from bit 0 alone made
/// cores run at CDO time -- the SP-4a cold-start root cause (#140).
const CORE_CONTROL_RESET: u32 = 0x2;

/// Pick the DMA channel field layout that matches a tile kind.
///
/// Per aie-rt `xaiemlgbl_params.h`:
///   - `XAIEMLGBL_MEM_TILE_DMA_*_TASK_QUEUE` Start_BD_ID is bits[5:0]
///     (6-bit, 48 BDs) -- use `reg_layout.memtile_channel`.
///   - `XAIEMLGBL_MEMORY_DMA_*_TASK_QUEUE` (compute) Start_BD_ID is
///     bits[3:0] (4-bit, 16 BDs) -- use `reg_layout.memory_channel`.
///   - Shim `XAIEMLGBL_NOC_MODULE_DMA_*_TASK_QUEUE` Start_BD_ID is also
///     4-bit; the compute layout is bit-compatible (same field widths)
///     so we route shim through `memory_channel` here.
///
/// Anti-pattern guard: any DMA channel handler that may run for a
/// MemTile must call this rather than reaching for `memory_channel`
/// directly. The 4-bit field silently truncates BDs 16+, and the BD-
/// channel invariant in `start_channel_with_repeat` then rejects the
/// (valid) MemTile BD as "even-only / odd-only" wiring -- masking the
/// real bug behind a misleading log line.
#[inline]
pub(super) fn channel_field_layout(
    reg_layout: &'static regdb::DeviceRegLayout,
    tile_kind: TileKind,
) -> &'static regdb::ChannelFieldLayout {
    if tile_kind.is_mem() {
        &reg_layout.memtile_channel
    } else {
        &reg_layout.memory_channel
    }
}

impl DeviceState {
    // =========================================================================
    // Lock helper functions (consolidate duplicate compute/memtile code)
    // =========================================================================

    /// Write a lock value (direct write, no mask).
    ///
    /// Lock registers are 16 bytes (LOCK_STRIDE) apart per AM025.
    /// Lock_Value field width is derived from the archspec LockValueLayout.
    pub(super) fn write_lock_value(
        tile: &mut Tile,
        tile_addr: TileAddress,
        base: u32,
        stride: u32,
        value: u32,
        tile_kind: &str,
    ) {
        let lock_idx = ((tile_addr.offset - base) / stride) as usize;
        if lock_idx < tile.locks.len() {
            let signed = crate::device::arch_handle::lock_value_layout().sign_extend(value);
            tile.locks[lock_idx].set(signed);
            if value != 0 {
                log::info!(
                    "CDO init {} lock {} on tile ({},{}) = {}",
                    tile_kind,
                    lock_idx,
                    tile_addr.col,
                    tile_addr.row,
                    signed
                );
            }
        }
    }

    /// Masked write to a lock value.
    pub(super) fn mask_write_lock_value(
        tile: &mut Tile,
        offset: u32,
        base: u32,
        stride: u32,
        mask: u32,
        value: u32,
    ) {
        let lock_idx = ((offset - base) / stride) as usize;
        if lock_idx < tile.locks.len() {
            // Read current value in unsigned representation for mask ops.
            // Mask is derived from the archspec LockValueLayout (6-bit field for AIE2).
            let layout = crate::device::arch_handle::lock_value_layout();
            let current_raw = (tile.locks[lock_idx].value as u8 & layout.mask as u8) as u32;
            let new_raw = (current_raw & !mask) | (value & mask);
            let signed = layout.sign_extend(new_raw);
            tile.locks[lock_idx].set(signed);
        }
    }

    /// Write BD data from a byte array.
    pub(super) fn dma_write_bd_data(&mut self, col: u8, row: u8, offset: u32, data: &[u8]) {
        let words: Vec<u32> = data
            .chunks(4)
            .map(|chunk| {
                let mut arr = [0u8; 4];
                arr[..chunk.len()].copy_from_slice(chunk);
                u32::from_le_bytes(arr)
            })
            .collect();

        log::debug!(
            "dma_write_bd_data tile({},{}) offset=0x{:05X} data_len={} words={:X?}",
            col,
            row,
            offset,
            data.len(),
            words
        );

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
    pub(super) fn write_dma_bd(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let tile_kind = self.array.arch().tile_kind(col, row);
        let reg_layout = regdb::device_reg_layout();

        // Select base/stride/max words based on tile type.
        // Shim and compute share the same BD base and stride, but shim uses
        // 8 words per BD while compute uses 6.
        let (bd_base, bd_stride, max_words) = match tile_kind {
            TileKind::ShimNoc | TileKind::ShimPl => {
                (reg_layout.shim_bd_base, reg_layout.shim_bd_stride, reg_layout.shim_bd_words)
            }
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

        log::trace!("  BD {} word {} = 0x{:08X} tile({},{}) marked dirty", bd_idx, word, value, col, row);
    }

    /// Write to a compute-tile DMA channel register.
    ///
    /// Handles both Ctrl and Start_Queue writes. This is the single
    /// source of truth for compute DMA channel effects; both the CDO
    /// path (via `apply_device_op::DmaStart` -> `write_register`)
    /// and non-CDO paths (NPU instructions, control packets, FFI
    /// via `write_tile_register`) reach this function.
    pub(super) fn write_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();
        // Pick layout by tile kind.  Dispatch routes MemTile DMA writes to
        // `write_memtile_dma_channel` (a separate handler) and Shim writes
        // to `write_shim_dma_channel`, so this site is reached only on
        // Compute today -- but using the helper rather than `memory_channel`
        // hardcoded means if dispatch ever changes (e.g. a future FFI or
        // control-packet path lands a MemTile write here) we won't silently
        // truncate BDs 16+.
        let tile_kind = self.array.arch().tile_kind(col, row);
        let lay = channel_field_layout(reg_layout, tile_kind);

        // Channel registers: base/stride derived from register database
        // Layout: S2MM_0, S2MM_1, MM2S_0, MM2S_1
        // Each channel has CTRL register and START_QUEUE register (4 bytes each)
        let rel = offset - reg_layout.memory_channel_base;
        let ch_idx = (rel / reg_layout.memory_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.memory_channel_stride) >= 4;
        let is_s2mm = ch_idx < xdna_archspec::aie2::compute::NUM_DMA_CHANNELS as usize;

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

                dma.set_channel_compression_config(ch_idx as u8, compress_en, decompression_en, ooo_en);
            }
        }

        // Start the DmaEngine channel when START_QUEUE is written (non-CDO
        // paths; CDO goes through DeviceOp::DmaStart).
        if is_start_queue {
            let bd_idx = lay.start_bd_id.extract(value) as u8;
            let repeat_count = lay.repeat_count.extract(value) as u8;

            // Read controller_id and fot_mode from the tile's channel state
            // (which was set earlier when the control register was written)
            let (controller_id, fot_mode) = self
                .array
                .get(col, row)
                .map(|t| {
                    let dma_ch = &t.dma_channels[ch_idx];
                    (dma_ch.controller_id, dma_ch.fot_mode)
                })
                .unwrap_or((0, 0));

            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                // Set task config before enqueuing the task
                dma.set_channel_task_config(ch_idx as u8, enable_token_issue, controller_id, fot_mode);

                // Enqueue to the channel's task queue (hardware has 8-deep queue).
                // Unlike start_channel_with_repeat(), this never rejects a busy
                // channel -- it queues the task for execution when the current
                // transfer finishes.
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, enable_token_issue) {
                    log::warn!(
                        "DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col,
                        row,
                        ch_idx,
                        bd_idx,
                    );
                } else if enable_token_issue {
                    log::debug!("CDO enqueued DMA channel {} BD {} repeat={} token_issue=true controller_id={} on tile ({},{})",
                        ch_idx, bd_idx, repeat_count, controller_id, col, row);
                } else {
                    log::info!(
                        "CDO enqueued DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx,
                        bd_idx,
                        repeat_count,
                        col,
                        row
                    );
                }
            }
        }
    }

    /// Masked write to a compute-tile DMA channel register.
    ///
    /// Reached by both CDO MaskWrites (via `apply_device_op::RegMask`
    /// -> `mask_write_register`) and non-CDO paths (NPU instructions,
    /// control packets, FFI). MaskWrite is never promoted to a typed
    /// op; mask-blend correctness lives in `mask_write_core_register`
    /// and per-channel handlers, applied to the merged value here.
    pub(super) fn mask_write_dma_channel(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();
        // Pick layout by tile kind.  See `write_dma_channel` for the
        // dispatch-contract rationale -- this site is compute-only today
        // but the helper makes the right choice if a non-CDO path (NPU
        // instructions, control packets, FFI) ever lands a MaskWrite to
        // a MemTile Start_Queue here.
        let tile_kind = self.array.arch().tile_kind(col, row);
        let lay = channel_field_layout(reg_layout, tile_kind);
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
        //
        // Note: MaskWrite to Start_Queue is not promoted to DmaStart
        // (promotion is gated to Write, not MaskWrite). If a MaskWrite
        // path ever reaches here, we fall back to the legacy inline
        // enqueue so that observed behavior stays identical.
        if let Some(queue_val) = new_start_queue {
            let bd_idx = lay.start_bd_id.extract(queue_val) as u8;
            let repeat_count = lay.repeat_count.extract(queue_val) as u8;
            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    log::warn!(
                        "DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col,
                        row,
                        ch_idx,
                        bd_idx,
                    );
                } else {
                    log::info!(
                        "CDO enqueued DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx,
                        bd_idx,
                        repeat_count,
                        col,
                        row
                    );
                }
            }
        }
    }

    /// Write to a shim DMA channel register.
    ///
    /// Shim tiles have the same 2 S2MM + 2 MM2S channel layout as compute
    /// tiles, but at a different base address (0x1D200 vs 0x1DE00).
    /// The register bit layout is the same (same ChannelFieldLayout).
    pub(super) fn write_shim_dma_channel(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();
        let lay = &reg_layout.memory_channel; // Same bit layout as compute

        let rel = offset - reg_layout.shim_channel_base;
        let ch_idx = (rel / reg_layout.shim_channel_stride) as usize;
        let is_start_queue = (rel % reg_layout.shim_channel_stride) >= 4;
        let is_s2mm = ch_idx < xdna_archspec::aie2::shim::NUM_DMA_CHANNELS as usize;

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

            let (controller_id, fot_mode) = self
                .array
                .get(col, row)
                .map(|t| {
                    let dma_ch = &t.dma_channels[ch_idx];
                    (dma_ch.controller_id, dma_ch.fot_mode)
                })
                .unwrap_or((0, 0));

            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);

            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                dma.set_channel_task_config(ch_idx as u8, enable_token_issue, controller_id, fot_mode);

                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, enable_token_issue) {
                    log::warn!(
                        "Shim DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col,
                        row,
                        ch_idx,
                        bd_idx,
                    );
                } else {
                    log::info!(
                        "CDO enqueued Shim DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx,
                        bd_idx,
                        repeat_count,
                        col,
                        row
                    );
                }
            }
        }
    }

    /// Masked write to a shim DMA channel register.
    pub(super) fn mask_write_shim_dma_channel(
        &mut self,
        col: u8,
        row: u8,
        offset: u32,
        mask: u32,
        value: u32,
    ) {
        let reg_layout = regdb::device_reg_layout();
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
            // Re-parse ALL dirty BDs so chained BDs are also configured
            self.reparse_all_dirty_bds(col, row);
            if let Some(dma) = self.array.dma_engine_mut(col, row) {
                if !dma.enqueue_task(ch_idx as u8, bd_idx, repeat_count, false) {
                    log::warn!(
                        "Shim DMA tile({},{}) ch{} task queue overflow (BD {} dropped)",
                        col,
                        row,
                        ch_idx,
                        bd_idx,
                    );
                } else {
                    log::info!(
                        "CDO enqueued Shim DMA channel {} BD {} repeat={} on tile ({},{})",
                        ch_idx,
                        bd_idx,
                        repeat_count,
                        col,
                        row
                    );
                }
            }
        }
    }

    /// Write to a core register.
    ///
    /// The CORE_CONTROL branch is the single source of truth for
    /// core-enable effects: both the CDO path (via
    /// `apply_device_op::CoreEnable` -> `write_register`) and
    /// non-CDO paths (NPU instructions, control packets, FFI via
    /// `write_tile_register`) reach this function.
    pub(super) fn write_core_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use xdna_archspec::aie2::registers as cm;

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // Core_Processor_Bus register (AM025, offset 0x32038)
        const CORE_PROCESSOR_BUS: u32 = 0x32038;

        match offset {
            cm::CORE_CONTROL => {
                let was_enabled = tile.core.enabled;
                tile.core.control = value;
                tile.core.enabled = value & 1 != 0 && value & CORE_CONTROL_RESET == 0;
                if tile.core.enabled != was_enabled {
                    log::info!(
                        "Core ({},{}) {}",
                        col,
                        row,
                        if tile.core.enabled { "ENABLED" } else { "DISABLED" }
                    );
                    self.pending_core_enables.push((col, row, tile.core.enabled));
                }
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
            CORE_PROCESSOR_BUS => {
                tile.processor_bus_enabled = value & 1 != 0;
                log::info!(
                    "Core ({},{}) processor bus {}",
                    col,
                    row,
                    if tile.processor_bus_enabled {
                        "ENABLED"
                    } else {
                        "DISABLED"
                    }
                );
            }
            _ => {}
        }
    }

    /// Firmware kernel-launch: deassert RESET on every enabled compute core.
    ///
    /// On real hardware the CDO leaves each core `CORE_CONTROL = 0x03`
    /// (ENABLE=1, RESET=1 = held); the XDNA firmware clears the reset bit when
    /// the kernel is launched, and THAT is what actually starts the cores.
    /// There is no reset-clearing register write anywhere in the CDO or the
    /// runtime `insts.bin` -- it is a pure firmware side effect -- so we model
    /// it explicitly here.
    ///
    /// This is the firmware-launch anchor for #140 SP-4a. Part 1 calls it at
    /// config-completion (behavior-preserving: cores start ~at CDO time, as
    /// before). Part 2 relocates it to the true launch timing; the long-term
    /// dream is running real NPU firmware on a simulated management processor,
    /// at which point this stub is replaced by the firmware's own reset write.
    pub fn release_core_resets(&mut self) {
        for col in 0..self.array.cols() {
            for row in 0..self.array.rows() {
                let Some(tile) = self.array.get_mut(col, row) else {
                    continue;
                };
                if !tile.is_compute() {
                    continue;
                }
                // Enable-intent (bit 0) but held in reset (bit 1): firmware
                // clears reset -> ENABLE=1 && RESET=0 -> the core executes.
                if tile.core.control & 1 != 0 && tile.core.control & CORE_CONTROL_RESET != 0 {
                    let was_enabled = tile.core.enabled;
                    tile.core.control &= !CORE_CONTROL_RESET;
                    tile.core.enabled = true;
                    if !was_enabled {
                        self.pending_core_enables.push((col, row, true));
                    }
                }
            }
        }
    }

    /// Masked write to a core register.
    pub(super) fn mask_write_core_register(&mut self, col: u8, row: u8, offset: u32, mask: u32, value: u32) {
        use xdna_archspec::aie2::registers as cm;
        const CORE_PROCESSOR_BUS: u32 = 0x32038;

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        match offset {
            cm::CORE_CONTROL => {
                let was_enabled = tile.core.enabled;
                tile.core.control = (tile.core.control & !mask) | (value & mask);
                tile.core.enabled = tile.core.control & 1 != 0 && tile.core.control & CORE_CONTROL_RESET == 0;
                if tile.core.enabled != was_enabled {
                    log::info!(
                        "Core ({},{}) {}",
                        col,
                        row,
                        if tile.core.enabled { "ENABLED" } else { "DISABLED" }
                    );
                    self.pending_core_enables.push((col, row, tile.core.enabled));
                }
            }
            cm::CORE_STATUS => {
                tile.core.status = (tile.core.status & !mask) | (value & mask);
            }
            CORE_PROCESSOR_BUS => {
                let masked = (value & mask) | (if tile.processor_bus_enabled { 1 } else { 0 } & !mask);
                tile.processor_bus_enabled = masked & 1 != 0;
                log::info!(
                    "Core ({},{}) processor bus {}",
                    col,
                    row,
                    if tile.processor_bus_enabled {
                        "ENABLED"
                    } else {
                        "DISABLED"
                    }
                );
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
    pub(super) fn write_stream_switch(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        use xdna_archspec::aie2::stream_switch::{ENABLE_BIT, SLAVE_SELECT_MASK};
        let ss = &regdb::device_reg_layout().memory_stream_switch;

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
                        let (dh, arb, msel) = tile
                            .stream_switch
                            .master_packet_cfg(port)
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
                        log::info!(
                            "Tile ({},{}) stream switch: slave[{}] packet mode enabled",
                            col,
                            row,
                            port
                        );
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
                log::info!(
                    "Tile ({},{}) stream switch: slave[{}] slot[{}] = 0x{:08X}",
                    col,
                    row,
                    slave_port,
                    slot,
                    value
                );
            }
        }
    }
}
