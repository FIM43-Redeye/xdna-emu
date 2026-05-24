//! Top-level register dispatch (write_register, mask_write_register, dma_write).

use anyhow::Result;

use super::*;

impl DeviceState {
    /// Hardware register bus -- single entry point for all register writes.
    ///
    /// This is the emulator's equivalent of the real NPU's per-tile register
    /// bus. A write to any offset produces the same side effects regardless
    /// of source (CDO, NPU instruction, control packet, FFI).
    ///
    /// Dispatches to module-specific handlers (stream switch, DMA engine,
    /// locks, etc.) and then runs tile-local side effects (trace config,
    /// shim mux, cascade, event broadcast) via `apply_tile_local_effects()`.
    ///
    /// All external callers MUST use this method. Never write to tile state
    /// directly.
    pub fn write_tile_register(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        // Warn-once-per-site on accesses to gated tiles.  Per spec, the
        // access still proceeds (silicon does not block it) -- we just
        // surface the bug pattern.  Clock-control register writes are
        // suppressed since they are the mechanism by which a tile is
        // ungated in the first place.
        let tile_kind = tile_kind_from_row(row);
        let subsystem = subsystem_from_offset(offset, tile_kind);
        if subsystem != SubsystemKind::ClockControl
            && self.array.clock_mut().warn_gated_access(col, row, offset)
            && warnings_enabled()
        {
            log::warn!(
                "access to gated tile ({}, {}) at offset 0x{:05X}; silicon would produce undefined results. \
                Set XDNA_EMU_WARN_GATED_ACCESS=0 to silence.",
                col,
                row,
                offset,
            );
        }

        let address = TileAddress::encode(col, row, offset);
        if let Err(e) = self.write_register(address, value) {
            log::error!("write_tile_register failed: tile({},{}) offset=0x{:05X}: {:?}", col, row, offset, e);
        }
    }

    /// Internal register bus dispatch. Decodes an encoded tile address and
    /// routes the write through module-specific handlers.
    pub(super) fn write_register(&mut self, address: u32, value: u32) -> Result<()> {
        let tile_addr = TileAddress::decode(address);

        // Raw register storage -- single source of truth, always first.
        // Must happen before module dispatch since handlers re-borrow self.
        if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
            tile.registers.insert(tile_addr.offset, value);
        } else {
            return Ok(());
        }

        // Dispatch based on subsystem classification.
        let reg_layout = regdb::device_reg_layout();
        let tile_kind = tile_kind_from_row(tile_addr.row);
        let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind);

        match subsystem {
            SubsystemKind::Lock => {
                if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
                    match tile_kind {
                        TileKind::Mem => {
                            Self::write_lock_value(
                                tile,
                                tile_addr,
                                reg_layout.memtile_lock_base,
                                reg_layout.memtile_lock_stride,
                                value,
                                "MemTile",
                            );
                        }
                        TileKind::Compute => {
                            Self::write_lock_value(
                                tile,
                                tile_addr,
                                reg_layout.memory_lock_base,
                                reg_layout.memory_lock_stride,
                                value,
                                "Compute",
                            );
                        }
                        TileKind::ShimNoc | TileKind::ShimPl => {
                            Self::write_lock_value(
                                tile,
                                tile_addr,
                                reg_layout.shim_lock_base,
                                reg_layout.shim_lock_stride,
                                value,
                                "Shim",
                            );
                        }
                    }
                }
            }

            SubsystemKind::Dma => {
                // Sub-dispatch: BD writes vs channel control writes.
                // The offset boundary between BDs and channels differs by tile type.
                match tile_kind {
                    TileKind::Mem => {
                        if tile_addr.offset < reg_layout.memtile_channel_s2mm_base {
                            self.write_memtile_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                        } else {
                            self.write_memtile_dma_channel(
                                tile_addr.col,
                                tile_addr.row,
                                tile_addr.offset,
                                value,
                            );
                        }
                    }
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        // Shim DMA: channel registers at shim_channel_base,
                        // BD registers at shim_bd_base (same as compute BD base).
                        let shim_ch_base = reg_layout.shim_channel_base;
                        let shim_ch_end = shim_ch_base + 4 * reg_layout.shim_channel_stride;
                        if (shim_ch_base..shim_ch_end).contains(&tile_addr.offset) {
                            self.write_shim_dma_channel(
                                tile_addr.col,
                                tile_addr.row,
                                tile_addr.offset,
                                value,
                            );
                        } else {
                            self.write_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                        }
                    }
                    TileKind::Compute => {
                        if tile_addr.offset < reg_layout.memory_channel_base {
                            self.write_dma_bd(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                        } else {
                            self.write_dma_channel(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                        }
                    }
                }
            }

            SubsystemKind::Processor => {
                self.write_core_register(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            SubsystemKind::StreamSwitch => match tile_kind {
                TileKind::Mem => {
                    self.write_memtile_stream_switch(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                }
                _ => {
                    self.write_stream_switch(tile_addr.col, tile_addr.row, tile_addr.offset, value);
                }
            },

            SubsystemKind::ProgramMemory => {
                // Write 32-bit value to program memory. Control packets
                // deliver ELF code word-by-word; CDO uses bulk dma_write.
                if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
                    let pm_offset = (tile_addr.offset - 0x20000) as usize;
                    tile.write_program(pm_offset, &value.to_le_bytes());
                }
            }

            SubsystemKind::DataMemory => {
                // Write 32-bit value to data memory. Control packets can
                // write to data memory word-by-word; CDO uses bulk dma_write.
                if let Some(tile) = self.array.get_mut(tile_addr.col, tile_addr.row) {
                    tile.write_data_u32(tile_addr.offset as usize, value);
                }
            }

            SubsystemKind::ClockControl => {
                self.array
                    .clock_mut()
                    .write_register(tile_addr.col, tile_addr.row, tile_addr.offset, value);
            }

            _ => {}
        }

        // Tile-local register side effects: trace config, edge detection,
        // event port selection, cascade config, shim mux, lock overflow/
        // underflow clear, event broadcast, and Event_Generate.
        self.apply_tile_local_effects(tile_addr.col, tile_addr.row, tile_addr.offset, value);

        // Propagate broadcast events to a fixed point: a delivered
        // broadcast can latch a shim L1, whose IRQ_NO output must itself
        // propagate to L2 within this dispatch (Tier A interrupt path).
        self.propagate_broadcasts_fixpoint(tile_addr.col, tile_addr.row);

        Ok(())
    }

    /// Internal masked register bus dispatch. Reads the current value,
    /// applies the mask, and delegates to `write_register()`.
    pub(super) fn mask_write_register(&mut self, address: u32, mask: u32, value: u32) -> Result<()> {
        let tile_addr = TileAddress::decode(address);
        let tile_kind = tile_kind_from_row(tile_addr.row);
        let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind);

        log::trace!(
            "mask_write_register: addr=0x{:08X} tile({},{}) offset=0x{:05X} subsystem={:?}",
            address,
            tile_addr.col,
            tile_addr.row,
            tile_addr.offset,
            subsystem
        );

        if self.array.get_mut(tile_addr.col, tile_addr.row).is_none() {
            log::trace!("mask_write_register: tile({},{}) not in array", tile_addr.col, tile_addr.row);
            return Ok(());
        }

        let reg_layout = regdb::device_reg_layout();

        // Track whether we need to run tile-local effects after the match.
        // Some arms do RMW and need the shim mux parser or other tile-local
        // handlers to fire on the merged value.
        let mut tile_local_value: Option<u32> = None;

        // Scope the tile borrow so it ends before apply_tile_local_effects.
        {
            let tile = self.array.get_mut(tile_addr.col, tile_addr.row).unwrap();

            match subsystem {
                SubsystemKind::Lock => match tile_kind {
                    TileKind::Mem => {
                        Self::mask_write_lock_value(
                            tile,
                            tile_addr.offset,
                            reg_layout.memtile_lock_base,
                            reg_layout.memtile_lock_stride,
                            mask,
                            value,
                        );
                    }
                    TileKind::Compute => {
                        Self::mask_write_lock_value(
                            tile,
                            tile_addr.offset,
                            reg_layout.memory_lock_base,
                            reg_layout.memory_lock_stride,
                            mask,
                            value,
                        );
                    }
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        Self::mask_write_lock_value(
                            tile,
                            tile_addr.offset,
                            reg_layout.shim_lock_base,
                            reg_layout.shim_lock_stride,
                            mask,
                            value,
                        );
                    }
                },

                _ => {}
            }
        }

        // Arms that take &mut self must be outside the tile borrow scope.
        match subsystem {
            SubsystemKind::Dma => {
                // Sub-dispatch: channel control mask writes.
                // BD mask writes are uncommon but possible; channel mask writes
                // are the primary use case (enabling/disabling channels).
                match tile_kind {
                    TileKind::Mem => {
                        if tile_addr.offset >= reg_layout.memtile_channel_s2mm_base {
                            self.mask_write_memtile_dma_channel(
                                tile_addr.col,
                                tile_addr.row,
                                tile_addr.offset,
                                mask,
                                value,
                            );
                        }
                        // BD mask writes for memtile: not implemented (uncommon in CDO)
                    }
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        let shim_ch_base = reg_layout.shim_channel_base;
                        let shim_ch_end = shim_ch_base + 4 * reg_layout.shim_channel_stride;
                        if (shim_ch_base..shim_ch_end).contains(&tile_addr.offset) {
                            self.mask_write_shim_dma_channel(
                                tile_addr.col,
                                tile_addr.row,
                                tile_addr.offset,
                                mask,
                                value,
                            );
                        }
                        // BD mask writes for shim: not implemented (uncommon in CDO)
                    }
                    TileKind::Compute => {
                        if tile_addr.offset >= reg_layout.memory_channel_base {
                            self.mask_write_dma_channel(
                                tile_addr.col,
                                tile_addr.row,
                                tile_addr.offset,
                                mask,
                                value,
                            );
                        }
                        // BD mask writes for compute: not implemented (uncommon in CDO)
                    }
                }
            }

            SubsystemKind::Processor => {
                self.mask_write_core_register(tile_addr.col, tile_addr.row, tile_addr.offset, mask, value);
            }

            SubsystemKind::Lock => {
                // Already handled in the scoped block above.
            }

            _ => {
                // Read-modify-write: preserve bits not covered by mask.
                // Multiple MaskWrites to the same register (e.g., Shim Mux config at
                // 0x1F000) must accumulate rather than clobber each other.
                let tile = self.array.get_mut(tile_addr.col, tile_addr.row).unwrap();
                let current = *tile.registers_ref().get(&tile_addr.offset).unwrap_or(&0);
                let new_value = (current & !mask) | (value & mask);
                log::debug!("CDO MaskWrite RMW: tile({},{}) offset=0x{:05X} current=0x{:08X} -> 0x{:08X} (mask=0x{:08X} val=0x{:08X})",
                    tile_addr.col, tile_addr.row, tile_addr.offset, current, new_value, mask, value);
                tile.registers.insert(tile_addr.offset, new_value);
                tile_local_value = Some(new_value);
            }
        }

        // Run tile-local effects for arms that did RMW (shim mux, generic fallthrough).
        if let Some(merged) = tile_local_value {
            self.apply_tile_local_effects(tile_addr.col, tile_addr.row, tile_addr.offset, merged);
        }

        Ok(())
    }

    /// DMA write to memory.
    ///
    /// Writes bulk data to program memory, data memory, or DMA BD registers
    /// depending on the address offset.
    pub(super) fn dma_write(&mut self, address: u32, data: &[u8]) -> Result<()> {
        let tile_addr = TileAddress::decode(address);

        if self.array.get(tile_addr.col, tile_addr.row).is_none() {
            return Ok(());
        }

        let offset = tile_addr.offset as usize;
        let tile_kind = tile_kind_from_row(tile_addr.row);
        let subsystem = subsystem_from_offset(tile_addr.offset, tile_kind);

        match subsystem {
            SubsystemKind::DataMemory => {
                // Write to data memory
                let tile = self.array.get_mut(tile_addr.col, tile_addr.row).unwrap();
                if tile.write_data(offset, data) {
                    self.stats.data_bytes += data.len();
                }
            }

            SubsystemKind::ProgramMemory => {
                use xdna_archspec::aie2::memory_map::PROGRAM_MEMORY_BASE;
                // Write to program memory
                let tile = self.array.get_mut(tile_addr.col, tile_addr.row).unwrap();
                let pm_offset = offset - PROGRAM_MEMORY_BASE as usize;
                if tile.write_program(pm_offset, data) {
                    self.stats.program_bytes += data.len();
                }
            }

            SubsystemKind::Dma => {
                // Write to DMA BD registers (bulk BD configuration).
                // Only BD writes happen via DmaWrite commands; channel config
                // uses Write/MaskWrite commands (handled in write_register).
                match tile_kind {
                    TileKind::Mem => {
                        self.dma_write_memtile_bd_data(tile_addr.col, tile_addr.row, tile_addr.offset, data);
                    }
                    _ => {
                        // Compute and shim tiles share the same BD write path
                        self.dma_write_bd_data(tile_addr.col, tile_addr.row, tile_addr.offset, data);
                    }
                }
            }

            _ => {
                use xdna_archspec::aie2::memory_map::MEM_TILE_DATA_MEMORY_END;
                // Could be register array writes - handle as data
                let tile = self.array.get_mut(tile_addr.col, tile_addr.row).unwrap();
                if offset <= MEM_TILE_DATA_MEMORY_END as usize && tile.write_data(offset, data) {
                    self.stats.data_bytes += data.len();
                }
            }
        }

        Ok(())
    }
}

/// Returns true unless `XDNA_EMU_WARN_GATED_ACCESS=0` is set.
/// Controls whether the gated-tile-access warning emits a log line;
/// the controller still records the site either way so tests can
/// observe dedup behavior.
fn warnings_enabled() -> bool {
    std::env::var("XDNA_EMU_WARN_GATED_ACCESS").as_deref() != Ok("0")
}
