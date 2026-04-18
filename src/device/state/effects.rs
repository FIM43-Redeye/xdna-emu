//! Tile-local register side effects and broadcast propagation.

use super::*;

impl DeviceState {
    /// Apply tile-local register side effects.
    ///
    /// Handles concerns that update tile-internal state without interacting
    /// with cross-tile subsystems (DMA engine, stream switch routing):
    /// cascade config, shim mux/demux, lock overflow/underflow clear, trace
    /// registers, edge detection, event port selection, and event broadcast
    /// with Event_Generate.
    ///
    /// Called from both `write_register()` and `mask_write_register()` after
    /// structured module dispatch has run.
    pub(super) fn apply_tile_local_effects(&mut self, col: u8, row: u8, offset: u32, value: u32) {
        let reg_layout = regdb::device_reg_layout();

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // 1. Cascade config (compute tiles only).
        // Accumulator_Control register per AM025 (offset from register database).
        //   Bit 0: cascade input direction (0=North, 1=West)
        //   Bit 1: cascade output direction (0=South, 1=East)
        if offset == reg_layout.cascade_config_offset && tile.is_compute() {
            tile.cascade_input_dir = (value & 0x1) as u8;
            tile.cascade_output_dir = ((value >> 1) & 0x1) as u8;
            log::info!(
                "Tile ({},{}) cascade config: input_dir={} output_dir={}",
                col, row,
                if tile.cascade_input_dir == 0 { "North" } else { "West" },
                if tile.cascade_output_dir == 0 { "South" } else { "East" },
            );
        }

        // 2. Shim mux/demux config (shim tiles only).
        // Mux_Config: selects source for switchbox South slave ports.
        // Demux_Config: selects destination for switchbox South master ports.
        if tile.is_shim() {
            if offset == reg_layout.shim_mux.mux_offset {
                tile.parse_shim_mux_config(value);
            } else if offset == reg_layout.shim_mux.demux_offset {
                tile.parse_shim_demux_config(value);
            }
        }

        // 3. Lock overflow/underflow status registers (write-to-clear).
        // Writing 1 to a bit clears that lock's overflow/underflow status.
        if tile.is_mem_tile() {
            if offset == reg_layout.memtile_locks_overflow_0 {
                tile.clear_lock_overflow_bits(0, 32, value);
            } else if offset == reg_layout.memtile_locks_overflow_1 {
                tile.clear_lock_overflow_bits(32, 64, value);
            } else if offset == reg_layout.memtile_locks_underflow_0 {
                tile.clear_lock_underflow_bits(0, 32, value);
            } else if offset == reg_layout.memtile_locks_underflow_1 {
                tile.clear_lock_underflow_bits(32, 64, value);
            }
        } else if tile.is_compute() {
            if offset == reg_layout.memory_locks_overflow {
                tile.clear_lock_overflow_bits(0, 16, value);
            } else if offset == reg_layout.memory_locks_underflow {
                tile.clear_lock_underflow_bits(0, 16, value);
            }
        } else if tile.is_shim() {
            if offset == reg_layout.shim_locks_overflow {
                tile.clear_lock_overflow_bits(0, 16, value);
            } else if offset == reg_layout.shim_locks_underflow {
                tile.clear_lock_underflow_bits(0, 16, value);
            }
        }

        // 4. Performance counter register routing.
        //
        // Offsets from aie-rt (xaiemlgbl_params.h). Each module's perf
        // counter block starts at a base address; we compute the in-block
        // offset and delegate to the Tile method that parses the fields.
        {
            use xdna_archspec::aie2::subsystems as subsystem;
            if tile.is_compute() {
                // Core module perf counters: 0x31500-0x3158C
                let base = subsystem::compute::core_performance::OFFSET_START;
                let end = subsystem::compute::core_performance::OFFSET_END;
                if offset >= base && offset < end {
                    tile.write_core_perf_register(offset - base, value);
                }
                // Memory module perf counters: 0x11000-0x11084
                let base = subsystem::compute::memory_performance::OFFSET_START;
                let end = subsystem::compute::memory_performance::OFFSET_END;
                if offset >= base && offset < end {
                    tile.write_mem_perf_register(offset - base, value);
                }
            } else if tile.is_mem_tile() {
                // MemTile perf counters: 0x91000-0x9108C
                let base = subsystem::memtile::performance::OFFSET_START;
                let end = subsystem::memtile::performance::OFFSET_END;
                if offset >= base && offset < end {
                    tile.write_mem_perf_register(offset - base, value);
                }
            } else if tile.is_shim() {
                // Shim PL module perf counters: 0x31000-0x31084
                let base = subsystem::shim::performance::OFFSET_START;
                let end = subsystem::shim::performance::OFFSET_END;
                if offset >= base && offset < end {
                    tile.write_core_perf_register(offset - base, value);
                }
            }
        }

        // 5. Trace register routing (offsets from register database).
        let ce = &reg_layout.core_events;
        let me = &reg_layout.memory_events;
        let mte = &reg_layout.memtile_events;
        if tile.is_compute() {
            if offset >= ce.trace_control_base && offset <= ce.trace_control_end {
                tile.core_trace.write_register(offset - ce.trace_control_base, value);
            }
            if offset >= me.trace_control_base && offset <= me.trace_control_end {
                tile.mem_trace.write_register(offset - me.trace_control_base, value);
            }
        } else if tile.is_mem_tile() {
            if offset >= mte.trace_control_base && offset <= mte.trace_control_end {
                tile.mem_trace.write_register(offset - mte.trace_control_base, value);
            }
        } else if tile.is_shim() {
            if offset >= ce.trace_control_base && offset <= ce.trace_control_end {
                tile.core_trace.write_register(offset - ce.trace_control_base, value);
            }
        }

        // 6. Edge detection event control registers (offsets from register database).
        if tile.is_compute() {
            if offset == ce.edge_detection {
                Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);
            }
            if offset == me.edge_detection {
                Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, false);
            }
        } else if tile.is_mem_tile() {
            if offset == mte.edge_detection {
                Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, true);
            }
        } else if tile.is_shim() {
            if offset == ce.edge_detection {
                Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);
            }
        }

        // 7. Event port selection registers (offsets from register database).
        // Configure which physical stream switch ports map to logical event
        // ports 0-7 for PORT_RUNNING/IDLE/STALLED trace events.
        // These registers live in the stream switch address space (0x3FF00),
        // not the event subsystem, so they can't go through EventModule.
        let port_sel_base = match tile.tile_type {
            TileType::Compute | TileType::Shim => {
                ce.event_port_select.map(|[r0, r1]| (r0, r1))
            }
            TileType::MemTile => {
                mte.event_port_select.map(|[r0, r1]| (r0, r1))
            }
        };
        if let Some((reg0, reg1)) = port_sel_base {
            if offset == reg0 || offset == reg1 {
                let base_slot = if offset == reg0 { 0 } else { 4 };
                for i in 0..4usize {
                    let byte = ((value >> (i * 8)) & 0xFF) as u8;
                    let port_idx = byte & 0x1F;
                    let is_master = (byte & 0x20) != 0;
                    tile.event_port_selection[base_slot + i] = Some((port_idx, is_master));
                }
                log::debug!(
                    "Tile({},{}) event port sel @0x{:X}: {:?}",
                    col, row, offset, &tile.event_port_selection[base_slot..base_slot+4]
                );
            }
        }

        // 8. Timer register routing.
        //
        // Each module has a timer block at a known base address. Offsets
        // within the block are consistent across modules (see timer.rs).
        // Compute tiles have core_timer + mem_timer; memtile has mem_timer;
        // shim has core_timer (PL module timer).
        {
            use xdna_archspec::aie2::subsystems as subsystem;
            if tile.is_compute() {
                let base = subsystem::compute::core_timer::OFFSET_START;
                let end = subsystem::compute::core_timer::OFFSET_END;
                if offset >= base && offset < end {
                    tile.core_timer.write_register(offset - base, value);
                }
                let base = subsystem::compute::memory_timer::OFFSET_START;
                let end = subsystem::compute::memory_timer::OFFSET_END;
                if offset >= base && offset < end {
                    tile.mem_timer.write_register(offset - base, value);
                }
            } else if tile.is_mem_tile() {
                let base = subsystem::memtile::timer::OFFSET_START;
                let end = subsystem::memtile::timer::OFFSET_END;
                if offset >= base && offset < end {
                    tile.mem_timer.write_register(offset - base, value);
                }
            } else if tile.is_shim() {
                let base = subsystem::shim::timer::OFFSET_START;
                let end = subsystem::shim::timer::OFFSET_END;
                if offset >= base && offset < end {
                    tile.core_timer.write_register(offset - base, value);
                }
            }
        }

        // 8. Core debug register routing (compute tiles only).
        //
        // CoreDebugState uses absolute tile offsets internally, matching the
        // hardware register map (Core_Control=0x32000, Core_Status=0x32004,
        // Debug_Control0=0x32010, etc.).
        if tile.is_compute() {
            tile.core_debug.write_register(offset, value);
        }

        // 9. Event module register routing.
        //
        // Each tile module has an event subsystem with broadcast channels,
        // combo events, group events, port events, and Event_Generate.
        // The EventModule handles all of these via its register interface,
        // which masks the offset with `& 0xFFFF` to get the subsystem-local
        // offset. We pass the full tile offset so the masking works correctly.
        //
        // Event subsystem offsets (from archspec):
        //   Compute core: 0x34008-0x34524 (module prefix 0x30000)
        //   Compute mem:  0x14008-0x14520 (module prefix 0x10000)
        //   MemTile:      0x94008-0x94524 (module prefix 0x90000)
        //   Shim (PL):    0x34008-0x34518 (module prefix 0x30000)
        {
            use xdna_archspec::aie2::subsystems as subsystem;
            if tile.is_compute() {
                let base = subsystem::compute::core_event::OFFSET_START;
                let end = subsystem::compute::core_event::OFFSET_END;
                if offset >= base && offset < end {
                    if let Some(ref mut em) = tile.core_events {
                        em.write_register(offset, value);
                    }
                }
                let base = subsystem::compute::memory_event::OFFSET_START;
                let end = subsystem::compute::memory_event::OFFSET_END;
                if offset >= base && offset < end {
                    if let Some(ref mut em) = tile.mem_events {
                        em.write_register(offset, value);
                    }
                }
            } else if tile.is_mem_tile() {
                let base = subsystem::memtile::event::OFFSET_START;
                let end = subsystem::memtile::event::OFFSET_END;
                if offset >= base && offset < end {
                    if let Some(ref mut em) = tile.mem_events {
                        em.write_register(offset, value);
                    }
                }
            } else if tile.is_shim() {
                let base = subsystem::shim::event::OFFSET_START;
                let end = subsystem::shim::event::OFFSET_END;
                if offset >= base && offset < end {
                    if let Some(ref mut em) = tile.core_events {
                        em.write_register(offset, value);
                    }
                }
            }
        }

        // 10. Interrupt controller register routing (shim tiles only).
        //
        // L1: per-switch interrupt controller in PL module (absolute offsets).
        // L2: NoC interrupt controller (absolute offsets).
        if tile.is_shim() {
            if let Some(ref mut l1) = tile.l1_irq {
                l1.write_register(offset, value);
            }
            if let Some(ref mut l2) = tile.l2_irq {
                l2.write_register(offset, value);
            }
        }

        // 11. Event_Generate: fire on trace units and broadcast.
        //
        // The EventModule above handles the event register write, but the
        // trace units and broadcast propagation also need to be notified.
        // Event_Generate offset is the first register in the event block.
        let is_event_generate = match tile.tile_type {
            TileType::Compute => offset == ce.event_generate || offset == me.event_generate,
            TileType::MemTile => offset == mte.event_generate,
            TileType::Shim => offset == ce.event_generate,
        };
        if is_event_generate {
            let event_id = (value & 0x7F) as u8;
            log::info!(
                "Tile({},{}) Event_Generate: event_id={} (offset=0x{:X})",
                col, row, event_id, offset
            );

            // Fire the event directly on local trace units.
            tile.core_trace.notify_event(event_id, 0);
            tile.mem_trace.notify_event(event_id, 0);

            // Check broadcast channel mapping in the EventModule: if the
            // generated event matches any broadcast channel's configured
            // event, queue the BROADCAST_N event for column propagation.
            let broadcast_base = match tile.tile_type {
                TileType::Compute | TileType::Shim => 107u8, // Core/PL module
                TileType::MemTile => 142u8,
            };
            let events_ref = match tile.tile_type {
                TileType::Compute | TileType::Shim => tile.core_events.as_ref(),
                TileType::MemTile => tile.mem_events.as_ref(),
            };
            if let Some(em) = events_ref {
                for ch in 0..16u8 {
                    let ch_event = em.broadcast.read_channel(ch as usize) as u8;
                    if ch_event == event_id && event_id != 0 {
                        let broadcast_hw_id = broadcast_base + ch;
                        log::info!(
                            "Tile({},{}) Event_Generate: event {} -> BROADCAST_{} (hw_id={})",
                            col, row, event_id, ch, broadcast_hw_id
                        );
                        tile.pending_broadcasts.push(broadcast_hw_id);
                    }
                }
            }
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
