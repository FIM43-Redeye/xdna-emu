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
        // Snapshot the simulation cycle up front so we can pass it to trace
        // units without needing a second borrow of self.array below.
        let current_cycle = self.array.current_cycle;

        let tile = match self.array.get_mut(col, row) {
            Some(t) => t,
            None => return,
        };

        // 0. Control_Packet_Handler_Status (write-1-to-clear sticky bits).
        // Compute: 0x3FF30, MemTile: 0xB0F30. Bits [3:0] are
        // [3] Tlast_Error / [2] SLVERR_On_Access /
        // [1] Second_Header_Parity_Error / [0] First_Header_Parity_Error
        // (AM025 Tile_Control_Packet_Handler_Status). Write-1-to-clear.
        // Reads return tile.pkt_handler_status; writes clear bits whose
        // mask bit is 1. tile.registers is bypassed for this offset.
        if (tile.is_compute() && offset == 0x3FF30) || (tile.is_mem() && offset == 0xB0F30) {
            tile.pkt_handler_status &= !(value & 0xF);
            tile.registers.remove(&offset);
        }

        // 0a. Tile_Control isolation bits.
        // Compute + Shim: 0x36030, MemTile: 0x96030. Low 4 bits = isolation
        // per AM025 Tile_Control field layout (S/W/N/E in bits 0..3),
        // matching aie-rt's `XAIE_ISOLATE_*_MASK` constants. Snapshot them
        // onto `tile.isolation` so cross-tile gates (stream-switch routing,
        // NeighborLocks, NeighborMemory) can consult a single byte instead
        // of re-decoding the register on every check. Higher bits of
        // Tile_Control (clock-gating etc.) still flow through the generic
        // register store unchanged.
        //
        // Shim tiles only participate in stream-switch routing gating;
        // NeighborMemory and NeighborLocks don't apply (shim has no
        // executing core that does cross-tile quadrant ops). Of the routing
        // directions, the gate that actually fires for shim is memtile->
        // shim south-bound, which checks shim's NORTH bit per the inbound-
        // direction rule. shim.SOUTH/WEST/EAST snapshot too, but no
        // current routing path consults them.
        let is_tile_control = (tile.is_compute() && offset == 0x36030)
            || (tile.is_mem() && offset == 0x96030)
            || (tile.is_shim() && offset == 0x36030);
        if is_tile_control {
            tile.isolation = (value & super::super::tile::isolation::ALL_DIRECTIONS as u32) as u8;
        }

        // 1. Cascade config (compute tiles only).
        // Accumulator_Control register per AM025 (offset from register database).
        //   Bit 0: cascade input direction (0=North, 1=West)
        //   Bit 1: cascade output direction (0=South, 1=East)
        if offset == reg_layout.cascade_config_offset && tile.is_compute() {
            tile.cascade_input_dir = (value & 0x1) as u8;
            tile.cascade_output_dir = ((value >> 1) & 0x1) as u8;
            log::info!(
                "Tile ({},{}) cascade config: input_dir={} output_dir={}",
                col,
                row,
                if tile.cascade_input_dir == 0 {
                    "North"
                } else {
                    "West"
                },
                if tile.cascade_output_dir == 0 {
                    "South"
                } else {
                    "East"
                },
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
        if tile.is_mem() {
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
            } else if tile.is_mem() {
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
        } else if tile.is_mem() {
            if offset >= mte.trace_control_base && offset <= mte.trace_control_end {
                tile.mem_trace.write_register(offset - mte.trace_control_base, value);
            }
        } else if tile.is_shim() {
            if offset >= ce.trace_control_base && offset <= ce.trace_control_end {
                tile.core_trace.write_register(offset - ce.trace_control_base, value);
            }
        }

        // 5b. MemTile DMA Event Channel Selection register (0xA06A0).
        //
        // Per AM020 / aie-rt xaiemlgbl_params.h, this register selects which
        // physical DMA channel feeds each of the four memtile DMA event
        // broadcast lines (S2MM_SEL0/SEL1, MM2S_SEL0/SEL1). At reset all
        // SEL slots point at channel 0; software must program this register
        // to redirect a SEL slot at any other channel.
        //
        // Read by `crate::trace::memtile_event_to_hw_ids` via the coordinator's
        // memtile DMA event dispatch path.
        if tile.is_mem() && offset == 0xA06A0 {
            tile.memtile_dma_event_chan_sel = value;
            log::info!(
                "Tile({},{}) DMA_Event_Channel_Selection = 0x{:08X} (S2MM_SEL0={} S2MM_SEL1={} MM2S_SEL0={} MM2S_SEL1={})",
                col,
                row,
                value,
                value & 0x7,
                (value >> 8) & 0x7,
                (value >> 16) & 0x7,
                (value >> 24) & 0x7,
            );
        }

        // 6. Edge detection event control registers (offsets from register database).
        if tile.is_compute() {
            if offset == ce.edge_detection {
                Tile::configure_edge_detectors(&mut tile.core_edge_detectors, value, false);
            }
            if offset == me.edge_detection {
                Tile::configure_edge_detectors(&mut tile.mem_edge_detectors, value, false);
            }
        } else if tile.is_mem() {
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
        let port_sel_base = match tile.tile_kind {
            TileKind::Compute | TileKind::ShimNoc | TileKind::ShimPl => {
                ce.event_port_select.map(|[r0, r1]| (r0, r1))
            }
            TileKind::Mem => mte.event_port_select.map(|[r0, r1]| (r0, r1)),
        };
        if let Some((reg0, reg1)) = port_sel_base {
            if offset == reg0 || offset == reg1 {
                let base_slot = if offset == reg0 { 0 } else { 4 };
                for i in 0..4usize {
                    let byte = ((value >> (i * 8)) & 0xFF) as u8;
                    let port_idx = byte & 0x1F;
                    let is_master = (byte & 0x20) != 0;
                    tile.event_port_selection[base_slot + i] = Some((port_idx, is_master));
                    // Reset edge-trigger memory for the reassigned slot so
                    // the first observed transition fires as a rising edge.
                    tile.prev_port_state[base_slot + i] = (false, false, false);
                }
                log::debug!(
                    "Tile({},{}) event port sel @0x{:X}: {:?}",
                    col,
                    row,
                    offset,
                    &tile.event_port_selection[base_slot..base_slot + 4]
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
            } else if tile.is_mem() {
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
            } else if tile.is_mem() {
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
        let is_event_generate = match tile.tile_kind {
            TileKind::Compute => offset == ce.event_generate || offset == me.event_generate,
            TileKind::Mem => offset == mte.event_generate,
            TileKind::ShimNoc | TileKind::ShimPl => offset == ce.event_generate,
        };
        if is_event_generate {
            let event_id = (value & 0x7F) as u8;
            log::info!(
                "Tile({},{}) Event_Generate: event_id={} (offset=0x{:X}) cycle={}",
                col,
                row,
                event_id,
                offset,
                current_cycle
            );

            // Fire the event directly on local trace units. Use the array's
            // current_cycle so trace unit deltas reflect real simulation time;
            // passing a hardcoded 0 here causes every generated event to look
            // like it fired at cycle 0.
            tile.core_trace.notify_event(event_id, current_cycle, None);
            tile.mem_trace.notify_event(event_id, current_cycle, None);

            // Check broadcast channel mapping in the EventModule: if the
            // generated event matches any broadcast channel's configured
            // event, queue the channel number for column propagation.
            //
            // Note: `pending_broadcasts` stores the channel number (0..15),
            // not a hw_id. Per-module hw_id translation happens at the
            // receiving tile in `propagate_broadcasts`, since each module
            // type sees BROADCAST_N at a different hw_id (compute core/mem
            // = 107+N, shim PL_A = 110+N, memtile = 142+N).
            let events_ref = match tile.tile_kind {
                TileKind::Compute | TileKind::ShimNoc | TileKind::ShimPl => tile.core_events.as_ref(),
                TileKind::Mem => tile.mem_events.as_ref(),
            };
            if let Some(em) = events_ref {
                for ch in 0..16u8 {
                    let ch_event = em.broadcast.read_channel(ch as usize) as u8;
                    if ch_event == event_id && event_id != 0 {
                        log::info!(
                            "Tile({},{}) Event_Generate: event {} -> BROADCAST channel {}",
                            col,
                            row,
                            event_id,
                            ch,
                        );
                        tile.pending_broadcasts.push(ch);
                    }
                }
            }
        }
    }

    /// Propagate pending broadcast events from a tile across the array.
    ///
    /// Real hardware's broadcast network spans the whole array in four
    /// directions (south/west/north/east). Each tile module has per-direction
    /// block-mask registers (`EVENT_BROADCAST_BLOCK_*`) that gate outbound
    /// propagation per channel. The BFS below walks tile-to-tile and at
    /// every hop consults the source-side tile's block mask for the channel
    /// in the relevant outbound direction; a blocked direction prunes that
    /// branch.
    ///
    /// Trace-prepare today emits no block-mask writes (so the masks stay at
    /// reset = 0 = no blocking, and the broadcast effectively floods the
    /// array), but honoring the masks here keeps the model HW-accurate for
    /// any CDO that does program them.
    ///
    /// Why a flood is needed: the trace planner can place `Event_Generate`
    /// on a spare column (e.g. (2,0) on npu1_2col) and expect every column's
    /// trace units to receive the corresponding BROADCAST_15 hw_id. The
    /// older column-only loop dropped the start signal for the application
    /// column's trace units. Re-enabling cross-column propagation depends
    /// on the multicast deadlock fix in stream_switch::step_packet_routes
    /// (task #28); see
    /// `docs/superpowers/findings/2026-05-11-emu-trace-widened-distributed-routing.md`.
    pub(crate) fn propagate_broadcasts(&mut self, col: u8, source_row: u8) {
        use crate::device::events::broadcast::BroadcastDir;

        let current_cycle = self.array.current_cycle;

        let channels = if let Some(tile) = self.array.get_mut(col, source_row) {
            tile.drain_pending_broadcasts()
        } else {
            return;
        };

        if channels.is_empty() {
            return;
        }

        // Per-module hw_id base for the BROADCAST_N event (per AM020 /
        // aie-rt xaie_events_aieml.h):
        //   Core module        : BROADCAST_0 = 107
        //   Compute mem module : BROADCAST_0 = 107
        //   Shim PL module     : BROADCAST_A_0 = 110
        //   MemTile event mod  : BROADCAST_0  = 142
        // hw_id 0 is the EVENT_NONE sentinel; notify_*_trace_event filters
        // it out for tile kinds that lack the corresponding module side.
        const CORE_BROADCAST_BASE: u8 = 107;
        const SHIM_PL_BROADCAST_BASE: u8 = 110;
        const MEMTILE_BROADCAST_BASE: u8 = 142;

        let cols = self.array.cols();
        let rows = self.array.rows();

        for &channel in &channels {
            log::info!(
                "Propagating BROADCAST channel {} from tile ({},{}) at cycle {}",
                channel,
                col,
                source_row,
                current_cycle,
            );

            // BFS across (col,row) tiles. `visited` is a flat (cols * rows)
            // bitmap stored row-major as Vec<bool>. Frontier holds (col,row)
            // coordinates to process.
            let mut visited = vec![false; cols as usize * rows as usize];
            let idx_of = |c: u8, r: u8| (c as usize) * (rows as usize) + (r as usize);

            let mut frontier: Vec<(u8, u8)> = vec![(col, source_row)];
            visited[idx_of(col, source_row)] = true;

            while let Some((c, r)) = frontier.pop() {
                // Notify this tile of the broadcast hit. notify_*_trace_event
                // filters hw_id=0 (no module on this side), so memtiles and
                // shims only receive on their valid module.
                let outbound_dirs: Vec<BroadcastDir>;
                {
                    let tile = match self.array.get_mut(c, r) {
                        Some(t) => t,
                        None => continue,
                    };
                    let core_pc = Some(tile.core.pc);
                    let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                        TileKind::Compute => (CORE_BROADCAST_BASE + channel, CORE_BROADCAST_BASE + channel),
                        TileKind::ShimNoc | TileKind::ShimPl => (SHIM_PL_BROADCAST_BASE + channel, 0),
                        TileKind::Mem => (0, MEMTILE_BROADCAST_BASE + channel),
                    };
                    tile.notify_core_trace_event(core_hw_id, current_cycle, core_pc);
                    tile.notify_mem_trace_event(mem_hw_id, current_cycle, None);

                    // Determine which directions propagation is allowed in
                    // from THIS tile (the source side of each outbound hop).
                    // Use whichever EventModule exists for this tile kind;
                    // both default to no-blocking until CDO programs the
                    // EVENT_BROADCAST_BLOCK_* registers.
                    let bcfg = tile.core_events.as_ref().or(tile.mem_events.as_ref()).map(|m| &m.broadcast);
                    outbound_dirs = match bcfg {
                        Some(b) => b.allowed_directions(channel as usize),
                        None => BroadcastDir::ALL.to_vec(),
                    };
                }

                // Enqueue unvisited neighbors in allowed directions.
                for dir in outbound_dirs {
                    let neighbor = match dir {
                        BroadcastDir::South => {
                            if r > 0 {
                                Some((c, r - 1))
                            } else {
                                None
                            }
                        }
                        BroadcastDir::North => {
                            if r + 1 < rows {
                                Some((c, r + 1))
                            } else {
                                None
                            }
                        }
                        BroadcastDir::East => {
                            if c + 1 < cols {
                                Some((c + 1, r))
                            } else {
                                None
                            }
                        }
                        BroadcastDir::West => {
                            if c > 0 {
                                Some((c - 1, r))
                            } else {
                                None
                            }
                        }
                    };
                    if let Some((nc, nr)) = neighbor {
                        let nidx = idx_of(nc, nr);
                        if !visited[nidx] {
                            visited[nidx] = true;
                            frontier.push((nc, nr));
                        }
                    }
                }
            }
        }
    }
}
