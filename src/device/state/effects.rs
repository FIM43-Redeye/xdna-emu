//! Tile-local register side effects and broadcast propagation.

use super::*;
use crate::device::tile::BroadcastProvenance;
#[cfg(test)]
use crate::device::tile::PendingBroadcast;

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

        // 11. Event_Generate: fire on trace units, broadcast, and (Tier B)
        // async-error pipeline.
        //
        // The EventModule above handles the event register write, but the
        // trace units, broadcast propagation, and the firmware async-error
        // path also need to be notified. Event_Generate offset is the first
        // register in the event block; the offset selects which module
        // fired (core vs mem on compute tiles), which in turn determines
        // the Tier B origin used for categorization.
        use xdna_archspec::aie2::async_errors::{is_error_event, AieErrorOrigin};
        let origin = match tile.tile_kind {
            TileKind::Compute if offset == ce.event_generate => Some(AieErrorOrigin::Core),
            TileKind::Compute if offset == me.event_generate => Some(AieErrorOrigin::Mem),
            TileKind::Mem if offset == mte.event_generate => Some(AieErrorOrigin::MemTile),
            TileKind::ShimNoc | TileKind::ShimPl if offset == ce.event_generate => Some(AieErrorOrigin::Pl),
            _ => None,
        };
        // Capture event_id and origin before the tile borrow ends so that
        // self.async_errors can be borrowed mutably after the tile scope closes.
        let tier_b = if let Some(origin) = origin {
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
            // = 107+N, shim PL_A = 110+N, memtile = 142+N). Shared with the
            // hardware error path so the scan logic cannot drift.
            tile.seed_broadcasts_for_event(event_id);
            // Tier A interrupt path: a software-generated event is also
            // offered to this tile's L1 interrupt controller (shim only).
            // On latch, L1 queues its IRQ_NO into pending_broadcasts so the
            // existing propagate_broadcasts transport carries it to L2.
            tile.tap_l1_interrupt(event_id);

            Some((origin, event_id))
        } else {
            None
        };
        // tile borrow ends here. self.async_errors can now be borrowed.

        // Tier B firmware async-error path: parallel to Tier A. On real
        // silicon, firmware delivers errors via mailbox regardless of
        // AIE L1/L2 enable state -- so this hook fires at event-generation,
        // not after L1 latches. The two paths are independent: an error
        // populates Tier B's cache + ring whether or not L1 was enabled.
        if let Some((origin, event_id)) = tier_b {
            if is_error_event(event_id, origin) {
                self.async_errors.record_error(col, row, origin, event_id, current_cycle);
            }
        }
    }

    /// Dijkstra wavefront over the broadcast adjacency for one channel.
    ///
    /// Returns `(col, row, origin_D)` for every reached tile, where `origin_D`
    /// is the minimum cumulative propagation delay from the source (edge cost
    /// `d_h` for an east/west hop, `d_v` for north/south). The OR-tree
    /// re-broadcasts on first arrival, so earliest arrival wins = shortest path
    /// (AM020 Ch2). Honors per-tile broadcast block masks (a blocked direction
    /// is a removed edge). Source `origin_D = 0`. With `d_h = d_v = 0` every
    /// `origin_D = 0` and the reached set equals the legacy flood's reach.
    pub(crate) fn broadcast_origin_d(
        &self,
        col: u8,
        source_row: u8,
        channel: u8,
        d_h: u32,
        d_v: u32,
    ) -> Vec<(u8, u8, u32)> {
        use crate::device::events::broadcast::BroadcastDir;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let cols = self.array.cols();
        let rows = self.array.rows();
        let idx_of = |c: u8, r: u8| (c as usize) * (rows as usize) + (r as usize);

        // best[idx] = settled min cost, or u32::MAX if unreached.
        let mut best = vec![u32::MAX; cols as usize * rows as usize];
        // Min-heap on (cost, col, row). Reverse for min-first.
        let mut heap: BinaryHeap<Reverse<(u32, u8, u8)>> = BinaryHeap::new();

        best[idx_of(col, source_row)] = 0;
        heap.push(Reverse((0, col, source_row)));

        let mut out: Vec<(u8, u8, u32)> = Vec::new();

        while let Some(Reverse((cost, c, r))) = heap.pop() {
            if cost > best[idx_of(c, r)] {
                continue; // stale heap entry
            }
            out.push((c, r, cost));

            // Allowed outbound directions from THIS tile (source side of the hop).
            let bcfg = self
                .array
                .get(c, r)
                .and_then(|t| t.core_events.as_ref().or(t.mem_events.as_ref()))
                .map(|m| &m.broadcast);
            let dirs = match bcfg {
                Some(b) => b.allowed_directions(channel as usize),
                None => BroadcastDir::ALL.to_vec(),
            };

            for dir in dirs {
                let (nc, nr, step) = match dir {
                    BroadcastDir::South if r > 0 => (c, r - 1, d_v),
                    BroadcastDir::North if r + 1 < rows => (c, r + 1, d_v),
                    BroadcastDir::East if c + 1 < cols => (c + 1, r, d_h),
                    BroadcastDir::West if c > 0 => (c - 1, r, d_h),
                    _ => continue,
                };
                let ncost = cost + step;
                let nidx = idx_of(nc, nr);
                if ncost < best[nidx] {
                    best[nidx] = ncost;
                    heap.push(Reverse((ncost, nc, nr)));
                }
            }
        }
        out
    }

    /// Per-module origin_D rows for the single-source flood, for the SP-4b
    /// sidecar. Maps each reached tile to its module-kind row(s) carrying
    /// `origin_D + intra_off` (= core_delay/mem_delay -- the physical timer-reset
    /// arrival, NOT the max_delay-complement `*_target`). Module kinds match the
    /// engine's decoder convention via the Python loader's module->pkt_type map.
    pub(crate) fn origin_d_table(
        &self,
        col: u8,
        source_row: u8,
        channel: u8,
        d_h: u32,
        d_v: u32,
        core_off: u32,
        mem_off: u32,
    ) -> Vec<(u8, u8, &'static str, u32)> {
        let mut out = Vec::new();
        for (c, r, origin_d) in self.broadcast_origin_d(col, source_row, channel, d_h, d_v) {
            match self.array.get(c, r).map(|t| t.tile_kind) {
                Some(TileKind::Compute) => {
                    out.push((c, r, "core", origin_d + core_off));
                    out.push((c, r, "mem", origin_d + mem_off));
                }
                Some(TileKind::Mem) => out.push((c, r, "memtile", origin_d + mem_off)),
                Some(TileKind::ShimNoc) | Some(TileKind::ShimPl) => {
                    out.push((c, r, "shim", origin_d + core_off))
                }
                None => {}
            }
        }
        out
    }

    /// Propagate pending broadcast events from a tile across the array.
    ///
    /// Real hardware's broadcast network spans the whole array in four
    /// directions (south/west/north/east). Each tile module has per-direction
    /// block-mask registers (`EVENT_BROADCAST_BLOCK_*`) that gate outbound
    /// propagation per channel. A Dijkstra wavefront (`broadcast_origin_d`)
    /// walks tile-to-tile and at every hop consults the source-side tile's
    /// block mask for the channel in the relevant outbound direction; a
    /// blocked direction prunes that branch.
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
        let d_h = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_HORIZONTAL as u32;
        let d_v = xdna_archspec::aie2::timing::BROADCAST_PER_HOP_VERTICAL as u32;
        let core_off = xdna_archspec::aie2::timing::BROADCAST_INTRA_TILE_CORE_OFFSET as u32;
        let mem_off = xdna_archspec::aie2::timing::BROADCAST_INTRA_TILE_MEM_OFFSET as u32;
        self.propagate_broadcasts_with_timing(col, source_row, d_h, d_v, core_off, mem_off);
    }

    pub(crate) fn propagate_broadcasts_with_timing(
        &mut self,
        col: u8,
        source_row: u8,
        d_h: u32,
        d_v: u32,
        core_off: u32,
        mem_off: u32,
    ) {
        let current_cycle = self.array.current_cycle;

        let channels = if let Some(tile) = self.array.get_mut(col, source_row) {
            tile.drain_pending_broadcasts()
        } else {
            return;
        };

        if channels.is_empty() {
            return;
        }

        // Per-module BROADCAST_N event id = EventModuleType::broadcast_event_base
        // + channel (aie-rt xaie_events_aieml.h, via the single events accessor):
        //   Core/Mem 107, Shim PL 110, MemTile 142.
        // hw_id 0 is the EVENT_NONE sentinel; notify_*_trace_event filters it for
        // tile kinds lacking that module side.
        use crate::device::events::EventModuleType;

        for &pb in &channels {
            // SP-5a: record a channel-15 flood SOURCE only for genuine
            // originations. An L1-relay re-emission of channel 15 (interrupt
            // transport reusing the timer-reset broadcast line) is not a
            // timer-reset source and must not pollute the single-source guard.
            if pb.channel == 15 && pb.provenance == BroadcastProvenance::Originated {
                self.channel15_flood_sources.insert((col, source_row));
            }

            log::info!(
                "Propagating BROADCAST channel {} from tile ({},{}) at cycle {}",
                pb.channel,
                col,
                source_row,
                current_cycle,
            );

            let reached = self.broadcast_origin_d(col, source_row, pb.channel, d_h, d_v);

            // max_delay over all reached modules: for each tile the worst-case
            // intra-tile offset is core_off.max(mem_off), so both
            // (max_delay - core_delay) and (max_delay - mem_delay) stay >= 0.
            // At zero consts max_delay = 0 and every target = 0, byte-identical
            // to the pre-SP-1 behavior.
            let max_delay = reached.iter().map(|&(_, _, o)| o + core_off.max(mem_off)).max().unwrap_or(0);

            for (c, r, origin_d) in reached {
                let core_delay = origin_d + core_off;
                let mem_delay = origin_d + mem_off;
                let core_target = (max_delay - core_delay) as u64;
                let mem_target = (max_delay - mem_delay) as u64;
                let tile = match self.array.get_mut(c, r) {
                    Some(t) => t,
                    None => continue,
                };
                let core_pc = Some(tile.core.pc);
                let (core_hw_id, mem_hw_id) = match tile.tile_kind {
                    TileKind::Compute => (
                        EventModuleType::Core.broadcast_event_base() + pb.channel,
                        EventModuleType::Memory.broadcast_event_base() + pb.channel,
                    ),
                    TileKind::ShimNoc | TileKind::ShimPl => {
                        (EventModuleType::Pl.broadcast_event_base() + pb.channel, 0)
                    }
                    TileKind::Mem => (0, EventModuleType::MemTile.broadcast_event_base() + pb.channel),
                };
                // SP-2: give the trace units the same skew baseline the timer
                // holds (core_target/mem_target = max_delay - module_delay). Set
                // BEFORE the notify below so a tile whose start_event is this
                // broadcast id arms with the offset already applied (design Sec.4.5).
                tile.core_trace.set_origin_offset(core_target);
                tile.mem_trace.set_origin_offset(mem_target);
                tile.notify_core_trace_event_with_target(core_hw_id, current_cycle, core_pc, core_target);
                tile.notify_mem_trace_event_with_target(mem_hw_id, current_cycle, None, mem_target);
                if let Some(ref mut l2) = tile.l2_irq {
                    l2.signal_interrupt(pb.channel);
                }
                // Received broadcasts also feed this tile's L1 (shim):
                // the PL module sees BROADCAST channel N as event id
                // EventModuleType::Pl.broadcast_event_base() (= 110) + N.
                // On latch L1 queues its IRQ_NO into this tile's pending_broadcasts;
                // the fixpoint driver re-propagates it (L1 output -> L2).
                if tile.l1_irq.is_some() {
                    let ev = EventModuleType::Pl.broadcast_event_base() + pb.channel;
                    tile.tap_l1_interrupt(ev);
                }
            }
        }
    }

    /// Drive `propagate_broadcasts` to a fixed point.
    ///
    /// A broadcast delivered to a shim tile can latch its L1 controller,
    /// which queues a new IRQ_NO broadcast (the L1 output). That second
    /// broadcast must also propagate (to reach L2) within the same
    /// dispatch. `propagate_broadcasts` drains once, so loop until no tile
    /// has pending broadcasts, bounded to avoid pathological cycles.
    ///
    /// The cap (8) comfortably exceeds the real L1->L2 chain depth (one
    /// hop); hitting it indicates a misconfiguration loop and is logged
    /// rather than silently spun.
    ///
    /// Deferred optimization (correctness-phase, not yet): `propagate_broadcasts`
    /// could return whether it queued new work, letting the common no-L1-latch
    /// case skip the all-tiles scan entirely. Not worth the signature churn until
    /// dispatch profiling shows this as a hotspot.
    pub(crate) fn propagate_broadcasts_fixpoint(&mut self, col: u8, source_row: u8) {
        const MAX_ITERS: u32 = 8;
        self.propagate_broadcasts(col, source_row);
        let mut pending: Vec<(u8, u8)> = Vec::new();
        for iter in 0..MAX_ITERS {
            pending.clear();
            for c in 0..self.array.cols() {
                for r in 0..self.array.rows() {
                    if let Some(t) = self.array.get(c, r) {
                        if !t.pending_broadcasts.is_empty() {
                            pending.push((c, r));
                        }
                    }
                }
            }
            if pending.is_empty() {
                return;
            }
            for &(c, r) in &pending {
                self.propagate_broadcasts(c, r);
            }
            // Warn AFTER the final propagation (don't skip work on the last
            // iteration); fires once, not per-iteration.
            if iter == MAX_ITERS - 1 {
                log::warn!(
                    "propagate_broadcasts_fixpoint hit iteration cap ({}) \
                     starting from ({},{}); possible broadcast/interrupt loop",
                    MAX_ITERS,
                    col,
                    source_row
                );
            }
        }
    }
}

#[cfg(test)]
mod interrupt_path_tests {
    use super::*;

    impl DeviceState {
        /// Fire Event_Generate on a tile by calling apply_tile_local_effects
        /// with the correct offset for that tile kind (test-only helper).
        ///
        /// For shim tiles, Event_Generate lives at `ce.event_generate`
        /// (same offset that the production code matches for ShimNoc/ShimPl).
        /// This keeps the test honest: it exercises the real production path.
        fn fire_event_generate_for_test(&mut self, col: u8, row: u8, event_id: u8) {
            let reg_layout = regdb::device_reg_layout();
            let offset = match self.array.get(col, row).map(|t| t.tile_kind).expect("tile must exist") {
                // mem-side (me.event_generate) intentionally omitted: Compute
                // tiles have no l1_irq, so this helper is shim-scoped.
                TileKind::Compute => reg_layout.core_events.event_generate,
                TileKind::ShimNoc | TileKind::ShimPl => reg_layout.core_events.event_generate,
                TileKind::Mem => reg_layout.memtile_events.event_generate,
            };
            self.apply_tile_local_effects(col, row, offset, event_id as u32);
        }
    }

    #[test]
    fn event_generate_on_shim_latches_l1_and_queues_irq_no() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (0u8, 0u8);
        {
            let t = dev.array.get_mut(col, row).unwrap();
            let l1 = t.l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 7);
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 5);
        }
        dev.fire_event_generate_for_test(col, row, 7);
        let t = dev.array.get(col, row).unwrap();
        let l1 = t.l1_irq.as_ref().unwrap();
        assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0, "L1 status must latch");
        assert!(t.pending_broadcasts.iter().any(|pb| pb.channel == 5), "IRQ_NO 5 must be queued");
    }

    #[test]
    fn broadcast_delivery_latches_shim_l2_on_matching_channel() {
        use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_STATUS};
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (0u8, 0u8);
        dev.array
            .get_mut(col, row)
            .unwrap()
            .l2_irq
            .as_mut()
            .unwrap()
            .write_register(L2_REG_ENABLE, 1 << 5);
        dev.array
            .get_mut(col, row)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(5));
        dev.propagate_broadcasts(col, row);
        let l2 = dev.array.get(col, row).unwrap().l2_irq.as_ref().unwrap();
        assert_ne!(
            l2.read_register(L2_REG_STATUS).unwrap() & (1 << 5),
            0,
            "L2 channel 5 must latch on broadcast delivery"
        );
    }

    #[test]
    fn broadcast_delivery_does_not_latch_disabled_l2_channel() {
        use crate::device::interrupts::L2_REG_STATUS;
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (0u8, 0u8);
        dev.array
            .get_mut(col, row)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(5));
        dev.propagate_broadcasts(col, row);
        let l2 = dev.array.get(col, row).unwrap().l2_irq.as_ref().unwrap();
        assert_eq!(l2.read_register(L2_REG_STATUS).unwrap(), 0, "disabled L2 channel must not latch");
    }

    #[test]
    fn broadcast_bases_resolve_from_event_module_accessor() {
        use crate::device::events::EventModuleType;
        // The de-duped flood must resolve the same per-module bases the single
        // accessor holds (aie-rt xaie_events_aieml.h).
        assert_eq!(EventModuleType::Core.broadcast_event_base(), 107);
        assert_eq!(EventModuleType::Memory.broadcast_event_base(), 107);
        assert_eq!(EventModuleType::Pl.broadcast_event_base(), 110);
        assert_eq!(EventModuleType::MemTile.broadcast_event_base(), 142);
    }

    #[test]
    fn broadcast_block_mask_prevents_l2_latch() {
        use crate::device::events::broadcast::BroadcastDir;
        use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_STATUS};
        // Topology: source = shim (0,0); target = adjacent shim (1,0).
        //
        // (1,0) has L2 channel 4 enabled. (0,0) has East blocked, so the
        // BFS frontier never adds (1,0). Because the L2 sink fires only on
        // BFS-visited tiles, (1,0) must not latch.
        //
        // Shim row is the bottom-most row (row 0), so South/West from (0,0)
        // are off-grid. Only East and North are live from (0,0). Blocking
        // East cuts the only direct path to (1,0); with (0,0)'s North
        // propagating upward into column 0 compute tiles, the sideways
        // re-entry into column 1 at row 0 would require traversing column 1
        // downward, which still could reach (1,0). To keep the test clean
        // and topology-independent, also block North on (0,0) so the BFS
        // visits exactly one tile: (0,0) itself. This directly proves the
        // sink cannot fire on an un-visited tile.
        let mut dev = DeviceState::new_npu1();
        let (src_col, src_row) = (0u8, 0u8);
        let (tgt_col, tgt_row) = (1u8, 0u8);
        // Enable channel 4 on the adjacent shim's L2.
        dev.array
            .get_mut(tgt_col, tgt_row)
            .unwrap()
            .l2_irq
            .as_mut()
            .unwrap()
            .write_register(L2_REG_ENABLE, 1 << 4);
        // Block East and North on the source shim so BFS stays at (0,0).
        // `core_events` is the shim's event module (shim tiles have core
        // module event state but no mem module; see Tile::new).
        {
            let em = dev.array.get_mut(src_col, src_row).unwrap().core_events.as_mut().unwrap();
            em.broadcast.block_channel(4, BroadcastDir::East);
            em.broadcast.block_channel(4, BroadcastDir::North);
        }
        dev.array
            .get_mut(src_col, src_row)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(4));
        dev.propagate_broadcasts(src_col, src_row);
        let l2 = dev.array.get(tgt_col, tgt_row).unwrap().l2_irq.as_ref().unwrap();
        assert_eq!(
            l2.read_register(L2_REG_STATUS).unwrap(),
            0,
            "block mask must prevent L2 latch on un-visited neighbor shim"
        );
    }

    #[test]
    fn received_broadcast_drives_shim_l1_then_l2_within_one_dispatch() {
        use crate::device::interrupts::{
            L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, L2_REG_ENABLE, L2_REG_STATUS, SwitchId,
        };
        let mut dev = DeviceState::new_npu1();
        let (scol, srow) = (0u8, 2u8); // compute source
        let (shim_col, shim_row) = (0u8, 0u8);
        {
            let l1 = dev.array.get_mut(shim_col, shim_row).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 110 + 2); // BROADCAST ch2 -> shim PL event 112
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 6);
        }
        dev.array
            .get_mut(shim_col, shim_row)
            .unwrap()
            .l2_irq
            .as_mut()
            .unwrap()
            .write_register(L2_REG_ENABLE, 1 << 6);
        dev.array
            .get_mut(scol, srow)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(2));
        dev.propagate_broadcasts_fixpoint(scol, srow);
        let l2 = dev.array.get(shim_col, shim_row).unwrap().l2_irq.as_ref().unwrap();
        assert_ne!(
            l2.read_register(L2_REG_STATUS).unwrap() & (1 << 6),
            0,
            "error/broadcast -> L1 -> L2 must complete within one dispatch"
        );
    }

    #[test]
    fn fixpoint_propagation_terminates_under_self_feeding_config() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (0u8, 0u8);
        {
            let l1 = dev.array.get_mut(col, row).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 110 + 3); // input event 113
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 3); // output ch3 -> feeds itself
            let _ = SwitchId::B;
        }
        dev.array
            .get_mut(col, row)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(3));
        dev.propagate_broadcasts_fixpoint(col, row); // must return, not hang
    }

    #[test]
    fn l1_switch_independence_only_configured_switch_latches() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, SwitchId};
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (0u8, 0u8);
        {
            let l1 = dev.array.get_mut(col, row).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 9);
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
        }
        dev.fire_event_generate_for_test(col, row, 9);
        let l1 = dev.array.get(col, row).unwrap().l1_irq.as_ref().unwrap();
        assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0);
        assert_eq!(l1.read_status(SwitchId::B), 0, "switch B must not latch");
    }

    // -- Task 8: Tier B async-error hook end-to-end tests --

    #[test]
    fn event_generate_for_instr_error_populates_async_cache_and_ring() {
        use xdna_archspec::aie2::async_errors::{self, AieErrorOrigin, AmdxdnaErrorModule, AmdxdnaErrorNum};
        let mut dev = DeviceState::new_npu1();
        // Drive simulated time so ts_us is nonzero -- proves the cycle-as-ts
        // conversion is wired (not a literal 0 from absent cycle plumbing).
        dev.array.set_dma_cycle(50_000);

        // Fire INSTR_ERROR on a compute tile via the real production path.
        let (col, row) = (1u8, 2u8);
        dev.fire_event_generate_for_test(col, row, 69);

        // Cache populated with the right decode.
        let cache = dev.async_errors.last_cache().expect("cache must populate");
        let expected_err = async_errors::build_critical_aie_error_code(
            AmdxdnaErrorNum::AieInstruction,
            AmdxdnaErrorModule::AieCore,
        );
        assert_eq!(cache.err_code, expected_err, "err_code must decode INSTR_ERROR");
        assert_eq!(cache.ex_err_code, ((row as u64) << 8) | col as u64, "ex_err_code packs row|col");
        assert_eq!(cache.ts_us, 50, "ts_us = 50_000 cycles / 1000 = 50 us");

        // Ring at col 1 has the wire-format record.
        let ring = dev.async_errors.ring(col).expect("ring must exist");
        assert_eq!(ring.header().err_cnt, 1);
        let rec = &ring.records()[0];
        assert_eq!(rec.event_id, 69);
        assert_eq!(rec.row, row);
        assert_eq!(rec.col, col);
        assert_eq!(rec.mod_type, AieErrorOrigin::Core.wire_mod_type());
    }

    #[test]
    fn event_generate_for_non_error_event_does_not_record_async() {
        let mut dev = DeviceState::new_npu1();
        let (col, row) = (1u8, 2u8);
        // Event 7 is NOT in any error table (it's a generic user event).
        dev.fire_event_generate_for_test(col, row, 7);
        assert!(dev.async_errors.last_cache().is_none(), "non-error event must not populate the async cache");
        assert_eq!(dev.async_errors.ring(col).unwrap().header().err_cnt, 0);
    }

    #[test]
    fn tier_a_fires_independently_of_tier_b_for_non_error_shim_event() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        let mut dev = DeviceState::new_npu1();
        // Configure shim L1 to latch on event 7 (use 7 not 69 so this test
        // doesn't depend on INSTR_ERROR being in any shim event-slot mapping).
        // Then drive event 7 on a SHIM tile and verify L1 latched.
        let (col, row) = (0u8, 0u8);
        {
            let t = dev.array.get_mut(col, row).unwrap();
            let l1 = t.l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 7);
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 5);
        }
        dev.fire_event_generate_for_test(col, row, 7);

        // Tier A: L1 latched + IRQ_NO queued.
        let t = dev.array.get(col, row).unwrap();
        let l1 = t.l1_irq.as_ref().unwrap();
        assert_ne!(l1.read_status(SwitchId::A) & (1 << 16), 0, "Tier A L1 must latch");
        assert!(t.pending_broadcasts.iter().any(|pb| pb.channel == 5), "Tier A IRQ_NO must queue");

        // Tier B: shim event 7 is NOT in the SHIM_EVENT_CAT table, so no
        // async record. This proves Tier A fires independently of Tier B
        // (a Tier-A-only event leaves Tier B's cache empty).
        assert!(
            dev.async_errors.last_cache().is_none(),
            "non-error shim event must not populate async cache; Tier B is independent"
        );
    }

    // -- Task 5: hardware error -> EventModule -> broadcast -> L1 -> L2 --

    impl DeviceState {
        /// Inject a hardware error event into the compute tile's core EventModule
        /// (test-only). Mirrors the production `raise_instr_error` path exactly:
        ///
        ///   1. `em.generate_event(ev)` -- records the event in the EventModule
        ///      status bits and pending queue (the same entry production uses).
        ///   2. `tile.seed_broadcasts_for_event(ev)` -- the SAME shared helper
        ///      `raise_instr_error` calls, which seeds `pending_broadcasts` for
        ///      any channel configured to carry `ev`.
        ///
        /// Without step 2, `propagate_broadcasts_fixpoint` has nothing to start
        /// from (it drains `pending_broadcasts`, not the EventModule pending queue).
        #[cfg(test)]
        fn raise_hardware_error_for_test(&mut self, col: u8, row: u8, ev: u8) {
            // Mirrors raise_instr_error's event-subsystem steps. Update here if
            // raise_instr_error gains additional event-subsystem steps.
            let tile = self.array.get_mut(col, row).expect("tile must exist");
            tile.core_events
                .as_mut()
                .expect("compute tile must have core EventModule")
                .generate_event(ev);
            tile.seed_broadcasts_for_event(ev);
        }
    }

    /// End-to-end: a hardware error on a compute tile reaches shim L2 via the
    /// event->broadcast->L1->L2 interrupt chain.
    ///
    /// Chain: generate_event(INSTR_ERROR=69) on compute (0,2)
    ///   -> broadcast ch2 configured to event 69
    ///   -> propagate_broadcasts_fixpoint carries broadcast ch2 south to shim (0,0)
    ///   -> shim L1 slot 0 configured for PL event 112 (= SHIM_PL_BROADCAST_BASE 110 + ch2)
    ///      -> L1 latches, queues IRQ_NO 7
    ///   -> propagate_broadcasts_fixpoint delivers IRQ_NO 7 to L2
    ///   -> L2 channel 7 enabled -> STATUS bit 7 latches.
    #[test]
    fn hardware_error_reaches_shim_l2_end_to_end() {
        use crate::device::interrupts::{
            L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, L2_REG_ENABLE, L2_REG_STATUS, SwitchId,
        };
        use xdna_archspec::aie2::trace_events::core_events;
        let mut dev = DeviceState::new_npu1();
        let (ccol, crow) = (0u8, 2u8); // compute tile
        let (shim_col, shim_row) = (0u8, 0u8);
        // INSTR_ERROR = 69, which is < 128 (core EventModule num_events),
        // so generate_event will always record it (out-of-range ids are
        // silently ignored).
        let err_ev = core_events::INSTR_ERROR; // = 69
                                               // Configure compute tile's broadcast ch2 to fire on INSTR_ERROR (69).
        {
            let em = dev.array.get_mut(ccol, crow).unwrap().core_events.as_mut().unwrap();
            em.broadcast.configure_channel(2, err_ev);
        }
        // Configure shim L1: slot 0 watches PL event 112 (BROADCAST_BASE 110 + ch 2).
        {
            let l1 = dev.array.get_mut(shim_col, shim_row).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 110 + 2); // event 112
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 7);
        }
        // Enable L2 channel 7.
        dev.array
            .get_mut(shim_col, shim_row)
            .unwrap()
            .l2_irq
            .as_mut()
            .unwrap()
            .write_register(L2_REG_ENABLE, 1 << 7);
        // Inject the hardware error event.
        dev.raise_hardware_error_for_test(ccol, crow, err_ev);
        // Propagate: event fires broadcast ch2 on (0,2) -> travels south -> shim
        // L1 latches -> IRQ_NO 7 queued -> L2 channel 7 latches.
        dev.propagate_broadcasts_fixpoint(ccol, crow);
        let l2 = dev.array.get(shim_col, shim_row).unwrap().l2_irq.as_ref().unwrap();
        assert_ne!(
            l2.read_register(L2_REG_STATUS).unwrap() & (1 << 7),
            0,
            "hardware error must reach shim L2 via event->broadcast->L1->L2"
        );
    }
}

#[cfg(test)]
mod broadcast_wavefront_tests {
    use super::*;

    #[test]
    fn broadcast_origin_d_weighted_manhattan_unblocked() {
        // Fresh NPU1: no CDO broadcast block config -> fully connected 5x6 grid.
        let dev = DeviceState::new_npu1();
        let (src_col, src_row) = (0u8, 2u8); // a compute-row source
        let d_h = 2u32;
        let d_v = 3u32;
        let map = dev.broadcast_origin_d(src_col, src_row, 0, d_h, d_v);

        // Source is 0; every tile dc cols, dr rows away costs dc*d_h + dr*d_v
        // (weighted Manhattan, valid because the unblocked grid lets you move
        // monotonically toward any target with non-negative per-axis weights).
        for &(c, r, o) in &map {
            let dc = (c as i32 - src_col as i32).unsigned_abs();
            let dr = (r as i32 - src_row as i32).unsigned_abs();
            assert_eq!(o, dc * d_h + dr * d_v, "origin_D at ({c},{r})");
        }
        assert!(map.iter().any(|&(c, r, o)| c == src_col && r == src_row && o == 0));
    }

    #[test]
    fn broadcast_origin_d_reached_set_all_zero_at_zero_delays() {
        // At d=0 every origin_D is 0; connectivity (reached set) is unchanged --
        // source and the far corner are both reached. (Full reach-equivalence vs.
        // the legacy flood is additionally guarded by interrupt_path_tests.)
        let dev = DeviceState::new_npu1();
        let map = dev.broadcast_origin_d(0, 2, 0, 0, 0);
        assert!(map.iter().all(|&(_, _, o)| o == 0), "d=0 -> all origin_D == 0");
        assert!(map.iter().any(|&(c, r, _)| c == 0 && r == 2), "source reached");
        let (fc, fr) = (dev.array.cols() - 1, dev.array.rows() - 1);
        assert!(map.iter().any(|&(c, r, _)| c == fc && r == fr), "far corner reached");
    }

    #[test]
    fn origin_d_table_keys_modules_and_uses_delay_not_target() {
        // Single-source flood with nonzero synthetic constants so the table is
        // not all-zero. d_h=d_v=1, core_off=0, mem_off=2 (the -2 intra signature).
        let st = DeviceState::new_npu1();
        let rows = st.origin_d_table(0, 0, /*channel*/ 15, 1, 1, 0, 2);
        // The shim source (0,0) has origin_D 0 -> shim delay 0.
        assert!(rows.iter().any(|&(c, r, k, d)| c == 0 && r == 0 && k == "shim" && d == 0));
        // A compute tile at Manhattan distance N has core_delay = N*1 + 0,
        // mem_delay = N*1 + 2 (delay = origin_D + intra_off, NOT max_delay - delay).
        // Assert a known compute module's mem delay exceeds its core delay by 2.
        let core = rows
            .iter()
            .find(|&&(c, r, k, _)| c == 1 && r == 2 && k == "core")
            .map(|&(_, _, _, d)| d);
        let mem = rows
            .iter()
            .find(|&&(c, r, k, _)| c == 1 && r == 2 && k == "mem")
            .map(|&(_, _, _, d)| d);
        if let (Some(co), Some(me)) = (core, mem) {
            assert_eq!(me, co + 2);
        }
    }
}

#[cfg(test)]
mod broadcast_flood_timing_tests {
    use super::*;
    use crate::device::events::EventModuleType;

    #[test]
    fn flood_sets_constant_skew_baseline_under_explicit_delays() {
        // Drive the flood with explicit d_v and assert two modules hold a constant
        // offset == their delay difference from the first tick. The injection seam
        // is propagate_broadcasts_with_timing (Step 4), which tests call with
        // explicit values since the shipped consts are zero.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel; // 112
        let src = (0u8, 2u8); // compute-row source
        let hop = (0u8, 3u8); // one vertical hop north
                              // Configure both core timers to auto-reset on the broadcast event
                              // (Timer_Control offset 0x000, Reset_Event in bits [14:8]).
        for &(c, r) in &[src, hop] {
            dev.array
                .get_mut(c, r)
                .unwrap()
                .core_timer
                .write_register(0x000, (bcast_id as u32) << 8);
        }
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(channel));
        // d_v = 4: one vertical hop = 4 cy of skew. d_h, offsets = 0.
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 4, 0, 0);
        // One tick each consumes the latch -> value = target.
        dev.array.get_mut(src.0, src.1).unwrap().core_timer.tick();
        dev.array.get_mut(hop.0, hop.1).unwrap().core_timer.tick();
        let v_src = dev.array.get(src.0, src.1).unwrap().core_timer.value();
        let v_hop = dev.array.get(hop.0, hop.1).unwrap().core_timer.value();
        // Source reset earlier (delay 0) -> higher baseline by exactly one hop.
        // (v_src - v_hop = (max_delay - 0) - (max_delay - 4) = 4, independent of max_delay.)
        assert_eq!(v_src - v_hop, 4, "one-vertical-hop skew == d_v");
    }

    #[test]
    fn flood_is_behavior_neutral_at_zero_delays() {
        // Shipped (zero) consts: every reached timer resets to 0 on the first tick,
        // exactly as before this change, regardless of prior value.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel;
        let tiles = [(0u8, 2u8), (0u8, 3u8), (1u8, 2u8)];
        for (i, &(c, r)) in tiles.iter().enumerate() {
            dev.array
                .get_mut(c, r)
                .unwrap()
                .core_timer
                .write_register(0x000, (bcast_id as u32) << 8);
            for _ in 0..(i * 10 + 5) {
                dev.array.get_mut(c, r).unwrap().core_timer.tick(); // diverge the timers
            }
        }
        dev.array
            .get_mut(0, 2)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(channel));
        dev.propagate_broadcasts(0, 2); // shipped consts = all zero
        for &(c, r) in &tiles {
            dev.array.get_mut(c, r).unwrap().core_timer.tick();
            assert_eq!(dev.array.get(c, r).unwrap().core_timer.value(), 0, "zero consts -> reset to 0");
        }
    }
}

#[cfg(test)]
mod flood_source_capture_tests {
    use super::*;

    #[test]
    fn channel_15_flood_records_its_source() {
        // SP-4b: only channel 15 (timer-reset) floods are recorded, since
        // that's the channel origin_d_table/the sidecar export key on.
        let mut dev = DeviceState::new_npu1();
        let src = (0u8, 0u8);
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(15));
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 0, 0);
        assert_eq!(dev.flood_sources(), &std::collections::HashSet::from([src]));
    }

    #[test]
    fn non_timer_reset_channel_is_not_recorded() {
        // Ordinary-event broadcasts (e.g. channel 5) are not timer resets and
        // must not pollute the single-source guard the sidecar relies on.
        let mut dev = DeviceState::new_npu1();
        let src = (0u8, 2u8);
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(5));
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 0, 0);
        assert!(dev.flood_sources().is_empty(), "channel 5 must not be recorded as a flood source");
    }

    #[test]
    fn distinct_channel_15_sources_both_accumulate() {
        // Two different tiles each firing channel 15 must both land in the
        // set -- this is exactly the multi-source case the export's
        // single-source guard must detect and fail loud on.
        let mut dev = DeviceState::new_npu1();
        let (a, b) = ((0u8, 0u8), (1u8, 0u8));
        dev.array
            .get_mut(a.0, a.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(15));
        dev.propagate_broadcasts_with_timing(a.0, a.1, 0, 0, 0, 0);
        dev.array
            .get_mut(b.0, b.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(15));
        dev.propagate_broadcasts_with_timing(b.0, b.1, 0, 0, 0, 0);
        assert_eq!(dev.flood_sources(), &std::collections::HashSet::from([a, b]));
    }

    #[test]
    fn drain_pending_broadcasts_preserves_provenance() {
        let mut dev = DeviceState::new_npu1();
        let t = dev.array.get_mut(0, 0).unwrap();
        t.pending_broadcasts.push(PendingBroadcast::originated(15));
        t.pending_broadcasts.push(PendingBroadcast::relayed(7));
        let drained = t.drain_pending_broadcasts();
        assert_eq!(
            drained.iter().map(|pb| (pb.channel, pb.provenance)).collect::<Vec<_>>(),
            vec![(15, BroadcastProvenance::Originated), (7, BroadcastProvenance::Relayed),],
        );
        assert!(t.pending_broadcasts.is_empty(), "drain must empty the queue");
    }

    #[test]
    fn fixpoint_channel15_relay_does_not_record_a_second_flood_source() {
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_IRQ_NO_A, SwitchId};
        // A single genuine channel-15 (timer-reset) flood originates at shim
        // (0,0). The reached shim (1,0) has its L1 configured to latch the
        // channel-15 broadcast event (Pl base 110 + 15 = 125) and drive
        // IRQ_NO 15 -- so the fixpoint re-floods channel 15 *from (1,0)* as
        // L1-interrupt transport (a relay, not a timer reset). Pre-fix the
        // recorder inserts (1,0) as a spurious second source; post-fix the
        // relay is skipped and only the genuine origin (0,0) counts.
        //
        // This config self-feeds (the relay flood re-taps (1,0)'s own L1), so
        // propagate_broadcasts_fixpoint runs to its MAX_ITERS cap and logs a
        // warning -- expected under this pathological config. Assert only on
        // flood_sources(), never on log output or iteration count.
        let mut dev = DeviceState::new_npu1();
        {
            let l1 = dev.array.get_mut(1, 0).unwrap().l1_irq.as_mut().unwrap();
            l1.set_irq_event_slot(SwitchId::A, 0, 110 + 15); // event 125
            l1.write_register(L1_REG_ENABLE_A, 1 << 16);
            l1.write_register(L1_REG_IRQ_NO_A, 15);
        }
        dev.array
            .get_mut(0, 0)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(15));
        dev.propagate_broadcasts_fixpoint(0, 0);

        let sources = dev.flood_sources();
        assert_eq!(
            sources.len(),
            1,
            "a single genuine timer-reset origin must record exactly one flood source; got {sources:?}",
        );
        assert!(
            sources.contains(&(0, 0)),
            "the genuine origin (0,0) must be the single recorded source; got {sources:?}",
        );
    }
}

#[cfg(test)]
mod broadcast_timing_consts_tests {
    #[test]
    fn broadcast_timing_consts_default_to_zero() {
        use xdna_archspec::aie2::timing::{
            BROADCAST_INTRA_TILE_CORE_OFFSET, BROADCAST_INTRA_TILE_MEM_OFFSET, BROADCAST_PER_HOP_HORIZONTAL,
            BROADCAST_PER_HOP_VERTICAL,
        };
        // SP-1 ships behavior-neutral: real values arrive in SP-5 (silicon).
        assert_eq!(BROADCAST_PER_HOP_HORIZONTAL, 0);
        assert_eq!(BROADCAST_PER_HOP_VERTICAL, 0);
        assert_eq!(BROADCAST_INTRA_TILE_CORE_OFFSET, 0);
        assert_eq!(BROADCAST_INTRA_TILE_MEM_OFFSET, 0);
    }
}

#[cfg(test)]
mod broadcast_origin_offset_tests {
    use super::*;
    use crate::device::events::EventModuleType;

    #[test]
    fn flood_sets_trace_origin_offset_cross_tile() {
        // Two tiles one vertical hop apart take origin offsets differing by the
        // hop skew d_v, independent of max_delay. (offset = max_delay - origin_d;
        // source origin_d=0, hop origin_d=d_v.)
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let src = (0u8, 2u8);
        let hop = (0u8, 3u8); // one vertical hop north
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(channel));
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 4, 0, 0); // d_v=4
        let off_src = dev.array.get(src.0, src.1).unwrap().core_trace.origin_offset();
        let off_hop = dev.array.get(hop.0, hop.1).unwrap().core_trace.origin_offset();
        assert_eq!(off_src - off_hop, 4, "cross-tile origin offset diff == d_v");
    }

    #[test]
    fn flood_sets_trace_origin_offset_intra_tile_asymmetry() {
        // On one tile, core and mem trace units differ by the intra-tile
        // pipeline asymmetry: core_target - mem_target = mem_off - core_off.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let src = (0u8, 2u8);
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(channel));
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 2, 4); // core_off=2, mem_off=4
        let tile = dev.array.get(src.0, src.1).unwrap();
        let core = tile.core_trace.origin_offset();
        let mem = tile.mem_trace.origin_offset();
        assert_eq!(core - mem, 2, "core vs mem origin offset diff == mem_off - core_off");
    }

    #[test]
    fn flood_sets_origin_offset_before_arming_reached_trace() {
        // Configure the source core trace to START on the broadcast event id, so
        // the flood's own notify (effects.rs:602) arms it. If set_origin_offset
        // ran AFTER that notify, the Start would encode offset 0. A fresh device
        // has array.current_cycle == 0, so the arm cycle is 0 and the Start
        // absolute equals the offset.
        let mut dev = DeviceState::new_npu1();
        let channel = 5u8;
        let bcast_id = EventModuleType::Core.broadcast_event_base() + channel;
        let src = (0u8, 2u8);
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .core_trace
            .write_register(0x00, (bcast_id as u32) << 16); // start_event = bcast_id
        dev.array
            .get_mut(src.0, src.1)
            .unwrap()
            .pending_broadcasts
            .push(PendingBroadcast::originated(channel));
        dev.propagate_broadcasts_with_timing(src.0, src.1, 0, 0, 2, 4); // core_off=2 -> nonzero offset
        let tile = dev.array.get(src.0, src.1).unwrap();
        let off = tile.core_trace.origin_offset();
        assert!(off > 0, "source core trace must receive a nonzero offset");
        let bytes = tile.core_trace.encoded_bytes();
        assert_eq!(bytes[0] & 0xF0, 0xF0, "Start marker emitted by the flood's arming notify");
        let start_abs = (0..7).fold(0u64, |v, i| (v << 8) | bytes[1 + i] as u64);
        assert_eq!(start_abs, off, "Start absolute carries the offset (set before arm)");
    }
}
