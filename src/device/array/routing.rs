//! Stream switch routing, cascade routing, and inter-tile data movement.

use super::*;
use xdna_archspec::aie2::stream_switch::{compute, mem_tile, shim};
use xdna_archspec::aie2::SHIM_ROW;
use crate::device::stream_switch::PortType;
use std::fmt;

/// Lazily-formatted description of a DMA MM2S -> stream switch slave route.
///
/// The description is only used in a `log::info!` call, which short-circuits
/// when the log level is disabled. Using an enum with a `Display` impl avoids
/// allocating a formatted `String` on every routed word when logging is off.
enum Mm2sRouteDesc {
    ShimMux { channel: u8 },
    ShimFallbackSouthNorth,
    DmaPort { channel: u8, port_type: PortType },
}

impl fmt::Display for Mm2sRouteDesc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShimMux { channel } => write!(f, "Shim Mux MM2S ch{}", channel),
            Self::ShimFallbackSouthNorth => f.write_str("fallback South->North"),
            Self::DmaPort { channel, port_type } => {
                write!(f, "ch {}, type={:?}", channel, port_type)
            }
        }
    }
}

impl TileArray {
    /// Step all per-tile stream switches, applying module-gate checks.
    ///
    /// This handles intra-tile routing: forwarding data from input (slave) ports
    /// to output (master) ports within each tile based on configured local routes.
    /// This is necessary for pass-through routing where data enters a tile from
    /// one direction and exits to another.
    ///
    /// Tiles whose StreamSwitch module is clock-gated (column gate, MCC bit 0,
    /// or adaptive SS gate engaged) are skipped.  Wake-on-event paths reset
    /// the adaptive counter so a gated SS resumes on the cycle after a
    /// stream beat or SS register access lands.
    ///
    /// Returns the total number of words forwarded across all tiles.
    pub fn step_tile_switches(&mut self) -> usize {
        self.step_tile_switches_impl(true)
    }

    /// Second-pass variant for one routing cycle: accept crossing-delivered
    /// words without re-advancing each tile's switch pipeline (see
    /// `StreamSwitch::step_accept_only`). `route_streams` runs the switch twice
    /// per cycle; only the first pass advances, so the AM020 per-stage latency
    /// is decremented exactly once per cycle (#140 send-cadence fix).
    pub fn step_tile_switches_accept_only(&mut self) -> usize {
        self.step_tile_switches_impl(false)
    }

    fn step_tile_switches_impl(&mut self, advance: bool) -> usize {
        use crate::device::clock_control::ModuleKind;
        let mut total_forwarded = 0;
        // Collect (col, row, ss_active) for adaptive tick; done in
        // step_data_movement where DMA activity is also known.
        for i in 0..self.tiles.len() {
            let col = self.tiles[i].col;
            let row = self.tiles[i].row;

            // Column gate (top tier).
            if !self.clock.is_column_active(col) {
                continue;
            }
            // Module gate (mid tier): stream switch MCC bit 0.
            if !self.clock.is_module_active(col, row, ModuleKind::StreamSwitch) {
                continue;
            }
            // Adaptive gate (bottom tier): skip if the SS has been idle long
            // enough for the adaptive counter to engage.  Wake 2 (stream
            // beat -> cycle_active -> Phase 5 tick) clears the counter on
            // the cycle that brought the beat in, so this skip lasts at
            // most one cycle past the wake event.
            if self.clock.is_adaptive_ss_engaged(col, row) {
                continue;
            }

            let forwarded = if advance {
                self.tiles[i].stream_switch.step()
            } else {
                self.tiles[i].stream_switch.step_accept_only()
            };
            total_forwarded += forwarded;
            // Drain any fatal errors from this tile's stream switch
            self.fatal_errors.append(&mut self.tiles[i].stream_switch.fatal_errors);
        }
        total_forwarded
    }

    /// Step the complete data movement system.
    ///
    /// This follows the NPU hardware architecture with 4 clean phases:
    ///
    /// 1. **Lock Snapshot**: Freeze lock values for consistent visibility
    /// 2. **DMA Step**: All DMA engines produce/consume stream data
    /// 3. **Stream Routing**: Route data through stream switches and inter-tile wires
    /// 4. **Lock Commit**: Apply accumulated lock changes for next cycle
    ///
    /// The stream routing phase handles:
    /// - DMA MM2S -> StreamSwitch slave ports
    /// - Core stream output -> StreamSwitch slave ports
    /// - StreamSwitch local routing (slave -> master via configured routes)
    /// - Inter-tile propagation (physical wires: North master -> South slave, etc.)
    /// - StreamSwitch master ports -> DMA S2MM
    ///
    /// This matches real NPU behavior where each tile has a StreamSwitch and
    /// tiles connect via physical inter-tile wires. There is no global router.
    ///
    /// Returns (dma_active, streams_moved, words_routed)
    pub fn step_data_movement(&mut self, host_memory: &mut HostMemory) -> (bool, bool, usize) {
        // Phase 0: Port activity tracking reset.
        // Seed cycle_active flags from pre-existing FIFO state before routing.
        // Ports that receive data during routing will also be marked active.
        // The coordinator reads these flags after routing to generate
        // PORT_RUNNING trace events.
        for tile in &mut self.tiles {
            tile.stream_switch.begin_routing_cycle();
        }

        // Phase 1: DMA Lock Request Submission
        // Each DMA engine submits lock acquire/release requests to tile arbiters.
        // Core lock releases from the coordinator's Phase 2 are already pending.
        self.submit_all_dma_lock_requests(host_memory);

        // Phase 2: Lock Arbiter Resolution
        // Resolve all tile arbiters using round-robin. Applies granted requests
        // directly to lock values. DMA channels check results in Phase 3.
        // Skip gated columns -- no DMA submitted requests for them (Phase 1
        // already skipped them), so resolving is also a no-op.
        let cycle = self.current_cycle;
        for tile in &mut self.tiles {
            if !self.clock.is_column_active(tile.col) {
                continue;
            }
            tile.resolve_lock_requests(cycle);
        }

        // Phase 2.5: Reset S2MM ingress drain counters.
        // Phase 4's can_accept_stream_in_for_routing uses these to enforce the
        // registered-FIFO ordering invariant: a slot freed by Phase 3 drain is
        // NOT available to the producer in the same cycle's Phase 4. On HW the
        // TREADY signal is registered; the producer sees the full state from the
        // end of the prior cycle (#140 phase-ordering fix).
        for dma in &mut self.dma_engines {
            dma.reset_cycle_drain_counters();
        }

        // Phase 3: DMA Step
        // All DMA engines advance channel FSMs, checking arbiter results.
        // MM2S channels produce stream words, S2MM channels consume them.
        let dma_active = self.step_all_dma(host_memory);

        // Phase 4: Stream Routing
        // Route all stream data through the stream switch network.
        let words_routed = self.route_streams();

        // Phase 4.5: Cascade Propagation
        // Dedicated point-to-point 384-bit links between compute tiles.
        // Entirely separate from the stream switch fabric.
        self.route_cascade();

        // Phase 5: Adaptive counter tick (silicon-accurate, per tile/module).
        //
        // Each module's idle counter advances only when that module's clock is
        // on (column ungated AND module MCC bit set AND adaptive gate not already
        // engaged).  A gated module's counter stays frozen -- silicon does not
        // run the idle detector without a clock.
        //
        // "DMA active" here means any channel on the tile is in a non-terminal
        // state (Active / WaitingForLock / WaitingForStream / Paused), not the
        // narrower "made progress this cycle" definition used by the global
        // dma_active return.  Silicon's idle detector observes engine clock
        // activity, which includes channels stalled on backpressure or locks --
        // they are still consuming a clock and should not engage the gate.
        // Without this, a sink S2MM waiting on its upstream producer would
        // engage the adaptive gate before the producer ever sent a word,
        // permanently blocking the transfer.
        //
        // SS activity is determined by whether any port on the tile had
        // cycle_active set after routing (set by begin_routing_cycle +
        // route_streams on ports that moved data this cycle).
        {
            use crate::device::clock_control::ModuleKind;
            for col in 0..self.cols {
                // Column gate: skip gated columns entirely (counters frozen).
                if !self.clock.is_column_active(col) {
                    continue;
                }
                for row in 0..self.rows {
                    // DMA module counter: advance only if the DMA module is ungated.
                    // Use has_pending_work, not is_active: a channel sitting in
                    // FSM=Idle with a queued task counts as "not yet idle" --
                    // otherwise the gate engages before the channel can promote
                    // Idle->BdSetup and step_all_dma deadlocks the task queue.
                    if self.clock.is_module_active(col, row, ModuleKind::Dma) {
                        let idx = self.tile_index(col, row);
                        let tile_dma_active = self.dma_engines[idx].any_channel_has_pending_work();
                        self.clock.tick_adaptive_dma(col, row, tile_dma_active);
                    }

                    // SS module counter: advance only if the SS module is ungated.
                    //
                    // This also implements Wake 2 (stream beat into slave SS
                    // port) from cycle-accuracy-mission.md item #8.  A slave
                    // push sets the port's cycle_active flag; any cycle_active
                    // port here produces ss_active=true, which resets the
                    // counter via tick_adaptive_ss(active=true).  The wake is
                    // therefore an end-of-cycle reset rather than an emit-site
                    // reset -- closer to silicon's small wake-up latency than
                    // an instantaneous wake would be.
                    if self.clock.is_module_active(col, row, ModuleKind::StreamSwitch) {
                        let idx = self.tile_index(col, row);
                        let ss = &self.tiles[idx].stream_switch;
                        let ss_active = ss.masters.iter().any(|p| p.cycle_active)
                            || ss.slaves.iter().any(|p| p.cycle_active);
                        self.clock.tick_adaptive_ss(col, row, ss_active);
                    }
                }
            }
        }

        let switch_pipelines_active = self.tiles.iter().any(|t| t.stream_switch.has_pipeline_data());
        let streams_active =
            words_routed > 0 || !self.inter_tile_pipeline.is_empty() || switch_pipelines_active;
        (dma_active, streams_active, words_routed)
    }

    /// Route cascade data between adjacent compute tiles.
    ///
    /// Cascade is a dedicated 512-bit point-to-point link entirely separate
    /// from the stream switch fabric. Each compute tile can send cascade data
    /// to one neighbor (South or East) and receive from one neighbor (North or
    /// West), as configured by the accumulator control register at 0x36060.
    ///
    /// Uses a two-phase collect-then-apply approach to avoid borrow issues.
    pub(super) fn route_cascade(&mut self) {
        let rows = self.rows;
        let cols = self.cols;

        // Phase 1: Collect pending transfers.
        // Each entry: (src_idx, dst_col, dst_row)
        let mut transfers: Vec<(usize, u8, u8)> = Vec::new();

        for col in 0..cols {
            // Column gate check: skip gated columns for cascade propagation.
            if !self.clock.is_column_active(col) {
                continue;
            }

            for row in 0..rows {
                let idx = (col as usize) * (rows as usize) + (row as usize);
                let tile = &self.tiles[idx];

                if !tile.is_compute() || !tile.has_cascade_output() {
                    continue;
                }

                // Determine destination based on output direction
                let (dst_col, dst_row) = match tile.cascade_output_dir {
                    0 => {
                        // South: (col, row - 1)
                        // Shim row has no south neighbor.
                        if row == SHIM_ROW {
                            continue;
                        }
                        (col, row - 1)
                    }
                    1 => {
                        // East: (col + 1, row)
                        if col + 1 >= cols {
                            continue;
                        }
                        (col + 1, row)
                    }
                    _ => continue,
                };

                // Verify destination exists and is a compute tile
                let dst_idx = (dst_col as usize) * (rows as usize) + (dst_row as usize);
                if dst_idx >= self.tiles.len() || !self.tiles[dst_idx].is_compute() {
                    continue;
                }

                // Verify destination's input direction matches
                let dst_input_dir = self.tiles[dst_idx].cascade_input_dir;
                let expected_dir = match tile.cascade_output_dir {
                    0 => 0, // South output -> North input (dir=0)
                    1 => 1, // East output -> West input (dir=1)
                    _ => continue,
                };
                if dst_input_dir != expected_dir {
                    continue;
                }

                // Check backpressure: destination SCD FIFO must have room (depth 4).
                if self.tiles[dst_idx].cascade_input.len() < 4 {
                    transfers.push((idx, dst_col, dst_row));
                }
            }
        }

        // Phase 2: Apply transfers
        for (src_idx, dst_col, dst_row) in transfers {
            let dst_idx = (dst_col as usize) * (rows as usize) + (dst_row as usize);

            if let Some(data) = self.tiles[src_idx].pop_cascade_output() {
                log::info!(
                    "[CASCADE] Route ({},{}) -> ({},{}) data[0]={:#X}",
                    self.tiles[src_idx].col,
                    self.tiles[src_idx].row,
                    dst_col,
                    dst_row,
                    data[0]
                );
                self.tiles[dst_idx].push_cascade_input(data);
            }
        }
    }

    /// Route all stream data through the NPU stream network.
    ///
    /// This implements the per-tile StreamSwitch model from AM020/AM025:
    /// - Each tile has a StreamSwitch with slave (input) and master (output) ports
    /// - DMA MM2S outputs to slave ports, DMA S2MM receives from master ports
    /// - Configured routes forward data from slaves to masters within each tile
    /// - Inter-tile wires connect adjacent tiles (North master -> South slave, etc.)
    ///
    /// Data flows: DMA MM2S -> slave -> [route] -> master -> [wire] -> slave -> [route] -> master -> DMA S2MM
    ///
    /// For multi-hop paths (e.g., Shim -> MemTile -> Compute), data takes multiple
    /// cycles to traverse. Inter-tile links add ROUTE_LATENCY_PER_HOP cycles per
    /// hop, matching the AM020 stream switch pipeline specification.
    fn route_streams(&mut self) -> usize {
        let mut words_routed = 0;

        // Step 1: DMA MM2S -> StreamSwitch slave ports
        // Data from DMA output buffers enters the stream switch network
        words_routed += self.route_dma_to_tile_switches();

        // Step 2: Core stream -> StreamSwitch slave ports
        // Data from core stream writes enters via the Core slave port (port 0)
        words_routed += self.route_core_to_tile_switches();

        // Step 2b: Trace unit -> StreamSwitch trace slave ports
        // Binary trace packets enter on dedicated trace slave ports
        // (indices from AM025 gen_stream_ranges.rs)
        words_routed += self.route_trace_to_tile_switches();

        // Step 2c: Control packet read responses -> TileCtrl slave ports
        // Queued OP_READ response words drain into TileCtrl slave ports
        // as FIFO space permits (backpressure-aware, same as trace).
        words_routed += self.drain_ctrl_responses();

        // Step 3: StreamSwitch local routing (first pass)
        // Apply configured routes within each tile: slave -> master
        words_routed += self.step_tile_switches();

        // Step 4: Inter-tile propagation
        // Physical wires carry data between adjacent tiles:
        // - North masters -> South slaves of tile above
        // - South masters -> North slaves of tile below
        words_routed += self.propagate_inter_tile();

        // Step 5: StreamSwitch local routing (second pass)
        // Forward newly arrived (crossing-delivered) data through local routes.
        // ACCEPT-ONLY: must not re-advance the per-stage latency pipeline, which
        // the first pass (Step 3) already advanced this cycle -- advancing twice
        // halves the AM020 register-slice latency (#140 send-cadence fix).
        words_routed += self.step_tile_switches_accept_only();

        // Step 6: StreamSwitch master ports -> DMA S2MM
        // Data reaching DMA master ports enters DMA input buffers
        words_routed += self.route_tile_switches_to_dma();

        // Step 7: StreamSwitch Core master -> tile stream_input
        // Data reaching the Core master port goes to tile.stream_input for core reads
        words_routed += self.route_tile_switches_to_core();

        // Step 8: StreamSwitch TileCtrl master -> tile control packet handler
        // Data reaching the TileCtrl master port gets processed as register writes
        words_routed += self.route_tile_switches_to_ctrl();

        // Step 9: Mark stalled ports. After all routing for this cycle, any port
        // still holding data that did not beat this cycle was backpressured --
        // that is the HW PORT_STALLED condition. Evaluated here (not mid-route)
        // so it reflects the final cycle_beat state and stays exclusive with
        // PORT_RUNNING. Without this, circuit-routed DMA ports never emitted
        // PORT_STALLED even though HW asserts it (confirmed on NPU1).
        for tile in &mut self.tiles {
            tile.stream_switch.mark_stalled_ports();
        }

        words_routed
    }

    /// Route data from StreamSwitch Core master port to tile stream input.
    ///
    /// When data arrives at the Core master port (port 0) on a compute tile,
    /// it should be delivered to the tile's stream_input buffer for core reads.
    fn route_tile_switches_to_core(&mut self) -> usize {
        use crate::device::stream_switch::PortType;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Only compute tiles have core stream I/O
            if !self.tiles[i].is_compute() {
                continue;
            }

            // Check Core master port (port 0) for data
            if self.tiles[i].stream_switch.masters[0].fifo.is_empty() {
                continue;
            }

            // Verify it's a Core port
            if !matches!(self.tiles[i].stream_switch.masters[0].port_type, PortType::Core) {
                continue;
            }

            // Pop from Core master port and push to tile stream_input
            if let Some(data) = self.tiles[i].stream_switch.masters[0].pop() {
                // Push to tile's stream input buffer (port 0 for core)
                self.tiles[i].push_stream_input(0, data);
                words_routed += 1;
                log::debug!(
                    "TileSwitch->Core: tile ({},{}) master[0] -> stream_input = 0x{:08X}",
                    self.tiles[i].col,
                    self.tiles[i].row,
                    data
                );
            }
        }

        words_routed
    }

    /// Route data from StreamSwitch TileCtrl master port to tile control packet handler.
    ///
    /// When data arrives at the TileCtrl master port (port 3 on compute, port 6 on
    /// memtile, port 0 on shim), it is fed to the tile's control packet state machine
    /// which interprets it as register writes.
    fn route_tile_switches_to_ctrl(&mut self) -> usize {
        use crate::device::stream_switch::PortType;
        use crate::device::control_packets::ReassembleResult;
        use crate::device::tile::CtrlPacketAction;
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Find TileCtrl master port index
            let ctrl_master = self.tiles[i]
                .stream_switch
                .masters
                .iter()
                .position(|p| matches!(p.port_type, PortType::TileCtrl));

            let master_idx = match ctrl_master {
                Some(idx) => idx,
                None => continue,
            };

            // Configure the reassembler's drop_header from the master port config
            // (once only). If the master port is in packet mode with
            // Drop_Header=false, the stream header is forwarded and must be
            // consumed by the reassembler before the actual control packet header.
            if self.ctrl_reassemblers[i].drop_header() {
                let pkt_cfg = self.tiles[i].stream_switch.master_packet_cfg(master_idx);
                if pkt_cfg.map_or(false, |c| c.packet_enable && !c.drop_header) {
                    self.ctrl_reassemblers[i].set_drop_header(false);
                }
            }

            // Drain all pending words from the TileCtrl master port
            while self.tiles[i].stream_switch.masters[master_idx].has_data() {
                if let Some((data, tlast)) = self.tiles[i].stream_switch.masters[master_idx].pop_with_tlast()
                {
                    let col = self.tiles[i].col;
                    let row = self.tiles[i].row;

                    match self.ctrl_reassemblers[i].feed_word(data, tlast) {
                        ReassembleResult::Complete(packet) => {
                            // Convert the complete packet to CtrlPacketActions
                            // for backward compatibility with the coordinator.
                            let actions = Self::packet_to_actions(col, row, &packet);
                            if !actions.is_empty() {
                                self.pending_ctrl_actions.extend(actions);
                            }
                        }
                        ReassembleResult::Pending => {}
                        ReassembleResult::HandlerError(e) => {
                            // Poll-only sticky-continue (aie-rt/AM025): latch
                            // the Control_Packet_Handler_Status bit and keep
                            // processing. NOT engine-fatal -- pushing
                            // CtrlPacketAction::Error here would route to
                            // fatal_errors -> EngineStatus::Error (the bug the
                            // SLVERR plan corrects). Firmware polls the bit.
                            log::error!(
                                "Tile ({},{}) ctrl_pkt handler error: {:?} (sets Control_Packet_Handler_Status bit 0x{:X})",
                                col, row, e, e.bit()
                            );
                            Self::latch_pkt_error(&mut self.tiles[i], e);
                        }
                        ReassembleResult::Error(msg) => {
                            // Structural rejection: a packet so malformed it
                            // cannot be reassembled into a coherent
                            // ControlPacket at all (distinct from a
                            // HandlerError on a well-formed packet above).
                            // Logged AND pushed as a fatal
                            // CtrlPacketAction::Error -- an unparseable
                            // control packet is an emulator-modeling failure,
                            // not a poll-able HW error class, so it stays
                            // engine-fatal (deliberately out of scope of the
                            // SLVERR poll-only correction). Per spec 3.2 it
                            // does NOT set Second_Header_Parity -- that bit is
                            // exclusively a true-parity signal (Task 5). No
                            // pkt_handler_status write.
                            log::error!("{}", msg);
                            self.pending_ctrl_actions.push(CtrlPacketAction::Error(msg));
                        }
                    }

                    words_routed += 1;
                    log::debug!(
                        "TileSwitch->Ctrl: tile ({},{}) master[{}] -> ctrl_pkt = 0x{:08X}{}",
                        col,
                        row,
                        master_idx,
                        data,
                        if tlast { " TLAST" } else { "" }
                    );
                }
            }
        }

        words_routed
    }

    /// The single Control_Packet_Handler_Status latch point. All faithful
    /// handler errors converge here so the bit map lives in exactly one
    /// place (`PktHandlerError::bit()`).
    fn latch_pkt_error(
        tile: &mut crate::device::tile::Tile,
        e: crate::device::control_packets::status::PktHandlerError,
    ) {
        tile.pkt_handler_status |= e.bit();
    }

    /// Convert a reassembled ControlPacket into legacy CtrlPacketAction(s).
    ///
    /// This bridges the new control_packets module with the existing coordinator
    /// dispatch path. Once the coordinator is updated to consume ControlPackets
    /// directly, this adapter can be removed.
    fn packet_to_actions(
        col: u8,
        row: u8,
        packet: &crate::device::control_packets::ControlPacket,
    ) -> Vec<crate::device::tile::CtrlPacketAction> {
        use crate::device::control_packets::CtrlOpCode;
        use crate::device::tile::CtrlPacketAction;

        let mut actions = Vec::new();
        match packet.opcode {
            CtrlOpCode::Write | CtrlOpCode::BlockWrite | CtrlOpCode::WriteIncr => {
                for (i, &value) in packet.data.iter().enumerate() {
                    let addr = packet.address + (i as u32) * 4;
                    log::info!(
                        "Tile ({},{}) ctrl_pkt {:?}: [0x{:05X}] = 0x{:08X}",
                        col,
                        row,
                        packet.opcode,
                        addr,
                        value
                    );
                    actions.push(CtrlPacketAction::WriteRegister { col, row, offset: addr, value });
                }
            }
            CtrlOpCode::Read => {
                log::info!(
                    "Tile ({},{}) ctrl_pkt READ: addr=0x{:05X} beats={} resp_id={}",
                    col,
                    row,
                    packet.address,
                    packet.beats,
                    packet.response_id
                );
                actions.push(CtrlPacketAction::ReadRegisters {
                    col,
                    row,
                    offset: packet.address,
                    count: packet.beats,
                    response_id: packet.response_id,
                });
            }
        }
        actions
    }

    /// Route core stream output to per-tile StreamSwitch slave ports.
    ///
    /// When a core executes StreamWriteScalar or StreamWritePacketHeader,
    /// data is pushed to tile.stream_output. This routes that data to
    /// the Core slave port (port 0) on the tile's StreamSwitch.
    fn route_core_to_tile_switches(&mut self) -> usize {
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            // Core stream output goes to slave port 0 (Core port)
            // The stream switch then routes it based on configured local routes
            for port in 0..8u8 {
                while let Some(data) = self.tiles[i].pop_stream_output(port) {
                    // For core-initiated streams, use the Core slave port (port 0)
                    // The actual destination is determined by local routes
                    if port == 0 {
                        // Direct core output goes to Core slave port
                        if self.tiles[i].stream_switch.slaves[0].push(data) {
                            words_routed += 1;
                            log::debug!(
                                "Core->TileSwitch: tile({},{}) slave[0] <- 0x{:08X}",
                                self.tiles[i].col,
                                self.tiles[i].row,
                                data
                            );
                        }
                    } else {
                        // Other ports: push to corresponding slave if it exists
                        // This handles multi-port core stream writes
                        let slave_port = port as usize;
                        if slave_port < self.tiles[i].stream_switch.slaves.len() {
                            if self.tiles[i].stream_switch.slaves[slave_port].push(data) {
                                words_routed += 1;
                                log::debug!(
                                    "Core->TileSwitch: tile({},{}) slave[{}] <- 0x{:08X}",
                                    self.tiles[i].col,
                                    self.tiles[i].row,
                                    slave_port,
                                    data
                                );
                            }
                        }
                    }
                }
            }
        }

        words_routed
    }

    /// Route trace unit packets to per-tile stream switch trace slave ports.
    ///
    /// Each tile has trace units that produce 8-word (32-byte) binary trace
    /// packets. These packets enter the stream switch via dedicated trace
    /// slave ports (indices from AM025 via gen_stream_ranges.rs):
    ///   - Compute tile: TRACE_SLAVE_START (core trace), TRACE_SLAVE_END (memory trace)
    ///   - MemTile: TRACE_SLAVE_START (memory trace)
    ///   - Shim: TRACE_SLAVE_START (PL trace, sourced from `core_trace` --
    ///     shim has only one trace unit and it's configured via the
    ///     0x340D0+ "core module" register block)
    ///
    /// Once on the slave port, the existing packet routing infrastructure
    /// handles forwarding to shim DMA and ultimately to host DDR.
    fn route_trace_to_tile_switches(&mut self) -> usize {
        let mut words_routed = 0;

        for i in 0..self.tiles.len() {
            let tile_kind = self.tiles[i].tile_kind;

            match tile_kind {
                TileKind::Compute => {
                    let aie_trace = compute::TRACE_SLAVE_START as usize;
                    let mem_trace = compute::TRACE_SLAVE_END as usize;

                    // Core trace -> AIE_TRACE slave port
                    while self.tiles[i].core_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[aie_trace].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].core_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[aie_trace].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                    // Memory trace -> MEM_TRACE slave port
                    while self.tiles[i].mem_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[mem_trace].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].mem_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[mem_trace].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
                TileKind::Mem => {
                    let trace_port = mem_tile::TRACE_SLAVE_START as usize;

                    while self.tiles[i].mem_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[trace_port].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].mem_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[trace_port].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
                TileKind::ShimNoc | TileKind::ShimPl => {
                    // Shim has a single PL-module trace unit. Its config
                    // registers (0x340D0+) write to `core_trace`, and shim
                    // DMA / lock / port events are routed there too -- so
                    // pull pending words from `core_trace`, not `mem_trace`
                    // (which is unused on shim).
                    let trace_port = shim::TRACE_SLAVE_START as usize;

                    while self.tiles[i].core_trace.has_pending_words()
                        && self.tiles[i].stream_switch.slaves[trace_port].can_accept()
                    {
                        let (word, tlast) = self.tiles[i].core_trace.pop_word().unwrap();
                        self.tiles[i].stream_switch.slaves[trace_port].push_with_tlast(word, tlast);
                        words_routed += 1;
                    }
                }
            }
        }

        words_routed
    }

    /// Route DMA MM2S output to per-tile stream switch slave ports (as arbiter sources).
    ///
    /// In AIE stream switches, DMA MM2S output is presented as a "slave source" that
    /// the arbiter can route to external master ports.  Local routes like
    /// `slave[1] -> master[5]` configure master[5] to receive from DMA0 MM2S output.
    ///
    /// On real hardware, each MM2S channel has independent credit-based flow
    /// control to its own slave port FIFO.  We mirror this by iterating
    /// channels independently: one stalled channel cannot block another
    /// because each channel's `stream_out` is its own queue.
    ///
    /// DMA slave port ranges come from gen_stream_ranges.rs (AM025-derived).
    fn route_dma_to_tile_switches(&mut self) -> usize {
        use crate::device::stream_switch::PortType;
        use crate::device::dma::StreamData;
        use std::collections::VecDeque;
        let mut words_routed = 0;

        // Split-borrow fields once so the hot loop doesn't re-index self.* per
        // word (which produced a large bounds-check / IndexMut pile in profiles).
        let tiles = &mut self.tiles;
        let dma_engines = &mut self.dma_engines;
        let fatal_errors = &mut self.fatal_errors;

        for i in 0..tiles.len() {
            let tile = &mut tiles[i];
            let dma = &mut dma_engines[i];
            let tile_kind = tile.tile_kind;
            let col = tile.col;
            let row = tile.row;
            let s2mm_count = dma.s2mm_channel_count() as u8;
            let mm2s_count = dma.mm2s_channel_count() as u8;

            // Iterate MM2S channels independently.  Each has its own
            // stream_out queue; backpressure on one does not gate any other.
            for mm2s_ch in 0..mm2s_count {
                let combined_ch = s2mm_count + mm2s_ch;

                // Resolve the destination slave port once per channel -- it's
                // a function of the channel index and tile topology, not the
                // specific data word.
                let target_slave =
                    if tile_kind.is_shim() {
                        let from_mux = tile.shim_mux_mm2s_slaves.get(mm2s_ch as usize).copied().flatten();
                        if let Some(slave_idx) = from_mux {
                            Some((slave_idx, Mm2sRouteDesc::ShimMux { channel: mm2s_ch }))
                        } else {
                            // Fallback: find South slave with circuit route to North
                            let ss = &tile.stream_switch;
                            let slaves = ss.slaves.as_slice();
                            let masters = ss.masters.as_slice();
                            let mut fallback_slave = None;
                            for route in &ss.local_routes {
                                let s = route.slave_idx as usize;
                                let m = route.master_idx as usize;
                                if s < slaves.len()
                                    && m < masters.len()
                                    && matches!(slaves[s].port_type, PortType::South)
                                    && matches!(masters[m].port_type, PortType::North)
                                {
                                    fallback_slave = Some(s);
                                    break;
                                }
                            }
                            if let Some(slave_idx) = fallback_slave {
                                Some((slave_idx, Mm2sRouteDesc::ShimFallbackSouthNorth))
                            } else {
                                // Only log + drop once per channel that has data
                                // pending and no route.  Without data the channel
                                // never reaches this branch.
                                if dma.has_stream_out_for_channel(combined_ch) {
                                    let msg = format!(
                                        "DMA_MM2S->TileSwitch: Shim ({},{}) no route for MM2S ch{} -- \
                                     no slave port or fallback available",
                                        col, row, mm2s_ch,
                                    );
                                    log::error!("{}", msg);
                                    fatal_errors.push(msg);
                                    // Drain the unroutable queue
                                    while dma.pop_stream_out_for_channel(combined_ch).is_some() {}
                                }
                                continue;
                            }
                        }
                    } else {
                        let slave_port = match tile_kind {
                            TileKind::Mem => (mem_tile::DMA_SLAVE_START + mm2s_ch) as usize,
                            TileKind::Compute => (compute::DMA_SLAVE_START + mm2s_ch) as usize,
                            TileKind::ShimNoc | TileKind::ShimPl => unreachable!(),
                        };

                        let slaves = tile.stream_switch.slaves.as_slice();
                        if slave_port < slaves.len() {
                            let port_type = slaves[slave_port].port_type;
                            if matches!(port_type, PortType::Dma(_)) {
                                Some((slave_port, Mm2sRouteDesc::DmaPort { channel: combined_ch, port_type }))
                            } else {
                                // Drain misrouted queue -- caller misconfigured.
                                if dma.has_stream_out_for_channel(combined_ch) {
                                    log::debug!(
                                    "DMA_MM2S->TileSwitch: tile ({},{}) slave[{}] rejected - wrong type {:?}",
                                    col, row, slave_port, port_type
                                );
                                    while dma.pop_stream_out_for_channel(combined_ch).is_some() {}
                                }
                                continue;
                            }
                        } else {
                            while dma.pop_stream_out_for_channel(combined_ch).is_some() {}
                            continue;
                        }
                    };

                let Some((slave_idx, desc)) = target_slave else {
                    continue;
                };

                // Drain this channel's stream_out, attempting delivery for
                // each word.  Retain blocked words for next cycle, preserving
                // FIFO order on the channel's own queue.
                let mut retained: VecDeque<StreamData> = VecDeque::new();
                while let Some(data) = dma.pop_stream_out_for_channel(combined_ch) {
                    let slave = &mut tile.stream_switch.slaves[slave_idx];
                    if slave.can_accept() {
                        slave.push_with_tlast(data.data, data.tlast);
                        words_routed += 1;
                        let prefix = if tile_kind.is_shim() { "Shim" } else { "tile" };
                        log::info!(
                            "DMA_MM2S->TileSwitch: {} ({},{}) slave[{}] <- 0x{:08X}{} ({})",
                            prefix,
                            col,
                            row,
                            slave_idx,
                            data.data,
                            if data.tlast { " TLAST" } else { "" },
                            desc
                        );
                    } else {
                        // Slave FIFO full -- stop trying for this channel
                        // (further words for the same slave would also
                        // bounce, and we want to preserve order).
                        retained.push_back(data);
                        // Drain the remaining words for this channel into
                        // retained too so order is preserved end-to-end.
                        while let Some(d) = dma.pop_stream_out_for_channel(combined_ch) {
                            retained.push_back(d);
                        }
                        break;
                    }
                }

                if !retained.is_empty() {
                    dma.prepend_stream_out_for_channel(combined_ch, retained);
                }
            }
        }

        words_routed
    }

    /// Route per-tile stream switch DMA master output to DMA S2MM input.
    ///
    /// For tiles with local routes, DMA master port output goes to DMA S2MM stream_in.
    /// For Shim tiles, South master ports (2-7) represent the DDR interface and
    /// route to S2MM channels (South0->ch0, South1->ch1, etc.).
    fn route_tile_switches_to_dma(&mut self) -> usize {
        use crate::device::stream_switch::PortType;
        use crate::device::dma::StreamData;
        let mut words_routed = 0;

        let tiles = &mut self.tiles;
        let dma_engines = &mut self.dma_engines;
        let fatal_errors = &mut self.fatal_errors;

        for i in 0..tiles.len() {
            // Process ALL tiles - DMA master ports can receive data from any slave source
            // via local routes or direct connections. Removing the local_route_count guard
            // ensures compute tiles can receive stream data from MemTile.
            let tile = &mut tiles[i];
            let dma = &mut dma_engines[i];
            let col = tile.col;
            let row = tile.row;
            let is_shim = row == xdna_archspec::aie2::SHIM_ROW;

            // Check DMA master ports for outgoing data
            let num_masters = tile.stream_switch.masters.len();
            for master_port in 0..num_masters {
                // Determine the DMA channel for this master port
                // - For Compute/MemTile: only PortType::Dma ports route to DMA
                // - For Shim: South ports (2-7) represent DDR interface -> S2MM
                let dma_channel = match &tile.stream_switch.masters[master_port].port_type {
                    PortType::Dma(ch) => Some(*ch),
                    PortType::South if is_shim => {
                        // Shim South masters represent DDR interface.
                        // The Demux config (0x1F004) determines which South master feeds
                        // each DMA S2MM channel. Match the master port index against
                        // the configured S2MM mappings.
                        let mut matched_ch = None;
                        for (ch, mapped_master) in tile.shim_mux_s2mm_masters.iter().enumerate() {
                            if *mapped_master == Some(master_port) {
                                matched_ch = Some(ch as u8);
                                break;
                            }
                        }
                        matched_ch // None = port not muxed to DMA, no S2MM routing
                    }
                    _ => None,
                };

                if let Some(ch) = dma_channel {
                    let master = &mut tile.stream_switch.masters[master_port];
                    // Debug: check if master has data
                    let fifo_len = master.fifo.len();
                    if fifo_len > 0 {
                        let can_accept = dma.can_accept_stream_in_for_routing(ch);
                        log::debug!(
                            "TileSwitch->DMA check: tile ({},{}) master[{}] ch {} fifo_len={} can_accept={}",
                            col,
                            row,
                            master_port,
                            ch,
                            fifo_len,
                            can_accept
                        );
                    }

                    // Per-channel backpressure: only pop from master if the
                    // target channel's FIFO has space. Each S2MM channel has its
                    // own buffer, so one channel can't block another.
                    //
                    // `can_accept_stream_in_for_routing` enforces two invariants:
                    // (1) The registered-FIFO ordering fix (#140): a slot freed by
                    //     Phase 3 drain is not available to the producer in the same
                    //     cycle -- the effective capacity is checked against
                    //     start-of-cycle occupancy (current + drained_this_cycle).
                    // (2) The per-BD TREADY deassert (#140 bd_switch_accept_block):
                    //     when a BD boundary is crossed, TREADY deasserts for one
                    //     cycle so the recv port traces `on16 off1` not front-loaded.
                    //     When the refusal is that gap, elapse one cycle of it here.
                    // An unstarted/unconfigured S2MM channel asserts no TREADY:
                    // it neither accepts nor buffers stream data. The terminal
                    // memtile->shim (of_out) send must not pre-fill the fabric
                    // before the runtime sequence dispatches the shim S2MM drain
                    // -- on HW that drain gets no static CDO config and is Idle
                    // until then. No BD is in flight, so there is nothing to
                    // elapse (unlike the bd_switch_accept_block gap below).
                    // (#140 SP-4a terminal-send.)
                    if !dma.channel_is_started(ch) {
                        continue;
                    }
                    if !dma.can_accept_stream_in_for_routing(ch) {
                        dma.consume_bd_switch_accept_block(ch as usize);
                        continue;
                    }

                    if let Some((data, tlast)) = master.pop_with_tlast() {
                        // Push to DMA S2MM stream_in
                        let stream_data = StreamData { data, tlast, channel: ch };
                        let push_result = dma.push_stream_in(stream_data);
                        let new_len = dma.stream_in_len();
                        if push_result {
                            words_routed += 1;
                            log::info!("TileSwitch->DMA: tile ({},{}) master[{}] -> DMA ch {} = 0x{:08X} (stream_in_len={})",
                                col, row, master_port, ch, data, new_len);
                        } else {
                            let msg = format!(
                                "TileSwitch->DMA: push_stream_in FAILED for tile ({},{}) ch {} data=0x{:08X} -- \
                                 DMA input buffer overflow (impossible with hardware backpressure)",
                                col, row, ch, data,
                            );
                            log::error!("{}", msg);
                            fatal_errors.push(msg);
                        }
                    }
                }
            }
        }

        words_routed
    }

    /// Propagate data between adjacent tiles via directional ports.
    ///
    /// This is the critical inter-tile data movement phase. When a tile's North
    /// master port has data, it flows to the South slave port of the tile above.
    /// Similarly for South->North (downward) and East/West (horizontal) connections.
    ///
    /// Per AM025, the North/South port mappings are:
    /// - **Shim->MemTile**: Shim North masters 12-17 -> MemTile South slaves 7-12
    /// - **MemTile->Compute**: MemTile North masters 11-16 -> Compute South slaves 5-10
    /// - **Compute->MemTile**: Compute South masters 5-8 -> MemTile North slaves 13-16
    /// - **MemTile->Shim**: MemTile South masters 7-10 -> Shim North slaves 14-17
    /// - **Compute->Compute**: Compute South masters 5-8 -> Compute North slaves 15-18 (below)
    ///
    /// Note the Compute South-master destination depends on the tile below it:
    /// above a MemTile it targets mem_tile::NORTH_SLAVE_START (13); above another
    /// Compute it targets compute::NORTH_SLAVE_START (15). The `(tile_kind,
    /// below_type)` match below keys on this; the static mirror in
    /// `route_graph::inter_tile_dest` does the same.
    ///
    /// East/West port mappings (same-type adjacency only, MemTiles have no E/W):
    /// - **Compute East->West**: East masters 19-22 -> West slaves 11-14
    /// - **Compute West->East**: West masters 9-12 -> East slaves 19-22
    /// - **Shim East->West**: East masters 18-21 -> West slaves 10-13
    /// - **Shim West->East**: West masters 8-11 -> East slaves 18-21
    ///
    /// Returns the number of words transferred between tiles.
    fn propagate_inter_tile(&mut self) -> usize {
        use crate::device::tile::isolation as iso;
        use xdna_archspec::aie2::timing::INTER_TILE_HOP_LATENCY;
        // Phase 1: Advance the pipeline -- deliver words that have completed
        // their inter-tile traversal and decrement countdown timers.
        let words_transferred = self.advance_inter_tile_pipeline();

        // Phase 2: Accept new words from source masters into the pipeline.
        // Each transfer: (src_col, src_row, src_master_idx, dst_col, dst_row, dst_slave_idx, data, tlast)
        let mut transfers: Vec<(u8, u8, usize, u8, u8, usize, u32, bool)> = Vec::new();

        // Iterate through all tiles and check North/South master ports
        for col in 0..self.cols {
            for row in 0..self.rows {
                let idx = self.tile_index(col, row);
                let tile_kind = self.tiles[idx].tile_kind;

                // Check North masters - data flows to tile above (row + 1)
                if row + 1 < self.rows {
                    let above_idx = self.tile_index(col, row + 1);
                    let above_type = self.tiles[above_idx].tile_kind;

                    // Determine port mappings based on tile types (AM025-derived constants)
                    let (north_master_start, north_master_count, south_slave_start) =
                        match (tile_kind, above_type) {
                            (TileKind::ShimNoc | TileKind::ShimPl, TileKind::Mem) => (
                                shim::NORTH_MASTER_START as usize,
                                (shim::NORTH_MASTER_END - shim::NORTH_MASTER_START + 1) as usize,
                                mem_tile::SOUTH_SLAVE_START as usize,
                            ),
                            (TileKind::Mem, TileKind::Compute) => (
                                mem_tile::NORTH_MASTER_START as usize,
                                (mem_tile::NORTH_MASTER_END - mem_tile::NORTH_MASTER_START + 1) as usize,
                                compute::SOUTH_SLAVE_START as usize,
                            ),
                            (TileKind::Compute, TileKind::Compute) => (
                                compute::NORTH_MASTER_START as usize,
                                (compute::NORTH_MASTER_END - compute::NORTH_MASTER_START + 1) as usize,
                                compute::SOUTH_SLAVE_START as usize,
                            ),
                            _ => continue,
                        };

                    // Transfer from each North master to corresponding South slave
                    for i in 0..north_master_count {
                        let master_idx = north_master_start + i;
                        let slave_idx = south_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.front(), port.peek_tlast()) {
                                // Tile_Control isolation: a north-bound
                                // word is incoming-from-south at the
                                // destination, so it's gated by dst's
                                // SOUTH bit. HW absorbs the word at the
                                // boundary; we drop the transfer here.
                                if self.tiles[above_idx].isolation & iso::SOUTH != 0 {
                                    continue;
                                }
                                // Admit only if the crossing's total committed words (dest
                                // FIFO occupancy + in-flight delay-line words) is below the
                                // crossing depth. The crossing is a transport delay line
                                // (INTER_TILE_HOP_LATENCY, the single master->slave wire hop =
                                // the destination slave's 2-cycle input latency, NOT the full
                                // within-switch ROUTE_PER_HOP traversal) feeding a FIFO
                                // (fifo_capacity); its depth is their sum = 4 + 2 = 6,
                                // HW-pinned by the add_one memtile send headroom (one-BD
                                // ingress 8 + crossing 6 = 14). The crossing sustains a
                                // contiguous run far longer than the FIFO alone when the
                                // destination drains (recv PORT_RUNNING [16,16,16,16]), so the
                                // delay-line occupancy must NOT be bounded by the FIFO capacity
                                // alone -- that throttles a draining destination to the hop
                                // latency (the [4,4,..] regression). It backpressures only when
                                // the full crossing depth is committed.
                                if slave_idx < self.tiles[above_idx].stream_switch.slaves.len() && {
                                    let slave = &self.tiles[above_idx].stream_switch.slaves[slave_idx];
                                    slave.fifo.len() + self.inflight_to(above_idx, slave_idx)
                                        < slave.fifo_capacity + INTER_TILE_HOP_LATENCY as usize
                                } {
                                    transfers.push((
                                        col,
                                        row,
                                        master_idx,
                                        col,
                                        row + 1,
                                        slave_idx,
                                        data,
                                        tlast,
                                    ));
                                }
                            }
                        }
                    }
                }

                // Check South masters - data flows to tile below (row - 1)
                if row > 0 {
                    let below_idx = self.tile_index(col, row - 1);
                    let below_type = self.tiles[below_idx].tile_kind;

                    // Determine port mappings based on tile types (AM025-derived constants)
                    let (south_master_start, south_master_count, north_slave_start) =
                        match (tile_kind, below_type) {
                            (TileKind::Mem, TileKind::ShimNoc | TileKind::ShimPl) => (
                                mem_tile::SOUTH_MASTER_START as usize,
                                (mem_tile::SOUTH_MASTER_END - mem_tile::SOUTH_MASTER_START + 1) as usize,
                                shim::NORTH_SLAVE_START as usize,
                            ),
                            (TileKind::Compute, TileKind::Mem) => (
                                compute::SOUTH_MASTER_START as usize,
                                (compute::SOUTH_MASTER_END - compute::SOUTH_MASTER_START + 1) as usize,
                                mem_tile::NORTH_SLAVE_START as usize,
                            ),
                            (TileKind::Compute, TileKind::Compute) => (
                                compute::SOUTH_MASTER_START as usize,
                                (compute::SOUTH_MASTER_END - compute::SOUTH_MASTER_START + 1) as usize,
                                compute::NORTH_SLAVE_START as usize,
                            ),
                            _ => continue,
                        };

                    // Transfer from each South master to corresponding North slave
                    for i in 0..south_master_count {
                        let master_idx = south_master_start + i;
                        let slave_idx = north_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.front(), port.peek_tlast()) {
                                // South-bound is incoming-from-north at
                                // dst -- gated by dst's NORTH bit.
                                if self.tiles[below_idx].isolation & iso::NORTH != 0 {
                                    continue;
                                }
                                // Admit only if the crossing's total committed words (dest
                                // FIFO occupancy + in-flight delay-line words) is below the
                                // crossing depth (fifo_capacity + ROUTE_PER_HOP); see the
                                // North-site comment for the full delay-line+FIFO rationale.
                                if slave_idx < self.tiles[below_idx].stream_switch.slaves.len() && {
                                    let slave = &self.tiles[below_idx].stream_switch.slaves[slave_idx];
                                    slave.fifo.len() + self.inflight_to(below_idx, slave_idx)
                                        < slave.fifo_capacity + INTER_TILE_HOP_LATENCY as usize
                                } {
                                    transfers.push((
                                        col,
                                        row,
                                        master_idx,
                                        col,
                                        row - 1,
                                        slave_idx,
                                        data,
                                        tlast,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Check East masters - data flows to tile to the right (col + 1)
        for col in 0..self.cols {
            for row in 0..self.rows {
                let idx = self.tile_index(col, row);
                let tile_kind = self.tiles[idx].tile_kind;

                if col + 1 < self.cols {
                    let right_idx = self.tile_index(col + 1, row);
                    let right_type = self.tiles[right_idx].tile_kind;

                    // East masters on source -> West slaves on destination (AM025-derived)
                    let (east_master_start, east_count, west_slave_start) = match (tile_kind, right_type) {
                        (TileKind::Compute, TileKind::Compute) => (
                            compute::EAST_MASTER_START as usize,
                            (compute::EAST_MASTER_END - compute::EAST_MASTER_START + 1) as usize,
                            compute::WEST_SLAVE_START as usize,
                        ),
                        (TileKind::ShimNoc | TileKind::ShimPl, TileKind::ShimNoc | TileKind::ShimPl) => (
                            shim::EAST_MASTER_START as usize,
                            (shim::EAST_MASTER_END - shim::EAST_MASTER_START + 1) as usize,
                            shim::WEST_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    for i in 0..east_count {
                        let master_idx = east_master_start + i;
                        let slave_idx = west_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.front(), port.peek_tlast()) {
                                // East-bound is incoming-from-west at
                                // dst -- gated by dst's WEST bit.
                                if self.tiles[right_idx].isolation & iso::WEST != 0 {
                                    continue;
                                }
                                // Admit only if the crossing's total committed words (dest
                                // FIFO occupancy + in-flight delay-line words) is below the
                                // crossing depth (fifo_capacity + ROUTE_PER_HOP); see the
                                // North-site comment for the full delay-line+FIFO rationale.
                                if slave_idx < self.tiles[right_idx].stream_switch.slaves.len() && {
                                    let slave = &self.tiles[right_idx].stream_switch.slaves[slave_idx];
                                    slave.fifo.len() + self.inflight_to(right_idx, slave_idx)
                                        < slave.fifo_capacity + INTER_TILE_HOP_LATENCY as usize
                                } {
                                    transfers.push((
                                        col,
                                        row,
                                        master_idx,
                                        col + 1,
                                        row,
                                        slave_idx,
                                        data,
                                        tlast,
                                    ));
                                }
                            }
                        }
                    }
                }

                // West masters on source -> East slaves on destination (col - 1)
                if col > 0 {
                    let left_idx = self.tile_index(col - 1, row);
                    let left_type = self.tiles[left_idx].tile_kind;

                    // West masters on source -> East slaves on destination (AM025-derived)
                    let (west_master_start, west_count, east_slave_start) = match (tile_kind, left_type) {
                        (TileKind::Compute, TileKind::Compute) => (
                            compute::WEST_MASTER_START as usize,
                            (compute::WEST_MASTER_END - compute::WEST_MASTER_START + 1) as usize,
                            compute::EAST_SLAVE_START as usize,
                        ),
                        (TileKind::ShimNoc | TileKind::ShimPl, TileKind::ShimNoc | TileKind::ShimPl) => (
                            shim::WEST_MASTER_START as usize,
                            (shim::WEST_MASTER_END - shim::WEST_MASTER_START + 1) as usize,
                            shim::EAST_SLAVE_START as usize,
                        ),
                        _ => continue,
                    };

                    for i in 0..west_count {
                        let master_idx = west_master_start + i;
                        let slave_idx = east_slave_start + i;

                        if master_idx < self.tiles[idx].stream_switch.masters.len() {
                            let port = &self.tiles[idx].stream_switch.masters[master_idx];
                            if let (Some(&data), Some(tlast)) = (port.fifo.front(), port.peek_tlast()) {
                                // West-bound is incoming-from-east at
                                // dst -- gated by dst's EAST bit.
                                if self.tiles[left_idx].isolation & iso::EAST != 0 {
                                    continue;
                                }
                                // Admit only if the crossing's total committed words (dest
                                // FIFO occupancy + in-flight delay-line words) is below the
                                // crossing depth (fifo_capacity + ROUTE_PER_HOP); see the
                                // North-site comment for the full delay-line+FIFO rationale.
                                if slave_idx < self.tiles[left_idx].stream_switch.slaves.len() && {
                                    let slave = &self.tiles[left_idx].stream_switch.slaves[slave_idx];
                                    slave.fifo.len() + self.inflight_to(left_idx, slave_idx)
                                        < slave.fifo_capacity + INTER_TILE_HOP_LATENCY as usize
                                } {
                                    transfers.push((
                                        col,
                                        row,
                                        master_idx,
                                        col - 1,
                                        row,
                                        slave_idx,
                                        data,
                                        tlast,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        // Pop words from source masters into the inter-tile pipeline.
        // Words will be delivered after ROUTE_LATENCY_PER_HOP cycles.
        use xdna_archspec::aie2::timing::INTER_TILE_HOP_LATENCY as ROUTE_LATENCY_PER_HOP;
        use crate::device::stream_switch::{PortRef, PortDir};

        for (src_col, src_row, src_master, dst_col, dst_row, dst_slave, _data, _tlast) in transfers {
            let src_idx = self.tile_index(src_col, src_row);
            let dst_idx = self.tile_index(dst_col, dst_row);

            // Pop from source master (with TLAST)
            if let Some((data, tlast)) =
                self.tiles[src_idx].stream_switch.masters[src_master].pop_with_tlast()
            {
                // Record enacted hop when recorder is enabled.
                // This branch is compiled away (zero cost) when hop_recorder is None.
                if let Some(rec) = &mut self.hop_recorder {
                    let src_kind = self.tiles[src_idx].stream_switch.masters[src_master]
                        .port_type
                        .as_kind_str()
                        .to_owned();
                    let dst_kind = self.tiles[dst_idx].stream_switch.slaves[dst_slave]
                        .port_type
                        .as_kind_str()
                        .to_owned();
                    rec.push((
                        PortRef {
                            col: src_col,
                            row: src_row,
                            port: src_master as u8,
                            dir: PortDir::Master,
                            kind: src_kind,
                        },
                        PortRef {
                            col: dst_col,
                            row: dst_row,
                            port: dst_slave as u8,
                            dir: PortDir::Slave,
                            kind: dst_kind,
                        },
                    ));
                }

                self.inter_tile_pipeline.push(InFlightWord {
                    dst_tile_idx: dst_idx,
                    dst_slave_idx: dst_slave,
                    data,
                    tlast,
                    cycles_remaining: ROUTE_LATENCY_PER_HOP,
                });
                log::debug!(
                    "InterTile: ({},{}) master[{}] -> pipeline({}) -> ({},{}) slave[{}] = 0x{:08X}{} delay={}cy",
                    src_col, src_row, src_master, ROUTE_LATENCY_PER_HOP,
                    dst_col, dst_row, dst_slave, data,
                    if tlast { " TLAST" } else { "" },
                    ROUTE_LATENCY_PER_HOP,
                );
            }
        }

        words_transferred
    }

    /// Advance the inter-tile pipeline by one cycle.
    ///
    /// Called at the start of `propagate_inter_tile()`. Decrements countdown
    /// timers and delivers words that have completed their traversal.
    /// Returns the number of words delivered.
    fn advance_inter_tile_pipeline(&mut self) -> usize {
        let mut delivered = 0;
        let mut i = 0;

        while i < self.inter_tile_pipeline.len() {
            let word = &mut self.inter_tile_pipeline[i];
            if word.cycles_remaining > 0 {
                word.cycles_remaining -= 1;
            }

            if word.cycles_remaining == 0 {
                // Try to deliver to destination slave
                let dst_idx = word.dst_tile_idx;
                let dst_slave = word.dst_slave_idx;

                if dst_slave < self.tiles[dst_idx].stream_switch.slaves.len()
                    && self.tiles[dst_idx].stream_switch.slaves[dst_slave].can_accept()
                {
                    let data = word.data;
                    let tlast = word.tlast;
                    self.tiles[dst_idx].stream_switch.slaves[dst_slave].push_with_tlast(data, tlast);
                    self.inter_tile_pipeline.remove(i);
                    delivered += 1;
                    // Don't increment i -- remove shifted later elements down
                } else {
                    // Destination can't accept -- word stays in pipeline (backpressure).
                    // It will be retried next cycle with cycles_remaining still at 0.
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        delivered
    }

    /// Count words already in the inter-tile pipeline that target a given
    /// destination slave. Inter-tile admission bounds `fifo.len() + inflight_to`
    /// against the crossing depth (`fifo_capacity + ROUTE_PER_HOP`), so the
    /// transport delay line is committed but the crossing still sustains
    /// contiguous flow when the destination drains (it backpressures only when
    /// the full crossing depth -- FIFO skid plus delay line -- is committed).
    pub(crate) fn inflight_to(&self, dst_idx: usize, dst_slave: usize) -> usize {
        self.inter_tile_pipeline
            .iter()
            .filter(|w| w.dst_tile_idx == dst_idx && w.dst_slave_idx == dst_slave)
            .count()
    }

    /// Total in-flight crossing words destined for a tile (any slave port),
    /// with the count still in transit (`cycles_remaining > 0`) vs landed-this-
    /// cycle. Inspection-only for the per-stage cascade probe
    /// (`XDNA_EMU_STAGE_PROBE`): shows how much the inter-tile delay line holds
    /// between a stalled consumer and its upstream producer.
    pub fn inflight_to_tile(&self, col: u8, row: u8) -> (usize, usize) {
        let dst = self.tile_index(col, row);
        let total = self.inter_tile_pipeline.iter().filter(|w| w.dst_tile_idx == dst).count();
        let in_transit = self
            .inter_tile_pipeline
            .iter()
            .filter(|w| w.dst_tile_idx == dst && w.cycles_remaining > 0)
            .count();
        (total, in_transit)
    }
}

#[cfg(test)]
mod isolation_tests {
    use super::*;
    use crate::device::tile::isolation as iso;
    use xdna_archspec::aie2::stream_switch::compute as compute_ranges;

    /// Helper: push one word with TLAST to the source tile's east-master
    /// port at index `compute::EAST_MASTER_START` and return the master/
    /// slave indices the test expects to use.
    fn seed_east_to_west_word(array: &mut TileArray, src_col: u8, src_row: u8) -> (usize, usize) {
        let master_idx = compute_ranges::EAST_MASTER_START as usize;
        let slave_idx = compute_ranges::WEST_SLAVE_START as usize;
        let src_idx = array.tile_index(src_col, src_row);
        let pushed = array.tiles[src_idx].stream_switch.masters[master_idx].push_with_tlast(0xDEADBEEF, true);
        assert!(pushed, "test setup: src east-master must accept the seed word");
        (master_idx, slave_idx)
    }

    /// Without isolation, propagate_inter_tile drains the source's east
    /// master into the inter-tile pipeline. The destination's west slave
    /// remains empty (delivery is delayed by ROUTE_PER_HOP cycles), but
    /// the source FIFO is now empty -- proving the transfer was accepted.
    #[test]
    fn east_to_west_transfer_passes_when_not_isolated() {
        let mut array = TileArray::npu1();
        let (src_col, src_row) = (1u8, 2u8); // compute tile
        let (master_idx, _slave_idx) = seed_east_to_west_word(&mut array, src_col, src_row);

        array.propagate_inter_tile();

        let src_idx = array.tile_index(src_col, src_row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "source east master must drain into pipeline when no isolation"
        );
    }

    /// With dst.isolation = WEST, the same transfer must be dropped at
    /// the boundary -- the source's master FIFO retains the word.
    #[test]
    fn east_to_west_transfer_blocked_when_dst_isolates_west() {
        let mut array = TileArray::npu1();
        let (src_col, src_row) = (1u8, 2u8);
        let dst_idx = array.tile_index(src_col + 1, src_row);
        array.tiles[dst_idx].isolation = iso::WEST;

        let (master_idx, _slave_idx) = seed_east_to_west_word(&mut array, src_col, src_row);
        array.propagate_inter_tile();

        let src_idx = array.tile_index(src_col, src_row);
        assert_eq!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.len(),
            1,
            "source east master must retain word when dst is isolated from west"
        );
    }

    /// Isolation bits are independent. ISOLATE_FROM_NORTH on dst doesn't
    /// affect an east-to-west transfer (which is gated by WEST). The
    /// source FIFO drains as in the unisolated case.
    #[test]
    fn unrelated_isolation_bit_does_not_block_transfer() {
        let mut array = TileArray::npu1();
        let (src_col, src_row) = (1u8, 2u8);
        let dst_idx = array.tile_index(src_col + 1, src_row);
        array.tiles[dst_idx].isolation = iso::NORTH | iso::SOUTH | iso::EAST;

        let (master_idx, _slave_idx) = seed_east_to_west_word(&mut array, src_col, src_row);
        array.propagate_inter_tile();

        let src_idx = array.tile_index(src_col, src_row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "WEST is the only bit that gates east-to-west; others must not block"
        );
    }

    /// Helper for cardinal directions other than east. Pushes one word with
    /// TLAST onto the chosen master at the source tile and returns the master
    /// index used. The destination's slave index isn't returned here -- the
    /// caller only inspects the source FIFO drain state.
    fn seed_master_word(array: &mut TileArray, col: u8, row: u8, master_idx: usize) -> usize {
        let src_idx = array.tile_index(col, row);
        let pushed = array.tiles[src_idx].stream_switch.masters[master_idx].push_with_tlast(0xCAFEBABE, true);
        assert!(pushed, "test setup: master[{master_idx}] must accept the seed word");
        master_idx
    }

    /// North-bound (data flows row -> row+1) is incoming-from-south at the
    /// destination, so dst.isolation = SOUTH must drop it. Without isolation
    /// the source drains; with SOUTH set the source retains the word.
    #[test]
    fn north_bound_transfer_blocked_when_dst_isolates_south() {
        let mut array = TileArray::npu1();
        // Compute->Compute boundary: source at (col, row=2), dst at row=3.
        let (col, src_row) = (1u8, 2u8);
        let dst_idx = array.tile_index(col, src_row + 1);
        let master_idx = compute_ranges::NORTH_MASTER_START as usize;

        // Without isolation: drains.
        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        let src_idx = array.tile_index(col, src_row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "baseline: north-bound must drain when dst is unisolated"
        );

        // With dst.SOUTH set: blocked.
        array.tiles[dst_idx].isolation = iso::SOUTH;
        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        assert_eq!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.len(),
            1,
            "north-bound must be blocked by dst.SOUTH"
        );
    }

    /// South-bound (data flows row -> row-1) is incoming-from-north at the
    /// destination, so dst.isolation = NORTH must drop it.
    #[test]
    fn south_bound_transfer_blocked_when_dst_isolates_north() {
        let mut array = TileArray::npu1();
        // Compute->Compute boundary: source at (col, row=3), dst at row=2.
        let (col, src_row) = (1u8, 3u8);
        let dst_idx = array.tile_index(col, src_row - 1);
        let master_idx = compute_ranges::SOUTH_MASTER_START as usize;

        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        let src_idx = array.tile_index(col, src_row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "baseline: south-bound must drain when dst is unisolated"
        );

        array.tiles[dst_idx].isolation = iso::NORTH;
        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        assert_eq!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.len(),
            1,
            "south-bound must be blocked by dst.NORTH"
        );
    }

    /// Memtile -> shim south-bound transit: source at row=1 (memtile),
    /// dst at row=0 (shim). Inbound direction is north at the dst, so
    /// shim.NORTH gates it. This is the only routing path where the
    /// shim's own isolation byte actually matters.
    #[test]
    fn memtile_to_shim_south_bound_blocked_when_shim_isolates_north() {
        use xdna_archspec::aie2::stream_switch::mem_tile as mem_tile_ranges;
        let mut array = TileArray::npu1();
        let col = 1u8;
        let src_row = 1u8; // memtile
        let dst_idx = array.tile_index(col, src_row - 1); // shim
        let master_idx = mem_tile_ranges::SOUTH_MASTER_START as usize;

        // Baseline: drains when shim is unisolated.
        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        let src_idx = array.tile_index(col, src_row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "baseline: memtile->shim south-bound must drain when shim is unisolated"
        );

        // With shim.NORTH set: blocked.
        array.tiles[dst_idx].isolation = iso::NORTH;
        seed_master_word(&mut array, col, src_row, master_idx);
        array.propagate_inter_tile();
        assert_eq!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.len(),
            1,
            "memtile->shim south-bound must be blocked by shim.NORTH"
        );
    }

    /// West-bound (data flows col -> col-1) is incoming-from-east at the
    /// destination, so dst.isolation = EAST must drop it.
    #[test]
    fn west_bound_transfer_blocked_when_dst_isolates_east() {
        let mut array = TileArray::npu1();
        // Compute->Compute boundary: source at (col=2, row), dst at col=1.
        let (src_col, row) = (2u8, 2u8);
        let dst_idx = array.tile_index(src_col - 1, row);
        let master_idx = compute_ranges::WEST_MASTER_START as usize;

        seed_master_word(&mut array, src_col, row, master_idx);
        array.propagate_inter_tile();
        let src_idx = array.tile_index(src_col, row);
        assert!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.is_empty(),
            "baseline: west-bound must drain when dst is unisolated"
        );

        array.tiles[dst_idx].isolation = iso::EAST;
        seed_master_word(&mut array, src_col, row, master_idx);
        array.propagate_inter_tile();
        assert_eq!(
            array.tiles[src_idx].stream_switch.masters[master_idx].fifo.len(),
            1,
            "west-bound must be blocked by dst.EAST"
        );
    }
}
