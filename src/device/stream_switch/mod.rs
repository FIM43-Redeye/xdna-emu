//! Stream switch model for DMA and inter-tile data movement.
//!
//! This module models the AIE2 stream switch, supporting both circuit-switched
//! and packet-switched routing modes with FIFO-based backpressure.
//!
//! # Architecture
//!
//! Each tile has a stream switch with:
//! - Master ports (output): send data to other tiles
//! - Slave ports (input): receive data from other tiles
//! - DMA integration: DMA channels connect to specific ports
//! - FIFOs for buffering
//! - Arbiter/msel packet routing with slot-based header matching
//!
//! ```text
//!                    North
//!                      ^
//!                +-----+-----+
//!                |           |
//!       West <---+  Stream   +---> East
//!                |  Switch   |
//!                |           |
//!                +-----+-----+
//!                      |
//!                      v
//!              South / Core / DMA
//! ```
//!
//! # What is modeled
//!
//! - Circuit-switched routing (slave -> master via slave index)
//! - Packet-switched routing (header match, arbiter locking, msel selection)
//! - Per-port FIFO buffering with backpressure
//! - Intra-tile pipeline latency (3-4 cycles per AM020)
//! - Drop-header mode for packet masters
//! - Port activity tracking for trace events (PORT_RUNNING, PORT_STALLED, PORT_TLAST)
//!
//! # Not yet modeled
//!
//! - Deterministic merge (registers absorbed, no behavioral effect)
//! - Per-tile-type port validity enforcement (CDO always programs valid routes)
//! - Micro-timing: NoC latency, memory bank conflicts

mod ports;
mod packet_types;
mod packet_switch;
mod route_graph;

pub use ports::*;
pub use packet_types::*;
pub use packet_switch::*;
pub use route_graph::*;

#[cfg(test)]
mod tests;

use xdna_archspec::aie2::timing as arch_timing;

/// Stream switch for a single tile.
#[derive(Debug, Clone)]
pub struct StreamSwitch {
    /// Tile column
    pub col: u8,
    /// Tile row
    pub row: u8,
    /// Master ports
    pub masters: Vec<StreamPort>,
    /// Slave ports
    pub slaves: Vec<StreamPort>,
    /// Local routes (slave -> master within this tile, circuit mode)
    pub local_routes: Vec<LocalRoute>,
    /// Latency in cycles for local-to-local routing
    pub local_latency: u8,
    /// Latency in cycles for external routing
    pub external_latency: u8,

    // === Packet routing state ===
    /// Per-slave packet slot config (4 slots per port), indexed by slave port.
    slave_slots: Vec<[PacketSlot; 4]>,
    /// Per-master packet config, indexed by master port.
    master_packet_config: Vec<MasterPacketConfig>,
    /// Per-slave active packet tracking for mid-packet forwarding.
    /// `Some(ActivePacket)` while forwarding data words between header and TLAST.
    active_packets: Vec<Option<ActivePacket>>,
    /// Per-arbiter lock: which slave port currently holds each arbiter.
    /// `None` = arbiter is free, `Some(slave_idx)` = locked by that slave
    /// until TLAST. Prevents packet interleaving when multiple slaves route
    /// through the same arbiter simultaneously. 8 arbiters max (3-bit field).
    pub(crate) arbiter_locks: [Option<usize>; 8],
    /// Per-arbiter round-robin priority pointer. After a packet completes
    /// (TLAST), the pointer advances past the releasing slave so the next
    /// contending slave gets priority. This matches the hardware's round-robin
    /// arbitration: after TLAST, the releasing slave goes to the back of the
    /// queue. Values are slave indices; the arbiter grants to the first
    /// eligible slave starting from this index (wrapping).
    arbiter_priority: [usize; 8],

    /// Words traversing the intra-tile switch pipeline.
    ///
    /// Models the AM020 switch pipeline latency: data takes 3-4 cycles to
    /// traverse from slave input to master output within a single tile's
    /// stream switch. Without this, data moves slave->master in 1 cycle.
    pub(crate) switch_pipeline: Vec<InSwitchWord>,

    /// Fatal errors accumulated during routing.
    ///
    /// Conditions that are impossible on real hardware (e.g., packet with no
    /// configured route) are collected here. The owning TileArray drains
    /// these after each step and propagates them to the coordinator.
    pub fatal_errors: Vec<String>,
}

impl StreamSwitch {
    /// Build ports from a spec layout.
    ///
    /// This takes a slice of encoded port types from the arch module and converts
    /// them into StreamPort instances with the correct indices and types.
    fn build_ports_from_spec(layout: &[u8], direction: PortDirection) -> Vec<StreamPort> {
        layout
            .iter()
            .enumerate()
            .map(|(idx, &encoded)| StreamPort::new(idx as u8, direction, PortType::from_spec(encoded)))
            .collect()
    }

    /// Create a new stream switch for a compute tile.
    ///
    /// Port layout is defined in arch::{COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS}.
    /// Tile_Ctrl port type is data-driven from AM025 register database.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::Compute);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);

        let num_slaves = slaves.len();
        let num_masters = masters.len();
        Self {
            col,
            row,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: arch_timing::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
            slave_slots: vec![[PacketSlot::default(); 4]; num_slaves],
            master_packet_config: vec![MasterPacketConfig::default(); num_masters],
            active_packets: vec![None; num_slaves],
            arbiter_locks: [None; 8],
            arbiter_priority: [0; 8],
            switch_pipeline: Vec::new(),
            fatal_errors: Vec::new(),
        }
    }

    /// Create a new stream switch for a memory tile.
    ///
    /// Port layout is defined in arch::{MEMTILE_MASTER_PORTS, MEMTILE_SLAVE_PORTS}.
    /// Tile_Ctrl port type is data-driven from AM025 register database.
    ///
    /// Note the asymmetry: 6 North masters but only 4 North slaves, and
    /// 4 South masters but 6 South slaves. This matches MemTile's role as
    /// a buffer between Shim (which has 6 North outputs) and Compute tiles.
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::Mem);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);

        let num_slaves = slaves.len();
        let num_masters = masters.len();
        Self {
            col,
            row,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: arch_timing::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
            slave_slots: vec![[PacketSlot::default(); 4]; num_slaves],
            master_packet_config: vec![MasterPacketConfig::default(); num_masters],
            active_packets: vec![None; num_slaves],
            arbiter_locks: [None; 8],
            arbiter_priority: [0; 8],
            switch_pipeline: Vec::new(),
            fatal_errors: Vec::new(),
        }
    }

    /// Create a new stream switch for a shim tile.
    ///
    /// Port layout is defined in arch::{SHIM_MASTER_PORTS, SHIM_SLAVE_PORTS}.
    /// Tile_Ctrl port type is data-driven from AM025 register database.
    ///
    /// The 6 North masters (12-17) connect 1:1 to MemTile South slaves (7-12).
    pub fn new_shim_tile(col: u8) -> Self {
        let ports = crate::device::arch_handle::stream_switch_topology()
            .for_tile(xdna_archspec::types::TileKind::ShimNoc);
        let masters = Self::build_ports_from_spec(ports.master_ports, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(ports.slave_ports, PortDirection::Slave);

        let num_slaves = slaves.len();
        let num_masters = masters.len();
        Self {
            col,
            row: 0,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: arch_timing::STREAM_EXTERNAL_TO_EXTERNAL_LATENCY,
            slave_slots: vec![[PacketSlot::default(); 4]; num_slaves],
            master_packet_config: vec![MasterPacketConfig::default(); num_masters],
            active_packets: vec![None; num_slaves],
            arbiter_locks: [None; 8],
            arbiter_priority: [0; 8],
            switch_pipeline: Vec::new(),
            fatal_errors: Vec::new(),
        }
    }

    /// Get a master port by index.
    pub fn master(&self, index: usize) -> Option<&StreamPort> {
        self.masters.get(index)
    }

    /// Get a mutable master port by index.
    pub fn master_mut(&mut self, index: usize) -> Option<&mut StreamPort> {
        self.masters.get_mut(index)
    }

    /// Get a slave port by index.
    pub fn slave(&self, index: usize) -> Option<&StreamPort> {
        self.slaves.get(index)
    }

    /// Get a mutable slave port by index.
    pub fn slave_mut(&mut self, index: usize) -> Option<&mut StreamPort> {
        self.slaves.get_mut(index)
    }

    /// Get the packet config for a master port.
    pub fn master_packet_cfg(&self, index: usize) -> Option<&MasterPacketConfig> {
        self.master_packet_config.get(index)
    }

    /// Find the TileCtrl slave port index.
    ///
    /// Returns the index into `self.slaves` for the port tagged as
    /// `PortType::TileCtrl`. This is port 3 on compute tiles, port 6 on
    /// mem tiles, and port 0 on shim tiles (per AM025).
    ///
    /// Used by OP_READ response injection: the response packet is pushed
    /// into this slave port so the packet routing infrastructure can
    /// deliver it to the configured destination.
    pub fn tile_ctrl_slave_port(&self) -> Option<usize> {
        self.slaves.iter().position(|p| matches!(p.port_type, PortType::TileCtrl))
    }

    /// Find a mutable reference to the TileCtrl slave port.
    pub fn tile_ctrl_slave_mut(&mut self) -> Option<&mut StreamPort> {
        self.slaves.iter_mut().find(|p| matches!(p.port_type, PortType::TileCtrl))
    }

    /// Find a DMA master port (for MM2S).
    pub fn dma_master(&self, channel: u8) -> Option<&StreamPort> {
        self.masters
            .iter()
            .find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a mutable DMA master port (for MM2S).
    pub fn dma_master_mut(&mut self, channel: u8) -> Option<&mut StreamPort> {
        self.masters
            .iter_mut()
            .find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a DMA slave port (for S2MM).
    pub fn dma_slave(&self, channel: u8) -> Option<&StreamPort> {
        self.slaves
            .iter()
            .find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a mutable DMA slave port (for S2MM).
    pub fn dma_slave_mut(&mut self, channel: u8) -> Option<&mut StreamPort> {
        self.slaves
            .iter_mut()
            .find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Check if any port or pipeline has pending data.
    pub fn has_pending_data(&self) -> bool {
        !self.switch_pipeline.is_empty()
            || self.masters.iter().any(|p| p.has_data())
            || self.slaves.iter().any(|p| p.has_data())
    }

    /// Check if the switch pipeline has in-flight words.
    pub fn has_pipeline_data(&self) -> bool {
        !self.switch_pipeline.is_empty()
    }

    /// Begin a new routing cycle for port activity tracking.
    ///
    /// Seeds `cycle_active` from pre-existing FIFO state (ports with data
    /// from backpressure in previous cycles are already "running") and
    /// clears it for empty ports. Called at the start of each
    /// `step_data_movement()` so that `push()` during routing also marks
    /// ports active. After routing completes, the coordinator reads
    /// `cycle_active` to generate PORT_RUNNING trace events.
    pub fn begin_routing_cycle(&mut self) {
        for port in &mut self.masters {
            port.cycle_active = port.has_data();
            port.cycle_beat = false;
            port.cycle_stalled = false;
            port.cycle_tlast = false;
        }
        for port in &mut self.slaves {
            port.cycle_active = port.has_data();
            port.cycle_beat = false;
            port.cycle_stalled = false;
            port.cycle_tlast = false;
        }
    }

    /// Mark ports stalled after all routing for this cycle has completed.
    ///
    /// A port is stalled when it holds buffered data but no beat crossed it
    /// this cycle -- it has something to move but was backpressured (downstream
    /// FIFO full / arbiter contention). This is the HW PORT_STALLED condition.
    /// Evaluated once at the end of the routing cycle so it reflects the final
    /// `cycle_beat` state (the local route runs in two passes, around inter-tile
    /// propagation, and a beat in either pass must win over a stall): a port
    /// that beat this cycle is running, not stalled -- the two are exclusive,
    /// matching HW where PORT_RUNNING and PORT_STALLED tile a transfer
    /// complementarily (confirmed on NPU1 add_one_using_dma: the memtile MM2S
    /// send port's RUNNING and STALLED fill each other's gaps exactly).
    ///
    /// Packet routes already set `cycle_stalled` mid-route (step_packet_routes);
    /// this is additive (`||`) so those are preserved. Circuit routes never set
    /// it, so before this pass PORT_STALLED never fired on circuit-routed DMA
    /// ports even though HW asserts it.
    pub fn mark_stalled_ports(&mut self) {
        for port in &mut self.masters {
            port.cycle_stalled = port.cycle_stalled || (port.has_data() && !port.cycle_beat);
        }
        for port in &mut self.slaves {
            port.cycle_stalled = port.cycle_stalled || (port.has_data() && !port.cycle_beat);
        }
    }

    /// Get total data in all FIFOs.
    pub fn total_fifo_level(&self) -> usize {
        self.masters.iter().map(|p| p.fifo_level()).sum::<usize>()
            + self.slaves.iter().map(|p| p.fifo_level()).sum::<usize>()
    }

    /// Clear all FIFOs.
    pub fn clear_all(&mut self) {
        for port in &mut self.masters {
            port.clear();
        }
        for port in &mut self.slaves {
            port.clear();
        }
    }

    /// Configure a route from slave to master within this switch.
    ///
    /// This sets up a local route that will forward data from the specified
    /// slave port to the specified master port when `step()` is called.
    pub fn configure_local_route(&mut self, slave_idx: usize, master_idx: usize) {
        // Enable the ports
        if let Some(master) = self.masters.get_mut(master_idx) {
            master.enabled = true;
        }
        if let Some(slave) = self.slaves.get_mut(slave_idx) {
            slave.enabled = true;
        }

        // Add the local route with latency based on port types (avoid duplicates)
        let slave_type = self.slaves.get(slave_idx).map(|p| p.port_type).unwrap_or(PortType::Core);
        let master_type = self.masters.get(master_idx).map(|p| p.port_type).unwrap_or(PortType::Core);
        let route =
            LocalRoute::with_port_latency(slave_idx as u8, master_idx as u8, &slave_type, &master_type);
        if !self
            .local_routes
            .iter()
            .any(|r| r.slave_idx == route.slave_idx && r.master_idx == route.master_idx)
        {
            self.local_routes.push(route);
        }
    }

    /// Remove a local route.
    pub fn remove_local_route(&mut self, slave_idx: usize, master_idx: usize) {
        self.local_routes
            .retain(|r| !(r.slave_idx == slave_idx as u8 && r.master_idx == master_idx as u8));
    }

    /// Clear all local routes.
    pub fn clear_local_routes(&mut self) {
        self.local_routes.clear();
    }

    /// Step the stream switch: forward data along circuit and packet routes.
    ///
    /// Circuit routes: for each configured route, if the source slave port
    /// has data and the destination master port has space, move one word.
    /// Packet routes: process packet headers and forward based on slot config.
    ///
    /// Returns the number of words forwarded.
    pub fn step(&mut self) -> usize {
        let mut words_forwarded = 0;

        // Phase 1: Advance pipeline -- deliver words that have completed traversal
        words_forwarded += self.advance_switch_pipeline();

        // Step packet-switched routes (they consume from slave FIFOs
        // that circuit routes should not also consume from)
        words_forwarded += self.step_packet_routes();

        // Phase 2: Accept new words from slaves into the pipeline.
        //
        // Circuit-mode multicast: a single slave port can connect to multiple
        // master ports (hardware duplicates the word in the switch fabric).
        // We group routes by slave_idx so we pop once and push a pipeline
        // entry for every destination master.
        //
        // Per-route backpressure: a route from slave to master models a
        // `latency`-cycle pipe feeding the master's FIFO.  The total in-flight
        // budget for a route is `latency + master.fifo_capacity` words --
        // `latency` in the pipe (one per cycle of overlap) plus the master's
        // own FIFO depth.  When in-flight reaches that budget, the slave can
        // no longer be popped for that route; the slave's FIFO holds the
        // word and back-pressures upstream (the DMA MM2S engine sees its
        // stream-out slave full and stalls in Transferring with
        // Stalled_Stream_Backpressure asserted).
        //
        // Without this gating, `step()` drained slaves unconditionally into an
        // unbounded `switch_pipeline` Vec, so MM2S never saw downstream
        // back-pressure and self-chained DMAs (the
        // memtile_dmas/blockwrite_using_locks pattern) ran forever -- the
        // engine never reached Halted/Stalled and natural completion never
        // fired even though the sync target was satisfied.
        //
        // For multicast: every destination must have room.  If any one peer
        // is at capacity, the slave is held until that peer drains.  This
        // matches HW's all-or-nothing multicast semantics (a switch-fabric
        // copy must land at every destination atomically).
        //
        // Collect which slaves have been consumed this cycle to avoid
        // double-popping when multiple routes share the same slave.
        let mut slave_consumed: u64 = 0; // bitset, up to 64 slave ports

        // Pre-compute pipeline occupancy per destination master so we don't
        // re-scan `switch_pipeline` for every (route, peer) check.
        let mut pipeline_to_master: Vec<usize> = vec![0; self.masters.len()];
        for w in &self.switch_pipeline {
            let m = w.master_idx as usize;
            if m < pipeline_to_master.len() {
                pipeline_to_master[m] += 1;
            }
        }

        for i in 0..self.local_routes.len() {
            let route = &self.local_routes[i];
            if !route.enabled {
                continue;
            }

            let slave_idx = route.slave_idx as usize;
            let master_idx = route.master_idx as usize;

            // Check bounds
            if slave_idx >= self.slaves.len() || master_idx >= self.masters.len() {
                continue;
            }

            // Already consumed by a prior route in this cycle (multicast peer)
            if slave_idx < 64 && (slave_consumed & (1u64 << slave_idx)) != 0 {
                continue;
            }

            let slave_has_data = self.slaves[slave_idx].has_data();

            log::trace!(
                "TileSwitch({},{}): route slave[{}]->master[{}] has_data={} fifo_len={}",
                self.col,
                self.row,
                slave_idx,
                master_idx,
                slave_has_data,
                self.slaves[slave_idx].fifo.len()
            );

            if !slave_has_data {
                continue;
            }

            // Per-peer backpressure check: every multicast destination must
            // have room for one more in-flight word.  Budget per peer is
            // `peer.latency + master.fifo_capacity`; in-flight is
            // `pipeline_to_master[m] + masters[m].fifo.len()`.
            let mut can_pop = true;
            for j in 0..self.local_routes.len() {
                let peer = &self.local_routes[j];
                if peer.slave_idx as usize != slave_idx || !peer.enabled {
                    continue;
                }
                let peer_master = peer.master_idx as usize;
                if peer_master >= self.masters.len() {
                    continue;
                }
                let in_flight = pipeline_to_master[peer_master] + self.masters[peer_master].fifo.len();
                let budget = peer.latency as usize + self.masters[peer_master].fifo_capacity;
                if in_flight >= budget {
                    log::trace!(
                        "TileSwitch({},{}): slave[{}]->master[{}] backpressured \
                         (in_flight={} budget={}=lat{}+cap{})",
                        self.col,
                        self.row,
                        slave_idx,
                        peer_master,
                        in_flight,
                        budget,
                        peer.latency,
                        self.masters[peer_master].fifo_capacity,
                    );
                    can_pop = false;
                    break;
                }
            }

            if !can_pop {
                continue;
            }

            // Pop once from slave
            if let Some((data, tlast)) = self.slaves[slave_idx].pop_with_tlast() {
                if slave_idx < 64 {
                    slave_consumed |= 1u64 << slave_idx;
                }

                // Find ALL enabled routes from this slave (multicast fanout).
                // Search the full route list -- routes for the same slave may
                // appear at any position.
                let mut dest_count: u32 = 0;
                for j in 0..self.local_routes.len() {
                    let peer = &self.local_routes[j];
                    if peer.slave_idx as usize != slave_idx || !peer.enabled {
                        continue;
                    }
                    let peer_master = peer.master_idx as usize;
                    if peer_master >= self.masters.len() {
                        continue;
                    }
                    self.switch_pipeline.push(InSwitchWord {
                        master_idx: peer.master_idx,
                        data,
                        tlast,
                        cycles_remaining: peer.latency,
                    });
                    pipeline_to_master[peer_master] += 1;
                    dest_count += 1;
                    log::debug!(
                        "TileSwitch({},{}): slave[{}] -> pipeline({}) -> master[{}] data=0x{:08X}{}{}",
                        self.col,
                        self.row,
                        slave_idx,
                        peer.latency,
                        peer_master,
                        data,
                        if tlast { " TLAST" } else { "" },
                        if dest_count > 1 { " (multicast)" } else { "" }
                    );
                }
                let _ = dest_count; // used in log messages above
            }
        }

        words_forwarded
    }

    /// Advance the intra-tile switch pipeline by one cycle.
    ///
    /// Decrements countdown timers and delivers words whose traversal is complete.
    /// Returns the number of words delivered to master ports.
    fn advance_switch_pipeline(&mut self) -> usize {
        let mut delivered = 0;
        let mut i = 0;

        while i < self.switch_pipeline.len() {
            let word = &mut self.switch_pipeline[i];
            if word.cycles_remaining > 0 {
                word.cycles_remaining -= 1;
            }

            if word.cycles_remaining == 0 {
                let master_idx = word.master_idx as usize;
                if master_idx < self.masters.len() && self.masters[master_idx].can_accept() {
                    let data = word.data;
                    let tlast = word.tlast;
                    self.masters[master_idx].push_with_tlast(data, tlast);
                    self.switch_pipeline.remove(i);
                    delivered += 1;
                } else {
                    // Master full -- backpressure, retry next cycle
                    i += 1;
                }
            } else {
                i += 1;
            }
        }

        delivered
    }

    /// Check if any local route has data pending (including in pipeline).
    pub fn has_pending_local(&self) -> bool {
        if !self.switch_pipeline.is_empty() {
            return true;
        }
        for route in &self.local_routes {
            if !route.enabled {
                continue;
            }
            let slave_idx = route.slave_idx as usize;
            if slave_idx < self.slaves.len() && self.slaves[slave_idx].has_data() {
                return true;
            }
        }
        false
    }

    /// Get the number of configured local routes.
    pub fn local_route_count(&self) -> usize {
        self.local_routes.len()
    }

    // ========================================================================
    // Packet routing configuration and forwarding
    // ========================================================================

    /// Configure a slave port's packet slot from a register write.
    ///
    /// Each slave port has 4 slots (0-3). The register value encodes
    /// ID, mask, enable, msel, and arbiter fields per aie-rt format.
    pub fn configure_slave_slot(&mut self, slave_port: usize, slot: usize, value: u32) {
        if slave_port < self.slave_slots.len() && slot < 4 {
            let parsed = PacketSlot::from_register(value);
            log::debug!(
                "TileSwitch({},{}): slave[{}] slot[{}] = id={} mask=0x{:02X} en={} msel={} arb={}",
                self.col,
                self.row,
                slave_port,
                slot,
                parsed.pkt_id,
                parsed.mask,
                parsed.enable,
                parsed.msel,
                parsed.arbiter
            );
            self.slave_slots[slave_port][slot] = parsed;
        }
    }

    /// Configure a master port's packet mode from its config register.
    ///
    /// Extracts packet_enable, drop_header, arbiter, and msel_enable.
    pub fn configure_master_packet(&mut self, master_port: usize, value: u32) {
        if master_port < self.master_packet_config.len() {
            let parsed = MasterPacketConfig::from_register(value);
            log::debug!(
                "TileSwitch({},{}): master[{}] pkt_en={} drop_hdr={} arb={} msel_en=0b{:04b}",
                self.col,
                self.row,
                master_port,
                parsed.packet_enable,
                parsed.drop_header,
                parsed.arbiter,
                parsed.msel_enable
            );
            self.master_packet_config[master_port] = parsed;
        }
    }

    /// Resolve which master ports a packet should route to.
    ///
    /// Scans the slave's 4 slots for a match on `pkt_id`, then finds all
    /// masters on the matching arbiter whose msel_enable includes the slot's msel.
    ///
    /// Returns `(target_masters, all_drop_header, arbiter)`.
    fn resolve_packet_route(&self, slave_port: usize, pkt_id: u8) -> Option<(Vec<u8>, bool, u8)> {
        if slave_port >= self.slave_slots.len() {
            return None;
        }

        // Find first matching slot
        for slot in &self.slave_slots[slave_port] {
            if !slot.matches(pkt_id) {
                continue;
            }

            // Find all master ports on this arbiter+msel
            let mut masters = Vec::new();
            let mut all_drop_header = true;
            for (idx, mcfg) in self.master_packet_config.iter().enumerate() {
                if mcfg.accepts(slot.arbiter, slot.msel) {
                    masters.push(idx as u8);
                    if !mcfg.drop_header {
                        all_drop_header = false;
                    }
                }
            }

            if !masters.is_empty() {
                log::debug!("TileSwitch({},{}): pkt_id={} matched slave[{}] slot(arb={},msel={}) -> masters {:?} drop_hdr={}",
                    self.col, self.row, pkt_id, slave_port,
                    slot.arbiter, slot.msel, masters, all_drop_header);
                return Some((masters, all_drop_header, slot.arbiter));
            }
        }

        None
    }

    /// Step packet-switched routing for all slave ports.
    ///
    /// Per slave with enabled packet slots:
    /// 1. Header arrival (idle -> active): peek + decode header, resolve
    ///    route, check arbiter lock / RR priority, pop header, build
    ///    `ActivePacket` with one `TargetState` per destination master
    ///    (header queued in each non-drop-header target's `pending`).
    /// 2. Mid-packet fill: pop one slave word and push a copy into every
    ///    target's `pending`, but only if every target's queue is below
    ///    `MAX_PENDING_PER_TARGET`. This bound replaces the old
    ///    all-or-nothing "every master can_accept" rule that deadlocked
    ///    multicast through diverging+reconverging paths.
    /// 3. Drain: each target independently pushes as many of its `pending`
    ///    words into its master FIFO as the master can accept this cycle.
    /// 4. Completion: when the slave has popped TLAST and every target's
    ///    `pending` is empty, release the arbiter and the active slot.
    ///
    /// Header arrival also runs steps 3-4 in the same cycle so a header
    /// reaches its master in the same step it was popped (preserving the
    /// 1-cycle-per-word throughput of the previous implementation when
    /// the multicast deadlock isn't engaged).
    ///
    /// Returns the number of words popped from slaves.
    pub fn step_packet_routes(&mut self) -> usize {
        let mut words_forwarded = 0;
        let num_slaves = self.slaves.len();

        for slave_idx in 0..num_slaves {
            // Skip slaves not in packet mode. The slave config register's
            // packet_enable bit (bit 30) determines whether this port uses
            // packet routing. Without this check, raw data on circuit-mode
            // ports would be misinterpreted as packet headers.
            if !self.slaves[slave_idx].packet_enable {
                continue;
            }
            let has_any_slot = self
                .slave_slots
                .get(slave_idx)
                .map_or(false, |slots| slots.iter().any(|s| s.enable));
            if !has_any_slot {
                continue;
            }

            // ----------------------------------------------------------------
            // Phase A: Header arrival (only when idle).
            // Pops the header word and creates an `ActivePacket` with per-
            // target pending queues, queueing the header for non-drop-header
            // targets. Skipped if a packet is already in progress.
            // ----------------------------------------------------------------
            if self.active_packets[slave_idx].is_none() && self.slaves[slave_idx].has_data() {
                let header_word = self.slaves[slave_idx].peek().unwrap();
                let (header, _parity_ok) = PacketHeader::decode(header_word);
                let pkt_id = header.stream_id;

                match self.resolve_packet_route(slave_idx, pkt_id) {
                    Some((masters, all_drop_header, arbiter)) => {
                        // Arbiter lock: another slave may hold this arbiter.
                        if let Some(locked_by) = self.arbiter_locks[arbiter as usize] {
                            if locked_by != slave_idx {
                                self.slaves[slave_idx].cycle_stalled = true;
                                log::trace!(
                                    "TileSwitch({},{}): slave[{}] waiting for arbiter {} (held by slave[{}])",
                                    self.col,
                                    self.row,
                                    slave_idx,
                                    arbiter,
                                    locked_by
                                );
                                continue;
                            }
                        }

                        // Round-robin priority check. Prevents a lower-indexed
                        // slave from starving higher-indexed ones by always
                        // winning a contested arbiter.
                        if self.arbiter_locks[arbiter as usize].is_none() {
                            let priority = self.arbiter_priority[arbiter as usize];
                            let mut dominated = false;
                            for other in 0..num_slaves {
                                if other == slave_idx
                                    || !self.slaves[other].packet_enable
                                    || self.active_packets[other].is_some()
                                    || !self.slaves[other].has_data()
                                {
                                    continue;
                                }
                                let other_header = match self.slaves[other].peek() {
                                    Some(w) => w,
                                    None => continue,
                                };
                                let (other_hdr, _) = PacketHeader::decode(other_header);
                                if let Some((_, _, other_arb)) =
                                    self.resolve_packet_route(other, other_hdr.stream_id)
                                {
                                    if other_arb == arbiter {
                                        let my_dist = (slave_idx + num_slaves - priority) % num_slaves;
                                        let other_dist = (other + num_slaves - priority) % num_slaves;
                                        if other_dist < my_dist {
                                            dominated = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if dominated {
                                self.slaves[slave_idx].cycle_stalled = true;
                                log::trace!(
                                    "TileSwitch({},{}): slave[{}] deferred by RR priority for arbiter {}",
                                    self.col,
                                    self.row,
                                    slave_idx,
                                    arbiter
                                );
                                continue;
                            }
                        }

                        // Pop the header from the slave and build per-target
                        // pendings. Header is queued only for masters that
                        // don't drop_header.
                        let (data, tlast) = self.slaves[slave_idx].pop_with_tlast().unwrap();
                        let mut targets = Vec::with_capacity(masters.len());
                        for &m_idx in &masters {
                            let mut t = TargetState::new(m_idx);
                            let m = m_idx as usize;
                            let drop = m < self.master_packet_config.len()
                                && self.master_packet_config[m].drop_header;
                            if !drop {
                                t.pending.push_back((data, tlast));
                            }
                            targets.push(t);
                        }
                        words_forwarded += 1;

                        log::info!(
                            "TileSwitch({},{}): pkt header 0x{:08X} (id={}) slave[{}] -> masters {:?} arb={} all_drop={}{}",
                            self.col, self.row, data, pkt_id, slave_idx, masters, arbiter,
                            all_drop_header,
                            if tlast { " TLAST" } else { "" }
                        );

                        // Lock the arbiter from header arrival through full
                        // drain (released later by completion check). Even a
                        // header-with-TLAST packet may need to drain its
                        // pendings over several cycles if any master is full.
                        self.arbiter_locks[arbiter as usize] = Some(slave_idx);
                        self.active_packets[slave_idx] =
                            Some(ActivePacket { targets, words_forwarded: 1, arbiter, tlast_seen: tlast });
                    }
                    None => {
                        // No route configured. On real hardware packets always
                        // have a valid route (CDO sets them up). This indicates
                        // a CDO parsing or configuration bug. Dump full slot
                        // config for this slave to aid diagnosis.
                        let port_type = &self.slaves[slave_idx].port_type;
                        let slots: Vec<String> = self.slave_slots[slave_idx]
                            .iter()
                            .enumerate()
                            .map(|(i, s)| {
                                format!(
                                    "slot[{}]: en={} id={} mask=0x{:02X} msel={} arb={}",
                                    i, s.enable, s.pkt_id, s.mask, s.msel, s.arbiter
                                )
                            })
                            .collect();
                        let msg = format!(
                            "TileSwitch({},{}): no packet route for pkt_id={} on slave[{}] ({:?}) \
                             header=0x{:08X} -- configured slots: [{}]",
                            self.col,
                            self.row,
                            pkt_id,
                            slave_idx,
                            port_type,
                            header_word,
                            slots.join(", "),
                        );
                        log::error!("{}", msg);
                        self.fatal_errors.push(msg);
                        self.slaves[slave_idx].pop();
                        // Nothing further to do this iteration for this slave.
                        continue;
                    }
                }
            }
            // ----------------------------------------------------------------
            // Phase B: Mid-packet fill (only when active, no fill in the
            // same cycle as Phase A above -- Phase A already enqueued the
            // header).
            // ----------------------------------------------------------------
            else if self.active_packets[slave_idx].is_some() {
                let (should_fill, room_blocked) = {
                    let active = self.active_packets[slave_idx].as_ref().unwrap();
                    let all_have_room =
                        active.targets.iter().all(|t| t.pending.len() < MAX_PENDING_PER_TARGET);
                    let want = !active.tlast_seen && self.slaves[slave_idx].has_data();
                    (want && all_have_room, want && !all_have_room)
                };
                if should_fill {
                    let (data, tlast) = self.slaves[slave_idx].pop_with_tlast().unwrap();
                    let active = self.active_packets[slave_idx].as_mut().unwrap();
                    for target in &mut active.targets {
                        target.pending.push_back((data, tlast));
                    }
                    active.words_forwarded += 1;
                    if tlast {
                        active.tlast_seen = true;
                    }
                    words_forwarded += 1;
                    log::trace!(
                        "TileSwitch({},{}): pkt data 0x{:08X} slave[{}] -> targets {:?}{}",
                        self.col,
                        self.row,
                        data,
                        slave_idx,
                        active.targets.iter().map(|t| t.master_idx).collect::<Vec<_>>(),
                        if tlast { " TLAST" } else { "" }
                    );
                } else if room_blocked {
                    self.slaves[slave_idx].cycle_stalled = true;
                    log::trace!(
                        "TileSwitch({},{}): slave[{}] mid-packet stall (per-target buffer full)",
                        self.col,
                        self.row,
                        slave_idx
                    );
                }
            }

            // ----------------------------------------------------------------
            // Phase C: Per-target drain. Each target independently pushes as
            // many of its `pending` words into its master FIFO as the master
            // can accept this cycle. A slow target does not block a fast one.
            // ----------------------------------------------------------------
            if self.active_packets[slave_idx].is_some() {
                let target_count = self.active_packets[slave_idx].as_ref().unwrap().targets.len();
                for t in 0..target_count {
                    let m_idx =
                        self.active_packets[slave_idx].as_ref().unwrap().targets[t].master_idx as usize;
                    if m_idx >= self.masters.len() {
                        continue;
                    }
                    loop {
                        let front = self.active_packets[slave_idx].as_ref().unwrap().targets[t]
                            .pending
                            .front()
                            .copied();
                        let Some((data, tlast)) = front else { break };
                        if !self.masters[m_idx].can_accept() {
                            break;
                        }
                        self.masters[m_idx].push_with_tlast(data, tlast);
                        self.active_packets[slave_idx].as_mut().unwrap().targets[t].pending.pop_front();
                    }
                }

                // ------------------------------------------------------------
                // Phase D: Completion. Release arbiter + active slot once
                // the slave has popped TLAST AND every target has fully
                // drained its pending into its master.
                // ------------------------------------------------------------
                let done = {
                    let active = self.active_packets[slave_idx].as_ref().unwrap();
                    active.tlast_seen && active.targets.iter().all(|t| t.pending.is_empty())
                };
                if done {
                    let arb = self.active_packets[slave_idx].as_ref().unwrap().arbiter as usize;
                    self.arbiter_locks[arb] = None;
                    self.arbiter_priority[arb] = (slave_idx + 1) % num_slaves;
                    log::debug!(
                        "TileSwitch({},{}): slave[{}] packet complete, arbiter {} released",
                        self.col,
                        self.row,
                        slave_idx,
                        arb
                    );
                    self.active_packets[slave_idx] = None;
                }
            }
        }

        words_forwarded
    }

    /// Check if any packet route has data pending.
    pub fn has_pending_packet(&self) -> bool {
        for slave_idx in 0..self.slaves.len() {
            if self.active_packets[slave_idx].is_some() {
                return true;
            }
            if !self.slaves[slave_idx].packet_enable {
                continue;
            }
            let has_any_slot = self
                .slave_slots
                .get(slave_idx)
                .map_or(false, |slots| slots.iter().any(|s| s.enable));
            if has_any_slot && self.slaves[slave_idx].has_data() {
                return true;
            }
        }
        false
    }
}
