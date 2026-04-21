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

pub use ports::*;
pub use packet_types::*;
pub use packet_switch::*;

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
            .map(|(idx, &encoded)| {
                StreamPort::new(idx as u8, direction, PortType::from_spec(encoded))
            })
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
        self.masters.iter().find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a mutable DMA master port (for MM2S).
    pub fn dma_master_mut(&mut self, channel: u8) -> Option<&mut StreamPort> {
        self.masters.iter_mut().find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a DMA slave port (for S2MM).
    pub fn dma_slave(&self, channel: u8) -> Option<&StreamPort> {
        self.slaves.iter().find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
    }

    /// Find a mutable DMA slave port (for S2MM).
    pub fn dma_slave_mut(&mut self, channel: u8) -> Option<&mut StreamPort> {
        self.slaves.iter_mut().find(|p| matches!(p.port_type, PortType::Dma(ch) if ch == channel))
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
            port.cycle_stalled = false;
            port.cycle_tlast = false;
        }
        for port in &mut self.slaves {
            port.cycle_active = port.has_data();
            port.cycle_stalled = false;
            port.cycle_tlast = false;
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
        let route = LocalRoute::with_port_latency(slave_idx as u8, master_idx as u8, &slave_type, &master_type);
        if !self.local_routes.iter().any(|r| {
            r.slave_idx == route.slave_idx && r.master_idx == route.master_idx
        }) {
            self.local_routes.push(route);
        }
    }

    /// Remove a local route.
    pub fn remove_local_route(&mut self, slave_idx: usize, master_idx: usize) {
        self.local_routes.retain(|r| {
            !(r.slave_idx == slave_idx as u8 && r.master_idx == master_idx as u8)
        });
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
        // Collect which slaves have been consumed this cycle to avoid
        // double-popping when multiple routes share the same slave.
        let mut slave_consumed: u64 = 0; // bitset, up to 64 slave ports

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

            log::trace!("TileSwitch({},{}): route slave[{}]->master[{}] has_data={} fifo_len={}",
                self.col, self.row, slave_idx, master_idx, slave_has_data,
                self.slaves[slave_idx].fifo.len());

            if slave_has_data {
                // Pop once from slave
                if let Some((data, tlast)) = self.slaves[slave_idx].pop_with_tlast() {
                    if slave_idx < 64 {
                        slave_consumed |= 1u64 << slave_idx;
                    }

                    // Find ALL enabled routes from this slave (multicast fanout)
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
                        dest_count += 1;
                        log::debug!("TileSwitch({},{}): slave[{}] -> pipeline({}) -> master[{}] data=0x{:08X}{}{}",
                            self.col, self.row, slave_idx, peer.latency, peer_master, data,
                            if tlast { " TLAST" } else { "" },
                            if dest_count > 1 { " (multicast)" } else { "" });
                    }
                    let _ = dest_count; // used in log messages above
                }
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
            log::debug!("TileSwitch({},{}): slave[{}] slot[{}] = id={} mask=0x{:02X} en={} msel={} arb={}",
                self.col, self.row, slave_port, slot,
                parsed.pkt_id, parsed.mask, parsed.enable, parsed.msel, parsed.arbiter);
            self.slave_slots[slave_port][slot] = parsed;
        }
    }

    /// Configure a master port's packet mode from its config register.
    ///
    /// Extracts packet_enable, drop_header, arbiter, and msel_enable.
    pub fn configure_master_packet(&mut self, master_port: usize, value: u32) {
        if master_port < self.master_packet_config.len() {
            let parsed = MasterPacketConfig::from_register(value);
            log::debug!("TileSwitch({},{}): master[{}] pkt_en={} drop_hdr={} arb={} msel_en=0b{:04b}",
                self.col, self.row, master_port,
                parsed.packet_enable, parsed.drop_header, parsed.arbiter, parsed.msel_enable);
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
    /// For each slave with enabled packet slots:
    /// - If idle: peek at first word, decode as header, resolve route, begin forwarding
    /// - If mid-packet: pop word, push to all target masters, end on TLAST
    ///
    /// Returns the number of words forwarded.
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

            // Also skip if no slots are configured (defensive)
            let has_any_slot = self.slave_slots.get(slave_idx)
                .map_or(false, |slots| slots.iter().any(|s| s.enable));
            if !has_any_slot {
                continue;
            }

            // Skip if slave has no data
            if !self.slaves[slave_idx].has_data() {
                continue;
            }

            if self.active_packets[slave_idx].is_none() {
                // === Idle: look for a new packet header ===
                let header_word = match self.slaves[slave_idx].peek() {
                    Some(w) => w,
                    None => continue,
                };

                // Decode the stream header to get pkt_id
                let (header, _parity_ok) = PacketHeader::decode(header_word);
                let pkt_id = header.stream_id;

                // Resolve route
                match self.resolve_packet_route(slave_idx, pkt_id) {
                    Some((masters, all_drop_header, arbiter)) => {
                        // Check arbiter lock: another slave may hold this arbiter.
                        if let Some(locked_by) = self.arbiter_locks[arbiter as usize] {
                            if locked_by != slave_idx {
                                self.slaves[slave_idx].cycle_stalled = true;
                                log::trace!("TileSwitch({},{}): slave[{}] waiting for arbiter {} (held by slave[{}])",
                                    self.col, self.row, slave_idx, arbiter, locked_by);
                                continue;
                            }
                        }

                        // Round-robin priority check: if another slave also
                        // wants this arbiter and has higher priority (closer
                        // to the priority pointer in round-robin order), defer.
                        // This prevents a lower-indexed slave from starving
                        // higher-indexed ones by always winning the arbiter.
                        if self.arbiter_locks[arbiter as usize].is_none() {
                            let priority = self.arbiter_priority[arbiter as usize];
                            // Check if any OTHER packet-enabled slave with data
                            // also wants this arbiter and has higher RR priority
                            let mut dominated = false;
                            for other in 0..num_slaves {
                                if other == slave_idx { continue; }
                                if !self.slaves[other].packet_enable { continue; }
                                if self.active_packets[other].is_some() { continue; }
                                if !self.slaves[other].has_data() { continue; }
                                // Check if 'other' also routes to this arbiter
                                let other_header = match self.slaves[other].peek() {
                                    Some(w) => w,
                                    None => continue,
                                };
                                let (other_hdr, _) = PacketHeader::decode(other_header);
                                if let Some((_, _, other_arb)) = self.resolve_packet_route(other, other_hdr.stream_id) {
                                    if other_arb == arbiter {
                                        // Both want the same arbiter. Compare RR distance.
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
                                log::trace!("TileSwitch({},{}): slave[{}] deferred by RR priority for arbiter {}",
                                    self.col, self.row, slave_idx, arbiter);
                                continue;
                            }
                        }

                        // Backpressure: check that all target masters can accept
                        // BEFORE consuming from slave. If any master is full,
                        // hold data in the slave FIFO until next cycle.
                        if !all_drop_header {
                            let all_can_accept = masters.iter().all(|&m| {
                                let m = m as usize;
                                m >= self.masters.len()
                                    || self.master_packet_config[m].drop_header
                                    || self.masters[m].can_accept()
                            });
                            if !all_can_accept {
                                self.slaves[slave_idx].cycle_stalled = true;
                                log::trace!("TileSwitch({},{}): slave[{}] header backpressure (master full)",
                                    self.col, self.row, slave_idx);
                                continue;
                            }
                        }

                        // Consume the header word from slave
                        let (data, tlast) = self.slaves[slave_idx].pop_with_tlast().unwrap();

                        // Forward header to masters that don't drop it
                        if !all_drop_header {
                            for &m in &masters {
                                let m = m as usize;
                                if m < self.masters.len() && !self.master_packet_config[m].drop_header {
                                    self.masters[m].push_with_tlast(data, tlast);
                                }
                            }
                        }
                        words_forwarded += 1;

                        log::info!("TileSwitch({},{}): pkt header 0x{:08X} (id={}) slave[{}] -> masters {:?} arb={} all_drop={}{}",
                            self.col, self.row, data, pkt_id, slave_idx, masters, arbiter,
                            all_drop_header,
                            if tlast { " TLAST" } else { "" });

                        if tlast {
                            // Single-word packet (header only with TLAST)
                            // No lock needed -- packet is complete
                        } else {
                            // Lock the arbiter for this multi-word packet
                            self.arbiter_locks[arbiter as usize] = Some(slave_idx);
                            self.active_packets[slave_idx] = Some(ActivePacket {
                                target_masters: masters,
                                words_forwarded: 0,
                                arbiter,
                            });
                        }
                    }
                    None => {
                        // No route configured. On real hardware, packets
                        // always have a valid route (CDO sets them up).
                        // This indicates a CDO parsing or configuration bug.
                        //
                        // Dump full slot config for this slave to aid diagnosis.
                        let port_type = &self.slaves[slave_idx].port_type;
                        let slots: Vec<String> = self.slave_slots[slave_idx].iter()
                            .enumerate()
                            .map(|(i, s)| format!(
                                "slot[{}]: en={} id={} mask=0x{:02X} msel={} arb={}",
                                i, s.enable, s.pkt_id, s.mask, s.msel, s.arbiter
                            ))
                            .collect();
                        let msg = format!(
                            "TileSwitch({},{}): no packet route for pkt_id={} on slave[{}] ({:?}) \
                             header=0x{:08X} -- configured slots: [{}]",
                            self.col, self.row, pkt_id, slave_idx, port_type,
                            header_word, slots.join(", "),
                        );
                        log::error!("{}", msg);
                        self.fatal_errors.push(msg);
                        self.slaves[slave_idx].pop();
                    }
                }
            } else {
                // === Mid-packet: forward data word ===

                // Backpressure: check that all target masters can accept
                // BEFORE consuming from slave. If any master is full,
                // hold data in the slave FIFO until next cycle.
                let active = self.active_packets[slave_idx].as_ref().unwrap();
                let all_can_accept = active.target_masters.iter().all(|&m| {
                    let m = m as usize;
                    m >= self.masters.len() || self.masters[m].can_accept()
                });
                if !all_can_accept {
                    self.slaves[slave_idx].cycle_stalled = true;
                    log::trace!("TileSwitch({},{}): slave[{}] data backpressure (master full)",
                        self.col, self.row, slave_idx);
                    continue;
                }

                let (data, tlast) = match self.slaves[slave_idx].pop_with_tlast() {
                    Some(dt) => dt,
                    None => continue,
                };

                // Push to all target masters (guaranteed to succeed via check above)
                let active = self.active_packets[slave_idx].as_ref().unwrap();
                let targets: Vec<u8> = active.target_masters.clone();
                for &m in &targets {
                    let m = m as usize;
                    if m < self.masters.len() {
                        self.masters[m].push_with_tlast(data, tlast);
                    }
                }
                words_forwarded += 1;

                if let Some(ref mut active) = self.active_packets[slave_idx] {
                    active.words_forwarded += 1;
                }

                if tlast {
                    // Release arbiter lock and advance round-robin pointer
                    // past this slave so the next contender gets priority.
                    if let Some(ref active) = self.active_packets[slave_idx] {
                        let arb = active.arbiter as usize;
                        self.arbiter_locks[arb] = None;
                        self.arbiter_priority[arb] = (slave_idx + 1) % num_slaves;
                    }
                    log::debug!("TileSwitch({},{}): pkt data 0x{:08X} slave[{}] -> {:?} TLAST (end of pkt)",
                        self.col, self.row, data, slave_idx, targets);
                    self.active_packets[slave_idx] = None;
                } else {
                    log::trace!("TileSwitch({},{}): pkt data 0x{:08X} slave[{}] -> {:?}",
                        self.col, self.row, data, slave_idx, targets);
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
            let has_any_slot = self.slave_slots.get(slave_idx)
                .map_or(false, |slots| slots.iter().any(|s| s.enable));
            if has_any_slot && self.slaves[slave_idx].has_data() {
                return true;
            }
        }
        false
    }
}
