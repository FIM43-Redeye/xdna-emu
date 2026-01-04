//! Stream switch stub for DMA integration.
//!
//! This module provides a simplified model of the AIE2 stream switch,
//! focusing on the functionality needed for DMA data movement.
//!
//! # Architecture
//!
//! Each tile has a stream switch with:
//! - Master ports (output): send data to other tiles
//! - Slave ports (input): receive data from other tiles
//! - DMA integration: DMA channels connect to specific ports
//! - FIFOs for buffering
//!
//! ```text
//!                    North
//!                      ↑
//!                ┌─────┴─────┐
//!                │           │
//!       West ◄───┤  Stream   ├───► East
//!                │  Switch   │
//!                │           │
//!                └─────┬─────┘
//!                      ↓
//!              South / Core / DMA
//! ```
//!
//! # Simplifications
//!
//! This stub does NOT model:
//! - Packet switching (only circuit switching)
//! - Backpressure propagation delays
//! - Route configuration complexity
//!
//! It DOES model:
//! - Port connectivity (which ports connect to what)
//! - FIFO buffering (data can be queued)
//! - Basic latency (cycles for data to traverse)

use crate::device::aie2_spec;

/// Stream port direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDirection {
    /// Master port (sends data)
    Master,
    /// Slave port (receives data)
    Slave,
}

/// Stream port type (what the port connects to).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortType {
    /// Connects to tile to the north
    North,
    /// Connects to tile to the south
    South,
    /// Connects to tile to the east
    East,
    /// Connects to tile to the west
    West,
    /// Connects to local DMA engine
    Dma(u8), // Channel index
    /// Connects to local core
    Core,
    /// Connects to cascade interface
    Cascade,
    /// Connects to FIFO
    Fifo,
    /// Connects to trace/debug interface
    Trace,
}

impl PortType {
    /// Convert from the u8 encoding used in aie2_spec port layouts.
    ///
    /// The encoding is:
    /// - 0: Core/Tile_Ctrl
    /// - 1: FIFO
    /// - 2: Trace
    /// - 10+n: North(n)
    /// - 20+n: South(n)
    /// - 30+n: East(n)
    /// - 40+n: West(n)
    /// - 50+n: DMA(n)
    pub fn from_spec(encoded: u8) -> Self {
        use aie2_spec::port_type;
        match encoded {
            port_type::CORE => PortType::Core,
            port_type::FIFO => PortType::Fifo,
            port_type::TRACE => PortType::Trace,
            n if n >= port_type::DMA_BASE => PortType::Dma(n - port_type::DMA_BASE),
            n if n >= port_type::WEST_BASE => PortType::West,
            n if n >= port_type::EAST_BASE => PortType::East,
            n if n >= port_type::SOUTH_BASE => PortType::South,
            n if n >= port_type::NORTH_BASE => PortType::North,
            _ => PortType::Core, // Fallback
        }
    }
}

/// A single stream port.
#[derive(Debug, Clone)]
pub struct StreamPort {
    /// Port index
    pub index: u8,
    /// Port direction
    pub direction: PortDirection,
    /// Port type
    pub port_type: PortType,
    /// FIFO buffer (data waiting to be sent/received)
    pub fifo: Vec<u32>,
    /// FIFO capacity
    pub fifo_capacity: usize,
    /// Raw configuration register value (from CDO)
    pub config: u32,
    /// Connected destination (for routing)
    pub route_to: Option<(u8, u8, u8)>, // (col, row, port_index)
    /// Port is enabled
    pub enabled: bool,
}

impl StreamPort {
    /// Create a new stream port.
    pub fn new(index: u8, direction: PortDirection, port_type: PortType) -> Self {
        let fifo_capacity = match direction {
            PortDirection::Master => aie2_spec::STREAM_LOCAL_MASTER_FIFO_DEPTH as usize,
            PortDirection::Slave => aie2_spec::STREAM_LOCAL_SLAVE_FIFO_DEPTH as usize,
        };

        Self {
            index,
            direction,
            port_type,
            fifo: Vec::with_capacity(fifo_capacity),
            fifo_capacity,
            config: 0,
            route_to: None,
            enabled: false,
        }
    }

    /// Check if FIFO has data.
    pub fn has_data(&self) -> bool {
        !self.fifo.is_empty()
    }

    /// Check if FIFO can accept more data.
    pub fn can_accept(&self) -> bool {
        self.fifo.len() < self.fifo_capacity
    }

    /// Check if FIFO is full (backpressure).
    pub fn is_full(&self) -> bool {
        self.fifo.len() >= self.fifo_capacity
    }

    /// Push data into FIFO (returns false if full).
    pub fn push(&mut self, data: u32) -> bool {
        if self.can_accept() {
            self.fifo.push(data);
            true
        } else {
            false
        }
    }

    /// Pop data from FIFO.
    pub fn pop(&mut self) -> Option<u32> {
        if self.fifo.is_empty() {
            None
        } else {
            Some(self.fifo.remove(0))
        }
    }

    /// Peek at front of FIFO without removing.
    pub fn peek(&self) -> Option<u32> {
        self.fifo.first().copied()
    }

    /// Get number of items in FIFO.
    pub fn fifo_level(&self) -> usize {
        self.fifo.len()
    }

    /// Clear the FIFO.
    pub fn clear(&mut self) {
        self.fifo.clear();
    }

    /// Set the route destination.
    pub fn set_route(&mut self, dest_col: u8, dest_row: u8, dest_port: u8) {
        self.route_to = Some((dest_col, dest_row, dest_port));
        self.enabled = true;
    }

    /// Clear the route.
    pub fn clear_route(&mut self) {
        self.route_to = None;
        self.enabled = false;
    }
}

/// A local route within a stream switch (slave to master).
#[derive(Debug, Clone, Copy)]
pub struct LocalRoute {
    /// Source slave port index
    pub slave_idx: u8,
    /// Destination master port index
    pub master_idx: u8,
    /// Is route enabled
    pub enabled: bool,
}

impl LocalRoute {
    /// Create a new local route.
    pub fn new(slave_idx: u8, master_idx: u8) -> Self {
        Self {
            slave_idx,
            master_idx,
            enabled: true,
        }
    }
}

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
    /// Local routes (slave → master within this tile)
    pub local_routes: Vec<LocalRoute>,
    /// Latency in cycles for local-to-local routing
    pub local_latency: u8,
    /// Latency in cycles for external routing
    pub external_latency: u8,
}

impl StreamSwitch {
    /// Build ports from a spec layout.
    ///
    /// This takes a slice of encoded port types from aie2_spec and converts
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
    /// Port layout is defined in aie2_spec::{COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS}.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let masters = Self::build_ports_from_spec(aie2_spec::COMPUTE_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(aie2_spec::COMPUTE_SLAVE_PORTS, PortDirection::Slave);

        Self {
            col,
            row,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: aie2_spec::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: aie2_spec::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
        }
    }

    /// Create a new stream switch for a memory tile.
    ///
    /// Port layout is defined in aie2_spec::{MEMTILE_MASTER_PORTS, MEMTILE_SLAVE_PORTS}.
    ///
    /// Note the asymmetry: 6 North masters but only 4 North slaves, and
    /// 4 South masters but 6 South slaves. This matches MemTile's role as
    /// a buffer between Shim (which has 6 North outputs) and Compute tiles.
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let masters = Self::build_ports_from_spec(aie2_spec::MEMTILE_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(aie2_spec::MEMTILE_SLAVE_PORTS, PortDirection::Slave);

        Self {
            col,
            row,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: aie2_spec::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: aie2_spec::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
        }
    }

    /// Create a new stream switch for a shim tile.
    ///
    /// Port layout is defined in aie2_spec::{SHIM_MASTER_PORTS, SHIM_SLAVE_PORTS}.
    ///
    /// The 6 North masters (12-17) connect 1:1 to MemTile South slaves (7-12).
    pub fn new_shim_tile(col: u8) -> Self {
        let masters = Self::build_ports_from_spec(aie2_spec::SHIM_MASTER_PORTS, PortDirection::Master);
        let slaves = Self::build_ports_from_spec(aie2_spec::SHIM_SLAVE_PORTS, PortDirection::Slave);

        Self {
            col,
            row: 0,
            masters,
            slaves,
            local_routes: Vec::new(),
            local_latency: aie2_spec::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: aie2_spec::STREAM_EXTERNAL_TO_EXTERNAL_LATENCY,
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

    /// Check if any port has pending data.
    pub fn has_pending_data(&self) -> bool {
        self.masters.iter().any(|p| p.has_data()) || self.slaves.iter().any(|p| p.has_data())
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

        // Add the local route (avoid duplicates)
        let route = LocalRoute::new(slave_idx as u8, master_idx as u8);
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

    /// Step the stream switch: forward data along configured local routes.
    ///
    /// For each configured route, if the source slave port has data and the
    /// destination master port has space, move one word from slave to master.
    ///
    /// Returns the number of words forwarded.
    pub fn step(&mut self) -> usize {
        let mut words_forwarded = 0;

        // Process each local route
        for route in &self.local_routes {
            if !route.enabled {
                continue;
            }

            let slave_idx = route.slave_idx as usize;
            let master_idx = route.master_idx as usize;

            // Check bounds
            if slave_idx >= self.slaves.len() || master_idx >= self.masters.len() {
                continue;
            }

            // Check if we can forward (slave has data, master has space)
            let slave_has_data = self.slaves[slave_idx].has_data();
            let master_can_accept = self.masters[master_idx].can_accept();

            // Debug: trace specific MemTile routes
            if self.row == 1 && (slave_idx == 11 || slave_idx == 13) {
                let fifo_len = self.slaves[slave_idx].fifo.len();
                log::trace!("TileSwitch({},{}): route slave[{}]->master[{}] has_data={} fifo_len={} can_accept={}",
                    self.col, self.row, slave_idx, master_idx, slave_has_data, fifo_len, master_can_accept);
            }

            if slave_has_data && master_can_accept {
                // Forward one word
                if let Some(data) = self.slaves[slave_idx].pop() {
                    if self.masters[master_idx].push(data) {
                        words_forwarded += 1;
                        log::debug!("TileSwitch({},{}): slave[{}] -> master[{}] data=0x{:08X}",
                            self.col, self.row, slave_idx, master_idx, data);
                    }
                }
            }
        }

        words_forwarded
    }

    /// Check if any local route has data pending.
    pub fn has_pending_local(&self) -> bool {
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
}

/// Data packet in the stream network.
#[derive(Debug, Clone, Copy)]
pub struct StreamPacket {
    /// Data word (32 bits)
    pub data: u32,
    /// Source tile column
    pub src_col: u8,
    /// Source tile row
    pub src_row: u8,
    /// Source port index
    pub src_port: u8,
    /// Destination tile column
    pub dest_col: u8,
    /// Destination tile row
    pub dest_row: u8,
    /// Destination port index
    pub dest_port: u8,
    /// Is this the last word in a transfer?
    pub is_last: bool,
}

impl StreamPacket {
    /// Create a new packet.
    pub fn new(
        data: u32,
        src_col: u8,
        src_row: u8,
        src_port: u8,
        dest_col: u8,
        dest_row: u8,
        dest_port: u8,
    ) -> Self {
        Self {
            data,
            src_col,
            src_row,
            src_port,
            dest_col,
            dest_row,
            dest_port,
            is_last: false,
        }
    }

    /// Mark as last packet in transfer.
    pub fn with_last(mut self) -> Self {
        self.is_last = true;
        self
    }
}

// ============================================================================
// Packet-Switched Routing (AM020 Ch2)
// ============================================================================

/// Packet type for packet-switched streams.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum PacketType {
    /// Normal data packet
    #[default]
    Data = 0,
    /// Control packet
    Control = 1,
    /// Configuration packet
    Config = 2,
    /// Trace packet
    Trace = 3,
    /// Reserved types (4-7)
    Reserved = 4,
}

impl PacketType {
    /// Convert from u8.
    pub fn from_u8(val: u8) -> Self {
        match val & 0x7 {
            0 => Self::Data,
            1 => Self::Control,
            2 => Self::Config,
            3 => Self::Trace,
            _ => Self::Reserved,
        }
    }
}

/// Packet header for packet-switched streams.
///
/// The 32-bit header contains routing and control information:
/// - Stream ID (5 bits): Identifies destination
/// - Packet Type (3 bits): Data, control, config, or trace
/// - Source Row (5 bits): Originating tile row
/// - Source Column (7 bits): Originating tile column
/// - Parity (1 bit): Odd parity for error detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PacketHeader {
    /// Stream ID (destination identifier)
    pub stream_id: u8,
    /// Packet type
    pub packet_type: PacketType,
    /// Source tile row
    pub src_row: u8,
    /// Source tile column
    pub src_col: u8,
}

impl PacketHeader {
    /// Create a new packet header.
    pub fn new(stream_id: u8, src_col: u8, src_row: u8) -> Self {
        Self {
            stream_id: stream_id & 0x1F, // 5 bits
            packet_type: PacketType::Data,
            src_row: src_row & 0x1F, // 5 bits
            src_col: src_col & 0x7F, // 7 bits
        }
    }

    /// Create with specific packet type.
    pub fn with_type(mut self, ptype: PacketType) -> Self {
        self.packet_type = ptype;
        self
    }

    /// Encode to 32-bit header word.
    ///
    /// Layout (AM020 Ch2, Table 2):
    /// | 31    | 30-28 | 27-21      | 20-16     | 15  | 14-12       | 11-5    | 4-0       |
    /// | Parity| Rsvd  | Src Column | Src Row   | Rsvd| Packet Type | Rsvd    | Stream ID |
    pub fn encode(&self) -> u32 {
        let mut word: u32 = 0;

        // Stream ID: bits 4-0
        word |= (self.stream_id as u32) & aie2_spec::PACKET_STREAM_ID_MASK;

        // Packet Type: bits 14-12
        word |= ((self.packet_type as u32) & aie2_spec::PACKET_TYPE_MASK)
            << aie2_spec::PACKET_TYPE_SHIFT;

        // Source Row: bits 20-16
        word |= ((self.src_row as u32) & aie2_spec::PACKET_SRC_ROW_MASK)
            << aie2_spec::PACKET_SRC_ROW_SHIFT;

        // Source Column: bits 27-21
        word |= ((self.src_col as u32) & aie2_spec::PACKET_SRC_COL_MASK)
            << aie2_spec::PACKET_SRC_COL_SHIFT;

        // Calculate odd parity over bits 30-0
        let parity = (word.count_ones() & 1) ^ 1; // Odd parity
        word |= parity << aie2_spec::PACKET_PARITY_SHIFT;

        word
    }

    /// Decode from 32-bit header word.
    ///
    /// Returns (header, parity_ok) tuple.
    pub fn decode(word: u32) -> (Self, bool) {
        // Extract fields
        let stream_id = (word & aie2_spec::PACKET_STREAM_ID_MASK) as u8;

        let packet_type = PacketType::from_u8(
            ((word >> aie2_spec::PACKET_TYPE_SHIFT) & aie2_spec::PACKET_TYPE_MASK) as u8,
        );

        let src_row =
            ((word >> aie2_spec::PACKET_SRC_ROW_SHIFT) & aie2_spec::PACKET_SRC_ROW_MASK) as u8;

        let src_col =
            ((word >> aie2_spec::PACKET_SRC_COL_SHIFT) & aie2_spec::PACKET_SRC_COL_MASK) as u8;

        // Check parity (odd parity means total 1-bits should be odd)
        let parity_ok = word.count_ones() & 1 == 1;

        let header = Self {
            stream_id,
            packet_type,
            src_row,
            src_col,
        };

        (header, parity_ok)
    }

    /// Check if this is a data packet.
    pub fn is_data(&self) -> bool {
        self.packet_type == PacketType::Data
    }
}

/// A packet route entry in the stream switch.
///
/// Maps a stream ID to one or more destination master ports.
/// Packet-switched routing allows multicast (one stream to many destinations).
#[derive(Debug, Clone)]
pub struct PacketRoute {
    /// Stream ID that triggers this route
    pub stream_id: u8,
    /// Destination master port indices
    pub dest_ports: Vec<u8>,
    /// Is route enabled
    pub enabled: bool,
}

impl PacketRoute {
    /// Create a new packet route.
    pub fn new(stream_id: u8, dest_port: u8) -> Self {
        Self {
            stream_id,
            dest_ports: vec![dest_port],
            enabled: true,
        }
    }

    /// Create a multicast route to multiple ports.
    pub fn multicast(stream_id: u8, dest_ports: Vec<u8>) -> Self {
        Self {
            stream_id,
            dest_ports,
            enabled: true,
        }
    }

    /// Add a destination port (for multicast).
    pub fn add_dest(&mut self, port: u8) {
        if !self.dest_ports.contains(&port) {
            self.dest_ports.push(port);
        }
    }
}

/// Packet switch state for a tile.
///
/// This handles packet-switched routing where the destination is
/// determined by the stream ID in the packet header.
#[derive(Debug, Clone, Default)]
pub struct PacketSwitch {
    /// Packet routes indexed by stream ID
    routes: Vec<PacketRoute>,
    /// Current packet being received (header + data count)
    current_packet: Option<(PacketHeader, usize)>,
    /// Arbitration overhead counter
    arb_delay: u8,
}

impl PacketSwitch {
    /// Create a new packet switch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a packet route.
    pub fn add_route(&mut self, stream_id: u8, dest_port: u8) {
        // Check if route for this stream ID exists
        if let Some(route) = self.routes.iter_mut().find(|r| r.stream_id == stream_id) {
            route.add_dest(dest_port);
        } else {
            self.routes.push(PacketRoute::new(stream_id, dest_port));
        }
    }

    /// Add a multicast route.
    pub fn add_multicast_route(&mut self, stream_id: u8, dest_ports: Vec<u8>) {
        self.routes.push(PacketRoute::multicast(stream_id, dest_ports));
    }

    /// Remove all routes for a stream ID.
    pub fn remove_route(&mut self, stream_id: u8) {
        self.routes.retain(|r| r.stream_id != stream_id);
    }

    /// Clear all routes.
    pub fn clear_routes(&mut self) {
        self.routes.clear();
    }

    /// Look up destinations for a stream ID.
    pub fn lookup(&self, stream_id: u8) -> Option<&[u8]> {
        self.routes
            .iter()
            .find(|r| r.stream_id == stream_id && r.enabled)
            .map(|r| r.dest_ports.as_slice())
    }

    /// Get the number of configured routes.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }

    /// Process a packet header word.
    ///
    /// Returns the decoded header and list of destination ports.
    pub fn process_header(&mut self, word: u32) -> Option<(PacketHeader, Vec<u8>)> {
        let (header, parity_ok) = PacketHeader::decode(word);

        if !parity_ok {
            // Parity error - drop packet
            return None;
        }

        // Look up destinations
        if let Some(dests) = self.lookup(header.stream_id) {
            let dest_vec = dests.to_vec();
            self.current_packet = Some((header, 0));
            self.arb_delay = aie2_spec::PACKET_ARBITRATION_OVERHEAD_CYCLES;
            Some((header, dest_vec))
        } else {
            // No route for this stream ID
            None
        }
    }

    /// Check if arbitration delay is pending.
    pub fn has_arb_delay(&self) -> bool {
        self.arb_delay > 0
    }

    /// Tick the arbitration delay counter.
    ///
    /// Returns true if arbitration is complete.
    pub fn tick_arb_delay(&mut self) -> bool {
        if self.arb_delay > 0 {
            self.arb_delay -= 1;
        }
        self.arb_delay == 0
    }

    /// Record that a data word was processed.
    pub fn count_data_word(&mut self) {
        if let Some((_, ref mut count)) = self.current_packet {
            *count += 1;
        }
    }

    /// Complete the current packet (called when TLAST is seen).
    ///
    /// Returns the header and word count of the completed packet.
    pub fn complete_packet(&mut self) -> Option<(PacketHeader, usize)> {
        self.current_packet.take()
    }

    /// Check if currently processing a packet.
    pub fn in_packet(&self) -> bool {
        self.current_packet.is_some()
    }

    /// Get current packet info if any.
    pub fn current_packet(&self) -> Option<&(PacketHeader, usize)> {
        self.current_packet.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_port_fifo() {
        let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

        assert!(!port.has_data());
        assert!(port.can_accept());

        port.push(0xDEADBEEF);
        assert!(port.has_data());
        assert_eq!(port.peek(), Some(0xDEADBEEF));

        let data = port.pop();
        assert_eq!(data, Some(0xDEADBEEF));
        assert!(!port.has_data());
    }

    #[test]
    fn test_stream_port_backpressure() {
        let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

        // Fill the FIFO
        while port.can_accept() {
            port.push(0x12345678);
        }

        assert!(port.is_full());
        assert!(!port.push(0xFFFFFFFF)); // Should fail
    }

    #[test]
    fn test_stream_switch_compute() {
        let ss = StreamSwitch::new_compute_tile(1, 2);

        // Per AM025 AIE_TILE_MODULE: Compute tile has 2 DMA channels (0-1).
        // S2MM (slaves) and MM2S (masters) are at the same channel indices.
        assert!(ss.dma_slave(0).is_some(), "Should have DMA S2MM channel 0");
        assert!(ss.dma_slave(1).is_some(), "Should have DMA S2MM channel 1");
        assert!(ss.dma_master(0).is_some(), "Should have DMA MM2S channel 0");
        assert!(ss.dma_master(1).is_some(), "Should have DMA MM2S channel 1");

        // Verify port counts per AM025 CORE_MODULE/STREAM_SWITCH:
        // Masters: 0=Core, 1-2=DMA, 3=Tile_Ctrl, 4=FIFO0, 5-10=South(6), 11-14=West(4),
        //          15-18=North(4), 19-22=East(4) = 23 total
        // Slaves:  0=Core, 1-2=DMA, 3=Tile_Ctrl, 4=FIFO0, 5-10=South(6), 11-14=West(4),
        //          15-18=North(4), 19-22=East(4), 23-24=Trace(2) = 25 total
        assert_eq!(ss.masters.len(), 23);
        assert_eq!(ss.slaves.len(), 25);
    }

    #[test]
    fn test_stream_switch_mem_tile() {
        let ss = StreamSwitch::new_mem_tile(0, 1);

        // Per AM025 MEMORY_TILE_MODULE: MemTile has 6 DMA channels (0-5).
        for i in 0..6 {
            assert!(ss.dma_slave(i).is_some(), "Should have DMA S2MM channel {}", i);
            assert!(ss.dma_master(i).is_some(), "Should have DMA MM2S channel {}", i);
        }

        // Verify port counts per AM025:
        // Masters: 0-5=DMA, 6=Tile_Ctrl, 7-10=South(4), 11-16=North(6) = 17 total
        // Slaves: 0-5=DMA, 6=Tile_Ctrl, 7-12=South(6), 13-16=North(4), 17=Trace = 18 total
        assert_eq!(ss.masters.len(), 17);
        assert_eq!(ss.slaves.len(), 18);

        // Verify asymmetric N/S connectivity (key architectural feature)
        let south_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::South)).count();
        let south_slaves = ss.slaves.iter().filter(|p| matches!(p.port_type, PortType::South)).count();
        let north_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::North)).count();
        let north_slaves = ss.slaves.iter().filter(|p| matches!(p.port_type, PortType::North)).count();

        assert_eq!(south_masters, 4, "MemTile should have 4 South masters");
        assert_eq!(south_slaves, 6, "MemTile should have 6 South slaves");
        assert_eq!(north_masters, 6, "MemTile should have 6 North masters");
        assert_eq!(north_slaves, 4, "MemTile should have 4 North slaves");
    }

    #[test]
    fn test_stream_switch_shim() {
        let ss = StreamSwitch::new_shim_tile(0);

        // Shim should be at row 0
        assert_eq!(ss.row, 0);

        // Verify port counts per AM025 PL_MODULE:
        // Masters: 0=Ctrl, 1=FIFO, 2-7=South(6), 8-11=West(4), 12-17=North(6), 18-21=East(4) = 22 total
        // Slaves: 0=Ctrl, 1=FIFO, 2-9=South(8), 10-13=West(4), 14-17=North(4), 18-21=East(4), 22=Trace = 23 total
        assert_eq!(ss.masters.len(), 22, "Shim should have 22 master ports");
        assert_eq!(ss.slaves.len(), 23, "Shim should have 23 slave ports");

        // Verify 6 North masters for connecting to MemTile
        let north_masters = ss.masters.iter().filter(|p| matches!(p.port_type, PortType::North)).count();
        assert_eq!(north_masters, 6, "Shim should have 6 North masters");
    }

    #[test]
    fn test_route_configuration() {
        let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

        assert!(!port.enabled);

        port.set_route(1, 2, 3);
        assert!(port.enabled);
        assert_eq!(port.route_to, Some((1, 2, 3)));

        port.clear_route();
        assert!(!port.enabled);
        assert!(port.route_to.is_none());
    }

    #[test]
    fn test_stream_packet() {
        let pkt = StreamPacket::new(0xCAFEBABE, 0, 1, 2, 1, 2, 3);

        assert_eq!(pkt.data, 0xCAFEBABE);
        assert_eq!(pkt.src_col, 0);
        assert_eq!(pkt.dest_col, 1);
        assert!(!pkt.is_last);

        let pkt_last = pkt.with_last();
        assert!(pkt_last.is_last);
    }

    #[test]
    fn test_local_route() {
        let route = LocalRoute::new(0, 1);
        assert_eq!(route.slave_idx, 0);
        assert_eq!(route.master_idx, 1);
        assert!(route.enabled);
    }

    #[test]
    fn test_configure_local_route() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        assert_eq!(ss.local_route_count(), 0);

        // Configure route from slave 0 to master 0
        ss.configure_local_route(0, 0);
        assert_eq!(ss.local_route_count(), 1);

        // Duplicate route should not be added
        ss.configure_local_route(0, 0);
        assert_eq!(ss.local_route_count(), 1);

        // Different route should be added
        ss.configure_local_route(1, 1);
        assert_eq!(ss.local_route_count(), 2);
    }

    #[test]
    fn test_switch_step_basic() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Configure route from slave 0 to master 0
        ss.configure_local_route(0, 0);

        // Put data in slave port
        ss.slaves[0].push(0xDEADBEEF);
        assert!(ss.slaves[0].has_data());
        assert!(!ss.masters[0].has_data());

        // Step should forward the data
        let forwarded = ss.step();
        assert_eq!(forwarded, 1);

        // Data should now be in master port
        assert!(!ss.slaves[0].has_data());
        assert!(ss.masters[0].has_data());
        assert_eq!(ss.masters[0].peek(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_switch_step_multiple_routes() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Configure two routes
        ss.configure_local_route(0, 0);
        ss.configure_local_route(1, 1);

        // Put data in both slave ports
        ss.slaves[0].push(0x11111111);
        ss.slaves[1].push(0x22222222);

        // Step should forward both
        let forwarded = ss.step();
        assert_eq!(forwarded, 2);

        assert_eq!(ss.masters[0].pop(), Some(0x11111111));
        assert_eq!(ss.masters[1].pop(), Some(0x22222222));
    }

    #[test]
    fn test_switch_step_backpressure() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);
        ss.configure_local_route(0, 0);

        // Fill the master port's FIFO
        while ss.masters[0].can_accept() {
            ss.masters[0].push(0x99999999);
        }

        // Put data in slave
        ss.slaves[0].push(0xDEADBEEF);

        // Step should not forward (backpressure)
        let forwarded = ss.step();
        assert_eq!(forwarded, 0);

        // Data still in slave
        assert!(ss.slaves[0].has_data());
    }

    #[test]
    fn test_switch_step_no_data() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);
        ss.configure_local_route(0, 0);

        // No data in slave - step does nothing
        let forwarded = ss.step();
        assert_eq!(forwarded, 0);
    }

    #[test]
    fn test_has_pending_local() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);
        ss.configure_local_route(0, 0);

        assert!(!ss.has_pending_local());

        ss.slaves[0].push(0x12345678);
        assert!(ss.has_pending_local());

        ss.step();
        assert!(!ss.has_pending_local());
    }

    #[test]
    fn test_remove_local_route() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        ss.configure_local_route(0, 0);
        ss.configure_local_route(1, 1);
        assert_eq!(ss.local_route_count(), 2);

        ss.remove_local_route(0, 0);
        assert_eq!(ss.local_route_count(), 1);

        ss.clear_local_routes();
        assert_eq!(ss.local_route_count(), 0);
    }

    // ========================================================================
    // Packet Header Tests
    // ========================================================================

    #[test]
    fn test_packet_header_new() {
        let header = PacketHeader::new(5, 2, 3);
        assert_eq!(header.stream_id, 5);
        assert_eq!(header.src_col, 2);
        assert_eq!(header.src_row, 3);
        assert_eq!(header.packet_type, PacketType::Data);
    }

    #[test]
    fn test_packet_header_with_type() {
        let header = PacketHeader::new(10, 1, 2).with_type(PacketType::Control);
        assert_eq!(header.stream_id, 10);
        assert_eq!(header.packet_type, PacketType::Control);
    }

    #[test]
    fn test_packet_header_encode_decode() {
        let original = PacketHeader::new(7, 3, 4);
        let encoded = original.encode();
        let (decoded, parity_ok) = PacketHeader::decode(encoded);

        assert!(parity_ok, "Parity check should pass");
        assert_eq!(decoded.stream_id, original.stream_id);
        assert_eq!(decoded.src_col, original.src_col);
        assert_eq!(decoded.src_row, original.src_row);
        assert_eq!(decoded.packet_type, original.packet_type);
    }

    #[test]
    fn test_packet_header_encode_decode_all_types() {
        for ptype in [
            PacketType::Data,
            PacketType::Control,
            PacketType::Config,
            PacketType::Trace,
        ] {
            let original = PacketHeader::new(15, 5, 6).with_type(ptype);
            let encoded = original.encode();
            let (decoded, parity_ok) = PacketHeader::decode(encoded);

            assert!(parity_ok);
            assert_eq!(decoded.packet_type, ptype);
        }
    }

    #[test]
    fn test_packet_header_field_masks() {
        // Test with maximum values for each field
        let header = PacketHeader {
            stream_id: 0x1F,   // 5 bits max
            packet_type: PacketType::Reserved,
            src_row: 0x1F,     // 5 bits max
            src_col: 0x7F,     // 7 bits max
        };

        let encoded = header.encode();
        let (decoded, _) = PacketHeader::decode(encoded);

        assert_eq!(decoded.stream_id, 0x1F);
        assert_eq!(decoded.src_row, 0x1F);
        assert_eq!(decoded.src_col, 0x7F);
    }

    #[test]
    fn test_packet_header_parity_error() {
        let header = PacketHeader::new(5, 2, 3);
        let mut encoded = header.encode();

        // Flip a bit to corrupt parity
        encoded ^= 0x100;

        let (_, parity_ok) = PacketHeader::decode(encoded);
        assert!(!parity_ok, "Parity should fail after bit flip");
    }

    // ========================================================================
    // Packet Switch Tests
    // ========================================================================

    #[test]
    fn test_packet_switch_new() {
        let ps = PacketSwitch::new();
        assert_eq!(ps.route_count(), 0);
        assert!(!ps.in_packet());
    }

    #[test]
    fn test_packet_switch_add_route() {
        let mut ps = PacketSwitch::new();

        ps.add_route(5, 0);
        assert_eq!(ps.route_count(), 1);

        // Adding same stream ID adds to existing route
        ps.add_route(5, 1);
        assert_eq!(ps.route_count(), 1);

        // Lookup should return both ports
        let dests = ps.lookup(5).unwrap();
        assert_eq!(dests.len(), 2);
        assert!(dests.contains(&0));
        assert!(dests.contains(&1));
    }

    #[test]
    fn test_packet_switch_multicast_route() {
        let mut ps = PacketSwitch::new();

        ps.add_multicast_route(10, vec![0, 1, 2]);
        assert_eq!(ps.route_count(), 1);

        let dests = ps.lookup(10).unwrap();
        assert_eq!(dests, &[0, 1, 2]);
    }

    #[test]
    fn test_packet_switch_lookup_not_found() {
        let ps = PacketSwitch::new();
        assert!(ps.lookup(5).is_none());
    }

    #[test]
    fn test_packet_switch_remove_route() {
        let mut ps = PacketSwitch::new();

        ps.add_route(5, 0);
        ps.add_route(10, 1);
        assert_eq!(ps.route_count(), 2);

        ps.remove_route(5);
        assert_eq!(ps.route_count(), 1);
        assert!(ps.lookup(5).is_none());
        assert!(ps.lookup(10).is_some());
    }

    #[test]
    fn test_packet_switch_process_header() {
        let mut ps = PacketSwitch::new();
        ps.add_route(5, 2);

        let header = PacketHeader::new(5, 1, 3);
        let encoded = header.encode();

        let result = ps.process_header(encoded);
        assert!(result.is_some());

        let (decoded, dests) = result.unwrap();
        assert_eq!(decoded.stream_id, 5);
        assert_eq!(dests, vec![2]);

        // Should be in packet now
        assert!(ps.in_packet());
        assert!(ps.has_arb_delay());
    }

    #[test]
    fn test_packet_switch_process_header_no_route() {
        let mut ps = PacketSwitch::new();
        // No routes configured

        let header = PacketHeader::new(5, 1, 3);
        let encoded = header.encode();

        let result = ps.process_header(encoded);
        assert!(result.is_none(), "Should return None for unknown stream ID");
    }

    #[test]
    fn test_packet_switch_arb_delay() {
        let mut ps = PacketSwitch::new();
        ps.add_route(5, 0);

        let header = PacketHeader::new(5, 1, 2);
        ps.process_header(header.encode());

        // With 1-cycle overhead, first tick should complete
        assert!(ps.has_arb_delay());

        // Tick until complete (may be immediate if overhead is 1)
        while ps.has_arb_delay() {
            ps.tick_arb_delay();
        }
        assert!(!ps.has_arb_delay());
    }

    #[test]
    fn test_packet_switch_complete_packet() {
        let mut ps = PacketSwitch::new();
        ps.add_route(5, 0);

        let header = PacketHeader::new(5, 1, 2);
        ps.process_header(header.encode());

        // Count some data words
        ps.count_data_word();
        ps.count_data_word();
        ps.count_data_word();

        // Complete packet
        let result = ps.complete_packet();
        assert!(result.is_some());

        let (completed_header, word_count) = result.unwrap();
        assert_eq!(completed_header.stream_id, 5);
        assert_eq!(word_count, 3);

        // No longer in packet
        assert!(!ps.in_packet());
    }

    #[test]
    fn test_packet_type_from_u8() {
        assert_eq!(PacketType::from_u8(0), PacketType::Data);
        assert_eq!(PacketType::from_u8(1), PacketType::Control);
        assert_eq!(PacketType::from_u8(2), PacketType::Config);
        assert_eq!(PacketType::from_u8(3), PacketType::Trace);
        assert_eq!(PacketType::from_u8(4), PacketType::Reserved);
        assert_eq!(PacketType::from_u8(7), PacketType::Reserved);
    }
}
