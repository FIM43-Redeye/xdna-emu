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
    /// Latency in cycles for local-to-local routing
    pub local_latency: u8,
    /// Latency in cycles for external routing
    pub external_latency: u8,
}

impl StreamSwitch {
    /// Create a new stream switch for a compute tile.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let mut masters = Vec::new();
        let mut slaves = Vec::new();

        // DMA ports (2 S2MM slaves, 2 MM2S masters)
        for i in 0..2 {
            slaves.push(StreamPort::new(i, PortDirection::Slave, PortType::Dma(i)));
        }
        for i in 0..2 {
            masters.push(StreamPort::new(i, PortDirection::Master, PortType::Dma(i + 2)));
        }

        // Directional ports
        masters.push(StreamPort::new(2, PortDirection::Master, PortType::North));
        masters.push(StreamPort::new(3, PortDirection::Master, PortType::South));
        masters.push(StreamPort::new(4, PortDirection::Master, PortType::East));
        masters.push(StreamPort::new(5, PortDirection::Master, PortType::West));

        slaves.push(StreamPort::new(2, PortDirection::Slave, PortType::North));
        slaves.push(StreamPort::new(3, PortDirection::Slave, PortType::South));
        slaves.push(StreamPort::new(4, PortDirection::Slave, PortType::East));
        slaves.push(StreamPort::new(5, PortDirection::Slave, PortType::West));

        // Core ports
        masters.push(StreamPort::new(6, PortDirection::Master, PortType::Core));
        slaves.push(StreamPort::new(6, PortDirection::Slave, PortType::Core));

        Self {
            col,
            row,
            masters,
            slaves,
            local_latency: aie2_spec::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: aie2_spec::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
        }
    }

    /// Create a new stream switch for a memory tile.
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let mut masters = Vec::new();
        let mut slaves = Vec::new();

        // DMA ports (6 S2MM slaves, 6 MM2S masters)
        for i in 0..6 {
            slaves.push(StreamPort::new(i, PortDirection::Slave, PortType::Dma(i)));
        }
        for i in 0..6 {
            masters.push(StreamPort::new(i, PortDirection::Master, PortType::Dma(i + 6)));
        }

        // Directional ports
        masters.push(StreamPort::new(6, PortDirection::Master, PortType::North));
        masters.push(StreamPort::new(7, PortDirection::Master, PortType::South));
        slaves.push(StreamPort::new(6, PortDirection::Slave, PortType::North));
        slaves.push(StreamPort::new(7, PortDirection::Slave, PortType::South));

        Self {
            col,
            row,
            masters,
            slaves,
            local_latency: aie2_spec::STREAM_LOCAL_TO_LOCAL_LATENCY,
            external_latency: aie2_spec::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
        }
    }

    /// Create a new stream switch for a shim tile.
    pub fn new_shim_tile(col: u8) -> Self {
        let mut masters = Vec::new();
        let mut slaves = Vec::new();

        // DMA ports for host interface
        for i in 0..2 {
            slaves.push(StreamPort::new(i, PortDirection::Slave, PortType::Dma(i)));
        }
        for i in 0..2 {
            masters.push(StreamPort::new(i, PortDirection::Master, PortType::Dma(i + 2)));
        }

        // North port (to mem tile / compute tiles)
        masters.push(StreamPort::new(2, PortDirection::Master, PortType::North));
        slaves.push(StreamPort::new(2, PortDirection::Slave, PortType::North));

        Self {
            col,
            row: 0,
            masters,
            slaves,
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
    pub fn configure_local_route(&mut self, slave_idx: usize, master_idx: usize) {
        if let Some(master) = self.masters.get_mut(master_idx) {
            master.enabled = true;
        }
        if let Some(slave) = self.slaves.get_mut(slave_idx) {
            slave.enabled = true;
        }
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

        // Should have DMA ports
        assert!(ss.dma_slave(0).is_some());
        assert!(ss.dma_slave(1).is_some());
        assert!(ss.dma_master(2).is_some());
        assert!(ss.dma_master(3).is_some());

        // Should have directional ports
        assert!(ss.masters.len() >= 6);
        assert!(ss.slaves.len() >= 6);
    }

    #[test]
    fn test_stream_switch_mem_tile() {
        let ss = StreamSwitch::new_mem_tile(0, 1);

        // Should have 6 DMA slave ports
        for i in 0..6 {
            assert!(ss.dma_slave(i).is_some());
        }
    }

    #[test]
    fn test_stream_switch_shim() {
        let ss = StreamSwitch::new_shim_tile(0);

        // Shim should be at row 0
        assert_eq!(ss.row, 0);

        // Should have north port for connecting to array
        assert!(ss.masters.iter().any(|p| matches!(p.port_type, PortType::North)));
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
}
