//! Global stream router for tile-to-tile data flow.
//!
//! The stream router handles data movement between tiles through the
//! stream switch fabric. It provides:
//!
//! - Circuit-switched routing (configured paths)
//! - FIFO buffering at each port
//! - Backpressure handling
//!
//! # Architecture
//!
//! ```text
//!                    Shim (row 0)
//!                         │
//!                    ┌────┴────┐
//!                    │ MemTile │  (row 1)
//!                    └────┬────┘
//!              ┌──────────┼──────────┐
//!         ┌────┴────┬────┴────┬────┴────┐
//!         │Compute 0│Compute 1│Compute 2│  (row 2+)
//!         └─────────┴─────────┴─────────┘
//! ```
//!
//! Each tile has north/south/east/west connections plus DMA and core ports.

use std::collections::VecDeque;

/// Stream data unit (32-bit word with metadata).
#[derive(Debug, Clone, Copy)]
pub struct StreamWord {
    /// Data word
    pub data: u32,
    /// TLAST - marks end of packet/transfer
    pub tlast: bool,
}

impl StreamWord {
    pub fn new(data: u32) -> Self {
        Self { data, tlast: false }
    }

    pub fn with_tlast(data: u32) -> Self {
        Self { data, tlast: true }
    }
}

/// Port identifier within the stream fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortId {
    pub col: u8,
    pub row: u8,
    pub port: u8,
    pub is_master: bool,
}

impl PortId {
    pub fn master(col: u8, row: u8, port: u8) -> Self {
        Self { col, row, port, is_master: true }
    }

    pub fn slave(col: u8, row: u8, port: u8) -> Self {
        Self { col, row, port, is_master: false }
    }
}

/// A configured route from source to destination.
#[derive(Debug, Clone)]
pub struct Route {
    /// Source port
    pub src: PortId,
    /// Destination port
    pub dest: PortId,
    /// Is route active
    pub enabled: bool,
}

/// FIFO buffer for stream data.
#[derive(Debug, Clone, Default)]
pub struct StreamFifo {
    /// Data queue
    data: VecDeque<StreamWord>,
    /// Maximum capacity
    capacity: usize,
}

impl StreamFifo {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, word: StreamWord) -> bool {
        if self.data.len() < self.capacity {
            self.data.push_back(word);
            true
        } else {
            false // Backpressure
        }
    }

    pub fn pop(&mut self) -> Option<StreamWord> {
        self.data.pop_front()
    }

    pub fn peek(&self) -> Option<&StreamWord> {
        self.data.front()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.data.len() >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }
}

/// Port type for DMA integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaPortType {
    /// S2MM: Stream to Memory (DMA receives from stream)
    S2MM(u8), // channel index
    /// MM2S: Memory to Stream (DMA sends to stream)
    MM2S(u8), // channel index
}

/// Stream router for a tile array.
///
/// Manages data flow between tiles through the stream switch fabric.
pub struct StreamRouter {
    /// Array dimensions
    cols: usize,
    rows: usize,

    /// Configured routes
    routes: Vec<Route>,

    /// Output FIFOs (master ports) - indexed by (col, row, port)
    /// Each tile has up to 8 master ports
    output_fifos: Vec<StreamFifo>,

    /// Input FIFOs (slave ports) - indexed by (col, row, port)
    input_fifos: Vec<StreamFifo>,

    /// FIFO depth (reserved for future use)
    #[allow(dead_code)]
    fifo_depth: usize,

    /// Ports per tile
    ports_per_tile: usize,
}

impl StreamRouter {
    /// Create a new stream router for the given array dimensions.
    pub fn new(cols: usize, rows: usize) -> Self {
        let ports_per_tile = 8; // 8 master + 8 slave ports typical
        let fifo_depth = 4; // Words of buffering

        let num_tiles = cols * rows;
        let num_ports = num_tiles * ports_per_tile;

        let output_fifos = (0..num_ports)
            .map(|_| StreamFifo::new(fifo_depth))
            .collect();
        let input_fifos = (0..num_ports)
            .map(|_| StreamFifo::new(fifo_depth))
            .collect();

        Self {
            cols,
            rows,
            routes: Vec::new(),
            output_fifos,
            input_fifos,
            fifo_depth,
            ports_per_tile,
        }
    }

    /// Get FIFO index for a port.
    fn fifo_index(&self, col: u8, row: u8, port: u8) -> usize {
        let tile_idx = (row as usize) * self.cols + (col as usize);
        tile_idx * self.ports_per_tile + (port as usize)
    }

    /// Configure a route from source to destination.
    pub fn add_route(&mut self, src_col: u8, src_row: u8, src_port: u8,
                     dest_col: u8, dest_row: u8, dest_port: u8) {
        let route = Route {
            src: PortId::master(src_col, src_row, src_port),
            dest: PortId::slave(dest_col, dest_row, dest_port),
            enabled: true,
        };
        self.routes.push(route);
    }

    /// Configure a north-south route (from tile to tile above).
    pub fn add_north_route(&mut self, col: u8, row: u8, src_port: u8, dest_port: u8) {
        if row + 1 < self.rows as u8 {
            self.add_route(col, row, src_port, col, row + 1, dest_port);
        }
    }

    /// Configure a south-north route (from tile to tile below).
    pub fn add_south_route(&mut self, col: u8, row: u8, src_port: u8, dest_port: u8) {
        if row > 0 {
            self.add_route(col, row, src_port, col, row - 1, dest_port);
        }
    }

    /// Write data to an output port (from DMA MM2S).
    pub fn write_output(&mut self, col: u8, row: u8, port: u8, word: StreamWord) -> bool {
        let idx = self.fifo_index(col, row, port);
        if idx < self.output_fifos.len() {
            self.output_fifos[idx].push(word)
        } else {
            false
        }
    }

    /// Read data from an input port (to DMA S2MM).
    pub fn read_input(&mut self, col: u8, row: u8, port: u8) -> Option<StreamWord> {
        let idx = self.fifo_index(col, row, port);
        if idx < self.input_fifos.len() {
            self.input_fifos[idx].pop()
        } else {
            None
        }
    }

    /// Check if input port has data.
    pub fn input_has_data(&self, col: u8, row: u8, port: u8) -> bool {
        let idx = self.fifo_index(col, row, port);
        if idx < self.input_fifos.len() {
            !self.input_fifos[idx].is_empty()
        } else {
            false
        }
    }

    /// Check if output port can accept data.
    pub fn output_can_accept(&self, col: u8, row: u8, port: u8) -> bool {
        let idx = self.fifo_index(col, row, port);
        if idx < self.output_fifos.len() {
            !self.output_fifos[idx].is_full()
        } else {
            false
        }
    }

    /// Step the router: move data along configured routes.
    ///
    /// Returns true if any data was moved.
    pub fn step(&mut self) -> bool {
        let mut any_moved = false;

        // Process each configured route
        for route in &self.routes {
            if !route.enabled {
                continue;
            }

            let src_idx = self.fifo_index(route.src.col, route.src.row, route.src.port);
            let dest_idx = self.fifo_index(route.dest.col, route.dest.row, route.dest.port);

            // Check if we can move data
            if src_idx < self.output_fifos.len() && dest_idx < self.input_fifos.len()
                && !self.output_fifos[src_idx].is_empty() &&
                   !self.input_fifos[dest_idx].is_full() {
                    if let Some(word) = self.output_fifos[src_idx].pop() {
                        self.input_fifos[dest_idx].push(word);
                        any_moved = true;
                    }
                }
        }

        any_moved
    }

    /// Get statistics about the router state.
    pub fn stats(&self) -> RouterStats {
        let output_words: usize = self.output_fifos.iter().map(|f| f.len()).sum();
        let input_words: usize = self.input_fifos.iter().map(|f| f.len()).sum();

        RouterStats {
            active_routes: self.routes.iter().filter(|r| r.enabled).count(),
            output_buffered: output_words,
            input_buffered: input_words,
        }
    }

    /// Clear all FIFOs and routes.
    pub fn reset(&mut self) {
        self.routes.clear();
        for fifo in &mut self.output_fifos {
            fifo.clear();
        }
        for fifo in &mut self.input_fifos {
            fifo.clear();
        }
    }
}

/// Router statistics.
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    pub active_routes: usize,
    pub output_buffered: usize,
    pub input_buffered: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_fifo() {
        let mut fifo = StreamFifo::new(4);

        assert!(fifo.is_empty());
        assert!(!fifo.is_full());

        fifo.push(StreamWord::new(0x1234));
        assert!(!fifo.is_empty());
        assert_eq!(fifo.len(), 1);

        let word = fifo.pop().unwrap();
        assert_eq!(word.data, 0x1234);
        assert!(fifo.is_empty());
    }

    #[test]
    fn test_fifo_backpressure() {
        let mut fifo = StreamFifo::new(2);

        assert!(fifo.push(StreamWord::new(1)));
        assert!(fifo.push(StreamWord::new(2)));
        assert!(!fifo.push(StreamWord::new(3))); // Should fail - full
        assert!(fifo.is_full());
    }

    #[test]
    fn test_router_basic() {
        let mut router = StreamRouter::new(4, 6);

        // Configure route from tile (0,2) port 0 to tile (0,1) port 0
        router.add_route(0, 2, 0, 0, 1, 0);

        // Write data to source
        assert!(router.write_output(0, 2, 0, StreamWord::new(0xDEAD)));
        assert!(router.write_output(0, 2, 0, StreamWord::new(0xBEEF)));

        // Step router
        assert!(router.step());

        // Data should be at destination
        assert!(router.input_has_data(0, 1, 0));
        let word = router.read_input(0, 1, 0).unwrap();
        assert_eq!(word.data, 0xDEAD);
    }

    #[test]
    fn test_router_chain() {
        let mut router = StreamRouter::new(4, 6);

        // Create a chain: (0,0) -> (0,1) -> (0,2)
        router.add_route(0, 0, 0, 0, 1, 0);
        router.add_route(0, 1, 0, 0, 2, 0);

        // Write to start
        router.write_output(0, 0, 0, StreamWord::new(0xCAFE));

        // Step 1: (0,0) -> (0,1)
        router.step();
        assert!(router.input_has_data(0, 1, 0));

        // Move from input FIFO to output FIFO of tile (0,1)
        // (In real system, this would be done by the tile's switch)
        let word = router.read_input(0, 1, 0).unwrap();
        router.write_output(0, 1, 0, word);

        // Step 2: (0,1) -> (0,2)
        router.step();
        assert!(router.input_has_data(0, 2, 0));

        let final_word = router.read_input(0, 2, 0).unwrap();
        assert_eq!(final_word.data, 0xCAFE);
    }

    #[test]
    fn test_router_stats() {
        let mut router = StreamRouter::new(4, 6);

        router.add_route(0, 0, 0, 0, 1, 0);
        router.write_output(0, 0, 0, StreamWord::new(0x1234));

        let stats = router.stats();
        assert_eq!(stats.active_routes, 1);
        assert_eq!(stats.output_buffered, 1);
        assert_eq!(stats.input_buffered, 0);

        router.step();

        let stats = router.stats();
        assert_eq!(stats.output_buffered, 0);
        assert_eq!(stats.input_buffered, 1);
    }
}
