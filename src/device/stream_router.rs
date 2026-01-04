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

use crate::device::aie2_spec::{
    ROUTE_LATENCY_LOCAL_TO_LOCAL,
    ROUTE_LATENCY_LOCAL_TO_EXTERNAL,
    ROUTE_LATENCY_EXTERNAL_TO_LOCAL,
    ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL,
    ROUTE_LATENCY_PER_HOP,
};

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

/// Port type for routing latency calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortLocation {
    /// Local port (core, DMA) within the tile
    Local,
    /// External port (north, south, east, west) to neighboring tile
    External,
}

impl PortLocation {
    /// Determine port location from port index.
    /// Ports 0-3 are typically DMA/core (local), 4-7 are directional (external).
    pub fn from_port_index(port: u8) -> Self {
        if port < 4 {
            PortLocation::Local
        } else {
            PortLocation::External
        }
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
    /// Cached hop count (Manhattan distance)
    pub hop_count: u8,
    /// Cached latency in cycles
    pub latency_cycles: u8,
}

/// Calculate routing latency based on source/destination port types and hop count.
pub fn calculate_route_latency(src: &PortId, dest: &PortId) -> (u8, u8) {
    // Calculate hop count (Manhattan distance)
    let col_diff = (src.col as i16 - dest.col as i16).unsigned_abs() as u8;
    let row_diff = (src.row as i16 - dest.row as i16).unsigned_abs() as u8;
    let hop_count = col_diff + row_diff;

    // Determine source and destination port locations
    let src_loc = PortLocation::from_port_index(src.port);
    let dest_loc = PortLocation::from_port_index(dest.port);

    // Base latency for the first hop (at source tile switch)
    let first_hop_latency = match (src_loc, dest_loc) {
        (PortLocation::Local, PortLocation::Local) => ROUTE_LATENCY_LOCAL_TO_LOCAL,
        (PortLocation::Local, PortLocation::External) => ROUTE_LATENCY_LOCAL_TO_EXTERNAL,
        (PortLocation::External, PortLocation::Local) => ROUTE_LATENCY_EXTERNAL_TO_LOCAL,
        (PortLocation::External, PortLocation::External) => ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL,
    };

    // If same tile, just return the base latency
    if hop_count == 0 {
        return (0, first_hop_latency);
    }

    // For multi-hop routes, add per-hop latency for intermediate tiles
    // Each intermediate tile adds external-to-external latency
    let intermediate_latency = if hop_count > 1 {
        (hop_count - 1) * ROUTE_LATENCY_PER_HOP
    } else {
        0
    };

    // Total latency = first hop + intermediate hops
    let total_latency = first_hop_latency + intermediate_latency;

    (hop_count, total_latency)
}

/// Data in flight between tiles.
#[derive(Debug, Clone)]
pub struct InFlightTransfer {
    /// The data word being transferred
    pub word: StreamWord,
    /// Route index this transfer follows
    pub route_idx: usize,
    /// Cycle when transfer will arrive at destination
    pub arrival_cycle: u64,
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
/// Supports both instant routing (for fast simulation) and cycle-accurate
/// routing with proper latency modeling.
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

    /// Data in flight (cycle-accurate mode only)
    in_flight: Vec<InFlightTransfer>,

    /// Current cycle (for cycle-accurate mode)
    current_cycle: u64,

    /// Enable cycle-accurate routing latency
    cycle_accurate: bool,

    /// FIFO depth (reserved for future use)
    #[allow(dead_code)]
    fifo_depth: usize,

    /// Ports per tile
    ports_per_tile: usize,
}

impl StreamRouter {
    /// Create a new stream router for the given array dimensions.
    /// Uses instant routing by default (no latency modeling).
    pub fn new(cols: usize, rows: usize) -> Self {
        // MemTile has 16 masters and 18 slaves, so we need at least 18 ports per tile
        let ports_per_tile = 18;
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
            in_flight: Vec::new(),
            current_cycle: 0,
            cycle_accurate: false,
            fifo_depth,
            ports_per_tile,
        }
    }

    /// Create a new stream router with cycle-accurate latency modeling.
    pub fn new_cycle_accurate(cols: usize, rows: usize) -> Self {
        let mut router = Self::new(cols, rows);
        router.cycle_accurate = true;
        router
    }

    /// Enable or disable cycle-accurate routing latency.
    pub fn set_cycle_accurate(&mut self, enabled: bool) {
        self.cycle_accurate = enabled;
    }

    /// Check if cycle-accurate mode is enabled.
    pub fn is_cycle_accurate(&self) -> bool {
        self.cycle_accurate
    }

    /// Get current cycle.
    pub fn current_cycle(&self) -> u64 {
        self.current_cycle
    }

    /// Get number of in-flight transfers.
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Get FIFO index for a port.
    fn fifo_index(&self, col: u8, row: u8, port: u8) -> usize {
        let tile_idx = (row as usize) * self.cols + (col as usize);
        tile_idx * self.ports_per_tile + (port as usize)
    }

    /// Configure a route from source to destination.
    /// Automatically calculates hop count and latency based on tile positions.
    pub fn add_route(&mut self, src_col: u8, src_row: u8, src_port: u8,
                     dest_col: u8, dest_row: u8, dest_port: u8) {
        let src = PortId::master(src_col, src_row, src_port);
        let dest = PortId::slave(dest_col, dest_row, dest_port);
        let (hop_count, latency_cycles) = calculate_route_latency(&src, &dest);

        let route = Route {
            src,
            dest,
            enabled: true,
            hop_count,
            latency_cycles,
        };
        self.routes.push(route);
    }

    /// Get the latency for a specific route index.
    pub fn route_latency(&self, route_idx: usize) -> Option<u8> {
        self.routes.get(route_idx).map(|r| r.latency_cycles)
    }

    /// Get the hop count for a specific route index.
    pub fn route_hop_count(&self, route_idx: usize) -> Option<u8> {
        self.routes.get(route_idx).map(|r| r.hop_count)
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
    /// In instant mode, data moves immediately from source to destination.
    /// In cycle-accurate mode, data is queued as in-flight with proper latency.
    ///
    /// Returns true if any data was moved or is still in flight.
    pub fn step(&mut self) -> bool {
        if self.cycle_accurate {
            self.step_cycle_accurate()
        } else {
            self.step_instant()
        }
    }

    /// Step with instant routing (no latency).
    fn step_instant(&mut self) -> bool {
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
                        log::debug!("StreamRouter: ({},{},{}) -> ({},{},{}) data=0x{:08X}",
                            route.src.col, route.src.row, route.src.port,
                            route.dest.col, route.dest.row, route.dest.port,
                            word.data);
                        self.input_fifos[dest_idx].push(word);
                        any_moved = true;
                    }
                }
        }

        any_moved
    }

    /// Step with cycle-accurate routing latency.
    ///
    /// The cycle semantics are:
    /// - current_cycle represents the cycle we're about to execute
    /// - Data launched at cycle N with latency L arrives at cycle N + L
    /// - After step(), current_cycle has been incremented
    fn step_cycle_accurate(&mut self) -> bool {
        let mut any_activity = false;

        // 1. Advance to current cycle (time progresses)
        self.current_cycle += 1;

        // 2. Deliver any in-flight data that has arrived
        let current = self.current_cycle;
        let mut arrived = Vec::new();
        self.in_flight.retain(|transfer| {
            if transfer.arrival_cycle <= current {
                arrived.push(transfer.clone());
                false // Remove from in_flight
            } else {
                true // Keep in in_flight
            }
        });

        for transfer in arrived {
            if let Some(route) = self.routes.get(transfer.route_idx) {
                let dest_idx = self.fifo_index(route.dest.col, route.dest.row, route.dest.port);
                if dest_idx < self.input_fifos.len() {
                    // Deliver to destination FIFO
                    self.input_fifos[dest_idx].push(transfer.word);
                    any_activity = true;
                }
            }
        }

        // 3. Launch new data from output FIFOs
        for (route_idx, route) in self.routes.iter().enumerate() {
            if !route.enabled {
                continue;
            }

            let src_idx = self.fifo_index(route.src.col, route.src.row, route.src.port);
            let dest_idx = self.fifo_index(route.dest.col, route.dest.row, route.dest.port);

            // Check if we can launch data (output has data)
            if src_idx < self.output_fifos.len() && dest_idx < self.input_fifos.len()
                && !self.output_fifos[src_idx].is_empty() {
                    if let Some(word) = self.output_fifos[src_idx].pop() {
                        // Calculate arrival time: data launched at current cycle
                        // arrives after latency_cycles - 1 more cycles
                        // (because it's already in transit for the current cycle)
                        let arrival_cycle = self.current_cycle + route.latency_cycles as u64 - 1;

                        // Add to in-flight
                        self.in_flight.push(InFlightTransfer {
                            word,
                            route_idx,
                            arrival_cycle,
                        });
                        any_activity = true;
                    }
                }
        }

        any_activity || !self.in_flight.is_empty()
    }

    /// Advance the router by multiple cycles without launching new data.
    /// Useful for draining in-flight transfers.
    pub fn advance_cycles(&mut self, cycles: u64) {
        if !self.cycle_accurate {
            return;
        }

        self.current_cycle += cycles;

        // Deliver any arrived data
        let current = self.current_cycle;
        let mut arrived = Vec::new();
        self.in_flight.retain(|transfer| {
            if transfer.arrival_cycle <= current {
                arrived.push(transfer.clone());
                false
            } else {
                true
            }
        });

        for transfer in arrived {
            if let Some(route) = self.routes.get(transfer.route_idx) {
                let dest_idx = self.fifo_index(route.dest.col, route.dest.row, route.dest.port);
                if dest_idx < self.input_fifos.len() {
                    self.input_fifos[dest_idx].push(transfer.word);
                }
            }
        }
    }

    /// Check if there's any data in flight.
    pub fn has_in_flight(&self) -> bool {
        !self.in_flight.is_empty()
    }

    /// Get the earliest arrival cycle for in-flight data.
    pub fn earliest_arrival(&self) -> Option<u64> {
        self.in_flight.iter().map(|t| t.arrival_cycle).min()
    }

    /// Get statistics about the router state.
    pub fn stats(&self) -> RouterStats {
        let output_words: usize = self.output_fifos.iter().map(|f| f.len()).sum();
        let input_words: usize = self.input_fifos.iter().map(|f| f.len()).sum();

        RouterStats {
            active_routes: self.routes.iter().filter(|r| r.enabled).count(),
            output_buffered: output_words,
            input_buffered: input_words,
            in_flight: self.in_flight.len(),
            current_cycle: self.current_cycle,
        }
    }

    /// Get all configured routes (for debugging).
    pub fn routes(&self) -> &[Route] {
        &self.routes
    }

    /// Clear all FIFOs, routes, and in-flight data.
    pub fn reset(&mut self) {
        self.routes.clear();
        self.in_flight.clear();
        self.current_cycle = 0;
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
    pub in_flight: usize,
    pub current_cycle: u64,
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

    // ========================================================================
    // Routing Latency Tests
    // ========================================================================

    #[test]
    fn test_port_location() {
        // Ports 0-3 are local (DMA/core)
        assert_eq!(PortLocation::from_port_index(0), PortLocation::Local);
        assert_eq!(PortLocation::from_port_index(1), PortLocation::Local);
        assert_eq!(PortLocation::from_port_index(2), PortLocation::Local);
        assert_eq!(PortLocation::from_port_index(3), PortLocation::Local);

        // Ports 4-7 are external (directional)
        assert_eq!(PortLocation::from_port_index(4), PortLocation::External);
        assert_eq!(PortLocation::from_port_index(5), PortLocation::External);
        assert_eq!(PortLocation::from_port_index(6), PortLocation::External);
        assert_eq!(PortLocation::from_port_index(7), PortLocation::External);
    }

    #[test]
    fn test_calculate_route_latency_same_tile() {
        // Same tile, local to local: 3 cycles
        let src = PortId::master(0, 0, 0);
        let dest = PortId::slave(0, 0, 1);
        let (hops, latency) = calculate_route_latency(&src, &dest);
        assert_eq!(hops, 0);
        assert_eq!(latency, ROUTE_LATENCY_LOCAL_TO_LOCAL);
    }

    #[test]
    fn test_calculate_route_latency_same_tile_external() {
        // Same tile, external to external: 4 cycles
        let src = PortId::master(0, 0, 4);
        let dest = PortId::slave(0, 0, 5);
        let (hops, latency) = calculate_route_latency(&src, &dest);
        assert_eq!(hops, 0);
        assert_eq!(latency, ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL);
    }

    #[test]
    fn test_calculate_route_latency_one_hop() {
        // One tile away (same column, adjacent row)
        // Local source, external destination: 4 cycles
        let src = PortId::master(0, 0, 0);
        let dest = PortId::slave(0, 1, 4);
        let (hops, latency) = calculate_route_latency(&src, &dest);
        assert_eq!(hops, 1);
        assert_eq!(latency, ROUTE_LATENCY_LOCAL_TO_EXTERNAL);
    }

    #[test]
    fn test_calculate_route_latency_multi_hop() {
        // Two tiles away (row 0 to row 2)
        // External to external base latency + 1 hop intermediate
        let src = PortId::master(0, 0, 4);
        let dest = PortId::slave(0, 2, 4);
        let (hops, latency) = calculate_route_latency(&src, &dest);
        assert_eq!(hops, 2);
        // 4 (external-to-external first hop) + 4 (intermediate hop) = 8
        assert_eq!(latency, ROUTE_LATENCY_EXTERNAL_TO_EXTERNAL + ROUTE_LATENCY_PER_HOP);
    }

    #[test]
    fn test_calculate_route_latency_manhattan() {
        // Diagonal: 2 columns + 3 rows = 5 hops
        let src = PortId::master(0, 0, 0);
        let dest = PortId::slave(2, 3, 0);
        let (hops, latency) = calculate_route_latency(&src, &dest);
        assert_eq!(hops, 5);
        // 3 (local-to-local first hop) + 4 * 4 (4 intermediate hops) = 19
        assert_eq!(latency, ROUTE_LATENCY_LOCAL_TO_LOCAL + 4 * ROUTE_LATENCY_PER_HOP);
    }

    #[test]
    fn test_cycle_accurate_basic() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        // Route from tile (0,2) port 0 (local) to tile (0,1) port 0 (local)
        // This is 1 hop, local-to-local = 3 cycles + 1 intermediate = 4 cycles?
        // Wait, 1 hop means same column adjacent row - that's local-to-external
        router.add_route(0, 2, 0, 0, 1, 0);

        // Check that latency was calculated
        let latency = router.route_latency(0).unwrap();
        assert!(latency > 0);

        // Write data
        router.write_output(0, 2, 0, StreamWord::new(0xDEAD));

        // Step once - data should be in flight
        assert!(router.step());
        assert!(router.has_in_flight());
        assert!(!router.input_has_data(0, 1, 0));

        // Advance remaining cycles
        let remaining = latency as u64 - 1; // Already stepped once
        for _ in 0..remaining {
            router.step();
        }

        // Data should now be available
        assert!(router.input_has_data(0, 1, 0));
        let word = router.read_input(0, 1, 0).unwrap();
        assert_eq!(word.data, 0xDEAD);
    }

    #[test]
    fn test_cycle_accurate_latency_measurement() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        // Same tile route (0 hops, local-to-local = 3 cycles)
        router.add_route(0, 0, 0, 0, 0, 1);

        let start_cycle = router.current_cycle();
        router.write_output(0, 0, 0, StreamWord::new(0xCAFE));

        // Step until data arrives
        while !router.input_has_data(0, 0, 1) {
            router.step();
        }

        let end_cycle = router.current_cycle();
        // Should be 3 cycles (local-to-local latency)
        assert_eq!(end_cycle - start_cycle, ROUTE_LATENCY_LOCAL_TO_LOCAL as u64);
    }

    #[test]
    fn test_cycle_accurate_multi_hop_latency() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        // Route from (0,0) to (0,3) - 3 hops
        // local-to-local base + 2 intermediate = 3 + 8 = 11 cycles
        router.add_route(0, 0, 0, 0, 3, 0);

        let expected_latency = router.route_latency(0).unwrap();

        let start_cycle = router.current_cycle();
        router.write_output(0, 0, 0, StreamWord::new(0xBEEF));

        // Step until data arrives
        while !router.input_has_data(0, 3, 0) {
            router.step();
        }

        let actual_latency = router.current_cycle() - start_cycle;
        assert_eq!(actual_latency, expected_latency as u64);
    }

    #[test]
    fn test_cycle_accurate_vs_instant() {
        // Same route, different modes
        let mut instant_router = StreamRouter::new(4, 6);
        let mut cycle_router = StreamRouter::new_cycle_accurate(4, 6);

        instant_router.add_route(0, 0, 0, 0, 2, 0);
        cycle_router.add_route(0, 0, 0, 0, 2, 0);

        instant_router.write_output(0, 0, 0, StreamWord::new(0x1111));
        cycle_router.write_output(0, 0, 0, StreamWord::new(0x2222));

        // Instant: data available after 1 step
        instant_router.step();
        assert!(instant_router.input_has_data(0, 2, 0));

        // Cycle-accurate: data in flight after 1 step
        cycle_router.step();
        assert!(!cycle_router.input_has_data(0, 2, 0));
        assert!(cycle_router.has_in_flight());
    }

    #[test]
    fn test_advance_cycles() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        // 2-hop route
        router.add_route(0, 0, 0, 0, 2, 0);
        let latency = router.route_latency(0).unwrap();

        router.write_output(0, 0, 0, StreamWord::new(0xABCD));
        router.step(); // Launch data

        // Advance to just before arrival
        router.advance_cycles(latency as u64 - 2);
        assert!(!router.input_has_data(0, 2, 0));

        // Advance past arrival
        router.advance_cycles(2);
        assert!(router.input_has_data(0, 2, 0));
    }

    #[test]
    fn test_in_flight_count() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        router.add_route(0, 0, 0, 0, 2, 0);

        assert_eq!(router.in_flight_count(), 0);

        router.write_output(0, 0, 0, StreamWord::new(1));
        router.step();
        assert_eq!(router.in_flight_count(), 1);

        router.write_output(0, 0, 0, StreamWord::new(2));
        router.step();
        assert_eq!(router.in_flight_count(), 2);
    }

    #[test]
    fn test_earliest_arrival() {
        let mut router = StreamRouter::new_cycle_accurate(4, 6);

        // Two routes with different latencies
        router.add_route(0, 0, 0, 0, 1, 0); // Short route
        router.add_route(0, 0, 1, 0, 3, 1); // Longer route

        let short_latency = router.route_latency(0).unwrap();
        let long_latency = router.route_latency(1).unwrap();
        assert!(short_latency < long_latency);

        // Launch on long route first
        router.write_output(0, 0, 1, StreamWord::new(1));
        router.step();
        let first_arrival = router.earliest_arrival().unwrap();

        // Launch on short route
        router.write_output(0, 0, 0, StreamWord::new(2));
        router.step();
        let second_arrival = router.earliest_arrival().unwrap();

        // Short route should arrive first
        assert!(second_arrival < first_arrival + long_latency as u64);
    }
}
