//! Port types and stream port implementation.

use crate::arch::timing as arch_timing;

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
    /// Connects to tile control packet handler
    TileCtrl,
    /// Connects to cascade interface
    Cascade,
    /// Connects to FIFO
    Fifo,
    /// Connects to trace/debug interface
    Trace,
}

impl PortType {
    /// Convert from the u8 encoding used in arch port layouts.
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
        use crate::arch::port_type;
        match encoded {
            port_type::CORE => PortType::Core,
            port_type::FIFO => PortType::Fifo,
            port_type::TRACE => PortType::Trace,
            port_type::CTRL => PortType::TileCtrl,
            n if n >= port_type::DMA_BASE => PortType::Dma(n - port_type::DMA_BASE),
            n if n >= port_type::WEST_BASE => PortType::West,
            n if n >= port_type::EAST_BASE => PortType::East,
            n if n >= port_type::SOUTH_BASE => PortType::South,
            n if n >= port_type::NORTH_BASE => PortType::North,
            _ => PortType::Core, // Fallback
        }
    }

    /// Whether this port connects to another tile (external/inter-tile).
    pub fn is_external(&self) -> bool {
        matches!(self, PortType::North | PortType::South | PortType::East | PortType::West)
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
    /// TLAST flags parallel to `fifo` -- true if word at same index has TLAST set.
    /// Kept in lock-step: push/pop/clear update both vectors together.
    tlast_flags: Vec<bool>,
    /// FIFO capacity
    pub fifo_capacity: usize,
    /// Raw configuration register value (from CDO)
    pub config: u32,
    /// Was this port active during the current routing cycle?
    /// Set true when data is pushed or when the FIFO was non-empty at the
    /// start of routing. Used by PORT_RUNNING trace events -- data flows
    /// through the entire stream switch within a single `route_streams()`
    /// call, so checking `has_data()` between calls always sees empty FIFOs.
    pub cycle_active: bool,
    /// Was this port stalled during the current routing cycle?
    /// Set true when a slave has data but cannot forward it due to arbiter
    /// contention or master backpressure. Used by PORT_STALLED trace events.
    pub cycle_stalled: bool,
    /// Was a TLAST seen on this port during the current routing cycle?
    /// Set true when a word with TLAST=true is pushed or popped.
    /// Used by PORT_TLAST trace events.
    pub cycle_tlast: bool,
    /// Connected destination (for routing)
    pub route_to: Option<(u8, u8, u8)>, // (col, row, port_index)
    /// Port is enabled
    pub enabled: bool,
    /// Port is in packet-switched mode (bit 30 of slave config register).
    /// When true, step_packet_routes() processes this port's data as packets.
    /// When false, data flows through circuit routes only.
    pub packet_enable: bool,
}

impl StreamPort {
    /// Create a new stream port.
    pub fn new(index: u8, direction: PortDirection, port_type: PortType) -> Self {
        let fifo_capacity = match direction {
            PortDirection::Master => arch_timing::STREAM_LOCAL_MASTER_FIFO_DEPTH as usize,
            PortDirection::Slave => arch_timing::STREAM_LOCAL_SLAVE_FIFO_DEPTH as usize,
        };

        Self {
            index,
            direction,
            port_type,
            fifo: Vec::with_capacity(fifo_capacity),
            tlast_flags: Vec::with_capacity(fifo_capacity),
            fifo_capacity,
            config: 0,
            route_to: None,
            enabled: false,
            packet_enable: false,
            cycle_active: false,
            cycle_stalled: false,
            cycle_tlast: false,
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

    /// Push data into FIFO (returns false if full). TLAST defaults to false.
    pub fn push(&mut self, data: u32) -> bool {
        if self.can_accept() {
            self.fifo.push(data);
            self.tlast_flags.push(false);
            self.cycle_active = true;
            true
        } else {
            false
        }
    }

    /// Push data with explicit TLAST flag (returns false if full).
    pub fn push_with_tlast(&mut self, data: u32, tlast: bool) -> bool {
        if self.can_accept() {
            self.fifo.push(data);
            self.tlast_flags.push(tlast);
            self.cycle_active = true;
            if tlast {
                self.cycle_tlast = true;
            }
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
            self.tlast_flags.remove(0);
            Some(self.fifo.remove(0))
        }
    }

    /// Pop data with its TLAST flag from FIFO.
    pub fn pop_with_tlast(&mut self) -> Option<(u32, bool)> {
        if self.fifo.is_empty() {
            None
        } else {
            let tlast = self.tlast_flags.remove(0);
            if tlast {
                self.cycle_tlast = true;
            }
            Some((self.fifo.remove(0), tlast))
        }
    }

    /// Peek at front of FIFO without removing.
    pub fn peek(&self) -> Option<u32> {
        self.fifo.first().copied()
    }

    /// Peek at front TLAST flag without removing.
    pub fn peek_tlast(&self) -> Option<bool> {
        self.tlast_flags.first().copied()
    }

    /// Get number of items in FIFO.
    pub fn fifo_level(&self) -> usize {
        self.fifo.len()
    }

    /// Clear the FIFO.
    pub fn clear(&mut self) {
        self.fifo.clear();
        self.tlast_flags.clear();
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
