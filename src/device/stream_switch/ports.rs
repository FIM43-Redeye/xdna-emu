//! Port types and stream port implementation.

use xdna_archspec::aie2::timing as arch_timing;
use std::collections::VecDeque;

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
        use xdna_archspec::aie2::port_type;
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

    /// Snake-case kind string for use in the route-graph JSON.
    ///
    /// These strings are part of the cross-language contract with `dump_model.py`;
    /// they are derived from the enum variants above — adding a new variant here
    /// requires a matching branch below (the exhaustive match enforces this).
    pub fn as_kind_str(&self) -> &'static str {
        match self {
            PortType::North => "north",
            PortType::South => "south",
            PortType::East => "east",
            PortType::West => "west",
            PortType::Dma(_) => "dma",
            PortType::Core => "core",
            PortType::TileCtrl => "tile_ctrl",
            PortType::Cascade => "cascade",
            PortType::Fifo => "fifo",
            PortType::Trace => "trace",
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
    /// FIFO buffer (data waiting to be sent/received).
    /// `VecDeque` because FIFO pop is the hot path and `Vec::remove(0)` is O(n).
    pub fifo: VecDeque<u32>,
    /// TLAST flags parallel to `fifo` -- true if word at same index has TLAST set.
    /// Kept in lock-step: push/pop/clear update both deques together.
    tlast_flags: VecDeque<bool>,
    /// FIFO capacity
    pub fifo_capacity: usize,
    /// Raw configuration register value (from CDO)
    pub config: u32,
    /// Was this port active during the current routing cycle?
    /// Set true when data is pushed or when the FIFO was non-empty at the
    /// start of routing. Drives adaptive clock gating (a port holding
    /// buffered data keeps the stream-switch clock awake, matching silicon's
    /// idle detector), NOT the PORT_RUNNING trace -- see `cycle_beat`.
    pub cycle_active: bool,
    /// Did a data beat actually cross this port during the current routing
    /// cycle? Set true only when a word is pushed or popped this cycle; reset
    /// (with no `has_data()` seed) at `begin_routing_cycle`. This is the
    /// HW-faithful "port running" signal: a stream-switch port asserts
    /// PORT_RUNNING only on cycles where a beat crosses it, not merely while
    /// it holds residual buffered data (that is idle-with-data, or stalled
    /// under backpressure). `cycle_active` conflates the two and over-marks
    /// S2MM receive ports as continuously running; `cycle_beat` does not.
    pub cycle_beat: bool,
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
        // FIFO depth per AM020 Ch2.  External masters drive the inter-tile
        // wires and have a deeper FIFO (4) than local masters (2) which feed
        // the DMA / core / trace / control consumers on the same tile.
        // All slaves share the same 4-deep depth.
        let fifo_capacity = match direction {
            PortDirection::Master if port_type.is_external() => {
                arch_timing::STREAM_EXTERNAL_MASTER_FIFO_DEPTH as usize
            }
            PortDirection::Master => arch_timing::STREAM_LOCAL_MASTER_FIFO_DEPTH as usize,
            PortDirection::Slave => arch_timing::STREAM_LOCAL_SLAVE_FIFO_DEPTH as usize,
        };

        Self {
            index,
            direction,
            port_type,
            fifo: VecDeque::with_capacity(fifo_capacity),
            tlast_flags: VecDeque::with_capacity(fifo_capacity),
            fifo_capacity,
            config: 0,
            route_to: None,
            enabled: false,
            packet_enable: false,
            cycle_active: false,
            cycle_beat: false,
            cycle_stalled: false,
            cycle_tlast: false,
        }
    }

    /// Check if FIFO has data.
    #[inline]
    pub fn has_data(&self) -> bool {
        !self.fifo.is_empty()
    }

    /// Check if FIFO can accept more data.
    #[inline]
    pub fn can_accept(&self) -> bool {
        self.fifo.len() < self.fifo_capacity
    }

    /// Check if FIFO is full (backpressure).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.fifo.len() >= self.fifo_capacity
    }

    /// Push data into FIFO (returns false if full). TLAST defaults to false.
    pub fn push(&mut self, data: u32) -> bool {
        if self.can_accept() {
            self.fifo.push_back(data);
            self.tlast_flags.push_back(false);
            self.cycle_active = true;
            if self.beats_on_push() {
                self.cycle_beat = true;
            }
            true
        } else {
            false
        }
    }

    /// Push data with explicit TLAST flag (returns false if full).
    pub fn push_with_tlast(&mut self, data: u32, tlast: bool) -> bool {
        if self.can_accept() {
            self.fifo.push_back(data);
            self.tlast_flags.push_back(tlast);
            self.cycle_active = true;
            if self.beats_on_push() {
                self.cycle_beat = true;
            }
            if tlast {
                self.cycle_tlast = true;
            }
            true
        } else {
            false
        }
    }

    /// Pop data from FIFO.
    ///
    /// A pop sets `cycle_beat` (the PORT_RUNNING signal) only for a MASTER
    /// port, whose pop drives its external downstream AXI interface. A SLAVE
    /// port's pop is the internal crossbar draining it toward a master and is
    /// NOT an external handshake -- see `beats_on_pop`. It does NOT set
    /// `cycle_active`: that flag intentionally tracks buffered-data presence for
    /// clock gating and is seeded from `has_data()` at cycle start.
    pub fn pop(&mut self) -> Option<u32> {
        if self.fifo.is_empty() {
            None
        } else {
            if self.beats_on_pop() {
                self.cycle_beat = true;
            }
            self.tlast_flags.pop_front();
            self.fifo.pop_front()
        }
    }

    /// Pop data with its TLAST flag from FIFO.
    pub fn pop_with_tlast(&mut self) -> Option<(u32, bool)> {
        if self.fifo.is_empty() {
            None
        } else {
            if self.beats_on_pop() {
                self.cycle_beat = true;
            }
            let tlast = self.tlast_flags.pop_front().unwrap_or(false);
            if tlast {
                self.cycle_tlast = true;
            }
            self.fifo.pop_front().map(|d| (d, tlast))
        }
    }

    /// Does a push cross this port's external AXI interface (so it should
    /// assert PORT_RUNNING)? True for a SLAVE port: its external interface is
    /// the input, filled by an upstream push. A master's push comes from the
    /// internal crossbar and is not externally visible.
    #[inline]
    fn beats_on_push(&self) -> bool {
        matches!(self.direction, PortDirection::Slave)
    }

    /// Does a pop cross this port's external AXI interface? True for a MASTER
    /// port: its external interface is the output, drained by a downstream pop.
    /// A slave's pop is the internal crossbar draining it and is not externally
    /// visible. PORT_RUNNING watches exactly one external interface per port
    /// (the master/slave bit of `Stream_Switch_Event_Port_Selection`), so a
    /// relayed word beats exactly once per port regardless of FIFO buffering.
    #[inline]
    fn beats_on_pop(&self) -> bool {
        matches!(self.direction, PortDirection::Master)
    }

    /// Peek at front of FIFO without removing.
    #[inline]
    pub fn peek(&self) -> Option<u32> {
        self.fifo.front().copied()
    }

    /// Peek at front TLAST flag without removing.
    #[inline]
    pub fn peek_tlast(&self) -> Option<bool> {
        self.tlast_flags.front().copied()
    }

    /// Get number of items in FIFO.
    #[inline]
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
