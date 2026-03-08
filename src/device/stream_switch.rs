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
        use crate::arch::port_type;
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

// ============================================================================
// Packet Routing Configuration (AM025 stream switch slave slot registers)
// ============================================================================

/// Per-slave-port packet slot configuration (4 slots per slave port).
///
/// Packet matching: `(incoming_pkt_id & mask) == (slot_pkt_id & mask)`
/// When a match is found, the packet is routed to all master ports on
/// the same arbiter whose `msel_enable` bit matches this slot's `msel`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PacketSlot {
    /// Packet ID to match (bits 28:24, 5 bits)
    pub pkt_id: u8,
    /// ID mask for matching (bits 20:16, 5 bits)
    pub mask: u8,
    /// Slot is enabled (bit 8)
    pub enable: bool,
    /// Master select index (bits 5:4, 2 bits)
    pub msel: u8,
    /// Arbiter number (bits 2:0, 3 bits)
    pub arbiter: u8,
}

impl PacketSlot {
    /// Parse from a 32-bit register value.
    ///
    /// Register layout (from aie-rt xaiemlgbl_params.h):
    /// - Bits 28:24 = ID (packet ID)
    /// - Bits 20:16 = MASK
    /// - Bit 8 = ENABLE
    /// - Bits 5:4 = MSEL
    /// - Bits 2:0 = ARBITOR
    pub fn from_register(value: u32) -> Self {
        Self {
            pkt_id: ((value >> 24) & 0x1F) as u8,
            mask: ((value >> 16) & 0x1F) as u8,
            enable: (value >> 8) & 1 != 0,
            msel: ((value >> 4) & 0x3) as u8,
            arbiter: (value & 0x7) as u8,
        }
    }

    /// Check if an incoming packet ID matches this slot.
    pub fn matches(&self, incoming_pkt_id: u8) -> bool {
        self.enable && ((incoming_pkt_id & self.mask) == (self.pkt_id & self.mask))
    }
}

/// Per-master-port packet configuration.
///
/// When `packet_enable` is true, this master operates in packet mode:
/// it receives data from the arbiter/msel routing system rather than
/// a directly-selected slave.
#[derive(Debug, Clone, Copy, Default)]
pub struct MasterPacketConfig {
    /// Packet switching enabled (bit 30 of master config)
    pub packet_enable: bool,
    /// Drop packet header before forwarding (bit 7 of config field)
    pub drop_header: bool,
    /// Arbiter this master belongs to (bits 2:0 of config field)
    pub arbiter: u8,
    /// Which msel values this master accepts (bits 6:3 of config field, 4-bit bitmap)
    pub msel_enable: u8,
}

impl MasterPacketConfig {
    /// Parse from master config register value.
    ///
    /// Master config register layout:
    /// - Bit 31: MASTER_ENABLE
    /// - Bit 30: PACKET_ENABLE
    /// - Bits 6:0: CONFIGURATION (when packet_enable=1):
    ///   - Bit 7: DROP_HEADER
    ///   - Bits 6:3: MSEL_ENABLE (4-bit bitmap)
    ///   - Bits 2:0: ARBITOR
    pub fn from_register(value: u32) -> Self {
        let packet_enable = (value >> 30) & 1 != 0;
        let config = value & 0xFF; // Lower 8 bits
        Self {
            packet_enable,
            drop_header: (config >> 7) & 1 != 0,
            arbiter: (config & 0x7) as u8,
            msel_enable: ((config >> 3) & 0xF) as u8,
        }
    }

    /// Check if this master accepts packets from the given arbiter and msel.
    pub fn accepts(&self, arbiter: u8, msel: u8) -> bool {
        self.packet_enable
            && self.arbiter == arbiter
            && (self.msel_enable >> msel) & 1 != 0
    }
}

/// Active packet tracking for a slave port currently forwarding packet data.
#[derive(Debug, Clone)]
pub struct ActivePacket {
    /// Master port indices this packet routes to
    pub target_masters: Vec<u8>,
    /// Number of data words forwarded so far
    pub words_forwarded: usize,
    /// Which arbiter this packet is using (for lock release on TLAST).
    pub arbiter: u8,
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
    /// Pipeline latency for this route (cycles from slave input to master output).
    /// Determined by source/destination port types per AM020:
    /// - local->local: STREAM_LOCAL_TO_LOCAL_LATENCY (3)
    /// - local->external: STREAM_LOCAL_TO_EXTERNAL_LATENCY (4)
    /// - external->local: STREAM_LOCAL_TO_LOCAL_LATENCY (3)
    /// - external->external: STREAM_EXTERNAL_TO_EXTERNAL_LATENCY (4)
    pub latency: u8,
}

/// A word traversing the intra-tile switch pipeline.
#[derive(Debug, Clone)]
struct InSwitchWord {
    master_idx: u8,
    data: u32,
    tlast: bool,
    cycles_remaining: u8,
}

impl LocalRoute {
    /// Create a new local route with default latency.
    pub fn new(slave_idx: u8, master_idx: u8) -> Self {
        Self {
            slave_idx,
            master_idx,
            enabled: true,
            latency: arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
        }
    }

    /// Create a route with latency determined by port types.
    pub fn with_port_latency(slave_idx: u8, master_idx: u8, slave_type: &PortType, master_type: &PortType) -> Self {
        let latency = match (slave_type.is_external(), master_type.is_external()) {
            (false, false) => arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
            (false, true)  => arch_timing::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
            (true, false)  => arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY, // ext->local same as local->local
            (true, true)   => arch_timing::STREAM_EXTERNAL_TO_EXTERNAL_LATENCY,
        };
        Self { slave_idx, master_idx, enabled: true, latency }
    }
}

impl PortType {
    /// Whether this port connects to another tile (external/inter-tile).
    pub fn is_external(&self) -> bool {
        matches!(self, PortType::North | PortType::South | PortType::East | PortType::West)
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
    /// Local routes (slave → master within this tile, circuit mode)
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
    arbiter_locks: [Option<usize>; 8],

    /// Words traversing the intra-tile switch pipeline.
    ///
    /// Models the AM020 switch pipeline latency: data takes 3-4 cycles to
    /// traverse from slave input to master output within a single tile's
    /// stream switch. Without this, data moves slave->master in 1 cycle.
    switch_pipeline: Vec<InSwitchWord>,

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
    /// Port layout is defined in arch::{COMPUTE_MASTER_PORTS, COMPUTE_SLAVE_PORTS}.
    /// Port 3 (master and slave) is Tile_Ctrl, not Core.
    pub fn new_compute_tile(col: u8, row: u8) -> Self {
        let mut masters = Self::build_ports_from_spec(crate::arch::COMPUTE_MASTER_PORTS, PortDirection::Master);
        let mut slaves = Self::build_ports_from_spec(crate::arch::COMPUTE_SLAVE_PORTS, PortDirection::Slave);

        // Tag port 3 as TileCtrl (AM025: Compute port 3 = Tile_Ctrl)
        if masters.len() > 3 { masters[3].port_type = PortType::TileCtrl; }
        if slaves.len() > 3 { slaves[3].port_type = PortType::TileCtrl; }

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
            switch_pipeline: Vec::new(),
            fatal_errors: Vec::new(),
        }
    }

    /// Create a new stream switch for a memory tile.
    ///
    /// Port layout is defined in arch::{MEMTILE_MASTER_PORTS, MEMTILE_SLAVE_PORTS}.
    /// Port 6 (master and slave) is Tile_Ctrl, not Core.
    ///
    /// Note the asymmetry: 6 North masters but only 4 North slaves, and
    /// 4 South masters but 6 South slaves. This matches MemTile's role as
    /// a buffer between Shim (which has 6 North outputs) and Compute tiles.
    pub fn new_mem_tile(col: u8, row: u8) -> Self {
        let mut masters = Self::build_ports_from_spec(crate::arch::MEMTILE_MASTER_PORTS, PortDirection::Master);
        let mut slaves = Self::build_ports_from_spec(crate::arch::MEMTILE_SLAVE_PORTS, PortDirection::Slave);

        // Tag port 6 as TileCtrl (AM025: MemTile port 6 = Tile_Ctrl)
        if masters.len() > 6 { masters[6].port_type = PortType::TileCtrl; }
        if slaves.len() > 6 { slaves[6].port_type = PortType::TileCtrl; }

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
            switch_pipeline: Vec::new(),
            fatal_errors: Vec::new(),
        }
    }

    /// Create a new stream switch for a shim tile.
    ///
    /// Port layout is defined in arch::{SHIM_MASTER_PORTS, SHIM_SLAVE_PORTS}.
    /// Port 0 (master and slave) is Tile_Ctrl, not Core.
    ///
    /// The 6 North masters (12-17) connect 1:1 to MemTile South slaves (7-12).
    pub fn new_shim_tile(col: u8) -> Self {
        let mut masters = Self::build_ports_from_spec(crate::arch::SHIM_MASTER_PORTS, PortDirection::Master);
        let mut slaves = Self::build_ports_from_spec(crate::arch::SHIM_SLAVE_PORTS, PortDirection::Slave);

        // Tag port 0 as TileCtrl (AM025: Shim port 0 = Tile_Ctrl)
        if !masters.is_empty() { masters[0].port_type = PortType::TileCtrl; }
        if !slaves.is_empty() { slaves[0].port_type = PortType::TileCtrl; }

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

        // Phase 2: Accept new words from slaves into the pipeline
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

            let slave_has_data = self.slaves[slave_idx].has_data();

            // Debug: trace all routes for tiles with local routes
            if slave_has_data || self.row >= 1 {
                let fifo_len = self.slaves[slave_idx].fifo.len();
                log::trace!("TileSwitch({},{}): route slave[{}]->master[{}] has_data={} fifo_len={}",
                    self.col, self.row, slave_idx, master_idx, slave_has_data, fifo_len);
            }

            if slave_has_data {
                // Pop from slave and enter the switch pipeline with route-specific latency
                if let Some((data, tlast)) = self.slaves[slave_idx].pop_with_tlast() {
                    self.switch_pipeline.push(InSwitchWord {
                        master_idx: master_idx as u8,
                        data,
                        tlast,
                        cycles_remaining: route.latency,
                    });
                    log::debug!("TileSwitch({},{}): slave[{}] -> pipeline({}) -> master[{}] data=0x{:08X}{}",
                        self.col, self.row, slave_idx, route.latency, master_idx, data,
                        if tlast { " TLAST" } else { "" });
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
                    self.switch_pipeline.swap_remove(i);
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
                        // Check arbiter lock: another slave may hold this arbiter
                        if let Some(locked_by) = self.arbiter_locks[arbiter as usize] {
                            if locked_by != slave_idx {
                                // Arbiter busy -- backpressure (hold data in FIFO)
                                self.slaves[slave_idx].cycle_stalled = true;
                                log::trace!("TileSwitch({},{}): slave[{}] waiting for arbiter {} (held by slave[{}])",
                                    self.col, self.row, slave_idx, arbiter, locked_by);
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
                    // Release arbiter lock
                    if let Some(ref active) = self.active_packets[slave_idx] {
                        self.arbiter_locks[active.arbiter as usize] = None;
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
        word |= (self.stream_id as u32) & crate::arch::packet::STREAM_ID_MASK;

        // Packet Type: bits 14-12
        word |= ((self.packet_type as u32) & crate::arch::packet::TYPE_MASK)
            << crate::arch::packet::TYPE_SHIFT as usize;

        // Source Row: bits 20-16
        word |= ((self.src_row as u32) & crate::arch::packet::SRC_ROW_MASK)
            << crate::arch::packet::SRC_ROW_SHIFT as usize;

        // Source Column: bits 27-21
        word |= ((self.src_col as u32) & crate::arch::packet::SRC_COL_MASK)
            << crate::arch::packet::SRC_COL_SHIFT as usize;

        // Calculate odd parity over bits 30-0
        let parity = (word.count_ones() & 1) ^ 1; // Odd parity
        word |= parity << crate::arch::packet::PARITY_SHIFT as usize;

        word
    }

    /// Decode from 32-bit header word.
    ///
    /// Returns (header, parity_ok) tuple.
    pub fn decode(word: u32) -> (Self, bool) {
        // Extract fields
        let stream_id = (word & crate::arch::packet::STREAM_ID_MASK) as u8;

        let packet_type = PacketType::from_u8(
            ((word >> crate::arch::packet::TYPE_SHIFT as usize) & crate::arch::packet::TYPE_MASK) as u8,
        );

        let src_row =
            ((word >> crate::arch::packet::SRC_ROW_SHIFT as usize) & crate::arch::packet::SRC_ROW_MASK) as u8;

        let src_col =
            ((word >> crate::arch::packet::SRC_COL_SHIFT as usize) & crate::arch::packet::SRC_COL_MASK) as u8;

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
            self.arb_delay = arch_timing::PACKET_ARBITRATION_OVERHEAD;
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

        // Configure route from slave 0 (Core) to master 0 (Core) -- local->local = 3 cycles
        ss.configure_local_route(0, 0);
        let latency = ss.local_routes[0].latency;
        assert_eq!(latency, 3, "Core->Core should be local->local latency");

        // Put data in slave port
        ss.slaves[0].push(0xDEADBEEF);
        assert!(ss.slaves[0].has_data());
        assert!(!ss.masters[0].has_data());

        // First step: data enters pipeline (popped from slave)
        let forwarded = ss.step();
        assert_eq!(forwarded, 0, "Data in pipeline, not yet delivered");
        assert!(!ss.slaves[0].has_data(), "Slave popped");
        assert!(!ss.masters[0].has_data(), "Master not yet received");

        // Step through pipeline: needs 'latency' more steps to deliver
        // (countdown: latency -> latency-1 -> ... -> 1 -> 0=deliver)
        for _ in 0..latency {
            ss.step();
        }

        // After latency+1 total steps, data should be in master
        assert!(ss.masters[0].has_data(), "Delivered after {} pipeline cycles", latency);
        assert_eq!(ss.masters[0].peek(), Some(0xDEADBEEF));
    }

    #[test]
    fn test_switch_step_multiple_routes() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Configure two routes (both local->local)
        ss.configure_local_route(0, 0);
        ss.configure_local_route(1, 1);
        let latency = ss.local_routes[0].latency;

        // Put data in both slave ports
        ss.slaves[0].push(0x11111111);
        ss.slaves[1].push(0x22222222);

        // Step through pipeline: 1 accept + latency delivery
        for _ in 0..=latency {
            ss.step();
        }

        assert_eq!(ss.masters[0].pop(), Some(0x11111111));
        assert_eq!(ss.masters[1].pop(), Some(0x22222222));
    }

    #[test]
    fn test_switch_step_backpressure() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);
        ss.configure_local_route(0, 0);
        let latency = ss.local_routes[0].latency;

        // Fill the master port's FIFO
        while ss.masters[0].can_accept() {
            ss.masters[0].push(0x99999999);
        }

        // Put data in slave
        ss.slaves[0].push(0xDEADBEEF);

        // Step through pipeline -- data should NOT be delivered (master full)
        for _ in 0..latency + 2 {
            ss.step();
        }
        assert!(!ss.slaves[0].has_data(), "Slave was popped into pipeline");
        // Data is stuck in pipeline (backpressure at master)
        assert!(!ss.switch_pipeline.is_empty(), "Data stuck in pipeline due to backpressure");

        // Make room in master
        ss.masters[0].pop();

        // Next step should deliver
        ss.step();
        assert!(ss.switch_pipeline.is_empty(), "Delivered after backpressure cleared");
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

        // After 1 step, data is in pipeline (still pending)
        ss.step();
        assert!(ss.has_pending_local(), "Pipeline has in-flight data");

        // Step through pipeline delivery
        let latency = ss.local_routes[0].latency;
        for _ in 0..latency {
            ss.step();
        }
        // Data delivered to master -- master has data so has_pending_data is true
        // but has_pending_local checks slaves and pipeline, not masters
        // Pipeline should be empty now, and slave is empty
        assert!(!ss.switch_pipeline.is_empty() || ss.masters[0].has_data(),
            "Data should be delivered or in pipeline");
        // Clear pipeline check
        ss.masters[0].pop();
        assert!(!ss.has_pending_local(), "Pipeline empty after delivery and drain");
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

    // ========================================================================
    // Arbiter Locking Tests
    // ========================================================================

    /// Helper: build a slave slot register value.
    /// Layout: id[28:24], mask[20:16], enable[8], msel[5:4], arbiter[2:0]
    fn make_slot_reg(pkt_id: u8, mask: u8, msel: u8, arbiter: u8) -> u32 {
        ((pkt_id as u32) << 24) | ((mask as u32) << 16) | (1 << 8) | ((msel as u32) << 4) | (arbiter as u32)
    }

    /// Helper: build a master packet config register value.
    /// Layout: enable[31], packet_enable[30], drop_header[7], msel_enable[6:3], arbiter[2:0]
    fn make_master_pkt_reg(arbiter: u8, msel_enable: u8, drop_header: bool) -> u32 {
        (1 << 31) | (1 << 30)
            | if drop_header { 1 << 7 } else { 0 }
            | ((msel_enable as u32) << 3)
            | (arbiter as u32)
    }

    #[test]
    fn test_arbiter_lock_prevents_interleave() {
        // Two slaves route through the SAME arbiter to the same master.
        // With locking, one completes its full packet before the other starts.
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Enable packet mode on trace slave ports
        ss.slaves[23].packet_enable = true;
        ss.slaves[24].packet_enable = true;
        // Slave 23 (core trace): pkt_id=1, arbiter=0, msel=0
        ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
        // Slave 24 (mem trace): pkt_id=2, arbiter=0, msel=0
        ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 0));

        // Master 7: packet mode, arbiter=0, msel_enable=0b0001 (accepts msel=0)
        ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));

        // Build two 4-word packets (header + 3 data + TLAST on last).
        // Slave FIFO is 4 deep, so these fit. Master FIFO is only 2 deep,
        // so we drain between steps to simulate downstream consumption.
        let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
        let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);

        // Packet 1 on slave 23
        ss.slaves[23].push_with_tlast(hdr1.encode(), false);
        ss.slaves[23].push_with_tlast(0xAAAA_0001, false);
        ss.slaves[23].push_with_tlast(0xAAAA_0002, false);
        ss.slaves[23].push_with_tlast(0xAAAA_0003, true);

        // Packet 2 on slave 24
        ss.slaves[24].push_with_tlast(hdr2.encode(), false);
        ss.slaves[24].push_with_tlast(0xBBBB_0001, false);
        ss.slaves[24].push_with_tlast(0xBBBB_0002, false);
        ss.slaves[24].push_with_tlast(0xBBBB_0003, true);

        // Step and drain the master each time (simulates downstream consumer)
        let mut output: Vec<u32> = Vec::new();
        for _ in 0..20 {
            ss.step();
            while let Some((word, _)) = ss.masters[7].pop_with_tlast() {
                output.push(word);
            }
        }

        // Should have all 8 words (2 x 4-word packets)
        assert_eq!(output.len(), 8, "expected 8 words, got {}: {:08X?}", output.len(), output);

        // First 4 words must ALL be from the same packet (no interleaving).
        // Slave 23 has lower index, so it gets priority.
        let first_pkt_id = output[0] & 0x1F;
        assert_eq!(first_pkt_id, 1, "slave 23 (lower index) should go first");

        // Words 1-3 of first packet: 0xAAAA_xxxx
        for i in 1..4 {
            assert_eq!(output[i] >> 16, 0xAAAA,
                "word {} should be from packet 1, got 0x{:08X}", i, output[i]);
        }

        // Second packet: words 4-7
        let second_pkt_id = output[4] & 0x1F;
        assert_eq!(second_pkt_id, 2, "packet 2 should follow");
        for i in 5..8 {
            assert_eq!(output[i] >> 16, 0xBBBB,
                "word {} should be from packet 2, got 0x{:08X}", i, output[i]);
        }
    }

    #[test]
    fn test_different_arbiters_no_contention() {
        // Two slaves use DIFFERENT arbiters -- both proceed simultaneously.
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Enable packet mode on trace slave ports
        ss.slaves[23].packet_enable = true;
        ss.slaves[24].packet_enable = true;
        // Slave 23: pkt_id=1, arbiter=0, msel=0
        ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
        // Slave 24: pkt_id=2, arbiter=1, msel=0
        ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 1));

        // Master 7: arbiter=0, master 8: arbiter=1
        ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));
        ss.configure_master_packet(8, make_master_pkt_reg(1, 0b0001, false));

        let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
        let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);

        // 2-word packets (header + data+TLAST) fit within 2-deep master FIFO
        ss.slaves[23].push_with_tlast(hdr1.encode(), false);
        ss.slaves[23].push_with_tlast(0xAAAA_0001, true);

        ss.slaves[24].push_with_tlast(hdr2.encode(), false);
        ss.slaves[24].push_with_tlast(0xBBBB_0001, true);

        // Step 1: both headers forward in parallel (different arbiters)
        let words = ss.step();
        assert_eq!(words, 2, "both slaves should forward in parallel");

        // Step 2: both TLAST data words forward in parallel
        let words2 = ss.step();
        assert_eq!(words2, 2, "both data words should forward in parallel");

        // Verify master 7 got packet 1, master 8 got packet 2
        let mut m7: Vec<u32> = Vec::new();
        let mut m8: Vec<u32> = Vec::new();
        while let Some(w) = ss.masters[7].pop() { m7.push(w); }
        while let Some(w) = ss.masters[8].pop() { m8.push(w); }

        assert_eq!(m7.len(), 2, "master 7 should have 2 words");
        assert_eq!(m8.len(), 2, "master 8 should have 2 words");
        assert_eq!(m7[0] & 0x1F, 1, "master 7 = packet 1");
        assert_eq!(m8[0] & 0x1F, 2, "master 8 = packet 2");
    }

    #[test]
    fn test_arbiter_lock_released_on_tlast() {
        // Verify the arbiter is freed after TLAST so the next packet can proceed.
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Enable packet mode on trace slave ports
        ss.slaves[23].packet_enable = true;
        ss.slaves[24].packet_enable = true;
        // Slave 23: arbiter=0
        ss.configure_slave_slot(23, 0, make_slot_reg(1, 0x1F, 0, 0));
        // Slave 24: arbiter=0 (same!)
        ss.configure_slave_slot(24, 0, make_slot_reg(2, 0x1F, 0, 0));
        // Master 7: arbiter=0
        ss.configure_master_packet(7, make_master_pkt_reg(0, 0b0001, false));

        // First packet: 2 words (header + data+TLAST) from slave 23
        let hdr1 = PacketHeader::new(1, 0, 2).with_type(PacketType::Trace);
        ss.slaves[23].push_with_tlast(hdr1.encode(), false);
        ss.slaves[23].push_with_tlast(0xAAAA_0001, true);

        // Step and drain: process first packet completely
        ss.step(); // header forwarded
        ss.masters[7].pop(); // drain header (make room)
        ss.step(); // data+TLAST forwarded, arbiter released
        ss.masters[7].pop(); // drain data

        // Arbiter should be free now
        assert!(ss.arbiter_locks[0].is_none(), "arbiter 0 should be free after TLAST");

        // Now slave 24 should be able to send through the same arbiter
        let hdr2 = PacketHeader::new(2, 0, 2).with_type(PacketType::Trace);
        ss.slaves[24].push_with_tlast(hdr2.encode(), false);
        ss.slaves[24].push_with_tlast(0xBBBB_0001, true);

        let words = ss.step(); // header from slave 24
        assert_eq!(words, 1, "slave 24 should now use arbiter 0");

        // Verify master 7 got packet 2's header
        let (w, _) = ss.masters[7].pop_with_tlast().unwrap();
        assert_eq!(w & 0x1F, 2, "should be packet 2 header");
    }

    #[test]
    fn test_cycle_active_tracks_port_activity() {
        // cycle_active captures whether a port had data at any point during
        // the routing cycle. This is needed for PORT_RUNNING trace events
        // because data enters and exits FIFOs within a single route_streams()
        // call, making between-step has_data() checks always see empty ports.
        let mut port = StreamPort::new(0, PortDirection::Slave, PortType::Dma(0));

        // Initially not active
        assert!(!port.cycle_active);
        assert!(!port.has_data());

        // Push marks active
        port.push_with_tlast(0xAAAA, false);
        assert!(port.cycle_active);

        // Pop drains data, but cycle_active persists
        port.pop();
        assert!(!port.has_data(), "FIFO should be empty after pop");
        assert!(port.cycle_active, "cycle_active should persist after pop");
    }

    #[test]
    fn test_begin_routing_cycle_seeds_from_fifo() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Put data in slave[1] (simulating backpressure holdover)
        ss.slaves[1].push_with_tlast(0x1234, false);

        // begin_routing_cycle seeds cycle_active from existing FIFO state
        ss.begin_routing_cycle();

        assert!(ss.slaves[1].cycle_active,
            "slave with existing data should be active");
        assert!(!ss.slaves[0].cycle_active,
            "empty slave should not be active");
        assert!(!ss.masters[0].cycle_active,
            "empty master should not be active");
    }

    #[test]
    fn test_begin_routing_cycle_clears_previous() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Push data to mark a port active
        ss.slaves[0].push_with_tlast(0xAAAA, false);
        assert!(ss.slaves[0].cycle_active);

        // Drain it
        ss.slaves[0].pop();

        // begin_routing_cycle clears the stale active flag
        ss.begin_routing_cycle();
        assert!(!ss.slaves[0].cycle_active,
            "empty port should not be active after begin_routing_cycle");
    }

    #[test]
    fn test_cycle_tlast_tracks_tlast_on_push() {
        let mut port = StreamPort::new(0, PortDirection::Master, PortType::Dma(0));

        // Push without TLAST: cycle_tlast stays false
        port.push_with_tlast(0x1111, false);
        assert!(!port.cycle_tlast);

        // Push with TLAST: cycle_tlast becomes true
        port.push_with_tlast(0x2222, true);
        assert!(port.cycle_tlast);
    }

    #[test]
    fn test_cycle_tlast_tracks_tlast_on_pop() {
        let mut port = StreamPort::new(0, PortDirection::Slave, PortType::Dma(0));

        // Push a word with TLAST (cycle_tlast set on push)
        port.push_with_tlast(0xAAAA, true);
        assert!(port.cycle_tlast);

        // Reset to test pop path
        port.cycle_tlast = false;
        let (_, tlast) = port.pop_with_tlast().unwrap();
        assert!(tlast);
        assert!(port.cycle_tlast, "pop_with_tlast should set cycle_tlast");
    }

    #[test]
    fn test_begin_routing_cycle_clears_stalled_and_tlast() {
        let mut ss = StreamSwitch::new_compute_tile(0, 2);

        // Manually set flags to verify they get cleared
        ss.slaves[0].cycle_stalled = true;
        ss.slaves[0].cycle_tlast = true;
        ss.masters[0].cycle_stalled = true;
        ss.masters[0].cycle_tlast = true;

        ss.begin_routing_cycle();

        assert!(!ss.slaves[0].cycle_stalled);
        assert!(!ss.slaves[0].cycle_tlast);
        assert!(!ss.masters[0].cycle_stalled);
        assert!(!ss.masters[0].cycle_tlast);
    }

    // ========================================================================
    // Packet routing integration tests (MemTile-focused)
    // ========================================================================

    #[test]
    fn test_packet_routing_basic_memtile() {
        // MemTile DMA slave[0] routes pkt_id=0 to North master[12].
        // Verifies the fundamental packet path: header match -> arbiter -> master.
        let mut ss = StreamSwitch::new_mem_tile(0, 1);

        // Enable packet mode on slave[0] (bit 31=enable, bit 30=packet_enable)
        ss.slaves[0].packet_enable = true;
        // Slave[0] (DMA:0): pkt_id=0, mask=0x1F, arbiter=0, msel=0
        ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
        // Master[12] (North:1): packet mode, arbiter=0, msel_enable=0b0001
        ss.configure_master_packet(12, make_master_pkt_reg(0, 0b0001, false));

        // Build a 4-word packet: header + 2 data + data+TLAST
        let hdr = PacketHeader::new(0, 0, 1).encode();
        ss.slaves[0].push_with_tlast(hdr, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0001, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0002, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0003, true);

        // Step and drain master each time (FIFO capacity is 2)
        let mut output = Vec::new();
        for _ in 0..8 {
            ss.step();
            while let Some((w, _)) = ss.masters[12].pop_with_tlast() {
                output.push(w);
            }
        }

        assert_eq!(output.len(), 4, "master[12] should have 4 words: {:08X?}", output);
        assert_eq!(output[0], hdr, "first word should be the header");
        assert_eq!(output[1], 0xDA7A_0001);
        assert_eq!(output[2], 0xDA7A_0002);
        assert_eq!(output[3], 0xDA7A_0003);

        // No other master should have data (spot-check DMA and South masters)
        for m in [0, 1, 7, 8, 9, 10, 11, 13] {
            assert!(!ss.masters[m].has_data(),
                "master[{}] should be empty but has data", m);
        }
    }

    #[test]
    fn test_packet_routing_multi_slave_memtile() {
        // Two MemTile slaves with different pkt_ids route to different masters.
        // slave[0] (DMA:0) pkt_id=0 -> master[9] (South:2)
        // slave[13] (North:0) pkt_id=1 -> master[0] (DMA:0)
        let mut ss = StreamSwitch::new_mem_tile(0, 1);

        // Enable packet mode on both slaves
        ss.slaves[0].packet_enable = true;
        ss.slaves[13].packet_enable = true;
        // Slave[0]: pkt_id=0, arbiter=0, msel=0
        ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
        // Slave[13]: pkt_id=1, arbiter=1, msel=0
        ss.configure_slave_slot(13, 0, make_slot_reg(1, 0x1F, 0, 1));

        // Master[9] (South:2): arbiter=0, msel_enable=0b0001
        ss.configure_master_packet(9, make_master_pkt_reg(0, 0b0001, false));
        // Master[0] (DMA:0): arbiter=1, msel_enable=0b0001
        ss.configure_master_packet(0, make_master_pkt_reg(1, 0b0001, false));

        // Packet from DMA slave (pkt_id=0): 2 words
        let hdr0 = PacketHeader::new(0, 0, 1).encode();
        ss.slaves[0].push_with_tlast(hdr0, false);
        ss.slaves[0].push_with_tlast(0xAAAA_0001, true);

        // Packet from North slave (pkt_id=1): 2 words
        let hdr1 = PacketHeader::new(1, 0, 2).encode();
        ss.slaves[13].push_with_tlast(hdr1, false);
        ss.slaves[13].push_with_tlast(0xBBBB_0001, true);

        // Step enough times
        for _ in 0..4 {
            ss.step();
        }

        // Master[9] should have pkt_id=0 data
        let (w, _) = ss.masters[9].pop_with_tlast().unwrap();
        assert_eq!(w & 0x1F, 0, "master[9] should have pkt_id=0");

        // Master[0] should have pkt_id=1 data
        let (w, _) = ss.masters[0].pop_with_tlast().unwrap();
        assert_eq!(w & 0x1F, 1, "master[0] should have pkt_id=1");
    }

    #[test]
    fn test_packet_routing_drop_header() {
        // When drop_header=true, the header word is consumed from the slave
        // but NOT forwarded to the master. Only data words appear in output.
        let mut ss = StreamSwitch::new_mem_tile(0, 1);

        // Enable packet mode on slave[0]
        ss.slaves[0].packet_enable = true;
        // Slave[0]: pkt_id=0, arbiter=0, msel=0
        ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
        // Master[0]: arbiter=0, msel_enable=0b0001, drop_header=TRUE
        ss.configure_master_packet(0, make_master_pkt_reg(0, 0b0001, true));

        // 3-word packet: header + data + data+TLAST
        let hdr = PacketHeader::new(0, 0, 1).encode();
        ss.slaves[0].push_with_tlast(hdr, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0001, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0002, true);

        for _ in 0..4 {
            ss.step();
        }

        // Master should have only the 2 data words (header dropped)
        let mut output = Vec::new();
        while let Some((w, _)) = ss.masters[0].pop_with_tlast() {
            output.push(w);
        }
        assert_eq!(output.len(), 2,
            "drop_header should remove header, leaving 2 data words: {:08X?}", output);
        assert_eq!(output[0], 0xDA7A_0001);
        assert_eq!(output[1], 0xDA7A_0002);

        // Slave should be empty (header was consumed, not left behind)
        assert!(!ss.slaves[0].has_data(), "slave should be drained");
    }

    #[test]
    fn test_packet_routing_no_route_is_fatal() {
        // A packet with no matching route should produce a fatal error.
        let mut ss = StreamSwitch::new_mem_tile(0, 1);

        // Enable packet mode on slave[0]
        ss.slaves[0].packet_enable = true;
        // Configure slave[0] for pkt_id=0 only
        ss.configure_slave_slot(0, 0, make_slot_reg(0, 0x1F, 0, 0));
        ss.configure_master_packet(9, make_master_pkt_reg(0, 0b0001, false));

        // Push a packet with pkt_id=1 (unmatched)
        let hdr = PacketHeader::new(1, 0, 1).encode();
        ss.slaves[0].push_with_tlast(hdr, false);
        ss.slaves[0].push_with_tlast(0xDA7A_0001, true);

        ss.step();

        // Should produce a fatal error
        assert!(!ss.fatal_errors.is_empty(),
            "unroutable packet should produce fatal error");
        assert!(ss.fatal_errors[0].contains("no packet route"),
            "error should mention 'no packet route': {}", ss.fatal_errors[0]);
    }
}
