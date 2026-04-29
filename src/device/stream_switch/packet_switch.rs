//! Packet switching logic: slot configuration, arbiter-based routing, and
//! the standalone PacketSwitch state machine.

use xdna_archspec::aie2::timing as arch_timing;
use super::packet_types::{PacketHeader, PacketRoute};

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
        self.packet_enable && self.arbiter == arbiter && (self.msel_enable >> msel) & 1 != 0
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
    pub fn with_port_latency(
        slave_idx: u8,
        master_idx: u8,
        slave_type: &super::PortType,
        master_type: &super::PortType,
    ) -> Self {
        let latency = match (slave_type.is_external(), master_type.is_external()) {
            (false, false) => arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY,
            (false, true) => arch_timing::STREAM_LOCAL_TO_EXTERNAL_LATENCY,
            (true, false) => arch_timing::STREAM_LOCAL_TO_LOCAL_LATENCY, // ext->local same as local->local
            (true, true) => arch_timing::STREAM_EXTERNAL_TO_EXTERNAL_LATENCY,
        };
        Self { slave_idx, master_idx, enabled: true, latency }
    }
}

/// A word traversing the intra-tile switch pipeline.
#[derive(Debug, Clone)]
pub(crate) struct InSwitchWord {
    pub(crate) master_idx: u8,
    pub(crate) data: u32,
    pub(crate) tlast: bool,
    pub(crate) cycles_remaining: u8,
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
