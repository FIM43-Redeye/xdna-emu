//! Packet data types for packet-switched streams.

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
        Self { data, src_col, src_row, src_port, dest_col, dest_row, dest_port, is_last: false }
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

/// Odd-parity check over a full 32-bit header word.
///
/// The AIE packet/control-packet headers use odd parity: the total
/// number of set bits (including the parity bit) is odd on a valid
/// header. Single source of truth for the parity formula -- both
/// `PacketHeader::decode` and the control-packet reassembler call this.
pub fn odd_parity_ok(word: u32) -> bool {
    word.count_ones() & 1 == 1
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
        word |= (self.stream_id as u32) & xdna_archspec::aie2::packet::STREAM_ID_MASK;

        // Packet Type: bits 14-12
        word |= ((self.packet_type as u32) & xdna_archspec::aie2::packet::TYPE_MASK)
            << xdna_archspec::aie2::packet::TYPE_SHIFT as usize;

        // Source Row: bits 20-16
        word |= ((self.src_row as u32) & xdna_archspec::aie2::packet::SRC_ROW_MASK)
            << xdna_archspec::aie2::packet::SRC_ROW_SHIFT as usize;

        // Source Column: bits 27-21
        word |= ((self.src_col as u32) & xdna_archspec::aie2::packet::SRC_COL_MASK)
            << xdna_archspec::aie2::packet::SRC_COL_SHIFT as usize;

        // Calculate odd parity over bits 30-0
        let parity = (word.count_ones() & 1) ^ 1; // Odd parity
        word |= parity << xdna_archspec::aie2::packet::PARITY_SHIFT as usize;

        word
    }

    /// Decode from 32-bit header word.
    ///
    /// Returns (header, parity_ok) tuple.
    pub fn decode(word: u32) -> (Self, bool) {
        // Extract fields
        let stream_id = (word & xdna_archspec::aie2::packet::STREAM_ID_MASK) as u8;

        let packet_type = PacketType::from_u8(
            ((word >> xdna_archspec::aie2::packet::TYPE_SHIFT as usize)
                & xdna_archspec::aie2::packet::TYPE_MASK) as u8,
        );

        let src_row = ((word >> xdna_archspec::aie2::packet::SRC_ROW_SHIFT as usize)
            & xdna_archspec::aie2::packet::SRC_ROW_MASK) as u8;

        let src_col = ((word >> xdna_archspec::aie2::packet::SRC_COL_SHIFT as usize)
            & xdna_archspec::aie2::packet::SRC_COL_MASK) as u8;

        let parity_ok = odd_parity_ok(word);

        let header = Self { stream_id, packet_type, src_row, src_col };

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
        Self { stream_id, dest_ports: vec![dest_port], enabled: true }
    }

    /// Create a multicast route to multiple ports.
    pub fn multicast(stream_id: u8, dest_ports: Vec<u8>) -> Self {
        Self { stream_id, dest_ports, enabled: true }
    }

    /// Add a destination port (for multicast).
    pub fn add_dest(&mut self, port: u8) {
        if !self.dest_ports.contains(&port) {
            self.dest_ports.push(port);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn odd_parity_ok_basic() {
        // count_ones() odd  -> true ; even -> false
        assert!(odd_parity_ok(0b1)); // 1 one  -> odd
        assert!(!odd_parity_ok(0b11)); // 2 ones -> even
        assert!(odd_parity_ok(0b111)); // 3 ones -> odd
        assert!(!odd_parity_ok(0)); // 0 ones -> even
        assert!(!odd_parity_ok(0xFFFF_FFFF)); // 32 ones -> even parity -> not ok
    }
}
