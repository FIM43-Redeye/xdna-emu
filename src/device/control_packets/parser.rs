//! Control packet header parsing.
//!
//! Parses the 32-bit control packet header into structured fields using
//! bit positions from `xdna_archspec::aie2::ctrl_packet::*` (generated from
//! xdna-archspec, derived from AM020/AM025).
//!
//! # Header Layout (AM020 Table 3)
//!
//! ```text
//! [31]    Parity
//! [30:24] Response_ID (7-bit stream ID for response routing)
//! [23:22] Operation   (2-bit opcode)
//! [21:20] Length      (2-bit, value+1 = number of data beats)
//! [19:0]  Address     (20-bit tile-local register offset)
//! ```
//!
//! # Relationship to `src/parser/*`
//!
//! Structurally distinct from `src/parser/*` (the XCLBIN / CDO / ELF
//! static-configuration parsers): control packets carry NPU runtime
//! commands as a 32-bit header + data beats, whereas CDO carries a
//! 20-byte container header + variable-length typed command stream,
//! and XCLBIN is a Xilinx axlf container. No framing primitives
//! overlap. Subsystem 8 audit §7 (`docs/arch/subsys8-audit.md`)
//! evaluated shared-module extraction and found zero overlap -- the
//! two parsers stay separate by design.

use std::fmt;

/// Control packet operation codes.
///
/// These map directly to the 2-bit operation field in the control packet
/// header. Values are derived from `xdna_archspec::aie2::ctrl_packet::OP_*` constants
/// (generated from xdna-archspec).
///
/// Note: There is no MaskWrite opcode in control packets. MaskWrite is a
/// CDO/host-level operation (see [`super::MaskWriteOp`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CtrlOpCode {
    /// Write data words to consecutive register addresses.
    /// Opcode value: 0 (arch::ctrl_packet::OP_WRITE)
    Write = 0,
    /// Read register(s) and send response back via TileCtrl master port.
    /// Opcode value: 1 (arch::ctrl_packet::OP_READ)
    Read = 1,
    /// Write with auto-incrementing address (same behavior as Write for
    /// the emulator, since both write consecutive addresses).
    /// Opcode value: 2 (arch::ctrl_packet::OP_WRITE_INCR)
    WriteIncr = 2,
    /// Block write to consecutive registers.
    /// Opcode value: 3 (arch::ctrl_packet::OP_BLOCK_WRITE)
    BlockWrite = 3,
}

impl CtrlOpCode {
    /// Convert from a raw 2-bit operation value.
    ///
    /// Returns `None` for values outside 0..=3, though the 2-bit field
    /// makes this impossible in practice.
    pub fn from_raw(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Write),
            1 => Some(Self::Read),
            2 => Some(Self::WriteIncr),
            3 => Some(Self::BlockWrite),
            _ => None,
        }
    }

    /// Check whether this opcode produces a response packet.
    pub fn has_response(&self) -> bool {
        matches!(self, Self::Read)
    }

    /// Check whether this opcode requires data payload beats.
    pub fn has_payload(&self) -> bool {
        !matches!(self, Self::Read)
    }
}

impl fmt::Display for CtrlOpCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Write => write!(f, "WRITE"),
            Self::Read => write!(f, "READ"),
            Self::WriteIncr => write!(f, "WRITE_INCR"),
            Self::BlockWrite => write!(f, "BLOCK_WRITE"),
        }
    }
}

/// Errors that can occur during control packet parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The operation field contained an invalid value.
    /// This should be impossible with a 2-bit field, but we handle it
    /// for robustness.
    InvalidOpCode(u8),
    /// Payload length does not match what the header declared.
    PayloadLengthMismatch {
        /// Number of beats declared in the header (1-4).
        expected: u8,
        /// Actual number of payload words provided.
        actual: usize,
    },
    /// Read operation was given a non-empty payload (reads have no data).
    ReadWithPayload(usize),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOpCode(v) => write!(f, "invalid control packet opcode: {}", v),
            Self::PayloadLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "payload length mismatch: header declares {} beat(s), got {} word(s)",
                    expected, actual
                )
            }
            Self::ReadWithPayload(n) => {
                write!(f, "OP_READ should have no payload, but got {} word(s)", n)
            }
        }
    }
}

impl std::error::Error for ParseError {}

/// Parsed control packet header fields.
///
/// Extracted from the raw 32-bit header using arch-derived bit positions.
/// This is a lightweight struct for inspecting header fields without
/// committing to a full packet parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderFields {
    /// Tile-local register address (bits 19:0).
    pub address: u32,
    /// Number of data beats (1-4, decoded from the 2-bit length field).
    pub beats: u8,
    /// Operation code.
    pub opcode: CtrlOpCode,
    /// Response/stream ID for response routing (bits 30:24).
    pub response_id: u8,
    /// Parity bit (bit 31).
    pub parity: bool,
}

/// Parse control packet header fields from a raw 32-bit word.
///
/// Uses `xdna_archspec::aie2::ctrl_packet::*` constants for all bit positions,
/// ensuring derivation from the toolchain rather than hardcoded values.
///
/// Returns `Err` only if the operation field is invalid (impossible with
/// a 2-bit field, but handled defensively).
pub fn parse_header(header: u32) -> Result<HeaderFields, ParseError> {
    use xdna_archspec::aie2::ctrl_packet::*;

    let address = header & ADDRESS_MASK;
    let beats = ((header >> LENGTH_SHIFT) & LENGTH_MASK) as u8 + 1;
    let op_raw = ((header >> OPERATION_SHIFT) & OPERATION_MASK) as u8;
    let response_id = ((header >> RESPONSE_ID_SHIFT) & RESPONSE_ID_MASK) as u8;
    let parity = (header >> PARITY_BIT) & 1 != 0;

    let opcode = CtrlOpCode::from_raw(op_raw).ok_or(ParseError::InvalidOpCode(op_raw))?;

    Ok(HeaderFields { address, beats, opcode, response_id, parity })
}

/// Build a control packet header word from structured fields.
///
/// This is the inverse of [`parse_header`]. Useful for test construction
/// and response packet generation.
pub fn build_header(address: u32, beats: u8, opcode: CtrlOpCode, response_id: u8) -> u32 {
    use xdna_archspec::aie2::ctrl_packet::*;

    let length_val = beats.saturating_sub(1) as u32;

    (address & ADDRESS_MASK)
        | ((length_val & LENGTH_MASK) << LENGTH_SHIFT)
        | (((opcode as u32) & OPERATION_MASK) << OPERATION_SHIFT)
        | (((response_id as u32) & RESPONSE_ID_MASK) << RESPONSE_ID_SHIFT)
    // Parity bit left as 0; caller can set it if needed.
}

/// A fully parsed control packet with header and payload.
///
/// Represents one complete control packet operation ready for processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlPacket {
    /// Operation code.
    pub opcode: CtrlOpCode,
    /// Tile-local register address.
    pub address: u32,
    /// Payload data words.
    ///
    /// - Write/WriteIncr/BlockWrite: 1-4 data words to write.
    /// - Read: empty (read address comes from header).
    pub data: Vec<u32>,
    /// Response/stream ID for response routing.
    pub response_id: u8,
    /// Number of beats declared in the header (1-4).
    ///
    /// For Read operations, this indicates how many registers to read.
    pub beats: u8,
}

impl ControlPacket {
    /// Parse a control packet from a header word and payload data.
    ///
    /// For write operations, `payload` must contain exactly the number of
    /// words declared in the header's length field. For read operations,
    /// `payload` must be empty.
    pub fn parse(header: u32, payload: &[u32]) -> Result<Self, ParseError> {
        let fields = parse_header(header)?;

        if fields.opcode == CtrlOpCode::Read {
            if !payload.is_empty() {
                return Err(ParseError::ReadWithPayload(payload.len()));
            }
        } else if payload.len() != fields.beats as usize {
            return Err(ParseError::PayloadLengthMismatch { expected: fields.beats, actual: payload.len() });
        }

        Ok(Self {
            opcode: fields.opcode,
            address: fields.address,
            data: payload.to_vec(),
            response_id: fields.response_id,
            beats: fields.beats,
        })
    }

    /// Create a Write packet programmatically.
    pub fn write(address: u32, value: u32) -> Self {
        Self { opcode: CtrlOpCode::Write, address, data: vec![value], response_id: 0, beats: 1 }
    }

    /// Create a Read packet programmatically.
    pub fn read(address: u32, beats: u8, response_id: u8) -> Self {
        Self { opcode: CtrlOpCode::Read, address, data: Vec::new(), response_id, beats }
    }

    /// Create a BlockWrite packet programmatically.
    pub fn block_write(address: u32, data: Vec<u32>) -> Self {
        let beats = data.len() as u8;
        Self { opcode: CtrlOpCode::BlockWrite, address, data, response_id: 0, beats }
    }

    /// Create a WriteIncr packet programmatically.
    pub fn write_incr(address: u32, data: Vec<u32>) -> Self {
        let beats = data.len() as u8;
        Self { opcode: CtrlOpCode::WriteIncr, address, data, response_id: 0, beats }
    }

    /// Whether this packet requires register writes.
    pub fn is_write_op(&self) -> bool {
        self.opcode.has_payload()
    }

    /// Whether this packet will produce a response.
    pub fn is_read_op(&self) -> bool {
        self.opcode.has_response()
    }

    /// Iterate over (address, value) pairs for write operations.
    ///
    /// Each successive word targets address + 4*i (consecutive 32-bit registers).
    /// Returns an empty iterator for Read operations.
    pub fn write_pairs(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.data
            .iter()
            .enumerate()
            .map(move |(i, &val)| (self.address + (i as u32) * 4, val))
    }
}

impl fmt::Display for ControlPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CtrlPkt({} addr=0x{:05X} beats={}", self.opcode, self.address, self.beats)?;
        if self.response_id != 0 {
            write!(f, " resp_id={}", self.response_id)?;
        }
        if !self.data.is_empty() {
            write!(f, " data=[")?;
            for (i, val) in self.data.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "0x{:08X}", val)?;
            }
            write!(f, "]")?;
        }
        write!(f, ")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a header word from raw field values.
    fn make_header(address: u32, length_raw: u8, operation: u8, response_id: u8) -> u32 {
        use xdna_archspec::aie2::ctrl_packet::*;
        (address & ADDRESS_MASK)
            | ((length_raw as u32 & LENGTH_MASK) << LENGTH_SHIFT)
            | ((operation as u32 & OPERATION_MASK) << OPERATION_SHIFT)
            | ((response_id as u32 & RESPONSE_ID_MASK) << RESPONSE_ID_SHIFT)
    }

    // -- CtrlOpCode tests --

    #[test]
    fn opcode_from_raw_all_valid() {
        assert_eq!(CtrlOpCode::from_raw(0), Some(CtrlOpCode::Write));
        assert_eq!(CtrlOpCode::from_raw(1), Some(CtrlOpCode::Read));
        assert_eq!(CtrlOpCode::from_raw(2), Some(CtrlOpCode::WriteIncr));
        assert_eq!(CtrlOpCode::from_raw(3), Some(CtrlOpCode::BlockWrite));
    }

    #[test]
    fn opcode_from_raw_invalid() {
        assert_eq!(CtrlOpCode::from_raw(4), None);
        assert_eq!(CtrlOpCode::from_raw(255), None);
    }

    #[test]
    fn opcode_has_response() {
        assert!(!CtrlOpCode::Write.has_response());
        assert!(CtrlOpCode::Read.has_response());
        assert!(!CtrlOpCode::WriteIncr.has_response());
        assert!(!CtrlOpCode::BlockWrite.has_response());
    }

    #[test]
    fn opcode_has_payload() {
        assert!(CtrlOpCode::Write.has_payload());
        assert!(!CtrlOpCode::Read.has_payload());
        assert!(CtrlOpCode::WriteIncr.has_payload());
        assert!(CtrlOpCode::BlockWrite.has_payload());
    }

    #[test]
    fn opcode_display() {
        assert_eq!(format!("{}", CtrlOpCode::Write), "WRITE");
        assert_eq!(format!("{}", CtrlOpCode::Read), "READ");
        assert_eq!(format!("{}", CtrlOpCode::WriteIncr), "WRITE_INCR");
        assert_eq!(format!("{}", CtrlOpCode::BlockWrite), "BLOCK_WRITE");
    }

    #[test]
    fn opcode_repr_matches_arch_constants() {
        use xdna_archspec::aie2::ctrl_packet::*;
        assert_eq!(CtrlOpCode::Write as u8, OP_WRITE);
        assert_eq!(CtrlOpCode::Read as u8, OP_READ);
        assert_eq!(CtrlOpCode::WriteIncr as u8, OP_WRITE_INCR);
        assert_eq!(CtrlOpCode::BlockWrite as u8, OP_BLOCK_WRITE);
    }

    // -- parse_header tests --

    #[test]
    fn parse_header_write_single_beat() {
        let header = make_header(0x1A000, 0, 0, 0); // addr=0x1A000, 1 beat, WRITE
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.address, 0x1A000);
        assert_eq!(fields.beats, 1);
        assert_eq!(fields.opcode, CtrlOpCode::Write);
        assert_eq!(fields.response_id, 0);
        assert!(!fields.parity);
    }

    #[test]
    fn parse_header_read_with_response_id() {
        let header = make_header(0x00100, 2, 1, 42); // addr=0x100, 3 beats, READ, rid=42
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.address, 0x00100);
        assert_eq!(fields.beats, 3);
        assert_eq!(fields.opcode, CtrlOpCode::Read);
        assert_eq!(fields.response_id, 42);
    }

    #[test]
    fn parse_header_block_write_four_beats() {
        let header = make_header(0x20000, 3, 3, 0); // 4 beats, BLOCK_WRITE
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.beats, 4);
        assert_eq!(fields.opcode, CtrlOpCode::BlockWrite);
    }

    #[test]
    fn parse_header_write_incr() {
        let header = make_header(0x340D0, 1, 2, 5); // 2 beats, WRITE_INCR, rid=5
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.address, 0x340D0);
        assert_eq!(fields.beats, 2);
        assert_eq!(fields.opcode, CtrlOpCode::WriteIncr);
        assert_eq!(fields.response_id, 5);
    }

    #[test]
    fn parse_header_parity_bit() {
        let header = make_header(0x100, 0, 0, 0) | (1u32 << 31);
        let fields = parse_header(header).unwrap();
        assert!(fields.parity);
    }

    #[test]
    fn parse_header_max_address() {
        // Maximum 20-bit address
        let header = make_header(0xFFFFF, 0, 0, 0);
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.address, 0xFFFFF);
    }

    #[test]
    fn parse_header_max_response_id() {
        // Maximum 7-bit response ID
        let header = make_header(0, 0, 1, 127);
        let fields = parse_header(header).unwrap();
        assert_eq!(fields.response_id, 127);
    }

    // -- build_header + round-trip tests --

    #[test]
    fn build_header_roundtrip() {
        let original = build_header(0x1A000, 3, CtrlOpCode::BlockWrite, 17);
        let parsed = parse_header(original).unwrap();
        assert_eq!(parsed.address, 0x1A000);
        assert_eq!(parsed.beats, 3);
        assert_eq!(parsed.opcode, CtrlOpCode::BlockWrite);
        assert_eq!(parsed.response_id, 17);
    }

    #[test]
    fn build_header_roundtrip_all_opcodes() {
        for opcode in [CtrlOpCode::Write, CtrlOpCode::Read, CtrlOpCode::WriteIncr, CtrlOpCode::BlockWrite] {
            let header = build_header(0x500, 2, opcode, 10);
            let parsed = parse_header(header).unwrap();
            assert_eq!(parsed.opcode, opcode);
            assert_eq!(parsed.beats, 2);
        }
    }

    // -- ControlPacket::parse tests --

    #[test]
    fn parse_op_write_packet() {
        let header = make_header(0x1A000, 0, 0, 0); // 1 beat, WRITE
        let pkt = ControlPacket::parse(header, &[0xDEAD_BEEF]).unwrap();
        assert_eq!(pkt.opcode, CtrlOpCode::Write);
        assert_eq!(pkt.address, 0x1A000);
        assert_eq!(pkt.data, vec![0xDEAD_BEEF]);
        assert_eq!(pkt.beats, 1);
        assert_eq!(pkt.response_id, 0);
    }

    #[test]
    fn parse_op_read_packet() {
        let header = make_header(0x00100, 1, 1, 42); // 2 beats, READ, rid=42
        let pkt = ControlPacket::parse(header, &[]).unwrap();
        assert_eq!(pkt.opcode, CtrlOpCode::Read);
        assert_eq!(pkt.address, 0x00100);
        assert!(pkt.data.is_empty());
        assert_eq!(pkt.beats, 2);
        assert_eq!(pkt.response_id, 42);
    }

    #[test]
    fn parse_op_blockwrite_packet() {
        let header = make_header(0x20000, 2, 3, 0); // 3 beats, BLOCK_WRITE
        let payload = [0x1111, 0x2222, 0x3333];
        let pkt = ControlPacket::parse(header, &payload).unwrap();
        assert_eq!(pkt.opcode, CtrlOpCode::BlockWrite);
        assert_eq!(pkt.data, vec![0x1111, 0x2222, 0x3333]);
        assert_eq!(pkt.beats, 3);
    }

    #[test]
    fn parse_op_write_incr_packet() {
        let header = make_header(0x340D0, 3, 2, 0); // 4 beats, WRITE_INCR
        let payload = [0xAA, 0xBB, 0xCC, 0xDD];
        let pkt = ControlPacket::parse(header, &payload).unwrap();
        assert_eq!(pkt.opcode, CtrlOpCode::WriteIncr);
        assert_eq!(pkt.data.len(), 4);
        assert_eq!(pkt.address, 0x340D0);
    }

    #[test]
    fn parse_read_with_payload_is_error() {
        let header = make_header(0x100, 0, 1, 0); // READ
        let err = ControlPacket::parse(header, &[0x1234]).unwrap_err();
        assert_eq!(err, ParseError::ReadWithPayload(1));
    }

    #[test]
    fn parse_payload_length_mismatch_too_few() {
        let header = make_header(0x100, 1, 0, 0); // 2 beats expected
        let err = ControlPacket::parse(header, &[0x1234]).unwrap_err(); // only 1 word
        assert_eq!(err, ParseError::PayloadLengthMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn parse_payload_length_mismatch_too_many() {
        let header = make_header(0x100, 0, 0, 0); // 1 beat expected
        let err = ControlPacket::parse(header, &[0x1234, 0x5678]).unwrap_err();
        assert_eq!(err, ParseError::PayloadLengthMismatch { expected: 1, actual: 2 });
    }

    // -- ControlPacket builder tests --

    #[test]
    fn write_builder() {
        let pkt = ControlPacket::write(0x1A000, 0xCAFE);
        assert_eq!(pkt.opcode, CtrlOpCode::Write);
        assert_eq!(pkt.address, 0x1A000);
        assert_eq!(pkt.data, vec![0xCAFE]);
        assert_eq!(pkt.beats, 1);
    }

    #[test]
    fn read_builder() {
        let pkt = ControlPacket::read(0x200, 3, 55);
        assert_eq!(pkt.opcode, CtrlOpCode::Read);
        assert_eq!(pkt.address, 0x200);
        assert!(pkt.data.is_empty());
        assert_eq!(pkt.beats, 3);
        assert_eq!(pkt.response_id, 55);
    }

    #[test]
    fn block_write_builder() {
        let pkt = ControlPacket::block_write(0x300, vec![1, 2, 3]);
        assert_eq!(pkt.opcode, CtrlOpCode::BlockWrite);
        assert_eq!(pkt.beats, 3);
        assert_eq!(pkt.data, vec![1, 2, 3]);
    }

    #[test]
    fn write_incr_builder() {
        let pkt = ControlPacket::write_incr(0x400, vec![10, 20]);
        assert_eq!(pkt.opcode, CtrlOpCode::WriteIncr);
        assert_eq!(pkt.beats, 2);
    }

    // -- write_pairs tests --

    #[test]
    fn write_pairs_consecutive_addresses() {
        let pkt = ControlPacket::block_write(0x1000, vec![0xAA, 0xBB, 0xCC]);
        let pairs: Vec<_> = pkt.write_pairs().collect();
        assert_eq!(pairs, vec![(0x1000, 0xAA), (0x1004, 0xBB), (0x1008, 0xCC),]);
    }

    #[test]
    fn write_pairs_empty_for_read() {
        let pkt = ControlPacket::read(0x100, 1, 0);
        let pairs: Vec<_> = pkt.write_pairs().collect();
        assert!(pairs.is_empty());
    }

    // -- is_write_op / is_read_op tests --

    #[test]
    fn is_write_op_flags() {
        assert!(ControlPacket::write(0, 0).is_write_op());
        assert!(!ControlPacket::read(0, 1, 0).is_write_op());
        assert!(ControlPacket::block_write(0, vec![1]).is_write_op());
        assert!(ControlPacket::write_incr(0, vec![1]).is_write_op());
    }

    #[test]
    fn is_read_op_flags() {
        assert!(!ControlPacket::write(0, 0).is_read_op());
        assert!(ControlPacket::read(0, 1, 0).is_read_op());
        assert!(!ControlPacket::block_write(0, vec![1]).is_read_op());
    }

    // -- Display tests --

    #[test]
    fn control_packet_display() {
        let pkt = ControlPacket::write(0x1A000, 0xDEAD_BEEF);
        let s = format!("{}", pkt);
        assert!(s.contains("WRITE"));
        assert!(s.contains("0x1A000"));
        assert!(s.contains("0xDEADBEEF"));
    }

    #[test]
    fn control_packet_display_read_no_data() {
        let pkt = ControlPacket::read(0x100, 2, 7);
        let s = format!("{}", pkt);
        assert!(s.contains("READ"));
        assert!(s.contains("resp_id=7"));
        // Should not contain "data=" since read has no payload
        assert!(!s.contains("data="));
    }

    // -- ParseError Display tests --

    #[test]
    fn parse_error_display() {
        let e = ParseError::InvalidOpCode(5);
        assert!(format!("{}", e).contains("invalid"));
        assert!(format!("{}", e).contains("5"));

        let e = ParseError::PayloadLengthMismatch { expected: 2, actual: 1 };
        assert!(format!("{}", e).contains("mismatch"));

        let e = ParseError::ReadWithPayload(3);
        assert!(format!("{}", e).contains("READ"));
    }
}
