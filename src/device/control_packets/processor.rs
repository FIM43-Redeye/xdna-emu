//! Control packet processor with register access callbacks.
//!
//! The [`ControlPacketProcessor`] is a stateless dispatcher that executes
//! control packet operations by calling register access callbacks. It
//! maintains a queue of pending response packets for OP_READ operations.
//!
//! # Design
//!
//! The processor does not own or know about tile state. Instead, it receives
//! closures for register read and write operations. This keeps the control
//! packet logic decoupled from the tile implementation, making it testable
//! in isolation.
//!
//! # Usage
//!
//! ```ignore
//! use xdna_emu::device::control_packets::*;
//!
//! let mut processor = ControlPacketProcessor::new();
//! let packet = ControlPacket::write(0x1A000, 0xDEAD_BEEF);
//!
//! let mut access = SimpleRegisterAccess::new();
//! processor.process(&packet, &mut access).unwrap();
//! ```

use std::collections::VecDeque;
use std::fmt;

use super::parser::{ControlPacket, CtrlOpCode};
use super::response::ControlPacketResponse;
use super::MaskWriteOp;

/// Trait for register access operations.
///
/// Implementors provide read and write access to tile registers. This
/// abstraction allows the control packet processor to work with any
/// register backend (real tile state, mock for testing, etc.).
pub trait RegisterAccess {
    /// Read a 32-bit register at the given tile-local offset.
    ///
    /// Returns the register value, or an error if the address is invalid
    /// or the read is not permitted.
    fn read_register(&self, offset: u32) -> Result<u32, String>;

    /// Write a 32-bit value to a register at the given tile-local offset.
    ///
    /// Returns an error if the address is invalid or the write is not
    /// permitted.
    fn write_register(&mut self, offset: u32, value: u32) -> Result<(), String>;
}

/// Errors from control packet processing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessError {
    /// A register read failed during OP_READ processing.
    ReadFailed {
        offset: u32,
        reason: String,
    },
    /// A register write failed during OP_WRITE/BLOCK_WRITE/WRITE_INCR.
    WriteFailed {
        offset: u32,
        value: u32,
        reason: String,
    },
    /// A register read failed during MaskWrite (read phase).
    MaskWriteReadFailed {
        offset: u32,
        reason: String,
    },
    /// A register write failed during MaskWrite (write phase).
    MaskWriteWriteFailed {
        offset: u32,
        new_value: u32,
        reason: String,
    },
}

impl fmt::Display for ProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReadFailed { offset, reason } => {
                write!(f, "register read failed at 0x{:05X}: {}", offset, reason)
            }
            Self::WriteFailed { offset, value, reason } => {
                write!(f, "register write failed at 0x{:05X} (value=0x{:08X}): {}",
                    offset, value, reason)
            }
            Self::MaskWriteReadFailed { offset, reason } => {
                write!(f, "mask-write read phase failed at 0x{:05X}: {}", offset, reason)
            }
            Self::MaskWriteWriteFailed { offset, new_value, reason } => {
                write!(f, "mask-write write phase failed at 0x{:05X} (new=0x{:08X}): {}",
                    offset, new_value, reason)
            }
        }
    }
}

impl std::error::Error for ProcessError {}

/// Control packet processor with response queue.
///
/// Processes control packets by dispatching register reads and writes
/// through a [`RegisterAccess`] implementation. OP_READ operations produce
/// response packets queued for retrieval.
///
/// This processor is intentionally stateless with respect to packet
/// reassembly -- it operates on fully parsed [`ControlPacket`] values.
/// The word-by-word reassembly state machine lives in `tile.rs`
/// (`ControlPacketState`).
#[derive(Debug)]
pub struct ControlPacketProcessor {
    /// Pending response packets from OP_READ operations.
    pending_responses: VecDeque<ControlPacketResponse>,
}

impl ControlPacketProcessor {
    /// Create a new processor with an empty response queue.
    pub fn new() -> Self {
        Self {
            pending_responses: VecDeque::new(),
        }
    }

    /// Process a control packet, performing register operations via the
    /// provided access callbacks.
    ///
    /// For write operations (Write, WriteIncr, BlockWrite), each data word
    /// is written to consecutive register addresses starting from the
    /// packet's base address.
    ///
    /// For read operations, register values are read and a response packet
    /// is queued. Retrieve it with [`pop_response`](Self::pop_response).
    ///
    /// Returns a list of (address, value) pairs that were written, for
    /// callers that need to track writes (e.g., for BD dirty tracking).
    pub fn process(
        &mut self,
        packet: &ControlPacket,
        access: &mut dyn RegisterAccess,
    ) -> Result<Vec<(u32, u32)>, ProcessError> {
        match packet.opcode {
            CtrlOpCode::Write | CtrlOpCode::WriteIncr | CtrlOpCode::BlockWrite => {
                self.process_write(packet, access)
            }
            CtrlOpCode::Read => {
                self.process_read(packet, access)?;
                Ok(Vec::new())
            }
        }
    }

    /// Process a CDO-level mask-write operation.
    ///
    /// This is NOT a control packet opcode. It is provided here for
    /// convenience since the emulator's CDO handler needs mask-write
    /// support and the processor already has the register access pattern.
    ///
    /// Semantics follow aie-rt: `new = (current & ~mask) | value`
    pub fn process_mask_write(
        &mut self,
        op: &MaskWriteOp,
        access: &mut dyn RegisterAccess,
    ) -> Result<(u32, u32), ProcessError> {
        let current = access.read_register(op.address)
            .map_err(|reason| ProcessError::MaskWriteReadFailed {
                offset: op.address,
                reason,
            })?;

        let new_value = op.apply(current);

        access.write_register(op.address, new_value)
            .map_err(|reason| ProcessError::MaskWriteWriteFailed {
                offset: op.address,
                new_value,
                reason,
            })?;

        Ok((op.address, new_value))
    }

    /// Check whether there are pending response packets.
    pub fn has_pending_response(&self) -> bool {
        !self.pending_responses.is_empty()
    }

    /// Remove and return the next pending response packet (FIFO order).
    pub fn pop_response(&mut self) -> Option<ControlPacketResponse> {
        self.pending_responses.pop_front()
    }

    /// Number of pending response packets.
    pub fn pending_count(&self) -> usize {
        self.pending_responses.len()
    }

    /// Clear all pending responses (e.g., on tile reset).
    pub fn clear_pending(&mut self) {
        self.pending_responses.clear();
    }

    // -- Internal helpers --

    /// Process a write-class operation (Write, WriteIncr, BlockWrite).
    ///
    /// All three opcodes write data words to consecutive register addresses.
    /// They differ only in how the hardware handles the stream -- for the
    /// emulator, the behavior is identical since we receive fully assembled
    /// packets.
    fn process_write(
        &self,
        packet: &ControlPacket,
        access: &mut dyn RegisterAccess,
    ) -> Result<Vec<(u32, u32)>, ProcessError> {
        let mut writes = Vec::with_capacity(packet.data.len());

        for (i, &value) in packet.data.iter().enumerate() {
            let addr = packet.address + (i as u32) * 4;
            access.write_register(addr, value)
                .map_err(|reason| ProcessError::WriteFailed {
                    offset: addr,
                    value,
                    reason,
                })?;
            writes.push((addr, value));
        }

        Ok(writes)
    }

    /// Process an OP_READ operation.
    ///
    /// Reads `beats` consecutive registers starting at the packet's address
    /// and queues a response packet.
    fn process_read(
        &mut self,
        packet: &ControlPacket,
        access: &dyn RegisterAccess,
    ) -> Result<(), ProcessError> {
        let mut values = Vec::with_capacity(packet.beats as usize);

        for i in 0..packet.beats {
            let addr = packet.address + (i as u32) * 4;
            let value = access.read_register(addr)
                .map_err(|reason| ProcessError::ReadFailed {
                    offset: addr,
                    reason,
                })?;
            values.push(value);
        }

        let response = ControlPacketResponse::multi(
            packet.address,
            values,
            packet.response_id,
        );

        self.pending_responses.push_back(response);
        Ok(())
    }
}

impl Default for ControlPacketProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Simple hash-map-backed register file for testing.
    struct SimpleRegisterAccess {
        registers: HashMap<u32, u32>,
        /// Track all writes for verification.
        write_log: Vec<(u32, u32)>,
        /// If set, reads from unlisted addresses return this instead of error.
        default_value: Option<u32>,
    }

    impl SimpleRegisterAccess {
        fn new() -> Self {
            Self {
                registers: HashMap::new(),
                write_log: Vec::new(),
                default_value: Some(0),
            }
        }

        fn with_registers(regs: &[(u32, u32)]) -> Self {
            let mut access = Self::new();
            for &(addr, val) in regs {
                access.registers.insert(addr, val);
            }
            access
        }

        /// Create an access that errors on unknown addresses.
        fn strict() -> Self {
            Self {
                registers: HashMap::new(),
                write_log: Vec::new(),
                default_value: None,
            }
        }
    }

    impl RegisterAccess for SimpleRegisterAccess {
        fn read_register(&self, offset: u32) -> Result<u32, String> {
            if let Some(&val) = self.registers.get(&offset) {
                Ok(val)
            } else if let Some(default) = self.default_value {
                Ok(default)
            } else {
                Err(format!("no register at 0x{:05X}", offset))
            }
        }

        fn write_register(&mut self, offset: u32, value: u32) -> Result<(), String> {
            self.registers.insert(offset, value);
            self.write_log.push((offset, value));
            Ok(())
        }
    }

    // -- Processor construction --

    #[test]
    fn new_processor_has_no_pending_responses() {
        let proc = ControlPacketProcessor::new();
        assert!(!proc.has_pending_response());
        assert_eq!(proc.pending_count(), 0);
    }

    #[test]
    fn default_processor_is_empty() {
        let proc = ControlPacketProcessor::default();
        assert!(!proc.has_pending_response());
    }

    // -- OP_WRITE tests --

    #[test]
    fn process_write_single_register() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();
        let pkt = ControlPacket::write(0x1A000, 0xDEAD_BEEF);

        let writes = proc.process(&pkt, &mut access).unwrap();

        assert_eq!(writes, vec![(0x1A000, 0xDEAD_BEEF)]);
        assert_eq!(access.registers[&0x1A000], 0xDEAD_BEEF);
        assert_eq!(access.write_log.len(), 1);
        assert!(!proc.has_pending_response());
    }

    #[test]
    fn process_write_updates_existing_register() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x100, 0x0000_1111)]);

        let pkt = ControlPacket::write(0x100, 0x2222_3333);
        proc.process(&pkt, &mut access).unwrap();

        assert_eq!(access.registers[&0x100], 0x2222_3333);
    }

    // -- OP_READ tests --

    #[test]
    fn process_read_single_register() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x100, 0xCAFE_BABE)]);
        let pkt = ControlPacket::read(0x100, 1, 42);

        let writes = proc.process(&pkt, &mut access).unwrap();

        // Read produces no writes
        assert!(writes.is_empty());

        // But produces a response
        assert!(proc.has_pending_response());
        assert_eq!(proc.pending_count(), 1);

        let resp = proc.pop_response().unwrap();
        assert_eq!(resp.address, 0x100);
        assert_eq!(resp.data, vec![0xCAFE_BABE]);
        assert_eq!(resp.response_id, 42);
    }

    #[test]
    fn process_read_multiple_consecutive_registers() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[
            (0x200, 0x1111_1111),
            (0x204, 0x2222_2222),
            (0x208, 0x3333_3333),
        ]);
        let pkt = ControlPacket::read(0x200, 3, 7);

        proc.process(&pkt, &mut access).unwrap();

        let resp = proc.pop_response().unwrap();
        assert_eq!(resp.data, vec![0x1111_1111, 0x2222_2222, 0x3333_3333]);
        assert_eq!(resp.response_id, 7);
    }

    #[test]
    fn process_read_generates_response_with_correct_address() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new(); // default=0
        let pkt = ControlPacket::read(0x340D0, 1, 0);

        proc.process(&pkt, &mut access).unwrap();

        let resp = proc.pop_response().unwrap();
        assert_eq!(resp.address, 0x340D0);
    }

    // -- OP_BLOCKWRITE tests --

    #[test]
    fn process_blockwrite_multiple_registers() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();
        let pkt = ControlPacket::block_write(0x1000, vec![0xAA, 0xBB, 0xCC]);

        let writes = proc.process(&pkt, &mut access).unwrap();

        assert_eq!(writes, vec![
            (0x1000, 0xAA),
            (0x1004, 0xBB),
            (0x1008, 0xCC),
        ]);
        assert_eq!(access.registers[&0x1000], 0xAA);
        assert_eq!(access.registers[&0x1004], 0xBB);
        assert_eq!(access.registers[&0x1008], 0xCC);
    }

    // -- OP_WRITE_INCR tests --

    #[test]
    fn process_write_incr_consecutive_addresses() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();
        let pkt = ControlPacket::write_incr(0x500, vec![10, 20]);

        let writes = proc.process(&pkt, &mut access).unwrap();

        assert_eq!(writes, vec![(0x500, 10), (0x504, 20)]);
        assert_eq!(access.registers[&0x500], 10);
        assert_eq!(access.registers[&0x504], 20);
    }

    // -- MaskWrite tests --

    #[test]
    fn process_mask_write_basic() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x100, 0xFFFF_FFFF)]);
        let op = MaskWriteOp::new(0x100, 0x0000_00AA, 0x0000_00FF);

        let (addr, new_val) = proc.process_mask_write(&op, &mut access).unwrap();

        assert_eq!(addr, 0x100);
        // (0xFFFF_FFFF & ~0xFF) | 0xAA = 0xFFFF_FF00 | 0xAA = 0xFFFF_FFAA
        assert_eq!(new_val, 0xFFFF_FFAA);
        assert_eq!(access.registers[&0x100], 0xFFFF_FFAA);
    }

    #[test]
    fn process_mask_write_full_mask() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x200, 0xAAAA_BBBB)]);
        let op = MaskWriteOp::new(0x200, 0x1234_5678, 0xFFFF_FFFF);

        let (_, new_val) = proc.process_mask_write(&op, &mut access).unwrap();

        assert_eq!(new_val, 0x1234_5678);
    }

    #[test]
    fn process_mask_write_zero_mask() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x300, 0xDEAD_BEEF)]);
        let op = MaskWriteOp::new(0x300, 0x0000_0000, 0x0000_0000);

        let (_, new_val) = proc.process_mask_write(&op, &mut access).unwrap();

        // No bits modified
        assert_eq!(new_val, 0xDEAD_BEEF);
    }

    #[test]
    fn process_mask_write_alternating() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x400, 0x0000_0000)]);
        let op = MaskWriteOp::new(0x400, 0x5555_5555, 0x5555_5555);

        let (_, new_val) = proc.process_mask_write(&op, &mut access).unwrap();

        assert_eq!(new_val, 0x5555_5555);
    }

    #[test]
    fn process_mask_write_high_nibble_only() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[(0x500, 0x1234_5678)]);
        // Modify only bits 31:16
        let op = MaskWriteOp::new(0x500, 0xABCD_0000, 0xFFFF_0000);

        let (_, new_val) = proc.process_mask_write(&op, &mut access).unwrap();

        assert_eq!(new_val, 0xABCD_5678);
    }

    // -- Error handling tests --

    #[test]
    fn process_write_error_propagated() {
        let mut proc = ControlPacketProcessor::new();
        // Use a wrapper that fails on specific addresses.
        struct FailOnWrite;
        impl RegisterAccess for FailOnWrite {
            fn read_register(&self, _offset: u32) -> Result<u32, String> { Ok(0) }
            fn write_register(&mut self, offset: u32, _value: u32) -> Result<(), String> {
                Err(format!("write blocked at 0x{:05X}", offset))
            }
        }

        let pkt = ControlPacket::write(0x100, 42);
        let err = proc.process(&pkt, &mut FailOnWrite).unwrap_err();
        match err {
            ProcessError::WriteFailed { offset, value, .. } => {
                assert_eq!(offset, 0x100);
                assert_eq!(value, 42);
            }
            _ => panic!("expected WriteFailed, got {:?}", err),
        }
    }

    #[test]
    fn process_read_error_propagated() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::strict(); // errors on unknown addrs
        let pkt = ControlPacket::read(0x999, 1, 0);

        let err = proc.process(&pkt, &mut access).unwrap_err();
        match err {
            ProcessError::ReadFailed { offset, .. } => {
                assert_eq!(offset, 0x999);
            }
            _ => panic!("expected ReadFailed, got {:?}", err),
        }
    }

    #[test]
    fn process_mask_write_read_error() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::strict();
        let op = MaskWriteOp::new(0x888, 0, 0xFF);

        let err = proc.process_mask_write(&op, &mut access).unwrap_err();
        match err {
            ProcessError::MaskWriteReadFailed { offset, .. } => {
                assert_eq!(offset, 0x888);
            }
            _ => panic!("expected MaskWriteReadFailed, got {:?}", err),
        }
    }

    #[test]
    fn process_mask_write_write_error() {
        let mut proc = ControlPacketProcessor::new();
        // Reads succeed but writes fail
        struct ReadOnlyAccess;
        impl RegisterAccess for ReadOnlyAccess {
            fn read_register(&self, _offset: u32) -> Result<u32, String> { Ok(0xFFFF_FFFF) }
            fn write_register(&mut self, _offset: u32, _value: u32) -> Result<(), String> {
                Err("read-only".to_string())
            }
        }

        let op = MaskWriteOp::new(0x100, 0xAA, 0xFF);
        let err = proc.process_mask_write(&op, &mut ReadOnlyAccess).unwrap_err();
        match err {
            ProcessError::MaskWriteWriteFailed { offset, .. } => {
                assert_eq!(offset, 0x100);
            }
            _ => panic!("expected MaskWriteWriteFailed"),
        }
    }

    // -- Response queue ordering tests --

    #[test]
    fn pending_response_fifo_ordering() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[
            (0x100, 0xAAAA),
            (0x200, 0xBBBB),
            (0x300, 0xCCCC),
        ]);

        // Issue three reads
        proc.process(&ControlPacket::read(0x100, 1, 1), &mut access).unwrap();
        proc.process(&ControlPacket::read(0x200, 1, 2), &mut access).unwrap();
        proc.process(&ControlPacket::read(0x300, 1, 3), &mut access).unwrap();

        assert_eq!(proc.pending_count(), 3);

        // Pop in FIFO order
        let r1 = proc.pop_response().unwrap();
        assert_eq!(r1.response_id, 1);
        assert_eq!(r1.data, vec![0xAAAA]);

        let r2 = proc.pop_response().unwrap();
        assert_eq!(r2.response_id, 2);
        assert_eq!(r2.data, vec![0xBBBB]);

        let r3 = proc.pop_response().unwrap();
        assert_eq!(r3.response_id, 3);
        assert_eq!(r3.data, vec![0xCCCC]);

        assert!(!proc.has_pending_response());
        assert!(proc.pop_response().is_none());
    }

    #[test]
    fn pop_response_from_empty_returns_none() {
        let mut proc = ControlPacketProcessor::new();
        assert!(proc.pop_response().is_none());
    }

    #[test]
    fn clear_pending_removes_all() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();

        proc.process(&ControlPacket::read(0x100, 1, 1), &mut access).unwrap();
        proc.process(&ControlPacket::read(0x200, 1, 2), &mut access).unwrap();
        assert_eq!(proc.pending_count(), 2);

        proc.clear_pending();
        assert_eq!(proc.pending_count(), 0);
        assert!(!proc.has_pending_response());
    }

    // -- Multiple operations in sequence --

    #[test]
    fn mixed_operations_in_sequence() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();

        // Write a value
        proc.process(&ControlPacket::write(0x100, 0xAAAA), &mut access).unwrap();
        assert_eq!(access.registers[&0x100], 0xAAAA);

        // Read it back -- should see the written value
        proc.process(&ControlPacket::read(0x100, 1, 5), &mut access).unwrap();
        let resp = proc.pop_response().unwrap();
        assert_eq!(resp.data, vec![0xAAAA]);

        // Block write multiple registers
        proc.process(
            &ControlPacket::block_write(0x200, vec![0x11, 0x22]),
            &mut access,
        ).unwrap();

        // Read them back
        proc.process(&ControlPacket::read(0x200, 2, 10), &mut access).unwrap();
        let resp = proc.pop_response().unwrap();
        assert_eq!(resp.data, vec![0x11, 0x22]);

        // MaskWrite to modify part of a register
        let op = MaskWriteOp::new(0x100, 0x0000_00BB, 0x0000_00FF);
        proc.process_mask_write(&op, &mut access).unwrap();
        assert_eq!(access.registers[&0x100], 0x0000_AABB);
    }

    #[test]
    fn writes_do_not_produce_responses() {
        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::new();

        proc.process(&ControlPacket::write(0x100, 1), &mut access).unwrap();
        proc.process(&ControlPacket::block_write(0x200, vec![2, 3]), &mut access).unwrap();
        proc.process(&ControlPacket::write_incr(0x300, vec![4]), &mut access).unwrap();

        assert!(!proc.has_pending_response());
    }

    // -- Response serialization round-trip --

    #[test]
    fn response_serialization_roundtrip() {
        use crate::device::control_packets::parser::parse_header;

        let mut proc = ControlPacketProcessor::new();
        let mut access = SimpleRegisterAccess::with_registers(&[
            (0x1A000, 0xDEAD_BEEF),
            (0x1A004, 0xCAFE_BABE),
        ]);

        proc.process(&ControlPacket::read(0x1A000, 2, 33), &mut access).unwrap();
        let resp = proc.pop_response().unwrap();

        // Serialize to stream words
        let stream_words = resp.to_stream_words();
        assert_eq!(stream_words.len(), 3); // header + 2 data

        // Parse the response header
        let (header_word, _) = stream_words[0];
        let fields = parse_header(header_word).unwrap();
        assert_eq!(fields.address, 0x1A000);
        assert_eq!(fields.beats, 2);
        assert_eq!(fields.opcode, CtrlOpCode::Read);
        assert_eq!(fields.response_id, 33);

        // Verify data words
        assert_eq!(stream_words[1].0, 0xDEAD_BEEF);
        assert_eq!(stream_words[2].0, 0xCAFE_BABE);

        // Verify tlast flags
        assert!(!stream_words[0].1);
        assert!(!stream_words[1].1);
        assert!(stream_words[2].1);
    }

    // -- ProcessError Display tests --

    #[test]
    fn process_error_display() {
        let e = ProcessError::ReadFailed { offset: 0x100, reason: "nope".into() };
        let s = format!("{}", e);
        assert!(s.contains("0x00100"));
        assert!(s.contains("nope"));

        let e = ProcessError::WriteFailed { offset: 0x200, value: 42, reason: "denied".into() };
        let s = format!("{}", e);
        assert!(s.contains("0x00200"));
        assert!(s.contains("0x0000002A"));

        let e = ProcessError::MaskWriteReadFailed { offset: 0x300, reason: "fail".into() };
        assert!(format!("{}", e).contains("mask-write read"));

        let e = ProcessError::MaskWriteWriteFailed { offset: 0x400, new_value: 0xFF, reason: "ro".into() };
        assert!(format!("{}", e).contains("mask-write write"));
    }

    // -- Edge case: blockwrite partial failure --

    #[test]
    fn blockwrite_stops_on_first_write_error() {
        let mut proc = ControlPacketProcessor::new();

        struct FailAtSecond { count: u32 }
        impl RegisterAccess for FailAtSecond {
            fn read_register(&self, _offset: u32) -> Result<u32, String> { Ok(0) }
            fn write_register(&mut self, offset: u32, _value: u32) -> Result<(), String> {
                self.count += 1;
                if self.count == 2 {
                    Err(format!("blocked at 0x{:05X}", offset))
                } else {
                    Ok(())
                }
            }
        }

        let pkt = ControlPacket::block_write(0x100, vec![0xAA, 0xBB, 0xCC]);
        let err = proc.process(&pkt, &mut FailAtSecond { count: 0 }).unwrap_err();

        match err {
            ProcessError::WriteFailed { offset, value, .. } => {
                assert_eq!(offset, 0x104); // second write
                assert_eq!(value, 0xBB);
            }
            _ => panic!("expected WriteFailed"),
        }
    }
}
