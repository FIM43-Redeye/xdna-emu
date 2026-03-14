//! Control packet parsing, processing, and response generation for AMD AIE2.
//!
//! Control packets are the mechanism for the host (or other tiles) to read/write
//! tile registers remotely via the stream switch network. They arrive on the
//! TileCtrl slave port and trigger register reads or writes.
//!
//! # Packet Format (AM020 Table 3)
//!
//! The control packet header is a single 32-bit word:
//!
//! ```text
//! Bit 31      : Parity
//! Bits 30:24  : Response_ID / Stream_ID (for response routing)
//! Bits 23:22  : Operation (2-bit opcode)
//! Bits 21:20  : Length (00=1 beat, 01=2, 10=3, 11=4)
//! Bits 19:0   : Address (tile-local register offset)
//! ```
//!
//! # Operations
//!
//! The 2-bit operation field encodes four opcodes (derived from
//! `crate::arch::ctrl_packet`):
//!
//! | Value | Operation    | Description                                    |
//! |-------|-------------|------------------------------------------------|
//! | 0     | Write       | Write data words to consecutive registers      |
//! | 1     | Read        | Read register(s), send response back            |
//! | 2     | WriteIncr   | Write with auto-incrementing address            |
//! | 3     | BlockWrite  | Block write to consecutive registers            |
//!
//! Note: MaskWrite (read-modify-write) is a CDO/host-level operation, not a
//! control packet opcode. The 2-bit operation field has no room for it. However,
//! this module provides a [`MaskWriteOp`] type for CDO-level mask-write support
//! since the emulator needs it for `DeviceState::write_tile_register` paths.
//!
//! # Response Packets (for Read)
//!
//! When a Read operation is processed, the tile generates a response packet
//! containing the register value(s). The response goes out the TileCtrl master
//! port and is routed back to the requestor via packet switching.
//!
//! # Lazy BD Parsing
//!
//! When control packets write BD registers word-by-word, the BD should NOT be
//! re-parsed after each word. The [`BdDirtyTracker`] trait defines the interface
//! for marking BDs as dirty and deferring re-parse until the DMA engine needs
//! them.
//!
//! # Deriving from the Toolchain
//!
//! All header field positions are read from `crate::arch::ctrl_packet::*`
//! (generated from xdna-archspec, which derives from AM020/AM025). Operation
//! codes match `OP_WRITE`, `OP_READ`, `OP_WRITE_INCR`, `OP_BLOCK_WRITE` from
//! the same source. MaskWrite semantics (`(old & ~mask) | value`) match the
//! aie-rt `XAie_BaremetalIO_MaskWrite32` implementation.

pub mod parser;
pub mod processor;
pub mod reassembler;
pub mod response;

pub use parser::{ControlPacket, CtrlOpCode, HeaderFields, ParseError, parse_header};
pub use processor::{ControlPacketProcessor, RegisterAccess, ProcessError};
pub use reassembler::{StreamReassembler, ReassembleResult};
pub use response::ControlPacketResponse;

/// Interface for lazy BD re-parsing on word-by-word control packet writes.
///
/// When control packets write BD registers one word at a time, the BD should
/// not be re-parsed after each individual word. Instead, the BD is marked
/// "dirty" and only re-parsed when the DMA engine actually needs the
/// configuration (e.g., when starting a transfer).
///
/// Implementors should track per-BD dirty state and clear it after re-parsing.
pub trait BdDirtyTracker {
    /// Mark a BD as dirty (needs re-parsing before next use).
    ///
    /// `bd_index` is the buffer descriptor index within the tile's DMA engine.
    fn mark_bd_dirty(&mut self, bd_index: usize);

    /// Check whether a BD needs re-parsing.
    fn is_bd_dirty(&self, bd_index: usize) -> bool;

    /// Clear the dirty flag after re-parsing.
    fn clear_bd_dirty(&mut self, bd_index: usize);
}

/// CDO-level mask-write operation.
///
/// This is NOT a control packet opcode (the 2-bit field has no room for it).
/// MaskWrite is a host-level CDO operation that performs a read-modify-write.
///
/// The aie-rt implementation (`XAie_BaremetalIO_MaskWrite32`) uses:
/// ```text
/// new_value = (current & ~mask) | value
/// ```
/// where the caller pre-masks `value` to the relevant field bits.
///
/// This module also provides [`MaskWriteOp::apply_masked`] for cases where
/// the caller has NOT pre-masked the value:
/// ```text
/// new_value = (current & ~mask) | (value & mask)
/// ```
///
/// Source: aie-rt `driver/src/io_backend/ext/xaie_baremetal.c:242-257`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaskWriteOp {
    /// Register address (tile-local offset).
    pub address: u32,
    /// Value to write (OR'd into the cleared mask bits).
    pub value: u32,
    /// Mask indicating which bits to modify.
    pub mask: u32,
}

impl MaskWriteOp {
    /// Create a new mask-write operation.
    pub fn new(address: u32, value: u32, mask: u32) -> Self {
        Self { address, value, mask }
    }

    /// Apply the mask-write to a current register value.
    ///
    /// Follows aie-rt semantics: `(current & ~mask) | value`
    ///
    /// The caller is responsible for pre-masking `value` if needed.
    /// aie-rt callers always pass `value` already masked to the relevant
    /// field bits, so `(current & ~mask) | value` is correct.
    pub fn apply(&self, current: u32) -> u32 {
        (current & !self.mask) | self.value
    }

    /// Apply with explicit value masking (defensive variant).
    ///
    /// Computes `(current & ~mask) | (value & mask)`, ensuring stray bits
    /// in `value` outside the mask are ignored. Use this when the caller
    /// may not have pre-masked the value.
    pub fn apply_masked(&self, current: u32) -> u32 {
        (current & !self.mask) | (self.value & self.mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- MaskWriteOp tests --

    #[test]
    fn mask_write_apply_aiert_semantics() {
        // aie-rt: (current & ~mask) | value
        let op = MaskWriteOp::new(0x100, 0x0000_00FF, 0x0000_00FF);
        assert_eq!(op.apply(0xDEAD_BEEF), 0xDEAD_BEFF);
    }

    #[test]
    fn mask_write_apply_masked_defensive() {
        // (current & ~mask) | (value & mask)
        // value has stray bits outside mask -- they should be ignored
        let op = MaskWriteOp::new(0x100, 0xFFFF_FFFF, 0x0000_00FF);
        assert_eq!(op.apply_masked(0xDEAD_BE00), 0xDEAD_BEFF);
        // apply() would let stray bits bleed through
        assert_eq!(op.apply(0xDEAD_BE00), 0xFFFF_FFFF);
    }

    #[test]
    fn mask_write_all_ones_mask() {
        let op = MaskWriteOp::new(0x200, 0x1234_5678, 0xFFFF_FFFF);
        assert_eq!(op.apply(0xAAAA_BBBB), 0x1234_5678);
        assert_eq!(op.apply_masked(0xAAAA_BBBB), 0x1234_5678);
    }

    #[test]
    fn mask_write_all_zeros_mask() {
        let op = MaskWriteOp::new(0x200, 0x0000_0000, 0x0000_0000);
        // No bits modified
        assert_eq!(op.apply(0xAAAA_BBBB), 0xAAAA_BBBB);
        assert_eq!(op.apply_masked(0xAAAA_BBBB), 0xAAAA_BBBB);
    }

    #[test]
    fn mask_write_all_zeros_mask_with_nonzero_value() {
        // Edge case: mask=0 but value has bits set.
        // apply: those bits leak through (aie-rt caller error)
        // apply_masked: value & 0 = 0, so current is preserved (defensive)
        let op = MaskWriteOp::new(0x200, 0x1234_5678, 0x0000_0000);
        assert_eq!(op.apply(0xAAAA_BBBB), 0xAAAA_BBBB | 0x1234_5678);
        assert_eq!(op.apply_masked(0xAAAA_BBBB), 0xAAAA_BBBB);
    }

    #[test]
    fn mask_write_alternating_mask() {
        let op = MaskWriteOp::new(0x300, 0x5555_5555, 0x5555_5555);
        let current = 0xAAAA_AAAA;
        // Even bits cleared by ~mask, odd bits set by value
        assert_eq!(op.apply(current), 0xFFFF_FFFF);
    }

    #[test]
    fn mask_write_preserves_unmasked_bits() {
        // Only modify bits 15:8
        let op = MaskWriteOp::new(0x400, 0x0000_AB00, 0x0000_FF00);
        assert_eq!(op.apply(0x1234_5678), 0x1234_AB78);
    }

    // -- BdDirtyTracker tests --

    struct TestBdTracker {
        dirty: [bool; 16],
    }

    impl TestBdTracker {
        fn new() -> Self {
            Self { dirty: [false; 16] }
        }
    }

    impl BdDirtyTracker for TestBdTracker {
        fn mark_bd_dirty(&mut self, bd_index: usize) {
            if bd_index < self.dirty.len() {
                self.dirty[bd_index] = true;
            }
        }

        fn is_bd_dirty(&self, bd_index: usize) -> bool {
            bd_index < self.dirty.len() && self.dirty[bd_index]
        }

        fn clear_bd_dirty(&mut self, bd_index: usize) {
            if bd_index < self.dirty.len() {
                self.dirty[bd_index] = false;
            }
        }
    }

    #[test]
    fn bd_dirty_tracker_mark_and_clear() {
        let mut tracker = TestBdTracker::new();
        assert!(!tracker.is_bd_dirty(0));
        tracker.mark_bd_dirty(0);
        assert!(tracker.is_bd_dirty(0));
        assert!(!tracker.is_bd_dirty(1));
        tracker.clear_bd_dirty(0);
        assert!(!tracker.is_bd_dirty(0));
    }

    #[test]
    fn bd_dirty_tracker_multiple_bds() {
        let mut tracker = TestBdTracker::new();
        tracker.mark_bd_dirty(3);
        tracker.mark_bd_dirty(7);
        tracker.mark_bd_dirty(15);
        assert!(tracker.is_bd_dirty(3));
        assert!(tracker.is_bd_dirty(7));
        assert!(tracker.is_bd_dirty(15));
        assert!(!tracker.is_bd_dirty(0));
        assert!(!tracker.is_bd_dirty(8));
        tracker.clear_bd_dirty(7);
        assert!(!tracker.is_bd_dirty(7));
        assert!(tracker.is_bd_dirty(3));
    }

    #[test]
    fn bd_dirty_tracker_out_of_range() {
        let mut tracker = TestBdTracker::new();
        // Should not panic on out-of-range indices
        tracker.mark_bd_dirty(999);
        assert!(!tracker.is_bd_dirty(999));
        tracker.clear_bd_dirty(999);
    }
}
