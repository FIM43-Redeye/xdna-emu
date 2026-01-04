//! Stream operations execution unit.
//!
//! Handles stream read/write operations for data movement between cores and
//! the stream fabric. AIE2 uses streams for tile-to-tile communication via
//! circuit-switched and packet-switched routing.
//!
//! # Operations
//!
//! - **StreamWriteScalar**: Write scalar register value to master stream port
//! - **StreamWritePacketHeader**: Write packet header to master stream port
//! - **StreamReadScalar**: Read from slave stream port into scalar register
//!
//! # Stream Architecture
//!
//! Each AIE2 tile has stream switch ports:
//! - Master ports: Output streams (data leaves the tile)
//! - Slave ports: Input streams (data enters the tile)
//!
//! Stream operations move data between scalar registers and stream ports.
//! The actual routing is configured by CDO commands via the stream switch.
//!
//! # Blocking Behavior
//!
//! Stream operations can be blocking or non-blocking:
//! - Blocking writes wait until the stream has capacity
//! - Blocking reads wait until data is available
//! - Non-blocking operations return immediately (may drop data or read stale)
//!
//! For now, this module implements stubs that log operations. Full stream
//! buffer integration requires wiring up the StreamSwitch from the Tile.

use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operation, Operand, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::ExecuteResult;

/// Stream operations execution unit.
///
/// Handles read/write operations between scalar registers and stream ports.
/// This unit requires access to the Tile's stream switch for buffer operations.
pub struct StreamOps;

/// Result of stream operation execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamResult {
    /// Operation completed successfully.
    Completed,
    /// Operation needs to stall (blocking read with empty buffer).
    Stall { port: u8 },
    /// Not a stream operation.
    NotStreamOp,
}

impl StreamOps {
    /// Execute a stream operation.
    ///
    /// Returns:
    /// - `StreamResult::Completed` if the operation was handled successfully
    /// - `StreamResult::Stall { port }` if blocking on empty stream
    /// - `StreamResult::NotStreamOp` if not a stream operation
    ///
    /// # Arguments
    ///
    /// * `op` - The slot operation to execute
    /// * `ctx` - Execution context with register files
    /// * `tile` - The tile containing stream buffers
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> StreamResult {
        match &op.op {
            Operation::StreamWriteScalar { blocking } => {
                // Get the source value from the scalar register
                let value = Self::get_source_value(op, ctx);

                // Log the operation for debugging/tracing
                #[cfg(any(test, debug_assertions))]
                eprintln!(
                    "[STREAM] WriteScalar: value=0x{:08X} blocking={} (tile {},{}) - STUB",
                    value, blocking, tile.col, tile.row
                );

                // TODO: Push value to tile's stream output buffer
                // This requires wiring up StreamSwitch from the Tile
                // For now, the operation succeeds but data is discarded
                let _ = (value, blocking, tile);

                StreamResult::Completed
            }

            Operation::StreamWritePacketHeader { blocking } => {
                // Get the header value from the scalar register
                let header = Self::get_source_value(op, ctx);

                // Log the operation for debugging/tracing
                #[cfg(any(test, debug_assertions))]
                eprintln!(
                    "[STREAM] WritePacketHeader: header=0x{:08X} blocking={} (tile {},{}) - STUB",
                    header, blocking, tile.col, tile.row
                );

                // TODO: Push packet header to tile's stream output buffer
                // Packet headers contain routing information for packet-switched mode
                let _ = (header, blocking, tile);

                StreamResult::Completed
            }

            Operation::StreamReadScalar { blocking } => {
                // Get the stream port from the operation (default to port 0)
                // TODO: Parse port from instruction encoding
                let port = 0u8;

                // Try to pop value from tile's stream input buffer
                if let Some(value) = tile.pop_stream_input(port) {
                    // Data available - write to destination register
                    Self::write_dest(op, ctx, value);
                    StreamResult::Completed
                } else if *blocking {
                    // No data and blocking - signal stall
                    // The interpreter will retry this instruction
                    StreamResult::Stall { port }
                } else {
                    // No data but non-blocking - write 0 and continue
                    Self::write_dest(op, ctx, 0);
                    StreamResult::Completed
                }
            }

            _ => StreamResult::NotStreamOp,
        }
    }

    /// Get the source value for a stream write operation.
    ///
    /// Stream writes typically take a single source operand (scalar register).
    fn get_source_value(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        op.sources
            .first()
            .map_or(0, |src| Self::read_operand(src, ctx))
    }

    /// Read an operand value from the execution context.
    fn read_operand(operand: &Operand, ctx: &ExecutionContext) -> u32 {
        match operand {
            Operand::ScalarReg(r) => ctx.scalar.read(*r),
            Operand::PointerReg(r) => ctx.pointer.read(*r),
            Operand::ModifierReg(r) => ctx.modifier.read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0, // Other operand types not valid for stream operations
        }
    }

    /// Write result to destination operand.
    ///
    /// Stream reads write to a scalar register destination.
    fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u32) {
        if let Some(dest) = &op.dest {
            match dest {
                Operand::ScalarReg(r) => ctx.scalar.write(*r, value),
                Operand::PointerReg(r) => ctx.pointer.write(*r, value),
                Operand::ModifierReg(r) => ctx.modifier.write(*r, value),
                _ => {} // Ignore invalid destinations
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    #[test]
    fn test_stream_write_scalar_handled() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(0, 0xDEADBEEF);

        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamWriteScalar { blocking: true },
        )
        .with_source(Operand::ScalarReg(0));

        // Should return Completed even though it's a stub
        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }

    #[test]
    fn test_stream_write_packet_header_handled() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(1, 0x12345678);

        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamWritePacketHeader { blocking: false },
        )
        .with_source(Operand::ScalarReg(1));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }

    #[test]
    fn test_stream_read_with_data() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Push data into the stream input buffer
        tile.push_stream_input(0, 0xCAFEBABE);

        ctx.scalar.write(5, 0xFFFFFFFF);

        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamReadScalar { blocking: true },
        )
        .with_dest(Operand::ScalarReg(5));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );

        // Should have read the actual data
        assert_eq!(ctx.scalar.read(5), 0xCAFEBABE);
    }

    #[test]
    fn test_stream_read_blocking_stall() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // No data in buffer - should stall
        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamReadScalar { blocking: true },
        )
        .with_dest(Operand::ScalarReg(5));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Stall { port: 0 }
        );
    }

    #[test]
    fn test_stream_read_nonblocking_empty() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(5, 0xFFFFFFFF);

        // Non-blocking read with no data should return 0
        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamReadScalar { blocking: false },
        )
        .with_dest(Operand::ScalarReg(5));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );

        // Non-blocking returns 0 when empty
        assert_eq!(ctx.scalar.read(5), 0);
    }

    #[test]
    fn test_non_stream_op_not_handled() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // A scalar add should not be handled by StreamOps
        let op = SlotOp::new(SlotIndex::Scalar0, Operation::ScalarAdd)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::NotStreamOp
        );
    }

    #[test]
    fn test_stream_write_with_immediate_source() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Edge case: immediate as source (unusual but should work)
        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamWriteScalar { blocking: true },
        )
        .with_source(Operand::Immediate(42));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }

    #[test]
    fn test_stream_read_without_dest() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Push data - even without dest, should read and discard
        tile.push_stream_input(0, 0x12345678);

        let op = SlotOp::new(
            SlotIndex::Scalar0,
            Operation::StreamReadScalar { blocking: false },
        );

        // Should return Completed (data consumed but discarded)
        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }
}
