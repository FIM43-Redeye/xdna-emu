//! Stream operations execution unit.
//!
//! Handles stream read/write operations for data movement between cores and
//! the stream fabric. AIE2 uses streams for tile-to-tile communication via
//! circuit-switched and packet-switched routing.
//!
//! # Architecture Note
//!
//! Like [`MemoryUnit`](super::MemoryUnit) and [`ControlUnit`](super::ControlUnit),
//! stream operations are NOT candidates for semantic dispatch - they interact
//! with the tile's stream switch hardware rather than just computing values.
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
//! Stream operations push/pop data to/from the Tile's stream buffers.
//! The TileArray routes this data through the stream switch fabric.

use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, SlotOp};
use crate::tablegen::SemanticOp;
use crate::interpreter::state::ExecutionContext;

use super::semantic::{read_operand, write_dest};

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
        let blocking = op.blocking;
        match op.semantic {
            Some(SemanticOp::StreamWrite) => {
                // Get the source value from the scalar register
                let value = Self::get_source_value(op, ctx);

                // Get the port from the second source operand if present, otherwise use port 0
                // AIE2 stream ops typically encode the port as an immediate operand
                let port = Self::get_port_from_operands(op);

                log::info!(
                    "[STREAM] WriteScalar: value=0x{:08X} port={} blocking={} (tile {},{})",
                    value, port, blocking, tile.col, tile.row
                );

                // Hardware stream output FIFO is 4 words deep per port.
                // If full and blocking, stall until the router drains it.
                const STREAM_FIFO_DEPTH: usize = 4;
                if blocking && tile.stream_output_len(port) >= STREAM_FIFO_DEPTH {
                    log::debug!(
                        "[STREAM] Write stall: port {} FIFO full (tile {},{}, pc=0x{:X})",
                        port, tile.col, tile.row, ctx.pc()
                    );
                    return StreamResult::Stall { port };
                }

                tile.push_stream_output(port, value);
                StreamResult::Completed
            }

            Some(SemanticOp::StreamWritePacketHeader) => {
                // Get the header value from the scalar register
                let header = Self::get_source_value(op, ctx);

                // Get the port from operands
                let port = Self::get_port_from_operands(op);

                log::info!(
                    "[STREAM] WritePacketHeader: header=0x{:08X} port={} blocking={} (tile {},{})",
                    header, port, blocking, tile.col, tile.row
                );

                // Hardware stream output FIFO is 4 words deep per port.
                const STREAM_FIFO_DEPTH: usize = 4;
                if blocking && tile.stream_output_len(port) >= STREAM_FIFO_DEPTH {
                    log::debug!(
                        "[STREAM] PacketHeader stall: port {} FIFO full (tile {},{}, pc=0x{:X})",
                        port, tile.col, tile.row, ctx.pc()
                    );
                    return StreamResult::Stall { port };
                }

                tile.push_stream_output(port, header);
                StreamResult::Completed
            }

            Some(SemanticOp::StreamRead) => {
                // Get the stream port from operands (default to port 0)
                let port = Self::get_port_from_operands(op);

                // Try to pop value from tile's stream input buffer
                if let Some(value) = tile.pop_stream_input(port) {
                    // Data available - write to destination register
                    write_dest(op, ctx, value);
                    StreamResult::Completed
                } else if blocking {
                    // No data and blocking - signal stall
                    // The interpreter will retry this instruction
                    StreamResult::Stall { port }
                } else {
                    // No data but non-blocking - write 0 and continue
                    log::debug!(
                        "[STREAM] Non-blocking read: port {} empty, writing 0 (tile {},{})",
                        port, tile.col, tile.row
                    );
                    write_dest(op, ctx, 0);
                    StreamResult::Completed
                }
            }

            _ => StreamResult::NotStreamOp,
        }
    }

    /// Pre-check whether this op would stall, without any side effects.
    ///
    /// VLIW bundles execute multiple slots in parallel and must commit
    /// atomically: either every slot completes or none does. If a stream
    /// op stalls mid-bundle, sibling slots that already executed would
    /// double-commit when the bundle is retried next cycle. Callers must
    /// pre-check every stream slot and bail out before executing any
    /// slot if one would block.
    ///
    /// Returns `Some(StreamResult::Stall)` if the op would stall, or
    /// `None` if the op is non-stream / non-blocking / would proceed.
    pub fn would_stall(op: &SlotOp, tile: &Tile) -> Option<StreamResult> {
        // Non-blocking ops never stall.
        if !op.blocking {
            return None;
        }
        const STREAM_FIFO_DEPTH: usize = 4;
        match op.semantic {
            Some(SemanticOp::StreamWrite) | Some(SemanticOp::StreamWritePacketHeader) => {
                let port = Self::get_port_from_operands(op);
                if tile.stream_output_len(port) >= STREAM_FIFO_DEPTH {
                    Some(StreamResult::Stall { port })
                } else {
                    None
                }
            }
            Some(SemanticOp::StreamRead) => {
                let port = Self::get_port_from_operands(op);
                if tile.stream_input_len(port) == 0 {
                    Some(StreamResult::Stall { port })
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Get the source value for a stream write operation.
    ///
    /// Stream writes typically take a single source operand (scalar register).
    fn get_source_value(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        op.sources
            .first()
            .map_or(0, |src| read_operand(src, ctx))
    }

    /// Get the stream port from operands.
    ///
    /// AIE2 stream operations may encode the port as:
    /// - An immediate operand (second source)
    /// - Part of the instruction encoding (in which case decoder should set it)
    /// - Default to port 0 if not specified
    fn get_port_from_operands(op: &SlotOp) -> u8 {
        // Check for a second source that's an immediate (port number)
        if op.sources.len() >= 2 {
            if let Operand::Immediate(port) = &op.sources[1] {
                return (*port as u8) & 0x7; // Ports 0-7
            }
        }
        // Default to port 0 (core port)
        0
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use crate::tablegen::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    #[test]
    fn test_stream_write_scalar_pushes_data() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(0, 0xDEADBEEF);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWrite)
            .with_blocking(true)
            .with_source(Operand::ScalarReg(0));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );

        // Verify data was actually pushed to stream output
        assert_eq!(tile.pop_stream_output(0), Some(0xDEADBEEF));
    }

    #[test]
    fn test_stream_write_scalar_with_port() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(0, 0x12345678);

        // Write to port 3 using immediate operand
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWrite)
            .with_blocking(true)
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::Immediate(3)); // Port 3

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );

        // Verify data went to port 3, not port 0
        assert_eq!(tile.pop_stream_output(0), None);
        assert_eq!(tile.pop_stream_output(3), Some(0x12345678));
    }

    #[test]
    fn test_stream_write_packet_header_pushes_data() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(1, 0x12345678);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWritePacketHeader)
            .with_blocking(false)
            .with_source(Operand::ScalarReg(1));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );

        // Verify packet header was pushed to stream output
        assert_eq!(tile.pop_stream_output(0), Some(0x12345678));
    }

    #[test]
    fn test_stream_read_with_data() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Push data into the stream input buffer
        tile.push_stream_input(0, 0xCAFEBABE);

        ctx.scalar.write(5, 0xFFFFFFFF);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamRead)
            .with_blocking(true)
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
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamRead)
            .with_blocking(true)
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
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamRead)
            .with_blocking(false)
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
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
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
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWrite)
            .with_blocking(true)
            .with_source(Operand::Immediate(42));

        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }

    #[test]
    fn test_stream_write_blocking_stalls_when_fifo_full() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(0, 0x11111111);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWrite)
            .with_blocking(true)
            .with_source(Operand::ScalarReg(0));

        // Fill the FIFO (depth 4)
        for _ in 0..4 {
            tile.push_stream_output(0, 0xAAAA);
        }

        // 5th write should stall
        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Stall { port: 0 }
        );
    }

    #[test]
    fn test_stream_write_nonblocking_always_succeeds() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.scalar.write(0, 0x22222222);

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamWrite)
            .with_blocking(false)
            .with_source(Operand::ScalarReg(0));

        // Fill the FIFO (depth 4)
        for _ in 0..4 {
            tile.push_stream_output(0, 0xBBBB);
        }

        // Non-blocking write should still succeed (unbounded for non-blocking)
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

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::StreamRead)
            .with_blocking(false);

        // Should return Completed (data consumed but discarded)
        assert_eq!(
            StreamOps::execute(&op, &mut ctx, &mut tile),
            StreamResult::Completed
        );
    }
}
