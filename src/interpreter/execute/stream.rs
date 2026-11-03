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
//! - **StreamWrite** (`mov ms, r`, `.tlast`, `.nb`): the register's word, with TLAST for the
//!   `.tlast` forms.
//! - **StreamWritePacketHeader** (`mov.ph`, `mov.cph`): the *hardware-built* header word, not a
//!   register copy: [`core_packet_header`] / [`core_control_header`] from the operands and the
//!   writing tile's coordinates.
//! - **StreamRead** (`mov r, SS`, `.nb`): the next input beat into the scalar register
//!   ([`SS_READ_DEST_LATENCY`] cycles later) and the read status `srSS0`.
//!
//! All of them use the core port (port 0). Blocking forms stall; `.nb` forms never do, and
//! leave their outcome in the status registers (`srMS0` for writes, `srSS0` for reads).
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

use crate::device::control_packets::parser::core_control_header;
use crate::device::stream_switch::core_packet_header;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, SlotOp};
use xdna_archspec::aie2::isa::SemanticOp;
use crate::interpreter::state::{ExecutionContext, MS_STATUS_REG_INDEX, SS_STATUS_REG_INDEX};

use super::semantic::read_operand;

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

/// The core's stream port (the only one a `mov ms` / `mov SS` instruction addresses).
const CORE_PORT: u8 = 0;

/// Words the core's output FIFO holds before a blocking write stalls (a non-blocking write
/// is refused instead). The depth is a modelling choice, not a measurement.
const STREAM_FIFO_DEPTH: usize = 4;

/// Cycles from a `mov r, SS` issue to its destination register (II_MOV_SS, operand 0 in
/// AIE2Schedule.td).
const SS_READ_DEST_LATENCY: u64 = 7;
/// Cycles from a `mov r, SS` issue to `srSS0` (II_MOV_SS, `def:srSS0`).
const SS_STATUS_LATENCY: u64 = 8;
/// Cycles from a non-blocking write's issue to `srMS0` (II_ST_MS_NB, `def:srMS0`).
const MS_STATUS_LATENCY: u64 = 3;

/// `srSS0` bit 1: the read took a word.
const SS_VALID: u32 = 0b10;
/// `srSS0` bit 0: the word carried TLAST.
const SS_TLAST: u32 = 0b01;
/// `srMS0` bit 0: the non-blocking write was accepted.
const MS_ACCEPTED: u32 = 1;

/// What a stream write puts on the wire, by operand shape (`mov ms, src`; `mov.ph ms, id,
/// type`; `mov.cph ms, addr, nw, op, id`).
enum WriteForm {
    Scalar,
    PacketHeader,
    ControlHeader,
}

impl StreamOps {
    /// Execute a stream operation.
    ///
    /// Returns:
    /// - `StreamResult::Completed` if the operation was handled successfully
    /// - `StreamResult::Stall { port }` if blocking on a full output or empty input
    /// - `StreamResult::NotStreamOp` if not a stream operation
    ///
    /// # Arguments
    ///
    /// * `op` - The slot operation to execute
    /// * `ctx` - Execution context with register files
    /// * `tile` - The tile containing stream buffers
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> StreamResult {
        match op.semantic {
            Some(SemanticOp::StreamWrite) | Some(SemanticOp::StreamWritePacketHeader) => {
                Self::execute_write(op, ctx, tile)
            }
            Some(SemanticOp::StreamRead) => Self::execute_read(op, ctx, tile),
            _ => StreamResult::NotStreamOp,
        }
    }

    fn write_form(op: &SlotOp) -> WriteForm {
        match (op.semantic, op.sources.len()) {
            (Some(SemanticOp::StreamWritePacketHeader), 4) => WriteForm::ControlHeader,
            (Some(SemanticOp::StreamWritePacketHeader), _) => WriteForm::PacketHeader,
            _ => WriteForm::Scalar,
        }
    }

    /// The word the write puts on the stream.
    fn write_word(op: &SlotOp, ctx: &ExecutionContext, tile: &Tile) -> u32 {
        let src = |i: usize| op.sources.get(i).map_or(0, |s| read_operand(s, ctx));
        match Self::write_form(op) {
            WriteForm::Scalar => src(0),
            WriteForm::PacketHeader => core_packet_header(src(0), src(1), tile.col, tile.row),
            WriteForm::ControlHeader => core_control_header(src(0), src(1), src(2), src(3)),
        }
    }

    fn execute_write(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> StreamResult {
        let full = tile.stream_output_len(CORE_PORT) >= STREAM_FIFO_DEPTH;
        if full && op.blocking {
            log::debug!(
                "[STREAM] Write stall: FIFO full (tile {},{}, pc=0x{:X})",
                tile.col,
                tile.row,
                ctx.pc()
            );
            return StreamResult::Stall { port: CORE_PORT };
        }
        let accepted = !full;
        if accepted {
            let word = Self::write_word(op, ctx, tile);
            log::info!(
                "[STREAM] Write: word=0x{:08X} tlast={} blocking={} (tile {},{})",
                word,
                op.tlast,
                op.blocking,
                tile.col,
                tile.row
            );
            tile.push_stream_output_beat(CORE_PORT, word, op.tlast);
        }
        if !op.blocking {
            let status = if accepted { MS_ACCEPTED } else { 0 };
            ctx.queue_scalar_load(Operand::ScalarReg(MS_STATUS_REG_INDEX), status, MS_STATUS_LATENCY);
        }
        StreamResult::Completed
    }

    fn execute_read(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> StreamResult {
        let (word, status) = match tile.pop_stream_input_beat(CORE_PORT) {
            Some((word, tlast)) => {
                ctx.set_stream_read_latch(word, tlast);
                (word, SS_VALID | if tlast { SS_TLAST } else { 0 })
            }
            None if op.blocking => return StreamResult::Stall { port: CORE_PORT },
            None => {
                // Silicon: an empty non-blocking read hands back the last word again, valid
                // bit clear, TLAST bit as that word had it.
                let (word, tlast) = ctx.stream_read_latch();
                log::debug!("[STREAM] Non-blocking read: empty (tile {},{})", tile.col, tile.row);
                (word, if tlast { SS_TLAST } else { 0 })
            }
        };
        if let Some(dest) = &op.dest {
            ctx.queue_scalar_load(dest.clone(), word, SS_READ_DEST_LATENCY);
        }
        ctx.queue_scalar_load(Operand::ScalarReg(SS_STATUS_REG_INDEX), status, SS_STATUS_LATENCY);
        StreamResult::Completed
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
        let stalls = match op.semantic {
            Some(SemanticOp::StreamWrite) | Some(SemanticOp::StreamWritePacketHeader) => {
                tile.stream_output_len(CORE_PORT) >= STREAM_FIFO_DEPTH
            }
            Some(SemanticOp::StreamRead) => tile.stream_input_len(CORE_PORT) == 0,
            _ => false,
        };
        stalls.then_some(StreamResult::Stall { port: CORE_PORT })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    /// Physical (1,3), where the NPU1 silicon probes ran their producer.
    fn make_tile() -> Tile {
        Tile::compute(1, 3)
    }

    fn write_op(blocking: bool, tlast: bool, src: u8) -> SlotOp {
        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::StreamWrite)
            .with_blocking(blocking)
            .with_source(Operand::ScalarReg(src));
        op.tlast = tlast;
        op
    }

    /// Run the pipeline forward until every queued write has landed.
    fn settle(ctx: &mut ExecutionContext) {
        ctx.cycles += 16;
        ctx.commit_pending_writes();
    }

    #[test]
    fn write_pushes_the_register_word_without_tlast() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        ctx.scalar.write(2, 0xDEADBEEF);
        assert_eq!(
            StreamOps::execute(&write_op(true, false, 2), &mut ctx, &mut tile),
            StreamResult::Completed
        );
        assert_eq!(tile.pop_stream_output_beat(0), Some((0xDEADBEEF, false)));
    }

    #[test]
    fn tlast_form_marks_the_beat() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        ctx.scalar.write(2, 7);
        StreamOps::execute(&write_op(true, true, 2), &mut ctx, &mut tile);
        assert_eq!(tile.pop_stream_output_beat(0), Some((7, true)));
    }

    #[test]
    fn blocking_write_stalls_on_a_full_fifo_and_nb_drops_with_status_zero() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        ctx.scalar.write(0, 0x11111111);
        for _ in 0..STREAM_FIFO_DEPTH {
            tile.push_stream_output(0, 0xAAAA);
        }
        let blocking = write_op(true, false, 0);
        assert_eq!(StreamOps::execute(&blocking, &mut ctx, &mut tile), StreamResult::Stall { port: 0 });
        assert!(StreamOps::would_stall(&blocking, &tile).is_some());

        let nb = write_op(false, false, 0);
        assert!(StreamOps::would_stall(&nb, &tile).is_none());
        assert_eq!(StreamOps::execute(&nb, &mut ctx, &mut tile), StreamResult::Completed);
        assert_eq!(tile.stream_output_len(0), STREAM_FIFO_DEPTH, "the refused word is dropped");
        ctx.scalar.write(MS_STATUS_REG_INDEX, 1);
        settle(&mut ctx);
        assert_eq!(ctx.scalar.read(MS_STATUS_REG_INDEX), 0, "srMS0 reports the refusal");
    }

    #[test]
    fn nb_write_status_lands_after_its_latency_and_reports_acceptance() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        StreamOps::execute(&write_op(false, false, 0), &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(MS_STATUS_REG_INDEX), 0, "not yet");
        ctx.cycles += MS_STATUS_LATENCY - 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(MS_STATUS_REG_INDEX), 0, "one cycle early");
        ctx.cycles += 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(MS_STATUS_REG_INDEX), 1);
    }

    #[test]
    fn packet_header_is_built_from_id_type_and_the_tile() {
        // NPU1 silicon, producer at physical (1,3): mov.ph with r = 0xab, #1 -> 0x0023100b
        // (id masked to 5 bits); r = 0x20, #4 -> 0x80234000 (id 0, parity bit set).
        for (reg, ptype, expect) in
            [(0xabu32, 1, 0x0023_100bu32), (0x20, 4, 0x8023_4000), (0xffff_ffff, 7, 0x0023_701f)]
        {
            let (mut ctx, mut tile) = (make_ctx(), make_tile());
            ctx.scalar.write(1, reg);
            let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::StreamWritePacketHeader)
                .with_blocking(true)
                .with_source(Operand::ScalarReg(1))
                .with_source(Operand::Immediate(ptype));
            StreamOps::execute(&op, &mut ctx, &mut tile);
            assert_eq!(tile.pop_stream_output_beat(0), Some((expect, false)), "r={reg:#x} type={ptype}");
        }
    }

    #[test]
    fn control_packet_header_carries_address_length_op_and_id() {
        // NPU1 silicon: m = 0x123, r = 3, nw 2, op 2 -> 0x83a00123; m = 0xfedcba98, r = -1,
        // nw 3, op 3 -> 0x1ffcba98 (20-bit address, 5-bit id).
        for (m, r, nw, opc, expect) in
            [(0x123u32, 3u32, 2, 2, 0x83a0_0123u32), (0xfedc_ba98, 0xffff_ffff, 3, 3, 0x1ffc_ba98)]
        {
            let (mut ctx, mut tile) = (make_ctx(), make_tile());
            ctx.modifier.write(1, m);
            ctx.scalar.write(2, r);
            let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::StreamWritePacketHeader)
                .with_blocking(true)
                .with_source(Operand::ModifierReg(1))
                .with_source(Operand::Immediate(nw))
                .with_source(Operand::Immediate(opc))
                .with_source(Operand::ScalarReg(2));
            op.tlast = true;
            StreamOps::execute(&op, &mut ctx, &mut tile);
            assert_eq!(tile.pop_stream_output_beat(0), Some((expect, true)));
        }
    }

    fn read_op(blocking: bool, dest: u8) -> SlotOp {
        SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::StreamRead)
            .with_blocking(blocking)
            .with_dest(Operand::ScalarReg(dest))
    }

    #[test]
    fn read_lands_after_seven_cycles_and_status_after_eight() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        tile.push_stream_input_beat(0, 0xCAFEBABE, true);
        ctx.scalar.write(5, 0xFFFF_FFFF);
        assert_eq!(StreamOps::execute(&read_op(true, 5), &mut ctx, &mut tile), StreamResult::Completed);
        ctx.cycles += SS_READ_DEST_LATENCY - 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 0xFFFF_FFFF);
        ctx.cycles += 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(5), 0xCAFEBABE);
        assert_eq!(ctx.scalar.read(SS_STATUS_REG_INDEX), 0, "srSS0 is one cycle behind");
        ctx.cycles += 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.scalar.read(SS_STATUS_REG_INDEX), 0b11, "valid and TLAST");
    }

    #[test]
    fn plain_word_reads_valid_without_tlast() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        tile.push_stream_input(0, 5);
        StreamOps::execute(&read_op(true, 5), &mut ctx, &mut tile);
        settle(&mut ctx);
        assert_eq!(ctx.scalar.read(SS_STATUS_REG_INDEX), 0b10);
    }

    #[test]
    fn blocking_read_of_an_empty_stream_stalls() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        let op = read_op(true, 5);
        assert_eq!(StreamOps::execute(&op, &mut ctx, &mut tile), StreamResult::Stall { port: 0 });
        assert!(StreamOps::would_stall(&op, &tile).is_some());
    }

    #[test]
    fn nb_read_of_an_empty_stream_returns_the_last_word_with_valid_clear() {
        // NPU1 silicon: after a TLAST word, an empty `mov.nb` returns that word again and
        // srSS0 = 1 (TLAST kept, valid clear); after a plain word, 0.
        for (tlast, status) in [(true, 1u32), (false, 0)] {
            let (mut ctx, mut tile) = (make_ctx(), make_tile());
            tile.push_stream_input_beat(0, 0x4444_0004, tlast);
            StreamOps::execute(&read_op(true, 1), &mut ctx, &mut tile);
            settle(&mut ctx);
            ctx.scalar.write(2, 0);
            let nb = read_op(false, 2);
            assert!(StreamOps::would_stall(&nb, &tile).is_none());
            assert_eq!(StreamOps::execute(&nb, &mut ctx, &mut tile), StreamResult::Completed);
            settle(&mut ctx);
            assert_eq!(ctx.scalar.read(2), 0x4444_0004);
            assert_eq!(ctx.scalar.read(SS_STATUS_REG_INDEX), status);
        }
    }

    #[test]
    fn non_stream_op_not_handled() {
        let (mut ctx, mut tile) = (make_ctx(), make_tile());
        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add)
            .with_dest(Operand::ScalarReg(2))
            .with_source(Operand::ScalarReg(0))
            .with_source(Operand::ScalarReg(1));
        assert_eq!(StreamOps::execute(&op, &mut ctx, &mut tile), StreamResult::NotStreamOp);
    }
}
