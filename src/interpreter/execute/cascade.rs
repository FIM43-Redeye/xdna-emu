//! Cascade stream execution unit.
//!
//! Handles cascade read/write operations for the dedicated 512-bit
//! point-to-point link between adjacent compute tiles. Cascade is
//! entirely separate from the stream switch fabric.
//!
//! # Hardware Facts
//!
//! - 512-bit data width = `CASCADE_WORDS` x u64 = 64 bytes (AIE2; AIE1 is 384).
//!   Source: mlir-aie `AIE2TargetModel::getAccumulatorCascadeSize()` == 512.
//! - Direction configured via accumulator control register at 0x36060
//!   (aie-rt xaie_core.c sets direction only, not width)
//! - Core stalls on empty SCD read or full MCD write
//! - Only present on architectures where `has_cascade_link` is true (AIE2, AIE2P).
//!
//! # Instructions
//!
//! - `VMOV_mv_scd` / `VMOV_HI` / `VMOV_LO`: read SCD into vector/accumulator
//! - `VMOV_mv_mcd`: write vector/accumulator to MCD
//!
//! # Data Packing
//!
//! The cascade link is 512 bits wide (`CASCADE_WORDS` u64 words). Register types:
//! - Vector register (256 bits = 8 x u32): packed into low 4 of `[u64; CASCADE_WORDS]`
//! - Accumulator register (512 bits = 8 x u64): all `CASCADE_WORDS` lanes carried

use crate::device::arch_handle;
use crate::device::tile::Tile;
use crate::interpreter::bundle::{Operand, SlotOp};
use xdna_archspec::aie2::{CASCADE_WORDS, isa::SemanticOp};
use crate::interpreter::state::ExecutionContext;

/// Cascade operations execution unit.
pub struct CascadeOps;

/// Result of cascade operation execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CascadeResult {
    /// Operation completed successfully.
    Completed,
    /// Core must stall on SCD read (empty input).
    StallRead,
    /// Core must stall on MCD write (full output).
    StallWrite,
    /// Not a cascade operation -- pass to next execution unit.
    NotCascadeOp,
    /// Fatal error in cascade operation.
    Error(String),
}

impl CascadeOps {
    /// Execute a cascade operation.
    ///
    /// Returns `Completed` if handled, `Stall` if blocking, or
    /// `NotCascadeOp` if the operation isn't a cascade instruction (or if
    /// the architecture has no cascade link -- always the case for AIE1).
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> CascadeResult {
        if !arch_handle::has_cascade_link() {
            return CascadeResult::NotCascadeOp;
        }
        match op.semantic {
            Some(SemanticOp::CascadeRead) => Self::execute_read(op, ctx, tile),
            Some(SemanticOp::CascadeWrite) => Self::execute_write(op, ctx, tile),
            _ => CascadeResult::NotCascadeOp,
        }
    }

    /// Pre-check whether this op would stall, without any side effects.
    ///
    /// VLIW bundles execute multiple slots in parallel and must commit
    /// atomically: either every slot completes or none does. If a cascade
    /// op stalls mid-bundle, sibling slots that already executed would
    /// double-commit when the bundle is retried next cycle. Callers must
    /// pre-check every cascade slot and bail out before executing any
    /// slot if one would block.
    ///
    /// Returns `Some(StallRead | StallWrite)` if the op would stall, or
    /// `None` if the op is non-cascade, would proceed, or if the
    /// architecture has no cascade link.
    pub fn would_stall(op: &SlotOp, tile: &Tile) -> Option<CascadeResult> {
        if !arch_handle::has_cascade_link() {
            return None;
        }
        match op.semantic {
            // A read never stalls at its own issue: it is held five bundles later
            // (`due_read_would_stall`).
            Some(SemanticOp::CascadeRead) => None,
            Some(SemanticOp::CascadeWrite) => {
                // Mirror the depth limit applied in execute_write.
                if tile.cascade_output_full() {
                    Some(CascadeResult::StallWrite)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Issue a cascade read: queue it for its hold stage (see
    /// [`PendingCascadeRead`](crate::interpreter::state::context::PendingCascadeRead)).
    /// The word is taken later, by [`Self::complete_due_read`].
    fn execute_read(op: &SlotOp, ctx: &mut ExecutionContext, _tile: &mut Tile) -> CascadeResult {
        match &op.dest {
            Some(dest @ (Operand::VectorReg(_) | Operand::AccumReg(_))) => {
                ctx.queue_pending_cascade_read(dest.clone());
                CascadeResult::Completed
            }
            other => CascadeResult::Error(format!(
                "[CASCADE] Read SCD: unexpected dest {other:?} -- impossible operand"
            )),
        }
    }

    /// True if the bundle about to issue is held by a due cascade read whose word has
    /// not arrived (SCD empty). Call before issuing a bundle; no side effects.
    pub fn due_read_would_stall(ctx: &ExecutionContext, tile: &Tile) -> bool {
        arch_handle::has_cascade_link() && ctx.due_cascade_read().is_some() && tile.cascade_input.is_empty()
    }

    /// Take the word for every cascade read that is due (normally at most one per
    /// bundle) and queue its destination write. Call once per issued bundle, after
    /// `ExecutionContext::advance_vector_bundle` and before the slots execute.
    pub fn complete_due_read(ctx: &mut ExecutionContext, tile: &mut Tile) {
        while ctx.due_cascade_read().is_some() {
            let Some(data) = tile.pop_cascade_input() else {
                return;
            };
            let (lo, hi) = cascade_to_vector(&data);
            let mut vec = [0u32; 16];
            vec[..8].copy_from_slice(&lo);
            vec[8..].copy_from_slice(&hi);
            log::info!("[CASCADE] SCD word taken (tile {},{}) data[0]={:#X}", tile.col, tile.row, data[0]);
            ctx.complete_due_cascade_read(vec, cascade_to_accumulator(&data));
        }
    }

    /// Execute cascade write: push source register data to MCD.
    fn execute_write(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> CascadeResult {
        // Backpressure: if the MCD FIFO is full, stall. Total depth is pinned by
        // silicon (see `CASCADE_TOTAL_DEPTH`).
        if tile.cascade_output_full() {
            log::info!(
                "[CASCADE] Write stall: MCD full (tile {},{}, pc=0x{:X})",
                tile.col,
                tile.row,
                ctx.pc()
            );
            return CascadeResult::StallWrite;
        }

        // Read source register (vector or accumulator)
        let data = if let Some(src) = op.sources.first() {
            match src {
                Operand::VectorReg(r) => {
                    // `vmov MCD, xN`: the operand is the 512-bit x register.
                    let lo = ctx.vector.read(*r);
                    let hi = ctx.vector.read(*r + 1);
                    let d = vector_to_cascade(&lo, &hi);
                    log::info!(
                        "[CASCADE] Write x (v{}, v{}) -> MCD (tile {},{})",
                        r,
                        r + 1,
                        tile.col,
                        tile.row
                    );
                    d
                }
                Operand::AccumReg(r) => {
                    let lanes = ctx.accumulator.read(*r);
                    let d = accumulator_to_cascade(&lanes);
                    log::info!(
                        "[CASCADE] Write acc{} -> MCD (tile {},{}) data[0]={:#X}",
                        r,
                        tile.col,
                        tile.row,
                        d[0]
                    );
                    d
                }
                _ => {
                    return CascadeResult::Error(format!(
                        "[CASCADE] Write MCD: unexpected src {:?} (tile {},{}) -- impossible operand",
                        src, tile.col, tile.row,
                    ));
                }
            }
        } else {
            return CascadeResult::Error(format!(
                "[CASCADE] Write MCD: no source operand (tile {},{}) -- malformed instruction",
                tile.col, tile.row,
            ));
        };

        tile.push_cascade_output(data);
        CascadeResult::Completed
    }
}

/// Pack a 512-bit x register, held as two 256-bit halves (8 x u32 each), into
/// 512-bit cascade data (8 x u64): the low half fills slots 0..4, the high half
/// slots 4..8.
fn vector_to_cascade(lo: &[u32; 8], hi: &[u32; 8]) -> [u64; CASCADE_WORDS] {
    let mut data = [0u64; CASCADE_WORDS];
    for i in 0..4 {
        data[i] = (lo[i * 2] as u64) | ((lo[i * 2 + 1] as u64) << 32);
        data[4 + i] = (hi[i * 2] as u64) | ((hi[i * 2 + 1] as u64) << 32);
    }
    data
}

/// Unpack 512-bit cascade data into the (low, high) 256-bit halves of an x register.
fn cascade_to_vector(data: &[u64; CASCADE_WORDS]) -> ([u32; 8], [u32; 8]) {
    let mut lo = [0u32; 8];
    let mut hi = [0u32; 8];
    for i in 0..4 {
        lo[i * 2] = data[i] as u32;
        lo[i * 2 + 1] = (data[i] >> 32) as u32;
        hi[i * 2] = data[4 + i] as u32;
        hi[i * 2 + 1] = (data[4 + i] >> 32) as u32;
    }
    (lo, hi)
}

/// Pack a 512-bit accumulator register (8 x u64) into 512-bit cascade data
/// (`CASCADE_WORDS` x u64). All 8 lanes are carried -- no truncation.
fn accumulator_to_cascade(lanes: &[u64; 8]) -> [u64; CASCADE_WORDS] {
    let mut data = [0u64; CASCADE_WORDS];
    data.copy_from_slice(&lanes[..CASCADE_WORDS]);
    data
}

/// Unpack 512-bit cascade data (`CASCADE_WORDS` x u64) into a 512-bit
/// accumulator register (8 x u64). All 8 lanes are preserved.
fn cascade_to_accumulator(data: &[u64; CASCADE_WORDS]) -> [u64; 8] {
    let mut lanes = [0u64; 8];
    lanes[..CASCADE_WORDS].copy_from_slice(data);
    lanes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use xdna_archspec::aie2::isa::SemanticOp;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    const WORD: [u64; CASCADE_WORDS] = [
        0x0000_0002_0000_0001,
        0x0000_0004_0000_0003,
        0x0000_0006_0000_0005,
        0x0000_0008_0000_0007,
        0xDEAD,
        0xBEEF,
        0xCAFE,
        0xF00D,
    ];

    fn read_op(dest: Operand) -> SlotOp {
        SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeRead).with_dest(dest)
    }

    /// Read a vector register as a store's data source does (use cycle 1, no bypass), the
    /// consumer the silicon probes used (`vst wl0/wh0`).
    fn vst_read(ctx: &ExecutionContext, reg: u8) -> [u32; 8] {
        ctx.vector.read_with(reg, 1, crate::interpreter::state::Bypass::No)
    }

    /// An accumulator reader with use cycle 1 (the `vmov x, bml` of the silicon probe).
    fn acc_read(ctx: &ExecutionContext, reg: u8) -> [u64; 8] {
        ctx.accumulator.read_with(reg, 1, crate::interpreter::state::Bypass::No)
    }

    /// Issue a cascade read at bundle `at` with the itinerary latency (7), as the executor does.
    fn issue_read(ctx: &mut ExecutionContext, tile: &mut Tile, dest: Operand, at: u64) {
        ctx.bundle_seq = at;
        ctx.result_latency = 7;
        assert_eq!(CascadeOps::execute(&read_op(dest), ctx, tile), CascadeResult::Completed);
    }

    /// Run the per-bundle part of the executor for the bundle numbered `seq`: stall check,
    /// advance, take the due word.
    fn begin_bundle(ctx: &mut ExecutionContext, tile: &mut Tile, seq: u64) -> bool {
        ctx.bundle_seq = seq;
        if CascadeOps::due_read_would_stall(ctx, tile) {
            return false;
        }
        ctx.advance_vector_bundle();
        CascadeOps::complete_due_read(ctx, tile);
        true
    }

    // Silicon (Phoenix, build/experiments/cascade-collisions/p2/hw_r1_sweep1.txt, use_N/nouse_N
    // "absent" rows): `vmov x, SCD` on an empty cascade does not hold the vmov or the next four
    // bundles (timestamps 1..=4 bundles later exclude the wait); the fifth bundle after it is held.
    #[test]
    fn cascade_read_does_not_stall_at_issue() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        let op = read_op(Operand::VectorReg(0));
        assert_eq!(CascadeOps::would_stall(&op, &tile), None);
        issue_read(&mut ctx, &mut tile, Operand::VectorReg(0), 10);
        for seq in 11..=14 {
            assert!(begin_bundle(&mut ctx, &mut tile, seq), "bundle {seq} must issue");
        }
    }

    #[test]
    fn cascade_read_holds_the_fifth_bundle_after_it_until_a_word_arrives() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        issue_read(&mut ctx, &mut tile, Operand::VectorReg(0), 10);
        for seq in 11..=14 {
            assert!(begin_bundle(&mut ctx, &mut tile, seq));
        }
        // Bundle 15 is held for as long as SCD is empty ...
        for _ in 0..3 {
            assert!(!begin_bundle(&mut ctx, &mut tile, 15), "held while SCD is empty");
        }
        // ... and issues once the word has arrived, taking it.
        tile.push_cascade_input(WORD);
        assert!(begin_bundle(&mut ctx, &mut tile, 15));
        assert!(!tile.has_cascade_input(), "the word was taken");
    }

    // Silicon (same sweeps, the `data=` flags): the destination is written so that it is visible
    // 7 issued bundles after the vmov (readers at +6 or earlier see the old value, at +7 the
    // word), with or without a hold, for x and bm destinations alike. No register interlock.
    #[test]
    fn cascade_read_vector_destination_is_visible_seven_bundles_after_issue() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.vector.write(4, [7; 8]);
        ctx.vector.write(5, [7; 8]);
        tile.push_cascade_input(WORD);
        issue_read(&mut ctx, &mut tile, Operand::VectorReg(4), 10);
        for seq in 11..=16 {
            assert!(begin_bundle(&mut ctx, &mut tile, seq));
            assert_eq!(vst_read(&ctx, 4), [7; 8], "bundle +{} still sees the old value", seq - 10);
        }
        assert!(begin_bundle(&mut ctx, &mut tile, 17));
        // `vmov x2, SCD`: the 256-bit pair VectorReg(4) (low) + VectorReg(5) (high); silicon carries
        // the whole 512-bit word through x (harness cascade smoke probe).
        assert_eq!(vst_read(&ctx, 4), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(vst_read(&ctx, 5), [0xDEAD, 0, 0xBEEF, 0, 0xCAFE, 0, 0xF00D, 0]);
    }

    #[test]
    fn cascade_read_visibility_is_independent_of_the_hold() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.vector.write(4, [7; 8]);
        issue_read(&mut ctx, &mut tile, Operand::VectorReg(4), 10);
        for seq in 11..=14 {
            assert!(begin_bundle(&mut ctx, &mut tile, seq));
        }
        assert!(!begin_bundle(&mut ctx, &mut tile, 15)); // held; bundle_seq does not advance
        tile.push_cascade_input(WORD);
        assert!(begin_bundle(&mut ctx, &mut tile, 15));
        assert!(begin_bundle(&mut ctx, &mut tile, 16));
        assert_eq!(vst_read(&ctx, 4), [7; 8], "+6: old value");
        assert!(begin_bundle(&mut ctx, &mut tile, 17));
        assert_eq!(vst_read(&ctx, 4), [1, 2, 3, 4, 5, 6, 7, 8], "+7: the word");
    }

    #[test]
    fn cascade_read_accumulator_destination_carries_all_lanes_seven_bundles_after_issue() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        tile.push_cascade_input([10, 20, 30, 40, 50, 60, 70, 80]);
        ctx.accumulator.write(3, [7; 8]);
        issue_read(&mut ctx, &mut tile, Operand::AccumReg(3), 10);
        for seq in 11..=16 {
            assert!(begin_bundle(&mut ctx, &mut tile, seq));
            assert_eq!(acc_read(&ctx, 3), [7; 8], "bundle +{} still sees the old value", seq - 10);
        }
        assert!(begin_bundle(&mut ctx, &mut tile, 17));
        assert_eq!(acc_read(&ctx, 3), [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn test_cascade_write_vector() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // `vmov MCD, x1` = VectorReg(2) (low 256 bits) + VectorReg(3) (high).
        ctx.vector.write(2, [0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10, 0x11]);
        ctx.vector.write(3, [0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27]);

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeWrite)
            .with_source(Operand::VectorReg(2));

        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::Completed);
        assert!(tile.has_cascade_output());

        let data = tile.pop_cascade_output().unwrap();
        // Verify packing: pairs of u32 into u64
        assert_eq!(data[0], 0x0000_000B_0000_000A);
        assert_eq!(data[1], 0x0000_000D_0000_000C);
        assert_eq!(data[2], 0x0000_000F_0000_000E);
        assert_eq!(data[3], 0x0000_0011_0000_0010);
        // The x register's high half fills the high 256 bits of the cascade word.
        assert_eq!(data[4], 0x0000_0021_0000_0020);
        assert_eq!(data[5], 0x0000_0023_0000_0022);
        assert_eq!(data[6], 0x0000_0025_0000_0024);
        assert_eq!(data[7], 0x0000_0027_0000_0026);
    }

    #[test]
    fn test_cascade_write_full_stalls() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Fill the MCD FIFO
        for _ in 0..crate::device::tile::CASCADE_OUTPUT_DEPTH {
            tile.push_cascade_output([0; CASCADE_WORDS]);
        }

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeWrite)
            .with_source(Operand::VectorReg(0));

        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::StallWrite);
    }

    #[test]
    fn test_not_cascade_op() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op = SlotOp::from_semantic(SlotIndex::Scalar0, SemanticOp::Add);
        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::NotCascadeOp);
    }

    #[test]
    fn test_vector_cascade_roundtrip() {
        let lo = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let hi = [9u32, 10, 11, 12, 13, 14, 15, 16];
        let cascade = vector_to_cascade(&lo, &hi);
        assert_eq!(cascade_to_vector(&cascade), (lo, hi));
    }

    #[test]
    fn test_accumulator_cascade_roundtrip() {
        // AIE2/AIE-ML cascade is 512-bit (getAccumulatorCascadeSize() == 512):
        // a full 8-lane accumulator must survive forwarding with no truncation.
        let original = [100u64, 200, 300, 400, 500, 600, 700, 800];
        let cascade = accumulator_to_cascade(&original);
        let recovered = cascade_to_accumulator(&cascade);
        assert_eq!(recovered, original, "all 512 bits (8 accumulator lanes) must round-trip through cascade");
    }
}
