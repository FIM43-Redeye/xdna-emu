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
            Some(SemanticOp::CascadeRead) => {
                if tile.cascade_input.is_empty() {
                    Some(CascadeResult::StallRead)
                } else {
                    None
                }
            }
            Some(SemanticOp::CascadeWrite) => {
                // Mirror the depth limit applied in execute_write.
                if tile.cascade_output.len() >= 4 {
                    Some(CascadeResult::StallWrite)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Execute cascade read: pop 512-bit data from SCD into destination register.
    fn execute_read(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> CascadeResult {
        let data = match tile.pop_cascade_input() {
            Some(d) => d,
            None => {
                log::info!(
                    "[CASCADE] Read stall: SCD empty (tile {},{}, pc=0x{:X})",
                    tile.col,
                    tile.row,
                    ctx.pc()
                );
                return CascadeResult::StallRead;
            }
        };

        // Write to destination register (vector or accumulator)
        if let Some(dest) = &op.dest {
            match dest {
                Operand::VectorReg(r) => {
                    let words = cascade_to_vector(&data);
                    ctx.vector.write(*r, words);
                    log::info!("[CASCADE] Read SCD -> v{} (tile {},{})", r, tile.col, tile.row);
                }
                Operand::AccumReg(r) => {
                    let lanes = cascade_to_accumulator(&data);
                    ctx.accumulator.write(*r, lanes);
                    log::info!(
                        "[CASCADE] Read SCD -> acc{} (tile {},{}) data[0]={:#X}",
                        r,
                        tile.col,
                        tile.row,
                        data[0]
                    );
                }
                _ => {
                    return CascadeResult::Error(format!(
                        "[CASCADE] Read SCD: unexpected dest {:?} (tile {},{}) -- impossible operand",
                        dest, tile.col, tile.row,
                    ));
                }
            }
        }

        CascadeResult::Completed
    }

    /// Execute cascade write: push source register data to MCD.
    fn execute_write(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> CascadeResult {
        // Backpressure: if MCD FIFO is full, stall.
        // Depth 4: hardware has a 2-deep 512-bit FIFO (AM020), and each
        // VMOV MCD writes one 256-bit half-register, so 4 entries covers
        // two full accumulator transfers with room for pipeline slack.
        if tile.cascade_output.len() >= 4 {
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
                    let words = ctx.vector.read(*r);
                    let d = vector_to_cascade(&words);
                    log::info!("[CASCADE] Write v{} -> MCD (tile {},{})", r, tile.col, tile.row);
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

/// Pack a 256-bit vector register (8 x u32) into 512-bit cascade data (8 x u64).
/// Low 4 slots hold the vector data, high 4 slots are zero-padded.
fn vector_to_cascade(words: &[u32; 8]) -> [u64; CASCADE_WORDS] {
    let mut data = [0u64; CASCADE_WORDS];
    for i in 0..4 {
        data[i] = (words[i * 2] as u64) | ((words[i * 2 + 1] as u64) << 32);
    }
    // data[4..] remain zero (padding -- a vector only fills the low 256 bits)
    data
}

/// Unpack 512-bit cascade data (8 x u64) into a 256-bit vector register (8 x u32).
/// Takes the low 4 slots (256 bits), ignores the high slots.
fn cascade_to_vector(data: &[u64; CASCADE_WORDS]) -> [u32; 8] {
    let mut words = [0u32; 8];
    for i in 0..4 {
        words[i * 2] = data[i] as u32;
        words[i * 2 + 1] = (data[i] >> 32) as u32;
    }
    words
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

    #[test]
    fn test_cascade_read_empty_stalls() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let op =
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeRead).with_dest(Operand::VectorReg(0));

        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::StallRead);
    }

    #[test]
    fn test_cascade_read_vector() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data: [u64; CASCADE_WORDS] = [
            0x0000_0002_0000_0001,
            0x0000_0004_0000_0003,
            0x0000_0006_0000_0005,
            0x0000_0008_0000_0007,
            0xDEAD,
            0xBEEF,
            0xCAFE,
            0xF00D,
        ];
        tile.push_cascade_input(test_data);

        let op =
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeRead).with_dest(Operand::VectorReg(5));

        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::Completed);

        let result = ctx.vector.read(5);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);
        assert_eq!(result[3], 4);
        assert_eq!(result[4], 5);
        assert_eq!(result[5], 6);
        assert_eq!(result[6], 7);
        assert_eq!(result[7], 8);
    }

    #[test]
    fn test_cascade_read_accumulator() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data: [u64; CASCADE_WORDS] = [10, 20, 30, 40, 50, 60, 70, 80];
        tile.push_cascade_input(test_data);

        let op =
            SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::CascadeRead).with_dest(Operand::AccumReg(3));

        assert_eq!(CascadeOps::execute(&op, &mut ctx, &mut tile), CascadeResult::Completed);

        // All 8 accumulator lanes are carried by the 512-bit cascade (no padding).
        let result = ctx.accumulator.read(3);
        assert_eq!(result, [10, 20, 30, 40, 50, 60, 70, 80]);
    }

    #[test]
    fn test_cascade_write_vector() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.vector.write(2, [0xA, 0xB, 0xC, 0xD, 0xE, 0xF, 0x10, 0x11]);

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
        // High 4 words zero-padded -- a vector fills only the low 256 bits.
        assert_eq!(data[4..], [0, 0, 0, 0]);
    }

    #[test]
    fn test_cascade_write_full_stalls() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Fill the MCD FIFO (depth 4)
        for _ in 0..4 {
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
        let original = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let cascade = vector_to_cascade(&original);
        let recovered = cascade_to_vector(&cascade);
        assert_eq!(original, recovered);
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
