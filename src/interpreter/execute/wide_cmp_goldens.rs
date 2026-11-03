//! The compare mask of the dual-result vector ops on x (512-bit) operands, from the movd probes
//! (`build/experiments/harness/movd`, p1, NPU1 silicon, r1 == r2): `vabs_gtz`, `vsub_lt` and
//! `vmax_lt` with x0 = every lane 0x66666666, x1 = every lane 0x7fffffff leave 0xffff (sixteen
//! 32-bit lanes, all true) in the mask register. The emulator wrote no mask at all for the wide
//! forms of every op but `vmax_lt`.

use super::cycle_accurate::CycleAccurateExecutor;
use crate::device::tile::Tile;
use crate::interpreter::bundle::VliwBundle;
use crate::interpreter::decode::InstructionDecoder;
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::traits::{Decoder, Executor};

fn bytes(hex: &str) -> Vec<u8> {
    hex.split_whitespace().map(|h| u8::from_str_radix(h, 16).unwrap()).collect()
}

/// Run one instruction with x0, x1 as the probe sets them and r16 = 0xaaaaaaaa; the mask register.
fn mask_after(hex: &str) -> u32 {
    let decoder = InstructionDecoder::load_default();
    let mut ctx = ExecutionContext::new();
    ctx.vector.write_wide(0, [0x6666_6666; 16]);
    ctx.vector.write_wide(2, [0x7fff_ffff; 16]);
    ctx.scalar.write(16, 0xaaaa_aaaa);
    let mut tile = Tile::compute(0, 2);
    let mut executor = CycleAccurateExecutor::new();
    let bundle = decoder.decode(&bytes(hex), 0).unwrap_or_else(|e| panic!("{hex}: {e:?}"));
    executor.execute(&bundle, &mut ctx, &mut tile);
    for _ in 0..8 {
        executor.execute(&VliwBundle::empty(), &mut ctx, &mut tile);
    }
    ctx.scalar.read(16)
}

#[test]
fn the_wide_dual_result_compares_write_their_mask() {
    for (asm, hex) in [
        ("vabs_gtz.s32 x2, r16, x1", "d9 80 03 19"),
        ("vsub_lt.s32 x2, r16, x0, x1", "d9 80 05 19"),
        ("vmax_lt.s32 x2, r16, x0, x1", "d9 80 0a 19"),
    ] {
        assert_eq!(mask_after(hex), 0xffff, "{asm}");
    }
}

// Not probed on silicon; the same structure with the compare each op's narrow form uses
// (a >= b for vsub_ge, a < b for vmaxdiff_lt, s > 0 for vneg_gtz).
#[test]
fn the_other_wide_dual_result_compares_follow_their_narrow_compare() {
    for (asm, hex, want) in [
        ("vneg_gtz32 x2, r16, x1", "d9 80 0d 19", 0xffff),
        ("vsub_ge.s32 x2, r16, x1, x0", "59 88 06 19", 0xffff),
        ("vmaxdiff_lt.s32 x2, r16, x0, x1", "d9 80 04 19", 0xffff),
    ] {
        assert_eq!(mask_after(hex), want, "{asm}");
    }
}
