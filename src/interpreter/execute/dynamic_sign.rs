//! Run-time signedness of the `d` vector forms.
//!
//! Several AIE2 vector instructions have an `s` form, whose lanes are signed by encoding,
//! and a `d` form that leaves the sign open (`vups.s32.d16`, `vsrs.d16.s32`, `vabs_gtz.d8`,
//! `vlt.d16`, `vextract.d8`, ...). The `d` form reads its sign from a control register that
//! its TableGen record lists in `Uses`: crUPSSign for the UPS forms, crSRSSign for the SRS
//! forms, crVaddSign for the compare / abs / min / max / sub-compare / extract forms. The
//! decoder labels every `d` lane type unsigned (`D16` -> `UInt16`); when the register's bit 0
//! is set the unsigned type fields of the op become signed (silicon, dynsign probe suite:
//! bit 0 decides, the higher bits are ignored).
//!
//! Pack and unpack resolve their sign through their own form tables
//! (`xdna_archspec::aie2::sign::SignSource`); they ignore the types resolved here.

use crate::interpreter::bundle::{ElementType, Operand, SlotOp};
use crate::interpreter::decode::register_map::classify_reg_name;
use crate::interpreter::state::ExecutionContext;

/// The sign control register an op reads (its TableGen `Uses`), as a control-register id.
pub(crate) fn sign_register(op: &SlotOp) -> Option<u8> {
    op.implicit_regs
        .iter()
        .filter(|r| r.is_use && r.reg_class.starts_with("cr") && r.reg_class.ends_with("Sign"))
        .find_map(|r| match classify_reg_name(&r.reg_class)?.operand {
            Operand::ControlReg(id) => Some(id),
            _ => None,
        })
}

/// The signedness an op's sign register holds now, if the op has one.
pub(crate) fn signed_now(op: &SlotOp, ctx: &ExecutionContext) -> Option<bool> {
    sign_register(op).map(|id| ctx.control_reg(id) & 1 != 0)
}

/// A decoded lane type under a run-time sign: an unsigned (`d`) type becomes signed when the
/// register says so; signed types and ops without a sign register are unchanged.
pub(crate) fn resolve(ty: ElementType, signed: Option<bool>) -> ElementType {
    match signed {
        Some(true) if !ty.is_signed() => ty.with_signedness(true),
        _ => ty,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;
    use xdna_archspec::aie2::isa::{ImplicitReg, SemanticOp};

    fn op_using(regs: &[&str]) -> SlotOp {
        SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::AbsGtz).with_implicit_regs(
            regs.iter()
                .map(|r| ImplicitReg { reg_class: r.to_string(), reg_num: 0, is_use: true }),
        )
    }

    #[test]
    fn the_sign_register_comes_from_the_uses_list() {
        assert_eq!(sign_register(&op_using(&["crVaddSign"])), Some(0));
        assert_eq!(sign_register(&op_using(&["crSat", "crUPSSign"])), Some(10));
        assert_eq!(sign_register(&op_using(&["crSat", "crRnd", "crSRSSign"])), Some(8));
        assert_eq!(sign_register(&op_using(&["crSat"])), None);
        assert_eq!(sign_register(&op_using(&[])), None);
    }

    #[test]
    fn a_def_of_a_sign_register_is_not_a_sign_source() {
        let op =
            SlotOp::from_semantic(SlotIndex::Vector, SemanticOp::AbsGtz).with_implicit_regs([ImplicitReg {
                reg_class: "crVaddSign".to_string(),
                reg_num: 0,
                is_use: false,
            }]);
        assert_eq!(sign_register(&op), None);
    }

    #[test]
    fn bit_zero_of_the_register_decides() {
        let op = op_using(&["crSat", "crUPSSign"]);
        let mut ctx = ExecutionContext::new();
        for (value, signed) in [(0, false), (1, true), (2, false), (3, true)] {
            ctx.set_control_reg_raw(10, value);
            assert_eq!(signed_now(&op, &ctx), Some(signed), "crUPSSign = {value}");
        }
        // crSRSSign lives in the SRS configuration.
        let srs = op_using(&["crSat", "crRnd", "crSRSSign"]);
        ctx.srs_config.srs_sign = true;
        assert_eq!(signed_now(&srs, &ctx), Some(true));
        ctx.srs_config.srs_sign = false;
        assert_eq!(signed_now(&srs, &ctx), Some(false));
        assert_eq!(signed_now(&op_using(&["crSat"]), &ctx), None);
    }

    #[test]
    fn only_unsigned_types_take_the_run_time_sign() {
        assert_eq!(resolve(ElementType::UInt16, Some(true)), ElementType::Int16);
        assert_eq!(resolve(ElementType::UInt16, Some(false)), ElementType::UInt16);
        assert_eq!(resolve(ElementType::Int32, Some(false)), ElementType::Int32);
        assert_eq!(resolve(ElementType::UInt8, None), ElementType::UInt8);
        assert_eq!(resolve(ElementType::BFloat16, Some(true)), ElementType::BFloat16);
    }
}

/// The `d` forms run end to end from their real encodings (Peano / Chess disassembly of the
/// dynsign probe kernels), against values silicon returned in that suite.
#[cfg(test)]
mod execution {
    use super::*;
    use crate::device::tile::Tile;
    use crate::interpreter::bundle::{SlotIndex, VliwBundle};
    use crate::interpreter::decode::InstructionDecoder;
    use crate::interpreter::execute::cycle_accurate::CycleAccurateExecutor;
    use crate::interpreter::traits::{Decoder, Executor};
    use xdna_archspec::aie2::isa::SemanticOp;

    const VABS_GTZ_D8: [u8; 4] = [0x19, 0x00, 0x43, 0x18]; // vabs_gtz.d8 x0, r25:r24, x0
    const VABS_GTZ_S8: [u8; 4] = [0x19, 0x80, 0x43, 0x18]; // vabs_gtz.s8 x0, r25:r24, x0
    const VLT_D8: [u8; 4] = [0x19, 0x01, 0x4c, 0x18]; // vlt.d8 r25:r24, x0, x2
    const VUPS_S32_D16: [u8; 4] = [0xd9, 0x07, 0x71, 0x18]; // vups.s32.d16 bmh0, wl0, s3
    const VSRS_D16_S32: [u8; 4] = [0x99, 0x03, 0x1c, 0x08]; // vsrs.d16.s32 wl0, bmh0, s0
    const VLDA_UPS_S64_D32: [u8; 4] = [0x19, 0x83, 0x2c, 0x07]; // vlda.ups.s64.d32 bmh0, s0, [p7, #0x20]
    const VST_SRS_D16_S32: [u8; 4] = [0x99, 0x03, 0x0c, 0x09]; // vst.srs.d16.s32 bmh0, s0, [p1, #0x0]
    const NOP: [u8; 2] = [0x01, 0x00];

    struct Rig {
        decoder: InstructionDecoder,
        exec: CycleAccurateExecutor,
        ctx: ExecutionContext,
        tile: Tile,
    }

    impl Rig {
        fn new() -> Self {
            Rig {
                decoder: InstructionDecoder::load_default(),
                exec: CycleAccurateExecutor::new(),
                ctx: ExecutionContext::new(),
                tile: Tile::compute(0, 2),
            }
        }
        fn bundle(&self, bytes: &[u8]) -> VliwBundle {
            self.decoder.decode(bytes, 0).expect("decode")
        }
        fn run(&mut self, bytes: &[u8]) {
            let b = self.bundle(bytes);
            self.exec.execute(&b, &mut self.ctx, &mut self.tile);
        }
        fn settle(&mut self, bundles: usize) {
            for _ in 0..bundles {
                self.run(&NOP);
            }
        }
        fn mask64(&self) -> u64 {
            self.ctx.scalar.read(24) as u64 | (self.ctx.scalar.read(25) as u64) << 32
        }
    }

    fn slot_with(bundle: &VliwBundle, sem: SemanticOp) -> SlotOp {
        SlotIndex::all()
            .iter()
            .filter_map(|s| bundle.slot(*s))
            .find(|op| op.semantic == Some(sem))
            .unwrap_or_else(|| panic!("no {sem:?} slot"))
            .clone()
    }

    #[test]
    fn real_encodings_carry_their_sign_register() {
        let rig = Rig::new();
        let cases: [(&[u8], SemanticOp, Option<u8>); 7] = [
            (&VABS_GTZ_D8, SemanticOp::AbsGtz, Some(0)),
            (&VABS_GTZ_S8, SemanticOp::AbsGtz, None),
            (&VLT_D8, SemanticOp::SetLt, Some(0)),
            (&VUPS_S32_D16, SemanticOp::Ups, Some(10)),
            (&VSRS_D16_S32, SemanticOp::Srs, Some(8)),
            (&VLDA_UPS_S64_D32, SemanticOp::Ups, Some(10)),
            (&VST_SRS_D16_S32, SemanticOp::Srs, Some(8)),
        ];
        for (bytes, sem, reg) in cases {
            let op = slot_with(&rig.bundle(bytes), sem);
            assert_eq!(sign_register(&op), reg, "{:?}", op.encoding_name);
        }
    }

    /// vlt.d8 of 0x80 against 0x01: unsigned 128 < 1 is false, signed -128 < 1 is true.
    #[test]
    fn vlt_d8_follows_crvaddsign() {
        for (sign, expect) in [(0, 0u64), (2, 0), (1, u64::MAX), (3, u64::MAX)] {
            let mut rig = Rig::new();
            rig.ctx.vector.write_wide(0, [0x8080_8080; 16]);
            rig.ctx.vector.write_wide(4, [0x0101_0101; 16]); // x2
            rig.ctx.set_control_reg_raw(0, sign);
            rig.run(&VLT_D8);
            rig.settle(4);
            assert_eq!(rig.mask64(), expect, "crVaddSign = {sign}");
        }
    }

    /// vabs_gtz.d8 on lanes 0x80, 0xff, 0x05, 0, ...: unsigned, every nonzero lane is > 0;
    /// signed, only 0x05 is.
    #[test]
    fn vabs_gtz_d8_follows_crvaddsign() {
        for (sign, expect) in [(0, 0b111u64), (1, 0b100)] {
            let mut rig = Rig::new();
            let mut v = [0u32; 16];
            v[0] = 0x0005_ff80;
            rig.ctx.vector.write_wide(0, v);
            rig.ctx.set_control_reg_raw(0, sign);
            rig.run(&VABS_GTZ_D8);
            rig.settle(4);
            assert_eq!(rig.mask64(), expect, "crVaddSign = {sign}");
        }
    }

    fn acc32_lanes(acc: [u64; 8]) -> Vec<u32> {
        acc.iter().flat_map(|w| [*w as u32, (*w >> 32) as u32]).collect()
    }

    /// vups.s32.d16: crUPSSign extends the source lanes; any nonzero crSat clamps to the signed
    /// 32-bit lane range, never symmetric (dynsign silicon: lane 0x8000 << 16 signed with
    /// crSat = 3 gives 0x80000000; unsigned 0xffff << 16 with crSat = 2 gives 0x7fffffff).
    #[test]
    fn vups_d16_follows_crupssign_and_saturates_on_any_crsat() {
        let run = |sign: u32, sat: u8, shift: u32| -> Vec<u32> {
            let mut rig = Rig::new();
            let mut v = [0u32; 8];
            v[0] = 0xffff_8000; // lanes 0x8000, 0xffff
            v[1] = 0x0001_7fff; // lanes 0x7fff, 0x0001
            rig.ctx.vector.write(0, v);
            rig.ctx.scalar.write(40 + 3, shift); // s3
            rig.ctx.set_control_reg_raw(10, sign);
            rig.ctx.srs_config.saturation_mode = sat;
            rig.run(&VUPS_S32_D16);
            rig.settle(4);
            acc32_lanes(rig.ctx.accumulator.read(1))[..4].to_vec()
        };
        assert_eq!(run(0, 0, 0), [0x8000, 0xffff, 0x7fff, 1]);
        assert_eq!(run(1, 0, 0), [0xffff_8000, 0xffff_ffff, 0x7fff, 1]);
        assert_eq!(run(2, 0, 0), [0x8000, 0xffff, 0x7fff, 1], "only bit 0 counts");
        assert_eq!(run(0, 1, 16), [0x7fff_ffff, 0x7fff_ffff, 0x7fff_0000, 0x1_0000]);
        assert_eq!(run(1, 1, 16), [0x8000_0000, 0xffff_0000, 0x7fff_0000, 0x1_0000]);
        assert_eq!(run(1, 3, 24), [0x8000_0000, 0xff00_0000, 0x7fff_ffff, 0x0100_0000]);
        assert_eq!(run(0, 2, 16), [0x7fff_ffff, 0x7fff_ffff, 0x7fff_0000, 0x1_0000]);
        assert_eq!(run(1, 0, 24), [0, 0xff00_0000, 0xff00_0000, 0x0100_0000], "no crSat: wraps");
    }

    /// vsrs.d16.s32 with saturation: crSRSSign picks the signed or unsigned 16-bit range.
    #[test]
    fn vsrs_d16_follows_crsrssign() {
        let run = |sign: bool| -> u32 {
            let mut rig = Rig::new();
            let mut acc = [0u64; 8];
            acc[0] = 0x0001_0000_ffff_ff00; // lanes -256, 0x10000
            rig.ctx.accumulator.write(1, acc);
            rig.ctx.scalar.write(40, 0); // s0
            rig.ctx.srs_config.saturation_mode = 1;
            rig.ctx.srs_config.srs_sign = sign;
            rig.run(&VSRS_D16_S32);
            rig.settle(4);
            rig.ctx.vector.read(0)[0]
        };
        assert_eq!(run(false), 0xffff_0000, "unsigned: -256 -> 0, 0x10000 -> 0xffff");
        assert_eq!(run(true), 0x7fff_ff00, "signed: -256 stays, 0x10000 -> 0x7fff");
    }

    /// vlda.ups reads crUPSSign with its shift at operand cycle 7 (issue+6) and crSat at
    /// operand cycle 8 (issue+7): Chess and Peano write both in the load's shadow. Lane 0 is
    /// the `lups` chess-full value 0x846d4520 << 7 (silicon: 0xffffffc2_36a29000 signed).
    #[test]
    fn vlda_ups_reads_its_sign_and_saturation_late() {
        // Lane 0 of bmh0, with `set` applied once `after` bundles have run past the load.
        let run = |set: &dyn Fn(&mut ExecutionContext), after: usize| -> u64 {
            let mut rig = Rig::new();
            let base = 0x100usize;
            let mut words = [0u32; 8];
            words[0] = 0x846d_4520;
            for (i, w) in words.iter().enumerate() {
                rig.tile.data_memory_mut()[base + 0x20 + 4 * i..base + 0x24 + 4 * i]
                    .copy_from_slice(&w.to_le_bytes());
            }
            rig.ctx.pointer.write(7, 0x70000 + base as u32);
            rig.ctx.scalar.write(40, 7); // s0
            rig.run(&VLDA_UPS_S64_D32);
            rig.settle(after);
            set(&mut rig.ctx);
            rig.settle(10 - after);
            rig.ctx.accumulator.read(1)[0]
        };
        let sign = |c: &mut ExecutionContext| c.set_control_reg_raw(10, 1);
        assert_eq!(run(&sign, 5), 0xffff_ffc2_36a2_9000, "written before the issue+6 read");
        assert_eq!(run(&sign, 6), 0x0000_0042_36a2_9000, "written after it");
        // Shifted by 7 the lane fits in 64 bits; shifted by 40 it overflows.
        let run_sat = |after: usize| -> u64 {
            let mut rig = Rig::new();
            let base = 0x100usize;
            rig.tile.data_memory_mut()[base + 0x20..base + 0x24]
                .copy_from_slice(&0x846d_4520u32.to_le_bytes());
            rig.ctx.pointer.write(7, 0x70000 + base as u32);
            rig.ctx.scalar.write(40, 40); // s0: 40-bit shift overflows the 64-bit lane
            rig.ctx.set_control_reg_raw(10, 1);
            rig.run(&VLDA_UPS_S64_D32);
            rig.settle(after);
            rig.ctx.srs_config.saturation_mode = 1;
            rig.settle(10 - after);
            rig.ctx.accumulator.read(1)[0]
        };
        assert_eq!(run_sat(6), 0x8000_0000_0000_0000, "crSat written before the issue+7 read");
        assert_eq!(run_sat(7), 0x6d45_2000_0000_0000, "written after it: wraps");
    }
}
