//! Producers of the status flags `srSRS_of`, `srUPS_of` and `srCarry`.
//!
//! TableGen lists `Defs = [srSRS_of]` on every SRS form (`vsrs`, `vsrsm`, `vst[.2d|.3d].srs`)
//! and `Defs = [srUPS_of]` on every UPS form (`vups`, `vlda[.2d|.3d].ups`). Silicon (satflags
//! probe suite, 5356 records over 74 forms) gives the semantics:
//!
//! - The flag is raised when **any lane** of the conversion leaves the output range: the
//!   shifted (and, for SRS, rounded) value before any clamp. crSat does not matter: crSat 0
//!   wraps the data and still raises the flag. The range is that of the output lane under the
//!   run-time sign of a `d` form; the symmetric SRS modes (crSat 2 and 3) exclude the signed
//!   minimum, so a lane landing exactly on it raises the flag there.
//! - It is **sticky**: a conversion that stays in range leaves it set; only a `mov` to the
//!   status register clears it.
//! - It becomes visible to a `mov rX, srY_of` issued [`SRS_OF_VISIBLE_AT`] bundles after an
//!   SRS (the itinerary's operand cycle for the def agrees), and as many bundles after a UPS
//!   as its accumulator result cycle (3 for `vups`, 9 for `vlda.ups`; the itinerary's 1 and 8
//!   for the flag are early by two and one).

//!
//! `srCarry` (satflags carry probe suite, 1112 records) is defined by the instructions whose
//! TableGen record lists `Defs = [srCarry]` (`add`, register and immediate, `sub`, `adc`,
//! `sbc`, `abs`, `jnzd`); every other instruction leaves it alone (`add.nc` included). It is
//! overwritten, not sticky, and a reader issued in the same bundle sees the old value, one
//! issued in the next bundle the new one (`adc` / `sbc` included). Values: the add family sets
//! the carry out of the 32-bit unsigned sum (an immediate is sign-extended; `jnzd` adds -1, so
//! its carry says the source was nonzero), the subtract family the **borrow** (1 when the
//! subtrahend plus borrow-in exceeds the minuend; `sbc` subtracts the flag as borrow-in), `abs`
//! the sign bit of its source.

/// Issued bundles after an SRS before its `srSRS_of` is visible (silicon; AIE2Schedule.td's
/// `/*def:srSRS_of*/3` for II_VSRS, II_VSRSM and II_VST*_SRS).
pub(crate) const SRS_OF_VISIBLE_AT: u64 = 3;

#[cfg(test)]
mod execution {
    use crate::device::tile::Tile;
    use crate::interpreter::bundle::VliwBundle;
    use crate::interpreter::decode::InstructionDecoder;
    use crate::interpreter::execute::cycle_accurate::CycleAccurateExecutor;
    use crate::interpreter::state::{ExecutionContext, SRS_OF_REG_INDEX, UPS_OF_REG_INDEX};
    use crate::interpreter::traits::{Decoder, Executor};

    const VSRS_D16_S32: [u8; 4] = [0x99, 0x03, 0x1c, 0x08]; // vsrs.d16.s32 wl0, bmh0, s0
    const VUPS_S32_D16: [u8; 4] = [0xd9, 0x07, 0x71, 0x18]; // vups.s32.d16 bmh0, wl0, s3
    const VLDA_UPS_S64_D32: [u8; 4] = [0x19, 0x83, 0x2c, 0x07]; // vlda.ups.s64.d32 bmh0, s0, [p7, #0x20]
    const MOV_R1_SRS_OF: [u8; 4] = [0x59, 0xb6, 0x46, 0x18]; // mov r1, srSRS_of
    const MOV_R1_UPS_OF: [u8; 4] = [0x59, 0xb6, 0x49, 0x18]; // mov r1, srUPS_of
    const MOV_SRS_OF_R0: [u8; 4] = [0x59, 0x16, 0x50, 0x1b]; // mov srSRS_of, r0
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
        /// What `mov r1, <flag>` issued `k` bundles after the last instruction returns.
        fn read_at(&mut self, k: usize, mov: &[u8]) -> u32 {
            self.settle(k - 1);
            self.run(mov);
            self.settle(4);
            self.ctx.scalar.read(1)
        }
        /// bmh0 (acc register 1) with 32-bit lanes 0 and 1.
        fn acc32(&mut self, lane0: i32, lane1: i32) {
            let mut acc = [0u64; 8];
            acc[0] = lane0 as u32 as u64 | (lane1 as u32 as u64) << 32;
            self.ctx.accumulator.write(1, acc);
        }
        fn srs(&mut self, sat: u8, sign: bool) {
            self.ctx.scalar.write(40, 0); // s0
            self.ctx.srs_config.saturation_mode = sat;
            self.ctx.srs_config.srs_sign = sign;
            self.run(&VSRS_D16_S32);
        }
    }

    /// vsrs.d16.s32 sets srSRS_of three bundles after its issue (silicon: a read issued 1 or
    /// 2 bundles later sees the old value).
    #[test]
    fn vsrs_raises_srs_of_three_bundles_later() {
        for k in 1..=6 {
            let mut rig = Rig::new();
            rig.acc32(0x1_0000, 0);
            rig.srs(1, true);
            assert_eq!(rig.read_at(k, &MOV_R1_SRS_OF), (k >= 3) as u32, "read {k} bundles later");
        }
    }

    /// Any lane out of range raises it, whatever crSat (crSat 0 wraps the data and still
    /// raises the flag); a lane in range does not.
    #[test]
    fn overflow_raises_the_flag_under_every_crsat() {
        for sat in 0..4 {
            let mut rig = Rig::new();
            rig.acc32(5, -40_000);
            rig.srs(sat, true);
            assert_eq!(rig.read_at(4, &MOV_R1_SRS_OF), 1, "crSat {sat}");
            let mut rig = Rig::new();
            rig.acc32(5, -32_767);
            rig.srs(sat, true);
            assert_eq!(rig.read_at(4, &MOV_R1_SRS_OF), 0, "crSat {sat}, in range");
        }
    }

    /// The symmetric modes (crSat 2, 3) exclude the signed minimum from the range.
    #[test]
    fn the_signed_minimum_overflows_the_symmetric_range() {
        for (sat, expect) in [(0, 0), (1, 0), (2, 1), (3, 1)] {
            let mut rig = Rig::new();
            rig.acc32(-32_768, 0);
            rig.srs(sat, true);
            assert_eq!(rig.read_at(4, &MOV_R1_SRS_OF), expect, "crSat {sat}");
        }
    }

    /// A `d` form judges the range with its run-time sign: -1 overflows an unsigned output only.
    #[test]
    fn a_d_form_takes_the_range_of_its_run_time_sign() {
        for (sign, expect) in [(false, 1), (true, 0)] {
            let mut rig = Rig::new();
            rig.acc32(-1, 0);
            rig.srs(1, sign);
            assert_eq!(rig.read_at(4, &MOV_R1_SRS_OF), expect, "crSRSSign {sign}");
        }
    }

    /// The flag is sticky: a later in-range conversion leaves it set; a move clears it.
    #[test]
    fn the_flag_is_sticky_until_written() {
        let mut rig = Rig::new();
        rig.acc32(0x1_0000, 0);
        rig.srs(1, true);
        rig.settle(4);
        rig.acc32(1, 2);
        rig.srs(1, true);
        assert_eq!(rig.read_at(4, &MOV_R1_SRS_OF), 1, "an in-range SRS does not clear it");
        rig.ctx.scalar.write(0, 0);
        rig.run(&MOV_SRS_OF_R0);
        assert_eq!(rig.read_at(2, &MOV_R1_SRS_OF), 0, "mov srSRS_of, r0 clears it");
        assert_eq!(rig.ctx.scalar.read(SRS_OF_REG_INDEX), 0);
    }

    /// vups.s32.d16: lane 0x8000 << 16 overflows an unsigned source, not a signed one; the flag
    /// is visible three bundles after the vups (its accumulator result cycle).
    #[test]
    fn vups_raises_ups_of_at_its_result_cycle() {
        for (sign, k, expect) in [(0, 2, 0), (0, 3, 1), (0, 6, 1), (1, 6, 0)] {
            let mut rig = Rig::new();
            let mut v = [0u32; 8];
            v[0] = 0x8000;
            rig.ctx.vector.write(0, v);
            rig.ctx.scalar.write(40 + 3, 16); // s3
            rig.ctx.set_control_reg_raw(10, sign);
            rig.run(&VUPS_S32_D16);
            assert_eq!(rig.read_at(k, &MOV_R1_UPS_OF), expect, "crUPSSign {sign}, read at {k}");
        }
        assert_eq!(Rig::new().ctx.scalar.read(UPS_OF_REG_INDEX), 0);
    }

    /// vlda.ups: visible nine bundles after the load (its accumulator result cycle), with the
    /// sign and crSat sampled late as for the data.
    #[test]
    fn vlda_ups_raises_ups_of_nine_bundles_later() {
        for k in [8, 9, 12] {
            let mut rig = Rig::new();
            let base = 0x100usize;
            rig.tile.data_memory_mut()[base + 0x20..base + 0x24]
                .copy_from_slice(&0x846d_4520u32.to_le_bytes());
            rig.ctx.pointer.write(7, 0x70000 + base as u32);
            rig.ctx.scalar.write(40, 40); // s0: a 40-bit shift overflows the 64-bit lane
            rig.ctx.set_control_reg_raw(10, 1);
            rig.run(&VLDA_UPS_S64_D32);
            assert_eq!(rig.read_at(k, &MOV_R1_UPS_OF), (k >= 9) as u32, "read {k} bundles later");
        }
    }
}

#[cfg(test)]
mod carry {
    use crate::device::tile::Tile;
    use crate::interpreter::decode::InstructionDecoder;
    use crate::interpreter::execute::cycle_accurate::CycleAccurateExecutor;
    use crate::interpreter::state::{ExecutionContext, CARRY_REG_INDEX};
    use crate::interpreter::traits::{Decoder, Executor};

    const ADD: [u8; 4] = [0x99, 0x30, 0x88, 0x10]; // add r4, r2, r3
    const SUB: [u8; 4] = [0x99, 0x31, 0x88, 0x10]; // sub r4, r2, r3
    const ADC: [u8; 4] = [0x99, 0x32, 0x88, 0x10]; // adc r4, r2, r3
    const SBC: [u8; 4] = [0x99, 0x33, 0x88, 0x10]; // sbc r4, r2, r3
    const ABS: [u8; 4] = [0x19, 0x1c, 0x89, 0x10]; // abs r4, r2
    const ADD_M1: [u8; 4] = [0x19, 0xff, 0x89, 0x10]; // add r4, r2, #-1
    const ADD_NC_M1: [u8; 4] = [0xf9, 0x7f, 0x01, 0x19]; // add.nc r4, r2, #-1
    const LTU: [u8; 4] = [0x99, 0x3c, 0x88, 0x10]; // ltu r4, r2, r3
    const JNZD: [u8; 4] = [0x19, 0x4c, 0x89, 0x10]; // jnzd r4, r2, p5
    const MOV_R1_CARRY: [u8; 4] = [0x59, 0xb6, 0x40, 0x18]; // mov r1, srCarry
    const MOV_CARRY_R6: [u8; 4] = [0x59, 0x16, 0x53, 0x18]; // mov srCarry, r6
    const ADD_WITH_READ: [u8; 8] = [0x43, 0x28, 0x5b, 0x00, 0x85, 0x41, 0x04, 0x00]; // add r4, r2, r3; mov r8, srCarry
    const NOP: [u8; 2] = [0x01, 0x00];

    struct Rig {
        decoder: InstructionDecoder,
        exec: CycleAccurateExecutor,
        ctx: ExecutionContext,
        tile: Tile,
    }

    impl Rig {
        /// r2 = a, r3 = b, srCarry = cin.
        fn new(a: u32, b: u32, cin: u32) -> Self {
            let mut rig = Rig {
                decoder: InstructionDecoder::load_default(),
                exec: CycleAccurateExecutor::new(),
                ctx: ExecutionContext::new(),
                tile: Tile::compute(0, 2),
            };
            rig.ctx.scalar.write(2, a);
            rig.ctx.scalar.write(3, b);
            rig.ctx.scalar.write(6, cin);
            rig.ctx.pointer.write(5, 0x100);
            rig.run(&MOV_CARRY_R6);
            rig.settle(2);
            rig
        }
        fn run(&mut self, bytes: &[u8]) {
            let b = self.decoder.decode(bytes, 0).expect("decode");
            self.exec.execute(&b, &mut self.ctx, &mut self.tile);
        }
        fn settle(&mut self, n: usize) {
            for _ in 0..n {
                self.run(&NOP);
            }
        }
        /// Run `op`, then read srCarry with a `mov` in the next bundle: (r4, carry).
        fn op(a: u32, b: u32, cin: u32, op: &[u8]) -> (u32, u32) {
            let mut rig = Rig::new(a, b, cin);
            rig.run(op);
            rig.run(&MOV_R1_CARRY);
            rig.settle(3);
            (rig.ctx.scalar.read(4), rig.ctx.scalar.read(1))
        }
    }

    #[test]
    fn add_sets_the_unsigned_carry_out() {
        assert_eq!(Rig::op(0xffff_ffff, 1, 0, &ADD), (0, 1));
        assert_eq!(Rig::op(0x7fff_ffff, 1, 1, &ADD), (0x8000_0000, 0), "signed overflow is not carry");
        assert_eq!(Rig::op(5, 0, 0, &ADD_M1), (4, 1), "the immediate is sign-extended");
        assert_eq!(Rig::op(0, 0, 1, &ADD_M1), (0xffff_ffff, 0));
    }

    #[test]
    fn sub_sets_the_borrow() {
        assert_eq!(Rig::op(0, 1, 0, &SUB), (0xffff_ffff, 1));
        assert_eq!(Rig::op(2, 1, 1, &SUB), (1, 0));
        assert_eq!(Rig::op(1, 1, 1, &SUB), (0, 0));
    }

    #[test]
    fn adc_adds_the_carry_in() {
        assert_eq!(Rig::op(1, 0xffff_ffff, 1, &ADC), (1, 1));
        assert_eq!(Rig::op(0, 0x7fff_ffff, 1, &ADC), (0x8000_0000, 0));
        assert_eq!(Rig::op(0, 0xffff_ffff, 1, &ADC), (0, 1));
    }

    #[test]
    fn sbc_subtracts_the_flag_as_borrow_in() {
        assert_eq!(Rig::op(0, 0xffff_ffff, 0, &SBC), (1, 1));
        assert_eq!(Rig::op(1, 0, 1, &SBC), (0, 0));
        assert_eq!(Rig::op(0, 0x7fff_ffff, 1, &SBC), (0x8000_0000, 1));
        assert_eq!(Rig::op(2, 1, 0, &SBC), (1, 0));
    }

    #[test]
    fn abs_and_jnzd() {
        assert_eq!(Rig::op(0x8000_0001, 0, 0, &ABS), (0x7fff_ffff, 1));
        assert_eq!(Rig::op(5, 0, 1, &ABS), (5, 0));
        assert_eq!(Rig::op(1, 0, 0, &JNZD).1, 1, "jnzd: the source was nonzero");
        assert_eq!(Rig::op(0, 0, 1, &JNZD).1, 0);
        assert_eq!(Rig::op(0x8000_0000, 0, 0, &JNZD), (0x7fff_ffff, 1), "the decrement wraps");
    }

    #[test]
    fn instructions_without_the_def_leave_it_alone() {
        for cin in [0, 1] {
            assert_eq!(Rig::op(0, 0, cin, &ADD_NC_M1).1, cin, "add.nc");
            assert_eq!(Rig::op(0, 1, cin, &LTU).1, cin, "ltu");
        }
    }

    #[test]
    fn a_same_bundle_reader_sees_the_old_carry() {
        let mut rig = Rig::new(0xffff_ffff, 1, 0);
        rig.run(&ADD_WITH_READ);
        rig.settle(3);
        assert_eq!(rig.ctx.scalar.read(8), 0, "same bundle: old");
        assert_eq!(rig.ctx.scalar.read(CARRY_REG_INDEX), 1);
    }

    #[test]
    fn adc_one_bundle_after_add_takes_its_carry() {
        let mut rig = Rig::new(0xffff_ffff, 1, 0);
        rig.run(&ADD);
        rig.run(&ADC);
        rig.settle(3);
        assert_eq!(rig.ctx.scalar.read(4), 1, "0xffffffff + 1 + carry 1");
    }
}
