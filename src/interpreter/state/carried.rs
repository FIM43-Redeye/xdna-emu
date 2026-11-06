//! The core state a launch's core reset keeps: what silicon carries from one kernel launch to
//! the next on the same core.
//!
//! NPU1 silicon keeps every data register of a core across launches (separate processes,
//! separate xclbins, launches on other tiles in between, idle gaps up to 300 s; probe
//! `build/experiments/harness/scrub/step1`, 17 launches) and resets only the control registers
//! (0, `crSCDEn` = `crMCDEn` = 1), the status registers (0) and `le` (0xFFFFF). The load-port
//! rings survive too (`build/experiments/harness/port-latch`). [`CarriedCoreState`] is exactly
//! the kept part; everything else is rebuilt by the reset (`ExecutionContext::after_reset`).
//!
//! The struct is also the schema of the cross-process device-state file
//! (`crate::interpreter::engine::device_state`), so its serialization is derived from it.

use serde::{Deserialize, Serialize};

use super::context::ExecutionContext;
use super::load_ring::LoadPortRings;
use super::registers::{
    DP_REG_INDEX, LC_REG_INDEX, LR_REG_INDEX, LS_REG_INDEX, NUM_ACCUMULATOR_REGS, NUM_MASK_REGS,
    NUM_MODIFIER_REGS, NUM_POINTER_REGS, NUM_SCALAR_GPRS, NUM_SHIFT_REGS, NUM_VECTOR_REGS, SHIFT_REG_BASE,
};

/// What a core keeps across a launch's core reset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CarriedCoreState {
    /// r0-r31.
    pub r: [u32; NUM_SCALAR_GPRS],
    /// The link register. Every launch's crt0 and call define it before it is read, so whether
    /// silicon keeps it is unobservable; it is a data register like its neighbours.
    pub lr: u32,
    pub ls: u32,
    pub lc: u32,
    pub dp: u32,
    /// s0-s3.
    pub s: [u32; NUM_SHIFT_REGS as usize],
    /// p0-p7.
    pub p: [u32; NUM_POINTER_REGS],
    /// The stack pointer. crt0 defines it before any read (unobservable, as `lr`).
    pub sp: u32,
    /// m0-m7, dn0-dn7, dj0-dj7, dc0-dc7 (the modifier file's index order).
    pub modifiers: [u32; NUM_MODIFIER_REGS],
    /// The 256-bit w file (wl0, wh0, wl1, ...), eight words each.
    pub w: [[u32; 8]; NUM_VECTOR_REGS],
    /// The 512-bit bm file (bml0, bmh0, ...), eight lanes each.
    pub bm: [[u64; 8]; NUM_ACCUMULATOR_REGS],
    /// q0-q3.
    pub q: [[u32; 4]; NUM_MASK_REGS],
    /// Both load ports' rings.
    pub load_rings: LoadPortRings,
}

impl CarriedCoreState {
    /// The kept state of `ctx`. Every write still in flight lands first, in place (silicon's
    /// pipeline completes them; a core that ran `done` has none left in practice). In place
    /// rather than on a clone: the context carries its event trace, and it is being retired.
    pub fn capture(c: &mut ExecutionContext) -> Self {
        c.force_commit_all_pending();
        let scalar = |i: u8| c.scalar.read(i);
        Self {
            r: std::array::from_fn(|i| scalar(i as u8)),
            lr: scalar(LR_REG_INDEX),
            ls: scalar(LS_REG_INDEX),
            lc: scalar(LC_REG_INDEX),
            dp: scalar(DP_REG_INDEX),
            s: std::array::from_fn(|i| scalar(SHIFT_REG_BASE + i as u8)),
            p: std::array::from_fn(|i| c.pointer.read(i as u8)),
            sp: c.sp(),
            modifiers: std::array::from_fn(|i| c.modifier.read(i as u8)),
            w: std::array::from_fn(|i| c.vector.read(i as u8)),
            bm: std::array::from_fn(|i| c.accumulator.read_committed(i as u8)),
            q: std::array::from_fn(|i| c.mask.read(i as u8)),
            load_rings: c.load_rings.clone(),
        }
    }

    /// Write the kept state into `ctx` (a context fresh from a reset).
    pub fn restore(&self, ctx: &mut ExecutionContext) {
        for (i, v) in self.r.iter().enumerate() {
            ctx.scalar.write(i as u8, *v);
        }
        ctx.scalar.write(LR_REG_INDEX, self.lr);
        ctx.scalar.write(LS_REG_INDEX, self.ls);
        ctx.scalar.write(LC_REG_INDEX, self.lc);
        ctx.scalar.write(DP_REG_INDEX, self.dp);
        for (i, v) in self.s.iter().enumerate() {
            ctx.scalar.write(SHIFT_REG_BASE + i as u8, *v);
        }
        for (i, v) in self.p.iter().enumerate() {
            ctx.pointer.write(i as u8, *v);
        }
        ctx.set_sp(self.sp);
        for (i, v) in self.modifiers.iter().enumerate() {
            ctx.modifier.write(i as u8, *v);
        }
        for (i, v) in self.w.iter().enumerate() {
            ctx.vector.write(i as u8, *v);
        }
        for (i, v) in self.bm.iter().enumerate() {
            ctx.accumulator.write(i as u8, *v);
        }
        for (i, v) in self.q.iter().enumerate() {
            ctx.mask.write(i as u8, *v);
        }
        ctx.load_rings = self.load_rings.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::state::load_ring::LoadPort;
    use crate::interpreter::state::registers::*;
    use crate::interpreter::state::scrub::CoreInit;

    fn ctrl_id(name: &str) -> u8 {
        use crate::interpreter::bundle::slot::Operand;
        match crate::interpreter::decode::register_map::classify_reg_name(name).map(|m| m.operand) {
            Some(Operand::ControlReg(id)) => id,
            other => panic!("{name}: {other:?}"),
        }
    }

    /// A context with a known value in every carried register, the rings filled, and every
    /// launch-reset register away from its reset value.
    fn dirty(col: u8, row: u8) -> ExecutionContext {
        let mut ctx = ExecutionContext::new_for_tile(col, row);
        for i in 0..NUM_SCALAR_GPRS as u8 {
            ctx.scalar.write(i, 0x1000 + i as u32);
        }
        ctx.scalar.write(LR_REG_INDEX, 0x2_0000);
        ctx.scalar.write(LS_REG_INDEX, 0x123);
        ctx.scalar.write(LC_REG_INDEX, 7);
        ctx.scalar.write(DP_REG_INDEX, 0x456);
        for i in 0..NUM_SHIFT_REGS {
            ctx.scalar.write(SHIFT_REG_BASE + i, 3 + i as u32);
        }
        for i in 0..NUM_POINTER_REGS as u8 {
            ctx.pointer.write(i, 0x7_0000 + 4 * i as u32);
        }
        ctx.set_sp(0x7_1000);
        for i in 0..NUM_MODIFIER_REGS as u8 {
            ctx.modifier.write(i, 0x30 + i as u32);
        }
        for i in 0..NUM_VECTOR_REGS as u8 {
            ctx.vector.write(i, std::array::from_fn(|k| (i as u32) << 8 | k as u32));
        }
        for i in 0..NUM_ACCUMULATOR_REGS as u8 {
            ctx.accumulator.write(i, std::array::from_fn(|k| (i as u64) << 40 | k as u64));
        }
        for i in 0..NUM_MASK_REGS as u8 {
            ctx.mask.write(i, [i as u32, 1, 2, 3]);
        }
        for k in 0..4u8 {
            ctx.load_rings.pass(LoadPort::A, (k & 1) as usize, Some([0xA0 | k; 16]));
            ctx.load_rings.pass(LoadPort::B, (k & 1) as usize, Some([0xB0 | k; 16]));
        }
        // launch-reset state, all away from reset
        ctx.scalar.write(LE_REG_INDEX, 0x40);
        for i in [
            SPARSE_OF_REG_INDEX,
            COMPR_UF_REG_INDEX,
            SS_STATUS_REG_INDEX,
            MS_STATUS_REG_INDEX,
            CARRY_REG_INDEX,
            SRS_OF_REG_INDEX,
            UPS_OF_REG_INDEX,
            FP_FLAGS_REG_INDEX,
            F2I_FLAGS_REG_INDEX,
            F2F_FLAGS_REG_INDEX,
        ] {
            ctx.scalar.write(i, 1);
        }
        ctx.set_control_reg_raw(ctrl_id("crSat"), 1);
        ctx.set_control_reg_raw(ctrl_id("crSCDEn"), 0);
        ctx.set_control_reg_raw(ctrl_id("crMCDEn"), 0);
        ctx
    }

    #[test]
    fn a_core_reset_keeps_every_data_register_and_both_rings() {
        let mut before = dirty(1, 2);
        let carried = CarriedCoreState::capture(&mut before);
        let after = ExecutionContext::after_reset(1, 2, &carried);
        for i in 0..NUM_SCALAR_GPRS as u8 {
            assert_eq!(after.scalar.read(i), 0x1000 + i as u32, "r{i}");
        }
        assert_eq!(after.lr(), 0x2_0000);
        assert_eq!(after.scalar.read(LS_REG_INDEX), 0x123);
        assert_eq!(after.scalar.read(LC_REG_INDEX), 7);
        assert_eq!(after.scalar.read(DP_REG_INDEX), 0x456);
        for i in 0..NUM_SHIFT_REGS {
            assert_eq!(after.scalar.read(SHIFT_REG_BASE + i), 3 + i as u32, "s{i}");
        }
        for i in 0..NUM_POINTER_REGS as u8 {
            assert_eq!(after.pointer.read(i), 0x7_0000 + 4 * i as u32, "p{i}");
        }
        assert_eq!(after.sp(), 0x7_1000);
        for i in 0..NUM_MODIFIER_REGS as u8 {
            assert_eq!(after.modifier.read(i), 0x30 + i as u32);
        }
        for i in 0..NUM_VECTOR_REGS as u8 {
            assert_eq!(after.vector.read(i), before.vector.read(i));
        }
        for i in 0..NUM_ACCUMULATOR_REGS as u8 {
            assert_eq!(after.accumulator.read(i), before.accumulator.read(i));
        }
        for i in 0..NUM_MASK_REGS as u8 {
            assert_eq!(after.mask.read(i), [i as u32, 1, 2, 3]);
        }
        assert_eq!(after.load_rings, before.load_rings);
        assert_eq!(CarriedCoreState::capture(&mut after.clone()), carried);
    }

    #[test]
    fn a_core_reset_resets_control_status_and_le() {
        let mut before = dirty(1, 2);
        let after = ExecutionContext::after_reset(1, 2, &CarriedCoreState::capture(&mut before));
        assert_eq!(after.scalar.read(LE_REG_INDEX), 0xF_FFFF);
        for i in [
            SPARSE_OF_REG_INDEX,
            COMPR_UF_REG_INDEX,
            SS_STATUS_REG_INDEX,
            MS_STATUS_REG_INDEX,
            CARRY_REG_INDEX,
            SRS_OF_REG_INDEX,
            UPS_OF_REG_INDEX,
            FP_FLAGS_REG_INDEX,
            F2I_FLAGS_REG_INDEX,
            F2F_FLAGS_REG_INDEX,
        ] {
            assert_eq!(after.scalar.read(i), 0, "status slot {i}");
        }
        assert_eq!(after.control_reg_raw(ctrl_id("crSat")), 0);
        assert_eq!(after.control_reg_raw(ctrl_id("crSCDEn")), 1);
        assert_eq!(after.control_reg_raw(ctrl_id("crMCDEn")), 1);
        assert_eq!(
            after.scalar.read(CORE_ID_REG_INDEX),
            ExecutionContext::new_for_tile(1, 2).scalar.read(CORE_ID_REG_INDEX)
        );
        assert_eq!(after.pc(), 0);
    }

    /// Every scalar-file slot is either carried or set by the reset: a new slot must be
    /// classified here before it can silently keep the sentinel or lose its value.
    #[test]
    fn every_scalar_slot_is_carried_or_reset() {
        let carried: Vec<u8> = (0..NUM_SCALAR_GPRS as u8)
            .chain([LR_REG_INDEX, LS_REG_INDEX, LC_REG_INDEX, DP_REG_INDEX])
            .chain((0..NUM_SHIFT_REGS).map(|i| SHIFT_REG_BASE + i))
            .collect();
        let reset = [
            LE_REG_INDEX,
            CORE_ID_REG_INDEX,
            SPARSE_OF_REG_INDEX,
            COMPR_UF_REG_INDEX,
            SS_STATUS_REG_INDEX,
            MS_STATUS_REG_INDEX,
            CARRY_REG_INDEX,
            SRS_OF_REG_INDEX,
            UPS_OF_REG_INDEX,
            FP_FLAGS_REG_INDEX,
            F2I_FLAGS_REG_INDEX,
            F2F_FLAGS_REG_INDEX,
        ];
        for i in 0..NUM_SCALAR_REGS as u8 {
            let n = carried.contains(&i) as u8 + reset.contains(&i) as u8;
            assert_eq!(n, 1, "scalar slot {i} must be exactly one of carried / reset");
        }
    }

    #[test]
    fn a_reset_does_not_reapply_the_power_on_init() {
        // XDNA_EMU_CORE_INIT is the power-on state of a fresh device; a reset keeps what the
        // previous launch left, not the scrub pattern.
        let mut ctx = ExecutionContext::new_for_tile(1, 2);
        CoreInit::Scrub.apply(&mut ctx);
        ctx.scalar.write(5, 0x55);
        let after = ExecutionContext::after_reset(1, 2, &CarriedCoreState::capture(&mut ctx));
        assert_eq!(after.scalar.read(5), 0x55);
        assert_eq!(after.scalar.read(6), crate::interpreter::state::scrub::pattern_word("r6", 0));
    }

    #[test]
    fn capture_lands_writes_in_flight() {
        let mut ctx = ExecutionContext::new_for_tile(1, 2);
        ctx.vector.write(3, [0; 8]);
        ctx.vector.queue_write(3, [9; 8], 6, xdna_archspec::aie2::Bypass::No);
        assert_eq!(CarriedCoreState::capture(&mut ctx).w[3], [9; 8]);
    }

    #[test]
    fn the_state_round_trips_through_json() {
        let c = CarriedCoreState::capture(&mut dirty(2, 3));
        let back: CarriedCoreState = serde_json::from_str(&serde_json::to_string(&c).unwrap()).unwrap();
        assert_eq!(back, c);
    }
}
