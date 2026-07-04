//! Register/immediate arithmetic execute: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`, plus the M2a integer/logical/conditional-move/min-max family.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`MoviN`/`Movi`/`MovN`/
/// `Or`/`Extui` plus the M2a arithmetic/logical/cmov/minmax group); `None`
/// otherwise, so `step()` tries the next category.
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::MoviN { t, imm } | Op::Movi { t, imm } => {
            cpu.regs.write_ar(*t, *imm as u32);
        }
        Op::MovN { t, s } => {
            let v = cpu.regs.read_ar(*s);
            cpu.regs.write_ar(*t, v);
        }
        Op::Or { r, s, t } => {
            let v = cpu.regs.read_ar(*s) | cpu.regs.read_ar(*t);
            cpu.regs.write_ar(*r, v);
        }
        Op::Extui { r, t, shiftimm, maskimm } => {
            // `maskimm` here is already the field WIDTH (1-16):
            // decode/arith.rs resolves the raw 4-bit encoded value (0-15) to
            // `maskimm_m1 + 1` before handing it to us. So the mask is
            // `(1<<maskimm)-1`, NOT `(1<<(maskimm+1))-1` -- that second
            // formula is for the raw encoded field decode/arith.rs has
            // already adjusted away.
            let mask = (1u32 << *maskimm) - 1;
            let v = (cpu.regs.read_ar(*t) >> *shiftimm) & mask;
            cpu.regs.write_ar(*r, v);
        }
        // add/add.n: AR[r] = AR[s] + AR[t], two's-complement wrap.
        Op::Add { r, s, t } | Op::AddN { r, s, t } => {
            let v = cpu.regs.read_ar(*s).wrapping_add(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        // addi/addi.n/addmi: AR[t] = AR[s] + imm. `imm` already carries
        // whatever pre-scaling the decoder applied (addmi's <<8), so all
        // three share this arm.
        Op::Addi { t, s, imm } | Op::AddiN { t, s, imm } | Op::Addmi { t, s, imm } => {
            let v = cpu.regs.read_ar(*s).wrapping_add(*imm as u32);
            cpu.regs.write_ar(*t, v);
        }
        // sub: AR[r] = AR[s] - AR[t].
        Op::Sub { r, s, t } => {
            let v = cpu.regs.read_ar(*s).wrapping_sub(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        // addx2/4/8: AR[r] = (AR[s]<<k) + AR[t], k=1/2/3.
        Op::Addx2 { r, s, t } => {
            let v = (cpu.regs.read_ar(*s) << 1).wrapping_add(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        Op::Addx4 { r, s, t } => {
            let v = (cpu.regs.read_ar(*s) << 2).wrapping_add(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        Op::Addx8 { r, s, t } => {
            let v = (cpu.regs.read_ar(*s) << 3).wrapping_add(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        // subx8: AR[r] = (AR[s]<<3) - AR[t] -- the shifted `s` is the
        // minuend, `t` the subtrahend (see decode/mod.rs's Op::Subx8 doc).
        Op::Subx8 { r, s, t } => {
            let v = (cpu.regs.read_ar(*s) << 3).wrapping_sub(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        Op::And { r, s, t } => {
            let v = cpu.regs.read_ar(*s) & cpu.regs.read_ar(*t);
            cpu.regs.write_ar(*r, v);
        }
        Op::Xor { r, s, t } => {
            let v = cpu.regs.read_ar(*s) ^ cpu.regs.read_ar(*t);
            cpu.regs.write_ar(*r, v);
        }
        // neg: AR[r] = -AR[t] (two's complement; wrapping_neg handles
        // i32::MIN without panicking, matching the no-traps policy).
        Op::Neg { r, t } => {
            let v = cpu.regs.read_ar(*t).wrapping_neg();
            cpu.regs.write_ar(*r, v);
        }
        // abs: AR[r] = |AR[t]| as signed (wrapping_abs: i32::MIN stays
        // i32::MIN, the two's-complement edge case, rather than panicking).
        Op::Abs { r, t } => {
            let v = (cpu.regs.read_ar(*t) as i32).wrapping_abs() as u32;
            cpu.regs.write_ar(*r, v);
        }
        // moveqz/movnez: conditional move on AR[t] == 0 / != 0.
        Op::Moveqz { r, s, t } => {
            if cpu.regs.read_ar(*t) == 0 {
                let v = cpu.regs.read_ar(*s);
                cpu.regs.write_ar(*r, v);
            }
        }
        Op::Movnez { r, s, t } => {
            if cpu.regs.read_ar(*t) != 0 {
                let v = cpu.regs.read_ar(*s);
                cpu.regs.write_ar(*r, v);
            }
        }
        // min: signed minimum.
        Op::Min { r, s, t } => {
            let (sv, tv) = (cpu.regs.read_ar(*s) as i32, cpu.regs.read_ar(*t) as i32);
            cpu.regs.write_ar(*r, if sv < tv { sv } else { tv } as u32);
        }
        // minu/maxu: unsigned minimum/maximum.
        Op::Minu { r, s, t } => {
            let v = cpu.regs.read_ar(*s).min(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        Op::Maxu { r, s, t } => {
            let v = cpu.regs.read_ar(*s).max(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

#[cfg(test)]
mod tests {
    use super::super::{Cpu, Step};
    use crate::firmware::mmio::Bus;

    // movi a2, 5 (0c 52 as movi.n) then or a3,a2,a2 (20 32 20) -> a3 == 5.
    #[test]
    fn executes_movi_n_then_or() {
        let rom = vec![0x0c, 0x52, /* movi.n a2,5 */ 0x30, 0x32, 0x20 /* or a3,a2,a2 */];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 5);
    }

    #[test]
    fn executes_wide_movi() {
        // The 3-byte movi form (distinct from movi.n above). Vector from
        // M1.1: d2 a0 ac = movi a13, 172.
        let rom = vec![0xd2, 0xa0, 0xac];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(13), 172);
    }

    #[test]
    fn executes_mov_n() {
        // mov.n a11, a3 -- `bd 03` (M1.1 vector).
        let rom = vec![0xbd, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0x1234);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(11), 0x1234);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_extui_extracts_low_16_bits() {
        // extui a3,a3,0,16 -- `30 30 f4` (M1.1 vector): shiftimm=0,
        // maskimm=16 (a 16-bit field). All-ones source isolates whether the
        // mask width is `maskimm` (correct, giving 0xffff) vs. an
        // off-by-one `maskimm+1` (would give 0x1ffff).
        let rom = vec![0x30, 0x30, 0xf4];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0xffff_ffff);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 0xffff);
    }

    #[test]
    fn executes_add() {
        // add a2,a6,a2 -- `20 26 80` (task-3 vector).
        let rom = vec![0x20, 0x26, 0x80];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(6, 7);
        cpu.regs.write_ar(2, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 12);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_add_n() {
        // add.n a14,a8,a7 -- `7a e8` (task-3 vector).
        let rom = vec![0x7a, 0xe8];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(8, 100);
        cpu.regs.write_ar(7, 23);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(14), 123);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_addi_sign_extends_and_wraps() {
        // addi a6,a15,0x27 (39) -- `62 cf 27` (task-3 vector). Also proves
        // two's-complement wraparound: AR[15] is near u32::MAX, and the sum
        // wraps rather than panicking or saturating.
        let rom = vec![0x62, 0xcf, 0x27];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(15, 0xffff_ffe0); // -32
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 7); // -32 + 39 = 7
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addi_n() {
        // addi.n a4,a2,1 -- `1b 42`, the DISTINCT-register synthetic vector
        // (objdump-confirmed `addi.n a4,a2,1`), NOT the s==t==2 firmware
        // vector -- so the exec test pins that the destination is `t` (a4)
        // and the source is `s` (a2), not a swapped/aliased mapping. a2 (src)
        // holds 10, a4 (dest) is left at 0 beforehand; a correct run writes
        // 11 to a4 and leaves a2 == 10.
        let rom = vec![0x1b, 0x42];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 10);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 11); // dest = src + 1
        assert_eq!(cpu.regs.read_ar(2), 10); // src untouched
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_addmi_shifts_immediate_by_8() {
        // addmi a2,a7,0x200 -- `22 d7 02` (task-3 vector). Pins the shift:
        // an unshifted-imm bug would add 2 instead of 0x200.
        let rom = vec![0x22, 0xd7, 0x02];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(7, 0x100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0x300);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_sub() {
        // sub a4,a4,a9 -- `90 44 c0` (task-3 vector).
        let rom = vec![0x90, 0x44, 0xc0];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(4, 10);
        cpu.regs.write_ar(9, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 7);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx4() {
        // addx4 a2,a2,a2 -- `20 22 a0` (task-3 vector), but written with
        // distinct source registers to pin the (AR[s]<<2)+AR[t] formula
        // (the real vector aliases all three to a2, which can't distinguish
        // "shift s" from "shift t"). Hand-encoded: addx4 a5,a2,a3 (r=5,s=2,
        // t=3) -- op1=0,op2=0xA per decode/arith.rs -- byte0=(t<<4)|0=0x30,
        // byte1=(r<<4)|s=0x52, byte2=(op2<<4)|op1=0xa0.
        let rom = vec![0x30, 0x52, 0xa0];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 3); // s
        cpu.regs.write_ar(3, 100); // t
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), (3 << 2) + 100); // 112
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx2() {
        // addx2 a3,a2,a2 -- `20 32 90` (task-3 vector).
        let rom = vec![0x20, 0x32, 0x90];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 9);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), (9 << 1) + 9); // 27
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx8() {
        // addx8 a5,a2,a5 -- `50 52 b0` (task-3 vector).
        let rom = vec![0x50, 0x52, 0xb0];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 4);
        cpu.regs.write_ar(5, 7);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), (4 << 3) + 7); // 39
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_subx8_pins_operand_order() {
        // subx8 a9,a8,a8 -- `80 98 f0` (task-3 vector), but hand-encoded
        // with distinct s/t to pin the operand order: subx8 a4,a5,a6
        // (r=4,s=5,t=6) -- op1=0,op2=0xF -- byte0=(t<<4)|0=0x60,
        // byte1=(r<<4)|s=0x45, byte2=(op2<<4)|op1=0xf0. If the minuend were
        // AR[t] instead of the shifted AR[s], or the shift applied to the
        // wrong operand, this would diverge from (AR[s]<<3)-AR[t].
        let rom = vec![0x60, 0x45, 0xf0];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(5, 20); // s (shifted minuend)
        cpu.regs.write_ar(6, 3); // t (subtrahend)
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), (20u32 << 3).wrapping_sub(3)); // 157
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_and() {
        // and a7,a2,a3 -- `30 72 10` (task-3 vector).
        let rom = vec![0x30, 0x72, 0x10];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xF0F0_F0F0);
        cpu.regs.write_ar(3, 0xFF00_FF00);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 0xF000_F000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_xor() {
        // xor a8,a5,a8 -- `80 85 30` (task-3 vector).
        let rom = vec![0x80, 0x85, 0x30];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(5, 0xFF00_FF00);
        cpu.regs.write_ar(8, 0x0F0F_0F0F);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), 0xF00F_F00F);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_neg() {
        // neg a8,a7 -- `70 80 60` (task-3 vector).
        let rom = vec![0x70, 0x80, 0x60];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(7, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), (-5i32) as u32);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_abs_on_negative() {
        // abs a2,a2 -- `20 21 60` (task-3 vector), aliased dest/src; the
        // absolute value must still land correctly on the shared register.
        let rom = vec![0x20, 0x21, 0x60];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, (-17i32) as u32);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 17);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_moveqz_both_conditions() {
        // moveqz a15,a2,a4 -- `40 f2 83` (task-3 vector). AR[t]==0 -> moved;
        // AR[t]!=0 -> unchanged.
        let rom = vec![0x40, 0xf2, 0x83];

        let mut bus = Bus::new(rom.clone());
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xAAAA);
        cpu.regs.write_ar(4, 0); // condition true
        cpu.regs.write_ar(15, 0x1111);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(15), 0xAAAA);

        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xAAAA);
        cpu.regs.write_ar(4, 1); // condition false
        cpu.regs.write_ar(15, 0x1111);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(15), 0x1111); // unchanged
    }

    #[test]
    fn executes_movnez_both_conditions() {
        // movnez a10,a3,a5 -- `50 a3 93` (task-3 vector). AR[t]!=0 ->
        // moved; AR[t]==0 -> unchanged.
        let rom = vec![0x50, 0xa3, 0x93];

        let mut bus = Bus::new(rom.clone());
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0xBBBB);
        cpu.regs.write_ar(5, 7); // condition true
        cpu.regs.write_ar(10, 0x2222);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(10), 0xBBBB);

        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0xBBBB);
        cpu.regs.write_ar(5, 0); // condition false
        cpu.regs.write_ar(10, 0x2222);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(10), 0x2222); // unchanged
    }

    #[test]
    fn executes_min_vs_minu_diverge_on_high_bit() {
        // min a2,a10,a2 -- `20 2a 43` (task-3 vector). One operand has the
        // high bit set (a large unsigned value / a negative signed value);
        // signed min must pick it, since it's the more-negative signed
        // value, while unsigned min (tested below with minu) must NOT.
        let rom = vec![0x20, 0x2a, 0x43];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(10, 0x8000_0000); // i32::MIN, the smaller signed value
        cpu.regs.write_ar(2, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0x8000_0000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_minu_treats_high_bit_as_large() {
        // minu a9,a4,a9 -- `90 94 63` (task-3 vector). Same high-bit-set
        // value as the min test above, but unsigned: it's the LARGER value,
        // so minu must pick the small operand instead.
        let rom = vec![0x90, 0x94, 0x63];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(4, 5);
        cpu.regs.write_ar(9, 0x8000_0000); // unsigned: huge
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 5);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_maxu() {
        // maxu a5,a7,a4 -- `40 57 73` (task-3 vector).
        let rom = vec![0x40, 0x57, 0x73];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(7, 0x8000_0000); // unsigned: huge
        cpu.regs.write_ar(4, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x8000_0000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_mov_as_or_with_s_eq_t() {
        // mov a4,a2 (== or a4,a2,a2) -- `20 42 20` (task-3 vector). Ghidra's
        // `mov` pseudo-op is exactly this Or arm with s==t; no separate
        // Mov variant/exec arm exists or is needed.
        let rom = vec![0x20, 0x42, 0x20];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xCAFE);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xCAFE);
        assert_eq!(cpu.pc, 3);
    }
}
