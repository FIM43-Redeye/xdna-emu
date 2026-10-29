//! Register/immediate arithmetic execute: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`, plus the M2a integer/logical/conditional-move/min-max family.

use super::{Cpu, Step, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO};
use crate::firmware::xtensa::decode::Op;

/// Execute `op` if it's one of this category's ops (`MoviN`/`Movi`/`MovN`/
/// `Or`/`Extui` plus the M2a arithmetic/logical/cmov/minmax group); `None`
/// otherwise, so `step()` tries the next category.
pub(super) fn exec(cpu: &mut Cpu, op: &Op, pc: u32, len: u8) -> Option<Step> {
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
        // slli: AR[r] = AR[s] << imm. `imm` is already the resolved final
        // shift count (decode/arith.rs), normally 1..31; `checked_shl`
        // guards the (unreachable via the encoding) imm==32 edge rather than
        // panicking, matching this interpreter's no-traps-on-decode policy.
        Op::Slli { r, s, imm } => {
            let v = cpu.regs.read_ar(*s).checked_shl(*imm as u32).unwrap_or(0);
            cpu.regs.write_ar(*r, v);
        }
        // srli: AR[r] = AR[t] >> imm, logical (imm 0..15, always in range).
        Op::Srli { r, t, imm } => {
            let v = cpu.regs.read_ar(*t) >> *imm;
            cpu.regs.write_ar(*r, v);
        }
        // srai: AR[r] = (AR[t] as i32) >> imm, arithmetic sign-fill (imm
        // 0..31, always in range).
        Op::Srai { r, t, imm } => {
            let v = (cpu.regs.read_ar(*t) as i32) >> *imm;
            cpu.regs.write_ar(*r, v as u32);
        }
        // sll: AR[r] = AR[s] << (32 - SAR) -- the left-shift half of the
        // ssl/sll SAR pair. `ssl` only ever sets SAR in 1..=32, so `32 -
        // SAR` is always 0..=31; `checked_shl` still guards a direct
        // `wsr.sar` write putting SAR out of that range, matching this
        // interpreter's no-traps-on-decode policy.
        Op::Sll { r, s } => {
            let shift = 32u32.wrapping_sub(cpu.regs.sar);
            let v = cpu.regs.read_ar(*s).checked_shl(shift).unwrap_or(0);
            cpu.regs.write_ar(*r, v);
        }
        // srl: AR[r] = AR[t] >> SAR -- the right-shift half of the ssr/srl
        // SAR pair. `ssr` only ever sets SAR in 0..=31 (always in range);
        // `checked_shr` guards a direct `wsr.sar` write putting SAR out of
        // range.
        Op::Srl { r, t } => {
            let v = cpu.regs.read_ar(*t).checked_shr(cpu.regs.sar).unwrap_or(0);
            cpu.regs.write_ar(*r, v);
        }
        // src: funnel shift right -- concatenate AR[s] (high) : AR[t] (low)
        // into a 64-bit value, shift right by SAR, take the low 32 bits.
        Op::Src { r, s, t } => {
            let hi = cpu.regs.read_ar(*s) as u64;
            let lo = cpu.regs.read_ar(*t) as u64;
            let combined = (hi << 32) | lo;
            let shifted = combined.checked_shr(cpu.regs.sar).unwrap_or(0);
            cpu.regs.write_ar(*r, shifted as u32);
        }
        // ssl: SAR = 32 - (AR[s] & 31) -- sets up a left shift by AR[s],
        // consumed by `Sll`.
        Op::Ssl { s } => {
            cpu.regs.sar = 32 - (cpu.regs.read_ar(*s) & 31);
        }
        // ssr: SAR = AR[s] & 31 -- sets up a right shift by AR[s], consumed
        // by `Srl`.
        Op::Ssr { s } => {
            cpu.regs.sar = cpu.regs.read_ar(*s) & 31;
        }
        // ssai: SAR = imm, a plain immediate (0..31).
        Op::Ssai { imm } => {
            cpu.regs.sar = *imm as u32;
        }
        // sext: sign-extend AR[s] from bit `imm` (treat bit `imm` as the new
        // sign bit) -- shift the field into the top of the word and back
        // down arithmetically, the same technique as decode's own
        // `sign_extend` helper (not reused directly: it's private to the
        // `decode` module, a sibling, not a descendant, of this one).
        Op::Sext { r, s, imm } => {
            let width = *imm as u32 + 1;
            let shift = 32 - width;
            let v = ((cpu.regs.read_ar(*s) << shift) as i32) >> shift;
            cpu.regs.write_ar(*r, v as u32);
        }
        // nsau: count of leading zero bits of AR[s] (32 when AR[s]==0,
        // exactly `u32::leading_zeros`'s own definition).
        Op::Nsau { t, s } => {
            let v = cpu.regs.read_ar(*s).leading_zeros();
            cpu.regs.write_ar(*t, v);
        }
        // mull: AR[r] = low 32 bits of AR[s] * AR[t]. `u32::wrapping_mul`
        // computes the full 32x32 multiply and truncates to the low word,
        // exactly the hardware's `mull` semantics -- sign-agnostic, since the
        // low word of a product is identical for signed/unsigned inputs.
        Op::Mull { r, s, t } => {
            let v = cpu.regs.read_ar(*s).wrapping_mul(cpu.regs.read_ar(*t));
            cpu.regs.write_ar(*r, v);
        }
        // mul16s: AR[r] = sign_extend16(AR[s]) * sign_extend16(AR[t]),
        // 16x16 signed multiply widened to a full 32-bit result (a 16x16
        // signed product always fits in 32 bits, so no truncation/overflow
        // guard is needed).
        Op::Mul16s { r, s, t } => {
            let sv = cpu.regs.read_ar(*s) as i16 as i32;
            let tv = cpu.regs.read_ar(*t) as i16 as i32;
            cpu.regs.write_ar(*r, (sv * tv) as u32);
        }
        // mul16u: AR[r] = (AR[s]&0xFFFF) * (AR[t]&0xFFFF), 16x16 unsigned
        // multiply (also always fits in 32 bits).
        Op::Mul16u { r, s, t } => {
            let sv = cpu.regs.read_ar(*s) & 0xFFFF;
            let tv = cpu.regs.read_ar(*t) & 0xFFFF;
            cpu.regs.write_ar(*r, sv * tv);
        }
        // quou/remu/rems: unsigned divide, unsigned remainder, signed
        // remainder. Divide-by-zero policy (see Op::Quou's doc): real
        // hardware raises INTEGER_DIVIDE_BY_ZERO_CAUSE BEFORE the divide
        // executes -- now that the general-exception-raise machinery exists
        // (M2a Task 9), this is the REAL raise, replacing the M2a Task 5
        // placeholder (a bare Rust `/0`/`%0` would still panic, so the
        // zero-divisor check must run before any Rust division regardless).
        // Each early-returns via `raise_general_exception`, bypassing this
        // function's common pc+len tail exactly like `system::exec`'s
        // `Syscall` handling.
        Op::Quou { r, s, t } => {
            let (sv, tv) = (cpu.regs.read_ar(*s), cpu.regs.read_ar(*t));
            if tv == 0 {
                return Some(cpu.raise_general_exception(pc, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO));
            }
            cpu.regs.write_ar(*r, sv / tv);
        }
        Op::Remu { r, s, t } => {
            let (sv, tv) = (cpu.regs.read_ar(*s), cpu.regs.read_ar(*t));
            if tv == 0 {
                return Some(cpu.raise_general_exception(pc, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO));
            }
            cpu.regs.write_ar(*r, sv % tv);
        }
        // rems additionally guards `i32::MIN % -1` (once past the
        // zero-divisor check above): unlike unsigned remainder, Rust's
        // plain `%` on `i32` PANICS for this specific input (confirmed
        // empirically -- it panics in every build profile, not just debug
        // overflow-checks, because the host `idiv` instruction itself can't
        // compute it) even though the true remainder is mathematically 0.
        // `wrapping_rem` handles it correctly (returns 0) without panicking,
        // matching real hardware (which has no such artifact -- REMS is
        // architecturally always 0 for that input).
        Op::Rems { r, s, t } => {
            let (sv, tv) = (cpu.regs.read_ar(*s) as i32, cpu.regs.read_ar(*t) as i32);
            if tv == 0 {
                return Some(cpu.raise_general_exception(pc, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO));
            }
            cpu.regs.write_ar(*r, sv.wrapping_rem(tv) as u32);
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

#[cfg(test)]
mod tests {
    use super::super::{mapped_cpu, Step, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO, ExceptionVectors};
    use crate::firmware::xtensa::flat_bus::FlatBus;

    // movi.n a7,43 (2c b7) then or a3,a7,a7 (70 37 20) -> a3 == 43.
    #[test]
    fn executes_movi_n_then_or() {
        let rom = vec![0x2c, 0xb7, /* movi.n a7,43 */ 0x70, 0x37, 0x20 /* or a3,a7,a7 */];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 43);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 43);
    }

    #[test]
    fn executes_wide_movi() {
        // The 3-byte movi form (distinct from movi.n above):
        // 52 ae d4 = movi a5, -300 (imm12 0xed4, sign-extended).
        let rom = vec![0x52, 0xae, 0xd4];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), (-300i32) as u32);
    }

    #[test]
    fn executes_mov_n() {
        // mov.n a6, a9 -- `6d 09`.
        let rom = vec![0x6d, 0x09];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(9, 0x1234);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 0x1234);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_extui_shifts_then_masks() {
        // extui a5,a9,4,8 -- `90 54 74`: shiftimm=4, maskimm=8 (an 8-bit
        // field). Source 0x1234_5678 >> 4 = 0x0123_4567; the mask must be
        // `maskimm` bits wide (giving 0x67), not an off-by-one `maskimm+1`
        // (would give 0x167).
        let rom = vec![0x90, 0x54, 0x74];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(9, 0x1234_5678);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x67);
    }

    #[test]
    fn executes_add() {
        // add a5,a9,a12 -- `c0 59 80`.
        let rom = vec![0xc0, 0x59, 0x80];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(9, 7);
        cpu.regs.write_ar(12, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 12);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_add_n() {
        // add.n a6,a11,a3 -- `3a 6b`.
        let rom = vec![0x3a, 0x6b];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(11, 100);
        cpu.regs.write_ar(3, 23);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 123);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_addi_sign_extends_and_wraps() {
        // addi a9,a4,0x15 (21) -- `92 c4 15`. Also proves
        // two's-complement wraparound: AR[4] is near u32::MAX, and the sum
        // wraps rather than panicking or saturating.
        let rom = vec![0x92, 0xc4, 0x15];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 0xffff_fff0); // -16
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 5); // -16 + 21 = 5
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addi_n() {
        // addi.n a4,a2,1 -- `1b 42`, a DISTINCT-register vector (not an
        // s==t aliased one) -- so the exec test pins that the destination is
        // `t` (a4) and the source is `s` (a2), not a swapped/aliased mapping.
        // a2 (src) holds 10, a4 (dest) is left at 0 beforehand; a correct run
        // writes 11 to a4 and leaves a2 == 10.
        let rom = vec![0x1b, 0x42];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 10);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 11); // dest = src + 1
        assert_eq!(cpu.regs.read_ar(2), 10); // src untouched
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_addmi_shifts_immediate_by_8() {
        // addmi a6,a3,0x300 -- `62 d3 03`. Pins the shift:
        // an unshifted-imm bug would add 3 instead of 0x300.
        let rom = vec![0x62, 0xd3, 0x03];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0x100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 0x400);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_sub() {
        // sub a6,a3,a10 -- `a0 63 c0`.
        let rom = vec![0xa0, 0x63, 0xc0];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 10);
        cpu.regs.write_ar(10, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 7);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx4() {
        // Distinct source registers pin the (AR[s]<<2)+AR[t] formula (an
        // aliased vector can't distinguish "shift s" from "shift t").
        // Hand-encoded: addx4 a5,a2,a3 (r=5,s=2,
        // t=3) -- op1=0,op2=0xA per decode/arith.rs -- byte0=(t<<4)|0=0x30,
        // byte1=(r<<4)|s=0x52, byte2=(op2<<4)|op1=0xa0.
        let rom = vec![0x30, 0x52, 0xa0];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 3); // s
        cpu.regs.write_ar(3, 100); // t
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), (3 << 2) + 100); // 112
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx2() {
        // addx2 a7,a4,a6 -- `60 74 90`.
        let rom = vec![0x60, 0x74, 0x90];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 9);
        cpu.regs.write_ar(6, 11);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), (9 << 1) + 11); // 29
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_addx8() {
        // addx8 a10,a7,a2 -- `20 a7 b0`.
        let rom = vec![0x20, 0xa7, 0xb0];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(7, 4);
        cpu.regs.write_ar(2, 7);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(10), (4 << 3) + 7); // 39
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_subx8_pins_operand_order() {
        // Hand-encoded with distinct s/t to pin the operand order: subx8 a4,a5,a6
        // (r=4,s=5,t=6) -- op1=0,op2=0xF -- byte0=(t<<4)|0=0x60,
        // byte1=(r<<4)|s=0x45, byte2=(op2<<4)|op1=0xf0. If the minuend were
        // AR[t] instead of the shifted AR[s], or the shift applied to the
        // wrong operand, this would diverge from (AR[s]<<3)-AR[t].
        let rom = vec![0x60, 0x45, 0xf0];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 20); // s (shifted minuend)
        cpu.regs.write_ar(6, 3); // t (subtrahend)
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), (20u32 << 3).wrapping_sub(3)); // 157
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_and() {
        // and a11,a5,a8 -- `80 b5 10`.
        let rom = vec![0x80, 0xb5, 0x10];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0xF0F0_F0F0);
        cpu.regs.write_ar(8, 0xFF00_FF00);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(11), 0xF000_F000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_xor() {
        // xor a12,a3,a7 -- `70 c3 30`.
        let rom = vec![0x70, 0xc3, 0x30];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0xFF00_FF00);
        cpu.regs.write_ar(7, 0x0F0F_0F0F);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(12), 0xF00F_F00F);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_neg() {
        // neg a9,a4 -- `40 90 60`.
        let rom = vec![0x40, 0x90, 0x60];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), (-5i32) as u32);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_abs_on_negative() {
        // abs a5,a13 -- `d0 51 60`.
        let rom = vec![0xd0, 0x51, 0x60];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(13, (-17i32) as u32);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 17);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_moveqz_both_conditions() {
        // moveqz a8,a5,a11 -- `b0 85 83`. AR[t]==0 -> moved;
        // AR[t]!=0 -> unchanged.
        let rom = vec![0xb0, 0x85, 0x83];

        let mut bus = FlatBus::new(rom.clone());
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0xAAAA);
        cpu.regs.write_ar(11, 0); // condition true
        cpu.regs.write_ar(8, 0x1111);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), 0xAAAA);

        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0xAAAA);
        cpu.regs.write_ar(11, 1); // condition false
        cpu.regs.write_ar(8, 0x1111);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), 0x1111); // unchanged
    }

    #[test]
    fn executes_movnez_both_conditions() {
        // movnez a3,a12,a6 -- `60 3c 93`. AR[t]!=0 ->
        // moved; AR[t]==0 -> unchanged.
        let rom = vec![0x60, 0x3c, 0x93];

        let mut bus = FlatBus::new(rom.clone());
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(12, 0xBBBB);
        cpu.regs.write_ar(6, 7); // condition true
        cpu.regs.write_ar(3, 0x2222);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 0xBBBB);

        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(12, 0xBBBB);
        cpu.regs.write_ar(6, 0); // condition false
        cpu.regs.write_ar(3, 0x2222);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 0x2222); // unchanged
    }

    #[test]
    fn executes_min_vs_minu_diverge_on_high_bit() {
        // min a12,a3,a9 -- `90 c3 43`. One operand has the
        // high bit set (a large unsigned value / a negative signed value);
        // signed min must pick it, since it's the more-negative signed
        // value, while unsigned min (tested below with minu) must NOT.
        let rom = vec![0x90, 0xc3, 0x43];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0x8000_0000); // i32::MIN, the smaller signed value
        cpu.regs.write_ar(9, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(12), 0x8000_0000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_minu_treats_high_bit_as_large() {
        // minu a4,a11,a7 -- `70 4b 63`. Same high-bit-set
        // value as the min test above, but unsigned: it's the LARGER value,
        // so minu must pick the small operand instead.
        let rom = vec![0x70, 0x4b, 0x63];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(11, 5);
        cpu.regs.write_ar(7, 0x8000_0000); // unsigned: huge
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 5);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_maxu() {
        // maxu a13,a6,a10 -- `a0 d6 73`.
        let rom = vec![0xa0, 0xd6, 0x73];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 0x8000_0000); // unsigned: huge
        cpu.regs.write_ar(10, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(13), 0x8000_0000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_mov_as_or_with_s_eq_t() {
        // mov a9,a3 (== or a9,a3,a3) -- `30 93 20`. The disassembler's
        // `mov` pseudo-op is exactly this Or arm with s==t; no separate
        // Mov variant/exec arm exists or is needed.
        let rom = vec![0x30, 0x93, 0x20];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0xCAFE);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0xCAFE);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_slli_shifts_left() {
        // slli a7,a6,27 -- `50 76 01` (synthetic, objdump-confirmed).
        let rom = vec![0x50, 0x76, 0x01];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 5u32 << 27);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_srli_shifts_right_logical() {
        // srli a7,a5,6 -- `50 76 41` (synthetic). A
        // high-bit-set value must NOT sign-fill (logical, not arithmetic).
        let rom = vec![0x50, 0x76, 0x41];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(5, 0x8000_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 0x8000_0000u32 >> 6);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_srai_sign_fills_negative() {
        // srai a3,a10,0x18 (24) -- `a0 38 31` (srai encoding vector).
        // Arithmetic shift of a negative value must sign-fill the vacated
        // top bits with 1s, unlike Srli's logical fill.
        let rom = vec![0xa0, 0x38, 0x31];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(10, 0x8000_0000); // i32::MIN
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), ((0x8000_0000u32 as i32) >> 24) as u32);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_ssl_then_sll_matches_direct_left_shift() {
        // REQUIRED invariant: ssl(n); sll == AR[s] << n. ssl a4 (`00 14 40`)
        // sets SAR from AR[4] (holding the shift count n=5); sll a3,a2 (`00
        // 32 a1`) then shifts AR[2] (a DIFFERENT register, holding the value
        // to shift) left by `32 - SAR`. a4 and a2 are deliberately distinct
        // from a3 (the destination) and from each other, so this can't pass
        // by accident (e.g. shifting the count register itself).
        let rom = vec![0x00, 0x14, 0x40, /* ssl a4 */ 0x00, 0x32, 0xa1 /* sll a3,a2 */];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 5); // shift count n
        cpu.regs.write_ar(2, 0x0000_0007); // value to shift
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // ssl
        assert_eq!(cpu.regs.sar, 32 - 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // sll
        assert_eq!(cpu.regs.read_ar(3), 0x0000_0007u32 << 5);
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn executes_ssr_then_srl_matches_direct_right_shift() {
        // REQUIRED invariant: ssr(n); srl == AR[t] >> n. ssr a4 (`00 04 40`)
        // sets SAR = AR[4] & 31 (n=5); srl a3,a2 (`20 30 91`) then shifts
        // AR[2] (a different register) right by SAR. A high-bit-set value
        // also confirms the shift is logical (matches Srl's own doc), not
        // arithmetic.
        let rom = vec![0x00, 0x04, 0x40, /* ssr a4 */ 0x20, 0x30, 0x91 /* srl a3,a2 */];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(4, 5); // shift count n
        cpu.regs.write_ar(2, 0xFFFF_FFF0); // value to shift
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // ssr
        assert_eq!(cpu.regs.sar, 5);
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // srl
        assert_eq!(cpu.regs.read_ar(3), 0xFFFF_FFF0u32 >> 5);
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn executes_src_funnels_across_the_32_bit_boundary() {
        // ssai 4 (`00 44 40`, sets SAR=4) then src a7,a6,a5 (`50 76 81`,
        // synthetic) -- funnel shift right by SAR,
        // concatenating AR[6] (high) : AR[5] (low) into a 64-bit value.
        // AR[6]'s LSB and AR[5]'s MSB are both set, so the expected result
        // depends on bits crossing from the high half into the low half --
        // a bug that shifted either operand in isolation (instead of the
        // true 64-bit concatenation) would diverge from this.
        let rom = vec![0x00, 0x44, 0x40, /* ssai 4 */ 0x50, 0x76, 0x81 /* src a7,a6,a5 */];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 0x0000_0001); // high
        cpu.regs.write_ar(5, 0x8000_0000); // low
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // ssai
        assert_eq!(cpu.regs.sar, 4);
        assert!(matches!(cpu.step(&mut bus), Step::Ran)); // src
                                                          // (0x1_8000_0000u64 >> 4) & 0xFFFF_FFFF == 0x1800_0000
        assert_eq!(cpu.regs.read_ar(7), 0x1800_0000);
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn executes_sext_from_bit_7_sign_extends() {
        // sext a9,a8,7 -- `00 98 23` (sext encoding vector). Bit 7 clear:
        // value passes through unchanged (positive). Bit 7 set: bits 8-31
        // sign-fill to 1 (negative).
        let rom = vec![0x00, 0x98, 0x23];

        let mut bus = FlatBus::new(rom.clone());
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(8, 0x0000_007F); // bit 7 clear
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0x0000_007F);

        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(8, 0x0000_00FF); // bit 7 set
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0xFFFF_FFFF);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_nsau_edge_cases() {
        // nsau a2,a2 -- `20 f2 40` (nsau encoding vector, aliased
        // dest/src -- the mapping itself is pinned separately at decode
        // level). REQUIRED edge cases: an all-zero input is 32 (not the
        // 32-bit-count overflow one might naively expect), and 0x1 has
        // exactly 31 leading zeros.
        let rom = vec![0x20, 0xf2, 0x40];

        let mut bus = FlatBus::new(rom.clone());
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 32);

        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0x0000_0001);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 31);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_mull_truncates_low_32_of_overflowing_product() {
        // mull a9,a6,a4 -- hand-encoded (r=9,s=6,t=4,op1=2,op2=8):
        // byte0=(t<<4)=0x40, byte1=(r<<4)|s=0x96, byte2=(op2<<4)|op1=0x82.
        // u32::MAX * 2 overflows 32 bits; the result must be the wrapped low
        // 32 bits (0xFFFF_FFFE), not a saturated or panicking value.
        let rom = vec![0x40, 0x96, 0x82];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 0xFFFF_FFFF);
        cpu.regs.write_ar(4, 2);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0xFFFF_FFFE);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_mul16s_two_negatives_give_positive_result() {
        // mul16s a5,a6,a7 -- hand-encoded (r=5,s=6,t=7,op1=1,op2=0xD):
        // byte0=(t<<4)=0x70, byte1=(r<<4)|s=0x56, byte2=(op2<<4)|op1=0xd1.
        // Both operands are -200 in their low 16 bits (upper bits carry
        // deliberate garbage to prove only the low halfword is used); the
        // signed product is +40000, which needs more than 16 bits so a
        // truncate-to-i16 bug would also be caught.
        let rom = vec![0x70, 0x56, 0xd1];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 0xABCD_FF38); // low16 = 0xFF38 = -200i16
        cpu.regs.write_ar(7, 0x1234_FF38); // low16 = 0xFF38 = -200i16
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 40000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_mul16u_high_bit_set_is_unsigned_not_sign_extended() {
        // mul16u a8,a2,a9 -- hand-encoded (r=8,s=2,t=9,op1=1,op2=0xC):
        // byte0=(t<<4)=0x90, byte1=(r<<4)|s=0x82, byte2=(op2<<4)|op1=0xc1.
        // AR[2]'s low 16 bits are 0x8000 (high bit set): as UNSIGNED that's
        // 32768; if the implementation mistakenly sign-extended instead of
        // masking, it would compute -32768*1 = 0xFFFF8000, not 0x8000 --
        // the two diverge, disambiguating the bug.
        let rom = vec![0x90, 0x82, 0xc1];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0xDEAD_8000); // low16 = 0x8000 (32768 unsigned)
        cpu.regs.write_ar(9, 0x0000_0001);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), 0x8000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_quou_large_unsigned_values() {
        // quou a4,a9,a11 -- `b0 49 c2`. Dividend has the
        // high bit set (would be negative as i32); unsigned division must
        // treat it as the large positive value, not -16.
        let rom = vec![0xb0, 0x49, 0xc2];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(9, 0xFFFF_FFF0); // 4294967280 unsigned
        cpu.regs.write_ar(11, 0x10); // 16
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0x0FFF_FFFF); // 268435455
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_remu_large_unsigned_values() {
        // remu a7,a3,a12 -- `c0 73 e2`. Dividend is
        // u32::MAX; as signed that's -1 (-1 % 10 == -1 in Rust), but
        // unsigned remainder must be 5 -- the two diverge, disambiguating
        // signed vs. unsigned.
        let rom = vec![0xc0, 0x73, 0xe2];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(3, 0xFFFF_FFFF);
        cpu.regs.write_ar(12, 10);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 5);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_rems_negative_dividend_keeps_negative_remainder() {
        // rems a10,a6,a13 -- `d0 a6 f2`. Dividend -7,
        // divisor 3: signed remainder is -7 - 3*(-2) = -1 (sign follows the
        // dividend). Unsigned remu on the same bit pattern would compute a
        // completely different (large, positive) value, disambiguating
        // signed vs. unsigned interpretation.
        let rom = vec![0xd0, 0xa6, 0xf2];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, (-7i32) as u32);
        cpu.regs.write_ar(13, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(10), (-1i32) as u32);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_rems_i32_min_by_minus_one_does_not_panic() {
        // rems a10,a6,a13 -- `d0 a6 f2`, with dividend
        // i32::MIN and divisor -1. This is DISTINCT from the zero-divisor
        // guard: Rust's plain `%` on i32 panics for this specific input in
        // every build profile (the host `idiv` instruction can't compute
        // it), even though the architectural remainder is 0. Confirmed
        // empirically before fixing: `sv % tv` panics here; `wrapping_rem`
        // does not. This test is the regression guard for that fix.
        let rom = vec![0xd0, 0xa6, 0xf2];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(6, 0x8000_0000); // i32::MIN
        cpu.regs.write_ar(13, 0xFFFF_FFFF); // -1
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(10), 0);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_quou_by_zero_divisor_raises_general_exception() {
        // quou a4,a9,a11 -- `b0 49 c2`, divisor forced to 0. Real hardware
        // raises INTEGER_DIVIDE_BY_ZERO_CAUSE (through the general-
        // exception path) -- assert the RAISED machine state, not a
        // returned value: AR[4] (the would-be destination) must be left
        // UNTOUCHED, since the divide never executes.
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0xb0, 0x49, 0xc2]);
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.vecbase = 0x2000;
        cpu.regs.write_ar(9, 123);
        cpu.regs.write_ar(11, 0);
        cpu.regs.write_ar(4, 0xdead_beef); // poison the destination
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
                assert_eq!(pc, 0x2000 + ExceptionVectors::default().kernel);
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
        assert_eq!(cpu.pc, 0x2000 + ExceptionVectors::default().kernel);
        assert_eq!(cpu.epc1, 0x100, "EPC1 = the faulting quou's own pc");
        assert_eq!(cpu.regs.exccause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
        assert!(cpu.regs.excm());
        assert_eq!(cpu.regs.read_ar(4), 0xdead_beef, "destination untouched -- the divide never ran");
    }

    #[test]
    fn executes_remu_by_zero_divisor_raises_general_exception() {
        // remu a7,a3,a12 -- `c0 73 e2`, divisor forced to 0. See
        // executes_quou_by_zero_divisor_raises_general_exception.
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0xc0, 0x73, 0xe2]);
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.vecbase = 0x2000;
        cpu.regs.write_ar(3, 123);
        cpu.regs.write_ar(12, 0);
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
                assert_eq!(pc, 0x2000 + ExceptionVectors::default().kernel);
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
        assert_eq!(cpu.regs.exccause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
        assert_eq!(cpu.epc1, 0x100);
        assert!(cpu.regs.excm());
    }

    #[test]
    fn executes_rems_by_zero_divisor_raises_general_exception() {
        // rems a10,a6,a13 -- `d0 a6 f2`, divisor forced to 0. See
        // executes_quou_by_zero_divisor_raises_general_exception.
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0xd0, 0xa6, 0xf2]);
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.vecbase = 0x2000;
        cpu.regs.write_ar(6, (-123i32) as u32);
        cpu.regs.write_ar(13, 0);
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
                assert_eq!(pc, 0x2000 + ExceptionVectors::default().kernel);
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
        assert_eq!(cpu.regs.exccause, EXCCAUSE_INTEGER_DIVIDE_BY_ZERO);
        assert_eq!(cpu.epc1, 0x100);
        assert!(cpu.regs.excm());
    }
}
