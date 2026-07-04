//! Register/immediate arithmetic execute: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`MoviN`/`Movi`/`MovN`/
/// `Or`/`Extui`); `None` otherwise, so `step()` tries the next category.
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
}
