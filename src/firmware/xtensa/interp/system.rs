//! System/MMU-config execute: `wsr.<sr>`, `isync`, `dsync`, `witlb`,
//! `wdtlb`, `iitlb`, `idtlb`.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`Isync`/`Dsync`/`Witlb`/
/// `Wdtlb`/`Iitlb`/`Idtlb`/`Wsr`); `None` otherwise, so `step()` tries the
/// next category.
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::Isync => {
            log::debug!("firmware interp: isync at 0x{:08x} (no modeled pipeline effect)", pc);
        }
        Op::Witlb { t, s } => {
            log::debug!(
                "firmware interp: witlb a{},a{} at 0x{:08x} (no-op; mmu.rs models TLB state later)",
                t,
                s,
                pc
            );
        }
        Op::Wdtlb { t, s } => {
            log::debug!(
                "firmware interp: wdtlb a{},a{} at 0x{:08x} (no-op; mmu.rs models TLB state later)",
                t,
                s,
                pc
            );
        }
        Op::Iitlb { s } => {
            log::debug!(
                "firmware interp: iitlb a{} (=0x{:08x}) at 0x{:08x} (no-op; mmu.rs later)",
                s,
                cpu.regs.read_ar(*s),
                pc
            );
        }
        Op::Idtlb { s } => {
            log::debug!(
                "firmware interp: idtlb a{} (=0x{:08x}) at 0x{:08x} (no-op; mmu.rs later)",
                s,
                cpu.regs.read_ar(*s),
                pc
            );
        }
        Op::Wsr { sr, t } => {
            let value = cpu.regs.read_ar(*t);
            cpu.write_sr(*sr, value);
        }
        Op::Dsync => {
            log::debug!("firmware interp: dsync at 0x{:08x} (no modeled pipeline effect)", pc);
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

    #[test]
    fn wsr_routes_modeled_special_registers() {
        // wsr.vecbase a2 (`20 e7 13`), wsr.ps a3 (`30 e6 13`): the modeled SRs
        // must land in cpu.vecbase / regs.ps. Oracle: SR numbers per the
        // Xtensa encoding (VECBASE=0xE7, PS=0xE6); decode verified in decode/system.rs.
        let rom = vec![0x20, 0xe7, 0x13, /* wsr.vecbase a2 */ 0x30, 0xe6, 0x13 /* wsr.ps a3 */];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0x0001_8000);
        cpu.regs.write_ar(3, 0x0004_0010);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, 0x0001_8000);
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.ps, 0x0004_0010);
    }

    #[test]
    fn wsr_to_unmodeled_mmu_config_is_a_no_op_that_advances() {
        // wsr.itlbcfg a2 (`20 5b 13`, boot vector): an unmodeled MMU-config
        // SR must not touch the interpreter's registers, but must still
        // advance pc like any executed instruction.
        let rom = vec![0x20, 0x5b, 0x13];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert_eq!(cpu.regs.ps, 0, "unmodeled SR write left PS untouched");
    }

    #[test]
    fn wdtlb_iitlb_idtlb_dsync_are_logged_no_ops() {
        // The remaining boot MMU-setup ops all advance pc with no modeled
        // register/memory effect: wdtlb a7,a4 (`70 e4 50`), iitlb a5
        // (`00 45 50`), idtlb a5 (`00 c5 50`), dsync (`30 20 00`).
        let rom = vec![0x70, 0xe4, 0x50, 0x00, 0x45, 0x50, 0x00, 0xc5, 0x50, 0x30, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        for expected_pc in [3u32, 6, 9, 12] {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
            assert_eq!(cpu.pc, expected_pc);
        }
    }

    #[test]
    fn witlb_and_isync_are_logged_no_ops() {
        // witlb a7,a4 (`70 64 50`) then isync (`00 20 00`) -- M1.1 vectors,
        // concatenated. Neither has a modeled register/memory effect; both
        // must still advance pc like any executed instruction.
        let rom = vec![0x70, 0x64, 0x50, 0x00, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 6);
    }
}
