//! System/MMU-config execute: `wsr.<sr>`, `rsr.<sr>`, `wur`, `isync`,
//! `dsync`, `rsync`, `memw`, `nop`/`nop.n`, `rsil`, `syscall`, `witlb`,
//! `wdtlb`, `iitlb`, `idtlb`, and the icache/dcache-maintenance group
//! (`dhwbi`/`dhi`/`dii`/`ihi`).

use super::{Cpu, Step, EXCCAUSE_SYSCALL};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`Isync`/`Dsync`/`Witlb`/
/// `Wdtlb`/`Iitlb`/`Idtlb`/`Wsr` plus the M2a Task 9 system-opcode group:
/// `Rsr`/`Wur`/`Rsil`/`Syscall`/`Memw`/`Nop`/`Rsync`/`NopN`/`Dhwbi`/`Dhi`/
/// `Dii`/`Ihi`); `None` otherwise, so `step()` tries the next category.
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
        // rsr: AR[t] = SR[sr], read via the shared SR router (see
        // Cpu::read_sr's doc: modeled SRs read their real value, unmodeled
        // ones log and return 0).
        Op::Rsr { sr, t } => {
            let v = cpu.read_sr(*sr);
            cpu.regs.write_ar(*t, v);
        }
        // wur: write AR[t] into the named user register, via the shared UR
        // router (see Cpu::write_ur's doc for the modeled/unmodeled split).
        Op::Wur { ur, t } => {
            let v = cpu.regs.read_ar(*t);
            cpu.write_ur(*ur, v);
        }
        // rsil: AR[t] = the FULL old PS (not just the level), then
        // PS.INTLEVEL = imm. Order matters -- AR[t] must capture PS BEFORE
        // the level is overwritten.
        Op::Rsil { t, imm } => {
            let old_ps = cpu.regs.ps;
            cpu.regs.write_ar(*t, old_ps);
            cpu.regs.set_intlevel(*imm);
        }
        // syscall: raises a general exception (EXCCAUSE_SYSCALL) -- an
        // early return, since raise_general_exception sets cpu.pc to the
        // vector itself, bypassing this function's common pc+len tail
        // (mirrors control::exec's Entry/Retw early-return-on-exception
        // pattern).
        Op::Syscall => {
            return Some(cpu.raise_general_exception(pc, EXCCAUSE_SYSCALL));
        }
        // memw/nop/rsync/nop.n: logged no-ops, same treatment as isync/dsync.
        Op::Memw => {
            log::debug!("firmware interp: memw at 0x{:08x} (no modeled memory reordering)", pc);
        }
        Op::Nop => {
            log::debug!("firmware interp: nop at 0x{:08x}", pc);
        }
        Op::Rsync => {
            log::debug!("firmware interp: rsync at 0x{:08x} (no modeled pipeline effect)", pc);
        }
        Op::NopN => {
            log::debug!("firmware interp: nop.n at 0x{:08x}", pc);
        }
        // dhwbi/dhi/dii/ihi: cache-maintenance ops, logged no-ops -- this
        // interpreter has no cache model to invalidate/writeback.
        Op::Dhwbi { s, imm } => {
            log::debug!(
                "firmware interp: dhwbi a{},{} (addr 0x{:08x}) at 0x{:08x} (no modeled cache)",
                s,
                imm,
                cpu.regs.read_ar(*s).wrapping_add(*imm),
                pc
            );
        }
        Op::Dhi { s, imm } => {
            log::debug!(
                "firmware interp: dhi a{},{} (addr 0x{:08x}) at 0x{:08x} (no modeled cache)",
                s,
                imm,
                cpu.regs.read_ar(*s).wrapping_add(*imm),
                pc
            );
        }
        Op::Dii { s, imm } => {
            log::debug!(
                "firmware interp: dii a{},{} (addr 0x{:08x}) at 0x{:08x} (no modeled cache)",
                s,
                imm,
                cpu.regs.read_ar(*s).wrapping_add(*imm),
                pc
            );
        }
        Op::Ihi { s, imm } => {
            log::debug!(
                "firmware interp: ihi a{},{} (addr 0x{:08x}) at 0x{:08x} (no modeled cache)",
                s,
                imm,
                cpu.regs.read_ar(*s).wrapping_add(*imm),
                pc
            );
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

    // -- M2a Task 9: system opcodes --------------------------------------

    #[test]
    fn rsr_reads_a_modeled_sr() {
        // Hand-built `rsr a2, EXCCAUSE` (t=2, sr=0xE8, op1=3/op2=0 -- same
        // byte1-is-whole-sr-number convention as Wsr): byte0=(t<<4)=0x20,
        // byte1=sr=0xe8, byte2=(op2<<4)|op1=0x03. EXCCAUSE is set directly
        // (as `raise_general_exception` would leave it) and must read back
        // exactly through `rsr`.
        let rom = vec![0x20, 0xe8, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.exccause = 6; // EXCCAUSE_INTEGER_DIVIDE_BY_ZERO
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 6);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn rsr_on_unmodeled_sr_returns_zero_without_panicking() {
        // Hand-built `rsr a2, 0x5b` (ITLBCFG -- unmodeled, same SR number as
        // the existing wsr-to-unmodeled-SR test): must not panic, and must
        // overwrite AR[2] with 0 (not leave the poisoned value behind).
        let rom = vec![0x20, 0x5b, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0, "unmodeled SR read returns 0, not the poisoned value");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn wur_vecbase_routes_to_cpu_vecbase() {
        // wur a3,VECBASE (`30 e7 f3`, firmware vector): AR[3] must land in
        // cpu.vecbase, the SAME state `wsr.vecbase` writes -- see
        // `Op::Wur`'s doc for why this WUR-based write is modeled that way
        // despite the naming discrepancy with AMD's vendored generic
        // xtensa-modules.c.
        let rom = vec![0x30, 0xe7, 0xf3];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(3, 0x0003_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, 0x0003_0000);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn wur_to_unmodeled_ur_is_a_no_op_that_advances() {
        // Hand-built `wur a2, 0x03` (an unmodeled UR number, distinct from
        // VECBASE=0xE7): byte0=(t<<4)=0x20, byte1=ur=0x03, byte2=0xf3
        // (op1=3,op2=0xF). Must not touch vecbase, but must still advance pc.
        let rom = vec![0x20, 0x03, 0xf3];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0xCAFE_BABE);
        let vecbase0 = cpu.vecbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, vecbase0, "unmodeled UR write left vecbase untouched");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn rsil_captures_full_old_ps_and_sets_intlevel() {
        // rsil a2,0x2 (`20 62 00`, firmware vector). PS is seeded with bits
        // OUTSIDE INTLEVEL set (0xDEAD_BEE5, intlevel nibble = 5) to prove
        // AR[2] captures the FULL old PS -- not just the 4-bit level (a bug
        // that returned `old_intlevel` instead of `old_ps` would produce 5
        // instead of 0xDEAD_BEE5, easily distinguished).
        let rom = vec![0x20, 0x62, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.ps = 0xDEAD_BEE5;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0xDEAD_BEE5, "AR[2] = the FULL old PS");
        assert_eq!(cpu.regs.intlevel(), 2, "PS.INTLEVEL set from imm4");
        assert_eq!(cpu.regs.ps, 0xDEAD_BEE2, "rest of PS preserved, only INTLEVEL changed");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn syscall_raises_general_exception_with_exccause_1() {
        // syscall (`00 50 00`, firmware vector) at a nonzero pc, so EPC1's
        // value is distinguishable from its zero default.
        use super::super::EXCCAUSE_SYSCALL;
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0x00, 0x50, 0x00]);
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0x100);
        cpu.vecbase = 0x2000;
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_SYSCALL);
                assert_eq!(pc, 0x2000 + 0x300, "vectors to VECBASE + kernel-exception offset");
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
        assert_eq!(cpu.pc, 0x2000 + 0x300);
        assert_eq!(cpu.epc1, 0x100, "EPC1 = the faulting syscall's own pc");
        assert_eq!(cpu.regs.exccause, EXCCAUSE_SYSCALL);
        assert!(cpu.regs.excm(), "PS.EXCM set entering the handler");
    }

    #[test]
    fn memw_nop_rsync_nopn_and_cache_ops_are_logged_no_ops() {
        // memw(`c0 20 00`), nop(`f0 20 00`), rsync(`10 20 00`), nop.n(`3d
        // f0`, len 2), dhwbi(`52 72 00`), dhi(`62 72 00`), dii(`72 72 00`),
        // ihi(`e2 72 00`) -- all advance pc by their length and touch no
        // modeled register/memory state (the cache ops use a2 as the
        // address register but must not write it).
        let rom = vec![
            0xc0, 0x20, 0x00, // memw
            0xf0, 0x20, 0x00, // nop
            0x10, 0x20, 0x00, // rsync
            0x3d, 0xf0, // nop.n
            0x52, 0x72, 0x00, // dhwbi a2,0
            0x62, 0x72, 0x00, // dhi a2,0
            0x72, 0x72, 0x00, // dii a2,0
            0xe2, 0x72, 0x00, // ihi a2,0
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.regs.write_ar(2, 0x0008_0000); // address register the cache ops read
        for expected_pc in [3u32, 6, 9, 11, 14, 17, 20, 23] {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
            assert_eq!(cpu.pc, expected_pc);
        }
        assert_eq!(cpu.regs.read_ar(2), 0x0008_0000, "cache ops must not write their address register");
        assert_eq!(cpu.regs.ps, 0, "none of these ops touch PS");
    }

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
