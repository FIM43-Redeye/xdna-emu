//! System/MMU-config execute: `wsr.<sr>`, `rsr.<sr>`, `wur`, `isync`,
//! `dsync`, `rsync`, `memw`, `nop`/`nop.n`, `rsil`, `syscall`, `witlb`,
//! `wdtlb`, `iitlb`, `idtlb`, and the icache/dcache-maintenance group
//! (`dhwbi`/`dhi`/`dii`/`ihi` plus the M2c Task 6a completion: `dpfr`/
//! `dpfw`/`dpfro`/`dpfwo`/`dhwb`/`dpfl`/`dhu`/`diu`/`diwb`/`diwbi`/`ipf`/
//! `ipfl`/`ihu`/`iiu`/`iii` -- 19 cache-maintenance ops total, all logged
//! no-ops since this interpreter models no cache).

use super::{Cpu, Step, EXCCAUSE_SYSCALL};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (`Isync`/`Dsync`/`Witlb`/
/// `Wdtlb`/`Iitlb`/`Idtlb`/`Wsr` plus the M2a Task 9 system-opcode group:
/// `Rsr`/`Wur`/`Rsil`/`Syscall`/`Memw`/`Nop`/`Rsync`/`NopN`/`Dhwbi`/`Dhi`/
/// `Dii`/`Ihi`, and the M2c Task 6a completion of the `r==7` cache group:
/// `Dpfr`/`Dpfw`/`Dpfro`/`Dpfwo`/`Dhwb`/`Dpfl`/`Dhu`/`Diu`/`Diwb`/`Diwbi`/
/// `Ipf`/`Ipfl`/`Ihu`/`Iiu`/`Iii`); `None` otherwise, so `step()` tries the
/// next category.
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::Isync => {
            log::debug!("firmware interp: isync at 0x{:08x} (no modeled pipeline effect)", pc);
        }
        // witlb/wdtlb: AS = AR[s] (way index + VPN), AT = AR[t] (paddr|attr,
        // ring in bits[5:4]) -- installs a real TLB entry via `Mmu::write_tlb`
        // (M2b Task 4; see that method's doc for the QEMU derivation).
        Op::Witlb { t, s } => {
            let as_ = cpu.regs.read_ar(*s);
            let at = cpu.regs.read_ar(*t);
            cpu.mmu.write_tlb(false, at, as_);
        }
        Op::Wdtlb { t, s } => {
            let as_ = cpu.regs.read_ar(*s);
            let at = cpu.regs.read_ar(*t);
            cpu.mmu.write_tlb(true, at, as_);
        }
        // iitlb/idtlb: AS = AR[s] (way index + VPN) -- invalidates the
        // targeted entry via `Mmu::invalidate_tlb`.
        Op::Iitlb { s } => {
            let as_ = cpu.regs.read_ar(*s);
            cpu.mmu.invalidate_tlb(false, as_);
        }
        Op::Idtlb { s } => {
            let as_ = cpu.regs.read_ar(*s);
            cpu.mmu.invalidate_tlb(true, as_);
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
        // rur: read the named user register into AR[t] (the destination `arr`,
        // resolved in decode) via the shared UR router (see Cpu::read_ur).
        Op::Rur { ur, t } => {
            let v = cpu.read_ur(*ur);
            cpu.regs.write_ar(*t, v);
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
        // The REST of the r==7 cache-maintenance group (M2c Task 6a):
        // dpfr/dpfw/dpfro/dpfwo/dhwb/dpfl/dhu/diu/diwb/diwbi/ipf/ipfl/ihu/
        // iiu/iii -- same treatment as dhwbi/dhi/dii/ihi above, one shared
        // logged no-op arm since this interpreter has no cache model for
        // any op in the 19-op group. `op`'s Debug impl names the specific
        // mnemonic+operands (e.g. `Iiu { s: 3, imm: 0 }`).
        op @ (Op::Dpfr { s, imm }
        | Op::Dpfw { s, imm }
        | Op::Dpfro { s, imm }
        | Op::Dpfwo { s, imm }
        | Op::Dhwb { s, imm }
        | Op::Dpfl { s, imm }
        | Op::Dhu { s, imm }
        | Op::Diu { s, imm }
        | Op::Diwb { s, imm }
        | Op::Diwbi { s, imm }
        | Op::Ipf { s, imm }
        | Op::Ipfl { s, imm }
        | Op::Ihu { s, imm }
        | Op::Iiu { s, imm }
        | Op::Iii { s, imm }) => {
            log::debug!(
                "firmware interp: {:?} (addr 0x{:08x}) at 0x{:08x} (no modeled cache)",
                op,
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
    use super::super::{mapped_cpu, Cpu, Step};
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
        let mut cpu = mapped_cpu(0);
        cpu.regs.exccause = 6; // EXCCAUSE_INTEGER_DIVIDE_BY_ZERO
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 6);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn depc_round_trips_through_wsr_and_rsr() {
        // wsr.depc a2; rsr a3,depc. The double-exception handler uses this
        // exact SR (0xc0) before returning through rfde.
        let rom = vec![0x20, 0xc0, 0x13, 0x30, 0xc0, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0x1234_5678);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.depc, 0x1234_5678);
        assert_eq!(cpu.regs.read_ar(3), 0x1234_5678);
    }

    #[test]
    fn rsr_on_unmodeled_sr_returns_zero_without_panicking() {
        // Hand-built `rsr a2, 0x00` (an SR this interpreter still doesn't
        // model -- 0x5b/ITLBCFG no longer qualifies as of M2b Task 4, which
        // routed the MMU-config SRs into `cpu.mmu`): must not panic, and must
        // overwrite AR[2] with 0 (not leave the poisoned value behind).
        let rom = vec![0x20, 0x00, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0, "unmodeled SR read returns 0, not the poisoned value");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn wur_threadptr_routes_to_cpu_threadptr_not_vecbase() {
        // wur a3,THREADPTR (`30 e7 f3`, firmware vector): AR[3] must land in
        // cpu.threadptr -- the Xtensa THREADPTR TIE register -- and MUST NOT
        // touch cpu.vecbase. UR 0xE7 (threadptr) and SR 0xE7 (vecbase) are
        // distinct register spaces; conflating them corrupted VECBASE and sent
        // the firmware's syscall into an unmapped page. See `Op::Wur`'s doc.
        let rom = vec![0x30, 0xe7, 0xf3];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.vecbase = 0x0000_0800; // the real prologue-set VECBASE
        cpu.regs.write_ar(3, 0x0003_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.threadptr, 0x0003_0000, "wur 0xE7 writes threadptr");
        assert_eq!(cpu.vecbase, 0x0000_0800, "wur 0xE7 must NOT clobber vecbase");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn wur_to_unmodeled_ur_is_a_no_op_that_advances() {
        // Hand-built `wur a2, 0x03` (an unmodeled UR number, distinct from
        // VECBASE=0xE7): byte0=(t<<4)=0x20, byte1=ur=0x03, byte2=0xf3
        // (op1=3,op2=0xF). Must not touch vecbase, but must still advance pc.
        let rom = vec![0x20, 0x03, 0xf3];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0xCAFE_BABE);
        let vecbase0 = cpu.vecbase;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, vecbase0, "unmodeled UR write left vecbase untouched");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn rur_threadptr_reads_cpu_threadptr() {
        // rur.threadptr a5 (`70 5e e3`: r=5, s=0xe, t=0x7 -> ur=(0xe<<4)|0x7=0xE7,
        // target arr=5): must read cpu.threadptr into AR[5] -- the READ sibling
        // of wur.threadptr.
        let rom = vec![0x70, 0x5e, 0xe3];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.threadptr = 0x1234_5678;
        cpu.regs.write_ar(5, 0xdead_beef); // poison the destination
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x1234_5678, "rur 0xE7 reads threadptr");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn rur_unmodeled_ur_reads_zero_and_advances() {
        // rur.fcr a2 (`80 2e e3`, the firmware's 0x2a09 wall): the FPU control
        // register is not modeled -> reads 0 (no FPU state), and pc advances.
        let rom = vec![0x80, 0x2e, 0xe3];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0xdead_beef); // poison the destination
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0, "rur of an unmodeled UR (fcr) reads 0");
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
        let mut cpu = mapped_cpu(0);
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
        use super::super::{EXCCAUSE_SYSCALL, KERNEL_EXCEPTION_VECTOR_OFFSET};
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0x00, 0x50, 0x00]);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.vecbase = 0x2000;
        let want = 0x2000 + KERNEL_EXCEPTION_VECTOR_OFFSET;
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_SYSCALL);
                // A non-double general exception vectors to the VECBASE-relative
                // general exception vector (iter23), whose real firmware stub
                // (wsr.excsave1; l32r a3,=0x28b4; jx a3) reaches the handler.
                assert_eq!(pc, want, "vectors to VECBASE + general-exception offset");
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
        assert_eq!(cpu.pc, want);
        assert_eq!(cpu.epc1, 0x100, "EPC1 = the faulting syscall's own pc");
        assert_eq!(cpu.regs.exccause, EXCCAUSE_SYSCALL);
        assert!(cpu.regs.excm(), "PS.EXCM set entering the handler");
    }

    #[test]
    fn general_exception_while_excm_vectors_to_double() {
        // A general exception raised while PS.EXCM is already set is a DOUBLE
        // fault: it vectors to DoubleExceptionVector, not the ordinary kernel
        // or user exception vector.
        use super::super::{DOUBLE_EXCEPTION_VECTOR_OFFSET, EXCCAUSE_SYSCALL};
        let mut rom = vec![0u8; 0x103];
        rom[0x100..0x103].copy_from_slice(&[0x00, 0x50, 0x00]); // syscall
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x100);
        cpu.vecbase = 0x2000;
        cpu.regs.set_excm(); // already in exception mode -> next fault is a double
        match cpu.step(&mut bus) {
            Step::Exception { cause, pc } => {
                assert_eq!(cause, EXCCAUSE_SYSCALL);
                assert_eq!(
                    pc,
                    0x2000 + DOUBLE_EXCEPTION_VECTOR_OFFSET,
                    "double fault vectors to VECBASE + DoubleException offset"
                );
            }
            other => panic!("expected Step::Exception, got {:?}", other),
        }
    }

    #[test]
    fn memw_nop_rsync_nopn_and_cache_ops_are_logged_no_ops() {
        // memw(`c0 20 00`), nop(`f0 20 00`), rsync(`10 20 00`), nop.n(`3d
        // f0`, len 2), dhwbi(`52 72 00`), dhi(`62 72 00`), dii(`72 72 00`),
        // ihi(`e2 72 00`), then the M2c Task 6a completion of the r==7
        // group -- dpfr/dpfw/dpfro/dpfwo/dhwb (t=0..4), dpfl/dhu/diu/diwb/
        // diwbi (t=8 sub 0/2/3/4/5), ipf (t=0xC), ipfl/ihu/iiu (t=0xD sub
        // 0/2/3), iii (t=0xF), all `as`=a2 (`s=2` -> byte1=0x72) -- all
        // advance pc by their length and touch no modeled register/memory
        // state (the cache ops read a2 as the address register but must not
        // write it).
        let rom = vec![
            0xc0, 0x20, 0x00, // memw
            0xf0, 0x20, 0x00, // nop
            0x10, 0x20, 0x00, // rsync
            0x3d, 0xf0, // nop.n
            0x52, 0x72, 0x00, // dhwbi a2,0
            0x62, 0x72, 0x00, // dhi a2,0
            0x72, 0x72, 0x00, // dii a2,0
            0xe2, 0x72, 0x00, // ihi a2,0
            0x02, 0x72, 0x00, // dpfr a2,0
            0x12, 0x72, 0x00, // dpfw a2,0
            0x22, 0x72, 0x00, // dpfro a2,0
            0x32, 0x72, 0x00, // dpfwo a2,0
            0x42, 0x72, 0x00, // dhwb a2,0
            0x82, 0x72, 0x00, // dpfl a2,0 (t=8 sub=0)
            0x82, 0x72, 0x02, // dhu a2,0 (t=8 sub=2)
            0x82, 0x72, 0x03, // diu a2,0 (t=8 sub=3)
            0x82, 0x72, 0x04, // diwb a2,0 (t=8 sub=4)
            0x82, 0x72, 0x05, // diwbi a2,0 (t=8 sub=5)
            0xc2, 0x72, 0x00, // ipf a2,0
            0xd2, 0x72, 0x00, // ipfl a2,0 (t=0xD sub=0)
            0xd2, 0x72, 0x02, // ihu a2,0 (t=0xD sub=2)
            0xd2, 0x72, 0x03, // iiu a2,0 (t=0xD sub=3)
            0xf2, 0x72, 0x00, // iii a2,0
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0x0008_0000); // address register the cache ops read
        let expected_pcs: Vec<u32> = [3u32, 6, 9, 11, 14, 17, 20, 23]
            .into_iter()
            .chain((0..15).map(|i| 23 + 3 * (i + 1)))
            .collect();
        for expected_pc in expected_pcs {
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
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0x0001_8000);
        cpu.regs.write_ar(3, 0x0004_0010);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.vecbase, 0x0001_8000);
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.ps, 0x0004_0010);
    }

    #[test]
    fn wsr_routes_mmu_config_special_registers() {
        // wsr.itlbcfg a2 (`20 5b 13`), wsr.ptevaddr a3 (`30 53 13`),
        // wsr.dtlbcfg a4 (`40 5c 13`): M2b Task 4 routes these into `cpu.mmu`
        // instead of dropping them, mirroring `wsr_routes_modeled_special_
        // registers` above for the interpreter's own SRs. wsr.rasid is
        // covered separately (`write_rasid` forces the ring-0 byte -- see
        // `mmu.rs`'s own test), so it isn't repeated here.
        let rom = vec![
            0x20, 0x5b, 0x13, // wsr.itlbcfg a2
            0x30, 0x53, 0x13, // wsr.ptevaddr a3
            0x40, 0x5c, 0x13, // wsr.dtlbcfg a4
        ];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(2, 0x0001_0000);
        cpu.regs.write_ar(3, 0x4000_0000);
        cpu.regs.write_ar(4, 0x0002_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.mmu.itlbcfg, 0x0001_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.mmu.ptevaddr, 0x4000_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.mmu.dtlbcfg, 0x0002_0000);
        assert_eq!(cpu.pc, 9);
        assert_eq!(cpu.regs.ps, 0, "MMU-config SR writes don't touch unrelated registers");
    }

    #[test]
    fn rsr_reads_back_mmu_config_special_registers() {
        // The read half of the above: rsr.itlbcfg a2 (`20 5b 03`) must return
        // the value `wsr.itlbcfg` stored, not the unmodeled-SR zero.
        let rom = vec![0x20, 0x5b, 0x03];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        cpu.mmu.itlbcfg = 0x0003_0000;
        cpu.regs.write_ar(2, 0xdead_beef);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0x0003_0000);
    }

    #[test]
    fn wdtlb_iitlb_idtlb_dsync_install_invalidate_and_are_logged_no_ops() {
        // wdtlb a7,a4 (`70 e4 50`) installs into the DTLB; iitlb a5
        // (`00 45 50`) / idtlb a5 (`00 c5 50`) invalidate; dsync (`30 20 00`)
        // remains an unmodeled logged no-op. All still advance pc.
        let rom = vec![0x70, 0xe4, 0x50, 0x00, 0x45, 0x50, 0x00, 0xc5, 0x50, 0x30, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        // wdtlb a7,a4: AS = AR[4] (way + VPN), AT = AR[7] (paddr|attr).
        cpu.regs.write_ar(4, 0x4000_1000 | 0); // way 0
        cpu.regs.write_ar(7, 0x0009_0000 | 0x3);
        for expected_pc in [3u32, 6, 9, 12] {
            assert!(matches!(cpu.step(&mut bus), Step::Ran));
            assert_eq!(cpu.pc, expected_pc);
        }
        // wdtlb installed a DTLB entry resolvable at pc==3.
        assert!(cpu.mmu.lookup(0x4000_1abc, true).is_ok());
    }

    #[test]
    fn witlb_installs_a_tlb_entry() {
        // witlb a7,a4 (`70 64 50`) then isync (`00 20 00`) -- M1.1 vectors,
        // concatenated. witlb now installs a real ITLB entry; isync remains
        // an unmodeled logged no-op. Both still advance pc.
        let rom = vec![0x70, 0x64, 0x50, 0x00, 0x20, 0x00];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        // witlb a7,a4: AS = AR[4] (way + VPN), AT = AR[7] (paddr|attr).
        cpu.regs.write_ar(4, 0x4000_1000 | 0); // way 0
        cpu.regs.write_ar(7, 0x0009_0000 | 0x3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert!(cpu.mmu.lookup(0x4000_1abc, false).is_ok());
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn witlb_installs_into_the_mmu() {
        use super::exec;
        use crate::firmware::mmio::Bus;
        use crate::firmware::xtensa::decode::Op;
        let mut cpu = Cpu::new(0);
        let mut bus = Bus::new(vec![0u8; 16]);
        // witlb a3,a4: AS = AR[4] (way 0 + VPN), AT = AR[3] (paddr|attr).
        cpu.regs.write_ar(4, 0x4000_1000 | 0);
        cpu.regs.write_ar(3, 0x0009_0000 | 0x3);
        let step = exec(&mut cpu, &mut bus, &Op::Witlb { t: 3, s: 4 }, 0, 3);
        assert!(matches!(step, Some(Step::Ran)));
        // The entry is now installed and resolvable.
        assert!(cpu.mmu.lookup(0x4000_1abc, false).is_ok());
    }

    #[test]
    fn idtlb_invalidates() {
        use super::exec;
        use crate::firmware::mmio::Bus;
        use crate::firmware::xtensa::decode::Op;
        let mut cpu = Cpu::new(0);
        let mut bus = Bus::new(vec![0u8; 16]);
        // wdtlb a3,a4 installs, then idtlb a4 (AS=AR[4]) invalidates the same entry.
        cpu.regs.write_ar(4, 0x4000_1000 | 0);
        cpu.regs.write_ar(3, 0x0009_0000 | 0x3);
        exec(&mut cpu, &mut bus, &Op::Wdtlb { t: 3, s: 4 }, 0, 3);
        assert!(cpu.mmu.lookup(0x4000_1abc, true).is_ok());
        let step = exec(&mut cpu, &mut bus, &Op::Idtlb { s: 4 }, 3, 3);
        assert!(matches!(step, Some(Step::Ran)));
        assert_eq!(cpu.mmu.lookup(0x4000_1abc, true), Err(24));
    }
}
