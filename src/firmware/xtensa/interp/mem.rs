//! Memory load/store execute: `l32i.n`, `l32i`, `l32r`, plus the M2a sibling
//! widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`, `s32ri`.

use super::{Access, Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (every `L*`/`S*` memory
/// op); `None` otherwise, so `step()` tries the next category. Every
/// effective address here is VIRTUAL -- each site routes through
/// [`Cpu::translate`] before touching `bus`, and a translation fault bails
/// out as `Some(Step::Exception)` without advancing `pc` (M2b Task 9).
pub(super) fn exec(cpu: &mut Cpu, bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match data_load32(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L32r { t, target } => {
            let v = match data_load32(cpu, bus, *target) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L8ui { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match data_load8(cpu, bus, vaddr) {
                Ok(v) => v as u32,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L16ui { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match load16(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v as u32);
        }
        Op::L16si { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match load16(cpu, bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            // Sign-extend the 16-bit load through i16 -> i32 -> u32.
            cpu.regs.write_ar(*t, v as i16 as i32 as u32);
        }
        Op::S8i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = data_store8(cpu, bus, vaddr, val) {
                return Some(step);
            }
        }
        Op::S16i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = cpu.regs.read_ar(*t) as u16;
            if let Err(step) = store16(cpu, bus, vaddr, v) {
                return Some(step);
            }
        }
        // s32i.n/s32i/s32ri all perform the identical 32-bit store; s32ri's
        // release-consistency ordering has no observable effect in this
        // single-threaded interpreter (see the `Op::S32ri` doc comment in
        // decode/mod.rs), so it shares this arm rather than duplicating it.
        Op::S32iN { t, s, imm } | Op::S32i { t, s, imm } | Op::S32ri { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = data_store32(cpu, bus, vaddr, val) {
                return Some(step);
            }
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

/// Compose a little-endian 16-bit load from two [`Bus::load8`] calls. `Bus`
/// has no native 16-bit accessor -- M2a keeps `Bus` changes out of scope (see
/// the M2a task-2 brief), so `l16ui`/`l16si` build the halfword from bytes
/// instead. Each byte's virtual address is translated independently (rather
/// than translating once and assuming +1 stays on the same physical page) so
/// a halfword straddling a page boundary faults faithfully -- mirroring the
/// fetch page-safety in `step()` (M2b Task 8).
fn load16(cpu: &mut Cpu, bus: &mut Bus, addr: u32) -> Result<u16, Step> {
    let lo = data_load8(cpu, bus, addr)? as u16;
    let hi = data_load8(cpu, bus, addr.wrapping_add(1))? as u16;
    Ok(lo | (hi << 8))
}

/// Compose a little-endian 16-bit store from two [`Bus::store8`] calls (see
/// [`load16`]).
fn store16(cpu: &mut Cpu, bus: &mut Bus, addr: u32, v: u16) -> Result<(), Step> {
    // Preserve the original no-half-write guarantee: validate both byte
    // destinations before writing either, so a fault on the high byte never
    // leaves the low byte's store applied. A local byte never faults; a
    // non-local byte is validated by a translate probe (raises on fault, no
    // write). Only after both are known routable do we write.
    let (lo, hi) = (addr, addr.wrapping_add(1));
    if !Bus::is_local_data(lo) {
        cpu.translate(bus, lo, Access::Store)?;
    }
    if !Bus::is_local_data(hi) {
        cpu.translate(bus, hi, Access::Store)?;
    }
    data_store8(cpu, bus, lo, (v & 0xFF) as u32)?;
    data_store8(cpu, bus, hi, (v >> 8) as u32)?;
    Ok(())
}

/// Route a 32-bit data LOAD: a low-window vaddr reads local data memory
/// (MMU-bypassed); anything else translates and reads the paddr backing.
fn data_load32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u32, Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 0 /*load*/);
        Ok(bus.load_local32(vaddr))
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Load)?;
        Ok(bus.load32(paddr))
    }
}

/// Route a byte data LOAD (see [`data_load32`]).
fn data_load8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32) -> Result<u8, Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 0 /*load*/);
        Ok(bus.load_local8(vaddr))
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Load)?;
        Ok(bus.load8(paddr))
    }
}

/// Route a 32-bit data STORE (see [`data_load32`]).
fn data_store32(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 1 /*store*/);
        bus.store_local32(vaddr, v);
        Ok(())
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Store)?;
        bus.store32(paddr, v);
        Ok(())
    }
}

/// Route a byte data STORE, storing the low byte of `v` (see [`data_load32`]).
fn data_store8(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, v: u32) -> Result<(), Step> {
    if Bus::is_local_data(vaddr) {
        assert_low_window_identity(cpu, bus, vaddr, 1 /*store*/);
        bus.store_local8(vaddr, v);
        Ok(())
    } else {
        let paddr = cpu.translate(bus, vaddr, Access::Store)?;
        bus.store8(paddr, v & 0xFF);
        Ok(())
    }
}

/// Safety net for the local-memory bypass: the low window is the varway56
/// way-6 identity across the whole probed boot (no autorefill, no fault), and
/// this task runs PAST that boot. Assert the low-window vaddr still identity-
/// maps so a FUTURE non-identity remap fails loudly instead of silently reading
/// the wrong backing. Lenient: passes on a TLB miss (unit-test CPUs don't map
/// the low window) and on an identity hit; fails only on a non-identity hit.
/// Uses the non-raising `cpu.mmu.translate` so the speculative check never
/// perturbs pc/epc1/exccause; compiled out in release.
fn assert_low_window_identity(cpu: &mut Cpu, bus: &mut Bus, vaddr: u32, mode: u8) {
    debug_assert!(
        !matches!(cpu.mmu.translate(bus, vaddr, mode, 0), Ok(t) if t.paddr != vaddr),
        "low-window data vaddr {vaddr:#x} translates to non-identity paddr -- local-memory bypass may be wrong"
    );
}

#[cfg(test)]
mod tests {
    use super::super::{mapped_cpu, Cpu, Step};
    use crate::firmware::mmio::Bus;

    /// Identity-map a data page into the DTLB (RWX, autorefill way 0) so a
    /// test's existing physical-looking addresses keep working now that
    /// load/store translate (M2b Task 9) -- the same role `mapped_cpu` plays
    /// for the ITLB. Way 0 never collides with `mapped_cpu`'s ITLB way-1
    /// mapping (separate TLBs); a test that maps two data pages sharing the
    /// same `(vaddr >> 12) & 0x3` slot would need a different way for the
    /// second, but none here do (each test touches a single data page).
    fn map_data(cpu: &mut Cpu, addr: u32) {
        let page = addr & 0xfffff000;
        cpu.mmu.write_tlb(true, page | 0x3, page | 0);
    }

    #[test]
    fn executes_l32i_n_loads_from_bus() {
        // l32i.n a4,a5,16 -- `48 45` (M1.1 vector).
        let rom = vec![0x48, 0x45];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b00010, 0xdeadbeef);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(5, 0x08b00000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xdeadbeef);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l32i_loads_from_bus() {
        // l32i a5,a2,40 -- `52 22 0a` (M1.1 vector).
        let rom = vec![0x52, 0x22, 0x0a];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b00028, 0x1122_3344); // base + 40 (0x28)
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(2, 0x08b00000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x1122_3344);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32i_n_stores_to_bus() {
        // s32i.n a6,a7,0x30 -- `69 c7` (task-2 vector).
        let rom = vec![0x69, 0xc7];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(7, 0x08b00000); // base
        cpu.regs.write_ar(6, 0x1122_3344); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.load32(0x08b00000 + 0x30), 0x1122_3344);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l8ui_zero_extends() {
        // l8ui a2,a2,0 -- `22 02 00` (task-2 vector). t==s==2: the base
        // (read from a2 before the op runs) and the destination (a2,
        // overwritten by the op) are the same register, as the real
        // encoding dictates.
        let rom = vec![0x22, 0x02, 0x00];
        let mut bus = Bus::new(rom);
        bus.store8(0x08b00100, 0xF7);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00100);
        cpu.regs.write_ar(2, 0x08b00100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0x0000_00F7); // zero-extended, not sign
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s8i_stores_low_byte_only() {
        // s8i a8,a4,44 -- `82 44 2c` (task-2 vector).
        let rom = vec![0x82, 0x44, 0x2c];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(4, 0x08b00000); // base
        cpu.regs.write_ar(8, 0x1234_56AB); // value; only the low byte should land
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.load8(0x08b00000 + 44), 0xAB);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32i_then_l32i_round_trips() {
        // s32i a2,a1,0x1c (`22 61 07`, task-2 vector) followed by a
        // hand-built l32i a5,a1,0x1c (`52 21 07`, encoding confirmed via
        // xtensa-lx106-elf-objdump: r=2/t=5/s=1/imm8=7) reading the same
        // address back -- proves the store and the (already-proven, M1.4)
        // load agree on address arithmetic.
        let rom = vec![0x22, 0x61, 0x07, 0x52, 0x21, 0x07];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00200);
        cpu.regs.write_ar(1, 0x08b00200); // base
        cpu.regs.write_ar(2, 0xDEAD_BEEF); // value to store
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0xDEAD_BEEF);
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn executes_l16ui_zero_extends() {
        // l16ui a3,a3,4 -- `32 13 02` (task-2 vector). t==s==3, same
        // same-register base/dest note as l8ui above.
        let rom = vec![0x32, 0x13, 0x02];
        let mut bus = Bus::new(rom);
        // Bit 15 set (0x8001): distinguishes zero-extend (-> 0x00008001)
        // from a wrongly sign-extended result (-> 0xFFFF8001).
        bus.store8(0x08b00104, 0x01);
        bus.store8(0x08b00105, 0x80);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00100);
        cpu.regs.write_ar(3, 0x08b00100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(3), 0x0000_8001);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_l16si_sign_extends() {
        // l16si a2,a2,0 -- `22 92 00` (task-2 vector). Bit 15 of the loaded
        // halfword is set, so a correct sign-extending load must produce
        // 0xFFFF8001, not 0x00008001.
        let rom = vec![0x22, 0x92, 0x00];
        let mut bus = Bus::new(rom);
        bus.store8(0x08b00100, 0x01);
        bus.store8(0x08b00101, 0x80);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00100);
        cpu.regs.write_ar(2, 0x08b00100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0xFFFF_8001);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s16i_stores_low_halfword_only() {
        // s16i a4,a7,4 -- `42 57 02` (task-2 vector).
        let rom = vec![0x42, 0x57, 0x02];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(7, 0x08b00000); // base
        cpu.regs.write_ar(4, 0x1234_ABCD); // value; only the low halfword should land
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.load8(0x08b00000 + 4), 0xCD); // low byte, little-endian
        assert_eq!(bus.load8(0x08b00000 + 5), 0xAB); // high byte
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32ri_stores_like_s32i() {
        // s32ri a10,a15,0x218 -- `a2 ff 86` (task-2 vector). Store-release is
        // a distinct Op from s32i, but in this single-threaded interpreter
        // the release ordering is a no-op -- the memory effect is identical.
        let rom = vec![0xa2, 0xff, 0x86];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(15, 0x08b00000); // base
        cpu.regs.write_ar(10, 0xCAFE_BABE); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.load32(0x08b00000 + 0x218), 0xCAFE_BABE);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_l32r_loads_from_resolved_target() {
        // l32r a2, 0x2d158 -- `21 bd e7` @ pc 0x33262 (M1.1 vector, target
        // already verified against real firmware disassembly in decode/mem.rs).
        // Build a ROM big enough to hold both the instruction and the
        // literal-pool value it targets, with zeros in between.
        let mut rom = vec![0u8; 0x33265];
        rom[0x2d158..0x2d158 + 4].copy_from_slice(&0xcafe_babeu32.to_le_bytes());
        rom[0x33262..0x33262 + 3].copy_from_slice(&[0x21, 0xbd, 0xe7]);
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0x33262);
        map_data(&mut cpu, 0x2d158); // literal-pool page, separate from code
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0xcafe_babe);
        assert_eq!(cpu.pc, 0x33265);
    }

    // -- M2b Task 9: load/store + l32r through MMU translation -------------

    #[test]
    fn load_translates_through_dtlb() {
        use crate::firmware::mmio::Bus;
        // l32i.n a4,a5,16 = `48 45` (VERIFIED vector from mem.rs's existing tests).
        // Reads from AR[5]+16 (virtual), which we map to physical RAM.
        let rom = vec![0x48, 0x45];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b0_0010, 0xfeed_face); // physical backing at base+16
        let mut cpu = Cpu::new(0);
        // Map code page 0 (R+X) so the fetch works; map virtual data page
        // 0x40000000 -> physical RAM 0x08b00000 (RWX).
        cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x0000_0000 | 0);
        cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x4000_0000 | 0);
        cpu.regs.write_ar(5, 0x4000_0000); // virtual base; +16 -> 0x40000010
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xfeed_face);
    }

    #[test]
    fn low_window_store_lands_in_local_data_not_image() {
        // s32i.n a6,a7,0x30 (`69 c7`) with a7 = 0x1000 -> effective vaddr 0x1030
        // (low window). The store must land in local_data and NOT corrupt the image.
        let mut bus = Bus::new(vec![0x69, 0xc7]);
        let mut cpu = mapped_cpu(0);
        cpu.regs.write_ar(7, 0x1000); // low-window data base
        cpu.regs.write_ar(6, 0x1122_3344); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        // Landed in local_data, read back by the same low vaddr (0x1000 + 0x30).
        assert_eq!(bus.load_local32(0x1030), 0x1122_3344);
        // The image (paddr Rom path) is untouched -- anti-aliasing.
        assert_eq!(bus.load32(0x1030), 0);
    }

    #[test]
    fn low_window_load_reads_local_data_blank_then_stored() {
        // l32i.n a4,a5,0x10 (`48 45`) with a5 = 0x1000 -> effective vaddr 0x1010.
        // Reads local_data: blank (0) until a prior store, then the stored value.
        let mut cpu = mapped_cpu(0);
        let mut bus = Bus::new(vec![0x48, 0x45]);
        cpu.regs.write_ar(5, 0x1000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0, "blank local_data reads 0, not the image");
        // Store into local_data at the same vaddr and re-load.
        bus.store_local32(0x1010, 0xcafe_babe);
        cpu.pc = 0;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xcafe_babe);
    }

    #[test]
    fn high_window_data_still_uses_translate_and_image() {
        // Regression: a data access at vaddr >= LOCAL_DATA_END is unchanged -- it
        // translates and hits the paddr backing (RAM here), NOT local_data. This is
        // the existing `executes_l32i_n_loads_from_bus` path, guarding that the new
        // local branch does NOT capture high addresses.
        let mut bus = Bus::new(vec![0x48, 0x45]); // l32i.n a4,a5,0x10
        bus.store32(0x08b0_0010, 0xdead_beef); // RAM aperture, paddr path
        let mut cpu = mapped_cpu(0);
        let page = 0x08b0_0000u32 & 0xffff_f000;
        cpu.mmu.write_tlb(true, page | 0x3, page | 0); // DTLB identity, way 0
        cpu.regs.write_ar(5, 0x08b0_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xdead_beef);
        assert!(!Bus::is_local_data(0x08b0_0010));
    }

    #[test]
    fn store_fault_raises_without_advancing_pc() {
        use crate::firmware::mmio::Bus;
        // s32i.n a6,a7,0x30 = `69 c7` (VERIFIED vector). AR[7]+0x30 is the store
        // target; point it at an unmapped page so the store faults.
        let rom = vec![0x69, 0xc7];
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x4000_0000;
        cpu.mmu.write_tlb(false, 0x0000_0000 | 0x1, 0x0000_0000 | 0); // code page
        cpu.regs.write_ar(7, 0x5000_0000); // unmapped data page -> DTLB miss on store
        match cpu.step(&mut bus) {
            Step::Exception { cause, .. } => assert_eq!(cause, 24), // LOAD_STORE_TLB_MISS
            other => panic!("expected store fault, got {:?}", other),
        }
        // The faulting store did NOT advance pc by the instruction's own
        // length (2, to 0x2) -- it vectored to the KernelExceptionVector
        // instead, exactly like a Fetch fault (`translate_raises_itlb_miss_as_exception`
        // in mod.rs): Task 7's `raise_general_exception` is the one chokepoint
        // for both, with no Task-9 special-casing (see `Cpu::translate`'s doc
        // comment). EPC1 holds the faulting instruction's own pc (0), not the
        // vector.
        assert_eq!(cpu.pc, 0x4000_0000 + 0x2e0);
        assert_eq!(cpu.epc1, 0);
    }
}
