//! Memory load/store execute: `l32i.n`, `l32i`, `l32r`, plus the M2a sibling
//! widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`, `s32ri`.

use super::{Access, Cpu, Step};
use crate::firmware::mmio::CpuBus;
use crate::firmware::xtensa::decode::Op;

/// Execute `op` if it's one of this category's ops (every `L*`/`S*` memory
/// op); `None` otherwise, so `step()` tries the next category. Every
/// effective address here is VIRTUAL -- each site routes through
/// [`Cpu::translate`] before touching `bus`, and a translation fault bails
/// out as `Some(Step::Exception)` without advancing `pc` (M2b Task 9).
pub(super) fn exec(cpu: &mut Cpu, bus: &mut CpuBus<'_>, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match cpu.data_read32_on(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L32r { t, target } => {
            let v = match l32r_load(cpu, bus, *target) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::L8ui { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match cpu.data_read8_on(bus, vaddr) {
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
            if let Err(step) = cpu.data_write8_on(bus, vaddr, val) {
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
            if let Err(step) = cpu.data_write32_on(bus, vaddr, val) {
                return Some(step);
            }
        }
        // s32c1i: atomic compare-and-store. tmp = MEM[vaddr]; if tmp ==
        // SCOMPARE1 then MEM[vaddr] = AR[t]; AR[t] = tmp. The load and the
        // conditional store are separate translate+fault steps (a fault on
        // either bails without advancing pc); non-atomic is observationally
        // identical in this single-threaded interpreter.
        Op::S32c1i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let new = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            let tmp = match cpu.data_read32_on(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            if tmp == cpu.scompare1 {
                if let Err(step) = cpu.data_write32_on(bus, vaddr, new) {
                    return Some(step);
                }
            }
            cpu.regs.write_ar(*t, tmp);
        }
        // lsi/ssi: FP load/store single. The FP register file holds raw bits
        // (no FP semantics -- see Cpu::fr), so these are plain 32-bit moves
        // between memory and fr[ft], sharing the same translate+fault path as
        // l32i.n/s32i.n.
        Op::Lsi { ft, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match cpu.data_read32_on(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.fr[*ft as usize] = v;
        }
        Op::Ssi { ft, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.fr[*ft as usize]; // read before the &mut cpu borrow
            if let Err(step) = cpu.data_write32_on(bus, vaddr, val) {
                return Some(step);
            }
        }
        // s32e/l32e: windowed-exception store/load. Identical to s32i/l32i
        // except for the small negative-only offset (already folded into `imm`)
        // and privilege -- the address/data registers are read through the
        // CURRENT (exception-rotated) window, so a plain windowed access is
        // correct. Used only inside the window overflow/underflow handlers.
        Op::L32e { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match cpu.data_read32_on(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::S32e { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = cpu.data_write32_on(bus, vaddr, val) {
                return Some(step);
            }
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

/// Compose a little-endian 16-bit load from two [`Cpu::data_read8`] calls.
/// `Bus` has no native 16-bit accessor -- M2a keeps `Bus` changes out of scope
/// (see the M2a task-2 brief), so `l16ui`/`l16si` build the halfword from
/// bytes instead. Each byte's virtual address is translated independently
/// (rather than translating once and assuming +1 stays on the same physical
/// page) so a halfword straddling a page boundary faults faithfully --
/// mirroring the fetch page-safety in `step()` (M2b Task 8).
fn load16(cpu: &mut Cpu, bus: &mut CpuBus<'_>, addr: u32) -> Result<u16, Step> {
    let lo = cpu.data_read8_on(bus, addr)? as u16;
    let hi = cpu.data_read8_on(bus, addr.wrapping_add(1))? as u16;
    Ok(lo | (hi << 8))
}

/// Compose a little-endian 16-bit store from two [`Cpu::data_write8`] calls
/// (see [`load16`]).
fn store16(cpu: &mut Cpu, bus: &mut CpuBus<'_>, addr: u32, v: u16) -> Result<(), Step> {
    // No-half-write: validate both byte destinations (translate) before writing
    // either, so a page-straddling store16 whose high byte faults never applies
    // the low byte. Under translation-authoritative BOTH bytes translate/fault.
    let (lo, hi) = (addr, addr.wrapping_add(1));
    cpu.translate(bus.bus(), lo, Access::Store)?;
    cpu.translate(bus.bus(), hi, Access::Store)?;
    cpu.data_write8_on(bus, lo, (v & 0xFF) as u32)?;
    cpu.data_write8_on(bus, hi, (v >> 8) as u32)?;
    Ok(())
}

/// Route an `l32r` literal load. Unlike a general data load, an `l32r` reads
/// its literal from the instruction-stream literal pool, which lives WITH the
/// code in instruction memory (IRAM) -- not in the mutable DRAM data scratch
/// (`local_data`) that a general D-side low-window access routes to. So
/// `l32r` reads via [`Bus::inst_load32`] (the I-side accessor), never
/// [`Cpu::data_read32`], even for a low-window target. This is the iter12
/// fix: the kernel exception-vector dispatch (`l32r a3,=dispatcher; jx a3`)
/// read its literal from `local_data`, which the boot's low-window DRAM
/// memset (`fill 0x4..0xff0`) had zeroed -- so the stub jumped to PC=0. On
/// silicon the memset zeroes DRAM and cannot touch the IRAM literal pool;
/// `l32r` reads the surviving literal. Grounded in Xtensa L32R semantics:
/// L32R is THE instruction-stream literal load.
fn l32r_load(cpu: &mut Cpu, bus: &mut CpuBus<'_>, target: u32) -> Result<u32, Step> {
    let paddr = cpu.translate(bus.bus(), target, Access::Load)?;
    Ok(bus.bus().inst_load32_overlay(target, paddr))
}

#[cfg(test)]
mod tests {
    use super::super::{mapped_cpu, Cpu, Step, KERNEL_EXCEPTION_VECTOR_OFFSET};
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
        bus.data_store32(0x08b00010, 0xdeadbeef);
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
        bus.data_store32(0x08b00028, 0x1122_3344); // base + 40 (0x28)
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(2, 0x08b00000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0x1122_3344);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_lsi_loads_into_fp_register() {
        // lsi f10,a12,0x18c -- `a3 0c 63` (the firmware's 0xd830 wall). Loads a
        // 32-bit word from AR[12]+0x18c into FP register f10 as OPAQUE bits (no
        // FP semantics -- this interpreter models FP registers as raw storage
        // for the exception handler's context save/restore).
        let rom = vec![0xa3, 0x0c, 0x63];
        let mut bus = Bus::new(rom);
        bus.data_store32(0x08b0_018c, 0x3f80_0000); // 1.0f bit pattern, treated as opaque
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(12, 0x08b0_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.fr[10], 0x3f80_0000, "lsi loads raw bits into fr[ft]");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_ssi_stores_fp_register() {
        // ssi f3,a2,0x10 -- `33 42 04` (ft=3, as=a2, r=4=ssi, imm8=4 -> off 0x10).
        // Stores the raw bits of fr[3] to AR[2]+0x10.
        let rom = vec![0x33, 0x42, 0x04];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(2, 0x08b0_0000);
        cpu.fr[3] = 0xcafe_f00d;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.data_load32(0x08b0_0010), 0xcafe_f00d, "ssi stores raw fr[ft] bits");
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
        assert_eq!(bus.data_load32(0x08b00000 + 0x30), 0x1122_3344);
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
        bus.data_store8(0x08b00100, 0xF7);
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
        assert_eq!(bus.data_load8(0x08b00000 + 44), 0xAB);
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
    fn firmware_step_borrows_the_array_interpreters_device() {
        // s32i a2,a1,0; l32i a5,a1,0. The firmware CPU must program and read
        // the same DeviceState the array interpreter owns, without moving or
        // cloning it into the firmware bus.
        let rom = vec![0x22, 0x61, 0x00, 0x52, 0x21, 0x00];
        let mut bus = Bus::new(rom);
        let mut device = crate::device::DeviceState::new_npu1();
        let mut cpu = mapped_cpu(0);
        let addr = 0x0400_0000 + (1 << 25) + (2 << 20) + 0x70000;
        map_data(&mut cpu, addr);
        cpu.regs.write_ar(1, addr);
        cpu.regs.write_ar(2, 0xABCD_1234);

        assert!(matches!(cpu.step_with_device(&mut bus, &mut device), Step::Ran));
        assert_eq!(device.read_tile_register(1, 2, 0x70000), 0xABCD_1234);
        assert!(matches!(cpu.step_with_device(&mut bus, &mut device), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0xABCD_1234);
    }

    #[test]
    fn executes_l16ui_zero_extends() {
        // l16ui a3,a3,4 -- `32 13 02` (task-2 vector). t==s==3, same
        // same-register base/dest note as l8ui above.
        let rom = vec![0x32, 0x13, 0x02];
        let mut bus = Bus::new(rom);
        // Bit 15 set (0x8001): distinguishes zero-extend (-> 0x00008001)
        // from a wrongly sign-extended result (-> 0xFFFF8001).
        bus.data_store8(0x08b00104, 0x01);
        bus.data_store8(0x08b00105, 0x80);
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
        bus.data_store8(0x08b00100, 0x01);
        bus.data_store8(0x08b00101, 0x80);
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
        assert_eq!(bus.data_load8(0x08b00000 + 4), 0xCD); // low byte, little-endian
        assert_eq!(bus.data_load8(0x08b00000 + 5), 0xAB); // high byte
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
        assert_eq!(bus.data_load32(0x08b00000 + 0x218), 0xCAFE_BABE);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32c1i_stores_on_scompare1_match() {
        // s32c1i a0,a5,0x308 -- `02 e5 c2` (firmware @0xd900). Atomic CAS:
        // stores AR[t] only when MEM == SCOMPARE1, and always returns the OLD
        // memory word in AR[t].
        let rom = vec![0x02, 0xe5, 0xc2];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        let addr = 0x08b00000 + 0x308;
        bus.data_store32(addr, 0x1111_1111); // memory word
        cpu.scompare1 = 0x1111_1111; // compare MATCHES
        cpu.regs.write_ar(5, 0x08b00000); // base
        cpu.regs.write_ar(0, 0xCAFE_BABE); // new value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.data_load32(addr), 0xCAFE_BABE, "stored on match");
        assert_eq!(cpu.regs.read_ar(0), 0x1111_1111, "returns old value");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32c1i_no_store_on_scompare1_mismatch() {
        // Same instruction; SCOMPARE1 differs from memory -> no store, and the
        // old (unchanged) word is returned in AR[t].
        let rom = vec![0x02, 0xe5, 0xc2];
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        let addr = 0x08b00000 + 0x308;
        bus.data_store32(addr, 0x2222_2222); // memory word
        cpu.scompare1 = 0x1111_1111; // compare MISMATCHES
        cpu.regs.write_ar(5, 0x08b00000);
        cpu.regs.write_ar(0, 0xCAFE_BABE);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.data_load32(addr), 0x2222_2222, "unchanged on mismatch");
        assert_eq!(cpu.regs.read_ar(0), 0x2222_2222, "returns old value");
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

    #[test]
    fn low_window_l32r_reads_image_not_clobbered_local_data() {
        // Regression (iter12): the kernel exception-vector dispatch is
        //   wsr.excsave1 a3;  l32r a3,=<dispatcher>;  jx a3
        // whose literal lives with the code in instruction memory (IRAM). The
        // firmware's low-window DRAM memset (`fill 0x4..0xff0`) must NOT clobber
        // that literal, because an `l32r` reads it from the pristine image, not
        // the mutable DRAM overlay (`local_data`). Before the fix, the memset
        // had zeroed the overlay over the literal, so `l32r` loaded 0 and the
        // stub `jx`-ed to PC=0 -- the iter12 boot wall.
        //
        // Reuse the verified low-window vector: l32r a2, 0x2d158 (`21 bd e7`
        // @ pc 0x33262). The image literal holds the real dispatcher 0x28b4.
        let mut rom = vec![0u8; 0x33265];
        rom[0x2d158..0x2d158 + 4].copy_from_slice(&0x0000_28b4u32.to_le_bytes());
        rom[0x33262..0x33262 + 3].copy_from_slice(&[0x21, 0xbd, 0xe7]);
        let mut bus = Bus::new(rom);
        // Simulate the DRAM memset clobbering the overlay at the literal's vaddr.
        bus.store_local32(0x2d158, 0);
        let mut cpu = mapped_cpu(0x33262);
        map_data(&mut cpu, 0x2d158);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(
            cpu.regs.read_ar(2),
            0x0000_28b4,
            "l32r must read the IRAM image literal, not the zeroed DRAM overlay"
        );
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
        bus.data_store32(0x08b0_0010, 0xfeed_face); // physical backing at base+16
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
        map_data(&mut cpu, 0x1000); // low-window DTLB identity (Task 4: translation now authoritative)
        cpu.regs.write_ar(7, 0x1000); // low-window data base
        cpu.regs.write_ar(6, 0x1122_3344); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        // Landed in local_data, read back by the same low vaddr (0x1000 + 0x30).
        assert_eq!(bus.load_local32(0x1030), 0x1122_3344);
        // The image (I-side path) is untouched -- anti-aliasing.
        assert_eq!(bus.inst_load32(0x1030), 0);
    }

    #[test]
    fn low_window_load_reads_local_data_blank_then_stored() {
        // l32i.n a4,a5,0x10 (`48 45`) with a5 = 0x1000 -> effective vaddr 0x1010.
        // Reads local_data: blank (0) until a prior store, then the stored value.
        let mut cpu = mapped_cpu(0);
        let mut bus = Bus::new(vec![0x48, 0x45]);
        map_data(&mut cpu, 0x1000); // low-window DTLB identity (Task 4: translation now authoritative)
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
        bus.data_store32(0x08b0_0010, 0xdead_beef); // RAM aperture, paddr path
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
        // length (2, to 0x2) -- it vectored to the unified general-exception
        // vector instead, exactly like a Fetch fault
        // (`translate_raises_itlb_miss_as_exception` in mod.rs): Task 7's
        // `raise_general_exception` is the one chokepoint for both, with no
        // Task-9 special-casing (see `Cpu::translate`'s doc comment). EPC1
        // holds the faulting instruction's own pc (0), not the vector.
        assert_eq!(cpu.pc, 0x4000_0000 + KERNEL_EXCEPTION_VECTOR_OFFSET);
        assert_eq!(cpu.epc1, 0);
    }

    // -- Task 4: migration regression/equivalence locks ---------------------

    #[test]
    fn store16_high_byte_fault_leaves_low_byte_unwritten() {
        // s16i a4,a7,0 with a7 at the last byte of a mapped page; the high byte spills
        // into the next (unmapped) page -> fault, and the low byte must NOT be applied.
        let rom = vec![0x42, 0x57, 0x00]; // s16i a4,a7,0
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(0);
        cpu.vecbase = 0x4000_0000;
        cpu.mmu.write_tlb(false, 0x0 | 0x1, 0x0 | 0); // code page R+X
                                                      // Map data page 0x10000 -> RAM 0x08b00000, but NOT the next page.
        cpu.mmu.write_tlb(true, 0x08b0_0000 | 0x3, 0x0001_0000 | 0);
        cpu.regs.write_ar(7, 0x0001_0fff); // last byte of the mapped page
        cpu.regs.write_ar(4, 0xABCD);
        match cpu.step(&mut bus) {
            Step::Exception { cause, .. } => assert_eq!(cause, 24), // LOAD_STORE_TLB_MISS on the high byte
            other => panic!("expected straddle fault, got {other:?}"),
        }
        assert_eq!(bus.data_load8(0x08b0_0fff), 0, "low byte must NOT be applied on high-byte fault");
    }

    #[test]
    fn executor_result_equals_probe_read_dside() {
        // Equivalence invariant (D-side loads only; l32r is I-side by design and excluded):
        // a store executed by the CPU is read back identically by cpu.data_read32.
        let rom = vec![0x69, 0xc7]; // s32i.n a6,a7,0x30
        let mut bus = Bus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x08b00000);
        cpu.regs.write_ar(7, 0x08b0_0000);
        cpu.regs.write_ar(6, 0x1234_5678);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.data_read32(&mut bus, 0x08b0_0030).unwrap(), 0x1234_5678, "probe == CPU");
    }
}
