//! Memory load/store execute: `l32i.n`, `l32i`, `l32r`, plus the M2a sibling
//! widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`, `s32ri`.

use super::{Access, Cpu, Step};
use super::super::bus::XtensaBus;
use crate::firmware::xtensa::decode::Op;

/// Execute `op` if it's one of this category's ops (every `L*`/`S*` memory
/// op); `None` otherwise, so `step()` tries the next category. Every
/// effective address here is VIRTUAL -- each site routes through
/// [`Cpu::translate`] before touching `bus`, and a translation fault bails
/// out as `Some(Step::Exception)` without advancing `pc` (M2b Task 9).
pub(super) fn exec<B: XtensaBus + ?Sized>(
    cpu: &mut Cpu,
    bus: &mut B,
    op: &Op,
    pc: u32,
    len: u8,
) -> Option<Step> {
    match op {
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = match cpu.data_read32(bus, vaddr) {
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
            let v = match cpu.data_read8(bus, vaddr) {
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
            if let Err(step) = cpu.data_write8(bus, vaddr, val) {
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
            if let Err(step) = cpu.data_write32(bus, vaddr, val) {
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
            let tmp = match cpu.data_read32(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            if tmp == cpu.scompare1 {
                if let Err(step) = cpu.data_write32(bus, vaddr, new) {
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
            let v = match cpu.data_read32(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.fr[*ft as usize] = v;
        }
        Op::Ssi { ft, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.fr[*ft as usize]; // read before the &mut cpu borrow
            if let Err(step) = cpu.data_write32(bus, vaddr, val) {
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
            let v = match cpu.data_read32(bus, vaddr) {
                Ok(v) => v,
                Err(step) => return Some(step),
            };
            cpu.regs.write_ar(*t, v);
        }
        Op::S32e { t, s, imm } => {
            let vaddr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let val = cpu.regs.read_ar(*t); // read before the &mut cpu borrow
            if let Err(step) = cpu.data_write32(bus, vaddr, val) {
                return Some(step);
            }
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

/// Compose a little-endian 16-bit load from two [`Cpu::data_read8`] calls.
/// `Bus` has no native 16-bit accessor, so `l16ui`/`l16si` build the halfword from
/// bytes instead. Each byte's virtual address is translated independently
/// (rather than translating once and assuming +1 stays on the same physical
/// page) so a halfword straddling a page boundary faults faithfully --
/// mirroring the fetch page-safety in `step()`.
fn load16<B: XtensaBus + ?Sized>(cpu: &mut Cpu, bus: &mut B, addr: u32) -> Result<u16, Step> {
    let lo = cpu.data_read8(bus, addr)? as u16;
    let hi = cpu.data_read8(bus, addr.wrapping_add(1))? as u16;
    Ok(lo | (hi << 8))
}

/// Compose a little-endian 16-bit store from two [`Cpu::data_write8`] calls
/// (see [`load16`]).
fn store16<B: XtensaBus + ?Sized>(cpu: &mut Cpu, bus: &mut B, addr: u32, v: u16) -> Result<(), Step> {
    // No-half-write: validate both byte destinations (translate) before writing
    // either, so a page-straddling store16 whose high byte faults never applies
    // the low byte. Under translation-authoritative BOTH bytes translate/fault.
    let (lo, hi) = (addr, addr.wrapping_add(1));
    cpu.translate(bus, lo, Access::Store)?;
    cpu.translate(bus, hi, Access::Store)?;
    cpu.data_write8(bus, lo, (v & 0xFF) as u32)?;
    cpu.data_write8(bus, hi, (v >> 8) as u32)?;
    Ok(())
}

/// Route an `l32r` literal load. Xtensa translates L32R as a D-side load, but
/// the resolved literal can live in the image view alongside its code rather
/// than in mutable local data. Translate with [`Access::Load`], then select
/// that literal view through [`XtensaBus::literal_load32`].
///
/// This keeps a literal alive when a data write covers the address range it
/// shares, and lets a platform key the literal view by
/// virtual address.
fn l32r_load<B: XtensaBus + ?Sized>(cpu: &mut Cpu, bus: &mut B, target: u32) -> Result<u32, Step> {
    let paddr = cpu.translate(bus, target, Access::Load)?;
    Ok(bus.literal_load32(target, paddr))
}

#[cfg(test)]
mod tests {
    use crate::firmware::xtensa::mmu::Tlb;
    use super::super::{mapped_cpu, unmapped_mmu, Cpu, ExceptionVectors, Step};
    use crate::firmware::xtensa::flat_bus::FlatBus;

    /// Identity-map a data page into the DTLB (RWX, autorefill way 0) so a
    /// test's existing physical-looking addresses keep working now that
    /// load/store translate (M2b Task 9) -- the same role `mapped_cpu` plays
    /// for the ITLB. Way 0 never collides with `mapped_cpu`'s ITLB way-1
    /// mapping (separate TLBs); a test that maps two data pages sharing the
    /// same `(vaddr >> 12) & 0x3` slot would need a different way for the
    /// second, but none here do (each test touches a single data page).
    fn map_data(cpu: &mut Cpu, addr: u32) {
        let page = addr & 0xfffff000;
        cpu.mmu.write_entry(Tlb::Data, page | 0, page | 0x3);
    }

    #[test]
    fn executes_l32i_n_loads_from_bus() {
        // l32i.n a7,a2,44 -- `78 b2`.
        let rom = vec![0x78, 0xb2];
        let mut bus = FlatBus::new(rom);
        bus.write32(0x0100002c, 0xdeadbeef);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(2, 0x01000000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 0xdeadbeef);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l32i_loads_from_bus() {
        // l32i a9,a6,100 -- `92 26 19`.
        let rom = vec![0x92, 0x26, 0x19];
        let mut bus = FlatBus::new(rom);
        bus.write32(0x01000064, 0x1122_3344); // base + 100 (0x64)
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(6, 0x01000000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0x1122_3344);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_lsi_loads_into_fp_register() {
        // lsi f6,a9,0x40 -- `63 09 10`. Loads a
        // 32-bit word from AR[9]+0x40 into FP register f6 as OPAQUE bits (no
        // FP semantics -- this interpreter models FP registers as raw storage
        // so lsi/ssi move bit patterns to and from memory unchanged).
        let rom = vec![0x63, 0x09, 0x10];
        let mut bus = FlatBus::new(rom);
        bus.write32(0x0100_0040, 0x3f80_0000); // 1.0f bit pattern, treated as opaque
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(9, 0x0100_0000);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.fr[6], 0x3f80_0000, "lsi loads raw bits into fr[ft]");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_ssi_stores_fp_register() {
        // ssi f12,a7,0x2c -- `c3 47 0b` (ft=12, as=a7, r=4=ssi, imm8=11 -> off 0x2c).
        // Stores the raw bits of fr[12] to AR[7]+0x2c.
        let rom = vec![0xc3, 0x47, 0x0b];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(7, 0x0100_0000);
        cpu.fr[12] = 0xcafe_f00d;
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read32(0x0100_002c), 0xcafe_f00d, "ssi stores raw fr[ft] bits");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32i_n_stores_to_bus() {
        // s32i.n a3,a10,0x24 -- `39 9a`.
        let rom = vec![0x39, 0x9a];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(10, 0x01000000); // base
        cpu.regs.write_ar(3, 0x1122_3344); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read32(0x01000000 + 0x24), 0x1122_3344);
        assert_eq!(cpu.pc, 2);
    }

    #[test]
    fn executes_l8ui_zero_extends() {
        // l8ui a6,a3,0x1f -- `62 03 1f`.
        let rom = vec![0x62, 0x03, 0x1f];
        let mut bus = FlatBus::new(rom);
        bus.write8(0x0100011f, 0xF7);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000100);
        cpu.regs.write_ar(3, 0x01000100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(6), 0x0000_00F7); // zero-extended, not sign
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s8i_stores_low_byte_only() {
        // s8i a11,a2,0x41 -- `b2 42 41`.
        let rom = vec![0xb2, 0x42, 0x41];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(2, 0x01000000); // base
        cpu.regs.write_ar(11, 0x1234_56AB); // value; only the low byte should land
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read8(0x01000000 + 0x41), 0xAB);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32i_then_l32i_round_trips() {
        // s32i a7,a4,0x48 (`72 64 12`) followed by
        // l32i a9,a4,0x48 (`92 24 12`: r=2/t=9/s=4/imm8=0x12) reading the
        // same address back -- proves the store and the load agree on
        // address arithmetic.
        let rom = vec![0x72, 0x64, 0x12, 0x92, 0x24, 0x12];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000200);
        cpu.regs.write_ar(4, 0x01000200); // base
        cpu.regs.write_ar(7, 0xDEAD_BEEF); // value to store
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.pc, 3);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(9), 0xDEAD_BEEF);
        assert_eq!(cpu.pc, 6);
    }

    #[test]
    fn executes_l16ui_zero_extends() {
        // l16ui a8,a5,26 -- `82 15 0d`.
        let rom = vec![0x82, 0x15, 0x0d];
        let mut bus = FlatBus::new(rom);
        // Bit 15 set (0x8001): distinguishes zero-extend (-> 0x00008001)
        // from a wrongly sign-extended result (-> 0xFFFF8001).
        bus.write8(0x0100011a, 0x01);
        bus.write8(0x0100011b, 0x80);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000100);
        cpu.regs.write_ar(5, 0x01000100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(8), 0x0000_8001);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_l16si_sign_extends() {
        // l16si a4,a12,42 -- `42 9c 15`. Bit 15 of the loaded
        // halfword is set, so a correct sign-extending load must produce
        // 0xFFFF8001, not 0x00008001.
        let rom = vec![0x42, 0x9c, 0x15];
        let mut bus = FlatBus::new(rom);
        bus.write8(0x0100012a, 0x01);
        bus.write8(0x0100012b, 0x80);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000100);
        cpu.regs.write_ar(12, 0x01000100);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(4), 0xFFFF_8001);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s16i_stores_low_halfword_only() {
        // s16i a10,a3,22 -- `a2 53 0b`.
        let rom = vec![0xa2, 0x53, 0x0b];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(3, 0x01000000); // base
        cpu.regs.write_ar(10, 0x1234_ABCD); // value; only the low halfword should land
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read8(0x01000000 + 22), 0xCD); // low byte, little-endian
        assert_eq!(bus.read8(0x01000000 + 23), 0xAB); // high byte
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32ri_stores_like_s32i() {
        // s32ri a3,a7,0x1c4 -- `32 f7 71`. Store-release is
        // a distinct Op from s32i, but in this single-threaded interpreter
        // the release ordering is a no-op -- the memory effect is identical.
        let rom = vec![0x32, 0xf7, 0x71];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(7, 0x01000000); // base
        cpu.regs.write_ar(3, 0xCAFE_BABE); // value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read32(0x01000000 + 0x1c4), 0xCAFE_BABE);
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32c1i_stores_on_scompare1_match() {
        // s32c1i a4,a6,0x2d0 -- `42 e6 b4`. Atomic CAS:
        // stores AR[t] only when MEM == SCOMPARE1, and always returns the OLD
        // memory word in AR[t].
        let rom = vec![0x42, 0xe6, 0xb4];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        let addr = 0x01000000 + 0x2d0;
        bus.write32(addr, 0x1111_1111); // memory word
        cpu.scompare1 = 0x1111_1111; // compare MATCHES
        cpu.regs.write_ar(6, 0x01000000); // base
        cpu.regs.write_ar(4, 0xCAFE_BABE); // new value
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read32(addr), 0xCAFE_BABE, "stored on match");
        assert_eq!(cpu.regs.read_ar(4), 0x1111_1111, "returns old value");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_s32c1i_no_store_on_scompare1_mismatch() {
        // Same instruction; SCOMPARE1 differs from memory -> no store, and the
        // old (unchanged) word is returned in AR[t].
        let rom = vec![0x42, 0xe6, 0xb4];
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        let addr = 0x01000000 + 0x2d0;
        bus.write32(addr, 0x2222_2222); // memory word
        cpu.scompare1 = 0x1111_1111; // compare MISMATCHES
        cpu.regs.write_ar(6, 0x01000000);
        cpu.regs.write_ar(4, 0xCAFE_BABE);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(bus.read32(addr), 0x2222_2222, "unchanged on mismatch");
        assert_eq!(cpu.regs.read_ar(4), 0x2222_2222, "returns old value");
        assert_eq!(cpu.pc, 3);
    }

    #[test]
    fn executes_l32r_loads_from_resolved_target() {
        // l32r a5, 0x1f00 -- `51 c0 ff` at pc 0x2000 (target arithmetic in
        // decode/mem.rs: ((0x2000+3)&~3) + (0xffffffc0 << 2) = 0x1f00).
        // Build a ROM big enough to hold both the instruction and the
        // literal value it targets, with zeros in between.
        let mut rom = vec![0u8; 0x2003];
        rom[0x1f00..0x1f00 + 4].copy_from_slice(&0xcafe_babeu32.to_le_bytes());
        rom[0x2000..0x2000 + 3].copy_from_slice(&[0x51, 0xc0, 0xff]);
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0x2000);
        map_data(&mut cpu, 0x1f00); // literal page, separate from code
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(5), 0xcafe_babe);
        assert_eq!(cpu.pc, 0x2003);
    }

    // -- M2b Task 9: load/store + l32r through MMU translation -------------

    #[test]
    fn load_translates_through_dtlb() {
        use crate::firmware::xtensa::flat_bus::FlatBus;
        // l32i.n a7,a2,44 = `78 b2`.
        // Reads from AR[2]+44 (virtual), which we map to physical RAM.
        let rom = vec![0x78, 0xb2];
        let mut bus = FlatBus::new(rom);
        bus.write32(0x0100_002c, 0xfeed_face); // physical backing at base+44
        let mut cpu = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        // Map code page 0 (R+X) so the fetch works; map virtual data page
        // 0x40000000 -> physical RAM 0x01000000 (RWX).
        cpu.mmu.write_entry(Tlb::Instruction, 0x0000_0000 | 0, 0x0000_0000 | 0x1);
        cpu.mmu.write_entry(Tlb::Data, 0x4000_0000 | 0, 0x0100_0000 | 0x3);
        cpu.regs.write_ar(2, 0x4000_0000); // virtual base; +44 -> 0x4000002c
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(7), 0xfeed_face);
    }

    #[test]
    fn store_fault_raises_without_advancing_pc() {
        use crate::firmware::xtensa::flat_bus::FlatBus;
        // s32i.n a6,a7,0x30 = `69 c7` (VERIFIED vector). AR[7]+0x30 is the store
        // target; point it at an unmapped page so the store faults.
        let rom = vec![0x69, 0xc7];
        let mut bus = FlatBus::new(rom);
        let mut cpu = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        cpu.vecbase = 0x4000_0000;
        cpu.mmu.write_entry(Tlb::Instruction, 0x0000_0000 | 0, 0x0000_0000 | 0x1); // code page
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
        assert_eq!(cpu.pc, 0x4000_0000 + ExceptionVectors::default().kernel);
        assert_eq!(cpu.epc1, 0);
    }

    // -- Task 4: migration regression/equivalence locks ---------------------

    #[test]
    fn store16_high_byte_fault_leaves_low_byte_unwritten() {
        // s16i a10,a3,0 with a3 at the last byte of a mapped page; the high byte spills
        // into the next (unmapped) page -> fault, and the low byte must NOT be applied.
        let rom = vec![0xa2, 0x53, 0x00]; // s16i a10,a3,0
        let mut bus = FlatBus::new(rom);
        let mut cpu = Cpu::new(0, unmapped_mmu(), ExceptionVectors::default());
        cpu.vecbase = 0x4000_0000;
        cpu.mmu.write_entry(Tlb::Instruction, 0x0 | 0, 0x0 | 0x1); // code page R+X
                                                                   // Map data page 0x10000 -> RAM 0x01000000, but NOT the next page.
        cpu.mmu.write_entry(Tlb::Data, 0x0001_0000 | 0, 0x0100_0000 | 0x3);
        cpu.regs.write_ar(3, 0x0001_0fff); // last byte of the mapped page
        cpu.regs.write_ar(10, 0xABCD);
        match cpu.step(&mut bus) {
            Step::Exception { cause, .. } => assert_eq!(cause, 24), // LOAD_STORE_TLB_MISS on the high byte
            other => panic!("expected straddle fault, got {other:?}"),
        }
        assert_eq!(bus.read8(0x0100_0fff), 0, "low byte must NOT be applied on high-byte fault");
    }

    #[test]
    fn executor_result_equals_probe_read_dside() {
        // Equivalence invariant (D-side loads only; l32r is I-side by design and excluded):
        // a store executed by the CPU is read back identically by cpu.data_read32.
        let rom = vec![0x39, 0x9a]; // s32i.n a3,a10,0x24
        let mut bus = FlatBus::new(rom);
        let mut cpu = mapped_cpu(0);
        map_data(&mut cpu, 0x01000000);
        cpu.regs.write_ar(10, 0x0100_0000);
        cpu.regs.write_ar(3, 0x1234_5678);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.data_read32(&mut bus, 0x0100_0024).unwrap(), 0x1234_5678, "probe == CPU");
    }
}
