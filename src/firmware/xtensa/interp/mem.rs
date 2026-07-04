//! Memory load/store execute: `l32i.n`, `l32i`, `l32r`, plus the M2a sibling
//! widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`, `s32ri`.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of this category's ops (every `L*`/`S*` memory
/// op); `None` otherwise, so `step()` tries the next category.
pub(super) fn exec(cpu: &mut Cpu, bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    match op {
        Op::L32iN { t, s, imm } | Op::L32i { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = bus.load32(addr);
            cpu.regs.write_ar(*t, v);
        }
        Op::L32r { t, target } => {
            let v = bus.load32(*target);
            cpu.regs.write_ar(*t, v);
        }
        Op::L8ui { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = bus.load8(addr) as u32;
            cpu.regs.write_ar(*t, v);
        }
        Op::L16ui { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            let v = load16(bus, addr) as u32;
            cpu.regs.write_ar(*t, v);
        }
        Op::L16si { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            // Sign-extend the 16-bit load through i16 -> i32 -> u32.
            let v = load16(bus, addr) as i16 as i32 as u32;
            cpu.regs.write_ar(*t, v);
        }
        Op::S8i { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            bus.store8(addr, cpu.regs.read_ar(*t) & 0xFF);
        }
        Op::S16i { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            store16(bus, addr, cpu.regs.read_ar(*t) as u16);
        }
        // s32i.n/s32i/s32ri all perform the identical 32-bit store; s32ri's
        // release-consistency ordering has no observable effect in this
        // single-threaded interpreter (see the `Op::S32ri` doc comment in
        // decode/mod.rs), so it shares this arm rather than duplicating it.
        Op::S32iN { t, s, imm } | Op::S32i { t, s, imm } | Op::S32ri { t, s, imm } => {
            let addr = cpu.regs.read_ar(*s).wrapping_add(*imm);
            bus.store32(addr, cpu.regs.read_ar(*t));
        }
        _ => return None,
    }
    cpu.pc = pc.wrapping_add(len as u32);
    Some(Step::Ran)
}

/// Compose a little-endian 16-bit load from two [`Bus::load8`] calls. `Bus`
/// has no native 16-bit accessor -- M2a keeps `Bus` changes out of scope (see
/// the M2a task-2 brief), so `l16ui`/`l16si` build the halfword from bytes
/// instead.
fn load16(bus: &mut Bus, addr: u32) -> u16 {
    let lo = bus.load8(addr) as u16;
    let hi = bus.load8(addr.wrapping_add(1)) as u16;
    lo | (hi << 8)
}

/// Compose a little-endian 16-bit store from two [`Bus::store8`] calls (see
/// [`load16`]).
fn store16(bus: &mut Bus, addr: u32, v: u16) {
    bus.store8(addr, (v & 0xFF) as u32);
    bus.store8(addr.wrapping_add(1), (v >> 8) as u32);
}

#[cfg(test)]
mod tests {
    use super::super::{mapped_cpu, Step};
    use crate::firmware::mmio::Bus;

    #[test]
    fn executes_l32i_n_loads_from_bus() {
        // l32i.n a4,a5,16 -- `48 45` (M1.1 vector).
        let rom = vec![0x48, 0x45];
        let mut bus = Bus::new(rom);
        bus.store32(0x08b00010, 0xdeadbeef);
        let mut cpu = mapped_cpu(0);
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
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        assert_eq!(cpu.regs.read_ar(2), 0xcafe_babe);
        assert_eq!(cpu.pc, 0x33265);
    }
}
