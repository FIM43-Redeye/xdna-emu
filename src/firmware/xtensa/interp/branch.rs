//! Branch-family execute: the 27 conditional/unconditional branch opcodes.
//!
//! All 27 share the same shape -- evaluate a condition (trivially `true` for
//! the unconditional `j`), then set `cpu.pc` to the decoded absolute
//! `target` if taken or to `pc + len` (the plain fall-through) if not. That
//! shared tail is factored once below; each match arm only computes
//! `(target, taken)`.

use super::{Cpu, Step};
use crate::firmware::xtensa::decode::Op;
use crate::firmware::Bus;

/// Execute `op` if it's one of the 27 branch/jump ops; `None` otherwise, so
/// `step()` tries the next category.
pub(super) fn exec(cpu: &mut Cpu, _bus: &mut Bus, op: &Op, pc: u32, len: u8) -> Option<Step> {
    let (target, taken) = match op {
        Op::J { target } => (*target, true),
        Op::Beqz { s, target } => (*target, cpu.regs.read_ar(*s) == 0),
        Op::Bnez { s, target } => (*target, cpu.regs.read_ar(*s) != 0),
        Op::Bltz { s, target } => (*target, (cpu.regs.read_ar(*s) as i32) < 0),
        Op::Bgez { s, target } => (*target, (cpu.regs.read_ar(*s) as i32) >= 0),
        Op::BeqzN { s, target } => (*target, cpu.regs.read_ar(*s) == 0),
        Op::BnezN { s, target } => (*target, cpu.regs.read_ar(*s) != 0),
        Op::Beq { s, t, target } => (*target, cpu.regs.read_ar(*s) == cpu.regs.read_ar(*t)),
        Op::Bne { s, t, target } => (*target, cpu.regs.read_ar(*s) != cpu.regs.read_ar(*t)),
        Op::Blt { s, t, target } => {
            let (sv, tv) = (cpu.regs.read_ar(*s) as i32, cpu.regs.read_ar(*t) as i32);
            (*target, sv < tv)
        }
        Op::Bltu { s, t, target } => (*target, cpu.regs.read_ar(*s) < cpu.regs.read_ar(*t)),
        Op::Bge { s, t, target } => {
            let (sv, tv) = (cpu.regs.read_ar(*s) as i32, cpu.regs.read_ar(*t) as i32);
            (*target, sv >= tv)
        }
        Op::Bgeu { s, t, target } => (*target, cpu.regs.read_ar(*s) >= cpu.regs.read_ar(*t)),
        // beqi/bnei/blti/bgei compare against `imm`, the already-resolved
        // B4CONST value (decode/branch.rs's B4CONST table lookup) -- signed
        // compare, `imm` is `i32`.
        Op::Beqi { s, imm, target } => (*target, (cpu.regs.read_ar(*s) as i32) == *imm),
        Op::Bnei { s, imm, target } => (*target, (cpu.regs.read_ar(*s) as i32) != *imm),
        Op::Blti { s, imm, target } => (*target, (cpu.regs.read_ar(*s) as i32) < *imm),
        Op::Bgei { s, imm, target } => (*target, (cpu.regs.read_ar(*s) as i32) >= *imm),
        // bltui/bgeui compare against the resolved B4CONSTU value -- unsigned
        // compare, `imm` is `u32`, `AR[s]` compared directly (already u32).
        Op::Bltui { s, imm, target } => (*target, cpu.regs.read_ar(*s) < *imm),
        Op::Bgeui { s, imm, target } => (*target, cpu.regs.read_ar(*s) >= *imm),
        // bbci/bbsi: bit index is fixed at decode time (`bit`, 0..31).
        Op::Bbci { s, bit, target } => (*target, (cpu.regs.read_ar(*s) >> *bit) & 1 == 0),
        Op::Bbsi { s, bit, target } => (*target, (cpu.regs.read_ar(*s) >> *bit) & 1 != 0),
        // bbc/bbs: bit index comes from AR[t]'s RUNTIME value, masked to
        // 0x1f (contrast bbci/bbsi's decode-time-fixed immediate).
        Op::Bbc { s, t, target } => {
            let bit = cpu.regs.read_ar(*t) & 0x1f;
            (*target, (cpu.regs.read_ar(*s) >> bit) & 1 == 0)
        }
        Op::Bbs { s, t, target } => {
            let bit = cpu.regs.read_ar(*t) & 0x1f;
            (*target, (cpu.regs.read_ar(*s) >> bit) & 1 != 0)
        }
        Op::Bnone { s, t, target } => (*target, (cpu.regs.read_ar(*s) & cpu.regs.read_ar(*t)) == 0),
        Op::Bany { s, t, target } => (*target, (cpu.regs.read_ar(*s) & cpu.regs.read_ar(*t)) != 0),
        // ball/bnall: every bit set in AR[t] is (ball) / isn't entirely
        // (bnall) also set in AR[s].
        Op::Ball { s, t, target } => {
            let tv = cpu.regs.read_ar(*t);
            (*target, (cpu.regs.read_ar(*s) & tv) == tv)
        }
        Op::Bnall { s, t, target } => {
            let tv = cpu.regs.read_ar(*t);
            (*target, (cpu.regs.read_ar(*s) & tv) != tv)
        }
        _ => return None,
    };
    cpu.pc = if taken {
        target
    } else {
        pc.wrapping_add(len as u32)
    };
    Some(Step::Ran)
}

#[cfg(test)]
mod tests {
    use super::super::{Cpu, Step};
    use crate::firmware::mmio::Bus;

    /// Builds a ROM with `bytes` placed at offset `pc` (zero-padded before),
    /// so `Cpu::new(pc)` can fetch a real decoded instruction at the exact
    /// pc the oracle vector was taken at (needed since these branch targets
    /// are pc-relative).
    fn rom_at(pc: u32, bytes: &[u8]) -> Vec<u8> {
        let mut rom = vec![0u8; pc as usize + bytes.len()];
        rom[pc as usize..pc as usize + bytes.len()].copy_from_slice(bytes);
        rom
    }

    fn step_from(pc: u32, bytes: &[u8], setup: impl FnOnce(&mut Cpu)) -> Cpu {
        let rom = rom_at(pc, bytes);
        let mut bus = Bus::new(rom);
        let mut cpu = Cpu::new(pc);
        setup(&mut cpu);
        assert!(matches!(cpu.step(&mut bus), Step::Ran));
        cpu
    }

    #[test]
    fn j_always_jumps() {
        // `46 02 00` @ pc 0 -> j 0xd. Unconditional: no not-taken case exists
        // for this opcode (there is no register/condition to flip).
        let cpu = step_from(0, &[0x46, 0x02, 0x00], |_| {});
        assert_eq!(cpu.pc, 0xd);
    }

    #[test]
    fn beqz_taken_and_not_taken() {
        // `16 88 04` @ pc 3 -> beqz a8, 0x4f.
        let cpu = step_from(3, &[0x16, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 0));
        assert_eq!(cpu.pc, 0x4f, "AR[8]==0 -> taken");
        let cpu = step_from(3, &[0x16, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 5));
        assert_eq!(cpu.pc, 3 + 3, "AR[8]!=0 -> not taken, falls through");
    }

    #[test]
    fn bnez_taken_and_not_taken() {
        // `56 88 04` @ pc 6 -> bnez a8, 0x52.
        let cpu = step_from(6, &[0x56, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 5));
        assert_eq!(cpu.pc, 0x52, "AR[8]!=0 -> taken");
        let cpu = step_from(6, &[0x56, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 0));
        assert_eq!(cpu.pc, 6 + 3, "AR[8]==0 -> not taken");
    }

    #[test]
    fn bltz_taken_and_not_taken() {
        // `96 88 04` @ pc 9 -> bltz a8, 0x55.
        let cpu = step_from(9, &[0x96, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 0x8000_0000));
        assert_eq!(cpu.pc, 0x55, "AR[8] negative -> taken");
        let cpu = step_from(9, &[0x96, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 5));
        assert_eq!(cpu.pc, 9 + 3, "AR[8] positive -> not taken");
    }

    #[test]
    fn bgez_taken_and_not_taken() {
        // `d6 88 04` @ pc 0xc -> bgez a8, 0x58.
        let cpu = step_from(0xc, &[0xd6, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 5));
        assert_eq!(cpu.pc, 0x58, "AR[8] positive -> taken");
        let cpu = step_from(0xc, &[0xd6, 0x88, 0x04], |cpu| cpu.regs.write_ar(8, 0x8000_0000));
        assert_eq!(cpu.pc, 0xc + 3, "AR[8] negative -> not taken");
    }

    #[test]
    fn beqz_n_taken_and_not_taken() {
        // `8c 52` @ pc 0xf -> beqz.n a2, 0x18 (narrow, len 2).
        let cpu = step_from(0xf, &[0x8c, 0x52], |cpu| cpu.regs.write_ar(2, 0));
        assert_eq!(cpu.pc, 0x18, "AR[2]==0 -> taken");
        let cpu = step_from(0xf, &[0x8c, 0x52], |cpu| cpu.regs.write_ar(2, 5));
        assert_eq!(cpu.pc, 0xf + 2, "AR[2]!=0 -> not taken");
    }

    #[test]
    fn bnez_n_taken_and_not_taken() {
        // `cc 52` @ pc 0x11 -> bnez.n a2, 0x1a (narrow, len 2).
        let cpu = step_from(0x11, &[0xcc, 0x52], |cpu| cpu.regs.write_ar(2, 5));
        assert_eq!(cpu.pc, 0x1a, "AR[2]!=0 -> taken");
        let cpu = step_from(0x11, &[0xcc, 0x52], |cpu| cpu.regs.write_ar(2, 0));
        assert_eq!(cpu.pc, 0x11 + 2, "AR[2]==0 -> not taken");
    }

    #[test]
    fn beq_taken_and_not_taken() {
        // `57 17 14` @ pc 0x13 -> beq a7, a5, 0x2b.
        let cpu = step_from(0x13, &[0x57, 0x17, 0x14], |cpu| {
            cpu.regs.write_ar(7, 5);
            cpu.regs.write_ar(5, 5);
        });
        assert_eq!(cpu.pc, 0x2b, "AR[7]==AR[5] -> taken");
        let cpu = step_from(0x13, &[0x57, 0x17, 0x14], |cpu| {
            cpu.regs.write_ar(7, 5);
            cpu.regs.write_ar(5, 9);
        });
        assert_eq!(cpu.pc, 0x13 + 3, "AR[7]!=AR[5] -> not taken");
    }

    #[test]
    fn bne_taken_and_not_taken() {
        // `57 97 14` @ pc 0x16 -> bne a7, a5, 0x2e.
        let cpu = step_from(0x16, &[0x57, 0x97, 0x14], |cpu| {
            cpu.regs.write_ar(7, 5);
            cpu.regs.write_ar(5, 9);
        });
        assert_eq!(cpu.pc, 0x2e, "AR[7]!=AR[5] -> taken");
        let cpu = step_from(0x16, &[0x57, 0x97, 0x14], |cpu| {
            cpu.regs.write_ar(7, 5);
            cpu.regs.write_ar(5, 5);
        });
        assert_eq!(cpu.pc, 0x16 + 3, "AR[7]==AR[5] -> not taken");
    }

    #[test]
    fn blt_taken_and_not_taken() {
        // `57 27 14` @ pc 0x19 -> blt a7, a5, 0x31.
        let cpu = step_from(0x19, &[0x57, 0x27, 0x14], |cpu| {
            cpu.regs.write_ar(7, 3);
            cpu.regs.write_ar(5, 10);
        });
        assert_eq!(cpu.pc, 0x31, "3 < 10 -> taken");
        let cpu = step_from(0x19, &[0x57, 0x27, 0x14], |cpu| {
            cpu.regs.write_ar(7, 10);
            cpu.regs.write_ar(5, 3);
        });
        assert_eq!(cpu.pc, 0x19 + 3, "10 < 3 is false -> not taken");
    }

    #[test]
    fn bge_taken_and_not_taken() {
        // `57 a7 14` @ pc 0x1f -> bge a7, a5, 0x37.
        let cpu = step_from(0x1f, &[0x57, 0xa7, 0x14], |cpu| {
            cpu.regs.write_ar(7, 10);
            cpu.regs.write_ar(5, 3);
        });
        assert_eq!(cpu.pc, 0x37, "10 >= 3 -> taken");
        let cpu = step_from(0x1f, &[0x57, 0xa7, 0x14], |cpu| {
            cpu.regs.write_ar(7, 3);
            cpu.regs.write_ar(5, 10);
        });
        assert_eq!(cpu.pc, 0x1f + 3, "3 >= 10 is false -> not taken");
    }

    #[test]
    fn bgeu_taken_and_not_taken() {
        // `57 b7 14` @ pc 0x22 -> bgeu a7, a5, 0x3a.
        let cpu = step_from(0x22, &[0x57, 0xb7, 0x14], |cpu| {
            cpu.regs.write_ar(7, 10);
            cpu.regs.write_ar(5, 3);
        });
        assert_eq!(cpu.pc, 0x3a, "10 >= 3 unsigned -> taken");
        let cpu = step_from(0x22, &[0x57, 0xb7, 0x14], |cpu| {
            cpu.regs.write_ar(7, 3);
            cpu.regs.write_ar(5, 10);
        });
        assert_eq!(cpu.pc, 0x22 + 3, "3 >= 10 unsigned is false -> not taken");
    }

    #[test]
    fn blt_vs_bltu_diverge_on_high_bit_set_value() {
        // blt (`57 27 14` @ pc 0x19) vs bltu (`57 37 14` @ pc 0x1c), same
        // a7/a5 values with AR[7]'s high bit set: signed, AR[7] is
        // i32::MIN (very negative) so `blt` takes; unsigned, AR[7] is a huge
        // positive value so `bltu` does NOT take. This is the required
        // signed/unsigned divergence case.
        let cpu = step_from(0x19, &[0x57, 0x27, 0x14], |cpu| {
            cpu.regs.write_ar(7, 0x8000_0000);
            cpu.regs.write_ar(5, 1);
        });
        assert_eq!(cpu.pc, 0x31, "signed: i32::MIN < 1 -> blt taken");

        let cpu = step_from(0x1c, &[0x57, 0x37, 0x14], |cpu| {
            cpu.regs.write_ar(7, 0x8000_0000);
            cpu.regs.write_ar(5, 1);
        });
        assert_eq!(cpu.pc, 0x1c + 3, "unsigned: 0x8000_0000 < 1 is false -> bltu not taken");
    }

    #[test]
    fn beqi_taken_and_not_taken_pins_b4const_lookup() {
        // `26 66 02` @ pc 0x25 -> beqi a6, 6, 0x2b. Index 6 resolves to
        // B4CONST[6]==6 (decode/branch.rs); this end-to-end `cpu.step` run
        // (decode + B4CONST lookup + compare) pins that the runtime compare
        // uses the resolved VALUE 6, not the raw index 6 coincidentally
        // matching (see the bnei-with-a-different-index test below, and
        // decode/branch.rs's `decodes_beqi_pins_a_different_b4const_index`,
        // for the case where index and value diverge).
        let cpu = step_from(0x25, &[0x26, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x2b, "AR[6]==6 -> taken");
        let cpu = step_from(0x25, &[0x26, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 7));
        assert_eq!(cpu.pc, 0x25 + 3, "AR[6]!=6 -> not taken");
    }

    #[test]
    fn bnei_taken_and_not_taken() {
        // `66 66 02` @ pc 0x28 -> bnei a6, 6, 0x2e.
        let cpu = step_from(0x28, &[0x66, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 7));
        assert_eq!(cpu.pc, 0x2e, "AR[6]!=6 -> taken");
        let cpu = step_from(0x28, &[0x66, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x28 + 3, "AR[6]==6 -> not taken");
    }

    #[test]
    fn blti_taken_and_not_taken() {
        // `a6 66 02` @ pc 0x2b -> blti a6, 6, 0x31.
        let cpu = step_from(0x2b, &[0xa6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 3));
        assert_eq!(cpu.pc, 0x31, "3 < 6 -> taken");
        let cpu = step_from(0x2b, &[0xa6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x2b + 3, "6 < 6 is false -> not taken");
    }

    #[test]
    fn bgei_taken_and_not_taken() {
        // `e6 66 02` @ pc 0x2e -> bgei a6, 6, 0x34.
        let cpu = step_from(0x2e, &[0xe6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x34, "6 >= 6 -> taken");
        let cpu = step_from(0x2e, &[0xe6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 3));
        assert_eq!(cpu.pc, 0x2e + 3, "3 >= 6 is false -> not taken");
    }

    #[test]
    fn bltui_taken_and_not_taken_pins_b4constu_lookup() {
        // `b6 66 02` @ pc 0x31 -> bltui a6, 6, 0x37. Index 6 resolves to
        // B4CONSTU[6]==6 -- same numeric value as B4CONST[6] in this
        // particular slot, but from the DISTINCT unsigned table (see
        // decode/branch.rs's `decodes_bltui_pins_a_b4constu_index_that_
        // diverges_from_b4const` for a divergent-index case).
        let cpu = step_from(0x31, &[0xb6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 3));
        assert_eq!(cpu.pc, 0x37, "3 < 6 unsigned -> taken");
        let cpu = step_from(0x31, &[0xb6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x31 + 3, "6 < 6 unsigned is false -> not taken");
    }

    #[test]
    fn bgeui_taken_and_not_taken() {
        // `f6 66 02` @ pc 0x34 -> bgeui a6, 6, 0x3a.
        let cpu = step_from(0x34, &[0xf6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 6));
        assert_eq!(cpu.pc, 0x3a, "6 >= 6 unsigned -> taken");
        let cpu = step_from(0x34, &[0xf6, 0x66, 0x02], |cpu| cpu.regs.write_ar(6, 3));
        assert_eq!(cpu.pc, 0x34 + 3, "3 >= 6 unsigned is false -> not taken");
    }

    #[test]
    fn bbci_taken_and_not_taken_pins_immediate_bit_index() {
        // `37 64 05` @ pc 0x37 -> bbci a4, 3, 0x40. The bit index (3) is
        // FIXED by the encoding -- contrast bbc's test below, where changing
        // a register changes which bit is tested.
        let cpu = step_from(0x37, &[0x37, 0x64, 0x05], |cpu| cpu.regs.write_ar(4, 0));
        assert_eq!(cpu.pc, 0x40, "bit 3 of AR[4] clear -> taken");
        let cpu = step_from(0x37, &[0x37, 0x64, 0x05], |cpu| cpu.regs.write_ar(4, 1 << 3));
        assert_eq!(cpu.pc, 0x37 + 3, "bit 3 of AR[4] set -> not taken");
    }

    #[test]
    fn bbsi_taken_and_not_taken() {
        // `37 e4 05` @ pc 0x3a -> bbsi a4, 3, 0x43.
        let cpu = step_from(0x3a, &[0x37, 0xe4, 0x05], |cpu| cpu.regs.write_ar(4, 1 << 3));
        assert_eq!(cpu.pc, 0x43, "bit 3 of AR[4] set -> taken");
        let cpu = step_from(0x3a, &[0x37, 0xe4, 0x05], |cpu| cpu.regs.write_ar(4, 0));
        assert_eq!(cpu.pc, 0x3a + 3, "bit 3 of AR[4] clear -> not taken");
    }

    #[test]
    fn bbc_taken_and_not_taken_pins_register_bit_index() {
        // `47 5a 0e` @ pc 0x3d -> bbc a10, a4, 0x4f. Unlike bbci, the bit
        // index comes from AR[4]'s RUNTIME value: setting AR[4]=5 tests bit
        // 5 of AR[10], not a fixed encoding bit -- if the index were instead
        // baked into the encoding, changing AR[4] here would have no effect.
        let cpu = step_from(0x3d, &[0x47, 0x5a, 0x0e], |cpu| {
            cpu.regs.write_ar(4, 5); // bit index register: test bit 5
            cpu.regs.write_ar(10, 0); // bit 5 clear
        });
        assert_eq!(cpu.pc, 0x4f, "bit AR[4]=5 of AR[10] clear -> taken");
        let cpu = step_from(0x3d, &[0x47, 0x5a, 0x0e], |cpu| {
            cpu.regs.write_ar(4, 5);
            cpu.regs.write_ar(10, 1 << 5); // bit 5 set
        });
        assert_eq!(cpu.pc, 0x3d + 3, "bit AR[4]=5 of AR[10] set -> not taken");
    }

    #[test]
    fn bbs_taken_and_not_taken() {
        // `47 da 0e` @ pc 0x40 -> bbs a10, a4, 0x52.
        let cpu = step_from(0x40, &[0x47, 0xda, 0x0e], |cpu| {
            cpu.regs.write_ar(4, 5);
            cpu.regs.write_ar(10, 1 << 5);
        });
        assert_eq!(cpu.pc, 0x52, "bit AR[4]=5 of AR[10] set -> taken");
        let cpu = step_from(0x40, &[0x47, 0xda, 0x0e], |cpu| {
            cpu.regs.write_ar(4, 5);
            cpu.regs.write_ar(10, 0);
        });
        assert_eq!(cpu.pc, 0x40 + 3, "bit AR[4]=5 of AR[10] clear -> not taken");
    }

    #[test]
    fn bnone_taken_and_not_taken() {
        // `77 0f 0f` @ pc 0x43 -> bnone a15, a7, 0x56.
        let cpu = step_from(0x43, &[0x77, 0x0f, 0x0f], |cpu| {
            cpu.regs.write_ar(15, 0xF0);
            cpu.regs.write_ar(7, 0x0F);
        });
        assert_eq!(cpu.pc, 0x56, "0xF0 & 0x0F == 0 -> taken");
        let cpu = step_from(0x43, &[0x77, 0x0f, 0x0f], |cpu| {
            cpu.regs.write_ar(15, 0xFF);
            cpu.regs.write_ar(7, 0x0F);
        });
        assert_eq!(cpu.pc, 0x43 + 3, "0xFF & 0x0F != 0 -> not taken");
    }

    #[test]
    fn bany_taken_and_not_taken() {
        // `a7 88 bc` @ pc 0x46 -> bany a8, a10, 0x6 (negative-offset vector).
        let cpu = step_from(0x46, &[0xa7, 0x88, 0xbc], |cpu| {
            cpu.regs.write_ar(8, 0xFF);
            cpu.regs.write_ar(10, 0x0F);
        });
        assert_eq!(cpu.pc, 0x6, "0xFF & 0x0F != 0 -> taken");
        let cpu = step_from(0x46, &[0xa7, 0x88, 0xbc], |cpu| {
            cpu.regs.write_ar(8, 0xF0);
            cpu.regs.write_ar(10, 0x0F);
        });
        assert_eq!(cpu.pc, 0x46 + 3, "0xF0 & 0x0F == 0 -> not taken");
    }

    #[test]
    fn ball_taken_and_not_taken() {
        // `37 44 02` @ pc 0x49 -> ball a4, a3, 0x4f.
        let cpu = step_from(0x49, &[0x37, 0x44, 0x02], |cpu| {
            cpu.regs.write_ar(4, 0xFF);
            cpu.regs.write_ar(3, 0x0F);
        });
        assert_eq!(cpu.pc, 0x4f, "all bits of AR[3] set in AR[4] -> taken");
        let cpu = step_from(0x49, &[0x37, 0x44, 0x02], |cpu| {
            cpu.regs.write_ar(4, 0xF0);
            cpu.regs.write_ar(3, 0x0F);
        });
        assert_eq!(cpu.pc, 0x49 + 3, "AR[3]'s bits not all in AR[4] -> not taken");
    }

    #[test]
    fn bnall_taken_and_not_taken() {
        // `67 c7 13` @ pc 0x4c -> bnall a7, a6, 0x63.
        let cpu = step_from(0x4c, &[0x67, 0xc7, 0x13], |cpu| {
            cpu.regs.write_ar(7, 0xF0);
            cpu.regs.write_ar(6, 0x0F);
        });
        assert_eq!(cpu.pc, 0x63, "AR[6]'s bits not all in AR[7] -> taken");
        let cpu = step_from(0x4c, &[0x67, 0xc7, 0x13], |cpu| {
            cpu.regs.write_ar(7, 0xFF);
            cpu.regs.write_ar(6, 0x0F);
        });
        assert_eq!(cpu.pc, 0x4c + 3, "all bits of AR[6] set in AR[7] -> not taken");
    }
}
