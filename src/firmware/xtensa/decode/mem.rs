//! Memory load/store decode: `l32i.n`, `l32i`, `l32r`, plus the M2a
//! sibling widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`,
//! `s32ri`, and the M2c iter15 FP load/store `lsi`/`ssi` (LSCI format).

use super::Op;

/// Narrow (2-byte) memory ops: `l32i.n at,as,imm4*4`. `None` if `op0` isn't
/// this category's narrow selector, so `decode()` tries the next category.
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        // l32i.n at,as,imm4*4 -- verified: `48 45` -> l32i.n a4,a5,0x10
        0x8 => Some(Op::L32iN { t: n1, s: n2, imm: (n3 as u32) * 4 }),
        // s32i.n at,as,imm4*4 -- sibling of l32i.n (same imm4*4
        // pre-scaling), narrow op0=0x9. Verified: `69 c7` -> s32i.n
        // a6,a7,0x30.
        0x9 => Some(Op::S32iN { t: n1, s: n2, imm: (n3 as u32) * 4 }),
        _ => None,
    }
}

/// RI16 format: `l32r at, <literal>`. imm16 = word>>8 is a negative WORD
/// offset -- sign-extend it to 32 bits FIRST (OR with 0xFFFF0000), THEN <<2
/// to convert words->bytes: `((pc+3)&~3) + ((0xFFFF0000 | imm16) << 2)`.
/// Order matters: OR-then-shift, not shift-then-OR -- verified against all
/// 2829 l32r instructions in the real firmware (xtensa-lx106-elf-objdump);
/// e.g. `21 bd e7` @ pc 0x33262 -> l32r a2, 0x2d158. op0=0x1 is exclusively
/// this format, so this always returns `Some`.
pub(super) fn decode_ri16(t: u8, word: u32, pc: u32) -> Option<Op> {
    let imm16 = word >> 8;
    let base = pc.wrapping_add(3) & !3u32;
    let target = base.wrapping_add((0xFFFF_0000u32 | imm16) << 2);
    Some(Op::L32r { t, target })
}

/// RRI8 format (LSAI group): `r` selects the sub-op; `l32i at,as,imm8*4` is
/// `r==0x2`. Verified: `52 22 0a` -> l32i a5,a2,40. This task (M2a memory
/// opcodes) adds the sibling load/store widths, each confirmed against its
/// own real-firmware vector (`decode/mem.rs` tests): `l8ui`/`s8i` (scale 0,
/// `r==0x0`/`0x4`), `l16ui`/`s16i`/`l16si` (scale 1, `r==0x1`/`0x5`/`0x9`),
/// `s32i`/`s32ri` (scale 2 like `l32i`, `r==0x6`/`0xF`). `None` for other `r`
/// values, so `decode()` tries the next category (`arith`'s `movi`, `r==0xA`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    match r {
        0x0 => Some(Op::L8ui { t, s, imm: b2 as u32 }),
        0x1 => Some(Op::L16ui { t, s, imm: (b2 as u32) * 2 }),
        0x2 => Some(Op::L32i { t, s, imm: (b2 as u32) * 4 }),
        0x4 => Some(Op::S8i { t, s, imm: b2 as u32 }),
        0x5 => Some(Op::S16i { t, s, imm: (b2 as u32) * 2 }),
        0x6 => Some(Op::S32i { t, s, imm: (b2 as u32) * 4 }),
        0x9 => Some(Op::L16si { t, s, imm: (b2 as u32) * 2 }),
        0xF => Some(Op::S32ri { t, s, imm: (b2 as u32) * 4 }),
        _ => None,
    }
}

/// LSCI format (op0=3): floating-point load/store single. `ft` = FP register
/// (the `t` field), `s` = address AR (the `s` field), `r` selects the sub-op
/// (0 = `lsi` load, 4 = `ssi` store), `imm8` = `b2` scaled *4 (word offset).
/// Only the immediate load/store are modeled -- the update forms (`lsiu`/`ssiu`,
/// r=8/0xC) and indexed forms would be added if a boot path needs them. Derived
/// from xtensa-modules.c's `Opcode_{lsi,ssi}_Slot_inst_encode` templates
/// (`0x3`/`0x4003`, distinguished by the `r` field). `None` for other `r`, so
/// `decode()` falls through to `Op::Unknown`.
pub(super) fn decode_lsci(ft: u8, s: u8, r: u8, b2: u8) -> Option<Op> {
    match r {
        0x0 => Some(Op::Lsi { ft, s, imm: (b2 as u32) * 4 }),
        0x4 => Some(Op::Ssi { ft, s, imm: (b2 as u32) * 4 }),
        _ => None,
    }
}

/// RRR-format windowed-exception memory ops (`op0=0`): `l32e` (`op1=9, op2=0`)
/// and `s32e` (`op1=9, op2=4`). The 4-bit `r` field is a negative word offset
/// `(r - 16) * 4` (range -64..-4); `t` = the data register `at`, `s` = the
/// address register `as`. Only the window spill/fill handlers use these. Byte-
/// verified against the firmware's window vectors: `00 c5 49` -> s32e a0,a5,-16;
/// `00 c5 09` -> l32e a0,a5,-16. `None` for other `(op1,op2)`.
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8) -> Option<Op> {
    if op1 != 0x9 {
        return None;
    }
    let imm = (((r as i32) - 16) * 4) as u32;
    match op2 {
        0x0 => Some(Op::L32e { t, s, imm }),
        0x4 => Some(Op::S32e { t, s, imm }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_l32i_n() {
        let d = decode(&[0x48, 0x45], 0x33259);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::L32iN { t: 4, s: 5, imm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32i() {
        let d = decode(&[0x52, 0x22, 0x0a], 0x3324a);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32i { t: 5, s: 2, imm: 40 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32r() {
        let d = decode(&[0x21, 0xbd, 0xe7], 0x33262);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32r { t: 2, target: 0x2d158 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32i_n() {
        // listing.txt: `69 c7` @0x27b6 -> s32i.n a6,a7,0x30.
        let d = decode(&[0x69, 0xc7], 0x27b6);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::S32iN { t: 6, s: 7, imm: 0x30 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_lsi_ssi() {
        // LSCI format (op0=3): FP load/store single. ft=n1, as=n2, r=n3
        // (0=lsi, 4=ssi), imm8=b2 scaled *4.
        // `a3 0c 63` -> lsi f10,a12,0x18c (the firmware's 0xd830 wall): ft=0xa,
        // as=0xc, r=0, imm8=0x63 -> off 0x63*4=0x18c.
        let d = decode(&[0xa3, 0x0c, 0x63], 0xd830);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Lsi { ft: 10, s: 12, imm: 0x18c }), "got {:?}", d.op);
        // `33 42 04` -> ssi f3,a2,0x10: ft=3, as=2, r=4, imm8=4 -> off 0x10.
        let d = decode(&[0x33, 0x42, 0x04], 0x1000);
        assert!(matches!(d.op, Op::Ssi { ft: 3, s: 2, imm: 0x10 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32e_l32e() {
        // Byte-verified against the firmware's window vectors (0x800 / 0x840):
        // `00 c5 49` -> s32e a0,a5,-16; `00 c5 09` -> l32e a0,a5,-16.
        let d = decode(&[0x00, 0xc5, 0x49], 0x800);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32e { t: 0, s: 5, imm } if imm == (-16i32) as u32), "got {:?}", d.op);
        let d = decode(&[0x00, 0xc5, 0x09], 0x840);
        assert!(matches!(d.op, Op::L32e { t: 0, s: 5, imm } if imm == (-16i32) as u32), "got {:?}", d.op);
        // A different offset/regs: `30 f5 49` -> s32e a3,a5,-4 (r=0xf -> -4).
        let d = decode(&[0x30, 0xf5, 0x49], 0x80c);
        assert!(matches!(d.op, Op::S32e { t: 3, s: 5, imm } if imm == (-4i32) as u32), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l8ui() {
        // listing.txt: `22 02 00` @0x279a -> l8ui a2,a2,0x0.
        let d = decode(&[0x22, 0x02, 0x00], 0x279a);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L8ui { t: 2, s: 2, imm: 0 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s8i() {
        // listing.txt: `82 44 2c` @0x2847 -> s8i a8,a4,0x2c.
        let d = decode(&[0x82, 0x44, 0x2c], 0x2847);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S8i { t: 8, s: 4, imm: 0x2c }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32i() {
        // listing.txt: `22 61 07` @0x2736 -> s32i a2,a1,0x1c.
        let d = decode(&[0x22, 0x61, 0x07], 0x2736);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32i { t: 2, s: 1, imm: 0x1c }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l16ui() {
        // listing.txt: `32 13 02` @0x4089 -> l16ui a3,a3,0x4.
        let d = decode(&[0x32, 0x13, 0x02], 0x4089);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L16ui { t: 3, s: 3, imm: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s16i() {
        // listing.txt: `42 57 02` @0x45bc -> s16i a4,a7,0x4.
        let d = decode(&[0x42, 0x57, 0x02], 0x45bc);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S16i { t: 4, s: 7, imm: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l16si() {
        // listing.txt: `22 92 00` @0x9f65 -> l16si a2,a2,0x0.
        let d = decode(&[0x22, 0x92, 0x00], 0x9f65);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L16si { t: 2, s: 2, imm: 0 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32ri() {
        // objdump can't decode this opcode (lx106 core lacks the Interrupt
        // option it belongs to) and prints `excw` regardless of the actual
        // bits -- verified via the captured Ghidra listing.txt instead: `a2
        // ff 86` @0x73e0 -> s32ri a10,a15,0x218.
        let d = decode(&[0xa2, 0xff, 0x86], 0x73e0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32ri { t: 10, s: 15, imm: 0x218 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32r_with_non_full_sign_extension() {
        // Regression for a shift-then-OR/OR-then-shift bug: this vector's
        // imm16 (0xa2e5) does NOT have both top bits set, so the two
        // formulas diverge (0xa2e5 top bits='10'; the earlier `decodes_l32r`
        // vector has imm16=0xe7bd, top bits='11', which happens to agree
        // under both formulas and so didn't catch the bug). Real firmware
        // instruction, verified via xtensa-lx106-elf-objdump: `a1 e5 a2` @
        // pc 0xe0 -> l32r a10, 0xfffe8c74.
        let d = decode(&[0xa1, 0xe5, 0xa2], 0xe0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32r { t: 10, target: 0xfffe8c74 }), "got {:?}", d.op);
    }
}
