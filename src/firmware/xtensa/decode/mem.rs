//! Memory load/store decode: `l32i.n`, `l32i`, `l32r`, plus the
//! sibling widths `s32i.n`, `l8ui`, `s8i`, `s32i`, `l16ui`, `s16i`, `l16si`,
//! `s32ri`, and the FP load/store `lsi`/`ssi` (LSCI format).

use super::Op;

/// Narrow (2-byte) memory ops: `l32i.n at,as,imm4*4`. `None` if `op0` isn't
/// this category's narrow selector, so `decode()` tries the next category.
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        // l32i.n at,as,imm4*4 -- example: `78 b2` -> l32i.n a7,a2,44
        0x8 => Some(Op::L32iN { t: n1, s: n2, imm: (n3 as u32) * 4 }),
        // s32i.n at,as,imm4*4 -- sibling of l32i.n (same imm4*4
        // pre-scaling), narrow op0=0x9. Example: `39 9a` -> s32i.n a3,a10,0x24.
        0x9 => Some(Op::S32iN { t: n1, s: n2, imm: (n3 as u32) * 4 }),
        _ => None,
    }
}

/// RI16 format: `l32r at, <literal>`. imm16 = word>>8 is a negative WORD
/// offset -- sign-extend it to 32 bits FIRST (OR with 0xFFFF0000), THEN <<2
/// to convert words->bytes: `((pc+3)&~3) + ((0xFFFF0000 | imm16) << 2)`.
/// Order matters: OR-then-shift, not shift-then-OR -- verified against
/// xtensa-lx106-elf-objdump, e.g. `51 c0 ff` at pc 0x2000 -> l32r a5,
/// 0x1f00. op0=0x1 is exclusively
/// this format, so this always returns `Some`.
pub(super) fn decode_ri16(t: u8, word: u32, pc: u32) -> Option<Op> {
    let imm16 = word >> 8;
    let base = pc.wrapping_add(3) & !3u32;
    let target = base.wrapping_add((0xFFFF_0000u32 | imm16) << 2);
    Some(Op::L32r { t, target })
}

/// RRI8 format (LSAI group): `r` selects the sub-op; `l32i at,as,imm8*4` is
/// `r==0x2`. Example: `92 26 19` -> l32i a9,a6,100. The sibling load/store
/// widths are each confirmed against their own vector (`decode/mem.rs`
/// tests): `l8ui`/`s8i` (scale 0,
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
        0xE => Some(Op::S32c1i { t, s, imm: (b2 as u32) * 4 }),
        0xF => Some(Op::S32ri { t, s, imm: (b2 as u32) * 4 }),
        _ => None,
    }
}

/// LSCI format (op0=3): floating-point load/store single. `ft` = FP register
/// (the `t` field), `s` = address AR (the `s` field), `r` selects the sub-op
/// (0 = `lsi` load, 4 = `ssi` store), `imm8` = `b2` scaled *4 (word offset).
/// Only the immediate load/store are modeled -- the update forms (`lsiu`/`ssiu`,
/// r=8/0xC) and indexed forms are not modeled. Derived
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
/// address register `as`. Only the window spill/fill handlers use these. Verified
/// (binutils libisa, this core's config): `00 c5 49` -> s32e a0,a5,-16;
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
        // l32i.n a7,a2,44: op0=8; t=7 -> byte0 0x78; imm4=11 (11*4=44), s=2 ->
        // byte1 0xb2.
        let d = decode(&[0x78, 0xb2], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::L32iN { t: 7, s: 2, imm: 44 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32i() {
        // l32i a9,a6,100: RRI8 r=2; imm8=25 (25*4=100) -> `92 26 19`.
        let d = decode(&[0x92, 0x26, 0x19], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32i { t: 9, s: 6, imm: 100 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32r() {
        // l32r a5 at pc 0x2000 reaching back to 0x1f00: base = (0x2000+3)&~3 =
        // 0x2000; offset = 0x1f00-0x2000 = -0x100 = -64 words; imm16 =
        // 0xffc0; word = (imm16<<8)|(t<<4)|op0=1 = 0xffc051 -> `51 c0 ff`.
        let d = decode(&[0x51, 0xc0, 0xff], 0x2000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32r { t: 5, target: 0x1f00 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32i_n() {
        // `39 9a` -> s32i.n a3,a10,0x24 (op0=9; imm4=9, 9*4=0x24).
        let d = decode(&[0x39, 0x9a], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::S32iN { t: 3, s: 10, imm: 0x24 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_lsi_ssi() {
        // LSCI format (op0=3): FP load/store single. ft=n1, as=n2, r=n3
        // (0=lsi, 4=ssi), imm8=b2 scaled *4.
        // `63 09 10` -> lsi f6,a9,0x40: ft=6, as=9, r=0, imm8=0x10 -> off
        // 0x10*4=0x40.
        let d = decode(&[0x63, 0x09, 0x10], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Lsi { ft: 6, s: 9, imm: 0x40 }), "got {:?}", d.op);
        // `c3 47 0b` -> ssi f12,a7,0x2c: ft=12, as=7, r=4, imm8=11 -> off 0x2c.
        let d = decode(&[0xc3, 0x47, 0x0b], 0x1000);
        assert!(matches!(d.op, Op::Ssi { ft: 12, s: 7, imm: 0x2c }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32e_l32e() {
        // Offset = (r - 16) * 4, with r the high nibble of byte1.
        // `40 d9 49` -> s32e a4,a9,-12 (r=0xd); `40 d9 09` -> l32e a4,a9,-12.
        let d = decode(&[0x40, 0xd9, 0x49], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32e { t: 4, s: 9, imm } if imm == (-12i32) as u32), "got {:?}", d.op);
        let d = decode(&[0x40, 0xd9, 0x09], 0x1000);
        assert!(matches!(d.op, Op::L32e { t: 4, s: 9, imm } if imm == (-12i32) as u32), "got {:?}", d.op);
        // A different offset/regs: `60 ea 49` -> s32e a6,a10,-8 (r=0xe -> -8).
        let d = decode(&[0x60, 0xea, 0x49], 0x1000);
        assert!(matches!(d.op, Op::S32e { t: 6, s: 10, imm } if imm == (-8i32) as u32), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l8ui() {
        // `62 03 1f` -> l8ui a6,a3,0x1f (RRI8 r=0).
        let d = decode(&[0x62, 0x03, 0x1f], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L8ui { t: 6, s: 3, imm: 0x1f }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s8i() {
        // `b2 42 41` -> s8i a11,a2,0x41 (RRI8 r=4).
        let d = decode(&[0xb2, 0x42, 0x41], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S8i { t: 11, s: 2, imm: 0x41 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32i() {
        // `72 64 12` -> s32i a7,a4,0x48 (RRI8 r=6; imm8=0x12, 0x12*4=0x48).
        let d = decode(&[0x72, 0x64, 0x12], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32i { t: 7, s: 4, imm: 0x48 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32c1i() {
        // `42 e6 b4` -> s32c1i a4,a6,0x2d0 (RRI8 r==0xE, imm8=0xb4*4=0x2d0).
        // SCOMPARE1 = SR 12.
        let d = decode(&[0x42, 0xe6, 0xb4], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32c1i { t: 4, s: 6, imm: 0x2d0 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l16ui() {
        // `82 15 0d` -> l16ui a8,a5,0x1a (RRI8 r=1; imm8=13, 13*2=26).
        let d = decode(&[0x82, 0x15, 0x0d], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L16ui { t: 8, s: 5, imm: 26 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s16i() {
        // `a2 53 0b` -> s16i a10,a3,0x16 (RRI8 r=5; imm8=11, 11*2=22).
        let d = decode(&[0xa2, 0x53, 0x0b], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S16i { t: 10, s: 3, imm: 22 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l16si() {
        // `42 9c 15` -> l16si a4,a12,0x2a (RRI8 r=9; imm8=21, 21*2=42).
        let d = decode(&[0x42, 0x9c, 0x15], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L16si { t: 4, s: 12, imm: 42 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_s32ri() {
        // objdump can't decode this opcode (lx106 core lacks the Interrupt
        // option it belongs to) and prints `excw` regardless of the actual
        // bits -- verified via binutils libisa, this core's config instead:
        // `32 f7 71` -> s32ri a3,a7,0x1c4 (imm8=0x71, 0x71*4=0x1c4).
        let d = decode(&[0x32, 0xf7, 0x71], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::S32ri { t: 3, s: 7, imm: 0x1c4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_l32r_with_non_full_sign_extension() {
        // Regression for a shift-then-OR/OR-then-shift bug: this vector's
        // imm16 (0x9000) does NOT have both top bits set, so the two
        // formulas diverge (0x9000 top bits='10'; the earlier `decodes_l32r`
        // vector has imm16=0xffc0, top bits='11', which happens to agree
        // under both formulas and so didn't catch the bug).
        // `a1 00 90` @ pc 0x4000 -> l32r a10: base = 0x4000;
        // OR-then-shift: (0xffff9000 << 2) = 0xfffe4000, target = 0x4000 +
        // 0xfffe4000 = 0xfffe8000 (shift-then-OR would give 0xffff4000 + 0x4000).
        let d = decode(&[0xa1, 0x00, 0x90], 0x4000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::L32r { t: 10, target: 0xfffe8000 }), "got {:?}", d.op);
    }
}
