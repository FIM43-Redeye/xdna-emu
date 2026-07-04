//! Memory-load decode: `l32i.n`, `l32i`, `l32r`.

use super::Op;

/// Narrow (2-byte) memory ops: `l32i.n at,as,imm4*4`. `None` if `op0` isn't
/// this category's narrow selector, so `decode()` tries the next category.
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        // l32i.n at,as,imm4*4 -- verified: `48 45` -> l32i.n a4,a5,0x10
        0x8 => Some(Op::L32iN { t: n1, s: n2, imm: (n3 as u32) * 4 }),
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
/// `r==0x2`. Verified: `52 22 0a` -> l32i a5,a2,40. `None` for other `r`
/// values, so `decode()` tries the next category (`arith`'s `movi`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    match r {
        0x2 => Some(Op::L32i { t, s, imm: (b2 as u32) * 4 }),
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
