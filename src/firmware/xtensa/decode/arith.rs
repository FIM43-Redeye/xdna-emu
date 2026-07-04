//! Register/immediate arithmetic decode: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`.

use super::{sign_extend, Op};

/// Narrow (2-byte) arithmetic ops: `movi.n` (op0=0xC, n1<=0x7) and `mov.n`
/// (op0=0xD, r==0). `None` otherwise, so `decode()` tries the next category.
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        // op0=0xC narrow-immediate family: n1 (byte0 high nibble)
        // selects MOVI.N (0x0..=0x7) vs BEQZ.N/BNEZ.N (0x8..=0xF, not
        // implemented). The 7-bit signed field is (n1<<4)|n3, wrapping
        // past 96 -- verified by sweeping xtensa-lx106-elf-objdump across
        // the full n1/n3 range (0x60 -> -32 ... 0x7f -> -1).
        0xC if n1 <= 0x7 => {
            let raw = ((n1 as u32) << 4) | (n3 as u32);
            let imm = if raw < 96 { raw as i32 } else { raw as i32 - 128 };
            Some(Op::MoviN { t: n2, imm })
        }
        // op0=0xD family: r (byte1 high nibble) == 0 selects MOV.N.
        0xD if n3 == 0 => Some(Op::MovN { t: n1, s: n2 }),
        _ => None,
    }
}

/// RRI8 format (LSAI group): `movi at,imm12` is `r==0xA` -- here "s" (n2) is
/// not a register, it's the immediate's high 4 bits. Verified: `d2 a0 ac` ->
/// movi a13,172. `None` for other `r` values, so `decode()` tries the next
/// category (`mem`'s `l32i`, tried first, or falls to `Op::Unknown`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    match r {
        0xA => {
            let raw12 = ((s as u32) << 8) | (b2 as u32);
            Some(Op::Movi { t, imm: sign_extend(raw12, 12) })
        }
        _ => None,
    }
}

/// RRR format: `or ar,as,at` (op1=0,op2=2) and `extui art,ars,shiftimm,
/// maskimm` (op1=4, any op2 -- op2 IS data here, the mask-width-1). `None`
/// for anything else in the RRR format, so `decode()` tries `system` then
/// `control`.
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, _word: u32) -> Option<Op> {
    match (op1, op2) {
        // or ar,as,at -- verified: `20 a2 20` -> or a10,a2,a2
        (0x0, 0x2) => Some(Op::Or { r, s, t }),
        // extui art,ars,shiftimm,maskimm -- op1==4 selects EXTUI regardless
        // of op2 (op2 IS data here, not a further selector). Register fields
        // follow the SAME convention as `or` above: r = dest, t = src; s =
        // shiftimm (0-15); op2 = maskimm-1 (1-16). Verified: `30 30 f4` ->
        // extui a3,a3,0,16 (dest==src in this vector, so it alone can't
        // disambiguate which nibble is which -- confirmed separately by
        // sweeping non-aliased registers against objdump: n1=1,n3=7 ->
        // `extui a7,a1,...`, i.e. r is always the first/dest operand).
        (0x4, maskimm_m1) => Some(Op::Extui { r, t, shiftimm: s, maskimm: maskimm_m1 + 1 }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_mov_n() {
        let d = decode(&[0xbd, 0x03], 0x33262);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::MovN { t: 11, s: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi_n() {
        let d = decode(&[0x0c, 0x52], 0x33278);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::MoviN { t: 2, imm: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi() {
        let d = decode(&[0xd2, 0xa0, 0xac], 0x33270);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Movi { t: 13, imm: 172 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_or() {
        let d = decode(&[0x20, 0xa2, 0x20], 0x33256);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Or { r: 10, s: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_extui() {
        let d = decode(&[0x30, 0x30, 0xf4], 0x33280);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Extui { r: 3, t: 3, shiftimm: 0, maskimm: 16 }), "got {:?}", d.op);
    }
}
