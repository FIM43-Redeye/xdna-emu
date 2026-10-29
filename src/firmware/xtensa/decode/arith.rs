//! Register/immediate arithmetic decode: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`, plus the integer/logical/conditional-move/min-max family
//! (`add`/`add.n`/`addi`/`addi.n`/`addmi`/`sub`/`addx2`/`addx4`/`addx8`/
//! `subx8`/`and`/`xor`/`neg`/`abs`/`moveqz`/`movnez`/`min`/`minu`/`maxu`).

use super::{sign_extend, Op};

/// Narrow (2-byte) arithmetic ops: `movi.n` (op0=0xC, n1<=0x7), `mov.n`
/// (op0=0xD, r==0), `add.n` (op0=0xA), `addi.n` (op0=0xB). `None` otherwise,
/// so `decode()` tries the next category.
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
        // add.n ar,as,at -- op0=0xA, same nibble positions as wide RRR
        // (r=n3 dest, s=n2, t=n1). Example: `3a 6b` -> add.n a6,a11,a3
        // (all three distinct, unambiguously pinning the mapping).
        0xA => Some(Op::AddN { r: n3, s: n2, t: n1 }),
        // addi.n at,as,imm -- op0=0xB. Confirmed by sweeping each nibble
        // independently against objdump (holding the others at a
        // non-aliasing value): byte1 high nibble (n3) = dest `t`, byte1 low
        // nibble (n2) = src `s`, byte0 high nibble (n1) = the immediate
        // selector -- a DIFFERENT nibble->field mapping than mov.n (a
        // different narrow format sharing this op0 range). The immediate
        // encoding has no representation for 0: raw nibble 0 means -1, raw
        // 1-15 mean themselves. Example: `4b 53` -> addi.n a5,a3,4 (raw nibble 4); raw 0 -> -1.
        0xB => {
            let imm = if n1 == 0 { -1 } else { n1 as i32 };
            Some(Op::AddiN { t: n3, s: n2, imm })
        }
        _ => None,
    }
}

/// RRI8 format (LSAI group): `movi at,imm12` is `r==0xA` -- here "s" (n2) is
/// not a register, it's the immediate's high 4 bits. Example: `52 ae d4` -> movi a5,-300. `addi at,as,imm8` is `r==0xC`; `addmi at,as,imm8` is
/// `r==0xD` (`imm` stored already `sign_extend8(imm8)<<8`, the final value
/// to add). `None` for other `r` values, so `decode()` tries the next
/// category (`mem`'s `l32i`, tried first, or falls to `Op::Unknown`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    match r {
        0xA => {
            let raw12 = ((s as u32) << 8) | (b2 as u32);
            Some(Op::Movi { t, imm: sign_extend(raw12, 12) })
        }
        // addi at,as,imm8 -- example: `92 c4 15` -> addi a9,a4,0x15 (21).
        0xC => Some(Op::Addi { t, s, imm: sign_extend(b2 as u32, 8) }),
        // addmi at,as,imm8 -- example: `62 d3 03` -> addmi a6,a3,0x300
        // (raw byte 0x03 -> sign_extend8=3, <<8 = 0x300).
        0xD => Some(Op::Addmi { t, s, imm: sign_extend(b2 as u32, 8) << 8 }),
        _ => None,
    }
}

/// RRR format: `or ar,as,at` (op1=0,op2=2), `extui art,ars,shiftimm,
/// maskimm` (op1=4, any op2 -- op2 IS data here, the mask-width-1), plus the
/// Integer/logical/cmov/minmax group (op1=0 sub-ops 1/3/6/8/9/0xA/0xB/
/// 0xC/0xF; op1=3 sub-ops 4/6/7/8/9). `None` for anything else in the RRR
/// format, so `decode()` tries `system` then `control`.
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, _word: u32) -> Option<Op> {
    match (op1, op2) {
        // and ar,as,at -- example: `80 b5 10` -> and a11,a5,a8
        (0x0, 0x1) => Some(Op::And { r, s, t }),
        // or ar,as,at -- example: `d0 69 20` -> or a6,a9,a13
        (0x0, 0x2) => Some(Op::Or { r, s, t }),
        // xor ar,as,at -- example: `70 c3 30` -> xor a12,a3,a7
        (0x0, 0x3) => Some(Op::Xor { r, s, t }),
        // neg/abs share op1=0,op2=6; `s` is a fixed selector nibble here
        // (not a register) -- s==0 selects neg, s==1 selects abs. Examples: `40 90 60` -> neg a9,a4 (s=0); `d0 51 60` -> abs a5,a13 (s=1).
        (0x0, 0x6) if s == 0 => Some(Op::Neg { r, t }),
        (0x0, 0x6) if s == 1 => Some(Op::Abs { r, t }),
        // add ar,as,at -- example: `c0 59 80` -> add a5,a9,a12
        (0x0, 0x8) => Some(Op::Add { r, s, t }),
        // addx2 ar,as,at = (AR[s]<<1)+AR[t] -- example: `60 74 90` -> addx2 a7,a4,a6
        (0x0, 0x9) => Some(Op::Addx2 { r, s, t }),
        // addx4 ar,as,at = (AR[s]<<2)+AR[t] -- example: `50 38 a0` -> addx4 a3,a8,a5
        (0x0, 0xA) => Some(Op::Addx4 { r, s, t }),
        // addx8 ar,as,at = (AR[s]<<3)+AR[t] -- example: `20 a7 b0` -> addx8 a10,a7,a2
        (0x0, 0xB) => Some(Op::Addx8 { r, s, t }),
        // sub ar,as,at -- example: `a0 63 c0` -> sub a6,a3,a10
        (0x0, 0xC) => Some(Op::Sub { r, s, t }),
        // subx8 ar,as,at = (AR[s]<<3)-AR[t] -- example: `90 4c f0` -> subx8 a4,a12,a9. (subx2/subx4 are op2=0xD/0xE.)
        (0x0, 0xF) => Some(Op::Subx8 { r, s, t }),
        // min ar,as,at: signed min -- example: `90 c3 43` -> min a12,a3,a9 (objdump prints excw). (max,
        // signed, is op2=5.)
        (0x3, 0x4) => Some(Op::Min { r, s, t }),
        // minu ar,as,at: unsigned min -- example: `70 4b 63` -> minu a4,a11,a7
        (0x3, 0x6) => Some(Op::Minu { r, s, t }),
        // maxu ar,as,at: unsigned max -- example: `a0 d6 73` -> maxu a13,a6,a10
        (0x3, 0x7) => Some(Op::Maxu { r, s, t }),
        // moveqz ar,as,at: if AR[t]==0 { AR[r]=AR[s] } -- example: `b0 85 83` -> moveqz a8,a5,a11
        (0x3, 0x8) => Some(Op::Moveqz { r, s, t }),
        // movnez ar,as,at: if AR[t]!=0 { AR[r]=AR[s] } -- example: `60 3c 93` -> movnez a3,a12,a6
        (0x3, 0x9) => Some(Op::Movnez { r, s, t }),
        // Shift-immediate/shift-register/funnel-shift family, RRR op1==1:
        // op2 selects the specific op. Field roles verified by an
        // exhaustive objdump sweep -- single-vector examples alias too many
        // operand nibbles to pin the roles alone. See each `Op` variant's
        // doc for the per-op field layout.
        //
        // slli ar,as,sa (sa 1..31): imm5 = ((op2&1)<<4)|t, sa = 32-imm5.
        // Example: `40 29 11` -> slli a2,a9,12 (imm5=20, sa=12). imm5==0
        // (op2&1==0, t==0) would give sa=32 -- outside the architectural
        // 1..31 range and reserved (xtensa-modules.c's Operand_msalp32_decode
        // formula produces it, but ISA RM SLLI, p.525, leaves a raw sa field
        // of 0 undefined), so that one raw value falls through to Op::Unknown
        // instead of a bogus Slli.
        (0x1, 0x0) | (0x1, 0x1) => {
            let imm5 = ((op2 & 1) << 4) | t;
            if imm5 == 0 {
                None
            } else {
                Some(Op::Slli { r, s, imm: 32 - imm5 })
            }
        }
        // srai ar,at,sa (sa 0..31): source is `t` (not `s`); imm5 =
        // ((op2&1)<<4)|s. Verified (binutils objdump): `a0 38 31` -> srai a3,a10,0x18 (imm5=24).
        (0x1, 0x2) | (0x1, 0x3) => {
            let imm5 = ((op2 & 1) << 4) | s;
            Some(Op::Srai { r, t, imm: imm5 })
        }
        // srli ar,at,imm4 (imm4 0..15): source is `t`; `s` is the plain
        // (unsplit) 4-bit shift count. Example: `b0 69 41` -> srli a6,a11,9.
        (0x1, 0x4) => Some(Op::Srli { r, t, imm: s }),
        // src ar,as,at: funnel shift right, all three roles in their usual
        // r/s/t positions. Verified (binutils objdump): `80 28 81` -> src a2,a8,a8.
        (0x1, 0x8) => Some(Op::Src { r, s, t }),
        // srl ar,at: `s` is a fixed selector nibble (must be 0) -- any
        // nonzero `s` is a different/invalid encoding (confirmed `excw` by
        // sweep). Verified (binutils objdump): `a0 d0 91` -> srl a13,a10 (s=0).
        (0x1, 0x9) if s == 0 => Some(Op::Srl { r, t }),
        // sll ar,as: `t` is the fixed selector nibble here (must be 0) --
        // the Sll/Srl pair swap which of s/t is the selector vs. the source
        // register. Verified (binutils objdump): `00 33 a1` -> sll a3,a3 (t=0).
        (0x1, 0xA) if t == 0 => Some(Op::Sll { r, s }),
        // mul16u ar,as,at: 16x16 unsigned multiply, RRR op1=1,op2=0xC --
        // shares op1 with the shift family above but a distinct op2. Both
        // objdump AND xtensa-modules.c (op1=1 table, op2=12) confirm this
        // (op1,op2) pair; see Op::Mul16u's doc. Verified (binutils objdump): `d0 23 c1` ->
        // mul16u a2,a3,a13.
        (0x1, 0xC) => Some(Op::Mul16u { r, s, t }),
        // mul16s ar,as,at: 16x16 signed multiply, RRR op1=1,op2=0xD. Both
        // objdump AND xtensa-modules.c (op1=1 table, op2=13) confirm this
        // pair; see Op::Mul16s's doc. Verified (binutils objdump): `30 22 d1` -> mul16s
        // a2,a2,a3.
        (0x1, 0xD) => Some(Op::Mul16s { r, s, t }),
        // Shift-amount-setting / normalize-shift-amount group, RRR
        // op1=0,op2=4: `r` is itself a sub-opcode selector within this
        // group (not a register, unlike every op1=0 arm above) -- verified
        // by an exhaustive objdump sweep over `r` (0..15): 0=ssr, 1=ssl,
        // 2=ssa8l, 3=ssa8b, 4=ssai, 6=rer, 7=wer, 14=nsa, 15=nsau (only
        // ssr/ssl/ssai/nsau are implemented; the rest aren't). `t` (and for `Ssai`, `t`'s range)
        // further gates
        // validity per sub-op -- see each `Op` variant's doc.
        (0x0, 0x4) => match r {
            // ssr as: `SAR = AR[s]&31`. Verified (binutils objdump): `00 06 40` -> ssr a6.
            0x0 if t == 0 => Some(Op::Ssr { s }),
            // ssl as: `SAR = 32-(AR[s]&31)`. Verified (binutils objdump): `00 14 40` -> ssl a4.
            0x1 if t == 0 => Some(Op::Ssl { s }),
            // ssai imm (0..31): imm = ((t&1)<<4)|s, t restricted to 0/1.
            // Verified (binutils objdump): `10 40 40` -> ssai 0x10 (t=1,s=0 -> imm=16).
            0x4 if t <= 1 => Some(Op::Ssai { imm: ((t & 1) << 4) | s }),
            // nsau ar,as: leading-zero count. Dest is `t` here (not `r`,
            // which is the fixed sub-op selector 15) -- verified by sweep
            // with distinct t/s. Verified (binutils objdump): `20 f2 40` -> nsau a2,a2.
            0xF => Some(Op::Nsau { t, s }),
            _ => None,
        },
        // sext ar,as,imm (imm 7..22): imm = t+7. `xtensa-lx106-elf-objdump`
        // can't decode this opcode (prints `excw`, same gap as `min`);
        // verified via binutils libisa, this core's config: `00 98 23` ->
        // sext a9,a8,7 (t=0 -> imm=7).
        (0x3, 0x2) => Some(Op::Sext { r, s, imm: t + 7 }),
        // mull ar,as,at: low-32 32x32 multiply, RRR op1=2,op2=8. Both
        // objdump AND xtensa-modules.c (op1=2 table, op2=8) confirm this
        // pair; see Op::Mull's doc. Verified (binutils objdump): `50 73 82` -> mull a7,a3,a5.
        (0x2, 0x8) => Some(Op::Mull { r, s, t }),
        // quou/remu/rems ar,as,at: unsigned divide/remainder + signed
        // remainder, RRR op1=2, op2=0xC/0xE/0xF. `xtensa-lx106-elf-objdump`
        // can't decode this trio (prints `excw`, the lx106 core lacks the
        // Divide option, same gap as `min`/`sext`/`s32ri`) -- verified
        // instead against xtensa-modules.c's op1=2 decode table: op2=12 ->
        // quou (457), op2=14 -> remu (459), op2=15 -> rems (460). See
        // Op::Quou's doc for the register-role and divide-by-zero notes.
        // Vectors: `b0 49 c2` -> quou a4,a9,a11; `c0 73 e2` -> remu
        // a7,a3,a12; `d0 a6 f2` -> rems a10,a6,a13.
        (0x2, 0xC) => Some(Op::Quou { r, s, t }),
        (0x2, 0xE) => Some(Op::Remu { r, s, t }),
        (0x2, 0xF) => Some(Op::Rems { r, s, t }),
        // extui art,ars,shiftimm,maskimm -- op1 IN {4,5} selects EXTUI (op2
        // IS data here, not a further selector, in both rows). Register
        // fields follow the SAME convention as `or` above: r = dest, t =
        // src; op2 = maskimm-1 (1-16). Example: `90 54 74` -> extui a5,a9,4,8 (dest==src in this vector, so it alone can't
        // disambiguate which nibble is which -- confirmed separately by
        // sweeping non-aliased registers against objdump: n1=1,n3=7 ->
        // `extui a7,a1,...`, i.e. r is always the first/dest operand).
        //
        // shiftimm is 5 bits (0-31), NOT the plain 4-bit `s` nibble (0-15)
        // this arm originally used: `s` supplies only the low 4 bits, and
        // op1's LSB supplies the 5th (`shiftimm = ((op1&1)<<4)|s`) -- the
        // same split-immediate pattern this file already uses for
        // Slli/Srai/Ssai. Matching op1==4 only would silently truncate
        // shifts >=16 (op1==5) to 4 bits. Verified against xtensa-modules.c's outer op1 dispatch
        // (`case 4: case 5: return 78 /* extui */` -- a DIFFERENT switch
        // than the op1==3 branch's own unrelated `case 4: return 444 /* min
        // */`) AND (binutils objdump) `30 30 f5` ->
        // extui a3,a3,0x10,0x10 (op1=5 -> shiftbit5=1, s=0 -> shiftimm=16).
        (0x4 | 0x5, maskimm_m1) => {
            Some(Op::Extui { r, t, shiftimm: ((op1 & 1) << 4) | s, maskimm: maskimm_m1 + 1 })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_mov_n() {
        // mov.n at,as: op0=0xD, r=0. t=6 -> byte0 0x6d; r=0,s=9 -> byte1 0x09.
        let d = decode(&[0x6d, 0x09], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::MovN { t: 6, s: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi_n() {
        // movi.n a7,43: op0=0xC; imm7 = (n1<<4)|n3 = (2<<4)|0xb = 43; t=7 is
        // byte1's low nibble -> `2c b7`.
        let d = decode(&[0x2c, 0xb7], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::MoviN { t: 7, imm: 43 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movi() {
        // movi a5,-300: RRI8 r=0xA, imm12 = (s<<8)|byte2 = 0xed4 -> -300 after
        // 12-bit sign extension (0xed4 - 0x1000). t=5, op0=2 -> byte0 0x52;
        // r=0xA, s=0xe -> byte1 0xae.
        let d = decode(&[0x52, 0xae, 0xd4], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Movi { t: 5, imm: -300 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_or() {
        // or a6,a9,a13: RRR op1=0,op2=2. t=13 -> byte0 0xd0; r=6,s=9 -> byte1
        // 0x69; op2=2,op1=0 -> byte2 0x20.
        let d = decode(&[0xd0, 0x69, 0x20], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Or { r: 6, s: 9, t: 13 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_extui() {
        // extui a5,a9,4,8: RRR op1=4, op2=maskimm-1=7, s=shiftimm low nibble=4,
        // t=9 (source), r=5 (dest) -> `90 54 74`.
        let d = decode(&[0x90, 0x54, 0x74], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Extui { r: 5, t: 9, shiftimm: 4, maskimm: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_extui_with_shift_ge_16() {
        // Shifts >=16 need op1==5: `30 30 f5` -> extui a3,a3,0x10,0x10
        // (binutils objdump). op1's byte here is 0xf5 -- op1
        // (low nibble) is 5, NOT 4; op1's LSB supplies the missing 5th bit of the shift
        // amount (shiftimm = ((op1&1)<<4)|s = (1<<4)|0 = 16).
        let d = decode(&[0x30, 0x30, 0xf5], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Extui { r: 3, t: 3, shiftimm: 16, maskimm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_add() {
        // `c0 59 80` -> add a5,a9,a12 (RRR: t=12, r=5, s=9, op2=8, op1=0)
        let d = decode(&[0xc0, 0x59, 0x80], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Add { r: 5, s: 9, t: 12 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_add_n() {
        // `3a 6b` -> add.n a6,a11,a3 (op0=0xA; t=3, r=6, s=11)
        let d = decode(&[0x3a, 0x6b], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::AddN { r: 6, s: 11, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi() {
        // `92 c4 15` -> addi a9,a4,0x15 (RRI8 r=0xC; t=9, s=4, imm8=0x15)
        let d = decode(&[0x92, 0xc4, 0x15], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addi { t: 9, s: 4, imm: 0x15 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n() {
        // `4b 53` -> addi.n a5,a3,4. Encoded imm nibble is 4, which is NOT
        // the -1 remap case (that's only raw==0).
        let d = decode(&[0x4b, 0x53], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::AddiN { t: 5, s: 3, imm: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n_distinct_dest_src_pins_nibble_mapping() {
        // A vector with s==t can't tell a correct dest/src mapping from a
        // swapped one. Synthetic vector with DISTINCT regs: `1b 42` ->
        // addi.n a4,a2,1 (dest=a4, src=a2). A swapped mapping would decode this as
        // AddiN { t: 2, s: 4 } and fail.
        let d = decode(&[0x1b, 0x42], 0);
        assert!(matches!(d.op, Op::AddiN { t: 4, s: 2, imm: 1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n_zero_selector_means_minus_one() {
        // Sweep-derived (binutils objdump: `0b 22` -> addi.n a2,a2,-1):
        // raw imm nibble 0 means -1, not 0 -- there is no encoding for +0.
        // `0b 22` -- same s/t as decodes_addi_n, imm nibble 0.
        let d = decode(&[0x0b, 0x22], 0);
        assert!(matches!(d.op, Op::AddiN { t: 2, s: 2, imm: -1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addmi() {
        // `62 d3 03` -> addmi a6,a3,0x300 (RRI8 r=0xD; raw byte 0x03 ->
        // sign_extend8=3, <<8 = 0x300).
        let d = decode(&[0x62, 0xd3, 0x03], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addmi { t: 6, s: 3, imm: 0x300 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sub() {
        // `a0 63 c0` -> sub a6,a3,a10 (RRR op2=0xC)
        let d = decode(&[0xa0, 0x63, 0xc0], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Sub { r: 6, s: 3, t: 10 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx2() {
        // `60 74 90` -> addx2 a7,a4,a6 (RRR op2=9)
        let d = decode(&[0x60, 0x74, 0x90], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx2 { r: 7, s: 4, t: 6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx4() {
        // `50 38 a0` -> addx4 a3,a8,a5 (RRR op2=0xA)
        let d = decode(&[0x50, 0x38, 0xa0], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx4 { r: 3, s: 8, t: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx8() {
        // `20 a7 b0` -> addx8 a10,a7,a2 (RRR op2=0xB)
        let d = decode(&[0x20, 0xa7, 0xb0], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx8 { r: 10, s: 7, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_subx8() {
        // `90 4c f0` -> subx8 a4,a12,a9 (RRR op2=0xF)
        let d = decode(&[0x90, 0x4c, 0xf0], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Subx8 { r: 4, s: 12, t: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_and() {
        // `80 b5 10` -> and a11,a5,a8 (RRR op2=1)
        let d = decode(&[0x80, 0xb5, 0x10], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::And { r: 11, s: 5, t: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_xor() {
        // `70 c3 30` -> xor a12,a3,a7 (RRR op2=3)
        let d = decode(&[0x70, 0xc3, 0x30], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Xor { r: 12, s: 3, t: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_neg() {
        // `40 90 60` -> neg a9,a4 (s field is 0,
        // disambiguating from Abs which shares op1=0,op2=6).
        let d = decode(&[0x40, 0x90, 0x60], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Neg { r: 9, t: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_abs() {
        // `d0 51 60` -> abs a5,a13 (s field is 1).
        let d = decode(&[0xd0, 0x51, 0x60], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Abs { r: 5, t: 13 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movnez() {
        // `60 3c 93` -> movnez a3,a12,a6 (RRR op1=3,op2=9)
        let d = decode(&[0x60, 0x3c, 0x93], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Movnez { r: 3, s: 12, t: 6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_moveqz() {
        // `b0 85 83` -> moveqz a8,a5,a11 (RRR op1=3,op2=8)
        let d = decode(&[0xb0, 0x85, 0x83], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Moveqz { r: 8, s: 5, t: 11 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_min() {
        // `90 c3 43` -> min a12,a3,a9 (RRR op1=3,op2=4; the lx106 objdump
        // prints excw for this opcode, same gap as s32ri).
        let d = decode(&[0x90, 0xc3, 0x43], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Min { r: 12, s: 3, t: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_minu() {
        // `70 4b 63` -> minu a4,a11,a7 (RRR op1=3,op2=6)
        let d = decode(&[0x70, 0x4b, 0x63], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Minu { r: 4, s: 11, t: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_maxu() {
        // `a0 d6 73` -> maxu a13,a6,a10 (RRR op1=3,op2=7)
        let d = decode(&[0xa0, 0xd6, 0x73], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Maxu { r: 13, s: 6, t: 10 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_mov_as_or_with_s_eq_t() {
        // Wide `mov as,at` is an assembler alias for `or as,at,at` (s==t),
        // NOT a separate opcode -- this is the existing `Or` arm, exercised
        // with s==t. `30 93 20` -> or a9,a3,a3, i.e.
        // `mov a9,a3`.
        let d = decode(&[0x30, 0x93, 0x20], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Or { r: 9, s: 3, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_slli() {
        // `40 29 11` -> slli a2,a9,12: op1=1, op2=1 so imm5 = (1<<4)|t =
        // 16|4 = 20, sa = 32-20 = 12; r=2 (dest), s=9 (src).
        let d = decode(&[0x40, 0x29, 0x11], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Slli { r: 2, s: 9, imm: 12 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_slli_distinct_dest_src_pins_mapping() {
        // Objdump-confirmed synthetic vector (t=5,s=6,r=7,op1=1,op2=0):
        // `50 76 01` -> slli a7,a6,27. Distinct dest(a7)/src(a6) pins r=dest,
        // s=src (a swapped mapping would report Slli{r:6,s:7,..}). imm5 =
        // ((op2&1)<<4)|t = (0<<4)|5 = 5, shift = 32-5 = 27.
        let d = decode(&[0x50, 0x76, 0x01], 0);
        assert!(matches!(d.op, Op::Slli { r: 7, s: 6, imm: 27 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_slli_reserved_sa32_is_unknown() {
        // sa (the architectural shift amount) is defined only for 1..31 --
        // xtensa-modules.c's Operand_msalp32_decode: `sal = raw&0x1f;
        // msalp32 = 0x20-sal`, so msalp32==32 only when the raw field is 0.
        // ISA RM SLLI, p.525, confirms this is genuinely reserved, not just
        // an odd-but-valid encoding: the shift amount is 1..31, encoded as
        // 32-shift, and "when the sa field is 0, the result of this
        // instruction is undefined". raw==0 means op2&1==0 (op2=0)
        // and t==0: `00 33 01` (r=s=3, arbitrary -- only t/op1/op2 matter
        // here).
        let d = decode(&[0x00, 0x33, 0x01], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_srli() {
        // `b0 69 41` -> srli a6,a11,9: op1=1, op2=4; r=6 (dest), t=11 (src),
        // s=9 is the plain shift count.
        let d = decode(&[0xb0, 0x69, 0x41], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Srli { r: 6, t: 11, imm: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_srli_distinct_dest_src_pins_mapping() {
        // Objdump-confirmed synthetic vector (t=5,s=6,r=7,op1=1,op2=4):
        // `50 76 41` -> srli a7,a5,6. Distinct dest(a7)/src(a5) pins r=dest,
        // t=src (NOT s -- unlike Slli, the source register here is the raw
        // `t` nibble; `s` instead carries the plain 4-bit shift count).
        let d = decode(&[0x50, 0x76, 0x41], 0);
        assert!(matches!(d.op, Op::Srli { r: 7, t: 5, imm: 6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_srai() {
        // `a0 38 31` -> srai a3,a10,0x18 (24) (binutils objdump). Dest(a3) and
        // src(a10) are distinct here, so this vector alone pins the
        // mapping: r=dest, t=src (same source-is-t layout as Srli). imm5 =
        // ((op2&1)<<4)|s = (1<<4)|8 = 24.
        let d = decode(&[0xa0, 0x38, 0x31], 0);
        assert!(matches!(d.op, Op::Srai { r: 3, t: 10, imm: 24 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sll() {
        // `00 33 a1` -> sll a3,a3 (binutils objdump; t=0 fixed selector).
        // Aliased dest/src (r=s=3); see the synthetic vector below.
        let d = decode(&[0x00, 0x33, 0xa1], 0);
        assert!(matches!(d.op, Op::Sll { r: 3, s: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sll_distinct_dest_src_pins_mapping() {
        // Objdump-confirmed synthetic vector (t=0,s=2,r=9,op1=1,op2=0xA):
        // `00 92 a1` -> sll a9,a2. Distinct dest(a9)/src(a2) pins r=dest,
        // s=src.
        let d = decode(&[0x00, 0x92, 0xa1], 0);
        assert!(matches!(d.op, Op::Sll { r: 9, s: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_srl() {
        // `a0 d0 91` -> srl a13,a10 (binutils objdump; s=0 fixed selector).
        // Dest(a13)/src(a10) are already distinct, pinning r=dest, t=src.
        let d = decode(&[0xa0, 0xd0, 0x91], 0);
        assert!(matches!(d.op, Op::Srl { r: 13, t: 10 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_src() {
        // `80 28 81` -> src a2,a8,a8 (binutils objdump). High(s)/low(t) both
        // alias to a8; see the synthetic vector below for the pinned roles.
        let d = decode(&[0x80, 0x28, 0x81], 0);
        assert!(matches!(d.op, Op::Src { r: 2, s: 8, t: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_src_distinct_high_low_pins_mapping() {
        // Objdump-confirmed synthetic vector (t=5,s=6,r=7,op1=1,op2=8):
        // `50 76 81` -> src a7,a6,a5. Distinct high(a6=s)/low(a5=t) pins
        // s=high-half, t=low-half (a swapped mapping would report
        // Src{s:5,t:6}).
        let d = decode(&[0x50, 0x76, 0x81], 0);
        assert!(matches!(d.op, Op::Src { r: 7, s: 6, t: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ssl() {
        // `00 14 40` -> ssl a4 (binutils objdump; r=1,t=0 fixed selectors).
        let d = decode(&[0x00, 0x14, 0x40], 0);
        assert!(matches!(d.op, Op::Ssl { s: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ssr() {
        // `00 06 40` -> ssr a6 (binutils objdump; r=0,t=0 fixed selectors).
        let d = decode(&[0x00, 0x06, 0x40], 0);
        assert!(matches!(d.op, Op::Ssr { s: 6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ssai() {
        // `10 40 40` -> ssai 0x10 (16) (binutils objdump).
        // t=1,s=0 -> imm = ((1&1)<<4)|0 = 16.
        let d = decode(&[0x10, 0x40, 0x40], 0);
        assert!(matches!(d.op, Op::Ssai { imm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sext() {
        // binutils libisa, this core's config (objdump prints `excw` for this
        // opcode, same gap as `min`): `00 98 23` -> sext a9,a8,7. Dest(a9)
        // and src(a8) are distinct here, pinning r=dest, s=src; t=0 -> imm =
        // 0+7 = 7.
        let d = decode(&[0x00, 0x98, 0x23], 0);
        assert!(matches!(d.op, Op::Sext { r: 9, s: 8, imm: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_nsau() {
        // `20 f2 40` -> nsau a2,a2 (binutils objdump; r=15 fixed selector).
        // Aliased dest/src (t=s=2); see the synthetic vector below.
        let d = decode(&[0x20, 0xf2, 0x40], 0);
        assert!(matches!(d.op, Op::Nsau { t: 2, s: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_nsau_distinct_dest_src_pins_mapping() {
        // Objdump-confirmed synthetic vector (t=0,s=7,r=15,op1=0,op2=4):
        // `00 f7 40` -> nsau a0,a7. Distinct dest(a0=t)/src(a7=s) pins
        // t=dest, s=src -- the swapped-from-usual convention since `r` here
        // is the sub-op selector, not a register.
        let d = decode(&[0x00, 0xf7, 0x40], 0);
        assert!(matches!(d.op, Op::Nsau { t: 0, s: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_mull() {
        // objdump: `50 73 82` -> mull a7,a3,a5.
        let d = decode(&[0x50, 0x73, 0x82], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Mull { r: 7, s: 3, t: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_mul16s() {
        // objdump: `30 22 d1` -> mul16s a2,a2,a3.
        let d = decode(&[0x30, 0x22, 0xd1], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Mul16s { r: 2, s: 2, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_mul16u() {
        // objdump: `d0 23 c1` -> mul16u a2,a3,a13.
        let d = decode(&[0xd0, 0x23, 0xc1], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Mul16u { r: 2, s: 3, t: 13 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_quou() {
        // xtensa-modules.c oracle (objdump prints excw, lx106 lacks Divide):
        // `b0 49 c2` -> quou a4,a9,a11 (op1=2,op2=0xC).
        let d = decode(&[0xb0, 0x49, 0xc2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Quou { r: 4, s: 9, t: 11 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_remu() {
        // xtensa-modules.c oracle: `c0 73 e2` -> remu a7,a3,a12 (op1=2,op2=0xE).
        let d = decode(&[0xc0, 0x73, 0xe2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Remu { r: 7, s: 3, t: 12 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_rems() {
        // xtensa-modules.c oracle: `d0 a6 f2` -> rems a10,a6,a13 (op1=2,op2=0xF).
        let d = decode(&[0xd0, 0xa6, 0xf2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rems { r: 10, s: 6, t: 13 }), "got {:?}", d.op);
    }
}
