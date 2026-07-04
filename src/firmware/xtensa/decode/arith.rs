//! Register/immediate arithmetic decode: `mov.n`, `movi.n`, `movi`, `or`,
//! `extui`, plus the M2a integer/logical/conditional-move/min-max family
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
        // (r=n3 dest, s=n2, t=n1). Verified: `7a e8` -> add.n a14,a8,a7
        // (all three distinct, unambiguously pinning the mapping).
        0xA => Some(Op::AddN { r: n3, s: n2, t: n1 }),
        // addi.n at,as,imm -- op0=0xB. Confirmed by sweeping each nibble
        // independently against objdump (holding the others at a
        // non-aliasing value): byte1 high nibble (n3) = dest `t`, byte1 low
        // nibble (n2) = src `s`, byte0 high nibble (n1) = the immediate
        // selector -- a DIFFERENT nibble->field mapping than mov.n (a
        // different narrow format sharing this op0 range). The immediate
        // encoding has no representation for 0: raw nibble 0 means -1, raw
        // 1-15 mean themselves. Verified: `1b 22` -> addi.n a2,a2,1 (raw
        // nibble 1); sweep confirms raw 0 -> -1.
        0xB => {
            let imm = if n1 == 0 { -1 } else { n1 as i32 };
            Some(Op::AddiN { t: n3, s: n2, imm })
        }
        _ => None,
    }
}

/// RRI8 format (LSAI group): `movi at,imm12` is `r==0xA` -- here "s" (n2) is
/// not a register, it's the immediate's high 4 bits. Verified: `d2 a0 ac` ->
/// movi a13,172. `addi at,as,imm8` is `r==0xC`; `addmi at,as,imm8` is
/// `r==0xD` (`imm` stored already `sign_extend8(imm8)<<8`, the final value
/// to add). `None` for other `r` values, so `decode()` tries the next
/// category (`mem`'s `l32i`, tried first, or falls to `Op::Unknown`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    match r {
        0xA => {
            let raw12 = ((s as u32) << 8) | (b2 as u32);
            Some(Op::Movi { t, imm: sign_extend(raw12, 12) })
        }
        // addi at,as,imm8 -- verified: `62 cf 27` -> addi a6,a15,0x27 (39).
        0xC => Some(Op::Addi { t, s, imm: sign_extend(b2 as u32, 8) }),
        // addmi at,as,imm8 -- verified: `22 d7 02` -> addmi a2,a7,0x200
        // (raw byte 0x02 -> sign_extend8=2, <<8 = 0x200).
        0xD => Some(Op::Addmi { t, s, imm: sign_extend(b2 as u32, 8) << 8 }),
        _ => None,
    }
}

/// RRR format: `or ar,as,at` (op1=0,op2=2), `extui art,ars,shiftimm,
/// maskimm` (op1=4, any op2 -- op2 IS data here, the mask-width-1), plus the
/// M2a integer/logical/cmov/minmax group (op1=0 sub-ops 1/3/6/8/9/0xA/0xB/
/// 0xC/0xF; op1=3 sub-ops 4/6/7/8/9). `None` for anything else in the RRR
/// format, so `decode()` tries `system` then `control`.
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, _word: u32) -> Option<Op> {
    match (op1, op2) {
        // and ar,as,at -- verified: `30 72 10` -> and a7,a2,a3
        (0x0, 0x1) => Some(Op::And { r, s, t }),
        // or ar,as,at -- verified: `20 a2 20` -> or a10,a2,a2
        (0x0, 0x2) => Some(Op::Or { r, s, t }),
        // xor ar,as,at -- verified: `80 85 30` -> xor a8,a5,a8
        (0x0, 0x3) => Some(Op::Xor { r, s, t }),
        // neg/abs share op1=0,op2=6; `s` is a fixed selector nibble here
        // (not a register) -- s==0 selects neg, s==1 selects abs. Verified:
        // `70 80 60` -> neg a8,a7 (s=0); `20 21 60` -> abs a2,a2 (s=1).
        (0x0, 0x6) if s == 0 => Some(Op::Neg { r, t }),
        (0x0, 0x6) if s == 1 => Some(Op::Abs { r, t }),
        // add ar,as,at -- verified: `20 26 80` -> add a2,a6,a2
        (0x0, 0x8) => Some(Op::Add { r, s, t }),
        // addx2 ar,as,at = (AR[s]<<1)+AR[t] -- verified: `20 32 90` ->
        // addx2 a3,a2,a2
        (0x0, 0x9) => Some(Op::Addx2 { r, s, t }),
        // addx4 ar,as,at = (AR[s]<<2)+AR[t] -- verified: `20 22 a0` ->
        // addx4 a2,a2,a2
        (0x0, 0xA) => Some(Op::Addx4 { r, s, t }),
        // addx8 ar,as,at = (AR[s]<<3)+AR[t] -- verified: `50 52 b0` ->
        // addx8 a5,a2,a5
        (0x0, 0xB) => Some(Op::Addx8 { r, s, t }),
        // sub ar,as,at -- verified: `90 44 c0` -> sub a4,a4,a9
        (0x0, 0xC) => Some(Op::Sub { r, s, t }),
        // subx8 ar,as,at = (AR[s]<<3)-AR[t] -- verified: `80 98 f0` ->
        // subx8 a9,a8,a8. Only subx8 appears in the firmware (no
        // subx2/subx4, which would be op2=0xD/0xE).
        (0x0, 0xF) => Some(Op::Subx8 { r, s, t }),
        // min ar,as,at: signed min -- verified (Ghidra listing.txt oracle,
        // objdump prints excw): `20 2a 43` -> min a2,a10,a2. (max, signed,
        // is op2=5 and absent from the firmware.)
        (0x3, 0x4) => Some(Op::Min { r, s, t }),
        // minu ar,as,at: unsigned min -- verified: `90 94 63` -> minu
        // a9,a4,a9
        (0x3, 0x6) => Some(Op::Minu { r, s, t }),
        // maxu ar,as,at: unsigned max -- verified: `40 57 73` -> maxu
        // a5,a7,a4
        (0x3, 0x7) => Some(Op::Maxu { r, s, t }),
        // moveqz ar,as,at: if AR[t]==0 { AR[r]=AR[s] } -- verified: `40 f2
        // 83` -> moveqz a15,a2,a4
        (0x3, 0x8) => Some(Op::Moveqz { r, s, t }),
        // movnez ar,as,at: if AR[t]!=0 { AR[r]=AR[s] } -- verified: `50 a3
        // 93` -> movnez a10,a3,a5
        (0x3, 0x9) => Some(Op::Movnez { r, s, t }),
        // Shift-immediate/shift-register/funnel-shift family, RRR op1==1:
        // op2 selects the specific op. Field roles verified by an
        // exhaustive objdump sweep and cross-checked against every real
        // instance of each op in the Ghidra listing (100% match across
        // 1346 slli / 193 srli / 49 srai / 160 sll / 107 srl / 19 src
        // instances) -- the firmware's own single-vector examples alias too
        // many operand nibbles to pin the roles alone. See each `Op`
        // variant's doc for the per-op field layout.
        //
        // slli ar,as,sa (sa 1..31): imm5 = ((op2&1)<<4)|t, sa = 32-imm5.
        // Verified: `20 33 01` -> slli a3,a3,0x1e (imm5=2, sa=30).
        (0x1, 0x0) | (0x1, 0x1) => {
            let imm5 = ((op2 & 1) << 4) | t;
            Some(Op::Slli { r, s, imm: 32 - imm5 })
        }
        // srai ar,at,sa (sa 0..31): source is `t` (not `s`); imm5 =
        // ((op2&1)<<4)|s. Verified: `a0 38 31` -> srai a3,a10,0x18 (imm5=24).
        (0x1, 0x2) | (0x1, 0x3) => {
            let imm5 = ((op2 & 1) << 4) | s;
            Some(Op::Srai { r, t, imm: imm5 })
        }
        // srli ar,at,imm4 (imm4 0..15): source is `t`; `s` is the plain
        // (unsplit) 4-bit shift count. Verified: `40 42 41` -> srli
        // a4,a4,0x2.
        (0x1, 0x4) => Some(Op::Srli { r, t, imm: s }),
        // src ar,as,at: funnel shift right, all three roles in their usual
        // r/s/t positions. Verified: `80 28 81` -> src a2,a8,a8.
        (0x1, 0x8) => Some(Op::Src { r, s, t }),
        // srl ar,at: `s` is a fixed selector nibble (must be 0) -- any
        // nonzero `s` is a different/invalid encoding (confirmed `excw` by
        // sweep). Verified: `a0 d0 91` -> srl a13,a10 (s=0).
        (0x1, 0x9) if s == 0 => Some(Op::Srl { r, t }),
        // sll ar,as: `t` is the fixed selector nibble here (must be 0) --
        // the Sll/Srl pair swap which of s/t is the selector vs. the source
        // register. Verified: `00 33 a1` -> sll a3,a3 (t=0).
        (0x1, 0xA) if t == 0 => Some(Op::Sll { r, s }),
        // mul16u ar,as,at: 16x16 unsigned multiply, RRR op1=1,op2=0xC --
        // shares op1 with the shift family above but a distinct op2. Both
        // objdump AND xtensa-modules.c (op1=1 table, op2=12) confirm this
        // (op1,op2) pair; see Op::Mul16u's doc. Verified: `d0 23 c1` ->
        // mul16u a2,a3,a13.
        (0x1, 0xC) => Some(Op::Mul16u { r, s, t }),
        // mul16s ar,as,at: 16x16 signed multiply, RRR op1=1,op2=0xD. Both
        // objdump AND xtensa-modules.c (op1=1 table, op2=13) confirm this
        // pair; see Op::Mul16s's doc. Verified: `30 22 d1` -> mul16s
        // a2,a2,a3.
        (0x1, 0xD) => Some(Op::Mul16s { r, s, t }),
        // Shift-amount-setting / normalize-shift-amount group, RRR
        // op1=0,op2=4: `r` is itself a sub-opcode selector within this
        // group (not a register, unlike every op1=0 arm above) -- verified
        // by an exhaustive objdump sweep over `r` (0..15): 0=ssr, 1=ssl,
        // 2=ssa8l, 3=ssa8b, 4=ssai, 6=rer, 7=wer, 14=nsa, 15=nsau (only
        // ssr/ssl/ssai/nsau are in the M2a opcode set; the rest aren't
        // implemented). `t` (and for `Ssai`, `t`'s range) further gates
        // validity per sub-op -- see each `Op` variant's doc.
        (0x0, 0x4) => match r {
            // ssr as: `SAR = AR[s]&31`. Verified: `00 06 40` -> ssr a6.
            0x0 if t == 0 => Some(Op::Ssr { s }),
            // ssl as: `SAR = 32-(AR[s]&31)`. Verified: `00 14 40` -> ssl a4.
            0x1 if t == 0 => Some(Op::Ssl { s }),
            // ssai imm (0..31): imm = ((t&1)<<4)|s, t restricted to 0/1.
            // Verified: `10 40 40` -> ssai 0x10 (t=1,s=0 -> imm=16).
            0x4 if t <= 1 => Some(Op::Ssai { imm: ((t & 1) << 4) | s }),
            // nsau ar,as: leading-zero count. Dest is `t` here (not `r`,
            // which is the fixed sub-op selector 15) -- verified by sweep
            // with distinct t/s. Verified: `20 f2 40` -> nsau a2,a2.
            0xF => Some(Op::Nsau { t, s }),
            _ => None,
        },
        // sext ar,as,imm (imm 7..22): imm = t+7. `xtensa-lx106-elf-objdump`
        // can't decode this opcode (prints `excw`, same BE-option gap as
        // `min`); verified via the Ghidra listing.txt oracle instead: `00
        // 98 23` -> sext a9,a8,7 (t=0 -> imm=7; every one of the firmware's
        // 17 real instances uses imm=7).
        (0x3, 0x2) => Some(Op::Sext { r, s, imm: t + 7 }),
        // mull ar,as,at: low-32 32x32 multiply, RRR op1=2,op2=8. Both
        // objdump AND xtensa-modules.c (op1=2 table, op2=8) confirm this
        // pair; see Op::Mull's doc. Verified: `50 73 82` -> mull a7,a3,a5.
        (0x2, 0x8) => Some(Op::Mull { r, s, t }),
        // quou/remu/rems ar,as,at: unsigned divide/remainder + signed
        // remainder, RRR op1=2, op2=0xC/0xE/0xF. `xtensa-lx106-elf-objdump`
        // can't decode this trio (prints `excw`, the lx106 core lacks the
        // Divide option, same gap as `min`/`sext`/`s32ri`) -- verified
        // instead against xtensa-modules.c's op1=2 decode table: op2=12 ->
        // quou (457), op2=14 -> remu (459), op2=15 -> rems (460). See
        // Op::Quou's doc for the register-role and divide-by-zero notes.
        // Vectors: `f0 2e c2` -> quou a2,a14,a15; `70 52 e2` -> remu
        // a5,a2,a7; `50 6a f2` -> rems a6,a10,a5.
        (0x2, 0xC) => Some(Op::Quou { r, s, t }),
        (0x2, 0xE) => Some(Op::Remu { r, s, t }),
        (0x2, 0xF) => Some(Op::Rems { r, s, t }),
        // extui art,ars,shiftimm,maskimm -- op1 IN {4,5} selects EXTUI (op2
        // IS data here, not a further selector, in both rows). Register
        // fields follow the SAME convention as `or` above: r = dest, t =
        // src; op2 = maskimm-1 (1-16). Verified: `30 30 f4` -> extui
        // a3,a3,0,16 (dest==src in this vector, so it alone can't
        // disambiguate which nibble is which -- confirmed separately by
        // sweeping non-aliased registers against objdump: n1=1,n3=7 ->
        // `extui a7,a1,...`, i.e. r is always the first/dest operand).
        //
        // shiftimm is 5 bits (0-31), NOT the plain 4-bit `s` nibble (0-15)
        // this arm originally used: `s` supplies only the low 4 bits, and
        // op1's LSB supplies the 5th (`shiftimm = ((op1&1)<<4)|s`) -- the
        // same split-immediate pattern this file already uses for
        // Slli/Srai/Ssai. Found missing by the M2a Task 10 exit-gate
        // coverage scan: 59 real firmware `extui` instances (all needing
        // shift>=16, i.e. op1==5) decoded to `Op::Unknown` because the
        // pre-fix arm matched op1==4 only, silently truncating the shift to
        // 4 bits. Verified against xtensa-modules.c's outer op1 dispatch
        // (`case 4: case 5: return 78 /* extui */` -- a DIFFERENT switch
        // than the op1==3 branch's own unrelated `case 4: return 444 /* min
        // */`) AND cross-checked against all 59 real occurrences, e.g. `30
        // 30 f5` @ 0xe0ac -> extui a3,a3,0x10,0x10 (op1=5 -> shiftbit5=1,
        // s=0 -> shiftimm=16).
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

    #[test]
    fn decodes_extui_with_shift_ge_16() {
        // Found by the M2a Task 10 exit-gate coverage scan: 59 real firmware
        // `extui` instances (all with shift>=16) decoded to Op::Unknown
        // before this fix. Real firmware vector @0xe0ac: `30 30 f5` ->
        // extui a3,a3,0x10,0x10 (Ghidra). op1's byte here is 0xf5 -- op1
        // (low nibble) is 5, NOT the 4 the pre-fix decoder required
        // exclusively; op1's LSB supplies the missing 5th bit of the shift
        // amount (shiftimm = ((op1&1)<<4)|s = (1<<4)|0 = 16).
        let d = decode(&[0x30, 0x30, 0xf5], 0xe0ac);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Extui { r: 3, t: 3, shiftimm: 16, maskimm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_add() {
        // listing.txt: `20 26 80` @0x3f4f -> add a2,a6,a2
        let d = decode(&[0x20, 0x26, 0x80], 0x3f4f);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Add { r: 2, s: 6, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_add_n() {
        // listing.txt: `7a e8` @0x2761 -> add.n a14,a8,a7
        let d = decode(&[0x7a, 0xe8], 0x2761);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::AddN { r: 14, s: 8, t: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi() {
        // listing.txt: `62 cf 27` @0x3f55 -> addi a6,a15,0x27
        let d = decode(&[0x62, 0xcf, 0x27], 0x3f55);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addi { t: 6, s: 15, imm: 0x27 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n() {
        // listing.txt: `1b 22` @0x2820 -> addi.n a2,a2,1. Encoded imm nibble
        // is 1, which is NOT the -1..1 remap case (that's only raw==0).
        let d = decode(&[0x1b, 0x22], 0x2820);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::AddiN { t: 2, s: 2, imm: 1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n_distinct_dest_src_pins_nibble_mapping() {
        // The firmware vector (`1b 22`) aliases s==t==2, so it can't tell a
        // correct dest/src mapping from a swapped one. Synthetic vector with
        // DISTINCT regs, confirmed against objdump: `1b 42` -> addi.n
        // a4,a2,1 (dest=a4, src=a2). A swapped mapping would decode this as
        // AddiN { t: 2, s: 4 } and fail.
        let d = decode(&[0x1b, 0x42], 0);
        assert!(matches!(d.op, Op::AddiN { t: 4, s: 2, imm: 1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addi_n_zero_selector_means_minus_one() {
        // Sweep-derived (not in the firmware table, but a real encoding):
        // raw imm nibble 0 means -1, not 0 -- there is no encoding for +0.
        // `0b 22` -- same s/t as decodes_addi_n, imm nibble 0.
        let d = decode(&[0x0b, 0x22], 0);
        assert!(matches!(d.op, Op::AddiN { t: 2, s: 2, imm: -1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addmi() {
        // listing.txt: `22 d7 02` @0x27c5 -> addmi a2,a7,0x200 (raw byte
        // 0x02 -> sign_extend8=2, <<8 = 0x200).
        let d = decode(&[0x22, 0xd7, 0x02], 0x27c5);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addmi { t: 2, s: 7, imm: 0x200 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sub() {
        // listing.txt: `90 44 c0` @0x28b1 -> sub a4,a4,a9
        let d = decode(&[0x90, 0x44, 0xc0], 0x28b1);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Sub { r: 4, s: 4, t: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx2() {
        // listing.txt: `20 32 90` @0x42f5 -> addx2 a3,a2,a2
        let d = decode(&[0x20, 0x32, 0x90], 0x42f5);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx2 { r: 3, s: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx4() {
        // listing.txt: `20 22 a0` @0x27e4 -> addx4 a2,a2,a2
        let d = decode(&[0x20, 0x22, 0xa0], 0x27e4);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx4 { r: 2, s: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_addx8() {
        // listing.txt: `50 52 b0` @0x59e5 -> addx8 a5,a2,a5
        let d = decode(&[0x50, 0x52, 0xb0], 0x59e5);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Addx8 { r: 5, s: 2, t: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_subx8() {
        // listing.txt: `80 98 f0` @0x4271 -> subx8 a9,a8,a8
        let d = decode(&[0x80, 0x98, 0xf0], 0x4271);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Subx8 { r: 9, s: 8, t: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_and() {
        // listing.txt: `30 72 10` @0x2759 -> and a7,a2,a3
        let d = decode(&[0x30, 0x72, 0x10], 0x2759);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::And { r: 7, s: 2, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_xor() {
        // listing.txt: `80 85 30` @0x2887 -> xor a8,a5,a8
        let d = decode(&[0x80, 0x85, 0x30], 0x2887);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Xor { r: 8, s: 5, t: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_neg() {
        // listing.txt: `70 80 60` @0xa433 -> neg a8,a7 (s field is 0,
        // disambiguating from Abs which shares op1=0,op2=6).
        let d = decode(&[0x70, 0x80, 0x60], 0xa433);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Neg { r: 8, t: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_abs() {
        // listing.txt: `20 21 60` @0x3b8a9 -> abs a2,a2 (s field is 1).
        let d = decode(&[0x20, 0x21, 0x60], 0x3b8a9);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Abs { r: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_movnez() {
        // listing.txt: `50 a3 93` @0x2918 -> movnez a10,a3,a5
        let d = decode(&[0x50, 0xa3, 0x93], 0x2918);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Movnez { r: 10, s: 3, t: 5 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_moveqz() {
        // listing.txt: `40 f2 83` @0x28b7 -> moveqz a15,a2,a4
        let d = decode(&[0x40, 0xf2, 0x83], 0x28b7);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Moveqz { r: 15, s: 2, t: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_min() {
        // listing.txt: `20 2a 43` @0x82f9 -> min a2,a10,a2 (Ghidra oracle;
        // objdump prints excw for this opcode, same gap as s32ri).
        let d = decode(&[0x20, 0x2a, 0x43], 0x82f9);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Min { r: 2, s: 10, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_minu() {
        // listing.txt: `90 94 63` @0x2819 -> minu a9,a4,a9
        let d = decode(&[0x90, 0x94, 0x63], 0x2819);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Minu { r: 9, s: 4, t: 9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_maxu() {
        // listing.txt: `40 57 73` @0x9d91 -> maxu a5,a7,a4
        let d = decode(&[0x40, 0x57, 0x73], 0x9d91);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Maxu { r: 5, s: 7, t: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_mov_as_or_with_s_eq_t() {
        // Ghidra prints `or as,at,at` (an `or` with s==t) as the pseudo-op
        // `mov`. Wide `mov` is NOT a separate opcode -- this is the existing
        // `Or` arm, exercised with s==t. `20 42 20` -> mov a4,a2 (== or
        // a4,a2,a2).
        let d = decode(&[0x20, 0x42, 0x20], 0x2820);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Or { r: 4, s: 2, t: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_slli() {
        // listing.txt: `20 33 01` @0x2750 -> slli a3,a3,0x1e (30). Aliased
        // dest/src (r=s=3); can't pin the roles alone, see the synthetic
        // vector below.
        let d = decode(&[0x20, 0x33, 0x01], 0x2750);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Slli { r: 3, s: 3, imm: 30 }), "got {:?}", d.op);
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
    fn decodes_srli() {
        // listing.txt: `40 42 41` @0x2753 -> srli a4,a4,0x2. Aliased dest/src
        // (r=t=4); see the synthetic vector below for the pinned mapping.
        let d = decode(&[0x40, 0x42, 0x41], 0x2753);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Srli { r: 4, t: 4, imm: 2 }), "got {:?}", d.op);
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
        // listing.txt: `a0 38 31` @0x51bd -> srai a3,a10,0x18 (24). Dest(a3) and
        // src(a10) are distinct here, so this real vector alone pins the
        // mapping: r=dest, t=src (same source-is-t layout as Srli). imm5 =
        // ((op2&1)<<4)|s = (1<<4)|8 = 24.
        let d = decode(&[0xa0, 0x38, 0x31], 0);
        assert!(matches!(d.op, Op::Srai { r: 3, t: 10, imm: 24 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sll() {
        // listing.txt vector `00 33 a1` -> sll a3,a3 (t=0 fixed selector).
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
        // listing.txt vector `a0 d0 91` -> srl a13,a10 (s=0 fixed selector).
        // Dest(a13)/src(a10) are already distinct, pinning r=dest, t=src.
        let d = decode(&[0xa0, 0xd0, 0x91], 0);
        assert!(matches!(d.op, Op::Srl { r: 13, t: 10 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_src() {
        // listing.txt vector `80 28 81` -> src a2,a8,a8. High(s)/low(t) both
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
        // listing.txt: `00 14 40` -> ssl a4 (r=1,t=0 fixed selectors).
        let d = decode(&[0x00, 0x14, 0x40], 0);
        assert!(matches!(d.op, Op::Ssl { s: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ssr() {
        // listing.txt: `00 06 40` -> ssr a6 (r=0,t=0 fixed selectors).
        let d = decode(&[0x00, 0x06, 0x40], 0);
        assert!(matches!(d.op, Op::Ssr { s: 6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ssai() {
        // Firmware's one real instance: `10 40 40` -> ssai 0x10 (16).
        // t=1,s=0 -> imm = ((1&1)<<4)|0 = 16.
        let d = decode(&[0x10, 0x40, 0x40], 0);
        assert!(matches!(d.op, Op::Ssai { imm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sext() {
        // Ghidra listing.txt oracle (objdump prints `excw` for this opcode,
        // same BE-option gap as `min`): `00 98 23` -> sext a9,a8,7. Dest(a9)
        // and src(a8) are distinct here, pinning r=dest, s=src; t=0 -> imm =
        // 0+7 = 7.
        let d = decode(&[0x00, 0x98, 0x23], 0);
        assert!(matches!(d.op, Op::Sext { r: 9, s: 8, imm: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_nsau() {
        // listing.txt: `20 f2 40` -> nsau a2,a2 (r=15 fixed selector).
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
        // `f0 2e c2` -> quou a2,a14,a15 (op1=2,op2=0xC).
        let d = decode(&[0xf0, 0x2e, 0xc2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Quou { r: 2, s: 14, t: 15 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_remu() {
        // xtensa-modules.c oracle: `70 52 e2` -> remu a5,a2,a7 (op1=2,op2=0xE).
        let d = decode(&[0x70, 0x52, 0xe2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Remu { r: 5, s: 2, t: 7 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_rems() {
        // xtensa-modules.c oracle: `50 6a f2` -> rems a6,a10,a5 (op1=2,op2=0xF).
        let d = decode(&[0x50, 0x6a, 0xf2], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rems { r: 6, s: 10, t: 5 }), "got {:?}", d.op);
    }
}
