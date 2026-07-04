//! Branch-family decode: the 27 conditional/unconditional branch opcodes
//! spanning two formats.
//!
//! **SI format (`op0=6`)**: shared with `entry` and the zero-overhead-loop
//! family (`loop`/`loopnez`, `control.rs`'s `decode_entry_fmt` /
//! `decode_loop_fmt`, M2a Task 7). The low 2 bits of the byte0-high nibble
//! ("t"/n1 field, called `n` in `xtensa-modules.c`'s
//! `Field_n_Slot_inst_get`) select a sub-family: `n==0` -> `j`
//! (unconditional; the OTHER 2 bits of that nibble, "m", are NOT a selector
//! here -- they're part of the 18-bit immediate); `n==1` -> BRI12-vs-zero
//! (`beqz`/`bnez`/`bltz`/`bgez`, `m` selects the mnemonic); `n==2` ->
//! BRI8-vs-B4CONST (`beqi`/`bnei`/`blti`/`bgei`, `m` selects); `n==3` ->
//! `m==0` is `entry`, `m==1` is `loop`/`loopnez` (`control::decode_loop_fmt`),
//! `m==2`/`m==3` are `bltui`/`bgeui`.
//!
//! **B format (`op0=7`, BRI8)**: `t` (byte0 high nibble), `s` (byte1 low
//! nibble), `r` (byte1 high nibble) select the comparison across the full
//! 0..15 range (0=bnone, 1=beq, 2=blt, 3=bltu, 4=ball, 5=bbc, 6/7=bbci,
//! 8=bany, 9=bne, 0xA=bge, 0xB=bgeu, 0xC=bnall, 0xD=bbs, 0xE/0xF=bbsi);
//! `imm8` (byte2) is the branch offset for every one of these.
//!
//! Every field layout here (the `n`/`m` split, the B-format `r` map, the
//! B4CONST/B4CONSTU tables, and the `bbci`/`bbsi` `r`-LSB-is-immediate-data
//! quirk) was cross-checked two ways: (1) `xtensa-modules.c`'s
//! `Field_*_Slot_inst_get`/`Operand_*_decode`/`CONST_TBL_*` (AMD's vendored
//! Xtensa ISA tables -- see module doc), and (2) an independent objdump
//! round-trip sweep (`xtensa-lx106-elf-objdump -D -b binary -m xtensa` fed
//! synthetic vectors built from those formulas) confirming every mnemonic,
//! register, and target this file decodes. Individual doc comments on each
//! `Op` variant in `decode/mod.rs` cite the specific vector.

use super::{sign_extend, Op};

/// `B4CONST`, the signed constant table indexed by the BRI8-vs-const
/// family's `r` field (`beqi`/`bnei`/`blti`/`bgei`). Reproduced from
/// `xtensa-modules.c`'s `CONST_TBL_b4c_0` (`Operand_b4const_decode`: `value
/// = CONST_TBL_b4c_0[r & 0xf]`).
pub(crate) const B4CONST: [i32; 16] = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128, 256];

/// `B4CONSTU`, the unsigned constant table for `bltui`/`bgeui`. Reproduced
/// from `xtensa-modules.c`'s `CONST_TBL_b4cu_0`.
pub(crate) const B4CONSTU: [u32; 16] = [32768, 65536, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 32, 64, 128, 256];

/// Shared `label8`/`label12`/`soffset` target formula: every branch/jump
/// offset in this file resolves to `pc + 4 + sign_extend(raw, bits)`
/// (`xtensa-modules.c`'s `Operand_label8_decode`/`_ator`/`_rtoa` and
/// siblings all share this `4 + sign_extend` shape).
fn pc_rel_target(pc: u32, raw: u32, bits: u32) -> u32 {
    pc.wrapping_add(4).wrapping_add(sign_extend(raw, bits) as u32)
}

/// SI-format (`op0=6`) branch/jump decode: `j`, the BRI12-vs-zero family
/// (`beqz`/`bnez`/`bltz`/`bgez`), the BRI8-vs-B4CONST family
/// (`beqi`/`bnei`/`blti`/`bgei`), and `bltui`/`bgeui`. `None` for `n==3`
/// with `m==0` (`entry`, left to `control::decode_entry_fmt`) or `m==1`
/// (`loop`/`loopnez`, left to `control::decode_loop_fmt`), so `decode()`
/// tries those next.
pub(super) fn decode_si(n1: u8, n2: u8, n3: u8, b2: u8, word: u32, pc: u32) -> Option<Op> {
    let n = n1 & 0x3;
    let m = (n1 >> 2) & 0x3;
    match n {
        // j: unconditional. imm18 = word>>6 (spans the "m" bits too -- they
        // are NOT a selector for this n value). Verified: `46 02 00` @ pc 0
        // -> j 0xd (objdump).
        0 => {
            let imm18 = word >> 6;
            Some(Op::J { target: pc_rel_target(pc, imm18, 18) })
        }
        // BRI12-vs-zero: s = n2, imm12 = n3 | (b2<<4) (same Field_imm12
        // extraction as control::decode_entry_fmt's frame-size field).
        1 => {
            let s = n2;
            let imm12 = (n3 as u32) | ((b2 as u32) << 4);
            let target = pc_rel_target(pc, imm12, 12);
            match m {
                0 => Some(Op::Beqz { s, target }),
                1 => Some(Op::Bnez { s, target }),
                2 => Some(Op::Bltz { s, target }),
                3 => Some(Op::Bgez { s, target }),
                _ => unreachable!("m is 2 bits, all 4 values covered"),
            }
        }
        // BRI8-vs-B4CONST(signed): s = n2, const index = n3, label8 = b2.
        2 => {
            let s = n2;
            let imm = B4CONST[n3 as usize];
            let target = pc_rel_target(pc, b2 as u32, 8);
            match m {
                0 => Some(Op::Beqi { s, imm, target }),
                1 => Some(Op::Bnei { s, imm, target }),
                2 => Some(Op::Blti { s, imm, target }),
                3 => Some(Op::Bgei { s, imm, target }),
                _ => unreachable!("m is 2 bits, all 4 values covered"),
            }
        }
        // n==3: m==0 is entry, m==1 is loop/loopnez -- both left to the next
        // decoders in decode()'s chain (control::decode_entry_fmt /
        // decode_loop_fmt respectively). m==2/3 are bltui/bgeui
        // (BRI8-vs-B4CONSTU), same field shape as the signed family above.
        3 => {
            if m == 2 || m == 3 {
                let s = n2;
                let imm = B4CONSTU[n3 as usize];
                let target = pc_rel_target(pc, b2 as u32, 8);
                if m == 2 {
                    Some(Op::Bltui { s, imm, target })
                } else {
                    Some(Op::Bgeui { s, imm, target })
                }
            } else {
                None
            }
        }
        _ => unreachable!("n is 2 bits, all 4 values covered"),
    }
}

/// B-format (`op0=7`, BRI8) decode: the register-compare
/// (`beq`/`bne`/`blt`/`bltu`/`bge`/`bgeu`), bit-test
/// (`bbc`/`bbs`/`bbci`/`bbsi`), and mask-test (`bnone`/`bany`/`ball`/
/// `bnall`) families. `t` = n1, `s` = n2, `r` = n3 selects the comparison
/// across the full 0..15 range (no gaps), so this always returns `Some` for
/// any `op0==7` instruction decode() routes here.
pub(super) fn decode_bri8(n1: u8, n2: u8, n3: u8, b2: u8, pc: u32) -> Option<Op> {
    let t = n1;
    let s = n2;
    let r = n3;
    let target = pc_rel_target(pc, b2 as u32, 8);
    match r {
        0 => Some(Op::Bnone { s, t, target }),
        1 => Some(Op::Beq { s, t, target }),
        2 => Some(Op::Blt { s, t, target }),
        3 => Some(Op::Bltu { s, t, target }),
        4 => Some(Op::Ball { s, t, target }),
        5 => Some(Op::Bbc { s, t, target }),
        // bbci: r's LSB is part of the 5-bit bit-index immediate, not a
        // further selector -- bit = ((r&1)<<4) | t.
        6 | 7 => Some(Op::Bbci { s, bit: ((r & 1) << 4) | t, target }),
        8 => Some(Op::Bany { s, t, target }),
        9 => Some(Op::Bne { s, t, target }),
        0xA => Some(Op::Bge { s, t, target }),
        0xB => Some(Op::Bgeu { s, t, target }),
        0xC => Some(Op::Bnall { s, t, target }),
        0xD => Some(Op::Bbs { s, t, target }),
        // bbsi: same r-LSB-as-immediate-data pairing as bbci.
        0xE | 0xF => Some(Op::Bbsi { s, bit: ((r & 1) << 4) | t, target }),
        _ => unreachable!("r is 4 bits, all 16 values covered"),
    }
}

/// Narrow (`op0=0xC`) `beqz.n`/`bnez.n` decode. `arith::decode_narrow`
/// claims `n1<=0x7` for `movi.n`, leaving `0x8..=0xF` (where the TOP 2 bits
/// of `n1`, `(n1>>2)&3`, are 2 or 3) to this decoder; the BOTTOM 2 bits of
/// `n1` are immediate data (`imm6`'s high 2 bits), not part of the opcode
/// identity -- confirmed by an objdump round-trip with a nonzero low-2-bits
/// value (`bc 52` still decodes `beqz.n`, not some other mnemonic). `None`
/// for any other `op0`, or for `op0==0xC` with the top 2 bits neither 2 nor
/// 3 (unreachable in practice: `arith` already claims 0/1 via `n1<=0x7`, and
/// this only receives `n1 in 0x8..=0xF` in the first place -- kept for
/// defensiveness, not because it's expected to trigger).
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8, pc: u32) -> Option<Op> {
    if op0 != 0xC {
        return None;
    }
    let ident = (n1 >> 2) & 0x3;
    let imm6 = (((n1 & 0x3) as u32) << 4) | (n3 as u32);
    let s = n2;
    // uimm6 = 4 + imm6 (Operand_uimm6_decode), target = pc + uimm6.
    let target = pc.wrapping_add(4).wrapping_add(imm6);
    match ident {
        2 => Some(Op::BeqzN { s, target }),
        3 => Some(Op::BnezN { s, target }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_j() {
        // `46 02 00` @ pc 0 -> j 0xd (objdump).
        let d = decode(&[0x46, 0x02, 0x00], 0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::J { target: 0xd }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beqz() {
        // `16 88 04` @ pc 3 -> beqz a8, 0x4f (objdump).
        let d = decode(&[0x16, 0x88, 0x04], 3);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Beqz { s: 8, target: 0x4f }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnez() {
        // `56 88 04` @ pc 6 -> bnez a8, 0x52 (objdump).
        let d = decode(&[0x56, 0x88, 0x04], 6);
        assert!(matches!(d.op, Op::Bnez { s: 8, target: 0x52 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bltz() {
        // `96 88 04` @ pc 9 -> bltz a8, 0x55 (objdump).
        let d = decode(&[0x96, 0x88, 0x04], 9);
        assert!(matches!(d.op, Op::Bltz { s: 8, target: 0x55 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bgez() {
        // `d6 88 04` @ pc 0xc -> bgez a8, 0x58 (objdump).
        let d = decode(&[0xd6, 0x88, 0x04], 0xc);
        assert!(matches!(d.op, Op::Bgez { s: 8, target: 0x58 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beqz_n() {
        // `8c 52` @ pc 0xf -> beqz.n a2, 0x18 (objdump).
        let d = decode(&[0x8c, 0x52], 0xf);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::BeqzN { s: 2, target: 0x18 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beqz_n_with_nonzero_imm6_hi_bits() {
        // `bc 52` @ pc 0 -> beqz.n a2, 0x39 (objdump). n1's low 2 bits are
        // nonzero here (imm6 hi bits == 3) -- this pins that they're
        // immediate data, not part of the opcode identity (only n1's TOP 2
        // bits distinguish beqz.n/bnez.n).
        let d = decode(&[0xbc, 0x52], 0);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::BeqzN { s: 2, target: 0x39 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnez_n() {
        // `cc 52` @ pc 0x11 -> bnez.n a2, 0x1a (objdump).
        let d = decode(&[0xcc, 0x52], 0x11);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::BnezN { s: 2, target: 0x1a }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnez_n_with_nonzero_imm6_hi_bits() {
        // `ec 52` @ pc 0 -> bnez.n a2, 0x29 (objdump).
        let d = decode(&[0xec, 0x52], 0);
        assert!(matches!(d.op, Op::BnezN { s: 2, target: 0x29 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beq() {
        // `57 17 14` @ pc 0x13 -> beq a7, a5, 0x2b (objdump).
        let d = decode(&[0x57, 0x17, 0x14], 0x13);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Beq { s: 7, t: 5, target: 0x2b }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bne() {
        // `57 97 14` @ pc 0x16 -> bne a7, a5, 0x2e (objdump).
        let d = decode(&[0x57, 0x97, 0x14], 0x16);
        assert!(matches!(d.op, Op::Bne { s: 7, t: 5, target: 0x2e }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_blt() {
        // `57 27 14` @ pc 0x19 -> blt a7, a5, 0x31 (objdump).
        let d = decode(&[0x57, 0x27, 0x14], 0x19);
        assert!(matches!(d.op, Op::Blt { s: 7, t: 5, target: 0x31 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bltu() {
        // `57 37 14` @ pc 0x1c -> bltu a7, a5, 0x34 (objdump).
        let d = decode(&[0x57, 0x37, 0x14], 0x1c);
        assert!(matches!(d.op, Op::Bltu { s: 7, t: 5, target: 0x34 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bge() {
        // `57 a7 14` @ pc 0x1f -> bge a7, a5, 0x37 (objdump).
        let d = decode(&[0x57, 0xa7, 0x14], 0x1f);
        assert!(matches!(d.op, Op::Bge { s: 7, t: 5, target: 0x37 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bgeu() {
        // `57 b7 14` @ pc 0x22 -> bgeu a7, a5, 0x3a (objdump).
        let d = decode(&[0x57, 0xb7, 0x14], 0x22);
        assert!(matches!(d.op, Op::Bgeu { s: 7, t: 5, target: 0x3a }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beqi() {
        // `26 66 02` @ pc 0x25 -> beqi a6, 6, 0x2b (objdump; index 6 ->
        // B4CONST[6]==6).
        let d = decode(&[0x26, 0x66, 0x02], 0x25);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Beqi { s: 6, imm: 6, target: 0x2b }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnei() {
        // `66 66 02` @ pc 0x28 -> bnei a6, 6, 0x2e (objdump).
        let d = decode(&[0x66, 0x66, 0x02], 0x28);
        assert!(matches!(d.op, Op::Bnei { s: 6, imm: 6, target: 0x2e }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_blti() {
        // `a6 66 02` @ pc 0x2b -> blti a6, 6, 0x31 (objdump).
        let d = decode(&[0xa6, 0x66, 0x02], 0x2b);
        assert!(matches!(d.op, Op::Blti { s: 6, imm: 6, target: 0x31 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bgei() {
        // `e6 66 02` @ pc 0x2e -> bgei a6, 6, 0x34 (objdump).
        let d = decode(&[0xe6, 0x66, 0x02], 0x2e);
        assert!(matches!(d.op, Op::Bgei { s: 6, imm: 6, target: 0x34 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_beqi_pins_a_different_b4const_index() {
        // Same shape as beqi above but index 9 -> B4CONST[9]==10, distinct
        // from the value at index 6 -- pins that the decoder truly looks up
        // the TABLE (not e.g. `index+something`). r(n3)=9: byte1 = (9<<4)|6
        // = 0x96.
        let d = decode(&[0x26, 0x96, 0x02], 0x25);
        assert!(matches!(d.op, Op::Beqi { s: 6, imm: 10, target: 0x2b }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bltui() {
        // `b6 66 02` @ pc 0x31 -> bltui a6, 6, 0x37 (objdump; index 6 ->
        // B4CONSTU[6]==6).
        let d = decode(&[0xb6, 0x66, 0x02], 0x31);
        assert!(matches!(d.op, Op::Bltui { s: 6, imm: 6, target: 0x37 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bgeui() {
        // `f6 66 02` @ pc 0x34 -> bgeui a6, 6, 0x3a (objdump).
        let d = decode(&[0xf6, 0x66, 0x02], 0x34);
        assert!(matches!(d.op, Op::Bgeui { s: 6, imm: 6, target: 0x3a }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bltui_pins_a_b4constu_index_that_diverges_from_b4const() {
        // Index 0: B4CONST[0]==-1 but B4CONSTU[0]==32768 -- a genuinely
        // different table, not a shared/reused one. r(n3)=0: byte1 =
        // (0<<4)|6 = 0x06.
        let d = decode(&[0xb6, 0x06, 0x02], 0x31);
        assert!(matches!(d.op, Op::Bltui { s: 6, imm: 32768, target: 0x37 }), "got {:?}", d.op);
    }

    #[test]
    fn bltui_does_not_shadow_entry_or_loop() {
        // n==3,m==0 (entry, n1==0x3) must still decode as Entry, not Bltui --
        // proves decode_si's m==2||m==3 guard doesn't swallow the whole
        // n==3 family.
        let d = decode(&[0x36, 0x41, 0x00], 0x33244);
        assert!(matches!(d.op, Op::Entry { s: 1, imm: 32 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbci() {
        // `37 64 05` @ pc 0x37 -> bbci a4, 3, 0x40 (objdump).
        let d = decode(&[0x37, 0x64, 0x05], 0x37);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Bbci { s: 4, bit: 3, target: 0x40 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbci_r_equals_7_variant_pins_lsb_as_immediate() {
        // `37 74 05` -> bbci a4, 19, 0x9 (objdump). r=7 instead of the
        // canonical r=6: bit index becomes 19 (16|3), mnemonic unchanged --
        // pins that r's LSB is immediate data, not a further selector.
        let d = decode(&[0x37, 0x74, 0x05], 0);
        assert!(matches!(d.op, Op::Bbci { s: 4, bit: 19, target: 0x9 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbsi() {
        // `37 e4 05` @ pc 0x3a -> bbsi a4, 3, 0x43 (objdump).
        let d = decode(&[0x37, 0xe4, 0x05], 0x3a);
        assert!(matches!(d.op, Op::Bbsi { s: 4, bit: 3, target: 0x43 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbsi_r_equals_15_variant_pins_lsb_as_immediate() {
        // `37 f4 05` @ pc 3 -> bbsi a4, 19, 0xc (objdump).
        let d = decode(&[0x37, 0xf4, 0x05], 3);
        assert!(matches!(d.op, Op::Bbsi { s: 4, bit: 19, target: 0xc }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbc() {
        // `47 5a 0e` @ pc 0x3d -> bbc a10, a4, 0x4f (objdump).
        let d = decode(&[0x47, 0x5a, 0x0e], 0x3d);
        assert!(matches!(d.op, Op::Bbc { s: 10, t: 4, target: 0x4f }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bbs() {
        // `47 da 0e` @ pc 0x40 -> bbs a10, a4, 0x52 (objdump).
        let d = decode(&[0x47, 0xda, 0x0e], 0x40);
        assert!(matches!(d.op, Op::Bbs { s: 10, t: 4, target: 0x52 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnone() {
        // `77 0f 0f` @ pc 0x43 -> bnone a15, a7, 0x56 (objdump).
        let d = decode(&[0x77, 0x0f, 0x0f], 0x43);
        assert!(matches!(d.op, Op::Bnone { s: 15, t: 7, target: 0x56 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bany() {
        // `a7 88 bc` @ pc 0x46 -> bany a8, a10, 0x6 (objdump; imm8=0xbc is
        // negative when sign-extended -- the only negative-offset oracle
        // vector in this task).
        let d = decode(&[0xa7, 0x88, 0xbc], 0x46);
        assert!(matches!(d.op, Op::Bany { s: 8, t: 10, target: 0x6 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_ball() {
        // `37 44 02` @ pc 0x49 -> ball a4, a3, 0x4f (objdump).
        let d = decode(&[0x37, 0x44, 0x02], 0x49);
        assert!(matches!(d.op, Op::Ball { s: 4, t: 3, target: 0x4f }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_bnall() {
        // `67 c7 13` @ pc 0x4c -> bnall a7, a6, 0x63 (objdump).
        let d = decode(&[0x67, 0xc7, 0x13], 0x4c);
        assert!(matches!(d.op, Op::Bnall { s: 7, t: 6, target: 0x63 }), "got {:?}", d.op);
    }
}
