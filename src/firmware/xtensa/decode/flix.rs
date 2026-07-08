//! FLIX (VLIW) bundle decode. This custom AMD Xtensa core packs a 3-slot or
//! 2-slot VLIW bundle behind two `op0` selectors: `op0` nibble (`byte0 & 0xf`)
//! `0xe` -> `xt_format1` (8 bytes, 3 slots), `0xf` -> `xt_format2` (8 bytes,
//! 2 slots). BOTH occur in this firmware -- an earlier census wrongly reported
//! "zero xt_format1"; the boot path hits a real `xt_format1` bundle (a triple-
//! `excw` barrier at phys 0xd903, atop a context-save routine).
//!
//! DERIVED FROM THE TOOLCHAIN. Every bit position here is derived from this
//! core's real ISA config (`binutils-2.37/bfd/xtensa-modules.c`, the
//! auto-generated FLIX tables) and byte-verified against the `xtdis` oracle
//! (`build/experiments/firmware-re/xtdis/`, which links that config through
//! libisa). The config-match is confirmed exact -- see
//! `docs/superpowers/findings/2026-07-05-iter8-flix-bundle-oracle.md` and the
//! design spec `docs/superpowers/specs/2026-07-05-m2c-flix-bundle-decode-design.md`.
//!
//! **Collapse model (`xt_format2`).** Every observed `xt_format2` bundle is
//! exactly one real op plus one inert slot (`excw` -- Exception Wait, a barrier
//! with no register/memory/PC effect -> a functional nop here -- or the slot's
//! designated `nop`). Because one slot is always inert, bundle execution is
//! never actually parallel, so we collapse the bundle to its single real `Op`
//! with `len = 8` and let the existing per-op `interp` execute it unchanged.
//! Two real ops in one bundle never occurs and would wall (`Op::Unknown`)
//! rather than silently drop a slot.
//!
//! **Parallel model (`xt_format1`).** The 3-slot format genuinely packs up to
//! three REAL ops that execute in parallel (reads see pre-bundle state, writes
//! commit together). `decode_format1` decodes each slot to an `Op` (dropping
//! inert `excw`/`nop` slots) and collapses: 0 real -> `Op::Nop`, 1 real -> that
//! op, >=2 real -> `Op::Flix1{ops}` (executed by the snapshot-diff parallel
//! executor `interp::Cpu::exec_flix1_bundle`). A slot that decodes to a real
//! opcode we don't yet map to an `Op` walls the whole bundle loudly
//! (`Op::Unknown`) rather than silently dropping it.
//!
//! **Slot layout (`xt_format2`).** The 8-byte bundle is two little-endian
//! 32-bit words `insn0`/`insn1` (libisa's `insnbuf` packing). Per
//! `Slot_xt_format2_Format_xt_flix64_slot0_4_get` /
//! `..._slot3_28_get`:
//! - slot0 (general slot, 24 bits) = `(insn0 >> 4) & 0xffffff`.
//! - slot3 (branch/`excw` slot) `slotbuf[0] = (insn0 >> 28) | ((insn1 &
//!   0xfffffff) << 4)`, `slotbuf[1] = (insn1 >> 28) & 0x7`.

use super::Op;

/// slot0 classification: a real op we can execute, or something we don't yet
/// recognize (walls). slot0 is the general-purpose slot; its own opcode is
/// selected by `op0 = (slot0 >> 20) & 0xf` (`Field_op0_xt_flix64_slot0`).
enum Slot0 {
    Real(Op),
    Unknown,
}

/// slot3 (== format2's slot1) classification. It is the wide-branch / `excw`
/// slot: `op0_s6 = (slotbuf[0] >> 27) & 0x1f` selects among 24 `.w18` branches
/// (cases 1..=24, each gated by a reserved-bits guard), `nop` (case 25), or the
/// `excw` default. We only need inert-vs-real here.
enum Slot3 {
    /// `excw` or `nop` -- no architectural effect.
    Inert,
    /// One of the `.w18` branches. Not yet decoded to an executable `Op`
    /// (no branch-slot bundle has been walked yet); surfaces as a wall so the
    /// next iteration implements it.
    RealBranch,
}

/// Decodes an `xt_format2` (`op0 == 0xf`) FLIX bundle to its single collapsed
/// `Op`. `bytes` must be at least 8 long (the caller guarantees this); `pc` is
/// the bundle's runtime address, needed for `l32r`'s PC-relative literal.
pub(super) fn decode_format2(bytes: &[u8], pc: u32) -> Op {
    let insn0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let insn1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);

    match (classify_slot0(insn0, pc), classify_slot3(insn1)) {
        // The observed shape: one real op in slot0, inert (excw/nop) slot3.
        (Slot0::Real(op), Slot3::Inert) => op,
        // Anything else -- unrecognized slot0, a real branch in slot3 (both a
        // not-yet-implemented op and, with a real slot0, a two-real-op bundle)
        // -- walls loudly rather than guessing. `word` carries insn0 to
        // identify the bundle in the `Step::Unknown` report.
        _ => Op::Unknown { word: insn0 },
    }
}

/// Classifies slot0 (the general-purpose slot).
fn classify_slot0(insn0: u32, pc: u32) -> Slot0 {
    let slot0 = (insn0 >> 4) & 0xff_ffff; // 24-bit slot, bits[4:27] of insn0
    let op0 = (slot0 >> 20) & 0xf; // Field_op0_xt_flix64_slot0
    match op0 {
        // `l32r at, <lit>` (slot0 `op0 == 1`). Register `at` = slot0 bits[0:3];
        // `imm16` = slot0 bits[4:19]; the offset formula is IDENTICAL to
        // base-ISA `l32r` (`((0xFFFF0000|imm16)<<2)` added to `(pc+3)&~3`),
        // confirmed against xtdis `undo_reloc` (0x28b4 bundle -> target
        // 0xfffe3094). See `decode::mem::decode_ri16`.
        1 => {
            let at = (slot0 & 0xf) as u8;
            let imm16 = (slot0 >> 4) & 0xffff;
            let base = pc.wrapping_add(3) & !3u32;
            let target = base.wrapping_add((0xFFFF_0000u32 | imm16) << 2);
            Slot0::Real(Op::L32r { t: at, target })
        }
        _ => Slot0::Unknown,
    }
}

/// Classifies slot3 (the branch/`excw` slot). A real `.w18` branch requires
/// `op0_s6 ∈ 1..=24` AND the reserved-bits guard `(slotbuf[1] & 7) == 0` (all
/// 24 branch cases share the identical `insn[1] & 7` guard in
/// `Slot_xt_flix64_slot3_decode`). `op0_s6 == 25` is `nop`; everything else --
/// including a failing guard, as in the dispatcher's `l32r` bundle
/// (`op0_s6 == 2`, `slotbuf[1] == 5`) -- is the `excw` default. Both `nop` and
/// `excw` are inert.
fn classify_slot3(insn1: u32) -> Slot3 {
    // slotbuf[0] top bits (27..31) come entirely from insn1 bits 23..27, and
    // slotbuf[1] from insn1 bits 28..30, so slot3's classification needs only
    // insn1 (slot3's low bits, which draw from insn0's top nibble, hold no
    // opcode-selecting field).
    let op0_s6 = (insn1 >> 23) & 0x1f; // (slotbuf[0] >> 27) & 0x1f
    let guard = (insn1 >> 28) & 0x7; // slotbuf[1] & 7
    if (1..=24).contains(&op0_s6) && guard == 0 {
        Slot3::RealBranch
    } else {
        Slot3::Inert // excw (default) or nop (op0_s6 == 25)
    }
}

// ===========================================================================
// xt_format1 (3-slot) decode.
//
// Slot bit-layout, from `Slot_xt_format1_Format_xt_flix64_slot{0,1,2}_*_get`
// (all set slotbuf[1]=0, so each slot word is a single u32):
//   slot0 (24-bit) = (insn0 >> 4)  & 0xff_ffff
//   slot1 (20-bit) = (insn0 >> 28) | ((insn1 & 0xffff) << 4)
//   slot2 (16-bit) = (insn1 >> 16) & 0xffff
// Each slot has its OWN opcode/operand field positions (below), ported one-for-
// one from `Slot_xt_flix64_slot{0,1,2}_decode` + their `Field_*_get` bodies.
// ===========================================================================

/// One slot's decode outcome.
enum SlotDecode {
    /// `excw` (opcode 0, each decode fn's default) or `nop` -- no architectural
    /// effect; dropped from the bundle.
    Inert,
    /// A real op mapped to our `Op`.
    Real(Op),
    /// A real op we identified but don't yet have an `Op`/exec for (`sra`,
    /// `movltz`/`movgez`, `subx2`/`subx4`, `max`, `clamps`, `ssa8l`/`ssa8b`,
    /// `nsa`). Walls the whole bundle loudly so it's added on first real
    /// occurrence rather than silently mis-executed -- OR an architecturally
    /// RESERVED encoding (e.g. `slli` with shift-amount 32, which real firmware
    /// never emits but xtdis still textually decodes). Both wall the bundle.
    /// Carries the mnemonic (read off the source at the wall site; the runtime
    /// wall reports `insn0`, which xtdis maps back to the mnemonic).
    Unmapped(#[allow(dead_code)] &'static str),
}

/// Decodes an `xt_format1` (`op0 == 0xe`) bundle. `bytes` >= 8 (caller
/// guarantees); `pc` resolves `l32r`'s literal and `j`'s target.
pub(super) fn decode_format1(bytes: &[u8], pc: u32) -> Op {
    let insn0 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let insn1 = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let s0 = (insn0 >> 4) & 0xff_ffff;
    let s1 = ((insn0 >> 28) & 0xf) | ((insn1 & 0xffff) << 4);
    let s2 = (insn1 >> 16) & 0xffff;

    let mut ops = Vec::with_capacity(3);
    for sd in [slot0(s0, pc), slot1(s1, pc), slot2(s2)] {
        match sd {
            SlotDecode::Inert => {}
            SlotDecode::Real(op) => ops.push(op),
            // Fail loud: a real slot op we can't execute yet -> wall the bundle.
            SlotDecode::Unmapped(_) => return Op::Unknown { word: insn0 },
        }
    }
    match ops.len() {
        0 => Op::Nop,
        1 => ops.into_iter().next().unwrap(),
        _ => Op::Flix1 { ops },
    }
}

/// slot0 (24-bit general slot). Field positions (`Field_*_Slot_xt_flix64_slot0`):
/// t=`w&0xf`, s=`(w>>4)&0xf`, r=`(w>>8)&0xf`, op1=`(w>>12)&0xf`,
/// op2=`(w>>16)&0xf`, op0=`(w>>20)&0xf`.
fn slot0(w: u32, pc: u32) -> SlotDecode {
    use SlotDecode::{Inert, Real, Unmapped};
    let t = (w & 0xf) as u8;
    let s = ((w >> 4) & 0xf) as u8;
    let r = ((w >> 8) & 0xf) as u8;
    let op1 = (w >> 12) & 0xf;
    let op2 = ((w >> 16) & 0xf) as u8;
    let op0 = (w >> 20) & 0xf;

    // op0 switch: a recognized op `return`s immediately; every non-match falls
    // OUT of the switch (mirroring the C `break`s) to the shared tail below,
    // which tries l32r (op0==1) / mov.n (op0==0) before defaulting to excw.
    match op0 {
        0 => {
            // extui: Field_combined..fld7 == 2, fld7 = (w>>13)&7.
            if (w >> 13) & 7 == 2 {
                // shiftimm(sae) = ((w>>12)&1)<<4 | ((w>>4)&0xf); maskimm = op2+1.
                let shiftimm = (((w >> 12) & 1) << 4 | ((w >> 4) & 0xf)) as u8;
                return Real(Op::Extui { r, t, shiftimm, maskimm: op2 + 1 });
            }
            match op1 {
                0 => match op2 {
                    0 => {
                        if r == 2 && s == 0 && t == 15 {
                            return Inert; // nop
                        }
                        // else: fall through to tail (mov.n/excw).
                    }
                    1 => return Real(Op::And { r, s, t }),
                    2 => return Real(Op::Or { r, s, t }),
                    3 => return Real(Op::Xor { r, s, t }),
                    4 => match r {
                        0 if t == 0 => return Real(Op::Ssr { s }),
                        1 if t == 0 => return Real(Op::Ssl { s }),
                        2 if t == 0 => return Unmapped("ssa8l"),
                        3 if t == 0 => return Unmapped("ssa8b"),
                        // ssai imm = ((t&1)<<4)|s, gated thi3==0 (thi3=(w>>1)&7).
                        4 if (w >> 1) & 7 == 0 => return Real(Op::Ssai { imm: ((t & 1) << 4) | s }),
                        14 => return Unmapped("nsa"),
                        15 => return Real(Op::Nsau { t, s }),
                        _ => {}
                    },
                    6 => match s {
                        0 => return Real(Op::Neg { r, t }),
                        1 => return Real(Op::Abs { r, t }),
                        _ => {}
                    },
                    8 => return Real(Op::Add { r, s, t }),
                    9 => return Real(Op::Addx2 { r, s, t }),
                    10 => return Real(Op::Addx4 { r, s, t }),
                    11 => return Real(Op::Addx8 { r, s, t }),
                    12 => return Real(Op::Sub { r, s, t }),
                    13 => return Unmapped("subx2"),
                    14 => return Unmapped("subx4"),
                    15 => return Real(Op::Subx8 { r, s, t }),
                    _ => {}
                },
                1 => {
                    // (w>>17)&7: ==1 -> srai, ==0 -> slli, else op2 switch.
                    let hi = (w >> 17) & 7;
                    if hi == 1 {
                        // srai imm5 = ((op2&1)<<4)|s.
                        return Real(Op::Srai { r, t, imm: ((op2 & 1) << 4) | s });
                    }
                    if hi == 0 {
                        // slli sa = 32 - imm5, imm5 = ((op2&1)<<4)|t. imm5==0
                        // (sa=32) is architecturally reserved -> wall.
                        let imm5 = ((op2 & 1) << 4) | t;
                        return if imm5 == 0 {
                            Unmapped("slli")
                        } else {
                            Real(Op::Slli { r, s, imm: 32 - imm5 })
                        };
                    }
                    match op2 {
                        4 => return Real(Op::Srli { r, t, imm: s }),
                        8 => return Real(Op::Src { r, s, t }),
                        9 if s == 0 => return Real(Op::Srl { r, t }),
                        10 if t == 0 => return Real(Op::Sll { r, s }),
                        11 if s == 0 => return Unmapped("sra"),
                        12 => return Real(Op::Mul16u { r, s, t }),
                        13 => return Real(Op::Mul16s { r, s, t }),
                        _ => {}
                    }
                }
                2 => {
                    if op2 == 8 {
                        return Real(Op::Mull { r, s, t });
                    }
                }
                3 => match op2 {
                    2 => return Real(Op::Sext { r, s, imm: t + 7 }),
                    3 => return Unmapped("clamps"),
                    4 => return Real(Op::Min { r, s, t }),
                    5 => return Unmapped("max"),
                    6 => return Real(Op::Minu { r, s, t }),
                    7 => return Real(Op::Maxu { r, s, t }),
                    8 => return Real(Op::Moveqz { r, s, t }),
                    9 => return Real(Op::Movnez { r, s, t }),
                    10 => return Unmapped("movltz"),
                    11 => return Unmapped("movgez"),
                    _ => {}
                },
                _ => {}
            }
        }
        2 => {
            // Loads/stores/movi/addi/addmi; imm8 = (w>>12)&0xff.
            let imm8 = (w >> 12) & 0xff;
            match r {
                0 => return Real(Op::L8ui { t, s, imm: imm8 }),
                1 => return Real(Op::L16ui { t, s, imm: imm8 << 1 }),
                2 => return Real(Op::L32i { t, s, imm: imm8 << 2 }),
                4 => return Real(Op::S8i { t, s, imm: imm8 }),
                5 => return Real(Op::S16i { t, s, imm: imm8 << 1 }),
                6 => return Real(Op::S32i { t, s, imm: imm8 << 2 }),
                9 => return Real(Op::L16si { t, s, imm: imm8 << 1 }),
                10 => {
                    // movi imm12 = ((w>>4)&0xf)<<8 | ((w>>12)&0xff), sign-ext 12.
                    let raw = ((w >> 4) & 0xf) << 8 | imm8;
                    return Real(Op::Movi { t, imm: super::sign_extend(raw, 12) });
                }
                12 => return Real(Op::Addi { t, s, imm: super::sign_extend(imm8, 8) }),
                13 => return Real(Op::Addmi { t, s, imm: super::sign_extend(imm8, 8) << 8 }),
                _ => {}
            }
        }
        _ => {}
    }

    // Tail (reached by fall-through from the switch, matching the C `break`s).
    // l32r: op0_s3 (== op0) == 1.
    if op0 == 1 {
        let imm16 = (w >> 4) & 0xffff;
        let base = pc.wrapping_add(3) & !3u32;
        let target = base.wrapping_add((0xFFFF_0000u32 | imm16) << 2);
        return Real(Op::L32r { t, target });
    }
    // mov.n: sae4==0 && fld8==3 && op0_s3==0 && fld49==0.
    // fld8=(w>>13)&7, op0_s3=(w>>20)&0xf, fld49=((w>>16)&0xf)<<4|((w>>8)&0xf).
    if (w >> 13) & 7 == 3 && (w >> 20) & 0xf == 0 && (((w >> 16) & 0xf) << 4 | ((w >> 8) & 0xf)) == 0 {
        return Real(Op::MovN { t, s });
    }
    Inert // excw (default)
}

/// slot1 (20-bit slot). Field positions (`Field_*_Slot_xt_flix64_slot1`):
/// t=`w&0xf`, r=`(w>>4)&0xf`, s=`(w>>8)&0xf`, op2=`(w>>8)&0xf`,
/// op0_s4=`(w>>18)&3`.
fn slot1(w: u32, pc: u32) -> SlotDecode {
    use SlotDecode::{Inert, Real, Unmapped};
    let t = (w & 0xf) as u8;
    let r = ((w >> 4) & 0xf) as u8;
    let s = ((w >> 8) & 0xf) as u8;
    let op2 = ((w >> 8) & 0xf) as u8;
    let op0_s4 = (w >> 18) & 3;

    // extui: fld19 (=(w>>17)&1) == 0 && op0_s4 == 1.
    if (w >> 17) & 1 == 0 && op0_s4 == 1 {
        // shiftimm(sae) = (w>>12)&0x1f; maskimm = op2+1.
        let shiftimm = ((w >> 12) & 0x1f) as u8;
        return Real(Op::Extui { r, t, shiftimm, maskimm: op2 + 1 });
    }
    // fld20 = (w>>16)&3.
    match (w >> 16) & 3 {
        0 if op0_s4 == 2 => {
            // movi imm12 = (w>>4)&0xfff, sign-ext 12.
            return Real(Op::Movi { t, imm: super::sign_extend((w >> 4) & 0xfff, 12) });
        }
        2 if op0_s4 == 1 => {
            // addi imm8 = ((w>>12)&0xf)<<4 | ((w>>4)&0xf), sign-ext 8.
            let imm8 = ((w >> 12) & 0xf) << 4 | ((w >> 4) & 0xf);
            return Real(Op::Addi { t, s, imm: super::sign_extend(imm8, 8) });
        }
        3 if op0_s4 == 1 => {
            let imm8 = ((w >> 12) & 0xf) << 4 | ((w >> 4) & 0xf);
            return Real(Op::Addmi { t, s, imm: super::sign_extend(imm8, 8) << 8 });
        }
        3 if op0_s4 == 2 && (w >> 12) & 0xf == 0 => return Real(Op::Xor { r, s, t }),
        _ => {}
    }
    // fld21 = (w>>13)&0x1f.
    match (w >> 13) & 0x1f {
        8 if op0_s4 == 2 => {
            // slli: sa = 32 - imm5, imm5 = sal = ((w>>12)&1)<<4 | (w&0xf).
            // imm5==0 (sa=32) is architecturally reserved -> wall.
            let imm5 = (((w >> 12) & 1) << 4 | (w & 0xf)) as u8;
            return if imm5 == 0 {
                Unmapped("slli")
            } else {
                Real(Op::Slli { r, s, imm: 32 - imm5 })
            };
        }
        16 if op0_s4 == 2 => {
            // srai imm5 = sargt = (w>>8)&0x1f.
            return Real(Op::Srai { r, t, imm: ((w >> 8) & 0x1f) as u8 });
        }
        // sll: fld57 = ((w>>12)&1)<<4 | (w&0xf) == 0.
        19 if op0_s4 == 2 && ((((w >> 12) & 1) << 4) | (w & 0xf)) == 0 => {
            return Real(Op::Sll { r, s });
        }
        _ => {}
    }
    // fld22 = (w>>12)&0x3f -- the big ALU dispatch.
    if op0_s4 == 2 {
        match (w >> 12) & 0x3f {
            18 => return Real(Op::Add { r, s, t }),
            19 => return Real(Op::Addx8 { r, s, t }),
            20 => return Real(Op::Addx2 { r, s, t }),
            21 => return Real(Op::And { r, s, t }),
            22 => return Real(Op::Moveqz { r, s, t }),
            23 => return Unmapped("movgez"),
            24 => return Real(Op::Addx4 { r, s, t }),
            25 => return Unmapped("movltz"),
            26 => return Real(Op::Movnez { r, s, t }),
            27 => return Real(Op::Mul16u { r, s, t }),
            28 => return Real(Op::Mul16s { r, s, t }),
            29 => return Real(Op::Mull { r, s, t }),
            30 => return Real(Op::Or { r, s, t }),
            31 => return Real(Op::Sext { r, s, imm: t + 7 }),
            34 => return Real(Op::Src { r, s, t }),
            36 => return Real(Op::Srli { r, t, imm: s }),
            _ => {}
        }
    }
    // Long-form single-op patterns (fldNN == const && op0_s4 == 2 && guard == 0).
    // fld23/fld25 = ((w>>12)&0x3f)<<3 | ((w>>4)&7).
    let fld23 = (((w >> 12) & 0x3f) << 3) | ((w >> 4) & 7);
    if op0_s4 == 2 {
        // mov.n: fld23==280 && fld51(=(w>>7)&1)==0.
        if fld23 == 280 && (w >> 7) & 1 == 0 {
            return Real(Op::MovN { t, s });
        }
        // jx: fld25==281 && fld52(=(w>>7)&1<<4 | w&0xf)==0.
        if fld23 == 281 && ((((w >> 7) & 1) << 4) | (w & 0xf)) == 0 {
            return Real(Op::Jx { s });
        }
        // ssl: fld26(=((w>>12)&0x3f)<<2 | (w>>5)&3)==141 && fld60(=((w>>7)&1)<<5 | w&0x1f)==0.
        let fld26 = (((w >> 12) & 0x3f) << 2) | ((w >> 5) & 3);
        if fld26 == 141 && ((((w >> 7) & 1) << 5) | (w & 0x1f)) == 0 {
            return Real(Op::Ssl { s });
        }
        // nop: fld28(=((w>>12)&0x3f)<<1 | (w>>6)&1)==71 && fld54(=((w>>7)&0x1f)<<6 | w&0x3f)==0.
        let fld28 = (((w >> 12) & 0x3f) << 1) | ((w >> 6) & 1);
        if fld28 == 71 && ((((w >> 7) & 0x1f) << 6) | (w & 0x3f)) == 0 {
            return Inert; // nop
        }
        // neg: fld30(=((w>>12)&0x3f)<<2 | (w>>8)&3)==148 && fld53(=(w>>10)&3)==0.
        let fld30 = (((w >> 12) & 0x3f) << 2) | ((w >> 8) & 3);
        if fld30 == 148 && (w >> 10) & 3 == 0 {
            return Real(Op::Neg { r, t });
        }
        // sra: fld32==149 && fld53==0.  (fld32 same body as fld30.)
        if fld30 == 149 && (w >> 10) & 3 == 0 {
            return Unmapped("sra");
        }
        // srl: fld33(=((w>>12)&0x3f)<<1 | (w>>9)&1)==75 && fld58(=((w>>10)&3)<<1 | (w>>8)&1)==0.
        let fld33 = (((w >> 12) & 0x3f) << 1) | ((w >> 9) & 1);
        if fld33 == 75 && ((((w >> 10) & 3) << 1) | ((w >> 8) & 1)) == 0 {
            return Real(Op::Srl { r, t });
        }
        // sub: fld35(=(w>>15)&7)==5 && fld62(=(w>>12)&7)==0.
        if (w >> 15) & 7 == 5 && (w >> 12) & 7 == 0 {
            return Real(Op::Sub { r, s, t });
        }
    }
    // j (op0_s4==3): offset = w & 0x3ffff (18-bit), target = pc+4+sign_extend.
    if op0_s4 == 3 {
        let off = w & 0x3ffff;
        let target = pc.wrapping_add(4).wrapping_add(super::sign_extend(off, 18) as u32);
        return Real(Op::J { target });
    }
    Inert // excw (default)
}

/// slot2 (16-bit slot). Field positions (`Field_*_Slot_xt_flix64_slot2`):
/// t=`w&0xf`, r=`(w>>4)&0xf`, s=`(w>>8)&0xf`, op0_s5=`(w>>13)&7`.
fn slot2(w: u32) -> SlotDecode {
    use SlotDecode::{Inert, Real, Unmapped};
    let t = (w & 0xf) as u8;
    let r = ((w >> 4) & 0xf) as u8;
    let s = ((w >> 8) & 0xf) as u8;
    let op0_s5 = (w >> 13) & 7;

    // fld36 = (w>>12)&1.
    match (w >> 12) & 1 {
        0 => match op0_s5 {
            1 => return Real(Op::Add { r, s, t }),
            5 => return Real(Op::Sub { r, s, t }),
            2 => return Real(Op::Addx2 { r, s, t }),
            3 => return Real(Op::And { r, s, t }),
            4 => return Real(Op::Sext { r, s, imm: t + 7 }),
            _ => {}
        },
        1 => match op0_s5 {
            // addi.n dest=r (bits4-7), src=s, imm=low nibble (t) with 0->-1 remap.
            1 => return Real(Op::AddiN { t: r, s, imm: if t == 0 { -1 } else { t as i32 } }),
            2 => return Real(Op::Addx4 { r, s, t }),
            3 => return Real(Op::Or { r, s, t }),
            5 => return Real(Op::Xor { r, s, t }),
            4 => return Real(Op::Srli { r, t, imm: s }),
            _ => {}
        },
        _ => {}
    }
    // movi.n: fld37(=((w>>12)&1)<<1 | (w>>7)&1)==0 && op0_s5==6. Dest register
    // is `s` (bits 8-11); the low bits hold imm7 (w & 0x7f), so it can't be `t`.
    if ((((w >> 12) & 1) << 1) | ((w >> 7) & 1)) == 0 && op0_s5 == 6 {
        let raw = w & 0x7f;
        let imm = if raw < 96 { raw as i32 } else { raw as i32 - 128 };
        return Real(Op::MoviN { t: s, imm });
    }
    // mov.n: fld39(=fld37<<1 | (w>>4)&1)==2 && op0_s5==6 && fld63(=(w>>5)&3)==0.
    let fld39 = (((((w >> 12) & 1) << 1) | ((w >> 7) & 1)) << 1) | ((w >> 4) & 1);
    if fld39 == 2 && op0_s5 == 6 && (w >> 5) & 3 == 0 {
        return Real(Op::MovN { t, s });
    }
    // nop: fld41(=fld39 body)==3 && op0_s5==6 && fld65==0.
    let fld65 = ((((w >> 8) & 0xf) << 2 | ((w >> 5) & 3)) << 4) | (w & 0xf);
    if fld39 == 3 && op0_s5 == 6 && fld65 == 0 {
        return Inert; // nop
    }
    // abs: fld42(=((w>>12)&1)<<3 | (w>>8)&7)==8 && op0_s5==6 && fld64(=(w>>11)&1)==0.
    let fld42 = (((w >> 12) & 1) << 3) | ((w >> 8) & 7);
    if fld42 == 8 && op0_s5 == 6 && (w >> 11) & 1 == 0 {
        return Real(Op::Abs { r, t });
    }
    // neg: fld44(=fld42 body)==9 && op0_s5==6 && fld64==0.
    if fld42 == 9 && op0_s5 == 6 && (w >> 11) & 1 == 0 {
        return Real(Op::Neg { r, t });
    }
    // sra: fld45(=((w>>12)&1)<<2 | (w>>9)&3)==5 && op0_s5==6 && fld66(=((w>>11)&1)<<1|(w>>8)&1)==0.
    let fld45 = (((w >> 12) & 1) << 2) | ((w >> 9) & 3);
    if fld45 == 5 && op0_s5 == 6 && ((((w >> 11) & 1) << 1) | ((w >> 8) & 1)) == 0 {
        return Unmapped("sra");
    }
    // srl: fld47(=((w>>12)&1)<<1 | (w>>10)&1)==3 && op0_s5==6 && fld68(=((w>>11)&1)<<2|(w>>8)&3)==0.
    let fld47 = (((w >> 12) & 1) << 1) | ((w >> 10) & 1);
    if fld47 == 3 && op0_s5 == 6 && ((((w >> 11) & 1) << 2) | ((w >> 8) & 3)) == 0 {
        return Real(Op::Srl { r, t });
    }
    // srai: op0_s5==7. imm5 = sargt = (w>>8)&0x1f.
    if op0_s5 == 7 {
        return Real(Op::Srai { r, t, imm: ((w >> 8) & 0x1f) as u8 });
    }
    Inert // excw (default)
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_l32r_bundle_collapsing_excw() {
        // Exception dispatcher entry @ runtime 0x28b4: an xt_format2 bundle
        // `ff f8 81 71 4e ff 68 51`. Oracle (xtdis, config-exact):
        //   slot0: l32r a15, <lit @ 0xfffe3094>
        //   slot1: excw (inert)
        // Collapses to Op::L32r{t:15, target:0xfffe3094}, len 8.
        let d = decode(&[0xff, 0xf8, 0x81, 0x71, 0x4e, 0xff, 0x68, 0x51], 0x28b4);
        assert_eq!(d.len, 8, "xt_format2 bundle is 8 bytes");
        assert!(matches!(d.op, Op::L32r { t: 15, target: 0xfffe_3094 }), "got {:?}", d.op);
    }

    #[test]
    fn xt_format1_all_inert_collapses_to_nop() {
        // An all-`excw` xt_format1 bundle (all three slots inert) is a pure
        // 8-byte NOP barrier -> collapses to Op::Nop, len 8. The boot path hits
        // exactly this at phys 0xd903 (`fe 0c 02 1d f0 36 41 00`, atop a
        // context-save routine); xtdis: slot0/1/2 all excw.
        let d = decode(&[0xfe, 0x0c, 0x02, 0x1d, 0xf0, 0x36, 0x41, 0x00], 0xd903);
        assert_eq!(d.len, 8);
        assert!(matches!(d.op, Op::Nop), "got {:?}", d.op);
        // A trivial all-zero-slot bundle likewise collapses to Nop.
        let d = decode(&[0x0e, 0, 0, 0, 0, 0, 0, 0], 0);
        assert_eq!(d.len, 8);
        assert!(matches!(d.op, Op::Nop), "got {:?}", d.op);
    }

    #[test]
    fn truncated_bundle_does_not_panic() {
        // Fewer than 8 bytes at a 0xf selector -> Unknown, no out-of-bounds.
        let d = decode(&[0xff, 0xf8, 0x81], 0x28b4);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    // ---- xt_format1 differential test against the xtdis oracle ----
    //
    // The decoder above is ported from `xtensa-modules.c`; this test proves the
    // port bit-exact by sweeping tens of thousands of pseudo-random format1
    // bundles through BOTH the config-exact `xtdis` oracle and our per-slot
    // decoders, asserting the same inert/real classification, mnemonic, and
    // operand values on every slot. Gated on the (gitignored) xtdis binary --
    // skips cleanly where it's absent (CI).

    use super::{slot0, slot1, slot2, SlotDecode};

    fn xtdis_path() -> Option<std::path::PathBuf> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("build/experiments/firmware-re/xtdis/xtdis");
        p.exists().then_some(p)
    }

    /// (mnemonic, operand values) for a decoded slot op, in xtdis's print order.
    /// Registers -> their index; immediates/targets -> their value (compared
    /// mod 2^32 so sign-extension and address forms align).
    fn op_sig(op: &Op) -> (&'static str, Vec<u32>) {
        use Op::*;
        let rrr = |m, r: &u8, s: &u8, t: &u8| (m, vec![*r as u32, *s as u32, *t as u32]);
        match op {
            Add { r, s, t } => rrr("add", r, s, t),
            Sub { r, s, t } => rrr("sub", r, s, t),
            Addx2 { r, s, t } => rrr("addx2", r, s, t),
            Addx4 { r, s, t } => rrr("addx4", r, s, t),
            Addx8 { r, s, t } => rrr("addx8", r, s, t),
            Subx8 { r, s, t } => rrr("subx8", r, s, t),
            And { r, s, t } => rrr("and", r, s, t),
            Or { r, s, t } => rrr("or", r, s, t),
            Xor { r, s, t } => rrr("xor", r, s, t),
            Min { r, s, t } => rrr("min", r, s, t),
            Minu { r, s, t } => rrr("minu", r, s, t),
            Maxu { r, s, t } => rrr("maxu", r, s, t),
            Moveqz { r, s, t } => rrr("moveqz", r, s, t),
            Movnez { r, s, t } => rrr("movnez", r, s, t),
            Mull { r, s, t } => rrr("mull", r, s, t),
            Mul16u { r, s, t } => rrr("mul16u", r, s, t),
            Mul16s { r, s, t } => rrr("mul16s", r, s, t),
            Src { r, s, t } => rrr("src", r, s, t),
            Neg { r, t } => ("neg", vec![*r as u32, *t as u32]),
            Abs { r, t } => ("abs", vec![*r as u32, *t as u32]),
            Ssr { s } => ("ssr", vec![*s as u32]),
            Ssl { s } => ("ssl", vec![*s as u32]),
            Ssai { imm } => ("ssai", vec![*imm as u32]),
            Nsau { t, s } => ("nsau", vec![*t as u32, *s as u32]),
            // shift ops: xtdis order = dest, src, amount.
            Slli { r, s, imm } => ("slli", vec![*r as u32, *s as u32, *imm as u32]),
            Srai { r, t, imm } => ("srai", vec![*r as u32, *t as u32, *imm as u32]),
            Srli { r, t, imm } => ("srli", vec![*r as u32, *t as u32, *imm as u32]),
            Sll { r, s } => ("sll", vec![*r as u32, *s as u32]),
            Srl { r, t } => ("srl", vec![*r as u32, *t as u32]),
            Sext { r, s, imm } => ("sext", vec![*r as u32, *s as u32, *imm as u32]),
            Extui { r, t, shiftimm, maskimm } => {
                ("extui", vec![*r as u32, *t as u32, *shiftimm as u32, *maskimm as u32])
            }
            L8ui { t, s, imm } => ("l8ui", vec![*t as u32, *s as u32, *imm]),
            L16ui { t, s, imm } => ("l16ui", vec![*t as u32, *s as u32, *imm]),
            L16si { t, s, imm } => ("l16si", vec![*t as u32, *s as u32, *imm]),
            L32i { t, s, imm } => ("l32i", vec![*t as u32, *s as u32, *imm]),
            S8i { t, s, imm } => ("s8i", vec![*t as u32, *s as u32, *imm]),
            S16i { t, s, imm } => ("s16i", vec![*t as u32, *s as u32, *imm]),
            S32i { t, s, imm } => ("s32i", vec![*t as u32, *s as u32, *imm]),
            Movi { t, imm } => ("movi", vec![*t as u32, *imm as u32]),
            MoviN { t, imm } => ("movi.n", vec![*t as u32, *imm as u32]),
            Addi { t, s, imm } => ("addi", vec![*t as u32, *s as u32, *imm as u32]),
            AddiN { t, s, imm } => ("addi.n", vec![*t as u32, *s as u32, *imm as u32]),
            Addmi { t, s, imm } => ("addmi", vec![*t as u32, *s as u32, *imm as u32]),
            L32r { t, target } => ("l32r", vec![*t as u32, *target]),
            MovN { t, s } => ("mov.n", vec![*t as u32, *s as u32]),
            J { target } => ("j", vec![*target]),
            Jx { s } => ("jx", vec![*s as u32]),
            other => panic!("op_sig: unmapped-in-test op {other:?}"),
        }
    }

    /// Parse one xtdis operand token (`a12`, `0x1f`, `18`) to its numeric value.
    fn parse_operand(tok: &str) -> Option<u32> {
        let tok = tok.trim();
        if let Some(reg) = tok.strip_prefix('a').and_then(|n| n.parse::<u32>().ok()) {
            return Some(reg);
        }
        if let Some(hex) = tok.strip_prefix("0x") {
            return u32::from_str_radix(hex, 16).ok();
        }
        tok.parse::<u32>().ok()
    }

    #[test]
    fn xt_format1_matches_xtdis_oracle() {
        let Some(xtdis) = xtdis_path() else {
            eprintln!("skip: xtdis oracle not present");
            return;
        };
        // Deterministic LCG (numerical-recipes constants) -- no Math.random,
        // fully reproducible.
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 32) as u32
        };

        const CHUNK: usize = 2000; // bundles per xtdis invocation (argv-safe)
        const CHUNKS: usize = 20; // 40k bundles total
        let mut compared = 0usize;
        for _ in 0..CHUNKS {
            // Build a chunk of 8-byte bundles, each forced to op0 nibble 0xe.
            let mut bytes = Vec::with_capacity(CHUNK * 8);
            for _ in 0..CHUNK {
                let mut b = [0u8; 8];
                for x in b.iter_mut() {
                    *x = (next() & 0xff) as u8;
                }
                b[0] = (b[0] & 0xf0) | 0x0e;
                bytes.extend_from_slice(&b);
            }
            let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            let out = std::process::Command::new(&xtdis)
                .arg("0x0")
                .arg(&hex)
                .output()
                .expect("run xtdis");
            let text = String::from_utf8_lossy(&out.stdout);

            // Parse xtdis: each bundle is a header line "ADDR: bytes [xt_format1,
            // 3 slots]" followed by three "slotN: ..." lines.
            let lines: Vec<&str> = text.lines().collect();
            let mut i = 0;
            while i < lines.len() {
                let Some(colon) = lines[i].find(':') else {
                    i += 1;
                    continue;
                };
                let Ok(addr) = u32::from_str_radix(lines[i][..colon].trim(), 16) else {
                    i += 1;
                    continue;
                };
                if !lines[i].contains("xt_format1") {
                    i += 1;
                    continue;
                }
                let idx = (addr / 8) as usize;
                // xtdis's main output prints the LOCATION-INDEPENDENT decoded
                // operand (`xtensa_operand_decode`), NOT the pc-relative target
                // (`undo_reloc`) -- so l32r/j print the raw offset. Pass pc=0 so
                // our decoder's (pc+3)&~3 / pc+4 anchoring reduces to the same
                // offset; the pc-addition itself is the shared, separately
                // verified `decode_ri16`/`J` formula, not what's under test here.
                let pc = 0u32;
                let base = idx * 8;
                let insn0 = u32::from_le_bytes(bytes[base..base + 4].try_into().unwrap());
                let insn1 = u32::from_le_bytes(bytes[base + 4..base + 8].try_into().unwrap());
                let s0 = (insn0 >> 4) & 0xff_ffff;
                let s1 = ((insn0 >> 28) & 0xf) | ((insn1 & 0xffff) << 4);
                let s2 = (insn1 >> 16) & 0xffff;
                let mine = [slot0(s0, pc), slot1(s1, pc), slot2(s2)];

                for (slot, decode_res) in mine.iter().enumerate() {
                    let line = lines[i + 1 + slot];
                    // "slotN: mnem op op ..." possibly with a trailing [NOP].
                    let body = line.split(':').nth(1).unwrap_or("").replace("[NOP]", "");
                    let mut toks = body.split_whitespace();
                    let mnem = toks.next().unwrap_or("excw");
                    let ora_inert = mnem == "excw" || mnem == "nop";
                    match decode_res {
                        SlotDecode::Inert => assert!(
                            ora_inert,
                            "bundle {hex_dbg} slot{slot}: I say inert, xtdis says '{line}'",
                            hex_dbg = &hex[base * 2..base * 2 + 16]
                        ),
                        SlotDecode::Unmapped(m) => assert!(
                            !ora_inert && mnem == *m,
                            "bundle slot{slot}: I say unmapped '{m}', xtdis says '{line}'"
                        ),
                        SlotDecode::Real(op) => {
                            let (my_mnem, my_ops) = op_sig(op);
                            assert!(
                                !ora_inert,
                                "bundle slot{slot}: I say real {my_mnem}, xtdis says inert '{line}'"
                            );
                            assert_eq!(
                                my_mnem, mnem,
                                "bundle slot{slot} @{addr:#x}: mnemonic mine={my_mnem} xtdis='{line}'"
                            );
                            let ora_ops: Vec<u32> = toks.filter_map(parse_operand).collect();
                            assert_eq!(
                                my_ops, ora_ops,
                                "bundle slot{slot} @{addr:#x} {my_mnem}: operands mine={my_ops:?} xtdis='{line}'"
                            );
                            compared += 1;
                        }
                    }
                }
                i += 4;
            }
        }
        eprintln!("xt_format1 vs xtdis: {compared} real-slot ops compared across {} bundles", CHUNK * CHUNKS);
        assert!(compared > 500, "too few real ops exercised ({compared}) -- oracle parse broken?");
    }
}
