//! Windowed-call ABI decode: `entry`, `call8`, `callx8`, `retw`, `retw.n`,
//! `jx`, plus the plain (non-windowed) call ABI (`call0`, `ret.n`) and the
//! software window-spill primitives (`rotw`, `waiti`).

use super::{sign_extend, Op};

/// Narrow (2-byte) control op: `retw.n`/`ret.n` (op0=0xD, the ST3.n group:
/// r==0xF; t selects RETW.N (1) vs RET.N (0), s==0 for both). `None`
/// otherwise, so `decode()` falls to `Op::Unknown` (narrow chain: `mem`,
/// `arith`, then this).
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        // r (n3) == 0xF is the ST3.n group: t (n1) selects RET.N (0),
        // RETW.N (1), BREAK.N/NOP.N/etc. Verified via the captured Ghidra
        // listing: `1d f0` -> retw.n. Verified via BOTH oracles for ret.n:
        // xtensa-lx106-elf-objdump decodes `0d f0` -> ret.n directly, and
        // xtensa-modules.c's Opcode_ret_n_Slot_inst16b_encode (0xf00d, i.e.
        // byte0=0x0d/byte1=0xf0) confirms the t==0 encoding.
        0xD if n3 == 0xF && n1 == 1 && n2 == 0 => Some(Op::RetwN),
        0xD if n3 == 0xF && n1 == 0 && n2 == 0 => Some(Op::RetN),
        _ => None,
    }
}

/// RRR format: the JR/CALLX/RETW/WAITI group (op1==0, op2==0; `r` selects
/// among sub-ops -- `r==0` for `jx`/`callx8`/`retw` (further split by `t`),
/// `r==7` for `waiti`) plus the ROTW sub-op (op1==0, op2==4, r==8). `None`
/// for anything else, so `decode()` falls to `Op::Unknown` (this is tried
/// last, after `arith` and `system`).
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, _word: u32) -> Option<Op> {
    match (op1, op2) {
        // JR/CALLX group (op1==0, op2==0, r==0): t (byte0 high nibble)
        // packs the (n,m) selector. t==0xA (n=2,m=2) is JX (jump to AR[s]).
        // Verified via the boot MMU sequence: `a0 03 00` -> jx a3.
        (0x0, 0x0) if t == 0xA && r == 0 => Some(Op::Jx { s }),
        // t==0xE is CALLX8 (m=3,n=2); its target register is s. Verified via
        // the captured Ghidra listing: `e0 09 00` -> callx8 a9, `e0 08 00`
        // -> callx8 a8 (182 instances, target reg = byte1 low nibble).
        (0x0, 0x0) if t == 0xE && r == 0 => Some(Op::Callx8 { s }),
        // retw (t=0x9,s=0,r=0): windowed return. Verified via the captured
        // Ghidra listing: `90 00 00` -> retw (t/s/r all 0).
        (0x0, 0x0) if t == 0x9 && s == 0 && r == 0 => Some(Op::Retw),
        // waiti imm4 (r==7, t==0 fixed, s carries the plain interrupt
        // level): a DIFFERENT sub-selector within the same op1=0,op2=0
        // dispatch family as jx/callx8/retw (which all fix r==0). Verified
        // via BOTH oracles: xtensa-lx106-elf-objdump decodes `00 70 00` ->
        // waiti 0 directly, and xtensa-modules.c's op1=0,op2=0,r=7 dispatch
        // (`Field_t_Slot_inst_get(insn)==0 -> 317 /* waiti */`).
        (0x0, 0x0) if r == 7 && t == 0 => Some(Op::Waiti { imm: s as u32 }),
        // rotw simm4 (r==8, s==0 fixed, t carries the SIGNED window-rotate
        // delta): shares the op1=0,op2=4 dispatch family with arith.rs's
        // Ssr/Ssl/Ssai/Nsau (r is a sub-opcode selector there too), but
        // rotw is a window-ABI primitive so it lives here.
        // xtensa-lx106-elf-objdump can't decode it (prints `excw`, the same
        // windowed-register-option gap as entry/call8/retw/callx8/jx in
        // this file) -- verified instead against xtensa-modules.c's op1=0
        // table (op2=4, r=8, `Field_s_Slot_inst_get(insn)==0` -> opcode 13,
        // rotw). Vector: `10 80 40` -> rotw 0x1 (t=1 -> simm4=+1).
        (0x0, 0x4) if r == 8 && s == 0 => Some(Op::Rotw { imm: sign_extend(t as u32, 4) }),
        _ => None,
    }
}

/// CALLN format: n (bits 5:4 of byte0, only 2 bits -- doesn't align to the
/// RRR/RRI8 nibble convention) selects call size: CALL0 (n==0, non-windowed,
/// [`Op::Call0`]) or CALL8 (n==2, windowed, [`Op::Call8`]); CALL4/CALL12
/// (n==1/3) don't appear in the firmware and aren't implemented. Both share
/// the identical target formula: `imm18 = word>>6`, PC-relative via the
/// fixed hardware formula `((pc+4)&~3) + (sign_extend(imm18,18)<<2)`.
/// CALL8 verified via the captured Ghidra listing (lx106 objdump cannot
/// decode this -- confirmed empirically, prints `excw`): `e5 20 f9` @ pc
/// 0x3a034 -> call8 0x33244. CALL0 verified via BOTH oracles:
/// xtensa-lx106-elf-objdump decodes it directly (`85 ec ff` @ pc 0 -> call0
/// 0xfffffecc) AND the firmware placement (same bytes @ pc 0xe1ce -> call0
/// 0xe098, matching this exact target formula). `None` if `n` isn't 0 or 2,
/// so `decode()` falls to `Op::Unknown`.
pub(super) fn decode_calln(b0: u8, word: u32, pc: u32) -> Option<Op> {
    let n = (b0 >> 4) & 0x3;
    if n == 0 || n == 2 {
        let imm18 = word >> 6;
        let base = pc.wrapping_add(4) & !3u32;
        let offset = (sign_extend(imm18, 18) << 2) as u32;
        let target = base.wrapping_add(offset);
        Some(if n == 0 {
            Op::Call0 { target }
        } else {
            Op::Call8 { target }
        })
    } else {
        None
    }
}

/// op0=0x6 format: `entry as, imm12*8` (frame size, always a multiple of 8).
/// `sel` (n1) selects this specific op0=6 encoding. Verified via the
/// captured Ghidra listing: `36 41 00` @ pc 0x33244 -> entry a1, 0x20.
/// `None` if `sel != 0x3`, so `decode()` falls to `Op::Unknown`.
pub(super) fn decode_entry_fmt(n1: u8, n2: u8, n3: u8, b2: u8) -> Option<Op> {
    let sel = n1;
    if sel == 0x3 {
        let raw = (n3 as u32) | ((b2 as u32) << 4);
        Some(Op::Entry { s: n2, imm: raw * 8 })
    } else {
        None
    }
}

/// op0=0x6 format: `loop`/`loopnez` zero-overhead-loop setup (`n1==0x7`,
/// i.e. `n==3,m==1` -- the sibling of `entry`'s `n==3,m==0` within the SI
/// format's `n==3` group; see `decode/branch.rs`'s module doc for the full
/// `n`/`m` map). `r` (n3) selects `loop`(0x8, [`Op::Loop`]) /
/// `loopnez`(0x9, [`Op::Loopnez`]); `loopgtz` (0xA) does not appear anywhere
/// in the firmware (zero occurrences in the captured Ghidra `listing.txt`),
/// so it's deliberately left unclaimed, falling through to `Op::Unknown`
/// like any other unimplemented opcode -- not a decode gap. `s` = n2 (byte1
/// low nibble); `imm8` = b2, UNSIGNED (Xtensa loops only ever span
/// forward). `end` (the absolute LEND) is `pc + 4 + imm8` -- see
/// [`Op::Loop`]'s doc for the LBEG/LEND asymmetry this must NOT collapse
/// into `pc + 3 + imm8`. `None` if `n1 != 0x7` or `r` isn't 0x8/0x9, so
/// `decode()` falls to `Op::Unknown`.
pub(super) fn decode_loop_fmt(n1: u8, n2: u8, n3: u8, b2: u8, pc: u32) -> Option<Op> {
    if n1 != 0x7 {
        return None;
    }
    let s = n2;
    let end = pc.wrapping_add(4).wrapping_add(b2 as u32);
    match n3 {
        0x8 => Some(Op::Loop { s, end }),
        0x9 => Some(Op::Loopnez { s, end }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

    #[test]
    fn decodes_entry() {
        let d = decode(&[0x36, 0x41, 0x00], 0x33244);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Entry { s: 1, imm: 32 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_call8() {
        let d = decode(&[0xe5, 0x20, 0xf9], 0x3a034);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Call8 { target: 0x33244 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_callx8() {
        // Ghidra listing: `e0 09 00` @ any pc -> callx8 a9 (target reg = s).
        let d = decode(&[0xe0, 0x09, 0x00], 0x2838);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Callx8 { s: 9 }), "got {:?}", d.op);
        // Second oracle vector with a different target register, so the s
        // field isn't fixed by one lucky value: `e0 08 00` -> callx8 a8.
        let d = decode(&[0xe0, 0x08, 0x00], 0x2909);
        assert!(matches!(d.op, Op::Callx8 { s: 8 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_retw() {
        // Ghidra listing: `90 00 00` -> retw.
        let d = decode(&[0x90, 0x00, 0x00], 0x59f1);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Retw), "got {:?}", d.op);
    }

    #[test]
    fn decodes_retw_n() {
        // Ghidra listing: `1d f0` -> retw.n (narrow, 2 bytes).
        let d = decode(&[0x1d, 0xf0], 0x27a8);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::RetwN), "got {:?}", d.op);
    }

    #[test]
    fn ret_n_is_not_misdecoded_as_retw_n() {
        // `0d f0` -> ret.n (the n1==0 sibling of retw.n), now implemented as
        // Op::RetN -- and it must stay distinct from Op::RetwN (never
        // silently aliased to it, which would corrupt the window-restore
        // delta on a real retw.n).
        let d = decode(&[0x0d, 0xf0], 0xe173);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::RetN), "got {:?}", d.op);
        assert!(!matches!(d.op, Op::RetwN));
    }

    #[test]
    fn decodes_jx() {
        // Boot MMU sequence: `a0 03 00` @ 0x399 -> jx a3 (JR group, n1==0xA).
        let d = decode(&[0xa0, 0x03, 0x00], 0x399);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Jx { s: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn jx_is_not_misdecoded_as_callx8() {
        // callx8 (`e0 09 00`, n1==0xE) and jx (`a0 03 00`, n1==0xA) share the
        // JR/CALLX group -- the n1 selector must keep them distinct.
        assert!(matches!(decode(&[0xe0, 0x09, 0x00], 0).op, Op::Callx8 { s: 9 }));
        assert!(matches!(decode(&[0xa0, 0x03, 0x00], 0).op, Op::Jx { s: 3 }));
    }

    #[test]
    fn decodes_loop() {
        // Ghidra listing.txt: `76 84 07` @ 0x3f8d -> loop a4, 0x3f98 (imm8=7,
        // LEND = 0x3f8d + 4 + 7 = 0x3f98).
        let d = decode(&[0x76, 0x84, 0x07], 0x3f8d);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Loop { s: 4, end: 0x3f98 }), "got {:?}", d.op);
    }

    #[test]
    fn loop_lend_is_pc_plus_4_plus_imm8_not_pc_plus_3_plus_imm8() {
        // Regression guard for the deliberate LBEG/LEND asymmetry: LEND must
        // be pc+4+imm8, NOT pc+3+imm8 -- the two formulas diverge here
        // (imm8=7 > 0), so this pins the correct one and would fail if a
        // future edit "simplified" the formula to match LBEG's pc+3.
        let d = decode(&[0x76, 0x84, 0x07], 0x3f8d);
        let wrong_end = 0x3f8d + 3 + 7;
        match d.op {
            Op::Loop { end, .. } => {
                assert_eq!(end, 0x3f8d + 4 + 7);
                assert_ne!(end, wrong_end, "LEND must not be pc+3+imm8");
            }
            other => panic!("expected Op::Loop, got {:?}", other),
        }
    }

    #[test]
    fn decodes_loopnez() {
        // Ghidra listing.txt: `76 93 05` @ 0x3b81d -> loopnez a3, 0x3b826
        // (imm8=5, LEND = 0x3b81d + 4 + 5 = 0x3b826).
        let d = decode(&[0x76, 0x93, 0x05], 0x3b81d);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Loopnez { s: 3, end: 0x3b826 }), "got {:?}", d.op);
    }

    #[test]
    fn loop_family_does_not_shadow_entry_or_bltui() {
        // n==3 is shared by entry (m==0), loop/loopnez (m==1, this task),
        // and bltui/bgeui (m==2/3) -- proves decode_loop_fmt's n1==0x7 guard
        // doesn't accidentally swallow the sibling m values. Reuses the
        // entry oracle vector (`36 41 00`, n1==0x3) and the bltui oracle
        // vector (`b6 66 02`, n1==0xB -- n==3,m==2) from decode/branch.rs.
        let d = decode(&[0x36, 0x41, 0x00], 0x33244);
        assert!(matches!(d.op, Op::Entry { s: 1, imm: 32 }), "got {:?}", d.op);
        let d = decode(&[0xb6, 0x66, 0x02], 0x31);
        assert!(matches!(d.op, Op::Bltui { .. }), "got {:?}", d.op);
    }

    #[test]
    fn loopgtz_is_left_unimplemented() {
        // r(n3)==0xA is loopgtz -- absent from the firmware entirely (zero
        // occurrences in the captured Ghidra listing.txt), so it's
        // deliberately left as Op::Unknown, not silently misdecoded as
        // Loop/Loopnez. Same n1/s as the loop oracle vector, r flipped to 0xA.
        let d = decode(&[0x76, 0xa4, 0x07], 0x3f8d);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_rotw() {
        // xtensa-modules.c oracle (objdump can't decode it, same windowed-
        // register-option gap as entry/retw): `10 80 40` -> rotw 0x1.
        let d = decode(&[0x10, 0x80, 0x40], 0x1234);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rotw { imm: 1 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_waiti() {
        // Both oracles agree: xtensa-lx106-elf-objdump decodes `00 70 00`
        // directly as `waiti 0`.
        let d = decode(&[0x00, 0x70, 0x00], 0x1234);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Waiti { imm: 0 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_call0() {
        // Firmware placement: `85 ec ff` @ pc 0xe1ce -> call0 0xe098 (same
        // ((pc+4)&~3) + (sign_extend(imm18,18)<<2) formula as call8).
        let d = decode(&[0x85, 0xec, 0xff], 0xe1ce);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Call0 { target: 0xe098 }), "got {:?}", d.op);
    }

    #[test]
    fn call0_is_not_misdecoded_as_call8() {
        // n==0 (call0) and n==2 (call8) share the CALLN format -- the n
        // selector must keep them distinct rather than collapsing to one.
        let d = decode(&[0x85, 0xec, 0xff], 0xe1ce);
        assert!(matches!(d.op, Op::Call0 { .. }), "got {:?}", d.op);
        let d = decode(&[0xe5, 0x20, 0xf9], 0x3a034);
        assert!(matches!(d.op, Op::Call8 { .. }), "got {:?}", d.op);
    }
}
