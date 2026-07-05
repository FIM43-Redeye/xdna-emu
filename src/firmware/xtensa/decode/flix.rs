//! FLIX (VLIW) bundle decode. This custom AMD Xtensa core packs a 3-slot or
//! 2-slot VLIW bundle behind two `op0` selectors: `op0` nibble (`byte0 & 0xf`)
//! `0xe` -> `xt_format1` (8 bytes, 3 slots), `0xf` -> `xt_format2` (8 bytes,
//! 2 slots). Only `xt_format2` occurs in this firmware (census: ~4-6 bundles,
//! zero `xt_format1`), so `xt_format1` walls as `Op::Unknown` in the caller.
//!
//! DERIVED FROM THE TOOLCHAIN. Every bit position here is derived from this
//! core's real ISA config (`binutils-2.37/bfd/xtensa-modules.c`, the
//! auto-generated FLIX tables) and byte-verified against the `xtdis` oracle
//! (`build/experiments/firmware-re/xtdis/`, which links that config through
//! libisa). The config-match is confirmed exact -- see
//! `docs/superpowers/findings/2026-07-05-iter8-flix-bundle-oracle.md` and the
//! design spec `docs/superpowers/specs/2026-07-05-m2c-flix-bundle-decode-design.md`.
//!
//! **Collapse model.** Every observed `xt_format2` bundle is exactly one real
//! op plus one inert slot (`excw` -- Exception Wait, a barrier with no
//! register/memory/PC effect -> a functional nop here -- or the slot's
//! designated `nop`). Because one slot is always inert, bundle execution is
//! never actually parallel, so we collapse the bundle to its single real `Op`
//! with `len = 8` and let the existing per-op `interp` execute it unchanged.
//! Two real ops in one bundle never occurs and would wall (`Op::Unknown`)
//! rather than silently drop a slot.
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
    fn xt_format1_bundle_walls() {
        // op0 == 0xe (xt_format1, 3-slot) never occurs in this firmware -- it
        // must wall as Unknown with the full 8-byte length consumed, not be
        // mistaken for a 1- or 3-byte op.
        let d = decode(&[0x0e, 0, 0, 0, 0, 0, 0, 0], 0);
        assert_eq!(d.len, 8);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn truncated_bundle_does_not_panic() {
        // Fewer than 8 bytes at a 0xf selector -> Unknown, no out-of-bounds.
        let d = decode(&[0xff, 0xf8, 0x81], 0x28b4);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }
}
