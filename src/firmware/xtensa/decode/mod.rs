//! Xtensa instruction decoder. Base ISA + the windowed-call ops the firmware
//! uses. DERIVED FROM THE TOOLCHAIN: every opcode here has a test vector taken
//! from the real firmware, disassembled by xtensa-lx106-elf-objdump (base) or
//! the captured Ghidra listing.txt (windowed ops -- lx106 objdump cannot
//! decode these; verified empirically, it prints `excw` regardless of the
//! actual bits).
//!
//! Split into per-category modules (`arith`/`mem`/`branch`/`control`/
//! `system`) as a scaffold for M2a's additional opcodes. `Op`/`Decoded` stay
//! here (an enum can't span files); this module keeps the instruction-format
//! skeleton (op0 dispatch, narrow/wide length, field extraction) and
//! delegates leaf construction to each category's `decode_*` helpers, which
//! return `None` when the bits aren't theirs so the next category (or
//! `Op::Unknown`) gets a turn.

mod arith;
mod branch;
mod control;
mod mem;
mod system;

/// A decoded Xtensa instruction: one variant per implemented opcode, each
/// carrying its decoded operands (register indices as `u8`, immediates as
/// `i32`/`u32`, branch/call targets as absolute `u32`). `Unknown` covers
/// anything the decoder doesn't (yet) implement or can't recognize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Entry {
        s: u8,
        imm: u32,
    },
    Call8 {
        target: u32,
    },
    /// Register-indirect windowed call, size 8 (`callx8 as`). Target comes
    /// from AR[s] at execute time; like `call8` it sets PS.CALLINC=2.
    Callx8 {
        s: u8,
    },
    /// Windowed return (`retw`). Restores the window using the call-size in
    /// `a0[31:30]` and returns to `a0[29:0]`.
    Retw,
    /// Narrow (2-byte) windowed return (`retw.n`); same semantics as `retw`.
    RetwN,
    L32iN {
        t: u8,
        s: u8,
        imm: u32,
    },
    MovN {
        t: u8,
        s: u8,
    },
    MoviN {
        t: u8,
        imm: i32,
    },
    Movi {
        t: u8,
        imm: i32,
    },
    L32i {
        t: u8,
        s: u8,
        imm: u32,
    },
    L32r {
        t: u8,
        target: u32,
    },
    Or {
        r: u8,
        s: u8,
        t: u8,
    },
    Extui {
        r: u8,
        t: u8,
        shiftimm: u8,
        maskimm: u8,
    },
    Witlb {
        t: u8,
        s: u8,
    },
    /// Write DTLB entry (`wdtlb at, as`) -- the data-side sibling of `witlb`,
    /// same operand layout. Boot uses it in the MMU-setup sequence.
    Wdtlb {
        t: u8,
        s: u8,
    },
    /// Invalidate ITLB entry (`iitlb as`); single register operand.
    Iitlb {
        s: u8,
    },
    /// Invalidate DTLB entry (`idtlb as`); single register operand.
    Idtlb {
        s: u8,
    },
    /// Write special register (`wsr.<sr> at`): `sr` is the 8-bit special-
    /// register number, `t` the source AR. Boot writes the MMU-config SRs
    /// (ITLBCFG/DTLBCFG/PTEVADDR); the interpreter routes the modeled ones
    /// (PS/VECBASE/WINDOWBASE/...) and logs the rest.
    Wsr {
        sr: u8,
        t: u8,
    },
    Isync,
    /// Data-memory synchronization barrier (`dsync`); no modeled pipeline
    /// effect, treated as a logged no-op like `isync`.
    Dsync,
    /// Jump register (`jx as`): unconditional jump to the address in AR[s].
    Jx {
        s: u8,
    },
    Unknown {
        word: u32,
    },
}

/// The result of decoding one instruction: the operation, plus how many
/// bytes it occupied in the instruction stream (1 for an undecodable op0
/// selector, 2 for narrow `.n` ops, 3 for standard ops).
#[derive(Debug)]
pub struct Decoded {
    pub op: Op,
    pub len: u8,
}

/// Sign-extends the low `bits` bits of `value` to i32, by shifting the field
/// into the top of the word and back down arithmetically. Shared by the
/// category decoders that carry a signed immediate (`arith::decode_rri8`'s
/// `movi`, `control::decode_calln`'s `call8`).
fn sign_extend(value: u32, bits: u32) -> i32 {
    let shift = 32 - bits;
    ((value << shift) as i32) >> shift
}

/// Decodes the instruction at `bytes[0..]`. `pc` resolves PC-relative targets
/// (`l32r`, `call8`). Reads at most 3 bytes, never panics: a short slice or an
/// unrecognized opcode decodes to `Op::Unknown`.
pub fn decode(bytes: &[u8], pc: u32) -> Decoded {
    let Some(&b0) = bytes.first() else {
        return Decoded { op: Op::Unknown { word: 0 }, len: 0 };
    };
    // op0 = low nibble of byte0. Narrow (.n) ops live in 0x8..=0xD (2 bytes);
    // 0x0..=0x7 select a standard 3-byte format. 0xE/0xF aren't valid
    // Xtensa format selectors at all -- treat as a single undecodable byte
    // rather than assuming a (possibly wrong) 3-byte length.
    let op0 = b0 & 0xF;
    if op0 == 0xE || op0 == 0xF {
        return Decoded { op: Op::Unknown { word: b0 as u32 }, len: 1 };
    }
    let narrow = (0x8..=0xD).contains(&op0);
    let need = if narrow { 2 } else { 3 };
    if bytes.len() < need {
        let word = bytes.iter().enumerate().fold(0u32, |w, (i, &b)| w | ((b as u32) << (8 * i)));
        return Decoded { op: Op::Unknown { word }, len: bytes.len() as u8 };
    }
    let b1 = bytes[1];

    if narrow {
        let n1 = (b0 >> 4) & 0xF; // byte0 high nibble
        let n2 = b1 & 0xF; // byte1 low nibble
        let n3 = (b1 >> 4) & 0xF; // byte1 high nibble
        let op = mem::decode_narrow(op0, n1, n2, n3)
            .or_else(|| arith::decode_narrow(op0, n1, n2, n3))
            .or_else(|| control::decode_narrow(op0, n1, n2, n3))
            .unwrap_or(Op::Unknown { word: (b0 as u32) | ((b1 as u32) << 8) });
        return Decoded { op, len: 2 };
    }

    let b2 = bytes[2];
    let word = (b0 as u32) | ((b1 as u32) << 8) | ((b2 as u32) << 16);
    // Nibbles common to the RRR/RRI8/RI16 formats (n1 = "t", n2 = "s",
    // n3 = "r", n4 = op1, n5 = op2). CALLN (op0=5) doesn't align to this --
    // its "n" field is only the low 2 bits of n1 -- so it's handled with its
    // own extraction below.
    let n1 = (b0 >> 4) & 0xF;
    let n2 = b1 & 0xF;
    let n3 = (b1 >> 4) & 0xF;
    let n4 = b2 & 0xF;
    let n5 = (b2 >> 4) & 0xF;

    let op = match op0 {
        // RI16 format: l32r at, <literal>.
        0x1 => mem::decode_ri16(n1, word, pc).unwrap_or(Op::Unknown { word }),
        // RRI8 format (LSAI group): r (n3) selects the sub-op.
        0x2 => mem::decode_rri8(n3, n1, n2, b2)
            .or_else(|| arith::decode_rri8(n3, n1, n2, b2))
            .unwrap_or(Op::Unknown { word }),
        // RRR format (op1 = n4, op2 = n5 select the specific op).
        0x0 => arith::decode_rrr(n4, n5, n3, n2, n1, word)
            .or_else(|| system::decode_rrr(n4, n5, n3, n2, n1, word))
            .or_else(|| control::decode_rrr(n4, n5, n3, n2, n1, word))
            .unwrap_or(Op::Unknown { word }),
        // CALLN format: n (bits 5:4 of byte0) selects call size; only CALL8
        // (n==2) is implemented.
        0x5 => control::decode_calln(b0, word, pc).unwrap_or(Op::Unknown { word }),
        // entry as, imm12*8 (frame size, always a multiple of 8).
        0x6 => control::decode_entry_fmt(n1, n2, n3, b2).unwrap_or(Op::Unknown { word }),
        _ => Op::Unknown { word },
    };
    Decoded { op, len: 3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These test the instruction-format skeleton itself (op0 handling, short
    // slices) rather than any one category's leaf op, so they stay here
    // rather than moving into a category file.

    #[test]
    fn unknown_opcode_is_reported_not_panicked() {
        // 0xff byte region (padding) must not panic.
        let d = decode(&[0xff, 0xff, 0xff], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn undecodable_op0_reports_single_byte_length() {
        // op0 0xE/0xF aren't valid Xtensa format selectors (0x0..=0x7 select
        // the six 3-byte formats, 0x8..=0xD the narrow 2-byte ones) -- a
        // single undecodable byte, not a false 3-byte instruction.
        let d = decode(&[0xff, 0xff, 0xff], 0);
        assert_eq!(d.len, 1);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
        let d = decode(&[0xee, 0x00, 0x00], 0);
        assert_eq!(d.len, 1);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }

    #[test]
    fn short_slice_does_not_panic() {
        // A single byte of what would otherwise be a 2- or 3-byte
        // instruction must decode to Unknown, never index out of bounds.
        let d = decode(&[0x48], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
        let d = decode(&[], 0);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
    }
}
