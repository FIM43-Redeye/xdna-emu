//! Xtensa instruction decoder. Base ISA + the windowed-call ops the firmware
//! uses. DERIVED FROM THE TOOLCHAIN: every opcode here has a test vector taken
//! from the real firmware, disassembled by xtensa-lx106-elf-objdump (base) or
//! the captured Ghidra listing.txt (windowed ops -- lx106 objdump cannot
//! decode these; verified empirically, it prints `excw` regardless of the
//! actual bits).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    Entry { s: u8, imm: u32 },
    Call8 { target: u32 },
    L32iN { t: u8, s: u8, imm: u32 },
    MovN { t: u8, s: u8 },
    MoviN { t: u8, imm: i32 },
    Movi { t: u8, imm: i32 },
    L32i { t: u8, s: u8, imm: u32 },
    L32r { t: u8, target: u32 },
    Or { r: u8, s: u8, t: u8 },
    Extui { r: u8, t: u8, shiftimm: u8, maskimm: u8 },
    Witlb { t: u8, s: u8 },
    Isync,
    Unknown { word: u32 },
}

#[derive(Debug)]
pub struct Decoded {
    pub op: Op,
    pub len: u8,
}

/// Sign-extends the low `bits` bits of `value` to i32, by shifting the field
/// into the top of the word and back down arithmetically.
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
    // everything else is a standard 3-byte instruction.
    let op0 = b0 & 0xF;
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
        let op = match op0 {
            // l32i.n at,as,imm4*4 -- verified: `48 45` -> l32i.n a4,a5,0x10
            0x8 => Op::L32iN { t: n1, s: n2, imm: (n3 as u32) * 4 },
            // op0=0xC narrow-immediate family: n1 (byte0 high nibble)
            // selects MOVI.N (0x0..=0x7) vs BEQZ.N/BNEZ.N (0x8..=0xF, not
            // implemented). The 7-bit signed field is (n1<<4)|n3, wrapping
            // past 96 -- verified by sweeping xtensa-lx106-elf-objdump across
            // the full n1/n3 range (0x60 -> -32 ... 0x7f -> -1).
            0xC if n1 <= 0x7 => {
                let raw = ((n1 as u32) << 4) | (n3 as u32);
                let imm = if raw < 96 { raw as i32 } else { raw as i32 - 128 };
                Op::MoviN { t: n2, imm }
            }
            // op0=0xD family: r (byte1 high nibble) == 0 selects MOV.N;
            // other r values are RET.N/RETW.N/BREAK.N/etc (not implemented).
            0xD if n3 == 0 => Op::MovN { t: n1, s: n2 },
            _ => Op::Unknown { word: (b0 as u32) | ((b1 as u32) << 8) },
        };
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
        // RI16 format: l32r at, <literal>. imm16 = word>>8; target uses the
        // fixed hardware formula (the literal is always addressed as if
        // "behind" pc): ((pc+3)&~3) + (0xFFFF0000 | (imm16<<2)). Verified:
        // `21 bd e7` @ pc 0x33262 -> l32r a2, 0x2d158 (xtensa-lx106-elf-objdump).
        0x1 => {
            let imm16 = word >> 8;
            let base = pc.wrapping_add(3) & !3u32;
            let target = base.wrapping_add(0xFFFF_0000u32 | (imm16 << 2));
            Op::L32r { t: n1, target }
        }
        // RRI8 format (LSAI group): r (n3) selects the sub-op.
        0x2 => match n3 {
            // l32i at,as,imm8*4 -- verified: `52 22 0a` -> l32i a5,a2,40
            0x2 => Op::L32i { t: n1, s: n2, imm: (b2 as u32) * 4 },
            // movi at,imm12 -- here "s" (n2) is not a register, it's the
            // immediate's high 4 bits. Verified: `d2 a0 ac` -> movi a13,172
            0xA => {
                let raw12 = ((n2 as u32) << 8) | (b2 as u32);
                Op::Movi { t: n1, imm: sign_extend(raw12, 12) }
            }
            _ => Op::Unknown { word },
        },
        // RRR format (op1/op2 select the specific op).
        0x0 => match (n4, n5) {
            // or ar,as,at -- verified: `20 a2 20` -> or a10,a2,a2
            (0x0, 0x2) => Op::Or { r: n3, s: n2, t: n1 },
            // witlb as,at (write ITLB entry) -- verified: `70 64 50` ->
            // witlb a7,a4
            (0x0, 0x5) if n3 == 0x6 => Op::Witlb { t: n1, s: n2 },
            // isync -- verified: `00 20 00` -> isync (t/s unused, always 0)
            (0x0, 0x0) if n3 == 0x2 && n1 == 0 && n2 == 0 => Op::Isync,
            // extui ar,at,shiftimm,maskimm -- op1==4 selects EXTUI regardless
            // of op2 (op2 IS data here, not a further selector). Register
            // roles differ from the op1==0 group above: t (n1) = dest, r
            // (n3) = src; s (n2) = shiftimm (0-15); op2 (n5) = maskimm-1
            // (1-16). Verified: `30 30 f4` -> extui a3,a3,0,16.
            (0x4, maskimm_m1) => Op::Extui { r: n3, t: n1, shiftimm: n2, maskimm: maskimm_m1 + 1 },
            _ => Op::Unknown { word },
        },
        // CALLN format: n (bits 5:4 of byte0, only 2 bits -- doesn't align to
        // the n1 nibble) selects call size; we only implement CALL8 (n==2).
        // imm18 = word>>6, PC-relative via the fixed hardware formula:
        // ((pc+4)&~3) + (sign_extend(imm18,18)<<2). Verified via the
        // captured Ghidra listing (lx106 objdump cannot decode this --
        // confirmed empirically, prints `excw`): `e5 20 f9` @ pc 0x3a034 ->
        // call8 0x33244.
        0x5 => {
            let n = (b0 >> 4) & 0x3;
            if n == 2 {
                let imm18 = word >> 6;
                let base = pc.wrapping_add(4) & !3u32;
                let offset = (sign_extend(imm18, 18) << 2) as u32;
                Op::Call8 { target: base.wrapping_add(offset) }
            } else {
                Op::Unknown { word }
            }
        }
        // entry as, imm12*8 (frame size, always a multiple of 8). sel (n1)
        // selects this specific op0=6 encoding. Verified via the captured
        // Ghidra listing: `36 41 00` @ pc 0x33244 -> entry a1, 0x20.
        0x6 => {
            let sel = n1;
            if sel == 0x3 {
                let raw = (n3 as u32) | ((b2 as u32) << 4);
                Op::Entry { s: n2, imm: raw * 8 }
            } else {
                Op::Unknown { word }
            }
        }
        _ => Op::Unknown { word },
    };
    Decoded { op, len: 3 }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Each case: (bytes, pc, expected Op, expected len). Vectors derived from
    // the real firmware via objdump (base ISA) / Ghidra listing.txt (windowed).
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
    fn decodes_l32i_n() {
        let d = decode(&[0x48, 0x45], 0x33259);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::L32iN { t: 4, s: 5, imm: 16 }), "got {:?}", d.op);
    }

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
    fn decodes_witlb() {
        let d = decode(&[0x70, 0x64, 0x50], 0x33290);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Witlb { t: 7, s: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_isync() {
        let d = decode(&[0x00, 0x20, 0x00], 0x332a0);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Isync), "got {:?}", d.op);
    }

    #[test]
    fn unknown_opcode_is_reported_not_panicked() {
        // 0xff byte region (padding) must not panic.
        let d = decode(&[0xff, 0xff, 0xff], 0);
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
