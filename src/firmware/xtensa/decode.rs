//! Xtensa instruction decoder. Base ISA + the windowed-call ops the firmware
//! uses. DERIVED FROM THE TOOLCHAIN: every opcode here has a test vector taken
//! from the real firmware, disassembled by xtensa-lx106-elf-objdump (base) or
//! the captured Ghidra listing.txt (windowed ops -- lx106 objdump cannot
//! decode these; verified empirically, it prints `excw` regardless of the
//! actual bits).

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
            // op0=0xD family: r (byte1 high nibble) == 0 selects MOV.N.
            0xD if n3 == 0 => Op::MovN { t: n1, s: n2 },
            // r (n3) == 0xF is the ST3.n group: t (n1) selects RET.N (0),
            // RETW.N (1), BREAK.N/NOP.N/etc. Only RETW.N is implemented (its
            // s field, n2, is 0). Verified via the captured Ghidra listing:
            // `1d f0` -> retw.n (and `0d f0` -> ret.n, the n1==0 sibling we
            // leave to Op::Unknown).
            0xD if n3 == 0xF && n1 == 1 && n2 == 0 => Op::RetwN,
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
        // RI16 format: l32r at, <literal>. imm16 = word>>8 is a negative
        // WORD offset -- sign-extend it to 32 bits FIRST (OR with
        // 0xFFFF0000), THEN <<2 to convert words->bytes:
        // ((pc+3)&~3) + ((0xFFFF0000 | imm16) << 2). Order matters: OR-then-
        // shift, not shift-then-OR -- verified against all 2829 l32r
        // instructions in the real firmware (xtensa-lx106-elf-objdump); e.g.
        // `21 bd e7` @ pc 0x33262 -> l32r a2, 0x2d158.
        0x1 => {
            let imm16 = word >> 8;
            let base = pc.wrapping_add(3) & !3u32;
            let target = base.wrapping_add((0xFFFF_0000u32 | imm16) << 2);
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
            // TLB-access group (op1==0, op2==5): r (n3) selects the specific
            // op -- 0x4 iitlb, 0x6 witlb, 0xC idtlb, 0xE wdtlb. Verified via
            // the boot MMU-setup sequence: `00 45 50` -> iitlb a5, `70 64 50`
            // -> witlb a7,a4, `00 c5 50` -> idtlb a5, `70 e4 50` -> wdtlb a7,a4.
            (0x0, 0x5) if n3 == 0x4 => Op::Iitlb { s: n2 },
            (0x0, 0x5) if n3 == 0x6 => Op::Witlb { t: n1, s: n2 },
            (0x0, 0x5) if n3 == 0xC => Op::Idtlb { s: n2 },
            (0x0, 0x5) if n3 == 0xE => Op::Wdtlb { t: n1, s: n2 },
            // SYNC group (op1==0, op2==0, r==2): t (n1) selects isync (0),
            // rsync (1), esync (2), dsync (3). Verified: `00 20 00` -> isync,
            // `30 20 00` -> dsync (t=3). Both are logged no-ops here.
            (0x0, 0x0) if n3 == 0x2 && n1 == 0 && n2 == 0 => Op::Isync,
            (0x0, 0x0) if n3 == 0x2 && n1 == 0x3 && n2 == 0 => Op::Dsync,
            // JR/CALLX group (op1==0, op2==0, r==0): byte0's high nibble n1
            // packs the (n,m) selector. n1==0xA (n=2,m=2) is JX (jump to
            // AR[s], s==n2). Verified via the boot MMU sequence: `a0 03 00`
            // -> jx a3. (n1==0xE is CALLX8, handled below.)
            (0x0, 0x0) if n1 == 0xA && n3 == 0 => Op::Jx { s: n2 },
            // WSR.<sr> at (op1==3, op2==1): the 8-bit SR number is byte1
            // ((r<<4)|s), the source AR is t (n1). Verified via the boot MMU
            // sequence: `20 5b 13` -> wsr.itlbcfg a2, `20 5c 13` ->
            // wsr.dtlbcfg a2, `50 53 13` -> wsr.ptevaddr a5.
            (0x3, 0x1) => Op::Wsr { sr: (word >> 8) as u8, t: n1 },
            // CALLX/RET group (op1==0, op2==0, r==0): the m,n encoded in
            // byte0's high nibble (n1) selects the specific op. n1==0xE is
            // CALLX8 (m=3,n=2); its target register is s (n2). Verified via
            // the captured Ghidra listing: `e0 09 00` -> callx8 a9, `e0 08
            // 00` -> callx8 a8 (182 instances, target reg = byte1 low nibble).
            (0x0, 0x0) if n1 == 0xE && n3 == 0 => Op::Callx8 { s: n2 },
            // retw (m=2,n=1): windowed return. Verified via the captured
            // Ghidra listing: `90 00 00` -> retw (t/s/r all 0).
            (0x0, 0x0) if n1 == 0x9 && n2 == 0 && n3 == 0 => Op::Retw,
            // extui art,ars,shiftimm,maskimm -- op1==4 selects EXTUI
            // regardless of op2 (op2 IS data here, not a further selector).
            // Register fields follow the SAME convention as `or` above: r
            // (n3) = dest, t (n1) = src; s (n2) = shiftimm (0-15); op2 (n5)
            // = maskimm-1 (1-16). Verified: `30 30 f4` -> extui a3,a3,0,16
            // (dest==src in this vector, so it alone can't disambiguate
            // which nibble is which -- confirmed separately by sweeping
            // non-aliased registers against objdump: n1=1,n3=7 -> `extui
            // a7,a1,...`, i.e. n3 is always the first/dest operand).
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
        // `0d f0` -> ret.n (the n1==0 sibling of retw.n). We don't implement
        // ret.n, so it must stay Op::Unknown -- never silently aliased to
        // retw.n, which would corrupt the window-restore delta.
        let d = decode(&[0x0d, 0xf0], 0xe173);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::Unknown { .. }), "got {:?}", d.op);
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
    fn decodes_dsync() {
        // Boot MMU sequence: `30 20 00` @ 0x32e -> dsync (SYNC group, t=3).
        let d = decode(&[0x30, 0x20, 0x00], 0x32e);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Dsync), "got {:?}", d.op);
    }

    #[test]
    fn decodes_wdtlb() {
        // Boot MMU sequence: `70 e4 50` @ 0x342 -> wdtlb a7,a4 (r==0xE).
        let d = decode(&[0x70, 0xe4, 0x50], 0x342);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Wdtlb { t: 7, s: 4 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_iitlb_and_idtlb() {
        // Boot MMU sequence: `00 45 50` -> iitlb a5 (r==0x4); `00 c5 50` ->
        // idtlb a5 (r==0xC). Single register operand s == byte1 low nibble.
        let d = decode(&[0x00, 0x45, 0x50], 0x357);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Iitlb { s: 5 }), "got {:?}", d.op);
        let d = decode(&[0x00, 0xc5, 0x50], 0x35a);
        assert!(matches!(d.op, Op::Idtlb { s: 5 }), "got {:?}", d.op);
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
    fn decodes_wsr_mmu_config_regs() {
        // Boot MMU sequence: SR number is byte1 ((r<<4)|s), source AR is t
        // (byte0 high nibble). `20 5b 13` -> wsr.itlbcfg(0x5B) a2;
        // `20 5c 13` -> wsr.dtlbcfg(0x5C) a2; `50 53 13` -> wsr.ptevaddr(0x53) a5.
        let d = decode(&[0x20, 0x5b, 0x13], 0x322);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Wsr { sr: 0x5b, t: 2 }), "got {:?}", d.op);
        assert!(matches!(decode(&[0x20, 0x5c, 0x13], 0x32b).op, Op::Wsr { sr: 0x5c, t: 2 }));
        assert!(matches!(decode(&[0x50, 0x53, 0x13], 0x334).op, Op::Wsr { sr: 0x53, t: 5 }));
    }

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
