//! System/MMU-config decode: `wsr.<sr>`, `isync`, `dsync`, `witlb`, `wdtlb`,
//! `iitlb`, `idtlb`.

use super::Op;

/// RRR format: the TLB-access group (op1==0, op2==5; r selects the specific
/// op), the SYNC group (op1==0, op2==0, r==2; t selects isync/dsync), and
/// `wsr.<sr> at` (op1==3, op2==1). `None` for anything else in the RRR
/// format, so `decode()` tries `control` next (`arith` is tried before this).
pub(super) fn decode_rrr(op1: u8, op2: u8, r: u8, s: u8, t: u8, word: u32) -> Option<Op> {
    match (op1, op2) {
        // TLB-access group (op1==0, op2==5): r selects the specific op --
        // 0x4 iitlb, 0x6 witlb, 0xC idtlb, 0xE wdtlb. Verified via the boot
        // MMU-setup sequence: `00 45 50` -> iitlb a5, `70 64 50` -> witlb
        // a7,a4, `00 c5 50` -> idtlb a5, `70 e4 50` -> wdtlb a7,a4.
        (0x0, 0x5) if r == 0x4 => Some(Op::Iitlb { s }),
        (0x0, 0x5) if r == 0x6 => Some(Op::Witlb { t, s }),
        (0x0, 0x5) if r == 0xC => Some(Op::Idtlb { s }),
        (0x0, 0x5) if r == 0xE => Some(Op::Wdtlb { t, s }),
        // SYNC group (op1==0, op2==0, r==2): t selects isync (0), rsync (1),
        // esync (2), dsync (3). Verified: `00 20 00` -> isync, `30 20 00` ->
        // dsync (t=3). Both are logged no-ops in the interpreter.
        (0x0, 0x0) if r == 0x2 && t == 0 && s == 0 => Some(Op::Isync),
        (0x0, 0x0) if r == 0x2 && t == 0x3 && s == 0 => Some(Op::Dsync),
        // WSR.<sr> at (op1==3, op2==1): the 8-bit SR number is byte1
        // ((r<<4)|s), the source AR is t. Verified via the boot MMU
        // sequence: `20 5b 13` -> wsr.itlbcfg a2, `20 5c 13` ->
        // wsr.dtlbcfg a2, `50 53 13` -> wsr.ptevaddr a5.
        (0x3, 0x1) => Some(Op::Wsr { sr: (word >> 8) as u8, t }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::decode;
    use super::Op;

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
}
