//! System/MMU-config decode: `wsr.<sr>`, `rsr.<sr>`, `wur`, `isync`, `dsync`,
//! `rsync`, `memw`, `nop`/`nop.n`, `rsil`, `syscall`, `witlb`, `wdtlb`,
//! `iitlb`, `idtlb`, and the icache/dcache-maintenance group (`dhwbi`/`dhi`/
//! `dii`/`ihi` plus the M2c Task 6a completion of the full 19-op `r==7`
//! group -- see `decode_rri8`'s doc for the complete table).

use super::Op;

/// RRR format: the TLB-access group (op1==0, op2==5; r selects the specific
/// op), the MISC group (op1==0, op2==0, r==2; t selects isync/rsync/dsync/
/// memw/nop), `rsil`/`syscall` (also op1==0, op2==0, distinguished by r),
/// `wsr.<sr> at` (op1==3, op2==1), `rsr.<sr> at` (op1==3, op2==0), and `wur
/// at,<ur>` (op1==3, op2==0xF). `None` for anything else in the RRR format,
/// so `decode()` tries `control` next (`arith` is tried before this).
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
        // MISC group (op1==0, op2==0, r==2): t selects isync (0), rsync (1),
        // esync (2, unimplemented -- absent from this task's opcode list),
        // dsync (3), memw (0xC), extw (0xD, unimplemented), nop (0xF).
        // Verified: `00 20 00` -> isync, `10 20 00` -> rsync, `30 20 00` ->
        // dsync, `c0 20 00` -> memw, `f0 20 00` -> nop (all cross-checked
        // against xtensa-modules.c's Opcode_{isync,rsync,dsync,memw,nop}_
        // Slot_inst_encode templates, which agree on every byte). All are
        // logged no-ops in the interpreter.
        (0x0, 0x0) if r == 0x2 && t == 0 && s == 0 => Some(Op::Isync),
        (0x0, 0x0) if r == 0x2 && t == 0x1 && s == 0 => Some(Op::Rsync),
        (0x0, 0x0) if r == 0x2 && t == 0x3 && s == 0 => Some(Op::Dsync),
        (0x0, 0x0) if r == 0x2 && t == 0xC && s == 0 => Some(Op::Memw),
        (0x0, 0x0) if r == 0x2 && t == 0xF && s == 0 => Some(Op::Nop),
        // RSIL at,imm4 (op1==0, op2==0, r==6): dest AR is t, imm4 is the
        // plain (non-register) s field. Verified against xtensa-modules.c's
        // Opcode_rsil_Slot_inst_encode (`0x006000` -> r=6) AND the firmware
        // vector: `20 62 00` -> rsil a2,0x2.
        (0x0, 0x0) if r == 6 => Some(Op::Rsil { t, imm: s as u32 }),
        // SYSCALL (op1==0, op2==0, r==5, s==0/t==0 fixed -- no operands).
        // Verified against xtensa-modules.c's Opcode_syscall_Slot_inst_encode
        // (`0x005000`) AND the firmware vector: `00 50 00` -> syscall.
        (0x0, 0x0) if r == 5 && s == 0 && t == 0 => Some(Op::Syscall),
        // WSR.<sr> at (op1==3, op2==1): the 8-bit SR number is byte1
        // ((r<<4)|s), the source AR is t. Verified via the boot MMU
        // sequence: `20 5b 13` -> wsr.itlbcfg a2, `20 5c 13` ->
        // wsr.dtlbcfg a2, `50 53 13` -> wsr.ptevaddr a5.
        (0x3, 0x1) => Some(Op::Wsr { sr: (word >> 8) as u8, t }),
        // RSR.<sr> at (op1==3, op2==0): the READ sibling of WSR, same
        // sr-number extraction. Verified against xtensa-modules.c's
        // Opcode_rsr_intenable_Slot_inst_encode (`0x03e400` -> op1=3,op2=0,
        // sr=0xE4) AND the firmware vector: `30 e4 03` -> rsr a3,INTENABLE.
        (0x3, 0x0) => Some(Op::Rsr { sr: (word >> 8) as u8, t }),
        // WUR at,<ur> (op1==3, op2==0xF): write a user (TIE) register --
        // `ur` is byte1 (same convention as the sr field above, but a
        // separate 8-bit namespace). Verified against xtensa-modules.c's
        // Opcode_wur_threadptr_Slot_inst_encode (`0xf3e700` -> op1=3,op2=0xF,
        // byte1=0xE7 -- confirms the field layout bit-for-bit) AND the
        // firmware vector: `30 e7 f3` -> wur a3,VECBASE (see `Op::Wur`'s doc
        // for the naming discrepancy between AMD's vendored generic table
        // and the real firmware's own Ghidra-derived naming).
        (0x3, 0xF) => Some(Op::Wur { ur: (word >> 8) as u8, t }),
        _ => None,
    }
}

/// Narrow (2-byte) `nop.n`: same ST3.n group as `control::decode_narrow`'s
/// `ret.n`/`retw.n` (op0=0xD, r(n3)==0xF), `t(n1)==3`. Kept in this module
/// (rather than extending `control::decode_narrow`'s existing match) since
/// `nop.n` is a system/no-op instruction, not a windowed-call-ABI one.
/// Verified against xtensa-modules.c's `Opcode_nop_n_Slot_inst16b_encode`
/// (`0xf03d` -> byte0=0x3d/byte1=0xf0, i.e. t=3,r=0xF) AND the firmware
/// vector: `3d f0` -> nop.n. `None` otherwise, so `decode()` tries `branch`
/// next (the narrow chain: `mem`, `arith`, `control`, this, `branch`).
pub(super) fn decode_narrow(op0: u8, n1: u8, n2: u8, n3: u8) -> Option<Op> {
    match op0 {
        0xD if n3 == 0xF && n1 == 3 && n2 == 0 => Some(Op::NopN),
        _ => None,
    }
}

/// RRI8 format: the icache/dcache-maintenance group (`r==7`) -- 19 ops
/// total (M2c Task 6a completed the group; M2a had implemented `dhwbi`/
/// `dhi`/`dii`/`ihi`). `s` is always the address register `as`. Two operand
/// shapes, per `xtensa-modules.c`'s `Iclass_xt_iclass_{dpf,dcache,
/// dcache_inv,icache,icache_inv}_args` (flat `t`, `uimm8x4`: imm = `b2*4`)
/// vs `Iclass_xt_iclass_{dcache_lock,dcache_ind,icache_lock}_args` (`t=0x8`/
/// `t=0xD` sub-selected by `b2&0xF`, `uimm4x16`: imm = `(b2>>4)*16`).
///
/// Flat `t` (`uimm8x4`, imm=`b2*4`): 0x0 dpfr, 0x1 dpfw, 0x2 dpfro, 0x3
/// dpfwo (iclass `xt_iclass_dpf`); 0x4 dhwb, 0x5 dhwbi (`xt_iclass_dcache`);
/// 0x6 dhi, 0x7 dii (`xt_iclass_dcache_inv`); 0xC ipf, 0xE ihi
/// (`xt_iclass_icache`); 0xF iii (`xt_iclass_icache_inv`).
///
/// Sub-selected `t=0x8` (`uimm4x16`): `b2&0xF` in `{0:dpfl,2:dhu,3:diu}`
/// (`xt_iclass_dcache_lock`) or `{4:diwb,5:diwbi}` (`xt_iclass_dcache_ind`).
/// Sub-selected `t=0xD` (`uimm4x16`, `xt_iclass_icache_lock`): `b2&0xF` in
/// `{0:ipfl,2:ihu,3:iiu}`.
///
/// `t=9`/`t=0xA`/`t=0xB` and any other `t=0x8`/`t=0xD` sub-selector are
/// UNDEFINED -- deliberately `None` (NOT a no-op), so `decode()` yields
/// `Op::Unknown` rather than silently swallowing an illegal encoding or a
/// future TIE opcode reusing the slot. Verified against xtensa-modules.c's
/// `Opcode_{dpfr,dpfw,dpfro,dpfwo,dhwb,dhwbi,dhi,dii,dpfl,dhu,diu,diwb,
/// diwbi,ipf,ipfl,ihu,iiu,ihi,iii}_Slot_inst_encode` templates (all share
/// byte1=0x70 baseline, i.e. r=7/op1=0/op2=0 with s=0, only `t`/`byte2`
/// differing) AND the firmware reset-head vectors: `52 72 00` -> dhwbi
/// a2,0; `62 72 00` -> dhi a2,0; `72 72 00` -> dii a2,0; `e2 72 00` -> ihi
/// a2,0; `d2 73 03` -> iiu a3,0; `f2 73 00` -> iii a3,0; `82 73 03` -> diu
/// a3,0. `None` for any other `r`, so `decode()` tries `Unknown` (this is
/// the last RRI8 handler tried, after `mem` and `arith`).
pub(super) fn decode_rri8(r: u8, t: u8, s: u8, b2: u8) -> Option<Op> {
    if r != 7 {
        return None;
    }
    let imm8x4 = b2 as u32 * 4;
    match t {
        0x0 => Some(Op::Dpfr { s, imm: imm8x4 }),
        0x1 => Some(Op::Dpfw { s, imm: imm8x4 }),
        0x2 => Some(Op::Dpfro { s, imm: imm8x4 }),
        0x3 => Some(Op::Dpfwo { s, imm: imm8x4 }),
        0x4 => Some(Op::Dhwb { s, imm: imm8x4 }),
        5 => Some(Op::Dhwbi { s, imm: imm8x4 }),
        6 => Some(Op::Dhi { s, imm: imm8x4 }),
        7 => Some(Op::Dii { s, imm: imm8x4 }),
        // t=0x8: byte2 low nibble picks the mnemonic, high nibble is the
        // uimm4x16 immediate (imm4*16) -- a DIFFERENT scaling from the
        // flat-t ops above. Only sub-values {0,2,3,4,5} are defined; every
        // other value (1, 6..=0xF) is an undefined slot and must stay None.
        0x8 => {
            let sub = b2 & 0xF;
            let imm4x16 = ((b2 >> 4) & 0xF) as u32 * 16;
            match sub {
                0 => Some(Op::Dpfl { s, imm: imm4x16 }),
                2 => Some(Op::Dhu { s, imm: imm4x16 }),
                3 => Some(Op::Diu { s, imm: imm4x16 }),
                4 => Some(Op::Diwb { s, imm: imm4x16 }),
                5 => Some(Op::Diwbi { s, imm: imm4x16 }),
                _ => None,
            }
        }
        0xC => Some(Op::Ipf { s, imm: imm8x4 }),
        // t=0xD: same sub-selection shape as t=0x8, but only {0,2,3} defined.
        0xD => {
            let sub = b2 & 0xF;
            let imm4x16 = ((b2 >> 4) & 0xF) as u32 * 16;
            match sub {
                0 => Some(Op::Ipfl { s, imm: imm4x16 }),
                2 => Some(Op::Ihu { s, imm: imm4x16 }),
                3 => Some(Op::Iiu { s, imm: imm4x16 }),
                _ => None,
            }
        }
        0xE => Some(Op::Ihi { s, imm: imm8x4 }),
        0xF => Some(Op::Iii { s, imm: imm8x4 }),
        // t=9/0xA/0xB: no opcode defined at all in this group -- None
        // (Op::Unknown), not a swallowed no-op.
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

    // -- M2a Task 9: system opcodes --------------------------------------

    #[test]
    fn decodes_rsr() {
        // `30 e4 03` -> rsr a3,INTENABLE (sr=0xE4=228).
        let d = decode(&[0x30, 0xe4, 0x03], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rsr { sr: 0xe4, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_wur() {
        // `30 e7 f3` -> wur a3,VECBASE (ur=0xE7).
        let d = decode(&[0x30, 0xe7, 0xf3], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Wur { ur: 0xe7, t: 3 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_rsil() {
        // `20 62 00` -> rsil a2,0x2.
        let d = decode(&[0x20, 0x62, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rsil { t: 2, imm: 2 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_syscall() {
        let d = decode(&[0x00, 0x50, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Syscall), "got {:?}", d.op);
    }

    #[test]
    fn decodes_memw() {
        let d = decode(&[0xc0, 0x20, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Memw), "got {:?}", d.op);
    }

    #[test]
    fn decodes_nop() {
        let d = decode(&[0xf0, 0x20, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Nop), "got {:?}", d.op);
    }

    #[test]
    fn decodes_rsync() {
        let d = decode(&[0x10, 0x20, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Rsync), "got {:?}", d.op);
    }

    #[test]
    fn decodes_nop_n() {
        let d = decode(&[0x3d, 0xf0], 0x1000);
        assert_eq!(d.len, 2);
        assert!(matches!(d.op, Op::NopN), "got {:?}", d.op);
    }

    #[test]
    fn nop_n_is_not_misdecoded_as_ret_n_or_retw_n() {
        // All three share the ST3.n group (op0=0xD, r==0xF); only `t`
        // differs (0=ret.n, 1=retw.n, 3=nop.n) -- proves the discriminator
        // doesn't collapse them.
        assert!(matches!(decode(&[0x0d, 0xf0], 0).op, Op::RetN));
        assert!(matches!(decode(&[0x1d, 0xf0], 0).op, Op::RetwN));
        assert!(matches!(decode(&[0x3d, 0xf0], 0).op, Op::NopN));
    }

    #[test]
    fn decodes_dii_dhi_dhwbi_ihi() {
        let d = decode(&[0x72, 0x72, 0x00], 0x1000);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Dii { s: 2, imm: 0 }), "got {:?}", d.op);

        let d = decode(&[0x62, 0x72, 0x00], 0x1000);
        assert!(matches!(d.op, Op::Dhi { s: 2, imm: 0 }), "got {:?}", d.op);

        let d = decode(&[0x52, 0x72, 0x00], 0x1000);
        assert!(matches!(d.op, Op::Dhwbi { s: 2, imm: 0 }), "got {:?}", d.op);

        let d = decode(&[0xe2, 0x72, 0x00], 0x1000);
        assert!(matches!(d.op, Op::Ihi { s: 2, imm: 0 }), "got {:?}", d.op);
    }

    #[test]
    fn cache_op_immediate_is_scaled_by_4() {
        // Hand-modified byte2 (imm8=1 instead of the firmware's 0) on the
        // dii vector proves the `imm8*4` pre-scaling, not just that a zero
        // immediate round-trips as zero.
        let d = decode(&[0x72, 0x72, 0x01], 0x1000);
        assert!(matches!(d.op, Op::Dii { s: 2, imm: 4 }), "got {:?}", d.op);
    }

    // -- M2c Task 6a: complete the r==7 cache-maintenance group ----------

    #[test]
    fn decodes_firmware_reset_head_cache_ops() {
        // The exact bytes the reset head's cache-init loops execute (M2c
        // Task 6, walled on `iiu` until this task). Verified against the
        // firmware image at these file offsets (see task-6a-brief.md).
        let d = decode(&[0xd2, 0x73, 0x03], 0x23d);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Iiu { s: 3, imm: 0 }), "got {:?}", d.op);

        let d = decode(&[0xf2, 0x73, 0x00], 0x251);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Iii { s: 3, imm: 0 }), "got {:?}", d.op);

        let d = decode(&[0x82, 0x73, 0x03], 0x265);
        assert_eq!(d.len, 3);
        assert!(matches!(d.op, Op::Diu { s: 3, imm: 0 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_sub_selected_uimm4x16_immediate() {
        // byte2=0x23 -> sub=3 (iiu), imm=(0x2)*16=32. Second real firmware
        // vector (`0x240`), proves the uimm4x16 scaling (not uimm8x4).
        let d = decode(&[0xd2, 0x73, 0x23], 0x240);
        assert!(matches!(d.op, Op::Iiu { s: 3, imm: 32 }), "got {:?}", d.op);

        // byte2=0x13 -> t=8 sub=3 (diu), imm=(0x1)*16=16.
        let d = decode(&[0x82, 0x73, 0x13], 0);
        assert!(matches!(d.op, Op::Diu { s: 3, imm: 16 }), "got {:?}", d.op);
    }

    #[test]
    fn decodes_all_new_r7_cache_ops() {
        // Baseline (s=0, imm=0) encodings for every op added in this task,
        // taken from xtensa-modules.c's Opcode_<mnemonic>_Slot_inst_encode
        // templates (see decode_rri8's doc for the per-op source lines).
        let cases: &[(&[u8], Op)] = &[
            (&[0x02, 0x70, 0x00], Op::Dpfr { s: 0, imm: 0 }),
            (&[0x12, 0x70, 0x00], Op::Dpfw { s: 0, imm: 0 }),
            (&[0x22, 0x70, 0x00], Op::Dpfro { s: 0, imm: 0 }),
            (&[0x32, 0x70, 0x00], Op::Dpfwo { s: 0, imm: 0 }),
            (&[0x42, 0x70, 0x00], Op::Dhwb { s: 0, imm: 0 }),
            (&[0x82, 0x70, 0x00], Op::Dpfl { s: 0, imm: 0 }),
            (&[0x82, 0x70, 0x02], Op::Dhu { s: 0, imm: 0 }),
            (&[0x82, 0x70, 0x03], Op::Diu { s: 0, imm: 0 }),
            (&[0x82, 0x70, 0x04], Op::Diwb { s: 0, imm: 0 }),
            (&[0x82, 0x70, 0x05], Op::Diwbi { s: 0, imm: 0 }),
            (&[0xc2, 0x70, 0x00], Op::Ipf { s: 0, imm: 0 }),
            (&[0xd2, 0x70, 0x00], Op::Ipfl { s: 0, imm: 0 }),
            (&[0xd2, 0x70, 0x02], Op::Ihu { s: 0, imm: 0 }),
            (&[0xd2, 0x70, 0x03], Op::Iiu { s: 0, imm: 0 }),
            (&[0xf2, 0x70, 0x00], Op::Iii { s: 0, imm: 0 }),
        ];
        for (bytes, expected) in cases {
            let d = decode(bytes, 0);
            assert_eq!(d.len, 3);
            assert_eq!(&d.op, expected, "decoding {:?} got {:?}", bytes, d.op);
        }
    }

    #[test]
    fn undefined_r7_slots_stay_unknown_not_no_op() {
        // t=9/0xA/0xB have no opcode at all in the r==7 group.
        for t in [0x9u8, 0xA, 0xB] {
            let d = decode(&[t << 4, 0x70, 0x00], 0);
            assert!(matches!(d.op, Op::Unknown { .. }), "t=0x{:x} got {:?}", t, d.op);
        }
        // t=9 case from the brief, byte2=0x00.
        assert!(matches!(decode(&[0x92, 0x73, 0x00], 0).op, Op::Unknown { .. }));

        // Within t=0x8, only sub {0,2,3,4,5} are defined -- every other
        // nibble value must stay Unknown, not silently collapse to a no-op.
        for sub in [1u8, 6, 7, 8, 9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF] {
            let d = decode(&[0x82, 0x70, sub], 0);
            assert!(matches!(d.op, Op::Unknown { .. }), "t=8 sub=0x{:x} got {:?}", sub, d.op);
        }

        // Within t=0xD, only sub {0,2,3} are defined.
        for sub in [1u8, 4, 5, 6, 7, 8, 9, 0xA, 0xB, 0xC, 0xD, 0xE, 0xF] {
            let d = decode(&[0xd2, 0x70, sub], 0);
            assert!(matches!(d.op, Op::Unknown { .. }), "t=0xD sub=0x{:x} got {:?}", sub, d.op);
        }
        // t=0xD sub=1 case from the brief, s=3.
        assert!(matches!(decode(&[0xd2, 0x73, 0x01], 0).op, Op::Unknown { .. }));
    }
}
