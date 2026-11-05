//! Per-encoding facts read from TableGen records: the encoding schema.
//!
//! Every [`InstrEncoding`](super::InstrEncoding) carries an [`EncodingSchema`] built at build
//! time from the instruction record and the classes it instantiates. Each field is a TableGen
//! record fact (a DAG, a field, a class the record derives from, a Pat<> it is selected by), never
//! a parse of the instruction name or mnemonic. The schema sits beside the older name-derived
//! fields of `InstrEncoding` (`operand_type`, `addressing_mode`, `mem_width`, ...): consumers
//! switch to it one package at a time, and `tools/datadriven_ledger.py` reports where the two
//! still disagree.

use super::resolver::OperandField;

/// TableGen facts about one encoding, beyond its bits.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EncodingSchema {
    /// The record's MC operands in LLVM order: the `outs` DAG, then the `ins` DAG. No AIE2
    /// operand has sub-operands (`MIOperandInfo` is empty for all of them), so the position in
    /// this list is the `MCInst` operand index.
    pub operands: Vec<SchemaOperand>,
    /// The format class: the class directly below the slot base class
    /// (`AIE2_inst_<slot>_instr<N>`), e.g. `AIE2_vmac_inst_vec`. Its template arguments are
    /// [`format_args`](Self::format_args).
    pub format_class: String,
    /// Every TableGen class the record instantiates, most general first (a class comes after
    /// every class it derives from; ties by name), without the classes all encodings share
    /// (`Instruction`, `AIE2Inst`, ...). Includes the slot base class, the format class, the
    /// classes derived from it (`AIE2_mv_scl2ms_nb_doTlast_reg`) and the mix-ins
    /// (`AIE2_ag_all__ag_idx_imm`, `vmac_cm_core_dense`, `mWsrs`): the class vocabulary a
    /// structural table can key on instead of the instruction name.
    pub classes: Vec<String>,
    /// The bit-typed template arguments of the format class that reach the encoding, with the
    /// value this record gives them (read back from its fixed encoding bits), in the order their
    /// bits appear in the encoding, most significant first: `[("mdm", 1), ("md1", 0)]` for a MAC,
    /// `[("op", 2)]` for `lda.s16`.
    pub format_args: Vec<(String, u64)>,
    /// The address-generator form, from the `*__ag_*` class the record instantiates; `None`
    /// for records without an address generator.
    pub addressing: Option<AgAddressing>,
    /// Memory access width in bits, from the `MemoryVT` of a Pat<> that selects this record or a
    /// record of the same format class with the same format arguments (they differ only in the
    /// address generator). Only the sub-word scalar accesses have such patterns (`sextloadi16`,
    /// `truncstorei8`, ...); `None` for everything else, including every vector access.
    pub mem_vt_bits: Option<u16>,
    /// Implicit register uses and definitions (`Uses` / `Defs`), with the hardware encoding of
    /// each register.
    pub implicit_regs: Vec<SchemaImplicitReg>,
    /// Blocking and TLAST behaviour of a stream access, from its derived format class
    /// (`*_b_*` / `*_nb_*`, `*_noTlast` / `*_doTlast` / `*_doTlast_reg`); `None` for
    /// everything else.
    pub stream: Option<StreamForm>,
}

impl EncodingSchema {
    /// The MC operand with this DAG name (no `$`).
    pub fn operand(&self, name: &str) -> Option<&SchemaOperand> {
        self.operands.iter().find(|o| o.name == name)
    }

    /// The value this record gives the format class template argument `name`.
    pub fn format_arg(&self, name: &str) -> Option<u64> {
        self.format_args.iter().find(|(n, _)| n == name).map(|&(_, v)| v)
    }

    /// The MC operand an encoding field fills, if the field is bound to one.
    pub fn operand_for_field(&self, field: &OperandField) -> Option<&SchemaOperand> {
        self.operands.iter().find(|o| o.encoded && o.name == field.name)
    }
}

/// One MC operand of an instruction record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaOperand {
    /// Index in the `MCInst` operand list.
    pub mc_index: u8,
    /// DAG operand name without the `$` (`ptr`, `mod`, `dj`, `imm`, `acc1`, `tlast`, ...).
    pub name: String,
    /// The TableGen record named in the DAG: a register class (`eR`, `mXm`), a register operand
    /// (`OP_mLdaScl`) or an immediate operand (`imm6x4`).
    pub class: String,
    /// True for an `outs` operand.
    pub is_def: bool,
    /// The other operand of a `Constraints` tie (`$ptr_out = $ptr`), set on both operands.
    pub tied_to: Option<u8>,
    /// True when encoding bits carry the operand (an [`OperandField`] has its name). False for
    /// operands fixed by their class (`eR28:$tlast`, `eR29:$idx`) and for the defining half of
    /// a tie.
    pub encoded: bool,
    /// What the class says about the operand's values.
    pub kind: OperandKind,
}

/// What an operand's TableGen class says about its values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandKind {
    /// A register class operand.
    Register(RegisterOperandClass),
    /// An immediate operand with the LLVM decoder method that turns its encoded bits into a value.
    Immediate(ImmDecode),
}

/// The register class behind a register operand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegisterOperandClass {
    /// The register class (for a `RegisterOperand` such as `OP_mLdaScl`, the class it wraps:
    /// `mLdaScl`); a key of [`RegisterModel::classes`](super::RegisterModel).
    pub class: String,
    /// The family of the class, from the `AIE2*RegisterClass` it derives from.
    pub family: RegClassFamily,
    /// Width in bits of one register of the class (`RegInfo.RegSize`).
    pub bits: u16,
    /// The `get<Name>OpValue` MC encoder method of a `RegisterOperand` with a custom encoding:
    /// the class mixes register files and the field is a composite code, decoded by the LLVM
    /// disassembler. `None` for a plain register class (field value = position in the class).
    pub encoder: Option<String>,
    /// The register a single-member class fixes (`eR27` -> `r27`).
    pub fixed: Option<String>,
}

/// The register class families of AIE2, from the `AIE2*RegisterClass` TableGen class a register
/// class derives from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegClassFamily {
    /// `AIE2ScalarRegisterClass`: 32-bit scalar classes (GPRs, shift registers, and the mixed
    /// scalar classes the moves and loads use).
    Scalar,
    /// `AIE220BitRegisterClass`: pointer, modifier and the 20-bit special registers.
    Bits20,
    /// `AIE2Vector128RegisterClass`: q registers.
    Vector128,
    /// `AIE2Vector256RegisterClass`: w registers.
    Vector256,
    /// `AIE2Vector512RegisterClass`: x registers.
    Vector512,
    /// `AIE2Vector640RegisterClass`: qx registers (x plus a sparsity mask).
    Vector640,
    /// `AIE2Vector1024RegisterClass`: y registers.
    Vector1024,
    /// `AIE2Acc256RegisterClass`: am registers.
    Acc256,
    /// `AIE2Acc512RegisterClass`: bm registers.
    Acc512,
    /// `AIE2Acc1024RegisterClass`: cm registers.
    Acc1024,
    /// `AIE2Dim2DRegisterClass`: 2D descriptors (`d0`..`d7`).
    Dim2D,
    /// `AIE2Dim3DRegisterClass`: 3D descriptors (`d0_3d`..`d3_3d`).
    Dim3D,
    /// A class with none of the families above: the scalar pairs `eL`, the control registers
    /// `mCRm` and the status registers `mSRm`.
    Other,
}

/// How an immediate operand's encoded bits become its value: the LLVM decoder method named in
/// the operand record's `DecoderMethod`.
///
/// - `decodeUImmOperand<N>`: the `N` bits, unsigned.
/// - `decodeSImmOperand<N>`: the `N` bits, sign-extended.
/// - `decodeSImmOperandXStep<N, step, isNegative, isUnsigned>`: the `N` bits shifted left by
///   log2(`step`); a negative operand has an implied sign bit above them; otherwise signed
///   unless `isUnsigned`.
/// - no decoder method (`addr20`): the raw field value, `bits` = the field width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImmDecode {
    /// Number of encoded bits.
    pub bits: u8,
    /// Multiplier of the encoded value (a power of two; 1 when the operand has no step).
    pub step: u16,
    /// Signed value (sign-extended); false for an unsigned operand.
    pub signed: bool,
    /// The value is negative with an implied sign bit above the encoded bits (`*_neg` operands).
    pub negative: bool,
}

impl ImmDecode {
    /// The operand value of an encoded field, as the LLVM disassembler computes it.
    pub fn decode(&self, raw: u64) -> i64 {
        let raw = raw & low_mask(self.bits as u32);
        let shift = self.step.trailing_zeros();
        if self.negative {
            let width = self.bits as u32 + 1 + shift;
            sign_extend((raw << shift) | (1 << (self.bits as u32 + shift)), width)
        } else if self.signed {
            sign_extend(raw << shift, self.bits as u32 + shift)
        } else {
            (raw << shift) as i64
        }
    }
}

fn low_mask(bits: u32) -> u64 {
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn sign_extend(value: u64, width: u32) -> i64 {
    if width >= 64 {
        value as i64
    } else {
        let unused = 64 - width;
        ((value << unused) as i64) >> unused
    }
}

/// The address-generator forms of AIE2 loads and stores: the suffix of the `*__ag_*` class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgAddressing {
    /// `ag_idx`: `[p, dj]`, pointer unchanged.
    Indexed,
    /// `ag_idx_imm`: `[p, imm]`, pointer unchanged.
    IndexedImm,
    /// `ag_pstm_nrm`: `[p], m`, the pointer then advances by the modifier.
    PostModify,
    /// `ag_pstm_nrm_imm`: `[p], imm`, the pointer then advances by the immediate.
    PostModifyImm,
    /// `ag_pstm_2d`: `[p], d` with a 2D descriptor.
    TwoDim,
    /// `ag_pstm_3d`: `[p], d` with a 3D descriptor.
    ThreeDim,
    /// `ag_spill`: `[sp, imm]`, stack-relative.
    Spill,
    /// `ag_pstm_sp_imm`: the stack pointer advances by the immediate.
    SpPostModifyImm,
}

/// An implicit register an instruction reads (`Uses`) or writes (`Defs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaImplicitReg {
    /// Register name (`crSat`, `srSRS_of`, `SP`).
    pub name: String,
    /// Its `HWEncoding`.
    pub hw_encoding: u16,
    /// True for a use, false for a definition.
    pub is_use: bool,
}

/// Blocking and TLAST behaviour of a stream access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamForm {
    /// `*_nb*`: the access does not stall on a full / empty stream.
    pub non_blocking: bool,
    /// What the write does to TLAST; `None` for a stream read.
    pub tlast: Option<StreamTlast>,
}

/// TLAST of a stream write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamTlast {
    /// `*_noTlast`: TLAST clear.
    Clear,
    /// `*_doTlast`: TLAST set.
    Set,
    /// `*_doTlast_reg`: TLAST from the `$tlast` operand (r28).
    FromRegister,
}

/// The build script's pure extraction rules, compiled here so their unit tests run.
#[cfg(test)]
#[path = "../../../build_helpers/schema_rules.rs"]
mod build_schema_rules;

#[cfg(test)]
mod tests {
    use super::*;

    fn imm(bits: u8, step: u16, signed: bool, negative: bool) -> ImmDecode {
        ImmDecode { bits, step, signed, negative }
    }

    /// The formulas of `AIEBaseDisassembler.h`, spot-checked.
    #[test]
    fn imm_decode_matches_the_llvm_decoder_methods() {
        // decodeUImmOperand<2>
        assert_eq!(imm(2, 1, false, false).decode(0b11), 3);
        // decodeSImmOperand<7>
        assert_eq!(imm(7, 1, true, false).decode(0x7f), -1);
        assert_eq!(imm(7, 1, true, false).decode(0x3f), 63);
        // decodeSImmOperandXStep<6, 4, 0, 0> (imm6x4): 8-bit signed, low 2 bits zero
        assert_eq!(imm(6, 4, true, false).decode(0b100000), -128);
        assert_eq!(imm(6, 4, true, false).decode(0b000011), 12);
        // decodeSImmOperandXStep<12, 32, 1, 0> (imm12x32_neg): -(2^17) + raw*32
        assert_eq!(imm(12, 32, true, true).decode(0), -(1 << 17));
        assert_eq!(imm(12, 32, true, true).decode(0xfff), -32);
        // an unsigned XStep operand
        assert_eq!(imm(3, 32, false, false).decode(0b111), 224);
        // bits above the field are ignored
        assert_eq!(imm(2, 1, false, false).decode(0b111), 3);
    }

    #[test]
    fn schema_lookups() {
        let schema = EncodingSchema {
            operands: vec![SchemaOperand {
                mc_index: 0,
                name: "dst".into(),
                class: "eR".into(),
                is_def: true,
                tied_to: None,
                encoded: true,
                kind: OperandKind::Register(RegisterOperandClass {
                    class: "eR".into(),
                    family: RegClassFamily::Scalar,
                    bits: 32,
                    encoder: None,
                    fixed: None,
                }),
            }],
            format_args: vec![("mdm".into(), 1), ("md1".into(), 0)],
            ..Default::default()
        };
        assert_eq!(schema.operand("dst").map(|o| o.mc_index), Some(0));
        assert!(schema.operand("src").is_none());
        assert_eq!(schema.format_arg("mdm"), Some(1));
        assert_eq!(schema.format_arg("md2"), None);
        assert_eq!(schema.operand_for_field(&OperandField::new("dst", 0, 5)).map(|o| o.mc_index), Some(0));
    }
}

/// The schema of the build-time generated encodings, checked against the TableGen records.
#[cfg(test)]
mod generated {
    use super::*;
    use crate::aie2::isa::{load_from_generated, InstrEncoding, TblgenOutput};
    use std::sync::OnceLock;

    fn output() -> &'static TblgenOutput {
        static OUT: OnceLock<TblgenOutput> = OnceLock::new();
        OUT.get_or_init(load_from_generated)
    }

    fn enc(name: &str) -> &'static InstrEncoding {
        output()
            .encodings_by_slot
            .values()
            .flatten()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("no encoding {name}"))
    }

    fn all() -> impl Iterator<Item = &'static InstrEncoding> {
        output().encodings_by_slot.values().flatten()
    }

    fn reg(name: &str, operand: &str) -> RegisterOperandClass {
        match &enc(name)
            .schema
            .operand(operand)
            .unwrap_or_else(|| panic!("{name}: no ${operand}"))
            .kind
        {
            OperandKind::Register(r) => r.clone(),
            other => panic!("{name}.{operand}: {other:?}"),
        }
    }

    fn imm(name: &str, operand: &str) -> ImmDecode {
        match &enc(name)
            .schema
            .operand(operand)
            .unwrap_or_else(|| panic!("{name}: no ${operand}"))
            .kind
        {
            OperandKind::Immediate(i) => *i,
            other => panic!("{name}.{operand}: {other:?}"),
        }
    }

    #[test]
    fn operands_follow_the_outs_then_ins_dags() {
        let add = &enc("ADD").schema;
        let names: Vec<(&str, u8, bool)> =
            add.operands.iter().map(|o| (o.name.as_str(), o.mc_index, o.is_def)).collect();
        assert_eq!(names, [("mRx", 0, true), ("mRx0", 1, false), ("mRy", 2, false)]);
        assert!(add.operands.iter().all(|o| o.encoded && o.tied_to.is_none() && o.class == "eR"));
        assert_eq!(
            reg("ADD", "mRx"),
            RegisterOperandClass {
                class: "eR".into(),
                family: RegClassFamily::Scalar,
                bits: 32,
                encoder: None,
                fixed: None,
            }
        );
    }

    #[test]
    fn constraint_ties_are_recorded_on_both_operands() {
        let lda = &enc("LDA_S16_ag_pstm_nrm").schema;
        let ptr_out = lda.operand("ptr_out").unwrap();
        let ptr = lda.operand("ptr").unwrap();
        assert_eq!((ptr_out.mc_index, ptr.mc_index), (1, 2));
        assert_eq!((ptr_out.tied_to, ptr.tied_to), (Some(2), Some(1)));
        assert!(ptr.encoded && !ptr_out.encoded);
        assert_eq!(lda.operand("mod").unwrap().tied_to, None);
    }

    #[test]
    fn composite_register_operands_name_their_encoder() {
        let r = reg("LDA_dms_lda_idx_imm", "mLdaScl");
        assert_eq!(r.class, "mLdaScl");
        assert_eq!(r.encoder.as_deref(), Some("getmLdaSclOpValue"));
        assert_eq!(enc("LDA_dms_lda_idx_imm").schema.operand("mLdaScl").unwrap().class, "OP_mLdaScl");
    }

    #[test]
    fn single_register_classes_fix_their_register() {
        let tlast = enc("MOV_NB_mv_scl2ms_doTlast_reg").schema.operand("tlast").unwrap();
        assert!(!tlast.encoded);
        assert_eq!(reg("MOV_NB_mv_scl2ms_doTlast_reg", "tlast").fixed.as_deref(), Some("r28"));
        assert_eq!(reg("ADD", "mRy").fixed, None);
    }

    /// The four classifier bugs of the 2026-11-04 audit (bug 2) are absent from the schema.
    #[test]
    fn schema_classes_fix_the_legacy_classifier_bugs() {
        // q registers are 128-bit vectors, not accumulators.
        assert_eq!(reg("LDA_2D_dmv_lda_q", "dst").family, RegClassFamily::Vector128);
        // cm registers are 1024-bit accumulators, not unknown.
        let cm = reg("VMOV_D", "csrc");
        assert_eq!((cm.family, cm.bits), (RegClassFamily::Acc1024, 1024));
        assert_eq!(reg("VLDA_2D_UPS_S32_D8", "dst").family, RegClassFamily::Acc1024);
        // imm2 is unsigned.
        assert_eq!(
            imm("MOV_NB_TLAST_mv_cph2ms", "op"),
            ImmDecode { bits: 2, step: 1, signed: false, negative: false }
        );
        // imm12x32_neg: 12 encoded bits, step 32, negative.
        assert_eq!(
            imm("VLDA_dmw_lda_am_ag_spill", "imm"),
            ImmDecode { bits: 12, step: 32, signed: true, negative: true }
        );
        // eL is the scalar pair class: its registers are pairs of GPRs from r16 on.
        let l = reg("MOV_CNTR", "dst");
        assert_eq!((l.class.as_str(), l.family, l.bits), ("eL", RegClassFamily::Other, 64));
        let model = &output().register_model;
        assert_eq!(model.registers["l1"].sub_regs, ["r18", "r19"]);
    }

    #[test]
    fn immediate_decoders_come_from_the_operand_records() {
        assert_eq!(
            imm("LDA_dms_lda_idx_imm", "imm"),
            ImmDecode { bits: 6, step: 4, signed: true, negative: false }
        );
        assert_eq!(
            imm("LDA_dmv_lda_q_ag_spill", "imm"),
            ImmDecode { bits: 12, step: 16, signed: true, negative: true }
        );
        assert_eq!(imm("EVENT", "val"), ImmDecode { bits: 2, step: 1, signed: false, negative: false });
    }

    #[test]
    fn format_class_and_mac_template_bits() {
        let add = &enc("ADD").schema;
        assert_eq!(add.format_class, "AIE2_alu_r_rr_inst_alu");
        assert_eq!(add.format_arg("op"), Some(0));
        for (name, mdm, md1) in [
            ("VMAC_vmac_cm_core_dense", 0, 0),
            ("VMSC_vmac_cm_core_dense", 1, 0),
            ("VNEGMAC_vmac_cm_core_dense", 1, 1),
            ("VNEGMSC_vmac_cm_core_dense", 0, 1),
        ] {
            let s = &enc(name).schema;
            assert_eq!(s.format_class, "AIE2_vmac_inst_vec", "{name}");
            assert_eq!((s.format_arg("mdm"), s.format_arg("md1")), (Some(mdm), Some(md1)), "{name}");
            assert!(s.classes.iter().any(|c| c == "vmac_cm_core_dense"), "{name}: {:?}", s.classes);
        }
        assert!(enc("VMAC_vmac_cm_core_sparse_wide")
            .schema
            .classes
            .iter()
            .any(|c| c == "vmac_cm_core_sparse_wide"));
        // Sub-word loads: the op argument selects s8 / u8 / s16 / u16.
        assert_eq!(enc("LDA_U16_ag_idx").schema.format_arg("op"), Some(0b11));
    }

    #[test]
    fn every_encoding_has_a_format_class_and_no_shared_base_classes() {
        for e in all() {
            assert!(!e.schema.format_class.is_empty(), "{}", e.name);
            assert!(e.schema.classes.contains(&e.schema.format_class), "{}", e.name);
            assert!(!e.schema.classes.iter().any(|c| c == "Instruction" || c == "AIE2Inst"), "{}", e.name);
        }
        // Classes come after the classes they derive from.
        let c = &enc("MOV_NB_mv_scl2ms_doTlast_reg").schema.classes;
        let pos = |n: &str| c.iter().position(|x| x == n).unwrap_or_else(|| panic!("{n} not in {c:?}"));
        assert!(pos("AIE2_inst_st_instr32") < pos("AIE2_mv_scl2ms_inst_st"));
        assert!(pos("AIE2_mv_scl2ms_inst_st") < pos("AIE2_mv_scl2ms_nb_doTlast_reg"));
    }

    #[test]
    fn addressing_comes_from_the_address_generator_class() {
        use crate::aie2::isa::AddressingMode;
        assert_eq!(enc("LDA_dms_lda_idx_imm").schema.addressing, Some(AgAddressing::IndexedImm));
        assert_eq!(enc("LDA_U8_ag_idx").schema.addressing, Some(AgAddressing::Indexed));
        assert_eq!(enc("LDA_S16_ag_pstm_nrm").schema.addressing, Some(AgAddressing::PostModify));
        assert_eq!(enc("LDA_S16_ag_pstm_nrm_imm").schema.addressing, Some(AgAddressing::PostModifyImm));
        assert_eq!(enc("LDA_2D_S16_dmhb_lda").schema.addressing, Some(AgAddressing::TwoDim));
        assert_eq!(enc("ST_3D_S8").schema.addressing, Some(AgAddressing::ThreeDim));
        assert_eq!(enc("LDA_dmv_lda_q_ag_spill").schema.addressing, Some(AgAddressing::Spill));
        assert_eq!(enc("ADD").schema.addressing, None);
        // The descriptor modes agree with the structural addressing_mode the decoder uses today.
        for e in all() {
            match e.schema.addressing {
                Some(AgAddressing::TwoDim) => {
                    assert_eq!(e.addressing_mode, AddressingMode::TwoDim, "{}", e.name)
                }
                Some(AgAddressing::ThreeDim) => {
                    assert_eq!(e.addressing_mode, AddressingMode::ThreeDim, "{}", e.name)
                }
                _ => {}
            }
        }
    }

    #[test]
    fn sub_word_widths_come_from_pattern_memory_types() {
        for (name, bits) in [
            ("LDA_S16_ag_idx_imm", 16),  // named by the pattern
            ("LDA_S16_ag_pstm_nrm", 16), // same format and op, other address generator
            ("LDA_2D_U8_dmhb_lda", 8),
            ("ST_S8_ag_idx_imm", 8),
            ("ST_3D_S16", 16),
        ] {
            assert_eq!(enc(name).schema.mem_vt_bits, Some(bits), "{name}");
        }
        assert_eq!(enc("VLDA_dmw_lda_w_ag_idx").schema.mem_vt_bits, None);
        assert_eq!(enc("LDA_dms_lda_idx_imm").schema.mem_vt_bits, None);
        let sub_word = all().filter(|e| e.schema.mem_vt_bits.is_some()).count();
        assert_eq!(sub_word, 36, "18 byte + 18 half-word accesses");
    }

    #[test]
    fn stream_forms_come_from_the_derived_format_class() {
        let form = |n: &str| enc(n).schema.stream;
        let w = |non_blocking, tlast| Some(StreamForm { non_blocking, tlast: Some(tlast) });
        assert_eq!(form("MOV_NB_mv_scl2ms_doTlast_reg"), w(true, StreamTlast::FromRegister));
        assert_eq!(form("MOV_mv_ph2ms_doTlast_reg"), w(false, StreamTlast::FromRegister));
        assert_eq!(form("MOV_TLAST_mv_ph2ms"), w(false, StreamTlast::Set));
        assert_eq!(form("MOV_NB_mv_cph2ms"), w(true, StreamTlast::Clear));
        assert_eq!(form("MOV_mv_ss2scl"), Some(StreamForm { non_blocking: false, tlast: None }));
        assert_eq!(form("MOV_NB_mv_ss2scl"), Some(StreamForm { non_blocking: true, tlast: None }));
        assert_eq!(form("ADD"), None);
        assert_eq!(all().filter(|e| e.schema.stream.is_some()).count(), 20);
        // TLAST from a register exactly when the record has a `$tlast` operand.
        for e in all() {
            let from_reg =
                matches!(e.schema.stream, Some(StreamForm { tlast: Some(StreamTlast::FromRegister), .. }));
            assert_eq!(from_reg, e.schema.operand("tlast").is_some(), "{}", e.name);
        }
    }

    /// Bug 3 of the audit: implicit registers were numbered from the digits in their names.
    #[test]
    fn implicit_registers_carry_their_hardware_encoding() {
        let model = &output().register_model;
        let conv = enc("VCONV_BF16_FP32");
        let f2f = conv
            .schema
            .implicit_regs
            .iter()
            .find(|r| r.name == "crF2FMask")
            .expect("crF2FMask use");
        assert!(f2f.is_use);
        assert_eq!(f2f.hw_encoding, model.registers["crF2FMask"].hw_encoding);
        assert_eq!(conv.schema.implicit_regs.len(), conv.implicit_regs.len());
        // The legacy field no longer reads "2" out of "crF2FMask".
        assert!(conv.implicit_regs.iter().all(|r| r.reg_num == 0), "{:?}", conv.implicit_regs);
    }

    #[test]
    fn register_model_carries_sub_registers_and_class_families() {
        let model = &output().register_model;
        assert_eq!(model.registers["x0"].sub_regs, ["wl0", "wh0"]);
        assert_eq!(model.registers["x0"].sub_reg_indices, ["sub_256_lo", "sub_256_hi"]);
        assert!(model.registers["r0"].sub_regs.is_empty());
        for (class, family) in [
            ("eR", RegClassFamily::Scalar),
            ("eP", RegClassFamily::Bits20),
            ("mQQa", RegClassFamily::Vector128),
            ("mWm", RegClassFamily::Vector256),
            ("mXm", RegClassFamily::Vector512),
            ("mQQXw", RegClassFamily::Vector640),
            ("eYs", RegClassFamily::Vector1024),
            ("mAMm", RegClassFamily::Acc256),
            ("mBMs", RegClassFamily::Acc512),
            ("eCM", RegClassFamily::Acc1024),
            ("eD", RegClassFamily::Dim2D),
            ("eDS", RegClassFamily::Dim3D),
            ("eL", RegClassFamily::Other),
            ("mCRm", RegClassFamily::Other),
        ] {
            assert_eq!(model.classes[class].family, family, "{class}");
        }
        assert_eq!(model.leaves("eRS8").first().copied(), Some("r16"));
        assert_eq!(model.base_hw_encoding("eRS8"), Some(16));
        assert_eq!(model.leaves("mWm").len(), 24);
        assert_eq!(model.hw_encodings("eS"), [0, 1, 2, 3]);
    }
}
