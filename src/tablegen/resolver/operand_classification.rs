//! Operand type classification and addressing mode detection.
//!
//! Classifies operand fields from TableGen into decode types (register kind,
//! immediate, composite, lock ID) and detects addressing modes and memory
//! widths from instruction naming conventions.

/// Addressing mode extracted from the TableGen instruction name.
///
/// The AIE2 instruction naming convention encodes the addressing mode:
/// - `_idx_imm` -> IndexedImmediate:   *(ptr + imm), no pointer update
/// - `_idx` (no _imm) -> IndexedRegister: *(ptr + mN), no pointer update
/// - `_pstm_*_imm` -> PostModifyImmediate: t = *ptr, ptr += imm
/// - `_pstm_*` (no _imm) -> PostModifyRegister: t = *ptr, ptr += mN
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AddressingMode {
    #[default]
    Unknown,
    /// *(ptr + imm), pointer unchanged
    IndexedImmediate,
    /// *(ptr + mN), pointer unchanged
    IndexedRegister,
    /// t = *ptr; ptr += imm
    PostModifyImmediate,
    /// t = *ptr; ptr += mN
    PostModifyRegister,
}

/// Memory access width extracted from the instruction mnemonic.
///
/// The mnemonic suffix determines the width:
/// - `lda.s8` / `lda.u8` -> Byte (8-bit)
/// - `lda.s16` / `lda.u16` -> HalfWord (16-bit)
/// - `lda` (no suffix) -> Word (32-bit)
/// - `vlda.128` / `vldb.128` / `vst.128` -> QuadWord (128-bit)
/// - `vlda` / `vldb` / `vst` -> Vector256 (256-bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InstrMemWidth {
    Byte,
    HalfWord,
    #[default]
    Word,
    /// 128-bit quadword access (vlda.128, vst.128, etc.)
    QuadWord,
    Vector256,
}

/// Detect addressing mode from the TableGen instruction name.
///
/// Uses the deterministic naming convention from llvm-aie's class hierarchy:
/// format class names propagate into instruction names as suffixes.
pub fn detect_addressing_mode(instr_name: &str) -> AddressingMode {
    let lower = instr_name.to_lowercase();
    if lower.contains("_pstm_") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            AddressingMode::PostModifyImmediate
        } else {
            AddressingMode::PostModifyRegister
        }
    } else if lower.contains("_idx") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            AddressingMode::IndexedImmediate
        } else {
            AddressingMode::IndexedRegister
        }
    } else {
        AddressingMode::Unknown
    }
}

/// Detect memory access width from the instruction mnemonic.
///
/// Uses the assembly mnemonic suffix convention: `.s8`, `.u16`, etc.
/// Vector operations (`vlda`, `vldb`, `vst`) are normally 256-bit,
/// but `.128` variants load/store only 128 bits (lower half of vector reg).
pub fn detect_mem_width(mnemonic: &str) -> InstrMemWidth {
    detect_mem_width_full("", mnemonic)
}

/// Detect memory width from instruction name and mnemonic.
///
/// Quad-word scalar loads/stores (dmv_lda_q / dmv_sts_q) and AM loads/stores
/// (dmw_lda_am / dmw_sts_am) use the same mnemonics as regular scalar/vector
/// operations ("lda"/"st"/"vlda"/"vst"), so the encoding name is needed to
/// distinguish them.
pub fn detect_mem_width_full(instr_name: &str, mnemonic: &str) -> InstrMemWidth {
    let lower_name = instr_name.to_lowercase();

    // Quad-word scalar: 128-bit load/store to q registers.
    if lower_name.contains("dmv_lda_q") || lower_name.contains("dmv_sts_q") {
        return InstrMemWidth::QuadWord;
    }

    let lower = mnemonic.to_lowercase();
    if lower.starts_with("vlda") || lower.starts_with("vldb")
        || lower.starts_with("vst")
    {
        // Check for 128-bit variant before defaulting to 256-bit.
        if lower.contains(".128") || lower.contains("_128") {
            InstrMemWidth::QuadWord
        } else {
            InstrMemWidth::Vector256
        }
    } else if lower.contains(".s8") || lower.contains(".u8") {
        InstrMemWidth::Byte
    } else if lower.contains(".s16") || lower.contains(".u16") {
        InstrMemWidth::HalfWord
    } else {
        InstrMemWidth::Word
    }
}

/// How a raw bit-field value should be decoded into an Operand.
///
/// Determined from the `OperandDef.reg_class` in the TableGen DAG pattern.
/// This replaces all hand-written heuristics in the decoder with a single
/// data-driven dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OperandType {
    /// Simple register: raw value + base_offset IS the register index.
    /// base_offset is non-zero for register subclasses (e.g., eRS8 = r16-r23
    /// uses base_offset=16, so a 3-bit field value of 3 maps to r19).
    Register(RegisterKind),
    /// Register with a base offset for subclass encoding.
    RegisterWithOffset(RegisterKind, u8),
    /// Composite register: multiple register classes share one field,
    /// discriminated by low bits. Requires an inverse encoder function.
    CompositeRegister(CompositeEncoder),
    /// Immediate value with sign and scale.
    Immediate { signed: bool, scale: i32 },
    /// Lock ID.
    LockId,
    /// Unknown (safe fallback: treated as unsigned immediate).
    Unknown,
}

/// Which simple register file a field encodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegisterKind {
    /// eR: scalar general-purpose registers (r0-r31)
    Scalar,
    /// eP: pointer registers (p0-p7)
    Pointer,
    /// eM: post-modify registers (m0-m7)
    ModifierM,
    /// eDN: dimension size registers (dn0-dn7)
    ModifierDN,
    /// eDJ: dimension stride/jump registers (dj0-dj7)
    ModifierDJ,
    /// eDC: dimension count registers (dc0-dc7)
    ModifierDC,
    /// eW*, mWm: 256-bit vector registers
    Vector256,
    /// mXm, eX*: 512-bit vector registers
    Vector512,
    /// eYs: 1024-bit vector registers (y2-y5)
    Vector1024,
    /// eAM, eBM, eCM, mAMm, mBMm: accumulator registers
    Accumulator,
    /// mCRm: control registers (crRnd, crSat, crSRSSign, crVaddSign, etc.)
    Control,
    /// mQX*: sparse composite registers qx0-qx3 (vector data + sparsity mask)
    SparseQx,
    /// eL: 64-bit scalar register pairs (l0..l7 = {r16:r17}..{r30:r31}).
    /// Encoding is pair index; decoded register = index * 2 + base_offset.
    /// The even register is emitted as ScalarReg; the execution handler
    /// reads reg+1 for the upper 32 bits.
    ScalarPair,
}

/// Which Peano composite encoder function was used.
///
/// Each variant corresponds to a `get<Name>OpValue` function in
/// `AIE2MCCodeEmitterRegOperandDef.h`. The decoder inverts these
/// to recover the original register from the encoded bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeEncoder {
    /// mLdaScl, mSclSt, mSclMS: load/store scalar composite
    LdaScl,
    /// mMvSclSrc, mMvSclDst, mMvSclDstCg: move scalar composite
    MvSclSrc,
    /// mLdaCg, mLdbCg: load destination composite group
    LdaCg,
    /// mAluCg: ALU destination composite group
    AluCg,
    /// mMvAMWQDst: accumulator/weight-queue destination
    MvAMWQDst,
    /// mMvAMWQSrc: accumulator/weight-queue source
    MvAMWQSrc,
    /// mMvBMXSrc: buffer/cross source
    MvBMXSrc,
    /// mMvBMXDst: buffer/cross destination
    MvBMXDst,
    /// eRS4: scalar register subset (r16-r31)
    ERS4,
    /// mShflDst: shuffle destination
    ShflDst,
    /// mWm_1: vector register with bit rearrangement
    Wm1,
    /// mQXHLb: cross high/low byte
    QXHLb,
}

/// Classify an operand's decode type from the OperandDef.reg_class.
///
/// Maps the `reg_class` string (from the TableGen DAG pattern) to the
/// appropriate `OperandType`. Falls back to field-name heuristics when
/// `reg_class` is empty (e.g., for immediates not captured in DAG patterns).
pub fn classify_operand_type(reg_class: &str, field_name: &str) -> OperandType {
    // 1. Composite register operands (OP_* prefix from TableGen operand classes).
    // Lookup table mapping reg_class names to their CompositeEncoder variant.
    const COMPOSITE_LOOKUP: &[(&str, CompositeEncoder)] = &[
        ("OP_mLdaScl",    CompositeEncoder::LdaScl),
        ("OP_mSclSt",     CompositeEncoder::LdaScl),
        ("OP_mSclMS",     CompositeEncoder::LdaScl),
        ("OP_mLdbScl",    CompositeEncoder::LdaScl),
        ("OP_mMvSclSrc",  CompositeEncoder::MvSclSrc),
        ("OP_mMvSclDst",  CompositeEncoder::MvSclSrc),
        ("OP_mMvSclDstCg",CompositeEncoder::MvSclSrc),
        ("OP_mLdaCg",     CompositeEncoder::LdaCg),
        ("OP_mLdbCg",     CompositeEncoder::LdaCg),
        ("OP_mAluCg",     CompositeEncoder::AluCg),
        ("OP_mMvAMWQDst", CompositeEncoder::MvAMWQDst),
        ("OP_mMvAMWQSrc", CompositeEncoder::MvAMWQSrc),
        ("OP_mMvBMXSrc",  CompositeEncoder::MvBMXSrc),
        ("OP_mMvBMXDst",  CompositeEncoder::MvBMXDst),
        ("OP_mMcdSrc",    CompositeEncoder::MvBMXDst),
        ("OP_mScdDst",    CompositeEncoder::MvBMXDst),
        ("OP_eRS4",       CompositeEncoder::ERS4),
        ("OP_mShflDst",   CompositeEncoder::ShflDst),
        ("OP_mWm_1",      CompositeEncoder::Wm1),
        ("OP_mQXHLb",     CompositeEncoder::QXHLb),
    ];
    if let Some((_, encoder)) = COMPOSITE_LOOKUP.iter().find(|(name, _)| *name == reg_class) {
        return OperandType::CompositeRegister(*encoder);
    }

    // 2. Simple register classes (exact match or prefix)
    match reg_class {
        "eR" => return OperandType::Register(RegisterKind::Scalar),
        "eP" => return OperandType::Register(RegisterKind::Pointer),
        "eM" => return OperandType::Register(RegisterKind::ModifierM),
        "eDN" => return OperandType::Register(RegisterKind::ModifierDN),
        "eDJ" => return OperandType::Register(RegisterKind::ModifierDJ),
        "eDC" => return OperandType::Register(RegisterKind::ModifierDC),
        "mCRm" => return OperandType::Register(RegisterKind::Control),
        // eL: 64-bit scalar pair (l0={r16,r17} .. l7={r30,r31}).
        // 3-bit field encodes pair index; stride=2, base=16.
        "eL" => return OperandType::RegisterWithOffset(RegisterKind::ScalarPair, 16),
        _ => {}
    }

    if reg_class.starts_with("eW") || reg_class == "mWm" {
        return OperandType::Register(RegisterKind::Vector256);
    }
    if reg_class.starts_with("mXm") || reg_class.starts_with("eX") {
        return OperandType::Register(RegisterKind::Vector512);
    }
    if reg_class.starts_with("eY") {
        return OperandType::Register(RegisterKind::Vector1024);
    }
    if reg_class.starts_with("eAM") || reg_class.starts_with("mAMm")
        || reg_class.starts_with("eBM") || reg_class.starts_with("mBMm")
        || reg_class.starts_with("eCM") || reg_class.starts_with("mBMS")
    {
        return OperandType::Register(RegisterKind::Accumulator);
    }
    // mQQX*, mQX*: sparse composite registers qx0-qx3 (vector data + mask).
    // Must be checked BEFORE mQQ (accumulator weight queue) since mQQXw
    // starts with mQQ but is NOT an accumulator.
    if reg_class.starts_with("mQQX") || reg_class.starts_with("mQX") {
        return OperandType::Register(RegisterKind::SparseQx);
    }
    // mQQm (weight queue) -- treat as accumulator for emulator purposes
    if reg_class.starts_with("mQQ") {
        return OperandType::Register(RegisterKind::Accumulator);
    }

    // 3. Lock ID fields: always unsigned, regardless of reg_class.
    // Lock IDs are 0-63 (6-bit unsigned).  The field may have an imm6
    // reg_class which parse_immediate_type() would classify as signed.
    if field_name == "id" || field_name == "mLockId" {
        return OperandType::LockId;
    }

    // 4. Immediate operands (parsed from reg_class name)
    if let Some(imm_type) = parse_immediate_type(reg_class) {
        return imm_type;
    }

    // 5. Field-name fallback (when reg_class is empty or unrecognized)
    if reg_class.is_empty() || reg_class == "?" {
        return classify_from_field_name(field_name);
    }

    // Unrecognized reg_class -- this should be addressed by adding the class
    // to the match above. Panicking at build time catches the miss early.
    panic!(
        "Unrecognized register class '{}' for field '{}'. \
         Add it to classify_operand_type() in operand_classification.rs.",
        reg_class, field_name
    )
}

/// Parse immediate type from reg_class name (e.g., "simm7", "imm12x4").
fn parse_immediate_type(reg_class: &str) -> Option<OperandType> {
    if reg_class.starts_with("simm") {
        // Signed immediate, check for scale suffix
        let rest = &reg_class[4..]; // after "simm"
        let scale = extract_scale_suffix(rest);
        return Some(OperandType::Immediate { signed: true, scale });
    }
    if reg_class.starts_with("imm") {
        // AIE2 convention: imm* classes (immx4, immx32, etc.) are SIGNED by default.
        // Only immx128_unsigned explicitly sets isUnsigned=true in TableGen.
        // Detect unsigned variants by checking for "_unsigned" suffix.
        let is_unsigned = reg_class.contains("unsigned");
        let rest = &reg_class[3..]; // after "imm"
        let scale = extract_scale_suffix(rest);
        return Some(OperandType::Immediate { signed: !is_unsigned, scale });
    }
    if reg_class.starts_with("addr") {
        return Some(OperandType::Immediate { signed: false, scale: 1 });
    }
    // Target-specific formats like "t5u" (5-bit unsigned)
    if reg_class.starts_with('t') && reg_class.len() >= 3 {
        let last = reg_class.as_bytes()[reg_class.len() - 1];
        if last == b'u' || last == b's' {
            let signed = last == b's';
            return Some(OperandType::Immediate { signed, scale: 1 });
        }
    }
    None
}

/// Extract scale factor from immediate suffix (e.g., "12x4" -> 4, "7" -> 1).
fn extract_scale_suffix(s: &str) -> i32 {
    if let Some(pos) = s.find('x') {
        s[pos + 1..].parse::<i32>().unwrap_or(1)
    } else {
        1
    }
}

/// Classify operand type purely from the field name (last resort fallback).
///
/// Some TableGen definitions (especially FixupInstrInfo patterns) use raw
/// `bits<N>` fields without associating a register class in the DAG. The
/// field name itself carries enough information to classify the operand.
fn classify_from_field_name(field_name: &str) -> OperandType {
    if field_name == "id" || field_name == "mLockId" {
        return OperandType::LockId;
    }
    // Sparse MAC operands: ys1 = 1024-bit y-register, qxs2 = sparse composite.
    // These appear as raw bits<2> fields in vmac_*_sparse_wide definitions
    // without a register class binding.
    if field_name == "ys1" {
        return OperandType::Register(RegisterKind::Vector1024);
    }
    if field_name == "qxs2" {
        return OperandType::Register(RegisterKind::SparseQx);
    }
    // Constant fields: c5s, c11s (signed), c5u, c6u (unsigned)
    if field_name.len() >= 3 && field_name.starts_with('c') {
        let bytes = field_name.as_bytes();
        if bytes[1].is_ascii_digit() {
            let last = bytes[field_name.len() - 1];
            if last == b's' {
                return OperandType::Immediate { signed: true, scale: 1 };
            }
            if last == b'u' {
                return OperandType::Immediate { signed: false, scale: 1 };
            }
        }
    }
    OperandType::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_addressing_mode() {
        // Post-modify immediate
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_pstm_nrm_imm"),
            AddressingMode::PostModifyImmediate
        );
        assert_eq!(
            detect_addressing_mode("ST_dms_st_pstm_nrm_imm"),
            AddressingMode::PostModifyImmediate
        );

        // Post-modify register
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_pstm_nrm"),
            AddressingMode::PostModifyRegister
        );

        // Indexed immediate
        assert_eq!(
            detect_addressing_mode("LDA_S8_ag_idx_imm"),
            AddressingMode::IndexedImmediate
        );
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_idx_imm"),
            AddressingMode::IndexedImmediate
        );

        // Indexed register
        assert_eq!(
            detect_addressing_mode("LDA_dms_lda_idx"),
            AddressingMode::IndexedRegister
        );

        // Unknown (non-memory instructions)
        assert_eq!(
            detect_addressing_mode("ADD_add_r_ri"),
            AddressingMode::Unknown
        );
    }

    #[test]
    fn test_detect_mem_width() {
        // Byte (8-bit)
        assert_eq!(detect_mem_width("lda.s8"), InstrMemWidth::Byte);
        assert_eq!(detect_mem_width("lda.u8"), InstrMemWidth::Byte);
        assert_eq!(detect_mem_width("st.s8"), InstrMemWidth::Byte);

        // HalfWord (16-bit)
        assert_eq!(detect_mem_width("lda.s16"), InstrMemWidth::HalfWord);
        assert_eq!(detect_mem_width("lda.u16"), InstrMemWidth::HalfWord);
        assert_eq!(detect_mem_width("st.u16"), InstrMemWidth::HalfWord);

        // Word (32-bit, default)
        assert_eq!(detect_mem_width("lda"), InstrMemWidth::Word);
        assert_eq!(detect_mem_width("st"), InstrMemWidth::Word);
        assert_eq!(detect_mem_width("add"), InstrMemWidth::Word);

        // Vector (256-bit)
        assert_eq!(detect_mem_width("vlda"), InstrMemWidth::Vector256);
        assert_eq!(detect_mem_width("vldb"), InstrMemWidth::Vector256);
        assert_eq!(detect_mem_width("vst"), InstrMemWidth::Vector256);
    }

    #[test]
    fn test_classify_simple_registers() {
        assert_eq!(
            classify_operand_type("eR", "mRx"),
            OperandType::Register(RegisterKind::Scalar)
        );
        assert_eq!(
            classify_operand_type("eP", "ptr"),
            OperandType::Register(RegisterKind::Pointer)
        );
        assert_eq!(
            classify_operand_type("eM", "mod"),
            OperandType::Register(RegisterKind::ModifierM)
        );
        assert_eq!(
            classify_operand_type("eDN", "dn"),
            OperandType::Register(RegisterKind::ModifierDN)
        );
        assert_eq!(
            classify_operand_type("eDJ", "dj"),
            OperandType::Register(RegisterKind::ModifierDJ)
        );
        assert_eq!(
            classify_operand_type("eDC", "dc"),
            OperandType::Register(RegisterKind::ModifierDC)
        );
    }

    #[test]
    fn test_classify_vector_accumulator_registers() {
        assert_eq!(
            classify_operand_type("eWLE", "mW"),
            OperandType::Register(RegisterKind::Vector256)
        );
        assert_eq!(
            classify_operand_type("mWm", "mWm"),
            OperandType::Register(RegisterKind::Vector256)
        );
        assert_eq!(
            classify_operand_type("mXm", "mXm"),
            OperandType::Register(RegisterKind::Vector512)
        );
        assert_eq!(
            classify_operand_type("eAM", "mAM"),
            OperandType::Register(RegisterKind::Accumulator)
        );
        assert_eq!(
            classify_operand_type("mAMm", "mAMm"),
            OperandType::Register(RegisterKind::Accumulator)
        );
        assert_eq!(
            classify_operand_type("mBMm", "mBMm"),
            OperandType::Register(RegisterKind::Accumulator)
        );
    }

    #[test]
    fn test_classify_composite_registers() {
        assert_eq!(
            classify_operand_type("OP_mLdaScl", "mLdaScl"),
            OperandType::CompositeRegister(CompositeEncoder::LdaScl)
        );
        assert_eq!(
            classify_operand_type("OP_mSclSt", "mSclSt"),
            OperandType::CompositeRegister(CompositeEncoder::LdaScl)
        );
        assert_eq!(
            classify_operand_type("OP_mMvSclSrc", "mMvSclSrc"),
            OperandType::CompositeRegister(CompositeEncoder::MvSclSrc)
        );
        assert_eq!(
            classify_operand_type("OP_mMvSclDstCg", "mMvSclDstCg"),
            OperandType::CompositeRegister(CompositeEncoder::MvSclSrc)
        );
        assert_eq!(
            classify_operand_type("OP_mLdaCg", "mLdaCg"),
            OperandType::CompositeRegister(CompositeEncoder::LdaCg)
        );
        assert_eq!(
            classify_operand_type("OP_mAluCg", "mAluCg"),
            OperandType::CompositeRegister(CompositeEncoder::AluCg)
        );
    }

    #[test]
    fn test_classify_immediates() {
        assert_eq!(
            classify_operand_type("simm7", "imm"),
            OperandType::Immediate { signed: true, scale: 1 }
        );
        // AIE2 convention: imm* classes are signed (immx4, immx32, etc.)
        assert_eq!(
            classify_operand_type("imm12x4", "imm"),
            OperandType::Immediate { signed: true, scale: 4 }
        );
        assert_eq!(
            classify_operand_type("imm6x32", "imm"),
            OperandType::Immediate { signed: true, scale: 32 }
        );
        assert_eq!(
            classify_operand_type("imm5", "imm"),
            OperandType::Immediate { signed: true, scale: 1 }
        );
        assert_eq!(
            classify_operand_type("addr20", "cpmaddr"),
            OperandType::Immediate { signed: false, scale: 1 }
        );
    }

    #[test]
    fn test_classify_field_name_fallback() {
        // Lock IDs -- must be LockId even when reg_class is a signed imm type.
        // Lock IDs are unsigned (0-63); the field name takes precedence.
        assert_eq!(classify_operand_type("", "id"), OperandType::LockId);
        assert_eq!(classify_operand_type("", "mLockId"), OperandType::LockId);
        assert_eq!(classify_operand_type("imm6", "id"), OperandType::LockId);
        assert_eq!(classify_operand_type("simm6", "mLockId"), OperandType::LockId);

        // Constant fields from field name
        assert_eq!(
            classify_operand_type("", "c11s"),
            OperandType::Immediate { signed: true, scale: 1 }
        );
        assert_eq!(
            classify_operand_type("", "c5u"),
            OperandType::Immediate { signed: false, scale: 1 }
        );

        // Unknown fallback
        assert_eq!(classify_operand_type("", "unknown"), OperandType::Unknown);
    }
}
