//! Build-time semantic inference for instruction encodings.
//!
//! Ported from `src/tablegen/resolver.rs`. All functions return `String`
//! holding the Rust expression text for the corresponding enum variant
//! (e.g., `"OperandType::Register(RegisterKind::Scalar)"`). The compiler
//! validates these strings when compiling the generated code in
//! `gen_tablegen.rs`.

use super::records::{BuildImplicitReg, BuildInstrEncoding, BuildInstrRecord};

// ---------------------------------------------------------------------------
// Operand type classification
// ---------------------------------------------------------------------------

/// Classify an operand's decode type from the reg_class string.
///
/// Returns a Rust expression string for the `OperandType` constructor.
pub fn classify_operand_type(reg_class: &str, field_name: &str) -> String {
    // 1. Composite register operands
    const COMPOSITE_LOOKUP: &[(&str, &str)] = &[
        ("OP_mLdaScl", "CompositeEncoder::LdaScl"),
        ("OP_mSclSt", "CompositeEncoder::LdaScl"),
        ("OP_mSclMS", "CompositeEncoder::LdaScl"),
        ("OP_mLdbScl", "CompositeEncoder::LdaScl"),
        ("OP_mMvSclSrc", "CompositeEncoder::MvSclSrc"),
        ("OP_mMvSclDst", "CompositeEncoder::MvSclSrc"),
        ("OP_mMvSclDstCg", "CompositeEncoder::MvSclSrc"),
        ("OP_mLdaCg", "CompositeEncoder::LdaCg"),
        ("OP_mLdbCg", "CompositeEncoder::LdaCg"),
        ("OP_mAluCg", "CompositeEncoder::AluCg"),
        ("OP_mMvAMWQDst", "CompositeEncoder::MvAMWQDst"),
        ("OP_mMvAMWQSrc", "CompositeEncoder::MvAMWQSrc"),
        ("OP_mMvBMXSrc", "CompositeEncoder::MvBMXSrc"),
        ("OP_mMvBMXDst", "CompositeEncoder::MvBMXDst"),
        ("OP_mMcdSrc", "CompositeEncoder::MvBMXDst"),
        ("OP_mScdDst", "CompositeEncoder::MvBMXDst"),
        ("OP_eRS4", "CompositeEncoder::ERS4"),
        ("OP_mRS4m", "CompositeEncoder::ERS4"),
        ("OP_mShflDst", "CompositeEncoder::ShflDst"),
        ("OP_mWm_1", "CompositeEncoder::Wm1"),
        ("OP_mQXHLb", "CompositeEncoder::QXHLb"),
    ];

    if let Some((_, encoder)) = COMPOSITE_LOOKUP.iter().find(|(name, _)| *name == reg_class) {
        return format!("OperandType::CompositeRegister({})", encoder);
    }

    // 2. Simple register classes
    match reg_class {
        "eR" => return "OperandType::Register(RegisterKind::Scalar)".to_string(),
        "eP" => return "OperandType::Register(RegisterKind::Pointer)".to_string(),
        "eM" => return "OperandType::Register(RegisterKind::ModifierM)".to_string(),
        "eDN" => return "OperandType::Register(RegisterKind::ModifierDN)".to_string(),
        "eDJ" => return "OperandType::Register(RegisterKind::ModifierDJ)".to_string(),
        "eDC" => return "OperandType::Register(RegisterKind::ModifierDC)".to_string(),
        "mCRm" => return "OperandType::Register(RegisterKind::Control)".to_string(),
        _ => {}
    }
    // Scalar register subclasses with base offsets:
    // eRS4 = r16-r19 (4 regs, 2-bit index, base=16)
    // eRS8 = r16-r23 (8 regs, 3-bit index, base=16)
    // eR29 = r29 only (fixed register, base=29)
    // eL = GPR pairs l0-l7 (64-bit, l0=r16:r17, l1=r18:r19, ...).
    //   Used only by VSEL_8 as 64-bit select mask. Treat as scalar with
    //   base=16 since l_i maps to r(16+2*i):r(16+2*i+1) and the select
    //   mask is read as a scalar value from the pair's even register.
    // eS = shift registers s0-s3 (scalar, 2-bit index, base=0).
    //   Used by VSRS, VUPS, VSHIFT_ALIGN as shift amount operand.
    match reg_class {
        "eRS4" => return "OperandType::RegisterWithOffset(RegisterKind::Scalar, 16)".to_string(),
        "eRS8" => return "OperandType::RegisterWithOffset(RegisterKind::Scalar, 16)".to_string(),
        "eR29" => return "OperandType::RegisterWithOffset(RegisterKind::Scalar, 29)".to_string(),
        "eL" => return "OperandType::RegisterWithOffset(RegisterKind::Scalar, 16)".to_string(),
        "eS" => return "OperandType::Register(RegisterKind::Scalar)".to_string(),
        _ => {}
    }
    // Generic eR* subclasses: treat as base scalar (no offset)
    if reg_class.starts_with("eR") {
        return "OperandType::Register(RegisterKind::Scalar)".to_string();
    }
    // Pointer register subclasses
    if reg_class.starts_with("eP") {
        return "OperandType::Register(RegisterKind::Pointer)".to_string();
    }
    // 256-bit vector register subclasses (mWm, mWn, eW*, etc.)
    if reg_class.starts_with("eW") || reg_class.starts_with("mW") {
        return "OperandType::Register(RegisterKind::Vector256)".to_string();
    }
    // 512-bit vector register subclasses (mXm, mXn, mXv, mXw, mXa, mXs, eX*, etc.)
    // All subsets of VEC512 = (add mXm, mXn, mXv, mXw, mXa, mXs)
    if reg_class.starts_with("mX") || reg_class.starts_with("eX") {
        return "OperandType::Register(RegisterKind::Vector512)".to_string();
    }
    // 1024-bit vector register subclasses (eYs = y-registers y2-y5).
    // Used by sparse MAC wide variants for the dense A operand (ys1).
    if reg_class.starts_with("eY") {
        return "OperandType::Register(RegisterKind::Vector1024)".to_string();
    }
    // mQQX*: sparse composite registers qx0-qx3 (vector data + sparsity mask).
    // Must be checked BEFORE mQQ (accumulator weight queue) since mQQXw
    // starts with mQQ but is NOT an accumulator.
    if reg_class.starts_with("mQQX") || reg_class.starts_with("mQX") {
        return "OperandType::Register(RegisterKind::SparseQx)".to_string();
    }
    // Accumulator register subclasses (eAM*, mAMm*, eBM*, mBMm*, eCM*, mBMS*, mQQ*)
    if reg_class.starts_with("eAM")
        || reg_class.starts_with("mAM")
        || reg_class.starts_with("eBM")
        || reg_class.starts_with("mBM")
        || reg_class.starts_with("eCM")
        || reg_class.starts_with("mQQ")
    {
        return "OperandType::Register(RegisterKind::Accumulator)".to_string();
    }
    // Shift register classes: mSs contains shift registers s0-s3 (scalar).
    // Used by VSRS, VUPS, VSHIFT_ALIGN for shift amount operands.
    // NOT stream registers despite the "mS" prefix -- the "m" prefix is
    // llvm-aie's naming convention for "machine" register classes.
    if reg_class.starts_with("mS") {
        return "OperandType::Register(RegisterKind::Scalar)".to_string();
    }
    // Modifier register subclasses
    if reg_class.starts_with("eD") || reg_class.starts_with("eM") {
        return "OperandType::Register(RegisterKind::ModifierM)".to_string();
    }

    // 3. Lock ID fields: always unsigned, regardless of reg_class.
    // Lock IDs are 0-63 (6-bit unsigned).  The field may have an imm6
    // reg_class which parse_immediate_type() would classify as signed.
    if field_name == "id" || field_name == "mLockId" {
        return "OperandType::LockId".to_string();
    }

    // 4. Immediate operands
    if let Some(imm_type) = parse_immediate_type(reg_class) {
        return imm_type;
    }

    // 5. Field-name fallback
    if reg_class.is_empty() || reg_class == "?" {
        return classify_from_field_name(field_name);
    }

    "OperandType::Unknown".to_string()
}

/// Parse immediate type from reg_class name (e.g., "simm7", "imm12x4").
fn parse_immediate_type(reg_class: &str) -> Option<String> {
    if reg_class.starts_with("simm") {
        let rest = &reg_class[4..];
        let scale = extract_scale_suffix(rest);
        return Some(format!(
            "OperandType::Immediate {{ signed: true, scale: {} }}",
            scale
        ));
    }
    if reg_class.starts_with("imm") {
        let is_unsigned = reg_class.contains("unsigned");
        let rest = &reg_class[3..];
        let scale = extract_scale_suffix(rest);
        return Some(format!(
            "OperandType::Immediate {{ signed: {}, scale: {} }}",
            !is_unsigned,
            scale
        ));
    }
    if reg_class.starts_with("addr") {
        return Some("OperandType::Immediate { signed: false, scale: 1 }".to_string());
    }
    if reg_class.starts_with('t') && reg_class.len() >= 3 {
        let last = reg_class.as_bytes()[reg_class.len() - 1];
        if last == b'u' || last == b's' {
            let signed = last == b's';
            return Some(format!(
                "OperandType::Immediate {{ signed: {}, scale: 1 }}",
                signed
            ));
        }
    }
    None
}

/// Extract scale factor from immediate suffix.
fn extract_scale_suffix(s: &str) -> i32 {
    if let Some(pos) = s.find('x') {
        s[pos + 1..].parse::<i32>().unwrap_or(1)
    } else {
        1
    }
}

/// Classify operand type from field name alone (last resort).
///
/// Some TableGen definitions (especially FixupInstrInfo patterns) use raw
/// `bits<N>` fields without associating a register class in the DAG. The
/// field name itself carries enough information to classify the operand.
fn classify_from_field_name(field_name: &str) -> String {
    if field_name == "id" || field_name == "mLockId" {
        return "OperandType::LockId".to_string();
    }
    // Sparse MAC operands: ys1 = 1024-bit y-register, qxs2 = sparse composite.
    // These appear as raw bits<2> fields in vmac_*_sparse_wide definitions
    // without a register class binding.
    if field_name == "ys1" {
        return "OperandType::Register(RegisterKind::Vector1024)".to_string();
    }
    if field_name == "qxs2" {
        return "OperandType::Register(RegisterKind::SparseQx)".to_string();
    }
    if field_name.len() >= 3 && field_name.starts_with('c') {
        let bytes = field_name.as_bytes();
        if bytes[1].is_ascii_digit() {
            let last = bytes[field_name.len() - 1];
            if last == b's' {
                return "OperandType::Immediate { signed: true, scale: 1 }".to_string();
            }
            if last == b'u' {
                return "OperandType::Immediate { signed: false, scale: 1 }".to_string();
            }
        }
    }
    "OperandType::Unknown".to_string()
}

// ---------------------------------------------------------------------------
// Addressing mode detection
// ---------------------------------------------------------------------------

/// Detect addressing mode from instruction name. Returns a Rust expression string.
pub fn detect_addressing_mode(instr_name: &str) -> String {
    let lower = instr_name.to_lowercase();
    if lower.contains("_pstm_") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            "AddressingMode::PostModifyImmediate"
        } else {
            "AddressingMode::PostModifyRegister"
        }
    } else if lower.contains("_idx") {
        if lower.ends_with("_imm") || lower.contains("_imm_") {
            "AddressingMode::IndexedImmediate"
        } else {
            "AddressingMode::IndexedRegister"
        }
    } else {
        "AddressingMode::Unknown"
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Memory width detection
// ---------------------------------------------------------------------------

/// Detect memory width from instruction name and mnemonic.
///
/// Quad-word scalar loads/stores (dmv_lda_q / dmv_sts_q) use the same "lda"/"st"
/// mnemonics as regular 32-bit scalar ops. The encoding name is needed to
/// distinguish 128-bit q-register operations.
pub fn detect_mem_width_full(instr_name: &str, mnemonic: &str) -> String {
    let lower_name = instr_name.to_lowercase();

    // Quad-word scalar: 128-bit load/store to q registers.
    if lower_name.contains("dmv_lda_q") || lower_name.contains("dmv_sts_q") {
        return "InstrMemWidth::QuadWord".to_string();
    }

    let lower = mnemonic.to_lowercase();
    if lower.starts_with("vlda") || lower.starts_with("vldb") || lower.starts_with("vst") {
        if lower.contains(".128") || lower.contains("_128") {
            "InstrMemWidth::QuadWord"
        } else {
            "InstrMemWidth::Vector256"
        }
    } else if lower.contains(".s8") || lower.contains(".u8") {
        "InstrMemWidth::Byte"
    } else if lower.contains(".s16") || lower.contains(".u16") {
        "InstrMemWidth::HalfWord"
    } else {
        "InstrMemWidth::Word"
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Structural semantic inference
// ---------------------------------------------------------------------------

/// Infer semantic from structural TableGen attributes.
///
/// Returns a Rust expression string for the SemanticOp variant, or None.
pub fn infer_semantic_from_structure(
    defs: &[String],
    uses: &[String],
    may_load: bool,
    may_store: bool,
    has_delay_slot: bool,
    parents: &[String],
) -> Option<String> {
    let defs_lr = defs.iter().any(|r| r == "lr");
    let uses_lr = uses.iter().any(|r| r == "lr");

    if has_delay_slot {
        if defs_lr {
            return Some("SemanticOp::Call".to_string());
        }
        if uses_lr {
            return Some("SemanticOp::Ret".to_string());
        }
        return Some("SemanticOp::Br".to_string());
    }

    if may_load && !may_store {
        return Some("SemanticOp::Load".to_string());
    }
    if may_store && !may_load {
        return Some("SemanticOp::Store".to_string());
    }

    for parent in parents {
        if parent.contains("_done_") {
            return Some("SemanticOp::Done".to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Element type inference
// ---------------------------------------------------------------------------

/// Infer element type from mnemonic suffix.
///
/// Delegates to shared logic in `element_type_logic.rs` and formats as a
/// Rust expression string for codegen.
pub fn infer_element_type(mnemonic: &str) -> Option<String> {
    super::element_type_logic::infer_type_tag_from_mnemonic(mnemonic).map(tag_to_string)
}

/// Infer both element types for dual-type instructions (SRS/UPS/CONV/FLOOR).
///
/// Delegates to shared logic in `element_type_logic.rs`.
/// Returns `(element_type, from_type)` as Rust expression strings.
pub fn infer_dual_element_types(name: &str) -> (Option<String>, Option<String>) {
    let (et, ft) = super::element_type_logic::infer_dual_type_tags(name);
    (et.map(tag_to_string), ft.map(tag_to_string))
}

/// Convert a TypeTag to a Rust expression string for codegen output.
fn tag_to_string(tag: super::element_type_logic::TypeTag) -> String {
    use super::element_type_logic::TypeTag;
    match tag {
        TypeTag::Int8 => "ElementType::Int8",
        TypeTag::UInt8 => "ElementType::UInt8",
        TypeTag::Int16 => "ElementType::Int16",
        TypeTag::UInt16 => "ElementType::UInt16",
        TypeTag::Int32 => "ElementType::Int32",
        TypeTag::UInt32 => "ElementType::UInt32",
        TypeTag::Int64 => "ElementType::Int64",
        TypeTag::UInt64 => "ElementType::UInt64",
        TypeTag::BFloat16 => "ElementType::BFloat16",
        TypeTag::Float32 => "ElementType::Float32",
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Branch semantic refinement
// ---------------------------------------------------------------------------

/// Refine a Br semantic to BrCond when the mnemonic indicates a condition.
pub fn refine_branch_semantic(mnemonic: &str, semantic: Option<String>) -> Option<String> {
    if semantic.as_deref() != Some("SemanticOp::Br") {
        return semantic;
    }
    let mn = mnemonic.to_lowercase();
    if mn.starts_with("jnz")
        || mn.starts_with("jz")
        || (mn.starts_with('b') && !mn.starts_with("bswap"))
    {
        Some("SemanticOp::BrCond".to_string())
    } else {
        semantic
    }
}

// ---------------------------------------------------------------------------
// Encoding-name semantic refinement (build-time)
// ---------------------------------------------------------------------------

/// Correct semantics for instructions where pattern/itinerary inference
/// assigns the wrong operation.
///
/// Unlike earlier layers, this function OVERRIDES existing semantics when
/// it has a definitive answer. This is necessary because:
/// - VMSC_* gets `Mac` from itinerary (II_VMAC) but is really `MatMulSub`
/// - VNEGMAC_* gets `NegMul` from patterns but is really `NegMatMul`
/// - ASHL gets `Sra` from patterns but is really `AshlBidir`
/// - EXTENDu8/u16 get `And` from patterns but are really `ZeroExtend`
/// - MOV_CNTR gets `Copy` from patterns but is really `ReadCycleCounter`
///
/// This runs at build time, so the name-matching cost is zero at runtime.
pub fn refine_encoding_semantic(encoding_name: &str) -> Option<String> {
    // MAC-family instructions: encoding names encode the operation variant.
    // These all target the accumulator slot. Earlier layers may assign a
    // wrong semantic (e.g., itinerary assigns Mac to all MAC-family).
    let n = encoding_name;
    if n.starts_with("VNEGMAC_") {
        return Some("SemanticOp::NegMatMul".to_string());
    }
    if n.starts_with("VNEGMSC_") {
        // VNEGMSC = NegMul semantic. HW testing shows VNEGMSC gives raw
        // products (no product negation), same as VMAC with acc negate.
        // sub0 = is_msc XOR is_neg = true XOR true = false (no product negate)
        // sub1 = is_neg = true (acc negate)
        // With zero_acc=1: products + (-0) = products.
        return Some("SemanticOp::NegMul".to_string());
    }
    if n.starts_with("VNEGMUL_") {
        return Some("SemanticOp::NegMul".to_string());
    }
    if n.starts_with("VADDMAC_") || n.starts_with("VADDMSC_") {
        return Some("SemanticOp::AddMac".to_string());
    }
    if n.starts_with("VSUBMAC_") || n.starts_with("VSUBMSC_") {
        return Some("SemanticOp::SubMac".to_string());
    }
    if n.starts_with("VMAC_") {
        return Some("SemanticOp::Mac".to_string());
    }
    if n.starts_with("VMSC_") {
        return Some("SemanticOp::MatMulSub".to_string());
    }
    // Matrix multiply VMUL: encoding names contain _vmac_ (accumulator slot).
    // Element-wise vmul (e.g., VMUL_S8) does NOT contain _vmac_.
    if n.starts_with("VMUL_") && n.contains("_vmac_") {
        return Some("SemanticOp::MatMul".to_string());
    }

    // Fused load+compute / compute+store operations.
    // Structural inference sees only mayLoad/mayStore and returns Load/Store.
    // The instruction NAME encodes the fused operation.
    if n.contains("_UPS_") {
        return Some("SemanticOp::Ups".to_string());
    }
    if n.contains("_SRS_") {
        return Some("SemanticOp::Srs".to_string());
    }
    if n.contains("_UNPACK_") {
        return Some("SemanticOp::Unpack".to_string());
    }
    // VST_PACK (fused store+pack) -- match _PACK_ with underscore to avoid
    // matching VUNPACK which has "PACK" as a substring.
    if n.contains("_PACK_") && !n.contains("UNPACK") {
        return Some("SemanticOp::Pack".to_string());
    }
    if n.contains("_CONV_") {
        return Some("SemanticOp::Convert".to_string());
    }
    // Standalone VUNPACK/VPACK (register-only, no mayLoad/mayStore).
    if n.starts_with("VUNPACK_") {
        return Some("SemanticOp::Unpack".to_string());
    }
    if n.starts_with("VPACK_") {
        return Some("SemanticOp::Pack".to_string());
    }

    // Instructions without Pat<> patterns that need explicit mapping.
    match n {
        "ASHL" => Some("SemanticOp::AshlBidir".to_string()),
        "LSHL" => Some("SemanticOp::LshlBidir".to_string()),
        "SBC" => Some("SemanticOp::Sbc".to_string()),
        "DIVS" => Some("SemanticOp::DivStep".to_string()),
        "SELEQZ" | "SELNEZ" => Some("SemanticOp::Select".to_string()),
        "EXTENDu8" | "EXTENDu16" => Some("SemanticOp::ZeroExtend".to_string()),
        "MOV_CNTR" => Some("SemanticOp::ReadCycleCounter".to_string()),
        // VBAND/VBOR share sched_class II_VBLOG -- itinerary can't distinguish.
        "VBAND" => Some("SemanticOp::And".to_string()),
        "VBOR" => Some("SemanticOp::Or".to_string()),
        // VADDSUB has no intrinsic pattern.
        "VADDSUB_8" | "VADDSUB_16" | "VADDSUB_32" => Some("SemanticOp::Add".to_string()),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Branch condition inference
// ---------------------------------------------------------------------------

/// Infer branch condition from mnemonic and semantic.
pub fn infer_branch_condition(mnemonic: &str, semantic: Option<&str>) -> Option<String> {
    if semantic != Some("SemanticOp::BrCond") {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(
        if mn == "jnzd" {
            "BranchCondition::NotZeroDecrement"
        } else if mn == "jnz" {
            "BranchCondition::NotZero"
        } else if mn == "jz" {
            "BranchCondition::Zero"
        } else if mn.starts_with("beq") || mn == "bz" {
            "BranchCondition::Equal"
        } else if mn.starts_with("bne") || mn == "bnz" {
            "BranchCondition::NotEqual"
        } else if mn.starts_with("blt") {
            "BranchCondition::Less"
        } else if mn.starts_with("bge") {
            "BranchCondition::GreaterEqual"
        } else {
            "BranchCondition::NotEqual"
        }
        .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Select variant inference
// ---------------------------------------------------------------------------

/// Infer select variant from mnemonic and semantic.
pub fn infer_select_variant(mnemonic: &str, semantic: Option<&str>) -> Option<String> {
    if semantic != Some("SemanticOp::Select") {
        return None;
    }
    let mn = mnemonic.to_lowercase();
    Some(
        if mn.contains("eqz") {
            "SelectVariant::EqualZero"
        } else if mn.contains("nez") {
            "SelectVariant::NotEqualZero"
        } else {
            "SelectVariant::Generic"
        }
        .to_string(),
    )
}

// ---------------------------------------------------------------------------
// BuildInstrRecord -> BuildInstrEncoding conversion
// ---------------------------------------------------------------------------

impl BuildInstrRecord {
    /// Convert to BuildInstrEncoding for code generation.
    ///
    /// Returns None for `isCodeGenOnly` instructions and `isComposite` records.
    pub fn to_build_encoding(&self) -> Option<BuildInstrEncoding> {
        if self.is_code_gen_only || self.is_composite {
            return None;
        }
        let enc = self.slot_encoding.as_ref()?;
        let (fixed_mask, fixed_bits) = enc.compute_fixed_bits();
        let mut operand_fields = enc.extract_operand_fields();

        // Populate operand_type using reg_class from inputs/outputs.
        let all_operands: Vec<(&str, &str)> = self
            .outputs
            .iter()
            .chain(self.inputs.iter())
            .map(|(cls, name)| (cls.as_str(), name.as_str()))
            .collect();

        for field in &mut operand_fields {
            if let Some((reg_class, _)) =
                all_operands.iter().find(|(_, name)| *name == field.name)
            {
                field.operand_type = classify_operand_type(reg_class, &field.name);
                if field.operand_type.contains("signed: true") {
                    field.signed = true;
                }
            } else {
                field.operand_type = classify_operand_type("", &field.name);
            }
        }

        let slot = match enc.slot.as_str() {
            "alu" | "lda" | "ldb" | "st" | "mv" | "vec" | "lng" => enc.slot.as_str(),
            _ => return None,
        };

        // Structural semantic inference from TableGen attributes.
        let semantic = if self.is_move_imm || self.is_move_reg {
            Some("SemanticOp::Copy".to_string())
        } else if self.is_slot_nop {
            Some("SemanticOp::Nop".to_string())
        } else {
            infer_semantic_from_structure(
                &self.defs,
                &self.uses,
                self.may_load,
                self.may_store,
                self.has_delay_slot,
                &self.parents,
            )
        };

        let semantic = refine_branch_semantic(&self.mnemonic, semantic);
        // Apply name-based overrides that are needed for downstream inference
        // (select_variant, etc.) before those inferences run.  The full
        // refine_encoding_semantic pipeline in extract.rs may re-set this
        // later, but the override here ensures infer_select_variant sees the
        // correct semantic.
        let semantic = refine_encoding_semantic(&self.name).or(semantic);

        let input_order: Vec<String> = self.inputs.iter().map(|(_, n)| n.clone()).collect();
        let output_order: Vec<String> = self.outputs.iter().map(|(_, n)| n.clone()).collect();

        let implicit_regs: Vec<BuildImplicitReg> = self
            .defs
            .iter()
            .map(|r| {
                let reg_num = r
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);
                BuildImplicitReg {
                    reg_class: r.clone(),
                    reg_num,
                    is_use: false,
                }
            })
            .chain(self.uses.iter().map(|r| {
                let reg_num = r
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .unwrap_or(0);
                BuildImplicitReg {
                    reg_class: r.clone(),
                    reg_num,
                    is_use: true,
                }
            }))
            .collect();

        let is_vector = self.mnemonic.starts_with('v') || self.mnemonic.starts_with('V');
        let is_sp_relative = self.uses.iter().any(|u| u == "SP");
        let (dual_et, dual_ft) = infer_dual_element_types(&self.name);
        let element_type = dual_et.or_else(|| infer_element_type(&self.mnemonic));
        let from_type = dual_ft;
        let branch_condition =
            infer_branch_condition(&self.mnemonic, semantic.as_deref());
        let select_variant =
            infer_select_variant(&self.mnemonic, semantic.as_deref());
        let is_ptr_arithmetic = semantic.as_deref() == Some("SemanticOp::PointerAdd");

        Some(BuildInstrEncoding {
            name: self.name.clone(),
            mnemonic: self.mnemonic.clone(),
            asm_string: self.asm_string.clone(),
            slot: slot.to_string(),
            width: enc.width,
            fixed_mask,
            fixed_bits,
            operand_fields,
            semantic,
            may_load: self.may_load,
            may_store: self.may_store,
            input_order,
            output_order,
            implicit_regs,
            addressing_mode: detect_addressing_mode(&self.name),
            mem_width: detect_mem_width_full(&self.name, &self.mnemonic),
            has_complete_decoder: self.has_complete_decoder,
            element_type,
            from_type,
            branch_condition,
            is_vector,
            select_variant,
            is_ptr_arithmetic,
            is_sp_relative,
            sched_class: self.itinerary_class.clone(),
        })
    }
}
