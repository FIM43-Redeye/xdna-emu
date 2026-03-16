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
    // Accumulator register subclasses (eAM*, mAMm*, eBM*, mBMm*, eCM*, mBMS*, mQQ*, mQX*)
    if reg_class.starts_with("eAM")
        || reg_class.starts_with("mAM")
        || reg_class.starts_with("eBM")
        || reg_class.starts_with("mBM")
        || reg_class.starts_with("eCM")
        || reg_class.starts_with("mQQ")
        || reg_class.starts_with("mQX")
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

    // 3. Immediate operands
    if let Some(imm_type) = parse_immediate_type(reg_class) {
        return imm_type;
    }

    // 4. Field-name fallback
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
fn classify_from_field_name(field_name: &str) -> String {
    if field_name == "id" || field_name == "mLockId" {
        return "OperandType::LockId".to_string();
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

/// Detect memory access width from mnemonic. Returns a Rust expression string.
pub fn detect_mem_width(mnemonic: &str) -> String {
    let lower = mnemonic.to_lowercase();
    if lower.starts_with("vlda") || lower.starts_with("vldb") || lower.starts_with("vst") {
        "InstrMemWidth::Vector256"
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
pub fn infer_element_type(mnemonic: &str) -> Option<String> {
    if mnemonic.ends_with("8") || mnemonic.contains(".i8") || mnemonic.contains(".u8") {
        if mnemonic.contains(".u") {
            Some("ElementType::UInt8".to_string())
        } else {
            Some("ElementType::Int8".to_string())
        }
    } else if mnemonic.ends_with("16") || mnemonic.contains(".i16") || mnemonic.contains(".u16") {
        if mnemonic.contains(".u") {
            Some("ElementType::UInt16".to_string())
        } else {
            Some("ElementType::Int16".to_string())
        }
    } else if mnemonic.ends_with("32") || mnemonic.contains(".i32") || mnemonic.contains(".u32") {
        if mnemonic.contains(".u") {
            Some("ElementType::UInt32".to_string())
        } else {
            Some("ElementType::Int32".to_string())
        }
    } else if mnemonic.contains("bf16") || mnemonic.contains(".bf") {
        Some("ElementType::BFloat16".to_string())
    } else if mnemonic.contains("f32") || mnemonic.contains("float") || mnemonic.ends_with(".f") {
        Some("ElementType::Float32".to_string())
    } else {
        None
    }
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
        let element_type = infer_element_type(&self.mnemonic);
        let branch_condition =
            infer_branch_condition(&self.mnemonic, semantic.as_deref());
        let select_variant =
            infer_select_variant(&self.mnemonic, semantic.as_deref());
        let is_ptr_arithmetic = semantic.as_deref() == Some("SemanticOp::PointerAdd");

        Some(BuildInstrEncoding {
            name: self.name.clone(),
            mnemonic: self.mnemonic.clone(),
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
            mem_width: detect_mem_width(&self.mnemonic),
            has_complete_decoder: self.has_complete_decoder,
            element_type,
            branch_condition,
            is_vector,
            select_variant,
            is_ptr_arithmetic,
            is_sp_relative,
            sched_class: self.itinerary_class.clone(),
        })
    }
}
