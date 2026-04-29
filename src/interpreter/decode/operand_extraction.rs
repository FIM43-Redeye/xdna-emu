//! Operand extraction from decoded instructions.
//!
//! Contains both the production LLVM FFI decode path and the legacy
//! bit-field extraction path (retained for cross-validation tests).

use std::collections::HashMap;

use crate::interpreter::bundle::{Operand, PostModify};
use crate::interpreter::state::SP_PTR_INDEX;
#[cfg(test)]
use crate::interpreter::state::{MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M};
use xdna_archspec::aie2::isa::{
    AddressingMode, InstrEncoding,
    decoder_ffi::{self, DecodedOperand, DecodeResult},
};
use crate::interpreter::decode::register_map::{AccumWidth, operand_from_reg_name};
#[cfg(test)]
use xdna_archspec::aie2::isa::{CompositeEncoder, OperandType, RegisterKind};

use super::decoder::{DecodedInstr, InstructionDecoder, can_be_dest};

/// Sign-extend a value from `bits` width to i32.
#[cfg(test)]
#[inline]
fn sign_extend(value: i32, bits: u32) -> i32 {
    if bits == 0 || bits >= 32 {
        return value;
    }
    let shift = 32 - bits;
    (value << shift) >> shift
}

/// Sign-extend an already-extracted unsigned raw value based on its logical width.
#[cfg(test)]
#[inline]
fn sign_extend_raw(value: u64, width: u8) -> i64 {
    if width == 0 || width >= 64 {
        return value as i64;
    }
    let sign_bit = 1u64 << (width - 1);
    if value & sign_bit != 0 {
        let mask = !((1u64 << width) - 1);
        (value | mask) as i64
    } else {
        value as i64
    }
}

impl InstructionDecoder {
    // -----------------------------------------------------------------------
    // LLVM FFI decode path
    // -----------------------------------------------------------------------

    /// Convert SlotType to decoder_ffi::Slot for LLVM decoding.
    pub(super) fn slot_type_to_ffi(
        slot_type: crate::interpreter::bundle::SlotType,
    ) -> Option<decoder_ffi::Slot> {
        use crate::interpreter::bundle::SlotType;
        match slot_type {
            SlotType::Alu => Some(decoder_ffi::Slot::Alu),
            SlotType::Lda => Some(decoder_ffi::Slot::Lda),
            SlotType::Ldb => Some(decoder_ffi::Slot::Ldb),
            SlotType::Lng => Some(decoder_ffi::Slot::Lng),
            SlotType::Mv => Some(decoder_ffi::Slot::Mv),
            SlotType::St => Some(decoder_ffi::Slot::St),
            SlotType::Vec => Some(decoder_ffi::Slot::Vec),
            SlotType::Nop => None,
        }
    }

    /// Try to decode a slot using LLVM FFI and build operands directly from
    /// LLVM's register names and output classification.
    ///
    /// Returns (DecodedInstr, dest, sources, post_modify, opcode, extra_defs, accum_width) on success.
    /// The DecodedInstr is still needed for build_slot_op() which reads
    /// metadata (semantic, element_type, etc.) from InstrEncoding.
    ///
    /// Falls back to None if:
    /// - LLVM can't decode the slot
    /// - The instruction name doesn't match any InstrEncoding
    /// - An LLVM register name can't be mapped to our Operand type
    pub(super) fn try_decode_via_ffi(
        &self,
        bits: u64,
        slot_type: crate::interpreter::bundle::SlotType,
    ) -> Option<(
        DecodedInstr,
        Option<Operand>,
        Vec<Operand>,
        Option<PostModify>,
        Option<u32>,
        Vec<Operand>,
        Option<AccumWidth>,
    )> {
        let ffi_slot = Self::slot_type_to_ffi(slot_type)?;
        let ffi_result = decoder_ffi::decode_slot(ffi_slot, bits)?;

        // Look up InstrEncoding by LLVM instruction name (still need for metadata).
        let slot_name = match slot_type {
            crate::interpreter::bundle::SlotType::Lda => "lda",
            crate::interpreter::bundle::SlotType::Ldb => "ldb",
            crate::interpreter::bundle::SlotType::Alu => "alu",
            crate::interpreter::bundle::SlotType::Mv => "mv",
            crate::interpreter::bundle::SlotType::St => "st",
            crate::interpreter::bundle::SlotType::Vec => "vec",
            crate::interpreter::bundle::SlotType::Lng => "lng",
            _ => return None,
        };

        let encoding = self.index.encoding_by_name(slot_name, &ffi_result.name);
        if encoding.is_none() {
            log::warn!("[FFI-LOOKUP MISS] slot={} name={} -- not in index", slot_name, ffi_result.name);
        }
        let encoding = encoding?;

        // Map LLVM operands to our Operand type.
        let (dest, sources, post_modify, extra_defs, accum_width) =
            Self::extract_operands_from_ffi(&ffi_result, encoding);

        // Build a minimal DecodedInstr with empty raw operands (we don't need
        // them since we're using LLVM's operands, but build_slot_op reads
        // encoding metadata from it).
        let decoded_instr = DecodedInstr { encoding: encoding.clone(), operands: HashMap::new() };

        Some((decoded_instr, dest, sources, post_modify, Some(ffi_result.opcode), extra_defs, accum_width))
    }

    /// Extract dest, sources, and post-modify from LLVM FFI decode result.
    ///
    /// Uses num_defs to split outputs from inputs, then applies addressing
    /// mode logic from InstrEncoding to construct Memory operands and
    /// PostModify metadata.
    pub(super) fn extract_operands_from_ffi(
        ffi_result: &DecodeResult,
        encoding: &InstrEncoding,
    ) -> (Option<Operand>, Vec<Operand>, Option<PostModify>, Vec<Operand>, Option<AccumWidth>) {
        let num_defs = ffi_result.num_defs as usize;

        // Map all LLVM operands to our Operand type.
        // Skip operands that can't be mapped (mask registers, DMA regs).
        //
        // LLVM's decoder tables already handle sign extension and scaling
        // internally.  The decoded immediate values are ready to use as-is.
        // Do NOT apply additional sign extension or scaling here -- that was
        // needed for our raw bit-field extraction path, not for LLVM output.
        //
        // Also capture the first AccumWidth seen from register class metadata
        // (e.g., bml -> Half, cm -> Full). This flows to SlotOp.accum_width
        // for use in accumulator execution paths.
        let mut first_accum_width: Option<AccumWidth> = None;
        let mapped: Vec<Option<Operand>> = ffi_result
            .operands
            .iter()
            .map(|op| match op {
                DecodedOperand::Reg { name, .. } => operand_from_reg_name(name).map(|m| {
                    if first_accum_width.is_none() {
                        first_accum_width = m.accum_width;
                    }
                    m.operand
                }),
                DecodedOperand::Imm(val) => Some(Operand::Immediate(*val as i32)),
            })
            .collect();

        // Split into defs (outputs) and uses (inputs).
        let split_at = num_defs.min(mapped.len());
        let def_ops: Vec<Operand> = mapped[..split_at].iter().filter_map(|o| o.clone()).collect();
        let mut use_ops: Vec<Operand> = mapped[split_at..].iter().filter_map(|o| o.clone()).collect();

        // Destination: first def operand (if any and writable).
        // Remaining defs (e.g., cmp register in dual-result instructions)
        // are saved in extra_defs for later attachment to SlotOp.
        let mut def_iter = def_ops.into_iter();
        let mut dest = def_iter.next();
        let extra_defs: Vec<Operand> = def_iter.collect();
        if let Some(ref d) = dest {
            if !can_be_dest(d) {
                // Non-writable in dest position (config immediate, etc.) -- move to sources.
                use_ops.insert(0, d.clone());
                dest = None;
            }
        }

        let mut post_modify: Option<PostModify> = None;

        // LLVM MCInst operand layout is deterministic (from AIEBaseDisassembler.h):
        //   Loads:  defs=[loaded_value, post_mod_ptr]  uses=[base_ptr, modify]
        //   Stores: defs=[post_mod_ptr]                uses=[data, base_ptr, modify]
        //
        // def[0] is always the right dest: loaded value for loads, post-mod
        // pointer for stores (which executors ignore via dest=None for stores
        // without register defs).  No heuristic swapping needed.

        // Some LLVM instruction definitions (e.g., LDA_dmv_lda_q) have all
        // operands in (ins) with (outs) empty, so num_defs=0 and dest=None.
        // For load instructions, promote the first writable non-address register
        // to dest so the executor writes the loaded data.
        if dest.is_none() && encoding.may_load {
            if let Some(pos) = use_ops.iter().position(|o| {
                matches!(
                    o,
                    Operand::ControlReg(_)
                        | Operand::AccumReg(_)
                        | Operand::VectorReg(_)
                        | Operand::ScalarReg(_)
                        | Operand::SparseQxReg(_)
                )
            }) {
                dest = Some(use_ops.remove(pos));
            }
        }

        // Handle pointer arithmetic: dest is the pointer reg, sources are offset.
        // LLVM reports the pointer as both a def (output) and a use (input) since
        // the instruction reads and writes the same register.  Deduplicate: keep
        // the pointer only in dest, remove the self-reference from sources so
        // executors see [offset] not [self_ptr, offset].
        if encoding.is_ptr_arithmetic {
            if dest.is_none() {
                // Find the first PointerReg in uses and promote it to dest.
                if let Some(pos) = use_ops.iter().position(|o| matches!(o, Operand::PointerReg(_))) {
                    dest = Some(use_ops.remove(pos));
                }
            }
            // Remove any self-referencing pointer from sources.
            if let Some(Operand::PointerReg(dest_p)) = &dest {
                let dp = *dest_p;
                use_ops.retain(|o| !matches!(o, Operand::PointerReg(p) if *p == dp));
            }
            // Sources are whatever remains (offset immediate or modifier reg).
            return (dest, use_ops, post_modify, extra_defs, first_accum_width);
        }

        // Handle memory addressing modes: combine pointer + offset into Memory
        // or PostModify, matching what decode_ag_field() does in the legacy path.
        match encoding.addressing_mode {
            AddressingMode::IndexedImmediate => {
                // Find PointerReg + Immediate pair in uses, combine into Memory.
                if let Some(ptr_pos) = use_ops.iter().position(|o| matches!(o, Operand::PointerReg(_))) {
                    let ptr_reg = match use_ops.remove(ptr_pos) {
                        Operand::PointerReg(p) => p,
                        _ => unreachable!(),
                    };
                    // Next Immediate after the pointer (now at ptr_pos since we removed ptr).
                    let offset = if let Some(imm_pos) =
                        use_ops[ptr_pos..].iter().position(|o| matches!(o, Operand::Immediate(_)))
                    {
                        match use_ops.remove(ptr_pos + imm_pos) {
                            Operand::Immediate(v) => v as i16,
                            _ => unreachable!(),
                        }
                    } else {
                        0
                    };
                    // SP-relative: handled naturally since SP maps to PointerReg(SP_PTR_INDEX).
                    use_ops.push(Operand::Memory { base: ptr_reg, offset });
                }
            }
            AddressingMode::PostModifyImmediate => {
                if let Some(ptr_pos) = use_ops.iter().position(|o| matches!(o, Operand::PointerReg(_))) {
                    let ptr_reg = use_ops.remove(ptr_pos);
                    // Find the modify amount.
                    let modify = if let Some(imm_pos) =
                        use_ops[ptr_pos..].iter().position(|o| matches!(o, Operand::Immediate(_)))
                    {
                        match use_ops.remove(ptr_pos + imm_pos) {
                            Operand::Immediate(v) => v as i16,
                            _ => 0,
                        }
                    } else {
                        0
                    };
                    use_ops.push(ptr_reg);
                    post_modify = Some(PostModify::Immediate(modify));
                }
            }
            AddressingMode::PostModifyRegister => {
                if let Some(ptr_pos) = use_ops.iter().position(|o| matches!(o, Operand::PointerReg(_))) {
                    let ptr_reg = use_ops.remove(ptr_pos);
                    // Find the modifier register.
                    if let Some(mod_pos) =
                        use_ops[ptr_pos..].iter().position(|o| matches!(o, Operand::ModifierReg(_)))
                    {
                        let mod_reg = match use_ops.remove(ptr_pos + mod_pos) {
                            Operand::ModifierReg(r) => r,
                            _ => 0,
                        };
                        post_modify = Some(PostModify::Register(mod_reg));
                    }
                    use_ops.push(ptr_reg);
                }
            }
            AddressingMode::IndexedRegister => {
                // ptr + dj register: both stay as separate sources.
                // (Current decoder does this too -- no Memory combining.)
            }
            AddressingMode::Unknown => {
                // No addressing mode -- pure compute instruction, or
                // unrecognized addressing pattern. Leave operands as-is.
            }
        }

        // SP-relative load/store: uses = [SP] in TableGen.
        // Convert standalone immediate to Memory { base: SP, offset: imm }.
        if encoding.is_sp_relative && !encoding.is_ptr_arithmetic {
            if let Some(imm_pos) = use_ops.iter().position(|o| matches!(o, Operand::Immediate(_))) {
                let offset = match use_ops.remove(imm_pos) {
                    Operand::Immediate(v) => v as i16,
                    _ => 0,
                };
                use_ops.push(Operand::Memory { base: SP_PTR_INDEX, offset });
            }
        }

        (dest, use_ops, post_modify, extra_defs, first_accum_width)
    }

    // -----------------------------------------------------------------------
    // Legacy bit-field extraction (test-only)
    // -----------------------------------------------------------------------

    /// Extract operands from decoded instruction using data-driven dispatch.
    ///
    /// Legacy bit-field extraction method. Retained for cross-validation tests
    /// against the LLVM FFI operand extraction. Not used in production.
    #[cfg(test)]
    pub(super) fn extract_operands(
        &self,
        decoded: &DecodedInstr,
    ) -> (Option<Operand>, Vec<Operand>, Option<PostModify>, Vec<Operand>) {
        let mut field_operands: HashMap<String, Operand> = HashMap::new();
        let mut direct_dest: Option<Operand> = None;
        let mut extra_sources: Vec<Operand> = Vec::new();
        let mut extracted_post_modify: Option<PostModify> = None;

        for field in &decoded.encoding.operand_fields {
            let raw = decoded.operand(&field.name).unwrap_or(0);

            // Address generator fields encode packed {ptr, offset/mod, mode} --
            // this is structural addressing, not a register encoding issue.
            if field.name.starts_with("ag") {
                self.decode_ag_field(
                    field,
                    raw,
                    decoded,
                    &mut direct_dest,
                    &mut extra_sources,
                    &mut extracted_post_modify,
                );
                continue;
            }

            // Data-driven dispatch on operand_type
            let operand = match &field.operand_type {
                OperandType::Register(kind) => Self::decode_register(*kind, raw as u8, 0),
                OperandType::RegisterWithOffset(kind, base) => Self::decode_register(*kind, raw as u8, *base),
                OperandType::CompositeRegister(encoder) => {
                    // Composite registers are decoded by the LLVM FFI layer
                    // in production. This minimal inline decode covers the
                    // common patterns needed by cross-validation tests.
                    Self::decode_composite_inline(*encoder, raw)
                }
                OperandType::Immediate { signed, scale } => {
                    let value = if *signed {
                        // Sign-extend the already-extracted raw value based on field width.
                        // (Don't reconstruct positioned bits -- breaks for split fields.)
                        sign_extend_raw(raw, field.width) as i32
                    } else {
                        raw as i32
                    };
                    Operand::Immediate(value * scale)
                }
                OperandType::LockId => Operand::Lock(raw as u8),
                OperandType::Unknown => {
                    // Fields named "dontcare*" are reserved/padding -- skip silently.
                    if field.name.starts_with("dontcare") {
                        continue;
                    }
                    panic!(
                        "Unhandled OperandType::Unknown for field '{}' (raw=0x{:X}) in encoding '{}'. \
                         Add the register class to classify_operand_type() in resolver.rs.",
                        field.name, raw, decoded.encoding.name
                    );
                }
            };

            field_operands.insert(field.name.clone(), operand);
        }

        // Combine separate ptr + imm fields into a Memory operand.
        // The tblgen-resolved encodings have separate "ptr" (PointerReg) and
        // "imm" (Immediate, already scaled) fields. Combining them here into
        // a single Memory { base, offset } operand means the execution side
        // doesn't need to know about the scale factor or field layout.
        //
        // IMPORTANT: Only handle IndexedImmediate here. IndexedRegister uses
        // a modifier register (dj) as the offset, which is decoded from the
        // AG field in decode_ag_field(). If we match IndexedRegister here and
        // find "ptr"+"imm" fields, we'd create a Memory { base, offset: 0 }
        // operand that ignores the dj register, producing wrong addresses.
        let addr_mode = decoded.encoding.addressing_mode;
        if matches!(addr_mode, AddressingMode::IndexedImmediate) {
            if let (Some(Operand::PointerReg(base)), Some(imm_op)) =
                (field_operands.get("ptr").cloned(), field_operands.get("imm").cloned())
            {
                let offset = match imm_op {
                    Operand::Immediate(v) => v as i16,
                    _ => 0,
                };
                if decoded.encoding.is_ptr_arithmetic {
                    // padd: ptr is destination, imm is source operand
                    direct_dest = Some(Operand::PointerReg(base));
                    extra_sources.push(Operand::Immediate(offset as i32));
                } else {
                    extra_sources.push(Operand::Memory { base, offset });
                }
                // Remove ptr and imm from field_operands so they don't
                // appear as separate sources in extract_ordered_operands
                field_operands.remove("ptr");
                field_operands.remove("imm");
            }
        } else if matches!(addr_mode, AddressingMode::PostModifyImmediate) {
            // Post-modify: ptr is the base address, imm is the modify amount
            if let (Some(Operand::PointerReg(base)), Some(imm_op)) =
                (field_operands.get("ptr").cloned(), field_operands.get("imm").cloned())
            {
                let modify_amount = match imm_op {
                    Operand::Immediate(v) => v as i16,
                    _ => 0,
                };
                extra_sources.push(Operand::PointerReg(base));
                extracted_post_modify = Some(PostModify::Immediate(modify_amount));
                field_operands.remove("ptr");
                field_operands.remove("imm");
            }
        } else if matches!(addr_mode, AddressingMode::PostModifyRegister) {
            if let (Some(Operand::PointerReg(base)), Some(mod_op)) =
                (field_operands.get("ptr").cloned(), field_operands.get("mod").cloned())
            {
                let mod_reg = match mod_op {
                    Operand::ModifierReg(r) => r,
                    _ => 0,
                };
                extra_sources.push(Operand::PointerReg(base));
                extracted_post_modify = Some(PostModify::Register(mod_reg));
                field_operands.remove("ptr");
                field_operands.remove("mod");
            }
        }

        // Pointer arithmetic with register offset (padda/paddb/padds [ptr], mod).
        // The addressing mode may be IndexedRegister or Unknown depending on the
        // instruction name, but for pointer arithmetic the semantics are the same:
        // ptr += mod (or ptr += dj from the AG field).
        if decoded.encoding.is_ptr_arithmetic && direct_dest.is_none() {
            if let Some(Operand::PointerReg(base)) = field_operands.get("ptr").cloned() {
                direct_dest = Some(Operand::PointerReg(base));
                // If there's a modifier register, use it as the offset source.
                if let Some(mod_op) = field_operands.get("mod").cloned() {
                    extra_sources.push(mod_op);
                    field_operands.remove("mod");
                }
                field_operands.remove("ptr");
            }
        }

        // SP-relative load/store (spill/fill): Uses = [SP] in TableGen.
        // The immediate field is a stack offset, not a standalone value.
        // Convert to Memory { base: SP_PTR_INDEX, offset: imm }.
        // AIE2's SP is a dedicated register (SPLReg<12>), not p6.
        // Exclude pointer arithmetic (padda/paddb [sp]) -- those have their
        // own implicit SP handling in execute_pointer_add.
        if decoded.encoding.is_sp_relative && !decoded.encoding.is_ptr_arithmetic {
            if let Some(Operand::Immediate(offset)) = field_operands.get("imm").cloned() {
                extra_sources.push(Operand::Memory {
                    base: crate::interpreter::state::SP_PTR_INDEX,
                    offset: offset as i16,
                });
                field_operands.remove("imm");
            }
        }

        // Extract dest and sources using TableGen ordering
        let (dest, sources) =
            self.extract_ordered_operands(decoded, &field_operands, direct_dest, extra_sources);

        // Legacy path doesn't extract extra defs (dual-result support is
        // FFI-only).
        (dest, sources, extracted_post_modify, vec![])
    }

    /// Decode a register operand from raw bits, applying subclass base offset.
    #[cfg(test)]
    fn decode_register(kind: RegisterKind, raw: u8, base_offset: u8) -> Operand {
        let reg = raw + base_offset;
        match kind {
            RegisterKind::Scalar => Operand::ScalarReg(reg),
            RegisterKind::Pointer => Operand::PointerReg(reg),
            RegisterKind::ModifierM => Operand::ModifierReg(reg + MOD_BASE_M),
            RegisterKind::ModifierDN => Operand::ModifierReg(reg + MOD_BASE_DN),
            RegisterKind::ModifierDJ => Operand::ModifierReg(reg + MOD_BASE_DJ),
            RegisterKind::ModifierDC => Operand::ModifierReg(reg + MOD_BASE_DC),
            RegisterKind::Vector256 => Operand::VectorReg(reg),
            // Vector512 (x-registers) span two 256-bit registers: x0={wl0,wh0}={v0,v1}.
            // The 4-bit encoding index needs *2 to address the 256-bit register file.
            RegisterKind::Vector512 => Operand::VectorReg(reg * 2),
            // Vector1024 (y-registers) span four 256-bit registers: y2={v8,v9,v10,v11}.
            // The 2-bit encoding index needs *4 to address the 256-bit register file.
            RegisterKind::Vector1024 => Operand::VectorReg(reg * 4),
            RegisterKind::Accumulator => Operand::AccumReg(reg),
            RegisterKind::Control => Operand::ControlReg(reg),
            RegisterKind::SparseQx => Operand::SparseQxReg(reg),
            // ScalarPair: 64-bit pair, encoding is pair index.
            // Emit the even register (index * 2 + base_offset); the execution
            // handler reads reg+1 for the upper 32 bits.
            RegisterKind::ScalarPair => Operand::ScalarReg(raw * 2 + base_offset),
        }
    }

    /// Minimal inline composite register decode for test cross-validation.
    ///
    /// Covers the scalar/pointer/accumulator patterns needed by tests.
    /// Full composite decoding is handled by the LLVM FFI layer in production.
    #[cfg(test)]
    fn decode_composite_inline(encoder: CompositeEncoder, raw: u64) -> Operand {
        use crate::interpreter::state::LR_REG_INDEX;
        match encoder {
            // MvSclSrc/MvSclDst: scalar=raw>>2 (low 2 bits=00), pointer=(raw>>4) (low 4=0011)
            CompositeEncoder::MvSclSrc => {
                if raw & 0x7 == 0b111 {
                    // Special registers: id = (raw >> 3) & 0xF
                    use crate::interpreter::state::*;
                    match (raw >> 3) & 0xF {
                        0 => Operand::ScalarReg(LS_REG_INDEX),
                        2 => Operand::ScalarReg(DP_REG_INDEX),
                        4 => Operand::ScalarReg(LR_REG_INDEX),
                        6 => Operand::ScalarReg(CORE_ID_REG_INDEX),
                        8 => Operand::ScalarReg(LE_REG_INDEX),
                        10 => Operand::ScalarReg(LC_REG_INDEX),
                        12 => Operand::PointerReg(SP_PTR_INDEX),
                        _ => Operand::ScalarReg(0),
                    }
                } else if raw & 0xF == 0b0011 {
                    Operand::PointerReg(((raw >> 4) & 0x7) as u8)
                } else if raw & 0x3 == 0b00 {
                    Operand::ScalarReg(((raw >> 2) & 0x1F) as u8)
                } else {
                    Operand::Immediate(raw as i32)
                }
            }
            // LdaScl: lr=0b0000101, pointer=(raw>>4)|0b1101, scalar=raw>>2
            CompositeEncoder::LdaScl => {
                if raw == 0b0000101 {
                    Operand::ScalarReg(LR_REG_INDEX)
                } else if raw & 0xF == 0b1101 {
                    Operand::PointerReg(((raw >> 4) & 0x7) as u8)
                } else if raw & 0x3 == 0b00 {
                    Operand::ScalarReg(((raw >> 2) & 0x1F) as u8)
                } else {
                    Operand::Immediate(raw as i32)
                }
            }
            // AluCg: LC=0b000001, scalar=raw>>1
            CompositeEncoder::AluCg => Operand::ScalarReg((raw >> 1) as u8),
            // ERS4: r16..r31
            CompositeEncoder::ERS4 => Operand::ScalarReg((raw as u8).wrapping_add(16)),
            // Fallback for other encoders
            _ => Operand::Immediate(raw as i32),
        }
    }

    /// AG fields encode a packed tuple of {pointer, offset/modifier, mode_bits}.
    #[cfg(test)]
    fn decode_ag_field(
        &self,
        field: &xdna_archspec::aie2::isa::OperandField,
        value: u64,
        decoded: &DecodedInstr,
        direct_dest: &mut Option<Operand>,
        extra_sources: &mut Vec<Operand>,
        extracted_post_modify: &mut Option<PostModify>,
    ) {
        let addr_mode = decoded.encoding.addressing_mode;
        let is_ag_all = field.width >= 13;

        match addr_mode {
            AddressingMode::IndexedImmediate => {
                let (mode_bits, imm_bits) = if is_ag_all { (4, 6) } else { (3, 3) };
                let data = value >> mode_bits;
                let ptr = ((data >> imm_bits) & 0x7) as u8;
                let imm_raw = (data & ((1 << imm_bits) - 1)) as i32;
                let imm_bytes = if is_ag_all { imm_raw * 4 } else { imm_raw };
                if decoded.encoding.is_ptr_arithmetic {
                    *direct_dest = Some(Operand::PointerReg(ptr));
                    extra_sources.push(Operand::Immediate(imm_bytes));
                } else {
                    extra_sources.push(Operand::Memory { base: ptr, offset: imm_bytes as i16 });
                }
            }
            AddressingMode::PostModifyImmediate => {
                let (mode_bits, imm_bits) = if is_ag_all { (3, 7) } else { (2, 4) };
                let data = value >> mode_bits;
                let ptr = ((data >> imm_bits) & 0x7) as u8;
                let imm_raw = sign_extend((data & ((1 << imm_bits) - 1)) as i32, imm_bits as u32);
                let imm_bytes = if is_ag_all { imm_raw * 4 } else { imm_raw };
                extra_sources.push(Operand::PointerReg(ptr));
                *extracted_post_modify = Some(PostModify::Immediate(imm_bytes as i16));
            }
            AddressingMode::IndexedRegister => {
                // Indexed addressing uses dj (stride) registers: addr = ptr + dj_n
                let ptr = ((value >> (field.width as u64 - 3)) & 0x7) as u8;
                let mod_reg = ((value >> 3) & 0x7) as u8 + MOD_BASE_DJ;
                if decoded.encoding.is_ptr_arithmetic {
                    *direct_dest = Some(Operand::PointerReg(ptr));
                    extra_sources.push(Operand::ModifierReg(mod_reg));
                } else {
                    extra_sources.push(Operand::PointerReg(ptr));
                    extra_sources.push(Operand::ModifierReg(mod_reg));
                }
            }
            AddressingMode::PostModifyRegister => {
                // Post-modify uses m (modifier) registers: addr = ptr, then ptr += m_n
                let ptr = ((value >> (field.width as u64 - 3)) & 0x7) as u8;
                let mod_reg = ((value >> 3) & 0x7) as u8 + MOD_BASE_M;
                extra_sources.push(Operand::PointerReg(ptr));
                *extracted_post_modify = Some(PostModify::Register(mod_reg));
            }
            AddressingMode::Unknown => {
                // Fallback: use mode-bit heuristic for ag_* fields without
                // a clear addressing mode (e.g., padd with no _idx/_pstm suffix)
                let mode = (value & 0xF) as u8;
                let (ptr_reg, offset_or_mod) = match mode {
                    0b1010 => {
                        let ag_ptr_imm = value >> 4;
                        let ptr = ((ag_ptr_imm >> 6) & 0x7) as u8;
                        let imm_words = (ag_ptr_imm & 0x3F) as i32;
                        (ptr, Operand::Immediate(imm_words * 4))
                    }
                    0b0110 => {
                        let ag_ptr_imm = value >> 3;
                        let ptr = ((ag_ptr_imm >> 7) & 0x7) as u8;
                        let imm = (ag_ptr_imm & 0x7F) as i32;
                        (ptr, Operand::Immediate(imm))
                    }
                    0b0010 => {
                        let ptr = ((value >> 10) & 0x7) as u8;
                        let mod_reg = ((value >> 3) & 0x7) as u8;
                        (ptr, Operand::ModifierReg(mod_reg))
                    }
                    _ => {
                        let ptr = (value & 0x7) as u8;
                        let mod_reg = ((value >> 3) & 0x1F) as u8;
                        (ptr, Operand::ModifierReg(mod_reg))
                    }
                };

                if decoded.encoding.is_ptr_arithmetic {
                    *direct_dest = Some(Operand::PointerReg(ptr_reg));
                    extra_sources.push(offset_or_mod);
                } else if let Operand::Immediate(imm) = offset_or_mod {
                    extra_sources.push(Operand::Memory { base: ptr_reg, offset: imm as i16 });
                } else {
                    extra_sources.push(Operand::PointerReg(ptr_reg));
                    extra_sources.push(offset_or_mod);
                }
            }
        }
    }

    /// Extract destination and sources using TableGen's output_order/input_order.
    #[cfg(test)]
    fn extract_ordered_operands(
        &self,
        decoded: &DecodedInstr,
        field_operands: &std::collections::HashMap<String, Operand>,
        direct_dest: Option<Operand>,
        mut extra_sources: Vec<Operand>,
    ) -> (Option<Operand>, Vec<Operand>) {
        let output_order = &decoded.encoding.output_order;
        let input_order = &decoded.encoding.input_order;

        // Extract destination from output_order, or use direct_dest from special handling
        let dest = if let Some(d) = direct_dest {
            Some(d)
        } else if !output_order.is_empty() {
            // Look up first output name in field_operands
            output_order.first().and_then(|name| field_operands.get(name).cloned())
                .or_else(|| {
                    log::trace!(
                        "[DECODE] output_order name '{}' not in field_operands for '{}', falling back to dest heuristic",
                        output_order.first().map(|s| s.as_str()).unwrap_or("?"),
                        decoded.encoding.name,
                    );
                    self.find_dest_heuristic(field_operands)
                })
        } else {
            // No output_order - use heuristic
            log::trace!(
                "[DECODE] empty output_order for '{}', falling back to dest heuristic",
                decoded.encoding.name,
            );
            self.find_dest_heuristic(field_operands)
        };

        // Validate: dest must be a writable operand type.
        // If an Immediate or other non-writable operand ended up here (from a
        // heuristic mismatch or bad output_order resolution), move it to sources.
        // This is expected for some Chess-compiled instructions (VMOV_mv_scd,
        // VMOV_mv_cm, VGE_D32) where the TableGen output_order puts a config
        // immediate in the dest position. The fallback is correct: treat it as
        // a source operand instead.
        let dest = match dest {
            Some(d) if !can_be_dest(&d) => {
                log::trace!(
                    "[DECODE] Non-writable {:?} in dest for '{}', moving to sources",
                    d,
                    decoded.encoding.name,
                );
                extra_sources.insert(0, d);
                None
            }
            other => other,
        };

        // Extract sources from input_order
        let mut sources = Vec::new();

        if !input_order.is_empty() {
            // Use TableGen ordering - this is the correct order
            for input_name in input_order {
                if let Some(operand) = field_operands.get(input_name) {
                    sources.push(operand.clone());
                } else {
                    log::trace!(
                        "[DECODE] input_order name '{}' not found in field_operands (keys: {:?})",
                        input_name,
                        field_operands.keys().collect::<Vec<_>>()
                    );
                }
            }
        } else {
            // No input_order - use all non-dest operands (heuristic fallback)
            log::trace!(
                "[DECODE] empty input_order for '{}', using is_dest_field heuristic for source filtering",
                decoded.encoding.name,
            );
            for (name, operand) in field_operands.iter() {
                if !self.is_dest_field(name) {
                    sources.push(operand.clone());
                }
            }
        }

        // Append extra sources from special handling (ag_*, etc.)
        sources.extend(extra_sources);

        (dest, sources)
    }

    /// Heuristic to find destination from field_operands map.
    #[cfg(test)]
    fn find_dest_heuristic(
        &self,
        field_operands: &std::collections::HashMap<String, Operand>,
    ) -> Option<Operand> {
        for (name, operand) in field_operands {
            if self.is_dest_field(name) {
                return Some(operand.clone());
            }
        }
        None
    }

    /// Legacy safety net: check if a field name represents a destination.
    #[cfg(test)]
    fn is_dest_field(&self, name: &str) -> bool {
        (name.contains("mRx") && !name.contains("mRx0"))
            || name.starts_with("d")
            || name == "dst"
            || name.contains("SclLd")
            || name.contains("mLdaScl")
            || name.contains("mLdbScl")
            || name.contains("mLdaCg")
            || name.contains("mLdbCg")
            || name.contains("mMvSclDstCg")
            || name == "mMvSclDst"
    }
}
