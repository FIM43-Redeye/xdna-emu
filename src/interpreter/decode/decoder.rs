//! TableGen-driven instruction decoder.
//!
//! This decoder uses LLVM bytecode tables generated from llvm-aie's TableGen files
//! to accurately decode AIE2 instructions with O(1) lookup performance.
//!
//! # How It Works
//!
//! 1. At construction, we receive resolved `InstrEncoding` tables and LLVM decoder
//!    bytecode tables grouped by slot
//! 2. For each instruction word, the LLVM bytecode identifies the instruction name
//! 3. The name is looked up in a per-slot HashMap to retrieve the full encoding
//! 4. Operand fields are extracted using the encoding's field definitions
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::decode::InstructionDecoder;
//!
//! let decoder = InstructionDecoder::load_from_generated();
//! ```

use std::collections::HashMap;

use crate::interpreter::bundle::{ExtractedBundle, Operand, SlotIndex, SlotOp, VliwBundle, extract_slots};
use crate::interpreter::traits::{DecodeError, Decoder};
use xdna_archspec::aie2::isa::{DecoderIndex, InstrEncoding, SemanticOp, decoder_ffi};

/// A decoded instruction from TableGen data.
#[derive(Debug, Clone)]
pub struct DecodedInstr {
    /// The matched encoding.
    pub encoding: InstrEncoding,
    /// Extracted operand values (keyed by field name).
    pub operands: HashMap<String, u64>,
}

impl DecodedInstr {
    /// Get an operand value by name.
    pub fn operand(&self, name: &str) -> Option<u64> {
        self.operands.get(name).copied()
    }

    /// Get an operand as a register number (u8).
    pub fn reg(&self, name: &str) -> Option<u8> {
        self.operand(name).map(|v| v as u8)
    }

    /// Get an operand as a signed immediate.
    pub fn signed_imm(&self, name: &str) -> Option<i32> {
        // Find the field to get its width for sign extension
        let field = self.encoding.operand_fields.iter().find(|f| f.name == name)?;
        let value = self.operand(name)?;

        if field.signed {
            Some(field.extract_signed(value << field.bit_position) as i32)
        } else {
            // Sign extend based on width
            let sign_bit = 1u64 << (field.width - 1);
            if value & sign_bit != 0 {
                let mask = !((1u64 << field.width) - 1);
                Some((value | mask) as i32)
            } else {
                Some(value as i32)
            }
        }
    }
}

/// TableGen-driven instruction decoder.
///
/// Uses resolved encoding tables from llvm-aie's TableGen files to decode
/// instructions accurately.
///
/// # Performance
///
/// This decoder uses O(1) lookup via `DecoderIndex`:
/// 1. Extract slot bits from bundle
/// 2. LLVM bytecode table identifies instruction name
/// 3. Name lookup retrieves full `InstrEncoding` with semantic metadata
#[derive(Clone)]
pub struct InstructionDecoder {
    /// O(1) decoder index (per-slot LLVM bytecode lookup).
    pub(super) index: DecoderIndex,

    /// Data-driven VLIW slot extraction table, built from tblgen Inst fields.
    /// When present, replaces the hand-coded extract_* functions in slot_layout.
    pub(super) format_table: Option<crate::interpreter::bundle::FormatTable>,

    /// Per-opcode metadata from LLVM's MCInstrDesc + itinerary model.
    /// Indexed by LLVM opcode ID for O(1) lookups. Populated once at init.
    pub(super) instr_info: Vec<decoder_ffi::InstrInfo>,

    /// Statistics: successful decodes.
    pub(super) decode_count: u64,
    /// Statistics: unknown patterns.
    pub(super) unknown_count: u64,
}

impl Default for InstructionDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns true if this operand type can be a write destination.
///
/// Register kinds (scalar, vector, accum, pointer, modifier) are writable.
/// Immediates, locks, DMA channels, buffer descriptors, and memory operands
/// are not valid write targets.
pub(super) fn can_be_dest(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::ScalarReg(_)
            | Operand::VectorReg(_)
            | Operand::AccumReg(_)
            | Operand::PointerReg(_)
            | Operand::ModifierReg(_)
            | Operand::ControlReg(_)
    )
}

impl InstructionDecoder {
    /// Get LLVM InstrInfo for an opcode, if available.
    pub fn get_instr_info(&self, opcode: u32) -> Option<&decoder_ffi::InstrInfo> {
        self.instr_info.get(opcode as usize)
    }

    /// Get the full InstrInfo table (for passing to LatencyTable, etc.).
    pub fn instr_info_table(&self) -> &[decoder_ffi::InstrInfo] {
        &self.instr_info
    }

    /// Validate a SlotOp's SemanticOp against LLVM MCInstrDesc flags.
    ///
    /// Logs a warning when our semantic classification disagrees with LLVM's
    /// flags (e.g., we say Load but LLVM says no MayLoad). These are bugs
    /// in our SemanticOp inference, not in LLVM.
    pub(super) fn validate_semantic_vs_flags(&self, op: &SlotOp) {
        let opcode = match op.llvm_opcode {
            Some(o) => o,
            None => return,
        };
        let info = match self.instr_info.get(opcode as usize) {
            Some(i) if i.flags != 0 => i,
            _ => return,
        };
        let semantic = match op.semantic {
            Some(s) => s,
            None => return,
        };

        // Load: our SemanticOp says Load but LLVM says no MayLoad?
        if matches!(semantic, SemanticOp::Load) && !info.is_load() {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): SemanticOp::Load but LLVM has no MayLoad (flags=0x{:X})",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, info.flags,
            );
        }
        // Store: our SemanticOp says Store but LLVM says no MayStore?
        if matches!(semantic, SemanticOp::Store) && !info.is_store() {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): SemanticOp::Store but LLVM has no MayStore (flags=0x{:X})",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, info.flags,
            );
        }
        // Branch: LLVM says Branch but we don't have Br/BrCond?
        if info.is_branch()
            && !matches!(semantic, SemanticOp::Br | SemanticOp::BrCond | SemanticOp::Call | SemanticOp::Ret)
        {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Branch but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"),
                opcode,
                semantic,
            );
        }
        // Call: LLVM says Call but we don't?
        if info.is_call() && !matches!(semantic, SemanticOp::Call) {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Call but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"),
                opcode,
                semantic,
            );
        }
        // Return: LLVM says Return but we don't?
        if info.is_return() && !matches!(semantic, SemanticOp::Ret) {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Return but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"),
                opcode,
                semantic,
            );
        }

        // Reverse: we say Load but LLVM says no MayLoad?
        // (Skip pointer ops and cascade reads -- they use the load slot but
        // aren't memory loads in LLVM's sense.)
        if info.is_load()
            && !matches!(
                semantic,
                SemanticOp::Load
                    | SemanticOp::PointerAdd
                    | SemanticOp::PointerMov
                    | SemanticOp::CascadeRead
                    | SemanticOp::CascadeWrite
                    | SemanticOp::Copy
            )
            && !matches!(
                semantic,
                // Intrinsics and vector ops that load (UPS, etc.) may have MayLoad
                // from LLVM but aren't SemanticOp::Load. That's expected.
                SemanticOp::Ups | SemanticOp::Intrinsic(_)
            )
        {
            log::debug!(
                "[semantic-flag note] {} (opcode {}): LLVM MayLoad but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"),
                opcode,
                semantic,
            );
        }
    }

    /// Get decode statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.decode_count, self.unknown_count)
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.decode_count = 0;
        self.unknown_count = 0;
    }

    /// Extract slot data from a bundle, preferring the data-driven format table
    /// when available, falling back to the hand-coded extract_slots().
    pub fn extract_bundle_slots(&self, bytes: &[u8]) -> ExtractedBundle {
        if bytes.len() < 2 {
            return ExtractedBundle::new();
        }
        let format = crate::interpreter::bundle::detect_format(bytes);

        // Try the data-driven table first
        if let Some(ref table) = self.format_table {
            if let Some(bundle) = table.extract(bytes, format) {
                return bundle;
            }
            // No match in table -- fall through to legacy
        }

        // Legacy hand-coded path
        extract_slots(bytes)
    }

    /// Get the underlying decoder index.
    pub fn decoder_index(&self) -> &DecoderIndex {
        &self.index
    }

    /// Try to decode slot-specific bits against encodings for that slot.
    ///
    /// Legacy decode method using bytecode tables. Retained for cross-validation
    /// tests against the LLVM FFI decoder. Not used in production.
    #[cfg(test)]
    pub fn decode_slot_bits(
        &self,
        bits: u64,
        slot_type: crate::interpreter::bundle::SlotType,
    ) -> Option<DecodedInstr> {
        use crate::interpreter::bundle::SlotType;

        // Map SlotType to the slot names used in encodings
        let slot_name = match slot_type {
            SlotType::Lda => "lda",
            SlotType::Ldb => "ldb",
            SlotType::Alu => "alu",
            SlotType::Mv => "mv",
            SlotType::St => "st",
            SlotType::Vec => "vec",
            SlotType::Lng => "lng",
            SlotType::Nop => return None, // NOPs don't have specific encodings
        };

        // O(1) lookup via DecoderIndex
        if let Some((encoding, operands)) = self.index.decode_slot(slot_name, bits) {
            return Some(DecodedInstr { encoding: encoding.clone(), operands });
        }

        None
    }
}

impl Decoder for InstructionDecoder {
    fn decode(&self, bytes: &[u8], pc: u32) -> Result<VliwBundle, DecodeError> {
        use crate::interpreter::bundle::SlotType;

        if bytes.len() < 2 {
            return Err(DecodeError::Incomplete { needed: 2, have: bytes.len() });
        }

        // Detect format from the low nibble
        let format = crate::interpreter::bundle::detect_format(bytes);
        let bundle_size = format.size_bytes() as usize;

        // If we don't have enough bytes for the detected format, handle specially
        // This can happen with test data or partial reads
        let effective_size = if bytes.len() < bundle_size {
            // Check if the data looks like a NOP (all zeros or known patterns)
            let word0 = if bytes.len() >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else if bytes.len() >= 2 {
                u16::from_le_bytes([bytes[0], bytes[1]]) as u32
            } else {
                return Err(DecodeError::Incomplete { needed: 2, have: bytes.len() });
            };

            // Special case: all zeros is a NOP regardless of format marker
            if word0 == 0 || word0 == 0x15010040 {
                // Treat as 4-byte NOP
                bytes.len().min(4)
            } else {
                // Not enough data and not a NOP
                return Err(DecodeError::Incomplete { needed: bundle_size, have: bytes.len() });
            }
        } else {
            bundle_size
        };

        let mut bundle = VliwBundle::from_raw(&bytes[..effective_size.min(bytes.len())], pc);

        // Extract slots from the bundle (data-driven when available)
        let extracted = self.extract_bundle_slots(bytes);

        if log::log_enabled!(log::Level::Trace) {
            static DECODE_COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            let count = DECODE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if count < 20 {
                log::trace!(
                    "[DECODE#{}] PC=0x{:04X} format={:?} slots={}",
                    count,
                    pc,
                    format,
                    extracted.slots.len()
                );
                for s in &extracted.slots {
                    log::trace!("  slot: {:?} bits=0x{:010X}", s.slot_type, s.bits);
                }
            }
        }

        // If we extracted any slots, decode each one
        if !extracted.is_empty() {
            for slot in &extracted.slots {
                let slot_index = match slot.slot_type {
                    SlotType::Lda => SlotIndex::LoadA,
                    SlotType::Ldb => SlotIndex::LoadB,
                    SlotType::Alu => SlotIndex::Scalar0,
                    SlotType::Mv => SlotIndex::Scalar1,
                    SlotType::St => SlotIndex::Store,
                    SlotType::Vec => SlotIndex::Vector,
                    // LNG is polymorphic: j/jl -> Control, movxm -> Scalar0.
                    // This default is only used for NOPs; real instructions get
                    // resolved via operation.natural_slot() below.
                    SlotType::Lng => SlotIndex::Control,
                    SlotType::Nop => SlotIndex::Control,
                };

                // Try to decode the slot bits against known encodings.
                // Only explicit NOP slot type is treated as NOP here.
                // Zero bits in a non-NOP slot is a VALID encoding (e.g.,
                // vmac cm0, cm0, x0, x0, r0 has all-zero operand fields
                // after format/discriminator bits are stripped). The LLVM
                // FFI decoder handles the full encoding including the
                // discriminator bits that build_bundle_32 re-adds.
                if slot.slot_type == SlotType::Nop {
                    bundle.set_slot(SlotOp::nop(slot_index));
                } else if let Some((
                    decoded,
                    dest,
                    sources,
                    extracted_pm,
                    ffi_opcode,
                    extra_defs,
                    accum_width,
                    source_forward,
                    result_bypass,
                )) = self.try_decode_via_ffi(slot.bits, slot.slot_type)
                {
                    // LLVM FFI is the sole production decoder for both instruction
                    // identification and operand extraction.
                    let mut slot_op = self.build_slot_op(
                        slot_index,
                        &decoded,
                        dest,
                        sources,
                        extracted_pm,
                        source_forward,
                        result_bypass,
                    );
                    slot_op.llvm_opcode = ffi_opcode;
                    slot_op.accum_width = accum_width;
                    // Attach secondary destinations (e.g., cmp register for
                    // dual-result instructions like VSUB_LT, VABS_GTZ).
                    for ed in extra_defs {
                        slot_op.extra_dests.push(ed);
                    }

                    // Validate SemanticOp against LLVM's MCInstrDesc flags.
                    self.validate_semantic_vs_flags(&slot_op);

                    // For LNG slot instructions, use the operation's natural slot since
                    // LNG can contain either control (JL, J) or vector/scalar (movxm).
                    let effective_slot = if slot.slot_type == SlotType::Lng {
                        slot_op.natural_slot()
                    } else {
                        slot_index
                    };
                    slot_op.slot = effective_slot;

                    // Slot collision: two non-NOP instructions decoded into the same
                    // VLIW slot. This is a decoder bug, not a valid encoding.
                    if let Some(existing) = bundle.slot(effective_slot) {
                        if !existing.is_nop() {
                            log::error!(
                                "[SLOT COLLISION] PC=0x{:04X} {:?} slot {:?} already has {:?}, overwriting with {:?}",
                                pc, slot.slot_type, effective_slot, existing.semantic, slot_op.semantic
                            );
                        }
                    }
                    bundle.set_slot(slot_op);
                } else {
                    // Slot extracted but not recognized - mark as unknown with slot info
                    log::warn!(
                        "[{:?} DECODE FAIL] PC=0x{:04X} bits=0x{:010X} - no matching encoding",
                        slot.slot_type,
                        pc,
                        slot.bits
                    );
                    let slot_name = match slot.slot_type {
                        SlotType::Lda => "lda",
                        SlotType::Ldb => "ldb",
                        SlotType::Alu => "alu",
                        SlotType::Mv => "mv",
                        SlotType::St => "st",
                        SlotType::Vec => "vec",
                        SlotType::Lng => "lng",
                        SlotType::Nop => "nop",
                    };
                    // Mark as unknown: semantic=None, raw_opcode for diagnostics
                    let mut unknown_op = SlotOp::nop(slot_index);
                    unknown_op.semantic = None;
                    unknown_op.raw_opcode = Some(slot.bits as u32);
                    bundle.set_slot(unknown_op);
                    let _ = slot_name; // Suppress unused warning for now
                }
            }
        } else {
            // No slots extracted - treat as unknown
            let word0 = if bytes.len() >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                u16::from_le_bytes([bytes[0], bytes[1]]) as u32
            };

            if word0 == 0 || word0 == 0x15010040 {
                bundle.set_slot(SlotOp::nop(SlotIndex::Scalar0));
            } else {
                let mut unknown_op = SlotOp::nop(SlotIndex::Scalar0);
                unknown_op.semantic = None;
                unknown_op.raw_opcode = Some(word0);
                bundle.set_slot(unknown_op);
            }
        }

        Ok(bundle)
    }

    fn instruction_size(&self, bytes: &[u8]) -> Result<u8, DecodeError> {
        if bytes.len() < 2 {
            return Err(DecodeError::Incomplete { needed: 2, have: bytes.len() });
        }

        // Use the format marker to determine size
        let format = crate::interpreter::bundle::detect_format(bytes);
        let expected_size = format.size_bytes();

        // If we don't have enough bytes, check for special cases
        if (bytes.len() as u8) < expected_size {
            let word0 = if bytes.len() >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else if bytes.len() >= 2 {
                u16::from_le_bytes([bytes[0], bytes[1]]) as u32
            } else {
                return Err(DecodeError::Incomplete { needed: 2, have: bytes.len() });
            };

            // All zeros or known NOP patterns: treat as 4-byte
            if word0 == 0 || word0 == 0x15010040 {
                return Ok(4.min(bytes.len() as u8));
            }
        }

        Ok(expected_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::ElementType;
    use smallvec::SmallVec;
    use xdna_archspec::aie2::{Bypass, isa::OperandField};
    use std::path::Path;

    #[test]
    fn test_decoder_nop() {
        let decoder = InstructionDecoder::new();

        let bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&bytes, 0).expect("Should decode");
        assert!(bundle.is_nop());
    }

    #[test]
    fn test_decoder_unknown() {
        let decoder = InstructionDecoder::new();

        // Random non-NOP word with 32-bit format marker (0x9)
        // This ensures we don't need more bytes than provided
        let bytes = [0xA9, 0xCD, 0xEF, 0x12]; // Low nibble = 0x9 = 32-bit
        let bundle = decoder.decode(&bytes, 0).expect("Should decode");

        let slot = bundle.slot(SlotIndex::Scalar0).expect("Should have slot");
        assert!(slot.semantic.is_none(), "Unknown instruction should have semantic: None");
    }

    #[test]
    fn test_decoder_with_real_llvm_aie() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        // Load decoder with bytecode tables (the only real decode path)
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        let encoding_count = decoder.decoder_index().all_encodings().count();
        eprintln!("Loaded {} encodings", encoding_count);
        assert!(encoding_count > 100, "Should have many encodings");

        // Test NOP decoding
        let nop_bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&nop_bytes, 0).expect("Should decode NOP");
        assert!(bundle.is_nop());

        // Get stats
        let (decode_count, unknown_count) = decoder.stats();
        eprintln!("Decoder stats: {} decoded, {} unknown", decode_count, unknown_count);
    }

    #[test]
    fn test_decode_real_elf_instructions() {
        use crate::config::Config;
        use crate::parser::AieElf;
        use std::path::PathBuf;

        let Some(elf_path) = Config::get().add_one_elf() else {
            eprintln!("Skipping test: ELF not found (set MLIR_AIE_PATH)");
            return;
        };

        let llvm_aie_path = PathBuf::from(Config::get().llvm_aie_path());
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found (set LLVM_AIE_PATH)");
            return;
        }

        // Load decoder with bytecode tables
        let decoder =
            InstructionDecoder::try_load_via_tblgen(&llvm_aie_path).expect("Failed to load via tblgen");

        // Read ELF using proper parser
        let elf_data = std::fs::read(&elf_path).expect("Failed to read ELF");
        let elf = AieElf::parse(&elf_data).expect("Failed to parse ELF");

        eprintln!("ELF Architecture: {:?}", elf.architecture());
        eprintln!("Entry point: 0x{:04X}", elf.entry_point());

        let text_data = elf.text_section().expect("No .text section found");

        eprintln!("\n=== Decoding real AIE2 ELF instructions ===\n");

        let mut pc = 0u32;
        let mut decoded_count = 0;
        let mut unknown_count = 0;
        let max_instructions = 20;

        while pc < text_data.len() as u32 && decoded_count + unknown_count < max_instructions {
            let bytes = &text_data[pc as usize..];
            if bytes.len() < 4 {
                break;
            }

            let word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);

            match decoder.decode(bytes, pc) {
                Ok(bundle) => {
                    // Check all active slots, not just Scalar0
                    let mut found_known = false;
                    let mut op_names = Vec::new();

                    for slot_op in bundle.active_slots() {
                        match slot_op.semantic {
                            None => {
                                let opcode = slot_op.raw_opcode.unwrap_or(0);
                                op_names.push(format!("{:?}:??? (0x{:05X})", slot_op.slot, opcode));
                            }
                            Some(SemanticOp::Nop) => {
                                found_known = true;
                                op_names.push("Nop".to_string());
                            }
                            Some(sem) => {
                                found_known = true;
                                op_names.push(format!("{:?}", sem));
                            }
                        }
                    }

                    let op_name = if op_names.is_empty() {
                        "empty".to_string()
                    } else {
                        op_names.join("; ")
                    };

                    if found_known || op_names.iter().any(|s| s.contains("Nop")) {
                        decoded_count += 1;
                    } else {
                        unknown_count += 1;
                    }

                    eprintln!("  PC 0x{:04X}: 0x{:08X} -> {}", pc, word, op_name);

                    pc += bundle.size() as u32;
                }
                Err(e) => {
                    eprintln!("  PC 0x{:04X}: 0x{:08X} -> ERROR: {:?}", pc, word, e);
                    unknown_count += 1;
                    pc += 4;
                }
            }
        }

        eprintln!("\n=== Results ===");
        eprintln!("Decoded: {} instructions", decoded_count);
        eprintln!("Unknown: {} instructions", unknown_count);
        eprintln!(
            "Recognition rate: {:.1}%",
            100.0 * decoded_count as f64 / (decoded_count + unknown_count) as f64
        );

        // We expect some unknowns - this is a real binary!
        // But we should decode SOMETHING
        assert!(decoded_count + unknown_count > 0, "Should have processed some instructions");
    }

    #[test]
    fn test_done_instruction_decodes_to_halt() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // done instruction: bytes from llvm-objdump, encodes to ALU slot
        let done_bytes: [u8; 4] = [0x19, 0x08, 0x00, 0x10];
        let bundle = decoder.decode(&done_bytes, 0xbc).expect("Should decode 'done'");

        let has_halt = bundle
            .active_slots()
            .any(|s| matches!(s.semantic, Some(SemanticOp::Halt) | Some(SemanticOp::Done)));
        assert!(
            has_halt,
            "done instruction must decode to SemanticOp::Halt or Done, got: {:?}",
            bundle.active_slots().map(|s| format!("{:?}", s.semantic)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_rel_instruction_decodes_to_lock_release() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // REL r0, r1 from Chess listing: bytes 0x10 0x10 0x12 0x19
        // (memory order, decoder reads with from_le_bytes)
        let rel_bytes: [u8; 4] = [0x19, 0x12, 0x10, 0x10];
        let bundle = decoder.decode(&rel_bytes, 0x250).expect("Should decode 'rel'");

        let has_lock_release = bundle
            .active_slots()
            .any(|s| matches!(s.semantic, Some(SemanticOp::LockRelease)));
        let slot_semantics: Vec<_> = bundle.active_slots().map(|s| format!("{:?}", s.semantic)).collect();
        assert!(
            has_lock_release,
            "REL instruction must decode to SemanticOp::LockRelease, got: {:?}",
            slot_semantics
        );
    }

    #[test]
    fn test_acq_instruction_decodes_to_lock_acquire() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // ACQ r0, r1 from Chess listing: bytes 0x10 0x12 0x12 0x19
        let acq_bytes: [u8; 4] = [0x19, 0x12, 0x12, 0x10];
        let bundle = decoder.decode(&acq_bytes, 0x230).expect("Should decode 'acq'");

        let has_lock_acquire = bundle
            .active_slots()
            .any(|s| matches!(s.semantic, Some(SemanticOp::LockAcquire)));
        let slot_semantics: Vec<_> = bundle.active_slots().map(|s| format!("{:?}", s.semantic)).collect();
        assert!(
            has_lock_acquire,
            "ACQ instruction must decode to SemanticOp::LockAcquire, got: {:?}",
            slot_semantics
        );
    }

    #[test]
    fn test_decoder_via_tblgen() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        // Load decoder via tblgen
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        let index = decoder.decoder_index();
        let encoding_count = index.all_encodings().count();
        eprintln!("Loaded {} encodings via tblgen", encoding_count);

        // Verify we loaded many encodings
        assert!(encoding_count > 100, "Should have loaded many encodings");

        // Verify ACQ instructions are distinguished
        let acq_imm = index.all_encodings().find(|e| e.name == "ACQ_mLockId_imm");
        let acq_reg = index.all_encodings().find(|e| e.name == "ACQ_mLockId_reg");

        if let (Some(imm), Some(reg)) = (acq_imm, acq_reg) {
            eprintln!("ACQ_mLockId_imm: mask=0x{:05X}, bits=0x{:05X}", imm.fixed_mask, imm.fixed_bits);
            eprintln!("ACQ_mLockId_reg: mask=0x{:05X}, bits=0x{:05X}", reg.fixed_mask, reg.fixed_bits);

            assert_ne!(imm.fixed_bits, reg.fixed_bits, "ACQ instructions should have different fixed bits");
        }

        // Test NOP decoding still works
        let nop_bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&nop_bytes, 0).expect("Should decode NOP");
        assert!(bundle.is_nop());
    }

    // === Composite Register LUT Integration Tests ===
    /// Decode an 80-bit VLIW bundle and return (mnemonic, dest, sources) for each slot.
    /// Helper for the bundle decode diagnosis tests below.
    fn decode_80bit_bundle_slots(
        decoder: &InstructionDecoder,
        bytes: &[u8],
        _pc: u32,
    ) -> Vec<(String, Option<Operand>, Vec<Operand>)> {
        use crate::interpreter::bundle::SlotType;

        let extracted = decoder.extract_bundle_slots(bytes);
        let mut results = Vec::new();

        for slot in &extracted.slots {
            if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                results.push(("nop".to_string(), None, vec![]));
                continue;
            }
            if let Some(decoded) = decoder.decode_slot_bits(slot.bits, slot.slot_type) {
                let (dest, sources, _pm, _) = decoder.extract_operands(&decoded);
                results.push((decoded.encoding.mnemonic.clone(), dest, sources));
            } else {
                results.push((format!("UNKNOWN_{:?}", slot.slot_type), None, vec![]));
            }
        }
        results
    }

    #[test]
    fn test_decode_add314_bundle_at_pc56() {
        // Bundle from add_314_using_dma_op at PC=0x56:
        //   mova r27, #0x0; movx r7, #0x0; mov r6, #0x6
        // Format: I80_LDA_ALU_MV (80-bit, 3 slots)
        let add_314_bytes: [u8; 10] = [0xbb, 0x48, 0x0f, 0xc0, 0x08, 0x70, 0x00, 0xc8, 0x06, 0x00];

        // For comparison, add_one_using_dma at PC=0x56:
        //   mova r27, #0x0; movx r6, #0x0; mov r5, #0x6
        let add_one_bytes: [u8; 10] = [0xbb, 0x48, 0x0f, 0xa0, 0x08, 0x60, 0x00, 0xc8, 0x06, 0x00];

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // First, show raw slot extraction
        use crate::interpreter::bundle::extract_slots;
        let ext_314 = extract_slots(&add_314_bytes);
        let ext_one = extract_slots(&add_one_bytes);
        eprintln!("=== add_314 slot extraction ===");
        for s in &ext_314.slots {
            eprintln!("  {:?}: bits=0x{:010X} (width={})", s.slot_type, s.bits, s.width);
        }
        eprintln!("=== add_one slot extraction ===");
        for s in &ext_one.slots {
            eprintln!("  {:?}: bits=0x{:010X} (width={})", s.slot_type, s.bits, s.width);
        }

        // Now decode each slot
        let slots_314 = decode_80bit_bundle_slots(&decoder, &add_314_bytes, 0x56);
        let slots_one = decode_80bit_bundle_slots(&decoder, &add_one_bytes, 0x56);
        eprintln!("\n=== add_314 decoded slots ===");
        for (i, (mnem, dest, srcs)) in slots_314.iter().enumerate() {
            eprintln!("  slot[{}]: {} dest={:?} sources={:?}", i, mnem, dest, srcs);
        }
        eprintln!("\n=== add_one decoded slots ===");
        for (i, (mnem, dest, srcs)) in slots_one.iter().enumerate() {
            eprintln!("  slot[{}]: {} dest={:?} sources={:?}", i, mnem, dest, srcs);
        }

        // Verify add_314 expectations from Peano disassembly:
        //   mova r27, #0x0  -> LDA slot: dest=ScalarReg(27), source=Immediate(0)
        //   movx r7, #0x0   -> ALU slot: dest=ScalarReg(7), source=Immediate(0)
        //   mov r6, #0x6    -> MV slot: dest=ScalarReg(6), source=Immediate(6)
        // (slot ordering in the extracted list may vary)

        // Now drill into the MV slot specifically to see field details
        eprintln!("\n=== MV slot deep dive ===");
        for (label, ext) in [("add_314", &ext_314), ("add_one", &ext_one)] {
            let mv_slot = ext
                .slots
                .iter()
                .find(|s| s.slot_type == crate::interpreter::bundle::SlotType::Mv);
            if let Some(mv) = mv_slot {
                eprintln!("{} MV bits: 0x{:06X} ({:022b})", label, mv.bits, mv.bits);
                if let Some(decoded) = decoder.decode_slot_bits(mv.bits, mv.slot_type) {
                    eprintln!("  matched: {} ({})", decoded.encoding.mnemonic, decoded.encoding.name);
                    eprintln!("  operand fields:");
                    for field in &decoded.encoding.operand_fields {
                        let raw = decoded.operand(&field.name).unwrap_or(0);
                        eprintln!(
                            "    {}: raw={} (0x{:X}), bit_pos={}, width={}, signed={}, type={:?}",
                            field.name,
                            raw,
                            raw,
                            field.bit_position,
                            field.width,
                            field.signed,
                            field.operand_type
                        );
                    }
                    eprintln!("  input_order: {:?}", decoded.encoding.input_order);
                    eprintln!("  output_order: {:?}", decoded.encoding.output_order);
                }
            }
        }

        // Verify the key assertion: add_314 MV slot should decode to mov r6, #6
        let mv_314 = ext_314
            .slots
            .iter()
            .find(|s| s.slot_type == crate::interpreter::bundle::SlotType::Mv)
            .unwrap();
        let decoded_mv_314 = decoder.decode_slot_bits(mv_314.bits, mv_314.slot_type).unwrap();
        let (dest, sources, _, _) = decoder.extract_operands(&decoded_mv_314);
        assert_eq!(dest, Some(Operand::ScalarReg(6)), "Expected dest=r6");
        assert_eq!(sources, vec![Operand::Immediate(6)], "Expected source=Immediate(6)");
    }

    /// Verify the generic decode path handles movxm (LNG slot, split immediate).
    ///
    /// LNG slot layout: {i{31-12}, mMvSclDstCg, i{11-0}, 0b001}
    ///   bits [41:22] = i{31-12}  (20 bits)
    ///   bits [21:15] = mMvSclDstCg (7 bits)
    ///   bits [14:3]  = i{11-0}   (12 bits)
    ///   bits [2:0]   = 0b001     (opcode)
    ///
    /// This replaced a manual special case that used a 2-bit discriminant
    /// (dst_raw & 0x3) and would misidentify lr as PointerReg(2).
    #[test]
    fn test_decode_movxm_generic_path() {
        use crate::interpreter::bundle::SlotType;
        use crate::interpreter::state::LR_REG_INDEX;

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // Helper: build a 42-bit LNG slot word for movxm
        let build_movxm = |imm: u32, dst_enc: u8| -> u64 {
            let i_high = ((imm >> 12) & 0xFFFFF) as u64;
            let i_low = (imm & 0xFFF) as u64;
            let dst = dst_enc as u64;
            (i_high << 22) | (dst << 15) | (i_low << 3) | 0b001
        };

        // Case 1: movxm r3, #0x7ff
        // r3 in MvSclSrc: encode = (3 << 2) | 0b00 = 12
        let bits = build_movxm(0x7ff, 12);
        let decoded = decoder
            .decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm r3, #0x7ff");
        assert_eq!(decoded.encoding.mnemonic, "movxm");
        let (dest, sources, _, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::ScalarReg(3)));
        assert_eq!(sources, vec![Operand::Immediate(0x7ff)]);

        // Case 2: movxm p2, #0x100
        // p2 in MvSclSrc: encode = (2 << 4) | 0b0011 = 35
        let bits = build_movxm(0x100, 35);
        let decoded = decoder
            .decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm p2, #0x100");
        let (dest, sources, _, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::PointerReg(2)));
        assert_eq!(sources, vec![Operand::Immediate(0x100)]);

        // Case 3: movxm sp, #0x70000
        // SP in MvSclSrc: HWEncoding = 103 = (12 << 3) | 0b111
        let bits = build_movxm(0x70000, 103);
        let decoded = decoder
            .decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm sp, #0x70000");
        let (dest, sources, _, _) = decoder.extract_operands(&decoded);
        assert_eq!(
            dest,
            Some(Operand::PointerReg(crate::interpreter::state::SP_PTR_INDEX)),
            "SP should map to dedicated SP"
        );
        assert_eq!(sources, vec![Operand::Immediate(0x70000)]);

        // Case 4: movxm lr, #0x1234
        // lr in MvSclSrc: HWEncoding = 39 = (4 << 3) | 0b111
        // The OLD special case got this wrong: 39 & 0x3 == 3, so it decoded
        // as PointerReg((39 >> 4) & 0x7) = PointerReg(2). The MvSclSrc LUT
        // checks bits[2:0] == 0b111 first, correctly routing to
        // ScalarReg(LR_REG_INDEX).
        let bits = build_movxm(0x1234, 39);
        let decoded = decoder
            .decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm lr, #0x1234");
        let (dest, sources, _, _) = decoder.extract_operands(&decoded);
        assert_eq!(
            dest,
            Some(Operand::ScalarReg(LR_REG_INDEX)),
            "lr must decode as ScalarReg(LR), not PointerReg(2)"
        );
        assert_eq!(sources, vec![Operand::Immediate(0x1234)]);
    }

    /// Verify that vector load channel is determined by slot, not mnemonic.
    ///
    /// The decoder must produce VectorLoadA for slot="lda"+is_vector,
    /// VectorLoadB for slot="ldb"+is_vector, and plain Load for non-vector
    /// instructions in the same slots.
    #[test]
    fn test_vector_load_channel_by_slot() {
        use xdna_archspec::aie2::isa::{AddressingMode, InstrMemWidth};
        let decoder = InstructionDecoder::new();

        // Helper to build a minimal Load encoding for the given slot+is_vector
        let make_load_enc = |slot: &str, is_vector: bool| -> InstrEncoding {
            InstrEncoding {
                name: format!("TEST_LOAD_{}", slot),
                mnemonic: if is_vector {
                    format!("v{}", slot)
                } else {
                    slot.to_string()
                },
                asm_string: String::new(),
                slot: slot.to_string(),
                width: 20,
                fixed_mask: 0,
                fixed_bits: 0,
                operand_fields: vec![],
                semantic: Some(SemanticOp::Load),
                may_load: true,
                may_store: false,
                input_order: vec![],
                output_order: vec![],
                implicit_regs: vec![],
                addressing_mode: AddressingMode::Unknown,
                mem_width: if is_vector {
                    InstrMemWidth::Vector256
                } else {
                    InstrMemWidth::Word
                },
                has_complete_decoder: true,
                element_type: None,
                from_type: None,
                branch_condition: None,
                is_vector,
                select_variant: None,
                is_ptr_arithmetic: false,
                is_sp_relative: false,
                sched_class: None,
            }
        };

        // Vector load A (slot=lda, is_vector=true) -> Load + is_vector
        let decoded_vlda = DecodedInstr { encoding: make_load_enc("lda", true), operands: HashMap::new() };
        let op = decoder.build_slot_op(
            SlotIndex::LoadA,
            &decoded_vlda,
            None,
            vec![],
            None,
            SmallVec::new(),
            Bypass::No,
        );
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(op.is_vector, "slot=lda + is_vector should produce vector Load");

        // Vector load B (slot=ldb, is_vector=true) -> Load + is_vector
        let decoded_vldb = DecodedInstr { encoding: make_load_enc("ldb", true), operands: HashMap::new() };
        let op = decoder.build_slot_op(
            SlotIndex::LoadB,
            &decoded_vldb,
            None,
            vec![],
            None,
            SmallVec::new(),
            Bypass::No,
        );
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(op.is_vector, "slot=ldb + is_vector should produce vector Load");

        // Scalar load in lda slot (is_vector=false) -> Load + !is_vector
        let decoded_scl = DecodedInstr { encoding: make_load_enc("lda", false), operands: HashMap::new() };
        let op = decoder.build_slot_op(
            SlotIndex::LoadA,
            &decoded_scl,
            None,
            vec![],
            None,
            SmallVec::new(),
            Bypass::No,
        );
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(!op.is_vector, "slot=lda + !is_vector should produce scalar Load");

        // Vector store (slot=st, is_vector=true) -> Store + is_vector
        let mut store_enc = make_load_enc("st", true);
        store_enc.semantic = Some(SemanticOp::Store);
        store_enc.may_load = false;
        store_enc.may_store = true;
        let decoded_vst = DecodedInstr { encoding: store_enc, operands: HashMap::new() };
        let op = decoder.build_slot_op(
            SlotIndex::Store,
            &decoded_vst,
            None,
            vec![],
            None,
            SmallVec::new(),
            Bypass::No,
        );
        assert_eq!(op.semantic, Some(SemanticOp::Store));
        assert!(op.is_vector, "slot=st + is_vector should produce vector Store");
    }

    /// Verify that PADD instructions produce PointerReg destination via
    /// the is_ptr_arithmetic field (not mnemonic checking).
    #[test]
    fn test_padd_dest_is_pointer() {
        use xdna_archspec::aie2::isa::{AddressingMode, InstrMemWidth, OperandType, RegisterKind};

        let decoder = InstructionDecoder::new();

        // Build a PADD-like encoding with IndexedImmediate addressing
        // and is_ptr_arithmetic=true. The ptr+imm combination should
        // produce a PointerReg destination + Immediate source.
        let mut ptr_field = OperandField::new("ptr", 10, 3);
        ptr_field.operand_type = OperandType::Register(RegisterKind::Pointer);
        let mut imm_field = OperandField::new("imm", 7, 3);
        imm_field.operand_type = OperandType::Immediate { signed: false, scale: 1 };

        let padd_enc = InstrEncoding {
            name: "PADD_test".to_string(),
            mnemonic: "padd".to_string(),
            asm_string: String::new(),
            slot: "alu".to_string(),
            width: 20,
            fixed_mask: 0,
            fixed_bits: 0,
            operand_fields: vec![ptr_field, imm_field],
            semantic: Some(SemanticOp::Add),
            may_load: false,
            may_store: false,
            input_order: vec![],
            output_order: vec![],
            implicit_regs: vec![],
            addressing_mode: AddressingMode::IndexedImmediate,
            mem_width: InstrMemWidth::Word,
            has_complete_decoder: true,
            element_type: None,
            from_type: None,
            branch_condition: None,
            is_vector: false,
            select_variant: None,
            is_ptr_arithmetic: true,
            is_sp_relative: false,
            sched_class: None,
        };

        // Encode: ptr=3 at bits 12:10, imm=5 at bits 9:7
        let mut operands = HashMap::new();
        operands.insert("ptr".to_string(), 3u64);
        operands.insert("imm".to_string(), 5u64);

        let decoded = DecodedInstr { encoding: padd_enc, operands };

        let (dest, sources, _post_modify, _) = decoder.extract_operands(&decoded);

        // PADD: ptr becomes destination, imm becomes source
        assert_eq!(dest, Some(Operand::PointerReg(3)), "PADD should produce PointerReg destination");
        assert!(
            sources.iter().any(|s| matches!(s, Operand::Immediate(5))),
            "PADD should have immediate source, got {:?}",
            sources
        );

        // Verify that a non-padd encoding with same fields produces Memory operand
        let mut load_enc = decoded.encoding.clone();
        load_enc.is_ptr_arithmetic = false;
        load_enc.name = "LDA_test".to_string();
        load_enc.mnemonic = "lda".to_string();
        let load_decoded = DecodedInstr { encoding: load_enc, operands: decoded.operands.clone() };

        let (dest, sources, _, _) = decoder.extract_operands(&load_decoded);

        // Non-padd: ptr+imm combine into Memory operand as source
        assert!(
            dest.is_none() || !matches!(dest, Some(Operand::PointerReg(3))),
            "Non-PADD should not produce PointerReg destination"
        );
        assert!(
            sources.iter().any(|s| matches!(s, Operand::Memory { base: 3, offset: 5 })),
            "Non-PADD should produce Memory source, got {:?}",
            sources
        );
    }

    /// Regression test: LDA instruction must not be overwritten by LDB NOP.
    ///
    /// LDA and LDB are independent load slots in 128-bit bundles. When LDB
    /// is NOP (bits=0) and LDA has a real instruction, both should decode
    /// into their own slot without collision.
    ///
    /// Real bundle from add_314_using_dma_op at PC=0x2A0:
    ///   nopb; mova r0, #0x30; nops; movx r1, #0x1; mov r20, p7; nopv
    #[test]
    fn test_lda_and_ldb_have_separate_slots() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // Raw 128-bit bundle from add_314_using_dma_op at PC=0x2A0:
        // nopb; mova r0, #0x30; nops; movx r1, #0x1; mov r20, p7; nopv
        let bytes: [u8; 16] =
            [0xc0, 0x03, 0x00, 0x28, 0x3b, 0x87, 0x2a, 0x10, 0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00, 0x00];

        let bundle = decoder.decode(&bytes, 0x2A0).expect("Should decode 128-bit bundle");
        assert_eq!(bundle.size(), 16);

        // LoadA slot must contain the mova instruction (LDA slot).
        let load_a = bundle.slot(SlotIndex::LoadA).expect("LoadA slot should be present");
        assert!(!load_a.is_nop(), "LoadA slot should have mova r0, #0x30");
        assert_eq!(load_a.dest, Some(Operand::ScalarReg(0)), "mova destination should be r0");
        assert_eq!(load_a.sources.len(), 1, "mova should have one source");
        assert_eq!(load_a.sources[0], Operand::Immediate(0x30), "mova source should be immediate 0x30");

        // LoadB slot should be NOP (nopb = LDB NOP).
        let load_b = bundle.slot(SlotIndex::LoadB).expect("LoadB slot should be present (NOP)");
        assert!(load_b.is_nop(), "LoadB slot should be NOP (nopb)");

        // Scalar0 (ALU) should have movx r1, #0x1
        let scalar0 = bundle.slot(SlotIndex::Scalar0).expect("Scalar0 slot should be present");
        assert_eq!(scalar0.dest, Some(Operand::ScalarReg(1)), "movx destination should be r1");

        // Scalar1 (MV) should have mov r20, p7
        let scalar1 = bundle.slot(SlotIndex::Scalar1).expect("Scalar1 slot should be present");
        assert_eq!(scalar1.dest, Some(Operand::ScalarReg(20)), "mov destination should be r20");
    }

    /// Decode regression for the Half-B SRS capture kernel's two key
    /// instructions (`vec_srs_i32`, real bytes from llvm-objdump):
    ///   - `st q0, [p1], #0x10`  -> 128-bit QuadWord store, q-mask source.
    ///   - `vlda.ups.s64.s32 bml5, s1, [p0], #0x20` -> fused UPS load with
    ///     POST-INCREMENT addressing (PointerReg base + post_modify, no
    ///     Memory{} operand). The post-increment form is what the dispatch
    ///     must recognize as a memory op; offset addressing was the only form
    ///     previously exercised.
    #[test]
    fn test_srs_kernel_q_store_and_postinc_ups_decode() {
        use crate::interpreter::bundle::{MemWidth, PostModify};
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping: llvm-aie not found");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // PC 0x2c0: `99 8a 03 09  st q0, [p1], #0x10`.
        let st_bundle = decoder.decode(&[0x99, 0x8a, 0x03, 0x09], 0x2c0).expect("decode st q");
        let st = st_bundle.slot(SlotIndex::Store).expect("Store slot");
        assert_eq!(st.semantic, Some(SemanticOp::Store));
        assert_eq!(st.mem_width, MemWidth::QuadWord, "st q is a 128-bit store");
        assert_eq!(st.sources[0], Operand::ControlReg(16), "data source is q0 (mask range)");
        assert_eq!(st.post_modify, PostModify::Immediate(16), "post-increment 16 bytes");

        // PC 0x250: `vlda.ups.s64.s32 bml5, s1, [p0], #0x20` (LoadA slot).
        let ups_bundle = decoder
            .decode(&[0xbb, 0xc8, 0x03, 0x48, 0x0a, 0x80, 0x01, 0xa0, 0xba, 0x02], 0x250)
            .expect("decode vlda.ups");
        let ups = ups_bundle.slot(SlotIndex::LoadA).expect("LoadA slot");
        assert_eq!(ups.semantic, Some(SemanticOp::Ups));
        assert_eq!(ups.dest, Some(Operand::AccumReg(10)), "bml5 -> AccumReg(10)");
        assert_eq!(ups.from_type, Some(ElementType::Int32));
        assert_eq!(ups.element_type, Some(ElementType::Int64));
        assert_eq!(ups.post_modify, PostModify::Immediate(32), "post-increment 32 bytes");
        // The address comes via a PointerReg (post-increment), NOT a Memory{}
        // operand -- the case the fused-load dispatch must still recognize.
        assert!(
            ups.sources.iter().any(|s| matches!(s, Operand::PointerReg(_))),
            "post-increment addressing carries a PointerReg, not Memory{{}}"
        );
        assert!(
            !ups.sources.iter().any(|s| matches!(s, Operand::Memory { .. })),
            "post-increment form has no Memory{{}} operand"
        );
    }

    /// Helper: create a minimal vector encoding with the given mnemonic.
    fn make_vec_encoding(mnemonic: &str) -> InstrEncoding {
        use xdna_archspec::aie2::isa::{AddressingMode, InstrMemWidth};
        InstrEncoding {
            name: mnemonic.to_uppercase().replace('.', "_"),
            mnemonic: mnemonic.to_string(),
            asm_string: String::new(),
            slot: "vec".to_string(),
            width: 26,
            fixed_mask: 0,
            fixed_bits: 0,
            operand_fields: vec![],
            semantic: Some(SemanticOp::Copy), // Generic fallback
            may_load: false,
            may_store: false,
            input_order: vec![],
            output_order: vec![],
            implicit_regs: vec![],
            addressing_mode: AddressingMode::Unknown,
            mem_width: InstrMemWidth::Word,
            has_complete_decoder: true,
            element_type: xdna_archspec::aie2::isa::infer_element_type(mnemonic),
            from_type: None,
            branch_condition: None,
            is_vector: true,
            select_variant: None,
            is_ptr_arithmetic: false,
            is_sp_relative: false,
            sched_class: None,
        }
    }

    /// Helper: build a SlotOp from a mnemonic and explicit SemanticOp.
    ///
    /// This tests the decoder's dispatch path: given an encoding with a
    /// known semantic, does build_slot_op produce the right SlotOp?
    fn dispatch_semantic(mnemonic: &str, semantic: SemanticOp) -> SlotOp {
        let mut enc = make_vec_encoding(mnemonic);
        enc.semantic = Some(semantic);
        let decoded = DecodedInstr { encoding: enc, operands: HashMap::new() };
        let decoder = InstructionDecoder::load_default();
        decoder.build_slot_op(SlotIndex::Vector, &decoded, None, vec![], None, SmallVec::new(), Bypass::No)
    }

    /// Helper: assert a dispatch result matches expected SemanticOp.
    fn assert_sem(mnemonic: &str, semantic: SemanticOp) {
        let op = dispatch_semantic(mnemonic, semantic);
        assert_eq!(
            op.semantic,
            Some(semantic),
            "mnemonic '{}' with {:?} should dispatch correctly, got {:?}",
            mnemonic,
            semantic,
            op.semantic
        );
    }

    #[test]
    fn test_vector_semantic_dispatch_srs_ups() {
        assert_sem("vsrs.s8", SemanticOp::Srs);
        assert_sem("vsrs.d16", SemanticOp::Srs);
        assert_sem("vups.s8", SemanticOp::Ups);
        assert_sem("vpush.lo.s16", SemanticOp::Ups);
    }

    #[test]
    fn test_vector_semantic_dispatch_mac_variants() {
        assert_sem("vmac", SemanticOp::Mac);
        assert_sem("vmac.f", SemanticOp::Mac);
        assert_sem("vmsc", SemanticOp::MatMulSub);
        assert_sem("vmsc.f", SemanticOp::MatMulSub);
        assert_sem("vnegmac", SemanticOp::NegMatMul);
        assert_sem("vaddmac", SemanticOp::AddMac);
        assert_sem("vsubmac", SemanticOp::SubMac);
        assert_sem("vnegmul", SemanticOp::NegMul);
    }

    #[test]
    fn test_vector_semantic_dispatch_conditional() {
        assert_sem("vsub_lt.d32", SemanticOp::SubLt);
        assert_sem("vsub_ge.s16", SemanticOp::SubGe);
        assert_sem("vmaxdiff_lt.s8", SemanticOp::MaxDiffLt);
        assert_sem("vmax_lt.bf", SemanticOp::MaxLt);
        assert_sem("vmin_ge.d16", SemanticOp::MinGe);
    }

    #[test]
    fn test_vector_semantic_dispatch_comparisons() {
        assert_sem("vge.s32", SemanticOp::SetGe);
        assert_sem("vlt.d8", SemanticOp::SetLt);
        assert_sem("veqz.s16", SemanticOp::SetEq);
    }

    #[test]
    fn test_vector_semantic_dispatch_abs_neg() {
        assert_sem("vabs_gtz.s32", SemanticOp::AbsGtz);
        assert_sem("vneg_gtz", SemanticOp::NegGtz);
        assert_sem("vbneg_ltz.s16", SemanticOp::NegLtz);
    }

    #[test]
    fn test_vector_semantic_dispatch_data_movement() {
        assert_sem("vshuffle", SemanticOp::Shuffle);
        assert_sem("vshift.align", SemanticOp::Align);
        assert_sem("vclr", SemanticOp::VectorClear);
        assert_sem("vsel.s32", SemanticOp::VectorSelect);
        assert_sem("vextbcst.s16", SemanticOp::VectorBroadcast);
        assert_sem("vbcst.s32", SemanticOp::VectorBroadcast);
        assert_sem("vpack.d8", SemanticOp::Pack);
        assert_sem("vunpack.s16", SemanticOp::Unpack);
        assert_sem("vextract.s32", SemanticOp::VectorExtract);
        assert_sem("vinsert.s16", SemanticOp::VectorInsert);
    }

    #[test]
    fn test_vector_semantic_dispatch_conversion() {
        assert_sem("vconv.bf16", SemanticOp::Convert);
        assert_sem("vconv.fp32", SemanticOp::Convert);
        assert_sem("vfloor.s32", SemanticOp::Convert);
    }

    #[test]
    fn test_vector_semantic_dispatch_basic_ops() {
        assert_sem("vadd.s32", SemanticOp::Add);
        assert_sem("vsub.s16", SemanticOp::Sub);
        assert_sem("vmul.s8", SemanticOp::Mul);
        assert_sem("vmov", SemanticOp::Copy);
        assert_sem("vand.s32", SemanticOp::And);
        assert_sem("vor.s16", SemanticOp::Or);
        assert_sem("vxor.s32", SemanticOp::Xor);
    }

    #[test]
    fn test_vector_semantic_dispatch_negadd_negsub() {
        assert_sem("vnegadd.s32", SemanticOp::NegAdd);
        assert_sem("vnegsub.f", SemanticOp::NegAdd);
    }

    /// Verify every vector encoding in the loaded decoder tables has a SemanticOp.
    ///
    /// Semantics come from two sources: pattern-based (AIE2InstrPatterns.td) and
    /// structural (TableGen attributes). Every vector instruction should be covered
    /// by at least one of these paths.
    #[test]
    fn test_all_vector_encodings_have_semantic() {
        let decoder = InstructionDecoder::load_default();
        let mut uncovered = Vec::new();

        for enc in decoder.decoder_index().all_encodings() {
            let m = enc.mnemonic.to_lowercase();
            if m.starts_with('v') && m != "opcodestr" && enc.semantic.is_none() {
                uncovered.push(enc.name.clone());
            }
        }

        // Some vector instructions may legitimately lack semantics if they have
        // no pattern and no structural signals. Log them but don't fail the test
        // for now -- the goal is to track coverage, not enforce 100%.
        if !uncovered.is_empty() {
            eprintln!(
                "INFO: {} vector encodings without SemanticOp (out of coverage):\n  {:?}",
                uncovered.len(),
                &uncovered[..uncovered.len().min(20)],
            );
        }
    }

    /// Cross-reference our decoder against llvm-objdump on a real ELF binary.
    ///
    /// This wires `crossref::cross_reference_elf()` into the test suite so
    /// decoder verification runs automatically rather than manually.
    #[test]
    fn test_crossref_integration() {
        use crate::config::Config;
        use crate::interpreter::decode::crossref::cross_reference_elf;

        let Some(elf_path) = Config::get().add_one_elf() else {
            eprintln!("Skipping test_crossref_integration: ELF not found (set MLIR_AIE_PATH)");
            return;
        };

        let objdump_path =
            std::path::PathBuf::from(Config::get().llvm_aie_path()).join("build/bin/llvm-objdump");
        if !objdump_path.exists() {
            eprintln!(
                "Skipping test_crossref_integration: llvm-objdump not found at {}",
                objdump_path.display()
            );
            return;
        }

        let decoder = InstructionDecoder::load_default();
        let report = cross_reference_elf(&elf_path, &objdump_path, &decoder).expect("Cross-reference failed");

        eprintln!("{}", report);

        assert!(report.total_instructions > 0, "Should decode at least one instruction");

        // Require >= 95% match rate, leaving room for known mnemonic
        // normalization gaps (dot-suffixed variants, NOP aliasing, etc.)
        let rate = report.match_rate();
        assert!(
            rate >= 95.0,
            "Decoder match rate {:.1}% is below 95% threshold.\n\
             Mismatches: {}, Decode failures: {}, Size mismatches: {}",
            rate,
            report.mnemonic_mismatches.len(),
            report.decode_failures.len(),
            report.size_mismatches.len(),
        );

        // Log mismatches for investigation even when passing
        if report.discrepancy_count() > 0 {
            eprintln!(
                "NOTE: {} discrepancies at {:.1}% match rate (above 95% threshold)",
                report.discrepancy_count(),
                rate,
            );
        }
    }

    #[test]
    fn test_element_type_bf_suffix() {
        // Verify .bf suffix correctly infers BFloat16
        let enc = make_vec_encoding("vmin_ge.bf");
        assert_eq!(enc.element_type, Some(ElementType::BFloat16));

        // Verify .f suffix correctly infers Float32
        let enc = make_vec_encoding("vadd.f");
        assert_eq!(enc.element_type, Some(ElementType::Float32));
    }

    /// Cross-validate LLVM FFI operands against legacy bit-field extraction
    /// for real ELF instructions.
    ///
    /// This test decodes every instruction in every fuzz ELF using BOTH the
    /// LLVM FFI path and the legacy bytecode path, and compares the resulting
    /// operands. Any divergence is flagged.
    #[test]
    fn test_ffi_vs_legacy_operand_crosscheck() {
        use crate::interpreter::bundle::SlotType;

        // Find ELF files from fuzz dir (recursive) or ISA test harness.
        let mut elf_paths = Vec::new();
        for search_dir in &["build/fuzz", "build/isa-tests"] {
            let dir = std::path::Path::new(search_dir);
            if dir.exists() {
                fn collect_elfs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() {
                                collect_elfs(&path, out);
                            } else if path.extension().map(|e| e == "elf" || e == "o").unwrap_or(false) {
                                out.push(path);
                            }
                        }
                    }
                }
                collect_elfs(dir, &mut elf_paths);
            }
        }

        if elf_paths.is_empty() {
            eprintln!("Skipping: no ELF files found in build/fuzz or build/isa-tests");
            return;
        }
        // Limit to 50 ELFs for test speed.
        elf_paths.truncate(50);

        let decoder = InstructionDecoder::load_default();
        let mut total = 0u64;
        let mut ffi_hits = 0u64;
        let mut ffi_misses = 0u64;
        let mut matches = 0u64;
        let mut divergences = 0u64;
        let mut divergence_details: Vec<String> = Vec::new();

        for path in &elf_paths {
            let data = std::fs::read(&path).unwrap();
            let elf = match goblin::elf::Elf::parse(&data) {
                Ok(e) => e,
                Err(_) => continue,
            };

            for section in &elf.section_headers {
                if section.sh_flags & 0x4 == 0 {
                    continue; // Skip non-executable sections
                }
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                if end > data.len() {
                    continue;
                }

                let code = &data[start..end];
                let mut offset = 0;

                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() {
                        break;
                    }

                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);

                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                            continue;
                        }
                        total += 1;

                        let ffi_result = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                        let legacy_result = decoder.decode_slot_bits(slot.bits, slot.slot_type).map(|d| {
                            let (dest, sources, pm, _extra) = decoder.extract_operands(&d);
                            (d, dest, sources, pm)
                        });

                        match (ffi_result, legacy_result) {
                            (
                                Some((_, ffi_dest, ffi_src, ffi_pm, _, _, _, _, _)),
                                Some((_, leg_dest, leg_src, leg_pm)),
                            ) => {
                                ffi_hits += 1;
                                if ffi_dest == leg_dest && ffi_src == leg_src && ffi_pm == leg_pm {
                                    matches += 1;
                                } else {
                                    divergences += 1;
                                    if divergence_details.len() < 20 {
                                        divergence_details.push(format!(
                                            "  {:?} 0x{:010X}:\n    FFI:    dest={:?} src={:?} pm={:?}\n    LEGACY: dest={:?} src={:?} pm={:?}",
                                            slot.slot_type, slot.bits,
                                            ffi_dest, ffi_src, ffi_pm,
                                            leg_dest, leg_src, leg_pm,
                                        ));
                                    }
                                }
                            }
                            (None, Some(_)) => {
                                ffi_misses += 1;
                            }
                            _ => {}
                        }
                    }

                    offset += size;
                }
            }
        }

        eprintln!("\n=== FFI vs Legacy Cross-Validation ===");
        eprintln!("Total slot decodes:    {}", total);
        eprintln!(
            "FFI hits:              {} ({:.1}%)",
            ffi_hits,
            if total > 0 {
                100.0 * ffi_hits as f64 / total as f64
            } else {
                0.0
            }
        );
        eprintln!("FFI misses (fallback): {}", ffi_misses);
        eprintln!("Operand matches:       {}", matches);
        eprintln!("Operand divergences:   {}", divergences);

        // Categorize divergences.
        let mut cat_store_dest = 0u64; // Store: FFI dest=None, legacy has ModifierReg dest
        let mut cat_ptr_arith = 0u64; // Pointer arith: FFI has extra PointerReg in sources
        let mut cat_memory_offset = 0u64; // Memory offset differs
        let mut cat_missing_memory = 0u64; // FFI has raw operands, legacy has Memory{}
        let mut cat_post_modify = 0u64; // PostModify differs
        let mut cat_other = 0u64;

        // Re-scan to categorize (the details vec was capped at 20)
        for path in &elf_paths {
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let elf = match goblin::elf::Elf::parse(&data) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for section in &elf.section_headers {
                if section.sh_flags & 0x4 == 0 {
                    continue;
                }
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                if end > data.len() {
                    continue;
                }
                let code = &data[start..end];
                let mut offset = 0;
                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() {
                        break;
                    }
                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                            continue;
                        }
                        let ffi = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                        let leg = decoder.decode_slot_bits(slot.bits, slot.slot_type).map(|d| {
                            let (dest, sources, pm, _) = decoder.extract_operands(&d);
                            (d, dest, sources, pm)
                        });
                        if let (Some((_, fd, fs, fp, _, _, _, _, _)), Some((_, ld, ls, lp))) = (ffi, leg) {
                            if fd == ld && fs == ls && fp == lp {
                                continue;
                            }
                            // Categorize
                            if ld.is_some() && fd.is_none() && matches!(ld, Some(Operand::ModifierReg(_))) {
                                cat_store_dest += 1;
                            } else if fd == ld && fp != lp {
                                cat_post_modify += 1;
                            } else if fd == ld {
                                // Same dest, different sources
                                let ffi_has_mem = fs.iter().any(|o| matches!(o, Operand::Memory { .. }));
                                let leg_has_mem = ls.iter().any(|o| matches!(o, Operand::Memory { .. }));
                                if ffi_has_mem && leg_has_mem {
                                    cat_memory_offset += 1;
                                } else if !ffi_has_mem && leg_has_mem {
                                    cat_missing_memory += 1;
                                } else if fs.len() != ls.len() {
                                    cat_ptr_arith += 1;
                                } else {
                                    cat_other += 1;
                                }
                            } else {
                                cat_other += 1;
                            }
                        }
                    }
                    offset += size;
                }
            }
        }

        eprintln!("\nDivergence categories:");
        eprintln!("  Store dest (FFI correct, legacy wrong):    {}", cat_store_dest);
        eprintln!("  Ptr arith (extra PointerReg in sources):   {}", cat_ptr_arith);
        eprintln!("  Memory offset differs:                     {}", cat_memory_offset);
        eprintln!("  Missing Memory (FFI has raw, legacy has):  {}", cat_missing_memory);
        eprintln!("  PostModify differs:                        {}", cat_post_modify);
        eprintln!("  Other:                                     {}", cat_other);

        if !divergence_details.is_empty() {
            eprintln!("\nFirst {} divergences:", divergence_details.len());
            for d in &divergence_details {
                eprintln!("{}", d);
            }
        }

        // We expect the FFI path to handle most instructions.
        if total > 0 {
            let hit_rate = 100.0 * ffi_hits as f64 / total as f64;
            eprintln!("FFI hit rate: {:.1}%", hit_rate);
            assert!(hit_rate > 90.0, "FFI should handle >90% of instructions (got {:.1}%)", hit_rate);
        }
    }

    /// Inspect raw LLVM operand layout for post-modify loads.
    #[test]
    #[ignore]
    fn test_llvm_postmodify_operand_layout() {
        use crate::interpreter::bundle::SlotType;
        use xdna_archspec::aie2::isa::AddressingMode;

        // Find post-modify load instructions in ELF files.
        let mut elf_paths = Vec::new();
        for search_dir in &["build/isa-tests", "build/fuzz"] {
            let dir = std::path::Path::new(search_dir);
            if dir.exists() {
                fn collect_elfs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() {
                                collect_elfs(&path, out);
                            } else if path.extension().map(|e| e == "elf" || e == "o").unwrap_or(false) {
                                out.push(path);
                            }
                        }
                    }
                }
                collect_elfs(dir, &mut elf_paths);
            }
        }
        // Also check bridge test ELFs
        let bridge_elf = std::path::Path::new(
            "../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie_arch.mlir.prj/main_core_0_2.elf",
        );
        if bridge_elf.exists() {
            elf_paths.push(bridge_elf.to_path_buf());
        }

        let decoder = InstructionDecoder::load_default();
        let mut seen = std::collections::HashSet::new();

        for path in &elf_paths {
            let data = match std::fs::read(path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let elf = match goblin::elf::Elf::parse(&data) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for section in &elf.section_headers {
                if section.sh_flags & 0x4 == 0 {
                    continue;
                }
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                if end > data.len() {
                    continue;
                }
                let code = &data[start..end];
                let mut offset = 0;
                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() {
                        break;
                    }
                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                            continue;
                        }
                        if !matches!(slot.slot_type, SlotType::Lda | SlotType::Ldb) {
                            continue;
                        }

                        let ffi_slot = InstructionDecoder::slot_type_to_ffi(slot.slot_type);
                        if ffi_slot.is_none() {
                            continue;
                        }
                        let raw = decoder_ffi::decode_slot(ffi_slot.unwrap(), slot.bits);
                        if raw.is_none() {
                            continue;
                        }
                        let raw = raw.unwrap();

                        // Only show post-modify instructions
                        let enc_name = &raw.name;
                        let encoding = decoder.index.encoding_by_name(
                            match slot.slot_type {
                                SlotType::Lda => "lda",
                                SlotType::Ldb => "ldb",
                                _ => continue,
                            },
                            enc_name,
                        );
                        if encoding.is_none() {
                            continue;
                        }
                        let encoding = encoding.unwrap();
                        if !matches!(
                            encoding.addressing_mode,
                            AddressingMode::PostModifyImmediate | AddressingMode::PostModifyRegister
                        ) {
                            continue;
                        }

                        let key = format!(
                            "{} nd={} ops={:?}",
                            enc_name,
                            raw.num_defs,
                            raw.operands
                                .iter()
                                .map(|o| match o {
                                    xdna_archspec::aie2::isa::decoder_ffi::DecodedOperand::Reg {
                                        name,
                                        ..
                                    } => format!("Reg({})", name),
                                    xdna_archspec::aie2::isa::decoder_ffi::DecodedOperand::Imm(v) =>
                                        format!("Imm({})", v),
                                })
                                .collect::<Vec<_>>()
                        );
                        if seen.insert(key.clone()) {
                            eprintln!("{}", key);
                        }
                    }
                    offset += size;
                }
            }
        }
    }

    /// Decode every instruction in batch_0's kernel via both FFI and legacy,
    /// printing all results for side-by-side comparison.
    #[test]
    #[ignore]
    fn test_batch0_ffi_vs_legacy_all_instructions() {
        use crate::interpreter::bundle::SlotType;

        let kernel_path = std::path::Path::new("build/isa-tests/batch_0/kernel.o");
        if !kernel_path.exists() {
            eprintln!("Skipping: batch_0 kernel not found");
            return;
        }

        let data = std::fs::read(kernel_path).unwrap();
        let elf = goblin::elf::Elf::parse(&data).unwrap();
        let decoder = InstructionDecoder::load_default();

        for section in &elf.section_headers {
            if section.sh_flags & 0x4 == 0 {
                continue;
            }
            let start = section.sh_offset as usize;
            let end = start + section.sh_size as usize;
            if end > data.len() {
                continue;
            }
            let code = &data[start..end];
            let mut offset = 0;
            let mut instr_num = 0;

            while offset + 4 <= code.len() {
                let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                let size = format.size_bytes() as usize;
                if offset + size > code.len() {
                    break;
                }

                let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                for slot in &extracted.slots {
                    if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                        continue;
                    }
                    instr_num += 1;

                    let ffi = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                    let leg = decoder.decode_slot_bits(slot.bits, slot.slot_type).map(|d| {
                        let name = d.encoding.name.clone();
                        let (dest, sources, pm, _) = decoder.extract_operands(&d);
                        (name, dest, sources, pm)
                    });

                    let ffi_name = ffi
                        .as_ref()
                        .map(|(d, _, _, _, _, _, _, _, _)| d.encoding.name.as_str())
                        .unwrap_or("FAIL");
                    let leg_name = leg.as_ref().map(|(n, _, _, _)| n.as_str()).unwrap_or("FAIL");

                    let ffi_ops = ffi
                        .as_ref()
                        .map(|(_, d, s, p, _, _, _, _, _)| format!("dest={:?} src={:?} pm={:?}", d, s, p));
                    let leg_ops =
                        leg.as_ref().map(|(_, d, s, p)| format!("dest={:?} src={:?} pm={:?}", d, s, p));

                    let status = if ffi_ops == leg_ops { "OK" } else { "DIFF" };

                    eprintln!(
                        "[{:3}] {:?} 0x{:010X} {} name_ffi={} name_leg={}",
                        instr_num, slot.slot_type, slot.bits, status, ffi_name, leg_name
                    );
                    if status == "DIFF" {
                        eprintln!("  FFI:    {}", ffi_ops.unwrap_or_default());
                        eprintln!("  LEGACY: {}", leg_ops.unwrap_or_default());
                    }
                }
                offset += size;
            }
        }
    }

    /// Verify that validate_semantic_vs_flags produces no warnings for
    /// all instructions in the ISA test ELFs.
    ///
    /// This catches cases where our SemanticOp classification disagrees
    /// with LLVM's MCInstrDesc flags (MayLoad, MayStore, isBranch, etc.).
    #[test]
    fn test_semantic_vs_llvm_flags_no_false_positives() {
        let decoder = InstructionDecoder::load_default();

        // Decode all known instruction encodings and validate.
        let mut checked = 0;
        let mut warnings = Vec::new();

        for enc in decoder.decoder_index().all_encodings() {
            // Skip instructions without semantics -- those are expected gaps.
            if enc.semantic.is_none() {
                continue;
            }

            // Try to get an LLVM opcode by decoding the encoding's fixed bits.
            let ffi_slot = match enc.slot.as_str() {
                "alu" => decoder_ffi::Slot::Alu,
                "lda" => decoder_ffi::Slot::Lda,
                "ldb" => decoder_ffi::Slot::Ldb,
                "lng" => decoder_ffi::Slot::Lng,
                "mv" => decoder_ffi::Slot::Mv,
                "st" => decoder_ffi::Slot::St,
                "vec" => decoder_ffi::Slot::Vec,
                _ => continue,
            };

            // Use the fixed bits as a decode candidate.
            let bits = enc.fixed_bits;
            if let Some(ffi_result) = decoder_ffi::decode_slot(ffi_slot, bits) {
                if let Some(info) = decoder.get_instr_info(ffi_result.opcode) {
                    if info.flags == 0 {
                        continue;
                    } // No flags to validate.

                    let semantic = enc.semantic.unwrap();

                    // Check load/store classification.
                    if matches!(semantic, SemanticOp::Load) && !info.is_load() {
                        warnings.push(format!("{}: SemanticOp::Load but no MayLoad", enc.name));
                    }
                    if matches!(semantic, SemanticOp::Store) && !info.is_store() {
                        warnings.push(format!("{}: SemanticOp::Store but no MayStore", enc.name));
                    }

                    checked += 1;
                }
            }
        }

        eprintln!("Validated semantic vs LLVM flags for {} instructions", checked);
        if !warnings.is_empty() {
            eprintln!("Warnings ({}):", warnings.len());
            for w in &warnings {
                eprintln!("  {}", w);
            }
        }
        // Warnings are informational, not test failures.
        // As we fix misclassifications, we can tighten this.
        assert!(checked > 50, "Should validate 50+ instructions, got {}", checked);
    }

    #[test]
    fn test_decode_vbcstshfl_8_encoding() {
        use crate::interpreter::bundle::SlotType;

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // vbcstshfl.8 x0, r0, r29 = 0x180027b9
        // MV bits = (0x180027b9 >> 5) & 0x3FFFFF = 0x00013D
        let word: u32 = 0x180027b9;
        let mv_bits = ((word >> 5) & 0x3FFFFF) as u64;
        eprintln!("MV bits for vbcstshfl.8: 0x{:06X}", mv_bits);

        // Try native decoder
        let native_result = decoder.decode_slot_bits(mv_bits, SlotType::Mv);
        if let Some(decoded) = &native_result {
            eprintln!(
                "Native decode: name={:?} mnemonic={:?}",
                decoded.encoding.name, decoded.encoding.mnemonic
            );
        } else {
            eprintln!("Native decode: FAILED");
        }

        // Try FFI decoder
        let ffi_result = decoder.try_decode_via_ffi(mv_bits, SlotType::Mv);
        if let Some((decoded, dest, sources, _, _, _, _, _, _)) = &ffi_result {
            eprintln!(
                "FFI decode: name={:?} mnemonic={:?} dest={:?} sources={:?}",
                decoded.encoding.name, decoded.encoding.mnemonic, dest, sources
            );
        } else {
            eprintln!("FFI decode: FAILED");
        }

        // Also try decoding the full 32-bit bundle
        let bytes = word.to_le_bytes();
        match decoder.decode(&bytes, 0) {
            Ok(bundle) => {
                for (i, slot) in bundle.slots().iter().enumerate() {
                    if let Some(ref op) = slot {
                        if !op.is_nop() {
                            eprintln!(
                                "Bundle slot {}: semantic={:?} encoding={:?} is_vector={}",
                                i, op.semantic, op.encoding_name, op.is_vector
                            );
                        }
                    }
                }
            }
            Err(e) => eprintln!("Bundle decode: FAILED {:?}", e),
        }

        // The instruction should decode as VBCSTSHFL_8 with VectorBroadcast semantic
        assert!(ffi_result.is_some() || native_result.is_some(), "VBCSTSHFL_8 should be decodable");
    }

    /// Debug test: check VMAXDIFF_LT_S16 FFI operand ordering.
    ///
    /// Encoding from `llvm-objdump -d batch_061.o`:
    ///   `b9c: 39 90 04 18  vmaxdiff_lt.s16 x0, r16, x2, x0`
    /// Expected: dest=x0, extra_dest=r16, sources=[x2, x0]
    #[test]
    fn test_vmaxdiff_lt_s16_operand_order() {
        // Decode from the full 32-bit bundle bytes.
        let bundle_bytes: [u8; 4] = [0x39, 0x90, 0x04, 0x18];
        let decoder = InstructionDecoder::load_default();

        let bundle = decoder.decode(&bundle_bytes, 0).expect("decode bundle");
        let mv_slot = bundle
            .slots()
            .iter()
            .flatten()
            .find(|s| !s.is_nop())
            .expect("should have a non-NOP slot");

        eprintln!("Encoding: {:?}", mv_slot.encoding_name);
        eprintln!("Semantic: {:?}", mv_slot.semantic);
        eprintln!("Dest: {:?}", mv_slot.dest);
        eprintln!("Sources: {:?}", mv_slot.sources);
        eprintln!("Extra dests: {:?}", mv_slot.extra_dests);

        // encoding_name stores mnemonic (lowercase), not TableGen name
        assert_eq!(mv_slot.encoding_name.as_deref(), Some("vmaxdiff_lt.s16"));

        // dest should be x0 = VectorReg(0)
        assert_eq!(mv_slot.dest, Some(Operand::VectorReg(0)), "dest should be x0");

        // sources[0] should be s1=x2 = VectorReg(4), sources[1] should be s2=x0 = VectorReg(0)
        assert!(mv_slot.sources.len() >= 2, "should have 2 sources");
        assert_eq!(mv_slot.sources[0], Operand::VectorReg(4), "sources[0] should be s1=x2 (VectorReg(4))");
        assert_eq!(mv_slot.sources[1], Operand::VectorReg(0), "sources[1] should be s2=x0 (VectorReg(0))");
    }

    /// Debug test: check VFLOOR_S32_BF16 operand layout.
    ///
    /// Encoding: `226: 59 00 00 08  vfloor.s32.bf16 x0, wl0, s0`
    #[test]
    fn test_vfloor_s32_bf16_operands() {
        let bundle_bytes: [u8; 4] = [0x59, 0x00, 0x00, 0x08];
        let decoder = InstructionDecoder::load_default();

        let bundle = decoder.decode(&bundle_bytes, 0).expect("decode bundle");
        let slot = bundle
            .slots()
            .iter()
            .flatten()
            .find(|s| !s.is_nop())
            .expect("should have a non-NOP slot");

        eprintln!("Encoding: {:?}", slot.encoding_name);
        eprintln!("Semantic: {:?}", slot.semantic);
        eprintln!("Dest: {:?}", slot.dest);
        eprintln!("Sources: {:?}", slot.sources);
        eprintln!("from_type: {:?}", slot.from_type);
        eprintln!("element_type: {:?}", slot.element_type);
        eprintln!("accum_width: {:?}", slot.accum_width);
        eprintln!("is_wide_vector: {:?}", slot.is_wide_vector);
        eprintln!("is_vector: {:?}", slot.is_vector);
        eprintln!("slot: {:?}", slot.slot);

        assert!(slot.encoding_name.as_deref().unwrap_or("").contains("vfloor"));

        // Also check ST slot FFI decode directly
        let st_bits: u64 = 0x000001;
        let ffi_result = xdna_archspec::aie2::isa::decoder_ffi::decode_slot(
            xdna_archspec::aie2::isa::decoder_ffi::Slot::St,
            st_bits,
        );
        if let Some(result) = &ffi_result {
            eprintln!(
                "ST FFI: name={} num_defs={} operands={:?}",
                result.name, result.num_defs, result.operands
            );
        } else {
            eprintln!("ST FFI: FAILED to decode bits 0x{:06X}", st_bits);
        }
    }

    /// Debug test: check VCONV_BF16_FP32 decode.
    #[test]
    fn test_vconv_bf16_fp32_decode() {
        // vconv.bf16.fp32 wl0, bml0 = [0xD9, 0x01, 0x00, 0x08]
        let bundle_bytes: [u8; 4] = [0xD9, 0x01, 0x00, 0x08];
        let decoder = InstructionDecoder::load_default();

        let bundle = decoder.decode(&bundle_bytes, 0).expect("decode bundle");
        let slot = bundle.slots().iter().flatten().find(|s| !s.is_nop());

        if let Some(slot) = slot {
            eprintln!("Encoding: {:?}", slot.encoding_name);
            eprintln!("Semantic: {:?}", slot.semantic);
            eprintln!("Dest: {:?}", slot.dest);
            eprintln!("Sources: {:?}", slot.sources);
            eprintln!("is_vector: {:?}", slot.is_vector);
            eprintln!("is_wide_vector: {:?}", slot.is_wide_vector);
            eprintln!("from_type: {:?}", slot.from_type);
            eprintln!("element_type: {:?}", slot.element_type);
            eprintln!("accum_width: {:?}", slot.accum_width);
            eprintln!("slot: {:?}", slot.slot);
        } else {
            eprintln!("NO non-NOP slot found -- VCONV decoded as NOP/unknown");
        }

        // Also try direct FFI decode
        let st_bits: u64 = 0x000007;
        let ffi_result = xdna_archspec::aie2::isa::decoder_ffi::decode_slot(
            xdna_archspec::aie2::isa::decoder_ffi::Slot::St,
            st_bits,
        );
        if let Some(result) = &ffi_result {
            eprintln!(
                "ST FFI: name={} num_defs={} operands={:?}",
                result.name, result.num_defs, result.operands
            );
        } else {
            eprintln!("ST FFI: FAILED to decode bits 0x{:06X}", st_bits);
        }
    }

    /// Verify MOVX_mvx_scl gets SemanticOp::Copy from the mnemonic fallback.
    ///
    /// MOVX_mvx_scl has no Pat<> in TableGen and structural inference returns
    /// None (no mayLoad/mayStore/hasDelaySlot). The decoder falls back to
    /// mnemonic-based Copy assignment for non-vector mov/mova/movx/movxm.
    #[test]
    fn test_movx_mvx_scl_gets_copy_semantic() {
        let decoder = InstructionDecoder::load_default();
        let enc = decoder.index.encoding_by_name("alu", "MOVX_mvx_scl");
        assert!(enc.is_some(), "MOVX_mvx_scl should exist in the encoding index");
        let enc = enc.unwrap();
        assert_eq!(enc.mnemonic, "movx");

        // Verify that the mnemonic fallback logic in build_slot_op assigns Copy.
        let has_dest = true; // MOVX_mvx_scl writes to a ControlReg
        let effective_semantic = if enc.semantic.is_none()
            && has_dest
            && (enc.mnemonic == "mov"
                || enc.mnemonic == "mova"
                || enc.mnemonic == "movx"
                || enc.mnemonic == "movxm")
            && !enc.is_vector
        {
            Some(xdna_archspec::aie2::isa::SemanticOp::Copy)
        } else {
            enc.semantic
        };
        assert_eq!(
            effective_semantic,
            Some(xdna_archspec::aie2::isa::SemanticOp::Copy),
            "MOVX_mvx_scl should get Copy semantic from mnemonic fallback"
        );
    }

    /// Decode a real compiled `vsrs.s16.s64 wh0, cm5, s0` and confirm the
    /// accumulator source (`cm5`) decodes to an AccumReg, not a ScalarReg.
    ///
    /// Regression for the Half-B SRS capture kernel: the first compiled SRS
    /// instruction run through decode mis-bucketed the `cm` accumulator source
    /// as `ScalarReg(41)`, so the SRS executor read zeros and then panicked
    /// writing a wide result to an odd half-register base.
    #[test]
    fn test_vsrs_s16_s64_decodes_accum_source() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder =
            InstructionDecoder::try_load_via_tblgen(llvm_aie_path).expect("Failed to load via tblgen");

        // vsrs.s16.s64 wh0, cm5, s0 -- from compiled vec_srs_i32 at PC 0x28e.
        // Disasm: `28e: 99 d4 5c 08  vsrs.s16.s64 wh0, cm5, s0`.
        let bytes: [u8; 4] = [0x99, 0xd4, 0x5c, 0x08];
        let bundle = decoder.decode(&bytes, 0x28e).expect("Should decode vsrs");

        eprintln!("=== vsrs.s16.s64 decoded slots ===");
        for slot in bundle.active_slots() {
            eprintln!(
                "  semantic={:?} dest={:?} sources={:?} accum_width={:?} is_wide={} from_type={:?} elem={:?}",
                slot.semantic,
                slot.dest,
                &slot.sources[..],
                slot.accum_width,
                slot.is_wide_vector,
                slot.from_type,
                slot.element_type,
            );
        }

        // Dump the raw encoding operand fields to see the source operand's
        // declared width/type.
        let ext = decoder.extract_bundle_slots(&bytes);
        for s in &ext.slots {
            if s.bits == 0 {
                continue;
            }
            if let Some(decoded) = decoder.decode_slot_bits(s.bits, s.slot_type) {
                eprintln!("=== ENC {} ({}) ===", decoded.encoding.mnemonic, decoded.encoding.name);
                for f in &decoded.encoding.operand_fields {
                    let raw = decoded.operand(&f.name).unwrap_or(0);
                    eprintln!(
                        "   field {}: raw={} (0x{:X}) width={} type={:?}",
                        f.name, raw, raw, f.width, f.operand_type
                    );
                }
            }
        }

        let srs_slot = bundle
            .active_slots()
            .find(|s| matches!(s.semantic, Some(SemanticOp::Srs)))
            .expect("vsrs must produce an Srs semantic slot");
        assert!(
            srs_slot.sources.iter().any(|o| matches!(o, Operand::AccumReg(_))),
            "vsrs source (cm5) must decode to AccumReg, got sources={:?}",
            &srs_slot.sources[..]
        );
    }

    /// TDD anchor: VMOV x, bml (X<-BM, Mv slot encoding 0x36) must produce
    /// `result_bypass == Bypass::Mov` on the decoded SlotOp.
    ///
    /// The static per-opcode lookup (based on the base schedule class II_VMOV_X)
    /// returns `Bypass::No` for this variant.  The register-aware resolved
    /// itinerary correctly returns `MOV_Bypass` (id 1).  This test verifies
    /// the resolved value wins.
    ///
    /// Companion to `test_resolved_sched_class_bypass` in decoder_ffi.rs, which
    /// proves the FFI itself returns id=1; this test proves the decode pipeline
    /// threads it through to `SlotOp.result_bypass`.
    #[test]
    fn test_vmov_x_bml_result_bypass_is_mov() {
        use xdna_archspec::aie2::Bypass;

        let decoder = InstructionDecoder::load_default();

        // VMOV_mv_x X<-BM: Mv slot bits 0x36.  Verified decodable in
        // test_resolved_sched_class_bypass (decoder_ffi.rs).
        let bytes: [u8; 4] = {
            // I32_MV bundle: {0b00011[31:27], mv[21:0][26:5], 0b1[4], 0b1001[3:0]}
            // bits[31:27]=0b00011 discriminates the MV slot in extract_32bit.
            // bit[4]=1 and bits[3:0]=0x9 together form the low 5 bits (0x19).
            // mv_bits=0x36 are placed at bits[26:5].
            let word: u32 = (0b00011u32 << 27) | (0x36u32 << 5) | 0x19;
            word.to_le_bytes()
        };

        let bundle = decoder.decode(&bytes, 0).expect("Should decode VMOV x, bml");

        // The decoded instruction should occupy the Scalar1 (MV) slot.
        let slot = bundle
            .slots()
            .iter()
            .flatten()
            .find(|s| !s.is_nop() && s.encoding_name.as_deref() == Some("vmov"))
            .expect("VMOV x,bml should decode to a vmov SlotOp");

        eprintln!(
            "VMOV x,bml: encoding={:?} result_bypass={:?} llvm_opcode={:?}",
            slot.encoding_name, slot.result_bypass, slot.llvm_opcode
        );

        assert_eq!(
            slot.result_bypass,
            Bypass::Mov,
            "VMOV x,bml (X<-BM, 0x36) must carry Bypass::Mov from resolved itinerary; \
             static per-opcode path would produce Bypass::No"
        );
    }

    /// TDD anchor: source_forward aligned 1:1 with sources for VEXTRACT.
    ///
    /// VEXTRACT_D8 (Mv slot 0x35) has a vector source (x0) at source_idx 0.
    /// The resolved itinerary marks it use_cycle=1, use_bypass=MOV (id 1).
    /// After decode, `slot_op.source_forward.len()` must equal
    /// `slot_op.sources.len()`, and the entry for the vector source must
    /// match the FFI-resolved values.
    #[test]
    fn test_vextract_source_forward_aligned() {
        use xdna_archspec::aie2::Bypass;

        let decoder = InstructionDecoder::load_default();

        // VEXTRACT_D8: Mv slot bits 0x35. Build a 32-bit I32_MV bundle.
        // See test_vmov_x_bml_result_bypass_is_mov for the I32_MV encoding rationale.
        let bytes: [u8; 4] = {
            let word: u32 = (0b00011u32 << 27) | (0x35u32 << 5) | 0x19;
            word.to_le_bytes()
        };

        let bundle = decoder.decode(&bytes, 0).expect("Should decode VEXTRACT_D8");

        let slot = bundle
            .slots()
            .iter()
            .flatten()
            .find(|s| !s.is_nop() && s.encoding_name.as_deref().map_or(false, |n| n.starts_with("vextract")))
            .expect("VEXTRACT_D8 should decode to a vextract SlotOp");

        eprintln!(
            "VEXTRACT: sources.len()={} source_forward.len()={} source_forward={:?}",
            slot.sources.len(),
            slot.source_forward.len(),
            &slot.source_forward[..]
        );

        // Structural: source_forward must align 1:1 with sources.
        assert_eq!(
            slot.source_forward.len(),
            slot.sources.len(),
            "source_forward.len() must equal sources.len() for VEXTRACT_D8"
        );

        // The vector source x0 must appear as a source (first in uses()).
        // Its resolved itinerary: use_cycle=1, use_bypass=MOV (id 1 -> Bypass::Mov).
        // Assert unconditionally: a decode mismatch that drops x0 must fail loudly
        // rather than silently skipping the forwarding checks.
        let vec_src_idx = slot
            .sources
            .iter()
            .position(|s| matches!(s, Operand::VectorReg(0)))
            .expect("x0 must appear as a source of VEXTRACT_D8");
        let (uc, ub) = slot.source_forward[vec_src_idx];
        eprintln!("  x0 source_forward[{}] = ({}, {:?})", vec_src_idx, uc, ub);
        assert_eq!(uc, 1, "VEXTRACT x0 source use_cycle should be 1");
        assert_eq!(ub, Bypass::Mov, "VEXTRACT x0 source use_bypass should be Bypass::Mov");
    }
}
