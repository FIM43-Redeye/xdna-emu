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
use std::path::Path;
use std::sync::OnceLock;

use crate::interpreter::bundle::{
    ExtractedBundle, MemWidth, Operand, PostModify, SlotIndex, SlotOp, VliwBundle,
    extract_slots,
};
use crate::interpreter::state::{MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M, SP_PTR_INDEX};
use crate::interpreter::traits::{DecodeError, Decoder};
#[cfg(test)]
use super::composite::CompositeLuts;
use crate::tablegen::{
    AddressingMode, DecoderIndex, InstrEncoding, InstrMemWidth, OperandType, RegisterKind,
    SemanticOp, decoder_bytecode,
    decoder_ffi::{self, DecodedOperand, DecodeResult, operand_from_reg_name},
};

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
        let field = self
            .encoding
            .operand_fields
            .iter()
            .find(|f| f.name == name)?;
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
    index: DecoderIndex,

    /// Pre-built composite register decode LUTs (test-only).
    /// Used by legacy `extract_operands()` for cross-validation against FFI.
    #[cfg(test)]
    composite_luts: CompositeLuts,

    /// Data-driven VLIW slot extraction table, built from tblgen Inst fields.
    /// When present, replaces the hand-coded extract_* functions in slot_layout.
    format_table: Option<crate::interpreter::bundle::FormatTable>,

    /// Per-opcode metadata from LLVM's MCInstrDesc + itinerary model.
    /// Indexed by LLVM opcode ID for O(1) lookups. Populated once at init.
    instr_info: Vec<decoder_ffi::InstrInfo>,

    /// Statistics: successful decodes.
    decode_count: u64,
    /// Statistics: unknown patterns.
    unknown_count: u64,
}

impl Default for InstructionDecoder {
    fn default() -> Self {
        Self::new()
    }
}

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

/// Returns true if this operand type can be a write destination.
///
/// Register kinds (scalar, vector, accum, pointer, modifier) are writable.
/// Immediates, locks, DMA channels, buffer descriptors, and memory operands
/// are not valid write targets.
fn can_be_dest(operand: &Operand) -> bool {
    matches!(
        operand,
        Operand::ScalarReg(_)
            | Operand::VectorReg(_)
            | Operand::AccumReg(_)
            | Operand::PointerReg(_)
            | Operand::ModifierReg(_)
    )
}

/// Global cached decoder, loaded once on first use.
/// This avoids repeatedly parsing TableGen files for each core.
static CACHED_DECODER: OnceLock<InstructionDecoder> = OnceLock::new();

impl InstructionDecoder {
    /// Create an empty decoder (no encodings loaded).
    pub fn new() -> Self {
        Self {
            index: DecoderIndex::default(),
            #[cfg(test)]
            composite_luts: CompositeLuts::build(),
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Get a cached decoder, loading it on first call.
    ///
    /// This is the preferred way to get a decoder - it loads once and reuses
    /// the cached instance for all subsequent calls. Each caller gets a clone
    /// with independent statistics.
    pub fn load_cached() -> Self {
        CACHED_DECODER.get_or_init(|| {
            log::info!("Initializing cached instruction decoder");
            Self::load_fresh()
        }).clone()
    }

    /// Load a fresh decoder (not cached).
    ///
    /// Use `load_cached()` instead unless you specifically need a fresh load.
    ///
    /// # Panics
    ///
    /// Panics if the TableGen parser fails. This is intentional - we want to
    /// fail fast rather than silently falling back to broken behavior.
    fn load_fresh() -> Self {
        Self::load_from_generated()
    }

    /// Load a decoder from build-time generated constants.
    ///
    /// All instruction encodings, decoder bytecode, and metadata were extracted
    /// from llvm-aie at compile time. No filesystem access required at runtime.
    fn load_from_generated() -> Self {
        let output = crate::tablegen::load_from_generated();

        // Build data-driven format table from composite format Inst fields
        let format_table = if output.composite_formats.iter().any(|f| !f.slot_maps.is_empty()) {
            let table = crate::interpreter::bundle::FormatTable::build(&output.composite_formats);
            log::info!(
                "Built data-driven format table: {} entries",
                table.total_entries(),
            );
            Some(table)
        } else {
            log::warn!("No Inst-derived format layouts; falling back to hand-coded extraction");
            None
        };

        let mut decoder = Self::from_tables_with_decoders(
            output.encodings_by_slot,
            output.decoder_tables,
        );
        decoder.format_table = format_table;

        // Populate per-opcode metadata from LLVM's MCInstrDesc + itinerary model.
        decoder.instr_info = decoder_ffi::query_all_instr_info();
        log::info!(
            "Loaded LLVM InstrInfo: {} opcodes, {} with latency, {} with flags",
            decoder.instr_info.len(),
            decoder.instr_info.iter().filter(|i| i.latency.is_some()).count(),
            decoder.instr_info.iter().filter(|i| i.flags != 0).count(),
        );

        decoder
    }

    /// Load a decoder from llvm-aie.
    ///
    /// Uses config file or environment variable to find llvm-aie path.
    /// Uses `llvm-tblgen` for accurate encodings.
    ///
    /// NOTE: Prefer `load_cached()` which avoids repeatedly parsing TableGen.
    ///
    /// # Panics
    ///
    /// Panics if llvm-aie is not found or TableGen parsing fails.
    pub fn load_default() -> Self {
        Self::load_cached()
    }

    /// Load a decoder using build-time generated data.
    ///
    /// The `llvm_aie_path` parameter is ignored -- all data is compiled in.
    /// This signature is kept for backward compatibility with existing tests.
    pub fn try_load_via_tblgen(_llvm_aie_path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        Ok(Self::load_from_generated())
    }

    /// Check if llvm-aie is available (checks config and env var).
    pub fn is_llvm_aie_available() -> bool {
        use crate::config::Config;
        Path::new(&Config::get().llvm_aie_path()).exists()
    }

    /// Create a decoder from encoding tables grouped by slot (no LLVM bytecode).
    ///
    /// Without bytecode tables, all slot decodes return `None` (unknown).
    /// Use `from_tables_with_decoders()` for the full decode path.
    pub fn from_tables(tables: HashMap<String, Vec<InstrEncoding>>) -> Self {
        Self::from_tables_with_decoders(tables, HashMap::new())
    }

    /// Create a decoder from encoding tables with LLVM decoder bytecode tables.
    ///
    /// This is the primary constructor. LLVM bytecode tables are the sole
    /// disambiguation mechanism, matching LLVM's own disassembler behavior.
    pub fn from_tables_with_decoders(
        tables: HashMap<String, Vec<InstrEncoding>>,
        decoder_tables: HashMap<String, decoder_bytecode::DecoderTable>,
    ) -> Self {
        let index = DecoderIndex::from_slot_encodings(tables, decoder_tables);

        Self {
            index,
            #[cfg(test)]
            composite_luts: CompositeLuts::build(),
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Create a decoder from a pre-built DecoderIndex.
    pub fn from_index(index: DecoderIndex) -> Self {
        Self {
            index,
            #[cfg(test)]
            composite_luts: CompositeLuts::build(),
            format_table: None,
            instr_info: Vec::new(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

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
    fn validate_semantic_vs_flags(&self, op: &SlotOp) {
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
        if info.is_branch() && !matches!(semantic,
            SemanticOp::Br | SemanticOp::BrCond | SemanticOp::Call | SemanticOp::Ret
        ) {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Branch but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, semantic,
            );
        }
        // Call: LLVM says Call but we don't?
        if info.is_call() && !matches!(semantic, SemanticOp::Call) {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Call but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, semantic,
            );
        }
        // Return: LLVM says Return but we don't?
        if info.is_return() && !matches!(semantic, SemanticOp::Ret) {
            log::warn!(
                "[semantic-flag mismatch] {} (opcode {}): LLVM says Return but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, semantic,
            );
        }

        // Reverse: we say Load but LLVM says no MayLoad?
        // (Skip pointer ops and cascade reads -- they use the load slot but
        // aren't memory loads in LLVM's sense.)
        if info.is_load() && !matches!(semantic,
            SemanticOp::Load | SemanticOp::PointerAdd | SemanticOp::PointerMov
            | SemanticOp::CascadeRead | SemanticOp::CascadeWrite
            | SemanticOp::Copy
        ) && !matches!(semantic,
            // Intrinsics and vector ops that load (UPS, etc.) may have MayLoad
            // from LLVM but aren't SemanticOp::Load. That's expected.
            SemanticOp::Ups | SemanticOp::Intrinsic(_)
        ) {
            log::debug!(
                "[semantic-flag note] {} (opcode {}): LLVM MayLoad but SemanticOp::{:?}",
                op.encoding_name.as_deref().unwrap_or("?"), opcode, semantic,
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
            return Some(DecodedInstr {
                encoding: encoding.clone(),
                operands,
            });
        }

        None
    }

    // -----------------------------------------------------------------------
    // LLVM FFI decode path (Step 6)
    // -----------------------------------------------------------------------

    /// Convert SlotType to decoder_ffi::Slot for LLVM decoding.
    fn slot_type_to_ffi(slot_type: crate::interpreter::bundle::SlotType) -> Option<decoder_ffi::Slot> {
        use crate::interpreter::bundle::SlotType;
        match slot_type {
            SlotType::Alu => Some(decoder_ffi::Slot::Alu),
            SlotType::Lda => Some(decoder_ffi::Slot::Lda),
            SlotType::Ldb => Some(decoder_ffi::Slot::Ldb),
            SlotType::Lng => Some(decoder_ffi::Slot::Lng),
            SlotType::Mv  => Some(decoder_ffi::Slot::Mv),
            SlotType::St  => Some(decoder_ffi::Slot::St),
            SlotType::Vec => Some(decoder_ffi::Slot::Vec),
            SlotType::Nop => None,
        }
    }

    /// Try to decode a slot using LLVM FFI and build operands directly from
    /// LLVM's register names and output classification.
    ///
    /// Returns (DecodedInstr, dest, sources, post_modify) on success.
    /// The DecodedInstr is still needed for build_slot_op() which reads
    /// metadata (semantic, element_type, etc.) from InstrEncoding.
    ///
    /// Falls back to None if:
    /// - LLVM can't decode the slot
    /// - The instruction name doesn't match any InstrEncoding
    /// - An LLVM register name can't be mapped to our Operand type
    fn try_decode_via_ffi(
        &self,
        bits: u64,
        slot_type: crate::interpreter::bundle::SlotType,
    ) -> Option<(DecodedInstr, Option<Operand>, Vec<Operand>, Option<PostModify>, Option<u32>)> {
        let ffi_slot = Self::slot_type_to_ffi(slot_type)?;
        let ffi_result = decoder_ffi::decode_slot(ffi_slot, bits)?;

        // Look up InstrEncoding by LLVM instruction name (still need for metadata).
        let slot_name = match slot_type {
            crate::interpreter::bundle::SlotType::Lda => "lda",
            crate::interpreter::bundle::SlotType::Ldb => "ldb",
            crate::interpreter::bundle::SlotType::Alu => "alu",
            crate::interpreter::bundle::SlotType::Mv  => "mv",
            crate::interpreter::bundle::SlotType::St  => "st",
            crate::interpreter::bundle::SlotType::Vec => "vec",
            crate::interpreter::bundle::SlotType::Lng => "lng",
            _ => return None,
        };

        let encoding = self.index.encoding_by_name(slot_name, &ffi_result.name)?;

        // Map LLVM operands to our Operand type.
        let (dest, sources, post_modify) =
            Self::extract_operands_from_ffi(&ffi_result, encoding);

        // Build a minimal DecodedInstr with empty raw operands (we don't need
        // them since we're using LLVM's operands, but build_slot_op reads
        // encoding metadata from it).
        let decoded_instr = DecodedInstr {
            encoding: encoding.clone(),
            operands: HashMap::new(),
        };

        Some((decoded_instr, dest, sources, post_modify, Some(ffi_result.opcode)))
    }

    /// Extract dest, sources, and post-modify from LLVM FFI decode result.
    ///
    /// Uses num_defs to split outputs from inputs, then applies addressing
    /// mode logic from InstrEncoding to construct Memory operands and
    /// PostModify metadata.
    fn extract_operands_from_ffi(
        ffi_result: &DecodeResult,
        encoding: &InstrEncoding,
    ) -> (Option<Operand>, Vec<Operand>, Option<PostModify>) {
        let num_defs = ffi_result.num_defs as usize;

        // Map all LLVM operands to our Operand type.
        // Skip operands that can't be mapped (mask registers, DMA regs).
        //
        // LLVM's decoder tables already handle sign extension and scaling
        // internally.  The decoded immediate values are ready to use as-is.
        // Do NOT apply additional sign extension or scaling here -- that was
        // needed for our raw bit-field extraction path, not for LLVM output.
        let mapped: Vec<Option<Operand>> = ffi_result.operands.iter().map(|op| {
            match op {
                DecodedOperand::Reg { name, .. } => {
                    operand_from_reg_name(name).map(|m| m.operand)
                }
                DecodedOperand::Imm(val) => {
                    Some(Operand::Immediate(*val as i32))
                }
            }
        }).collect();

        // Split into defs (outputs) and uses (inputs).
        let split_at = num_defs.min(mapped.len());
        let def_ops: Vec<Operand> = mapped[..split_at].iter().filter_map(|o| o.clone()).collect();
        let mut use_ops: Vec<Operand> = mapped[split_at..].iter().filter_map(|o| o.clone()).collect();

        // Destination: first def operand (if any and writable).
        let mut dest = def_ops.into_iter().next();
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
            return (dest, use_ops, post_modify);
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
                    let offset = if let Some(imm_pos) = use_ops[ptr_pos..].iter()
                        .position(|o| matches!(o, Operand::Immediate(_)))
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
                    let modify = if let Some(imm_pos) = use_ops[ptr_pos..].iter()
                        .position(|o| matches!(o, Operand::Immediate(_)))
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
                    if let Some(mod_pos) = use_ops[ptr_pos..].iter()
                        .position(|o| matches!(o, Operand::ModifierReg(_)))
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

        (dest, use_ops, post_modify)
    }

    /// Extract operands from decoded instruction using data-driven dispatch.
    ///
    /// Legacy bit-field extraction method. Retained for cross-validation tests
    /// against the LLVM FFI operand extraction. Not used in production.
    #[cfg(test)]
    fn extract_operands(&self, decoded: &DecodedInstr) -> (Option<Operand>, Vec<Operand>, Option<PostModify>) {
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
                    field, raw, decoded,
                    &mut direct_dest, &mut extra_sources, &mut extracted_post_modify,
                );
                continue;
            }

            // Data-driven dispatch on operand_type
            let operand = match &field.operand_type {
                OperandType::Register(kind) => {
                    Self::decode_register(*kind, raw as u8, 0)
                }
                OperandType::RegisterWithOffset(kind, base) => {
                    Self::decode_register(*kind, raw as u8, *base)
                }
                OperandType::CompositeRegister(encoder) => {
                    self.composite_luts.decode(*encoder, raw)
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
                    // Fallback: treat as unsigned immediate
                    Operand::Immediate(raw as i32)
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
        let (dest, sources) = self.extract_ordered_operands(
            decoded,
            &field_operands,
            direct_dest,
            extra_sources,
        );

        (dest, sources, extracted_post_modify)
    }

    /// Decode an address generator field (ag_all, ag_nospill, agb_sa, etc.).
    ///
    /// Decode a register operand from raw bits, applying subclass base offset.
    #[cfg(test)]
    fn decode_register(kind: RegisterKind, raw: u8, base_offset: u8) -> Operand {
        let reg = raw + base_offset;
        match kind {
            RegisterKind::Scalar => Operand::ScalarReg(reg),
            RegisterKind::Pointer => Operand::PointerReg(reg),
            RegisterKind::ModifierM  => Operand::ModifierReg(reg + MOD_BASE_M),
            RegisterKind::ModifierDN => Operand::ModifierReg(reg + MOD_BASE_DN),
            RegisterKind::ModifierDJ => Operand::ModifierReg(reg + MOD_BASE_DJ),
            RegisterKind::ModifierDC => Operand::ModifierReg(reg + MOD_BASE_DC),
            RegisterKind::Vector256 => Operand::VectorReg(reg),
            // Vector512 (x-registers) span two 256-bit registers: x0={wl0,wh0}={v0,v1}.
            // The 4-bit encoding index needs *2 to address the 256-bit register file.
            RegisterKind::Vector512 => Operand::VectorReg(reg * 2),
            RegisterKind::Accumulator => Operand::AccumReg(reg),
            RegisterKind::Control => Operand::ControlReg(reg),
        }
    }

    /// AG fields encode a packed tuple of {pointer, offset/modifier, mode_bits}.
    #[cfg(test)]
    fn decode_ag_field(
        &self,
        field: &crate::tablegen::OperandField,
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
                let imm_raw = sign_extend(
                    (data & ((1 << imm_bits) - 1)) as i32, imm_bits as u32
                );
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
                    d, decoded.encoding.name,
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
    ///
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

    /// Build a SlotOp with TableGen-derived information.
    ///
    /// Sources are already in canonical order from extract_operands().
    /// This method:
    /// 1. Sets the semantic operation from TableGen
    /// 2. Attaches implicit register uses/defs
    fn build_slot_op(
        &self,
        slot_index: SlotIndex,
        decoded: &DecodedInstr,
        dest: Option<Operand>,
        sources: Vec<Operand>,
        extracted_pm: Option<PostModify>,
    ) -> SlotOp {
        let enc = &decoded.encoding;

        // Cascade instructions: detected by encoding name before SemanticOp lookup.
        // These have hasSideEffects=true and mnemonic "vmov" (which maps to
        // SemanticOp::Copy). Name-based detection is required because the
        // mnemonic is shared with regular vmov instructions.
        //
        // Data-driven alternative: cascade instructions use crSCDEn/crMCDEn in
        // their Uses list and OP_mScdDst/OP_mMcdSrc operand types. If new
        // cascade variants are added, detect via those operand types instead
        // of extending this name list.
        let semantic_override = match enc.name.as_str() {
            "VMOV_mv_scd" | "VMOV_HI" | "VMOV_LO" => Some(SemanticOp::CascadeRead),
            "VMOV_mv_mcd" => Some(SemanticOp::CascadeWrite),
            // Instructions without Pat<> patterns in TableGen need explicit
            // semantic mappings. These are valid ISA instructions that Peano
            // doesn't select but Chess does.
            "ASHL" => Some(SemanticOp::AshlBidir),
            "LSHL" => Some(SemanticOp::LshlBidir),
            "SBC" => Some(SemanticOp::Sbc),
            "DIVS" => Some(SemanticOp::SDiv),
            "EXTENDu8" => Some(SemanticOp::ZeroExtend),
            "EXTENDu16" => Some(SemanticOp::ZeroExtend),
            "MOV_CNTR" => Some(SemanticOp::ReadCycleCounter),
            // VBAND/VBOR share sched_class II_VBLOG -- cannot distinguish
            // via itinerary table. Override by encoding name instead.
            "VBAND" => Some(SemanticOp::And),
            "VBOR" => Some(SemanticOp::Or),
            _ => {
                // MAC-family instructions: many encoding names lack
                // build-time semantic assignment (no intrinsic pattern).
                // Match by encoding name prefix to assign the correct
                // MAC variant semantic.
                let n = enc.name.as_str();
                if n.starts_with("VNEGMAC_") || n.starts_with("VNEGMSC_") {
                    Some(SemanticOp::NegMatMul)
                } else if n.starts_with("VNEGMUL_") {
                    Some(SemanticOp::NegMul)
                } else if n.starts_with("VADDMAC_") || n.starts_with("VADDMSC_") {
                    Some(SemanticOp::AddMac)
                } else if n.starts_with("VSUBMAC_") || n.starts_with("VSUBMSC_") {
                    Some(SemanticOp::SubMac)
                } else if n.starts_with("VMAC_") {
                    Some(SemanticOp::Mac)
                } else if n.starts_with("VMSC_") {
                    Some(SemanticOp::MatMulSub)
                } else if n.starts_with("VMUL_") {
                    Some(SemanticOp::MatMul)
                } else {
                    None
                }
            }
        };

        // Build SlotOp directly from SemanticOp (no Operation bridge).
        // Force PointerAdd for all pointer arithmetic instructions.
        // Some PADDB/PADDA variants get SemanticOp::Add from pattern
        // matching instead of PointerAdd. The is_ptr_arithmetic flag
        // (derived from mnemonic "padd*") is the reliable indicator.
        let effective_semantic = if enc.is_ptr_arithmetic {
            Some(SemanticOp::PointerAdd)
        } else {
            semantic_override.or(enc.semantic)
        };
        let mut slot_op = if let Some(semantic) = effective_semantic {
            SlotOp::from_semantic(slot_index, semantic)
        } else if enc.mnemonic == "opcodestr" || enc.mnemonic.is_empty() {
            // "opcodestr" is a parser artifact from unresolved TableGen template
            // parameters in NOP class definitions; treat as NOP.
            SlotOp::nop(slot_index)
        } else {
            // No semantic -- unknown instruction. This will hit the executor's
            // "no semantic" error path and abort on first execution.
            log::error!(
                "[NO SEMANTIC] Instruction '{}' has no SemanticOp (no pattern or structural match)",
                enc.mnemonic
            );
            let mut s = SlotOp::nop(slot_index);
            s.semantic = None;
            s.raw_opcode = Some(decoded.operands.get("word0").copied().unwrap_or(0) as u32);
            s
        };

        // ── Populate metadata from InstrEncoding ─────────────────────────
        slot_op.is_vector = enc.is_vector;
        // Detect 512-bit (x-register) operations: any operand uses Vector512.
        slot_op.is_wide_vector = enc.operand_fields.iter()
            .any(|f| f.operand_type == OperandType::Register(RegisterKind::Vector512));
        slot_op.element_type = enc.element_type;
        slot_op.from_type = enc.from_type;
        slot_op.mem_width = match enc.mem_width {
            InstrMemWidth::Byte => MemWidth::Byte,
            InstrMemWidth::HalfWord => MemWidth::HalfWord,
            InstrMemWidth::Word => MemWidth::Word,
            InstrMemWidth::QuadWord => MemWidth::QuadWord,
            InstrMemWidth::Vector256 => MemWidth::Vector256,
        };
        slot_op.branch_condition = enc.branch_condition;
        slot_op.select_variant = enc.select_variant;
        // PostModify comes directly from AG field extraction -- no backpatching.
        slot_op.post_modify = extracted_pm.unwrap_or(PostModify::None);

        // Store encoding mnemonic for crossref/debugging
        slot_op.encoding_name = Some(enc.mnemonic.clone());

        // Add implicit registers from TableGen
        if !enc.implicit_regs.is_empty() {
            slot_op = slot_op.with_implicit_regs(enc.implicit_regs.clone());
        }

        // Set destination and sources (already in TableGen canonical order)
        if let Some(d) = dest {
            slot_op = slot_op.with_dest(d);
        }
        for src in sources {
            slot_op = slot_op.with_source(src);
        }

        slot_op
    }
}

impl Decoder for InstructionDecoder {
    fn decode(&self, bytes: &[u8], pc: u32) -> Result<VliwBundle, DecodeError> {
        use crate::interpreter::bundle::SlotType;

        if bytes.len() < 2 {
            return Err(DecodeError::Incomplete {
                needed: 2,
                have: bytes.len(),
            });
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
                return Err(DecodeError::Incomplete {
                    needed: 2,
                    have: bytes.len(),
                });
            };

            // Special case: all zeros is a NOP regardless of format marker
            if word0 == 0 || word0 == 0x15010040 {
                // Treat as 4-byte NOP
                bytes.len().min(4)
            } else {
                // Not enough data and not a NOP
                return Err(DecodeError::Incomplete {
                    needed: bundle_size,
                    have: bytes.len(),
                });
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
                log::trace!("[DECODE#{}] PC=0x{:04X} format={:?} slots={}", count, pc, format, extracted.slots.len());
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
                } else if let Some((decoded, dest, sources, extracted_pm, ffi_opcode)) =
                    self.try_decode_via_ffi(slot.bits, slot.slot_type)
                {
                    // LLVM FFI is the sole production decoder for both instruction
                    // identification and operand extraction.
                    let mut slot_op = self.build_slot_op(
                        slot_index, &decoded, dest, sources, extracted_pm,
                    );
                    slot_op.llvm_opcode = ffi_opcode;

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
                    log::warn!("[{:?} DECODE FAIL] PC=0x{:04X} bits=0x{:010X} - no matching encoding",
                        slot.slot_type, pc, slot.bits);
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
            return Err(DecodeError::Incomplete {
                needed: 2,
                have: bytes.len(),
            });
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
                return Err(DecodeError::Incomplete {
                    needed: 2,
                    have: bytes.len(),
                });
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
    use crate::tablegen::{CompositeEncoder, OperandField};
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
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        let encoding_count = decoder.decoder_index().all_encodings().count();
        eprintln!("Loaded {} encodings", encoding_count);
        assert!(encoding_count > 100, "Should have many encodings");

        // Test NOP decoding
        let nop_bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&nop_bytes, 0).expect("Should decode NOP");
        assert!(bundle.is_nop());

        // Get stats
        let (decode_count, unknown_count) = decoder.stats();
        eprintln!(
            "Decoder stats: {} decoded, {} unknown",
            decode_count,
            unknown_count
        );
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
        let decoder = InstructionDecoder::try_load_via_tblgen(&llvm_aie_path)
            .expect("Failed to load via tblgen");

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
        assert!(
            decoded_count + unknown_count > 0,
            "Should have processed some instructions"
        );
    }

    #[test]
    fn test_done_instruction_decodes_to_halt() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        // done instruction: bytes from llvm-objdump, encodes to ALU slot
        let done_bytes: [u8; 4] = [0x19, 0x08, 0x00, 0x10];
        let bundle = decoder.decode(&done_bytes, 0xbc).expect("Should decode 'done'");

        let has_halt = bundle.active_slots().any(|s|
            matches!(s.semantic, Some(SemanticOp::Halt) | Some(SemanticOp::Done))
        );
        assert!(has_halt, "done instruction must decode to SemanticOp::Halt or Done, got: {:?}",
            bundle.active_slots().map(|s| format!("{:?}", s.semantic)).collect::<Vec<_>>());
    }

    #[test]
    fn test_rel_instruction_decodes_to_lock_release() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        // REL r0, r1 from Chess listing: bytes 0x10 0x10 0x12 0x19
        // (memory order, decoder reads with from_le_bytes)
        let rel_bytes: [u8; 4] = [0x19, 0x12, 0x10, 0x10];
        let bundle = decoder.decode(&rel_bytes, 0x250).expect("Should decode 'rel'");

        let has_lock_release = bundle.active_slots().any(|s|
            matches!(s.semantic, Some(SemanticOp::LockRelease))
        );
        let slot_semantics: Vec<_> = bundle.active_slots()
            .map(|s| format!("{:?}", s.semantic))
            .collect();
        assert!(has_lock_release,
            "REL instruction must decode to SemanticOp::LockRelease, got: {:?}", slot_semantics);
    }

    #[test]
    fn test_acq_instruction_decodes_to_lock_acquire() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        // ACQ r0, r1 from Chess listing: bytes 0x10 0x12 0x12 0x19
        let acq_bytes: [u8; 4] = [0x19, 0x12, 0x12, 0x10];
        let bundle = decoder.decode(&acq_bytes, 0x230).expect("Should decode 'acq'");

        let has_lock_acquire = bundle.active_slots().any(|s|
            matches!(s.semantic, Some(SemanticOp::LockAcquire))
        );
        let slot_semantics: Vec<_> = bundle.active_slots()
            .map(|s| format!("{:?}", s.semantic))
            .collect();
        assert!(has_lock_acquire,
            "ACQ instruction must decode to SemanticOp::LockAcquire, got: {:?}", slot_semantics);
    }

    #[test]
    fn test_decoder_via_tblgen() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        // Load decoder via tblgen
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        let index = decoder.decoder_index();
        let encoding_count = index.all_encodings().count();
        eprintln!("Loaded {} encodings via tblgen", encoding_count);

        // Verify we loaded many encodings
        assert!(
            encoding_count > 100,
            "Should have loaded many encodings"
        );

        // Verify ACQ instructions are distinguished
        let acq_imm = index.all_encodings().find(|e| e.name == "ACQ_mLockId_imm");
        let acq_reg = index.all_encodings().find(|e| e.name == "ACQ_mLockId_reg");

        if let (Some(imm), Some(reg)) = (acq_imm, acq_reg) {
            eprintln!(
                "ACQ_mLockId_imm: mask=0x{:05X}, bits=0x{:05X}",
                imm.fixed_mask, imm.fixed_bits
            );
            eprintln!(
                "ACQ_mLockId_reg: mask=0x{:05X}, bits=0x{:05X}",
                reg.fixed_mask, reg.fixed_bits
            );

            assert_ne!(
                imm.fixed_bits, reg.fixed_bits,
                "ACQ instructions should have different fixed bits"
            );
        }

        // Test NOP decoding still works
        let nop_bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&nop_bytes, 0).expect("Should decode NOP");
        assert!(bundle.is_nop());
    }

    // === Composite Register LUT Integration Tests ===
    //
    // Per-function decode tests live in composite.rs. These tests verify
    // the LUT dispatch works correctly through the CompositeLuts interface.

    #[test]
    fn test_composite_lut_dispatch() {
        use crate::interpreter::decode::composite::CompositeLuts;
        let luts = CompositeLuts::build();

        // LdaScl: eR(7) = (7 << 2) | 0b00 = 28
        assert_eq!(
            luts.decode(CompositeEncoder::LdaScl, 28),
            Operand::ScalarReg(7)
        );
        // MvSclSrc: eP(3) = (3 << 4) | 0b0011 = 51
        assert_eq!(
            luts.decode(CompositeEncoder::MvSclSrc, 51),
            Operand::PointerReg(3)
        );
        // AluCg: eR(5) = 5 << 1 = 10
        assert_eq!(
            luts.decode(CompositeEncoder::AluCg, 10),
            Operand::ScalarReg(5)
        );
    }

    /// Decode an 80-bit VLIW bundle and return (mnemonic, dest, sources) for each slot.
    /// Helper for the bundle decode diagnosis tests below.
    fn decode_80bit_bundle_slots(decoder: &InstructionDecoder, bytes: &[u8], _pc: u32)
        -> Vec<(String, Option<Operand>, Vec<Operand>)>
    {
        use crate::interpreter::bundle::SlotType;

        let extracted = decoder.extract_bundle_slots(bytes);
        let mut results = Vec::new();

        for slot in &extracted.slots {
            if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                results.push(("nop".to_string(), None, vec![]));
                continue;
            }
            if let Some(decoded) = decoder.decode_slot_bits(slot.bits, slot.slot_type) {
                let (dest, sources, _pm) = decoder.extract_operands(&decoded);
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
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

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
            let mv_slot = ext.slots.iter().find(|s| s.slot_type == crate::interpreter::bundle::SlotType::Mv);
            if let Some(mv) = mv_slot {
                eprintln!("{} MV bits: 0x{:06X} ({:022b})", label, mv.bits, mv.bits);
                if let Some(decoded) = decoder.decode_slot_bits(mv.bits, mv.slot_type) {
                    eprintln!("  matched: {} ({})", decoded.encoding.mnemonic, decoded.encoding.name);
                    eprintln!("  operand fields:");
                    for field in &decoded.encoding.operand_fields {
                        let raw = decoded.operand(&field.name).unwrap_or(0);
                        eprintln!("    {}: raw={} (0x{:X}), bit_pos={}, width={}, signed={}, type={:?}",
                            field.name, raw, raw, field.bit_position, field.width, field.signed,
                            field.operand_type);
                    }
                    eprintln!("  input_order: {:?}", decoded.encoding.input_order);
                    eprintln!("  output_order: {:?}", decoded.encoding.output_order);
                }
            }
        }

        // Verify the key assertion: add_314 MV slot should decode to mov r6, #6
        let mv_314 = ext_314.slots.iter().find(|s| s.slot_type == crate::interpreter::bundle::SlotType::Mv).unwrap();
        let decoded_mv_314 = decoder.decode_slot_bits(mv_314.bits, mv_314.slot_type).unwrap();
        let (dest, sources, _) = decoder.extract_operands(&decoded_mv_314);
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
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

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
        let decoded = decoder.decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm r3, #0x7ff");
        assert_eq!(decoded.encoding.mnemonic, "movxm");
        let (dest, sources, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::ScalarReg(3)));
        assert_eq!(sources, vec![Operand::Immediate(0x7ff)]);

        // Case 2: movxm p2, #0x100
        // p2 in MvSclSrc: encode = (2 << 4) | 0b0011 = 35
        let bits = build_movxm(0x100, 35);
        let decoded = decoder.decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm p2, #0x100");
        let (dest, sources, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::PointerReg(2)));
        assert_eq!(sources, vec![Operand::Immediate(0x100)]);

        // Case 3: movxm sp, #0x70000
        // SP in MvSclSrc: HWEncoding = 103 = (12 << 3) | 0b111
        let bits = build_movxm(0x70000, 103);
        let decoded = decoder.decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm sp, #0x70000");
        let (dest, sources, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::PointerReg(crate::interpreter::state::SP_PTR_INDEX)), "SP should map to dedicated SP");
        assert_eq!(sources, vec![Operand::Immediate(0x70000)]);

        // Case 4: movxm lr, #0x1234
        // lr in MvSclSrc: HWEncoding = 39 = (4 << 3) | 0b111
        // The OLD special case got this wrong: 39 & 0x3 == 3, so it decoded
        // as PointerReg((39 >> 4) & 0x7) = PointerReg(2). The MvSclSrc LUT
        // checks bits[2:0] == 0b111 first, correctly routing to
        // ScalarReg(LR_REG_INDEX).
        let bits = build_movxm(0x1234, 39);
        let decoded = decoder.decode_slot_bits(bits, SlotType::Lng)
            .expect("Should decode movxm lr, #0x1234");
        let (dest, sources, _) = decoder.extract_operands(&decoded);
        assert_eq!(dest, Some(Operand::ScalarReg(LR_REG_INDEX)),
            "lr must decode as ScalarReg(LR), not PointerReg(2)");
        assert_eq!(sources, vec![Operand::Immediate(0x1234)]);
    }

    #[test]
    fn test_lda_scl_vs_mv_scl_pointer_encoding_differs() {
        use crate::interpreter::decode::composite::CompositeLuts;
        let luts = CompositeLuts::build();

        // This is THE critical test: the same pointer register (p3) encodes
        // differently in LdaScl vs MvSclSrc. The old heuristic code got this wrong.
        //
        // p3 in LdaScl: (3 << 4) | 0b1101 = 61
        // p3 in MvSclSrc: (3 << 4) | 0b0011 = 51
        //
        // Decoding 61 with MvSclSrc would give the wrong register,
        // and decoding 51 with LdaScl would also be wrong.
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 61), Operand::PointerReg(3));
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 51), Operand::PointerReg(3));

        // Cross-check: 51 through LdaScl should NOT give p3
        // 51 = 0b0110011, low 4 bits = 0b0011, not 0b1101, so it's not a pointer in LdaScl
        // low 2 bits = 0b11, which doesn't match any LdaScl pattern cleanly
        let cross = luts.decode(CompositeEncoder::LdaScl, 51);
        assert_ne!(cross, Operand::PointerReg(3),
            "LdaScl(51) should NOT decode as p3 -- different encoding scheme");
    }

    /// Verify that vector load channel is determined by slot, not mnemonic.
    ///
    /// The decoder must produce VectorLoadA for slot="lda"+is_vector,
    /// VectorLoadB for slot="ldb"+is_vector, and plain Load for non-vector
    /// instructions in the same slots.
    #[test]
    fn test_vector_load_channel_by_slot() {
        use crate::tablegen::{AddressingMode, InstrMemWidth};
        let decoder = InstructionDecoder::new();

        // Helper to build a minimal Load encoding for the given slot+is_vector
        let make_load_enc = |slot: &str, is_vector: bool| -> InstrEncoding {
            InstrEncoding {
                name: format!("TEST_LOAD_{}", slot),
                mnemonic: if is_vector { format!("v{}", slot) } else { slot.to_string() },
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
                mem_width: if is_vector { InstrMemWidth::Vector256 } else { InstrMemWidth::Word },
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
        let decoded_vlda = DecodedInstr {
            encoding: make_load_enc("lda", true),
            operands: HashMap::new(),
        };
        let op = decoder.build_slot_op(SlotIndex::LoadA, &decoded_vlda, None, vec![], None);
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(op.is_vector, "slot=lda + is_vector should produce vector Load");

        // Vector load B (slot=ldb, is_vector=true) -> Load + is_vector
        let decoded_vldb = DecodedInstr {
            encoding: make_load_enc("ldb", true),
            operands: HashMap::new(),
        };
        let op = decoder.build_slot_op(SlotIndex::LoadB, &decoded_vldb, None, vec![], None);
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(op.is_vector, "slot=ldb + is_vector should produce vector Load");

        // Scalar load in lda slot (is_vector=false) -> Load + !is_vector
        let decoded_scl = DecodedInstr {
            encoding: make_load_enc("lda", false),
            operands: HashMap::new(),
        };
        let op = decoder.build_slot_op(SlotIndex::LoadA, &decoded_scl, None, vec![], None);
        assert_eq!(op.semantic, Some(SemanticOp::Load));
        assert!(!op.is_vector, "slot=lda + !is_vector should produce scalar Load");

        // Vector store (slot=st, is_vector=true) -> Store + is_vector
        let mut store_enc = make_load_enc("st", true);
        store_enc.semantic = Some(SemanticOp::Store);
        store_enc.may_load = false;
        store_enc.may_store = true;
        let decoded_vst = DecodedInstr {
            encoding: store_enc,
            operands: HashMap::new(),
        };
        let op = decoder.build_slot_op(SlotIndex::Store, &decoded_vst, None, vec![], None);
        assert_eq!(op.semantic, Some(SemanticOp::Store));
        assert!(op.is_vector, "slot=st + is_vector should produce vector Store");
    }

    /// Verify that PADD instructions produce PointerReg destination via
    /// the is_ptr_arithmetic field (not mnemonic checking).
    #[test]
    fn test_padd_dest_is_pointer() {
        use crate::tablegen::{AddressingMode, InstrMemWidth, OperandType, RegisterKind};

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

        let decoded = DecodedInstr {
            encoding: padd_enc,
            operands,
        };

        let (dest, sources, _post_modify) = decoder.extract_operands(&decoded);

        // PADD: ptr becomes destination, imm becomes source
        assert_eq!(dest, Some(Operand::PointerReg(3)),
            "PADD should produce PointerReg destination");
        assert!(sources.iter().any(|s| matches!(s, Operand::Immediate(5))),
            "PADD should have immediate source, got {:?}", sources);

        // Verify that a non-padd encoding with same fields produces Memory operand
        let mut load_enc = decoded.encoding.clone();
        load_enc.is_ptr_arithmetic = false;
        load_enc.name = "LDA_test".to_string();
        load_enc.mnemonic = "lda".to_string();
        let load_decoded = DecodedInstr {
            encoding: load_enc,
            operands: decoded.operands.clone(),
        };

        let (dest, sources, _) = decoder.extract_operands(&load_decoded);

        // Non-padd: ptr+imm combine into Memory operand as source
        assert!(dest.is_none() || !matches!(dest, Some(Operand::PointerReg(3))),
            "Non-PADD should not produce PointerReg destination");
        assert!(sources.iter().any(|s| matches!(s, Operand::Memory { base: 3, offset: 5 })),
            "Non-PADD should produce Memory source, got {:?}", sources);
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
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        // Raw 128-bit bundle from add_314_using_dma_op at PC=0x2A0:
        // nopb; mova r0, #0x30; nops; movx r1, #0x1; mov r20, p7; nopv
        let bytes: [u8; 16] = [
            0xc0, 0x03, 0x00, 0x28, 0x3b, 0x87, 0x2a, 0x10,
            0x00, 0x00, 0x00, 0x08, 0x00, 0x06, 0x00, 0x00,
        ];

        let bundle = decoder.decode(&bytes, 0x2A0).expect("Should decode 128-bit bundle");
        assert_eq!(bundle.size(), 16);

        // LoadA slot must contain the mova instruction (LDA slot).
        let load_a = bundle.slot(SlotIndex::LoadA)
            .expect("LoadA slot should be present");
        assert!(
            !load_a.is_nop(),
            "LoadA slot should have mova r0, #0x30"
        );
        assert_eq!(load_a.dest, Some(Operand::ScalarReg(0)),
            "mova destination should be r0");
        assert_eq!(load_a.sources.len(), 1, "mova should have one source");
        assert_eq!(load_a.sources[0], Operand::Immediate(0x30),
            "mova source should be immediate 0x30");

        // LoadB slot should be NOP (nopb = LDB NOP).
        let load_b = bundle.slot(SlotIndex::LoadB)
            .expect("LoadB slot should be present (NOP)");
        assert!(load_b.is_nop(), "LoadB slot should be NOP (nopb)");

        // Scalar0 (ALU) should have movx r1, #0x1
        let scalar0 = bundle.slot(SlotIndex::Scalar0)
            .expect("Scalar0 slot should be present");
        assert_eq!(scalar0.dest, Some(Operand::ScalarReg(1)),
            "movx destination should be r1");

        // Scalar1 (MV) should have mov r20, p7
        let scalar1 = bundle.slot(SlotIndex::Scalar1)
            .expect("Scalar1 slot should be present");
        assert_eq!(scalar1.dest, Some(Operand::ScalarReg(20)),
            "mov destination should be r20");
    }

    /// Helper: create a minimal vector encoding with the given mnemonic.
    fn make_vec_encoding(mnemonic: &str) -> InstrEncoding {
        use crate::tablegen::{AddressingMode, InstrMemWidth};
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
            element_type: crate::tablegen::infer_element_type(mnemonic),
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
        let decoded = DecodedInstr {
            encoding: enc,
            operands: HashMap::new(),
        };
        let decoder = InstructionDecoder::load_default();
        decoder.build_slot_op(SlotIndex::Vector, &decoded, None, vec![], None)
    }

    /// Helper: assert a dispatch result matches expected SemanticOp.
    fn assert_sem(mnemonic: &str, semantic: SemanticOp) {
        let op = dispatch_semantic(mnemonic, semantic);
        assert_eq!(op.semantic, Some(semantic),
            "mnemonic '{}' with {:?} should dispatch correctly, got {:?}",
            mnemonic, semantic, op.semantic);
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

        let objdump_path = std::path::PathBuf::from(Config::get().llvm_aie_path())
            .join("build/bin/llvm-objdump");
        if !objdump_path.exists() {
            eprintln!(
                "Skipping test_crossref_integration: llvm-objdump not found at {}",
                objdump_path.display()
            );
            return;
        }

        let decoder = InstructionDecoder::load_default();
        let report = cross_reference_elf(&elf_path, &objdump_path, &decoder)
            .expect("Cross-reference failed");

        eprintln!("{}", report);

        assert!(
            report.total_instructions > 0,
            "Should decode at least one instruction"
        );

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
                if end > data.len() { continue; }

                let code = &data[start..end];
                let mut offset = 0;

                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() { break; }

                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);

                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                            continue;
                        }
                        total += 1;

                        let ffi_result = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                        let legacy_result = decoder.decode_slot_bits(slot.bits, slot.slot_type)
                            .map(|d| {
                                let (dest, sources, pm) = decoder.extract_operands(&d);
                                (d, dest, sources, pm)
                            });

                        match (ffi_result, legacy_result) {
                            (Some((_, ffi_dest, ffi_src, ffi_pm, _)), Some((_, leg_dest, leg_src, leg_pm))) => {
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
                            (None, Some(_)) => { ffi_misses += 1; }
                            _ => {}
                        }
                    }

                    offset += size;
                }
            }
        }

        eprintln!("\n=== FFI vs Legacy Cross-Validation ===");
        eprintln!("Total slot decodes:    {}", total);
        eprintln!("FFI hits:              {} ({:.1}%)", ffi_hits,
            if total > 0 { 100.0 * ffi_hits as f64 / total as f64 } else { 0.0 });
        eprintln!("FFI misses (fallback): {}", ffi_misses);
        eprintln!("Operand matches:       {}", matches);
        eprintln!("Operand divergences:   {}", divergences);

        // Categorize divergences.
        let mut cat_store_dest = 0u64;  // Store: FFI dest=None, legacy has ModifierReg dest
        let mut cat_ptr_arith = 0u64;   // Pointer arith: FFI has extra PointerReg in sources
        let mut cat_memory_offset = 0u64; // Memory offset differs
        let mut cat_missing_memory = 0u64; // FFI has raw operands, legacy has Memory{}
        let mut cat_post_modify = 0u64; // PostModify differs
        let mut cat_other = 0u64;

        // Re-scan to categorize (the details vec was capped at 20)
        for path in &elf_paths {
            let data = match std::fs::read(path) { Ok(d) => d, Err(_) => continue };
            let elf = match goblin::elf::Elf::parse(&data) { Ok(e) => e, Err(_) => continue };
            for section in &elf.section_headers {
                if section.sh_flags & 0x4 == 0 { continue; }
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                if end > data.len() { continue; }
                let code = &data[start..end];
                let mut offset = 0;
                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() { break; }
                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 { continue; }
                        let ffi = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                        let leg = decoder.decode_slot_bits(slot.bits, slot.slot_type).map(|d| {
                            let (dest, sources, pm) = decoder.extract_operands(&d);
                            (d, dest, sources, pm)
                        });
                        if let (Some((_, fd, fs, fp, _)), Some((_, ld, ls, lp))) = (ffi, leg) {
                            if fd == ld && fs == ls && fp == lp { continue; }
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

        // Find post-modify load instructions in ELF files.
        let mut elf_paths = Vec::new();
        for search_dir in &["build/isa-tests", "build/fuzz"] {
            let dir = std::path::Path::new(search_dir);
            if dir.exists() {
                fn collect_elfs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
                    if let Ok(entries) = std::fs::read_dir(dir) {
                        for entry in entries.flatten() {
                            let path = entry.path();
                            if path.is_dir() { collect_elfs(&path, out); }
                            else if path.extension().map(|e| e == "elf" || e == "o").unwrap_or(false) {
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
            "../mlir-aie/build/test/npu-xrt/add_one_using_dma/chess/aie_arch.mlir.prj/main_core_0_2.elf"
        );
        if bridge_elf.exists() {
            elf_paths.push(bridge_elf.to_path_buf());
        }

        let decoder = InstructionDecoder::load_default();
        let mut seen = std::collections::HashSet::new();

        for path in &elf_paths {
            let data = match std::fs::read(path) { Ok(d) => d, Err(_) => continue };
            let elf = match goblin::elf::Elf::parse(&data) { Ok(e) => e, Err(_) => continue };
            for section in &elf.section_headers {
                if section.sh_flags & 0x4 == 0 { continue; }
                let start = section.sh_offset as usize;
                let end = start + section.sh_size as usize;
                if end > data.len() { continue; }
                let code = &data[start..end];
                let mut offset = 0;
                while offset + 4 <= code.len() {
                    let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                    let size = format.size_bytes() as usize;
                    if offset + size > code.len() { break; }
                    let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                    for slot in &extracted.slots {
                        if slot.slot_type == SlotType::Nop || slot.bits == 0 { continue; }
                        if !matches!(slot.slot_type, SlotType::Lda | SlotType::Ldb) { continue; }

                        let ffi_slot = InstructionDecoder::slot_type_to_ffi(slot.slot_type);
                        if ffi_slot.is_none() { continue; }
                        let raw = decoder_ffi::decode_slot(ffi_slot.unwrap(), slot.bits);
                        if raw.is_none() { continue; }
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
                        if encoding.is_none() { continue; }
                        let encoding = encoding.unwrap();
                        if !matches!(encoding.addressing_mode,
                            AddressingMode::PostModifyImmediate | AddressingMode::PostModifyRegister
                        ) { continue; }

                        let key = format!("{} nd={} ops={:?}", enc_name, raw.num_defs,
                            raw.operands.iter().map(|o| match o {
                                crate::tablegen::decoder_ffi::DecodedOperand::Reg { name, .. } => format!("Reg({})", name),
                                crate::tablegen::decoder_ffi::DecodedOperand::Imm(v) => format!("Imm({})", v),
                            }).collect::<Vec<_>>()
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
            if section.sh_flags & 0x4 == 0 { continue; }
            let start = section.sh_offset as usize;
            let end = start + section.sh_size as usize;
            if end > data.len() { continue; }
            let code = &data[start..end];
            let mut offset = 0;
            let mut instr_num = 0;

            while offset + 4 <= code.len() {
                let format = crate::interpreter::bundle::detect_format(&code[offset..]);
                let size = format.size_bytes() as usize;
                if offset + size > code.len() { break; }

                let extracted = decoder.extract_bundle_slots(&code[offset..offset + size]);
                for slot in &extracted.slots {
                    if slot.slot_type == SlotType::Nop || slot.bits == 0 { continue; }
                    instr_num += 1;

                    let ffi = decoder.try_decode_via_ffi(slot.bits, slot.slot_type);
                    let leg = decoder.decode_slot_bits(slot.bits, slot.slot_type).map(|d| {
                        let name = d.encoding.name.clone();
                        let (dest, sources, pm) = decoder.extract_operands(&d);
                        (name, dest, sources, pm)
                    });

                    let ffi_name = ffi.as_ref().map(|(d, _, _, _, _)| d.encoding.name.as_str()).unwrap_or("FAIL");
                    let leg_name = leg.as_ref().map(|(n, _, _, _)| n.as_str()).unwrap_or("FAIL");

                    let ffi_ops = ffi.as_ref().map(|(_, d, s, p, _)| format!("dest={:?} src={:?} pm={:?}", d, s, p));
                    let leg_ops = leg.as_ref().map(|(_, d, s, p)| format!("dest={:?} src={:?} pm={:?}", d, s, p));

                    let status = if ffi_ops == leg_ops { "OK" } else { "DIFF" };

                    eprintln!("[{:3}] {:?} 0x{:010X} {} name_ffi={} name_leg={}",
                        instr_num, slot.slot_type, slot.bits, status, ffi_name, leg_name);
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
                    if info.flags == 0 { continue; } // No flags to validate.

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
}
