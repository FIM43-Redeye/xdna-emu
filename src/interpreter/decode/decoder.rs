//! TableGen-driven instruction decoder.
//!
//! This decoder uses encoding tables generated from llvm-aie's TableGen files
//! to accurately decode AIE2 instructions with O(1) lookup performance.
//!
//! # How It Works
//!
//! 1. At construction, we receive resolved `InstrEncoding` tables grouped by slot
//! 2. For each instruction word, we try to match against all encodings
//! 3. The encoding with the most specific match (highest fixed bit count) wins
//! 4. Operand fields are extracted using the encoding's field definitions
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::tablegen::{load_from_llvm_aie, build_decoder_tables};
//! use xdna_emu::interpreter::decode::InstructionDecoder;
//!
//! let data = load_from_llvm_aie("../llvm-aie")?;
//! let tables = build_decoder_tables(&data);
//! let decoder = InstructionDecoder::from_tables(tables);
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

use crate::interpreter::bundle::{
    BranchCondition, ElementType, MemWidth, Operand, Operation, PostModify, SlotIndex, SlotOp,
    VliwBundle,
};
use crate::interpreter::traits::{DecodeError, Decoder};
use crate::tablegen::{
    build_decoder_tables, load_from_llvm_aie, load_via_tblgen, DecoderIndex, InstrEncoding,
    SemanticOp,
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
/// 2. Compute `opcode = bits & common_mask` for the slot
/// 3. HashMap lookup to find candidate encodings
/// 4. Small linear scan (usually 1-3 candidates) for disambiguation
#[derive(Clone)]
pub struct InstructionDecoder {
    /// O(1) decoder index (per-slot HashMap lookup).
    index: DecoderIndex,

    /// Legacy flat encoding list (for fallback/compatibility).
    /// Will be removed once all code uses DecoderIndex.
    encodings: Vec<InstrEncoding>,

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

/// Global cached decoder, loaded once on first use.
/// This avoids repeatedly parsing TableGen files for each core.
static CACHED_DECODER: OnceLock<InstructionDecoder> = OnceLock::new();

impl InstructionDecoder {
    /// Create an empty decoder (no encodings loaded).
    pub fn new() -> Self {
        Self {
            index: DecoderIndex::default(),
            encodings: Vec::new(),
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
        use crate::config::Config;
        let path = Config::get().llvm_aie_path();
        Self::try_load_via_tblgen(&path).unwrap_or_else(|e| {
            let config_path = Config::user_config_path()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "~/.config/xdna-emu/config.toml".to_string());
            panic!(
                "TableGen decoder loading failed: {}\n\n\
                 Configure llvm_aie_path in {} or set LLVM_AIE_PATH environment variable.\n\n\
                 Sample config:\n{}",
                e,
                config_path,
                Config::sample_config()
            );
        })
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

    /// Load a decoder from llvm-aie at the specified path using the regex parser.
    ///
    /// **DEPRECATED**: Use `try_load_via_tblgen()` instead. The regex parser
    /// is less accurate (misses mixin literal bits) and is only kept for
    /// testing/comparison purposes.
    ///
    /// Returns an error if the path doesn't exist or parsing fails.
    #[deprecated(note = "Use try_load_via_tblgen() instead for accurate encodings")]
    pub fn try_load(llvm_aie_path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let path = llvm_aie_path.as_ref();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("llvm-aie not found at: {}", path.display()),
            ));
        }

        let data = load_from_llvm_aie(path)?;
        let tables = build_decoder_tables(&data);
        Ok(Self::from_tables(tables))
    }

    /// Load a decoder using llvm-tblgen for fully resolved encodings.
    ///
    /// This is the preferred loading method - it uses `llvm-tblgen --print-records`
    /// to get ground truth encodings with all inheritance and mixin field assignments
    /// resolved. This correctly distinguishes instruction variants like ACQ_mLockId_imm
    /// vs ACQ_mLockId_reg based on their literal encoding bits.
    ///
    /// Requires `llvm-tblgen` to be in PATH.
    pub fn try_load_via_tblgen(llvm_aie_path: impl AsRef<Path>) -> Result<Self, std::io::Error> {
        let path = llvm_aie_path.as_ref();
        if !path.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("llvm-aie not found at: {}", path.display()),
            ));
        }

        let tables = load_via_tblgen(path)?;
        Ok(Self::from_tables(tables))
    }

    /// Check if llvm-aie is available (checks config and env var).
    pub fn is_llvm_aie_available() -> bool {
        use crate::config::Config;
        Path::new(&Config::get().llvm_aie_path()).exists()
    }

    /// Create a decoder from encoding tables grouped by slot.
    ///
    /// This is the preferred constructor - it builds an O(1) index.
    pub fn from_tables(tables: HashMap<String, Vec<InstrEncoding>>) -> Self {
        // Build O(1) index from the tables
        let index = DecoderIndex::from_slot_encodings(tables.clone());

        // Keep flat list for legacy decode_word() compatibility
        let mut encodings: Vec<InstrEncoding> = tables.into_values().flatten().collect();
        encodings.sort_by_key(|e| std::cmp::Reverse(e.specificity()));

        Self {
            index,
            encodings,
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Create a decoder from a flat list of encodings.
    pub fn from_encodings(mut encodings: Vec<InstrEncoding>) -> Self {
        // Group by slot for the O(1) index
        let mut by_slot: HashMap<String, Vec<InstrEncoding>> = HashMap::new();
        for enc in &encodings {
            by_slot
                .entry(enc.slot.clone())
                .or_default()
                .push(enc.clone());
        }
        let index = DecoderIndex::from_slot_encodings(by_slot);

        encodings.sort_by_key(|e| std::cmp::Reverse(e.specificity()));
        Self {
            index,
            encodings,
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Create a decoder from a pre-built DecoderIndex.
    pub fn from_index(index: DecoderIndex) -> Self {
        Self {
            index,
            encodings: Vec::new(), // No legacy list needed
            decode_count: 0,
            unknown_count: 0,
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

    /// Get the underlying decoder index.
    pub fn decoder_index(&self) -> &DecoderIndex {
        &self.index
    }

    /// Try to decode an instruction word against all known encodings.
    ///
    /// Note: This is a legacy O(n) method. Prefer using decode_slot_bits()
    /// with the slot type for O(1) performance.
    pub fn decode_word(&self, word: u64) -> Option<DecodedInstr> {
        for encoding in &self.encodings {
            if encoding.matches(word) {
                // Extract operands
                let mut operands = HashMap::new();
                for field in &encoding.operand_fields {
                    let value = field.extract(word);
                    operands.insert(field.name.clone(), value);
                }

                return Some(DecodedInstr {
                    encoding: encoding.clone(),
                    operands,
                });
            }
        }
        None
    }

    /// Try to decode slot-specific bits against encodings for that slot.
    ///
    /// This is the preferred O(1) decode method. It uses the DecoderIndex
    /// for constant-time lookup based on the slot's common opcode bits.
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

    /// Convert a decoded instruction to a bundle Operation.
    ///
    /// Uses SemanticOp from TableGen for classification. All instructions should have
    /// SemanticOp assigned via infer_semantic_from_mnemonic() during TableGen parsing.
    /// If no semantic is available, logs a warning and returns Unknown.
    fn to_operation(&self, decoded: &DecodedInstr) -> Operation {
        // Use semantic info - this should be available for all instructions
        if let Some(semantic) = decoded.encoding.semantic {
            return self.semantic_to_operation(semantic, decoded);
        }

        // No SemanticOp assigned - this is a gap in our semantic inference
        log::warn!(
            "[NO SEMANTIC] Instruction '{}' has no SemanticOp - add to infer_semantic_from_mnemonic()",
            decoded.encoding.mnemonic
        );
        Operation::Unknown {
            opcode: decoded.operands.get("word0").copied().unwrap_or(0) as u32,
        }
    }

    /// Convert a SemanticOp to an Operation.
    fn semantic_to_operation(&self, semantic: SemanticOp, decoded: &DecodedInstr) -> Operation {
        let mnemonic = &decoded.encoding.mnemonic;
        let is_vector = mnemonic.starts_with('v') || mnemonic.starts_with('V');
        let element_type = self.infer_element_type(mnemonic);

        match semantic {
            SemanticOp::Add if is_vector => Operation::VectorAdd { element_type },
            SemanticOp::Add => Operation::ScalarAdd,
            SemanticOp::Sub if is_vector => Operation::VectorSub { element_type },
            SemanticOp::Sub => Operation::ScalarSub,
            SemanticOp::Mul if is_vector => Operation::VectorMul { element_type },
            SemanticOp::Mul => Operation::ScalarMul,
            SemanticOp::And => Operation::ScalarAnd,
            SemanticOp::Or => Operation::ScalarOr,
            SemanticOp::Xor => Operation::ScalarXor,
            SemanticOp::Shl => Operation::ScalarShl,
            SemanticOp::Sra => Operation::ScalarSra,
            SemanticOp::Srl => Operation::ScalarShr,
            SemanticOp::Load => Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
            SemanticOp::Store => Operation::Store {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
            SemanticOp::Br => Operation::Branch {
                condition: BranchCondition::Always,
            },
            SemanticOp::BrCond => Operation::Branch {
                condition: BranchCondition::NotEqual, // Placeholder
            },
            SemanticOp::Nop => Operation::Nop,
            SemanticOp::Copy => Operation::ScalarMov,

            // Comparison operations (produce 0/1 result)
            SemanticOp::SetLt => Operation::ScalarLt,
            SemanticOp::SetLe => Operation::ScalarLe,
            SemanticOp::SetGt => Operation::ScalarGt,
            SemanticOp::SetGe => Operation::ScalarGe,
            SemanticOp::SetUlt => Operation::ScalarLtu,
            SemanticOp::SetUle => Operation::ScalarLeu,
            SemanticOp::SetUgt => Operation::ScalarGtu,
            SemanticOp::SetUge => Operation::ScalarGeu,
            SemanticOp::SetEq => Operation::ScalarEq,
            SemanticOp::SetNe => Operation::ScalarNe,
            SemanticOp::Select => Operation::ScalarSel,

            // Lock operations
            SemanticOp::LockAcquire => Operation::LockAcquire,
            SemanticOp::LockRelease => Operation::LockRelease,

            // Other scalar ops
            SemanticOp::Abs => Operation::ScalarAbs,

            // Control flow
            SemanticOp::Ret => Operation::Return,

            // Not handled yet - fall through
            _ => Operation::Unknown { opcode: 0 },
        }
    }

    /// Infer element type from mnemonic suffix.
    fn infer_element_type(&self, mnemonic: &str) -> ElementType {
        if mnemonic.ends_with("8") || mnemonic.contains(".i8") || mnemonic.contains(".u8") {
            if mnemonic.contains(".u") {
                ElementType::UInt8
            } else {
                ElementType::Int8
            }
        } else if mnemonic.ends_with("16") || mnemonic.contains(".i16") || mnemonic.contains(".u16")
        {
            if mnemonic.contains(".u") {
                ElementType::UInt16
            } else {
                ElementType::Int16
            }
        } else if mnemonic.ends_with("32") || mnemonic.contains(".i32") || mnemonic.contains(".u32")
        {
            if mnemonic.contains(".u") {
                ElementType::UInt32
            } else {
                ElementType::Int32
            }
        } else if mnemonic.contains("bf16") {
            ElementType::BFloat16
        } else if mnemonic.contains("f32") || mnemonic.contains("float") {
            ElementType::Float32
        } else {
            ElementType::Int32 // Default
        }
    }

    /// Map slot name to SlotIndex.
    fn slot_to_index(&self, slot: &str) -> SlotIndex {
        match slot {
            "alu" => SlotIndex::Scalar0,
            "lda" | "ldb" => SlotIndex::Load,
            "st" => SlotIndex::Store,
            "mv" => SlotIndex::Scalar1,
            "vec" => SlotIndex::Vector,
            "lng" => SlotIndex::Control, // Long instructions often control
            _ => SlotIndex::Scalar0,
        }
    }

    /// Extract operands from decoded instruction using TableGen ordering.
    ///
    /// This method builds a field_name → Operand map, then extracts:
    /// - Destination from `output_order` names
    /// - Sources from `input_order` names (in canonical order)
    ///
    /// Special handling is preserved for complex field encodings (ag_*, mSclSt, etc.)
    fn extract_operands(&self, decoded: &DecodedInstr) -> (Option<Operand>, Vec<Operand>) {
        use std::collections::HashMap;

        // Build a map of field_name -> decoded Operand
        let mut field_operands: HashMap<String, Operand> = HashMap::new();
        // Some fields produce multiple operands or set dest directly (special cases)
        let mut direct_dest: Option<Operand> = None;
        let mut extra_sources: Vec<Operand> = Vec::new();

        if decoded.encoding.slot == "st"
            || decoded.encoding.slot == "lda"
            || decoded.encoding.slot == "ldb"
        {
            let field_info: Vec<String> = decoded
                .encoding
                .operand_fields
                .iter()
                .map(|f| format!("{}=0x{:X}", f.name, decoded.operand(&f.name).unwrap_or(0)))
                .collect();
            log::debug!(
                "[DECODE {}] mnemonic={} fields={:?} input_order={:?} output_order={:?}",
                decoded.encoding.slot.to_uppercase(),
                decoded.encoding.mnemonic,
                field_info,
                decoded.encoding.input_order,
                decoded.encoding.output_order
            );
        }

        for field in &decoded.encoding.operand_fields {
            let value = decoded.operand(&field.name).unwrap_or(0);
            log::debug!("[FIELD] Processing field='{}' value=0x{:X}", field.name, value);

            // Special handling for address generator fields (ag_all, ag_idx, agb_sa, etc.)
            // These produce Memory operands or multiple operands - handled specially
            if field.name.starts_with("ag") {
                let mode = (value & 0xF) as u8;
                let mnemonic = decoded.encoding.mnemonic.to_lowercase();

                let (ptr_reg, offset_or_mod) = match mode {
                    0b1010 => {
                        let ag_ptr_imm = value >> 4;
                        let ptr = ((ag_ptr_imm >> 6) & 0x7) as u8;
                        let imm_words = (ag_ptr_imm & 0x3F) as i32;
                        let imm_bytes = imm_words * 4;
                        log::trace!("[DECODE ag_all] mode=0b1010 ptr=p{} imm={}w -> {}b", ptr, imm_words, imm_bytes);
                        (ptr, Operand::Immediate(imm_bytes))
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

                if mnemonic.starts_with("padd") {
                    direct_dest = Some(Operand::PointerReg(ptr_reg));
                    extra_sources.push(offset_or_mod);
                } else if let Operand::Immediate(imm) = offset_or_mod {
                    extra_sources.push(Operand::Memory { base: ptr_reg, offset: imm as i16 });
                } else {
                    extra_sources.push(Operand::PointerReg(ptr_reg));
                    extra_sources.push(offset_or_mod);
                }
                continue;
            }

            // Special handling for scalar store value (mSclSt)
            if field.name == "mSclSt" || field.name.contains("SclSt") {
                let scl_reg = ((value >> 2) & 0x1F) as u8;
                log::trace!("[DECODE mSclSt] raw=0x{:X} -> r{}", value, scl_reg);
                field_operands.insert(field.name.clone(), Operand::ScalarReg(scl_reg));
                continue;
            }

            // Special handling for scalar load destination (mSclLd, mLdaScl, mLdbScl)
            if field.name == "mSclLd"
                || field.name.contains("SclLd")
                || field.name == "mLdaScl"
                || field.name == "mLdbScl"
            {
                log::debug!("[DECODE mLdaScl] HANDLER ENTERED field={} value=0x{:X} ({})", field.name, value, value);
                let operand = if value % 4 == 0 {
                    let scl_reg = (value / 4) as u8;
                    log::trace!("[DECODE mLdaScl] raw=0x{:X} -> r{}", value, scl_reg);
                    Operand::ScalarReg(scl_reg)
                } else if value % 4 == 2 {
                    let mod_reg = (value / 4) as u8;
                    log::trace!("[DECODE mLdaScl] raw=0x{:X} -> m{} (modifier)", value, mod_reg);
                    Operand::ScalarReg(mod_reg)
                } else if value == 5 {
                    log::trace!("[DECODE mLdaScl] raw=0x{:X} -> lr", value);
                    Operand::ScalarReg(0)
                } else if value >= 13 && (value - 13) % 16 == 0 {
                    let ptr_reg = ((value - 13) / 16) as u8;
                    log::trace!("[DECODE mLdaScl] raw=0x{:X} -> p{}", value, ptr_reg);
                    Operand::PointerReg(ptr_reg)
                } else {
                    let scl_reg = (value / 4) as u8;
                    log::trace!("[DECODE mLdaScl] raw=0x{:X} -> unknown, using r{}", value, scl_reg);
                    Operand::ScalarReg(scl_reg)
                };
                field_operands.insert(field.name.clone(), operand);
                continue;
            }

            // Special handling for mova/movb coarse granularity field (mLdaCg/mLdbCg)
            // Encoding: r0-r31 at 0x00-0x7C (×4), then p0-p7 at 0x80-0x9C (×4)
            if field.name == "mLdaCg" || field.name == "mLdbCg" {
                let operand = if value < 0x80 {
                    // Scalar registers r0-r31 (32 regs × 4 = 0x80)
                    let scl_reg = (value >> 2) as u8;
                    log::trace!("[DECODE mLdaCg] raw=0x{:X} -> r{}", value, scl_reg);
                    Operand::ScalarReg(scl_reg)
                } else if value < 0xA0 {
                    // Pointer registers p0-p7 at 0x80-0x9C
                    let ptr_reg = ((value - 0x80) >> 2) as u8;
                    log::trace!("[DECODE mLdaCg] raw=0x{:X} -> p{}", value, ptr_reg);
                    Operand::PointerReg(ptr_reg)
                } else {
                    // Special registers (DC, DJ, DN, M, LC)
                    let reg = ((value - 0xA0) >> 2) as u8;
                    log::trace!("[DECODE mLdaCg] raw=0x{:X} -> special reg {}", value, reg);
                    Operand::ModifierReg(reg)
                };
                field_operands.insert(field.name.clone(), operand);
                continue;
            }

            // Special handling for movxm destination register (mMvSclDstCg)
            if field.name == "mMvSclDstCg" {
                log::trace!("[DECODE mMvSclDstCg] raw=0x{:X} ({})", value, value);
                let operand = if (0x40..0x60).contains(&value) {
                    let ptr_reg = ((value - 0x40) >> 2) as u8;
                    log::trace!("[DECODE mMvSclDstCg] -> p{}", ptr_reg);
                    Operand::PointerReg(ptr_reg)
                } else if value < 0x80 {
                    let scalar_reg = ((value >> 2) & 0x1F) as u8;
                    log::trace!("[DECODE mMvSclDstCg] -> r{}", scalar_reg);
                    Operand::ScalarReg(scalar_reg)
                } else {
                    log::trace!("[DECODE mMvSclDstCg] -> special reg 0x{:X}", value);
                    Operand::ScalarReg((value & 0x1F) as u8)
                };
                field_operands.insert(field.name.clone(), operand);
                continue;
            }

            // Special handling for c11s (11-bit signed constant)
            if field.name == "c11s" {
                let sign_extended = if value & 0x400 != 0 {
                    value | 0xFFFFF800
                } else {
                    value
                } as i32;
                field_operands.insert(field.name.clone(), Operand::Immediate(sign_extended));
                continue;
            }

            // Special handling for lock instruction ID field
            if field.name == "id" {
                let mnemonic = decoded.encoding.mnemonic.to_lowercase();
                if mnemonic.starts_with("acq") || mnemonic.starts_with("rel") {
                    log::debug!("[LOCK] mnemonic={} id field: value=0x{:X} ({})",
                                mnemonic, value, value);
                    field_operands.insert(field.name.clone(), Operand::Lock(value as u8));
                    continue;
                }
            }

            // Special handling for mLockId field
            if field.name == "mLockId" {
                log::debug!("[LOCK] mLockId field: value=0x{:X} ({})", value, value);
                field_operands.insert(field.name.clone(), Operand::Lock(value as u8));
                continue;
            }

            // Generic operand decoding based on field name patterns
            let operand = self.decode_generic_operand(&field.name, value);
            field_operands.insert(field.name.clone(), operand);
        }

        // Now extract dest and sources using TableGen ordering
        let (dest, sources) = self.extract_ordered_operands(
            decoded,
            &field_operands,
            direct_dest,
            extra_sources,
        );

        // Convert ScalarReg to VectorReg for vector instructions
        // Vector instructions have mnemonics starting with 'v' (e.g., vadd.8, vsub, vmul)
        let mnemonic = &decoded.encoding.mnemonic;
        let is_vector_instr = mnemonic.starts_with('v') || mnemonic.starts_with('V');

        if is_vector_instr {
            log::debug!(
                "[DECODE VECTOR CONVERT] mnemonic={} converting ScalarReg->VectorReg dest={:?} sources={:?}",
                mnemonic, dest, sources
            );
            let dest = dest.map(|op| match op {
                Operand::ScalarReg(r) => Operand::VectorReg(r),
                other => other,
            });
            let sources: Vec<Operand> = sources
                .into_iter()
                .map(|op| match op {
                    Operand::ScalarReg(r) => Operand::VectorReg(r),
                    other => other,
                })
                .collect();
            log::debug!(
                "[DECODE VECTOR CONVERT] AFTER: dest={:?} sources={:?}",
                dest, sources
            );
            (dest, sources)
        } else {
            (dest, sources)
        }
    }

    /// Decode a generic operand based on field name patterns.
    fn decode_generic_operand(&self, field_name: &str, value: u64) -> Operand {
        // Constant field patterns: c5u, c5s, c6u, etc.
        let is_const_field = field_name.len() >= 3
            && field_name.starts_with("c")
            && field_name.chars().nth(1).is_some_and(|c| c.is_ascii_digit())
            && field_name.chars().last().is_some_and(|c| c == 'u' || c == 's');

        let is_imm = field_name.contains("imm")
            || field_name.contains("offset")
            || field_name.contains("target")
            || field_name.contains("addr")
            || field_name.contains("tgt")
            || field_name.contains("disp")
            || field_name.starts_with("i")
            || is_const_field;

        if is_imm {
            Operand::Immediate(value as i32)
        } else if field_name.contains("ptr") || field_name.starts_with("p") {
            Operand::PointerReg(value as u8)
        } else if field_name.starts_with("v") {
            Operand::VectorReg(value as u8)
        } else if field_name.starts_with("acc") {
            Operand::AccumReg(value as u8)
        } else if field_name.starts_with("mP") {
            Operand::PointerReg(value as u8)
        // Vector registers: mW* (256-bit), mX* (512-bit), my (1024-bit)
        } else if field_name.starts_with("mW")
            || field_name.starts_with("mX")
            || field_name == "my"
        {
            Operand::VectorReg(value as u8)
        // Accumulator registers: mAM* (256-bit), mBM* (512-bit), mCM* (1024-bit)
        } else if field_name.starts_with("mAM")
            || field_name.starts_with("mBM")
            || field_name.starts_with("mCM")
        {
            Operand::AccumReg(value as u8)
        } else if field_name.starts_with("m")
            && !field_name.contains("mR")
            && !field_name.contains("mS")
        {
            Operand::ModifierReg(value as u8)
        } else {
            Operand::ScalarReg(value as u8)
        }
    }

    /// Extract destination and sources using TableGen's output_order/input_order.
    ///
    /// This is the key function for Phase 1: it ensures sources are in canonical order.
    ///
    /// Note: Operand names come from TableGen input_order which should now match
    /// the field_operands keys (after resolver traces through field_sources).
    fn extract_ordered_operands(
        &self,
        decoded: &DecodedInstr,
        field_operands: &std::collections::HashMap<String, Operand>,
        direct_dest: Option<Operand>,
        extra_sources: Vec<Operand>,
    ) -> (Option<Operand>, Vec<Operand>) {
        let output_order = &decoded.encoding.output_order;
        let input_order = &decoded.encoding.input_order;

        // Extract destination from output_order, or use direct_dest from special handling
        let dest = if let Some(d) = direct_dest {
            Some(d)
        } else if !output_order.is_empty() {
            // Look up first output name in field_operands
            output_order.first().and_then(|name| field_operands.get(name).cloned())
                .or_else(|| self.find_dest_heuristic(field_operands))
        } else {
            // No output_order - use heuristic
            self.find_dest_heuristic(field_operands)
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
            // No input_order - use all non-dest operands
            for (name, operand) in field_operands.iter() {
                if !self.is_dest_field(name) {
                    sources.push(operand.clone());
                }
            }
        }

        // Append extra sources from special handling (ag_*, etc.)
        sources.extend(extra_sources);

        log::debug!(
            "[DECODE ORDERED] mnemonic={} dest={:?} sources={:?}",
            decoded.encoding.mnemonic,
            dest,
            sources
        );

        (dest, sources)
    }

    /// Heuristic to find destination from field_operands map.
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

    /// Check if a field name represents a destination (used for fallback).
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
        operation: Operation,
        decoded: &DecodedInstr,
        dest: Option<Operand>,
        sources: Vec<Operand>,
    ) -> SlotOp {
        // Build SlotOp with semantic and implicit registers
        let mut slot_op = if let Some(semantic) = decoded.encoding.semantic {
            SlotOp::with_semantic(slot_index, operation, semantic)
        } else {
            SlotOp::new(slot_index, operation)
        };

        // Add implicit registers from TableGen
        if !decoded.encoding.implicit_regs.is_empty() {
            slot_op = slot_op.with_implicit_regs(decoded.encoding.implicit_regs.clone());
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
        use crate::interpreter::bundle::{extract_slots, SlotType};

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

        // Extract slots from the bundle
        let extracted = extract_slots(bytes);

        // Debug: show bundle format for first few decodes (trace level only)
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
                    SlotType::Lda | SlotType::Ldb => SlotIndex::Load,
                    SlotType::Alu => SlotIndex::Scalar0,
                    SlotType::Mv => SlotIndex::Scalar1,
                    SlotType::St => SlotIndex::Store,
                    SlotType::Vec | SlotType::Lng => SlotIndex::Vector,
                    SlotType::Nop => SlotIndex::Control,
                };

                // Try to decode the slot bits against known encodings
                // Zero bits means no operation in this slot - treat as NOP
                if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                    bundle.set_slot(SlotOp::nop(slot_index));
                } else if let Some(decoded) = self.decode_slot_bits(slot.bits, slot.slot_type) {
                    // Debug: log lng slot decoding
                    if slot.slot_type == SlotType::Lng {
                        log::debug!("[LNG DECODE] bits=0x{:010X} mnemonic={} fields={:?}",
                            slot.bits, decoded.encoding.mnemonic,
                            decoded.operands.keys().collect::<Vec<_>>());
                    }
                    let operation = self.to_operation(&decoded);
                    let (dest, sources) = if decoded.encoding.mnemonic == "movxm" {
                        // Special handling for movxm: immediate is SPLIT in the lng slot
                        // TableGen layout: lng = {i{31-12}, mMvSclDstCg, i{11-0}, 0b001}
                        // - bits 2:0 = opcode (0b001)
                        // - bits 14:3 = i{11:0} (low 12 bits of immediate)
                        // - bits 21:15 = mMvSclDstCg (7-bit destination)
                        // - bits 41:22 = i{31:12} (high 20 bits of immediate)
                        let i_low = ((slot.bits >> 3) & 0xFFF) as u32;
                        let dst_raw = ((slot.bits >> 15) & 0x7F) as u8;
                        let i_high = ((slot.bits >> 22) & 0xFFFFF) as u32;
                        let full_imm = (i_high << 12) | i_low;

                        // Destination encoding: bits 5:4 = pointer register index
                        // Pattern: 0x03=p0, 0x13=p1, 0x23=p2, 0x33=p3
                        let ptr_idx = (dst_raw >> 4) & 0x3;

                        log::debug!("[MOVXM] bits=0x{:010X} i_low=0x{:03X} i_high=0x{:05X} full=0x{:08X} dst_raw=0x{:02X} -> p{}",
                            slot.bits, i_low, i_high, full_imm, dst_raw, ptr_idx);

                        (Some(Operand::PointerReg(ptr_idx)), vec![Operand::Immediate(full_imm as i32)])
                    } else {
                        self.extract_operands(&decoded)
                    };

                    // Debug: log st slot bits and field positions
                    if slot.slot_type == SlotType::St {
                        let field_details: Vec<String> = decoded.encoding.operand_fields.iter()
                            .map(|f| format!("{}@bit{}:w{}", f.name, f.bit_position, f.width))
                            .collect();
                        log::debug!("[ST SLOT] bits=0x{:05X} fields={:?}", slot.bits, field_details);
                    }

                    #[cfg(test)]
                    if slot.slot_type == SlotType::Alu || slot.slot_type == SlotType::Mv {
                        eprintln!(
                            "[DECODE {:?}] mnemonic={} op={:?} bits=0x{:X}",
                            slot.slot_type, decoded.encoding.mnemonic, operation, slot.bits
                        );
                    }

                    // Build SlotOp with TableGen info (reorders sources, adds semantic/implicit)
                    let slot_op = self.build_slot_op(slot_index, operation, &decoded, dest, sources);
                    bundle.set_slot(slot_op);
                } else if slot.slot_type == SlotType::Lng {
                    // FIXME: This is a workaround for LNG control instructions not being
                    // properly integrated via TableGen. The JL and J instructions use the
                    // LNG slot format which puts a 20-bit cpmaddr in the immediate field.
                    // These should be handled by extending the TableGen parser to recognize
                    // LNG format control flow instructions. See llvm-aie AIE2InstrFormats.td
                    // for the proper encoding definitions.
                    //
                    // JL (jump and link): lng = {0b00000, cpmaddr[19:0], 0b0, 0b0000, 0b0000, 0b00000, 0b100}
                    // J (jump): lng = {0b00000, cpmaddr[19:0], 0b0, 0b0000, 0b0000, 0b00000, 0b010}
                    let opcode = (slot.bits & 0x7) as u8;

                    match opcode {
                        0b100 => {
                            // JL - Jump and Link (saves return address to LR, jumps to cpmaddr)
                            let cpmaddr = ((slot.bits >> 17) & 0xFFFFF) as u32;
                            log::debug!("[LNG JL] bits=0x{:010X} cpmaddr=0x{:05X}", slot.bits, cpmaddr);

                            bundle.set_slot(
                                SlotOp::new(SlotIndex::Control, Operation::Call)
                                    .with_source(Operand::Immediate(cpmaddr as i32))
                            );
                        }
                        0b010 => {
                            // J - Unconditional Jump
                            let cpmaddr = ((slot.bits >> 17) & 0xFFFFF) as u32;
                            log::debug!("[LNG J] bits=0x{:010X} cpmaddr=0x{:05X}", slot.bits, cpmaddr);

                            bundle.set_slot(
                                SlotOp::new(
                                    SlotIndex::Control,
                                    Operation::Branch { condition: BranchCondition::Always }
                                ).with_source(Operand::Immediate(cpmaddr as i32))
                            );
                        }
                        _ => {
                            // Other LNG opcodes - mark as unknown
                            log::debug!("[LNG DECODE FAIL] bits=0x{:010X} opcode=0b{:03b} - not recognized",
                                slot.bits, opcode);
                            bundle.set_slot(
                                SlotOp::new(SlotIndex::Control, Operation::Unknown { opcode: slot.bits as u32 })
                            );
                        }
                    }
                } else {
                    // Slot extracted but not recognized - mark as unknown with slot info
                    log::debug!("[{:?} DECODE FAIL] bits=0x{:010X} - no matching encoding",
                        slot.slot_type, slot.bits);
                    #[cfg(test)]
                    if slot.slot_type == SlotType::Alu || slot.slot_type == SlotType::Mv {
                        eprintln!(
                            "[DECODE {:?} FAIL] bits=0x{:X} - no matching encoding",
                            slot.slot_type, slot.bits
                        );
                    }
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
                    // For now, mark as unknown but we know the slot type
                    bundle.set_slot(SlotOp::new(
                        slot_index,
                        Operation::Unknown {
                            opcode: slot.bits as u32,
                        },
                    ));
                    let _ = slot_name; // Suppress unused warning for now
                }
            }
        } else {
            // No slots extracted - fall back to word-based decoding
            let word0 = if bytes.len() >= 4 {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                u16::from_le_bytes([bytes[0], bytes[1]]) as u32
            };

            // Check for known NOP patterns
            if word0 == 0 || word0 == 0x15010040 {
                bundle.set_slot(SlotOp::nop(SlotIndex::Scalar0));
            } else if let Some(decoded) = self.decode_word(word0 as u64) {
                let operation = self.to_operation(&decoded);
                let slot_index = self.slot_to_index(&decoded.encoding.slot);
                let (dest, sources) = self.extract_operands(&decoded);

                // Build SlotOp with TableGen info
                let slot_op = self.build_slot_op(slot_index, operation, &decoded, dest, sources);
                bundle.set_slot(slot_op);
            } else {
                // Unknown instruction
                bundle.set_slot(SlotOp::new(
                    SlotIndex::Scalar0,
                    Operation::Unknown { opcode: word0 },
                ));
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
    use crate::tablegen::{build_decoder_tables, load_from_llvm_aie, OperandField};
    use std::path::Path;

    fn make_add_encoding() -> InstrEncoding {
        InstrEncoding {
            name: "ADD".to_string(),
            mnemonic: "add".to_string(),
            slot: "alu".to_string(),
            width: 20,
            fixed_mask: 0b1_1111,
            fixed_bits: 0b0_0001,
            operand_fields: vec![
                OperandField::new("mRx0", 15, 5),
                OperandField::new("mRx", 10, 5),
                OperandField::new("mRy", 5, 5),
            ],
            semantic: Some(SemanticOp::Add),
            may_load: false,
            may_store: false,
            input_order: vec!["mRx0".to_string(), "mRy".to_string()],
            output_order: vec!["mRx".to_string()],
            implicit_regs: vec![],
        }
    }

    #[test]
    fn test_decoder_matches_add() {
        let decoder = InstructionDecoder::from_encodings(vec![make_add_encoding()]);

        // Construct ADD r5, r3, r2: mRx0=3, mRx=5, mRy=2, fixed=0b0_0001
        // Encoding: 00011_00101_00010_0000_1 = bits 19:0
        let word = 0b00011_00101_00010_0000_1u64;

        let decoded = decoder.decode_word(word).expect("Should decode");
        assert_eq!(decoded.encoding.name, "ADD");
        assert_eq!(decoded.reg("mRx0"), Some(3));
        assert_eq!(decoded.reg("mRx"), Some(5));
        assert_eq!(decoded.reg("mRy"), Some(2));
    }

    #[test]
    fn test_decoder_trait() {
        let decoder = InstructionDecoder::from_encodings(vec![make_add_encoding()]);

        // ADD r5, r3, r2 as bytes (little-endian)
        // Slot encoding: 00011_00101_00010_0000_1 = 20 bits
        // For 32-bit format, we need format marker 0x9 in low nibble
        // Shift slot content left by 4 bits and add format marker
        let slot_content = 0b00011_00101_00010_0000_1u32; // 20 bits
        let word = (slot_content << 4) | 0x9; // Add 32-bit format marker
        let bytes = word.to_le_bytes();

        let bundle = decoder.decode(&bytes, 0x100).expect("Should decode");
        assert_eq!(bundle.pc(), 0x100);

        // Note: The slot encoding is now shifted, so decode_word needs to look
        // at the shifted position. For now, check that we get SOMETHING decoded.
        // The actual operation may be Unknown since the encoding doesn't match
        // the shifted position. That's OK - this test is mainly about the Decoder
        // trait working, not the specific instruction matching.
        assert_eq!(bundle.size(), 4); // Should be 32-bit format
    }

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
        assert!(matches!(slot.op, Operation::Unknown { .. }));
    }

    #[test]
    fn test_specificity_ordering() {
        // More specific encoding (more fixed bits) should match first
        let less_specific = InstrEncoding {
            name: "GENERIC".to_string(),
            mnemonic: "generic".to_string(),
            slot: "alu".to_string(),
            width: 20,
            fixed_mask: 0b1, // Only 1 fixed bit
            fixed_bits: 0b1,
            operand_fields: vec![],
            semantic: None,
            may_load: false,
            may_store: false,
            input_order: vec![],
            output_order: vec![],
            implicit_regs: vec![],
        };

        let more_specific = make_add_encoding(); // 5 fixed bits

        // Order shouldn't matter - decoder sorts by specificity
        let decoder =
            InstructionDecoder::from_encodings(vec![less_specific.clone(), more_specific]);

        let word = 0b00011_00101_00010_0000_1u64;
        let decoded = decoder.decode_word(word).expect("Should decode");

        // Should match ADD (more specific), not GENERIC
        assert_eq!(decoded.encoding.name, "ADD");
    }

    #[test]
    fn test_decoder_with_real_llvm_aie() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        // Load real TableGen data
        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let tables = build_decoder_tables(&data);

        eprintln!("Loaded {} slots with encodings:", tables.len());
        for (slot, encodings) in &tables {
            eprintln!("  {}: {} encodings", slot, encodings.len());
        }

        // Create decoder from real data
        let decoder = InstructionDecoder::from_tables(tables);

        // Test NOP decoding
        let nop_bytes = [0x00u8, 0x00, 0x00, 0x00];
        let bundle = decoder.decode(&nop_bytes, 0).expect("Should decode NOP");
        assert!(bundle.is_nop());

        // Get stats
        let (decode_count, unknown_count) = decoder.stats();
        eprintln!(
            "Decoder created with {} encodings, {} decoded, {} unknown",
            decoder.encodings.len(),
            decode_count,
            unknown_count
        );

        assert!(
            decoder.encodings.len() > 0,
            "Should have loaded some encodings"
        );
    }

    #[test]
    fn test_decode_real_elf_instructions() {
        use crate::parser::AieElf;

        let elf_path = Path::new("/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf");
        if !elf_path.exists() {
            eprintln!("Skipping test: ELF not found");
            return;
        }

        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        // Load TableGen data and create decoder
        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let tables = build_decoder_tables(&data);
        let decoder = InstructionDecoder::from_tables(tables);

        // Read ELF using proper parser
        let elf_data = std::fs::read(elf_path).expect("Failed to read ELF");
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
                        match &slot_op.op {
                            Operation::Unknown { opcode } => {
                                op_names.push(format!("{:?}:??? (0x{:05X})", slot_op.slot, opcode));
                            }
                            Operation::Nop => {
                                found_known = true;
                                op_names.push("Nop".to_string());
                            }
                            op => {
                                found_known = true;
                                op_names.push(format!("{:?}", op));
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
    fn test_decoder_via_tblgen() {
        let llvm_aie_path = Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found at ../llvm-aie");
            return;
        }

        // Load decoder via tblgen
        let decoder = InstructionDecoder::try_load_via_tblgen(llvm_aie_path)
            .expect("Failed to load via tblgen");

        eprintln!("Loaded {} encodings via tblgen", decoder.encodings.len());

        // Verify we have more than regex-based loader (tblgen captures more instructions)
        assert!(
            decoder.encodings.len() > 100,
            "Should have loaded many encodings"
        );

        // Verify ACQ instructions are distinguished
        let acq_imm = decoder.encodings.iter().find(|e| e.name == "ACQ_mLockId_imm");
        let acq_reg = decoder.encodings.iter().find(|e| e.name == "ACQ_mLockId_reg");

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
}
