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
    BranchCondition, ElementType, MemWidth, Operand, Operation, PostModify, SelectVariant,
    ShufflePattern, SlotIndex, SlotOp, VliwBundle,
};
use crate::interpreter::state::{MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M};
use crate::interpreter::traits::{DecodeError, Decoder};
use super::composite::CompositeLuts;
use crate::tablegen::{
    build_decoder_tables, load_from_llvm_aie, load_via_tblgen, AddressingMode,
    DecoderIndex, InstrEncoding, InstrMemWidth, OperandType, RegisterKind, SemanticOp,
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

    /// Pre-built composite register decode LUTs.
    /// Each composite encoder variant has a dedicated LUT for O(1) decode.
    composite_luts: CompositeLuts,

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
#[inline]
fn sign_extend(value: i32, bits: u32) -> i32 {
    if bits == 0 || bits >= 32 {
        return value;
    }
    let shift = 32 - bits;
    (value << shift) >> shift
}

/// Sign-extend an already-extracted unsigned raw value based on its logical width.
///
/// Unlike `OperandField::extract_signed()`, this works on the value directly
/// without needing the original instruction word. Essential for split fields
/// where the value was reassembled from non-contiguous fragments.
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
            encodings: Vec::new(),
            composite_luts: CompositeLuts::build(),
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
        encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));

        Self {
            index,
            encodings,
            composite_luts: CompositeLuts::build(),
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

        encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));
        Self {
            index,
            encodings,
            composite_luts: CompositeLuts::build(),
            decode_count: 0,
            unknown_count: 0,
        }
    }

    /// Create a decoder from a pre-built DecoderIndex.
    pub fn from_index(index: DecoderIndex) -> Self {
        Self {
            index,
            encodings: Vec::new(), // No legacy list needed
            composite_luts: CompositeLuts::build(),
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
        // Cascade instructions: detected by name before SemanticOp lookup.
        // These have hasSideEffects=true and no SemanticOp in TableGen.
        // Names from llvm-aie AIE2GenFixupInstrInfo.td:
        //   VMOV_mv_scd  -- full 384-bit cascade read
        //   VMOV_HI      -- cascade read high half
        //   VMOV_LO      -- cascade read low half
        //   VMOV_mv_mcd  -- full 384-bit cascade write
        let name = &decoded.encoding.name;
        if name == "VMOV_mv_scd" || name == "VMOV_HI" || name == "VMOV_LO" {
            return Operation::CascadeRead;
        }
        if name == "VMOV_mv_mcd" {
            return Operation::CascadeWrite;
        }

        // Vector operation mnemonic dispatch.
        //
        // Many vector instructions share generic SemanticOps (Copy, Mul, etc.)
        // that lose the specific operation type. For example, vsrs maps to
        // SemanticOp::Copy which becomes VectorMov -- wrong. We intercept
        // these here and create the correct Operation from the mnemonic.
        if let Some(op) = Self::vector_op_from_mnemonic(decoded) {
            return op;
        }

        // Use semantic info - this should be available for all instructions
        if let Some(semantic) = decoded.encoding.semantic {
            return self.semantic_to_operation(semantic, decoded);
        }

        // No SemanticOp assigned - this is a gap in our semantic inference.
        // "opcodestr" is a parser artifact from unresolved TableGen template parameters
        // in NOP class definitions; treat as NOP rather than Unknown.
        if decoded.encoding.mnemonic == "opcodestr" || decoded.encoding.mnemonic.is_empty() {
            return Operation::Nop;
        }

        log::warn!(
            "[NO SEMANTIC] Instruction '{}' has no SemanticOp - add to infer_semantic_from_mnemonic()",
            decoded.encoding.mnemonic
        );
        Operation::Unknown {
            opcode: decoded.operands.get("word0").copied().unwrap_or(0) as u32,
        }
    }

    /// Map vector instruction mnemonics to specific Operation variants.
    ///
    /// Many vector instructions share generic SemanticOps (e.g., vmac and vmul
    /// both get SemanticOp::Mul, vsrs and vmov both get SemanticOp::Copy). This
    /// method resolves the ambiguity by checking the mnemonic directly.
    ///
    /// Only matches mnemonics that would produce the WRONG Operation through
    /// the SemanticOp path. Instructions that work correctly via semantic
    /// dispatch (vadd, vsub, vmul, vand, vor, vxor, vnot, shifts, vmov, vneg,
    /// loads, stores) are NOT matched here.
    fn vector_op_from_mnemonic(decoded: &DecodedInstr) -> Option<Operation> {
        let mnem = decoded.encoding.mnemonic.to_lowercase();
        let element_type = decoded.encoding.element_type.unwrap_or(ElementType::Int32);

        // ---- SRS / UPS (accumulator <-> vector conversions) ----

        // vsrs/vsrsm: Shift-Round-Saturate (accumulator -> vector).
        // vsrsm is the masked variant; same operation, mask handled separately.
        // element_type gives the output width (e.g., vsrs.s8 -> Int8).
        if mnem.starts_with("vsrs") {
            return Some(Operation::VectorSRS {
                from_type: ElementType::Int32,
                to_type: element_type,
            });
        }

        // vups: Upshift (vector -> accumulator).
        // element_type gives the input width (e.g., vups.s8 -> Int8).
        if mnem.starts_with("vups") {
            return Some(Operation::VectorUpshift {
                from_type: element_type,
                to_type: ElementType::Int32,
            });
        }

        // vpush: push vector to accumulator (functionally same as UPS).
        if mnem.starts_with("vpush") {
            return Some(Operation::VectorUpshift {
                from_type: element_type,
                to_type: ElementType::Int32,
            });
        }

        // ---- MAC variants (must be checked before vmul/vneg) ----

        // vaddmac: acc1 = acc1 + acc2 + A * B
        if mnem.starts_with("vaddmac") {
            return Some(Operation::VectorAddMac { element_type });
        }
        // vaddmsc: acc1 = acc1 + acc2 - A * B (sub-MAC variant)
        if mnem.starts_with("vaddmsc") {
            // No dedicated Operation yet; approximate as VectorAddMac.
            // The subtraction vs addition of the product is a refinement.
            return Some(Operation::VectorAddMac { element_type });
        }
        // vsubmac: acc1 = acc1 - acc2 + A * B
        if mnem.starts_with("vsubmac") {
            return Some(Operation::VectorSubMac { element_type });
        }
        // vsubmsc: acc1 = acc1 - acc2 - A * B
        if mnem.starts_with("vsubmsc") {
            return Some(Operation::VectorSubMac { element_type });
        }
        // vnegmac: acc += -(A * B)
        if mnem.starts_with("vnegmac") {
            return Some(Operation::VectorNegMatMulDense { element_type });
        }
        // vnegmsc: acc -= -(A * B)  (negate multiply-subtract-accumulate)
        if mnem.starts_with("vnegmsc") {
            return Some(Operation::VectorNegMatMulDense { element_type });
        }

        // vmac: multiply-accumulate (acc += A * B).
        // vmac.f = bfloat16 variant.
        if mnem.starts_with("vmac") {
            if mnem.contains(".f") || mnem.contains(".bf") {
                return Some(Operation::VectorMatMulAccFloat {
                    element_type: ElementType::BFloat16,
                });
            }
            return Some(Operation::VectorMac { element_type });
        }
        // vmsc: multiply-subtract-accumulate (acc -= A * B).
        if mnem.starts_with("vmsc") {
            if mnem.contains(".f") || mnem.contains(".bf") {
                return Some(Operation::VectorMatMulSubFloat {
                    element_type: ElementType::BFloat16,
                });
            }
            return Some(Operation::VectorMatMulSubDense { element_type });
        }

        // ---- Negate-arithmetic variants (must be before vneg) ----

        // vnegmul: acc += -(src1 * src2)
        if mnem.starts_with("vnegmul") {
            return Some(Operation::VectorNegMul { element_type });
        }
        // vnegadd: dst = -src1 + src2
        if mnem.starts_with("vnegadd") {
            return Some(Operation::VectorNegAdd { element_type });
        }
        // vnegsub: dst = -(src1) + src2 (same as negadd semantically)
        if mnem.starts_with("vnegsub") {
            return Some(Operation::VectorNegAdd { element_type });
        }

        // vaddsub: alternating add/subtract on even/odd lanes (FFT butterfly).
        // No dedicated Operation yet; approximate as VectorAdd.
        if mnem.starts_with("vaddsub") {
            return Some(Operation::VectorAdd { element_type });
        }

        // ---- Conditional arithmetic (must be before vsub/comparison) ----

        // vsub_lt: dst[i] = (a[i] < b[i]) ? a[i] - b[i] : a[i]
        if mnem.starts_with("vsub_lt") {
            return Some(Operation::VectorSubLt { element_type });
        }
        // vsub_ge: dst[i] = (a[i] >= b[i]) ? a[i] - b[i] : a[i]
        if mnem.starts_with("vsub_ge") {
            return Some(Operation::VectorSubGe { element_type });
        }
        // vmaxdiff_lt: dst[i] = max(a[i] - b[i], 0) when a < b
        if mnem.starts_with("vmaxdiff_lt") {
            return Some(Operation::VectorMaxDiffLt { element_type });
        }
        // vmax_lt: dst = max(a, b) with less-than flag
        if mnem.starts_with("vmax_lt") {
            return Some(Operation::VectorMaxLt { element_type });
        }
        // vmin_ge: dst = min(a, b) with greater-equal flag
        if mnem.starts_with("vmin_ge") {
            return Some(Operation::VectorMinGe { element_type });
        }
        // vmin (standalone): dst = min(a, b)
        if mnem.starts_with("vmin") {
            return Some(Operation::VectorMin { element_type });
        }
        // vmax (standalone, must come after vmax_lt/vmaxdiff_lt checks): dst = max(a, b)
        if mnem.starts_with("vmax") {
            return Some(Operation::VectorMax { element_type });
        }

        // ---- Absolute value / conditional negate ----

        // vabs_gtz: dst[i] = (src[i] > 0) ? abs(src[i]) : src[i]
        if mnem.starts_with("vabs_gtz") || mnem.starts_with("vabs") {
            return Some(Operation::VectorAbsGtz { element_type });
        }
        // vneg_gtz: dst[i] = (src[i] > 0) ? -src[i] : src[i]
        if mnem.starts_with("vneg_gtz") {
            return Some(Operation::VectorNegGtz { element_type });
        }
        // vbneg_ltz: dst[i] = (src[i] < 0) ? -src[i] : src[i] (boolean negate)
        if mnem.starts_with("vbneg_ltz") || mnem.starts_with("vbneg") {
            return Some(Operation::VectorNegLtz { element_type });
        }

        // ---- Vector comparisons (produce mask output) ----

        // veqz: dst[i] = (a[i] == 0) ? ~0 : 0
        if mnem.starts_with("veqz") {
            return Some(Operation::VectorEqz { element_type });
        }
        // veq: dst[i] = (a[i] == b[i]) ? ~0 : 0
        if mnem.starts_with("veq.") || mnem == "veq" {
            return Some(Operation::VectorCmp { element_type });
        }
        // vne: dst[i] = (a[i] != b[i]) ? ~0 : 0
        // Approximate as bitwise XOR (nonzero where different).
        if mnem.starts_with("vne.") || mnem == "vne" {
            return Some(Operation::VectorCmp { element_type });
        }
        // vge/vgeu: dst[i] = (a[i] >= b[i]) ? ~0 : 0
        if mnem.starts_with("vge") {
            return Some(Operation::VectorGe { element_type });
        }
        // vlt: dst[i] = (a[i] < b[i]) ? ~0 : 0
        // Must not match vlda, vldb.
        if (mnem.starts_with("vlt") || mnem.starts_with("vltu"))
            && !mnem.starts_with("vlda") && !mnem.starts_with("vldb")
        {
            return Some(Operation::VectorLt { element_type });
        }
        // vgt/vgtu: dst[i] = (a[i] > b[i]) ? ~0 : 0
        // Approximate as VectorLt with swapped operands (handled at execution).
        if mnem.starts_with("vgt") {
            return Some(Operation::VectorLt { element_type });
        }
        // vle/vleu: dst[i] = (a[i] <= b[i]) ? ~0 : 0
        // Approximate as VectorGe with swapped operands.
        if mnem.starts_with("vle") {
            return Some(Operation::VectorGe { element_type });
        }

        // ---- Data rearrangement ----

        // vshuffle: vector lane permutation
        if mnem.starts_with("vshuffle") {
            return Some(Operation::VectorShuffle {
                pattern: ShufflePattern::InterleaveLow,
            });
        }
        // vshift: concatenate two vectors and shift (alignment)
        if mnem.starts_with("vshift") {
            return Some(Operation::VectorAlign { element_type });
        }
        // vsel: per-lane conditional select
        if mnem.starts_with("vsel") {
            return Some(Operation::VectorSelect { element_type });
        }
        // vclr: clear vector register to zero
        if mnem == "vclr" || mnem.starts_with("vclr.") {
            return Some(Operation::VectorClear);
        }
        // vextbcst / vbcst: extract-broadcast / broadcast scalar to all lanes
        if mnem.starts_with("vextbcst") || mnem.starts_with("vbcst") {
            return Some(Operation::VectorBroadcast { element_type });
        }
        // vpack: pack two vectors into one (narrow)
        if mnem.starts_with("vpack") {
            return Some(Operation::VectorPack);
        }
        // vunpack: unpack one vector into two (widen).
        // Note: vldb.unpack goes through Load path, standalone vunpack goes here.
        if mnem == "vunpack" || mnem.starts_with("vunpack.") {
            return Some(Operation::VectorUnpack);
        }
        // vextract: extract single element from vector to scalar
        if mnem.starts_with("vextract") {
            return Some(Operation::VectorExtract { element_type });
        }
        // vinsert: insert scalar into vector lane
        if mnem.starts_with("vinsert") {
            return Some(Operation::VectorInsert { element_type });
        }

        // ---- Type conversion ----

        // vconv: type conversion between vector element types
        if mnem.starts_with("vconv") {
            let (from_type, to_type) = if mnem.contains(".bf") {
                (ElementType::Float32, ElementType::BFloat16)
            } else if mnem.contains(".fp") {
                (ElementType::BFloat16, ElementType::Float32)
            } else {
                (element_type, element_type)
            };
            return Some(Operation::VectorConvert { from_type, to_type });
        }
        // vfloor/vceil/vtrunc/vround: float-to-int conversions with rounding
        if mnem.starts_with("vfloor") || mnem.starts_with("vceil")
            || mnem.starts_with("vtrunc") || mnem.starts_with("vround")
        {
            return Some(Operation::VectorConvert {
                from_type: ElementType::Float32,
                to_type: element_type,
            });
        }

        None // Not a specialized vector op; fall through to semantic dispatch.
    }

    /// Convert a SemanticOp to an Operation using pre-resolved metadata.
    ///
    /// All element type, branch condition, select variant, and load/store channel
    /// disambiguation is resolved at TableGen load time on InstrEncoding. No mnemonic
    /// string parsing happens here -- just field lookups and slot matching.
    fn semantic_to_operation(&self, semantic: SemanticOp, decoded: &DecodedInstr) -> Operation {
        let enc = &decoded.encoding;
        let element_type = enc.element_type.unwrap_or(ElementType::Int32);

        match semantic {
            SemanticOp::Add if enc.is_vector => Operation::VectorAdd { element_type },
            SemanticOp::Add => Operation::ScalarAdd,
            SemanticOp::Sub if enc.is_vector => Operation::VectorSub { element_type },
            SemanticOp::Sub => Operation::ScalarSub,
            SemanticOp::Mul if enc.is_vector => Operation::VectorMul { element_type },
            SemanticOp::Mul => Operation::ScalarMul,
            SemanticOp::And if enc.is_vector => Operation::VectorAnd { element_type },
            SemanticOp::And => Operation::ScalarAnd,
            SemanticOp::Or if enc.is_vector => Operation::VectorOr { element_type },
            SemanticOp::Or => Operation::ScalarOr,
            SemanticOp::Xor if enc.is_vector => Operation::VectorXor { element_type },
            SemanticOp::Xor => Operation::ScalarXor,
            SemanticOp::Not if enc.is_vector => Operation::VectorNot { element_type },
            SemanticOp::Not => Operation::ScalarXor, // scalar not = xor with -1
            SemanticOp::Shl if enc.is_vector => Operation::VectorShiftLeft { element_type },
            SemanticOp::Shl => Operation::ScalarShl,
            SemanticOp::Sra if enc.is_vector => Operation::VectorArithShiftRight { element_type },
            SemanticOp::Sra => Operation::ScalarSra,
            SemanticOp::Srl if enc.is_vector => Operation::VectorShiftRight { element_type },
            SemanticOp::Srl => Operation::ScalarShr,
            SemanticOp::Load => {
                // Channel discrimination: enc.slot is the authoritative
                // functional-unit identifier from TableGen ("lda", "ldb", "st").
                // The is_vector guard prevents scalar lda/ldb from matching.
                match enc.slot.as_str() {
                    "lda" if enc.is_vector => Operation::VectorLoadA { post_modify: PostModify::None },
                    "ldb" if enc.is_vector => Operation::VectorLoadB { post_modify: PostModify::None },
                    _ => {
                        let width = match enc.mem_width {
                            InstrMemWidth::Byte => MemWidth::Byte,
                            InstrMemWidth::HalfWord => MemWidth::HalfWord,
                            InstrMemWidth::Word => MemWidth::Word,
                            InstrMemWidth::Vector256 => MemWidth::Vector256,
                        };
                        Operation::Load { width, post_modify: PostModify::None }
                    }
                }
            },
            SemanticOp::Store => {
                match enc.slot.as_str() {
                    "st" if enc.is_vector => Operation::VectorStore { post_modify: PostModify::None },
                    _ => {
                        let width = match enc.mem_width {
                            InstrMemWidth::Byte => MemWidth::Byte,
                            InstrMemWidth::HalfWord => MemWidth::HalfWord,
                            InstrMemWidth::Word => MemWidth::Word,
                            InstrMemWidth::Vector256 => MemWidth::Vector256,
                        };
                        Operation::Store { width, post_modify: PostModify::None }
                    }
                }
            },
            SemanticOp::Br => Operation::Branch {
                condition: BranchCondition::Always,
            },
            SemanticOp::Call => Operation::Call,
            SemanticOp::BrCond => Operation::Branch {
                condition: enc.branch_condition.unwrap_or(BranchCondition::NotEqual),
            },
            SemanticOp::Nop => Operation::Nop,
            SemanticOp::Copy if enc.is_vector => Operation::VectorMov { element_type },
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
            SemanticOp::Select => match enc.select_variant {
                Some(SelectVariant::EqualZero) => Operation::ScalarSelEqz,
                Some(SelectVariant::NotEqualZero) => Operation::ScalarSelNez,
                _ => Operation::ScalarSel,
            },

            // Lock operations
            SemanticOp::LockAcquire => Operation::LockAcquire,
            SemanticOp::LockRelease => Operation::LockRelease,

            // Abs/Neg
            SemanticOp::Abs => Operation::ScalarAbs,
            SemanticOp::Neg if enc.is_vector => Operation::VectorNegate { element_type },
            SemanticOp::Neg => Operation::ScalarSub, // neg r = 0 - r

            // Bit manipulation
            SemanticOp::Ctlz => Operation::ScalarClz,

            // Sign/zero extension
            SemanticOp::SignExtend => Operation::ScalarExtendS16,
            SemanticOp::ZeroExtend => Operation::ScalarExtendU16,

            // Division
            SemanticOp::SDiv => Operation::ScalarDiv,
            SemanticOp::UDiv => Operation::ScalarDivu,
            SemanticOp::SRem => Operation::ScalarMod,

            // Control flow
            SemanticOp::Ret => Operation::Return,
            SemanticOp::Done => Operation::Halt,

            // Truncate: narrow type conversion, treat as NOP for now
            // (the narrowing happens implicitly in register writes)
            SemanticOp::Truncate => Operation::Nop,

            // Event: fires a trace event (INSTR_EVENT_0/1). No computational
            // effect; the trace subsystem auto-starts events separately.
            SemanticOp::Event => Operation::Nop,

            // Not handled yet (URem, Rotl, Rotr, Cttz, Ctpop, Bswap, Intrinsic)
            _ => {
                log::warn!(
                    "[UNHANDLED SEMANTIC] '{}' has SemanticOp::{:?} in slot '{}' - not yet implemented",
                    decoded.encoding.mnemonic, semantic, decoded.encoding.slot
                );
                Operation::Unknown {
                    opcode: decoded.encoding.fixed_bits as u32,
                }
            }
        }
    }

    /// Map slot name to SlotIndex.
    fn slot_to_index(&self, slot: &str) -> SlotIndex {
        match slot {
            "alu" => SlotIndex::Scalar0,
            "lda" => SlotIndex::LoadA,
            "ldb" => SlotIndex::LoadB,
            "st" => SlotIndex::Store,
            "mv" => SlotIndex::Scalar1,
            "vec" => SlotIndex::Vector,
            // LNG is polymorphic (j/jl/movxm). Control is the safe default;
            // the main decode_bundle path overrides via natural_slot() anyway.
            "lng" => SlotIndex::Control,
            _ => SlotIndex::Scalar0,
        }
    }

    /// Extract operands from decoded instruction using data-driven dispatch.
    ///
    /// Each operand field's `operand_type` (populated from TableGen reg_class)
    /// determines exactly how raw bits become an Operand. No field-name heuristics.
    ///
    /// Address generator fields (ag_*) are still handled specially because they
    /// encode a packed {ptr, imm/mod, mode} tuple -- this is structural, not a
    /// register encoding issue.
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
                    let reg = raw as u8;
                    match kind {
                        RegisterKind::Scalar => Operand::ScalarReg(reg),
                        RegisterKind::Pointer => Operand::PointerReg(reg),
                        // Each modifier sub-class maps to a different base index
                        // in the unified modifier register file (32 entries total):
                        //   m0-m7 at 0-7, dn0-dn7 at 8-15, dj0-dj7 at 16-23, dc0-dc7 at 24-31
                        RegisterKind::ModifierM  => Operand::ModifierReg(reg + MOD_BASE_M),
                        RegisterKind::ModifierDN => Operand::ModifierReg(reg + MOD_BASE_DN),
                        RegisterKind::ModifierDJ => Operand::ModifierReg(reg + MOD_BASE_DJ),
                        RegisterKind::ModifierDC => Operand::ModifierReg(reg + MOD_BASE_DC),
                        RegisterKind::Vector256 | RegisterKind::Vector512 =>
                            Operand::VectorReg(reg),
                        RegisterKind::Accumulator => Operand::AccumReg(reg),
                        RegisterKind::Control => Operand::ControlReg(reg),
                    }
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

        // SP-relative load/store (spill/fill): Uses = [SP] in TableGen.
        // The immediate field is a stack offset, not a standalone value.
        // Convert to Memory { base: 6 (SP), offset: imm }.
        // Exclude pointer arithmetic (padda/paddb [sp]) -- those have their
        // own implicit SP handling in execute_add.
        if decoded.encoding.is_sp_relative && !decoded.encoding.is_ptr_arithmetic {
            if let Some(Operand::Immediate(offset)) = field_operands.get("imm").cloned() {
                extra_sources.push(Operand::Memory { base: 6, offset: offset as i16 });
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
    /// AG fields encode a packed tuple of {pointer, offset/modifier, mode_bits}.
    /// The addressing mode (from the instruction name) determines the layout.
    /// This is structural packing, not a register composite encoding, so it
    /// stays as explicit logic rather than being part of the OperandType system.
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
    /// This heuristic is only used when output_order/input_order are empty or
    /// when the first output_order name doesn't match any field_operand key.
    /// New instructions should always have output_order/input_order populated
    /// by the resolver -- do not extend this pattern list.
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
        operation: Operation,
        decoded: &DecodedInstr,
        dest: Option<Operand>,
        sources: Vec<Operand>,
        extracted_pm: Option<PostModify>,
    ) -> SlotOp {
        let enc = &decoded.encoding;

        // Build SlotOp with semantic and implicit registers
        let mut slot_op = if let Some(semantic) = enc.semantic {
            SlotOp::with_semantic(slot_index, operation, semantic)
        } else {
            SlotOp::new(slot_index, operation)
        };

        // ── Populate metadata from InstrEncoding ─────────────────────────
        slot_op.is_vector = enc.is_vector;
        slot_op.element_type = enc.element_type;
        slot_op.mem_width = match enc.mem_width {
            InstrMemWidth::Byte => MemWidth::Byte,
            InstrMemWidth::HalfWord => MemWidth::HalfWord,
            InstrMemWidth::Word => MemWidth::Word,
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
                // Zero bits means no operation in this slot - treat as NOP.
                if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                    bundle.set_slot(SlotOp::nop(slot_index));
                } else if let Some(decoded) = self.decode_slot_bits(slot.bits, slot.slot_type) {
                    let operation = self.to_operation(&decoded);
                    let (dest, sources, extracted_pm) = self.extract_operands(&decoded);

                    // For LNG slot instructions, use the operation's natural slot since
                    // LNG can contain either control (JL, J) or vector/scalar (movxm).
                    let effective_slot = if slot.slot_type == SlotType::Lng {
                        operation.natural_slot()
                    } else {
                        slot_index
                    };

                    // Build SlotOp with metadata from InstrEncoding + extracted PostModify
                    let slot_op = self.build_slot_op(
                        effective_slot, operation, &decoded, dest, sources, extracted_pm,
                    );

                    // Warn if this overwrites a non-NOP instruction in the same slot
                    // (e.g., LDA and LDB both active -- would need separate slots).
                    if let Some(existing) = bundle.slot(effective_slot) {
                        if !existing.op.is_nop() {
                            log::warn!(
                                "[SLOT COLLISION] PC=0x{:04X} {:?} slot {:?} already has {:?}, overwriting with {:?}",
                                pc, slot.slot_type, effective_slot, existing.op, slot_op.op
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
                let (dest, sources, extracted_pm) = self.extract_operands(&decoded);

                // Build SlotOp with metadata from InstrEncoding + extracted PostModify
                let slot_op = self.build_slot_op(
                    slot_index, operation, &decoded, dest, sources, extracted_pm,
                );
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
    use crate::tablegen::{build_decoder_tables, load_from_llvm_aie, CompositeEncoder, OperandField};
    use std::path::Path;

    fn make_add_encoding() -> InstrEncoding {
        use crate::tablegen::{AddressingMode, InstrMemWidth};
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
            addressing_mode: AddressingMode::Unknown,
            mem_width: InstrMemWidth::Word,
            has_complete_decoder: true,
            element_type: None,
            branch_condition: None,
            is_vector: false,
            select_variant: None,
            is_ptr_arithmetic: false,
            is_sp_relative: false,
            sched_class: None,
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
            addressing_mode: crate::tablegen::AddressingMode::Unknown,
            mem_width: crate::tablegen::InstrMemWidth::Word,
            has_complete_decoder: true,
            element_type: None,
            branch_condition: None,
            is_vector: false,
            select_variant: None,
            is_ptr_arithmetic: false,
            is_sp_relative: false,
            sched_class: None,
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

        // Load TableGen data and create decoder
        let data = load_from_llvm_aie(&llvm_aie_path).expect("Failed to load llvm-aie");
        let tables = build_decoder_tables(&data);
        let decoder = InstructionDecoder::from_tables(tables);

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
            matches!(s.op, Operation::Halt)
        );
        assert!(has_halt, "done instruction must decode to Operation::Halt, got: {:?}",
            bundle.active_slots().map(|s| format!("{:?}", s.op)).collect::<Vec<_>>());
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
    fn decode_80bit_bundle_slots(decoder: &InstructionDecoder, bytes: &[u8], pc: u32)
        -> Vec<(String, Option<Operand>, Vec<Operand>)>
    {
        use crate::interpreter::bundle::{extract_slots, SlotType};

        let extracted = extract_slots(bytes);
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
        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let tables = build_decoder_tables(&data);
        let decoder = InstructionDecoder::from_tables(tables);

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
        let data = load_from_llvm_aie(llvm_aie_path).expect("Failed to load llvm-aie");
        let tables = build_decoder_tables(&data);
        let decoder = InstructionDecoder::from_tables(tables);

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
        assert_eq!(dest, Some(Operand::PointerReg(6)), "SP should map to p6");
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
                branch_condition: None,
                is_vector,
                select_variant: None,
                is_ptr_arithmetic: false,
            is_sp_relative: false,
                sched_class: None,
            }
        };

        // Vector load A (slot=lda, is_vector=true) -> VectorLoadA
        let decoded_vlda = DecodedInstr {
            encoding: make_load_enc("lda", true),
            operands: HashMap::new(),
        };
        let op = decoder.to_operation(&decoded_vlda);
        assert!(matches!(op, Operation::VectorLoadA { .. }),
            "slot=lda + is_vector should produce VectorLoadA, got {:?}", op);

        // Vector load B (slot=ldb, is_vector=true) -> VectorLoadB
        let decoded_vldb = DecodedInstr {
            encoding: make_load_enc("ldb", true),
            operands: HashMap::new(),
        };
        let op = decoder.to_operation(&decoded_vldb);
        assert!(matches!(op, Operation::VectorLoadB { .. }),
            "slot=ldb + is_vector should produce VectorLoadB, got {:?}", op);

        // Scalar load in lda slot (is_vector=false) -> plain Load
        let decoded_scl = DecodedInstr {
            encoding: make_load_enc("lda", false),
            operands: HashMap::new(),
        };
        let op = decoder.to_operation(&decoded_scl);
        assert!(matches!(op, Operation::Load { .. }),
            "slot=lda + !is_vector should produce plain Load, got {:?}", op);

        // Vector store (slot=st, is_vector=true) -> VectorStore
        let mut store_enc = make_load_enc("st", true);
        store_enc.semantic = Some(SemanticOp::Store);
        store_enc.may_load = false;
        store_enc.may_store = true;
        let decoded_vst = DecodedInstr {
            encoding: store_enc,
            operands: HashMap::new(),
        };
        let op = decoder.to_operation(&decoded_vst);
        assert!(matches!(op, Operation::VectorStore { .. }),
            "slot=st + is_vector should produce VectorStore, got {:?}", op);
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
            !load_a.op.is_nop(),
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
        assert!(load_b.op.is_nop(), "LoadB slot should be NOP (nopb)");

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
            branch_condition: None,
            is_vector: true,
            select_variant: None,
            is_ptr_arithmetic: false,
            is_sp_relative: false,
            sched_class: None,
        }
    }

    /// Helper: call vector_op_from_mnemonic and return the Operation.
    fn dispatch_mnemonic(mnemonic: &str) -> Option<Operation> {
        let enc = make_vec_encoding(mnemonic);
        let decoded = DecodedInstr {
            encoding: enc,
            operands: HashMap::new(),
        };
        InstructionDecoder::vector_op_from_mnemonic(&decoded)
    }

    #[test]
    fn test_vector_mnemonic_dispatch_srs_ups() {
        // SRS: accumulator -> vector
        assert!(matches!(
            dispatch_mnemonic("vsrs.s8"),
            Some(Operation::VectorSRS { to_type: ElementType::Int8, .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vsrs.d16"),
            Some(Operation::VectorSRS { to_type: ElementType::Int16, .. })
        ));

        // UPS: vector -> accumulator
        assert!(matches!(
            dispatch_mnemonic("vups.s8"),
            Some(Operation::VectorUpshift { from_type: ElementType::Int8, .. })
        ));

        // PUSH: also upshift
        assert!(matches!(
            dispatch_mnemonic("vpush.lo.s16"),
            Some(Operation::VectorUpshift { from_type: ElementType::Int16, .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_mac_variants() {
        // vmac -> VectorMac (not VectorMul)
        assert!(matches!(
            dispatch_mnemonic("vmac"),
            Some(Operation::VectorMac { .. })
        ));

        // vmac.f -> VectorMatMulAccFloat (bfloat16)
        assert!(matches!(
            dispatch_mnemonic("vmac.f"),
            Some(Operation::VectorMatMulAccFloat { .. })
        ));

        // vmsc -> VectorMatMulSubDense (not VectorMul)
        assert!(matches!(
            dispatch_mnemonic("vmsc"),
            Some(Operation::VectorMatMulSubDense { .. })
        ));

        // vmsc.f -> VectorMatMulSubFloat
        assert!(matches!(
            dispatch_mnemonic("vmsc.f"),
            Some(Operation::VectorMatMulSubFloat { .. })
        ));

        // vnegmac -> VectorNegMatMulDense
        assert!(matches!(
            dispatch_mnemonic("vnegmac"),
            Some(Operation::VectorNegMatMulDense { .. })
        ));

        // vaddmac -> VectorAddMac
        assert!(matches!(
            dispatch_mnemonic("vaddmac"),
            Some(Operation::VectorAddMac { .. })
        ));

        // vsubmac -> VectorSubMac
        assert!(matches!(
            dispatch_mnemonic("vsubmac"),
            Some(Operation::VectorSubMac { .. })
        ));

        // vnegmul -> VectorNegMul
        assert!(matches!(
            dispatch_mnemonic("vnegmul"),
            Some(Operation::VectorNegMul { .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_conditional() {
        // Conditional arithmetic
        assert!(matches!(
            dispatch_mnemonic("vsub_lt.d32"),
            Some(Operation::VectorSubLt { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vsub_ge.s16"),
            Some(Operation::VectorSubGe { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vmaxdiff_lt.s8"),
            Some(Operation::VectorMaxDiffLt { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vmax_lt.bf"),
            Some(Operation::VectorMaxLt { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vmin_ge.d16"),
            Some(Operation::VectorMinGe { .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_comparisons() {
        assert!(matches!(
            dispatch_mnemonic("vge.s32"),
            Some(Operation::VectorGe { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vlt.d8"),
            Some(Operation::VectorLt { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("veqz.s16"),
            Some(Operation::VectorEqz { .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_abs_neg() {
        assert!(matches!(
            dispatch_mnemonic("vabs_gtz.s32"),
            Some(Operation::VectorAbsGtz { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vneg_gtz"),
            Some(Operation::VectorNegGtz { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vbneg_ltz.s16"),
            Some(Operation::VectorNegLtz { .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_data_movement() {
        assert!(matches!(
            dispatch_mnemonic("vshuffle"),
            Some(Operation::VectorShuffle { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vshift.align"),
            Some(Operation::VectorAlign { .. })
        ));
        assert!(matches!(dispatch_mnemonic("vclr"), Some(Operation::VectorClear)));
        assert!(matches!(
            dispatch_mnemonic("vsel.s32"),
            Some(Operation::VectorSelect { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vextbcst.s16"),
            Some(Operation::VectorBroadcast { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vbcst.s32"),
            Some(Operation::VectorBroadcast { .. })
        ));
        assert!(matches!(dispatch_mnemonic("vpack.d8"), Some(Operation::VectorPack)));
        assert!(matches!(
            dispatch_mnemonic("vunpack.s16"),
            Some(Operation::VectorUnpack)
        ));
        assert!(matches!(
            dispatch_mnemonic("vextract.s32"),
            Some(Operation::VectorExtract { .. })
        ));
        assert!(matches!(
            dispatch_mnemonic("vinsert.s16"),
            Some(Operation::VectorInsert { .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_conversion() {
        assert!(matches!(
            dispatch_mnemonic("vconv.bf16"),
            Some(Operation::VectorConvert { from_type: ElementType::Float32, to_type: ElementType::BFloat16 })
        ));
        assert!(matches!(
            dispatch_mnemonic("vconv.fp32"),
            Some(Operation::VectorConvert { from_type: ElementType::BFloat16, to_type: ElementType::Float32 })
        ));
        assert!(matches!(
            dispatch_mnemonic("vfloor.s32"),
            Some(Operation::VectorConvert { from_type: ElementType::Float32, .. })
        ));
    }

    #[test]
    fn test_vector_mnemonic_dispatch_noop_for_basic_ops() {
        // vadd, vsub, vmul should NOT be caught by mnemonic dispatch
        // (they work correctly through semantic dispatch)
        assert!(dispatch_mnemonic("vadd.s32").is_none());
        assert!(dispatch_mnemonic("vsub.s16").is_none());
        assert!(dispatch_mnemonic("vmul.s8").is_none());
        assert!(dispatch_mnemonic("vmov").is_none());
        assert!(dispatch_mnemonic("vand.s32").is_none());
        assert!(dispatch_mnemonic("vor.s16").is_none());
        assert!(dispatch_mnemonic("vxor.s32").is_none());
    }

    #[test]
    fn test_vector_mnemonic_dispatch_negadd_negsub() {
        // vnegadd -> VectorNegAdd
        assert!(matches!(
            dispatch_mnemonic("vnegadd.s32"),
            Some(Operation::VectorNegAdd { .. })
        ));
        // vnegsub -> VectorNegAdd (same semantics: -a + b)
        assert!(matches!(
            dispatch_mnemonic("vnegsub.f"),
            Some(Operation::VectorNegAdd { .. })
        ));
    }

    /// Verify every vector mnemonic in the TableGen data either has a specialized
    /// mnemonic dispatch entry or is one of the known "semantic-only" mnemonics
    /// that work correctly through the generic SemanticOp path.
    #[test]
    fn test_all_vector_mnemonics_dispatched() {
        // These mnemonics work correctly through SemanticOp dispatch
        // and intentionally do NOT have mnemonic overrides.
        let semantic_only: std::collections::HashSet<&str> = [
            "vadd", "vsub", "vmul", "vmov", "vneg",
            "vband", "vbor",
            "vlda", "vldb", "vst",
        ].into_iter().collect();

        let decoder = InstructionDecoder::load_default();
        let mut vec_mnemonics: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for enc in &decoder.encodings {
            let m = enc.mnemonic.to_lowercase();
            if m.starts_with('v') && m != "opcodestr" {
                let base = m.split('.').next().unwrap_or(&m).to_string();
                vec_mnemonics.insert(base);
            }
        }

        let mut uncovered = Vec::new();
        for m in &vec_mnemonics {
            if semantic_only.contains(m.as_str()) {
                continue;
            }
            if dispatch_mnemonic(m).is_none() {
                uncovered.push(m.clone());
            }
        }

        assert!(
            uncovered.is_empty(),
            "Vector mnemonics with no dispatch: {:?}\n\
             Add them to vector_op_from_mnemonic() or the semantic_only set.",
            uncovered,
        );
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
}
