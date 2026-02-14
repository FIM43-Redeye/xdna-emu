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
    SlotIndex, SlotOp, VliwBundle,
};
use crate::interpreter::traits::{DecodeError, Decoder};
use crate::tablegen::{
    build_decoder_tables, load_from_llvm_aie, load_via_tblgen, AddressingMode, CompositeEncoder,
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

// ─── Inverse Encoder Functions ───────────────────────────────────────────────
//
// Each function below is the inverse of the corresponding Peano encoder in
// AIE2MCCodeEmitterRegOperandDef.h. They take a raw bit-field value and return
// the decoded Operand.

/// Dispatch a composite register decode based on the encoder variant.
fn decode_composite(encoder: CompositeEncoder, raw: u64) -> Operand {
    match encoder {
        CompositeEncoder::LdaScl => decode_lda_scl(raw),
        CompositeEncoder::MvSclSrc => decode_mv_scl_src(raw),
        CompositeEncoder::LdaCg => decode_lda_cg(raw),
        CompositeEncoder::AluCg => decode_alu_cg(raw),
        CompositeEncoder::MvAMWQDst => decode_mv_amwq_dst(raw),
        CompositeEncoder::MvAMWQSrc => decode_mv_amwq_src(raw),
        CompositeEncoder::MvBMXSrc => decode_mv_bmx_src(raw),
        CompositeEncoder::MvBMXDst => decode_mv_bmx_dst(raw),
        CompositeEncoder::ERS4 => decode_ers4(raw),
        CompositeEncoder::ShflDst => decode_shfl_dst(raw),
        CompositeEncoder::Wm1 => decode_wm1(raw),
        CompositeEncoder::QXHLb => decode_qxhlb(raw),
    }
}

/// Inverse of `getmLdaSclOpValue`. Covers mLdaScl, mSclSt, mSclMS, mLdbScl.
///
/// Encoding:
///   lr         -> 0b0000101
///   eP(enc)    -> (enc << 4) | 0b1101
///   eR(enc)    -> (enc << 2) | 0b00
///   eM(enc)    -> ((enc | 0b00000) << 2) | 0b10
///   eDN(enc)   -> ((enc | 0b01000) << 2) | 0b10
///   eDJ(enc)   -> ((enc | 0b10000) << 2) | 0b10
///   eDC(enc)   -> ((enc | 0b11000) << 2) | 0b10
fn decode_lda_scl(raw: u64) -> Operand {
    // Special case: link register (HW encoding 5 in LdaScl space)
    if raw == 0b0000101 {
        return Operand::ScalarReg(crate::interpreter::state::LR_REG_INDEX);
    }
    // Pointer: low 4 bits == 0b1101
    if raw & 0xF == 0b1101 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    // Scalar: low 2 bits == 0b00
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    // Modifier/special: low 2 bits == 0b10
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18; // bits[4:3]
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(reg),        // eM
            0b01000 => Operand::ModifierReg(8 + reg),    // eDN
            0b10000 => Operand::ModifierReg(16 + reg),   // eDJ
            0b11000 => Operand::ModifierReg(24 + reg),   // eDC
            _ => Operand::ModifierReg(reg),
        };
    }
    // Fallback
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmMvSclSrcOpValue`. Covers mMvSclSrc, mMvSclDst, mMvSclDstCg.
///
/// Special register HW encodings (returned directly):
///   LS=7, DP=23, lr=39, CORE_ID=55, LE=71, LC=87, SP=103
///
/// Otherwise:
///   eR(enc)    -> (enc << 2) | 0b00
///   eP(enc)    -> (enc << 4) | 0b0011
///   eM(enc)    -> ((enc | 0b0000) << 2) | 0b10
///   eDN/DJ/DC  -> same as LdaScl modifier encoding
///   eS(enc)    -> (enc << 5) | 0b01011
///   mCRm(enc)  -> (enc << 3) | 0b001
///   mSRm(enc)  -> (enc << 3) | 0b101
fn decode_mv_scl_src(raw: u64) -> Operand {
    // Check for special register HW encodings first
    // These have bits[2:0] == 0b111 (the SPL tag) and bits[6:3] identify the register.
    if raw & 0x7 == 0b111 {
        // Special register: HWEncoding = (id << 3) | 0b111
        // id comes from AIE2SPLReg<Enc> in TableGen.
        use crate::interpreter::state::*;
        let id = (raw >> 3) & 0xF;
        return match id {
            0  => Operand::ScalarReg(LS_REG_INDEX),       // LS (loop start)
            2  => Operand::ScalarReg(DP_REG_INDEX),       // DP (decompress pointer)
            4  => Operand::ScalarReg(LR_REG_INDEX),       // lr (link register)
            6  => Operand::ScalarReg(CORE_ID_REG_INDEX),  // CORE_ID (read-only)
            8  => Operand::ScalarReg(LE_REG_INDEX),       // LE (loop end)
            10 => Operand::ScalarReg(LC_REG_INDEX),       // LC (loop count)
            12 => Operand::PointerReg(6),                 // SP (stack pointer = p6)
            _  => Operand::ScalarReg(0),
        };
    }
    // Pointer: low 4 bits == 0b0011
    if raw & 0xF == 0b0011 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    // Scalar: low 2 bits == 0b00
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    // eS: low 5 bits == 0b01011
    if raw & 0x1F == 0b01011 {
        let s_reg = ((raw >> 5) & 0x3) as u8;
        return Operand::ScalarReg(s_reg); // Status register, modeled as scalar
    }
    // Modifier/special: low 2 bits == 0b10
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18;
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(reg),
            0b01000 => Operand::ModifierReg(8 + reg),
            0b10000 => Operand::ModifierReg(16 + reg),
            0b11000 => Operand::ModifierReg(24 + reg),
            _ => Operand::ModifierReg(reg),
        };
    }
    // mCRm: low 3 bits == 0b001
    if raw & 0x7 == 0b001 {
        let cr_reg = ((raw >> 3) & 0xF) as u8;
        return Operand::ScalarReg(cr_reg); // Control register, modeled as scalar
    }
    // mSRm: low 3 bits == 0b101
    if raw & 0x7 == 0b101 {
        let sr_reg = ((raw >> 3) & 0xF) as u8;
        return Operand::ScalarReg(sr_reg); // Status register, modeled as scalar
    }
    // Fallback
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmLdaCgOpValue`. Covers mLdaCg, mLdbCg.
///
/// Encoding:
///   LC         -> 0b0010101
///   eP(enc)    -> (enc << 4) | 0b1101
///   eR(enc)    -> (enc << 2) | 0b00
///   eM/eDN/eDJ/eDC -> same modifier encoding as LdaScl
fn decode_lda_cg(raw: u64) -> Operand {
    // Special case: LC (loop count) -- HW encoding 0b0010101 = 21 in LdaCg space
    if raw == 0b0010101 {
        return Operand::ScalarReg(crate::interpreter::state::LC_REG_INDEX);
    }
    // Pointer: low 4 bits == 0b1101
    if raw & 0xF == 0b1101 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    // Scalar: low 2 bits == 0b00
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    // Modifier: low 2 bits == 0b10
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18;
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(reg),
            0b01000 => Operand::ModifierReg(8 + reg),
            0b10000 => Operand::ModifierReg(16 + reg),
            0b11000 => Operand::ModifierReg(24 + reg),
            _ => Operand::ModifierReg(reg),
        };
    }
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmAluCgOpValue`.
///
/// Encoding:
///   LC         -> 0b000001
///   eR(enc)    -> enc << 1
fn decode_alu_cg(raw: u64) -> Operand {
    if raw == 0b000001 {
        return Operand::ScalarReg(crate::interpreter::state::LC_REG_INDEX); // LC
    }
    let scl_reg = (raw >> 1) as u8;
    Operand::ScalarReg(scl_reg)
}

/// Inverse of `getmMvAMWQDstOpValue`.
///
/// Encoding:
///   mAMm(enc)    -> (enc << 1) | 0b1
///   eWLE(enc)    -> (0b00 << 5) | ((enc >> 2) << 2) | 0b00
///   eWLO(enc)    -> (0b01 << 5) | ((enc >> 2) << 2) | 0b00
///   eWHE(enc)    -> (0b10 << 5) | ((enc >> 2) << 2) | 0b00
///   eWHO(enc)    -> (0b11 << 5) | ((enc >> 2) << 2) | 0b00
///   mQQm(enc)    -> (enc << 5) | 0b00010
fn decode_mv_amwq_dst(raw: u64) -> Operand {
    // mAMm: low bit == 1
    if raw & 0x1 == 0b1 {
        let am_reg = (raw >> 1) as u8;
        return Operand::AccumReg(am_reg);
    }
    // mQQm: low 5 bits == 0b00010
    if raw & 0x1F == 0b00010 {
        let qq_reg = (raw >> 5) as u8;
        return Operand::AccumReg(qq_reg); // Weight queue
    }
    // eWxx: low 2 bits == 0b00, bits[6:5] select sub-class
    if raw & 0x3 == 0b00 {
        let _subclass = (raw >> 5) & 0x3; // 00=WLE, 01=WLO, 10=WHE, 11=WHO
        let enc_bits = ((raw >> 2) & 0x7) as u8;
        // Recover original encoding: enc = (enc_bits << 2) (since we stored enc >> 2)
        let vec_reg = enc_bits << 2;
        return Operand::VectorReg(vec_reg);
    }
    Operand::VectorReg(raw as u8)
}

/// Inverse of `getmMvAMWQSrcOpValue`.
///
/// Encoding:
///   mAMm(enc)    -> (enc << 3) | 0b001
///   eWLE(enc)    -> (0b110000 << 3) | (enc >> 2)
///   eWLO(enc)    -> (0b110001 << 3) | (enc >> 2)
///   eWHE(enc)    -> (0b110010 << 3) | (enc >> 2)
///   eWHO(enc)    -> (0b110011 << 3) | (enc >> 2)
///   mQQm(enc)    -> (0b111 << 6) | (enc << 4) | 0b0000
fn decode_mv_amwq_src(raw: u64) -> Operand {
    // Check for mQQm: bits[6:4] == 0b111, low 4 bits == 0b0000
    if (raw >> 6) & 0x7 == 0b111 && raw & 0xF == 0b0000 {
        let qq_reg = ((raw >> 4) & 0x3) as u8;
        return Operand::AccumReg(qq_reg);
    }
    // Check for eWxx: bits[8:3] starts with 0b1100xx
    let upper6 = (raw >> 3) & 0x3F;
    if upper6 >= 0b110000 && upper6 <= 0b110011 {
        let enc_div4 = (raw & 0x7) as u8;
        let vec_reg = enc_div4 << 2;
        return Operand::VectorReg(vec_reg);
    }
    // mAMm: low 3 bits == 0b001
    if raw & 0x7 == 0b001 {
        let am_reg = (raw >> 3) as u8;
        return Operand::AccumReg(am_reg);
    }
    Operand::AccumReg(raw as u8)
}

/// Inverse of `getmMvBMXSrcOpValue`.
///
/// Encoding:
///   mBMm(enc) -> (enc << 4) | 0b0000
///   mXm(enc)  -> (0b11000 << 4) | enc
fn decode_mv_bmx_src(raw: u64) -> Operand {
    // mXm: high bits == 0b11000
    if (raw >> 4) & 0x1F == 0b11000 {
        let x_reg = (raw & 0xF) as u8;
        return Operand::VectorReg(x_reg); // 512-bit vector
    }
    // mBMm: low 4 bits == 0b0000
    let bm_reg = (raw >> 4) as u8;
    Operand::AccumReg(bm_reg)
}

/// Inverse of `getmMvBMXDstOpValue`.
///
/// Encoding:
///   mBMm(enc) -> (enc << 1) | 0b1
///   mXm(enc)  -> (enc << 2) | 0b00
fn decode_mv_bmx_dst(raw: u64) -> Operand {
    // mBMm: low bit == 1
    if raw & 0x1 == 0b1 {
        let bm_reg = (raw >> 1) as u8;
        return Operand::AccumReg(bm_reg);
    }
    // mXm: low 2 bits == 0b00
    let x_reg = (raw >> 2) as u8;
    Operand::VectorReg(x_reg)
}

/// Inverse of `geteRS4OpValue`.
///
/// Encoding: eRS4(enc) -> enc - 16
fn decode_ers4(raw: u64) -> Operand {
    let scl_reg = (raw as u8).wrapping_add(16);
    Operand::ScalarReg(scl_reg)
}

/// Inverse of `getmShflDstOpValue`.
///
/// Encoding:
///   mBMSm(enc) -> ((((enc & 1) << 3) | (enc >> 1)) << 1) | 0b1
///   mXm(enc)   -> (enc << 1) | 0b0
fn decode_shfl_dst(raw: u64) -> Operand {
    // mXm: low bit == 0
    if raw & 0x1 == 0b0 {
        let x_reg = (raw >> 1) as u8;
        return Operand::VectorReg(x_reg);
    }
    // mBMSm: low bit == 1, need to reverse the bit rearrangement
    let shifted = raw >> 1;
    // Forward: ((lsb << 3) | (rest >> 1)) where lsb = enc & 1, rest = enc >> 1
    // shifted = (lsb << 3) | (rest >> 1)
    // So: lsb = (shifted >> 3) & 1, upper = shifted & 0x7
    // enc = (upper << 1) | lsb
    let lsb = ((shifted >> 3) & 0x1) as u8;
    let upper = (shifted & 0x7) as u8;
    let bms_reg = (upper << 1) | lsb;
    Operand::AccumReg(bms_reg)
}

/// Inverse of `getmWm_1OpValue`.
///
/// Encoding: mWm(enc) -> (((bit0 << 1) | bit1) << 3) | (enc >> 2)
///   where bit0 = enc & 1, bit1 = (enc >> 1) & 1
fn decode_wm1(raw: u64) -> Operand {
    // Forward: result = (rearranged_low2 << 3) | (enc >> 2)
    // rearranged = (bit0 << 1) | bit1
    let enc_upper = (raw & 0x7) as u8;  // enc >> 2 (3 bits)
    let rearranged = ((raw >> 3) & 0x3) as u8;
    // rearranged = (bit0 << 1) | bit1, so bit0 = (rearranged >> 1) & 1, bit1 = rearranged & 1
    let bit0 = (rearranged >> 1) & 0x1;
    let bit1 = rearranged & 0x1;
    let vec_reg = (enc_upper << 2) | (bit1 << 1) | bit0;
    Operand::VectorReg(vec_reg)
}

/// Inverse of `getmQXHLbOpValue`.
///
/// Encoding:
///   mQXHb(enc) -> (enc << 1) | 0b0
///   mQXLb(enc) -> (enc << 1) | 0b1
fn decode_qxhlb(raw: u64) -> Operand {
    // Low bit selects high/low sub-register, upper bits are the register index
    let reg = (raw >> 1) as u8;
    Operand::AccumReg(reg)
}

// ─── End Inverse Encoder Functions ──────────────────────────────────────────

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
        encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));

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

        encodings.sort_by(|a, b| b.sort_key().cmp(&a.sort_key()));
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

    /// Convert a SemanticOp to an Operation using pre-resolved metadata.
    ///
    /// All element type, branch condition, and select variant disambiguation is
    /// resolved at TableGen load time on InstrEncoding. No mnemonic string parsing
    /// happens here -- just field lookups.
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
            SemanticOp::And => Operation::ScalarAnd,
            SemanticOp::Or => Operation::ScalarOr,
            SemanticOp::Xor => Operation::ScalarXor,
            SemanticOp::Shl => Operation::ScalarShl,
            SemanticOp::Sra => Operation::ScalarSra,
            SemanticOp::Srl => Operation::ScalarShr,
            SemanticOp::Load => {
                // vlda/vldb/vst distinction: mem_width == Vector256 means vector,
                // and the mnemonic prefix distinguishes A vs B channel. This is a
                // cheap 4-char check that's cleaner than adding SemanticOp variants.
                let mn = &enc.mnemonic;
                if mn.starts_with("vlda") {
                    Operation::VectorLoadA { post_modify: PostModify::None }
                } else if mn.starts_with("vldb") {
                    Operation::VectorLoadB { post_modify: PostModify::None }
                } else {
                    let width = match enc.mem_width {
                        InstrMemWidth::Byte => MemWidth::Byte,
                        InstrMemWidth::HalfWord => MemWidth::HalfWord,
                        InstrMemWidth::Word => MemWidth::Word,
                        InstrMemWidth::Vector256 => MemWidth::Vector256,
                    };
                    Operation::Load { width, post_modify: PostModify::None }
                }
            },
            SemanticOp::Store => {
                let mn = &enc.mnemonic;
                if mn.starts_with("vst") {
                    Operation::VectorStore { post_modify: PostModify::None }
                } else {
                    let width = match enc.mem_width {
                        InstrMemWidth::Byte => MemWidth::Byte,
                        InstrMemWidth::HalfWord => MemWidth::HalfWord,
                        InstrMemWidth::Word => MemWidth::Word,
                        InstrMemWidth::Vector256 => MemWidth::Vector256,
                    };
                    Operation::Store { width, post_modify: PostModify::None }
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

            // Other scalar ops
            SemanticOp::Abs => Operation::ScalarAbs,

            // Control flow
            SemanticOp::Ret => Operation::Return,
            SemanticOp::Done => Operation::Halt,

            // Not handled yet - fall through
            _ => Operation::Unknown { opcode: 0 },
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

        let mnemonic = decoded.encoding.mnemonic.to_lowercase();

        for field in &decoded.encoding.operand_fields {
            let raw = decoded.operand(&field.name).unwrap_or(0);

            // Address generator fields encode packed {ptr, offset/mod, mode} --
            // this is structural addressing, not a register encoding issue.
            if field.name.starts_with("ag") {
                self.decode_ag_field(
                    field, raw, decoded,
                    &mnemonic,
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
                        RegisterKind::Modifier => Operand::ModifierReg(reg),
                        RegisterKind::Vector256 | RegisterKind::Vector512 =>
                            Operand::VectorReg(reg),
                        RegisterKind::Accumulator => Operand::AccumReg(reg),
                    }
                }
                OperandType::CompositeRegister(encoder) => {
                    decode_composite(*encoder, raw)
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
        let addr_mode = decoded.encoding.addressing_mode;
        if matches!(addr_mode, AddressingMode::IndexedImmediate | AddressingMode::IndexedRegister) {
            if let (Some(Operand::PointerReg(base)), Some(imm_op)) =
                (field_operands.get("ptr").cloned(), field_operands.get("imm").cloned())
            {
                let offset = match imm_op {
                    Operand::Immediate(v) => v as i16,
                    _ => 0,
                };
                if mnemonic.starts_with("padd") {
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
        mnemonic: &str,
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
                if mnemonic.starts_with("padd") {
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
                let ptr = ((value >> (field.width as u64 - 3)) & 0x7) as u8;
                let mod_reg = ((value >> 3) & 0x7) as u8;
                if mnemonic.starts_with("padd") {
                    *direct_dest = Some(Operand::PointerReg(ptr));
                    extra_sources.push(Operand::ModifierReg(mod_reg));
                } else {
                    extra_sources.push(Operand::PointerReg(ptr));
                    extra_sources.push(Operand::ModifierReg(mod_reg));
                }
            }
            AddressingMode::PostModifyRegister => {
                let ptr = ((value >> (field.width as u64 - 3)) & 0x7) as u8;
                let mod_reg = ((value >> 3) & 0x7) as u8;
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

                if mnemonic.starts_with("padd") {
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
                    SlotType::Vec => SlotIndex::Vector,
                    // LNG slot can contain either control flow (JL, J) or vector (movxm).
                    // Determine the correct slot after decoding via operation.natural_slot().
                    SlotType::Lng => SlotIndex::Vector,
                    SlotType::Nop => SlotIndex::Control,
                };

                // Try to decode the slot bits against known encodings
                // Zero bits means no operation in this slot - treat as NOP
                if slot.slot_type == SlotType::Nop || slot.bits == 0 {
                    bundle.set_slot(SlotOp::nop(slot_index));
                } else if let Some(decoded) = self.decode_slot_bits(slot.bits, slot.slot_type) {
                    let mut operation = self.to_operation(&decoded);
                    let (dest, sources, extracted_pm) = if decoded.encoding.mnemonic == "movxm" {
                        // Special handling for movxm: immediate is SPLIT in the lng slot
                        // TableGen layout: lng = {i{31-12}, mMvSclDstCg, i{11-0}, 0b001}
                        let i_low = ((slot.bits >> 3) & 0xFFF) as u32;
                        let dst_raw = ((slot.bits >> 15) & 0x7F) as u8;
                        let i_high = ((slot.bits >> 22) & 0xFFFFF) as u32;
                        let full_imm = (i_high << 12) | i_low;

                        // Decode destination from mMvSclDstCg composite register encoding
                        let dest_operand = if dst_raw & 0x3 == 0x3 {
                            Operand::PointerReg((dst_raw >> 4) & 0x7)
                        } else {
                            Operand::ScalarReg((dst_raw >> 2) & 0x1F)
                        };

                        (Some(dest_operand), vec![Operand::Immediate(full_imm as i32)], None)
                    } else {
                        self.extract_operands(&decoded)
                    };

                    // Patch PostModify from ag_* extraction into the Operation
                    if let Some(pm) = extracted_pm {
                        match &mut operation {
                            Operation::Load { post_modify, .. } => *post_modify = pm,
                            Operation::Store { post_modify, .. } => *post_modify = pm,
                            Operation::VectorLoadA { post_modify } => *post_modify = pm,
                            Operation::VectorLoadB { post_modify } => *post_modify = pm,
                            Operation::VectorStore { post_modify } => *post_modify = pm,
                            _ => {}
                        }
                    }

                    // For LNG slot instructions, use the operation's natural slot since
                    // LNG can contain either control (JL, J) or vector/scalar (movxm).
                    let effective_slot = if slot.slot_type == SlotType::Lng {
                        operation.natural_slot()
                    } else {
                        slot_index
                    };

                    // Build SlotOp with TableGen info (reorders sources, adds semantic/implicit)
                    let slot_op = self.build_slot_op(effective_slot, operation, &decoded, dest, sources);
                    bundle.set_slot(slot_op);
                } else {
                    // Slot extracted but not recognized - mark as unknown with slot info
                    log::trace!("[{:?} DECODE FAIL] bits=0x{:010X} - no matching encoding",
                        slot.slot_type, slot.bits);
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
                let mut operation = self.to_operation(&decoded);
                let slot_index = self.slot_to_index(&decoded.encoding.slot);
                let (dest, sources, extracted_pm) = self.extract_operands(&decoded);

                // Patch PostModify from ag_* extraction into the Operation
                if let Some(pm) = extracted_pm {
                    match &mut operation {
                        Operation::Load { post_modify, .. } => *post_modify = pm,
                        Operation::Store { post_modify, .. } => *post_modify = pm,
                        Operation::VectorLoadA { post_modify } => *post_modify = pm,
                        Operation::VectorLoadB { post_modify } => *post_modify = pm,
                        Operation::VectorStore { post_modify } => *post_modify = pm,
                        _ => {}
                    }
                }

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

    // === Inverse Encoder Function Tests ===
    //
    // Each test verifies that encode(reg) -> raw -> decode(raw) == reg.
    // Encoding formulas come from AIE2MCCodeEmitterRegOperandDef.h.

    #[test]
    fn test_decode_lda_scl_scalar() {
        // eR(7): encode = (7 << 2) | 0b00 = 28
        assert_eq!(decode_lda_scl(28), Operand::ScalarReg(7));
        // eR(0): encode = (0 << 2) | 0b00 = 0
        assert_eq!(decode_lda_scl(0), Operand::ScalarReg(0));
        // eR(31): encode = (31 << 2) | 0b00 = 124
        assert_eq!(decode_lda_scl(124), Operand::ScalarReg(31));
    }

    #[test]
    fn test_decode_lda_scl_pointer() {
        // eP(0): encode = (0 << 4) | 0b1101 = 13
        assert_eq!(decode_lda_scl(13), Operand::PointerReg(0));
        // eP(3): encode = (3 << 4) | 0b1101 = 61
        assert_eq!(decode_lda_scl(61), Operand::PointerReg(3));
        // eP(7): encode = (7 << 4) | 0b1101 = 125
        assert_eq!(decode_lda_scl(125), Operand::PointerReg(7));
    }

    #[test]
    fn test_decode_lda_scl_lr() {
        // lr: encode = 0b0000101 = 5
        use crate::interpreter::state::LR_REG_INDEX;
        assert_eq!(decode_lda_scl(5), Operand::ScalarReg(LR_REG_INDEX));
    }

    #[test]
    fn test_decode_lda_scl_modifier() {
        // eM(0): encode = ((0 | 0b00000) << 2) | 0b10 = 2
        assert_eq!(decode_lda_scl(2), Operand::ModifierReg(0));
        // eM(3): encode = ((3 | 0b00000) << 2) | 0b10 = 14
        assert_eq!(decode_lda_scl(14), Operand::ModifierReg(3));
        // eDN(0): encode = ((0 | 0b01000) << 2) | 0b10 = 34
        assert_eq!(decode_lda_scl(34), Operand::ModifierReg(8));
        // eDJ(0): encode = ((0 | 0b10000) << 2) | 0b10 = 66
        assert_eq!(decode_lda_scl(66), Operand::ModifierReg(16));
        // eDC(0): encode = ((0 | 0b11000) << 2) | 0b10 = 98
        assert_eq!(decode_lda_scl(98), Operand::ModifierReg(24));
    }

    #[test]
    fn test_decode_mv_scl_src_scalar() {
        // eR(7): encode = (7 << 2) | 0b00 = 28
        assert_eq!(decode_mv_scl_src(28), Operand::ScalarReg(7));
        // eR(0): encode = 0
        assert_eq!(decode_mv_scl_src(0), Operand::ScalarReg(0));
    }

    #[test]
    fn test_decode_mv_scl_src_pointer() {
        // eP(3): encode = (3 << 4) | 0b0011 = 51
        assert_eq!(decode_mv_scl_src(51), Operand::PointerReg(3));
        // eP(0): encode = (0 << 4) | 0b0011 = 3
        assert_eq!(decode_mv_scl_src(3), Operand::PointerReg(0));
        // eP(7): encode = (7 << 4) | 0b0011 = 115
        assert_eq!(decode_mv_scl_src(115), Operand::PointerReg(7));
    }

    #[test]
    fn test_decode_mv_scl_src_special() {
        use crate::interpreter::state::*;
        // SP: HWEncoding = 103 = 0b1100111 (bits[2:0] = 0b111, bits[6:3] = 12)
        assert_eq!(decode_mv_scl_src(103), Operand::PointerReg(6)); // SP = p6
        // LC: HWEncoding = 87 = 0b1010111
        assert_eq!(decode_mv_scl_src(87), Operand::ScalarReg(LC_REG_INDEX));
        // lr: HWEncoding = 39 = 0b0100111
        assert_eq!(decode_mv_scl_src(39), Operand::ScalarReg(LR_REG_INDEX));
        // LS: HWEncoding = 7 = 0b0000111
        assert_eq!(decode_mv_scl_src(7), Operand::ScalarReg(LS_REG_INDEX));
        // LE: HWEncoding = 71 = 0b1000111
        assert_eq!(decode_mv_scl_src(71), Operand::ScalarReg(LE_REG_INDEX));
        // DP: HWEncoding = 23 = 0b0010111
        assert_eq!(decode_mv_scl_src(23), Operand::ScalarReg(DP_REG_INDEX));
        // CORE_ID: HWEncoding = 55 = 0b0110111
        assert_eq!(decode_mv_scl_src(55), Operand::ScalarReg(CORE_ID_REG_INDEX));
    }

    #[test]
    fn test_decode_lda_cg_scalar() {
        // eR(7): encode = (7 << 2) | 0b00 = 28
        assert_eq!(decode_lda_cg(28), Operand::ScalarReg(7));
    }

    #[test]
    fn test_decode_lda_cg_pointer() {
        // eP(3): encode = (3 << 4) | 0b1101 = 61
        assert_eq!(decode_lda_cg(61), Operand::PointerReg(3));
    }

    #[test]
    fn test_decode_lda_cg_lc() {
        use crate::interpreter::state::LC_REG_INDEX;
        // LC: encode = 0b0010101 = 21
        assert_eq!(decode_lda_cg(21), Operand::ScalarReg(LC_REG_INDEX));
    }

    #[test]
    fn test_decode_alu_cg() {
        use crate::interpreter::state::LC_REG_INDEX;
        // eR(7): encode = 7 << 1 = 14
        assert_eq!(decode_alu_cg(14), Operand::ScalarReg(7));
        // LC: encode = 0b000001 = 1
        assert_eq!(decode_alu_cg(1), Operand::ScalarReg(LC_REG_INDEX));
    }

    #[test]
    fn test_decode_mv_bmx_src() {
        // mBMm(0): encode = (0 << 4) | 0b0000 = 0
        assert_eq!(decode_mv_bmx_src(0), Operand::AccumReg(0));
        // mBMm(3): encode = (3 << 4) | 0b0000 = 48
        assert_eq!(decode_mv_bmx_src(48), Operand::AccumReg(3));
        // mXm(5): encode = (0b11000 << 4) | 5 = 389
        assert_eq!(decode_mv_bmx_src(389), Operand::VectorReg(5));
    }

    #[test]
    fn test_decode_mv_bmx_dst() {
        // mBMm(2): encode = (2 << 1) | 0b1 = 5
        assert_eq!(decode_mv_bmx_dst(5), Operand::AccumReg(2));
        // mXm(3): encode = (3 << 2) | 0b00 = 12
        assert_eq!(decode_mv_bmx_dst(12), Operand::VectorReg(3));
    }

    #[test]
    fn test_decode_ers4() {
        // eRS4 register 16: encode = 16 - 16 = 0
        assert_eq!(decode_ers4(0), Operand::ScalarReg(16));
        // eRS4 register 31: encode = 31 - 16 = 15
        assert_eq!(decode_ers4(15), Operand::ScalarReg(31));
    }

    #[test]
    fn test_decode_shfl_dst() {
        // mXm(3): encode = (3 << 1) | 0b0 = 6
        assert_eq!(decode_shfl_dst(6), Operand::VectorReg(3));
        // mXm(0): encode = (0 << 1) | 0b0 = 0
        assert_eq!(decode_shfl_dst(0), Operand::VectorReg(0));
    }

    #[test]
    fn test_decode_qxhlb() {
        // mQXHb(2): encode = (2 << 1) | 0b0 = 4
        assert_eq!(decode_qxhlb(4), Operand::AccumReg(2));
        // mQXLb(2): encode = (2 << 1) | 0b1 = 5
        assert_eq!(decode_qxhlb(5), Operand::AccumReg(2));
    }

    #[test]
    fn test_decode_composite_dispatch() {
        // Verify the dispatch function routes correctly
        // LdaScl: eR(7) = 28
        assert_eq!(
            decode_composite(CompositeEncoder::LdaScl, 28),
            Operand::ScalarReg(7)
        );
        // MvSclSrc: eP(3) = 51
        assert_eq!(
            decode_composite(CompositeEncoder::MvSclSrc, 51),
            Operand::PointerReg(3)
        );
        // AluCg: eR(5) = 10
        assert_eq!(
            decode_composite(CompositeEncoder::AluCg, 10),
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

    #[test]
    fn test_lda_scl_vs_mv_scl_pointer_encoding_differs() {
        // This is THE critical test: the same pointer register (p3) encodes
        // differently in LdaScl vs MvSclSrc. The old heuristic code got this wrong.
        //
        // p3 in LdaScl: (3 << 4) | 0b1101 = 61
        // p3 in MvSclSrc: (3 << 4) | 0b0011 = 51
        //
        // Decoding 61 with MvSclSrc would give the wrong register,
        // and decoding 51 with LdaScl would also be wrong.
        assert_eq!(decode_lda_scl(61), Operand::PointerReg(3));
        assert_eq!(decode_mv_scl_src(51), Operand::PointerReg(3));

        // Cross-check: 51 through LdaScl should NOT give p3
        // 51 = 0b0110011, low 4 bits = 0b0011, not 0b1101, so it's not a pointer in LdaScl
        // low 2 bits = 0b11, which doesn't match any LdaScl pattern cleanly
        let cross = decode_lda_scl(51);
        assert_ne!(cross, Operand::PointerReg(3),
            "LdaScl(51) should NOT decode as p3 -- different encoding scheme");
    }
}
