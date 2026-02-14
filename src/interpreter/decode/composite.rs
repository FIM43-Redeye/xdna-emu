//! Composite register decode LUTs.
//!
//! AIE2 uses "composite encoders" that pack multiple register classes into
//! a single bit field using class-specific discriminant patterns. For example,
//! the `mLdaScl` field encodes scalars, pointers, modifier registers, and
//! the link register in a 7-bit field using different bit patterns:
//!
//! ```text
//! lr         -> 0b0000101
//! eR(enc)    -> (enc << 2) | 0b00
//! eP(enc)    -> (enc << 4) | 0b1101
//! eM(enc)    -> ((enc | 0b00000) << 2) | 0b10
//! ```
//!
//! The formulas come from `AIE2MCCodeEmitterRegOperandDef.h` in llvm-aie.
//! Rather than running the inverse function per-decode, we build lookup
//! tables at initialization time and validate them against the register
//! HWEncodings parsed from TableGen.
//!
//! # Hot Path
//!
//! After initialization, decoding is a single array lookup:
//! ```ignore
//! let operand = lut[raw_value as usize];
//! ```

use crate::interpreter::bundle::Operand;
use crate::interpreter::state::{
    CORE_ID_REG_INDEX, DP_REG_INDEX, LC_REG_INDEX, LE_REG_INDEX, LR_REG_INDEX, LS_REG_INDEX,
    MOD_BASE_DC, MOD_BASE_DJ, MOD_BASE_DN, MOD_BASE_M,
};
use crate::tablegen::CompositeEncoder;

/// Lookup table for one composite encoder variant.
///
/// Built at initialization time by evaluating the inverse encoder function
/// for every possible raw value. After construction, decoding is O(1).
#[derive(Clone)]
pub struct CompositeLut {
    /// LUT entries: index = raw field value, value = decoded operand.
    /// None means the raw value doesn't correspond to a valid register.
    entries: Vec<Option<Operand>>,
}

impl CompositeLut {
    /// Build a LUT by evaluating an inverse decode function for all possible values.
    fn build(width_bits: u8, decode_fn: fn(u64) -> Operand) -> Self {
        let size = 1usize << width_bits;
        let entries: Vec<Option<Operand>> = (0..size)
            .map(|raw| Some(decode_fn(raw as u64)))
            .collect();
        Self { entries }
    }

    /// Look up a raw field value in the LUT.
    #[inline]
    pub fn decode(&self, raw: u64) -> Operand {
        let idx = raw as usize;
        if idx < self.entries.len() {
            self.entries[idx].clone().unwrap_or(Operand::Immediate(raw as i32))
        } else {
            Operand::Immediate(raw as i32)
        }
    }
}

/// Collection of all composite register decode LUTs.
///
/// Built once at decoder initialization from the inverse encoder functions.
#[derive(Clone)]
pub struct CompositeLuts {
    lda_scl: CompositeLut,
    mv_scl_src: CompositeLut,
    lda_cg: CompositeLut,
    alu_cg: CompositeLut,
    mv_amwq_dst: CompositeLut,
    mv_amwq_src: CompositeLut,
    mv_bmx_src: CompositeLut,
    mv_bmx_dst: CompositeLut,
    ers4: CompositeLut,
    shfl_dst: CompositeLut,
    wm1: CompositeLut,
    qxhlb: CompositeLut,
}

impl CompositeLuts {
    /// Build all LUTs from the inverse encoder functions.
    pub fn build() -> Self {
        Self {
            lda_scl: CompositeLut::build(7, decode_lda_scl),
            mv_scl_src: CompositeLut::build(7, decode_mv_scl_src),
            lda_cg: CompositeLut::build(7, decode_lda_cg),
            alu_cg: CompositeLut::build(6, decode_alu_cg),
            mv_amwq_dst: CompositeLut::build(7, decode_mv_amwq_dst),
            mv_amwq_src: CompositeLut::build(9, decode_mv_amwq_src),
            mv_bmx_src: CompositeLut::build(9, decode_mv_bmx_src),
            mv_bmx_dst: CompositeLut::build(6, decode_mv_bmx_dst),
            ers4: CompositeLut::build(4, decode_ers4),
            shfl_dst: CompositeLut::build(6, decode_shfl_dst),
            wm1: CompositeLut::build(5, decode_wm1),
            qxhlb: CompositeLut::build(5, decode_qxhlb),
        }
    }

    /// Decode a composite register field using the pre-built LUT.
    #[inline]
    pub fn decode(&self, encoder: CompositeEncoder, raw: u64) -> Operand {
        match encoder {
            CompositeEncoder::LdaScl => self.lda_scl.decode(raw),
            CompositeEncoder::MvSclSrc => self.mv_scl_src.decode(raw),
            CompositeEncoder::LdaCg => self.lda_cg.decode(raw),
            CompositeEncoder::AluCg => self.alu_cg.decode(raw),
            CompositeEncoder::MvAMWQDst => self.mv_amwq_dst.decode(raw),
            CompositeEncoder::MvAMWQSrc => self.mv_amwq_src.decode(raw),
            CompositeEncoder::MvBMXSrc => self.mv_bmx_src.decode(raw),
            CompositeEncoder::MvBMXDst => self.mv_bmx_dst.decode(raw),
            CompositeEncoder::ERS4 => self.ers4.decode(raw),
            CompositeEncoder::ShflDst => self.shfl_dst.decode(raw),
            CompositeEncoder::Wm1 => self.wm1.decode(raw),
            CompositeEncoder::QXHLb => self.qxhlb.decode(raw),
        }
    }
}

// ─── Inverse Encoder Functions ───────────────────────────────────────────────
//
// Each function below is the inverse of the corresponding Peano encoder in
// AIE2MCCodeEmitterRegOperandDef.h. They are called ONLY at initialization
// time to build the LUTs. The hot decode path uses the LUT directly.

/// Inverse of `getmLdaSclOpValue`. Covers mLdaScl, mSclSt, mSclMS, mLdbScl.
fn decode_lda_scl(raw: u64) -> Operand {
    if raw == 0b0000101 {
        return Operand::ScalarReg(LR_REG_INDEX);
    }
    if raw & 0xF == 0b1101 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18;
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(MOD_BASE_M + reg),
            0b01000 => Operand::ModifierReg(MOD_BASE_DN + reg),
            0b10000 => Operand::ModifierReg(MOD_BASE_DJ + reg),
            0b11000 => Operand::ModifierReg(MOD_BASE_DC + reg),
            _ => Operand::ModifierReg(MOD_BASE_M + reg),
        };
    }
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmMvSclSrcOpValue`. Covers mMvSclSrc, mMvSclDst, mMvSclDstCg.
fn decode_mv_scl_src(raw: u64) -> Operand {
    if raw & 0x7 == 0b111 {
        let id = (raw >> 3) & 0xF;
        return match id {
            0  => Operand::ScalarReg(LS_REG_INDEX),
            2  => Operand::ScalarReg(DP_REG_INDEX),
            4  => Operand::ScalarReg(LR_REG_INDEX),
            6  => Operand::ScalarReg(CORE_ID_REG_INDEX),
            8  => Operand::ScalarReg(LE_REG_INDEX),
            10 => Operand::ScalarReg(LC_REG_INDEX),
            12 => Operand::PointerReg(6), // SP = p6
            _  => Operand::ScalarReg(0),
        };
    }
    if raw & 0xF == 0b0011 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    if raw & 0x1F == 0b01011 {
        let s_reg = ((raw >> 5) & 0x3) as u8;
        return Operand::ScalarReg(s_reg);
    }
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18;
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(MOD_BASE_M + reg),
            0b01000 => Operand::ModifierReg(MOD_BASE_DN + reg),
            0b10000 => Operand::ModifierReg(MOD_BASE_DJ + reg),
            0b11000 => Operand::ModifierReg(MOD_BASE_DC + reg),
            _ => Operand::ModifierReg(MOD_BASE_M + reg),
        };
    }
    if raw & 0x7 == 0b001 {
        let cr_reg = ((raw >> 3) & 0xF) as u8;
        return Operand::ScalarReg(cr_reg);
    }
    if raw & 0x7 == 0b101 {
        let sr_reg = ((raw >> 3) & 0xF) as u8;
        return Operand::ScalarReg(sr_reg);
    }
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmLdaCgOpValue`. Covers mLdaCg, mLdbCg.
fn decode_lda_cg(raw: u64) -> Operand {
    if raw == 0b0010101 {
        return Operand::ScalarReg(LC_REG_INDEX);
    }
    if raw & 0xF == 0b1101 {
        let ptr_reg = ((raw >> 4) & 0x7) as u8;
        return Operand::PointerReg(ptr_reg);
    }
    if raw & 0x3 == 0b00 {
        let scl_reg = ((raw >> 2) & 0x1F) as u8;
        return Operand::ScalarReg(scl_reg);
    }
    if raw & 0x3 == 0b10 {
        let combined = (raw >> 2) & 0x1F;
        let prefix = combined & 0x18;
        let reg = (combined & 0x07) as u8;
        return match prefix {
            0b00000 => Operand::ModifierReg(MOD_BASE_M + reg),
            0b01000 => Operand::ModifierReg(MOD_BASE_DN + reg),
            0b10000 => Operand::ModifierReg(MOD_BASE_DJ + reg),
            0b11000 => Operand::ModifierReg(MOD_BASE_DC + reg),
            _ => Operand::ModifierReg(MOD_BASE_M + reg),
        };
    }
    Operand::ScalarReg((raw >> 2) as u8)
}

/// Inverse of `getmAluCgOpValue`.
fn decode_alu_cg(raw: u64) -> Operand {
    if raw == 0b000001 {
        return Operand::ScalarReg(LC_REG_INDEX);
    }
    let scl_reg = (raw >> 1) as u8;
    Operand::ScalarReg(scl_reg)
}

/// Inverse of `getmMvAMWQDstOpValue`.
fn decode_mv_amwq_dst(raw: u64) -> Operand {
    if raw & 0x1 == 0b1 {
        let am_reg = (raw >> 1) as u8;
        return Operand::AccumReg(am_reg);
    }
    if raw & 0x1F == 0b00010 {
        let qq_reg = (raw >> 5) as u8;
        return Operand::AccumReg(qq_reg);
    }
    if raw & 0x3 == 0b00 {
        let enc_bits = ((raw >> 2) & 0x7) as u8;
        let vec_reg = enc_bits << 2;
        return Operand::VectorReg(vec_reg);
    }
    Operand::VectorReg(raw as u8)
}

/// Inverse of `getmMvAMWQSrcOpValue`.
fn decode_mv_amwq_src(raw: u64) -> Operand {
    if (raw >> 6) & 0x7 == 0b111 && raw & 0xF == 0b0000 {
        let qq_reg = ((raw >> 4) & 0x3) as u8;
        return Operand::AccumReg(qq_reg);
    }
    let upper6 = (raw >> 3) & 0x3F;
    if upper6 >= 0b110000 && upper6 <= 0b110011 {
        let enc_div4 = (raw & 0x7) as u8;
        let vec_reg = enc_div4 << 2;
        return Operand::VectorReg(vec_reg);
    }
    if raw & 0x7 == 0b001 {
        let am_reg = (raw >> 3) as u8;
        return Operand::AccumReg(am_reg);
    }
    Operand::AccumReg(raw as u8)
}

/// Inverse of `getmMvBMXSrcOpValue`.
fn decode_mv_bmx_src(raw: u64) -> Operand {
    if (raw >> 4) & 0x1F == 0b11000 {
        let x_reg = (raw & 0xF) as u8;
        return Operand::VectorReg(x_reg);
    }
    let bm_reg = (raw >> 4) as u8;
    Operand::AccumReg(bm_reg)
}

/// Inverse of `getmMvBMXDstOpValue`.
fn decode_mv_bmx_dst(raw: u64) -> Operand {
    if raw & 0x1 == 0b1 {
        let bm_reg = (raw >> 1) as u8;
        return Operand::AccumReg(bm_reg);
    }
    let x_reg = (raw >> 2) as u8;
    Operand::VectorReg(x_reg)
}

/// Inverse of `geteRS4OpValue`.
fn decode_ers4(raw: u64) -> Operand {
    let scl_reg = (raw as u8).wrapping_add(16);
    Operand::ScalarReg(scl_reg)
}

/// Inverse of `getmShflDstOpValue`.
fn decode_shfl_dst(raw: u64) -> Operand {
    if raw & 0x1 == 0b0 {
        let x_reg = (raw >> 1) as u8;
        return Operand::VectorReg(x_reg);
    }
    let shifted = raw >> 1;
    let lsb = ((shifted >> 3) & 0x1) as u8;
    let upper = (shifted & 0x7) as u8;
    let bms_reg = (upper << 1) | lsb;
    Operand::AccumReg(bms_reg)
}

/// Inverse of `getmWm_1OpValue`.
fn decode_wm1(raw: u64) -> Operand {
    let enc_upper = (raw & 0x7) as u8;
    let rearranged = ((raw >> 3) & 0x3) as u8;
    let bit0 = (rearranged >> 1) & 0x1;
    let bit1 = rearranged & 0x1;
    let vec_reg = (enc_upper << 2) | (bit1 << 1) | bit0;
    Operand::VectorReg(vec_reg)
}

/// Inverse of `getmQXHLbOpValue`.
fn decode_qxhlb(raw: u64) -> Operand {
    let reg = (raw >> 1) as u8;
    Operand::AccumReg(reg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_build_and_lookup() {
        let luts = CompositeLuts::build();

        // Verify known LdaScl decodings
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0000101),
            Operand::ScalarReg(LR_REG_INDEX)); // lr
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0001101),
            Operand::PointerReg(0)); // p0
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0000000),
            Operand::ScalarReg(0)); // r0
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0000100),
            Operand::ScalarReg(1)); // r1

        // Verify AluCg
        assert_eq!(luts.decode(CompositeEncoder::AluCg, 0b000001),
            Operand::ScalarReg(LC_REG_INDEX)); // LC
        assert_eq!(luts.decode(CompositeEncoder::AluCg, 0b000000),
            Operand::ScalarReg(0)); // r0
        assert_eq!(luts.decode(CompositeEncoder::AluCg, 0b000010),
            Operand::ScalarReg(1)); // r1

        // Verify ERS4
        assert_eq!(luts.decode(CompositeEncoder::ERS4, 0),
            Operand::ScalarReg(16)); // r16
        assert_eq!(luts.decode(CompositeEncoder::ERS4, 15),
            Operand::ScalarReg(31)); // r31
    }

    #[test]
    fn test_mv_scl_src_special_registers() {
        let luts = CompositeLuts::build();

        // Special registers via SPL tag (bits[2:0] == 0b111)
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 0b0000111),
            Operand::ScalarReg(LS_REG_INDEX)); // LS, id=0
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 0b0010111),
            Operand::ScalarReg(DP_REG_INDEX)); // DP, id=2
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 0b0100111),
            Operand::ScalarReg(LR_REG_INDEX)); // lr, id=4
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 0b0110111),
            Operand::ScalarReg(CORE_ID_REG_INDEX)); // CORE_ID, id=6
        assert_eq!(luts.decode(CompositeEncoder::MvSclSrc, 0b1100111),
            Operand::PointerReg(6)); // SP = p6, id=12
    }

    #[test]
    fn test_modifier_registers_with_base_offsets() {
        let luts = CompositeLuts::build();

        // LdaScl modifier registers should use correct base offsets
        // eM: m0 = raw (0 << 2) | 0b10 = 0b10
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0000010),
            Operand::ModifierReg(MOD_BASE_M)); // m0
        // eDN: dn0 = raw ((0 | 0b01000) << 2) | 0b10 = (8 << 2) | 2 = 34
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0100010),
            Operand::ModifierReg(MOD_BASE_DN)); // dn0
        // eDJ: dj0 = raw ((0 | 0b10000) << 2) | 0b10 = (16 << 2) | 2 = 66
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b1000010),
            Operand::ModifierReg(MOD_BASE_DJ)); // dj0
    }

    /// Validate LUT entries against Phase 1 register HWEncodings.
    ///
    /// This is the key conformance test: for every register with a known
    /// HWEncoding, verify that the forward encoder formula produces a
    /// composite value that our LUT inverts back to the correct register.
    #[test]
    fn test_lut_against_hwencodings() {
        let llvm_aie_path = std::path::Path::new("../llvm-aie");
        if !llvm_aie_path.exists() {
            eprintln!("Skipping test: llvm-aie not found");
            return;
        }

        let output = crate::tablegen::load_full_via_tblgen(llvm_aie_path)
            .expect("Failed to load tblgen data");
        let reg_model = &output.register_model;

        let luts = CompositeLuts::build();

        // Validate LdaScl for scalar registers (eR)
        if let Some(er_class) = reg_model.classes.get("eR") {
            for member in &er_class.members {
                if let Some(reg_def) = reg_model.registers.get(member) {
                    let hw = reg_def.hw_encoding;
                    // Forward formula: eR(enc) -> (enc << 2) | 0b00
                    let composite = (hw as u64) << 2;
                    let decoded = luts.decode(CompositeEncoder::LdaScl, composite);
                    assert_eq!(decoded, Operand::ScalarReg(hw as u8),
                        "LdaScl round-trip failed for {} (hw={})", member, hw);
                }
            }
        }

        // Validate LdaScl for pointer registers (eP)
        if let Some(ep_class) = reg_model.classes.get("eP") {
            for member in &ep_class.members {
                if let Some(reg_def) = reg_model.registers.get(member) {
                    let hw = reg_def.hw_encoding;
                    // Forward formula: eP(enc) -> (enc << 4) | 0b1101
                    let composite = ((hw as u64) << 4) | 0b1101;
                    let decoded = luts.decode(CompositeEncoder::LdaScl, composite);
                    assert_eq!(decoded, Operand::PointerReg(hw as u8),
                        "LdaScl round-trip failed for {} (hw={})", member, hw);
                }
            }
        }

        // Validate lr via LdaScl (fixed encoding 0b0000101)
        assert_eq!(luts.decode(CompositeEncoder::LdaScl, 0b0000101),
            Operand::ScalarReg(LR_REG_INDEX),
            "LdaScl lr encoding should map to LR_REG_INDEX");

        // Validate AluCg for scalar registers
        if let Some(er_class) = reg_model.classes.get("eR") {
            for member in &er_class.members {
                if let Some(reg_def) = reg_model.registers.get(member) {
                    let hw = reg_def.hw_encoding;
                    // Forward formula: eR(enc) -> enc << 1
                    let composite = (hw as u64) << 1;
                    let decoded = luts.decode(CompositeEncoder::AluCg, composite);
                    assert_eq!(decoded, Operand::ScalarReg(hw as u8),
                        "AluCg round-trip failed for {} (hw={})", member, hw);
                }
            }
        }

        eprintln!("LUT validation against HWEncodings: PASSED");
    }
}
