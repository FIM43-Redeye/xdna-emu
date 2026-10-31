//! Per-form facts of the AIE2 pack instructions.
//!
//! AIE2 narrows integers two ways: the fused store `vst.pack.*` (a 512-bit x register
//! narrowed into 256 bits of memory) and the standalone `vpack.*` (a 512-bit x register
//! narrowed into a 256-bit w register). Each comes in four forms, `s8.s16`, `s4.s8`, `d8.d16`
//! and `d4.d8`, and the fused store in four addressing flavours (index, post-modify, 2D, 3D).
//!
//! The table is extracted from llvm-aie TableGen at build time (`gen_pack_forms.rs`):
//!
//! - lane widths from the mnemonic, cross-checked against the `sz` encoding bit;
//! - the sign source from `sgn` and `Uses`: a form that uses a `*PackSign` control register
//!   takes its signedness from that register at run time (the `d` forms, `sgn = 0`); every
//!   other form has it fixed by the encoding (the `s` forms, `sgn = 1`). The sign source is
//!   derived from `Uses`, never from the `d` / `s` letter.
//!
//! The extraction fails the build if a record named like a pack (`VPACK_*`, `VST_PACK_*`,
//! `VST_2D_PACK_*`, `VST_3D_PACK_*`) is not reached through the two encoding classes or the
//! reverse, and [`PACK_RECORDS`] lists every record the table covers, so tests can check the
//! table against the instruction set. Only AIE2 is described: AIE2P has a different layout
//! (`vpack` with `packSign0` / `packSign1` and a `crPackSize` register).
//!
//! What the sign and saturation registers mean, observed on NPU1 silicon (`vec_pack_lane_*`,
//! `vec_pack_addr`, `vec_pack_crtiming`):
//!
//! - lane `i` of the narrowed result is lane `i` of the source, `bits_in` wide; 4-bit lanes
//!   are packed low nibble first; the fused store and the standalone instruction agree for
//!   every form, value and register setting;
//! - the sign decides how the input lane is read and which clamp range it gets: a signed
//!   lane is read two's complement and clamps to the signed output range, an unsigned lane
//!   is read as an unsigned number and clamps to `[0, 2^bits_out - 1]`. The `s` forms are
//!   signed; a `d` form is signed when `crPackSign` is 1 and unsigned when it is 0;
//! - `crSat` selects the narrowing: see `PackMode` in the emulator's `vector_pack` for the
//!   per-value table (signed and unsigned differ at `crSat = 2`).

use super::sign::SignSource;

/// Fused store or register-to-register pack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackKind {
    /// `vst.pack.*`: a 512-bit x register narrowed into 256 bits of memory.
    FusedStore,
    /// `vpack.*`: a 512-bit x register narrowed into a 256-bit w register.
    Standalone,
}

/// How a fused store updates its pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackAddressing {
    /// `vpack`: no memory operand.
    Register,
    /// `[p, #imm]`, `[p, dj]` (index: the pointer is unchanged) or `[p], m` / `[p], #imm`
    /// (post-modify). The four share a mnemonic; the decoder tells them apart.
    Linear,
    /// `[p], d`: store at the pointer, then a two-dimensional pointer update.
    TwoDim,
    /// `[p], d`: store at the pointer, then a three-dimensional pointer update.
    ThreeDim,
}

/// One pack mnemonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackForm {
    /// Assembly mnemonic (`vst.pack.d8.d16`).
    pub mnemonic: &'static str,
    pub kind: PackKind,
    pub addressing: PackAddressing,
    /// Source lane width in bits.
    pub bits_in: u8,
    /// Narrowed (destination) lane width in bits.
    pub bits_out: u8,
    pub sign: SignSource,
}

include!(concat!(env!("OUT_DIR"), "/gen_pack_forms.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    fn form(mnemonic: &str) -> &'static PackForm {
        pack_form(mnemonic).unwrap_or_else(|| panic!("{mnemonic} is not a pack mnemonic"))
    }

    #[test]
    fn every_form_of_both_families_is_present() {
        let fused = PACK_FORMS.iter().filter(|f| f.kind == PackKind::FusedStore).count();
        let standalone = PACK_FORMS.iter().filter(|f| f.kind == PackKind::Standalone).count();
        // Four forms x three pointer-update flavours (linear, 2D, 3D), and four standalone forms.
        assert_eq!((fused, standalone), (12, 4));
    }

    #[test]
    fn every_tablegen_record_is_covered() {
        // 4 standalone + 4 forms x (idx_imm, idx, pstm_nrm, pstm_nrm_imm, 2d, 3d).
        assert_eq!(PACK_RECORDS.len(), 4 + 4 * 6);
        for (record, mnemonic) in PACK_RECORDS {
            assert!(pack_form(mnemonic).is_some(), "{record}: {mnemonic} is not in the table");
        }
        let mut names: Vec<&str> = PACK_RECORDS.iter().map(|r| r.0).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), PACK_RECORDS.len(), "a record is listed twice");
    }

    #[test]
    fn widths_come_from_the_mnemonic() {
        let w = |m: &str| (form(m).bits_in, form(m).bits_out);
        assert_eq!(w("vpack.s4.s8"), (8, 4));
        assert_eq!(w("vpack.d8.d16"), (16, 8));
        assert_eq!(w("vst.pack.s8.s16"), (16, 8));
        assert_eq!(w("vst.2d.pack.d4.d8"), (8, 4));
        assert_eq!(w("vst.3d.pack.d8.d16"), (16, 8));
    }

    #[test]
    fn s_forms_are_fixed_signed_and_d_forms_read_crpacksign() {
        for f in PACK_FORMS {
            if f.mnemonic.contains(".s") {
                assert_eq!(f.sign, SignSource::Fixed(true), "{}", f.mnemonic);
            } else {
                // crPackSign is control register 5 (AIE2GenRegisterInfo.td).
                assert_eq!(f.sign, SignSource::Register { name: "crPackSign", id: 5 }, "{}", f.mnemonic);
            }
        }
    }

    #[test]
    fn addressing_follows_the_mnemonic() {
        assert_eq!(form("vst.pack.s8.s16").addressing, PackAddressing::Linear);
        assert_eq!(form("vst.2d.pack.s4.s8").addressing, PackAddressing::TwoDim);
        assert_eq!(form("vst.3d.pack.d8.d16").addressing, PackAddressing::ThreeDim);
        assert_eq!(form("vpack.d4.d8").addressing, PackAddressing::Register);
        assert_eq!(form("vpack.d4.d8").kind, PackKind::Standalone);
    }

    #[test]
    fn non_pack_mnemonics_are_not_forms() {
        assert!(pack_form("vst.srs.s16.s32").is_none());
        assert!(pack_form("vunpack.s16.s8").is_none());
        assert!(pack_form("vst.unpack.s16.s8").is_none());
    }
}
