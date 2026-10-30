//! Per-form facts of the AIE2 unpack instructions.
//!
//! AIE2 widens packed integers two ways: the fused load `vldb.unpack.*` (a 256-bit
//! memory load widened into a 512-bit x register) and the standalone `vunpack.*` (a
//! 256-bit w register widened into a 512-bit x register). Each comes in four forms,
//! `s8.s4`, `s16.s8`, `d8.d4` and `d16.d8`, and the fused load in four addressing
//! flavours (index, post-modify, 2D, 3D).
//!
//! The table is extracted from llvm-aie TableGen at build time (`gen_unpack_forms.rs`):
//!
//! - lane widths from the mnemonic, cross-checked against the `sz` encoding bit;
//! - the sign source from `sgn` and `Uses`: a form that uses a `*UnpackSign` control
//!   register takes its signedness from that register at run time (the `d` forms,
//!   `sgn = 0`); every other form has it fixed by the encoding (the `s` forms, `sgn = 1`).
//!
//! The control register is data, not an assumption: AIE2P splits it into `unpackSign0`
//! and `unpackSign1`, so [`UnpackSign::Register`] names the register a form reads and
//! callers resolve its value through their own control-register file.
//!
//! What a sign value means, observed on NPU1 silicon (`vec_unpack_lane_*`,
//! `vec_unpack_crtiming`): the `s` forms always sign-extend whatever the register holds
//! (they do not read it); a register-driven form sign-extends when the register is 1 and
//! zero-extends when it is 0. The register is read at pipeline stage 7 (II_VUNPACK and
//! II_VLDB_UNPACK operand cycle 7), not at issue.

/// Fused load or register-to-register unpack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackKind {
    /// `vldb.unpack.*`: 256 bits from memory into a 512-bit x register.
    FusedLoad,
    /// `vunpack.*`: a 256-bit w register into a 512-bit x register.
    Standalone,
}

/// How a fused load updates its pointer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackAddressing {
    /// `vunpack`: no memory operand.
    Register,
    /// `[p, dj]` (index: the pointer is unchanged) or `[p], m` (post-modify: add an m
    /// register after the load). The two share a mnemonic; the decoder tells them apart.
    Linear,
    /// `[p], d`: load at the pointer, then a two-dimensional pointer update.
    TwoDim,
    /// `[p], d`: load at the pointer, then a three-dimensional pointer update.
    ThreeDim,
}

/// Where a form's signedness comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnpackSign {
    /// Fixed by the encoding.
    Fixed(bool),
    /// Read at run time from a control register: its TableGen name and hardware id.
    Register { name: &'static str, id: u8 },
}

impl UnpackSign {
    /// Resolve the signedness, reading the control register through `read` when the form
    /// takes it from one. A register value of 1 is signed, 0 unsigned (silicon: values 0
    /// and 1 only; higher bits are unobserved and ignored).
    pub fn is_signed(self, read: impl FnOnce(u8) -> u32) -> bool {
        match self {
            UnpackSign::Fixed(signed) => signed,
            UnpackSign::Register { id, .. } => read(id) & 1 != 0,
        }
    }
}

/// One unpack mnemonic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnpackForm {
    /// Assembly mnemonic (`vldb.unpack.s16.s8`).
    pub mnemonic: &'static str,
    pub kind: UnpackKind,
    pub addressing: UnpackAddressing,
    /// Packed (source) lane width in bits.
    pub bits_in: u8,
    /// Widened (destination) lane width in bits.
    pub bits_out: u8,
    pub sign: UnpackSign,
}

include!(concat!(env!("OUT_DIR"), "/gen_unpack_forms.rs"));

#[cfg(test)]
mod tests {
    use super::*;

    fn form(mnemonic: &str) -> &'static UnpackForm {
        unpack_form(mnemonic).unwrap_or_else(|| panic!("{mnemonic} is not an unpack mnemonic"))
    }

    #[test]
    fn every_form_of_both_families_is_present() {
        let fused = UNPACK_FORMS.iter().filter(|f| f.kind == UnpackKind::FusedLoad).count();
        let standalone = UNPACK_FORMS.iter().filter(|f| f.kind == UnpackKind::Standalone).count();
        // Four forms x three pointer-update flavours (linear, 2D, 3D), and four standalone forms.
        assert_eq!((fused, standalone), (12, 4));
    }

    #[test]
    fn widths_come_from_the_mnemonic() {
        let w = |m: &str| (form(m).bits_in, form(m).bits_out);
        assert_eq!(w("vunpack.s8.s4"), (4, 8));
        assert_eq!(w("vunpack.d16.d8"), (8, 16));
        assert_eq!(w("vldb.2d.unpack.s16.s8"), (8, 16));
        assert_eq!(w("vldb.3d.unpack.d8.d4"), (4, 8));
        assert_eq!(w("vldb.unpack.d16.d8"), (8, 16));
    }

    #[test]
    fn s_forms_are_fixed_signed_and_d_forms_read_crunpacksign() {
        for f in UNPACK_FORMS {
            if f.mnemonic.contains(".s") {
                assert_eq!(f.sign, UnpackSign::Fixed(true), "{}", f.mnemonic);
            } else {
                // crUnpackSign is control register 11 (AIE2GenRegisterInfo.td).
                assert_eq!(f.sign, UnpackSign::Register { name: "crUnpackSign", id: 11 }, "{}", f.mnemonic);
            }
        }
    }

    #[test]
    fn addressing_follows_the_mnemonic() {
        assert_eq!(form("vldb.unpack.s16.s8").addressing, UnpackAddressing::Linear);
        assert_eq!(form("vldb.2d.unpack.s8.s4").addressing, UnpackAddressing::TwoDim);
        assert_eq!(form("vldb.3d.unpack.d16.d8").addressing, UnpackAddressing::ThreeDim);
        assert_eq!(form("vunpack.d8.d4").addressing, UnpackAddressing::Register);
    }

    #[test]
    fn sign_resolution_reads_the_register_only_for_register_forms() {
        let reads = std::cell::Cell::new(0);
        let read = |v: u32| {
            let reads = &reads;
            move |_id: u8| {
                reads.set(reads.get() + 1);
                v
            }
        };
        assert!(UnpackSign::Fixed(true).is_signed(read(0)));
        assert_eq!(reads.get(), 0, "a fixed-sign form must not read the register");
        let reg = UnpackSign::Register { name: "crUnpackSign", id: 11 };
        assert!(reg.is_signed(read(1)));
        assert!(!reg.is_signed(read(0)));
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn non_unpack_mnemonics_are_not_forms() {
        assert!(unpack_form("vldb.128").is_none());
        assert!(unpack_form("vpack.s8.s16").is_none());
    }
}
