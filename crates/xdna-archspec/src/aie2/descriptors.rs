//! The 2D / 3D descriptor registers behind the `[p], d` addressing mode.
//!
//! An access whose post-modify names a descriptor takes the next step of an
//! address pattern instead of adding a fixed amount to the pointer. TableGen says which
//! registers a descriptor operand stands for:
//!
//! - a 2D descriptor `d<k>` (register class `eD`) is made of four registers: a modifier `m`,
//!   a size `dn`, a stride `dj` and a count `dc`, reached through the sub-register indices
//!   `sub_mod`, `sub_dim_size`, `sub_dim_stride` and `sub_dim_count`;
//! - a 3D descriptor `d<k>_3d` (class `eDS`) is a pair of 2D descriptors, an inner one
//!   (`sub_lo_dim`) and an outer one (`sub_hi_dim`).
//!
//! The tables are extracted at build time (`gen_descriptors.rs`) together with the list of
//! instruction records that take a descriptor, which is how the build and the tests prove
//! that every such instruction is classified [`AddressingMode::TwoDim`] or
//! [`AddressingMode::ThreeDim`]. What the update does with the registers (the rule observed
//! on NPU1 silicon) lives with the interpreter, in `execute::memory::agu`.
//!
//! This is the AIE2 layout. AIE2P gets its own table from its own TableGen; nothing here
//! should be read as holding for it.
//!
//! [`AddressingMode::TwoDim`]: super::isa::AddressingMode::TwoDim
//! [`AddressingMode::ThreeDim`]: super::isa::AddressingMode::ThreeDim

/// Which of the four registers of a 2D descriptor a sub-register is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModifierKind {
    /// `m<i>`: the amount added when the pattern wraps.
    Modifier,
    /// `dn<i>`: the size of the dimension.
    Size,
    /// `dj<i>`: the stride of the dimension.
    Stride,
    /// `dc<i>`: the running count in the dimension.
    Count,
}

/// A register of the modifier file: its kind and its index within that kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubRegister {
    pub kind: ModifierKind,
    pub index: u8,
}

/// A 2D descriptor `d<index>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Descriptor2D {
    pub register: &'static str,
    pub index: u8,
    pub modifier: SubRegister,
    pub size: SubRegister,
    pub stride: SubRegister,
    pub count: SubRegister,
}

/// A 3D descriptor `d<index>_3d`: the 2D descriptors `inner` and `outer`, by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Descriptor3D {
    pub register: &'static str,
    pub index: u8,
    pub inner: u8,
    pub outer: u8,
}

include!(concat!(env!("OUT_DIR"), "/gen_descriptors.rs"));

/// The 2D descriptor with this index.
pub fn descriptor_2d(index: u8) -> Option<&'static Descriptor2D> {
    DESCRIPTORS_2D.iter().find(|d| d.index == index)
}

/// The 3D descriptor with this index.
pub fn descriptor_3d(index: u8) -> Option<&'static Descriptor3D> {
    DESCRIPTORS_3D.iter().find(|d| d.index == index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aie2::isa::{load_from_generated, AddressingMode};

    #[test]
    fn a_2d_descriptor_is_the_four_registers_of_its_own_index() {
        assert_eq!(DESCRIPTORS_2D.len(), 8);
        for d in DESCRIPTORS_2D {
            assert_eq!(d.register, format!("d{}", d.index));
            for (sub, kind) in [
                (d.modifier, ModifierKind::Modifier),
                (d.size, ModifierKind::Size),
                (d.stride, ModifierKind::Stride),
                (d.count, ModifierKind::Count),
            ] {
                assert_eq!(sub, SubRegister { kind, index: d.index }, "{}", d.register);
            }
        }
    }

    #[test]
    fn a_3d_descriptor_is_an_inner_and_an_outer_2d_descriptor() {
        assert_eq!(DESCRIPTORS_3D.len(), 4);
        for d in DESCRIPTORS_3D {
            assert_eq!(d.register, format!("d{}_3d", d.index));
            assert_eq!((d.inner, d.outer), (d.index, d.index + 4), "{}", d.register);
            assert!(descriptor_2d(d.inner).is_some() && descriptor_2d(d.outer).is_some());
        }
    }

    #[test]
    fn lookups_by_index() {
        assert_eq!(descriptor_2d(5).unwrap().register, "d5");
        assert_eq!(descriptor_3d(2).unwrap().register, "d2_3d");
        assert!(descriptor_2d(8).is_none() && descriptor_3d(4).is_none());
    }

    /// Every TableGen record of a descriptor-addressed access is classified, and nothing
    /// else is. The build already fails when the addressing class and the operand register
    /// class disagree; this closes the loop from the record lists to the decoder tables.
    #[test]
    fn every_descriptor_access_record_is_classified_and_nothing_else_is() {
        let output = load_from_generated();
        let encodings: Vec<_> = output.encodings_by_slot.values().flatten().collect();
        for (records, mode) in
            [(ACCESS_RECORDS_2D, AddressingMode::TwoDim), (ACCESS_RECORDS_3D, AddressingMode::ThreeDim)]
        {
            assert!(!records.is_empty());
            for name in records {
                let enc = encodings
                    .iter()
                    .find(|e| e.name == *name)
                    .unwrap_or_else(|| panic!("{name}: no decoder encoding"));
                assert_eq!(enc.addressing_mode, mode, "{name}");
            }
            let classified: Vec<_> = encodings.iter().filter(|e| e.addressing_mode == mode).collect();
            assert_eq!(classified.len(), records.len(), "{mode:?}: classified encodings vs TableGen records");
            eprintln!("{mode:?}: {} TableGen records, all classified", records.len());
        }
    }

    /// Independent of the build-time check: the mnemonics say 2d / 3d exactly where the
    /// classification does (padda / paddb / padds included).
    #[test]
    fn mnemonics_agree_with_the_classification() {
        let output = load_from_generated();
        for enc in output.encodings_by_slot.values().flatten() {
            let dotted = |tag: &str| enc.mnemonic.split('.').any(|part| part == tag);
            let expected = if dotted("2d") {
                AddressingMode::TwoDim
            } else if dotted("3d") {
                AddressingMode::ThreeDim
            } else {
                continue;
            };
            assert_eq!(enc.addressing_mode, expected, "{} ({})", enc.name, enc.mnemonic);
        }
    }
}
