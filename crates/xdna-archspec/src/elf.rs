//! ELF format constants shared across AIE architectures.
//!
//! Toolchain-derived identifiers (from llvm-aie):
//! - `EM_AIE` is the ELF machine type number LLVM emits for AIE ELFs.
//! - `AieArchitecture` is the per-arch flag value in `e_flags`.
//!
//! Kept in archspec (not xdna-emu's parser) so a future AIE1
//! implementation populates its arch constants in the same place as
//! memory-map, DMA model, and ISA data. Subsystem 8 Task 4 migration
//! per docs/arch/subsys8-audit.md §Closing Summary.

/// ELF machine type for AIE cores.
///
/// See: llvm-aie llvm/include/llvm/BinaryFormat/ELF.h
pub const EM_AIE: u16 = 264; // 0x108

/// AIE architecture flags (e_flags field)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AieArchitecture {
    /// AIE1 - Original AI Engine (Versal)
    Aie1 = 0x01,
    /// AIE2 - AI Engine ML (Phoenix/HawkPoint NPU)
    Aie2 = 0x02,
    /// AIE2P - AI Engine ML+ (Strix/Krackan NPU)
    Aie2P = 0x03,
    /// Unknown architecture
    Unknown = 0x00,
}

impl From<u32> for AieArchitecture {
    fn from(flags: u32) -> Self {
        match flags & 0x0F {
            0x01 => AieArchitecture::Aie1,
            0x02 => AieArchitecture::Aie2,
            0x03 => AieArchitecture::Aie2P,
            _ => AieArchitecture::Unknown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_em_aie_constant() {
        assert_eq!(EM_AIE, 264);
        assert_eq!(EM_AIE, 0x108);
    }

    #[test]
    fn test_aie_architecture_from_flags() {
        assert_eq!(AieArchitecture::from(0x01), AieArchitecture::Aie1);
        assert_eq!(AieArchitecture::from(0x02), AieArchitecture::Aie2);
        assert_eq!(AieArchitecture::from(0x03), AieArchitecture::Aie2P);
        assert_eq!(AieArchitecture::from(0x00), AieArchitecture::Unknown);
        assert_eq!(AieArchitecture::from(0x12), AieArchitecture::Aie2); // Mask test
    }
}
