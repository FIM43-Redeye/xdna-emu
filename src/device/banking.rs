//! AIE2 Memory Banking Utilities
//!
//! Physical memory-bank layout and bank-conflict-detection helpers.
//! Bank counts and sizes are in `xdna_archspec::aie2` (generated from ArchModel).

/// Physical memory-bank layout of a tile's data memory.
///
/// AIE2 compute-tile data memory is 64 KB as eight 8 KB physical banks
/// (512 word x 128-bit, single-port). Every two physical banks are interleaved
/// at 16-byte granularity to form one contiguous 16 KB logical bank, giving the
/// four banks the compiler allocates in (AM020 ch.2:164; AIETargetModel
/// getNumBanks == 4 for compute tiles).
///
/// Arbitration is per PHYSICAL bank -- each has its own round-robin arbiter
/// (AM020 ch.2:166), and the hardware exposes eight CONFLICT_DM_BANK events.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BankLayout {
    Compute,
    /// MemTile geometry is NOT validated against hardware; preserved as-is.
    MemTile,
    /// Shim tiles have no local data memory banks.
    None,
}

/// Number of physical banks in a compute-tile data memory.
pub const COMPUTE_PHYSICAL_BANKS: u32 = xdna_archspec::aie2::compute::PHYSICAL_BANKS as u32;
/// Number of physical banks in a mem-tile data memory.
const MEMTILE_PHYSICAL_BANKS: u32 = xdna_archspec::aie2::memtile::PHYSICAL_BANKS as u32;
/// Size of one contiguous logical bank: a pair of interleaved physical banks
/// (AM020 ch.2:164). Derived from the physical bank size, never hardcoded.
const COMPUTE_LOGICAL_BANK_SHIFT: u32 =
    (2 * xdna_archspec::aie2::compute::PHYSICAL_BANK_SIZE).trailing_zeros();
/// Physical banks of a logical pair alternate every bank-width word (128 bits
/// = 16 bytes). Derived from the physical bank width.
const COMPUTE_INTERLEAVE_SHIFT: u32 =
    ((xdna_archspec::aie2::compute::PHYSICAL_BANK_WIDTH_BITS / 8) as u32).trailing_zeros();

impl BankLayout {
    /// Physical bank index for a tile-local byte offset.
    #[inline]
    pub fn physical_bank(&self, addr: u32) -> u8 {
        match self {
            BankLayout::Compute => {
                let logical = (addr >> COMPUTE_LOGICAL_BANK_SHIFT) & 0x3;
                let half = (addr >> COMPUTE_INTERLEAVE_SHIFT) & 0x1;
                (2 * logical + half) as u8
            }
            // Unvalidated: preserve the previous flat interleave for memtiles.
            BankLayout::MemTile => ((addr >> COMPUTE_INTERLEAVE_SHIFT) & 0xF) as u8,
            BankLayout::None => 0,
        }
    }

    /// Number of physical banks this layout arbitrates over.
    #[inline]
    pub fn num_banks(&self) -> u32 {
        match self {
            BankLayout::Compute => COMPUTE_PHYSICAL_BANKS,
            BankLayout::MemTile => MEMTILE_PHYSICAL_BANKS,
            BankLayout::None => 0,
        }
    }
}

/// Bitmask of every physical bank an access touches.
///
/// An access spans one or more 128-bit words; each word lives in one physical
/// bank, so a wide (vector) or unaligned access can touch several.
#[inline]
pub fn banks_for_access(addr: u32, bytes: usize, layout: BankLayout) -> u16 {
    if bytes == 0 || layout == BankLayout::None {
        return 0;
    }
    let mut mask = 0u16;
    let end = addr.saturating_add(bytes as u32);
    let mut word = addr & !0xF; // 128-bit word containing the first byte
    while word < end {
        mask |= 1 << layout.physical_bank(word);
        word += 16;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_max_value() {
        // Lock max value is now defined in Lock::MAX_VALUE (tile.rs)
        // and validated against the mlir-aie model at test time.
        assert_eq!(crate::device::tile::Lock::MAX_VALUE, 63);
    }

    #[test]
    fn test_compute_banking_constants() {
        assert_eq!(xdna_archspec::aie2::compute::PHYSICAL_BANKS, 8);
        assert_eq!(xdna_archspec::aie2::compute::PHYSICAL_BANK_SIZE, 8 * 1024);
        // Bank width: 128 bits per aie-rt XAIEMLGBL_MEMORY_MODULE_DATAMEMORY_WIDTH.
        assert_eq!(xdna_archspec::aie2::compute::PHYSICAL_BANK_WIDTH_BITS, 128);
        // The two shifts are derived, not hardcoded: 16 KB logical bank
        // (a pair of 8 KB physical banks) interleaved every 16-byte word.
        assert_eq!(COMPUTE_LOGICAL_BANK_SHIFT, 14);
        assert_eq!(COMPUTE_INTERLEAVE_SHIFT, 4);
    }

    #[test]
    fn test_memtile_banking_constants() {
        assert_eq!(xdna_archspec::aie2::memtile::PHYSICAL_BANKS, 16);
        assert_eq!(xdna_archspec::aie2::memtile::PHYSICAL_BANK_SIZE, 32 * 1024);
        assert_eq!(
            xdna_archspec::aie2::memtile::PHYSICAL_BANKS as u64
                * xdna_archspec::aie2::memtile::PHYSICAL_BANK_SIZE,
            512 * 1024
        );
        // Bank width: 128 bits per aie-rt XAIEMLGBL_MEM_TILE_MODULE_DATAMEMORY_WIDTH.
        assert_eq!(xdna_archspec::aie2::memtile::PHYSICAL_BANK_WIDTH_BITS, 128);
    }

    // AM020 ch.2:164 -- 64 KB as eight 8 KB single-port physical banks; every
    // two are interleaved (16-byte granularity) into one 16 KB logical bank.
    // Confirmed by HW: of_q0_rich buffers at 0x400..0x5ff fire CONFLICT_DM_BANK
    // on physical banks 0 and 1 only, near-evenly.
    #[test]
    fn compute_physical_bank_interleaves_pair_every_16_bytes() {
        assert_eq!(BankLayout::Compute.physical_bank(0x0000), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x0010), 1);
        assert_eq!(BankLayout::Compute.physical_bank(0x0020), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x0030), 1);
        // within a 16-byte word the bank does not change
        assert_eq!(BankLayout::Compute.physical_bank(0x0004), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x001C), 1);
    }

    #[test]
    fn compute_logical_banks_are_contiguous_16kb() {
        // logical 0 -> physical {0,1}; logical 1 -> {2,3}; 2 -> {4,5}; 3 -> {6,7}
        assert_eq!(BankLayout::Compute.physical_bank(0x0000), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x4000), 2);
        assert_eq!(BankLayout::Compute.physical_bank(0x4010), 3);
        assert_eq!(BankLayout::Compute.physical_bank(0x8000), 4);
        assert_eq!(BankLayout::Compute.physical_bank(0xC000), 6);
        assert_eq!(BankLayout::Compute.physical_bank(0xC010), 7);
    }

    #[test]
    fn repro_kernel_buffers_land_in_banks_0_and_1_only() {
        // of_q0_rich consumer buffers: 0x400..0x5ff (ei/eo). HW fired conflicts
        // on banks 0 and 1 ONLY -- banks 2-7 silent.
        let mut seen = 0u16;
        for addr in (0x400u32..0x600).step_by(16) {
            seen |= 1 << BankLayout::Compute.physical_bank(addr);
        }
        assert_eq!(seen, 0b0000_0011, "buffers must occupy exactly banks 0 and 1");
    }

    #[test]
    fn banks_for_access_covers_every_16_byte_word_touched() {
        // a 4-byte scalar access touches one bank
        assert_eq!(banks_for_access(0x400, 4, BankLayout::Compute), 1 << 0);
        // a 32-byte vector access spans two 16-byte words -> two physical banks
        assert_eq!(banks_for_access(0x400, 32, BankLayout::Compute), (1 << 0) | (1 << 1));
        // unaligned access straddling a 16-byte boundary touches both
        assert_eq!(banks_for_access(0x40C, 8, BankLayout::Compute), (1 << 0) | (1 << 1));
    }
}
