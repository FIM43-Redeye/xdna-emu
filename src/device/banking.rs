//! AIE2 Memory Banking Utilities
//!
//! Interleaved banking functions for memory conflict detection.
//! Bank counts and sizes are in `crate::arch` (generated from ArchModel).
//!
//! AIE2 uses 128-bit (16-byte) interleaved banking: consecutive 16-byte
//! lines map to different physical banks, enabling parallel access from
//! core load/store units and DMA engines.

/// Compute the bank index for a local memory address.
///
/// AIE2 uses interleaved banking at 128-bit (16-byte) boundaries.
/// Consecutive 16-byte lines map to different banks, enabling parallel
/// access to sequential addresses from separate load/store units.
///
/// `addr` is the byte offset within tile data memory.
/// `num_banks` is 8 for compute tiles, 16 for MemTiles.
#[inline]
pub fn addr_to_bank(addr: u32, num_banks: usize) -> u8 {
    ((addr as usize >> 4) % num_banks) as u8
}

/// Compute a bitmask of all banks touched by a memory access.
///
/// A 32-byte (256-bit) vector access spans two 128-bit bank rows and may
/// touch two different banks. This function returns a u16 bitmask with one
/// bit set per bank touched.
#[inline]
pub fn banks_for_access(addr: u32, bytes: usize, num_banks: usize) -> u16 {
    if bytes == 0 {
        return 0;
    }
    let mut mask = 0u16;
    // Align down to bank row boundary
    let start = (addr & !0xF) as usize;
    let end = (addr as usize) + bytes;
    let mut a = start;
    while a < end {
        let bank = (a >> 4) % num_banks;
        mask |= 1 << bank;
        a += 16;
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
        assert_eq!(crate::arch::compute::PHYSICAL_BANKS, 8);
        assert_eq!(crate::arch::compute::PHYSICAL_BANK_SIZE, 8 * 1024);
    }

    #[test]
    fn test_memtile_banking_constants() {
        assert_eq!(crate::arch::memtile::PHYSICAL_BANKS, 16);
        assert_eq!(crate::arch::memtile::PHYSICAL_BANK_SIZE, 32 * 1024);
        assert_eq!(
            crate::arch::memtile::PHYSICAL_BANKS as u64 * crate::arch::memtile::PHYSICAL_BANK_SIZE,
            512 * 1024
        );
    }

    #[test]
    fn test_addr_to_bank_interleaved() {
        // Consecutive 16-byte rows map to consecutive banks
        assert_eq!(addr_to_bank(0x00, 8), 0);
        assert_eq!(addr_to_bank(0x10, 8), 1);
        assert_eq!(addr_to_bank(0x20, 8), 2);
        assert_eq!(addr_to_bank(0x70, 8), 7);
        // Wraps around after 8 banks
        assert_eq!(addr_to_bank(0x80, 8), 0);
        assert_eq!(addr_to_bank(0x90, 8), 1);
    }

    #[test]
    fn test_addr_to_bank_within_row() {
        // All bytes within a 16-byte bank row map to the same bank
        for offset in 0..16 {
            assert_eq!(addr_to_bank(0x30 + offset, 8), 3);
        }
    }

    #[test]
    fn test_addr_to_bank_memtile_16_banks() {
        assert_eq!(addr_to_bank(0x00, 16), 0);
        assert_eq!(addr_to_bank(0xF0, 16), 15);
        assert_eq!(addr_to_bank(0x100, 16), 0); // wraps at 16
    }

    #[test]
    fn test_banks_for_access_scalar() {
        // 4-byte scalar load within one bank row -> one bank
        let mask = banks_for_access(0x00, 4, 8);
        assert_eq!(mask, 0b0000_0001); // bank 0

        let mask = banks_for_access(0x14, 4, 8);
        assert_eq!(mask, 0b0000_0010); // bank 1
    }

    #[test]
    fn test_banks_for_access_vector() {
        // 32-byte vector access at 0x00: spans banks 0 and 1
        let mask = banks_for_access(0x00, 32, 8);
        assert_eq!(mask, 0b0000_0011); // banks 0,1

        // 32-byte vector access at 0x70: spans banks 7 and 0 (wraps)
        let mask = banks_for_access(0x70, 32, 8);
        assert_eq!(mask, 0b1000_0001); // banks 7,0
    }

    #[test]
    fn test_banks_for_access_zero_bytes() {
        assert_eq!(banks_for_access(0x00, 0, 8), 0);
    }
}
