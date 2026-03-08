//! Memory timing model for AIE2.
//!
//! Models memory bank conflicts, access latencies, and alignment per AM020.
//!
//! # Memory Architecture (AM020 Ch2, Ch4)
//!
//! - Data memory per tile: 64 KB
//! - Organized as 8 physical banks of 8 KB each
//! - From programmer's view: 4 logical banks of 16 KB (pairs interleaved)
//! - Bank width: 128 bits (16 bytes)
//! - Two 256-bit load ports + one 256-bit store port
//! - Base latency: 5 cycles
//!
//! # Bank Conflict Rules (AM020 Ch2)
//!
//! - Same bank accessed by multiple ports: stall 1 cycle
//! - Different banks: parallel access (no stall)
//! - Round-robin arbitration on conflict
//! - "Only one request per cycle is allowed to access the memory"
//!
//! # Bank Mapping
//!
//! Address bits determine physical bank:
//! - Bits [3:0]: byte within 16-byte block (128-bit word)
//! - Bits [6:4]: physical bank select (0-7)
//! - Bits [15:7]: row within bank
//!
//! Physical banks are paired into logical banks:
//! - Physical 0,1 -> Logical 0
//! - Physical 2,3 -> Logical 1
//! - Physical 4,5 -> Logical 2
//! - Physical 6,7 -> Logical 3
//!
//! # Alignment Requirements (AM020 Ch4)
//!
//! - "Two 256-bit load and one 256-bit store units with aligned addresses"
//! - 256-bit (32-byte) vector: requires 32-byte alignment
//! - 32-bit word: requires 4-byte alignment
//! - 16-bit halfword: requires 2-byte alignment
//! - Unaligned access is undefined behavior (may fault or produce wrong results)

/// Number of physical memory banks per compute tile (for conflict detection).
pub const NUM_BANKS: usize = crate::arch::compute::PHYSICAL_BANKS as usize;

// ============================================================================
// Cross-Tile Memory Access Latency (AM020 Ch4)
// ============================================================================
//
// Each compute tile can access 256KB of data memory across 4 tiles:
// - Quadrant 0 (0x00000-0x0FFFF): Local tile memory (64KB)
// - Quadrant 1 (0x10000-0x1FFFF): West neighbor tile (col-1, same row)
// - Quadrant 2 (0x20000-0x2FFFF): North neighbor tile (same col, row+1)
// - Quadrant 3 (0x30000-0x3FFFF): East neighbor tile (col+1, same row)
//
// Note: the hardware's AGU quadrant mapping per aie-rt and mlir-aie is
// actually 0=South, 1=West, 2=North, 3=East(local). The emulator uses
// quadrant 0 as local because linker-assigned addresses (0x70000+) are
// masked to 16-bit offsets landing in quadrant 0 (see decode_data_address
// in execute/memory.rs). This simplified mapping works for all current
// code paths.
//
// Cross-tile memory access incurs routing latency through the stream switch.
// Per AM020 Ch2, local-to-external routing is 4 cycles per hop.
//

/// Size of each memory quadrant (64KB = compute tile data memory).
pub const QUADRANT_SIZE: u32 = crate::arch::compute::MEMORY_SIZE as u32;

/// Latency for accessing neighbor tile memory (1 hop).
/// Per AM020 Ch2: local-to-external routing is 4 cycles.
pub const CROSS_TILE_LATENCY: u8 = crate::arch::timing::ROUTE_LOCAL_TO_EXTERNAL;

/// Memory quadrant (which tile's memory is being accessed).
///
/// Mapping matches execute/memory.rs NeighborMemory and decode_data_address().
/// See comment block above for the relationship to the hardware's actual
/// AGU quadrant numbering (South/West/North/East per aie-rt).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryQuadrant {
    /// Local tile memory (0x00000-0x0FFFF) - no routing latency.
    Local,
    /// West neighbor tile (0x10000-0x1FFFF) - 1 hop (col-1, same row).
    West,
    /// North neighbor tile (0x20000-0x2FFFF) - 1 hop (same col, row+1).
    North,
    /// East neighbor tile (0x30000-0x3FFFF) - 1 hop (col+1, same row).
    East,
}

impl MemoryQuadrant {
    /// Determine which quadrant an address falls into.
    ///
    /// Address bits [17:16] determine the quadrant:
    /// - 0b00: Local (own tile memory)
    /// - 0b01: West neighbor (col-1, same row)
    /// - 0b10: North neighbor (same col, row+1)
    /// - 0b11: East neighbor (col+1, same row)
    #[inline]
    pub fn from_address(address: u32) -> Self {
        match (address >> 16) & 0x3 {
            0 => Self::Local,
            1 => Self::West,
            2 => Self::North,
            3 => Self::East,
            _ => unreachable!(),
        }
    }

    /// Get the number of hops to reach this quadrant.
    ///
    /// All cross-tile neighbors (West, North, East) are 1 hop away.
    #[inline]
    pub fn hop_count(&self) -> u8 {
        match self {
            Self::Local => 0,
            Self::West | Self::North | Self::East => 1,
        }
    }

    /// Get the routing latency for accessing this quadrant.
    ///
    /// Based on AM020 Ch2 stream switch latencies:
    /// - Local: 0 cycles (no routing)
    /// - 1 hop: 4 cycles (local-to-external)
    /// - 2 hops: 8 cycles (4 per hop)
    #[inline]
    pub fn routing_latency(&self) -> u8 {
        self.hop_count() * CROSS_TILE_LATENCY
    }

    /// Check if this is a cross-tile access.
    #[inline]
    pub fn is_cross_tile(&self) -> bool {
        !matches!(self, Self::Local)
    }
}

/// Size of each bank in bytes.
pub const BANK_SIZE: usize = crate::arch::compute::PHYSICAL_BANK_SIZE as usize;

/// Bank width in bytes (128 bits = 16 bytes).
pub const BANK_WIDTH_BYTES: usize = 128 / 8; // TODO: generate from ArchModel bank_width_bits

/// Base memory access latency (cycles).
pub const BASE_LATENCY: u8 = crate::arch::timing::DATA_MEMORY_LATENCY;

/// Additional stall cycles on bank conflict.
pub const CONFLICT_PENALTY: u8 = 1;

/// Alignment error when access doesn't meet alignment requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlignmentError {
    /// The unaligned address.
    pub address: u32,
    /// The access width in bytes.
    pub width: u8,
    /// Required alignment in bytes.
    pub required_alignment: u8,
}

impl std::fmt::Display for AlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unaligned access: address 0x{:08X} with width {} requires {}-byte alignment",
            self.address, self.width, self.required_alignment
        )
    }
}

/// Memory access descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryAccess {
    /// Starting address.
    pub address: u32,
    /// Access width in bytes.
    pub width: u8,
    /// Is this a write (true) or read (false)?
    pub is_write: bool,
    /// Port index (0 = load A, 1 = load B, 2 = store).
    pub port: u8,
}

impl MemoryAccess {
    /// Create a load access.
    pub fn load(address: u32, width: u8) -> Self {
        Self {
            address,
            width,
            is_write: false,
            port: 0, // Default to port 0
        }
    }

    /// Create a store access.
    pub fn store(address: u32, width: u8) -> Self {
        Self {
            address,
            width,
            is_write: true,
            port: 2, // Store port
        }
    }

    /// Get the required alignment for this access width.
    ///
    /// Per AM020 Ch4:
    /// - 256-bit (32-byte) vector: 32-byte alignment
    /// - 128-bit (16-byte) quadword: 16-byte alignment
    /// - 64-bit (8-byte) doubleword: 8-byte alignment
    /// - 32-bit (4-byte) word: 4-byte alignment
    /// - 16-bit (2-byte) halfword: 2-byte alignment
    /// - 8-bit byte: no alignment required
    #[inline]
    pub fn required_alignment(&self) -> u8 {
        match self.width {
            32 => 32, // 256-bit vector
            16 => 16, // 128-bit quadword
            8 => 8,   // 64-bit doubleword
            4 => 4,   // 32-bit word
            2 => 2,   // 16-bit halfword
            _ => 1,   // byte or unknown
        }
    }

    /// Check if the access is properly aligned.
    #[inline]
    pub fn is_aligned(&self) -> bool {
        let alignment = self.required_alignment() as u32;
        (self.address & (alignment - 1)) == 0
    }

    /// Check alignment and return error if unaligned.
    pub fn check_alignment(&self) -> Result<(), AlignmentError> {
        if self.is_aligned() {
            Ok(())
        } else {
            Err(AlignmentError {
                address: self.address,
                width: self.width,
                required_alignment: self.required_alignment(),
            })
        }
    }

    /// Get the physical bank index for this access (0-7).
    ///
    /// Bank is determined by bits [6:4] of address.
    /// Each bank is 8 KB (512 × 128-bit words).
    #[inline]
    pub fn bank(&self) -> u8 {
        ((self.address >> 4) & 0x7) as u8
    }

    /// Get the logical bank index (0-3).
    ///
    /// Physical banks are paired: 0+1, 2+3, 4+5, 6+7 form logical banks.
    #[inline]
    pub fn logical_bank(&self) -> u8 {
        self.bank() >> 1
    }

    /// Get all physical banks touched by this access (for wide accesses).
    ///
    /// A 256-bit (32-byte) vector access touches 2 consecutive banks
    /// since each bank row is 128 bits (16 bytes).
    pub fn banks_touched(&self) -> Vec<u8> {
        let start_bank = self.bank();
        let bytes_per_bank = BANK_WIDTH_BYTES as u32;

        // How many banks does this access span?
        let start_offset = self.address & (bytes_per_bank - 1);
        let end_offset = start_offset + self.width as u32;
        let num_banks = ((end_offset + bytes_per_bank - 1) / bytes_per_bank) as u8;

        (0..num_banks)
            .map(|i| (start_bank + i) % NUM_BANKS as u8)
            .collect()
    }

    /// Get the memory quadrant for this access.
    ///
    /// Determines which of the 4 tiles (local or neighbor) this address
    /// falls into based on address bits [17:16].
    #[inline]
    pub fn quadrant(&self) -> MemoryQuadrant {
        MemoryQuadrant::from_address(self.address)
    }

    /// Get the cross-tile routing latency for this access.
    ///
    /// Returns 0 for local tile access, or 4-8 cycles for neighbor tiles.
    #[inline]
    pub fn cross_tile_latency(&self) -> u8 {
        self.quadrant().routing_latency()
    }

    /// Check if this is a cross-tile memory access.
    #[inline]
    pub fn is_cross_tile(&self) -> bool {
        self.quadrant().is_cross_tile()
    }
}

/// Result of bank conflict detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BankConflict {
    /// Number of stall cycles due to conflicts.
    pub stall_cycles: u8,
    /// Which banks had conflicts (bitmask).
    pub conflict_banks: u8,
}

impl BankConflict {
    /// No conflict.
    pub const NONE: Self = Self {
        stall_cycles: 0,
        conflict_banks: 0,
    };

    /// Check if there was any conflict.
    #[inline]
    pub fn has_conflict(&self) -> bool {
        self.stall_cycles > 0
    }
}

/// Memory timing model.
///
/// Tracks per-cycle bank usage to detect conflicts and cross-tile latency.
#[derive(Debug, Clone)]
pub struct MemoryModel {
    /// Last cycle each bank was accessed.
    bank_last_access: [u64; NUM_BANKS],

    /// Current cycle (for conflict detection).
    current_cycle: u64,

    /// Statistics: total bank conflicts.
    total_conflicts: u64,

    /// Statistics: total memory accesses.
    total_accesses: u64,

    /// Statistics: total cross-tile accesses.
    total_cross_tile: u64,

    /// Statistics: total cross-tile latency cycles.
    total_cross_tile_cycles: u64,
}

/// Sentinel value indicating "bank never accessed".
/// Using u64::MAX ensures no false conflicts at cycle 0.
const NEVER_ACCESSED: u64 = u64::MAX;

impl MemoryModel {
    /// Create a new memory model.
    pub fn new() -> Self {
        Self {
            bank_last_access: [NEVER_ACCESSED; NUM_BANKS],
            current_cycle: 0,
            total_conflicts: 0,
            total_accesses: 0,
            total_cross_tile: 0,
            total_cross_tile_cycles: 0,
        }
    }

    /// Advance the model to the specified cycle.
    pub fn advance_to(&mut self, cycle: u64) {
        self.current_cycle = cycle;
    }

    /// Check for bank conflicts without recording the access.
    pub fn check_conflict(&self, access: &MemoryAccess) -> BankConflict {
        let banks = access.banks_touched();
        let mut conflict_banks = 0u8;
        let mut max_stall = 0u8;

        for bank in banks {
            let bank_idx = bank as usize;
            if bank_idx < NUM_BANKS {
                let last = self.bank_last_access[bank_idx];
                if last == self.current_cycle {
                    // Bank was already accessed this cycle
                    conflict_banks |= 1 << bank;
                    max_stall = max_stall.max(CONFLICT_PENALTY);
                }
            }
        }

        BankConflict {
            stall_cycles: max_stall,
            conflict_banks,
        }
    }

    /// Record a memory access and return any conflict penalty.
    pub fn record_access(&mut self, access: &MemoryAccess) -> BankConflict {
        let conflict = self.check_conflict(access);

        // Mark banks as accessed
        let banks = access.banks_touched();
        for bank in banks {
            let bank_idx = bank as usize;
            if bank_idx < NUM_BANKS {
                self.bank_last_access[bank_idx] = self.current_cycle;
            }
        }

        // Update statistics
        self.total_accesses += 1;
        if conflict.has_conflict() {
            self.total_conflicts += 1;
        }

        // Track cross-tile access statistics
        if access.is_cross_tile() {
            self.total_cross_tile += 1;
            self.total_cross_tile_cycles += access.cross_tile_latency() as u64;
        }

        conflict
    }

    /// Check conflicts for multiple accesses in the same cycle.
    ///
    /// Returns total stall cycles needed for all accesses.
    pub fn check_conflicts(&self, accesses: &[MemoryAccess]) -> BankConflict {
        let mut used_banks = 0u8;
        let mut conflict_banks = 0u8;
        let mut stall_cycles = 0u8;

        for access in accesses {
            for bank in access.banks_touched() {
                let bank_mask = 1u8 << bank;
                if used_banks & bank_mask != 0 {
                    // This bank already used by another access in this cycle
                    conflict_banks |= bank_mask;
                    stall_cycles += CONFLICT_PENALTY;
                }
                used_banks |= bank_mask;
            }
        }

        // Also check against previous cycle's accesses
        for access in accesses {
            let prev_conflict = self.check_conflict(access);
            conflict_banks |= prev_conflict.conflict_banks;
            stall_cycles = stall_cycles.saturating_add(prev_conflict.stall_cycles);
        }

        BankConflict {
            stall_cycles,
            conflict_banks,
        }
    }

    /// Get total latency for a memory access (base + conflict penalty + cross-tile).
    ///
    /// Total latency includes:
    /// - Base memory access latency (5 cycles)
    /// - Bank conflict penalty (if any)
    /// - Cross-tile routing latency (0, 4, or 8 cycles based on quadrant)
    pub fn access_latency(&self, access: &MemoryAccess) -> u8 {
        let conflict = self.check_conflict(access);
        let cross_tile = access.cross_tile_latency();
        BASE_LATENCY
            .saturating_add(conflict.stall_cycles)
            .saturating_add(cross_tile)
    }

    /// Get just the cross-tile routing latency for an access.
    pub fn cross_tile_latency(&self, access: &MemoryAccess) -> u8 {
        access.cross_tile_latency()
    }

    /// Reset the model.
    pub fn reset(&mut self) {
        self.bank_last_access = [NEVER_ACCESSED; NUM_BANKS];
        self.current_cycle = 0;
        self.total_conflicts = 0;
        self.total_accesses = 0;
        self.total_cross_tile = 0;
        self.total_cross_tile_cycles = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_accesses: self.total_accesses,
            total_conflicts: self.total_conflicts,
            conflict_rate: if self.total_accesses > 0 {
                self.total_conflicts as f64 / self.total_accesses as f64
            } else {
                0.0
            },
            total_cross_tile: self.total_cross_tile,
            total_cross_tile_cycles: self.total_cross_tile_cycles,
            cross_tile_rate: if self.total_accesses > 0 {
                self.total_cross_tile as f64 / self.total_accesses as f64
            } else {
                0.0
            },
        }
    }
}

impl Default for MemoryModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory access statistics.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    /// Total memory accesses.
    pub total_accesses: u64,
    /// Total bank conflicts.
    pub total_conflicts: u64,
    /// Bank conflict rate (0.0 - 1.0).
    pub conflict_rate: f64,
    /// Total cross-tile memory accesses.
    pub total_cross_tile: u64,
    /// Total cycles spent on cross-tile routing latency.
    pub total_cross_tile_cycles: u64,
    /// Cross-tile access rate (0.0 - 1.0).
    pub cross_tile_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_calculation() {
        // Address 0x00 should be bank 0
        let access = MemoryAccess::load(0x00, 4);
        assert_eq!(access.bank(), 0);

        // Address 0x10 (16) should be bank 1
        let access = MemoryAccess::load(0x10, 4);
        assert_eq!(access.bank(), 1);

        // Address 0x70 (112) should be bank 7
        let access = MemoryAccess::load(0x70, 4);
        assert_eq!(access.bank(), 7);

        // Address 0x80 (128) should wrap to bank 0
        let access = MemoryAccess::load(0x80, 4);
        assert_eq!(access.bank(), 0);
    }

    #[test]
    fn test_no_conflict() {
        let mut model = MemoryModel::new();
        model.advance_to(100);

        // Access bank 0
        let access1 = MemoryAccess::load(0x00, 4);
        let conflict1 = model.record_access(&access1);
        assert!(!conflict1.has_conflict());

        // Access bank 1 (different bank, no conflict)
        let access2 = MemoryAccess::load(0x10, 4);
        let conflict2 = model.record_access(&access2);
        assert!(!conflict2.has_conflict());
    }

    #[test]
    fn test_bank_conflict() {
        let mut model = MemoryModel::new();
        model.advance_to(100);

        // Access bank 0
        let access1 = MemoryAccess::load(0x00, 4);
        let conflict1 = model.record_access(&access1);
        assert!(!conflict1.has_conflict());

        // Access bank 0 again in same cycle (conflict!)
        let access2 = MemoryAccess::load(0x80, 4); // Also bank 0
        let conflict2 = model.record_access(&access2);
        assert!(conflict2.has_conflict());
        assert_eq!(conflict2.stall_cycles, CONFLICT_PENALTY);
    }

    #[test]
    fn test_wide_access_banks() {
        // 32-byte (256-bit) vector access starting at address 0
        // Should touch banks 0 and 1 (16 bytes each)
        let access = MemoryAccess::load(0x00, 32);
        let banks = access.banks_touched();
        assert_eq!(banks.len(), 2);
        assert!(banks.contains(&0));
        assert!(banks.contains(&1));
    }

    #[test]
    fn test_access_latency() {
        let model = MemoryModel::new();

        let access = MemoryAccess::load(0x00, 4);
        let latency = model.access_latency(&access);
        assert_eq!(latency, BASE_LATENCY);
    }

    #[test]
    fn test_stats() {
        let mut model = MemoryModel::new();
        model.advance_to(1);

        // Two accesses to same bank
        model.record_access(&MemoryAccess::load(0x00, 4));
        model.record_access(&MemoryAccess::load(0x80, 4));

        let stats = model.stats();
        assert_eq!(stats.total_accesses, 2);
        assert_eq!(stats.total_conflicts, 1);
    }

    // ========== Alignment Tests ==========

    #[test]
    fn test_alignment_requirements() {
        // Byte access - no alignment needed
        assert_eq!(MemoryAccess::load(0x01, 1).required_alignment(), 1);

        // Halfword - 2-byte alignment
        assert_eq!(MemoryAccess::load(0x00, 2).required_alignment(), 2);

        // Word - 4-byte alignment
        assert_eq!(MemoryAccess::load(0x00, 4).required_alignment(), 4);

        // Doubleword - 8-byte alignment
        assert_eq!(MemoryAccess::load(0x00, 8).required_alignment(), 8);

        // Quadword - 16-byte alignment
        assert_eq!(MemoryAccess::load(0x00, 16).required_alignment(), 16);

        // Vector256 - 32-byte alignment
        assert_eq!(MemoryAccess::load(0x00, 32).required_alignment(), 32);
    }

    #[test]
    fn test_aligned_access() {
        // Word at 4-byte aligned address
        let access = MemoryAccess::load(0x100, 4);
        assert!(access.is_aligned());
        assert!(access.check_alignment().is_ok());

        // Vector at 32-byte aligned address
        let access = MemoryAccess::load(0x20, 32);
        assert!(access.is_aligned());
        assert!(access.check_alignment().is_ok());

        // Byte at any address
        let access = MemoryAccess::load(0x33, 1);
        assert!(access.is_aligned());
    }

    #[test]
    fn test_unaligned_access() {
        // Word at unaligned address
        let access = MemoryAccess::load(0x101, 4);
        assert!(!access.is_aligned());
        let err = access.check_alignment().unwrap_err();
        assert_eq!(err.address, 0x101);
        assert_eq!(err.width, 4);
        assert_eq!(err.required_alignment, 4);

        // Vector at 16-byte aligned but not 32-byte aligned
        let access = MemoryAccess::load(0x10, 32);
        assert!(!access.is_aligned());

        // Halfword at odd address
        let access = MemoryAccess::load(0x11, 2);
        assert!(!access.is_aligned());
    }

    #[test]
    fn test_logical_bank() {
        // Physical banks 0,1 -> logical 0
        assert_eq!(MemoryAccess::load(0x00, 4).logical_bank(), 0);
        assert_eq!(MemoryAccess::load(0x10, 4).logical_bank(), 0);

        // Physical banks 2,3 -> logical 1
        assert_eq!(MemoryAccess::load(0x20, 4).logical_bank(), 1);
        assert_eq!(MemoryAccess::load(0x30, 4).logical_bank(), 1);

        // Physical banks 4,5 -> logical 2
        assert_eq!(MemoryAccess::load(0x40, 4).logical_bank(), 2);
        assert_eq!(MemoryAccess::load(0x50, 4).logical_bank(), 2);

        // Physical banks 6,7 -> logical 3
        assert_eq!(MemoryAccess::load(0x60, 4).logical_bank(), 3);
        assert_eq!(MemoryAccess::load(0x70, 4).logical_bank(), 3);
    }

    #[test]
    fn test_vector_banks_touched() {
        // 256-bit vector at address 0 touches banks 0 and 1
        let access = MemoryAccess::load(0x00, 32);
        let banks = access.banks_touched();
        assert_eq!(banks.len(), 2);
        assert_eq!(banks[0], 0);
        assert_eq!(banks[1], 1);

        // 256-bit vector at address 0x20 touches banks 2 and 3
        let access = MemoryAccess::load(0x20, 32);
        let banks = access.banks_touched();
        assert_eq!(banks.len(), 2);
        assert_eq!(banks[0], 2);
        assert_eq!(banks[1], 3);
    }

    #[test]
    fn test_alignment_error_display() {
        let err = AlignmentError {
            address: 0x123,
            width: 4,
            required_alignment: 4,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("0x00000123"));
        assert!(msg.contains("width 4"));
        assert!(msg.contains("4-byte alignment"));
    }

    // ========== Cross-Tile Memory Access Tests ==========

    #[test]
    fn test_memory_quadrant_from_address() {
        // Quadrant 0: Local (0x00000-0x0FFFF)
        assert_eq!(MemoryQuadrant::from_address(0x00000), MemoryQuadrant::Local);
        assert_eq!(MemoryQuadrant::from_address(0x08000), MemoryQuadrant::Local);
        assert_eq!(MemoryQuadrant::from_address(0x0FFFF), MemoryQuadrant::Local);

        // Quadrant 1: West (0x10000-0x1FFFF)
        assert_eq!(MemoryQuadrant::from_address(0x10000), MemoryQuadrant::West);
        assert_eq!(MemoryQuadrant::from_address(0x1FFFF), MemoryQuadrant::West);

        // Quadrant 2: North (0x20000-0x2FFFF)
        assert_eq!(MemoryQuadrant::from_address(0x20000), MemoryQuadrant::North);
        assert_eq!(MemoryQuadrant::from_address(0x2FFFF), MemoryQuadrant::North);

        // Quadrant 3: East (0x30000-0x3FFFF)
        assert_eq!(MemoryQuadrant::from_address(0x30000), MemoryQuadrant::East);
        assert_eq!(MemoryQuadrant::from_address(0x3FFFF), MemoryQuadrant::East);
    }

    #[test]
    fn test_quadrant_hop_count() {
        assert_eq!(MemoryQuadrant::Local.hop_count(), 0);
        assert_eq!(MemoryQuadrant::West.hop_count(), 1);
        assert_eq!(MemoryQuadrant::North.hop_count(), 1);
        assert_eq!(MemoryQuadrant::East.hop_count(), 1);
    }

    #[test]
    fn test_quadrant_routing_latency() {
        // Local: 0 cycles
        assert_eq!(MemoryQuadrant::Local.routing_latency(), 0);

        // 1 hop: 4 cycles (LOCAL_TO_EXTERNAL) for all neighbors
        assert_eq!(MemoryQuadrant::West.routing_latency(), CROSS_TILE_LATENCY);
        assert_eq!(MemoryQuadrant::North.routing_latency(), CROSS_TILE_LATENCY);
        assert_eq!(MemoryQuadrant::East.routing_latency(), CROSS_TILE_LATENCY);
    }

    #[test]
    fn test_memory_access_quadrant() {
        // Local access
        let access = MemoryAccess::load(0x1000, 4);
        assert_eq!(access.quadrant(), MemoryQuadrant::Local);
        assert!(!access.is_cross_tile());
        assert_eq!(access.cross_tile_latency(), 0);

        // West neighbor access
        let access = MemoryAccess::load(0x11000, 4);
        assert_eq!(access.quadrant(), MemoryQuadrant::West);
        assert!(access.is_cross_tile());
        assert_eq!(access.cross_tile_latency(), CROSS_TILE_LATENCY);

        // North neighbor access
        let access = MemoryAccess::load(0x21000, 4);
        assert_eq!(access.quadrant(), MemoryQuadrant::North);
        assert!(access.is_cross_tile());
        assert_eq!(access.cross_tile_latency(), CROSS_TILE_LATENCY);

        // East neighbor access
        let access = MemoryAccess::load(0x31000, 4);
        assert_eq!(access.quadrant(), MemoryQuadrant::East);
        assert!(access.is_cross_tile());
        assert_eq!(access.cross_tile_latency(), CROSS_TILE_LATENCY);
    }

    #[test]
    fn test_access_latency_local() {
        let model = MemoryModel::new();

        // Local access: base latency only (5 cycles)
        let access = MemoryAccess::load(0x1000, 4);
        let latency = model.access_latency(&access);
        assert_eq!(latency, BASE_LATENCY);
    }

    #[test]
    fn test_access_latency_cross_tile_west() {
        let model = MemoryModel::new();

        // West neighbor: base + 4 cycles routing (5 + 4 = 9)
        let access = MemoryAccess::load(0x11000, 4);
        let latency = model.access_latency(&access);
        assert_eq!(latency, BASE_LATENCY + CROSS_TILE_LATENCY);
    }

    #[test]
    fn test_access_latency_cross_tile_east() {
        let model = MemoryModel::new();

        // East neighbor: base + 4 cycles routing (5 + 4 = 9)
        let access = MemoryAccess::load(0x31000, 4);
        let latency = model.access_latency(&access);
        assert_eq!(latency, BASE_LATENCY + CROSS_TILE_LATENCY);
    }

    #[test]
    fn test_cross_tile_stats() {
        let mut model = MemoryModel::new();
        model.advance_to(1);

        // Local access (no cross-tile)
        model.record_access(&MemoryAccess::load(0x1000, 4));

        // West neighbor access (1 hop)
        model.record_access(&MemoryAccess::load(0x11000, 4));

        // North neighbor access (1 hop)
        model.record_access(&MemoryAccess::load(0x21000, 4));

        // East neighbor access (1 hop)
        model.record_access(&MemoryAccess::load(0x31000, 4));

        let stats = model.stats();
        assert_eq!(stats.total_accesses, 4);
        assert_eq!(stats.total_cross_tile, 3);
        // 1 hop + 1 hop + 1 hop = 3 hops total = 12 cycles
        assert_eq!(stats.total_cross_tile_cycles, 3 * CROSS_TILE_LATENCY as u64);
        assert!((stats.cross_tile_rate - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_access_latency_combined() {
        let mut model = MemoryModel::new();
        model.advance_to(1);

        // First access to bank 0 in local memory
        let access1 = MemoryAccess::load(0x00, 4);
        model.record_access(&access1);

        // Second access to bank 0 (conflict) + cross-tile (west)
        // Should have: base (5) + conflict (1) + cross-tile (4) = 10
        let access2 = MemoryAccess::load(0x10080, 4);
        let latency = model.access_latency(&access2);

        // Cross-tile to west + bank 0 conflict
        assert_eq!(latency, BASE_LATENCY + CONFLICT_PENALTY + CROSS_TILE_LATENCY);
    }
}
