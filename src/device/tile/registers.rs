//! Register read/write methods on Tile.

use super::*;

impl Tile {
    /// Get an immutable reference to the register map.
    ///
    /// Used by mask_write_register in state.rs to read current values without
    /// triggering side effects (unlike read_register which executes lock operations).
    pub fn registers_ref(&self) -> &std::collections::HashMap<u32, u32> {
        &self.registers
    }

    /// Read a 32-bit value from a register offset.
    ///
    /// Returns 0 for unwritten registers (default state).
    pub fn read_register(&mut self, offset: u32) -> u32 {
        use xdna_archspec::aie2::registers::mem_tile as mt;
        use xdna_archspec::aie2::registers::memory as mm;
        let reg_layout = super::super::regdb::device_reg_layout();

        // Lock_Request register - address encodes operation parameters
        // Reading performs the lock operation and returns result
        if self.is_mem() {
            if (mt::LOCK_REQUEST_BASE..mt::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, true);
            }
            // Lock status registers
            if offset == reg_layout.memtile_locks_overflow_0 {
                return self.get_lock_overflow_bits(0, 32);
            }
            if offset == reg_layout.memtile_locks_overflow_1 {
                return self.get_lock_overflow_bits(32, 64);
            }
            if offset == reg_layout.memtile_locks_underflow_0 {
                return self.get_lock_underflow_bits(0, 32);
            }
            if offset == reg_layout.memtile_locks_underflow_1 {
                return self.get_lock_underflow_bits(32, 64);
            }
        } else if self.is_compute() {
            if (mm::LOCK_REQUEST_BASE..mm::LOCK_REQUEST_END).contains(&offset) {
                return self.handle_lock_request(offset, false);
            }
            // Lock status registers
            if offset == reg_layout.memory_locks_overflow {
                return self.get_lock_overflow_bits(0, 16);
            }
            if offset == reg_layout.memory_locks_underflow {
                return self.get_lock_underflow_bits(0, 16);
            }
        }

        // Check specific subsystem state first: DMA BD range.
        // Base and stride are per-tile-type from the register database.
        let (bd_base, bd_stride) = self.bd_layout(reg_layout);
        let bd_end = bd_base + (self.dma_bds.len() as u32) * bd_stride;
        if offset >= bd_base && offset < bd_end {
            let bd_offset = offset - bd_base;
            let bd_index = (bd_offset / bd_stride) as usize;
            let reg_in_bd = (bd_offset % bd_stride) as usize / 4;

            if bd_index < self.dma_bds.len() {
                let bd = &self.dma_bds[bd_index];
                // Legacy struct has 6 fields; words 6-7 (shim/memtile iteration
                // and lock/valid) fall through to the register HashMap below.
                match reg_in_bd {
                    0 => return bd.addr_low,
                    1 => return bd.addr_high,
                    2 => return bd.length,
                    3 => return bd.control,
                    4 => return bd.d0,
                    5 => return bd.d1,
                    _ => {} // Fall through to register map for words 6-7
                }
            }
        }

        // Control_Packet_Handler_Status sticky bits (compute 0x3FF30,
        // memtile 0xB0F30). Source-of-truth lives on the dedicated field;
        // tile.registers is not used for this offset.
        if (self.is_compute() && offset == 0x3FF30) || (self.is_mem() && offset == 0xB0F30) {
            return self.pkt_handler_status & 0xF;
        }

        // Fall back to register map
        self.registers.get(&offset).copied().unwrap_or(0)
    }

    /// Read a register value without side effects.
    ///
    /// Unlike `read_register()`, this does NOT execute lock operations.
    /// Used for MMIO loads from the memory unit where mutable tile access
    /// is not available during instruction execution.
    pub fn read_register_pure(&self, offset: u32) -> u32 {
        let reg_layout = super::super::regdb::device_reg_layout();

        // Data memory: read from the flat byte array where cores and DMA
        // actually store data. Control packet OP_READ targets this space
        // for addresses below the data memory size (0x10000 for compute,
        // 0x80000 for memtile).
        let mem_size = self.data_memory.len() as u32;
        if mem_size > 0 && offset + 4 <= mem_size {
            return u32::from_le_bytes([
                self.data_memory[offset as usize],
                self.data_memory[offset as usize + 1],
                self.data_memory[offset as usize + 2],
                self.data_memory[offset as usize + 3],
            ]);
        }

        // DMA BD range (per-tile-type from register database)
        let (bd_base, bd_stride) = self.bd_layout(reg_layout);
        let bd_end = bd_base + (self.dma_bds.len() as u32) * bd_stride;
        if offset >= bd_base && offset < bd_end {
            let bd_offset = offset - bd_base;
            let bd_index = (bd_offset / bd_stride) as usize;
            let reg_in_bd = (bd_offset % bd_stride) as usize / 4;
            if bd_index < self.dma_bds.len() {
                let bd = &self.dma_bds[bd_index];
                return match reg_in_bd {
                    0 => bd.addr_low,
                    1 => bd.addr_high,
                    2 => bd.length,
                    3 => bd.control,
                    4 => bd.d0,
                    5 => bd.d1,
                    _ => self.registers.get(&offset).copied().unwrap_or(0),
                };
            }
        }

        // DMA channel control (per-tile-type from register database).
        // Compute tiles have a single channel base (S2MM and MM2S interleaved).
        // MemTiles have separate S2MM/MM2S bases. The stride is the same.
        let (ch_base, ch_stride) = self.channel_layout(reg_layout);
        let ch_end = ch_base + (self.dma_channels.len() as u32) * ch_stride;
        if offset >= ch_base && offset < ch_end {
            let ch_offset = offset - ch_base;
            let ch_index = (ch_offset / ch_stride) as usize;
            if ch_index < self.dma_channels.len() {
                return if ch_offset % ch_stride == 0 {
                    self.dma_channels[ch_index].control
                } else {
                    self.dma_channels[ch_index].start_queue
                };
            }
        }

        // Lock value registers (read-only, no acquire side effect)
        let lock_base = if self.is_mem() {
            reg_layout.memtile_lock_base
        } else {
            reg_layout.memory_lock_base
        };
        let lock_stride = if self.is_mem() {
            reg_layout.memtile_lock_stride
        } else {
            reg_layout.memory_lock_stride
        };
        let lock_end = lock_base + (self.locks.len() as u32) * lock_stride;
        if (lock_base..lock_end).contains(&offset) {
            let lock_id = ((offset - lock_base) / lock_stride) as usize;
            if lock_id < self.locks.len() {
                return self.locks[lock_id].value as u32
                    & crate::device::arch_handle::lock_value_layout().mask;
            }
        }

        // Control_Packet_Handler_Status sticky bits (compute 0x3FF30,
        // memtile 0xB0F30). See read_register.
        if (self.is_compute() && offset == 0x3FF30) || (self.is_mem() && offset == 0xB0F30) {
            return self.pkt_handler_status & 0xF;
        }

        // Fall back to register map
        self.registers.get(&offset).copied().unwrap_or(0)
    }

    /// Get a reference to the raw register map.
    ///
    /// Useful for debugging and inspection.
    pub fn registers(&self) -> &std::collections::HashMap<u32, u32> {
        &self.registers
    }

    // === Performance Counter Register Handling ===

    /// Handle a write to a core module performance counter register.
    ///
    /// Dispatches based on the register offset within the core performance
    /// counter range. Compute tiles use offsets 0x31500-0x3158C (4 counters);
    /// shim tiles use PL module offsets 0x31000-0x31084 (2 counters).
    ///
    /// Register layout per aie-rt (xaiemlgbl_params.h):
    ///   Control0 (+0x00): cnt0/cnt1 start/stop events
    ///   Control1 (+0x04): cnt2/cnt3 start/stop events (4-counter only)
    ///   Control2 (+0x08): cnt0-3 reset events (4-counter only)
    ///   Counter0-3 (+0x20..+0x2C): counter values
    ///   EventValue0-3 (+0x80..+0x8C): event value thresholds
    /// Handle a write to a core module performance counter register.
    /// Delegates to the PerfCounterBank's register interface.
    pub fn write_core_perf_register(&mut self, offset_in_block: u32, value: u32) {
        // Core/PL modules use 7-bit event fields
        self.core_perf_counters.write_register(offset_in_block, value, 7);
    }

    /// Handle a write to a memory module performance counter register.
    /// Delegates to the PerfCounterBank's register interface.
    pub fn write_mem_perf_register(&mut self, offset_in_block: u32, value: u32) {
        // MemTile uses 8-bit event fields; compute memory module uses 7-bit
        let event_width = if self.is_mem() { 8 } else { 7 };
        self.mem_perf_counters.write_register(offset_in_block, value, event_width);
    }

    // === Lock_Request Register Handling ===

    /// Handle a Lock_Request register read.
    ///
    /// The address encodes the lock operation:
    /// - Lock_Id: bits [13:10] (compute) or [15:10] (memtile)
    /// - Acq_Rel: bit [9] (1=acquire, 0=release)
    /// - Change_Value: bits [8:2] (7-bit signed)
    ///
    /// Reading from this address performs the operation and returns:
    /// - Bit 0: 1 if operation succeeded, 0 if it would stall/fail
    fn handle_lock_request(&mut self, offset: u32, is_memtile: bool) -> u32 {
        use xdna_archspec::aie2::registers::mem_tile as mt;
        use xdna_archspec::aie2::registers::memory as mm;

        let base = if is_memtile {
            mt::LOCK_REQUEST_BASE
        } else {
            mm::LOCK_REQUEST_BASE
        };
        let addr = offset - base;

        // Extract fields from address
        let id_shift = if is_memtile {
            mt::LOCK_REQUEST_ID_SHIFT
        } else {
            mm::LOCK_REQUEST_ID_SHIFT
        };
        let id_mask = if is_memtile {
            mt::LOCK_REQUEST_ID_MASK
        } else {
            mm::LOCK_REQUEST_ID_MASK
        };

        let lock_id = ((addr >> id_shift) & id_mask) as usize;
        let is_acquire = (addr >> mm::LOCK_REQUEST_ACQ_REL_BIT) & 1 != 0;
        let change_raw = ((addr >> mm::LOCK_REQUEST_VALUE_SHIFT) & mm::LOCK_REQUEST_VALUE_MASK) as i8;

        // Sign-extend 7-bit value
        let change_value = if change_raw & 0x40 != 0 {
            change_raw | !0x7F_i8 // Sign extend
        } else {
            change_raw
        };

        // Bounds check against actual lock count for this tile
        if lock_id >= self.locks.len() {
            return 0; // Invalid lock ID
        }

        // Perform the operation.
        //
        // AIE-ML lock semantics (matching DMA engine in dma/engine.rs):
        // - change_value < 0: acq_ge -- wait until lock >= |value|, then decrement
        // - change_value > 0: acq_eq -- wait until lock == value, then set to 0
        // - change_value == 0: simple acquire (decrement if > 0)
        let result = if is_acquire {
            if change_value < 0 {
                // acq_ge: wait until lock >= |value|, then decrement by |value|
                let expected = (-change_value) as i8;
                self.locks[lock_id].acquire_with_value(expected, change_value)
            } else if change_value > 0 {
                // acq_eq: wait until lock == value, then decrement to 0
                let expected = change_value as i8;
                let delta = -expected;
                self.locks[lock_id].acquire_equal(expected, delta)
            } else {
                // Simple acquire: decrement by 1 if > 0
                self.locks[lock_id].acquire_with_value(1, -1)
            }
        } else {
            // Release: apply delta (typically positive)
            self.locks[lock_id].release_with_value(change_value)
        };

        // Return success bit
        if matches!(result, LockResult::Success) {
            1
        } else {
            0
        }
    }

    /// Get lock overflow bits for a range of locks.
    ///
    /// Returns a bitmask where bit N is set if lock (start + N) has overflowed.
    fn get_lock_overflow_bits(&self, start: usize, end: usize) -> u32 {
        let mut bits = 0u32;
        for i in start..end.min(self.locks.len()) {
            if self.locks[i].overflow {
                bits |= 1 << (i - start);
            }
        }
        bits
    }

    /// Get lock underflow bits for a range of locks.
    ///
    /// Returns a bitmask where bit N is set if lock (start + N) has underflowed.
    fn get_lock_underflow_bits(&self, start: usize, end: usize) -> u32 {
        let mut bits = 0u32;
        for i in start..end.min(self.locks.len()) {
            if self.locks[i].underflow {
                bits |= 1 << (i - start);
            }
        }
        bits
    }

    /// Clear lock overflow bits for a range (write-to-clear behavior).
    pub(crate) fn clear_lock_overflow_bits(&mut self, start: usize, end: usize, bits: u32) {
        for i in start..end.min(self.locks.len()) {
            if bits & (1 << (i - start)) != 0 {
                self.locks[i].overflow = false;
            }
        }
    }

    /// Clear lock underflow bits for a range (write-to-clear behavior).
    pub(crate) fn clear_lock_underflow_bits(&mut self, start: usize, end: usize, bits: u32) {
        for i in start..end.min(self.locks.len()) {
            if bits & (1 << (i - start)) != 0 {
                self.locks[i].underflow = false;
            }
        }
    }
}

#[cfg(test)]
mod pkt_handler_status_tests {
    use super::*;

    #[test]
    fn read_returns_pkt_handler_status_compute() {
        let mut tile = Tile::compute(0, 2);
        tile.pkt_handler_status = 0b0010; // Second_Header_Parity
        assert_eq!(tile.read_register(0x3FF30), 0b0010);
        assert_eq!(tile.read_register_pure(0x3FF30), 0b0010);
    }

    #[test]
    fn read_returns_pkt_handler_status_memtile() {
        let mut tile = Tile::mem_tile(0, 1);
        tile.pkt_handler_status = 0b1000; // Tlast_Error
        assert_eq!(tile.read_register(0xB0F30), 0b1000);
        assert_eq!(tile.read_register_pure(0xB0F30), 0b1000);
    }

    #[test]
    fn read_masks_to_low_4_bits() {
        let mut tile = Tile::compute(0, 2);
        tile.pkt_handler_status = 0xFFFF_FFFF;
        assert_eq!(tile.read_register(0x3FF30), 0xF);
    }

    #[test]
    fn read_and_clear_first_header_parity_bit0_compute() {
        let mut tile = Tile::compute(0, 2);
        tile.pkt_handler_status = 0b0001; // First_Header_Parity_Error
        assert_eq!(tile.read_register(0x3FF30), 0b0001);
    }

    #[test]
    fn read_tlast_error_bit3_memtile() {
        let mut tile = Tile::mem_tile(0, 1);
        tile.pkt_handler_status = 0b1000; // Tlast_Error
        assert_eq!(tile.read_register(0xB0F30), 0b1000);
        assert_eq!(tile.read_register_pure(0xB0F30), 0b1000);
    }
}
