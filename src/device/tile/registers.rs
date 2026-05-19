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

        // Core debug register dispatch (compute tiles only).
        //
        // Mirrors the identical dispatch in read_register_pure (Phase B Unit 1,
        // commit e0ec922). The two paths MUST return the same value for any
        // given core_debug state -- divergence is a Phase B Unit 1b bug.
        //
        // Core_Status (0x32004) and Debug_Status (0x3201C) are live-computed
        // from CoreDebugState fields; a raw HashMap lookup always returns 0
        // even when the core is halted (DEBUG_HALT bit 16 comes from
        // core_debug.read_status(), not from tile.registers). The same applies
        // to Debug_Control0 (halted+single_step_count) and Debug_Control2
        // (stall-to-halt enables stored in core_debug.debug_ctrl2).
        //
        // The injected MASKPOLL polls Core_Status (0x32004) via this mutable
        // path. Without this dispatch the poll always read 0 and never
        // satisfied, so the emulator could not observe the DEBUG_HALT set by
        // the Unit-1 pre-execute seam (Phase B Unit 1b, spec §4.2/§5.3).
        //
        // Offsets from aie-rt xaiemlgbl_params.h, cross-checked against
        // aie_registers_aie2.json (same offsets as read_register_pure):
        //   Core_Status 0x32004, Debug_Control0..2 0x32010..0x32018,
        //   Debug_Status 0x3201C, PC_Event0..3 0x32020..0x3202C.
        if self.is_compute() {
            if let Some(val) = self.core_debug.read_register(offset) {
                return val;
            }
        }

        // Interrupt controller read routing (shim tiles only). Mirrors the
        // write routing in effects.rs::apply_tile_local_effects. L1 and L2
        // occupy disjoint offset ranges (0x35xxx vs 0x15xxx) so order is
        // immaterial; each returns None for offsets it does not own.
        if self.is_shim() {
            if let Some(ref l1) = self.l1_irq {
                if let Some(val) = l1.read_register(offset) {
                    return val;
                }
            }
            if let Some(ref l2) = self.l2_irq {
                if let Some(val) = l2.read_register(offset) {
                    return val;
                }
            }
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

        // Core debug register dispatch (compute tiles only).
        //
        // Core_Status (0x32004) and Debug_Status (0x3201C) are live-computed
        // registers: their values are derived from the CoreDebugState struct
        // fields (halt flag, stall flags, cause latches) and are never written
        // to tile.registers, so a raw HashMap lookup returns 0 even when the
        // core is halted. Delegating to core_debug.read_register() returns the
        // correct live value.
        //
        // The other registers in this range (PC_Event*, Debug_Control*) store
        // raw values in both tile.registers and core_debug fields; either path
        // returns the same value. Routing them through core_debug.read_register()
        // here is consistent and correct.
        //
        // Offsets derived from aie-rt xaiemlgbl_params.h, cross-checked against
        // the AM025 register database (aie_registers_aie2.json):
        //   Core_Control 0x32000  Core_Status 0x32004  (and 0x32008, 0x3200C)
        //   Debug_Control0..2  0x32010, 0x32014, 0x32018
        //   Debug_Status       0x3201C
        //   PC_Event0..3       0x32020, 0x32024, 0x32028, 0x3202C
        //   Core_PC/SP/LR      0x31100, 0x31120, 0x31130
        //
        // Phase B Unit 1 routing-gap fix (spec §5.1/§5.3, G1-derived 2026-05-18).
        if self.is_compute() {
            if let Some(val) = self.core_debug.read_register(offset) {
                return val;
            }
        }

        // Interrupt controller registers are stateful (write-1-to-clear,
        // enable->mask); they are intentionally NOT routed through this
        // side-effect-free path -- guest interrupt reads use read_register.

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

/// Unit 1b read-path reconciliation tests (spec §4.2/§5.3, Phase B Unit 1b).
///
/// Asserts that the mutable `tile.read_register` path agrees with
/// `read_register_pure` for Core_Status (0x32004) and the debug-register
/// offsets 0x32010–0x3202C. Both paths must return the live-computed
/// `core_debug` value, not a raw HashMap lookup (which always returns 0 for
/// these registers since they are never written to the HashMap).
///
/// Offsets from aie-rt xaiemlgbl_params.h, cross-checked against
/// aie_registers_aie2.json:
///   Core_Status    0x32004  (live-computed; includes DEBUG_HALT bit 16)
///   Debug_Control0 0x32010  (stored in core_debug.halted + single_step_count)
///   Debug_Control1 0x32014  (raw event config)
///   Debug_Control2 0x32018  (stall-to-halt enables)
///   Debug_Status   0x3201C  (live-computed halt-cause latches)
///   PC_Event0      0x32020  (armed VALID|PC_ADDRESS)
///   PC_Event1      0x32024
///   PC_Event2      0x32028
///   PC_Event3      0x3202C
#[cfg(test)]
mod unit1b_read_path_tests {
    use super::*;

    // -------------------------------------------------------------------
    // Core_Status (0x32004) — reflects core_debug halt state, both paths
    // -------------------------------------------------------------------

    /// The mutable read path must reflect DEBUG_HALT (bit 16) when the
    /// core is halted. Before Unit 1b this always returned 0 (raw HashMap).
    #[test]
    fn mutable_read_register_reflects_debug_halt_bit() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = true;
        tile.core_debug.enabled = true;
        let status = tile.read_register(0x32004);
        assert_ne!(status & (1 << 16), 0, "read_register(Core_Status) must reflect DEBUG_HALT when halted");
    }

    /// The mutable path must NOT set DEBUG_HALT when the core is not halted.
    #[test]
    fn mutable_read_register_no_debug_halt_when_not_halted() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = false;
        tile.core_debug.enabled = true;
        let status = tile.read_register(0x32004);
        assert_eq!(
            status & (1 << 16),
            0,
            "read_register(Core_Status) must NOT set DEBUG_HALT when not halted"
        );
    }

    /// `read_register` and `read_register_pure` must agree on Core_Status
    /// for the same core_debug state (halted).
    #[test]
    fn mutable_and_pure_agree_on_core_status_halted() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = true;
        tile.core_debug.enabled = true;
        // read_register takes &mut self; read_register_pure takes &self.
        // Save the pure value first (immutable borrow), then mutable.
        let pure_val = tile.read_register_pure(0x32004);
        let mut_val = tile.read_register(0x32004);
        assert_eq!(
            pure_val, mut_val,
            "read_register and read_register_pure must agree on Core_Status (halted)"
        );
    }

    /// Both paths must agree on Core_Status when the core is not halted.
    #[test]
    fn mutable_and_pure_agree_on_core_status_not_halted() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = false;
        tile.core_debug.enabled = true;
        let pure_val = tile.read_register_pure(0x32004);
        let mut_val = tile.read_register(0x32004);
        assert_eq!(
            pure_val, mut_val,
            "read_register and read_register_pure must agree on Core_Status (not halted)"
        );
    }

    // -------------------------------------------------------------------
    // Debug_Control0 (0x32010) — halted + single_step_count packed
    // -------------------------------------------------------------------

    #[test]
    fn mutable_and_pure_agree_on_debug_control0() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = true;
        tile.core_debug.single_step_count = 3;
        let pure_val = tile.read_register_pure(0x32010);
        let mut_val = tile.read_register(0x32010);
        assert_eq!(pure_val, mut_val, "read_register and read_register_pure must agree on Debug_Control0");
        // Smoke: halt bit (bit 0) + count (bits [5:2]=3 → 0x0C)
        assert_ne!(mut_val & 1, 0, "halt bit must be set");
        assert_ne!(mut_val & 0x0C, 0, "single_step_count must be non-zero");
    }

    // -------------------------------------------------------------------
    // Debug_Control2 (0x32018) — raw value stored in core_debug
    // -------------------------------------------------------------------

    #[test]
    fn mutable_and_pure_agree_on_debug_control2() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.debug_ctrl2 = 0x01; // PC_Event_Halt enabled
        let pure_val = tile.read_register_pure(0x32018);
        let mut_val = tile.read_register(0x32018);
        assert_eq!(pure_val, mut_val, "read_register and read_register_pure must agree on Debug_Control2");
        assert_eq!(mut_val, 0x01, "Debug_Control2 value must match what was set");
    }

    // -------------------------------------------------------------------
    // Debug_Status (0x3201C) — live-computed halt-cause latches
    // -------------------------------------------------------------------

    #[test]
    fn mutable_and_pure_agree_on_debug_status() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.halted = true;
        tile.core_debug.halt_cause_pc_event = true;
        let pure_val = tile.read_register_pure(0x3201C);
        let mut_val = tile.read_register(0x3201C);
        assert_eq!(pure_val, mut_val, "read_register and read_register_pure must agree on Debug_Status");
        assert_ne!(mut_val & 1, 0, "Debug_Status halted bit must be set");
        assert_ne!(mut_val & 2, 0, "Debug_Status PC_Event_halted bit must be set");
    }

    // -------------------------------------------------------------------
    // PC_Event0 (0x32020) — raw VALID|PC_ADDRESS stored in core_debug
    // -------------------------------------------------------------------

    #[test]
    fn mutable_and_pure_agree_on_pc_event0() {
        let mut tile = Tile::compute(0, 2);
        tile.core_debug.pc_event0 = 0x8000_0184; // VALID | TRAP_PC14=0x184
        let pure_val = tile.read_register_pure(0x32020);
        let mut_val = tile.read_register(0x32020);
        assert_eq!(pure_val, mut_val, "read_register and read_register_pure must agree on PC_Event0");
        assert_eq!(mut_val, 0x8000_0184, "PC_Event0 value must match what was set");
    }

    // -------------------------------------------------------------------
    // Non-debug registers must be unaffected (still HashMap)
    // -------------------------------------------------------------------

    /// A non-debug register written to the HashMap must still be read
    /// through the HashMap by the mutable path.
    #[test]
    fn non_debug_register_unaffected_reads_from_hashmap() {
        let mut tile = Tile::compute(0, 2);
        let arbitrary_offset = 0x1234_5678u32;
        tile.registers.insert(arbitrary_offset, 0xDEAD_BEEF);
        assert_eq!(
            tile.read_register(arbitrary_offset),
            0xDEAD_BEEF,
            "non-debug registers must still be read from the HashMap"
        );
        // Ensure the debug-register dispatch does not intercept it.
        assert_ne!(arbitrary_offset, 0x32004, "test setup: offset must not be a debug register");
    }

    /// Mem-tile read_register must not dispatch to core_debug (compute-only).
    #[test]
    fn memtile_core_status_offset_not_dispatched_to_core_debug() {
        let mut tile = Tile::mem_tile(0, 1);
        // Mem tiles have no core_debug; write the offset to the HashMap to
        // give it a non-zero value and verify it round-trips without dispatch.
        tile.registers.insert(0x32004, 0xABCD_EF01);
        assert_eq!(
            tile.read_register(0x32004),
            0xABCD_EF01,
            "mem-tile must not dispatch Core_Status offset to core_debug"
        );
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
    fn read_first_header_parity_bit0_compute() {
        let mut tile = Tile::compute(0, 2);
        tile.pkt_handler_status = 0b0001; // First_Header_Parity_Error
        assert_eq!(tile.read_register(0x3FF30), 0b0001);
        assert_eq!(tile.read_register_pure(0x3FF30), 0b0001);
    }
}

/// Task 1: L1/L2 read-path routing (interrupt close-out, 2026-05-19).
///
/// Verifies that `tile.read_register` routes L1 and L2 register reads through
/// the dedicated interrupt controller objects on shim tiles, so that
/// write-1-to-clear status and enable→mask reflection are visible to a guest
/// read.  Before Task 1 both reads fell through to the generic HashMap
/// (returning 0).
#[cfg(test)]
mod interrupt_read_routing_tests {
    use super::*;

    /// Writing to L2 Enable must make the mask reflect via `read_register`.
    #[test]
    fn shim_l2_enable_then_read_mask_reflects() {
        let mut tile = Tile::shim(0, 0);
        use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_MASK};
        assert!(tile.l2_irq.as_mut().unwrap().write_register(L2_REG_ENABLE, 0b1001));
        assert_eq!(tile.read_register(L2_REG_MASK), 0b1001);
        // A non-interrupt offset must still fall through (the routing block
        // must not intercept offsets the controllers don't own).
        assert_eq!(tile.read_register(0x0001_FFFF), 0, "non-L2 offset must fall through");
    }

    /// Writing to L1 Enable (Switch A) must make the mask reflect via
    /// `read_register`.  Mirrors the L2 enable→mask test for L1.
    #[test]
    fn shim_l1_enable_then_read_mask_reflects() {
        let mut tile = Tile::shim(0, 0);
        use crate::device::interrupts::{L1_REG_ENABLE_A, L1_REG_MASK_A};
        assert!(tile.l1_irq.as_mut().unwrap().write_register(L1_REG_ENABLE_A, 0b1010));
        assert_eq!(tile.read_register(L1_REG_MASK_A), 0b1010);
    }

    /// L2 status write-1-to-clear must be visible through `read_register`.
    #[test]
    fn shim_l2_status_write_one_to_clear_visible_on_read() {
        let mut tile = Tile::shim(0, 0);
        use crate::device::interrupts::{L2_REG_ENABLE, L2_REG_STATUS};
        tile.l2_irq.as_mut().unwrap().write_register(L2_REG_ENABLE, 0b1);
        tile.l2_irq.as_mut().unwrap().signal_interrupt(0);
        assert_eq!(tile.read_register(L2_REG_STATUS), 0b1);
        tile.l2_irq.as_mut().unwrap().write_register(L2_REG_STATUS, 0b1);
        assert_eq!(tile.read_register(L2_REG_STATUS), 0);
    }
}
